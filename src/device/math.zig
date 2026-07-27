//! The libm subset that survives GPU codegen.
//!
//! NVPTX and AMDGCN emit no libcalls, so any `llvm.<libm>` they cannot map to a
//! hardware instruction is a hard error at the IR->ISA stage — `no libcall
//! available for fexp`, `Cannot select: f32 = fsin` — which lands late and names
//! clang internals rather than your kernel. `@exp @log @log2 @log10 @sin @cos`
//! and `std.math.sinh/cosh/pow/tanh/atan/expm1` all trip it. This module is the
//! replacement; it also compiles on the host, so one kernel source builds both
//! ways.
//!
//! What each target can actually do (measured, zig 0.16 / LLVM 21):
//!
//!   f32 amdgcn   LLVM already expands exp/exp2/log/log2/log10/sin/cos to the
//!                v_*_f32 hardware approximations, so the builtins are used.
//!   f32 nvptx64  only `@exp2` lowers (to `ex2.approx.f32`); log2/sin/cos come
//!                from `llvm.nvvm.*.approx.f` and the rest is derived.
//!   f64 both     nothing but `@sqrt`. Software (musl ports) all the way.
//!
//! So accuracy is not uniform, and device f32 is NOT IEEE. Numbers below are
//! measured on an RTX 4060 (sm_89) over x in (0, 8], 1024 samples, against
//! glibc; the per-function doc comments carry the individual bounds.
//!
//!   device f32  <=4 ulp relative, EXCEPT log/log2/log10/sin/cos, which the
//!               hardware bounds absolutely (~2^-21 for lg2, ~2^-20 for sin/cos)
//!               rather than relatively — so relative accuracy collapses at
//!               their zeros: log near x=1, sin near k*pi, cos near pi/2+k*pi.
//!   device f64  <=1.4 ulp (pow <=4), musl ports, checked against the builtins
//!               in the test below.
//!   host        libm, forwarded to the builtins / `std.math`.
//!
//! Want IEEE-grade f32 on device? Compute in f64 and `@floatCast` — that path is
//! software and correct, just slow (1/64 rate on consumer NVIDIA).
//!
//! Already safe everywhere and deliberately absent here: `@sqrt @abs @trunc
//! @round @floor @ceil @copysign @min @max @mulAdd`, `std.math.scalbn/frexp/modf`.

const std = @import("std");
const builtin = @import("builtin");

const arch = builtin.cpu.arch;
const dev = arch == .nvptx64 or arch == .amdgcn;

fn Check(comptime T: type) type {
    return switch (T) {
        f32, f64 => T,
        else => @compileError("gompute.math supports f32 and f64, not " ++
            @typeName(T) ++ " — cast first"),
    };
}

// ---------------------------------------------------------------------------
// Public API. Host forwards to libm; device dispatches on the element type.
// ---------------------------------------------------------------------------

/// e^x. Measured sm_89: f32 <=3.7 ulp, f64 <=1 ulp.
pub inline fn exp(x: anytype) @TypeOf(x) {
    const T = Check(@TypeOf(x));
    if (!dev) return @exp(x);
    return if (T == f32) @exp2(x * @as(f32, log2e)) else softExp(x);
}

/// 2^x, exact for integer x. Measured sm_89: f32 <=1 ulp, f64 <=1 ulp.
pub inline fn exp2(x: anytype) @TypeOf(x) {
    const T = Check(@TypeOf(x));
    if (!dev) return @exp2(x);
    return if (T == f32) @exp2(x) else softExp2(x);
}

/// Natural log. Measured sm_89: f32 <=3 ulp relative, f64 <=1 ulp.
///
/// ponytail: the f32 back end bounds `lg2` ABSOLUTELY (~2^-21), not relatively,
/// so log(x) for x near 1 is 2^-21 of noise on a near-zero answer. Anything
/// doing log1p-style work in that neighbourhood wants f64 (or a real logf port).
pub inline fn log(x: anytype) @TypeOf(x) {
    const T = Check(@TypeOf(x));
    if (!dev) return @log(x);
    return if (T == f32) hwLog2(x) * @as(f32, ln2) else softLog(x);
}

/// Base-2 log. Measured sm_89: f32 <=4 ulp relative (same near-1 caveat as
/// `log`), f64 <=1 ulp.
pub inline fn log2(x: anytype) @TypeOf(x) {
    const T = Check(@TypeOf(x));
    if (!dev) return @log2(x);
    // ponytail: f64 is softLog * 1/ln2, so a power of two can come back 1 ulp
    // off and a subnormal argument ~22 ulp (log's own half-ulp, rebased onto a
    // 20x larger result). Port musl's log2.c if a caller needs it exact.
    return if (T == f32) hwLog2(x) else softLog(x) * log2e;
}

/// Base-10 log. Measured sm_89: f32 <=2 ulp relative (same near-1 caveat as
/// `log`), f64 <=1.1 ulp.
pub inline fn log10(x: anytype) @TypeOf(x) {
    const T = Check(@TypeOf(x));
    if (!dev) return @log10(x);
    return if (T == f32) hwLog2(x) * @as(f32, log10_2) else softLog(x) * log10e;
}

/// Measured sm_89 f32: ~1e-6 ABSOLUTE, which is <=29 ulp relative away from the
/// zeros and unbounded at them — `sin.approx.f32`/`v_sin_f32` are absolute-error
/// devices, and their own range reduction gives out for large |x|. f64 matched
/// glibc bit-for-bit over (0,8]; see `remPio2` for its ceiling.
pub inline fn sin(x: anytype) @TypeOf(x) {
    const T = Check(@TypeOf(x));
    if (!dev) return @sin(x);
    return if (T == f32) hwSin(x) else softSin(x);
}

/// Accuracy as `sin`.
pub inline fn cos(x: anytype) @TypeOf(x) {
    const T = Check(@TypeOf(x));
    if (!dev) return @cos(x);
    return if (T == f32) hwCos(x) else softCos(x);
}

/// ponytail: sin/cos, so error blows up near the poles where cos goes to zero
/// (measured 0.16 absolute at x = 3pi/2 in f32). A dedicated tan with its own
/// argument reduction is worth writing only if someone is actually near pi/2.
pub inline fn tan(x: anytype) @TypeOf(x) {
    _ = Check(@TypeOf(x));
    return sin(x) / cos(x);
}

/// Cephes rational below 0.625, else 1 - 2/(e^2|x| + 1).
/// Measured sm_89: f32 <=1.8 ulp, f64 <=1 ulp.
pub inline fn tanh(x: anytype) @TypeOf(x) {
    const T = Check(@TypeOf(x));
    if (!dev) return std.math.tanh(x);
    const ax = @abs(x);
    if (ax < 0.625) {
        const z = x * x;
        const p = ((@as(T, -9.64399179425052238628e-1) * z +
            @as(T, -9.92877231001918586564e1)) * z + @as(T, -1.61468768441708447952e3)) * z;
        const q = ((z + @as(T, 1.12811678491632931402e2)) * z +
            @as(T, 2.23548839060100448583e3)) * z + @as(T, 4.84406305325125486048e3);
        return x + x * (p / q);
    }
    // exp overflows to inf for large ax, which lands on exactly +-1. No branch.
    const r = 1 - @as(T, 2) / (exp(2 * ax) + 1);
    return if (x < 0) -r else r;
}

/// Taylor in x^2 below 0.5 (the (e^x - e^-x)/2 cancellation region), else the
/// exponentials. Measured sm_89: f32 <=3.7 ulp, f64 <=1.4 ulp.
pub inline fn sinh(x: anytype) @TypeOf(x) {
    const T = Check(@TypeOf(x));
    if (!dev) return std.math.sinh(x);
    const ax = @abs(x);
    if (ax < 0.5) {
        const z = x * x;
        return x * (1 + z * (@as(T, 1.0 / 6.0) + z * (@as(T, 1.0 / 120.0) +
            z * (@as(T, 1.0 / 5040.0) + z * (@as(T, 1.0 / 362880.0) +
                z * (@as(T, 1.0 / 39916800.0) + z * @as(T, 1.0 / 6227020800.0)))))));
    }
    const e = exp(ax);
    const r = @as(T, 0.5) * e - @as(T, 0.5) / e;
    return if (x < 0) -r else r;
}

/// ponytail: overflows to inf just under |x| = 710 (f64) instead of 710.48 —
/// musl splits the exponential to buy those last ulps of range. Split it too if
/// anything ever survives a cosh that large.
pub inline fn cosh(x: anytype) @TypeOf(x) {
    const T = Check(@TypeOf(x));
    if (!dev) return std.math.cosh(x);
    const e = exp(@abs(x));
    return @as(T, 0.5) * e + @as(T, 0.5) / e;
}

/// Measured sm_89 (y = 1.7): f32 <=3 ulp, f64 <=4 ulp.
///
/// ponytail: 2^(y*log2 x) rather than a double-double split, so relative error
/// grows as |y*log2 x| ulps (~1e-14 at y*ln x = -80). Integer |y| <= 64 is
/// square-and-multiply instead, which is exact — `**2`/`**3` are everywhere and
/// the exp/log route is 2 ulp off on them.
pub inline fn pow(x: anytype, y: @TypeOf(x)) @TypeOf(x) {
    const T = Check(@TypeOf(x));
    if (!dev) return std.math.pow(T, x, y);
    if (y == 0 or x == 1) return 1;
    if (x == 0) return if (y > 0) 0 else std.math.inf(T);
    if (y == @trunc(y) and @abs(y) <= 64) {
        var n: u32 = @intFromFloat(@abs(y));
        var base = x;
        var acc: T = 1;
        while (n != 0) : (n >>= 1) {
            if (n & 1 != 0) acc *= base;
            base *= base;
        }
        return if (y < 0) 1 / acc else acc;
    }
    if (x > 0) return exp2(y * log2(x));
    // Negative base is defined only for an integer exponent.
    if (y != @trunc(y) or @abs(y) >= std.math.maxInt(std.meta.Int(.unsigned, std.math.floatMantissaBits(T) + 1)))
        return std.math.nan(T);
    const m = exp2(y * log2(-x));
    return if (@rem(@abs(y), @as(T, 2)) == 1) -m else m;
}

/// Native instruction on both back ends; here so callers need not remember
/// which builtins are device-safe.
pub inline fn sqrt(x: anytype) @TypeOf(x) {
    _ = Check(@TypeOf(x));
    return @sqrt(x);
}

/// ponytail: 1/sqrt, i.e. two IEEE ops. Swap in `rsqrt.approx.f32`/`v_rsq_f32`
/// if a normalization loop ever shows up hot in a profile.
pub inline fn rsqrt(x: anytype) @TypeOf(x) {
    const T = Check(@TypeOf(x));
    return @as(T, 1) / @sqrt(x);
}

// ---------------------------------------------------------------------------
// f32 device primitives — hardware approximations.
// ---------------------------------------------------------------------------

extern fn @"llvm.nvvm.lg2.approx.f"(f32) callconv(.c) f32;
extern fn @"llvm.nvvm.sin.approx.f"(f32) callconv(.c) f32;
extern fn @"llvm.nvvm.cos.approx.f"(f32) callconv(.c) f32;

inline fn hwLog2(x: f32) f32 {
    return if (arch == .nvptx64) @"llvm.nvvm.lg2.approx.f"(x) else @log2(x);
}
inline fn hwSin(x: f32) f32 {
    return if (arch == .nvptx64) @"llvm.nvvm.sin.approx.f"(x) else @sin(x);
}
inline fn hwCos(x: f32) f32 {
    return if (arch == .nvptx64) @"llvm.nvvm.cos.approx.f"(x) else @cos(x);
}

const ln2 = 0.69314718055994530942;
const log2e = 1.44269504088896338700;
const log10e = 0.43429448190325182765;
const log10_2 = 0.30102999566398119521;

// ---------------------------------------------------------------------------
// exp — musl exp.c
// ---------------------------------------------------------------------------

const ln2hi = 6.93147180369123816490e-01;
const ln2lo = 1.90821492927058770002e-10;
const P1 = 1.66666666666666019037e-01;
const P2 = -2.77777777770155933842e-03;
const P3 = 6.61375632143793436117e-05;
const P4 = -1.65339022054652515390e-06;
const P5 = 4.13813679705723846039e-08;

fn softExp(x: f64) f64 {
    const bits: u64 = @bitCast(x);
    const neg = bits >> 63 != 0;
    const ax: u32 = @truncate((bits >> 32) & 0x7fffffff);

    if (ax >= 0x4086232b) { // |x| >~ 708.39
        if (std.math.isNan(x)) return x;
        if (x > 709.782712893383973096) return std.math.inf(f64);
        if (x < -745.13321910194110842) return 0;
    }

    var k: i32 = 0;
    var hi: f64 = x;
    var lo: f64 = 0;
    var r = x;
    if (ax > 0x3fd62e42) { // |x| > 0.5 ln2
        k = if (ax >= 0x3ff0a2b2) // |x| >= 1.5 ln2
            @intFromFloat(log2e * x + if (neg) @as(f64, -0.5) else 0.5)
        else if (neg) -1 else 1;
        const kf: f64 = @floatFromInt(k);
        hi = x - kf * ln2hi;
        lo = kf * ln2lo;
        r = hi - lo;
    } else if (ax <= 0x3e300000) { // |x| <= 2^-28: 1+x is already correct
        return 1 + x;
    }

    const rr = r * r;
    const c = r - rr * (P1 + rr * (P2 + rr * (P3 + rr * (P4 + rr * P5))));
    const y = 1 + (r * c / (2 - c) - lo + hi);
    return if (k == 0) y else std.math.scalbn(y, k);
}

/// 2^x. Splitting off the integer part keeps integer x exact and leaves the
/// (inexact) ln2 multiply acting on |f| <= 0.5, where its error is negligible.
fn softExp2(x: f64) f64 {
    const k = @round(x);
    // Not-a-number and the overflow ends go straight to softExp, which
    // saturates correctly; the `!(<=)` spelling routes NaN here too.
    if (!(@abs(k) <= 1100)) return softExp(x * ln2);
    return std.math.scalbn(softExp((x - k) * ln2), @intFromFloat(k));
}

// ---------------------------------------------------------------------------
// log — musl log.c
// ---------------------------------------------------------------------------

const Lg1 = 6.666666666666735130e-01;
const Lg2 = 3.999999999940941908e-01;
const Lg3 = 2.857142874366239149e-01;
const Lg4 = 2.222219843214978396e-01;
const Lg5 = 1.818357216161805012e-01;
const Lg6 = 1.531383769920937332e-01;
const Lg7 = 1.479819860511658591e-01;

fn softLog(x: f64) f64 {
    var u: u64 = @bitCast(x);
    var hx: u32 = @truncate(u >> 32);
    var k: i32 = 0;

    if (hx < 0x00100000 or hx >> 31 != 0) {
        if (u << 1 == 0) return -std.math.inf(f64); // log(+-0)
        if (hx >> 31 != 0) return std.math.nan(f64); // log(negative)
        u = @bitCast(x * 0x1p54); // subnormal: scale into range
        hx = @truncate(u >> 32);
        k -= 54;
    } else if (hx >= 0x7ff00000) {
        return x; // inf / nan
    } else if (hx == 0x3ff00000 and u << 32 == 0) {
        return 0; // log(1)
    }

    // reduce into [sqrt(2)/2, sqrt(2)]
    hx +%= 0x3ff00000 - 0x3fe6a09e;
    k += @as(i32, @intCast(hx >> 20)) - 0x3ff;
    hx = (hx & 0x000fffff) + 0x3fe6a09e;
    u = (@as(u64, hx) << 32) | (u & 0xffffffff);

    const f = @as(f64, @bitCast(u)) - 1.0;
    const hfsq = 0.5 * f * f;
    const s = f / (2.0 + f);
    const z = s * s;
    const w = z * z;
    const t1 = w * (Lg2 + w * (Lg4 + w * Lg6));
    const t2 = z * (Lg1 + w * (Lg3 + w * (Lg5 + w * Lg7)));
    const dk: f64 = @floatFromInt(k);
    return s * (hfsq + t2 + t1) + dk * ln2lo - hfsq + f + dk * ln2hi;
}

// ---------------------------------------------------------------------------
// sin / cos — musl k_sin.c, k_cos.c, and the medium branch of __rem_pio2
// ---------------------------------------------------------------------------

const pio4 = 0x1.921fb54442d18p-1;
const pio2 = 0x1.921fb54442d18p+0;
const invpio2 = 6.36619772367581382433e-01;
const pio2_1 = 1.57079632673412561417e+00;
const pio2_1t = 6.07710050650619224932e-11;
const pio2_2 = 6.07710050630396597660e-11;
const pio2_2t = 2.02226624879595063154e-21;
const pio2_3 = 2.02226624871116645580e-21;
const pio2_3t = 8.47842766036889956997e-32;
const tau_hi = 6.28318530717958623200e+00;
const tau_lo = 2.44929359829470635445e-16;

const S1 = -1.66666666666666324348e-01;
const S2 = 8.33333333332248946124e-03;
const S3 = -1.98412698298579493134e-04;
const S4 = 2.75573137070700676789e-06;
const S5 = -2.50507602534068634195e-08;
const S6 = 1.58969099521155010221e-10;

const C1 = 4.16666666666666019037e-02;
const C2 = -1.38888888888741095749e-03;
const C3 = 2.48015872894767294178e-05;
const C4 = -2.75573143513906633035e-07;
const C5 = 2.08757232129817482790e-09;
const C6 = -1.13596475577881948265e-11;

/// sin on [-pi/4, pi/4]; `y` is the low half of a double-double argument.
fn kernelSin(x: f64, y: f64, tail: bool) f64 {
    const z = x * x;
    const w = z * z;
    const r = S2 + z * (S3 + z * S4) + z * w * (S5 + z * S6);
    const v = z * x;
    if (!tail) return x + v * (S1 + z * r);
    return x - ((z * (0.5 * y - v * r) - y) - v * S1);
}

/// cos on [-pi/4, pi/4]; `y` is the low half of a double-double argument.
fn kernelCos(x: f64, y: f64) f64 {
    const z = x * x;
    const zz = z * z;
    const r = z * (C1 + z * (C2 + z * C3)) + zz * zz * (C4 + z * (C5 + z * C6));
    const hz = 0.5 * z;
    const w = 1.0 - hz;
    return w + (((1.0 - w) - hz) + (z * r - x * y));
}

/// x = y[0] + y[1] + n*(pi/2), with |y[0]| <= pi/4. Returns n.
fn remPio2(x: f64, y: *[2]f64) i32 {
    // ponytail: no Payne-Hanek. Past 2^20*(pi/2) ~= 1.6e6 rad the Cody-Waite
    // splits stop being exact, so pre-reduce mod 2pi in double-double instead;
    // phase accuracy then decays ~1 bit per octave. A source that needs exact
    // phase beyond that wants musl's __rem_pio2_large table.
    var xr = x;
    if (@abs(xr) >= 0x1p20 * pio2) {
        const q = @round(xr / (tau_hi + tau_lo));
        xr = (xr - q * tau_hi) - q * tau_lo;
    }

    var q: f64 = @round(xr * invpio2);
    var n: i32 = @intFromFloat(q);
    var r = xr - q * pio2_1;
    var w = q * pio2_1t; // 1st round, good to 85 bits
    if (r - w < -pio4) {
        n -= 1;
        q -= 1;
        r = xr - q * pio2_1;
        w = q * pio2_1t;
    } else if (r - w > pio4) {
        n += 1;
        q += 1;
        r = xr - q * pio2_1;
        w = q * pio2_1t;
    }
    y[0] = r - w;

    const ex = expOf(xr);
    if (ex - expOf(y[0]) > 16) { // 2nd round, good to 118 bits
        const t = r;
        w = q * pio2_2;
        r = t - w;
        w = q * pio2_2t - ((t - r) - w);
        y[0] = r - w;
        if (ex - expOf(y[0]) > 49) { // 3rd round, covers the rest
            const t3 = r;
            w = q * pio2_3;
            r = t3 - w;
            w = q * pio2_3t - ((t3 - r) - w);
            y[0] = r - w;
        }
    }
    y[1] = (r - y[0]) - w;
    return n;
}

fn expOf(x: f64) i32 {
    return @intCast(@as(u64, @bitCast(x)) >> 52 & 0x7ff);
}

fn softSin(x: f64) f64 {
    const ax = @abs(x);
    if (ax < pio4) return if (ax < 0x1p-27) x else kernelSin(x, 0.0, false);
    if (!std.math.isFinite(x)) return std.math.nan(f64);
    var y: [2]f64 = undefined;
    const n = remPio2(x, &y);
    return switch (@as(u32, @bitCast(n)) & 3) {
        0 => kernelSin(y[0], y[1], true),
        1 => kernelCos(y[0], y[1]),
        2 => -kernelSin(y[0], y[1], true),
        else => -kernelCos(y[0], y[1]),
    };
}

fn softCos(x: f64) f64 {
    const ax = @abs(x);
    if (ax < pio4) return if (ax < 0x1p-27) 1.0 else kernelCos(x, 0.0);
    if (!std.math.isFinite(x)) return std.math.nan(f64);
    var y: [2]f64 = undefined;
    const n = remPio2(x, &y);
    return switch (@as(u32, @bitCast(n)) & 3) {
        0 => kernelCos(y[0], y[1]),
        1 => -kernelSin(y[0], y[1], true),
        2 => -kernelCos(y[0], y[1]),
        else => kernelSin(y[0], y[1], true),
    };
}

// ---------------------------------------------------------------------------
// Tests. The soft paths never run on the host, so check them against the
// builtins directly. The derived functions (tanh/sinh/cosh/pow/tan) run their
// device algorithm here with the host's exp/log underneath — composition costs
// a couple of ulp on top of what softExp/softLog are proven to give above.
// The other half of the check is that nvptx/amdgcn objects build at all.
// ---------------------------------------------------------------------------

fn ulpErr(comptime T: type, got: T, want: T) T {
    if (got == want) return 0;
    const scale = @max(@abs(want), std.math.floatMin(T));
    return @abs(got - want) / (scale * std.math.floatEps(T));
}

test "musl core matches the builtins" {
    const ulp = struct {
        fn e(got: f64, want: f64) f64 {
            return ulpErr(f64, got, want);
        }
    }.e;

    // exp: a wide dynamic range plus the edges of the argument reduction.
    for ([_]f64{ 0, 1e-30, -1e-30, 1e-9, 0.1, -0.34, 0.35, 1, -1, 40, 80, -80, 700, -700, -745, 710 }) |x| {
        try std.testing.expect(ulp(softExp(x), @exp(x)) <= 2);
        try std.testing.expect(ulp(softExp2(x), @exp2(x)) <= 2);
    }
    // exp2 must be exact on integers.
    for ([_]f64{ -60, -1, 0, 1, 10, 63, 1023 }) |x| {
        try std.testing.expectEqual(@exp2(x), softExp2(x));
    }
    try std.testing.expect(std.math.isNan(softExp2(std.math.nan(f64))));
    try std.testing.expectEqual(std.math.inf(f64), softExp2(5000));
    try std.testing.expectEqual(@as(f64, 0), softExp2(-5000));

    // log: subnormal, exactly 1, and both sides of the sqrt(2) fold.
    for ([_]f64{ 0x1p-1060, 1e-30, 0.5, 0.7071, 1.0, 1.5, 2.0, 1e6, 1e300 }) |x| {
        try std.testing.expect(ulp(softLog(x), @log(x)) <= 2);
    }
    // log2/log10 are log * a rounded constant. Subnormals are excluded: the
    // rebased result there is ~20x larger than log's own, so log's half-ulp
    // lands as ~22 ulp of it — the ceiling named on `log2` above.
    for ([_]f64{ 1e-30, 0.5, 0.7071, 1.5, 2.0, 1e6, 1e300 }) |x| {
        try std.testing.expect(ulp(softLog(x) * log2e, @log2(x)) <= 4);
        try std.testing.expect(ulp(softLog(x) * log10e, @log10(x)) <= 4);
    }
    try std.testing.expect(std.math.isNegativeInf(softLog(0)));
    try std.testing.expect(std.math.isNan(softLog(-1)));

    // sin/cos: below pi/4, across the quadrant switches, and into the
    // pre-reduced range above 2^20*(pi/2).
    for ([_]f64{ 0, 1e-9, 0.3, 0.786, 1.5708, 3.14159, -2.5, 100.0, 1e5, 1.5e6 }) |x| {
        try std.testing.expect(ulp(softSin(x), @sin(x)) <= 4);
        try std.testing.expect(ulp(softCos(x), @cos(x)) <= 4);
    }
    // Past the Cody-Waite ceiling accuracy is only asserted loosely.
    try std.testing.expectApproxEqAbs(@sin(1e9), softSin(1e9), 1e-6);
}

test "derived functions match std.math" {
    inline for (.{ f32, f64 }) |T| {
        const ulp = struct {
            fn e(got: T, want: T) T {
                return ulpErr(T, got, want);
            }
        }.e;

        for ([_]T{ 0, 1e-9, 0.1, -0.3, 0.624, 0.626, 1, -1, 3, -3, 9, 40 }) |x| {
            try std.testing.expect(ulp(tanh(x), std.math.tanh(x)) <= 4);
        }
        try std.testing.expectEqual(@as(T, 1), tanh(std.math.inf(T)));
        try std.testing.expectEqual(@as(T, -1), tanh(-std.math.inf(T)));

        for ([_]T{ 0, 1e-12, 0.4, -0.4, 0.5, 1, -3, 20, -20 }) |x| {
            try std.testing.expect(ulp(sinh(x), std.math.sinh(x)) <= 4);
            try std.testing.expect(ulp(cosh(x), std.math.cosh(x)) <= 4);
        }

        for ([_]T{ 1e-6, 0.5, 1.5, 2, 10, 1e6 }) |x| {
            for ([_]T{ 0, 1, 2, 2.5, -1.5, 0.33, -3 }) |y| {
                try std.testing.expect(ulp(pow(x, y), std.math.pow(T, x, y)) <= 64);
            }
        }
        try std.testing.expectEqual(@as(T, -8), pow(@as(T, -2), 3));
        try std.testing.expectEqual(@as(T, 4), pow(@as(T, -2), 2));
        try std.testing.expect(std.math.isNan(pow(@as(T, -2), 0.5)));

        for ([_]T{ 0.1, 1, -1, 2 }) |x| {
            try std.testing.expect(ulp(tan(x), @sin(x) / @cos(x)) <= 2);
        }
        for ([_]T{ 0.25, 1, 4, 1e6 }) |x| {
            try std.testing.expectEqual(@sqrt(x), sqrt(x));
            try std.testing.expect(ulp(rsqrt(x), 1 / @sqrt(x)) <= 1);
        }
        // The dispatchers themselves, on the host, are pass-throughs.
        for ([_]T{ 0.5, 1, 2, 7.25 }) |x| {
            try std.testing.expectEqual(@exp(x), exp(x));
            try std.testing.expectEqual(@exp2(x), exp2(x));
            try std.testing.expectEqual(@log(x), log(x));
            try std.testing.expectEqual(@log2(x), log2(x));
            try std.testing.expectEqual(@log10(x), log10(x));
            try std.testing.expectEqual(@sin(x), sin(x));
            try std.testing.expectEqual(@cos(x), cos(x));
        }
    }
}
