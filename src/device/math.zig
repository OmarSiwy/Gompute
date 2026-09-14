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
//! `expm1` and `atan` fail differently and only on AMD, which is how they went
//! unnoticed: both assemble to PTX cleanly, and both die on AMDGCN. Not for
//! want of a libcall — std's ports raise the subnormal underflow flag through
//! `std.mem.doNotOptimizeAway`, which for a float is `asm volatile ("" :: "rm"
//! (v))`, and the AMDGPU backend cannot match the `m` alternative. That flag is
//! a register no GPU exposes, so the idiom is dead weight on device and a hard
//! error there. See `expm1` for the port and `atan` for the cheaper dodge.
//!
//! `exp exp2 log log2 log10 pow` are ports of ARM optimized-routines, ONE body
//! each for host and device, f32 and f64 (only f32 `exp2` still uses hardware).
//! Table-driven, branch-light, no libm and no f64 builtin beyond `+ - * /` and
//! bit casts, so the same source survives NVPTX and AMDGCN unchanged.
//!
//! That is a correctness decision before it is a speed one. glibc >= 2.28 IS
//! ARM optimized-routines for exactly these functions, and the simulators that
//! consume this module are scored against ngspice, which links glibc. What Zig
//! gives you instead — `@exp` and friends, and `extern "c" fn exp` too, since
//! compiler_rt's static definition beats the shared libm — is musl: a
//! DIFFERENT algorithm that disagrees with the reference in the low bits of
//! every model evaluation. Porting moves the arithmetic toward the reference,
//! not away from it. (Measured: compiler_rt's `log2` is ~23 ulp out on
//! subnormals, where glibc's and this one are exact.)
//!
//! Measured against the real glibc (dlsym'd past compiler_rt) at the bottom of
//! this file, >=1e6 points each over the full domain including subnormals, the
//! saturation thresholds and the near-1 window:
//!
//!   exp exp2 log log2 pow        <=1 ulp, f32 and f64
//!   expf logf log2f log10f       <=1 ulp
//!   log10 f64                    <=2 ulp against glibc, but 0.52 ulp against
//!                                a 60-digit reference: the slack is glibc's,
//!                                whose log10 is not optimized-routines.
//!
//! What is still hardware, and still not IEEE (measured on an RTX 4060 / sm_89
//! over x in (0, 8], 1024 samples):
//!
//!   f32 exp2       `ex2.approx.f32` / `v_exp_f32`, <=1 ulp RELATIVE, kept.
//!   f32/f64 sin cos  `sin.approx.f32` is ~1e-6 ABSOLUTE, so it has no
//!                  relative accuracy near k*pi; f64 goes through the musl
//!                  ports below, <=1.4 ulp. See `sin` and `remPio2`.
//!
//! Want IEEE-grade f32 for something not in the list? Compute in f64 and
//! `@floatCast` — that path is software and correct, just slow (1/64 rate on
//! consumer NVIDIA).
//!
//! Every entry point takes a scalar `f32`/`f64` or a `@Vector` of either. That
//! is not decoration: the CPU backend instantiates a generic map body at
//! `@Vector` width, so a kernel written the way the guide recommends hands
//! these functions vectors. Only the builtins are elementwise (`sin`, `cos`,
//! `tan`, `sqrt`, `rsqrt`, f32 `exp2`); the ported bodies run lane by lane.
//! See `apply`.
//!
//! Already safe everywhere and deliberately absent here: `@abs @trunc @round
//! @floor @ceil @copysign @min @max`, `std.math.scalbn/frexp/modf`. `sqrt` and
//! `rsqrt` ARE here despite being safe, so that a kernel need not keep two
//! lists in its head — they are `@sqrt` with the argument checked.
//!
//! `@mulAdd` is NOT in that list. It is one instruction only where the target
//! actually has an FMA; anywhere else it lowers to a `fma()` CALL into libm,
//! which is slow on the host and a link error on device. Gate it on `fast_fma`
//! below, never use it unconditionally.

const std = @import("std");
const builtin = @import("builtin");

const arch = builtin.cpu.arch;
const dev = arch == .nvptx64 or arch == .amdgcn;

/// The element type a body computes in: `T` itself for a scalar, the child for
/// a vector.
///
/// Vectors are accepted because the CPU backend instantiates a generic map body
/// at `@Vector` width -- see `Spec.is_generic` and `lanes` in
/// `src/host/kernel.zig`. Rejecting them made `g.math` a compile error in
/// exactly the kernel shape the guide recommends for CPU speed.
fn Elem(comptime T: type) type {
    const E = switch (@typeInfo(T)) {
        .vector => |v| v.child,
        else => T,
    };
    return switch (E) {
        f32, f64 => E,
        else => @compileError("gompute.math supports f32, f64 and vectors of them, not " ++
            @typeName(T) ++ " — cast first"),
    };
}

fn isVec(comptime T: type) bool {
    return @typeInfo(T) == .vector;
}

/// A scalar body `f`, applied to a scalar, or one lane at a time to a vector.
///
/// `f` is always the scalar body, never the public entry point above it: Zig
/// rejects `exp -> apply -> exp` as an inline recursion cycle even though the
/// inner call is the scalar instantiation that terminates it.
///
/// ponytail: a lane loop, not a vector algorithm. Every ported body below
/// indexes a table with the input (`exp_tab[idx]`) and branches on it (the
/// saturation ends) -- a gather and a divergence, the two things `@Vector` does
/// not have. Lane-parallel transcendentals exist (ARM's own vector routines,
/// SLEEF) and are a rewrite of every body here, not a wrapper. This is what
/// makes a vectorized map body COMPILE and be correct; if a profile ever blames
/// the lane loop, that rewrite is the upgrade path. `sin` and `cos` on the
/// host, `tan`, `sqrt`, `rsqrt` and f32 `exp2` never reach it -- their builtins
/// are already elementwise.
inline fn apply(comptime f: anytype, x: anytype) @TypeOf(x) {
    if (comptime !isVec(@TypeOf(x))) return f(x);
    var out: @TypeOf(x) = undefined;
    inline for (0..@typeInfo(@TypeOf(x)).vector.len) |i| out[i] = f(x[i]);
    return out;
}

/// `apply` for `pow`, the one entry point that takes two arguments.
inline fn apply2(comptime f: anytype, x: anytype, y: @TypeOf(x)) @TypeOf(x) {
    if (comptime !isVec(@TypeOf(x))) return f(x, y);
    var out: @TypeOf(x) = undefined;
    inline for (0..@typeInfo(@TypeOf(x)).vector.len) |i| out[i] = f(x[i], y[i]);
    return out;
}

// ---------------------------------------------------------------------------
// Public API. exp/exp2/log/log2/log10/pow are one body each, host and device;
// the rest dispatch on the element type.
//
// Every entry point here is a dispatcher: it picks a scalar body and hands it
// to `apply`, which runs it once or once per lane. Bodies never see a vector,
// which is why `tanh`, `sinh`, `cosh` and `pow` keep theirs in a `*Body`
// function rather than inline in the public one.
// ---------------------------------------------------------------------------

/// e^x, <=1 ulp on host AND device from one body: ARM optimized-routines
/// `exp`/`expf`, which is what glibc >= 2.28 ships. See the note on `pow` for
/// why matching the reference's arithmetic is worth more than the speed.
pub inline fn exp(x: anytype) @TypeOf(x) {
    if (Elem(@TypeOf(x)) == f32) return apply(softExpf, x);
    return apply(softExp, x);
}

/// 2^x, exact for integer x. Same table as `exp`, its own reduction.
///
/// ponytail: f32 stays on `@exp2`. `ex2.approx.f32`/`v_exp_f32` are RELATIVE
/// error devices measured at <=1 ulp, so unlike log there is no hole to close,
/// and one instruction beats ten f64 ops at 1/64 rate. Port `exp2f.c` if an
/// f32 caller ever needs the last half ulp.
pub inline fn exp2(x: anytype) @TypeOf(x) {
    if (Elem(@TypeOf(x)) == f32) return @exp2(x); // elementwise on a vector
    return apply(softExp2, x);
}

/// Natural log, <=1 ulp on host AND device from one body: ARM
/// optimized-routines `log` for f64, `logf` for f32.
///
/// The f32 half is what closed this module's worst accuracy hole. It used to
/// be `lg2.approx.f * ln2`, and that hardware is bounded ABSOLUTELY (~2^-21),
/// not relatively — so log(x) for x near 1 was 2^-21 of noise on a near-zero
/// answer, which is exactly where a SPICE junction sits.
pub inline fn log(x: anytype) @TypeOf(x) {
    if (Elem(@TypeOf(x)) == f32) return apply(softLnf, x);
    return apply(softLog, x);
}

/// Base-2 log, <=1 ulp on host AND device: ARM optimized-routines
/// `log2`/`log2f`, with its own 64-entry table rather than `log * 1/ln2`.
/// Powers of two come back exact, which the scaled form could not manage.
pub inline fn log2(x: anytype) @TypeOf(x) {
    if (Elem(@TypeOf(x)) == f32) return apply(softLog2f, x);
    return apply(softLog2, x);
}

/// Base-10 log.
///
/// f32 is ARM's `log10f`, which is `logf` with 1/ln10 folded into the binary64
/// accumulator before the single rounding — a real log10, not a scaled log.
///
/// f64 is `log(x) * 1/ln10`, and it is the one function here that is not a
/// port: ARM optimized-routines has no f64 log10, and neither does glibc —
/// glibc's is still its own `e_log10.c`, so there is no reference arithmetic
/// to converge on. That costs one extra rounding on top of log's 0.52 ulp
/// (analytically ~1.02 ulp; the measured worst is in the differential test at
/// the bottom of this file). ponytail: the fix, if a caller ever needs it, is
/// to have `softLog` hand back its hi/lo pair and scale THAT, not a new table.
pub inline fn log10(x: anytype) @TypeOf(x) {
    if (Elem(@TypeOf(x)) == f32) return apply(softLog10f, x);
    return apply(softLog10, x);
}

/// Measured sm_89 f32: ~1e-6 ABSOLUTE, which is <=29 ulp relative away from the
/// zeros and unbounded at them — `sin.approx.f32`/`v_sin_f32` are absolute-error
/// devices, and their own range reduction gives out for large |x|. f64 matched
/// glibc bit-for-bit over (0,8]; see `remPio2` for its ceiling.
pub inline fn sin(x: anytype) @TypeOf(x) {
    const E = Elem(@TypeOf(x));
    if (!dev) return @sin(x); // elementwise on a vector
    if (E == f32) return apply(hwSin, x);
    return apply(softSin, x);
}

/// Accuracy as `sin`.
pub inline fn cos(x: anytype) @TypeOf(x) {
    const E = Elem(@TypeOf(x));
    if (!dev) return @cos(x); // elementwise on a vector
    if (E == f32) return apply(hwCos, x);
    return apply(softCos, x);
}

/// ponytail: sin/cos, so error blows up near the poles where cos goes to zero
/// (measured 0.16 absolute at x = 3pi/2 in f32). A dedicated tan with its own
/// argument reduction is worth writing only if someone is actually near pi/2.
pub inline fn tan(x: anytype) @TypeOf(x) {
    _ = Elem(@TypeOf(x));
    return sin(x) / cos(x); // both already handle a vector
}

/// Cephes rational below 0.625, else 1 - 2/(e^2|x| + 1).
/// Measured sm_89: f32 <=1.8 ulp, f64 <=1 ulp.
pub inline fn tanh(x: anytype) @TypeOf(x) {
    _ = Elem(@TypeOf(x));
    return apply(tanhBody, x);
}

inline fn tanhBody(x: anytype) @TypeOf(x) {
    const T = @TypeOf(x);
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
    _ = Elem(@TypeOf(x));
    return apply(sinhBody, x);
}

inline fn sinhBody(x: anytype) @TypeOf(x) {
    const T = @TypeOf(x);
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
    _ = Elem(@TypeOf(x));
    return apply(coshBody, x);
}

inline fn coshBody(x: anytype) @TypeOf(x) {
    const T = @TypeOf(x);
    if (!dev) return std.math.cosh(x);
    const e = exp(@abs(x));
    return @as(T, 0.5) * e + @as(T, 0.5) / e;
}

/// e^x - 1, without the cancellation `exp(x) - 1` suffers near zero.
///
/// Here because `std.math.expm1` DOES NOT COMPILE FOR AMDGCN. Its tiny-argument
/// branch raises the underflow flag through `std.mem.doNotOptimizeAway`, which
/// for a float lowers to `asm volatile ("" :: "rm" (v))`, and the AMDGPU backend
/// cannot match the `m` alternative: `LLVM ERROR: Could not match memory
/// address. Inline asm failure!`. NVPTX assembles the same source to PTX
/// without complaint, so the hole is AMD-only and invisible on an NVIDIA box.
///
/// The body is std's own musl port with that one line dropped. The line only
/// ever set an IEEE exception flag, and no GPU exposes one to read, so nothing
/// on either target loses a value it could have observed — every return here is
/// bit-identical to `std.math.expm1`.
///
/// ponytail: `log1p` is std's on both targets. Its port happens not to contain
/// the idiom, so it compiles for AMDGCN today; bring it here if that changes.
/// Upstream, the real fix is AMDGPU joining the carve-out list `doNotOptimizeAway`
/// already keeps for LoongArch and stage2_c.
pub inline fn expm1(x: anytype) @TypeOf(x) {
    if (Elem(@TypeOf(x)) == f32) return apply(expm1f, x);
    return apply(softExpm1, x);
}

/// f32 through the f64 body, per the module header: one rounding of a correctly
/// rounded f64 result is itself correctly rounded, and it saves porting the
/// second half of the reference.
fn expm1f(x: f32) f32 {
    return @floatCast(softExpm1(x));
}

/// arctangent.
///
/// Same AMDGCN hole as `expm1` — std's SCALAR atan raises the subnormal
/// underflow flag through `doNotOptimizeAway` — but the fix is different,
/// because `std.math.atan` carries a VECTOR path that never reaches it.
///
/// The two paths are not bit-identical: over 400k samples in [-200, 200] they
/// disagree on 32% of inputs, by at most 2.22e-16 relative — one ulp, the
/// ordinary gap between two polynomial approximations. The host therefore keeps
/// the scalar body, so nothing already shipped moves.
pub inline fn atan(x: anytype) @TypeOf(x) {
    _ = Elem(@TypeOf(x));
    return apply(atanBody, x);
}

inline fn atanBody(x: anytype) @TypeOf(x) {
    if (!dev) return std.math.atan(x);
    // ponytail: TWO lanes, and the width is the whole point. `@Vector(1, f64)`
    // does not merely fail to select — it SEGVs the AMDGPU backend, so one lane
    // is not an option and two is the cheapest that is. The second result is
    // discarded, so device `atan` costs twice what it should. No GPU-eligible
    // device calls `atan` today; if one ever does, port std's `atanBinary64`
    // and `atanBinary32` minus their `doNotOptimizeAway` line, the way
    // `softExpm1` does. `tests/device_entries.zig` compiles this for AMDGCN, so
    // if the backend's handling of narrow vectors shifts again, the build says
    // so rather than a GPU silently returning the wrong angle.
    const v: @Vector(2, @TypeOf(x)) = @splat(x);
    return std.math.atan(v)[0];
}

/// x^y, <=0.52 ulp on host AND device, from one body.
///
/// Everything past the two fast paths is `softPow` below: the ARM
/// optimized-routines algorithm, which is what glibc >= 2.28 ships as its own
/// `pow`. It replaced two different things at once —
///
///   host    `std.math.pow`, the Go/FreeBSD algorithm: `modf`, `exp(yf*ln x)`
///           for the fractional part, `frexp`, a data-dependent binary-
///           exponentiation loop and `scalbn`, behind ~15 special-case
///           branches. Callgrind on an i9-14900HX: 265 instructions/call,
///           3 ulp.
///   device  `exp2(y * log2 x)`, whose relative error grows as |y*log2 x| ulps
///           (~1e-14 at y*ln x = -80) — measured 4 ulp.
///
/// It matters because every SPICE junction grading coefficient is a
/// non-integer model parameter in (0, 1.2) — NC=1.1739, 1-MJ=0.693,
/// 1-MJSW=0.351 — so none of them hit a `y == 0.5 -> @sqrt` fast path and
/// every call pays the full route. ARPice devices/mos6_inverter issues ~299k
/// of them: 8.2% of the whole run. It also moves the arithmetic TOWARD the
/// reference rather than away: ngspice calls this same algorithm via glibc.
///
/// Deliberately NOT `extern "c" fn pow`. gompute has to build freestanding and
/// has to build for NVPTX/AMDGCN, where there is no libm at all; and one Zig
/// body can later be inlined and vectorized where a libm call cannot.
///
/// Two fast paths stay in front of it, because they beat it on their own
/// inputs:
///
///   y == 0 or x == 1   C99 says 1 even when the other operand is NaN. One
///           compare each, and it takes the commonest exits off the table path.
///   integer |y| <= 64  square-and-multiply. `x*x` is a single correctly
///           rounded multiply where the table route is 0.52 ulp, `**2`/`**3`
///           are everywhere in device models, and matching what the naive
///           expression computes is worth more there than the 0.02 ulp
///           `softPow` would buy back on `**3` — at ~130 instructions less.
///
/// f32 goes through the f64 body: one algorithm, one test surface, and a
/// single f32 rounding of a 0.52-ulp f64 result is correctly rounded. Nothing
/// in gompute or its consumers calls `pow` on f32 (Verilog-A `real` is f64),
/// so the device-side f64 rate penalty buys accuracy nobody pays for.
/// ponytail: if an f32 kernel ever wants a cheap pow, `exp2(y * log2 x)` on
/// the f32 hardware path is the thing to bring back, for f32 only.
pub inline fn pow(x: anytype, y: @TypeOf(x)) @TypeOf(x) {
    _ = Elem(@TypeOf(x));
    return apply2(powBody, x, y);
}

inline fn powBody(x: anytype, y: @TypeOf(x)) @TypeOf(x) {
    const T = @TypeOf(x);
    if (y == 0 or x == 1) return 1;
    if (y == @trunc(y) and @abs(y) <= 64) {
        var n: u32 = @intFromFloat(@abs(y));
        var base = x;
        var acc: T = 1;
        while (n != 0) : (n >>= 1) {
            if (n & 1 != 0) acc *= base;
            base *= base;
        }
        if (y > 0) return acc;
        // `1/acc` is the answer only if acc neither overflowed nor underflowed
        // on the way there: pow(1e160, -2) squares to inf and 1/inf = 0 throws
        // away a perfectly representable subnormal. When it did, fall through —
        // softPow scales in the exponent field and gets it right.
        if (acc != 0 and !std.math.isInf(acc)) return 1 / acc;
    }
    if (T == f32) return @floatCast(softPow(@floatCast(x), @floatCast(y)));
    return softPow(x, y);
}

/// Native instruction on both back ends; here so callers need not remember
/// which builtins are device-safe.
pub inline fn sqrt(x: anytype) @TypeOf(x) {
    _ = Elem(@TypeOf(x));
    return @sqrt(x); // elementwise on a vector
}

/// ponytail: 1/sqrt, i.e. two IEEE ops. Swap in `rsqrt.approx.f32`/`v_rsq_f32`
/// if a normalization loop ever shows up hot in a profile.
pub inline fn rsqrt(x: anytype) @TypeOf(x) {
    const T = @TypeOf(x);
    _ = Elem(T);
    const one: T = if (comptime isVec(T)) @splat(1) else 1;
    return one / @sqrt(x); // elementwise on a vector
}

// ---------------------------------------------------------------------------
// f32 device primitives — hardware approximations. Only sin/cos still use
// them; log2 lost its `lg2.approx.f` when `softLog2f` landed below.
// ---------------------------------------------------------------------------

extern fn @"llvm.nvvm.sin.approx.f"(f32) callconv(.c) f32;
extern fn @"llvm.nvvm.cos.approx.f"(f32) callconv(.c) f32;

inline fn hwSin(x: f32) f32 {
    return if (arch == .nvptx64) @"llvm.nvvm.sin.approx.f"(x) else @sin(x);
}
inline fn hwCos(x: f32) f32 {
    return if (arch == .nvptx64) @"llvm.nvvm.cos.approx.f"(x) else @cos(x);
}

/// 1/ln10, correctly rounded, for the f32 `log10` — ARM's `log10f` folds it
/// into the binary64 accumulator, so one multiply is enough there.
const invln10 = 0x1.bcb7b1526e50ep-2;

/// 1/ln10 again, as a double-double: `l10hi` keeps 25 significant bits and
/// `l10hi + l10lo` is 1/ln10 to 2^-82. `softLog10` needs both.
const l10hi = 0x1.bcb7b10000000p-2;
const l10lo = 0x1.49b9438ca9aaep-28;

// ---------------------------------------------------------------------------
// pow — ARM optimized-routines math/pow.c (MIT), the glibc >= 2.28 algorithm.
//
// log(x) to ~68 bits as a double-double out of a 128-entry table, y * that in
// double-double, then a 128-entry exp with the exponent folded into the table
// entry's bit pattern. No loops, no libm, no f64 builtin beyond +-*/ and bit
// casts, so it survives NVPTX and AMDGCN codegen unchanged.
//
// Published worst case 0.54 ulp; see `math_data.zig` for the tables and the
// test at the bottom of this file for what we measure.
// ---------------------------------------------------------------------------

const md = @import("math_data.zig");

/// True where `@mulAdd(f64, ...)` is one instruction. Anywhere else it lowers
/// to a `fma()` CALL into libm, which is both catastrophic for speed and a
/// link error on device — so the reference's `#if __FP_FAST_FMA` split is a
/// correctness gate here, not just a tuning knob. Both variants are ported.
const fast_fma = switch (arch) {
    .x86_64 => std.Target.x86.featureSetHas(builtin.cpu.features, .fma),
    .aarch64, .aarch64_be => true,
    .nvptx64, .amdgcn => true,
    else => false,
};

/// Top 12 bits of a double: sign and biased exponent.
inline fn top12(x: f64) u32 {
    return @truncate(@as(u64, @bitCast(x)) >> 52);
}

/// 0 if not an integer, 1 if an odd integer, 2 if an even one. `iy` must be
/// the bit pattern of a non-zero finite double.
fn checkint(iy: u64) u32 {
    const e: u32 = @truncate(iy >> 52 & 0x7ff);
    if (e < 0x3ff) return 0; // |y| < 1
    if (e > 0x3ff + 52) return 2; // no fractional bits left, and even
    const sh: u6 = @intCast(0x3ff + 52 - e);
    if (iy & ((@as(u64, 1) << sh) - 1) != 0) return 0;
    if (iy & (@as(u64, 1) << sh) != 0) return 1;
    return 2;
}

/// True for the bit pattern of 0, inf or nan.
inline fn zeroinfnan(i: u64) bool {
    return 2 *% i -% 1 >= 2 *% @as(u64, 0x7ff0000000000000) - 1;
}

const one_bits: u64 = 0x3ff0000000000000;

/// log(x) as y + tail, carrying about 15 bits past the double. `ix` is x's bit
/// pattern, already normalized out of the subnormal range by the caller.
fn logInline(ix: u64, tail: *f64) f64 {
    // x = 2^k z with z in [OFF, 2*OFF) exactly; the range is cut into 128
    // subintervals and c sits near the centre of the one z lands in.
    const off: u64 = 0x3fe6955500000000;
    const tmp = ix -% off;
    const i: usize = @intCast((tmp >> (52 - 7)) & 127);
    const k = @as(i64, @bitCast(tmp)) >> 52; // arithmetic
    const iz = ix -% (tmp & (@as(u64, 0xfff) << 52));
    const z: f64 = @bitCast(iz);
    const kd: f64 = @floatFromInt(k);

    const e = md.powlog_tab[i];
    const invc = e.invc;

    // 1/c is j/128 or j/256 for integer j, and |z/c - 1| < 1/128, so
    // r = z/c - 1 is exactly representable. Without an FMA it takes a split of
    // z into halves whose products are exact. Both are computed; the unused
    // half of the pair is dead code the backend drops.
    const zhi: f64 = @bitCast((iz +% (1 << 31)) & (~@as(u64, 0) << 32));
    const zlo = z - zhi;
    const rhi = zhi * invc - 1.0;
    const rlo = zlo * invc;
    const r = if (fast_fma) @mulAdd(f64, z, invc, -1.0) else rhi + rlo;

    // k*ln2 + log(c) + r, in double-double.
    const t1 = kd * md.powlog_ln2hi + e.logc;
    const t2 = t1 + r;
    const lo1 = kd * md.powlog_ln2lo + e.logctail;
    const lo2 = t1 - t2 + r;

    // Ordered for a superscalar pipeline, not for readability.
    const a = md.powlog_poly;
    const ar = a[0] * r; // a[0] = -0.5
    const ar2 = r * ar;
    const ar3 = r * ar2;
    const hi, const lo3, const lo4 = if (fast_fma) blk: {
        const hi = t2 + ar2;
        break :blk .{ hi, @mulAdd(f64, ar, r, -ar2), t2 - hi + ar2 };
    } else blk: {
        const arhi = a[0] * rhi;
        const arhi2 = rhi * arhi;
        const hi = t2 + arhi2;
        break :blk .{ hi, rlo * (ar + arhi), t2 - hi + arhi2 };
    };
    // p = log1p(r) - r - a[0]*r*r.
    const p = ar3 * (a[1] + r * a[2] + ar2 * (a[3] + r * a[4] + ar2 * (a[5] + r * a[6])));

    const lo = lo1 + lo2 + lo3 + lo4 + p;
    const y = hi + lo;
    tail.* = hi - y + lo;
    return y;
}

const sign_bias_bit: u32 = 0x800 << 7;

/// The exponent of `scale` may have run into the sign bit, so it is passed as
/// bits and fixed up before use. Positive k means the result may overflow,
/// negative k means it may land in the subnormals.
///
/// `oflow` is how far the exponent can have run past the top: 1009 for `exp`
/// and `pow`, whose k comes from a 128-way split of a ~1420-wide range, but
/// only 1 for `exp2`, whose k IS the exponent. Shared because the subnormal
/// half — the delicate half — is identical.
fn expSpecial(comptime oflow: comptime_int, tmp: f64, sbits_in: u64, ki: u64) f64 {
    var sbits = sbits_in;
    if (ki & 0x80000000 == 0) {
        // k > 0: back the exponent off, then put the scale back afterwards.
        const rescale: f64 = comptime @bitCast(@as(u64, 1023 + oflow) << 52);
        sbits -%= @as(u64, oflow) << 52;
        const scale: f64 = @bitCast(sbits);
        return (scale + scale * tmp) * rescale;
    }
    sbits +%= @as(u64, 1022) << 52; // sbits is a signed scale here
    const scale: f64 = @bitCast(sbits);
    var y = scale + scale * tmp;
    if (@abs(y) < 1.0) {
        // Round y to full precision BEFORE scaling it into the subnormals;
        // otherwise the double rounding costs 0.5 + E/2 ulp instead of E.
        const one: f64 = if (y < 0.0) -1.0 else 1.0;
        var lo = scale - y + scale * tmp;
        const hi = one + y;
        lo = one - hi + y + lo;
        y = (hi + lo) - one;
        if (y == 0.0) y = @bitCast(sbits & 0x8000000000000000); // sign of zero
    }
    return 0x1p-1022 * y;
}

/// sign * exp(x + xtail), with |xtail| < 2^-15 and |xtail| <= |x|.
/// `sign_bias` is `sign_bias_bit` for a negative result and 0 otherwise.
fn expInline(x: f64, xtail: f64, sign_bias: u32) f64 {
    const tiny_top = comptime top12(0x1p-54);
    var abstop: u32 = top12(x) & 0x7ff;
    if (abstop -% tiny_top >= comptime top12(512.0) - top12(0x1p-54)) {
        if (abstop -% tiny_top >= 0x80000000) {
            // |x| < 2^-54, and 0 is a common input: no spurious underflow.
            const one = 1.0 + x;
            return if (sign_bias != 0) -one else one;
        }
        if (abstop >= comptime top12(1024.0)) {
            // inf and nan were handled by the caller.
            if (@as(u64, @bitCast(x)) >> 63 != 0)
                return if (sign_bias != 0) -@as(f64, 0) else 0;
            return if (sign_bias != 0) -std.math.inf(f64) else std.math.inf(f64);
        }
        abstop = 0; // large x: special-cased after the polynomial
    }

    // exp(x) = 2^(k/128) * exp(r), r in [-ln2/256, ln2/256].
    const z = md.invln2N * x;
    const shifted = z + md.shift; // forces the round-to-nearest-int
    const ki: u64 = @bitCast(shifted);
    const kd = shifted - md.shift;
    const r = x + kd * md.negln2hiN + kd * md.negln2loN + xtail;

    const idx: usize = @intCast(2 * (ki & 127));
    const top = (ki +% sign_bias) << (52 - 7);
    const tail: f64 = @bitCast(md.exp_tab[idx]);
    const sbits = md.exp_tab[idx + 1] +% top; // valid while -1023*128 < k < 1024*128

    const c = md.exp_poly;
    const r2 = r * r;
    const tmp = tail + r + r2 * (c[0] + r * c[1]) + r2 * r2 * (c[2] + r * c[3]);
    if (abstop == 0) return expSpecial(1009, tmp, sbits, ki);
    const scale: f64 = @bitCast(sbits);
    // tmp is 0 or |tmp| > 2^-200 and scale > 2^-739, so no spurious underflow.
    return scale + scale * tmp;
}

fn softPow(x: f64, y: f64) f64 {
    var sign_bias: u32 = 0;
    var ix: u64 = @bitCast(x);
    const iy: u64 = @bitCast(y);
    var topx = top12(x);
    const topy = top12(y);

    // |y| > 1075*ln2*2^53 makes the answer inf or 0; |y| < 2^-54/1075 makes it
    // +-1. Everything else here is a genuine special value.
    if (topx -% 1 >= 0x7ff - 1 or (topy & 0x7ff) -% 0x3be >= 0x43e - 0x3be) {
        if (zeroinfnan(iy)) {
            if (2 *% iy == 0) return 1.0;
            if (ix == one_bits) return 1.0;
            if (2 *% ix > 2 *% @as(u64, 0x7ff0000000000000) or
                2 *% iy > 2 *% @as(u64, 0x7ff0000000000000)) return x + y; // nan
            if (2 *% ix == 2 *% one_bits) return 1.0; // pow(-1, +-inf)
            // |x| < 1 && y == inf, or |x| > 1 && y == -inf.
            if ((2 *% ix < 2 *% one_bits) == (iy >> 63 == 0)) return 0.0;
            return y * y; // +inf
        }
        if (zeroinfnan(ix)) {
            var x2 = x * x;
            if (ix >> 63 != 0 and checkint(iy) == 1) x2 = -x2;
            return if (iy >> 63 != 0) 1 / x2 else x2;
        }
        // x and y are non-zero and finite from here.
        if (ix >> 63 != 0) {
            const yint = checkint(iy);
            if (yint == 0) return std.math.nan(f64); // negative^non-integer
            if (yint == 1) sign_bias = sign_bias_bit;
            ix &= 0x7fffffffffffffff;
            topx &= 0x7ff;
        }
        if ((topy & 0x7ff) -% 0x3be >= 0x43e - 0x3be) {
            // sign_bias is 0 here: such a y cannot be an odd integer.
            if (ix == one_bits) return 1.0;
            // |y| < 2^-65: x^y ~= 1 + y*log(x), and the sign of the correction
            // is all that survives the rounding.
            if ((topy & 0x7ff) < 0x3be) return if (ix > one_bits) 1.0 + y else 1.0 - y;
            return if ((ix > one_bits) == (topy < 0x800)) std.math.inf(f64) else 0;
        }
        if (topx == 0) {
            // Normalize a subnormal x so its exponent goes negative.
            ix = @bitCast(x * 0x1p52);
            ix &= 0x7fffffffffffffff;
            ix -%= @as(u64, 52) << 52;
        }
    }

    var lo: f64 = undefined;
    const hi = logInline(ix, &lo);
    const ehi, const elo = if (fast_fma) blk: {
        const e = y * hi;
        break :blk .{ e, y * lo + @mulAdd(f64, y, hi, -e) };
    } else blk: {
        const yhi: f64 = @bitCast(iy & (~@as(u64, 0) << 27));
        const ylo = y - yhi;
        const lhi: f64 = @bitCast(@as(u64, @bitCast(hi)) & (~@as(u64, 0) << 27));
        const llo = hi - lhi + lo;
        break :blk .{ yhi * lhi, ylo * lhi + y * llo }; // |elo| < |ehi| * 2^-25
    };
    return expInline(ehi, elo, sign_bias);
}

// ---------------------------------------------------------------------------
// exp and exp2 — ARM optimized-routines math/exp.c and math/exp2.c, on the
// same 128-entry `exp_tab` `pow` already carries. exp is literally pow's
// `expInline` with the inf/nan cases the caller there had already peeled off,
// so the whole of exp costs five lines on top of what was here for pow.
//
// Published worst case 0.509 ulp (0.511 without fma) for exp, 0.507 (0.511)
// for exp2.
// ---------------------------------------------------------------------------

/// e^x. Worst case measured against libm at the bottom of this file.
fn softExp(x: f64) f64 {
    // expInline saturates the finite overflow/underflow ends itself and is
    // documented as not handling inf/nan, because pow's own special-value
    // block already had. Do that here instead.
    const abstop = top12(x) & 0x7ff;
    if (abstop >= comptime top12(1024.0)) {
        if (@as(u64, @bitCast(x)) == comptime @as(u64, @bitCast(-std.math.inf(f64)))) return 0;
        if (abstop >= comptime top12(std.math.inf(f64))) return 1.0 + x; // nan, +inf
    }
    return expInline(x, 0, 0);
}

/// e^x - 1 — musl `expm1.c`, by way of `std.math.expm1`, MINUS the one line
/// that does not compile for AMDGCN. See `expm1` above for why that line is
/// there and why dropping it changes no return value.
///
/// Not built on `softExp`: the whole point of expm1 is that the `-1` happens
/// INSIDE the reduced-argument polynomial, where `exp(x) - 1` would cancel away
/// most of the significand for small x.
fn softExpm1(x_: f64) f64 {
    if (std.math.isNan(x_)) return std.math.nan(f64);

    const o_threshold: f64 = 7.09782712893383973096e+02;
    const ln2_hi: f64 = 6.93147180369123816490e-01;
    const ln2_lo: f64 = 1.90821492927058770002e-10;
    const invln2: f64 = 1.44269504088896338700e+00;
    const Q1: f64 = -3.33333333333331316428e-02;
    const Q2: f64 = 1.58730158725481460165e-03;
    const Q3: f64 = -7.93650757867487942473e-05;
    const Q4: f64 = 4.00821782732936239552e-06;
    const Q5: f64 = -2.01099218183624371326e-07;

    var x = x_;
    const ux: u64 = @bitCast(x);
    const hx: u32 = @as(u32, @intCast(ux >> 32)) & 0x7FFFFFFF;
    const sign = ux >> 63;

    if (std.math.isNegativeInf(x)) return -1.0;

    // |x| >= 56 * ln2
    if (hx >= 0x4043687A) {
        if (hx > 0x7FF00000) return x; // nan
        if (sign != 0) return -1; // expm1(-big) = -1
        if (x > o_threshold) return std.math.inf(f64);
    }

    var hi: f64 = undefined;
    var lo: f64 = undefined;
    var c: f64 = undefined;
    var k: i32 = undefined;

    if (hx > 0x3FD62E42) { // |x| > 0.5 * ln2
        if (hx < 0x3FF0A2B2) { // |x| < 1.5 * ln2
            if (sign == 0) {
                hi = x - ln2_hi;
                lo = ln2_lo;
                k = 1;
            } else {
                hi = x + ln2_hi;
                lo = -ln2_lo;
                k = -1;
            }
        } else {
            var kf = invln2 * x;
            if (sign != 0) kf -= 0.5 else kf += 0.5;
            k = @intFromFloat(kf);
            const t = @as(f64, @floatFromInt(k));
            hi = x - t * ln2_hi;
            lo = t * ln2_lo;
        }
        x = hi - lo;
        c = (hi - x) - lo;
    } else if (hx < 0x3C900000) {
        // |x| < 2^-54, where expm1(x) == x. std raises the underflow flag for
        // a subnormal here; that is the line this port drops.
        return x;
    } else {
        k = 0;
    }

    const hfx = 0.5 * x;
    const hxs = x * hfx;
    const r1 = 1.0 + hxs * (Q1 + hxs * (Q2 + hxs * (Q3 + hxs * (Q4 + hxs * Q5))));
    const t = 3.0 - r1 * hfx;
    var e = hxs * ((r1 - t) / (6.0 - x * t));

    if (k == 0) return x - (x * e - hxs); // c is 0

    e = x * (e - c) - c;
    e -= hxs;

    // exp(x) ~ 2^k (x_reduced - e + 1)
    if (k == -1) return 0.5 * (x - e) - 0.5;
    if (k == 1) {
        if (x < -0.25) return -2.0 * (e - (x + 0.5));
        return 1.0 + 2.0 * (x - e);
    }

    const twopk: f64 = @bitCast(@as(u64, @intCast(0x3FF +% k)) << 52);

    if (k < 0 or k > 56) {
        var y = x - e + 1.0;
        if (k == 1024) y = y * 2.0 * 0x1.0p1023 else y = y * twopk;
        return y - 1.0;
    }

    const uf: f64 = @bitCast(@as(u64, @intCast(0x3FF -% k)) << 52);
    if (k < 20) return (x - e + (1 - uf)) * twopk;
    return (x - (e + uf) + 1) * twopk;
}

/// 2^x. Not `softExp(x * ln2)`: reducing on k/N directly keeps integer x
/// exact and leaves r in [-1/256, 1/256] with no rounding at all, where the
/// scaled-argument route would spend a whole rounding on `x * ln2` first.
fn softExp2(x: f64) f64 {
    const tiny_top = comptime top12(0x1p-54);
    const ix: u64 = @bitCast(x);
    var abstop: u32 = top12(x) & 0x7ff;
    if (abstop -% tiny_top >= comptime top12(512.0) - top12(0x1p-54)) {
        // |x| < 2^-54: 1 + x, and no spurious underflow. 0 is a common input.
        if (abstop -% tiny_top >= 0x80000000) return 1.0 + x;
        if (abstop >= comptime top12(1024.0)) {
            if (ix == comptime @as(u64, @bitCast(-std.math.inf(f64)))) return 0;
            if (abstop >= comptime top12(std.math.inf(f64))) return 1.0 + x; // nan, +inf
            if (ix >> 63 == 0) return std.math.inf(f64);
            // x <= -1075 underflows to zero; between -1024 and -1075 the
            // result is subnormal and falls through to the slow path.
            if (ix >= comptime @as(u64, @bitCast(@as(f64, -1075.0)))) return 0;
        }
        // Only above 928 can `sbits` overflow its exponent field.
        if (2 *% ix > comptime 2 *% @as(u64, @bitCast(@as(f64, 928.0)))) abstop = 0;
    }

    // 2^x = 2^(k/128) * 2^r with integer k and |r| <= 1/256, both exact.
    const shifted = x + md.exp2_shift;
    const ki: u64 = @bitCast(shifted);
    const r = x - (shifted - md.exp2_shift);

    const idx: usize = @intCast(2 * (ki & 127));
    const tail: f64 = @bitCast(md.exp_tab[idx]);
    const sbits = md.exp_tab[idx + 1] +% (ki << (52 - 7));

    const c = md.exp2_poly;
    const r2 = r * r;
    const tmp = tail + r * c[0] + r2 * (c[1] + r * c[2]) + r2 * r2 * (c[3] + r * c[4]);
    if (abstop == 0) return expSpecial(1, tmp, sbits, ki);
    const scale: f64 = @bitCast(sbits);
    // tmp is 0 or |tmp| > 2^-65 and scale > 2^-928: no spurious underflow.
    return scale + scale * tmp;
}

// ---------------------------------------------------------------------------
// log and log2 — ARM optimized-routines math/log.c and math/log2.c.
//
// Same shape as pow's `logInline`, minus the double-double tail pow needs and
// plus a separate near-1 polynomial: on |log x| < 2^-4 the table route's
// log(c) + poly(r) cancels, so those inputs get a 12th-order log1p series in
// (x - 1) instead. That branch is why this is not `logInline` with a cast —
// and it is the branch a SPICE junction lives in.
//
// Published worst case 0.519 ulp (0.520 without fma) for log, 0.547 (0.550)
// for log2.
// ---------------------------------------------------------------------------

/// Where the table's z lands: [OFF, 2*OFF) is cut into 128 (log) or 64 (log2)
/// subintervals. Note this is NOT pow's OFF — pow centres its split
/// differently because it needs log(c) as a double-double.
const log_off: u64 = 0x3fe6000000000000;

/// Shared special-value prologue for the two f64 logs. Returns the value to
/// return, or null with `ix` normalized out of the subnormals.
inline fn logSpecial(x: f64, ix: *u64) ?f64 {
    const top: u32 = @truncate(ix.* >> 48);
    if (top -% 0x0010 < 0x7ff0 - 0x0010) return null; // the common case
    if (ix.* *% 2 == 0) return -std.math.inf(f64); // log(+-0) = -inf
    if (ix.* == comptime @as(u64, @bitCast(std.math.inf(f64)))) return x;
    if (top & 0x8000 != 0 or top & 0x7ff0 == 0x7ff0) return std.math.nan(f64); // x<0, nan
    ix.* = @bitCast(x * 0x1p52); // subnormal: scale up, then fix k
    ix.* -%= @as(u64, 52) << 52;
    return null;
}

/// log(x) as an unnormalized `hi + lo`, good to ~2^-68 relative — or the
/// finished answer, for the special values, where there is no pair to give.
///
/// Split out because `log10` needs the two halves: scaling a once-rounded
/// log by 1/ln10 costs a second full rounding and lands at 1.6 ulp, which is
/// worse than glibc's own log10 rather than better.
inline fn logCore(x: f64, hi: *f64, lo: *f64) ?f64 {
    var ix: u64 = @bitCast(x);

    const near1_lo = comptime @as(u64, @bitCast(@as(f64, 1.0 - 0x1p-4)));
    const near1_hi = comptime @as(u64, @bitCast(@as(f64, 1.0 + 0x1.09p-4)));
    if (ix -% near1_lo < near1_hi - near1_lo) {
        // |log x| < 2^-4, where log(c) + poly(r) would cancel: a 12th-order
        // log1p series in x - 1 instead. This is the branch a SPICE junction
        // lives in.
        if (ix == one_bits) return 0; // +0, not -0, under downward rounding
        const r = x - 1.0;
        const r2 = r * r;
        const r3 = r * r2;
        const b = md.log_poly1;
        var y = r3 * (b[1] + r * b[2] + r2 * b[3] +
            r3 * (b[4] + r * b[5] + r2 * b[6] +
                r3 * (b[7] + r * b[8] + r2 * b[9] + r3 * b[10])));
        // Dekker split of r so that rhi*rhi is exact; `r + w0 - w0` is NOT
        // removable under IEEE rules, and Zig never reassociates float ops.
        const w0 = r * 0x1p27;
        const rhi = r + w0 - w0;
        const rlo = r - rhi;
        const w = rhi * rhi * b[0]; // b[0] == -0.5
        hi.* = r + w;
        y += r - hi.* + w;
        y += b[0] * rlo * (rhi + r);
        lo.* = y;
        return null;
    }
    if (logSpecial(x, &ix)) |v| return v;

    // x = 2^k z with z in [OFF, 2*OFF) exactly, z near c = 1/invc.
    const tmp = ix -% log_off;
    const i: usize = @intCast((tmp >> (52 - 7)) & 127);
    const k = @as(i64, @bitCast(tmp)) >> 52; // arithmetic
    const z: f64 = @bitCast(ix -% (tmp & (@as(u64, 0xfff) << 52)));
    const e = md.log_tab[i];

    // r = z/c - 1, |r| < 1/256. Without an FMA the reference subtracts c as a
    // double-double first, which is why `log_tab2` exists.
    const r = if (fast_fma)
        @mulAdd(f64, z, e.invc, -1.0)
    else
        (z - md.log_tab2[i].chi - md.log_tab2[i].clo) * e.invc;

    // hi + lo = r + log(c) + k*ln2, exactly — that is what the table's
    // rounding of logc buys.
    const kd: f64 = @floatFromInt(k);
    const w = kd * md.log_ln2hi + e.logc;
    hi.* = w + r;
    const t = w - hi.* + r + kd * md.log_ln2lo;

    const a = md.log_poly;
    const r2 = r * r;
    lo.* = t + r2 * a[0] + r * r2 * (a[1] + r * a[2] + r2 * (a[3] + r * a[4]));
    return null;
}

fn softLog(x: f64) f64 {
    var hi: f64 = undefined;
    var lo: f64 = undefined;
    if (logCore(x, &hi, &lo)) |v| return v;
    return lo + hi;
}

/// log10(x) = (hi + lo)/ln10, with the leading product kept EXACT.
///
/// `l10hi` carries 25 significant bits and a 27-bit Dekker split of `y` leaves
/// 26, so `y1 * l10hi` fits in 51 bits: no rounding, no FMA, and no `fast_fma`
/// branch. Everything after it is 2^-27 of the answer, so the only rounding
/// that reaches the result is the final add.
///
/// The Knuth two-sum first is not decoration. `logCore`'s pair is NOT
/// normalized — in the table branch `hi = k*ln2 + log(c) + r` cancels for x
/// near 1, leaving |lo| comparable to |hi| — and scaling an unnormalized pair
/// puts a full rounding back where this was supposed to remove one (measured:
/// 2 ulp against glibc without it, 1 with).
///
/// The special values need no scaling: log and log10 agree on all of them
/// (+-0 -> -inf, 1 -> +0, inf -> inf, negative -> nan).
fn softLog10(x: f64) f64 {
    var hi: f64 = undefined;
    var lo: f64 = undefined;
    if (logCore(x, &hi, &lo)) |v| return v;
    const y = hi + lo;
    const b = y - hi;
    const yl = (hi - (y - b)) + (lo - b); // y + yl = hi + lo, exactly
    const y1: f64 = @bitCast(@as(u64, @bitCast(y)) & (~@as(u64, 0) << 27));
    const y2 = y - y1;
    return y1 * l10hi + (y2 * l10hi + (y * l10lo + yl * invln10));
}

fn softLog2(x: f64) f64 {
    var ix: u64 = @bitCast(x);

    const near1_lo = comptime @as(u64, @bitCast(@as(f64, 1.0 - 0x1.5b51p-5)));
    const near1_hi = comptime @as(u64, @bitCast(@as(f64, 1.0 + 0x1.6ab2p-5)));
    if (ix -% near1_lo < near1_hi - near1_lo) {
        if (ix == one_bits) return 0;
        const r = x - 1.0;
        // r/ln2 in double-double.
        const h, const l = if (fast_fma) blk: {
            const h = r * md.log2_invln2hi;
            break :blk .{ h, r * md.log2_invln2lo + @mulAdd(f64, r, md.log2_invln2hi, -h) };
        } else blk: {
            const rhi: f64 = @bitCast(@as(u64, @bitCast(r)) & (~@as(u64, 0) << 32));
            const rlo = r - rhi;
            break :blk .{ rhi * md.log2_invln2hi, rlo * md.log2_invln2hi + r * md.log2_invln2lo };
        };
        const r2 = r * r;
        const r4 = r2 * r2;
        const b = md.log2_poly1;
        const p = r2 * (b[0] + r * b[1]);
        const y = h + p;
        var lo = l + (h - y + p);
        lo += r4 * (b[2] + r * b[3] + r2 * (b[4] + r * b[5]) +
            r4 * (b[6] + r * b[7] + r2 * (b[8] + r * b[9])));
        return y + lo;
    }
    if (logSpecial(x, &ix)) |v| return v;

    // 64 subintervals here, not 128: log2's k folds in as an exact integer
    // rather than through a k*ln2 double-double, so half the table reaches
    // the same accuracy.
    const tmp = ix -% log_off;
    const i: usize = @intCast((tmp >> (52 - 6)) & 63);
    const k = @as(i64, @bitCast(tmp)) >> 52;
    const z: f64 = @bitCast(ix -% (tmp & (@as(u64, 0xfff) << 52)));
    const e = md.log2_tab[i];

    const r, const t1, const t2 = if (fast_fma) blk: {
        const r = @mulAdd(f64, z, e.invc, -1.0);
        const t1 = r * md.log2_invln2hi;
        break :blk .{ r, t1, r * md.log2_invln2lo + @mulAdd(f64, r, md.log2_invln2hi, -t1) };
    } else blk: {
        const r = (z - md.log2_tab2[i].chi - md.log2_tab2[i].clo) * e.invc;
        const rhi: f64 = @bitCast(@as(u64, @bitCast(r)) & (~@as(u64, 0) << 32));
        const rlo = r - rhi;
        break :blk .{ r, rhi * md.log2_invln2hi, rlo * md.log2_invln2hi + r * md.log2_invln2lo };
    };

    // hi + lo = r/ln2 + log2(c) + k, exactly: `k + logc` is why logc is
    // rounded so that 1024 + logc has no rounding error.
    const t3 = @as(f64, @floatFromInt(k)) + e.logc;
    const hi = t3 + t1;
    const lo = t3 - hi + t1 + t2;

    const a = md.log2_poly;
    const r2 = r * r;
    const r4 = r2 * r2;
    const p = a[0] + r * a[1] + r2 * (a[2] + r * a[3]) + r4 * (a[4] + r * a[5]);
    return lo + r2 * p + hi;
}

// ---------------------------------------------------------------------------
// f32 exp / log / log2 / log10 — ARM optimized-routines math/expf.c,
// math/logf.c, math/log2f.c, math/log10f.c.
//
// These accumulate in binary64 and round once at the end, which is what makes
// them <=0.82 ulp where the f32 hardware approximations are bounded only
// ABSOLUTELY (~2^-21 for `lg2.approx.f`/`v_log_f32`) and therefore have no
// relative accuracy at all near log's zero at x = 1.
//
// ponytail: that binary64 middle is 1/64 rate on consumer NVIDIA. It is ~10
// f64 ops here rather than the ~25 a full f64 log would cost, and Verilog-A
// `real` is f64 so nothing in the current consumers takes this path — but a
// throughput-bound f32 kernel that genuinely does not care about x near 1
// wants `@log2` back, for f32 only.
// ---------------------------------------------------------------------------

inline fn top12f(x: f32) u32 {
    return @as(u32, @bitCast(x)) >> 20;
}

fn softExpf(x: f32) f32 {
    const ux: u32 = @bitCast(x);
    const abstop = top12f(x) & 0x7ff;
    if (abstop >= comptime top12f(88.0) & 0x7ff) {
        if (ux == comptime @as(u32, @bitCast(-std.math.inf(f32)))) return 0;
        if (abstop >= comptime top12f(std.math.inf(f32)) & 0x7ff) return x + x;
        if (x > 0x1.62e42ep6) return std.math.inf(f32); // > log(2^128)
        if (x < -0x1.9fe368p6) return 0; // < log(2^-150)
    }

    // x*32/ln2 = k + r with integer k and |r| <= 1/2; the 32-entry table is
    // every 4th entry of the f64 one.
    const z = md.expf_invln2N * @as(f64, x);
    const shifted = z + md.expf_shift;
    const ki: u64 = @bitCast(shifted);
    const r = z - (shifted - md.expf_shift);

    const s: f64 = @bitCast(md.expf_tab[@intCast(ki & 31)] +% (ki << (52 - 5)));
    const c = md.expf_poly;
    const r2 = r * r;
    return @floatCast(((c[0] * r + c[1]) * r2 + (c[2] * r + 1)) * s);
}

/// Everything ARM's logf.c and log2f.c do before their polynomials, which is
/// the same thing twice: the same near-1 exit, the same special values, the
/// same subnormal scale-up, the same 16-way split. Only the table and the
/// weight on `k` differ.
///
/// `kscale` is ln2 for logf, whose k arrives scaled, and 1 for log2f, where k
/// folds in exactly — multiplying by 1.0 is exact in IEEE, so sharing the line
/// costs log2f nothing.
///
/// Returns the finished answer for the inputs that have no `(r, y0)` pair to
/// hand back, null otherwise: the same shape as `logSpecial`, which is the f64
/// half of this.
inline fn logfSplit(
    x: f32,
    tab: *const [16]md.LogTab,
    comptime kscale: f64,
    r: *f64,
    y0: *f64,
) ?f32 {
    var ix: u32 = @bitCast(x);
    if (ix == 0x3f800000) return 0; // +0, not -0, under downward rounding
    if (ix -% 0x00800000 >= 0x7f800000 - 0x00800000) {
        if (ix *% 2 == 0) return -std.math.inf(f32);
        if (ix == 0x7f800000) return x;
        if (ix & 0x80000000 != 0 or ix *% 2 >= 0xff000000) return std.math.nan(f32);
        ix = @bitCast(x * 0x1p23); // subnormal: scale up, then fix k
        ix -%= 23 << 23;
    }

    const tmp = ix -% 0x3f330000;
    const i: usize = @intCast((tmp >> (23 - 4)) & 15);
    const k = @as(i32, @bitCast(tmp)) >> 23; // arithmetic
    const z: f64 = @as(f32, @bitCast(ix -% (tmp & 0xff800000)));
    const e = tab[i];

    r.* = z * e.invc - 1;
    y0.* = e.logc + @as(f64, @floatFromInt(k)) * kscale;
    return null;
}

/// The two scales `softLogf` is ever called at. `apply` takes a one-argument
/// body, and naming them beats a bare `1.0` at the call site.
fn softLnf(x: f32) f32 {
    return softLogf(x, 1.0);
}
fn softLog10f(x: f32) f32 {
    return softLogf(x, invln10);
}

/// `scale` is 1 for log and 1/ln10 for log10 — ARM's log10f is byte for byte
/// its logf with the constant folded into the binary64 accumulator, before
/// the single rounding, so it is genuinely a log10 and not a scaled log.
fn softLogf(x: f32, comptime scale: f64) f32 {
    var r: f64 = undefined;
    var y0: f64 = undefined;
    if (logfSplit(x, &md.logf_tab, md.logf_ln2, &r, &y0)) |v| return v;

    const a = md.logf_poly;
    const r2 = r * r;
    const y = (a[0] * r2 + (a[1] * r + a[2])) * r2 + (y0 + r);
    return @floatCast(if (scale == 1.0) y else y * scale);
}

/// Not `softLogf(x, 1/ln2)`: log2f has its own table and a fourth coefficient,
/// and it is the `a[3]*r + y0` tail — k added as an exact integer, never
/// through a scaled ln2 — that keeps powers of two exact.
fn softLog2f(x: f32) f32 {
    var r: f64 = undefined;
    var y0: f64 = undefined;
    if (logfSplit(x, &md.log2f_tab, 1.0, &r, &y0)) |v| return v;

    const a = md.log2f_poly;
    const r2 = r * r;
    return @floatCast((a[0] * r2 + (a[1] * r + a[2])) * r2 + (a[3] * r + y0));
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
// Tests. The coarse ones here are a smoke screen against the builtins, which
// are only good to a couple of ulp themselves; the differentials against libm
// at the bottom are what actually certifies the tables and they need `-lc`.
// The other half of the check is that nvptx/amdgcn objects build at all.
// ---------------------------------------------------------------------------

fn ulpErr(comptime T: type, got: T, want: T) T {
    if (got == want) return 0;
    const scale = @max(@abs(want), std.math.floatMin(T));
    return @abs(got - want) / (scale * std.math.floatEps(T));
}

test "ported cores agree with the builtins" {
    const ulp = struct {
        fn e(got: f64, want: f64) f64 {
            return ulpErr(f64, got, want);
        }
    }.e;

    // exp: a wide dynamic range plus the edges of the argument reduction.
    for ([_]f64{ 0, 1e-30, -1e-30, 1e-9, 0.1, -0.34, 0.35, 1, -1, 40, 80, -80, 700, -700, -745, 710 }) |x| {
        try std.testing.expect(ulp(softExp(x), @exp(x)) <= 2);
        try std.testing.expect(ulp(softExp2(x), @exp2(x)) <= 2);
        try std.testing.expect(ulpErr(f32, softExpf(@floatCast(x)), @exp(@as(f32, @floatCast(x)))) <= 2);
    }
    // exp2 must be exact on integers, all the way into the subnormals.
    for ([_]f64{ -1074, -1050, -60, -1, 0, 1, 10, 63, 1023 }) |x| {
        try std.testing.expectEqual(@exp2(x), softExp2(x));
    }
    try std.testing.expect(std.math.isNan(softExp2(std.math.nan(f64))));
    try std.testing.expect(std.math.isNan(softExp(std.math.nan(f64))));
    try std.testing.expectEqual(std.math.inf(f64), softExp2(5000));
    try std.testing.expectEqual(@as(f64, 0), softExp2(-5000));
    try std.testing.expectEqual(std.math.inf(f64), softExp(std.math.inf(f64)));
    try std.testing.expectEqual(@as(f64, 0), softExp(-std.math.inf(f64)));

    // log: exactly 1 and both sides of the near-1 window. Subnormals are NOT
    // in this loop: the builtins are compiler_rt, whose log2 is musl's
    // log * log2e and lands ~23 ulp out down there — @log2(0x1p-1060) comes
    // back as -1059.999999999995 — which is one of the things this replaces.
    for ([_]f64{ 1e-30, 0.5, 0.7071, 0.94, 1.0, 1.06, 1.5, 2.0, 1e6, 1e300 }) |x| {
        try std.testing.expect(ulp(softLog(x), @log(x)) <= 2);
        try std.testing.expect(ulp(softLog2(x), @log2(x)) <= 2);
        try std.testing.expect(ulpErr(f32, softLogf(@floatCast(x), 1.0), @log(@as(f32, @floatCast(x)))) <= 2);
        try std.testing.expect(ulpErr(f32, softLog2f(@floatCast(x)), @log2(@as(f32, @floatCast(x)))) <= 2);
    }
    // Every power of two exact, against the integer rather than a builtin,
    // subnormals included.
    for ([_]i32{ -1074, -1060, -1023, -1022, -2, 0, 1, 10, 1023 }) |e| {
        try std.testing.expectEqual(@as(f64, @floatFromInt(e)), softLog2(std.math.ldexp(@as(f64, 1), e)));
    }
    for ([_]i32{ -149, -140, -127, -126, -2, 0, 1, 10, 127 }) |e| {
        try std.testing.expectEqual(@as(f32, @floatFromInt(e)), softLog2f(std.math.ldexp(@as(f32, 1), e)));
    }
    // log10 exact on every exactly-representable power of ten. This is the
    // property the old `softLog * log10e` could not hold, and the differential
    // below cannot check it because glibc's own log10 is the loose one there.
    {
        var p10: f64 = 1;
        for (0..23) |e| {
            try std.testing.expectEqual(@as(f64, @floatFromInt(e)), softLog10(p10));
            try std.testing.expectEqual(-@as(f64, @floatFromInt(e)), softLog10(1 / p10));
            p10 *= 10;
        }
    }
    for ([_]f64{ 0, -0.0 }) |x| {
        try std.testing.expect(std.math.isNegativeInf(softLog(x)));
        try std.testing.expect(std.math.isNegativeInf(softLog2(x)));
        try std.testing.expect(std.math.isNegativeInf(softLogf(@floatCast(x), 1.0)));
        try std.testing.expect(std.math.isNegativeInf(softLog2f(@floatCast(x))));
    }
    for ([_]f64{ -1, -0x1p-1074, -std.math.inf(f64), std.math.nan(f64) }) |x| {
        try std.testing.expect(std.math.isNan(softLog(x)));
        try std.testing.expect(std.math.isNan(softLog2(x)));
    }
    // -0x1p-1074 casts to -0.0 in f32, which is a -inf not a nan, so the f32
    // list is its own.
    for ([_]f32{ -1, -0x1p-149, -std.math.inf(f32), std.math.nan(f32) }) |x| {
        try std.testing.expect(std.math.isNan(softLogf(x, 1.0)));
        try std.testing.expect(std.math.isNan(softLog2f(x)));
    }
    try std.testing.expectEqual(std.math.inf(f64), softLog(std.math.inf(f64)));
    try std.testing.expectEqual(std.math.inf(f64), softLog2(std.math.inf(f64)));
    // log(1) is +0, not -0: the sign matters to anything that then divides.
    try std.testing.expectEqual(@as(u64, 0), @as(u64, @bitCast(softLog(1.0))));
    try std.testing.expectEqual(@as(u64, 0), @as(u64, @bitCast(softLog2(1.0))));

    // sin/cos: below pi/4, across the quadrant switches, and into the
    // pre-reduced range above 2^20*(pi/2).
    for ([_]f64{ 0, 1e-9, 0.3, 0.786, 1.5708, 3.14159, -2.5, 100.0, 1e5, 1.5e6 }) |x| {
        try std.testing.expect(ulp(softSin(x), @sin(x)) <= 4);
        try std.testing.expect(ulp(softCos(x), @cos(x)) <= 4);
    }
    // Past the Cody-Waite ceiling accuracy is only asserted loosely.
    try std.testing.expectApproxEqAbs(@sin(1e9), softSin(1e9), 1e-6);
}

test "expm1 is bit-identical to std, and atan is within an ulp" {
    // The whole claim of the port: it drops a line that touched only the FP
    // flag register, so every VALUE must match std exactly. Bit equality, over
    // the branch boundaries the algorithm actually switches on -- 2^-54, the
    // 0.5*ln2 and 1.5*ln2 reduction splits, 56*ln2, and the overflow threshold.
    const edges = [_]f64{
        0,          -0.0,         0x1p-60,     -0x1p-60,   0x1p-54,
        -0x1p-54,   0x1p-53,      1e-300,      -1e-300,    0.3465,
        -0.3465,    0.3466,       -0.3466,     1.0397,     -1.0397,
        1.0398,     -1.0398,      1,           -1,         0.25,
        -0.25,      -0.2501,      2,           -2,         38.8,
        -38.8,      38.9,         709.78,      709.79,     710,
        -745,       1e300,        -1e300,
    };
    for (edges) |x| {
        try std.testing.expectEqual(
            @as(u64, @bitCast(std.math.expm1(x))),
            @as(u64, @bitCast(expm1(x))),
        );
    }
    // A sweep, not just the edges: 200k points across the whole reduction range.
    var i: i32 = -100_000;
    while (i < 100_000) : (i += 1) {
        const x = @as(f64, @floatFromInt(i)) * 1e-4;
        try std.testing.expectEqual(
            @as(u64, @bitCast(std.math.expm1(x))),
            @as(u64, @bitCast(expm1(x))),
        );
    }
    try std.testing.expectEqual(@as(f64, -1), expm1(-std.math.inf(f64)));
    try std.testing.expectEqual(std.math.inf(f64), expm1(std.math.inf(f64)));
    try std.testing.expect(std.math.isNan(expm1(std.math.nan(f64))));
    // f32 rides the f64 body; one rounding of a correct f64 is correct.
    for ([_]f32{ 0, 1e-10, -1e-10, 0.5, -0.5, 3, -3, 88, -88 }) |x| {
        try std.testing.expect(ulpErr(f32, expm1(x), @floatCast(std.math.expm1(@as(f64, x)))) <= 1);
    }

    // atan on the host is std's scalar body, so it is std exactly.
    for ([_]f64{ 0, 1e-20, 0.3, -0.4375, 1, -1, 1.5, 1e6, -1e6 }) |x| {
        try std.testing.expectEqual(std.math.atan(x), atan(x));
    }
}

test "every entry point takes a vector, lane for lane" {
    // The CPU backend instantiates a generic map body at `@Vector` width, so
    // these are the calls a vectorized kernel actually makes -- and every one
    // of them used to be a compile error.
    //
    // Bit-equality against the scalar path, not a tolerance. A vector call must
    // be the SAME arithmetic whether it arrives through an elementwise builtin
    // (`@sqrt`, `@sin`, f32 `@exp2`) or through `lanewise`; anything looser
    // would let a lane quietly take a different route.
    // `apply` instantiates a whole body per lane, and `powBody` is a big one;
    // fifteen entry points at two widths walks past the default in Debug. A
    // caller that lane-loops `pow` over a wide vector needs the same raise.
    @setEvalBranchQuota(20_000);
    inline for (.{ f32, f64 }) |T| {
        const V = @Vector(4, T);
        const xs: V = .{ 0.25, 1.0, 3.5, 40.0 };
        const ys: V = .{ 2.0, -1.5, 0.33, 3.0 };

        inline for (.{ exp, exp2, expm1, log, log2, log10, atan, sin, cos, tan, tanh, sinh, cosh, sqrt, rsqrt }) |f| {
            const got: V = f(xs);
            inline for (0..4) |i| try std.testing.expectEqual(f(xs[i]), got[i]);
        }
        const got: V = pow(xs, ys);
        inline for (0..4) |i| try std.testing.expectEqual(pow(xs[i], ys[i]), got[i]);
    }
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

        // 8, not the 64 this used to allow: `pow` is now the 0.52-ulp
        // algorithm and `std.math.pow` is the 3-ulp one, so the gap is the
        // reference's own error. The real bound is the libm differential
        // below.
        for ([_]T{ 1e-6, 0.5, 1.5, 2, 10, 1e6 }) |x| {
            for ([_]T{ 0, 1, 2, 2.5, -1.5, 0.33, -3 }) |y| {
                try std.testing.expect(ulp(pow(x, y), std.math.pow(T, x, y)) <= 8);
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
        // The dispatchers pick the right body for the type. exp/log/log2 are
        // now ports on the host too, so they are checked against the builtin
        // to a couple of ulp rather than for equality; sin/cos still forward.
        for ([_]T{ 0.5, 1, 2, 7.25 }) |x| {
            try std.testing.expect(ulp(exp(x), @exp(x)) <= 2);
            try std.testing.expectEqual(@exp2(x), exp2(x));
            try std.testing.expect(ulp(log(x), @log(x)) <= 2);
            try std.testing.expect(ulp(log2(x), @log2(x)) <= 2);
            try std.testing.expect(ulp(log10(x), @log10(x)) <= 2);
            try std.testing.expectEqual(@sin(x), sin(x));
            try std.testing.expectEqual(@cos(x), cos(x));
        }
    }
}

// ---------------------------------------------------------------------------
// The differentials against libm. These are the only tests that can actually
// see 1 ulp, so they are what certifies the tables: a single mistyped hex
// digit anywhere in `math_data.zig` moves the error by orders of magnitude.
// Host-only — device builds have no oracle — and they need libc, which
// build.zig turns on for the root project's test module only.
// ---------------------------------------------------------------------------

const libm = struct {
    // Safe as a plain extern: compiler_rt has no `pow`, so this one really is
    // the system's.
    extern "c" fn pow(f64, f64) f64;

    extern "c" fn dlopen(?[*:0]const u8, c_int) ?*anyopaque;
    extern "c" fn dlsym(?*anyopaque, [*:0]const u8) ?*anyopaque;

    /// The system libm's own `name`, fetched the long way round.
    ///
    /// `extern "c" fn log` does NOT reach glibc from a Zig binary. Zig always
    /// links compiler_rt, compiler_rt exports `exp exp2 log log2 log10` (and
    /// the f32 forms) as musl ports, and a static definition beats a shared
    /// one — so the naive spelling silently benchmarks and validates this
    /// port against the very algorithm it replaces. Going through the shared
    /// object by hand is the only way to get the reference glibc actually
    /// runs, which is the whole point of the exercise: ngspice links that one.
    ///
    /// Returns null off glibc; the callers skip rather than assert.
    fn real(comptime F: type, name: [*:0]const u8) ?*const F {
        const h = dlopen("libm.so.6", 0x2) orelse return null; // RTLD_NOW
        return @ptrCast(@alignCast(dlsym(h, name) orelse return null));
    }
};

const Fn64 = fn (f64) callconv(.c) f64;
const Fn32 = fn (f32) callconv(.c) f32;

/// A running worst-case ulp gap between one of our bodies and libm's, for
/// either width. Sampling lives here rather than in the tests because every
/// function wants the same four shapes and only the ranges differ.
fn Diff(comptime T: type) type {
    const U = std.meta.Int(.unsigned, @bitSizeOf(T));
    const I = std.meta.Int(.signed, @bitSizeOf(T));
    return struct {
        const Self = @This();

        name: []const u8,
        mine: *const fn (T) T,
        theirs: *const fn (T) callconv(.c) T,
        worst: f64 = 0,
        arg: T = 0,
        n: usize = 0,

        /// Floats ordered as signed integers, so |key(a) - key(b)| is the
        /// number of representable values between them. Exact across binade
        /// boundaries and into the subnormals, where a |got-want|/ulp(want)
        /// formula needs special cases.
        const sign_bit: U = @as(U, 1) << (@bitSizeOf(T) - 1);

        fn key(x: T) I {
            const u: U = @bitCast(x);
            return @bitCast(if (u & sign_bit != 0) sign_bit -% u else u);
        }
        fn unkey(k: I) T {
            const u: U = @bitCast(k);
            return @bitCast(if (k < 0) sign_bit -% u else u);
        }
        fn apart(got: T, want: T) f64 {
            const g = std.math.isNan(got);
            const w = std.math.isNan(want);
            if (g or w) return if (g and w) 0 else std.math.inf(f64);
            const d = @abs(@as(i128, key(got)) - @as(i128, key(want)));
            return @floatFromInt(@min(d, 1 << 62));
        }

        fn at(d: *Self, x: T) void {
            d.n += 1;
            const e = apart(d.mine(x), d.theirs(x));
            if (e > d.worst) {
                d.worst = e;
                d.arg = x;
            }
        }

        /// `count` values with a uniform random mantissa and a uniform random
        /// BIASED exponent in [elo, ehi], so elo = 0 walks into the
        /// subnormals -- which a log-uniform sweep in the value never reaches
        /// in any useful density.
        fn binades(d: *Self, rnd: std.Random, count: usize, elo: u32, ehi: u32, neg: bool) void {
            const mant = @typeInfo(T).float.bits - std.math.floatExponentBits(T) - 1;
            for (0..count) |_| {
                const e: U = rnd.intRangeAtMost(u32, elo, ehi);
                const sign: U = if (neg and rnd.boolean()) sign_bit else 0;
                d.at(@bitCast(sign | (e << mant) | (rnd.int(U) >> (@bitSizeOf(T) - mant))));
            }
        }

        /// `count` values uniform in [lo, hi].
        fn uniform(d: *Self, rnd: std.Random, count: usize, lo: T, hi: T) void {
            for (0..count) |_| d.at(lo + rnd.float(T) * (hi - lo));
        }

        /// Every representable value walking out from `x0`, both ways. This is
        /// how the near-1 window and the saturation thresholds get covered at
        /// the density where an off-by-one branch bound actually shows up.
        fn around(d: *Self, x0: T, count: usize) void {
            const k0 = key(x0);
            for (0..count) |i| {
                const off: I = @intCast(i);
                d.at(unkey(k0 +| off));
                d.at(unkey(k0 -| off));
            }
        }

        fn report(d: Self, bound: f64) !void {
            std.debug.print("    {s:<7} {d:>9} pts, worst {d:.3} ulp at {x}\n", .{ d.name, d.n, d.worst, d.arg });
            try std.testing.expect(d.worst <= bound);
        }

        /// libm's answer is the conformance reference: bit-exact where it
        /// returns 0, inf or nan (sign of zero included), inside `bound`
        /// elsewhere.
        fn specials(d: *Self, xs: []const T, bound: f64) !void {
            for (xs) |x| {
                const want = d.theirs(x);
                const got = d.mine(x);
                d.n += 1;
                if (std.math.isNan(want)) {
                    try std.testing.expect(std.math.isNan(got));
                } else if (want == 0 or std.math.isInf(want)) {
                    try std.testing.expectEqual(@as(U, @bitCast(want)), @as(U, @bitCast(got)));
                } else {
                    try std.testing.expect(apart(got, want) <= bound);
                }
            }
        }
    };
}

test "exp, exp2, log, log2 and log10 match libm to 1 ulp" {
    if (dev or !builtin.link_libc) return error.SkipZigTest;
    const c_exp = libm.real(Fn64, "exp") orelse return error.SkipZigTest;
    const D = Diff(f64);
    var prng = std.Random.DefaultPrng.init(0x5EED_C0FFEE);
    const rnd = prng.random();
    std.debug.print("\n  f64 vs libm (dlsym'd, NOT compiler_rt):\n", .{});

    const inf = std.math.inf(f64);
    const nan = std.math.nan(f64);
    const edges = [_]f64{
        0.0,        -0.0,    1.0,     -1.0,      2.0,                     -2.0,
        0.5,        inf,     -inf,    nan,       -nan,                    1e300,
        -1e300,     1e-300,  -1e-300, 0x1p-1022, -0x1p-1022,              0x1p-1074,
        -0x1p-1074, 709.0,   710.0,   -745.0,    -746.0,                  1024.0,
        1025.0,     -1074.0, -1075.0, -1076.0,   0x1.fffffffffffffp+1023,
    };

    // exp / exp2. 400k across the whole live range, 300k over every binade
    // from subnormal up (the |x| < 2^-54 shortcut and the 1+x path), and 300k
    // packed against the saturation thresholds, where being one representable
    // value out is the entire failure mode.
    inline for (.{
        .{ "exp", softExp, "exp", -745.2, 709.79, 709.782712893384, -745.1332191019411 },
        .{ "exp2", softExp2, "exp2", -1080.0, 1024.0, 1024.0, -1075.0 },
    }) |spec| {
        const mine: *const fn (f64) f64 = spec[1];
        var d = D{ .name = spec[0], .mine = mine, .theirs = libm.real(Fn64, spec[2]).? };
        d.uniform(rnd, 400_000, spec[3], spec[4]);
        d.binades(rnd, 300_000, 0, 1033, true);
        d.around(spec[5], 75_000);
        d.around(spec[6], 75_000);
        try d.specials(&edges, 1.0);
        try d.report(1.0);
    }

    // log / log2 / log10. 400k over every binade of the positive finite range
    // (biased exponent 0..2046, so ~1/2047 of the points are subnormal), 400k
    // inside the near-1 window each body special-cases, and 200k walking the
    // representable values immediately around 1.0 -- where the f32 hardware
    // path had no significant digits at all, and where a junction sits.
    inline for (.{
        .{ "log", softLog, "log", 1.0 },
        .{ "log2", softLog2, "log2", 1.0 },
        // 2, not 1, and the slack is GLIBC's. Its log10 is not from
        // optimized-routines — it is still `e_log10.c` — and measures 1.57
        // ulp from a 60-digit reference over the same 120k points where this
        // one measures 0.52. Tightening this bound would be asserting that we
        // reproduce the reference's error.
        .{ "log10", softLog10, "log10", 2.0 },
    }) |spec| {
        const mine: *const fn (f64) f64 = spec[1];
        var d = D{ .name = spec[0], .mine = mine, .theirs = libm.real(Fn64, spec[2]).? };
        d.binades(rnd, 400_000, 0, 2046, false);
        d.uniform(rnd, 400_000, 1.0 - 0x1.1p-4, 1.0 + 0x1.1p-4);
        d.around(1.0, 100_000);
        try d.specials(&edges, spec[3]);
        try d.report(spec[3]);
    }
    _ = c_exp;
}

test "f32 exp, log, log2 and log10 match libm to 1 ulp" {
    if (dev or !builtin.link_libc) return error.SkipZigTest;
    const c_expf = libm.real(Fn32, "expf") orelse return error.SkipZigTest;
    const D = Diff(f32);
    const wrap = struct {
        fn logf_(x: f32) f32 {
            return softLogf(x, 1.0);
        }
        fn log10f_(x: f32) f32 {
            return softLogf(x, invln10);
        }
    };

    var prng = std.Random.DefaultPrng.init(0xF32_5EED);
    const rnd = prng.random();
    std.debug.print("  f32 vs libm:\n", .{});

    const inf = std.math.inf(f32);
    const edges = [_]f32{
        0.0,          -0.0,      1.0,      -1.0, inf,  -inf,   std.math.nan(f32),
        0x1p-149,     -0x1p-149, 0x1p-126, 88.0, 89.0, -104.0, -105.0,
        3.4028235e38,
    };

    // 600k random bit patterns each: every binade, the subnormals and the
    // negative half (nan for the logs, underflow for exp) all in proportion.
    inline for (.{
        .{ "expf", softExpf, "expf", false },
        .{ "logf", wrap.logf_, "logf", true },
        .{ "log2f", softLog2f, "log2f", true },
        .{ "log10f", wrap.log10f_, "log10f", true },
    }) |spec| {
        const mine: *const fn (f32) f32 = spec[1];
        var d = D{ .name = spec[0], .mine = mine, .theirs = libm.real(Fn32, spec[2]).? };
        d.binades(rnd, 600_000, 0, 254, true);
        if (spec[3]) {
            d.uniform(rnd, 400_000, 1.0 - 0x1.1p-4, 1.0 + 0x1.1p-4);
            d.around(1.0, 100_000);
        } else {
            d.uniform(rnd, 400_000, -104.0, 89.0);
            d.around(0x1.62e42ep6, 50_000);
            d.around(-0x1.9fe368p6, 50_000);
        }
        try d.specials(&edges, 1.0);
        try d.report(1.0);
    }
    _ = c_expf;
}

test "pow matches libm to 1 ulp" {
    if (dev or !builtin.link_libc) return error.SkipZigTest;

    var worst: f64 = 0;
    var wx: f64 = 0;
    var wy: f64 = 0;
    var finite: usize = 0;
    var n: usize = 0;
    const check = struct {
        fn f(x: f64, y: f64, w: *f64, ax: *f64, ay: *f64, fin: *usize, cnt: *usize) void {
            const want = libm.pow(x, y);
            const e = Diff(f64).apart(pow(x, y), want);
            cnt.* += 1;
            if (std.math.isFinite(want) and want != 0) fin.* += 1;
            if (e > w.*) {
                w.* = e;
                ax.* = x;
                ay.* = y;
            }
        }
    }.f;

    // 1e6 points: x log-uniform over [1e-30, 1e30], y uniform over [-40, 40].
    // Most of that plane overflows or underflows; `finite` reports how many
    // actually exercised the table path.
    var prng = std.Random.DefaultPrng.init(0x5EED_C0FFEE);
    const rnd = prng.random();
    const ln10 = 2.302585092994045684;
    for (0..1_000_000) |_| {
        const x = @exp(ln10 * (rnd.float(f64) * 60.0 - 30.0));
        const y = rnd.float(f64) * 80.0 - 40.0;
        check(x, y, &worst, &wx, &wy, &finite, &n);
    }
    const random_worst = worst;
    const random_finite = finite;

    // The domain that actually shows up in SPICE: junction grading and
    // ideality exponents against a normalized voltage ratio.
    for ([_]f64{ 1.1739, 0.87225, 0.693, 0.99, 0.649, 0.351 }) |y| {
        for (0..100_000) |i| {
            const u = @as(f64, @floatFromInt(i)) / 100_000.0;
            const x = @exp(ln10 * (-3.1249 + u * (0.4771 + 3.1249)));
            check(x, y, &worst, &wx, &wy, &finite, &n);
        }
    }

    // C99 special values, cross product. Where libm returns 0, inf or nan the
    // match must be exact (sign of zero included); elsewhere the ulp bound
    // applies. glibc is the conformance reference, so no expected-value table.
    const inf = std.math.inf(f64);
    const nan = std.math.nan(f64);
    const xs = [_]f64{
        0.0,       -0.0,       1.0,       -1.0,       2.0,                     -2.0,   0.5,    -0.5,
        inf,       -inf,       nan,       -nan,       1e300,                   -1e300, 1e-300, -1e-300,
        0x1p-1022, -0x1p-1022, 0x1p-1070, -0x1p-1070, 0x1.fffffffffffffp+1023,
    };
    const ys = [_]f64{
        0.0,     -0.0,     1.0,    -1.0,    2.0,    -2.0,    3.0,  -3.0,
        0.5,     -0.5,     2.5,    -2.5,    64.0,   -64.0,   65.0, -65.0,
        inf,     -inf,     nan,    1023.0,  1024.0, -1074.0, 1e10, -1e10,
        0x1p-70, -0x1p-70, 0x1p70, -0x1p70, 1e308,
    };
    for (xs) |x| for (ys) |y| {
        const want = libm.pow(x, y);
        const got = pow(x, y);
        n += 1;
        if (std.math.isNan(want)) {
            try std.testing.expect(std.math.isNan(got));
        } else if (want == 0 or std.math.isInf(want)) {
            // Bit compare: pow(-0.0, -3) is -inf, not +inf, and pow(-0.0, 3)
            // is -0.0, not +0.0.
            try std.testing.expectEqual(@as(u64, @bitCast(want)), @as(u64, @bitCast(got)));
        } else {
            const e = Diff(f64).apart(got, want);
            if (e > worst) {
                worst = e;
                wx = x;
                wy = y;
            }
        }
    };

    std.debug.print(
        "\n  pow vs libm: {d} points, worst {d:.2} ulp at pow({e}, {e})\n" ++
            "    random 1e6 (x log-uniform 1e-30..1e30, y -40..40): {d:.2} ulp, {d} finite non-zero\n",
        .{ n, worst, wx, wy, random_worst, random_finite },
    );
    try std.testing.expect(worst <= 1.0);
}

test "tables keep the exactness the algorithms assume" {
    // pow: 1/c must have at most 8 fractional mantissa bits, or z/c - 1 is not
    // exactly representable and logInline's whole error budget is void.
    for (md.powlog_tab) |e| {
        const m: u64 = @bitCast(e.invc);
        try std.testing.expectEqual(@as(u64, 0), m & ((1 << 44) - 1));
        // logc is rounded to a multiple of 2^-43 so k*ln2hi + logc is exact.
        try std.testing.expectEqual(e.logc, @trunc(e.logc * 0x1p43) * 0x1p-43);
    }
    // ln2hi and negln2hiN have their low bits cleared for the same reason.
    // negln2hiN keeps 36 significant bits, so kd*negln2hiN is exact for every
    // |kd| < 2^17 = 131072 -- exactly the 1024*128 the exp reduction reaches.
    try std.testing.expectEqual(@as(u64, 0), @as(u64, @bitCast(md.powlog_ln2hi)) & ((1 << 11) - 1));
    try std.testing.expectEqual(@as(u64, 0), @as(u64, @bitCast(md.negln2hiN)) & ((1 << 17) - 1));
    // Entry 0 of the exp table is exactly 1.0 with a zero tail, and the f32
    // table is its every-4th-entry slice.
    try std.testing.expectEqual(@as(u64, 0), md.exp_tab[0]);
    try std.testing.expectEqual(one_bits, md.exp_tab[1]);
    for (md.expf_tab, 0..) |t, i| try std.testing.expectEqual(md.exp_tab[8 * i + 1], t);

    // log: `k*ln2hi + logc` is error-free only because logc was rounded so
    // that `0x1.8p9 + logc` is, and ln2hi's low 11 bits are clear. log2 wants
    // `k + logc` error-free, hence 0x1.8p10 there. One digit wrong in either
    // table breaks these before it breaks the ulp bound.
    try std.testing.expectEqual(@as(u64, 0), @as(u64, @bitCast(md.log_ln2hi)) & ((1 << 11) - 1));
    for (md.log_tab) |e| {
        try std.testing.expectEqual(e.logc, (0x1.8p9 + e.logc) - 0x1.8p9);
    }
    for (md.log2_tab) |e| {
        try std.testing.expectEqual(e.logc, (0x1.8p10 + e.logc) - 0x1.8p10);
    }
    // The non-fma paths need chi + clo to reproduce c = 1/invc; chi alone is
    // the rounded c, so chi*invc must land within a rounding of 1.
    for (md.log_tab, md.log_tab2) |e, c| {
        try std.testing.expectApproxEqAbs(@as(f64, 1), (c.chi + c.clo) * e.invc, 0x1p-50);
    }
    for (md.log2_tab, md.log2_tab2) |e, c| {
        try std.testing.expectApproxEqAbs(@as(f64, 1), (c.chi + c.clo) * e.invc, 0x1p-50);
    }
    // Both f32 log tables split the same 16 subintervals, so they share invc
    // and differ only in the base of logc. Entry 9 is the one holding x = 1.
    for (md.logf_tab, md.log2f_tab) |a, b| try std.testing.expectEqual(a.invc, b.invc);
    try std.testing.expectEqual(@as(f64, 1), md.logf_tab[9].invc);
    try std.testing.expectEqual(@as(f64, 0), md.logf_tab[9].logc);
    try std.testing.expectEqual(@as(f64, 0), md.log2f_tab[9].logc);

    // l10hi keeps 25 significant bits so `y1 * l10hi` in softLog10 is exact,
    // and l10hi + l10lo is 1/ln10 to 2^-82.
    try std.testing.expectEqual(@as(u64, 0), @as(u64, @bitCast(@as(f64, l10hi))) & ((1 << 28) - 1));
    try std.testing.expectEqual(@as(f64, invln10), @as(f64, l10hi) + @as(f64, l10lo));
}
