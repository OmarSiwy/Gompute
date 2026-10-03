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
//! `expm1`, `log1p` and `atan` fail differently and only on AMD, which is how they went
//! unnoticed: both assemble to PTX cleanly, and both die on AMDGCN. Not for
//! want of a libcall — std's ports raise the subnormal underflow flag through
//! `std.mem.doNotOptimizeAway`, which for a float is `asm volatile ("" :: "rm"
//! (v))`, and the AMDGPU backend cannot match the `m` alternative. That flag is
//! a register no GPU exposes, so the idiom is dead weight on device and a hard
//! error there. See `expm1` and `log1p` for the ports and `atan` for the cheaper dodge.
//!
//! `exp exp2 log log2 log10 pow` are ports of ARM optimized-routines (what
//! glibc >= 2.28 ships), f32 sin/cos follow musl's `sinf` in binary64, f32
//! tanh/sinh/cosh/pow are binary64 arithmetic on the f64 exp/log2 rounded
//! once, and f64 sin/cos/tanh/sinh/cosh/expm1 are musl's. Table-driven, no
//! libm and no float builtin beyond `+ - * /`, `@sqrt` and bit casts.
//!
//! ONE body per function on every target: no FMA, no hardware approximation,
//! no inline asm, no intrinsic. The same input gives the same bits on the
//! host (with or without FMA hardware), NVPTX, AMDGCN and at comptime, which
//! is what VerA's constant folding and ESPice's CPU/GPU comparison rely on;
//! `examples/exhaustive` checks CUDA against the CPU bit for bit.
//!
//! Accuracy, as tested at the bottom of this file:
//!
//!   f64 exp log pow              0.505 / 0.502 / 0.502 ulp vs an f128
//!                                oracle; faithful, and monotone.
//!   f64 exp2 log2 log10          <=1 / <=1 / <=2 ulp vs glibc (log10's
//!                                slack is glibc's own).
//!   f64 sin cos                  musl's bits, = compiler_rt's `@sin`/`@cos`.
//!   f64 tanh sinh / cosh         musl's bits (2.05 / 1.75 ulp) / 0.998 ulp.
//!   f32 everything               faithful over ALL 2^32 inputs: exp exp2
//!                                tanh cosh 0.50, sin cos sinh 0.501, log
//!                                log2 log10 <= 0.82 ulp.
//!
//! Every entry point takes a scalar `f32`/`f64` or a `@Vector` of either. The
//! CPU backend instantiates a generic map body at `@Vector` width, so a kernel
//! written the way the guide recommends hands these functions vectors. Real
//! SIMD, lane for lane the scalar's bits: exp, exp2, log, log2, log10 and pow
//! at f32 and f64, f32 sin/cos/tanh/sinh/cosh, sqrt and rsqrt. The vector form
//! is the scalar's own width-generic main path (`expMain`, `logMain`, ...);
//! a vector holding any special lane takes the scalar body lane by lane. f64
//! sin/cos/tanh/sinh/cosh/expm1, log1p and atan still run lane by lane.
//!
//! Already safe everywhere and deliberately absent here: `@abs @trunc @round
//! @floor @ceil @copysign @min @max`, `std.math.scalbn/frexp/modf`. `sqrt` and
//! `rsqrt` ARE here despite being safe, so that a kernel need not keep two
//! lists in its head — they are `@sqrt` with the argument checked.
//!
//! `@mulAdd` is NOT in that list, and nothing here uses it. Every body is the
//! reference's no-FMA variant, so the bits are the same on every target: host
//! with or without FMA hardware, NVPTX and AMDGCN. That bit-identity is the
//! contract VerA's constant folding and ESPice's CPU/GPU comparison rely on.
//! `@mulAdd` would also lower to a `fma()` libm CALL where there is no FMA,
//! which is a link error on device.

const std = @import("std");
const builtin = @import("builtin");

const arch = builtin.target.cpu.arch;
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
    if (Elem(@TypeOf(x)) == f32) return if (comptime isVec(@TypeOf(x))) expfVec(x) else softExpf(x);
    if (comptime isVec(@TypeOf(x))) return expVec(x, false);
    return softExp(x);
}

/// 2^x, exact for integer x. Same table as `exp`, its own reduction. f32 is
/// one rounding of the f64 body, so the bits are the same on every target --
/// `@exp2` would be `ex2.approx.f32` on NVIDIA and something else elsewhere.
pub inline fn exp2(x: anytype) @TypeOf(x) {
    if (Elem(@TypeOf(x)) == f32) {
        if (comptime !isVec(@TypeOf(x))) return viaF64(softExp2)(x);
        // Widen, run the f64 vector path, round once: each lane is exactly
        // the scalar's `viaF64`.
        return @floatCast(expVec(@as(At(@TypeOf(x), f64), @floatCast(x)), true));
    }
    if (comptime isVec(@TypeOf(x))) return expVec(x, true);
    return softExp2(x);
}

/// Natural log, <=1 ulp on host AND device from one body: ARM
/// optimized-routines `log` for f64, `logf` for f32.
///
/// The f32 half is what closed this module's worst accuracy hole. It used to
/// be `lg2.approx.f * ln2`, and that hardware is bounded ABSOLUTELY (~2^-21),
/// not relatively — so log(x) for x near 1 was 2^-21 of noise on a near-zero
/// answer, which is exactly where a SPICE junction sits.
pub inline fn log(x: anytype) @TypeOf(x) {
    if (Elem(@TypeOf(x)) == f32) return if (comptime isVec(@TypeOf(x))) logfVec(x, .ln) else softLnf(x);
    if (comptime isVec(@TypeOf(x))) return logVec(x, .ln);
    return softLog(x);
}

/// Base-2 log, <=1 ulp on host AND device: ARM optimized-routines
/// `log2`/`log2f`, with its own 64-entry table rather than `log * 1/ln2`.
/// Powers of two come back exact, which the scaled form could not manage.
pub inline fn log2(x: anytype) @TypeOf(x) {
    if (Elem(@TypeOf(x)) == f32) return if (comptime isVec(@TypeOf(x))) logfVec(x, .log2) else softLog2f(x);
    if (comptime isVec(@TypeOf(x))) return logVec(x, .log2);
    return softLog2(x);
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
    if (Elem(@TypeOf(x)) == f32) return if (comptime isVec(@TypeOf(x))) logfVec(x, .log10) else softLog10f(x);
    if (comptime isVec(@TypeOf(x))) return logVec(x, .log10);
    return softLog10(x);
}

/// musl's sin (via `softSin`), one body on every target; f32 is one rounding
/// of it. Not `@sin`: that is compiler_rt on the host and `sin.approx.f32`
/// (~1e-6 ABSOLUTE, no relative accuracy near k*pi) on NVIDIA, so the same
/// kernel gave different bits per target.
pub inline fn sin(x: anytype) @TypeOf(x) {
    if (Elem(@TypeOf(x)) == f32) return trigf(x, false);
    return apply(softSin, x);
}

/// musl's cos, as `sin`.
pub inline fn cos(x: anytype) @TypeOf(x) {
    if (Elem(@TypeOf(x)) == f32) return trigf(x, true);
    return apply(softCos, x);
}

/// ponytail: sin/cos, so error blows up near the poles where cos goes to zero.
/// A dedicated tan with its own argument reduction is worth writing only if
/// someone is actually near pi/2.
pub inline fn tan(x: anytype) @TypeOf(x) {
    _ = Elem(@TypeOf(x));
    return sin(x) / cos(x); // both already handle a vector
}

/// musl's tanh (std's `tanh64`), with expm1 the one-body `softExpm1`.
pub inline fn tanh(x: anytype) @TypeOf(x) {
    if (Elem(@TypeOf(x)) == f32) return hypf(x, .tanh);
    return apply(softTanh, x);
}

/// musl's sinh (std's `sinh64`), with expm1 and exp the one-body ports.
pub inline fn sinh(x: anytype) @TypeOf(x) {
    if (Elem(@TypeOf(x)) == f32) return hypf(x, .sinh);
    return apply(softSinh, x);
}

/// musl's cosh (std's `cosh64`), as `sinh`.
pub inline fn cosh(x: anytype) @TypeOf(x) {
    if (Elem(@TypeOf(x)) == f32) return hypf(x, .cosh);
    return apply(softCosh, x);
}

/// e^x - 1, without the cancellation `exp(x) - 1` suffers near zero.
///
/// Here because `std.math.expm1` DOES NOT COMPILE FOR AMDGCN. Its tiny-argument
/// branch raises the underflow flag through `std.mem.doNotOptimizeAway`, which
/// for a float lowers to `asm volatile ("" :: "rm" (v))`, and the AMDGPU backend
/// cannot match the `m` alternative. The body is std's own musl port with that
/// one line dropped, so every return is bit-identical to `std.math.expm1`.
pub inline fn expm1(x: anytype) @TypeOf(x) {
    if (Elem(@TypeOf(x)) == f32) return apply(viaF64(softExpm1), x);
    return apply(softExpm1, x);
}

/// log(1 + x), std's musl port with the same one line dropped as `expm1`.
///
/// Zig 0.17's `std.math.log1p` gained the subnormal `doNotOptimizeAway`, so it
/// stopped compiling for AMDGCN too. Every return is bit-identical to std.
pub inline fn log1p(x: anytype) @TypeOf(x) {
    if (Elem(@TypeOf(x)) == f32) return apply(viaF64(softLog1p), x);
    return apply(softLog1p, x);
}

fn softLog1p(x: f64) f64 {
    const ln2_hi: f64 = 6.93147180369123816490e-01;
    const ln2_lo: f64 = 1.90821492927058770002e-10;
    const Lg1: f64 = 6.666666666666735130e-01;
    const Lg2: f64 = 3.999999999940941908e-01;
    const Lg3: f64 = 2.857142874366239149e-01;
    const Lg4: f64 = 2.222219843214978396e-01;
    const Lg5: f64 = 1.818357216161805012e-01;
    const Lg6: f64 = 1.531383769920937332e-01;
    const Lg7: f64 = 1.479819860511658591e-01;

    const ix: u64 = @bitCast(x);
    const hx: u32 = @intCast(ix >> 32);
    var k: i32 = 1;
    var c: f64 = undefined;
    var f: f64 = undefined;

    if (hx < 0x3FDA827A or hx >> 31 != 0) { // 1 + x < sqrt(2)
        if (ix == 0xBFF0000000000000) return x / 0.0; // log1p(-1) = -inf
        if (hx >= 0xBFF00000) return (x - x) / 0.0; // x < -1: nan
        // |x| < 2^-53. std raises underflow here for a subnormal x; dropped.
        if ((hx << 1) < (0x3CA00000 << 1)) return x;
        if (hx <= 0xBFD2BEC4) { // sqrt(2)/2- <= 1 + x < sqrt(2)+
            k = 0;
            c = 0;
            f = x;
        }
    } else if (hx >= 0x7FF00000) {
        return x;
    }

    if (k != 0) {
        const uf = 1 + x;
        const hu: u64 = @bitCast(uf);
        var iu: u32 = @intCast(hu >> 32);
        iu += 0x3FF00000 - 0x3FE6A09E;
        k = @as(i32, @intCast(iu >> 20)) - 0x3FF;

        // correction to avoid underflow in c / u
        if (k < 54) {
            c = if (k >= 2) 1 - (uf - x) else x - (uf - 1);
            c /= uf;
        } else {
            c = 0;
        }

        // u into [sqrt(2)/2, sqrt(2)]
        iu = (iu & 0x000FFFFF) + 0x3FE6A09E;
        const iq = (@as(u64, iu) << 32) | (hu & 0xFFFFFFFF);
        f = @as(f64, @bitCast(iq)) - 1;
    }

    const hfsq = 0.5 * f * f;
    const s = f / (2.0 + f);
    const z = s * s;
    const w = z * z;
    const t1 = w * (Lg2 + w * (Lg4 + w * Lg6));
    const t2 = z * (Lg1 + w * (Lg3 + w * (Lg5 + w * Lg7)));
    const R = t2 + t1;
    const dk: f64 = @floatFromInt(k);

    return s * (hfsq + R) + (dk * ln2_lo + c) - hfsq + f + dk * ln2_hi;
}

/// arctangent: `std.math.atan`'s VECTOR body on every target, scalars as a
/// two-lane splat. std's scalar body raises the underflow flag through
/// `doNotOptimizeAway`, which does not compile for AMDGCN; the vector body
/// never reaches it. Two lanes because `@Vector(1, f64)` crashes the AMDGPU
/// backend. Within an ulp of std's scalar atan.
pub inline fn atan(x: anytype) @TypeOf(x) {
    _ = Elem(@TypeOf(x));
    if (comptime isVec(@TypeOf(x))) return std.math.atan(x);
    const v: @Vector(2, @TypeOf(x)) = @splat(x);
    return std.math.atan(v)[0];
}

/// f32 tanh, sinh and cosh: binary64 arithmetic on one f64 `exp`, rounded
/// once. Branch-free -- both sides of each split are computed and selected --
/// so the scalar and every vector width are this one function, and a vector
/// gets `exp`'s SIMD path. Near 0 an odd series replaces the cancelling
/// difference; its first dropped term is under 2^-30 relative, and the
/// exponential side loses at most ~3 bits to cancellation, so the f32 result
/// is faithful with ~25 bits to spare.
inline fn hypf(x: anytype, comptime which: enum { tanh, sinh, cosh }) @TypeOf(x) {
    const T = @TypeOf(x);
    const D = At(T, f64);
    const U = At(T, u64);
    const xd: D = @floatCast(x);
    const ax = @abs(xd);
    const x2 = xd * xd;
    const sign = @as(U, @bitCast(xd)) & sp(U, 1 << 63);
    // Every NaN returns x itself, explicitly: NVPTX gave -inf for a
    // negative-NaN sinh when the NaN was left to propagate through the
    // arithmetic, and the bits must not depend on the backend.
    const r: D = switch (which) {
        .tanh => blk: {
            const small = xd + xd * x2 * (sp(D, -1.0 / 3.0) + x2 * (sp(D, 2.0 / 15.0) + x2 * sp(D, -17.0 / 315.0)));
            // tanh(10) rounds to 1 in f32; clamping keeps e finite.
            const e = exp(sp(D, 2) * @min(ax, sp(D, 10)));
            const big: D = @bitCast(@as(U, @bitCast((e - sp(D, 1)) / (e + sp(D, 1)))) | sign);
            // @min drops a NaN, so put it back.
            // The sign bit again so -0 stays -0: the series sums -0 and +0.
            const near: D = @bitCast(@as(U, @bitCast(small)) | sign);
            break :blk sel(xd != xd, xd, sel(ax < sp(D, 0.0625), near, big));
        },
        .sinh => blk: {
            const small = xd + xd * x2 * (sp(D, 1.0 / 6.0) + x2 * (sp(D, 1.0 / 120.0) + x2 * sp(D, 1.0 / 5040.0)));
            const e = exp(ax);
            const big: D = @bitCast(@as(U, @bitCast(sp(D, 0.5) * e - sp(D, 0.5) / e)) | sign);
            break :blk sel(xd != xd, xd, sel(ax < sp(D, 0.25), small, big));
        },
        .cosh => blk: {
            const e = exp(ax);
            break :blk sel(xd != xd, xd, sp(D, 0.5) * e + sp(D, 0.5) / e);
        },
    };
    return @floatCast(r);
}

/// f32 sin or cos, musl's `sinf`/`cosf` design in binary64: n = round(x*2/pi),
/// y = x - n*pi/2 in two parts (exact first product: pio2_1 has 25 bits and
/// |n| < 2^28), then musl's double-precision `__sindf`/`__cosdf` polynomials,
/// picked by the quadrant with a select. One rounding to f32 at the end.
/// |x| >= 2^28*pi/2, inf and NaN take Payne-Hanek through `rem`, lane by lane.
inline fn trigf(x: anytype, comptime is_cos: bool) @TypeOf(x) {
    @setEvalBranchQuota(100_000);
    const T = @TypeOf(x);
    const D = At(T, f64);
    const xd: D = @floatCast(x);
    if (comptime !isVec(T)) {
        if (!(@abs(xd) < 0x1p28 * 1.5707963267948966)) return @floatCast(trigfLarge(xd, is_cos));
    } else if (!@reduce(.And, @abs(xd) < sp(D, 0x1p28 * 1.5707963267948966))) {
        @branchHint(.unlikely);
        return apply(if (is_cos) cosfLane else sinfLane, x);
    }
    const U = At(T, u64);
    const shifted = xd * sp(D, 6.36619772367581382433e-01) + sp(D, 0x1.8p52); // round to nearest even
    const n: U = @bitCast(shifted);
    const nd = shifted - sp(D, 0x1.8p52);
    const y = (xd - nd * sp(D, 1.57079631090164184570e+00)) - nd * sp(D, 1.58932547735281966916e-08);
    return @floatCast(trigfQuadrant(D, U, y, n, is_cos));
}

/// sin(y) or cos(y) by the quadrant `n` (the low two bits), |y| <= ~pi/4.
inline fn trigfQuadrant(comptime D: type, comptime U: type, y: D, n: U, comptime is_cos: bool) D {
    const z = y * y;
    const w = z * z;
    // musl k_sinf.c: |sin(y)/y - s(y)| < 2^-37.5.
    const s3 = sp(D, -0x1a00f9e2cae774.0p-65) + z * sp(D, 0x16cd878c3b46a7.0p-71);
    const sz = z * y;
    const sv = (y + sz * (sp(D, -0x15555554cbac77.0p-55) + z * sp(D, 0x111110896efbb2.0p-59))) + sz * w * s3;
    // musl k_cosf.c: |cos(y) - c(y)| < 2^-34.1.
    const c3 = sp(D, -0x16c087e80f1e27.0p-62) + z * sp(D, 0x199342e0ee5069.0p-68);
    const cv = ((sp(D, 1.0) + z * sp(D, -0x1ffffffd0c5e81.0p-54)) + w * sp(D, 0x155553e1053a42.0p-57)) + (w * z) * c3;
    // cos(x) = sin(x + pi/2): one more quarter turn.
    const q = n +% sp(U, @intFromBool(is_cos));
    const odd = q & sp(U, 1) != sp(U, 0);
    const neg = q & sp(U, 2) != sp(U, 0);
    const v = sel(odd, cv, sv);
    return sel(neg, -v, v);
}

fn sinfLane(x: f32) f32 {
    return trigf(x, false);
}
fn cosfLane(x: f32) f32 {
    return trigf(x, true);
}

/// The scalar route for |x| >= 2^28*pi/2, inf and NaN.
fn trigfLarge(xd: f64, comptime is_cos: bool) f64 {
    var y: [2]f64 = undefined;
    const n = rem.remPio2(xd, &y); // nan for inf/nan
    return trigfQuadrant(f64, u64, y[0], @bitCast(@as(i64, n)), is_cos);
}

/// `c ? a : b` lane-wise, or the plain choice for a scalar.
inline fn sel(c: anytype, a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    return if (comptime isVec(@TypeOf(a))) @select(ElemOf(@TypeOf(a)), c, a, b) else if (c) a else b;
}

/// An f64 body as an f32 one: widen, run, round once.
fn viaF64(comptime f: fn (f64) f64) fn (f32) f32 {
    return struct {
        fn g(x: f32) f32 {
            return @floatCast(f(x));
        }
    }.g;
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
/// One fast path stays in front of it: y == 0 or x == 1 is 1, as C99 says,
/// even when the other operand is NaN.
///
/// Integer y goes through the table route like every other y. It used to take
/// square-and-multiply for |y| <= 64, which is faster but rounds differently:
/// pow stepped DOWN across y = -64 (a prover evaluating at interval ends
/// relies on monotone pow), and six squarings are not faithful. The table
/// route is faithful (0.50 ulp measured), so an exact x^y still comes back
/// exact: pow(2, 10) is 1024.
///
/// f32 goes through the f64 body: one algorithm, one test surface, and a
/// single f32 rounding of a 0.52-ulp f64 result is correctly rounded. Nothing
/// in gompute or its consumers calls `pow` on f32 (Verilog-A `real` is f64),
/// so the device-side f64 rate penalty buys accuracy nobody pays for.
/// ponytail: if an f32 kernel ever wants a cheap pow, `exp2(y * log2 x)` on
/// the f32 hardware path is the thing to bring back, for f32 only.
pub inline fn pow(x: anytype, y: @TypeOf(x)) @TypeOf(x) {
    const T = @TypeOf(x);
    if (Elem(T) == f32) return powf(x, y);
    if (comptime !isVec(T)) return powBody(x, y);
    return powVec(x, y);
}

/// f32 x^y: exp2(y * log2(x)) in binary64 for x positive and finite and y
/// finite. f64 log2 and exp2 are faithful and |y*log2 x| < ~150 anywhere the
/// f32 result is finite and non-zero, so the double result is good to ~2^-46
/// relative and its one rounding to f32 is faithful -- correctly rounded but
/// for inputs within 2^-46 of a midpoint, which no exact case is. Everything
/// else (x <= 0, inf, NaN) goes the f64 `pow` way, so C99's table holds.
inline fn powf(x: anytype, y: @TypeOf(x)) @TypeOf(x) {
    @setEvalBranchQuota(100_000);
    const T = @TypeOf(x);
    const D = At(T, f64);
    const xd: D = @floatCast(x);
    const yd: D = @floatCast(y);
    const inf = sp(D, std.math.inf(f64));
    const ok = if (comptime isVec(T))
        @reduce(.And, xd > sp(D, 0)) and @reduce(.And, xd < inf) and @reduce(.And, @abs(yd) < inf)
    else
        xd > 0 and xd < inf and @abs(yd) < inf;
    if (ok) return @floatCast(exp2(yd * log2(xd)));
    if (comptime !isVec(T)) return @floatCast(powBody(xd, yd));
    return apply2(powfLane, x, y);
}

fn powfLane(x: f32, y: f32) f32 {
    return powf(x, y);
}

inline fn powBody(x: anytype, y: @TypeOf(x)) @TypeOf(x) {
    const T = @TypeOf(x);
    if (y == 0 or x == 1) return 1;
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
const rem = @import("math_pio2.zig");

/// Top 12 bits of a double: sign and biased exponent.
inline fn top12(x: f64) u32 {
    return @truncate(@as(u64, @bitCast(x)) >> 52);
}

// ---------------------------------------------------------------------------
// Width-generic arithmetic. `expMain` and `logMain` are the ordinary-input
// halves of `exp` and `log`, written once for `f64` and `@Vector(n, f64)`.
// The scalar bodies call them at width 1 and the vector entry points at width
// n, so a vector lane is the scalar call's result bit for bit by construction,
// not by a test that happens to pass. Special inputs never reach them: the
// scalar body branches around them, the vector entry falls back to the scalar
// body lane by lane when any lane is special.
// ---------------------------------------------------------------------------

/// `u64` for `f64`, `@Vector(n, u64)` for `@Vector(n, f64)`.
fn Bits(comptime T: type) type {
    return if (isVec(T)) @Vector(@typeInfo(T).vector.len, u64) else u64;
}
/// `i64` likewise.
fn SBits(comptime T: type) type {
    return if (isVec(T)) @Vector(@typeInfo(T).vector.len, i64) else i64;
}
/// `c` at `T`'s width.
inline fn sp(comptime T: type, c: anytype) T {
    return if (comptime isVec(T)) @splat(c) else c;
}
/// A shift amount at `U`'s width: Zig wants a vector of `u6` for a vector shift.
inline fn shamt(comptime U: type, comptime n: comptime_int) At(U, std.math.Log2Int(ElemOf(U))) {
    return sp(At(U, std.math.Log2Int(ElemOf(U))), n);
}
/// `E` at the width of `T`: itself for a scalar, `@Vector(n, E)` for n lanes.
fn At(comptime T: type, comptime E: type) type {
    return if (isVec(T)) @Vector(@typeInfo(T).vector.len, E) else E;
}
fn ElemOf(comptime T: type) type {
    return if (isVec(T)) @typeInfo(T).vector.child else T;
}
/// `tab[idx]` for each lane, as one array of K columns. Each lane is one
/// K-wide load of its row and the rows are transposed in registers: K loads
/// for W lanes instead of K*W scalar loads, which was a third of vector `log`.
inline fn gatherRows(comptime E: type, comptime K: usize, comptime tab: []const [K]E, idx: anytype) [K]if (isVec(@TypeOf(idx))) @Vector(@typeInfo(@TypeOf(idx)).vector.len, E) else E {
    if (comptime !isVec(@TypeOf(idx))) return tab[@intCast(idx)];
    const W = @typeInfo(@TypeOf(idx)).vector.len;
    var cols: [K]@Vector(W, E) = undefined;
    if (comptime W % 4 != 0 or (K != 2 and K != 4)) {
        inline for (0..W) |l| {
            const row: @Vector(K, E) = tab[@intCast(idx[l])];
            inline for (0..K) |c| cols[c][l] = row[c];
        }
        return cols;
    }
    // Four lanes at a time: four whole-row loads, then an explicit transpose.
    // Element-by-element assembly let LLVM split every row into scalar loads.
    inline for (0..W / 4) |g| {
        const r: [4]@Vector(K, E) = .{
            tab[@intCast(idx[4 * g + 0])], tab[@intCast(idx[4 * g + 1])],
            tab[@intCast(idx[4 * g + 2])], tab[@intCast(idx[4 * g + 3])],
        };
        const q: [K]@Vector(4, E) = if (K == 2) blk: {
            // {r0, r1} and {r2, r3} side by side, then split even/odd.
            const a = @shuffle(E, r[0], r[1], [4]i32{ 0, 1, -1, -2 });
            const b = @shuffle(E, r[2], r[3], [4]i32{ 0, 1, -1, -2 });
            break :blk .{
                @shuffle(E, a, b, [4]i32{ 0, 2, -1, -3 }),
                @shuffle(E, a, b, [4]i32{ 1, 3, -2, -4 }),
            };
        } else blk: {
            // The AVX 4x4 transpose: unpack lo/hi pairs, then swap 128-bit halves.
            const t0 = @shuffle(E, r[0], r[1], [4]i32{ 0, -1, 2, -3 });
            const t1 = @shuffle(E, r[0], r[1], [4]i32{ 1, -2, 3, -4 });
            const t2 = @shuffle(E, r[2], r[3], [4]i32{ 0, -1, 2, -3 });
            const t3 = @shuffle(E, r[2], r[3], [4]i32{ 1, -2, 3, -4 });
            break :blk .{
                @shuffle(E, t0, t2, [4]i32{ 0, 1, -1, -2 }),
                @shuffle(E, t1, t3, [4]i32{ 0, 1, -1, -2 }),
                @shuffle(E, t0, t2, [4]i32{ 2, 3, -3, -4 }),
                @shuffle(E, t1, t3, [4]i32{ 2, 3, -3, -4 }),
            };
        };
        inline for (0..K) |c| inline for (0..4) |l| {
            cols[c][4 * g + l] = q[c][l];
        };
    }
    return cols;
}

/// The f32 tables as rows: `expf_tab` one entry wide, the two f32 log tables
/// their (invc, logc) pairs.
const expf_rows: *const [32][1]u64 = @ptrCast(&md.expf_tab);
const logf_rows: *const [16][2]f64 = @ptrCast(&md.logf_tab);
const log2f_rows: *const [16][2]f64 = @ptrCast(&md.log2f_tab);

/// log2's two tables fused the same way.
const log2_rows: [64][4]f64 align(32) = blk: {
    var t: [64][4]f64 = undefined;
    for (&t, md.log2_tab, md.log2_tab2) |*r, a, c| r.* = .{ a.invc, a.logc, c.chi, c.clo };
    break :blk t;
};

/// `exp_tab` as the (tail, sbits) pairs `expMain` reads together.
const exp_rows: *const [128][2]u64 = @ptrCast(&md.exp_tab);

/// `log_tab` and `log_tab2` fused into the one row `logMain` reads:
/// invc, logc, chi, clo. Built at comptime, so the data is unchanged.
const log_rows: [128][4]f64 align(32) = blk: {
    var t: [128][4]f64 = undefined;
    for (&t, md.log_tab, md.log_tab2) |*r, a, c| r.* = .{ a.invc, a.logc, c.chi, c.clo };
    break :blk t;
};

/// exp(x + xtail) ~= scale * (1 + tmp), scale = `sbits` as a double. The
/// reduction exp(x) = 2^(k/128) * exp(r), r in [-ln2/256, ln2/256].
inline fn expMain(comptime T: type, x: T, xtail: T, sign_bias: u64) struct { tmp: T, sbits: Bits(T), ki: Bits(T) } {
    const U = Bits(T);
    const z = sp(T, md.invln2N) * x;
    const shifted = z + sp(T, md.shift); // forces the round-to-nearest-int
    const ki: U = @bitCast(shifted);
    const kd = shifted - sp(T, md.shift);
    const r = x + kd * sp(T, md.negln2hiN) + kd * sp(T, md.negln2loN) + xtail;

    const row = gatherRows(u64, 2, exp_rows, ki & sp(U, 127));
    const top = (ki +% sp(U, sign_bias)) << shamt(U, 52 - 7);
    const tail: T = @bitCast(row[0]);
    const sbits = row[1] +% top; // valid while -1023*128 < k < 1024*128

    const c = md.exp_poly;
    const r2 = r * r;
    const tmp = tail + r + r2 * (sp(T, c[0]) + r * sp(T, c[1])) + r2 * r2 * (sp(T, c[2]) + r * sp(T, c[3]));
    return .{ .tmp = tmp, .sbits = sbits, .ki = ki };
}

/// log(x) as an unnormalized `hi + lo` for a positive, normal, finite x off
/// the band around 1, where the table path applies.
inline fn logMain(comptime T: type, ix: Bits(T)) struct { hi: T, lo: T } {
    const U = Bits(T);
    // x = 2^k z with z in [OFF, 2*OFF) exactly, z near c = 1/invc.
    const tmp = ix -% sp(U, log_off);
    const i = (tmp >> shamt(U, 52 - 7)) & sp(U, 127);
    const k = @as(SBits(T), @bitCast(tmp)) >> shamt(U, 52); // arithmetic
    const z: T = @bitCast(ix -% (tmp & sp(U, @as(u64, 0xfff) << 52)));

    // r = z/c - 1, |r| < 1/256. Without an FMA the reference subtracts c as a
    // double-double first, which is why `log_tab2` exists.
    const row = gatherRows(f64, 4, &log_rows, i);
    const r = (z - row[2] - row[3]) * row[0];

    // hi + lo = r + log(c) + k*ln2, exactly -- that is what the table's
    // rounding of logc buys.
    const kd = kAsF64(T, k);
    const w = kd * sp(T, md.log_ln2hi) + row[1];
    const hi = w + r;
    const t = w - hi + r + kd * sp(T, md.log_ln2lo);

    const a = md.log_poly;
    const r2 = r * r;
    const lo = t + r2 * sp(T, a[0]) + r * r2 * (sp(T, a[1]) + r * sp(T, a[2]) + r2 * (sp(T, a[3]) + r * sp(T, a[4])));
    return .{ .hi = hi, .lo = lo };
}

/// `exp` at vector width: one `expMain` over the whole vector when every lane
/// is ordinary, which is the scalar body's own main path, so the lanes match it
/// bit for bit. Any tiny, huge, inf or NaN lane sends the vector to the scalar
/// body lane by lane.
inline fn expVec(x: anytype, comptime two: bool) @TypeOf(x) {
    @setEvalBranchQuota(100_000); // the lane unrolls at W = 16
    const T = @TypeOf(x);
    const U = Bits(T);
    const abstop = (@as(U, @bitCast(x)) >> shamt(U, 52)) & sp(U, 0x7ff);
    const tiny = comptime top12(0x1p-54);
    if (!@reduce(.And, abstop -% sp(U, tiny) < sp(U, comptime top12(512.0) - tiny))) {
        @branchHint(.unlikely);
        return apply(if (two) softExp2 else softExp, x);
    }
    const m = if (two) exp2Main(T, x) else expMain(T, x, sp(T, 0.0), 0);
    const scale: T = @bitCast(m.sbits);
    return scale + scale * m.tmp;
}

/// `log` at vector width, by the same rule as `expVec`: the table path over
/// the whole vector when every lane takes it, the scalar body otherwise.
inline fn logVec(x: anytype, comptime which: enum { ln, log2, log10 }) @TypeOf(x) {
    @setEvalBranchQuota(100_000); // the lane unrolls at W = 16
    const T = @TypeOf(x);
    const U = Bits(T);
    const ix: U = @bitCast(x);
    const lo_band, const hi_band = if (which == .log2) .{ log2_near1_lo, log2_near1_hi } else .{ log_near1_lo, log_near1_hi };
    const normal = (ix >> shamt(U, 48)) -% sp(U, 0x0010) < sp(U, 0x7ff0 - 0x0010);
    const near1 = ix -% sp(U, lo_band) < sp(U, hi_band - lo_band);
    if (!@reduce(.And, normal) or @reduce(.Or, near1)) {
        @branchHint(.unlikely);
        return apply(switch (which) {
            .ln => softLog,
            .log2 => softLog2,
            .log10 => softLog10,
        }, x);
    }
    if (which == .log2) return log2Main(T, ix);
    const p = logMain(T, ix);
    return if (which == .ln) p.lo + p.hi else log10Tail(T, p.hi, p.lo);
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
    const l = powLogMain(f64, ix);
    tail.* = l.tail;
    return l.y;
}

/// `md.powlog_tab` as rows: invc, pad, logc, logctail -- already the 32 bytes
/// one lane reads.
const powlog_rows: *const [128][4]f64 = @ptrCast(&md.powlog_tab);

/// pow's log(x) as `y + tail` to ~68 bits, at any width.
inline fn powLogMain(comptime T: type, ix: Bits(T)) struct { y: T, tail: T } {
    const U = Bits(T);
    // x = 2^k z with z in [OFF, 2*OFF) exactly; the range is cut into 128
    // subintervals and c sits near the centre of the one z lands in.
    const off: u64 = 0x3fe6955500000000;
    const tmp = ix -% sp(U, off);
    const i = (tmp >> shamt(U, 52 - 7)) & sp(U, 127);
    const k = @as(SBits(T), @bitCast(tmp)) >> shamt(U, 52); // arithmetic
    const iz = ix -% (tmp & sp(U, @as(u64, 0xfff) << 52));
    const z: T = @bitCast(iz);
    const kd = kAsF64(T, k);

    const row = gatherRows(f64, 4, powlog_rows, i);
    const invc = row[0];

    // 1/c is j/128 or j/256 for integer j, and |z/c - 1| < 1/128, so
    // r = z/c - 1 is exactly representable. Without an FMA it takes a split of
    // z into halves whose products are exact.
    const zhi: T = @bitCast((iz +% sp(U, 1 << 31)) & sp(U, ~@as(u64, 0) << 32));
    const zlo = z - zhi;
    const rhi = zhi * invc - sp(T, 1.0);
    const rlo = zlo * invc;
    const r = rhi + rlo;

    // k*ln2 + log(c) + r, in double-double.
    const t1 = kd * sp(T, md.powlog_ln2hi) + row[2];
    const t2 = t1 + r;
    const lo1 = kd * sp(T, md.powlog_ln2lo) + row[3];
    const lo2 = t1 - t2 + r;

    // Ordered for a superscalar pipeline, not for readability.
    const a = md.powlog_poly;
    const ar = sp(T, a[0]) * r; // a[0] = -0.5
    const ar2 = r * ar;
    const ar3 = r * ar2;
    const arhi = sp(T, a[0]) * rhi;
    const arhi2 = rhi * arhi;
    const hi = t2 + arhi2;
    const lo3 = rlo * (ar + arhi);
    const lo4 = t2 - hi + arhi2;
    // p = log1p(r) - r - a[0]*r*r.
    const p = ar3 * (sp(T, a[1]) + r * sp(T, a[2]) + ar2 * (sp(T, a[3]) + r * sp(T, a[4]) + ar2 * (sp(T, a[5]) + r * sp(T, a[6]))));

    const lo = lo1 + lo2 + lo3 + lo4 + p;
    const y = hi + lo;
    return .{ .y = y, .tail = hi - y + lo };
}

/// f64 `pow` at vector width: log, the y split and exp over the whole vector
/// when every lane is ordinary (x positive, normal and finite, |y| in the
/// table range, y*log(x) inside exp's main range), which is softPow's own
/// main path; the scalar body lane by lane otherwise.
inline fn powVec(x: anytype, y: @TypeOf(x)) @TypeOf(x) {
    @setEvalBranchQuota(100_000); // the lane unrolls at W = 16
    const T = @TypeOf(x);
    const U = Bits(T);
    const ix: U = @bitCast(x);
    const iy: U = @bitCast(y);
    const topy = (iy >> shamt(U, 52)) & sp(U, 0x7ff);
    const ok_x = (ix >> shamt(U, 52)) -% sp(U, 1) < sp(U, 0x7ff - 1);
    const ok_y = topy -% sp(U, 0x3be) < sp(U, 0x43e - 0x3be);
    if (@reduce(.And, ok_x) and @reduce(.And, ok_y)) {
        const l = powLogMain(T, ix);
        const mask27 = sp(U, ~@as(u64, 0) << 27);
        const yhi: T = @bitCast(iy & mask27);
        const ylo = y - yhi;
        const lhi: T = @bitCast(@as(U, @bitCast(l.y)) & mask27);
        const llo = l.y - lhi + l.tail;
        const ehi = yhi * lhi;
        const elo = ylo * lhi + y * llo; // |elo| < |ehi| * 2^-25
        const abstop = (@as(U, @bitCast(ehi)) >> shamt(U, 52)) & sp(U, 0x7ff);
        const tiny = comptime top12(0x1p-54);
        if (@reduce(.And, abstop -% sp(U, tiny) < sp(U, comptime top12(512.0) - tiny))) {
            const m = expMain(T, ehi, elo, 0);
            const scale: T = @bitCast(m.sbits);
            return scale + scale * m.tmp;
        }
    }
    return apply2(powBody, x, y);
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

    const m = expMain(f64, x, xtail, sign_bias);
    if (abstop == 0) return expSpecial(1009, m.tmp, m.sbits, m.ki);
    const scale: f64 = @bitCast(m.sbits);
    const tmp = m.tmp;
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
    const yhi: f64 = @bitCast(iy & (~@as(u64, 0) << 27));
    const ylo = y - yhi;
    const lhi: f64 = @bitCast(@as(u64, @bitCast(hi)) & (~@as(u64, 0) << 27));
    const llo = hi - lhi + lo;
    const ehi = yhi * lhi;
    const elo = ylo * lhi + y * llo; // |elo| < |ehi| * 2^-25
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

/// musl `tanh.c`, by way of std's `tanh64`, with `softExpm1` for expm1. The
/// subnormal branch's `doNotOptimizeAway` only raised a flag and is dropped.
fn softTanh(x: f64) f64 {
    const u: u64 = @bitCast(x);
    const ux = u & 0x7FFFFFFFFFFFFFFF;
    const w: u32 = @intCast(ux >> 32);
    const ax: f64 = @bitCast(ux);
    var t: f64 = undefined;
    if (w > 0x3FE193EA) { // |x| > log(3)/2 ~= 0.5493, or nan
        if (w > 0x40340000) { // |x| > 20, or nan
            t = 1.0 - 0 / ax;
        } else {
            t = softExpm1(2 * ax);
            t = 1 - 2 / (t + 2);
        }
    } else if (w > 0x3FD058AE) { // |x| > log(5/3)/2 ~= 0.2554
        t = softExpm1(2 * ax);
        t = t / (t + 2);
    } else if (w >= 0x00100000) { // |x| >= 0x1p-1022
        t = softExpm1(-2 * ax);
        t = -t / (t + 2);
    } else t = ax; // subnormal
    return if (u >> 63 != 0) -t else t;
}

/// musl `sinh.c` (std's `sinh64`), with the one-body expm1 and exp.
fn softSinh(x: f64) f64 {
    const u: u64 = @bitCast(x);
    const w: u32 = @as(u32, @intCast(u >> 32)) & 0x7FFFFFFF;
    const ax: f64 = @bitCast(u & 0x7FFFFFFFFFFFFFFF);
    if (x == 0.0 or std.math.isNan(x)) return x;
    const h: f64 = if (u >> 63 != 0) -0.5 else 0.5;
    if (w < 0x40862E42) { // |x| < log(DBL_MAX)
        const t = softExpm1(ax);
        if (w < 0x3FF00000) {
            if (w < 0x3FF00000 - (26 << 20)) return x;
            return h * (2 * t - t * t / (t + 1));
        }
        return h * (t + t / (t + 1));
    }
    return 2 * h * expo2(ax);
}

/// musl `cosh.c` (std's `cosh64`), with the one-body expm1 and exp.
fn softCosh(x: f64) f64 {
    const u: u64 = @bitCast(x);
    const w: u32 = @as(u32, @intCast(u >> 32)) & 0x7FFFFFFF;
    const ax: f64 = @bitCast(u & 0x7FFFFFFFFFFFFFFF);
    if (x == 0.0) return 1.0;
    if (w < 0x3FE62E42) { // |x| < log(2)
        if (w < 0x3FF00000 - (26 << 20)) return 1.0;
        const t = softExpm1(ax);
        return 1 + t * t / (2 * (1 + t));
    }
    if (w < 0x40862E42) { // |x| < log(DBL_MAX)
        const t = softExp(ax);
        return 0.5 * (t + 1 / t);
    }
    return expo2(ax);
}

/// exp(x)/2 without overflowing for x past log(DBL_MAX): musl `__expo2`.
fn expo2(x: f64) f64 {
    const kln2 = 0x1.62066151ADD8BP+10; // k = 2043
    const scale: f64 = @bitCast(@as(u64, 0x3FF + 2043 / 2) << 52);
    return softExp(x - kln2) * scale * scale;
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
/// 2^x = 2^(k/128) * 2^r with integer k and |r| <= 1/256, both exact; at
/// any width, as `expMain`.
inline fn exp2Main(comptime T: type, x: T) struct { tmp: T, sbits: Bits(T), ki: Bits(T) } {
    const U = Bits(T);
    const shifted = x + sp(T, md.exp2_shift);
    const ki: U = @bitCast(shifted);
    const r = x - (shifted - sp(T, md.exp2_shift));
    const row = gatherRows(u64, 2, exp_rows, ki & sp(U, 127));
    const tail: T = @bitCast(row[0]);
    const sbits = row[1] +% (ki << shamt(U, 52 - 7));
    const c = md.exp2_poly;
    const r2 = r * r;
    const tmp = tail + r * sp(T, c[0]) + r2 * (sp(T, c[1]) + r * sp(T, c[2])) + r2 * r2 * (sp(T, c[3]) + r * sp(T, c[4]));
    return .{ .tmp = tmp, .sbits = sbits, .ki = ki };
}

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

    const m = exp2Main(f64, x);
    if (abstop == 0) return expSpecial(1, m.tmp, m.sbits, m.ki);
    const scale: f64 = @bitCast(m.sbits);
    const tmp = m.tmp;
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

/// The band around 1, [LO, HI) as bit patterns, where log switches from the
/// table to a direct series in x - 1.
const log_near1_lo: u64 = @bitCast(@as(f64, 1.0 - 0x1p-4));
const log_near1_hi: u64 = @bitCast(@as(f64, 1.0 + 0x1.09p-4));

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

    if (ix -% log_near1_lo < log_near1_hi - log_near1_lo) {
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
    const m = logMain(f64, ix);
    hi.* = m.hi;
    lo.* = m.lo;
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
/// 26, so `y1 * l10hi` fits in 51 bits: no rounding and no FMA. Everything after it is 2^-27 of the answer, so the only rounding
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
    return log10Tail(f64, hi, lo);
}

/// (hi + lo)/ln10 at any width; see `softLog10`.
inline fn log10Tail(comptime T: type, hi: T, lo: T) T {
    const U = Bits(T);
    const y = hi + lo;
    const b = y - hi;
    const yl = (hi - (y - b)) + (lo - b); // y + yl = hi + lo, exactly
    const y1: T = @bitCast(@as(U, @bitCast(y)) & sp(U, ~@as(u64, 0) << 27));
    const y2 = y - y1;
    return y1 * sp(T, l10hi) + (y2 * sp(T, l10hi) + (y * sp(T, l10lo) + yl * sp(T, invln10)));
}

fn softLog2(x: f64) f64 {
    var ix: u64 = @bitCast(x);

    if (ix -% log2_near1_lo < log2_near1_hi - log2_near1_lo) {
        if (ix == one_bits) return 0;
        const r = x - 1.0;
        // r/ln2 in double-double.
        const rhi: f64 = @bitCast(@as(u64, @bitCast(r)) & (~@as(u64, 0) << 32));
        const rlo = r - rhi;
        const h = rhi * md.log2_invln2hi;
        const l = rlo * md.log2_invln2hi + r * md.log2_invln2lo;
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

    return log2Main(f64, ix);
}

/// log2's table path at any width, for x positive, normal, finite and off
/// its band around 1. 64 subintervals here, not 128: log2's k folds in as an
/// exact integer rather than through a k*ln2 double-double, so half the
/// table reaches the same accuracy.
inline fn log2Main(comptime T: type, ix: Bits(T)) T {
    const U = Bits(T);
    const tmp = ix -% sp(U, log_off);
    const i = (tmp >> shamt(U, 52 - 6)) & sp(U, 63);
    const k = @as(SBits(T), @bitCast(tmp)) >> shamt(U, 52);
    const z: T = @bitCast(ix -% (tmp & sp(U, @as(u64, 0xfff) << 52)));
    const row = gatherRows(f64, 4, &log2_rows, i);

    const r = (z - row[2] - row[3]) * row[0];
    const rhi: T = @bitCast(@as(U, @bitCast(r)) & sp(U, ~@as(u64, 0) << 32));
    const rlo = r - rhi;
    const t1 = rhi * sp(T, md.log2_invln2hi);
    const t2 = rlo * sp(T, md.log2_invln2hi) + r * sp(T, md.log2_invln2lo);

    // hi + lo = r/ln2 + log2(c) + k, exactly: `k + logc` is why logc is
    // rounded so that 1024 + logc has no rounding error.
    const t3 = kAsF64(T, k) + row[1];
    const hi = t3 + t1;
    const lo = t3 - hi + t1 + t2;

    const a = md.log2_poly;
    const r2 = r * r;
    const r4 = r2 * r2;
    const p = sp(T, a[0]) + r * sp(T, a[1]) + r2 * (sp(T, a[2]) + r * sp(T, a[3])) + r4 * (sp(T, a[4]) + r * sp(T, a[5]));
    return lo + r2 * p + hi;
}

/// An exponent k (|k| < 2^51) as a double, exactly, by the magic-number trick:
/// AVX2 has no i64 -> f64 vector convert, and `@floatFromInt` went lane by lane.
inline fn kAsF64(comptime T: type, k: SBits(T)) T {
    const magic = comptime @as(u64, @bitCast(@as(f64, 0x1.8p52)));
    return @as(T, @bitCast(@as(Bits(T), @bitCast(k)) +% sp(Bits(T), magic))) - sp(T, 0x1.8p52);
}

/// log2's band around 1, where it switches to a series in x - 1.
const log2_near1_lo: u64 = @bitCast(@as(f64, 1.0 - 0x1.5b51p-5));
const log2_near1_hi: u64 = @bitCast(@as(f64, 1.0 + 0x1.6ab2p-5));

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

    return expfMain(f32, x);
}

/// expf's ordinary path, |x| < 88, at any width: in binary64, rounded once.
/// x*32/ln2 = k + r with integer k and |r| <= 1/2; the 32-entry table is
/// every 4th entry of the f64 one.
inline fn expfMain(comptime T: type, x: T) T {
    const D = At(T, f64);
    const U = At(T, u64);
    const z = sp(D, md.expf_invln2N) * @as(D, @floatCast(x));
    const shifted = z + sp(D, md.expf_shift);
    const ki: U = @bitCast(shifted);
    const r = z - (shifted - sp(D, md.expf_shift));

    const s: D = @bitCast(gatherRows(u64, 1, expf_rows, ki & sp(U, 31))[0] +% (ki << shamt(U, 52 - 5)));
    const c = md.expf_poly;
    const r2 = r * r;
    return @floatCast(((sp(D, c[0]) * r + sp(D, c[1])) * r2 + (sp(D, c[2]) * r + sp(D, 1))) * s);
}

/// f32 `exp` at vector width: `expfMain` over the whole vector when no lane
/// is past 88 in magnitude, the scalar body lane by lane otherwise.
inline fn expfVec(x: anytype) @TypeOf(x) {
    @setEvalBranchQuota(100_000); // the lane unrolls at W = 16
    const U = At(@TypeOf(x), u32);
    const abstop = (@as(U, @bitCast(x)) >> shamt(U, 20)) & sp(U, 0x7ff);
    if (!@reduce(.And, abstop < sp(U, comptime top12f(88.0) & 0x7ff))) {
        @branchHint(.unlikely);
        return apply(softExpf, x);
    }
    return expfMain(@TypeOf(x), x);
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

    const m = logfMain(f32, ix, @ptrCast(tab), kscale);
    r.* = m.r;
    y0.* = m.y0;
    return null;
}

/// logfSplit's ordinary path (x positive, normal, finite) at any width:
/// x = 2^k z, r = z/c - 1 off a 16-entry table, y0 = log(c) + k*kscale.
inline fn logfMain(comptime T: type, ix: At(T, u32), comptime tab: *const [16][2]f64, comptime kscale: f64) struct { r: At(T, f64), y0: At(T, f64) } {
    const D = At(T, f64);
    const U = At(T, u32);
    const tmp = ix -% sp(U, 0x3f330000);
    const i = (tmp >> shamt(U, 23 - 4)) & sp(U, 15);
    const k = @as(At(T, i32), @bitCast(tmp)) >> shamt(U, 23); // arithmetic
    const z: D = @floatCast(@as(T, @bitCast(ix -% (tmp & sp(U, 0xff800000)))));
    const row = gatherRows(f64, 2, tab, i);
    return .{
        .r = z * row[0] - sp(D, 1),
        .y0 = row[1] + @as(D, @floatFromInt(k)) * sp(D, kscale),
    };
}

/// f32 log, log2 or log10 at vector width: the table path over the whole
/// vector when every lane is positive, normal and finite, the scalar body
/// lane by lane otherwise. x = 1 takes the table path too, and gives the +0
/// the scalar's own early exit does.
inline fn logfVec(x: anytype, comptime which: enum { ln, log2, log10 }) @TypeOf(x) {
    @setEvalBranchQuota(100_000); // the lane unrolls at W = 16
    const T = @TypeOf(x);
    const D = At(T, f64);
    const U = At(T, u32);
    const ix: U = @bitCast(x);
    if (!@reduce(.And, ix -% sp(U, 0x00800000) < sp(U, 0x7f800000 - 0x00800000))) {
        @branchHint(.unlikely);
        return apply(switch (which) {
            .ln => softLnf,
            .log2 => softLog2f,
            .log10 => softLog10f,
        }, x);
    }
    if (which == .log2) {
        const m = logfMain(T, ix, log2f_rows, 1.0);
        const a = md.log2f_poly;
        const r2 = m.r * m.r;
        return @floatCast((sp(D, a[0]) * r2 + (sp(D, a[1]) * m.r + sp(D, a[2]))) * r2 + (sp(D, a[3]) * m.r + m.y0));
    }
    const m = logfMain(T, ix, logf_rows, md.logf_ln2);
    const a = md.logf_poly;
    const r2 = m.r * m.r;
    const y = (sp(D, a[0]) * r2 + (sp(D, a[1]) * m.r + sp(D, a[2]))) * r2 + (m.y0 + m.r);
    return @floatCast(if (which == .ln) y else y * sp(D, invln10));
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
// sin / cos -- musl k_sin.c and k_cos.c here, __rem_pio2 (with Payne-Hanek
// for large |x|) in math_pio2.zig
// ---------------------------------------------------------------------------

const pio4 = 0x1.921fb54442d18p-1;

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

fn softSin(x: f64) f64 {
    const ax = @abs(x);
    if (ax < pio4) return if (ax < 0x1p-27) x else kernelSin(x, 0.0, false);
    if (!std.math.isFinite(x)) return std.math.nan(f64);
    var y: [2]f64 = undefined;
    const n = rem.remPio2(x, &y);
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
    const n = rem.remPio2(x, &y);
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
        try std.testing.expect(ulpErr(f32, softLnf(@floatCast(x)), @log(@as(f32, @floatCast(x)))) <= 2);
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
        try std.testing.expect(std.math.isNegativeInf(softLnf(@floatCast(x))));
        try std.testing.expect(std.math.isNegativeInf(softLog2f(@floatCast(x))));
    }
    for ([_]f64{ -1, -0x1p-1074, -std.math.inf(f64), std.math.nan(f64) }) |x| {
        try std.testing.expect(std.math.isNan(softLog(x)));
        try std.testing.expect(std.math.isNan(softLog2(x)));
    }
    // -0x1p-1074 casts to -0.0 in f32, which is a -inf not a nan, so the f32
    // list is its own.
    for ([_]f32{ -1, -0x1p-149, -std.math.inf(f32), std.math.nan(f32) }) |x| {
        try std.testing.expect(std.math.isNan(softLnf(x)));
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
    // Large arguments take Payne-Hanek, the same reduction as compiler_rt's
    // `@sin`, so they agree bit for bit all the way up.
    for ([_]f64{ 1e9, 0x1p20 * 1.5707963267948966, 1e22, 1e100, 1e300, -1e308, std.math.floatMax(f64) }) |x| {
        try std.testing.expectEqual(@as(u64, @bitCast(@sin(x))), @as(u64, @bitCast(softSin(x))));
        try std.testing.expectEqual(@as(u64, @bitCast(@cos(x))), @as(u64, @bitCast(softCos(x))));
    }
}

test "log1p is bit-identical to std" {
    // Same claim as expm1's port. Edges are the branch splits: -1, 2^-53,
    // the sqrt(2)/2 and sqrt(2) bounds, k = 54, a subnormal, inf.
    for ([_]f64{
        0,      -0.0,    -1,     -0.9999, 0x1p-54, -0x1p-54, 0x1p-53, 4.9e-324,
        -0.293, -0.2929, 0.4142, 0.4143,  1,       0x1p54,   1e300,   std.math.inf(f64),
    }) |x| try std.testing.expectEqual(
        @as(u64, @bitCast(std.math.log1p(x))),
        @as(u64, @bitCast(log1p(x))),
    );
    var i: i32 = -99_999;
    while (i < 200_000) : (i += 1) {
        const x = @as(f64, @floatFromInt(i)) * 1e-5;
        try std.testing.expectEqual(
            @as(u64, @bitCast(std.math.log1p(x))),
            @as(u64, @bitCast(log1p(x))),
        );
    }
    try std.testing.expect(std.math.isNan(log1p(@as(f64, -2))));
}

test "expm1 is bit-identical to std, and atan is within an ulp" {
    // The whole claim of the port: it drops a line that touched only the FP
    // flag register, so every VALUE must match std exactly. Bit equality, over
    // the branch boundaries the algorithm actually switches on -- 2^-54, the
    // 0.5*ln2 and 1.5*ln2 reduction splits, 56*ln2, and the overflow threshold.
    const edges = [_]f64{
        0,        -0.0,    0x1p-60, -0x1p-60, 0x1p-54,
        -0x1p-54, 0x1p-53, 1e-300,  -1e-300,  0.3465,
        -0.3465,  0.3466,  -0.3466, 1.0397,   -1.0397,
        1.0398,   -1.0398, 1,       -1,       0.25,
        -0.25,    -0.2501, 2,       -2,       38.8,
        -38.8,    38.9,    709.78,  709.79,   710,
        -745,     1e300,   -1e300,
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
    const U = @Int(.unsigned, @bitSizeOf(T));
    const I = @Int(.signed, @bitSizeOf(T));
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
    if (libm.real(Fn64, "exp") == null) return error.SkipZigTest;
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
}

test "f32 exp, log, log2 and log10 match libm to 1 ulp" {
    if (dev or !builtin.link_libc) return error.SkipZigTest;
    if (libm.real(Fn32, "expf") == null) return error.SkipZigTest;
    const D = Diff(f32);

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
        .{ "logf", softLnf, "logf", true },
        .{ "log2f", softLog2f, "log2f", true },
        .{ "log10f", softLog10f, "log10f", true },
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

// The tables' exactness, checked at compile time on every target -- device
// builds included -- rather than only by a host `zig build test`. One digit
// wrong in a pasted table breaks these before it breaks an ulp bound.
comptime {
    @setEvalBranchQuota(20_000);
    const need = struct {
        fn f(ok: bool, comptime what: []const u8) void {
            if (!ok) @compileError("math tables: " ++ what);
        }
    }.f;
    // pow: 1/c must have at most 8 fractional mantissa bits, or z/c - 1 is not
    // exactly representable and logInline's whole error budget is void.
    for (md.powlog_tab) |e| {
        need(@as(u64, @bitCast(e.invc)) & ((1 << 44) - 1) == 0, "powlog invc has more than 8 fraction bits");
        // logc is rounded to a multiple of 2^-43 so k*ln2hi + logc is exact.
        need(e.logc == @trunc(e.logc * 0x1p43) * 0x1p-43, "powlog logc is not a multiple of 2^-43");
    }
    // ln2hi and negln2hiN have their low bits cleared for the same reason.
    // negln2hiN keeps 36 significant bits, so kd*negln2hiN is exact for every
    // |kd| < 2^17 = 131072 -- exactly the 1024*128 the exp reduction reaches.
    need(@as(u64, @bitCast(md.powlog_ln2hi)) & ((1 << 11) - 1) == 0, "ln2hi has low bits set");
    need(@as(u64, @bitCast(md.negln2hiN)) & ((1 << 17) - 1) == 0, "negln2hiN has low bits set");
    // Entry 0 of the exp table is exactly 1.0 with a zero tail, and the f32
    // table is its every-4th-entry slice.
    need(md.exp_tab[0] == 0 and md.exp_tab[1] == one_bits, "exp_tab[0] is not 1.0");
    for (md.expf_tab, 0..) |t, i| need(md.exp_tab[8 * i + 1] == t, "expf_tab is not exp_tab's slice");

    // log: `k*ln2hi + logc` is error-free only because logc was rounded so
    // that `0x1.8p9 + logc` is. log2 wants `k + logc` error-free, hence 0x1.8p10.
    for (md.log_tab) |e| need(e.logc == (0x1.8p9 + e.logc) - 0x1.8p9, "log_tab logc is not exact");
    for (md.log2_tab) |e| need(e.logc == (0x1.8p10 + e.logc) - 0x1.8p10, "log2_tab logc is not exact");
    // The non-fma paths need chi + clo to reproduce c = 1/invc; chi alone is
    // the rounded c, so chi*invc must land within a rounding of 1.
    for (md.log_tab, md.log_tab2) |e, c| need(@abs((c.chi + c.clo) * e.invc - 1) <= 0x1p-50, "log_tab2 does not invert log_tab");
    for (md.log2_tab, md.log2_tab2) |e, c| need(@abs((c.chi + c.clo) * e.invc - 1) <= 0x1p-50, "log2_tab2 does not invert log2_tab");
    // Both f32 log tables split the same 16 subintervals, so they share invc
    // and differ only in the base of logc. Entry 9 is the one holding x = 1.
    for (md.logf_tab, md.log2f_tab) |a, b| need(a.invc == b.invc, "logf and log2f tables disagree on invc");
    need(md.logf_tab[9].invc == 1 and md.logf_tab[9].logc == 0 and md.log2f_tab[9].logc == 0, "entry 9 does not hold x = 1");

    // l10hi keeps 25 significant bits so `y1 * l10hi` in softLog10 is exact,
    // and l10hi + l10lo is 1/ln10 to 2^-82.
    need(@as(u64, @bitCast(@as(f64, l10hi))) & ((1 << 28) - 1) == 0, "l10hi has more than 25 bits");
    need(@as(f64, invln10) == @as(f64, l10hi) + @as(f64, l10lo), "l10hi + l10lo is not 1/ln10");
}

test "vector f64 exp, exp2, log, log2 and log10 are the scalar call, lane for lane" {
    var prng = std.Random.DefaultPrng.init(0x1A4E_E4AC);
    const rnd = prng.random();
    const specials = [_]f64{ 0, -0.0, 1, -1, 0x1p-1074, 0x1p-1022, 0x1p-60, 709.78, 710, -745.2, -746, 1e308, std.math.inf(f64), -std.math.inf(f64), std.math.nan(f64), 1.0 + 0x1p-5, 1.0 - 0x1p-5 };
    inline for (.{ 2, 4, 8 }) |w| {
        const V = @Vector(w, f64);
        for (0..50_000) |n| {
            var xe: V = undefined;
            var xl: V = undefined;
            inline for (0..w) |l| {
                // Mostly all-ordinary vectors (the vector path), every 8th with
                // a special lane (the fallback), which must agree just the same.
                xe[l] = (rnd.float(f64) - 0.5) * 1400;
                xl[l] = @exp((rnd.float(f64) - 0.5) * 1400);
                if (n % 8 == 0 and rnd.boolean()) {
                    xe[l] = specials[rnd.uintLessThan(usize, specials.len)];
                    xl[l] = specials[rnd.uintLessThan(usize, specials.len)];
                }
            }
            const ye = exp(xe);
            const y2 = exp2(xe);
            const yl = log(xl);
            const yl2 = log2(xl);
            const yl10 = log10(xl);
            inline for (0..w) |l| {
                try std.testing.expectEqual(@as(u64, @bitCast(softExp(xe[l]))), @as(u64, @bitCast(ye[l])));
                try std.testing.expectEqual(@as(u64, @bitCast(softExp2(xe[l]))), @as(u64, @bitCast(y2[l])));
                try std.testing.expectEqual(@as(u64, @bitCast(softLog(xl[l]))), @as(u64, @bitCast(yl[l])));
                try std.testing.expectEqual(@as(u64, @bitCast(softLog2(xl[l]))), @as(u64, @bitCast(yl2[l])));
                try std.testing.expectEqual(@as(u64, @bitCast(softLog10(xl[l]))), @as(u64, @bitCast(yl10[l])));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The contract VerA's constant folding and prover rely on, from VerA's own
// harness (tools/contract.zig): faithful against an f128 oracle, comptime ==
// runtime, C99 F.10 special and exact values, and monotone.
// ---------------------------------------------------------------------------

/// Measured-and-promised max error against the f128 oracle, per function.
/// Faithful (< 1) is the contract; these are the tighter numbers we keep.
pub const max_ulp = struct {
    pub const exp: f64 = 0.52;
    pub const log: f64 = 0.52;
    pub const pow: f64 = 0.55;
};

const oracle = struct {
    const ln2: f128 = 0x1.62e42fefa39ef35793c7673007e6p-1;
    fn expQ(t: f128) f128 {
        const k = @round(t / ln2);
        const r = t - k * ln2;
        var term: f128 = 1;
        var sum: f128 = 1;
        var n: f128 = 1;
        while (n < 36) : (n += 1) {
            term = term * r / n;
            sum += term;
        }
        return std.math.ldexp(sum, @intFromFloat(k));
    }
    fn logQ(x: f64) f128 {
        const fr = std.math.frexp(@as(f128, x));
        var m = fr.significand;
        var e: f128 = @floatFromInt(fr.exponent);
        if (m < 0.70710678) {
            m *= 2;
            e -= 1;
        }
        const u = (m - 1) / (m + 1);
        const uu = u * u;
        var term = u;
        var sum: f128 = 0;
        var k: f128 = 1;
        while (k < 100) : (k += 2) {
            sum += term / k;
            term *= uu;
        }
        return 2 * sum + e * ln2;
    }
    /// |got - ref| in ulps of the double nearest ref.
    fn ulp(got: f64, ref: f128) f64 {
        const rd: f64 = @floatCast(ref);
        const e = @max(std.math.ilogb(rd) - 52, -1074);
        return @floatCast(@abs(@as(f128, got) - ref) / std.math.ldexp(@as(f128, 1.0), e));
    }
};

test "exp, log and pow stay inside max_ulp of an f128 oracle" {
    if (dev) return error.SkipZigTest;
    var prng = std.Random.DefaultPrng.init(0x7a11);
    const r = prng.random();
    var worst = [3]f64{ 0, 0, 0 };
    for (0..20_000) |_| {
        const x = r.float(f64) * 1440.0 - 735.0;
        worst[0] = @max(worst[0], oracle.ulp(exp(x), oracle.expQ(x)));
        const lx: f64 = @bitCast((r.int(u64) % 0x7fe0000000000000) + 0x0010000000000000);
        worst[1] = @max(worst[1], oracle.ulp(log(lx), oracle.logQ(lx)));
        const nx = 1.0 + (r.float(f64) - 0.5) * 0x1p-3; // log's band around 1
        worst[1] = @max(worst[1], oracle.ulp(log(nx), oracle.logQ(nx)));
        const px = std.math.pow(f64, 10.0, r.float(f64) * 15.0 - 3.0);
        const py = r.float(f64) * 21.0 - 3.0;
        worst[2] = @max(worst[2], oracle.ulp(pow(px, py), oracle.expQ(@as(f128, py) * oracle.logQ(px))));
    }
    std.debug.print("\n  f128 oracle: exp {d:.3}, log {d:.3}, pow {d:.3} ulp\n", .{ worst[0], worst[1], worst[2] });
    try std.testing.expect(worst[0] < max_ulp.exp);
    try std.testing.expect(worst[1] < max_ulp.log);
    try std.testing.expect(worst[2] < max_ulp.pow);
}

test "exp, log and pow fold at comptime to the bits they run to" {
    // No FMA and no target intrinsic in any body, so comptime IEEE f64 is the
    // same arithmetic every backend runs.
    const S = struct {
        fn rt(x: f64) f64 {
            var v = x;
            std.mem.doNotOptimizeAway(&v);
            return v;
        }
    };
    const xs = [_]f64{ -744.9, -700.25, -20.5, -0.75, -1e-9, 0.0, 3e-17, 0.693, 1.0, 22.0, 512.5, 709.7 };
    const ls = [_]f64{ 0x1p-1060, 1e-300, 1e-9, 0.5, 0.999999, 1.0, 1.0000001, 3.0, 1e18, 1e308 };
    inline for (xs) |x| {
        const folded = comptime blk: {
            @setEvalBranchQuota(1_000_000);
            break :blk exp(x);
        };
        try std.testing.expectEqual(@as(u64, @bitCast(folded)), @as(u64, @bitCast(exp(S.rt(x)))));
    }
    inline for (ls) |x| {
        const fl, const fp = comptime blk: {
            @setEvalBranchQuota(1_000_000);
            break :blk .{ log(x), pow(x, 1.4552480184709202) };
        };
        try std.testing.expectEqual(@as(u64, @bitCast(fl)), @as(u64, @bitCast(log(S.rt(x)))));
        try std.testing.expectEqual(@as(u64, @bitCast(fp)), @as(u64, @bitCast(pow(S.rt(x), 1.4552480184709202))));
    }
}

test "exp, log and pow: C99 F.10 special values and exact cases" {
    const inf = std.math.inf(f64);
    const nan = std.math.nan(f64);
    try std.testing.expectEqual(@as(f64, 1), exp(@as(f64, 0)));
    try std.testing.expectEqual(@as(f64, 1), exp(@as(f64, -0.0)));
    try std.testing.expectEqual(inf, exp(inf));
    try std.testing.expectEqual(@as(u64, 0), @as(u64, @bitCast(exp(-inf))));
    try std.testing.expect(std.math.isNan(exp(nan)));
    try std.testing.expectEqual(@as(f64, 0x1p-1074), exp(@as(f64, -0x1.74910d52d3051p9)));
    try std.testing.expectEqual(@as(f64, 0), exp(@as(f64, -0x1.74910d52d3052p9)));
    try std.testing.expect(exp(@as(f64, 0x1.62e42fefa39efp9)) < inf);
    try std.testing.expectEqual(inf, exp(@as(f64, 0x1.62e42fefa39f0p9)));
    try std.testing.expectEqual(std.math.e, exp(@as(f64, 1)));

    try std.testing.expectEqual(@as(u64, 0), @as(u64, @bitCast(log(@as(f64, 1))))); // +0
    try std.testing.expectEqual(-inf, log(@as(f64, 0)));
    try std.testing.expectEqual(-inf, log(@as(f64, -0.0)));
    try std.testing.expectEqual(inf, log(inf));
    try std.testing.expect(std.math.isNan(log(@as(f64, -1))));
    try std.testing.expect(std.math.isNan(log(nan)));

    try std.testing.expectEqual(@as(f64, 1), pow(@as(f64, -3), 0));
    try std.testing.expectEqual(@as(f64, 1), pow(@as(f64, 1), nan));
    try std.testing.expectEqual(@as(f64, 1024), pow(@as(f64, 2), 10));
    try std.testing.expectEqual(@as(f64, -8), pow(@as(f64, -2), 3));
    try std.testing.expectEqual(@as(f64, 0.0625), pow(@as(f64, -2), -4));
    try std.testing.expectEqual(@as(f64, 2), pow(@as(f64, 4), 0.5));
    try std.testing.expectEqual(@as(f64, 0.1), pow(@as(f64, 0.1), 1));
    try std.testing.expect(std.math.isNan(pow(@as(f64, -2), 0.5)));
    try std.testing.expectEqual(inf, pow(@as(f64, 0), -1));
    try std.testing.expectEqual(-inf, pow(@as(f64, -0.0), -1));
    try std.testing.expectEqual(@as(u64, 1 << 63), @as(u64, @bitCast(pow(@as(f64, -0.0), 3))));
    try std.testing.expectEqual(@as(f64, 0), pow(@as(f64, 0.5), inf));
    try std.testing.expectEqual(inf, pow(@as(f64, 2), inf));
    try std.testing.expectEqual(inf, pow(@as(f64, 10), 400));
    try std.testing.expectEqual(@as(f64, 0), pow(@as(f64, 10), -400));
}

/// f(x) <= f(next(x)) for `n` adjacent-ulp steps either side of `x0`.
fn nonDecreasingAround(comptime f: fn (f64) f64, x0: f64, n: usize) !void {
    var x = x0;
    for (0..n) |_| x = std.math.nextAfter(f64, x, -std.math.inf(f64));
    var prev = f(x);
    for (0..2 * n) |_| {
        x = std.math.nextAfter(f64, x, std.math.inf(f64));
        const y = f(x);
        if (std.math.isNan(y) or std.math.isNan(prev)) {
            prev = y;
            continue;
        }
        if (y < prev) {
            std.debug.print("not monotone at {e}: {e} after {e}\n", .{ x, y, prev });
            return error.NotMonotone;
        }
        prev = y;
    }
}

test "exp, log, pow, expm1, log1p and sinh are monotone across every table seam and branch" {
    if (dev) return error.SkipZigTest;
    const F = struct {
        fn e(x: f64) f64 {
            return exp(x);
        }
        fn l(x: f64) f64 {
            return log(x);
        }
        fn m1(x: f64) f64 {
            return expm1(x);
        }
        fn sh(x: f64) f64 {
            return sinh(x);
        }
        fn l1p(x: f64) f64 {
            return log1p(x);
        }
        var py: f64 = 0;
        var px: f64 = 0;
        fn powX(x: f64) f64 {
            return pow(x, py);
        }
        fn powY(y: f64) f64 {
            return pow(px, y);
        }
    };
    // exp: every k/128 rounding seam over the whole range, plus the branch
    // edges (2^-54, 512, the overflow and underflow ends).
    var j: f64 = -1075 * 128;
    while (j < 1024 * 128) : (j += 37) try nonDecreasingAround(F.e, (j + 0.5) / md.invln2N, 3);
    for ([_]f64{ 0x1p-54, -0x1p-54, 512, -512, 709.78, -708.4, -745.13, 0 }) |b| try nonDecreasingAround(F.e, b, 64);
    // log: every table subinterval edge in a few binades, both band edges,
    // and powers of two.
    for ([_]i32{ -1022, -300, -1, 0, 1, 300, 1023 }) |e2| {
        for (0..128) |i| {
            const bits = (log_off + (@as(u64, i) << 45)) & 0x000fffffffffffff | (@as(u64, @intCast(1023 + e2)) << 52);
            try nonDecreasingAround(F.l, @bitCast(bits), 3);
        }
        try nonDecreasingAround(F.l, std.math.ldexp(@as(f64, 1), e2), 32);
    }
    for ([_]f64{ 1.0 - 0x1p-4, 1.0 + 0x1.09p-4, 1, 0x1p-1022 }) |b| try nonDecreasingAround(F.l, b, 64);
    // expm1 and sinh at their reduction and branch edges.
    for ([_]f64{ 0, 0x1p-54, -0x1p-54, 0.3465, -0.3465, 1.0397, -1.0397, 38.8, -38.8, 709.78 }) |b| try nonDecreasingAround(F.m1, b, 32);
    for ([_]f64{ 0, 0.5, -0.5, 22, -22, 709.78 }) |b| try nonDecreasingAround(F.sh, b, 32);
    // log1p: musl switches at sqrt(2)/2-1, sqrt(2)-1, 2^-29 and 2^53.
    for ([_]f64{ -0.29289321881345254, 0.41421356237309503, 0x1p-29, -0x1p-29, 0, 0x1p53, -0.9999999999, 1e-300 }) |b| try nonDecreasingAround(F.l1p, b, 32);
    // pow in x for fixed y >= 0 (and decreasing in x for y < 0, checked as
    // increasing in 1/x... kept to y >= 0 here), and in y for fixed x > 1.
    for ([_]f64{ 0.5, 1.4552480184709202, 2, 3, 0.693, 7.25 }) |y| {
        F.py = y;
        for ([_]f64{ 0x1p-1022, 1e-10, 0.5, 1, 1.0 - 0x1p-4, 1.0 + 0x1.09p-4, 2, 1e10 }) |b| try nonDecreasingAround(F.powX, b, 32);
    }
    for ([_]f64{ 1.0000001, 1.5, 2, 10, 1e10 }) |x| {
        F.px = x;
        for ([_]f64{ -64.5, -64, -1, -0.5, 0, 0.5, 1, 63.5, 64, 64.5, 100 }) |b| try nonDecreasingAround(F.powY, b, 32);
    }
    // Dense random sample, sorted: exp and log over their whole domains.
    var prng = std.Random.DefaultPrng.init(0x3070_7070);
    const r = prng.random();
    var xs: [20_000]f64 = undefined;
    for (&xs) |*x| x.* = r.float(f64) * 1440.0 - 735.0;
    std.mem.sort(f64, &xs, {}, std.sort.asc(f64));
    for (xs[1..], xs[0 .. xs.len - 1]) |b, a| try std.testing.expect(exp(b) >= exp(a));
    for (&xs) |*x| x.* = @bitCast((r.int(u64) % 0x7fe0000000000000) + 0x0010000000000000);
    std.mem.sort(f64, &xs, {}, std.sort.asc(f64));
    for (xs[1..], xs[0 .. xs.len - 1]) |b, a| try std.testing.expect(log(b) >= log(a));
}

test "tanh, sinh, cosh, sin and cos keep std's values on the host" {
    if (dev) return error.SkipZigTest;
    // tanh and sinh are std's musl bodies with std's own expm1 algorithm, so
    // they are std's bits (f128-measured: tanh 2.05, sinh 1.75 ulp -- musl's
    // own error, not faithful). cosh's middle range calls our 0.505-ulp exp
    // where std calls compiler_rt's, so they part by up to 2 ulp and ours is
    // the closer: 0.998 ulp against f128, std 1.131. sin/cos are musl against
    // compiler_rt's musl, and agree on every bit sampled.
    var prng = std.Random.DefaultPrng.init(0x7a4b);
    const r = prng.random();
    var cosh_diff: usize = 0;
    var trig_diff: usize = 0;
    for (0..200_000) |n| {
        const x = if (n % 2 == 0) (r.float(f64) - 0.5) * 60 else (r.float(f64) - 0.5) * 1440;
        try std.testing.expectEqual(@as(u64, @bitCast(std.math.tanh(x))), @as(u64, @bitCast(tanh(x))));
        if (@abs(x) < 709) try std.testing.expectEqual(@as(u64, @bitCast(std.math.sinh(x))), @as(u64, @bitCast(sinh(x))));
        const c = ulpErr(f64, cosh(x), std.math.cosh(x));
        try std.testing.expect(c <= 2);
        if (c != 0) cosh_diff += 1;
        const s = ulpErr(f64, sin(x), @sin(x));
        const k = ulpErr(f64, cos(x), @cos(x));
        try std.testing.expect(s <= 1 and k <= 1);
        if (s != 0 or k != 0) trig_diff += 1;
    }
    std.debug.print("\n  vs host std/compiler_rt over 200k: cosh differs on {d}, sin/cos on {d}\n", .{ cosh_diff, trig_diff });
}

test "vector f32 exp, exp2, log, log2, log10, sin, cos and pow are the scalar call, lane for lane" {
    var prng = std.Random.DefaultPrng.init(0xF32_1A4E);
    const rnd = prng.random();
    const specials = [_]f32{ 0, -0.0, 1, -1, 0x1p-149, 0x1p-126, 88, 89, -104, -105, std.math.floatMax(f32), std.math.inf(f32), -std.math.inf(f32), std.math.nan(f32) };
    inline for (.{ 4, 8, 16 }) |w| {
        const V = @Vector(w, f32);
        for (0..50_000) |n| {
            var xe: V = undefined;
            var xl: V = undefined;
            inline for (0..w) |l| {
                xe[l] = (rnd.float(f32) - 0.5) * 200;
                xl[l] = @bitCast(rnd.int(u32) & 0x7fffffff);
                if (n % 8 == 0 and rnd.boolean()) {
                    xe[l] = specials[rnd.uintLessThan(usize, specials.len)];
                    xl[l] = specials[rnd.uintLessThan(usize, specials.len)];
                }
            }
            const e = exp(xe);
            const e2 = exp2(xe);
            inline for (0..w) |l| try std.testing.expectEqual(@as(u32, @bitCast(viaF64(softExp2)(xe[l]))), @as(u32, @bitCast(e2[l])));
            const sn = sin(xl);
            const cs = cos(xe);
            const pw = pow(xl, xe);
            inline for (0..w) |l| {
                try std.testing.expectEqual(@as(u32, @bitCast(sinfLane(xl[l]))), @as(u32, @bitCast(sn[l])));
                try std.testing.expectEqual(@as(u32, @bitCast(cosfLane(xe[l]))), @as(u32, @bitCast(cs[l])));
                const pl = powfLane(xl[l], xe[l]);
                if (!(std.math.isNan(pl) and std.math.isNan(pw[l])))
                    try std.testing.expectEqual(@as(u32, @bitCast(pl)), @as(u32, @bitCast(pw[l])));
            }
            const l1 = log(xl);
            const l2 = log2(xl);
            const l10 = log10(xl);
            inline for (0..w) |l| {
                try std.testing.expectEqual(@as(u32, @bitCast(softExpf(xe[l]))), @as(u32, @bitCast(e[l])));
                try std.testing.expectEqual(@as(u32, @bitCast(softLnf(xl[l]))), @as(u32, @bitCast(l1[l])));
                try std.testing.expectEqual(@as(u32, @bitCast(softLog2f(xl[l]))), @as(u32, @bitCast(l2[l])));
                try std.testing.expectEqual(@as(u32, @bitCast(softLog10f(xl[l]))), @as(u32, @bitCast(l10[l])));
            }
        }
    }
}

test "vector pow is the scalar call, lane for lane, f64 and f32" {
    var prng = std.Random.DefaultPrng.init(0x9013_1A4E);
    const rnd = prng.random();
    const sx = [_]f64{ 0, -0.0, 1, -1, -2, 0x1p-1074, std.math.inf(f64), std.math.nan(f64), 1e300 };
    const sy = [_]f64{ 0, 1, -1, 2, 3, 0.5, 1e300, -1e300, std.math.nan(f64), 0x1p-70 };
    inline for (.{ f64, f32 }) |T| inline for (.{ 4, 8 }) |w| {
        const V = @Vector(w, T);
        for (0..30_000) |n| {
            var x: V = undefined;
            var y: V = undefined;
            inline for (0..w) |l| {
                x[l] = @floatCast(std.math.pow(f64, 10.0, rnd.float(f64) * 30 - 15));
                y[l] = @floatCast(rnd.float(f64) * 40 - 20);
                if (n % 8 == 0 and rnd.boolean()) {
                    x[l] = @floatCast(sx[rnd.uintLessThan(usize, sx.len)]);
                    y[l] = @floatCast(sy[rnd.uintLessThan(usize, sy.len)]);
                }
            }
            const got = pow(x, y);
            const U = @Int(.unsigned, @bitSizeOf(T));
            inline for (0..w) |l| {
                const want = pow(x[l], y[l]);
                if (!(std.math.isNan(want) and std.math.isNan(got[l])))
                    try std.testing.expectEqual(@as(U, @bitCast(want)), @as(U, @bitCast(got[l])));
            }
        }
    };
}

test "f32 tanh, sinh and cosh are faithful, and the vector is the scalar lane for lane" {
    if (dev) return error.SkipZigTest;
    var prng = std.Random.DefaultPrng.init(0xF32_4A11);
    const r = prng.random();
    var worst = [3]f64{ 0, 0, 0 };
    for (0..300_000) |n| {
        const x: f32 = switch (n % 3) {
            0 => (r.float(f32) - 0.5) * 2,
            1 => (r.float(f32) - 0.5) * 200,
            else => @bitCast(r.int(u32)),
        };
        if (std.math.isNan(x)) continue;
        const xd: f64 = x;
        // The f64 musl bodies are ~1e-16 relative: an exact oracle at f32.
        inline for (.{ .{ tanh(x), softTanh(xd) }, .{ sinh(x), softSinh(xd) }, .{ cosh(x), softCosh(xd) } }, 0..) |c, k| {
            const got: f32 = c[0];
            const want: f64 = c[1];
            if (std.math.isInf(want) or @abs(want) > std.math.floatMax(f32)) {
                try std.testing.expect(std.math.isInf(got) or @abs(got) == std.math.floatMax(f32));
            } else if (want != 0) {
                const e = @abs(@as(f64, got) - want) / std.math.ldexp(@as(f64, 1), @max(std.math.ilogb(@as(f32, @floatCast(want))) - 23, -149));
                worst[k] = @max(worst[k], e);
            }
        }
    }
    std.debug.print("\n  f32 hyperbolics vs f64: tanh {d:.3}, sinh {d:.3}, cosh {d:.3} ulp\n", .{ worst[0], worst[1], worst[2] });
    for (worst) |w| try std.testing.expect(w < 1);
    try std.testing.expect(std.math.isNan(tanh(std.math.nan(f32))));
    try std.testing.expectEqual(@as(f32, 1), tanh(std.math.inf(f32)));
    try std.testing.expectEqual(@as(f32, -1), tanh(-std.math.inf(f32)));
    try std.testing.expectEqual(@as(u32, 1 << 31), @as(u32, @bitCast(tanh(@as(f32, -0.0)))));
    try std.testing.expectEqual(std.math.inf(f32), cosh(-std.math.inf(f32)));
    // Lane for lane.
    inline for (.{ 4, 8, 16 }) |w| {
        var v: @Vector(w, f32) = undefined;
        for (0..20_000) |_| {
            inline for (0..w) |l| v[l] = if (r.boolean()) (r.float(f32) - 0.5) * 30 else @bitCast(r.int(u32));
            const t = tanh(v);
            const sh = sinh(v);
            const ch = cosh(v);
            inline for (0..w) |l| {
                if (!std.math.isNan(v[l])) {
                    try std.testing.expectEqual(@as(u32, @bitCast(tanh(v[l]))), @as(u32, @bitCast(t[l])));
                    try std.testing.expectEqual(@as(u32, @bitCast(sinh(v[l]))), @as(u32, @bitCast(sh[l])));
                    try std.testing.expectEqual(@as(u32, @bitCast(cosh(v[l]))), @as(u32, @bitCast(ch[l])));
                }
            }
        }
    }
}
