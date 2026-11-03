//! Raw kernels exported for a GPU target. Never run.
//!
//! `src/device.zig` is reachable from no other test artifact, so this is what
//! puts the device root, `builtins.zig` and every `g.math` entry point through
//! the compiler for `nvptx64` and `amdgcn` -- all the way to PTX and ISA, where
//! `build.zig` checks the instructions the builtins promise.

const g = @import("gompute");

/// Every `g.math` entry point, three families at two widths.
///
/// `src/device/math.zig` exists because `@exp @log @sin` and friends fail at
/// the IR->ISA stage on NVPTX and AMDGCN — `no libcall available for fexp`,
/// `Cannot select: f32 = fsin`. That is a claim about the BACK END, so
/// `build.zig` assembles this probe down to real PTX rather than stopping at
/// the frontend. Split into families rather than one body, and the split is
/// load bearing: every entry point is `inline`, and at fifteen in one body the
/// AMDGCN backend SEGV'd the compiler under `zig build`'s server protocol.
fn expFamily(x: anytype) @TypeOf(x) {
    return g.math.exp(x) + g.math.exp2(x) + g.math.expm1(x) + g.math.pow(x, x);
}
fn logFamily(x: anytype) @TypeOf(x) {
    return g.math.log(x) + g.math.log2(x) + g.math.log10(x) + g.math.log1p(x) +
        g.math.sqrt(x) + g.math.rsqrt(x);
}
fn trigFamily(x: anytype) @TypeOf(x) {
    return g.math.sin(x) + g.math.cos(x) + g.math.tan(x) + g.math.atan(x) +
        g.math.tanh(x) + g.math.sinh(x) + g.math.cosh(x);
}

fn MathKernel(comptime T: type, comptime f: anytype) type {
    return struct {
        fn run(data: g.GlobalPtr(T), len: u64) callconv(g.kernel_callconv) void {
            const i = g.globalIdX(256);
            if (i < len) data[i] = f(data[i]);
        }
    };
}

/// `globalIdX` reads the launched block size out of the dispatch packet on AMD.
fn rawDouble(data: g.GlobalPtr(f32), len: u64) callconv(g.kernel_callconv) void {
    const i = g.globalIdX(256);
    if (i >= len) return;
    data[i] *= 2;
}

/// Pins the atomic lowerings `builtins.zig` documents.
fn rawAtomics(ticket: *addrspace(.global) u32, flag: *addrspace(.global) u32, out: g.GlobalPtr(u32)) callconv(g.kernel_callconv) void {
    const t = @atomicRmw(u32, ticket, .Add, 1, .monotonic);
    _ = @atomicRmw(u32, ticket, .Min, t, .monotonic);
    while (@atomicLoad(u32, flag, .acquire) != t) g.builtins.spinPause();
    out[t] = t;
    @atomicStore(u32, flag, t + 1, .release);
    while (g.builtins.loadAcquireDevice(flag) != t + 1) g.builtins.spinPause();
    g.builtins.storeReleaseDevice(flag, t + 2);
}

/// A block-wide sum, 256 threads: a `shuffleXor` fold per 32-lane group, the
/// group totals through shared memory across a `barrier`, the first group
/// folds those. Pins the shuffle and barrier lowerings.
var fold_scratch: [256 / 32]f32 addrspace(.shared) = undefined;
fn rawBlockSum(data: g.GlobalPtr(f32), out: g.GlobalPtr(f32)) callconv(g.kernel_callconv) void {
    const tid = g.builtins.localIdX();
    var v = data[g.globalIdX(256)];
    inline for (.{ 16, 8, 4, 2, 1 }) |m| v += g.builtins.shuffleXor(f32, v, m);
    if (tid % 32 == 0) fold_scratch[tid / 32] = v;
    g.builtins.barrier();
    if (tid < 32) {
        var w = if (tid < 256 / 32) fold_scratch[tid] else 0;
        inline for (.{ 16, 8, 4, 2, 1 }) |m| w += g.builtins.shuffleXor(f32, w, m);
        if (tid == 0) out[g.builtins.blockIdX()] = w;
    }
}

comptime {
    if (g.is_device) {
        g.exportRaw("t_raw_double", &rawDouble);
        g.exportRaw("t_raw_atomics", &rawAtomics);
        g.exportRaw("t_raw_block_sum", &rawBlockSum);
        g.exportRaw("t_math_exp_f32", &MathKernel(f32, expFamily).run);
        g.exportRaw("t_math_exp_f64", &MathKernel(f64, expFamily).run);
        g.exportRaw("t_math_log_f32", &MathKernel(f32, logFamily).run);
        g.exportRaw("t_math_log_f64", &MathKernel(f64, logFamily).run);
        g.exportRaw("t_math_trig_f32", &MathKernel(f32, trigFamily).run);
        g.exportRaw("t_math_trig_f64", &MathKernel(f64, trigFamily).run);
    }
}
