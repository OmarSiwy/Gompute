//! One spec of every `Kind`, exported for a GPU target.
//!
//! This file is never run. It exists so that `zig build test` puts all six
//! entry generators in `src/device/export.zig` through the Zig frontend for
//! `nvptx64` -- before this, only `Entry` (the `map` shape) had ever been
//! compiled for a device at all, because every kernels root in the repo and in
//! both examples uses `g.map` exclusively. `ReduceEntry`'s
//! `addrspace(.shared)` scratch array, `builtins.barrier`, `builtins.localIdX`
//! and `builtins.blockIdX` had no compiled call site anywhere.
//!
//! It also names every `g.math` entry point, which is the only device call site
//! that module has. See `allMath` for why that one matters more than the rest.

const g = @import("gompute");

const Scale = struct { k: f32 };
const NoParams = struct {};

fn scale(x: f32, p: Scale) f32 {
    return x * p.k;
}
fn scaleAt(x: f32, i: u64, p: Scale) f32 {
    return x * p.k + @as(f32, @floatFromInt(i));
}
fn toInt(x: f32, _: Scale) i32 {
    return @intFromFloat(x);
}
fn addUp(a: f32, b: f32, _: Scale) f32 {
    return a + b;
}
fn maxOf(a: f32, b: f32) f32 {
    return @max(a, b);
}
fn negate(x: f32, _: NoParams) f32 {
    return -x;
}

pub const map_entry = g.map("t_map", f32, Scale, scale, .{ .block_size = 256 });
pub const indexed_entry = g.mapIndexed("t_map_indexed", f32, Scale, scaleAt, .{ .block_size = 128 });
pub const map_to_entry = g.mapTo("t_map_to", f32, i32, Scale, toInt, .{ .block_size = 64 });
pub const zip_entry = g.zip("t_zip", f32, f32, f32, Scale, addUp, .{ .block_size = 256 });
pub const reduce_entry = g.reduce("t_reduce", f32, Scale, maxOf, -1e30, .{ .block_size = 256 });
pub const sum_entry = g.sum("t_sum", f32, Scale, .{ .block_size = 100 });
pub const gather_entry = g.gather("t_gather", f32, u32, .{ .block_size = 256 });
pub const scatter_entry = g.scatter("t_scatter", f32, u32, .{ .block_size = 256 });

/// Every `g.math` entry point in one body, at one width.
///
/// `src/device/math.zig` exists because `@exp @log @sin` and friends fail at
/// the IR->ISA stage on NVPTX and AMDGCN — `no libcall available for fexp`,
/// `Cannot select: f32 = fsin`. That is a claim about the BACK END, and it went
/// unchecked: no example, test or kernels root in this repo called `g.math`
/// from a device target, so the module's whole reason to exist had never been
/// compiled the way it is meant to be used. `build.zig` assembles this probe
/// down to real PTX for that reason, rather than stopping at the frontend.
/// Split into families rather than one body per width, and the split is load
/// bearing. Every entry point is `inline`, so one body naming all fifteen is a
/// single enormous function; at fifteen the AMDGCN backend stopped returning an
/// error and started SEGV-ing the compiler under `zig build`'s server protocol
/// (the same `zig build-obj` on its own still succeeded). Three smaller bodies
/// compile, and when one does break it names which family did it.
fn expFamily(comptime T: type) fn (T, NoParams) T {
    return struct {
        fn f(x: T, _: NoParams) T {
            return g.math.exp(x) + g.math.exp2(x) + g.math.expm1(x) + g.math.pow(x, x);
        }
    }.f;
}

fn logFamily(comptime T: type) fn (T, NoParams) T {
    return struct {
        fn f(x: T, _: NoParams) T {
            return g.math.log(x) + g.math.log2(x) + g.math.log10(x) +
                g.math.sqrt(x) + g.math.rsqrt(x);
        }
    }.f;
}

fn trigFamily(comptime T: type) fn (T, NoParams) T {
    return struct {
        fn f(x: T, _: NoParams) T {
            return g.math.sin(x) + g.math.cos(x) + g.math.tan(x) + g.math.atan(x) +
                g.math.tanh(x) + g.math.sinh(x) + g.math.cosh(x);
        }
    }.f;
}

pub const math_exp_f32 = g.map("t_math_exp_f32", f32, NoParams, expFamily(f32), .{ .block_size = 256 });
pub const math_exp_f64 = g.map("t_math_exp_f64", f64, NoParams, expFamily(f64), .{ .block_size = 256 });
pub const math_log_f32 = g.map("t_math_log_f32", f32, NoParams, logFamily(f32), .{ .block_size = 256 });
pub const math_log_f64 = g.map("t_math_log_f64", f64, NoParams, logFamily(f64), .{ .block_size = 256 });
pub const math_trig_f32 = g.map("t_math_trig_f32", f32, NoParams, trigFamily(f32), .{ .block_size = 256 });
pub const math_trig_f64 = g.map("t_math_trig_f64", f64, NoParams, trigFamily(f64), .{ .block_size = 256 });

/// `abi.Boundary(struct{})` is a zero-field extern struct, so the emitted entry
/// takes one fewer parameter than the host pushes. Compiled here so the shape
/// at least exists in an artifact; see the launch-frame note in `exportOne`.
pub const empty_params_entry = g.map("t_no_params", f32, NoParams, negate, .{ .block_size = 256 });

comptime {
    g.exportKernels(@This());
}

// The device half of the host/device mirror: `GlobalPtr`, `kernel_callconv`,
// `globalIdX` and `exportRaw` all resolve to something different in
// `src/root.zig`, and only a device compilation sees these versions.
fn rawDouble(data: g.GlobalPtr(f32), len: u64) callconv(g.kernel_callconv) void {
    const i = g.globalIdX(256);
    if (i >= len) return;
    data[i] *= 2;
}

comptime {
    if (g.is_device) g.exportRaw("t_raw_double", &rawDouble);
}
