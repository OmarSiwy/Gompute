//! The kernels the bench measures. The elementwise bodies are the cheapest
//! thing that still touches their bytes, so the numbers are bandwidth and
//! overhead, not arithmetic.
const std = @import("std");
const g = @import("gompute");

fn rawMapF32(data: g.GlobalPtr(f32), len: u64, scale: f32) callconv(g.kernel_callconv) void {
    const i = g.globalIdX(256);
    if (i >= len) return;
    const y = data[i] * scale;
    data[i] = if (y > 0) y else 0;
}
fn rawMapF64(data: g.GlobalPtr(f64), len: u64, scale: f64) callconv(g.kernel_callconv) void {
    const i = g.globalIdX(256);
    if (i >= len) return;
    data[i] = data[i] * scale;
}
fn rawZipF32(a: g.GlobalPtr(f32), b: g.GlobalPtr(f32), out: g.GlobalPtr(f32), len: u64, scale: f32) callconv(g.kernel_callconv) void {
    const i = g.globalIdX(256);
    if (i >= len) return;
    out[i] = a[i] * scale + b[i];
}

comptime {
    if (g.is_device) {
        g.exportRaw("raw_map_f32", &rawMapF32);
        g.exportRaw("raw_map_f64", &rawMapF64);
        g.exportRaw("raw_zip_f32", &rawZipF32);
    }
}

// ── g.math throughput: 8 calls per element, so the arithmetic, not the
// bytes, is what a launch measures. Inputs live in [0.1, 4) and each call
// adds 0.25*i, so every function stays on its ordinary path.

pub const math_fns = .{ "exp", "exp2", "expm1", "log", "log2", "log10", "log1p", "sin", "cos", "tan", "tanh", "sinh", "cosh", "atan", "pow" };

/// `v` at `X`'s width: `X` itself for a scalar, every lane for a vector.
fn splat(comptime X: type, v: anytype) X {
    return if (@typeInfo(X) == .vector) @splat(v) else v;
}

fn mathCall(comptime name: []const u8, x: anytype) @TypeOf(x) {
    if (comptime std.mem.eql(u8, name, "pow")) return g.math.pow(x, splat(@TypeOf(x), 1.4552480184709202));
    return @field(g.math, name)(x);
}

/// Eight calls of `name` at `x`; `x` is a scalar or a vector.
pub fn mathBody(comptime name: []const u8, x: anytype) @TypeOf(x) {
    const X = @TypeOf(x);
    var acc = splat(X, 0);
    inline for (0..8) |i| acc += mathCall(name, x + splat(X, 0.25 * @as(comptime_float, i)));
    return acc;
}

fn MathKernel(comptime T: type, comptime name: []const u8) type {
    return struct {
        fn run(data: g.GlobalPtr(T), len: u64) callconv(g.kernel_callconv) void {
            const i = g.globalIdX(256);
            if (i < len) data[i] = mathBody(name, data[i]);
        }
    };
}

comptime {
    if (g.is_device) for (math_fns) |name| for (.{ f32, f64 }) |T|
        g.exportRaw("bm_" ++ name ++ "_" ++ @typeName(T), &MathKernel(T, name).run);
}
