//! The kernels OVERHEAD.md is measured on. Every elementwise body is the
//! cheapest thing that still touches its bytes, so the numbers are bandwidth
//! and overhead, not arithmetic.
const std = @import("std");
const g = @import("gompute");

pub const P = struct { scale: f32 };
pub const P64 = struct { scale: f64 };

fn scaleRelu(x: f32, p: P) f32 {
    const y = x * p.scale;
    return if (y > 0) y else 0;
}
/// The same body written generically: the CPU path runs it at vector width.
fn scaleReluV(x: anytype, p: P) @TypeOf(x) {
    const V = @TypeOf(x);
    return @max(x * g.splat(V, p.scale), g.splat(V, @as(f32, 0)));
}
fn axpy(a: f32, b: f32, p: P) f32 {
    return a * p.scale + b;
}
fn axpyV(a: anytype, b: anytype, p: P) @TypeOf(a) {
    return a * g.splat(@TypeOf(a), p.scale) + b;
}
fn scale64(x: f64, p: P64) f64 {
    return x * p.scale;
}
fn sat16(x: i16, _: P) i16 {
    return if (x > 100) 100 else if (x < -100) -100 else x;
}
fn expf(x: f32, p: P) f32 {
    return g.math.exp(x * p.scale);
}

pub const map_f32 = g.map("b_map_f32", f32, P, scaleRelu, .{});
pub const map_f32_v = g.map("b_map_f32_v", f32, P, scaleReluV, .{});
pub const zip_f32 = g.zip("b_zip_f32", f32, f32, f32, P, axpy, .{});
pub const zip_f32_v = g.zip("b_zip_f32_v", f32, f32, f32, P, axpyV, .{});
pub const sum_f32 = g.sum("b_sum_f32", f32, P, .{});
pub const map_f64 = g.map("b_map_f64", f64, P64, scale64, .{});
pub const map_i16 = g.map("b_map_i16", i16, P, sat16, .{});
pub const exp_f32 = g.map("b_exp_f32", f32, P, expf, .{});

comptime {
    g.exportKernels(@This());
}

// ── The 1.4.0 entry shape, one element per thread, as raw kernels ───────
// `main.zig` alternates these with the generated 16-byte-lane entries in one
// process, so a clock-drifting laptop GPU cannot bias the comparison.

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

fn mathCall(comptime name: []const u8, x: anytype) @TypeOf(x) {
    if (comptime std.mem.eql(u8, name, "pow")) return g.math.pow(x, g.splat(@TypeOf(x), 1.4552480184709202));
    return @field(g.math, name)(x);
}

fn MathBody(comptime T: type, comptime name: []const u8) type {
    return struct {
        fn scalar(x: T, _: P) T {
            var acc: T = 0;
            inline for (0..8) |i| acc += mathCall(name, x + @as(T, 0.25 * @as(comptime_float, i)));
            return acc;
        }
        fn vector(x: anytype, _: P) @TypeOf(x) {
            const V = @TypeOf(x);
            var acc = g.splat(V, @as(T, 0));
            inline for (0..8) |i| acc += mathCall(name, x + g.splat(V, @as(T, 0.25 * @as(comptime_float, i))));
            return acc;
        }
    };
}

fn mathKernels(comptime T: type, comptime vector: bool) [math_fns.len]type {
    var out: [math_fns.len]type = undefined;
    for (math_fns, 0..) |name, i| {
        const B = MathBody(T, name);
        const kname = "bm_" ++ name ++ "_" ++ @typeName(T) ++ (if (vector) "_v" else "");
        out[i] = if (vector) g.map(kname, T, P, B.vector, .{}) else g.map(kname, T, P, B.scalar, .{});
    }
    return out;
}

pub const math_f32 = mathKernels(f32, false);
pub const math_f64 = mathKernels(f64, false);
pub const math_f32_v = mathKernels(f32, true);
pub const math_f64_v = mathKernels(f64, true);

comptime {
    g.exportKernels(math_f32);
    g.exportKernels(math_f64);
}
