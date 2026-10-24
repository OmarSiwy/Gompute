//! The kernels OVERHEAD.md is measured on. Every elementwise body is the
//! cheapest thing that still touches its bytes, so the numbers are bandwidth
//! and overhead, not arithmetic.
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
