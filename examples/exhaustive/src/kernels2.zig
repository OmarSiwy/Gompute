//! A SECOND kernel root, compiled to its own artifact.
//!
//! Exists so the multi-root pipeline is exercised on every run of this example:
//! separate blob, merged name map, and a kernel reachable only by a name chosen
//! at run time. Kernel names must not collide with `kernels.zig`.

const g = @import("gompute");

pub const Params = struct { offset: f32 };

fn addOffset(x: f32, p: Params) f32 {
    return x + p.offset;
}
pub const add_offset = g.map("add_offset", f32, Params, addOffset, .{ .block_size = 256 });

fn rawTriple(data: g.GlobalPtr(f32), len: u64) callconv(g.kernel_callconv) void {
    const i = g.globalIdX(256);
    if (i < len) data[i] = data[i] * 3;
}

comptime {
    if (g.is_device) g.exportRaw("raw_triple", &rawTriple);
    g.exportKernels(@This());
}

/// Which `g.math` function `math_f64`/`math_f32` apply, for the CPU/GPU
/// bit-identity section of the exhaustive run.
pub const MathParams = struct { f: u32 };

fn mathOf(comptime T: type) fn (T, MathParams) T {
    return struct {
        fn call(x: T, p: MathParams) T {
            return switch (p.f) {
                0 => g.math.exp(x),
                1 => g.math.log(x),
                2 => g.math.exp2(x),
                3 => g.math.log2(x),
                4 => g.math.log10(x),
                5 => g.math.sin(x),
                6 => g.math.cos(x),
                7 => g.math.tan(x),
                8 => g.math.tanh(x),
                9 => g.math.sinh(x),
                10 => g.math.cosh(x),
                11 => g.math.expm1(x),
                12 => g.math.log1p(x),
                13 => g.math.atan(x),
                14 => g.math.pow(x, @as(T, 1.4552480184709202)),
                15 => g.math.sqrt(x),
                else => x,
            };
        }
    }.call;
}
pub const math_function_count = 16;
pub const math_f64 = g.map("math_f64", f64, MathParams, mathOf(f64), .{});
pub const math_f32 = g.map("math_f32", f32, MathParams, mathOf(f32), .{});

// ── Every kind, for the GPU-against-CPU sweep in main.zig ────────────────
// One spec per kind and per lane width the generated entries take (f32: 4
// per thread, f64: 2, i16: 8, u8: 16, a 16-byte vector: 1), so the vector
// path, its straddling tail and the n == 1 path all run on hardware.

fn axpy(a: f32, b: f32, p: Params) f32 {
    return a * p.offset + b;
}
fn toByte(x: f32, p: Params) u8 {
    return @intFromFloat(@min(@max(x + p.offset, 0), 255));
}
fn widen(x: f32, p: Params) f64 {
    return @as(f64, x) * 1.5 + p.offset;
}
fn stripe(x: f32, i: u64, p: Params) f32 {
    return x + @as(f32, @floatFromInt(i % 7)) * p.offset;
}
fn incByte(x: u8, _: Params) u8 {
    return x +% 3;
}
fn scaleV4(x: @Vector(4, f32), p: Params) @Vector(4, f32) {
    return x * @as(@Vector(4, f32), @splat(p.offset));
}
fn xorFold(a: u32, b: u32) u32 {
    return a ^ b;
}

pub const x_sum_f32 = g.sum("x_sum_f32", f32, Params, .{});
pub const x_max_i32 = g.max("x_max_i32", i32, Params, .{ .block_size = 100 });
pub const x_xor_u32 = g.reduce("x_xor_u32", u32, Params, xorFold, 0, .{ .block_size = 64 });
pub const x_zip_f32 = g.zip("x_zip_f32", f32, f32, f32, Params, axpy, .{});
pub const x_to_u8 = g.mapTo("x_to_u8", f32, u8, Params, toByte, .{ .block_size = 64 });
pub const x_widen = g.mapTo("x_widen", f32, f64, Params, widen, .{});
pub const x_indexed = g.mapIndexed("x_indexed", f32, Params, stripe, .{ .block_size = 128 });
pub const x_map_u8 = g.map("x_map_u8", u8, Params, incByte, .{});
pub const x_map_v4 = g.map("x_map_v4", @Vector(4, f32), Params, scaleV4, .{});
pub const x_gather = g.gather("x_gather", f32, u32, .{});
pub const x_scatter = g.scatter("x_scatter", f32, u32, .{});
