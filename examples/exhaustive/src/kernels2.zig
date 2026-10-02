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
