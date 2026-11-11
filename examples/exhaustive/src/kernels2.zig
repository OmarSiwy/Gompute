//! A SECOND kernel root, compiled to its own artifact.
//!
//! Exists so the multi-root pipeline is exercised on every run of this example:
//! separate blob, merged name map, and a kernel reachable only by a name chosen
//! at run time. Kernel names must not collide with `kernels.zig`.

const g = @import("gompute");

fn rawAddOffset(data: g.GlobalPtr(f32), len: u64, offset: f32) callconv(g.kernel_callconv) void {
    const i = g.globalIdX(256);
    if (i < len) data[i] += offset;
}

fn rawTriple(data: g.GlobalPtr(f32), len: u64) callconv(g.kernel_callconv) void {
    const i = g.globalIdX(256);
    if (i < len) data[i] = data[i] * 3;
}

/// `g.math` function number `f` at `x`: the body both the `math_f64`/`math_f32`
/// kernels and the CPU side of the bit-identity sections run.
pub fn mathOf(comptime T: type) fn (T, u32) T {
    return struct {
        fn call(x: T, f: u32) T {
            return switch (f) {
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
                16 => g.math.asinh(x),
                17 => g.math.acosh(x),
                18 => g.math.atanh(x),
                19 => g.math.atan2(x, @as(T, -0.7)),
                else => x,
            };
        }
    }.call;
}
pub const math_function_count = 20;
/// The `mathOf` indices whose f32 result is correctly rounded, so the GPU
/// body may differ from the CPU one and must still match it on all 2^32 inputs.
pub const math_cr32 = [_]u32{ 0, 1, 3, 4, 5, 6 };
fn MathKernel(comptime T: type) type {
    return struct {
        fn run(data: g.GlobalPtr(T), len: u64, f: u32) callconv(g.kernel_callconv) void {
            const i = g.globalIdX(256);
            if (i < len) data[i] = mathOf(T)(data[i], f);
        }
    };
}

comptime {
    if (g.is_device) {
        g.exportRaw("raw_add_offset", &rawAddOffset);
        g.exportRaw("raw_triple", &rawTriple);
        g.exportRaw("math_f64", &MathKernel(f64).run);
        g.exportRaw("math_f32", &MathKernel(f32).run);
    }
}
