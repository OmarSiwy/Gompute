const g = @import("gompute");

const Params = struct { scale: f32 };
fn op(x: f32, p: Params) f32 {
    const y = x * p.scale;
    return if (y > 0) y else 0;
}
const Spec = g.map("codegen_scale_relu", f32, Params, op, .{});

export fn gompute_scale_relu(data: [*]f32, len: usize, scale: f32) void {
    var kernel = g.Kernel(Spec, .cpu).init(0) catch unreachable;
    kernel.run(data[0..len], .{ .scale = scale }) catch unreachable;
}

export fn manual_scale_relu(data: [*]f32, len: usize, scale: f32) void {
    for (data[0..len]) |*value| {
        const y = value.* * scale;
        value.* = if (y > 0) y else 0;
    }
}

/// Same operation written generically: the CPU backend runs it at vector width.
fn opGeneric(x: anytype, p: Params) @TypeOf(x) {
    const V = @TypeOf(x);
    return @max(x * g.splat(V, p.scale), g.splat(V, @as(f32, 0)));
}
const SimdSpec = g.map("codegen_scale_relu_simd", f32, Params, opGeneric, .{});

export fn gompute_scale_relu_simd(data: [*]f32, len: usize, scale: f32) void {
    var kernel = g.Kernel(SimdSpec, .cpu).init(0) catch unreachable;
    kernel.run(data[0..len], .{ .scale = scale }) catch unreachable;
}

/// Inferring constructor: T and Params come from `op`'s signature.
const InferredSpec = g.mapFn("codegen_inferred", op, .{ .block_size = 128 });

export fn gompute_inferred(data: [*]f32, len: usize, scale: f32) void {
    var kernel = g.Kernel(InferredSpec, .cpu).init(0) catch unreachable;
    kernel.run(data[0..len], .{ .scale = scale }) catch unreachable;
}
