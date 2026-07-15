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
