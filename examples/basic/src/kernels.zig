const g = @import("gompute");

pub const Params = struct {
    scale: f32,
};

fn scaleRelu(x: f32, p: Params) f32 {
    const y = x * p.scale;
    return if (y > 0) y else 0;
}

pub const scale_relu = g.map("scale_relu", f32, Params, scaleRelu, .{ .block_size = 256 });

comptime {
    // Every pub map spec in this file; no second list to keep in sync.
    g.exportKernels(@This());
}
