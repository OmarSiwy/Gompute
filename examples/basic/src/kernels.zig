const g = @import("gompute");

/// The element function: plain Zig, so the CPU calls it directly too.
pub fn scaleRelu(x: f32, scale: f32) f32 {
    return @max(x * scale, 0);
}

fn scaleReluKernel(data: g.GlobalPtr(f32), len: u64, scale: f32) callconv(g.kernel_callconv) void {
    const i = g.globalIdX(256);
    if (i < len) data[i] = scaleRelu(data[i], scale);
}

comptime {
    if (g.is_device) g.exportRaw("scale_relu", &scaleReluKernel);
}
