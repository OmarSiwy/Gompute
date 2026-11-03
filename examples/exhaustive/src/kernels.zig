const g = @import("gompute");

fn rawIncrement(data: g.GlobalPtr(f32), len: u64) callconv(g.kernel_callconv) void {
    const i = g.globalIdX(256);
    if (i < len) data[i] += 1;
}

fn rawScale(data: g.GlobalPtr(f32), len: u64, scale: f32) callconv(g.kernel_callconv) void {
    const i = g.globalIdX(256);
    if (i < len) data[i] *= scale;
}

comptime {
    if (g.is_device) {
        g.exportRaw("raw_increment", &rawIncrement);
        g.exportRaw("raw_scale", &rawScale);
    }
}
