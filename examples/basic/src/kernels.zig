const g = @import("gompute");

pub const Params = struct {
    scale: f32,
};

fn scaleRelu(x: f32, p: Params) f32 {
    const y = x * p.scale;
    return if (y > 0) y else 0;
}

pub const scale_relu = g.map("scale_relu", f32, Params, scaleRelu, .{ .block_size = 256 });

const Scale = struct {
    pub inline fn eval(x: f32, p: Params) f32 {
        return x * p.scale;
    }
};
const Square = struct {
    pub inline fn eval(x: f32, _: Params) f32 {
        return x * x;
    }
};
const ScaleSquare = g.Fused(f32, Params, .{ Scale, Square });

fn scaleSquare(x: f32, p: Params) f32 {
    return ScaleSquare.eval(x, p);
}

pub const scale_square = g.map("scale_square", f32, Params, scaleSquare, .{ .block_size = 256 });

fn rawAdd(data: g.GlobalPtr(f32), len: u64) callconv(g.kernel_callconv) void {
    const i = g.globalIdX(256);
    if (i < len) data[i] += 1;
}

comptime {
    if (g.is_device) g.exportRaw("raw_add", &rawAdd);
    // Every pub map spec in this file; no second list to keep in sync.
    g.exportKernels(@This());
}
