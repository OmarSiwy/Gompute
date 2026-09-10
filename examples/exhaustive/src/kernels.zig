const g = @import("gompute");

// ── 1. Various param struct shapes ──────────────────────────────────────

pub const ScalarParams = struct { scale: f32 };
pub const MultiParams = struct { scale: f32, bias: f32, enabled: bool };
pub const IntParams = struct { shift: u32, mask: u32 };
pub const ArrayParams = struct { weights: [4]f32 };
pub const NestedParams = struct { inner: ScalarParams, offset: f32 };

// ── 2. Map kernels: f32 ─────────────────────────────────────────────────

fn scaleRelu(x: f32, p: ScalarParams) f32 {
    const y = x * p.scale;
    return if (y > 0) y else 0;
}
pub const scale_relu = g.map("scale_relu", f32, ScalarParams, scaleRelu, .{ .block_size = 256 });

fn affine(x: f32, p: MultiParams) f32 {
    const y = x * p.scale + p.bias;
    return y;
}
pub const affine_transform = g.map("affine_transform", f32, MultiParams, affine, .{ .block_size = 128 });

fn clamp01(x: f32, _: ScalarParams) f32 {
    return @max(0.0, @min(1.0, x));
}
pub const clamp = g.map("clamp01", f32, ScalarParams, clamp01, .{ .block_size = 256 });

fn negate(x: f32, _: ScalarParams) f32 {
    return -x;
}
pub const neg = g.map("neg", f32, ScalarParams, negate, .{ .block_size = 512 });

fn square(x: f32, _: ScalarParams) f32 {
    return x * x;
}
pub const sq = g.map("sq", f32, ScalarParams, square, .{ .block_size = 64 });

fn weightedSum(x: f32, p: ArrayParams) f32 {
    return x * p.weights[0] + p.weights[1] * p.weights[2] + p.weights[3];
}
pub const weighted = g.map("weighted", f32, ArrayParams, weightedSum, .{ .block_size = 256 });

fn nestedOp(x: f32, p: NestedParams) f32 {
    return x * p.inner.scale + p.offset;
}
pub const nested = g.map("nested", f32, NestedParams, nestedOp, .{ .block_size = 256 });

// ── 3. Map kernels: other types ─────────────────────────────────────────

fn doubleF64(x: f64, _: ScalarParams) f64 {
    return x * 2.0;
}
pub const double_f64 = g.map("double_f64", f64, ScalarParams, doubleF64, .{ .block_size = 256 });

fn shiftMask(x: u32, p: IntParams) u32 {
    return (x >> @intCast(p.shift)) & p.mask;
}
pub const shift_mask = g.map("shift_mask", u32, IntParams, shiftMask, .{ .block_size = 256 });

fn saturateI16(x: i16, _: ScalarParams) i16 {
    return if (x > 100) 100 else if (x < -100) -100 else x;
}
pub const saturate_i16 = g.map("saturate_i16", i16, ScalarParams, saturateI16, .{ .block_size = 256 });

// ── 4. Fused pipelines ──────────────────────────────────────────────────

const ScaleOp = struct {
    pub inline fn eval(x: f32, p: ScalarParams) f32 {
        return x * p.scale;
    }
};
const SquareOp = struct {
    pub inline fn eval(x: f32, _: ScalarParams) f32 {
        return x * x;
    }
};
const AbsOp = struct {
    pub inline fn eval(x: f32, _: ScalarParams) f32 {
        return @abs(x);
    }
};
const ReluOp = struct {
    pub inline fn eval(x: f32, _: ScalarParams) f32 {
        return @max(0.0, x);
    }
};

pub const ScaleSquare = g.Fused(f32, ScalarParams, .{ ScaleOp, SquareOp });
pub const AbsRelu = g.Fused(f32, ScalarParams, .{ AbsOp, ReluOp });
pub const ScaleAbsSquare = g.Fused(f32, ScalarParams, .{ ScaleOp, AbsOp, SquareOp });

fn scaleSquareFn(x: f32, p: ScalarParams) f32 {
    return ScaleSquare.eval(x, p);
}
pub const scale_square = g.map("scale_square", f32, ScalarParams, scaleSquareFn, .{ .block_size = 256 });

fn tripleChain(x: f32, p: ScalarParams) f32 {
    return ScaleAbsSquare.eval(x, p);
}
pub const triple_fused = g.map("triple_fused", f32, ScalarParams, tripleChain, .{ .block_size = 256 });

// ── 5. Unary wrapper ────────────────────────────────────────────────────

fn absFunc(x: f32, _: ScalarParams) f32 {
    return @abs(x);
}
const AbsUnary = g.Unary(f32, ScalarParams, absFunc);
pub const AbsThenScale = g.Fused(f32, ScalarParams, .{ AbsUnary, ScaleOp });

fn absThenScaleFn(x: f32, p: ScalarParams) f32 {
    return AbsThenScale.eval(x, p);
}
pub const abs_then_scale = g.map("abs_then_scale", f32, ScalarParams, absThenScaleFn, .{ .block_size = 256 });

// ── 6. Raw kernel ───────────────────────────────────────────────────────

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
    // Every pub map spec in this file; no second list to keep in sync. The
    // pub Fused/Unary types and the param structs have no `entry_name`, so
    // they are not specs and are skipped.
    g.exportKernels(@This());
}
