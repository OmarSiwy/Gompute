const std = @import("std");
const g = @import("gompute");
const kernels = @import("kernels");

pub fn main() !void {
    var data = [_]f32{ -1, 2, -3, 4 };
    var kernel = try g.Kernel(kernels.scale_relu, .cpu).init(0);
    try kernel.run(&data, .{ .scale = 2 });
    std.debug.print("{any}\n", .{data});
}

/// Kept exported so both fixed GPU paths are fully compiled without requiring
/// a GPU on the build machine.
export fn gompute_gpu_compile_probe() void {
    var cuda_kernel = g.Kernel(kernels.scale_relu, .cuda).init(0) catch return;
    defer cuda_kernel.deinit();
    var one = [_]f32{1};
    cuda_kernel.run(&one, .{ .scale = 2 }) catch return;

    var hip_kernel = g.Kernel(kernels.scale_relu, .hip).init(0) catch return;
    defer hip_kernel.deinit();
    hip_kernel.run(&one, .{ .scale = 2 }) catch return;

    var raw_cuda = g.RawKernel("raw_add", .cuda).init(0) catch return;
    defer raw_cuda.deinit();
    var raw_hip = g.RawKernel("raw_add", .hip).init(0) catch return;
    defer raw_hip.deinit();
}
