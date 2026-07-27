const std = @import("std");
const g = @import("gompute");
const kernels = @import("kernels");

pub fn main() !void {
    var data = [_]f32{ -1, 2, -3, 4 };
    var kernel = try g.Kernel(kernels.scale_relu, .cpu).init(0);
    try kernel.run(&data, .{ .scale = 2 });
    std.debug.print("{any}\n", .{data});
}

/// Kept exported so both fixed GPU paths are fully compiled whenever this build
/// actually emitted them.
///
/// `Kernel(spec, .cuda)` is a compile error when the build emitted no CUDA, so
/// a probe that wants to compile on any machine has to ask first. `.available`
/// is that question; `AutoKernel` asks it for you.
export fn gompute_gpu_compile_probe() void {
    const Auto = g.AutoKernel(kernels.scale_relu);
    var one = [_]f32{1};

    if (comptime Auto.Cuda.available) {
        var cuda_kernel = g.Kernel(kernels.scale_relu, .cuda).init(0) catch return;
        defer cuda_kernel.deinit();
        cuda_kernel.run(&one, .{ .scale = 2 }) catch return;

        var raw_cuda = g.RawKernel("raw_add", .cuda).init(0) catch return;
        defer raw_cuda.deinit();
    }

    if (comptime Auto.Hip.available) {
        var hip_kernel = g.Kernel(kernels.scale_relu, .hip).init(0) catch return;
        defer hip_kernel.deinit();
        hip_kernel.run(&one, .{ .scale = 2 }) catch return;

        var raw_hip = g.RawKernel("raw_add", .hip).init(0) catch return;
        defer raw_hip.deinit();
    }
}
