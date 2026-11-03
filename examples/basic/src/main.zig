const std = @import("std");
const g = @import("gompute");
const kernels = @import("kernels");

pub fn main() !void {
    var data = [_]f32{ -1, 2, -3, 4 };
    var scale: f32 = 2;

    // `available` is comptime: without CUDA artifacts, `RawKernel(.., .cuda)`
    // is not even compiled. Without a card at run time, `init` fails.
    if (comptime g.available(.cuda)) gpu: {
        var kernel = g.RawKernel("scale_relu", .cuda).init(0) catch break :gpu;
        defer kernel.deinit();
        var buf = try kernel.alloc(@sizeOf(@TypeOf(data)));
        defer buf.free();
        try buf.upload(&data, @sizeOf(@TypeOf(data)));
        var len: u64 = data.len;
        try kernel.launch(g.Dim3.linear(data.len, 256), .{ .x = 256 }, 0, &.{ buf.argPtr(), g.interface.arg(&len), g.interface.arg(&scale) });
        try kernel.synchronize();
        try buf.download(&data, @sizeOf(@TypeOf(data)));
        std.debug.print("cuda: {any}\n", .{data});
        return;
    }
    // The CPU path is a plain loop; LLVM vectorizes it.
    for (&data) |*x| x.* = kernels.scaleRelu(x.*, scale);
    std.debug.print("cpu: {any}\n", .{data});
}
