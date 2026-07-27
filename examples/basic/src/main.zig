const std = @import("std");
const g = @import("gompute");
const kernels = @import("kernels");

pub fn main() !void {
    var data = [_]f32{ -1, 2, -3, 4 };

    // The backend is part of the type. This is a direct inlined CPU loop.
    var kernel = try g.Kernel(kernels.scale_relu, .cpu).init(0);
    defer kernel.deinit();

    try kernel.run(&data, .{ .scale = 2 });
    std.debug.print("{any}\n", .{data});
}
