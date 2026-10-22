const std = @import("std");
const gompute_build = @import("gompute");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const dep = b.dependency("gompute", .{ .target = target, .optimize = optimize });

    const kernels_mod = b.createModule(.{
        .root_source_file = b.path("src/kernels.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "gompute", .module = dep.module("gompute") }},
    });
    const exe = b.addExecutable(.{
        .name = "example",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "gompute", .module = dep.module("gompute") },
                .{ .name = "kernels", .module = kernels_mod },
            },
        }),
    });

    exe.root_module.linkSystemLibrary("c", .{});

    // Null: SASS exactly when ptxas and fatbinary are on PATH.
    const sass = b.option(bool, "sass", "Embed SASS for the CUDA target beside the PTX (default: when ptxas and fatbinary are on PATH)");
    gompute_build.emitKernels(b, dep, exe, .{
        .kernels_root = b.path("src/kernels.zig"),
        .target = target,
        .optimize = optimize,
        .cuda = .{ .sass = sass },
    });
    b.installArtifact(exe);

    const run = b.addRunArtifact(exe);
    const run_step = b.step("run", "Run CPU example");
    run_step.dependOn(&run.step);
}
