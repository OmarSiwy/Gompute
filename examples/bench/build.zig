//! The source of the numbers in skill/gompute/OVERHEAD.md: `zig build run`
//! on a machine with a GPU, with `-Doptimize=ReleaseFast`; `-Dcuda-arch=sm_89`
//! pins the device target the way a release build should.
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
        .name = "bench",
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

    const cuda_arch = b.option([]const u8, "cuda-arch", "CUDA device target, e.g. sm_89 (default: probe this machine)");
    const hip_arch = b.option([]const u8, "hip-arch", "HIP device target, e.g. gfx1100 (default: probe this machine)");
    gompute_build.emitKernels(b, dep, exe, .{
        .kernels_root = b.path("src/kernels.zig"),
        .target = target,
        .optimize = optimize,
        .cuda = .{ .gpu = if (cuda_arch) |a| .{ .name = a } else .auto },
        .hip = .{ .gpu = if (hip_arch) |a| .{ .name = a } else .auto },
    });
    b.installArtifact(exe);

    const run = b.addRunArtifact(exe);
    run.step.dependOn(b.getInstallStep());
    b.step("run", "Run the benchmark").dependOn(&run.step);
}
