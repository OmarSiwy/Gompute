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
        .name = "exhaustive",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "gompute", .module = dep.module("gompute") },
                .{ .name = "kernels", .module = kernels_mod },
                .{ .name = "kernels2", .module = b.createModule(.{
                    .root_source_file = b.path("src/kernels2.zig"),
                    .target = target,
                    .optimize = optimize,
                    .imports = &.{.{ .name = "gompute", .module = dep.module("gompute") }},
                }) },
            },
        }),
    });
    exe.root_module.linkSystemLibrary("c", .{});

    // Two roots: separately compiled, separately cached, merged into one name
    // map. `.kernels_root` is implicitly the root named "kernels".
    gompute_build.emitKernels(b, dep, exe, .{
        .kernels_root = b.path("src/kernels.zig"),
        .kernel_roots = &.{
            .{ .name = "extra", .root = b.path("src/kernels2.zig") },
        },
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(exe);

    const run = b.addRunArtifact(exe);
    const run_step = b.step("run", "Run exhaustive test");
    run_step.dependOn(&run.step);
}
