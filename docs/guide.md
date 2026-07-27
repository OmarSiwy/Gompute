# Guide

Gompute is a Zig **0.16.0** library for defining a one-dimensional GPU/CPU
kernel once and specializing the whole abstraction at compile time.

## Dependencies

- **LLVM** — supplied by Zig itself; the PTX/HSACO steps call `zig cc` and
  `zig ld.lld`. Nothing to install.
- **libc, linked into your executable** — `exe.root_module.linkSystemLibrary("c", .{})`
  is **required for the CUDA and HIP backends**. Without libc, Zig 0.16's
  `std.DynLib` resolves to `ElfDynLib` instead of `DlDynLib`; it opens
  `libcuda.so` but cannot resolve symbols out of it, and you get
  `cuda: symbol not found: cuInit` at run time, which looks exactly like a
  driver mismatch and is not one. CPU-only builds do not need it.
- **A CUDA or HIP driver at run time**, if you use those backends. Both are
  `dlopen`'d; neither is needed to build.

## 1. Add the dependency

```sh
zig fetch --save git+https://github.com/OmarSiwy/Gompute.git
```

## 2. Define shared kernels

```zig
// src/kernels.zig
const g = @import("gompute");

pub const Params = struct {
    scale: f32,
};

fn scaleRelu(x: f32, p: Params) f32 {
    const y = x * p.scale;
    return if (y > 0) y else 0;
}

pub const scale_relu = g.map(
    "scale_relu",
    f32,
    Params,
    scaleRelu,
    .{ .block_size = 256 },
);

comptime {
    g.exportKernels(@This());
}
```

`kernels.zig` is compiled twice: once as an ordinary host module and once for
each requested GPU target. The public source is shared; target-specific
address spaces and thread indexing stay inside Gompute's generated entry point.

`g.exportKernels(@This())` exports every public map spec in the file. Pass a
tuple instead — `g.exportKernels(.{scale_relu})` — if you want an explicit
subset.

## 3. Add one build call

```zig
// build.zig
const std = @import("std");
const gompute_build = @import("gompute");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const dep = b.dependency("gompute", .{
        .target = target,
        .optimize = optimize,
    });

    const kernels_mod = b.createModule(.{
        .root_source_file = b.path("src/kernels.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "gompute", .module = dep.module("gompute") },
        },
    });

    const exe = b.addExecutable(.{
        .name = "app",
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

    // Required for the CUDA/HIP backends. Without libc, std.DynLib is
    // ElfDynLib, which opens libcuda.so but cannot resolve symbols from it:
    // you get "cuda: symbol not found: cuInit" at run time. Harmless if you
    // only ever use .cpu.
    exe.root_module.linkSystemLibrary("c", .{});

    gompute_build.emitKernels(b, dep, exe, .{
        .kernels_root = b.path("src/kernels.zig"),
        .target = target,
        .optimize = optimize,
        // .gpu defaults to .auto: the build probes the GPU in THIS machine and
        // disables a backend when its GPU is absent. Pin it for CI, Docker,
        // Nix and releases -- see the build integration page.
        // .cuda = .{ .gpu = .{ .name = "sm_89" } },
        // .hip = .{ .gpu = .{ .name = "gfx1100" } },
    });

    b.installArtifact(exe);
}
```

## 4. Select a backend

```zig
const g = @import("gompute");
const kernels = @import("kernels");

pub fn main() !void {
    var data = [_]f32{ -1, 2, -3, 4 };

    // The backend is part of the type. This is a direct inlined CPU loop.
    var kernel = try g.Kernel(kernels.scale_relu, .cpu).init(0);
    defer kernel.deinit();

    try kernel.run(&data, .{ .scale = 2 });
    // data == .{ 0, 4, 0, 8 }
}
```

Change `.cpu` to `.cuda` or `.hip` without changing the call site:

```zig
var kernel = try g.Kernel(kernels.scale_relu, .cuda).init(0);
defer kernel.deinit();
try kernel.run(&data, .{ .scale = 2 });
```

For runtime probing and CPU fallback:

```zig
var kernel = g.AutoKernel(kernels.scale_relu).init();
defer kernel.deinit();
try kernel.run(&data, .{ .scale = 2 });
```

`AutoKernel` probes CUDA, then HIP, once during initialization. Its `run` method
has one tagged-union switch. Use a fixed backend when zero runtime dispatch is
the requirement.

`AutoKernel` distinguishes *no GPU here* from *this build is broken*. A missing
driver falls back to the CPU silently, as it should. But if the artifact is in
the binary and still fails to start — wrong device arch, a kernel missing from
`exportKernels` — it warns, because that is a ~100x slowdown caused by a build
mistake rather than by hardware. Use `initStrict()` to make that fatal, and
`kernel.selected()` to assert which backend you actually got.

## CPU-only builds

Turn both GPU backends off to drop the PTX/HSACO sub-compilations from the
graph entirely. This is the supported way to force CPU-only, and it also
silences the `.auto` detection warning:

```zig
gompute_build.emitKernels(b, dep, exe, .{
    .kernels_root = b.path("src/kernels.zig"),
    .cuda = .{ .enabled = false },
    .hip = .{ .enabled = false },
});
```

`Kernel(spec, .cpu)` and `AutoKernel` keep working; `Kernel(spec, .cuda)`
becomes a compile error naming both causes and both fixes.

## Commands

```sh
# Library tests
zig build test

# Documentation site -> zig-out/docs (add -Dopen to view it in a browser)
zig build docs
zig build docs -Dopen

# Basic example (CPU path)
cd examples/basic
zig build run

# Exhaustive example (GPU — requires libc + CUDA/HIP driver)
cd examples/exhaustive
zig build run
```

With Nix:

```sh
nix build        # generated API docs
nix flake check  # zig build test
nix develop      # Zig + ROCm/CUDA library paths
```
