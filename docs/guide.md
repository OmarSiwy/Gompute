# Guide

Gompute compiles GPU kernels you write in Zig to PTX (CUDA) and HSACO (HIP) at
build time, embeds them in your executable, and launches them through a thin
runtime that loads the driver at run time. It pins one Zig build,
0.18.0-dev.349+d924f83bf at the moment; see [Versions](reference.html#versions).

## Dependencies

LLVM comes from Zig itself: PTX is Zig's own output and HSACO is `zig ld.lld`,
so there is nothing to install.

The CUDA and HIP backends do need libc linked into your executable, via
`exe.root_module.linkSystemLibrary("c", .{})`. Without libc, Zig's
`std.DynLib` resolves to `ElfDynLib` instead of `DlDynLib`; that opens
`libcuda.so` but cannot resolve symbols out of it, and you get
`cuda: symbol not found: cuInit` at run time. The message looks exactly like a
driver mismatch and is not one. CPU-only builds do not need libc.

Those two backends also need a CUDA or HIP driver at run time. Both are
`dlopen`'d, so neither is needed to build.

## 1. Add the dependency

```sh
zig fetch --save git+https://github.com/OmarSiwy/Gompute.git
```

## 2. Write a kernel

```zig
// src/kernels.zig
const g = @import("gompute");

/// The element function: plain Zig, so the CPU calls it directly too.
pub fn scaleRelu(x: f32, scale: f32) f32 {
    return @max(x * scale, 0);
}

fn scaleReluKernel(data: g.GlobalPtr(f32), len: u64, scale: f32) callconv(g.kernel_callconv) void {
    const i = g.globalIdX(256);
    if (i < len) data[i] = scaleRelu(data[i], scale);
}

comptime {
    if (g.is_device) g.exportRaw("scale_relu", &scaleReluKernel);
}
```

`kernels.zig` is compiled twice: once as an ordinary host module, against
`src/root.zig`, and once per GPU target, against `src/device.zig`. Five names
change between the two so one file parses for both: `is_device`,
`kernel_callconv` (`.auto` on the host, `.kernel` on the device), `GlobalPtr`
(`[*]T`, or a pointer into the global address space), `globalIdX` and
`exportRaw` (a no-op on the host).

The kernel function is only referenced under `if (g.is_device)`, so the host
compilation never analyzes it. `globalIdX` is a compile error in host code,
which is what you get if host code names the kernel by mistake.

`exportRaw` takes a pointer to the function, `&scaleReluKernel`. The name it
exports under is the one the host loads it by.

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
    // you get "cuda: symbol not found: cuInit" at run time. Harmless in a
    // CPU-only build.
    exe.root_module.linkSystemLibrary("c", .{});

    gompute_build.emitKernels(b, dep, exe, .{
        .kernels_root = b.path("src/kernels.zig"),
        .target = target,
        .optimize = optimize,
        // .gpu defaults to .auto: the build probes the GPU in THIS machine and
        // disables a backend when its GPU is absent. Pin it for CI, Docker,
        // Nix and releases; see the build integration page.
        // .cuda = .{ .gpu = .{ .name = "sm_89" } },
        // .hip = .{ .gpu = .{ .name = "gfx1100" } },
    });

    b.installArtifact(exe);
}
```

## 4. Launch it, or run the loop

```zig
const std = @import("std");
const g = @import("gompute");
const kernels = @import("kernels");

pub fn main() !void {
    var data = [_]f32{ -1, 2, -3, 4 };
    var scale: f32 = 2;

    if (comptime g.available(.cuda)) gpu: {
        var kernel = g.RawKernel("scale_relu", .cuda).init(0) catch break :gpu;
        defer kernel.deinit();
        var buf = try kernel.alloc(@sizeOf(@TypeOf(data)));
        defer buf.free();
        try buf.upload(&data, @sizeOf(@TypeOf(data)));
        var len: u64 = data.len;
        try kernel.launch(g.Dim3.linear(data.len, 256), .{ .x = 256 }, 0, &.{
            buf.argPtr(), g.interface.arg(&len), g.interface.arg(&scale),
        });
        try kernel.synchronize();
        try buf.download(&data, @sizeOf(@TypeOf(data)));
        return; // data == .{ 0, 4, 0, 8 }
    }
    // The CPU path is a plain loop; LLVM vectorizes it.
    for (&data) |*x| x.* = kernels.scaleRelu(x.*, scale);
}
```

`g.available(.cuda)` is a comptime question: did this build emit CUDA
artifacts? `RawKernel("scale_relu", .cuda)` is a compile error when it did
not, so the guard is what lets the same `main` build on a machine with no GPU.
`init(0)` opens device 0 and fails at run time on a machine with no driver or
no card; `catch break :gpu` sends that to the loop. For HIP, write `.hip` in
both places. Nothing has executed on AMD hardware yet; see
[Backend status](reference.html#backend-status).

The launch is yours to describe: `Dim3.linear(n, 256)` is the grid of
256-thread blocks covering `n` elements, and the block is `.{ .x = 256 }`.
Each argument is a pointer to a variable holding it, in the kernel's parameter
order; `buf.argPtr()` is the buffer's device pointer. `alloc`, `upload` and
`download` take bytes. `launch` returns once the launch is queued;
`synchronize` waits for it, and is where a fault inside the kernel shows up.

`catch break :gpu` takes the CPU path on any error, including a build mistake
such as a device arch the card cannot load. The library logs those at
`std.log.err`; to make them fatal, fall back only on `error.InitFailed` and
`error.NoDevice`. See [Troubleshooting](troubleshooting.html).

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

`g.available(.cuda)` is then false and the guarded branch compiles out. An
unguarded `RawKernel(name, .cuda)` becomes a compile error naming both causes
and both fixes.

## Commands

```sh
# Library tests
zig build test

# Documentation site -> zig-out/docs (add -Dopen to view it in a browser)
zig build docs
zig build docs -Dopen

# Basic example (the GPU if there is one, else the CPU loop)
cd examples/basic
zig build run

# Exhaustive example (GPU: requires libc + a CUDA/HIP driver)
cd examples/exhaustive
zig build run
```

With Nix:

```sh
nix build        # documentation site
nix flake check  # zig build test
nix develop      # Zig + ROCm/CUDA library paths
```

## Where to go next

- [Build integration](build.html): pinning the device arch (do this before you
  ship anything), multiple kernel roots, what `.auto` gets wrong in CI.
- [Reference](reference.html): `RawKernel`, the error set, kernel arguments,
  device math accuracy.
- [Advanced](advanced.html): graph replay, the device builtins and atomics,
  vectorized CPU loops over `g.math`.
- [Runtime](runtime.html): the layer underneath `RawKernel`, meaning modules,
  buffers, streams and several devices.
- [Troubleshooting](troubleshooting.html): start here when something fails at
  run time.
