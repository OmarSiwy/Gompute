# Troubleshooting

Symptoms, in the order people hit them. Most of these are build-time mistakes
that only show up at run time, which is why the library shouts about them.

## `cuda: symbol not found: cuInit`

Your executable is not linked against libc.

```zig
exe.root_module.linkSystemLibrary("c", .{});
```

Without libc, Zig's `std.DynLib` resolves to `ElfDynLib` rather than
`DlDynLib`. It opens `libcuda.so` successfully and then cannot resolve a single
symbol out of it. The message reads exactly like a driver version mismatch and
is not one. CPU-only builds do not need this.

## `error.InitFailed`

No driver on this machine. The library `dlopen`s `libcuda.so` /
`libamdhip64.so` and neither is present, or `cuInit` itself failed.

This is the *normal* answer on a machine with no GPU. It is the error to catch
on `init` to take your CPU path, and it is what you should expect in CI.

If you believe a GPU is there, check that the driver library is on the loader
path. Gompute also looks in `/run/opengl-driver/lib`, which is where NixOS puts
it.

## `error.ModuleLoadFailed`, and a loud log line about arch mismatch

The artifact in the binary was compiled for a different GPU than the one it is
running on. `sm_89` code does not load on an `sm_75` card.

The usual cause is `.auto`. It probes the GPU in the **build** machine, which
is the wrong machine whenever you build in CI, in Docker, in a Nix sandbox, or
on a laptop for a cluster. Pin it:

```zig
gompute_build.emitKernels(b, dep, exe, .{
    .kernels_root = b.path("src/kernels.zig"),
    .cuda = .{ .gpu = .{ .name = "sm_89" } },
    .hip = .{ .gpu = .{ .name = "gfx1100" } },
});
```

## `error.KernelNotFound`

The image loaded, but holds no symbol by that name. Two things to check, and
the log line names both:

1. Is the kernel exported? A kernel root needs
   `comptime { if (g.is_device) g.exportRaw("scale_relu", &scaleReluKernel); }`.
2. Does the `kernels_root` you passed to `emitKernels` point at the same file
   your host code imports as `kernels`? Two files, one exporting and one
   imported, is the quiet version of this bug.

From `rawKernelByName`, it just means the runtime name matched nothing in the
compiled set.

## Compile error: `RawKernel(name, .cuda) was asked for, but this build emitted no cuda artifacts`

Either `.auto` found no NVIDIA GPU on the build machine, or you passed
`.cuda = .{ .enabled = false }`. Pin the arch you deploy to, as above, or
guard the GPU branch with `if (comptime g.available(.cuda))` so a build
without the backend compiles it out.

If the message says the build never emitted any artifacts, `emitKernels` (or
`addKernels`) is missing from your `build.zig`.

## It silently ran on the CPU

`catch break :gpu` on `init` takes the CPU path on every error, and two of
them look alike from there:

- No driver, or no device at that ordinal: `error.InitFailed` or
  `error.NoDevice`. Falling back is correct; there is no GPU.
- The artifact is in the binary and failed to load: `error.ModuleLoadFailed`
  or `error.KernelNotFound`. That is a build mistake, and the library logs it
  at `std.log.err` with what to check.

To make the second case fatal, fall back on the first two only:

```zig
var kernel = g.RawKernel("scale_relu", .cuda).init(0) catch |err| switch (err) {
    error.InitFailed, error.NoDevice => break :gpu,
    else => return err,
};
```

## `error.LaunchFailed`

The driver refused the launch. In order of likelihood: a block past the
device's limit (both vendors cap at 1024), a grid past `gridDim.x = 2^31-1`, or
argument storage that went out of scope before the launch. Kernel arguments are
passed as *pointers to* your variables, so every one of them needs to outlive
the call.

## `error.SyncFailed`

Almost always a fault from an *earlier* launch. `launch` is asynchronous and
does not report device-side faults; the next `synchronize` does. If a
synchronize fails and the launch before it looked fine, the launch before it is
the suspect. `lastDriverError().code` has the vendor's own code.

## `error.Unsupported` from `enablePeerAccess`

The two devices have no direct link (no NVLink, PCIe P2P or XGMI path between
them), or they are two `GOMPUTE_FAKE_DEVICES` contexts on one card. Nothing is
broken: a `copyFrom` between their buffers still works, staged by the driver
through host memory. `canAccessPeer` answers the question without the error.

## Getting the driver's own answer

When the library's error is not specific enough:

```zig
const drv = g.lastDriverError();
std.debug.print("{t} code {d}\n", .{ drv.backend, drv.code });
```

Thread-local, and cleared on every successful driver call, so it always
describes the failure you just saw. The codes are `CUDA_ERROR_*` and
`hipError_t` values; look them up in the vendor's headers.
