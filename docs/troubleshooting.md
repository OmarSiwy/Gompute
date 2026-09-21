# Troubleshooting

Symptoms, in the order people hit them. Most of these are build-time mistakes
that only show up at run time, which is why the library shouts about them.

## `cuda: symbol not found: cuInit`

Your executable is not linked against libc.

```zig
exe.root_module.linkSystemLibrary("c", .{});
```

Without libc, Zig 0.16's `std.DynLib` resolves to `ElfDynLib` rather than
`DlDynLib`. It opens `libcuda.so` successfully and then cannot resolve a single
symbol out of it. The message reads exactly like a driver version mismatch and
is not one. CPU-only builds do not need this.

## `error.InitFailed`

No driver on this machine. The library `dlopen`s `libcuda.so` /
`libamdhip64.so` and neither is present, or `cuInit` itself failed.

This is the *normal* answer on a machine with no GPU. It is what `AutoKernel`
silently falls back to the CPU on, and it is what you should expect in CI.

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

1. Is the spec exported? A kernel root needs
   `comptime { g.exportKernels(@This()); }`, or an explicit
   `g.exportKernels(.{ scale_relu })`.
2. Does the `kernels_root` you passed to `emitKernels` point at the same file
   your host code imports as `kernels`? Two files, one exporting and one
   imported, is the quiet version of this bug.

From `rawKernelByName`, it just means the runtime name matched nothing in the
compiled set.

## `error.BackendUnavailable`

`Kernel(spec, .cuda).init` in a build that emitted no CUDA artifact. Either
`.auto` found no NVIDIA GPU on the build machine, or you passed
`.cuda = .{ .enabled = false }`.

`Kernel(spec, .cuda)` itself is a *compile* error in that case, with the fix in
the message. You only reach the runtime error through `AutoKernel`, which
compiles all three backends by design.

## It silently ran on the CPU and took 100× longer

`AutoKernel.init()` distinguishes two things that look alike:

- No driver at all. Falls back to the CPU without a word, which is correct:
  there is no GPU.
- Artifact present but it failed to load. Logs a warning, because that is a
  build mistake and the cost of not noticing is enormous.

If you want the second case to be fatal:

```zig
var kernel = try g.AutoKernel(kernels.scale_relu).initStrict();
```

To assert what you actually got:

```zig
std.debug.assert(kernel.selected() == .cuda);
```

## `error.LaunchFailed`

The driver refused the launch. In order of likelihood: a `block_size` past the
device's limit (both vendors cap at 1024), a grid past `gridDim.x = 2^31-1`, or
argument storage that went out of scope before the launch. Kernel arguments are
passed as *pointers to* your variables, so every one of them needs to outlive
the call.

## `error.SyncFailed`

Almost always a fault from an *earlier* launch. `launch` is asynchronous and
does not report device-side faults; the next `synchronize` does. If a
synchronize fails and the launch before it looked fine, the launch before it is
the suspect. `lastDriverError().code` has the vendor's own code.

## A raw kernel is right on CUDA and wrong on HIP

`globalIdX(block_size)` is asymmetric. NVPTX reads the real block dimension out
of hardware and ignores the argument; AMDGCN uses the argument, because it has
no portable way to read it. Pass a 256 and launch with 128 and NVIDIA covers
for you while AMD computes wrong indices.

Generated `map` kernels guarantee the match by construction. Raw kernels do not,
so the `block_size` you pass `globalIdX` **must** be the `block.x` you launch
with.

## Compile error naming a `Params` field

```
gompute: `Params.inner.w` has type `usize`, which cannot cross the GPU boundary.
```

Some field cannot cross the host/device boundary. `usize` and `isize` are the
common ones: pointer width may differ between host and device, so they are
rejected rather than guessed at. Use a fixed-width integer, or give the type a
[custom boundary](advanced.html).

The path is the full field path, not just the type it happened to reach.

## The GPU and the CPU disagree in the last few bits of a reduce

Expected. `reduce` reassociates freely across threads and blocks, and float
addition is not associative. A GPU sum and a strict left-to-right CPU sum
differ in the low bits, and the GPU answer changes with the launch geometry.

Compare with a tolerance, or reduce in `f64`.

If the disagreement is *large*, `combine` is probably not associative, or
`identity` is not its neutral element. Both are your side of the contract.

## `scatter` gives different results every run

Also expected, if two indices are equal. Two threads writing one slot race, and
which one lands is unspecified on both vendors. There is no `scatterAdd`;
combining duplicates needs a device float atomic, which is gfx9+ and
`--unsafe-fp-atomics` on AMD.

## ≈29 compile errors from `std/dynamic_library.zig`

You are building for Windows or wasm and something instantiated a GPU backend.
Zig 0.16's `std.DynLib` only supports Linux and the Darwin/BSD family;
everything else is `@compileError("unsupported platform")`.

`.cpu` alone compiles everywhere. `AutoKernel` does not count as CPU-only,
because it instantiates all three backends. See [Platform support](reference.html#platform-support).

## Getting the driver's own answer

When the library's error is not specific enough:

```zig
const drv = g.lastDriverError();
std.debug.print("{t} code {d}\n", .{ drv.backend, drv.code });
```

Thread-local, and cleared on every successful driver call, so it always
describes the failure you just saw. The codes are `CUDA_ERROR_*` and
`hipError_t` values; look them up in the vendor's headers.
