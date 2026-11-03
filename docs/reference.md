# Reference

## Versions

Gompute pins one Zig build at a time; every Zig release so far has moved
`std.Build`, `@typeInfo` or the GPU backends under it. `main` follows Zig
master at a pinned nightly, the same build ESPice pins. A release is a `v<version>` tag (the older tags carried the Zig
release in their name); the package version is separate.

| Git tag         | Gompute | Zig                      |
| --------------- | ------- | ------------------------ |
| `0.16.0`        | 1.0.0   | 0.16.0                   |
| `0.17.0-v1.0.0` | 1.0.0   | 0.17.0                   |
| `0.17.0-v1.4.0` | 1.4.0   | 0.17.0                   |
| `v1.5.0`        | 1.5.0   | 0.18.0-dev.349+d924f83bf |

`minimum_zig_version` in `build.zig.zon`, the overlay revision in `flake.lock`
and the version in `.github/workflows/*` name the same build and move
together. Nothing older or newer is supported.

## Platform support

| Platform      | Host code | CUDA / HIP backends                                         |
| ------------- | --------- | ----------------------------------------------------------- |
| Linux         | Yes       | Yes                                                         |
| Windows       | Yes       | Yes: `nvcuda.dll`, `amdhip64_7.dll` / `_6` / unsuffixed     |
| macOS         | Yes       | Compiles; only finds NVIDIA's last driver (CUDA 10.2)       |
| `wasm32-wasi` | Yes       | Every GPU init returns `error.InitFailed`. Link no libc: Zig ships no wasi-libc |

The drivers are loaded at run time: `std.DynLib` on Linux, macOS and the BSDs,
`LoadLibraryA` on Windows (Zig's std has no Windows `DynLib`), and a stub
that never opens anything on wasm (`src/runtime/dynlib.zig`). A missing
driver is the same `error.InitFailed` on every platform, so one `catch` on
`init` sends every platform to your CPU path.

`emitKernels` skips the GPU sub-compilations on wasm, so there is nothing
to launch there anyway.

## Behaviour worth knowing

Things that are easy to get wrong because nothing in the API says them out loud.

| | |
| --- | --- |
| `init(0)` | The CUDA/HIP **device ordinal**. |
| `launch` | Returns once the launch is queued, on the NULL stream. It does not synchronize, and a fault inside the kernel surfaces at the next `synchronize`. The arguments are pointers to your variables, so each must outlive the call. |
| `deinit` | Always safe, and safe to call twice. It releases *your handle*, not the shared device state. |
| Contexts and modules | Shared process-wide, per device. Two `RawKernel`s on one GPU share one CUDA primary context and JIT the artifact once, so buffers allocated through one are usable by the other. `runtime.cuda.shutdown()` is the only real teardown. |
| Several devices | One handle per ordinal. A `copyFrom` between buffers on two ordinals is a peer copy. See [Several devices](runtime.html#several-devices). |
| Threads | Safe. A context is made current per thread on every `launch`. Buffers are not: two threads writing one race. |
| `Buffer` | `free()` is idempotent. Use-after-free is not checked. `upload`/`download` do not bounds-check against `Buffer.bytes`; the driver catches the overrun and you get `error.CopyFailed`. |
| Errors | One flat set of 15, listed [below](#errors). `lastDriverError()` returns the raw CUDA/HIP code behind the last failure, which is the only way to tell "no driver" from "out of memory". It is cleared on success. |
| Block size | Must be 1…1024. Not required to be a multiple of the warp/wave size, but anything else wastes part of every warp. `globalIdX` reads the size you launched with, so its argument is ignored. |
| Empty input | `Dim3.linear(0, b)` is a zero grid, which the driver refuses. Return before the launch. |

## Errors

Every fallible call in the library returns from one set, `g.Error`. It is
closed and flat on purpose: a caller that wants to distinguish "the machine has
no GPU" from "this build shipped no PTX" should not have to match on a nested
union.

| Error | Raised when |
| --- | --- |
| `error.InitFailed` | The driver library is absent, or `cuInit`/`hipInit` failed. This is the *no GPU here* error. |
| `error.NoDevice` | The ordinal passed to `init` names no device (`< 0`, or past the last one), or a device attribute query failed. |
| `error.ContextFailed` | Retaining or making current the device's primary context failed. |
| `error.SyncFailed` | `synchronize` on a context or stream failed, or a stream could not be created. Usually the *previous* launch faulted. |
| `error.AllocFailed` | Device allocation failed: out of memory, or a zero-byte request. Also returned by `runtime.dynamic` `alloc` on the `.cpu` backend, which has no device memory. |
| `error.ModuleLoadFailed` | The PTX/HSACO image was rejected by the driver. Almost always an arch mismatch: `sm_89` code on an `sm_75` card. |
| `error.KernelNotFound` | The image loaded but holds no symbol by that name: a kernel not exported with `exportRaw`, or a bad name passed to `rawKernelByName`. |
| `error.LaunchFailed` | The driver refused the launch: block size over the device limit, a grid past `gridDim.x`, or bad arguments. |
| `error.CopyFailed` | A host↔device or device↔device copy failed. An overrun past `Buffer.bytes` lands here, since the copy is not bounds-checked in-process. |
| `error.InvalidArgument` | Caller-side validation: a null or freed `Buffer`, or a launch geometry `Dim3.linearChecked` will not express. |
| `error.BackendUnavailable` | A backend this build emitted no artifact for. `RawKernel` and `rawKernelByName` make that a compile error first, so no call in 2.0 returns it; ask `g.available` instead. |
| `error.CaptureFailed` | A stream capture failed to begin or end, or was invalidated in between (a host-blocking call ran during it). |
| `error.GraphFailed` | A graph failed to instantiate, launch or update for a reason other than a topology change, which `GraphExec.update` reports as `false`. |
| `error.EventFailed` | Creating, recording, waiting on or timing an event failed; timing an event created with `timing = false` lands here. |
| `error.Unsupported` | The driver is too old for the call: graphs and events need CUDA 11.4+ or a ROCm that exports them (`hasGraphs()` / `hasEvents()` say so first), and `info`, `fp64Ratio`, `computeCapability` and `multiprocessorCount` need ROCm 6.0+ on HIP. Also `enablePeerAccess` between two devices with no direct link, including a `GOMPUTE_FAKE_DEVICES` pair; copies between them still work. |

The error tells you *which step* failed; `lastDriverError()` tells you *why*.

```zig
var kernel = g.RawKernel("scale_relu", .cuda).init(0) catch |err| {
    const drv = g.lastDriverError();
    std.debug.print("{t}: {t} code {d}\n", .{ err, drv.backend, drv.code });
    return err;
};
```

`DriverError` is `struct { code: i64 = 0, backend: enum { none, cuda, hip } = .none }`.
It is thread-local, and it is cleared on every *successful* driver call, so it
describes the failure you just saw, not one from three calls ago. Codes are the
vendor's own: `CUDA_ERROR_*` and `hipError_t`.

See [Troubleshooting](troubleshooting.html) for what to do about each.

## Launch geometry

`Dim3` is the `extern struct { x: u32 = 1, y: u32 = 1, z: u32 = 1 }` both
drivers expect: `launch` takes one for the grid, measured in blocks, and one
for the block, measured in threads.

| | |
| --- | --- |
| `Dim3.linear(n, block_x)` | The grid of `block_x`-wide blocks covering `n` elements. `n == 0` gives a zero grid, since an empty range is not a fault. |
| `Dim3.linearChecked(n, block_x)` | Same, as `Error!Dim3`. Rejects `n == 0`, `block_x == 0`, and any grid past `max_grid_x`. |
| `Dim3.max_grid_x` | `2^31 - 1`, the cap both vendors put on `gridDim.x`. |

`linear` **panics** where `linearChecked` returns `error.InvalidArgument`.
That is deliberate: clamping would launch fewer blocks than there are elements
and silently leave the tail of the buffer unprocessed, which is a wrong answer
with no signal. Use `linearChecked` whenever the count comes from outside the
program: a file header, a socket, argv.

## Naming

**`exportRaw`** goes in your *kernels* file and names an entry point.
**`emitKernels`** goes in your *build.zig* and wires up the compilation. They
are different files, and `@import("gompute")` means a different thing in each:
gompute's build API in `build.zig`; in `kernels.zig`, the host root or the
device root, depending on which compilation is reading it. Forgetting either
produces a different error.

## RawKernel

`g.RawKernel(name, backend)` is a handle on one exported kernel. `backend` is
`.cuda` or `.hip`; `.cpu` is a compile error, because the CPU path is your own
loop. Naming a backend this build emitted no artifacts for is a compile error
too, so guard it with `if (comptime g.available(.cuda))`. A `name` that no
kernel root exports is a compile error naming it.

| | |
| --- | --- |
| `init(ordinal: c_int) Error!Self` | Retains the device's primary context and loads the root that exports `name`. Cheap after the first handle on a device. |
| `deinit(self) void` | Releases your handle. Safe twice. Not device teardown; see `shutdown()`. |
| `alloc(bytes: usize) Error!Buffer` | **Bytes**. Caller frees the buffer. |
| `allocPinned(bytes) Error![]u8` / `freePinned(block)` | Page-locked host memory, which the async copies need. Release it with `freePinned` only. |
| `launch(grid, block, shared_bytes, args) Error!void` | On the NULL stream. Returns once queued. |
| `launchOn(&stream, grid, block, shared_bytes, args) Error!void` | The same, ordered on `stream`. |
| `synchronize() Error!void` | Wait for every launch and copy queued on this device. |
| `createStream()` / `createStreamNonBlocking()` | A stream; the non-blocking kind is what a graph capture needs. Caller `deinit`s it. |
| `createEvent(timing: bool)` | An event; `false` orders but cannot be timed. |
| `hasGraphs()` / `hasEvents()` | False on a driver too old for them. |
| `context` | The underlying `runtime.cuda.Context` / `runtime.hip.Context`, for what the handle does not wrap: `info`, peer access. |
| `Buffer`, `Stream`, `Graph`, `GraphExec`, `Event`, `CaptureMode` | The backend's own types. See [Runtime](runtime.html). |

`g.rawKernelByName(backend, name, ordinal)` is the same handle for a name
decided at run time; it returns `g.RawByName(backend)`, which has no `init`,
and an unknown name is `error.KernelNotFound`. See
[Picking a kernel at run time](build.html#picking-a-kernel-at-run-time).

`g.available(backend)` is the comptime guard: true when this build emitted
artifacts for `backend`, always false for `.cpu`.

## Kernel arguments

`args` is CUDA's and HIP's `void **kernelParams`: one `g.interface.Arg` per
kernel parameter, in order, each a pointer to the storage holding that value.
`g.interface.arg(&x)` wraps a variable; `buffer.argPtr()` is a buffer's device
pointer. Nothing is copied, so every variable must stay alive and unmoved
until `launch` returns, and a pointer to a temporary hands the kernel garbage.

Pass each parameter as its own scalar argument; there is no struct
marshalling. Use fixed-width types for scalars, as the examples do with
`len: u64`: `usize` is the host's width, which need not be the device's.

## Device math

`@exp`, `@log`, `@sin` and friends do not compile for NVPTX or AMDGCN: those
targets emit no libcalls, so the backend fails with `no libcall available for
fexp` or `Cannot select: fsin`. `g.math` provides device-safe replacements, and
they compile on the host too, so one kernel source builds both ways.

Every one is `inline fn (x: anytype) @TypeOf(x)`, except `pow(x, y)` where
`y` has `x`'s type. They take `f32`, `f64`, and `@Vector`s of either. A bare
literal or `const lim = 80.0` is a `comptime_float` and a compile error;
write `@as(f64, 80.0)`. That's deliberate: choosing f64 silently could differ
from what an f32 kernel computes. The set:
`exp exp2 log log2 log10 log1p pow sin cos tan tanh sinh cosh expm1 atan sqrt
rsqrt`.

**The same bits everywhere.** No function uses a hardware approximation,
inline asm or a target intrinsic, so a given input gives the same bits on the
host (with or without FMA hardware), on NVPTX, on AMDGCN, and when evaluated
at comptime. Most functions get there by being one body with no FMA.
`examples/exhaustive` checks the CUDA result against the CPU result bit for
bit for every function over 64K inputs.

f32 `exp`, `log`, `log2`, `log10`, `sin` and `cos` get there differently:
they are **correctly rounded**, so any two bodies agree. The GPU runs an f32
body (with `@mulAdd`, exact in software too) and falls back to binary64 on
the rare input it cannot settle; the CPU runs a body shaped for its SIMD
units. Both are
checked against an f128 oracle on all 2³² inputs, and `examples/exhaustive`
checks the CUDA result against the CPU on all 2³² as well. On an RTX 4060
Laptop (Zig 0.18.0-dev.349, `examples/bench` with `BENCH=math`) that is
89–111 Gcalls/s for `exp` and the logs and 67 for `sin`/`cos`.

Every other f32 function is one rounding of the f64 body. On a GPU that
costs f64 throughput (1/64 rate on consumer NVIDIA): on the same card f32
`exp2`, `expm1`, `log1p`, `tan`, `tanh`, `sinh`, `cosh` and `pow` run 1.5–4
Gcalls/s. That is the price of the bits agreeing.

The math is also its own std-only module, `b.dependency("gompute", .{}).module("math")`,
for code that wants it without the GPU layer, such as a compiler folding
constants.

**Vector forms.** At f32 and f64, `exp`, `exp2`, `log`, `log2`, `log10` and
`pow` run as real SIMD, and so do f32 `sin`, `cos`, `tanh`, `sinh` and
`cosh`. Each vector lane is the scalar call's result bit for bit, because the
scalar body is the width-1 instance of the same code. A vector with any
special lane (zero, subnormal, inf, NaN, huge) falls back to the scalar body
for each lane. f64 `sin`/`cos`/`tanh`/`sinh`/`cosh`/`expm1`, `log1p` and
`atan` still run one lane at a time.

**Accuracy**, as the maximum error measured in tests:

| | Max error | Reference | Notes |
| --- | --- | --- | --- |
| `exp` | 0.505 ulp | f128 oracle | ARM optimized-routines. Faithful; promised < 0.52 (`math.max_ulp`). |
| `log` | 0.502 ulp | f128 oracle | Faithful; promised < 0.52. `log(1)` is +0. |
| `pow` | 0.502 ulp | f128 oracle | Faithful; promised < 0.55. Exact results come back exact. |
| `exp2`, `log2`, `log10` | ≤1 / ≤1 / ≤2 ulp | glibc | f64 `log10` is 0.52 ulp against a 60-digit reference; the gap is glibc's error. |
| f64 `sin`, `cos` | bit-identical to `@sin`/`@cos` | compiler_rt | musl, including Payne-Hanek for huge arguments. |
| f64 `cosh` | 0.998 ulp | f128 oracle | std's is 1.131. |
| f64 `tanh`, `sinh` | 2.05 / 1.75 ulp | f128 oracle | Bit-identical to std's musl bodies; not faithful. |
| f32 `exp`, `log`, `log2`, `log10`, `sin`, `cos` | correctly rounded | f128 oracle, **all 2³² inputs** | Host and device bodies both, scalar and vector. |
| f32 `exp2`, `tanh`, `sinh`, `cosh` | ≤ 0.501 ulp | f64, **all 2³² inputs** | exp2/tanh/cosh 0.50, sinh 0.501. f32 `tan` is the ratio of two roundings, not checked exhaustively. |
| `expm1`, `log1p` | std's bits | std | `expm1` minus one line that does not compile for AMDGCN. |
| `atan` | within 1 ulp of std | std | std's vector body on every target. |

`exp`, `log`, `pow` (in both arguments), `expm1`, `log1p` and `sinh` are
**monotone**. That is tested at every table seam and branch edge, so a prover
that evaluates only at interval endpoints can rely on it. That is why integer
`y` goes through the same table route as any other `y`: the old
square-and-multiply fast path stepped down across y = -64.

Why not `@exp`/`@sin`? Those builtins emit libcalls, and NVPTX/AMDGCN have no
libm (`no libcall available for fexp`). std's `expm1` and scalar `atan` also do
not compile for AMDGCN, because they raise an FP flag through inline asm the
AMDGPU backend cannot match. **An NVIDIA-only test matrix will not see this.**

## Backend status

| Backend | Device code |              Host runtime | Validation in this package                     |
| ------- | ----------: | ------------------------: | ---------------------------------------------- |
| CUDA    |         PTX | Runtime-loaded driver API | Cross-compiled and executed; PTX entries inspected |
| HIP     |       HSACO |    Runtime-loaded HIP API | Cross-compiled; ELF symbols/metadata inspected |

HIP is cross-compiled and its symbols inspected, but **nothing in this package
has been executed on AMD hardware**; no such device was available. Its device
queries (`Context.info`, `fp64Ratio`, `computeCapability`,
`multiprocessorCount`) read `hipGetDevicePropertiesR0600`, the properties
layout ROCm 6.0 froze, and `hipDeviceComputeCapability`; an older ROCm gets
`error.Unsupported` from them. The FP64 ratio comes from the gfx family:
gfx906, gfx908, gfx94x and gfx950 report 2, gfx90a 1, every other gfx9, gfx10,
gfx11 and gfx12 part 16, and an unknown family `error.Unsupported`.
