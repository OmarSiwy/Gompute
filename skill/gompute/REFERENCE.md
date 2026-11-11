# Gompute API reference

`g` is `@import("gompute")`. In a kernels file the same name resolves to the
host root (`src/root.zig`) in one compilation and to the device root
(`src/device.zig`) in the other. The device root is a subset: `math`,
`builtins` (device only) and the five switches listed under "Device-side
names". Everything else below is host only.

## Host

| | |
| --- | --- |
| `g.Backend` | `enum { cpu, cuda, hip }`. `.cpu` names no kernel: `RawKernel(name, .cpu)` is a compile error |
| `g.available(comptime backend) bool` | whether this build emitted artifacts for `backend`; always false for `.cpu`. Ask it before naming `RawKernel(name, backend)` |
| `g.RawKernel(entry_name, .cuda \| .hip)` | a handle type for one exported entry; compile error when the build emitted no artifact for that backend or no root exports `entry_name` |
| `g.rawKernelByName(backend, name, ordinal) Error!RawByName(backend)` | the same handle for a name known only at run time; an unknown name is `error.KernelNotFound` |
| `g.RawByName(backend)` | the type `rawKernelByName` returns; it has no `init` |
| `g.interface.Arg`, `g.interface.arg(&x)` | one `kernelParams` entry: a pointer to one argument's storage, not a copy |
| `g.Dim3`, `g.Error`, `g.lastDriverError()` | below |
| `g.runtime.{dynamic, cuda, hip}` | the driver layer `RawKernel` sits on |
| `g.math` | device math, also on the host |

### `RawKernel` / `RawByName` members

| Member | Notes |
| --- | --- |
| `init(ordinal: c_int) Error!Self` | `RawKernel` only. Retains the primary context and loads (or reuses) the module of the root that exports the entry; only that root is JIT'd |
| `deinit()` | drops this handle only; safe twice; not device teardown |
| `alloc(bytes) Error!Buffer` | **bytes**, 256-byte aligned on both drivers; the caller `free`s it |
| `allocPinned(bytes) ![]u8`, `freePinned(mem)` | page-locked host memory, so async copies are actually async; `freePinned` takes only what `allocPinned` returned |
| `createStream()`, `createStreamNonBlocking()`, `createEvent(timing: bool)` | the caller owns each and must `deinit` it; a capture needs a non-blocking stream |
| `hasGraphs()`, `hasEvents()` | false on old drivers; the calls then return `error.Unsupported` |
| `launch(grid, block, shared_bytes, args)` | NULL stream; returns once queued |
| `launchOn(&stream, grid, block, shared_bytes, args)` | ordered on `stream`; returns once queued |
| `synchronize()` | whole device |
| `context` | `runtime.cuda.Context` / `runtime.hip.Context`: `info`, peer access, everything below |
| `Buffer`, `Stream`, `Graph`, `GraphExec`, `Event`, `CaptureMode` | types |

`launch`/`launchOn` make the handle's device current on the calling thread,
allocate nothing and copy nothing. `args` is one entry per kernel parameter,
in order: `buffer.argPtr()` for a buffer, `g.interface.arg(&value)` for a
scalar, with the parameter's exact type. Nothing checks the list against the
kernel's signature. The driver reads the argument values when you enqueue
the launch (and records them when capturing), so the storage must outlive
the call.

## Driver objects (`g.runtime.cuda` / `g.runtime.hip`: same shape)

**Context**: `init(ordinal)`, `deinit`, `makeCurrent`, `synchronize`, `alloc`,
`allocPinned`/`freePinned`, `loadModuleFromMemory(image)` (cached per device
and image), `createStream`, `createStreamNonBlocking`, `createEvent`,
`hasGraphs`, `hasEvents`, `info() !DeviceInfo` (name, uuid, PCI address,
SMs, compute capability, FP64 ratio, clocks, memory; `bandwidth()` is peak
bytes/s), `computeCapability() !u32`
(`major * 10 + minor`: 89 on sm_89, 110 on gfx1100), `fp64Ratio() !u32`
(FP32:FP64; 2 on NVIDIA datacenter parts, 64 on consumer parts; 1, 2 or 16 by
gfx family on AMD), `multiprocessorCount() !u32` (SMs, or compute units),
`canAccessPeer(&peer) !bool`, `enablePeerAccess(&peer) !void` (one
direction, idempotent, `error.Unsupported` without a direct link). On HIP the
four device queries need ROCm 6.0+ (`hipGetDevicePropertiesR0600`); an older
runtime gets `error.Unsupported`.

**Buffer**: `upload(host, n)`, `download(host, n)`, `uploadAt/downloadAt(host,
offset, n)`, `copyFrom(src, src_off, dst_off, n)`, `copyFromAsync(..., &stream)`
(a peer copy when the two buffers are on different ordinals),
`uploadAtAsync/downloadAtAsync(host, offset, n, &stream)` (host must be pinned),
`fillAsync(byte, n, &stream)`, `argPtr()`, `deviceAddr()`, `free()`
(idempotent). Sizes are bytes and are not bounds-checked in-process. A null
or freed buffer is `error.InvalidArgument`.

**Stream**: `synchronize`, `query() !bool` (never blocks), `waitEvent(&ev)`,
`beginCapture(mode)`, `endCapture() !Graph`, `isCapturing`, `deinit`.

**Graph**: `instantiate() !GraphExec`, `deinit`. **GraphExec**:
`launch(&stream)`, `update(&graph) !bool` (false means the topology changed,
so re-instantiate), `deinit`.

**Event**: `record(&stream)`, `synchronize`, `query`,
`Event.elapsedUs(&start, &end) !f32`, `deinit`.

**Module**: `getKernel(name) !Kernel`, `deinit`. **Kernel**:
`launch(grid, block, shared, args)`, `launchOnStream(..., stream_handle)`.

`deviceCount() !u32` counts the ordinals (`GOMPUTE_FAKE_DEVICES=N` adds fake
ones: extra contexts on device 0, `info().fake` set). `shutdown()` unloads
every module and releases every context. Call it only once nothing else will
touch the device. `cuda.loadedModuleCount()` and `hip.loadedModuleCount()`
report how many (device, image) modules are JIT'd.

`g.runtime.dynamic.Compute` is one tagged-union facade over CUDA, HIP and a
`.cpu` arm that fails politely. It offers `init(?Backend)`, `initDevice`,
`alloc`, `loadModule`, `createStream`, `synchronize`, `info`, `fp64Ratio`,
`computeCapability`, `multiprocessorCount`, `canAccessPeer` and
`enablePeerAccess`; `dynamic.deviceCount(backend)` counts the ordinals. It
has no events or graphs; for those, use the vendor module.

## Errors

`g.Error` is one flat set: `InitFailed` (no driver/GPU), `NoDevice`,
`ContextFailed`, `SyncFailed` (usually an earlier launch faulted),
`AllocFailed`, `ModuleLoadFailed` (arch mismatch), `KernelNotFound`,
`LaunchFailed`, `CopyFailed`, `InvalidArgument`, `BackendUnavailable`,
`CaptureFailed`, `GraphFailed`, `EventFailed` and `Unsupported` (driver too
old, a HIP runtime before ROCm 6.0 for the device queries, or no direct link
for `enablePeerAccess`).
`g.lastDriverError() -> { code: i64, backend: .none | .cuda | .hip }` is
thread-local and cleared on success.

## Geometry

`g.Dim3 = extern struct { x = 1, y = 1, z = 1 }`.
`Dim3.linear(n, block_x)` returns a zero grid for n = 0 (no driver launches
it, and `RawKernel` does not check, so return first) and **panics** when the
grid is out of range. `Dim3.linearChecked(n, block_x) Error!Dim3` returns
an error instead, so use it for counts that come from outside.
`Dim3.max_grid_x = 2^31 - 1`.

## Device-side names

`g.is_device` (bool), `g.kernel_callconv`, `g.GlobalPtr(T)`
(`[*]addrspace(.global) T` on device, `[*]T` on the host),
`g.globalIdX(block_size)` and `g.exportRaw(name, &fn)`. These are the five
names whose meaning differs between the roots. On the host `globalIdX` is a
compile error and `exportRaw` is a no-op, so guard the export with
`if (g.is_device)`.

`g.builtins` exists in the device compilation only:

| | |
| --- | --- |
| `globalIdX(bs)`, `localIdX()`, `blockIdX()` | thread indices from the launched geometry on both vendors; `bs` is ignored, but pass the block size you launch with |
| `shuffleXor(T, v, mask)` | the value of lane `this ^ mask` in a 32-lane group (`shfl.sync.bfly`; DPP / `ds_swizzle` / `v_permlanex16` on AMD); every lane of the group must reach it, so launch a multiple of 32 |
| `gridDimX()` | NVPTX only; on AMD, pass the stride as an argument |
| `barrier()` | `__syncthreads`, with the fences included on AMDGCN; every thread must reach it |
| `spinPause()` | ~64 ns backoff; a no-op below sm_70/PTX 6.3 |
| `loadAcquireDevice(p)`, `storeReleaseDevice(p, v)` | u32 at `.gpu` scope (cheaper than Zig's `.sys` on NVPTX) |

Block-shared scratch is a file-scope `var x: [N]T addrspace(.shared) = undefined;`.

Device math (`g.math`, also the std-only module `dep.module("math")`): `exp
exp2 log log2 log10 log1p pow sin cos tan tanh sinh cosh expm1 atan atan2
asinh acosh atanh sqrt rsqrt`, for `f32`/`f64`/vectors; `pow(x, y: @TypeOf(x))`,
`atan2(y, x: @TypeOf(y))`. Each gives the
same bits on host, NVPTX, AMDGCN and at comptime within one FMA class: f32
exp, log, log2, log10, sin and cos by being correctly rounded (checked on all
2³² inputs), f64 exp, log and pow by an FMA body and an FMA-free body picked
at comptime from the target (FMA class: x86_64 with `fma`, aarch64, NVPTX,
AMDGCN; the FMA-free class keeps git bd8800b's bits), the rest by being one body
with no FMA or intrinsic (no optional contraction in the IR, checked on four
targets). f64 `asinh` and `acosh` call f64 `log`, so they follow its class;
their f32 forms round the split away. A baseline host build is in the
FMA-free class, so it can disagree with the GPU by one ulp on f64
exp/log/pow, asinh and acosh. `atan(-0)` is -0 (+0
before git bd8800b). `asinh`, `acosh`, `atanh` and `atan2` are musl's, over
this module's own `log`/`log1p`/`atan`. `math.max_ulp` holds the f64
bounds: exp 0.52, log 0.52, pow 0.55 (measured 0.505 / 0.502 / 0.502 against
f128); asinh 1.5, acosh 2.3, atanh 1.8, atan2 1.5 (measured 1.259 / 2.152 /
1.574 / 1.300). The vector forms are lane-exact with the scalar call; a CPU loop
reaches them only by stepping `@Vector`s by hand.

## Build API (`@import("gompute")` in build.zig)

`emitKernels(b, dep, exe, EmitOptions)` is for one artifact-consuming
executable per dependency; a second call on the same `dep` panics.
`addKernels(b, dep, KernelsOptions) Kernels { kernels, gompute }` is for
several: import `k.gompute` in place of `dep.module("gompute")`, in the exe
and in its host-side kernels module.

`EmitOptions` fields:

| Field | Meaning |
| --- | --- |
| `kernels_root` | one root, named `"kernels"` |
| `kernel_roots: []KernelRoot` | several roots, compiled in parallel; each `KernelRoot` is `{ name, root, imports, imports_ctx, heavy }`. Kernel names must be unique across roots |
| `imports: ?DeviceImportsFn`, `imports_ctx` | `fn (b, target, optimize, ctx) []Import`. A callback, because a module carries its own target |
| `heavy_lanes: u8 = 1` | how many `.heavy` roots may compile at once |
| `cuda`, `hip: { enabled = true, gpu = .auto \| .{ .name = "sm_89" }, optimize = null, sass = null }` | per-backend device settings. `sass` (CUDA only) also embeds SASS for `gpu` beside the PTX so that card skips the JIT; null does it when `ptxas` and `fatbinary` are on PATH |
| `target`, `optimize` | default to the exe's |

`KernelsOptions` takes `root_source_file` in place of `kernels_root`, and
`target` and `optimize` are required.

`.auto` probes the **build** machine (`nvidia-smi`, `amdgpu-arch`), and when
it finds nothing it compiles that backend out with a warning. Pin `.name` for
CI, Docker, Nix and releases. Device `Debug` is promoted to `ReleaseFast`.

The generated `gompute_kernels` module provides `emitted`, `has_cuda`,
`has_hip`, `cuda_sass`, `root_names`, `cuda_images`, `hip_images` and
`cuda_index`/`hip_index` (name → `{ blob, symbol }`).
