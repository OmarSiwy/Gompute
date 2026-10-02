# Gompute API reference

`g` is `@import("gompute")`. In a kernels file the same name resolves to the
host root (`src/root.zig`) in one compilation and to the device root
(`src/device.zig`) in the other. Both roots expose the same names, except
for six switches, listed at the end.

## Specs

| | |
| --- | --- |
| `g.map(name, T, P, f, o: MapOptions)` | `f: fn (T, P) T` or `fn (x: anytype, p: P) @TypeOf(x)` (generic → CPU SIMD) |
| `g.mapFn(name, f, o)` | `map` with `T`, `P` inferred; concrete `f` only |
| `g.mapTo(name, In, Out, P, f, o)` | out of place; `f` is `fn (In, P) Out` or generic `fn (x: anytype, p: P) g.Lanes(@TypeOf(x), Out)` |
| `g.zip(name, A, B, Out, P, f, o)` | two input buffers; `f` is `fn (A, B, P) Out` or generic in both inputs |
| `g.mapIndexed(name, T, P, f, o)` | linear index, no 2-D; `f` is `fn (T, u64, P) T` or generic (`i` is then a vector of u64) |
| `g.reduce(name, T, P, combine: fn (T, T) T, identity: T, o: ReduceOptions(T, P))` | associative fold |
| `g.sum` / `g.min` / `g.max` / `g.any` / `g.all` `(name, T, P, o: ReduceOptions(T, P))` | presets; identities 0 / max / min / 0 / ~0; `any`/`all` integer only |
| `g.gather(name, T, Idx, o)` / `g.scatter(name, T, Idx, o)` | `Idx` is u8/u16/u32/u64 |
| `g.MapOptions` | `{ block_size: u32 = 256 }`, 1…1024 |
| `g.ReduceOptions(T, P)` | `{ block_size = 256, pre: ?*const fn (T, P) T = null, simd_op: ?ReduceOp = null }`. The presets set `simd_op` for you |
| `g.splat(V, scalar)` | `@splat` when `V` is a vector, identity otherwise |
| `g.Lanes(X, E)` | `E` at `X`'s width: the return type of a generic `mapTo`/`zip` body whose output type differs |
| `g.Fused(T, P, .{ Ops... })`, `g.Unary(T, P, f)` | compile-time composition; `.eval(x, p)` |
| `g.Kind`, `g.spec.kindOf(Spec)` | which kind a spec is |

A spec exposes `entry_name`, `block_size`, `Value`, `Parameters`,
`BoundaryParameters` and `eval`. Depending on the kind it also has `In`,
`Out`, `A`, `B`, `Index`, `identity`, `combine`, `pre` and `simd_op`.

## Handles

### `g.Kernel(spec, .cpu)`

`init(ordinal) !Self` (never fails), `deinit()`, `run(...)`. `Buffer = void`.
There is no `alloc`/`launch`.

### `g.Kernel(spec, .cuda | .hip)`

This is a compile error if the build emitted no artifact for that backend.

| Member | Notes |
| --- | --- |
| `init(ordinal: c_int) Error!Self` | retains the primary context and loads (or reuses) this root's module |
| `deinit()` | resets the handle; safe twice; not device teardown |
| `run(...)` | upload, launch, sync, download; device buffers are kept on the handle and grown on demand, so one `run` at a time per handle |
| `alloc(count) Error!Buffer` | **elements** of `Spec.Value` |
| `launch(...) Error!void` | no alloc, no copy, no sync; per-kind signature below |
| `reduceBlocks(count) u32` | `min(ceil(count / block_size), 1024)` |
| `context` | `runtime.cuda.Context` / `runtime.hip.Context` |
| `stream: Stream` | the stream `launch` enqueues on; default NULL stream |
| `Buffer`, `Stream`, `backend`, `available` | |

`launch` signatures:

| Kind | `launch(` |
| --- | --- |
| `map`, `mapIndexed` | `&buf, count, params)` |
| `mapTo` | `&in, &out, count, params)` |
| `zip` | `&a, &b, &out, count, params)` |
| `reduce` | `&data, count, &partials, blocks, params)`: one partial per block; you fold them |
| `gather` | `&src, &idx, &out, count = idx.len, bound = src.len)` |
| `scatter` | `&src, &idx, &out, count = src.len, bound = out.len)` |

A `count` of 0 returns without launching. Past `(2^31-1) * block_size`
elements, `Dim3.linear` panics.

### `g.AutoKernel(spec)`

`init() Self` (infallible; it warns when a present artifact fails to start),
`initStrict() Error!Self` (that failure becomes an error), `deinit()`,
`run(...)` and `selected() Backend`. The three backend handle types are
`Cpu`, `Cuda` and `Hip`, and `Cuda.available` / `Hip.available` tell you which
artifacts this build has. It is a `union(Backend)`, so a `switch` on it lets you
reach the GPU handle's `launch`.

### `g.RawKernel(entry_name, .cuda | .hip)` / `g.rawKernelByName(backend, name, ordinal)`

| Member | Notes |
| --- | --- |
| `init(ordinal)` | `RawKernel` only; `rawKernelByName` returns an already-open `RawByName(backend)` |
| `alloc(bytes)` | **bytes** |
| `allocPinned(bytes) ![]u8`, `freePinned(mem)` | page-locked host memory, so async copies are actually async |
| `createStream()`, `createStreamNonBlocking()`, `createEvent(timing: bool)` | the caller owns each and must `deinit` it |
| `hasGraphs()`, `hasEvents()` | false on old drivers; the calls then return `error.Unsupported` |
| `launch(grid, block, shared_bytes, args)` | NULL stream |
| `launchOn(&stream, grid, block, shared_bytes, args)` | |
| `synchronize()` | whole device |
| `Buffer`, `Stream`, `Graph`, `GraphExec`, `Event`, `CaptureMode` | types |

To build kernel arguments, call `buffer.argPtr()` for a buffer and
`g.interface.arg(&value)` for anything else, and pack params with
`g.abi.pack(P, p)`. The driver copies the argument values when you enqueue the
launch.

## Driver objects (`g.runtime.cuda` / `g.runtime.hip`: same shape)

**Context**: `init(ordinal)`, `deinit`, `makeCurrent`, `synchronize`, `alloc`,
`allocPinned`/`freePinned`, `loadModuleFromMemory(image)` (cached per device
and image), `createStream`, `createStreamNonBlocking`, `createEvent`,
`hasGraphs`, `hasEvents`, `fp64Ratio() !u32` (2 on datacenter parts, 64 on
consumer parts).

**Buffer**: `upload(host, n)`, `download(host, n)`, `uploadAt/downloadAt(host,
offset, n)`, `copyFrom(src, src_off, dst_off, n)`, `copyFromAsync(..., &stream)`,
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

`shutdown()` unloads every module and releases every context. Call it only
once nothing else will touch the device. `cuda.loadedModuleCount()` reports
how many (device, image) modules are JIT'd.

`g.runtime.dynamic.Compute` is one tagged-union facade over CUDA, HIP and a
`.cpu` arm that fails politely. It offers `init(?Backend)`, `initDevice`,
`alloc`, `loadModule`, `createStream`, `synchronize` and `fp64Ratio`. It
has no events or graphs; for those, use the vendor module.

## Errors

`g.Error` is one flat set: `InitFailed` (no driver/GPU), `NoDevice`,
`ContextFailed`, `SyncFailed` (usually an earlier launch faulted),
`AllocFailed`, `ModuleLoadFailed` (arch mismatch), `KernelNotFound`,
`LaunchFailed`, `CopyFailed`, `InvalidArgument`, `BackendUnavailable`,
`UnsupportedBackend` (reserved), `CaptureFailed`, `GraphFailed`,
`EventFailed` and `Unsupported` (driver too old).
`g.lastDriverError() -> { code: i64, backend: .none | .cuda | .hip }` is
thread-local and cleared on success.

## Geometry

`g.Dim3 = extern struct { x = 1, y = 1, z = 1 }`.
`Dim3.linear(n, block_x)` returns a zero grid for n = 0 and **panics** when
the grid is out of range. `Dim3.linearChecked(n, block_x) Error!Dim3` returns
an error instead, so use it for counts that come from outside.
`Dim3.max_grid_x = 2^31 - 1`.

## Device-side names

`g.is_device` (bool), `g.kernel_callconv`, `g.GlobalPtr(T)`
(`[*]addrspace(.global) T` on device), `g.globalIdX(block_size)`,
`g.exportRaw(name, &fn)` and `g.exportKernels(@This() or .{ specs })`. These
are the six names whose meaning differs between the roots. On the host
`globalIdX` is a compile error and both exports are no-ops.

`g.builtins` exists in the device compilation only:

| | |
| --- | --- |
| `globalIdX(bs)`, `localIdX()`, `blockIdX()` | thread indices; AMDGCN traps if a launch is wider than `bs` |
| `gridDimX()` | NVPTX only; on AMD, pass the stride as an argument |
| `barrier()` | `__syncthreads`, with the fences included on AMDGCN; every thread must reach it |
| `spinPause()` | ~64 ns backoff; a no-op below sm_70/PTX 6.3 |
| `loadAcquireDevice(p)`, `storeReleaseDevice(p, v)` | u32 at `.gpu` scope (cheaper than Zig's `.sys` on NVPTX) |

Device math (`g.math`, also the std-only module `dep.module("math")`): `exp
exp2 log log2 log10 log1p pow sin cos tan tanh sinh cosh expm1 atan sqrt
rsqrt`, for `f32`/`f64`/vectors; `pow(x, y: @TypeOf(x))`. Each is one body,
with no FMA or intrinsic, so it gives the same bits on host, NVPTX, AMDGCN and
at comptime. `math.max_ulp` holds the promised bounds: exp 0.52, log 0.52,
pow 0.55 (measured 0.505 / 0.502 / 0.502 against f128).

## ABI (`g.abi`)

`Boundary(T)` gives the wire type, `pack(T, v)` / `unpack(T, w)` convert, and
`assertStable(T)` is a comptime guard. A custom boundary declares all three of
`gpu_layout` (an `extern` or `packed` struct), `toGpu` and `fromGpu`.

## Build API (`@import("gompute")` in build.zig)

`emitKernels(b, dep, exe, EmitOptions)` is for one artifact-consuming
executable per dependency; a second call on the same `dep` panics.
`addKernels(b, dep, KernelsOptions) Kernels { kernels, gompute }` is for
several: import `k.gompute` in place of `dep.module("gompute")`.

`EmitOptions` fields:

| Field | Meaning |
| --- | --- |
| `kernels_root` | one root, named `"kernels"` |
| `kernel_roots: []KernelRoot` | several roots, compiled in parallel; each `KernelRoot` is `{ name, root, imports, imports_ctx, heavy }` |
| `imports: ?DeviceImportsFn`, `imports_ctx` | `fn (b, target, optimize, ctx) []Import`. A callback, because a module carries its own target |
| `heavy_lanes: u8 = 1` | how many `.heavy` roots may compile at once |
| `cuda`, `hip: { enabled = true, gpu = .auto \| .{ .name = "sm_89" }, optimize = null }` | per-backend device settings |
| `target`, `optimize` | default to the exe's |

`.auto` probes the **build** machine (`nvidia-smi`, `amdgpu-arch`), and when
it finds nothing it compiles that backend out with a warning. Pin `.name` for
CI, Docker, Nix and releases. Device `Debug` is promoted to `ReleaseFast`.

The generated `gompute_kernels` module provides `emitted`, `has_cuda`,
`has_hip`, `root_names`, `cuda_images`, `hip_images` and
`cuda_index`/`hip_index` (name → `{ blob, symbol }`).
