---
name: gompute
description: Write, build and launch one-dimensional GPU/CPU kernels with the Gompute Zig 0.17 library. Use when a Zig project imports `gompute`, when the user wants to run a map/zip/reduce/gather/scatter over a slice on CUDA, HIP or the CPU from one source, wire `emitKernels` into build.zig, hoist allocations out of `run`, capture launches into a CUDA/HIP graph, write a raw device kernel, or choose between `Kernel`, `AutoKernel`, `RawKernel` and `g.runtime`. Also use when deciding what a Gompute feature costs at run time.
---

# Gompute

You write a pure scalar function once. At compile time Gompute turns it into a
CPU loop (vectorized where it can be), a PTX entry point and an HSACO entry
point, all from the same source file. The backend is part of the type, so
`Kernel(spec, .cpu)` compiles to the loop you would have written yourself;
`zig build test` checks that against the emitted assembly.

Pinned to **Zig 0.17.0**. GPU backends work on Linux only (macOS compiles them,
but there is no driver). Windows and wasm are CPU-only.

## Pick the right layer

Work down the table and stop at the first row that fits. Each row down costs
more code and gives more control.

| You need | Use | Run-time cost over a hand-written loop/launch |
| --- | --- | --- |
| One elementwise/reduce op, data on the host, called occasionally | `g.Kernel(spec, b).run(...)` | CPU: none. GPU: copy in + launch + sync + copy out **every call** (device buffers are cached on the handle) |
| The same, but the deploy machine decides the backend | `g.AutoKernel(spec)` | One probe at `init`, then one tagged-union switch per `run` |
| Data stays on the device across many launches | `kernel.alloc` + `kernel.launch` | Launch only: three stack args, no allocation |
| The same launch sequence repeated many times | `kernel.stream` + capture into a graph | One driver call per replay of the whole sequence |
| Shared memory, barriers, atomics, your own ABI | `g.RawKernel(name, b)` | Whatever your kernel does |
| A module/blob you load yourself, no spec at all | `g.runtime.dynamic` / `.cuda` / `.hip` | Raw driver calls |

Measured costs are in [OVERHEAD.md](OVERHEAD.md). Read it before you put
`run()` in a loop.

## Quick start

Three files. Each one is required.

```zig
// src/kernels.zig -- compiled once for the host and once per GPU target
const g = @import("gompute");

pub const Params = struct { scale: f32 };

// `anytype` makes the body generic: the CPU backend runs it at SIMD width,
// and the GPU runs it at scalar width, one thread per element.
fn scaleRelu(x: anytype, p: Params) @TypeOf(x) {
    const V = @TypeOf(x);
    return @max(x * g.splat(V, p.scale), g.splat(V, @as(f32, 0)));
}
pub const scale_relu = g.map("scale_relu", f32, Params, scaleRelu, .{});

comptime { g.exportKernels(@This()); } // device side: emits one entry per pub spec
```

```zig
// build.zig
const gompute_build = @import("gompute");
// ...
const dep = b.dependency("gompute", .{ .target = target, .optimize = optimize });
// import dep.module("gompute") as "gompute" into BOTH the exe and the kernels module
exe.root_module.linkSystemLibrary("c", .{}); // required for CUDA/HIP (dlopen)
gompute_build.emitKernels(b, dep, exe, .{
    .kernels_root = b.path("src/kernels.zig"),
    .cuda = .{ .gpu = .{ .name = "sm_89" } },  // pin it; .auto probes the BUILD machine
    .hip = .{ .gpu = .{ .name = "gfx1100" } },
});
```

```zig
// src/main.zig
var kernel = try g.Kernel(kernels.scale_relu, .cuda).init(0); // or .cpu / .hip
defer kernel.deinit();
try kernel.run(&data, .{ .scale = 2 });
```

Install with `zig fetch --save git+https://github.com/OmarSiwy/Gompute.git`.

## The operation set

Every constructor returns a comptime **type** (a spec), even though the names
are lowercase. All of them take `.{ .block_size = N }` with N in 1…1024
(default 256). `name` must be a C identifier, and it must be unique across
every kernel root.

| Constructor | `run` arguments | CPU path |
| --- | --- | --- |
| `g.map(name, T, P, f, o)` | `(data: []T, p)`, in place | SIMD if `f` takes `anytype`, else scalar |
| `g.mapFn(name, f, o)` | same; `T` and `P` read off `f` | same (concrete `f` only) |
| `g.mapTo(name, In, Out, P, f, o)` | `(in: []const In, out: []Out, p)` | SIMD if `f` takes `anytype`, else scalar |
| `g.zip(name, A, B, Out, P, f, o)` | `(a, b, out, p)`, all the same length | SIMD if `f` takes `anytype`, else scalar |
| `g.mapIndexed(name, T, P, f, o)` | `(data: []T, p)`; `f(x, i: u64, p)` | SIMD if `f` takes `anytype` (`i` is then a vector of indices), else scalar |
| `g.reduce(name, T, P, combine, identity, o)` | `(data: []const T, p) !T` | scalar fold |
| `g.sum` `g.min` `g.max` `g.any` `g.all` `(name, T, P, o)` | `(data, p) !T` | 4-register vector accumulator, one `@reduce` at the end |
| `g.gather(name, T, Idx, o)` | `(src, idx, out)`: `out[i] = src[idx[i]]` | scalar, bounds-skipping |
| `g.scatter(name, T, Idx, o)` | `(src, idx, out)`: `out[idx[i]] = src[i]` | scalar, bounds-skipping |

"Scalar" means scalar machine code. **Zig 0.16 and 0.17 ship LLVM's loop vectorizer
disabled**, so a plain loop over a concrete `fn (f32, P) f32` is `vaddss`, one
element at a time, even at `ReleaseFast -mcpu=native`. A generic (`anytype`)
body is the only way to get SIMD on the CPU. When a generic `mapTo`/`zip`
body's result type differs from its input's, name it with
`g.Lanes(@TypeOf(x), Out)`.

The rules you can't see from the signatures:

- **`T`** is a fixed-width int (8/16/32/64-bit), `f16`, `f32` or `f64`. No `usize`.
- **`Params`** crosses to the device by value. It may hold ints, floats, bools,
  enums, arrays, vectors and nested structs of those. Pointers, slices,
  optionals and `usize` are compile errors that name the field's full path.
  `gpu_layout`/`toGpu`/`fromGpu` is the escape hatch (see REFERENCE.md).
- **`reduce`**: `combine` must be associative and `identity` must be its neutral
  element. A float sum differs from a left-to-right sum in the last bits, on
  the CPU as well, because the vector accumulator reassociates.
  `.pre = &f` is `transform_reduce` (sum of squares, count-if) in one pass.
- **`any`/`all`** are integer only. Write the predicate as `.pre` returning 0/1
  (`any`) or 0/all-ones (`all`).
- **`scatter`** with duplicate indices is nondeterministic on the GPU. An
  out-of-range index is skipped, never written.
- A length mismatch between paired slices is `error.InvalidArgument`, not a
  truncated run. An empty slice returns without launching.
- Inside a kernel, use **`g.math.*`** instead of `@exp`, `@log`, `@sin` and the
  rest: those builtins do not compile for NVPTX/AMDGCN. `g.math` covers `exp`,
  `exp2`, `log`, `log2`, `log10`, `log1p`, `pow`, `sin`, `cos`, `tan`, `tanh`,
  `sinh`, `cosh`, `expm1`, `atan`, `sqrt` and `rsqrt`, for `f32`, `f64` and
  `@Vector`s of either. **Each is one body on every target**, so CPU and GPU
  give the same bits, and comptime folds to them as well. `exp`/`log`/`pow`
  are faithful (≤0.505 ulp) and monotone. To use the math without the GPU
  layer, import `dep.module("math")`.

Compose ops without an intermediate buffer using `g.Fused(T, P, .{ OpA, OpB })`
(each op is a struct with `pub inline fn eval(T, P) T`, or a function lifted
with `g.Unary`). Pass the result's `eval` as the map function. You get one
loop and one kernel.

## Writing kernels that are fast on both sides

1. **Make every elementwise body generic** (`x: anytype`, `g.splat` for
   scalars), whether it's a `map`, `mapTo`, `zip` or `mapIndexed`. A concrete
   function is correct but scalar on the CPU: measured, generic `zip` is 2.8×
   faster and a generic `mapTo` calling `exp` is 2.45× faster.
2. **Prefer a preset over a custom `reduce`.** Presets get the vector
   accumulator (3.6× faster than per-chunk `@reduce` on 64K f32). A custom
   `combine` stays scalar on the CPU.
3. **Use `zip`, not an array of structs.** Two separate buffers coalesce on
   the GPU; `[]struct{a, b}` halves effective bandwidth.
4. **Fuse, don't chain.** Two `map`s cost two launches and two passes over
   memory. A `Fused` costs one of each.
5. **Branch-light bodies.** One thread per element means a data-dependent `if`
   diverges warps on the GPU and blocks SIMD on the CPU. Prefer
   `@select`/`@max`/`@min`.
6. **`g.math` is SIMD at vector width**: exp/exp2/log/log2/log10/pow at f32
   and f64, plus f32 sin/cos/tanh/sinh/cosh, each lane-exact with the scalar.
   Against glibc's scalar libm that's 2–4× faster at vector width, and the
   scalar forms are on par. f64 sin/cos/tanh run one lane at a time. On the
   GPU, f32 math computes in f64 to keep the bits identical, so expect f64
   throughput (1/64 rate on consumer NVIDIA).

## GPU: keep data on the device

```zig
var kernel = try g.Kernel(kernels.scale_relu, .cuda).init(0);
defer kernel.deinit();
var buf = try kernel.alloc(n);            // ELEMENTS
defer buf.free();
try buf.upload(data.ptr, n * @sizeOf(f32)); // BYTES
for (0..steps) |_| try kernel.launch(&buf, n, .{ .scale = 2 }); // no sync, no copy
try kernel.context.synchronize();         // a device fault surfaces here
try buf.download(data.ptr, n * @sizeOf(f32));
```

The `launch` signature depends on the kind: `map`/`mapIndexed`
`(&buf, n, p)`; `mapTo` `(&in, &out, n, p)`; `zip` `(&a, &b, &out, n, p)`;
`reduce` `(&data, n, &partials, blocks, p)`, after which you download
`blocks` partials and fold them yourself (`reduceBlocks(n)` sizes it, at
most 1024); `gather`/`scatter` `(&src, &idx, &out, n, bound)`.
`AutoKernel` has only `run`. For `launch`, name the backend.

## GPU: graph replay

```zig
kernel.stream = try kernel.context.createStreamNonBlocking();
defer kernel.stream.deinit();
if (!kernel.context.hasGraphs()) return plainLaunches(); // CUDA < 11.4

try kernel.stream.beginCapture(.thread_local);
try kernel.launch(&buf, n, p);       // recorded, not run
try other_kernel_same_device.launch(...);  // set its .stream too
var graph = try kernel.stream.endCapture();
defer graph.deinit();
var exec = try graph.instantiate();
defer exec.deinit();

for (0..iters) |_| try exec.launch(&kernel.stream);
try kernel.stream.synchronize();
```

- `params` and lengths are **copied at capture**. To change them, recapture
  and call `exec.update(&new_graph)`. It returns `false` if the topology
  changed; in that case re-`instantiate`.
- Between `beginCapture` and `endCapture`, never call anything that waits on
  the host: `run`, `synchronize`, a blocking `upload`/`download`. Any of these
  invalidates the capture with `error.CaptureFailed`. For async copies, use
  `buf.uploadAtAsync`/`downloadAtAsync` with pinned memory
  (`kernel.context.allocPinned`).
- `deinit` clears `kernel.stream` but does not destroy it. You own the stream.

## Raw kernels

When a pure scalar function can't express the kernel: shared memory,
barriers, atomics, a spin-wait, or a custom argument list.

```zig
fn rawAdd(data: g.GlobalPtr(f32), len: u64) callconv(g.kernel_callconv) void {
    const i = g.globalIdX(256);      // MUST equal the block.x you launch with
    if (i < len) data[i] += 1;
}
comptime { if (g.is_device) g.exportRaw("raw_add", &rawAdd); }
```

Load it with `g.RawKernel("raw_add", .cuda).init(0)`, then call
`raw.launch(g.Dim3.linear(n, 256), .{ .x = 256 }, 0, &args)` or `raw.launchOn(&stream, ...)`.
A `globalIdX(256)` launched with a different block size is correct on CUDA and
wrong on HIP. Device intrinsics (`localIdX`, `blockIdX`, `barrier`,
`spinPause`, `loadAcquireDevice`/`storeReleaseDevice`) live in `g.builtins`,
which exists only in the device compilation, so guard uses with
`if (g.is_device)`. On NVPTX, every `@atomicRmw` runs relaxed whatever
ordering you pass. Order through an acquire load or a release store instead.

## Failure modes, in the order you will hit them

| Symptom | Cause | Fix |
| --- | --- | --- |
| `symbol not found: cuInit` | exe not linked with libc | `linkSystemLibrary("c", .{})` |
| Compile error on `Kernel(spec, .cuda)` | the build emitted no CUDA (`.auto` found no GPU, or disabled) | pin `.cuda.gpu.name`, or use `AutoKernel` |
| `error.ModuleLoadFailed` + loud log | artifact built for a different arch | pin the arch you deploy to |
| `KernelNotFound` / compile error naming the kernel | missing `exportKernels(@This())` or root not passed to `emitKernels` | add both |
| Silently 100× slower | `AutoKernel` fell back on a broken artifact | `initStrict()`, assert `selected()` |
| `error.SyncFailed` | the *previous* launch faulted | check `g.lastDriverError()` |
| Two executables ship each other's PTX | `emitKernels` twice on one dep (it panics now) | `addKernels`, one per exe |

`g.lastDriverError()` returns the raw `CUresult`/`hipError_t` behind the last
failure on this thread. It is cleared on every success.

Full API: [REFERENCE.md](REFERENCE.md). Costs: [OVERHEAD.md](OVERHEAD.md).
