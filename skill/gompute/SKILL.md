---
name: gompute
description: Write, build and launch hand-written CUDA/HIP kernels from Zig with the Gompute library, and call its device math (same bits on CPU and GPU). Use when a Zig project imports `gompute`, when wiring `emitKernels`/`addKernels` into build.zig, writing a kernel with `g.exportRaw`/`g.GlobalPtr`, launching it with `RawKernel` or `rawKernelByName`, keeping data on the device, using streams, graphs, events, pinned memory or several devices, calling `g.math` in a kernel, or deciding what a Gompute feature costs at run time.
---

# Gompute

You write each GPU kernel by hand in Zig, in a kernels root that also
compiles for the host. At build time Gompute compiles that root to PTX and
HSACO and embeds them in your executable; at run time `g.RawKernel` loads an
entry by name and launches it through a thin, dlopen'd CUDA/HIP driver layer.
The CPU path is your own plain loop over the same element function.

Pinned to **Zig 0.18.0-dev.349+d924f83bf** (Zig master; the pin moves with
`build.zig.zon`). GPU backends work on Linux and Windows (macOS compiles them,
but there is no driver). wasm has no GPU backend. HIP compiles and links, but
**nothing has executed on AMD hardware yet**.

## Pick the right layer

Work down the table and stop at the first row that fits.

| You need | Use |
| --- | --- |
| The CPU path | a plain `for` loop calling the element function; LLVM vectorizes a concrete body. Hand-widen with `@Vector` only to reach `g.math`'s vector forms |
| A GPU kernel whose name you know at build time | `g.RawKernel(name, .cuda \| .hip)`, behind `if (comptime g.available(.cuda))` |
| A kernel picked at run time (config, netlist, flag) | `g.rawKernelByName(backend, name, ordinal)` |
| The same launch sequence repeated many times | capture `launchOn` calls on a stream into a graph |
| A module/blob you load yourself | `g.runtime.dynamic` / `.cuda` / `.hip` |

Measured costs are in [OVERHEAD.md](OVERHEAD.md).

## Quick start

Three files. Each one is required.

```zig
// src/kernels.zig -- compiled for the host (as a module) and once per GPU target
const g = @import("gompute");

/// The element function: plain Zig, so the CPU loop calls it too.
pub fn scaleRelu(x: f32, scale: f32) f32 {
    return @max(x * scale, 0);
}

fn scaleReluKernel(data: g.GlobalPtr(f32), len: u64, scale: f32) callconv(g.kernel_callconv) void {
    const i = g.globalIdX(256); // pass the block size you launch with
    if (i < len) data[i] = scaleRelu(data[i], scale);
}

comptime {
    if (g.is_device) g.exportRaw("scale_relu", &scaleReluKernel);
}
```

```zig
// build.zig
const gompute_build = @import("gompute");
// ...
const dep = b.dependency("gompute", .{ .target = target, .optimize = optimize });
// a "kernels" module from src/kernels.zig importing dep.module("gompute");
// the exe imports both "gompute" and "kernels"
exe.root_module.linkSystemLibrary("c", .{}); // required for CUDA/HIP (dlopen)
gompute_build.emitKernels(b, dep, exe, .{
    .kernels_root = b.path("src/kernels.zig"),
    .cuda = .{ .gpu = .{ .name = "sm_89" } },  // pin it; .auto probes the BUILD machine
    .hip = .{ .gpu = .{ .name = "gfx1100" } },
});
```

```zig
// src/main.zig
var data = [_]f32{ -1, 2, -3, 4 };
var scale: f32 = 2;
if (comptime g.available(.cuda)) gpu: {
    var kernel = g.RawKernel("scale_relu", .cuda).init(0) catch break :gpu; // no card: CPU
    defer kernel.deinit();
    var buf = try kernel.alloc(@sizeOf(@TypeOf(data))); // BYTES
    defer buf.free();
    try buf.upload(&data, @sizeOf(@TypeOf(data)));
    var len: u64 = data.len;
    try kernel.launch(g.Dim3.linear(data.len, 256), .{ .x = 256 }, 0, &.{
        buf.argPtr(), g.interface.arg(&len), g.interface.arg(&scale),
    });
    try kernel.synchronize(); // a device fault surfaces here
    try buf.download(&data, @sizeOf(@TypeOf(data)));
    return;
}
for (&data) |*x| x.* = kernels.scaleRelu(x.*, scale);
```

Install with `zig fetch --save git+https://github.com/OmarSiwy/Gompute.git`.
`examples/basic` in the repo is this program.

## Writing a kernels root

- An entry point is a function with `callconv(g.kernel_callconv)` whose
  buffers are `g.GlobalPtr(T)`, exported with
  `comptime { if (g.is_device) g.exportRaw("name", &fn); }`. Pass `&fn`, not
  `fn`. `name` must be unique across every kernel root.
- In a kernels root, name only `g.math`, `g.builtins` and the five switches
  (`is_device`, `kernel_callconv`, `GlobalPtr`, `globalIdX`, `exportRaw`).
  The device root has nothing else: no `Dim3`, `RawKernel` or `runtime`.
- `g.builtins` exists only in the device compilation. Name it only from
  kernels you export under `if (g.is_device)`, so the host never analyzes them.
- Pass each parameter as its own scalar argument. There is no struct
  marshalling: the launch's `args` is one `g.interface.arg(&x)` per parameter
  (`buf.argPtr()` for a buffer), in the kernel's parameter order and with its
  exact types. **Nothing checks the list against the signature.** A wrong
  count, order or width is a garbage argument, not an error.
- `g.interface.arg` does not copy. The pointee must be alive and unmoved
  until `launch`/`launchOn` returns (the driver copies the values then).
- Guard the tail: `if (i < len)`, with `len` an argument and the grid from
  `g.Dim3.linear(n, block)`. `Dim3.linear(0, b)` is a zero grid, which no
  driver launches, so return before launching when `n` can be 0.
- Inside a kernel, use **`g.math.*`** instead of `@exp`, `@log`, `@sin` and the
  rest: those builtins do not compile for NVPTX/AMDGCN. `g.math` covers `exp`,
  `exp2`, `log`, `log2`, `log10`, `log1p`, `pow`, `sin`, `cos`, `tan`, `tanh`,
  `sinh`, `cosh`, `expm1`, `atan`, `sqrt` and `rsqrt`, for `f32`, `f64` and
  `@Vector`s of either. CPU, GPU and comptime give the same bits. f32 `exp`,
  `log`, `log2`, `log10`, `sin`, `cos` are correctly rounded (all 2³² inputs);
  f64 `exp`/`log`/`pow` are faithful (≤0.505 ulp) and monotone. To use the
  math without the GPU layer, import `dep.module("math")`.

## Writing kernels that are fast on both sides

1. **Struct of arrays on the GPU.** One `g.GlobalPtr(T)` per field, never a
   pointer to `[]struct{a, b}`: separate buffers coalesce, an array of structs
   halves effective bandwidth.
2. **Branch-light bodies.** One thread per element means a data-dependent `if`
   diverges warps on the GPU and blocks vectorization on the CPU. Prefer
   `@select`/`@max`/`@min`.
3. **One element per thread is enough for a bandwidth-bound kernel.** The
   bench's plain kernels read 76–77% of the RTX 4060 Laptop's peak at 16M
   elements (OVERHEAD.md).
4. **CPU: leave a concrete loop to LLVM.** Zig 0.18's loop vectorizer widens
   and unrolls a plain loop over a concrete arithmetic body. It does not
   widen a call into `g.math` (table-driven): to get the vector forms, step
   the loop `@Vector(8, f32)` / `@Vector(4, f64)` at a time and pass the
   vector to `g.math` (lane-exact with the scalar call). On an i9-14900HX
   (Zig 0.18.0-dev.349) it pays for every f32 function except
   `expm1`/`log1p` (f32 `log` 5.4 → 2.1 ns/call, `tanh` 27.4 → 3.0), and for
   f64 `exp`, `exp2`, `log`, `log2`, `log10`, `atan` and `pow`. f64 `sin`,
   `cos`, `tan`, `tanh`, `sinh`, `cosh`, `expm1`, `log1p` gain nothing.
5. **GPU math rates.** f32 `exp`, `log`, `log2`, `log10`, `sin`, `cos`,
   `atan` run in f32 (67–181 Gcalls/s on an RTX 4060 Laptop, Zig 0.18.0-dev.349). Every other f32
   function computes in f64 to keep the bits identical (1.5–4 Gcalls/s; f64 is
   1/64 rate on consumer NVIDIA, `context.fp64Ratio()` tells you).
6. **No hard-coded warp width.** AMD is wave64 on GCN/CDNA and wave32 on
   RDNA. `g.builtins.shuffleXor` works in 32-lane groups on both; launch a
   block size that is a multiple of 32 and call it outside any branch that
   splits a group.

## GPU: keep data on the device

Allocate once, upload once, launch many times, synchronize and download once.
`launch` and `launchOn` allocate nothing, copy nothing and do not wait.

```zig
var kernel = try g.RawKernel("scale_relu", .cuda).init(0);
defer kernel.deinit();
var buf = try kernel.alloc(n * @sizeOf(f32));  // BYTES
defer buf.free();
try buf.upload(data.ptr, n * @sizeOf(f32));
var len: u64 = n;
const args = [_]g.interface.Arg{ buf.argPtr(), g.interface.arg(&len), g.interface.arg(&scale) };
for (0..steps) |_| try kernel.launch(g.Dim3.linear(n, 256), .{ .x = 256 }, 0, &args);
try kernel.synchronize();
try buf.download(data.ptr, n * @sizeOf(f32));
```

Every handle on one device shares one context and one JIT'd copy of each root, so a
`Buffer` from any handle is valid in all of them. For async copies, upload
from pinned memory (`kernel.allocPinned`, released with `kernel.freePinned`)
with `buf.uploadAtAsync`/`downloadAtAsync` on a stream from
`kernel.createStreamNonBlocking()`, and `launchOn` that stream.

## GPU: graph replay

```zig
if (!kernel.hasGraphs()) return plainLaunches(); // CUDA < 11.4
var stream = try kernel.createStreamNonBlocking();
defer stream.deinit();

try stream.beginCapture(.thread_local);
try kernel.launchOn(&stream, grid, .{ .x = 256 }, 0, &args);  // recorded, not run
try other_kernel_same_device.launchOn(&stream, ...);
var graph = try stream.endCapture();
defer graph.deinit();
var exec = try graph.instantiate();
defer exec.deinit();

for (0..iters) |_| try exec.launch(&stream);
try stream.synchronize();
```

- Argument values are **copied at capture**. To change them, recapture and
  call `exec.update(&new_graph)`. It returns `false` if the topology changed;
  in that case re-`instantiate`.
- Between `beginCapture` and `endCapture`, never call anything that waits on
  the host: `synchronize`, a blocking `upload`/`download`, `launch` on the
  NULL stream. Any of these invalidates the capture with
  `error.CaptureFailed`. For copies inside a graph, use
  `buf.uploadAtAsync`/`downloadAtAsync` with pinned memory.

## GPU: several devices

One handle per ordinal; every handle carries its device and makes it current
on the calling thread. To overlap N devices, queue async copies and launches
on one non-blocking stream per device, then wait:

```zig
const K = g.RawKernel("scale_relu", .cuda);
const count = try g.runtime.dynamic.deviceCount(.cuda); // GOMPUTE_FAKE_DEVICES=2 fakes a second one
for (0..count) |d| {
    handle[d] = try K.init(@intCast(d));
    stream[d] = try handle[d].createStreamNonBlocking();
    dev[d] = try handle[d].alloc(part * @sizeOf(f32));
    pinned[d] = try handle[d].allocPinned(part * @sizeOf(f32));
    // ... fill pinned[d] ...
    try dev[d].uploadAtAsync(pinned[d].ptr, 0, pinned[d].len, &stream[d]);
    try handle[d].launchOn(&stream[d], g.Dim3.linear(part, 256), .{ .x = 256 }, 0, &.{
        dev[d].argPtr(), g.interface.arg(&len), g.interface.arg(&scale),
    });
    try dev[d].downloadAtAsync(pinned[d].ptr, 0, pinned[d].len, &stream[d]);
}
for (0..count) |d| try stream[d].synchronize();
```

`buf.copyFrom(&other_device_buf, ...)` between two ordinals is a peer copy:
direct after `context.enablePeerAccess(&other.context)` (one direction per
call, idempotent), staged through host memory by the driver otherwise.
`canAccessPeer` says which; `enablePeerAccess` returns `error.Unsupported`
without a link, which a fake-device pair also answers. `context.info()` picks
devices: a real second card has its own `uuid`, a fake one the same.

## Shared memory, barriers, atomics

```zig
var scratch: [256 / 32]f32 addrspace(.shared) = undefined;

/// One partial sum per 256-thread block; `n` is a multiple of 256.
fn blockSum(data: g.GlobalPtr(f32), out: g.GlobalPtr(f32)) callconv(g.kernel_callconv) void {
    const tid = g.builtins.localIdX();
    var v = data[g.globalIdX(256)];
    inline for (.{ 16, 8, 4, 2, 1 }) |m| v += g.builtins.shuffleXor(f32, v, m);
    if (tid % 32 == 0) scratch[tid / 32] = v;
    g.builtins.barrier(); // every thread of the block must reach it
    if (tid < 32) {
        var w = if (tid < 256 / 32) scratch[tid] else 0;
        inline for (.{ 16, 8, 4, 2, 1 }) |m| w += g.builtins.shuffleXor(f32, w, m);
        if (tid == 0) out[g.builtins.blockIdX()] = w;
    }
}
```

`g.builtins` also has `gridDimX` (NVPTX only; on AMD pass the grid stride
as an argument), `spinPause`, and `loadAcquireDevice`/`storeReleaseDevice`.
On NVPTX, every `@atomicRmw` runs relaxed whatever ordering you pass. Order
through an acquire load or a release store instead. Pass the dynamic shared
size as `launch`'s `shared_bytes`; a static `addrspace(.shared)` array needs
0.

## Failure modes, in the order you will hit them

| Symptom | Cause | Fix |
| --- | --- | --- |
| `symbol not found: cuInit` | exe not linked with libc | `linkSystemLibrary("c", .{})` |
| Compile error on `RawKernel(name, .cuda)` | the build emitted no CUDA (`.auto` found no GPU, or disabled) | pin `.cuda.gpu.name`, or guard with `if (comptime g.available(.cuda))` |
| Compile error naming the kernel | missing `exportRaw`, or its root not passed to `emitKernels` | add both |
| `error.ModuleLoadFailed` + loud log | artifact built for a different arch | pin the arch you deploy to |
| Wrong numbers, no error | `args` disagree with the kernel's parameters (count, order, type), or point at a dead temporary | match the signature; keep argument storage alive past the launch |
| `error.SyncFailed` | the *previous* launch faulted | check `g.lastDriverError()` |
| Two executables ship each other's PTX | `emitKernels` twice on one dep (it panics now) | `addKernels`, one per exe |
| `error.Unsupported` from `enablePeerAccess` | no direct link between the two devices, or a fake-device pair | nothing: `copyFrom` still works, staged by the driver |

`g.lastDriverError()` returns the raw `CUresult`/`hipError_t` behind the last
failure on this thread. It is cleared on every success.

Full API: [REFERENCE.md](REFERENCE.md). Costs: [OVERHEAD.md](OVERHEAD.md).
