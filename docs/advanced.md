# Advanced

## Graph replay

A graph records a sequence of launches once and replays it with one driver
call. Capture needs a non-blocking stream, since the NULL stream cannot be
captured; launch onto it with `launchOn`:

```zig
var raw = try g.RawKernel("scale_relu", .cuda).init(0);
defer raw.deinit();

var stream = try raw.createStreamNonBlocking();
defer stream.deinit();
var len: u64 = count;
var scale: f32 = 2;
const args = [_]g.interface.Arg{ buffer.argPtr(), g.interface.arg(&len), g.interface.arg(&scale) };

try stream.beginCapture(.thread_local);
try raw.launchOn(&stream, g.Dim3.linear(count, 256), .{ .x = 256 }, 0, &args);
try raw.launchOn(&stream, g.Dim3.linear(count, 256), .{ .x = 256 }, 0, &args);
var graph = try stream.endCapture();
defer graph.deinit();
var exec = try graph.instantiate();
defer exec.deinit();

for (0..1000) |_| try exec.launch(&stream); // one driver call per replay
try stream.synchronize();
```

Capture records, it does not run: the buffer is untouched until the first
replay. Arguments are copied at capture time, so a replay always runs with the
values you captured; `exec.update(&new_graph)` swaps them in place when only
the arguments changed, and returns `false` when the topology did. Nothing that
waits on the host may run between `beginCapture` and `endCapture`:
`synchronize` or a blocking `upload` invalidates the capture.
`raw.hasGraphs()` is false on a CUDA driver older than 11.4; every graph call
then returns `error.Unsupported`.

## Device builtins

`g.globalIdX` reads the block size the kernel was launched with on both
vendors (`%ntid.x`, and on AMD the dispatch packet), so any `block` works. Its
argument is ignored; it stays because kernels written against git abd75e4
(the last Zig 0.17 build) pass one.

`g.builtins` has the rest of the device intrinsics. It exists only in the
device compilation, so name it under `if (g.is_device)` or inside a kernel
only the device compilation analyzes.

- `localIdX()` is the thread's index within its block, `blockIdX()` the
  block's index within the grid.
- `barrier()` is a block-wide execution barrier and memory fence,
  `__syncthreads`. Every thread in the block must reach it, or NVIDIA hangs
  and AMD is undefined.
- `shuffleXor(T, v, mask)` swaps a value with the lane `mask` away inside a
  32-lane group, `__shfl_xor_sync` on NVIDIA and DPP / `ds_swizzle` /
  `v_permlanex16` on AMD. Every lane of the group must reach it, so launch
  with a block that is a multiple of 32. Groups are 32 lanes on both wave
  sizes, so nothing depends on AMD's wave width.
- `gridDimX()` exists on NVPTX only, since Zig cannot read AMD's grid size.
  Pass the stride as a kernel argument instead.
- `spinPause()` backs off inside a spin loop. It compiles to nothing on NVPTX
  targets below sm_70 with PTX 6.3.

## Atomics

For atomics, use Zig's `@atomicRmw`, `@atomicLoad` and `@atomicStore` on global
pointers. One caveat: on NVPTX every `@atomicRmw` runs relaxed, whatever
ordering you pass. Order through an acquire load or a release store instead.
`g.builtins.loadAcquireDevice` and `storeReleaseDevice` are those two at
device scope rather than Zig's system scope, which on NVPTX skips the
system-scope fence.

## Picking a kernel at run time

If the kernel to launch is named by a config file rather than by your source,
`g.rawKernelByName(.cuda, name, 0)` resolves it against the same compiled set
and returns `error.KernelNotFound` for a name that is not there. See
[Build integration](build.html#picking-a-kernel-at-run-time).

## Vectorized CPU loops

The CPU path is your own loop over the element function, and LLVM's loop
vectorizer, on in Zig 0.18, widens a concrete one. It cannot widen a loop
that calls `g.math`: the table-driven bodies have no loop of their own to
widen. Every `g.math` function also takes a `@Vector`, each lane the scalar
call's bits, so widen that loop by hand:

```zig
const w = std.simd.suggestVectorLength(f32) orelse 1;
var i: usize = 0;
while (i + w <= xs.len) : (i += w) {
    const v: @Vector(w, f32) = xs[i..][0..w].*;
    xs[i..][0..w].* = g.math.exp(v);
}
for (xs[i..]) |*x| x.* = g.math.exp(x.*);
```

On an i9-14900HX (Zig 0.18.0-dev.349, `examples/bench` with `BENCH=math`, 8
lanes for f32), f32 `log` went from 5.43 to 2.10 ns per call and `sin` from
7.40 to 2.64. Which functions have a real SIMD body is in
[Device math](reference.html#device-math).
