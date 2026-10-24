# Runtime

`Kernel` and `AutoKernel` are one policy over a smaller layer: `g.runtime`.
Reach for the runtime directly when you want something the map API does not
express: streams, buffer-to-buffer copies, partial transfers, or launching a
[raw kernel](advanced.html) whose ABI is your own.

Nothing here is required for ordinary use. If `run` fits, use `run`.

- `g.runtime.dynamic` is one tagged-union facade over both vendors, plus a `.cpu`
  arm that fails politely. Pick this unless you have a reason not to.
- `g.runtime.cuda` and `g.runtime.hip` are the vendor APIs, unwrapped. Same
  shape, same method names, no union tag to switch on.

## The dynamic facade

```zig
const rt = g.runtime.dynamic;

// null means "probe": CUDA, then HIP, then CPU. Pass .cuda to demand one.
var gpu = try rt.Compute.init(null);
defer gpu.deinit();

std.debug.print("running on {t}\n", .{gpu.backend});
```

| | |
| --- | --- |
| `Compute.init(preferred: ?Backend) Error!Compute` | Probe, or demand a specific backend. Falls back to `.cpu`. |
| `Compute.initDevice(preferred: ?Backend, ordinal: c_int) Error!Compute` | The same, on a chosen device. |
| `compute.backend` | `.cuda`, `.hip`, or `.cpu`. Check it before doing anything device-shaped. |
| `compute.alloc(bytes: usize) Error!Buffer` | **Bytes**, not elements. `error.AllocFailed` on the `.cpu` arm. |
| `compute.loadModule(image: [:0]const u8) Error!Module` | PTX or HSACO. Cached per (device, image), so the same blob JITs once per process. |
| `compute.createStream() Error!Stream` | |
| `compute.synchronize() Error!void` | Block until the device is idle. |
| `compute.info() Error!DeviceInfo` | Name, uuid, PCI address, SMs or compute units, compute capability, FP64 ratio, clocks, bus width, memory. `error.Unsupported` on `.cpu`. |
| `compute.fp64Ratio()`, `compute.multiprocessorCount()`, `compute.computeCapability()` | `Error!u32` each. `error.Unsupported` on `.cpu`, and on a HIP runtime older than ROCm 6.0. |
| `compute.canAccessPeer(&other) Error!bool`, `compute.enablePeerAccess(&other) Error!void` | A direct link to another device's memory; see [Several devices](#several-devices). |
| `compute.deinit() void` | Releases your handle only. |
| `rt.deviceCount(backend) Error!u32` | The ordinals `initDevice` accepts; 0 on `.cpu`. |
| `rt.shutdown()` | Unload every module, release every context. Process teardown, not handle teardown. |

## Getting at the compiled kernels

`emitKernels` generates a module named `gompute_kernels` and wires it into your
`gompute` import. It is what `Kernel.init` reads, and you can read it too:

```zig
pub const emitted: bool;                        // was emitKernels called at all
pub const has_cuda: bool;
pub const has_hip: bool;
pub const root_names: []const []const u8;       // kernel roots, in blob order
pub const Entry = struct { blob: u16, symbol: [:0]const u8 };
pub const cuda_images: []const [:0]const u8;    // one PTX blob per root
pub const hip_images: []const [:0]const u8;     // one HSACO blob per root
pub const cuda_index: std.StaticStringMap(Entry);
pub const hip_index: std.StaticStringMap(Entry);
```

One image per kernel root, so the index maps a kernel name to the blob that
holds it and the symbol inside it. Since Zig 0.17 the symbol is always the
kernel name; under 0.16 HIP kept Zig's mangled one.

```zig
const artifacts = @import("gompute_kernels");

const entry = artifacts.cuda_index.get("scale_relu") orelse return error.KernelNotFound;
var module = try gpu.loadModule(artifacts.cuda_images[entry.blob]);
defer module.deinit();

var kernel = try module.getKernel(entry.symbol);
```

## A launch, by hand

The generated entry point for a `map` takes `(data, len, params)`. Arguments
are passed the way both drivers want them: an array of pointers to the storage
holding each value, so every argument needs a variable with an address.

```zig
const N: usize = 1024;
var buffer = try gpu.alloc(N * @sizeOf(f32));
defer buffer.free();
try buffer.upload(@ptrCast(&host_data), N * @sizeOf(f32));

var len: u64 = N;
var params = g.abi.pack(kernels.Params, .{ .scale = 2 });
var args = [_]g.interface.Arg{
    buffer.argPtr(),
    g.interface.arg(&len),
    g.interface.arg(&params),
};

try kernel.launch(g.Dim3.linear(N, 256), .{ .x = 256 }, 0, &args);
try gpu.synchronize();
try buffer.download(@ptrCast(&host_data), N * @sizeOf(f32));
```

The block size in the launch must match the spec's `block_size`. For generated
kernels that is `.{ .block_size = 256 }` in the spec, 256 here. For raw
kernels, see the warning in [Advanced](advanced.html): the two backends disagree
about which one wins.

`launch` does not synchronize. Nothing tells you a kernel faulted until the
next `synchronize`, which is where you will see `error.SyncFailed` for a fault
that happened one launch earlier.

## Buffers

| | |
| --- | --- |
| `upload(host: *const anyopaque, n: usize) Error!void` | Host → device, `n` bytes from offset 0. |
| `download(host: *anyopaque, n: usize) Error!void` | Device → host. |
| `uploadAt(host, offset: usize, n: usize) Error!void` | The same, starting `offset` **bytes** into the buffer. |
| `downloadAt(host, offset: usize, n: usize) Error!void` | |
| `copyFrom(src: *const Buffer, src_offset, dst_offset, n) Error!void` | Device → device, no host round trip of yours. Between buffers on two ordinals it is a peer copy: over the direct link after `enablePeerAccess`, staged by the driver otherwise. |
| `copyFromAsync(src, src_offset, dst_offset, n, &stream) Error!void` | The same, stream-ordered and not waited on. |
| `argPtr() Arg` | The buffer as a kernel argument. |
| `deviceAddr() u64` | The raw device pointer, for kernels that want it as a scalar. |
| `free() void` | Idempotent. Use-after-free is *not* checked. |

Transfers are not bounds-checked against `Buffer.bytes` in process; an overrun
comes back from the driver as `error.CopyFailed`.

## Streams

```zig
var stream = try gpu.createStream();
defer stream.deinit();

try kernel.launchOnStream(grid, block, 0, &args, &stream);
try stream.synchronize();
```

A stream orders the work put on it and runs independently of work on other
streams. Two streams let a copy overlap a launch. `compute.synchronize()`
waits for the whole device; `stream.synchronize()` waits for just that queue.

The facade has no events and no graphs. `g.runtime.cuda` and `g.runtime.hip`
both do, with the same shape: `Event`, `Stream.waitEvent`, `beginCapture` /
`endCapture`, `Graph.instantiate`. See [Graph replay](advanced.html#graph-replay).

## Several devices

Every handle carries the ordinal it was made on and makes that device's
context current on the calling thread, so several devices are several handles,
and nothing else changes:

- `rt.deviceCount(backend)` says how many ordinals there are.
  `GOMPUTE_FAKE_DEVICES=k` makes ordinals up to `k-1` extra contexts on device
  0, so a two-device program runs on one card; `Context.info` tells a real
  second card (its own `uuid`) from a fake one (`fake`, the same `uuid`).
- `Compute.initDevice(backend, ordinal)` or `Kernel(spec, .cuda).init(ordinal)`
  opens one handle per device.
- `run` synchronizes, so launching through it on N devices serializes them.
  To overlap, give each device a non-blocking stream and pinned host memory,
  queue the upload, the launch and the download on each stream, then wait for
  every stream:

```zig
const n = try rt.deviceCount(.cuda);
var handle: [16]g.Kernel(kernels.scale_relu, .cuda) = undefined;
var stream: [16]g.runtime.cuda.Stream = undefined;
var dev: [16]g.runtime.cuda.Buffer = undefined;
var pinned: [16][]u8 = undefined;
const part = data.len / n;
for (0..n) |d| {
    handle[d] = try g.Kernel(kernels.scale_relu, .cuda).init(@intCast(d));
    stream[d] = try handle[d].context.createStreamNonBlocking();
    handle[d].stream = stream[d];
    dev[d] = try handle[d].alloc(part);
    pinned[d] = try handle[d].context.allocPinned(part * @sizeOf(f32));
    @memcpy(pinned[d], std.mem.sliceAsBytes(data[d * part ..][0..part]));
    try dev[d].uploadAtAsync(pinned[d].ptr, 0, pinned[d].len, &stream[d]);
    try handle[d].launch(&dev[d], part, .{ .scale = 2 });
    try dev[d].downloadAtAsync(pinned[d].ptr, 0, pinned[d].len, &stream[d]);
}
for (0..n) |d| try stream[d].synchronize();
```

- A `copyFrom` between buffers on two ordinals is a peer copy. After
  `context.enablePeerAccess(&other.context)` it goes over the direct link
  (NVLink, PCIe P2P, XGMI); without one the driver stages it through host
  memory, which is correct and slower. `canAccessPeer` says which you will
  get; `enablePeerAccess` returns `error.Unsupported` when there is no link,
  which is also what a fake-device pair answers. Enable it in both directions
  if both devices read each other's buffers.
- Events order streams across devices: record on one device's stream, wait on
  the other's.

## Lifetime

Contexts and modules are process-wide and per device, and they are shared:

- Two `Compute`s on one GPU get the same primary context, so a buffer allocated
  through one is valid in the other.
- Two `loadModule` calls with the same image on the same device JIT once.
- `deinit` on a `Compute`, `Module`, or `Kernel` releases *your handle*. It does
  not unload the module or release the context, because something else may
  still be holding one.

`rt.shutdown()` (or `g.runtime.cuda.shutdown()` / `g.runtime.hip.shutdown()`) is
the real teardown. Call it once, at exit, if you care, and only once nothing
else is going to touch the device.

Contexts are made current per thread on first use, so the runtime is safe to
call from several threads. Buffers are not: two threads writing one buffer race
exactly as you would expect.

## Loading a kernel by name at run time

If the kernel to run comes from a config file or argv rather than from source,
`rawKernelByName` resolves it against the same compiled set:

```zig
var kernel = try g.rawKernelByName(.cuda, name_from_config, 0);
defer kernel.deinit();
```

The set of kernels is still closed at build time, since everything was compiled
from your kernel roots. Only the *choice* is deferred, and an unknown name is
`error.KernelNotFound` rather than a panic. The returned handle is
`g.RawByName(.cuda)`, which has the same `alloc` / `launch` / `synchronize` as
[`RawKernel`](advanced.html) but no `init`, since it is already initialized.
