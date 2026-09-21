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
| `compute.deinit() void` | Releases your handle only. |
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
holds it and the symbol inside it. The symbol is not always the kernel name,
because HIP keeps Zig's mangled name, which is why the map stores both.

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
| `copyFrom(src: *const Buffer, src_offset, dst_offset, n) Error!void` | Device → device, no host round trip. |
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

There is no event API and no cross-stream dependency mechanism. If you need
those, use `g.runtime.cuda` directly. The facade only carries what both vendors
spell the same way.

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
