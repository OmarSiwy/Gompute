# Advanced

## Allocation-free GPU launch path

`run` is intentionally convenient, not magical: it uploads, launches,
synchronizes and downloads on every call. Its device buffers are kept on the
handle and grown on demand, so only the first call (or a bigger one) pays for
an allocation, and `deinit` frees them. When the data can stay on the device,
skip the copies too:

```zig
var data = [_]f32{ -1, 2, -3, 4 };
const bytes = data.len * @sizeOf(f32);

var kernel = try g.Kernel(kernels.scale_relu, .cuda).init(0);
defer kernel.deinit();

var buffer = try kernel.alloc(data.len);
defer buffer.free();

try buffer.upload(&data, bytes);
try kernel.launch(&buffer, data.len, .{ .scale = 2 });
try kernel.context.synchronize();
try buffer.download(&data, bytes);
```

`launch` creates only three stack-resident argument values: device pointer,
length, and the generated extern parameter struct.

`kernel.alloc` takes a count of **elements**; `Buffer` methods take **bytes**.
`kernel.context` is the underlying `runtime.cuda.Context` / `runtime.hip.Context`,
which is how you wait for a bare `launch`. `launch` does not synchronize, and a
device-side fault will not surface until you do. For partial transfers,
device-to-device copies and streams, see [Runtime](runtime.html).

## Graph replay

`kernel.stream` is the stream `launch` enqueues on; it defaults to the NULL
stream, which cannot be captured. Point it at a non-blocking stream and the same
`launch` calls are recorded into a graph instead of run:

```zig
kernel.stream = try kernel.context.createStreamNonBlocking();
defer kernel.stream.deinit();

try kernel.stream.beginCapture(.thread_local);
try kernel.launch(&buffer, data.len, .{ .scale = 2 });
try kernel.launch(&buffer, data.len, .{ .scale = 2 });
var graph = try kernel.stream.endCapture();
defer graph.deinit();
var exec = try graph.instantiate();
defer exec.deinit();

for (0..1000) |_| try exec.launch(&kernel.stream); // one driver call per replay
try kernel.stream.synchronize();
```

Arguments, `params` included, are copied at capture time, so a replay always
runs with the values you captured; `exec.update(&new_graph)` swaps them in place
when only the parameters changed. Nothing that waits on the host may run
between `beginCapture` and `endCapture` -- `run`, `synchronize`, or a blocking
`upload` invalidate the capture. `kernel.context.hasGraphs()` is false on a
CUDA driver older than 11.4; every graph call then returns `error.Unsupported`.

## Compile-time fusion

Each operation type exposes `eval`. `Fused` expands an `inline for`, keeping
intermediate values in the same scalar expression/register chain:

```zig
const Scale = struct {
    pub inline fn eval(x: f32, p: Params) f32 {
        return x * p.scale;
    }
};

const Square = struct {
    pub inline fn eval(x: f32, _: Params) f32 {
        return x * x;
    }
};

const Pipeline = g.Fused(f32, Params, .{ Scale, Square });

fn scaleSquare(x: f32, p: Params) f32 {
    return Pipeline.eval(x, p);
}
```

Putting `scaleSquare` in one `map` spec emits one CPU loop and one GPU kernel,
not two launches and not an intermediate buffer.

## Vectorized CPU kernels

The CPU backend runs a generic map function at the target's vector width. Write
the operation over `anytype` and use `g.splat` so the same source compiles at
both scalar and vector width. `x * p.scale` is illegal when `x` is a vector and
`p.scale` is an `f32`:

```zig
fn scaleRelu(x: anytype, p: Params) @TypeOf(x) {
    const V = @TypeOf(x);
    return @max(x * g.splat(V, p.scale), g.splat(V, @as(f32, 0)));
}

pub const scale_relu = g.map("scale_relu", f32, Params, scaleRelu, .{});
```

The GPU path still instantiates it at scalar type, one thread per element, so
one definition serves both. A non-generic `fn (T, Params) T` keeps the plain
scalar loop, unchanged.

## Hand-written raw kernels

The same artifact pipeline can carry manually written CUDA/HIP kernels.
`GlobalPtr` and `kernel_callconv` deliberately change between host and device
compilations so the source file remains parseable on both targets:

```zig
const g = @import("gompute");

fn rawAdd(data: g.GlobalPtr(f32), len: u64) callconv(g.kernel_callconv) void {
    const i = g.globalIdX(256);
    if (i < len) data[i] += 1;
}

comptime {
    if (g.is_device) g.exportRaw("raw_add", &rawAdd);
}
```

Load the embedded entry with a fixed raw handle:

```zig
var raw = try g.RawKernel("raw_add", .cuda).init(0);
defer raw.deinit();

var len: u64 = count;
var args = [_]g.interface.Arg{
    buffer.argPtr(),
    g.interface.arg(&len),
};
try raw.launch(g.Dim3.linear(count, 256), .{ .x = 256 }, 0, &args);
try raw.synchronize();
```

The block size passed to `globalIdX` **must** match the block size you launch
with. Generated `map` kernels guarantee that by construction; raw kernels do
not, and the mismatch is asymmetric: NVPTX reads the real block dimension and
ignores the argument, while AMDGCN uses it. So the same code can be correct on
CUDA and silently wrong on HIP.

For shared memory, barriers, multidimensional indexing, textures, or a custom
ABI, use a raw kernel and the low-level [runtime](runtime.html) modules under
`g.runtime`.

Inside a raw kernel, `g.builtins` has the rest of the device intrinsics:
`localIdX()` for the thread's index within its block, `blockIdX()`, and
`barrier()` for a block-wide execution barrier and shared-memory fence. Every
thread in the block must reach that barrier, or NVIDIA hangs and AMD is
undefined.
`gridDimX()` exists on NVPTX only, since AMDGCN has no portable way to read it;
pass the stride as a kernel argument instead, which is what the generated
`reduce` kernels do. `spinPause()` backs off inside a spin loop; it compiles to
nothing on NVPTX targets below sm_70 with PTX 6.3. `g.builtins` is only present
in the device compilation.

For atomics, use Zig's `@atomicRmw`, `@atomicLoad` and `@atomicStore` on global
pointers. One caveat: on NVPTX every read-modify-write runs relaxed, whatever
ordering you pass. Order through an acquire load or a release store instead.

If the kernel to launch is named by a config file rather than by your source,
`g.rawKernelByName(.cuda, name, 0)` resolves it against the same compiled set
and returns `error.KernelNotFound` for a name that is not there.

## Custom ABI boundaries

When a `Params` field has no stable GPU representation, decompose it yourself:

```zig
const Handle = struct {
    id: u64,
    gain: f64,

    pub const gpu_layout = extern struct { id_lo: u32, id_hi: u32, gain: f32 };

    pub fn toGpu(v: @This()) gpu_layout {
        return .{
            .id_lo = @truncate(v.id),
            .id_hi = @truncate(v.id >> 32),
            .gain = @floatCast(v.gain),
        };
    }
    pub fn fromGpu(w: gpu_layout) @This() {
        return .{ .id = @as(u64, w.id_hi) << 32 | w.id_lo, .gain = w.gain };
    }
};
```

All three declarations are required together, and `gpu_layout` must be `extern`
or `packed`.
