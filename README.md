# Gompute

A Zig library for GPU kernels written in Zig: you write the kernel, the build
compiles it to PTX (CUDA) and HSACO (HIP) and embeds both in your executable, a
thin runtime loads the driver at run time and launches it, and `g.math` gives
device-safe math with the same bits on the host and the GPU.

[Documentation](https://omarsiwy.github.io/Gompute/) ·
[Guide](https://omarsiwy.github.io/Gompute/guide.html) ·
[Build integration](https://omarsiwy.github.io/Gompute/build.html) ·
[Reference](https://omarsiwy.github.io/Gompute/reference.html) ·
[API](https://omarsiwy.github.io/Gompute/api/)

I am aware that, as the zig library evolves, this library will need to be updated
greatly. Hence, I'll attempt to maintain the interface, only ADDING features, rather
than removing any features. 2.0.0 is the exception: it removed the generated
kernel layer (`map`, `zip`, `reduce`, `Kernel`, `AutoKernel` and the rest),
because Zig 0.18's loop vectorizer makes a plain CPU loop as fast as its
generated one.

## Install

```sh
zig fetch --save git+https://github.com/OmarSiwy/Gompute.git
```

Requires Zig 0.18.0-dev.349+d924f83bf (Zig master, pinned; the `0.17.0-v1.4.0` tag is
the last Zig 0.17 build). Builds on Linux, Windows, macOS and wasm. The CUDA/HIP drivers are loaded at run time (`dlopen` / `LoadLibraryA`); on
a machine without one `init` returns `error.InitFailed` and your code takes its
CPU path. See [Platform support](docs/reference.md#platform-support).
The GPU backends need `exe.root_module.linkSystemLibrary("c", .{})`.

## Example

```zig
// src/kernels.zig
const g = @import("gompute");

/// Plain Zig, so the CPU path calls it too.
pub fn scaleRelu(x: f32, scale: f32) f32 {
    return @max(x * scale, 0);
}

fn scaleReluKernel(data: g.GlobalPtr(f32), len: u64, scale: f32) callconv(g.kernel_callconv) void {
    const i = g.globalIdX(256);
    if (i < len) data[i] = scaleRelu(data[i], scale);
}

comptime {
    if (g.is_device) g.exportRaw("scale_relu", &scaleReluKernel);
}
```

```zig
// src/main.zig
var data = [_]f32{ -1, 2, -3, 4 };
var scale: f32 = 2;

if (comptime g.available(.cuda)) gpu: {
    var kernel = g.RawKernel("scale_relu", .cuda).init(0) catch break :gpu;
    defer kernel.deinit();
    var buf = try kernel.alloc(@sizeOf(@TypeOf(data)));
    defer buf.free();
    try buf.upload(&data, @sizeOf(@TypeOf(data)));
    var len: u64 = data.len;
    try kernel.launch(g.Dim3.linear(data.len, 256), .{ .x = 256 }, 0, &.{
        buf.argPtr(), g.interface.arg(&len), g.interface.arg(&scale),
    });
    try kernel.synchronize();
    try buf.download(&data, @sizeOf(@TypeOf(data)));
    return; // data == .{ 0, 4, 0, 8 }
}
// No CUDA artifacts in this build, or no card here: a plain loop.
for (&data) |*x| x.* = kernels.scaleRelu(x.*, scale);
```

`g.available(.cuda)` is comptime: a build that emitted no CUDA artifacts never
compiles the GPU branch. One `emitKernels` call in your `build.zig` wires up
the device compilation; the
[guide](https://omarsiwy.github.io/Gompute/guide.html) walks through it.

## Commands

```sh
zig build test    # library tests and the device probes
zig build docs    # documentation site -> zig-out/docs
zig build docs -Dopen   # ...and open it in a browser

cd examples/basic     && zig build run   # GPU if present, else the CPU loop
cd examples/exhaustive && zig build run  # GPU, needs libc + a driver
cd examples/bench && zig build run -Doptimize=ReleaseFast   # the numbers in skill/gompute/OVERHEAD.md
```

## Documentation

The site under [`docs/`](docs/) is the full documentation; it is published to
GitHub Pages on every push to `main`, together with the generated API
reference.

| Page                               |                                                                                                   |
| ---------------------------------- | ------------------------------------------------------------------------------------------------- |
| [Guide](docs/guide.md)             | Install, writing a kernel, the build call, launching it, the CPU path                             |
| [Build integration](docs/build.md) | `.auto` pinning, multiple executables, kernel roots with their own imports, the artifact pipeline |
| [Reference](docs/reference.md)     | Platform support, behaviour, errors, `RawKernel`, kernel arguments, device math, backend status   |
| [Advanced](docs/advanced.md)       | Graph replay, device builtins, atomics, vectorized CPU loops over `g.math`                        |

## License

MIT. See [LICENSE](LICENSE).
