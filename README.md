# Gompute

A Zig **0.16.0** library for defining a one-dimensional GPU/CPU kernel once and
specializing the whole abstraction at compile time. CUDA (PTX), HIP (HSACO) and
native CPU backends.

The backend is part of the type, and the CPU path compiles down to the loop you
would have written by hand. That is checked on every build, not just claimed:
`zig build test` compares the emitted assembly and fails on a difference.

**📖 [Documentation](https://omarsiwy.github.io/Gompute/)** ·
[Guide](https://omarsiwy.github.io/Gompute/guide.html) ·
[Build integration](https://omarsiwy.github.io/Gompute/build.html) ·
[Reference](https://omarsiwy.github.io/Gompute/reference.html) ·
[API](https://omarsiwy.github.io/Gompute/api/)

I am aware, as the zig library evolves that this library will need to be updated
greatly. Hence, I'll attempt to maintain the interface, only ADDING features, rather
than removing any features.

## Install

```sh
zig fetch --save git+https://github.com/OmarSiwy/Gompute.git
```

Requires Zig 0.16.0 exactly. Linux and macOS; Windows and wasm are CPU-only and
the GPU backends do not compile there. The CUDA/HIP backends need
`exe.root_module.linkSystemLibrary("c", .{})` — the drivers are `dlopen`'d.

## Example

```zig
// src/kernels.zig
const g = @import("gompute");

pub const Params = struct { scale: f32 };

fn scaleRelu(x: f32, p: Params) f32 {
    const y = x * p.scale;
    return if (y > 0) y else 0;
}

pub const scale_relu = g.map("scale_relu", f32, Params, scaleRelu, .{});

comptime { g.exportKernels(@This()); }
```

```zig
// src/main.zig
var data = [_]f32{ -1, 2, -3, 4 };

var kernel = try g.Kernel(kernels.scale_relu, .cpu).init(0);
defer kernel.deinit();

try kernel.run(&data, .{ .scale = 2 });
// data == .{ 0, 4, 0, 8 }
```

Change `.cpu` to `.cuda` or `.hip` without touching the call site, or use
`AutoKernel` to probe at run time. One `emitKernels` call in your `build.zig`
wires up the device compilation — see the
**[guide](https://omarsiwy.github.io/Gompute/guide.html)**.

## Commands

```sh
zig build test    # library tests, including the codegen comparison
zig build docs    # API reference -> zig-out/docs

cd examples/basic     && zig build run   # CPU path
cd examples/exhaustive && zig build run  # GPU, needs libc + a driver
```

## Documentation

The site under [`docs/`](docs/) is the full documentation; it is published to
GitHub Pages on every push to `main`, together with the generated API
reference.

| Page                               |                                                                                                   |
| ---------------------------------- | ------------------------------------------------------------------------------------------------- |
| [Guide](docs/guide.md)             | Install, defining kernels, the build call, selecting a backend                                    |
| [Build integration](docs/build.md) | `.auto` pinning, multiple executables, kernel roots with their own imports, the artifact pipeline |
| [Reference](docs/reference.md)     | Platform support, behaviour, the generated ABI, device math, backend status                       |
| [Advanced](docs/advanced.md)       | Allocation-free launches, fusion, vectorized CPU kernels, raw kernels, custom ABIs                |

## License

MIT. See [LICENSE](LICENSE).
