# Gompute

Gompute is a Zig **0.16.0** library for defining a one-dimensional GPU/CPU
kernel once and specializing the whole abstraction at compile time.

I am aware, as the zig library evolves that this library will need to be updated
greatly. Hence, I'll attempt to maintain the interface, only ADDING features, rather
than removing any features.

Dependencies:

- LLVM
- libC

No WASM support.

## Quick start

### 1. Add the dependency

```sh
zig fetch --save git+https://github.com/OmarSiwy/Gompute.git
```

### 2. Define shared kernels

```zig
// src/kernels.zig
const g = @import("gompute");

pub const Params = struct {
    scale: f32,
};

fn scaleRelu(x: f32, p: Params) f32 {
    const y = x * p.scale;
    return if (y > 0) y else 0;
}

pub const scale_relu = g.map(
    "scale_relu",
    f32,
    Params,
    scaleRelu,
    .{ .block_size = 256 },
);

comptime {
    g.exportKernels(.{scale_relu});
}
```

`kernels.zig` is compiled twice: once as an ordinary host module and once for
each requested GPU target. The public source is shared; target-specific
address spaces and thread indexing stay inside Gompute's generated entry point.

### 3. Add one build call

```zig
// build.zig
const std = @import("std");
const gompute_build = @import("gompute");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const dep = b.dependency("gompute", .{
        .target = target,
        .optimize = optimize,
    });

    const kernels_mod = b.createModule(.{
        .root_source_file = b.path("src/kernels.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "gompute", .module = dep.module("gompute") },
        },
    });

    const exe = b.addExecutable(.{
        .name = "app",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "gompute", .module = dep.module("gompute") },
                .{ .name = "kernels", .module = kernels_mod },
            },
        }),
    });

    gompute_build.emitKernels(b, dep, exe, .{
        .kernels_root = b.path("src/kernels.zig"),
        // .gpu defaults to .auto: the build detects the GPU on this machine
        // and disables a backend when its GPU is absent. Pin explicitly with:
        // .cuda = .{ .gpu = .{ .name = "sm_80" } },
        // .hip = .{ .gpu = .{ .name = "gfx1030" } },
    });

    b.installArtifact(exe);
}
```

Backends can be omitted from the artifact graph:

```zig
.cuda = .{ .enabled = false },
.hip = .{ .enabled = false },
```

Call `emitKernels` once per Gompute dependency instance. It attaches a private
`gompute_kernels` module to the host module and embeds all emitted blobs.

### Kernel roots that import their own modules

If `kernels.zig` imports anything besides `gompute`, supply an `.imports`
callback. Gompute calls it once per enabled backend with that backend's
resolved device target and optimize mode; build every module — including
nested ones — from the values it hands you:

```zig
fn deviceImports(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    _: ?*anyopaque,
) []const std.Build.Module.Import {
    const contract = b.createModule(.{
        .root_source_file = b.path("src/contract.zig"),
        .target = target,
        .optimize = optimize,
    });
    const models = b.createModule(.{
        .root_source_file = b.path("src/models.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "contract", .module = contract }},
    });
    const out = b.allocator.alloc(std.Build.Module.Import, 2) catch @panic("OOM");
    out[0] = .{ .name = "models", .module = models };
    out[1] = .{ .name = "contract", .module = contract };
    return out;
}

gompute_build.emitKernels(b, dep, exe, .{
    .kernels_root = b.path("src/kernels.zig"),
    .imports = &deviceImports,
});
```

It is a callback rather than a plain list of modules because a
`std.Build.Module` carries its own target and optimize mode. One prebuilt
module cannot serve both the `nvptx64` and `amdgcn` compilations, and a
host-built module dragged into device code would silently keep the host's
optimize mode — which is how a `Debug` import ends up inside an otherwise
`ReleaseFast` device build.

### 4. Select a backend

```zig
const g = @import("gompute");
const kernels = @import("kernels");

pub fn main() !void {
    var data = [_]f32{ -1, 2, -3, 4 };

    // The backend is part of the type. This is a direct inlined CPU loop.
    var kernel = try g.Kernel(kernels.scale_relu, .cpu).init(0);
    defer kernel.deinit();

    try kernel.run(&data, .{ .scale = 2 });
    // data == .{ 0, 4, 0, 8 }
}
```

Change `.cpu` to `.cuda` or `.hip` without changing the call site:

```zig
var kernel = try g.Kernel(kernels.scale_relu, .cuda).init(0);
defer kernel.deinit();
try kernel.run(&data, .{ .scale = 2 });
```

For runtime probing and CPU fallback:

```zig
var kernel = g.AutoKernel(kernels.scale_relu).init();
defer kernel.deinit();
try kernel.run(&data, .{ .scale = 2 });
```

`AutoKernel` probes CUDA, then HIP, once during initialization. Its `run` method
has one tagged-union switch. Use a fixed backend when zero runtime dispatch is
the requirement.

## Allocation-free GPU launch path

`run` is intentionally convenient, not magical. Reuse a device buffer to avoid
allocating and copying on every launch:

```zig
var kernel = try g.Kernel(kernels.scale_relu, .cuda).init(0);
defer kernel.deinit();

var buffer = try kernel.alloc(data.len);
defer buffer.free();

try buffer.upload(data.ptr, data.len * @sizeOf(f32));
try kernel.launch(&buffer, data.len, .{ .scale = 2 });
try kernel.context.synchronize();
try buffer.download(data.ptr, data.len * @sizeOf(f32));
```

`launch` creates only three stack-resident argument values: device pointer,
length, and the generated extern parameter struct.

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

## Generated ABI

`Params` may be a normal Zig struct. At compile time, Gompute derives a matching
`extern struct`, recursively packs it on the host, and unpacks it in device
code.

Supported generated fields:

- fixed-width integers: 8, 16, 32, or 64 bits;
- `f16`, `f32`, and `f64`;
- booleans, represented as `u8` across the boundary;
- enums with a supported fixed-width tag;
- arrays, vectors, and nested structs composed from those types.

Rejected by default:

- `usize` and `isize`, because target pointer widths may differ;
- pointers, slices, optionals, error unions, allocators, and opaque handles;
- non-byte-sized integers and extended host-only floats.

A type can declare `gpu_layout`, `toGpu`, and `fromGpu` for an explicit custom
boundary. This is the escape hatch for manually decomposing a higher-level
host value into a stable wire representation.

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
try raw.launch(.{ .x = grid_x }, .{ .x = 256 }, 0, &args);
try raw.synchronize();
```

For shared memory, barriers, multidimensional indexing, textures, or a custom
ABI, use a raw kernel and the low-level runtime modules.

## Artifact pipeline

### CUDA

1. Cross-compile `kernels.zig` to NVPTX LLVM IR.
2. Run `tools/kernel_ir_tool.zig`.
3. Delete Zig 0.16's exported alias lines.
4. Rename each internal `ptx_kernel` definition to its requested entry name.
5. Assemble the rewritten IR to PTX with Zig's bundled LLVM tools.
6. Embed the PTX in the host executable.

This implements the workaround for Zig 0.16's NVPTX alias form, which NVPTX
code generation rejects.

### HIP

1. Cross-compile `kernels.zig` to an AMDGCN object and LLVM IR.
2. Link the object into an HAS code object with Zig's bundled `ld.lld`.
3. Generate a Zig compile-time map from public names to AMDGPU metadata names.
4. Embed both the HAS code object and name map.

The map lets `hipModuleGetFunction` request the internal kernel name recorded in
AMDGPU metadata while the public API continues to use `scale_relu`.

## Backend status

| Backend | Generated `map` path |              Host runtime | Validation in this package                     |
| ------- | -------------------: | ------------------------: | ---------------------------------------------- |
| CPU     |                  Yes |                Native Zig | Executed and codegen-compared                  |
| CUDA    |             Yes, PTX | Runtime-loaded driver API | Cross-compiled; PTX entries inspected          |
| HIP     |           Yes, HSACO |    Runtime-loaded HIP API | Cross-compiled; ELF symbols/metadata inspected |

## Commands

```sh
# Library tests
zig build test

# Basic example (CPU path)
cd examples/basic
zig build run

# Exhaustive example (GPU — requires libc + CUDA/HIP driver)
cd examples/exhaustive
zig build run
```
