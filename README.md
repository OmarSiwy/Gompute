# Gompute

Gompute is a Zig **0.16.0** library for defining a one-dimensional GPU/CPU
kernel once and specializing the whole abstraction at compile time.

I am aware, as the zig library evolves that this library will need to be updated
greatly. Hence, I'll attempt to maintain the interface, only ADDING features, rather
than removing any features.

## Versions

Gompute pins one Zig release at a time; Zig 0.16 moved `std.Build`,
`std.DynLib` and the NVPTX backend under it.

| Gompute | Zig    |
| ------- | ------ |
| 0.1.x   | 0.16.0 |

`minimum_zig_version` in `build.zig.zon` is the same value. Nothing older or
newer is supported.

## Dependencies

- **LLVM** — supplied by Zig itself; the PTX/HSACO steps call `zig cc` and
  `zig ld.lld`. Nothing to install.
- **libc, linked into your executable** — `exe.root_module.linkSystemLibrary("c", .{})`
  is **required for the CUDA and HIP backends**. Without libc, Zig 0.16's
  `std.DynLib` resolves to `ElfDynLib` instead of `DlDynLib`; it opens
  `libcuda.so` but cannot resolve symbols out of it, and you get
  `cuda: symbol not found: cuInit` at run time, which looks exactly like a
  driver mismatch and is not one. CPU-only builds do not need it.
- **A CUDA or HIP driver at run time**, if you use those backends. Both are
  `dlopen`'d; neither is needed to build.

## Platform support

| Platform            | CPU backend | CUDA / HIP backends                |
| ------------------- | ----------- | ---------------------------------- |
| Linux               | Yes         | Yes                                |
| macOS               | Yes         | Compiles, but no such driver exists |
| `wasm32-wasi`       | Yes         | **Does not compile**                |
| Windows             | Yes         | **Does not compile**                |

Zig 0.16's `std.DynLib` only has an implementation for Linux and the
Darwin/BSD family; every other target hits `@compileError("unsupported
platform")`. Gompute `dlopen`s the CUDA and HIP drivers, so on Windows and
wasm anything that instantiates `Kernel(spec, .cuda)`, `Kernel(spec, .hip)`,
`RawKernel` or `AutoKernel` fails to compile (≈29 errors, all from
`std/dynamic_library.zig`). The `nvcuda.dll` entry in `src/runtime/cuda.zig`
is aspirational; **Windows is not supported.**

A program that only ever names `.cpu` compiles and runs on all four —
`emitKernels` already skips the GPU sub-compilations on wasm. `AutoKernel`
does *not* count as CPU-only: it instantiates all three backends.

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

    // Required for the CUDA/HIP backends. Without libc, std.DynLib is
    // ElfDynLib, which opens libcuda.so but cannot resolve symbols from it:
    // you get "cuda: symbol not found: cuInit" at run time. Harmless if you
    // only ever use .cpu.
    exe.root_module.linkSystemLibrary("c", .{});

    gompute_build.emitKernels(b, dep, exe, .{
        .kernels_root = b.path("src/kernels.zig"),
        .target = target,
        .optimize = optimize,
        // .gpu defaults to .auto: the build probes the GPU in THIS machine and
        // disables a backend when its GPU is absent. Pin it for CI, Docker,
        // Nix and releases -- see "Pin .auto off the build machine" below.
        // .cuda = .{ .gpu = .{ .name = "sm_89" } },
        // .hip = .{ .gpu = .{ .name = "gfx1100" } },
    });

    b.installArtifact(exe);
}
```

### CPU-only builds

Turn both GPU backends off to drop the PTX/HSACO sub-compilations from the
graph entirely. This is the supported way to force CPU-only, and it also
silences the `.auto` detection warning:

```zig
gompute_build.emitKernels(b, dep, exe, .{
    .kernels_root = b.path("src/kernels.zig"),
    .cuda = .{ .enabled = false },
    .hip = .{ .enabled = false },
});
```

`Kernel(spec, .cpu)` and `AutoKernel` keep working; `Kernel(spec, .cuda)`
returns `error.BackendUnavailable`.

### Pin `.auto` off the build machine

`.auto` runs `nvidia-smi` / `amdgpu-arch` **on the machine running
`zig build`**, not on the machine that will run the binary. When the probe
finds nothing, the backend is compiled out and the build still succeeds — the
binary can then never use a GPU, and you find out in production.

Gompute now warns when this happens. Treat the warning as an error in any
build whose output leaves the machine:

- **CI, Docker, Nix** — build hosts almost never have a GPU. Always pin
  `.cuda = .{ .gpu = .{ .name = "sm_89" } }` (or the `gfx*` you target).
- **Releases** — pin, or you ship whatever card the release runner happened
  to have.
- **Genuinely CPU-only** — set `.enabled = false` rather than relying on
  detection failing.

`.auto` is also why the Nix dev shell is not hermetic: `nvidia-smi` comes from
the ambient `PATH`, so the same source tree can produce different artifacts on
two machines. The flake's `packages.default` and `checks.default` never call
`emitKernels`; anything of yours that does must pin the GPU.

### One executable per dependency, or `addKernels`

`emitKernels` attaches `gompute_kernels` to `dep.module("gompute")`, which is
**shared**. Two executables calling `emitKernels` against the same `dep` do not
get one artifact set each — the last call overwrites the first, the build is
green, and the first executable ships the second one's PTX and fails at run
time with `error.KernelNotFound`.

For two or more executables, use `addKernels`, which returns a private
`gompute` module carrying only that root's artifacts:

```zig
const k = gompute_build.addKernels(b, dep, .{
    .root_source_file = b.path("src/kernels_a.zig"),
    .target = target,
    .optimize = optimize,
});

const kernels_mod = b.createModule(.{
    .root_source_file = b.path("src/kernels_a.zig"),
    .target = target,
    .optimize = optimize,
    // note: k.gompute, not dep.module("gompute")
    .imports = &.{.{ .name = "gompute", .module = k.gompute }},
});

const exe = b.addExecutable(.{
    .name = "a",
    .root_module = b.createModule(.{
        .root_source_file = b.path("src/main_a.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "gompute", .module = k.gompute },
            .{ .name = "kernels", .module = kernels_mod },
        },
    }),
});
exe.root_module.linkSystemLibrary("c", .{});
b.installArtifact(exe);
```

Repeat verbatim for the second executable with its own kernel root; the two
instances do not collide. `k.kernels` is the generated artifact module
(`has_cuda`, `has_hip`, the blobs) if you want to read it directly.

`emitKernels` is unchanged and stays supported for the single-executable case.

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
| CPU     |                  Yes |                Native Zig | Executed by `zig build test` and the examples  |
| CUDA    |             Yes, PTX | Runtime-loaded driver API | Cross-compiled; PTX entries inspected          |
| HIP     |           Yes, HSACO |    Runtime-loaded HIP API | Cross-compiled; ELF symbols/metadata inspected |

`tests/codegen.zig` is only *compiled* (`addObject`, `-fno-emit-bin`) by
`zig build test`. It proves the specializations type-check and instantiate; it
is not linked, not run, and no emitted code is diffed against a reference.
There is no codegen comparison in this package.

## Commands

```sh
# Library tests
zig build test

# API documentation -> zig-out/docs
zig build docs

# Basic example (CPU path)
cd examples/basic
zig build run

# Exhaustive example (GPU — requires libc + CUDA/HIP driver)
cd examples/exhaustive
zig build run
```

With Nix:

```sh
nix build        # generated API docs
nix flake check  # zig build test
nix develop      # Zig + ROCm/CUDA library paths
```
