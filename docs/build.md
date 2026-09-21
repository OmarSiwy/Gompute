# Build integration

Everything beyond the single `emitKernels` call in the [guide](guide.html).

## Pin `.auto` off the build machine

`.auto` runs `nvidia-smi` / `amdgpu-arch` **on the machine running
`zig build`**, not on the machine that will run the binary. When the probe
finds nothing, the backend is compiled out and the build still succeeds. The
binary can then never use a GPU, and you find out in production.

Gompute warns when this happens. Treat the warning as an error in any build
whose output leaves the machine:

- CI, Docker and Nix hosts almost never have a GPU, so always pin
  `.cuda = .{ .gpu = .{ .name = "sm_89" } }` (or the `gfx*` you target).
- For releases, pin, or you ship whatever card the release runner happened to
  have.
- If the build is genuinely CPU-only, set `.enabled = false` rather than relying
  on detection failing.

`.auto` is also why the Nix dev shell is not hermetic: `nvidia-smi` comes from
the ambient `PATH`, so the same source tree can produce different artifacts on
two machines. The flake's `packages.default` and `checks.default` never call
`emitKernels`; anything of yours that does must pin the GPU.

## One executable per dependency, or `addKernels`

`emitKernels` attaches `gompute_kernels` to `dep.module("gompute")`, which is
**shared**. Two executables calling `emitKernels` against the same `dep` do not
get one artifact set each: the last call would overwrite the first, the build
would stay green, and the first executable would ship the second one's PTX and
fail at run time with `error.KernelNotFound`. Gompute panics on the second call
rather than letting that happen.

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

`emitKernels` stays supported for the single-executable case.

## Kernel roots that import their own modules

If `kernels.zig` imports anything besides `gompute`, supply an `.imports`
callback. Gompute calls it once per enabled backend with that backend's
resolved device target and optimize mode; build every module, including nested
ones, from the values it hands you:

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
optimize mode, which is how a `Debug` import ends up inside an otherwise
`ReleaseFast` device build.

## Several kernel roots

One kernel root is one `zig build-obj` per backend: no parallelism, and an edit
to any kernel recompiles all of them. `.kernel_roots` splits that into one
sub-compilation per root, which the build runner schedules on its thread pool
and caches independently.

```zig
gompute_build.emitKernels(b, dep, exe, .{
    .kernel_roots = &.{
        .{ .name = "hisim", .root = b.path("src/hisim.zig"), .heavy = true },
        .{ .name = "bsimsoi", .root = b.path("src/bsimsoi.zig"), .heavy = true },
        .{ .name = "vbic", .root = b.path("src/vbic.zig") },
    },
    .heavy_lanes = 2,
});
```

Each root may carry its own `.imports`/`.imports_ctx`. `.kernels_root` still
works and may be combined with `.kernel_roots`; it becomes a root named
`"kernels"`. `addKernels` takes the same two fields.

`.heavy = true` marks a root whose device compilation is big enough that running
it next to the other big ones costs memory rather than saving time. Heavy roots
are chained into `heavy_lanes` serial lanes with ordinary build-graph edges;
light roots run unconstrained.

Kernel names must be unique across roots, because the name is what run-time
dispatch looks up. A duplicate is a compile error naming both roots.

Measured on eight generated roots shaped like real device models, one of which
was edited:

| | wall | recompiled |
| --- | --- | --- |
| 8 roots | **1.05 s** | 7 of 8 cached; only the edited root |
| same kernels, 1 root | **11.22 s** | everything |

Cold builds went 12.3 s → 8.2 s at `heavy_lanes = 1` and → 6.5 s at
`heavy_lanes = 2`. The cold speedup is bounded by the largest root, so it is
well under the root count whenever the cost is skewed.

## Picking a kernel at run time

The set of kernels is closed at build time, but which one a given run launches
need not be. `rawKernelByName` looks a name up in a `StaticStringMap` built at
compile time from every root's name table, then loads only the blob that holds
it:

```zig
const model = netlist.deviceModel();          // decided at run time
var kernel = g.rawKernelByName(.cuda, model, 0) catch |err| switch (err) {
    error.KernelNotFound => return reportUnknownModel(model),
    else => return err,
};
defer kernel.deinit();
try kernel.launch(grid, block, 0, &args);
```

An unknown name is `error.KernelNotFound`, not a panic. The comptime paths
(`Kernel(spec, .cuda)` and `RawKernel(name, .cuda)`) resolve through the same
map at compile time, so a kernel that is in no root is still a compile error.

Either way only the root that exports the kernel is JIT'd, and the runtime
caches it per `(device, blob)`, so a process that touches 3 of 37 models pays
for 3.

## Device optimize mode

`Debug` device code drags `std.builtin` panic globals into the module, and
LLVM's NVPTX backend emits invalid PTX types (`.u2`/`.u4`/`.u5`) for them. So a
`Debug` build compiles device code at `ReleaseFast`, which removes the panic
machinery outright rather than merely optimizing it.

`ReleaseSafe` would keep it, and is pathologically expensive: it turns a large
straight-line kernel into a CFG with a panic edge per operation and something
in the LLVM pipeline goes superlinear. Measured over 37 Verilog-A device models
on `nvptx64-cuda`/`sm_89`:

| model | Debug | ReleaseSafe | ReleaseFast |
| --- | --- | --- | --- |
| `hisimhv_va` | 4.2 s | **443.1 s** | 13.6 s |
| `bsim4va` | 0.9 s | **38.1 s** | 3.4 s |
| `hicumL2_va` | 0.4 s | **17.1 s** | 2.4 s |

Override per backend with `.cuda = .{ .optimize = ... }` if you need to.

## Artifact pipeline

Run once per kernel root; the per-root name tables are then merged into one
comptime name -> `(blob, symbol)` map in the generated `gompute_kernels` module.

### CUDA

1. Cross-compile `kernels.zig` to NVPTX LLVM IR.
2. Run `tools/kernel_ir_tool.zig`.
3. Delete Zig 0.16's exported alias lines.
4. Rename each internal `ptx_kernel` definition to its requested entry name.
5. Assemble the rewritten IR to PTX with Zig's bundled LLVM tools.
6. Embed the PTX in the host executable, with a compile-time name map.

This implements the workaround for Zig 0.16's NVPTX alias form, which NVPTX
code generation rejects.

### HIP

1. Cross-compile `kernels.zig` to an AMDGCN object and LLVM IR.
2. Link the object into an HSA code object with Zig's bundled `ld.lld`.
3. Generate a Zig compile-time map from public names to AMDGPU metadata names.
4. Embed both the code object and the name map.

The map lets `hipModuleGetFunction` request the internal kernel name recorded in
AMDGPU metadata while the public API continues to use `scale_relu`.

Both backends consult their name map at compile time, so a kernel you launch
but never exported is a compile error naming it, not a runtime
`error.KernelNotFound`.
