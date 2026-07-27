# Reference

## Versions

Gompute pins one Zig release at a time; Zig 0.16 moved `std.Build`,
`std.DynLib` and the NVPTX backend under it.

| Gompute | Zig    |
| ------- | ------ |
| 1.0.x   | 0.16.0 |

`minimum_zig_version` in `build.zig.zon` is the same value. Nothing older or
newer is supported.

## Platform support

| Platform            | CPU backend | CUDA / HIP backends                 |
| ------------------- | ----------- | ----------------------------------- |
| Linux               | Yes         | Yes                                 |
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

## Behaviour worth knowing

Things that are easy to get wrong because nothing in the API says them out loud.

| | |
| --- | --- |
| `init(0)` | The CUDA/HIP **device ordinal**. Ignored by `.cpu`. |
| `run` | Allocates a device buffer, uploads, launches, **synchronizes**, downloads, frees — every call. Convenient, not cheap; see [Advanced](advanced.html) to hoist it. |
| `deinit` | Always safe, and safe to call twice. On `.cpu` it is a no-op. It releases *your handle*, not the shared device state. |
| Contexts and modules | Shared process-wide, per device. Two `Kernel`s on one GPU share one CUDA primary context and JIT the artifact once, so buffers allocated through one are usable by the other. `runtime.cuda.shutdown()` is the only real teardown. |
| Threads | Safe. A context is made current per thread on first use. |
| `Buffer` | `free()` is idempotent. Use-after-free is not checked. `upload`/`download` do not bounds-check against `Buffer.bytes` — the driver catches the overrun and you get `error.CopyFailed`. |
| Errors | `Error` has 12 members. `lastDriverError()` returns the raw CUDA/HIP code behind the last failure, which is the only way to tell "no driver" from "out of memory". It is cleared on success. |
| `block_size` | Must be 1…1024. Not required to be a multiple of the warp/wave size, but anything else wastes part of every warp. |
| Empty input | `run` on a zero-length slice returns without launching. |

## Naming

Two pairs are easy to confuse:

- **`exportKernels`** goes in your *kernels* file and names the entry points to
  emit. **`emitKernels`** goes in your *build.zig* and wires up the
  compilation. One word apart, different files, and `@import("gompute")` means
  a different module in each — the host module in `build.zig`, the device
  module inside `kernels.zig`. Forgetting either produces a different error.
- **`map` returns a comptime *type***, not a value, despite the lowercase name.
  `pub const scale_relu = g.map(...)` binds a type, which is why
  `Kernel(kernels.scale_relu, .cpu)` takes it as a parameter. `mapFn` is the
  same thing with `T` and `Params` inferred from the function.

`Unary` is `Fused` with a single operation; it exists so a bare `fn (T, Params) T`
can be dropped into a `Fused` pipeline alongside op structs.

## Operations

Every constructor below returns a spec; `g.Kernel(spec, backend)` /
`g.AutoKernel(spec)` give it a `run` whose arguments are the operation's
buffers, positionally. All of them take `.{ .block_size = 256 }`-style options.

| Constructor | `run` | Body |
| --- | --- | --- |
| `g.map(name, T, P, fn (T, P) T, o)` | `run(data: []T, p)` | in place |
| `g.mapTo(name, In, Out, P, fn (In, P) Out, o)` | `run(in: []const In, out: []Out, p)` | out of place, type may change |
| `g.zip(name, A, B, Out, P, fn (A, B, P) Out, o)` | `run(a: []const A, b: []const B, out: []Out, p)` | two inputs, separate buffers |
| `g.mapIndexed(name, T, P, fn (T, u64, P) T, o)` | `run(data: []T, p)` | value plus linear index |
| `g.reduce(name, T, P, combine, identity, o)` | `run(data: []const T, p) !T` | whole-buffer fold |
| `g.sum` / `g.min` / `g.max` / `g.any` / `g.all` `(name, T, P, o)` | `run(data: []const T, p) !T` | `reduce` presets |
| `g.gather(name, T, Idx, o)` | `run(src, idx, out)` | `out[i] = src[idx[i]]` |
| `g.scatter(name, T, Idx, o)` | `run(src, idx, out)` | `out[idx[i]] = src[i]` |

`zip` exists so `c = a + b` does not force you to pack the operands into one
`[]struct { a: f32, b: f32 }`. That is array-of-structs, and it interleaves two
streams that each want to be coalesced on their own.

### Reduce

`combine` must be associative and `identity` must be its neutral element: the
device reassociates across threads and blocks, so the answer to a
non-associative op changes with the launch geometry. Float addition is only
approximately associative — expect the last bits to differ from a strict
left-to-right CPU sum.

`.pre` turns a reduce into Thrust's `transform_reduce` — sum of squares, L2
norm, "how many match", "does any exceed k", each in one launch:

```zig
fn square(x: f64, _: Params) f64 { return x * x; }
pub const l2sq = g.sum("l2sq", f64, Params, .{ .pre = &square });
```

The presets also set the `@reduce` op their CPU path uses. A custom `combine`
stays scalar on the CPU: a generic `fn (T, T) T` cannot be lane-widened.

### Gather and scatter

Fixed bodies, no user function — reading `src` at an arbitrary offset needs a
raw device pointer, and handing one to user code breaks the pure-scalar-function
model that lets the same source run on the CPU.

**Scatter with duplicate indices is nondeterministic.** Two threads writing one
slot race, and which lands is unspecified on both vendors. There is no
`scatterAdd`: combining duplicates needs a device float atomic, and
`global_atomic_add_f32` is gfx9+/`--unsafe-fp-atomics` on AMD.

An index at or past the end of the buffer it subscripts is skipped rather than
written — it can never corrupt memory, but the two backends do not agree on
what the untouched slot then holds. For `gather` that slot is unspecified; for
`scatter`, elements no index selects keep their prior value.

### Not provided

Scan/prefix-sum, sort, 2-D/3-D launches, dynamic shared memory, and warp
shuffles are deliberately absent. A correct decoupled-lookback scan needs
memory-ordering guarantees this API cannot express; `shfl.sync` versus
`ds_bpermute`/DPP with wave32-vs-wave64 has no portable spelling. Write those
against [`RawKernel`](advanced.html).

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

A rejection names the offending field by its full path — `Params.inner.w`, not
just the type it happened to reach.

A type can declare `gpu_layout`, `toGpu`, and `fromGpu` for an explicit custom
boundary. This is the escape hatch for manually decomposing a higher-level
host value into a stable wire representation. All three must be declared
together, and `gpu_layout` must be an `extern` or `packed` struct — a
default-layout struct guarantees nothing about field order or padding across
two separate compilations.

## Device math

`@exp`, `@log`, `@sin` and friends do not compile for NVPTX or AMDGCN: those
targets emit no libcalls, so the backend fails with `no libcall available for
fexp` or `Cannot select: fsin`. `g.math` provides device-safe replacements that
forward to libm on the host:

`exp exp2 log log2 log10 sin cos tan tanh sinh cosh pow sqrt rsqrt`

Device `f32` uses the hardware approximation instructions; device `f64` uses
software implementations; the host uses libm. Accuracy is documented per
function — note that `log`/`sin` and friends are bounded *absolutely* by the
hardware on device `f32`, so relative error is unbounded near their zeros.
Compute in `f64` and cast if that matters.

## Backend status

| Backend | Generated `map` path |              Host runtime | Validation in this package                     |
| ------- | -------------------: | ------------------------: | ---------------------------------------------- |
| CPU     |                  Yes |                Native Zig | Executed, and codegen-compared                 |
| CUDA    |             Yes, PTX | Runtime-loaded driver API | Cross-compiled and executed; PTX entries inspected |
| HIP     |           Yes, HSACO |    Runtime-loaded HIP API | Cross-compiled; ELF symbols/metadata inspected |

"Codegen-compared" means `zig build test` compiles `tests/codegen.zig` at
`ReleaseFast`, emits its assembly, and asserts that the generated
`Kernel(Spec, .cpu)` and the hand-written loop beside it end up as the *same
machine code*. In practice LLVM folds them into one symbol, along with the
`mapFn` variant, which is the strongest form that assertion can take. If they
ever diverge the build fails with an instruction-level diff.

That is the library's central claim — the abstraction is resolved at compile
time and leaves no runtime residue — so it is checked rather than asserted.

HIP is cross-compiled and its symbols inspected, but **nothing in this package
has been executed on AMD hardware**; no such device was available.
