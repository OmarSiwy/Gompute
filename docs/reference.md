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
| Errors | One flat set of 12, listed [below](#errors). `lastDriverError()` returns the raw CUDA/HIP code behind the last failure, which is the only way to tell "no driver" from "out of memory". It is cleared on success. |
| `block_size` | Must be 1…1024. Not required to be a multiple of the warp/wave size, but anything else wastes part of every warp. |
| Empty input | `run` on a zero-length slice returns without launching. |

## Errors

Every fallible call in the library returns from one set, `g.Error`. It is
closed and flat on purpose: a caller that wants to distinguish "the machine has
no GPU" from "this build shipped no PTX" should not have to match on a nested
union.

| Error | Raised when |
| --- | --- |
| `error.InitFailed` | The driver library is absent, or `cuInit`/`hipInit` failed. This is the *no GPU here* error. |
| `error.NoDevice` | The ordinal passed to `init` names no device (`< 0`, or past the last one), or a device attribute query failed. |
| `error.ContextFailed` | Retaining or making current the device's primary context failed. |
| `error.SyncFailed` | `synchronize` on a context or stream failed, or a stream could not be created. Usually the *previous* launch faulted. |
| `error.AllocFailed` | Device allocation failed — out of memory, or a zero-byte request. Also returned by `runtime.dynamic` `alloc` on the `.cpu` backend, which has no device memory. |
| `error.ModuleLoadFailed` | The PTX/HSACO image was rejected by the driver. Almost always an arch mismatch: `sm_89` code on an `sm_75` card. |
| `error.KernelNotFound` | The image loaded but holds no symbol by that name — a kernel missing from `exportKernels`, or a bad name passed to `rawKernelByName`. |
| `error.LaunchFailed` | The driver refused the launch: block size over the device limit, a grid past `gridDim.x`, or bad arguments. |
| `error.CopyFailed` | A host↔device or device↔device copy failed. An overrun past `Buffer.bytes` lands here, since the copy is not bounds-checked in-process. |
| `error.InvalidArgument` | Caller-side validation: a null or freed `Buffer`, or a launch geometry `Dim3.linearChecked` will not express. |
| `error.BackendUnavailable` | `Kernel(spec, .cuda)` was instantiated in a build that emitted no CUDA artifact. The type still compiles; `init` refuses. |
| `error.UnsupportedBackend` | Reserved. Nothing returns it today. |

The error tells you *which step* failed; `lastDriverError()` tells you *why*.

```zig
var kernel = g.Kernel(kernels.scale_relu, .cuda).init(0) catch |err| {
    const drv = g.lastDriverError();
    std.debug.print("{t}: {t} code {d}\n", .{ err, drv.backend, drv.code });
    return err;
};
```

`DriverError` is `struct { code: i64 = 0, backend: enum { none, cuda, hip } = .none }`.
It is thread-local, and it is cleared on every *successful* driver call — so it
describes the failure you just saw, not one from three calls ago. Codes are the
vendor's own: `CUDA_ERROR_*` and `hipError_t`.

See [Troubleshooting](troubleshooting.html) for what to do about each.

## Launch geometry

`Dim3` is the `extern struct { x: u32 = 1, y: u32 = 1, z: u32 = 1 }` both
drivers expect. Generated kernels compute their own grid; you need this only
for [`RawKernel`](advanced.html).

| | |
| --- | --- |
| `Dim3.linear(n, block_x)` | The grid of `block_x`-wide blocks covering `n` elements. `n == 0` gives a zero grid — an empty range is not a fault. |
| `Dim3.linearChecked(n, block_x)` | Same, as `Error!Dim3`. Rejects `n == 0`, `block_x == 0`, and any grid past `max_grid_x`. |
| `Dim3.max_grid_x` | `2^31 - 1`, the cap both vendors put on `gridDim.x`. |

`linear` **panics** where `linearChecked` returns `error.InvalidArgument`.
That is deliberate: clamping would launch fewer blocks than there are elements
and silently leave the tail of the buffer unprocessed, which is a wrong answer
with no signal. Use `linearChecked` whenever the count comes from outside the
program — a file header, a socket, argv.

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

## Kernel handles

Three ways to hold a compiled spec. All of them expose the same `run`; they
differ in when the backend is decided.

| | Backend decided | |
| --- | --- | --- |
| `g.Kernel(spec, .cpu)` | Compile time | A direct inlined loop. `init` cannot fail; `deinit` is a no-op. |
| `g.Kernel(spec, .cuda)` / `(spec, .hip)` | Compile time | `init(ordinal)` retains the device's primary context and loads the artifact. |
| `g.AutoKernel(spec)` | First `init` call | Probes CUDA, then HIP, then falls back to the CPU. |

`Kernel(spec, backend)` gives you:

| | |
| --- | --- |
| `init(ordinal: c_int) Error!Self` | `ordinal` is the device index; `.cpu` ignores it. |
| `deinit(self) void` | Releases your handle. Safe twice. Not device teardown — see `shutdown()`. |
| `run(...) Error!...` | The whole operation, buffers included. Arguments per the [operation table](#operations). |
| `backend: Backend` | Comptime constant. |
| `available: bool` | Comptime constant: false if this build emitted no artifact for the backend, in which case `init` returns `error.BackendUnavailable`. |

GPU instantiations add the pieces `run` is built out of, for when you want to
hoist the allocation and the copies — see [Advanced](advanced.html):

| | |
| --- | --- |
| `alloc(count: usize) Error!Buffer` | `count` **elements**, not bytes. (`RawKernel.alloc` takes bytes.) |
| `launch(...) Error!void` | Launch only: no allocation, no copy, no synchronize. |
| `context` | The underlying `runtime.cuda.Context` / `runtime.hip.Context`. `context.synchronize()` is how you wait for a bare `launch`. |
| `Buffer` | The backend's buffer type. `void` on `.cpu`. |
| `reduceBlocks(count: usize) u32` | Block count for a `reduce` launch, so you can size the partials buffer. |

`AutoKernel(spec)` adds:

| | |
| --- | --- |
| `init() Self` | Infallible. Cannot fail because the CPU path always exists. |
| `initStrict() Error!Self` | Same probe, but a *broken* GPU artifact is fatal instead of a warning. |
| `selected() Backend` | Which backend the probe actually chose. |
| `Cpu` / `Cuda` / `Hip` | The three underlying handle types. |

The distinction `initStrict` exists for: a missing driver means *no GPU on this
machine*, and falling back to the CPU is correct. An artifact that is present
and still fails to load means *this build is wrong* — wrong device arch, or a
kernel left out of `exportKernels` — and quietly running 100× slower is not a
service. `init` warns in that case; `initStrict` returns the error.

```zig
var kernel = g.AutoKernel(kernels.scale_relu).init();
defer kernel.deinit();
std.debug.assert(kernel.selected() == .cuda);
```

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
two separate compilations. [Advanced](advanced.html) has a worked example.

Generated `map` kernels do all of this for you. `g.abi` is exposed for the
cases where you are building the launch yourself — a [`RawKernel`](advanced.html)
that wants the same parameter struct, or a
[low-level launch](runtime.html):

| | |
| --- | --- |
| `g.abi.Boundary(T)` | The wire type derived from `T`. A compile error if `T` cannot cross, naming the offending field by its full path. |
| `g.abi.pack(T, value)` | Host value → `Boundary(T)`. |
| `g.abi.unpack(T, wire)` | `Boundary(T)` → host value. |
| `g.abi.assertStable(T)` | Comptime-only check. Put it in a `comptime` block to fail the build the moment a field is added that cannot cross. |

```zig
comptime { g.abi.assertStable(Params); }

var wire = g.abi.pack(Params, .{ .scale = 2 });
var args = [_]g.interface.Arg{ buffer.argPtr(), g.interface.arg(&wire) };
```

## Device math

`@exp`, `@log`, `@sin` and friends do not compile for NVPTX or AMDGCN: those
targets emit no libcalls, so the backend fails with `no libcall available for
fexp` or `Cannot select: fsin`. `g.math` provides device-safe replacements, and
they compile on the host too, so one kernel source builds both ways.

Every one is `inline fn (x: anytype) @TypeOf(x)`, except `pow(x, y)`. They take
`f32`, `f64`, and `@Vector`s of either; anything else is a compile error telling
you to cast first. Vectors matter because the CPU backend instantiates a generic
map body at vector width, so that is what a vectorized kernel passes them.

The vector form is correct rather than fast. The ones that are a builtin —
`sin`, `cos`, `tan`, `sqrt`, `rsqrt` and `f32` `exp2` — stay elementwise; the
ported transcendentals index a table with the input and branch on it, so they
run a lane at a time.

`exp`, `exp2`, `log`, `log2`, `log10` and `pow` are ports of [ARM
optimized-routines][aor] — ONE body each, used unchanged on the host, on NVPTX
and on AMDGCN, for both widths (only `f32` `exp2` still uses the hardware
instruction). Everything else is a musl port for device `f64` and a hardware
approximation for device `f32`.

`expm1` and `atan` are here for a different reason, and it is worth knowing if
you reach for `std.math` in a kernel. Neither needs a libcall — but both of
std's ports raise the subnormal underflow flag through
`std.mem.doNotOptimizeAway`, which for a float becomes `asm volatile ("" :: "rm"
(v))`. The AMDGPU backend cannot match the `m` alternative, so both are a hard
error on AMDGCN while assembling to PTX without complaint. **An NVIDIA-only test
matrix will not see this.** `g.math.expm1` is std's own algorithm with that line
dropped (values bit-identical); `g.math.atan` keeps std's scalar body on the
host and takes its vector path on device. `std.math.log1p` happens not to
contain the idiom and is fine as-is.

[aor]: https://github.com/ARM-software/optimized-routines

That is a correctness decision before it is a speed one: glibc ≥ 2.28 *is* ARM
optimized-routines for exactly those functions, so a simulator scored against a
glibc-linked reference now shares its arithmetic. Zig's own `@exp`/`@log` —
and `extern "c" fn exp` too, because compiler_rt's static definition wins over
the shared `libm` — are musl, a different algorithm that disagrees in the low
bits.

Ulp figures below are measured against the real glibc (`dlsym`'d past
compiler_rt), ≥1e6 points per function over the full domain including
subnormals, the saturation thresholds and the near-1 window; `sin`/`cos`/`tanh`
and friends are measured on `sm_89` over x in (0, 8].

| | `f32` | `f64` | Notes |
| --- | --- | --- | --- |
| `exp` | ≤1 ulp | ≤1 ulp | One body, host and device. |
| `exp2` | ≤1 ulp | ≤1 ulp | Exact for integer `x`. `f32` is the hardware `exp2`. |
| `log` | ≤1 ulp | ≤1 ulp | One body. Relative, not absolute — see below. |
| `log2` | ≤1 ulp | ≤1 ulp | Own table; powers of two exact. |
| `log10` | ≤1 ulp | ≤2 ulp vs glibc | 0.52 ulp against a 60-digit reference: the slack is glibc's, whose `log10` is not optimized-routines. |
| `pow` | ≤1 ulp | ≤1 ulp | Integer y in ±64 is square-and-multiply, so exact. |
| `sin` | ~1e-6 **absolute** | ≤1 ulp | f64 matches glibc bit-for-bit over (0, 8]. |
| `cos` | ~1e-6 **absolute** | ≤1 ulp | |
| `tan` | sin/cos | sin/cos | 0.16 absolute at 3π/2 in f32; error blows up at the poles. |
| `tanh` | ≤1.8 ulp | ≤1 ulp | |
| `sinh` | ≤3.7 ulp | ≤1.4 ulp | |
| `cosh` | ≤3.7 ulp | similar | Overflows to infinity just under x = ±710 in f64. |
| `sqrt` | exact | exact | Native instruction on both backends. |
| `rsqrt` | exact | exact | `1/sqrt`, two IEEE ops — not the hardware approximation. |

**The absolute-error caveat.** `sin` and `cos` are bounded *absolutely* by the
f32 hardware, not relatively. `sin.approx.f32` is good to about 2^-20 of
absolute error, which is fine at x = 3 and is pure noise at x = π. Relative
accuracy therefore collapses near each function's zeros: `sin` near multiples
of π, `cos` near π/2 + kπ. If you are working in that neighbourhood, compute in
`f64` and cast the result.

The `log` family used to have the same problem — `lg2.approx.f32` bounds its
error absolutely at ~2^-21, so `log(1.0000001)` was noise — which is what the
`logf`/`log2f`/`log10f` ports fixed. They accumulate in binary64 and round
once, so device `f32` `log` now costs about ten `f64` operations (1/64 rate on
consumer NVIDIA) and is worth it only because it is otherwise wrong where it
matters. A throughput-bound `f32` kernel that genuinely never goes near x = 1
should call `@log2` directly.

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
