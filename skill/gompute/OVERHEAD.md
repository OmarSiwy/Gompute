# What each feature costs

Measured on an RTX 4060 Laptop (sm_89, driver CUDA 12.x) with a Ryzen host,
`ReleaseFast -Dcpu=native`, median of 200 runs. The kernel is `add_offset`
(`x + p.offset`, f32), so these numbers are pure overhead and bandwidth.
Expect the same shape on other hardware, though the exact numbers will move.

## Compile time: free at run time

| Feature | Run-time cost | Paid at |
| --- | --- | --- |
| Spec constructors (`map`, `zip`, `sum`, ...) | 0, zero-sized types | comptime |
| Kind dispatch (`run`/`launch` picks a body per kind) | 0, `switch` on a comptime enum | comptime |
| `Params` -> extern wire struct (`abi.pack`) | a by-value copy of the struct, folded away on the CPU | comptime layout |
| `Fused` / `Unary` | 0, `inline for` into one expression | comptime |
| Kernel-name lookup (`Kernel.init` -> which blob, which symbol) | 0, a comptime `StaticStringMap`; a missing kernel is a compile error | comptime |
| Math table exactness checks | 0, a `comptime` block that fails the build on a bad digit | every build, device included |
| `Kernel(spec, .cpu)` | 0. `zig build test` asserts it is the same machine code as the hand loop | checked by codegen test |

## CPU backend

| Path | Cost |
| --- | --- |
| `init`/`deinit` | nothing, both inline to no code |
| generic `map` | `suggestVectorLength(T)` lanes per step, scalar tail of < lanes elements |
| generic `mapTo`, `zip`, `mapIndexed` | the same SIMD chunking as `map`. Measured: `zip` a·s+b 0.40 → 0.15 ns/elem (2.8×); `mapTo` of `exp` 2.93 → 1.20 (2.45×) |
| concrete `map`, `mapTo`, `zip`, `mapIndexed`, `gather`, `scatter` | a plain scalar loop. **Zig 0.16 ships LLVM's loop vectorizer disabled**, so this is one element per instruction |
| `sum`/`min`/`max`/`any`/`all` | 4 vector registers of accumulators, a single `@reduce` at the end; tail < 4×lanes scalar. **1.8 µs for 64K f32** (old per-chunk `@reduce`: 9.9 µs) |
| custom `reduce` | scalar fold, one dependent `combine` per element |
| `.pre` on a reduce | one scalar call per element, inlined |
| `g.math` at vector width | see "Device math against the libraries it replaces" below |

`map` n=1M f32 on the CPU: **266 µs** (memory-bound). `map` n=1024: **0.29 µs**.

## GPU backend

| Operation | n = 1024 | n = 1M f32 | What it does |
| --- | ---: | ---: | --- |
| first `Kernel.init` in the process | 220–255 ms | | `cuInit` 55–145 ms + primary context 75–90 ms (NVIDIA driver) + PTX JIT 9 ms cold / 0.1 ms from the driver's cache |
| first `Kernel.init` after a background warm-up | 0.29 ms | | see "Startup" below |
| later `Kernel.init`, same device/root | 0.4 µs | | table lookup: context and module are process-wide and cached |
| `AutoKernel.init` (warm) | 0.3 µs | | probe, CUDA first, then HIP, then CPU |
| `run()` | **12.3 µs** | **834 µs** | upload + launch + sync + download; buffers reused from the handle (was 82 / 983 µs when it allocated per call) |
| `launch` + `context.synchronize()` | 4.8 µs | 11.8 µs | the launch itself |
| 10 × `launch` + sync | 19.9 µs | 90.5 µs | ~1.5 µs CPU-side per extra launch |
| graph of 10 launches, replay + sync | 13.0 µs | 85.2 µs | one driver call for all ten |

What to take from it:

- **`run()` is a convenience, not a hot-path API.** At n=1024 it costs 2.5× a
  bare launch, and at 1M it is 4× slower than the CPU doing the same `map`,
  because both copies go through pageable memory. The GPU only wins once the
  data *stays* there: hoist with `alloc`/`launch`.
- **The first `init` is driver start-up, not Gompute.** `cuInit` and the
  primary context are ~99% of it, and no library can make them 0. Take them
  off the critical path instead (next section). Every later handle on that
  device and root costs ~0.5 µs. Roots load lazily, so only the root holding
  the kernel is JIT'd, and the driver caches the JIT on disk
  (`~/.nv/ComputeCache`) after the first run on a machine.
- **Graphs pay off when launches are small.** For 10 small launches, a replay
  saves about 35% (19.9 → 13.0 µs). When each kernel is bandwidth-bound, the
  saving shrinks to ~6%, because the GPU time dominates. Graphs remove
  CPU-side submission cost, not kernel time.
- `reduce` on the GPU: a grid-stride kernel capped at 1024 blocks, then one
  shared-memory tree per block. The host downloads ≤1024 partials and folds
  them, which costs microseconds next to the download that carries them.
  Shared memory, not warp shuffles: about 15% slower than a shuffle tree, but
  one body on both vendors.
- `gather`/`scatter` `run` uploads `out` as well (unselected slots keep their
  value), so expect one more copy than `mapTo`.
- Device math is one software body per function so CPU and GPU agree on
  every bit. On the GPU, f32 functions without an f32 port (sin, cos, tan,
  tanh, sinh, cosh, exp2, expm1, log1p, pow) compute in f64, and f64 runs at
  1/64 rate on consumer NVIDIA (`context.fp64Ratio()` tells you). A
  throughput-bound f32 kernel that can live with hardware approximations and
  target-dependent bits can call `@sin`/`@exp2` directly. Accuracy is ≤0.505
  ulp for exp/log/pow against an f128 oracle.
- `pow` with an integer exponent takes the full table route, like any other
  exponent (a square-and-multiply shortcut broke monotonicity). In a hot
  kernel, write `x * x` yourself for `pow(x, 2)`.

## Device math against the libraries it replaces

The same i9-14900HX (AVX2, so W = 4 for f64 and 8 for f32), pinned to one
P-core, ReleaseFast `-mcpu=native`, 8K inputs, best of 80. Units are ns per
element, lower is better.

- **gompute**: `g.math.f(x)` in a scalar loop.
- **g.vector**: `g.math.f(v)` on `@Vector(W)`.
- **glibc**: libm through `dlopen`, the way a C program calls it. This is the
  equivalent library: glibc ≥ 2.28 *is* ARM optimized-routines for
  exp/exp2/log/log2/pow, the same algorithms gompute ports.
- **zig@**: Zig's `@exp`/`@log`/`@sin` (compiler_rt, musl).
- **std**: `std.math.pow`.

| f64 | gompute | g.vector | glibc | zig@ | std |
| --- | ---: | ---: | ---: | ---: | ---: |
| exp | 3.29 | **1.55** | 3.88 | 10.52 | |
| exp2 | 2.50 | **1.29** | 2.67 | 7.86 | |
| log | 3.77 | **2.25** | 4.13 | 4.28 | |
| log2 | 4.82 | **2.60** | 3.39 | 6.77 | |
| log10 | 6.15 | **2.95** | 6.78 | 7.18 | |
| pow | 13.80 | **5.26** | 10.91 | | 51.02 |
| sin | 14.34 | 13.72 | 13.72 | 13.78 | |
| cos | 14.55 | 13.74 | 11.58 | 13.09 | |
| tanh | 11.97 | 11.85 | 15.64 | 11.95 | |

| f32 | gompute | g.vector | glibc | zig@ | std |
| --- | ---: | ---: | ---: | ---: | ---: |
| exp | 2.38 | **1.09** | 2.38 | 8.29 | |
| exp2 | 3.00 | **1.21** | 2.31 | 3.34 | |
| log | 2.55 | **1.33** | 2.59 | 3.53 | |
| log2 | 2.55 | **1.37** | 2.22 | 4.42 | |
| log10 | 2.69 | **1.35** | 3.63 | 4.88 | |
| pow | 11.78 | **5.22** | 5.22 | | 49.13 |
| sin | 6.63 | **1.38** | 5.70 | 9.57 | |
| cos | 7.08 | **1.43** | 6.08 | 9.54 | |
| tanh | 7.06 | **2.12** | 4.42 | 10.30 | |

What it says:

- **At vector width, gompute beats glibc's scalar libm on everything except
  f64 sin/cos/tanh.** It's 2–2.5× faster on f64 exp/log/pow, and 2–4× on
  f32, including sin/cos (4×).
- **Scalar, gompute is on par with glibc** for exp/exp2/log/log10. It's
  slower where glibc uses FMA, which gompute is barred from by its
  bit-identity contract: f64 `log2` 4.8 vs 3.4, f64 `pow` 13.8 vs 10.9. It's
  also slower on f32 `pow` scalar (11.8 vs 5.2, where glibc has a dedicated
  f32 algorithm) and f32 `tanh` scalar (7.1 vs 4.4).
- **Accuracy is the same class or better.** f64 exp/log/pow are 0.505 /
  0.502 / 0.502 ulp against an f128 oracle. Every f32 function is
  exhaustively faithful over all 2³² inputs (max 0.82 ulp, most ~0.50).
  glibc promises < 1 ulp for these and doesn't give identical bits across
  CPU/GPU.
- **What the others can't do:** the same bits on the CPU, on NVIDIA, on AMD
  and at comptime, plus monotone exp/log/pow/expm1/log1p/sinh. glibc's
  results depend on FMA hardware, and there's no glibc on a GPU.
- `std.math.pow` is 4–10× slower than everything else here. Never use it in
  a kernel.

Where the remaining time goes: each vector lane still does one table-row
load (exp: 16 bytes, log: 32 bytes), so a 4-lane f64 `log` is four 256-bit
loads plus a transpose before the arithmetic. f64 sin/cos/tanh are musl's
branchy scalar bodies run one lane at a time; vectorizing them lane-exact
means reproducing musl's per-range reduction choices with selects. That's
the next upgrade if a profile asks for it.

## Startup: hide the driver's 200 ms

Start the driver on a thread as the very first thing `main` does, and do the
rest of your startup while it runs:

```zig
const warm = try std.Thread.spawn(.{}, struct {
    fn f() void {
        var c = g.runtime.cuda.Context.init(0) catch return; // no GPU: fine, init reports it later
        c.deinit(); // the primary context stays retained process-wide
    }
}.f, .{});
// ... parse config, load files, build the model ...
warm.join();
var kernel = try g.Kernel(spec, .cuda).init(0); // 0.29 ms: makeCurrent + cached module load
```

Measured: 255 ms becomes 0.29 ms on the critical path, as long as the app has
~250 ms of its own work to overlap. On a headless Linux box, `nvidia-smi -pm 1`
(persistence mode) keeps the driver initialized between processes and cuts
`cuInit` itself.

## Memory and threads

- Nothing in the library allocates on the host. `run` keeps up to three device
  buffers on the handle (grown on demand, freed by `deinit`); `launch`
  allocates nothing. One handle must not `run` from two threads at once.
- Contexts and modules are shared process-wide per device (16 devices,
  64 module images; past that, a module is loaded uncached, which is slower
  but still correct). `deinit` releases your handle only.
  `g.runtime.cuda.shutdown()` is the real teardown.
- One global spin lock guards init-time tables only. Launches never take it.
- Safe to use from several threads: a context is made current per thread on
  first use (one TLS check per call after that). Two threads writing one
  buffer race.
- `g.lastDriverError()` is thread-local, so writing it costs one TLS store per
  driver call.

## Build time

- Each kernel root is one `zig build-obj` per backend, cached separately and
  compiled in parallel. Mark huge roots `.heavy = true` and set `heavy_lanes`
  to bound peak memory.
- Device code always builds `ReleaseFast`, even when you ask for Debug.
  `ReleaseSafe` is honored if you ask for it, but it was measured 32× slower to
  compile on a 140k-line device model.
