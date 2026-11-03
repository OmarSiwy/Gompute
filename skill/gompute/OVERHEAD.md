# What each feature costs

Three measurement sets, all on one machine: an RTX 4060 Laptop GPU (sm_89,
24 SMs, peak 256 GB/s from `Context.info`) with an i9-14900HX host.

- **2.0.0** (bandwidth, `BENCH=math`): `examples/bench`,
  `zig build run -Doptimize=ReleaseFast`, Zig 0.18.0-dev.349+d924f83bf,
  measured 2026-11-03. Bandwidth is 20 back-to-back `launchOn`s on a
  non-blocking stream, timed with events after a warm-up.
- **1.4.0** (first `init`, launch, graphs, startup): Zig 0.17.0,
  `ReleaseFast -Dcpu=native`, median of 200 runs, kernel `add_offset`
  (`x + offset`, f32, one element per thread). Measured through the removed
  `Kernel` handle, whose `init` and `launch` took the same path `RawKernel`
  takes now (the same `openModule`, then `makeCurrent` + `cuLaunchKernel`).
  Not re-measured on 2.0.0.
- **libm comparison** ("Device math against the libraries it replaces"):
  the f64 table from 1.4.0 on Zig 0.17.0, the f32 table from 1.5.0; the
  glibc and zig@ columns have not been re-measured since.

Expect the same shape on other hardware, though the exact numbers will move.
Nothing has executed on AMD hardware, so every number here is NVIDIA's.

## Compile time: free at run time

| Feature | Run-time cost | Paid at |
| --- | --- | --- |
| Kernel-name lookup (`RawKernel.init` -> which blob, which symbol) | 0, a comptime `StaticStringMap`; a missing kernel is a compile error | comptime |
| `g.available(backend)` | 0, a comptime bool | comptime |
| Math table exactness checks | 0, a `comptime` block that fails the build on a bad digit | every build, device included |

`rawKernelByName` does the same lookup at run time: one `StaticStringMap`
probe per call.

## CPU

There is no CPU backend: the CPU path is your loop. A plain loop over a
concrete arithmetic body is left to Zig 0.18's LLVM loop vectorizer, which
widens and unrolls it. It does not widen calls into `g.math`; step the loop
in `@Vector`s by hand for those (the `BENCH=math` table below shows what that
buys per function). A strict float sum never auto-vectorizes: keep several
vector accumulators yourself and `@reduce` once at the end. Read a
vectorization claim off the assembly, never off a timing.

## GPU backend

Bandwidth of one-element-per-thread raw kernels (`examples/bench/src/kernels.zig`),
GB/s from the event-timed µs per launch, counting each byte read once and
written once:

| Kernel | n = 16M | |
| --- | ---: | --- |
| `raw_map_f32` (`max(x * s, 0)` in place) | 195 GB/s | 76% of peak |
| `raw_map_f64` (`x * s` in place) | 197 GB/s | 77% |
| `raw_zip_f32` (`out = a * s + b`) | 198 GB/s | 77% |

At n = 1M the buffers sit in the 24 MB L2 and read over 100% of DRAM peak,
so that size measures the cache, not the bus. At n = 1024 a launch is
~2.2–2.4 µs device-side whatever the kernel: overhead-bound.

Driver and launch costs (1.4.0, see the top):

| Operation | n = 1024 | n = 1M f32 | What it does |
| --- | ---: | ---: | --- |
| first `init` in the process | 220–255 ms | | `cuInit` 55–145 ms + primary context 75–90 ms (NVIDIA driver) + PTX JIT 9 ms cold / 0.1 ms from the driver's cache |
| first `init` after a background warm-up | 0.29 ms | | see "Startup" below |
| later `init`, same device/root | 0.4 µs | | table lookup: context and module are process-wide and cached |
| upload + launch + sync + download, pageable memory | 12.3 µs | 834 µs | 1.4.0's removed `run()`; the copies, not the launch |
| `launch` + `synchronize()` | 4.8 µs | 11.8 µs | the launch itself |
| 10 × `launch` + sync | 19.9 µs | 90.5 µs | ~1.5 µs CPU-side per extra launch |
| graph of 10 launches, replay + sync | 13.0 µs | 85.2 µs | one driver call for all ten |

What to take from it:

- **Keep data on the device.** A round trip through pageable memory cost 70×
  the launch at 1M f32. The GPU only wins once the data *stays* there:
  allocate and upload once, `launch` many times, download once. For copies
  that overlap work, use pinned memory and the `*Async` copies on a stream.
- **The first `init` is driver start-up, not Gompute.** `cuInit` and the
  primary context are ~99% of it, and no library can make them 0. Take them
  off the critical path instead (next section). Every later handle on that
  device and root costs ~0.5 µs. Roots load lazily, so only the root holding
  the kernel is JIT'd, and the driver caches the JIT on disk
  (`~/.nv/ComputeCache`) after the first run on a machine. `sass = true`
  embeds SASS for the build's arch so a card of that arch skips the JIT.
- **Graphs pay off when launches are small.** For 10 small launches, a replay
  saves about 35% (19.9 → 13.0 µs). When each kernel is bandwidth-bound, the
  saving shrinks to ~6%, because the GPU time dominates. Graphs remove
  CPU-side submission cost, not kernel time.
- Device math gives CPU and GPU the same bits. f32 exp, log, log2, log10,
  sin, cos and atan run in f32 on the GPU (2.0.0 `BENCH=math`: exp 111, log
  99, log2/log10 89, sin/cos 67, atan 181 Gcalls/s). Every other f32 function
  (exp2, expm1, log1p, tan, tanh, sinh, cosh, pow) computes in f64, 1.5–4
  Gcalls/s, because f64 runs at 1/64 rate on consumer NVIDIA
  (`context.fp64Ratio()` tells you). A throughput-bound kernel that can live
  with hardware approximations and target-dependent bits can call `@exp2`
  directly. Accuracy is ≤0.505 ulp for f64 exp/log/pow against an f128
  oracle.
- `pow` with an integer exponent takes the full table route, like any other
  exponent (a square-and-multiply shortcut broke monotonicity). In a hot
  kernel, write `x * x` yourself for `pow(x, 2)`.

## Device math throughput (`BENCH=math`)

2.0.0, Zig 0.18.0-dev.349, RTX 4060 Laptop and i9-14900HX (not pinned to a
core). Each element makes 8 calls. GPU: Gcalls/s over 4M elements, best of 5
event-timed rounds. CPU: ns per call over 64K elements, median of 5, the
scalar call in a plain loop and the vector form 8 (f32) or 4 (f64) lanes wide.

| fn | f32 GPU | f32 scalar | f32 vector | f64 GPU | f64 scalar | f64 vector |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exp | 111.36 | 1.82 | 1.47 | 4.87 | 5.13 | 1.80 |
| exp2 | 4.05 | 4.46 | 1.72 | 5.64 | 4.16 | 1.66 |
| expm1 | 1.48 | 12.07 | 13.44 | 2.00 | 12.74 | 12.94 |
| log | 98.70 | 5.43 | 2.10 | 4.05 | 6.75 | 2.86 |
| log2 | 89.04 | 5.67 | 2.30 | 3.42 | 7.44 | 3.05 |
| log10 | 88.97 | 5.68 | 2.28 | 2.71 | 9.50 | 3.53 |
| log1p | 1.72 | 8.83 | 9.15 | 1.72 | 9.53 | 8.22 |
| sin | 67.70 | 7.40 | 2.64 | 3.11 | 8.26 | 8.00 |
| cos | 67.01 | 7.48 | 2.57 | 3.12 | 7.96 | 7.96 |
| tan | 1.75 | 8.20 | 2.42 | 1.48 | 15.93 | 15.64 |
| tanh | 2.46 | 27.40 | 3.02 | 1.59 | 10.37 | 10.17 |
| sinh | 2.52 | 21.15 | 2.81 | 1.63 | 10.73 | 9.75 |
| cosh | 3.39 | 21.06 | 2.28 | 3.24 | 5.03 | 4.91 |
| atan | 181.04 | 4.10 | 0.57 | 3.23 | 4.69 | 1.20 |
| pow | 2.18 | 18.95 | 5.53 | 1.51 | 12.28 | 4.42 |

The vector form buys nothing for expm1 and log1p at either width, or for f64
sin, cos, tan, tanh, sinh and cosh. This harness differs from the libm
comparison below (8 calls per element, no core pinning), so
compare within a table, not across.

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

1.5.0 numbers for f32. 1.6.0's correctly rounded exp, log, log2, log10, sin
and cos cost about 1.5× these on the CPU (`BENCH=math`, scalar and vector);
the glibc and zig@ columns have not been re-measured.

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
  0.502 / 0.502 ulp against an f128 oracle. f32 exp, log, log2, log10, sin
  and cos are correctly rounded over all 2³² inputs, and f32 exp2, tanh,
  sinh and cosh are ≤0.501 ulp there.
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
var kernel = try g.RawKernel("scale_relu", .cuda).init(0); // makeCurrent + cached module load
```

Measured on 1.4.0: 255 ms became 0.29 ms on the critical path, as long as the
app has ~250 ms of its own work to overlap. On a headless Linux box,
`nvidia-smi -pm 1` (persistence mode) keeps the driver initialized between
processes and cuts `cuInit` itself.

## Memory and threads

- `launch` and `launchOn` allocate nothing, on the host or the device. A
  handle holds no buffers; every `Buffer` is yours to `free`.
- Contexts and modules are shared process-wide per device (16 devices; the
  module list grows, and a failed growth loads that module uncached, which is
  slower but still correct). `deinit` releases your handle only.
  `g.runtime.cuda.shutdown()` is the real teardown.
- A `copyFrom` between buffers on two ordinals is a peer copy: over the
  direct link after `enablePeerAccess`, staged through host memory by the
  driver otherwise.
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
