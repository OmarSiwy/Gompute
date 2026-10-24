# Agents

Gompute is a library: one scalar kernel source, compiled at build time into a
CPU loop, a PTX entry point and an HSACO entry point. Its consumers (ESPice)
pin it by commit and content hash in `build.zig.zon`, so a change here reaches
nobody until they move the pin. Two rules decide everything below: **every
test names its oracle**, and **main is green by machine**.

## Testing

An oracle is the independent thing that says an output is wrong. The main bug
class here is a wrong number with no crash, on a backend nobody can see into,
so every layer below names its oracle and the two that run on hardware are
the ones that count.

| Layer | Oracle | Where | Tier |
|---|---|---|---|
| CPU residue | the hand-written loop beside the generated one; LLVM must fold both into one symbol, else an instruction diff fails the build | `tests/codegen.zig`, `tools/codegen_check.zig` | `zig build test` |
| device code shape | one spec of every kind compiled to PTX and gfx1100 ISA; `build.zig` pins the instructions `src/device/export.zig` and `builtins.zig` promise (`ld.global.v4`, `global_load_b128`, the atomics, the barrier fences, `s_trap`) | `tests/device_entries.zig` | `zig build test` |
| device math | glibc libm dlsym'd at 1 ulp (Linux only; skips elsewhere), an f128 oracle at 0.52 ulp, bit identity with std where promised, monotonicity at every table seam, comptime fold equal to run time | `src/device/math.zig` tests | `zig build test` |
| CPU paths | the scalar body over every length from 0 past three vector widths and a non-multiple block | `src/host/kernel.zig` tests | `zig build test` |
| driver bindings | hand-built handles never reach an undefined driver; graphs, events, peer copies and fake devices on the card | `src/runtime/cuda.zig` tests | `zig build test`, skips without a GPU |
| every kind on the GPU | the CPU path, at every edge length (0, 1, lane group, block, a few blocks, each one less, exact and one more, and 70001) | `examples/exhaustive` `[kinds]` | by hand, on a card |
| GPU math | the CPU bits, over 64K inputs per function | `examples/exhaustive` `[math bit-identity]` | by hand |
| several devices | sharded map, peer copy, peer access, with `GOMPUTE_FAKE_DEVICES=2` on one card | `examples/exhaustive` `[devices]` | by hand |
| speed | fraction of the card's peak bandwidth (`Context.info`), concrete against generic CPU bodies | `examples/bench` | by hand; its numbers are what `skill/gompute/OVERHEAD.md` quotes |

- An instruction check proves the instruction is there, not that the kernel is right. The kinds sweep is the correctness oracle; run it after touching anything under `src/host`, `src/device` or `src/runtime`.
- HIP compiles, links through `ld.lld`, and its handle-less tests run. **Nothing has executed on AMD hardware.** Every HIP claim in the docs says so; keep it that way until a card has run the exhaustive suite.
- A bug found by hand, by a user or by review means the oracle set has a hole: land the fix with its regression test and the oracle or length that would have caught the whole class. Check a new test goes red on the bug before trusting it green.
- Integers compare bit for bit. Floats compare within `same` (1024 ulp) because the kernels are checked for the operation, not the rounding, except `g.math`, whose contract is the bits. A reduce reorders, so it gets a length-scaled relative tolerance, never a bitwise one.
- Every `examples/exhaustive` section declares how many checks it reaches; a bailed section is a failure, not a pass with fewer checks. A section that cannot run here calls `skip` before its first check.
- Fix a flaky test or quarantine it the same day. Never retry silently. Gate on the exit code: `zig build test 2>&1 | tail` swallows it.

## Maintaining

- `main` is the only long-lived branch. Work lands through a pull request whose merge result passed `.github/workflows/test.yaml` (`test` on three OSes, `examples`, `nix`). No `dev` branch.
- The public surface only grows. A `pub` signature that a consumer could call is kept; a replacement is added beside it. The package version in `build.zig.zon` moves in the same commit as the change, and the commit message says what changed and the version.
- The toolchain is a pinned nightly of Zig master, the same build ESPice pins. It is named in four places that move together: `build.zig.zon` (`minimum_zig_version`), `flake.lock` (the zig-overlay revision whose `master` is that build), `.github/workflows/*.yaml` (`ZIG_VERSION`) and `README.md`. A release is a `v<version>` tag (`v1.5.0`) whose version equals `build.zig.zon`'s; the docs' Versions table says which Zig it builds with.
- Actions are pinned by commit SHA with the tag in a comment.
- `docs/*.md` is the site (`zig build docs`) and `skill/gompute/` is the consumer skill; both describe the same API, so a `pub` change edits both. Every number in them is measured by `examples/bench` or a named test and says which Zig and which card.

## Building

- `nix develop` gives the pinned Zig, the CUDA and ROCm library paths, and a global cache per Zig store path (`ZIG_GLOBAL_CACHE_DIR=~/.cache/zig-<store path>`). Outside nix, use that exact Zig and set the same cache dir: two Zig builds sharing `~/.cache/zig` evict each other's build runner, and every `zig build` then recompiles it for about two minutes.
- Steps: `zig build test` (every unit suite, the codegen fold, both device probes, the docs renderer's tests); `zig build docs` (the site into `zig-out/docs`, `-Dopen` to view). Examples: `cd examples/basic && zig build run`; `cd examples/exhaustive && zig build run` (the hardware suite; `-Dcuda-arch=sm_89 -Dhip-arch=gfx1100` pins both device targets, which is how CI compiles them without a GPU; `GOMPUTE_FAKE_DEVICES=2 zig build run` exercises the devices section on one card); `cd examples/bench && zig build run -Doptimize=ReleaseFast`.
- Edit loop: `zig build test --watch` reruns on every save. The device probes compile in a few seconds; the libm oracle tests dominate a full run.
- Device code is always ReleaseFast, whatever the host mode (`deviceOptimize` in `build.zig`): Debug device code drags the panic machinery in, and ReleaseSafe was measured 32x slower to compile on a 140k-line device model.
- One `.zig-cache` per checkout; a worktree symlinks the main checkout's. Never run two `zig build`s with different Zig binaries in one tree at once. Separately, an uncached `zig build test` prints the math tests' stderr under a `failed command:` line even though the step passed and the exit code is 0; the rerun is cached and silent. Read the exit code, not that line.
- Rebuild times on this machine (i9-14900HX, Zig 0.18.0-dev.349, warm global cache, measured 2026-10-24):

| Change | `zig build test` |
|---|---|
| nothing | 0.2 s (7 s when the configure cache is cold) |
| a comment in `src/host/kernel.zig` (host module: unit tests, codegen probe, docs object) | 11.7 s |
| a comment in `src/device/export.zig` (both device probes) | 3.0 s |
| fresh local `.zig-cache`, warm global | 3.7 s |
| `examples/exhaustive` device code for sm_89 and gfx1100, fresh local cache | 15.2 s (0.2 s no-op) |

## Layout

- `src/root.zig` is the host root and `src/device.zig` the device root. A kernels file compiles against both; the two export the same names except six polarity switches (`is_device`, `kernel_callconv`, `GlobalPtr`, `globalIdX`, `exportRaw`, `exportKernels`). A name added to one is added to the other.
- `src/core/`: `spec.zig` (the constructors, each a zero-sized type whose decls are the contract both sides read; `deviceLanes` is the one number the host grid and the device entry share), `abi.zig` (the host/device boundary derivation), `fusion.zig`, `interface.zig` (`Dim3`, `Error`, the thread-local driver error).
- `src/host/`: `kernel.zig` (`CpuKernel`, `GpuKernel`, `AutoKernel`, module opening), `raw.zig` (`RawKernel`, `rawKernelByName`).
- `src/device/`: `export.zig` (one entry shape per kind; each `run` signature pairs with a `*Launch` argument array in `kernel.zig`, and nothing checks the pairing), `builtins.zig` (thread indices, barrier, device-scope atomics), `math.zig` with `math_data.zig` and `math_pio2.zig`.
- `src/runtime/`: `cuda.zig` and `hip.zig` are twins, method for method with the vendor types swapped; a change to one lands with its twin in the same change. `compute.zig` is the tagged-union facade over both; `dynlib.zig` the loader.
- `build.zig`: the consumer API (`emitKernels`, `addKernels`) above the `pkg_hash` early return, the dev tooling below it. `tools/`: the IR name table, the codegen check, the Markdown renderer. `tests/`: the two probes. `examples/`: `basic`, `exhaustive`, `bench`.

## How a change is made

The order is fixed by what a later step cannot undo: the boundary first, then where a piece lives, then the data, then the zero-cost proof, then contracts, then tests, then branches, then lanes. Three rulings settle conflicts: data layout beats code brevity; the residue probe comes before the abstraction; measure before accepting a performance rung.

- GPU: struct of arrays only (`zip`, never `[]struct { a, b }`). Name the ceiling before touching a kernel (bandwidth, compute, latency, overhead) with the arithmetic shown, then measure the fraction of peak in `examples/bench`. No hard-coded warp width: AMD is wave64 on GCN and CDNA, wave32 on RDNA. Reductions reorder only with the launch geometry, through the fixed shared-memory tree, never through atomics. Device math stays one body per function with no FMA and no intrinsic; that bit identity between CPU, NVPTX, AMDGCN and comptime is what VerA's constant folding and ESPice's CPU/GPU comparison rely on.
- CPU: a concrete body is left to LLVM's loop vectorizer (on since 0.18, and it unrolls); a generic (`anytype`) body is widened by hand only because `g.math`'s lane-exact vector forms cannot be found by a vectorizer; the reduce presets keep their explicit accumulators because a strict float sum never vectorizes. A vectorization claim is read off the assembly, never off a timing.
- A cleanup pass is one agent per unit with a disjoint file set, three steps in order: shape (delete first; then the data pass with sizes and widths stated; doc comments carry a fact the signature cannot), testability (every bug class gets an oracle and a layer), implementation. Nothing in a pass simplifies away a NaN gate, an `errdefer`, validation at a trust boundary, a tolerance, or the ceiling a `ponytail:` comment names. Every deliberate shortcut carries a `ponytail:` comment with its ceiling and its upgrade path.
- Each agent runs one `zig build` at a time and only its own unit's step; the full suite, the exhaustive example and the bench run between waves. The `.zig-cache` is shared, so an agent mid-edit in `src/runtime` breaks every other build until it compiles again; sequence those.
