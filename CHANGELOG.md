# Changelog

What changes for code that uses Gompute, newest first. A release is a
`v<version>` tag; `docs/reference.md`'s Versions table names the Zig each
one builds with. 1.0.0 is the first release under this numbering; the
earlier tags were retired, and every commit they named is on `main`.

## 1.0.0

- **Kernels.** Hand-written Zig kernels compile at build time to PTX and
  HSACO and are embedded in the executable; a thin CUDA/HIP runtime loads
  them by name. HIP compiles and links but has not run on AMD hardware.
- **Device math** (`g.math`: `exp exp2 log log2 log10 log1p pow sin cos tan
  tanh sinh cosh expm1 atan atan2 asinh acosh atanh sqrt rsqrt`, for `f32`,
  `f64` and vectors of them) gives the same bits on the host, NVPTX, AMDGCN
  and at comptime within one FMA class, which the target fixes at comptime
  (x86_64 with `fma`, aarch64, NVPTX and AMDGCN are the FMA class). f64
  `exp`, `log` and `pow` have an FMA body and an FMA-free body and measure
  0.505 / 0.502 / 0.502 ulp against an f128 reference in both; f64 `asinh`
  and `acosh` call f64 `log` and follow its class. A host built for baseline
  x86_64 can differ from the GPU by one ulp: 30, 2 and 23 of 65,536 inputs
  for `exp`, `log` and `pow`. `math.max_ulp` promises `asinh` 1.5 ulp,
  `acosh` 2.3, `atanh` 1.8 and `atan2` 1.5, with C99 F.10's special cases.
- **Speed.** On `x86_64_v3` a scalar f64 `exp`, `log` and `pow` call takes
  2.17 / 2.74 / 8.48 ns, and an RTX 4060 Laptop runs them at 7.26 / 6.22 /
  2.36 Gcalls/s.
