# Changelog

What changes for code that uses Gompute, newest first. A release is a
`v<version>` tag; `docs/reference.md`'s Versions table names the Zig each
one builds with. Releases before 2.1.0 are described by their tags.

## 3.0.0

- **f64 `exp`, `log` and `pow` take an FMA body on an FMA target** (x86_64
  with `fma`, aarch64, NVPTX, AMDGCN), picked at comptime from the target,
  never the build host, so their bits, and those of f64 `asinh` and
  `acosh`, which call f64 `log`, are the same within one FMA class rather
  than on every target. The FMA-free class keeps 2.0.1's and 2.1.0's bits. A host
  built for baseline x86_64 can now differ from the GPU by one ulp: 30, 2
  and 23 of 65,536 inputs for `exp`, `log` and `pow`. Both classes stay at
  0.505 / 0.502 / 0.502 ulp against an f128 reference. On `x86_64_v3` a
  scalar call takes 2.17 / 2.74 / 8.48 ns instead of 3.04 / 3.79 / 12.18,
  and an RTX 4060 Laptop runs 7.26 / 6.22 / 2.36 Gcalls/s instead of 4.84 /
  4.03 / 1.50 (#6).

## 2.1.0

- **`math` gains `asinh`, `acosh`, `atanh` and `atan2`** for `f32`, `f64`
  and vectors of them, device-safe like the rest of `math` (no libm; host,
  NVPTX and AMDGCN checked free of optional contraction). Special cases follow C99 F.10:
  signed zeros and infinities, `acosh(x < 1)` and `atanh(|x| > 1)` are NaN,
  `atanh(±1)` is ±inf, and `atan2` returns the quadrant-correct angle in
  [-π, π] for every sign of zero and infinity. Accuracy against an f128
  reference, now listed in `math.max_ulp`: `asinh` 1.5 ulp, `acosh` 2.3
  (its worst case is near `x = 1`), `atanh` 1.8, `atan2` 1.5. Code that
  carried its own copies of these (VerA did) can drop them.
