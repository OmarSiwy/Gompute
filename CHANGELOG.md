# Changelog

What changes for code that uses Gompute, newest first. A release is a
`v<version>` tag; `docs/reference.md`'s Versions table names the Zig each
one builds with. Releases before 2.1.0 are described by their tags.

## Unreleased (2.1.0)

- **`math` gains `asinh`, `acosh`, `atanh` and `atan2`** for `f32`, `f64`
  and vectors of them, device-safe like the rest of `math` (no libm; host,
  NVPTX and AMDGCN checked free of optional contraction). Special cases follow C99 F.10:
  signed zeros and infinities, `acosh(x < 1)` and `atanh(|x| > 1)` are NaN,
  `atanh(±1)` is ±inf, and `atan2` returns the quadrant-correct angle in
  [-π, π] for every sign of zero and infinity. Accuracy against an f128
  reference, now listed in `math.max_ulp`: `asinh` 1.5 ulp, `acosh` 2.3
  (its worst case is near `x = 1`), `atanh` 1.8, `atan2` 1.5. Code that
  carried its own copies of these (VerA did) can drop them.
