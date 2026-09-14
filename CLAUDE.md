## How this codebase is written

Follow the `codebase-rules` skill — it carries the build order and the rulings
where the engineering skills disagree. Read it before starting a new module.

Project specifics that override it:

- Zig 0.16.0 exactly. No version shims for older or newer compilers.
- `zig build test` diffs the emitted CPU assembly against `tests/codegen.zig`.
  A codegen change is a failing test, not a warning — fix it or update the
  expectation deliberately.
- The public interface is additive. Names in `src/root.zig` and
  `src/device.zig` get added, not removed or renamed.
- Those two roots are mirrors. A name added to one belongs in the other unless
  it is one of the six documented polarity switches.
