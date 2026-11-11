//! Every `g.math` entry point, exported for four targets. Never run.
//!
//! The same bits on the host, NVPTX, AMDGCN and at comptime survive an
//! explicit `@mulAdd` (one rounding everywhere: hardware, compiler_rt, the
//! comptime folder) but not OPTIONAL contraction: `llvm.fmuladd` or a
//! `contract`/`fast` flag lets each backend decide whether `a * b + c` rounds
//! once or twice. `build.zig` reads this file's IR for both; the `canary`
//! build compiles one optimized-mode `a * b + c`, which must show up, or the
//! check is blind on that target.

const m = @import("math");
const canary = @import("options").canary;

fn expFamily(x: anytype) @TypeOf(x) {
    return m.exp(x) + m.exp2(x) + m.expm1(x) + m.pow(x, x);
}
fn logFamily(x: anytype) @TypeOf(x) {
    return m.log(x) + m.log2(x) + m.log10(x) + m.log1p(x) + m.sqrt(x) + m.rsqrt(x);
}
fn trigFamily(x: anytype) @TypeOf(x) {
    return m.sin(x) + m.cos(x) + m.tan(x) + m.atan(x) + m.atan2(x, x);
}
fn hypFamily(x: anytype) @TypeOf(x) {
    return m.tanh(x) + m.sinh(x) + m.cosh(x) + m.asinh(x) + m.acosh(x) + m.atanh(x);
}
fn contracted(x: anytype) @TypeOf(x) {
    @setFloatMode(.optimized);
    return x * x + x;
}

fn Entry(comptime T: type, comptime f: anytype) type {
    return struct {
        fn run(p: *T) callconv(.c) void {
            p.* = f(p.*);
        }
    };
}

comptime {
    const families = if (canary) .{contracted} else .{ expFamily, logFamily, trigFamily, hypFamily };
    for (families, 0..) |f, i| for (.{ f32, f64, @Vector(8, f32), @Vector(4, f64) }, 0..) |T, j| {
        @export(&Entry(T, f).run, .{ .name = std.fmt.comptimePrint("f{d}_{d}", .{ i, j }) });
    };
}

const std = @import("std");
