//! One spec of every `Kind`, exported for a GPU target.
//!
//! This file is never run. It exists so that `zig build test` puts all six
//! entry generators in `src/device/export.zig` through the Zig frontend for
//! `nvptx64` -- before this, only `Entry` (the `map` shape) had ever been
//! compiled for a device at all, because every kernels root in the repo and in
//! both examples uses `g.map` exclusively. `ReduceEntry`'s
//! `addrspace(.shared)` scratch array, `builtins.barrier`, `builtins.localIdX`
//! and `builtins.blockIdX` had no compiled call site anywhere.
//!
//! Deliberately does NOT name `g.math`: that module is mid-rewrite, and a
//! failure there is not a failure of this block.

const g = @import("gompute");

const Scale = struct { k: f32 };
const NoParams = struct {};

fn scale(x: f32, p: Scale) f32 {
    return x * p.k;
}
fn scaleAt(x: f32, i: u64, p: Scale) f32 {
    return x * p.k + @as(f32, @floatFromInt(i));
}
fn toInt(x: f32, _: Scale) i32 {
    return @intFromFloat(x);
}
fn addUp(a: f32, b: f32, _: Scale) f32 {
    return a + b;
}
fn maxOf(a: f32, b: f32) f32 {
    return @max(a, b);
}
fn negate(x: f32, _: NoParams) f32 {
    return -x;
}

pub const map_entry = g.map("t_map", f32, Scale, scale, .{ .block_size = 256 });
pub const indexed_entry = g.mapIndexed("t_map_indexed", f32, Scale, scaleAt, .{ .block_size = 128 });
pub const map_to_entry = g.mapTo("t_map_to", f32, i32, Scale, toInt, .{ .block_size = 64 });
pub const zip_entry = g.zip("t_zip", f32, f32, f32, Scale, addUp, .{ .block_size = 256 });
pub const reduce_entry = g.reduce("t_reduce", f32, Scale, maxOf, -1e30, .{ .block_size = 256 });
pub const sum_entry = g.sum("t_sum", f32, Scale, .{ .block_size = 100 });
pub const gather_entry = g.gather("t_gather", f32, u32, .{ .block_size = 256 });
pub const scatter_entry = g.scatter("t_scatter", f32, u32, .{ .block_size = 256 });

/// `abi.Boundary(struct{})` is a zero-field extern struct, so the emitted entry
/// takes one fewer parameter than the host pushes. Compiled here so the shape
/// at least exists in an artifact; see the launch-frame note in `exportOne`.
pub const empty_params_entry = g.map("t_no_params", f32, NoParams, negate, .{ .block_size = 256 });

comptime {
    g.exportKernels(@This());
}
