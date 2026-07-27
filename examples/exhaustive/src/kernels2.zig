//! A SECOND kernel root, compiled to its own artifact.
//!
//! Exists so the multi-root pipeline is exercised on every run of this example:
//! separate blob, merged name map, and a kernel reachable only by a name chosen
//! at run time. Kernel names must not collide with `kernels.zig`.

const g = @import("gompute");

pub const Params = struct { offset: f32 };

fn addOffset(x: f32, p: Params) f32 {
    return x + p.offset;
}
pub const add_offset = g.map("add_offset", f32, Params, addOffset, .{ .block_size = 256 });

fn rawTriple(data: g.GlobalPtr(f32), len: u64) callconv(g.kernel_callconv) void {
    const i = g.globalIdX(256);
    if (i < len) data[i] = data[i] * 3;
}

comptime {
    if (g.is_device) g.exportRaw("raw_triple", &rawTriple);
    g.exportKernels(@This());
}
