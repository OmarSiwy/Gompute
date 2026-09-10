//! Codegen probe: one `export fn` per way of building a spec, plus a hand-written
//! loop beside the two that are compared.
//!
//! Nothing links or runs this object. `build.zig` takes only its
//! `getEmittedAsm()` and hands that to `tools/codegen_check.zig`, which asserts
//! that `gompute_scale_relu` matches `manual_scale_relu` and `gompute_zip`
//! matches `manual_zip` -- the library's central claim, that
//! `Kernel(Spec, .cpu)` leaves no runtime residue. Those four names are the
//! interface: rename one and `zig build test` fails with SymbolNotFound, change
//! one body without the other and it reports a mismatch.
//!
//! The exports nothing compares are not spare. They are the only thing that
//! forces zip/sum/mapTo/mapIndexed/gather/scatter/mapFn and the generic-body map
//! through the CPU backend at all; delete one and that constructor stops being
//! compiled anywhere.
//!
//! `Kernel(Spec, .cpu).init(0)` is restated in every export deliberately. A
//! shared helper would put a call between the export boundary and the loop,
//! which is the inlining this file exists to measure.

const g = @import("gompute");

const Params = struct { scale: f32 };
fn op(x: f32, p: Params) f32 {
    const y = x * p.scale;
    return if (y > 0) y else 0;
}
const Spec = g.map("codegen_scale_relu", f32, Params, op, .{});

export fn gompute_scale_relu(data: [*]f32, len: usize, scale: f32) void {
    var kernel = g.Kernel(Spec, .cpu).init(0) catch unreachable;
    kernel.run(data[0..len], .{ .scale = scale }) catch unreachable;
}

export fn manual_scale_relu(data: [*]f32, len: usize, scale: f32) void {
    for (data[0..len]) |*value| {
        const y = value.* * scale;
        value.* = if (y > 0) y else 0;
    }
}

/// Same operation written generically: the CPU backend runs it at vector width.
fn opGeneric(x: anytype, p: Params) @TypeOf(x) {
    const V = @TypeOf(x);
    return @max(x * g.splat(V, p.scale), g.splat(V, @as(f32, 0)));
}
const SimdSpec = g.map("codegen_scale_relu_simd", f32, Params, opGeneric, .{});

export fn gompute_scale_relu_simd(data: [*]f32, len: usize, scale: f32) void {
    var kernel = g.Kernel(SimdSpec, .cpu).init(0) catch unreachable;
    kernel.run(data[0..len], .{ .scale = scale }) catch unreachable;
}

/// Inferring constructor: T and Params come from `op`'s signature.
const InferredSpec = g.mapFn("codegen_inferred", op, .{ .block_size = 128 });

export fn gompute_inferred(data: [*]f32, len: usize, scale: f32) void {
    var kernel = g.Kernel(InferredSpec, .cpu).init(0) catch unreachable;
    kernel.run(data[0..len], .{ .scale = scale }) catch unreachable;
}

// ── The rest of the operation set, same test: `Kernel(Spec, .cpu)` has to
// come out as the loop next to it, with no dispatch left over. ──────────────

fn axpy(a: f32, b: f32, p: Params) f32 {
    return a + b * p.scale;
}
const ZipSpec = g.zip("codegen_zip", f32, f32, f32, Params, axpy, .{});

export fn gompute_zip(a: [*]const f32, b: [*]const f32, out: [*]f32, len: usize, scale: f32) void {
    var kernel = g.Kernel(ZipSpec, .cpu).init(0) catch unreachable;
    kernel.run(a[0..len], b[0..len], out[0..len], .{ .scale = scale }) catch unreachable;
}

export fn manual_zip(a: [*]const f32, b: [*]const f32, out: [*]f32, len: usize, scale: f32) void {
    for (a[0..len], b[0..len], out[0..len]) |x, y, *o| o.* = x + y * scale;
}

const SumSpec = g.sum("codegen_sum", f32, Params, .{});

export fn gompute_sum(data: [*]const f32, len: usize) f32 {
    var kernel = g.Kernel(SumSpec, .cpu).init(0) catch unreachable;
    return kernel.run(data[0..len], .{ .scale = 0 }) catch unreachable;
}

fn toU32(x: f32, p: Params) u32 {
    return @intFromFloat(@max(x * p.scale, 0));
}
const MapToSpec = g.mapTo("codegen_map_to", f32, u32, Params, toU32, .{});

export fn gompute_map_to(in: [*]const f32, out: [*]u32, len: usize, scale: f32) void {
    var kernel = g.Kernel(MapToSpec, .cpu).init(0) catch unreachable;
    kernel.run(in[0..len], out[0..len], .{ .scale = scale }) catch unreachable;
}

fn stripe(x: f32, i: u64, p: Params) f32 {
    return x + @as(f32, @floatFromInt(i)) * p.scale;
}
const IndexedSpec = g.mapIndexed("codegen_indexed", f32, Params, stripe, .{});

export fn gompute_indexed(data: [*]f32, len: usize, scale: f32) void {
    var kernel = g.Kernel(IndexedSpec, .cpu).init(0) catch unreachable;
    kernel.run(data[0..len], .{ .scale = scale }) catch unreachable;
}

const GatherSpec = g.gather("codegen_gather", f32, u32, .{});

export fn gompute_gather(src: [*]const f32, idx: [*]const u32, out: [*]f32, n: usize, src_len: usize) void {
    var kernel = g.Kernel(GatherSpec, .cpu).init(0) catch unreachable;
    kernel.run(src[0..src_len], idx[0..n], out[0..n]) catch unreachable;
}

const ScatterSpec = g.scatter("codegen_scatter", f32, u32, .{});

export fn gompute_scatter(src: [*]const f32, idx: [*]const u32, out: [*]f32, n: usize, out_len: usize) void {
    var kernel = g.Kernel(ScatterSpec, .cpu).init(0) catch unreachable;
    kernel.run(src[0..n], idx[0..n], out[0..out_len]) catch unreachable;
}
