//! Bandwidth of plain raw kernels on the GPU, through `launchOn` on a
//! non-blocking stream with events timing the device side only, as a fraction
//! of the card's peak memory bandwidth (from `Context.info`); and `g.math`
//! throughput with `BENCH=math`.
//!
//! Build with `-Doptimize=ReleaseFast`; the Debug numbers measure nothing.

const std = @import("std");
const g = @import("gompute");
const k = @import("kernels");

const print = std.debug.print;
const sizes = [_]usize{ 1 << 10, 1 << 20, 1 << 24 };
const reps = 20;
const B = .cuda;

pub fn main(init: std.process.Init) !void {
    if (init.environ_map.get("BENCH")) |which| if (std.mem.eql(u8, which, "math")) return mathBench(init.io);
    try gpu();
}

const Case = enum { map_f32, map_f64, zip_f32 };

var map32: g.RawKernel("raw_map_f32", B) = undefined;
var map64: g.RawKernel("raw_map_f64", B) = undefined;
var zip32: g.RawKernel("raw_zip_f32", B) = undefined;
var stream: g.runtime.cuda.Stream = undefined;
var bufs: [3]g.runtime.cuda.Buffer = undefined;
var cur_n: usize = 0;
var cur: Case = .map_f32;

/// Bytes a launch over `cur_n` elements must move: every byte read once and
/// written once, the speed-of-light count.
fn bytes() usize {
    return switch (cur) {
        .map_f32 => cur_n * 4 * 2,
        .map_f64 => cur_n * 8 * 2,
        .zip_f32 => cur_n * 4 * 3,
    };
}

fn launchCurrent() !void {
    var len: u64 = cur_n;
    var scale32: f32 = 1.5;
    var scale64: f64 = 1.5;
    const grid = g.Dim3.linear(cur_n, 256);
    switch (cur) {
        .map_f32 => try map32.launchOn(&stream, grid, .{ .x = 256 }, 0, &.{ bufs[0].argPtr(), g.interface.arg(&len), g.interface.arg(&scale32) }),
        .map_f64 => try map64.launchOn(&stream, grid, .{ .x = 256 }, 0, &.{ bufs[0].argPtr(), g.interface.arg(&len), g.interface.arg(&scale64) }),
        .zip_f32 => try zip32.launchOn(&stream, grid, .{ .x = 256 }, 0, &.{ bufs[0].argPtr(), bufs[1].argPtr(), bufs[2].argPtr(), g.interface.arg(&len), g.interface.arg(&scale32) }),
    }
}

/// Microseconds per launch of `launch_fn` over `reps` back-to-back launches,
/// measured with events on `on`.
fn timeLaunches(ctx: *g.runtime.cuda.Context, on: *g.runtime.cuda.Stream, launch_fn: *const fn () anyerror!void) !f64 {
    var start = try ctx.createEvent(true);
    defer start.deinit();
    var end = try ctx.createEvent(true);
    defer end.deinit();
    try launch_fn(); // warm: JIT, first touch, instruction cache
    try on.synchronize();
    try start.record(on);
    for (0..reps) |_| try launch_fn();
    try end.record(on);
    try end.synchronize();
    return @as(f64, try g.runtime.cuda.Event.elapsedUs(&start, &end)) / reps;
}

fn gpu() !void {
    if (comptime !g.available(B)) return print("GPU: this build emitted no CUDA artifacts\n", .{});
    map32 = @TypeOf(map32).init(0) catch return print("GPU: no CUDA device\n", .{});
    defer map32.deinit();
    map64 = try @TypeOf(map64).init(0);
    defer map64.deinit();
    zip32 = try @TypeOf(zip32).init(0);
    defer zip32.deinit();
    const info = try map32.context.info();
    const peak = info.bandwidth();
    print("GPU: {s}, {d} SMs, peak {d:.0} GB/s (bus {d} bit at {d} MHz)\n", .{ info.name(), info.sms, @as(f64, @floatFromInt(peak)) / 1e9, info.bus_width_bits, info.mem_clock_khz / 1000 });
    stream = try map32.createStreamNonBlocking();
    defer stream.deinit();
    const max = sizes[sizes.len - 1];
    for (&bufs) |*b| {
        b.* = try map32.alloc(max * 8);
        try b.fillAsync(0, max * 8, &stream);
    }
    defer for (&bufs) |*b| b.free();

    for (sizes) |n| {
        cur_n = n;
        inline for (std.enums.values(Case)) |which| {
            cur = which;
            const us = try timeLaunches(&map32.context, &stream, launchCurrent);
            const gbs = @as(f64, @floatFromInt(bytes())) / (us * 1e3);
            const pct = if (peak == 0) 0 else gbs * 1e9 / @as(f64, @floatFromInt(peak)) * 100;
            print("  {s:<8} n={d:>9}  {d:>9.2} us  {d:>7.1} GB/s  {d:>5.1}% of peak\n", .{ @tagName(which), n, us, gbs, pct });
        }
    }
    try stream.synchronize();
}

// ---------------------------------------------------------------------------
// g.math throughput: `BENCH=math zig build run -Doptimize=ReleaseFast`. Each
// element makes 8 calls (see kernels.zig), so these are calls per second, not
// bytes. GPU: best of 5 event-timed rounds. CPU: the scalar call in a plain
// loop and the vector form 8 (f32) or 4 (f64) lanes wide, median of 5.
// ---------------------------------------------------------------------------

const math_n_gpu = 1 << 22;
const math_n_cpu = 1 << 16;

fn fillMath(comptime T: type, xs: []T) void {
    for (xs, 0..) |*x, i| x.* = 0.1 + @as(T, @floatFromInt(i % 3901)) * (3.9 / 3901.0);
}

fn mathGpuOne(comptime T: type, comptime name: []const u8, on: *g.runtime.cuda.Stream, buf: *g.runtime.cuda.Buffer, host_in: []const T) !f64 {
    var kk = try g.RawKernel("bm_" ++ name ++ "_" ++ @typeName(T), B).init(0);
    defer kk.deinit();
    var len: u64 = host_in.len;
    var best: f64 = 1e30;
    for (0..5) |_| {
        try buf.upload(host_in.ptr, host_in.len * @sizeOf(T));
        var start = try kk.createEvent(true);
        defer start.deinit();
        var end = try kk.createEvent(true);
        defer end.deinit();
        try start.record(on);
        try kk.launchOn(on, g.Dim3.linear(host_in.len, 256), .{ .x = 256 }, 0, &.{ buf.argPtr(), g.interface.arg(&len) });
        try end.record(on);
        try end.synchronize();
        best = @min(best, @as(f64, try g.runtime.cuda.Event.elapsedUs(&start, &end)));
    }
    return best;
}

/// `w` lanes at a time (1: the scalar call in a plain loop), median of 5, ns
/// per call.
fn mathCpuOne(comptime T: type, comptime name: []const u8, comptime w: usize, io: std.Io, xs: []T, src: []const T) !f64 {
    var t: [5]i96 = undefined;
    for (&t) |*slot| {
        @memcpy(xs, src);
        const start = std.Io.Clock.Timestamp.now(io, .awake);
        if (w == 1) {
            for (xs) |*x| x.* = k.mathBody(name, x.*);
        } else {
            var i: usize = 0;
            while (i < xs.len) : (i += w) {
                const v: @Vector(w, T) = xs[i..][0..w].*;
                xs[i..][0..w].* = k.mathBody(name, v);
            }
        }
        slot.* = start.durationTo(std.Io.Clock.Timestamp.now(io, .awake)).raw.nanoseconds;
    }
    std.mem.doNotOptimizeAway(xs[0]);
    std.mem.sort(i96, &t, {}, std.sort.asc(i96));
    return @as(f64, @floatFromInt(t[2])) / @as(f64, @floatFromInt(xs.len * 8));
}

fn mathBench(io: std.Io) !void {
    const gpa = std.heap.page_allocator;
    var ctx: ?g.runtime.cuda.Context = if (comptime g.available(B)) g.runtime.cuda.Context.init(0) catch null else null;
    defer if (ctx) |*c| c.deinit();
    print("g.math: GPU Gcalls/s (n={d}) | CPU ns/call scalar, vector (n={d})\n", .{ math_n_gpu, math_n_cpu });
    inline for (.{ f32, f64 }) |T| {
        const w = if (T == f32) 8 else 4;
        const host_in = try gpa.alloc(T, math_n_gpu);
        defer gpa.free(host_in);
        fillMath(T, host_in);
        const xs = try gpa.alloc(T, math_n_cpu);
        defer gpa.free(xs);
        var buf: ?g.runtime.cuda.Buffer = null;
        var on: ?g.runtime.cuda.Stream = null;
        if (ctx) |*c| {
            buf = try c.alloc(math_n_gpu * @sizeOf(T));
            on = try c.createStreamNonBlocking();
        }
        defer if (buf) |*b| b.free();
        defer if (on) |*st| st.deinit();
        inline for (k.math_fns) |name| {
            const gcalls = if (buf) |*b| @as(f64, math_n_gpu * 8) / (try mathGpuOne(T, name, &on.?, b, host_in)) / 1e3 else 0;
            const s_ns = try mathCpuOne(T, name, 1, io, xs, host_in[0..math_n_cpu]);
            const v_ns = try mathCpuOne(T, name, w, io, xs, host_in[0..math_n_cpu]);
            print("  {s:<6} {s}  {d:>8.2} Gcalls/s  | {d:>6.2} {d:>6.2} ns\n", .{ name, @typeName(T), gcalls, s_ns, v_ns });
        }
    }
}
