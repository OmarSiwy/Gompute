//! Bandwidth and overhead of the generated kernels, on the GPU through
//! `alloc` + `launch` on a non-blocking stream (events time the device side
//! only), and on the CPU for the concrete and the generic body. Prints what
//! OVERHEAD.md quotes; the GPU rows also say what fraction of the card's peak
//! memory bandwidth (from `Context.info`) each kernel reaches.
//!
//! Build with `-Doptimize=ReleaseFast`; the Debug numbers measure nothing.

const std = @import("std");
const g = @import("gompute");
const k = @import("kernels");

const print = std.debug.print;
const sizes = [_]usize{ 1 << 10, 1 << 20, 1 << 24 };
const reps = 20;

pub fn main(init: std.process.Init) !void {
    try gpu();
    try ab();
    try cpu(init.io);
}

// ---------------------------------------------------------------------------
// GPU. One handle per kernel, all on one non-blocking stream; the buffers are
// the largest size and every case reads a prefix of them.
// ---------------------------------------------------------------------------

const B = .cuda;
const Case = enum { map_f32, zip_f32, sum_f32, map_f64, map_i16, exp_f32 };

const Gpu = struct {
    map: g.Kernel(k.map_f32, B),
    zip: g.Kernel(k.zip_f32, B),
    sum: g.Kernel(k.sum_f32, B),
    map64: g.Kernel(k.map_f64, B),
    map16: g.Kernel(k.map_i16, B),
    exp: g.Kernel(k.exp_f32, B),
    a: g.runtime.cuda.Buffer,
    b: g.runtime.cuda.Buffer,
    c: g.runtime.cuda.Buffer,
    partials: g.runtime.cuda.Buffer,
    n: usize = 0,
    which: Case = .map_f32,

    /// Bytes a `which` launch over `n` elements must move: every byte read
    /// once and written once, the speed-of-light count.
    fn bytes(self: *const Gpu) usize {
        return switch (self.which) {
            .map_f32, .exp_f32 => self.n * 4 * 2,
            .zip_f32 => self.n * 4 * 3,
            .sum_f32 => self.n * 4,
            .map_f64 => self.n * 8 * 2,
            .map_i16 => self.n * 2 * 2,
        };
    }

    fn launch(self: *Gpu) !void {
        const p: k.P = .{ .scale = 1.5 };
        switch (self.which) {
            .map_f32 => try self.map.launch(&self.a, self.n, p),
            .zip_f32 => try self.zip.launch(&self.a, &self.b, &self.c, self.n, p),
            .sum_f32 => try self.sum.launch(&self.a, self.n, &self.partials, @TypeOf(self.sum).reduceBlocks(self.n), p),
            .map_f64 => try self.map64.launch(&self.a, self.n, .{ .scale = 1.5 }),
            .map_i16 => try self.map16.launch(&self.a, self.n, p),
            .exp_f32 => try self.exp.launch(&self.a, self.n, p),
        }
    }
};

var dev: Gpu = undefined;

fn launchCurrent() !void {
    return dev.launch();
}

/// Microseconds per launch of `launch_fn` over `reps` back-to-back launches,
/// measured with events on `stream`.
fn timeLaunches(ctx: *g.runtime.cuda.Context, stream: *g.runtime.cuda.Stream, launch_fn: *const fn () anyerror!void) !f64 {
    var start = try ctx.createEvent(true);
    defer start.deinit();
    var end = try ctx.createEvent(true);
    defer end.deinit();
    try launch_fn(); // warm: JIT, first touch, instruction cache
    try stream.synchronize();
    try start.record(stream);
    for (0..reps) |_| try launch_fn();
    try end.record(stream);
    try end.synchronize();
    return @as(f64, try g.runtime.cuda.Event.elapsedUs(&start, &end)) / reps;
}

fn gpu() !void {
    if (comptime !g.AutoKernel(k.map_f32).Cuda.available) return print("GPU: this build emitted no CUDA artifacts\n", .{});
    dev.map = g.Kernel(k.map_f32, B).init(0) catch return print("GPU: no CUDA device\n", .{});
    defer dev.map.deinit();
    const info = try dev.map.context.info();
    const peak = info.bandwidth();
    print("GPU: {s}, {d} SMs, peak {d:.0} GB/s (bus {d} bit at {d} MHz)\n", .{ info.name(), info.sms, @as(f64, @floatFromInt(peak)) / 1e9, info.bus_width_bits, info.mem_clock_khz / 1000 });
    var stream = try dev.map.context.createStreamNonBlocking();
    defer stream.deinit();
    dev.zip = try g.Kernel(k.zip_f32, B).init(0);
    defer dev.zip.deinit();
    dev.sum = try g.Kernel(k.sum_f32, B).init(0);
    defer dev.sum.deinit();
    dev.map64 = try g.Kernel(k.map_f64, B).init(0);
    defer dev.map64.deinit();
    dev.map16 = try g.Kernel(k.map_i16, B).init(0);
    defer dev.map16.deinit();
    dev.exp = try g.Kernel(k.exp_f32, B).init(0);
    defer dev.exp.deinit();
    dev.map.stream = stream;
    dev.zip.stream = stream;
    dev.sum.stream = stream;
    dev.map64.stream = stream;
    dev.map16.stream = stream;
    dev.exp.stream = stream;

    const max = sizes[sizes.len - 1];
    dev.a = try dev.map.context.alloc(max * 8);
    defer dev.a.free();
    dev.b = try dev.map.context.alloc(max * 8);
    defer dev.b.free();
    dev.c = try dev.map.context.alloc(max * 8);
    defer dev.c.free();
    dev.partials = try dev.map.context.alloc(1024 * 4);
    defer dev.partials.free();
    try dev.a.fillAsync(0, max * 8, &stream);
    try dev.b.fillAsync(0, max * 8, &stream);

    for (sizes) |n| {
        dev.n = n;
        inline for (std.enums.values(Case)) |which| {
            dev.which = which;
            const us = try timeLaunches(&dev.map.context, &stream, launchCurrent);
            const gbs = @as(f64, @floatFromInt(dev.bytes())) / (us * 1e3);
            const pct = if (peak == 0) 0 else gbs * 1e9 / @as(f64, @floatFromInt(peak)) * 100;
            print("  {s:<8} n={d:>9}  {d:>9.2} us  {d:>7.1} GB/s  {d:>5.1}% of peak\n", .{ @tagName(which), n, us, gbs, pct });
        }
    }
    try stream.synchronize();
}

// ---------------------------------------------------------------------------
// A/B: the 1.4.0 one-element-per-thread shape (raw kernels in kernels.zig)
// against the generated 16-byte-lane entries, alternated in one process so
// clock drift hits both the same. Five rounds; read the minimum of each.
// ---------------------------------------------------------------------------

const Raw = g.RawKernel("raw_map_f32", B);
var raw_which: u8 = 0;
var raw_n: usize = 0;
var raw_map32: Raw = undefined;
var raw_map64: g.RawKernel("raw_map_f64", B) = undefined;
var raw_zip32: g.RawKernel("raw_zip_f32", B) = undefined;
var raw_stream: g.runtime.cuda.Stream = undefined;

fn launchRaw() !void {
    var len: u64 = raw_n;
    var scale32: f32 = 1.5;
    var scale64: f64 = 1.5;
    const grid = g.Dim3.linear(raw_n, 256);
    switch (raw_which) {
        0 => try raw_map32.launchOn(&raw_stream, grid, .{ .x = 256 }, 0, &.{ dev.a.argPtr(), g.interface.arg(&len), g.interface.arg(&scale32) }),
        1 => try raw_map64.launchOn(&raw_stream, grid, .{ .x = 256 }, 0, &.{ dev.a.argPtr(), g.interface.arg(&len), g.interface.arg(&scale64) }),
        else => try raw_zip32.launchOn(&raw_stream, grid, .{ .x = 256 }, 0, &.{ dev.a.argPtr(), dev.b.argPtr(), dev.c.argPtr(), g.interface.arg(&len), g.interface.arg(&scale32) }),
    }
}

fn ab() !void {
    if (comptime !g.AutoKernel(k.map_f32).Cuda.available) return;
    dev.map = g.Kernel(k.map_f32, B).init(0) catch return;
    defer dev.map.deinit();
    dev.zip = try g.Kernel(k.zip_f32, B).init(0);
    defer dev.zip.deinit();
    dev.map64 = try g.Kernel(k.map_f64, B).init(0);
    defer dev.map64.deinit();
    raw_map32 = try Raw.init(0);
    defer raw_map32.deinit();
    raw_map64 = try @TypeOf(raw_map64).init(0);
    defer raw_map64.deinit();
    raw_zip32 = try @TypeOf(raw_zip32).init(0);
    defer raw_zip32.deinit();
    raw_stream = try dev.map.context.createStreamNonBlocking();
    defer raw_stream.deinit();
    dev.map.stream = raw_stream;
    dev.zip.stream = raw_stream;
    dev.map64.stream = raw_stream;
    const max = sizes[sizes.len - 1];
    dev.a = try dev.map.context.alloc(max * 8);
    defer dev.a.free();
    dev.b = try dev.map.context.alloc(max * 8);
    defer dev.b.free();
    dev.c = try dev.map.context.alloc(max * 8);
    defer dev.c.free();
    try dev.a.fillAsync(0, max * 8, &raw_stream);
    try dev.b.fillAsync(0, max * 8, &raw_stream);

    print("A/B, one process, alternating; us per launch, 5 rounds (1 per thread | 16 bytes per thread)\n", .{});
    for ([_]usize{ 1 << 20, 1 << 24 }) |n| {
        dev.n = n;
        raw_n = n;
        inline for (.{ .{ "map f32", 0, Case.map_f32 }, .{ "map f64", 1, Case.map_f64 }, .{ "zip f32", 2, Case.zip_f32 } }) |c| {
            raw_which = c[1];
            dev.which = c[2];
            var old_min: f64 = 1e30;
            var new_min: f64 = 1e30;
            print("  {s:<8} n={d:>9} ", .{ c[0], n });
            for (0..5) |_| {
                const old = try timeLaunches(&dev.map.context, &raw_stream, launchRaw);
                const new = try timeLaunches(&dev.map.context, &raw_stream, launchCurrent);
                old_min = @min(old_min, old);
                new_min = @min(new_min, new);
                print(" {d:.0}|{d:.0}", .{ old, new });
            }
            print("   min {d:.1} | {d:.1} us  ({d:.2}x)\n", .{ old_min, new_min, old_min / new_min });
        }
    }
}

// ---------------------------------------------------------------------------
// CPU. The concrete body against the generic one, now that 0.18's loop
// vectorizer sees the concrete loop too.
// ---------------------------------------------------------------------------

const CpuCase = enum { map_concrete, map_generic, zip_concrete, zip_generic, sum_preset, exp_concrete };

const Cpu = struct {
    a: []f32,
    b: []f32,
    c: []f32,
    sink: f32 = 0,
    which: CpuCase = .map_concrete,

    fn run(self: *Cpu) !void {
        const p: k.P = .{ .scale = 1 };
        switch (self.which) {
            .map_concrete => {
                var kk = try g.Kernel(k.map_f32, .cpu).init(0);
                try kk.run(self.a, p);
            },
            .map_generic => {
                var kk = try g.Kernel(k.map_f32_v, .cpu).init(0);
                try kk.run(self.a, p);
            },
            .zip_concrete => {
                var kk = try g.Kernel(k.zip_f32, .cpu).init(0);
                try kk.run(self.a, self.b, self.c, p);
            },
            .zip_generic => {
                var kk = try g.Kernel(k.zip_f32_v, .cpu).init(0);
                try kk.run(self.a, self.b, self.c, p);
            },
            .sum_preset => {
                var kk = try g.Kernel(k.sum_f32, .cpu).init(0);
                self.sink += try kk.run(self.a, p);
            },
            .exp_concrete => {
                var kk = try g.Kernel(k.exp_f32, .cpu).init(0);
                try kk.run(self.a, p);
            },
        }
    }
};

var host: Cpu = undefined;

fn runCurrent() !void {
    return host.run();
}

/// Median of `runs` monotonic-clock timings of `f`, in nanoseconds per element.
fn cpuTime(io: std.Io, n: usize, comptime runs: usize, f: *const fn () anyerror!void) !f64 {
    var t: [runs]i96 = undefined;
    for (&t) |*slot| {
        const start = std.Io.Clock.Timestamp.now(io, .awake);
        try f();
        slot.* = start.durationTo(std.Io.Clock.Timestamp.now(io, .awake)).raw.nanoseconds;
    }
    std.mem.sort(i96, &t, {}, std.sort.asc(i96));
    return @as(f64, @floatFromInt(t[runs / 2])) / @as(f64, @floatFromInt(n));
}

fn cpu(io: std.Io) !void {
    const gpa = std.heap.page_allocator;
    print("CPU: median of 11, ns per element\n", .{});
    for ([_]usize{ 1 << 18, 1 << 24 }) |n| {
        host.a = try gpa.alloc(f32, n);
        defer gpa.free(host.a);
        host.b = try gpa.alloc(f32, n);
        defer gpa.free(host.b);
        host.c = try gpa.alloc(f32, n);
        defer gpa.free(host.c);
        for (host.a, host.b, 0..) |*x, *y, i| {
            x.* = @as(f32, @floatFromInt(i % 1000)) * 1e-3 - 0.5;
            y.* = 1;
        }
        inline for (std.enums.values(CpuCase)) |which| {
            host.which = which;
            const ns = try cpuTime(io, n, 11, runCurrent);
            print("  {s:<13} n={d:>9}  {d:>6.3} ns/elem\n", .{ @tagName(which), n, ns });
        }
    }
    std.mem.doNotOptimizeAway(host.sink);
}
