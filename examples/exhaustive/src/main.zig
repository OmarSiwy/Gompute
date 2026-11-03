//! The GPU integration suite, run by hand: `cd examples/exhaustive && zig build
//! run`. Nothing in CI builds it, so it can break and stay broken -- run it
//! after touching anything on the host or device path.
//!
//! Exits 1 on any failure. A section that hits an error part-way through counts
//! as a failure too; see `section` for why that needs saying.

const std = @import("std");
const g = @import("gompute");
const k2 = @import("kernels2");

const print = std.debug.print;

var pass_count: u32 = 0;
var fail_count: u32 = 0;
var section_skipped = false;

fn check(name: []const u8, ok: bool) void {
    if (ok) {
        pass_count += 1;
    } else {
        fail_count += 1;
        print("  FAIL: {s}\n", .{name});
    }
}

/// Every GPU section bails out of a `catch` with a bare `return`, which records
/// nothing -- so a broken driver used to print "55 passed, 0 failed" and exit 0.
/// A section therefore declares how many checks it must reach, and reaching a
/// different number is itself a failure. One floor covers every bail site at
/// once, including bail sites nobody has written yet.
///
/// The count is exact, not a minimum, so adding a `check` to a section without
/// updating its floor fails loudly instead of quietly widening the hole.
fn section(name: []const u8, expected_checks: u32, comptime body: fn () void) void {
    const before = pass_count + fail_count;
    section_skipped = false;
    body();
    if (section_skipped) return;
    const reached = pass_count + fail_count - before;
    if (reached != expected_checks) {
        fail_count += 1;
        print("  FAIL: {s} reached {d} of {d} checks\n", .{ name, reached, expected_checks });
    }
}

/// The one exit from a section that is not a failure: the hardware or the build
/// artifacts it needs do not exist on this machine. Asserts nothing about what
/// already ran, so call it before the section's first `check`.
fn skip(reason: []const u8) void {
    section_skipped = true;
    print("  skipped: {s}\n", .{reason});
}

/// Floats compare with an absolute tolerance, everything else exactly. The
/// tolerance is deliberately loose: these check that the kernel ran the right
/// operation, not how it rounded. Scaling it off `floatEps` keeps f64 from
/// silently inheriting a tolerance picked for f32.
fn same(comptime T: type, a: T, b: T) bool {
    return switch (@typeInfo(T)) {
        .float => std.math.approxEqAbs(T, a, b, 1024 * std.math.floatEps(T)),
        .vector => |v| blk: {
            inline for (0..v.len) |l| if (!same(v.child, a[l], b[l])) break :blk false;
            break :blk true;
        },
        else => a == b,
    };
}

fn allSame(comptime T: type, actual: []const T, expected: []const T) bool {
    if (actual.len != expected.len) return false;
    for (actual, expected) |a, e| if (!same(T, a, e)) return false;
    return true;
}

pub fn main() !void {
    print("=== Gompute exhaustive GPU test ===\n\n", .{});

    // ── Runtime dynamic backend ─────────────────────────────────────────
    section("runtime.dynamic", 11, testRuntimeDynamic);

    // ── Second kernel root + run-time kernel selection ──────────────────
    section("second root", 5, testSecondRoot);

    // ── Every device at once, and copies between them ───────────────────
    section("devices", 3, testDevices);

    // ── A kernel captured into a graph and replayed ─────────────────────
    section("graph replay", 3, testGraphReplay);

    // ── g.math: the same bits on the GPU as on the CPU ──────────────────
    section("math bit-identity", 2, testMathBits);

    // ── g.math f32: every input, on the GPU, for the correctly rounded ones ─
    section("math f32 sweep", k2.math_cr32.len, testMathSweep);

    // ── Hand-written raw kernel through the artifact pipeline ───────────
    section("raw kernel", 1, testRawKernel);

    // ── Pure comptime: Dim3 ─────────────────────────────────────────────
    testDim3Linear();

    print("\n=== Results: {} passed, {} failed ===\n", .{ pass_count, fail_count });
    if (fail_count > 0) std.process.exit(1);
}

/// One elementwise launch: `data` up, the kernel over it with `extra` (or
/// nothing, for `{}`) after the buffer and length, `data` down. Blocks until
/// done.
fn runOn(raw: anytype, buf: anytype, data: anytype, extra: anytype) !void {
    const bytes = data.len * @sizeOf(@TypeOf(data[0]));
    try buf.upload(data.ptr, bytes);
    var len: u64 = data.len;
    var e = extra;
    const args = if (@TypeOf(extra) == void)
        &[_]g.interface.Arg{ buf.argPtr(), g.interface.arg(&len) }
    else
        &[_]g.interface.Arg{ buf.argPtr(), g.interface.arg(&len), g.interface.arg(&e) };
    try raw.launch(g.Dim3.linear(data.len, 256), .{ .x = 256 }, 0, args);
    try raw.synchronize();
    try buf.download(data.ptr, bytes);
}

/// Two launches recorded once on a non-blocking stream, replayed twice, add
/// the offset four times.
fn testGraphReplay() void {
    print("\n[graph replay]\n", .{});
    if (comptime !g.available(.cuda)) return skip("this build emitted no CUDA artifacts");
    var raw = g.RawKernel("raw_add_offset", .cuda).init(0) catch return skip("no CUDA device");
    defer raw.deinit();
    if (!raw.hasGraphs()) return skip("driver has no graphs");

    var data = [_]f32{ 1, 2, 3 };
    var buf = raw.alloc(@sizeOf(@TypeOf(data))) catch return;
    defer buf.free();
    buf.upload(&data, @sizeOf(@TypeOf(data))) catch return;

    var stream = raw.createStreamNonBlocking() catch return;
    defer stream.deinit();
    var len: u64 = data.len;
    var offset: f32 = 1;
    const args = [_]g.interface.Arg{ buf.argPtr(), g.interface.arg(&len), g.interface.arg(&offset) };
    stream.beginCapture(.thread_local) catch return;
    raw.launchOn(&stream, .{ .x = 1 }, .{ .x = 256 }, 0, &args) catch return;
    raw.launchOn(&stream, .{ .x = 1 }, .{ .x = 256 }, 0, &args) catch return;
    var graph = stream.endCapture() catch return;
    defer graph.deinit();
    // Capture records, it does not run.
    var probe: [3]f32 = undefined;
    buf.download(&probe, @sizeOf(@TypeOf(probe))) catch return;
    check("capture_ran_nothing", allSame(f32, &probe, &data));

    var exec = graph.instantiate() catch return;
    defer exec.deinit();
    for (0..2) |_| exec.launch(&stream) catch return;
    stream.synchronize() catch return;
    buf.download(&data, @sizeOf(@TypeOf(data))) catch return;
    check("replayed_twice", allSame(f32, &data, &[_]f32{ 5, 6, 7 }));
    check("stream_idle", stream.query() catch false);
    print("  {any}\n", .{data});
}

/// Every `g.math` function, f64 and f32, over random bit patterns plus each
/// function's interesting ranges: the CUDA result must be the CPU's, bit for
/// bit.
fn testMathBits() void {
    print("\n[math bit-identity]\n", .{});
    if (comptime !g.available(.cuda)) return skip("this build emitted no CUDA artifacts");
    inline for (.{ f64, f32 }) |T| {
        const U = @Int(.unsigned, @bitSizeOf(T));
        var gpu = g.RawKernel(if (T == f64) "math_f64" else "math_f32", .cuda).init(0) catch return skip("no CUDA device");
        defer gpu.deinit();
        const n = 1 << 16;
        var buf = gpu.alloc(n * @sizeOf(T)) catch return;
        defer buf.free();
        var a: [n]T = undefined;
        var b: [n]T = undefined;
        var prng = std.Random.DefaultPrng.init(0xB17_1D);
        const r = prng.random();
        var mismatches: usize = 0;
        for (0..k2.math_function_count) |f| {
            for (&a, 0..) |*x, i| x.* = switch (i % 4) {
                0 => @bitCast(r.int(U)), // any bit pattern: specials, subnormals
                1 => (r.float(T) - 0.5) * 40,
                2 => (r.float(T) - 0.5) * 1400,
                else => 1 + (r.float(T) - 0.5) * 0.25, // around 1
            };
            const input = a;
            runOn(&gpu, &buf, @as([]T, &a), @as(u32, @intCast(f))) catch return;
            for (&b, input) |*y, x| y.* = k2.mathOf(T)(x, @intCast(f));
            var bad: usize = 0;
            for (a, b, input) |x, y, in| {
                if (@as(U, @bitCast(x)) == @as(U, @bitCast(y)) or (std.math.isNan(x) and std.math.isNan(y))) continue;
                if (bad < 4) print("    x={x} gpu={x} cpu={x}\n", .{ @as(U, @bitCast(in)), @as(U, @bitCast(x)), @as(U, @bitCast(y)) });
                bad += 1;
            }
            if (bad != 0) print("  {s} f{d}: {d} of {d} differ\n", .{ @typeName(T), f, bad, n });
            mismatches += bad;
        }
        check("math_bits_" ++ @typeName(T), mismatches == 0);
    }
    print("  done\n", .{});
}

/// Every one of the 2^32 f32 inputs through each correctly rounded `g.math`
/// function, CUDA against the CPU. The CPU bodies are proven correctly rounded
/// on every input by math.zig's tests; the device bodies differ from them, so
/// this is what proves the GPU codegen (no FTZ, no contraction) is too.
fn testMathSweep() void {
    print("\n[math f32 sweep]\n", .{});
    if (comptime !g.available(.cuda)) return skip("this build emitted no CUDA artifacts");
    var gpu = g.RawKernel("math_f32", .cuda).init(0) catch return skip("no CUDA device");
    defer gpu.deinit();
    const chunk = 1 << 24;
    var buf = gpu.alloc(chunk * @sizeOf(f32)) catch return;
    defer buf.free();
    const a = std.heap.page_allocator.alloc(f32, chunk) catch return;
    defer std.heap.page_allocator.free(a);
    for (k2.math_cr32) |f| {
        var bad: usize = 0;
        var base: u64 = 0;
        while (base < 1 << 32) : (base += chunk) {
            for (a, 0..) |*x, i| x.* = @bitCast(@as(u32, @intCast(base + i)));
            runOn(&gpu, &buf, a, f) catch return;
            for (a, 0..) |x, i| {
                const in: f32 = @bitCast(@as(u32, @intCast(base + i)));
                const y = k2.mathOf(f32)(in, f);
                if (@as(u32, @bitCast(x)) == @as(u32, @bitCast(y)) or (x != x and y != y)) continue;
                if (bad < 4) print("    f{d} x={x} gpu={x} cpu={x}\n", .{ f, base + i, @as(u32, @bitCast(x)), @as(u32, @bitCast(y)) });
                bad += 1;
            }
        }
        print("  f{d}: {d} of 2^32 differ\n", .{ f, bad });
        check("math_sweep_f32", bad == 0);
    }
}

// ── Several devices: one handle, stream and pinned slice per device ──────

/// Shards one kernel over every device the backend has, launches on all of
/// them without waiting, then waits once per device. `GOMPUTE_FAKE_DEVICES=2`
/// makes a one-card machine run it with two contexts; one real card skips.
fn testDevices() void {
    print("\n[devices]\n", .{});
    if (comptime !g.available(.cuda)) return skip("this build emitted no CUDA artifacts");
    const count = g.runtime.dynamic.deviceCount(.cuda) catch return skip("no CUDA driver");
    if (count < 2) return skip("one device; set GOMPUTE_FAKE_DEVICES=2 to shard across two contexts");
    const K = g.RawKernel("raw_add_offset", .cuda);
    const Dev = struct { kernel: K, stream: K.Stream, buf: K.Buffer, host: []u8 };
    const n: usize = 1000;
    const per = n / 2;
    var devs: [2]Dev = undefined;
    var input: [n]f32 = undefined;
    for (&input, 0..) |*x, i| x.* = @floatFromInt(i);

    for (&devs, 0..) |*d, o| {
        d.kernel = K.init(@intCast(o)) catch return;
        d.stream = d.kernel.createStreamNonBlocking() catch return;
        d.buf = d.kernel.alloc(per * 4) catch return;
        d.host = d.kernel.allocPinned(per * 4) catch return;
        @memcpy(d.host, std.mem.sliceAsBytes(input[o * per ..][0..per]));
    }
    defer for (&devs) |*d| {
        d.kernel.freePinned(d.host);
        d.buf.free();
        d.stream.deinit();
        d.kernel.deinit();
    };
    // Enqueue everything on both devices, then wait: the two halves overlap.
    var len: u64 = per;
    var offset: f32 = 10;
    for (&devs) |*d| {
        d.buf.uploadAtAsync(d.host.ptr, 0, per * 4, &d.stream) catch return;
        d.kernel.launchOn(&d.stream, g.Dim3.linear(per, 256), .{ .x = 256 }, 0, &.{ d.buf.argPtr(), g.interface.arg(&len), g.interface.arg(&offset) }) catch return;
        d.buf.downloadAtAsync(d.host.ptr, 0, per * 4, &d.stream) catch return;
    }
    for (&devs) |*d| d.stream.synchronize() catch return;
    var ok = true;
    for (&devs, 0..) |*d, o| {
        const got: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, d.host));
        for (got, 0..) |v, i| ok = ok and v == input[o * per + i] + 10;
    }
    check("sharded_kernel_on_every_device", ok);

    // A copy between the two is a peer copy; the first device's buffer ends
    // up holding the second's half.
    devs[0].buf.copyFrom(&devs[1].buf, 0, 0, per * 4) catch return;
    devs[0].buf.download(devs[0].host.ptr, per * 4) catch return;
    const copied: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, devs[0].host));
    var copy_ok = true;
    for (copied, 0..) |v, i| copy_ok = copy_ok and v == input[per + i] + 10;
    check("peer_copy", copy_ok);

    // Direct access is a property of the pair of cards: granted, or
    // `error.Unsupported` (always, for two contexts on one card). Either is a
    // valid answer; anything else is a bug.
    const link = devs[0].kernel.context.enablePeerAccess(&devs[1].kernel.context);
    check("peer_access_answer", if (link) true else |e| e == error.Unsupported);
    print("  {d} devices, peer access: {s}\n", .{ count, if (link) "direct" else |_| "staged" });
}

/// The multi-root pipeline: `src/kernels2.zig` is a second, separately compiled
/// artifact. Proves the merged name map reaches both roots, that a name decided
/// at run time resolves into the right blob, and that an unknown one is an
/// error rather than a panic.
fn testSecondRoot() void {
    print("\n[second root]\n", .{});
    if (comptime !g.available(.cuda)) return skip("this build emitted no CUDA artifacts");
    const loaded = g.runtime.cuda.loadedModuleCount;
    const before = loaded();

    // Comptime-named, from the second root.
    var kernel = g.RawKernel("raw_add_offset", .cuda).init(0) catch |e| {
        print("  init failed: {s}\n", .{@errorName(e)});
        return;
    };
    defer kernel.deinit();
    var buf = kernel.alloc(3 * @sizeOf(f32)) catch return;
    defer buf.free();
    var data = [_]f32{ 1, 2, 3 };
    runOn(&kernel, &buf, @as([]f32, &data), @as(f32, 10)) catch |e| {
        print("  run failed: {s}\n", .{@errorName(e)});
        return;
    };
    check("second_root_result", allSame(f32, &data, &[_]f32{ 11, 12, 13 }));
    // One root loaded, not both: the other blob is never JIT'd.
    check("second_root_loaded_one_blob", loaded() == before + 1);

    // A name the compiler cannot see: chosen from a run-time value.
    var name_buf: [16]u8 = undefined;
    var name_buf2: [16]u8 = undefined;
    const chosen = std.fmt.bufPrint(&name_buf, "raw_{s}", .{if (data[0] > 0) "triple" else "nope"}) catch return;
    var by_name = g.rawKernelByName(.cuda, chosen, 0) catch |e| {
        print("  rawKernelByName failed: {s}\n", .{@errorName(e)});
        return;
    };
    defer by_name.deinit();
    check("runtime_named_kernel_resolved", true);

    var device_buf = by_name.alloc(3 * @sizeOf(f32)) catch return;
    defer device_buf.free();
    var host_data = [_]f32{ 1, 2, 3 };
    runOn(&by_name, &device_buf, @as([]f32, &host_data), {}) catch return;
    check("runtime_named_kernel_result", allSame(f32, &host_data, &[_]f32{ 3, 6, 9 }));

    // An unknown name is an error, not a panic.
    const bogus = std.fmt.bufPrint(&name_buf2, "raw_{s}", .{if (data[0] > 0) "nope" else "triple"}) catch return;
    if (g.rawKernelByName(.cuda, bogus, 0)) |_| {
        check("unknown_runtime_name_errors", false);
    } else |err| {
        check("unknown_runtime_name_errors", err == error.KernelNotFound);
        print("  \"{s}\" -> {s}\n", .{ bogus, @errorName(err) });
    }
    print("  raw_add_offset {any}, {s} {any}, blobs loaded {d}\n", .{ data, chosen, host_data, loaded() });
}

// ── Runtime dynamic: module load + buffer + kernel launch ───────────────

/// The generated index is built from the same kernel roots as the artifacts, so
/// a name that is missing while the artifacts exist means a renamed kernel or a
/// mismatched root -- a regression to report, not a `.?` panic. `docs/runtime.md`
/// shows the same shape as `orelse return error.KernelNotFound`.
fn notInIndex() void {
    print("  \"scale_relu\" is not in the generated kernel index\n", .{});
}

fn testRuntimeDynamic() void {
    print("\n[runtime.dynamic] ", .{});
    const rt = g.runtime.dynamic;

    var gpu = rt.Compute.init(null) catch |e| return skip(@errorName(e));
    defer gpu.deinit();
    print("backend={s}\n", .{@tagName(gpu.backend)});

    if (gpu.backend == .cpu) return skip("no GPU backend on this machine");

    // Load the compiled kernel module. One artifact per kernel root, so ask the
    // generated index which blob holds the kernel we are about to launch.
    const artifacts = @import("gompute_kernels");
    const image: [:0]const u8 = switch (gpu.backend) {
        .cuda => if (artifacts.has_cuda)
            artifacts.cuda_images[(artifacts.cuda_index.get("raw_scale") orelse return notInIndex()).blob]
        else
            return skip("this build emitted no CUDA artifacts"),
        .hip => if (artifacts.has_hip)
            artifacts.hip_images[(artifacts.hip_index.get("raw_scale") orelse return notInIndex()).blob]
        else
            return skip("this build emitted no HIP artifacts"),
        .cpu => unreachable,
    };

    var module = gpu.loadModule(image) catch |e| {
        const derr = g.lastDriverError();
        print("  loadModule failed: {s}, driver error={}\n", .{ @errorName(e), derr.code });
        return;
    };
    defer module.deinit();
    check("runtime_load_module", true);

    // Get kernel handle
    var kernel = module.getKernel("raw_scale") catch |e| {
        print("  getKernel failed: {s}\n", .{@errorName(e)});
        return;
    };
    check("runtime_get_kernel", true);

    // Alloc + upload
    const N: usize = 1024;
    const byte_size = N * @sizeOf(f32);
    var buf = gpu.alloc(byte_size) catch |e| {
        print("  alloc failed: {s}\n", .{@errorName(e)});
        return;
    };
    defer buf.free();
    check("runtime_alloc", true);

    var host_data: [N]f32 = undefined;
    for (&host_data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) - 512;
    buf.upload(@ptrCast(&host_data), byte_size) catch |e| {
        print("  upload failed: {s}\n", .{@errorName(e)});
        return;
    };
    check("runtime_upload", true);

    var len: u64 = N;
    var scale: f32 = 2;
    var args = [_]g.interface.Arg{
        buf.argPtr(),
        g.interface.arg(&len),
        g.interface.arg(&scale),
    };
    const grid = g.Dim3.linear(N, 256);
    kernel.launch(grid, .{ .x = 256 }, 0, &args) catch |e| {
        print("  launch failed: {s}\n", .{@errorName(e)});
        return;
    };
    gpu.synchronize() catch |e| {
        print("  sync failed: {s}\n", .{@errorName(e)});
        return;
    };
    check("runtime_launch", true);

    // Download + verify
    var result: [N]f32 = undefined;
    buf.download(@ptrCast(&result), byte_size) catch |e| {
        print("  download failed: {s}\n", .{@errorName(e)});
        return;
    };
    check("runtime_download", true);

    var ok = true;
    for (result, 0..) |v, i| {
        const orig: f32 = @as(f32, @floatFromInt(i)) - 512;
        const expected = orig * 2.0;
        if (!same(f32, v, expected)) {
            ok = false;
            break;
        }
    }
    check("runtime_result_correct", ok);

    // Buffer-to-buffer copy
    var buf2 = gpu.alloc(byte_size) catch return;
    defer buf2.free();
    buf2.copyFrom(&buf, 0, 0, byte_size) catch |e| {
        print("  copyFrom failed: {s}\n", .{@errorName(e)});
        return;
    };
    var copy_result: [N]f32 = undefined;
    buf2.download(@ptrCast(&copy_result), byte_size) catch return;
    check("runtime_copy", allSame(f32, &copy_result, &result));

    // Partial upload/download (uploadAt/downloadAt)
    var partial = [_]f32{ 99, 88, 77 };
    const offset = 10 * @sizeOf(f32);
    buf.uploadAt(@ptrCast(&partial), offset, 3 * @sizeOf(f32)) catch |e| {
        print("  uploadAt failed: {s}\n", .{@errorName(e)});
        return;
    };
    var readback: [3]f32 = undefined;
    buf.downloadAt(@ptrCast(&readback), offset, 3 * @sizeOf(f32)) catch |e| {
        print("  downloadAt failed: {s}\n", .{@errorName(e)});
        return;
    };
    check("runtime_partial_xfer", allSame(f32, &readback, &.{ 99, 88, 77 }));

    // Stream
    var stream = gpu.createStream() catch |e| {
        print("  createStream failed: {s}\n", .{@errorName(e)});
        return;
    };
    defer stream.deinit();
    stream.synchronize() catch |e| {
        print("  stream sync failed: {s}\n", .{@errorName(e)});
        return;
    };
    check("runtime_stream", true);

    // lastDriverError
    const derr = g.lastDriverError();
    check("driver_error_backend", derr.backend == .none or derr.backend == .cuda or derr.backend == .hip);

    print("  runtime tests done\n", .{});
}

// ── Raw kernel ──────────────────────────────────────────────────────────

/// `raw_increment` is exported by kernels.zig via `exportRaw` and launched by
/// hand: the caller owns the grid, the block, and the argument list.
///
/// `g.available(.cuda)` is the comptime question "did this build emit CUDA
/// artifacts?" -- `RawKernel(name, .cuda)` is a compile error when it did not,
/// so a probe that must compile on any machine has to ask first.
fn testRawKernel() void {
    print("\n[raw kernel] ", .{});
    if (comptime !g.available(.cuda)) return skip("this build emitted no CUDA artifacts");

    // Not a skip: the artifacts exist, so failing to load one is a regression.
    var raw = g.RawKernel("raw_increment", .cuda).init(0) catch |e| {
        print("init failed: {s}\n", .{@errorName(e)});
        return;
    };
    defer raw.deinit();
    var data = [_]f32{ 1, 2, 3, 4, 5 };
    var buffer = raw.alloc(@sizeOf(@TypeOf(data))) catch |e| {
        print("alloc failed: {s}\n", .{@errorName(e)});
        return;
    };
    defer buffer.free();
    runOn(&raw, &buffer, @as([]f32, &data), {}) catch |e| {
        print("run failed: {s}\n", .{@errorName(e)});
        return;
    };
    check("raw_increment", allSame(f32, &data, &.{ 2, 3, 4, 5, 6 }));
    print("{any}\n", .{data});
}

// ── Dim3 ────────────────────────────────────────────────────────────────

fn testDim3Linear() void {
    print("[Dim3.linear] ", .{});
    const d1 = g.Dim3.linear(1024, 256);
    check("dim3_exact", d1.x == 4 and d1.y == 1 and d1.z == 1);
    check("dim3_ceil", g.Dim3.linear(1025, 256).x == 5);
    check("dim3_one", g.Dim3.linear(1, 256).x == 1);
    check("dim3_zero", g.Dim3.linear(0, 256).x == 0);
    print("ok\n", .{});
}
