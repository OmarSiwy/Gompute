const std = @import("std");
const g = @import("gompute");
const k = @import("kernels");
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

    // ── AutoKernel tests (CUDA → HIP → CPU) ────────────────────────────
    // Spec, label, input, params, expected. The axes worth covering are the
    // element type (f32/f64/u32/i16), the param shape (scalar/multi/array/
    // nested), fusion depth, and the length edges 0 and 1.
    autoCase(k.scale_relu, "scale_relu", &.{ -3, -1, 0, 1, 3 }, .{ .scale = 2 }, &.{ 0, 0, 0, 2, 6 });
    autoCase(k.affine_transform, "affine", &.{ 0, 1, -1, 10 }, .{ .scale = 3, .bias = 1, .enabled = true }, &.{ 1, 4, -2, 31 });
    autoCase(k.clamp, "clamp", &.{ -5, 0, 0.5, 1, 2 }, .{ .scale = 0 }, &.{ 0, 0, 0.5, 1, 1 });
    // -0.0, not 0: negating a zero keeps the sign, and the printed output says so.
    autoCase(k.neg, "negate", &.{ -3, 0, 7 }, .{ .scale = 0 }, &.{ 3, -0.0, -7 });
    autoCase(k.sq, "square", &.{ -2, 0, 3, 0.5 }, .{ .scale = 0 }, &.{ 4, 0, 9, 0.25 });
    autoCase(k.weighted, "weighted", &.{ 5, 0 }, .{ .weights = .{ 2, 3, 4, 1 } }, &.{ 23, 13 });
    autoCase(k.nested, "nested", &.{ 2, -1 }, .{ .inner = .{ .scale = 3 }, .offset = 10 }, &.{ 16, 7 });
    autoCase(k.scale_square, "fused_scale_sq", &.{ 3, -2 }, .{ .scale = 2 }, &.{ 36, 16 });
    autoCase(k.triple_fused, "triple_fused", &.{ -3, 2 }, .{ .scale = -2 }, &.{ 36, 16 });
    autoCase(k.abs_then_scale, "abs_then_scale", &.{ -5, 3 }, .{ .scale = 10 }, &.{ 50, 30 });
    autoCase(k.double_f64, "f64", &.{ 1.5, -2.25, 0 }, .{ .scale = 0 }, &.{ 3, -4.5, 0 });
    autoCase(k.shift_mask, "u32_shift_mask", &.{ 0xFF00, 0xABCD, 0 }, .{ .shift = 8, .mask = 0xFF }, &.{ 0xFF, 0xAB, 0 });
    autoCase(k.saturate_i16, "i16_saturate", &.{ -200, -50, 0, 50, 200 }, .{ .scale = 0 }, &.{ -100, -50, 0, 50, 100 });
    autoCase(k.scale_relu, "empty", &.{}, .{ .scale = 2 }, &.{});
    autoCase(k.sq, "single", &.{42}, .{ .scale = 0 }, &.{1764});
    testAutoLarge();

    // ── Runtime dynamic backend ─────────────────────────────────────────
    section("runtime.dynamic", 11, testRuntimeDynamic);

    // ── Second kernel root + run-time kernel selection ──────────────────
    section("second root", 5, testSecondRoot);

    // ── Comptime CPU reference (sanity baseline) ────────────────────────
    testCpuReference();

    // ── Hand-written raw kernel through the same artifact pipeline ──────
    section("raw kernel", 1, testRawKernel);

    // ── Pure comptime: ABI, fusion, Dim3 ────────────────────────────────
    testAbiPack();
    testFusionDirect();
    testDim3Linear();

    print("\n=== Results: {} passed, {} failed ===\n", .{ pass_count, fail_count });
    if (fail_count > 0) std.process.exit(1);
}

// ── AutoKernel: runs on GPU when available ──────────────────────────────

/// Every auto test is the same seven lines: pick a backend, run the spec over a
/// literal input, compare against a literal expectation. Only `testAutoLarge`
/// differs, because it generates its input rather than listing it.
///
/// `input` is copied into scratch storage so the caller can pass a literal.
/// Asserts `input.len <= 8` -- these are shape probes, not workloads.
fn autoCase(
    comptime Spec: type,
    label: []const u8,
    input: []const Spec.Value,
    params: Spec.Parameters,
    expected: []const Spec.Value,
) void {
    std.debug.assert(input.len <= 8);
    print("[auto {s}] ", .{label});
    var kern = g.AutoKernel(Spec).init();
    defer kern.deinit();
    print("({s}) ", .{@tagName(kern.selected())});

    var storage: [8]Spec.Value = undefined;
    const data = storage[0..input.len];
    @memcpy(data, input);
    kern.run(data, params) catch unreachable;

    check(label, allSame(Spec.Value, data, expected));
    print("{any}\n", .{data});
}

fn testAutoLarge() void {
    print("[auto large 10k] ", .{});
    var kern = g.AutoKernel(k.scale_relu).init();
    defer kern.deinit();
    print("({s}) ", .{@tagName(kern.selected())});
    var buf: [10000]f32 = undefined;
    for (&buf, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) - 5000;
    kern.run(&buf, .{ .scale = 1 }) catch unreachable;
    var ok = true;
    for (buf, 0..) |v, i| {
        const orig: f32 = @as(f32, @floatFromInt(i)) - 5000;
        const expected: f32 = if (orig > 0) orig else 0;
        if (!same(f32, v, expected)) {
            ok = false;
            break;
        }
    }
    check("large_10k", ok);
    print("ok\n", .{});
}

// ── Runtime dynamic: module load + buffer + kernel launch ───────────────

/// The multi-root pipeline: `src/kernels2.zig` is a second, separately compiled
/// artifact. Proves the merged name map reaches both roots, that a name decided
/// at run time resolves into the right blob, and that an unknown one is an
/// error rather than a panic.
fn testSecondRoot() void {
    print("\n[second root]\n", .{});
    if (!g.AutoKernel(k2.add_offset).Cuda.available) return skip("this build emitted no CUDA artifacts");
    const loaded = g.runtime.cuda.loadedModuleCount;
    const before = loaded();

    // Comptime-named, from the second root.
    var kernel = g.Kernel(k2.add_offset, .cuda).init(0) catch |e| {
        print("  init failed: {s}\n", .{@errorName(e)});
        return;
    };
    defer kernel.deinit();
    var data = [_]f32{ 1, 2, 3 };
    kernel.run(&data, .{ .offset = 10 }) catch |e| {
        print("  run failed: {s}\n", .{@errorName(e)});
        return;
    };
    check("second_root_result", allSame(f32, &data, &[_]f32{ 11, 12, 13 }));
    // One root loaded, not both: the other blob is never JIT'd.
    check("second_root_loaded_one_blob", loaded() == before + 1);

    // A name the compiler cannot see: chosen from a run-time value.
    var buf: [16]u8 = undefined;
    var buf2: [16]u8 = undefined;
    const chosen = std.fmt.bufPrint(&buf, "raw_{s}", .{if (data[0] > 0) "triple" else "nope"}) catch return;
    var by_name = g.rawKernelByName(.cuda, chosen, 0) catch |e| {
        print("  rawKernelByName failed: {s}\n", .{@errorName(e)});
        return;
    };
    defer by_name.deinit();
    check("runtime_named_kernel_resolved", true);

    var device_buf = by_name.alloc(3 * @sizeOf(f32)) catch return;
    defer device_buf.free();
    var host_data = [_]f32{ 1, 2, 3 };
    device_buf.upload(@ptrCast(&host_data), @sizeOf(@TypeOf(host_data))) catch return;
    var len: u64 = 3;
    by_name.launch(.{ .x = 1 }, .{ .x = 256 }, 0, &.{ device_buf.argPtr(), g.interface.arg(&len) }) catch return;
    by_name.synchronize() catch return;
    device_buf.download(@ptrCast(&host_data), @sizeOf(@TypeOf(host_data))) catch return;
    check("runtime_named_kernel_result", allSame(f32, &host_data, &[_]f32{ 3, 6, 9 }));

    // An unknown name is an error, not a panic.
    const bogus = std.fmt.bufPrint(&buf2, "raw_{s}", .{if (data[0] > 0) "nope" else "triple"}) catch return;
    if (g.rawKernelByName(.cuda, bogus, 0)) |_| {
        check("unknown_runtime_name_errors", false);
    } else |err| {
        check("unknown_runtime_name_errors", err == error.KernelNotFound);
        print("  \"{s}\" -> {s}\n", .{ bogus, @errorName(err) });
    }
    print("  add_offset {any}, {s} {any}, blobs loaded {d}\n", .{
        data,
        chosen,
        host_data,
        loaded(),
    });
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
        .cuda => if (artifacts.has_cuda) artifacts.cuda_images[artifacts.cuda_index.get("scale_relu").?.blob] else return skip("this build emitted no CUDA artifacts"),
        .hip => if (artifacts.has_hip) artifacts.hip_images[artifacts.hip_index.get("scale_relu").?.blob] else return skip("this build emitted no HIP artifacts"),
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
    var kernel = module.getKernel("scale_relu") catch |e| {
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
    var packed_params = g.abi.pack(k.ScalarParams, .{ .scale = @as(f32, 2.0) });
    var args = [_]g.interface.Arg{
        buf.argPtr(),
        g.interface.arg(&len),
        g.interface.arg(&packed_params),
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
        const y = orig * 2.0;
        const expected: f32 = if (y > 0) y else 0;
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

// ── CPU reference baseline ──────────────────────────────────────────────

fn testCpuReference() void {
    print("\n[cpu reference] ", .{});
    var data = [_]f32{ -3, -1, 0, 1, 3 };
    var kern = g.Kernel(k.scale_relu, .cpu).init(0) catch unreachable;
    kern.run(&data, .{ .scale = 2 }) catch unreachable;
    check("cpu_ref", allSame(f32, &data, &.{ 0, 0, 0, 2, 6 }));
    print("ok\n", .{});
}

// ── ABI pack/unpack ─────────────────────────────────────────────────────

/// `raw_increment` is exported by kernels.zig via `exportRaw` and rides the same
/// PTX/HSACO pipeline as the generated kernels, but is launched by hand: the
/// caller owns the grid, the block, and the argument list.
///
/// `AutoKernel(...).Cuda.available` is the comptime question "did this build
/// emit CUDA artifacts?" -- `RawKernel(name, .cuda)` is a compile error when it
/// did not, so a probe that must compile on any machine has to ask first.
fn testRawKernel() void {
    print("[raw kernel] ", .{});
    if (comptime !g.AutoKernel(k.scale_relu).Cuda.available) return skip("this build emitted no CUDA artifacts");

    // Not a skip: the artifacts exist, so failing to load one is a regression.
    var raw = g.RawKernel("raw_increment", .cuda).init(0) catch |e| {
        print("init failed: {s}\n", .{@errorName(e)});
        return;
    };
    defer raw.deinit();

    var data = [_]f32{ 1, 2, 3, 4, 5 };
    const bytes = data.len * @sizeOf(f32);
    const block: u32 = 256;

    var buffer = raw.alloc(bytes) catch |e| {
        print("alloc failed: {s}\n", .{@errorName(e)});
        return;
    };
    defer buffer.free();

    var len: u64 = data.len;
    var args = [_]g.interface.Arg{ buffer.argPtr(), g.interface.arg(&len) };

    buffer.upload(&data, bytes) catch |e| {
        print("upload failed: {s}\n", .{@errorName(e)});
        return;
    };
    raw.launch(g.Dim3.linear(data.len, block), .{ .x = block }, 0, &args) catch |e| {
        print("launch failed: {s}\n", .{@errorName(e)});
        return;
    };
    raw.synchronize() catch |e| {
        print("sync failed: {s}\n", .{@errorName(e)});
        return;
    };
    buffer.download(&data, bytes) catch |e| {
        print("download failed: {s}\n", .{@errorName(e)});
        return;
    };

    check("raw_increment", allSame(f32, &data, &.{ 2, 3, 4, 5, 6 }));
    print("{any}\n", .{data});
}

fn testAbiPack() void {
    print("[abi pack/unpack] ", .{});
    const abi = g.abi;

    const packed_scalar = abi.pack(k.ScalarParams, .{ .scale = 3.14 });
    const unpacked_scalar = abi.unpack(k.ScalarParams, packed_scalar);
    check("abi_scalar", same(f32, unpacked_scalar.scale, 3.14));

    const packed_multi = abi.pack(k.MultiParams, .{ .scale = 2, .bias = -1, .enabled = true });
    const unpacked_multi = abi.unpack(k.MultiParams, packed_multi);
    check("abi_multi_scale", same(f32, unpacked_multi.scale, 2));
    check("abi_multi_bias", same(f32, unpacked_multi.bias, -1));
    check("abi_multi_bool", unpacked_multi.enabled == true);

    const packed_false = abi.pack(k.MultiParams, .{ .scale = 0, .bias = 0, .enabled = false });
    const unpacked_false = abi.unpack(k.MultiParams, packed_false);
    check("abi_bool_false", unpacked_false.enabled == false);

    const packed_arr = abi.pack(k.ArrayParams, .{ .weights = .{ 1, 2, 3, 4 } });
    const unpacked_arr = abi.unpack(k.ArrayParams, packed_arr);
    check("abi_array", unpacked_arr.weights[0] == 1 and unpacked_arr.weights[3] == 4);

    const packed_nested = abi.pack(k.NestedParams, .{ .inner = .{ .scale = 7 }, .offset = -3 });
    const unpacked_nested = abi.unpack(k.NestedParams, packed_nested);
    check("abi_nested", same(f32, unpacked_nested.inner.scale, 7) and same(f32, unpacked_nested.offset, -3));

    const B = abi.Boundary(k.MultiParams);
    check("abi_extern_layout", @typeInfo(B).@"struct".layout == .@"extern");

    print("ok\n", .{});
}

// ── Fusion direct eval ──────────────────────────────────────────────────

fn testFusionDirect() void {
    print("[fusion direct] ", .{});
    check("fused_scale_sq", same(f32, k.ScaleSquare.eval(3, .{ .scale = 2 }), 36));
    check("fused_abs_relu", same(f32, k.AbsRelu.eval(-5, .{ .scale = 0 }), 5));
    check("fused_triple", same(f32, k.ScaleAbsSquare.eval(-3, .{ .scale = -2 }), 36));
    check("fused_abs_scale", same(f32, k.AbsThenScale.eval(-4, .{ .scale = 3 }), 12));
    print("ok\n", .{});
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
