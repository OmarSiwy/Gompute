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

fn approxEq(a: f32, b: f32) bool {
    return @abs(a - b) < 1e-4;
}

fn allApproxEq(actual: []const f32, expected: []const f32) bool {
    if (actual.len != expected.len) return false;
    for (actual, expected) |a, e| if (!approxEq(a, e)) return false;
    return true;
}

pub fn main() !void {
    print("=== Gompute exhaustive GPU test ===\n\n", .{});

    // ── AutoKernel tests (CUDA → HIP → CPU) ────────────────────────────
    testAutoScaleRelu();
    testAutoAffine();
    testAutoClamp();
    testAutoNegate();
    testAutoSquare();
    testAutoWeighted();
    testAutoNested();
    testAutoFusedScaleSquare();
    testAutoTripleFused();
    testAutoAbsThenScale();
    testAutoF64();
    testAutoU32();
    testAutoI16();
    testAutoEmpty();
    testAutoSingle();
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

fn testAutoScaleRelu() void {
    print("[auto scale_relu] ", .{});
    var kern = g.AutoKernel(k.scale_relu).init();
    defer kern.deinit();
    print("({s}) ", .{@tagName(kern.selected())});
    var data = [_]f32{ -3, -1, 0, 1, 3 };
    kern.run(&data, .{ .scale = 2 }) catch unreachable;
    check("scale_relu", allApproxEq(&data, &.{ 0, 0, 0, 2, 6 }));
    print("{any}\n", .{data});
}

fn testAutoAffine() void {
    print("[auto affine] ", .{});
    var kern = g.AutoKernel(k.affine_transform).init();
    defer kern.deinit();
    print("({s}) ", .{@tagName(kern.selected())});
    var data = [_]f32{ 0, 1, -1, 10 };
    kern.run(&data, .{ .scale = 3, .bias = 1, .enabled = true }) catch unreachable;
    check("affine", allApproxEq(&data, &.{ 1, 4, -2, 31 }));
    print("{any}\n", .{data});
}

fn testAutoClamp() void {
    print("[auto clamp] ", .{});
    var kern = g.AutoKernel(k.clamp).init();
    defer kern.deinit();
    print("({s}) ", .{@tagName(kern.selected())});
    var data = [_]f32{ -5, 0, 0.5, 1, 2 };
    kern.run(&data, .{ .scale = 0 }) catch unreachable;
    check("clamp", allApproxEq(&data, &.{ 0, 0, 0.5, 1, 1 }));
    print("{any}\n", .{data});
}

fn testAutoNegate() void {
    print("[auto negate] ", .{});
    var kern = g.AutoKernel(k.neg).init();
    defer kern.deinit();
    print("({s}) ", .{@tagName(kern.selected())});
    var data = [_]f32{ -3, 0, 7 };
    kern.run(&data, .{ .scale = 0 }) catch unreachable;
    check("negate", allApproxEq(&data, &.{ 3, 0, -7 }));
    print("{any}\n", .{data});
}

fn testAutoSquare() void {
    print("[auto square] ", .{});
    var kern = g.AutoKernel(k.sq).init();
    defer kern.deinit();
    print("({s}) ", .{@tagName(kern.selected())});
    var data = [_]f32{ -2, 0, 3, 0.5 };
    kern.run(&data, .{ .scale = 0 }) catch unreachable;
    check("square", allApproxEq(&data, &.{ 4, 0, 9, 0.25 }));
    print("{any}\n", .{data});
}

fn testAutoWeighted() void {
    print("[auto weighted] ", .{});
    var kern = g.AutoKernel(k.weighted).init();
    defer kern.deinit();
    print("({s}) ", .{@tagName(kern.selected())});
    var data = [_]f32{ 5, 0 };
    kern.run(&data, .{ .weights = .{ 2, 3, 4, 1 } }) catch unreachable;
    check("weighted[0]", approxEq(data[0], 23));
    check("weighted[1]", approxEq(data[1], 13));
    print("{any}\n", .{data});
}

fn testAutoNested() void {
    print("[auto nested] ", .{});
    var kern = g.AutoKernel(k.nested).init();
    defer kern.deinit();
    print("({s}) ", .{@tagName(kern.selected())});
    var data = [_]f32{ 2, -1 };
    kern.run(&data, .{ .inner = .{ .scale = 3 }, .offset = 10 }) catch unreachable;
    check("nested", allApproxEq(&data, &.{ 16, 7 }));
    print("{any}\n", .{data});
}

fn testAutoFusedScaleSquare() void {
    print("[auto fused scale*sq] ", .{});
    var kern = g.AutoKernel(k.scale_square).init();
    defer kern.deinit();
    print("({s}) ", .{@tagName(kern.selected())});
    var data = [_]f32{ 3, -2 };
    kern.run(&data, .{ .scale = 2 }) catch unreachable;
    check("fused_scale_sq", allApproxEq(&data, &.{ 36, 16 }));
    print("{any}\n", .{data});
}

fn testAutoTripleFused() void {
    print("[auto triple fused] ", .{});
    var kern = g.AutoKernel(k.triple_fused).init();
    defer kern.deinit();
    print("({s}) ", .{@tagName(kern.selected())});
    var data = [_]f32{ -3, 2 };
    kern.run(&data, .{ .scale = -2 }) catch unreachable;
    check("triple_fused", allApproxEq(&data, &.{ 36, 16 }));
    print("{any}\n", .{data});
}

fn testAutoAbsThenScale() void {
    print("[auto abs->scale] ", .{});
    var kern = g.AutoKernel(k.abs_then_scale).init();
    defer kern.deinit();
    print("({s}) ", .{@tagName(kern.selected())});
    var data = [_]f32{ -5, 3 };
    kern.run(&data, .{ .scale = 10 }) catch unreachable;
    check("abs_then_scale", allApproxEq(&data, &.{ 50, 30 }));
    print("{any}\n", .{data});
}

fn testAutoF64() void {
    print("[auto f64] ", .{});
    var kern = g.AutoKernel(k.double_f64).init();
    defer kern.deinit();
    print("({s}) ", .{@tagName(kern.selected())});
    var data = [_]f64{ 1.5, -2.25, 0 };
    kern.run(&data, .{ .scale = 0 }) catch unreachable;
    check("f64[0]", @abs(data[0] - 3.0) < 1e-10);
    check("f64[1]", @abs(data[1] - -4.5) < 1e-10);
    check("f64[2]", data[2] == 0);
    print("{any}\n", .{data});
}

fn testAutoU32() void {
    print("[auto u32 shift_mask] ", .{});
    var kern = g.AutoKernel(k.shift_mask).init();
    defer kern.deinit();
    print("({s}) ", .{@tagName(kern.selected())});
    var data = [_]u32{ 0xFF00, 0xABCD, 0 };
    kern.run(&data, .{ .shift = 8, .mask = 0xFF }) catch unreachable;
    check("u32[0]", data[0] == 0xFF);
    check("u32[1]", data[1] == 0xAB);
    check("u32[2]", data[2] == 0);
    print("{any}\n", .{data});
}

fn testAutoI16() void {
    print("[auto i16 saturate] ", .{});
    var kern = g.AutoKernel(k.saturate_i16).init();
    defer kern.deinit();
    print("({s}) ", .{@tagName(kern.selected())});
    var data = [_]i16{ -200, -50, 0, 50, 200 };
    kern.run(&data, .{ .scale = 0 }) catch unreachable;
    check("i16_sat", data[0] == -100 and data[1] == -50 and data[2] == 0 and data[3] == 50 and data[4] == 100);
    print("{any}\n", .{data});
}

fn testAutoEmpty() void {
    print("[auto empty] ", .{});
    var kern = g.AutoKernel(k.scale_relu).init();
    defer kern.deinit();
    print("({s}) ", .{@tagName(kern.selected())});
    var data = [_]f32{};
    kern.run(&data, .{ .scale = 2 }) catch unreachable;
    check("empty", data.len == 0);
    print("ok\n", .{});
}

fn testAutoSingle() void {
    print("[auto single] ", .{});
    var kern = g.AutoKernel(k.sq).init();
    defer kern.deinit();
    print("({s}) ", .{@tagName(kern.selected())});
    var data = [_]f32{42};
    kern.run(&data, .{ .scale = 0 }) catch unreachable;
    check("single", approxEq(data[0], 1764));
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
        if (!approxEq(v, expected)) {
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
    check("second_root_result", allApproxEq(&data, &[_]f32{ 11, 12, 13 }));
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
    check("runtime_named_kernel_result", allApproxEq(&host_data, &[_]f32{ 3, 6, 9 }));

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
        if (!approxEq(v, expected)) {
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
    check("runtime_copy", allApproxEq(&copy_result, &result));

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
    check("runtime_partial_xfer", allApproxEq(&readback, &.{ 99, 88, 77 }));

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
    check("cpu_ref", allApproxEq(&data, &.{ 0, 0, 0, 2, 6 }));
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

    check("raw_increment", allApproxEq(&data, &.{ 2, 3, 4, 5, 6 }));
    print("{any}\n", .{data});
}

fn testAbiPack() void {
    print("[abi pack/unpack] ", .{});
    const abi = g.abi;

    const packed_scalar = abi.pack(k.ScalarParams, .{ .scale = 3.14 });
    const unpacked_scalar = abi.unpack(k.ScalarParams, packed_scalar);
    check("abi_scalar", approxEq(unpacked_scalar.scale, 3.14));

    const packed_multi = abi.pack(k.MultiParams, .{ .scale = 2, .bias = -1, .enabled = true });
    const unpacked_multi = abi.unpack(k.MultiParams, packed_multi);
    check("abi_multi_scale", approxEq(unpacked_multi.scale, 2));
    check("abi_multi_bias", approxEq(unpacked_multi.bias, -1));
    check("abi_multi_bool", unpacked_multi.enabled == true);

    const packed_false = abi.pack(k.MultiParams, .{ .scale = 0, .bias = 0, .enabled = false });
    const unpacked_false = abi.unpack(k.MultiParams, packed_false);
    check("abi_bool_false", unpacked_false.enabled == false);

    const packed_arr = abi.pack(k.ArrayParams, .{ .weights = .{ 1, 2, 3, 4 } });
    const unpacked_arr = abi.unpack(k.ArrayParams, packed_arr);
    check("abi_array", unpacked_arr.weights[0] == 1 and unpacked_arr.weights[3] == 4);

    const packed_nested = abi.pack(k.NestedParams, .{ .inner = .{ .scale = 7 }, .offset = -3 });
    const unpacked_nested = abi.unpack(k.NestedParams, packed_nested);
    check("abi_nested", approxEq(unpacked_nested.inner.scale, 7) and approxEq(unpacked_nested.offset, -3));

    const B = abi.Boundary(k.MultiParams);
    check("abi_extern_layout", @typeInfo(B).@"struct".layout == .@"extern");

    print("ok\n", .{});
}

// ── Fusion direct eval ──────────────────────────────────────────────────

fn testFusionDirect() void {
    print("[fusion direct] ", .{});
    check("fused_scale_sq", approxEq(k.ScaleSquare.eval(3, .{ .scale = 2 }), 36));
    check("fused_abs_relu", approxEq(k.AbsRelu.eval(-5, .{ .scale = 0 }), 5));
    check("fused_triple", approxEq(k.ScaleAbsSquare.eval(-3, .{ .scale = -2 }), 36));
    check("fused_abs_scale", approxEq(k.AbsThenScale.eval(-4, .{ .scale = 3 }), 12));
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
