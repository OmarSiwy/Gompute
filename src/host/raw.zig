//! Thin loader for hand-written GPU kernels emitted by the same build pipeline.

const std = @import("std");
const iface = @import("../core/interface.zig");
const host = @import("kernel.zig");

pub fn RawKernel(comptime entry_name: [:0]const u8, comptime backend: host.Backend) type {
    return switch (backend) {
        .cuda => host.requireArtifacts(host.gpu_cuda, GpuRaw(entry_name, host.gpu_cuda)),
        .hip => host.requireArtifacts(host.gpu_hip, GpuRaw(entry_name, host.gpu_hip)),
        .cpu => @compileError("RawKernel is device-only; use a normal Zig function on CPU"),
    };
}

/// The handle `rawKernelByName` returns: a `RawKernel` whose kernel was picked
/// at run time, so it has no comptime `init` of its own.
pub fn RawByName(comptime backend: host.Backend) type {
    return switch (backend) {
        .cuda => host.requireArtifacts(host.gpu_cuda, GpuRaw(null, host.gpu_cuda)),
        .hip => host.requireArtifacts(host.gpu_hip, GpuRaw(null, host.gpu_hip)),
        .cpu => @compileError("kernel-by-name is device-only; use a normal Zig function on CPU"),
    };
}

/// Load the kernel called `name`, decided at run time -- from a parsed netlist,
/// a config file, a CLI flag. The set of kernels is closed at build time, so an
/// unknown name is `error.KernelNotFound` rather than a panic, and only the one
/// root that exports it is JIT'd.
pub fn rawKernelByName(
    comptime backend: host.Backend,
    name: []const u8,
    ordinal: c_int,
) iface.Error!RawByName(backend) {
    const gpu = switch (backend) {
        .cuda => host.gpu_cuda,
        .hip => host.gpu_hip,
        .cpu => comptime unreachable, // RawByName already rejected .cpu
    };
    const opened = try host.openModuleByName(gpu, name, ordinal);
    return .{ .context = opened.context, .module = opened.module, .kernel = opened.kernel };
}

/// `entry_name` is null for a handle whose kernel is chosen at run time.
fn GpuRaw(comptime entry_name: ?[:0]const u8, comptime gpu: host.Gpu) type {
    return struct {
        const Self = @This();
        pub const Buffer = gpu.rt.Buffer;
        pub const Stream = gpu.rt.Stream;

        context: gpu.rt.Context = .{},
        module: gpu.rt.Module = .{},
        kernel: gpu.rt.Kernel = .{},

        /// (#3) Handles on the same device share one primary context and one
        /// JIT'd copy of the artifact, so this is cheap after the first one and
        /// a `Buffer` from any handle is valid in all of them.
        pub fn init(ordinal: c_int) iface.Error!Self {
            const name = entry_name orelse
                @compileError("gompute: this handle came from rawKernelByName, which already " ++
                    "loaded its kernel; there is nothing for init() to name.");
            const opened = try host.openModule(gpu, name, ordinal);
            return .{ .context = opened.context, .module = opened.module, .kernel = opened.kernel };
        }

        /// (#3) Drops this handle only; the shared context and module stay up for
        /// the process. See `gompute.runtime.<backend>.shutdown()`.
        ///
        /// Resets rather than `undefined`, so a stray double-deinit stays a
        /// no-op instead of a driver-level double free in ReleaseFast.
        pub fn deinit(self: *Self) void {
            self.module.deinit();
            self.context.deinit();
            self.* = .{};
        }

        pub fn alloc(self: *Self, bytes: usize) iface.Error!Buffer {
            return self.context.alloc(bytes);
        }

        /// Page-locked host memory. Driver memory, not a Zig allocator's, so it
        /// is released with `freePinned`; see `Context.allocPinned` for why the
        /// async copies need it.
        pub fn allocPinned(self: *Self, bytes: usize) iface.Error![]u8 {
            return self.context.allocPinned(bytes);
        }

        pub fn freePinned(self: *Self, bytes: []u8) void {
            self.context.freePinned(bytes);
        }

        /// A stream to order copies and launches on. Anything issued here stays
        /// off the NULL stream, which implicitly synchronizes against every
        /// other blocking stream on the device.
        pub fn createStream(self: *Self) iface.Error!Stream {
            return self.context.createStream();
        }

        pub fn launch(
            self: *Self,
            grid: iface.Dim3,
            block: iface.Dim3,
            shared_bytes: u32,
            args: []const iface.Arg,
        ) iface.Error!void {
            // Worker threads start with no current context; see GpuKernel.launch.
            try self.context.makeCurrent();
            return self.kernel.launch(grid, block, shared_bytes, args);
        }

        /// `launch`, ordered on `stream` instead of the NULL stream. Returns as
        /// soon as the launch is queued, so `args` -- which are pointers to the
        /// caller's argument storage -- must outlive the call.
        pub fn launchOn(
            self: *Self,
            stream: *Stream,
            grid: iface.Dim3,
            block: iface.Dim3,
            shared_bytes: u32,
            args: []const iface.Arg,
        ) iface.Error!void {
            // Same reason as `launch`: a worker thread has no current context.
            try self.context.makeCurrent();
            return self.kernel.launchOnStream(grid, block, shared_bytes, args, stream.stream);
        }

        pub fn synchronize(self: *Self) iface.Error!void {
            return self.context.synchronize();
        }
    };
}

// ---------------------------------------------------------------------------
// Seams. None of this needs a GPU, and none of it can go through `RawKernel`:
// the library's own test build emits no artifacts, so `requireArtifacts` makes
// every public entry point here a compile error. `GpuRaw` is reachable, and
// naming it is most of the point -- these tests are what pulls this file's
// bodies through semantic analysis at all (see the `test` at the end of
// kernel.zig for why that needs help).
// ---------------------------------------------------------------------------

test GpuRaw {
    inline for (.{ host.gpu_cuda, host.gpu_hip }) |gpu| {
        const Raw = GpuRaw("gompute_seam_probe", gpu);
        std.testing.refAllDecls(Raw);
        try std.testing.expect(Raw.Buffer == gpu.rt.Buffer);
        try std.testing.expect(Raw.Stream == gpu.rt.Stream);

        // A reset handle's methods are driver-level no-ops; see `deinit`.
        var raw: Raw = .{};
        var empty: [0]u8 = .{};
        raw.freePinned(&empty);
        raw.deinit();
    }
}

test "the async surface still lines up with the blocking one" {
    // `Stream`, `createStream`, `allocPinned`/`freePinned` and `launchOn` all
    // landed in one commit and nothing calls them yet. Until something does,
    // this is the oracle: the pairs have to keep matching, or whoever finishes
    // the overlap path inherits three mismatches at once.
    inline for (.{ host.gpu_cuda, host.gpu_hip }) |gpu| {
        const Raw = GpuRaw("gompute_seam_probe", gpu);

        const stream = @typeInfo(@TypeOf(Raw.createStream)).@"fn".return_type.?;
        try std.testing.expect(@typeInfo(stream).error_union.payload == Raw.Stream);

        // Driver memory: `freePinned` is the only legal way to release it, so it
        // has to accept exactly what `allocPinned` hands back.
        const pinned = @typeInfo(@TypeOf(Raw.allocPinned)).@"fn".return_type.?;
        try std.testing.expect(@typeInfo(pinned).error_union.payload ==
            @typeInfo(@TypeOf(Raw.freePinned)).@"fn".params[1].type.?);

        // `launchOn` is `launch` with a stream spliced in after `self`.
        const blocking = @typeInfo(@TypeOf(Raw.launch)).@"fn".params;
        const ordered = @typeInfo(@TypeOf(Raw.launchOn)).@"fn".params;
        try std.testing.expectEqual(blocking.len + 1, ordered.len);
        try std.testing.expect(ordered[1].type.? == *Raw.Stream);
        inline for (blocking[1..], ordered[2..]) |a, b|
            try std.testing.expect(a.type.? == b.type.?);
    }
}
