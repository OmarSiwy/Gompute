//! Thin loader for hand-written GPU kernels emitted by the same build pipeline.

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
