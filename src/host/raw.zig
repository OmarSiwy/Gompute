//! Thin loader for hand-written GPU kernels emitted by the same build pipeline.

const iface = @import("../core/interface.zig");
const host = @import("kernel.zig");

pub fn RawKernel(comptime entry_name: [:0]const u8, comptime backend: host.Backend) type {
    return switch (backend) {
        .cuda => GpuRaw(entry_name, host.gpu_cuda),
        .hip => GpuRaw(entry_name, host.gpu_hip),
        .cpu => @compileError("RawKernel is device-only; use a normal Zig function on CPU"),
    };
}

fn GpuRaw(comptime entry_name: [:0]const u8, comptime gpu: host.Gpu) type {
    return struct {
        const Self = @This();
        pub const Buffer = gpu.rt.Buffer;

        context: gpu.rt.Context,
        module: gpu.rt.Module,
        kernel: gpu.rt.Kernel,

        /// (#3) Handles on the same device share one primary context and one
        /// JIT'd copy of the artifact, so this is cheap after the first one and
        /// a `Buffer` from any handle is valid in all of them.
        pub fn init(ordinal: c_int) iface.Error!Self {
            const opened = try host.openModule(gpu, entry_name, ordinal);
            return .{ .context = opened.context, .module = opened.module, .kernel = opened.kernel };
        }

        /// (#3) Drops this handle only; the shared context and module stay up for
        /// the process. See `gompute.runtime.<backend>.shutdown()`.
        pub fn deinit(self: *Self) void {
            self.module.deinit();
            self.context.deinit();
            self.* = undefined;
        }

        pub fn alloc(self: *Self, bytes: usize) iface.Error!Buffer {
            return self.context.alloc(bytes);
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

        pub fn synchronize(self: *Self) iface.Error!void {
            return self.context.synchronize();
        }
    };
}
