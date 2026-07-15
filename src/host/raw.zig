//! Thin loader for hand-written GPU kernels emitted by the same build pipeline.

const iface = @import("../core/interface.zig");
const host = @import("kernel.zig");
const cuda = @import("../runtime/cuda.zig");
const hip = @import("../runtime/hip.zig");

pub fn RawKernel(comptime entry_name: [:0]const u8, comptime backend: host.Backend) type {
    return switch (backend) {
        .cuda => CudaRaw(entry_name),
        .hip => HipRaw(entry_name),
        .cpu => @compileError("RawKernel is device-only; use a normal Zig function on CPU"),
    };
}

fn CudaRaw(comptime entry_name: [:0]const u8) type {
    return struct {
        const Self = @This();
        pub const Buffer = cuda.Buffer;

        context: cuda.Context,
        module: cuda.Module,
        kernel: cuda.Kernel,

        pub fn init(ordinal: c_int) iface.Error!Self {
            const artifacts = @import("gompute_kernels");
            if (comptime !artifacts.has_cuda) return error.BackendUnavailable;
            var context = try cuda.Context.init(ordinal);
            errdefer context.deinit();
            var module = try context.loadModuleFromMemory(artifacts.cuda);
            errdefer module.deinit();
            return .{
                .context = context,
                .module = module,
                .kernel = try module.getKernel(entry_name.ptr),
            };
        }

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
            return self.kernel.launch(grid, block, shared_bytes, args);
        }

        pub fn synchronize(self: *Self) iface.Error!void {
            return self.context.synchronize();
        }
    };
}

fn HipRaw(comptime entry_name: [:0]const u8) type {
    return struct {
        const Self = @This();
        pub const Buffer = hip.Buffer;

        context: hip.Context,
        module: hip.Module,
        kernel: hip.Kernel,

        pub fn init(ordinal: c_int) iface.Error!Self {
            const artifacts = @import("gompute_kernels");
            if (comptime !artifacts.has_hip) return error.BackendUnavailable;
            var context = try hip.Context.init(ordinal);
            errdefer context.deinit();
            var module = try context.loadModuleFromMemory(artifacts.hip);
            errdefer module.deinit();
            const internal_name = artifacts.hip_names.resolve(entry_name);
            return .{
                .context = context,
                .module = module,
                .kernel = try module.getKernel(internal_name.ptr),
            };
        }

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
            return self.kernel.launch(grid, block, shared_bytes, args);
        }

        pub fn synchronize(self: *Self) iface.Error!void {
            return self.context.synchronize();
        }
    };
}
