//! Compile-time-specialized host handles.

const std = @import("std");
const abi = @import("../core/abi.zig");
const iface = @import("../core/interface.zig");
const cuda = @import("../runtime/cuda.zig");
const hip = @import("../runtime/hip.zig");

pub const Backend = enum { cpu, cuda, hip };

pub fn Kernel(comptime Spec: type, comptime backend: Backend) type {
    return switch (backend) {
        .cpu => CpuKernel(Spec),
        .cuda => CudaKernel(Spec),
        .hip => HipKernel(Spec),
    };
}

fn CpuKernel(comptime Spec: type) type {
    return struct {
        const Self = @This();
        pub const backend: Backend = .cpu;
        pub const Buffer = void;

        pub inline fn init(_: c_int) iface.Error!Self {
            return .{};
        }

        pub inline fn deinit(_: *Self) void {}

        pub inline fn run(_: *Self, data: []Spec.Value, params: Spec.Parameters) iface.Error!void {
            for (data) |*value| value.* = Spec.eval(value.*, params);
        }
    };
}

fn CudaKernel(comptime Spec: type) type {
    return struct {
        const Self = @This();
        pub const backend: Backend = .cuda;
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
            const kernel = try module.getKernel(Spec.entry_name.ptr);
            return .{ .context = context, .module = module, .kernel = kernel };
        }

        pub fn deinit(self: *Self) void {
            self.module.deinit();
            self.context.deinit();
            self.* = undefined;
        }

        pub fn alloc(self: *Self, count: usize) iface.Error!Buffer {
            return self.context.alloc(count * @sizeOf(Spec.Value));
        }

        pub fn launch(
            self: *Self,
            buffer: *Buffer,
            count: usize,
            params: Spec.Parameters,
        ) iface.Error!void {
            if (count == 0) return;
            var len: u64 = @intCast(count);
            var packed_params = abi.pack(Spec.Parameters, params);
            var args = [_]iface.Arg{
                buffer.argPtr(),
                iface.arg(&len),
                iface.arg(&packed_params),
            };
            try self.kernel.launch(
                iface.Dim3.linear(count, Spec.block_size),
                .{ .x = Spec.block_size },
                0,
                &args,
            );
        }

        pub fn run(self: *Self, data: []Spec.Value, params: Spec.Parameters) iface.Error!void {
            if (data.len == 0) return;
            var buffer = try self.alloc(data.len);
            defer buffer.free();
            try buffer.upload(data.ptr, data.len * @sizeOf(Spec.Value));
            try self.launch(&buffer, data.len, params);
            try self.context.synchronize();
            try buffer.download(data.ptr, data.len * @sizeOf(Spec.Value));
        }
    };
}

fn HipKernel(comptime Spec: type) type {
    return struct {
        const Self = @This();
        pub const backend: Backend = .hip;
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
            const internal_name = artifacts.hip_names.resolve(Spec.entry_name);
            const kernel = try module.getKernel(internal_name.ptr);
            return .{ .context = context, .module = module, .kernel = kernel };
        }

        pub fn deinit(self: *Self) void {
            self.module.deinit();
            self.context.deinit();
            self.* = undefined;
        }

        pub fn alloc(self: *Self, count: usize) iface.Error!Buffer {
            return self.context.alloc(count * @sizeOf(Spec.Value));
        }

        pub fn launch(
            self: *Self,
            buffer: *Buffer,
            count: usize,
            params: Spec.Parameters,
        ) iface.Error!void {
            if (count == 0) return;
            var len: u64 = @intCast(count);
            var packed_params = abi.pack(Spec.Parameters, params);
            var args = [_]iface.Arg{
                buffer.argPtr(),
                iface.arg(&len),
                iface.arg(&packed_params),
            };
            try self.kernel.launch(
                iface.Dim3.linear(count, Spec.block_size),
                .{ .x = Spec.block_size },
                0,
                &args,
            );
        }

        pub fn run(self: *Self, data: []Spec.Value, params: Spec.Parameters) iface.Error!void {
            if (data.len == 0) return;
            var buffer = try self.alloc(data.len);
            defer buffer.free();
            try buffer.upload(data.ptr, data.len * @sizeOf(Spec.Value));
            try self.launch(&buffer, data.len, params);
            try self.context.synchronize();
            try buffer.download(data.ptr, data.len * @sizeOf(Spec.Value));
        }
    };
}

pub fn AutoKernel(comptime Spec: type) type {
    const Cpu = Kernel(Spec, .cpu);
    const Cuda = Kernel(Spec, .cuda);
    const Hip = Kernel(Spec, .hip);

    return union(enum) {
        cpu: Cpu,
        cuda: Cuda,
        hip: Hip,

        const Self = @This();

        pub fn init() Self {
            if (Cuda.init(0)) |value| return .{ .cuda = value } else |_| {}
            if (Hip.init(0)) |value| return .{ .hip = value } else |_| {}
            return .{ .cpu = Cpu.init(0) catch unreachable };
        }

        pub fn deinit(self: *Self) void {
            switch (self.*) {
                .cpu => |*k| k.deinit(),
                .cuda => |*k| k.deinit(),
                .hip => |*k| k.deinit(),
            }
        }

        pub fn run(self: *Self, data: []Spec.Value, params: Spec.Parameters) iface.Error!void {
            switch (self.*) {
                .cpu => |*k| try k.run(data, params),
                .cuda => |*k| try k.run(data, params),
                .hip => |*k| try k.run(data, params),
            }
        }

        pub fn selected(self: *const Self) Backend {
            return switch (self.*) {
                .cpu => .cpu,
                .cuda => .cuda,
                .hip => .hip,
            };
        }
    };
}
