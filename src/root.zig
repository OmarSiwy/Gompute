//! Gompute: comptime map/fusion plus thin CUDA/HIP runtime backends.

const std = @import("std");

pub const abi = @import("core/abi.zig");
pub const fusion = @import("core/fusion.zig");
pub const spec = @import("core/spec.zig");
pub const interface = @import("core/interface.zig");

const host = @import("host/kernel.zig");

pub const Backend = host.Backend;
pub const Kernel = host.Kernel;
pub const AutoKernel = host.AutoKernel;
pub const RawKernel = @import("host/raw.zig").RawKernel;
pub const MapOptions = spec.MapOptions;
pub const is_device = false;
pub const kernel_callconv: std.builtin.CallingConvention = .auto;

pub fn GlobalPtr(comptime T: type) type {
    return [*]T;
}

pub inline fn globalIdX(comptime _: u32) usize {
    @compileError("globalIdX is only available in a device compilation");
}

pub fn exportRaw(comptime _: []const u8, comptime _: anytype) void {}
pub const Dim3 = interface.Dim3;
pub const Error = interface.Error;

pub fn map(
    comptime name: [:0]const u8,
    comptime T: type,
    comptime Params: type,
    comptime func: anytype,
    comptime options: MapOptions,
) type {
    return spec.Map(name, T, Params, func, options);
}

pub const Fused = fusion.Fused;
pub const Unary = fusion.Unary;

/// Host-side no-op. The same kernels module is recompiled with `gompute_device`,
/// where this call instantiates actual GPU entry points.
pub fn exportKernels(comptime _: anytype) void {}

pub const runtime = struct {
    pub const dynamic = @import("runtime/compute.zig");
    pub const cuda = @import("runtime/cuda.zig");
    pub const hip = @import("runtime/hip.zig");
};

pub fn lastDriverError() interface.DriverError {
    return interface.last_driver_error;
}

test {
    _ = abi;
    _ = fusion;
    _ = spec;
    _ = runtime.cuda.Context;
    _ = runtime.hip.Context;
    _ = runtime.dynamic.Compute;
}
