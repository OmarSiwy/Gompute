//! Gompute: comptime map/fusion plus thin CUDA/HIP runtime backends.

const std = @import("std");

pub const abi = @import("core/abi.zig");
pub const fusion = @import("core/fusion.zig");
pub const spec = @import("core/spec.zig");
pub const interface = @import("core/interface.zig");
/// Device-safe libm. Same source compiles host-side, forwarding to `std.math`.
pub const math = @import("device/math.zig");

const host = @import("host/kernel.zig");

pub const Backend = host.Backend;
pub const Kernel = host.Kernel;
pub const AutoKernel = host.AutoKernel;
pub const RawKernel = @import("host/raw.zig").RawKernel;
/// Load a kernel whose name is only known at run time. See `host/raw.zig`.
pub const rawKernelByName = @import("host/raw.zig").rawKernelByName;
pub const RawByName = @import("host/raw.zig").RawByName;
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

/// The operation set. Aliases, not wrappers: repeating these signatures in both
/// roots is two places to get them wrong.
pub const map = spec.Map;
/// `map` with `T` and `Params` read off `func`'s signature.
pub const mapFn = spec.MapFn;
pub const splat = spec.splat;
pub const mapTo = spec.MapTo;
pub const zip = spec.Zip;
pub const mapIndexed = spec.MapIndexed;
pub const reduce = spec.Reduce;
pub const sum = spec.Sum;
pub const min = spec.Min;
pub const max = spec.Max;
pub const any = spec.Any;
pub const all = spec.All;
pub const gather = spec.Gather;
pub const scatter = spec.Scatter;
pub const ReduceOptions = spec.ReduceOptions;
pub const Kind = spec.Kind;

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
    _ = math;
    _ = fusion;
    _ = spec;
    _ = host;
    _ = runtime.cuda.Context;
    _ = runtime.hip.Context;
    _ = runtime.dynamic.Compute;
}
