//! Import this module as `gompute` when cross-compiling the user's kernel root.

const std = @import("std");

pub const abi = @import("core/abi.zig");
pub const fusion = @import("core/fusion.zig");
pub const spec = @import("core/spec.zig");
pub const builtins = @import("device/builtins.zig");
pub const math = @import("device/math.zig");

pub const MapOptions = spec.MapOptions;
pub const is_device = true;
pub const kernel_callconv: std.builtin.CallingConvention = .kernel;

pub fn GlobalPtr(comptime T: type) type {
    return [*]addrspace(.global) T;
}

pub inline fn globalIdX(comptime block_size: u32) usize {
    return builtins.globalIdX(block_size);
}

pub fn exportRaw(comptime name: []const u8, comptime function: anytype) void {
    @export(function, .{ .name = name });
}

pub fn map(
    comptime name: [:0]const u8,
    comptime T: type,
    comptime Params: type,
    comptime func: anytype,
    comptime options: MapOptions,
) type {
    return spec.Map(name, T, Params, func, options);
}

/// `map` with `T` and `Params` read off `func`'s signature.
pub fn mapFn(
    comptime name: [:0]const u8,
    comptime func: anytype,
    comptime options: MapOptions,
) type {
    return spec.MapFn(name, func, options);
}

pub const splat = spec.splat;

/// The rest of the operation set. Aliases, not wrappers: repeating these
/// signatures in both roots is two places to get them wrong.
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
pub const exportKernels = @import("device/export.zig").exportAll;
