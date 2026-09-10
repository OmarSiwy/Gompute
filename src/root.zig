//! Gompute: comptime map/fusion plus thin CUDA/HIP runtime backends.

const std = @import("std");

/// The host/device calling convention: how a kernel's parameters are laid out.
pub const abi = @import("core/abi.zig");
/// Chaining operations into a single pass. See `Fused` and `Unary`.
pub const fusion = @import("core/fusion.zig");
/// The operation constructors themselves; `map` and friends are aliases into it.
pub const spec = @import("core/spec.zig");
/// Backend-independent vocabulary: `Dim3`, `Error`, driver-error recording.
pub const interface = @import("core/interface.zig");
/// Device-safe libm. Same source compiles host-side, forwarding to `std.math`.
pub const math = @import("device/math.zig");

const host = @import("host/kernel.zig");

/// Where a kernel runs. `.cpu` always exists; the GPU tags need the matching
/// backend to have been emitted by `build.zig`.
pub const Backend = host.Backend;
/// A launchable instance of a spec on one named backend.
///
/// Naming a GPU backend the build did not emit is a compile error on purpose:
/// it is a build.zig bug and belongs at build time. Use `AutoKernel` when the
/// answer should be decided on the deploy machine instead.
pub const Kernel = host.Kernel;
/// A kernel that picks the fastest backend that actually works, falling back to
/// the CPU. Compiles whether or not the build emitted any GPU artifacts.
pub const AutoKernel = host.AutoKernel;
/// A hand-written GPU entry point, loaded by its exported name.
pub const RawKernel = @import("host/raw.zig").RawKernel;
/// Load a kernel whose name is only known at run time. See `host/raw.zig`.
pub const rawKernelByName = @import("host/raw.zig").rawKernelByName;
/// The handle `rawKernelByName` returns.
pub const RawByName = @import("host/raw.zig").RawByName;
/// Per-operation knobs shared by `map`, `mapFn` and `mapTo`.
pub const MapOptions = spec.MapOptions;

/// False here, true in `src/device.zig`. Branch on it in a kernel file that has
/// to compile for both.
pub const is_device = false;
/// `.auto` here, `.kernel` in `src/device.zig`. Put it on a raw entry point.
pub const kernel_callconv: std.builtin.CallingConvention = .auto;

/// A plain `[*]T` here; `[*]addrspace(.global) T` in `src/device.zig`.
pub fn GlobalPtr(comptime T: type) type {
    return [*]T;
}

/// Device-only. Asserts at compile time that this is a device compilation, so
/// naming it in host code names the mistake instead of silently returning 0.
pub inline fn globalIdX(comptime _: u32) usize {
    @compileError("globalIdX is only available in a device compilation");
}

/// Host-side no-op; the device root `@export`s the function. See `exportKernels`.
pub fn exportRaw(comptime _: []const u8, comptime _: anytype) void {}

/// A launch geometry, in the layout both drivers expect.
pub const Dim3 = interface.Dim3;
/// Every error gompute's host side can return.
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
/// Knobs for `reduce`, `sum`, `min`, `max`, `any` and `all`.
pub const ReduceOptions = spec.ReduceOptions;
/// Which operation a spec is, for code that inspects one generically.
pub const Kind = spec.Kind;

/// Several operations run as a single pass over the data.
pub const Fused = fusion.Fused;
/// One element-wise step inside a `Fused` chain.
pub const Unary = fusion.Unary;

/// Host-side no-op. The same kernels module is recompiled with `gompute_device`,
/// where this call instantiates actual GPU entry points.
pub fn exportKernels(comptime _: anytype) void {}

/// The driver bindings underneath `Kernel`. Reach for these only to do
/// something the kernel layer does not cover -- they are raw driver surface.
pub const runtime = struct {
    /// Picks whichever driver is present at run time.
    pub const dynamic = @import("runtime/compute.zig");
    /// The CUDA driver API, dlopen'd.
    pub const cuda = @import("runtime/cuda.zig");
    /// The HIP driver API, dlopen'd.
    pub const hip = @import("runtime/hip.zig");
};

/// The driver's own code for the most recent failed call *on this thread*, for
/// reporting alongside a returned `Error`. Cleared by the next call that
/// succeeds, so a zero code means "no failure", not "no information".
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
