//! Gompute: hand-written GPU kernels compiled at build time, a thin CUDA/HIP
//! runtime to launch them, and device-safe math.

const std = @import("std");

/// Backend-independent vocabulary: `Dim3`, `Error`, driver-error recording.
pub const interface = @import("core/interface.zig");
/// Device-safe libm. Same source compiles host-side, forwarding to `std.math`.
pub const math = @import("device/math.zig");

const raw = @import("host/raw.zig");

/// Where a kernel runs. `RawKernel` takes the GPU tags; naming `.cpu` there is
/// a compile error.
pub const Backend = raw.Backend;
/// A hand-written GPU entry point, loaded by its exported name.
pub const RawKernel = raw.RawKernel;
/// Load a kernel whose name is only known at run time. See `host/raw.zig`.
pub const rawKernelByName = raw.rawKernelByName;
/// The handle `rawKernelByName` returns.
pub const RawByName = raw.RawByName;

/// Whether this build emitted device artifacts for `backend`: ask before
/// naming `RawKernel(name, backend)`, which is a compile error when it did not.
/// Always false for `.cpu`.
pub fn available(comptime backend: Backend) bool {
    return switch (backend) {
        .cpu => false,
        .cuda => raw.available(raw.gpu_cuda),
        .hip => raw.available(raw.gpu_hip),
    };
}

/// False here, true in `src/device.zig`. Branch on it in a kernel file that has
/// to compile for both.
pub const is_device = false;
/// `.auto` here, `.kernel` in `src/device.zig`. Put it on a raw entry point.
pub const kernel_callconv: std.lang.CallingConvention = .auto;

/// A plain `[*]T` here; `[*]addrspace(.global) T` in `src/device.zig`.
pub fn GlobalPtr(comptime T: type) type {
    return [*]T;
}

/// Device-only. Asserts at compile time that this is a device compilation, so
/// naming it in host code names the mistake instead of silently returning 0.
pub inline fn globalIdX(comptime _: u32) usize {
    @compileError("globalIdX is only available in a device compilation");
}

/// Host-side no-op; the device root `@export`s the function.
pub fn exportRaw(comptime _: []const u8, comptime _: anytype) void {}

/// A launch geometry, in the layout both drivers expect.
pub const Dim3 = interface.Dim3;
/// Every error gompute's host side can return.
pub const Error = interface.Error;

/// The driver bindings underneath `RawKernel`, for what it does not cover:
/// raw driver surface.
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
    _ = math;
    _ = raw;
    _ = runtime.cuda.Context;
    _ = runtime.hip.Context;
    _ = runtime.dynamic.Compute;
}
