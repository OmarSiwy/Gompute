//! Import this module as `gompute` when cross-compiling the user's kernel root.
//!
//! The API-shaped mirror of `src/root.zig`: the same kernels file is compiled
//! twice, host-side against that root and device-side against this one. Every
//! name below either aliases the same thing as its counterpart there, or is one
//! of the five deliberate polarity switches -- `is_device`, `kernel_callconv`,
//! `GlobalPtr`, `globalIdX`, `exportRaw` -- which are the whole reason two
//! roots exist. Adding a name here without adding it there
//! breaks a kernels file that compiles for both.

const std = @import("std");

/// Thread-index intrinsics. Device-only, so it has no counterpart in
/// `src/root.zig` -- a kernels file may only name it under `if (is_device)`.
pub const builtins = @import("device/builtins.zig");
/// Device-safe libm. Same source compiles host-side, forwarding to `std.math`.
pub const math = @import("device/math.zig");

/// True here, false in `src/root.zig`. Branch on it in a kernel file that has
/// to compile for both.
pub const is_device = true;
/// `.kernel` here, `.auto` in `src/root.zig`. Put it on a raw entry point.
pub const kernel_callconv: std.lang.CallingConvention = .kernel;

/// `[*]addrspace(.global) T` here; a plain `[*]T` in `src/root.zig`.
pub fn GlobalPtr(comptime T: type) type {
    return [*]addrspace(.global) T;
}

/// Alias, not a wrapper, so `g.globalIdX` carries `builtins.globalIdX`'s
/// `block_size` hazard doc at the name users actually type.
pub const globalIdX = builtins.globalIdX;

/// Export a hand-written entry point under `name`, for `RawKernel` to load.
/// A no-op in `src/root.zig`, so guard the call with `if (is_device)`.
///
/// `function` must be a POINTER to the function (`&myKernel`). `@export` takes
/// one, but `anytype` accepts a bare function value, and the host root accepts
/// literally anything -- so the mistake compiles clean host-side and fails here
/// with an error pointing into this file rather than the caller's.
pub fn exportRaw(comptime name: []const u8, comptime function: anytype) void {
    @export(function, .{ .name = name });
}
