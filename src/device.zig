//! Import this module as `gompute` when cross-compiling the user's kernel root.
//!
//! The API-shaped mirror of `src/root.zig`: the same kernels file is compiled
//! twice, host-side against that root and device-side against this one. Every
//! name below either aliases the same thing as its counterpart there, or is one
//! of the six deliberate polarity switches -- `is_device`, `kernel_callconv`,
//! `GlobalPtr`, `globalIdX`, `exportRaw`, `exportKernels` -- which are the
//! whole reason two roots exist. Adding a name here without adding it there
//! breaks a kernels file that compiles for both.

const std = @import("std");

/// The host/device calling convention: how a kernel's parameters are laid out.
pub const abi = @import("core/abi.zig");
/// Chaining operations into a single pass. See `Fused` and `Unary`.
pub const fusion = @import("core/fusion.zig");
/// The operation constructors themselves; `map` and friends are aliases into it.
pub const spec = @import("core/spec.zig");
/// Thread-index intrinsics. Device-only, so it has no counterpart in
/// `src/root.zig` -- a kernels file may only name it under `if (is_device)`.
pub const builtins = @import("device/builtins.zig");
/// Device-safe libm. Same source compiles host-side, forwarding to `std.math`.
pub const math = @import("device/math.zig");

/// Per-operation knobs shared by `map`, `mapFn` and `mapTo`.
pub const MapOptions = spec.MapOptions;
/// True here, false in `src/root.zig`. Branch on it in a kernel file that has
/// to compile for both.
pub const is_device = true;
/// `.kernel` here, `.auto` in `src/root.zig`. Put it on a raw entry point.
pub const kernel_callconv: std.builtin.CallingConvention = .kernel;

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
/// Emit a `callconv(.kernel)` entry point for every spec handed to it: a tuple
/// of specs, or a module type whose pub decls are scanned for them.
///
/// A no-op in `src/root.zig`. Asserts at compile time that the target is
/// `nvptx64` or `amdgcn`.
///
/// Passing `@This()` force-analyses every pub decl in that file for the GPU
/// target, so a pub decl the device cannot compile -- or a `pub var`, which
/// cannot be read by `@field` at comptime at all -- errors inside
/// `device/export.zig` rather than at the decl.
pub const exportKernels = @import("device/export.zig").exportAll;
