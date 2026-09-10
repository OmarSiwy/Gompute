//! Small ABI shared by the runtime backends.

const std = @import("std");

/// Which vendor driver produced the code in `last_driver_error`. `.none` means
/// no failure has been recorded on this thread, not "no backend selected".
pub const DriverBackend = enum { none, cuda, hip };

/// The last raw driver status this thread saw. `code` is the vendor's own value
/// -- a `CUresult` or a `hipError_t` -- so it is only meaningful read together
/// with `backend`. The default `.{}` means no failure.
pub const DriverError = struct {
    code: i64 = 0,
    backend: DriverBackend = .none,
};

/// Written by every backend `check()`, read through `gompute.lastDriverError()`.
///
/// `threadlocal` is load-bearing, not defensive: host threads submit work
/// independently, so a plain global would race and would report another
/// thread's failure as yours.
pub threadlocal var last_driver_error: DriverError = .{};

/// Every backend `check()` reports its raw result here, success included.
/// Recording only failures left the last code standing forever, so
/// `lastDriverError()` could not tell a fresh failure from one made three
/// calls ago -- and reported one on a thread that had never failed at all.
pub inline fn recordDriverResult(backend: DriverBackend, code: i64) void {
    last_driver_error = if (code == 0) .{} else .{ .code = code, .backend = backend };
}

/// Every error gompute itself returns. Deliberately small and vendor-neutral:
/// the driver's own status code is not a member, it goes to
/// `last_driver_error`, so one error set covers both CUDA and HIP.
pub const Error = error{
    InitFailed,
    NoDevice,
    ContextFailed,
    SyncFailed,
    AllocFailed,
    ModuleLoadFailed,
    KernelNotFound,
    LaunchFailed,
    CopyFailed,
    InvalidArgument,
    BackendUnavailable,
    UnsupportedBackend,
};

/// A launch geometry: a grid measured in blocks, or a block measured in
/// threads, depending on which argument it is.
///
/// `extern`, and in this field order, because it is handed straight to
/// `cuLaunchKernel`/`hipModuleLaunchKernel`. The defaults of 1 make
/// `.{ .x = n }` a 1-D launch.
pub const Dim3 = extern struct {
    x: u32 = 1,
    y: u32 = 1,
    z: u32 = 1,

    /// CUDA and HIP both cap `gridDim.x` at 2^31-1; the driver rejects a
    /// launch past it.
    pub const max_grid_x: u32 = (1 << 31) - 1;

    /// The grid of `block_x`-wide blocks that covers `n` elements.
    ///
    /// `n == 0` is an empty range, not a fault: it yields a zero grid, which no
    /// driver will launch, so callers that can see an empty range must return
    /// before launching (`Kernel.launch` does). Everything else this cannot
    /// express -- a grid past `max_grid_x`, or `block_x == 0` -- is the caller
    /// sizing a launch no GPU can run, so it panics.
    ///
    /// ponytail: panic, not a clamp and not a new signature. Clamping would
    /// launch fewer blocks than elements and quietly leave the tail of the
    /// buffer unprocessed -- a wrong answer with no signal, the worst of the
    /// three. Returning `Error!Dim3` is the honest form but this is public API
    /// with callers that use the result directly, so it lives next door as
    /// `linearChecked`; use it wherever the count comes from outside.
    pub fn linear(n: usize, block_x: u32) Dim3 {
        if (n == 0) return .{ .x = 0 };
        return linearChecked(n, block_x) catch @panic(
            "gompute: Dim3.linear cannot express this launch: block_x is 0, or the grid " ++
                "exceeds gridDim.x = 2^31-1. Raise block_x, split the range, or call " ++
                "Dim3.linearChecked to handle it as an error.",
        );
    }

    /// `linear` for counts that come from outside the program. Rejects an empty
    /// range, a zero block, and any grid the driver would refuse -- including
    /// the ones that used to wrap a u32 in ReleaseFast (n > 2^32 * block_x).
    pub fn linearChecked(n: usize, block_x: u32) Error!Dim3 {
        if (n == 0 or block_x == 0) return error.InvalidArgument;
        const blocks = (@as(u64, n) + block_x - 1) / block_x;
        if (blocks > max_grid_x) return error.InvalidArgument;
        return .{ .x = @intCast(blocks) };
    }
};

/// One element in CUDA/HIP's `void **kernelParams` array: a pointer to the
/// storage holding one kernel argument.
pub const Arg = *anyopaque;

/// Wrap a pointer to one argument's storage for the `kernelParams` array.
///
/// Does not copy. The driver reads through the pointer when the launch is
/// submitted, so the pointee must stay alive and unmoved until then -- passing
/// a pointer to a temporary is how a kernel receives a garbage argument.
pub inline fn arg(value_ptr: anytype) Arg {
    return @ptrCast(@constCast(value_ptr));
}

test "Dim3.linear covers the range and keeps the empty grid" {
    try std.testing.expectEqual(@as(u32, 4), Dim3.linear(1024, 256).x);
    try std.testing.expectEqual(@as(u32, 5), Dim3.linear(1025, 256).x);
    try std.testing.expectEqual(@as(u32, 1), Dim3.linear(1, 256).x);
    try std.testing.expectEqual(@as(u32, 0), Dim3.linear(0, 256).x);
    const d = Dim3.linear(1, 256);
    try std.testing.expectEqual(@as(u32, 1), d.y);
    try std.testing.expectEqual(@as(u32, 1), d.z);
}

test "Dim3.linearChecked rejects the launches no driver accepts" {
    // Empty range: a zero grid is an argument error to cuLaunchKernel.
    try std.testing.expectError(error.InvalidArgument, Dim3.linearChecked(0, 256));
    try std.testing.expectError(error.InvalidArgument, Dim3.linearChecked(1, 0));

    // Past gridDim.x = 2^31-1. Exactly at the cap still launches.
    try std.testing.expectEqual(
        @as(u32, Dim3.max_grid_x),
        (try Dim3.linearChecked(@as(usize, Dim3.max_grid_x) * 256, 256)).x,
    );
    try std.testing.expectError(
        error.InvalidArgument,
        Dim3.linearChecked(@as(usize, Dim3.max_grid_x) * 256 + 1, 256),
    );
    try std.testing.expectError(error.InvalidArgument, Dim3.linearChecked(1 << 39, 256));

    // Used to overflow the u32 outright: silently in ReleaseFast, panicking
    // otherwise. block_size = 1 is legal, so 2^32 u8 elements reach it.
    try std.testing.expectError(error.InvalidArgument, Dim3.linearChecked(1 << 32, 1));
    try std.testing.expectError(error.InvalidArgument, Dim3.linearChecked(1 << 40, 256));
}

test recordDriverResult {
    recordDriverResult(.cuda, 700);
    try std.testing.expectEqual(DriverBackend.cuda, last_driver_error.backend);
    try std.testing.expectEqual(@as(i64, 700), last_driver_error.code);

    recordDriverResult(.cuda, 0);
    try std.testing.expectEqual(DriverError{}, last_driver_error);
}
