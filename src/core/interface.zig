//! Small ABI shared by the runtime backends.

pub const DriverBackend = enum { none, cuda, hip };

pub const DriverError = struct {
    code: i64 = 0,
    backend: DriverBackend = .none,
};

pub threadlocal var last_driver_error: DriverError = .{};

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

pub const Dim3 = extern struct {
    x: u32 = 1,
    y: u32 = 1,
    z: u32 = 1,

    pub fn linear(n: usize, block_x: u32) Dim3 {
        const n64: u64 = @intCast(n);
        const b64: u64 = block_x;
        return .{ .x = @intCast((n64 + b64 - 1) / b64) };
    }
};

/// One element in CUDA/HIP's `void **kernelParams` array: a pointer to the
/// storage holding one kernel argument.
pub const Arg = *anyopaque;

pub inline fn arg(value_ptr: anytype) Arg {
    return @ptrCast(@constCast(value_ptr));
}
