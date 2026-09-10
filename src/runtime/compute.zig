//! Uniform GPU facade: CUDA -> HIP -> CPU fallback.

const std = @import("std");
const cuda = @import("cuda.zig");
const hip = @import("hip.zig");
const iface = @import("../core/interface.zig");

/// Which backend a handle came from. Distinct from `gompute.Backend`
/// (`host/kernel.zig`), which is a separate enum with the same three tags --
/// they do not coerce to each other.
pub const Backend = enum { cpu, cuda, hip };
/// Launch geometry; see `core/interface.zig`.
pub const Dim3 = iface.Dim3;
/// One kernel argument: a pointer to caller-owned storage that must outlive the
/// launch. See `core/interface.zig`.
pub const Arg = iface.Arg;
/// The error set every call in this file returns; see `core/interface.zig`.
pub const Error = iface.Error;

/// A device handle over whichever backend `init` found. Holds a context for
/// both GPU backends and uses the one `backend` names.
pub const Compute = struct {
    backend: Backend,
    cuda_ctx: cuda.Context = .{},
    hip_ctx: hip.Context = .{},

    /// Probe CUDA, then HIP, then settle for `.cpu`. Never fails: a machine with
    /// no GPU at all is an ordinary outcome, which is also why the three probe
    /// results are `std.log.debug` and not a print to stderr -- see f08b34b,
    /// which did the same to cuda.zig's own "no NVIDIA driver here" line.
    ///
    /// Pass a `preferred` backend to skip the probe and take that one or fail.
    pub fn init(preferred: ?Backend) Error!Compute {
        if (preferred) |p| return initBackend(p);
        if (initBackend(.cuda)) |c| return c else |e| {
            std.log.debug("gompute: cuda init failed: {s}", .{@errorName(e)});
        }
        if (initBackend(.hip)) |c| return c else |e| {
            std.log.debug("gompute: hip init failed: {s}", .{@errorName(e)});
        }
        std.log.debug("gompute: no GPU backend available, using CPU fallback", .{});
        return .{ .backend = .cpu };
    }

    fn initBackend(b: Backend) Error!Compute {
        return switch (b) {
            .cuda => .{ .backend = .cuda, .cuda_ctx = try cuda.Context.init(0) },
            .hip => .{ .backend = .hip, .hip_ctx = try hip.Context.init(0) },
            .cpu => .{ .backend = .cpu },
        };
    }

    /// Init on a chosen device ordinal.
    ///
    /// Unlike `init`, this does not probe and does not fall back: a null
    /// `preferred` means `.cuda`, and on a machine without CUDA it returns
    /// `error.InitFailed` rather than trying HIP or settling for `.cpu`.
    pub fn initDevice(preferred: ?Backend, ordinal: c_int) Error!Compute {
        const b = preferred orelse .cuda;
        return switch (b) {
            .cuda => .{ .backend = .cuda, .cuda_ctx = try cuda.Context.init(ordinal) },
            .hip => .{ .backend = .hip, .hip_ctx = try hip.Context.init(ordinal) },
            .cpu => .{ .backend = .cpu },
        };
    }

    /// (#3) Drops this handle only; the device's primary context is shared with
    /// every other handle in the process and stays retained. Use `shutdown()`
    /// for real teardown.
    pub fn deinit(self: *Compute) void {
        switch (self.backend) {
            .cuda => self.cuda_ctx.deinit(),
            .hip => self.hip_ctx.deinit(),
            .cpu => {},
        }
    }

    /// Block until every launch on this device has finished. A no-op on `.cpu`.
    pub fn synchronize(self: *Compute) Error!void {
        switch (self.backend) {
            .cuda => try self.cuda_ctx.synchronize(),
            .hip => try self.hip_ctx.synchronize(),
            .cpu => {},
        }
    }

    /// Allocate device memory. Caller owns the returned Buffer and must
    /// `free` it. Returns `error.AllocFailed` on the `.cpu` backend, which has
    /// no device to allocate on.
    pub fn alloc(self: *Compute, bytes: usize) Error!Buffer {
        return switch (self.backend) {
            .cuda => .{ .cuda = try self.cuda_ctx.alloc(bytes) },
            .hip => .{ .hip = try self.hip_ctx.alloc(bytes) },
            .cpu => error.AllocFailed,
        };
    }

    /// JIT a module image, cached per (device, image) by the backend. The
    /// sentinel on `image` is load-bearing for CUDA -- `cuModuleLoadData` reads
    /// PTX up to a NUL. Returns `error.ModuleLoadFailed` on `.cpu`.
    pub fn loadModule(self: *Compute, image: [:0]const u8) Error!Module {
        return switch (self.backend) {
            .cuda => .{ .cuda = try self.cuda_ctx.loadModuleFromMemory(image) },
            .hip => .{ .hip = try self.hip_ctx.loadModuleFromMemory(image) },
            .cpu => error.ModuleLoadFailed,
        };
    }

    /// Create a stream for ordered asynchronous work. Caller owns the returned
    /// Stream and must `deinit` it. Returns `error.InitFailed` on `.cpu`.
    pub fn createStream(self: *Compute) Error!Stream {
        return switch (self.backend) {
            .cuda => .{ .cuda = try self.cuda_ctx.createStream() },
            .hip => .{ .hip = try self.hip_ctx.createStream() },
            .cpu => error.InitFailed,
        };
    }
};

/// Tear down every shared context and cached module on both backends. Nothing
/// calls this for you; `deinit` deliberately leaves shared state alone. Only
/// call it once every handle in the process is done.
pub fn shutdown() void {
    cuda.shutdown();
    hip.shutdown();
}

/// Device memory on whichever backend allocated it.
///
/// Asserts the backend is not `.cpu` in every method except `free`: a `.cpu`
/// Buffer cannot exist, because `Compute.alloc` refuses to make one.
pub const Buffer = union(Backend) {
    cpu: void,
    cuda: cuda.Buffer,
    hip: hip.Buffer,

    /// Copy `n` bytes from host memory into the buffer, and wait.
    pub fn upload(self: *Buffer, host: *const anyopaque, n: usize) Error!void {
        switch (self.*) {
            .cuda => |*b| try b.upload(host, n),
            .hip => |*b| try b.upload(host, n),
            .cpu => unreachable,
        }
    }
    /// Copy `n` bytes out of the buffer into host memory, and wait.
    pub fn download(self: *Buffer, host: *anyopaque, n: usize) Error!void {
        switch (self.*) {
            .cuda => |*b| try b.download(host, n),
            .hip => |*b| try b.download(host, n),
            .cpu => unreachable,
        }
    }
    /// `download`, starting `offset` bytes into the buffer. Not bounds-checked
    /// against the allocation.
    pub fn downloadAt(self: *Buffer, host: *anyopaque, offset: usize, n: usize) Error!void {
        switch (self.*) {
            .cuda => |*b| try b.downloadAt(host, offset, n),
            .hip => |*b| try b.downloadAt(host, offset, n),
            .cpu => unreachable,
        }
    }
    /// `upload`, starting `offset` bytes into the buffer. Not bounds-checked
    /// against the allocation.
    pub fn uploadAt(self: *Buffer, host: *const anyopaque, offset: usize, n: usize) Error!void {
        switch (self.*) {
            .cuda => |*b| try b.uploadAt(host, offset, n),
            .hip => |*b| try b.uploadAt(host, offset, n),
            .cpu => unreachable,
        }
    }
    /// Release the device memory. Idempotent, and a no-op on `.cpu`.
    pub fn free(self: *Buffer) void {
        switch (self.*) {
            .cuda => |*b| b.free(),
            .hip => |*b| b.free(),
            .cpu => {},
        }
    }
    /// The buffer's handle as a kernel argument. Borrows `self`: the pointer is
    /// into the Buffer, which must outlive the launch it is passed to.
    pub fn argPtr(self: *Buffer) Arg {
        return switch (self.*) {
            .cuda => |*b| b.argPtr(),
            .hip => |*b| b.argPtr(),
            .cpu => unreachable,
        };
    }
    /// Device-to-device copy of `n` bytes. Asserts `src` is on the same backend
    /// as `self`.
    pub fn copyFrom(self: *Buffer, src: *const Buffer, src_offset: usize, dst_offset: usize, n: usize) Error!void {
        switch (self.*) {
            .cuda => |*b| try b.copyFrom(&src.cuda, src_offset, dst_offset, n),
            .hip => |*b| try b.copyFrom(&src.hip, src_offset, dst_offset, n),
            .cpu => unreachable,
        }
    }
    /// The raw device address, for printing or for passing to code outside this
    /// library. Not dereferenceable from the host.
    pub fn deviceAddr(self: *const Buffer) u64 {
        return switch (self.*) {
            .cuda => |b| b.handle,
            .hip => |b| @intFromPtr(b.handle),
            .cpu => unreachable,
        };
    }
};

/// A loaded module. Asserts the backend is not `.cpu` in `getKernel`.
pub const Module = union(Backend) {
    cpu: void,
    cuda: cuda.Module,
    hip: hip.Module,

    /// Look up an entry point by its mangled symbol name.
    pub fn getKernel(self: *Module, name: [*:0]const u8) Error!Kernel {
        return switch (self.*) {
            .cuda => |*m| .{ .cuda = try m.getKernel(name) },
            .hip => |*m| .{ .hip = try m.getKernel(name) },
            .cpu => unreachable,
        };
    }
    /// (#3) Drops this handle. A cached module stays loaded for the process;
    /// `shutdown()` unloads it.
    pub fn deinit(self: *Module) void {
        switch (self.*) {
            .cuda => |*m| m.deinit(),
            .hip => |*m| m.deinit(),
            .cpu => {},
        }
    }
};

/// An entry point. Asserts the backend is not `.cpu` in both launch methods.
pub const Kernel = union(Backend) {
    cpu: void,
    cuda: cuda.Kernel,
    hip: hip.Kernel,

    /// Launch on the default stream. `args` must outlive the call.
    pub fn launch(self: Kernel, grid: Dim3, block: Dim3, shared_bytes: u32, args: []const Arg) Error!void {
        switch (self) {
            .cuda => |k| try k.launch(grid, block, shared_bytes, args),
            .hip => |k| try k.launch(grid, block, shared_bytes, args),
            .cpu => unreachable,
        }
    }
    /// Launch on `stream` and return without waiting. `args` must stay put until
    /// `stream.synchronize()` returns. Asserts `stream` is on the same backend
    /// as the kernel.
    pub fn launchOnStream(self: Kernel, grid: Dim3, block: Dim3, shared_bytes: u32, args: []const Arg, stream: *Stream) Error!void {
        switch (self) {
            .cuda => |k| try k.launchOnStream(grid, block, shared_bytes, args, stream.cuda.stream),
            .hip => |k| try k.launchOnStream(grid, block, shared_bytes, args, stream.hip.stream),
            .cpu => unreachable,
        }
    }
};

/// A queue of ordered asynchronous work. Asserts the backend is not `.cpu` in
/// `synchronize`.
pub const Stream = union(Backend) {
    cpu: void,
    cuda: cuda.Stream,
    hip: hip.Stream,

    /// Block until everything queued on this stream has finished.
    pub fn synchronize(self: *Stream) Error!void {
        switch (self.*) {
            .cuda => |*s| try s.synchronize(),
            .hip => |*s| try s.synchronize(),
            .cpu => unreachable,
        }
    }
    /// Destroy the stream. A no-op on `.cpu`.
    pub fn deinit(self: *Stream) void {
        switch (self.*) {
            .cuda => |*s| s.deinit(),
            .hip => |*s| s.deinit(),
            .cpu => {},
        }
    }
};

test "the .cpu arm fails politely instead of pretending" {
    // No driver of any kind involved: this is the fallback `init` lands on when
    // a machine has neither, and every call on it has to be an ordinary error.
    var c: Compute = .{ .backend = .cpu };
    defer c.deinit();
    try c.synchronize();
    try std.testing.expectError(error.AllocFailed, c.alloc(16));
    try std.testing.expectError(error.ModuleLoadFailed, c.loadModule(""));
    try std.testing.expectError(error.InitFailed, c.createStream());
}

test "Compute.init picks a backend and never fails" {
    // The contract is that probing never fails, whatever the machine has: on a
    // box with no driver this is `.cpu`, on one with a driver it is that driver.
    var probed = try Compute.init(null);
    defer probed.deinit();

    var cpu = try Compute.init(.cpu);
    defer cpu.deinit();
    try std.testing.expectEqual(Backend.cpu, cpu.backend);
}
