//! Uniform GPU facade: CUDA -> HIP -> CPU fallback.

const std = @import("std");
const cuda = @import("cuda.zig");
const hip = @import("hip.zig");
const iface = @import("../core/interface.zig");

pub const Backend = enum { cpu, cuda, hip };
pub const Dim3 = iface.Dim3;
pub const Arg = iface.Arg;
pub const Error = iface.Error;

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

    pub fn synchronize(self: *Compute) Error!void {
        switch (self.backend) {
            .cuda => try self.cuda_ctx.synchronize(),
            .hip => try self.hip_ctx.synchronize(),
            .cpu => {},
        }
    }

    pub fn alloc(self: *Compute, bytes: usize) Error!Buffer {
        return switch (self.backend) {
            .cuda => .{ .cuda = try self.cuda_ctx.alloc(bytes) },
            .hip => .{ .hip = try self.hip_ctx.alloc(bytes) },
            .cpu => error.AllocFailed,
        };
    }

    pub fn loadModule(self: *Compute, image: [:0]const u8) Error!Module {
        return switch (self.backend) {
            .cuda => .{ .cuda = try self.cuda_ctx.loadModuleFromMemory(image) },
            .hip => .{ .hip = try self.hip_ctx.loadModuleFromMemory(image) },
            .cpu => error.ModuleLoadFailed,
        };
    }

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

pub const Buffer = union(Backend) {
    cpu: void,
    cuda: cuda.Buffer,
    hip: hip.Buffer,

    pub fn upload(self: *Buffer, host: *const anyopaque, n: usize) Error!void {
        switch (self.*) {
            .cuda => |*b| try b.upload(host, n),
            .hip => |*b| try b.upload(host, n),
            .cpu => unreachable,
        }
    }
    pub fn download(self: *Buffer, host: *anyopaque, n: usize) Error!void {
        switch (self.*) {
            .cuda => |*b| try b.download(host, n),
            .hip => |*b| try b.download(host, n),
            .cpu => unreachable,
        }
    }
    pub fn downloadAt(self: *Buffer, host: *anyopaque, offset: usize, n: usize) Error!void {
        switch (self.*) {
            .cuda => |*b| try b.downloadAt(host, offset, n),
            .hip => |*b| try b.downloadAt(host, offset, n),
            .cpu => unreachable,
        }
    }
    pub fn uploadAt(self: *Buffer, host: *const anyopaque, offset: usize, n: usize) Error!void {
        switch (self.*) {
            .cuda => |*b| try b.uploadAt(host, offset, n),
            .hip => |*b| try b.uploadAt(host, offset, n),
            .cpu => unreachable,
        }
    }
    pub fn free(self: *Buffer) void {
        switch (self.*) {
            .cuda => |*b| b.free(),
            .hip => |*b| b.free(),
            .cpu => {},
        }
    }
    pub fn argPtr(self: *Buffer) Arg {
        return switch (self.*) {
            .cuda => |*b| b.argPtr(),
            .hip => |*b| b.argPtr(),
            .cpu => unreachable,
        };
    }
    pub fn copyFrom(self: *Buffer, src: *const Buffer, src_offset: usize, dst_offset: usize, n: usize) Error!void {
        switch (self.*) {
            .cuda => |*b| try b.copyFrom(&src.cuda, src_offset, dst_offset, n),
            .hip => |*b| try b.copyFrom(&src.hip, src_offset, dst_offset, n),
            .cpu => unreachable,
        }
    }
    pub fn deviceAddr(self: *const Buffer) u64 {
        return switch (self.*) {
            .cuda => |b| b.handle,
            .hip => |b| @intFromPtr(b.handle),
            .cpu => unreachable,
        };
    }
};

pub const Module = union(Backend) {
    cpu: void,
    cuda: cuda.Module,
    hip: hip.Module,

    pub fn getKernel(self: *Module, name: [*:0]const u8) Error!Kernel {
        return switch (self.*) {
            .cuda => |*m| .{ .cuda = try m.getKernel(name) },
            .hip => |*m| .{ .hip = try m.getKernel(name) },
            .cpu => unreachable,
        };
    }
    pub fn deinit(self: *Module) void {
        switch (self.*) {
            .cuda => |*m| m.deinit(),
            .hip => |*m| m.deinit(),
            .cpu => {},
        }
    }
};

pub const Kernel = union(Backend) {
    cpu: void,
    cuda: cuda.Kernel,
    hip: hip.Kernel,

    pub fn launch(self: Kernel, grid: Dim3, block: Dim3, shared_bytes: u32, args: []const Arg) Error!void {
        switch (self) {
            .cuda => |k| try k.launch(grid, block, shared_bytes, args),
            .hip => |k| try k.launch(grid, block, shared_bytes, args),
            .cpu => unreachable,
        }
    }
    pub fn launchOnStream(self: Kernel, grid: Dim3, block: Dim3, shared_bytes: u32, args: []const Arg, stream: *Stream) Error!void {
        switch (self) {
            .cuda => |k| try k.launchOnStream(grid, block, shared_bytes, args, stream.cuda.stream),
            .hip => |k| try k.launchOnStream(grid, block, shared_bytes, args, stream.hip.stream),
            .cpu => unreachable,
        }
    }
};

pub const Stream = union(Backend) {
    cpu: void,
    cuda: cuda.Stream,
    hip: hip.Stream,

    pub fn synchronize(self: *Stream) Error!void {
        switch (self.*) {
            .cuda => |*s| try s.synchronize(),
            .hip => |*s| try s.synchronize(),
            .cpu => unreachable,
        }
    }
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
