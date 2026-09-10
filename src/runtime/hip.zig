///! HIP backend via runtime loading (dlopen), mirroring backend_cuda.zig.
const std = @import("std");
const builtin = @import("builtin");
const iface = @import("../core/interface.zig");
const Dim3 = iface.Dim3;
const Error = iface.Error;

const hipError_t = c_int;
const hipDevice_t = c_int;
const hipCtx_t = ?*anyopaque;
const hipModule_t = ?*anyopaque;
const hipFunction_t = ?*anyopaque;
const hipStream_t = ?*anyopaque;
const hipDeviceptr_t = ?*anyopaque;

const Api = struct {
    lib: std.DynLib,
    hipInit: *const fn (c_uint) callconv(.c) hipError_t,
    hipDeviceGet: *const fn (*hipDevice_t, c_int) callconv(.c) hipError_t,
    hipCtxCreate: *const fn (*hipCtx_t, c_uint, hipDevice_t) callconv(.c) hipError_t,
    hipCtxDestroy: *const fn (hipCtx_t) callconv(.c) hipError_t,
    // Optional: present on ROCm >= 4.2, and the fields above are the fallback.
    // An absent symbol must not sink the whole dlopen.
    hipDevicePrimaryCtxRetain: ?*const fn (*hipCtx_t, hipDevice_t) callconv(.c) hipError_t,
    hipDevicePrimaryCtxRelease: ?*const fn (hipDevice_t) callconv(.c) hipError_t,
    hipCtxSetCurrent: ?*const fn (hipCtx_t) callconv(.c) hipError_t,
    hipDeviceSynchronize: *const fn () callconv(.c) hipError_t,
    hipModuleLoadData: *const fn (*hipModule_t, *const anyopaque) callconv(.c) hipError_t,
    hipModuleUnload: *const fn (hipModule_t) callconv(.c) hipError_t,
    hipModuleGetFunction: *const fn (*hipFunction_t, hipModule_t, [*:0]const u8) callconv(.c) hipError_t,
    hipMalloc: *const fn (*hipDeviceptr_t, usize) callconv(.c) hipError_t,
    hipFree: *const fn (hipDeviceptr_t) callconv(.c) hipError_t,
    hipMemcpyHtoD: *const fn (hipDeviceptr_t, *const anyopaque, usize) callconv(.c) hipError_t,
    hipMemcpyDtoH: *const fn (*anyopaque, hipDeviceptr_t, usize) callconv(.c) hipError_t,
    hipModuleLaunchKernel: *const fn (hipFunction_t, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, hipStream_t, ?[*]iface.Arg, ?[*]iface.Arg) callconv(.c) hipError_t,
    hipMemcpyDtoD: *const fn (hipDeviceptr_t, hipDeviceptr_t, usize) callconv(.c) hipError_t,
    hipMemcpyHtoDAsync: *const fn (hipDeviceptr_t, *const anyopaque, usize, hipStream_t) callconv(.c) hipError_t,
    hipMemcpyDtoHAsync: *const fn (*anyopaque, hipDeviceptr_t, usize, hipStream_t) callconv(.c) hipError_t,
    // `int` value, not `u8`: hipMemsetAsync follows memset(3), not cuMemsetD8Async.
    hipMemsetAsync: *const fn (hipDeviceptr_t, c_int, usize, hipStream_t) callconv(.c) hipError_t,
    hipHostMalloc: *const fn (*?*anyopaque, usize, c_uint) callconv(.c) hipError_t,
    hipHostFree: *const fn (*anyopaque) callconv(.c) hipError_t,
    hipStreamCreate: *const fn (*hipStream_t, c_uint) callconv(.c) hipError_t,
    hipStreamDestroy: *const fn (hipStream_t) callconv(.c) hipError_t,
    hipStreamSynchronize: *const fn (hipStream_t) callconv(.c) hipError_t,
};

var g: Api = undefined;
var loaded = false;

/// ponytail: see cuda.zig -- one spin-and-yield lock for dlopen plus both
/// tables, taken on init-time paths only.
var lock: std.atomic.Mutex = .unlocked;

fn acquire() void {
    while (!lock.tryLock()) std.Thread.yield() catch std.atomic.spinLoopHint();
}

const lib_names = switch (builtin.os.tag) {
    .windows => &[_][]const u8{"amdhip64.dll"},
    else => &[_][]const u8{ "libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so.5" },
};

fn openFirst(names: []const []const u8) ?std.DynLib {
    for (names) |n| {
        if (std.DynLib.open(n)) |l| return l else |_| {}
    }
    return null;
}

/// Caller holds `lock`.
fn loadApiLocked() Error!void {
    if (loaded) return;
    var lib = openFirst(lib_names) orelse return error.InitFailed;
    errdefer lib.close();
    inline for (@typeInfo(Api).@"struct".fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "lib")) continue;
        // An optional field is one we can live without; anything else is fatal.
        const optional = comptime @typeInfo(field.type) == .optional;
        const Fn = comptime if (optional) @typeInfo(field.type).optional.child else field.type;
        if (lib.lookup(Fn, field.name)) |sym| {
            @field(g, field.name) = sym;
        } else if (optional) {
            @field(g, field.name) = null;
        } else return error.InitFailed;
    }
    g.lib = lib;
    loaded = true;
}

inline fn check(rc: hipError_t, err: Error) Error!void {
    if (rc != 0) {
        iface.last_driver_error = .{ .code = rc, .backend = .hip };
        return err;
    }
}

// ---- Process-wide device state ----
//
// Mirrors cuda.zig: device pointers are context-scoped, so every handle on a
// device shares one retained primary context and one JIT'd copy of the image.
//
// ponytail: fixed tables sized for one node -- 16 devices, 64 distinct images
// (one per kernel root, per device). Overflow degrades to uncached, not to wrong.
const max_devices = 16;
const max_modules = 64;

const CtxSlot = struct {
    device: hipDevice_t = 0,
    ctx: hipCtx_t = null,
    /// False when we had to fall back to `hipCtxCreate`, so teardown differs.
    primary: bool = false,
};
var ctx_slots: [max_devices]CtxSlot = @splat(.{});

/// First ordinal anyone retained, for handles that carry no ordinal of their own.
var default_ordinal: std.atomic.Value(c_int) = .init(-1);

/// Contexts are current *per thread*: a worker that never made one current has
/// none, and every launch from it fails.
threadlocal var current_ordinal: ?c_int = null;

fn retainPrimary(ordinal: c_int) Error!CtxSlot {
    if (ordinal < 0 or ordinal >= max_devices) return error.NoDevice;
    acquire();
    defer lock.unlock();
    const slot = &ctx_slots[@intCast(ordinal)];
    if (slot.ctx != null) return slot.*;

    try loadApiLocked();
    try check(g.hipInit(0), error.InitFailed);
    var found: CtxSlot = .{};
    try check(g.hipDeviceGet(&found.device, ordinal), error.NoDevice);
    if (g.hipDevicePrimaryCtxRetain) |retain| {
        try check(retain(&found.ctx, found.device), error.ContextFailed);
        found.primary = true;
    } else {
        // ponytail: pre-4.2 ROCm. One shared private context per device still
        // fixes the cross-handle buffer bug, it just is not shared with cuBLAS.
        try check(g.hipCtxCreate(&found.ctx, 0, found.device), error.ContextFailed);
    }
    slot.* = found;
    _ = default_ordinal.cmpxchgStrong(-1, ordinal, .monotonic, .monotonic);
    return found;
}

fn setCurrent(ordinal: c_int) Error!void {
    if (current_ordinal) |o| if (o == ordinal) return;
    const slot = try retainPrimary(ordinal);
    if (g.hipCtxSetCurrent) |set| try check(set(slot.ctx), error.ContextFailed);
    current_ordinal = ordinal;
}

/// Handles that carry no ordinal still need *a* context current on this thread.
inline fn ensureCurrent() void {
    if (current_ordinal != null) return;
    const d = default_ordinal.load(.monotonic);
    if (d >= 0) setCurrent(d) catch {};
}

const ModuleSlot = struct {
    ordinal: c_int = -1,
    image: []const u8 = &.{},
    module: hipModule_t = null,
};
var module_slots: [max_modules]ModuleSlot = @splat(.{});

/// ponytail: keyed on image identity (ptr + len), not contents -- artifacts come
/// from `@embedFile` and live for the process.
fn loadModuleCached(ordinal: c_int, image: []const u8) Error!Module {
    acquire();
    defer lock.unlock();
    var free_slot: ?*ModuleSlot = null;
    for (&module_slots) |*s| {
        if (s.module == null) {
            if (free_slot == null) free_slot = s;
        } else if (s.ordinal == ordinal and s.image.ptr == image.ptr and s.image.len == image.len) {
            return .{ .module = s.module, .cached = true };
        }
    }
    var m: hipModule_t = null;
    try check(g.hipModuleLoadData(&m, image.ptr), error.ModuleLoadFailed);
    const slot = free_slot orelse return .{ .module = m, .cached = false };
    slot.* = .{ .ordinal = ordinal, .image = image, .module = m };
    return .{ .module = m, .cached = true };
}

/// Unload every cached module and release every retained context. Nothing calls
/// this for you: `Context.deinit` and `Module.deinit` deliberately leave shared
/// state alone. Only call it once every handle in the process is done.
pub fn shutdown() void {
    acquire();
    defer lock.unlock();
    if (!loaded) return;
    for (&module_slots) |*s| {
        if (s.module != null) _ = g.hipModuleUnload(s.module);
        s.* = .{};
    }
    for (&ctx_slots) |*s| {
        if (s.ctx != null) {
            if (s.primary) {
                if (g.hipDevicePrimaryCtxRelease) |release| _ = release(s.device);
            } else _ = g.hipCtxDestroy(s.ctx);
        }
        s.* = .{};
    }
    current_ordinal = null;
    default_ordinal.store(-1, .monotonic);
}

pub const Context = struct {
    device: hipDevice_t = 0,
    ctx: hipCtx_t = null,
    ordinal: c_int = 0,

    /// (#3) Retains the device's primary context rather than creating a private
    /// one, so buffers allocated through one handle are valid in all the others.
    pub fn init(ordinal: c_int) Error!Context {
        const slot = try retainPrimary(ordinal);
        var self: Context = .{ .device = slot.device, .ctx = slot.ctx, .ordinal = ordinal };
        try self.makeCurrent();
        return self;
    }
    /// (#3) Releases this handle only. The context is shared and stays retained
    /// for the process; use `shutdown()` for real teardown. Safe to repeat.
    pub fn deinit(self: *Context) void {
        self.* = .{};
    }
    /// Make this device's context current on the calling thread. Idempotent and
    /// near-free after the first call on a given thread.
    pub fn makeCurrent(self: *Context) Error!void {
        return setCurrent(self.ordinal);
    }
    pub fn synchronize(self: *Context) Error!void {
        try self.makeCurrent();
        try check(g.hipDeviceSynchronize(), error.SyncFailed);
    }
    pub fn alloc(self: *Context, bytes: usize) Error!Buffer {
        try self.makeCurrent();
        var b: Buffer = .{ .bytes = bytes };
        try check(g.hipMalloc(&b.handle, bytes), error.AllocFailed);
        return b;
    }
    /// Page-locked host memory, which is what makes an async copy against it
    /// actually asynchronous: on ordinary pageable memory the runtime stages the
    /// transfer through an internal pinned buffer and blocks, so
    /// `uploadAtAsync`/`downloadAtAsync` are asynchronous in name only.
    ///
    /// Runtime memory, not the caller's allocator's -- release it with
    /// `freePinned` and nothing else.
    pub fn allocPinned(self: *Context, bytes: usize) Error![]u8 {
        try self.makeCurrent();
        var p: ?*anyopaque = null;
        // Flags 0 (hipHostMallocDefault): plain page-locked. Not
        // `WriteCombined` -- these blocks are the landing area for downloads,
        // and write-combined memory reads back at uncached speed on the host.
        try check(g.hipHostMalloc(&p, bytes, 0), error.AllocFailed);
        return @as([*]u8, @ptrCast(p.?))[0..bytes];
    }
    /// Counterpart to `allocPinned`. Like `Buffer.free`, a no-op before the
    /// runtime has ever loaded, so a hand-built slice cannot call through `g`
    /// while it is still `undefined`.
    pub fn freePinned(self: *Context, mem: []u8) void {
        if (!loaded or mem.len == 0) return;
        self.makeCurrent() catch return;
        _ = g.hipHostFree(mem.ptr);
    }
    /// (#3) Cached per (device, image): JITs once per process, not per handle.
    pub fn loadModuleFromMemory(self: *Context, image: []const u8) Error!Module {
        try self.makeCurrent();
        return loadModuleCached(self.ordinal, image);
    }
    pub fn createStream(self: *Context) Error!Stream {
        try self.makeCurrent();
        var s: Stream = .{};
        try check(g.hipStreamCreate(&s.stream, 0), error.SyncFailed);
        return s;
    }
};

pub const Buffer = struct {
    handle: hipDeviceptr_t = null,
    bytes: usize = 0,

    /// `free` nulls the handle and a default-constructed Buffer never had one.
    /// Either way it is a caller mistake, not a safety panic (or, in
    /// ReleaseFast, a copy through a dangling pointer).
    fn devicePtr(self: *const Buffer) Error!hipDeviceptr_t {
        if (self.handle == null) return error.InvalidArgument;
        return self.handle;
    }

    fn offsetPtr(self: *const Buffer, offset: usize) Error!hipDeviceptr_t {
        return @ptrFromInt(@intFromPtr(try self.devicePtr()) + offset);
    }

    pub fn upload(self: *Buffer, host: *const anyopaque, n: usize) Error!void {
        ensureCurrent();
        try check(g.hipMemcpyHtoD(try self.devicePtr(), host, n), error.CopyFailed);
    }
    pub fn download(self: *Buffer, host: *anyopaque, n: usize) Error!void {
        ensureCurrent();
        try check(g.hipMemcpyDtoH(host, try self.devicePtr(), n), error.CopyFailed);
    }
    pub fn free(self: *Buffer) void {
        // Also covers a default-constructed Buffer, where `g` is still
        // `undefined` because nothing ever dlopen'd the runtime.
        if (self.handle == null) return;
        ensureCurrent();
        _ = g.hipFree(self.handle);
        self.* = .{};
    }
    pub fn argPtr(self: *Buffer) iface.Arg {
        return @ptrCast(&self.handle);
    }
    pub fn copyFrom(self: *Buffer, src: *const Buffer, src_offset: usize, dst_offset: usize, n: usize) Error!void {
        ensureCurrent();
        try check(g.hipMemcpyDtoD(try self.offsetPtr(dst_offset), try src.offsetPtr(src_offset), n), error.CopyFailed);
    }
    pub fn downloadAt(self: *Buffer, host: *anyopaque, offset: usize, n: usize) Error!void {
        ensureCurrent();
        try check(g.hipMemcpyDtoH(host, try self.offsetPtr(offset), n), error.CopyFailed);
    }
    pub fn uploadAt(self: *Buffer, host: *const anyopaque, offset: usize, n: usize) Error!void {
        ensureCurrent();
        try check(g.hipMemcpyHtoD(try self.offsetPtr(offset), host, n), error.CopyFailed);
    }
    /// Enqueued on `stream` and not waited on; `host` must be `allocPinned`
    /// memory and must stay put until `stream.synchronize()` returns.
    pub fn downloadAtAsync(self: *Buffer, host: *anyopaque, offset: usize, n: usize, stream: *Stream) Error!void {
        ensureCurrent();
        try check(g.hipMemcpyDtoHAsync(host, try self.offsetPtr(offset), n, stream.stream), error.CopyFailed);
    }
    pub fn uploadAtAsync(self: *Buffer, host: *const anyopaque, offset: usize, n: usize, stream: *Stream) Error!void {
        ensureCurrent();
        try check(g.hipMemcpyHtoDAsync(try self.offsetPtr(offset), host, n, stream.stream), error.CopyFailed);
    }
    /// Set the first `n` bytes to `value`, on the device and on `stream`. The
    /// memory controller does it in place: zeroing this way costs no bus
    /// traffic, where copying a resident block of zeros over it does.
    pub fn fillAsync(self: *Buffer, value: u8, n: usize, stream: *Stream) Error!void {
        ensureCurrent();
        try check(g.hipMemsetAsync(try self.devicePtr(), value, n, stream.stream), error.CopyFailed);
    }
};

pub const Module = struct {
    module: hipModule_t = null,
    /// Owned by the process-wide cache; `deinit` must leave it alone.
    cached: bool = false,

    pub fn getKernel(self: *Module, name: [*:0]const u8) Error!Kernel {
        ensureCurrent();
        var k: Kernel = .{};
        try check(g.hipModuleGetFunction(&k.func, self.module, name), error.KernelNotFound);
        return k;
    }
    /// (#3) Drops this handle. A cached module stays loaded for the process;
    /// `shutdown()` unloads it.
    pub fn deinit(self: *Module) void {
        if (!self.cached and self.module != null) {
            ensureCurrent();
            _ = g.hipModuleUnload(self.module);
        }
        self.* = .{};
    }
};

pub const Kernel = struct {
    func: hipFunction_t = null,
    pub fn launch(self: Kernel, grid: Dim3, block: Dim3, shared_bytes: u32, args: []const iface.Arg) Error!void {
        try self.launchOnStream(grid, block, shared_bytes, args, null);
    }
    pub fn launchOnStream(self: Kernel, grid: Dim3, block: Dim3, shared_bytes: u32, args: []const iface.Arg, stream: hipStream_t) Error!void {
        ensureCurrent();
        try check(g.hipModuleLaunchKernel(self.func, grid.x, grid.y, grid.z, block.x, block.y, block.z, shared_bytes, stream, @constCast(args.ptr), null), error.LaunchFailed);
    }
};

pub const Stream = struct {
    stream: hipStream_t = null,
    pub fn synchronize(self: *Stream) Error!void {
        ensureCurrent();
        try check(g.hipStreamSynchronize(self.stream), error.SyncFailed);
    }
    pub fn deinit(self: *Stream) void {
        ensureCurrent();
        _ = g.hipStreamDestroy(self.stream);
        self.* = .{};
    }
};

test "a handle-less Buffer errors instead of panicking, and free is idempotent" {
    // Runs on any machine: none of these paths reach the driver, which is the
    // point -- before, each was `self.handle.?` on a null handle.
    var buffer: Buffer = .{};
    var byte: u8 = 0;
    try std.testing.expectError(error.InvalidArgument, buffer.upload(&byte, 1));
    try std.testing.expectError(error.InvalidArgument, buffer.download(&byte, 1));
    try std.testing.expectError(error.InvalidArgument, buffer.uploadAt(&byte, 4, 1));
    try std.testing.expectError(error.InvalidArgument, buffer.downloadAt(&byte, 4, 1));
    try std.testing.expectError(error.InvalidArgument, buffer.copyFrom(&buffer, 0, 0, 1));
    // Same for the stream-ordered forms: they resolve the device pointer before
    // they reach the runtime, so a null handle is an error and not a copy from
    // address `offset`.
    var stream: Stream = .{};
    try std.testing.expectError(error.InvalidArgument, buffer.uploadAtAsync(&byte, 4, 1, &stream));
    try std.testing.expectError(error.InvalidArgument, buffer.downloadAtAsync(&byte, 4, 1, &stream));
    try std.testing.expectError(error.InvalidArgument, buffer.fillAsync(0, 1, &stream));
    buffer.free();
    buffer.free();
}
