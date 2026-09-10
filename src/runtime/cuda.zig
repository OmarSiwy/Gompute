//! CUDA backend via runtime dlopen.

const std = @import("std");
const builtin = @import("builtin");
const iface = @import("../core/interface.zig");
const Dim3 = iface.Dim3;
const Error = iface.Error;

const CUresult = c_int;
const CUdevice = c_int;
const CUcontext = ?*anyopaque;
const CUmodule = ?*anyopaque;
const CUfunction = ?*anyopaque;
const CUstream = ?*anyopaque;
const CUdeviceptr = c_ulonglong;

const Api = struct {
    lib: std.DynLib,
    // Core
    cuInit: *const fn (c_uint) callconv(.c) CUresult,
    cuDeviceGet: *const fn (*CUdevice, c_int) callconv(.c) CUresult,
    cuDevicePrimaryCtxRetain: *const fn (*CUcontext, CUdevice) callconv(.c) CUresult,
    cuDevicePrimaryCtxRelease_v2: *const fn (CUdevice) callconv(.c) CUresult,
    cuCtxSetCurrent: *const fn (CUcontext) callconv(.c) CUresult,
    cuCtxSynchronize: *const fn () callconv(.c) CUresult,
    // Module
    cuModuleLoadData: *const fn (*CUmodule, *const anyopaque) callconv(.c) CUresult,
    cuModuleUnload: *const fn (CUmodule) callconv(.c) CUresult,
    cuModuleGetFunction: *const fn (*CUfunction, CUmodule, [*:0]const u8) callconv(.c) CUresult,
    // Memory
    cuMemAlloc_v2: *const fn (*CUdeviceptr, usize) callconv(.c) CUresult,
    cuMemFree_v2: *const fn (CUdeviceptr) callconv(.c) CUresult,
    cuMemcpyHtoD_v2: *const fn (CUdeviceptr, *const anyopaque, usize) callconv(.c) CUresult,
    cuMemcpyDtoH_v2: *const fn (*anyopaque, CUdeviceptr, usize) callconv(.c) CUresult,
    cuMemcpyDtoD_v2: *const fn (CUdeviceptr, CUdeviceptr, usize) callconv(.c) CUresult,
    cuMemcpyHtoDAsync_v2: *const fn (CUdeviceptr, *const anyopaque, usize, CUstream) callconv(.c) CUresult,
    cuMemcpyDtoHAsync_v2: *const fn (*anyopaque, CUdeviceptr, usize, CUstream) callconv(.c) CUresult,
    cuMemsetD8Async: *const fn (CUdeviceptr, u8, usize, CUstream) callconv(.c) CUresult,
    // Page-locked host memory. Neither is `_v2`-remapped in cuda.h, unlike the
    // copies above -- `cuMemAllocHost` is the one that got a version suffix.
    cuMemHostAlloc: *const fn (*?*anyopaque, usize, c_uint) callconv(.c) CUresult,
    cuMemFreeHost: *const fn (*anyopaque) callconv(.c) CUresult,
    // Launch
    cuLaunchKernel: *const fn (CUfunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, CUstream, ?[*]iface.Arg, ?[*]iface.Arg) callconv(.c) CUresult,
    // Streams
    cuStreamCreate: *const fn (*CUstream, c_uint) callconv(.c) CUresult,
    cuStreamDestroy_v2: *const fn (CUstream) callconv(.c) CUresult,
    cuStreamSynchronize: *const fn (CUstream) callconv(.c) CUresult,
};

var g: Api = undefined;
var loaded = false;

/// ponytail: one global lock covers the dlopen, the context table and the
/// module table. All three are init-time paths; launches never take it. Split
/// it only if someone profiles `Kernel.init` storms as contended.
///
/// ponytail: spin-and-yield -- 0.16 dropped `std.Thread.Mutex` and `std.Io.Mutex`
/// wants an `Io` this layer does not have. Swap it in if one ever reaches here.
var lock: std.atomic.Mutex = .unlocked;

fn acquire() void {
    while (!lock.tryLock()) std.Thread.yield() catch std.atomic.spinLoopHint();
}

const lib_names = switch (builtin.os.tag) {
    .windows => &[_][]const u8{"nvcuda.dll"},
    else => &[_][]const u8{
        "libcuda.so",                          "libcuda.so.1",
        "/run/opengl-driver/lib/libcuda.so.1", "/run/opengl-driver/lib/libcuda.so",
    },
};

/// The candidate list as one literal, so the debug line costs no formatting.
const tried_paths = blk: {
    var s: []const u8 = "";
    for (lib_names, 0..) |n, i| s = s ++ (if (i == 0) "" else ", ") ++ n;
    break :blk s;
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
    // "No NVIDIA driver here" is an ordinary outcome -- AutoKernel probes CUDA
    // on every machine -- so this is debug, not a print to stderr on the normal
    // path of a non-NVIDIA box. The caller gets error.InitFailed either way.
    var lib = openFirst(lib_names) orelse {
        std.log.debug("gompute: no CUDA driver; tried " ++ tried_paths, .{});
        return error.InitFailed;
    };
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
        } else {
            // Opening the driver and then failing to resolve a symbol out of it
            // is almost never a driver problem: without libc, Zig 0.16's
            // std.DynLib is ElfDynLib, which opens libcuda.so but cannot
            // resolve from it. Stays loud -- the bare old message
            // ("symbol not found: cuInit") sent people hunting a version
            // mismatch that does not exist.
            std.log.err(
                "gompute: opened the CUDA driver but symbol '{s}' is missing.\n" ++
                    "  This almost always means the executable was not linked against libc.\n" ++
                    "  Add to your build.zig:  exe.root_module.linkSystemLibrary(\"c\", .{{}});",
                .{field.name},
            );
            return error.InitFailed;
        }
    }
    g.lib = lib;
    loaded = true;
}

// ---- Process-wide device state ----
//
// CUDA device pointers are context-scoped, so every handle on a device has to
// share one context or buffers silently do not travel between them. We retain
// the device's *primary* context: refcounted by the driver, and the same one
// the CUDA runtime API and libraries like cuBLAS use, so we interoperate.
//
// ponytail: fixed tables sized for one node -- 16 devices, 64 distinct module
// images. Overflow degrades to uncached, not to wrong. Swap in a hash map only
// if someone ships a box past that ceiling.
//
// 64 rather than 16 because one artifact per kernel root means the image count
// is now the root count times the devices in use, not 1: a 37-model consumer on
// one GPU used to blow a 16-slot table and re-JIT on every handle.
const max_devices = 16;
const max_modules = 64;

const CtxSlot = struct { device: CUdevice = 0, ctx: CUcontext = null };
var ctx_slots: [max_devices]CtxSlot = @splat(.{});

/// First ordinal anyone retained, for handles that carry no ordinal of their own.
var default_ordinal: std.atomic.Value(c_int) = .init(-1);

/// A CUDA context is current *per thread*: a worker that never called
/// `cuCtxSetCurrent` has none, and every launch from it fails. Cheap after the
/// first call on a thread.
threadlocal var current_ordinal: ?c_int = null;

fn retainPrimary(ordinal: c_int) Error!CtxSlot {
    if (ordinal < 0 or ordinal >= max_devices) return error.NoDevice;
    acquire();
    defer lock.unlock();
    const slot = &ctx_slots[@intCast(ordinal)];
    if (slot.ctx != null) return slot.*;

    try loadApiLocked();
    try check(g.cuInit(0), error.InitFailed);
    var found: CtxSlot = .{};
    try check(g.cuDeviceGet(&found.device, ordinal), error.NoDevice);
    try check(g.cuDevicePrimaryCtxRetain(&found.ctx, found.device), error.ContextFailed);
    slot.* = found;
    _ = default_ordinal.cmpxchgStrong(-1, ordinal, .monotonic, .monotonic);
    return found;
}

fn setCurrent(ordinal: c_int) Error!void {
    if (current_ordinal) |o| if (o == ordinal) return;
    const slot = try retainPrimary(ordinal);
    try check(g.cuCtxSetCurrent(slot.ctx), error.ContextFailed);
    current_ordinal = ordinal;
}

/// Handles that carry no ordinal (`Buffer`, `Module`, `Kernel`, `Stream`) still
/// need *a* context current on this thread; adopt the first device retained.
inline fn ensureCurrent() void {
    if (current_ordinal != null) return;
    const d = default_ordinal.load(.monotonic);
    if (d >= 0) setCurrent(d) catch {};
}

const ModuleSlot = struct {
    ordinal: c_int = -1,
    image: []const u8 = &.{},
    module: CUmodule = null,
};
var module_slots: [max_modules]ModuleSlot = @splat(.{});

/// JIT the image once per (device, image) instead of once per kernel handle --
/// the artifact is one blob holding every kernel, so N handles used to mean N
/// compiles of the whole thing.
///
/// ponytail: keyed on image identity (ptr + len), not contents. Artifacts come
/// from `@embedFile` and live for the process; hash the bytes instead if anyone
/// ever loads a module from a buffer they then free and reuse.
fn loadModuleCached(ordinal: c_int, image: [:0]const u8) Error!Module {
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

    // cuModuleLoadData requires null-terminated PTX, and the sentinel is now in
    // the parameter type -- so there is nothing to probe and nothing to copy.
    // Probing it as `image.ptr[image.len]` read one byte past the slice on every
    // call, which segfaults when the image ends exactly on a guard page.
    var m: CUmodule = null;
    try check(g.cuModuleLoadData(&m, image.ptr), error.ModuleLoadFailed);

    const slot = free_slot orelse return .{ .module = m, .cached = false };
    slot.* = .{ .ordinal = ordinal, .image = image, .module = m };
    return .{ .module = m, .cached = true };
}

/// How many distinct (device, image) modules are currently JIT'd.
///
/// With one artifact per kernel root, this is the count of roots this process
/// has actually paid to compile -- the thing a consumer with 37 device models
/// wants to stay at 1. ponytail: CUDA only; add the HIP twin when someone has
/// the hardware to check it on.
pub fn loadedModuleCount() usize {
    acquire();
    defer lock.unlock();
    var n: usize = 0;
    for (&module_slots) |*s| {
        if (s.module != null) n += 1;
    }
    return n;
}

/// Unload every cached module and release every retained primary context.
/// Nothing calls this for you: `Context.deinit` and `Module.deinit` deliberately
/// leave shared state alone, so this is the only real teardown. Only call it
/// once every handle in the process is done.
pub fn shutdown() void {
    acquire();
    defer lock.unlock();
    if (!loaded) return;
    for (&module_slots) |*s| {
        if (s.module != null) _ = g.cuModuleUnload(s.module);
        s.* = .{};
    }
    for (&ctx_slots) |*s| {
        if (s.ctx != null) _ = g.cuDevicePrimaryCtxRelease_v2(s.device);
        s.* = .{};
    }
    current_ordinal = null;
    default_ordinal.store(-1, .monotonic);
}

/// Check CUresult, stash raw code for #7 error detail.
inline fn check(rc: CUresult, err: Error) Error!void {
    // Records on failure AND clears on success -- assigning only on failure left
    // lastDriverError() reporting a stale code indefinitely, so a caller could
    // not tell a fresh failure from one several calls ago.
    iface.recordDriverResult(.cuda, rc);
    if (rc != 0) return err;
}

// ---- Public API ----

pub const Context = struct {
    device: CUdevice = 0,
    ctx: CUcontext = null,
    ordinal: c_int = 0,

    /// (#6) Init with any device ordinal, not just 0.
    ///
    /// (#3) Retains the device's *primary* context rather than creating a
    /// private one, so every handle on a device shares it and buffers allocated
    /// through one are valid in all the others. Also makes it current on the
    /// calling thread.
    pub fn init(ordinal: c_int) Error!Context {
        const slot = try retainPrimary(ordinal);
        var self: Context = .{ .device = slot.device, .ctx = slot.ctx, .ordinal = ordinal };
        try self.makeCurrent();
        return self;
    }

    /// (#3) Releases this handle only. The primary context is shared and stays
    /// retained for the process; use `shutdown()` for real teardown. Safe to
    /// call any number of times.
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
        try check(g.cuCtxSynchronize(), error.SyncFailed);
    }

    pub fn createStream(self: *Context) Error!Stream {
        try self.makeCurrent();
        var s: Stream = .{};
        try check(g.cuStreamCreate(&s.stream, 0), error.SyncFailed);
        return s;
    }
    pub fn alloc(self: *Context, bytes: usize) Error!Buffer {
        try self.makeCurrent();
        var b: Buffer = .{ .bytes = bytes };
        try check(g.cuMemAlloc_v2(&b.handle, bytes), error.AllocFailed);
        return b;
    }

    /// Page-locked host memory, which is what makes an async copy against it
    /// actually asynchronous: on ordinary pageable memory the driver stages the
    /// transfer through an internal pinned buffer and blocks, so
    /// `uploadAtAsync`/`downloadAtAsync` are asynchronous in name only.
    ///
    /// Driver memory, not the caller's allocator's -- release it with
    /// `freePinned` and nothing else.
    pub fn allocPinned(self: *Context, bytes: usize) Error![]u8 {
        try self.makeCurrent();
        var p: ?*anyopaque = null;
        // Flags 0: plain page-locked. Not WRITECOMBINED -- these blocks are the
        // landing area for downloads, and write-combined memory reads back at
        // uncached speed on the host.
        try check(g.cuMemHostAlloc(&p, bytes, 0), error.AllocFailed);
        return @as([*]u8, @ptrCast(p.?))[0..bytes];
    }

    /// Counterpart to `allocPinned`. Like `Buffer.free`, a no-op before the
    /// driver has ever loaded, so a hand-built slice cannot call through `g`
    /// while it is still `undefined`.
    pub fn freePinned(self: *Context, mem: []u8) void {
        if (!loaded or mem.len == 0) return;
        self.makeCurrent() catch return;
        _ = g.cuMemFreeHost(mem.ptr);
    }
    /// (#3) Cached per (device, image): the artifact holds every kernel, so this
    /// JITs once per process instead of once per handle.
    pub fn loadModuleFromMemory(self: *Context, image: [:0]const u8) Error!Module {
        try self.makeCurrent();
        return loadModuleCached(self.ordinal, image);
    }
};

pub const Buffer = struct {
    handle: CUdeviceptr = 0,
    bytes: usize = 0,

    pub fn upload(self: *Buffer, host: *const anyopaque, n: usize) Error!void {
        ensureCurrent();
        try check(g.cuMemcpyHtoD_v2(self.handle, host, n), error.CopyFailed);
    }
    pub fn download(self: *Buffer, host: *anyopaque, n: usize) Error!void {
        ensureCurrent();
        try check(g.cuMemcpyDtoH_v2(host, self.handle, n), error.CopyFailed);
    }
    pub fn downloadAt(self: *Buffer, host: *anyopaque, offset: usize, n: usize) Error!void {
        ensureCurrent();
        try check(g.cuMemcpyDtoH_v2(host, self.handle + offset, n), error.CopyFailed);
    }
    pub fn uploadAt(self: *Buffer, host: *const anyopaque, offset: usize, n: usize) Error!void {
        ensureCurrent();
        try check(g.cuMemcpyHtoD_v2(self.handle + offset, host, n), error.CopyFailed);
    }
    pub fn free(self: *Buffer) void {
        // `g` is undefined until loadApi succeeds, and every handle type here is
        // pub with all-default fields -- so a hand-constructed `Buffer{}` freed
        // on a machine with no driver would call through garbage.
        if (!loaded) return;
        ensureCurrent();
        _ = g.cuMemFree_v2(self.handle);
        self.* = .{};
    }
    pub fn argPtr(self: *Buffer) iface.Arg {
        return @ptrCast(&self.handle);
    }
    pub fn copyFrom(self: *Buffer, src: *const Buffer, src_offset: usize, dst_offset: usize, n: usize) Error!void {
        ensureCurrent();
        try check(g.cuMemcpyDtoD_v2(self.handle + dst_offset, src.handle + src_offset, n), error.CopyFailed);
    }
    /// Enqueued on `stream` and not waited on; `host` must be `allocPinned`
    /// memory and must stay put until `stream.synchronize()` returns.
    pub fn downloadAtAsync(self: *Buffer, host: *anyopaque, offset: usize, n: usize, stream: *Stream) Error!void {
        ensureCurrent();
        try check(g.cuMemcpyDtoHAsync_v2(host, self.handle + offset, n, stream.stream), error.CopyFailed);
    }
    pub fn uploadAtAsync(self: *Buffer, host: *const anyopaque, offset: usize, n: usize, stream: *Stream) Error!void {
        ensureCurrent();
        try check(g.cuMemcpyHtoDAsync_v2(self.handle + offset, host, n, stream.stream), error.CopyFailed);
    }
    /// Set the first `n` bytes to `value`, on the device and on `stream`. The
    /// memory controller does it in place: zeroing this way costs no bus
    /// traffic, where copying a resident block of zeros over it does.
    pub fn fillAsync(self: *Buffer, value: u8, n: usize, stream: *Stream) Error!void {
        ensureCurrent();
        try check(g.cuMemsetD8Async(self.handle, value, n, stream.stream), error.CopyFailed);
    }

    pub fn deviceAddr(self: *const Buffer) u64 {
        return self.handle;
    }
};

pub const Module = struct {
    module: CUmodule = null,
    /// Owned by the process-wide cache; `deinit` must leave it alone.
    cached: bool = false,

    pub fn getKernel(self: *Module, name: [*:0]const u8) Error!Kernel {
        ensureCurrent();
        var k: Kernel = .{};
        try check(g.cuModuleGetFunction(&k.func, self.module, name), error.KernelNotFound);
        return k;
    }
    /// (#3) Drops this handle. A cached module is shared with every other handle
    /// on the device and stays loaded for the process; `shutdown()` unloads it.
    pub fn deinit(self: *Module) void {
        if (!self.cached and self.module != null) {
            ensureCurrent();
            _ = g.cuModuleUnload(self.module);
        }
        self.* = .{};
    }
};

pub const Kernel = struct {
    func: CUfunction = null,
    pub fn launch(self: Kernel, grid: Dim3, block: Dim3, shared_bytes: u32, args: []const iface.Arg) Error!void {
        try self.launchOnStream(grid, block, shared_bytes, args, null);
    }
    pub fn launchOnStream(self: Kernel, grid: Dim3, block: Dim3, shared_bytes: u32, args: []const iface.Arg, stream: CUstream) Error!void {
        ensureCurrent();
        try check(g.cuLaunchKernel(self.func, grid.x, grid.y, grid.z, block.x, block.y, block.z, shared_bytes, stream, @constCast(args.ptr), null), error.LaunchFailed);
    }
};

pub const Stream = struct {
    stream: CUstream = null,
    pub fn synchronize(self: *Stream) Error!void {
        ensureCurrent();
        try check(g.cuStreamSynchronize(self.stream), error.SyncFailed);
    }
    pub fn deinit(self: *Stream) void {
        if (!loaded) return;
        ensureCurrent();
        _ = g.cuStreamDestroy_v2(self.stream);
        self.* = .{};
    }
};

test loadedModuleCount {
    // Nothing in this suite JITs anything, and a fresh process starts empty --
    // this is the number a consumer with 37 device models wants to stay at 1.
    try std.testing.expectEqual(@as(usize, 0), loadedModuleCount());
}

test "hand-built handles do not call through an undefined driver" {
    // `g` is undefined until loadApiLocked succeeds, and every handle type is
    // pub with all-default fields, so `Buffer{}` and `Stream{}` are values a
    // caller can make on a machine with no driver at all.
    if (loaded) return error.SkipZigTest; // this box has one; nothing to prove
    var b: Buffer = .{};
    b.free();
    b.free();
    var s: Stream = .{};
    s.deinit();
}
