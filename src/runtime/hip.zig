//! HIP backend via runtime dlopen, mirroring cuda.zig.

const std = @import("std");
const builtin = @import("builtin");
const iface = @import("../core/interface.zig");
const DynLib = @import("dynlib.zig").DynLib;
const Dim3 = iface.Dim3;
const Error = iface.Error;
/// See `core/interface.zig`; the same type on both backends.
pub const CaptureMode = iface.CaptureMode;

const hipError_t = c_int;
const hipDevice_t = c_int;
const hipCtx_t = ?*anyopaque;
const hipModule_t = ?*anyopaque;
const hipFunction_t = ?*anyopaque;
const hipStream_t = ?*anyopaque;
const hipDeviceptr_t = ?*anyopaque;
const hipGraph_t = ?*anyopaque;
const hipGraphExec_t = ?*anyopaque;
const hipGraphNode_t = ?*anyopaque;
const hipEvent_t = ?*anyopaque;

const hipErrorNotReady: hipError_t = 600;
const hipErrorGraphExecUpdateFailure: hipError_t = 910;
const hipStreamNonBlocking: c_uint = 1;
const hipEventDisableTiming: c_uint = 2;

const Api = struct {
    lib: DynLib,
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
    hipStreamCreateWithFlags: *const fn (*hipStream_t, c_uint) callconv(.c) hipError_t,
    hipStreamQuery: *const fn (hipStream_t) callconv(.c) hipError_t,
    hipMemcpyDtoDAsync: *const fn (hipDeviceptr_t, hipDeviceptr_t, usize, hipStream_t) callconv(.c) hipError_t,
    // Graphs, optional so an older ROCm still runs plain launches
    // (`Context.hasGraphs`). The same shapes as cuda.zig's, down to
    // `hipGraphExecUpdate`'s error-node and result out-parameters.
    hipStreamBeginCapture: ?*const fn (hipStream_t, CaptureMode) callconv(.c) hipError_t,
    hipStreamEndCapture: ?*const fn (hipStream_t, *hipGraph_t) callconv(.c) hipError_t,
    hipStreamIsCapturing: ?*const fn (hipStream_t, *c_int) callconv(.c) hipError_t,
    hipGraphInstantiateWithFlags: ?*const fn (*hipGraphExec_t, hipGraph_t, c_ulonglong) callconv(.c) hipError_t,
    hipGraphExecUpdate: ?*const fn (hipGraphExec_t, hipGraph_t, *hipGraphNode_t, *c_int) callconv(.c) hipError_t,
    hipGraphLaunch: ?*const fn (hipGraphExec_t, hipStream_t) callconv(.c) hipError_t,
    hipGraphExecDestroy: ?*const fn (hipGraphExec_t) callconv(.c) hipError_t,
    hipGraphDestroy: ?*const fn (hipGraph_t) callconv(.c) hipError_t,
    // Events, optional (`Context.hasEvents`). `hipEventCreate` takes no flags;
    // `WithFlags` is the cuEventCreate twin.
    hipEventCreateWithFlags: ?*const fn (*hipEvent_t, c_uint) callconv(.c) hipError_t,
    hipEventRecord: ?*const fn (hipEvent_t, hipStream_t) callconv(.c) hipError_t,
    hipEventSynchronize: ?*const fn (hipEvent_t) callconv(.c) hipError_t,
    hipEventQuery: ?*const fn (hipEvent_t) callconv(.c) hipError_t,
    hipEventElapsedTime: ?*const fn (*f32, hipEvent_t, hipEvent_t) callconv(.c) hipError_t,
    hipEventDestroy: ?*const fn (hipEvent_t) callconv(.c) hipError_t,
    hipStreamWaitEvent: ?*const fn (hipStream_t, hipEvent_t, c_uint) callconv(.c) hipError_t,
};

const graph_fns = [_][]const u8{ "hipStreamBeginCapture", "hipStreamEndCapture", "hipStreamIsCapturing", "hipGraphInstantiateWithFlags", "hipGraphExecUpdate", "hipGraphLaunch", "hipGraphExecDestroy", "hipGraphDestroy" };
const event_fns = [_][]const u8{ "hipEventCreateWithFlags", "hipEventRecord", "hipEventSynchronize", "hipEventQuery", "hipEventElapsedTime", "hipEventDestroy", "hipStreamWaitEvent" };

/// Whether the loaded driver exports every name in `names`.
fn present(comptime names: []const []const u8) bool {
    if (!loaded) return false;
    inline for (names) |name| if (@field(g, name) == null) return false;
    return true;
}

var g: Api = undefined;
var loaded = false;

/// ponytail: see cuda.zig -- one spin-and-yield lock for dlopen plus both
/// tables, taken on init-time paths only.
var lock: std.atomic.Mutex = .unlocked;

fn acquire() void {
    while (!lock.tryLock()) std.Thread.yield() catch std.atomic.spinLoopHint();
}

const lib_names = switch (builtin.target.os.tag) {
    // The HIP SDK suffixes the DLL with its major version since 6.0.
    .windows => &[_][]const u8{ "amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll" },
    else => &[_][]const u8{ "libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so.5" },
};

fn openFirst(names: []const []const u8) ?DynLib {
    for (names) |n| {
        if (DynLib.open(n)) |l| return l else |_| {}
    }
    return null;
}

/// Caller holds `lock`.
fn loadApiLocked() Error!void {
    if (loaded) return;
    var lib = openFirst(lib_names) orelse return error.InitFailed;
    errdefer lib.close();
    const api = @typeInfo(Api).@"struct";
    inline for (api.field_names, api.field_types) |field_name, field_type| {
        if (comptime std.mem.eql(u8, field_name, "lib")) continue;
        // An optional field is one we can live without; anything else is fatal.
        const optional = comptime @typeInfo(field_type) == .optional;
        const Fn = comptime if (optional) @typeInfo(field_type).optional.child else field_type;
        if (lib.lookup(Fn, field_name)) |sym| {
            @field(g, field_name) = sym;
        } else if (optional) {
            @field(g, field_name) = null;
        } else return error.InitFailed;
    }
    g.lib = lib;
    loaded = true;
}

/// Check hipError_t, stash raw code for #7 error detail.
inline fn check(rc: hipError_t, err: Error) Error!void {
    // Records on failure AND clears on success, same as cuda.zig: assigning
    // only on failure left lastDriverError() reporting a stale code
    // indefinitely, so a caller could not tell a fresh failure from one made
    // several calls ago.
    iface.recordDriverResult(.hip, rc);
    if (rc != 0) return err;
}

/// `check` for the query calls, where "not finished yet" is an answer and not
/// a failure.
inline fn ready(rc: hipError_t, err: Error) Error!bool {
    if (rc == hipErrorNotReady) {
        iface.recordDriverResult(.hip, 0);
        return false;
    }
    try check(rc, err);
    return true;
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

/// One device. Cheap to copy; the expensive part -- the retained primary context
/// -- is process-wide and shared. Every method makes this device's context
/// current on the calling thread first, so a Context is usable from any thread.
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
    /// Allocate device memory. Caller owns the returned Buffer and must `free`
    /// it; nothing here tracks it.
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
    /// Caller owns the returned block. It is runtime memory, not the caller's
    /// allocator's -- release it with `freePinned` and nothing else.
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
    /// True when the driver has stream capture and graphs. When false, every
    /// capture and graph call returns `error.Unsupported`; fall back to plain
    /// launches.
    pub fn hasGraphs(self: *const Context) bool {
        _ = self;
        return present(&graph_fns);
    }
    /// True when the driver has events. When false, every event call returns
    /// `error.Unsupported`.
    pub fn hasEvents(self: *const Context) bool {
        _ = self;
        return present(&event_fns);
    }
    /// See `cuda.Context.fp64Ratio`. Always `error.Unsupported` for now.
    ///
    /// ponytail: HIP has no stable way to ask. The CUDA-compat attribute is not
    /// implemented on AMD, `hipDeviceAttribute_t` values moved between ROCm 5
    /// and 6, and the gfx name lives in `hipDeviceProp_t`, whose layout moved
    /// too (`hipGetDevicePropertiesR0600`). Map the gfx name to a ratio once
    /// someone has a ROCm box to check the layout on.
    pub fn fp64Ratio(self: *Context) Error!u32 {
        _ = self;
        return error.Unsupported;
    }
    /// Caller owns the returned Stream and must `deinit` it.
    pub fn createStream(self: *Context) Error!Stream {
        try self.makeCurrent();
        var s: Stream = .{};
        try check(g.hipStreamCreate(&s.stream, 0), error.SyncFailed);
        return s;
    }
    /// See `cuda.Context.createStreamNonBlocking`.
    pub fn createStreamNonBlocking(self: *Context) Error!Stream {
        try self.makeCurrent();
        var s: Stream = .{};
        try check(g.hipStreamCreateWithFlags(&s.stream, hipStreamNonBlocking), error.SyncFailed);
        return s;
    }
    /// See `cuda.Context.createEvent`.
    pub fn createEvent(self: *Context, timing: bool) Error!Event {
        try self.makeCurrent();
        var e: Event = .{};
        try check((g.hipEventCreateWithFlags orelse return error.Unsupported)(&e.event, if (timing) 0 else hipEventDisableTiming), error.EventFailed);
        return e;
    }
};

/// Device memory. All-default fields on purpose -- a hand-built or already-freed
/// Buffer has a null handle, and every method here reports that as
/// `error.InvalidArgument` rather than handing the runtime an address of `offset`.
///
/// `bytes` records the allocation size but nothing checks against it: transfers
/// are not bounds-checked in process.
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
    /// Release the device memory. Idempotent.
    pub fn free(self: *Buffer) void {
        // Also covers a default-constructed Buffer, where `g` is still
        // `undefined` because nothing ever dlopen'd the runtime.
        if (self.handle == null) return;
        ensureCurrent();
        _ = g.hipFree(self.handle);
        self.* = .{};
    }
    /// The handle as a kernel argument. Borrows `self`: the pointer is into the
    /// Buffer, which must outlive every launch it is passed to.
    pub fn argPtr(self: *Buffer) iface.Arg {
        return @ptrCast(&self.handle);
    }
    pub fn copyFrom(self: *Buffer, src: *const Buffer, src_offset: usize, dst_offset: usize, n: usize) Error!void {
        ensureCurrent();
        try check(g.hipMemcpyDtoD(try self.offsetPtr(dst_offset), try src.offsetPtr(src_offset), n), error.CopyFailed);
    }
    /// `copyFrom`, enqueued on `stream` and not waited on. Legal under capture.
    pub fn copyFromAsync(self: *Buffer, src: *const Buffer, src_offset: usize, dst_offset: usize, n: usize, stream: *Stream) Error!void {
        ensureCurrent();
        try check(g.hipMemcpyDtoDAsync(try self.offsetPtr(dst_offset), try src.offsetPtr(src_offset), n, stream.stream), error.CopyFailed);
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

/// A loaded module. `getKernel` looks entry points up by symbol name.
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

/// An entry point in a loaded module.
pub const Kernel = struct {
    func: hipFunction_t = null,
    pub fn launch(self: Kernel, grid: Dim3, block: Dim3, shared_bytes: u32, args: []const iface.Arg) Error!void {
        try self.launchOnStream(grid, block, shared_bytes, args, null);
    }
    /// Launch on `stream` and return without waiting. `args` must stay put
    /// until the launch has actually run -- `stream.synchronize()`.
    pub fn launchOnStream(self: Kernel, grid: Dim3, block: Dim3, shared_bytes: u32, args: []const iface.Arg, stream: hipStream_t) Error!void {
        ensureCurrent();
        try check(g.hipModuleLaunchKernel(self.func, grid.x, grid.y, grid.z, block.x, block.y, block.z, shared_bytes, stream, @constCast(args.ptr), null), error.LaunchFailed);
    }
};

/// A queue of ordered asynchronous work. `null` is the default stream.
pub const Stream = struct {
    stream: hipStream_t = null,
    pub fn synchronize(self: *Stream) Error!void {
        ensureCurrent();
        try check(g.hipStreamSynchronize(self.stream), error.SyncFailed);
    }
    /// See `cuda.Stream.query`.
    pub fn query(self: *Stream) Error!bool {
        ensureCurrent();
        return ready(g.hipStreamQuery(self.stream), error.SyncFailed);
    }
    /// See `cuda.Stream.waitEvent`.
    pub fn waitEvent(self: *Stream, event: *const Event) Error!void {
        ensureCurrent();
        try check((g.hipStreamWaitEvent orelse return error.Unsupported)(self.stream, event.event, 0), error.EventFailed);
    }
    /// See `cuda.Stream.beginCapture`.
    pub fn beginCapture(self: *Stream, mode: CaptureMode) Error!void {
        ensureCurrent();
        try check((g.hipStreamBeginCapture orelse return error.Unsupported)(self.stream, mode), error.CaptureFailed);
    }
    /// See `cuda.Stream.endCapture`.
    pub fn endCapture(self: *Stream) Error!Graph {
        ensureCurrent();
        var graph: Graph = .{};
        try check((g.hipStreamEndCapture orelse return error.Unsupported)(self.stream, &graph.graph), error.CaptureFailed);
        return graph;
    }
    /// See `cuda.Stream.isCapturing`.
    pub fn isCapturing(self: *Stream) Error!bool {
        ensureCurrent();
        var status: c_int = 0;
        try check((g.hipStreamIsCapturing orelse return error.Unsupported)(self.stream, &status), error.CaptureFailed);
        return status != 0;
    }
    pub fn deinit(self: *Stream) void {
        // Same reason as `Buffer.free`: `Stream{}` is a value any caller can
        // build, and `g` is undefined until something dlopen'd the runtime.
        if (!loaded) return;
        ensureCurrent();
        _ = g.hipStreamDestroy(self.stream);
        self.* = .{};
    }
};

/// See `cuda.Graph`.
pub const Graph = struct {
    graph: hipGraph_t = null,

    pub fn instantiate(self: *const Graph) Error!GraphExec {
        ensureCurrent();
        var exec: GraphExec = .{};
        try check((g.hipGraphInstantiateWithFlags orelse return error.Unsupported)(&exec.exec, self.graph, 0), error.GraphFailed);
        return exec;
    }
    pub fn deinit(self: *Graph) void {
        if (!loaded or self.graph == null) return;
        ensureCurrent();
        if (g.hipGraphDestroy) |destroy| _ = destroy(self.graph);
        self.* = .{};
    }
};

/// See `cuda.GraphExec`.
pub const GraphExec = struct {
    exec: hipGraphExec_t = null,

    pub fn launch(self: *GraphExec, stream: *Stream) Error!void {
        ensureCurrent();
        try check((g.hipGraphLaunch orelse return error.Unsupported)(self.exec, stream.stream), error.GraphFailed);
    }
    pub fn update(self: *GraphExec, graph: *const Graph) Error!bool {
        ensureCurrent();
        var node: hipGraphNode_t = null;
        var result: c_int = 0;
        const rc = (g.hipGraphExecUpdate orelse return error.Unsupported)(self.exec, graph.graph, &node, &result);
        if (rc == hipErrorGraphExecUpdateFailure) {
            iface.recordDriverResult(.hip, rc);
            return false;
        }
        try check(rc, error.GraphFailed);
        return true;
    }
    pub fn deinit(self: *GraphExec) void {
        if (!loaded or self.exec == null) return;
        ensureCurrent();
        if (g.hipGraphExecDestroy) |destroy| _ = destroy(self.exec);
        self.* = .{};
    }
};

/// See `cuda.Event`.
pub const Event = struct {
    event: hipEvent_t = null,

    pub fn record(self: *Event, stream: *Stream) Error!void {
        ensureCurrent();
        try check((g.hipEventRecord orelse return error.Unsupported)(self.event, stream.stream), error.EventFailed);
    }
    pub fn synchronize(self: *Event) Error!void {
        ensureCurrent();
        try check((g.hipEventSynchronize orelse return error.Unsupported)(self.event), error.EventFailed);
    }
    pub fn query(self: *Event) Error!bool {
        ensureCurrent();
        return ready((g.hipEventQuery orelse return error.Unsupported)(self.event), error.EventFailed);
    }
    pub fn elapsedUs(start: *const Event, end: *const Event) Error!f32 {
        ensureCurrent();
        var ms: f32 = 0;
        try check((g.hipEventElapsedTime orelse return error.Unsupported)(&ms, start.event, end.event), error.EventFailed);
        return ms * 1000;
    }
    pub fn deinit(self: *Event) void {
        if (!loaded or self.event == null) return;
        ensureCurrent();
        if (g.hipEventDestroy) |destroy| _ = destroy(self.event);
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
    // And the same again for the Stream, which used to reach hipStreamDestroy
    // through an undefined `g` on a machine with no ROCm.
    if (!loaded) stream.deinit();
}

test "a successful call clears the last driver error" {
    // hip used to assign `iface.last_driver_error` by hand and only on failure,
    // so a success after a failure left the old code standing forever --
    // core/interface.zig documents that as fixed, and it was, on cuda only.
    iface.recordDriverResult(.hip, 101);
    try check(0, error.LaunchFailed);
    try std.testing.expectEqual(iface.DriverError{}, iface.last_driver_error);

    try std.testing.expectError(error.LaunchFailed, check(101, error.LaunchFailed));
    try std.testing.expectEqual(@as(i64, 101), iface.last_driver_error.code);
    iface.recordDriverResult(.hip, 0);
}

test "the graph and event surface matches cuda.zig's, and handle-less deinit is a no-op" {
    // No AMD device has run this file. What can be checked without one: every
    // new declaration compiles, each signature is cuda.zig's with the backend
    // types swapped, and a default-built handle never reaches the runtime.
    const cuda = @import("cuda.zig");
    inline for (.{
        .{ Graph, cuda.Graph },   .{ GraphExec, cuda.GraphExec }, .{ Event, cuda.Event },
        .{ Stream, cuda.Stream }, .{ Buffer, cuda.Buffer },       .{ Context, cuda.Context },
    }) |pair| {
        std.testing.refAllDecls(pair[0]);
        inline for (.{ "instantiate", "launch", "update", "record", "synchronize", "query", "elapsedUs", "waitEvent", "beginCapture", "endCapture", "isCapturing", "copyFromAsync", "createStreamNonBlocking", "createEvent" }) |name| {
            if (@hasDecl(pair[1], name)) try std.testing.expectEqual(
                @typeInfo(@TypeOf(@field(pair[1], name))).@"fn".param_types.len,
                @typeInfo(@TypeOf(@field(pair[0], name))).@"fn".param_types.len,
            );
        }
    }
    try std.testing.expect(CaptureMode == cuda.CaptureMode);
    if (loaded) return;
    var graph: Graph = .{};
    graph.deinit();
    var exec: GraphExec = .{};
    exec.deinit();
    var event: Event = .{};
    event.deinit();
}
