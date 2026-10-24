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
const hipErrorPeerAccessAlreadyEnabled: hipError_t = 704;
const hipErrorGraphExecUpdateFailure: hipError_t = 910;
const hipStreamNonBlocking: c_uint = 1;
const hipEventDisableTiming: c_uint = 2;

/// `hipDeviceProp_tR0600`: the device-properties layout ROCm 6.0 froze, and
/// the one `hipGetDevicePropertiesR0600` fills, so a binary built against one
/// header keeps working on later runtimes. It is also the only stable way to
/// ask: `hipDeviceAttribute_t` was renumbered in ROCm 6.0, so an attribute
/// number cannot be hardcoded across versions. Named up to the last field
/// read; the runs nothing here reads are opaque padding. Measured against
/// ROCm 7.2.3's `hip_runtime_api.h` on x86_64 Linux: 1472 bytes, 8-aligned,
/// at the offsets the comptime block below pins.
const DeviceProp = extern struct {
    name: [256]u8,
    uuid: [16]u8,
    luid: [8]u8,
    luidDeviceNodeMask: c_uint,
    _pad0: [4]u8,
    totalGlobalMem: usize,
    sharedMemPerBlock: usize,
    regsPerBlock: c_int,
    warpSize: c_int,
    memPitch: usize,
    maxThreadsPerBlock: c_int,
    maxThreadsDim: [3]c_int,
    maxGridSize: [3]c_int,
    /// kHz.
    clockRate: c_int,
    totalConstMem: usize,
    major: c_int,
    minor: c_int,
    textureAlignment: usize,
    texturePitchAlignment: usize,
    deviceOverlap: c_int,
    multiProcessorCount: c_int,
    kernelExecTimeoutEnabled: c_int,
    integrated: c_int,
    canMapHostMemory: c_int,
    computeMode: c_int,
    _pad1: [176]u8,
    pciBusID: c_int,
    pciDeviceID: c_int,
    pciDomainID: c_int,
    _pad2: [8]u8,
    unifiedAddressing: c_int,
    /// kHz.
    memoryClockRate: c_int,
    /// Bits.
    memoryBusWidth: c_int,
    l2CacheSize: c_int,
    _pad3: [4]u8,
    maxThreadsPerMultiProcessor: c_int,
    _pad4: [532]u8,
    /// NUL-terminated, e.g. "gfx1100:sramecc+:xnack-".
    gcnArchName: [256]u8,
    _pad5: [56]u8,
};

comptime {
    if (@sizeOf(DeviceProp) != 1472 or @alignOf(DeviceProp) != 8) @compileError("DeviceProp is not the 1472-byte, 8-aligned hipDeviceProp_tR0600");
    for (.{
        .{ "name", 0 },                 .{ "uuid", 256 },                        .{ "luid", 272 },
        .{ "luidDeviceNodeMask", 280 }, .{ "totalGlobalMem", 288 },              .{ "sharedMemPerBlock", 296 },
        .{ "regsPerBlock", 304 },       .{ "warpSize", 308 },                    .{ "memPitch", 312 },
        .{ "maxThreadsPerBlock", 320 }, .{ "maxThreadsDim", 324 },               .{ "maxGridSize", 336 },
        .{ "clockRate", 348 },          .{ "totalConstMem", 352 },               .{ "major", 360 },
        .{ "minor", 364 },              .{ "textureAlignment", 368 },            .{ "texturePitchAlignment", 376 },
        .{ "deviceOverlap", 384 },      .{ "multiProcessorCount", 388 },         .{ "kernelExecTimeoutEnabled", 392 },
        .{ "integrated", 396 },         .{ "canMapHostMemory", 400 },            .{ "computeMode", 404 },
        .{ "pciBusID", 584 },           .{ "pciDeviceID", 588 },                 .{ "pciDomainID", 592 },
        .{ "unifiedAddressing", 604 },  .{ "memoryClockRate", 608 },             .{ "memoryBusWidth", 612 },
        .{ "l2CacheSize", 616 },        .{ "maxThreadsPerMultiProcessor", 624 }, .{ "gcnArchName", 1160 },
    }) |f| if (@offsetOf(DeviceProp, f[0]) != f[1]) @compileError("DeviceProp." ++ f[0] ++ " is not at its hipDeviceProp_tR0600 offset");
}

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
    // Peer to peer (HIP 1.x): `Buffer.copyFrom` across devices and
    // `Context.enablePeerAccess`. Device ids, where CUDA takes contexts.
    hipDeviceCanAccessPeer: *const fn (*c_int, c_int, c_int) callconv(.c) hipError_t,
    hipDeviceEnablePeerAccess: *const fn (c_int, c_uint) callconv(.c) hipError_t,
    hipMemcpyPeer: *const fn (hipDeviceptr_t, c_int, hipDeviceptr_t, c_int, usize) callconv(.c) hipError_t,
    hipMemcpyPeerAsync: *const fn (hipDeviceptr_t, c_int, hipDeviceptr_t, c_int, usize, hipStream_t) callconv(.c) hipError_t,
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
    // Device properties, optional: the versioned call is ROCm 6.0+
    // (`Context.properties`), `hipDeviceComputeCapability` older than that.
    hipGetDevicePropertiesR0600: ?*const fn (*DeviceProp, c_int) callconv(.c) hipError_t,
    hipDeviceComputeCapability: ?*const fn (*c_int, *c_int, hipDevice_t) callconv(.c) hipError_t,
    // Device enumeration and identity, optional (`deviceCount`, `Context.info`).
    hipGetDeviceCount: ?*const fn (*c_int) callconv(.c) hipError_t,
    hipDeviceGetName: ?*const fn ([*]u8, c_int, hipDevice_t) callconv(.c) hipError_t,
    hipDeviceGetUuid: ?*const fn (*[16]u8, hipDevice_t) callconv(.c) hipError_t,
    hipDeviceGetPCIBusId: ?*const fn ([*]u8, c_int, c_int) callconv(.c) hipError_t,
    hipDeviceTotalMem: ?*const fn (*usize, hipDevice_t) callconv(.c) hipError_t,
    hipMemGetInfo: ?*const fn (*usize, *usize) callconv(.c) hipError_t,
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

/// Check a hipError_t, recording the raw code for `lastDriverError`.
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
// ponytail: as cuda.zig -- 16 devices, a module list that grows; a failed
// growth degrades to uncached, not to wrong.
const max_devices = 16;

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
    if (ordinal >= try realCountLocked()) {
        if (ordinal >= fakeCount()) return error.NoDevice;
        // A fake device (see `fakeCount`): a private context on device 0,
        // which hipCtxCreate makes current behind `current_ordinal`'s back.
        try check(g.hipDeviceGet(&found.device, 0), error.NoDevice);
        try check(g.hipCtxCreate(&found.ctx, 0, found.device), error.ContextFailed);
        current_ordinal = null;
    } else if (g.hipDevicePrimaryCtxRetain) |retain| {
        try check(g.hipDeviceGet(&found.device, ordinal), error.NoDevice);
        try check(retain(&found.ctx, found.device), error.ContextFailed);
        found.primary = true;
    } else {
        // ponytail: pre-4.2 ROCm. One shared private context per device still
        // fixes the cross-handle buffer bug, it just is not shared with cuBLAS.
        try check(g.hipDeviceGet(&found.device, ordinal), error.NoDevice);
        try check(g.hipCtxCreate(&found.ctx, 0, found.device), error.ContextFailed);
    }
    slot.* = found;
    _ = default_ordinal.cmpxchgStrong(-1, ordinal, .monotonic, .monotonic);
    return found;
}

/// See cuda.zig's: the runtime's device count, -1 until asked. Caller holds
/// `lock` and has loaded the API. 1 on a runtime without `hipGetDeviceCount`.
var real_count: c_int = -1;

fn realCountLocked() Error!c_int {
    if (real_count < 0) {
        var n: c_int = 1;
        if (g.hipGetDeviceCount) |count| try check(count(&n), error.NoDevice);
        real_count = n;
    }
    return real_count;
}

/// See cuda.zig's `fakeCount`: `GOMPUTE_FAKE_DEVICES`, a test knob.
var fake_devices: ?c_int = null;

fn fakeCount() c_int {
    if (fake_devices) |k| return k;
    const v = std.c.getenv("GOMPUTE_FAKE_DEVICES") orelse "";
    const k = std.fmt.parseInt(c_int, std.mem.sliceTo(v, 0), 10) catch 0;
    fake_devices = std.math.clamp(k, 0, max_devices);
    return fake_devices.?;
}

/// See `cuda.deviceCount`.
pub fn deviceCount() Error!u32 {
    acquire();
    defer lock.unlock();
    try loadApiLocked();
    try check(g.hipInit(0), error.InitFailed);
    const real = try realCountLocked();
    if (real == 0) return 0;
    return @intCast(@max(real, fakeCount()));
}

/// See cuda.zig's `use`.
inline fn use(ordinal: c_int) void {
    if (ordinal < 0) return ensureCurrent();
    setCurrent(ordinal) catch {};
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

/// The device of a live handle's ordinal, which the peer calls take where
/// CUDA takes a context. No lock: a slot is written once, under `lock`, before
/// any handle with that ordinal exists, and cleared only by `shutdown`.
inline fn deviceOf(ordinal: c_int) hipDevice_t {
    return ctx_slots[@intCast(ordinal)].device;
}

/// Two handles on different devices. Hand-built handles (ordinal -1) adopt the
/// default device, so they never count as a second one.
inline fn crossDevice(a: c_int, b: c_int) bool {
    return a != b and a >= 0 and b >= 0;
}

const ModuleSlot = struct {
    ordinal: c_int = -1,
    image: []const u8 = &.{},
    module: hipModule_t = null,
};
var module_list: std.ArrayList(ModuleSlot) = .empty;

/// ponytail: keyed on image identity (ptr + len), not contents -- artifacts come
/// from `@embedFile` and live for the process.
fn loadModuleCached(ordinal: c_int, image: []const u8) Error!Module {
    acquire();
    defer lock.unlock();
    for (module_list.items) |*s| {
        if (s.ordinal == ordinal and s.image.ptr == image.ptr and s.image.len == image.len) {
            return .{ .module = s.module, .cached = true, .ordinal = ordinal };
        }
    }
    var m: hipModule_t = null;
    try check(g.hipModuleLoadData(&m, image.ptr), error.ModuleLoadFailed);
    module_list.append(std.heap.page_allocator, .{ .ordinal = ordinal, .image = image, .module = m }) catch return .{ .module = m, .cached = false, .ordinal = ordinal };
    return .{ .module = m, .cached = true, .ordinal = ordinal };
}

/// See `cuda.loadedModuleCount`: how many distinct (device, image) modules
/// are currently loaded.
pub fn loadedModuleCount() usize {
    acquire();
    defer lock.unlock();
    return module_list.items.len;
}

/// Unload every cached module and release every retained context. Nothing calls
/// this for you: `Context.deinit` and `Module.deinit` deliberately leave shared
/// state alone. Only call it once every handle in the process is done.
pub fn shutdown() void {
    acquire();
    defer lock.unlock();
    if (!loaded) return;
    for (module_list.items) |*s| _ = g.hipModuleUnload(s.module);
    module_list.clearAndFree(std.heap.page_allocator);
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
    real_count = -1;
}

/// A `c_int` the runtime documents as non-negative, as the `u32` DeviceInfo
/// stores it; a negative one, which would be a runtime bug, clamps to 0
/// instead of tripping `@intCast`.
inline fn nonNeg(v: c_int) u32 {
    return @intCast(@max(v, 0));
}

/// A gfx target as `parseGfx` reads it off a `gcnArchName`.
const Gfx = struct { major: u32, minor: u8, stepping: u8 };

/// The target in a `gcnArchName`: "gfx" + digits, then an optional ":feature"
/// list. The last two digits are the hex minor and stepping and the rest is
/// the decimal major, so "gfx90a:sramecc+:xnack-" is 9, 0, 0xa and "gfx1100"
/// is 11, 0, 0. Stops at the first NUL too, so the struct's fixed-width field
/// passes as is. Null when the name does not parse.
fn parseGfx(name: []const u8) ?Gfx {
    const end = std.mem.findAny(u8, name, &.{ ':', 0 }) orelse name.len;
    const arch = name[0..end];
    if (!std.mem.startsWith(u8, arch, "gfx") or arch.len < 6) return null;
    const digits = arch[3..];
    return .{
        .major = std.fmt.parseInt(u32, digits[0 .. digits.len - 2], 10) catch return null,
        .minor = std.fmt.charToDigit(digits[digits.len - 2], 16) catch return null,
        .stepping = std.fmt.charToDigit(digits[digits.len - 1], 16) catch return null,
    };
}

/// The vendor's peak vector FP32:FP64 ratio of a gfx family: 2 on the MI50
/// and MI60 (gfx906), the MI100 (gfx908), the MI300 (gfx94x) and the MI350
/// (gfx95x); 1 on the MI200 (gfx90a); 16 on every other Vega and on all of
/// RDNA (gfx10 to gfx12). Null for a family outside the table, which is
/// everything before Vega. Datasheet figures, unverified on hardware.
fn ratioOf(major: u32, minor: u8, stepping: u8) ?u32 {
    return switch (major) {
        9 => switch (minor) {
            0 => switch (stepping) {
                6, 8 => 2,
                0xa => 1,
                else => 16,
            },
            4, 5 => 2,
            else => 16,
        },
        10, 11, 12 => 16,
        else => null,
    };
}

/// `ratioOf` for a `gcnArchName`; null when the name does not parse or the
/// family is outside the table.
fn archRatio(name: []const u8) ?u32 {
    const gfx = parseGfx(name) orelse return null;
    return ratioOf(gfx.major, gfx.minor, gfx.stepping);
}

/// One device. Cheap to copy; the expensive part -- the retained primary context
/// -- is process-wide and shared. Every method makes this device's context
/// current on the calling thread first, so a Context is usable from any thread.
pub const Context = struct {
    device: hipDevice_t = 0,
    ctx: hipCtx_t = null,
    ordinal: c_int = 0,

    /// Retains the device's primary context rather than creating a private
    /// one, so buffers allocated through one handle are valid in all the others.
    pub fn init(ordinal: c_int) Error!Context {
        const slot = try retainPrimary(ordinal);
        var self: Context = .{ .device = slot.device, .ctx = slot.ctx, .ordinal = ordinal };
        try self.makeCurrent();
        return self;
    }
    /// Releases this handle only. The context is shared and stays retained
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
        var b: Buffer = .{ .bytes = bytes, .ordinal = self.ordinal };
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
    /// Cached per (device, image): JITs once per process, not per handle.
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
    /// See `cuda.Context.fp64Ratio`: 16 on an RX 7900 XTX (gfx1100), 2 on an
    /// MI300X (gfx942), 1 on an MI250X (gfx90a). HIP has no attribute for it,
    /// so this is `ratioOf` applied to the device's `gcnArchName`: the
    /// vendor's datasheet figure, not a measurement. `error.Unsupported`
    /// before ROCm 6.0 and for a gfx family the table does not know.
    pub fn fp64Ratio(self: *Context) Error!u32 {
        try self.makeCurrent();
        const p = try self.properties();
        return archRatio(&p.gcnArchName) orelse error.Unsupported;
    }
    /// See `cuda.Context.computeCapability`, except that on HIP it is the gfx
    /// major and minor: 110 on a gfx1100, 94 on a gfx942, a number no `sm_`
    /// target shares. `error.Unsupported` on a runtime with neither
    /// `hipDeviceComputeCapability` nor the ROCm 6.0 properties call.
    pub fn computeCapability(self: *Context) Error!u32 {
        try self.makeCurrent();
        var major: c_int = 0;
        var minor: c_int = 0;
        if (g.hipDeviceComputeCapability) |get| {
            try check(get(&major, &minor, self.device), error.InvalidArgument);
        } else {
            const p = try self.properties();
            major = p.major;
            minor = p.minor;
        }
        return nonNeg(major * 10 + minor);
    }
    /// See `cuda.Context.canAccessPeer`. False for the same device, which
    /// includes a `GOMPUTE_FAKE_DEVICES` context.
    pub fn canAccessPeer(self: *const Context, peer: *const Context) Error!bool {
        var v: c_int = 0;
        try check(g.hipDeviceCanAccessPeer(&v, self.device, peer.device), error.InvalidArgument);
        return v != 0;
    }
    /// See `cuda.Context.enablePeerAccess`. HIP grants it to the *current*
    /// device, so `self` is made current first. `error.Unsupported` when
    /// `canAccessPeer` is false, asked here because the runtime reports that
    /// case as an invalid device, not as unsupported; cross-device copies
    /// still work, staged through the host.
    pub fn enablePeerAccess(self: *Context, peer: *const Context) Error!void {
        try self.makeCurrent();
        if (!try self.canAccessPeer(peer)) return error.Unsupported;
        const rc = g.hipDeviceEnablePeerAccess(peer.device, 0);
        if (rc == hipErrorPeerAccessAlreadyEnabled) return iface.recordDriverResult(.hip, 0);
        try check(rc, error.ContextFailed);
    }
    /// See `cuda.Context.multiprocessorCount`: compute units on AMD (96 on a
    /// gfx1100 RX 7900 XTX), or WGPs, half as many, when the device runs in
    /// WGP mode. `error.Unsupported` before ROCm 6.0.
    pub fn multiprocessorCount(self: *Context) Error!u32 {
        try self.makeCurrent();
        return nonNeg((try self.properties()).multiProcessorCount);
    }
    /// `hipGetDevicePropertiesR0600` for this device, zero-filled first so a
    /// field the runtime leaves alone reads as 0. `error.Unsupported` before
    /// ROCm 6.0, which does not export the versioned call.
    fn properties(self: *const Context) Error!DeviceProp {
        const get = g.hipGetDevicePropertiesR0600 orelse return error.Unsupported;
        var p = std.mem.zeroes(DeviceProp);
        try check(get(&p, self.device), error.InvalidArgument);
        return p;
    }
    /// See `cuda.Context.info`. Everything is read off `properties` on ROCm
    /// 6.0 and later. Before that, the calls that predate the frozen layout
    /// still fill the name, uuid, PCI address and memory, and the rest is 0.
    pub fn info(self: *Context) Error!iface.DeviceInfo {
        try self.makeCurrent();
        var d: iface.DeviceInfo = .{ .ordinal = self.ordinal };
        if (self.properties()) |p| {
            d.name_buf = p.name;
            d.uuid = p.uuid;
            d.pci_domain = nonNeg(p.pciDomainID);
            d.pci_bus = nonNeg(p.pciBusID);
            d.pci_device = nonNeg(p.pciDeviceID);
            d.compute_capability = nonNeg(p.major * 10 + p.minor);
            d.sms = nonNeg(p.multiProcessorCount);
            d.fp64_ratio = archRatio(&p.gcnArchName) orelse 0;
            d.clock_khz = nonNeg(p.clockRate);
            d.mem_clock_khz = nonNeg(p.memoryClockRate);
            d.bus_width_bits = nonNeg(p.memoryBusWidth);
            d.mem_total = p.totalGlobalMem;
        } else |err| {
            if (err != error.Unsupported) return err;
            if (g.hipDeviceGetName) |get| try check(get(&d.name_buf, d.name_buf.len, self.device), error.InvalidArgument);
            if (g.hipDeviceGetPCIBusId) |get| {
                // "dddd:bb:dd.f", hexadecimal.
                var buf: [32]u8 = @splat(0);
                try check(get(&buf, buf.len, self.device), error.InvalidArgument);
                var it = std.mem.tokenizeAny(u8, std.mem.sliceTo(&buf, 0), ":.");
                for ([_]*u32{ &d.pci_domain, &d.pci_bus, &d.pci_device }) |field|
                    field.* = std.fmt.parseInt(u32, it.next() orelse break, 16) catch 0;
            }
            if (g.hipDeviceTotalMem) |get| {
                var total: usize = 0;
                try check(get(&total, self.device), error.InvalidArgument);
                d.mem_total = total;
            }
        }
        d.name_len = std.mem.findScalar(u8, &d.name_buf, 0) orelse d.name_buf.len;
        if (g.hipDeviceGetUuid) |get| try check(get(&d.uuid, self.device), error.InvalidArgument);
        if (g.hipMemGetInfo) |get| {
            var free: usize = 0;
            var total: usize = 0;
            try check(get(&free, &total), error.InvalidArgument);
            d.mem_free = free;
        }
        acquire();
        defer lock.unlock();
        d.fake = self.ordinal >= real_count;
        return d;
    }

    /// Caller owns the returned Stream and must `deinit` it.
    pub fn createStream(self: *Context) Error!Stream {
        try self.makeCurrent();
        var s: Stream = .{ .ordinal = self.ordinal };
        try check(g.hipStreamCreate(&s.stream, 0), error.SyncFailed);
        return s;
    }
    /// See `cuda.Context.createStreamNonBlocking`.
    pub fn createStreamNonBlocking(self: *Context) Error!Stream {
        try self.makeCurrent();
        var s: Stream = .{ .ordinal = self.ordinal };
        try check(g.hipStreamCreateWithFlags(&s.stream, hipStreamNonBlocking), error.SyncFailed);
        return s;
    }
    /// See `cuda.Context.createEvent`.
    pub fn createEvent(self: *Context, timing: bool) Error!Event {
        try self.makeCurrent();
        var e: Event = .{ .ordinal = self.ordinal };
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
    /// The device it was made on; -1 (hand-built) adopts the first one
    /// retained (`use`).
    ordinal: c_int = -1,

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
        use(self.ordinal);
        try check(g.hipMemcpyHtoD(try self.devicePtr(), host, n), error.CopyFailed);
    }
    pub fn download(self: *Buffer, host: *anyopaque, n: usize) Error!void {
        use(self.ordinal);
        try check(g.hipMemcpyDtoH(host, try self.devicePtr(), n), error.CopyFailed);
    }
    /// Release the device memory. Idempotent.
    pub fn free(self: *Buffer) void {
        // Also covers a default-constructed Buffer, where `g` is still
        // `undefined` because nothing ever dlopen'd the runtime.
        if (self.handle == null) return;
        use(self.ordinal);
        _ = g.hipFree(self.handle);
        self.* = .{};
    }
    /// The handle as a kernel argument. Borrows `self`: the pointer is into the
    /// Buffer, which must outlive every launch it is passed to.
    pub fn argPtr(self: *Buffer) iface.Arg {
        return @ptrCast(&self.handle);
    }
    /// Device-to-device copy of `n` bytes. Across two devices it is a peer
    /// copy: direct when `Context.enablePeerAccess` was granted, staged by
    /// the runtime otherwise. Either way it is `n` bytes in and `n` bytes
    /// out, never a host round trip of the caller's.
    pub fn copyFrom(self: *Buffer, src: *const Buffer, src_offset: usize, dst_offset: usize, n: usize) Error!void {
        use(self.ordinal);
        const dst = try self.offsetPtr(dst_offset);
        const from = try src.offsetPtr(src_offset);
        if (crossDevice(self.ordinal, src.ordinal))
            return check(g.hipMemcpyPeer(dst, deviceOf(self.ordinal), from, deviceOf(src.ordinal), n), error.CopyFailed);
        try check(g.hipMemcpyDtoD(dst, from, n), error.CopyFailed);
    }
    /// `copyFrom`, enqueued on `stream` and not waited on. Legal under capture.
    pub fn copyFromAsync(self: *Buffer, src: *const Buffer, src_offset: usize, dst_offset: usize, n: usize, stream: *Stream) Error!void {
        use(self.ordinal);
        const dst = try self.offsetPtr(dst_offset);
        const from = try src.offsetPtr(src_offset);
        if (crossDevice(self.ordinal, src.ordinal))
            return check(g.hipMemcpyPeerAsync(dst, deviceOf(self.ordinal), from, deviceOf(src.ordinal), n, stream.stream), error.CopyFailed);
        try check(g.hipMemcpyDtoDAsync(dst, from, n, stream.stream), error.CopyFailed);
    }
    pub fn downloadAt(self: *Buffer, host: *anyopaque, offset: usize, n: usize) Error!void {
        use(self.ordinal);
        try check(g.hipMemcpyDtoH(host, try self.offsetPtr(offset), n), error.CopyFailed);
    }
    pub fn uploadAt(self: *Buffer, host: *const anyopaque, offset: usize, n: usize) Error!void {
        use(self.ordinal);
        try check(g.hipMemcpyHtoD(try self.offsetPtr(offset), host, n), error.CopyFailed);
    }
    /// Enqueued on `stream` and not waited on; `host` must be `allocPinned`
    /// memory and must stay put until `stream.synchronize()` returns.
    pub fn downloadAtAsync(self: *Buffer, host: *anyopaque, offset: usize, n: usize, stream: *Stream) Error!void {
        use(self.ordinal);
        try check(g.hipMemcpyDtoHAsync(host, try self.offsetPtr(offset), n, stream.stream), error.CopyFailed);
    }
    pub fn uploadAtAsync(self: *Buffer, host: *const anyopaque, offset: usize, n: usize, stream: *Stream) Error!void {
        use(self.ordinal);
        try check(g.hipMemcpyHtoDAsync(try self.offsetPtr(offset), host, n, stream.stream), error.CopyFailed);
    }
    /// Set the first `n` bytes to `value`, on the device and on `stream`. The
    /// memory controller does it in place: zeroing this way costs no bus
    /// traffic, where copying a resident block of zeros over it does.
    pub fn fillAsync(self: *Buffer, value: u8, n: usize, stream: *Stream) Error!void {
        use(self.ordinal);
        try check(g.hipMemsetAsync(try self.devicePtr(), value, n, stream.stream), error.CopyFailed);
    }
};

/// A loaded module. `getKernel` looks entry points up by symbol name.
pub const Module = struct {
    module: hipModule_t = null,
    /// Owned by the process-wide cache; `deinit` must leave it alone.
    cached: bool = false,
    /// The device it was made on; -1 (hand-built) adopts the first one
    /// retained (`use`).
    ordinal: c_int = -1,

    pub fn getKernel(self: *Module, name: [*:0]const u8) Error!Kernel {
        use(self.ordinal);
        var k: Kernel = .{ .ordinal = self.ordinal };
        try check(g.hipModuleGetFunction(&k.func, self.module, name), error.KernelNotFound);
        return k;
    }
    /// Drops this handle. A cached module stays loaded for the process;
    /// `shutdown()` unloads it.
    pub fn deinit(self: *Module) void {
        if (!self.cached and self.module != null) {
            use(self.ordinal);
            _ = g.hipModuleUnload(self.module);
        }
        self.* = .{};
    }
};

/// An entry point in a loaded module.
pub const Kernel = struct {
    func: hipFunction_t = null,
    /// The device it was made on; -1 (hand-built) adopts the first one
    /// retained (`use`).
    ordinal: c_int = -1,
    pub fn launch(self: Kernel, grid: Dim3, block: Dim3, shared_bytes: u32, args: []const iface.Arg) Error!void {
        try self.launchOnStream(grid, block, shared_bytes, args, null);
    }
    /// Launch on `stream` and return without waiting. `args` must stay put
    /// until the launch has actually run -- `stream.synchronize()`.
    pub fn launchOnStream(self: Kernel, grid: Dim3, block: Dim3, shared_bytes: u32, args: []const iface.Arg, stream: hipStream_t) Error!void {
        use(self.ordinal);
        try check(g.hipModuleLaunchKernel(self.func, grid.x, grid.y, grid.z, block.x, block.y, block.z, shared_bytes, stream, @constCast(args.ptr), null), error.LaunchFailed);
    }
};

/// A queue of ordered asynchronous work. `null` is the default stream.
pub const Stream = struct {
    stream: hipStream_t = null,
    /// The device it was made on; -1 (hand-built) adopts the first one
    /// retained (`use`).
    ordinal: c_int = -1,
    pub fn synchronize(self: *Stream) Error!void {
        use(self.ordinal);
        try check(g.hipStreamSynchronize(self.stream), error.SyncFailed);
    }
    /// See `cuda.Stream.query`.
    pub fn query(self: *Stream) Error!bool {
        use(self.ordinal);
        return ready(g.hipStreamQuery(self.stream), error.SyncFailed);
    }
    /// See `cuda.Stream.waitEvent`.
    pub fn waitEvent(self: *Stream, event: *const Event) Error!void {
        use(self.ordinal);
        try check((g.hipStreamWaitEvent orelse return error.Unsupported)(self.stream, event.event, 0), error.EventFailed);
    }
    /// See `cuda.Stream.beginCapture`.
    pub fn beginCapture(self: *Stream, mode: CaptureMode) Error!void {
        use(self.ordinal);
        try check((g.hipStreamBeginCapture orelse return error.Unsupported)(self.stream, mode), error.CaptureFailed);
    }
    /// See `cuda.Stream.endCapture`.
    pub fn endCapture(self: *Stream) Error!Graph {
        use(self.ordinal);
        var graph: Graph = .{ .ordinal = self.ordinal };
        try check((g.hipStreamEndCapture orelse return error.Unsupported)(self.stream, &graph.graph), error.CaptureFailed);
        return graph;
    }
    /// See `cuda.Stream.isCapturing`.
    pub fn isCapturing(self: *Stream) Error!bool {
        use(self.ordinal);
        var status: c_int = 0;
        try check((g.hipStreamIsCapturing orelse return error.Unsupported)(self.stream, &status), error.CaptureFailed);
        return status != 0;
    }
    pub fn deinit(self: *Stream) void {
        // Same reason as `Buffer.free`: `Stream{}` is a value any caller can
        // build, and `g` is undefined until something dlopen'd the runtime.
        if (!loaded) return;
        use(self.ordinal);
        _ = g.hipStreamDestroy(self.stream);
        self.* = .{};
    }
};

/// See `cuda.Graph`.
pub const Graph = struct {
    graph: hipGraph_t = null,
    /// The device it was made on; -1 (hand-built) adopts the first one
    /// retained (`use`).
    ordinal: c_int = -1,

    pub fn instantiate(self: *const Graph) Error!GraphExec {
        use(self.ordinal);
        var exec: GraphExec = .{ .ordinal = self.ordinal };
        try check((g.hipGraphInstantiateWithFlags orelse return error.Unsupported)(&exec.exec, self.graph, 0), error.GraphFailed);
        return exec;
    }
    pub fn deinit(self: *Graph) void {
        if (!loaded or self.graph == null) return;
        use(self.ordinal);
        if (g.hipGraphDestroy) |destroy| _ = destroy(self.graph);
        self.* = .{};
    }
};

/// See `cuda.GraphExec`.
pub const GraphExec = struct {
    exec: hipGraphExec_t = null,
    /// The device it was made on; -1 (hand-built) adopts the first one
    /// retained (`use`).
    ordinal: c_int = -1,

    pub fn launch(self: *GraphExec, stream: *Stream) Error!void {
        use(self.ordinal);
        try check((g.hipGraphLaunch orelse return error.Unsupported)(self.exec, stream.stream), error.GraphFailed);
    }
    pub fn update(self: *GraphExec, graph: *const Graph) Error!bool {
        use(self.ordinal);
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
        use(self.ordinal);
        if (g.hipGraphExecDestroy) |destroy| _ = destroy(self.exec);
        self.* = .{};
    }
};

/// See `cuda.Event`.
pub const Event = struct {
    event: hipEvent_t = null,
    /// The device it was made on; -1 (hand-built) adopts the first one
    /// retained (`use`).
    ordinal: c_int = -1,

    pub fn record(self: *Event, stream: *Stream) Error!void {
        use(self.ordinal);
        try check((g.hipEventRecord orelse return error.Unsupported)(self.event, stream.stream), error.EventFailed);
    }
    pub fn synchronize(self: *Event) Error!void {
        use(self.ordinal);
        try check((g.hipEventSynchronize orelse return error.Unsupported)(self.event), error.EventFailed);
    }
    pub fn query(self: *Event) Error!bool {
        use(self.ordinal);
        return ready((g.hipEventQuery orelse return error.Unsupported)(self.event), error.EventFailed);
    }
    pub fn elapsedUs(start: *const Event, end: *const Event) Error!f32 {
        use(start.ordinal);
        var ms: f32 = 0;
        try check((g.hipEventElapsedTime orelse return error.Unsupported)(&ms, start.event, end.event), error.EventFailed);
        return ms * 1000;
    }
    pub fn deinit(self: *Event) void {
        if (!loaded or self.event == null) return;
        use(self.ordinal);
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
    // Two hand-built buffers share the default device (ordinal -1 on both),
    // so this is the DtoD path, and it fails on the null handle before any
    // peer or copy call could be reached.
    const other: Buffer = .{};
    try std.testing.expectError(error.InvalidArgument, buffer.copyFromAsync(&other, 0, 0, 1, &stream));
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
        inline for (.{ "instantiate", "launch", "update", "record", "synchronize", "query", "elapsedUs", "waitEvent", "beginCapture", "endCapture", "isCapturing", "copyFrom", "copyFromAsync", "createStreamNonBlocking", "createEvent", "canAccessPeer", "enablePeerAccess", "computeCapability", "fp64Ratio", "multiprocessorCount", "info" }) |name| {
            if (@hasDecl(pair[1], name)) try std.testing.expectEqual(
                @typeInfo(@TypeOf(@field(pair[1], name))).@"fn".param_types.len,
                @typeInfo(@TypeOf(@field(pair[0], name))).@"fn".param_types.len,
            );
        }
    }
    try std.testing.expect(CaptureMode == cuda.CaptureMode);
    // Nothing in this suite loads a HIP module, so the cache is empty.
    try std.testing.expectEqual(@as(usize, 0), loadedModuleCount());
    if (loaded) return;
    var graph: Graph = .{};
    graph.deinit();
    var exec: GraphExec = .{};
    exec.deinit();
    var event: Event = .{};
    event.deinit();
    // The device queries make the context current before they read `g`, so
    // without a runtime they fail at the dlopen and never touch it undefined.
    var ctx: Context = .{};
    try std.testing.expectError(error.InitFailed, ctx.computeCapability());
    try std.testing.expectError(error.InitFailed, ctx.fp64Ratio());
    try std.testing.expectError(error.InitFailed, ctx.multiprocessorCount());
    try std.testing.expectError(error.InitFailed, ctx.info());
    try std.testing.expectError(error.InitFailed, ctx.enablePeerAccess(&ctx));
}

test "the gfx family in a gcnArchName maps to the vendor's FP64 ratio" {
    try std.testing.expectEqual(Gfx{ .major = 9, .minor = 0, .stepping = 6 }, parseGfx("gfx906").?);
    try std.testing.expectEqual(Gfx{ .major = 9, .minor = 0, .stepping = 0xa }, parseGfx("gfx90a").?);
    try std.testing.expectEqual(Gfx{ .major = 9, .minor = 4, .stepping = 2 }, parseGfx("gfx942").?);
    try std.testing.expectEqual(Gfx{ .major = 11, .minor = 0, .stepping = 0 }, parseGfx("gfx1100:sramecc+:xnack-").?);
    try std.testing.expectEqual(Gfx{ .major = 12, .minor = 0, .stepping = 1 }, parseGfx("gfx1201").?);
    try std.testing.expectEqual(@as(?Gfx, null), parseGfx(""));
    try std.testing.expectEqual(@as(?Gfx, null), parseGfx("gfx"));
    try std.testing.expectEqual(@as(?Gfx, null), parseGfx("gfx9-generic"));

    for ([_]struct { []const u8, ?u32 }{
        .{ "gfx906", 2 },    .{ "gfx908", 2 },   .{ "gfx90a", 1 },                   .{ "gfx942", 2 },
        .{ "gfx950", 2 },    .{ "gfx900", 16 },  .{ "gfx90c", 16 },                  .{ "gfx1030", 16 },
        .{ "gfx1100", 16 },  .{ "gfx1201", 16 }, .{ "gfx1100:sramecc+:xnack-", 16 }, .{ "gfx803", null },
        .{ "gfx000", null }, .{ "", null },
    }) |case| try std.testing.expectEqual(case[1], archRatio(case[0]));

    // The real input: the struct's NUL-padded field, passed whole.
    var arch: [256]u8 = @splat(0);
    @memcpy(arch[0..6], "gfx906");
    try std.testing.expectEqual(@as(?u32, 2), archRatio(&arch));
}
