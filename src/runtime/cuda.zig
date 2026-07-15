//! CUDA backend via runtime dlopen.
//! Changes from v1:  #6 multi-GPU, #7 error detail, #8 pinned/unified mem,
//! #10 events/timing, #19 occupancy API.

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
const CUevent = ?*anyopaque;
const CUgraph = ?*anyopaque;
const CUgraphExec = ?*anyopaque;
const CUdeviceptr = c_ulonglong;
const CUtexObject = u64;
const CUsurfObject = u64;
const CUmemoryPool = ?*anyopaque;

/// CUDA_RESOURCE_DESC for texture/surface creation.
pub const ResourceDesc = extern struct {
    res_type: c_uint = 0, // 0=ARRAY, 1=MIPMAPPED_ARRAY, 2=LINEAR, 3=PITCH2D
    // Union: for LINEAR (type 2), fields are:
    dev_ptr: CUdeviceptr = 0,
    format: c_uint = 0, // CU_AD_FORMAT_FLOAT = 0x20
    num_channels: c_uint = 1,
    size_in_bytes: usize = 0,
    // Pad to match CUDA struct layout (192 bytes total)
    _pad: [192 - 32]u8 = [_]u8{0} ** (192 - 32),
};

/// CUDA_TEXTURE_DESC for texture creation.
pub const TextureDesc = extern struct {
    address_mode: [3]c_uint = .{ 0, 0, 0 }, // CU_TR_ADDRESS_MODE_WRAP=0
    filter_mode: c_uint = 0, // CU_TR_FILTER_MODE_POINT=0
    flags: c_uint = 0,
    max_anisotropy: c_uint = 1,
    mipmap_filter_mode: c_uint = 0,
    mipmap_level_bias: f32 = 0,
    min_mipmap_level_clamp: f32 = 0,
    max_mipmap_level_clamp: f32 = 0,
    border_color: [4]f32 = .{ 0, 0, 0, 0 },
    _reserved: [12]c_int = [_]c_int{0} ** 12,
};

const Api = struct {
    lib: std.DynLib,
    // Core
    cuInit: *const fn (c_uint) callconv(.c) CUresult,
    cuDeviceGet: *const fn (*CUdevice, c_int) callconv(.c) CUresult,
    cuDeviceGetCount: *const fn (*c_int) callconv(.c) CUresult,
    cuDeviceGetName: *const fn ([*]u8, c_int, CUdevice) callconv(.c) CUresult,
    cuCtxCreate_v2: *const fn (*CUcontext, c_uint, CUdevice) callconv(.c) CUresult,
    cuCtxDestroy_v2: *const fn (CUcontext) callconv(.c) CUresult,
    cuCtxSynchronize: *const fn () callconv(.c) CUresult,
    cuCtxSetCurrent: *const fn (CUcontext) callconv(.c) CUresult,
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
    // Pinned host memory (#8)
    cuMemAllocHost_v2: *const fn (**anyopaque, usize) callconv(.c) CUresult,
    cuMemFreeHost: *const fn (*anyopaque) callconv(.c) CUresult,
    // Managed / unified memory (#8)
    cuMemAllocManaged: *const fn (*CUdeviceptr, usize, c_uint) callconv(.c) CUresult,
    // Launch
    cuLaunchKernel: *const fn (CUfunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, CUstream, ?[*]iface.Arg, ?[*]iface.Arg) callconv(.c) CUresult,
    cuLaunchCooperativeKernel: *const fn (CUfunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, CUstream, ?[*]iface.Arg) callconv(.c) CUresult,
    // Occupancy (#19)
    cuOccupancyMaxActiveBlocksPerMultiprocessor: *const fn (*c_int, CUfunction, c_int, usize) callconv(.c) CUresult,
    cuOccupancyMaxPotentialBlockSize: *const fn (*c_int, *c_int, CUfunction, ?*const anyopaque, usize, c_int) callconv(.c) CUresult,
    // Device attributes
    cuDeviceGetAttribute: *const fn (*c_int, c_int, CUdevice) callconv(.c) CUresult,
    // Streams
    cuStreamCreate: *const fn (*CUstream, c_uint) callconv(.c) CUresult,
    cuStreamDestroy_v2: *const fn (CUstream) callconv(.c) CUresult,
    cuStreamSynchronize: *const fn (CUstream) callconv(.c) CUresult,
    // Events (#10)
    cuEventCreate: *const fn (*CUevent, c_uint) callconv(.c) CUresult,
    cuEventDestroy_v2: *const fn (CUevent) callconv(.c) CUresult,
    cuEventRecord: *const fn (CUevent, CUstream) callconv(.c) CUresult,
    cuEventSynchronize: *const fn (CUevent) callconv(.c) CUresult,
    cuEventElapsedTime: *const fn (*f32, CUevent, CUevent) callconv(.c) CUresult,
    // Texture / surface objects (#5)
    cuTexObjectCreate: *const fn (*CUtexObject, *const ResourceDesc, *const TextureDesc, ?*const anyopaque) callconv(.c) CUresult,
    cuTexObjectDestroy: *const fn (CUtexObject) callconv(.c) CUresult,
    cuSurfObjectCreate: *const fn (*CUsurfObject, *const ResourceDesc) callconv(.c) CUresult,
    cuSurfObjectDestroy: *const fn (CUsurfObject) callconv(.c) CUresult,
    // Memory pools (#8) — optional, CUDA 11.2+
    cuMemPoolCreate: ?*const fn (*CUmemoryPool, ?*const anyopaque) callconv(.c) CUresult = null,
    cuMemPoolDestroy: ?*const fn (CUmemoryPool) callconv(.c) CUresult = null,
    cuMemAllocAsync: ?*const fn (*CUdeviceptr, usize, CUstream) callconv(.c) CUresult = null,
    cuMemFreeAsync: ?*const fn (CUdeviceptr, CUstream) callconv(.c) CUresult = null,
    // Graph capture
    cuStreamBeginCapture_v2: *const fn (CUstream, c_uint) callconv(.c) CUresult,
    cuStreamEndCapture: *const fn (CUstream, *CUgraph) callconv(.c) CUresult,
    cuGraphInstantiate_v2: *const fn (*CUgraphExec, CUgraph, ?*anyopaque, ?*anyopaque, usize) callconv(.c) CUresult,
    cuGraphLaunch: *const fn (CUgraphExec, CUstream) callconv(.c) CUresult,
    cuGraphExecDestroy: *const fn (CUgraphExec) callconv(.c) CUresult,
    cuGraphDestroy: *const fn (CUgraph) callconv(.c) CUresult,
};

var g: Api = undefined;
var loaded = false;

const lib_names = switch (builtin.os.tag) {
    .windows => &[_][]const u8{"nvcuda.dll"},
    else => &[_][]const u8{
        "libcuda.so",                          "libcuda.so.1",
        "/run/opengl-driver/lib/libcuda.so.1", "/run/opengl-driver/lib/libcuda.so",
    },
};

fn openFirst(names: []const []const u8) ?std.DynLib {
    for (names) |n| {
        if (std.DynLib.open(n)) |l| return l else |_| {}
    }
    return null;
}

fn loadApi() Error!void {
    if (loaded) return;
    var lib = openFirst(lib_names) orelse {
        std.debug.print("cuda: failed to open any of: ", .{});
        for (lib_names) |n| std.debug.print("{s} ", .{n});
        std.debug.print("\n", .{});
        return error.InitFailed;
    };
    errdefer lib.close();
    inline for (@typeInfo(Api).@"struct".fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "lib")) continue;
        const FT = @TypeOf(@field(g, field.name));
        if (comptime @typeInfo(FT) == .optional) {
            @field(g, field.name) = lib.lookup(@typeInfo(FT).optional.child, field.name);
        } else {
            @field(g, field.name) = lib.lookup(FT, field.name) orelse {
                std.debug.print("cuda: symbol not found: {s}\n", .{field.name});
                return error.InitFailed;
            };
        }
    }
    g.lib = lib;
    loaded = true;
}

/// Check CUresult, stash raw code for #7 error detail.
inline fn check(rc: CUresult, err: Error) Error!void {
    if (rc != 0) {
        iface.last_driver_error = .{ .code = rc, .backend = .cuda };
        return err;
    }
}

// ---- Public API ----

pub const Context = struct {
    device: CUdevice = 0,
    ctx: CUcontext = null,

    /// (#6) Init with any device ordinal, not just 0.
    pub fn init(ordinal: c_int) Error!Context {
        try loadApi();
        try check(g.cuInit(0), error.InitFailed);
        var self: Context = .{};
        try check(g.cuDeviceGet(&self.device, ordinal), error.NoDevice);
        try check(g.cuCtxCreate_v2(&self.ctx, 0, self.device), error.ContextFailed);
        return self;
    }

    pub fn deinit(self: *Context) void {
        _ = g.cuCtxDestroy_v2(self.ctx);
        self.* = .{};
    }
    pub fn synchronize(_: *Context) Error!void {
        try check(g.cuCtxSynchronize(), error.SyncFailed);
    }

    /// (#6) Make this context current on the calling thread.
    pub fn makeCurrent(self: *Context) Error!void {
        try check(g.cuCtxSetCurrent(self.ctx), error.ContextFailed);
    }

    /// (#6) Query number of CUDA devices.
    pub fn deviceCount() Error!c_int {
        try loadApi();
        try check(g.cuInit(0), error.InitFailed);
        var count: c_int = 0;
        try check(g.cuDeviceGetCount(&count), error.NoDevice);
        return count;
    }

    /// (#6) Get device name.
    pub fn deviceName(self: *Context, buf: []u8) Error![]u8 {
        try check(g.cuDeviceGetName(buf.ptr, @intCast(buf.len), self.device), error.NoDevice);
        const len = std.mem.indexOfScalar(u8, buf, 0) orelse buf.len;
        return buf[0..len];
    }

    pub fn createStream(_: *Context) Error!Stream {
        var s: Stream = .{};
        try check(g.cuStreamCreate(&s.stream, 0), error.SyncFailed);
        return s;
    }
    pub fn alloc(_: *Context, bytes: usize) Error!Buffer {
        var b: Buffer = .{ .bytes = bytes };
        try check(g.cuMemAlloc_v2(&b.handle, bytes), error.AllocFailed);
        return b;
    }
    pub fn loadModuleFromMemory(_: *Context, image: []const u8) Error!Module {
        // cuModuleLoadData requires null-terminated PTX; @embedFile doesn't guarantee it.
        const ptr: [*]const u8 = if (image.len > 0 and image.ptr[image.len] == 0)
            image.ptr
        else blk: {
            const buf = std.heap.page_allocator.alloc(u8, image.len + 1) catch return error.ModuleLoadFailed;
            @memcpy(buf[0..image.len], image);
            buf[image.len] = 0;
            break :blk buf.ptr;
        };
        var m: Module = .{};
        try check(g.cuModuleLoadData(&m.module, ptr), error.ModuleLoadFailed);
        return m;
    }

    /// (#8) Allocate page-locked (pinned) host memory.
    pub fn allocPinned(_: *Context, bytes: usize) Error!PinnedBuffer {
        var ptr: *anyopaque = undefined;
        try check(g.cuMemAllocHost_v2(&ptr, bytes), error.AllocFailed);
        return .{ .ptr = ptr, .bytes = bytes };
    }

    /// (#8) Allocate managed (unified) memory.
    /// Flags: 1 = CU_MEM_ATTACH_GLOBAL, 2 = CU_MEM_ATTACH_HOST.
    pub fn allocManaged(_: *Context, bytes: usize) Error!Buffer {
        var b: Buffer = .{ .bytes = bytes };
        try check(g.cuMemAllocManaged(&b.handle, bytes, 1), error.AllocFailed);
        return b;
    }

    /// (#10) Create an event for timing.
    pub fn createEvent(_: *Context) Error!Event {
        var ev: Event = .{};
        try check(g.cuEventCreate(&ev.event, 0), error.InitFailed);
        return ev;
    }

    // -- Texture / surface objects (#5) --

    /// Create a bindless texture object over a linear device buffer.
    /// Pass the returned u64 handle as a kernel argument; use
    /// gpu.tex1Dfetch_f32(handle, idx) on device.
    pub fn createTexObject(_: *Context, dev_ptr: CUdeviceptr, num_elems: usize, num_channels: u32) Error!CUtexObject {
        var rd: ResourceDesc = .{};
        rd.res_type = 2; // CU_RESOURCE_TYPE_LINEAR
        rd.dev_ptr = dev_ptr;
        rd.format = 0x20; // CU_AD_FORMAT_FLOAT
        rd.num_channels = num_channels;
        rd.size_in_bytes = num_elems * num_channels * @sizeOf(f32);
        var td: TextureDesc = .{};
        td.filter_mode = 0; // CU_TR_FILTER_MODE_POINT
        var obj: CUtexObject = 0;
        try check(g.cuTexObjectCreate(&obj, &rd, &td, null), error.InitFailed);
        return obj;
    }

    pub fn destroyTexObject(_: *Context, obj: CUtexObject) void {
        _ = g.cuTexObjectDestroy(obj);
    }

    pub fn createSurfObject(_: *Context, dev_ptr: CUdeviceptr, size_bytes: usize) Error!CUsurfObject {
        var rd: ResourceDesc = .{};
        rd.res_type = 2;
        rd.dev_ptr = dev_ptr;
        rd.format = 0x20;
        rd.num_channels = 1;
        rd.size_in_bytes = size_bytes;
        var obj: CUsurfObject = 0;
        try check(g.cuSurfObjectCreate(&obj, &rd), error.InitFailed);
        return obj;
    }

    pub fn destroySurfObject(_: *Context, obj: CUsurfObject) void {
        _ = g.cuSurfObjectDestroy(obj);
    }

    // -- Memory pools (#8, CUDA 11.2+) --

    /// Async alloc from the default pool. Returns error.AllocFailed
    /// if pool APIs are not available (pre-11.2 driver).
    pub fn allocAsync(_: *Context, bytes: usize, stream: *Stream) Error!Buffer {
        const f = g.cuMemAllocAsync orelse return error.AllocFailed;
        var b: Buffer = .{ .bytes = bytes };
        try check(f(&b.handle, bytes, stream.stream), error.AllocFailed);
        return b;
    }

    /// Async free to the default pool.
    pub fn freeAsync(_: *Context, buf: *Buffer, stream: *Stream) Error!void {
        const f = g.cuMemFreeAsync orelse return error.AllocFailed;
        try check(f(buf.handle, stream.stream), error.AllocFailed);
        buf.* = .{};
    }

    pub const attr_multiprocessor_count: c_int = 16;
    pub const attr_cooperative_launch: c_int = 95;
    pub const attr_max_threads_per_block: c_int = 1;
    pub const attr_max_shared_memory_per_block: c_int = 8;
    pub const attr_warp_size: c_int = 10;

    pub fn deviceAttribute(self: *Context, attrib: c_int) Error!c_int {
        var v: c_int = 0;
        try check(g.cuDeviceGetAttribute(&v, attrib, self.device), error.NoDevice);
        return v;
    }
    pub fn maxCoopBlocks(self: *Context, k: Kernel, block_dim: u32, shared_bytes: usize) Error!u32 {
        const coop = self.deviceAttribute(attr_cooperative_launch) catch 0;
        if (coop == 0) return 0;
        var per_sm: c_int = 0;
        try check(g.cuOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k.func, @intCast(block_dim), shared_bytes), error.LaunchFailed);
        const sms = try self.deviceAttribute(attr_multiprocessor_count);
        return @intCast(per_sm * sms);
    }

    /// (#19) Query optimal block size for a kernel.
    pub fn optimalBlockSize(_: *Context, k: Kernel, shared_bytes: usize) Error!struct { grid: c_int, block: c_int } {
        var min_grid: c_int = 0;
        var block_size: c_int = 0;
        try check(g.cuOccupancyMaxPotentialBlockSize(&min_grid, &block_size, k.func, null, shared_bytes, 0), error.LaunchFailed);
        return .{ .grid = min_grid, .block = block_size };
    }
};

pub const Buffer = struct {
    handle: CUdeviceptr = 0,
    bytes: usize = 0,

    pub fn upload(self: *Buffer, host: *const anyopaque, n: usize) Error!void {
        try check(g.cuMemcpyHtoD_v2(self.handle, host, n), error.CopyFailed);
    }
    pub fn download(self: *Buffer, host: *anyopaque, n: usize) Error!void {
        try check(g.cuMemcpyDtoH_v2(host, self.handle, n), error.CopyFailed);
    }
    pub fn downloadAt(self: *Buffer, host: *anyopaque, offset: usize, n: usize) Error!void {
        try check(g.cuMemcpyDtoH_v2(host, self.handle + offset, n), error.CopyFailed);
    }
    pub fn uploadAt(self: *Buffer, host: *const anyopaque, offset: usize, n: usize) Error!void {
        try check(g.cuMemcpyHtoD_v2(self.handle + offset, host, n), error.CopyFailed);
    }
    pub fn free(self: *Buffer) void {
        _ = g.cuMemFree_v2(self.handle);
        self.* = .{};
    }
    pub fn argPtr(self: *Buffer) iface.Arg {
        return @ptrCast(&self.handle);
    }
    pub fn copyFrom(self: *Buffer, src: *const Buffer, src_offset: usize, dst_offset: usize, n: usize) Error!void {
        try check(g.cuMemcpyDtoD_v2(self.handle + dst_offset, src.handle + src_offset, n), error.CopyFailed);
    }
    pub fn downloadAtAsync(self: *Buffer, host: *anyopaque, offset: usize, n: usize, stream: CUstream) Error!void {
        try check(g.cuMemcpyDtoHAsync_v2(host, self.handle + offset, n, stream), error.CopyFailed);
    }
    pub fn uploadAtAsync(self: *Buffer, host: *const anyopaque, offset: usize, n: usize, stream: CUstream) Error!void {
        try check(g.cuMemcpyHtoDAsync_v2(self.handle + offset, host, n, stream), error.CopyFailed);
    }

    pub fn deviceAddr(self: *const Buffer) u64 {
        return self.handle;
    }
};

/// (#8) Pinned (page-locked) host memory for async transfers.
pub const PinnedBuffer = struct {
    ptr: *anyopaque,
    bytes: usize,

    pub fn free(self: *PinnedBuffer) void {
        _ = g.cuMemFreeHost(self.ptr);
        self.* = undefined;
    }
    pub fn slice(self: *PinnedBuffer, comptime T: type) []T {
        const typed: [*]T = @ptrCast(@alignCast(self.ptr));
        return typed[0 .. self.bytes / @sizeOf(T)];
    }
};

pub const Module = struct {
    module: CUmodule = null,
    pub fn getKernel(self: *Module, name: [*:0]const u8) Error!Kernel {
        var k: Kernel = .{};
        try check(g.cuModuleGetFunction(&k.func, self.module, name), error.KernelNotFound);
        return k;
    }
    pub fn deinit(self: *Module) void {
        _ = g.cuModuleUnload(self.module);
        self.* = .{};
    }
};

pub const Kernel = struct {
    func: CUfunction = null,
    pub fn launch(self: Kernel, grid: Dim3, block: Dim3, shared_bytes: u32, args: []const iface.Arg) Error!void {
        try self.launchOnStream(grid, block, shared_bytes, args, null);
    }
    pub fn launchOnStream(self: Kernel, grid: Dim3, block: Dim3, shared_bytes: u32, args: []const iface.Arg, stream: CUstream) Error!void {
        try check(g.cuLaunchKernel(self.func, grid.x, grid.y, grid.z, block.x, block.y, block.z, shared_bytes, stream, @constCast(args.ptr), null), error.LaunchFailed);
    }
    pub fn launchCooperative(self: Kernel, grid: Dim3, block: Dim3, shared_bytes: u32, args: []const iface.Arg, stream: CUstream) Error!void {
        try check(g.cuLaunchCooperativeKernel(self.func, grid.x, grid.y, grid.z, block.x, block.y, block.z, shared_bytes, stream, @constCast(args.ptr)), error.LaunchFailed);
    }
};

pub const Stream = struct {
    stream: CUstream = null,
    pub fn synchronize(self: *Stream) Error!void {
        try check(g.cuStreamSynchronize(self.stream), error.SyncFailed);
    }
    pub fn deinit(self: *Stream) void {
        _ = g.cuStreamDestroy_v2(self.stream);
        self.* = .{};
    }
};

/// (#10) Event for GPU-side timing.
pub const Event = struct {
    event: CUevent = null,
    pub fn record(self: *Event, stream: *Stream) Error!void {
        try check(g.cuEventRecord(self.event, stream.stream), error.SyncFailed);
    }
    pub fn synchronize(self: *Event) Error!void {
        try check(g.cuEventSynchronize(self.event), error.SyncFailed);
    }
    pub fn deinit(self: *Event) void {
        _ = g.cuEventDestroy_v2(self.event);
        self.* = undefined;
    }
    /// Elapsed time in milliseconds between two recorded events.
    pub fn elapsedMs(start: *Event, stop: *Event) Error!f32 {
        var ms: f32 = 0;
        try check(g.cuEventElapsedTime(&ms, start.event, stop.event), error.SyncFailed);
        return ms;
    }
};

pub const Graph = struct {
    exec: CUgraphExec = null,
    graph: CUgraph = null,
    pub fn deinit(self: *Graph) void {
        if (self.exec != null) _ = g.cuGraphExecDestroy(self.exec);
        if (self.graph != null) _ = g.cuGraphDestroy(self.graph);
        self.* = .{};
    }
    pub fn launch(self: *Graph, stream: *Stream) Error!void {
        try check(g.cuGraphLaunch(self.exec, stream.stream), error.LaunchFailed);
    }
};

pub fn beginCapture(stream: *Stream) Error!void {
    try check(g.cuStreamBeginCapture_v2(stream.stream, 0), error.LaunchFailed);
}
pub fn endCapture(stream: *Stream) Error!Graph {
    var gr: Graph = .{};
    try check(g.cuStreamEndCapture(stream.stream, &gr.graph), error.LaunchFailed);
    errdefer gr.deinit();
    try check(g.cuGraphInstantiate_v2(&gr.exec, gr.graph, null, null, 0), error.LaunchFailed);
    return gr;
}
