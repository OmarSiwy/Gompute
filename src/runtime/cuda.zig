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
    cuCtxCreate_v2: *const fn (*CUcontext, c_uint, CUdevice) callconv(.c) CUresult,
    cuCtxDestroy_v2: *const fn (CUcontext) callconv(.c) CUresult,
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
    // Launch
    cuLaunchKernel: *const fn (CUfunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, CUstream, ?[*]iface.Arg, ?[*]iface.Arg) callconv(.c) CUresult,
    // Device attributes
    cuDeviceGetAttribute: *const fn (*c_int, c_int, CUdevice) callconv(.c) CUresult,
    // Streams
    cuStreamCreate: *const fn (*CUstream, c_uint) callconv(.c) CUresult,
    cuStreamDestroy_v2: *const fn (CUstream) callconv(.c) CUresult,
    cuStreamSynchronize: *const fn (CUstream) callconv(.c) CUresult,
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
        @field(g, field.name) = lib.lookup(@TypeOf(@field(g, field.name)), field.name) orelse {
            std.debug.print("cuda: symbol not found: {s}\n", .{field.name});
            return error.InitFailed;
        };
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
