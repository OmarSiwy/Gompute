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
    hipLaunchCooperativeKernel: *const fn (hipFunction_t, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, hipStream_t, ?[*]iface.Arg) callconv(.c) hipError_t,
    hipOccupancyMaxActiveBlocksPerMultiprocessor: *const fn (*c_int, hipFunction_t, c_int, usize) callconv(.c) hipError_t,
    hipDeviceGetAttribute: *const fn (*c_int, c_int, hipDevice_t) callconv(.c) hipError_t,
    hipStreamCreate: *const fn (*hipStream_t, c_uint) callconv(.c) hipError_t,
    hipStreamDestroy: *const fn (hipStream_t) callconv(.c) hipError_t,
    hipStreamSynchronize: *const fn (hipStream_t) callconv(.c) hipError_t,
};

var g: Api = undefined;
var loaded = false;

const lib_names = switch (builtin.os.tag) {
    .windows => &[_][]const u8{"amdhip64.dll"},
    else => &[_][]const u8{ "libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.5" },
};

fn openFirst(names: []const []const u8) ?std.DynLib {
    for (names) |n| {
        if (std.DynLib.open(n)) |l| return l else |_| {}
    }
    return null;
}

fn loadApi() Error!void {
    if (loaded) return;
    var lib = openFirst(lib_names) orelse return error.InitFailed;
    errdefer lib.close();
    inline for (@typeInfo(Api).@"struct".fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "lib")) continue;
        @field(g, field.name) = lib.lookup(@TypeOf(@field(g, field.name)), field.name) orelse return error.InitFailed;
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

pub const Context = struct {
    device: hipDevice_t = 0,
    ctx: hipCtx_t = null,

    pub fn init(ordinal: c_int) Error!Context {
        try loadApi();
        try check(g.hipInit(0), error.InitFailed);
        var self: Context = .{};
        try check(g.hipDeviceGet(&self.device, ordinal), error.NoDevice);
        try check(g.hipCtxCreate(&self.ctx, 0, self.device), error.ContextFailed);
        return self;
    }
    pub fn deinit(self: *Context) void {
        _ = g.hipCtxDestroy(self.ctx);
        self.* = .{};
    }
    pub fn synchronize(_: *Context) Error!void {
        try check(g.hipDeviceSynchronize(), error.SyncFailed);
    }
    pub fn alloc(_: *Context, bytes: usize) Error!Buffer {
        var b: Buffer = .{ .bytes = bytes };
        try check(g.hipMalloc(&b.handle, bytes), error.AllocFailed);
        return b;
    }
    pub fn loadModuleFromMemory(_: *Context, image: []const u8) Error!Module {
        var m: Module = .{};
        try check(g.hipModuleLoadData(&m.module, image.ptr), error.ModuleLoadFailed);
        return m;
    }
    pub fn createStream(_: *Context) Error!Stream {
        var s: Stream = .{};
        try check(g.hipStreamCreate(&s.stream, 0), error.SyncFailed);
        return s;
    }
    pub const attr_multiprocessor_count: c_int = 16;
    pub const attr_cooperative_launch: c_int = 97;
    pub fn deviceAttribute(self: *Context, attrib: c_int) Error!c_int {
        var v: c_int = 0;
        try check(g.hipDeviceGetAttribute(&v, attrib, self.device), error.NoDevice);
        return v;
    }
    pub fn maxCoopBlocks(self: *Context, k: Kernel, block_dim: u32, shared_bytes: usize) Error!u32 {
        const coop = self.deviceAttribute(attr_cooperative_launch) catch 0;
        if (coop == 0) return 0;
        var per_sm: c_int = 0;
        try check(g.hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k.func, @intCast(block_dim), shared_bytes), error.LaunchFailed);
        const sms = try self.deviceAttribute(attr_multiprocessor_count);
        return @intCast(per_sm * sms);
    }
};

pub const Buffer = struct {
    handle: hipDeviceptr_t = null,
    bytes: usize = 0,

    pub fn upload(self: *Buffer, host: *const anyopaque, n: usize) Error!void {
        try check(g.hipMemcpyHtoD(self.handle, host, n), error.CopyFailed);
    }
    pub fn download(self: *Buffer, host: *anyopaque, n: usize) Error!void {
        try check(g.hipMemcpyDtoH(host, self.handle, n), error.CopyFailed);
    }
    pub fn free(self: *Buffer) void {
        _ = g.hipFree(self.handle);
        self.* = .{};
    }
    pub fn argPtr(self: *Buffer) iface.Arg {
        return @ptrCast(&self.handle);
    }
    pub fn copyFrom(self: *Buffer, src: *const Buffer, src_offset: usize, dst_offset: usize, n: usize) Error!void {
        const sp: usize = @intFromPtr(src.handle.?) + src_offset;
        const dp: usize = @intFromPtr(self.handle.?) + dst_offset;
        try check(g.hipMemcpyDtoD(@ptrFromInt(dp), @ptrFromInt(sp), n), error.CopyFailed);
    }
    pub fn downloadAt(self: *Buffer, host: *anyopaque, offset: usize, n: usize) Error!void {
        const src: usize = @intFromPtr(self.handle.?) + offset;
        try check(g.hipMemcpyDtoH(host, @ptrFromInt(src), n), error.CopyFailed);
    }
    pub fn uploadAt(self: *Buffer, host: *const anyopaque, offset: usize, n: usize) Error!void {
        const dst: usize = @intFromPtr(self.handle.?) + offset;
        try check(g.hipMemcpyHtoD(@ptrFromInt(dst), host, n), error.CopyFailed);
    }
};

pub const Module = struct {
    module: hipModule_t = null,
    pub fn getKernel(self: *Module, name: [*:0]const u8) Error!Kernel {
        var k: Kernel = .{};
        try check(g.hipModuleGetFunction(&k.func, self.module, name), error.KernelNotFound);
        return k;
    }
    pub fn deinit(self: *Module) void {
        _ = g.hipModuleUnload(self.module);
        self.* = .{};
    }
};

pub const Kernel = struct {
    func: hipFunction_t = null,
    pub fn launch(self: Kernel, grid: Dim3, block: Dim3, shared_bytes: u32, args: []const iface.Arg) Error!void {
        try self.launchOnStream(grid, block, shared_bytes, args, null);
    }
    pub fn launchOnStream(self: Kernel, grid: Dim3, block: Dim3, shared_bytes: u32, args: []const iface.Arg, stream: hipStream_t) Error!void {
        try check(g.hipModuleLaunchKernel(self.func, grid.x, grid.y, grid.z, block.x, block.y, block.z, shared_bytes, stream, @constCast(args.ptr), null), error.LaunchFailed);
    }
    pub fn launchCooperative(self: Kernel, grid: Dim3, block: Dim3, shared_bytes: u32, args: []const iface.Arg, stream: hipStream_t) Error!void {
        try check(g.hipLaunchCooperativeKernel(self.func, grid.x, grid.y, grid.z, block.x, block.y, block.z, shared_bytes, stream, @constCast(args.ptr)), error.LaunchFailed);
    }
};

pub const Stream = struct {
    stream: hipStream_t = null,
    pub fn synchronize(self: *Stream) Error!void {
        try check(g.hipStreamSynchronize(self.stream), error.SyncFailed);
    }
    pub fn deinit(self: *Stream) void {
        _ = g.hipStreamDestroy(self.stream);
        self.* = .{};
    }
};
