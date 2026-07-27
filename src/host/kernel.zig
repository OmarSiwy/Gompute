//! Compile-time-specialized host handles.

const std = @import("std");
const abi = @import("../core/abi.zig");
const iface = @import("../core/interface.zig");
const cuda = @import("../runtime/cuda.zig");
const hip = @import("../runtime/hip.zig");

pub const Backend = enum { cpu, cuda, hip };

pub fn Kernel(comptime Spec: type, comptime backend: Backend) type {
    return switch (backend) {
        .cpu => CpuKernel(Spec),
        .cuda => GpuKernel(Spec, gpu_cuda),
        .hip => GpuKernel(Spec, gpu_hip),
    };
}

fn CpuKernel(comptime Spec: type) type {
    return struct {
        const Self = @This();
        pub const backend: Backend = .cpu;
        pub const Buffer = void;

        pub inline fn init(_: c_int) iface.Error!Self {
            return .{};
        }

        pub inline fn deinit(_: *Self) void {}

        /// A generic map body (`fn (x: anytype, p: Params)`) instantiates at
        /// vector width; a strict `fn (T, Params) T` cannot, so it stays scalar.
        const lanes: usize = if (@hasDecl(Spec, "is_generic") and Spec.is_generic)
            std.simd.suggestVectorLength(Spec.Value) orelse 1
        else
            1;

        pub inline fn run(_: *Self, data: []Spec.Value, params: Spec.Parameters) iface.Error!void {
            if (comptime lanes == 1) {
                for (data) |*value| value.* = Spec.eval(value.*, params);
                return;
            }
            var i: usize = 0;
            while (i + lanes <= data.len) : (i += lanes) {
                const chunk: @Vector(lanes, Spec.Value) = data[i..][0..lanes].*;
                data[i..][0..lanes].* = Spec.eval(chunk, params);
            }
            for (data[i..]) |*value| value.* = Spec.eval(value.*, params);
        }
    };
}

/// The two GPU backends differ only in which runtime module they call and how
/// a kernel's entry name is spelled in the emitted artifact.
pub const Gpu = struct {
    backend: Backend,
    rt: type,
    /// Field names in the generated `gompute_kernels` module.
    has: []const u8,
    image: []const u8,
    /// HIP entry points keep their mangled Zig name; look it up in `hip_names`.
    mangled: bool,
};

pub const gpu_cuda: Gpu = .{ .backend = .cuda, .rt = cuda, .has = "has_cuda", .image = "cuda", .mangled = false };
pub const gpu_hip: Gpu = .{ .backend = .hip, .rt = hip, .has = "has_hip", .image = "hip", .mangled = true };

/// Open the artifact for `gpu` and resolve `entry_name` in it.
pub fn openModule(comptime gpu: Gpu, comptime entry_name: [:0]const u8, ordinal: c_int) iface.Error!struct {
    context: gpu.rt.Context,
    module: gpu.rt.Module,
    kernel: gpu.rt.Kernel,
} {
    const artifacts = @import("gompute_kernels");
    if (comptime !@field(artifacts, gpu.has)) return error.BackendUnavailable;

    var context = try gpu.rt.Context.init(ordinal);
    errdefer context.deinit();
    var module = try context.loadModuleFromMemory(@field(artifacts, gpu.image));
    errdefer module.deinit();
    const internal_name = comptime if (gpu.mangled)
        artifacts.hip_names.resolve(entry_name)
    else
        entry_name;
    return .{ .context = context, .module = module, .kernel = try module.getKernel(internal_name.ptr) };
}

fn GpuKernel(comptime Spec: type, comptime gpu: Gpu) type {
    return struct {
        const Self = @This();
        pub const backend: Backend = gpu.backend;
        pub const Buffer = gpu.rt.Buffer;

        context: gpu.rt.Context,
        module: gpu.rt.Module,
        kernel: gpu.rt.Kernel,

        /// (#3) Handles on the same device share one primary context and one
        /// JIT'd copy of the artifact, so this is cheap after the first one and
        /// a `Buffer` from any handle is valid in all of them.
        pub fn init(ordinal: c_int) iface.Error!Self {
            const opened = try openModule(gpu, Spec.entry_name, ordinal);
            return .{ .context = opened.context, .module = opened.module, .kernel = opened.kernel };
        }

        /// (#3) Drops this handle only. The device context and the loaded module
        /// are process-wide and shared with every other handle on the device, so
        /// this no longer tears them down; call `gompute.runtime.<backend>
        /// .shutdown()` if you genuinely want that. Still safe to call, still
        /// safe to call on a handle nobody else shares.
        pub fn deinit(self: *Self) void {
            self.module.deinit();
            self.context.deinit();
            self.* = undefined;
        }

        pub fn alloc(self: *Self, count: usize) iface.Error!Buffer {
            return self.context.alloc(count * @sizeOf(Spec.Value));
        }

        pub fn launch(
            self: *Self,
            buffer: *Buffer,
            count: usize,
            params: Spec.Parameters,
        ) iface.Error!void {
            if (count == 0) return;
            // A worker thread that has not touched this device yet has no current
            // context; without this every launch off the main thread fails.
            try self.context.makeCurrent();
            var len: u64 = @intCast(count);
            var packed_params = abi.pack(Spec.Parameters, params);
            var args = [_]iface.Arg{
                buffer.argPtr(),
                iface.arg(&len),
                iface.arg(&packed_params),
            };
            try self.kernel.launch(
                iface.Dim3.linear(count, Spec.block_size),
                .{ .x = Spec.block_size },
                0,
                &args,
            );
        }

        pub fn run(self: *Self, data: []Spec.Value, params: Spec.Parameters) iface.Error!void {
            if (data.len == 0) return;
            var buffer = try self.alloc(data.len);
            defer buffer.free();
            try buffer.upload(data.ptr, data.len * @sizeOf(Spec.Value));
            try self.launch(&buffer, data.len, params);
            try self.context.synchronize();
            try buffer.download(data.ptr, data.len * @sizeOf(Spec.Value));
        }
    };
}

pub fn AutoKernel(comptime Spec: type) type {
    const Cpu = Kernel(Spec, .cpu);
    const Cuda = Kernel(Spec, .cuda);
    const Hip = Kernel(Spec, .hip);

    return union(enum) {
        cpu: Cpu,
        cuda: Cuda,
        hip: Hip,

        const Self = @This();

        pub fn init() Self {
            if (Cuda.init(0)) |value| return .{ .cuda = value } else |_| {}
            if (Hip.init(0)) |value| return .{ .hip = value } else |_| {}
            return .{ .cpu = Cpu.init(0) catch unreachable };
        }

        pub fn deinit(self: *Self) void {
            switch (self.*) {
                .cpu => |*k| k.deinit(),
                .cuda => |*k| k.deinit(),
                .hip => |*k| k.deinit(),
            }
        }

        pub fn run(self: *Self, data: []Spec.Value, params: Spec.Parameters) iface.Error!void {
            switch (self.*) {
                .cpu => |*k| try k.run(data, params),
                .cuda => |*k| try k.run(data, params),
                .hip => |*k| try k.run(data, params),
            }
        }

        pub fn selected(self: *const Self) Backend {
            return switch (self.*) {
                .cpu => .cpu,
                .cuda => .cuda,
                .hip => .hip,
            };
        }
    };
}

test "vectorized cpu run matches the scalar loop at every tail boundary" {
    const spec = @import("../core/spec.zig");
    const P = struct { scale: f32 };
    const Generic = spec.Map("simd_probe", f32, P, struct {
        fn call(x: anytype, p: P) @TypeOf(x) {
            const y = x * spec.splat(@TypeOf(x), p.scale);
            return @max(y, spec.splat(@TypeOf(x), @as(f32, 0)));
        }
    }.call, .{});
    try std.testing.expect(Generic.is_generic);

    const K = CpuKernel(Generic);
    const params: P = .{ .scale = 2 };
    var buf: [3 * 64 + 1]f32 = undefined;
    var expected: [buf.len]f32 = undefined;

    for (0..3 * K.lanes + 2) |len| {
        for (buf[0..len], 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) - 8;
        for (buf[0..len], expected[0..len]) |v, *e| e.* = Generic.eval(v, params);

        var kernel = try K.init(0);
        try kernel.run(buf[0..len], params);
        try std.testing.expectEqualSlices(f32, expected[0..len], buf[0..len]);
    }
}
