//! Compile-time-specialized host handles.

const std = @import("std");
const abi = @import("../core/abi.zig");
const iface = @import("../core/interface.zig");
const cuda = @import("../runtime/cuda.zig");
const hip = @import("../runtime/hip.zig");

pub const Backend = enum { cpu, cuda, hip };

/// Naming an explicit backend is a promise made in build.zig. If the build did
/// not keep it, that is a build bug and belongs at build time, not at run time.
/// `AutoKernel` deliberately bypasses this and gates on `.available` instead.
pub fn Kernel(comptime Spec: type, comptime backend: Backend) type {
    return switch (backend) {
        .cpu => CpuKernel(Spec),
        .cuda => requireArtifacts(gpu_cuda, GpuKernel(Spec, gpu_cuda)),
        .hip => requireArtifacts(gpu_hip, GpuKernel(Spec, gpu_hip)),
    };
}

fn CpuKernel(comptime Spec: type) type {
    return struct {
        const Self = @This();
        pub const backend: Backend = .cpu;
        pub const available = true;
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
    /// One device image per kernel root, in blob order.
    images: []const u8,
    /// Kernel name -> `.{ .blob, .symbol }`, merged from every root at comptime.
    /// Both backends consult it: it says which blob to load, and the lookup
    /// itself is what turns a missing kernel into a compile error rather than a
    /// runtime KernelNotFound. See `openModule`.
    index: []const u8,
    /// A device arch to name in diagnostics, so the fix is copy-pasteable.
    example_cpu: []const u8,
};

pub const gpu_cuda: Gpu = .{ .backend = .cuda, .rt = cuda, .has = "has_cuda", .images = "cuda_images", .index = "cuda_index", .example_cpu = "sm_89" };
pub const gpu_hip: Gpu = .{ .backend = .hip, .rt = hip, .has = "has_hip", .images = "hip_images", .index = "hip_index", .example_cpu = "gfx1100" };

/// Whether this build actually carries a device artifact for `gpu`.
///
/// The only public way to ask before writing `Kernel(Spec, .cuda)`, which is a
/// compile error when the answer is `false`. Reachable as
/// `AutoKernel(Spec).Cuda.available` / `.Hip.available`.
///
/// The import stays inside the function on purpose: the library's own test
/// build has no `gompute_kernels` module at all, and a file-scope import would
/// be analyzed there.
pub fn available(comptime gpu: Gpu) bool {
    return @field(@import("gompute_kernels"), gpu.has);
}

/// Whether `emitKernels`/`addKernels` was ever called. The generated module says
/// `emitted = true`, build.zig's default stub says `false`; a build.zig old
/// enough to declare neither can only have resolved the real generated module.
fn emitted() bool {
    const artifacts = @import("gompute_kernels");
    return !@hasDecl(artifacts, "emitted") or artifacts.emitted;
}

/// Passes `T` through, or explains at compile time why `gpu` has no artifact.
pub fn requireArtifacts(comptime gpu: Gpu, comptime T: type) type {
    if (comptime available(gpu)) return T;
    const tag = @tagName(gpu.backend);
    if (comptime !emitted()) @compileError(
        "gompute: Kernel(Spec, ." ++ tag ++ ") needs device artifacts, but this build never " ++
            "emitted any. Add to build.zig:\n" ++
            "    const gompute_build = @import(\"gompute\");\n" ++
            "    gompute_build.emitKernels(b, dep, exe, .{ .kernels_root = b.path(\"src/kernels.zig\") });\n" ++
            "(or gompute_build.addKernels for more than one executable).",
    );
    @compileError(
        "gompute: Kernel(Spec, ." ++ tag ++ ") was asked for, but this build emitted no " ++ tag ++
            " artifacts. Either .auto detected no " ++ tag ++ " GPU on the BUILD machine, or you " ++
            "passed ." ++ tag ++ " = .{ .enabled = false }. Fix it by pinning the arch you deploy " ++
            "to -- ." ++ tag ++ " = .{ .gpu = .{ .name = \"" ++ gpu.example_cpu ++ "\" } } in " ++
            "emitKernels -- or switch to AutoKernel(Spec), which compiles either way and falls " ++
            "back to the CPU at run time.",
    );
}

/// A device, the loaded blob it came from, and one kernel inside it.
pub fn Opened(comptime gpu: Gpu) type {
    return struct {
        context: gpu.rt.Context,
        module: gpu.rt.Module,
        kernel: gpu.rt.Kernel,
    };
}

/// Open the artifact for `gpu` and resolve `entry_name` in it.
///
/// The name is resolved at compile time: a kernel that is not in any emitted
/// root is a `@compileError` naming it, rather than a runtime `KernelNotFound`
/// discovered on a customer's machine. Nothing else in the pipeline verifies
/// that a requested kernel was actually emitted.
pub fn openModule(
    comptime gpu: Gpu,
    comptime entry_name: [:0]const u8,
    ordinal: c_int,
) iface.Error!Opened(gpu) {
    const artifacts = @import("gompute_kernels");
    if (comptime !@field(artifacts, gpu.has)) return error.BackendUnavailable;
    const entry = comptime @field(artifacts, gpu.index).get(entry_name) orelse @compileError(
        "gompute: " ++ @tagName(gpu.backend) ++ " kernel \"" ++ entry_name ++ "\" is not in " ++
            "any emitted kernel root; export it from the root file with " ++
            "`comptime { g.exportKernels(@This()); }`, and check that the root is listed in " ++
            "your emitKernels call.",
    );
    // Only this kernel's blob is loaded: one root out of N is JIT'd, not all N.
    return open(gpu, entry_name, entry, ordinal);
}

/// `openModule` for a kernel name that is only known at run time -- a name read
/// out of a config file or a netlist. The set of kernels is still closed at
/// build time, so an unknown name is `error.KernelNotFound`, not a panic.
pub fn openModuleByName(
    comptime gpu: Gpu,
    name: []const u8,
    ordinal: c_int,
) iface.Error!Opened(gpu) {
    const artifacts = @import("gompute_kernels");
    if (comptime !@field(artifacts, gpu.has)) return error.BackendUnavailable;
    const entry = @field(artifacts, gpu.index).get(name) orelse return error.KernelNotFound;
    return open(gpu, name, entry, ordinal);
}

/// `entry` is `gompute_kernels.Entry`, duck-typed so the generated module owns
/// the definition.
fn open(comptime gpu: Gpu, name: []const u8, entry: anytype, ordinal: c_int) iface.Error!Opened(gpu) {
    const artifacts = @import("gompute_kernels");
    const tag = @tagName(gpu.backend);

    var context = try gpu.rt.Context.init(ordinal);
    errdefer context.deinit();
    // Past this point the artifact is in the binary and the device is up, so
    // every remaining failure is a build bug. Callers are allowed to swallow the
    // error (AutoKernel does); they are not allowed to swallow the reason.
    var module = context.loadModuleFromMemory(@field(artifacts, gpu.images)[entry.blob]) catch |err| {
        std.log.err(
            "gompute: this binary's " ++ tag ++ " artifact for kernel root \"{s}\" will not " ++
                "load on this device ({t}, driver code {d}). The usual cause is a device arch " ++
                "mismatch -- the build compiled for one GPU and this machine has another. Pin " ++
                "the arch you deploy to with ." ++ tag ++ " = .{{ .gpu = .{{ .name = \"" ++
                gpu.example_cpu ++ "\" }} }} in emitKernels.",
            .{ artifacts.root_names[entry.blob], err, iface.last_driver_error.code },
        );
        return err;
    };
    errdefer module.deinit();
    const kernel = module.getKernel(entry.symbol.ptr) catch |err| {
        std.log.err(
            "gompute: " ++ tag ++ " kernel \"{s}\" is in the name table but not in the emitted " ++
                "artifact ({t}, driver code {d}). Two things to check: is it listed in the " ++
                "gompute.exportKernels(.{{ ... }}) call in your kernels root, and does the " ++
                "root you passed to emitKernels point at that same file?",
            .{ name, err, iface.last_driver_error.code },
        );
        return err;
    };
    return .{ .context = context, .module = module, .kernel = kernel };
}

/// `GpuKernel` declares its own `available`, which would shadow the function.
const host_available = available;

fn GpuKernel(comptime Spec: type, comptime gpu: Gpu) type {
    return struct {
        const Self = @This();
        pub const backend: Backend = gpu.backend;
        /// False when this build emitted no artifact for `gpu`; `init` then
        /// always returns `error.BackendUnavailable`.
        pub const available = host_available(gpu);
        pub const Buffer = gpu.rt.Buffer;

        context: gpu.rt.Context = .{},
        module: gpu.rt.Module = .{},
        kernel: gpu.rt.Kernel = .{},

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
        ///
        /// Resets rather than `undefined`: in ReleaseFast `undefined` leaves the
        /// old live handles in place, so a stray double-deinit becomes a
        /// driver-level double free. Cleared handles make it a no-op, which is
        /// what the runtime structs already do.
        pub fn deinit(self: *Self) void {
            self.module.deinit();
            self.context.deinit();
            self.* = .{};
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

/// A backend that is simply not on this machine is a legitimate quiet CPU
/// fallback. Anything else means the artifact IS in the binary and is wrong,
/// and a ~100x downgrade must never be silent.
fn absent(err: iface.Error) bool {
    return switch (err) {
        // No artifact in the build, no driver to dlopen, no such device.
        error.BackendUnavailable, error.InitFailed, error.NoDevice => true,
        else => false,
    };
}

/// `openModule` has already logged the specifics; this adds the consequence.
fn reportSkip(comptime tag: []const u8, err: iface.Error, strict: bool) iface.Error!void {
    if (absent(err)) return;
    if (strict) return err;
    std.log.warn(
        "gompute: this build contains " ++ tag ++ " artifacts but the " ++ tag ++ " backend " ++
            "failed to start ({t}), so AutoKernel fell back to the CPU -- typically ~100x " ++
            "slower. That is a broken build, not a missing GPU. Use initStrict() to make it fatal.",
        .{err},
    );
}

pub fn AutoKernel(comptime Spec: type) type {
    return union(enum) {
        cpu: Cpu,
        cuda: Cuda,
        hip: Hip,

        const Self = @This();

        pub const Cpu = Kernel(Spec, .cpu);
        /// Not `Kernel(Spec, .cuda)`: that is a compile error when the build
        /// emitted no CUDA, and the entire point of `AutoKernel` is to compile
        /// either way. Ask `Cuda.available` for the answer instead.
        pub const Cuda = GpuKernel(Spec, gpu_cuda);
        pub const Hip = GpuKernel(Spec, gpu_hip);

        /// Picks the fastest backend that works, silently falling back to the
        /// CPU when this machine has no GPU -- but warning loudly when it has
        /// one and the artifact is broken. Use `initStrict` to refuse instead.
        pub fn init() Self {
            // `reportSkip` only errors when strict; the CPU tail is infallible.
            return probe(false) catch .{ .cpu = Cpu.init(0) catch unreachable };
        }

        /// `init`, except a build that ships an artifact the device rejects is
        /// an error rather than a quiet CPU downgrade. A machine with no GPU at
        /// all still returns `.cpu`.
        pub fn initStrict() iface.Error!Self {
            return probe(true);
        }

        fn probe(strict: bool) iface.Error!Self {
            if (comptime Cuda.available) {
                if (Cuda.init(0)) |value| return .{ .cuda = value } else |err| try reportSkip("cuda", err, strict);
            }
            if (comptime Hip.available) {
                if (Hip.init(0)) |value| return .{ .hip = value } else |err| try reportSkip("hip", err, strict);
            }
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

test "a broken build is distinguishable from a machine that simply has no GPU" {
    // The whole of C5: get this classification wrong and a ~100x CPU downgrade
    // goes back to being silent. `strict = true` returns before it logs.
    for ([_]iface.Error{ error.BackendUnavailable, error.InitFailed, error.NoDevice }) |absence| {
        try std.testing.expect(absent(absence));
        try reportSkip("cuda", absence, true);
    }
    for ([_]iface.Error{ error.ModuleLoadFailed, error.KernelNotFound, error.LaunchFailed }) |broken| {
        try std.testing.expect(!absent(broken));
        try std.testing.expectError(broken, reportSkip("cuda", broken, true));
    }
}
