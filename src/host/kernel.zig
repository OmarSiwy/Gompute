//! Compile-time-specialized host handles.

const std = @import("std");
const abi = @import("../core/abi.zig");
const iface = @import("../core/interface.zig");
const spec = @import("../core/spec.zig");
const cuda = @import("../runtime/cuda.zig");
const hip = @import("../runtime/hip.zig");

pub const Backend = enum { cpu, cuda, hip };

/// The most partial results a `reduce` will ever produce, and therefore the
/// biggest final fold the host does itself.
///
/// ponytail: a fixed 1024-block grid instead of a second-pass kernel or a
/// device atomic. Folding <=1024 values on the CPU costs microseconds and is
/// dwarfed by the download that carries them; a second kernel is another launch
/// plus another buffer, and `atomicAdd` on floats means `global_atomic_add_f32`,
/// which is gfx9+ and needs `--unsafe-fp-atomics` on AMD. Raise this if a
/// profile ever shows the fold mattering -- the device side already scales, it
/// is a grid-stride loop.
const max_reduce_blocks: u32 = 1024;

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

        const vec_lanes: usize = std.simd.suggestVectorLength(Spec.Value) orelse 1;

        /// A generic map body (`fn (x: anytype, p: Params)`) instantiates at
        /// vector width; a strict `fn (T, Params) T` cannot, so it stays scalar.
        const lanes: usize = if (@hasDecl(Spec, "is_generic") and Spec.is_generic)
            vec_lanes
        else
            1;

        /// Every `run` below is the loop a caller would have written by hand,
        /// with no dispatch left at run time. That is the point of naming
        /// `.cpu` at compile time.
        pub const run = switch (spec.kindOf(Spec)) {
            .map => mapRun,
            .map_to => mapToRun,
            .zip => zipRun,
            .map_indexed => mapIndexedRun,
            .reduce => reduceRun,
            .gather => gatherRun,
            .scatter => scatterRun,
        };

        inline fn mapRun(_: *Self, data: []Spec.Value, params: Spec.Parameters) iface.Error!void {
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

        inline fn mapToRun(
            _: *Self,
            in: []const Spec.In,
            out: []Spec.Out,
            params: Spec.Parameters,
        ) iface.Error!void {
            if (in.len != out.len) return error.InvalidArgument;
            for (in, out) |x, *o| o.* = Spec.eval(x, params);
        }

        inline fn zipRun(
            _: *Self,
            a: []const Spec.A,
            b: []const Spec.B,
            out: []Spec.Out,
            params: Spec.Parameters,
        ) iface.Error!void {
            if (a.len != b.len or a.len != out.len) return error.InvalidArgument;
            for (a, b, out) |x, y, *o| o.* = Spec.eval(x, y, params);
        }

        inline fn mapIndexedRun(_: *Self, data: []Spec.Value, params: Spec.Parameters) iface.Error!void {
            for (data, 0..) |*value, i| value.* = Spec.eval(value.*, @intCast(i), params);
        }

        /// Scalar for a custom `combine` -- a generic `fn (T, T) T` cannot be
        /// lane-widened, and float sum is not reassociative so LLVM will not do
        /// it either. The presets set `simd_op` and take the `@reduce` path,
        /// which reassociates exactly the way the device already does.
        inline fn reduceRun(
            _: *Self,
            data: []const Spec.Value,
            params: Spec.Parameters,
        ) iface.Error!Spec.Value {
            var acc: Spec.Value = Spec.identity;
            var i: usize = 0;
            if (comptime Spec.simd_op != null and vec_lanes > 1) {
                const V = @Vector(vec_lanes, Spec.Value);
                while (i + vec_lanes <= data.len) : (i += vec_lanes) {
                    var chunk: V = data[i..][0..vec_lanes].*;
                    // `pre` is scalar by construction, so it is applied per lane
                    // and only the combining tree is widened.
                    inline for (0..vec_lanes) |lane| chunk[lane] = Spec.pre(chunk[lane], params);
                    acc = Spec.combine(acc, @reduce(Spec.simd_op.?, chunk));
                }
            }
            for (data[i..]) |x| acc = Spec.combine(acc, Spec.pre(x, params));
            return acc;
        }

        inline fn gatherRun(
            _: *Self,
            src: []const Spec.Value,
            idx: []const Spec.Index,
            out: []Spec.Value,
        ) iface.Error!void {
            if (idx.len != out.len) return error.InvalidArgument;
            for (idx, out) |j, *o| if (j < src.len) {
                o.* = src[j];
            };
        }

        inline fn scatterRun(
            _: *Self,
            src: []const Spec.Value,
            idx: []const Spec.Index,
            out: []Spec.Value,
        ) iface.Error!void {
            if (src.len != idx.len) return error.InvalidArgument;
            for (src, idx) |x, j| if (j < out.len) {
                out[j] = x;
            };
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
    /// HIP entry points keep their mangled Zig name; look it up in the name map.
    mangled: bool,
    /// The generated name map. Both backends consult it -- only HIP needs the
    /// result, but the lookup is what turns a missing kernel into a compile
    /// error instead of a runtime KernelNotFound. See `openModule`.
    names: []const u8,
    /// A device arch to name in diagnostics, so the fix is copy-pasteable.
    example_cpu: []const u8,
};

pub const gpu_cuda: Gpu = .{ .backend = .cuda, .rt = cuda, .has = "has_cuda", .image = "cuda", .names = "cuda_names", .mangled = false, .example_cpu = "sm_89" };
pub const gpu_hip: Gpu = .{ .backend = .hip, .rt = hip, .has = "has_hip", .image = "hip", .names = "hip_names", .mangled = true, .example_cpu = "gfx1100" };

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

/// Open the artifact for `gpu` and resolve `entry_name` in it.
pub fn openModule(comptime gpu: Gpu, comptime entry_name: [:0]const u8, ordinal: c_int) iface.Error!struct {
    context: gpu.rt.Context,
    module: gpu.rt.Module,
    kernel: gpu.rt.Kernel,
} {
    const artifacts = @import("gompute_kernels");
    if (comptime !@field(artifacts, gpu.has)) return error.BackendUnavailable;
    const tag = @tagName(gpu.backend);

    var context = try gpu.rt.Context.init(ordinal);
    errdefer context.deinit();
    // Past this point the artifact is in the binary and the device is up, so
    // every remaining failure is a build bug. Callers are allowed to swallow the
    // error (AutoKernel does); they are not allowed to swallow the reason.
    var module = context.loadModuleFromMemory(@field(artifacts, gpu.image)) catch |err| {
        std.log.err(
            "gompute: this binary's " ++ tag ++ " artifact will not load on this device " ++
                "({t}, driver code {d}). The usual cause is a device arch mismatch -- the build " ++
                "compiled for one GPU and this machine has another. Pin the arch you deploy to " ++
                "with ." ++ tag ++ " = .{{ .gpu = .{{ .name = \"" ++ gpu.example_cpu ++
                "\" }} }} in emitKernels.",
            .{ err, iface.last_driver_error.code },
        );
        return err;
    };
    errdefer module.deinit();
    // Resolve on BOTH backends. HIP needs the mangled symbol; CUDA launches the
    // public name and throws the result away -- but the lookup itself is the
    // check: a kernel missing from the artifact is a @compileError naming it,
    // rather than a runtime KernelNotFound discovered on a customer's machine.
    // Nothing else in the pipeline verified that a requested kernel was emitted.
    const internal_name = comptime blk: {
        const resolved = @field(artifacts, gpu.names).resolve(entry_name);
        break :blk if (gpu.mangled) resolved else entry_name;
    };
    const kernel = module.getKernel(internal_name.ptr) catch |err| {
        std.log.err(
            "gompute: " ++ tag ++ " kernel \"" ++ entry_name ++ "\" is not in the emitted " ++
                "artifact ({t}, driver code {d}). Two things to check: is it listed in the " ++
                "gompute.exportKernels(.{{ ... }}) call in your kernels root, and does the " ++
                ".kernels_root you passed to emitKernels point at that same file?",
            .{ err, iface.last_driver_error.code },
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

        /// A device copy of `slice`, already uploaded. Callers must have ruled
        /// out an empty slice: a zero-byte allocation is an error to both drivers.
        fn staged(self: *Self, comptime E: type, slice: []const E) iface.Error!Buffer {
            const bytes = slice.len * @sizeOf(E);
            var buffer = try self.context.alloc(bytes);
            errdefer buffer.free();
            try buffer.upload(slice.ptr, bytes);
            return buffer;
        }

        /// A worker thread that has not touched this device yet has no current
        /// context; without this every launch off the main thread fails.
        fn dispatch(self: *Self, grid: iface.Dim3, args: []const iface.Arg) iface.Error!void {
            try self.context.makeCurrent();
            try self.kernel.launch(grid, .{ .x = Spec.block_size }, 0, args);
        }

        /// Same story as `CpuKernel.run`: one signature per kind, chosen at
        /// compile time, nothing left to branch on at run time.
        pub const launch = switch (spec.kindOf(Spec)) {
            .map, .map_indexed => mapLaunch,
            .map_to => mapToLaunch,
            .zip => zipLaunch,
            .reduce => reduceLaunch,
            .gather, .scatter => indexedCopyLaunch,
        };

        pub const run = switch (spec.kindOf(Spec)) {
            .map, .map_indexed => mapRun,
            .map_to => mapToRun,
            .zip => zipRun,
            .reduce => reduceRun,
            .gather => gatherRun,
            .scatter => scatterRun,
        };

        fn mapLaunch(
            self: *Self,
            buffer: *Buffer,
            count: usize,
            params: Spec.Parameters,
        ) iface.Error!void {
            if (count == 0) return;
            var len: u64 = @intCast(count);
            var packed_params = abi.pack(Spec.Parameters, params);
            var args = [_]iface.Arg{
                buffer.argPtr(),
                iface.arg(&len),
                iface.arg(&packed_params),
            };
            try self.dispatch(iface.Dim3.linear(count, Spec.block_size), &args);
        }

        fn mapRun(self: *Self, data: []Spec.Value, params: Spec.Parameters) iface.Error!void {
            if (data.len == 0) return;
            var buffer = try self.staged(Spec.Value, data);
            defer buffer.free();
            try self.launch(&buffer, data.len, params);
            try self.context.synchronize();
            try buffer.download(data.ptr, data.len * @sizeOf(Spec.Value));
        }

        fn mapToLaunch(
            self: *Self,
            in: *Buffer,
            out: *Buffer,
            count: usize,
            params: Spec.Parameters,
        ) iface.Error!void {
            if (count == 0) return;
            var len: u64 = @intCast(count);
            var packed_params = abi.pack(Spec.Parameters, params);
            var args = [_]iface.Arg{
                in.argPtr(),
                out.argPtr(),
                iface.arg(&len),
                iface.arg(&packed_params),
            };
            try self.dispatch(iface.Dim3.linear(count, Spec.block_size), &args);
        }

        fn mapToRun(
            self: *Self,
            in: []const Spec.In,
            out: []Spec.Out,
            params: Spec.Parameters,
        ) iface.Error!void {
            if (in.len != out.len) return error.InvalidArgument;
            if (in.len == 0) return;
            var in_buf = try self.staged(Spec.In, in);
            defer in_buf.free();
            var out_buf = try self.context.alloc(out.len * @sizeOf(Spec.Out));
            defer out_buf.free();
            try self.launch(&in_buf, &out_buf, in.len, params);
            try self.context.synchronize();
            try out_buf.download(out.ptr, out.len * @sizeOf(Spec.Out));
        }

        fn zipLaunch(
            self: *Self,
            a: *Buffer,
            b: *Buffer,
            out: *Buffer,
            count: usize,
            params: Spec.Parameters,
        ) iface.Error!void {
            if (count == 0) return;
            var len: u64 = @intCast(count);
            var packed_params = abi.pack(Spec.Parameters, params);
            var args = [_]iface.Arg{
                a.argPtr(),
                b.argPtr(),
                out.argPtr(),
                iface.arg(&len),
                iface.arg(&packed_params),
            };
            try self.dispatch(iface.Dim3.linear(count, Spec.block_size), &args);
        }

        fn zipRun(
            self: *Self,
            a: []const Spec.A,
            b: []const Spec.B,
            out: []Spec.Out,
            params: Spec.Parameters,
        ) iface.Error!void {
            if (a.len != b.len or a.len != out.len) return error.InvalidArgument;
            if (a.len == 0) return;
            var a_buf = try self.staged(Spec.A, a);
            defer a_buf.free();
            var b_buf = try self.staged(Spec.B, b);
            defer b_buf.free();
            var out_buf = try self.context.alloc(out.len * @sizeOf(Spec.Out));
            defer out_buf.free();
            try self.launch(&a_buf, &b_buf, &out_buf, a.len, params);
            try self.context.synchronize();
            try out_buf.download(out.ptr, out.len * @sizeOf(Spec.Out));
        }

        fn reduceLaunch(
            self: *Self,
            data: *Buffer,
            count: usize,
            partials: *Buffer,
            blocks: u32,
            params: Spec.Parameters,
        ) iface.Error!void {
            if (count == 0 or blocks == 0) return;
            var len: u64 = @intCast(count);
            // The device cannot portably ask how big its own grid is, so tell it.
            var stride: u64 = @as(u64, blocks) * Spec.block_size;
            var packed_params = abi.pack(Spec.Parameters, params);
            var args = [_]iface.Arg{
                data.argPtr(),
                iface.arg(&len),
                partials.argPtr(),
                iface.arg(&stride),
                iface.arg(&packed_params),
            };
            // A fixed grid, not one block per element: the kernel is a
            // grid-stride loop, and the cap is what keeps the final fold small.
            try self.dispatch(.{ .x = blocks }, &args);
        }

        /// How many blocks `count` elements get. Saturates at
        /// `max_reduce_blocks`; the grid-stride loop covers the rest.
        pub fn reduceBlocks(count: usize) u32 {
            const covering = (count + Spec.block_size - 1) / Spec.block_size;
            return @intCast(@min(covering, max_reduce_blocks));
        }

        fn reduceRun(
            self: *Self,
            data: []const Spec.Value,
            params: Spec.Parameters,
        ) iface.Error!Spec.Value {
            if (data.len == 0) return Spec.identity;
            const blocks = reduceBlocks(data.len);

            var data_buf = try self.staged(Spec.Value, data);
            defer data_buf.free();
            var partial_buf = try self.context.alloc(blocks * @sizeOf(Spec.Value));
            defer partial_buf.free();
            try self.launch(&data_buf, data.len, &partial_buf, blocks, params);
            try self.context.synchronize();

            var partials: [max_reduce_blocks]Spec.Value = undefined;
            try partial_buf.download(&partials, blocks * @sizeOf(Spec.Value));
            var acc: Spec.Value = Spec.identity;
            for (partials[0..blocks]) |partial| acc = Spec.combine(acc, partial);
            return acc;
        }

        fn indexedCopyLaunch(
            self: *Self,
            src: *Buffer,
            idx: *Buffer,
            out: *Buffer,
            count: usize,
            bound: usize,
        ) iface.Error!void {
            if (count == 0) return;
            var len: u64 = @intCast(count);
            var limit: u64 = @intCast(bound);
            var args = [_]iface.Arg{
                src.argPtr(),
                idx.argPtr(),
                out.argPtr(),
                iface.arg(&len),
                iface.arg(&limit),
            };
            try self.dispatch(iface.Dim3.linear(count, Spec.block_size), &args);
        }

        fn gatherRun(
            self: *Self,
            src: []const Spec.Value,
            idx: []const Spec.Index,
            out: []Spec.Value,
        ) iface.Error!void {
            if (idx.len != out.len) return error.InvalidArgument;
            if (out.len == 0 or src.len == 0) return;
            var src_buf = try self.staged(Spec.Value, src);
            defer src_buf.free();
            var idx_buf = try self.staged(Spec.Index, idx);
            defer idx_buf.free();
            var out_buf = try self.context.alloc(out.len * @sizeOf(Spec.Value));
            defer out_buf.free();
            try self.launch(&src_buf, &idx_buf, &out_buf, out.len, src.len);
            try self.context.synchronize();
            try out_buf.download(out.ptr, out.len * @sizeOf(Spec.Value));
        }

        fn scatterRun(
            self: *Self,
            src: []const Spec.Value,
            idx: []const Spec.Index,
            out: []Spec.Value,
        ) iface.Error!void {
            if (src.len != idx.len) return error.InvalidArgument;
            if (src.len == 0 or out.len == 0) return;
            var src_buf = try self.staged(Spec.Value, src);
            defer src_buf.free();
            var idx_buf = try self.staged(Spec.Index, idx);
            defer idx_buf.free();
            // Uploaded, not just allocated: an element of `out` that no index
            // selects must keep the value the caller put there.
            var out_buf = try self.staged(Spec.Value, out);
            defer out_buf.free();
            try self.launch(&src_buf, &idx_buf, &out_buf, src.len, out.len);
            try self.context.synchronize();
            try out_buf.download(out.ptr, out.len * @sizeOf(Spec.Value));
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

        /// Forwards to whichever backend `init` picked. One arm per kind
        /// because the argument list is the kind: all three backends agree on
        /// the signature, so the body is the same `inline else` every time.
        pub const run = switch (spec.kindOf(Spec)) {
            .map, .map_indexed => struct {
                fn run(self: *Self, data: []Spec.Value, params: Spec.Parameters) iface.Error!void {
                    switch (self.*) {
                        inline else => |*k| return k.run(data, params),
                    }
                }
            }.run,
            .map_to => struct {
                fn run(
                    self: *Self,
                    in: []const Spec.In,
                    out: []Spec.Out,
                    params: Spec.Parameters,
                ) iface.Error!void {
                    switch (self.*) {
                        inline else => |*k| return k.run(in, out, params),
                    }
                }
            }.run,
            .zip => struct {
                fn run(
                    self: *Self,
                    a: []const Spec.A,
                    b: []const Spec.B,
                    out: []Spec.Out,
                    params: Spec.Parameters,
                ) iface.Error!void {
                    switch (self.*) {
                        inline else => |*k| return k.run(a, b, out, params),
                    }
                }
            }.run,
            .reduce => struct {
                fn run(
                    self: *Self,
                    data: []const Spec.Value,
                    params: Spec.Parameters,
                ) iface.Error!Spec.Value {
                    switch (self.*) {
                        inline else => |*k| return k.run(data, params),
                    }
                }
            }.run,
            .gather, .scatter => struct {
                fn run(
                    self: *Self,
                    src: []const Spec.Value,
                    idx: []const Spec.Index,
                    out: []Spec.Value,
                ) iface.Error!void {
                    switch (self.*) {
                        inline else => |*k| return k.run(src, idx, out),
                    }
                }
            }.run,
        };

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

// ---------------------------------------------------------------------------
// CPU paths for the rest of the operation set.
//
// Every sweep runs the same lengths: 0, 1, either side of the vector width the
// reduce presets use, and either side of a small block_size -- the two places a
// tail or a partially-filled block can go wrong. The GPU paths are checked on
// real hardware by the consumer under examples/, not here; these pin the
// reference semantics both backends have to agree on.
// ---------------------------------------------------------------------------

const TestParams = struct { scale: f32 };

/// 8 is a block_size, so `block_size ± 1` lands inside the sweep, and it is
/// not a multiple of any suggested vector length on x86 or aarch64.
const sweep_lengths = [_]usize{ 0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65 };

test "mapTo writes a different element type without touching the input" {
    const Spec = spec.MapTo("t_map_to", f32, u32, TestParams, struct {
        fn call(x: f32, p: TestParams) u32 {
            return @intFromFloat(@max(x * p.scale, 0));
        }
    }.call, .{ .block_size = 8 });

    var kernel = try CpuKernel(Spec).init(0);
    var in: [65]f32 = undefined;
    var out: [65]u32 = undefined;
    for (sweep_lengths) |len| {
        for (in[0..len], 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) - 8;
        @memset(out[0..len], 0xdead);
        try kernel.run(in[0..len], out[0..len], .{ .scale = 2 });
        for (in[0..len], out[0..len]) |x, y|
            try std.testing.expectEqual(@as(u32, @intFromFloat(@max(x * 2, 0))), y);
    }
    // A length disagreement is the caller's bug, not a truncated run.
    try std.testing.expectError(error.InvalidArgument, kernel.run(in[0..4], out[0..3], .{ .scale = 1 }));
}

test "zip reads two buffers positionally" {
    const Spec = spec.Zip("t_zip", f32, f32, f32, TestParams, struct {
        fn call(a: f32, b: f32, p: TestParams) f32 {
            return a + b * p.scale;
        }
    }.call, .{ .block_size = 8 });

    var kernel = try CpuKernel(Spec).init(0);
    var a: [65]f32 = undefined;
    var b: [65]f32 = undefined;
    var out: [65]f32 = undefined;
    for (sweep_lengths) |len| {
        for (a[0..len], b[0..len], 0..) |*x, *y, i| {
            x.* = @floatFromInt(i);
            y.* = @as(f32, @floatFromInt(i)) * -0.5;
        }
        try kernel.run(a[0..len], b[0..len], out[0..len], .{ .scale = 3 });
        for (a[0..len], b[0..len], out[0..len]) |x, y, o|
            try std.testing.expectEqual(x + y * 3, o);
    }
    try std.testing.expectError(
        error.InvalidArgument,
        kernel.run(a[0..4], b[0..3], out[0..4], .{ .scale = 1 }),
    );
}

test "mapIndexed sees the linear index" {
    const Spec = spec.MapIndexed("t_map_indexed", f32, TestParams, struct {
        fn call(x: f32, i: u64, p: TestParams) f32 {
            return x + @as(f32, @floatFromInt(i % 8)) * p.scale;
        }
    }.call, .{ .block_size = 8 });

    var kernel = try CpuKernel(Spec).init(0);
    var data: [65]f32 = undefined;
    for (sweep_lengths) |len| {
        for (data[0..len], 0..) |*v, i| v.* = @floatFromInt(i);
        try kernel.run(data[0..len], .{ .scale = 10 });
        for (data[0..len], 0..) |v, i|
            try std.testing.expectEqual(
                @as(f32, @floatFromInt(i)) + @as(f32, @floatFromInt(i % 8)) * 10,
                v,
            );
    }
}

test "reduce folds with a custom combine and returns the identity when empty" {
    // Integer product: exact, so a reassociating backend must agree bit for bit.
    const Spec = spec.Reduce("t_reduce", u32, TestParams, struct {
        fn call(a: u32, b: u32) u32 {
            return a *% b;
        }
    }.call, 1, .{ .block_size = 8 });
    try std.testing.expectEqual(@as(?std.builtin.ReduceOp, null), Spec.simd_op);

    var kernel = try CpuKernel(Spec).init(0);
    var data: [65]u32 = undefined;
    for (sweep_lengths) |len| {
        var expected: u32 = 1;
        for (data[0..len], 0..) |*v, i| {
            v.* = @intCast(i % 5 + 1);
            expected *%= v.*;
        }
        try std.testing.expectEqual(expected, try kernel.run(data[0..len], .{ .scale = 0 }));
    }
    try std.testing.expectEqual(@as(u32, 1), try kernel.run(&.{}, .{ .scale = 0 }));
}

test "reduce presets take the @reduce path and agree with the scalar fold" {
    const Sum = spec.Sum("t_sum", i32, TestParams, .{ .block_size = 8 });
    const Min = spec.Min("t_min", i32, TestParams, .{ .block_size = 8 });
    const Max = spec.Max("t_max", i32, TestParams, .{ .block_size = 8 });
    const Any = spec.Any("t_any", u32, TestParams, .{ .block_size = 8 });
    const All = spec.All("t_all", u32, TestParams, .{ .block_size = 8 });

    try std.testing.expectEqual(std.builtin.ReduceOp.Add, Sum.simd_op.?);
    try std.testing.expectEqual(@as(i32, std.math.maxInt(i32)), Min.identity);
    try std.testing.expectEqual(@as(i32, std.math.minInt(i32)), Max.identity);
    try std.testing.expectEqual(@as(u32, 0), Any.identity);
    try std.testing.expectEqual(~@as(u32, 0), All.identity);

    var sum_k = try CpuKernel(Sum).init(0);
    var min_k = try CpuKernel(Min).init(0);
    var max_k = try CpuKernel(Max).init(0);
    var any_k = try CpuKernel(Any).init(0);
    var all_k = try CpuKernel(All).init(0);
    const p: TestParams = .{ .scale = 0 };

    var signed: [65]i32 = undefined;
    var bits: [65]u32 = undefined;
    for (sweep_lengths) |len| {
        var want_sum: i32 = 0;
        var want_min: i32 = std.math.maxInt(i32);
        var want_max: i32 = std.math.minInt(i32);
        var want_any: u32 = 0;
        var want_all: u32 = ~@as(u32, 0);
        for (signed[0..len], bits[0..len], 0..) |*s, *b, i| {
            s.* = @as(i32, @intCast(i)) * (if (i % 3 == 0) @as(i32, -1) else 1);
            b.* = if (i % 4 == 0) 0 else ~@as(u32, 0);
            want_sum +%= s.*;
            want_min = @min(want_min, s.*);
            want_max = @max(want_max, s.*);
            want_any |= b.*;
            want_all &= b.*;
        }
        try std.testing.expectEqual(want_sum, try sum_k.run(signed[0..len], p));
        try std.testing.expectEqual(want_min, try min_k.run(signed[0..len], p));
        try std.testing.expectEqual(want_max, try max_k.run(signed[0..len], p));
        try std.testing.expectEqual(want_any, try any_k.run(bits[0..len], p));
        try std.testing.expectEqual(want_all, try all_k.run(bits[0..len], p));
    }
}

test "reduce pre is transform_reduce: sum of squares in one pass" {
    const SumSq = spec.Sum("t_sumsq", f64, TestParams, .{
        .block_size = 8,
        .pre = &struct {
            fn square(x: f64, p: TestParams) f64 {
                return x * x * p.scale;
            }
        }.square,
    });
    var kernel = try CpuKernel(SumSq).init(0);
    var data: [65]f64 = undefined;
    for (sweep_lengths) |len| {
        var expected: f64 = 0;
        for (data[0..len], 0..) |*v, i| {
            v.* = @floatFromInt(i);
            expected += v.* * v.* * 2;
        }
        const got = try kernel.run(data[0..len], .{ .scale = 2 });
        try std.testing.expectApproxEqRel(expected, got, 1e-12);
    }
}

test "gather and scatter copy through an index buffer" {
    const Gather = spec.Gather("t_gather", f32, u32, .{ .block_size = 8 });
    const Scatter = spec.Scatter("t_scatter", f32, u32, .{ .block_size = 8 });
    var gather_k = try CpuKernel(Gather).init(0);
    var scatter_k = try CpuKernel(Scatter).init(0);

    var src: [65]f32 = undefined;
    var idx: [65]u32 = undefined;
    var out: [65]f32 = undefined;
    for (sweep_lengths) |len| {
        if (len == 0) continue;
        for (src[0..len], idx[0..len], 0..) |*s, *j, i| {
            s.* = @floatFromInt(i);
            j.* = @intCast(len - 1 - i); // reverse
        }
        @memset(out[0..len], -1);
        try gather_k.run(src[0..len], idx[0..len], out[0..len]);
        for (out[0..len], 0..) |o, i|
            try std.testing.expectEqual(@as(f32, @floatFromInt(len - 1 - i)), o);

        @memset(out[0..len], -1);
        try scatter_k.run(src[0..len], idx[0..len], out[0..len]);
        for (out[0..len], 0..) |o, i|
            try std.testing.expectEqual(@as(f32, @floatFromInt(len - 1 - i)), o);
    }

    // Empty is a no-op on both, and a length disagreement is an error.
    try gather_k.run(src[0..4], idx[0..0], out[0..0]);
    try scatter_k.run(src[0..0], idx[0..0], out[0..4]);
    try std.testing.expectError(error.InvalidArgument, gather_k.run(src[0..4], idx[0..3], out[0..4]));
    try std.testing.expectError(error.InvalidArgument, scatter_k.run(src[0..4], idx[0..3], out[0..4]));
}

test "an index past the end is skipped, never a write outside the buffer" {
    const Gather = spec.Gather("t_gather_oob", f32, u32, .{});
    const Scatter = spec.Scatter("t_scatter_oob", f32, u32, .{});
    var gather_k = try CpuKernel(Gather).init(0);
    var scatter_k = try CpuKernel(Scatter).init(0);

    const src = [_]f32{ 10, 20, 30 };
    const idx = [_]u32{ 2, 99, 0 };
    var out = [_]f32{ -1, -1, -1 };
    try gather_k.run(&src, &idx, &out);
    // Slot 1's index is out of range: unspecified by contract, and on the CPU
    // that means untouched. The neighbours must still be right.
    try std.testing.expectEqual(@as(f32, 30), out[0]);
    try std.testing.expectEqual(@as(f32, 10), out[2]);

    out = .{ -1, -1, -1 };
    try scatter_k.run(&src, &idx, &out);
    try std.testing.expectEqual(@as(f32, 30), out[0]);
    try std.testing.expectEqual(@as(f32, 10), out[2]);
    // Nothing targeted slot 1, so it keeps what the caller left there.
    try std.testing.expectEqual(@as(f32, -1), out[1]);
}

test "reduce grid saturates at the block cap that keeps the host fold small" {
    const Spec = spec.Sum("t_blocks", f32, TestParams, .{ .block_size = 256 });
    const K = GpuKernel(Spec, gpu_cuda);
    try std.testing.expectEqual(@as(u32, 1), K.reduceBlocks(1));
    try std.testing.expectEqual(@as(u32, 1), K.reduceBlocks(256));
    try std.testing.expectEqual(@as(u32, 2), K.reduceBlocks(257));
    try std.testing.expectEqual(@as(u32, max_reduce_blocks), K.reduceBlocks(256 * 1024));
    // Past the cap the grid stops growing; the kernel's grid-stride loop covers
    // the rest, which is the whole reason the fold buffer can be fixed-size.
    try std.testing.expectEqual(@as(u32, max_reduce_blocks), K.reduceBlocks(1 << 30));
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
