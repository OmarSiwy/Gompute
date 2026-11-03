//! Hand-written GPU kernels: loading the artifact `build.zig` emitted, and
//! launching an entry point out of it by name.

const std = @import("std");
const iface = @import("../core/interface.zig");
const cuda = @import("../runtime/cuda.zig");
const hip = @import("../runtime/hip.zig");

/// Where a kernel runs. `.cpu` always exists; the GPU tags need the matching
/// backend to have been emitted by build.zig.
pub const Backend = enum { cpu, cuda, hip };

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

/// The two descriptors that exist. Every generic here is instantiated with one
/// of them, and `gpuOf` is how a `Backend` tag picks.
pub const gpu_cuda: Gpu = .{ .backend = .cuda, .rt = cuda, .has = "has_cuda", .images = "cuda_images", .index = "cuda_index", .example_cpu = "sm_89" };
pub const gpu_hip: Gpu = .{ .backend = .hip, .rt = hip, .has = "has_hip", .images = "hip_images", .index = "hip_index", .example_cpu = "gfx1100" };

/// The descriptor for a backend tag, for the device-only entry points below.
/// `what` names the caller so the `.cpu` diagnostic can say which one
/// was device-only, rather than just that the tag was wrong.
pub fn gpuOf(comptime backend: Backend, comptime what: []const u8) Gpu {
    return switch (backend) {
        .cuda => gpu_cuda,
        .hip => gpu_hip,
        .cpu => @compileError(what ++ " is device-only; use a normal Zig function on CPU"),
    };
}

/// Whether this build actually carries a device artifact for `gpu`: the
/// question to ask before naming `RawKernel(name, .cuda)`, which is a compile
/// error when the answer is false. Public as `gompute.available(.cuda)`.
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
        "gompute: RawKernel(name, ." ++ tag ++ ") needs device artifacts, but this build never " ++
            "emitted any. Add to build.zig:\n" ++
            "    const gompute_build = @import(\"gompute\");\n" ++
            "    gompute_build.emitKernels(b, dep, exe, .{ .kernels_root = b.path(\"src/kernels.zig\") });\n" ++
            "(or gompute_build.addKernels for more than one executable).",
    );
    @compileError(
        "gompute: RawKernel(name, ." ++ tag ++ ") was asked for, but this build emitted no " ++ tag ++
            " artifacts. Either .auto detected no " ++ tag ++ " GPU on the BUILD machine, or you " ++
            "passed ." ++ tag ++ " = .{ .enabled = false }. Fix it by pinning the arch you deploy " ++
            "to -- ." ++ tag ++ " = .{ .gpu = .{ .name = \"" ++ gpu.example_cpu ++ "\" } } in " ++
            "emitKernels -- or guard the call with `if (comptime gompute.available(." ++ tag ++ "))`.",
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
            "`comptime { if (g.is_device) g.exportRaw(\"" ++ entry_name ++ "\", &fn); }`, and check " ++
            "that the root is listed in " ++
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
    // error; they are not allowed to swallow the reason.
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
                "artifact ({t}, driver code {d}). Two things to check: does your kernels root " ++
                "export it with gompute.exportRaw, and does the " ++
                "root you passed to emitKernels point at that same file?",
            .{ name, err, iface.last_driver_error.code },
        );
        return err;
    };
    return .{ .context = context, .module = module, .kernel = kernel };
}

/// A hand-written device entry point, loaded by the name it was exported under.
/// Device-only: naming `.cpu` is a compile error, because there is nothing for
/// this to load and a plain Zig function is what you want instead.
pub fn RawKernel(comptime entry_name: [:0]const u8, comptime backend: Backend) type {
    const gpu = gpuOf(backend, "RawKernel");
    return requireArtifacts(gpu, GpuRaw(entry_name, gpu));
}

/// The handle `rawKernelByName` returns: a `RawKernel` whose kernel was picked
/// at run time, so it has no comptime `init` of its own.
pub fn RawByName(comptime backend: Backend) type {
    const gpu = gpuOf(backend, "kernel-by-name");
    return requireArtifacts(gpu, GpuRaw(null, gpu));
}

/// Load the kernel called `name`, decided at run time -- from a parsed netlist,
/// a config file, a CLI flag. The set of kernels is closed at build time, so an
/// unknown name is `error.KernelNotFound` rather than a panic, and only the one
/// root that exports it is JIT'd.
pub fn rawKernelByName(
    comptime backend: Backend,
    name: []const u8,
    ordinal: c_int,
) iface.Error!RawByName(backend) {
    const gpu = comptime gpuOf(backend, "kernel-by-name");
    const opened = try openModuleByName(gpu, name, ordinal);
    return .{ .context = opened.context, .module = opened.module, .kernel = opened.kernel };
}

/// `entry_name` is null for a handle whose kernel is chosen at run time.
fn GpuRaw(comptime entry_name: ?[:0]const u8, comptime gpu: Gpu) type {
    return struct {
        const Self = @This();
        /// Device memory, valid on every handle for the same device.
        pub const Buffer = gpu.rt.Buffer;
        /// The queue type `createStream` returns and `launchOn` takes.
        pub const Stream = gpu.rt.Stream;
        /// What `Stream.endCapture` records; `instantiate` it to replay.
        pub const Graph = gpu.rt.Graph;
        /// An instantiated `Graph`: `launch` it on a stream, `update` it in place.
        pub const GraphExec = gpu.rt.GraphExec;
        /// The type `createEvent` returns.
        pub const Event = gpu.rt.Event;
        /// `Stream.beginCapture`'s argument.
        pub const CaptureMode = gpu.rt.CaptureMode;

        context: gpu.rt.Context = .{},
        module: gpu.rt.Module = .{},
        kernel: gpu.rt.Kernel = .{},

        /// Handles on the same device share one primary context and one
        /// JIT'd copy of the artifact, so this is cheap after the first one and
        /// a `Buffer` from any handle is valid in all of them.
        pub fn init(ordinal: c_int) iface.Error!Self {
            const name = entry_name orelse
                @compileError("gompute: this handle came from rawKernelByName, which already " ++
                    "loaded its kernel; there is nothing for init() to name.");
            const opened = try openModule(gpu, name, ordinal);
            return .{ .context = opened.context, .module = opened.module, .kernel = opened.kernel };
        }

        /// Drops this handle only; the shared context and module stay up for
        /// the process. See `gompute.runtime.<backend>.shutdown()`.
        ///
        /// Resets rather than `undefined`, so a stray double-deinit stays a
        /// no-op instead of a driver-level double free in ReleaseFast.
        pub fn deinit(self: *Self) void {
            self.module.deinit();
            self.context.deinit();
            self.* = .{};
        }

        /// `bytes`, not elements: nothing here knows the kernel's element type.
        ///
        /// Caller owns the returned buffer and must `free` it.
        pub fn alloc(self: *Self, bytes: usize) iface.Error!Buffer {
            return self.context.alloc(bytes);
        }

        /// Page-locked host memory. Driver memory, not a Zig allocator's, so it
        /// is released with `freePinned`; see `Context.allocPinned` for why the
        /// async copies need it.
        ///
        /// Caller owns the returned block.
        pub fn allocPinned(self: *Self, bytes: usize) iface.Error![]u8 {
            return self.context.allocPinned(bytes);
        }

        /// Releases a block from `allocPinned`, and nothing else -- passing a
        /// Zig allocator's memory here hands the driver a pointer it never gave
        /// out. A no-op on an empty slice or before the driver has loaded.
        pub fn freePinned(self: *Self, bytes: []u8) void {
            self.context.freePinned(bytes);
        }

        /// A stream to order copies and launches on. Anything issued here stays
        /// off the NULL stream, which implicitly synchronizes against every
        /// other blocking stream on the device.
        ///
        /// Caller owns the returned stream and must `deinit` it.
        pub fn createStream(self: *Self) iface.Error!Stream {
            return self.context.createStream();
        }

        /// See `Context.hasGraphs`: false on a driver too old for graphs.
        pub fn hasGraphs(self: *const Self) bool {
            return self.context.hasGraphs();
        }

        /// See `Context.hasEvents`.
        pub fn hasEvents(self: *const Self) bool {
            return self.context.hasEvents();
        }

        /// A stream that does not synchronize with the NULL stream, which is
        /// the kind a capture needs. Caller owns it and must `deinit` it.
        pub fn createStreamNonBlocking(self: *Self) iface.Error!Stream {
            return self.context.createStreamNonBlocking();
        }

        /// `timing` false makes an event that orders but cannot be timed.
        /// Caller owns it and must `deinit` it.
        pub fn createEvent(self: *Self, timing: bool) iface.Error!Event {
            return self.context.createEvent(timing);
        }

        /// Launch on the NULL stream and return once the launch is queued. The
        /// grid and block are yours to pick: nothing here knows the kernel's
        /// shape. Pass `globalIdX` the same block size.
        pub fn launch(
            self: *Self,
            grid: iface.Dim3,
            block: iface.Dim3,
            shared_bytes: u32,
            args: []const iface.Arg,
        ) iface.Error!void {
            // A worker thread starts with no current context; make this device current.
            try self.context.makeCurrent();
            return self.kernel.launch(grid, block, shared_bytes, args);
        }

        /// `launch`, ordered on `stream` instead of the NULL stream. Returns as
        /// soon as the launch is queued, so `args` -- which are pointers to the
        /// caller's argument storage -- must outlive the call.
        pub fn launchOn(
            self: *Self,
            stream: *Stream,
            grid: iface.Dim3,
            block: iface.Dim3,
            shared_bytes: u32,
            args: []const iface.Arg,
        ) iface.Error!void {
            // Same reason as `launch`: a worker thread has no current context.
            try self.context.makeCurrent();
            return self.kernel.launchOnStream(grid, block, shared_bytes, args, stream.stream);
        }

        /// Block until every launch and copy queued on this device is done.
        pub fn synchronize(self: *Self) iface.Error!void {
            return self.context.synchronize();
        }
    };
}

// ---------------------------------------------------------------------------
// Seams. None of this needs a GPU, and none of it can go through `RawKernel`:
// the library's own test build emits no artifacts, so `requireArtifacts` makes
// every public entry point here a compile error. `GpuRaw` is reachable, and
// naming it is most of the point -- these tests are what pulls this file's
// bodies through semantic analysis at all: `root.zig` reaches this file only
// through lazily analyzed decls.
// ---------------------------------------------------------------------------

test GpuRaw {
    inline for (.{ gpu_cuda, gpu_hip }) |gpu| {
        const Raw = GpuRaw("gompute_seam_probe", gpu);
        std.testing.refAllDecls(Raw);
        try std.testing.expect(Raw.Buffer == gpu.rt.Buffer);
        try std.testing.expect(Raw.Stream == gpu.rt.Stream);

        // A reset handle's methods are driver-level no-ops; see `deinit`.
        var raw: Raw = .{};
        var empty: [0]u8 = .{};
        raw.freePinned(&empty);
        raw.deinit();
    }
}

test "the async surface still lines up with the blocking one" {
    // `Stream`, `createStream`, `allocPinned`/`freePinned` and `launchOn` all
    // landed in one commit and nothing calls them yet. Until something does,
    // this is the oracle: the pairs have to keep matching, or whoever finishes
    // the overlap path inherits three mismatches at once.
    inline for (.{ gpu_cuda, gpu_hip }) |gpu| {
        const Raw = GpuRaw("gompute_seam_probe", gpu);

        const stream = @typeInfo(@TypeOf(Raw.createStream)).@"fn".return_type.?;
        try std.testing.expect(@typeInfo(stream).error_union.payload == Raw.Stream);

        // Driver memory: `freePinned` is the only legal way to release it, so it
        // has to accept exactly what `allocPinned` hands back.
        const pinned = @typeInfo(@TypeOf(Raw.allocPinned)).@"fn".return_type.?;
        try std.testing.expect(@typeInfo(pinned).error_union.payload ==
            @typeInfo(@TypeOf(Raw.freePinned)).@"fn".param_types[1].?);

        // `launchOn` is `launch` with a stream spliced in after `self`.
        const blocking = @typeInfo(@TypeOf(Raw.launch)).@"fn".param_types;
        const ordered = @typeInfo(@TypeOf(Raw.launchOn)).@"fn".param_types;
        try std.testing.expectEqual(blocking.len + 1, ordered.len);
        try std.testing.expect(ordered[1].? == *Raw.Stream);
        inline for (blocking[1..], ordered[2..]) |a, b|
            try std.testing.expect(a.? == b.?);
    }
}
