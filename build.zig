const std = @import("std");

pub const Gpu = union(enum) {
    /// Detect the GPU on the BUILD machine; the backend is compiled out (with a
    /// warning) if none is found. Not the deploy machine -- pin `.name` in CI,
    /// Docker, Nix and releases.
    auto,
    /// Explicit target CPU accepted by Zig, e.g. `sm_89` (CUDA) or `gfx1100` (HIP).
    name: []const u8,
};

pub const CudaOptions = struct {
    enabled: bool = true,
    gpu: Gpu = .auto,
    /// Optimize mode for CUDA device code. Overrides `EmitOptions.optimize`.
    optimize: ?std.builtin.OptimizeMode = null,
};

pub const HipOptions = struct {
    enabled: bool = true,
    gpu: Gpu = .auto,
    /// Optimize mode for HIP device code. Overrides `EmitOptions.optimize`.
    optimize: ?std.builtin.OptimizeMode = null,
};

/// ponytail: nvidia-smi ships with every NVIDIA driver; querying it is the
/// lightest reliable probe. Compute cap "8.9" maps directly to "sm_89".
fn detectCudaGpu(b: *std.Build) ?[]const u8 {
    var code: u8 = undefined;
    const out = b.runAllowFail(
        &.{ "nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader" },
        &code,
        .ignore,
    ) catch return null;
    const line = std.mem.trim(u8, std.mem.sliceTo(out, '\n'), " \r\t");
    // nvidia-smi can exit 0 and still print "[N/A]" or "[Not Supported]" for a
    // card it cannot report. Passing that through produced "sm_[N/A]", which
    // reached Target.Query.parse and killed the build. An unreadable capability
    // means "no usable GPU detected" -- the same answer as no nvidia-smi at all.
    if (!isComputeCap(line)) return null;
    var buf: std.ArrayList(u8) = .empty;
    buf.appendSlice(b.allocator, "sm_") catch @panic("OOM");
    for (line) |c| if (c != '.') buf.append(b.allocator, c) catch @panic("OOM");
    return buf.items;
}

/// A compute capability is exactly `<digits>.<digits>`, e.g. "8.9".
fn isComputeCap(s: []const u8) bool {
    const dot = std.mem.indexOfScalar(u8, s, '.') orelse return false;
    const major = s[0..dot];
    const minor = s[dot + 1 ..];
    if (major.len == 0 or minor.len == 0) return false;
    for (major) |c| if (!std.ascii.isDigit(c)) return false;
    for (minor) |c| if (!std.ascii.isDigit(c)) return false;
    return true;
}

fn detectHipGpu(b: *std.Build) ?[]const u8 {
    for ([_][]const []const u8{
        &.{"amdgpu-arch"},
        &.{"rocm_agent_enumerator"},
    }) |argv| {
        var code: u8 = undefined;
        const out = b.runAllowFail(argv, &code, .ignore) catch continue;
        var it = std.mem.tokenizeAny(u8, out, " \r\n\t");
        while (it.next()) |arch| {
            // gfx000 is the CPU agent reported by rocm_agent_enumerator.
            if (std.mem.startsWith(u8, arch, "gfx") and !std.mem.eql(u8, arch, "gfx000"))
                return arch;
        }
    }
    return null;
}

/// Resolve one backend's device CPU, shouting when `.auto` comes up empty.
///
/// `.auto` probes the BUILD machine. A failed probe used to produce a green
/// build whose GPU backend was quietly compiled out -- invisible until the
/// deploy machine hit `error.BackendUnavailable`. Detection failing is normal
/// in CI/Docker/Nix, so it must be loud rather than fatal.
fn resolveGpu(
    b: *std.Build,
    comptime backend: []const u8,
    comptime example: []const u8,
    gpu: Gpu,
    detect: fn (*std.Build) ?[]const u8,
) ?[]const u8 {
    switch (gpu) {
        .name => |n| return n,
        .auto => {
            if (detect(b)) |n| return n;
            std.log.warn(
                "gompute: .auto found no " ++ backend ++ " GPU on this BUILD machine, so the " ++
                    backend ++ " backend is compiled out. Kernel(spec, ." ++ backend ++
                    ") is now a compile error, and AutoKernel will fall back to the CPU on " ++
                    "every deploy machine no matter what hardware it has. " ++
                    "Pin it with ." ++ backend ++ " = .{{ .gpu = .{{ .name = \"" ++ example ++
                    "\" }} }}, or silence this with ." ++ backend ++ " = .{{ .enabled = false }}.",
                .{},
            );
            return null;
        },
    }
}

/// Debug device code drags std.builtin panic globals into the module and
/// LLVM's NVPTX backend emits invalid PTX types (.u2/.u4/.u5) for them.
/// ReleaseFast removes the panic machinery outright rather than optimizing it,
/// so it serves that goal strictly better than ReleaseSafe -- and far cheaper:
/// ReleaseSafe keeps a panic edge per operation, which sends the LLVM pipeline
/// superlinear on large straight-line kernels (32x on a 140k-line device model,
/// 443s vs 13.6s; measured by the ARPice consumer across 37 models).
///
/// ponytail: no way to ask for on-device safety checks. Add an EmitOptions
/// knob if someone wants them, but quote the compile cost above first.
fn deviceOptimize(mode: std.builtin.OptimizeMode) std.builtin.OptimizeMode {
    return if (mode == .Debug) .ReleaseFast else mode;
}

/// Builds the extra imports for the kernel root, for one backend.
///
/// Called once per enabled backend, with that backend's resolved device target
/// (`nvptx64-cuda`/`sm_*` or `amdgcn-amdhsa`/`gfx*`) and its post-`deviceOptimize`
/// mode. Create every module inside this function using the `target` and
/// `optimize` handed to you, including nested imports.
///
/// It is a callback rather than a plain module list because a `std.Build.Module`
/// carries its own target and optimize mode: one prebuilt module cannot serve
/// both the nvptx64 and amdgcn compilations, and a host-built module dragged
/// into device code keeps the host's optimize mode.
pub const DeviceImportsFn = *const fn (
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    ctx: ?*anyopaque,
) []const std.Build.Module.Import;

pub const EmitOptions = struct {
    kernels_root: std.Build.LazyPath,
    /// Extra imports for the kernel root, on top of `gompute`. Leave null when
    /// the kernel root imports nothing of its own.
    imports: ?DeviceImportsFn = null,
    /// Passed through to `imports` untouched.
    imports_ctx: ?*anyopaque = null,
    cuda: CudaOptions = .{},
    hip: HipOptions = .{},
    /// Host target (from standardTargetOptions). Defaults to the host
    /// artifact's target. Native GPU backends are skipped on wasm.
    target: ?std.Build.ResolvedTarget = null,
    /// Optimize mode for device code (from standardOptimizeOption).
    /// Defaults to the host artifact's optimize mode.
    optimize: ?std.builtin.OptimizeMode = null,
};

/// ponytail: one build graph per process, so a plain list is enough to catch
/// the double-call. Keyed on the dependency pointer.
var emitted_deps: std.ArrayList(*std.Build.Dependency) = .empty;

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const host_mod = b.addModule("gompute", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    _ = b.addModule("gompute_device", .{
        .root_source_file = b.path("src/device.zig"),
    });

    // A stub so `gompute` always resolves `gompute_kernels`. Without it, merely
    // naming Kernel(spec, .cuda) in a build that never called emitKernels failed
    // with "no module named 'gompute_kernels' available within module 'gompute'"
    // -- an internal module the user never wrote. `emitted = false` lets the
    // host layer say what to do instead. emitKernels/addKernels replace this
    // import with the real artifacts; addImport with a duplicate name overwrites.
    //
    // It also makes the GPU backends reachable from `zig build test`: the test
    // module previously had no gompute_kernels at all, so CudaKernel, HipKernel,
    // AutoKernel and RawKernel were never semantically analyzed and a type error
    // in any of them shipped undetected.
    host_mod.addImport("gompute_kernels", b.createModule(.{
        .root_source_file = b.addWriteFiles().add("gompute_kernels.zig",
            \\//! Placeholder: gompute.emitKernels was never called in this build.
            \\pub const emitted = false;
            \\pub const has_cuda = false;
            \\pub const has_hip = false;
            \\pub const cuda: [:0]const u8 = "";
            \\pub const hip: [:0]const u8 = "";
            \\pub const cuda_names = struct {
            \\    pub fn resolve(comptime name: []const u8) [:0]const u8 {
            \\        @compileError("no GPU artifacts were emitted for: " ++ name);
            \\    }
            \\};
            \\pub const hip_names = struct {
            \\    pub fn resolve(comptime name: []const u8) [:0]const u8 {
            \\        @compileError("no GPU artifacts were emitted for: " ++ name);
            \\    }
            \\};
            \\
        ),
    }));

    const unit_tests = b.addTest(.{ .root_module = host_mod });
    const run_unit_tests = b.addRunArtifact(unit_tests);

    const tool_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/kernel_ir_tool.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_tool_tests = b.addRunArtifact(tool_tests);

    const codegen_probe = b.addObject(.{
        .name = "gompute-codegen-probe",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/codegen.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{.{ .name = "gompute", .module = host_mod }},
        }),
    });

    // The probe used to be compiled and then never read, so the README's claim
    // that the generated CPU kernel matches the hand-written loop was enforced
    // by nobody. Compare the emitted assembly instead.
    const codegen_check = b.addExecutable(.{
        .name = "gompute-codegen-check",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/codegen_check.zig"),
            .target = b.graph.host,
            .optimize = .Debug,
        }),
    });
    const run_codegen_check = b.addRunArtifact(codegen_check);
    run_codegen_check.addFileArg(codegen_probe.getEmittedAsm());

    const check_tests = b.addTest(.{ .root_module = codegen_check.root_module });
    const run_check_tests = b.addRunArtifact(check_tests);

    const test_step = b.step("test", "Run Gompute unit tests");
    test_step.dependOn(&run_unit_tests.step);
    test_step.dependOn(&run_tool_tests.step);
    test_step.dependOn(&run_check_tests.step);
    test_step.dependOn(&run_codegen_check.step);

    const docs_obj = b.addObject(.{ .name = "gompute", .root_module = host_mod });
    const docs_step = b.step("docs", "Emit API documentation to zig-out/docs");
    docs_step.dependOn(&b.addInstallDirectory(.{
        .source_dir = docs_obj.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs",
    }).step);
}

/// `gompute` plus whatever `options.imports` builds for this backend.
fn deviceImports(
    b: *std.Build,
    dep: *std.Build.Dependency,
    options: EmitOptions,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) []const std.Build.Module.Import {
    // `&.{base}` would return a pointer to a stack temporary; always allocate.
    const extra: []const std.Build.Module.Import = if (options.imports) |build_extra|
        build_extra(b, target, optimize, options.imports_ctx)
    else
        &.{};

    const all = b.allocator.alloc(std.Build.Module.Import, extra.len + 1) catch @panic("OOM");
    all[0] = .{ .name = "gompute", .module = dep.module("gompute_device") };
    @memcpy(all[1..], extra);
    return all;
}

/// Builds the CUDA PTX and HIP HSACO sub-compilations for one kernel root and
/// wraps them in a generated `gompute_kernels` module. Shared by `emitKernels`
/// and `addKernels`; the two differ only in who gets the resulting module.
fn buildArtifacts(
    b: *std.Build,
    dep: *std.Build.Dependency,
    options: EmitOptions,
    host_target: std.Build.ResolvedTarget,
    host_optimize: std.builtin.OptimizeMode,
) *std.Build.Module {
    const tool = b.addExecutable(.{
        .name = "gompute-kernel-ir-tool",
        .root_module = b.createModule(.{
            .root_source_file = dep.path("tools/kernel_ir_tool.zig"),
            .target = b.graph.host,
            .optimize = .Debug,
        }),
    });

    var cuda_ptx: ?std.Build.LazyPath = null;
    var hip_hsaco: ?std.Build.LazyPath = null;
    var hip_names: ?std.Build.LazyPath = null;
    var cuda_names: ?std.Build.LazyPath = null;

    const optimize = deviceOptimize(host_optimize);
    // CUDA/HIP drivers do not exist on the web; skip the native backends there.
    const is_wasm = host_target.result.cpu.arch.isWasm();

    const cuda_cpu: ?[]const u8 = if (is_wasm or !options.cuda.enabled)
        null
    else
        resolveGpu(b, "cuda", "sm_89", options.cuda.gpu, detectCudaGpu);
    const hip_cpu: ?[]const u8 = if (is_wasm or !options.hip.enabled)
        null
    else
        resolveGpu(b, "hip", "gfx1100", options.hip.gpu, detectHipGpu);

    if (cuda_cpu) |cpu| {
        const query = std.Target.Query.parse(.{
            .arch_os_abi = "nvptx64-cuda",
            .cpu_features = cpu,
        }) catch |err| std.debug.panic(
            "gompute: invalid CUDA target CPU \"{s}\" ({t}). Expected an NVPTX CPU name such " ++
                "as sm_70, sm_80, sm_89 or sm_90; run `zig targets` for the full list.",
            .{ cpu, err },
        );
        const target = b.resolveTargetQuery(query);
        const mode = if (options.cuda.optimize) |m| deviceOptimize(m) else optimize;
        const gpu_mod = b.createModule(.{
            .root_source_file = options.kernels_root,
            .target = target,
            .optimize = mode,
            // Debug info in device IR makes the PTX claim DWARF it doesn't
            // have; the CUDA driver then rejects the module (error 218).
            .strip = true,
            .imports = deviceImports(b, dep, options, target, mode),
        });
        const object = b.addObject(.{ .name = "gompute_cuda_ir", .root_module = gpu_mod });
        const rewrite = b.addRunArtifact(tool);
        rewrite.addFileArg(object.getEmittedLlvmIr());
        const rewritten_ir = rewrite.addOutputFileArg("gompute_cuda.ll");
        cuda_names = rewrite.addOutputFileArg("gompute_cuda_names.zig");

        const assemble = b.addSystemCommand(&.{
            b.graph.zig_exe,
            "cc",
            "-target",
            "nvptx64-cuda",
            b.fmt("-mcpu={s}", .{cuda_cpu.?}),
            "-S",
            "-g0", // nvptx rejects dwarf debug info; keeps stderr clean
            "-Wno-unused-command-line-argument",
        });
        assemble.addFileArg(rewritten_ir);
        cuda_ptx = assemble.addPrefixedOutputFileArg("-o", "gompute.ptx");
    }

    if (hip_cpu) |cpu| {
        const query = std.Target.Query.parse(.{
            .arch_os_abi = "amdgcn-amdhsa",
            .cpu_features = cpu,
        }) catch |err| std.debug.panic(
            "gompute: invalid HIP target CPU \"{s}\" ({t}). Expected an AMDGCN CPU name such " ++
                "as gfx900, gfx1030 or gfx1100; run `zig targets` for the full list.",
            .{ cpu, err },
        );
        const target = b.resolveTargetQuery(query);
        const mode = if (options.hip.optimize) |m| deviceOptimize(m) else optimize;
        const gpu_mod = b.createModule(.{
            .root_source_file = options.kernels_root,
            .target = target,
            .optimize = mode,
            .strip = true,
            .imports = deviceImports(b, dep, options, target, mode),
        });
        const object = b.addObject(.{ .name = "gompute_hip_obj", .root_module = gpu_mod });

        const names_run = b.addRunArtifact(tool);
        names_run.addFileArg(object.getEmittedLlvmIr());
        _ = names_run.addOutputFileArg("gompute_hip_rewritten.ll");
        hip_names = names_run.addOutputFileArg("gompute_hip_names.zig");

        const link = b.addSystemCommand(&.{ b.graph.zig_exe, "ld.lld", "-shared" });
        link.addFileArg(object.getEmittedBin());
        hip_hsaco = link.addPrefixedOutputFileArg("-o", "gompute.hsaco");
    }

    const write = b.addWriteFiles();
    const artifacts_source = write.add("gompute_kernels.zig", b.fmt(
        \\//! Generated by gompute.emitKernels.
        \\pub const emitted = true;
        \\pub const has_cuda = {};
        \\pub const has_hip = {};
        \\pub const cuda: [:0]const u8 = if (has_cuda) @embedFile("cuda_blob") else "";
        \\pub const hip: [:0]const u8 = if (has_hip) @embedFile("hip_blob") else "";
        \\pub const cuda_names = if (has_cuda) @import("cuda_names") else struct {{
        \\    pub fn resolve(comptime name: []const u8) [:0]const u8 {{
        \\        @compileError("no CUDA artifacts were emitted for: " ++ name);
        \\    }}
        \\}};
        \\pub const hip_names = if (has_hip) @import("hip_names") else struct {{
        \\    pub fn resolve(comptime _: []const u8) [:0]const u8 {{
        \\        @compileError("HIP artifacts were not emitted");
        \\    }}
        \\}};
        \\
    , .{ cuda_cpu != null, hip_cpu != null }));

    const artifacts_mod = b.createModule(.{ .root_source_file = artifacts_source });
    if (cuda_ptx) |path| artifacts_mod.addAnonymousImport("cuda_blob", .{ .root_source_file = path });
    if (hip_hsaco) |path| artifacts_mod.addAnonymousImport("hip_blob", .{ .root_source_file = path });
    if (hip_names) |path| artifacts_mod.addAnonymousImport("hip_names", .{ .root_source_file = path });
    if (cuda_names) |path| artifacts_mod.addAnonymousImport("cuda_names", .{ .root_source_file = path });
    return artifacts_mod;
}

/// Add CUDA PTX and HIP HSACO sub-compilations for a designated kernel root,
/// then attach them as `gompute_kernels` to the dependency's shared `gompute`
/// module -- the same module the consumer imported into `host`.
///
/// Supports ONE artifact-consuming executable per `gompute` dependency
/// instance. `dep.module("gompute")` is shared, so a second `emitKernels` call
/// against the same `dep` overwrites the first one's artifacts and the earlier
/// executable ships the later one's PTX (green build, `error.KernelNotFound` at
/// run time). For two or more executables, use `addKernels` instead.
///
/// Consumer build.zig:
///
///     const gompute_build = @import("gompute");
///     const dep = b.dependency("gompute", .{});
///     exe.root_module.addImport("gompute", dep.module("gompute"));
///     gompute_build.emitKernels(b, dep, exe, .{
///         .kernels_root = b.path("src/kernels.zig"),
///     });
///
/// If the kernel root imports modules of its own, also set `.imports`.
pub fn emitKernels(
    b: *std.Build,
    dep: *std.Build.Dependency,
    host: *std.Build.Step.Compile,
    options: EmitOptions,
) void {
    // Calling this twice against one dependency used to fail with
    // "file exists in modules 'gompute_kernels' and 'gompute_kernels0'", which
    // names neither emitKernels nor the user's build.zig. Say what happened.
    for (emitted_deps.items) |d| if (d == dep) @panic("gompute.emitKernels was called twice against the same dependency. " ++
        "dep.module(\"gompute\") is shared, so the second call would overwrite the " ++
        "first executable's artifacts. Use gompute.addKernels for two or more " ++
        "artifact-consuming executables -- it returns a private gompute module per call.");
    emitted_deps.append(b.allocator, dep) catch @panic("OOM");

    const artifacts_mod = buildArtifacts(
        b,
        dep,
        options,
        options.target orelse host.root_module.resolved_target orelse b.graph.host,
        options.optimize orelse host.root_module.optimize orelse .ReleaseFast,
    );
    // Resolution happens inside the shared `gompute` module; the host import is
    // only so a consumer's own code may `@import("gompute_kernels")` directly.
    dep.module("gompute").addImport("gompute_kernels", artifacts_mod);
    host.root_module.addImport("gompute_kernels", artifacts_mod);
}

pub const KernelsOptions = struct {
    root_source_file: std.Build.LazyPath,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    /// Extra imports for the kernel root, on top of `gompute`. Leave null when
    /// the kernel root imports nothing of its own.
    imports: ?DeviceImportsFn = null,
    /// Passed through to `imports` untouched.
    imports_ctx: ?*anyopaque = null,
    cuda: CudaOptions = .{},
    hip: HipOptions = .{},
};

/// What `addKernels` hands back.
pub const Kernels = struct {
    /// The generated artifact module: `has_cuda`, `has_hip`, and the blobs.
    kernels: *std.Build.Module,
    /// A private `gompute` instance wired to `kernels`. Import THIS wherever
    /// the consumer would have used `dep.module("gompute")` -- both in the
    /// executable's root module and in its host-side kernels module.
    gompute: *std.Build.Module,
};

/// Like `emitKernels`, but returns a fresh `gompute` module carrying only this
/// root's artifacts instead of mutating the dependency's shared one.
///
/// Call once per executable; instances do not collide, so two executables in
/// one build can each have their own kernels.
///
///     const k = gompute_build.addKernels(b, dep, .{
///         .root_source_file = b.path("src/kernels.zig"),
///         .target = target,
///         .optimize = optimize,
///     });
///     exe.root_module.addImport("gompute", k.gompute);
pub fn addKernels(b: *std.Build, dep: *std.Build.Dependency, o: KernelsOptions) Kernels {
    const artifacts_mod = buildArtifacts(b, dep, .{
        .kernels_root = o.root_source_file,
        .imports = o.imports,
        .imports_ctx = o.imports_ctx,
        .cuda = o.cuda,
        .hip = o.hip,
        .target = o.target,
        .optimize = o.optimize,
    }, o.target, o.optimize);

    const gompute_mod = b.createModule(.{
        .root_source_file = dep.path("src/root.zig"),
        .target = o.target,
        .optimize = o.optimize,
    });
    gompute_mod.addImport("gompute_kernels", artifacts_mod);
    return .{ .kernels = artifacts_mod, .gompute = gompute_mod };
}
