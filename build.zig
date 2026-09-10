const std = @import("std");

pub const Gpu = union(enum) {
    /// Detect the GPU on the BUILD machine; the backend is compiled out (with a
    /// warning) if none is found. Not the deploy machine -- pin `.name` in CI,
    /// Docker, Nix and releases.
    auto,
    /// Explicit target CPU accepted by Zig, e.g. `sm_89` (CUDA) or `gfx1100` (HIP).
    name: []const u8,
};

/// One backend's device-code settings. `CudaOptions` and `HipOptions` are the
/// same shape and always have been; the two names exist so that `.cuda = .{...}`
/// and `.hip = .{...}` read as what they are.
const BackendOptions = struct {
    enabled: bool = true,
    gpu: Gpu = .auto,
    /// Optimize mode for this backend's device code. Overrides
    /// `EmitOptions.optimize`.
    optimize: ?std.builtin.OptimizeMode = null,
};

pub const CudaOptions = BackendOptions;
pub const HipOptions = BackendOptions;

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
    const dot = computeCapDot(line) orelse return null;
    return b.fmt("sm_{s}{s}", .{ line[0..dot], line[dot + 1 ..] });
}

/// Index of the `.` in a compute capability, which is exactly
/// `<digits>.<digits>`, e.g. "8.9". Null when `s` is not one.
fn computeCapDot(s: []const u8) ?usize {
    const dot = std.mem.indexOfScalar(u8, s, '.') orelse return null;
    if (dot == 0 or dot + 1 == s.len) return null;
    for (s, 0..) |c, i| if (i != dot and !std.ascii.isDigit(c)) return null;
    return dot;
}

test computeCapDot {
    // What nvidia-smi prints on a working card, and what detectCudaGpu makes
    // of it. `sm_` ++ digits, dot dropped.
    for ([_]struct { []const u8, ?usize }{
        .{ "8.9", 1 },
        .{ "12.0", 2 },
        .{ "7.5", 1 },
        // The reason this guard exists: nvidia-smi exits 0 and prints these for
        // a card it cannot report. "sm_[N/A]" used to reach Target.Query.parse
        // and kill the build.
        .{ "[N/A]", null },
        .{ "[Not Supported]", null },
        .{ "", null },
        .{ "8", null },
        .{ ".9", null },
        .{ "8.", null },
        .{ "8.9.1", null },
        .{ "8 .9", null },
        .{ "sm_89", null },
    }) |case| {
        const s, const want = case;
        try std.testing.expectEqual(want, computeCapDot(s));
    }
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

/// One independently compiled kernel root.
///
/// Each root becomes its own `zig build-obj` per backend, so the build runner
/// compiles them on its thread pool and caches them separately: editing one root
/// rebuilds that root only. Kernel names must not repeat across roots.
pub const KernelRoot = struct {
    /// Artifact id, unique within one emitKernels/addKernels call. Shows up in
    /// object names and in diagnostics; keep it a plain identifier.
    name: []const u8,
    root: std.Build.LazyPath,
    /// Extra imports for this root, on top of `gompute`.
    imports: ?DeviceImportsFn = null,
    /// Passed through to `imports` untouched.
    imports_ctx: ?*anyopaque = null,
    /// Marks a root whose device compilation is big enough that running it
    /// alongside the other big ones is a memory problem rather than a speedup.
    /// Heavy roots are chained into `heavy_lanes` serial lanes; light roots run
    /// unconstrained.
    heavy: bool = false,
};

pub const EmitOptions = struct {
    /// The single-root form. Equivalent to one `kernel_roots` entry; the two may
    /// be combined, and at least one of them must be set.
    kernels_root: ?std.Build.LazyPath = null,
    /// Extra imports for `kernels_root`, on top of `gompute`. Leave null when
    /// the kernel root imports nothing of its own.
    imports: ?DeviceImportsFn = null,
    /// Passed through to `imports` untouched.
    imports_ctx: ?*anyopaque = null,
    /// Roots compiled independently and in parallel. See `KernelRoot`.
    kernel_roots: []const KernelRoot = &.{},
    /// How many `heavy` roots may compile at once. 1 means fully serial.
    heavy_lanes: u8 = 1,
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
            \\const std = @import("std");
            \\pub const emitted = false;
            \\pub const has_cuda = false;
            \\pub const has_hip = false;
            \\pub const Entry = struct { blob: u16, symbol: [:0]const u8 };
            \\pub const root_names: []const []const u8 = &.{};
            \\pub const cuda_images: []const [:0]const u8 = &.{};
            \\pub const hip_images: []const [:0]const u8 = &.{};
            \\pub const cuda_index = std.StaticStringMap(Entry).initComptime(.{});
            \\pub const hip_index = std.StaticStringMap(Entry).initComptime(.{});
            \\
        ),
    }));

    // The two modules and the stub above are the product; everything below is
    // dev-only. A consumer's `b.dependency("gompute", ...)` runs this build()
    // too, and `docs/` is deliberately absent from `.paths` (same reasoning as
    // `examples`), so buildDocs panicked on a perfectly good dependency —
    // "gompute: docs/ is missing", pointing at gompute's own build.zig from a
    // consumer that did nothing wrong. pkg_hash is "" only for the root
    // project, so this also spares consumers the test artifacts at configure.
    if (b.pkg_hash.len != 0) return;

    // device/math.zig checks `pow` against libm at 1 ulp, and libm is the only
    // oracle precise enough to see that. Set here, past the early return, so
    // only gompute's own build gets it -- a consumer's `gompute` module stays
    // freestanding, which is the whole point of that file.
    host_mod.link_libc = true;

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

    // build.zig compiled as a plain module, for the handful of pure helpers in
    // it. `pub fn build` is never called here -- only `test` decls run -- but
    // this is the only automated coverage anything in this file has.
    const build_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("build.zig"),
            .target = b.graph.host,
            .optimize = .Debug,
        }),
    });
    const run_build_tests = b.addRunArtifact(build_tests);

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
    test_step.dependOn(&run_build_tests.step);
    test_step.dependOn(&run_check_tests.step);
    test_step.dependOn(&run_codegen_check.step);

    buildDocs(b, host_mod, test_step);
}

/// Assemble the whole documentation site into `zig-out/docs`: the hand-written
/// landing page, every `docs/*.md` rendered to HTML, and Zig's generated API
/// reference under `api/`.
///
/// Rendering is done by `tools/md2html.zig` rather than a system Markdown tool
/// so that the site builds with nothing but Zig -- the docs can be previewed on
/// any machine that can build the library, and CI has no second rendering path
/// to drift from.
fn buildDocs(b: *std.Build, host_mod: *std.Build.Module, test_step: *std.Build.Step) void {
    const docs_step = b.step("docs", "Build the documentation site into zig-out/docs");

    // Everything the site is made of hangs off `site`, not off `docs` directly,
    // so that `-Dopen` can be sequenced *after* the site without `docs`
    // depending on the opener and the opener depending on `docs`.
    const site = b.allocator.create(std.Build.Step) catch @panic("OOM");
    site.* = std.Build.Step.init(.{ .id = .custom, .name = "docs site", .owner = b });
    docs_step.dependOn(site);

    const api = b.addObject(.{ .name = "gompute", .root_module = host_mod });
    site.dependOn(&b.addInstallDirectory(.{
        .source_dir = api.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs/api",
    }).step);

    for ([_][]const u8{
        "index.html",
        "style.css",
        // Subset Iosevka, checked in: see docs/fonts/README.md. 36 KB total,
        // so the site needs no network and no font on the reader's machine.
        "fonts/iosevka-400.woff2",
        "fonts/iosevka-aile-400.woff2",
        "fonts/iosevka-aile-400-italic.woff2",
        "fonts/iosevka-aile-600.woff2",
    }) |asset| {
        site.dependOn(&b.addInstallFileWithDir(
            b.path(b.fmt("docs/{s}", .{asset})),
            .prefix,
            b.fmt("docs/{s}", .{asset}),
        ).step);
    }

    const md2html = b.addExecutable(.{
        .name = "gompute-md2html",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/md2html.zig"),
            .target = b.graph.host,
            .optimize = .Debug,
        }),
    });

    // Every docs/*.md becomes a page. Adding one needs no build.zig change.
    const io = b.graph.io;
    var dir = b.build_root.handle.openDir(io, "docs", .{ .iterate = true }) catch
        @panic("gompute: docs/ is missing; the documentation site cannot be built");
    defer dir.close(io);
    var it = dir.iterate();
    while (it.next(io) catch @panic("gompute: cannot read docs/")) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".md")) continue;
        const stem = entry.name[0 .. entry.name.len - ".md".len];

        const run = b.addRunArtifact(md2html);
        run.addFileArg(b.path(b.fmt("docs/{s}", .{entry.name})));
        run.addFileArg(b.path("docs/page.template.html"));
        const out = run.addOutputFileArg(b.fmt("{s}.html", .{stem}));
        site.dependOn(&b.addInstallFileWithDir(
            out,
            .prefix,
            b.fmt("docs/{s}.html", .{stem}),
        ).step);
    }

    // The renderer's own tests belong to `test`, not just `docs`, so
    // `nix flake check` covers them too.
    const md_tests = b.addRunArtifact(b.addTest(.{ .root_module = md2html.root_module }));
    site.dependOn(&md_tests.step);
    test_step.dependOn(&md_tests.step);

    // `zig build docs -Dopen` previews it. Opt-in, so CI can build the site
    // without a browser trying to launch on a headless runner.
    if (b.option(bool, "open", "Open the built documentation in a browser") orelse false) {
        const opener = switch (b.graph.host.result.os.tag) {
            .macos => "open",
            else => "xdg-open",
        };
        const open = b.addSystemCommand(&.{opener});
        // The install path, not `b.path`: the file only exists once `site` ran.
        open.addArg(b.getInstallPath(.prefix, "docs/index.html"));
        open.has_side_effects = true;
        open.step.dependOn(site);
        docs_step.dependOn(&open.step);
    }
}

/// `gompute` plus whatever `root.imports` builds for this backend.
fn deviceImports(
    b: *std.Build,
    dep: *std.Build.Dependency,
    root: KernelRoot,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) []const std.Build.Module.Import {
    // `&.{base}` would return a pointer to a stack temporary; always allocate.
    const extra: []const std.Build.Module.Import = if (root.imports) |build_extra|
        build_extra(b, target, optimize, root.imports_ctx)
    else
        &.{};

    const all = b.allocator.alloc(std.Build.Module.Import, extra.len + 1) catch @panic("OOM");
    all[0] = .{ .name = "gompute", .module = dep.module("gompute_device") };
    @memcpy(all[1..], extra);
    return all;
}

/// `options.kernels_root` and `options.kernel_roots` as one list, checked.
fn normalizeRoots(b: *std.Build, options: EmitOptions) []const KernelRoot {
    var roots: std.ArrayList(KernelRoot) = .empty;
    if (options.kernels_root) |path| roots.append(b.allocator, .{
        .name = "kernels",
        .root = path,
        .imports = options.imports,
        .imports_ctx = options.imports_ctx,
    }) catch @panic("OOM");
    roots.appendSlice(b.allocator, options.kernel_roots) catch @panic("OOM");

    if (roots.items.len == 0) @panic("gompute: no kernels to emit. Set .kernels_root (one root) " ++
        "or .kernel_roots (several, compiled in parallel and cached separately).");
    // Names become object-file names and blob indices; a collision would make
    // one root silently overwrite the other's artifact.
    for (roots.items, 0..) |a, i| for (roots.items[i + 1 ..]) |c| {
        if (std.mem.eql(u8, a.name, c.name)) std.debug.panic(
            "gompute: two kernel roots are both named \"{s}\". Root names must be unique " ++
                "(note that .kernels_root takes the name \"kernels\").",
            .{a.name},
        );
    };
    return roots.items;
}

/// The two device backends. `@tagName` is also the prefix of the generated
/// import names (`cuda_blob_0`, `hip_names_1`), so it must keep matching what
/// `artifactsSource` writes.
const Device = enum { cuda, hip };

/// Serializes `heavy` roots into `lanes` chains, leaving light roots free.
///
/// A plain `dependOn` edge between two unrelated compilations is a false
/// dependency, which only constrains ordering -- exactly what is wanted. 37
/// concurrent LLVM processes on 140k-line device models is an OOM, not a
/// speedup.
const HeavyLanes = struct {
    last: []?*std.Build.Step,
    next: usize = 0,

    fn init(b: *std.Build, lanes: u8) HeavyLanes {
        const n = @max(lanes, 1);
        const slots = b.allocator.alloc(?*std.Build.Step, n) catch @panic("OOM");
        @memset(slots, null);
        return .{ .last = slots };
    }

    fn chain(self: *HeavyLanes, step: *std.Build.Step) void {
        const lane = self.next % self.last.len;
        self.next += 1;
        if (self.last[lane]) |prev| step.dependOn(prev);
        self.last[lane] = step;
    }
};

/// Builds the CUDA PTX and HIP HSACO sub-compilations for every kernel root and
/// wraps them in a generated `gompute_kernels` module. Shared by `emitKernels`
/// and `addKernels`; the two differ only in who gets the resulting module.
fn buildArtifacts(
    b: *std.Build,
    dep: *std.Build.Dependency,
    options: EmitOptions,
    host_target: std.Build.ResolvedTarget,
    host_optimize: std.builtin.OptimizeMode,
) *std.Build.Module {
    const roots = normalizeRoots(b, options);
    const tool = b.addExecutable(.{
        .name = "gompute-kernel-ir-tool",
        .root_module = b.createModule(.{
            .root_source_file = dep.path("tools/kernel_ir_tool.zig"),
            .target = b.graph.host,
            .optimize = .Debug,
        }),
    });

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

    // Created up front so each backend loop can register its blobs on the spot.
    // `has_cuda`/`has_hip` are already known, and the generated source depends
    // on nothing else, so there is no reason to stage the LazyPaths anywhere.
    const write = b.addWriteFiles();
    const artifacts_mod = b.createModule(.{
        .root_source_file = write.add("gompute_kernels.zig", artifactsSource(
            b,
            roots,
            cuda_cpu != null,
            hip_cpu != null,
        )),
    });

    // The two backends run the same pipeline -- device module, build-obj, IR
    // tool, then one final command -- and differ only in the values below plus
    // that last step. Kept as one loop so a change to the pipeline cannot be
    // applied to CUDA and forgotten for HIP; nothing in `zig build test`
    // exercises either path.
    for ([_]struct {
        device: Device,
        triple: []const u8,
        expected_cpu: []const u8,
        cpu: ?[]const u8,
        optimize: ?std.builtin.OptimizeMode,
    }{
        .{
            .device = .cuda,
            .triple = "nvptx64-cuda",
            .expected_cpu = "an NVPTX CPU name such as sm_70, sm_80, sm_89 or sm_90",
            .cpu = cuda_cpu,
            .optimize = options.cuda.optimize,
        },
        .{
            .device = .hip,
            .triple = "amdgcn-amdhsa",
            .expected_cpu = "an AMDGCN CPU name such as gfx900, gfx1030 or gfx1100",
            .cpu = hip_cpu,
            .optimize = options.hip.optimize,
        },
    }) |d| {
        const cpu = d.cpu orelse continue;
        const tag = @tagName(d.device);

        const query = std.Target.Query.parse(.{
            .arch_os_abi = d.triple,
            .cpu_features = cpu,
        }) catch |err| std.debug.panic(
            "gompute: invalid {s} target CPU \"{s}\" ({t}). Expected {s}; " ++
                "run `zig targets` for the full list.",
            .{ tag, cpu, err, d.expected_cpu },
        );
        const target = b.resolveTargetQuery(query);
        const mode = if (d.optimize) |m| deviceOptimize(m) else optimize;
        var lanes: HeavyLanes = .init(b, options.heavy_lanes);

        for (roots, 0..) |root, i| {
            const object = b.addObject(.{
                .name = b.fmt("gompute_{s}_obj_{s}", .{ tag, root.name }),
                .root_module = b.createModule(.{
                    .root_source_file = root.root,
                    .target = target,
                    .optimize = mode,
                    // Debug info in device IR makes the PTX claim DWARF it
                    // doesn't have; the CUDA driver then rejects the module
                    // (error 218).
                    .strip = true,
                    .imports = deviceImports(b, dep, root, target, mode),
                }),
            });
            if (root.heavy) lanes.chain(&object.step);

            // The tool always writes both outputs. CUDA assembles the rewritten
            // IR; HIP links the object itself and keeps only the name table.
            const rewrite = b.addRunArtifact(tool);
            rewrite.addFileArg(object.getEmittedLlvmIr());
            const rewritten_ir = rewrite.addOutputFileArg(b.fmt("gompute_{s}_{s}.ll", .{ tag, root.name }));
            const names = rewrite.addOutputFileArg(b.fmt("gompute_{s}_names_{s}.zig", .{ tag, root.name }));

            const blob = switch (d.device) {
                .cuda => blk: {
                    const assemble = b.addSystemCommand(&.{
                        b.graph.zig_exe,
                        "cc",
                        "-target",
                        d.triple,
                        b.fmt("-mcpu={s}", .{cpu}),
                        "-S",
                        "-g0", // nvptx rejects dwarf debug info; keeps stderr clean
                        "-Wno-unused-command-line-argument",
                    });
                    assemble.addFileArg(rewritten_ir);
                    break :blk assemble.addPrefixedOutputFileArg("-o", b.fmt("gompute_{s}.ptx", .{root.name}));
                },
                .hip => blk: {
                    const link = b.addSystemCommand(&.{ b.graph.zig_exe, "ld.lld", "-shared" });
                    link.addFileArg(object.getEmittedBin());
                    break :blk link.addPrefixedOutputFileArg("-o", b.fmt("gompute_{s}.hsaco", .{root.name}));
                },
            };

            artifacts_mod.addAnonymousImport(b.fmt("{s}_blob_{d}", .{ tag, i }), .{ .root_source_file = blob });
            artifacts_mod.addAnonymousImport(b.fmt("{s}_names_{d}", .{ tag, i }), .{ .root_source_file = names });
        }
    }

    return artifacts_mod;
}

/// The generated `gompute_kernels` module: the blobs, plus one comptime
/// name -> (blob, symbol) map per backend merged from the per-root tables the
/// IR tool emitted. `src/host/kernel.zig` reads both.
fn artifactsSource(
    b: *std.Build,
    roots: []const KernelRoot,
    has_cuda: bool,
    has_hip: bool,
) []const u8 {
    var out: std.Io.Writer.Allocating = .init(b.allocator);
    const w = &out.writer;

    w.print(
        \\//! Generated by gompute.emitKernels. Do not edit.
        \\const std = @import("std");
        \\
        \\pub const emitted = true;
        \\pub const has_cuda = {};
        \\pub const has_hip = {};
        \\
        \\/// Kernel roots, in blob order.
        \\pub const root_names = [_][]const u8{{
        \\
    , .{ has_cuda, has_hip }) catch @panic("OOM");
    for (roots) |root| w.print("    \"{f}\",\n", .{std.zig.fmtString(root.name)}) catch @panic("OOM");

    w.writeAll(
        \\};
        \\
        \\/// Where a kernel lives: which blob, and its symbol name inside it.
        \\pub const Entry = struct { blob: u16, symbol: [:0]const u8 };
        \\const KV = struct { []const u8, Entry };
        \\
        \\/// Fold one root's name table into the merged map. `mangled` is HIP,
        \\/// whose entry points keep the mangled Zig symbol; a CUDA entry point is
        \\/// the exported name itself.
        \\///
        \\/// ponytail: O(kernels^2) duplicate scan, at comptime. Fine into the
        \\/// hundreds; sort first if a consumer ever gets to thousands.
        \\fn merge(
        \\    comptime kvs: []const KV,
        \\    comptime blob: u16,
        \\    comptime table: anytype,
        \\    comptime mangled: bool,
        \\) []const KV {
        \\    @setEvalBranchQuota(100_000);
        \\    var out = kvs;
        \\    for (table) |e| {
        \\        for (out) |prev| if (std.mem.eql(u8, prev[0], e.exported)) @compileError(
        \\            "gompute: kernel \"" ++ e.exported ++ "\" is exported by two kernel roots (" ++
        \\                root_names[prev[1].blob] ++ " and " ++ root_names[blob] ++
        \\                "). A kernel name is the run-time dispatch key, so it must name one root.",
        \\        );
        \\        out = out ++ .{KV{ e.exported, .{
        \\            .blob = blob,
        \\            .symbol = if (mangled) e.internal else e.exported,
        \\        } }};
        \\    }
        \\    return out;
        \\}
        \\
        \\
    ) catch @panic("OOM");

    for ([_]struct { []const u8, bool, bool }{
        .{ "cuda", has_cuda, false },
        .{ "hip", has_hip, true },
    }) |backend| {
        const tag, const present, const mangled = backend;
        if (!present) {
            w.print(
                \\pub const {s}_images: []const [:0]const u8 = &.{{}};
                \\pub const {s}_index = std.StaticStringMap(Entry).initComptime(.{{}});
                \\
                \\
            , .{ tag, tag }) catch @panic("OOM");
            continue;
        }
        w.print("pub const {s}_images = [_][:0]const u8{{\n", .{tag}) catch @panic("OOM");
        for (roots, 0..) |_, i| w.print("    @embedFile(\"{s}_blob_{d}\"),\n", .{ tag, i }) catch @panic("OOM");
        w.print(
            \\}};
            \\pub const {s}_index = std.StaticStringMap(Entry).initComptime(blk: {{
            \\    var kv: []const KV = &.{{}};
            \\
        , .{tag}) catch @panic("OOM");
        for (roots, 0..) |_, i| w.print(
            "    kv = merge(kv, {d}, @import(\"{s}_names_{d}\").entries, {});\n",
            .{ i, tag, i, mangled },
        ) catch @panic("OOM");
        w.writeAll("    break :blk kv;\n});\n\n") catch @panic("OOM");
    }
    return out.written();
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
///
/// For several roots, use `.kernel_roots`: each is its own sub-compilation, so
/// they compile on the build runner's thread pool and cache separately -- an
/// edit to one root rebuilds that root alone. Kernel names must be unique
/// across roots, since a name is what run-time dispatch looks up.
///
///     gompute_build.emitKernels(b, dep, exe, .{
///         .kernel_roots = &.{
///             .{ .name = "bsim4", .root = b.path("src/bsim4.zig"), .heavy = true },
///             .{ .name = "hisim", .root = b.path("src/hisim.zig"), .heavy = true },
///             .{ .name = "vbic", .root = b.path("src/vbic.zig") },
///         },
///         .heavy_lanes = 2,
///     });
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
    /// The single-root form. Equivalent to one `kernel_roots` entry; the two may
    /// be combined, and at least one of them must be set.
    root_source_file: ?std.Build.LazyPath = null,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    /// Extra imports for `root_source_file`, on top of `gompute`. Leave null
    /// when the kernel root imports nothing of its own.
    imports: ?DeviceImportsFn = null,
    /// Passed through to `imports` untouched.
    imports_ctx: ?*anyopaque = null,
    /// Roots compiled independently and in parallel. See `KernelRoot`.
    kernel_roots: []const KernelRoot = &.{},
    /// How many `heavy` roots may compile at once. 1 means fully serial.
    heavy_lanes: u8 = 1,
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
        .kernel_roots = o.kernel_roots,
        .heavy_lanes = o.heavy_lanes,
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
