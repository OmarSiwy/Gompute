//! Device-only thread-index builtins for Zig 0.16.0.
//!
//! Almost no atomics here: Zig's own `@atomicRmw`, `@atomicLoad` and `@atomicStore`
//! on `addrspace(.global)` pointers lower correctly on both targets, with one
//! hole. On NVPTX, LLVM 21 drops the ordering of every read-modify-write:
//! `@atomicRmw(.., .acq_rel)`, `.seq_cst` and `@cmpxchgStrong` all emit a plain
//! relaxed `atom.global.*` with no fence. Pair a relaxed RMW with an
//! `@atomicLoad(.acquire)` / `@atomicStore(.release)` instead, which do lower
//! to `ld.acquire.sys` / `st.release.sys` -- system scope, stronger than the
//! `.gpu` a kernel needs, since Zig cannot name a sync scope --
//! `loadAcquireDevice` / `storeReleaseDevice` below are the `.gpu` pair.
//! `build.zig` pins these lowerings against `tests/device_entries.zig`.

const std = @import("std");
const builtin = @import("builtin");

extern fn @"llvm.amdgcn.workitem.id.x"() callconv(.c) u32;
extern fn @"llvm.amdgcn.workgroup.id.x"() callconv(.c) u32;
extern fn @"llvm.amdgcn.s.barrier"() callconv(.c) void;
extern fn @"llvm.amdgcn.s.sleep"(u32) callconv(.c) void;

extern fn @"llvm.nvvm.read.ptx.sreg.tid.x"() callconv(.c) u32;
extern fn @"llvm.nvvm.read.ptx.sreg.ctaid.x"() callconv(.c) u32;
extern fn @"llvm.nvvm.read.ptx.sreg.ntid.x"() callconv(.c) u32;
extern fn @"llvm.nvvm.read.ptx.sreg.nctaid.x"() callconv(.c) u32;
extern fn @"llvm.nvvm.barrier0"() callconv(.c) void;
extern fn @"llvm.nvvm.nanosleep"(u32) callconv(.c) void;

/// The flat thread index, `blockIdx.x * blockDim.x + threadIdx.x`.
///
/// `block_size` MUST equal the block size the kernel is actually launched with.
/// Generated `map` kernels satisfy that by construction -- the host always
/// launches at `Spec.block_size` -- but `RawKernel.launch` takes an arbitrary
/// `block: Dim3`, so a hand-written kernel calling `globalIdX(256)` and launched
/// with `block.x = 128` is wrong.
///
/// Worse, it is wrong asymmetrically: NVPTX reads the real block dimension from
/// `ntid.x` and ignores `block_size` entirely, so the same code gives correct
/// indices on CUDA and wrong ones on HIP.
///
/// AMDGCN traps when a launch is *wider* than `block_size`: a thread whose
/// `threadIdx.x` is past the end proves the mismatch, and the next sync on the
/// host reports the aborted kernel. A *narrower* launch is still silent -- no
/// thread can see it without the real block size, which is the wall below.
///
/// ponytail: AMDGCN should read `workgroup_size_x` out of the HSA dispatch
/// packet, which is the exact counterpart of `ntid.x`. It cannot be expressed
/// here yet: `llvm.amdgcn.dispatch.ptr` returns `ptr addrspace(4)` and Zig
/// 0.16 rejects `addrspace(.constant)` on amdgcn ("pointers with address space
/// 'constant' are not supported on amdgcn"). Declaring it `.global` would
/// mis-type the intrinsic. Revisit when Zig supports the constant address
/// space; until then the contract above is the mitigation, and it is untested
/// on hardware -- no AMD device was available.
pub inline fn globalIdX(comptime block_size: u32) usize {
    return switch (builtin.cpu.arch) {
        .nvptx64 => @as(usize, @"llvm.nvvm.read.ptx.sreg.ctaid.x"()) *
            @"llvm.nvvm.read.ptx.sreg.ntid.x"() + @"llvm.nvvm.read.ptx.sreg.tid.x"(),
        .amdgcn => blk: {
            const tid = @"llvm.amdgcn.workitem.id.x"();
            if (tid >= block_size) @trap();
            break :blk @as(usize, @"llvm.amdgcn.workgroup.id.x"()) * block_size + tid;
        },
        else => @compileError("device globalIdX used on a non-GPU target"),
    };
}

/// The thread's index within its own block, `threadIdx.x`. Indexes block-local
/// shared scratch, so it is always in `0..block_size`.
pub inline fn localIdX() u32 {
    return switch (builtin.cpu.arch) {
        .nvptx64 => @"llvm.nvvm.read.ptx.sreg.tid.x"(),
        .amdgcn => @"llvm.amdgcn.workitem.id.x"(),
        else => @compileError("device localIdX used on a non-GPU target"),
    };
}

/// The block's index within the grid, `blockIdx.x`.
pub inline fn blockIdX() u32 {
    return switch (builtin.cpu.arch) {
        .nvptx64 => @"llvm.nvvm.read.ptx.sreg.ctaid.x"(),
        .amdgcn => @"llvm.amdgcn.workgroup.id.x"(),
        else => @compileError("device blockIdX used on a non-GPU target"),
    };
}

/// The number of blocks in the grid, `gridDim.x`.
///
/// NVPTX ONLY. There is no portable counterpart, which is why the generated
/// `reduce` kernel takes its grid stride as a kernel argument instead of
/// calling this: the host computed the grid, so it can just say what it is.
///
/// ponytail: AMDGCN is missing on purpose, not by oversight.
/// `llvm.amdgcn.grid.size.x` looks like the answer and is not -- it is a *clang*
/// builtin (`__builtin_amdgcn_grid_size_x`) that clang expands to a dispatch-packet
/// load, not an LLVM intrinsic. Declaring it `extern` in Zig compiles, and
/// `ld.lld -shared` links it, because a shared object is allowed undefined
/// symbols; the HSACO then carries a real `@gotpcrel32` call to a function that
/// does not exist and the device rejects it at load. The genuine route is
/// `llvm.amdgcn.dispatch.ptr`, which returns `ptr addrspace(4)` and needs the
/// constant address space Zig 0.16 rejects on amdgcn -- the same wall
/// `globalIdX` documents above.
pub inline fn gridDimX() usize {
    return switch (builtin.cpu.arch) {
        .nvptx64 => @"llvm.nvvm.read.ptx.sreg.nctaid.x"(),
        else => @compileError(
            "gridDimX is nvptx64-only; pass the grid stride as a kernel argument for portability",
        ),
    };
}

/// Block-wide execution barrier plus a block-scope memory fence: every thread in
/// the block reaches it before any thread passes, and shared *and* global writes
/// made before it are visible to the whole block after it -- `__syncthreads`.
///
/// Must be reached by every thread in the block. Calling it inside a branch that
/// only some threads take hangs the block on NVIDIA and is undefined on AMD.
///
/// NVPTX `bar.sync` orders memory by itself. AMDGCN `s_barrier` does not, so
/// the fences HIP puts around it are spelled out; see `wg_fence`.
pub inline fn barrier() void {
    switch (builtin.cpu.arch) {
        .nvptx64 => @"llvm.nvvm.barrier0"(),
        .amdgcn => {
            asm volatile (wg_fence.release ::: .{ .memory = true });
            @"llvm.amdgcn.s.barrier"();
            asm volatile (wg_fence.acquire ::: .{ .memory = true });
        },
        else => @compileError("device barrier used on a non-GPU target"),
    }
}

/// What LLVM 21 emits for `fence syncscope("workgroup") release` / `acquire`
/// -- the pair clang puts around `s_barrier` for `__syncthreads` -- per gfx
/// family, in WGP mode, read off `clang -mcpu=gfx906/gfx90a/gfx1030/gfx1100/
/// gfx1201`. Zig has no fence and cannot name a sync scope, so it is inline asm.
/// gfx9 also waits `vmcnt` on release, which LLVM skips on gfx90a; harmless.
///
/// ponytail: tgsplit (gfx90a/gfx94x, off by default) spreads a workgroup over
/// CUs and needs an L1 invalidate on acquire; refused rather than guessed.
const wg_fence: struct { release: []const u8, acquire: []const u8 = "" } = if (builtin.cpu.arch != .amdgcn) .{ .release = "" } else blk: {
    const has = struct {
        fn f(comptime feature: std.Target.amdgcn.Feature) bool {
            return builtin.cpu.features.isEnabled(@intFromEnum(feature));
        }
    }.f;
    if (has(.tgsplit)) @compileError("barrier: tgsplit is not supported");
    if (has(.gfx12_insts)) break :blk .{
        .release = "s_wait_loadcnt 0x0\n\ts_wait_storecnt 0x0\n\ts_wait_dscnt 0x0",
        .acquire = "global_inv scope:SCOPE_SE",
    };
    if (has(.gfx10_insts)) break :blk .{
        .release = "s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_waitcnt_vscnt null, 0x0",
        .acquire = "buffer_gl0_inv",
    };
    break :blk .{ .release = "s_waitcnt vmcnt(0) lgkmcnt(0)" };
};

/// Back off for a moment inside a spin loop, so a waiting thread stops
/// competing for issue slots with the one it is waiting on. Purely a hint:
/// correctness never depends on it, and a loop without it is still correct.
///
/// ponytail: fixed ~64 ns (`nanosleep 64`, `s_sleep 1` = 64 clocks). Take the
/// duration as a parameter if a profile shows the backoff is wrong.
///
/// A no-op on NVPTX below PTX 6.3 or sm_70, where `nanosleep` does not exist.
/// That includes the default `sm_70` target, which Zig pins to PTX 6.0.
pub inline fn spinPause() void {
    switch (builtin.cpu.arch) {
        .nvptx64 => if (comptime nv.ptx >= 63 and nv.sm >= 70) @"llvm.nvvm.nanosleep"(64),
        .amdgcn => @"llvm.amdgcn.s.sleep"(1),
        else => @compileError("device spinPause used on a non-GPU target"),
    }
}

/// `@atomicLoad(.acquire)` at device (`.gpu` / agent) scope instead of Zig's
/// `.sys`: the other half of a flag another block publishes with
/// `storeReleaseDevice`. On NVPTX it skips the system-scope fence, which is the
/// whole point (an LU publish/acquire refactor went from 7.1 to 3.7 ms).
///
/// ponytail: AMDGCN uses Zig's system-scope atomics. LLVM emits the same code
/// for agent and system scope on gfx9, gfx10 and gfx11; only gfx90a, gfx94x
/// and gfx12 pay extra (an L2 writeback/invalidate). Hand-written agent-scope
/// asm for those when someone has one to test on. NVPTX below sm_70 has no
/// `ld.acquire` and falls back the same way.
pub inline fn loadAcquireDevice(p: *addrspace(.global) const u32) u32 {
    if (builtin.cpu.arch == .nvptx64 and comptime nv.sm >= 70)
        return asm volatile ("ld.acquire.gpu.global.u32 %[v], [%[p]];"
            : [v] "=r" (-> u32),
            : [p] "l" (@intFromPtr(p)),
            : .{ .memory = true });
    return @atomicLoad(u32, p, .acquire);
}

/// `@atomicStore(.release)` at device scope; see `loadAcquireDevice`.
pub inline fn storeReleaseDevice(p: *addrspace(.global) u32, v: u32) void {
    if (builtin.cpu.arch == .nvptx64 and comptime nv.sm >= 70)
        return asm volatile ("st.release.gpu.global.u32 [%[p]], %[v];"
            :
            : [p] "l" (@intFromPtr(p)),
              [v] "r" (v),
            : .{ .memory = true });
    @atomicStore(u32, p, v, .release);
}

/// The target's PTX ISA and SM versions, e.g. 60 and 70. Zig's `ptxNN` and
/// `sm_NN` features do not imply each other, so take the highest of each.
const nv = blk: {
    var v: struct { ptx: u32 = 0, sm: u32 = 0 } = .{};
    if (builtin.cpu.arch != .nvptx64) break :blk v;
    for (@typeInfo(std.Target.nvptx.Feature).@"enum".fields) |f| {
        if (!builtin.cpu.features.isEnabled(f.value)) continue;
        const digits = std.mem.trimEnd(u8, f.name, "af"); // sm_90a, sm_100f
        if (std.mem.startsWith(u8, digits, "ptx"))
            v.ptx = @max(v.ptx, std.fmt.parseInt(u32, digits[3..], 10) catch 0);
        if (std.mem.startsWith(u8, digits, "sm_"))
            v.sm = @max(v.sm, std.fmt.parseInt(u32, digits[3..], 10) catch 0);
    }
    break :blk v;
};
