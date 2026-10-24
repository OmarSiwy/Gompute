//! Device-only thread indices, barrier, warp shuffle and device-scope atomics.
//!
//! Almost no atomics here: Zig's own `@atomicRmw`, `@atomicLoad` and `@atomicStore`
//! on `addrspace(.global)` pointers lower correctly on both targets, with one
//! hole. On NVPTX, LLVM drops the ordering of `@atomicRmw`: `.acq_rel` and
//! `.seq_cst` emit a plain relaxed `atom.global.*` with no fence (LLVM 21 did
//! the same to `@cmpxchgStrong`; LLVM 22 fences that one). Pair a relaxed RMW with an
//! `@atomicLoad(.acquire)` / `@atomicStore(.release)` instead, which do lower
//! to `ld.acquire.sys` / `st.release.sys` -- system scope, stronger than the
//! `.gpu` a kernel needs, since Zig cannot name a sync scope --
//! `loadAcquireDevice` / `storeReleaseDevice` below are the `.gpu` pair.
//! `build.zig` pins these lowerings against `tests/device_entries.zig`.

const std = @import("std");
const builtin = @import("builtin");

extern fn @"llvm.amdgcn.s.barrier"() callconv(.c) void;
extern fn @"llvm.amdgcn.s.sleep"(u32) callconv(.c) void;
extern fn @"llvm.amdgcn.ds.bpermute"(i32, i32) callconv(.c) i32;
extern fn @"llvm.amdgcn.mbcnt.lo"(u32, u32) callconv(.c) u32;
extern fn @"llvm.amdgcn.mbcnt.hi"(u32, u32) callconv(.c) u32;

extern fn @"llvm.nvvm.read.ptx.sreg.nctaid.x"() callconv(.c) u32;
extern fn @"llvm.nvvm.shfl.sync.bfly.i32"(u32, i32, i32, i32) callconv(.c) i32;
extern fn @"llvm.nvvm.barrier0"() callconv(.c) void;
extern fn @"llvm.nvvm.nanosleep"(u32) callconv(.c) void;

/// The flat thread index, `blockIdx.x * blockDim.x + threadIdx.x`.
///
/// Reads the block size the kernel was actually launched with on both
/// vendors: `%ntid.x` on NVPTX, the dispatch packet's `workgroup_size_x` on
/// AMDGCN (Zig's `@workGroupSize` emits the load the language cannot spell,
/// `addrspace(.constant)` being rejected on amdgcn). `block_size` is ignored;
/// it stays in the signature because every kernel written against 1.4.0
/// passes it, and it used to be what AMDGCN multiplied by.
pub inline fn globalIdX(comptime block_size: u32) usize {
    _ = block_size;
    return @as(usize, @workGroupId(0)) * @workGroupSize(0) + @workItemId(0);
}

/// The thread's index within its own block, `threadIdx.x`. Indexes block-local
/// shared scratch, so it is always in `0..blockDim.x`.
pub inline fn localIdX() u32 {
    return @workItemId(0);
}

/// The block's index within the grid, `blockIdx.x`.
pub inline fn blockIdX() u32 {
    return @workGroupId(0);
}

/// `v` from the lane whose index is this lane's XOR `mask`, within groups of
/// 32 lanes: `__shfl_xor_sync(~0, v, mask)` on NVPTX, and on AMDGCN a
/// `ds_bpermute` that LLVM lowers to DPP, `v_permlanex16` or `ds_swizzle`
/// for the masks that have one. 32-lane groups on both vendors, so a wave64
/// is two independent groups and nothing here depends on the wave width.
///
/// Every lane of the group must be active and reach the call: launch with a
/// block size that is a multiple of 32, and call it outside any branch that
/// splits a group. `T` is any type of 32 bits or less, or a multiple of 32
/// bits, moved 32 bits at a time.
pub inline fn shuffleXor(comptime T: type, v: T, comptime mask: u5) T {
    const bits = @bitSizeOf(T);
    if (bits <= 32) {
        const U = @Int(.unsigned, bits);
        const r = shfl32(@as(U, @bitCast(v)), mask);
        return @bitCast(@as(U, @truncate(r)));
    }
    if (bits % 32 != 0) @compileError("shuffleXor: " ++ @typeName(T) ++ " is not a multiple of 32 bits");
    var words: [bits / 32]u32 = @bitCast(v);
    inline for (&words) |*w| w.* = shfl32(w.*, mask);
    return @bitCast(words);
}

inline fn shfl32(v: u32, comptime mask: u5) u32 {
    return switch (builtin.target.cpu.arch) {
        .nvptx64 => @bitCast(@"llvm.nvvm.shfl.sync.bfly.i32"(0xffff_ffff, @bitCast(v), mask, 0x1f)),
        .amdgcn => blk: {
            // The lane within the wave, 0..63 on wave64 and 0..31 on wave32,
            // where `mbcnt.hi` adds nothing. `ds_bpermute` takes a byte index.
            const lane = @"llvm.amdgcn.mbcnt.hi"(~@as(u32, 0), @"llvm.amdgcn.mbcnt.lo"(~@as(u32, 0), 0));
            break :blk @bitCast(@"llvm.amdgcn.ds.bpermute"(@bitCast((lane ^ mask) << 2), @bitCast(v)));
        },
        else => @compileError("device shuffleXor used on a non-GPU target"),
    };
}

/// The number of blocks in the grid, `gridDim.x`.
///
/// NVPTX ONLY. Zig has `@workGroupSize` but no grid-size builtin, and the
/// AMDGCN answer is the dispatch packet's `grid_size_x`, whose pointer lives
/// in the constant address space Zig rejects on amdgcn. The generated
/// `reduce` takes its grid stride as a kernel argument instead: the host
/// computed the grid, so it can just say what it is.
///
/// ponytail: declaring `llvm.amdgcn.grid.size.x` does not work around it: it
/// is a clang builtin, not an LLVM intrinsic, and the HSACO ends up calling a
/// function that does not exist. Revisit when Zig grows `@gridSize` or
/// accepts `addrspace(.constant)` on amdgcn.
pub inline fn gridDimX() usize {
    return switch (builtin.target.cpu.arch) {
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
    switch (builtin.target.cpu.arch) {
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
const wg_fence: struct { release: []const u8, acquire: []const u8 = "" } = if (builtin.target.cpu.arch != .amdgcn) .{ .release = "" } else blk: {
    const has = struct {
        fn f(comptime feature: std.Target.amdgcn.Feature) bool {
            return builtin.target.cpu.features.isEnabled(@backingInt(feature));
        }
    }.f;
    // Zig 0.18's amdgcn has no `tgsplit` mode feature, only `tgsplit_support`.
    if (@hasField(std.Target.amdgcn.Feature, "tgsplit") and has(.tgsplit)) @compileError("barrier: tgsplit is not supported");
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
    switch (builtin.target.cpu.arch) {
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
    if (builtin.target.cpu.arch == .nvptx64 and comptime nv.sm >= 70)
        return asm volatile ("ld.acquire.gpu.global.u32 %[v], [%[p]];"
            : [v] "=r" (-> u32),
            : [p] "l" (@intFromPtr(p)),
            : .{ .memory = true });
    return @atomicLoad(u32, p, .acquire);
}

/// `@atomicStore(.release)` at device scope; see `loadAcquireDevice`.
pub inline fn storeReleaseDevice(p: *addrspace(.global) u32, v: u32) void {
    if (builtin.target.cpu.arch == .nvptx64 and comptime nv.sm >= 70)
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
    if (builtin.target.cpu.arch != .nvptx64) break :blk v;
    const info = @typeInfo(std.Target.nvptx.Feature).@"enum";
    for (info.field_names, info.field_values) |name, value| {
        if (!builtin.target.cpu.features.isEnabled(value)) continue;
        const digits = std.mem.trimEnd(u8, name, "af"); // sm_90a, sm_100f
        if (std.mem.startsWith(u8, digits, "ptx"))
            v.ptx = @max(v.ptx, std.fmt.parseInt(u32, digits[3..], 10) catch 0);
        if (std.mem.startsWith(u8, digits, "sm_"))
            v.sm = @max(v.sm, std.fmt.parseInt(u32, digits[3..], 10) catch 0);
    }
    break :blk v;
};
