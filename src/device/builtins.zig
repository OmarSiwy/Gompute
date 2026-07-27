//! Device-only thread-index builtins for Zig 0.16.0.

const std = @import("std");
const builtin = @import("builtin");

extern fn @"llvm.amdgcn.workitem.id.x"() callconv(.c) u32;
extern fn @"llvm.amdgcn.workgroup.id.x"() callconv(.c) u32;
extern fn @"llvm.amdgcn.s.barrier"() callconv(.c) void;

extern fn @"llvm.nvvm.read.ptx.sreg.tid.x"() callconv(.c) u32;
extern fn @"llvm.nvvm.read.ptx.sreg.ctaid.x"() callconv(.c) u32;
extern fn @"llvm.nvvm.read.ptx.sreg.ntid.x"() callconv(.c) u32;
extern fn @"llvm.nvvm.read.ptx.sreg.nctaid.x"() callconv(.c) u32;
extern fn @"llvm.nvvm.barrier0"() callconv(.c) void;

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
/// indices on CUDA and silently wrong ones on HIP.
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
        .amdgcn => @as(usize, @"llvm.amdgcn.workgroup.id.x"()) * block_size +
            @"llvm.amdgcn.workitem.id.x"(),
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

/// Block-wide execution barrier plus a shared-memory fence: every thread in the
/// block reaches it before any thread passes, and shared writes made before it
/// are visible to the whole block after it.
///
/// Must be reached by every thread in the block. Calling it inside a branch that
/// only some threads take hangs the block on NVIDIA and is undefined on AMD.
pub inline fn barrier() void {
    switch (builtin.cpu.arch) {
        .nvptx64 => @"llvm.nvvm.barrier0"(),
        .amdgcn => @"llvm.amdgcn.s.barrier"(),
        else => @compileError("device barrier used on a non-GPU target"),
    }
}
