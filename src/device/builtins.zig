//! Device-only thread-index builtins for Zig 0.16.0.

const std = @import("std");
const builtin = @import("builtin");

extern fn @"llvm.amdgcn.workitem.id.x"() callconv(.c) u32;
extern fn @"llvm.amdgcn.workgroup.id.x"() callconv(.c) u32;

extern fn @"llvm.nvvm.read.ptx.sreg.tid.x"() callconv(.c) u32;
extern fn @"llvm.nvvm.read.ptx.sreg.ctaid.x"() callconv(.c) u32;
extern fn @"llvm.nvvm.read.ptx.sreg.ntid.x"() callconv(.c) u32;

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
