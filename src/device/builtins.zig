//! Device-only thread-index builtins for Zig 0.16.0.

const std = @import("std");
const builtin = @import("builtin");

extern fn @"llvm.amdgcn.workitem.id.x"() callconv(.c) u32;
extern fn @"llvm.amdgcn.workgroup.id.x"() callconv(.c) u32;

extern fn @"llvm.nvvm.read.ptx.sreg.tid.x"() callconv(.c) u32;
extern fn @"llvm.nvvm.read.ptx.sreg.ctaid.x"() callconv(.c) u32;
extern fn @"llvm.nvvm.read.ptx.sreg.ntid.x"() callconv(.c) u32;

pub inline fn globalIdX(comptime block_size: u32) usize {
    return switch (builtin.cpu.arch) {
        .nvptx64 => @as(usize, @"llvm.nvvm.read.ptx.sreg.ctaid.x"()) *
            @"llvm.nvvm.read.ptx.sreg.ntid.x"() + @"llvm.nvvm.read.ptx.sreg.tid.x"(),
        .amdgcn => @as(usize, @"llvm.amdgcn.workgroup.id.x"()) * block_size +
            @"llvm.amdgcn.workitem.id.x"(),
        else => @compileError("device globalIdX used on a non-GPU target"),
    };
}
