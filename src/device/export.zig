//! Instantiate and export generated device kernels.

const builtin = @import("builtin");
const abi = @import("../core/abi.zig");
const builtins = @import("builtins.zig");

fn Entry(comptime Spec: type) type {
    return struct {
        pub fn run(
            data: [*]addrspace(.global) Spec.Value,
            len: u64,
            packed_params: Spec.BoundaryParameters,
        ) callconv(.kernel) void {
            const i = builtins.globalIdX(Spec.block_size);
            if (i >= len) return;
            const params = abi.unpack(Spec.Parameters, packed_params);
            data[i] = Spec.eval(data[i], params);
        }
    };
}

pub fn exportAll(comptime specs: anytype) void {
    if (builtin.cpu.arch != .nvptx64 and builtin.cpu.arch != .amdgcn)
        @compileError("exportAll must be compiled for nvptx64 or amdgcn");

    inline for (specs) |Spec| {
        const E = Entry(Spec);
        @export(&E.run, .{ .name = Spec.entry_name });
    }
}
