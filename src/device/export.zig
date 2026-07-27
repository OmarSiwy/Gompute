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

/// Accepts a tuple of map specs, or a module/struct type — in which case every
/// pub decl that looks like a map spec is exported, in declaration order.
pub fn exportAll(comptime specs: anytype) void {
    if (builtin.cpu.arch != .nvptx64 and builtin.cpu.arch != .amdgcn)
        @compileError("exportAll must be compiled for nvptx64 or amdgcn");

    if (@TypeOf(specs) == type) {
        inline for (@typeInfo(specs).@"struct".decls) |decl| {
            const Spec = @field(specs, decl.name);
            if (comptime isSpec(Spec)) exportOne(Spec);
        }
    } else {
        inline for (specs) |Spec| exportOne(Spec);
    }
}

fn isSpec(comptime Spec: anytype) bool {
    if (@TypeOf(Spec) != type) return false;
    if (@typeInfo(Spec) != .@"struct") return false;
    return @hasDecl(Spec, "entry_name") and @hasDecl(Spec, "eval");
}

fn exportOne(comptime Spec: type) void {
    const E = Entry(Spec);
    @export(&E.run, .{ .name = Spec.entry_name });
}
