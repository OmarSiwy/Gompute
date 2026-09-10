//! Instantiate and export generated device kernels.

const std = @import("std");
const builtin = @import("builtin");
const abi = @import("../core/abi.zig");
const spec = @import("../core/spec.zig");
const builtins = @import("builtins.zig");

/// `map` and `map_indexed` share one entry shape: same in-place signature, same
/// bounds guard, and `eval` differs only by the extra index argument. The host
/// already treats them as one case (`.map, .map_indexed => mapLaunch`).
fn Entry(comptime Spec: type, comptime indexed: bool) type {
    return struct {
        pub fn run(
            data: [*]addrspace(.global) Spec.Value,
            len: u64,
            packed_params: Spec.BoundaryParameters,
        ) callconv(.kernel) void {
            const i = builtins.globalIdX(Spec.block_size);
            if (i >= len) return;
            const params = abi.unpack(Spec.Parameters, packed_params);
            data[i] = if (indexed)
                Spec.eval(data[i], @as(u64, i), params)
            else
                Spec.eval(data[i], params);
        }
    };
}

fn MapToEntry(comptime Spec: type) type {
    return struct {
        pub fn run(
            in: [*]addrspace(.global) const Spec.In,
            out: [*]addrspace(.global) Spec.Out,
            len: u64,
            packed_params: Spec.BoundaryParameters,
        ) callconv(.kernel) void {
            const i = builtins.globalIdX(Spec.block_size);
            if (i >= len) return;
            const params = abi.unpack(Spec.Parameters, packed_params);
            out[i] = Spec.eval(in[i], params);
        }
    };
}

fn ZipEntry(comptime Spec: type) type {
    return struct {
        pub fn run(
            a: [*]addrspace(.global) const Spec.A,
            b: [*]addrspace(.global) const Spec.B,
            out: [*]addrspace(.global) Spec.Out,
            len: u64,
            packed_params: Spec.BoundaryParameters,
        ) callconv(.kernel) void {
            const i = builtins.globalIdX(Spec.block_size);
            if (i >= len) return;
            const params = abi.unpack(Spec.Parameters, packed_params);
            out[i] = Spec.eval(a[i], b[i], params);
        }
    };
}

/// Grid-stride accumulate into a register, then one shared-memory tree per
/// block, then one partial per block. The host folds the partials.
///
/// ponytail: shared memory, not warp shuffles. `shfl.sync` (NVPTX) and
/// `ds_bpermute`/DPP (AMDGCN) do not have a common spelling, and AMD's wave
/// width is 32 on RDNA and 64 on GCN/CDNA, so the shuffle version needs a
/// per-target reduction written three ways. This one is ~15% slower and
/// identical on both vendors.
fn ReduceEntry(comptime Spec: type) type {
    const T = Spec.Value;
    const bs = Spec.block_size;
    // Halving starts at the next power of two so a non-power-of-two block still
    // pairs every live slot; the `tid + s < bs` guard drops the ones past the end.
    const tree: u32 = std.math.ceilPowerOfTwo(u32, bs) catch unreachable;

    return struct {
        var scratch: [bs]T addrspace(.shared) = undefined;

        /// `stride` is the total thread count, handed over by the host rather
        /// than read back off the device. `gridDim.x` has no portable spelling
        /// -- see `builtins.gridDimX` -- and the host computed the grid anyway,
        /// so asking the hardware what it was just told is the long way round.
        pub fn run(
            data: [*]addrspace(.global) const T,
            len: u64,
            partials: [*]addrspace(.global) T,
            stride: u64,
            packed_params: Spec.BoundaryParameters,
        ) callconv(.kernel) void {
            const params = abi.unpack(Spec.Parameters, packed_params);
            const tid = builtins.localIdX();

            // The grid is capped well below the element count, so every thread
            // walks a strided run rather than owning one element.
            var acc: T = Spec.identity;
            var i = builtins.globalIdX(bs);
            while (i < len) : (i += stride)
                acc = Spec.combine(acc, Spec.pre(data[i], params));

            scratch[tid] = acc;
            builtins.barrier();

            comptime var s: u32 = tree / 2;
            inline while (s > 0) : (s /= 2) {
                if (tid < s and tid + s < bs)
                    scratch[tid] = Spec.combine(scratch[tid], scratch[tid + s]);
                // Outside the `if` on purpose: every thread in the block must
                // reach every barrier.
                builtins.barrier();
            }

            if (tid == 0) partials[builtins.blockIdX()] = scratch[0];
        }
    };
}

/// `gather` and `scatter` differ by one line, so they share one entry shape:
/// `bound` is the length of whichever buffer the index subscripts.
fn IndexedCopyEntry(comptime Spec: type, comptime gathering: bool) type {
    const T = Spec.Value;
    return struct {
        pub fn run(
            src: [*]addrspace(.global) const T,
            idx: [*]addrspace(.global) const Spec.Index,
            out: [*]addrspace(.global) T,
            len: u64,
            bound: u64,
        ) callconv(.kernel) void {
            const i = builtins.globalIdX(Spec.block_size);
            if (i >= len) return;
            const j: usize = idx[i];
            // ponytail: skip, do not clamp and do not fault. An index past the
            // end is user data, and without this it is an out-of-bounds device
            // write -- silent corruption of whatever else is on the GPU. The
            // host cannot pre-validate without an O(n) pass that defeats the
            // point of the launch, so the check rides along here.
            if (j >= bound) return;
            if (gathering) out[i] = src[j] else out[j] = src[i];
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
    // `eval` covers every spec with a user function; `kind` covers gather and
    // scatter, whose bodies are fixed and take none.
    return @hasDecl(Spec, "entry_name") and (@hasDecl(Spec, "eval") or @hasDecl(Spec, "kind"));
}

fn exportOne(comptime Spec: type) void {
    const E = switch (comptime spec.kindOf(Spec)) {
        .map => Entry(Spec, false),
        .map_indexed => Entry(Spec, true),
        .map_to => MapToEntry(Spec),
        .zip => ZipEntry(Spec),
        .reduce => ReduceEntry(Spec),
        .gather => IndexedCopyEntry(Spec, true),
        .scatter => IndexedCopyEntry(Spec, false),
    };
    @export(&E.run, .{ .name = Spec.entry_name });
}
