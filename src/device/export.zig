//! Instantiate and export generated device kernels.
//!
//! Every elementwise entry moves its buffers 16 bytes per thread per access:
//! `spec.deviceLanes` consecutive elements, loaded as one aligned vector,
//! evaluated one lane at a time (the body is scalar; a wider load is what the
//! memory pipeline wants, a wider body is not what a GPU thread wants), and
//! stored as one vector. The last partial group of a buffer is the one thread
//! whose group straddles `len`, which handles it element by element. A warp
//! then issues `ld.global.v4`/`global_load_b128` and moves 512 bytes per
//! instruction. `tests/device_entries.zig` is checked for those instructions.
//!
//! The vector access assumes every buffer is 16-byte aligned at element 0,
//! which a driver allocation is (256 bytes on both vendors); `host/kernel.zig`
//! only ever passes whole buffers. A raw launch of these entries with a
//! pointer into the middle of an allocation is the caller's fault line.

const std = @import("std");
const builtin = @import("builtin");
const abi = @import("../core/abi.zig");
const spec = @import("../core/spec.zig");
const builtins = @import("builtins.zig");

/// The elements `[i, i + n)` of `p` as one vector, from a `[*]addrspace(.global)`
/// pointer. `i` is a multiple of `n` and `n * @sizeOf(T)` divides 16, so the
/// access is naturally aligned and lowers to a single vector load.
inline fn loadVec(comptime n: u32, comptime T: type, p: anytype, i: usize) @Vector(n, T) {
    const vp: *align(n * @sizeOf(T)) addrspace(.global) const @Vector(n, T) = @ptrCast(@alignCast(p + i));
    return vp.*;
}

inline fn storeVec(comptime n: u32, comptime T: type, p: [*]addrspace(.global) T, i: usize, v: @Vector(n, T)) void {
    const vp: *align(n * @sizeOf(T)) addrspace(.global) @Vector(n, T) = @ptrCast(@alignCast(p + i));
    vp.* = v;
}

/// `map` and `map_indexed` share one entry shape: same in-place signature, same
/// bounds guard, and `eval` differs only by the extra index argument. The host
/// already treats them as one case (`.map, .map_indexed => mapLaunch`).
fn Entry(comptime Spec: type, comptime indexed: bool) type {
    const T = Spec.Value;
    const n = spec.deviceLanes(Spec);
    return struct {
        inline fn one(x: T, i: usize, params: Spec.Parameters) T {
            return if (indexed) Spec.eval(x, @as(u64, i), params) else Spec.eval(x, params);
        }

        pub fn run(
            data: [*]addrspace(.global) T,
            len: u64,
            packed_params: Spec.BoundaryParameters,
        ) callconv(.kernel) void {
            const params = abi.unpack(Spec.Parameters, packed_params);
            var i = builtins.globalIdX(Spec.block_size) * n;
            if (n == 1) {
                if (i < len) data[i] = one(data[i], i, params);
                return;
            }
            if (i + n <= len) {
                var v = loadVec(n, T, data, i);
                inline for (0..n) |l| v[l] = one(v[l], i + l, params);
                storeVec(n, T, data, i, v);
                return;
            }
            // The one thread whose group straddles `len`: fewer than n
            // elements, spelled as n - 1 guarded steps so LLVM does not
            // unroll an open loop eight times for them.
            inline for (0..n - 1) |_| {
                if (i >= len) return;
                data[i] = one(data[i], i, params);
                i += 1;
            }
        }
    };
}

fn MapToEntry(comptime Spec: type) type {
    const n = spec.deviceLanes(Spec);
    return struct {
        pub fn run(
            in: [*]addrspace(.global) const Spec.In,
            out: [*]addrspace(.global) Spec.Out,
            len: u64,
            packed_params: Spec.BoundaryParameters,
        ) callconv(.kernel) void {
            const params = abi.unpack(Spec.Parameters, packed_params);
            var i = builtins.globalIdX(Spec.block_size) * n;
            if (n == 1) {
                if (i < len) out[i] = Spec.eval(in[i], params);
                return;
            }
            if (i + n <= len) {
                const x = loadVec(n, Spec.In, in, i);
                var y: @Vector(n, Spec.Out) = undefined;
                inline for (0..n) |l| y[l] = Spec.eval(x[l], params);
                storeVec(n, Spec.Out, out, i, y);
                return;
            }
            inline for (0..n - 1) |_| {
                if (i >= len) return;
                out[i] = Spec.eval(in[i], params);
                i += 1;
            }
        }
    };
}

fn ZipEntry(comptime Spec: type) type {
    const n = spec.deviceLanes(Spec);
    return struct {
        pub fn run(
            a: [*]addrspace(.global) const Spec.A,
            b: [*]addrspace(.global) const Spec.B,
            out: [*]addrspace(.global) Spec.Out,
            len: u64,
            packed_params: Spec.BoundaryParameters,
        ) callconv(.kernel) void {
            const params = abi.unpack(Spec.Parameters, packed_params);
            var i = builtins.globalIdX(Spec.block_size) * n;
            if (n == 1) {
                if (i < len) out[i] = Spec.eval(a[i], b[i], params);
                return;
            }
            if (i + n <= len) {
                const x = loadVec(n, Spec.A, a, i);
                const y = loadVec(n, Spec.B, b, i);
                var z: @Vector(n, Spec.Out) = undefined;
                inline for (0..n) |l| z[l] = Spec.eval(x[l], y[l], params);
                storeVec(n, Spec.Out, out, i, z);
                return;
            }
            inline for (0..n - 1) |_| {
                if (i >= len) return;
                out[i] = Spec.eval(a[i], b[i], params);
                i += 1;
            }
        }
    };
}

/// Grid-stride accumulate into a register, then fold the block, then one
/// partial per block. The host folds the partials.
///
/// A block size that is a multiple of 32 folds through `builtins.shuffleXor`:
/// five shuffles inside each 32-lane group, one shared slot per group, one
/// barrier, and the first group folds the slots the same way. That is one
/// barrier where the shared-memory tree takes log2(block_size), and the
/// shuffles lower to DPP / `v_permlanex16` / `ds_swizzle` on AMD and
/// `shfl.sync.bfly` on NVIDIA. Any other block size takes the tree. The order
/// of combination is fixed by the launch geometry either way.
fn ReduceEntry(comptime Spec: type) type {
    const T = Spec.Value;
    const bs = Spec.block_size;
    const n = spec.deviceLanes(Spec);
    // Halving starts at the next power of two so a non-power-of-two block still
    // pairs every live slot; the `tid + s < bs` guard drops the ones past the end.
    const tree: u32 = std.math.ceilPowerOfTwo(u32, bs) catch unreachable;
    const bits = @bitSizeOf(T);
    const shuffled = bs % 32 == 0 and (bits <= 32 or bits % 32 == 0);

    return struct {
        var scratch: [if (shuffled) bs / 32 else bs]T addrspace(.shared) = undefined;

        /// `acc` folded across this lane's 32-lane group; every lane gets it.
        inline fn groupFold(acc: T) T {
            var v = acc;
            inline for (.{ 16, 8, 4, 2, 1 }) |m| v = Spec.combine(v, builtins.shuffleXor(T, v, m));
            return v;
        }

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
            // walks a strided run of n-element groups rather than owning one.
            var acc: T = Spec.identity;
            var i = builtins.globalIdX(bs) * n;
            if (n > 1) {
                const step = stride * n;
                while (i + n <= len) : (i += step) {
                    const v = loadVec(n, T, data, i);
                    inline for (0..n) |l| acc = Spec.combine(acc, Spec.pre(v[l], params));
                }
                // Only the thread whose last group straddles `len` has any of
                // these, fewer than n; everyone else left the loop with
                // `i >= len`.
                inline for (0..n - 1) |_| {
                    if (i < len) {
                        acc = Spec.combine(acc, Spec.pre(data[i], params));
                        i += 1;
                    }
                }
            } else {
                while (i < len) : (i += stride) acc = Spec.combine(acc, Spec.pre(data[i], params));
            }

            if (comptime shuffled) {
                const folded = groupFold(acc);
                if (tid % 32 == 0) scratch[tid / 32] = folded;
                builtins.barrier();
                // Only the first group reads the slots; it is whole, so every
                // lane in it reaches the shuffles.
                if (tid < 32) {
                    const last = groupFold(if (tid < bs / 32) scratch[tid] else Spec.identity);
                    if (tid == 0) partials[builtins.blockIdX()] = last;
                }
                return;
            }

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
/// `bound` is the length of whichever buffer the index subscripts. One element
/// per thread: the subscripted side is a random access, which no vector load
/// helps.
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
///
/// Asserts at compile time that the target is `nvptx64` or `amdgcn`.
///
/// The module form force-analyses every pub decl for the GPU target, `@field`
/// first and `isSpec` second, so a pub decl that cannot compile for a device --
/// or a `pub var`, which `@field` cannot read at comptime at all -- errors
/// inside this file rather than at the decl that caused it.
///
/// No duplicate-`entry_name` check: two specs sharing a name become two strong
/// symbols, which surfaces as an `ld.lld` duplicate-symbol error on HIP and is
/// undefined on CUDA. `build.zig` does the equivalent check for root names.
pub fn exportAll(comptime specs: anytype) void {
    if (builtin.target.cpu.arch != .nvptx64 and builtin.target.cpu.arch != .amdgcn)
        @compileError("exportAll must be compiled for nvptx64 or amdgcn");

    if (@TypeOf(specs) == type) {
        inline for (@typeInfo(specs).@"struct".decl_names) |name| {
            const Spec = @field(specs, name);
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

/// The kind -> entry-shape table, and the one `@export` in the library. Each
/// `run` signature must match its `*Launch` argument array in
/// `src/host/kernel.zig` in count, order and width; nothing checks the pairing,
/// and a mismatch is garbage on the GPU rather than a compile error.
///
/// One asymmetry to know about: a `Parameters` of `struct {}` makes
/// `BoundaryParameters` zero-sized, and Zig drops zero-sized parameters from
/// the emitted ABI, so the entry takes one fewer argument than `mapLaunch`
/// pushes. Harmless today -- both drivers read only as many argument pointers
/// as the kernel declares -- but it is a reliance on driver behaviour, not on
/// anything this code guarantees.
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
