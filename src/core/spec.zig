//! A map specification is metadata plus one pure scalar function.

const std = @import("std");
const abi = @import("abi.zig");

pub const MapOptions = struct {
    block_size: u32 = 256,
};

/// Which generated kernel body and which host `run` signature a spec asks for.
///
/// Read through `kindOf`, never directly: specs built before this existed have
/// no `kind` decl at all and mean `.map`.
pub const Kind = enum { map, map_to, zip, reduce, map_indexed, gather, scatter };

pub fn kindOf(comptime Spec: type) Kind {
    return if (@hasDecl(Spec, "kind")) Spec.kind else .map;
}

/// Broadcast a scalar so one generic map body compiles at scalar and vector
/// width: `x * splat(@TypeOf(x), p.scale)`.
pub inline fn splat(comptime T: type, value: anytype) T {
    return if (@typeInfo(T) == .vector) @splat(value) else value;
}

pub fn Map(
    comptime name: [:0]const u8,
    comptime T: type,
    comptime Params: type,
    comptime func: anytype,
    comptime options: MapOptions,
) type {
    validateName(name);
    validateValue(T);
    const generic = validateFunction(T, Params, func);
    _ = abi.Boundary(Params);
    validateBlockSize(options.block_size);

    return struct {
        pub const kind: Kind = .map;
        pub const entry_name: [:0]const u8 = name;
        pub const Value = T;
        pub const Parameters = Params;
        pub const BoundaryParameters = abi.Boundary(Params);
        pub const block_size: u32 = options.block_size;
        /// The map function takes `anytype`, so `eval` also instantiates at
        /// `@Vector(n, Value)`; the CPU backend uses that to vectorize.
        pub const is_generic: bool = generic;
        pub const eval = Eval(generic, T, Params, func).eval;
    };
}

/// Out-of-place map: `out[i] = func(in[i], params)`, with a result type that
/// need not match the input's.
///
/// Unlike `Map`, `func` is a concrete `fn (In, Params) Out` -- there is no
/// generic form, so the CPU path stays scalar. Vectorizing it needs a body that
/// instantiates at `@Vector`, which is what in-place `map` already offers.
pub fn MapTo(
    comptime name: [:0]const u8,
    comptime Input: type,
    comptime Output: type,
    comptime Params: type,
    comptime func: fn (Input, Params) Output,
    comptime options: MapOptions,
) type {
    validateName(name);
    validateValue(Input);
    validateValue(Output);
    _ = abi.Boundary(Params);
    validateBlockSize(options.block_size);

    return struct {
        pub const kind: Kind = .map_to;
        pub const entry_name: [:0]const u8 = name;
        pub const In = Input;
        pub const Out = Output;
        /// What `alloc` sizes: for an out-of-place op, the result element.
        pub const Value = Output;
        pub const Parameters = Params;
        pub const BoundaryParameters = abi.Boundary(Params);
        pub const block_size: u32 = options.block_size;

        pub inline fn eval(x: Input, params: Params) Output {
            return @call(.always_inline, func, .{ x, params });
        }
    };
}

/// Elementwise binary map: `out[i] = func(a[i], b[i], params)`.
///
/// Two buffers, two positional arguments. Packing the operands into one
/// `[]struct { a: A, b: B }` would work through `map` alone, but that is
/// array-of-structs -- it halves effective bandwidth on both vendors by
/// interleaving two streams that each want to be coalesced on their own.
pub fn Zip(
    comptime name: [:0]const u8,
    comptime First: type,
    comptime Second: type,
    comptime Output: type,
    comptime Params: type,
    comptime func: fn (First, Second, Params) Output,
    comptime options: MapOptions,
) type {
    validateName(name);
    validateValue(First);
    validateValue(Second);
    validateValue(Output);
    _ = abi.Boundary(Params);
    validateBlockSize(options.block_size);

    return struct {
        pub const kind: Kind = .zip;
        pub const entry_name: [:0]const u8 = name;
        pub const A = First;
        pub const B = Second;
        pub const Out = Output;
        pub const Value = Output;
        pub const Parameters = Params;
        pub const BoundaryParameters = abi.Boundary(Params);
        pub const block_size: u32 = options.block_size;

        pub inline fn eval(a: First, b: Second, params: Params) Output {
            return @call(.always_inline, func, .{ a, b, params });
        }
    };
}

/// In-place map that also sees the element's linear index:
/// `data[i] = func(data[i], i, params)`.
///
/// ponytail: one linear index, no `globalIdY`/`globalIdZ` and no 2-D launch.
/// A caller who wants (row, col) divides by the row stride, which is free next
/// to the global load it sits on, and a u64 index does not run out until ~1.8e19
/// elements. Real 2-D kernels want block tiling and shared-memory staging, which
/// is a different library, not a second grid dimension.
pub fn MapIndexed(
    comptime name: [:0]const u8,
    comptime T: type,
    comptime Params: type,
    comptime func: fn (T, u64, Params) T,
    comptime options: MapOptions,
) type {
    validateName(name);
    validateValue(T);
    _ = abi.Boundary(Params);
    validateBlockSize(options.block_size);

    return struct {
        pub const kind: Kind = .map_indexed;
        pub const entry_name: [:0]const u8 = name;
        pub const Value = T;
        pub const Parameters = Params;
        pub const BoundaryParameters = abi.Boundary(Params);
        pub const block_size: u32 = options.block_size;

        pub inline fn eval(x: T, index: u64, params: Params) T {
            return @call(.always_inline, func, .{ x, index, params });
        }
    };
}

pub fn ReduceOptions(comptime T: type, comptime Params: type) type {
    return struct {
        block_size: u32 = 256,
        /// Applied to each element before it is combined -- Thrust's
        /// `transform_reduce`. Sum-of-squares, L2 norm, "how many match" and
        /// "does any exceed k" are all one launch with this set, instead of a
        /// `mapTo` into a scratch buffer followed by a `reduce` over it.
        pre: ?*const fn (T, Params) T = null,
        /// Set by the presets so the CPU path can use `@reduce` over a
        /// `@Vector`. Left null for a custom `combine`: a generic `fn (T, T) T`
        /// cannot be lane-widened, and float addition is not reassociative, so
        /// LLVM will not widen it either.
        simd_op: ?std.builtin.ReduceOp = null,
    };
}

/// Whole-buffer reduction with an associative `combine` and its `identity`.
///
/// `combine` must be associative and `identity` must be its neutral element:
/// the device reassociates freely across threads and blocks, so a
/// non-associative op gives an answer that changes with the launch geometry.
/// Float addition is only approximately associative -- expect a result that
/// differs from a strict left-to-right CPU sum in the last bits.
pub fn Reduce(
    comptime name: [:0]const u8,
    comptime T: type,
    comptime Params: type,
    comptime combine_fn: fn (T, T) T,
    comptime identity_value: T,
    comptime options: ReduceOptions(T, Params),
) type {
    validateName(name);
    validateValue(T);
    _ = abi.Boundary(Params);
    validateBlockSize(options.block_size);

    // Always a real function, so `pre` needs no null check at any use site and
    // the default inlines away to nothing.
    const pre_fn: *const fn (T, Params) T = options.pre orelse struct {
        fn identityPre(x: T, _: Params) T {
            return x;
        }
    }.identityPre;

    return struct {
        pub const kind: Kind = .reduce;
        pub const entry_name: [:0]const u8 = name;
        pub const Value = T;
        pub const Parameters = Params;
        pub const BoundaryParameters = abi.Boundary(Params);
        pub const block_size: u32 = options.block_size;
        pub const identity: T = identity_value;
        pub const simd_op: ?std.builtin.ReduceOp = options.simd_op;

        pub inline fn combine(a: T, b: T) T {
            return @call(.always_inline, combine_fn, .{ a, b });
        }

        pub inline fn pre(x: T, params: Params) T {
            return @call(.always_inline, pre_fn, .{ x, params });
        }
    };
}

/// `out[i] = src[idx[i]]`.
///
/// A fixed body rather than a user function: reading `src` at an arbitrary
/// offset needs a raw device pointer, and handing one to user code breaks the
/// pure-scalar-function model that lets the same source run on the CPU.
///
/// An index at or past `src.len` leaves `out[i]` unspecified -- the device
/// checks the bound and skips, so it can never read outside the buffer, but it
/// does not agree with the CPU path on what the untouched slot holds.
pub fn Gather(
    comptime name: [:0]const u8,
    comptime T: type,
    comptime Idx: type,
    comptime options: MapOptions,
) type {
    return IndexedCopy(.gather, name, T, Idx, options);
}

/// `out[idx[i]] = src[i]`.
///
/// Duplicate indices are NONDETERMINISTIC: two threads writing the same slot
/// race, and which one lands is unspecified on both vendors. An index at or
/// past `out.len` drops that element. Elements of `out` no index selects keep
/// their prior value -- `out` is uploaded, not just allocated.
///
/// ponytail: no `scatterAdd`. Combining duplicates needs a device float atomic,
/// and `global_atomic_add_f32` is gfx9+/`--unsafe-fp-atomics` on AMD -- a
/// portability swamp for one operation. Use `RawKernel` if you need it.
pub fn Scatter(
    comptime name: [:0]const u8,
    comptime T: type,
    comptime Idx: type,
    comptime options: MapOptions,
) type {
    return IndexedCopy(.scatter, name, T, Idx, options);
}

fn IndexedCopy(
    comptime k: Kind,
    comptime name: [:0]const u8,
    comptime T: type,
    comptime Idx: type,
    comptime options: MapOptions,
) type {
    validateName(name);
    validateValue(T);
    validateIndex(Idx);
    validateBlockSize(options.block_size);

    return struct {
        pub const kind: Kind = k;
        pub const entry_name: [:0]const u8 = name;
        pub const Value = T;
        pub const Index = Idx;
        pub const block_size: u32 = options.block_size;
    };
}

/// `Map` with `T` and `Params` read off the function signature.
pub fn MapFn(
    comptime name: [:0]const u8,
    comptime func: anytype,
    comptime options: MapOptions,
) type {
    const f = fnInfo(func);
    const T = f.params[0].type orelse @compileError(
        "cannot infer the value type of a generic map function; use the " ++
            "5-argument form g.map(name, T, Params, func, options)",
    );
    const P = f.params[1].type orelse @compileError(
        "cannot infer the parameter type; use g.map(name, T, Params, func, options)",
    );
    return Map(name, T, P, func, options);
}

// ---------------------------------------------------------------------------
// Reduce presets. The five reductions people actually write, each with the
// `simd_op` its CPU path needs -- a custom `combine` cannot supply one.
// All of them still take `pre`, which is where the interesting ones live:
// sum-of-squares, L2 norm, "how many match", "does any exceed k".
// ---------------------------------------------------------------------------

pub fn Sum(
    comptime name: [:0]const u8,
    comptime T: type,
    comptime Params: type,
    comptime options: ReduceOptions(T, Params),
) type {
    return Reduce(name, T, Params, Arith(T).add, 0, withOp(options, .Add));
}

pub fn Min(
    comptime name: [:0]const u8,
    comptime T: type,
    comptime Params: type,
    comptime options: ReduceOptions(T, Params),
) type {
    return Reduce(name, T, Params, Arith(T).min, Arith(T).largest, withOp(options, .Min));
}

pub fn Max(
    comptime name: [:0]const u8,
    comptime T: type,
    comptime Params: type,
    comptime options: ReduceOptions(T, Params),
) type {
    return Reduce(name, T, Params, Arith(T).max, Arith(T).smallest, withOp(options, .Max));
}

/// Bitwise OR, i.e. "is any element nonzero". Integer only; write the predicate
/// as `pre` returning 0 or 1: `any(x > k)` is `.pre = &gtK`.
pub fn Any(
    comptime name: [:0]const u8,
    comptime T: type,
    comptime Params: type,
    comptime options: ReduceOptions(T, Params),
) type {
    requireInt(T, "any");
    return Reduce(name, T, Params, Arith(T).bitOr, 0, withOp(options, .Or));
}

/// Bitwise AND, i.e. "is every element all-ones". Integer only; write the
/// predicate as `pre` returning `0` or `~@as(T, 0)`.
pub fn All(
    comptime name: [:0]const u8,
    comptime T: type,
    comptime Params: type,
    comptime options: ReduceOptions(T, Params),
) type {
    requireInt(T, "all");
    return Reduce(name, T, Params, Arith(T).bitAnd, ~@as(T, 0), withOp(options, .And));
}

fn requireInt(comptime T: type, comptime what: []const u8) void {
    if (@typeInfo(T) != .int)
        @compileError(what ++ " combines bitwise and needs an integer value type, not " ++ @typeName(T));
}

fn withOp(comptime options: anytype, comptime op: std.builtin.ReduceOp) @TypeOf(options) {
    var out = options;
    out.simd_op = op;
    return out;
}

fn Arith(comptime T: type) type {
    const int = @typeInfo(T) == .int;
    return struct {
        /// Wrapping for integers: the device runs ReleaseFast and would wrap
        /// silently anyway, so the CPU path must not disagree by panicking.
        pub fn add(a: T, b: T) T {
            return if (int) a +% b else a + b;
        }
        pub fn min(a: T, b: T) T {
            return @min(a, b);
        }
        pub fn max(a: T, b: T) T {
            return @max(a, b);
        }
        pub fn bitOr(a: T, b: T) T {
            return if (int) a | b else unreachable;
        }
        pub fn bitAnd(a: T, b: T) T {
            return if (int) a & b else unreachable;
        }
        pub const largest: T = if (int) std.math.maxInt(T) else std.math.inf(T);
        pub const smallest: T = if (int) std.math.minInt(T) else -std.math.inf(T);
    };
}

fn Eval(comptime generic: bool, comptime T: type, comptime P: type, comptime func: anytype) type {
    return if (generic) struct {
        pub inline fn eval(x: anytype, params: P) @TypeOf(x) {
            return @call(.always_inline, func, .{ x, params });
        }
    } else struct {
        pub inline fn eval(x: T, params: P) T {
            return @call(.always_inline, func, .{ x, params });
        }
    };
}

fn validateName(comptime name: []const u8) void {
    if (name.len == 0) @compileError("kernel entry name cannot be empty");
    if (!isCIdentifier(name)) @compileError("kernel names must be C identifiers: " ++ name);
}

/// The entry name is emitted verbatim as a PTX/HSA symbol and looked up by
/// string, so anything a C compiler would not accept cannot be a kernel.
/// Split out from `validateName` because a `@compileError` cannot be tested.
fn isCIdentifier(name: []const u8) bool {
    if (name.len == 0) return false;
    for (name, 0..) |c, i| switch (c) {
        'a'...'z', 'A'...'Z', '_' => {},
        '0'...'9' => if (i == 0) return false,
        else => return false,
    };
    return true;
}

/// 1024 is the maximum threads-per-block on every CUDA compute capability and
/// on every AMD gfx target gompute targets.
fn blockSizeValid(n: u32) bool {
    return n > 0 and n <= 1024;
}

fn validateBlockSize(comptime n: u32) void {
    if (!blockSizeValid(n)) @compileError("block_size must be in 1...1024");
}

/// Gather/scatter subscripts. Unsigned only -- a negative subscript has no
/// meaning and would silently wrap into the bounds check on the device.
fn validateIndex(comptime T: type) void {
    if (!abi.fixedWidthInt(T) or @typeInfo(T).int.signedness != .unsigned)
        @compileError("index type must be u8, u16, u32, or u64, not " ++ @typeName(T));
}

fn validateValue(comptime T: type) void {
    switch (@typeInfo(T)) {
        .int => if (!abi.fixedWidthInt(T))
            @compileError("map value must be a fixed-width 8/16/32/64-bit integer, not " ++ @typeName(T)),
        .float => if (!abi.fixedWidthFloat(T))
            @compileError("map value must be f16, f32, or f64, not " ++ @typeName(T)),
        .vector => |v| validateValue(v.child),
        else => @compileError("map value must be an integer, float, or vector: " ++ @typeName(T)),
    }
}

fn fnInfo(comptime func: anytype) std.builtin.Type.Fn {
    const info = @typeInfo(@TypeOf(func));
    if (info != .@"fn") @compileError("map operation must be a function");
    const f = info.@"fn";
    if (f.is_var_args or f.params.len != 2)
        @compileError("map operation must have signature fn (T, Params) T");
    return f;
}

/// Returns true for a generic map function (`fn (x: anytype, p: Params)`),
/// which the CPU backend may instantiate at vector width.
fn validateFunction(comptime T: type, comptime P: type, comptime func: anytype) bool {
    const f = fnInfo(func);
    const b = f.params[1].type orelse @compileError("the params argument cannot be anytype");
    if (b != P)
        @compileError("map operation must take " ++ @typeName(P) ++ " as its second argument");
    if (f.params[0].type == null) return true;
    const r = f.return_type orelse @compileError("map operation must return T");
    if (f.params[0].type.? != T or r != T)
        @compileError("map operation must have signature fn (" ++ @typeName(T) ++ ", " ++ @typeName(P) ++ ") " ++ @typeName(T));
    return false;
}

test "map spec validates and evaluates" {
    const P = struct { scale: f32 };
    const op = struct {
        fn call(x: f32, p: P) f32 {
            return x * p.scale;
        }
    }.call;
    const S = Map("scale", f32, P, op, .{});
    try std.testing.expectEqual(@as(f32, 6), S.eval(3, .{ .scale = 2 }));
    try std.testing.expectEqual(@as(u32, 256), S.block_size);
    try std.testing.expectEqualStrings("scale", S.entry_name);

    // The bounds are real: both ends compile, and they reach the spec.
    try std.testing.expectEqual(@as(u32, 1), Map("a", f32, P, op, .{ .block_size = 1 }).block_size);
    try std.testing.expectEqual(@as(u32, 1024), Map("b", f32, P, op, .{ .block_size = 1024 }).block_size);
}

test "kernel names must be C identifiers" {
    for ([_][]const u8{ "scale", "_x", "a0", "MAP_9" }) |ok|
        try std.testing.expect(isCIdentifier(ok));
    // Leading digit, separators, namespacing, and anything a linker would choke
    // on -- the name is emitted as a symbol verbatim.
    for ([_][]const u8{ "", "0scale", "my kernel", "my-kernel", "my.kernel", "add()", "über" }) |bad|
        try std.testing.expect(!isCIdentifier(bad));
}

test "block_size bounds" {
    try std.testing.expect(!blockSizeValid(0));
    try std.testing.expect(blockSizeValid(1));
    try std.testing.expect(blockSizeValid(256));
    try std.testing.expect(blockSizeValid(1024));
    try std.testing.expect(!blockSizeValid(1025));
    try std.testing.expect(!blockSizeValid(std.math.maxInt(u32)));
}
