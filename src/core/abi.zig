//! Compile-time host/device boundary derivation.

const std = @import("std");

pub fn Boundary(comptime T: type) type {
    return BoundaryAt(T, @typeName(T));
}

/// The integer widths that mean the same thing in both compilations: exactly
/// 8, 16, 32 or 64 bits, and never `usize`/`isize`, whose width is the target's
/// pointer width and so is not a property of the type at all.
///
/// This predicate is the width half of the boundary rule. `spec.zig` validates
/// value and index types against it too, so the rule has one definition rather
/// than one per caller to drift out of sync.
pub fn fixedWidthInt(comptime T: type) bool {
    const info = @typeInfo(T);
    if (info != .int or T == usize or T == isize) return false;
    return switch (info.int.bits) {
        8, 16, 32, 64 => true,
        else => false,
    };
}

/// f16, f32 and f64 -- the float widths both vendors' hardware implements.
/// The float half of the boundary rule; see `fixedWidthInt`.
pub fn fixedWidthFloat(comptime T: type) bool {
    const info = @typeInfo(T);
    if (info != .float) return false;
    return switch (info.float.bits) {
        16, 32, 64 => true,
        else => false,
    };
}

/// `at` is the dotted path from the root parameter type down to whatever we are
/// currently deriving, so a rejection names the field the user actually wrote
/// rather than the innermost type the recursion happened to reach.
fn BoundaryAt(comptime T: type, comptime at: []const u8) type {
    return switch (@typeInfo(T)) {
        .bool => u8,
        .int => blk: {
            // Split from the width check so the pointer-width case, which is the
            // one people actually hit, gets its own diagnostic.
            if (T == usize or T == isize) @compileError(reject(at, T, "host and device pointer widths may differ, so usize/isize have no fixed size across the boundary; use u32/u64 or i32/i64"));
            if (!fixedWidthInt(T))
                @compileError(reject(at, T, "integers must be exactly 8, 16, 32, or 64 bits wide"));
            break :blk T;
        },
        .float => blk: {
            if (!fixedWidthFloat(T))
                @compileError(reject(at, T, "floats must be f16, f32, or f64"));
            break :blk T;
        },
        .@"enum" => |e| BoundaryAt(e.tag_type, at),
        .array => |a| [a.len]BoundaryAt(a.child, at ++ "[_]"),
        .vector => |v| @Vector(v.len, BoundaryAt(v.child, at ++ "[_]")),
        .@"struct" => |s| boundaryStruct(T, s, at),
        else => @compileError(reject(at, T, "only scalars and aggregates of scalars have a stable GPU representation")),
    };
}

fn reject(comptime at: []const u8, comptime T: type, comptime why: []const u8) []const u8 {
    return "gompute: `" ++ at ++ "` has type `" ++ @typeName(T) ++ "`, which cannot cross the GPU boundary.\n" ++
        "  " ++ why ++ ".\n" ++
        "  Supported: 8/16/32/64-bit integers, f16/f32/f64, bool, enums with those tag types,\n" ++
        "  and arrays, vectors and structs composed of those.\n" ++
        "  To pass anything else, declare on the struct: `pub const gpu_layout`,\n" ++
        "  `pub fn toGpu(Self) gpu_layout` and `pub fn fromGpu(gpu_layout) Self`.";
}

fn boundaryStruct(comptime T: type, comptime info: std.builtin.Type.Struct, comptime at: []const u8) type {
    // All three of gpu_layout/toGpu/fromGpu are required together. Declaring a
    // subset used to derive a field-wise layout and then fail inside pack() with
    // a raw type mismatch pointing into this file.
    const has_layout = @hasDecl(T, "gpu_layout");
    const has_to = @hasDecl(T, "toGpu");
    const has_from = @hasDecl(T, "fromGpu");
    if ((has_layout or has_to or has_from) and !(has_layout and has_to and has_from))
        @compileError("gompute: `" ++ @typeName(T) ++ "` declares " ++
            (if (has_layout) "`gpu_layout`" else if (has_to) "`toGpu`" else "`fromGpu`") ++
            " but not all three parts of a custom GPU boundary. Declare:\n" ++
            "  pub const gpu_layout = <an extern-safe wire type>;\n" ++
            "  pub fn toGpu(value: " ++ @typeName(T) ++ ") gpu_layout { ... }\n" ++
            "  pub fn fromGpu(wire: gpu_layout) " ++ @typeName(T) ++ " { ... }\n" ++
            "  Or remove all three and let gompute derive the layout field by field.");

    if (has_layout) {
        const U = T.gpu_layout;
        const path = @typeName(T) ++ ".gpu_layout";
        assertFixedLayout(U, path);
        // A packed struct is one integer with a fixed bit layout, so it is that
        // backing integer that crosses, not the sub-byte fields.
        const info_u = @typeInfo(U);
        if (info_u == .@"struct" and info_u.@"struct".layout == .@"packed")
            _ = BoundaryAt(info_u.@"struct".backing_integer.?, path)
        else
            _ = BoundaryAt(U, path);
        return U;
    }

    comptime var names: [info.fields.len][]const u8 = undefined;
    comptime var types: [info.fields.len]type = undefined;
    comptime var attrs: [info.fields.len]std.builtin.Type.StructField.Attributes = undefined;
    inline for (info.fields, 0..) |field, i| {
        if (field.is_comptime)
            @compileError("gompute: `" ++ at ++ "." ++ field.name ++ "` is a comptime field, which has no runtime representation to send to the GPU. Remove it from the params struct.");
        names[i] = field.name;
        types[i] = BoundaryAt(field.type, at ++ "." ++ field.name);
        attrs[i] = .{};
    }

    return @Struct(.@"extern", null, &names, &types, &attrs);
}

/// `gpu_layout` is the one type gompute does not derive, so it is the one type
/// whose layout has to be checked by hand. `BoundaryAt` only vets the element
/// types and throws its result away, which leaves an `.auto` struct -- no
/// guaranteed field order, no guaranteed padding -- accepted by the very hatch
/// that exists to make the boundary safe. Host and device are separate
/// compilations for different targets, so both ends must agree by construction.
///
/// Only the outermost type needs checking: Zig already rejects a non-extern
/// struct field inside an `extern struct` and a non-packed one inside a
/// `packed struct`.
fn assertFixedLayout(comptime U: type, comptime at: []const u8) void {
    if (!hasFixedLayout(U)) @compileError(
        "gompute: `" ++ at ++ "` is `" ++ @typeName(U) ++ "`, a default-layout (`.auto`) struct.\n" ++
            "  A GPU wire type must have a layout both compilations agree on; `.auto` fixes\n" ++
            "  neither field order nor padding, and the host and the device are compiled\n" ++
            "  separately for different targets.\n" ++
            "  Declare it as `extern struct` (C layout) or `packed struct` (bit layout).",
    );
}

fn hasFixedLayout(comptime U: type) bool {
    return switch (@typeInfo(U)) {
        .@"struct" => |s| s.layout != .auto,
        .array => |a| hasFixedLayout(a.child),
        else => true,
    };
}

pub fn assertStable(comptime T: type) void {
    _ = Boundary(T);
}

pub fn pack(comptime T: type, value: T) Boundary(T) {
    return switch (@typeInfo(T)) {
        .bool => @intFromBool(value),
        .int, .float => value,
        .@"enum" => @intFromEnum(value),
        // A bool vector is bit-packed -- @Vector(4, bool) is 4 bits, its wire
        // form @Vector(4, u8) is 32 -- so it is the one vector a @bitCast
        // cannot carry. Both builtins are elementwise on vectors.
        .vector => |v| if (v.child == bool) @intCast(@intFromBool(value)) else @bitCast(value),
        .array => |a| blk: {
            var out: Boundary(T) = undefined;
            inline for (0..a.len) |i| out[i] = pack(a.child, value[i]);
            break :blk out;
        },
        .@"struct" => |s| blk: {
            if (@hasDecl(T, "toGpu")) break :blk T.toGpu(value);
            var out: Boundary(T) = undefined;
            inline for (s.fields) |field|
                @field(out, field.name) = pack(field.type, @field(value, field.name));
            break :blk out;
        },
        else => unreachable,
    };
}

pub fn unpack(comptime T: type, value: Boundary(T)) T {
    return switch (@typeInfo(T)) {
        .bool => value != 0,
        .int, .float => value,
        .@"enum" => @enumFromInt(value),
        .vector => |v| if (v.child == bool)
            value != @as(Boundary(T), @splat(0))
        else
            @bitCast(value),
        .array => |a| blk: {
            var out: T = undefined;
            inline for (0..a.len) |i| out[i] = unpack(a.child, value[i]);
            break :blk out;
        },
        .@"struct" => |s| blk: {
            if (@hasDecl(T, "fromGpu")) break :blk T.fromGpu(value);
            var out: T = undefined;
            inline for (s.fields) |field|
                @field(out, field.name) = unpack(field.type, @field(value, field.name));
            break :blk out;
        },
        else => unreachable,
    };
}

// The tests below lock in what an on-hardware audit confirmed already works on
// NVPTX. They are regressions locks for the derivation, not open questions.

test fixedWidthInt {
    // The rejections themselves are `@compileError`s and cannot be tested, so
    // the rule they consult is tested instead -- the same split as
    // `hasFixedLayout`, and the only coverage `spec.zig`'s index and value
    // checks have.
    inline for ([_]type{ u8, i8, u16, i16, u32, i32, u64, i64 }) |T|
        try std.testing.expect(fixedWidthInt(T));
    inline for ([_]type{ u1, u7, i24, u128, u0, usize, isize, f32, bool }) |T|
        try std.testing.expect(!fixedWidthInt(T));

    // usize/isize are 64-bit on this host and must still be rejected: the width
    // is the target's, not the type's.
    try std.testing.expectEqual(@as(u16, 64), @typeInfo(usize).int.bits);

    inline for ([_]type{ f16, f32, f64 }) |T|
        try std.testing.expect(fixedWidthFloat(T));
    inline for ([_]type{ f80, f128, u32, bool }) |T|
        try std.testing.expect(!fixedWidthFloat(T));
}

test "enums cross as their tag type, signed and negative included" {
    const Mode = enum(u8) { off, on };
    const Signed = enum(i16) { back = -3, forward = 4 };

    try std.testing.expectEqual(u8, Boundary(Mode));
    try std.testing.expectEqual(i16, Boundary(Signed));
    try std.testing.expectEqual(@as(u8, 1), pack(Mode, .on));
    try std.testing.expectEqual(@as(i16, -3), pack(Signed, .back));
    try std.testing.expectEqual(Mode.on, unpack(Mode, pack(Mode, .on)));
    try std.testing.expectEqual(Signed.back, unpack(Signed, pack(Signed, .back)));
}

test "arrays and vectors round-trip element by element" {
    const Flags = [3]bool;
    try std.testing.expectEqual([3]u8, Boundary(Flags));
    const flags: Flags = .{ true, false, true };
    try std.testing.expectEqual([3]u8{ 1, 0, 1 }, pack(Flags, flags));
    try std.testing.expectEqual(flags, unpack(Flags, pack(Flags, flags)));

    const V4 = @Vector(4, f32);
    try std.testing.expectEqual(V4, Boundary(V4));
    const v: V4 = .{ 1, 2, 3, 4 };
    try std.testing.expectEqual(v, unpack(V4, pack(V4, v)));

    // Enum arrays reduce to tag arrays, not to a derived struct.
    const Mode = enum(u32) { a, b };
    try std.testing.expectEqual([2]u32, Boundary([2]Mode));
    try std.testing.expectEqual([2]u32{ 1, 0 }, pack([2]Mode, .{ .b, .a }));

    // A bool vector is bit-packed on the host: @Vector(4, bool) is 4 bits and
    // the wire form is 32, so this used to fail inside pack() with a raw
    // @bitCast size mismatch pointing into this file.
    const Mask = @Vector(4, bool);
    try std.testing.expectEqual(@Vector(4, u8), Boundary(Mask));
    const mask: Mask = .{ true, false, true, false };
    try std.testing.expectEqual(@Vector(4, u8){ 1, 0, 1, 0 }, pack(Mask, mask));
    try std.testing.expectEqual(mask, unpack(Mask, pack(Mask, mask)));
}

test "nested structs, f16 and a zero-field params struct" {
    const Inner = struct { half: f16, mode: enum(i8) { lo = -1, hi = 1 } };
    const Outer = struct { inner: Inner, samples: [2]u16, weights: @Vector(2, f32) };

    const B = Boundary(Outer);
    try std.testing.expect(@typeInfo(B).@"struct".layout == .@"extern");
    try std.testing.expect(@typeInfo(@FieldType(B, "inner")).@"struct".layout == .@"extern");

    const value: Outer = .{
        .inner = .{ .half = 0.5, .mode = .lo },
        .samples = .{ 7, 9 },
        .weights = .{ 1.5, 2.5 },
    };
    const wire = pack(Outer, value);
    try std.testing.expectEqual(@as(f16, 0.5), wire.inner.half);
    try std.testing.expectEqual(@as(i8, -1), wire.inner.mode);

    const back = unpack(Outer, wire);
    try std.testing.expectEqual(value.inner.mode, back.inner.mode);
    try std.testing.expectEqual(value.samples, back.samples);
    try std.testing.expectEqual(value.weights, back.weights);

    // A kernel that takes no parameters still has to pack something.
    const Empty = struct {};
    try std.testing.expectEqual(@as(usize, 0), @typeInfo(Boundary(Empty)).@"struct".fields.len);
    _ = unpack(Empty, pack(Empty, .{}));
}

test "a custom gpu_layout must have a layout both compilations agree on" {
    // `.auto` fixes neither field order nor padding, and it is the one type the
    // derivation skips, so it is the one that has to be checked by hand.
    try std.testing.expect(!hasFixedLayout(struct { lo: u32, hi: u32 }));
    try std.testing.expect(!hasFixedLayout([2]struct { lo: u32 }));
    try std.testing.expect(hasFixedLayout(extern struct { lo: u32, hi: u32 }));
    try std.testing.expect(hasFixedLayout(packed struct { lo: u32, hi: u32 }));
    try std.testing.expect(hasFixedLayout([2]extern struct { lo: u32 }));
    try std.testing.expect(hasFixedLayout(u32));

    // A packed wire type is accepted end to end.
    const Bits = struct {
        a: bool,
        b: bool,

        pub const gpu_layout = packed struct(u32) { a: u1, b: u1, _pad: u30 = 0 };
        pub fn toGpu(v: @This()) gpu_layout {
            return .{ .a = @intFromBool(v.a), .b = @intFromBool(v.b) };
        }
        pub fn fromGpu(w: gpu_layout) @This() {
            return .{ .a = w.a == 1, .b = w.b == 1 };
        }
    };
    try std.testing.expectEqual(@as(usize, 4), @sizeOf(Boundary(Bits)));
    const back = unpack(Bits, pack(Bits, .{ .a = false, .b = true }));
    try std.testing.expect(!back.a and back.b);
}

test "gpu_layout/toGpu/fromGpu round-trips a host-only type" {
    // The documented escape hatch: a host value that cannot cross the boundary
    // on its own (f64 range, an index) decomposed into a stable wire form.
    const Handle = struct {
        id: u64,
        gain: f64,

        pub const gpu_layout = extern struct { id_lo: u32, id_hi: u32, gain: f32 };

        pub fn toGpu(value: @This()) gpu_layout {
            return .{
                .id_lo = @truncate(value.id),
                .id_hi = @truncate(value.id >> 32),
                .gain = @floatCast(value.gain),
            };
        }
        pub fn fromGpu(wire: gpu_layout) @This() {
            return .{
                .id = @as(u64, wire.id_hi) << 32 | wire.id_lo,
                .gain = wire.gain,
            };
        }
    };

    const B = Boundary(Handle);
    try std.testing.expect(@typeInfo(B).@"struct".layout == .@"extern");

    const original: Handle = .{ .id = 0xDEAD_BEEF_CAFE, .gain = 0.5 };
    const wire = pack(Handle, original);
    try std.testing.expectEqual(@as(u32, 0xBEEF_CAFE), wire.id_lo);
    try std.testing.expectEqual(@as(u32, 0xDEAD), wire.id_hi);

    const back = unpack(Handle, wire);
    try std.testing.expectEqual(original.id, back.id);
    try std.testing.expectEqual(@as(f64, 0.5), back.gain);
}

test "custom boundary nests inside a params struct" {
    const Inner = struct {
        flag: bool,

        pub const gpu_layout = extern struct { flag: u32 };
        pub fn toGpu(v: @This()) gpu_layout {
            return .{ .flag = @intFromBool(v.flag) };
        }
        pub fn fromGpu(w: gpu_layout) @This() {
            return .{ .flag = w.flag != 0 };
        }
    };
    const Params = struct { scale: f32, inner: Inner };

    const value: Params = .{ .scale = 2, .inner = .{ .flag = true } };
    const wire = pack(Params, value);
    try std.testing.expectEqual(@as(u32, 1), wire.inner.flag);
    try std.testing.expect(unpack(Params, wire).inner.flag);
}

test "plain structs become extern boundary structs" {
    const P = struct {
        scale: f32,
        passes: u32,
        enabled: bool,
    };
    const B = Boundary(P);
    try std.testing.expect(@typeInfo(B).@"struct".layout == .@"extern");
    const packed_value = pack(P, .{ .scale = 2, .passes = 3, .enabled = true });
    try std.testing.expectEqual(@as(f32, 2), packed_value.scale);
    try std.testing.expectEqual(@as(u8, 1), packed_value.enabled);
    const roundtrip = unpack(P, packed_value);
    try std.testing.expect(roundtrip.enabled);
}
