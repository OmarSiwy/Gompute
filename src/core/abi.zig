//! Compile-time host/device boundary derivation.

const std = @import("std");

pub fn Boundary(comptime T: type) type {
    return BoundaryAt(T, @typeName(T));
}

/// `at` is the dotted path from the root parameter type down to whatever we are
/// currently deriving, so a rejection names the field the user actually wrote
/// rather than the innermost type the recursion happened to reach.
fn BoundaryAt(comptime T: type, comptime at: []const u8) type {
    return switch (@typeInfo(T)) {
        .bool => u8,
        .int => |i| blk: {
            if (T == usize or T == isize) @compileError(reject(at, T, "host and device pointer widths may differ, so usize/isize have no fixed size across the boundary; use u32/u64 or i32/i64"));
            if (i.bits != 8 and i.bits != 16 and i.bits != 32 and i.bits != 64)
                @compileError(reject(at, T, "integers must be exactly 8, 16, 32, or 64 bits wide"));
            break :blk T;
        },
        .float => |f| blk: {
            if (f.bits != 16 and f.bits != 32 and f.bits != 64)
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
        _ = BoundaryAt(U, @typeName(T) ++ ".gpu_layout");
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

pub fn assertStable(comptime T: type) void {
    _ = Boundary(T);
}

pub fn pack(comptime T: type, value: T) Boundary(T) {
    return switch (@typeInfo(T)) {
        .bool => @intFromBool(value),
        .int, .float => value,
        .@"enum" => @intFromEnum(value),
        .vector => @bitCast(value),
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
        .vector => @bitCast(value),
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
