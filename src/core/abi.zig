//! Compile-time host/device boundary derivation.

const std = @import("std");

pub fn Boundary(comptime T: type) type {
    return switch (@typeInfo(T)) {
        .bool => u8,
        .int => |i| blk: {
            if (T == usize or T == isize)
                @compileError("usize/isize cannot cross a GPU boundary; use u32/u64 or i32/i64");
            if (i.bits != 8 and i.bits != 16 and i.bits != 32 and i.bits != 64)
                @compileError("GPU boundary integers must be 8, 16, 32, or 64 bits: " ++ @typeName(T));
            break :blk T;
        },
        .float => |f| blk: {
            if (f.bits != 16 and f.bits != 32 and f.bits != 64)
                @compileError("GPU boundary floats must be f16, f32, or f64: " ++ @typeName(T));
            break :blk T;
        },
        .@"enum" => |e| Boundary(e.tag_type),
        .array => |a| [a.len]Boundary(a.child),
        .vector => |v| @Vector(v.len, Boundary(v.child)),
        .@"struct" => |s| boundaryStruct(T, s),
        else => @compileError(@typeName(T) ++ " has no stable generated GPU ABI"),
    };
}

fn boundaryStruct(comptime T: type, comptime info: std.builtin.Type.Struct) type {
    if (@hasDecl(T, "gpu_layout")) {
        const U = T.gpu_layout;
        assertStable(U);
        return U;
    }

    comptime var names: [info.fields.len][]const u8 = undefined;
    comptime var types: [info.fields.len]type = undefined;
    comptime var attrs: [info.fields.len]std.builtin.Type.StructField.Attributes = undefined;
    inline for (info.fields, 0..) |field, i| {
        if (field.is_comptime)
            @compileError("comptime field cannot cross GPU boundary: " ++ field.name);
        names[i] = field.name;
        types[i] = Boundary(field.type);
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
