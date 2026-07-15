//! A map specification is metadata plus one pure scalar function.

const std = @import("std");
const abi = @import("abi.zig");

pub const MapOptions = struct {
    block_size: u32 = 256,
};

pub fn Map(
    comptime name: [:0]const u8,
    comptime T: type,
    comptime Params: type,
    comptime func: anytype,
    comptime options: MapOptions,
) type {
    validateName(name);
    validateValue(T);
    validateFunction(T, Params, func);
    _ = abi.Boundary(Params);
    if (options.block_size == 0 or options.block_size > 1024)
        @compileError("block_size must be in 1...1024");

    return struct {
        pub const entry_name: [:0]const u8 = name;
        pub const Value = T;
        pub const Parameters = Params;
        pub const BoundaryParameters = abi.Boundary(Params);
        pub const block_size: u32 = options.block_size;

        pub inline fn eval(x: T, params: Params) T {
            return @call(.always_inline, func, .{ x, params });
        }
    };
}

fn validateName(comptime name: []const u8) void {
    if (name.len == 0) @compileError("kernel entry name cannot be empty");
    for (name, 0..) |c, i| {
        const valid = switch (c) {
            'a'...'z', 'A'...'Z', '_' => true,
            '0'...'9' => i != 0,
            else => false,
        };
        if (!valid) @compileError("kernel names must be C identifiers: " ++ name);
    }
}

fn validateValue(comptime T: type) void {
    switch (@typeInfo(T)) {
        .int => |i| {
            if (T == usize or T == isize or (i.bits != 8 and i.bits != 16 and i.bits != 32 and i.bits != 64))
                @compileError("map value must be a fixed-width 8/16/32/64-bit integer");
        },
        .float => |f| if (f.bits != 16 and f.bits != 32 and f.bits != 64)
            @compileError("map value must be f16, f32, or f64"),
        .vector => |v| validateValue(v.child),
        else => @compileError("map value must be an integer, float, or vector: " ++ @typeName(T)),
    }
}

fn validateFunction(comptime T: type, comptime P: type, comptime func: anytype) void {
    const info = @typeInfo(@TypeOf(func));
    if (info != .@"fn") @compileError("map operation must be a function");
    const f = info.@"fn";
    if (f.is_var_args or f.params.len != 2)
        @compileError("map operation must have signature fn (T, Params) T");
    const a = f.params[0].type orelse @compileError("generic function parameters are not supported");
    const b = f.params[1].type orelse @compileError("generic function parameters are not supported");
    const r = f.return_type orelse @compileError("map operation must return T");
    if (a != T or b != P or r != T)
        @compileError("map operation must have signature fn (" ++ @typeName(T) ++ ", " ++ @typeName(P) ++ ") " ++ @typeName(T));
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
}
