//! A map specification is metadata plus one pure scalar function.

const std = @import("std");
const abi = @import("abi.zig");

pub const MapOptions = struct {
    block_size: u32 = 256,
};

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
    if (options.block_size == 0 or options.block_size > 1024)
        @compileError("block_size must be in 1...1024");

    return struct {
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
}
