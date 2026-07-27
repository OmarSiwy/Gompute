//! Compile-time operation fusion. Each operation type exposes:
//! `pub inline fn eval(x: T, params: Params) T`.

pub fn Fused(comptime T: type, comptime Params: type, comptime ops: anytype) type {
    if (ops.len == 0) @compileError("a fused pipeline needs at least one operation");
    return struct {
        pub inline fn eval(value: T, params: Params) T {
            var x = value;
            inline for (ops) |Op| x = @call(.always_inline, Op.eval, .{ x, params });
            return x;
        }
    };
}

pub fn Unary(comptime T: type, comptime Params: type, comptime func: anytype) type {
    return struct {
        pub inline fn eval(x: T, params: Params) T {
            return @call(.always_inline, func, .{ x, params });
        }
    };
}

test "fused operations are straight-line comptime composition" {
    const P = struct { gain: f32 };
    const Scale = struct {
        pub inline fn eval(x: f32, p: P) f32 {
            return x * p.gain;
        }
    };
    const Square = struct {
        pub inline fn eval(x: f32, _: P) f32 {
            return x * x;
        }
    };
    const Pipe = Fused(f32, P, .{ Scale, Square });
    const std = @import("std");
    try std.testing.expectEqual(@as(f32, 36), Pipe.eval(3, .{ .gain = 2 }));

    // One op is the identity case: the pipeline is exactly that op.
    try std.testing.expectEqual(@as(f32, 6), Fused(f32, P, .{Scale}).eval(3, .{ .gain = 2 }));

    // Order matters, and it is left to right.
    const Offset = struct {
        pub inline fn eval(x: f32, _: P) f32 {
            return x + 1;
        }
    };
    const Long = Fused(f32, P, .{ Scale, Offset, Square, Offset });
    try std.testing.expectEqual(@as(f32, 50), Long.eval(3, .{ .gain = 2 })); // 6, 7, 49, 50
    try std.testing.expectEqual(@as(f32, 65), Fused(f32, P, .{ Offset, Scale, Square, Offset }).eval(3, .{ .gain = 2 })); // 4, 8, 64, 65

    // Fused nests: it is an op like any other.
    try std.testing.expectEqual(@as(f32, 1296), Fused(f32, P, .{ Pipe, Square }).eval(3, .{ .gain = 2 }));

    // Unary lifts a plain function into the same shape.
    const Half = Unary(f32, P, struct {
        fn call(x: f32, _: P) f32 {
            return x / 2;
        }
    }.call);
    try std.testing.expectEqual(@as(f32, 18), Fused(f32, P, .{ Pipe, Half }).eval(3, .{ .gain = 2 }));
}
