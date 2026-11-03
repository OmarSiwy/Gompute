//! Fail if a file holds any of the given patterns, or with `--present`, if it
//! holds none. `addCheckFile` can only demand a match; this is the inverse,
//! for `tests/no_contract.zig`.
//!
//! Usage: absent [--present] <file> <pattern>...

const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const present = std.mem.eql(u8, args[1], "--present");
    const at: usize = @intFromBool(present);
    const path = args[1 + at];
    const patterns = args[2 + at ..];
    const text = try std.Io.Dir.cwd().readFileAlloc(init.io, path, init.arena.allocator(), .limited(256 << 20));

    var found = false;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| for (patterns) |p| {
        if (std.mem.indexOf(u8, line, p) == null) continue;
        found = true;
        if (!present) std.debug.print("{s}: '{s}' in: {s}\n", .{ path, p, std.mem.trim(u8, line, " \t") });
    };
    if (found != present) {
        if (present) std.debug.print("{s}: none of {f} found; the check is blind on this target\n", .{ path, std.json.fmt(patterns, .{}) });
        std.process.exit(1);
    }
}
