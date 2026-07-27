//! Fix Zig 0.16 GPU kernel aliases and emit a host-side alias map.
//!
//! Usage: kernel_ir_tool <input.ll> <rewritten.ll> <aliases.zig>

const std = @import("std");

const Alias = struct {
    name: []const u8,
    aliasee: []const u8,
};

fn identEnd(s: []const u8) usize {
    for (s, 0..) |c, i| switch (c) {
        'a'...'z', 'A'...'Z', '0'...'9', '_', '.', '$', '-' => {},
        else => return i,
    };
    return s.len;
}

fn parseAlias(line: []const u8) ?Alias {
    if (!std.mem.startsWith(u8, line, "@")) return null;
    if (std.mem.indexOf(u8, line, " alias ") == null) return null;

    const name_end = identEnd(line[1..]);
    if (name_end == 0) return null;
    const name = line[1 .. 1 + name_end];

    const at = std.mem.lastIndexOfScalar(u8, line, '@') orelse return null;
    const rest = line[at + 1 ..];
    const aliasee = if (rest.len > 0 and rest[0] == '"') blk: {
        const close = std.mem.indexOfScalarPos(u8, rest, 1, '"') orelse return null;
        break :blk rest[0 .. close + 1];
    } else rest[0..identEnd(rest)];

    if (aliasee.len == 0) return null;
    return .{ .name = name, .aliasee = aliasee };
}

fn decodeLlvmName(arena: std.mem.Allocator, llvm_name: []const u8) ![]const u8 {
    const raw = if (llvm_name.len >= 2 and llvm_name[0] == '"' and llvm_name[llvm_name.len - 1] == '"')
        llvm_name[1 .. llvm_name.len - 1]
    else
        llvm_name;

    var out: std.ArrayList(u8) = .empty;
    try out.ensureTotalCapacity(arena, raw.len);
    var i: usize = 0;
    while (i < raw.len) {
        if (raw[i] == '\\' and i + 2 < raw.len) {
            const hi = std.fmt.charToDigit(raw[i + 1], 16) catch null;
            const lo = std.fmt.charToDigit(raw[i + 2], 16) catch null;
            if (hi != null and lo != null) {
                try out.append(arena, @intCast(hi.? * 16 + lo.?));
                i += 3;
                continue;
            }
        }
        try out.append(arena, raw[i]);
        i += 1;
    }
    return out.items;
}

fn appendZigString(out: *std.ArrayList(u8), arena: std.mem.Allocator, text: []const u8) !void {
    try out.append(arena, '"');
    for (text) |c| {
        switch (c) {
            '"' => try out.appendSlice(arena, "\\\""),
            '\\' => try out.appendSlice(arena, "\\\\"),
            '\n' => try out.appendSlice(arena, "\\n"),
            '\r' => try out.appendSlice(arena, "\\r"),
            '\t' => try out.appendSlice(arena, "\\t"),
            else => if (c >= 0x20 and c <= 0x7e) {
                try out.append(arena, c);
            } else {
                const escaped = try std.fmt.allocPrint(arena, "\\x{x:0>2}", .{c});
                try out.appendSlice(arena, escaped);
            },
        }
    }
    try out.append(arena, '"');
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(arena);
    if (args.len != 4) {
        std.debug.print("usage: kernel_ir_tool <input.ll> <rewritten.ll> <aliases.zig>\n", .{});
        return error.BadUsage;
    }

    const input = try std.Io.Dir.cwd().readFileAlloc(
        io,
        args[1],
        arena,
        .limited(512 * 1024 * 1024),
    );

    var aliases: std.ArrayList(Alias) = .empty;
    var it = std.mem.splitScalar(u8, input, '\n');
    while (it.next()) |line| {
        if (parseAlias(line)) |alias| try aliases.append(arena, alias);
    }
    // "alias" is an internal detail of the Zig 0.16 NVPTX workaround. A user who
    // forgot exportKernels has never heard the word, so say what to do instead.
    if (aliases.items.len == 0) {
        std.debug.print(
            \\gompute: the device compilation of your kernels file produced no GPU entry points.
            \\
            \\Every kernel launched with Kernel(spec, .cuda) or Kernel(spec, .hip) must be
            \\exported from the kernels file named by `.kernels_root` in your build.zig:
            \\
            \\    comptime {{ g.exportKernels(@This()); }}          // all kernels in this file
            \\    comptime {{ g.exportKernels(.{{ my_kernel }}); }}   // or an explicit list
            \\
            \\Check that the file exporting them is the same file `.kernels_root` points at.
            \\
        , .{});
        return error.NoKernelAliasesFound;
    }

    var rewritten: std.ArrayList(u8) = .empty;
    try rewritten.ensureTotalCapacity(arena, input.len);
    var first = true;
    it = std.mem.splitScalar(u8, input, '\n');
    while (it.next()) |line| {
        if (parseAlias(line) != null) continue;
        if (!first) try rewritten.append(arena, '\n');
        first = false;

        if (std.mem.startsWith(u8, line, "define ")) rewrite: {
            for (aliases.items) |alias| {
                const marker = try std.fmt.allocPrint(arena, "@{s}(", .{alias.aliasee});
                const pos = std.mem.indexOf(u8, line, marker) orelse continue;
                const head = line[0..pos];
                var body = head["define ".len..];
                for ([_][]const u8{ "internal ", "private " }) |linkage| {
                    if (std.mem.startsWith(u8, body, linkage)) {
                        body = body[linkage.len..];
                        break;
                    }
                }
                try rewritten.appendSlice(arena, "define ");
                try rewritten.appendSlice(arena, body);
                try rewritten.append(arena, '@');
                try rewritten.appendSlice(arena, alias.name);
                try rewritten.appendSlice(arena, line[pos + marker.len - 1 ..]);
                break :rewrite;
            }
            try rewritten.appendSlice(arena, line);
        } else {
            try rewritten.appendSlice(arena, line);
        }
    }

    // When multiple aliases target the same definition, only the first got
    // renamed. Clone the full function body for each additional alias.
    {
        var renamed = std.StringHashMap([]const u8).init(arena);
        for (aliases.items) |alias| {
            const gop = try renamed.getOrPut(alias.aliasee);
            if (!gop.found_existing) {
                gop.value_ptr.* = alias.name;
            } else {
                const primary = gop.value_ptr.*;
                const needle = try std.fmt.allocPrint(arena, " @{s}(", .{primary});
                const fn_start = std.mem.indexOf(u8, rewritten.items, needle) orelse continue;
                // Walk back to "define"
                const define_start = std.mem.lastIndexOf(u8, rewritten.items[0..fn_start], "define ") orelse continue;
                // Find closing "}" — functions end with "\n}\n" or "}\n"
                const search_from = fn_start + needle.len;
                var end = search_from;
                while (end < rewritten.items.len) : (end += 1) {
                    if (rewritten.items[end] == '}' and
                        (end + 1 >= rewritten.items.len or rewritten.items[end + 1] == '\n'))
                    {
                        break;
                    }
                }
                if (end >= rewritten.items.len) continue;
                const fn_body = rewritten.items[define_start .. end + 1];
                // Clone with the new name
                const old_name = try std.fmt.allocPrint(arena, "@{s}(", .{primary});
                const new_name = try std.fmt.allocPrint(arena, "@{s}(", .{alias.name});
                const cloned = try std.mem.replaceOwned(u8, arena, fn_body, old_name, new_name);
                try rewritten.append(arena, '\n');
                try rewritten.appendSlice(arena, cloned);
            }
        }
    }

    var map_source: std.ArrayList(u8) = .empty;
    try map_source.appendSlice(arena,
        \\//! Generated from LLVM aliases. Do not edit.
        \\const std = @import("std");
        \\
        \\pub fn resolve(comptime exported: []const u8) [:0]const u8 {
        \\
    );
    for (aliases.items) |alias| {
        try map_source.appendSlice(arena, "    if (comptime std.mem.eql(u8, exported, ");
        try appendZigString(&map_source, arena, alias.name);
        try map_source.appendSlice(arena, ")) return ");
        try appendZigString(&map_source, arena, try decodeLlvmName(arena, alias.aliasee));
        try map_source.appendSlice(arena, ";\n");
    }
    try map_source.appendSlice(arena,
        \\    @compileError("kernel was not exported into the HIP artifact: " ++ exported);
        \\}
        \\
    );

    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = args[2], .data = rewritten.items });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = args[3], .data = map_source.items });
}

test "parse Zig-style alias" {
    const a = parseAlias("@add = alias void (ptr), ptr @kernels.add").?;
    try std.testing.expectEqualStrings("add", a.name);
    try std.testing.expectEqualStrings("kernels.add", a.aliasee);
}
