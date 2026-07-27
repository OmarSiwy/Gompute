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

    const name_end = identEnd(line[1..]);
    if (name_end == 0) return null;
    const name = line[1 .. 1 + name_end];

    // Anchor on the header, not on " alias " anywhere in the line: a string
    // constant whose *contents* spell " alias " would otherwise be parsed as an
    // alias and silently deleted from the module. Everything that may precede
    // the kind keyword (linkage, visibility, unnamed_addr, addrspace(N), ...)
    // is a single space-free word, so the first word that names a kind decides.
    if (!std.mem.startsWith(u8, line[1 + name_end ..], " = ")) return null;
    var words = std.mem.tokenizeScalar(u8, line[1 + name_end + " = ".len ..], ' ');
    while (words.next()) |word| {
        if (std.mem.eql(u8, word, "alias")) break;
        // ponytail: the kinds Zig's backends actually emit. A new kind just
        // means the scan runs off the end and returns null, which is the safe
        // direction -- the global stays in the module.
        for ([_][]const u8{ "global", "constant", "ifunc" }) |kind| {
            if (std.mem.eql(u8, word, kind)) return null;
        }
    } else return null;

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

/// Drop the alias lines and rename each aliased definition to its public name.
fn rewriteIr(arena: std.mem.Allocator, input: []const u8, aliases: []const Alias) ![]const u8 {
    var rewritten: std.ArrayList(u8) = .empty;
    try rewritten.ensureTotalCapacity(arena, input.len);
    var first = true;
    var it = std.mem.splitScalar(u8, input, '\n');
    while (it.next()) |line| {
        if (parseAlias(line) != null) continue;
        if (!first) try rewritten.append(arena, '\n');
        first = false;

        if (std.mem.startsWith(u8, line, "define ")) rewrite: {
            for (aliases) |alias| {
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

    // When multiple aliases target the same definition (LLVM merges identical
    // kernel bodies), only the first got renamed. Clone the full function body
    // for each additional alias.
    var renamed = std.StringHashMap([]const u8).init(arena);
    for (aliases) |alias| {
        const gop = try renamed.getOrPut(alias.aliasee);
        if (!gop.found_existing) {
            gop.value_ptr.* = alias.name;
            continue;
        }
        const primary = gop.value_ptr.*;
        const needle = try std.fmt.allocPrint(arena, " @{s}(", .{primary});
        // Take the occurrence that is on a "define " line; a call site would
        // otherwise walk back to the *previous* function and clone that.
        var scan: usize = 0;
        var fn_start: usize = undefined;
        const define_start = while (std.mem.indexOfPos(u8, rewritten.items, scan, needle)) |pos| {
            const line_start = if (std.mem.lastIndexOfScalar(u8, rewritten.items[0..pos], '\n')) |nl|
                nl + 1
            else
                0;
            if (std.mem.startsWith(u8, rewritten.items[line_start..], "define ")) {
                fn_start = pos;
                break line_start;
            }
            scan = pos + needle.len;
        } else continue;

        // A function only ever closes with a line that is exactly "}". Matching
        // any "}" followed by a newline truncated the body at ordinary lines
        // such as `%s = alloca { i32, i32 }`, emitting IR with no terminator.
        var end = fn_start + needle.len;
        while (end < rewritten.items.len) : (end += 1) {
            if (rewritten.items[end] != '}') continue;
            if (end == 0 or rewritten.items[end - 1] != '\n') continue;
            if (end + 1 == rewritten.items.len or rewritten.items[end + 1] == '\n') break;
        }
        if (end >= rewritten.items.len) continue;

        const fn_body = rewritten.items[define_start .. end + 1];
        const old_name = try std.fmt.allocPrint(arena, "@{s}(", .{primary});
        const new_name = try std.fmt.allocPrint(arena, "@{s}(", .{alias.name});
        const cloned = try std.mem.replaceOwned(u8, arena, fn_body, old_name, new_name);
        try rewritten.append(arena, '\n');
        try rewritten.appendSlice(arena, cloned);
    }

    return rewritten.items;
}

/// Public entry name -> symbol as it appears in the *unrewritten* object, which
/// is what HIP loads. CUDA loads the rewritten module, where the entry is the
/// public name itself, so a CUDA caller discards the result and keeps the name
/// it asked for -- it calls `resolve` only for the missing-kernel compile error.
fn emitMap(arena: std.mem.Allocator, aliases: []const Alias) ![]const u8 {
    var map_source: std.ArrayList(u8) = .empty;
    try map_source.appendSlice(arena,
        \\//! Generated from LLVM aliases. Do not edit.
        \\//!
        \\//! `resolve` returns the symbol name in the HIP artifact. In the CUDA
        \\//! artifact the entry is the exported name itself; call it there only
        \\//! to turn a missing kernel into a compile error.
        \\const std = @import("std");
        \\
        \\pub fn resolve(comptime exported: []const u8) [:0]const u8 {
        \\
    );
    for (aliases) |alias| {
        try map_source.appendSlice(arena, "    if (comptime std.mem.eql(u8, exported, ");
        try appendZigString(&map_source, arena, alias.name);
        try map_source.appendSlice(arena, ")) return ");
        try appendZigString(&map_source, arena, try decodeLlvmName(arena, alias.aliasee));
        try map_source.appendSlice(arena, ";\n");
    }
    try map_source.appendSlice(arena,
        \\    @compileError("kernel \"" ++ exported ++ "\" is not in the GPU artifact; " ++
        \\        "export it from your `.kernels_root` file with `comptime { g.exportKernels(@This()); }`");
        \\}
        \\
    );
    return map_source.items;
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

    const rewritten = try rewriteIr(arena, input, aliases.items);
    const map_source = try emitMap(arena, aliases.items);

    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = args[2], .data = rewritten });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = args[3], .data = map_source });
}

test "parse Zig-style alias" {
    const a = parseAlias("@add = alias void (ptr), ptr @kernels.add").?;
    try std.testing.expectEqualStrings("add", a.name);
    try std.testing.expectEqualStrings("kernels.add", a.aliasee);
}

test "a global whose contents say ' alias ' is not an alias" {
    try std.testing.expect(parseAlias(
        \\@.str = private unnamed_addr constant [16 x i8] c"x alias y\00", align 1
    ) == null);
    try std.testing.expect(parseAlias("@g = external global i32") == null);
    try std.testing.expect(parseAlias("@r = internal unnamed_addr alias i32, ptr @g") != null);
}

/// Everything below drives the two rewrite passes through an arena, the way
/// main() does.
fn testRewrite(arena: std.mem.Allocator, input: []const u8) ![]const u8 {
    var aliases: std.ArrayList(Alias) = .empty;
    var it = std.mem.splitScalar(u8, input, '\n');
    while (it.next()) |line| {
        if (parseAlias(line)) |alias| try aliases.append(arena, alias);
    }
    return rewriteIr(arena, input, aliases.items);
}

test "string constants survive the rewrite and stay out of the name map" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const input =
        \\@.str = private unnamed_addr constant [16 x i8] c"x alias y\00", align 1
        \\@add = alias void (ptr), ptr @kernels.add
        \\
        \\define private ptx_kernel void @kernels.add(ptr %0) {
        \\  ret void
        \\}
        \\
    ;
    const out = try testRewrite(arena, input);
    try std.testing.expect(std.mem.indexOf(u8, out, "@.str = private") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "@add = alias") == null);
    try std.testing.expect(std.mem.indexOf(u8, out, "define ptx_kernel void @add(ptr %0) {") != null);

    const map = try emitMap(arena, &.{.{ .name = "add", .aliasee = "kernels.add" }});
    try std.testing.expect(std.mem.indexOf(u8, map, ".str") == null);
    try std.testing.expect(std.mem.indexOf(u8, map, "\"add\")) return \"kernels.add\";") != null);
}

test "cloning a shared definition copies the whole body" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // The alloca's trailing "}" used to be mistaken for the end of the function.
    const input =
        \\@one = alias void (ptr), ptr @impl
        \\@two = alias void (ptr), ptr @impl
        \\
        \\define private ptx_kernel void @impl(ptr %0) {
        \\entry:
        \\  %s = alloca { i32, i32 }
        \\  store i32 7, ptr %s
        \\  ret void
        \\}
        \\
    ;
    const out = try testRewrite(arena, input);
    try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, out, "define ptx_kernel void @"));
    try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, out, "store i32 7, ptr %s"));
    try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, out, "  ret void\n}"));
    try std.testing.expect(std.mem.indexOf(u8, out, "@one(ptr %0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "@two(ptr %0)") != null);
    try std.testing.expect(std.mem.endsWith(u8, std.mem.trimEnd(u8, out, "\n"), "}"));
}

test "a call site does not make the clone pick the previous function" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const input =
        \\@one = alias void (ptr), ptr @impl
        \\@two = alias void (ptr), ptr @impl
        \\
        \\define internal void @caller(ptr %0) {
        \\  call void @impl(ptr %0)
        \\  ret void
        \\}
        \\
        \\define private ptx_kernel void @impl(ptr %0) {
        \\  ret void
        \\}
        \\
    ;
    const out = try testRewrite(arena, input);
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, out, "define internal void @caller"));
    try std.testing.expect(std.mem.indexOf(u8, out, "define ptx_kernel void @two(ptr %0) {\n  ret void\n}") != null);
}

test "hex escapes decode, including one at the very end of the name" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    try std.testing.expectEqualStrings("a.b", try decodeLlvmName(arena, "\"a\\2Eb\""));
    try std.testing.expectEqualStrings("kernels.a\"", try decodeLlvmName(arena, "\"kernels.a\\22\""));
    // A truncated escape is not one; it stays literal.
    try std.testing.expectEqualStrings("a\\2", try decodeLlvmName(arena, "a\\2"));
}
