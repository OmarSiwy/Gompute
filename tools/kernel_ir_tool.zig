//! List the GPU entry points in a kernel root's LLVM IR as a Zig name table.
//!
//! Usage: kernel_ir_tool <input.ll> <names.zig>
//!
//! Zig 0.16 exported a kernel as an LLVM alias of a mangled definition, which
//! NVPTX rejects; this tool used to rewrite the IR around that. Since 0.17 the
//! definition carries the exported name itself, so all that is left is reading
//! the names off: `emitKernels` needs them at comptime to map a kernel name to
//! the blob that holds it.

const std = @import("std");

fn identEnd(s: []const u8) usize {
    for (s, 0..) |c, i| switch (c) {
        'a'...'z', 'A'...'Z', '0'...'9', '_', '.', '$', '-' => {},
        else => return i,
    };
    return s.len;
}

/// The symbol of an externally visible kernel definition, still LLVM-escaped
/// if quoted, or null for any other line.
fn parseKernel(line: []const u8) ?[]const u8 {
    if (!std.mem.startsWith(u8, line, "define ")) return null;
    const at = std.mem.indexOfScalar(u8, line, '@') orelse return null;
    // Everything before the name is space-separated keywords and the return
    // type; a kernel returns void, so no word here can hide a space.
    var kernel = false;
    var words = std.mem.tokenizeScalar(u8, line["define ".len..at], ' ');
    while (words.next()) |word| {
        if (std.mem.eql(u8, word, "internal") or std.mem.eql(u8, word, "private")) return null;
        if (std.mem.eql(u8, word, "ptx_kernel") or std.mem.eql(u8, word, "amdgpu_kernel")) kernel = true;
    }
    if (!kernel) return null;

    const rest = line[at + 1 ..];
    const name = if (rest.len > 0 and rest[0] == '"') blk: {
        const close = std.mem.indexOfScalarPos(u8, rest, 1, '"') orelse return null;
        break :blk rest[0 .. close + 1];
    } else rest[0..identEnd(rest)];
    return if (name.len == 0) null else name;
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

/// A plain table rather than a `resolve` function: `emitKernels` merges one of
/// these per kernel root into a single comptime name -> (blob, symbol) map, and
/// a chain of comptime `if`s cannot be merged.
fn emitNames(arena: std.mem.Allocator, input: []const u8) ![]const u8 {
    var out: std.Io.Writer.Allocating = .init(arena);
    try out.writer.writeAll(
        \\//! Generated from the kernel definitions in LLVM IR. Do not edit.
        \\pub const entries = [_][:0]const u8{
        \\
    );
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, input, '\n');
    while (it.next()) |line| if (parseKernel(line)) |name| {
        try out.writer.print("    \"{f}\",\n", .{std.zig.fmtString(try decodeLlvmName(arena, name))});
        n += 1;
    };
    if (n == 0) return error.NoKernelsFound;
    try out.writer.writeAll("};\n");
    return out.written();
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(arena);
    if (args.len != 3) {
        std.debug.print("usage: kernel_ir_tool <input.ll> <names.zig>\n", .{});
        return error.BadUsage;
    }

    const input = try std.Io.Dir.cwd().readFileAlloc(io, args[1], arena, .limited(512 * 1024 * 1024));
    const names = emitNames(arena, input) catch |err| switch (err) {
        // A user who forgot exportKernels gets told what to do, not an error name.
        error.NoKernelsFound => {
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
            return err;
        },
        else => |e| return e,
    };
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = args[2], .data = names });
}

test parseKernel {
    try std.testing.expectEqualStrings("t_map", parseKernel(
        "define ptx_kernel void @t_map(ptr addrspace(1) align 4 captures(none) %0, i64 %1) local_unnamed_addr #2 {",
    ).?);
    try std.testing.expectEqualStrings("\"a b\"", parseKernel(
        "define dso_local amdgpu_kernel void @\"a b\"(ptr addrspace(1) %0) #1 {",
    ).?);
    // Device helpers, declarations, and kernels that are not exported.
    try std.testing.expect(parseKernel("define private fastcc double @device.math.softSin(double %0) unnamed_addr #7 {") == null);
    try std.testing.expect(parseKernel("define internal ptx_kernel void @hidden(ptr %0) {") == null);
    try std.testing.expect(parseKernel("declare ptx_kernel void @ext(ptr)") == null);
    try std.testing.expect(parseKernel("@.str = private unnamed_addr constant [9 x i8] c\"ptx_kernel\\00\"") == null);
}

test emitNames {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    // Escaped as a Zig string, and `\22` in the LLVM name decoded first.
    const names = try emitNames(arena,
        \\define ptx_kernel void @add(ptr %0) {
        \\}
        \\define ptx_kernel void @"k\22x"(ptr %0) {
        \\}
    );
    try std.testing.expect(std.mem.indexOf(u8, names, "    \"add\",\n    \"k\\\"x\",\n") != null);
    try std.testing.expectError(error.NoKernelsFound, emitNames(arena, "define void @f() {\n}\n"));
}
