//! Assert that the generated CPU kernel compiles to the same machine code as
//! the hand-written loop.
//!
//! Usage: codegen_check <probe.s>
//!
//! This is the library's central claim -- `Kernel(Spec, .cpu)` is a comptime
//! abstraction that leaves no runtime residue -- and it is the reason the host
//! code is written out per backend instead of hiding behind a vtable. It was
//! previously only asserted in the README: `tests/codegen.zig` was compiled and
//! then nothing read the result, so a regression would keep the build green.

const std = @import("std");

/// The two symbols must end up as the same code. In practice LLVM notices they
/// are identical and folds them into one symbol, which is the strongest
/// possible evidence; if it ever stops doing that we fall back to comparing the
/// instruction streams.
const generated = "gompute_scale_relu";
const handwritten = "manual_scale_relu";

fn endsWithSymbol(line: []const u8, symbol: []const u8) bool {
    return std.mem.endsWith(u8, line, symbol) and
        (line.len == symbol.len or !isSymbolChar(line[line.len - symbol.len - 1]));
}

/// `.` and `$` are separators in the mangled names Zig emits (`cg.gompute_...`),
/// not part of the trailing identifier, so they end a symbol rather than
/// continue it. Treating them as symbol characters made every real match fail.
fn isSymbolChar(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_';
}

/// The target of an `X = Y` assembler equate whose left side is `symbol`.
fn aliasTarget(text: []const u8, symbol: []const u8) ?[]const u8 {
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        const eq = std.mem.indexOfScalar(u8, line, '=') orelse continue;
        const lhs = std.mem.trim(u8, line[0..eq], " \t");
        const rhs = std.mem.trim(u8, line[eq + 1 ..], " \t");
        if (lhs.len == 0 or rhs.len == 0) continue;
        if (std.mem.indexOfAny(u8, rhs, " \t+-") != null) continue; // an expression, not an alias
        if (std.mem.eql(u8, lhs, symbol) or endsWithSymbol(lhs, symbol)) return rhs;
    }
    return null;
}

/// Follow equates to the symbol that actually owns the code.
fn resolve(text: []const u8, symbol: []const u8) []const u8 {
    var current = symbol;
    // ponytail: alias chains are one or two hops in practice; the bound just
    // stops a malformed file from looping forever.
    for (0..8) |_| {
        const next = aliasTarget(text, current) orelse return current;
        if (std.mem.eql(u8, next, current)) return current;
        current = next;
    }
    return current;
}

/// True when both names end up owned by the same symbol. LLVM folds identical
/// function bodies, so the usual outcome is that both are equated to a third
/// name -- which is the strongest evidence available that the code is the same.
fn foldedTogether(text: []const u8) bool {
    // Compare the trailing identifier: the same symbol shows up both bare and
    // module-qualified (`gompute_scale_relu` vs `codegen.gompute_scale_relu`).
    return std.mem.eql(u8, trailingName(resolve(text, generated)), trailingName(resolve(text, handwritten)));
}

fn trailingName(s: []const u8) []const u8 {
    const dot = std.mem.lastIndexOfScalar(u8, s, '.') orelse return s;
    return s[dot + 1 ..];
}

/// ponytail: the fallback compares branch targets (`.LBB0_8`) literally, so two
/// functions that are identical apart from label numbering would be reported as
/// different. Unreachable in practice -- LLVM folds identical bodies, and
/// `foldedTogether` returns before we get here. If it ever false-fires the diff
/// says so plainly; normalize label operands then, not before.
///
/// Instruction lines of one function: everything between `symbol:` and the
/// `.size` directive that closes it, minus assembler directives, labels and
/// comments. Register names and mnemonics are kept -- identical codegen means
/// identical text here.
fn body(arena: std.mem.Allocator, text: []const u8, symbol: []const u8) !?[]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    var it = std.mem.splitScalar(u8, text, '\n');
    var inside = false;
    while (it.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (!inside) {
            if (line.len > 1 and line[line.len - 1] == ':' and
                endsWithSymbol(line[0 .. line.len - 1], symbol)) inside = true;
            continue;
        }
        if (std.mem.startsWith(u8, line, ".size")) break;
        if (line.len == 0) continue;
        if (line[0] == '.' or line[0] == '#') continue; // directive or comment
        if (line[line.len - 1] == ':') continue; // local label
        const code = if (std.mem.indexOfScalar(u8, line, '#')) |h|
            std.mem.trim(u8, line[0..h], " \t")
        else
            line;
        if (code.len != 0) try out.append(arena, code);
    }
    return if (inside) out.items else null;
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(arena);
    if (args.len != 2) {
        std.debug.print("usage: codegen_check <probe.s>\n", .{});
        return error.BadUsage;
    }

    const text = try std.Io.Dir.cwd().readFileAlloc(io, args[1], arena, .limited(64 * 1024 * 1024));

    if (foldedTogether(text)) return;

    // Each name may itself be an equate onto whichever identical function won
    // the fold, so look up the body under the resolved symbol, not the export.
    const gen_sym = resolve(text, generated);
    const manual_sym = resolve(text, handwritten);

    const gen = try body(arena, text, gen_sym) orelse {
        std.debug.print("codegen_check: symbol '{s}' (from '{s}') not found in {s}\n", .{ gen_sym, generated, args[1] });
        return error.SymbolNotFound;
    };
    const manual = try body(arena, text, manual_sym) orelse {
        std.debug.print("codegen_check: symbol '{s}' (from '{s}') not found in {s}\n", .{ manual_sym, handwritten, args[1] });
        return error.SymbolNotFound;
    };

    var same = gen.len == manual.len;
    if (same) for (gen, manual) |a, b| {
        if (!std.mem.eql(u8, a, b)) {
            same = false;
            break;
        }
    };
    if (same) return;

    std.debug.print(
        \\codegen_check: the generated CPU kernel no longer matches the hand-written loop.
        \\
        \\Gompute's core claim is that Kernel(Spec, .cpu) is a comptime abstraction with no
        \\runtime residue. These two should be the same machine code:
        \\
        \\  {s}  ({d} instructions)
        \\  {s}  ({d} instructions)
        \\
    , .{ generated, gen.len, handwritten, manual.len });

    const n = @max(gen.len, manual.len);
    for (0..n) |i| {
        const a = if (i < gen.len) gen[i] else "<end>";
        const b = if (i < manual.len) manual[i] else "<end>";
        const mark: []const u8 = if (std.mem.eql(u8, a, b)) " " else ">";
        std.debug.print("{s} {s:<40}  {s}\n", .{ mark, a, b });
    }
    return error.CodegenMismatch;
}

test "folded alias is recognised directly and through a shared target" {
    try std.testing.expect(foldedTogether("\tmanual_scale_relu = cg.gompute_scale_relu\n"));
    try std.testing.expect(foldedTogether("probe.gompute_scale_relu = probe.manual_scale_relu\n"));

    // What LLVM actually emits: both names equated to a third symbol, because a
    // third identical function won the fold.
    try std.testing.expect(foldedTogether(
        "manual_scale_relu = codegen.gompute_inferred\n" ++
            "gompute_scale_relu = codegen.gompute_inferred\n",
    ));

    // Only one of the two folded away -> they are not the same code.
    try std.testing.expect(!foldedTogether("\tmanual_scale_relu = something_else\n"));
    // A symbol that merely ends with the name must not count.
    try std.testing.expect(!foldedTogether("\tx_manual_scale_relu = y_gompute_scale_relu\n"));
}

test "body extraction stops at .size and drops directives and labels" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // Written with escapes: a multiline literal cannot contain a tab, and real
    // assembler output is tab-indented.
    const asm_text = "\t.type\tprobe.manual_scale_relu,@function\n" ++
        "probe.manual_scale_relu:\n" ++
        "\t.cfi_startproc\n" ++
        "\tvmulss\txmm2, xmm0, dword ptr [rdi]  # comment\n" ++
        ".LBB0_1:\n" ++
        "\tvmaxss\txmm2, xmm2, xmm1\n" ++
        "\t.size\tprobe.manual_scale_relu, 16\n" ++
        "\tretq\n";

    const lines = (try body(arena, asm_text, handwritten)).?;
    try std.testing.expectEqual(@as(usize, 2), lines.len);
    try std.testing.expectEqualStrings("vmulss\txmm2, xmm0, dword ptr [rdi]", lines[0]);
    try std.testing.expectEqualStrings("vmaxss\txmm2, xmm2, xmm1", lines[1]);
}

test "a missing symbol is reported, not silently passed" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    try std.testing.expect((try body(arena_state.allocator(), "nothing here\n", generated)) == null);
}
