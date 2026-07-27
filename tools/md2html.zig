//! Render one documentation page from Markdown to HTML.
//!
//! Usage: md2html <page.md> <template.html> <out.html>
//!
//! Deliberately not a general Markdown implementation. It covers exactly the
//! subset `docs/*.md` uses -- headings, fenced code, tables, lists, paragraphs,
//! and inline code/bold/italic/links -- so that `zig build docs` produces the
//! whole site with no external tool. A hosted renderer would mean the docs could
//! only be previewed on a machine that had it installed, and would give CI a
//! second rendering path to drift from.
//!
//! ponytail: anything outside the subset falls through as paragraph text rather
//! than erroring. If the docs ever need blockquotes, nested lists or footnotes,
//! add them here -- or switch to a real renderer and accept the dependency.

const std = @import("std");

const Writer = std.ArrayList(u8);

fn put(out: *Writer, a: std.mem.Allocator, text: []const u8) void {
    out.appendSlice(a, text) catch @panic("OOM");
}

/// HTML-escape. Applied to every literal run, so nothing in the source can
/// inject markup.
fn escape(out: *Writer, a: std.mem.Allocator, text: []const u8) void {
    for (text) |c| switch (c) {
        '&' => put(out, a, "&amp;"),
        '<' => put(out, a, "&lt;"),
        '>' => put(out, a, "&gt;"),
        '"' => put(out, a, "&quot;"),
        else => out.append(a, c) catch @panic("OOM"),
    };
}

/// Inline spans, in one pass. Code spans are handled first by construction:
/// once a backtick opens, everything to the closing backtick is literal, so
/// `**` or `[` inside a code span cannot be misread as markup.
fn inlineSpans(out: *Writer, a: std.mem.Allocator, src: []const u8) void {
    var i: usize = 0;
    while (i < src.len) {
        // `code`
        if (src[i] == '`') {
            if (std.mem.indexOfScalarPos(u8, src, i + 1, '`')) |end| {
                put(out, a, "<code>");
                escape(out, a, src[i + 1 .. end]);
                put(out, a, "</code>");
                i = end + 1;
                continue;
            }
        }
        // [text](href)
        if (src[i] == '[') {
            if (std.mem.indexOfScalarPos(u8, src, i, ']')) |close| {
                if (close + 1 < src.len and src[close + 1] == '(') {
                    if (std.mem.indexOfScalarPos(u8, src, close + 2, ')')) |paren| {
                        put(out, a, "<a href=\"");
                        escape(out, a, src[close + 2 .. paren]);
                        put(out, a, "\">");
                        inlineSpans(out, a, src[i + 1 .. close]);
                        put(out, a, "</a>");
                        i = paren + 1;
                        continue;
                    }
                }
            }
        }
        // **bold**
        if (i + 1 < src.len and src[i] == '*' and src[i + 1] == '*') {
            if (std.mem.indexOfPos(u8, src, i + 2, "**")) |end| {
                put(out, a, "<strong>");
                inlineSpans(out, a, src[i + 2 .. end]);
                put(out, a, "</strong>");
                i = end + 2;
                continue;
            }
        }
        // *italic* -- never an unmatched or space-led star, which is usually
        // multiplication in the prose around this codebase.
        if (src[i] == '*' and i + 1 < src.len and src[i + 1] != ' ') {
            if (std.mem.indexOfScalarPos(u8, src, i + 1, '*')) |end| {
                put(out, a, "<em>");
                inlineSpans(out, a, src[i + 1 .. end]);
                put(out, a, "</em>");
                i = end + 1;
                continue;
            }
        }
        escape(out, a, src[i .. i + 1]);
        i += 1;
    }
}

fn slug(a: std.mem.Allocator, text: []const u8) []const u8 {
    var s: Writer = .empty;
    var dash = false;
    for (text) |c| {
        if (std.ascii.isAlphanumeric(c)) {
            s.append(a, std.ascii.toLower(c)) catch @panic("OOM");
            dash = false;
        } else if (!dash and s.items.len != 0) {
            s.append(a, '-') catch @panic("OOM");
            dash = true;
        }
    }
    if (s.items.len > 0 and s.items[s.items.len - 1] == '-') _ = s.pop();
    return s.items;
}

fn isTableRow(line: []const u8) bool {
    return line.len > 0 and line[0] == '|';
}

/// `| --- | ---: |` -- the row that separates a table's head from its body.
fn isTableRule(line: []const u8) bool {
    if (!isTableRow(line)) return false;
    for (line) |c| if (c != '|' and c != '-' and c != ':' and c != ' ') return false;
    return true;
}

fn emitCells(out: *Writer, a: std.mem.Allocator, line: []const u8, tag: []const u8) void {
    const trimmed = std.mem.trim(u8, line, "| \t");
    var it = std.mem.splitScalar(u8, trimmed, '|');
    put(out, a, "<tr>");
    while (it.next()) |cell| {
        put(out, a, "<");
        put(out, a, tag);
        put(out, a, ">");
        inlineSpans(out, a, std.mem.trim(u8, cell, " \t"));
        put(out, a, "</");
        put(out, a, tag);
        put(out, a, ">");
    }
    put(out, a, "</tr>\n");
}

const Rendered = struct { body: []const u8, toc: []const u8, title: []const u8 };

pub fn render(a: std.mem.Allocator, src: []const u8) Rendered {
    var body: Writer = .empty;
    var toc: Writer = .empty;
    var title: []const u8 = "Gompute";

    var lines: std.ArrayList([]const u8) = .empty;
    var split = std.mem.splitScalar(u8, src, '\n');
    while (split.next()) |l| lines.append(a, std.mem.trimEnd(u8, l, "\r")) catch @panic("OOM");

    var i: usize = 0;
    while (i < lines.items.len) {
        const line = lines.items[i];

        if (line.len == 0) {
            i += 1;
            continue;
        }

        // Fenced code. Contents are literal: no inline processing at all.
        if (std.mem.startsWith(u8, line, "```")) {
            put(&body, a, "<pre><code>");
            i += 1;
            while (i < lines.items.len and !std.mem.startsWith(u8, lines.items[i], "```")) : (i += 1) {
                escape(&body, a, lines.items[i]);
                put(&body, a, "\n");
            }
            i += 1; // closing fence
            put(&body, a, "</code></pre>\n");
            continue;
        }

        if (line[0] == '#') {
            var level: usize = 0;
            while (level < line.len and line[level] == '#') level += 1;
            const text = std.mem.trim(u8, line[level..], " \t");
            if (level == 1) title = text;
            const id = slug(a, text);
            const tag = switch (level) {
                1 => "h1",
                2 => "h2",
                else => "h3",
            };
            put(&body, a, "<");
            put(&body, a, tag);
            put(&body, a, " id=\"");
            put(&body, a, id);
            put(&body, a, "\">");
            inlineSpans(&body, a, text);
            put(&body, a, "</");
            put(&body, a, tag);
            put(&body, a, ">\n");
            if (level == 2) {
                put(&toc, a, "<li><a href=\"#");
                put(&toc, a, id);
                put(&toc, a, "\">");
                inlineSpans(&toc, a, text);
                put(&toc, a, "</a></li>\n");
            }
            i += 1;
            continue;
        }

        if (isTableRow(line)) {
            put(&body, a, "<table>\n");
            const has_head = i + 1 < lines.items.len and isTableRule(lines.items[i + 1]);
            if (has_head) {
                put(&body, a, "<thead>\n");
                emitCells(&body, a, line, "th");
                put(&body, a, "</thead>\n");
                i += 2;
            }
            put(&body, a, "<tbody>\n");
            while (i < lines.items.len and isTableRow(lines.items[i])) : (i += 1) {
                if (isTableRule(lines.items[i])) continue;
                emitCells(&body, a, lines.items[i], "td");
            }
            put(&body, a, "</tbody>\n</table>\n");
            continue;
        }

        // Lists. A continuation line is indented; anything else ends the list.
        const is_ul = std.mem.startsWith(u8, line, "- ");
        const is_ol = ordered(line) != null;
        if (is_ul or is_ol) {
            const tag = if (is_ul) "ul" else "ol";
            put(&body, a, "<");
            put(&body, a, tag);
            put(&body, a, ">\n");
            while (i < lines.items.len) {
                const cur = lines.items[i];
                const start = if (std.mem.startsWith(u8, cur, "- "))
                    @as(?usize, 2)
                else
                    ordered(cur);
                if (start == null) break;

                var item: Writer = .empty;
                put(&item, a, cur[start.?..]);
                i += 1;
                // Wrapped continuation lines belong to the same item.
                while (i < lines.items.len and lines.items[i].len > 0 and
                    (lines.items[i][0] == ' ' or lines.items[i][0] == '\t')) : (i += 1)
                {
                    put(&item, a, " ");
                    put(&item, a, std.mem.trim(u8, lines.items[i], " \t"));
                }
                put(&body, a, "<li>");
                inlineSpans(&body, a, item.items);
                put(&body, a, "</li>\n");
            }
            put(&body, a, "</");
            put(&body, a, tag);
            put(&body, a, ">\n");
            continue;
        }

        // Paragraph: consume until a blank line or the start of another block.
        var para: Writer = .empty;
        while (i < lines.items.len) : (i += 1) {
            const cur = lines.items[i];
            if (cur.len == 0 or cur[0] == '#' or isTableRow(cur) or
                std.mem.startsWith(u8, cur, "```") or std.mem.startsWith(u8, cur, "- ") or
                ordered(cur) != null) break;
            if (para.items.len != 0) put(&para, a, " ");
            put(&para, a, std.mem.trim(u8, cur, " \t"));
        }
        put(&body, a, "<p>");
        inlineSpans(&body, a, para.items);
        put(&body, a, "</p>\n");
    }

    return .{ .body = body.items, .toc = toc.items, .title = title };
}

/// Offset of the text in `1. item`, or null when the line is not an ordered item.
fn ordered(line: []const u8) ?usize {
    var j: usize = 0;
    while (j < line.len and std.ascii.isDigit(line[j])) j += 1;
    if (j == 0 or j + 1 >= line.len) return null;
    if (line[j] != '.' or line[j + 1] != ' ') return null;
    return j + 2;
}

fn substitute(a: std.mem.Allocator, template: []const u8, r: Rendered) []const u8 {
    var out = std.mem.replaceOwned(u8, a, template, "$title$", r.title) catch @panic("OOM");
    out = std.mem.replaceOwned(u8, a, out, "$toc$", r.toc) catch @panic("OOM");
    out = std.mem.replaceOwned(u8, a, out, "$body$", r.body) catch @panic("OOM");
    return out;
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 4) {
        std.debug.print("usage: md2html <page.md> <template.html> <out.html>\n", .{});
        return error.BadUsage;
    }
    const cwd = std.Io.Dir.cwd();
    const src = try cwd.readFileAlloc(io, args[1], a, .limited(8 * 1024 * 1024));
    const template = try cwd.readFileAlloc(io, args[2], a, .limited(1024 * 1024));
    const page = substitute(a, template, render(a, src));
    try cwd.writeFile(io, .{ .sub_path = args[3], .data = page });
}

// ---- tests ----

fn renderBody(a: std.mem.Allocator, src: []const u8) []const u8 {
    return render(a, src).body;
}

test "headings carry slug ids and the h1 becomes the title" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const r = render(arena.allocator(), "# Build integration\n\n## Pin `.auto` off\n");
    try std.testing.expectEqualStrings("Build integration", r.title);
    try std.testing.expect(std.mem.indexOf(u8, r.body, "<h1 id=\"build-integration\">") != null);
    try std.testing.expect(std.mem.indexOf(u8, r.body, "<h2 id=\"pin-auto-off\">") != null);
    // Only h2s go in the table of contents.
    try std.testing.expect(std.mem.indexOf(u8, r.toc, "#pin-auto-off") != null);
    try std.testing.expect(std.mem.indexOf(u8, r.toc, "build-integration") == null);
}

test "fenced code is literal: markup inside it is not interpreted" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const body = renderBody(arena.allocator(), "```zig\nconst x = a[i] * *p; // **not bold**\n```\n");
    try std.testing.expect(std.mem.indexOf(u8, body, "<strong>") == null);
    try std.testing.expect(std.mem.indexOf(u8, body, "<em>") == null);
    try std.testing.expect(std.mem.indexOf(u8, body, "a[i] * *p; // **not bold**") != null);
}

test "html in the source is escaped, not passed through" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const body = renderBody(arena.allocator(), "Use `if (y > 0)` and <script>alert(1)</script>\n");
    try std.testing.expect(std.mem.indexOf(u8, body, "<script>") == null);
    try std.testing.expect(std.mem.indexOf(u8, body, "&lt;script&gt;") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "<code>if (y &gt; 0)</code>") != null);
}

test "a code span protects its contents from inline markup" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const body = renderBody(arena.allocator(), "call `f(**a**, [b](c))` now\n");
    try std.testing.expect(std.mem.indexOf(u8, body, "<strong>") == null);
    try std.testing.expect(std.mem.indexOf(u8, body, "<a href") == null);
}

test "tables get a head when a rule row follows" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const body = renderBody(arena.allocator(), "| A | B |\n| --- | ---: |\n| `x` | y |\n");
    try std.testing.expect(std.mem.indexOf(u8, body, "<thead>") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "<th>A</th>") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "<td><code>x</code></td>") != null);
    // The rule row itself must never become a data row.
    try std.testing.expect(std.mem.indexOf(u8, body, "<td>---</td>") == null);
}

test "headerless tables still render" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const body = renderBody(arena.allocator(), "| a | b |\n| c | d |\n");
    try std.testing.expect(std.mem.indexOf(u8, body, "<thead>") == null);
    try std.testing.expect(std.mem.indexOf(u8, body, "<td>a</td>") != null);
}

test "lists absorb wrapped continuation lines" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const body = renderBody(arena.allocator(), "- first item\n  wrapped on\n- second\n");
    try std.testing.expect(std.mem.indexOf(u8, body, "<li>first item wrapped on</li>") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "<li>second</li>") != null);
}

test "ordered lists are recognised and 1.5 is not a list" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const body = renderBody(arena.allocator(), "1. one\n2. two\n");
    try std.testing.expect(std.mem.indexOf(u8, body, "<ol>") != null);
    try std.testing.expect(ordered("1.5 is a number") == null);
    try std.testing.expect(ordered("- dash") == null);
}

test "links render and nest inline markup" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const body = renderBody(arena.allocator(), "see [the **guide**](guide.html) now\n");
    try std.testing.expect(std.mem.indexOf(u8, body, "<a href=\"guide.html\">the <strong>guide</strong></a>") != null);
}

test "paragraphs join wrapped lines and stop at the next block" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const body = renderBody(arena.allocator(), "one line\nand another\n\n## Next\n");
    try std.testing.expect(std.mem.indexOf(u8, body, "<p>one line and another</p>") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "<h2 id=\"next\">Next</h2>") != null);
}

test "template substitution fills every placeholder" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const page = substitute(a, "<title>$title$</title>$toc$$body$", render(a, "# T\n\n## S\n\nhi\n"));
    try std.testing.expect(std.mem.indexOf(u8, page, "<title>T</title>") != null);
    try std.testing.expect(std.mem.indexOf(u8, page, "$body$") == null);
    try std.testing.expect(std.mem.indexOf(u8, page, "$toc$") == null);
    try std.testing.expect(std.mem.indexOf(u8, page, "<p>hi</p>") != null);
}
