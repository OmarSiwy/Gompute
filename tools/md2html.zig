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

/// Every writer here is `std.Io.Writer.Allocating` over the caller's arena, so
/// the only error it can produce is OOM and there is nothing a build-time tool
/// can usefully do about that.
fn put(w: *std.Io.Writer, text: []const u8) void {
    w.writeAll(text) catch @panic("OOM");
}

fn fmt(w: *std.Io.Writer, comptime format: []const u8, args: anytype) void {
    w.print(format, args) catch @panic("OOM");
}

/// HTML-escape. Applied to every literal run, so nothing in the source can
/// inject markup.
fn escape(w: *std.Io.Writer, text: []const u8) void {
    for (text) |c| switch (c) {
        '&' => put(w, "&amp;"),
        '<' => put(w, "&lt;"),
        '>' => put(w, "&gt;"),
        '"' => put(w, "&quot;"),
        else => w.writeByte(c) catch @panic("OOM"),
    };
}

/// Inline spans, in one pass. Code spans are handled first by construction:
/// once a backtick opens, everything to the closing backtick is literal, so
/// `**` or `[` inside a code span cannot be misread as markup.
fn inlineSpans(w: *std.Io.Writer, src: []const u8) void {
    var i: usize = 0;
    while (i < src.len) {
        // `code`
        if (src[i] == '`') {
            if (std.mem.indexOfScalarPos(u8, src, i + 1, '`')) |end| {
                put(w, "<code>");
                escape(w, src[i + 1 .. end]);
                put(w, "</code>");
                i = end + 1;
                continue;
            }
        }
        // [text](href)
        if (src[i] == '[') {
            if (std.mem.indexOfScalarPos(u8, src, i, ']')) |close| {
                if (close + 1 < src.len and src[close + 1] == '(') {
                    if (std.mem.indexOfScalarPos(u8, src, close + 2, ')')) |paren| {
                        put(w, "<a href=\"");
                        escape(w, src[close + 2 .. paren]);
                        put(w, "\">");
                        inlineSpans(w, src[i + 1 .. close]);
                        put(w, "</a>");
                        i = paren + 1;
                        continue;
                    }
                }
            }
        }
        // **bold**
        if (i + 1 < src.len and src[i] == '*' and src[i + 1] == '*') {
            if (std.mem.indexOfPos(u8, src, i + 2, "**")) |end| {
                put(w, "<strong>");
                inlineSpans(w, src[i + 2 .. end]);
                put(w, "</strong>");
                i = end + 2;
                continue;
            }
        }
        // *italic* -- never an unmatched or space-led star, which is usually
        // multiplication in the prose around this codebase.
        if (src[i] == '*' and i + 1 < src.len and src[i + 1] != ' ') {
            if (std.mem.indexOfScalarPos(u8, src, i + 1, '*')) |end| {
                put(w, "<em>");
                inlineSpans(w, src[i + 1 .. end]);
                put(w, "</em>");
                i = end + 1;
                continue;
            }
        }
        escape(w, src[i .. i + 1]);
        i += 1;
    }
}

/// Which stylesheet class a token gets, or null to leave it in the body colour.
/// Six classes total -- keyword, builtin, string, number, type, comment -- which
/// is as far as colour helps before it turns into noise.
fn tokenClass(tag: std.zig.Token.Tag, text: []const u8) ?[]const u8 {
    return switch (tag) {
        .builtin => "hl-b",
        .string_literal, .multiline_string_literal_line, .char_literal => "hl-s",
        .number_literal => "hl-n",
        .doc_comment, .container_doc_comment => "hl-c",
        // Zig has no reserved type names, so this is a convention check, not a
        // parse: primitives, plus the TitleCase names the ecosystem uses.
        .identifier => if (std.zig.primitives.isPrimitive(text) or std.ascii.isUpper(text[0]))
            "hl-t"
        else
            null,
        else => if (std.mem.startsWith(u8, @tagName(tag), "keyword_")) "hl-k" else null,
    };
}

fn span(w: *std.Io.Writer, class: []const u8, text: []const u8) void {
    fmt(w, "<span class=\"{s}\">", .{class});
    escape(w, text);
    put(w, "</span>");
}

/// The text between two tokens: whitespace, and `//` comments, which the
/// tokenizer skips rather than reporting.
fn betweenTokens(w: *std.Io.Writer, text: []const u8) void {
    var i: usize = 0;
    while (std.mem.indexOfPos(u8, text, i, "//")) |start| {
        escape(w, text[i..start]);
        const end = std.mem.indexOfScalarPos(u8, text, start, '\n') orelse text.len;
        span(w, "hl-c", text[start..end]);
        i = end;
    }
    escape(w, text[i..]);
}

/// Colour a ```zig block with the compiler's own tokenizer, so the docs cannot
/// disagree with the language about what a keyword is. Everything is escaped on
/// the way out, exactly as an unhighlighted block would be.
fn highlightZig(w: *std.Io.Writer, a: std.mem.Allocator, src: []const u8) void {
    const buf = a.dupeZ(u8, src) catch @panic("OOM");
    var tokenizer = std.zig.Tokenizer.init(buf);
    var prev: usize = 0;
    while (true) {
        const token = tokenizer.next();
        betweenTokens(w, buf[prev..token.loc.start]);
        if (token.tag == .eof) break;
        const text = buf[token.loc.start..token.loc.end];
        if (tokenClass(token.tag, text)) |class| span(w, class, text) else escape(w, text);
        prev = token.loc.end;
    }
}

/// Arena-scoped, like everything else here: the result is a slice into memory
/// that is never freed, and it stays valid as long as `a`'s arena does.
fn slug(a: std.mem.Allocator, text: []const u8) []const u8 {
    var s: std.ArrayList(u8) = .empty;
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

fn emitCells(w: *std.Io.Writer, line: []const u8, tag: []const u8) void {
    const trimmed = std.mem.trim(u8, line, "| \t");
    var it = std.mem.splitScalar(u8, trimmed, '|');
    put(w, "<tr>");
    while (it.next()) |cell| {
        fmt(w, "<{s}>", .{tag});
        inlineSpans(w, std.mem.trim(u8, cell, " \t"));
        fmt(w, "</{s}>", .{tag});
    }
    put(w, "</tr>\n");
}

const Rendered = struct { body: []const u8, toc: []const u8, title: []const u8 };

/// Render one page. `a` must be an arena: nothing here is ever freed, and the
/// three slices in the result point into it.
fn render(a: std.mem.Allocator, src: []const u8) Rendered {
    var body_out: std.Io.Writer.Allocating = .init(a);
    var toc_out: std.Io.Writer.Allocating = .init(a);
    const body = &body_out.writer;
    const toc = &toc_out.writer;
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

        // Fenced code. Contents are literal: no inline processing at all. The
        // info string picks the highlighter; anything but `zig` stays plain.
        if (std.mem.startsWith(u8, line, "```")) {
            const lang = std.mem.trim(u8, line[3..], " \t");
            var block: std.Io.Writer.Allocating = .init(a);
            i += 1;
            while (i < lines.items.len and !std.mem.startsWith(u8, lines.items[i], "```")) : (i += 1)
                fmt(&block.writer, "{s}\n", .{lines.items[i]});
            i += 1; // closing fence
            put(body, "<pre><code>");
            if (std.mem.eql(u8, lang, "zig"))
                highlightZig(body, a, block.written())
            else
                escape(body, block.written());
            put(body, "</code></pre>\n");
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
            fmt(body, "<{s} id=\"{s}\">", .{ tag, id });
            inlineSpans(body, text);
            fmt(body, "</{s}>\n", .{tag});
            if (level == 2) {
                fmt(toc, "<li><a href=\"#{s}\">", .{id});
                inlineSpans(toc, text);
                put(toc, "</a></li>\n");
            }
            i += 1;
            continue;
        }

        if (isTableRow(line)) {
            put(body, "<table>\n");
            const has_head = i + 1 < lines.items.len and isTableRule(lines.items[i + 1]);
            if (has_head) {
                put(body, "<thead>\n");
                emitCells(body, line, "th");
                put(body, "</thead>\n");
                i += 2;
            }
            put(body, "<tbody>\n");
            while (i < lines.items.len and isTableRow(lines.items[i])) : (i += 1) {
                if (isTableRule(lines.items[i])) continue;
                emitCells(body, lines.items[i], "td");
            }
            put(body, "</tbody>\n</table>\n");
            continue;
        }

        // Lists. A continuation line is indented; anything else ends the list.
        const is_ul = std.mem.startsWith(u8, line, "- ");
        const is_ol = ordered(line) != null;
        if (is_ul or is_ol) {
            const tag = if (is_ul) "ul" else "ol";
            fmt(body, "<{s}>\n", .{tag});
            while (i < lines.items.len) {
                const cur = lines.items[i];
                const start = if (std.mem.startsWith(u8, cur, "- "))
                    @as(?usize, 2)
                else
                    ordered(cur);
                if (start == null) break;

                var item: std.Io.Writer.Allocating = .init(a);
                put(&item.writer, cur[start.?..]);
                i += 1;
                // Wrapped continuation lines belong to the same item.
                while (i < lines.items.len and lines.items[i].len > 0 and
                    (lines.items[i][0] == ' ' or lines.items[i][0] == '\t')) : (i += 1)
                    fmt(&item.writer, " {s}", .{std.mem.trim(u8, lines.items[i], " \t")});
                put(body, "<li>");
                inlineSpans(body, item.written());
                put(body, "</li>\n");
            }
            fmt(body, "</{s}>\n", .{tag});
            continue;
        }

        // Paragraph: consume until a blank line or the start of another block.
        var para: std.Io.Writer.Allocating = .init(a);
        while (i < lines.items.len) : (i += 1) {
            const cur = lines.items[i];
            if (cur.len == 0 or cur[0] == '#' or isTableRow(cur) or
                std.mem.startsWith(u8, cur, "```") or std.mem.startsWith(u8, cur, "- ") or
                ordered(cur) != null) break;
            if (para.written().len != 0) put(&para.writer, " ");
            put(&para.writer, std.mem.trim(u8, cur, " \t"));
        }
        put(body, "<p>");
        inlineSpans(body, para.written());
        put(body, "</p>\n");
    }

    return .{ .body = body_out.written(), .toc = toc_out.written(), .title = title };
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
    const body = renderBody(arena.allocator(), "```\nconst x = a[i] * *p; // **not bold**\n```\n");
    try std.testing.expect(std.mem.indexOf(u8, body, "<strong>") == null);
    try std.testing.expect(std.mem.indexOf(u8, body, "<em>") == null);
    try std.testing.expect(std.mem.indexOf(u8, body, "a[i] * *p; // **not bold**") != null);
}

test "a zig fence is highlighted, and highlighting still escapes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const body = renderBody(
        arena.allocator(),
        "```zig\nconst n: u32 = 1; // **not bold** <b>\nconst S = @import(\"s\");\n```\n",
    );
    try std.testing.expect(std.mem.indexOf(u8, body, "<span class=\"hl-k\">const</span>") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "<span class=\"hl-t\">u32</span>") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "<span class=\"hl-n\">1</span>") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "<span class=\"hl-b\">@import</span>") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "<span class=\"hl-s\">&quot;s&quot;</span>") != null);
    // The comment runs to end of line, is one span, and is still escaped.
    try std.testing.expect(std.mem.indexOf(u8, body, "<span class=\"hl-c\">// **not bold** &lt;b&gt;</span>") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "<strong>") == null);
    try std.testing.expect(std.mem.indexOf(u8, body, "<b>") == null);
}

test "an unterminated string in a zig fence does not swallow the rest" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    // The tokenizer reports `.invalid` here; the block must still round-trip.
    const body = renderBody(arena.allocator(), "```zig\nconst s = \"oops\nconst n = 2;\n```\n");
    try std.testing.expect(std.mem.indexOf(u8, body, "oops") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "2") != null);
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
