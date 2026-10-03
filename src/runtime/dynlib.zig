//! `std.DynLib` where std has one; `LoadLibraryA` on Windows, which std
//! dropped. Same `open`/`lookup`/`close` surface, so the backends never branch.

const std = @import("std");
const builtin = @import("builtin");

pub const DynLib = switch (builtin.os.tag) {
    .windows => WinDynLib,
    // The list `std.DynLib` implements; keep in step with std/dynamic_library.zig.
    .linux, .driverkit, .ios, .maccatalyst, .macos, .tvos, .visionos, .watchos, .freebsd, .netbsd, .openbsd, .dragonfly, .illumos => std.DynLib,
    // No loader at all (wasm, freestanding). Opening fails, which the
    // backends already read as "no driver here".
    else => NoDynLib,
};

const HMODULE = *opaque {};
extern "kernel32" fn LoadLibraryA(name: [*:0]const u8) callconv(.winapi) ?HMODULE;
extern "kernel32" fn GetProcAddress(module: HMODULE, name: [*:0]const u8) callconv(.winapi) ?*anyopaque;
extern "kernel32" fn FreeLibrary(module: HMODULE) callconv(.winapi) c_int;

const WinDynLib = struct {
    module: HMODULE,

    pub fn open(path: []const u8) error{ NameTooLong, FileNotFound }!WinDynLib {
        // LoadLibraryA is MAX_PATH-bound anyway.
        var buf: [260:0]u8 = undefined;
        if (path.len >= buf.len) return error.NameTooLong;
        @memcpy(buf[0..path.len], path);
        buf[path.len] = 0;
        return .{ .module = LoadLibraryA(&buf) orelse return error.FileNotFound };
    }

    pub fn close(self: *WinDynLib) void {
        _ = FreeLibrary(self.module);
    }

    pub fn lookup(self: *WinDynLib, comptime T: type, name: [:0]const u8) ?T {
        return @ptrCast(@alignCast(GetProcAddress(self.module, name) orelse return null));
    }
};

const NoDynLib = struct {
    pub fn open(_: []const u8) error{FileNotFound}!NoDynLib {
        return error.FileNotFound;
    }
    pub fn close(_: *NoDynLib) void {}
    pub fn lookup(_: *NoDynLib, comptime T: type, _: [:0]const u8) ?T {
        return null;
    }
};
