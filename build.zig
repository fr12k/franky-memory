const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    // Link system sqlite3 (requires libsqlite3-dev / libsqlite3).
    // We use the system library so we don't have to vendor the 9MB amalgamation.
    // FTS5 support is compiled into the system libsqlite3 on Debian/Ubuntu.
    lib_mod.linkSystemLibrary("sqlite3", .{});

    // Export the module under the name "agent_memory" so that dependents
    // can call `agent_memory_dep.module("agent_memory")` to import it.
    _ = b.addModule("agent_memory", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    // NOTE: the module needs sqlite3 linked too — dependents who use this
    // module must also link sqlite3 (or we need to put linkSystemLibrary on
    // the exported module). For now the exported module above does NOT link
    // sqlite3; the consuming project (franky) is responsible for linking it.
    // The self-tests below link it via test_mod.

    const lib = b.addLibrary(.{
        .name = "agent_memory",
        .root_module = lib_mod,
    });

    // ── Tests ──────────────────────────────────────────────────────────
    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    test_mod.linkSystemLibrary("sqlite3", .{});

    const test_bin = b.addTest(.{
        .name = "agent-memory-tests",
        .root_module = test_mod,
    });

    const run_tests = b.addRunArtifact(test_bin);
    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&run_tests.step);

    // ── Integration tests (separate test binary) ──────────────────────
    const itest_mod = b.createModule(.{
        .root_source_file = b.path("test/sqlite_store_test.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    itest_mod.linkSystemLibrary("sqlite3", .{});
    itest_mod.addImport("agent_memory", lib_mod);

    const itest_bin = b.addTest(.{
        .name = "agent-memory-itests",
        .root_module = itest_mod,
    });

    const run_itests = b.addRunArtifact(itest_bin);
    const itest_step = b.step("test-integration", "Run integration tests");
    itest_step.dependOn(&run_itests.step);

    // Combined "test-all" step.
    const test_all_step = b.step("test-all", "Run all tests");
    test_all_step.dependOn(&run_tests.step);
    test_all_step.dependOn(&run_itests.step);

    b.installArtifact(lib);
}