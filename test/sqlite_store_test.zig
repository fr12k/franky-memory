//! Integration tests for the embedded SQLite memory store.
//!
//! These tests create a real temporary SQLite database and exercise
//! the L0/L1/L2/L3 CRUD + search operations end-to-end.

const std = @import("std");
const agent_memory = @import("agent_memory");
const types = agent_memory.types;
const sqlite_store = agent_memory.embedded;

// ============================
// Test helpers
// ============================

var test_counter: std.atomic.Value(u64) = .init(0);

/// Create a Threaded Io for test filesystem operations.
fn makeIo() std.Io.Threaded {
    return std.Io.Threaded.init(std.testing.allocator, .{});
}

/// Create a temp directory + database path. Returns the paths;
/// caller cleans up via `cleanupTempDir`.
fn makeTempDir(allocator: std.mem.Allocator, io: std.Io) !struct {
    dir: []const u8,
    db_path: [:0]u8,
} {
    const epoch = test_counter.fetchAdd(1, .monotonic);
    const dir = try std.fmt.allocPrint(allocator, "/tmp/agent-memory-test-{d}", .{epoch});
    std.Io.Dir.cwd().createDirPath(io, dir) catch |e| switch (e) {
        error.PathAlreadyExists => {},
        else => return e,
    };
    const db_path = try std.fmt.allocPrintSentinel(allocator, "{s}/memory.db", .{dir}, 0);
    return .{ .dir = dir, .db_path = db_path };
}

fn cleanupTempDir(allocator: std.mem.Allocator, io: std.Io, dir: []const u8) void {
    // Best-effort cleanup.
    std.Io.Dir.cwd().deleteTree(io, dir) catch {};
    allocator.free(dir);
}

const TestCtx = struct {
    allocator: std.mem.Allocator,
    threaded: std.Io.Threaded,
    io: std.Io,
    dir: []const u8,
    db_path: [:0]u8,
    store: sqlite_store.SqliteStore,

    fn init() !TestCtx {
        const allocator = std.testing.allocator;
        var threaded = makeIo();
        const io = threaded.io();
        const tmp = try makeTempDir(allocator, io);
        const store = try sqlite_store.SqliteStore.init(allocator, io, tmp.db_path, tmp.dir);
        return .{
            .allocator = allocator,
            .threaded = threaded,
            .io = io,
            .dir = tmp.dir,
            .db_path = tmp.db_path,
            .store = store,
        };
    }

    fn deinit(self: *TestCtx) void {
        self.store.deinit();
        self.allocator.free(self.db_path);
        cleanupTempDir(self.allocator, self.io, self.dir);
        self.threaded.deinit();
    }
};

/// Free all owned strings in an L0Record.
fn freeL0Record(allocator: std.mem.Allocator, r: types.L0Record) void {
    allocator.free(r.id);
    allocator.free(r.session_key);
    allocator.free(r.session_id);
    allocator.free(r.team_id);
    allocator.free(r.user_id);
    allocator.free(r.agent_id);
    allocator.free(r.task_id);
    allocator.free(r.role);
    allocator.free(r.message_text);
    allocator.free(r.recorded_at);
}

// ============================
// Tests
// ============================

test "SqliteStore init creates schema" {
    var ctx = try TestCtx.init();
    defer ctx.deinit();

    // The store should report FTS5 capability.
    const caps = ctx.store.capabilities;
    try std.testing.expect(caps.fts_search);
}

test "L0 add + query round-trip" {
    var ctx = try TestCtx.init();
    defer ctx.deinit();

    const iso = types.IsolationContext{ .session_id = "s1" };
    const records = [_]types.L0Record{
        .{
            .id = "msg-001",
            .session_key = "sk1",
            .session_id = "s1",
            .role = "user",
            .message_text = "hello world",
            .recorded_at = "2025-01-15T10:00:00Z",
            .timestamp = 1736932800000,
        },
        .{
            .id = "msg-002",
            .session_key = "sk1",
            .session_id = "s1",
            .role = "assistant",
            .message_text = "hi there! how can I help?",
            .recorded_at = "2025-01-15T10:00:01Z",
            .timestamp = 1736932801000,
        },
    };

    try ctx.store.addConversation(&records, iso);

    // Query them back.
    const result = try ctx.store.queryConversation(ctx.allocator, .{ .limit = 10 }, iso);
    defer {
        for (result) |r| freeL0Record(ctx.allocator, r);
        ctx.allocator.free(result);
    }

    try std.testing.expectEqual(@as(usize, 2), result.len);
    try std.testing.expectEqualStrings("msg-001", result[0].id);
    try std.testing.expectEqualStrings("hello world", result[0].message_text);
    try std.testing.expectEqualStrings("msg-002", result[1].id);
    try std.testing.expectEqualStrings("hi there! how can I help?", result[1].message_text);
}

test "L0 query with updated_after filter" {
    var ctx = try TestCtx.init();
    defer ctx.deinit();

    const iso = types.IsolationContext{ .session_id = "s1" };
    const records = [_]types.L0Record{
        .{
            .id = "msg-001",
            .session_key = "sk1",
            .session_id = "s1",
            .role = "user",
            .message_text = "first message",
            .recorded_at = "2025-01-15T10:00:00Z",
            .timestamp = 1736932800000,
        },
        .{
            .id = "msg-002",
            .session_key = "sk1",
            .session_id = "s1",
            .role = "assistant",
            .message_text = "second message",
            .recorded_at = "2025-01-15T11:00:00Z",
            .timestamp = 1736936400000,
        },
    };

    try ctx.store.addConversation(&records, iso);

    // Query only messages after 10:30.
    const result = try ctx.store.queryConversation(
        ctx.allocator,
        .{ .updated_after = "2025-01-15T10:30:00Z", .limit = 10 },
        iso,
    );
    defer {
        for (result) |r| freeL0Record(ctx.allocator, r);
        ctx.allocator.free(result);
    }

    try std.testing.expectEqual(@as(usize, 1), result.len);
    try std.testing.expectEqualStrings("msg-002", result[0].id);
}

test "L1 upsert + FTS search" {
    var ctx = try TestCtx.init();
    defer ctx.deinit();

    const iso = types.IsolationContext{ .session_id = "s1" };
    const record = types.L1Record{
        .record_id = "mem-001",
        .content = "User decided to use PostgreSQL for their database",
        .type = .episodic,
        .priority = 75,
        .scene_name = "database setup",
        .session_key = "sk1",
        .session_id = "s1",
        .team_id = "default",
        .task_id = "",
        .user_id = "default",
        .agent_id = "default",
        .version = 1,
        .timestamp_str = "2025-01-15T10:00:00Z",
        .timestamp_start = "2025-01-15T10:00:00Z",
        .timestamp_end = "2025-01-15T10:05:00Z",
        .created_time = "2025-01-15T10:05:00Z",
        .updated_time = "2025-01-15T10:05:00Z",
        .metadata_json = "{}",
    };

    _ = try ctx.store.upsertL1(record, null, iso);

    // Search for "PostgreSQL".
    const results = try ctx.store.searchL1Fts(ctx.allocator, "PostgreSQL", 5, iso);
    defer {
        for (results) |r| r.deinit(ctx.allocator);
        ctx.allocator.free(results);
    }

    try std.testing.expectEqual(@as(usize, 1), results.len);
    try std.testing.expectEqualStrings("mem-001", results[0].record_id);
    try std.testing.expectEqualStrings("User decided to use PostgreSQL for their database", results[0].content);
}

test "L1 FTS search with no match returns empty" {
    var ctx = try TestCtx.init();
    defer ctx.deinit();

    const iso = types.IsolationContext{ .session_id = "s1" };
    const record = types.L1Record{
        .record_id = "mem-001",
        .content = "User likes Python",
        .type = .persona,
        .priority = 80,
        .scene_name = "preferences",
        .session_key = "sk1",
        .session_id = "s1",
        .team_id = "default",
        .task_id = "",
        .user_id = "default",
        .agent_id = "default",
        .version = 1,
        .timestamp_str = "",
        .timestamp_start = "",
        .timestamp_end = "",
        .created_time = "",
        .updated_time = "",
        .metadata_json = "{}",
    };

    _ = try ctx.store.upsertL1(record, null, iso);

    // Search for something unrelated.
    const results = try ctx.store.searchL1Fts(ctx.allocator, "Java", 5, iso);
    defer {
        for (results) |r| r.deinit(ctx.allocator);
        ctx.allocator.free(results);
    }

    try std.testing.expectEqual(@as(usize, 0), results.len);
}

test "L1 upsert replaces existing record" {
    var ctx = try TestCtx.init();
    defer ctx.deinit();

    const iso = types.IsolationContext{ .session_id = "s1" };

    // Insert first version.
    const r1 = types.L1Record{
        .record_id = "mem-001",
        .content = "User uses MySQL",
        .type = .episodic,
        .priority = 50,
        .scene_name = "database",
        .session_key = "sk1",
        .session_id = "s1",
        .team_id = "default",
        .task_id = "",
        .user_id = "default",
        .agent_id = "default",
        .version = 1,
        .timestamp_str = "",
        .timestamp_start = "",
        .timestamp_end = "",
        .created_time = "",
        .updated_time = "",
        .metadata_json = "{}",
    };
    _ = try ctx.store.upsertL1(r1, null, iso);

    // Upsert with new content.
    const r2 = types.L1Record{
        .record_id = "mem-001",
        .content = "User switched to PostgreSQL",
        .type = .episodic,
        .priority = 75,
        .scene_name = "database",
        .session_key = "sk1",
        .session_id = "s1",
        .team_id = "default",
        .task_id = "",
        .user_id = "default",
        .agent_id = "default",
        .version = 2,
        .timestamp_str = "",
        .timestamp_start = "",
        .timestamp_end = "",
        .created_time = "",
        .updated_time = "",
        .metadata_json = "{}",
    };
    _ = try ctx.store.upsertL1(r2, null, iso);

    // Search — should find only the updated content.
    const results = try ctx.store.searchL1Fts(ctx.allocator, "PostgreSQL", 5, iso);
    defer {
        for (results) |r| r.deinit(ctx.allocator);
        ctx.allocator.free(results);
    }
    try std.testing.expectEqual(@as(usize, 1), results.len);
    try std.testing.expectEqualStrings("mem-001", results[0].record_id);
    try std.testing.expectEqualStrings("User switched to PostgreSQL", results[0].content);

    // Old content should NOT be searchable.
    const old_results = try ctx.store.searchL1Fts(ctx.allocator, "MySQL", 5, iso);
    defer {
        for (old_results) |r| r.deinit(ctx.allocator);
        ctx.allocator.free(old_results);
    }
    try std.testing.expectEqual(@as(usize, 0), old_results.len);
}

test "L3 persona write + read" {
    var ctx = try TestCtx.init();
    defer ctx.deinit();

    const iso = types.IsolationContext{};
    const persona_content = "User's name is Alice. Prefers concise answers. Uses PostgreSQL.";

    try ctx.store.writeCore(persona_content, iso);

    const read = try ctx.store.readCore(ctx.allocator, iso);
    defer if (read) |r| ctx.allocator.free(r);

    try std.testing.expect(read != null);
    try std.testing.expectEqualStrings(persona_content, read.?);
}

test "L3 persona read when missing returns null" {
    var ctx = try TestCtx.init();
    defer ctx.deinit();

    const iso = types.IsolationContext{};
    const read = try ctx.store.readCore(ctx.allocator, iso);
    try std.testing.expectEqual(@as(?[]u8, null), read);
}

test "L2 scenario write + read" {
    var ctx = try TestCtx.init();
    defer ctx.deinit();

    const iso = types.IsolationContext{};
    const scenario_content = "# Debugging Auth Module\n\nFound nil pointer in middleware.\nAdded tests.";

    try ctx.store.writeScenario("debugging-auth.md", scenario_content, iso);

    const read = try ctx.store.readScenario(ctx.allocator, "debugging-auth.md", iso);
    defer if (read) |r| r.deinit(ctx.allocator);

    try std.testing.expect(read != null);
    try std.testing.expectEqualStrings("debugging-auth.md", read.?.path);
    try std.testing.expectEqualStrings(scenario_content, read.?.content);
}

test "L2 scenario list" {
    var ctx = try TestCtx.init();
    defer ctx.deinit();

    const iso = types.IsolationContext{};

    try ctx.store.writeScenario("scenario-a.md", "content a", iso);
    try ctx.store.writeScenario("scenario-b.md", "content b", iso);

    const list = try ctx.store.listScenarios(ctx.allocator, null, iso);
    defer {
        for (list) |f| f.deinit(ctx.allocator);
        ctx.allocator.free(list);
    }

    try std.testing.expectEqual(@as(usize, 2), list.len);
}

test "checkpoint set + get round-trip" {
    var ctx = try TestCtx.init();
    defer ctx.deinit();

    const checkpoint = types.Checkpoint{
        .last_processed_timestamp = "2025-01-15T10:05:00Z",
        .last_scene_name = "database setup",
    };
    try ctx.store.setCheckpoint(checkpoint);

    const read = try ctx.store.getCheckpoint(ctx.allocator);
    // Note: parseCheckpoint is a stub in Phase 1; it returns empty.
    // This test verifies set doesn't crash. Phase 2 will implement parse.
    _ = read;
}

test "recall returns L3 + L2 + L1" {
    var ctx = try TestCtx.init();
    defer ctx.deinit();

    const iso = types.IsolationContext{};

    // Write L3.
    try ctx.store.writeCore("User prefers dark mode.", iso);

    // Write L2.
    try ctx.store.writeScenario("setup.md", "Setting up the project.", iso);

    // Write L1.
    const l1 = types.L1Record{
        .record_id = "mem-001",
        .content = "User uses PostgreSQL",
        .type = .episodic,
        .priority = 75,
        .scene_name = "database",
        .session_key = "sk1",
        .session_id = "s1",
        .team_id = "default",
        .task_id = "",
        .user_id = "default",
        .agent_id = "default",
        .version = 1,
        .timestamp_str = "",
        .timestamp_start = "",
        .timestamp_end = "",
        .created_time = "",
        .updated_time = "",
        .metadata_json = "{}",
    };
    _ = try ctx.store.upsertL1(l1, null, iso);

    // Recall.
    var result = try ctx.store.recall(ctx.allocator, "PostgreSQL", 5, iso);
    defer result.deinit(ctx.allocator);

    // L3 should be present.
    try std.testing.expect(result.persona != null);
    try std.testing.expectEqualStrings("User prefers dark mode.", result.persona.?);

    // L2 should have at least 1 file.
    try std.testing.expect(result.scenario_files.len >= 1);

    // L1 should have 1 hit.
    try std.testing.expectEqual(@as(usize, 1), result.l1_results.len);
    try std.testing.expectEqualStrings("mem-001", result.l1_results[0].record_id);

    // total_chars should be non-zero.
    try std.testing.expect(result.total_chars > 0);
}

test "toMemoryStore + MemoryContext round-trip" {
    var ctx = try TestCtx.init();
    defer ctx.deinit();

    // Wrap the SqliteStore in a MemoryStore vtable.
    const mem_store = ctx.store.toMemoryStore();

    // Use the vtable to save a memory.
    const iso = types.IsolationContext{ .session_id = "s1" };
    const record = types.L1Record{
        .record_id = "mem-vtable-001",
        .content = "User likes Zig",
        .type = .persona,
        .priority = 85,
        .scene_name = "preferences",
        .session_key = "sk1",
        .session_id = "s1",
        .team_id = "default",
        .task_id = "",
        .user_id = "default",
        .agent_id = "default",
        .version = 1,
        .timestamp_str = "",
        .timestamp_start = "",
        .timestamp_end = "",
        .created_time = "",
        .updated_time = "",
        .metadata_json = "{}",
    };
    _ = try mem_store.upsertL1(record, null, iso);

    // Search via the vtable.
    const results = try mem_store.searchL1(ctx.allocator, "Zig", 5, iso);
    defer {
        for (results) |r| r.deinit(ctx.allocator);
        ctx.allocator.free(results);
    }

    try std.testing.expectEqual(@as(usize, 1), results.len);
    try std.testing.expectEqualStrings("mem-vtable-001", results[0].record_id);
    try std.testing.expectEqualStrings("User likes Zig", results[0].content);

    // Recall via the vtable.
    var recall = try mem_store.recall(ctx.allocator, "Zig", 5, iso);
    defer recall.deinit(ctx.allocator);
    try std.testing.expectEqual(@as(usize, 1), recall.l1_results.len);
    try std.testing.expect(recall.total_chars > 0);
}