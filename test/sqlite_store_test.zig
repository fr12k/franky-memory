//! Integration tests for the embedded SQLite memory store.
//!
//! These tests create a real temporary SQLite database and exercise
//! the L1 CRUD + search operations end-to-end.

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
        const store = try sqlite_store.SqliteStore.init(allocator, io, tmp.db_path);
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

// ============================
// Tests
// ============================

/// Build a minimal L1 record with the given id/content (all other fields
/// have safe defaults), for delete/search tests.
fn makeRecord(record_id: []const u8, content: []const u8) types.L1Record {
    return .{
        .record_id = record_id,
        .content = content,
        .type = .episodic,
        .priority = 75,
        .scene_name = "test",
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
}

test "SqliteStore init creates schema" {
    var ctx = try TestCtx.init();
    defer ctx.deinit();

    // The store should report FTS5 capability.
    const caps = ctx.store.capabilities;
    try std.testing.expect(caps.fts_search);
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

    _ = try ctx.store.upsertL1(record, iso);

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

    _ = try ctx.store.upsertL1(record, iso);

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
    _ = try ctx.store.upsertL1(r1, iso);

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
    _ = try ctx.store.upsertL1(r2, iso);

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

test "recall returns L1" {
    var ctx = try TestCtx.init();
    defer ctx.deinit();

    const iso = types.IsolationContext{};

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
    _ = try ctx.store.upsertL1(l1, iso);

    // Recall.
    var result = try ctx.store.recall(ctx.allocator, "PostgreSQL", 5, iso);
    defer result.deinit(ctx.allocator);

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
    _ = try mem_store.upsertL1(record, iso);

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
test "L1 FTS search finds only matching content" {
    var ctx = try TestCtx.init();
    defer ctx.deinit();

    const iso = types.IsolationContext{ .session_id = "s1" };

    const rec1 = types.L1Record{
        .record_id = "vec-001",
        .content = "User likes PostgreSQL",
        .type = .persona,
        .priority = 80,
        .scene_name = "db",
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
    var rec2 = rec1;
    rec2.record_id = "vec-002";
    rec2.content = "User likes MySQL";
    var rec3 = rec1;
    rec3.record_id = "vec-003";
    rec3.content = "User likes SQLite";

    _ = try ctx.store.upsertL1(rec1, iso);
    _ = try ctx.store.upsertL1(rec2, iso);
    _ = try ctx.store.upsertL1(rec3, iso);

    // Search — FTS ranks by BM25 relevance.
    const results = try ctx.store.searchL1Fts(ctx.allocator, "PostgreSQL", 3, iso);
    defer {
        for (results) |r| r.deinit(ctx.allocator);
        ctx.allocator.free(results);
    }

    try std.testing.expectEqual(@as(usize, 1), results.len);
    try std.testing.expectEqualStrings("vec-001", results[0].record_id);
}

test "L1 hybrid search merges FTS results" {
    var ctx = try TestCtx.init();
    defer ctx.deinit();

    const iso = types.IsolationContext{ .session_id = "s1" };

    const rec1 = types.L1Record{
        .record_id = "hybrid-001",
        .content = "User prefers PostgreSQL database",
        .type = .persona,
        .priority = 80,
        .scene_name = "db",
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
    var rec2 = rec1;
    rec2.record_id = "hybrid-002";
    rec2.content = "User likes MySQL";

    _ = try ctx.store.upsertL1(rec1, iso);
    _ = try ctx.store.upsertL1(rec2, iso);

    // Hybrid search (FTS-only mode: no vector search in this build).
    const results = try ctx.store.searchL1Hybrid(ctx.allocator, "PostgreSQL", 5, iso);
    defer {
        for (results) |r| r.deinit(ctx.allocator);
        ctx.allocator.free(results);
    }

    // Should return results.
    try std.testing.expect(results.len > 0);
    // hybrid-001 matches the query.
    try std.testing.expectEqualStrings("hybrid-001", results[0].record_id);
}

test "L1 hybrid search without embedding returns FTS only" {
    var ctx = try TestCtx.init();
    defer ctx.deinit();

    const iso = types.IsolationContext{ .session_id = "s1" };

    const rec1 = types.L1Record{
        .record_id = "hybrid-003",
        .content = "User prefers dark mode",
        .type = .persona,
        .priority = 70,
        .scene_name = "ui",
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
    _ = try ctx.store.upsertL1(rec1, iso);

    // Hybrid search (FTS-only mode — no embeddings in this build).
    const results = try ctx.store.searchL1Hybrid(ctx.allocator, "dark", 5, iso);
    defer {
        for (results) |r| r.deinit(ctx.allocator);
        ctx.allocator.free(results);
    }

    try std.testing.expectEqual(@as(usize, 1), results.len);
    try std.testing.expectEqualStrings("hybrid-003", results[0].record_id);
}

test "recallWithBudget caps total content" {
    var ctx = try TestCtx.init();
    defer ctx.deinit();

    const iso = types.IsolationContext{};

    // Write L1 records.
    const rec1 = types.L1Record{
        .record_id = "budget-001",
        .content = "User prefers PostgreSQL over MySQL", // 36 chars
        .type = .persona,
        .priority = 80,
        .scene_name = "db",
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
    var rec2 = rec1;
    rec2.record_id = "budget-002";
    rec2.content = "User uses PostgreSQL for their Python backend"; // matches query

    _ = try ctx.store.upsertL1(rec1, iso);
    _ = try ctx.store.upsertL1(rec2, iso);

    // Recall with unlimited budget — should get everything.
    var unlimited = try ctx.store.recallWithBudget(ctx.allocator, "PostgreSQL", 5, iso, 0);
    defer unlimited.deinit(ctx.allocator);
    try std.testing.expect(unlimited.total_chars > 0);
    try std.testing.expectEqual(@as(usize, 2), unlimited.l1_results.len);

    // Recall with tight budget (50 chars) — only enough L1 fits.
    var capped = try ctx.store.recallWithBudget(ctx.allocator, "PostgreSQL", 5, iso, 50);
    defer capped.deinit(ctx.allocator);
    try std.testing.expect(capped.total_chars <= 50);
}

test "recallWithBudget with zero budget returns everything" {
    var ctx = try TestCtx.init();
    defer ctx.deinit();

    const iso = types.IsolationContext{};

    const rec = types.L1Record{
        .record_id = "budget-003",
        .content = "Some memory content",
        .type = .episodic,
        .priority = 50,
        .scene_name = "",
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
    _ = try ctx.store.upsertL1(rec, iso);

    var result = try ctx.store.recallWithBudget(ctx.allocator, "memory", 5, iso, 0);
    defer result.deinit(ctx.allocator);
    try std.testing.expectEqual(@as(usize, 1), result.l1_results.len);
}

// ============================
// Delete (soft + hard) tests
// ============================

test "soft delete hides record from search and recall" {
    var ctx = try TestCtx.init();
    defer ctx.deinit();

    const iso = types.IsolationContext{};
    _ = try ctx.store.upsertL1(makeRecord("del-001", "User prefers PostgreSQL over MySQL"), iso);

    // Sanity: it is searchable before the delete.
    {
        const results = try ctx.store.searchL1Fts(ctx.allocator, "PostgreSQL", 5, iso);
        defer {
            for (results) |r| r.deinit(ctx.allocator);
            ctx.allocator.free(results);
        }
        try std.testing.expectEqual(@as(usize, 1), results.len);
    }

    // Soft delete (default options).
    const deleted = try ctx.store.deleteL1("del-001", .{}, iso);
    try std.testing.expect(deleted);

    // No longer searchable.
    {
        const results = try ctx.store.searchL1Fts(ctx.allocator, "PostgreSQL", 5, iso);
        defer {
            for (results) |r| r.deinit(ctx.allocator);
            ctx.allocator.free(results);
        }
        try std.testing.expectEqual(@as(usize, 0), results.len);
    }

    // No longer recalled.
    {
        var recall = try ctx.store.recall(ctx.allocator, "PostgreSQL", 5, iso);
        defer recall.deinit(ctx.allocator);
        try std.testing.expectEqual(@as(usize, 0), recall.l1_results.len);
    }
}

test "soft delete twice returns false" {
    var ctx = try TestCtx.init();
    defer ctx.deinit();

    const iso = types.IsolationContext{};
    _ = try ctx.store.upsertL1(makeRecord("del-002", "User likes SQLite"), iso);

    try std.testing.expect(try ctx.store.deleteL1("del-002", .{}, iso));
    // Already soft-deleted → false.
    try std.testing.expect(!(try ctx.store.deleteL1("del-002", .{}, iso)));
}

test "delete unknown record returns false" {
    var ctx = try TestCtx.init();
    defer ctx.deinit();

    const iso = types.IsolationContext{};
    try std.testing.expect(!(try ctx.store.deleteL1("does-not-exist", .{}, iso)));
    try std.testing.expect(!(try ctx.store.restoreL1("does-not-exist", iso)));
}

test "restore revives soft-deleted record" {
    var ctx = try TestCtx.init();
    defer ctx.deinit();

    const iso = types.IsolationContext{};
    _ = try ctx.store.upsertL1(makeRecord("del-003", "User prefers dark mode"), iso);

    try std.testing.expect(try ctx.store.deleteL1("del-003", .{}, iso));

    // Restore.
    try std.testing.expect(try ctx.store.restoreL1("del-003", iso));

    // Searchable again with the original content.
    {
        const results = try ctx.store.searchL1Fts(ctx.allocator, "dark", 5, iso);
        defer {
            for (results) |r| r.deinit(ctx.allocator);
            ctx.allocator.free(results);
        }
        try std.testing.expectEqual(@as(usize, 1), results.len);
        try std.testing.expectEqualStrings("del-003", results[0].record_id);
    }

    // Restoring a live record returns false.
    try std.testing.expect(!(try ctx.store.restoreL1("del-003", iso)));
}

test "hard delete removes record permanently" {
    var ctx = try TestCtx.init();
    defer ctx.deinit();

    const iso = types.IsolationContext{};
    _ = try ctx.store.upsertL1(makeRecord("del-004", "User uses MySQL in production"), iso);

    // Hard delete.
    try std.testing.expect(try ctx.store.deleteL1("del-004", .{ .soft = false }, iso));

    // Not searchable.
    {
        const results = try ctx.store.searchL1Fts(ctx.allocator, "MySQL", 5, iso);
        defer {
            for (results) |r| r.deinit(ctx.allocator);
            ctx.allocator.free(results);
        }
        try std.testing.expectEqual(@as(usize, 0), results.len);
    }

    // Cannot be restored (row is gone).
    try std.testing.expect(!(try ctx.store.restoreL1("del-004", iso)));

    // Hard-deleting the same id again returns false (row is physically gone).
    try std.testing.expect(!(try ctx.store.deleteL1("del-004", .{ .soft = false }, iso)));
}

test "hard delete force-removes a soft-deleted record" {
    var ctx = try TestCtx.init();
    defer ctx.deinit();

    const iso = types.IsolationContext{};
    _ = try ctx.store.upsertL1(makeRecord("del-005", "User uses MongoDB in staging"), iso);

    // Soft delete first — the row is now deleted=1.
    try std.testing.expect(try ctx.store.deleteL1("del-005", .{}, iso));

    // A second SOFT delete is a no-op (already soft-deleted).
    try std.testing.expect(!(try ctx.store.deleteL1("del-005", .{}, iso)));

    // A HARD delete force-removes the soft-deleted row.
    try std.testing.expect(try ctx.store.deleteL1("del-005", .{ .soft = false }, iso));

    // Gone for good: not restorable, not searchable.
    try std.testing.expect(!(try ctx.store.restoreL1("del-005", iso)));
    const results = try ctx.store.searchL1Fts(ctx.allocator, "MongoDB", 5, iso);
    defer {
        for (results) |r| r.deinit(ctx.allocator);
        ctx.allocator.free(results);
    }
    try std.testing.expectEqual(@as(usize, 0), results.len);

    // And it no longer counts as soft-deleted for a purge.
    try std.testing.expectEqual(@as(u32, 0), try ctx.store.purgeDeletedL1(iso));
}

test "purge removes soft-deleted records and returns count" {
    var ctx = try TestCtx.init();
    defer ctx.deinit();

    const iso = types.IsolationContext{};
    _ = try ctx.store.upsertL1(makeRecord("purge-001", "User likes Redis"), iso);
    _ = try ctx.store.upsertL1(makeRecord("purge-002", "User likes Memcached"), iso);
    _ = try ctx.store.upsertL1(makeRecord("purge-003", "User likes SQLite"), iso);

    // Soft-delete two of the three.
    try std.testing.expect(try ctx.store.deleteL1("purge-001", .{}, iso));
    try std.testing.expect(try ctx.store.deleteL1("purge-002", .{}, iso));

    // Purge — should remove exactly the two soft-deleted rows.
    const purged = try ctx.store.purgeDeletedL1(iso);
    try std.testing.expectEqual(@as(u32, 2), purged);

    // Nothing left to purge.
    try std.testing.expectEqual(@as(u32, 0), try ctx.store.purgeDeletedL1(iso));

    // Purged rows cannot be restored.
    try std.testing.expect(!(try ctx.store.restoreL1("purge-001", iso)));
    try std.testing.expect(!(try ctx.store.restoreL1("purge-002", iso)));

    // The live record survives the purge and is still searchable.
    {
        const results = try ctx.store.searchL1Fts(ctx.allocator, "SQLite", 5, iso);
        defer {
            for (results) |r| r.deinit(ctx.allocator);
            ctx.allocator.free(results);
        }
        try std.testing.expectEqual(@as(usize, 1), results.len);
        try std.testing.expectEqualStrings("purge-003", results[0].record_id);
    }
}

test "delete respects isolation context" {
    var ctx = try TestCtx.init();
    defer ctx.deinit();

    const iso_a = types.IsolationContext{ .user_id = "alice" };
    const iso_b = types.IsolationContext{ .user_id = "bob" };

    // alice owns the record.
    var rec = makeRecord("iso-001", "Alice prefers PostgreSQL");
    rec.user_id = "alice";
    _ = try ctx.store.upsertL1(rec, iso_a);

    // bob cannot delete alice's record.
    try std.testing.expect(!(try ctx.store.deleteL1("iso-001", .{}, iso_b)));

    // alice can.
    try std.testing.expect(try ctx.store.deleteL1("iso-001", .{}, iso_a));

    // bob cannot restore it either.
    try std.testing.expect(!(try ctx.store.restoreL1("iso-001", iso_b)));

    // alice can.
    try std.testing.expect(try ctx.store.restoreL1("iso-001", iso_a));
}

test "purge is scoped to isolation context" {
    var ctx = try TestCtx.init();
    defer ctx.deinit();

    const iso_alice = types.IsolationContext{ .user_id = "alice" };
    const iso_bob = types.IsolationContext{ .user_id = "bob" };

    // alice and bob each soft-delete one of their own records.
    var rec_a = makeRecord("purge-iso-a", "Alice likes Redis");
    rec_a.user_id = "alice";
    _ = try ctx.store.upsertL1(rec_a, iso_alice);
    try std.testing.expect(try ctx.store.deleteL1("purge-iso-a", .{}, iso_alice));

    var rec_b = makeRecord("purge-iso-b", "Bob likes SQLite");
    rec_b.user_id = "bob";
    _ = try ctx.store.upsertL1(rec_b, iso_bob);
    try std.testing.expect(try ctx.store.deleteL1("purge-iso-b", .{}, iso_bob));

    // Bob purges — only his own soft-deleted row is removed.
    try std.testing.expectEqual(@as(u32, 1), try ctx.store.purgeDeletedL1(iso_bob));

    // Alice's soft-deleted record is still restorable.
    try std.testing.expect(try ctx.store.restoreL1("purge-iso-a", iso_alice));
}

test "upsert revives soft-deleted record" {
    var ctx = try TestCtx.init();
    defer ctx.deinit();

    const iso = types.IsolationContext{};
    _ = try ctx.store.upsertL1(makeRecord("revive-001", "User uses MySQL"), iso);

    // Soft delete.
    try std.testing.expect(try ctx.store.deleteL1("revive-001", .{}, iso));

    // A soft-deleted row is restorable (restore semantics sanity check).
    try std.testing.expect(try ctx.store.restoreL1("revive-001", iso));

    // Soft delete again, then revive via upsert instead of restore.
    try std.testing.expect(try ctx.store.deleteL1("revive-001", .{}, iso));
    _ = try ctx.store.upsertL1(makeRecord("revive-001", "User uses PostgreSQL now"), iso);

    // Searchable again with the new content.
    {
        const results = try ctx.store.searchL1Fts(ctx.allocator, "PostgreSQL", 5, iso);
        defer {
            for (results) |r| r.deinit(ctx.allocator);
            ctx.allocator.free(results);
        }
        try std.testing.expectEqual(@as(usize, 1), results.len);
        try std.testing.expectEqualStrings("User uses PostgreSQL now", results[0].content);
    }

    // The revived row is live — restore now returns false.
    try std.testing.expect(!(try ctx.store.restoreL1("revive-001", iso)));
}

// ============================
// Migration tests
// ============================

/// Create a legacy (pre-soft-delete) database: the v0.4.2 schema WITHOUT the
/// `deleted` column, with the unguarded l1_au trigger, and one seeded row.
fn createLegacyDb(db_path: [:0]const u8) !void {
    var db = try agent_memory.sqlite.Db.open(db_path);
    defer db.close();

    try db.exec(
        \\CREATE TABLE l1_records (
        \\  record_id TEXT PRIMARY KEY,
        \\  content TEXT NOT NULL,
        \\  type TEXT NOT NULL,
        \\  priority REAL NOT NULL DEFAULT 50,
        \\  scene_name TEXT NOT NULL DEFAULT '',
        \\  session_key TEXT NOT NULL,
        \\  session_id TEXT NOT NULL,
        \\  team_id TEXT NOT NULL DEFAULT 'default',
        \\  task_id TEXT NOT NULL DEFAULT '',
        \\  user_id TEXT NOT NULL DEFAULT 'default',
        \\  agent_id TEXT NOT NULL DEFAULT 'default',
        \\  version INTEGER NOT NULL DEFAULT 1,
        \\  timestamp_str TEXT NOT NULL DEFAULT '',
        \\  timestamp_start TEXT NOT NULL DEFAULT '',
        \\  timestamp_end TEXT NOT NULL DEFAULT '',
        \\  created_time TEXT NOT NULL DEFAULT '',
        \\  updated_time TEXT NOT NULL DEFAULT '',
        \\  metadata_json TEXT NOT NULL DEFAULT '{}'
        \\)
    );
    try db.exec(
        \\CREATE VIRTUAL TABLE l1_fts USING fts5(
        \\  content,
        \\  content='l1_records',
        \\  content_rowid='rowid'
        \\)
    );
    try db.exec(
        \\CREATE TRIGGER l1_ai AFTER INSERT ON l1_records BEGIN
        \\  INSERT INTO l1_fts(rowid, content) VALUES (new.rowid, new.content);
        \\END
    );
    try db.exec(
        \\CREATE TRIGGER l1_ad AFTER DELETE ON l1_records BEGIN
        \\  INSERT INTO l1_fts(l1_fts, rowid, content) VALUES('delete', old.rowid, old.content);
        \\END
    );
    try db.exec(
        \\CREATE TRIGGER l1_au AFTER UPDATE ON l1_records BEGIN
        \\  INSERT INTO l1_fts(l1_fts, rowid, content) VALUES('delete', old.rowid, old.content);
        \\  INSERT INTO l1_fts(rowid, content) VALUES (new.rowid, new.content);
        \\END
    );

    // Seed one row (via the legacy insert, fires l1_ai).
    const insert =
        "INSERT INTO l1_records " ++
        "(record_id, content, type, priority, scene_name, session_key, session_id, " ++
        "team_id, task_id, user_id, agent_id, version, " ++
        "timestamp_str, timestamp_start, timestamp_end, created_time, updated_time, metadata_json) " ++
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";
    var stmt = try db.prepare(insert);
    defer stmt.finalize();
    try stmt.bindText(1, "legacy-001");
    try stmt.bindText(2, "User uses PostgreSQL (legacy row)");
    try stmt.bindText(3, "episodic");
    try stmt.bindFloat(4, 75);
    try stmt.bindText(5, "legacy");
    try stmt.bindText(6, "sk1");
    try stmt.bindText(7, "s1");
    try stmt.bindText(8, "default");
    try stmt.bindText(9, "");
    try stmt.bindText(10, "default");
    try stmt.bindText(11, "default");
    try stmt.bindInt(12, 1);
    try stmt.bindText(13, "");
    try stmt.bindText(14, "");
    try stmt.bindText(15, "");
    try stmt.bindText(16, "");
    try stmt.bindText(17, "");
    try stmt.bindText(18, "{}");
    _ = try stmt.step();
}

test "init migrates legacy database in place" {
    const allocator = std.testing.allocator;
    var threaded = makeIo();
    defer threaded.deinit();
    const io = threaded.io();

    const tmp = try makeTempDir(allocator, io);
    defer {
        std.Io.Dir.cwd().deleteTree(io, tmp.dir) catch {};
        allocator.free(tmp.dir);
        allocator.free(tmp.db_path);
    }

    // Create the legacy database (schema WITHOUT the deleted column).
    try createLegacyDb(tmp.db_path);

    // Open with the current SqliteStore — init() must migrate it.
    var store = try sqlite_store.SqliteStore.init(allocator, io, tmp.db_path);
    defer store.deinit();

    // The pre-existing row must still be visible (deleted defaults to 0).
    const iso = types.IsolationContext{};
    {
        const results = try store.searchL1Fts(allocator, "PostgreSQL", 5, iso);
        defer {
            for (results) |r| r.deinit(allocator);
            allocator.free(results);
        }
        try std.testing.expectEqual(@as(usize, 1), results.len);
        try std.testing.expectEqualStrings("legacy-001", results[0].record_id);
    }

    // Soft delete the legacy row — works on the migrated column.
    try std.testing.expect(try store.deleteL1("legacy-001", .{}, iso));
    {
        const after = try store.searchL1Fts(allocator, "PostgreSQL", 5, iso);
        defer {
            for (after) |r| r.deinit(allocator);
            allocator.free(after);
        }
        try std.testing.expectEqual(@as(usize, 0), after.len);
    }

    // Restore works after migration.
    try std.testing.expect(try store.restoreL1("legacy-001", iso));
    {
        const revived = try store.searchL1Fts(allocator, "PostgreSQL", 5, iso);
        defer {
            for (revived) |r| r.deinit(allocator);
            allocator.free(revived);
        }
        try std.testing.expectEqual(@as(usize, 1), revived.len);
    }

    // Hard delete still works (l1_ad trigger intact after migration).
    try std.testing.expect(try store.deleteL1("legacy-001", .{ .soft = false }, iso));
    {
        const gone = try store.searchL1Fts(allocator, "PostgreSQL", 5, iso);
        defer {
            for (gone) |r| r.deinit(allocator);
            allocator.free(gone);
        }
        try std.testing.expectEqual(@as(usize, 0), gone.len);
    }

    // Re-opening a migrated database is idempotent (no double-migration errors).
    store.deinit();
    var reopened = try sqlite_store.SqliteStore.init(allocator, io, tmp.db_path);
    defer reopened.deinit();
    const results2 = try reopened.searchL1Fts(allocator, "PostgreSQL", 5, iso);
    defer {
        for (results2) |r| r.deinit(allocator);
        allocator.free(results2);
    }
    try std.testing.expectEqual(@as(usize, 0), results2.len);
}

test "MemoryContext delete/restore/purge delegate through vtable" {
    var ctx = try TestCtx.init();
    defer ctx.deinit();

    var mem_ctx = agent_memory.MemoryContext{
        .store = ctx.store.toMemoryStore(),
        .iso = .{},
    };

    _ = try mem_ctx.save(ctx.allocator, "User prefers tabs over spaces", .persona, 70, "editor");

    // Find the saved record id via search.
    const results = try mem_ctx.search(ctx.allocator, "tabs", 5);
    defer {
        for (results) |r| r.deinit(ctx.allocator);
        ctx.allocator.free(results);
    }
    try std.testing.expectEqual(@as(usize, 1), results.len);
    const record_id = results[0].record_id;

    // Soft delete via MemoryContext.
    try std.testing.expect(try mem_ctx.delete(record_id));

    // Hidden from search.
    {
        const after = try mem_ctx.search(ctx.allocator, "tabs", 5);
        defer {
            for (after) |r| r.deinit(ctx.allocator);
            ctx.allocator.free(after);
        }
        try std.testing.expectEqual(@as(usize, 0), after.len);
    }

    // Restore via MemoryContext.
    try std.testing.expect(try mem_ctx.restore(record_id));

    // Hard delete via MemoryContext.
    try std.testing.expect(try mem_ctx.deleteHard(record_id));
    try std.testing.expect(!(try mem_ctx.restore(record_id)));

    // Purge via MemoryContext (nothing soft-deleted → 0).
    try std.testing.expectEqual(@as(u32, 0), try mem_ctx.purgeDeleted());
}
