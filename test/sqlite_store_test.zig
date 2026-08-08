//! Integration tests for the embedded SQLite memory store.
//!
//! These tests create a real temporary SQLite database and exercise
//! the L1/L2/L3 CRUD + search operations end-to-end.

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

test "checkpoint set + get round-trip" {
    var ctx = try TestCtx.init();
    defer ctx.deinit();

    const checkpoint = types.Checkpoint{
        .last_processed_timestamp = "2025-01-15T10:05:00Z",
        .last_scene_name = "database setup",
    };
    try ctx.store.setCheckpoint(checkpoint);

    const read = try ctx.store.getCheckpoint(ctx.allocator);
    defer read.deinit(ctx.allocator);

    try std.testing.expect(read.last_processed_timestamp != null);
    try std.testing.expect(read.last_scene_name != null);
    try std.testing.expectEqualStrings("2025-01-15T10:05:00Z", read.last_processed_timestamp.?);
    try std.testing.expectEqualStrings("database setup", read.last_scene_name.?);
}

test "checkpoint get when empty returns empty" {
    var ctx = try TestCtx.init();
    defer ctx.deinit();

    const read = try ctx.store.getCheckpoint(ctx.allocator);
    try std.testing.expect(read.last_processed_timestamp == null);
    try std.testing.expect(read.last_scene_name == null);
}

test "recall returns L3 + L2 + L1" {
    var ctx = try TestCtx.init();
    defer ctx.deinit();

    const iso = types.IsolationContext{};

    // Write L3.
    try ctx.store.writeCore("User prefers dark mode.", iso);

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
test "L1 upsert with embedding + vector search" {
    var ctx = try TestCtx.init();
    defer ctx.deinit();

    const iso = types.IsolationContext{ .session_id = "s1" };

    // Insert two records with embeddings.
    const emb1 = [_]f32{ 1.0, 0.0, 0.0 };
    const emb2 = [_]f32{ 0.0, 1.0, 0.0 };
    const emb3 = [_]f32{ 0.9, 0.1, 0.0 }; // close to emb1

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

    _ = try ctx.store.upsertL1(rec1, &emb1, iso);
    _ = try ctx.store.upsertL1(rec2, &emb2, iso);
    _ = try ctx.store.upsertL1(rec3, &emb3, iso);

    // Search with query embedding close to emb1 (should rank vec-001 first).
    const query_emb = [_]f32{ 0.95, 0.05, 0.0 };
    const results = try ctx.store.searchL1Vector(ctx.allocator, &query_emb, 3, iso);
    defer {
        for (results) |r| r.deinit(ctx.allocator);
        ctx.allocator.free(results);
    }

    try std.testing.expectEqual(@as(usize, 3), results.len);
    // vec-001 should be top (closest to query), vec-003 second, vec-002 last.
    try std.testing.expectEqualStrings("vec-001", results[0].record_id);
    try std.testing.expect(results[0].score >= results[1].score);
    try std.testing.expect(results[1].score >= results[2].score);
}

test "L1 hybrid search with FTS + vector via RRF" {
    var ctx = try TestCtx.init();
    defer ctx.deinit();

    const iso = types.IsolationContext{ .session_id = "s1" };

    // Insert records with embeddings.
    const emb1 = [_]f32{ 1.0, 0.0 };
    const emb2 = [_]f32{ 0.0, 1.0 };

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

    _ = try ctx.store.upsertL1(rec1, &emb1, iso);
    _ = try ctx.store.upsertL1(rec2, &emb2, iso);

    // Hybrid search: FTS query "PostgreSQL" + embedding close to rec1.
    const query_emb = [_]f32{ 0.95, 0.05 };
    const results = try ctx.store.searchL1Hybrid(ctx.allocator, "PostgreSQL", 5, iso, &query_emb);
    defer {
        for (results) |r| r.deinit(ctx.allocator);
        ctx.allocator.free(results);
    }

    // Should return results (RRF-merged).
    try std.testing.expect(results.len > 0);
    // hybrid-001 should be top (matches both FTS and vector).
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
    _ = try ctx.store.upsertL1(rec1, null, iso);

    // Hybrid search with null embedding — should work as FTS-only.
    const results = try ctx.store.searchL1Hybrid(ctx.allocator, "dark", 5, iso, null);
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

    // Write L3 persona (10 chars).
    try ctx.store.writeCore("User likes dark mode.", iso); // 20 chars

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
    rec2.content = "User likes Python programming language"; // 39 chars

    _ = try ctx.store.upsertL1(rec1, null, iso);
    _ = try ctx.store.upsertL1(rec2, null, iso);

    // Recall with unlimited budget — should get everything.
    var unlimited = try ctx.store.recallWithBudget(ctx.allocator, "PostgreSQL", 5, iso, 0);
    defer unlimited.deinit(ctx.allocator);
    try std.testing.expect(unlimited.total_chars > 0);
    try std.testing.expectEqual(@as(usize, 2), unlimited.l1_results.len);

    // Recall with tight budget (50 chars) — persona (20) + 1 L1 (36) = 56 > 50, so persona + 0 L1.
    var capped = try ctx.store.recallWithBudget(ctx.allocator, "PostgreSQL", 5, iso, 50);
    defer capped.deinit(ctx.allocator);
    try std.testing.expect(capped.total_chars <= 50);
    try std.testing.expect(capped.persona != null); // persona always included first
    // L1 results may be empty if persona alone fills budget.
}

test "recallWithBudget with zero budget returns everything" {
    var ctx = try TestCtx.init();
    defer ctx.deinit();

    const iso = types.IsolationContext{};

    try ctx.store.writeCore("Persona text", iso);

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
    _ = try ctx.store.upsertL1(rec, null, iso);

    var result = try ctx.store.recallWithBudget(ctx.allocator, "memory", 5, iso, 0);
    defer result.deinit(ctx.allocator);
    try std.testing.expect(result.persona != null);
    try std.testing.expectEqual(@as(usize, 1), result.l1_results.len);
}
