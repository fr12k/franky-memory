//! MemoryContext — bundles a memory store with an isolation context.
//!
//! This is the struct that franky (or any LLM harness) passes as the `ctx`
//! pointer to memory tools (memory_search, memory_save, etc.). It has zero
//! dependencies on any LLM harness — it's pure agent-memory-zig.
//!
//! Usage:
//!   var store = try agent_memory.SqliteStore.init(allocator, io, db_path, data_dir);
//!   defer store.deinit();
//!   var mem_ctx = agent_memory.MemoryContext{
//!       .store = store.toMemoryStore(),
//!       .iso = .{ .session_id = "s1" },
//!   };
//!   // Now pass &mem_ctx as the tool's ctx pointer.

const std = @import("std");
const types = @import("types.zig");
const store_mod = @import("store.zig");
const sqlite_store = @import("embedded/sqlite_store.zig");

/// Bundles a `MemoryStore` (vtable dispatch) with an `IsolationContext`
/// so tools have everything they need in one pointer.
pub const MemoryContext = struct {
    store: store_mod.MemoryStore,
    iso: types.IsolationContext,

    /// Search L1 memories by keyword (FTS5 BM25).
    /// Returns owned SearchResult slice — caller frees each entry + the slice.
    pub fn search(self: *MemoryContext, allocator: std.mem.Allocator, query: []const u8, top_k: u32) ![]types.SearchResult {
        return self.store.searchL1(allocator, query, top_k, self.iso);
    }

    /// Save a memory to L1. The agent calls this when it decides something
    /// is worth remembering. No extraction LLM — the agent IS the extractor.
    ///
    /// The `session_id` is derived from `self.iso.session_id`. If the
    /// isolation context has no session_id, `"default"` is used.
    ///
    /// **Contract**: `store.upsertL1` MUST deep-copy all string fields in
    /// the `L1Record` (the SqliteStore does this via `sqlite3_bind_text`
    /// with `SQLITE_TRANSIENT`). The record's string fields are freed
    /// before this function returns; the store must not retain pointers.
    pub fn save(
        self: *MemoryContext,
        allocator: std.mem.Allocator,
        content: []const u8,
        mem_type: types.MemoryType,
        priority: f32,
        scene_name: []const u8,
    ) !bool {
        const sid = self.iso.session_id orelse "default";

        // Allocate each string field separately so the store can copy them
        // independently. A single shared buffer would be freed before the
        // store finishes if it retained pointers.
        const record_id = try generateId(allocator);
        defer allocator.free(record_id);

        const ts_str = try nowMillisStr(allocator);
        defer allocator.free(ts_str);

        const ts_start = try allocator.dupe(u8, ts_str);
        defer allocator.free(ts_start);
        const ts_end = try allocator.dupe(u8, ts_str);
        defer allocator.free(ts_end);
        const created = try allocator.dupe(u8, ts_str);
        defer allocator.free(created);
        const updated = try allocator.dupe(u8, ts_str);
        defer allocator.free(updated);

        const record = types.L1Record{
            .record_id = record_id,
            .content = content,
            .type = mem_type,
            .priority = priority,
            .scene_name = scene_name,
            .session_key = sid,
            .session_id = sid,
            .team_id = self.iso.team_id,
            .task_id = self.iso.task_id orelse "",
            .user_id = self.iso.user_id,
            .agent_id = self.iso.agent_id,
            .version = 1,
            .timestamp_str = ts_str,
            .timestamp_start = ts_start,
            .timestamp_end = ts_end,
            .created_time = created,
            .updated_time = updated,
            .metadata_json = "{}",
        };

        return self.store.upsertL1(record, null, self.iso);
    }

    /// Recall — the main entry point for system prompt injection.
    /// Searches L1 + reads L2/L3 and returns a bounded RecallResult.
    pub fn recall(self: *MemoryContext, allocator: std.mem.Allocator, query: []const u8, top_k: u32) !types.RecallResult {
        return self.store.recall(allocator, query, top_k, self.iso);
    }

    /// Recall with a maximum character budget for prompt injection.
    /// A budget of zero means unlimited.
    pub fn recallWithBudget(self: *MemoryContext, allocator: std.mem.Allocator, query: []const u8, top_k: u32, max_chars: usize) !types.RecallResult {
        return self.store.recallWithBudget(allocator, query, top_k, self.iso, max_chars);
    }
};

// ============================
// Vtable adapter for SqliteStore
// ============================

/// The vtable implementation for `SqliteStore`. These functions cast the
/// `*anyopaque` back to `*SqliteStore` and delegate.
pub const SqliteStoreVTable = store_mod.MemoryStore.VTable{
    .deinit = vtableDeinit,
    .capabilities = vtableCapabilities,
    .upsert_l1 = vtableUpsertL1,
    .search_l1 = vtableSearchL1,
    .read_core = vtableReadCore,
    .write_core = vtableWriteCore,
    .read_scenario = vtableReadScenario,
    .write_scenario = vtableWriteScenario,
    .list_scenarios = vtableListScenarios,
    .recall = vtableRecall,
    .recall_with_budget = vtableRecallWithBudget,
    .get_checkpoint = vtableGetCheckpoint,
    .set_checkpoint = vtableSetCheckpoint,
};

fn vtableDeinit(ctx: *anyopaque) void {
    const self: *sqlite_store.SqliteStore = @ptrCast(@alignCast(ctx));
    self.deinit();
}

fn vtableCapabilities(ctx: *anyopaque) types.StoreCapabilities {
    const self: *sqlite_store.SqliteStore = @ptrCast(@alignCast(ctx));
    return self.capabilities;
}

fn vtableUpsertL1(ctx: *anyopaque, record: types.L1Record, embedding: ?[]const f32, iso: types.IsolationContext) !bool {
    const self: *sqlite_store.SqliteStore = @ptrCast(@alignCast(ctx));
    return self.upsertL1(record, embedding, iso);
}

fn vtableSearchL1(ctx: *anyopaque, allocator: std.mem.Allocator, query: []const u8, top_k: u32, iso: types.IsolationContext) ![]types.SearchResult {
    const self: *sqlite_store.SqliteStore = @ptrCast(@alignCast(ctx));
    return self.searchL1Hybrid(allocator, query, top_k, iso, null);
}

fn vtableReadCore(ctx: *anyopaque, allocator: std.mem.Allocator, iso: types.IsolationContext) !?[]u8 {
    const self: *sqlite_store.SqliteStore = @ptrCast(@alignCast(ctx));
    return self.readCore(allocator, iso);
}

fn vtableWriteCore(ctx: *anyopaque, content: []const u8, iso: types.IsolationContext) !void {
    const self: *sqlite_store.SqliteStore = @ptrCast(@alignCast(ctx));
    return self.writeCore(content, iso);
}

fn vtableReadScenario(ctx: *anyopaque, allocator: std.mem.Allocator, path: []const u8, iso: types.IsolationContext) !?types.ScenarioFile {
    const self: *sqlite_store.SqliteStore = @ptrCast(@alignCast(ctx));
    return self.readScenario(allocator, path, iso);
}

fn vtableWriteScenario(ctx: *anyopaque, path: []const u8, content: []const u8, iso: types.IsolationContext) !void {
    const self: *sqlite_store.SqliteStore = @ptrCast(@alignCast(ctx));
    return self.writeScenario(path, content, iso);
}

fn vtableListScenarios(ctx: *anyopaque, allocator: std.mem.Allocator, path_prefix: ?[]const u8, iso: types.IsolationContext) ![]types.ScenarioFile {
    const self: *sqlite_store.SqliteStore = @ptrCast(@alignCast(ctx));
    return self.listScenarios(allocator, path_prefix, iso);
}

fn vtableRecall(ctx: *anyopaque, allocator: std.mem.Allocator, query: []const u8, top_k: u32, iso: types.IsolationContext) !types.RecallResult {
    const self: *sqlite_store.SqliteStore = @ptrCast(@alignCast(ctx));
    return self.recall(allocator, query, top_k, iso);
}

fn vtableRecallWithBudget(ctx: *anyopaque, allocator: std.mem.Allocator, query: []const u8, top_k: u32, iso: types.IsolationContext, max_chars: usize) !types.RecallResult {
    const self: *sqlite_store.SqliteStore = @ptrCast(@alignCast(ctx));
    return self.recallWithBudget(allocator, query, top_k, iso, max_chars);
}

fn vtableGetCheckpoint(ctx: *anyopaque, allocator: std.mem.Allocator) !types.Checkpoint {
    const self: *sqlite_store.SqliteStore = @ptrCast(@alignCast(ctx));
    return self.getCheckpoint(allocator);
}

fn vtableSetCheckpoint(ctx: *anyopaque, checkpoint: types.Checkpoint) !void {
    const self: *sqlite_store.SqliteStore = @ptrCast(@alignCast(ctx));
    return self.setCheckpoint(checkpoint);
}

// ============================
// Helpers
// ============================

/// Generate a unique memory record ID: "mem-<timestamp>-<counter>"
///
/// The counter type is selected at comptime: `u64` on 64-bit targets and
/// `u32` on 32-bit targets. 32-bit x86 with the baseline CPU (no cmpxchg8b)
/// cannot perform 64-bit atomic read-modify-write operations, so
/// `@atomicRmw` (used by `fetchAdd`) rejects `u64` there. A `u32` counter
/// is ample for a per-process monotonic counter that is only combined
/// with a millisecond timestamp to disambiguate IDs within the same ms.
const CounterType = if (@sizeOf(usize) <= 4) u32 else u64;
var id_counter: std.atomic.Value(CounterType) = .init(0);

fn generateId(allocator: std.mem.Allocator) ![]u8 {
    const c = id_counter.fetchAdd(1, .monotonic);
    return std.fmt.allocPrint(allocator, "mem-{d}-{d}", .{ nowMillis(), c });
}

/// Current time as a millisecond timestamp (epoch).
/// Uses clock_gettime on Linux, falls back to 0 on failure or unsupported
/// platforms. Matches franky's `ai.stream.nowMillis` signature exactly.
fn nowMillis() i64 {
    const builtin = @import("builtin");
    if (builtin.os.tag == .linux) {
        const linux = std.os.linux;
        var ts: linux.timespec = undefined;
        if (linux.clock_gettime(.REALTIME, &ts) == 0) {
            return @as(i64, ts.sec) * 1000 + @divFloor(@as(i64, ts.nsec), std.time.ns_per_ms);
        }
        return 0;
    }
    // POSIX (Darwin / BSD / other) via libc.
    if (builtin.link_libc) {
        var ts: std.c.timespec = undefined;
        if (std.c.clock_gettime(.REALTIME, &ts) == 0) {
            return @as(i64, ts.sec) * 1000 + @divFloor(@as(i64, ts.nsec), std.time.ns_per_ms);
        }
    }
    return 0;
}

/// Current time as a millisecond epoch string (NOT ISO 8601).
/// Used for timestamp fields in L1Record. Returns an owned string.
fn nowMillisStr(allocator: std.mem.Allocator) ![]u8 {
    return std.fmt.allocPrint(allocator, "{d}", .{nowMillis()});
}

// ============================
// Tests
// ============================

test "MemoryContext search returns empty on fresh store" {
    const allocator = std.testing.allocator;
    var threaded = std.Io.Threaded.init(allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const tmp_dir = "/tmp/agent-memory-ctx-test";
    std.Io.Dir.cwd().createDirPath(io, tmp_dir) catch {};
    defer std.Io.Dir.cwd().deleteTree(io, tmp_dir) catch {};

    const db_path = try std.fmt.allocPrintSentinel(allocator, "{s}/memory.db", .{tmp_dir}, 0);
    defer allocator.free(db_path);

    var store = try sqlite_store.SqliteStore.init(allocator, io, db_path, tmp_dir);
    defer store.deinit();

    var mem_ctx = MemoryContext{
        .store = .{ .ctx = @ptrCast(&store), .vtable = &SqliteStoreVTable },
        .iso = .{},
    };

    const results = try mem_ctx.search(allocator, "anything", 5);
    defer {
        for (results) |r| r.deinit(allocator);
        allocator.free(results);
    }
    try std.testing.expectEqual(@as(usize, 0), results.len);
}

test "MemoryContext save + search round-trip" {
    const allocator = std.testing.allocator;
    var threaded = std.Io.Threaded.init(allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const tmp_dir = "/tmp/agent-memory-ctx-save-test";
    std.Io.Dir.cwd().createDirPath(io, tmp_dir) catch {};
    defer std.Io.Dir.cwd().deleteTree(io, tmp_dir) catch {};

    const db_path = try std.fmt.allocPrintSentinel(allocator, "{s}/memory.db", .{tmp_dir}, 0);
    defer allocator.free(db_path);

    var store = try sqlite_store.SqliteStore.init(allocator, io, db_path, tmp_dir);
    defer store.deinit();

    var mem_ctx = MemoryContext{
        .store = .{ .ctx = @ptrCast(&store), .vtable = &SqliteStoreVTable },
        .iso = .{ .session_id = "test-session" },
    };

    // Save a memory.
    _ = try mem_ctx.save(allocator, "User prefers PostgreSQL over MySQL", .persona, 80, "database preferences");

    // Search for it.
    const results = try mem_ctx.search(allocator, "PostgreSQL", 5);
    defer {
        for (results) |r| r.deinit(allocator);
        allocator.free(results);
    }
    try std.testing.expectEqual(@as(usize, 1), results.len);
    try std.testing.expectEqualStrings("User prefers PostgreSQL over MySQL", results[0].content);
    try std.testing.expectEqual(types.MemoryType.persona, results[0].type);
}

test "MemoryContext recallWithBudget delegates through vtable" {
    const allocator = std.testing.allocator;
    var threaded: std.Io.Threaded = .init_single_threaded;
    defer threaded.deinit();
    const io = threaded.io();

    const tmp_dir = "/tmp/agent-memory-ctx-budget-test";
    std.Io.Dir.cwd().createDirPath(io, tmp_dir) catch {};
    defer std.Io.Dir.cwd().deleteTree(io, tmp_dir) catch {};

    const db_path = try std.fmt.allocPrintSentinel(allocator, "{s}/memory.db", .{tmp_dir}, 0);
    defer allocator.free(db_path);

    var sqlite = try sqlite_store.SqliteStore.init(allocator, io, db_path, tmp_dir);
    defer sqlite.deinit();

    var mem_ctx = MemoryContext{
        .store = .{ .ctx = @ptrCast(&sqlite), .vtable = &SqliteStoreVTable },
        .iso = .{},
    };

    try sqlite.writeCore("short persona", .{});
    var result = try mem_ctx.recallWithBudget(allocator, "test query", 5, 20);
    defer result.deinit(allocator);

    try std.testing.expect(result.persona != null);
    try std.testing.expect(result.total_chars <= 20);
}

test "MemoryContext recall returns empty on fresh store" {
    const allocator = std.testing.allocator;
    var threaded = std.Io.Threaded.init(allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const tmp_dir = "/tmp/agent-memory-ctx-recall-test";
    std.Io.Dir.cwd().createDirPath(io, tmp_dir) catch {};
    defer std.Io.Dir.cwd().deleteTree(io, tmp_dir) catch {};

    const db_path = try std.fmt.allocPrintSentinel(allocator, "{s}/memory.db", .{tmp_dir}, 0);
    defer allocator.free(db_path);

    var store = try sqlite_store.SqliteStore.init(allocator, io, db_path, tmp_dir);
    defer store.deinit();

    var mem_ctx = MemoryContext{
        .store = .{ .ctx = @ptrCast(&store), .vtable = &SqliteStoreVTable },
        .iso = .{},
    };

    var result = try mem_ctx.recall(allocator, "test query", 5);
    defer result.deinit(allocator);

    // Fresh store → no persona, no scenarios, no L1 hits.
    try std.testing.expectEqual(@as(?[]const u8, null), result.persona);
    try std.testing.expectEqual(@as(usize, 0), result.scenario_files.len);
    try std.testing.expectEqual(@as(usize, 0), result.l1_results.len);
    try std.testing.expectEqual(@as(usize, 0), result.total_chars);
}