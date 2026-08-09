//! SQLite-based embedded memory store.
//!
//! Implements the `MemoryStore` vtable using SQLite + FTS5 for L1
//! storage.
//!
//! Design (ported from TencentDB Agent Memory's `sqlite.ts`):
//! - WAL mode for concurrent read performance.
//! - FTS5 for full-text keyword search (BM25-ranked).
//! - L1 table + FTS5 virtual table with triggers to keep them in sync.
//! - Optional vector embeddings (brute-force cosine — see vector.zig).
//! - Pipeline checkpoint in a SQLite table.

const std = @import("std");
const sqlite = @import("sqlite.zig");
const types = @import("../types.zig");
const rrf = @import("rrf.zig");
const store = @import("../store.zig");
const context = @import("../context.zig");

const TAG = "[agent-memory]";

// ============================
// Store
// ============================

pub const SqliteStore = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    db: sqlite.Db,
    capabilities: types.StoreCapabilities,
    degraded: bool = false,

    /// Open (or create) a memory store at `db_path`.
    pub fn init(allocator: std.mem.Allocator, io: std.Io, db_path: [:0]const u8) !SqliteStore {
        var db = try sqlite.Db.open(db_path);
        errdefer db.close();

        // PRAGMAs — same as the TS implementation.
        try db.exec("PRAGMA journal_mode = WAL");
        try db.exec("PRAGMA busy_timeout = 5000");
        try db.exec("PRAGMA cache_size = -65536"); // 64 MB
        try db.exec("PRAGMA foreign_keys = ON");

        // Detect FTS5 support.
        var caps = types.StoreCapabilities{ .fts_search = false };
        if (detectFts5(&db)) {
            caps.fts_search = true;
        } else |_| {
            // FTS5 not available — degrade to no search.
        }

        // Create schema.
        try createSchema(&db, caps.fts_search);

        return .{
            .allocator = allocator,
            .io = io,
            .db = db,
            .capabilities = caps,
        };
    }

    pub fn deinit(self: *SqliteStore) void {
        self.db.close();
    }

    /// Wrap this store in a `MemoryStore` vtable for use with `MemoryContext`.
    /// The caller is responsible for keeping `self` alive for the lifetime
    /// of the returned `MemoryStore`.
    pub fn toMemoryStore(self: *SqliteStore) store.MemoryStore {
        return .{ .ctx = @ptrCast(self), .vtable = &context.SqliteStoreVTable };
    }

    // ============================
    // L1 — Structured Memories
    // ============================

    /// Upsert an L1 record. If `embedding` is provided, it's stored in l1_embeddings.
    pub fn upsertL1(
        self: *SqliteStore,
        record: types.L1Record,
        iso: types.IsolationContext,
    ) !bool {
        _ = iso; // isolation fields are already in the record
        try self.db.exec("BEGIN IMMEDIATE");
        errdefer self.db.exec("ROLLBACK") catch {};

        // Delete existing (upsert = delete + insert for FTS sync).
        const del_sql = "DELETE FROM l1_records WHERE record_id = ?";
        var del_stmt = try self.db.prepare(del_sql);
        defer del_stmt.finalize();
        try del_stmt.bindText(1, record.record_id);
        _ = try del_stmt.step();

        // Insert new.
        const sql =
            "INSERT INTO l1_records " ++
            "(record_id, content, type, priority, scene_name, session_key, session_id, " ++
            "team_id, task_id, user_id, agent_id, version, " ++
            "timestamp_str, timestamp_start, timestamp_end, created_time, updated_time, metadata_json) " ++
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";

        var stmt = try self.db.prepare(sql);
        defer stmt.finalize();

        try stmt.bindText(1, record.record_id);
        try stmt.bindText(2, record.content);
        try stmt.bindText(3, record.type.toString());
        try stmt.bindFloat(4, @floatCast(record.priority));
        try stmt.bindText(5, record.scene_name);
        try stmt.bindText(6, record.session_key);
        try stmt.bindText(7, record.session_id);
        try stmt.bindText(8, record.team_id);
        try stmt.bindText(9, record.task_id);
        try stmt.bindText(10, record.user_id);
        try stmt.bindText(11, record.agent_id);
        try stmt.bindInt(12, @intCast(record.version));
        try stmt.bindText(13, record.timestamp_str);
        try stmt.bindText(14, record.timestamp_start);
        try stmt.bindText(15, record.timestamp_end);
        try stmt.bindText(16, record.created_time);
        try stmt.bindText(17, record.updated_time);
        try stmt.bindText(18, record.metadata_json);

        _ = try stmt.step();

        try self.db.exec("COMMIT");
        return true;
    }

    /// FTS5 keyword search on L1 records.
    pub fn searchL1Fts(
        self: *SqliteStore,
        allocator: std.mem.Allocator,
        query: []const u8,
        top_k: u32,
        iso: types.IsolationContext,
    ) ![]types.SearchResult {
        if (!self.capabilities.fts_search) return &.{};

        const fts_query = try buildFtsQuery(allocator, query);
        defer allocator.free(fts_query);

        const sql =
            "SELECT l1.record_id, l1.content, l1.type, l1.priority, l1.scene_name, " ++
            "bm25(l1_fts) AS score, l1.session_id, l1.team_id, l1.user_id, l1.agent_id " ++
            "FROM l1_fts JOIN l1_records l1 ON l1_fts.rowid = l1.rowid " ++
            "WHERE l1_fts MATCH ? AND l1.team_id = ? AND l1.agent_id = ? AND l1.user_id = ? " ++
            "ORDER BY score LIMIT ?";

        var stmt = try self.db.prepare(sql);
        defer stmt.finalize();

        try stmt.bindText(1, fts_query);
        try stmt.bindText(2, iso.team_id);
        try stmt.bindText(3, iso.agent_id);
        try stmt.bindText(4, iso.user_id);
        try stmt.bindInt(5, @intCast(top_k));

        var results: std.ArrayList(types.SearchResult) = .empty;
        defer results.deinit(allocator);
        errdefer {
            for (results.items) |r| r.deinit(allocator);
        }

        while (try stmt.step()) {
            const mem_type = types.MemoryType.fromString(stmt.columnText(2)) orelse .episodic;
            const result = types.SearchResult{
                .record_id = try allocator.dupe(u8, stmt.columnText(0)),
                .content = try allocator.dupe(u8, stmt.columnText(1)),
                .type = mem_type,
                .priority = @floatCast(stmt.columnFloat(3)),
                .scene_name = try allocator.dupe(u8, stmt.columnText(4)),
                .score = @floatCast(stmt.columnFloat(5)),
                .session_id = try allocator.dupe(u8, stmt.columnText(6)),
                .team_id = try allocator.dupe(u8, stmt.columnText(7)),
                .user_id = try allocator.dupe(u8, stmt.columnText(8)),
                .agent_id = try allocator.dupe(u8, stmt.columnText(9)),
            };
            try results.append(allocator, result);
        }

        return try results.toOwnedSlice(allocator);
    }

    /// Hybrid search: FTS + optional vector, merged with RRF.
    /// When `query_embedding` is provided, FTS and vector results are
    /// fused via Reciprocal Rank Fusion. Otherwise, FTS-only.
    pub fn searchL1Hybrid(
        self: *SqliteStore,
        allocator: std.mem.Allocator,
        query: []const u8,
        top_k: u32,
        iso: types.IsolationContext,
    ) ![]types.SearchResult {
        if (!self.capabilities.fts_search) return &.{};

        // Phase 1: FTS5 search.
        const fts_results = try self.searchL1Fts(allocator, query, top_k, iso);

        return fts_results;
    }

    // ============================
    // Recall
    // ============================

    /// The main entry point for prompt injection.
    /// Searches L1 (hybrid) and returns a bounded RecallResult.
    /// No character budget capping — use recallWithBudget for that.
    pub fn recall(
        self: *SqliteStore,
        allocator: std.mem.Allocator,
        query: []const u8,
        top_k: u32,
        iso: types.IsolationContext,
    ) !types.RecallResult {
        return self.recallWithBudget(allocator, query, top_k, iso, 0);
    }

    /// Recall with character budget capping. When `max_chars` > 0, the result
    /// is truncated to fit within the budget. L1 results are included most
    /// relevant first; items that don't fit are dropped entirely (not
    /// partially truncated). Use `max_chars = 0` for unlimited (same as recall()).
    pub fn recallWithBudget(
        self: *SqliteStore,
        allocator: std.mem.Allocator,
        query: []const u8,
        top_k: u32,
        iso: types.IsolationContext,
        max_chars: usize,
    ) !types.RecallResult {
        // L1 hybrid search.
        const all_l1 = try self.searchL1Hybrid(allocator, query, top_k, iso);

        // If no budget, return everything.
        if (max_chars == 0) {
            var total_chars: usize = 0;
            for (all_l1) |r| total_chars += r.content.len;
            return .{
                .l1_results = all_l1,
                .total_chars = total_chars,
            };
        }

        // Budget capping: L1 results (most relevant first).
        var remaining = max_chars;
        var total_chars: usize = 0;

        var capped_l1: []types.SearchResult = all_l1;
        var l1_count: usize = 0;
        for (all_l1) |r| {
            if (r.content.len <= remaining) {
                remaining -= r.content.len;
                total_chars += r.content.len;
                l1_count += 1;
            } else {
                break; // stop at first that doesn't fit
            }
        }
        if (l1_count < all_l1.len) {
            // Free the ones that didn't fit.
            for (all_l1[l1_count..]) |r| r.deinit(allocator);
            // Shrink the slice — need to realloc.
            capped_l1 = try allocator.realloc(all_l1, l1_count);
        }

        return .{
            .l1_results = capped_l1,
            .total_chars = total_chars,
        };
    }

    // ============================
    // Schema Creation
    // ============================

    fn createSchema(db: *sqlite.Db, fts_available: bool) !void {
        // L1 table.
        try db.exec(
            \\CREATE TABLE IF NOT EXISTS l1_records (
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
        try db.exec("CREATE INDEX IF NOT EXISTS idx_l1_session ON l1_records(session_id)");
        try db.exec("CREATE INDEX IF NOT EXISTS idx_l1_type ON l1_records(type)");

        // FTS5 tables + triggers (only if FTS5 is available).
        if (fts_available) {
            try db.exec(
                \\CREATE VIRTUAL TABLE IF NOT EXISTS l1_fts USING fts5(
                \\  content,
                \\  content='l1_records',
                \\  content_rowid='rowid'
                \\)
            );

            // Triggers to keep FTS in sync.
            try db.exec(
                \\CREATE TRIGGER IF NOT EXISTS l1_ai AFTER INSERT ON l1_records BEGIN
                \\  INSERT INTO l1_fts(rowid, content) VALUES (new.rowid, new.content);
                \\END
            );
            try db.exec(
                \\CREATE TRIGGER IF NOT EXISTS l1_ad AFTER DELETE ON l1_records BEGIN
                \\  INSERT INTO l1_fts(l1_fts, rowid, content) VALUES('delete', old.rowid, old.content);
                \\END
            );
            try db.exec(
                \\CREATE TRIGGER IF NOT EXISTS l1_au AFTER UPDATE ON l1_records BEGIN
                \\  INSERT INTO l1_fts(l1_fts, rowid, content) VALUES('delete', old.rowid, old.content);
                \\  INSERT INTO l1_fts(rowid, content) VALUES (new.rowid, new.content);
                \\END
            );
        }
    }

    fn detectFts5(db: *sqlite.Db) !void {
        // Try creating a throwaway FTS5 table to detect support.
        try db.exec("CREATE VIRTUAL TABLE IF NOT EXISTS _fts5_detect USING fts5(x)");
        try db.exec("DROP TABLE IF EXISTS _fts5_detect");
    }
};

// ============================
// Helpers
// ============================

/// Build a safe FTS5 query string from user input.
/// Wraps each token in double quotes to prevent FTS5 syntax injection.
fn buildFtsQuery(allocator: std.mem.Allocator, query: []const u8) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);

    // Split on whitespace, quote each token.
    var it = std.mem.tokenizeAny(u8, query, " \t\n\r");
    var first = true;
    while (it.next()) |token| {
        if (!first) try buf.append(allocator, ' ');
        first = false;
        try buf.append(allocator, '"');
        // Escape internal double-quotes by doubling them (FTS5 convention).
        for (token) |ch| {
            if (ch == '"') {
                try buf.append(allocator, '"');
                try buf.append(allocator, '"');
            } else {
                try buf.append(allocator, ch);
            }
        }
        try buf.append(allocator, '"');
    }
    if (first) {
        // Empty query — match everything.
        try buf.appendSlice(allocator, "\"\"");
    }
    return try buf.toOwnedSlice(allocator);
}

/// Parse a checkpoint JSON value: {"timestamp": "...", "scene_name": "..."}
fn parseCheckpoint(allocator: std.mem.Allocator, json: []const u8) !types.Checkpoint {
    if (json.len == 0) return .{};

    var parsed = std.json.parseFromSlice(std.json.Value, allocator, json, .{}) catch return .{};
    defer parsed.deinit();

    if (parsed.value != .object) return .{};
    const obj = parsed.value.object;

    var checkpoint: types.Checkpoint = .{};

    if (obj.get("timestamp")) |ts| {
        if (ts == .string) {
            checkpoint.last_processed_timestamp = try allocator.dupe(u8, ts.string);
        }
    }
    if (obj.get("scene_name")) |sn| {
        if (sn == .string) {
            checkpoint.last_scene_name = try allocator.dupe(u8, sn.string);
        }
    }
    return checkpoint;
}

/// Serialize a checkpoint to JSON.
fn serializeCheckpoint(allocator: std.mem.Allocator, checkpoint: types.Checkpoint) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);
    try buf.append(allocator, '{');
    try buf.appendSlice(allocator, "\"timestamp\":");
    if (checkpoint.last_processed_timestamp) |ts| {
        const s = try std.fmt.allocPrint(allocator, "\"{s}\"", .{ts});
        defer allocator.free(s);
        try buf.appendSlice(allocator, s);
    } else {
        try buf.appendSlice(allocator, "null");
    }
    try buf.appendSlice(allocator, ",\"scene_name\":");
    if (checkpoint.last_scene_name) |sn| {
        const s = try std.fmt.allocPrint(allocator, "\"{s}\"", .{sn});
        defer allocator.free(s);
        try buf.appendSlice(allocator, s);
    } else {
        try buf.appendSlice(allocator, "null");
    }
    try buf.append(allocator, '}');
    return try buf.toOwnedSlice(allocator);
}
