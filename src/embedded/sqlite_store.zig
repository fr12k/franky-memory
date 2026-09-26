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

        // Migrate existing databases (adds the `deleted` column and
        // normalizes the FTS update trigger when needed).
        try migrateSchema(&db, caps.fts_search);

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

    /// Delete an L1 record.
    ///
    /// With `DeleteOptions{ .soft = true }` (the default) the row is only
    /// marked `deleted = 1`: it disappears from search/recall but stays in
    /// the table and can be recovered with `restoreL1`. The FTS5 index entry
    /// is kept — soft-deleted rows are filtered at query time. Only a *live*
    /// row can be soft-deleted (deleting an already-deleted row returns false).
    ///
    /// With `.soft = false` the row is physically removed (irreversible),
    /// whether it is live or soft-deleted — a force-delete. The `l1_ad`
    /// trigger evicts the FTS5 entry.
    ///
    /// The isolation context (team/agent/user) is enforced, so a record of
    /// another tenant can never be deleted.
    ///
    /// Returns true when a row was affected; false when no matching record
    /// exists (unknown id, or isolation mismatch — and for soft delete,
    /// already deleted).
    pub fn deleteL1(
        self: *SqliteStore,
        record_id: []const u8,
        options: types.DeleteOptions,
        iso: types.IsolationContext,
    ) !bool {
        // The two paths differ only in the SQL statement.
        const sql = if (options.soft)
            "UPDATE l1_records SET deleted = 1 " ++
                "WHERE record_id = ? AND team_id = ? AND agent_id = ? AND user_id = ? " ++
                "AND deleted = 0"
        else
            "DELETE FROM l1_records " ++
                "WHERE record_id = ? AND team_id = ? AND agent_id = ? AND user_id = ?";
        return self.execRecordStatement(sql, record_id, iso);
    }

    /// Restore a soft-deleted L1 record (sets `deleted` back to 0).
    /// Returns true when a soft-deleted row was revived; false when the id
    /// is unknown, the row is live, or the isolation context does not match.
    pub fn restoreL1(
        self: *SqliteStore,
        record_id: []const u8,
        iso: types.IsolationContext,
    ) !bool {
        return self.execRecordStatement(
            "UPDATE l1_records SET deleted = 0 " ++
                "WHERE record_id = ? AND team_id = ? AND agent_id = ? AND user_id = ? " ++
                "AND deleted = 1",
            record_id,
            iso,
        );
    }

    /// Execute a single-row, isolation-scoped statement (binds: record_id,
    /// team_id, agent_id, user_id) and report whether a row was affected.
    /// Shared by `deleteL1` and `restoreL1`.
    fn execRecordStatement(
        self: *SqliteStore,
        sql: []const u8,
        record_id: []const u8,
        iso: types.IsolationContext,
    ) !bool {
        var stmt = try self.db.prepare(sql);
        defer stmt.finalize();
        try stmt.bindText(1, record_id);
        try stmt.bindText(2, iso.team_id);
        try stmt.bindText(3, iso.agent_id);
        try stmt.bindText(4, iso.user_id);
        _ = try stmt.step();
        return self.db.changes() > 0;
    }

    /// Physically remove all soft-deleted records within the isolation scope
    /// (team/agent/user). This is the garbage-collection pass — after it runs,
    /// the purged rows can no longer be restored.
    ///
    /// Returns the number of purged rows.
    pub fn purgeDeletedL1(self: *SqliteStore, iso: types.IsolationContext) !u32 {
        const sql =
            "DELETE FROM l1_records " ++
            "WHERE team_id = ? AND agent_id = ? AND user_id = ? AND deleted = 1";
        var stmt = try self.db.prepare(sql);
        defer stmt.finalize();
        try stmt.bindText(1, iso.team_id);
        try stmt.bindText(2, iso.agent_id);
        try stmt.bindText(3, iso.user_id);
        _ = try stmt.step();
        return @intCast(self.db.changes());
    }

    /// Enumerate live (non-deleted) L1 memories in scope, returning only the
    /// metadata projection (`scene_name`, `created_time`, `updated_time`,
    /// `metadata_json`). The `filter` narrows the result set: `session_id`,
    /// `type`, `time_start`/`time_end` (compared against `created_time`,
    /// lexicographically — both bounds optional), and `limit`/`offset`.
    ///
    /// `filter.limit = 0` means **all** matching records (the SQL LIMIT
    /// clause is omitted); the default `L1QueryFilter.limit` is 100.
    ///
    /// Returns an owned `[]MemorySummary` — the caller frees each entry's
    /// strings via `deinit` and then frees the slice.
    ///
    /// Ordering is `created_time DESC` (newest first), with `record_id` as a
    /// deterministic tie-breaker. This is a full scan filtered by the
    /// isolation scope — appropriate for enumerating a single tenant's
    /// memories, not for paginating millions of rows.
    pub fn listL1(
        self: *SqliteStore,
        allocator: std.mem.Allocator,
        filter: types.L1QueryFilter,
        iso: types.IsolationContext,
    ) ![]types.MemorySummary {
        // Build the WHERE clause. The isolation fragment is always present;
        // the filter fragments are appended conditionally. Bind parameters
        // are accumulated in order so the caller can bind them in sequence.
        var where: std.ArrayList(u8) = .empty;
        defer where.deinit(allocator);

        try where.appendSlice(allocator, "deleted = 0 AND team_id = ? AND agent_id = ? AND user_id = ?");
        var param_idx: c_int = 4; // 3 isolation params come first

        if (filter.session_id) |sid| {
            _ = sid;
            try where.appendSlice(allocator, " AND session_id = ?");
            param_idx += 1;
        }
        if (filter.type) |t| {
            _ = t;
            try where.appendSlice(allocator, " AND type = ?");
            param_idx += 1;
        }
        if (filter.time_start) |ts| {
            _ = ts;
            try where.appendSlice(allocator, " AND created_time >= ?");
            param_idx += 1;
        }
        if (filter.time_end) |te| {
            _ = te;
            try where.appendSlice(allocator, " AND created_time <= ?");
            param_idx += 1;
        }

        // Build the SQL. limit=0 means "all" → omit the LIMIT clause
        // entirely (SQLite LIMIT 0 returns zero rows, not all).
        var sql_buf: std.ArrayList(u8) = .empty;
        defer sql_buf.deinit(allocator);
        try sql_buf.appendSlice(allocator, "SELECT scene_name, created_time, updated_time, metadata_json FROM l1_records WHERE ");
        try sql_buf.appendSlice(allocator, where.items);
        try sql_buf.appendSlice(allocator, " ORDER BY created_time DESC, record_id DESC");
        if (filter.limit != 0) {
            try sql_buf.appendSlice(allocator, " LIMIT ? OFFSET ?");
        }
        const sql = try sql_buf.toOwnedSlice(allocator);
        defer allocator.free(sql);

        var stmt = try self.db.prepare(sql);
        defer stmt.finalize();

        // Bind parameters in the order they appear in the WHERE clause.
        var bind_idx: c_int = 1;
        try stmt.bindText(bind_idx, iso.team_id);
        bind_idx += 1;
        try stmt.bindText(bind_idx, iso.agent_id);
        bind_idx += 1;
        try stmt.bindText(bind_idx, iso.user_id);
        bind_idx += 1;
        if (filter.session_id) |sid| {
            try stmt.bindText(bind_idx, sid);
            bind_idx += 1;
        }
        if (filter.type) |t| {
            try stmt.bindText(bind_idx, t.toString());
            bind_idx += 1;
        }
        if (filter.time_start) |ts| {
            try stmt.bindText(bind_idx, ts);
            bind_idx += 1;
        }
        if (filter.time_end) |te| {
            try stmt.bindText(bind_idx, te);
            bind_idx += 1;
        }
        // LIMIT / OFFSET — only bound when limit != 0 (limit = 0 means
        // "all", so the LIMIT clause is omitted from the SQL entirely).
        if (filter.limit != 0) {
            try stmt.bindInt(bind_idx, @intCast(filter.limit));
            bind_idx += 1;
            try stmt.bindInt(bind_idx, @intCast(filter.offset));
        }

        var results: std.ArrayList(types.MemorySummary) = .empty;
        errdefer {
            for (results.items) |r| r.deinit(allocator);
            results.deinit(allocator);
        }

        while (try stmt.step()) {
            const summary = types.MemorySummary{
                .scene_name = try allocator.dupe(u8, stmt.columnText(0)),
                .created_time = try allocator.dupe(u8, stmt.columnText(1)),
                .updated_time = try allocator.dupe(u8, stmt.columnText(2)),
                .metadata_json = try allocator.dupe(u8, stmt.columnText(3)),
            };
            try results.append(allocator, summary);
        }

        return try results.toOwnedSlice(allocator);
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

        // Empty / whitespace-only query: return all live rows in scope (newest
        // first), up to top_k. FTS5 MATCH with an empty phrase does not match
        // everything — it matches nothing — so we short-circuit here and query
        // l1_records directly. This is the "what do I already know?" path.
        if (isBlankQuery(query)) {
            return self.searchL1All(allocator, top_k, iso);
        }

        const fts_query = try buildFtsQuery(allocator, query);
        defer allocator.free(fts_query);

        const sql =
            "SELECT l1.record_id, l1.content, l1.type, l1.priority, l1.scene_name, " ++
            "bm25(l1_fts) AS score, l1.session_id, l1.team_id, l1.user_id, l1.agent_id " ++
            "FROM l1_fts JOIN l1_records l1 ON l1_fts.rowid = l1.rowid " ++
            "WHERE l1_fts MATCH ? AND l1.team_id = ? AND l1.agent_id = ? AND l1.user_id = ? " ++
            "AND l1.deleted = 0 " ++
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

    /// Return all live L1 records in scope (newest first), up to `top_k`.
    /// Used as the fallback for empty / whitespace-only queries — FTS5 MATCH
    /// with an empty phrase matches nothing, so we bypass FTS and query
    /// l1_records directly. Score is set to 0 (no relevance ranking).
    fn searchL1All(
        self: *SqliteStore,
        allocator: std.mem.Allocator,
        top_k: u32,
        iso: types.IsolationContext,
    ) ![]types.SearchResult {
        const sql =
            "SELECT record_id, content, type, priority, scene_name, " ++
            "0.0 AS score, session_id, team_id, user_id, agent_id " ++
            "FROM l1_records " ++
            "WHERE team_id = ? AND agent_id = ? AND user_id = ? AND deleted = 0 " ++
            "ORDER BY created_time DESC, record_id DESC LIMIT ?";

        var stmt = try self.db.prepare(sql);
        defer stmt.finalize();

        try stmt.bindText(1, iso.team_id);
        try stmt.bindText(2, iso.agent_id);
        try stmt.bindText(3, iso.user_id);
        try stmt.bindInt(4, @intCast(top_k));

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
            \\  metadata_json TEXT NOT NULL DEFAULT '{}',
            \\  deleted INTEGER NOT NULL DEFAULT 0
            \\)
        );
        try db.exec("CREATE INDEX IF NOT EXISTS idx_l1_session ON l1_records(session_id)");
        try db.exec("CREATE INDEX IF NOT EXISTS idx_l1_type ON l1_records(type)");
        // NOTE: idx_l1_deleted is created in migrateSchema — it can only be
        // built once the `deleted` column is guaranteed to exist, which for
        // databases created before this column is only after the ALTER TABLE
        // in migrateSchema runs.

        // FTS5 tables + triggers (only if FTS5 is available).
        //
        // The FTS5 index covers BOTH `content` (the memory body) AND
        // `scene_name` (the scenario label). Indexing scene_name is essential
        // — without it, searching for a term that appears only in the scene
        // name (e.g. "htmx" when the scene is "franky htmx implementation")
        // returns zero results even though the memory exists. See the
        // false-negative bug report for the full catalogue of failures this
        // caused.
        if (fts_available) {
            try db.exec(
                \\CREATE VIRTUAL TABLE IF NOT EXISTS l1_fts USING fts5(
                \\  content, scene_name,
                \\  content='l1_records',
                \\  content_rowid='rowid'
                \\)
            );

            // Triggers to keep FTS in sync. Both content and scene_name are
            // mirrored into the FTS index on insert/delete.
            try db.exec(
                \\CREATE TRIGGER IF NOT EXISTS l1_ai AFTER INSERT ON l1_records BEGIN
                \\  INSERT INTO l1_fts(rowid, content, scene_name)
                \\  VALUES (new.rowid, new.content, new.scene_name);
                \\END
            );
            try db.exec(
                \\CREATE TRIGGER IF NOT EXISTS l1_ad AFTER DELETE ON l1_records BEGIN
                \\  INSERT INTO l1_fts(l1_fts, rowid, content, scene_name)
                \\  VALUES('delete', old.rowid, old.content, old.scene_name);
                \\END
            );
            // NOTE: `l1_au` (AFTER UPDATE OF content, scene_name) is created
            // in migrateSchema — the single creation site. Older databases
            // may carry an unguarded variant, so it is always normalized
            // there rather than IF NOT EXISTS'd here.
        }
    }

    /// Migrate an existing database to the current schema in place.
    /// Adds the `deleted` column to `l1_records` when missing (idempotent:
    /// a PRAGMA table_info probe runs first, so the ALTER runs at most once).
    /// Runs after createSchema, so the table is guaranteed to exist.
    fn migrateSchema(db: *sqlite.Db, fts_available: bool) !void {
        if (!try hasL1DeletedColumn(db)) {
            try db.exec("ALTER TABLE l1_records ADD COLUMN deleted INTEGER NOT NULL DEFAULT 0");
        }
        // Safe here for both fresh and old databases: the column exists in
        // either case (CREATE TABLE for fresh, ALTER TABLE just above for old).
        try db.exec("CREATE INDEX IF NOT EXISTS idx_l1_deleted ON l1_records(deleted)");

        // Normalize the FTS update trigger. This is the single creation
        // site for `l1_au`. Databases created before the soft-delete feature
        // carry an unguarded `l1_au` (AFTER UPDATE ON ...); CREATE TRIGGER IF
        // NOT EXISTS would leave it in place, causing needless FTS
        // evict/re-insert churn on every soft delete/restore. Drop and
        // recreate unconditionally — idempotent for databases that already
        // have the guarded form.
        //
        // The trigger fires on updates to EITHER content OR scene_name, since
        // both are indexed by the FTS table. A single UPDATE OF (content,
        // scene_name) clause covers both columns.
        if (fts_available) {
            // Migrate the FTS schema and normalize all triggers atomically.
            //
            // migrateFtsSchema detects legacy content-only FTS schemas and
            // rebuilds the index with both content + scene_name columns. It
            // also drops and recreates l1_ai and l1_ad. The l1_au trigger is
            // dropped/recreated here (not inside migrateFtsSchema) so that the
            // entire operation — FTS rebuild + all three triggers — is within
            // a single transaction. If any step fails, the database rolls back
            // to its pre-migration state.
            //
            // l1_au normalization: databases created before the soft-delete
            // feature carry an unguarded `l1_au` (AFTER UPDATE ON ...); CREATE
            // TRIGGER IF NOT EXISTS would leave it in place, causing needless
            // FTS evict/re-insert churn on every soft delete/restore. Drop and
            // recreate unconditionally — idempotent for databases that already
            // have the guarded form.
            //
            // The trigger fires on updates to EITHER content OR scene_name,
            // since both are indexed by the FTS table.
            try db.exec("BEGIN IMMEDIATE");
            errdefer db.exec("ROLLBACK") catch {};
            try migrateFtsSchema(db);
            try db.exec("DROP TRIGGER IF EXISTS l1_au");
            try db.exec(
                \\CREATE TRIGGER l1_au AFTER UPDATE OF content, scene_name ON l1_records BEGIN
                \\  INSERT INTO l1_fts(l1_fts, rowid, content, scene_name)
                \\  VALUES('delete', old.rowid, old.content, old.scene_name);
                \\  INSERT INTO l1_fts(rowid, content, scene_name)
                \\  VALUES (new.rowid, new.content, new.scene_name);
                \\END
            );
            try db.exec("COMMIT");
        }
    }

    /// Returns true when `l1_records` already carries a `deleted` column.
    fn hasL1DeletedColumn(db: *sqlite.Db) !bool {
        var stmt = try db.prepare("PRAGMA table_info(l1_records)");
        defer stmt.finalize();
        while (try stmt.step()) {
            if (std.mem.eql(u8, stmt.columnText(1), "deleted")) return true;
        }
        return false;
    }

    /// Migrate a legacy FTS5 schema to the current one (content + scene_name).
    ///
    /// Older databases have an `l1_fts` table with only a `content` column.
    /// This detects that by probing `pragma_table_info('l1_fts')` for a
    /// `scene_name` column. When the column is missing, the FTS index is
    /// dropped and rebuilt from `l1_records` with both columns.
    ///
    /// Idempotent: when the schema is already current, this is a no-op (the
    /// probe finds `scene_name` and returns early).
    fn migrateFtsSchema(db: *sqlite.Db) !void {
        if (try ftsHasSceneNameColumn(db)) return;

        // Drop the old FTS index and its triggers, then recreate with both
        // columns and backfill from l1_records.
        try db.exec("DROP TRIGGER IF EXISTS l1_ai");
        try db.exec("DROP TRIGGER IF EXISTS l1_ad");
        try db.exec("DROP TRIGGER IF EXISTS l1_au");
        try db.exec("DROP TABLE IF EXISTS l1_fts");

        try db.exec(
            \\CREATE VIRTUAL TABLE l1_fts USING fts5(
            \\  content, scene_name,
            \\  content='l1_records',
            \\  content_rowid='rowid'
            \\)
        );

        // Backfill: insert every live + soft-deleted row into the new FTS
        // index. Soft-deleted rows are included so that a subsequent restore
        // finds the FTS entry already present (matching the trigger behavior
        // where soft delete does NOT evict the FTS entry).
        try db.exec(
            \\INSERT INTO l1_fts(rowid, content, scene_name)
            \\  SELECT rowid, content, scene_name FROM l1_records
        );

        // Recreate the sync triggers (same as createSchema).
        try db.exec(
            \\CREATE TRIGGER l1_ai AFTER INSERT ON l1_records BEGIN
            \\  INSERT INTO l1_fts(rowid, content, scene_name)
            \\  VALUES (new.rowid, new.content, new.scene_name);
            \\END
        );
        try db.exec(
            \\CREATE TRIGGER l1_ad AFTER DELETE ON l1_records BEGIN
            \\  INSERT INTO l1_fts(l1_fts, rowid, content, scene_name)
            \\  VALUES('delete', old.rowid, old.content, old.scene_name);
            \\END
        );
        // l1_au is recreated by the caller (migrateSchema) after this returns.
    }

    /// Returns true when the `l1_fts` FTS5 table already has a `scene_name`
    /// column (i.e. the schema is current). Returns false when the table is
    /// absent or has only the legacy `content`-only schema.
    ///
    /// Uses `PRAGMA table_info(l1_fts)` — for FTS5 virtual tables this reports
    /// the user-defined columns (content, scene_name) as rows, with the column
    /// name in position 1 (same layout as for regular tables).
    fn ftsHasSceneNameColumn(db: *sqlite.Db) !bool {
        var stmt = try db.prepare("PRAGMA table_info(l1_fts)");
        defer stmt.finalize();
        while (try stmt.step()) {
            // Column 1 is the name (same layout as PRAGMA table_info).
            if (std.mem.eql(u8, stmt.columnText(1), "scene_name")) return true;
        }
        return false;
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

/// Returns true when `query` is empty or contains only whitespace characters.
/// Used to short-circuit FTS5 search — an empty FTS5 MATCH phrase matches
/// nothing, so we fall back to a full table scan (newest first) instead.
fn isBlankQuery(query: []const u8) bool {
    for (query) |ch| {
        switch (ch) {
            ' ', '\t', '\n', '\r' => {},
            else => return false,
        }
    }
    return true;
}

/// Build a safe FTS5 query string from user input.
///
/// Each whitespace-delimited token is wrapped in double quotes (to prevent
/// FTS5 syntax injection) and the tokens are joined with the FTS5 `OR`
/// operator. OR semantics are critical for memory search usability: a
/// multi-word natural-language query like "franky htmx implementation" should
/// return any memory that contains *any* of those terms, not silently return
/// zero results because no single memory happens to contain *all* of them.
/// The AND (default) join caused widespread false-negatives — see the
/// memory_search false-negative bug report.
///
/// Empty / whitespace-only queries are handled by the caller (`searchL1Fts`
/// short-circuits via `isBlankQuery` before calling this function). The
/// `if (first)` guard below is a defensive fallback that emits an empty
/// phrase if this function is ever called with a blank query.
fn buildFtsQuery(allocator: std.mem.Allocator, query: []const u8) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);

    // Split on whitespace, quote each token, join with OR.
    var it = std.mem.tokenizeAny(u8, query, " \t\n\r");
    var first = true;
    while (it.next()) |token| {
        if (!first) try buf.appendSlice(allocator, " OR ");
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
