//! SQLite-based embedded memory store.
//!
//! Implements the `MemoryStore` vtable using SQLite + FTS5 for L0/L1
//! storage and the filesystem for L2/L3 markdown files.
//!
//! Design (ported from TencentDB Agent Memory's `sqlite.ts`):
//! - WAL mode for concurrent read performance.
//! - FTS5 for full-text keyword search (BM25-ranked).
//! - L0/L1 tables + FTS5 virtual tables with triggers to keep them in sync.
//! - Optional vector embeddings (brute-force cosine — see vector.zig).
//! - L2/L3 stored as markdown files under `data_dir`.
//! - Pipeline checkpoint in a SQLite table.

const std = @import("std");
const sqlite = @import("sqlite.zig");
const types = @import("../types.zig");
const rrf = @import("rrf.zig");
const vector = @import("vector.zig");
const store = @import("../store.zig");
const context = @import("../context.zig");

const TAG = "[agent-memory]";

// ============================
// Tokenizer helpers (shared by scenario relevance ranking)
// ============================

/// Split text into lowercase alphanumeric tokens. Caller owns each slice.
fn tokenizeLower(allocator: std.mem.Allocator, text: []const u8, out: *std.ArrayList([]const u8)) !void {
    var start: ?usize = null;
    for (text, 0..) |c, i| {
        const is_alnum = std.ascii.isAlphanumeric(c);
        if (is_alnum and start == null) start = i;
        if (!is_alnum and start != null) {
            const lower = try std.ascii.allocLowerString(allocator, text[start.?..i]);
            try out.append(allocator, lower);
            start = null;
        }
    }
    if (start != null) {
        const lower = try std.ascii.allocLowerString(allocator, text[start.?..]);
        try out.append(allocator, lower);
    }
}

/// Score a scenario file against a query by token overlap.
/// Path tokens are weighted higher than content tokens (titles are more
/// discriminative than body text). Returns 0 if no overlap.
fn scenarioRelevance(query_tokens: []const []const u8, path: []const u8, content: []const u8, allocator: std.mem.Allocator) !f32 {
    if (query_tokens.len == 0) return 0;

    var path_toks: std.ArrayList([]const u8) = .empty;
    defer {
        for (path_toks.items) |t| allocator.free(t);
        path_toks.deinit(allocator);
    }
    try tokenizeLower(allocator, path, &path_toks);

    var content_toks: std.ArrayList([]const u8) = .empty;
    defer {
        for (content_toks.items) |t| allocator.free(t);
        content_toks.deinit(allocator);
    }
    try tokenizeLower(allocator, content, &content_toks);

    var score: f32 = 0;
    for (query_tokens) |qt| {
        // Path hits count double (title is more informative).
        for (path_toks.items) |pt| {
            if (std.mem.eql(u8, qt, pt)) {
                score += 2.0;
                break;
            }
        }
        for (content_toks.items) |ct| {
            if (std.mem.eql(u8, qt, ct)) {
                score += 1.0;
                break;
            }
        }
    }
    return score;
}

// ============================
// Store
// ============================

pub const SqliteStore = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    db: sqlite.Db,
    data_dir: []const u8,
    capabilities: types.StoreCapabilities,
    degraded: bool = false,

    /// Open (or create) a memory store at `db_path` with L2/L3 files under `data_dir`.
    pub fn init(allocator: std.mem.Allocator, io: std.Io, db_path: [:0]const u8, data_dir: []const u8) !SqliteStore {
        // Ensure data_dir exists for L2/L3 markdown files.
        std.Io.Dir.cwd().createDirPath(io, data_dir) catch |e| switch (e) {
            error.PathAlreadyExists => {},
            else => return e,
        };

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
            .data_dir = try allocator.dupe(u8, data_dir),
            .capabilities = caps,
        };
    }

    pub fn deinit(self: *SqliteStore) void {
        self.db.close();
        self.allocator.free(self.data_dir);
    }

    /// Wrap this store in a `MemoryStore` vtable for use with `MemoryContext`.
    /// The caller is responsible for keeping `self` alive for the lifetime
    /// of the returned `MemoryStore`.
    pub fn toMemoryStore(self: *SqliteStore) store.MemoryStore {
        return .{ .ctx = @ptrCast(self), .vtable = &context.SqliteStoreVTable };
    }

    // ============================
    // L0 — Raw Conversations
    // ============================

    /// Insert one or more L0 conversation records.
    pub fn addConversation(
        self: *SqliteStore,
        records: []const types.L0Record,
        iso: types.IsolationContext,
    ) !void {
        // Use a transaction for atomicity.
        try self.db.exec("BEGIN IMMEDIATE");
        errdefer self.db.exec("ROLLBACK") catch {};

        const sql =
            "INSERT INTO l0_conversations " ++
            "(record_id, session_key, session_id, team_id, user_id, agent_id, task_id, " ++
            "role, message_text, recorded_at, timestamp) " ++
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";

        var stmt = try self.db.prepare(sql);
        defer stmt.finalize();

        for (records) |r| {
            stmt.reset();
            try stmt.bindText(1, r.id);
            try stmt.bindText(2, r.session_key);
            try stmt.bindText(3, r.session_id);
            try stmt.bindText(4, if (r.team_id.len > 0) r.team_id else iso.team_id);
            try stmt.bindText(5, if (r.user_id.len > 0) r.user_id else iso.user_id);
            try stmt.bindText(6, if (r.agent_id.len > 0) r.agent_id else iso.agent_id);
            try stmt.bindText(7, r.task_id);
            try stmt.bindText(8, r.role);
            try stmt.bindText(9, r.message_text);
            try stmt.bindText(10, r.recorded_at);
            try stmt.bindInt(11, r.timestamp);
            const step_rc = stmt.step() catch |e| {
                return e;
            };
            _ = step_rc;
        }

        try self.db.exec("COMMIT");
    }

    /// Query L0 conversations with filtering.
    /// Caller owns the returned slice and each record's strings.
    pub fn queryConversation(
        self: *SqliteStore,
        allocator: std.mem.Allocator,
        filter: types.L0QueryFilter,
        iso: types.IsolationContext,
    ) ![]types.L0Record {
        var time_start_ms_val: ?i64 = null;
        _ = &time_start_ms_val;

        var sql_buf: std.ArrayList(u8) = .empty;
        defer sql_buf.deinit(allocator);
        try sql_buf.appendSlice(
            allocator,
            "SELECT record_id, session_key, session_id, team_id, user_id, agent_id, task_id, " ++
                "role, message_text, recorded_at, timestamp FROM l0_conversations WHERE ",
        );
        try iso.whereClause(allocator, &sql_buf);

        var params = std.ArrayList([]const u8).empty;
        defer params.deinit(allocator);
        try params.append(allocator, iso.team_id);
        try params.append(allocator, iso.agent_id);
        try params.append(allocator, iso.user_id);

        // whereClause already added session_id / task_id conditions if present.
        if (iso.session_id) |sid| {
            try params.append(allocator, sid);
        }
        if (iso.task_id) |tid| {
            try params.append(allocator, tid);
        }
        if (filter.updated_after) |ua| {
            try sql_buf.appendSlice(allocator, " AND recorded_at > ?");
            try params.append(allocator, ua);
        }
        if (filter.time_start_ms) |ts| {
            try sql_buf.appendSlice(allocator, " AND timestamp >= ?");
            // Store the int value to bind after string params.
            time_start_ms_val = ts;
        }

        // Order + limit.
        try sql_buf.appendSlice(allocator, " ORDER BY timestamp ASC");
        if (filter.limit > 0) {
            const limit_str = try std.fmt.allocPrint(allocator, " LIMIT {d}", .{filter.limit});
            defer allocator.free(limit_str);
            try sql_buf.appendSlice(allocator, limit_str);
            if (filter.offset > 0) {
                const offset_str = try std.fmt.allocPrint(allocator, " OFFSET {d}", .{filter.offset});
                defer allocator.free(offset_str);
                try sql_buf.appendSlice(allocator, offset_str);
            }
        }

        var stmt = try self.db.prepare(sql_buf.items);
        defer stmt.finalize();

        // Bind string params (1-based index).
        for (params.items, 1..) |p, i| {
            try stmt.bindText(@intCast(i), p);
        }

        // Bind int param for time_start_ms (if any) at the next position.
        if (time_start_ms_val) |ts| {
            try stmt.bindInt(@intCast(params.items.len + 1), ts);
        }

        var results: std.ArrayList(types.L0Record) = .empty;
        defer results.deinit(allocator);
        errdefer {
            for (results.items) |r| {
                allocator.free(r.id);
                allocator.free(r.session_key);
                allocator.free(r.session_id);
                allocator.free(r.role);
                allocator.free(r.message_text);
                allocator.free(r.recorded_at);
            }
        }

        while (try stmt.step()) {
            const record = types.L0Record{
                .id = try allocator.dupe(u8, stmt.columnText(0)),
                .session_key = try allocator.dupe(u8, stmt.columnText(1)),
                .session_id = try allocator.dupe(u8, stmt.columnText(2)),
                .team_id = try allocator.dupe(u8, stmt.columnText(3)),
                .user_id = try allocator.dupe(u8, stmt.columnText(4)),
                .agent_id = try allocator.dupe(u8, stmt.columnText(5)),
                .task_id = try allocator.dupe(u8, stmt.columnText(6)),
                .role = try allocator.dupe(u8, stmt.columnText(7)),
                .message_text = try allocator.dupe(u8, stmt.columnText(8)),
                .recorded_at = try allocator.dupe(u8, stmt.columnText(9)),
                .timestamp = stmt.columnInt(10),
            };
            try results.append(allocator, record);
        }

        return try results.toOwnedSlice(allocator);
    }

    /// FTS5 keyword search on L0 conversations.
    /// Returns BM25-ranked results.
    pub fn searchConversationFts(
        self: *SqliteStore,
        allocator: std.mem.Allocator,
        query: []const u8,
        top_k: u32,
        iso: types.IsolationContext,
    ) ![]types.SearchResult {
        if (!self.capabilities.fts_search) return &.{};

        // FTS5 query: escape special characters by quoting.
        const fts_query = try buildFtsQuery(allocator, query);
        defer allocator.free(fts_query);

        const sql =
            "SELECT l0.record_id, l0.message_text, 'episodic' AS type, 50.0 AS priority, " ++
            "'' AS scene_name, bm25(l0_fts) AS score, l0.session_id, l0.team_id, l0.user_id, l0.agent_id " ++
            "FROM l0_fts JOIN l0_conversations l0 ON l0_fts.rowid = l0.rowid " ++
            "WHERE l0_fts MATCH ? AND l0.team_id = ? AND l0.agent_id = ? AND l0.user_id = ? " ++
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
            const result = types.SearchResult{
                .record_id = try allocator.dupe(u8, stmt.columnText(0)),
                .content = try allocator.dupe(u8, stmt.columnText(1)),
                .type = .episodic, // L0 doesn't have types; use episodic as default
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

    // ============================
    // L1 — Structured Memories
    // ============================

    /// Upsert an L1 record. If `embedding` is provided, it's stored in l1_embeddings.
    pub fn upsertL1(
        self: *SqliteStore,
        record: types.L1Record,
        embedding: ?[]const f32,
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

        // Store embedding if provided.
        if (embedding) |emb| {
            // Pack f32[] as little-endian bytes.
            const blob_len = emb.len * @sizeOf(f32);
            const blob = try self.allocator.alloc(u8, blob_len);
            defer self.allocator.free(blob);
            for (emb, 0..) |v, i| {
                const bytes = std.mem.toBytes(v);
                @memcpy(blob[i * 4 ..][0..4], &bytes);
            }

            // Delete + insert for embedding.
            const del_emb = "DELETE FROM l1_embeddings WHERE record_id = ?";
            var del_emb_stmt = try self.db.prepare(del_emb);
            defer del_emb_stmt.finalize();
            try del_emb_stmt.bindText(1, record.record_id);
            _ = try del_emb_stmt.step();

            const emb_sql = "INSERT INTO l1_embeddings (record_id, embedding, dimensions, provider, model) VALUES (?, ?, ?, '', '')";
            var emb_stmt = try self.db.prepare(emb_sql);
            defer emb_stmt.finalize();
            try emb_stmt.bindText(1, record.record_id);
            try emb_stmt.bindBlob(2, blob);
            try emb_stmt.bindInt(3, @intCast(emb.len));
            _ = try emb_stmt.step();
        }

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

    /// Vector similarity search on L1 records using cosine similarity.
    /// `query_embedding` is the f32 embedding vector for the query.
    /// Returns SearchResult slice sorted by similarity descending.
    pub fn searchL1Vector(
        self: *SqliteStore,
        allocator: std.mem.Allocator,
        query_embedding: []const f32,
        top_k: u32,
        iso: types.IsolationContext,
    ) ![]types.SearchResult {
        if (query_embedding.len == 0) return &.{};

        // Fetch all L1 records that have embeddings, scoped by isolation.
        const sql =
            "SELECT e.record_id, e.embedding, e.dimensions, " ++
            "l.content, l.type, l.priority, l.scene_name, " ++
            "l.session_id, l.team_id, l.user_id, l.agent_id " ++
            "FROM l1_embeddings e JOIN l1_records l ON e.record_id = l.record_id " ++
            "WHERE l.team_id = ? AND l.agent_id = ? AND l.user_id = ?";

        var stmt = try self.db.prepare(sql);
        defer stmt.finalize();

        try stmt.bindText(1, iso.team_id);
        try stmt.bindText(2, iso.agent_id);
        try stmt.bindText(3, iso.user_id);

        // Collect candidate embeddings + metadata.
        const Candidate = struct {
            record_id: []u8,
            content: []u8,
            mem_type: types.MemoryType,
            priority: f32,
            scene_name: []u8,
            session_id: []u8,
            team_id: []u8,
            user_id: []u8,
            agent_id: []u8,
            embedding: []f32,
        };

        var candidates: std.ArrayList(Candidate) = .empty;
        defer {
            for (candidates.items) |c| {
                allocator.free(c.record_id);
                allocator.free(c.content);
                allocator.free(c.scene_name);
                allocator.free(c.session_id);
                allocator.free(c.team_id);
                allocator.free(c.user_id);
                allocator.free(c.agent_id);
                allocator.free(c.embedding);
            }
            candidates.deinit(allocator);
        }

        while (try stmt.step()) {
            const blob = stmt.columnBlob(1);
            const emb = vector.decodeBlob(allocator, blob) catch continue;
            const mem_type = types.MemoryType.fromString(stmt.columnText(4)) orelse .episodic;
            try candidates.append(allocator, .{
                .record_id = try allocator.dupe(u8, stmt.columnText(0)),
                .content = try allocator.dupe(u8, stmt.columnText(3)),
                .mem_type = mem_type,
                .priority = @floatCast(stmt.columnFloat(5)),
                .scene_name = try allocator.dupe(u8, stmt.columnText(6)),
                .session_id = try allocator.dupe(u8, stmt.columnText(7)),
                .team_id = try allocator.dupe(u8, stmt.columnText(8)),
                .user_id = try allocator.dupe(u8, stmt.columnText(9)),
                .agent_id = try allocator.dupe(u8, stmt.columnText(10)),
                .embedding = emb,
            });
        }

        if (candidates.items.len == 0) return &.{};

        // Compute top-k by cosine similarity.
        var embeddings = try allocator.alloc([]const f32, candidates.items.len);
        defer allocator.free(embeddings);
        for (candidates.items, 0..) |c, i| embeddings[i] = c.embedding;

        const scored = try vector.topK(allocator, query_embedding, embeddings, top_k);
        defer allocator.free(scored);

        var results: std.ArrayList(types.SearchResult) = .empty;
        defer results.deinit(allocator);
        errdefer {
            for (results.items) |r| r.deinit(allocator);
        }

        for (scored) |s| {
            const c = candidates.items[s.index];
            try results.append(allocator, .{
                .record_id = try allocator.dupe(u8, c.record_id),
                .content = try allocator.dupe(u8, c.content),
                .type = c.mem_type,
                .priority = c.priority,
                .scene_name = try allocator.dupe(u8, c.scene_name),
                .score = s.score,
                .session_id = try allocator.dupe(u8, c.session_id),
                .team_id = try allocator.dupe(u8, c.team_id),
                .user_id = try allocator.dupe(u8, c.user_id),
                .agent_id = try allocator.dupe(u8, c.agent_id),
            });
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
        query_embedding: ?[]const f32,
    ) ![]types.SearchResult {
        if (!self.capabilities.fts_search) return &.{};

        // Phase 1: FTS5 search.
        const fts_results = try self.searchL1Fts(allocator, query, top_k, iso);

        // If no query embedding, return FTS-only.
        if (query_embedding == null) return fts_results;

        // Phase 2: Vector search.
        const vec_results = try self.searchL1Vector(allocator, query_embedding.?, top_k, iso);

        // If either result set is empty, return the non-empty one.
        if (fts_results.len == 0) return vec_results;
        if (vec_results.len == 0) return fts_results;

        // RRF-merge FTS + vector results.
        // Both lists need to be freed after merge, but merge deep-copies.
        defer {
            for (fts_results) |r| r.deinit(allocator);
            allocator.free(fts_results);
        }
        defer {
            for (vec_results) |r| r.deinit(allocator);
            allocator.free(vec_results);
        }

        const lists = [_][]const types.SearchResult{ fts_results, vec_results };
        return try rrf.merge(allocator, &lists, top_k);
    }

    // ============================
    // L2/L3 — Markdown Files
    // ============================

    /// Read the L3 persona file.
    pub fn readCore(self: *SqliteStore, allocator: std.mem.Allocator, iso: types.IsolationContext) !?[]u8 {
        _ = iso; // single-user mode: one persona per data_dir
        const path = try std.fmt.allocPrint(allocator, "{s}/persona.md", .{self.data_dir});
        defer allocator.free(path);

        const cwd = std.Io.Dir.cwd();
        const file = cwd.openFile(self.io, path, .{}) catch |e| switch (e) {
            error.FileNotFound => return null,
            else => return e,
        };
        defer file.close(self.io);

        const stat = try file.stat(self.io);
        const content = try allocator.alloc(u8, @intCast(stat.size));
        _ = file.readPositionalAll(self.io, content, 0) catch |e| {
            allocator.free(content);
            return e;
        };
        return content;
    }

    /// Write the L3 persona file.
    pub fn writeCore(self: *SqliteStore, content: []const u8, iso: types.IsolationContext) !void {
        _ = iso;
        const path = try std.fmt.allocPrint(self.allocator, "{s}/persona.md", .{self.data_dir});
        defer self.allocator.free(path);

        const cwd = std.Io.Dir.cwd();
        const file = try cwd.createFile(self.io, path, .{});
        defer file.close(self.io);
        try file.writeStreamingAll(self.io, content);
    }

    /// Read an L2 scenario file by path (relative to data_dir).
    pub fn readScenario(
        self: *SqliteStore,
        allocator: std.mem.Allocator,
        path: []const u8,
        iso: types.IsolationContext,
    ) !?types.ScenarioFile {
        _ = iso;
        const full_path = try std.fmt.allocPrint(allocator, "{s}/scene_blocks/{s}", .{ self.data_dir, path });
        defer allocator.free(full_path);

        const cwd = std.Io.Dir.cwd();
        const file = cwd.openFile(self.io, full_path, .{}) catch |e| switch (e) {
            error.FileNotFound => return null,
            else => return e,
        };
        defer file.close(self.io);

        const stat = try file.stat(self.io);
        const content = try allocator.alloc(u8, @intCast(stat.size));
        _ = file.readPositionalAll(self.io, content, 0) catch |e| {
            allocator.free(content);
            return e;
        };

        return .{
            .path = try allocator.dupe(u8, path),
            .content = content,
            .version = 1,
        };
    }

    /// Write an L2 scenario file.
    pub fn writeScenario(
        self: *SqliteStore,
        path: []const u8,
        content: []const u8,
        iso: types.IsolationContext,
    ) !void {
        _ = iso;
        const dir_path = try std.fmt.allocPrint(self.allocator, "{s}/scene_blocks", .{self.data_dir});
        defer self.allocator.free(dir_path);

        const cwd = std.Io.Dir.cwd();
        cwd.createDirPath(self.io, dir_path) catch |e| switch (e) {
            error.PathAlreadyExists => {},
            else => return e,
        };

        const full_path = try std.fmt.allocPrint(self.allocator, "{s}/scene_blocks/{s}", .{ self.data_dir, path });
        defer self.allocator.free(full_path);

        const file = try cwd.createFile(self.io, full_path, .{});
        defer file.close(self.io);
        try file.writeStreamingAll(self.io, content);
    }

    /// List L2 scenario files.
    pub fn listScenarios(
        self: *SqliteStore,
        allocator: std.mem.Allocator,
        path_prefix: ?[]const u8,
        iso: types.IsolationContext,
    ) ![]types.ScenarioFile {
        _ = iso;
        const dir_path = try std.fmt.allocPrint(allocator, "{s}/scene_blocks", .{self.data_dir});
        defer allocator.free(dir_path);

        const cwd = std.Io.Dir.cwd();
        var dir = cwd.openDir(self.io, dir_path, .{ .iterate = true }) catch |e| switch (e) {
            error.FileNotFound => return &.{},
            else => return e,
        };
        defer dir.close(self.io);

        var results: std.ArrayList(types.ScenarioFile) = .empty;
        defer results.deinit(allocator);
        errdefer {
            for (results.items) |f| f.deinit(allocator);
        }

        var iter = dir.iterate();
        while (try iter.next(self.io)) |entry| {
            if (entry.kind != .file) continue;
            if (!std.mem.endsWith(u8, entry.name, ".md")) continue;

            // Apply prefix filtering if provided.
            if (path_prefix) |prefix| {
                if (!std.mem.startsWith(u8, entry.name, prefix)) continue;
            }

            const content = self.readScenario(allocator, entry.name, .{}) catch continue;
            if (content) |c| {
                try results.append(allocator, c);
            }
        }

        return try results.toOwnedSlice(allocator);
    }

    /// List L2 scenario files ranked by relevance to a query.
    /// Scenario path tokens are weighted double vs content tokens.
    /// Returns scenarios sorted by relevance score descending, with
    /// zero-scoring scenarios (no token overlap) appended last.
    /// When `top_n` > 0, only the top-N scoring scenarios are returned.
    /// Caller owns the returned slice and each file's strings.
    pub fn recallScenarios(
        self: *SqliteStore,
        allocator: std.mem.Allocator,
        query: []const u8,
        top_n: usize,
        iso: types.IsolationContext,
    ) ![]types.ScenarioFile {
        const all = try self.listScenarios(allocator, null, iso);
        if (all.len == 0) return &.{};
        if (query.len == 0 or top_n == 0) return all;

        // Tokenize the query once.
        var query_toks: std.ArrayList([]const u8) = .empty;
        defer {
            for (query_toks.items) |t| allocator.free(t);
            query_toks.deinit(allocator);
        }
        try tokenizeLower(allocator, query, &query_toks);

        // Score each scenario.
        const Scored = struct { file: types.ScenarioFile, score: f32 };
        var scored = try allocator.alloc(Scored, all.len);
        defer allocator.free(scored);
        for (all, 0..) |f, i| {
            const s = try scenarioRelevance(query_toks.items, f.path, f.content, allocator);
            scored[i] = .{ .file = f, .score = s };
        }

        // Stable sort by score descending (zero-scorers sink to the end).
        std.sort.block(Scored, scored, {}, struct {
            fn lt(_: void, a: Scored, b: Scored) bool {
                return a.score > b.score;
            }
        }.lt);

        const keep = @min(top_n, all.len);
        var result = try allocator.alloc(types.ScenarioFile, keep);
        for (scored[0..keep], 0..) |s, i| {
            result[i] = s.file;
        }
        // Free the dropped scenarios (the ones beyond keep).
        for (scored[keep..]) |s| s.file.deinit(allocator);
        return result;
    }

    // ============================
    // Recall
    // ============================

    /// The main entry point for prompt injection.
    /// Searches L1 (hybrid), reads L2/L3, and returns a bounded RecallResult.
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
    /// is truncated to fit within the budget. Priority: L3 persona always included,
    /// then L1 results (most relevant first), then L2 scenarios. Items that don't
    /// fit are dropped entirely (not partially truncated).
    /// Use `max_chars = 0` for unlimited (same as recall()).
    pub fn recallWithBudget(
        self: *SqliteStore,
        allocator: std.mem.Allocator,
        query: []const u8,
        top_k: u32,
        iso: types.IsolationContext,
        max_chars: usize,
    ) !types.RecallResult {
        // L3 persona — always included (if exists).
        const persona = try self.readCore(allocator, iso);

        // L2 scenarios ranked by relevance to the query.
        const all_scenarios = try self.recallScenarios(allocator, query, 0, iso);

        // L1 hybrid search.
        const all_l1 = try self.searchL1Hybrid(allocator, query, top_k, iso, null);

        // If no budget, return everything.
        if (max_chars == 0) {
            var total_chars: usize = 0;
            if (persona) |p| total_chars += p.len;
            for (all_scenarios) |s| total_chars += s.content.len;
            for (all_l1) |r| total_chars += r.content.len;
            return .{
                .persona = persona,
                .scenario_files = all_scenarios,
                .l1_results = all_l1,
                .total_chars = total_chars,
            };
        }

        // Budget capping: persona first, then L1 (most relevant), then L2.
        var remaining = max_chars;
        var total_chars: usize = 0;

        // Persona.
        var capped_persona: ?[]u8 = null;
        if (persona) |p| {
            if (p.len <= remaining) {
                capped_persona = p;
                remaining -= p.len;
                total_chars += p.len;
            } else {
                // Persona doesn't fit — drop it (and its allocation).
                allocator.free(p);
            }
        }

        // L1 results (already sorted by relevance from searchL1Hybrid).
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

        // L2 scenarios.
        var capped_scenarios: []types.ScenarioFile = all_scenarios;
        var scenario_count: usize = 0;
        for (all_scenarios) |s| {
            if (s.content.len <= remaining) {
                remaining -= s.content.len;
                total_chars += s.content.len;
                scenario_count += 1;
            } else {
                break;
            }
        }
        if (scenario_count < all_scenarios.len) {
            for (all_scenarios[scenario_count..]) |s| s.deinit(allocator);
            capped_scenarios = try allocator.realloc(all_scenarios, scenario_count);
        }

        return .{
            .persona = capped_persona,
            .scenario_files = capped_scenarios,
            .l1_results = capped_l1,
            .total_chars = total_chars,
        };
    }

    // ============================
    // Checkpoint
    // ============================

    pub fn getCheckpoint(self: *SqliteStore, allocator: std.mem.Allocator) !types.Checkpoint {
        const sql = "SELECT value FROM pipeline_checkpoint WHERE key = 'last_processed'";
        var stmt = try self.db.prepare(sql);
        defer stmt.finalize();

        if (try stmt.step()) {
            const value = stmt.columnText(0);
            // Parse JSON: {"timestamp": "...", "scene_name": "..."}
            return parseCheckpoint(allocator, value);
        }
        return .{};
    }

    pub fn setCheckpoint(self: *SqliteStore, checkpoint: types.Checkpoint) !void {
        const json = try serializeCheckpoint(self.allocator, checkpoint);
        defer self.allocator.free(json);

        const sql = "INSERT OR REPLACE INTO pipeline_checkpoint (key, value) VALUES ('last_processed', ?)";
        var stmt = try self.db.prepare(sql);
        defer stmt.finalize();
        try stmt.bindText(1, json);
        _ = try stmt.step();
    }

    // ============================
    // Schema Creation
    // ============================

    fn createSchema(db: *sqlite.Db, fts_available: bool) !void {
        // L0 table.
        try db.exec(
            \\CREATE TABLE IF NOT EXISTS l0_conversations (
            \\  record_id TEXT PRIMARY KEY,
            \\  session_key TEXT NOT NULL,
            \\  session_id TEXT NOT NULL,
            \\  team_id TEXT NOT NULL DEFAULT 'default',
            \\  user_id TEXT NOT NULL DEFAULT 'default',
            \\  agent_id TEXT NOT NULL DEFAULT 'default',
            \\  task_id TEXT NOT NULL DEFAULT '',
            \\  role TEXT NOT NULL,
            \\  message_text TEXT NOT NULL,
            \\  recorded_at TEXT NOT NULL,
            \\  timestamp INTEGER NOT NULL
            \\)
        );
        try db.exec("CREATE INDEX IF NOT EXISTS idx_l0_session ON l0_conversations(session_id)");
        try db.exec("CREATE INDEX IF NOT EXISTS idx_l0_timestamp ON l0_conversations(timestamp)");

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

        // L1 embeddings table (optional, for vector search).
        try db.exec(
            \\CREATE TABLE IF NOT EXISTS l1_embeddings (
            \\  record_id TEXT PRIMARY KEY REFERENCES l1_records(record_id) ON DELETE CASCADE,
            \\  embedding BLOB NOT NULL,
            \\  dimensions INTEGER NOT NULL,
            \\  provider TEXT NOT NULL DEFAULT '',
            \\  model TEXT NOT NULL DEFAULT ''
            \\)
        );

        // Pipeline checkpoint table.
        try db.exec(
            \\CREATE TABLE IF NOT EXISTS pipeline_checkpoint (
            \\  key TEXT PRIMARY KEY,
            \\  value TEXT NOT NULL
            \\)
        );

        // FTS5 tables + triggers (only if FTS5 is available).
        if (fts_available) {
            try db.exec(
                \\CREATE VIRTUAL TABLE IF NOT EXISTS l0_fts USING fts5(
                \\  message_text,
                \\  content='l0_conversations',
                \\  content_rowid='rowid'
                \\)
            );
            try db.exec(
                \\CREATE VIRTUAL TABLE IF NOT EXISTS l1_fts USING fts5(
                \\  content,
                \\  content='l1_records',
                \\  content_rowid='rowid'
                \\)
            );

            // Triggers to keep FTS in sync.
            try db.exec(
                \\CREATE TRIGGER IF NOT EXISTS l0_ai AFTER INSERT ON l0_conversations BEGIN
                \\  INSERT INTO l0_fts(rowid, message_text) VALUES (new.rowid, new.message_text);
                \\END
            );
            try db.exec(
                \\CREATE TRIGGER IF NOT EXISTS l0_ad AFTER DELETE ON l0_conversations BEGIN
                \\  INSERT INTO l0_fts(l0_fts, rowid, message_text) VALUES('delete', old.rowid, old.message_text);
                \\END
            );
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