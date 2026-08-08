//! Core type definitions for agent-memory-zig.
//!
//! These types define the memory record shapes (L0–L3) and the isolation
//! context used to scope multi-tenant data. All types are plain structs;
//! every value is serializable to JSON.

const std = @import("std");

// ============================
// Memory Type
// ============================

/// The three memory categories extracted from L0 conversations.
///
/// Mirrors the `type` field in TencentDB Agent Memory's L1 records.
pub const MemoryType = enum {
    persona,
    episodic,
    instruction,

    pub fn fromString(s: []const u8) ?MemoryType {
        if (std.mem.eql(u8, s, "persona")) return .persona;
        if (std.mem.eql(u8, s, "episodic")) return .episodic;
        if (std.mem.eql(u8, s, "instruction")) return .instruction;
        return null;
    }

    pub fn toString(self: MemoryType) []const u8 {
        return switch (self) {
            .persona => "persona",
            .episodic => "episodic",
            .instruction => "instruction",
        };
    }
};

// ============================
// Isolation Context
// ============================

/// Five-dimensional tenancy scoping for all memory writes/queries.
///
/// Every write into L0/L1/Profile MUST carry an IsolationContext.
/// Every query accepts one to narrow which dimensions are filtered.
///
/// For single-user embedded mode, all fields default to "default".
/// The schema keeps these columns so multi-agent support can be added
/// later without migration.
pub const IsolationContext = struct {
    team_id: []const u8 = "default",
    agent_id: []const u8 = "default",
    user_id: []const u8 = "default",
    session_id: ?[]const u8 = null,
    task_id: ?[]const u8 = null,

    /// Build the isolation fragment of a SQL WHERE clause.
    /// Returns a string like: `team_id = ? AND agent_id = ? AND user_id = ?`
    /// The caller is responsible for binding the values in order:
    /// team_id, agent_id, user_id.
    pub fn whereClause(self: IsolationContext, allocator: std.mem.Allocator, buf: *std.ArrayList(u8)) !void {
        try buf.appendSlice(allocator, "team_id = ? AND agent_id = ? AND user_id = ?");
        if (self.session_id) |sid| {
            try buf.appendSlice(allocator, " AND session_id = ?");
            _ = sid; // bound by caller
        }
        if (self.task_id) |tid| {
            try buf.appendSlice(allocator, " AND task_id = ?");
            _ = tid; // bound by caller
        }
    }

    /// Number of bind parameters the whereClause produces.
    pub fn whereParamCount(self: IsolationContext) usize {
        var n: usize = 3; // team, agent, user
        if (self.session_id != null) n += 1;
        if (self.task_id != null) n += 1;
        return n;
    }
};

// ============================
// L1 — Structured Memories
// ============================

/// A structured memory atom distilled from L0 by the extraction pipeline.
/// Each record is a self-contained fact that "makes sense without context."
pub const L1Record = struct {
    record_id: []const u8,
    content: []const u8,
    type: MemoryType,
    priority: f32,
    scene_name: []const u8,
    session_key: []const u8,
    session_id: []const u8,
    team_id: []const u8,
    task_id: []const u8,
    user_id: []const u8,
    agent_id: []const u8,
    version: u32,
    timestamp_str: []const u8,
    timestamp_start: []const u8,
    timestamp_end: []const u8,
    created_time: []const u8,
    updated_time: []const u8,
    metadata_json: []const u8,

    pub fn deinit(self: L1Record, allocator: std.mem.Allocator) void {
        // Caller-owned strings — free each one.
        allocator.free(self.record_id);
        allocator.free(self.content);
        allocator.free(self.scene_name);
        allocator.free(self.session_key);
        allocator.free(self.session_id);
        allocator.free(self.team_id);
        allocator.free(self.task_id);
        allocator.free(self.user_id);
        allocator.free(self.agent_id);
        allocator.free(self.timestamp_str);
        allocator.free(self.timestamp_start);
        allocator.free(self.timestamp_end);
        allocator.free(self.created_time);
        allocator.free(self.updated_time);
        allocator.free(self.metadata_json);
    }
};

// ============================
// Search Result
// ============================

/// A single hit from a memory search (L0 or L1).
/// `score` is either a BM25 rank or an RRF-fused score.
pub const SearchResult = struct {
    record_id: []const u8,
    content: []const u8,
    type: MemoryType,
    priority: f32,
    scene_name: []const u8,
    score: f32,
    session_id: []const u8,
    team_id: []const u8,
    user_id: []const u8,
    agent_id: []const u8,

    pub fn deinit(self: SearchResult, allocator: std.mem.Allocator) void {
        allocator.free(self.record_id);
        allocator.free(self.content);
        allocator.free(self.scene_name);
        allocator.free(self.session_id);
        allocator.free(self.team_id);
        allocator.free(self.user_id);
        allocator.free(self.agent_id);
    }
};

// ============================
// Capabilities
// ============================

/// Describes what search capabilities a store backend supports.
/// Callers use this to select search strategies and degrade gracefully.
pub const StoreCapabilities = struct {
    vector_search: bool = false,
    fts_search: bool = false,
};

// ============================
// Recall Result
// ============================

/// The aggregated result of a recall operation — what gets injected
/// into the system prompt before the next LLM call.
pub const RecallResult = struct {
    /// L3 persona content (markdown), or null if no persona file exists.
    persona: ?[]const u8 = null,
    /// L2 scenario files relevant to the query.
    scenario_files: []ScenarioFile = &.{},
    /// L1 hybrid search hits (vector + BM25 + RRF).
    l1_results: []SearchResult = &.{},
    /// Total character count of all content (for budget capping).
    total_chars: usize = 0,

    pub fn deinit(self: *RecallResult, allocator: std.mem.Allocator) void {
        if (self.persona) |p| allocator.free(p);
        for (self.scenario_files) |f| f.deinit(allocator);
        if (self.scenario_files.len > 0) allocator.free(self.scenario_files);
        for (self.l1_results) |r| r.deinit(allocator);
        if (self.l1_results.len > 0) allocator.free(self.l1_results);
    }
};

/// An L2 scenario markdown file.
pub const ScenarioFile = struct {
    path: []const u8,
    content: []const u8,
    version: u32,

    pub fn deinit(self: ScenarioFile, allocator: std.mem.Allocator) void {
        allocator.free(self.path);
        allocator.free(self.content);
    }
};

// ============================
// Query Filters
// ============================

/// Filter for querying L1 records.
pub const L1QueryFilter = struct {
    session_id: ?[]const u8 = null,
    type: ?MemoryType = null,
    time_start: ?[]const u8 = null,
    time_end: ?[]const u8 = null,
    limit: u32 = 100,
    offset: u32 = 0,
};

// ============================
// Checkpoint
// ============================

/// Pipeline checkpoint — tracks which L0 messages have been processed
/// by the L1 extraction pipeline, so we only extract new ones.
pub const Checkpoint = struct {
    /// ISO 8601 timestamp of the last processed L0 message.
    last_processed_timestamp: ?[]const u8 = null,
    /// Last scene name detected by the extraction LLM.
    last_scene_name: ?[]const u8 = null,

    pub fn deinit(self: Checkpoint, allocator: std.mem.Allocator) void {
        if (self.last_processed_timestamp) |t| allocator.free(t);
        if (self.last_scene_name) |s| allocator.free(s);
    }
};

// ============================
// Extraction Result
// ============================

/// Result of a single L1 extraction run.
pub const ExtractionResult = struct {
    extracted_count: u32,
    stored_count: u32,
};

// ============================
// Dedup Decision
// ============================

/// How to handle a newly extracted memory relative to existing L1 records.
pub const DedupAction = enum { store, update, merge, skip };

pub const DedupDecision = struct {
    action: DedupAction,
    /// The final content to persist (may differ from extracted if merged).
    merged_content: ?[]const u8 = null,
    /// The final priority (may differ if merged).
    merged_priority: ?f32 = null,
    /// The record_id of the existing record to delete (for update/merge).
    existing_record_id: ?[]const u8 = null,
};

// ============================
// Tests
// ============================

test "MemoryType round-trip" {
    try std.testing.expectEqual(@as(?MemoryType, .persona), MemoryType.fromString("persona"));
    try std.testing.expectEqual(@as(?MemoryType, .episodic), MemoryType.fromString("episodic"));
    try std.testing.expectEqual(@as(?MemoryType, .instruction), MemoryType.fromString("instruction"));
    try std.testing.expectEqual(@as(?MemoryType, null), MemoryType.fromString("unknown"));
    try std.testing.expectEqualStrings("persona", MemoryType.toString(.persona));
}

test "IsolationContext defaults" {
    const iso = IsolationContext{};
    try std.testing.expectEqualStrings("default", iso.team_id);
    try std.testing.expectEqualStrings("default", iso.agent_id);
    try std.testing.expectEqualStrings("default", iso.user_id);
    try std.testing.expectEqual(@as(?[]const u8, null), iso.session_id);
}

test "IsolationContext whereClause without optional fields" {
    const iso = IsolationContext{};
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(std.testing.allocator);
    try iso.whereClause(std.testing.allocator, &buf);
    try std.testing.expectEqualStrings("team_id = ? AND agent_id = ? AND user_id = ?", buf.items);
    try std.testing.expectEqual(@as(usize, 3), iso.whereParamCount());
}

test "IsolationContext whereClause with session_id" {
    const iso = IsolationContext{ .session_id = "s1" };
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(std.testing.allocator);
    try iso.whereClause(std.testing.allocator, &buf);
    try std.testing.expectEqualStrings(
        "team_id = ? AND agent_id = ? AND user_id = ? AND session_id = ?",
        buf.items,
    );
    try std.testing.expectEqual(@as(usize, 4), iso.whereParamCount());
}

test "IsolationContext whereClause with session_id and task_id" {
    const iso = IsolationContext{ .session_id = "s1", .task_id = "t1" };
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(std.testing.allocator);
    try iso.whereClause(std.testing.allocator, &buf);
    try std.testing.expectEqualStrings(
        "team_id = ? AND agent_id = ? AND user_id = ? AND session_id = ? AND task_id = ?",
        buf.items,
    );
    try std.testing.expectEqual(@as(usize, 5), iso.whereParamCount());
}