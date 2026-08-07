//! agent-memory-zig — embedded memory store for LLM agents.
//!
//! A Zig implementation of the "Database Memory" concept from TencentDB
//! Agent Memory (https://github.com/TencentCloud/TencentDB-Agent-Memory).
//!
//! Provides layered persistent memory (L0–L3) backed by SQLite + FTS5,
//! designed to be linked into an LLM harness (like franky) as a package
//! dependency.
//!
//! See ANALYSIS.md for the full design document and implementation status.

const std = @import("std");

// Public API
pub const types = @import("types.zig");
pub const store = @import("store.zig");
pub const context = @import("context.zig");
pub const embedded = @import("embedded/sqlite_store.zig");
pub const rrf = @import("embedded/rrf.zig");
pub const vector = @import("embedded/vector.zig");

// Re-export key types at the top level for convenience.
pub const MemoryType = types.MemoryType;
pub const IsolationContext = types.IsolationContext;
pub const L0Record = types.L0Record;
pub const L1Record = types.L1Record;
pub const SearchResult = types.SearchResult;
pub const StoreCapabilities = types.StoreCapabilities;
pub const RecallResult = types.RecallResult;
pub const ScenarioFile = types.ScenarioFile;
pub const L0QueryFilter = types.L0QueryFilter;
pub const L1QueryFilter = types.L1QueryFilter;
pub const Checkpoint = types.Checkpoint;
pub const ExtractionResult = types.ExtractionResult;
pub const DedupAction = types.DedupAction;
pub const DedupDecision = types.DedupDecision;
pub const MemoryStore = store.MemoryStore;
pub const SqliteStore = embedded.SqliteStore;
pub const MemoryContext = context.MemoryContext;

test {
    // Pull in all test files when running `zig test`.
    std.testing.refAllDecls(types);
    std.testing.refAllDecls(rrf);
    std.testing.refAllDecls(vector);
    std.testing.refAllDecls(embedded);
    std.testing.refAllDecls(context);
}