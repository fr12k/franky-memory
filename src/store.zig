//! Memory store interface — a vtable-based dispatch so callers can use
//! any concrete implementation behind a uniform API.
//!
//! Currently only `SqliteStore` is implemented (embedded mode).
//! The vtable keeps the door open for a future remote (HTTP) backend.

const std = @import("std");
const types = @import("types.zig");

pub const MemoryStore = struct {
    ctx: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        deinit: *const fn (ctx: *anyopaque) void,
        capabilities: *const fn (ctx: *anyopaque) types.StoreCapabilities,

        // L1
        upsert_l1: *const fn (
            ctx: *anyopaque,
            record: types.L1Record,
            iso: types.IsolationContext,
        ) anyerror!bool,

        search_l1: *const fn (
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
            query: []const u8,
            top_k: u32,
            iso: types.IsolationContext,
        ) anyerror![]types.SearchResult,

        delete_l1: *const fn (
            ctx: *anyopaque,
            record_id: []const u8,
            options: types.DeleteOptions,
            iso: types.IsolationContext,
        ) anyerror!bool,

        restore_l1: *const fn (
            ctx: *anyopaque,
            record_id: []const u8,
            iso: types.IsolationContext,
        ) anyerror!bool,

        purge_deleted_l1: *const fn (
            ctx: *anyopaque,
            iso: types.IsolationContext,
        ) anyerror!u32,

        // L1 — list
        list_l1: *const fn (
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
            filter: types.L1QueryFilter,
            iso: types.IsolationContext,
        ) anyerror![]types.MemorySummary,

        // Recall
        recall: *const fn (
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
            query: []const u8,
            top_k: u32,
            iso: types.IsolationContext,
        ) anyerror!types.RecallResult,

        recall_with_budget: *const fn (
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
            query: []const u8,
            top_k: u32,
            iso: types.IsolationContext,
            max_chars: usize,
        ) anyerror!types.RecallResult,
    };

    pub fn deinit(self: MemoryStore) void {
        self.vtable.deinit(self.ctx);
    }

    pub fn capabilities(self: MemoryStore) types.StoreCapabilities {
        return self.vtable.capabilities(self.ctx);
    }

    pub fn upsertL1(self: MemoryStore, record: types.L1Record, iso: types.IsolationContext) !bool {
        return self.vtable.upsert_l1(self.ctx, record, iso);
    }

    pub fn searchL1(self: MemoryStore, allocator: std.mem.Allocator, query: []const u8, top_k: u32, iso: types.IsolationContext) ![]types.SearchResult {
        return self.vtable.search_l1(self.ctx, allocator, query, top_k, iso);
    }

    /// Delete an L1 record (soft by default). Returns true when a row was affected.
    pub fn deleteL1(self: MemoryStore, record_id: []const u8, options: types.DeleteOptions, iso: types.IsolationContext) !bool {
        return self.vtable.delete_l1(self.ctx, record_id, options, iso);
    }

    /// Restore a soft-deleted L1 record. Returns true when a row was revived.
    pub fn restoreL1(self: MemoryStore, record_id: []const u8, iso: types.IsolationContext) !bool {
        return self.vtable.restore_l1(self.ctx, record_id, iso);
    }

    /// Physically remove all soft-deleted records in scope. Returns the purge count.
    pub fn purgeDeletedL1(self: MemoryStore, iso: types.IsolationContext) !u32 {
        return self.vtable.purge_deleted_l1(self.ctx, iso);
    }

    /// Enumerate live (non-deleted) L1 memories in scope, returning only the
    /// metadata projection (`scene_name`, `created_time`, `updated_time`,
    /// `metadata_json`). The `filter` narrows the result set (session/type/
    /// time range/limit/offset); pass `.{}` for all live records up to the
    /// default limit of 100.
    ///
    /// Returns an owned `[]MemorySummary` — the caller MUST free each entry
    /// via `deinit` and then free the slice:
    ///   ```zig
    ///   const items = try store.listL1(allocator, .{}, iso);
    ///   defer { for (items) |it| it.deinit(allocator); allocator.free(items); }
    ///   ```
    pub fn listL1(
        self: MemoryStore,
        allocator: std.mem.Allocator,
        filter: types.L1QueryFilter,
        iso: types.IsolationContext,
    ) ![]types.MemorySummary {
        return self.vtable.list_l1(self.ctx, allocator, filter, iso);
    }

    pub fn recall(self: MemoryStore, allocator: std.mem.Allocator, query: []const u8, top_k: u32, iso: types.IsolationContext) !types.RecallResult {
        return self.vtable.recall(self.ctx, allocator, query, top_k, iso);
    }

    pub fn recallWithBudget(self: MemoryStore, allocator: std.mem.Allocator, query: []const u8, top_k: u32, iso: types.IsolationContext, max_chars: usize) !types.RecallResult {
        return self.vtable.recall_with_budget(self.ctx, allocator, query, top_k, iso, max_chars);
    }
};
