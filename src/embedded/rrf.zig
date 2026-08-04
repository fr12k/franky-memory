//! Reciprocal Rank Fusion — merges multiple ranked search result lists
//! into a single ranking.
//!
//! Ported from TencentDB Agent Memory's `search-utils.ts` (RRF_K = 60).
//!
//! Algorithm: for each item in each ranked list, compute
//!   score = 1 / (k + rank + 1)
//! and sum scores across lists. Higher total = higher final rank.
//! Items appearing in multiple lists get boosted (fusion effect).

const std = @import("std");
const types = @import("../types.zig");

/// Standard RRF constant from the original RRF paper.
/// Higher k → more weight on lower-ranked items (smoother distribution).
pub const RRF_K: u32 = 60;

const Entry = struct {
    item: types.SearchResult,
    rrf_score: f32,
};

/// Merge `lists` of search results using RRF and return the top `top_k`.
///
/// `getId` extracts the dedup key (record_id) from a SearchResult.
/// Caller owns the returned slice.
pub fn merge(
    allocator: std.mem.Allocator,
    lists: []const []const types.SearchResult,
    top_k: u32,
) ![]types.SearchResult {
    if (lists.len == 0) return &.{};

    // Accumulate RRF scores per record_id.
    var map = std.StringHashMap(Entry).init(allocator);
    defer map.deinit();

    for (lists) |list| {
        for (list, 0..) |item, rank| {
            const score = 1.0 / @as(f32, @floatFromInt(RRF_K + @as(u32, @intCast(rank)) + 1));
            const gop = try map.getOrPut(item.record_id);
            if (gop.found_existing) {
                gop.value_ptr.rrf_score += score;
            } else {
                gop.value_ptr.* = .{
                    .item = item,
                    .rrf_score = score,
                };
            }
        }
    }

    // Collect entries for sorting.
    var entries = try allocator.alloc(Entry, map.count());
    defer allocator.free(entries);
    {
        var iter = map.iterator();
        var i: usize = 0;
        while (iter.next()) |kv| : (i += 1) {
            entries[i] = kv.value_ptr.*;
        }
    }

    // Sort by RRF score descending.
    std.sort.block(Entry, entries, {}, struct {
        fn lt(_: void, a: Entry, b: Entry) bool {
            return a.rrf_score > b.rrf_score;
        }
    }.lt);

    // Build result slice (top_k, capped).
    const result_len = @min(top_k, entries.len);
    var result = try allocator.alloc(types.SearchResult, result_len);
    for (entries[0..result_len], 0..) |e, i| {
        // Deep-copy the item so the result is independently owned.
        result[i] = try dupeSearchResult(allocator, e.item);
        result[i].score = e.rrf_score; // overwrite with RRF score
    }
    return result;
}

/// Deep-copy a SearchResult (all owned strings duplicated).
fn dupeSearchResult(
    allocator: std.mem.Allocator,
    src: types.SearchResult,
) !types.SearchResult {
    return .{
        .record_id = try allocator.dupe(u8, src.record_id),
        .content = try allocator.dupe(u8, src.content),
        .type = src.type,
        .priority = src.priority,
        .scene_name = try allocator.dupe(u8, src.scene_name),
        .score = src.score,
        .session_id = try allocator.dupe(u8, src.session_id),
        .team_id = try allocator.dupe(u8, src.team_id),
        .user_id = try allocator.dupe(u8, src.user_id),
        .agent_id = try allocator.dupe(u8, src.agent_id),
    };
}

// ============================
// Tests
// ============================

test "RRF empty lists returns empty" {
    const allocator = std.testing.allocator;
    const result = try merge(allocator, &.{}, 5);
    defer allocator.free(result);
    try std.testing.expectEqual(@as(usize, 0), result.len);
}

test "RRF single list passes through" {
    const allocator = std.testing.allocator;
    var r1 = types.SearchResult{
        .record_id = "a",
        .content = "alpha",
        .type = .episodic,
        .priority = 50,
        .scene_name = "s1",
        .score = 0.9,
        .session_id = "s",
        .team_id = "t",
        .user_id = "u",
        .agent_id = "a",
    };
    _ = &r1;
    const list = &[_]types.SearchResult{r1};
    const result = try merge(allocator, &.{list}, 5);
    defer {
        for (result) |r| r.deinit(allocator);
        allocator.free(result);
    }
    try std.testing.expectEqual(@as(usize, 1), result.len);
    try std.testing.expectEqualStrings("a", result[0].record_id);
}

test "RRF merges two lists with overlapping items" {
    const allocator = std.testing.allocator;
    const r1 = types.SearchResult{
        .record_id = "a",
        .content = "alpha",
        .type = .episodic,
        .priority = 50,
        .scene_name = "s1",
        .score = 0.9,
        .session_id = "s",
        .team_id = "t",
        .user_id = "u",
        .agent_id = "a",
    };
    const r2 = types.SearchResult{
        .record_id = "b",
        .content = "beta",
        .type = .persona,
        .priority = 60,
        .scene_name = "s2",
        .score = 0.8,
        .session_id = "s",
        .team_id = "t",
        .user_id = "u",
        .agent_id = "a",
    };
    // List 1: [a, b]  → ranks 0, 1
    // List 2: [b, a]  → ranks 0, 1
    // a: 1/(60+0+1) + 1/(60+1+1) = 1/61 + 1/62
    // b: 1/(60+1+1) + 1/(60+0+1) = 1/62 + 1/61
    // a == b (symmetric), so both have the same RRF score.
    const list1 = &[_]types.SearchResult{ r1, r2 };
    const list2 = &[_]types.SearchResult{ r2, r1 };
    const result = try merge(allocator, &.{ list1, list2 }, 5);
    defer {
        for (result) |r| r.deinit(allocator);
        allocator.free(result);
    }
    try std.testing.expectEqual(@as(usize, 2), result.len);
    // Both should have the same score (symmetric overlap).
    try std.testing.expectApproxEqAbs(result[0].score, result[1].score, 0.0001);
}

test "RRF item in both lists ranks higher than item in one" {
    const allocator = std.testing.allocator;
    const r_common = types.SearchResult{
        .record_id = "common",
        .content = "c",
        .type = .episodic,
        .priority = 50,
        .scene_name = "s",
        .score = 0.5,
        .session_id = "s",
        .team_id = "t",
        .user_id = "u",
        .agent_id = "a",
    };
    const r_only1 = types.SearchResult{
        .record_id = "only1",
        .content = "x",
        .type = .persona,
        .priority = 40,
        .scene_name = "s",
        .score = 0.5,
        .session_id = "s",
        .team_id = "t",
        .user_id = "u",
        .agent_id = "a",
    };
    // "common" is rank 0 in both lists; "only1" is rank 1 in list1 only.
    // common: 1/61 + 1/61 = 2/61 ≈ 0.0328
    // only1:  1/62        ≈ 0.0161
    // common should rank first.
    const list1 = &[_]types.SearchResult{ r_common, r_only1 };
    const list2 = &[_]types.SearchResult{r_common};
    const result = try merge(allocator, &.{ list1, list2 }, 5);
    defer {
        for (result) |r| r.deinit(allocator);
        allocator.free(result);
    }
    try std.testing.expectEqual(@as(usize, 2), result.len);
    try std.testing.expectEqualStrings("common", result[0].record_id);
    try std.testing.expect(result[0].score > result[1].score);
}

test "RRF top_k caps results" {
    const allocator = std.testing.allocator;
    var records: [5]types.SearchResult = undefined;
    for (&records, 0..) |*r, i| {
        r.* = .{
            .record_id = try std.fmt.allocPrint(allocator, "id{d}", .{i}),
            .content = "c",
            .type = .episodic,
            .priority = 50,
            .scene_name = "s",
            .score = 0.5,
            .session_id = "s",
            .team_id = "t",
            .user_id = "u",
            .agent_id = "a",
        };
    }
    defer for (&records) |*r| {
        allocator.free(r.record_id);
    };
    const list = &records;
    const result = try merge(allocator, &.{list}, 3);
    defer {
        for (result) |r| r.deinit(allocator);
        allocator.free(result);
    }
    try std.testing.expectEqual(@as(usize, 3), result.len);
}