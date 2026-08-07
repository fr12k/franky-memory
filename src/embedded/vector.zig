//! Brute-force vector similarity (cosine) for L1 embedding search.
//!
//! No external dependencies — pure Zig. Designed for small-to-medium
//! embedding stores (thousands of records). For larger scale, a
//! sqlite-vec or ANN index would be needed (future work).
//!
//! Embeddings are stored as packed f32 little-endian blobs in the
//! `l1_embeddings` table. This module provides the decode + cosine
//! similarity + top-k selection logic.

const std = @import("std");

/// Decode a packed f32 blob (little-endian) into a slice of f32.
/// The returned slice is caller-owned (allocated via `allocator`).
pub fn decodeBlob(allocator: std.mem.Allocator, blob: []const u8) ![]f32 {
    if (blob.len % @sizeOf(f32) != 0) return error.InvalidEmbeddingBlob;
    const dims = blob.len / @sizeOf(f32);
    const result = try allocator.alloc(f32, dims);
    for (0..dims) |i| {
        const bytes: [4]u8 = blob[i * 4 ..][0..4].*;
        result[i] = std.mem.bytesToValue(f32, &bytes);
    }
    return result;
}

/// Encode a f32 slice into a packed little-endian blob.
/// The returned slice is caller-owned.
pub fn encodeBlob(allocator: std.mem.Allocator, vec: []const f32) ![]u8 {
    const blob = try allocator.alloc(u8, vec.len * @sizeOf(f32));
    for (vec, 0..) |v, i| {
        const bytes = std.mem.toBytes(v);
        @memcpy(blob[i * 4 ..][0..4], &bytes);
    }
    return blob;
}

/// Compute cosine similarity between two vectors.
/// Returns a value in [-1.0, 1.0]. Returns 0.0 if either vector has zero magnitude.
pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    if (a.len != b.len or a.len == 0) return 0.0;

    var dot: f32 = 0.0;
    var norm_a: f32 = 0.0;
    var norm_b: f32 = 0.0;

    for (a, b) |av, bv| {
        dot += av * bv;
        norm_a += av * av;
        norm_b += bv * bv;
    }

    const denom = @sqrt(norm_a) * @sqrt(norm_b);
    if (denom == 0.0) return 0.0;
    return dot / denom;
}

/// A single scored entry from vector search.
pub const ScoredEntry = struct {
    index: usize,
    score: f32,
};

/// Compute top-k indices by cosine similarity against a query vector.
/// `candidates` is a slice where each element is a decoded embedding.
/// Returns indices into `candidates`, sorted by score descending.
/// The returned slice is caller-owned.
pub fn topK(
    allocator: std.mem.Allocator,
    query: []const f32,
    candidates: []const []const f32,
    k: u32,
) ![]ScoredEntry {
    if (candidates.len == 0 or k == 0) return &.{};

    var entries = try allocator.alloc(ScoredEntry, candidates.len);
    defer allocator.free(entries);

    for (candidates, 0..) |c, i| {
        entries[i] = .{
            .index = i,
            .score = cosineSimilarity(query, c),
        };
    }

    // Sort by score descending.
    std.sort.block(ScoredEntry, entries, {}, struct {
        fn lt(_: void, a: ScoredEntry, b: ScoredEntry) bool {
            return a.score > b.score;
        }
    }.lt);

    const result_len = @min(k, @as(u32, @intCast(candidates.len)));
    const result = try allocator.alloc(ScoredEntry, result_len);
    @memcpy(result, entries[0..result_len]);
    return result;
}

test "cosineSimilarity identical vectors" {
    const a = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ 1.0, 2.0, 3.0 };
    try std.testing.expectApproxEqAbs(cosineSimilarity(&a, &b), 1.0, 0.0001);
}

test "cosineSimilarity orthogonal vectors" {
    const a = [_]f32{ 1.0, 0.0 };
    const b = [_]f32{ 0.0, 1.0 };
    try std.testing.expectApproxEqAbs(cosineSimilarity(&a, &b), 0.0, 0.0001);
}

test "cosineSimilarity opposite vectors" {
    const a = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ -1.0, -2.0, -3.0 };
    try std.testing.expectApproxEqAbs(cosineSimilarity(&a, &b), -1.0, 0.0001);
}

test "cosineSimilarity zero vector" {
    const a = [_]f32{ 0.0, 0.0, 0.0 };
    const b = [_]f32{ 1.0, 2.0, 3.0 };
    try std.testing.expectApproxEqAbs(cosineSimilarity(&a, &b), 0.0, 0.0001);
}

test "cosineSimilarity mismatched dimensions" {
    const a = [_]f32{ 1.0, 2.0 };
    const b = [_]f32{ 1.0, 2.0, 3.0 };
    try std.testing.expectApproxEqAbs(cosineSimilarity(&a, &b), 0.0, 0.0001);
}

test "encodeDecodeRoundTrip" {
    const original = [_]f32{ 0.1, 0.2, 0.3, -0.4, 0.5 };
    const blob = try encodeBlob(std.testing.allocator, &original);
    defer std.testing.allocator.free(blob);
    const decoded = try decodeBlob(std.testing.allocator, blob);
    defer std.testing.allocator.free(decoded);
    try std.testing.expectEqualSlices(f32, &original, decoded);
}

test "topK basic" {
    const query = [_]f32{ 1.0, 0.0 };
    const c1 = [_]f32{ 0.9, 0.1 };
    const c2 = [_]f32{ 0.1, 0.9 };
    const c3 = [_]f32{ 1.0, 0.0 };
    const candidates = [_][]const f32{ &c1, &c2, &c3 };
    const result = try topK(std.testing.allocator, &query, &candidates, 2);
    defer std.testing.allocator.free(result);
    try std.testing.expectEqual(@as(usize, 2), result.len);
    // c3 (index 2) has highest similarity, then c1 (index 0)
    try std.testing.expectEqual(@as(usize, 2), result[0].index);
    try std.testing.expectEqual(@as(usize, 0), result[1].index);
}

test "topK empty candidates" {
    const query = [_]f32{ 1.0, 0.0 };
    const candidates = [_][]const f32{};
    const result = try topK(std.testing.allocator, &query, &candidates, 5);
    try std.testing.expectEqual(@as(usize, 0), result.len);
}