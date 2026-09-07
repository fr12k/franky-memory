# franky-memory

Embedded persistent memory store for the [franky](https://github.com/fr12k/franky) LLM agent framework.

A Zig implementation of the "Database Memory" concept from [TencentDB Agent Memory](https://github.com/TencentCloud/TencentDB-Agent-Memory), adapted for single-binary, embedded operation — no Node.js sidecar, no external HTTP gateway, no separate API keys.

## What it does

Provides a single-layer persistent memory store for LLM agents:

| Layer | Stores | Storage | Purpose |
|-------|--------|---------|---------|
| **L1 Atom** | Facts, preferences, decisions | SQLite rows + FTS5 index | Compact, self-contained memory atoms |

## Architecture

```
franky Agent Loop
  │
  │ 1. Save → store facts/preferences/decisions (SQLite INSERT + FTS5)
  │ 2. Before next prompt → recall L1 (FTS5 + optional vector), inject as [Memory Context]
  ▼
SQLite database (~/.franky/memory.db)
```

**Persistence** is SQLite + FTS5 (full-text search with BM25 ranking).
**Intelligence** (L1 extraction) calls franky's own `ai.registry.Registry` — no separate LLM client.

## Usage

### As a dependency (in franky's `build.zig.zon`)

```zig
.dependencies = .{
    .agent_memory = .{
        .url = "https://github.com/franky-agent/franky-memory/archive/refs/tags/v0.5.0.tar.gz",
        .hash = "agent_memory-0.5.0-yf36fgpinQAZnwEvO_CLblpYsvJlD6cBxjTkw8T8JygT",
    },
},
```

### Direct API

```zig
const agent_memory = @import("agent_memory");

var store = try agent_memory.SqliteStore.init(
    allocator,
    io,
    "/home/user/.franky/memory.db",
);
defer store.deinit();

// L1 — store extracted memory
_ = try store.upsertL1(.{
    .record_id = "mem-1", .content = "User uses PostgreSQL",
    .type = .episodic, .priority = 75, .scene_name = "database",
    .session_key = "sk1", .session_id = "s1",
    .team_id = "default", .task_id = "", .user_id = "default",
    .agent_id = "default", .version = 1,
    .timestamp_str = "", .timestamp_start = "", .timestamp_end = "",
    .created_time = "", .updated_time = "", .metadata_json = "{}",
}, .{});

// L1 — hybrid search (FTS5 BM25)
const results = try store.searchL1Fts(allocator, "PostgreSQL", 5, .{});
defer { for (results) |r| r.deinit(allocator); allocator.free(results); }

// Recall — the main entry point for prompt injection
var recall = try store.recall(allocator, "database setup", 10, .{});
defer recall.deinit(allocator);

// L1 — delete a memory (soft by default, see "Memory deletion" below)
const deleted = try store.deleteL1("mem-1", .{}, .{});
```

## Memory deletion

Memories can be deleted either **soft** (default) or **hard**, controlled by
`DeleteOptions`:

```zig
const agent_memory = @import("agent_memory");
const DeleteOptions = agent_memory.DeleteOptions;
const iso = agent_memory.IsolationContext{};

// Soft delete (default): the record is marked deleted and disappears from
// search and recall, but the row is kept and can be recovered.
const soft_deleted = try store.deleteL1("mem-1", .{}, iso);

// Explicit soft delete (same as above)
_ = try store.deleteL1("mem-1", .{ .soft = true }, iso);

// Hard delete: the row and its FTS5 entry are physically removed —
// a force-delete that works on live and soft-deleted rows alike.
// This is irreversible.
_ = try store.deleteL1("mem-1", .{ .soft = false }, iso);

// Restore a soft-deleted record (undoes the soft delete).
const restored = try store.restoreL1("mem-1", iso);

// Physically remove ALL soft-deleted records under this isolation context.
// Returns the number of purged rows. After this, restore is no longer possible.
const purged = try store.purgeDeletedL1(iso);
```

**Semantics:**

| Aspect | Soft delete (`.soft = true`) | Hard delete (`.soft = false`) |
|--------|------------------------------|-------------------------------|
| Row in `l1_records` | kept, `deleted = 1` | removed |
| Visible in `searchL1Fts` / `recall` | no | no |
| Recoverable via `restoreL1` | yes | no |
| Removed by `purgeDeletedL1` | yes | n/a (already gone) |
| FTS5 index entry | kept (filtered out at query time) | removed via trigger |

- All delete variants **respect the isolation context** (`team_id`, `agent_id`,
  `user_id`): a record that belongs to a different tenant cannot be deleted.
- Soft delete only affects a **live** row (deleting an already-soft-deleted
  record returns `false`). Hard delete is a **force-delete**: it removes the
  row whether it is live or soft-deleted.
- All delete/restore functions return `true` when a row was affected, `false`
  when no matching record exists (id not found, already deleted/restored, or
  tenant mismatch).
- `upsertL1` on a soft-deleted `record_id` re-creates the row with
  `deleted = 0` — an upsert revives a soft-deleted memory.

### Through the `MemoryStore` vtable / `MemoryContext`

```zig
var mem_ctx = agent_memory.MemoryContext{ .store = store.toMemoryStore(), .iso = .{} };

// Soft delete via MemoryContext (delegates through the vtable)
_ = try mem_ctx.delete("mem-1");

// Hard delete via MemoryContext
_ = try mem_ctx.deleteHard("mem-1");

// Restore
_ = try mem_ctx.restore("mem-1");

// Purge all soft-deleted records in this isolation scope
_ = try mem_ctx.purgeDeleted();
```

## Build

Requires:
- Zig 0.17.0-dev (master)
- No system SQLite needed — the SQLite 3.53.4 amalgamation is vendored in `vendor/` and compiled from source (FTS5 enabled).

```bash
zig build              # Build the library
zig build test         # Run unit tests
zig build test-integration  # Run integration tests (creates temp SQLite DBs)
zig build test-all     # Run all tests
```

## Design document

See [ANALYSIS.md](./ANALYSIS.md) for the full deep analysis of TencentDB Agent Memory and the implementation plan.

## License

MIT

## Acknowledgements

This project is inspired by and ports key concepts from [TencentDB Agent Memory](https://github.com/TencentCloud/TencentDB-Agent-Memory) by Tencent Cloud. The memory pyramid (L0–L3), FTS5 + RRF hybrid search, and L1 extraction pipeline design originate from that project.

The vendored SQLite amalgamation (`vendor/sqlite3.c`, `vendor/sqlite3.h`) is SQLite 3.53.4 (2026-07-24), from <https://sqlite.org/2026/sqlite-amalgamation-3530400.zip>.