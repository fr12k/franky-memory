# franky-memory

Embedded persistent memory store for the [franky](https://github.com/fr12k/franky) LLM agent framework.

A Zig implementation of the "Database Memory" concept from [TencentDB Agent Memory](https://github.com/TencentCloud/TencentDB-Agent-Memory), adapted for single-binary, embedded operation — no Node.js sidecar, no external HTTP gateway, no separate API keys.

## What it does

Provides layered persistent memory for LLM agents:

| Layer | Stores | Storage | Purpose |
|-------|--------|---------|---------|
| **L0 Conversation** | Raw messages (role, text, timestamp) | SQLite rows | Verbatim conversation log |
| **L1 Atom** | Facts, preferences, decisions | SQLite rows + FTS5 index | Compact, self-contained memory atoms |
| **L2 Scenario** | Per-project context blocks | Markdown files | Restore working context |
| **L3 Persona** | Long-term user profile | Markdown file | Rapid context bootstrap |

## Architecture

```
franky Agent Loop
  │
  │ 1. Turn completes → capture messages to L0 (SQLite INSERT, no LLM)
  │ 2. Before next prompt → recall L1/L2/L3, inject as [Memory Context]
  │ 3. On session end → L1 extraction pipeline (calls franky's LLM)
  ▼
SQLite database (~/.franky/memory.db)
  + scene_blocks/*.md   (L2)
  + persona.md          (L3)
```

**Persistence** is SQLite + FTS5 (full-text search with BM25 ranking).
**Intelligence** (L1 extraction) calls franky's own `ai.registry.Registry` — no separate LLM client.

## Usage

### As a dependency (in franky's `build.zig.zon`)

```zig
.dependencies = .{
    .agent_memory = .{
        .url = "https://github.com/franky-agent/franky-memory/archive/refs/tags/v0.1.0.tar.gz",
        .hash = "agent_memory-0.0.1-<hash>",
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
    "/home/user/.franky/memory",
);
defer store.deinit();

// L0 — capture raw conversation (no LLM, cheap)
try store.addConversation(&[_]agent_memory.L0Record{
    .{ .id = "msg-1", .session_key = "sk1", .session_id = "s1",
       .role = "user", .message_text = "I use PostgreSQL",
       .recorded_at = "2025-01-15T10:00:00Z", .timestamp = 1736932800000 },
}, .{ .session_id = "s1" });

// L1 — store extracted memory
_ = try store.upsertL1(.{
    .record_id = "mem-1", .content = "User uses PostgreSQL",
    .type = .episodic, .priority = 75, .scene_name = "database",
    .session_key = "sk1", .session_id = "s1",
    .team_id = "default", .task_id = "", .user_id = "default",
    .agent_id = "default", .version = 1,
    .timestamp_str = "", .timestamp_start = "", .timestamp_end = "",
    .created_time = "", .updated_time = "", .metadata_json = "{}",
}, null, .{});

// L1 — hybrid search (FTS5 BM25)
const results = try store.searchL1Fts(allocator, "PostgreSQL", 5, .{});
defer { for (results) |r| r.deinit(allocator); allocator.free(results); }

// L3 — persona
try store.writeCore("User's name is Alice. Prefers concise answers.", .{});

// Recall — the main entry point for prompt injection
var recall = try store.recall(allocator, "database setup", 10, .{});
defer recall.deinit(allocator);
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