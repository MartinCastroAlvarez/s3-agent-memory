# Architecture

```text
s3_agent_memory/
  keys.py            key layout + allow-list validation (namespaces, ids, prefixes, object keys)
  limits.py          Limits: every size/count bound
  model.py           Memory (JSON schema 1), RecallHit (what agents receive)
  text.py            tokeniser + Okapi BM25 (pure Python)
  index.py           NamespaceIndex: per-namespace _index.json (term freqs + ranking metadata)
  vectors.py         Embedder protocol, fastembed / sentence-transformers adapters, VectorShard (.npz)
  ranking.py         Ranking: relevance + recency (per-kind half-life) + importance
  store.py           MemoryStore: remember/get/update/forget/list/recall/reindex/export/import
  consolidate.py     duplicate grouping (Jaccard / cosine, union-find) + summarisers (ollama, longest)
  server.py          MCP server (mcp 2.x MCPServer), bound to one namespace
  langgraph_store.py LangGraph BaseStore adapter (S3Store)
  cli.py             s3-agent-memory command
  backends/
    base.py          ObjectStore: get / put(if_match, if_none_match) / delete / list
    memory.py        dict + lock
    filesystem.py    files + flock + atomic rename, confined to its root
    s3.py            boto3, PutObject IfNoneMatch / IfMatch, SSE
```

## Data model

One object per memory, under `<prefix>/<namespace...>/<kind>/<id>.json`.
`kind` is `semantic`, `episodic` or `procedural`. The object is the source of
truth; everything else is derived:

| Object | Holds | Rebuilt by |
| --- | --- | --- |
| `_index.json` | per memory: kind, term frequencies (content + tags), created/updated, importance, tags, expiry, version | `reindex` |
| `_vectors.npz` | ids (`<U128`) + float16 L2-normalised matrix + model name | `reindex` (re-embeds) |
| `_tombstones/<id>.json` | id, kind, deleted_at, reason, version | (history; not derived) |

The index never stores the text, so a reader of the index alone learns terms,
not sentences. A namespace "exists" once it has an index object; `namespaces()`
lists them.

## Concurrency

All shared objects change through `MemoryStore._cas`: GET (data + ETag) ->
apply a pure function -> PUT with `If-Match: <etag>` (or `If-None-Match: *` if
the object did not exist). A `412` makes the loop sleep with jittered
exponential backoff and re-run the function on fresh state, up to
`max_retries` (12). The function must be side-effect free; it returns
`(result, changed)` so no-op changes skip the write.

`remember` order: reserve id in the index (fails if the id exists in any kind)
-> create the object with `If-None-Match: *` (on failure the reservation is
rolled back) -> add the vector. A crash between steps leaves at worst an index
entry without an object (recall skips it) or an object without an index entry
(found by `reindex`). Index entries carry the memory's `version`; an older
version never overwrites a newer one, so racing `update`s converge.

`update` uses `If-Match` on the memory object itself. Without
`expected_version` it retries by re-reading (last writer wins per field); with
it, any concurrent change raises `ConflictError`.

The filesystem backend implements the same contract with an exclusive `flock`
on `<root>/.lock` around check-and-write and atomic `os.replace`, so separate
processes on one host are safe. The in-memory backend uses a lock.

## Recall

1. Read `_index.json` (size-bounded) and keep eligible entries (not expired,
   kind, all requested tags).
2. BM25 over their term frequencies (k1 = 1.2, b = 0.75), normalised to the best
   hit. With an embedder, cosine similarity against the shard; a memory with no
   keyword match is a candidate when similarity >= `min_similarity` (0.35).
3. `score = 0.7 * relevance + 0.15 * recency + 0.15 * importance`; relevance mixes
   BM25 and cosine 50/50 when both exist. Recency halves every 30 days
   (episodic), 180 days (semantic) or never (procedural).
4. Fetch objects best-first until `limit` hits are loaded, skipping any that
   vanished or expired since the index was written.

Cost: one GET for the index (+ one for the shard) and `limit` GETs, whatever
the namespace size. Writes cost 2-3 PUTs. The index grows with the vocabulary
of a namespace (roughly 100-300 bytes per memory); at the default 100 000
memories cap it stays in the tens of MB.

## Interfaces

- **MCP** (`server.py`): `MemoryTools` holds the logic, `build_server` wraps it
  in `MCPServer` tools with annotations. The namespace is fixed per process.
- **LangGraph** (`langgraph_store.py`): `S3Store.batch` maps `GetOp`, `PutOp`,
  `SearchOp`, `ListNamespacesOp` onto `MemoryStore`; `abatch` runs it in a
  thread.
- **CLI** (`cli.py`): one subcommand per `MemoryStore` operation, JSON out.

## Extending

- A new backend implements the four `ObjectStore` methods with the same
  conditional semantics and runs the parametrised backend tests.
- A new embedder is any object with `name: str` and
  `embed(texts) -> list[list[float]]`; changing `name` starts a new shard.
- A new summariser is any object with `merge(texts) -> str`.
