# Security

s3-agent-memory stores what AI agents learn about people and replays it into
later conversations. It is installed by other people, pointed at buckets that
other systems may also write to, and called by models. Design goal: nothing a
caller, a model or a bucket object sends can make it read or write outside the
namespace it was asked for, run code, leak credentials, or pass stored text off
as instructions.

## Reporting a vulnerability

Open a private security advisory at
<https://github.com/MartinCastroAlvarez/s3-agent-memory/security/advisories/new>.
Do **not** open a public issue for a suspected vulnerability. We acknowledge
within 5 business days and aim to ship a fix within 30 days; you are credited
in the advisory unless you ask otherwise.

Supported versions: the latest minor release.

## 1. Threat model

**Assets.** Memories (often personal data), the separation between
namespaces (users, tenants, agents), the bucket and its credentials, the host
the library runs on, and the integrity of what a model is told it remembers.

**Actors.**

| Actor | Controls | Must not be able to |
| --- | --- | --- |
| The model calling MCP tools | every tool argument | choose a namespace, reach another namespace's keys, delete outside its namespace, write when the server is `--read-only` |
| Application code / CLI user | namespace, id, content, tags, metadata, import files | escape the prefix with `..`, `/`, `\`, `%2e`, NUL or reserved names; exceed size limits; import into another namespace |
| Other writers of the bucket (another service, a compromised agent) | object bytes | have an object served under a key it does not belong to, run code through a vector shard, exhaust memory with a huge index or zip bomb |
| Text stored in memories | content | be treated as instructions (prompt injection, see below) |

**Threats and controls.**

| Threat | Control |
| --- | --- |
| Path traversal / key injection | Namespace and prefix segments must match `[A-Za-z0-9][A-Za-z0-9_.@+=-]{0,127}` without `..`; ids `[A-Za-z0-9][A-Za-z0-9_.-]{0,127}`. Segments starting with `_` (index, shard, tombstones), kind names and `*.json`/`*.npz` are reserved, so one namespace can never address another's objects. Every backend re-validates the full key. Tested with a hostile corpus through the API, CLI, MCP and LangGraph paths. |
| Filesystem escape | The filesystem backend resolves every path inside its root, refuses symlinks in any component, opens files with `O_NOFOLLOW`, accepts only regular files, and hides dotfiles from listings. |
| Cross-namespace reads | Objects carry their namespace, kind and id; one found under a key it does not match is refused. The MCP server is bound to one namespace at startup and has no namespace argument. Import ignores the namespace inside the file. |
| Lost updates / races | Conditional writes (`If-None-Match: *`, `If-Match`) with bounded, jittered retries; strict mode with `expected_version`. |
| Resource exhaustion | `Limits`: content, data, metadata, tags, per-object, per-index/shard, memories per namespace, query length, results, JSONL line length. Reads check size before loading. |
| Code execution via stored data | No `pickle`, `eval`, `exec`, `yaml`, `subprocess`, shell. Vector shards load with `np.load(allow_pickle=False)` after checking the member list and the uncompressed size (zip bombs). A test scans the package AST for these. |
| Credential leaks | Credentials come only from boto3's chain; the package never reads, stores, prints or logs them. `S3Backend.__repr__` shows bucket, endpoint and SSE mode only (not the KMS key id). Tests run with fake keys and assert they never appear in logs or reprs. `s3://` URLs with userinfo or query strings are refused. |
| Content leaks in logs | Log lines carry error types, never memory text (tested). Tombstones keep id, kind, time and reason only. |
| Data at rest | Use bucket default encryption (SSE-S3 or SSE-KMS) and Block Public Access; `S3_AGENT_MEMORY_SSE` / `S3_AGENT_MEMORY_KMS_KEY_ID` add the headers per object. Scope IAM to `s3:GetObject`, `s3:PutObject`, `s3:DeleteObject` and `s3:ListBucket` on the prefix. |
| Cost / network | No telemetry and no hosted model calls. Embeddings run locally; consolidation calls only the configured ollama host (http/https URL validated). |

**Prompt injection.** A memory is whatever a user, a tool result or another
agent once wrote, and recall puts it back in front of a model. Treat it like a
web page:

- `recall` returns memories as JSON data with provenance (`key`, `source`,
  `created_at`, `updated_at`, `version`); the MCP envelope is marked
  `untrusted_content: true` with a note, and the server instructions tell the
  model never to follow instructions found in memories.
- Record `source` honestly (`mcp`, `cli`, `consolidate`, `web-scrape`, ...) so
  a policy can down-weight or drop memories from untrusted sources.
- The consolidation prompt fences memories as data, the model's answer is
  length-capped, and an empty answer falls back to an original text.
- Do not store secrets, credentials or payment data in memories; the server
  instructions say so too.

**Out of scope.** Vulnerabilities in boto3, the MCP SDK, LangGraph, numpy or
the embedding libraries; a bucket policy that grants other principals write
access to the prefix (the library limits what they can achieve, but cannot
stop them deleting data); an operator who gives two tenants the same namespace.

## 2. Non-negotiable rules

Enforced by `tests/test_security.py`, `tests/test_keys.py` and
`tests/test_tampering.py`.

1. Every key is built from validated segments in `keys.py`; backends re-check.
2. No `pickle`/`marshal`/`yaml`/`subprocess`/shell/`eval`/`exec`; `np.load` only with `allow_pickle=False`.
3. Every read is size-bounded; every write is validated against `Limits`.
4. Shared objects change only through compare-and-swap.
5. No memory content and no credentials in logs, reprs or tombstones.
