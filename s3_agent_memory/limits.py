"""Size and count limits. Everything read from the bucket or sent by a caller is bounded."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Limits:
    """Hard limits enforced on writes and reads. Override per :class:`MemoryStore`."""

    max_content_bytes: int = 16_384
    """UTF-8 size of ``Memory.content``."""
    max_data_bytes: int = 32_768
    """Serialised size of the optional structured ``Memory.data`` payload."""
    max_metadata_bytes: int = 4_096
    """Serialised size of ``Memory.metadata``."""
    max_tags: int = 32
    max_tag_chars: int = 64
    max_source_chars: int = 256
    max_object_bytes: int = 1_048_576
    """Largest memory object accepted when reading from the bucket."""
    max_index_bytes: int = 64 * 1_048_576
    """Largest index (or vector shard) object accepted when reading."""
    max_memories: int = 100_000
    """Memories per namespace; a ``remember`` beyond this is refused."""
    max_query_chars: int = 2_000
    max_results: int = 100
    max_jsonl_line_bytes: int = 131_072
