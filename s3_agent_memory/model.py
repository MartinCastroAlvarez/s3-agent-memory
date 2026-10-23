"""The :class:`Memory` record and its JSON form."""

from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from datetime import UTC
from datetime import datetime
from typing import Any

from s3_agent_memory.errors import LimitExceededError
from s3_agent_memory.errors import ValidationError
from s3_agent_memory.keys import Kind
from s3_agent_memory.keys import Namespace
from s3_agent_memory.keys import validate_id
from s3_agent_memory.keys import validate_kind
from s3_agent_memory.keys import validate_namespace
from s3_agent_memory.limits import Limits

SCHEMA = 1
_CONTROL = {chr(c) for c in range(32)} | {"\x7f"}


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    """Time-ordered, unguessable id: 12 hex digits of milliseconds + 12 random hex digits."""
    return f"{int(time.time() * 1000):012x}{secrets.token_hex(6)}"


def iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def parse_time(value: object, what: str) -> datetime:
    if not isinstance(value, str):
        raise ValidationError(f"{what} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError(f"{what} is not ISO-8601: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _json_size(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())


def _check_text(value: object, what: str, max_chars: int) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{what} must be a string")
    if len(value) > max_chars:
        raise LimitExceededError(f"{what} longer than {max_chars} characters")
    if any(ch in _CONTROL for ch in value):
        raise ValidationError(f"{what} may not contain control characters")
    return value


@dataclass(frozen=True)
class Memory:
    """One long-term memory.

    ``kind`` is ``semantic`` (facts about the user or world), ``episodic``
    (something that happened) or ``procedural`` (instructions, preferences,
    how-tos). ``content`` is the searchable text; ``data`` is an optional JSON
    payload (used by the LangGraph adapter). ``etag`` is filled when read from
    a backend and is not serialised.
    """

    id: str
    namespace: Namespace
    kind: Kind
    content: str
    created_at: datetime
    updated_at: datetime
    source: str | None = None
    importance: float = 0.5
    tags: tuple[str, ...] = ()
    expires_at: datetime | None = None
    data: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    version: int = 1
    etag: str | None = field(default=None, compare=False)

    # -- validation ---------------------------------------------------------- #
    def validated(self, limits: Limits) -> Memory:
        """Return ``self`` after checking every field against ``limits`` (raises otherwise)."""
        validate_id(self.id)
        validate_namespace(self.namespace)
        validate_kind(self.kind)
        if not isinstance(self.content, str):
            raise ValidationError("content must be a string")
        if len(self.content.encode()) > limits.max_content_bytes:
            raise LimitExceededError(f"content larger than {limits.max_content_bytes} bytes")
        if "\x00" in self.content:
            raise ValidationError("content may not contain NUL")
        if self.source is not None:
            _check_text(self.source, "source", limits.max_source_chars)
        if isinstance(self.importance, bool) or not isinstance(self.importance, int | float):
            raise ValidationError("importance must be a number")
        if not 0.0 <= float(self.importance) <= 1.0:
            raise ValidationError("importance must be between 0 and 1")
        if len(self.tags) > limits.max_tags:
            raise LimitExceededError(f"more than {limits.max_tags} tags")
        for tag in self.tags:
            if not _check_text(tag, "tag", limits.max_tag_chars).strip():
                raise ValidationError("tags may not be blank")
        if self.data is not None:
            if not isinstance(self.data, dict):
                raise ValidationError("data must be a JSON object")
            if _json_size(self.data) > limits.max_data_bytes:
                raise LimitExceededError(f"data larger than {limits.max_data_bytes} bytes")
        if not isinstance(self.metadata, dict):
            raise ValidationError("metadata must be a JSON object")
        if _json_size(self.metadata) > limits.max_metadata_bytes:
            raise LimitExceededError(f"metadata larger than {limits.max_metadata_bytes} bytes")
        if not isinstance(self.version, int) or self.version < 1:
            raise ValidationError("version must be a positive integer")
        return self

    def is_expired(self, now: datetime | None = None) -> bool:
        return self.expires_at is not None and self.expires_at <= (now or utcnow())

    def with_etag(self, etag: str | None) -> Memory:
        return replace(self, etag=etag)

    # -- JSON ---------------------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "id": self.id,
            "namespace": list(self.namespace),
            "kind": self.kind,
            "content": self.content,
            "source": self.source,
            "importance": self.importance,
            "tags": list(self.tags),
            "created_at": iso(self.created_at),
            "updated_at": iso(self.updated_at),
            "expires_at": iso(self.expires_at) if self.expires_at else None,
            "data": self.data,
            "metadata": self.metadata,
            "version": self.version,
        }

    def to_bytes(self) -> bytes:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":")).encode()

    @classmethod
    def from_dict(cls, raw: object, limits: Limits, *, etag: str | None = None) -> Memory:
        if not isinstance(raw, dict):
            raise ValidationError("memory must be a JSON object")
        if raw.get("schema", SCHEMA) != SCHEMA:
            raise ValidationError(f"unsupported memory schema {raw.get('schema')!r}")
        try:
            tags = raw.get("tags") or []
            if not isinstance(tags, list):
                raise ValidationError("tags must be a list")
            namespace = raw.get("namespace")
            if not isinstance(namespace, list | tuple | str):
                raise ValidationError("namespace must be a list")
            memory = cls(
                id=raw["id"],
                namespace=validate_namespace(namespace),
                kind=raw["kind"],
                content=raw["content"],
                source=raw.get("source"),
                importance=raw.get("importance", 0.5),
                tags=tuple(tags),
                created_at=parse_time(raw["created_at"], "created_at"),
                updated_at=parse_time(raw["updated_at"], "updated_at"),
                expires_at=(
                    parse_time(raw["expires_at"], "expires_at") if raw.get("expires_at") else None
                ),
                data=raw.get("data"),
                metadata=raw.get("metadata") or {},
                version=raw.get("version", 1),
                etag=etag,
            )
        except KeyError as exc:
            raise ValidationError(f"memory is missing field {exc.args[0]!r}") from exc
        return memory.validated(limits)

    @classmethod
    def from_bytes(cls, data: bytes, limits: Limits, *, etag: str | None = None) -> Memory:
        try:
            raw = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValidationError("memory object is not valid JSON") from exc
        return cls.from_dict(raw, limits, etag=etag)


@dataclass(frozen=True)
class RecallHit:
    """A recalled memory plus how it scored. ``to_dict`` is what agents receive."""

    memory: Memory
    score: float
    components: dict[str, float]
    key: str

    def to_dict(self) -> dict[str, Any]:
        m = self.memory
        return {
            "id": m.id,
            "kind": m.kind,
            "content": m.content,
            "score": round(self.score, 4),
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "importance": m.importance,
            "tags": list(m.tags),
            "provenance": {
                "namespace": "/".join(m.namespace),
                "key": self.key,
                "source": m.source,
                "created_at": iso(m.created_at),
                "updated_at": iso(m.updated_at),
                "version": m.version,
            },
        }
