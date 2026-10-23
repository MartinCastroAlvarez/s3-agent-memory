"""Object-key layout and validation.

Layout::

    <prefix>/<namespace...>/<kind>/<id>.json   one memory
    <prefix>/<namespace...>/_index.json        BM25 index for the namespace
    <prefix>/<namespace...>/_vectors.npz       optional embedding shard
    <prefix>/<namespace...>/_tombstones/<id>.json

Every segment that reaches a key is checked against a strict allow-list, so a
namespace or id can never contain ``/``, ``..``, ``\\``, NUL, ``%`` escapes,
whitespace or start with ``_`` (reserved for the files above). Namespace
segments also cannot be a memory kind or end in ``.json``, so one namespace can
never address another namespace's objects.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from collections.abc import Sequence
from typing import Literal

from s3_agent_memory.errors import InvalidKeyError

Kind = Literal["semantic", "episodic", "procedural"]
KINDS: tuple[Kind, ...] = ("semantic", "episodic", "procedural")

_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@+=-]{0,127}")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_OBJECT_SEGMENT = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.@+=-]{0,132}")
MAX_NAMESPACE_DEPTH = 8
MAX_PREFIX_DEPTH = 8
MAX_KEY_CHARS = 1024

INDEX_NAME = "_index.json"
VECTORS_NAME = "_vectors.npz"
TOMBSTONES_DIR = "_tombstones"

Namespace = tuple[str, ...]


def _segment(value: object, what: str) -> str:
    if not isinstance(value, str):
        raise InvalidKeyError(f"{what} must be a string, got {type(value).__name__}")
    if not _SEGMENT.fullmatch(value) or ".." in value:
        raise InvalidKeyError(
            f"invalid {what} {value!r}: use 1-128 of [A-Za-z0-9_.@+=-], "
            "starting with a letter or digit, without '..'"
        )
    return value


def validate_namespace(namespace: str | Sequence[str]) -> Namespace:
    """Normalise ``"users/alice"`` or ``("users", "alice")`` to a validated tuple."""
    parts: Sequence[object]
    if isinstance(namespace, str):
        parts = namespace.split("/")
    elif isinstance(namespace, (tuple, list)):
        parts = namespace
    else:
        raise InvalidKeyError(
            f"namespace must be a string or tuple, got {type(namespace).__name__}"
        )
    if not parts or len(parts) > MAX_NAMESPACE_DEPTH:
        raise InvalidKeyError(f"namespace must have 1-{MAX_NAMESPACE_DEPTH} segments")
    out = []
    for part in parts:
        seg = _segment(part, "namespace segment")
        if seg in KINDS:
            raise InvalidKeyError(f"namespace segment {seg!r} is reserved (a memory kind)")
        if seg.lower().endswith(".json") or seg.lower().endswith(".npz"):
            raise InvalidKeyError(f"namespace segment {seg!r} may not end in .json/.npz")
        out.append(seg)
    return tuple(out)


def validate_prefix(prefix: str) -> str:
    """Validate a bucket prefix such as ``"agents/prod"``; ``""`` means the bucket root."""
    if not isinstance(prefix, str):
        raise InvalidKeyError("prefix must be a string")
    prefix = prefix.strip("/")
    if not prefix:
        return ""
    parts = prefix.split("/")
    if len(parts) > MAX_PREFIX_DEPTH:
        raise InvalidKeyError(f"prefix may have at most {MAX_PREFIX_DEPTH} segments")
    for part in parts:
        _segment(part, "prefix segment")
    return prefix


def validate_kind(kind: object) -> Kind:
    for known in KINDS:
        if kind == known:
            return known
    raise InvalidKeyError(f"kind must be one of {', '.join(KINDS)}; got {kind!r}")


def validate_id(memory_id: object) -> str:
    if not isinstance(memory_id, str) or not _ID.fullmatch(memory_id) or ".." in memory_id:
        raise InvalidKeyError(
            f"invalid memory id {memory_id!r}: use 1-128 of [A-Za-z0-9_.-], "
            "starting with a letter or digit, without '..'"
        )
    if memory_id.lower().endswith(".json"):
        raise InvalidKeyError("memory id may not end in .json")
    return memory_id


def validate_object_key(key: str) -> str:
    """Defence in depth for backends: every key they touch is relative and segment-safe."""
    if not isinstance(key, str) or not key or len(key) > MAX_KEY_CHARS:
        raise InvalidKeyError("object key must be a non-empty string of at most 1024 chars")
    for part in key.split("/"):
        if not _OBJECT_SEGMENT.fullmatch(part) or ".." in part:
            raise InvalidKeyError(f"unsafe object key {key!r}")
    return key


def validate_list_prefix(prefix: str) -> str:
    """A listing prefix: ``""`` or a safe key ending in ``/``."""
    if prefix == "":
        return prefix
    if not prefix.endswith("/"):
        raise InvalidKeyError("list prefix must end with '/'")
    validate_object_key(prefix[:-1])
    return prefix


def _join(*parts: str | Iterable[str]) -> str:
    flat: list[str] = []
    for part in parts:
        if isinstance(part, str):
            if part:
                flat.append(part)
        else:
            flat.extend(part)
    return "/".join(flat)


def namespace_dir(prefix: str, namespace: Namespace) -> str:
    return _join(prefix, namespace) + "/"


def memory_key(prefix: str, namespace: Namespace, kind: str, memory_id: str) -> str:
    return _join(prefix, namespace, validate_kind(kind), validate_id(memory_id) + ".json")


def kind_dir(prefix: str, namespace: Namespace, kind: str) -> str:
    return _join(prefix, namespace, validate_kind(kind)) + "/"


def index_key(prefix: str, namespace: Namespace) -> str:
    return _join(prefix, namespace, INDEX_NAME)


def vectors_key(prefix: str, namespace: Namespace) -> str:
    return _join(prefix, namespace, VECTORS_NAME)


def tombstone_key(prefix: str, namespace: Namespace, memory_id: str) -> str:
    return _join(prefix, namespace, TOMBSTONES_DIR, validate_id(memory_id) + ".json")


def tombstones_dir(prefix: str, namespace: Namespace) -> str:
    return _join(prefix, namespace, TOMBSTONES_DIR) + "/"


def namespace_from_index_key(prefix: str, key: str) -> Namespace | None:
    """Inverse of :func:`index_key`; ``None`` for keys that are not a valid index object."""
    head = prefix + "/" if prefix else ""
    if not key.startswith(head) or not key.endswith("/" + INDEX_NAME):
        return None
    middle = key[len(head) : -len(INDEX_NAME) - 1]
    try:
        return validate_namespace(tuple(middle.split("/")))
    except InvalidKeyError:
        return None
