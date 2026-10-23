"""In-process backend: a dict behind a lock. For tests, notebooks and ephemeral agents."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Iterator

from s3_agent_memory.backends.base import ObjectStore
from s3_agent_memory.backends.base import StoredObject
from s3_agent_memory.errors import ConflictError
from s3_agent_memory.errors import LimitExceededError
from s3_agent_memory.keys import validate_list_prefix
from s3_agent_memory.keys import validate_object_key


def etag_of(data: bytes) -> str:
    """Content-derived ETag (quoted, like S3's)."""
    return '"' + hashlib.sha256(data).hexdigest()[:32] + '"'


class InMemoryBackend(ObjectStore):
    def __init__(self) -> None:
        self._objects: dict[str, StoredObject] = {}
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"InMemoryBackend(objects={len(self._objects)})"

    def get(self, key: str, *, max_bytes: int) -> StoredObject | None:
        validate_object_key(key)
        with self._lock:
            obj = self._objects.get(key)
        if obj is not None and len(obj.data) > max_bytes:
            raise LimitExceededError(f"{key}: {len(obj.data)} bytes > limit {max_bytes}")
        return obj

    def put(
        self,
        key: str,
        data: bytes,
        *,
        if_match: str | None = None,
        if_none_match: bool = False,
        content_type: str = "application/json",
    ) -> str:
        validate_object_key(key)
        with self._lock:
            current = self._objects.get(key)
            if if_none_match and current is not None:
                raise ConflictError(f"{key} already exists")
            if if_match is not None and (current is None or current.etag != if_match):
                raise ConflictError(f"{key} changed (ETag mismatch)")
            etag = etag_of(data)
            self._objects[key] = StoredObject(bytes(data), etag)
            return etag

    def delete(self, key: str) -> None:
        validate_object_key(key)
        with self._lock:
            self._objects.pop(key, None)

    def list(self, prefix: str, *, recursive: bool = True) -> Iterator[str]:
        validate_list_prefix(prefix)
        with self._lock:
            keys = sorted(k for k in self._objects if k.startswith(prefix))
        for key in keys:
            if recursive or "/" not in key[len(prefix) :]:
                yield key
