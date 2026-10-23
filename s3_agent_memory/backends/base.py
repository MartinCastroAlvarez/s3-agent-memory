"""The object-store contract every backend implements.

Four operations, with S3 semantics:

* ``get`` returns the bytes and an opaque ETag, or ``None`` when the key is absent.
* ``put`` is conditional: ``if_none_match=True`` creates only if absent
  (S3 ``If-None-Match: *``), ``if_match=<etag>`` replaces only if unchanged
  (S3 ``If-Match``). A failed condition raises :class:`ConflictError`.
* ``delete`` is idempotent.
* ``list`` yields keys under a prefix, directly under it when ``recursive=False``.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class StoredObject:
    data: bytes
    etag: str


class ObjectStore(ABC):
    """Minimal S3-shaped key/value store with optimistic concurrency."""

    @abstractmethod
    def get(self, key: str, *, max_bytes: int) -> StoredObject | None:
        """Read ``key``; raise :class:`LimitExceededError` above ``max_bytes``."""

    @abstractmethod
    def put(
        self,
        key: str,
        data: bytes,
        *,
        if_match: str | None = None,
        if_none_match: bool = False,
        content_type: str = "application/json",
    ) -> str:
        """Write ``key`` and return its new ETag (conditionally, see the module docstring)."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove ``key``; a missing key is not an error."""

    @abstractmethod
    def list(self, prefix: str, *, recursive: bool = True) -> Iterator[str]:
        """Yield keys starting with ``prefix`` (``""`` or ending in ``/``), sorted."""
