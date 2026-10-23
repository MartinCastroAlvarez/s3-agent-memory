"""Object-store backends, and :func:`open_backend` for ``s3://``, ``file://``, ``memory://``."""

from __future__ import annotations

import os
from urllib.parse import unquote
from urllib.parse import urlsplit

from s3_agent_memory.backends.base import ObjectStore
from s3_agent_memory.backends.base import StoredObject
from s3_agent_memory.backends.filesystem import FilesystemBackend
from s3_agent_memory.backends.memory import InMemoryBackend
from s3_agent_memory.errors import BackendError

__all__ = [
    "FilesystemBackend",
    "InMemoryBackend",
    "ObjectStore",
    "StoredObject",
    "open_backend",
]


def open_backend(url: str) -> tuple[ObjectStore, str]:
    """Return ``(backend, prefix)`` for a store URL.

    * ``s3://bucket/optional/prefix`` - S3 or any S3-compatible endpoint
      (``S3_AGENT_MEMORY_ENDPOINT_URL`` or boto3's ``AWS_ENDPOINT_URL_S3`` for
      MinIO / R2; ``S3_AGENT_MEMORY_SSE`` and ``S3_AGENT_MEMORY_KMS_KEY_ID``
      for server-side encryption).
    * ``file:///abs/dir``, ``file://./rel/dir`` or a plain path - local directory.
    * ``memory://`` - in-process dict (lost on exit).
    """
    parts = urlsplit(url)
    if parts.scheme == "s3":
        from s3_agent_memory.backends.s3 import S3Backend

        if parts.query or parts.fragment or "@" in parts.netloc:
            raise BackendError("s3:// URLs take no credentials, query or fragment")
        backend = S3Backend(
            parts.netloc,
            endpoint_url=os.environ.get("S3_AGENT_MEMORY_ENDPOINT_URL") or None,
            region_name=os.environ.get("S3_AGENT_MEMORY_REGION") or None,
            sse=os.environ.get("S3_AGENT_MEMORY_SSE") or None,
            kms_key_id=os.environ.get("S3_AGENT_MEMORY_KMS_KEY_ID") or None,
        )
        return backend, parts.path.strip("/")
    if parts.scheme == "memory":
        return InMemoryBackend(), ""
    if parts.scheme == "file":
        path = unquote(parts.netloc + parts.path)
        return FilesystemBackend(path), ""
    if parts.scheme == "" or (len(parts.scheme) == 1 and os.name == "nt"):  # pragma: no branch
        return FilesystemBackend(url), ""
    raise BackendError(f"unsupported store URL scheme {parts.scheme!r} (use s3, file or memory)")
