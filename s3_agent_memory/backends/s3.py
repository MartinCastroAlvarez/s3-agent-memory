"""AWS S3 / MinIO / Cloudflare R2 backend (``pip install "s3-agent-memory[s3]"``).

Uses S3 conditional writes: ``PutObject`` with ``If-None-Match: *`` to create
and ``If-Match: <etag>`` to replace, both answered ``412 Precondition Failed``
when another writer got there first. AWS S3 supports both on general-purpose
buckets since late 2024; MinIO and R2 support them as well.

Credentials come from boto3's normal chain (environment, profile, instance
role). They are never passed through, stored or logged by this package.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING
from typing import Any

from s3_agent_memory.backends.base import ObjectStore
from s3_agent_memory.backends.base import StoredObject
from s3_agent_memory.errors import BackendError
from s3_agent_memory.errors import ConflictError
from s3_agent_memory.errors import LimitExceededError
from s3_agent_memory.keys import validate_list_prefix
from s3_agent_memory.keys import validate_object_key

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client

log = logging.getLogger(__name__)

_CONFLICT_CODES = {"PreconditionFailed", "ConditionalRequestConflict", "412", "409"}
_MISSING_CODES = {"NoSuchKey", "404", "NotFound"}
_SSE_MODES = {"AES256", "aws:kms", "aws:kms:dsse"}


def _code(exc: Exception) -> str:
    response = getattr(exc, "response", None) or {}
    code = str(response.get("Error", {}).get("Code", ""))
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code or str(status or "")


class S3Backend(ObjectStore):
    """Objects in ``bucket``. Pass ``client`` to reuse a configured boto3 S3 client."""

    def __init__(
        self,
        bucket: str,
        *,
        client: S3Client | None = None,
        endpoint_url: str | None = None,
        region_name: str | None = None,
        sse: str | None = None,
        kms_key_id: str | None = None,
    ) -> None:
        if not bucket or "/" in bucket:
            raise BackendError("S3 bucket name must be non-empty and contain no '/'")
        if sse is not None and sse not in _SSE_MODES:
            raise BackendError(f"sse must be one of {sorted(_SSE_MODES)}")
        if kms_key_id and sse not in {"aws:kms", "aws:kms:dsse"}:
            raise BackendError("kms_key_id requires sse='aws:kms' or 'aws:kms:dsse'")
        self.bucket = bucket
        self.endpoint_url = endpoint_url
        self.sse = sse
        self._kms_key_id = kms_key_id
        if client is None:
            try:
                import boto3
                from botocore.config import Config
            except ImportError as exc:  # pragma: no cover - depends on extras
                raise BackendError('S3 support needs: pip install "s3-agent-memory[s3]"') from exc
            config = Config(
                retries={"mode": "standard", "max_attempts": 5},
                s3={"addressing_style": "path"} if endpoint_url else {},
            )
            client = boto3.client(
                "s3", endpoint_url=endpoint_url, region_name=region_name, config=config
            )
        self._client = client

    def __repr__(self) -> str:
        # Never include credentials or the KMS key id.
        return (
            f"S3Backend(bucket={self.bucket!r}, endpoint_url={self.endpoint_url!r}, "
            f"sse={self.sse!r})"
        )

    def get(self, key: str, *, max_bytes: int) -> StoredObject | None:
        validate_object_key(key)
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            if _code(exc) in _MISSING_CODES:
                return None
            raise BackendError(f"get {key}: {_code(exc) or type(exc).__name__}") from exc
        body = response["Body"]
        try:
            if int(response.get("ContentLength", 0)) > max_bytes:
                raise LimitExceededError(f"{key}: larger than {max_bytes} bytes")
            data = body.read(max_bytes + 1)
        finally:
            body.close()
        if len(data) > max_bytes:
            raise LimitExceededError(f"{key}: larger than {max_bytes} bytes")
        return StoredObject(data, str(response["ETag"]))

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
        kwargs: dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": key,
            "Body": data,
            "ContentType": content_type,
        }
        if if_none_match:
            kwargs["IfNoneMatch"] = "*"
        if if_match is not None:
            kwargs["IfMatch"] = if_match
        if self.sse:
            kwargs["ServerSideEncryption"] = self.sse
            if self._kms_key_id:
                kwargs["SSEKMSKeyId"] = self._kms_key_id
        try:
            response = self._client.put_object(**kwargs)
        except Exception as exc:
            code = _code(exc)
            if code in _CONFLICT_CODES or (if_match is not None and code in _MISSING_CODES):
                raise ConflictError(f"{key}: conditional write lost ({code})") from exc
            raise BackendError(f"put {key}: {code or type(exc).__name__}") from exc
        return str(response["ETag"])

    def delete(self, key: str) -> None:
        validate_object_key(key)
        try:
            self._client.delete_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            if _code(exc) in _MISSING_CODES:
                return
            raise BackendError(f"delete {key}: {_code(exc) or type(exc).__name__}") from exc

    def list(self, prefix: str, *, recursive: bool = True) -> Iterator[str]:
        validate_list_prefix(prefix)
        kwargs: dict[str, Any] = {"Bucket": self.bucket, "Prefix": prefix}
        if not recursive:
            kwargs["Delimiter"] = "/"
        paginator = self._client.get_paginator("list_objects_v2")
        keys: list[str] = []
        try:
            for page in paginator.paginate(**kwargs):
                keys.extend(obj["Key"] for obj in page.get("Contents", []) if "Key" in obj)
        except Exception as exc:
            raise BackendError(f"list {prefix}: {_code(exc) or type(exc).__name__}") from exc
        yield from sorted(keys)
