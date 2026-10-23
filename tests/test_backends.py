from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from s3_agent_memory.backends import FilesystemBackend
from s3_agent_memory.backends import InMemoryBackend
from s3_agent_memory.backends import ObjectStore
from s3_agent_memory.backends import open_backend
from s3_agent_memory.errors import BackendError
from s3_agent_memory.errors import ConflictError
from s3_agent_memory.errors import InvalidKeyError
from s3_agent_memory.errors import LimitExceededError
from tests.conftest import BUCKET
from tests.conftest import FAKE_KEY_ID
from tests.conftest import FAKE_SECRET


def test_conditional_writes(backend: ObjectStore) -> None:
    etag = backend.put("ns/a.json", b"one", if_none_match=True)
    with pytest.raises(ConflictError):
        backend.put("ns/a.json", b"two", if_none_match=True)
    obj = backend.get("ns/a.json", max_bytes=100)
    assert obj is not None and obj.data == b"one" and obj.etag == etag
    etag2 = backend.put("ns/a.json", b"two", if_match=etag)
    assert etag2 != etag
    with pytest.raises(ConflictError):  # stale ETag
        backend.put("ns/a.json", b"three", if_match=etag)
    with pytest.raises(ConflictError):  # If-Match on a missing key
        backend.put("ns/missing.json", b"x", if_match=etag)
    backend.put("ns/a.json", b"four")  # unconditional overwrite
    current = backend.get("ns/a.json", max_bytes=100)
    assert current is not None and current.data == b"four"


def test_get_missing_delete_and_limits(backend: ObjectStore) -> None:
    assert backend.get("nope/x.json", max_bytes=10) is None
    backend.delete("nope/x.json")  # idempotent
    backend.put("big/x.json", b"x" * 50)
    with pytest.raises(LimitExceededError):
        backend.get("big/x.json", max_bytes=10)
    backend.delete("big/x.json")
    assert backend.get("big/x.json", max_bytes=100) is None


def test_listing(backend: ObjectStore) -> None:
    for key in ("p/a/1.json", "p/a/2.json", "p/a/deep/3.json", "p/b/4.json", "q/5.json"):
        backend.put(key, b"{}")
    assert list(backend.list("p/a/", recursive=False)) == ["p/a/1.json", "p/a/2.json"]
    assert list(backend.list("p/a/")) == ["p/a/1.json", "p/a/2.json", "p/a/deep/3.json"]
    assert len(list(backend.list(""))) == 5
    assert list(backend.list("missing/")) == []
    with pytest.raises(InvalidKeyError):
        list(backend.list("../"))


@pytest.mark.parametrize("key", ["../escape.json", "/abs.json", "a/../../b.json", "a\\..\\b"])
def test_backends_refuse_unsafe_keys(backend: ObjectStore, key: str) -> None:
    with pytest.raises(InvalidKeyError):
        backend.put(key, b"x")
    with pytest.raises(InvalidKeyError):
        backend.get(key, max_bytes=10)
    with pytest.raises(InvalidKeyError):
        backend.delete(key)


def test_filesystem_refuses_symlinks(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.json").write_text("top secret")
    root = tmp_path / "root"
    fs = FilesystemBackend(root)
    (root / "ns").mkdir()
    os.symlink(outside, root / "ns" / "link")
    os.symlink(outside / "secret.json", root / "ns" / "file.json")
    with pytest.raises(InvalidKeyError):
        fs.get("ns/link/secret.json", max_bytes=100)
    with pytest.raises(InvalidKeyError):
        fs.put("ns/link/new.json", b"x")
    with pytest.raises(InvalidKeyError):
        fs.get("ns/file.json", max_bytes=100)
    assert not (outside / "new.json").exists()
    assert list(fs.list("ns/")) == []  # symlinks are invisible to listing
    with pytest.raises(BackendError):
        FilesystemBackend(root / "ns" / "link")


def test_filesystem_atomic_files_and_hidden_entries(tmp_path: Path) -> None:
    fs = FilesystemBackend(tmp_path / "s")
    fs.put("a/b.json", b"{}")
    (tmp_path / "s" / "a" / ".tmp-leftover").write_text("partial")
    (tmp_path / "s" / "a" / "bad name.json").write_text("{}")
    assert list(fs.list("")) == ["a/b.json"]
    assert sorted(p.name for p in (tmp_path / "s").iterdir()) == [".lock", "a"]
    with pytest.raises(InvalidKeyError):
        (tmp_path / "s" / "dir.json").mkdir()
        fs.get("dir.json", max_bytes=10)
    assert repr(fs).startswith("FilesystemBackend(")


def test_s3_backend_sends_sse_and_hides_credentials(
    s3_backend: ObjectStore, caplog: pytest.LogCaptureFixture
) -> None:
    from s3_agent_memory.backends.s3 import S3Backend

    assert isinstance(s3_backend, S3Backend)
    caplog.set_level(logging.DEBUG)
    encrypted = S3Backend(BUCKET, client=s3_backend._client, sse="AES256")
    encrypted.put("enc/x.json", b"{}")
    head = s3_backend._client.head_object(Bucket=BUCKET, Key="enc/x.json")
    assert head["ServerSideEncryption"] == "AES256"
    text = repr(encrypted) + caplog.text
    assert FAKE_SECRET not in text and FAKE_KEY_ID not in text
    kms = S3Backend(BUCKET, client=s3_backend._client, sse="aws:kms", kms_key_id="alias/x")
    assert "alias/x" not in repr(kms)
    with pytest.raises(BackendError):
        S3Backend(BUCKET, client=s3_backend._client, sse="rot13")
    with pytest.raises(BackendError):
        S3Backend(BUCKET, client=s3_backend._client, kms_key_id="k")
    with pytest.raises(BackendError):
        S3Backend("a/b", client=s3_backend._client)


def test_s3_backend_errors_are_mapped(s3_backend: ObjectStore) -> None:
    from s3_agent_memory.backends.s3 import S3Backend

    assert isinstance(s3_backend, S3Backend)
    missing = S3Backend("no-such-bucket-here", client=s3_backend._client)
    with pytest.raises(BackendError):
        missing.put("a/b.json", b"x")
    with pytest.raises(BackendError):
        list(missing.list("a/"))
    with pytest.raises(BackendError):
        missing.get("a/b.json", max_bytes=10)
    with pytest.raises(BackendError):
        missing.delete("a/b.json")


def test_s3_backend_builds_its_own_client(aws_env: None) -> None:
    from moto import mock_aws

    from s3_agent_memory.backends.s3 import S3Backend

    with mock_aws():
        backend = S3Backend(BUCKET, region_name="us-east-1", endpoint_url=None)
        backend._client.create_bucket(Bucket=BUCKET)
        backend.put("x/y.json", b"1", if_none_match=True)
        assert list(backend.list("x/")) == ["x/y.json"]


def test_open_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, aws_env: None) -> None:
    from moto import mock_aws

    backend, prefix = open_backend("memory://")
    assert isinstance(backend, InMemoryBackend) and prefix == ""
    backend, prefix = open_backend(f"file://{tmp_path}/a")
    assert isinstance(backend, FilesystemBackend) and backend.root == (tmp_path / "a").resolve()
    backend, _ = open_backend(str(tmp_path / "b"))
    assert isinstance(backend, FilesystemBackend)
    monkeypatch.setenv("S3_AGENT_MEMORY_SSE", "AES256")
    with mock_aws():
        backend, prefix = open_backend("s3://bucket-x/agents/prod/")
        assert prefix == "agents/prod"
        assert "bucket-x" in repr(backend) and "AES256" in repr(backend)
    for bad in ("s3://key:secret@bucket/x", "s3://bucket/x?endpoint=evil", "ftp://x/y"):
        with pytest.raises(BackendError):
            open_backend(bad)


def test_s3_listing_skips_directory_markers(s3_backend: ObjectStore) -> None:
    from s3_agent_memory.backends.s3 import S3Backend

    assert isinstance(s3_backend, S3Backend)
    s3_backend._client.put_object(Bucket=BUCKET, Key="ns/_tombstones/", Body=b"")
    s3_backend.put("ns/_tombstones/a.json", b"{}")
    assert list(s3_backend.list("ns/_tombstones/", recursive=False)) == ["ns/_tombstones/a.json"]
