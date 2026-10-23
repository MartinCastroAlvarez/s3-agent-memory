"""Local-directory backend with the same conditional-write semantics as S3.

Each key is a file under ``root``. Writes go to a temporary file that is
``fsync``-ed and atomically renamed into place. Conditional writes and deletes
take an exclusive ``flock`` on ``root/.lock`` (plus a thread lock), so
concurrent processes on one host get compare-and-swap semantics. On platforms
without ``fcntl`` only threads in one process are serialised.

Path confinement: keys are validated segment by segment, the resolved path
must stay inside ``root``, and no component may be a symlink.
"""

from __future__ import annotations

import contextlib
import os
import stat
import sys
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path

from s3_agent_memory.backends.base import ObjectStore
from s3_agent_memory.backends.base import StoredObject
from s3_agent_memory.backends.memory import etag_of
from s3_agent_memory.errors import BackendError
from s3_agent_memory.errors import ConflictError
from s3_agent_memory.errors import InvalidKeyError
from s3_agent_memory.errors import LimitExceededError
from s3_agent_memory.keys import validate_list_prefix
from s3_agent_memory.keys import validate_object_key

if sys.platform != "win32":  # pragma: no branch
    import fcntl

_LOCK_NAME = ".lock"
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


class FilesystemBackend(ObjectStore):
    def __init__(self, root: str | os.PathLike[str], *, create: bool = True) -> None:
        path = Path(root)
        if create:
            path.mkdir(parents=True, exist_ok=True)
        if path.is_symlink() or not path.is_dir():
            raise BackendError(f"filesystem backend root must be a real directory: {path}")
        self.root = path.resolve()
        self._thread_lock = threading.RLock()

    def __repr__(self) -> str:
        return f"FilesystemBackend(root={str(self.root)!r})"

    # -- paths ------------------------------------------------------------- #
    def _path(self, key: str) -> Path:
        validate_object_key(key)
        path = self.root.joinpath(*key.split("/"))
        current = self.root
        for part in key.split("/"):
            current = current / part
            if current.is_symlink():
                raise InvalidKeyError(f"refusing symlink in object path: {key!r}")
        if not path.resolve().is_relative_to(self.root):
            raise InvalidKeyError(f"object key escapes the backend root: {key!r}")
        return path

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        with self._thread_lock:
            if sys.platform == "win32":  # pragma: no cover - no flock on Windows
                yield
                return
            fd = os.open(self.root / _LOCK_NAME, os.O_RDWR | os.O_CREAT | _O_NOFOLLOW, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)

    @staticmethod
    def _read(path: Path, max_bytes: int) -> bytes | None:
        try:
            fd = os.open(path, os.O_RDONLY | _O_NOFOLLOW)
        except FileNotFoundError:
            return None
        except OSError as exc:  # ELOOP from O_NOFOLLOW, EISDIR, ...
            raise InvalidKeyError(f"cannot read {path.name}: {exc.strerror}") from exc
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            os.close(fd)
            raise InvalidKeyError(f"not a regular file: {path.name}")
        with os.fdopen(fd, "rb") as fh:
            if info.st_size > max_bytes:
                raise LimitExceededError(f"{path.name}: {info.st_size} bytes > limit {max_bytes}")
            data = fh.read(min(max_bytes, info.st_size) + 1)
        if len(data) > max_bytes:  # pragma: no cover - file grew while reading
            raise LimitExceededError(f"{path.name}: larger than {max_bytes} bytes")
        return data

    # -- API ---------------------------------------------------------------- #
    def get(self, key: str, *, max_bytes: int) -> StoredObject | None:
        data = self._read(self._path(key), max_bytes)
        return None if data is None else StoredObject(data, etag_of(data))

    def put(
        self,
        key: str,
        data: bytes,
        *,
        if_match: str | None = None,
        if_none_match: bool = False,
        content_type: str = "application/json",
    ) -> str:
        path = self._path(key)
        with self._locked():
            if if_none_match or if_match is not None:
                current = self._read(path, max_bytes=2**62)
                if if_none_match and current is not None:
                    raise ConflictError(f"{key} already exists")
                if if_match is not None and (current is None or etag_of(current) != if_match):
                    raise ConflictError(f"{key} changed (ETag mismatch)")
            path.parent.mkdir(parents=True, exist_ok=True)
            self._path(key)  # re-check: a directory created above must not be a symlink
            fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(data)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, path)
            except BaseException:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(tmp)
                raise
        return etag_of(data)

    def delete(self, key: str) -> None:
        path = self._path(key)
        with self._locked(), contextlib.suppress(FileNotFoundError):
            path.unlink()

    def list(self, prefix: str, *, recursive: bool = True) -> Iterator[str]:
        validate_list_prefix(prefix)
        base = self.root.joinpath(*prefix.rstrip("/").split("/")) if prefix else self.root
        if base.is_symlink() or not base.is_dir():
            return
        found: list[str] = []
        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            rel_dir = Path(dirpath).relative_to(self.root).as_posix()
            if not recursive:
                dirnames.clear()
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for name in filenames:
                if name.startswith(".") or (Path(dirpath) / name).is_symlink():
                    continue
                key = name if rel_dir == "." else f"{rel_dir}/{name}"
                try:
                    validate_object_key(key)
                except InvalidKeyError:
                    continue
                found.append(key)
        yield from sorted(found)
