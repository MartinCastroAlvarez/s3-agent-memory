"""Exceptions raised by s3-agent-memory. All derive from :class:`AgentMemoryError`."""

from __future__ import annotations


class AgentMemoryError(Exception):
    """Base class for every error this package raises on purpose."""


class InvalidKeyError(AgentMemoryError, ValueError):
    """A namespace, id, prefix or object key failed validation (traversal, injection, length)."""


class ValidationError(AgentMemoryError, ValueError):
    """A memory, query or import line is malformed."""


class LimitExceededError(AgentMemoryError, ValueError):
    """A size or count limit from :class:`~s3_agent_memory.limits.Limits` was hit."""


class ConflictError(AgentMemoryError):
    """A conditional write lost the race (S3 ``412 Precondition Failed``) or retries ran out."""


class NotFoundError(AgentMemoryError, KeyError):
    """The memory does not exist in the namespace."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else "not found"


class BackendError(AgentMemoryError):
    """The object store failed in a way the caller cannot fix by retrying the same call."""
