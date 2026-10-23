"""s3-agent-memory: long-term memory for AI agents as plain objects in S3.

>>> from s3_agent_memory import MemoryStore
>>> store = MemoryStore.from_url("memory://")
>>> m = store.remember("users/alice", "Alice prefers metric units", kind="procedural")
>>> [h.memory.id == m.id for h in store.recall("users/alice", "which units?")]
[True]
"""

from __future__ import annotations

__version__ = "0.1.0"

from s3_agent_memory.backends import FilesystemBackend
from s3_agent_memory.backends import InMemoryBackend
from s3_agent_memory.backends import ObjectStore
from s3_agent_memory.backends import open_backend
from s3_agent_memory.errors import AgentMemoryError
from s3_agent_memory.errors import ConflictError
from s3_agent_memory.errors import InvalidKeyError
from s3_agent_memory.errors import LimitExceededError
from s3_agent_memory.errors import NotFoundError
from s3_agent_memory.errors import ValidationError
from s3_agent_memory.limits import Limits
from s3_agent_memory.model import Memory
from s3_agent_memory.model import RecallHit
from s3_agent_memory.ranking import Ranking
from s3_agent_memory.store import UNSET
from s3_agent_memory.store import MemoryStore

__all__ = [
    "UNSET",
    "AgentMemoryError",
    "ConflictError",
    "FilesystemBackend",
    "InMemoryBackend",
    "InvalidKeyError",
    "LimitExceededError",
    "Limits",
    "Memory",
    "MemoryStore",
    "NotFoundError",
    "ObjectStore",
    "Ranking",
    "RecallHit",
    "ValidationError",
    "__version__",
    "open_backend",
]
