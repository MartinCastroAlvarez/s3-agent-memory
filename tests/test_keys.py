from __future__ import annotations

import pytest

from s3_agent_memory.errors import InvalidKeyError
from s3_agent_memory.keys import index_key
from s3_agent_memory.keys import memory_key
from s3_agent_memory.keys import namespace_from_index_key
from s3_agent_memory.keys import tombstone_key
from s3_agent_memory.keys import validate_id
from s3_agent_memory.keys import validate_kind
from s3_agent_memory.keys import validate_list_prefix
from s3_agent_memory.keys import validate_namespace
from s3_agent_memory.keys import validate_object_key
from s3_agent_memory.keys import validate_prefix
from s3_agent_memory.keys import vectors_key

HOSTILE_SEGMENTS = [
    "",
    ".",
    "..",
    "../bob",
    "a/../b",
    "..%2fbob",
    "%2e%2e",
    "a\\b",
    "/abs",
    "a\x00b",
    "a b",
    "a\nb",
    "_index.json",
    "_tombstones",
    "-flag",
    "x" * 129,
    "café",
    "a..b",
    "semantic",
    "episodic",
    "procedural",
    "notes.json",
    "vecs.npz",
    "a?b",
    "a#b",
    "a*b",
]


@pytest.mark.parametrize("segment", HOSTILE_SEGMENTS)
def test_hostile_namespace_segments_are_refused(segment: str) -> None:
    with pytest.raises(InvalidKeyError):
        validate_namespace(("users", segment))


@pytest.mark.parametrize(
    "namespace",
    ["../../etc", "users/../bob", "users//bob", "/users/bob", "users/bob/", "a/" * 9 + "a"],
)
def test_hostile_namespace_strings_are_refused(namespace: str) -> None:
    with pytest.raises(InvalidKeyError):
        validate_namespace(namespace)


def test_namespace_forms() -> None:
    assert validate_namespace("users/alice") == ("users", "alice")
    assert validate_namespace(["users", "alice"]) == ("users", "alice")
    assert validate_namespace("agent-7@team.x") == ("agent-7@team.x",)
    with pytest.raises(InvalidKeyError):
        validate_namespace(())
    with pytest.raises(InvalidKeyError):
        validate_namespace(42)  # type: ignore[arg-type]
    with pytest.raises(InvalidKeyError):
        validate_namespace(("users", 1))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "memory_id", ["", "..", "../x", "a/b", "a\\b", "x.json", "_x", "a b", "a\x00", "z" * 129, 7]
)
def test_hostile_ids_are_refused(memory_id: object) -> None:
    with pytest.raises(InvalidKeyError):
        validate_id(memory_id)


def test_kind_and_prefix() -> None:
    assert validate_kind("episodic") == "episodic"
    with pytest.raises(InvalidKeyError):
        validate_kind("secret")
    assert validate_prefix("/agents/prod/") == "agents/prod"
    assert validate_prefix("") == ""
    for bad in ("../x", "a/../b", "a\\b", "/".join(["p"] * 9)):
        with pytest.raises(InvalidKeyError):
            validate_prefix(bad)
    with pytest.raises(InvalidKeyError):
        validate_prefix(None)  # type: ignore[arg-type]


def test_layout() -> None:
    ns = ("users", "alice")
    assert memory_key("m", ns, "semantic", "abc") == "m/users/alice/semantic/abc.json"
    assert memory_key("", ns, "episodic", "abc") == "users/alice/episodic/abc.json"
    assert index_key("m", ns) == "m/users/alice/_index.json"
    assert vectors_key("m", ns) == "m/users/alice/_vectors.npz"
    assert tombstone_key("m", ns, "abc") == "m/users/alice/_tombstones/abc.json"
    assert namespace_from_index_key("m", "m/users/alice/_index.json") == ns
    assert namespace_from_index_key("", "users/alice/_index.json") == ns
    assert namespace_from_index_key("m", "other/users/_index.json") is None
    assert namespace_from_index_key("m", "m/users/semantic/_index.json") is None
    assert namespace_from_index_key("m", "m/users/alice/semantic/x.json") is None


def test_object_key_defence_in_depth() -> None:
    assert validate_object_key("a/_index.json") == "a/_index.json"
    for bad in ("", "/a", "a/", "a//b", "a/../b", "a/./b", "a\\b", "a/b\x00", "x" * 1025):
        with pytest.raises(InvalidKeyError):
            validate_object_key(bad)
    assert validate_list_prefix("") == ""
    assert validate_list_prefix("a/b/") == "a/b/"
    for bad in ("a/b", "../", "/"):
        with pytest.raises(InvalidKeyError):
            validate_list_prefix(bad)
