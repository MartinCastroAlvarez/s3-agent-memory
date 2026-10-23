from __future__ import annotations

import json
from pathlib import Path

import pytest

from s3_agent_memory.cli import main


def run(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, object, str]:
    code = main(list(args))
    out, err = capsys.readouterr()
    try:
        payload = json.loads(out) if out.strip() else None
    except json.JSONDecodeError:
        payload = out
    return code, payload, err


def test_cli_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    url = ["--url", f"file://{tmp_path}/mem", "-n", "users/alice"]
    code, m, _ = run(
        capsys,
        "remember",
        "Alice prefers metric units",
        "--kind",
        "procedural",
        "--tag",
        "prefs",
        *url,
    )
    assert code == 0 and isinstance(m, dict) and m["kind"] == "procedural"
    mid = m["id"]
    run(capsys, "remember", "Alice prefers metric units everywhere", "--ttl-days", "30", *url)
    code, hits, _ = run(capsys, "recall", "metric units", *url)
    assert code == 0 and isinstance(hits, list) and hits[0]["id"] == mid
    code, got, _ = run(capsys, "get", mid, *url)
    assert code == 0 and isinstance(got, dict) and got["key"].endswith(f"{mid}.json")
    code, upd, _ = run(
        capsys,
        "update",
        mid,
        "--importance",
        "0.9",
        "--tag",
        "x",
        "--content",
        "Alice uses metric",
        *url,
    )
    assert code == 0 and isinstance(upd, dict) and upd["version"] == 2
    code, _, err = run(capsys, "update", mid, "--content", "stale", "--expected-version", "1", *url)
    assert code == 3 and "conflict" in err
    code, listed, _ = run(capsys, "list", "--kind", "procedural", *url)
    assert code == 0 and isinstance(listed, list) and len(listed) == 1
    code, stats, _ = run(capsys, "stats", *url)
    assert isinstance(stats, dict) and stats["memories"] == 2
    code, names, _ = run(capsys, "namespaces", *url)
    assert names == ["users/alice"]
    code, report, _ = run(capsys, "consolidate", "--dry-run", *url)
    assert code == 0 and isinstance(report, dict) and report["dry_run"] is True

    out = tmp_path / "export.jsonl"
    code, _, err = run(capsys, "export", "-o", str(out), *url)
    assert code == 0 and json.loads(err) == {"exported": 2}
    code, imported, _ = run(
        capsys, "import", str(out), "--url", f"file://{tmp_path}/mem", "-n", "users/carol"
    )
    assert isinstance(imported, dict) and imported["imported"] == 2

    code, rep, _ = run(capsys, "reindex", *url)
    assert isinstance(rep, dict) and rep["indexed"] == 2
    code, pruned, _ = run(capsys, "prune", *url)
    assert pruned == {"forgotten": []}
    code, gone, _ = run(capsys, "forget", mid, *url)
    assert code == 0 and isinstance(gone, dict) and gone["forgotten"] is True
    code, _, _ = run(capsys, "forget", mid, *url)
    assert code == 1
    code, _, _ = run(capsys, "get", mid, *url)
    assert code == 1


def test_cli_consolidate_with_longest_and_stdout_export(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("S3_AGENT_MEMORY_URL", f"file://{tmp_path}")
    monkeypatch.setenv("S3_AGENT_MEMORY_NAMESPACE", "bot")
    run(capsys, "remember", "the deploy runs on fridays")
    run(capsys, "remember", "the deploy runs on fridays at noon")
    code, report, _ = run(capsys, "consolidate", "--summariser", "longest")
    assert code == 0 and isinstance(report, dict) and len(report["merged"]) == 1
    assert report["merged"][0]["content"] == "the deploy runs on fridays at noon"
    code, out, err = run(capsys, "export")
    assert code == 0 and json.loads(err)["exported"] == 1


def test_cli_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    url = ["--url", f"file://{tmp_path}"]
    code, _, err = run(capsys, "remember", "x", "-n", "../escape", *url)
    assert code == 2 and "InvalidKeyError" in err
    code, _, err = run(capsys, "recall", "x", "-n", "a", "--url", "ftp://nope/x")
    assert code == 3 and "backend" in err
    code, _, err = run(capsys, "remember", "x", "--importance", "7", *url)
    assert code == 2
    with pytest.raises(SystemExit):
        main(["--version"])
    assert not (tmp_path.parent / "escape").exists()


def test_cli_import_from_stdin(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import io

    line = json.dumps(
        {
            "id": "abc",
            "kind": "semantic",
            "content": "from stdin",
            "namespace": ["x"],
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        }
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(line + "\n"))
    code, report, _ = run(capsys, "import", "-", "--url", f"file://{tmp_path}", "-n", "y")
    assert code == 0 and isinstance(report, dict) and report["imported"] == 1
