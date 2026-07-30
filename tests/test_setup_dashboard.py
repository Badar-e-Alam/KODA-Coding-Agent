"""Tests for the /setup dashboard backend and the fs host-path fix.

Exercises:
  - koda.env_file: read_env_file / update_env_file / delete_env_keys against a
    throwaway .env (comments and unrelated keys must survive)
  - koda.bridge._mask_secret + Bridge._env_list event shape (secrets masked,
    ".env" vs ambient "env" source reported)
  - koda.tools.fs._resolve accepts real absolute paths under the workspace root

Self-contained: each test creates its own tempdir so it runs identically
standalone (``python tests/test_setup_dashboard.py``) and under pytest.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _with_tmpdir(fn):
    """Run fn(path) inside a fresh TemporaryDirectory, restoring cwd after."""
    cwd = os.getcwd()

    def wrapper():
        with tempfile.TemporaryDirectory() as d:
            try:
                fn(Path(d))
            finally:
                os.chdir(cwd)
    wrapper.__name__ = fn.__name__
    return wrapper


@_with_tmpdir
def test_env_file_roundtrip(tmpdir: Path) -> None:
    from koda import env_file

    os.chdir(tmpdir)
    path = tmpdir / ".env"
    path.write_text(
        "# my keys\n"
        "OPENAI_API_KEY=sk-old\n"
        "UNRELATED=keep-me\n",
        encoding="utf-8",
    )

    assert env_file.read_env_file(path) == {
        "OPENAI_API_KEY": "sk-old",
        "UNRELATED": "keep-me",
    }, env_file.read_env_file(path)

    # update in place + append a new key
    env_file.update_env_file({"OPENAI_API_KEY": "sk-new", "ANTHROPIC_API_KEY": "sk-ant-1"}, path)
    vals = env_file.read_env_file(path)
    assert vals["OPENAI_API_KEY"] == "sk-new", vals
    assert vals["ANTHROPIC_API_KEY"] == "sk-ant-1", vals
    assert vals["UNRELATED"] == "keep-me", vals
    assert "# my keys" in path.read_text(encoding="utf-8"), "comment must survive"

    # quoted values round-trip
    env_file.update_env_file({"KODA_DEFAULT_MODEL": "ollama:kimi k2"}, path)
    assert env_file.read_env_file(path)["KODA_DEFAULT_MODEL"] == "ollama:kimi k2"

    # delete one key, leave the rest
    env_file.delete_env_keys(["OPENAI_API_KEY"], path)
    vals = env_file.read_env_file(path)
    assert "OPENAI_API_KEY" not in vals, vals
    assert vals["ANTHROPIC_API_KEY"] == "sk-ant-1", vals
    assert vals["UNRELATED"] == "keep-me", vals

    # missing file reads as empty, deleting from it is a no-op
    assert env_file.read_env_file(tmpdir / "nope.env") == {}
    env_file.delete_env_keys(["X"], tmpdir / "nope.env")
    print("  env_file read/update/delete OK")


def test_mask_secret() -> None:
    import koda.bridge as b

    assert b._mask_secret("") == ""
    assert b._mask_secret("short") == "•••••", b._mask_secret("short")
    masked = b._mask_secret("sk-ant-api03-abcdefghijkl")
    assert masked.startswith("sk-a") and masked.endswith("ijkl"), masked
    assert "api03" not in masked, "middle of the token must not leak"
    print("  _mask_secret OK")


@_with_tmpdir
def test_bridge_env_list_shape(tmpdir: Path) -> None:
    """_env_list masks secrets and reports where each value came from."""
    import koda.bridge as b

    os.chdir(tmpdir)
    (tmpdir / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-secret-value-123\n", encoding="utf-8")
    os.environ["OLLAMA_HOST"] = "http://localhost:11434"
    os.environ.pop("OPENAI_API_KEY", None)

    class _Stub:
        _model = "ollama:test"

    captured: list[dict] = []
    orig = b.emit
    b.emit = captured.append  # type: ignore[assignment]
    try:
        b.Bridge._env_list(self=_Stub())  # type: ignore[arg-type]
    finally:
        b.emit = orig  # type: ignore[assignment]
        os.environ.pop("OLLAMA_HOST", None)

    assert len(captured) == 1, captured
    ev = captured[0]
    assert ev["type"] == "env_state", ev
    assert ev["model"] == "ollama:test", ev
    rows = {r["key"]: r for r in ev["keys"]}

    anthropic = rows["ANTHROPIC_API_KEY"]
    assert anthropic["set"] is True and anthropic["source"] == ".env", anthropic
    assert "secret-value" not in anthropic["value"], anthropic

    host = rows["OLLAMA_HOST"]
    assert host["source"] == "env", host
    assert host["value"] == "http://localhost:11434", host  # non-secret shown verbatim

    unset = rows["OPENAI_API_KEY"]
    assert unset["set"] is False and unset["source"] == "" and unset["value"] == "", unset
    print("  Bridge._env_list shape OK")


def test_bridge_env_set_rejects_unknown_key() -> None:
    """Only SETUP_KEYS are writable — the dashboard is not an env-var shell."""
    import asyncio

    import koda.bridge as b

    captured: list[dict] = []
    orig = b.emit
    b.emit = captured.append  # type: ignore[assignment]
    try:
        asyncio.run(b.Bridge._env_set(self=None, cmd={"key": "PATH", "value": "/evil"}))  # type: ignore[arg-type]
    finally:
        b.emit = orig  # type: ignore[assignment]

    assert len(captured) == 1, captured
    assert captured[0]["type"] == "error", captured
    assert "not a settable key" in captured[0]["message"], captured
    assert os.environ.get("PATH") != "/evil"
    print("  Bridge._env_set key allowlist OK")


@_with_tmpdir
def test_fs_resolve_accepts_root_absolute_paths(tmpdir: Path) -> None:
    """A real absolute path under the workspace root resolves like a jail path."""
    from koda.tools import fs

    (tmpdir / "pkg").mkdir()
    (tmpdir / "pkg" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    fs.set_workspace_root(tmpdir)
    root = str(fs._ROOT)  # resolved (macOS /var -> /private/var)

    jailed = fs._resolve("/pkg/mod.py")
    host = fs._resolve(f"{root}/pkg/mod.py")
    relative = fs._resolve("pkg/mod.py")
    assert jailed == host == relative, (jailed, host, relative)
    assert fs._resolve(root) == fs._ROOT, fs._resolve(root)

    # escaping the root is still refused
    try:
        fs._resolve("../outside.txt")
        raise AssertionError("expected ValueError for path escaping the root")
    except ValueError:
        pass
    print("  fs._resolve host-absolute paths OK")


if __name__ == "__main__":
    tests = [
        test_env_file_roundtrip,
        test_mask_secret,
        test_bridge_env_list_shape,
        test_bridge_env_set_rejects_unknown_key,
        test_fs_resolve_accepts_root_absolute_paths,
    ]
    for t in tests:
        print(f"· {t.__name__}")
        t()
    print(f"\n{len(tests)} test(s) passed")
