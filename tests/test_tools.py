"""Tests for coding_agent/tools.py: multi_edit, read-only git, project detection,
background bash. No network; git tests that need a real repo skip without git."""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from coding_agent import tools

# ── multi_edit ─────────────────────────────────────────────────────────


def _multi_edit(path: Path, edits: list[tuple[str, str]]) -> str:
    return tools.multi_edit.invoke(
        {"path": str(path), "edits": [{"old": o, "new": n} for o, n in edits]}
    )


def test_multi_edit_applies_edits_in_order(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("alpha = 1\nbeta = 2\n")
    # The second edit only matches the text the first edit produced.
    out = _multi_edit(f, [("alpha = 1", "gamma = 1"), ("gamma = 1", "gamma = 10")])
    assert out.startswith("applied 2 edits")
    assert f.read_text() == "gamma = 10\nbeta = 2\n"


def test_multi_edit_missing_find_string_leaves_file_unchanged(tmp_path):
    f = tmp_path / "a.py"
    original = "alpha = 1\nbeta = 2\n"
    f.write_text(original)
    out = _multi_edit(f, [("alpha", "ALPHA"), ("does-not-exist", "x")])
    assert out.startswith("[error] edit 2")
    assert "matched 0 times" in out
    assert f.read_text() == original


def test_multi_edit_repeated_find_string_leaves_file_unchanged(tmp_path):
    f = tmp_path / "a.py"
    original = "x = 1\nx = 1\n"
    f.write_text(original)
    out = _multi_edit(f, [("x = 1", "y = 1")])
    assert out.startswith("[error] edit 1")
    assert "matched 2 times" in out
    assert f.read_text() == original


def test_multi_edit_missing_file(tmp_path):
    out = _multi_edit(tmp_path / "nope.py", [("a", "b")])
    assert out.startswith("[error] file not found")


# ── git / git_diff are read-only ───────────────────────────────────────


@pytest.fixture
def no_git_run(monkeypatch):
    """Fail the test if any git process would be started."""

    def _boom(*args, **kwargs):  # pragma: no cover - only runs on a bug
        raise AssertionError(f"git must not run: {args[0]!r}")

    monkeypatch.setattr(tools.subprocess, "run", _boom)


@pytest.fixture
def git_calls(monkeypatch):
    calls: list[list[str]] = []

    def _fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(tools.subprocess, "run", _fake_run)
    return calls


@pytest.mark.parametrize(
    ("subcommand", "extra_args"),
    [
        ("commit", "-m x"),
        ("push", ""),
        ("reset", "--hard"),
        ("checkout", "main"),
        ("branch", "-D main"),
        ("branch", "-d main"),
        ("branch", "--delete main"),
        ("branch", "-m old new"),
        ("branch", "new-branch"),
        ("tag", "-d v1"),
        ("tag", "-a v2 -m msg"),
        ("tag", "v2"),
        ("remote", "add evil https://example.invalid/x.git"),
        ("remote", "remove origin"),
        ("remote", "set-url origin https://example.invalid/x.git"),
        ("config", "user.name someone"),
        ("config", "--unset user.name"),
        ("config", "--add core.hooksPath /tmp/hooks"),
        ("reflog", "expire --expire=now --all"),
        ("reflog", "delete HEAD@{0}"),
        ("log", "--output=/tmp/koda-pwned"),
        ("show", "--output /tmp/koda-pwned HEAD"),
        # git accepts unique long-option prefixes and bundled short options.
        ("branch", "-v --del main"),
        ("branch", "-v -vD main"),
        ("branch", "-r -rd origin/x"),
        ("tag", "-l --del v1"),
        ("config", "--get --unset-a user.name"),
        ("log", "--out=/tmp/koda-pwned"),
    ],
)
def test_git_refuses_mutating_command(no_git_run, subcommand, extra_args):
    out = tools.git.invoke({"subcommand": subcommand, "extra_args": extra_args})
    assert out.startswith("[error]")


@pytest.mark.parametrize(
    ("subcommand", "extra_args"),
    [
        ("status", "--short --branch"),
        ("log", "-3 --oneline"),
        ("log", "-10 --stat"),
        ("branch", "-vv"),
        ("tag", "-n5 -l"),
        ("show", "HEAD"),
        ("branch", ""),
        ("branch", "-a"),
        ("branch", "--list feat*"),
        ("branch", "--contains HEAD"),
        ("tag", ""),
        ("tag", "-l v*"),
        ("remote", "-v"),
        ("remote", "get-url origin"),
        ("config", "--get user.name"),
        ("config", "--list"),
        ("reflog", ""),
        ("rev-parse", "HEAD"),
    ],
)
def test_git_allows_read_forms(git_calls, subcommand, extra_args):
    out = tools.git.invoke({"subcommand": subcommand, "extra_args": extra_args})
    assert not out.startswith("[error]"), out
    assert git_calls == [["git", subcommand, *extra_args.split()]]


def test_git_diff_only_runs_diff(git_calls):
    tools.git_diff.invoke({"staged": True, "file": "--output=/tmp/koda-pwned"})
    # The file argument sits after ``--`` so git treats it as a path, not a flag.
    assert git_calls == [["git", "diff", "--cached", "--", "--output=/tmp/koda-pwned"]]


@pytest.fixture
def repo(tmp_path, monkeypatch):
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    for k, v in {
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
        "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
    }.items():
        monkeypatch.setenv(k, v)
    r = tmp_path / "repo"
    r.mkdir()

    def run(*args):
        subprocess.run(["git", *args], cwd=r, check=True, capture_output=True)

    run("init", "-q", "-b", "main")
    (r / "f.txt").write_text("one\n")
    run("add", "f.txt")
    run("commit", "-q", "-m", "first")
    run("branch", "keep")
    run("tag", "v1")
    return r


def _git_out(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout


def test_git_real_repo_mutations_leave_repo_unchanged(repo):
    before = (_git_out(repo, "branch"), _git_out(repo, "tag"), _git_out(repo, "remote"))
    for sub, extra in [("branch", "-D keep"), ("tag", "-d v1"),
                       ("remote", "add evil https://example.invalid/x.git"),
                       ("config", "user.name changed"), ("commit", "--allow-empty -m x")]:
        out = tools.git.invoke({"subcommand": sub, "extra_args": extra, "path": str(repo)})
        assert out.startswith("[error]"), (sub, extra, out)
    after = (_git_out(repo, "branch"), _git_out(repo, "tag"), _git_out(repo, "remote"))
    assert after == before
    assert "changed" not in (repo / ".git" / "config").read_text()
    assert _git_out(repo, "rev-list", "--count", "HEAD").strip() == "1"


def test_git_real_repo_read_and_diff(repo):
    assert "first" in tools.git.invoke(
        {"subcommand": "log", "extra_args": "--oneline", "path": str(repo)}
    )
    assert "keep" in tools.git.invoke({"subcommand": "branch", "path": str(repo)})
    (repo / "f.txt").write_text("two\n")
    diff = tools.git_diff.invoke({"path": str(repo)})
    assert "-one" in diff and "+two" in diff


# ── framework / checker / linter detection ─────────────────────────────


def _make(root: Path, files: dict[str, str]) -> Path:
    for name, content in files.items():
        p = root / name
        if name.endswith("/"):
            p.mkdir(parents=True, exist_ok=True)
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
    return root


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        ({"pytest.ini": "[pytest]\n"}, "pytest"),
        ({"tests/": ""}, "pytest"),
        ({"pyproject.toml": "[tool.pytest.ini_options]\n"}, "pytest"),
        ({"package.json": json.dumps({"devDependencies": {"jest": "^29"}})}, "jest"),
        ({"package.json": json.dumps({"scripts": {"test": "vitest"}})}, "npm-test"),
        ({"Cargo.toml": "[package]\n"}, "cargo"),
        ({"go.mod": "module x\n"}, "go"),
        ({"README.md": "hi\n"}, ""),
    ],
)
def test_detect_test_framework(tmp_path, files, expected):
    assert tools._detect_test_framework(_make(tmp_path, files)) == expected


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        ({"mypy.ini": "[mypy]\n"}, "mypy"),
        ({"pyrightconfig.json": "{}"}, "pyright"),
        ({"pyproject.toml": "[tool.mypy]\n"}, "mypy"),
        ({"pyproject.toml": "[tool.pyright]\n"}, "pyright"),
        ({"pyproject.toml": "[project]\n", "pkg/m.py": "x = 1\n"}, "mypy"),
        ({"tsconfig.json": "{}"}, "tsc"),
        ({"README.md": "hi\n"}, ""),
    ],
)
def test_detect_type_checker(tmp_path, files, expected):
    assert tools._detect_type_checker(_make(tmp_path, files)) == expected


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        ({"ruff.toml": ""}, "ruff"),
        ({"pyproject.toml": "[tool.ruff]\nline-length = 100\n"}, "ruff"),
        ({"eslint.config.js": "export default [];\n"}, "eslint"),
        ({".eslintrc.json": "{}"}, "eslint"),
        ({"main.py": "print(1)\n"}, "ruff"),
        ({"package.json": "{}"}, "eslint"),
        ({"README.md": "hi\n"}, ""),
    ],
)
def test_detect_linter(tmp_path, files, expected):
    assert tools._detect_linter(_make(tmp_path, files)) == expected


# ── background bash ────────────────────────────────────────────────────

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")


def _wait_until(predicate, timeout: float = 10.0, interval: float = 0.05):
    """Poll ``predicate`` until it returns a truthy value or ``timeout`` expires."""
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value or time.monotonic() > deadline:
            return value
        time.sleep(interval)


def _alive(pid: int) -> bool:
    """True if ``pid`` exists and is not a zombie (zombies linger in containers)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    stat = Path(f"/proc/{pid}/stat")
    if stat.parent.exists():
        try:
            return stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"
        except (OSError, IndexError):
            return False
    ps = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    state = ps.stdout.strip()
    return bool(state) and not state.startswith("Z")


@pytest.fixture
def bg(monkeypatch):
    """Isolate the background-process registry and kill anything left behind."""
    monkeypatch.setattr(tools, "_BG_PROCS", {})
    started: list[int] = []
    yield started
    tools._cleanup_background_procs()
    for pgid in started:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGKILL)


def _start(bg, command: str, path: Path) -> str:
    out = tools.bash_background.invoke({"command": command, "path": str(path)})
    m = re.match(r"started bash_id=(\S+) pid=(\d+)", out)
    assert m, out
    bg.append(int(m.group(2)))
    return m.group(1)


def _body(reply: str) -> str:
    """The new-output part of a bash_output reply (the header repeats the command)."""
    return reply.split("--new output--\n", 1)[1]


def _collect(bash_id: str, needle: str) -> str:
    """Poll bash_output until ``needle`` appears in the output; return all output."""
    seen: list[str] = []

    def _poll():
        seen.append(_body(tools.bash_output.invoke({"bash_id": bash_id})))
        return needle in "".join(seen)

    assert _wait_until(_poll), "".join(seen)
    return "".join(seen)


@posix_only
def test_bash_background_start_read_kill(bg, tmp_path):
    bash_id = _start(bg, "echo hello; echo oops 1>&2; sleep 30", tmp_path)
    pid = bg[-1]
    out = _collect(bash_id, "[stderr] oops")
    assert "hello" in out
    # Output already read is not returned again.
    again = tools.bash_output.invoke({"bash_id": bash_id})
    assert "status=running" in again
    assert _body(again) == "(no new output)"

    killed = tools.kill_bash.invoke({"bash_id": bash_id})
    assert killed.startswith(f"killed bash_id={bash_id}")
    assert _wait_until(lambda: not _alive(pid))
    assert bash_id not in tools._BG_PROCS
    assert tools.bash_output.invoke({"bash_id": bash_id}).startswith("[error] unknown bash_id")


@posix_only
def test_bash_background_finished_process_is_dropped(bg, tmp_path):
    bash_id = _start(bg, "echo done", tmp_path)
    replies: list[str] = []

    def _poll():
        replies.append(tools.bash_output.invoke({"bash_id": bash_id}))
        return "status=exited (code 0)" in replies[-1]

    assert _wait_until(_poll), replies
    assert "done" in "".join(_body(r) for r in replies)
    assert bash_id not in tools._BG_PROCS


@posix_only
def test_kill_bash_reaps_child_processes(bg, tmp_path):
    bash_id = _start(bg, "sleep 30 & echo child=$!; wait", tmp_path)
    child = int(re.search(r"child=(\d+)", _collect(bash_id, "child=")).group(1))
    assert _alive(child)
    tools.kill_bash.invoke({"bash_id": bash_id})
    assert _wait_until(lambda: not _alive(child))


@posix_only
def test_cleanup_on_exit_kills_running_processes(bg, tmp_path):
    _start(bg, "sleep 30", tmp_path)
    pid = bg[-1]
    tools._cleanup_background_procs()
    assert tools._BG_PROCS == {}
    assert _wait_until(lambda: not _alive(pid))


@posix_only
def test_cleanup_on_exit_kills_children_after_shell_exited(bg, tmp_path):
    # The shell exits at once and leaves ``sleep`` running in its process group.
    bash_id = _start(bg, "sleep 30 & echo child=$!", tmp_path)
    proc = tools._BG_PROCS[bash_id]
    child = int(re.search(r"child=(\d+)", _wait_until(
        lambda: "".join(proc._buf) if "child=" in "".join(proc._buf) else ""
    )).group(1))
    assert _wait_until(lambda: proc.proc.poll() is not None)
    tools._cleanup_background_procs()
    assert _wait_until(lambda: not _alive(child)), "child outlived cleanup"


@posix_only
def test_bash_output_keeps_id_while_children_run(bg, tmp_path):
    bash_id = _start(bg, "sleep 30 & echo child=$!", tmp_path)
    proc = tools._BG_PROCS[bash_id]
    assert _wait_until(lambda: proc.proc.poll() is not None)
    out = _collect(bash_id, "child=")
    child = int(re.search(r"child=(\d+)", out).group(1))
    # The shell has exited but its child still runs, so kill_bash must still work.
    assert tools.kill_bash.invoke({"bash_id": bash_id}).startswith("killed")
    assert _wait_until(lambda: not _alive(child))
