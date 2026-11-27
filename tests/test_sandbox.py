"""KODA_SANDBOX=project shell isolation (KODA-10).

The OS-level tests run real commands under bwrap (Linux) / sandbox-exec
(macOS) and skip when that tool is missing or cannot run here (e.g. bwrap in a
container without unprivileged user namespaces). The unit tests below them
cover mode parsing, argv construction and the fallback warning everywhere.
"""

from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

import pytest

from coding_agent import sandbox
from coding_agent.backend import ReapingShellBackend


@pytest.fixture(autouse=True)
def _fresh_sandbox_state(monkeypatch):
    monkeypatch.delenv("KODA_SANDBOX", raising=False)
    monkeypatch.delenv("KODA_SANDBOX_NETWORK", raising=False)
    sandbox._reset_for_tests()
    yield
    sandbox._reset_for_tests()


@pytest.fixture
def dirs():
    """A project dir and a sibling "outside" dir, both outside temp.

    They live next to this test file rather than under /tmp, because temp is
    writable inside the sandbox by design and would make the checks trivial.
    """
    base = Path(__file__).resolve().parent / f".sbx-{uuid.uuid4().hex[:8]}"
    project = base / "project"
    outside = base / "outside"
    project.mkdir(parents=True)
    outside.mkdir()
    yield project, outside
    shutil.rmtree(base, ignore_errors=True)


def _backend(project: Path) -> ReapingShellBackend:
    return ReapingShellBackend(root_dir=project, virtual_mode=True, timeout=30, inherit_env=True)


requires_sandbox = pytest.mark.skipif(
    not sandbox.sandbox_available(),
    reason="sandbox tool (bwrap / sandbox-exec) missing or unusable here",
)


# ── OS-level behaviour ────────────────────────────────────────────────────


@requires_sandbox
def test_write_outside_project_fails(dirs, monkeypatch):
    project, outside = dirs
    monkeypatch.setenv("KODA_SANDBOX", "project")
    target = outside / "escaped.txt"
    res = _backend(project).execute(f"echo pwned > '{target}'")
    assert res.exit_code != 0
    assert not target.exists()


@requires_sandbox
def test_write_inside_project_and_temp_works(dirs, monkeypatch):
    project, _ = dirs
    monkeypatch.setenv("KODA_SANDBOX", "project")
    res = _backend(project).execute(
        "echo ok > inside.txt && t=$(mktemp) && echo ok > \"$t\" && rm \"$t\""
    )
    assert res.exit_code == 0, res.output
    assert (project / "inside.txt").read_text().strip() == "ok"


@requires_sandbox
def test_network_blocked_by_default(dirs, monkeypatch):
    project, _ = dirs
    monkeypatch.setenv("KODA_SANDBOX", "project")
    script = (
        "import socket,sys\n"
        "s=socket.socket(); s.settimeout(3)\n"
        "try:\n"
        "    s.connect(('1.1.1.1', 53))\n"
        "except OSError as e:\n"
        "    print('blocked', e); sys.exit(7)\n"
        "print('connected')\n"
    )
    script_file = project / "net.py"
    script_file.write_text(script)
    res = _backend(project).execute(f"'{sys.executable}' net.py")
    assert res.exit_code == 7, res.output
    assert "blocked" in res.output


# ── Mode parsing, argv construction, fallback ─────────────────────────────


def test_off_is_default_and_passes_command_through():
    w = sandbox.wrap_command("echo hi", root="/proj")
    assert w.args == "echo hi"
    assert w.shell is True
    assert w.sandboxed is False
    assert w.warning is None


def test_off_mode_execute_unchanged(dirs, monkeypatch):
    project, outside = dirs
    monkeypatch.setenv("KODA_SANDBOX", "off")
    target = outside / "ok.txt"
    res = _backend(project).execute(f"echo hi > '{target}' && echo done")
    assert res.exit_code == 0
    assert target.exists()
    assert "[sandbox]" not in res.output


def test_unknown_mode_treated_as_off(monkeypatch):
    monkeypatch.setenv("KODA_SANDBOX", "bogus")
    assert sandbox.sandbox_mode() == sandbox.MODE_OFF


def test_bwrap_argv_keeps_command_as_single_arg_and_blocks_net():
    cmd = "echo 'a b' ; rm -rf \"$HOME\""
    argv = sandbox.bwrap_argv(cmd, "/proj", "/proj", network=False)
    assert argv[0] == "bwrap"
    assert argv[-3:] == ["/bin/sh", "-c", cmd]
    assert "--unshare-net" in argv
    assert ["--ro-bind", "/", "/"] == argv[1:4]
    i = argv.index("/proj")
    assert argv[i - 1] == "--bind"


def test_bwrap_argv_network_opt_in():
    argv = sandbox.bwrap_argv("true", "/proj", "/proj", network=True)
    assert "--unshare-net" not in argv


def test_sandbox_exec_argv_profile(monkeypatch):
    argv = sandbox.sandbox_exec_argv("echo hi", "/proj", network=False)
    assert argv[:2] == ["sandbox-exec", "-p"]
    profile = argv[2]
    assert "(deny file-write*)" in profile
    assert '(subpath (param "PROJECT_ROOT"))' in profile
    assert "(deny network*)" in profile
    assert "PROJECT_ROOT=/proj" in argv
    assert argv[-3:] == ["/bin/sh", "-c", "echo hi"]
    net = sandbox.sandbox_exec_argv("echo hi", "/proj", network=True)
    assert "(deny network*)" not in net[2]


def test_network_env(monkeypatch):
    assert sandbox.network_allowed() is False
    monkeypatch.setenv("KODA_SANDBOX_NETWORK", "1")
    assert sandbox.network_allowed() is True


def test_missing_tool_falls_back_with_one_time_warning(dirs, monkeypatch, capsys):
    project, _ = dirs
    monkeypatch.setenv("KODA_SANDBOX", "project")
    monkeypatch.setattr(sandbox.shutil, "which", lambda _name: None)
    backend = _backend(project)

    first = backend.execute("echo hi")
    assert first.exit_code == 0
    assert "NOT isolated" in first.output
    assert "hi" in first.output
    assert "NOT isolated" in capsys.readouterr().err

    second = backend.execute("echo again")
    assert "NOT isolated" not in second.output
    assert "NOT isolated" not in capsys.readouterr().err


def test_sandboxed_wrap_uses_argv(monkeypatch):
    monkeypatch.setenv("KODA_SANDBOX", "project")
    monkeypatch.setattr(sandbox, "sandbox_available", lambda: True)
    monkeypatch.setattr(sandbox, "_tool_name", lambda: "bwrap")
    w = sandbox.wrap_command("echo hi", root="/proj")
    assert w.shell is False
    assert w.sandboxed is True
    assert isinstance(w.args, list) and w.args[0] == "bwrap"
