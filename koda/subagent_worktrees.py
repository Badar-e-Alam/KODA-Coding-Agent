"""Git workspaces owned by local background subagents."""

from __future__ import annotations

import logging
import shutil
import sqlite3
import subprocess
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator

task_cwd: ContextVar[Path | None] = ContextVar("subagent_worktree", default=None)
task_project_root: ContextVar[Path | None] = ContextVar("subagent_project_root", default=None)

_log = logging.getLogger(__name__)
_RETENTION = 7 * 24 * 60 * 60


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True, text=True, errors="replace", check=check, timeout=30,
    )


def repository_root(cwd: Path) -> Path | None:
    if not shutil.which("git"):
        return None
    rev_parse = _git(cwd, "rev-parse", "--show-toplevel", check=False)
    return Path(rev_parse.stdout.rstrip("\n")).resolve() if rev_parse.returncode == 0 else None


def _leases(root: Path) -> Path:
    git_dir = _git(root, "rev-parse", "--git-common-dir").stdout.rstrip("\n")
    return root / git_dir / "koda-task-leases"


@contextmanager
def _lease(path: Path, *, create: bool = False) -> Iterator[None]:
    # Keep the lease outside the checkout so Git can remove it while locked.
    mode = "rwc" if create else "rw"
    connection = sqlite3.connect(f"{path.as_uri()}?mode={mode}", uri=True, timeout=0)
    try:
        try:
            connection.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            if exc.sqlite_errorcode == sqlite3.SQLITE_BUSY:
                raise BlockingIOError(f"Task worktree is in use: {path.name}") from exc
            raise
        yield
    finally:
        connection.close()


class TaskWorktree:
    def __init__(self, root: Path, task_id: str) -> None:
        self.root = root
        self.path = root / ".koda" / "worktrees" / task_id
        self.branch = f"koda/{task_id}"
        self.lease_path = _leases(root) / task_id
        self.base = _git(root, "rev-parse", "--verify", "HEAD").stdout.strip()

    def _checkout(self) -> None:
        if self.path.exists():
            return
        branch = _git(self.root, "show-ref", "--verify", "--quiet",
                      f"refs/heads/{self.branch}", check=False)
        if branch.returncode == 0:
            _git(self.root, "worktree", "add", str(self.path), self.branch)
        else:
            _git(self.root, "worktree", "add", "-b", self.branch, str(self.path), self.base)

    @contextmanager
    def activate(self) -> Iterator[None]:
        self.lease_path.parent.mkdir(parents=True, exist_ok=True)
        with _lease(self.lease_path, create=True):
            self._checkout()
            token = task_cwd.set(self.path)
            project_token = task_project_root.set(self.root)
            try:
                yield
            finally:
                task_cwd.reset(token)
                task_project_root.reset(project_token)
                self.lease_path.touch()

    def finish(self) -> str:
        ref = f"refs/heads/{self.branch}"
        head = _git(self.path, "symbolic-ref", "--quiet", "HEAD", check=False)
        if head.stdout.strip() != ref:
            raise RuntimeError(f"Task left its worktree on a different ref. Inspect {self.path}.")
        _git(self.path, "add", "-A", "--", ".")
        staged = _git(self.path, "diff", "--cached", "--quiet", check=False)
        if staged.returncode == 1:
            _git(self.path, "-c", "user.name=KODA", "-c", "user.email=koda@localhost",
                 "commit", "-m", "Save background subagent changes")
        else:
            staged.check_returncode()
        diff = _git(self.path, "diff", "--name-status", self.base, ref, "--").stdout.strip()
        return f"Branch `{self.branch}`\n{diff or 'No changed files.'}"


def cleanup_worktrees(root: Path) -> None:
    """Sweep inactive KODA worktrees, keeping their branches available."""
    directory = root / ".koda" / "worktrees"
    if not directory.is_dir():
        return
    leases = _leases(root)
    for path in directory.glob("task_*"):
        lease = leases / path.name
        if not lease.is_file() or not (path / ".git").is_file() or path.is_symlink():
            continue
        try:
            with _lease(lease):
                if not path.exists():
                    continue
                expired = time.time() - lease.stat().st_mtime >= _RETENTION
                branch = f"refs/heads/koda/{path.name}"
                if expired or _merged_and_clean(root, path, branch):
                    _git(root, "worktree", "remove", "--force", str(path))
        except (FileNotFoundError, BlockingIOError):
            continue
        except (OSError, sqlite3.Error, subprocess.SubprocessError):
            _log.warning("Could not clean up task worktree %s", path, exc_info=True)


def _merged_and_clean(root: Path, path: Path, branch: str) -> bool:
    head = _git(path, "symbolic-ref", "--quiet", "HEAD", check=False)
    if head.stdout.strip() != branch:
        return False
    merged = _git(root, "merge-base", "--is-ancestor", branch, "HEAD", check=False)
    # Ignore caches, while preserving new edits until the inactivity deadline.
    return merged.returncode == 0 and not _git(path, "status", "--porcelain").stdout
