"""Path limits of koda/tools/fs.py: nothing outside the workspace root is read,
listed, or written. One test per escape case (../, absolute, symlink, ~).

By design, absolute paths and ``~`` are *not* errors: ``/x`` maps to ``{root}/x``
and ``~`` is a literal folder name inside the root (see ``fs._resolve``). For
those cases the tests assert that the outside file is never touched.
"""

from __future__ import annotations

import os
import sys

import pytest

from koda.tools import fs

SECRET = "TOP-SECRET-OUTSIDE"

needs_symlinks = pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges")


@pytest.fixture
def jail(tmp_path, monkeypatch):
    """A workspace root with a sibling ``outside`` folder holding a secret."""
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text(SECRET + "\n")
    monkeypatch.setattr(fs, "_permission_check", lambda *a, **k: None)
    previous = fs._ROOT
    fs.set_workspace_root(root)
    (fs._ROOT / "inside.txt").write_text("inside\n")
    yield fs._ROOT, outside.resolve()
    fs._ROOT = previous
    fs._READ_CACHE.clear()


def _outside_untouched(outside):
    assert sorted(p.name for p in outside.iterdir()) == ["secret.txt"]
    assert (outside / "secret.txt").read_text() == SECRET + "\n"


# ── ../ ────────────────────────────────────────────────────────────────


def test_dotdot_read_refused(jail):
    with pytest.raises(ValueError, match="escapes workspace root"):
        fs.read_file.invoke({"file_path": "../outside/secret.txt"})


def test_dotdot_nested_read_refused(jail):
    with pytest.raises(ValueError, match="escapes workspace root"):
        fs.read_file.invoke({"file_path": "/sub/../../outside/secret.txt"})


def test_dotdot_write_refused(jail):
    _root, outside = jail
    with pytest.raises(ValueError, match="escapes workspace root"):
        fs.write_file.invoke({"file_path": "../outside/new.txt", "content": "x"})
    _outside_untouched(outside)


def test_dotdot_edit_refused(jail):
    _root, outside = jail
    with pytest.raises(ValueError, match="escapes workspace root"):
        fs.edit_file.invoke({"file_path": "../outside/secret.txt",
                             "old_string": SECRET, "new_string": "pwned"})
    _outside_untouched(outside)


def test_dotdot_ls_refused(jail):
    with pytest.raises(ValueError, match="escapes workspace root"):
        fs.ls.invoke({"path": "../"})


def test_dotdot_glob_base_refused(jail):
    with pytest.raises(ValueError, match="escapes workspace root"):
        fs.glob.invoke({"pattern": "*", "path": "../outside"})


def test_dotdot_glob_pattern_does_not_list_outside(jail):
    out = fs.glob.invoke({"pattern": "../outside/*"})
    assert "secret.txt" not in out


def test_dotdot_grep_path_refused(jail):
    with pytest.raises(ValueError, match="escapes workspace root"):
        fs.grep.invoke({"pattern": SECRET, "path": "../outside"})


# ── absolute paths ─────────────────────────────────────────────────────


def test_absolute_outside_path_read_stays_in_root(jail):
    _root, outside = jail
    out = fs.read_file.invoke({"file_path": str(outside / "secret.txt")})
    assert out.startswith("Error: file not found")
    assert SECRET not in out


def test_absolute_outside_path_write_stays_in_root(jail):
    root, outside = jail
    target = outside / "new.txt"
    fs.write_file.invoke({"file_path": str(target), "content": "x"})
    _outside_untouched(outside)
    assert (root / str(target).lstrip("/")).read_text() == "x"


def test_absolute_outside_dir_ls_stays_in_root(jail):
    _root, outside = jail
    out = fs.ls.invoke({"path": str(outside)})
    assert out.startswith("Error: path not found")


def test_absolute_root_prefixed_path_reads_inside(jail):
    root, _outside = jail
    assert "inside" in fs.read_file.invoke({"file_path": str(root / "inside.txt")})


# ── symlinks pointing outside ──────────────────────────────────────────


@needs_symlinks
def test_symlink_file_read_refused(jail):
    root, outside = jail
    os.symlink(outside / "secret.txt", root / "link.txt")
    with pytest.raises(ValueError, match="symlink"):
        fs.read_file.invoke({"file_path": "/link.txt"})


@needs_symlinks
def test_symlink_dir_read_refused(jail):
    root, outside = jail
    os.symlink(outside, root / "linkdir")
    with pytest.raises(ValueError, match="symlink"):
        fs.read_file.invoke({"file_path": "/linkdir/secret.txt"})


@needs_symlinks
def test_symlink_dir_write_refused(jail):
    root, outside = jail
    os.symlink(outside, root / "linkdir")
    with pytest.raises(ValueError, match="symlink"):
        fs.write_file.invoke({"file_path": "/linkdir/new.txt", "content": "x"})
    _outside_untouched(outside)


@needs_symlinks
def test_symlink_dir_ls_refused(jail):
    root, outside = jail
    os.symlink(outside, root / "linkdir")
    with pytest.raises(ValueError, match="symlink"):
        fs.ls.invoke({"path": "/linkdir"})


@needs_symlinks
def test_symlink_file_not_read_by_grep(jail):
    root, outside = jail
    os.symlink(outside / "secret.txt", root / "link.txt")
    out = fs.grep.invoke({"pattern": SECRET, "output_mode": "content"})
    assert out.startswith("(no matches"), out


@needs_symlinks
def test_symlink_file_not_listed_by_glob(jail):
    root, outside = jail
    os.symlink(outside / "secret.txt", root / "link.txt")
    out = fs.glob.invoke({"pattern": "*.txt"})
    assert "link.txt" not in out
    assert "/inside.txt" in out


# ── ~ expansion ────────────────────────────────────────────────────────


def test_tilde_read_is_not_expanded(jail, monkeypatch):
    _root, outside = jail
    monkeypatch.setenv("HOME", str(outside))
    out = fs.read_file.invoke({"file_path": "~/secret.txt"})
    assert out.startswith("Error: file not found")
    assert SECRET not in out


def test_tilde_write_is_not_expanded(jail, monkeypatch):
    root, outside = jail
    monkeypatch.setenv("HOME", str(outside))
    fs.write_file.invoke({"file_path": "~/new.txt", "content": "x"})
    _outside_untouched(outside)
    assert (root / "~" / "new.txt").read_text() == "x"


def test_tilde_ls_is_not_expanded(jail, monkeypatch):
    _root, outside = jail
    monkeypatch.setenv("HOME", str(outside))
    out = fs.ls.invoke({"path": "~"})
    assert out.startswith("Error: path not found")
