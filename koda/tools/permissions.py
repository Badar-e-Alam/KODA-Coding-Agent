"""
Permission policy for mutating tools.

KODA gates the agent's mutating tools (``write_file`` / ``edit_file`` /
``multi_edit`` / ``execute``) through LangGraph's human-in-the-loop
``interrupt()`` mechanism: the graph pauses *before* running a gated tool,
checkpoints its state, and the adapter surfaces a ``PermissionRequest`` to
the TUI. Nothing blocks — neither the event loop nor a worker thread.

This module is the single source of truth for the *policy* that decides,
per gated tool call, whether to:

  * ``"approve"`` — let it run (auto, no prompt),
  * ``"reject"``  — refuse it (auto, no prompt), or
  * ``"ask"``     — surface a prompt and wait for the user.

Behaviour depends on the current :class:`~koda.modes.Mode`:

  * ``PLAN``    — every mutating tool is auto-rejected (advisory-only;
                  the user reviews the plan and presses Shift+A to apply).
  * ``EDITS``   — file writes/edits auto-approve; shell calls auto-approve
                  too, *except* ``rm`` (file removal), which still asks.
  * ``DEFAULT`` — every mutating tool asks on first use, unless the user
                  has already "always allowed" it for this session (an
                  explicit "always" overrides the EDITS ``rm`` guard).

The decision is consumed by ``koda.adapters.langgraph.LangGraphAdapter``,
which turns ``"ask"`` results into a prompt and the user's answer into a
LangGraph resume command.

A separate *soft-pause* primitive (``wait_until_unpaused`` /
``mark_prompt_*``) is still used by the ``ask_user`` tool bridge so that
its worker-thread → UI-loop handoff coordinates focus; it is unrelated to
the permission policy above.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import re
import threading
from typing import Literal

from koda.modes import Mode
from koda.web.policy import INTERRUPT_ON, SessionPolicy

_log = logging.getLogger("koda.permissions")


# ── Soft pause (used only by the ``ask_user`` bridge) ──────────────────
#
# When the ``ask_user`` tool puts a question on screen from its worker
# thread, this lets any other in-flight worker-thread tool wait so it
# doesn't steal focus. The permission gate no longer uses this — it pauses
# the *whole graph* via ``interrupt()`` instead — but ask_user still does.
#
# Lazy-created because the module imports before the event loop exists, and
# ``asyncio.Event()`` binds to the current loop on first wait.

_pause_event: asyncio.Event | None = None
_pause_lock = threading.Lock()


def _ensure_pause_event() -> asyncio.Event:
    """Lazily create the pause event. Default state: set (no pause)."""
    global _pause_event
    with _pause_lock:
        if _pause_event is None:
            _pause_event = asyncio.Event()
            _pause_event.set()
    return _pause_event


async def wait_until_unpaused() -> None:
    """Block while an ``ask_user`` prompt is being shown. No-op when none."""
    await _ensure_pause_event().wait()


def mark_prompt_pending() -> None:
    """Pause: signal that an ``ask_user`` prompt is now visible.

    MUST be called on the asyncio loop's thread (``asyncio.Event.clear`` is
    not thread-safe in the general case). The TUI bridge uses
    ``App.call_from_thread`` to satisfy this from the worker thread.
    """
    _ensure_pause_event().clear()


def mark_prompt_resolved() -> None:
    """Resume: signal that the ``ask_user`` prompt has been dismissed."""
    _ensure_pause_event().set()


# ── Module state — single-process truth source ─────────────────────────
#
# (Legacy module-level state removed; all policy state now lives in
# ``SessionPolicy``. The default policy below keeps single-process TUI/CLI
# behaviour unchanged.)


# ── Tool classification ────────────────────────────────────────────────

# Tools that change the filesystem or run shell commands. These are the
# tools wired into ``create_deep_agent(interrupt_on=…)`` (see
# ``coding_agent/agent.py``); every other tool (ls, read_file, glob, grep,
# web_search, git, …) is read-only and never interrupts.
#
# DANGER: these are string names, and ``decide()`` FAILS OPEN — a tool that
# isn't listed here is approved without a prompt. So if one of these names ever
# drifts from the live tool set (a deepagents rename, a local refactor), that
# tool silently stops being gated and PLAN mode stops blocking it, with no
# error anywhere. ``tests/test_permission_tool_names.py`` diffs this set
# against the real graph's tools to make that a test failure instead.
MUTATING_TOOLS: set[str] = {"write_file", "edit_file", "multi_edit", "execute", "bash_background"}

# Subset of MUTATING_TOOLS that count as "file edits" — these pass through
# silently in EDITS mode.
FILE_EDIT_TOOLS: set[str] = {"write_file", "edit_file", "multi_edit"}

# ``execute`` commands that are dangerous enough to ALWAYS prompt, even when
# the user has session-allowed ``execute`` or is in EDITS mode. This is the
# defense-in-depth backstop for the class of bug where a single shell call
# walks the entire disk or destroys files: an orphaned
# ``glob('/**/.env', recursive=True)`` once pegged a CPU for 75 minutes. The
# session-allow escape hatch is keyed by tool *name* only (args aren't part
# of the key), so without this a single "always allow execute" would wave
# such a command straight through. Patterns are deliberately broad — a false
# "ask" costs one keypress; a false "approve" can cost the machine.
_DANGEROUS_EXECUTE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Filesystem-root- or home-rooted walks (find / grep -r / rg / ls -R).
    re.compile(r"\b(find|grep|rg|ls|du|chmod|chown)\b[^\n|;&]*\s(/|~|\$HOME)(\s|$)"),
    # Python recursive walks anchored at root/home: glob('/**…'), os.walk('/'),
    # Path('/').rglob(…), pathlib rglob from root.
    re.compile(r"""glob\(\s*['"](/|~|\$HOME|/\*\*)"""),
    re.compile(r"""(os\.walk|rglob)\(\s*['"]?(/|~|os\.path\.expanduser)"""),
    re.compile(r"recursive\s*=\s*True[^\n]*['\"]/\*\*"),
    # Root-anchored shell globs: /**/…  or  /*  walks.
    re.compile(r"(^|\s)/\*\*?/"),
    # Destructive classics.
    re.compile(r"\brm\b[^\n|;&]*\s-[a-zA-Z]*[rf][a-zA-Z]*\s+(/|~|\$HOME)(\s|$)"),
    re.compile(r"\b(mkfs|dd)\b[^\n]*\bif=|\bof=/dev/"),
    re.compile(r">\s*/dev/[sh]d[a-z]"),
    re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}"),  # fork bomb
)


def is_dangerous_execute(command: str) -> bool:
    """True if a shell command matches a known catastrophic pattern.

    Matching forces an ``"ask"`` verdict regardless of session-allow / mode
    (except PLAN, which already rejects everything). See
    ``_DANGEROUS_EXECUTE_PATTERNS`` for the rationale.
    """
    if not command:
        return False
    return any(p.search(command) for p in _DANGEROUS_EXECUTE_PATTERNS)


# ``rm`` (file removal) is the one shell verb that ACCEPT-EDITS mode still
# refuses to auto-approve. Removing a file is easy to trigger by accident,
# hard to undo, and unlike a build/edit can't be recovered from version
# control if it was never tracked. The pattern matches ``rm`` as a *command*
# word (optionally prefixed by sudo / nohup / exec / xargs or chained after
# ``;``/``&&``/``|``), so ``grep rm …`` or ``ls rm`` don't trip it.
_RM_CMD_PATTERN = re.compile(
    r"(?:^|[|;&]+\s*|\b(?:sudo|nohup|exec|xargs)\s+)rm\b"
)


def _is_rm_command(command: str) -> bool:
    """True if ``command`` removes files via ``rm`` (ACCEPT-EDITS guard).

    Deliberately narrow to the ``rm`` verb so ordinary shell calls
    (``ls``, ``git status``, ``pytest``, ``npm run build`` …) auto-approve
    in ACCEPT-EDITS mode; only ``rm`` still surfaces a prompt there.
    """
    if not command:
        return False
    return bool(_RM_CMD_PATTERN.search(command))


# The exact mapping handed to deepagents' ``interrupt_on``. Keeping it here
# keeps the gated-tool list and the policy in one place.
INTERRUPT_ON: dict[str, dict] = {
    name: {"allowed_decisions": ["approve", "reject"]} for name in MUTATING_TOOLS
}

Verdict = Literal["approve", "reject", "ask"]


# ── Context-aware default policy (legacy TUI compatibility) ────────────

# The web server creates a SessionPolicy per run and injects it into adapters
# via a context variable. The legacy TUI/bridge runs a single session per
# process, so it continues to use the module-level default policy below.
_policy_ctx: contextvars.ContextVar[SessionPolicy | None] = contextvars.ContextVar(
    "koda_active_policy", default=None
)
_default_policy = SessionPolicy()


def _active_policy() -> SessionPolicy:
    return _policy_ctx.get() or _default_policy


@contextlib.contextmanager
def policy_context(policy: SessionPolicy):
    """Return a context manager that sets the active policy for this run."""
    token = _policy_ctx.set(policy)
    try:
        yield
    finally:
        _policy_ctx.reset(token)


# ── Public API ─────────────────────────────────────────────────────────


def current_mode() -> Mode:
    return _active_policy().mode


def set_mode(mode: Mode) -> None:
    _active_policy().set_mode(mode)
    _log.info("mode → %s", mode.value)


def set_auto_approve(on: bool) -> None:
    _active_policy().set_auto_approve(on)
    _log.info("auto-approve → %s", on)


def is_auto_approve() -> bool:
    return _active_policy().auto_approve


def allow_tool(tool_name: str) -> None:
    _active_policy().allow_tool(tool_name)
    _log.info("session-allow: %s", tool_name)


def is_allowed(tool_name: str) -> bool:
    return _active_policy().is_allowed(tool_name)


def clear_session_allow() -> None:
    _active_policy().allowed.clear()
    _active_policy().set_auto_approve(False)


def reject_message(tool_name: str) -> str:
    return _active_policy().reject_message(tool_name)


def decide(tool_name: str, args: dict | None = None) -> Verdict:
    return _active_policy().decide(tool_name, args)




