"""Regression guard: the permission policy's tool names must match reality.

``permissions.decide()`` FAILS OPEN — it returns ``"approve"`` for any tool
that isn't in ``MUTATING_TOOLS``, and ``INTERRUPT_ON`` is derived from that
same set. So if a gated name ever drifts out of sync with the live graph (a
deepagents rename, a local refactor, a dropped builtin), the tool silently
falls out of ``interrupt_on``, the graph never interrupts, and **PLAN mode
stops blocking writes with no error anywhere**.

Nothing else in the codebase cross-checks those hand-maintained name sets
against the tools actually wired into the graph. This test does.

Builds the real agent (no network: a chat model is only *constructed*, never
invoked) and diffs the policy's names against the live tool surface.
"""

from __future__ import annotations

import pytest

from coding_agent.agent import build_agent
from koda.adapters.langgraph import _introspect_graph_tools
from koda.tools import permissions as perms

# The tools deepagents' middleware stack is contracted to provide out of the
# box (FilesystemMiddleware + TodoListMiddleware + SubAgentMiddleware). Pinned
# here so a version bump that drops or renames one fails loudly rather than
# quietly shrinking the agent's capabilities.
DEEPAGENTS_BUILTINS = {
    "write_todos",
    "task",
    "ls",
    "read_file",
    "write_file",
    "edit_file",
    "glob",
    "grep",
    "execute",
}


async def _live_tool_names(tmp_path) -> set[str]:
    """Build the real graph and return the tool names the model can see."""
    graph = await build_agent(model="anthropic:claude-sonnet-4-6", cwd=tmp_path)
    try:
        return {t.name for t in _introspect_graph_tools(graph)}
    finally:
        conn = getattr(graph, "_koda_checkpointer_conn", None)
        if conn is not None:
            await conn.close()


@pytest.mark.asyncio
async def test_gated_tools_all_exist_on_the_live_graph(tmp_path, monkeypatch):
    """Every tool the policy gates must actually exist, or PLAN mode is a no-op."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-used-model-is-never-invoked")
    live = await _live_tool_names(tmp_path)

    assert live, "no tools introspected from the graph — _introspect_graph_tools broke"

    missing = perms.MUTATING_TOOLS - live
    assert not missing, (
        f"{sorted(missing)} is gated by permissions.MUTATING_TOOLS but no longer "
        f"exists on the graph. decide() fails open, so these tools are now "
        f"UNGATED: PLAN mode will not block them and no permission prompt will "
        f"appear. Live tools: {sorted(live)}"
    )


@pytest.mark.asyncio
async def test_deepagents_still_provides_its_builtin_tools(tmp_path, monkeypatch):
    """A deepagents upgrade that drops/renames a builtin must fail here, loudly."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-used-model-is-never-invoked")
    live = await _live_tool_names(tmp_path)

    missing = DEEPAGENTS_BUILTINS - live
    assert not missing, (
        f"deepagents no longer provides {sorted(missing)}. KODA relies on its "
        f"default middleware for these. Live tools: {sorted(live)}"
    )


def test_interrupt_on_covers_exactly_the_mutating_tools():
    """INTERRUPT_ON is what deepagents actually gates on; it must not drift."""
    assert set(perms.INTERRUPT_ON) == perms.MUTATING_TOOLS
    assert perms.FILE_EDIT_TOOLS <= perms.MUTATING_TOOLS


def test_mutating_tools_are_split_between_deepagents_and_koda():
    """Pin *which* gated tools come from where.

    Catches a rename on either side of the split: if KODA renames
    ``multi_edit``/``bash_background`` in coding_agent/tools.py, or deepagents
    renames one of its three mutating builtins, the expected residue changes.
    """
    from coding_agent.tools import EXTRA_TOOLS

    koda_names = {t.name for t in EXTRA_TOOLS}
    from_deepagents = perms.MUTATING_TOOLS - koda_names

    assert from_deepagents == {"write_file", "edit_file", "execute"}, (
        f"expected exactly write_file/edit_file/execute to come from deepagents' "
        f"FilesystemMiddleware, got {sorted(from_deepagents)}"
    )
