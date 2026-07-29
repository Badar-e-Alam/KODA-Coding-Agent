"""Workspace introspection — what KODA is mounted to, and what it can do there.

One source of truth for the three questions the TUI's left bar asks:

* **CONTEXT** — which files the agent auto-loads for this project, and which
  on-disk memories exist. Mirrors ``memory=["/AGENTS.md", "/user_preferences.md"]``
  in :mod:`coding_agent.agent` and the ``/memories/`` mount in
  :mod:`coding_agent.backend`.
* **SKILLS** — the ``*/SKILL.md`` bundles under ``coding_agent/skills/``.
  These are package-global (shared by every project), not project-scoped.
* **TOOLS** — deepagents' built-ins + ``EXTRA_TOOLS`` + whatever the configured
  MCP servers contribute.

Everything is resolved against an explicit ``root`` so the answer is always
"...for *this* directory", never "...for wherever the process happens to be".

This is a **script**, not a hot path. It is free to import the tool registry
(~0.9 s) because a human ran it and is waiting for the answer. Nothing in the
TUI's startup path calls it — the banner shows no live data, and the running
UI gets tools and skills from the bridge's ``ready`` event, which already
carries them. There is deliberately no cache: a cross-process JSON cache
existed here once, purely so a decorative banner could dodge that 0.9 s, and it
cost more in drift than it ever bought in latency.

Run it standalone::

    python -m koda.workspace                 # pretty panel for the cwd
    python -m koda.workspace -C ~/proj --json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# There is deliberately NO hardcoded copy of deepagents' builtin tool names here.
# An earlier version kept one, which meant a deepagents rename would silently
# render a wrong-but-plausible tool list that no test could catch.


def auto_loaded() -> tuple[str, ...]:
    """Files deepagents injects into context every turn.

    Derived from ``coding_agent.agent.MEMORY_FILES`` — the same list handed to
    ``create_deep_agent(memory=…)`` — so the banner cannot drift from reality.
    """
    try:
        from coding_agent.agent import MEMORY_FILES

        return tuple(p.lstrip("/") for p in MEMORY_FILES)
    except Exception:
        return ("AGENTS.md", "user_preferences.md")


# ── context ───────────────────────────────────────────────────────────────


def git_branch(root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    branch = out.stdout.strip()
    return branch if out.returncode == 0 and branch else None


def context_for(root: Path) -> list[dict[str, Any]]:
    """The project files that make up the agent's starting context."""
    items: list[dict[str, Any]] = []
    for name in auto_loaded():
        items.append(
            {
                "name": name,
                "path": str(root / name),
                "exists": (root / name).is_file(),
                # deepagents injects these every turn — when they exist.
                "loaded": (root / name).is_file(),
            }
        )

    mem_dir = root / ".koda" / "memories"
    memories = sorted(mem_dir.glob("*.md")) if mem_dir.is_dir() else []
    for md in memories:
        items.append(
            {
                "name": f".koda/memories/{md.name}",
                "path": str(md),
                "exists": True,
                # On-disk but read on demand, not auto-injected.
                "loaded": False,
            }
        )
    return items


# ── skills ────────────────────────────────────────────────────────────────
# Discovery lives in koda/skills.py — the one module that knows what a skill is
# and how deepagents merges the two sources. This used to reach into
# koda.bridge for the frontmatter parser, which had the dependency backwards.

from koda.skills import bundled_dir as skills_dir  # noqa: E402
from koda.skills import list_skills as skills_for  # noqa: E402
from koda.skills import project_dir as project_skills_dir  # noqa: E402


# ── mcp ───────────────────────────────────────────────────────────────────


def mcp_for(root: Path) -> list[dict[str, str]]:
    """MCP servers configured for this project (``<root>/.mcp.json``)."""
    cfg = root / ".mcp.json"
    if not cfg.is_file():
        return []
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    servers = data.get("mcpServers") or data.get("mcp_servers") or {}
    if not isinstance(servers, dict):
        return []
    return [
        {
            "name": name,
            "transport": str((spec or {}).get("transport") or "stdio"),
            "url": str((spec or {}).get("url") or (spec or {}).get("command") or ""),
        }
        for name, spec in servers.items()
    ]


# ── tools ─────────────────────────────────────────────────────────────────


def _extra_tool_names() -> list[str]:
    """Names from ``coding_agent.tools.EXTRA_TOOLS`` (~0.9 s import — call late)."""
    try:
        from coding_agent.tools import EXTRA_TOOLS

        return [str(getattr(t, "name", "")) for t in EXTRA_TOOLS if getattr(t, "name", "")]
    except Exception:
        return []


def tools_static() -> list[dict[str, str]]:
    """KODA's own tools, without building a graph.

    Deliberately does NOT include deepagents' builtins: enumerating them would
    mean hardcoding names this repo doesn't own. The complete, live list (incl.
    MCP) only exists once the graph is standing — the bridge sends it on
    ``ready``, and ``/tools`` shows it.
    """
    return [{"name": n, "origin": "extra"} for n in _extra_tool_names()]


# ── mounts ────────────────────────────────────────────────────────────────


def mounts_for(root: Path) -> list[dict[str, str]]:
    """The backend routes, i.e. what the agent's virtual paths actually hit.

    Mirrors ``build_backend`` in coding_agent/backend.py — and is pinned to it
    by ``tests/test_workspace_mounts.py``, so a route added there without being
    added here fails the suite. Note the default route is ``virtual_mode``: a
    model-supplied ``/tmp/x`` lands at ``<root>/tmp/x``.
    """
    return [
        {"route": "/", "target": str(root), "scope": "project"},
        {"route": "/memories/", "target": str(root / ".koda" / "memories"), "scope": "project"},
        {"route": "/skills/", "target": str(skills_dir()), "scope": "global"},
        {
            "route": "/project_skills/",
            "target": str(project_skills_dir(root)),
            "scope": "project",
        },
    ]


# ── the whole picture ─────────────────────────────────────────────────────


def inspect_workspace(root: str | os.PathLike[str]) -> dict[str, Any]:
    """Everything KODA has in ``root``, resolved from the filesystem.

    Never builds an agent graph, so ``tools`` lists KODA's own ``EXTRA_TOOLS``
    only: deepagents' builtins and MCP-contributed tools are not knowable
    without standing the graph up, and are reported as unknown rather than
    guessed at. The running agent's full list comes from the bridge's ``ready``
    event (``/tools``).
    """
    root = Path(root).expanduser().resolve()
    return {
        "root": str(root),
        "name": root.name,
        "git_branch": git_branch(root),
        "context": context_for(root),
        "skills": skills_for(root),
        "mcp": mcp_for(root),
        "mounts": mounts_for(root),
        "tools": tools_static(),
    }


# ── CLI ───────────────────────────────────────────────────────────────────

_DIM = "\x1b[2m"
_BOLD = "\x1b[1m"
_ORANGE = "\x1b[38;2;232;149;90m"
_TEAL = "\x1b[38;2;90;209;196m"
_RESET = "\x1b[0m"


def _render(ws: dict[str, Any], color: bool) -> str:
    def c(code: str, s: str) -> str:
        return f"{code}{s}{_RESET}" if color else s

    out: list[str] = []
    branch = f"  on {ws['git_branch']}" if ws["git_branch"] else ""
    out.append("")
    out.append(c(_BOLD + _ORANGE, f"KODA workspace — {ws['name']}") + c(_DIM, branch))
    out.append(c(_DIM, ws["root"]))

    # Pad the raw text *before* colouring — padding a coloured string would
    # count the ANSI escape bytes toward the width and skew every column.
    out.append("")
    out.append(c(_BOLD, "MOUNTS"))
    for m in ws["mounts"]:
        route = c(_TEAL, f"{m['route']:<12}")
        out.append(f"  {route} {m['target']}  {c(_DIM, '(' + m['scope'] + ')')}")

    out.append("")
    loaded = [i for i in ws["context"] if i["loaded"]]
    out.append(c(_BOLD, "CONTEXT") + c(_DIM, f"  {len(loaded)} auto-loaded"))
    for i in ws["context"]:
        name = f"{i['name']:<28}"
        if i["loaded"]:
            out.append(f"  {c(_TEAL, '•')} {name} {c(_DIM, 'auto-loaded each turn')}")
        elif i["exists"]:
            out.append(f"  {c(_DIM, '◦')} {name} {c(_DIM, 'on disk, read on demand')}")
        else:
            out.append(f"  {c(_DIM, '◦')} {c(_DIM, name)} {c(_DIM, 'absent')}")

    out.append("")
    out.append(c(_BOLD, "SKILLS") + c(_DIM, f"  {len(ws['skills'])} available"))
    for s in ws["skills"]:
        desc = s["description"]
        if len(desc) > 60:
            desc = desc[:57] + "…"
        name = f"{s['name']:<20}"
        out.append(f"  {c(_TEAL, '•')} {name} {c(_DIM, desc)}")
    if not ws["skills"]:
        out.append(f"  {c(_DIM, '◦ none installed')}")

    out.append("")
    out.append(c(_BOLD, "MCP") + c(_DIM, f"  {len(ws['mcp'])} server(s)"))
    for m in ws["mcp"]:
        name = f"{m['name']:<20}"
        out.append(f"  {c(_TEAL, '•')} {name} {c(_DIM, m['transport'] + '  ' + m['url'])}")
    if not ws["mcp"]:
        out.append(f"  {c(_DIM, '◦ none configured')}")

    tools = ws["tools"]
    # We can only see KODA's own tools here — deepagents' builtins and MCP tools
    # both need the graph. Say so rather than printing a partial list as if it
    # were the whole thing.
    out.append("")
    out.append(
        c(_BOLD, "TOOLS")
        + c(_DIM, f"  {len(tools)} from KODA  (+ built-in and MCP tools at runtime — see /tools)")
    )
    for i in range(0, len(tools), 3):
        row = "".join(f"{c(_TEAL, '•')} {t['name']:<22}" for t in tools[i : i + 3])
        out.append(f"  {row}")

    out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="koda.workspace",
        description="Report the context, skills and tools KODA has in a directory.",
    )
    p.add_argument("--cwd", "-C", default=".", help="directory to inspect (default: .)")
    p.add_argument("--json", action="store_true", help="emit JSON instead of a panel")
    args = p.parse_args(argv)

    root = Path(args.cwd).expanduser().resolve()
    if not root.is_dir():
        print(f"koda.workspace: not a directory: {root}", file=sys.stderr)
        return 2

    ws = inspect_workspace(root)

    if args.json:
        print(json.dumps(ws, indent=2))
    else:
        print(_render(ws, color=sys.stdout.isatty() and os.environ.get("NO_COLOR") is None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
