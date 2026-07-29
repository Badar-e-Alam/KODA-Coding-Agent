"""Skill discovery and authoring — the one place that knows what a skill is.

A skill is a ``<name>/SKILL.md`` directory with YAML frontmatter. They come from
two sources, matching ``SKILL_SOURCES`` in :mod:`coding_agent.agent`:

* **bundled** — ``coding_agent/skills/``, shipped with the package, shared by
  every project. Read-only in a pip install, and wiped on upgrade.
* **project** — ``<root>/.koda/skills/``, travels with the repo.

deepagents' ``SkillsMiddleware`` merges those sources left-to-right and lets a
later one win on a name collision, so a project skill overrides a bundled skill
of the same name. :func:`list_skills` reproduces that merge exactly — which is
what makes ``/skill list`` and the banner report what the *model* actually has,
rather than a superset.

This lives apart from :mod:`koda.bridge` on purpose. The bridge translates
events; it should not also be the authority on skills. Both it and
:mod:`koda.workspace` read from here, so there is a single definition to keep
in step with deepagents.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

SKILL_AUTHOR_PROMPT = """You are authoring an Agent Skill file (SKILL.md) that follows Anthropic's Agent Skills specification.

Write ONLY the complete SKILL.md content — no preamble, no explanation, and no surrounding code fences.

It MUST begin with YAML frontmatter delimited by --- lines:
---
name: <lowercase-hyphenated identifier, 1-64 chars, only a-z 0-9 and single hyphens>
description: <one or two sentences saying WHAT the skill does AND WHEN to use it; include trigger keywords>
---

Then a markdown body with these sections:
# <Skill Title>
## When to Use   — bullet list of situations that should trigger this skill
## Instructions  — a concrete, numbered, step-by-step workflow the agent should follow
## Notes         — optional pitfalls, tips, or a short example

Make it specific and actionable — this is instructions for an AI coding agent, not marketing copy.

The skill to author:
{brief}
"""


# ── where skills live ─────────────────────────────────────────────────────


def bundled_dir() -> Path:
    """Package-bundled skills, shared by every project."""
    from coding_agent.backend import SKILLS_DIR

    return Path(SKILLS_DIR)


def project_dir(root: Path) -> Path:
    """Skills belonging to one project (``<root>/.koda/skills/``)."""
    from coding_agent.backend import project_skills_dir

    return Path(project_skills_dir(root))


# ── parsing ───────────────────────────────────────────────────────────────


def frontmatter(content: str) -> dict[str, Any]:
    """The YAML frontmatter mapping from SKILL.md content (or ``{}``)."""
    m = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if not m:
        return {}
    try:
        import yaml

        data = yaml.safe_load(m.group(1))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def slug(s: str) -> str:
    """Normalize an arbitrary string into a spec-valid skill name."""
    s = re.sub(r"[^a-z0-9]+", "-", s.strip().lower())
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s[:64]


def clean_markdown(text: str) -> str:
    """Strip a wrapping code fence the model may have added; ensure trailing NL."""
    t = text.strip()
    m = re.match(r"^```[a-zA-Z]*\s*\n(.*)\n```$", t, re.DOTALL)
    if m:
        t = m.group(1).strip()
    return t + "\n"


# ── discovery ─────────────────────────────────────────────────────────────


def _scan(directory: Path, scope: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for md in sorted(directory.glob("*/SKILL.md")):
        try:
            fm = frontmatter(md.read_text(encoding="utf-8"))
        except OSError:
            fm = {}  # unreadable → fall back to the directory name
        out.append(
            {
                "name": str(fm.get("name") or md.parent.name),
                "description": str(fm.get("description") or ""),
                "path": str(md),
                "scope": scope,
            }
        )
    return out


def list_skills(root: Path) -> list[dict[str, str]]:
    """Every skill the agent can see in ``root``, sorted by name.

    Later sources win on a name collision — the same rule deepagents applies, so
    a project skill shadows a bundled one instead of appearing twice.
    """
    merged: dict[str, dict[str, str]] = {}
    for scope, d in (("bundled", bundled_dir()), ("project", project_dir(root))):
        for skill in _scan(d, scope):
            merged[skill["name"]] = skill
    return [merged[k] for k in sorted(merged)]


# ── authoring ─────────────────────────────────────────────────────────────


def author(model: str, brief: str) -> str:
    """One-shot call to the configured model to write a SKILL.md body."""
    from koda.summarizer import create_chat_model

    llm = create_chat_model(model)
    resp = llm.invoke(SKILL_AUTHOR_PROMPT.format(brief=brief))
    text = getattr(resp, "content", resp)
    if isinstance(text, list):  # some providers return content blocks
        text = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in text)
    return clean_markdown(str(text))


def save(root: Path, content: str, brief: str) -> tuple[Path | None, str]:
    """Write an authored skill into the project. Returns ``(path, error)``.

    Authored skills land in the PROJECT (``.koda/skills/``), never the package:
    ``coding_agent/skills/`` is read-only in a pip install and is wiped on
    upgrade, whereas ``.koda/skills/`` travels with the repo and — being the
    last skill source — overrides a bundled skill of the same name.
    """
    fm = frontmatter(content)
    name = slug(str(fm.get("name") or brief.split(":", 1)[0]))
    if not name:
        return None, "could not determine a valid skill name — try naming it explicitly."

    # The spec requires the frontmatter name and the directory to match, and the
    # model may have slugged it differently.
    if str(fm.get("name") or "") != name:
        content = re.sub(r"(?m)^name:.*$", f"name: {name}", content, count=1)

    dest = project_dir(root) / name
    if (dest / "SKILL.md").exists():
        return None, f"skill '{name}' already exists — pick another name or delete it first."
    try:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "SKILL.md").write_text(content, encoding="utf-8")
    except OSError as e:
        return None, f"could not write skill: {e}"
    return dest / "SKILL.md", ""
