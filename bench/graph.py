"""Terminal-Bench entrypoint: KODA's coding agent as a Harbor LangGraph target.

Harbor's ``langgraph`` agent (``harbor/agents/installed/langgraph.py``) stages
this directory into the task container, ``uv pip install -e .`` it, and hands
the graph declared by ``langgraph.json`` a single user message — the task
instruction — via ``graph.ainvoke({"messages": [...]}, config)``. The model
comes in as ``config["configurable"]["model"]``, sourced from ``harbor run -m``.

That is exactly ``coding_agent.agent.build_agent``'s shape, so this module is a
thin factory rather than a reimplementation: the graph under evaluation is the
same one the KODA TUI drives, which is the whole point — we are measuring KODA's
real harness, not a benchmark-only variant of it.

Two things must differ inside a benchmark container, and only two:

1. **No human is watching.** KODA gates mutating tools through deepagents'
   ``HumanInTheLoopMiddleware``, which calls LangGraph's ``interrupt()`` — the
   graph *stops* and checkpoints rather than blocking, so ``ainvoke`` returns a
   half-finished state carrying ``__interrupt__`` and the trial scores 0 with no
   error to explain it. Nothing in the container can answer that prompt, so the
   gate is emptied here.
2. **The agent owns the whole filesystem.** KODA normally jails its file tools
   to the project root, so a model-supplied ``/etc/nginx/nginx.conf`` lands at
   ``<cwd>/etc/nginx/nginx.conf``. Terminal-Bench tasks address ``/etc``,
   ``/usr/local`` and ``/app`` by absolute path, so the jail is turned off here.

Everything else — prompt, tools, middleware, context policy — is untouched.
Those are the knobs the optimization loop tunes; changing them here would mean
measuring something other than KODA.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from coding_agent.agent import build_agent, _render_system_prompt
from koda.tools import permissions

from prompt_fixes import align_prompt_with_bench_semantics
from variant import Variant

logger = logging.getLogger(__name__)

# Used only when harbor is invoked without -m. Kept explicit so a bench run
# never silently inherits KODA_DEFAULT_MODEL from the container env.
DEFAULT_MODEL = "ollama:glm-5.2"


def _resolve_model(config: dict[str, Any] | None) -> str:
    """Pick the model spec, preferring what harbor passed via ``-m``.

    Harbor normalizes ``-m ollama/glm-5.2`` to ``ollama:glm-5.2`` before it
    reaches ``configurable`` (``LangGraph._normalized_model_name``), which is
    already KODA's ``provider:model`` spec format — so it passes through as-is.
    """
    configurable = (config or {}).get("configurable") or {}
    model = configurable.get("model") or os.environ.get("KODA_BENCH_MODEL")
    return model or DEFAULT_MODEL


async def make_graph(config: dict[str, Any] | None = None):
    """Build the KODA coding agent for one benchmark trial."""
    model = _resolve_model(config)

    # The harness configuration under test, staged into this container with the
    # rest of bench/. Baseline when absent. See bench/variant.py and bench/rci/.
    variant = Variant.load()
    variant.apply_env()

    # See module docstring, point 1. The interrupt_on={} below is what actually
    # stops the graph halting; this covers the separate KODA-side policy that
    # non-deepagents call sites (koda/tools/permissions.decide) consult.
    permissions.set_auto_approve(True)

    # The container's working directory is where harbor's exec lands us, which
    # is the task's own WORKDIR — the same place a human solving the task would
    # start. Anchoring relative paths there keeps instructions like
    # "edit config.yaml" resolving the way the task author intended.
    cwd = os.environ.get("KODA_BENCH_CWD") or os.getcwd()

    # Render the prompt here rather than letting build_agent do it, so the
    # corrections and the variant's edits apply to the same fully-substituted
    # text the agent would otherwise have received. Order matters: make the
    # prompt truthful first (prompt_fixes), then let the variant tune it.
    system_prompt = align_prompt_with_bench_semantics(_render_system_prompt(Path(cwd)))
    system_prompt = variant.apply_prompt(system_prompt)

    logger.info("bench graph: model=%s cwd=%s variant=%s", model, cwd, variant.name)
    return await build_agent(
        model=model,
        cwd=cwd,
        system_prompt=system_prompt,
        # See module docstring, point 2.
        virtual_mode=False,
        # See module docstring, point 1 — gate nothing, nobody is there to approve.
        interrupt_on={},
    )
