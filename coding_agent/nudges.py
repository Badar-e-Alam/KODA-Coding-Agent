"""In-context behavioral nudges for the coding agent.

Langfuse trace analysis (see the project memory ``koda-harness-serial-plodding``)
showed KODA fires exactly ONE tool call on ~85% of turns — reading files
one-by-one, each in its own model→tools round-trip, instead of batching
independent reads/greps into a single turn. The system prompt already tells
it to batch; prompt wording alone moved the needle 0%.

So this is a *harness* nudge, not a prompt tweak: a middleware that watches the
live message stream and, when it sees N single-read turns in a row, injects a
short reminder RIGHT WHERE THE MODEL READS IT — in the context for the very next
turn — instead of relying on a rule buried 200 lines up in the system prompt.

The nudge is intentionally minimal and self-limiting: it fires at most once per
"streak", carries no state across sessions, and degrades to a silent no-op on
any introspection failure. It never blocks or mutates a tool call.
"""

from __future__ import annotations

import logging
import os

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, SystemMessage

_log = logging.getLogger(__name__)

# Read-only navigation tools. A turn that emits exactly one of these and
# nothing else is a candidate "plod" — the model looked at one thing and
# stopped, when it could have batched several.
_READ_ONLY_TOOLS = frozenset({"read_file", "grep", "glob", "ls"})

# How many single-read turns in a row before we nudge. 3 is the point where
# a human reviewer would say "why aren't you reading these together?" — one or
# two serial reads are often legitimately dependent.
_DEFAULT_THRESHOLD = 3

_REMINDER = (
    "[harness reminder] You've made {n} single-tool read turns in a row "
    "(one file/grep per turn). If your next reads/greps are independent — you "
    "already know their arguments without waiting for a prior result — emit "
    "them together in ONE turn instead of one at a time. Batch the reads, then "
    "reason over them."
)


def _single_read_tool(msg: AIMessage) -> bool:
    """True iff this assistant turn emitted exactly one read-only tool call."""
    calls = getattr(msg, "tool_calls", None) or []
    if len(calls) != 1:
        return False
    name = calls[0].get("name") if isinstance(calls[0], dict) else None
    return name in _READ_ONLY_TOOLS


class SerialReadNudgeMiddleware(AgentMiddleware):
    """Inject a batching reminder after a streak of single-read turns.

    Runs in ``before_model``: it inspects the recent assistant turns, counts a
    trailing streak of single read-only tool calls, and — once per streak —
    appends a short :class:`SystemMessage` the model sees on the turn it's about
    to take. Returns ``None`` (no state change) in the common case, so it's
    effectively free when the agent is already batching.
    """

    def __init__(self, threshold: int = _DEFAULT_THRESHOLD) -> None:
        super().__init__()
        self.threshold = max(2, int(threshold))

    def _trailing_streak(self, messages: list) -> int:
        """Count consecutive most-recent assistant turns that were single reads.

        Walks backward over AI messages (skipping the Tool results between
        them). Stops at the first assistant turn that wasn't a single read.
        """
        streak = 0
        for msg in reversed(messages):
            if not isinstance(msg, AIMessage):
                # Tool results / human messages sit between assistant turns;
                # skip them without breaking the streak.
                continue
            if _single_read_tool(msg):
                streak += 1
            else:
                break
        return streak

    def _already_nudged_for(self, messages: list, streak: int) -> bool:
        """Don't re-nudge within the same unbroken streak.

        If a reminder SystemMessage already sits within the last ``streak``
        assistant turns, we've fired for this streak already — stay quiet until
        the model breaks the pattern and (possibly) starts a new one.
        """
        seen_ai = 0
        for msg in reversed(messages):
            if isinstance(msg, SystemMessage) and "[harness reminder]" in str(
                getattr(msg, "content", "")
            ):
                return True
            if isinstance(msg, AIMessage):
                seen_ai += 1
                if seen_ai >= streak:
                    return False
        return False

    def before_model(self, state, runtime):  # type: ignore[override]
        if os.environ.get("KODA_DISABLE_NUDGES"):
            return None
        try:
            messages = state.get("messages") if isinstance(state, dict) else None
            if not messages:
                return None
            streak = self._trailing_streak(messages)
            if streak < self.threshold:
                return None
            if self._already_nudged_for(messages, streak):
                return None
            _log.debug("serial-read nudge firing at streak=%d", streak)
            return {"messages": [SystemMessage(content=_REMINDER.format(n=streak))]}
        except Exception:  # noqa: BLE001 — a nudge must never break a turn
            _log.debug("SerialReadNudgeMiddleware skipped", exc_info=True)
            return None


def build_serial_read_nudge_middleware() -> SerialReadNudgeMiddleware | None:
    """Construct the nudge middleware unless disabled via env.

    ``KODA_DISABLE_NUDGES=1`` turns it off entirely (returns ``None`` so it's
    never added to the chain). ``KODA_NUDGE_THRESHOLD`` overrides the streak
    length that triggers a reminder.
    """
    if os.environ.get("KODA_DISABLE_NUDGES"):
        return None
    threshold = _DEFAULT_THRESHOLD
    raw = os.environ.get("KODA_NUDGE_THRESHOLD")
    if raw:
        try:
            threshold = int(raw)
        except ValueError:
            _log.warning("ignoring non-integer KODA_NUDGE_THRESHOLD=%r", raw)
    return SerialReadNudgeMiddleware(threshold=threshold)
