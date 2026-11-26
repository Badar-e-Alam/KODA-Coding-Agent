"""
koda.tool_stats — measure the tool calls an agent makes.

:class:`ToolCallTracker` pairs ``start`` / ``finish`` calls by ``tool_id``
and summarises them: how many calls, which failed, which never finished,
results that had no matching start, per-tool timings and output sizes, and
identical calls the agent repeated (a sign it is looping).

The bridge (``koda/bridge.py``) feeds it every ``ToolStart`` / ``ToolResult``
of a turn and reports :meth:`ToolCallTracker.summary` as a ``tool_stats``
event. Tracking must never break a turn, so nothing here raises on odd
input (non-dict arguments, values that are not JSON-serialisable, …).

Choices the ticket left open:

* ``calls`` counts every ``start`` (finished or not).
* An orphan result (``finish`` for an unknown ``tool_id``) is counted in
  ``orphan_results`` only; it does not add to ``calls`` or ``errors``.
* A second ``start`` with a ``tool_id`` that is still open counts as a new
  call; the earlier one stays unfinished.
* ``output_chars`` is the length of the full output (``str()`` of it when it
  is not a string).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

# Identical name + arguments called this many times (or more) is a repeat.
REPEAT_THRESHOLD = 3


def _args_key(arguments: Any) -> str:
    """A stable comparison key for tool arguments (dict key order ignored)."""
    try:
        return json.dumps(arguments, sort_keys=True, default=repr)
    except Exception:  # unsortable mixed keys, recursion, …
        try:
            return repr(arguments)
        except Exception:
            return f"<unrepresentable {type(arguments).__name__}>"


def _wire_safe(value: Any, seen: set[int] | None = None) -> Any:
    """Return a JSON-safe copy for the event payload without changing keys."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    seen = seen if seen is not None else set()
    ident = id(value)
    if ident in seen:
        return "<recursive>"
    if isinstance(value, dict):
        seen.add(ident)
        try:
            return {str(k): _wire_safe(v, seen) for k, v in value.items()}
        finally:
            seen.remove(ident)
    if isinstance(value, (list, tuple)):
        seen.add(ident)
        try:
            return [_wire_safe(v, seen) for v in value]
        finally:
            seen.remove(ident)
    try:
        return repr(value)
    except Exception:
        return f"<unrepresentable {type(value).__name__}>"


@dataclass
class _Call:
    name: str
    arguments: Any
    key: str
    started: float
    finished: bool = False
    is_error: bool = False
    duration_ms: float = 0.0
    output_chars: int = 0


class ToolCallTracker:
    """Record tool calls and summarise them. ``clock`` returns seconds."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self.reset()

    def reset(self) -> None:
        """Forget every call (start of a new turn)."""
        self._calls: list[_Call] = []
        self._open: dict[str, _Call] = {}
        self._orphans = 0

    def start(self, tool_id: str, name: str, arguments: Any) -> None:
        call = _Call(
            name=str(name),
            arguments=arguments,
            key=_args_key(arguments),
            started=self._clock(),
        )
        self._calls.append(call)
        self._open[tool_id] = call

    def finish(self, tool_id: str, output: Any, is_error: bool = False) -> None:
        call = self._open.pop(tool_id, None)
        if call is None:
            self._orphans += 1
            return
        call.finished = True
        call.is_error = bool(is_error)
        call.duration_ms = max(0.0, (self._clock() - call.started) * 1000.0)
        try:
            call.output_chars = len(output if isinstance(output, str) else str(output))
        except Exception:
            call.output_chars = 0

    def summary(self) -> dict[str, Any]:
        by_tool: dict[str, dict[str, Any]] = {}
        seen: dict[tuple[str, str], dict[str, Any]] = {}
        for c in self._calls:
            t = by_tool.setdefault(
                c.name,
                {"calls": 0, "errors": 0, "total_ms": 0.0, "max_ms": 0.0, "output_chars": 0},
            )
            t["calls"] += 1
            if c.finished:
                t["errors"] += int(c.is_error)
                t["total_ms"] += c.duration_ms
                t["max_ms"] = max(t["max_ms"], c.duration_ms)
                t["output_chars"] += c.output_chars
            r = seen.setdefault(
                (c.name, c.key), {"name": c.name, "arguments": c.arguments, "count": 0}
            )
            r["count"] += 1
        for t in by_tool.values():
            t["total_ms"] = round(t["total_ms"], 3)
            t["max_ms"] = round(t["max_ms"], 3)
        return {
            "calls": len(self._calls),
            "errors": sum(1 for c in self._calls if c.finished and c.is_error),
            "unfinished": [c.name for c in self._calls if not c.finished],
            "orphan_results": self._orphans,
            "by_tool": by_tool,
            "repeats": [
                {**r, "arguments": _wire_safe(r["arguments"])}
                for r in seen.values()
                if r["count"] >= REPEAT_THRESHOLD
            ],
        }
