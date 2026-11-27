"""Retry classification, backoff and session-cost bookkeeping.

Pure helpers (no SDK imports) so they work across Anthropic, OpenAI-style,
Ollama and httpx errors by looking at status codes and class names.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any

from koda.provider_models import cost_usd

_log = logging.getLogger("koda.reliability")

MAX_ATTEMPTS = 4  # total tries per model (1 call + 3 retries)
BASE_DELAY = 1.0
MAX_DELAY = 30.0
MAX_RETRY_AFTER = 120.0

_RETRY_STATUS = {429, 529}
_RETRY_CLASS_NAMES = {
    "ratelimiterror", "overloadederror", "apiconnectionerror", "connecterror",
    "connectionerror", "connectionreseterror", "readerror", "remoteprotocolerror",
    "serverdisconnectederror", "clientconnectionerror",
}


def _status_of(exc: BaseException) -> int | None:
    for obj in (exc, getattr(exc, "response", None)):
        for attr in ("status_code", "status"):
            v = getattr(obj, attr, None)
            if isinstance(v, int) and not isinstance(v, bool):
                return v
    return None


def _retry_after_of(exc: BaseException) -> float | None:
    raw = getattr(exc, "retry_after", None)
    if raw is None:
        headers = getattr(getattr(exc, "response", None), "headers", None)
        try:
            raw = headers.get("retry-after") if headers is not None else None
        except Exception:
            raw = None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    return v if v >= 0 else None


def classify(exc: BaseException) -> tuple[bool, float | None]:
    """``(retryable, retry_after_seconds)`` for an exception.

    Retryable: HTTP 429/529, overloaded errors, connection resets/failures.
    Everything else (auth, bad request, other 4xx/5xx) is not retried.
    """
    seen = 0
    cur: BaseException | None = exc
    while cur is not None and seen < 4:
        status = _status_of(cur)
        if status is not None:
            return status in _RETRY_STATUS, _retry_after_of(cur)
        if type(cur).__name__.lower() in _RETRY_CLASS_NAMES or isinstance(cur, ConnectionError):
            return True, _retry_after_of(cur)
        if "overloaded" in str(cur).lower():
            return True, _retry_after_of(cur)
        cur = cur.__cause__ or cur.__context__
        seen += 1
    return False, None


def backoff_delay(attempt: int, retry_after: float | None = None) -> float:
    """Seconds to wait before retry number ``attempt`` (1-based).

    Exponential (``BASE_DELAY * 2**(attempt-1)``, capped) with jitter in
    [50%, 100%]; a server ``Retry-After`` is a floor (capped at 120s).
    """
    base = min(MAX_DELAY, BASE_DELAY * (2 ** (attempt - 1)))
    delay = base * (0.5 + random.random() / 2)
    if retry_after is not None:
        delay = max(delay, min(retry_after, MAX_RETRY_AFTER))
    return delay


def fallback_model() -> str | None:
    return (os.environ.get("KODA_FALLBACK_MODEL") or "").strip() or None


def session_budget_usd() -> float | None:
    """``KODA_SESSION_BUDGET_USD`` as a positive float; unset/invalid → no limit."""
    raw = (os.environ.get("KODA_SESSION_BUDGET_USD") or "").strip()
    if not raw:
        return None
    try:
        v = float(raw)
    except ValueError:
        _log.warning("ignoring invalid KODA_SESSION_BUDGET_USD=%r", raw)
        return None
    return v if v > 0 else None


@dataclass
class SessionCost:
    """Accumulates per-turn cost from each turn's final ``Done`` usage."""

    usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    unpriced_turns: int = 0
    models: set = field(default_factory=set)

    def add_turn(self, model: str, usage: Any) -> float | None:
        """Record one turn; returns its dollar cost or ``None`` if unpriced."""
        if usage is None:
            return None
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        c = cost_usd(model, usage)
        if c is None:
            self.unpriced_turns += 1
            return None
        self.usd += c
        self.models.add(model)
        return c

    def budget_exceeded(self, limit: float | None = None) -> bool:
        limit = session_budget_usd() if limit is None else limit
        return limit is not None and self.usd >= limit

    def summary(self) -> str:
        """``/usage`` text: tokens always; dollars only for priced models."""
        line = f"Session usage — input: {self.input_tokens:,}  output: {self.output_tokens:,}"
        if self.models:
            line += f"  cost: ${self.usd:.4f}"
            if self.unpriced_turns:
                line += f" (+{self.unpriced_turns} turn(s) on models with no known price)"
        elif self.unpriced_turns:
            line += "  (cost unavailable: no price known for this model)"
        limit = session_budget_usd()
        if limit is not None:
            line += f"  budget: ${limit:.2f}"
        return line
