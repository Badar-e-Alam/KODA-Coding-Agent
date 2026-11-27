"""Retry/backoff, fallback model, price table, and session budget."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from koda import bridge as bridge_mod
from koda import reliability
from koda.adapters.base import BaseAdapter
from koda.agent_api import Done, Notice, TextDelta, ToolResult, Usage
from koda.provider_models import cost_usd, price_for
from koda.reliability import SessionCost, backoff_delay, classify


class RateLimit(Exception):
    def __init__(self, retry_after: float | None = None) -> None:
        super().__init__("429 rate limited")
        self.status_code = 429
        self.retry_after = retry_after


class AuthError(Exception):
    status_code = 401


def _extract(chunk: Any):
    if isinstance(chunk, str):
        yield TextDelta(chunk)
    elif isinstance(chunk, Usage):
        yield chunk


class FakeAdapter(BaseAdapter):
    """Fails with ``error`` for the first ``fail_times`` calls per model
    (``None`` = always for models in ``bad``)."""

    def __init__(self, model="primary", fail_times=2, error=None, bad=("primary",)):
        super().__init__(model=model)
        self._extractors = (_extract,)
        self.fail_times = fail_times
        self.error = error or RateLimit()
        self.bad = set(bad)
        self.calls: list[str] = []
        self.sleeps: list[float] = []

    async def _backoff_sleep(self, delay: float) -> bool:  # injected: no real waiting
        self.sleeps.append(delay)
        return False

    async def _native_stream(self, message, history):
        self.calls.append(self._model)
        n = sum(1 for m in self.calls if m == self._model)
        if self._model in self.bad and (self.fail_times is None or n <= self.fail_times):
            raise self.error
        yield f"ok from {self._model}"
        yield Usage(input_tokens=10, output_tokens=5)


def _run(adapter, text="hi"):
    async def go():
        return [ev async for ev in adapter.stream(text, [])]

    return asyncio.run(go())


def test_two_429s_then_success_one_turn_two_retry_events(monkeypatch):
    monkeypatch.delenv("KODA_FALLBACK_MODEL", raising=False)
    a = FakeAdapter(fail_times=2)
    evs = _run(a)
    notices = [e for e in evs if isinstance(e, Notice)]
    assert len(notices) == 2 and all(n.retry_in is not None for n in notices)
    assert "retrying in" in notices[0].message
    assert [e.content for e in evs if isinstance(e, TextDelta)] == ["ok from primary"]
    assert not [e for e in evs if isinstance(e, ToolResult)]
    assert isinstance(evs[-1], Done) and evs[-1].usage.output_tokens == 5
    assert len(a.sleeps) == 2


def test_always_fails_uses_fallback_for_rest_of_turn_then_resets(monkeypatch):
    monkeypatch.setenv("KODA_FALLBACK_MODEL", "backup")
    a = FakeAdapter(fail_times=None)
    evs = _run(a)
    assert a.calls == ["primary"] * 4 + ["backup"]
    assert len(a.sleeps) == 3  # 4 attempts -> 3 backoffs
    assert any(isinstance(e, Notice) and "fallback model backup" in e.message for e in evs)
    assert [e.content for e in evs if isinstance(e, TextDelta)] == ["ok from backup"]
    assert a.model_name() == "backup"
    # next turn starts on the primary model again
    _run(a)
    assert a.calls[5] == "primary"


def test_always_fails_without_fallback_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("KODA_FALLBACK_MODEL", raising=False)
    a = FakeAdapter(fail_times=None)
    evs = _run(a)
    errs = [e for e in evs if isinstance(e, ToolResult)]
    assert len(errs) == 1 and errs[0].is_error
    assert "gave up after 4 attempts" in errs[0].output
    assert "KODA_FALLBACK_MODEL" in errs[0].output
    assert isinstance(evs[-1], Done)
    assert len(a.calls) == 4


def test_non_retryable_error_is_not_retried(monkeypatch):
    monkeypatch.delenv("KODA_FALLBACK_MODEL", raising=False)
    a = FakeAdapter(fail_times=None, error=AuthError("bad key"))
    evs = _run(a)
    assert a.calls == ["primary"] and a.sleeps == []
    assert any(isinstance(e, ToolResult) and "AuthError" in e.output for e in evs)


def test_midstream_failure_is_not_retried(monkeypatch):
    class Mid(FakeAdapter):
        async def _native_stream(self, message, history):
            self.calls.append(self._model)
            yield "partial"
            raise RateLimit()

    a = Mid()
    evs = _run(a)
    assert a.calls == ["primary"] and a.sleeps == []
    assert [e.content for e in evs if isinstance(e, TextDelta)] == ["partial"]
    assert any(isinstance(e, ToolResult) and e.is_error for e in evs)


def test_cancel_interrupts_backoff():
    class Real(FakeAdapter):
        _backoff_sleep = BaseAdapter._backoff_sleep

    a = Real(fail_times=None)

    async def go():
        out = []
        async def consume():
            async for ev in a.stream("x", []):
                out.append(ev)
        t = asyncio.create_task(consume())
        await asyncio.sleep(0.05)  # inside first backoff (>=0.5s)
        await a.interrupt()
        await asyncio.wait_for(t, timeout=2)
        return out

    evs = asyncio.run(go())
    assert len(a.calls) == 1
    assert isinstance(evs[-1], Done)


def test_classify():
    assert classify(RateLimit(3)) == (True, 3.0)
    e = Exception("x")
    e.status_code = 529
    assert classify(e)[0] is True
    assert classify(ConnectionResetError())[0] is True
    assert classify(Exception("Overloaded"))[0] is True
    assert classify(AuthError())[0] is False
    assert classify(ValueError("nope"))[0] is False
    # status on the cause chain / response headers
    class Resp:
        status_code = 429
        headers = {"retry-after": "7"}
    wrapped = RuntimeError("wrapped")
    inner = Exception("http")
    inner.response = Resp()
    wrapped.__cause__ = inner
    assert classify(wrapped) == (True, 7.0)


def test_backoff_exponential_with_jitter_and_retry_after(monkeypatch):
    monkeypatch.setattr(reliability.random, "random", lambda: 1.0)
    assert [backoff_delay(n) for n in (1, 2, 3)] == [1.0, 2.0, 4.0]
    monkeypatch.setattr(reliability.random, "random", lambda: 0.0)
    assert [backoff_delay(n) for n in (1, 2, 3)] == [0.5, 1.0, 2.0]
    assert backoff_delay(1, retry_after=10) == 10  # Retry-After is honoured
    assert backoff_delay(1, retry_after=0.1) == 0.5  # never shorter than backoff
    assert backoff_delay(1, retry_after=10_000) == reliability.MAX_RETRY_AFTER
    monkeypatch.undo()
    for _ in range(50):
        assert 2.0 <= backoff_delay(3) <= 4.0


def test_retry_after_drives_notice_delay(monkeypatch):
    monkeypatch.delenv("KODA_FALLBACK_MODEL", raising=False)
    a = FakeAdapter(fail_times=1, error=RateLimit(retry_after=12))
    _run(a)
    assert a.sleeps == [12]


# ── prices ───────────────────────────────────────────────────────────


def test_price_known_models_and_name_forms():
    assert price_for("claude-sonnet-4-5").input == 3.0
    assert price_for("anthropic:claude-sonnet-4-20250514").output == 15.0
    assert price_for("openai/gpt-4o-mini").input == 0.15
    assert price_for("gpt-4o").input == 2.5  # not shadowed by gpt-4o-mini


def test_unknown_model_has_no_price_never_zero():
    for m in ("llama3:8b", "claude-opus-4-7", "", None, "gpt-4o-mini-foo"):
        assert price_for(m) is None
        assert cost_usd(m, Usage(input_tokens=1000, output_tokens=1000)) is None


def test_cost_includes_cache_rates():
    u = Usage(input_tokens=1_000_000, output_tokens=1_000_000, cache_read_tokens=1_000_000, cache_write_tokens=1_000_000)
    assert cost_usd("claude-sonnet-4-5", u) == pytest.approx(3 + 15 + 0.30 + 3.75)
    # no cache price (gpt-4o write) -> input rate
    assert cost_usd("gpt-4o", Usage(cache_write_tokens=1_000_000)) == pytest.approx(2.5)


def test_usage_summary_dollars_vs_tokens_only():
    c = SessionCost()
    c.add_turn("claude-sonnet-4-5", Usage(input_tokens=1000, output_tokens=500))
    assert "cost: $0.0105" in c.summary()
    u = SessionCost()
    u.add_turn("some-local-model", Usage(input_tokens=1000, output_tokens=500))
    s = u.summary()
    assert "input: 1,000" in s and "$" not in s.replace("$)", "")
    assert u.usd == 0.0


# ── budget ───────────────────────────────────────────────────────────


def test_budget_env_parsing(monkeypatch):
    monkeypatch.delenv("KODA_SESSION_BUDGET_USD", raising=False)
    assert reliability.session_budget_usd() is None
    for bad in ("abc", "-1", "0", "nan "):
        monkeypatch.setenv("KODA_SESSION_BUDGET_USD", bad)
        assert reliability.session_budget_usd() is None
    monkeypatch.setenv("KODA_SESSION_BUDGET_USD", "1.5")
    assert reliability.session_budget_usd() == 1.5


def test_budget_stops_new_turns(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("KODA_SESSION_BUDGET_USD", "0.01")
    events: list[dict] = []
    monkeypatch.setattr(bridge_mod, "emit", events.append)

    class Adapter(FakeAdapter):
        pass

    b = bridge_mod.Bridge(factory=None, model="claude-sonnet-4-5", thread_id="t")
    b._adapter = Adapter(model="claude-sonnet-4-5", fail_times=0)

    # turn 1 runs (under budget) and costs 1M*3$ -> over the budget
    b._adapter._native_stream = _usage_stream(2_000_000)
    asyncio.run(b._run_turn("first"))
    assert b._cost.usd > 0.01
    assert any(e.get("type") == "info" and "cost:" in e["message"] for e in events)

    events.clear()
    calls_before = list(b._adapter.calls)
    asyncio.run(b._run_turn("second"))
    assert [e["type"] for e in events] == ["error", "turn_end"]
    assert "budget" in events[0]["message"]
    assert b._adapter.calls == calls_before  # model never called


def test_no_budget_means_no_limit(monkeypatch):
    monkeypatch.delenv("KODA_SESSION_BUDGET_USD", raising=False)
    c = SessionCost(usd=1e6)
    assert c.budget_exceeded() is False


def test_notice_event_json():
    assert bridge_mod._event_to_json(Notice("retrying in 2s", retry_in=2.0)) == {
        "type": "info",
        "message": "retrying in 2s",
        "retry_in": 2.0,
    }
    assert "retry_in" not in bridge_mod._event_to_json(Notice("fallback"))


def _usage_stream(tokens: int):
    async def gen(message, history):
        yield "hello"
        yield Usage(input_tokens=tokens, output_tokens=0)

    return gen
