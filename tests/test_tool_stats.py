"""koda.tool_stats — the tool-call analyser, and its wiring into the bridge."""

from __future__ import annotations

import asyncio

import pytest

from koda import bridge as bridge_mod
from koda.agent_api import Done, TextDelta, ToolResult, ToolStart
from koda.tool_stats import ToolCallTracker


class FakeClock:
    def __init__(self, t: float = 100.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


# ── ToolCallTracker ──────────────────────────────────────────────────────


def test_empty_summary() -> None:
    assert ToolCallTracker().summary() == {
        "calls": 0,
        "errors": 0,
        "unfinished": [],
        "orphan_results": 0,
        "by_tool": {},
        "repeats": [],
    }


def test_pairs_by_tool_id_out_of_order(clock: FakeClock) -> None:
    t = ToolCallTracker(clock=clock)
    t.start("a", "ls", {"path": "."})
    t.start("b", "read_file", {"path": "x"})
    clock.advance(0.5)
    t.finish("b", "contents")  # finishes before "a"
    clock.advance(0.5)
    t.finish("a", "file1\nfile2")
    s = t.summary()
    assert s["calls"] == 2
    assert s["unfinished"] == []
    assert s["orphan_results"] == 0
    assert s["by_tool"]["read_file"]["total_ms"] == pytest.approx(500.0)
    assert s["by_tool"]["ls"]["total_ms"] == pytest.approx(1000.0)


def test_durations_from_fake_clock(clock: FakeClock) -> None:
    t = ToolCallTracker(clock=clock)
    t.start("1", "grep", {"q": "a"})
    clock.advance(0.25)
    t.finish("1", "x" * 10)
    t.start("2", "grep", {"q": "b"})
    clock.advance(1.5)
    t.finish("2", "y" * 5)
    g = t.summary()["by_tool"]["grep"]
    assert g["calls"] == 2
    assert g["total_ms"] == pytest.approx(1750.0)
    assert g["max_ms"] == pytest.approx(1500.0)
    assert g["output_chars"] == 15


def test_errors_counted(clock: FakeClock) -> None:
    t = ToolCallTracker(clock=clock)
    t.start("1", "execute", {"cmd": "false"})
    t.finish("1", "exit 1", is_error=True)
    t.start("2", "execute", {"cmd": "true"})
    t.finish("2", "")
    t.start("3", "ls", {})
    t.finish("3", "ok")
    s = t.summary()
    assert s["errors"] == 1
    assert s["by_tool"]["execute"]["errors"] == 1
    assert s["by_tool"]["ls"]["errors"] == 0


def test_unfinished_calls(clock: FakeClock) -> None:
    t = ToolCallTracker(clock=clock)
    t.start("1", "ls", {})
    t.start("2", "web_search", {"q": "x"})
    t.finish("1", "ok")
    s = t.summary()
    assert s["calls"] == 2
    assert s["unfinished"] == ["web_search"]
    # an unfinished call adds no time
    assert s["by_tool"]["web_search"] == {
        "calls": 1,
        "errors": 0,
        "total_ms": 0.0,
        "max_ms": 0.0,
        "output_chars": 0,
    }


def test_orphan_results(clock: FakeClock) -> None:
    t = ToolCallTracker(clock=clock)
    t.finish("nope", "boom", is_error=True)
    t.start("1", "ls", {})
    t.finish("1", "ok")
    t.finish("1", "again")  # second finish for an already-finished id
    s = t.summary()
    assert s["orphan_results"] == 2
    assert s["calls"] == 1
    assert s["errors"] == 0


def test_repeat_detection_three_identical(clock: FakeClock) -> None:
    t = ToolCallTracker(clock=clock)
    for i in range(3):
        # key order must not matter
        args = {"path": "a.py", "offset": 0} if i % 2 else {"offset": 0, "path": "a.py"}
        t.start(str(i), "read_file", args)
        t.finish(str(i), "x")
    repeats = t.summary()["repeats"]
    assert len(repeats) == 1
    assert repeats[0]["name"] == "read_file"
    assert repeats[0]["count"] == 3
    assert repeats[0]["arguments"] == {"path": "a.py", "offset": 0}


def test_repeat_detection_different_arguments(clock: FakeClock) -> None:
    t = ToolCallTracker(clock=clock)
    for i in range(3):
        t.start(str(i), "read_file", {"path": f"f{i}.py"})
        t.finish(str(i), "x")
    assert t.summary()["repeats"] == []


def test_repeat_needs_three(clock: FakeClock) -> None:
    t = ToolCallTracker(clock=clock)
    for i in range(2):
        t.start(str(i), "ls", {"path": "."})
    assert t.summary()["repeats"] == []


def test_non_serialisable_arguments_do_not_raise(clock: FakeClock) -> None:
    t = ToolCallTracker(clock=clock)
    runtime = object()
    for i in range(3):
        t.start(str(i), "tool", {"runtime": runtime, "x": 1})
        t.finish(str(i), None)
    t.start("odd", "tool2", "not-a-dict")  # type: ignore[arg-type]
    s = t.summary()
    assert s["calls"] == 4
    assert s["repeats"][0]["count"] == 3
    # The summary is also sent over the JSON bridge protocol.
    import json

    json.dumps(s)


def test_recursive_arguments_are_safe_for_wire(clock: FakeClock) -> None:
    import json

    arguments = {}
    arguments["self"] = arguments
    t = ToolCallTracker(clock=clock)
    for i in range(3):
        t.start(str(i), "tool", arguments)
    repeat = t.summary()["repeats"][0]
    assert repeat["arguments"] == {"self": "<recursive>"}
    json.dumps(t.summary())


def test_reset(clock: FakeClock) -> None:
    t = ToolCallTracker(clock=clock)
    t.start("1", "ls", {})
    t.finish("x", "orphan")
    t.reset()
    assert t.summary() == ToolCallTracker().summary()
    t.start("2", "ls", {})
    t.finish("2", "ok")
    assert t.summary()["calls"] == 1


# ── bridge wiring ────────────────────────────────────────────────────────


class FakeAdapter:
    def __init__(self, events: list) -> None:
        self.events = events

    async def stream(self, text, history):
        for ev in self.events:
            yield ev

    async def interrupt(self) -> None:
        pass


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    # Keep the session store out of the real ~/.koda.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    sent: list[dict] = []
    monkeypatch.setattr(bridge_mod, "emit", sent.append)
    b = bridge_mod.Bridge(factory=None, model="fake:model", thread_id="t1")
    b.sent = sent
    return b


def test_run_turn_emits_tool_stats_before_turn_end(bridge) -> None:
    bridge._adapter = FakeAdapter(
        [
            ToolStart("w", "write_todos", {"todos": [{"content": "a", "status": "pending"}]}),
            ToolResult("w", "updated"),
            ToolStart("1", "ls", {"path": "."}),
            ToolResult("1", "a.py"),
            ToolStart("2", "read_file", {"path": "a.py"}),
            ToolResult("2", "denied", is_error=True),
            ToolStart("3", "execute", {"cmd": "sleep 9"}),  # never finishes
            ToolResult("ghost", "stray"),  # orphan
            TextDelta("done"),
            Done(),
        ]
    )
    asyncio.run(bridge._run_turn("hi"))
    types = [e["type"] for e in bridge.sent]
    assert types[-2:] == ["tool_stats", "turn_end"]
    assert types.count("tool_stats") == 1
    stats = bridge.sent[-2]
    assert stats["scope"] == "turn"
    assert stats["calls"] == 4  # write_todos included
    assert stats["errors"] == 1
    assert stats["unfinished"] == ["execute"]
    assert stats["orphan_results"] == 1
    assert set(stats["by_tool"]) == {"write_todos", "ls", "read_file", "execute"}
    # existing event shapes are unchanged
    hidden = next(e for e in bridge.sent if e["type"] == "tool_start" and e["name"] == "write_todos")
    assert hidden == {
        "type": "tool_start",
        "tool_id": "w",
        "name": "write_todos",
        "arguments": {},
        "hidden": True,
    }
    assert bridge.sent[-1] == {"type": "turn_end", "reply": "done"}


def test_turn_stats_reset_and_session_total(bridge) -> None:
    bridge._adapter = FakeAdapter([ToolStart("1", "ls", {}), ToolResult("1", "ok")])
    asyncio.run(bridge._run_turn("one"))
    bridge._adapter = FakeAdapter([ToolStart("2", "ls", {}), ToolResult("2", "ok")])
    asyncio.run(bridge._run_turn("two"))
    turn_stats = [e for e in bridge.sent if e["type"] == "tool_stats"]
    assert [s["calls"] for s in turn_stats] == [1, 1]

    bridge.sent.clear()
    asyncio.run(bridge._dispatch({"type": "tool_stats"}))
    assert bridge.sent == [
        {"type": "tool_stats", "scope": "session", **bridge._session_tools.summary()}
    ]
    assert bridge.sent[0]["calls"] == 2
    assert bridge.sent[0]["by_tool"]["ls"]["calls"] == 2


def test_empty_turn_emits_zero_stats(bridge) -> None:
    asyncio.run(bridge._run_turn("   "))
    assert [e["type"] for e in bridge.sent] == ["tool_stats", "turn_end"]
    assert bridge.sent[0]["calls"] == 0
