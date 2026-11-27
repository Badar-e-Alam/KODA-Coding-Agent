import asyncio
import json
import sys

import pytest

from koda import __main__ as m
from koda.agent_api import Done, TextDelta, ToolResult, ToolStart, Usage


class FakeAdapter:
    def __init__(self, script, exc=None):
        self.script, self.exc, self.closed = script, exc, False

    async def stream(self, message, history):
        for ev in self.script:
            yield ev
        if self.exc:
            raise self.exc

    async def aclose(self):
        self.closed = True


def run(script, capsys, exc=None, **kw):
    adapter = FakeAdapter(script, exc)
    code = asyncio.run(m._run_oneshot(
        factory=lambda **_: adapter, model="fake", prompt="hi", **kw))
    out = capsys.readouterr()
    assert adapter.closed
    return code, out.out, out.err


SCRIPT = [
    TextDelta("he"), ToolStart("1", "ls", {}), ToolResult("1", "ok"),
    TextDelta("llo"), Done(usage=Usage(10, 5)),
]


@pytest.fixture(autouse=True)
def _no_prices(monkeypatch):
    monkeypatch.delenv("KODA_INPUT_PRICE_PER_MTOK", raising=False)
    monkeypatch.delenv("KODA_OUTPUT_PRICE_PER_MTOK", raising=False)


def test_text_default(capsys):
    code, out, _ = run(SCRIPT, capsys)
    assert code == 0 and out == "hello"


def test_json_shape(capsys):
    code, out, _ = run(SCRIPT, capsys, output_format="json")
    obj = json.loads(out)
    assert code == 0
    assert obj["result"] == "hello" and obj["turns"] == 2
    assert obj["tool_calls"] == 1 and obj["is_error"] is False
    assert obj["input_tokens"] == 10 and obj["output_tokens"] == 5
    assert obj["session_id"].startswith("oneshot-") and "cost_usd" not in obj


def test_json_cost_when_prices_known(capsys, monkeypatch):
    monkeypatch.setenv("KODA_INPUT_PRICE_PER_MTOK", "1000000")
    monkeypatch.setenv("KODA_OUTPUT_PRICE_PER_MTOK", "1000000")
    _, out, _ = run(SCRIPT, capsys, output_format="json")
    assert json.loads(out)["cost_usd"] == 15


def test_stream_json_lines(capsys):
    code, out, _ = run(SCRIPT, capsys, output_format="stream-json")
    events = [json.loads(line) for line in out.splitlines()]
    assert code == 0
    assert [e["type"] for e in events] == [
        "text_delta", "tool_start", "tool_result", "text_delta", "done", "result"]
    assert events[-1]["result"] == "hello" and events[-1]["is_error"] is False


def test_agent_error_exit_1(capsys):
    code, out, _ = run([TextDelta("a")], capsys, exc=RuntimeError("boom"),
                       output_format="json")
    obj = json.loads(out)
    assert code == 1 and obj["is_error"] and obj["reason"] == "error"
    assert "boom" in obj["error"]


def test_stream_json_error_still_ends_with_result(capsys):
    code, out, _ = run([], capsys, exc=RuntimeError("boom"),
                       output_format="stream-json")
    events = [json.loads(line) for line in out.splitlines()]
    assert code == 1
    assert events[0]["type"] == "error" and events[-1]["type"] == "result"


def test_max_turns_exit_3(capsys):
    code, out, _ = run(SCRIPT, capsys, output_format="json", max_turns=1)
    obj = json.loads(out)
    assert code == 3 and obj["is_error"] and obj["reason"] == "max_turns"
    assert obj["turns"] == 1 and obj["result"] == "he"


def test_max_turns_not_hit(capsys):
    code, _, _ = run(SCRIPT, capsys, output_format="json", max_turns=2)
    assert code == 0


def test_max_budget_exit_3(capsys, monkeypatch):
    monkeypatch.setenv("KODA_INPUT_PRICE_PER_MTOK", "1000000")
    monkeypatch.setenv("KODA_OUTPUT_PRICE_PER_MTOK", "1000000")
    script = [TextDelta("a"), Usage(1, 1), TextDelta("b"), Done(usage=None)]
    code, out, _ = run(script, capsys, output_format="json", max_budget_usd=1.5)
    obj = json.loads(out)
    assert code == 3 and obj["reason"] == "max_budget_usd" and obj["is_error"]


def test_budget_unknown_cost_warns_and_does_not_trigger(capsys):
    code, _, err = run(SCRIPT, capsys, output_format="json", max_budget_usd=0.01)
    assert code == 0 and "cannot be enforced" in err


@pytest.mark.parametrize("argv", [
    ["--prompt", "x", "--output-format", "xml"],
    ["--prompt", "x", "--max-turns", "0"],
    ["--prompt", "x", "--max-turns", "abc"],
    ["--prompt", "x", "--max-budget-usd", "-1"],
    ["--output-format", "json"],
])
def test_bad_args_exit_2(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["koda", *argv])
    monkeypatch.setattr(m, "_load_dotenv", lambda: None)
    with pytest.raises(SystemExit) as e:
        m.main()
    assert e.value.code == 2
