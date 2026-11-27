"""coding_agent/compaction.py: manual /compact and the clear-tool-uses rule.

Uses the real deepagents summarization engine with a fake chat model and a fake
graph, so no network and no LLM.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from coding_agent import compaction

SUMMARY = "SUMMARY: user wants a CLI; files a.py and b.py edited."


class FakeGraph:
    def __init__(self, messages, event=None, nodes=("model",)):
        self.values = {"messages": messages, "_summarization_event": event}
        self.nodes = {n: object() for n in nodes}
        self.updates: list[tuple] = []

    async def aget_state(self, config):
        return SimpleNamespace(values=self.values)

    async def aupdate_state(self, config, values, as_node=None):
        self.updates.append((config, values, as_node))

    def get_graph(self):
        return SimpleNamespace(nodes=self.nodes)


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.delenv("KODA_COMPACT_KEEP_MESSAGES", raising=False)
    model = FakeListChatModel(responses=[SUMMARY])
    return compaction.build_manual_compaction_engine(model, backend=None)


def _conversation(n: int) -> list:
    msgs = []
    for i in range(n):
        cls = HumanMessage if i % 2 == 0 else AIMessage
        msgs.append(cls(content=f"message {i}", id=f"m{i}"))
    return msgs


CONFIG = {"configurable": {"thread_id": "t1"}}


def test_compact_thread_keeps_recent_tail_and_writes_summary_event(engine):
    messages = _conversation(10)
    graph = FakeGraph(messages)

    result = asyncio.run(compaction.compact_thread(graph, engine, CONFIG))

    keep = compaction._DEFAULT_KEEP_MESSAGES
    assert result.compacted is True
    assert result.summary == SUMMARY
    assert result.summarized_messages == len(messages) - keep

    assert len(graph.updates) == 1
    config, values, as_node = graph.updates[0]
    assert config == CONFIG
    assert as_node == "model"
    event = values["_summarization_event"]
    assert event["cutoff_index"] == len(messages) - keep
    assert event["file_path"] is None
    assert SUMMARY in str(event["summary_message"].content)

    # What the model sees next: the summary, then the untouched recent tail.
    effective = engine._apply_event_to_messages(messages, event)
    assert SUMMARY in str(effective[0].content)
    assert [m.id for m in effective[1:]] == [m.id for m in messages[-keep:]]


def test_compact_thread_attributes_write_to_idle_node(engine):
    graph = FakeGraph(_conversation(10), nodes=("model", compaction._IDLE_NODE))
    asyncio.run(compaction.compact_thread(graph, engine, CONFIG))
    assert graph.updates[0][2] == compaction._IDLE_NODE


def test_compact_thread_short_conversation_does_nothing(engine):
    graph = FakeGraph(_conversation(1))
    result = asyncio.run(compaction.compact_thread(graph, engine, CONFIG))
    assert result.compacted is False
    assert graph.updates == []


def test_compact_thread_within_keep_window_does_nothing(engine):
    graph = FakeGraph(_conversation(compaction._DEFAULT_KEEP_MESSAGES))
    result = asyncio.run(compaction.compact_thread(graph, engine, CONFIG))
    assert result.compacted is False
    assert graph.updates == []


# ── clear-tool-uses rule ───────────────────────────────────────────────


def _tool_transcript(n: int) -> list:
    msgs: list = [HumanMessage(content="read the files")]
    for i in range(n):
        msgs.append(AIMessage(
            content="",
            tool_calls=[{"name": "read_file", "args": {"file_path": f"/f{i}.py"},
                         "id": f"call{i}"}],
        ))
        msgs.append(ToolMessage(content=f"contents of f{i} " * 50, tool_call_id=f"call{i}"))
    return msgs


def _rule(monkeypatch, trigger="10", keep="2"):
    monkeypatch.setenv("KODA_CONTEXT_EDIT_TRIGGER_TOKENS", trigger)
    monkeypatch.setenv("KODA_CONTEXT_EDIT_KEEP", keep)
    middleware = compaction.build_context_editing_middleware()
    assert middleware is not None
    return middleware.edits[0]


def test_clear_tool_uses_replaces_old_results_and_keeps_latest(monkeypatch):
    rule = _rule(monkeypatch, keep="2")
    msgs = _tool_transcript(5)
    originals = [m.content for m in msgs if isinstance(m, ToolMessage)]

    rule.apply(msgs, count_tokens=lambda _m: 1_000_000)

    results = [m.content for m in msgs if isinstance(m, ToolMessage)]
    assert results[:3] == ["[cleared]"] * 3
    assert results[3:] == originals[3:]
    # Tool inputs stay intact so the model still sees what it asked for.
    calls = [m.tool_calls[0]["args"] for m in msgs if isinstance(m, AIMessage)]
    assert calls == [{"file_path": f"/f{i}.py"} for i in range(5)]


def test_clear_tool_uses_below_trigger_changes_nothing(monkeypatch):
    rule = _rule(monkeypatch, trigger="1000000")
    msgs = _tool_transcript(5)
    before = [m.content for m in msgs]
    rule.apply(msgs, count_tokens=lambda _m: 10)
    assert [m.content for m in msgs] == before


@pytest.mark.parametrize("value", ["0", "off"])
def test_clear_tool_uses_can_be_disabled(monkeypatch, value):
    monkeypatch.setenv("KODA_CONTEXT_EDIT_TRIGGER_TOKENS", value)
    assert compaction.build_context_editing_middleware() is None
