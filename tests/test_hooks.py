"""Exercise project hooks through the default graph and turn adapter."""

import json
import shlex
import sys

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage

from coding_agent import agent
from coding_agent.hooks import OUTPUT_LIMIT
from koda.adapters.coding_agent import CodingAgentAdapter
from koda.agent_api import Done, Notice, PermissionRequest


class ScriptedModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def _settings(root, **hooks):
    settings = root / ".koda" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({"hooks": hooks}))


def _python(script):
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"


async def _turn(root, monkeypatch, *, tool=None, inherit_env=True, decision="approve"):
    responses = [AIMessage(content="done")]
    if tool is not None:
        name, args = tool
        responses.insert(0, AIMessage(content="", tool_calls=[
            {"name": name, "args": args, "id": "call-1", "type": "tool_call"},
        ]))
    monkeypatch.setattr(agent, "resolve_model", lambda _: ScriptedModel(responses=responses))
    monkeypatch.setattr(agent._perms, "interrupt_on_map",
                        lambda: {"execute": True, "edit_file": True})
    adapter = CodingAgentAdapter(model="test:model")
    adapter._graph = await agent.build_agent(cwd=root, inherit_env=inherit_env)
    try:
        events = []
        async for event in adapter.stream("Run the test turn", []):
            events.append(event)
            if isinstance(event, PermissionRequest):
                adapter.provide_decisions([{"type": decision} for _ in event.items])
        state = await adapter._graph.aget_state(
            {"configurable": {"thread_id": adapter._thread_id}}
        )
        messages = [m for m in state.values["messages"] if isinstance(m, ToolMessage)]
        assert isinstance(events[-1], Done)
        if tool is not None:
            assert any(isinstance(event, PermissionRequest) for event in events)
        return events, messages
    finally:
        await adapter.aclose()


@pytest.mark.asyncio
async def test_pre_hook_blocks_execute(tmp_path, monkeypatch):
    _settings(tmp_path, pre_tool=[
        {"matcher": "exec*", "command": "cat > pre.json; echo 'Denied by project' >&2; exit 2"},
    ])
    _, messages = await _turn(
        tmp_path, monkeypatch, tool=("execute", {"command": "touch forbidden"}),
    )
    assert not (tmp_path / "forbidden").exists()
    assert messages[-1].status == "error"
    assert "Denied by project" in messages[-1].content
    assert json.loads((tmp_path / "pre.json").read_text()) == {
        "event": "pre_tool", "tool_name": "execute", "tool_input": {"command": "touch forbidden"},
    }


@pytest.mark.asyncio
async def test_pre_hook_blocks_execute_with_non_utf8_stderr(tmp_path, monkeypatch):
    _settings(tmp_path, pre_tool=[{"matcher": "execute", "command": _python(
        "import sys; sys.stderr.buffer.write(b'Denied by project\\xff'); sys.exit(2)"
    )}])
    _, messages = await _turn(
        tmp_path, monkeypatch, tool=("execute", {"command": "touch forbidden"}),
    )
    assert not (tmp_path / "forbidden").exists()
    assert messages[-1].status == "error"
    assert "Denied by project" in messages[-1].content


@pytest.mark.asyncio
async def test_rejected_tool_does_not_run_hooks(tmp_path, monkeypatch):
    _settings(tmp_path, pre_tool=[{"matcher": "*", "command": "touch pre-ran"}],
              post_tool=[{"matcher": "*", "command": "touch post-ran"}])
    await _turn(tmp_path, monkeypatch, tool=("execute", {"command": "touch executed"}),
                decision="reject")
    assert not (tmp_path / "pre-ran").exists()
    assert not (tmp_path / "post-ran").exists()
    assert not (tmp_path / "executed").exists()


@pytest.mark.asyncio
async def test_post_formatter_sees_edit_and_caps_output(tmp_path, monkeypatch):
    target = tmp_path / "sample.txt"
    target.write_text("before")
    _settings(tmp_path, post_tool=[{"matcher": "edit_file", "command": _python(
        "import json, sys; from pathlib import Path; "
        "event = json.load(sys.stdin); "
        "path = Path(event['tool_input']['file_path'].lstrip('/')); "
        "assert path.read_text() == 'after'; "
        "path.write_text('FORMATTED'); print('formatter ran'); print('x' * 6000)"
    )}])
    _, messages = await _turn(tmp_path, monkeypatch, tool=("edit_file", {
        "file_path": "/sample.txt", "old_string": "before", "new_string": "after",
    }))
    assert target.read_text() == "FORMATTED"
    hook_output = messages[-1].content.split("formatter ran", 1)[1]
    assert len("formatter ran" + hook_output) == OUTPUT_LIMIT
    assert hook_output.endswith("[Hook output truncated]")


@pytest.mark.asyncio
async def test_stop_receives_event_at_turn_end(tmp_path, monkeypatch):
    _settings(tmp_path, stop=[
        {"matcher": "execute", "command": "touch wrong-match"},
        {"matcher": "stop", "command": "test -f executed && cat > stop.json && echo finished"},
    ])
    events, _ = await _turn(
        tmp_path, monkeypatch, tool=("execute", {"command": "touch executed"}),
    )
    assert json.loads((tmp_path / "stop.json").read_text()) == {"event": "stop"}
    assert not (tmp_path / "wrong-match").exists()
    assert isinstance(events[-2], Notice)
    assert events[-2].message == "finished"


@pytest.mark.asyncio
async def test_timeout_and_failure_leave_turn_running(tmp_path, monkeypatch):
    _settings(tmp_path, pre_tool=[
        {"matcher": "execute", "command": "sleep 60", "timeout": 0.05},
        {"matcher": "execute", "command": "echo 'hook error' >&2; exit 7"},
    ])
    _, messages = await _turn(
        tmp_path, monkeypatch, tool=("execute", {"command": "touch executed"}),
    )
    assert (tmp_path / "executed").exists()
    assert "timed out after 0.05 seconds" in messages[-1].content
    assert "exit code 7" in messages[-1].content
    assert "hook error" in messages[-1].content


@pytest.mark.parametrize("inherit_env", [False, True])
@pytest.mark.asyncio
async def test_hook_cwd_and_environment_match_execute(tmp_path, monkeypatch, inherit_env):
    monkeypatch.setenv("KODA_TEST_HOOK_ENV", "inherited")
    snapshot = 'pwd; echo "${KODA_TEST_HOOK_ENV:-empty}"'
    _settings(tmp_path, post_tool=[
        {"matcher": "execute", "command": f"({snapshot}) > hook-env"},
    ])
    await _turn(tmp_path, monkeypatch, inherit_env=inherit_env,
                tool=("execute", {"command": f"({snapshot}) > execute-env"}))
    expected = "inherited" if inherit_env else "empty"
    assert (tmp_path / "hook-env").read_text() == (tmp_path / "execute-env").read_text()
    assert (tmp_path / "hook-env").read_text().splitlines() == [str(tmp_path.resolve()), expected]


@pytest.mark.asyncio
async def test_missing_settings_leaves_execute_unchanged(tmp_path, monkeypatch):
    events, messages = await _turn(
        tmp_path, monkeypatch, tool=("execute", {"command": "echo normal"}),
    )
    assert messages[-1].content.strip() == "normal\n\n[Command succeeded with exit code 0]"
    assert not any(isinstance(event, Notice) for event in events)
