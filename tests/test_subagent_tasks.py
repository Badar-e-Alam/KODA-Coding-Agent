"""Background-subagent execution-control registry: the spawn / stop / resume /
restart lifecycle that gives subagents interrupt-and-resume 'memory'.

Uses a fake adapter so the lifecycle is exercised deterministically without a
real model."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import time

import pytest

from koda.agent_api import Done, TextDelta, ToolResult, ToolStart
from koda.subagent_tasks import BackgroundTaskRegistry
from koda.subagent_worktrees import cleanup_worktrees


class FakeAdapter:
    """Minimal KodaAgent-shaped stub. Replays a scripted event list, optionally
    slowly so a test can cancel it mid-stream."""

    def __init__(self, model=None, thread_id=None, script=None, slow=False):
        self.model = model
        self.thread_id = thread_id
        self._script = script or [TextDelta("hello"), Done(usage=None)]
        self._slow = slow
        self.interrupted = False

    async def stream(self, message, history):
        for ev in self._script:
            if self._slow:
                await asyncio.sleep(0.15)
            yield ev

    async def interrupt(self):
        self.interrupted = True

    def provide_decisions(self, decisions):
        pass

    async def aclose(self):
        pass


def make_factory(script=None, slow=False):
    def factory(model, thread_id):
        return FakeAdapter(model=model, thread_id=thread_id, script=script, slow=slow)

    return factory


async def _drain(task):
    if task._task is not None:
        await asyncio.gather(task._task, return_exceptions=True)


@pytest.mark.asyncio
async def test_spawn_runs_to_done_and_captures_result():
    script = [
        ToolStart(tool_id="t1", name="ls", arguments={}),
        ToolResult(tool_id="t1", output="ok", is_error=False),
        TextDelta("final "),
        TextDelta("answer"),
        Done(usage=None),
    ]
    reg = BackgroundTaskRegistry(factory=make_factory(script), model="m", on_update=None)
    tid = reg.spawn("do a thing", "explore")
    task = reg.get(tid)
    await _drain(task)
    assert task.summary.state == "success"
    assert task.summary.tool_count == 1
    assert task.final_text == "final answer"
    assert reg.active_count() == 0


@pytest.mark.asyncio
async def test_stop_cancels_and_marks_stopped():
    long_script = [TextDelta("x")] * 20 + [Done(usage=None)]
    reg = BackgroundTaskRegistry(factory=make_factory(long_script, slow=True), model="m", on_update=None)
    tid = reg.spawn("long job", "plan")
    task = reg.get(tid)
    # let it start
    for _ in range(20):
        await asyncio.sleep(0.05)
        if task.summary.state == "running":
            break
    assert reg.stop(tid) is True
    await _drain(task)
    assert task.summary.state == "cancelled"
    assert task.adapter.interrupted is True  # best-effort unwind was called


@pytest.mark.asyncio
async def test_resume_reruns_same_task():
    reg = BackgroundTaskRegistry(factory=make_factory([TextDelta("hi"), Done(usage=None)]), model="m", on_update=None)
    tid = reg.spawn("job", "general-purpose")
    task = reg.get(tid)
    await _drain(task)
    assert task.summary.state == "success"
    first_adapter = task.adapter
    assert reg.resume(tid, "keep going") is True
    await _drain(task)
    assert task.summary.state == "success"
    assert task.adapter is first_adapter  # resume keeps the SAME thread (memory)


@pytest.mark.asyncio
async def test_restart_uses_fresh_adapter():
    reg = BackgroundTaskRegistry(factory=make_factory([TextDelta("hi"), Done(usage=None)]), model="m", on_update=None)
    tid = reg.spawn("job", "explore")
    task = reg.get(tid)
    await _drain(task)
    first_adapter = task.adapter
    assert reg.restart(tid) is True
    await _drain(task)
    assert task.adapter is not first_adapter  # restart = clean slate, new thread
    assert task.summary.state == "success"


@pytest.mark.asyncio
async def test_update_callback_fires_and_reports_done():
    events: list[tuple[str, bool]] = []

    async def on_update(task, done):
        events.append((task.summary.state, done))

    reg = BackgroundTaskRegistry(
        factory=make_factory([TextDelta("hi"), Done(usage=None)]), model="m", on_update=on_update
    )
    tid = reg.spawn("job", "explore")
    await _drain(reg.get(tid))
    assert any(done for _, done in events)  # a done=True update was emitted
    assert events[-1][0] == "success"


@pytest.mark.asyncio
async def test_list_and_get_unknown():
    reg = BackgroundTaskRegistry(factory=make_factory(), model="m", on_update=None)
    assert reg.get("nope") is None
    assert reg.stop("nope") is False
    assert reg.resume("nope") is False
    tid = reg.spawn("job")
    await _drain(reg.get(tid))
    assert [s.id for s in reg.list()] == [tid]


def git(cwd, *args):
    return subprocess.run(
        ["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def git_project(tmp_path):
    if not shutil.which("git"):
        pytest.skip("requires git")
    git(tmp_path, "init", "-q")
    (tmp_path / "shared.txt").write_text("base")
    git(tmp_path, "add", "shared.txt")
    git(tmp_path, "-c", "user.name=Test", "-c", "user.email=test@localhost", "commit", "-qm", "base")
    return tmp_path


@pytest.mark.asyncio
async def test_editing_tasks_isolate_files_and_publish_completion(git_project, monkeypatch):
    from coding_agent.backend import ReapingShellBackend
    from coding_agent.tools import multi_edit
    from koda import bridge, subagent_tools

    both_running = asyncio.Event()
    edits = []

    class EditingAdapter(FakeAdapter):
        async def stream(self, message, history):
            value = message.rsplit("Task: ", 1)[-1]
            assert await asyncio.to_thread(
                multi_edit.invoke,
                {"path": "shared.txt", "edits": [{"old": "base", "new": value}]},
            ) == "applied 1 edits to shared.txt"
            shell = ReapingShellBackend(root_dir=self.cwd, virtual_mode=True)
            response = await asyncio.to_thread(shell.execute, "cat shared.txt")
            assert response.exit_code == 0
            assert response.output.strip() == value
            edits.append(value)
            if len(edits) == 2:
                both_running.set()
            await asyncio.wait_for(both_running.wait(), timeout=5)
            yield TextDelta(value)

    def factory(model, thread_id, cwd):
        adapter = EditingAdapter(model=model, thread_id=thread_id)
        adapter.cwd = cwd
        return adapter

    events = []
    monkeypatch.setattr(bridge, "emit", events.append)
    ui = bridge.Bridge.__new__(bridge.Bridge)
    ui._pending_task_notes = []
    reg = BackgroundTaskRegistry(
        factory=factory, model="m", cwd=git_project, on_update=ui._on_task_update
    )
    subagent_tools.bind(reg, asyncio.get_running_loop())
    try:
        tasks = [reg.get(reg.spawn(value, role)) for value, role in
                 [("one", "edit"), ("two", "general-purpose")]]
        await asyncio.gather(*(_drain(task) for task in tasks))
        for value, task in zip(("one", "two"), tasks, strict=True):
            path = git_project / ".koda" / "worktrees" / task.id
            branch = f"koda/{task.id}"
            assert task.adapter.cwd == path
            assert (path / "shared.txt").read_text() == value
            assert git(git_project, "show", f"{branch}:shared.txt") == value
            report = json.loads(subagent_tools.check_async_task.invoke({"task_id": task.id}))
            assert report["status"] == "success"
            assert report["branch"] == branch
            assert "shared.txt" in report["diff_summary"]
            done = next(e["task"] for e in events if e["type"] == "task_done" and e["task"]["id"] == task.id)
            assert done["worktree_path"] == str(path)
            assert branch in done["diff_summary"]
            assert any(branch in note and "shared.txt" in note for note in ui._pending_task_notes)
        assert (git_project / "shared.txt").read_text() == "base"
    finally:
        subagent_tools.unbind()
        await reg.aclose()


@pytest.mark.asyncio
async def test_cleanup_merged_expired_and_active_worktrees(git_project):
    class EditingAdapter(FakeAdapter):
        async def stream(self, message, history):
            (self.cwd / "shared.txt").write_text(message)
            yield TextDelta(message)

    def factory(model, thread_id, cwd):
        adapter = EditingAdapter(model=model, thread_id=thread_id)
        adapter.cwd = cwd
        return adapter

    reg = BackgroundTaskRegistry(factory=factory, model="m", cwd=git_project)
    try:
        merged, expired = [reg.get(reg.spawn(message)) for message in ("merge me", "expire me")]
        await asyncio.gather(_drain(merged), _drain(expired))
        assert merged.summary.state == expired.summary.state == "success"
        git(git_project, "merge", "--ff-only", merged.summary.branch)
        marker = expired.workspace.lease_path
        with expired.workspace.activate():
            old = time.time() - 8 * 24 * 60 * 60
            os.utime(marker, (old, old))
            cleanup_worktrees(git_project)
            assert not merged.workspace.path.exists()
            assert expired.workspace.path.exists()
        os.utime(marker, (old, old))
    finally:
        await reg.aclose()
    # Startup also sweeps worktrees left by a different registry.
    later = BackgroundTaskRegistry(factory=factory, model="m", cwd=git_project)
    try:
        assert not expired.workspace.path.exists()
        assert "expire me" in git(git_project, "show", f"{expired.summary.branch}:shared.txt")
    finally:
        await later.aclose()


@pytest.mark.asyncio
async def test_read_only_tasks_and_restarts_keep_their_original_workspace(git_project):
    reg = BackgroundTaskRegistry(factory=lambda **kwargs: FakeAdapter(), model="m", cwd=git_project)
    try:
        for role in ("explore", "plan"):
            task = reg.get(reg.spawn("inspect", role))
            await _drain(task)
            assert task.summary.state == "success"
            assert task.summary.worktree_path is None
        task = reg.get(reg.spawn("edit"))
        await _drain(task)
        path = task.workspace.path
        branch = task.summary.branch
        adapter = task.adapter
        cleanup_worktrees(git_project)
        assert not path.exists()
        assert reg.resume(task.id)
        await _drain(task)
        assert task.adapter is adapter
        assert path.is_dir()
        assert reg.restart(task.id)
        await _drain(task)
        assert task.summary.state == "success"
        assert task.adapter is not adapter
        assert git(path, "branch", "--show-current") == branch
    finally:
        await reg.aclose()


@pytest.mark.asyncio
async def test_non_git_editing_tasks_keep_old_factory_and_emit_one_notice(tmp_path):
    notices = []
    reg = BackgroundTaskRegistry(
        factory=make_factory(), model="m", cwd=tmp_path, on_notice=notices.append
    )
    for _ in range(2):
        task = reg.get(reg.spawn("edit", "edit"))
        await _drain(task)
        assert task.summary.state == "success"
        assert task.summary.worktree_path is None
    assert len(notices) == 1


def test_worktree_lease_blocks_a_second_owner(tmp_path):
    from koda.subagent_worktrees import _lease

    lease = tmp_path / "task"
    with _lease(lease, create=True):
        with pytest.raises(BlockingIOError):
            with _lease(lease):
                pytest.fail("Acquired an active worktree lease")
    with _lease(lease):
        assert lease.is_file()


@pytest.mark.asyncio
async def test_coding_adapter_uses_task_directory_for_file_and_shell_tools(tmp_path, monkeypatch):
    import koda.adapters.coding_agent as adapter_module
    from coding_agent.backend import build_backend
    from coding_agent.tools import multi_edit
    from koda.__main__ import _build_adapter_factory
    from koda.subagent_worktrees import task_cwd

    (tmp_path / "example.txt").write_text("original")

    async def build_agent(*, model, cwd):
        assert cwd == tmp_path
        return build_backend(cwd)

    monkeypatch.setattr(adapter_module, "build_agent", build_agent)
    adapter = _build_adapter_factory("coding_agent")("m", "thread", cwd=tmp_path)
    token = task_cwd.set(tmp_path)
    try:
        await adapter._ensure_graph()
        await asyncio.to_thread(
            multi_edit.invoke,
            {"path": "example.txt", "edits": [{"old": "original", "new": "edited"}]},
        )
        response = await asyncio.to_thread(adapter._graph.execute, "cat example.txt")
    finally:
        task_cwd.reset(token)
    assert response.exit_code == 0
    assert response.output.strip() == "edited"
