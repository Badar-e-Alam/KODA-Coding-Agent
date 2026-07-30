"""One failing MCP server must not cost you the other servers' tools.

``MultiServerMCPClient.get_tools()`` fans out over every configured server
inside one anyio task group, and the first exception aborts the group — so a
single cloud server the user hadn't signed into yet returned **zero** tools for
the whole session. Context7's two tools vanished because an unrelated Notion
server answered 401, which made "add it now, sign in later" a trap.

``coding_agent.mcp.load_mcp_tools`` now gives each server its own client and
gathers them concurrently. These tests pin that, plus the two things per-server
loading newly has to decide: what happens to duplicate tool names, and what the
caller can find out about the failures.
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class _FakeTool:
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = f"fake {name}"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<FakeTool {self.name}>"


def _install_fake_client(monkeypatch, behaviour: dict[str, object]):
    """Patch MultiServerMCPClient so each server behaves as `behaviour` says.

    A value that is an Exception is raised; a list of names yields those tools;
    the string "hang" never returns (to exercise the per-server timeout).
    """
    import langchain_mcp_adapters.client as client_mod

    class FakeClient:
        def __init__(self, connections: dict):
            # Per-server isolation is the property under test: if the
            # production code ever batches servers into one client again, this
            # assertion fails loudly instead of the test quietly still passing.
            assert len(connections) == 1, f"expected one server per client, got {list(connections)}"
            self._name = next(iter(connections))

        async def get_tools(self):
            outcome = behaviour[self._name]
            if isinstance(outcome, BaseException):
                raise outcome
            if outcome == "hang":
                await asyncio.sleep(3600)
            return [_FakeTool(n) for n in outcome]  # type: ignore[union-attr]

    monkeypatch.setattr(client_mod, "MultiServerMCPClient", FakeClient)


def _write_config(tmpdir: Path, names: list[str]) -> None:
    (tmpdir / ".mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    n: {"transport": "http", "url": f"https://{n}.example/mcp"} for n in names
                }
            }
        )
    )


def test_one_unauthorized_server_keeps_the_others(monkeypatch) -> None:
    """A 401 from one server costs only that server's tools."""
    import httpx

    from coding_agent.mcp import last_load_report, load_mcp_tools

    request = httpx.Request("POST", "https://motion.example/mcp")
    unauthorized = ExceptionGroup(
        "unhandled errors in a TaskGroup (1 sub-exception)",
        [
            httpx.HTTPStatusError(
                "401", request=request, response=httpx.Response(401, request=request)
            )
        ],
    )

    with tempfile.TemporaryDirectory() as d:
        tmpdir = Path(d)
        _write_config(tmpdir, ["context7", "motion"])
        _install_fake_client(
            monkeypatch,
            {"context7": ["resolve-library-id", "query-docs"], "motion": unauthorized},
        )

        tools = asyncio.run(load_mcp_tools(tmpdir))

    names = sorted(t.name for t in tools)
    assert names == ["query-docs", "resolve-library-id"], names

    report = {e["name"]: e for e in last_load_report()}
    assert report["context7"]["ok"] is True
    assert report["context7"]["tool_count"] == 2
    assert report["motion"]["ok"] is False
    # The failure is classified, not just stringified — that's what lets the UI
    # say "press o to sign in" instead of dumping an ExceptionGroup.
    assert report["motion"]["needs_auth"] is True, report["motion"]
    assert "TaskGroup" not in report["motion"]["error"], report["motion"]
    print("  one unauthorized server keeps the others OK")


def test_every_server_failing_yields_no_tools_but_a_full_report(monkeypatch) -> None:
    from coding_agent.mcp import last_load_report, load_mcp_tools

    with tempfile.TemporaryDirectory() as d:
        tmpdir = Path(d)
        _write_config(tmpdir, ["a", "b"])
        _install_fake_client(
            monkeypatch,
            {"a": ConnectionRefusedError("refused"), "b": RuntimeError("boom")},
        )
        tools = asyncio.run(load_mcp_tools(tmpdir))

    assert tools == []
    report = last_load_report()
    assert len(report) == 2, report
    assert all(e["ok"] is False for e in report), report
    assert all(e["error"] for e in report), report
    print("  all-servers-down still reports per server OK")


def test_slow_server_does_not_starve_a_fast_one(monkeypatch) -> None:
    """A hanging server is bounded by its own clock, not a shared budget."""
    import coding_agent.mcp as mcp_mod
    from coding_agent.mcp import last_load_report, load_mcp_tools

    monkeypatch.setattr(mcp_mod, "_PER_SERVER_TIMEOUT", 1)

    with tempfile.TemporaryDirectory() as d:
        tmpdir = Path(d)
        _write_config(tmpdir, ["fast", "stuck"])
        _install_fake_client(monkeypatch, {"fast": ["quick-tool"], "stuck": "hang"})

        loop_start = asyncio.run(_timed(load_mcp_tools(tmpdir)))
    elapsed, tools = loop_start

    assert [t.name for t in tools] == ["quick-tool"], tools
    # Concurrent, so the whole load is bounded by the slowest server's own
    # timeout rather than the sum of them.
    assert elapsed < 2.5, f"took {elapsed:.1f}s — servers may be loading serially"
    report = {e["name"]: e for e in last_load_report()}
    assert report["stuck"]["ok"] is False
    assert "timeout" in report["stuck"]["error"], report["stuck"]
    print("  slow server does not starve a fast one OK")


async def _timed(coro):
    loop = asyncio.get_running_loop()
    start = loop.time()
    result = await coro
    return loop.time() - start, result


def test_duplicate_tool_names_are_deduped_deterministically(monkeypatch) -> None:
    """Two servers exposing the same bare tool name: first alphabetically wins.

    MCP tools arrive without a server prefix, so this is reachable — pointing a
    second entry at the same upstream is enough. Keeping both would hand the
    model two identically-named tools and make the permission policy (also
    keyed by bare name) ambiguous.
    """
    from coding_agent.mcp import last_load_report, load_mcp_tools

    with tempfile.TemporaryDirectory() as d:
        tmpdir = Path(d)
        _write_config(tmpdir, ["alpha", "zulu"])
        _install_fake_client(
            monkeypatch,
            {"alpha": ["query-docs", "only-in-alpha"], "zulu": ["query-docs", "only-in-zulu"]},
        )
        tools = asyncio.run(load_mcp_tools(tmpdir))

    names = sorted(t.name for t in tools)
    assert names == ["only-in-alpha", "only-in-zulu", "query-docs"], names
    report = {e["name"]: e for e in last_load_report()}
    # Both servers are healthy; the shadowed duplicate just isn't counted twice.
    assert report["alpha"]["ok"] is True and report["alpha"]["tool_count"] == 2
    assert report["zulu"]["ok"] is True and report["zulu"]["tool_count"] == 1
    print("  duplicate tool names deduped deterministically OK")


def test_no_config_means_no_report(monkeypatch) -> None:
    from coding_agent.mcp import last_load_report, load_mcp_tools

    with tempfile.TemporaryDirectory() as d:
        monkeypatch.delenv("CONTEXT7_API_KEY", raising=False)
        tools = asyncio.run(load_mcp_tools(Path(d)))
    assert tools == []
    assert last_load_report() == []
    print("  unconfigured MCP stays silent OK")
