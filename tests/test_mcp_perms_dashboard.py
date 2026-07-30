"""Smoke test for the /mcp + /perms backend wiring.

Exercises:
  - koda.mcp_config: read_servers / write_server / remove_server against a
    throwaway .mcp.json
  - koda.tools.permissions.allowed_tools() accessor
  - Bridge._build_mcp_spec (static) + _perms_list / _mcp_list shape

Self-contained: each test creates its own tempdir so it runs identically
standalone (``python tests/test_mcp_perms_dashboard.py``) and under pytest.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

# ensure project root importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _with_tmpdir(fn):
    """Run fn(path) inside a fresh TemporaryDirectory, restoring cwd after."""
    cwd = os.getcwd()

    def wrapper():
        with tempfile.TemporaryDirectory() as d:
            try:
                fn(Path(d))
            finally:
                os.chdir(cwd)
    wrapper.__name__ = fn.__name__
    return wrapper


@_with_tmpdir
def test_mcp_config_roundtrip(tmpdir: Path) -> None:
    from koda import mcp_config

    os.chdir(tmpdir)
    assert mcp_config.read_servers() == [], "expected empty initially"

    # add a cloud http server (Notion-style)
    mcp_config.write_server(
        "notion",
        {"transport": "http", "url": "https://mcp.notion.example/mcp", "headers": {"Authorization": "Bearer x"}},
    )
    servers = mcp_config.read_servers()
    assert len(servers) == 1, servers
    s = servers[0]
    assert s["name"] == "notion", s
    assert s["transport"] == "http", s
    assert s["url"] == "https://mcp.notion.example/mcp", s
    assert s["status"] == "configured", s

    # add a stdio server
    mcp_config.write_server("local", {"transport": "stdio", "command": "node", "args": ["server.js"]})
    servers = mcp_config.read_servers()
    assert {x["name"] for x in servers} == {"notion", "local"}, servers

    # invalid spec raises
    try:
        mcp_config.write_server("bad", {"transport": "wat"})
        raise AssertionError("expected ValueError for bad spec")
    except ValueError:
        pass

    # empty name raises
    try:
        mcp_config.write_server("", {"transport": "http", "url": "x"})
        raise AssertionError("expected ValueError for empty name")
    except ValueError:
        pass

    # remove
    assert mcp_config.remove_server("notion") is True
    assert mcp_config.remove_server("nope") is False
    servers = mcp_config.read_servers()
    assert [x["name"] for x in servers] == ["local"], servers

    # file is valid JSON and human-editable shape
    raw = json.loads((tmpdir / ".mcp.json").read_text(encoding="utf-8"))
    assert "mcpServers" in raw, raw
    print("  mcp_config roundtrip OK")


def test_permissions_accessor() -> None:
    from koda.tools import permissions as p

    p.clear_session_allow()
    p.allow_tool("edit_file")
    p.allow_tool("execute")
    assert p.allowed_tools() == ["edit_file", "execute"], p.allowed_tools()
    assert p.is_allowed("edit_file") is True
    assert p.is_allowed("write_file") is False
    p.clear_session_allow()
    assert p.allowed_tools() == [], p.allowed_tools()
    print("  permissions.allowed_tools() OK")


def test_bridge_build_mcp_spec() -> None:
    import koda.bridge as b

    # Every spec carries the nested `koda` block holding the starting per-tool
    # policy. It defaults to "ask" so a newly added third-party server can't
    # run anything unattended before the user reviews its tools in /mcp.
    ASK = {"defaultPermission": "ask"}

    # http
    spec = b.Bridge._build_mcp_spec({"transport": "http", "url": "https://x.example/mcp"})
    assert spec == {"transport": "http", "url": "https://x.example/mcp", "koda": ASK}, spec
    # http with headers
    spec = b.Bridge._build_mcp_spec({"transport": "http", "url": "https://x", "headers": {"Authorization": "Bearer y"}})
    assert spec == {
        "transport": "http",
        "url": "https://x",
        "headers": {"Authorization": "Bearer y"},
        "koda": ASK,
    }, spec
    # stdio
    spec = b.Bridge._build_mcp_spec({"transport": "stdio", "command": "node", "args": ["a", "b"]})
    assert spec == {"transport": "stdio", "command": "node", "args": ["a", "b"], "koda": ASK}, spec
    # empty headers/args dropped
    spec = b.Bridge._build_mcp_spec({"transport": "http", "url": "https://x", "headers": {}})
    assert spec == {"transport": "http", "url": "https://x", "koda": ASK}, spec
    # an explicit default policy is honoured; anything unrecognised falls back
    spec = b.Bridge._build_mcp_spec({"transport": "http", "url": "https://x", "default_policy": "allow"})
    assert spec["koda"] == {"defaultPermission": "allow"}, spec
    spec = b.Bridge._build_mcp_spec({"transport": "http", "url": "https://x", "default_policy": "bogus"})
    assert spec["koda"] == ASK, spec
    print("  Bridge._build_mcp_spec OK")


def test_bridge_perms_list_shape() -> None:
    """_perms_list emits the expected event shape (uses a stub emit)."""
    import koda.bridge as b
    from koda.tools import permissions as p

    p.clear_session_allow()
    p.allow_tool("edit_file")

    captured: list[dict] = []
    orig = b.emit
    b.emit = captured.append  # type: ignore[assignment]
    try:
        b.Bridge._perms_list(self=None)  # type: ignore[arg-type]
    finally:
        b.emit = orig  # type: ignore[assignment]

    assert len(captured) == 1, captured
    ev = captured[0]
    assert ev["type"] == "perms_state", ev
    assert ev["allowed_tools"] == ["edit_file"], ev
    assert "mutating_tools" in ev and "file_edit_tools" in ev, ev
    p.clear_session_allow()
    print("  Bridge._perms_list shape OK")


@_with_tmpdir
def test_bridge_mcp_list_shape(tmpdir: Path) -> None:
    import koda.bridge as b
    from koda import mcp_config

    os.chdir(tmpdir)
    mcp_config.write_server("ctx7", {"transport": "http", "url": "https://mcp.context7.com/mcp"})

    captured: list[dict] = []
    orig = b.emit
    b.emit = captured.append  # type: ignore[assignment]
    try:
        b.Bridge._mcp_list(self=None)  # type: ignore[arg-type]
    finally:
        b.emit = orig  # type: ignore[assignment]

    assert len(captured) == 1, captured
    ev = captured[0]
    assert ev["type"] == "mcp_servers", ev
    names = {s["name"] for s in ev["servers"]}
    assert "ctx7" in names, ev
    print("  Bridge._mcp_list shape OK")


@_with_tmpdir
def test_mcp_tool_permissions(tmpdir: Path) -> None:
    """Per-tool allow/ask/deny survives a round-trip through .mcp.json."""
    from koda import mcp_config

    os.chdir(tmpdir)
    mcp_config.write_server(
        "notion",
        {"transport": "http", "url": "https://mcp.notion.example/mcp", "koda": {"defaultPermission": "ask"}},
    )
    # Cache what the server exposes, then override two of the three tools.
    assert mcp_config.cache_tools("notion", ["notion-search", "notion-update-page", "notion-fetch"])
    assert mcp_config.set_tool_permission("notion", "notion-search", "allow")
    assert mcp_config.set_tool_permission("notion", "notion-update-page", "deny")

    # Unset tools fall back to the server default rather than to "allow".
    policy = mcp_config.tool_permissions()
    assert policy == {
        "notion-search": "allow",
        "notion-update-page": "deny",
        "notion-fetch": "ask",
    }, policy

    # Bad policies are refused, not silently written.
    assert mcp_config.set_tool_permission("notion", "notion-fetch", "maybe") is False
    assert mcp_config.set_tool_permission("nope", "notion-fetch", "allow") is False
    assert mcp_config.tool_permissions()["notion-fetch"] == "ask"

    # tool="*" applies to every known tool and becomes the new default.
    assert mcp_config.set_tool_permission("notion", "*", "allow")
    assert set(mcp_config.tool_permissions().values()) == {"allow"}
    assert mcp_config.read_servers()[0]["default_policy"] == "allow"

    # Rewriting the server (edited url) keeps the tool list + policy.
    mcp_config.write_server("notion", {"transport": "http", "url": "https://mcp.notion.example/v2"})
    server = mcp_config.read_servers()[0]
    assert server["url"] == "https://mcp.notion.example/v2", server
    assert sorted(server["tool_names"]) == ["notion-fetch", "notion-search", "notion-update-page"], server
    assert server["tool_permissions"]["notion-search"] == "allow", server

    # The file stays valid, standard .mcp.json — extras nest under "koda".
    raw = json.loads((tmpdir / ".mcp.json").read_text())
    spec = raw["mcpServers"]["notion"]
    assert spec["url"] == "https://mcp.notion.example/v2", spec
    assert set(spec["koda"]) == {"defaultPermission", "tools", "permissions"}, spec
    print("  mcp_config tool permissions OK")


@_with_tmpdir
def test_mcp_legacy_config_is_not_retroactively_gated(tmpdir: Path) -> None:
    """A server predating per-tool permissions keeps working ungated.

    Retroactively gating an existing .mcp.json would turn every configured
    server into a prompt on every call, so a spec with no explicit
    `defaultPermission` resolves to allow. Merely *viewing* the tool list
    caches the names but must not change that.
    """
    from koda import mcp_config

    os.chdir(tmpdir)
    (tmpdir / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"ctx7": {"transport": "http", "url": "https://c7.example/mcp"}}})
    )
    assert mcp_config.read_servers()[0]["default_policy"] == "allow"

    mcp_config.cache_tools("ctx7", ["query-docs", "resolve-library-id"])
    assert mcp_config.tool_permissions() == {"query-docs": "allow", "resolve-library-id": "allow"}

    # Order is preserved so the dashboard cursor doesn't jump between reloads.
    assert mcp_config.read_servers()[0]["tool_names"] == ["query-docs", "resolve-library-id"]

    # Expressing intent for one tool leaves the others grandfathered.
    mcp_config.set_tool_permission("ctx7", "query-docs", "deny")
    assert mcp_config.tool_permissions() == {"query-docs": "deny", "resolve-library-id": "allow"}
    print("  legacy .mcp.json not retroactively gated OK")


def test_permissions_decide_mcp() -> None:
    """decide() honours the MCP policy, and PLAN/auto-approve still win."""
    from koda.modes import Mode
    from koda.tools import permissions as p

    p.clear_session_allow()
    p.set_mode(Mode.DEFAULT)
    p.set_mcp_policy({"notion-search": "allow", "notion-update-page": "ask", "notion-delete": "deny"})
    try:
        assert p.decide("notion-search") == "approve"
        assert p.decide("notion-update-page") == "ask"
        assert p.decide("notion-delete") == "reject"
        # An unknown tool is not an MCP tool and not mutating → never gates.
        assert p.decide("some-other-tool") == "approve"

        # Only ask/deny tools need to pause the graph.
        assert p.gated_mcp_tools() == {"notion-update-page", "notion-delete"}
        gated = p.interrupt_on_map()
        assert "notion-update-page" in gated and "notion-delete" in gated, gated
        assert "notion-search" not in gated, gated
        assert set(gated) >= p.MUTATING_TOOLS, gated

        # "always allow" during a session upgrades an ask, but never a deny.
        p.allow_tool("notion-update-page")
        p.allow_tool("notion-delete")
        assert p.decide("notion-update-page") == "approve"
        assert p.decide("notion-delete") == "reject"

        # Plan mode is advisory for MCP tools too.
        p.set_mode(Mode.PLAN)
        assert p.decide("notion-search") == "reject"
        p.set_mode(Mode.DEFAULT)

        # Unattended runs approve everything, including denied tools.
        p.set_auto_approve(True)
        assert p.decide("notion-delete") == "approve"

        assert "deny" in p.reject_message("notion-delete")
    finally:
        p.set_auto_approve(False)
        p.set_mcp_policy({})
        p.clear_session_allow()
        p.set_mode(Mode.DEFAULT)
    print("  permissions.decide() MCP policy OK")


if __name__ == "__main__":
    tests = [
        test_mcp_config_roundtrip,
        test_mcp_tool_permissions,
        test_mcp_legacy_config_is_not_retroactively_gated,
        test_permissions_accessor,
        test_permissions_decide_mcp,
        test_bridge_build_mcp_spec,
        test_bridge_perms_list_shape,
        test_bridge_mcp_list_shape,
    ]
    fails = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            fails += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print()
    if fails:
        print(f"{fails} FAILURE(S)")
        sys.exit(1)
    print("ALL PASS")