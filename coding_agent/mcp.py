"""MCP (Model Context Protocol) tool integration for the coding agent.

KODA consumes external MCP servers as LangGraph tools via the official
``langchain-mcp-adapters`` package (LangChain-authored, the recommended
path documented by both LangGraph and Deep Agents — see
https://docs.langchain.com/oss/python/deepagents/tools). The pattern is:

    async with MultiServerMCPClient(config) as client:
        tools = await client.get_tools()          # List[BaseTool]
        agent = create_deep_agent(..., tools=tools)

This module keeps all MCP wiring in one place: it loads the server config
from ``.mcp.json`` at the project root (the standard convention used by
Claude Code, Cursor, VS Code, and the Deep Agents repo itself), builds a
:class:`~langchain_mcp_adapters.client.MultiServerMCPClient`, and returns
the merged tool list. ``build_agent`` merges those tools with the built-in
``EXTRA_TOOLS`` before calling ``create_deep_agent``.

Everything is **optional and non-fatal**:

  * If ``langchain-mcp-adapters`` isn't installed → return ``[]`` (log once).
  * If no ``.mcp.json`` exists → return ``[]`` (silent — MCP is opt-in).
  * If a server is unreachable / times out → log a warning, skip it, and
    return whatever tools the *other* servers provided. One bad server never
    breaks the agent.

Context7 (https://github.com/upstash/context7) is the default server: it
exposes up-to-date, version-specific library documentation so the agent can
fetch fresh API docs before implementing with an unfamiliar library instead
of relying on (possibly stale) training data. Its two tools are
``resolve-library-id`` and ``query-docs``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

_log = logging.getLogger("koda.mcp")

# Default Context7 config — used when .mcp.json doesn't exist but the user
# has set CONTEXT7_API_KEY (or wants the anonymous, rate-limited endpoint).
# The API key is optional; without it Context7 still works but with lower
# rate limits. We never hardcode a key — it comes from the environment.
_DEFAULT_CONTEXT7_URL = "https://mcp.context7.com/mcp"


def _load_mcp_config(root: Path) -> dict[str, dict[str, Any]]:
    """Load MCP server definitions from ``<root>/.mcp.json``.

    The file uses the standard ``{"mcpServers": {name: spec, ...}}`` shape
    that Claude Code / Cursor / VS Code all share. Specs are translated to
    the form ``MultiServerMCPClient`` expects:

      * stdio  -> ``{"transport": "stdio", "command": ..., "args": [...]}``
      * http   -> ``{"transport": "http", "url": ..., "headers": {...}}``
      * sse    -> ``{"transport": "sse", "url": ..., "headers": {...}}``

    Returns ``{}`` (no servers) when the file is absent or unparseable, so
    the agent still starts cleanly without MCP configured.
    """
    config_path = root / ".mcp.json"
    if not config_path.is_file():
        # No config file — fall back to Context7 via env if an API key is set.
        api_key = os.environ.get("CONTEXT7_API_KEY", "").strip()
        if api_key:
            _log.info("no .mcp.json found; using Context7 with CONTEXT7_API_KEY from env")
            return {
                "context7": {
                    "transport": "http",
                    "url": _DEFAULT_CONTEXT7_URL,
                    "headers": {"CONTEXT7_API_KEY": api_key},
                }
            }
        _log.info("no .mcp.json and no CONTEXT7_API_KEY — MCP tools disabled")
        return {}

    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        _log.warning("failed to parse %s: %s — MCP tools disabled", config_path, exc)
        return {}

    servers_raw = raw.get("mcpServers") or raw.get("mcp_servers") or {}
    if not isinstance(servers_raw, dict):
        _log.warning(".mcp.json 'mcpServers' is not an object — MCP tools disabled")
        return {}

    servers: dict[str, dict[str, Any]] = {}
    for name, spec in servers_raw.items():
        if not isinstance(spec, dict):
            _log.warning("skipping MCP server %r: spec is not an object", name)
            continue
        translated = _translate_spec(name, spec)
        if translated is not None:
            servers[name] = translated
    return servers


def _translate_spec(name: str, spec: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a ``.mcp.json`` server spec to ``MultiServerMCPClient`` form.

    Recognizes both the ``"type": "http"|"sse"`` key (Claude Code style)
    and the bare ``"url"`` / ``"command"`` keys (Cursor style).
    """
    transport = spec.get("transport") or spec.get("type")

    # HTTP / streamable-HTTP
    if transport in ("http", "streamable-http", "streamable_http"):
        url = spec.get("url")
        if not url:
            _log.warning("skipping MCP server %r: http transport missing 'url'", name)
            return None
        out: dict[str, Any] = {"transport": "http", "url": url}
        if spec.get("headers"):
            out["headers"] = spec["headers"]
        return out

    # SSE (deprecated by the MCP spec but still supported)
    if transport == "sse":
        url = spec.get("url")
        if not url:
            _log.warning("skipping MCP server %r: sse transport missing 'url'", name)
            return None
        out = {"transport": "sse", "url": url}
        if spec.get("headers"):
            out["headers"] = spec["headers"]
        return out

    # stdio — spawn a local subprocess
    if transport == "stdio" or "command" in spec:
        command = spec.get("command")
        if not command:
            _log.warning("skipping MCP server %r: stdio transport missing 'command'", name)
            return None
        out = {
            "transport": "stdio",
            "command": command,
            "args": spec.get("args", []),
        }
        if spec.get("env"):
            out["env"] = spec["env"]
        return out

    _log.warning(
        "skipping MCP server %r: unrecognized spec (need 'url'/'command' or 'transport')", name
    )
    return None


def _with_stored_auth(servers: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Attach any stored OAuth token to each http/sse server connection.

    Import is local and failure is swallowed: MCP is optional, and a problem
    reading the credential store must not stop the agent from starting with
    its built-in tools.
    """
    try:
        from koda.mcp_auth import silent_provider
    except Exception:
        return servers

    out: dict[str, dict[str, Any]] = {}
    for name, conn in servers.items():
        out[name] = conn
        if conn.get("transport") not in ("http", "sse") or not conn.get("url"):
            continue
        headers = {k.lower(): v for k, v in (conn.get("headers") or {}).items()}
        if "authorization" in headers:
            continue  # user supplied their own key; leave it alone
        try:
            provider = silent_provider(str(conn["url"]), name)
        except Exception:
            _log.warning("could not load stored MCP credentials for %r", name, exc_info=True)
            continue
        if provider is not None:
            out[name] = {**conn, "auth": provider}
    return out


# Per-server connect budget. Each server gets its own clock, so one slow or
# hanging server can't consume the whole startup budget and starve the others.
_PER_SERVER_TIMEOUT = 30

# Outcome of the most recent :func:`load_mcp_tools` call, one entry per
# configured server: ``{name, ok, tool_count, error?, needs_auth?}``. The
# bridge reads this after a build to tell the user which servers actually came
# up (see ``koda/bridge.py``); without it a server that failed to connect is
# invisible except in the log.
_last_report: list[dict[str, Any]] = []


def last_load_report() -> list[dict[str, Any]]:
    """Per-server outcome of the last tool load, for surfacing in the UI."""
    return [dict(entry) for entry in _last_report]


async def _load_one_server(name: str, connection: dict[str, Any]) -> tuple[list, dict[str, Any]]:
    """Connect to a single server and return ``(tools, report_entry)``.

    Never raises: a failure is returned as a report entry with ``ok: False``
    so one dead server is a missing row rather than a missing tool set.
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    try:
        client = MultiServerMCPClient({name: connection})
        tools = await asyncio.wait_for(client.get_tools(), timeout=_PER_SERVER_TIMEOUT)
    except TimeoutError:
        _log.warning("MCP server %r timed out after %ds", name, _PER_SERVER_TIMEOUT)
        return [], {
            "name": name,
            "ok": False,
            "tool_count": 0,
            "error": f"timeout ({_PER_SERVER_TIMEOUT}s)",
            "needs_auth": False,
        }
    except Exception as exc:
        try:
            from koda.mcp_auth import explain

            detail, needs_auth = explain(exc, name)
        except Exception:
            detail, needs_auth = f"{type(exc).__name__}: {exc}", False
        _log.warning(
            "MCP server %r failed: %s%s", name, detail, " [run /mcp and sign in]" if needs_auth else ""
        )
        return [], {
            "name": name,
            "ok": False,
            "tool_count": 0,
            "error": detail,
            "needs_auth": needs_auth,
        }

    tools = list(tools)
    _log.info("MCP server %r: %d tool(s)", name, len(tools))
    return tools, {"name": name, "ok": True, "tool_count": len(tools)}


async def load_mcp_tools(root: Path) -> list:
    """Load and return all tools from configured MCP servers.

    Returns a flat ``list`` of LangChain ``BaseTool`` objects ready to pass
    to ``create_deep_agent(tools=...)``. Returns ``[]`` when MCP is
    unavailable or unconfigured — the agent then runs with its built-in
    tools only. Never raises: every failure path degrades so a missing or
    failed MCP server can't block agent startup.

    **Each server is loaded through its own client, concurrently.**
    ``MultiServerMCPClient.get_tools()`` fans out over every server inside one
    ``anyio`` task group and lets the first exception abort the group, so a
    single unauthorized server (a cloud one the user hasn't signed into yet)
    used to return *zero* tools for the whole session — Context7's two tools
    disappeared because an unrelated Notion server answered 401. That made
    "add it now, sign in later" a trap. One client per server means a failure
    costs you that server's tools and nothing else.

    See :func:`last_load_report` for the per-server outcome.
    """
    global _last_report
    _last_report = []

    servers = _load_mcp_config(root)
    if not servers:
        return []

    try:
        import langchain_mcp_adapters.client  # noqa: F401
    except ImportError:
        _log.warning(
            "langchain-mcp-adapters not installed — MCP tools disabled. "
            "Install with: pip install langchain-mcp-adapters"
        )
        return []

    # Attach stored OAuth tokens (Notion, Linear, Sentry, …). Servers the user
    # hasn't signed into yet get no provider and will simply fail to connect;
    # `/mcp` reports that as "needs authorization" rather than blocking the
    # agent build on a browser. See koda/mcp_auth.py.
    servers = _with_stored_auth(servers)

    _log.info("loading MCP tools from servers: %s", ", ".join(sorted(servers)))
    ordered = sorted(servers)
    results = await asyncio.gather(
        *(_load_one_server(name, servers[name]) for name in ordered),
        return_exceptions=True,
    )

    tools: list = []
    seen: dict[str, str] = {}  # tool name -> the server that first claimed it
    report: list[dict[str, Any]] = []

    for name, result in zip(ordered, results, strict=False):
        if isinstance(result, BaseException):
            # _load_one_server handles its own errors, so this is a bug or a
            # cancellation rather than a server problem. Record it and move on.
            _log.warning("MCP server %r raised unexpectedly: %r", name, result)
            report.append(
                {
                    "name": name,
                    "ok": False,
                    "tool_count": 0,
                    "error": f"{type(result).__name__}: {result}",
                    "needs_auth": False,
                }
            )
            continue
        server_tools, entry = result
        kept = 0
        for tool in server_tools:
            tool_name = getattr(tool, "name", "")
            owner = seen.get(tool_name)
            if owner is not None:
                # MCP tools arrive under their bare name, so two servers can
                # both expose e.g. `query-docs`. Keeping both would give the
                # model two identically-named tools and make the permission
                # policy (also keyed by bare name) ambiguous. First server in
                # alphabetical order wins, and we say which one lost.
                _log.warning(
                    "MCP tool %r from %r shadowed by %r — skipping the duplicate",
                    tool_name,
                    name,
                    owner,
                )
                continue
            seen[tool_name] = name
            tools.append(tool)
            kept += 1
        entry["tool_count"] = kept
        report.append(entry)

    _last_report = report
    ok = [e["name"] for e in report if e["ok"]]
    failed = [e["name"] for e in report if not e["ok"]]
    _log.info(
        "loaded %d MCP tool(s) from %d/%d server(s)%s",
        len(tools),
        len(ok),
        len(report),
        f" — failed: {', '.join(failed)}" if failed else "",
    )
    return tools
