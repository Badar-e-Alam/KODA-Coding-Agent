"""MCP server config read/write + single-server connectivity test.

The persistence layer for the ``/mcp`` dashboard. KODA stores MCP server
definitions in ``<cwd>/.mcp.json`` (the standard ``{"mcpServers": {...}}``
shape shared by Claude Code, Cursor, VS Code, and the Deep Agents repo).
The agent reads that file at graph-build time (``coding_agent/mcp.py`` →
``coding_agent/agent.py:build_agent``), so a dashboard *add/remove* must
write the file **and** trigger a graph rebuild (see ``Bridge._rebuild``).

This module wraps the translation logic already living in
``coding_agent/mcp.py`` so the bridge doesn't reach across package
boundaries, and adds:

  * :func:`read_servers`  — current config as a dict, with a ``status``
    field per server (``configured`` / ``error`` + reason).
  * :func:`write_server`  — upsert one server spec into ``.mcp.json``.
  * :func:`remove_server` — delete one server from ``.mcp.json``.
  * :func:`test_server`   — connect to a single server and report the
    tool count (or the failure). Used by the dashboard's "test" action so
    the user can verify a cloud server (Notion, GitHub, …) before
    rebuilding the whole agent around it.
  * :func:`list_tools`    — a server's tools (live or from cache) so the
    dashboard can show a per-tool permission list.
  * :func:`set_tool_permission` / :func:`tool_permissions` — the per-tool
    ``allow`` / ``ask`` / ``deny`` policy the agent enforces.

Per-tool permissions live inside each server's spec under a ``koda`` key::

    "notion": {
      "transport": "http",
      "url": "https://mcp.notion.com/mcp",
      "koda": {
        "defaultPermission": "ask",
        "tools": ["notion-search", "notion-update-page"],
        "permissions": {"notion-search": "allow", "notion-update-page": "ask"}
      }
    }

The nested ``koda`` block keeps our extras out of the way of the standard
keys, so the same ``.mcp.json`` still works in Claude Code / Cursor / VS
Code — they ignore the block, and ``_translate_spec`` never reads it.
``tools`` is a cache of the last successful discovery so the dashboard can
render the permission list instantly (and offline); ``permissions`` is the
policy, consulted by ``koda.tools.permissions.decide``.

All file I/O is atomic-ish (write to a temp then replace) and tolerant of
a missing file — we create one on first write.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from koda import mcp_auth

_log = logging.getLogger("koda.mcp_config")

# ── config file location ────────────────────────────────────────────────


def config_path(cwd: str | Path | None = None) -> Path:
    """Path to the ``.mcp.json`` the agent reads (workspace root)."""
    return Path(cwd or os.getcwd()) / ".mcp.json"


def _load_raw(path: Path) -> dict[str, Any]:
    """Read ``.mcp.json`` as a dict; return ``{}`` if absent/unparseable."""
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        _log.warning("failed to parse %s: %s", path, exc)
        return {}
    if not isinstance(raw, dict):
        return {}
    return raw


def _save_raw(path: Path, raw: dict[str, Any]) -> None:
    """Write ``.mcp.json`` atomically (temp + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".mcp.json.tmp")
    tmp.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _servers_dict(raw: dict[str, Any]) -> dict[str, Any]:
    """Pull the ``mcpServers`` object out of the raw config."""
    s = raw.get("mcpServers") or raw.get("mcp_servers") or {}
    return s if isinstance(s, dict) else {}


# ── per-tool permissions (the nested "koda" block) ─────────────────────

POLICIES = ("allow", "ask", "deny")

#: Written as ``defaultPermission`` for a server added through the dashboard.
#: A third-party MCP server is untrusted code acting on the user's accounts,
#: so nothing it exposes runs unattended until the user opts in.
DEFAULT_POLICY = "ask"

#: Used when a server has **no** ``defaultPermission`` at all — a config
#: written before per-tool permissions existed, or one hand-edited into
#: ``.mcp.json``. Those keep working exactly as they did (MCP tools were
#: never gated), because retroactively gating them would turn every existing
#: setup into a prompt on every call. Recording any intent — adding the
#: server in ``/mcp``, or touching one tool's policy — replaces this with a
#: real ``defaultPermission``.
UNCONFIGURED_POLICY = "allow"


def _koda_block(spec: dict[str, Any]) -> dict[str, Any]:
    """The ``koda`` extras block of a server spec (never ``None``)."""
    block = spec.get("koda")
    return block if isinstance(block, dict) else {}


def _perm_map(spec: dict[str, Any]) -> dict[str, str]:
    """``{tool_name: policy}`` for a server spec, dropping bad entries."""
    raw = _koda_block(spec).get("permissions")
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items() if str(v) in POLICIES}


def _default_policy(spec: dict[str, Any]) -> str:
    """Policy for tools of this server with no explicit entry.

    See :data:`UNCONFIGURED_POLICY` for why a server with no
    ``defaultPermission`` resolves to ``allow`` rather than ``ask``.
    """
    value = str(_koda_block(spec).get("defaultPermission") or "")
    return value if value in POLICIES else UNCONFIGURED_POLICY


def _requires_auth(spec: dict[str, Any]) -> bool:
    """True once we've actually seen this server answer 401/403.

    Distinct from "could use OAuth": plenty of http servers (Context7, for
    one) work anonymously, and telling the user to sign in to those would be
    noise. Only an observed 401 sets this.
    """
    return bool(_koda_block(spec).get("requiresAuth"))


def set_requires_auth(server: str, required: bool, cwd: str | Path | None = None) -> bool:
    """Record (or clear) the observation that a server demands OAuth."""

    def mutate(block: dict[str, Any]) -> None:
        if required:
            block["requiresAuth"] = True
        else:
            block.pop("requiresAuth", None)

    return _mutate_koda_block(server, mutate, cwd)


def _cached_tools(spec: dict[str, Any]) -> list[str]:
    """Tool names remembered from the last successful discovery."""
    raw = _koda_block(spec).get("tools")
    if not isinstance(raw, list):
        return []
    return [str(t) for t in raw if isinstance(t, str)]


# ── public read ────────────────────────────────────────────────────────


def read_servers(cwd: str | Path | None = None) -> list[dict[str, Any]]:
    """Return the configured servers as wire-shaped dicts for the TUI.

    Each entry: ``{name, transport, url?, command?, args?, headers?, env?,
    status, tool_names, tool_permissions, default_policy}`` where ``status``
    is ``"configured"`` (present in the file) — the live connection status is
    only known after :func:`test_server` or a rebuild.
    """
    from coding_agent.mcp import _translate_spec

    path = config_path(cwd)
    raw = _load_raw(path)
    out: list[dict[str, Any]] = []
    for name, spec in _servers_dict(raw).items():
        if not isinstance(spec, dict):
            continue
        translated = _translate_spec(name, spec)
        if translated is None:
            out.append({"name": name, "transport": "?", "status": "error", "error": "unrecognized spec"})
            continue
        entry: dict[str, Any] = {"name": name, "status": "configured"}
        entry.update(translated)
        entry["tool_names"] = _cached_tools(spec)
        entry["tool_permissions"] = _perm_map(spec)
        entry["default_policy"] = _default_policy(spec)
        # Whether we hold an OAuth token. Only meaningful for http/sse; a
        # server carrying its own Authorization header doesn't use OAuth.
        headers = {k.lower(): v for k, v in (translated.get("headers") or {}).items()}
        entry["oauth"] = bool(
            translated.get("transport") in ("http", "sse")
            and translated.get("url")
            and "authorization" not in headers
        )
        entry["authorized"] = bool(
            entry["oauth"] and mcp_auth.has_credentials(str(translated.get("url")))
        )
        entry["requires_auth"] = bool(entry["oauth"] and _requires_auth(spec))
        out.append(entry)
    return out


# ── public write ───────────────────────────────────────────────────────


def write_server(
    name: str,
    spec: dict[str, Any],
    cwd: str | Path | None = None,
) -> None:
    """Upsert one server spec into ``.mcp.json``.

    ``spec`` is accepted in the *user* shape (Claude Code / Cursor style —
    ``transport``/``url``/``command``/``args``/``headers``/``env``) and
    stored verbatim so the file stays human-editable and compatible with
    other MCP clients. Validates via ``_translate_spec`` and raises
    ``ValueError`` for an unrecognised transport / missing field.

    Re-writing an existing server keeps its ``koda`` block, so editing a URL
    or a header never silently drops the tool permissions the user set.
    """
    from coding_agent.mcp import _translate_spec

    name = (name or "").strip()
    if not name:
        raise ValueError("server name is required")
    if _translate_spec(name, spec) is None:
        raise ValueError(
            "unrecognized server spec — need transport 'http'|'sse'|'stdio' "
            "with a 'url' (http/sse) or 'command' (stdio)"
        )

    path = config_path(cwd)
    raw = _load_raw(path)
    if "mcpServers" not in raw and "mcp_servers" not in raw:
        raw["mcpServers"] = {}
    servers = raw.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}
        raw["mcpServers"] = servers
    previous = servers.get(name)
    spec = dict(spec)
    old_block = _koda_block(previous) if isinstance(previous, dict) else {}
    if old_block:
        # Keep the cached tool list and the per-tool policy; anything the
        # caller supplied (e.g. a new defaultPermission) wins.
        spec["koda"] = {**old_block, **_koda_block(spec)}
    servers[name] = spec
    _save_raw(path, raw)


def remove_server(name: str, cwd: str | Path | None = None) -> bool:
    """Delete a server from ``.mcp.json``. Returns True if it existed."""
    path = config_path(cwd)
    raw = _load_raw(path)
    servers = _servers_dict(raw)
    if name not in servers:
        return False
    del servers[name]
    # Re-attach under whichever key was present (prefer mcpServers).
    if "mcpServers" in raw:
        raw["mcpServers"] = servers
    else:
        raw["mcpServers"] = servers
        raw.pop("mcp_servers", None)
    _save_raw(path, raw)
    return True


# ── per-tool permissions ───────────────────────────────────────────────


def _mutate_koda_block(
    name: str,
    mutate: Any,
    cwd: str | Path | None = None,
) -> bool:
    """Apply ``mutate(block)`` to one server's ``koda`` block and save.

    Returns False when the server isn't in ``.mcp.json``. The block is
    created on first use.
    """
    path = config_path(cwd)
    raw = _load_raw(path)
    servers = _servers_dict(raw)
    spec = servers.get(name)
    if not isinstance(spec, dict):
        return False
    block = spec.get("koda")
    if not isinstance(block, dict):
        block = {}
    mutate(block)
    # Don't leave an empty block behind: probing a server that needs no
    # config shouldn't add noise to a file the user reads and edits.
    if block:
        spec["koda"] = block
    else:
        spec.pop("koda", None)
    raw["mcpServers"] = servers
    raw.pop("mcp_servers", None)
    _save_raw(path, raw)
    return True


def set_tool_permission(
    server: str,
    tool: str,
    policy: str,
    cwd: str | Path | None = None,
) -> bool:
    """Set one tool's policy. ``tool="*"`` sets every known tool + default.

    Returns False if the server isn't configured or the policy is invalid.
    """
    if policy not in POLICIES:
        return False

    def mutate(block: dict[str, Any]) -> None:
        perms = block.get("permissions")
        if not isinstance(perms, dict):
            perms = {}
        if tool == "*":
            # Bulk-apply to every discovered tool, and make it the default so
            # tools the server adds later inherit the same answer.
            known = block.get("tools")
            names = [str(t) for t in known] if isinstance(known, list) else list(perms)
            perms = {n: policy for n in names}
            block["defaultPermission"] = policy
        else:
            perms[tool] = policy
        block["permissions"] = perms

    return _mutate_koda_block(server, mutate, cwd)


#: Tool descriptions from this process's live discoveries, keyed by server.
#: They are deliberately *not* written to ``.mcp.json`` — a couple of hundred
#: characters per tool would bury the handful of lines a human edits there.
#: The dashboard re-reads the cached name list constantly (every permission
#: toggle), so without this the descriptions would blink out after the first
#: keypress.
_descriptions: dict[str, dict[str, str]] = {}


def cache_tools(server: str, tools: list[str], cwd: str | Path | None = None) -> bool:
    """Remember a server's tool names so the dashboard works offline.

    Discovery order is preserved: the dashboard indexes rows positionally, so
    re-ordering the list under the cursor would move the selection onto a
    different tool between the keypress and the reload.

    Tools seen for the first time get no ``permissions`` entry — they fall
    through to the server's ``defaultPermission`` (``ask``), so a server that
    grows a new tool can't silently inherit an ``allow``.
    """

    def mutate(block: dict[str, Any]) -> None:
        block["tools"] = list(tools)

    return _mutate_koda_block(server, mutate, cwd)


def _with_descriptions(server: str, names: list[str]) -> list[dict[str, str]]:
    """Pair cached tool names with any description seen this session."""
    known = _descriptions.get(server, {})
    return [{"name": n, "description": known.get(n, "")} for n in names]


def tool_permissions(cwd: str | Path | None = None) -> dict[str, str]:
    """Flat ``{tool_name: policy}`` across every configured server.

    MCP tools reach LangChain under their bare MCP name (no server prefix),
    so the agent-side gate is keyed the same way. Tools that only exist in a
    server's cached list — never explicitly set — resolve to that server's
    ``defaultPermission``, which is how a freshly added server ends up fully
    gated without the user having to touch every row.
    """
    raw = _load_raw(config_path(cwd))
    out: dict[str, str] = {}
    for spec in _servers_dict(raw).values():
        if not isinstance(spec, dict):
            continue
        default = _default_policy(spec)
        for tool in _cached_tools(spec):
            out[tool] = default
        out.update(_perm_map(spec))
    return out


async def list_tools(
    server: str,
    refresh: bool = False,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    """A server's tools plus its current policy, for the dashboard.

    Uses the names cached in ``.mcp.json`` unless ``refresh`` is set or the
    cache is empty, in which case it connects and re-caches. Returns
    ``{server, tools: [{name, description}], permissions, cached, error?}``
    and never raises — a dead server yields an ``error`` and whatever the
    cache still holds, so the user can keep editing permissions offline.
    """
    raw = _load_raw(config_path(cwd))
    spec = _servers_dict(raw).get(server)
    if not isinstance(spec, dict):
        return {
            "server": server,
            "tools": [],
            "permissions": {},
            "cached": True,
            "error": f"server '{server}' not in .mcp.json",
        }

    default = _default_policy(spec)
    cached = _cached_tools(spec)
    perms = {name: default for name in cached}
    perms.update(_perm_map(spec))

    # Serve from cache only when it can actually fill the table. Descriptions
    # are session-scoped (they're not written to .mcp.json), so the first open
    # after a restart has names but no descriptions — rendering that leaves the
    # dashboard's DESCRIPTION column blank, which reads as broken. Reconnect
    # instead; it also re-confirms the server is reachable. If it fails we fall
    # through to the cached names below, so this is never worse than before.
    have_descriptions = any(_descriptions.get(server, {}).get(name) for name in cached)
    if not refresh and cached and have_descriptions:
        return {
            "server": server,
            "tools": _with_descriptions(server, cached),
            "permissions": perms,
            "cached": True,
        }

    result = await test_server(server, cwd)
    if not result.get("ok"):
        return {
            "server": server,
            "tools": _with_descriptions(server, cached),
            "permissions": perms,
            "cached": True,
            "error": str(result.get("error") or "connection failed"),
            "needs_auth": bool(result.get("needs_auth")),
        }

    discovered = result.get("tool_details") or [
        {"name": n, "description": ""} for n in result.get("tools", [])
    ]
    names = [str(t["name"]) for t in discovered]
    _descriptions[server] = {str(t["name"]): str(t.get("description") or "") for t in discovered}
    cache_tools(server, names, cwd)
    perms = {name: default for name in names}
    perms.update({k: v for k, v in _perm_map(spec).items() if k in set(names)})
    return {"server": server, "tools": discovered, "permissions": perms, "cached": False}


# ── connectivity test ──────────────────────────────────────────────────


async def test_server(name: str, cwd: str | Path | None = None) -> dict[str, Any]:
    """Connect to a single configured server and report its tools.

    Returns ``{name, ok, tool_count?, tools?, tool_details?, error?}`` where
    ``tools`` is the bare name list and ``tool_details`` carries the one-line
    descriptions the permission view shows. Used by the dashboard's *test*
    action so the user can verify a cloud server (Notion / GitHub / Slack …)
    before rebuilding the agent around it.
    """
    from coding_agent.mcp import _translate_spec

    path = config_path(cwd)
    raw = _load_raw(path)
    spec = _servers_dict(raw).get(name)
    if not isinstance(spec, dict):
        return {"name": name, "ok": False, "error": f"server '{name}' not in .mcp.json"}
    translated = _translate_spec(name, spec)
    if translated is None:
        return {"name": name, "ok": False, "error": "unrecognized spec"}

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError:
        return {"name": name, "ok": False, "error": "langchain-mcp-adapters not installed"}

    # Attach a bearer token if we already have one (and let it refresh). A
    # server that needs a first-time browser sign-in gets no provider, so the
    # 401 comes back as needs_auth rather than silently opening a browser
    # nobody asked for.
    translated = attach_auth(name, spec, translated)

    try:
        client = MultiServerMCPClient({name: translated})
        tools = await asyncio.wait_for(client.get_tools(), timeout=30)
        # Connected anonymously or with a valid token — either way the row
        # shouldn't nag about signing in.
        set_requires_auth(name, False, cwd)
        return {
            "name": name,
            "ok": True,
            "tool_count": len(tools),
            "tools": [t.name for t in tools],
            "tool_details": [
                {
                    "name": t.name,
                    "description": " ".join((getattr(t, "description", "") or "").split())[:160],
                }
                for t in tools
            ],
        }
    except TimeoutError:
        return {"name": name, "ok": False, "error": "timeout (30s)"}
    except Exception as exc:  # auth fail, unreachable, …
        message, needs_auth = mcp_auth.explain(exc, name)
        if needs_auth:
            set_requires_auth(name, True, cwd)
        return {"name": name, "ok": False, "error": message, "needs_auth": needs_auth}


def attach_auth(
    name: str,
    spec: dict[str, Any],
    translated: dict[str, Any],
) -> dict[str, Any]:
    """Add an OAuth ``auth`` provider to a translated http/sse connection.

    ``langchain-mcp-adapters`` passes ``auth`` straight to its httpx client
    (``StreamableHttpConnection``/``SSEConnection``), and the SDK's
    ``OAuthClientProvider`` *is* an ``httpx.Auth`` — so a stored token rides
    along on every request and refreshes itself. Returns ``translated``
    unchanged for stdio, for a server with an explicit ``Authorization``
    header (an API key the user supplied — don't second-guess it), or when we
    hold no credentials.
    """
    if translated.get("transport") not in ("http", "sse"):
        return translated
    url = translated.get("url")
    if not url:
        return translated
    headers = {k.lower(): v for k, v in (translated.get("headers") or {}).items()}
    if "authorization" in headers:
        return translated
    provider = mcp_auth.silent_provider(str(url), name)
    if provider is None:
        return translated
    out = dict(translated)
    out["auth"] = provider
    return out


async def authorize_server(
    name: str,
    notify: Any = None,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    """Run the interactive OAuth flow for one server, then verify it.

    Opens the user's browser, waits for the redirect, stores the tokens, and
    confirms the connection by listing tools. Returns
    ``{name, ok, tool_count?, tools?, tool_details?, error?}``.
    """
    from coding_agent.mcp import _translate_spec

    raw = _load_raw(config_path(cwd))
    spec = _servers_dict(raw).get(name)
    if not isinstance(spec, dict):
        return {"name": name, "ok": False, "error": f"server '{name}' not in .mcp.json"}
    translated = _translate_spec(name, spec)
    if translated is None:
        return {"name": name, "ok": False, "error": "unrecognized spec"}
    if translated.get("transport") not in ("http", "sse"):
        return {"name": name, "ok": False, "error": "only http/sse servers use OAuth"}
    url = str(translated.get("url") or "")
    if not url:
        return {"name": name, "ok": False, "error": "server has no url"}

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError:
        return {"name": name, "ok": False, "error": "langchain-mcp-adapters not installed"}

    provider, cleanup = mcp_auth.interactive_provider(url, notify)
    connection = dict(translated)
    connection["auth"] = provider
    try:
        client = MultiServerMCPClient({name: connection})
        # Generous: the clock includes the human reading a consent screen.
        tools = await asyncio.wait_for(client.get_tools(), timeout=330)
    except TimeoutError:
        return {"name": name, "ok": False, "error": "timed out waiting for sign-in"}
    except Exception as exc:
        message, _ = mcp_auth.explain(exc, name)
        return {"name": name, "ok": False, "error": message}
    finally:
        await cleanup()

    details = [
        {
            "name": t.name,
            "description": " ".join((getattr(t, "description", "") or "").split())[:160],
        }
        for t in tools
    ]
    names = [str(t["name"]) for t in details]
    _descriptions[name] = {d["name"]: d["description"] for d in details}
    cache_tools(name, names, cwd)
    return {
        "name": name,
        "ok": True,
        "tool_count": len(tools),
        "tools": names,
        "tool_details": details,
    }