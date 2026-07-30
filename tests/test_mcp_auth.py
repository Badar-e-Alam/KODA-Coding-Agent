"""Tests for OAuth support on remote MCP servers (koda/mcp_auth.py).

Covers the parts that don't need a real authorization server:

  - explain(): unwrapping the anyio ExceptionGroup a 401 arrives in, which is
    the difference between "needs authorization — press o to sign in" and
    "ExceptionGroup: unhandled errors in a TaskGroup (1 sub-exception)".
  - FileTokenStorage: round-trip, 0600 permissions, port-change invalidation.
  - the loopback callback server: a real HTTP GET carrying ?code=&state=.
  - requires_auth bookkeeping in .mcp.json.

The browser consent step itself needs a human and a real provider account, so
it is not covered here.
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import stat
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _isolated_home(fn):
    """Point HOME at a tempdir so we never touch the real ~/.koda."""

    def wrapper():
        cwd = os.getcwd()
        old_home = os.environ.get("HOME")
        with tempfile.TemporaryDirectory() as d:
            os.environ["HOME"] = d
            try:
                fn(Path(d))
            finally:
                if old_home is not None:
                    os.environ["HOME"] = old_home
                os.chdir(cwd)

    wrapper.__name__ = fn.__name__
    return wrapper


def test_explain_unwraps_exception_groups() -> None:
    """A 401 buried in a task group reads as 'needs authorization'."""
    import httpx

    from koda import mcp_auth

    request = httpx.Request("POST", "https://mcp.notion.com/mcp")
    for status in (401, 403):
        inner = httpx.HTTPStatusError(
            f"{status}", request=request, response=httpx.Response(status, request=request)
        )
        group = ExceptionGroup("unhandled errors in a TaskGroup (1 sub-exception)", [inner])
        message, needs_auth = mcp_auth.explain(group, "motion")
        assert needs_auth is True, (status, message)
        assert "authoriz" in message, message
        assert "TaskGroup" not in message, message

    # Nested groups are flattened too.
    inner = httpx.HTTPStatusError("401", request=request, response=httpx.Response(401, request=request))
    nested = ExceptionGroup("outer", [ExceptionGroup("inner", [inner])])
    assert mcp_auth.explain(nested)[1] is True

    # A non-auth failure keeps its own message and is not mislabelled.
    refused = ExceptionGroup(
        "unhandled errors in a TaskGroup (1 sub-exception)",
        [ConnectionRefusedError("[Errno 61] Connection refused")],
    )
    message, needs_auth = mcp_auth.explain(refused, "local")
    assert needs_auth is False, message
    assert "Connection refused" in message, message
    assert "TaskGroup" not in message, message

    # AuthorizationRequired from the silent provider is recognised directly.
    wrapped = ExceptionGroup("g", [mcp_auth.AuthorizationRequired("motion", "sign-in expired")])
    assert mcp_auth.explain(wrapped)[1] is True
    print("  explain() unwraps ExceptionGroups OK")


@_isolated_home
def test_token_storage_roundtrip(home: Path) -> None:
    from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

    from koda import mcp_auth

    url = "https://mcp.notion.com/mcp"
    store = mcp_auth.FileTokenStorage(url, redirect_port=33418)

    assert asyncio.run(store.get_tokens()) is None
    assert mcp_auth.has_credentials(url) is False

    asyncio.run(store.set_tokens(OAuthToken(access_token="tok-abc", refresh_token="ref-xyz")))
    assert mcp_auth.has_credentials(url) is True
    got = asyncio.run(store.get_tokens())
    assert got is not None and got.access_token == "tok-abc", got
    assert got.refresh_token == "ref-xyz", got

    # A trailing slash is the same server.
    assert mcp_auth.has_credentials(url + "/") is True

    # Secrets must not be world-readable.
    mode = stat.S_IMODE(mcp_auth.auth_path().stat().st_mode)
    assert mode == 0o600, oct(mode)
    # …and must not live in the project's .mcp.json.
    assert mcp_auth.auth_path().is_relative_to(home), mcp_auth.auth_path()

    info = OAuthClientInformationFull(
        client_id="cid-1",
        redirect_uris=["http://localhost:33418/oauth/callback"],  # type: ignore[list-item]
    )
    asyncio.run(store.set_client_info(info))
    assert asyncio.run(store.get_client_info()).client_id == "cid-1"

    # A registration is bound to its redirect_uri: listening on a different
    # port must drop it so the SDK re-registers instead of being rejected.
    other = mcp_auth.FileTokenStorage(url, redirect_port=33419)
    assert asyncio.run(other.get_client_info()) is None
    # A reader that isn't about to listen (the silent provider) still sees it.
    assert asyncio.run(mcp_auth.FileTokenStorage(url).get_client_info()) is not None

    assert mcp_auth.clear_credentials(url) is True
    assert mcp_auth.has_credentials(url) is False
    assert mcp_auth.clear_credentials(url) is False
    print("  FileTokenStorage round-trip + 0600 OK")


@_isolated_home
def test_silent_provider_needs_credentials(home: Path) -> None:
    """No token → no provider (so the caller reports 'needs authorization')."""
    from mcp.shared.auth import OAuthToken

    from koda import mcp_auth

    url = "https://mcp.notion.com/mcp"
    assert mcp_auth.silent_provider(url, "motion") is None

    asyncio.run(mcp_auth.FileTokenStorage(url).set_tokens(OAuthToken(access_token="t")))
    provider = mcp_auth.silent_provider(url, "motion")
    assert provider is not None

    import httpx

    assert isinstance(provider, httpx.Auth), type(provider)

    # It must never try to open a browser.
    try:
        asyncio.run(provider.context.redirect_handler("https://example.com/consent"))
    except mcp_auth.AuthorizationRequired as exc:
        assert exc.server == "motion", exc
    else:
        raise AssertionError("silent provider tried to start an interactive flow")
    print("  silent_provider gating OK")


@_isolated_home
def test_callback_server_captures_code(home: Path) -> None:
    """The loopback listener returns (code, state) from a real HTTP request."""
    from koda import mcp_auth

    async def scenario():
        port = mcp_auth._pick_port(None)
        server = mcp_auth._CallbackServer(port)
        await server.start()

        async def fetch(path: str) -> bytes:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(f"GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n".encode())
            await writer.drain()
            body = await reader.read()
            writer.close()
            return body

        # An unrelated path is a 404 and must not resolve the flow.
        assert b"404" in await fetch("/favicon.ico")
        assert not server._result.done()

        response = await fetch("/oauth/callback?code=the-code&state=the-state")
        assert b"200 OK" in response, response[:120]
        assert b"You can close this tab" in response, response[:200]

        code, state = await server.wait(timeout=5)
        await server.close()
        return code, state

    code, state = asyncio.run(scenario())
    assert code == "the-code", code
    assert state == "the-state", state
    print("  loopback callback server OK")


@_isolated_home
def test_callback_server_reports_denial(home: Path) -> None:
    """A user who clicks 'deny' gets an error, not a five-minute hang."""
    from koda import mcp_auth

    async def scenario():
        port = mcp_auth._pick_port(None)
        server = mcp_auth._CallbackServer(port)
        await server.start()
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(
            b"GET /oauth/callback?error=access_denied&error_description=User%20said%20no "
            b"HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
        )
        await writer.drain()
        await reader.read()
        writer.close()
        try:
            await server.wait(timeout=5)
        except RuntimeError as exc:
            return str(exc)
        finally:
            await server.close()
        raise AssertionError("expected the denial to raise")

    message = asyncio.run(scenario())
    assert "access_denied" in message or "said no" in message, message
    print("  callback denial surfaces OK")


@_isolated_home
def test_pick_port_skips_busy_ports(home: Path) -> None:
    from koda import mcp_auth

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as taken:
        taken.bind(("127.0.0.1", 0))
        busy = taken.getsockname()[1]
        taken.listen(1)
        chosen = mcp_auth._pick_port(busy)
        assert chosen != busy, chosen
        assert mcp_auth._port_free(chosen), chosen
    print("  _pick_port avoids busy ports OK")


@_isolated_home
def test_requires_auth_is_recorded_not_guessed(home: Path) -> None:
    """`requires_auth` reflects an observed 401, never 'is an http server'."""
    from koda import mcp_config

    os.chdir(home)
    Path(".mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "cloud": {"transport": "http", "url": "https://cloud.example/mcp"},
                    "keyed": {
                        "transport": "http",
                        "url": "https://keyed.example/mcp",
                        "headers": {"Authorization": "Bearer mine"},
                    },
                    "local": {"transport": "stdio", "command": "node"},
                }
            }
        )
    )

    by_name = {s["name"]: s for s in mcp_config.read_servers()}
    # Nothing probed yet → nobody is told to sign in.
    assert by_name["cloud"]["requires_auth"] is False
    assert by_name["cloud"]["oauth"] is True
    # A user-supplied Authorization header opts out of OAuth entirely.
    assert by_name["keyed"]["oauth"] is False
    assert by_name["local"]["oauth"] is False

    mcp_config.set_requires_auth("cloud", True)
    assert {s["name"]: s["requires_auth"] for s in mcp_config.read_servers()}["cloud"] is True

    # A later successful connect clears it, and leaves no empty koda block.
    mcp_config.set_requires_auth("cloud", False)
    assert {s["name"]: s["requires_auth"] for s in mcp_config.read_servers()}["cloud"] is False
    spec = json.loads(Path(".mcp.json").read_text())["mcpServers"]["cloud"]
    assert "koda" not in spec, spec

    # A header-carrying server is never given an OAuth provider.
    from coding_agent.mcp import _with_stored_auth

    conns = _with_stored_auth(
        {"keyed": {"transport": "http", "url": "https://keyed.example/mcp",
                   "headers": {"Authorization": "Bearer mine"}}}
    )
    assert "auth" not in conns["keyed"], conns
    print("  requires_auth bookkeeping OK")


if __name__ == "__main__":
    tests = [
        test_explain_unwraps_exception_groups,
        test_token_storage_roundtrip,
        test_silent_provider_needs_credentials,
        test_callback_server_captures_code,
        test_callback_server_reports_denial,
        test_pick_port_skips_busy_ports,
        test_requires_auth_is_recorded_not_guessed,
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
