"""OAuth for remote MCP servers — browser sign-in, token storage, refresh.

Cloud MCP servers (Notion, Linear, Sentry, GitHub's hosted server, …) don't
take an API key in a header. They speak the MCP authorization spec: the first
unauthenticated request gets a ``401`` with a ``WWW-Authenticate`` header, and
the client is expected to discover the authorization server, register itself
dynamically, run an OAuth 2.1 authorization-code flow with PKCE in the user's
browser, and then send ``Authorization: Bearer …``.

Without this, adding ``https://mcp.notion.com/mcp`` in ``/mcp`` fails with
``ExceptionGroup: unhandled errors in a TaskGroup (1 sub-exception)`` — an
``httpx`` 401 wrapped in the MCP client's task group. See :func:`explain`,
which unwraps that into something a human can act on.

The protocol work is done by the official SDK's
:class:`mcp.client.auth.OAuthClientProvider` (discovery per RFC 8414 / RFC
9728, dynamic client registration per RFC 7591, PKCE, refresh). This module
supplies the three pieces the SDK leaves to the host application:

  * :class:`FileTokenStorage` — where the client registration and tokens live.
  * a **redirect handler** — opens the system browser at the consent page.
  * a **callback handler** — a one-shot loopback HTTP server that catches the
    redirect and hands the authorization code back.

Two provider flavours, because the same server is reached from two very
different places:

  * :func:`interactive_provider` — used only by ``/mcp``'s *authorize* action.
    May open a browser and block for up to five minutes.
  * :func:`silent_provider` — used by agent startup, ``test``, and tool
    listing. Refreshes an existing token but **never** opens a browser; if
    sign-in is needed it raises :class:`AuthorizationRequired` so the caller
    can say "press o to sign in" instead of hanging the agent build on a
    browser nobody is watching.

Tokens are secrets, so they never go near ``.mcp.json`` (a project file people
commit). They live in ``~/.koda/mcp_auth.json``, mode ``0600``, keyed by
server URL.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import socket
import urllib.parse
import webbrowser
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

_log = logging.getLogger("koda.mcp_auth")

# ── where credentials live ─────────────────────────────────────────────

#: Loopback port for the OAuth redirect. Fixed rather than random because the
#: port is baked into the ``redirect_uris`` we register with the authorization
#: server — a different port on the next run would no longer match the
#: registered client. The port actually used is recorded per server so a
#: re-auth reuses it; if it's occupied we register a fresh client (see
#: :func:`_pick_port`).
DEFAULT_REDIRECT_PORT = 33418
_PORT_ATTEMPTS = 8

CLIENT_NAME = "KODA Coding Agent"


def auth_path() -> Path:
    """Path to the credential store (``~/.koda/mcp_auth.json``)."""
    return Path.home() / ".koda" / "mcp_auth.json"


class AuthorizationRequired(Exception):
    """Raised when a server needs an interactive browser sign-in.

    Carries the server name so the UI can point at the right row.
    """

    def __init__(self, server: str, detail: str = "") -> None:
        self.server = server
        self.detail = detail
        super().__init__(f"'{server}' requires authorization{f': {detail}' if detail else ''}")


def _load_store() -> dict[str, Any]:
    path = auth_path()
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        _log.warning("could not read %s: %s", path, exc)
        return {}
    return raw if isinstance(raw, dict) else {}


def _save_store(store: dict[str, Any]) -> None:
    """Write the credential store with owner-only permissions."""
    path = auth_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    # Create with 0600 from the start — never briefly world-readable.
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(store, fh, indent=2)
    os.replace(tmp, path)
    os.chmod(path, 0o600)


def _key(server_url: str) -> str:
    """Store key for a server URL — normalised so trailing slashes match."""
    return server_url.rstrip("/")


def has_credentials(server_url: str) -> bool:
    """True if we hold an access token for this server (fresh or refreshable)."""
    entry = _load_store().get(_key(server_url))
    return bool(isinstance(entry, dict) and (entry.get("tokens") or {}).get("access_token"))


def clear_credentials(server_url: str) -> bool:
    """Forget the tokens and client registration for one server."""
    store = _load_store()
    if _key(server_url) not in store:
        return False
    del store[_key(server_url)]
    _save_store(store)
    return True


# ── TokenStorage ───────────────────────────────────────────────────────


class FileTokenStorage:
    """:class:`mcp.client.auth.TokenStorage` backed by ``mcp_auth.json``.

    Reads and writes are per-call rather than cached: the bridge and the agent
    graph both hold providers for the same server, and a token refreshed by
    one should be visible to the other rather than being clobbered by a stale
    in-memory copy.
    """

    def __init__(self, server_url: str, redirect_port: int | None = None) -> None:
        self._key = _key(server_url)
        self._redirect_port = redirect_port

    def _entry(self) -> dict[str, Any]:
        entry = _load_store().get(self._key)
        return entry if isinstance(entry, dict) else {}

    def _update(self, **fields: Any) -> None:
        store = _load_store()
        entry = store.get(self._key)
        if not isinstance(entry, dict):
            entry = {}
        entry.update(fields)
        if self._redirect_port is not None:
            entry["redirect_port"] = self._redirect_port
        store[self._key] = entry
        _save_store(store)

    async def get_tokens(self):  # -> OAuthToken | None
        from mcp.shared.auth import OAuthToken

        raw = self._entry().get("tokens")
        if not isinstance(raw, dict):
            return None
        try:
            return OAuthToken.model_validate(raw)
        except Exception:
            _log.warning("discarding unreadable tokens for %s", self._key)
            return None

    async def set_tokens(self, tokens) -> None:
        self._update(tokens=tokens.model_dump(exclude_none=True))

    async def get_client_info(self):  # -> OAuthClientInformationFull | None
        from mcp.shared.auth import OAuthClientInformationFull

        entry = self._entry()
        raw = entry.get("client_info")
        if not isinstance(raw, dict):
            return None
        # A registration is tied to the redirect_uri it was created with. If
        # we're about to listen on a different port, that client is useless —
        # drop it so the SDK registers a new one instead of failing the
        # redirect_uri check at the authorization server.
        if self._redirect_port is not None and entry.get("redirect_port") not in (None, self._redirect_port):
            _log.info("redirect port changed for %s — re-registering client", self._key)
            return None
        try:
            return OAuthClientInformationFull.model_validate(raw)
        except Exception:
            _log.warning("discarding unreadable client registration for %s", self._key)
            return None

    async def set_client_info(self, client_info) -> None:
        self._update(client_info=client_info.model_dump(exclude_none=True, mode="json"))


# ── loopback redirect catcher ──────────────────────────────────────────

_SUCCESS_PAGE = (
    "<!doctype html><meta charset=utf-8><title>KODA — signed in</title>"
    "<style>body{font:16px/1.6 -apple-system,system-ui,sans-serif;display:grid;"
    "place-items:center;height:100vh;margin:0;background:#111;color:#eee}"
    "div{text-align:center}b{color:#fb923c}</style>"
    "<div><p><b>KODA</b> is connected.</p><p>You can close this tab and go back to the terminal.</p></div>"
)
_FAILURE_PAGE = (
    "<!doctype html><meta charset=utf-8><title>KODA — sign-in failed</title>"
    "<style>body{font:16px/1.6 -apple-system,system-ui,sans-serif;display:grid;"
    "place-items:center;height:100vh;margin:0;background:#111;color:#eee}"
    "div{text-align:center}b{color:#d97474}</style>"
    "<div><p><b>Sign-in failed.</b></p><p>%s</p><p>Close this tab and try again in KODA.</p></div>"
)


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _pick_port(preferred: int | None) -> int:
    """First free loopback port, starting from the one we used last time."""
    start = preferred or DEFAULT_REDIRECT_PORT
    for offset in range(_PORT_ATTEMPTS):
        port = start + offset
        if _port_free(port):
            return port
    # Nothing in the range is free; let the OS choose and accept that we'll
    # have to register a new client.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _CallbackServer:
    """One-shot loopback HTTP server that captures ``?code=…&state=…``.

    Speaks just enough HTTP to read the request line and answer with a small
    page — no framework, and nothing is ever served beyond the two static
    responses below.
    """

    def __init__(self, port: int, path: str = "/oauth/callback") -> None:
        self.port = port
        self.path = path
        self._result: asyncio.Future[tuple[str, str | None]] | None = None
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._result = loop.create_future()
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", self.port)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=10)
        except (TimeoutError, ConnectionError):
            writer.close()
            return
        try:
            target = line.decode("latin-1").split(" ")[1]
        except (IndexError, UnicodeDecodeError):
            target = "/"
        parsed = urllib.parse.urlparse(target)
        params = urllib.parse.parse_qs(parsed.query)
        code = (params.get("code") or [""])[0]
        state = (params.get("state") or [None])[0]
        error = (params.get("error") or [""])[0]
        description = (params.get("error_description") or [""])[0]

        if parsed.path != self.path:
            body, status = "not found", "404 Not Found"
        elif error:
            body, status = _FAILURE_PAGE % urllib.parse.quote(description or error), "400 Bad Request"
        elif code:
            body, status = _SUCCESS_PAGE, "200 OK"
        else:
            body, status = _FAILURE_PAGE % "No authorization code in the redirect.", "400 Bad Request"

        payload = body.encode("utf-8")
        writer.write(
            f"HTTP/1.1 {status}\r\nContent-Type: text/html; charset=utf-8\r\n"
            f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode("latin-1")
            + payload
        )
        with contextlib.suppress(ConnectionError):
            await writer.drain()
        writer.close()

        if self._result is not None and not self._result.done():
            if code:
                self._result.set_result((code, state))
            elif error:
                self._result.set_exception(
                    RuntimeError(f"authorization denied: {description or error}")
                )

    async def wait(self, timeout: float) -> tuple[str, str | None]:
        assert self._result is not None
        return await asyncio.wait_for(self._result, timeout=timeout)

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()


# ── providers ──────────────────────────────────────────────────────────


def _client_metadata(redirect_uri: str):
    from mcp.shared.auth import OAuthClientMetadata

    return OAuthClientMetadata(
        client_name=CLIENT_NAME,
        redirect_uris=[redirect_uri],  # type: ignore[list-item]
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="client_secret_post",
    )


def interactive_provider(
    server_url: str,
    notify: Callable[[str], None] | None = None,
) -> tuple[Any, Callable[[], Awaitable[None]]]:
    """An ``httpx.Auth`` that will open a browser to sign the user in.

    Returns ``(provider, cleanup)``. Call ``cleanup()`` when the connection
    attempt is over — it shuts the loopback listener down whether or not the
    flow got that far.

    ``notify`` receives human-readable progress (the consent URL, in case the
    browser didn't open — a headless box or a remote SSH session).
    """
    from mcp.client.auth import OAuthClientProvider

    stored_port = (_load_store().get(_key(server_url)) or {}).get("redirect_port")
    port = _pick_port(stored_port if isinstance(stored_port, int) else None)
    redirect_uri = f"http://localhost:{port}/oauth/callback"
    server = _CallbackServer(port)
    say = notify or (lambda _m: None)

    async def redirect_handler(authorization_url: str) -> None:
        # Listen *before* sending the user to the consent page: a fast
        # provider can redirect back before a later start() would be ready.
        await server.start()
        opened = False
        try:
            opened = webbrowser.open(authorization_url)
        except Exception:
            opened = False
        if opened:
            say("opened your browser to finish sign-in…")
        else:
            say("could not open a browser — visit this URL to sign in:")
        say(authorization_url)

    async def callback_handler() -> tuple[str, str | None]:
        try:
            return await server.wait(timeout=300)
        except TimeoutError as exc:
            raise RuntimeError("timed out after 5 minutes waiting for sign-in") from exc

    async def cleanup() -> None:
        await server.close()

    provider = OAuthClientProvider(
        server_url=server_url,
        client_metadata=_client_metadata(redirect_uri),
        storage=FileTokenStorage(server_url, redirect_port=port),
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )
    return provider, cleanup


def silent_provider(server_url: str, server_name: str = "") -> Any | None:
    """An ``httpx.Auth`` that refreshes tokens but never opens a browser.

    Returns ``None`` when we hold no credentials at all — there is nothing to
    attach, and the caller should surface "needs authorization" rather than
    start a flow. When credentials do exist the provider refreshes them as
    needed; if the refresh fails and a full sign-in would be required, the
    handlers raise :class:`AuthorizationRequired` so the failure names the
    server instead of surfacing as an opaque task-group error.
    """
    from mcp.client.auth import OAuthClientProvider

    if not has_credentials(server_url):
        return None

    entry = _load_store().get(_key(server_url)) or {}
    port = entry.get("redirect_port") if isinstance(entry.get("redirect_port"), int) else DEFAULT_REDIRECT_PORT
    name = server_name or server_url

    async def redirect_handler(_authorization_url: str) -> None:
        raise AuthorizationRequired(name, "sign-in expired")

    async def callback_handler() -> tuple[str, str | None]:
        raise AuthorizationRequired(name, "sign-in expired")

    return OAuthClientProvider(
        server_url=server_url,
        client_metadata=_client_metadata(f"http://localhost:{port}/oauth/callback"),
        # No redirect_port here: a silent provider must never invalidate the
        # stored client registration, it only reads and refreshes.
        storage=FileTokenStorage(server_url),
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )


# ── error explanation ──────────────────────────────────────────────────

#: Substrings that mean "the server wants OAuth and we haven't done it".
_AUTH_HINTS = ("401", "unauthorized", "invalid_token", "www-authenticate", "403", "forbidden")


def _flatten(exc: BaseException, depth: int = 0) -> list[BaseException]:
    """Every leaf exception inside (possibly nested) ExceptionGroups."""
    if depth > 6:
        return [exc]
    group = getattr(exc, "exceptions", None)
    if group:
        out: list[BaseException] = []
        for sub in group:
            out.extend(_flatten(sub, depth + 1))
        return out
    cause = exc.__cause__
    if cause is not None and type(exc).__name__ in ("TaskGroupError", "ExceptionGroup"):
        return _flatten(cause, depth + 1)
    return [exc]


def explain(exc: BaseException, server_name: str = "") -> tuple[str, bool]:
    """Turn a connection failure into ``(message, needs_auth)``.

    The MCP client runs its transport inside an ``anyio`` task group, so a
    plain 401 surfaces as ``ExceptionGroup: unhandled errors in a TaskGroup
    (1 sub-exception)`` — technically accurate and completely useless on a
    dashboard row. This digs out the leaves and reports the informative one,
    flagging the "you just need to sign in" case separately so the UI can
    offer that instead of an error.
    """
    leaves = _flatten(exc)
    for leaf in leaves:
        if isinstance(leaf, AuthorizationRequired):
            return ("needs authorization — press o to sign in", True)

    messages: list[str] = []
    for leaf in leaves:
        status = getattr(getattr(leaf, "response", None), "status_code", None)
        text = f"{type(leaf).__name__}: {leaf}".strip()
        if status in (401, 403):
            return ("needs authorization — press o to sign in", True)
        messages.append(text)

    blob = " ".join(messages).lower()
    if any(hint in blob for hint in _AUTH_HINTS):
        return ("needs authorization — press o to sign in", True)

    # Prefer a leaf that says something over a bare group wrapper.
    for text in messages:
        if "unhandled errors in a TaskGroup" not in text:
            return (text[:200], False)
    return (f"{type(exc).__name__}: {exc}"[:200], False)
