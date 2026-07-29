"""Extra tools layered on top of the deepagents built-ins.

deepagents already provides: `execute` (shell), `read_file`, `write_file`,
`edit_file`, `ls`, `glob`, `grep`, `write_todos`, `task`. This module only
defines tools with **no deepagents equivalent**: `think`, `multi_edit`, web
access, image analysis (Gemma vision via Ollama), read-only git, and the
background-shell trio.

There is deliberately no ``run_tests`` / ``run_type_check`` / ``run_lint``.
They used to live here — ~310 lines that guessed the project's framework from
files on disk, shelled out, and regex-parsed the output into a summary. But
``execute`` already runs the command, and a model reads raw pytest output more
reliably than our regexes summarised it. Every tool definition also ships its
schema on every request, so a tool that duplicates ``execute`` costs tokens on
every turn and buys nothing. The agent just runs ``execute("pytest -q")``.

All tools here use LangChain's `@tool` decorator so they slot directly
into `create_deep_agent(..., tools=EXTRA_TOOLS)`.
"""

import atexit
import base64
import os
import re
import signal
import subprocess
import threading
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel
from tavily import TavilyClient


class FindReplace(BaseModel):
    """One find/replace operation for `multi_edit`."""

    old: str
    new: str


def _enriched_env() -> dict[str, str]:
    """Return os.environ with user-local toolchain bins prepended to PATH.

    `subprocess.run(..., shell=True)` invokes /bin/sh (or cmd on Windows),
    which never sources ~/.bashrc, so version-manager-installed tools
    (nvm, pyenv, cargo, pipx) are invisible by default. Prepend their bin
    dirs so the agent can run things it just installed without restarting.
    """
    home = Path.home()
    candidates: list[str] = []
    nvm_versions = home / ".nvm" / "versions" / "node"
    if nvm_versions.is_dir():
        versions = sorted(
            (p for p in nvm_versions.iterdir() if p.is_dir() and (p / "bin" / "node").exists()),
            key=lambda p: p.name,
        )
        if versions:
            candidates.append(str(versions[-1] / "bin"))
    for sub in (".local/bin", ".cargo/bin", ".pyenv/shims", ".pyenv/bin", ".bun/bin", ".deno/bin"):
        d = home / sub
        if d.is_dir():
            candidates.append(str(d))

    env = os.environ.copy()
    existing = env.get("PATH", "")
    parts = [c for c in candidates if c and c not in existing.split(os.pathsep)]
    if parts:
        env["PATH"] = os.pathsep.join(parts + ([existing] if existing else []))
    return env


# ── think ──────────────────────────────────────────────────────────────


@tool
def think(thought: str) -> str:
    """Scratchpad for reasoning. Use before actions or when stuck.

    Nothing is executed — the act of writing forces structured thinking
    and the thought stays in the conversation for later steps to reference.
    Use for: planning an approach, debugging hypotheses, weighing trade-offs.
    """
    return f"noted: {thought[:80]}{'...' if len(thought) > 80 else ''}"


# ── multi_edit ─────────────────────────────────────────────────────────


@tool
def multi_edit(path: str, edits: list[FindReplace]) -> str:
    """Apply multiple find/replace edits to one file atomically.

    All edits succeed or none are written — if any `old` fails to match
    uniquely (in the file state *after* prior edits in this batch), the
    file is left untouched and an error is returned.

    Args:
        path: target file.
        edits: list of {"old": str, "new": str} entries, applied in order.
    """
    p = Path(path)
    if not p.exists():
        return f"[error] file not found: {path}"
    text = p.read_text()
    original = text
    for i, e in enumerate(edits, 1):
        n = text.count(e.old)
        if n != 1:
            return (
                f"[error] edit {i}: `old` matched {n} times in current state, need exactly 1. "
                f"No changes written — widen `old` with surrounding context."
            )
        text = text.replace(e.old, e.new)
    if text == original:
        return "no changes (edits resolved to no-op)"
    p.write_text(text)
    return f"applied {len(edits)} edits to {path}"


# ── web tools ──────────────────────────────────────────────────────────

# Max redirect hops for web_fetch. Each hop is re-validated (see _public_url_error).
_MAX_REDIRECTS = 5


def _public_url_error(url: str) -> str | None:
    """Return an error string if ``url`` isn't a safe public http(s) URL, else None.

    SSRF guard. ``web_fetch`` runs inside the user's network, so without this a
    model (or a prompt-injected page) could reach loopback, RFC1918 hosts, or the
    cloud metadata endpoint at 169.254.169.254 and exfiltrate credentials.
    """
    import ipaddress
    import socket
    from urllib.parse import urlparse

    try:
        parsed = urlparse(url)
    except ValueError:
        return "[error] invalid URL"
    if parsed.scheme not in ("http", "https"):
        return f"[error] only http/https URLs allowed (got {parsed.scheme!r})"
    host = parsed.hostname
    if not host:
        return "[error] URL missing host"
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return f"[error] cannot resolve host {host!r}"
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        # link_local covers 169.254.0.0/16 — the cloud metadata service.
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return f"[error] refusing to fetch private/internal address ({ip}) for {host!r}"
    return None


@tool
def web_fetch(url: str, max_chars: int = 20_000) -> str:
    """Fetch a URL and return its text content (truncated to `max_chars`).

    Use when you need to read external docs, an API reference, a Stack
    Overflow answer, or any page to inform your work. HTML is stripped to
    body text. Set `max_chars` higher for long pages, lower to save tokens.

    Only public http(s) addresses are reachable; private/loopback/link-local
    hosts are refused, including via redirect.
    """
    try:
        import httpx
    except ImportError:
        return "[error] httpx not installed; cannot fetch URLs"

    # Redirects are followed MANUALLY so every hop gets the SSRF check. With
    # httpx's follow_redirects=True, a public URL that 302s to 169.254.169.254
    # would sail straight past a check done only on the original URL.
    seen = url
    try:
        for _ in range(_MAX_REDIRECTS + 1):
            err = _public_url_error(seen)
            if err:
                return err
            r = httpx.get(
                seen,
                follow_redirects=False,
                timeout=15.0,
                headers={"User-Agent": "koda-coding-agent/1.0"},
            )
            if not r.is_redirect:
                break
            location = r.headers.get("location")
            if not location:
                return f"[error] HTTP {r.status_code} redirect with no Location for {seen}"
            seen = str(httpx.URL(seen).join(location))
        else:
            return f"[error] too many redirects (>{_MAX_REDIRECTS}) starting at {url}"
    except Exception as e:
        return f"[error] fetch failed: {e}"

    if r.status_code >= 400:
        return f"[error] HTTP {r.status_code} for {seen}"
    url = seen
    text = r.text
    ct = (r.headers.get("content-type") or "").lower()
    if "html" in ct:
        text = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", "", text, flags=re.I)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\n[ \t]*\n+", "\n\n", text).strip()
    overflow = max(0, len(text) - max_chars)
    if overflow:
        text = text[:max_chars] + f"\n... [truncated, {overflow} chars omitted]"
    return f"# {url} ({r.status_code}, {ct or 'unknown content-type'})\n{text}"


@tool
def web_search(query: str, max_results: int = 10) -> str:
    """Search the web via Tavily and return results as a numbered list.

    Uses Tavily's ``advanced`` search depth with ``include_answer="advanced"``,
    so the response carries both a synthesised answer (rendered first when
    present) and per-source snippets. Requires ``TAVILY_API_KEY`` in the
    environment.

    Args:
        query: The search query string.
        max_results: Maximum number of result rows to return (default 10).

    Returns:
        A formatted string — an ``Answer:`` block when Tavily synthesised one,
        followed by numbered ``title / url / content`` rows. Errors come back
        prefixed with ``[error]``.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "[error] TAVILY_API_KEY is not set in the environment"



    try:
        client = TavilyClient(api_key)
        response = client.search(
            query=query,
            include_answer="advanced",
            search_depth="advanced",
            max_results=max_results,
            # Hard upper bound so a slow Tavily response can't silently
            # wedge an agent turn. 20 s is comfortably above the p99 of
            # ``advanced`` queries and well below the model's per-turn
            # patience. Override via ``KODA_WEB_SEARCH_TIMEOUT`` if a
            # specific deployment needs longer.
            timeout=int(os.environ.get("KODA_WEB_SEARCH_TIMEOUT", "20")),
        )
    except Exception as e:  # noqa: BLE001
        return f"[error] web_search failed: {e}"

    results = response.get("results") or []
    answer = (response.get("answer") or "").strip()
    if not results and not answer:
        return f"No results found for: {query}"

    blocks: list[str] = []
    if answer:
        blocks.append(f"Answer: {answer}")
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "")
        blocks.append(f"{i}. {title}\n   {url}\n   {content}")
    return "\n\n".join(blocks)


# ── visual tools (Gemma vision via Ollama / Ollama Cloud) ────────────────

_SUPPORTED_IMAGE_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif",
})


def _resolve_ollama_host() -> str:
    """Return the Ollama base URL for the visual tool.

    Resolution order:
    1. ``OLLAMA_BASE_URL`` — explicit override (used as-is).
    2. ``OLLAMA_HOST`` — explicit host (protocol-prefixed if bare).
    3. ``OLLAMA_API_KEY`` present — Ollama Cloud via ``OLLAMA_CLOUD_HOST``
       (defaults to ``https://api.ollama.com`` if not set).
    4. Otherwise raises ``RuntimeError`` (local Ollama daemon is not used).

    Reuses the same resolution logic as ``coding_agent.model`` so the
    tool and the agent talk to the same endpoint.
    """
    explicit = os.environ.get("OLLAMA_BASE_URL")
    if explicit:
        return explicit.rstrip("/")
    host = os.environ.get("OLLAMA_HOST")
    if host:
        if not host.startswith(("http://", "https://")):
            host = f"http://{host}"
        return host.rstrip("/")
    if os.environ.get("OLLAMA_API_KEY"):
        cloud_host = os.environ.get("OLLAMA_CLOUD_HOST", "https://api.ollama.com")
        if not cloud_host.startswith(("http://", "https://")):
            cloud_host = f"https://{cloud_host}"
        return cloud_host.rstrip("/")
    raise RuntimeError(
        "Ollama is not configured. Set OLLAMA_API_KEY for Ollama Cloud, "
        "or set OLLAMA_HOST / OLLAMA_BASE_URL for a custom endpoint."
    )


@tool
def visual_analyze(
    image_path: str,
    prompt: str = (
        "First, extract and list ALL readable text verbatim. Then describe: "
        "1. UI Type: Terminal, web app, IDE, CLI? "
        "2. Layout: Windows, panels, bars, input areas - describe structure. "
        "3. Components: Buttons, menus, status indicators, command history. "
        "4. Text Content: Commands, status messages, error messages, labels. "
        "5. Context: What app/system? What is it doing/trying to do? "
        "Be specific, technical, and complete. For UIs, note colors, fonts, states."
    ),
    model: str | None = None,
    max_tokens: int = 4096,
) -> str:
    """Analyze an image using a vision model via Ollama Cloud or a custom endpoint.

    Sends an image file to a multimodal model and returns the model's
    analysis.  When ``OLLAMA_API_KEY`` is set the tool connects to
    **Ollama Cloud** (``https://api.ollama.com``) and defaults to
    ``gemma4:31b``.  For a custom Ollama endpoint, set ``OLLAMA_HOST`` or
    ``OLLAMA_BASE_URL`` and pass the model tag explicitly.

    Useful for understanding screenshots, UI mockups, diagrams, charts,
    or any visual content the agent cannot read as text.

    Args:
        image_path: Path to the image file to analyze. Supports PNG, JPG,
            JPEG, GIF, WebP, BMP, TIFF formats. Use virtual-absolute paths
            rooted at the project (e.g. ``/screenshots/ui.png``).
        prompt: What to ask about the image. Be specific for best results:
            - ``"Describe the UI layout and components"`` for screenshots
            - ``"What error messages are visible?"`` for error screenshots
            - ``"Describe the diagram and relationships shown"`` for diagrams
            - ``"Extract all text visible in this image"`` for OCR-like tasks
            Default is a general description prompt.
        model: Ollama model tag to use.  Defaults to ``gemma4:31b`` when
            using Ollama Cloud (``OLLAMA_API_KEY`` set).  Required when
            using a custom Ollama endpoint.
        max_tokens: Maximum response tokens. Default 4096.

    Returns:
        The model's text analysis of the image. Errors are prefixed
        with ``[error]``.
    """
    img = Path(image_path)
    if not img.exists():
        return f"[error] image not found: {image_path}"
    if img.suffix.lower() not in _SUPPORTED_IMAGE_EXTENSIONS:
        return (
            f"[error] unsupported image format: {img.suffix}. "
            f"Supported: {sorted(_SUPPORTED_IMAGE_EXTENSIONS)}"
        )

    try:
        import ollama
    except ImportError:
        return "[error] ollama Python package not installed; run: pip install ollama"

    # Validate the file is a readable image (check first few bytes).
    try:
        raw = img.read_bytes()
    except OSError as e:
        return f"[error] cannot read image: {e}"
    if len(raw) < 16:
        return "[error] file is too small to be a valid image"

    # Determine endpoint and default model.
    host = _resolve_ollama_host()
    is_cloud = host.startswith("https://api.ollama.com")
    if is_cloud:
        resolved_model = model or "gemma4:31b"
    else:
        if not model:
            return (
                "[error] no model specified for custom Ollama endpoint; "
                "pass `model` explicitly."
            )
        resolved_model = model

    # For cloud, images must be base64-encoded (server can't read local files).
    # For local Ollama, pass the file path directly.
    if is_cloud:
        image_data = base64.b64encode(raw).decode("utf-8")
        images_payload: list[str] = [image_data]
    else:
        images_payload = [image_path]

    try:
        client = ollama.Client(host=host)
        response = client.chat(
            model=resolved_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": images_payload,
                },
            ],
            options={"num_predict": max_tokens},
        )
    except ollama.ResponseError as e:
        hint = ""
        if "not found" in str(e).lower() or "404" in str(e):
            hint = (
                f"\nHint: model {resolved_model!r} may not be available. "
                f"{'Ensure your Ollama Cloud plan includes this model.' if is_cloud else f'Run: ollama pull {resolved_model}'}"
            )
        return f"[error] Ollama request failed: {e}{hint}"
    except Exception as e:  # noqa: BLE001
        return f"[error] visual analysis failed: {e}"

    content = response.message.content
    if not content:
        return "(model returned empty response)"

    # Truncate very long responses to keep context manageable.
    max_chars = 8000
    overflow = max(0, len(content) - max_chars)
    if overflow:
        content = content[:max_chars] + f"\n... [truncated, {overflow} chars omitted]"

    return content


# ── git tools ──────────────────────────────────────────────────────────


def _git(args: list[str], cwd: str = ".") -> str:
    try:
        r = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except FileNotFoundError:
        return "[error] git is not installed or not on PATH"
    except subprocess.TimeoutExpired:
        return "[error] git timed out"
    if r.returncode != 0 and not r.stdout:
        return f"[error] git {' '.join(args)} exit={r.returncode}\n{r.stderr.strip()}"
    out = r.stdout
    if r.stderr.strip():
        out += f"\n[stderr]\n{r.stderr.strip()}"
    return out or "(no output)"


_GIT_READ_ONLY_SUBCOMMANDS = frozenset({
    "status", "log", "blame", "show", "branch", "tag",
    "ls-files", "rev-parse", "rev-list", "describe", "remote",
    "shortlog", "reflog", "config",  # config is read-only when invoked with --get*
})


@tool
def git(subcommand: str, extra_args: str = "", path: str = ".") -> str:
    """Run a **read-only** git subcommand.

    Replaces the prior ``git_status`` / ``git_log`` / ``git_blame`` trio with
    one slot. The model gets full git semantics for inspection while a
    whitelist keeps it from mutating state — anything that could change
    the repo (``commit``, ``push``, ``reset``, ``checkout``, ``rebase``,
    ``merge``, ``add``, ``rm``, ``stash`` save, ``tag -d``, ``branch -d``)
    is rejected. For diffs see ``git_diff`` (kept separate because its
    flag shape is distinct).

    Args:
        subcommand: a single git subcommand from the read-only set
            (``status``, ``log``, ``blame``, ``show``, ``branch``,
            ``tag``, ``ls-files``, ``rev-parse``, ``rev-list``,
            ``describe``, ``remote``, ``shortlog``, ``reflog``,
            ``config``).
        extra_args: flags / paths to append, e.g. ``"-10 --oneline"`` for
            log, ``"-L 40,80 path/to/file.py"`` for blame, ``"--short
            --branch"`` for status.
        path: repo root. Defaults to the cwd.

    Examples:
        ``git("status", "--short --branch")``
        ``git("log", "-10 --pretty=format:'%h %an %s'")``
        ``git("blame", "-L 40,80 koda/tui/app.py")``
    """
    sc = (subcommand or "").strip().split()[0] if subcommand and subcommand.strip() else ""
    if sc not in _GIT_READ_ONLY_SUBCOMMANDS:
        return (
            f"[error] subcommand {sc!r} is not in the read-only whitelist. "
            f"Allowed: {sorted(_GIT_READ_ONLY_SUBCOMMANDS)}. "
            "For mutating commands, use the `execute` tool with explicit intent."
        )
    # ``shlex.split`` mirrors how a shell would tokenize the extra flags,
    # so quoted values like ``--pretty=format:'%h %s'`` survive intact.
    import shlex
    parts = [sc, *shlex.split(extra_args)] if extra_args else [sc]
    return _git(parts, cwd=path)


@tool
def git_diff(path: str = ".", staged: bool = False, file: str = "") -> str:
    """Show unified diff of unstaged (default) or staged changes.

    Kept separate from the generic ``git`` tool because its flag shape
    (``--cached``, ``-- <path>``) is the one the model gets wrong most
    often when forced to spell out diffs through a generic interface.

    Args:
        path: repo root.
        staged: True for staged diff, False for unstaged.
        file: optional path to limit the diff to one file.
    """
    args = ["diff"] + (["--cached"] if staged else [])
    if file:
        args += ["--", file]
    return _git(args, cwd=path)


# ── background shell (run_in_background / poll / kill) ───────────────────
#
# The framework's ``execute`` tool is synchronous — it blocks until the
# command finishes or its timeout fires. That's wrong for dev servers,
# watchers, and slow builds the agent wants to *start* and then keep working
# alongside. These three tools add the missing capability:
#
#   bash_background(command)  → launch detached, return a `bash_id`
#   bash_output(bash_id)      → read NEW output since the last poll + status
#   kill_bash(bash_id)        → terminate the whole process group
#
# Each command runs in its own session (``start_new_session=True``) so a
# single ``killpg`` reaps the shell *and* every child it spawned — the same
# orphan-avoidance the ReapingShellBackend uses for foreground timeouts.
# Reader threads drain stdout/stderr into a bounded buffer so a chatty
# process can never deadlock on a full pipe or balloon memory.

# Cap on unconsumed output retained per process. A server logging forever
# must not grow this without bound; past the cap we keep the tail and mark it.
_BG_MAX_BUFFER_CHARS = 500_000

_BG_LOCK = threading.Lock()
_BG_PROCS: dict[str, "_BackgroundProc"] = {}
_BG_COUNTER = 0


class _BackgroundProc:
    """A detached subprocess plus its drained, pollable output buffer."""

    def __init__(self, bash_id: str, command: str, proc: subprocess.Popen) -> None:
        self.bash_id = bash_id
        self.command = command
        self.proc = proc
        self._buf: list[str] = []
        self._chars = 0
        self._dropped = False
        self._lock = threading.Lock()
        for stream, prefix in ((proc.stdout, ""), (proc.stderr, "[stderr] ")):
            if stream is not None:
                t = threading.Thread(target=self._pump, args=(stream, prefix), daemon=True)
                t.start()

    def _pump(self, stream, prefix: str) -> None:
        """Forward a stream line-by-line into the buffer until EOF."""
        try:
            for line in iter(stream.readline, ""):
                chunk = f"{prefix}{line}" if prefix else line
                with self._lock:
                    self._buf.append(chunk)
                    self._chars += len(chunk)
                    # Drop oldest chunks past the cap, keeping the tail.
                    while self._chars > _BG_MAX_BUFFER_CHARS and len(self._buf) > 1:
                        self._chars -= len(self._buf.pop(0))
                        self._dropped = True
        except (ValueError, OSError):
            # Stream closed underneath us (process killed) — nothing to drain.
            pass
        finally:
            try:
                stream.close()
            except OSError:
                pass

    def drain(self) -> str:
        """Return buffered output and clear it, so each poll sees only new text."""
        with self._lock:
            out = "".join(self._buf)
            dropped = self._dropped
            self._buf.clear()
            self._chars = 0
            self._dropped = False
        if dropped:
            out = "... [earlier output dropped — buffer cap reached]\n" + out
        return out

    def status(self) -> str:
        rc = self.proc.poll()
        return "running" if rc is None else f"exited (code {rc})"

    def terminate(self) -> None:
        """SIGTERM then SIGKILL the whole process group."""
        try:
            pgid = os.getpgid(self.proc.pid)
        except ProcessLookupError:
            return
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                return
            try:
                self.proc.wait(timeout=3)
                return
            except subprocess.TimeoutExpired:
                continue


def _cleanup_background_procs() -> None:
    """Kill any still-running background processes at interpreter exit.

    Without this, a backgrounded dev server outlives KODA and keeps its port
    bound (and, worse, keeps running unattended). Daemon reader threads die
    with the process; the children would not, so reap them explicitly.
    """
    with _BG_LOCK:
        procs = list(_BG_PROCS.values())
        _BG_PROCS.clear()
    for bp in procs:
        bp.terminate()


atexit.register(_cleanup_background_procs)


@tool
def bash_background(command: str, path: str = ".") -> str:
    """Start a shell command in the **background** and return a ``bash_id``.

    Use this for anything that should keep running while you do other work —
    dev servers, file watchers, long builds, log tailing. Unlike ``execute``
    (which blocks until the command finishes), this returns immediately. Poll
    its output with ``bash_output(bash_id)`` and stop it with
    ``kill_bash(bash_id)``.

    The command runs in its own process session, so ``kill_bash`` reaps it and
    every child it spawned. Output is captured (stdout + stderr) and buffered
    until you poll for it.

    Args:
        command: The shell command to launch (e.g. ``"npm run dev"``,
            ``"python -m http.server 8000"``, ``"pytest -x --lf"``).
        path: Working directory to run in. Defaults to the project root.

    Returns:
        ``started bash_id=<id> pid=<pid>: <command>`` on success, or a string
        prefixed with ``[error]`` if the process could not be launched.

    Examples:
        ``bash_background("npm run dev")`` → start a dev server, keep coding.
        Then ``bash_output("bg_1")`` to see new logs, ``kill_bash("bg_1")`` to stop.
    """
    if not command or not command.strip():
        return "[error] command must be a non-empty string"
    global _BG_COUNTER
    try:
        proc = subprocess.Popen(  # noqa: S602
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered so readline() yields promptly
            cwd=path,
            env=_enriched_env(),
            start_new_session=True,
        )
    except Exception as e:  # noqa: BLE001
        return f"[error] failed to launch: {type(e).__name__}: {e}"

    with _BG_LOCK:
        _BG_COUNTER += 1
        bash_id = f"bg_{_BG_COUNTER}"
        _BG_PROCS[bash_id] = _BackgroundProc(bash_id, command, proc)
    return f"started bash_id={bash_id} pid={proc.pid}: {command}"


@tool
def bash_output(bash_id: str) -> str:
    """Read **new** output from a backgrounded command since the last poll.

    Returns only output produced since the previous ``bash_output`` call for
    this ``bash_id`` (each poll consumes what it returns), plus the current
    status. Poll periodically while a background process runs to watch its
    logs; an empty output block with status ``running`` just means nothing new
    has been printed yet.

    Args:
        bash_id: The id returned by ``bash_background``.

    Returns:
        A header line (``bash_id``, status, command) followed by any new
        output, or ``[error]`` if the id is unknown.
    """
    with _BG_LOCK:
        bp = _BG_PROCS.get(bash_id)
    if bp is None:
        known = ", ".join(sorted(_BG_PROCS)) or "(none)"
        return f"[error] unknown bash_id {bash_id!r}. Active: {known}"
    new_output = bp.drain()
    status = bp.status()
    header = f"bash_id={bash_id} status={status}\ncommand: {bp.command}"
    body = new_output if new_output else "(no new output)"
    # Once the process has exited and its final output has been delivered,
    # drop it from the registry so the id list stays meaningful.
    if status.startswith("exited"):
        with _BG_LOCK:
            _BG_PROCS.pop(bash_id, None)
    return f"{header}\n--new output--\n{body}"


@tool
def kill_bash(bash_id: str) -> str:
    """Terminate a backgrounded command and reap its whole process group.

    Sends SIGTERM (then SIGKILL if it doesn't exit) to the process group
    started by ``bash_background``, so the shell and every child die together.
    Always kill background processes you started once you're done with them —
    a leftover dev server keeps its port bound.

    Args:
        bash_id: The id returned by ``bash_background``.

    Returns:
        Confirmation, any final buffered output, or ``[error]`` for an unknown id.
    """
    with _BG_LOCK:
        bp = _BG_PROCS.pop(bash_id, None)
    if bp is None:
        known = ", ".join(sorted(_BG_PROCS)) or "(none)"
        return f"[error] unknown bash_id {bash_id!r}. Active: {known}"
    final = bp.drain()
    bp.terminate()
    tail = f"\n--final output--\n{final}" if final else ""
    return f"killed bash_id={bash_id}: {bp.command}{tail}"


# ── Registry ───────────────────────────────────────────────────────────


@tool
def ask_user(question: str, options: list[str] | None = None) -> str:
    """Ask the user a clarifying question and wait for their answer.

    Encouraged in **PLAN mode** before drafting — when the user's
    request is ambiguous, the agent should ask one focused question
    instead of guessing at a requirement. Also useful at decision
    points anywhere (which approach, which library, where to put X)
    when an arbitrary pick would be costly to revisit.

    Args:
        question: A focused, single-sentence question. Don't stack
            multiple questions — call ``ask_user`` again if you need
            another answer.
        options: Optional list of 2-9 short choices. The user selects one
            with the arrow keys + Enter — OR ignores the options entirely
            and types a free-text reply in the card's "say something else"
            field, which is returned verbatim. If omitted, the user can
            still free-type, acknowledge with Enter, or cancel with Esc.

    Returns:
        The user's typed reply if they wrote one; otherwise the chosen
        option's verbatim text; ``"(acknowledged)"`` for Enter on an
        options-less prompt with no text typed; or an empty string on Esc
        (declined). Because the user can ALWAYS free-type, treat the answer
        as free-form text — do not assume it is one of the options you
        offered.

    Examples:
        ``ask_user("Should this use SQLite or Postgres for storage?", ["SQLite", "Postgres"])``
        ``ask_user("This change touches public API. Add a deprecation period?", ["Yes, deprecate first", "No, break now"])``
    """
    from koda.tools import ask_user as _ask

    return _ask.ask(question, options or [])


def _background_subagent_tools():
    """Background-subagent control tools (spawn_task / task_status / …).

    They degrade to an "unavailable" message unless a registry is bound (only
    the inline bridge binds one), so they're harmless to expose in every build.
    """
    try:
        from koda.subagent_tools import SUBAGENT_TASK_TOOLS

        return SUBAGENT_TASK_TOOLS
    except Exception:  # pragma: no cover - never break tool assembly
        return []


EXTRA_TOOLS = [
    think,
    multi_edit,
    web_fetch,
    web_search,
    visual_analyze,
    git,
    git_diff,
    bash_background,
    bash_output,
    kill_bash,
    ask_user,
    *_background_subagent_tools(),
]


if __name__ == "__main__":
    # Direct-script entrypoint: load .env so TAVILY_API_KEY etc. are visible.
    # override=True lets the .env value win over any stale value already
    # exported in the current shell (e.g. LANGSMITH_TRACING from a prior run).
    from dotenv import load_dotenv

    load_dotenv(override=True)
    query = "What is the price of the tea in China?"
    print(web_search.invoke(query))