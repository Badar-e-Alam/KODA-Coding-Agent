"""Optional OS-level isolation for shell commands the agent runs.

The file tools (``read_file``/``write_file``/``edit_file``/…) are confined to
the project by the backend's ``virtual_mode``. Shell commands are not: by
default an approved ``execute`` (or ``bash_background``) call runs with the
user's full access to the machine and the network.

``KODA_SANDBOX=project`` opts into wrapping those commands:

- **Linux** → ``bwrap`` (bubblewrap): the whole filesystem is bind-mounted
  read-only, the project directory and temp dirs are bound writable, and the
  network namespace is unshared.
- **macOS** → ``sandbox-exec`` with an SBPL profile that allows everything
  except file writes outside the project/temp and (by default) networking.

``KODA_SANDBOX_NETWORK=1`` keeps the network available inside the sandbox
(package installs, ``git push``…). Reads are not restricted in either mode.

When ``project`` is requested but the sandbox tool is missing (or present but
unusable, e.g. bwrap without unprivileged user namespaces), commands run
**unsandboxed** and a loud warning is emitted once per process.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass

_log = logging.getLogger(__name__)

MODE_OFF = "off"
MODE_PROJECT = "project"

_warn_lock = threading.Lock()
_warned = False
_probe_cache: dict[str, bool] = {}


def sandbox_mode() -> str:
    """``off`` (default) or ``project``, from ``KODA_SANDBOX``."""
    raw = os.environ.get("KODA_SANDBOX", "").strip().lower()
    if raw == MODE_PROJECT:
        return MODE_PROJECT
    if raw not in ("", MODE_OFF):
        _log.warning("unknown KODA_SANDBOX=%r; expected off|project — treating as off", raw)
    return MODE_OFF


def network_allowed() -> bool:
    """True when ``KODA_SANDBOX_NETWORK=1`` keeps the network on in the sandbox."""
    return os.environ.get("KODA_SANDBOX_NETWORK", "").strip() == "1"


def _temp_dirs() -> list[str]:
    """Writable temp locations, resolved through symlinks, deduplicated."""
    candidates = ["/tmp", "/var/tmp", tempfile.gettempdir(), os.environ.get("TMPDIR", "")]
    if sys.platform == "darwin":
        candidates += ["/private/tmp", "/private/var/folders"]
    out: list[str] = []
    for c in candidates:
        if not c or not os.path.isdir(c):
            continue
        real = os.path.realpath(c)
        if real not in out:
            out.append(real)
    return out


def bwrap_argv(command: str, root: str, cwd: str, *, network: bool) -> list[str]:
    """argv that runs ``command`` via ``/bin/sh -c`` under bubblewrap.

    The command is passed as a single argv element, never concatenated, so it
    is not re-parsed by anything but the inner shell.
    """
    root = os.path.realpath(root)
    argv = [
        "bwrap",
        "--ro-bind", "/", "/",
        "--dev", "/dev",
        "--bind", root, root,
    ]
    for d in _temp_dirs():
        if d != root:
            argv += ["--bind", d, d]
    if not network:
        argv.append("--unshare-net")
    argv += [
        "--die-with-parent",
        "--chdir", os.path.realpath(cwd),
        "/bin/sh", "-c", command,
    ]
    return argv


# SBPL profile. Paths arrive as ``-D`` parameters so they never need quoting.
_SBPL_BASE = """(version 1)
(allow default)
(deny file-write*)
(allow file-write*
  (subpath (param "PROJECT_ROOT"))
{temp_rules}
  (literal "/dev/null")
  (literal "/dev/zero")
  (literal "/dev/tty")
  (regex #"^/dev/ttys[0-9]+$")
  (regex #"^/dev/fd/[0-9]+$"))
"""


def sandbox_exec_argv(command: str, root: str, *, network: bool) -> list[str]:
    """argv that runs ``command`` via ``/bin/sh -c`` under macOS sandbox-exec."""
    temps = _temp_dirs()
    temp_rules = "\n".join(f'  (subpath (param "TMP{i}"))' for i in range(len(temps)))
    profile = _SBPL_BASE.format(temp_rules=temp_rules)
    if not network:
        profile += "(deny network*)\n"
    argv = ["sandbox-exec", "-p", profile, "-D", f"PROJECT_ROOT={os.path.realpath(root)}"]
    for i, d in enumerate(temps):
        argv += ["-D", f"TMP{i}={d}"]
    argv += ["/bin/sh", "-c", command]
    return argv


def _tool_name() -> str | None:
    if sys.platform.startswith("linux"):
        return "bwrap"
    if sys.platform == "darwin":
        return "sandbox-exec"
    return None


def sandbox_available() -> bool:
    """True when the platform's sandbox tool exists *and* actually works.

    bwrap can be installed yet fail at runtime when unprivileged user
    namespaces are disabled (common in containers/CI), so run a trivial probe
    once and cache the answer.
    """
    tool = _tool_name()
    if tool is None or shutil.which(tool) is None:
        return False
    if tool in _probe_cache:
        return _probe_cache[tool]
    cwd = os.getcwd()
    if tool == "bwrap":
        probe = bwrap_argv("true", cwd, cwd, network=False)
    else:
        probe = sandbox_exec_argv("true", cwd, network=False)
    try:
        ok = subprocess.run(probe, capture_output=True, timeout=10).returncode == 0  # noqa: S603
    except Exception:  # noqa: BLE001
        ok = False
    _probe_cache[tool] = ok
    return ok


def _fallback_warning() -> str | None:
    """The fallback warning text the first time it is needed, else ``None``."""
    global _warned
    with _warn_lock:
        if _warned:
            return None
        _warned = True
    tool = _tool_name() or "a sandbox tool"
    msg = (
        f"WARNING: KODA_SANDBOX=project was requested but {tool} is missing or "
        f"unusable on this system. Shell commands are NOT isolated: they run with "
        f"your full access to the filesystem and network."
    )
    _log.warning(msg)
    print(f"[koda] {msg}", file=sys.stderr, flush=True)
    return msg


@dataclass
class WrappedCommand:
    """How to launch a shell command: ``Popen(args, shell=shell)``."""

    args: str | list[str]
    shell: bool
    sandboxed: bool
    warning: str | None = None


def wrap_command(command: str, *, root: str, cwd: str | None = None) -> WrappedCommand:
    """Wrap ``command`` for the configured sandbox mode.

    ``off`` returns the command unchanged (``shell=True``), so behaviour is
    exactly what it was before sandboxing existed.
    """
    if sandbox_mode() != MODE_PROJECT:
        return WrappedCommand(command, shell=True, sandboxed=False)
    if not sandbox_available():
        return WrappedCommand(command, shell=True, sandboxed=False, warning=_fallback_warning())
    net = network_allowed()
    if _tool_name() == "bwrap":
        argv = bwrap_argv(command, root, cwd or root, network=net)
    else:
        argv = sandbox_exec_argv(command, root, network=net)
    return WrappedCommand(argv, shell=False, sandboxed=True)


def _reset_for_tests() -> None:
    global _warned
    _warned = False
    _probe_cache.clear()
