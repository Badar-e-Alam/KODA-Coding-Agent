"""Project-configured shell hooks for the coding agent."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import subprocess
from dataclasses import dataclass, replace
from fnmatch import fnmatchcase
from pathlib import Path

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from coding_agent.backend import ReapingShellBackend

_log = logging.getLogger(__name__)
OUTPUT_LIMIT = 4096
DEFAULT_TIMEOUT = 30


@dataclass(frozen=True)
class Hook:
    matcher: str
    command: str
    timeout: float


def _parse_hooks(entries: object, event: str) -> tuple[Hook, ...]:
    if not isinstance(entries, list):
        _log.warning("hooks.%s must be a list", event)
        return ()
    hooks = []
    for index, entry in enumerate(entries):
        try:
            matcher, command = entry["matcher"], entry["command"]
            timeout = entry.get("timeout", DEFAULT_TIMEOUT)
            if not isinstance(matcher, str) or not isinstance(command, str) or not command.strip():
                raise ValueError
            if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
                raise ValueError
        except (KeyError, TypeError, ValueError, OverflowError):
            _log.warning("Invalid hook at hooks.%s[%d]", event, index)
            continue
        hooks.append(Hook(matcher, command, timeout))
    return tuple(hooks)


def load_hooks(root: Path, backend: ReapingShellBackend) -> HookMiddleware | None:
    try:
        settings = json.loads((root / ".koda" / "settings.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError, RecursionError) as exc:
        _log.warning("Could not read .koda/settings.json (%s)", type(exc).__name__)
        return None
    hooks = settings.get("hooks", {}) if isinstance(settings, dict) else None
    if not isinstance(hooks, dict):
        _log.warning("Expected a hooks object in .koda/settings.json")
        return None
    configured = {event: _parse_hooks(hooks.get(event, []), event)
                  for event in ("pre_tool", "post_tool", "stop")}
    return HookMiddleware(backend, configured) if any(configured.values()) else None


def _cap(output: str) -> str:
    marker = "\n[Hook output truncated]"
    if len(output) > OUTPUT_LIMIT:
        return output[:OUTPUT_LIMIT - len(marker)] + marker
    return output


def _append_output(response, output: str):
    if not output:
        return response
    if isinstance(response, ToolMessage):
        content = response.content
        if isinstance(content, str):
            content = f"{content}\n\n{output}"
        else:
            content = [*content, {"type": "text", "text": output}]
        return response.model_copy(update={"content": content})
    if isinstance(response, Command) and isinstance(response.update, dict) and "messages" in response.update:
        messages = response.update["messages"]
        updated = (
            [_append_output(message, output) for message in messages]
            if isinstance(messages, (list, tuple)) else _append_output(messages, output)
        )
        return replace(response, update={
            **response.update, "messages": updated,
        })
    return response


class HookMiddleware(AgentMiddleware):
    def __init__(self, backend: ReapingShellBackend, hooks: dict[str, tuple[Hook, ...]]):
        self._backend = backend
        self._hooks = hooks

    async def _run(self, event: str, name: str, **fields) -> tuple[str | None, str]:
        output = []
        for hook in self._hooks[event]:
            if not fnmatchcase(name, hook.matcher):
                continue
            try:
                payload = json.dumps({"event": event, **fields})
                code, stdout, stderr = await asyncio.to_thread(
                    self._backend.run_hook, hook.command, payload, hook.timeout
                )
            except subprocess.TimeoutExpired:
                output.append(f"{event} hook timed out after {hook.timeout:g} seconds.")
                continue
            except (OSError, TypeError, ValueError, OverflowError, RecursionError) as exc:
                output.append(f"{event} hook could not run ({type(exc).__name__}).")
                continue
            if event == "pre_tool" and code == 2:
                return stderr.strip() or "The hook exited with code 2.", "\n".join(output)
            if code:
                output.append(f"{event} hook failed with exit code {code}.")
            if event != "pre_tool" or code:
                output.extend(part.strip() for part in (stdout, stderr) if part.strip())
        return None, "\n".join(output)

    async def awrap_tool_call(self, request, handler):
        call = request.tool_call
        fields = {"tool_name": call["name"], "tool_input": call["args"]}
        reason, pre_output = await self._run("pre_tool", call["name"], **fields)
        if reason is not None:
            content = _cap(f"Blocked by pre_tool hook: {reason}\n{pre_output}".rstrip())
            return ToolMessage(content=content, tool_call_id=call["id"],
                               name=call["name"], status="error")
        response = await handler(request)
        _, post_output = await self._run("post_tool", call["name"], **fields)
        return _append_output(response, _cap("\n".join(filter(None, (pre_output, post_output)))))

    async def stop(self) -> str:
        _, output = await self._run("stop", "stop")
        return _cap(output)
