import React, { useEffect, useReducer, useRef, useState } from "react";
import { Box, Static, Text, useApp, useInput } from "ink";
import { spawn } from "node:child_process";
import { Bridge, type BridgeOptions } from "./bridge.js";
import type { BridgeEvent, EnvState, Item, McpServer, McpTestResult, McpToolList, McpToolPolicy, PermsState, PermissionItem, ToolInfo } from "./types.js";
import { transcriptReducer, nextId, type TState } from "./transcript.js";
import { getTheme, THEMES, DEFAULT_THEME } from "./theme.js";
import { nextMode, styleFor, type Mode } from "./modes.js";
import { MessageView } from "./components/Message.js";
import { Input } from "./components/Input.js";
import { Permission } from "./components/Permission.js";
import { StatusBar } from "./components/StatusBar.js";
import { TaskBar } from "./components/TaskBar.js";
import { Dashboard } from "./components/Dashboard.js";
import { McpDashboard } from "./components/McpDashboard.js";
import { PermsDashboard } from "./components/PermsDashboard.js";
import { SetupDashboard } from "./components/SetupDashboard.js";
import { AskUser } from "./components/AskUser.js";
import { SessionPicker } from "./components/SessionPicker.js";
import { ModelPicker } from "./components/ModelPicker.js";
import { TreePopup } from "./components/TreePopup.js";
import { getModels, setModels } from "./completer.js";
import type { SessionInfo, TaskSummary, TreeNode } from "./types.js";

function Spinner({ color, muted, start }: { color: string; muted: string; start: number }) {
  const [f, setF] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setF((x) => x + 1), 150);
    return () => clearInterval(t);
  }, []);
  const frames = ["·   ", "··  ", "··· ", " ···", "  ··", "   ·"];
  const el = Math.floor((Date.now() - start) / 1000);
  const ts = `${Math.floor(el / 60)}:${String(el % 60).padStart(2, "0")}`;
  return (
    <Text color={color}>
      ★ <Text color={muted} italic>{`Thinking ${ts} ${frames[f % frames.length]}`}</Text>
    </Text>
  );
}

// Cross-platform clipboard: pbcopy on macOS; wl-copy/xclip/xsel on Linux.
// Never throws — spawn "error" events are handled so a missing binary can't
// crash the app (audit finding: bare pbcopy on Linux killed the whole UI).
function copyToClipboard(text: string, done: (ok: boolean, detail?: string) => void) {
  const candidates: Array<[string, string[]]> =
    process.platform === "darwin"
      ? [["pbcopy", []]]
      : process.env.WAYLAND_DISPLAY
        ? [["wl-copy", []], ["xclip", ["-selection", "clipboard"]], ["xsel", ["--clipboard", "--input"]]]
        : [["xclip", ["-selection", "clipboard"]], ["xsel", ["--clipboard", "--input"]], ["wl-copy", []]];

  const tryNext = (i: number) => {
    if (i >= candidates.length) {
      done(false, "no clipboard tool found");
      return;
    }
    const [cmd, args] = candidates[i];
    const p = spawn(cmd, args, { stdio: ["pipe", "ignore", "ignore"] });
    let failed = false;
    p.on("error", () => {
      failed = true;
      tryNext(i + 1);
    });
    p.on("close", (code) => {
      if (!failed) done(code === 0, code === 0 ? undefined : `${cmd} exited ${code}`);
    });
    p.stdin.on("error", () => {}); // EPIPE if the tool dies mid-write
    p.stdin.write(text);
    p.stdin.end();
  };
  tryNext(0);
}

export interface AppProps {
  bridgeOptions: BridgeOptions;
  initialModel: string;
  // Launch-time resume (koda -r / -c): open the picker or continue the latest.
  startupResume?: "pick" | "latest";
}

export function App({ bridgeOptions, initialModel, startupResume }: AppProps) {
  const { exit } = useApp();
  const [tstate, dispatch] = useReducer(transcriptReducer, { committed: [], live: [] } as TState);
  const stateRef = useRef(tstate);
  stateRef.current = tstate;

  const [themeName, setThemeName] = useState(DEFAULT_THEME);
  const palette = getTheme(themeName);
  const paletteRef = useRef(palette);
  paletteRef.current = palette;

  const [model, setModel] = useState(initialModel);
  const [backend, setBackend] = useState("");
  const [mode, setMode] = useState<Mode>("default");
  const [tools, setTools] = useState<ToolInfo[]>([]);
  const [usageIn, setUsageIn] = useState(0);
  const [usageOut, setUsageOut] = useState(0);
  const [streaming, setStreaming] = useState(false);
  const [thinkStart, setThinkStart] = useState(0);

  const [permItems, setPermItems] = useState<PermissionItem[] | null>(null);
  const [permIdx, setPermIdx] = useState(0);
  const permOutcomes = useRef<Array<"allow" | "always" | "deny">>([]);

  const [tasks, setTasks] = useState<Record<string, TaskSummary>>({});
  const [askReq, setAskReq] = useState<{ question: string; options: string[] } | null>(null);
  const [sessionList, setSessionList] = useState<SessionInfo[] | null>(null);
  const [dashboardOpen, setDashboardOpen] = useState(false);
  const [mcpOpen, setMcpOpen] = useState(false);
  const [mcpServers, setMcpServers] = useState<McpServer[]>([]);
  const [mcpTestResults, setMcpTestResults] = useState<Record<string, McpTestResult>>({});
  // Per-server tool inventory + permission policy, keyed by server name.
  const [mcpToolLists, setMcpToolLists] = useState<Record<string, McpToolList>>({});
  const [permDashOpen, setPermDashOpen] = useState(false);
  const [permsState, setPermsState] = useState<PermsState | null>(null);
  const [setupOpen, setSetupOpen] = useState(false);
  const [envState, setEnvState] = useState<EnvState | null>(null);
  // /model popup: opens when /model is run with no arg; carries the discovered
  // model list and the current selection so the picker can highlight it.
  const [modelOpen, setModelOpen] = useState(false);
  const [modelsList, setModelsList] = useState<string[]>([]);
  // /tree popup: opens when /tree is run with no arg; payload arrives on the
  // `tree` bridge event (see handleSlash / case "tree" below).
  const [treeOpen, setTreeOpen] = useState(false);
  const [treeData, setTreeData] = useState<{
    sessionId: string; messageCount: number; leaf: string | null; nodes: TreeNode[];
  } | null>(null);
  // Track when /tree fired without args so the next `tree` event knows to
  // open the popup rather than treat the data as an info refresh.
  const treePending = useRef(false);
  // /setup waits for its data, then opens on a short timer.
  //
  // Ink 5.2.1 loses a frame rendered within 32ms of the previous one: the
  // throttled writer skips the write, but `lastOutput` is still assigned the
  // skipped frame (ink/build/ink.js, onRender), so the next identical render
  // compares equal and is skipped too — that frame never lands. The bridge
  // answers `env_list` in a few ms, well inside the window, which left the
  // dashboard stuck on "Loading…" forever. Opening with the payload already in
  // state, one tick past the window, makes the first painted frame the
  // finished one.
  const setupPending = useRef(false);
  const setupTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [taskPerm, setTaskPerm] = useState<{ taskId: string; items: PermissionItem[] } | null>(null);
  const [taskPermIdx, setTaskPermIdx] = useState(0);
  const taskPermOutcomes = useRef<Array<"allow" | "always" | "deny">>([]);
  const rows = process.stdout.rows || 24;
  // Table-based overlays size their columns against this, so a stale value
  // shears the layout. Ink re-renders on stdout "resize", which re-reads it.
  const columns = process.stdout.columns || 80;

  const toolsRef = useRef<ToolInfo[]>([]);
  const modelRef = useRef(model);
  modelRef.current = model;
  const bridgeRef = useRef<Bridge | null>(null);
  const didResume = useRef(false);

  function commit(item: Item) {
    dispatch({ type: "commit", item });
  }
  function info(text: string) {
    commit({ kind: "info", id: nextId(), text });
  }
  function error(text: string) {
    commit({ kind: "error", id: nextId(), text });
  }

  // ── bridge lifecycle ──────────────────────────────────────────────
  useEffect(() => {
    const bridge = new Bridge(bridgeOptions);
    bridgeRef.current = bridge;

    bridge.onEvent((ev: BridgeEvent) => {
      switch (ev.type) {
        case "ready":
          setModel(ev.model);
          setBackend(ev.backend);
          setMode((ev.mode as Mode) ?? "default");
          setTools(ev.tools);
          toolsRef.current = ev.tools;
          // koda -r / -c: once the backend is ready, either open the session
          // picker (pick) or continue the most recent session (latest). Guarded
          // so a reconnect's second "ready" doesn't re-trigger it.
          if (startupResume && !didResume.current) {
            didResume.current = true;
            bridge.send(
              startupResume === "latest"
                ? { type: "resume", session_id: "__latest__" }
                : { type: "resume" },
            );
          }
          break;
        case "text_delta":
          dispatch({ type: "text_delta", content: ev.content });
          break;
        case "thinking_delta":
          // reasoning is streamed but kept quiet in the transcript (spinner covers it)
          break;
        case "tool_start":
          if (ev.hidden) break;
          dispatch({ type: "tool_start", tool_id: ev.tool_id, name: ev.name, arguments: ev.arguments });
          break;
        case "tool_result":
          dispatch({ type: "tool_result", tool_id: ev.tool_id, output: ev.output, is_error: ev.is_error });
          break;
        case "todos":
          dispatch({ type: "todos", todos: ev.todos });
          break;
        case "usage":
          if (ev.input_tokens) setUsageIn(ev.input_tokens);
          if (ev.output_tokens) setUsageOut(ev.output_tokens);
          break;
        case "done":
          if (ev.usage) {
            if (ev.usage.input_tokens) setUsageIn(ev.usage.input_tokens);
            if (ev.usage.output_tokens) setUsageOut(ev.usage.output_tokens);
          }
          break;
        case "permission_request":
          permOutcomes.current = [];
          setPermIdx(0);
          setPermItems(ev.items);
          break;
        case "turn_end":
          dispatch({ type: "turn_end" });
          setStreaming(false);
          break;
        case "model_changed":
          setModel(ev.model);
          info(`model → ${ev.model}`);
          break;
        case "mode_changed":
          setMode(ev.mode as Mode);
          break;
        case "cleared":
          dispatch({ type: "reset" });
          info("started a new session");
          setUsageIn(0);
          setUsageOut(0);
          break;
        case "info":
          info(ev.message);
          break;
        case "error":
          error(ev.message);
          setStreaming(false);
          break;
        case "task_update":
          setTasks((prev) => ({ ...prev, [ev.task.id]: ev.task }));
          break;
        case "task_done":
          setTasks((prev) => ({ ...prev, [ev.task.id]: ev.task }));
          info(
            `subagent ${ev.task.id} (${ev.task.subagent_type}) ${ev.task.state}` +
              (ev.task.state === "success" ? ` — open /dashboard for the result` : ""),
          );
          break;
        case "task_list":
          setTasks(() => Object.fromEntries(ev.tasks.map((t) => [t.id, t])));
          break;
        case "task_permission":
          taskPermOutcomes.current = [];
          setTaskPermIdx(0);
          setTaskPerm({ taskId: ev.task_id, items: ev.items });
          break;
        case "models":
          setModels(ev.models);
          setModelsList(ev.models);
          break;
        case "ask_user":
          setAskReq({ question: ev.question, options: ev.options });
          break;
        case "sessions":
          setSessionList(ev.sessions);
          break;
        case "resumed": {
          dispatch({ type: "reset" });
          for (const m of ev.messages) {
            commit(
              m.role === "user"
                ? { kind: "user", id: nextId(), text: m.content }
                : { kind: "assistant", id: nextId(), text: m.content },
            );
          }
          info(`resumed session ${ev.session_id.slice(0, 8)} — ${ev.messages.length} message(s) restored`);
          setSessionList(null);
          break;
        }
        case "mcp_servers":
          setMcpServers(ev.servers);
          break;
        case "mcp_status":
          setMcpTestResults((prev) => ({
            ...prev,
            [ev.name]: {
              name: ev.name,
              ok: ev.ok,
              tool_count: ev.tool_count,
              tools: ev.tools,
              error: ev.error,
              needs_auth: ev.needs_auth,
            },
          }));
          if (ev.ok) info(`mcp '${ev.name}' OK — ${ev.tool_count} tool(s)`);
          else error(`mcp '${ev.name}' failed: ${ev.error || "unknown"}`);
          break;
        case "mcp_tools":
          setMcpToolLists((prev) => ({
            ...prev,
            [ev.server]: {
              server: ev.server,
              loading: false,
              tools: ev.tools,
              permissions: ev.permissions,
              error: ev.error,
              cached: ev.cached,
            },
          }));
          break;
        case "env_state":
          setEnvState({ path: ev.path, model: ev.model, keys: ev.keys });
          if (setupPending.current) {
            setupPending.current = false;
            if (setupTimer.current) clearTimeout(setupTimer.current);
            setupTimer.current = setTimeout(() => setSetupOpen(true), 60);
          }
          break;
        case "perms_state":
          setPermsState({
            mode: ev.mode,
            auto_approve: ev.auto_approve,
            allowed_tools: ev.allowed_tools,
            mutating_tools: ev.mutating_tools,
            file_edit_tools: ev.file_edit_tools,
          });
          break;
        case "tree":
          // Structured tree payload for the /tree popup. The popup is opened
          // in the slash handler (it may still be loading for a tick until
          // this event lands — same pattern as /setup + env_state).
          setTreeData({
            sessionId: ev.session_id,
            messageCount: ev.message_count,
            leaf: ev.leaf,
            nodes: ev.nodes,
          });
          if (treePending.current) {
            treePending.current = false;
            setTreeOpen(true);
          }
          break;
      }
    });

    bridge.onExit((code) => {
      if (code && code !== 0) {
        error(`agent process exited (code ${code}). ${bridge.stderrTail.slice(-3).join(" | ")}`);
      }
    });

    return () => bridge.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── permission answering ──────────────────────────────────────────
  function decidePermission(outcome: "allow" | "always" | "deny") {
    if (!permItems) return;
    permOutcomes.current.push(outcome);
    const next = permIdx + 1;
    if (next < permItems.length) {
      setPermIdx(next);
    } else {
      bridgeRef.current?.send({ type: "decisions", outcomes: permOutcomes.current });
      setPermItems(null);
      setPermIdx(0);
    }
  }

  function decideTaskPermission(outcome: "allow" | "always" | "deny") {
    if (!taskPerm) return;
    taskPermOutcomes.current.push(outcome);
    const next = taskPermIdx + 1;
    if (next < taskPerm.items.length) {
      setTaskPermIdx(next);
    } else {
      bridgeRef.current?.send({ type: "task_answer", task_id: taskPerm.taskId, outcomes: taskPermOutcomes.current });
      setTaskPerm(null);
      setTaskPermIdx(0);
    }
  }

  /** Patch one server's cached tool→policy map (used for optimistic toggles). */
  function setMcpToolPerms(
    server: string,
    update: (list: McpToolList) => Record<string, McpToolPolicy>,
  ) {
    setMcpToolLists((prev) => {
      const list = prev[server];
      if (!list) return prev;
      return { ...prev, [server]: { ...list, permissions: update(list) } };
    });
  }

  function taskControl(action: "stop" | "resume" | "restart", taskId: string) {
    const bridge = bridgeRef.current;
    if (!bridge) return;
    if (action === "stop") bridge.send({ type: "task_stop", task_id: taskId });
    else if (action === "resume") bridge.send({ type: "task_resume", task_id: taskId });
    else bridge.send({ type: "task_restart", task_id: taskId });
    info(`${action} ${taskId}`);
  }

  // ── input handling ────────────────────────────────────────────────
  function startTurn(text: string) {
    commit({ kind: "user", id: nextId(), text });
    setStreaming(true);
    setThinkStart(Date.now());
    bridgeRef.current?.send({ type: "user", text });
  }

  function runShell(raw: string) {
    const local = raw.startsWith("!!");
    const cmd = raw.slice(local ? 2 : 1).trim();
    if (!cmd) return;
    const chunks: string[] = [];
    const child = spawn("bash", ["-lc", cmd], { cwd: process.cwd() });
    child.stdout.on("data", (d) => chunks.push(d.toString()));
    child.stderr.on("data", (d) => chunks.push(d.toString()));
    child.on("close", (code) => {
      commit({
        kind: "tool",
        id: nextId(),
        toolId: "shell-" + nextId(),
        name: "shell",
        args: { command: cmd },
        output: chunks.join("") || "(no output)",
        isError: code !== 0,
        running: false,
      });
    });
    child.on("error", (e) => error(`shell: ${e.message}`));
  }

  function cycleMode() {
    const m = nextMode(mode);
    setMode(m);
    bridgeRef.current?.send({ type: "set_mode", mode: m });
    info(`mode → ${styleFor(m).label.toLowerCase()}`);
  }

  function interrupt() {
    if (streaming) {
      bridgeRef.current?.send({ type: "interrupt" });
      info("interrupting…");
    }
  }

  function quit() {
    bridgeRef.current?.close();
    exit();
  }

  // Global keyboard fallbacks — fire regardless of which overlay is open.
  // Ink's exitOnCtrlC:false means the OS won't kill us; we must handle it.
  useInput((input, key) => {
    // Esc closes any open popup (the popup's own useInput also handles this,
    // but doing it here too makes the dismissal feel instant — the user
    // doesn't see a frame where the Esc got "swallowed" by an inner handler).
    if (key.escape) {
      if (dashboardOpen) { setDashboardOpen(false); return; }
      if (mcpOpen) { setMcpOpen(false); return; }
      if (permDashOpen) { setPermDashOpen(false); return; }
      if (setupOpen) { setSetupOpen(false); return; }
      if (modelOpen) { setModelOpen(false); return; }
      if (treeOpen) { setTreeOpen(false); return; }
      if (sessionList) { setSessionList(null); return; }
    }
    if (key.ctrl && input === "c") {
      // Close any open overlay first.
      if (dashboardOpen) { setDashboardOpen(false); return; }
      if (mcpOpen) { setMcpOpen(false); return; }
      if (permDashOpen) { setPermDashOpen(false); return; }
      if (setupOpen) { setSetupOpen(false); return; }
      if (modelOpen) { setModelOpen(false); return; }
      if (treeOpen) { setTreeOpen(false); return; }
      if (sessionList) { setSessionList(null); return; }
      // If nothing is open, let the Input component handle it
      // (interrupt on empty, clear on non-empty).
    }
  });

  function lastAssistantText(): string {
    const all = [...stateRef.current.committed, ...stateRef.current.live];
    for (let i = all.length - 1; i >= 0; i--) {
      if (all[i].kind === "assistant") return (all[i] as any).text as string;
    }
    return "";
  }

  function handleSlash(text: string) {
    const [name, ...rest] = text.slice(1).split(" ");
    const args = rest.join(" ").trim();
    const bridge = bridgeRef.current;
    switch (name.toLowerCase()) {
      case "help": {
        const lines = [
          "Slash commands:",
          "  /clear          start a new chat session",
          "  /model [m]      switch model or show current",
          "  /theme [name]   switch color theme (or list)",
          "  /plan /edits /default   switch agent mode",
          "  /compact        summarize older messages",
          "  /copy           copy last assistant reply",
          "  /usage          show token usage",
          "  /agents /tools  describe the agent / list tools",
          "  /tasks          list background subagents",
          "  /dashboard      full-screen subagent manager",
          "  /mcp            MCP servers — add, browser sign-in, per-tool permissions",
          "  /perms          permission dashboard (mode, allowed tools, grants)",
          "  /tree [id]      show the session tree · jump to a node to branch",
          "  /resume [id]    pick a past session and continue it",
          "  /skill [new …]  list skills, or author one with the current model",
          "  /setup          API-key dashboard (add/clear keys, writes .env)",
          "  /help /quit /exit",
          "",
          "Input: @path attaches a file · !cmd runs shell · Shift+Tab cycles mode",
        ];
        info(lines.join("\n"));
        break;
      }
      case "theme":
        if (!args) {
          info("Themes: " + Object.keys(THEMES).sort().join(", "));
        } else if (THEMES[args]) {
          setThemeName(args);
          info(`theme → ${args}`);
        } else {
          error(`unknown theme: ${args}`);
        }
        break;
      case "clear":
        bridge?.send({ type: "clear" });
        break;
      case "quit":
      case "exit":
        quit();
        break;
      case "copy": {
        const t = lastAssistantText();
        if (!t) {
          info("nothing to copy yet");
          break;
        }
        copyToClipboard(t, (ok, detail) =>
          ok ? info(`copied ${t.length} chars`) : error(`clipboard unavailable${detail ? ` (${detail})` : ""}`),
        );
        break;
      }
      case "usage":
        info(`Session usage — input: ${usageIn.toLocaleString()}  output: ${usageOut.toLocaleString()}`);
        break;
      case "model":
        if (!args) {
          // No arg → open the model picker. If the bridge hasn't sent the
          // discovered list yet (the very first /model before `ready`), seed
          // the picker with the current model so it's never empty.
          const m = getModels();
          setModelsList(m);
          setModelOpen(true);
        } else {
          bridge?.send({ type: "switch_model", model: args });
        }
        break;
      case "plan":
      case "edits":
      case "default": {
        const m = (name.toLowerCase() as Mode);
        setMode(m);
        bridge?.send({ type: "set_mode", mode: m });
        info(`mode → ${styleFor(m).label.toLowerCase()}`);
        break;
      }
      case "compact":
        bridge?.send({ type: "compact" });
        break;
      case "agents":
        info(
          `Agent — backend: ${backend}  model: ${modelRef.current}  tools: ${toolsRef.current.length}`,
        );
        break;
      case "tools": {
        const names = toolsRef.current.map((t) => t.name);
        info(`Tools (${names.length}):\n  ` + names.join(", "));
        break;
      }
      case "tree":
        if (!args) {
          // No arg → pop up the session tree. The bridge will reply with a
          // `tree` event containing the structured payload; treePending tells
          // that handler to flip the popup open once it lands.
          treePending.current = true;
          bridge?.send({ type: "tree" });
        } else {
          // With a node id → jump to it; the next message branches from there.
          bridge?.send({ type: "tree", node: args });
        }
        break;
      case "resume":
        // No arg → picker of this project's past sessions; with an id →
        // resume it directly (appends to the same session file).
        bridge?.send({ type: "resume", session_id: args || undefined });
        break;
      case "skill":
      case "skills": {
        // No arg → list skills. `/skill new <name>: <what>` (or just
        // `/skill <brief>`) → author one with the configured model & save it.
        const brief = args.replace(/^new\s+/i, "").trim();
        if (!brief) bridge?.send({ type: "skill", action: "list" });
        else bridge?.send({ type: "skill", action: "create", brief });
        break;
      }
      case "tasks": {
        const all = Object.values(tasks);
        if (all.length === 0) {
          info("No background subagents. The agent starts them with start_async_task — ask it to run something in the background.");
        } else {
          info(
            "Subagents:\n" +
              all
                .map((t) => `  ${t.id} [${t.state}] ${t.subagent_type} — ${t.current} · ${t.tool_count} tools\n    ${t.description.slice(0, 70)}`)
                .join("\n") +
              "\n(/dashboard to manage)",
          );
        }
        break;
      }
      case "dashboard":
        setDashboardOpen(true);
        break;
      case "mcp":
        bridge?.send({ type: "mcp_list" });
        setMcpOpen(true);
        break;
      case "perms":
      case "permissions":
        bridge?.send({ type: "perms_list" });
        setPermDashOpen(true);
        break;
      case "setup":
      case "keys":
        // Opens from the env_state handler, once the keys are in hand.
        setupPending.current = true;
        bridge?.send({ type: "env_list" });
        break;
      default:
        error(`unknown command: /${name}`);
    }
  }

  function onSubmit(text: string) {
    if (text.startsWith("!")) {
      runShell(text);
      return;
    }
    if (text.startsWith("/")) {
      handleSlash(text);
      return;
    }
    startTurn(text);
  }

  // ── render ────────────────────────────────────────────────────────
  const taskList = Object.values(tasks).sort((a, b) => a.id.localeCompare(b.id, undefined, { numeric: true }));

  // Full-screen-on-demand overlays (/dashboard, /mcp, /setup, /perms). These
  // replace the *live* section only — never the whole tree.
  //
  // Returning a different root while an overlay is open used to unmount
  // <Static>, and that is a heap bomb: Ink caches `rootNode.staticNode` when
  // the <Static> mounts (reconciler.js finalizeInitialChildren) but never
  // clears it on unmount, while the reconciler frees that node's Yoga memory
  // (cleanupYogaNode → freeRecursive). The next render still reads
  // `staticNode.yogaNode.getComputedWidth()/getComputedHeight()`
  // (renderer.js) off the freed pointer, gets garbage dimensions, and
  // Output.get() then allocates width × height cell objects — multiple GB
  // within a couple of seconds, killing node with "Ineffective mark-compacts
  // near heap limit". Keeping <Static> mounted for the whole session keeps
  // that Yoga node alive, and as a bonus stops the transcript being
  // re-printed every time an overlay closes.
  const overlay = dashboardOpen ? (
    <Dashboard
      tasks={taskList}
      palette={palette}
      rows={rows}
      onClose={() => setDashboardOpen(false)}
      onControl={taskControl}
    />
  ) : modelOpen ? (
    <ModelPicker
      models={modelsList}
      current={model}
      palette={palette}
      rows={rows}
      onPick={(m) => {
        bridgeRef.current?.send({ type: "switch_model", model: m });
        setModelOpen(false);
      }}
      onClose={() => setModelOpen(false)}
    />
  ) : treeOpen ? (
    <TreePopup
      sessionId={treeData?.sessionId ?? ""}
      messageCount={treeData?.messageCount ?? 0}
      leaf={treeData?.leaf ?? null}
      nodes={treeData?.nodes ?? []}
      palette={palette}
      rows={rows}
      onJump={(id) => {
        bridgeRef.current?.send({ type: "tree", node: id });
        setTreeOpen(false);
      }}
      onClose={() => setTreeOpen(false)}
    />
  ) : mcpOpen ? (
    <McpDashboard
      servers={mcpServers}
      testResults={mcpTestResults}
      toolLists={mcpToolLists}
      palette={palette}
      rows={rows}
      columns={columns}
      onClose={() => setMcpOpen(false)}
      onRefresh={() => bridgeRef.current?.send({ type: "mcp_list" })}
      onAdd={(spec) => bridgeRef.current?.send({ type: "mcp_add", ...spec })}
      onRemove={(name) => bridgeRef.current?.send({ type: "mcp_remove", name })}
      onTest={(name) => bridgeRef.current?.send({ type: "mcp_test", name })}
      onLoadTools={(name, refresh) => {
        setMcpToolLists((prev) => ({
          ...prev,
          [name]: {
            server: name,
            tools: prev[name]?.tools ?? [],
            permissions: prev[name]?.permissions ?? {},
            loading: true,
          },
        }));
        bridgeRef.current?.send({ type: "mcp_tools", name, refresh });
      }}
      // Both toggles paint optimistically: the bridge queues the write
      // behind any running turn and then rebuilds the graph, which can take
      // seconds — long enough for the row to look broken. The bridge's
      // `mcp_tools` reply overwrites this with the persisted truth.
      onSetToolPerm={(server, tool, policy) => {
        setMcpToolPerms(server, (list) => ({ ...list.permissions, [tool]: policy }));
        bridgeRef.current?.send({ type: "mcp_set_tool_perm", server, tool, policy });
      }}
      onSetAllToolPerms={(server, policy) => {
        setMcpToolPerms(server, (list) =>
          Object.fromEntries(list.tools.map((t) => [t.name, policy])),
        );
        bridgeRef.current?.send({ type: "mcp_set_tool_perm", server, tool: "*", policy });
      }}
      onAuthorize={(name) => bridgeRef.current?.send({ type: "mcp_authorize", name })}
      onSignOut={(name) => bridgeRef.current?.send({ type: "mcp_sign_out", name })}
    />
  ) : setupOpen ? (
    <SetupDashboard
      state={envState}
      palette={palette}
      rows={rows}
      onClose={() => setSetupOpen(false)}
      onRefresh={() => bridgeRef.current?.send({ type: "env_list" })}
      onSet={(key, value) => bridgeRef.current?.send({ type: "env_set", key, value })}
    />
  ) : permDashOpen ? (
    <PermsDashboard
      state={permsState}
      palette={palette}
      rows={rows}
      onClose={() => setPermDashOpen(false)}
      onRefresh={() => bridgeRef.current?.send({ type: "perms_list" })}
      onClearAllowed={() => bridgeRef.current?.send({ type: "perms_clear_allowed" })}
      onSetMode={(m) => bridgeRef.current?.send({ type: "set_mode", mode: m })}
    />
  ) : null;

  const permActive = !!permItems;
  const taskPermActive = !permActive && !!taskPerm;
  const askActive = !permActive && !taskPermActive && !!askReq;
  const pickerActive = !permActive && !taskPermActive && !askActive && !!sessionList;
  const inputActive = !overlay && !permActive && !taskPermActive && !askActive && !pickerActive;

  return (
    <Box flexDirection="column">
      <Static items={tstate.committed}>
        {(item, i) => (
          <MessageView
            key={item.id}
            item={item}
            prevKind={tstate.committed[i - 1]?.kind}
            palette={palette}
          />
        )}
      </Static>

      {overlay}

      {overlay ? null : (
      <Box flexDirection="column">
        {tstate.live.map((item, i) => (
          <MessageView
            key={item.id}
            item={item}
            prevKind={
              i > 0 ? tstate.live[i - 1].kind : tstate.committed[tstate.committed.length - 1]?.kind
            }
            palette={palette}
          />
        ))}

        {streaming && !permActive && !taskPermActive ? (
          <Box marginTop={1}>
            <Spinner color={palette.accent} muted={palette.muted} start={thinkStart} />
          </Box>
        ) : null}

        {permActive && permItems ? (
          <Permission
            key={`perm-${permIdx}`}
            item={permItems[permIdx]}
            index={permIdx}
            total={permItems.length}
            palette={palette}
            onDecide={decidePermission}
          />
        ) : null}

        {taskPermActive && taskPerm ? (
          <Box flexDirection="column" marginTop={1}>
            <Text color={palette.accent}>Background task {taskPerm.taskId} needs approval:</Text>
            <Permission
              key={`taskperm-${taskPermIdx}`}
              item={taskPerm.items[taskPermIdx]}
              index={taskPermIdx}
              total={taskPerm.items.length}
              palette={palette}
              onDecide={decideTaskPermission}
            />
          </Box>
        ) : null}

        {askActive && askReq ? (
          <AskUser
            question={askReq.question}
            options={askReq.options}
            palette={palette}
            onAnswer={(value) => {
              bridgeRef.current?.send({ type: "ask_answer", value });
              commit({ kind: "info", id: nextId(), text: `answered: ${value}` });
              setAskReq(null);
            }}
          />
        ) : null}

        {pickerActive && sessionList ? (
          <SessionPicker
            sessions={sessionList}
            palette={palette}
            onPick={(id) => bridgeRef.current?.send({ type: "resume", session_id: id })}
            onClose={() => setSessionList(null)}
          />
        ) : null}

        <Box marginTop={1} flexDirection="column">
          <Input
            active={inputActive}
            palette={palette}
            mode={mode}
            streaming={streaming}
            onSubmit={onSubmit}
            onInterrupt={interrupt}
            onCycleMode={cycleMode}
            onExit={quit}
          />
          <TaskBar tasks={taskList} palette={palette} />
          <StatusBar model={model} mode={mode} inputTokens={usageIn} outputTokens={usageOut} palette={palette} />
        </Box>
      </Box>
      )}
    </Box>
  );
}
