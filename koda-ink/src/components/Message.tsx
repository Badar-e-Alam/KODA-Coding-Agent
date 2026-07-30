import React from "react";
import { Box, Text } from "ink";
import type { Item, Todo } from "../types.js";
import type { Palette } from "../theme.js";
import { Markdown } from "../markdown.js";

const TODO_GLYPH: Record<Todo["status"], string> = {
  pending: "○",
  in_progress: "◐",
  completed: "✓",
};

// ── tool call rendering ──────────────────────────────────────────────
// One tool call = one header line ("what am I doing, to what") plus one
// indented result line ("what came of it"). We deliberately do NOT dump the
// whole kwargs map: a header like
//   grep(-n="true", output_mode="content", path="/koda-ink/src", pattern="mcp…
// costs a full terminal row and says less than `Grep(mcp_list)`. Each tool
// declares its display verb and its ONE meaningful argument; everything else
// is noise the transcript doesn't need.
const TOOL_VERBS: Record<string, string> = {
  read_file: "Read",
  write_file: "Write",
  edit_file: "Update",
  multi_edit: "Update",
  execute: "Bash",
  bash_background: "Bash",
  bash_output: "BashOutput",
  kill_bash: "KillBash",
  grep: "Grep",
  glob: "Glob",
  ls: "List",
  git: "Git",
  git_diff: "Diff",
  run_tests: "Tests",
  run_lint: "Lint",
  run_type_check: "Typecheck",
  web_search: "WebSearch",
  web_fetch: "Fetch",
  visual_analyze: "Analyze",
  ask_user: "Ask",
  think: "Think",
  write_todos: "Todos",
};

// Which argument names carry the "what" of a call, most specific first.
const TARGET_KEYS = [
  "file_path", "command", "pattern", "url", "query", "subcommand", "extra_args",
  "thought", "question", "path", "checker", "framework", "linter", "bash_id",
];

function toolVerb(name: string): string {
  return TOOL_VERBS[name] ?? name;
}

// Trim absolute/virtual paths down to project-relative: the leading
// /Users/…/project (or the tools' virtual "/" root) is the same on
// every line, so it only steals width from the part that differs.
function prettyPath(p: string): string {
  const cwd = process.cwd();
  if (p.startsWith(cwd)) p = p.slice(cwd.length);
  return p.replace(/^\/+/, "") || ".";
}

function toolTarget(item: Extract<Item, { kind: "tool" }>): string {
  const args = item.args ?? {};
  // `git` splits one command across two args — show it the way you'd type it.
  if (item.name === "git" && typeof args.subcommand === "string") {
    return clip(`${args.subcommand} ${str(args.extra_args)}`.trim(), 68);
  }
  for (const k of TARGET_KEYS) {
    const v = args[k];
    if (typeof v !== "string" || !v.trim()) continue;
    const value = k === "file_path" || k === "path" ? prettyPath(v) : v;
    return clip(value, 68);
  }
  return "";
}

function lineCount(s: unknown): number {
  const t = typeof s === "string" ? s : "";
  if (!t) return 0;
  return t.replace(/\n$/, "").split("\n").length;
}

function plural(n: number, one: string, many = one + "s"): string {
  return `${n} ${n === 1 ? one : many}`;
}

// The result line. Per-tool metrics (lines added/removed, matches found, lines
// read) beat a raw output preview — they're what the reader is checking for,
// and they fit in a few words instead of a wrapped blob.
function toolDetail(item: Extract<Item, { kind: "tool" }>): string {
  const out = item.output ?? "";
  const args = item.args ?? {};
  switch (item.name) {
    case "edit_file": {
      const added = lineCount(args.new_string);
      const removed = lineCount(args.old_string);
      if (added || removed) return `Added ${plural(added, "line")}, removed ${plural(removed, "line")}`;
      break;
    }
    case "multi_edit": {
      const edits = Array.isArray(args.edits) ? args.edits.length : 0;
      if (edits) return `Applied ${plural(edits, "edit")}`;
      break;
    }
    case "write_file": {
      const n = lineCount(args.content);
      if (n) return `Wrote ${plural(n, "line")}`;
      break;
    }
    case "read_file":
      return `Read ${plural(lineCount(out), "line")}`;
    case "grep": {
      if (/^\(no (matches|content matches)/.test(out)) return "No matches";
      const mode = String(args.output_mode ?? "");
      const n = lineCount(out);
      return mode === "files_with_matches" ? `${plural(n, "file")} matched` : `${plural(n, "match", "matches")}`;
    }
    case "glob":
      return /^\(no files match/.test(out) ? "No files found" : `${plural(lineCount(out), "file")}`;
    case "ls":
      return /^\(empty directory/.test(out) ? "Empty" : `${plural(lineCount(out), "entry", "entries")}`;
  }
  return preview(out);
}

function preview(text: string): string {
  const clean = (text ?? "").trim();
  if (!clean) return "(empty)";
  const lines = clean.split("\n");
  let p = lines[0];
  if (p.length > 80) p = p.slice(0, 79) + "…";
  if (lines.length > 1) p += `  (+${lines.length - 1} lines)`;
  return p;
}

// The two subagent-spawning tools. Both take {description, subagent_type} and
// dispatch a specialist agent — the only difference is whether they block the
// main agent (inline `task`) or run detached (`start_async_task`, which also
// shows up in /dashboard). We render both with ONE card style so they read as
// the same kind of thing in the transcript.
const SUBAGENT_TOOLS: Record<string, "inline" | "background"> = {
  task: "inline",
  start_async_task: "background",
};

function str(v: unknown): string {
  return typeof v === "string" ? v : v == null ? "" : String(v);
}

function clip(s: string, n: number): string {
  s = s.replace(/\s+/g, " ").trim();
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

// Shared subagent block for both the blocking `task` tool and the async
// `start_async_task` tool. Same glyph, label, and layout; a trailing badge is
// the only tell for inline vs background.
function SubagentCard({
  item,
  kind,
  palette,
}: {
  item: Extract<Item, { kind: "tool" }>;
  kind: "inline" | "background";
  palette: Palette;
}) {
  const type = str(item.args?.subagent_type) || "general-purpose";
  const desc = clip(str(item.args?.description), 74);
  const headColor = item.isError ? palette.toolErr : item.running ? palette.tool : palette.toolOk;
  const glyph = item.running ? "◐" : item.isError ? "✗" : "◇";
  const taskId = /task_id:\s*(\S+)/.exec(item.output ?? "")?.[1];

  // Second line: what happened. Background never blocks, so it reports that the
  // main agent stays free; inline shows the subagent's returned report.
  let status: React.ReactNode;
  if (item.isError) {
    status = <Text color={palette.toolErr}> ↳ {preview(item.output ?? "")}</Text>;
  } else if (kind === "background") {
    status = (
      <Text color={palette.muted}>
        {" ↳ running in background — main agent is free"}
        {taskId ? <Text color={palette.accent}> · {taskId}</Text> : null}
        <Text color={palette.muted}> · /dashboard</Text>
      </Text>
    );
  } else if (item.running) {
    status = <Text color={palette.muted}> ↳ working…</Text>;
  } else {
    status = <Text color={palette.muted}> ↳ {preview(item.output ?? "")}</Text>;
  }

  return (
    <Box flexDirection="column" marginTop={1}>
      <Text color={headColor}>
        {glyph} subagent <Text color={palette.muted}>·</Text> {type}
        <Text color={kind === "background" ? palette.accent : palette.muted}>
          {kind === "background" ? "  [background → /dashboard]" : "  [inline]"}
        </Text>
      </Text>
      {desc ? <Text color={palette.assistant}> ↳ {desc}</Text> : null}
      {status}
    </Box>
  );
}

export function MessageView({
  item,
  palette,
  prevKind,
}: {
  item: Item;
  palette: Palette;
  // Kind of the item directly above, so consecutive tool calls can pack
  // together instead of each opening with a blank line.
  prevKind?: Item["kind"];
}) {
  switch (item.kind) {
    case "user":
      return (
        <Box marginTop={1}>
          <Text color={palette.user} bold>
            {"› "}
          </Text>
          <Text color={palette.assistant}>{item.text}</Text>
        </Box>
      );

    case "assistant":
      return (
        <Box marginTop={1} flexDirection="column">
          <Markdown text={item.text} palette={palette} />
        </Box>
      );

    case "tool": {
      // Blocking `task` and async `start_async_task` share one subagent card so
      // they read identically in the transcript (async also lives in /dashboard).
      const subKind = SUBAGENT_TOOLS[item.name];
      if (subKind) return <SubagentCard item={item} kind={subKind} palette={palette} />;

      const headColor = item.isError ? palette.toolErr : item.running ? palette.tool : palette.toolOk;
      const glyph = item.running ? "◐" : item.isError ? "✗" : "●";
      const target = toolTarget(item);
      return (
        // A run of tool calls is one block: no blank line between them, so ten
        // calls cost ten rows instead of twenty.
        <Box flexDirection="column" marginTop={prevKind === "tool" ? 0 : 1}>
          <Text color={headColor}>
            {glyph} <Text bold>{toolVerb(item.name)}</Text>
            {target ? `(${target})` : ""}
          </Text>
          {item.running ? (
            <Text color={palette.muted}> ↳ …</Text>
          ) : (
            <Text color={item.isError ? palette.toolErr : palette.muted}>
              {" ↳ "}
              {item.isError ? preview(item.output ?? "") : toolDetail(item)}
            </Text>
          )}
        </Box>
      );
    }

    case "todos": {
      // Never trust the shape here. This component also renders items restored
      // from a past session, which may predate the normalization upstream, and
      // an exception in a render kills the entire UI — so a malformed payload
      // renders as nothing rather than taking the session with it.
      const todos = Array.isArray(item.todos) ? item.todos : [];
      if (todos.length === 0) return null;
      const done = todos.filter((t) => t?.status === "completed").length;
      return (
        <Box flexDirection="column" marginTop={1}>
          <Text bold color={palette.primary}>
            Tasks{" "}
            <Text color={palette.muted}>
              ({done}/{todos.length})
            </Text>
          </Text>
          {todos.map((t, i) => {
            const g = TODO_GLYPH[t?.status] ?? "○";
            // Coerce before rendering: an object here is a React child, and
            // "Objects are not valid as a React child" escapes the render.
            const text = typeof t?.content === "string" ? t.content : String(t?.content ?? "");
            if (t?.status === "completed")
              return (
                <Text key={i} color={palette.muted} strikethrough>
                  {"  "}
                  {g} {text}
                </Text>
              );
            if (t?.status === "in_progress")
              return (
                <Text key={i} bold color={palette.accent}>
                  {"  "}
                  {g} {text}
                </Text>
              );
            return (
              <Text key={i} color={palette.assistant}>
                {"  "}
                {g} {text}
              </Text>
            );
          })}
        </Box>
      );
    }

    case "info":
      return (
        <Text color={palette.muted}>· {item.text}</Text>
      );

    case "error":
      return (
        <Text color={palette.error}>⚠ {item.text}</Text>
      );
  }
}
