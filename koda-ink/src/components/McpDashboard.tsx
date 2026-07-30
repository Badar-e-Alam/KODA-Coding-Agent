// /mcp — manage MCP servers and the permissions on the tools they expose.
//
// Four views, one at a time, each owning its own keyboard handling:
//
//   list    servers from .mcp.json, with connection + permission summary
//   add     the add-server wizard (http / sse / stdio)
//   tools   one server's tools, each set to allow | ask | deny
//   remove  a confirmation step, since removing rebuilds the agent
//
// Only one useInput is live at a time — every view returns early rather than
// rendering siblings — so a key never lands in two handlers at once.

import React, { useEffect, useState } from "react";
import { Box, Text, useInput } from "ink";
import type { McpServer, McpTestResult, McpTool, McpToolList, McpToolPolicy } from "../types.js";
import type { Palette } from "../theme.js";

export interface AddSpec {
  name: string;
  transport: string;
  url?: string;
  command?: string;
  args?: string[];
  headers?: Record<string, string>;
  env?: Record<string, string>;
  default_policy?: McpToolPolicy;
}

interface Props {
  servers: McpServer[];
  testResults: Record<string, McpTestResult>;
  toolLists: Record<string, McpToolList>;
  palette: Palette;
  rows: number;
  columns: number;
  onClose: () => void;
  onRefresh: () => void;
  onAdd: (spec: AddSpec) => void;
  onRemove: (name: string) => void;
  onTest: (name: string) => void;
  onLoadTools: (name: string, refresh: boolean) => void;
  onSetToolPerm: (server: string, tool: string, policy: McpToolPolicy) => void;
  onSetAllToolPerms: (server: string, policy: McpToolPolicy) => void;
  onAuthorize: (name: string) => void;
  onSignOut: (name: string) => void;
}

const POLICIES: McpToolPolicy[] = ["allow", "ask", "deny"];
const TRANSPORTS = ["http", "sse", "stdio"];

function policyColor(p: McpToolPolicy, palette: Palette): string {
  if (p === "allow") return palette.toolOk;
  if (p === "deny") return palette.toolErr;
  return palette.tool;
}

function policyGlyph(p: McpToolPolicy): string {
  if (p === "allow") return "●";
  if (p === "deny") return "✕";
  return "◐";
}

/** Width that fits the longest label, clamped to [min, max]. */
function fitWidth(labels: string[], min: number, max: number): number {
  const longest = labels.reduce((widest, label) => Math.max(widest, label.length), 0);
  return Math.max(min, Math.min(max, longest));
}

// A window of `size` rows around `sel`, so a 30-tool server still fits.
function windowed<T>(items: T[], sel: number, size: number): [T[], number] {
  if (items.length <= size) return [items, 0];
  const start = Math.max(0, Math.min(sel - Math.floor(size / 2), items.length - size));
  return [items.slice(start, start + size), start];
}

// ── table layout ─────────────────────────────────────────────────────
//
// Every row is laid out to the same fixed column widths and truncated to fit
// the terminal. Both matter: unpadded cells put each row's later columns at a
// different x, and any row that overflows the frame gets soft-wrapped by Yoga
// onto a second line, which shears the table in half. So each cell is padded
// to its width, one column absorbs the slack, and nothing is allowed to
// exceed the frame's inner width.

const GUTTER = 2; // the "❯ " / "  " selection marker

/** Pad to `width`, or truncate with an ellipsis when too long. */
function cell(text: string, width: number, pad = true): string {
  const s = text ?? "";
  if (width <= 0) return "";
  if (s.length > width) return width > 1 ? s.slice(0, width - 1) + "…" : s.slice(0, width);
  return pad ? s.padEnd(width) : s;
}

interface Column {
  header: string;
  /** Fixed width, or omitted for the one column that takes the remainder. */
  width?: number;
  /** Smallest useful width for the flexible column. */
  min?: number;
}

/**
 * Resolve column widths against the space actually available.
 *
 * `inner` is the frame's content width (terminal minus border and padding).
 * The flexible column gets whatever the fixed ones leave; if that is less
 * than its `min`, later fixed columns are dropped rather than wrapped — a
 * narrow terminal shows fewer columns instead of a mangled table.
 */
function layout(cols: Column[], inner: number): number[] {
  const usable = Math.max(0, inner - GUTTER);
  const flexAt = cols.findIndex((c) => c.width === undefined);
  const spacing = Math.max(0, cols.length - 1); // one space between columns
  const fixed = cols.reduce((sum, c) => sum + (c.width ?? 0), 0) + spacing;

  if (flexAt < 0) return cols.map((c) => c.width ?? 0);

  let widths = cols.map((c) => c.width ?? Math.max(0, usable - fixed));
  const min = cols[flexAt].min ?? 8;
  // Too tight: drop trailing fixed columns (rightmost first) to buy room.
  for (let last = cols.length - 1; widths[flexAt] < min && last > flexAt; last--) {
    widths[last] = 0;
    const dropped = (cols[last].width ?? 0) + 1;
    widths[flexAt] += dropped;
  }
  widths[flexAt] = Math.max(0, widths[flexAt]);
  return widths;
}

function HeaderRow({
  cols,
  widths,
  palette,
}: {
  cols: Column[];
  widths: number[];
  palette: Palette;
}) {
  const line = cols
    .map((c, i) => (widths[i] > 0 ? cell(c.header.toUpperCase(), widths[i]) : null))
    .filter((s): s is string => s !== null)
    .join(" ");
  const total = Math.min(line.length, widths.reduce((a, b) => a + b, 0) + widths.filter((w) => w > 0).length - 1);
  return (
    <Box flexDirection="column">
      <Text color={palette.muted}>{" ".repeat(GUTTER) + line.trimEnd()}</Text>
      <Text color={palette.muted}>{" ".repeat(GUTTER) + "─".repeat(Math.max(0, total))}</Text>
    </Box>
  );
}

function Hints({ pairs, palette }: { pairs: Array<[string, string]>; palette: Palette }) {
  return (
    <Box marginTop={1} flexWrap="wrap">
      {pairs.map(([key, label]) => (
        <Text key={key + label}>
          <Text color={palette.accent}>{key}</Text>
          <Text color={palette.muted}>{` ${label}   `}</Text>
        </Text>
      ))}
    </Box>
  );
}

function Frame({
  title,
  subtitle,
  palette,
  children,
}: {
  title: React.ReactNode;
  subtitle?: string;
  palette: Palette;
  children: React.ReactNode;
}) {
  return (
    <Box flexDirection="column" borderStyle="round" borderColor={palette.accent} paddingX={1}>
      <Text color={palette.accent} bold>
        {title}
      </Text>
      {subtitle ? <Text color={palette.muted}>{subtitle}</Text> : null}
      {children}
    </Box>
  );
}

// ── add-server wizard ────────────────────────────────────────────────

type FieldKind = "text" | "choice";
interface Field {
  key: string;
  label: string;
  kind: FieldKind;
  hint?: string;
  choices?: string[];
  /** Hidden for the current transport (stdio has no url, http has no command). */
  skip?: (values: Record<string, string>) => boolean;
}

const FIELDS: Field[] = [
  { key: "name", label: "name", kind: "text", hint: "how you'll refer to it, e.g. notion" },
  { key: "transport", label: "transport", kind: "choice", choices: TRANSPORTS, hint: "←/→ to change" },
  {
    key: "url",
    label: "url",
    kind: "text",
    hint: "https://mcp.notion.com/mcp",
    skip: (v) => v.transport === "stdio",
  },
  {
    key: "command",
    label: "command",
    kind: "text",
    hint: "npx",
    skip: (v) => v.transport !== "stdio",
  },
  {
    key: "args",
    label: "args",
    kind: "text",
    hint: "space-separated, e.g. -y @modelcontextprotocol/server-github",
    skip: (v) => v.transport !== "stdio",
  },
  {
    key: "headers",
    label: "headers",
    kind: "text",
    // Cloud servers use OAuth, not a header — leave this empty and press `o`.
    // Setting an Authorization header opts out of OAuth entirely.
    hint: 'leave empty for OAuth sign-in · or JSON, e.g. {"Authorization": "Bearer …"}',
    skip: (v) => v.transport === "stdio",
  },
  {
    key: "env",
    label: "env",
    kind: "text",
    hint: 'optional JSON, e.g. {"GITHUB_TOKEN": "…"}',
    skip: (v) => v.transport !== "stdio",
  },
  {
    key: "default_policy",
    label: "tool permission",
    kind: "choice",
    choices: POLICIES,
    hint: "starting policy for every tool this server exposes",
  },
];

function AddForm({
  palette,
  existing,
  onCancel,
  onSubmit,
}: {
  palette: Palette;
  existing: string[];
  onCancel: () => void;
  onSubmit: (spec: AddSpec) => void;
}) {
  const [values, setValues] = useState<Record<string, string>>({
    name: "",
    transport: "http",
    url: "",
    command: "",
    args: "",
    headers: "",
    env: "",
    default_policy: "ask",
  });
  const [idx, setIdx] = useState(0);
  const [err, setErr] = useState("");

  const visible = FIELDS.filter((f) => !f.skip?.(values));
  const active = visible[Math.min(idx, visible.length - 1)];

  const set = (key: string, value: string) => setValues((v) => ({ ...v, [key]: value }));

  // Validate and build the spec. Returns an error string instead of failing
  // silently — the old form dropped malformed input on the floor, so "add"
  // looked like it did nothing at all.
  function build(): AddSpec | string {
    const name = values.name.trim();
    if (!name) return "name is required";
    if (!/^[A-Za-z0-9_.-]+$/.test(name)) return "name may only contain letters, digits, - _ .";
    if (existing.includes(name)) return `'${name}' already exists — remove it first, or pick another name`;

    const transport = values.transport;
    const spec: AddSpec = { name, transport };

    if (transport === "stdio") {
      const command = values.command.trim();
      if (!command) return "stdio needs a command (e.g. npx)";
      spec.command = command;
      const args = values.args.trim();
      if (args) spec.args = args.split(/\s+/);
      if (values.env.trim()) {
        const parsed = parseJsonObject(values.env);
        if (typeof parsed === "string") return `env: ${parsed}`;
        spec.env = parsed;
      }
    } else {
      const url = values.url.trim();
      if (!url) return `${transport} needs a url`;
      if (!/^https?:\/\//i.test(url)) return "url must start with http:// or https://";
      spec.url = url;
      if (values.headers.trim()) {
        const parsed = parseJsonObject(values.headers);
        if (typeof parsed === "string") return `headers: ${parsed}`;
        spec.headers = parsed;
      }
    }
    spec.default_policy = values.default_policy as McpToolPolicy;
    return spec;
  }

  useInput((input, key) => {
    if (key.escape) {
      onCancel();
      return;
    }
    if (key.upArrow || (key.tab && key.shift)) {
      setErr("");
      setIdx((i) => (i + visible.length - 1) % visible.length);
      return;
    }
    if (key.downArrow || key.tab) {
      setErr("");
      setIdx((i) => (i + 1) % visible.length);
      return;
    }
    if (active.kind === "choice" && (key.leftArrow || key.rightArrow)) {
      const choices = active.choices!;
      const at = Math.max(0, choices.indexOf(values[active.key]));
      const next = (at + (key.rightArrow ? 1 : choices.length - 1)) % choices.length;
      set(active.key, choices[next]);
      setErr("");
      return;
    }
    if (key.return) {
      // Enter advances until the last field, then submits. Ctrl+S submits
      // from anywhere once the required fields are filled.
      if (idx < visible.length - 1) {
        setErr("");
        setIdx(idx + 1);
        return;
      }
      const result = build();
      if (typeof result === "string") {
        setErr(result);
        return;
      }
      onSubmit(result);
      return;
    }
    if (key.ctrl && input === "s") {
      const result = build();
      if (typeof result === "string") setErr(result);
      else onSubmit(result);
      return;
    }
    if (active.kind !== "text") return;
    if (key.backspace || key.delete) {
      set(active.key, values[active.key].slice(0, -1));
      setErr("");
      return;
    }
    if (input && !key.ctrl && !key.meta) {
      set(active.key, values[active.key] + input);
      setErr("");
    }
  });

  return (
    <Frame
      palette={palette}
      title="Add MCP server"
      subtitle="↑↓/tab move · ←/→ change choice · enter next / add · ctrl+s add now · esc cancel"
    >
      <Box flexDirection="column" marginTop={1}>
        {visible.map((f) => {
          const on = f.key === active.key;
          const value = values[f.key];
          return (
            <Box key={f.key} flexDirection="column">
              <Text>
                <Text color={on ? palette.accent : palette.muted}>{on ? "❯ " : "  "}</Text>
                <Text color={on ? palette.accent : palette.muted}>{f.label.padEnd(16)}</Text>
                {f.kind === "choice" ? (
                  <Text>
                    {f.choices!.map((c) => (
                      <Text
                        key={c}
                        bold={c === value}
                        color={
                          c === value
                            ? f.key === "default_policy"
                              ? policyColor(c as McpToolPolicy, palette)
                              : palette.assistant
                            : palette.muted
                        }
                      >
                        {cell(c === value ? `[${c}]` : ` ${c} `, 9)}
                      </Text>
                    ))}
                  </Text>
                ) : (
                  <Text color={value ? palette.assistant : palette.muted}>
                    {value || (on ? "▌" : "—")}
                  </Text>
                )}
              </Text>
              {on && f.hint ? <Text color={palette.muted}>{`    ${f.hint}`}</Text> : null}
            </Box>
          );
        })}
      </Box>
      {err ? (
        <Box marginTop={1}>
          <Text color={palette.error}>✗ {err}</Text>
        </Box>
      ) : null}
    </Frame>
  );
}

/** Parse a JSON object literal; returns the object or an error string. */
function parseJsonObject(text: string): Record<string, string> | string {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (e) {
    return `not valid JSON (${(e as Error).message})`;
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    return "must be a JSON object";
  }
  return Object.fromEntries(Object.entries(parsed as Record<string, unknown>).map(([k, v]) => [k, String(v)]));
}

// ── tool permission view ─────────────────────────────────────────────

function ToolPerms({
  server,
  list,
  palette,
  rows,
  columns,
  onBack,
  onRefresh,
  onSet,
  onSetAll,
}: {
  server: McpServer;
  list: McpToolList | undefined;
  palette: Palette;
  rows: number;
  columns: number;
  onBack: () => void;
  onRefresh: () => void;
  onSet: (tool: string, policy: McpToolPolicy) => void;
  onSetAll: (policy: McpToolPolicy) => void;
}) {
  // Selection is tracked by tool *name*, not row index: every toggle round-
  // trips through the bridge and re-emits the list, and a reordered list would
  // otherwise slide the cursor onto a different tool mid-edit.
  const [selName, setSelName] = useState<string | null>(null);
  const tools: McpTool[] = list?.tools ?? [];
  const perms = list?.permissions ?? {};
  const sel = Math.max(0, tools.findIndex((t) => t.name === selName));

  const policyOf = (name: string): McpToolPolicy =>
    (perms[name] as McpToolPolicy) ?? (server.default_policy as McpToolPolicy) ?? "ask";

  const cycle = (dir: 1 | -1) => {
    const tool = tools[sel];
    if (!tool) return;
    const at = POLICIES.indexOf(policyOf(tool.name));
    const next = POLICIES[(at + (dir === 1 ? 1 : POLICIES.length - 1)) % POLICIES.length];
    onSet(tool.name, next);
  };

  useInput((input, key) => {
    if (key.escape || input === "q") {
      onBack();
      return;
    }
    if (input === "r") {
      onRefresh();
      return;
    }
    if (input === "A") {
      onSetAll("allow");
      return;
    }
    if (input === "D") {
      onSetAll("deny");
      return;
    }
    if (input === "K") {
      onSetAll("ask");
      return;
    }
    if (!tools.length) return;
    if (key.upArrow) setSelName(tools[(sel + tools.length - 1) % tools.length].name);
    else if (key.downArrow) setSelName(tools[(sel + 1) % tools.length].name);
    else if (key.rightArrow || input === " " || key.return) cycle(1);
    else if (key.leftArrow) cycle(-1);
  });

  const counts = tools.reduce<Record<string, number>>((acc, t) => {
    const p = policyOf(t.name);
    acc[p] = (acc[p] ?? 0) + 1;
    return acc;
  }, {});

  // One line per tool. A server like Notion exposes 25+, so the description
  // shares the row instead of taking one of its own; the selected tool's full
  // description goes under the table.
  const capacity = Math.max(3, rows - 13);
  const [shown, offset] = windowed(tools, sel, capacity);

  const summary = POLICIES.filter((p) => counts[p])
    .map((p) => `${counts[p]} ${p}`)
    .join(" · ");

  const nameWidth = fitWidth(["tool", ...tools.map((t) => t.name)], 10, 30);
  const cols: Column[] = [
    // "policy" not "permission": the header has to fit the column, and the
    // widest value ("● allow") only needs 7.
    { header: "tool", width: nameWidth },
    { header: "policy", width: 8 },
    { header: "description", min: 16 },
  ];
  const widths = layout(cols, Math.max(20, columns - 4));
  const selectedTool = tools[sel];

  return (
    <Frame
      palette={palette}
      title={
        <Text>
          {server.name}
          <Text color={palette.muted}>{"  tools & permissions"}</Text>
        </Text>
      }
      subtitle={
        list?.loading
          ? "connecting…"
          : summary
            ? `${tools.length} tool(s) · ${summary}${list?.cached ? " · cached" : ""}`
            : "no tools discovered yet"
      }
    >
      {list?.error ? (
        <Text color={list.needs_auth ? palette.tool : palette.error}>
          {list.needs_auth ? "⚿ " : "✗ "}
          {list.error}
        </Text>
      ) : null}

      <Box flexDirection="column" marginTop={1}>
        {tools.length === 0 && !list?.loading ? (
          list?.needs_auth ? (
            <Text color={palette.muted}>
              Go back with <Text color={palette.accent}>esc</Text> and press{" "}
              <Text color={palette.accent}>o</Text> to sign in — the tool list arrives with the
              token.
            </Text>
          ) : (
            <Text color={palette.muted}>
              Nothing to show. Press <Text color={palette.accent}>r</Text> to connect and list this
              server's tools.
            </Text>
          )
        ) : (
          <>
            <HeaderRow cols={cols} widths={widths} palette={palette} />
            {shown.map((tool, i) => {
              const on = offset + i === sel;
              const policy = policyOf(tool.name);
              return (
                <Text key={tool.name}>
                  <Text color={on ? palette.accent : palette.muted}>{on ? "❯ " : "  "}</Text>
                  <Text color={on ? palette.accent : palette.assistant} bold={on}>
                    {cell(tool.name, widths[0])}
                  </Text>
                  <Text color={policyColor(policy, palette)}>
                    {widths[1] ? " " + cell(`${policyGlyph(policy)} ${policy}`, widths[1]) : ""}
                  </Text>
                  <Text color={palette.muted}>
                    {widths[2] ? " " + cell(tool.description || "", widths[2], false) : ""}
                  </Text>
                </Text>
              );
            })}
            {tools.length > shown.length ? (
              <Text color={palette.muted}>
                {`  … ${tools.length - shown.length} more (↑↓ to scroll)`}
              </Text>
            ) : null}
            {selectedTool?.description ? (
              <Box marginTop={1}>
                <Text color={palette.muted}>
                  {cell(selectedTool.description, Math.max(20, columns - 6), false)}
                </Text>
              </Box>
            ) : null}
          </>
        )}
      </Box>

      <Hints
        palette={palette}
        pairs={[
          ["↑↓", "select"],
          ["←/→/space", "allow · ask · deny"],
          ["A", "allow all"],
          ["K", "ask all"],
          ["D", "deny all"],
          ["r", "reload from server"],
          ["esc", "back"],
        ]}
      />
    </Frame>
  );
}

// ── main dashboard ───────────────────────────────────────────────────

type View = "list" | "add" | "tools" | "remove";

export function McpDashboard({
  servers,
  testResults,
  toolLists,
  palette,
  rows,
  columns,
  onClose,
  onRefresh,
  onAdd,
  onRemove,
  onTest,
  onLoadTools,
  onSetToolPerm,
  onSetAllToolPerms,
  onAuthorize,
  onSignOut,
}: Props) {
  const [sel, setSel] = useState(0);
  const [view, setView] = useState<View>("list");

  // Servers can disappear under us (remove + rebuild), so keep `sel` valid.
  const index = servers.length ? Math.min(sel, servers.length - 1) : 0;
  const current = servers[index];

  useEffect(() => {
    if (view !== "list" && !current) setView("list");
  }, [view, current]);

  useInput(
    (input, key) => {
      if (key.escape || input === "q") {
        onClose();
        return;
      }
      if (input === "a") {
        setView("add");
        return;
      }
      if (input === "r") {
        onRefresh();
        return;
      }
      if (!servers.length) return;
      if (key.upArrow) setSel((s) => (s + servers.length - 1) % servers.length);
      else if (key.downArrow) setSel((s) => (s + 1) % servers.length);
      else if (input === "t") onTest(current.name);
      else if (input === "o" && current.oauth) onAuthorize(current.name);
      else if (input === "O" && current.authorized) onSignOut(current.name);
      else if (input === "d") setView("remove");
      else if (key.return || input === "p") {
        // Show whatever is cached immediately; only reach out to the server
        // when we have nothing to show.
        onLoadTools(current.name, !(current.tool_names?.length));
        setView("tools");
      }
    },
    { isActive: view === "list" },
  );

  useInput(
    (input, key) => {
      if (key.escape || input === "n") {
        setView("list");
        return;
      }
      if (input === "y" || key.return) {
        onRemove(current.name);
        setView("list");
      }
    },
    { isActive: view === "remove" },
  );

  if (view === "add") {
    return (
      <AddForm
        palette={palette}
        existing={servers.map((s) => s.name)}
        onCancel={() => setView("list")}
        onSubmit={(spec) => {
          setView("list");
          onAdd(spec);
        }}
      />
    );
  }

  if (view === "tools" && current) {
    return (
      <ToolPerms
        server={current}
        list={toolLists[current.name]}
        palette={palette}
        rows={rows}
        columns={columns}
        onBack={() => setView("list")}
        onRefresh={() => onLoadTools(current.name, true)}
        onSet={(tool, policy) => onSetToolPerm(current.name, tool, policy)}
        onSetAll={(policy) => onSetAllToolPerms(current.name, policy)}
      />
    );
  }

  if (view === "remove" && current) {
    return (
      <Frame palette={palette} title="Remove MCP server">
        <Box marginTop={1} flexDirection="column">
          <Text color={palette.assistant}>
            Remove <Text bold>{current.name}</Text> from .mcp.json and rebuild the agent?
          </Text>
          <Text color={palette.muted}>Its tools and their permissions go away with it.</Text>
        </Box>
        <Hints
          palette={palette}
          pairs={[
            ["y/enter", "remove"],
            ["n/esc", "keep"],
          ]}
        />
      </Frame>
    );
  }

  // One line per server: the table stays aligned and more rows fit. Anything
  // that needs prose (an error, a sign-in nudge) goes in the detail line under
  // the table, for the selected row only.
  const capacity = Math.max(3, rows - 12);
  const [shown, offset] = windowed(servers, index, capacity);

  const nameWidth = fitWidth(["server", ...servers.map((s) => s.name)], 6, 18);
  const cols: Column[] = [
    { header: "server", width: nameWidth },
    { header: "type", width: 5 },
    { header: "endpoint", min: 14 },
    { header: "auth", width: 9 },
    { header: "tools", width: 13 },
  ];
  const widths = layout(cols, Math.max(20, columns - 4));

  const describe = (s: McpServer): { auth: string; authColor: string; needsAuth: boolean } => {
    const test = testResults[s.name];
    const needsAuth = !s.authorized && ((test && !test.ok && test.needs_auth) || !!s.requires_auth);
    if (s.authorized) return { auth: "⚿ ok", authColor: palette.toolOk, needsAuth: false };
    if (needsAuth) return { auth: "sign in", authColor: palette.tool, needsAuth: true };
    // "—" covers both "doesn't use OAuth" (stdio, or its own auth header) and
    // "hasn't been probed yet". Neither warrants a claim about whether the
    // server would accept us — only an observed 401 does.
    return { auth: "—", authColor: palette.muted, needsAuth: false };
  };

  const selected = current ? describe(current) : null;
  const selectedTest = current ? testResults[current.name] : undefined;
  const detail = !current
    ? ""
    : selected?.needsAuth
      ? "needs authorization — press o to sign in with your browser"
      : selectedTest && !selectedTest.ok
        ? selectedTest.error || "connection failed"
        : current.status === "error"
          ? current.error || "unrecognized entry in .mcp.json"
          : current.tool_names?.length
            ? `${current.url || current.command || ""}`
            : "tools not listed yet — press enter to load them";
  const detailColor = selected?.needsAuth
    ? palette.tool
    : (selectedTest && !selectedTest.ok) || current?.status === "error"
      ? palette.toolErr
      : palette.muted;

  return (
    <Frame
      palette={palette}
      title={
        <Text>
          MCP Servers{" "}
          <Text color={palette.muted}>({servers.length} configured · .mcp.json)</Text>
        </Text>
      }
      subtitle="Connect any MCP server — Notion, GitHub, Slack, Context7, or a local stdio one"
    >
      <Box flexDirection="column" marginTop={1}>
        {servers.length === 0 ? (
          <Text color={palette.muted}>
            No servers yet. Press <Text color={palette.accent}>a</Text> to add one.
          </Text>
        ) : (
          <>
            <HeaderRow cols={cols} widths={widths} palette={palette} />
            {shown.map((s, i) => {
              const on = offset + i === index;
              const { auth, authColor } = describe(s);
              const perms = s.tool_permissions ?? {};
              const names = s.tool_names ?? [];
              const fallback = (s.default_policy as McpToolPolicy) ?? "ask";
              const counts = names.reduce<Record<string, number>>((acc, n) => {
                const p = (perms[n] as McpToolPolicy) ?? fallback;
                acc[p] = (acc[p] ?? 0) + 1;
                return acc;
              }, {});
              // "4 · 3a/1k" — count, then the policy split. Short by design:
              // this column has to survive a narrow terminal.
              const split = POLICIES.filter((p) => counts[p])
                .map((p) => `${counts[p]}${p[0]}`)
                .join("/");
              const toolsCell = names.length ? `${names.length} · ${split}` : "—";
              const endpoint =
                s.url || [s.command, ...(s.args ?? [])].filter(Boolean).join(" ") || "";

              return (
                <Text key={s.name}>
                  <Text color={on ? palette.accent : palette.muted}>{on ? "❯ " : "  "}</Text>
                  <Text color={on ? palette.accent : palette.assistant} bold={on}>
                    {cell(s.name, widths[0])}
                  </Text>
                  <Text color={palette.muted}>{widths[1] ? " " + cell(s.transport, widths[1]) : ""}</Text>
                  <Text color={palette.muted}>{widths[2] ? " " + cell(endpoint, widths[2]) : ""}</Text>
                  <Text color={authColor}>{widths[3] ? " " + cell(auth, widths[3]) : ""}</Text>
                  <Text color={palette.muted}>
                    {widths[4] ? " " + cell(toolsCell, widths[4], false) : ""}
                  </Text>
                </Text>
              );
            })}
            {servers.length > shown.length ? (
              <Text color={palette.muted}>
                {`  … ${servers.length - shown.length} more (↑↓ to scroll)`}
              </Text>
            ) : null}
            {detail ? (
              <Box marginTop={1}>
                <Text color={detailColor}>{cell(detail, Math.max(20, columns - 6), false)}</Text>
              </Box>
            ) : null}
          </>
        )}
      </Box>

      <Hints
        palette={palette}
        pairs={[
          ["enter", "tools & permissions"],
          ["a", "add"],
          ["o", "sign in"],
          ...(current?.authorized ? ([["O", "sign out"]] as Array<[string, string]>) : []),
          ["t", "test"],
          ["r", "refresh"],
          ["d", "remove"],
          ["↑↓", "select"],
          ["q/esc", "close"],
        ]}
      />
    </Frame>
  );
}
