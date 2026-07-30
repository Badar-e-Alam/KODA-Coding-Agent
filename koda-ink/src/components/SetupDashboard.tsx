import React, { useEffect, useState } from "react";
import { Box, Text, useInput } from "ink";
import type { EnvState } from "../types.js";
import type { Palette } from "../theme.js";

interface Props {
  state: EnvState | null;
  palette: Palette;
  rows: number;
  onClose: () => void;
  onRefresh: () => void;
  onSet: (key: string, value: string) => void;
}

// /setup — interactive credential editor. ↑↓ picks a key, Enter opens an
// inline field for it, Enter saves (empty value clears), Esc backs out. The
// bridge writes the value to the project .env and rebuilds the agent, so a
// key entered here takes effect on the very next turn.
export function SetupDashboard({ state, palette, rows, onClose, onRefresh, onSet }: Props) {
  const keys = state?.keys ?? [];
  const [sel, setSel] = useState(0);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");

  const cur = keys[Math.min(sel, Math.max(0, keys.length - 1))];

  // Ink 5.2.1 can silently drop the frame that shows freshly arrived data: the
  // write is skipped when it lands too close to the previous paint, yet Ink
  // still records that frame as "last written", so re-rendering the same
  // content is skipped as a no-op and the dashboard sits on "Loading…".
  // These settle ticks re-render a few times after `state` changes, each with a
  // one-character difference (an invisible trailing space), so at least one
  // frame is both far enough from the previous paint and textually different
  // from what Ink believes is on screen.
  const [settle, setSettle] = useState(0);
  useEffect(() => {
    let n = 0;
    const id = setInterval(() => {
      setSettle((s) => s + 1);
      if (++n >= 3) clearInterval(id);
    }, 60);
    return () => clearInterval(id);
  }, [state]);
  const settlePad = settle % 2 === 1 ? " " : "";

  useInput((input, key) => {
    if (editing) {
      if (key.escape) {
        setEditing(false);
        return;
      }
      if (key.return) {
        setEditing(false);
        if (cur) onSet(cur.key, draft.trim());
        return;
      }
      if (key.backspace || key.delete) {
        setDraft((d) => d.slice(0, -1));
        return;
      }
      // Ink delivers a paste as one multi-char chunk — keep it, minus newlines
      // (API keys are often pasted with a trailing newline).
      if (input && !key.ctrl && !key.meta) setDraft((d) => d + input.replace(/[\r\n]/g, ""));
      return;
    }

    if (key.escape || input === "q") {
      onClose();
      return;
    }
    if (input === "r") {
      onRefresh();
      return;
    }
    if (keys.length === 0) return;
    if (key.upArrow) setSel((s) => (s + keys.length - 1) % keys.length);
    else if (key.downArrow) setSel((s) => (s + 1) % keys.length);
    else if (key.return || input === "e") {
      setDraft("");
      setEditing(true);
    } else if (input === "d") {
      onSet(keys[sel].key, ""); // clear
    }
  });

  const hints: Array<[string, string]> = editing
    ? [["enter", "save"], ["esc", "cancel"], ["empty", "clears the key"]]
    : [
        ["↑↓", "select"],
        ["enter/e", "edit"],
        ["d", "clear"],
        ["r", "refresh"],
        ["q/esc", "close"],
      ];

  // Leave room for the header, footer and (when editing) the input row.
  const maxRows = Math.max(4, rows - (editing ? 11 : 9));
  const start = Math.max(0, Math.min(sel - maxRows + 1, keys.length - maxRows));
  const visible = keys.slice(Math.max(0, start), Math.max(0, start) + maxRows);

  return (
    <Box flexDirection="column" borderStyle="round" borderColor={palette.accent} paddingX={1}>
      <Text color={palette.accent} bold>
        Setup — API keys & defaults{" "}
        <Text color={palette.muted}>({keys.filter((k) => k.set).length}/{keys.length} set)</Text>
      </Text>
      <Text color={palette.muted}>
        writes to {state?.path || ".env"} · model {state?.model || "—"}
      </Text>

      <Box flexDirection="column" marginTop={1}>
        {keys.length === 0 ? (
          <Text color={palette.muted}>Loading… (press r to refresh)</Text>
        ) : (
          visible.map((k) => {
            const active = keys.indexOf(k) === sel;
            const status = k.set ? k.value || "set" : "not set";
            const statusCol = k.set ? palette.toolOk : palette.muted;
            return (
              <Text key={k.key} color={active ? palette.accent : palette.assistant} inverse={active}>
                {active ? "❯ " : "  "}
                <Text bold>{k.key.padEnd(20)}</Text>{" "}
                <Text color={statusCol}>{status}</Text>{" "}
                <Text color={palette.muted}>
                  {k.source ? `[${k.source}]` : ""} {k.label}
                </Text>
              </Text>
            );
          })
        )}
      </Box>

      {editing && cur ? (
        <Box flexDirection="column" marginTop={1}>
          <Text color={palette.accent}>
            {cur.key} <Text color={palette.muted}>({cur.hint})</Text>
          </Text>
          <Text color={palette.assistant}>
            {"› "}
            {cur.secret ? "•".repeat(draft.length) : draft}
            <Text inverse> </Text>
          </Text>
        </Box>
      ) : null}

      <Box marginTop={1}>
        {hints.map(([k, label]) => (
          <Text key={k}>
            <Text color={palette.accent}>{k}</Text>
            <Text color={palette.muted}> {label}  </Text>
          </Text>
        ))}
      </Box>
      {!editing ? (
        <Text color={palette.muted}>Saved keys apply immediately · /model switches provider{settlePad}</Text>
      ) : null}
    </Box>
  );
}
