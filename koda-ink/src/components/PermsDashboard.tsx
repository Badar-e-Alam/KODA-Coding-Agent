import React, { useState } from "react";
import { Box, Text, useInput } from "ink";
import type { PermsState } from "../types.js";
import type { Palette } from "../theme.js";

interface Props {
  state: PermsState | null;
  palette: Palette;
  rows: number;
  onClose: () => void;
  onRefresh: () => void;
  onClearAllowed: () => void;
  onSetMode: (mode: string) => void;
}

const MODES = ["default", "edits", "plan"] as const;

export function PermsDashboard({
  state,
  palette,
  rows,
  onClose,
  onRefresh,
  onClearAllowed,
  onSetMode,
}: Props) {
  const [sel, setSel] = useState(0);

  useInput((input, key) => {
    if (key.escape || input === "q") {
      onClose();
      return;
    }
    if (input === "r") {
      onRefresh();
      return;
    }
    if (input === "c") {
      onClearAllowed();
      return;
    }
    if (state && state.allowed_tools.length > 0) {
      if (key.upArrow) setSel((s) => (s + state.allowed_tools.length - 1) % state.allowed_tools.length);
      else if (key.downArrow) setSel((s) => (s + 1) % state.allowed_tools.length);
    }
  });

  const mode = state?.mode ?? "default";
  const allowed = state?.allowed_tools ?? [];
  const mutating = state?.mutating_tools ?? [];
  const fileEdits = state?.file_edit_tools ?? [];

  return (
    <Box flexDirection="column" borderStyle="round" borderColor={palette.accent} paddingX={1}>
      <Text color={palette.accent} bold>
        Permissions{" "}
        <Text color={palette.muted}>(session state — grants reset on /clear or restart)</Text>
      </Text>

      {/* Mode */}
      <Box marginTop={1}>
        <Text color={palette.assistant}>
          Mode:{" "}
          <Text bold color={palette.accent}>{mode}</Text>
        </Text>
        <Text color={palette.muted}>  [</Text>
        {MODES.map((m, i) => (
          <Text key={m}>
            {i > 0 && <Text color={palette.muted}>|</Text>}
            <Text
              color={m === mode ? palette.accent : palette.muted}
              bold={m === mode}
            >
              {m === mode ? ` ${m} ` : ` ${m} `}
            </Text>
          </Text>
        ))}
        <Text color={palette.muted}>]</Text>
      </Box>

      {/* Auto-approve (read-only) */}
      <Text color={state?.auto_approve ? palette.toolErr : palette.muted}>
        auto-approve: {state ? (state.auto_approve ? "ON (unattended mode)" : "off") : "?"}
      </Text>

      {/* Always-allowed tools */}
      <Box flexDirection="column" marginTop={1}>
        <Text color={palette.assistant} bold>
          Always-allowed this session ({allowed.length}):
        </Text>
        {allowed.length === 0 ? (
          <Text color={palette.muted}>  none — each gated tool will prompt</Text>
        ) : (
          allowed.map((t, i) => {
            const active = i === sel;
            return (
              <Text key={t} color={active ? palette.accent : palette.assistant} inverse={active}>
                {active ? "❯ " : "  "}
                {t}
              </Text>
            );
          })
        )}
        {allowed.length > 0 && (
          <Box marginTop={1}>
            <Text color={palette.muted}>
              <Text color={palette.accent}>c</Text> clears all session grants
            </Text>
          </Box>
        )}
      </Box>

      {/* Gated tool classes */}
      <Box flexDirection="column" marginTop={1}>
        <Text color={palette.assistant} bold>
          Gated tools (prompt by default):
        </Text>
        <Text color={palette.muted}>  mutating: {mutating.join(", ")}</Text>
        <Text color={palette.muted}>  file-edits: {fileEdits.join(", ")}</Text>
      </Box>

      <Box marginTop={1}>
        <Text>
          <Text color={palette.accent}>r</Text>
          <Text color={palette.muted}> refresh  </Text>
          <Text color={palette.accent}>c</Text>
          <Text color={palette.muted}> clear grants  </Text>
          <Text color={palette.accent}>q/esc</Text>
          <Text color={palette.muted}> close</Text>
        </Text>
      </Box>
    </Box>
  );
}