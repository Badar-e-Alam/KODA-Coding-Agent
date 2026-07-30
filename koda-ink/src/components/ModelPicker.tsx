import React, { useMemo, useState } from "react";
import { Box, Text, useInput } from "ink";
import type { Palette } from "../theme.js";

interface Props {
  models: string[];
  current: string;
  palette: Palette;
  rows: number;
  onPick: (model: string) => void;
  onClose: () => void;
}

// /model popup — switch the agent's model. The list is the bridge's discovered
// provider:model set (sent on `ready`); the current one is highlighted and
// pinned to the top. ↑/↓ selects, Enter switches (the bridge emits
// model_changed), Esc/q cancels, typing filters by substring.
//
// Grouped by provider (the part before ':') with a subtle section header, so a
// long flat list stays navigable. Counts are shown per provider.
export function ModelPicker({ models, current, palette, rows, onPick, onClose }: Props) {
  const [filter, setFilter] = useState("");
  const [sel, setSel] = useState(() => {
    const i = models.indexOf(current);
    return i >= 0 ? i : 0;
  });

  // Build a grouped, filtered list. The current model is always included even
  // if it isn't in the discovered set (e.g. set via --model before discovery).
  const grouped = useMemo(() => {
    const all = models.length ? models : [current];
    const seen = new Set(all);
    if (!seen.has(current)) all.unshift(current);
    const f = filter.trim().toLowerCase();
    const filtered = f ? all.filter((m) => m.toLowerCase().includes(f)) : all;
    const order: string[] = [];
    const byProv: Record<string, string[]> = {};
    for (const m of filtered) {
      const p = m.includes(":") ? m.split(":")[0] : "other";
      if (!byProv[p]) { byProv[p] = []; order.push(p); }
      byProv[p].push(m);
    }
    return { order, byProv, flat: filtered };
  }, [models, current, filter]);

  const n = grouped.flat.length;
  const idx = n ? Math.min(sel, n - 1) : 0;
  const picked = grouped.flat[idx];

  useInput((input, key) => {
    if (key.escape || input === "q") { onClose(); return; }
    if (key.upArrow) { setSel((s) => (n + s - 1) % Math.max(1, n)); return; }
    if (key.downArrow) { setSel((s) => (n + s + 1) % Math.max(1, n)); return; }
    if (key.return && picked) { onPick(picked); return; }
    if (key.backspace || key.delete) { setFilter((f) => f.slice(0, -1)); setSel(0); return; }
    if (input && !key.ctrl && !key.meta && !key.return && !key.tab) {
      setFilter((f) => f + input);
      setSel(0);
    }
  });

  // Window the list to the available height (header + footer + filter line).
  const headerRows = 3;
  const footerRows = 1;
  const maxRows = Math.max(3, rows - headerRows - footerRows - 2);
  const start = Math.max(0, Math.min(idx - Math.floor(maxRows / 2), Math.max(0, n - maxRows)));

  // Render the grouped list, computing the global running index to keep `idx`
  // aligned with `start` windowing across provider headers.
  const rendered: React.ReactNode[] = [];
  let running = -1;
  for (const prov of grouped.order) {
    rendered.push(
      <Text key={`h-${prov}`} color={palette.muted} bold>
        {prov} <Text color={palette.muted}>({grouped.byProv[prov].length})</Text>
      </Text>,
    );
    for (const m of grouped.byProv[prov]) {
      running += 1;
      const globalIdx = running;
      if (globalIdx < start || globalIdx >= start + maxRows) continue;
      const active = globalIdx === idx;
      const isCur = m === current;
      rendered.push(
        <Box key={m}>
          <Text color={active ? palette.accent : palette.muted}>{active ? "❯ " : "  "}</Text>
          <Text color={isCur ? palette.toolOk : active ? palette.accent : palette.assistant} inverse={active} bold={isCur}>
            {m}
            {isCur ? " ✓" : ""}
          </Text>
        </Box>,
      );
    }
  }

  return (
    <Box flexDirection="column" borderStyle="round" borderColor={palette.accent} paddingX={1} marginTop={1}>
      <Text color={palette.primary} bold>
        Model <Text color={palette.muted}>(current: {current})</Text>
      </Text>
      <Box>
        <Text color={palette.muted}>filter: </Text>
        <Text color={palette.assistant}>{filter || "…"}</Text>
      </Box>
      <Box flexDirection="column" marginTop={1}>
        {n === 0 ? (
          <Text color={palette.muted}>No models match “{filter}”.</Text>
        ) : (
          rendered
        )}
      </Box>
      <Box marginTop={1}>
        <Text>
          <Text color={palette.accent}>↑↓</Text>
          <Text color={palette.muted}> select  </Text>
          <Text color={palette.accent}>enter</Text>
          <Text color={palette.muted}> switch  </Text>
          <Text color={palette.accent}>type</Text>
          <Text color={palette.muted}> filter  </Text>
          <Text color={palette.accent}>q/esc</Text>
          <Text color={palette.muted}> close</Text>
        </Text>
      </Box>
    </Box>
  );
}