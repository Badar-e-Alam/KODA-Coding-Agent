import React, { useMemo, useState } from "react";
import { Box, Text, useInput } from "ink";
import type { TreeNode } from "../types.js";
import type { Palette } from "../theme.js";

interface Props {
  sessionId: string;
  messageCount: number;
  leaf: string | null;
  nodes: TreeNode[];
  palette: Palette;
  rows: number;
  onJump: (nodeId: string) => void;
  onClose: () => void;
}

// ── /tree popup ───────────────────────────────────────────────────
//
// Renders the branchable session tree, marks the active path through the
// graph, and lets the user jump to any node to branch from there. Visual:
// the flat node list is grouped by parent, with a tree-drawn prefix per row
// (├── / └──) so the structure reads at a glance. The current leaf is pinned
// to the bottom of the list with a "you are here" star, and a row stays in
// view by scrolling when the tree is taller than the terminal.

interface Layout {
  key: string;
  prefix: string;       // "    " / "  ├─ " / "  └─ " / "    "
  node: TreeNode;
  hasNext: boolean;     // siblings after this one (controls ├ vs └)
}

function layoutTree(nodes: TreeNode[]): Layout[] {
  // Reconstruct parent → children index from the flat list.
  const byParent: Record<string, TreeNode[]> = {};
  for (const n of nodes) {
    const k = n.parent_id ?? "__root__";
    (byParent[k] ??= []).push(n);
  }
  // Render the tree depth-first. The prefix grows as we descend, each level
  // adding either "│  " (more siblings below) or "   " (last sibling).
  const out: Layout[] = [];
  function walk(parentId: string | null, depth: number, prefix: string, isRoot: boolean) {
    const k = parentId ?? "__root__";
    const siblings = byParent[k] ?? [];
    siblings.forEach((n, i) => {
      const isLast = i === siblings.length - 1;
      const branch = isRoot ? "" : isLast ? "└─ " : "├─ ";
      out.push({ key: n.id, prefix: prefix + branch, node: n, hasNext: !isLast });
      walk(n.id, depth + 1, prefix + (isRoot ? "" : isLast ? "   " : "│  "), false);
    });
  }
  walk(null, 0, "", true);
  return out;
}

function roleGlyph(node: TreeNode, palette: Palette): React.ReactNode {
  if (node.type === "compaction") {
    return <Text color={palette.muted}>∑</Text>;
  }
  if (node.role === "user") {
    return <Text color={palette.user}>›</Text>;
  }
  return <Text color={palette.assistant}>·</Text>;
}

function previewText(node: TreeNode, max: number): string {
  if (node.type === "compaction") return "compaction";
  const t = (node.preview ?? "").replace(/\s+/g, " ").trim();
  if (t.length <= max) return t;
  return t.slice(0, Math.max(0, max - 1)) + "…";
}

export function TreePopup({ sessionId, messageCount, leaf, nodes, palette, rows, onJump, onClose }: Props) {
  const layout = useMemo(() => layoutTree(nodes), [nodes]);
  const [sel, setSel] = useState(() => {
    // Land the cursor on the leaf if known, else the last node, else 0.
    if (leaf) {
      const i = layout.findIndex((l) => l.node.id === leaf);
      if (i >= 0) return i;
    }
    return Math.max(0, layout.length - 1);
  });

  useInput((input, key) => {
    if (key.escape || input === "q") { onClose(); return; }
    if (!layout.length) return;
    if (key.upArrow)   setSel((s) => (s + layout.length - 1) % layout.length);
    if (key.downArrow) setSel((s) => (s + 1) % layout.length);
    if (key.return)    onJump(layout[sel].node.id);
  });

  if (!layout.length) {
    return (
      <Box flexDirection="column" borderStyle="round" borderColor={palette.accent} paddingX={1} marginTop={1}>
        <Text color={palette.primary} bold>Session tree</Text>
        <Text color={palette.muted}>session {sessionId || "—"} · 0 messages</Text>
        <Box marginTop={1}>
          <Text color={palette.muted}>This session has no branches yet — make a few turns first.</Text>
        </Box>
        <Box marginTop={1}>
          <Text color={palette.accent}>esc</Text>
          <Text color={palette.muted}> close</Text>
        </Box>
      </Box>
    );
  }

  // Reserve rows for the header (2) + footer (1) + border padding. Scrolling
  // keeps the selected row roughly centered, the same trick the picker uses.
  const reserveTop = 2;
  const reserveBottom = 2;
  const maxRows = Math.max(3, rows - reserveTop - reserveBottom);
  const start = Math.max(0, Math.min(sel - Math.floor(maxRows / 2), Math.max(0, layout.length - maxRows)));
  const visible = layout.slice(start, start + maxRows);

  return (
    <Box flexDirection="column" borderStyle="round" borderColor={palette.accent} paddingX={1} marginTop={1}>
      <Box>
        <Text color={palette.primary} bold>Session tree</Text>
        <Text color={palette.muted}>
          {"  ·  "}session <Text color={palette.toolOk}>{sessionId || "—"}</Text>
          {"  ·  "}{messageCount} message{messageCount === 1 ? "" : "s"}
          {leaf ? <>{"  ·  "}leaf <Text color={palette.toolOk}>{leaf.slice(0, 8)}</Text></> : null}
        </Text>
      </Box>

      <Box flexDirection="column" marginTop={1}>
        {visible.map((row, i) => {
          const real = start + i;
          const active = real === sel;
          const n = row.node;
          const onActivePath = n.active;
          const isLeaf = n.leaf;
          // A node that's both on the active path and the leaf is the
          // "you are here" cursor; mark it loudly.
          const tone = onActivePath ? palette.accent : palette.assistant;
          return (
            <Box key={row.key} flexDirection="row">
              <Text color={palette.muted}>{row.prefix}</Text>
              <Text color={active ? palette.accent : palette.muted}>{active ? "❯ " : "  "}</Text>
              {roleGlyph(n, palette)}
              <Text> </Text>
              <Text color={tone} inverse={active} bold={onActivePath}>
                {previewText(n, 60)}
              </Text>
              {isLeaf ? <Text color={palette.toolOk}> ★</Text> : null}
              {onActivePath && !isLeaf ? <Text color={palette.muted}> ·</Text> : null}
            </Box>
          );
        })}
      </Box>

      <Box marginTop={1}>
        <Text>
          <Text color={palette.accent}>↑↓</Text>
          <Text color={palette.muted}> select  </Text>
          <Text color={palette.accent}>enter</Text>
          <Text color={palette.muted}> branch from here  </Text>
          <Text color={palette.accent}>q/esc</Text>
          <Text color={palette.muted}> close</Text>
        </Text>
      </Box>
    </Box>
  );
}
