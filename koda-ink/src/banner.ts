// The KODA banner, printed once to stdout before Ink mounts so it lands at the
// very top of scrollback (permanent, selectable) — matching koda/tui/widgets/banner.py.

import type { Palette } from "./theme.js";

const ART = [
  "  ██╗  ██╗  ██████╗  ██████╗   █████╗",
  "  ██║ ██╔╝ ██╔═══██╗ ██╔══██╗ ██╔══██╗",
  "  █████╔╝  ██║   ██║ ██║  ██║ ███████║",
  "  ██╔═██╗  ██║   ██║ ██║  ██║ ██╔══██║",
  "  ██║  ██╗ ╚██████╔╝ ██████╔╝ ██║  ██║",
  "  ╚═╝  ╚═╝  ╚═════╝  ╚═════╝  ╚═╝  ╚═╝",
];

const TIPS = [
  "Select & copy text and click links like a normal terminal",
  "Use /model to switch between LLM providers on the fly",
  "Use /theme <name> to change the color theme",
  "Use ! prefix for shell commands (e.g. !ls)",
  "Use @ to attach a file to your message",
  "Shift+Tab cycles default → accept-edits → plan mode",
  "Use /help to list all slash commands",
];

function hex(h: string): string {
  const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(h);
  if (!m) return "";
  const [r, g, b] = [1, 2, 3].map((i) => parseInt(m[i], 16));
  return `\x1b[38;2;${r};${g};${b}m`;
}
const RESET = "\x1b[0m";
const DIM = "\x1b[2m";

export interface BannerInfo {
  version: string;
  model: string;
  mode: string;
  palette: Palette;
  color: boolean;
}

export function bannerString(info: BannerInfo): string {
  const { palette, color } = info;
  const c = (h: string) => (color ? hex(h) : "");
  const reset = color ? RESET : "";
  const dim = color ? DIM : "";
  const accent = c(palette.accent);
  const primary = c(palette.primary);
  const muted = c(palette.muted);

  const tip = TIPS[Math.floor(Math.random() * TIPS.length)];
  const lines: string[] = [""];
  for (const row of ART) {
    lines.push(`${accent}${row}${reset}`);
  }
  lines.push("");
  lines.push(`  ${dim}Tip: ${tip}${reset}`);
  lines.push(`  ${dim}/help for commands · @ files · ! shell · Ctrl+C to interrupt${reset}`);
  lines.push("");
  return lines.join("\n");
}
