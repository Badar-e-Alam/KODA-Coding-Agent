import React, { useState } from "react";
import { Box, Text, useInput } from "ink";
import type { Palette } from "../theme.js";

interface Props {
  question: string;
  options: string[];
  palette: Palette;
  onAnswer: (value: string) => void;
}

// Inline prompt for the agent's ask_user tool. With options: ↑/↓ + Enter picks
// one; typing switches to free-text. Without options: free-text only.
export function AskUser({ question, options: rawOptions, palette, onAnswer }: Props) {
  const [sel, setSel] = useState(0);
  const [text, setText] = useState("");
  // `options` comes from the model's own tool call, so the declared string[] is
  // a hope, not a guarantee. A non-string entry rendered as a React child
  // throws ("Objects are not valid as a React child") and takes the session
  // down. The bridge normalizes this too; this is the render-side backstop.
  const options = React.useMemo(
    () =>
      (Array.isArray(rawOptions) ? rawOptions : [])
        .map((o) => (typeof o === "string" ? o : String(o ?? "")))
        .map((o) => o.trim())
        .filter(Boolean),
    [rawOptions],
  );
  const safeQuestion = typeof question === "string" ? question : String(question ?? "");
  const freeText = text.length > 0 || options.length === 0;

  useInput((input, key) => {
    if (key.return) {
      if (freeText) {
        if (text.trim()) onAnswer(text.trim());
      } else {
        onAnswer(options[Math.min(sel, options.length - 1)] ?? "");
      }
      return;
    }
    if (!freeText && key.upArrow) {
      setSel((s) => (s + options.length - 1) % options.length);
      return;
    }
    if (!freeText && key.downArrow) {
      setSel((s) => (s + 1) % options.length);
      return;
    }
    if (key.backspace || key.delete) {
      setText((t) => t.slice(0, -1));
      return;
    }
    if (key.escape) {
      setText("");
      return;
    }
    if (input && !key.ctrl && !key.meta) {
      setText((t) => t + input.replace(/[\r\n]/g, " "));
    }
  });

  return (
    <Box flexDirection="column" borderStyle="round" borderColor={palette.accent} paddingX={1} marginTop={1}>
      <Text color={palette.accent} bold>
        The agent has a question
      </Text>
      <Text color={palette.assistant}>{safeQuestion}</Text>
      {options.length > 0 ? (
        <Box flexDirection="column" marginTop={1}>
          {options.map((o, i) => (
            <Text
              key={i}
              color={!freeText && i === sel ? palette.accent : palette.assistant}
              inverse={!freeText && i === sel}
              dimColor={freeText}
            >
              {!freeText && i === sel ? "❯ " : "  "}
              {o}
            </Text>
          ))}
        </Box>
      ) : null}
      <Box marginTop={1}>
        <Text color={palette.muted}>{"answer: "}</Text>
        <Text color={palette.assistant}>{text}</Text>
        <Text inverse> </Text>
      </Box>
      <Text color={palette.muted}>
        {options.length > 0 ? "↑/↓ + Enter to pick · or type a custom answer + Enter" : "type your answer + Enter"}
      </Text>
    </Box>
  );
}
