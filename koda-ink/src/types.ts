// Wire protocol shared with koda/bridge.py (NDJSON over stdio).

export interface Usage {
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  cache_write_tokens: number;
}

export interface ToolInfo {
  name: string;
  description: string;
}

export interface PermissionItem {
  tool_name: string;
  args: Record<string, unknown>;
  allowed_decisions: string[];
  description: string;
  /** Real on-disk target when the tool's path is workspace-jailed. */
  resolved_path?: string;
}

// /mcp dashboard — one configured MCP server (from .mcp.json).
export interface McpServer {
  name: string;
  transport: string; // "http" | "sse" | "stdio"
  url?: string;
  command?: string;
  args?: string[];
  headers?: Record<string, string>;
  env?: Record<string, string>;
  status: string; // "configured" | "error"
  error?: string;
  /** Tool names cached in .mcp.json from the last successful discovery. */
  tool_names?: string[];
  /** Per-tool policy from .mcp.json (tool name → allow | ask | deny). */
  tool_permissions?: Record<string, McpToolPolicy>;
  /** Policy applied to tools with no explicit entry. */
  default_policy?: McpToolPolicy;
  /** True when this server can use OAuth (http/sse, no manual auth header). */
  oauth?: boolean;
  /** True when a stored OAuth token exists for it. */
  authorized?: boolean;
  /** True once the server has actually answered 401/403 — not just "could". */
  requires_auth?: boolean;
}

// /mcp "test server" result.
export interface McpTestResult {
  name: string;
  ok: boolean;
  tool_count?: number;
  tools?: string[];
  error?: string;
  /** The failure was a 401/403 — the user needs to sign in, not debug config. */
  needs_auth?: boolean;
}

// How a single MCP tool is gated. `allow` runs without prompting, `ask`
// surfaces a permission prompt, `deny` refuses the call outright.
export type McpToolPolicy = "allow" | "ask" | "deny";

export interface McpTool {
  name: string;
  description: string;
}

// /mcp tool-permission view — one server's tool inventory + policy.
export interface McpToolList {
  server: string;
  tools: McpTool[];
  permissions: Record<string, McpToolPolicy>;
  loading?: boolean;
  error?: string;
  /** The listing failed because the server needs a browser sign-in. */
  needs_auth?: boolean;
  /** True when the list came from .mcp.json rather than a live connection. */
  cached?: boolean;
}

// /perms dashboard — live permission state.
export interface PermsState {
  mode: string; // "default" | "edits" | "plan"
  auto_approve: boolean;
  allowed_tools: string[];
  mutating_tools: string[];
  file_edit_tools: string[];
}

// /tree popup — one node in the branchable session tree.
export interface TreeNode {
  id: string;
  parent_id: string | null;
  depth: number;
  role: "user" | "assistant" | null;
  type: "message" | "compaction";
  preview: string;
  active: boolean; // on the current active path
  leaf: boolean;   // current position
}

// /setup dashboard — one credential/config row. `value` is already masked
// server-side for secrets, so it is safe to render as-is.
export interface EnvKey {
  key: string;
  label: string;
  secret: boolean;
  hint: string;
  set: boolean;
  value: string;
  source: string; // ".env" | "env" | ""
}

export interface EnvState {
  path: string; // .env file the dashboard writes to
  model: string;
  keys: EnvKey[];
}

// Status vocabulary matches deepagents' official async-subagents middleware
// ("running" | "success" | "error" | "cancelled"); "queued"/"paused" are
// Extensions to the standard subagent status vocabulary.
export interface TaskSummary {
  id: string;
  description: string;
  subagent_type: string;
  state: "queued" | "running" | "paused" | "success" | "error" | "cancelled";
  tool_count: number;
  current: string;
  reply_chars: number;
  elapsed: number;
  error: string;
  awaiting_permission: boolean;
  input_tokens?: number;
  output_tokens?: number;
  /** Peek inside the agent: its most recent tool calls (up to 10). */
  recent_tools?: string[];
  /** Tail of the agent's output so far (up to 400 chars). */
  preview?: string;
  /** Chronological log of what the agent did — one line per tool call. */
  activity?: string[];
}

// Bridge → client events.
export type BridgeEvent =
  | { type: "ready"; model: string; backend: string; cwd: string; mode: string; supports_thinking: boolean; supports_vision: boolean; tools: ToolInfo[] }
  | { type: "text_delta"; content: string }
  | { type: "thinking_delta"; content: string }
  | { type: "tool_start"; tool_id: string; name: string; arguments: Record<string, unknown>; hidden?: boolean }
  | { type: "tool_result"; tool_id: string; output: string; is_error: boolean }
  | { type: "todos"; todos: Todo[] }
  | { type: "usage"; input_tokens: number; output_tokens: number; cache_read_tokens: number; cache_write_tokens: number }
  | { type: "permission_request"; items: PermissionItem[] }
  | { type: "done"; usage: Usage | null }
  | { type: "turn_end"; reply: string }
  | { type: "model_changed"; model: string }
  | { type: "mode_changed"; mode: string }
  | { type: "cleared" }
  | { type: "info"; message: string }
  | { type: "error"; message: string }
  | { type: "task_update"; task: TaskSummary }
  | { type: "task_done"; task: TaskSummary; result: string }
  | { type: "task_list"; tasks: TaskSummary[] }
  | { type: "task_permission"; task_id: string; items: PermissionItem[] }
  | { type: "models"; models: string[] }
  | { type: "ask_user"; question: string; options: string[] }
  | { type: "sessions"; sessions: SessionInfo[] }
  | { type: "resumed"; session_id: string; messages: Array<{ role: string; content: string }> }
  | { type: "mcp_servers"; servers: McpServer[] }
  | { type: "mcp_status"; name: string; ok: boolean; tool_count?: number; tools?: string[]; error?: string; needs_auth?: boolean }
  | { type: "mcp_tools"; server: string; tools: McpTool[]; permissions: Record<string, McpToolPolicy>; cached?: boolean; error?: string; needs_auth?: boolean }
  | { type: "perms_state"; mode: string; auto_approve: boolean; allowed_tools: string[]; mutating_tools: string[]; file_edit_tools: string[] }
  | { type: "env_state"; path: string; model: string; keys: EnvKey[] }
  | { type: "tree"; session_id: string; message_count: number; leaf: string | null; nodes: TreeNode[] }

export interface SessionInfo {
  id: string;
  started: string;
  messages: number;
  preview: string;
}

export interface Todo {
  content: string;
  status: "pending" | "in_progress" | "completed";
}

// Client → bridge commands.
export type BridgeCommand =
  | { type: "user"; text: string }
  | { type: "interrupt" }
  | { type: "decisions"; outcomes: Array<"allow" | "always" | "deny"> }
  | { type: "set_mode"; mode: string }
  | { type: "switch_model"; model: string }
  | { type: "compact" }
  | { type: "clear" }
  | { type: "describe" }
  | { type: "task_stop"; task_id: string }
  | { type: "task_resume"; task_id: string; message?: string }
  | { type: "task_restart"; task_id: string }
  | { type: "task_answer"; task_id: string; outcomes: Array<"allow" | "always" | "deny"> }
  | { type: "task_list" }
  // /mcp dashboard
  | { type: "mcp_list" }
  | { type: "mcp_add"; name: string; transport: string; url?: string; command?: string; args?: string[]; headers?: Record<string, string>; env?: Record<string, string>; default_policy?: McpToolPolicy }
  | { type: "mcp_remove"; name: string }
  | { type: "mcp_test"; name: string }
  // List a server's tools. `refresh` forces a live connect instead of using
  // the tool names cached in .mcp.json.
  | { type: "mcp_tools"; name: string; refresh?: boolean }
  // Set one tool's policy. `tool: "*"` applies the policy to every tool the
  // server exposes (and becomes the default for tools discovered later).
  | { type: "mcp_set_tool_perm"; server: string; tool: string; policy: McpToolPolicy }
  // Run the browser OAuth flow for a cloud server (Notion, Linear, Sentry, …).
  | { type: "mcp_authorize"; name: string }
  | { type: "mcp_sign_out"; name: string }
  // /perms dashboard
  | { type: "perms_list" }
  | { type: "perms_clear_allowed" }
  // /setup dashboard
  | { type: "env_list" }
  | { type: "env_set"; key: string; value: string }
  | { type: "ask_answer"; value: string }
  | { type: "tree"; node?: string }
  | { type: "resume"; session_id?: string }
  | { type: "skill"; action: "list" | "create"; brief?: string }
  | { type: "quit" };

// ── Transcript items (what the UI renders) ──────────────────────────────

export type Item =
  | { kind: "user"; id: string; text: string }
  | { kind: "assistant"; id: string; text: string }
  | { kind: "tool"; id: string; toolId: string; name: string; args: Record<string, unknown>; output?: string; isError?: boolean; running: boolean }
  | { kind: "todos"; id: string; todos: Todo[] }
  | { kind: "info"; id: string; text: string }
  | { kind: "error"; id: string; text: string };
