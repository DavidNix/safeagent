# Initial Findings: Agent Harnesses

Date: 2026-04-27

## Scope

This document summarizes an initial deep dive into six agent harnesses:

| Repository | Focus |
| --- | --- |
| `openclaw` | Embedded PI agent runner, harness lifecycle, tool policy, context engine |
| `claude-code` | Query loop, tool orchestration, prompt caching, context recovery |
| `pi-mono` | Coding agent only: `packages/coding-agent`, `packages/agent`, `packages/ai` |
| `hermes-agent` | `AIAgent`, transports, tool execution, context compression, memory |
| `zeroclaw` | Rust runtime loop, provider traits, dispatcher, policy, receipts |
| `opencode` | Session prompt loop, AI SDK streaming, processor, tools, permissions |

The goal is not to clone any existing coding agent. The goal is to identify production-grade patterns for an agentic loop, provider APIs, tool calling, prompt construction, prompt caching, context management, memory boundaries, and safety controls. The target implementation is a Go harness in this repository, starting with `safeagent.Lite`, which must not expose bash or shell command execution.

## Executive Summary

Production LLM harnesses converge on the same shape: a session-scoped loop that calls a provider, streams or receives an assistant message, extracts tool calls, executes validated tools, appends tool results, and repeats until the model returns a final answer or the loop reaches a hard budget.

The most important commonality is not the specific tool list. It is the invariant that every provider response, tool call, tool result, context mutation, permission decision, and final answer is represented in durable, replayable state.

The highest-value practices for `safeagent.Lite` are:

1. Build a small, testable loop first.
2. Normalize provider streaming into one internal event model.
3. Preserve tool-call IDs and pair every tool call with exactly one tool result.
4. Treat tool validation, authorization, truncation, and redaction as runtime controls, not prompt instructions.
5. Keep system prompts stable and deterministic to support provider prompt caching later.
6. Enforce workspace and file safety in tool implementations.
7. Start without bash, external process execution, browser automation, MCP, subagents, or long-term memory.
8. Use fake providers and deterministic event streams for loop tests.

## Common Agent Loop

All studied harnesses use a variation of this loop:

```text
initialize session
load stable system prompt
load active tool schemas
append user message

for step in 1..max_steps:
    assemble provider request from system, messages, tools, model options
    call model, usually streaming
    normalize assistant text, reasoning, usage, and tool calls
    persist assistant message

    if there are no tool calls:
        return final assistant answer

    validate tool names and arguments
    authorize each tool call at runtime
    execute safe tools, sequentially or in safe parallel batches
    truncate and redact tool output
    append tool result messages with original tool_call_id
    continue

if max_steps exhausted:
    ask for final summary without tools or return bounded exhaustion error
```

The loop should be explicit. Avoid hidden recursion, provider-specific branches inside the core loop, and implicit state in UI callbacks.

## At A Glance

| Repo | Loop Entry | Provider Boundary | Tool Boundary | Notable Harness Pattern |
| --- | --- | --- | --- | --- |
| `openclaw` | `src/agents/pi-embedded-runner/run.ts`, `run/attempt.ts` | Transport-aware stream selection in `provider-transport-stream.ts` and `stream-resolution.ts` | `pi-tool-definition-adapter.ts`, `pi-tools.ts` | Lifecycle phases: prepare, start, send, resolve, cleanup |
| `claude-code` | `query.ts`, `QueryEngine.ts` | `services/api/claude.ts`, `services/api/client.ts` | `services/tools/toolExecution.ts`, `StreamingToolExecutor.ts` | Layered context recovery and strict tool-result pairing |
| `pi-mono` | `packages/agent/src/agent-loop.ts` | `packages/ai/src/stream.ts`, provider files | `packages/agent/src/agent-loop.ts`, coding-agent tools | Small reusable loop, provider-normalized events |
| `hermes-agent` | `run_agent.py` `AIAgent.run_conversation()` | `agent/transports/*`, dispatch in `run_agent.py` | `model_tools.py`, tool registry, `_execute_tool_calls()` | Defensive recovery for malformed, empty, truncated, and oversized responses |
| `zeroclaw` | `crates/zeroclaw-runtime/src/agent/loop_.rs` | `crates/zeroclaw-api/src/provider.rs` | dispatcher and `tool_execution.rs` | Rust traits, security policy, tool receipts |
| `opencode` | `packages/opencode/src/session/prompt.ts` `runLoop` | `session/llm.ts`, provider transforms | `session/processor.ts`, tool registry | Durable message parts and per-session run serialization |

## Repository Findings

### OpenClaw

Key files:

| Area | Files |
| --- | --- |
| CLI and gateway entry | `src/cli/program/register.agent.ts`, `src/gateway/server-methods/agent.ts`, `src/gateway/server-methods/chat.ts` |
| Main runner | `src/agents/pi-embedded-runner/run.ts`, `src/agents/pi-embedded-runner/run/attempt.ts` |
| Harness lifecycle | `src/agents/harness/v2.ts`, `src/agents/harness/builtin-pi.ts` |
| Provider streams | `src/agents/provider-stream.ts`, `src/agents/provider-transport-stream.ts`, `src/agents/pi-embedded-runner/stream-resolution.ts` |
| Tool adapter | `src/agents/pi-tool-definition-adapter.ts`, `src/agents/pi-tools.ts` |
| System prompt | `src/agents/system-prompt.ts`, `src/agents/pi-embedded-runner/system-prompt.ts` |
| Prompt cache | `src/agents/pi-embedded-runner/prompt-cache-observability.ts` |
| Context engine | `src/context-engine/types.ts` |
| Safety | `src/agents/tool-policy.ts`, `src/agents/tool-policy-pipeline.ts`, sandbox-aware wrappers in `pi-tools.ts` |

OpenClaw has the most explicit harness lifecycle. The harness phase model is useful for Go because it separates preparation, model send, result resolution, and cleanup. It also makes diagnostics precise. A failed `prepare` is different from a failed model stream, a failed tool, or a failed cleanup.

The runner also separates provider stream selection from the loop. Provider adapters can choose OpenAI Responses, OpenAI WebSocket, Anthropic, Google, Azure, or other transports without changing the loop.

Tool execution runs through one adapter. The adapter handles hooks, validation, result normalization, pending client-side tools, thrown errors, and tool result sanitization. This is the right direction for `safeagent.Lite`: the loop should never call arbitrary tool functions directly.

Prompt construction is deterministic. Context files and sections are sorted in fixed order, and prompt-cache observability tracks prompt and tool digests. This matters because accidental ordering changes can destroy provider cache hit rates.

Useful lessons:

1. Use lifecycle phases and per-phase diagnostics.
2. Keep provider transport selection outside the loop.
3. Funnel all tools through one adapter.
4. Track stable prompt and tool digests.
5. Treat transcript repair and session write locks as safety features.
6. Keep context assembly and compaction behind a context engine contract.

### Claude Code

Key files:

| Area | Files |
| --- | --- |
| CLI and REPL | `entrypoints/cli.tsx`, `replLauncher.tsx`, `screens/REPL.tsx` |
| Headless harness | `QueryEngine.ts` |
| Core loop | `query.ts` |
| Provider API | `services/api/client.ts`, `services/api/claude.ts`, `services/api/withRetry.ts` |
| Tools | `tools.ts`, `services/tools/toolExecution.ts`, `services/tools/toolOrchestration.ts`, `services/tools/StreamingToolExecutor.ts` |
| Prompts | `constants/prompts.ts`, `utils/queryContext.ts`, `utils/api.ts` |
| Prompt cache | `services/api/promptCacheBreakDetection.ts`, cache code in `services/api/claude.ts` |
| Context | `utils/context.ts`, `utils/toolResultStorage.ts` |
| Memory | `memdir/memdir.ts`, memory attachment code in `utils/attachments.ts` |

Claude Code treats the loop as a resumable state machine. `query.ts` tracks messages, tool context, compaction state, recovery counts, turn count, and transition reason. This statefulness is more important than the UI.

The loop does not trust `stop_reason` alone. It records streamed `tool_use` blocks and decides whether to continue based on actual tool calls. This is a valuable guardrail because provider stop reasons vary and can be wrong or incomplete.

Tool execution is highly defensive. Inputs are schema-validated, permission failures return structured `tool_result` errors, and streaming tool execution can begin before the full assistant response finishes while preserving output order and concurrency rules.

Prompt caching is treated as a first-class invariant. The system prompt is split into stable and dynamic sections. Cache control flags are latched to avoid mid-session churn. Tool schemas are frozen or normalized to prevent cache breaks caused by irrelevant schema drift.

Useful lessons:

1. Make loop continuation depend on parsed tool calls, not only provider stop reasons.
2. Return tool validation and permission failures to the model as tool errors.
3. Keep cache-sensitive prompt sections stable and ordered.
4. Add layered context pressure handling: tool-result budgeting, microcompaction, autocompaction, reactive compaction, and output-token recovery.
5. Add stream watchdogs and cleanup paths for hung provider streams.
6. Test with fake providers and deterministic streams.

### Pi Mono Coding Agent

Key files:

| Area | Files |
| --- | --- |
| CLI | `packages/coding-agent/src/cli.ts`, `packages/coding-agent/src/main.ts` |
| SDK/session factory | `packages/coding-agent/src/core/sdk.ts` |
| Core loop | `packages/agent/src/agent-loop.ts`, `packages/agent/src/agent.ts` |
| Provider abstraction | `packages/ai/src/stream.ts`, `packages/ai/src/types.ts`, provider files under `packages/ai/src/providers` |
| Tools | `packages/coding-agent/src/core/tools/index.ts`, `read.ts`, `write.ts`, `edit.ts`, `bash.ts` |
| Prompt | `packages/coding-agent/src/core/system-prompt.ts`, `resource-loader.ts`, `agent-session.ts` |
| Context | `packages/coding-agent/src/core/compaction/compaction.ts`, `session-manager.ts` |
| Tests | `packages/agent/test/agent-loop.test.ts`, `packages/agent/test/agent.test.ts` |

Pi Mono is the cleanest small-loop reference. The reusable `agentLoop()` has two nested concepts: continue while there are tool calls or steering messages, then continue for queued follow-up messages. The model stream boundary, tool execution boundary, and event emission are compact and testable.

Provider handling lives in `packages/ai`, which normalizes Anthropic, OpenAI Responses, and other provider streams into a shared event protocol. This is close to what a first Go version should do.

The tool set is conventional: `read`, `bash`, `edit`, `write`, `grep`, `find`, and `ls`. The default SDK enables a smaller active set. The file mutation queue serializes writes per file, which avoids race conditions during parallel tool execution.

Prompt caching support is simpler than Claude Code or OpenClaw but still important. Cache retention and session IDs are part of provider options. Anthropic maps retention to ephemeral cache control; OpenAI Responses gets `prompt_cache_key` and retention.

Useful lessons:

1. Keep the loop small and reusable.
2. Normalize providers into one event stream.
3. Pass cancellation through provider calls, hooks, and tools.
4. Serialize file mutations by path.
5. Store compactions as explicit session entries, not destructive rewrites.
6. Use deterministic fake providers for tests.

### Hermes Agent

Key files:

| Area | Files |
| --- | --- |
| Entry | `pyproject.toml`, `run_agent.py` |
| Main harness | `run_agent.py` `AIAgent`, `AIAgent.run_conversation()` |
| Provider transports | `agent/transports/base.py`, `agent/transports/chat_completions.py`, `agent/transports/anthropic.py` |
| Tools | `tools/registry.py`, `model_tools.py`, tool execution code in `run_agent.py` |
| Prompt | `_build_system_prompt()` in `run_agent.py`, `agent/prompt_builder.py` |
| Prompt cache | `agent/prompt_caching.py`, cache policy in `run_agent.py` |
| Context | `agent/context_compressor.py`, context recovery in `run_agent.py` |
| Safety | `tools/approval.py`, terminal safety in `tools/terminal_tool.py` |
| Memory | `agent/memory_manager.py`, memory setup in `run_agent.py` |

Hermes is a defensive production harness. It supports several transport families and normalizes provider responses into a common shape. The main loop is bounded by `max_iterations` and a shared iteration budget, then performs one final toolless summary request when exhausted.

The tool path validates tool names, repairs JSON arguments where possible, and injects tool errors so the model can self-correct. Tools can run concurrently only when considered safe and independent. Result order is preserved.

Prompt caching is careful. The system prompt is built once per session and reused. Ephemeral prompt context is injected at API-call time rather than changing the stored system prompt. Whitespace and tool-call JSON are normalized for cache stability.

Hermes also demonstrates strong recovery behavior: empty response recovery, reasoning-only recovery, output-cap reduction, context-overflow detection, compression, payload cleanup for invalid Unicode, and fallback between streaming and non-streaming paths.

Useful lessons:

1. Build the system prompt once per session when possible.
2. Inject ephemeral context outside the stable prompt prefix.
3. Add one explicit toolless final-summary path after loop exhaustion.
4. Treat provider errors as recovery signals, not just fatal errors.
5. Run only safe independent tools concurrently.
6. Make hard safety blocks impossible to bypass with a permissive mode.

### Zeroclaw

Key files:

| Area | Files |
| --- | --- |
| CLI | `src/main.rs` |
| Main loop | `crates/zeroclaw-runtime/src/agent/loop_.rs` |
| Library agent | `crates/zeroclaw-runtime/src/agent/agent.rs` |
| Provider traits | `crates/zeroclaw-api/src/provider.rs` |
| Providers | `crates/zeroclaw-providers/src/*` |
| Tool trait | `crates/zeroclaw-api/src/tool.rs` |
| Dispatchers | `crates/zeroclaw-runtime/src/agent/dispatcher.rs` |
| Tool execution | `crates/zeroclaw-runtime/src/agent/tool_execution.rs` |
| Tools | `crates/zeroclaw-runtime/src/tools/mod.rs` |
| Prompt | `crates/zeroclaw-runtime/src/agent/system_prompt.rs`, `prompt.rs` |
| Context | `history.rs`, `history_pruner.rs`, `context_compressor.rs` |
| Safety | `crates/zeroclaw-config/src/policy.rs`, `approval/mod.rs`, `security/prompt_guard.rs` |
| Receipts | `crates/zeroclaw-runtime/src/agent/tool_receipts.rs` |

Zeroclaw is the best source for Rust-style interface design and explicit security policy. It has provider traits with capability flags, a tool trait, native and XML dispatchers, and separate tool execution helpers.

The dispatcher abstraction is important. Native provider tool calls and prompt-guided XML or JSON calls are different protocols, but the core loop should see the same `ToolCall` and `ToolResult` shape. A Go harness should copy this boundary.

Zeroclaw also has the clearest policy model: autonomy level, workspace boundary, command allowlist, forbidden paths, rate limits, and risk gates. Even though `safeagent.Lite` will not have shell tools, the same policy idea applies to file reads and writes.

Tool receipts are notable. An HMAC receipt over tool name, args, result, and timestamp can help detect tampering or replay when tool results are later audited. This is not required for the first Lite version, but it fits the safety direction.

Useful lessons:

1. Use explicit provider and tool interfaces with capability flags.
2. Separate native tool calling from fallback text parsing behind a dispatcher.
3. Preserve provider-native `tool_call_id` exactly.
4. Trim assistant/tool groups atomically.
5. Make non-interactive approval safe by default: deny when approval is required but unavailable.
6. Redact secrets from tool outputs before logging or feeding them back.

### Opencode

Key files:

| Area | Files |
| --- | --- |
| HTTP entry | `packages/opencode/src/server/routes/instance/session.ts`, `httpapi/session.ts` |
| Main loop | `packages/opencode/src/session/prompt.ts` |
| Run state | `packages/opencode/src/session/run-state.ts` |
| Provider call | `packages/opencode/src/session/llm.ts` |
| Provider transforms | `packages/opencode/src/provider/transform.ts` |
| Processor | `packages/opencode/src/session/processor.ts` |
| Message conversion | `packages/opencode/src/session/message-v2.ts` |
| Tools | `packages/opencode/src/tool/registry.ts`, `tool.ts`, `read.ts`, `write.ts`, `bash.ts`, `truncate.ts` |
| Prompt | `packages/opencode/src/session/system.ts`, `session/prompt/default.txt` |
| Context | `packages/opencode/src/session/overflow.ts`, `session/compaction.ts` |
| Safety | `packages/opencode/src/permission/index.ts`, `permission/evaluate.ts`, `tool/external-directory.ts` |

Opencode is centered on durable message parts. Text, tool input, tool result, tool errors, reasoning, and step boundaries are persisted as structured parts. This is more robust than treating a turn as one opaque blob.

The run state serializes work per session and exposes cancellation. That is a critical production feature because concurrent runs against one session can corrupt history or tool-result pairing.

Tools return a richer result shape: title, metadata, output, and attachments. The output can be truncated and saved to disk while returning a concise model-readable reference.

Prompt caching marks stable system messages and recent non-system messages. Provider transforms handle Anthropic, OpenRouter, Bedrock, OpenAI-compatible, Copilot, and other cache metadata differences.

Useful lessons:

1. Store messages as structured parts, not only provider-shaped messages.
2. Serialize one active run per session.
3. Keep provider transforms in one boundary layer.
4. Save large tool outputs outside context and return references.
5. Test tool schema wire shapes with snapshots.
6. Capture pre-tool snapshots where mutations can race with streamed events.

## Common Toolsets

Across systems, the common coding-agent tools are:

| Tool | Common Purpose | Lite Recommendation |
| --- | --- | --- |
| `read` | Read files with offsets, limits, line numbers, image support in some agents | Include in v1, text only, bounded |
| `list` or `ls` | Directory listing | Include in v1 as `list_dir` |
| `glob` or `find` | File discovery by pattern | Include in v1, workspace-rooted |
| `grep` or `search` | Content search with regex and include filters | Include in v1, bounded results |
| `write` | Create or replace files | Include only behind mutation policy, or defer |
| `edit` | Exact replacement edits | Optional; patch may be enough |
| `apply_patch` | Structured multi-file patch | Prefer for mutations in Lite |
| `bash` or `shell` | Execute commands | Exclude from Lite |
| `web_fetch` or `fetch` | Retrieve URLs | Defer unless needed |
| `todo` | Maintain session task state | Optional, useful but not core |
| `question` or `clarify` | Ask user for missing information | Useful if the host UI supports it |
| `task` or subagent | Delegate work | Exclude from Lite |
| `memory` | Store or retrieve long-term context | Exclude from Lite v1, leave interface seam |

For `safeagent.Lite`, the recommended first toolset is:

| Tool | Behavior |
| --- | --- |
| `list_dir` | List one workspace directory, sorted, with type and size metadata |
| `read_file` | Read UTF-8 text with `offset`, `limit`, max bytes, and line numbers |
| `glob` | Match workspace-relative file paths with max result count |
| `grep` | Regex search over workspace files with include filters and max match count |
| `apply_patch` | Apply a structured patch only inside the workspace, gated by mutation policy |
| `write_file` | Optional; create new files only, or require an explicit overwrite flag and policy approval |

The first version can be read-only if the immediate goal is introspection. If mutation is included, prefer `apply_patch` over free-form `write_file` because patches are easier to audit, bound, and reject.

## Tool Calling Best Practices

Tool calling should use a canonical internal shape independent of provider APIs:

```go
type ToolCall struct {
    ID        string
    Name      string
    Arguments json.RawMessage
}

type ToolResult struct {
    ToolCallID string
    Name       string
    Content    string
    IsError    bool
    Metadata   map[string]any
}
```

The tool layer should enforce these rules:

1. Unknown tools return a tool result error to the model.
2. Invalid JSON returns a tool result error to the model.
3. Schema validation failures return a tool result error to the model.
4. Authorization failures return a tool result error to the model unless policy says to abort the run.
5. Tool panics or internal errors are caught and returned as bounded tool result errors.
6. Every tool call gets exactly one result with the same `tool_call_id`.
7. Tool output is truncated before storage, logging, or model reinjection.
8. Tool output is redacted for common secrets before logging or model reinjection.
9. Parallel execution is allowed only for read-only tools or tools with non-overlapping mutation paths.
10. Result messages are appended in original call order even if execution is parallel.

For Lite, start with sequential execution. Add safe parallelism later for `read_file`, `glob`, and `grep`.

## Provider API Design

The Go harness should hide provider differences behind a small interface:

```go
type Provider interface {
    Stream(ctx context.Context, req ModelRequest) (EventStream, error)
    Capabilities() ProviderCapabilities
}

type ProviderCapabilities struct {
    NativeTools     bool
    Streaming       bool
    PromptCaching   bool
    Reasoning       bool
    MaxInputTokens  int
    MaxOutputTokens int
}
```

The internal event model should support:

| Event | Purpose |
| --- | --- |
| `message_start` | Start assistant turn |
| `text_delta` | Stream assistant text |
| `reasoning_delta` | Optional reasoning or thinking stream if provider exposes it |
| `tool_call_start` | Begin tool call with ID and name |
| `tool_call_delta` | Incremental JSON arguments |
| `tool_call_done` | Final parsed tool call |
| `usage` | Input, output, cache read, cache write tokens |
| `message_done` | Final assistant message and stop reason |
| `error` | Provider or stream error |

Native tool providers should map directly into this event model. Non-native or text-based tool calling should be handled by a dispatcher outside the core loop.

## System Prompt Best Practices

The strongest systems share these prompt habits:

1. Keep a stable base prompt with deterministic section order.
2. Separate stable prompt content from dynamic runtime content.
3. Put tool usage rules near the tool descriptions.
4. Include explicit tool honesty rules: do not claim a tool was used if it was not.
5. Include autonomy rules that encourage action, but do not override safety gates.
6. Include workspace rules, path rules, and verification expectations.
7. Include current date, cwd, platform, and model only in a dynamic section.
8. Keep user, project, and memory context clearly fenced.
9. Do not rely on prompts to enforce security.
10. Track a digest of the final system prompt and tool schema set.

Recommended Lite prompt structure:

```text
1. Identity and role
2. Safety and workspace rules
3. Agent loop and tool-use rules
4. Available tools and when to use them
5. Editing rules if mutation tools are enabled
6. Response style
7. Dynamic runtime context: cwd, date, platform, model
8. Optional project instructions from trusted files
```

Lite should load project instruction files only after path validation and size limits. It should scan or fence these files as untrusted instructions below the harness-level system rules.

## Prompt Caching

Prompt caching is a production optimization, but it affects architecture early.

Common patterns:

1. Stable system prompt prefix.
2. Stable tool schema order.
3. Stable cache-control placement.
4. Session-latched cache options.
5. Last-message or recent-message cache markers where providers support them.
6. Prompt and tool digest logging to diagnose cache breaks.

Lite should not depend on prompt caching in v1, but it should be cache-ready:

1. Sort system sections deterministically.
2. Sort tool schemas by tool name.
3. Avoid random IDs in prompts.
4. Avoid timestamps in the stable system prefix.
5. Keep dynamic context in a separate section.
6. Compute and log `system_prompt_sha256` and `tool_schema_sha256` per request.
7. Preserve provider usage fields for cache read and write tokens.

## Context Management

Context management appears in every mature harness. The common layers are:

1. Preflight token estimate before provider calls.
2. Tool output truncation before appending to context.
3. Aggregate per-turn tool result budget.
4. History pruning that preserves assistant/tool-result groups.
5. Compaction or summarization when near threshold.
6. Reactive recovery after provider context-overflow errors.
7. Final no-tools summary when the loop budget is exhausted.

Lite should start simpler:

1. Hard cap file read output.
2. Hard cap grep output.
3. Hard cap tool result content.
4. Keep full outputs outside model context only if needed later.
5. Estimate tokens with a conservative heuristic such as 4 chars per token plus message overhead.
6. Abort with a clear error if the assembled request is too large.
7. Add summarizing compaction after the core loop is stable.

The key invariant is group integrity. Do not trim an assistant message that contains tool calls without also trimming its corresponding tool results.

## Safety And Security

`safeagent.Lite` should implement safety in code, not only in the prompt.

Required v1 controls:

| Control | Recommendation |
| --- | --- |
| No shell | Do not expose bash, process execution, package installs, network shells, or command wrappers |
| Workspace root | All file tools must resolve paths under a configured workspace root |
| Symlink policy | Resolve symlinks and reject paths that escape the workspace |
| Secret paths | Deny or require approval for `.env`, credential files, private keys, SSH keys, tokens, and cloud config paths |
| Git internals | Deny writes under `.git` |
| Size limits | Bound file read, grep, glob, patch, and write sizes |
| Binary files | Deny binary reads and writes in Lite unless explicitly supported |
| Mutation policy | Gate `apply_patch` and `write_file`; default can be deny, ask, or allow only for explicit user task |
| Atomic writes | Write through temp file plus rename where practical |
| Conflict checks | For patches, fail if expected context does not match current file contents |
| Redaction | Scrub common secret patterns from logs and tool outputs |
| Audit log | Record model request metadata, tool calls, policy decisions, and result sizes |
| Cancellation | Pass `context.Context` through provider and tool execution |

If Lite runs non-interactively and a tool requires approval, deny by default. Never auto-approve because there is no UI.

## Agent Memory

Memory is not required for the first iteration, but the harness should leave clean seams for it.

The studied systems show three memory categories:

| Memory Type | Description | Examples |
| --- | --- | --- |
| Session memory | Durable transcript, compactions, todos, permissions, branch state | Pi Mono, Opencode |
| Project context | `AGENTS.md`, `CLAUDE.md`, instruction files, workspace docs | OpenClaw, Claude Code, Pi Mono, Zeroclaw, Opencode |
| Long-term semantic memory | User profile, external provider, retrieved snippets, memory tools | Hermes, Zeroclaw, OpenClaw, Claude Code |

Recommendations for Lite:

1. Implement session memory first as an append-only transcript.
2. Do not implement long-term semantic memory in v1.
3. Define a future `MemoryProvider` interface but keep it unused by default.
4. Inject retrieved memory into ephemeral user-message context, not the stable system prompt.
5. Fence memory as informational context, not instructions.
6. Bound memory snippets by count, bytes, and age.
7. Never store secrets or full tool outputs in long-term memory by default.
8. Add tests that memory cannot override system safety rules before enabling it.

Possible future interface:

```go
type MemoryProvider interface {
    Retrieve(ctx context.Context, query string, limit int) ([]MemoryItem, error)
    Commit(ctx context.Context, event MemoryEvent) error
}
```

## Recommended `safeagent.Lite` Design

Start with a package that has a small public API and explicit dependency injection.

```go
package safeagent

type Lite struct {
    Provider Provider
    Tools    ToolRegistry
    Policy   PolicyEngine
    Store    SessionStore
    Prompt   PromptBuilder
    Context  ContextManager
    Observe  Observer
    Config   LiteConfig
}

func (a *Lite) Run(ctx context.Context, req RunRequest) (*RunResult, error)
```

Core interfaces:

```go
type Tool interface {
    Name() string
    Description() string
    InputSchema() json.RawMessage
    Execute(ctx context.Context, call ToolCall, env ToolEnv) (ToolResult, error)
}

type PolicyEngine interface {
    AuthorizeTool(ctx context.Context, call ToolCall, env ToolEnv) PolicyDecision
}

type SessionStore interface {
    Append(ctx context.Context, sessionID string, entries ...Entry) error
    Load(ctx context.Context, sessionID string) ([]Entry, error)
}
```

Recommended first loop:

```go
func (a *Lite) Run(ctx context.Context, req RunRequest) (*RunResult, error) {
    session, err := a.Store.Load(ctx, req.SessionID)
    if err != nil {
        return nil, err
    }

    userEntry := UserEntry(req.UserMessage)
    if err := a.Store.Append(ctx, req.SessionID, userEntry); err != nil {
        return nil, err
    }
    session = append(session, userEntry)

    for step := 0; step < a.Config.MaxSteps; step++ {
        modelReq, err := a.Context.Assemble(ctx, session, a.Prompt, a.Tools)
        if err != nil {
            return nil, err
        }

        assistant, toolCalls, usage, err := a.callAndCollect(ctx, modelReq)
        if err != nil {
            return nil, err
        }

        assistantEntry := AssistantEntry(assistant, usage)
        if err := a.Store.Append(ctx, req.SessionID, assistantEntry); err != nil {
            return nil, err
        }
        session = append(session, assistantEntry)

        if len(toolCalls) == 0 {
            return &RunResult{Text: assistant.Text, Usage: usage}, nil
        }

        results := a.executeToolCalls(ctx, toolCalls)
        resultEntries := ToolResultEntries(results)
        if err := a.Store.Append(ctx, req.SessionID, resultEntries...); err != nil {
            return nil, err
        }
        session = append(session, resultEntries...)
    }

    return nil, ErrStepLimitExceeded
}
```

The production version should add final no-tools summary, compaction, retries, and provider fallback. Do not add those until the tool-call invariant tests pass.

## Build Order

Recommended implementation order:

1. Define internal message, event, tool call, and tool result types.
2. Implement append-only session store in memory or JSONL.
3. Implement `read_file`, `list_dir`, `glob`, and `grep` with workspace guards.
4. Implement fake provider and deterministic streaming tests.
5. Implement provider adapter for one model API with native tools.
6. Implement the Lite loop with max step limit and cancellation.
7. Add tool schema validation and structured tool errors.
8. Add `apply_patch` behind a mutation policy.
9. Add prompt builder with stable and dynamic sections.
10. Add prompt and tool schema digests.
11. Add context size estimation and hard output limits.
12. Add audit logging and redaction.
13. Add compaction only after the base loop is stable.

## Test Plan

The test suite should focus on harness invariants, not model quality.

Minimum tests:

| Test | Purpose |
| --- | --- |
| Final answer without tools | Loop exits cleanly when no tool calls exist |
| Single tool call | Tool result is appended and loop continues |
| Multiple tool calls | Every call gets one result with the same ID |
| Unknown tool | Unknown call returns structured tool error |
| Invalid JSON args | Validation error returns structured tool error |
| Permission denied | Denial returns tool error or aborts according to policy |
| Step limit | Loop stops with bounded error or final no-tools summary |
| Cancellation | Context cancellation stops provider and tools |
| Path escape | `../`, symlink escape, and absolute external paths are denied |
| Secret path | `.env` and key files are denied by default |
| Prompt digest stability | Same inputs produce same prompt and tool digests |
| Tool-result pairing trim | Context trimming never leaves orphan tool messages |
| Output truncation | Large tool outputs are bounded and marked truncated |
| Fake stream fragments | Partial tool JSON deltas assemble correctly |

Avoid paid provider APIs in harness tests. Use fake event streams.

## Open Questions

1. Should `safeagent.Lite` be read-only first, or should it include `apply_patch` from the first version?
2. Should Lite store sessions in JSONL files, SQLite, or memory first?
3. Which provider should be implemented first: Anthropic Messages, OpenAI Responses, or a provider-agnostic adapter over an existing Go SDK?
4. Should mutation approval be interactive, config-based, or disabled in non-interactive mode?
5. Should Lite include project instruction files such as `AGENTS.md` in v1, or only explicit user-provided context?

## Initial Recommendation

Implement `safeagent.Lite` as a small, no-bash, workspace-bound harness with a provider-normalized streaming loop, append-only session state, deterministic prompt construction, and a read-mostly toolset.

The first useful version should include `list_dir`, `read_file`, `glob`, `grep`, and optionally `apply_patch` behind a mutation policy. It should not include bash, subagents, MCP, web automation, browser tools, plugin systems, or long-term memory.

This gives the repository a safe foundation for experimentation while preserving the core production invariant: model output, tool calls, tool results, context updates, and policy decisions remain explicit, testable, bounded, and auditable.
