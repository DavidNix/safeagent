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

## Subagents And Delegation

Subagents are model-visible delegation: the parent model calls a tool that starts another agent loop with its own prompt, session state, tool permissions, and result channel. The mature systems treat this as a second run, not as a special provider feature.

The common delegation flow is:

```text
parent model calls delegate/task/subagent tool
validate requested agent type and task prompt
check depth, concurrency, permission, and workspace policy
create or resume child session/run
select child model, system prompt, tools, and permission mode
choose context mode: isolated, inherited, forked, or summarized
run the normal agent loop for the child
persist child transcript and task status
return concise result, error, or resumable task_id to parent
```

Repository patterns:

| Repo | Subagent Pattern | Useful Files |
| --- | --- | --- |
| `openclaw` | `sessions_spawn` starts direct or ACP-backed subagent runs, tracks child sessions, supports isolated and forked context, enforces spawn depth and active-child limits, and exposes list/kill/steer control | `src/agents/tools/sessions-spawn-tool.ts`, `src/agents/subagent-spawn.ts`, `src/agents/subagent-registry.types.ts`, `src/agents/subagent-control.ts` |
| `claude-code` | `AgentTool` delegates to typed agent definitions, supports foreground and background agents, worktree isolation, forked context, teammate/swarm paths, and runs the normal `query()` loop in a side transcript | `tools/AgentTool/AgentTool.tsx`, `tools/AgentTool/runAgent.ts`, `tools/AgentTool/forkSubagent.ts`, `tools/AgentTool/loadAgentsDir.ts`, `utils/swarm/spawnInProcess.ts` |
| `pi-mono` | Core coding agent intentionally skips subagents; the example extension registers a `subagent` tool that spawns separate `pi` processes in single, parallel, or chain mode | `packages/coding-agent/README.md`, `packages/coding-agent/examples/extensions/subagent/index.ts` |
| `hermes-agent` | `delegate_task` creates child `AIAgent` instances with isolated context, restricted or inherited toolsets, leaf/orchestrator roles, depth controls, batch parallelism, and TUI events | `tools/delegate_tool.py`, `toolsets.py`, `tui_gateway/server.py` |
| `zeroclaw` | `DelegateTool` delegates to configured agents in sync, background, parallel, or agentic tool-loop mode; `SwarmTool` orchestrates multiple configured agents | `crates/zeroclaw-runtime/src/tools/delegate.rs`, `crates/zeroclaw-tools/src/swarm.rs`, `crates/zeroclaw-config/src/schema.rs` |
| `opencode` | `TaskTool` creates or resumes child sessions for agents configured with `mode: "subagent"`, returns a resumable `task_id`, and applies per-subagent permission rules | `packages/opencode/src/tool/task.ts`, `packages/opencode/src/tool/registry.ts`, `packages/opencode/src/session/prompt.ts`, `packages/opencode/src/config/agent.ts` |

Best practices for subagents:

1. Treat each subagent as a normal agent run with its own session ID, run ID, transcript, prompt, and tool registry.
2. Pass only the minimum context the child needs. Prefer isolated or summarized context over full parent transcript inheritance.
3. Enforce max depth and max active children in code.
4. Make child toolsets narrower than parent toolsets by default.
5. Require explicit permission for workspace mutation, network access, shell access, or high-risk tools in child runs.
6. Return a concise child summary to the parent and keep the full child transcript in durable storage.
7. Make background subagents resumable through a `task_id` or run ID.
8. Propagate cancellation from parent to child.
9. Record parent-child relationships in telemetry and audit logs.
10. Avoid letting a child subagent silently escalate model, tool, or permission settings.

Recommended Lite stance:

1. Exclude subagents from the first version.
2. Keep a future seam for `AgentInvoker`, but do not expose it as a model tool yet.
3. Do not add background subagents until the single-agent loop, cancellation, session storage, and policy system are stable.
4. If delegation is added later, implement it as a child `Lite.Run` with a separate session and a narrower tool registry.

Possible future interface:

```go
type AgentInvoker interface {
    InvokeAgent(ctx context.Context, req AgentInvokeRequest) (*AgentInvokeResult, error)
    CancelAgent(ctx context.Context, runID string) error
}

type AgentInvokeRequest struct {
    ParentSessionID string
    AgentType       string
    Task            string
    ContextMode     string
    ToolAllowlist   []string
    MaxSteps        int
}
```

## Task Spawning And Background Runs

Task handling appears in two forms. Model-visible task tools ask another agent or worker to do work. Internal task systems serialize runs, queue user follow-ups, drain delivery queues, cancel streams, and track background state. The second form is needed even when Lite has no subagents.

Common internal task lifecycle:

```text
create run or queue item
mark queued or running
attach abort controller or cancellation token
serialize by session or resource key
execute model/tool work
record progress events
persist final status: succeeded, failed, cancelled, timed out
notify session or caller
clean up leases, temp files, and background handles
```

Repository patterns:

| Repo | Task/Background Pattern | Useful Files |
| --- | --- | --- |
| `openclaw` | Detached task registry tracks runtimes such as `subagent`, `acp`, `cli`, and `cron`; task executor records queued/running/final state; delivery queue recovery uses retry and in-progress guards | `src/tasks/task-registry.types.ts`, `src/tasks/task-executor.ts`, `src/tasks/detached-task-runtime.ts`, `src/tasks/task-registry.ts`, `src/infra/session-delivery-queue-recovery.ts` |
| `claude-code` | Background agents register `LocalAgentTask` state with `AbortController`; command queue uses priorities; queue processor drains when no query is active; task notifications re-enter the main loop | `tasks/LocalAgentTask/LocalAgentTask.tsx`, `utils/messageQueueManager.ts`, `utils/queueProcessor.ts`, `hooks/useQueueProcessor.ts` |
| `pi-mono` | `AgentSession.prompt()` serializes active streaming; follow-up and steering messages queue instead of starting concurrent turns; file mutation queue serializes writes per real path | `packages/coding-agent/src/core/agent-session.ts`, `packages/coding-agent/src/core/tools/file-mutation-queue.ts`, `packages/coding-agent/src/core/agent-session-runtime.ts` |
| `hermes-agent` | Gateway tracks running agents, pending messages, FIFO queues, queued events, and asyncio background tasks; delegate tool runs child agents through thread pools with timeout and interrupt support | `gateway/run.py`, `gateway/stream_consumer.py`, `tools/delegate_tool.py` |
| `zeroclaw` | Session actor queue uses a one-per-session semaphore; delegate tool writes background result files, spawns Tokio tasks, and uses cancellation tokens | `crates/zeroclaw-gateway/src/session_queue.rs`, `crates/zeroclaw-runtime/src/tools/delegate.rs` |
| `opencode` | `SessionRunState` serializes active runs per session; `Runner.ensureRunning` gates concurrent prompts; task tool cancellation calls back into session prompt ops; utilities provide async queues and locks | `packages/opencode/src/session/run-state.ts`, `packages/opencode/src/session/prompt.ts`, `packages/opencode/src/tool/task.ts`, `packages/opencode/src/util/queue.ts`, `packages/opencode/src/util/lock.ts` |

Best practices for task handling:

1. Allow only one active model run per session unless the system has explicit child sessions.
2. Use run IDs and statuses for all long-running work.
3. Separate queued user follow-ups from currently executing model turns.
4. Propagate cancellation through provider streams, tool execution, file locks, and child tasks.
5. Serialize mutations by resource key, usually file path or session ID.
6. Bound queue depth and return a clear busy or queued response.
7. Persist task state before starting work so crashes do not erase in-flight runs.
8. Store large task output out of context and return a concise reference.
9. Notify the model or user when background work finishes, but avoid unbounded automatic re-entry.
10. Make non-interactive cancellation and timeout behavior deterministic.

Recommended Lite stance:

1. Implement one active run per session from the beginning.
2. Add `RunID`, `Status`, `StartedAt`, `EndedAt`, and `Cancel` to the session store or runner state.
3. Support cancellation with `context.Context` before adding any queue.
4. Reject concurrent prompts for the same session at first, or queue one explicit follow-up with a bounded depth.
5. Serialize file mutations by canonical path if `apply_patch` or `write_file` is enabled.
6. Do not add detached/background work until foreground runs are durable and cancellable.

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

## System Prompt Best Practices And Commonalities

The mature harnesses treat the system prompt as a versioned runtime artifact. It is assembled from ordered sections, split into stable and dynamic parts, cached or hashed, and combined with runtime policy that enforces the rules in code.

Repository patterns:

| Repo | Prompt Pattern | Useful Files |
| --- | --- | --- |
| `openclaw` | `buildAgentSystemPrompt` inserts stable context before an explicit cache boundary and dynamic additions after it; context files and capability IDs are sorted for stable ordering | `src/agents/system-prompt.ts`, `src/agents/system-prompt-cache-boundary.ts`, `src/agents/pi-embedded-runner/system-prompt.ts` |
| `claude-code` | `getSystemPrompt` and `buildEffectiveSystemPrompt` split stable and dynamic sections, use section-level cache controls, and load managed/user/project instructions through a precedence model | `constants/prompts.ts`, `utils/systemPrompt.ts`, `constants/systemPromptSections.ts`, `utils/api.ts`, `utils/markdownConfigLoader.ts` |
| `pi-mono` | `buildSystemPrompt` creates a compact coding-agent prompt; session rebuilds the base prompt when active tools change; project context loads `AGENTS.md` and `CLAUDE.md` from agent and ancestor directories | `packages/coding-agent/src/core/system-prompt.ts`, `packages/coding-agent/src/core/agent-session.ts`, `packages/coding-agent/src/core/resource-loader.ts` |
| `hermes-agent` | `_build_system_prompt` builds and reuses a session-stable prompt; ephemeral context is appended at API-call time; prompt builder scans and truncates context files | `run_agent.py`, `agent/prompt_builder.py`, `agent/prompt_caching.py`, `website/docs/developer-guide/prompt-assembly.md` |
| `zeroclaw` | System prompt builder composes identity, safety, tools, autonomy, and workspace sections; prompt guard scans risky inputs; bootstrap files are bounded | `crates/zeroclaw-runtime/src/agent/system_prompt.rs`, `crates/zeroclaw-runtime/src/agent/prompt.rs`, `crates/zeroclaw-runtime/src/security/prompt_guard.rs` |
| `opencode` | Final system message combines provider/model prompt, environment, skills, custom system text, per-user system text, plugin transforms, and resolved instruction files | `packages/opencode/src/session/system.ts`, `packages/opencode/src/session/prompt.ts`, `packages/opencode/src/session/llm.ts`, `packages/opencode/src/session/instruction.ts` |

Common prompt layers:

| Layer | Stable Or Dynamic | Purpose |
| --- | --- | --- |
| Identity and role | Stable | Define the agent's job, tone, and operating domain |
| Safety and authority | Stable | State instruction hierarchy, workspace boundaries, approval expectations, and forbidden behavior |
| Tool-use rules | Stable | Explain when to use tools, how to report tool use, and how to handle failures |
| Tool schemas | Stable per session | Give provider-native tool definitions or prompt-format tool descriptions |
| Autonomy rules | Stable | Encourage persistence, investigation, verification, and completion within policy limits |
| Project instructions | Stable or semi-stable | Include trusted project files such as `AGENTS.md`, `CLAUDE.md`, or repo-specific docs |
| Runtime environment | Dynamic | Add cwd, platform, date, model, sandbox, and permission mode |
| User/session context | Dynamic | Add current request, selected files, memory snippets, todos, and recent state |
| Response style | Stable | Keep answers concise, direct, and CLI-friendly |

Best practices:

1. Keep stable identity, safety, tool rules, and project instructions at the start of the prompt.
2. Put date, cwd, model, active session state, and ephemeral reminders after the stable cache boundary.
3. Make cache boundaries explicit in the prompt builder, not implicit in string concatenation.
4. Sort and dedupe tools, capabilities, skills, and context files before insertion.
5. Apply provider cache metadata as an overlay. Do not mutate the canonical prompt or tool schemas.
6. Fence project instructions and label their source paths.
7. Treat project instructions, memory, and retrieved context as lower priority than harness rules.
8. Scan or bound project context for prompt-injection patterns, but enforce security in runtime policy.
9. Include tool honesty rules: do not claim to have read, edited, tested, or run anything unless a tool actually did it.
10. Include explicit tool-result handling: report errors, do not hide failed tools, and continue only when safe.
11. Include autonomy rules that promote action, but never let them override approval or workspace limits.
12. Include verification expectations, especially after code edits.
13. Keep response-style rules short and below safety/tool rules.
14. Hash the final stable system prefix and tool schema set for cache diagnostics and replay.
15. Version the prompt so evals and trace replay can explain behavior changes.

Recommended Lite prompt structure:

```text
[stable]
1. Identity and role
2. Instruction hierarchy and safety rules
3. Workspace and file-access rules
4. Agent loop and tool-use rules
5. Tool honesty and verification rules
6. Available tools and mutation policy
7. Response style
8. Optional project instructions, fenced and source-labeled
<!-- SAFEAGENT_CACHE_BOUNDARY -->

[dynamic]
9. Current date, cwd, platform, provider, model
10. Session status, run ID, permission mode
11. Current user request and explicit attachments
12. Optional memory or retrieved context, fenced as informational
```

Lite should load project instruction files only after path validation and size limits. It should treat those files as untrusted project context below the harness-level system rules. The prompt can tell the model not to exfiltrate secrets or escape the workspace, but tool implementations must enforce those controls.

## Prompt Caching

Prompt caching is mostly provider-side. The provider must implement cache storage, prefix matching, KV reuse, TTL, billing discounts, and usage reporting. The harness still matters because it determines whether requests are cacheable.

The harness should keep stable content at the beginning of the request, place provider-specific cache markers or keys, preserve byte-identical prompt and tool schema prefixes, and measure cache hit rates. Prompt caching is not response caching; the provider still generates new output.

### Cache Types

| Cache Type | Implemented By | What It Reuses | Lite Recommendation |
| --- | --- | --- | --- |
| Prompt/KV cache | LLM provider | Repeated prompt prefix, often attention KV tensors | Use through provider APIs |
| Explicit context cache | Some providers, especially Gemini/Vertex | Named cached content resource | Defer until lifecycle is needed |
| Application response cache | Harness | Final response for identical request | Avoid initially for agent loops |

### Provider API Summary

| Provider | Main Mechanism | Harness Control | Usage Fields |
| --- | --- | --- | --- |
| Anthropic Claude | `cache_control` on request or content blocks | Explicit breakpoints, 5m default TTL, optional 1h TTL | `cache_creation_input_tokens`, `cache_read_input_tokens` |
| OpenAI | Automatic prefix caching | Stable prefix, `prompt_cache_key`, `prompt_cache_retention` where supported | `cached_tokens` in token details |
| Google Gemini / Vertex AI | Implicit caching plus explicit context cache resources | Stable prefix for implicit; create/use/update/delete cache resources for explicit | `cachedContentTokenCount` |
| Amazon Bedrock | `cachePoint` in Converse, `cache_control` for Claude InvokeModel | Explicit checkpoints, TTL where supported | `CacheReadInputTokens`, `CacheWriteInputTokens` |
| OpenRouter and proxies | Provider-dependent | Anthropic-style markers or OpenAI-style keys only when capability-gated | Varies |

### Anthropic

Anthropic supports automatic caching and explicit cache breakpoints.

Important behavior:

1. Cached prefix order is `tools`, then `system`, then `messages`.
2. Cache hits require exact matching content up to the cache breakpoint.
3. Default TTL is 5 minutes.
4. Optional 1-hour TTL is available at higher write cost.
5. Cache read tokens are much cheaper than normal input tokens.
6. Cache write tokens cost more than normal input tokens.
7. Caching only works above model-specific minimum token thresholds.
8. Explicit block-level caching supports up to 4 breakpoints.
9. Anthropic can look back roughly 20 content blocks from a breakpoint to find a prior cache entry.
10. Tool definitions, system blocks, messages, images, tool uses, and tool results can be cached.

Harness implications:

1. Put tool schemas first and keep their order stable.
2. Put stable system prompt content before dynamic runtime content.
3. Put `cache_control` on the last stable block, not on a timestamp or current user message unless using automatic conversation caching deliberately.
4. Do not add cache markers to thinking blocks.
5. Track `cache_creation_input_tokens` and `cache_read_input_tokens`.
6. Avoid random JSON key order in tool calls and tool schemas.

Short retention marker:

```json
{ "cache_control": { "type": "ephemeral" } }
```

Long retention marker:

```json
{ "cache_control": { "type": "ephemeral", "ttl": "1h" } }
```

### OpenAI

OpenAI prompt caching is automatic for recent models and prompts of at least 1024 tokens. There is no Anthropic-style block-level `cache_control` for direct OpenAI APIs.

Important behavior:

1. Caching works on exact prefix matches.
2. Static content should be at the beginning of the request.
3. Dynamic user-specific content should be at the end.
4. `prompt_cache_key` can improve routing for shared prefixes.
5. `prompt_cache_retention` can request in-memory or extended retention where supported.
6. OpenAI reports cache hits through `cached_tokens` in usage details.
7. Prompt caching does not change output generation.
8. Cached prompts still count toward rate limits according to OpenAI docs.

Harness implications:

1. Build OpenAI requests with stable prefix order.
2. Use `prompt_cache_key` derived from a stable session or stable prompt family.
3. Do not generate a new cache key on every request.
4. Use retention controls only when supported by the selected endpoint and model.
5. Log `cached_tokens` and compute cache hit percentage.

Recommended Lite behavior:

```go
PromptCacheKey := stableSessionID
PromptCacheRetention := "in_memory" // or "24h" only when supported and desired
```

### Gemini / Vertex AI

Vertex AI offers implicit and explicit context caching.

Important behavior:

1. Implicit caching is automatic for supported Gemini models.
2. Implicit caching rewards large common prompt prefixes sent close together in time.
3. Explicit caching requires creating a context cache resource.
4. Explicit caches can be used, updated, inspected, and deleted through the API.
5. Explicit caches have storage costs based on lifetime.
6. `cachedContentTokenCount` reports cached token count.

Harness implications:

1. Start with implicit caching only.
2. Normalize and log `cachedContentTokenCount` as cache read tokens.
3. Do not implement explicit context cache resources until the harness owns session lifecycle, cleanup, and TTL refresh semantics.
4. Consider explicit context caching later for large project context, documentation bundles, or long-lived read-only corpora.

### Amazon Bedrock

Bedrock supports prompt caching through explicit cache checkpoints for supported models and APIs.

Important behavior:

1. Bedrock uses cache checkpoints to define prompt prefixes.
2. Converse APIs use `cachePoint`.
3. Claude InvokeModel-style payloads can use Anthropic `cache_control`.
4. Cache checkpoints require model-specific minimum token counts.
5. Many Claude models allow up to 4 checkpoints.
6. TTL is commonly 5 minutes, with 1-hour TTL on supported Claude models.
7. Bedrock reports cache read and write token counts.
8. Claude on Bedrock has simplified cache management with lookback near a specified breakpoint.

Harness implications:

1. Add `cachePoint` after stable system content, stable tool definitions, or a stable message prefix.
2. Capability-gate by provider, model, endpoint, and region.
3. Avoid cache checkpoints below token minimums because they silently fail to create useful cache entries.
4. Normalize Bedrock cache usage into common usage buckets.

### Proxies

Proxy support varies. Some routes accept Anthropic-compatible `cache_control`; some OpenAI-compatible routes accept `prompt_cache_key`; some ignore or reject unsupported fields.

Harness rules:

1. Never blindly send provider-specific cache fields to unknown proxies.
2. Add provider capability flags such as `SupportsAnthropicCacheControl`, `SupportsPromptCacheKey`, and `SupportsLongCacheRetention`.
3. Default to provider-automatic behavior when compatibility is unknown.
4. Parse usage fields defensively because proxies may return Anthropic, OpenAI, or custom cache metrics.

### Submodule Lessons

| Repo | Prompt Caching Pattern |
| --- | --- |
| `claude-code` | Strongest cache stability system: session-latched TTL, split system blocks, exactly one Anthropic message marker, cached base tool schemas, cache-break detection |
| `openclaw` | Explicit system prompt cache boundary, proxy safety rules, prompt/tool digest observability |
| `pi-mono` | Clean provider-neutral `cacheRetention` and `sessionId` options across Anthropic, OpenAI, Bedrock, and Google usage accounting |
| `hermes-agent` | Practical Anthropic strategy: system prompt plus last three non-system messages, stored system prompt reuse, whitespace/tool JSON normalization |
| `zeroclaw` | Provider capability flag and cached-token usage field, partial Anthropic and Bedrock marker support |
| `opencode` | Provider transform layer marks first two system messages and last two non-system messages, with provider-specific option shapes |

### Lite Cache Design

Add cache policy as a first-class part of model request assembly.

```go
type CacheRetention string

const (
    CacheRetentionNone  CacheRetention = "none"
    CacheRetentionShort CacheRetention = "short"
    CacheRetentionLong  CacheRetention = "long"
)

type CachePolicy struct {
    Enabled   bool
    Retention CacheRetention
    SessionID string
}

type CacheCapabilities struct {
    Automatic             bool
    AnthropicCacheControl bool
    OpenAIPromptCacheKey  bool
    OpenAIRetention       bool
    BedrockCachePoint     bool
    GeminiExplicitContext bool
    LongRetention         bool
}
```

Normalize usage:

```go
type Usage struct {
    InputFreshTokens int64
    CacheReadTokens  int64
    CacheWriteTokens int64
    OutputTokens     int64
    TotalTokens      int64
}
```

Track cache diagnostics:

```go
type CacheDiagnostics struct {
    Enabled          bool
    ProviderMode     string
    Retention        CacheRetention
    SessionID        string
    SystemDigest     string
    ToolSchemaDigest string
    RequestDigest    string
    CacheReadTokens  int64
    CacheWriteTokens int64
    SuspectedBreak   string
}
```

Recommended stable prompt layout:

```text
[stable]
1. Harness identity and safety rules
2. Tool-use rules
3. Tool schema descriptions, if provider serializes them here
4. Project-level instructions that are stable for this session
<!-- SAFEAGENT_CACHE_BOUNDARY -->

[dynamic]
5. Current date
6. Current cwd
7. Model/provider/runtime information
8. Current user message and recent ephemeral context
```

Stable tool schema rules:

1. Sort tools by name.
2. Serialize JSON schema with sorted keys.
3. Do not include runtime-only values in tool descriptions.
4. Do not include timestamps, random IDs, absolute temp paths, or feature flags in schema text.
5. Keep a session-stable copy of base tool schemas.
6. Apply provider cache metadata as an overlay, not by mutating the base schema.

Cache-affecting state should be latched when a session starts:

1. Provider and model.
2. Cache retention mode.
3. Tool list and schemas.
4. Stable system prompt prefix.
5. Provider headers and beta flags that affect request shape.
6. Reasoning or thinking mode if it changes request formatting.

Cache-break detection should compare current and prior observations:

1. Compute digests for stable system prompt, tool schemas, cache config, provider options, and message prefix.
2. Store the prior observation per session.
3. Compare current cache read tokens to prior cache read tokens.
4. If cache read drops by more than 5% and more than 1,000 tokens, flag a possible break.
5. If prompt digests changed, classify it as client-side prompt mutation.
6. If digests did not change but TTL elapsed, classify it as likely TTL expiry.
7. If neither changed nor elapsed, classify it as provider routing or eviction.

Recommended Lite defaults:

| Setting | Default |
| --- | --- |
| Cache enabled | Yes when provider supports it |
| Retention | Short |
| Long retention | Off unless explicitly configured |
| Cache key | Stable session ID |
| Explicit Gemini context cache | Off |
| Unknown proxy cache fields | Off |
| Stable prompt digest logging | On |
| Tool schema digest logging | On |
| Cache usage normalization | On |
| Cache break detection | On after initial provider integration |

### Prompt Caching Tests

| Test | Purpose |
| --- | --- |
| Stable prompt digest | Same stable inputs produce same digest |
| Dynamic suffix excluded | Date/cwd changes do not alter stable prefix digest |
| Tool schema ordering | Tool order is deterministic regardless of registration order |
| JSON key ordering | Schema serialization is stable |
| Anthropic marker placement | Cache marker lands on stable prefix or configured trailing block |
| OpenAI cache key | Stable session ID maps to `prompt_cache_key` |
| Unknown proxy | Unsupported cache fields are not sent |
| Usage normalization | Provider-specific fields map to common usage buckets |
| Cache break detection | Digest change classifies client-side cache break |
| TTL break classification | Same digest after TTL classifies likely expiration |

The recommendation is to build cache-friendly requests and provider adapters, not a local prompt cache.

## Telemetry And LLM Tracing

Telemetry should be part of the harness from the beginning because agent failures are hard to diagnose from final text alone. The most useful telemetry captures the causal tree: user request, prompt assembly, model calls, streamed output, tool calls, policy decisions, cache behavior, context compaction, retries, and final result.

### OpenTelemetry And Langfuse

Langfuse can receive OpenTelemetry traces through its OTLP HTTP endpoint:

```text
https://cloud.langfuse.com/api/public/otel
```

For signal-specific exporters, the trace endpoint is:

```text
https://cloud.langfuse.com/api/public/otel/v1/traces
```

Langfuse supports OTLP over HTTP using JSON or protobuf. It does not currently support OTLP gRPC ingestion on that endpoint.

Authentication uses Basic Auth with the Langfuse public and secret keys:

```text
OTEL_EXPORTER_OTLP_ENDPOINT=https://cloud.langfuse.com/api/public/otel
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic <base64(pk-lf-...:sk-lf-...)>,x-langfuse-ingestion-version=4
```

For Go, the practical path is direct OpenTelemetry instrumentation using `go.opentelemetry.io/otel` and the OTLP HTTP exporter. Do not make Langfuse a hard dependency of the core harness. Emit standard OpenTelemetry spans and let users route OTLP to Langfuse, Honeycomb, Tempo, Grafana, Datadog, or a collector.

### OpenTelemetry GenAI Semantics

OpenTelemetry GenAI semantic conventions are still in development, but they are already useful. Use them where stable enough, and isolate them in one telemetry adapter so attribute names can evolve.

Important span attributes:

| Attribute | Purpose |
| --- | --- |
| `gen_ai.operation.name` | Operation type, such as `chat`, `execute_tool`, `invoke_agent`, `retrieval` |
| `gen_ai.provider.name` | Provider, such as `openai`, `anthropic`, `aws.bedrock`, `gcp.vertex_ai` |
| `gen_ai.conversation.id` | Session or conversation ID |
| `gen_ai.request.model` | Requested model |
| `gen_ai.response.model` | Actual response model |
| `gen_ai.request.temperature` | Request temperature |
| `gen_ai.request.max_tokens` | Output token limit |
| `gen_ai.response.finish_reasons` | Stop reasons |
| `gen_ai.usage.input_tokens` | Total input tokens, including cached tokens |
| `gen_ai.usage.output_tokens` | Output tokens |
| `gen_ai.usage.cache_read.input_tokens` | Provider cache read tokens |
| `gen_ai.usage.cache_creation.input_tokens` | Provider cache write tokens |
| `error.type` | Low-cardinality error classification |

Content attributes are opt-in because they may contain secrets or PII:

| Attribute | Risk |
| --- | --- |
| `gen_ai.input.messages` | Full prompt and tool results can contain user data and secrets |
| `gen_ai.output.messages` | Model output can contain sensitive data |
| `gen_ai.system_instructions` | System prompt can expose proprietary policy and instructions |
| `gen_ai.tool.definitions` | Tool schemas can be large and reveal internal capabilities |

Recommended metrics:

| Metric | Purpose |
| --- | --- |
| `gen_ai.client.operation.duration` | Model or tool operation duration |
| `gen_ai.client.token.usage` | Token usage histogram with `gen_ai.token.type=input|output` |

Additional useful harness metrics:

| Metric | Purpose |
| --- | --- |
| `safeagent.run.duration` | End-to-end run latency |
| `safeagent.loop.steps` | Loop step count |
| `safeagent.tool.duration` | Tool latency |
| `safeagent.tool.errors` | Tool failure count by tool and error class |
| `safeagent.policy.denials` | Policy denial count |
| `safeagent.cache.hit_tokens` | Cache read tokens |
| `safeagent.cache.write_tokens` | Cache write tokens |
| `safeagent.context.tokens` | Estimated context size at request time |

### Langfuse Mapping

Langfuse maps OpenTelemetry spans into its trace and observation model. It supports standard `gen_ai.*` attributes and a `langfuse.*` namespace for precise mapping.

Trace-level attributes:

| Langfuse Attribute | Purpose |
| --- | --- |
| `langfuse.trace.name` | Human-readable trace name |
| `langfuse.user.id` or `user.id` | User identifier |
| `langfuse.session.id` or `session.id` | Session identifier |
| `langfuse.trace.tags` | Tags |
| `langfuse.trace.metadata.*` | Filterable trace metadata |
| `langfuse.version` | Application or prompt version |
| `langfuse.release` | Release identifier |

Observation-level attributes:

| Langfuse Attribute | Purpose |
| --- | --- |
| `langfuse.observation.type` | `span`, `generation`, or `event` |
| `langfuse.observation.model.name` | Model name for a generation |
| `langfuse.observation.model.parameters` | JSON model parameters |
| `langfuse.observation.usage_details` | JSON token usage |
| `langfuse.observation.cost_details` | JSON cost details |
| `langfuse.observation.input` | JSON input, if content capture is enabled |
| `langfuse.observation.output` | JSON output, if content capture is enabled |
| `langfuse.observation.prompt.name` | Langfuse prompt name, if using prompt management |
| `langfuse.observation.prompt.version` | Langfuse prompt version |

Langfuse recommends propagating trace-level attributes to every span for reliable filtering and aggregation. OpenTelemetry baggage can do this, but baggage crosses service boundaries. Do not put secrets, API keys, raw prompts, or personal data in baggage.

### Recommended Trace Shape

For `safeagent.Lite`, use this hierarchy:

```text
safeagent.run
  safeagent.prompt.assemble
  safeagent.context.estimate
  gen_ai.chat <model>
    safeagent.stream.collect
  safeagent.tool.execute <tool_name>
    safeagent.policy.authorize
    safeagent.tool.read_file | safeagent.tool.grep | safeagent.tool.apply_patch
  gen_ai.chat <model>
  safeagent.result.finalize
```

Root span attributes:

| Attribute | Example |
| --- | --- |
| `langfuse.trace.name` | `safeagent.lite.run` |
| `langfuse.session.id` | Session ID |
| `langfuse.user.id` | User ID if available |
| `langfuse.trace.tags` | `safeagent,lite,no-bash` |
| `safeagent.run.id` | Run ID |
| `safeagent.workspace.id` | Stable workspace hash, not raw path by default |
| `safeagent.loop.max_steps` | Max step budget |
| `safeagent.tools.enabled` | Tool names, bounded |

Model span attributes:

| Attribute | Example |
| --- | --- |
| `gen_ai.operation.name` | `chat` |
| `gen_ai.provider.name` | `anthropic` |
| `gen_ai.request.model` | `claude-sonnet-4-5` |
| `gen_ai.conversation.id` | Session ID |
| `gen_ai.request.max_tokens` | Output token limit |
| `gen_ai.usage.input_tokens` | Total input tokens |
| `gen_ai.usage.output_tokens` | Output tokens |
| `gen_ai.usage.cache_read.input_tokens` | Cache read tokens |
| `gen_ai.usage.cache_creation.input_tokens` | Cache write tokens |
| `safeagent.prompt.system_sha256` | Stable prompt digest |
| `safeagent.tools.schema_sha256` | Tool schema digest |
| `safeagent.cache.retention` | `short` or `long` |

Tool span attributes:

| Attribute | Example |
| --- | --- |
| `gen_ai.operation.name` | `execute_tool` |
| `tool.name` | `read_file` |
| `safeagent.tool.call_id` | Provider tool call ID |
| `safeagent.tool.read_only` | `true` |
| `safeagent.policy.decision` | `allow`, `deny`, `ask` |
| `safeagent.tool.input_bytes` | Bounded input size |
| `safeagent.tool.output_bytes` | Bounded output size |
| `safeagent.tool.output_truncated` | `true` or `false` |
| `error.type` | Low-cardinality error class |

### Privacy And Security

Telemetry can easily become a data leak. The default should be safe.

Rules:

1. Telemetry should be opt-in for external export.
2. Raw prompts, model outputs, tool arguments, and tool results should be redacted by default.
3. Add a separate explicit setting for content capture.
4. Even when content capture is enabled, truncate content and redact secrets.
5. Use hashes for prompt/tool schema identity by default.
6. Do not put raw paths in high-cardinality metric attributes; use hashes or record raw paths only in opt-in trace events.
7. Do not put secrets or user prompt content in OpenTelemetry baggage.
8. Bound attribute sizes because OTLP backends often reject or truncate large attributes.
9. Prefer low-cardinality attributes for metrics.
10. Keep full replay transcripts in the local session store, not in telemetry, unless the user explicitly enables remote content capture.

### Submodule Telemetry Lessons

| Repo | Telemetry Pattern |
| --- | --- |
| `claude-code` | Mature OTel setup with env-configured exporters, console/OTLP support, interaction/LLM/tool/hook spans, redaction by default, hash dedupe for large content, AsyncLocalStorage context, orphan span cleanup, Perfetto local traces |
| `opencode` | Uses Effect tracing and Vercel AI SDK `experimental_telemetry`, injects `session.id`, wraps HTTP handlers with route spans |
| `zeroclaw` | Observer abstraction supports `noop`, `log`, `verbose`, Prometheus, and OTLP; creates spans and metrics but should move toward GenAI semantic attributes |
| `openclaw` | Strong local trajectory tracing and diagnostic trace context; useful for replay and debugging even when remote telemetry is disabled |
| `pi-mono` | Minimal install telemetry and provider attribution; not a deep LLM tracing reference |
| `hermes-agent` | Tool traces and plugin hooks exist; Langfuse appears as plugin metadata in tests, but no first-class OTel harness was found |

### Lite Telemetry Design

Keep observability provider-neutral in the core package.

```go
type Observer interface {
    StartRun(ctx context.Context, attrs RunAttrs) (context.Context, SpanEnd)
    StartModelCall(ctx context.Context, attrs ModelCallAttrs) (context.Context, SpanEnd)
    StartToolCall(ctx context.Context, attrs ToolCallAttrs) (context.Context, SpanEnd)
    RecordEvent(ctx context.Context, name string, attrs map[string]any)
    RecordMetric(ctx context.Context, name string, value float64, attrs map[string]any)
}

type SpanEnd func(err error, attrs map[string]any)
```

Provide implementations:

| Observer | Purpose |
| --- | --- |
| `NoopObserver` | Default safe fallback |
| `LogObserver` | Local debugging without remote export |
| `OTelObserver` | Production tracing and metrics through OpenTelemetry |

For Go, use OpenTelemetry directly:

```go
import (
    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
    "go.opentelemetry.io/otel/sdk/resource"
    sdktrace "go.opentelemetry.io/otel/sdk/trace"
)
```

Langfuse integration should be configuration, not a compile-time dependency:

```text
SAFEAGENT_OTEL_EXPORTER=otlp
OTEL_EXPORTER_OTLP_ENDPOINT=https://cloud.langfuse.com/api/public/otel
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic <base64>,x-langfuse-ingestion-version=4
```

Recommended Lite defaults:

| Setting | Default |
| --- | --- |
| Remote telemetry | Off |
| Local trace log | Optional |
| Raw prompt capture | Off |
| Raw output capture | Off |
| Tool argument capture | Redacted or hashed |
| Tool result capture | Redacted, truncated, off for remote by default |
| Token/cost metrics | On when provider reports usage |
| Cache diagnostics | On |
| Prompt/tool digests | On |
| User/session IDs | Session ID on; user ID only if provided by host |

Telemetry tests:

| Test | Purpose |
| --- | --- |
| Noop observer | Harness runs without telemetry configured |
| Span hierarchy | Run contains model and tool child spans |
| Error status | Provider and tool errors mark spans failed with `error.type` |
| Redaction default | Raw prompt and tool result are not exported by default |
| Content opt-in | Raw content is exported only when explicitly enabled |
| Attribute bounds | Large content is truncated or hashed |
| Cache usage attrs | Cache read/write tokens appear on model spans |
| Langfuse attrs | Session/user/tags use `langfuse.*` or standard equivalents |
| Baggage safety | No prompt, secret, or tool output enters baggage |

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

## Evals

Agent evals should test harness invariants before model quality. The first eval suite should prove that the loop, provider adapter, tool execution, policy gates, context limits, and replay format behave deterministically with fake providers.

Repository patterns:

| Repo | Eval Pattern | Useful Files |
| --- | --- | --- |
| `openclaw` | Human-readable QA scenario packs in markdown with YAML fences; flow runner supports calls, variables, assertions, conditionals, loops, and recovery scenarios | `qa/scenarios/index.md`, `extensions/qa-lab/src/scenario-catalog.ts`, `extensions/qa-lab/src/scenario-flow-runner.ts`, `extensions/qa-lab/src/scenario.ts` |
| `claude-code` | VCR records and replays live provider calls, including streaming output and token counts, with normalized fixture hashes and CI replay behavior | `services/vcr.ts`, `services/api/claude.ts` |
| `pi-mono` | Coding-agent harness registers a faux provider, in-memory settings, temporary sessions, event capture, and deterministic scripted responses | `packages/coding-agent/test/suite/harness.ts`, `packages/ai/src/providers/faux.ts`, `packages/agent/test/agent-loop.test.ts` |
| `hermes-agent` | Benchmark environments wrap agent loops in isolated task environments, verifier scripts, deterministic seeds, and aggregate scoring | `environments/hermes_base_env.py`, `environments/agent_loop.py`, `environments/benchmarks/terminalbench_2/terminalbench2_env.py`, `environments/benchmarks/yc_bench/yc_bench_env.py` |
| `zeroclaw` | Trace replay fixtures define provider turns and declarative expectations; mock providers record requests and fail when traces are exhausted | `tests/support/mock_provider.rs`, `tests/support/trace.rs`, `tests/support/assertions.rs`, `tests/support/mock_tools.rs`, `tests/fixtures/traces/*.json` |
| `opencode` | In-process fake OpenAI-compatible HTTP/SSE server exercises real serialization and stream parsing; fake provider layers support faster unit tests | `packages/opencode/test/lib/llm-server.ts`, `packages/opencode/test/fake/provider.ts`, `packages/opencode/test/fixture/fixture.ts` |

Recommended Lite eval architecture:

| Component | Purpose |
| --- | --- |
| `ScriptedProvider` | FIFO provider responses for fast unit tests; records requests; fails on response exhaustion |
| `TraceReplayProvider` | Loads JSON traces and returns scripted text, tool calls, errors, and usage |
| `FakeLLMServer` | Local OpenAI-compatible HTTP/SSE server for provider adapter integration tests |
| `EvalRunner` | Creates temp workspace, in-memory session store, tool registry, policy, observer, and event capture |
| `ScenarioCatalog` | Markdown or JSON scenario pack for human-readable behavior specs |
| `VCRProvider` | Optional wrapper for live-provider recording and replay, disabled by default in CI unless fixtures exist |
| `Expect` | Declarative assertions over final text, tool calls, tool errors, file changes, policy decisions, and telemetry |

Minimal trace fixture shape:

```json
{
  "name": "single-tool-read",
  "turns": [
    {
      "user": "Read README.md and summarize it.",
      "steps": [
        {
          "assistant": {
            "tool_calls": [
              { "id": "call_1", "name": "read_file", "arguments": { "path": "README.md" } }
            ]
          }
        },
        {
          "assistant": { "text": "README.md describes ..." }
        }
      ]
    }
  ],
  "expects": {
    "tools_used": ["read_file"],
    "tools_not_used": ["apply_patch"],
    "response_contains": ["README"],
    "max_tool_calls": 1
  }
}
```

Core assertion types:

| Assertion | Checks |
| --- | --- |
| `response_contains` / `response_not_contains` | Final answer content |
| `response_matches` | Final answer regex |
| `tools_used` / `tools_not_used` | Tool selection |
| `tool_call_count` / `max_tool_calls` | Loop/tool budget |
| `tool_args_match` | Exact or partial argument shape |
| `all_tool_calls_have_results` | Provider tool-call pairing invariant |
| `policy_denied` | Safety gate triggered for forbidden action |
| `file_equals` / `file_contains` | Workspace mutation result |
| `no_external_path_access` | Workspace boundary enforcement |
| `telemetry_has_span` | Trace shape and key attributes |
| `prompt_digest_stable` | Prompt/cache stability |

Eval levels:

| Level | Runs In CI | Provider | Purpose |
| --- | --- | --- | --- |
| Unit | Yes | Scripted provider | Loop invariants, tool-result pairing, policy errors, cancellation |
| Trace replay | Yes | Trace replay provider | Regressions across known agent transcripts |
| Fake HTTP/SSE | Yes | Local fake server | Provider adapter serialization and streaming |
| Scenario pack | Yes for deterministic scenarios | Scripted provider or fake server | Higher-level workflows and safety cases |
| VCR replay | Optional | Recorded live calls | Provider compatibility without paid calls |
| Live model eval | Manual or nightly | Real provider | Model quality, prompt changes, tool-choice behavior |
| Sandbox benchmark | Manual or nightly | Real or scripted provider | End-to-end tasks with isolated verifiers |

Safety evals should be mandatory before mutation tools ship:

1. Path traversal with `../` is denied.
2. Symlink escape is denied.
3. Absolute path outside workspace is denied.
4. `.env`, private keys, and credential files are denied or require approval.
5. Writes under `.git` are denied.
6. Patch attempts that do not match file context fail safely.
7. Unknown tools return structured tool errors.
8. Invalid JSON arguments return structured tool errors.
9. Tool output is truncated and marked as truncated.
10. Redaction removes common secret patterns from logs and telemetry.

Avoid arbitrary code evaluation in eval files. OpenClaw's QA flow is flexible, but Lite should start with a small declarative assertion DSL so evals are safe to run on untrusted scenario packs.

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
