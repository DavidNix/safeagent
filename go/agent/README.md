# agent

A Go port of the core runtime of the [OpenAI Agents SDK](https://github.com/openai/openai-agents-js)
(`@openai/agents-core`): agents with instructions, function tools, handoffs,
guardrails, and a run loop that drives a model until a final output.

The `agent` package is provider-agnostic. The `agent/llm` subpackage
implements the `Model` interface against any OpenAI-compatible Chat
Completions API (OpenAI, OpenRouter, Ollama, vLLM, etc.) over plain HTTP —
no vendor SDK. It owns its wire types, so provider extensions such as vLLM's
`reasoning_content` are first-class.

## Quick start

```go
import (
	"github.com/DavidNix/safeagent/agent"
	"github.com/DavidNix/safeagent/agent/llm"
)

assistant := &agent.Agent{
	Name:         "Assistant",
	Instructions: "You are a helpful assistant.",
	Model:        llm.NewModel("gpt-4.1"), // reads OPENAI_API_KEY
}

result, err := agent.Run(ctx, assistant, "What is the capital of France?")
if err != nil {
	return err
}
fmt.Println(result.FinalOutput)
```

`agent.Run` uses a default `Runner`. Construct a `Runner` to configure the
run:

```go
runner := &agent.Runner{
	MaxTurns: 5,          // default 10; exceeded -> *MaxTurnsExceededError
	Context:  myAppState, // arbitrary value passed to tools and guardrails
}
result, err := runner.Run(ctx, assistant, "hello")
```

## The run loop

One **turn** is one model invocation. Each turn the runner sends the system
instructions, the full item history, and the agent's tools and handoffs to
the model, then acts on the response:

- **Tool calls** → every named tool is invoked (concurrently when the model
  issues several calls), outputs are appended to the history, and the loop
  continues.
- **A handoff call** → control transfers to the target agent, which continues
  the same run with the same history.
- **A plain assistant message** → the run is final. Output guardrails run,
  and `RunResult` is returned.

`MaxTurns` caps the loop; a run that never settles fails with
`*MaxTurnsExceededError` rather than spinning forever.

## Tools

A tool is a name, a JSON schema, and a Go function:

```go
weather := agent.Tool{
	Name:        "get_weather",
	Description: "Get the current weather for a city",
	Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
	OnInvoke: func(ctx context.Context, rc *agent.RunContext, arguments string) (string, error) {
		var args struct{ City string `json:"city"` }
		if err := json.Unmarshal([]byte(arguments), &args); err != nil {
			return "", err
		}
		return lookupWeather(ctx, args.City)
	},
}
```

The `RunContext` passed to `OnInvoke` exposes the `Runner.Context` value and
the run's accumulated `Usage`.

### Tool errors and timeouts

By default a failed `OnInvoke` does **not** abort the run: the error becomes
a model-visible tool output ("An error occurred while running the tool.
Please try again. Error: ...") and the loop continues, so the model can retry
or work around it. Customize the message with `OnInvokeError`, or set
`PropagateErrors: true` to abort the run with a `*ToolCallError` instead
(matching the JS SDK's `errorFunction` semantics).

`Timeout` caps a single invocation. A timed out tool likewise becomes a
model-visible output (`Tool 'name' timed out after 100ms.`) unless
`PropagateTimeout: true`, which aborts the run with a `*ToolTimeoutError`;
`OnTimeout` customizes the message. The invocation's ctx is cancelled at the
deadline — `OnInvoke` implementations should honor it, because a goroutine
that ignores ctx leaks until it returns. Cancellation of the run's own
context is never converted into tool output; it always propagates.

## Multi-agent coordination

Three patterns are supported, and loops are possible in all of them.

### 1. Handoffs (peer-to-peer transfer)

An agent's `Handoffs` are presented to the model as `transfer_to_<name>`
tools. When the model calls one, the run loop swaps the active agent and
continues with the full history. Cycles are allowed — wire mutual references
after construction:

```go
triage := &agent.Agent{Name: "Triage", Model: m, Instructions: "Route the user."}
billing := &agent.Agent{Name: "Billing", Model: m, HandoffDescription: "Handles invoices."}

triage.Handoffs = []agent.Handoff{agent.HandoffTo(billing)}
billing.Handoffs = []agent.Handoff{agent.HandoffTo(triage)} // loop back
```

Every transfer consumes a turn, so `MaxTurns` bounds handoff ping-pong. If a
response contains several handoff calls, the first wins and the rest receive
rejection outputs. `Handoff.OnHandoff` runs before control transfers;
`ToolName`, `ToolDescription`, and `InputSchema` override the defaults.
`RunResult.LastAgent` reports which agent finished the run.

### 2. Agent-as-tool (orchestrator)

`Agent.AsTool` wraps an agent in a function tool. The parent agent keeps
control, can invoke sub-agents repeatedly across turns — in parallel when the
model issues multiple calls — and receives each sub-agent's final output as
the tool result. Nested usage rolls up to the parent run.

```go
orchestrator := &agent.Agent{
	Name:  "Orchestrator",
	Model: m,
	Tools: []agent.Tool{
		translator.AsTool("", "Translate text to Spanish"),
		summarizer.AsTool("", "Summarize text"),
	},
}
```

### 3. Deterministic loops in your own code

`Runner.RunItems` accepts a full item history, and `RunResult` exposes
`History`, `NewItems`, and `LastAgent` — so evaluator-optimizer style loops
need no framework support:

```go
history := []agent.Item{agent.UserMessage(draftRequest)}
for range maxRounds {
	draft, err := runner.RunItems(ctx, writer, history)
	// feed the draft to a judge agent; break when it approves,
	// otherwise append its critique to history and loop
}
```

`RunItems` is also how you continue a conversation: append the previous
`result.History` plus the next user message and run again.

### Caveats

- **Nested runs get fresh budgets.** `AsTool` starts a new `Runner` with its
  own default `MaxTurns`, and nothing caps recursion *depth*. Mutual
  agent-as-tool references (A calls B, B calls A) are turn-bounded per level
  but not depth-bounded. Add a depth guard via `Runner.Context` if you build
  cyclic tool graphs.
- **Handoffs carry the entire history, unfiltered.** The JS SDK's handoff
  `inputFilter` was not ported, so long multi-hop runs grow context
  monotonically.

## Guardrails

Input guardrails run concurrently before the first model call; output
guardrails run against the final output. A triggered tripwire halts the run
with a typed error:

```go
ag.InputGuardrails = []agent.InputGuardrail{{
	Name: "blocklist",
	Execute: func(ctx context.Context, rc *agent.RunContext, a *agent.Agent, input []agent.Item) (agent.GuardrailOutput, error) {
		return agent.GuardrailOutput{TripwireTriggered: containsBanned(input)}, nil
	},
}}

_, err := agent.Run(ctx, ag, userInput)
var tripwire *agent.InputGuardrailTripwireTriggeredError
if errors.As(err, &tripwire) {
	// blocked; tripwire.Result has the guardrail's OutputInfo
}
```

Note: guardrails only run for the agent that starts (input) or finishes
(output) the run — not for every agent along a handoff chain.

## Dynamic instructions and model settings

`InstructionsFunc` resolves the system prompt per turn and takes precedence
over `Instructions` — useful for injecting per-run state from
`RunContext.Context`:

```go
ag.InstructionsFunc = func(ctx context.Context, rc *agent.RunContext, a *agent.Agent) (string, error) {
	user := rc.Context.(*User)
	return "You are helping " + user.Name, nil
}
```

`ModelSettings` tunes generation per agent: `Temperature`, `TopP`,
`MaxTokens`, `ParallelToolCalls` (nil pointers are omitted from the request),
and `ToolChoice` (`"auto"`, `"required"`, `"none"`, or a tool name to force).

A forced `ToolChoice` is a one-shot constraint: once an agent has executed
tool calls in a run, any `ToolChoice` other than `"none"` is dropped from
subsequent requests so the model can produce a final answer — otherwise a
`"required"` or named tool choice would force tool calls every turn until
`MaxTurns`. Tracking is per-agent, so a handoff target's own forced choice is
still honored on its first call. Set `Agent.DisableToolChoiceReset` to opt
out (mirrors the JS SDK's `resetToolChoice`).

## Results, usage, and errors

`RunResult` carries `FinalOutput`, `NewItems` (everything generated),
`History` (input + generated), `LastAgent`, aggregated `Usage` (requests and
token counts across all model calls, including nested agent-as-tool runs),
and the guardrail results.

Failures are typed — match with `errors.As`: `*MaxTurnsExceededError`,
`*ModelBehaviorError` (e.g. the model called a nonexistent tool),
`*ToolCallError`, `*ToolTimeoutError`, `*GuardrailExecutionError`, the two
tripwire errors, and `*UserError` for misconfiguration (nil agent, missing
model).

## Tracing

Inject an optional `Tracer` via `Runner.Tracer` to observe the run lifecycle:
`RunStarted`, `TurnStarted`, `ModelCallEnded`, `ToolStarted`/`ToolEnded`
(these two fire from parallel goroutines — implementations must be
concurrency-safe), `Handoff`, and `RunEnded`. `ToolEnded` reports the raw
invocation result before error recovery rewrites it for the model. The
tracer propagates to nested runs started by `Agent.AsTool`.

Embed `NoopTracer` to implement only the methods you need, or use the
built-in slog adapter:

```go
runner := &agent.Runner{
	Tracer: agent.NewSlogTracer(slog.Default()),
}
```

`SlogTracer` logs run boundaries and handoffs at info, turns, model calls,
and tool activity at debug, and failures at error.

## Custom models and testing

`Model` is a one-method interface, which makes agents trivial to unit test —
no HTTP, no API key:

```go
type Model interface {
	GetResponse(ctx context.Context, req ModelRequest) (*ModelResponse, error)
}
```

Script a fake that returns tool calls, handoffs, or messages and assert on
the requests it received (see `run_test.go` for a reference implementation).
Any backend — a different vendor API, a local model, a rules engine — plugs
in the same way.

The `llm` package options: `llm.WithBaseURL` (point it at any
OpenAI-compatible server), `llm.WithAPIKey`, `llm.WithMaxRetries` and
`llm.WithRetryDelay` (retryable failures — HTTP 429, 5xx, and transport
errors — are retried with exponential backoff, honoring `Retry-After`; the
default is 2 retries), and `llm.WithClient`, which accepts any `llm.Doer`:

```go
type Doer interface {
	Do(req *http.Request) (*http.Response, error)
}
```

`*http.Client` satisfies it directly; wrap one for logging, custom auth, or
other middleware, or inject a fake in tests to skip HTTP entirely.

Reasoning models served with a reasoning parser (for example
`vllm serve ... --reasoning-parser deepseek_r1`) return their thinking in a
`reasoning_content` field; the `llm` package surfaces it as an
`agent.Reasoning` item in `ModelResponse.Output` — visible to tracers via
`ModelCallEnded` — and never replays reasoning items back to the model when
converting history into requests.

## Not ported (yet)

Streaming, sessions/memory, MCP servers, structured output types, handoff
input filters, `toolUseBehavior` (stop-on-first-tool and friends),
human-in-the-loop interruptions, OpenAI dashboard trace export, and the
Responses API transport. The `Model` interface is the seam for most of
these.
