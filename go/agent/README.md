# agent

A Go port of the core runtime of the [OpenAI Agents SDK](https://github.com/openai/openai-agents-js)
(`@openai/agents-core`): agents with instructions, function tools, handoffs,
guardrails, and a run loop that drives a model until a final output.

The `agent` package is provider-agnostic. The `agent/llm` subpackage
implements the `Model` interface against any OpenAI-compatible Chat
Completions API (OpenAI, OpenRouter, Ollama, vLLM, etc.). You construct and
configure the client — typically the official
[openai-go](https://github.com/openai/openai-go) SDK — and inject it via the
`llm.Client` interface.

## Quick start

```go
import (
	"github.com/openai/openai-go/v3"

	"github.com/DavidNix/safeagent/agent"
	"github.com/DavidNix/safeagent/agent/llm"
)

client := openai.NewClient() // reads OPENAI_API_KEY
assistant := &agent.Agent{
	Name:         "Assistant",
	Instructions: "You are a helpful assistant.",
	Model:        llm.NewModel("gpt-4.1", &client.Chat.Completions),
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

Tool errors abort the run wrapped in `*ToolCallError`. The `RunContext`
passed to `OnInvoke` exposes the `Runner.Context` value and the run's
accumulated `Usage`.

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

## Results, usage, and errors

`RunResult` carries `FinalOutput`, `NewItems` (everything generated),
`History` (input + generated), `LastAgent`, aggregated `Usage` (requests and
token counts across all model calls, including nested agent-as-tool runs),
and the guardrail results.

Failures are typed — match with `errors.As`: `*MaxTurnsExceededError`,
`*ModelBehaviorError` (e.g. the model called a nonexistent tool),
`*ToolCallError`, `*GuardrailExecutionError`, the two tripwire errors, and
`*UserError` for misconfiguration (nil agent, missing model).

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

The `llm` package accepts any implementation of its one-method `llm.Client`
interface. `&client.Chat.Completions` from an `openai.Client` satisfies it,
configured however you need (`option.WithBaseURL` for compatible servers,
`option.WithAPIKey`, `option.WithHTTPClient`, retries, middleware) — or
substitute your own fake in tests without any HTTP server at all.

## Not ported (yet)

Streaming, tracing, sessions/memory, MCP servers, structured output types,
handoff input filters, human-in-the-loop interruptions, and the Responses
API transport. The `Model` interface is the seam for most of these.
