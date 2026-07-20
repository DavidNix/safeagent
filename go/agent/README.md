# agent

A Go port of the core runtime of the [OpenAI Agents SDK](https://github.com/openai/openai-agents-js)
(`@openai/agents-core`): agents with instructions, function tools, handoffs,
guardrails, and a run loop that drives a model until a final output.

The `agent` package is provider-agnostic. The top-level `llm` package provides
plain-HTTP clients for OpenAI-compatible Chat Completions and Embeddings APIs
(OpenAI, OpenRouter, Ollama, vLLM, etc.). It owns the wire types, so provider
extensions such as vLLM's `reasoning_content` are first-class.

## Quick start

```go
import (
	"os"

	"github.com/DavidNix/safeagent/go/agent"
	"github.com/DavidNix/safeagent/go/llm"
)

model := llm.NewOpenRouter(llm.OpenRouterConfig{
	APIKey:    os.Getenv("OPENROUTER_API_KEY"),
	ChatModel: "openai/gpt-4.1",
})
assistant := &agent.Agent{
	Name:         "Assistant",
	Instructions: "You are a helpful assistant.",
	Model:        model,
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

A tool is a name, a JSON schema, and a Go function. Its name is required and
must be unique across the agent's tools and resolved handoff names:

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

`ModelSettings` tunes requests per agent: `RequestTimeout`, `Temperature`,
`TopP`, `MaxTokens`, `ReasoningTokenBudget`, `ParallelToolCalls` (nil pointers
are omitted from the request), `ToolChoice` (`"auto"`, `"required"`, `"none"`,
or a tool name to force), and `StructuredOutput`. `RequestTimeout` applies
separately to each provider attempt; zero uses the client's default and a
negative duration disables the client timeout.

`StructuredOutput` requests modern JSON Schema structured output. The client
does not expose the legacy `json_object` mode and does not validate the model's
response locally; `RunResult.FinalOutput` remains a JSON string. Provider and
model support varies.

```go
schema := json.RawMessage(`{
  "type": "object",
  "properties": {
    "answer": {"type": "string"}
  },
  "required": ["answer"],
  "additionalProperties": false
}`)

assistant := &agent.Agent{
  Name:  "Structured assistant",
  Model: model,
  ModelSettings: agent.ModelSettings{
    StructuredOutput: &llm.StructuredOutput{
      Name:        "answer",
      Description: "A concise answer",
      Schema:      schema,
      Strict:      true,
    },
  },
}
```

The same type can be set directly on `llm.ChatRequest.StructuredOutput`. An
explicit structured output takes precedence over a conflicting static
`response_format` in `WithExtraFields`.

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

`SlogTracer` does not attach raw prompts, responses, reasoning, tool arguments,
or tool outputs by default. Tool starts are logged at info with the tool name
and call ID regardless of content capture; turns, model calls, and tool
completions are logged at debug, and failures at error.

Raw trace content can contain secrets or personal data. Enable it explicitly
only when the log destination is trusted:

```go
tracer := agent.NewSlogTracer(slog.Default())
tracer.IncludeSensitiveData = true

runner := &agent.Runner{Tracer: tracer}
```

When enabled, run input, complete model requests and responses (including
assistant replies and reasoning), tool arguments, raw tool outputs, and final
output are attached to their lifecycle events. Configure the flag before
starting a run; a tracer may be called concurrently for parallel tools.

## Custom models and testing

`Model` is a one-method interface, which makes agents trivial to unit test —
no HTTP, no API key:

```go
type Model interface {
	Complete(ctx context.Context, req llm.ChatRequest) (*llm.ChatResponse, error)
}
```

Script a fake that returns tool calls, handoffs, or messages and assert on
the requests it received (see `run_test.go` for a reference implementation).
Any backend — a different vendor API, a local model, a rules engine — plugs
in the same way.

The `llm` package options include `llm.WithBaseURL` (point it at any
OpenAI-compatible server), `llm.WithAPIKey`, `llm.WithHTTPClient`, static
headers/query params, and constructor-level JSON extra fields for provider
extensions. Each client is attempted once and has a 60-second request timeout
by default. Use `llm.WithRequestTimeout` to change the client default; zero or
a negative duration disables it.

Set `ChatRequest.RequestTimeout` or `EmbeddingRequest.RequestTimeout` to
override that default for one logical request. A positive duration applies
separately to each provider attempt, zero inherits the client default, and a
negative duration disables the client timeout for that request. Agent runs set
the same override through `ModelSettings.RequestTimeout`:

```go
ctx, cancel := context.WithTimeout(ctx, 45*time.Second)
defer cancel()

assistant.ModelSettings.RequestTimeout = 10 * time.Second
result, err := agent.Run(ctx, assistant, "Help me troubleshoot this request.")
```

The package-created HTTP client has no separate `http.Client.Timeout`. A custom
client passed through `llm.WithHTTPClient` keeps its own timeout, so each
attempt's effective limit is the earliest of the caller context deadline, the
request override or client default, and the custom HTTP timeout.

Wrap clients in `llm.CircuitBreaker` to try configured fallbacks when a request
fails. Each provider gets a fresh request timeout bounded by the caller's
context. Caller cancellation, caller expiration, and request serialization
errors stop the chain immediately. HTTP 400, 412, 422, other unclassified 4xx
responses, and local response-size limit errors fall back without counting
against the primary. HTTP 401, 403, 404, 408, 413, 429, 5xx responses, transport
errors, provider timeouts, and unusable successful responses count against it.

Use `WithProviderID` or the provider configuration's `ProviderID` field with
`BreakerConfig.OnFailover` to observe errors that cause another provider to be
attempted. Routing directly to a fallback while the circuit is already open
does not emit an event. The callback runs synchronously outside the breaker
lock and must be safe for concurrent use:

```go
breaker := llm.NewCircuitBreakerWithConfig(llm.BreakerConfig{
	OnFailover: func(ctx context.Context, event llm.FailoverEvent) {
		slog.WarnContext(ctx, "Failing over LLM provider",
			"from", event.FromProvider,
			"from_index", event.FromProviderIndex,
			"to", event.ToProvider,
			"error", event.Error,
			"counted", event.CountedAgainstBreaker,
		)
	},
}, primary, fallback)
```

For embeddings, use `llm.NewEmbeddingClient`, `llm.NewVLLMEmbedding`, or
`llm.NewOpenRouterEmbedding`. `llm.EmbeddingCircuitBreaker` provides the same
failover behavior:

```go
primary := llm.NewVLLMEmbedding(llm.VLLMEmbeddingConfig{
	ProviderID: "vllm-primary",
	BaseURL:    "http://ai1-inference:8001/v1",
	Model:      "BAAI/bge-m3",
})
fallback := llm.NewOpenRouterEmbedding(llm.OpenRouterEmbeddingConfig{
	ProviderID: "openrouter-fallback",
	APIKey:     os.Getenv("OPENROUTER_API_KEY"),
	Model:      "baai/bge-m3",
})
embedder := llm.NewEmbeddingCircuitBreaker(primary, fallback)
response, err := embedder.Embed(ctx, llm.EmbeddingRequest{
	Input: []string{"first document", "second document"},
})
```

The library does not validate embedding-model compatibility. Callers must
configure every fallback to use the same model weights and embedding settings
as the primary. Equal vector dimensions and similar model names do not make
embeddings from different models compatible.

Reasoning models served with a reasoning parser (for example
`vllm serve ... --reasoning-parser deepseek_r1`) return their thinking in a
`reasoning` field (`reasoning_content` on compatible legacy endpoints). The
runner converts it into an `agent.Reasoning` item and never replays reasoning
items back to the model when converting history into requests.

Reasoning is disabled by default. Set a positive per-agent token budget to
enable it:

```go
assistant.ModelSettings.ReasoningTokenBudget = 256
```

The runner copies this value to `llm.ChatRequest.ReasoningTokenBudget` on every
turn. OpenRouter receives `reasoning.max_tokens`; vLLM receives
`thinking_token_budget` and an enabled chat template. A non-positive value
sends each provider's reasoning-off controls. Handoff targets and agents used
as tools use their own `ModelSettings`.

## Not ported (yet)

Streaming, sessions/memory, MCP servers, structured output types, handoff
input filters, `toolUseBehavior` (stop-on-first-tool and friends),
human-in-the-loop interruptions, OpenAI dashboard trace export, and the
Responses API transport. The `Model` interface is the seam for most of
these.
