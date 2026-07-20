package agent

import (
	"context"
	"log/slog"

	"github.com/DavidNix/safeagent/go/llm"
)

// Tracer observes the lifecycle of agent runs. Inject one via Runner.Tracer;
// a nil Tracer disables tracing. Implementations must be safe for concurrent
// use: ToolStarted and ToolEnded fire from parallel goroutines when the model
// issues multiple tool calls in one turn. Embed NoopTracer to implement only
// the methods you need.
type Tracer interface {
	// RunStarted fires once per run, before the first turn.
	RunStarted(ctx context.Context, agent *Agent, input []Item)
	// TurnStarted fires at the beginning of each turn. Turns count from 1.
	TurnStarted(ctx context.Context, agent *Agent, turn int)
	// ModelCallEnded fires after every model invocation, successful or not.
	ModelCallEnded(ctx context.Context, agent *Agent, req llm.ChatRequest, resp *llm.ChatResponse, err error)
	// ToolStarted fires before a function tool invocation. Handoff calls do
	// not fire tool events; see Handoff.
	ToolStarted(ctx context.Context, agent *Agent, call ToolCall)
	// ToolEnded fires after a function tool invocation with the raw result,
	// before any error recovery converts a failure into model-visible output.
	ToolEnded(ctx context.Context, agent *Agent, call ToolCall, output string, err error)
	// Handoff fires when control transfers from one agent to another.
	Handoff(ctx context.Context, from *Agent, to *Agent)
	// RunEnded fires once per run with the final result or error.
	RunEnded(ctx context.Context, result *RunResult, err error)
}

// NoopTracer implements Tracer with no-op methods. Embed it in a custom
// tracer to override only the events you care about.
type NoopTracer struct{}

// RunStarted implements Tracer.
func (NoopTracer) RunStarted(context.Context, *Agent, []Item) {}

// TurnStarted implements Tracer.
func (NoopTracer) TurnStarted(context.Context, *Agent, int) {}

// ModelCallEnded implements Tracer.
func (NoopTracer) ModelCallEnded(context.Context, *Agent, llm.ChatRequest, *llm.ChatResponse, error) {
}

// ToolStarted implements Tracer.
func (NoopTracer) ToolStarted(context.Context, *Agent, ToolCall) {}

// ToolEnded implements Tracer.
func (NoopTracer) ToolEnded(context.Context, *Agent, ToolCall, string, error) {}

// Handoff implements Tracer.
func (NoopTracer) Handoff(context.Context, *Agent, *Agent) {}

// RunEnded implements Tracer.
func (NoopTracer) RunEnded(context.Context, *RunResult, error) {}

// SlogTracer logs run lifecycle events to a slog.Logger: run boundaries,
// handoffs, and tool starts at info, turns, model calls, and tool completions
// at debug, and failures at error.
type SlogTracer struct {
	// IncludeSensitiveData enables logging prompts, model responses, reasoning,
	// tool arguments, and tool outputs. Configure it before starting a run.
	IncludeSensitiveData bool
	logger               *slog.Logger
}

// NewSlogTracer builds a SlogTracer that logs to logger. The logger must not
// be nil.
func NewSlogTracer(logger *slog.Logger) *SlogTracer {
	return &SlogTracer{logger: logger}
}

// RunStarted implements Tracer.
func (t *SlogTracer) RunStarted(ctx context.Context, agent *Agent, input []Item) {
	attrs := []any{"agent", agent.Name, "input_items", len(input)}
	if t.IncludeSensitiveData {
		attrs = append(attrs, "input", input)
	}
	t.logger.InfoContext(ctx, "Run started", attrs...)
}

// TurnStarted implements Tracer.
func (t *SlogTracer) TurnStarted(ctx context.Context, agent *Agent, turn int) {
	t.logger.DebugContext(ctx, "Turn started", "agent", agent.Name, "turn", turn)
}

// ModelCallEnded implements Tracer.
func (t *SlogTracer) ModelCallEnded(ctx context.Context, agent *Agent, req llm.ChatRequest, resp *llm.ChatResponse, err error) {
	attrs := []any{"agent", agent.Name}
	if t.IncludeSensitiveData {
		attrs = append(attrs, "request", req, "response", resp)
	}
	if err != nil {
		attrs = append(attrs, "error", err)
		t.logger.ErrorContext(ctx, "Model call failed", attrs...)
		return
	}
	attrs = append(attrs,
		"choices", len(resp.Choices),
		"input_tokens", resp.Usage.PromptTokens,
		"output_tokens", resp.Usage.CompletionTokens,
	)
	t.logger.DebugContext(ctx, "Model call completed", attrs...)
}

// ToolStarted implements Tracer.
func (t *SlogTracer) ToolStarted(ctx context.Context, agent *Agent, call ToolCall) {
	attrs := []any{"agent", agent.Name, "tool", call.Name, "call_id", call.ID}
	if t.IncludeSensitiveData {
		attrs = append(attrs, "arguments", call.Arguments)
	}
	t.logger.InfoContext(ctx, "Tool started", attrs...)
}

// ToolEnded implements Tracer.
func (t *SlogTracer) ToolEnded(ctx context.Context, agent *Agent, call ToolCall, output string, err error) {
	attrs := []any{"agent", agent.Name, "tool", call.Name, "call_id", call.ID}
	if t.IncludeSensitiveData {
		attrs = append(attrs, "output", output)
	}
	if err != nil {
		attrs = append(attrs, "error", err)
		t.logger.ErrorContext(ctx, "Tool failed", attrs...)
		return
	}
	attrs = append(attrs, "output_bytes", len(output))
	t.logger.DebugContext(ctx, "Tool completed", attrs...)
}

// Handoff implements Tracer.
func (t *SlogTracer) Handoff(ctx context.Context, from *Agent, to *Agent) {
	t.logger.InfoContext(ctx, "Handoff", "from", from.Name, "to", to.Name)
}

// RunEnded implements Tracer.
func (t *SlogTracer) RunEnded(ctx context.Context, result *RunResult, err error) {
	if err != nil {
		attrs := []any{"error", err}
		if t.IncludeSensitiveData && result != nil {
			attrs = append(attrs, "final_output", result.FinalOutput)
		}
		t.logger.ErrorContext(ctx, "Run failed", attrs...)
		return
	}
	attrs := []any{
		"agent", result.LastAgent.Name,
		"new_items", len(result.NewItems),
		"requests", result.Usage.Requests,
		"total_tokens", result.Usage.TotalTokens,
	}
	if t.IncludeSensitiveData {
		attrs = append(attrs, "final_output", result.FinalOutput)
	}
	t.logger.InfoContext(ctx, "Run completed", attrs...)
}
