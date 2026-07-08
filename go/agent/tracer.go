package agent

import (
	"context"
	"log/slog"
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
	ModelCallEnded(ctx context.Context, agent *Agent, req ModelRequest, resp *ModelResponse, err error)
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
func (NoopTracer) ModelCallEnded(context.Context, *Agent, ModelRequest, *ModelResponse, error) {}

// ToolStarted implements Tracer.
func (NoopTracer) ToolStarted(context.Context, *Agent, ToolCall) {}

// ToolEnded implements Tracer.
func (NoopTracer) ToolEnded(context.Context, *Agent, ToolCall, string, error) {}

// Handoff implements Tracer.
func (NoopTracer) Handoff(context.Context, *Agent, *Agent) {}

// RunEnded implements Tracer.
func (NoopTracer) RunEnded(context.Context, *RunResult, error) {}

// SlogTracer logs run lifecycle events to a slog.Logger: run boundaries and
// handoffs at info, turns, model calls, and tool activity at debug, and
// failures at error.
type SlogTracer struct {
	logger *slog.Logger
}

// NewSlogTracer builds a SlogTracer that logs to logger. The logger must not
// be nil.
func NewSlogTracer(logger *slog.Logger) *SlogTracer {
	return &SlogTracer{logger: logger}
}

// RunStarted implements Tracer.
func (t *SlogTracer) RunStarted(ctx context.Context, agent *Agent, input []Item) {
	t.logger.InfoContext(ctx, "Run started", "agent", agent.Name, "input_items", len(input))
}

// TurnStarted implements Tracer.
func (t *SlogTracer) TurnStarted(ctx context.Context, agent *Agent, turn int) {
	t.logger.DebugContext(ctx, "Turn started", "agent", agent.Name, "turn", turn)
}

// ModelCallEnded implements Tracer.
func (t *SlogTracer) ModelCallEnded(ctx context.Context, agent *Agent, _ ModelRequest, resp *ModelResponse, err error) {
	if err != nil {
		t.logger.ErrorContext(ctx, "Model call failed", "agent", agent.Name, "error", err)
		return
	}
	t.logger.DebugContext(ctx, "Model call completed",
		"agent", agent.Name,
		"output_items", len(resp.Output),
		"input_tokens", resp.Usage.InputTokens,
		"output_tokens", resp.Usage.OutputTokens,
	)
}

// ToolStarted implements Tracer.
func (t *SlogTracer) ToolStarted(ctx context.Context, agent *Agent, call ToolCall) {
	t.logger.DebugContext(ctx, "Tool started", "agent", agent.Name, "tool", call.Name, "call_id", call.ID)
}

// ToolEnded implements Tracer.
func (t *SlogTracer) ToolEnded(ctx context.Context, agent *Agent, call ToolCall, output string, err error) {
	if err != nil {
		t.logger.ErrorContext(ctx, "Tool failed", "agent", agent.Name, "tool", call.Name, "call_id", call.ID, "error", err)
		return
	}
	t.logger.DebugContext(ctx, "Tool completed", "agent", agent.Name, "tool", call.Name, "call_id", call.ID, "output_bytes", len(output))
}

// Handoff implements Tracer.
func (t *SlogTracer) Handoff(ctx context.Context, from *Agent, to *Agent) {
	t.logger.InfoContext(ctx, "Handoff", "from", from.Name, "to", to.Name)
}

// RunEnded implements Tracer.
func (t *SlogTracer) RunEnded(ctx context.Context, result *RunResult, err error) {
	if err != nil {
		t.logger.ErrorContext(ctx, "Run failed", "error", err)
		return
	}
	t.logger.InfoContext(ctx, "Run completed",
		"agent", result.LastAgent.Name,
		"new_items", len(result.NewItems),
		"requests", result.Usage.Requests,
		"total_tokens", result.Usage.TotalTokens,
	)
}
