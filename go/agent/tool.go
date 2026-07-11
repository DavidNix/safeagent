package agent

import (
	"context"
	"encoding/json"
	"time"
)

// Tool is a function tool the model can invoke during a run.
type Tool struct {
	// Name uniquely identifies the tool within an agent.
	Name string
	// Description tells the model when to use the tool.
	Description string
	// Parameters is the JSON schema for the tool arguments. Empty means an
	// object schema with no properties.
	Parameters json.RawMessage
	// Strict requests strict JSON schema validation from providers that
	// support it.
	Strict bool
	// OnInvoke executes the tool. Arguments is the raw JSON string produced
	// by the model. Implementations should honor ctx cancellation: when
	// Timeout is set, an invocation that ignores ctx leaks its goroutine
	// until it returns.
	OnInvoke func(ctx context.Context, rc *RunContext, arguments string) (string, error)

	// Timeout caps a single OnInvoke execution. Zero means no timeout.
	Timeout time.Duration
	// PropagateTimeout aborts the run with a *ToolTimeoutError when the
	// invocation times out, instead of converting the timeout into a
	// model-visible tool output.
	PropagateTimeout bool
	// OnTimeout overrides the model-visible output for a timed out
	// invocation. Nil uses the ToolTimeoutError message.
	OnTimeout func(ctx context.Context, rc *RunContext, err *ToolTimeoutError) string

	// PropagateErrors aborts the run with a *ToolCallError when OnInvoke
	// fails, instead of converting the error into a model-visible tool
	// output the model can react to.
	PropagateErrors bool
	// OnInvokeError overrides the model-visible output produced when
	// OnInvoke fails. Nil uses a default message that asks the model to try
	// again.
	OnInvokeError func(ctx context.Context, rc *RunContext, err error) string
}

var emptyObjectSchema = json.RawMessage(`{"type":"object","properties":{}}`)

func (t Tool) definition() ToolDefinition {
	params := t.Parameters
	if len(params) == 0 {
		params = emptyObjectSchema
	}
	return ToolDefinition{
		Name:        t.Name,
		Description: t.Description,
		Parameters:  params,
		Strict:      t.Strict,
	}
}
