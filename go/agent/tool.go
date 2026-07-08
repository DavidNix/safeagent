package agent

import (
	"context"
	"encoding/json"
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
	// by the model.
	OnInvoke func(ctx context.Context, rc *RunContext, arguments string) (string, error)
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
