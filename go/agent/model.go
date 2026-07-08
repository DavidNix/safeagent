package agent

import (
	"context"
	"encoding/json"
)

// ToolDefinition is the serialized form of a tool sent to the model.
type ToolDefinition struct {
	Name        string
	Description string
	Parameters  json.RawMessage
	Strict      bool
}

// HandoffDefinition is the serialized form of a handoff sent to the model.
type HandoffDefinition struct {
	ToolName        string
	ToolDescription string
	InputSchema     json.RawMessage
}

// ModelRequest is a single request to a language model.
type ModelRequest struct {
	SystemInstructions string
	Input              []Item
	ModelSettings      ModelSettings
	Tools              []ToolDefinition
	Handoffs           []HandoffDefinition
}

// ModelResponse is the model output for a single request.
type ModelResponse struct {
	Output     []Item
	Usage      Usage
	ResponseID string
}

// Model is the interface for calling a language model.
type Model interface {
	// GetResponse returns a single response for the request.
	GetResponse(ctx context.Context, req ModelRequest) (*ModelResponse, error)
}
