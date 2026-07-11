package agent

import (
	"context"
	"encoding/json"

	"github.com/DavidNix/safeagent/llm"
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

// Model is the agent call-site interface for an OpenAI-compatible Chat
// Completions model.
type Model interface {
	// Complete returns a single Chat Completions response for the request.
	Complete(ctx context.Context, req llm.ChatRequest) (*llm.ChatResponse, error)
}
