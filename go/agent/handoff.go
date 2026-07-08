package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// Handoff lets an agent delegate the remainder of a run to another agent.
// The model sees the handoff as a callable tool.
type Handoff struct {
	// Agent receives control when the handoff triggers.
	Agent *Agent
	// ToolName overrides the default transfer_to_<agent> tool name.
	ToolName string
	// ToolDescription overrides the default tool description.
	ToolDescription string
	// InputSchema is the JSON schema of the handoff input, if any.
	InputSchema json.RawMessage
	// OnHandoff, when set, is called with the raw tool arguments before
	// control transfers.
	OnHandoff func(ctx context.Context, rc *RunContext, input string) error
}

// HandoffTo builds a handoff to agent with the default tool name and
// description.
func HandoffTo(agent *Agent) Handoff {
	return Handoff{Agent: agent}
}

func (h Handoff) toolName() string {
	if h.ToolName != "" {
		return h.ToolName
	}
	return "transfer_to_" + toFunctionToolName(h.Agent.Name)
}

func (h Handoff) toolDescription() string {
	if h.ToolDescription != "" {
		return h.ToolDescription
	}
	desc := fmt.Sprintf("Handoff to the %s agent to handle the request.", h.Agent.Name)
	if h.Agent.HandoffDescription != "" {
		desc += " " + h.Agent.HandoffDescription
	}
	return desc
}

func (h Handoff) definition() HandoffDefinition {
	schema := h.InputSchema
	if len(schema) == 0 {
		schema = emptyObjectSchema
	}
	return HandoffDefinition{
		ToolName:        h.toolName(),
		ToolDescription: h.toolDescription(),
		InputSchema:     schema,
	}
}

func toFunctionToolName(name string) string {
	var b strings.Builder
	for _, r := range name {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '_', r == '-':
			b.WriteRune(r)
		default:
			b.WriteRune('_')
		}
	}
	return b.String()
}
