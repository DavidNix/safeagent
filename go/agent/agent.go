package agent

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
)

// Agent is a configured LLM agent: instructions, a model, and the tools,
// handoffs, and guardrails available during a run.
type Agent struct {
	// Name identifies the agent in handoffs and results.
	Name string
	// Instructions is the static system prompt for the agent.
	Instructions string
	// InstructionsFunc, when set, resolves the system prompt dynamically and
	// takes precedence over Instructions.
	InstructionsFunc func(ctx context.Context, rc *RunContext, agent *Agent) (string, error)
	// HandoffDescription describes the agent when it is a handoff target.
	HandoffDescription string
	// Model produces responses for the agent.
	Model Model
	// ModelSettings tunes generation parameters.
	ModelSettings ModelSettings
	// Tools are the function tools available to the model.
	Tools []Tool
	// Handoffs are the agents this agent can delegate to.
	Handoffs []Handoff
	// InputGuardrails run against the run input before the first model call.
	InputGuardrails []InputGuardrail
	// OutputGuardrails run against the final output of the run.
	OutputGuardrails []OutputGuardrail
	// DisableToolChoiceReset keeps sending ModelSettings.ToolChoice on every
	// turn. By default any ToolChoice other than "none" is dropped from
	// requests once this agent has executed tool calls in the run, so a
	// forced tool choice cannot loop the run to MaxTurns.
	DisableToolChoiceReset bool
}

// AsTool exposes the agent as a function tool that runs the agent to
// completion on a string input and returns its final output. The nested
// run's usage is added to the parent run.
func (a *Agent) AsTool(name, description string) Tool {
	toolName := cmp.Or(name, toFunctionToolName(a.Name))
	return Tool{
		Name:        toolName,
		Description: description,
		Parameters:  json.RawMessage(`{"type":"object","properties":{"input":{"type":"string"}},"required":["input"]}`),
		OnInvoke: func(ctx context.Context, rc *RunContext, arguments string) (string, error) {
			var args struct {
				Input string `json:"input"`
			}
			if err := json.Unmarshal([]byte(arguments), &args); err != nil {
				return "", fmt.Errorf("parse input for agent tool %q: %w", toolName, err)
			}
			runner := &Runner{Context: rc.Context, Tracer: rc.tracer}
			result, err := runner.Run(ctx, a, args.Input)
			if err != nil {
				return "", fmt.Errorf("run agent %q as tool: %w", a.Name, err)
			}
			rc.AddUsage(result.Usage)
			return result.FinalOutput, nil
		},
	}
}

func (a *Agent) resolveInstructions(ctx context.Context, rc *RunContext) (string, error) {
	if a.InstructionsFunc != nil {
		instructions, err := a.InstructionsFunc(ctx, rc, a)
		if err != nil {
			return "", fmt.Errorf("resolve instructions for agent %q: %w", a.Name, err)
		}
		return instructions, nil
	}
	return a.Instructions, nil
}

func (a *Agent) toolByName(name string) (Tool, bool) {
	for _, tool := range a.Tools {
		if tool.Name == name {
			return tool, true
		}
	}
	return Tool{}, false
}

func (a *Agent) handoffByToolName(name string) (Handoff, bool) {
	for _, handoff := range a.Handoffs {
		if handoff.toolName() == name {
			return handoff, true
		}
	}
	return Handoff{}, false
}

func (a *Agent) toolDefinitions() []ToolDefinition {
	defs := make([]ToolDefinition, 0, len(a.Tools))
	for _, tool := range a.Tools {
		defs = append(defs, tool.definition())
	}
	return defs
}

func (a *Agent) handoffDefinitions() []HandoffDefinition {
	defs := make([]HandoffDefinition, 0, len(a.Handoffs))
	for _, handoff := range a.Handoffs {
		defs = append(defs, handoff.definition())
	}
	return defs
}

func (a *Agent) validate() error {
	if a.Model == nil {
		return &UserError{Message: fmt.Sprintf("agent %q has no model", a.Name)}
	}

	functionNames := make(map[string]struct{}, len(a.Tools)+len(a.Handoffs))
	for _, tool := range a.Tools {
		if tool.OnInvoke == nil {
			return &UserError{Message: fmt.Sprintf("agent %q tool %q has no OnInvoke callback", a.Name, tool.Name)}
		}
		if _, exists := functionNames[tool.Name]; exists {
			return &UserError{Message: fmt.Sprintf("agent %q has duplicate function name %q", a.Name, tool.Name)}
		}
		functionNames[tool.Name] = struct{}{}
	}
	for _, handoff := range a.Handoffs {
		if handoff.Agent == nil {
			return &UserError{Message: fmt.Sprintf("agent %q handoff %q has no target agent", a.Name, handoff.ToolName)}
		}
		name := handoff.toolName()
		if _, exists := functionNames[name]; exists {
			return &UserError{Message: fmt.Sprintf("agent %q has duplicate function name %q", a.Name, name)}
		}
		functionNames[name] = struct{}{}
	}
	for _, guardrail := range a.InputGuardrails {
		if guardrail.Execute == nil {
			return &UserError{Message: fmt.Sprintf("agent %q input guardrail %q has no Execute callback", a.Name, guardrail.Name)}
		}
	}
	for _, guardrail := range a.OutputGuardrails {
		if guardrail.Execute == nil {
			return &UserError{Message: fmt.Sprintf("agent %q output guardrail %q has no Execute callback", a.Name, guardrail.Name)}
		}
	}
	return nil
}
