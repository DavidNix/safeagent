package agent

import (
	"context"
	"fmt"
	"slices"

	"golang.org/x/sync/errgroup"
)

// DefaultMaxTurns is the turn budget used when Runner.MaxTurns is zero.
const DefaultMaxTurns = 10

// Runner executes agent runs.
type Runner struct {
	// MaxTurns caps model invocations per run. Zero means DefaultMaxTurns.
	MaxTurns int
	// Context is an arbitrary value passed to tools and guardrails via
	// RunContext.
	Context any
}

// Run executes agent with a plain text user input using a default Runner.
func Run(ctx context.Context, agent *Agent, input string) (*RunResult, error) {
	runner := &Runner{}
	return runner.Run(ctx, agent, input)
}

// Run executes agent with a plain text user input until the agent produces a
// final output. It returns an error when the run exceeds MaxTurns, a
// guardrail tripwire triggers, a tool fails, or the model misbehaves.
func (r *Runner) Run(ctx context.Context, agent *Agent, input string) (*RunResult, error) {
	return r.RunItems(ctx, agent, []Item{UserMessage(input)})
}

// RunItems executes agent with a full input item list. See Run.
func (r *Runner) RunItems(ctx context.Context, agent *Agent, input []Item) (*RunResult, error) {
	if agent == nil {
		return nil, &UserError{Message: "agent is nil"}
	}
	maxTurns := r.MaxTurns
	if maxTurns <= 0 {
		maxTurns = DefaultMaxTurns
	}
	rc := &RunContext{Context: r.Context}
	history := make([]Item, len(input))
	copy(history, input)
	result := &RunResult{LastAgent: agent}
	current := agent

	inputResults, err := runInputGuardrails(ctx, rc, current, input)
	result.InputGuardrailResults = inputResults
	if err != nil {
		return nil, err
	}

	for turn := 1; turn <= maxTurns; turn++ {
		if current.Model == nil {
			return nil, &UserError{Message: fmt.Sprintf("agent %q has no model", current.Name)}
		}
		instructions, err := current.resolveInstructions(ctx, rc)
		if err != nil {
			return nil, err
		}
		req := ModelRequest{
			SystemInstructions: instructions,
			Input:              history,
			ModelSettings:      current.ModelSettings,
			Tools:              current.toolDefinitions(),
			Handoffs:           current.handoffDefinitions(),
		}
		resp, err := current.Model.GetResponse(ctx, req)
		if err != nil {
			return nil, fmt.Errorf("model response for agent %q: %w", current.Name, err)
		}
		rc.AddUsage(resp.Usage)
		history = append(history, resp.Output...)
		result.NewItems = append(result.NewItems, resp.Output...)

		next, outputs, err := processTurn(ctx, rc, current, resp.Output)
		if err != nil {
			return nil, err
		}
		history = append(history, outputs...)
		result.NewItems = append(result.NewItems, outputs...)
		if next != nil {
			current = next
			result.LastAgent = current
			continue
		}
		if len(outputs) > 0 {
			continue
		}

		result.FinalOutput = lastAssistantText(resp.Output)
		result.History = history
		result.Usage = rc.Usage()
		outputResults, err := runOutputGuardrails(ctx, rc, current, result.FinalOutput)
		result.OutputGuardrailResults = outputResults
		if err != nil {
			return nil, err
		}
		return result, nil
	}
	return nil, &MaxTurnsExceededError{MaxTurns: maxTurns}
}

type pendingToolCall struct {
	call ToolCall
	tool Tool
}

// processTurn executes the tool calls in a model response. It returns the
// next agent when a handoff triggered and the tool output items to append to
// the history. Empty outputs and a nil next agent mean the turn is final.
func processTurn(ctx context.Context, rc *RunContext, agent *Agent, output []Item) (*Agent, []Item, error) {
	var pending []pendingToolCall
	var handoff *Handoff
	var handoffCall ToolCall
	var rejectedHandoffs []Item

	for _, item := range output {
		call, ok := item.(ToolCall)
		if !ok {
			continue
		}
		if h, found := agent.handoffByToolName(call.Name); found {
			if handoff == nil {
				handoff = &h
				handoffCall = call
				continue
			}
			rejectedHandoffs = append(rejectedHandoffs, ToolOutput{
				CallID: call.ID,
				Name:   call.Name,
				Output: "Multiple handoffs detected, ignoring this one.",
			})
			continue
		}
		tool, found := agent.toolByName(call.Name)
		if !found {
			return nil, nil, &ModelBehaviorError{
				Message: fmt.Sprintf("tool %q not found on agent %q", call.Name, agent.Name),
			}
		}
		pending = append(pending, pendingToolCall{call: call, tool: tool})
	}

	toolOutputs := make([]Item, len(pending))
	g, gctx := errgroup.WithContext(ctx)
	for i, p := range pending {
		g.Go(func() error {
			out, err := p.tool.OnInvoke(gctx, rc, p.call.Arguments)
			if err != nil {
				return &ToolCallError{ToolName: p.call.Name, Err: err}
			}
			toolOutputs[i] = ToolOutput{CallID: p.call.ID, Name: p.call.Name, Output: out}
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return nil, nil, err
	}

	outputs := append(toolOutputs, rejectedHandoffs...)
	if handoff == nil {
		return nil, outputs, nil
	}
	if handoff.OnHandoff != nil {
		if err := handoff.OnHandoff(ctx, rc, handoffCall.Arguments); err != nil {
			return nil, nil, fmt.Errorf("handoff to agent %q: %w", handoff.Agent.Name, err)
		}
	}
	outputs = append(outputs, ToolOutput{
		CallID: handoffCall.ID,
		Name:   handoffCall.Name,
		Output: fmt.Sprintf(`{"assistant":%q}`, handoff.Agent.Name),
	})
	return handoff.Agent, outputs, nil
}

func runInputGuardrails(ctx context.Context, rc *RunContext, agent *Agent, input []Item) ([]InputGuardrailResult, error) {
	if len(agent.InputGuardrails) == 0 {
		return nil, nil
	}
	results := make([]InputGuardrailResult, len(agent.InputGuardrails))
	g, gctx := errgroup.WithContext(ctx)
	for i, guardrail := range agent.InputGuardrails {
		g.Go(func() error {
			output, err := guardrail.Execute(gctx, rc, agent, input)
			if err != nil {
				return &GuardrailExecutionError{GuardrailName: guardrail.Name, Err: err}
			}
			results[i] = InputGuardrailResult{GuardrailName: guardrail.Name, Output: output}
			if output.TripwireTriggered {
				return &InputGuardrailTripwireTriggeredError{Result: results[i]}
			}
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return results, err
	}
	return results, nil
}

func runOutputGuardrails(ctx context.Context, rc *RunContext, agent *Agent, finalOutput string) ([]OutputGuardrailResult, error) {
	if len(agent.OutputGuardrails) == 0 {
		return nil, nil
	}
	results := make([]OutputGuardrailResult, len(agent.OutputGuardrails))
	g, gctx := errgroup.WithContext(ctx)
	for i, guardrail := range agent.OutputGuardrails {
		g.Go(func() error {
			output, err := guardrail.Execute(gctx, rc, agent, finalOutput)
			if err != nil {
				return &GuardrailExecutionError{GuardrailName: guardrail.Name, Err: err}
			}
			results[i] = OutputGuardrailResult{GuardrailName: guardrail.Name, Output: output}
			if output.TripwireTriggered {
				return &OutputGuardrailTripwireTriggeredError{Result: results[i]}
			}
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return results, err
	}
	return results, nil
}

func lastAssistantText(output []Item) string {
	for _, o := range slices.Backward(output) {
		msg, ok := o.(Message)
		if ok && msg.Role == RoleAssistant {
			return msg.Content
		}
	}
	return ""
}
