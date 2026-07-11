package agent

import "context"

// GuardrailOutput is the decision produced by a guardrail check.
type GuardrailOutput struct {
	// TripwireTriggered halts the run when true.
	TripwireTriggered bool
	// OutputInfo carries optional details about the checks performed.
	OutputInfo any
}

// InputGuardrail validates the run input before the first model call.
type InputGuardrail struct {
	Name    string
	Execute func(ctx context.Context, rc *RunContext, agent *Agent, input []Item) (GuardrailOutput, error)
}

// OutputGuardrail validates the final output of a run.
type OutputGuardrail struct {
	Name    string
	Execute func(ctx context.Context, rc *RunContext, agent *Agent, output string) (GuardrailOutput, error)
}

// InputGuardrailResult pairs an input guardrail with its output.
type InputGuardrailResult struct {
	GuardrailName string
	Output        GuardrailOutput
}

// OutputGuardrailResult pairs an output guardrail with its output.
type OutputGuardrailResult struct {
	GuardrailName string
	Output        GuardrailOutput
}
