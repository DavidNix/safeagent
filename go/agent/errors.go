package agent

import "fmt"

// MaxTurnsExceededError reports that a run exceeded its turn budget.
type MaxTurnsExceededError struct {
	MaxTurns int
}

func (e *MaxTurnsExceededError) Error() string {
	return fmt.Sprintf("max turns (%d) exceeded", e.MaxTurns)
}

// ModelBehaviorError reports unexpected model behavior, such as calling a
// tool that does not exist.
type ModelBehaviorError struct {
	Message string
}

func (e *ModelBehaviorError) Error() string {
	return "model behavior error: " + e.Message
}

// UserError reports a misconfiguration by the library user.
type UserError struct {
	Message string
}

func (e *UserError) Error() string {
	return e.Message
}

// ToolCallError reports a failed tool invocation.
type ToolCallError struct {
	ToolName string
	Err      error
}

func (e *ToolCallError) Error() string {
	return fmt.Sprintf("tool %q failed: %v", e.ToolName, e.Err)
}

// Unwrap returns the underlying tool error.
func (e *ToolCallError) Unwrap() error {
	return e.Err
}

// GuardrailExecutionError reports a guardrail that failed to execute.
type GuardrailExecutionError struct {
	GuardrailName string
	Err           error
}

func (e *GuardrailExecutionError) Error() string {
	return fmt.Sprintf("guardrail %q failed: %v", e.GuardrailName, e.Err)
}

// Unwrap returns the underlying guardrail error.
func (e *GuardrailExecutionError) Unwrap() error {
	return e.Err
}

// InputGuardrailTripwireTriggeredError reports that an input guardrail
// tripwire halted the run.
type InputGuardrailTripwireTriggeredError struct {
	Result InputGuardrailResult
}

func (e *InputGuardrailTripwireTriggeredError) Error() string {
	return fmt.Sprintf("input guardrail %q tripwire triggered", e.Result.GuardrailName)
}

// OutputGuardrailTripwireTriggeredError reports that an output guardrail
// tripwire halted the run.
type OutputGuardrailTripwireTriggeredError struct {
	Result OutputGuardrailResult
}

func (e *OutputGuardrailTripwireTriggeredError) Error() string {
	return fmt.Sprintf("output guardrail %q tripwire triggered", e.Result.GuardrailName)
}
