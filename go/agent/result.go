package agent

// RunResult is the outcome of an agent run.
type RunResult struct {
	// FinalOutput is the text of the final assistant message.
	FinalOutput string
	// NewItems are the items generated during the run: model output and tool
	// outputs, in order.
	NewItems []Item
	// History is the full conversation: the run input followed by NewItems.
	History []Item
	// LastAgent is the agent that produced the final output.
	LastAgent *Agent
	// Usage is the total model usage across the run.
	Usage Usage
	// InputGuardrailResults are the results of the input guardrails that ran.
	InputGuardrailResults []InputGuardrailResult
	// OutputGuardrailResults are the results of the output guardrails that ran.
	OutputGuardrailResults []OutputGuardrailResult
}
