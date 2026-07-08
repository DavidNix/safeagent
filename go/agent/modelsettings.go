package agent

// ModelSettings holds tunable generation parameters passed to the model.
// Nil pointer fields are omitted from provider requests.
type ModelSettings struct {
	Temperature       *float64
	TopP              *float64
	MaxTokens         *int
	ParallelToolCalls *bool
	// ToolChoice is "auto", "required", "none", or the name of a specific
	// tool the model must call.
	ToolChoice string
}
