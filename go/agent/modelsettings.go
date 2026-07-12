package agent

import "github.com/DavidNix/safeagent/llm"

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
	// StructuredOutput constrains model output to an OpenAI-compatible JSON
	// Schema. The final output remains a JSON string.
	//
	// Example:
	//
	//	settings := agent.ModelSettings{
	//		StructuredOutput: &llm.StructuredOutput{
	//			Name:   "answer",
	//			Schema: json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}`),
	//			Strict: true,
	//		},
	//	}
	StructuredOutput *llm.StructuredOutput
}
