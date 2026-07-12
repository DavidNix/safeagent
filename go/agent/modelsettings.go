package agent

import "github.com/DavidNix/safeagent/go/llm"

// ModelSettings holds tunable generation parameters passed to the model.
// Nil pointer fields are omitted from provider requests.
type ModelSettings struct {
	// Temperature controls output randomness when supported by the provider.
	Temperature *float64
	// TopP controls nucleus sampling when supported by the provider.
	TopP *float64
	// MaxTokens limits the number of tokens generated for a completion.
	MaxTokens *int
	// ParallelToolCalls controls whether the model may request tools in parallel.
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
