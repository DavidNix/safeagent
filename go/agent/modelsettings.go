package agent

import (
	"time"

	"github.com/DavidNix/safeagent/go/llm"
)

// ModelSettings holds request and generation parameters passed to the model.
// Nil pointer fields are omitted from provider requests.
type ModelSettings struct {
	// RequestTimeout overrides the model client's timeout for each provider
	// attempt. Zero uses the client default; a negative duration disables it.
	RequestTimeout time.Duration
	// Temperature controls output randomness when supported by the provider.
	Temperature *float64
	// TopP controls nucleus sampling when supported by the provider.
	TopP *float64
	// MaxTokens limits the number of tokens generated for a completion.
	MaxTokens *int
	// ReasoningTokenBudget enables reasoning when greater than zero and limits
	// the number of reasoning tokens when supported by the provider.
	ReasoningTokenBudget int
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
