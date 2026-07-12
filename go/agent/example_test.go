package agent_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/DavidNix/safeagent/go/agent"
	"github.com/DavidNix/safeagent/go/llm"
)

type modelFunc func(context.Context, llm.ChatRequest) (*llm.ChatResponse, error)

func (f modelFunc) Complete(ctx context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {
	return f(ctx, req)
}

func textResponse(content string) *llm.ChatResponse {
	return &llm.ChatResponse{
		Choices: []llm.ChatChoice{{
			Message:      llm.ChatMessage{Role: "assistant", Content: content},
			FinishReason: "stop",
		}},
	}
}

func toolResponse(id, name, arguments string) *llm.ChatResponse {
	return &llm.ChatResponse{
		Choices: []llm.ChatChoice{{
			Message: llm.ChatMessage{Role: "assistant", ToolCalls: []llm.ChatToolCall{{
				ID:   id,
				Type: "function",
				Function: llm.ChatFunctionCall{
					Name:      name,
					Arguments: arguments,
				},
			}}},
			FinishReason: "tool_calls",
		}},
	}
}

func ExampleRun() {
	greeter := &agent.Agent{
		Name:         "greeter",
		Instructions: "Greet the user concisely.",
		Model: modelFunc(func(_ context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {
			fmt.Println(req.Messages[0].Content)
			return textResponse("Hello, Ada!"), nil
		}),
	}

	result, err := agent.Run(context.Background(), greeter, "My name is Ada.")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(result.FinalOutput)

	// Output:
	// Greet the user concisely.
	// Hello, Ada!
}

func ExampleRunner_RunItems() {
	assistant := &agent.Agent{
		Name: "assistant",
		Model: modelFunc(func(_ context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {
			fmt.Printf("%s: %s\n", req.Messages[0].Role, req.Messages[0].Content)
			return textResponse("The earlier topic was Go."), nil
		}),
	}
	runner := &agent.Runner{}

	result, err := runner.RunItems(context.Background(), assistant, []agent.Item{
		agent.UserMessage("Let's discuss Go."),
		agent.AssistantMessage("Sure."),
		agent.UserMessage("What was the topic?"),
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(result.FinalOutput)

	// Output:
	// user: Let's discuss Go.
	// The earlier topic was Go.
}

func ExampleTool() {
	responses := []*llm.ChatResponse{
		toolResponse("call-1", "weather", `{"city":"Lisbon"}`),
		textResponse("It is sunny in Lisbon."),
	}
	weather := agent.Tool{
		Name:        "weather",
		Description: "Get the current weather for a city.",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
		OnInvoke: func(_ context.Context, _ *agent.RunContext, arguments string) (string, error) {
			var input struct {
				City string `json:"city"`
			}
			if err := json.Unmarshal([]byte(arguments), &input); err != nil {
				return "", err
			}
			return input.City + ": sunny", nil
		},
	}
	assistant := &agent.Agent{
		Name:  "assistant",
		Tools: []agent.Tool{weather},
		Model: modelFunc(func(_ context.Context, _ llm.ChatRequest) (*llm.ChatResponse, error) {
			response := responses[0]
			responses = responses[1:]
			return response, nil
		}),
	}

	result, err := agent.Run(context.Background(), assistant, "What is the weather in Lisbon?")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(result.FinalOutput)
	fmt.Println(result.Usage.Requests)

	// Output:
	// It is sunny in Lisbon.
	// 2
}

func ExampleHandoffTo() {
	spanish := &agent.Agent{
		Name: "spanish",
		Model: modelFunc(func(_ context.Context, _ llm.ChatRequest) (*llm.ChatResponse, error) {
			return textResponse("Hola, ¿en qué puedo ayudarte?"), nil
		}),
	}
	triage := &agent.Agent{
		Name:     "triage",
		Handoffs: []agent.Handoff{agent.HandoffTo(spanish)},
		Model: modelFunc(func(_ context.Context, _ llm.ChatRequest) (*llm.ChatResponse, error) {
			return toolResponse("handoff-1", "transfer_to_spanish", `{}`), nil
		}),
	}

	result, err := agent.Run(context.Background(), triage, "Necesito ayuda.")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(result.LastAgent.Name)
	fmt.Println(result.FinalOutput)

	// Output:
	// spanish
	// Hola, ¿en qué puedo ayudarte?
}

func ExampleAgent_AsTool() {
	summarizer := &agent.Agent{
		Name: "summarizer",
		Model: modelFunc(func(_ context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {
			return textResponse("A short summary of: " + req.Messages[0].Content), nil
		}),
	}
	tool := summarizer.AsTool("summarize", "Summarize the supplied text.")

	output, err := tool.OnInvoke(
		context.Background(),
		&agent.RunContext{},
		`{"input":"Go favors composition."}`,
	)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(output)

	// Output:
	// A short summary of: Go favors composition.
}

func ExampleInputGuardrail() {
	guarded := &agent.Agent{
		Name: "guarded",
		Model: modelFunc(func(_ context.Context, _ llm.ChatRequest) (*llm.ChatResponse, error) {
			return textResponse("This response is never reached."), nil
		}),
		InputGuardrails: []agent.InputGuardrail{{
			Name: "blocked topic",
			Execute: func(_ context.Context, _ *agent.RunContext, _ *agent.Agent, _ []agent.Item) (agent.GuardrailOutput, error) {
				return agent.GuardrailOutput{TripwireTriggered: true}, nil
			},
		}},
	}

	_, err := agent.Run(context.Background(), guarded, "blocked input")
	if tripwire, ok := errors.AsType[*agent.InputGuardrailTripwireTriggeredError](err); ok {
		fmt.Println(tripwire.Result.GuardrailName)
	}

	// Output:
	// blocked topic
}
