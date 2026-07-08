package agent_test

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/DavidNix/safeagent/agent"
	"github.com/DavidNix/safeagent/llm"
	"github.com/stretchr/testify/require"
)

type fakeModel struct {
	mu          sync.Mutex
	responses   []fakeModelResponse
	requests    []capturedModelRequest
	rawRequests []llm.ChatRequest
}

type fakeModelResponse struct {
	Output     []agent.Item
	Usage      agent.Usage
	ResponseID string
	Raw        *llm.ChatResponse
}

type capturedModelRequest struct {
	SystemInstructions string
	Input              []agent.Item
	ModelSettings      agent.ModelSettings
	Tools              []agent.ToolDefinition
	Handoffs           []agent.HandoffDefinition
}

func (m *fakeModel) Complete(_ context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.rawRequests = append(m.rawRequests, cloneChatRequest(req))
	m.requests = append(m.requests, captureModelRequest(req))
	if len(m.responses) == 0 {
		return nil, errors.New("fake model has no scripted responses")
	}
	resp := m.responses[0]
	m.responses = m.responses[1:]
	if resp.Raw != nil {
		return resp.Raw, nil
	}
	return chatResponseFromItems(resp), nil
}

func cloneChatRequest(req llm.ChatRequest) llm.ChatRequest {
	req.Messages = append([]llm.ChatMessage(nil), req.Messages...)
	req.Tools = append([]llm.ChatTool(nil), req.Tools...)
	return req
}

func captureModelRequest(req llm.ChatRequest) capturedModelRequest {
	captured := capturedModelRequest{
		ModelSettings: agent.ModelSettings{
			Temperature:       req.Temperature,
			TopP:              req.TopP,
			MaxTokens:         req.MaxTokens,
			ParallelToolCalls: req.ParallelToolCalls,
			ToolChoice:        toolChoiceString(req.ToolChoice),
		},
	}
	toolNamesByCallID := map[string]string{}
	for _, msg := range req.Messages {
		switch msg.Role {
		case "system":
			if captured.SystemInstructions == "" {
				captured.SystemInstructions = msg.Content
				continue
			}
			captured.Input = append(captured.Input, agent.SystemMessage(msg.Content))
		case "user", "assistant":
			if msg.Content != "" {
				captured.Input = append(captured.Input, agent.Message{Role: agent.Role(msg.Role), Content: msg.Content})
			}
			for _, call := range msg.ToolCalls {
				toolNamesByCallID[call.ID] = call.Function.Name
				captured.Input = append(captured.Input, agent.ToolCall{
					ID:        call.ID,
					Name:      call.Function.Name,
					Arguments: call.Function.Arguments,
				})
			}
		case "tool":
			captured.Input = append(captured.Input, agent.ToolOutput{
				CallID: msg.ToolCallID,
				Name:   toolNamesByCallID[msg.ToolCallID],
				Output: msg.Content,
			})
		}
	}
	for _, tool := range req.Tools {
		if strings.HasPrefix(tool.Function.Name, "transfer_to_") {
			captured.Handoffs = append(captured.Handoffs, agent.HandoffDefinition{
				ToolName:        tool.Function.Name,
				ToolDescription: tool.Function.Description,
				InputSchema:     tool.Function.Parameters,
			})
			continue
		}
		captured.Tools = append(captured.Tools, agent.ToolDefinition{
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
			Parameters:  tool.Function.Parameters,
			Strict:      tool.Function.Strict,
		})
	}
	return captured
}

func toolChoiceString(choice any) string {
	s, ok := choice.(string)
	if ok {
		return s
	}
	choiceMap, ok := choice.(map[string]any)
	if !ok {
		return ""
	}
	function, ok := choiceMap["function"].(map[string]any)
	if !ok {
		return ""
	}
	name, _ := function["name"].(string)
	return name
}

func chatResponseFromItems(resp fakeModelResponse) *llm.ChatResponse {
	choice := llm.ChatChoice{FinishReason: "stop"}
	for _, item := range resp.Output {
		switch v := item.(type) {
		case agent.Message:
			if v.Role == agent.RoleAssistant {
				choice.Message.Content = v.Content
			}
		case agent.Reasoning:
			choice.Message.ReasoningContent = v.Content
		case agent.ToolCall:
			choice.FinishReason = "tool_calls"
			choice.Message.ToolCalls = append(choice.Message.ToolCalls, llm.ChatToolCall{
				ID:       v.ID,
				Type:     "function",
				Function: llm.ChatFunctionCall{Name: v.Name, Arguments: v.Arguments},
			})
		}
	}
	return &llm.ChatResponse{
		ID:      resp.ResponseID,
		Choices: []llm.ChatChoice{choice},
		Usage: llm.ChatUsage{
			PromptTokens:     resp.Usage.InputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		},
	}
}

func TestRunner_Run(t *testing.T) {
	t.Parallel()

	t.Run("happy path", func(t *testing.T) {
		model := &fakeModel{responses: []fakeModelResponse{
			{
				Output: []agent.Item{agent.AssistantMessage("Hello!")},
				Usage:  agent.Usage{Requests: 1, InputTokens: 10, OutputTokens: 5, TotalTokens: 15},
			},
		}}
		ag := &agent.Agent{Name: "assistant", Instructions: "Be helpful", Model: model}

		runner := &agent.Runner{}
		result, err := runner.Run(t.Context(), ag, "Hi")

		require.NoError(t, err)
		require.Equal(t, "Hello!", result.FinalOutput)
		require.Same(t, ag, result.LastAgent)
		require.Equal(t, agent.Usage{Requests: 1, InputTokens: 10, OutputTokens: 5, TotalTokens: 15}, result.Usage)
		require.Equal(t, []agent.Item{agent.AssistantMessage("Hello!")}, result.NewItems)
		require.Equal(t, []agent.Item{agent.UserMessage("Hi"), agent.AssistantMessage("Hello!")}, result.History)
		require.Len(t, model.requests, 1)
		require.Equal(t, "Be helpful", model.requests[0].SystemInstructions)
		require.Equal(t, []agent.Item{agent.UserMessage("Hi")}, model.requests[0].Input)
	})

	t.Run("tool call loop", func(t *testing.T) {
		var gotArgs string
		var gotContext any
		tool := agent.Tool{
			Name:        "get_weather",
			Description: "Get the weather for a city",
			OnInvoke: func(_ context.Context, rc *agent.RunContext, arguments string) (string, error) {
				gotArgs = arguments
				gotContext = rc.Context
				return "sunny", nil
			},
		}
		model := &fakeModel{responses: []fakeModelResponse{
			{
				Output: []agent.Item{agent.ToolCall{ID: "call_1", Name: "get_weather", Arguments: `{"city":"SF"}`}},
				Usage:  agent.Usage{Requests: 1, InputTokens: 10, OutputTokens: 3, TotalTokens: 13},
			},
			{
				Output: []agent.Item{agent.AssistantMessage("It is sunny.")},
				Usage:  agent.Usage{Requests: 1, InputTokens: 20, OutputTokens: 4, TotalTokens: 24},
			},
		}}
		ag := &agent.Agent{Name: "assistant", Model: model, Tools: []agent.Tool{tool}}

		runner := &agent.Runner{Context: "user-ctx"}
		result, err := runner.Run(t.Context(), ag, "Weather in SF?")

		require.NoError(t, err)
		require.Equal(t, "It is sunny.", result.FinalOutput)
		require.Equal(t, `{"city":"SF"}`, gotArgs)
		require.Equal(t, "user-ctx", gotContext)
		require.Equal(t, agent.Usage{Requests: 2, InputTokens: 30, OutputTokens: 7, TotalTokens: 37}, result.Usage)
		require.Len(t, model.requests, 2)
		require.Equal(t, "get_weather", model.requests[0].Tools[0].Name)
		require.JSONEq(t, `{"type":"object","properties":{}}`, string(model.requests[0].Tools[0].Parameters))
		require.Contains(t, model.requests[1].Input, agent.Item(agent.ToolOutput{CallID: "call_1", Name: "get_weather", Output: "sunny"}))
	})

	t.Run("handoff", func(t *testing.T) {
		spanishModel := &fakeModel{responses: []fakeModelResponse{
			{Output: []agent.Item{agent.AssistantMessage("¡Hola!")}, Usage: agent.Usage{Requests: 1}},
		}}
		spanish := &agent.Agent{Name: "Spanish agent", Model: spanishModel, HandoffDescription: "Speaks Spanish."}

		triageModel := &fakeModel{responses: []fakeModelResponse{
			{
				Output: []agent.Item{agent.ToolCall{ID: "call_1", Name: "transfer_to_Spanish_agent", Arguments: "{}"}},
				Usage:  agent.Usage{Requests: 1},
			},
		}}
		var handoffInput string
		handoff := agent.HandoffTo(spanish)
		handoff.OnHandoff = func(_ context.Context, _ *agent.RunContext, input string) error {
			handoffInput = input
			return nil
		}
		triage := &agent.Agent{Name: "Triage", Model: triageModel, Handoffs: []agent.Handoff{handoff}}

		runner := &agent.Runner{}
		result, err := runner.Run(t.Context(), triage, "Hola")

		require.NoError(t, err)
		require.Same(t, spanish, result.LastAgent)
		require.Equal(t, "¡Hola!", result.FinalOutput)
		require.Equal(t, "{}", handoffInput)
		require.Contains(t, result.NewItems, agent.Item(agent.ToolOutput{
			CallID: "call_1",
			Name:   "transfer_to_Spanish_agent",
			Output: `{"assistant":"Spanish agent"}`,
		}))
		require.Len(t, triageModel.requests, 1)
		require.Equal(t, "transfer_to_Spanish_agent", triageModel.requests[0].Handoffs[0].ToolName)
		require.Equal(t,
			"Handoff to the Spanish agent agent to handle the request. Speaks Spanish.",
			triageModel.requests[0].Handoffs[0].ToolDescription)
		require.Len(t, spanishModel.requests, 1)
		require.Contains(t, spanishModel.requests[0].Input, agent.Item(agent.UserMessage("Hola")))
	})

	t.Run("multiple handoffs first wins", func(t *testing.T) {
		first := &agent.Agent{Name: "First", Model: &fakeModel{responses: []fakeModelResponse{
			{Output: []agent.Item{agent.AssistantMessage("first here")}},
		}}}
		second := &agent.Agent{Name: "Second", Model: &fakeModel{}}
		triage := &agent.Agent{
			Name: "Triage",
			Model: &fakeModel{responses: []fakeModelResponse{
				{Output: []agent.Item{
					agent.ToolCall{ID: "call_1", Name: "transfer_to_First", Arguments: "{}"},
					agent.ToolCall{ID: "call_2", Name: "transfer_to_Second", Arguments: "{}"},
				}},
			}},
			Handoffs: []agent.Handoff{agent.HandoffTo(first), agent.HandoffTo(second)},
		}

		result, err := agent.Run(t.Context(), triage, "hello")

		require.NoError(t, err)
		require.Same(t, first, result.LastAgent)
		require.Contains(t, result.NewItems, agent.Item(agent.ToolOutput{
			CallID: "call_2",
			Name:   "transfer_to_Second",
			Output: "Multiple handoffs detected, ignoring this one.",
		}))
	})

	t.Run("dynamic instructions", func(t *testing.T) {
		model := &fakeModel{responses: []fakeModelResponse{
			{Output: []agent.Item{agent.AssistantMessage("ok")}},
		}}
		ag := &agent.Agent{
			Name:  "assistant",
			Model: model,
			InstructionsFunc: func(_ context.Context, _ *agent.RunContext, a *agent.Agent) (string, error) {
				return "You are " + a.Name, nil
			},
		}

		_, err := agent.Run(t.Context(), ag, "Hi")

		require.NoError(t, err)
		require.Equal(t, "You are assistant", model.requests[0].SystemInstructions)
	})

	t.Run("dynamic instructions error", func(t *testing.T) {
		ag := &agent.Agent{
			Name:  "assistant",
			Model: &fakeModel{},
			InstructionsFunc: func(_ context.Context, _ *agent.RunContext, _ *agent.Agent) (string, error) {
				return "", errors.New("boom")
			},
		}

		_, err := agent.Run(t.Context(), ag, "Hi")

		require.EqualError(t, err, `resolve instructions for agent "assistant": boom`)
	})

	t.Run("max turns exceeded", func(t *testing.T) {
		tool := agent.Tool{
			Name: "loop",
			OnInvoke: func(_ context.Context, _ *agent.RunContext, _ string) (string, error) {
				return "again", nil
			},
		}
		model := &fakeModel{responses: []fakeModelResponse{
			{Output: []agent.Item{agent.ToolCall{ID: "call_1", Name: "loop", Arguments: "{}"}}},
			{Output: []agent.Item{agent.ToolCall{ID: "call_2", Name: "loop", Arguments: "{}"}}},
		}}
		ag := &agent.Agent{Name: "assistant", Model: model, Tools: []agent.Tool{tool}}

		runner := &agent.Runner{MaxTurns: 2}
		_, err := runner.Run(t.Context(), ag, "go")

		require.EqualError(t, err, "max turns (2) exceeded")
		var maxTurnsErr *agent.MaxTurnsExceededError
		require.ErrorAs(t, err, &maxTurnsErr)
		require.Equal(t, 2, maxTurnsErr.MaxTurns)
	})

	t.Run("unknown tool", func(t *testing.T) {
		model := &fakeModel{responses: []fakeModelResponse{
			{Output: []agent.Item{agent.ToolCall{ID: "call_1", Name: "nope", Arguments: "{}"}}},
		}}
		ag := &agent.Agent{Name: "assistant", Model: model}

		_, err := agent.Run(t.Context(), ag, "Hi")

		require.EqualError(t, err, `model behavior error: tool "nope" not found on agent "assistant"`)
	})

	t.Run("tool call error propagated", func(t *testing.T) {
		kaboom := errors.New("kaboom")
		tool := agent.Tool{
			Name:            "explode",
			PropagateErrors: true,
			OnInvoke: func(_ context.Context, _ *agent.RunContext, _ string) (string, error) {
				return "", kaboom
			},
		}
		model := &fakeModel{responses: []fakeModelResponse{
			{Output: []agent.Item{agent.ToolCall{ID: "call_1", Name: "explode", Arguments: "{}"}}},
		}}
		ag := &agent.Agent{Name: "assistant", Model: model, Tools: []agent.Tool{tool}}

		_, err := agent.Run(t.Context(), ag, "Hi")

		require.EqualError(t, err, `tool "explode" failed: kaboom`)
		require.ErrorIs(t, err, kaboom)
	})

	t.Run("tool error recovered by default", func(t *testing.T) {
		tool := agent.Tool{
			Name: "explode",
			OnInvoke: func(_ context.Context, _ *agent.RunContext, _ string) (string, error) {
				return "", errors.New("kaboom")
			},
		}
		model := &fakeModel{responses: []fakeModelResponse{
			{Output: []agent.Item{agent.ToolCall{ID: "call_1", Name: "explode", Arguments: "{}"}}},
			{Output: []agent.Item{agent.AssistantMessage("recovered")}},
		}}
		ag := &agent.Agent{Name: "assistant", Model: model, Tools: []agent.Tool{tool}}

		result, err := agent.Run(t.Context(), ag, "Hi")

		require.NoError(t, err)
		require.Equal(t, "recovered", result.FinalOutput)
		require.Contains(t, model.requests[1].Input, agent.Item(agent.ToolOutput{
			CallID: "call_1",
			Name:   "explode",
			Output: "An error occurred while running the tool. Please try again. Error: kaboom",
		}))
	})

	t.Run("tool error custom handler", func(t *testing.T) {
		tool := agent.Tool{
			Name: "explode",
			OnInvoke: func(_ context.Context, _ *agent.RunContext, _ string) (string, error) {
				return "", errors.New("kaboom")
			},
			OnInvokeError: func(_ context.Context, _ *agent.RunContext, err error) string {
				return "custom recovery: " + err.Error()
			},
		}
		model := &fakeModel{responses: []fakeModelResponse{
			{Output: []agent.Item{agent.ToolCall{ID: "call_1", Name: "explode", Arguments: "{}"}}},
			{Output: []agent.Item{agent.AssistantMessage("ok")}},
		}}
		ag := &agent.Agent{Name: "assistant", Model: model, Tools: []agent.Tool{tool}}

		_, err := agent.Run(t.Context(), ag, "Hi")

		require.NoError(t, err)
		require.Contains(t, model.requests[1].Input, agent.Item(agent.ToolOutput{
			CallID: "call_1",
			Name:   "explode",
			Output: "custom recovery: kaboom",
		}))
	})

	t.Run("tool timeout recovered by default", func(t *testing.T) {
		tool := agent.Tool{
			Name:    "slow",
			Timeout: 10 * time.Millisecond,
			OnInvoke: func(ctx context.Context, _ *agent.RunContext, _ string) (string, error) {
				<-ctx.Done()
				return "", ctx.Err()
			},
		}
		model := &fakeModel{responses: []fakeModelResponse{
			{Output: []agent.Item{agent.ToolCall{ID: "call_1", Name: "slow", Arguments: "{}"}}},
			{Output: []agent.Item{agent.AssistantMessage("moving on")}},
		}}
		ag := &agent.Agent{Name: "assistant", Model: model, Tools: []agent.Tool{tool}}

		result, err := agent.Run(t.Context(), ag, "Hi")

		require.NoError(t, err)
		require.Equal(t, "moving on", result.FinalOutput)
		require.Contains(t, model.requests[1].Input, agent.Item(agent.ToolOutput{
			CallID: "call_1",
			Name:   "slow",
			Output: "Tool 'slow' timed out after 10ms.",
		}))
	})

	t.Run("tool timeout propagated", func(t *testing.T) {
		tool := agent.Tool{
			Name:             "slow",
			Timeout:          10 * time.Millisecond,
			PropagateTimeout: true,
			OnInvoke: func(ctx context.Context, _ *agent.RunContext, _ string) (string, error) {
				<-ctx.Done()
				return "", ctx.Err()
			},
		}
		model := &fakeModel{responses: []fakeModelResponse{
			{Output: []agent.Item{agent.ToolCall{ID: "call_1", Name: "slow", Arguments: "{}"}}},
		}}
		ag := &agent.Agent{Name: "assistant", Model: model, Tools: []agent.Tool{tool}}

		_, err := agent.Run(t.Context(), ag, "Hi")

		require.EqualError(t, err, "Tool 'slow' timed out after 10ms.")
		var timeoutErr *agent.ToolTimeoutError
		require.ErrorAs(t, err, &timeoutErr)
		require.Equal(t, "slow", timeoutErr.ToolName)
	})

	t.Run("tool choice reset after tool use", func(t *testing.T) {
		model := &fakeModel{responses: []fakeModelResponse{
			{Output: []agent.Item{agent.ToolCall{ID: "call_1", Name: "loop", Arguments: "{}"}}},
			{Output: []agent.Item{agent.AssistantMessage("done")}},
		}}
		ag := &agent.Agent{
			Name:          "assistant",
			Model:         model,
			Tools:         []agent.Tool{loopTool()},
			ModelSettings: agent.ModelSettings{ToolChoice: "required"},
		}

		_, err := agent.Run(t.Context(), ag, "go")

		require.NoError(t, err)
		require.Equal(t, "required", model.requests[0].ModelSettings.ToolChoice)
		require.Equal(t, "", model.requests[1].ModelSettings.ToolChoice)
	})

	t.Run("tool choice reset disabled", func(t *testing.T) {
		model := &fakeModel{responses: []fakeModelResponse{
			{Output: []agent.Item{agent.ToolCall{ID: "call_1", Name: "loop", Arguments: "{}"}}},
			{Output: []agent.Item{agent.AssistantMessage("done")}},
		}}
		ag := &agent.Agent{
			Name:                   "assistant",
			Model:                  model,
			Tools:                  []agent.Tool{loopTool()},
			ModelSettings:          agent.ModelSettings{ToolChoice: "required"},
			DisableToolChoiceReset: true,
		}

		_, err := agent.Run(t.Context(), ag, "go")

		require.NoError(t, err)
		require.Equal(t, "required", model.requests[1].ModelSettings.ToolChoice)
	})

	t.Run("tool choice none never reset", func(t *testing.T) {
		model := &fakeModel{responses: []fakeModelResponse{
			{Output: []agent.Item{agent.ToolCall{ID: "call_1", Name: "loop", Arguments: "{}"}}},
			{Output: []agent.Item{agent.AssistantMessage("done")}},
		}}
		ag := &agent.Agent{
			Name:          "assistant",
			Model:         model,
			Tools:         []agent.Tool{loopTool()},
			ModelSettings: agent.ModelSettings{ToolChoice: "none"},
		}

		_, err := agent.Run(t.Context(), ag, "go")

		require.NoError(t, err)
		require.Equal(t, "none", model.requests[1].ModelSettings.ToolChoice)
	})

	t.Run("tracer events", func(t *testing.T) {
		model := &fakeModel{responses: []fakeModelResponse{
			{Output: []agent.Item{agent.ToolCall{ID: "call_1", Name: "loop", Arguments: "{}"}}},
			{Output: []agent.Item{agent.AssistantMessage("done")}},
		}}
		ag := &agent.Agent{Name: "assistant", Model: model, Tools: []agent.Tool{loopTool()}}

		tracer := &recordingTracer{}
		runner := &agent.Runner{Tracer: tracer}
		_, err := runner.Run(t.Context(), ag, "go")

		require.NoError(t, err)
		require.Equal(t, []string{
			"run_started:assistant",
			"turn_started:1",
			"model_call_ended",
			"tool_started:loop",
			"tool_ended:loop",
			"turn_started:2",
			"model_call_ended",
			"run_ended",
		}, tracer.log())
	})

	t.Run("tracer handoff event", func(t *testing.T) {
		spanish := &agent.Agent{Name: "Spanish agent", Model: &fakeModel{responses: []fakeModelResponse{
			{Output: []agent.Item{agent.AssistantMessage("¡Hola!")}},
		}}}
		triage := &agent.Agent{
			Name: "Triage",
			Model: &fakeModel{responses: []fakeModelResponse{
				{Output: []agent.Item{agent.ToolCall{ID: "call_1", Name: "transfer_to_Spanish_agent", Arguments: "{}"}}},
			}},
			Handoffs: []agent.Handoff{agent.HandoffTo(spanish)},
		}

		tracer := &recordingTracer{}
		runner := &agent.Runner{Tracer: tracer}
		_, err := runner.Run(t.Context(), triage, "Hola")

		require.NoError(t, err)
		require.Equal(t, []string{
			"run_started:Triage",
			"turn_started:1",
			"model_call_ended",
			"handoff:Triage->Spanish agent",
			"turn_started:2",
			"model_call_ended",
			"run_ended",
		}, tracer.log())
	})

	t.Run("tracer run ended with error", func(t *testing.T) {
		ag := &agent.Agent{Name: "assistant", Model: &fakeModel{}}

		tracer := &recordingTracer{}
		runner := &agent.Runner{Tracer: tracer}
		_, err := runner.Run(t.Context(), ag, "Hi")

		require.Error(t, err)
		events := tracer.log()
		require.Equal(t, "run_ended_error", events[len(events)-1])
	})

	t.Run("model error", func(t *testing.T) {
		ag := &agent.Agent{Name: "assistant", Model: &fakeModel{}}

		_, err := agent.Run(t.Context(), ag, "Hi")

		require.EqualError(t, err, `model response for agent "assistant": fake model has no scripted responses`)
	})

	t.Run("model response with no choices", func(t *testing.T) {
		model := &fakeModel{responses: []fakeModelResponse{{Raw: &llm.ChatResponse{}}}}
		ag := &agent.Agent{Name: "assistant", Model: model}

		_, err := agent.Run(t.Context(), ag, "Hi")

		require.EqualError(t, err, `model response for agent "assistant": chat completions returned no choices`)
	})

	t.Run("model refusal", func(t *testing.T) {
		model := &fakeModel{responses: []fakeModelResponse{{Raw: &llm.ChatResponse{Choices: []llm.ChatChoice{{Message: llm.ChatMessage{Refusal: "nope"}}}}}}}
		ag := &agent.Agent{Name: "assistant", Model: model}

		_, err := agent.Run(t.Context(), ag, "Hi")

		require.EqualError(t, err, `model response for agent "assistant": model behavior error: model refused to produce output: nope`)
	})

	t.Run("input guardrail tripwire", func(t *testing.T) {
		model := &fakeModel{responses: []fakeModelResponse{
			{Output: []agent.Item{agent.AssistantMessage("should not matter")}},
		}}
		ag := &agent.Agent{
			Name:  "assistant",
			Model: model,
			InputGuardrails: []agent.InputGuardrail{{
				Name: "blocklist",
				Execute: func(_ context.Context, _ *agent.RunContext, _ *agent.Agent, _ []agent.Item) (agent.GuardrailOutput, error) {
					return agent.GuardrailOutput{TripwireTriggered: true, OutputInfo: "blocked"}, nil
				},
			}},
		}

		_, err := agent.Run(t.Context(), ag, "bad input")

		require.EqualError(t, err, `input guardrail "blocklist" tripwire triggered`)
		var tripwireErr *agent.InputGuardrailTripwireTriggeredError
		require.ErrorAs(t, err, &tripwireErr)
		require.Equal(t, "blocked", tripwireErr.Result.Output.OutputInfo)
		require.Empty(t, model.requests)
	})

	t.Run("input guardrail execution error", func(t *testing.T) {
		ag := &agent.Agent{
			Name:  "assistant",
			Model: &fakeModel{},
			InputGuardrails: []agent.InputGuardrail{{
				Name: "flaky",
				Execute: func(_ context.Context, _ *agent.RunContext, _ *agent.Agent, _ []agent.Item) (agent.GuardrailOutput, error) {
					return agent.GuardrailOutput{}, errors.New("kaboom")
				},
			}},
		}

		_, err := agent.Run(t.Context(), ag, "Hi")

		require.EqualError(t, err, `guardrail "flaky" failed: kaboom`)
	})

	t.Run("output guardrail tripwire", func(t *testing.T) {
		model := &fakeModel{responses: []fakeModelResponse{
			{Output: []agent.Item{agent.AssistantMessage("rude words")}},
		}}
		ag := &agent.Agent{
			Name:  "assistant",
			Model: model,
			OutputGuardrails: []agent.OutputGuardrail{{
				Name: "profanity",
				Execute: func(_ context.Context, _ *agent.RunContext, _ *agent.Agent, output string) (agent.GuardrailOutput, error) {
					return agent.GuardrailOutput{TripwireTriggered: output == "rude words"}, nil
				},
			}},
		}

		_, err := agent.Run(t.Context(), ag, "Hi")

		require.EqualError(t, err, `output guardrail "profanity" tripwire triggered`)
	})

	t.Run("nil agent", func(t *testing.T) {
		_, err := agent.Run(t.Context(), nil, "Hi")

		require.EqualError(t, err, "agent is nil")
	})

	t.Run("agent without model", func(t *testing.T) {
		ag := &agent.Agent{Name: "assistant"}

		_, err := agent.Run(t.Context(), ag, "Hi")

		require.EqualError(t, err, `agent "assistant" has no model`)
	})

	t.Run("e2e openrouter client", func(t *testing.T) {
		skipE2EInShortMode(t)
		apiKey := requireE2EEnv(t, openRouterAPIKeyEnv)
		ctx, cancel := e2eContext(t)
		defer cancel()

		client := llm.NewOpenRouter(llm.OpenRouterConfig{
			APIKey:    apiKey,
			ChatModel: openRouterE2EModel,
			SiteURL:   "https://github.com/DavidNix/safeagent",
			AppTitle:  "SafeAgent E2E",
		})

		runE2EAgent(t, ctx, "OpenRouter E2E", client, "SAFEAGENT_OPENROUTER_E2E")
	})

	t.Run("e2e vllm client", func(t *testing.T) {
		skipE2EInShortMode(t)
		baseURL := requireE2EEnv(t, vllmBaseURLEnv)
		ctx, cancel := e2eContext(t)
		defer cancel()

		client := newVLLME2EClient(baseURL)

		runE2EAgent(t, ctx, "vLLM E2E", client, "SAFEAGENT_VLLM_E2E")
	})

	t.Run("e2e circuit breaker falls back to vllm", func(t *testing.T) {
		skipE2EInShortMode(t)
		apiKey := requireE2EEnv(t, openRouterAPIKeyEnv)
		baseURL := requireE2EEnv(t, vllmBaseURLEnv)
		ctx, cancel := e2eContext(t)
		defer cancel()

		var primaryHits atomic.Int32
		primaryServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			primaryHits.Add(1)
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte("primary intentionally unavailable"))
		}))
		t.Cleanup(primaryServer.Close)

		primary := llm.NewOpenRouter(llm.OpenRouterConfig{
			APIKey:    apiKey,
			BaseURL:   primaryServer.URL,
			ChatModel: "intentionally-failing-primary",
		})
		fallback := newVLLME2EClient(baseURL)
		breaker := llm.NewCircuitBreakerWithConfig(llm.BreakerConfig{FailureThreshold: 1}, primary, fallback)

		runE2EAgent(t, ctx, "Circuit Breaker E2E", breaker, "SAFEAGENT_CIRCUIT_BREAKER_E2E")
		require.Equal(t, int32(1), primaryHits.Load())
	})
}

const (
	openRouterAPIKeyEnv        = "OPENROUTER_API_KEY"
	openRouterE2EModel         = "qwen/qwen3.6-35b-a3b"
	vllmBaseURLEnv             = "VLLM_BASE_URL"
	vllmE2EModel               = "qwen3.6-35b-a3b"
	e2eTimeout                 = 2 * time.Minute
	e2eAgentMaxCompletionToken = 32
)

func skipE2EInShortMode(t *testing.T) {
	t.Helper()
	if testing.Short() {
		t.Skip("skipping e2e test in short mode")
	}
}

func e2eContext(t *testing.T) (context.Context, context.CancelFunc) {
	t.Helper()
	return context.WithTimeout(t.Context(), e2eTimeout)
}

func requireE2EEnv(t *testing.T, name string) string {
	t.Helper()
	value := strings.TrimSpace(os.Getenv(name))
	require.NotEmpty(t, value, "%s must be set for e2e tests", name)
	return value
}

func newVLLME2EClient(baseURL string) *llm.Client {
	return llm.NewVLLM(llm.VLLMConfig{
		ChatBaseURL: baseURL,
		ChatModel:   vllmE2EModel,
	})
}

func runE2EAgent(t *testing.T, ctx context.Context, name string, model agent.Model, marker string) *agent.RunResult {
	t.Helper()
	temperature := 0.0
	maxTokens := e2eAgentMaxCompletionToken
	ag := &agent.Agent{
		Name:         name,
		Instructions: "You are running a SafeAgent end-to-end test. Reply with exactly the requested marker and no other text.",
		Model:        model,
		ModelSettings: agent.ModelSettings{
			Temperature: &temperature,
			MaxTokens:   &maxTokens,
		},
	}

	result, err := agent.Run(ctx, ag, "Return exactly this marker: "+marker)

	require.NoError(t, err)
	require.Contains(t, result.FinalOutput, marker)
	require.Equal(t, 1, result.Usage.Requests)
	return result
}

func TestAgent_AsTool(t *testing.T) {
	t.Parallel()

	t.Run("happy path", func(t *testing.T) {
		innerModel := &fakeModel{responses: []fakeModelResponse{
			{Output: []agent.Item{agent.AssistantMessage("short summary")}, Usage: agent.Usage{Requests: 1, TotalTokens: 5}},
		}}
		inner := &agent.Agent{Name: "Summarizer", Model: innerModel}

		outerModel := &fakeModel{responses: []fakeModelResponse{
			{
				Output: []agent.Item{agent.ToolCall{ID: "call_1", Name: "Summarizer", Arguments: `{"input":"long text"}`}},
				Usage:  agent.Usage{Requests: 1, TotalTokens: 10},
			},
			{Output: []agent.Item{agent.AssistantMessage("done: short summary")}, Usage: agent.Usage{Requests: 1, TotalTokens: 7}},
		}}
		outer := &agent.Agent{
			Name:  "Writer",
			Model: outerModel,
			Tools: []agent.Tool{inner.AsTool("", "Summarize text")},
		}

		result, err := agent.Run(t.Context(), outer, "Summarize this")

		require.NoError(t, err)
		require.Equal(t, "done: short summary", result.FinalOutput)
		require.Contains(t, result.NewItems, agent.Item(agent.ToolOutput{CallID: "call_1", Name: "Summarizer", Output: "short summary"}))
		require.Equal(t, []agent.Item{agent.UserMessage("long text")}, innerModel.requests[0].Input)
		require.Equal(t, agent.Usage{Requests: 3, TotalTokens: 22}, result.Usage)
	})

	t.Run("invalid tool input recovered", func(t *testing.T) {
		inner := &agent.Agent{Name: "Summarizer", Model: &fakeModel{}}
		outerModel := &fakeModel{responses: []fakeModelResponse{
			{Output: []agent.Item{agent.ToolCall{ID: "call_1", Name: "Summarizer", Arguments: "not json"}}},
			{Output: []agent.Item{agent.AssistantMessage("could not summarize")}},
		}}
		outer := &agent.Agent{
			Name:  "Writer",
			Model: outerModel,
			Tools: []agent.Tool{inner.AsTool("", "Summarize text")},
		}

		result, err := agent.Run(t.Context(), outer, "Summarize this")

		require.NoError(t, err)
		require.Equal(t, "could not summarize", result.FinalOutput)
		require.Len(t, outerModel.requests, 2)
		output, ok := findToolOutput(outerModel.requests[1].Input, "call_1")
		require.True(t, ok)
		require.Contains(t, output.Output, `parse input for agent tool "Summarizer"`)
	})
}

func findToolOutput(items []agent.Item, callID string) (agent.ToolOutput, bool) {
	for _, item := range items {
		output, ok := item.(agent.ToolOutput)
		if ok && output.CallID == callID {
			return output, true
		}
	}
	return agent.ToolOutput{}, false
}

func loopTool() agent.Tool {
	return agent.Tool{
		Name: "loop",
		OnInvoke: func(_ context.Context, _ *agent.RunContext, _ string) (string, error) {
			return "ok", nil
		},
	}
}

type recordingTracer struct {
	agent.NoopTracer
	mu     sync.Mutex
	events []string
}

func (t *recordingTracer) record(event string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.events = append(t.events, event)
}

func (t *recordingTracer) log() []string {
	t.mu.Lock()
	defer t.mu.Unlock()
	return append([]string(nil), t.events...)
}

func (t *recordingTracer) RunStarted(_ context.Context, a *agent.Agent, _ []agent.Item) {
	t.record("run_started:" + a.Name)
}

func (t *recordingTracer) TurnStarted(_ context.Context, _ *agent.Agent, turn int) {
	t.record(fmt.Sprintf("turn_started:%d", turn))
}

func (t *recordingTracer) ModelCallEnded(_ context.Context, _ *agent.Agent, _ llm.ChatRequest, _ *llm.ChatResponse, _ error) {
	t.record("model_call_ended")
}

func (t *recordingTracer) ToolStarted(_ context.Context, _ *agent.Agent, call agent.ToolCall) {
	t.record("tool_started:" + call.Name)
}

func (t *recordingTracer) ToolEnded(_ context.Context, _ *agent.Agent, call agent.ToolCall, _ string, _ error) {
	t.record("tool_ended:" + call.Name)
}

func (t *recordingTracer) Handoff(_ context.Context, from *agent.Agent, to *agent.Agent) {
	t.record(fmt.Sprintf("handoff:%s->%s", from.Name, to.Name))
}

func (t *recordingTracer) RunEnded(_ context.Context, _ *agent.RunResult, err error) {
	switch err {
	case nil:
		t.record("run_ended")
	default:
		t.record("run_ended_error")
	}
}

func TestSlogTracer(t *testing.T) {
	t.Parallel()

	t.Run("happy path", func(t *testing.T) {
		var buf bytes.Buffer
		logger := slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug}))
		model := &fakeModel{responses: []fakeModelResponse{
			{Output: []agent.Item{agent.ToolCall{ID: "call_1", Name: "loop", Arguments: "{}"}}},
			{Output: []agent.Item{agent.AssistantMessage("done")}, Usage: agent.Usage{Requests: 1, TotalTokens: 9}},
		}}
		ag := &agent.Agent{Name: "assistant", Model: model, Tools: []agent.Tool{loopTool()}}

		runner := &agent.Runner{Tracer: agent.NewSlogTracer(logger)}
		_, err := runner.Run(t.Context(), ag, "go")

		require.NoError(t, err)
		logged := buf.String()
		for _, want := range []string{"Run started", "Turn started", "Model call completed", "Tool started", "Tool completed", "Run completed"} {
			require.True(t, strings.Contains(logged, want), "missing %q in logs:\n%s", want, logged)
		}
	})

	t.Run("run error", func(t *testing.T) {
		var buf bytes.Buffer
		logger := slog.New(slog.NewTextHandler(&buf, nil))
		ag := &agent.Agent{Name: "assistant", Model: &fakeModel{}}

		runner := &agent.Runner{Tracer: agent.NewSlogTracer(logger)}
		_, err := runner.Run(t.Context(), ag, "Hi")

		require.Error(t, err)
		require.Contains(t, buf.String(), "Model call failed")
		require.Contains(t, buf.String(), "Run failed")
	})
}
