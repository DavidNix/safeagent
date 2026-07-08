package agent_test

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/DavidNix/safeagent/agent"
	"github.com/stretchr/testify/require"
)

type fakeModel struct {
	mu        sync.Mutex
	responses []agent.ModelResponse
	requests  []agent.ModelRequest
}

func (m *fakeModel) GetResponse(_ context.Context, req agent.ModelRequest) (*agent.ModelResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	snapshot := req
	snapshot.Input = make([]agent.Item, len(req.Input))
	copy(snapshot.Input, req.Input)
	m.requests = append(m.requests, snapshot)
	if len(m.responses) == 0 {
		return nil, errors.New("fake model has no scripted responses")
	}
	resp := m.responses[0]
	m.responses = m.responses[1:]
	return &resp, nil
}

func TestRunner_Run(t *testing.T) {
	t.Parallel()

	t.Run("happy path", func(t *testing.T) {
		model := &fakeModel{responses: []agent.ModelResponse{
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
		model := &fakeModel{responses: []agent.ModelResponse{
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
		spanishModel := &fakeModel{responses: []agent.ModelResponse{
			{Output: []agent.Item{agent.AssistantMessage("¡Hola!")}, Usage: agent.Usage{Requests: 1}},
		}}
		spanish := &agent.Agent{Name: "Spanish agent", Model: spanishModel, HandoffDescription: "Speaks Spanish."}

		triageModel := &fakeModel{responses: []agent.ModelResponse{
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
		first := &agent.Agent{Name: "First", Model: &fakeModel{responses: []agent.ModelResponse{
			{Output: []agent.Item{agent.AssistantMessage("first here")}},
		}}}
		second := &agent.Agent{Name: "Second", Model: &fakeModel{}}
		triage := &agent.Agent{
			Name: "Triage",
			Model: &fakeModel{responses: []agent.ModelResponse{
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
		model := &fakeModel{responses: []agent.ModelResponse{
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
		model := &fakeModel{responses: []agent.ModelResponse{
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
		model := &fakeModel{responses: []agent.ModelResponse{
			{Output: []agent.Item{agent.ToolCall{ID: "call_1", Name: "nope", Arguments: "{}"}}},
		}}
		ag := &agent.Agent{Name: "assistant", Model: model}

		_, err := agent.Run(t.Context(), ag, "Hi")

		require.EqualError(t, err, `model behavior error: tool "nope" not found on agent "assistant"`)
	})

	t.Run("tool call error", func(t *testing.T) {
		kaboom := errors.New("kaboom")
		tool := agent.Tool{
			Name: "explode",
			OnInvoke: func(_ context.Context, _ *agent.RunContext, _ string) (string, error) {
				return "", kaboom
			},
		}
		model := &fakeModel{responses: []agent.ModelResponse{
			{Output: []agent.Item{agent.ToolCall{ID: "call_1", Name: "explode", Arguments: "{}"}}},
		}}
		ag := &agent.Agent{Name: "assistant", Model: model, Tools: []agent.Tool{tool}}

		_, err := agent.Run(t.Context(), ag, "Hi")

		require.EqualError(t, err, `tool "explode" failed: kaboom`)
		require.ErrorIs(t, err, kaboom)
	})

	t.Run("model error", func(t *testing.T) {
		ag := &agent.Agent{Name: "assistant", Model: &fakeModel{}}

		_, err := agent.Run(t.Context(), ag, "Hi")

		require.EqualError(t, err, `model response for agent "assistant": fake model has no scripted responses`)
	})

	t.Run("input guardrail tripwire", func(t *testing.T) {
		model := &fakeModel{responses: []agent.ModelResponse{
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
		model := &fakeModel{responses: []agent.ModelResponse{
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
}

func TestAgent_AsTool(t *testing.T) {
	t.Parallel()

	t.Run("happy path", func(t *testing.T) {
		innerModel := &fakeModel{responses: []agent.ModelResponse{
			{Output: []agent.Item{agent.AssistantMessage("short summary")}, Usage: agent.Usage{Requests: 1, TotalTokens: 5}},
		}}
		inner := &agent.Agent{Name: "Summarizer", Model: innerModel}

		outerModel := &fakeModel{responses: []agent.ModelResponse{
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

	t.Run("invalid tool input", func(t *testing.T) {
		inner := &agent.Agent{Name: "Summarizer", Model: &fakeModel{}}
		outerModel := &fakeModel{responses: []agent.ModelResponse{
			{Output: []agent.Item{agent.ToolCall{ID: "call_1", Name: "Summarizer", Arguments: "not json"}}},
		}}
		outer := &agent.Agent{
			Name:  "Writer",
			Model: outerModel,
			Tools: []agent.Tool{inner.AsTool("", "Summarize text")},
		}

		_, err := agent.Run(t.Context(), outer, "Summarize this")

		require.ErrorContains(t, err, `tool "Summarizer" failed: parse input for agent tool "Summarizer"`)
	})
}
