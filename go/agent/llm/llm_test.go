package llm_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/stretchr/testify/require"

	"github.com/DavidNix/safeagent/agent"
	"github.com/DavidNix/safeagent/agent/llm"
)

func newTestClient(baseURL string, extra ...option.RequestOption) llm.Client {
	opts := append([]option.RequestOption{
		option.WithBaseURL(baseURL),
		option.WithAPIKey("test-key"),
	}, extra...)
	client := openai.NewClient(opts...)
	return &client.Chat.Completions
}

func TestModel_GetResponse(t *testing.T) {
	t.Parallel()

	t.Run("happy path", func(t *testing.T) {
		var gotBody []byte
		var gotPath, gotAuth string
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			gotPath = r.URL.Path
			gotAuth = r.Header.Get("Authorization")
			body, err := io.ReadAll(r.Body)
			if err != nil {
				w.WriteHeader(http.StatusInternalServerError)
				return
			}
			gotBody = body
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{
				"id": "chatcmpl-123",
				"choices": [{
					"message": {
						"content": "It is sunny.",
						"tool_calls": [{"id":"call_2","type":"function","function":{"name":"get_weather","arguments":"{}"}}]
					},
					"finish_reason": "tool_calls"
				}],
				"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
			}`))
		}))
		defer srv.Close()

		model := llm.NewModel("gpt-test", newTestClient(srv.URL))
		resp, err := model.GetResponse(t.Context(), agent.ModelRequest{
			SystemInstructions: "Be helpful",
			Input: []agent.Item{
				agent.UserMessage("What is the weather in SF?"),
				agent.ToolCall{ID: "call_1", Name: "get_weather", Arguments: `{"city":"SF"}`},
				agent.ToolOutput{CallID: "call_1", Name: "get_weather", Output: "sunny"},
			},
			ModelSettings: agent.ModelSettings{
				Temperature:       new(0.5),
				TopP:              new(0.9),
				MaxTokens:         new(100),
				ParallelToolCalls: new(true),
				ToolChoice:        "auto",
			},
			Tools: []agent.ToolDefinition{{
				Name:        "get_weather",
				Description: "Get weather",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
				Strict:      true,
			}},
			Handoffs: []agent.HandoffDefinition{{
				ToolName:        "transfer_to_Support",
				ToolDescription: "Handles support",
				InputSchema:     json.RawMessage(`{"type":"object","properties":{}}`),
			}},
		})

		require.NoError(t, err)
		require.Equal(t, "/chat/completions", gotPath)
		require.Equal(t, "Bearer test-key", gotAuth)
		require.JSONEq(t, `{
			"model": "gpt-test",
			"messages": [
				{"role":"system","content":"Be helpful"},
				{"role":"user","content":"What is the weather in SF?"},
				{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"SF\"}"}}]},
				{"role":"tool","content":"sunny","tool_call_id":"call_1"}
			],
			"tools": [
				{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]},"strict":true}},
				{"type":"function","function":{"name":"transfer_to_Support","description":"Handles support","parameters":{"type":"object","properties":{}}}}
			],
			"temperature": 0.5,
			"top_p": 0.9,
			"max_completion_tokens": 100,
			"parallel_tool_calls": true,
			"tool_choice": "auto"
		}`, string(gotBody))
		require.Equal(t, []agent.Item{
			agent.AssistantMessage("It is sunny."),
			agent.ToolCall{ID: "call_2", Name: "get_weather", Arguments: "{}"},
		}, resp.Output)
		require.Equal(t, agent.Usage{Requests: 1, InputTokens: 10, OutputTokens: 5, TotalTokens: 15}, resp.Usage)
		require.Equal(t, "chatcmpl-123", resp.ResponseID)
	})

	t.Run("specific tool choice", func(t *testing.T) {
		var gotBody []byte
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			gotBody = body
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"x","choices":[{"message":{"content":"ok"}}],"usage":{}}`))
		}))
		defer srv.Close()

		model := llm.NewModel("gpt-test", newTestClient(srv.URL))
		_, err := model.GetResponse(t.Context(), agent.ModelRequest{
			Input:         []agent.Item{agent.UserMessage("hi")},
			ModelSettings: agent.ModelSettings{ToolChoice: "get_weather"},
		})

		require.NoError(t, err)
		require.JSONEq(t, `{
			"model": "gpt-test",
			"messages": [{"role":"user","content":"hi"}],
			"tool_choice": {"type":"function","function":{"name":"get_weather"}}
		}`, string(gotBody))
	})

	t.Run("status error", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte("boom"))
		}))
		defer srv.Close()

		model := llm.NewModel("gpt-test", newTestClient(srv.URL, option.WithMaxRetries(0)))
		_, err := model.GetResponse(t.Context(), agent.ModelRequest{Input: []agent.Item{agent.UserMessage("hi")}})

		require.ErrorContains(t, err, "chat completion: ")
		require.ErrorContains(t, err, "500")
	})

	t.Run("no choices", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"x","choices":[],"usage":{}}`))
		}))
		defer srv.Close()

		model := llm.NewModel("gpt-test", newTestClient(srv.URL))
		_, err := model.GetResponse(t.Context(), agent.ModelRequest{Input: []agent.Item{agent.UserMessage("hi")}})

		require.EqualError(t, err, "chat completions returned no choices")
	})

	t.Run("refusal", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"x","choices":[{"message":{"refusal":"nope"}}],"usage":{}}`))
		}))
		defer srv.Close()

		model := llm.NewModel("gpt-test", newTestClient(srv.URL))
		_, err := model.GetResponse(t.Context(), agent.ModelRequest{Input: []agent.Item{agent.UserMessage("hi")}})

		require.EqualError(t, err, "model behavior error: model refused to produce output: nope")
	})
}
