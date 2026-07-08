package llm_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/DavidNix/safeagent/agent"
	"github.com/DavidNix/safeagent/agent/llm"
)

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

		model := llm.NewModel("gpt-test", llm.WithBaseURL(srv.URL), llm.WithAPIKey("test-key"))
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
			_, _ = w.Write([]byte(`{"id":"x","choices":[{"message":{"content":"ok"}}],"usage":{}}`))
		}))
		defer srv.Close()

		model := llm.NewModel("gpt-test", llm.WithBaseURL(srv.URL), llm.WithAPIKey("test-key"))
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

	t.Run("reasoning content extracted", func(t *testing.T) {
		var gotBody []byte
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			gotBody = body
			_, _ = w.Write([]byte(`{
				"id": "x",
				"choices": [{"message": {
					"reasoning_content": "The user greeted me, I should greet back.",
					"content": "Hello!"
				}}],
				"usage": {}
			}`))
		}))
		defer srv.Close()

		model := llm.NewModel("r1-test", llm.WithBaseURL(srv.URL), llm.WithAPIKey("test-key"))
		resp, err := model.GetResponse(t.Context(), agent.ModelRequest{
			Input: []agent.Item{
				agent.UserMessage("hi"),
				agent.Reasoning{Content: "prior turn thinking"},
				agent.AssistantMessage("earlier reply"),
			},
		})

		require.NoError(t, err)
		require.Equal(t, []agent.Item{
			agent.Reasoning{Content: "The user greeted me, I should greet back."},
			agent.AssistantMessage("Hello!"),
		}, resp.Output)
		require.JSONEq(t, `{
			"model": "r1-test",
			"messages": [
				{"role":"user","content":"hi"},
				{"role":"assistant","content":"earlier reply"}
			]
		}`, string(gotBody))
	})

	t.Run("retries then succeeds", func(t *testing.T) {
		var attempts atomic.Int32
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			if attempts.Add(1) <= 2 {
				w.WriteHeader(http.StatusInternalServerError)
				_, _ = w.Write([]byte("boom"))
				return
			}
			_, _ = w.Write([]byte(`{"id":"x","choices":[{"message":{"content":"ok"}}],"usage":{}}`))
		}))
		defer srv.Close()

		model := llm.NewModel("gpt-test",
			llm.WithBaseURL(srv.URL),
			llm.WithAPIKey("test-key"),
			llm.WithMaxRetries(2),
			llm.WithRetryDelay(time.Millisecond),
		)
		resp, err := model.GetResponse(t.Context(), agent.ModelRequest{Input: []agent.Item{agent.UserMessage("hi")}})

		require.NoError(t, err)
		require.Equal(t, []agent.Item{agent.AssistantMessage("ok")}, resp.Output)
		require.Equal(t, int32(3), attempts.Load())
	})

	t.Run("retries exhausted", func(t *testing.T) {
		var attempts atomic.Int32
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			attempts.Add(1)
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte("boom"))
		}))
		defer srv.Close()

		model := llm.NewModel("gpt-test",
			llm.WithBaseURL(srv.URL),
			llm.WithAPIKey("test-key"),
			llm.WithMaxRetries(1),
			llm.WithRetryDelay(time.Millisecond),
		)
		_, err := model.GetResponse(t.Context(), agent.ModelRequest{Input: []agent.Item{agent.UserMessage("hi")}})

		require.EqualError(t, err, "chat completions failed after 2 attempts: chat completions returned status 500: boom")
		require.Equal(t, int32(2), attempts.Load())
	})

	t.Run("client error not retried", func(t *testing.T) {
		var attempts atomic.Int32
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			attempts.Add(1)
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte("bad request"))
		}))
		defer srv.Close()

		model := llm.NewModel("gpt-test",
			llm.WithBaseURL(srv.URL),
			llm.WithAPIKey("test-key"),
			llm.WithRetryDelay(time.Millisecond),
		)
		_, err := model.GetResponse(t.Context(), agent.ModelRequest{Input: []agent.Item{agent.UserMessage("hi")}})

		require.EqualError(t, err, "chat completions returned status 400: bad request")
		require.Equal(t, int32(1), attempts.Load())
	})

	t.Run("no choices", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`{"id":"x","choices":[],"usage":{}}`))
		}))
		defer srv.Close()

		model := llm.NewModel("gpt-test", llm.WithBaseURL(srv.URL), llm.WithAPIKey("test-key"))
		_, err := model.GetResponse(t.Context(), agent.ModelRequest{Input: []agent.Item{agent.UserMessage("hi")}})

		require.EqualError(t, err, "chat completions returned no choices")
	})

	t.Run("refusal", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`{"id":"x","choices":[{"message":{"refusal":"nope"}}],"usage":{}}`))
		}))
		defer srv.Close()

		model := llm.NewModel("gpt-test", llm.WithBaseURL(srv.URL), llm.WithAPIKey("test-key"))
		_, err := model.GetResponse(t.Context(), agent.ModelRequest{Input: []agent.Item{agent.UserMessage("hi")}})

		require.EqualError(t, err, "model behavior error: model refused to produce output: nope")
	})
}
