package llm_test

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/DavidNix/safeagent/llm"
	"github.com/stretchr/testify/require"
)

func TestClient_Complete(t *testing.T) {
	t.Parallel()

	t.Run("default base url points to openrouter", func(t *testing.T) {
		require.Equal(t, "https://openrouter.ai/api/v1", llm.DefaultBaseURL)
	})

	t.Run("e2e openrouter client", func(t *testing.T) {
		skipE2EInShortMode(t)
		apiKey := requireE2EEnv(t, openRouterAPIKeyEnv)
		ctx, cancel := e2eContext(t)
		defer cancel()

		client := newOpenRouterE2EClient(apiKey, "")

		runLLME2ECompletion(t, ctx, client, "SAFEAGENT_LLM_OPENROUTER_E2E")
	})

	t.Run("e2e openrouter reasoning", func(t *testing.T) {
		skipE2EInShortMode(t)
		apiKey := requireE2EEnv(t, openRouterAPIKeyEnv)
		ctx, cancel := e2eContext(t)
		defer cancel()

		client := newOpenRouterReasoningE2EClient(apiKey, "")

		runLLMReasoningE2ECompletion(t, ctx, client, "SAFEAGENT_LLM_OPENROUTER_REASONING_E2E")
	})

	t.Run("e2e vllm client", func(t *testing.T) {
		skipE2EInShortMode(t)
		baseURL := requireE2EEnv(t, vllmBaseURLEnv)
		ctx, cancel := e2eContext(t)
		defer cancel()

		client := newVLLME2EClient(t, ctx, baseURL)

		runLLME2ECompletion(t, ctx, client, "SAFEAGENT_LLM_VLLM_E2E")
	})

	t.Run("e2e vllm reasoning", func(t *testing.T) {
		skipE2EInShortMode(t)
		baseURL := requireE2EEnv(t, vllmBaseURLEnv)
		ctx, cancel := e2eContext(t)
		defer cancel()

		client := newVLLMReasoningE2EClient(t, ctx, baseURL)

		runLLMReasoningE2ECompletion(t, ctx, client, "SAFEAGENT_LLM_VLLM_REASONING_E2E")
	})

	t.Run("happy path sends chat completion request and decodes api response", func(t *testing.T) {
		request := &capturedRequest{}
		server := chatCompletionServer(t, http.StatusOK, "It is sunny.", request, nil)
		t.Cleanup(server.Close)

		client := llm.NewClient("gpt-test", llm.WithBaseURL(server.URL), llm.WithAPIKey("test-key"))
		resp, err := client.Complete(t.Context(), llm.ChatRequest{
			Messages: []llm.ChatMessage{
				{Role: "system", Content: "Be helpful"},
				{Role: "user", Content: "What is the weather in SF?"},
				{Role: "assistant", ToolCalls: []llm.ChatToolCall{{ID: "call_1", Type: "function", Function: llm.ChatFunctionCall{Name: "get_weather", Arguments: `{"city":"SF"}`}}}},
				{Role: "tool", Content: "sunny", ToolCallID: "call_1"},
			},
			Tools: []llm.ChatTool{{
				Type: "function",
				Function: llm.ChatFunctionDefinition{
					Name:        "get_weather",
					Description: "Get weather",
					Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
					Strict:      true,
				},
			}},
			Temperature:       new(0.5),
			TopP:              new(0.9),
			MaxTokens:         new(100),
			ParallelToolCalls: new(true),
			ToolChoice:        "auto",
		})

		require.NoError(t, err)
		require.Equal(t, "/chat/completions", request.Path)
		require.Equal(t, "Bearer test-key", request.Header.Get("Authorization"))
		require.JSONEq(t, `{
			"model": "gpt-test",
			"messages": [
				{"role":"system","content":"Be helpful"},
				{"role":"user","content":"What is the weather in SF?"},
				{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"SF\"}"}}]},
				{"role":"tool","content":"sunny","tool_call_id":"call_1"}
			],
			"tools": [
				{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]},"strict":true}}
			],
			"temperature": 0.5,
			"top_p": 0.9,
			"max_completion_tokens": 100,
			"parallel_tool_calls": true,
			"tool_choice": "auto"
		}`, string(request.Body))
		require.Equal(t, "chatcmpl-test", resp.ID)
		require.Equal(t, "It is sunny.", resp.Choices[0].Message.Content)
		require.Equal(t, []llm.ChatToolCall{{ID: "call_2", Type: "function", Function: llm.ChatFunctionCall{Name: "get_weather", Arguments: "{}"}}}, resp.Choices[0].Message.ToolCalls)
		require.Equal(t, llm.ChatUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15}, resp.Usage)
	})

	t.Run("decodes vllm reasoning field", func(t *testing.T) {
		server := rawJSONServer(t, http.StatusOK, `{
			"id":"chatcmpl-test",
			"choices":[{"message":{"role":"assistant","content":"final","reasoning":"hidden chain"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}
		}`)
		t.Cleanup(server.Close)

		client := llm.NewClient("gpt-test", llm.WithBaseURL(server.URL), llm.WithAPIKey("test-key"))

		resp, err := client.Complete(t.Context(), llm.ChatRequest{Messages: []llm.ChatMessage{{Role: "user", Content: "hello"}}})

		require.NoError(t, err)
		require.Equal(t, "hidden chain", resp.Choices[0].Message.ReasoningContent)
		require.Equal(t, "final", resp.Choices[0].Message.Content)
	})

	t.Run("provider constructors apply openrouter and vllm configuration", func(t *testing.T) {
		primaryRequest := &capturedRequest{}
		primaryServer := chatCompletionServer(t, http.StatusServiceUnavailable, "", primaryRequest, nil)
		t.Cleanup(primaryServer.Close)

		fallbackRequest := &capturedRequest{}
		fallbackServer := chatCompletionServer(t, http.StatusOK, `{"fallback":true}`, fallbackRequest, nil)
		t.Cleanup(fallbackServer.Close)

		budget := 128
		primary := llm.NewVLLM(llm.VLLMConfig{
			ChatBaseURL:          primaryServer.URL,
			ChatModel:            "vllm-chat",
			ReasoningTokenBudget: &budget,
		})
		fallback := llm.NewOpenRouter(llm.OpenRouterConfig{
			APIKey:                   "openrouter-key",
			BaseURL:                  fallbackServer.URL,
			ChatModel:                "openrouter-chat",
			SiteURL:                  "https://safeagent.example",
			AppTitle:                 "SafeAgent",
			RequireZeroDataRetention: true,
			Headers:                  map[string]string{"X-OpenRouter-Test": "yes"},
			QueryParams:              map[string]string{"route": "openrouter"},
		})
		breaker := llm.NewCircuitBreaker(primary, fallback)

		resp, err := breaker.Complete(t.Context(), llm.ChatRequest{Messages: []llm.ChatMessage{{Role: "user", Content: "hello"}}})

		require.NoError(t, err)
		require.NotNil(t, resp)
		require.Equal(t, "vllm-chat", primaryRequest.JSONBody["model"])
		require.InDelta(t, 128, primaryRequest.JSONBody["thinking_token_budget"], 0.0001)
		reasoning, ok := primaryRequest.JSONBody["reasoning"].(map[string]any)
		require.True(t, ok)
		require.InDelta(t, 128, reasoning["max_tokens"], 0.0001)

		require.Equal(t, "openrouter-chat", fallbackRequest.JSONBody["model"])
		require.Equal(t, "Bearer openrouter-key", fallbackRequest.Header.Get("Authorization"))
		require.Equal(t, "https://safeagent.example", fallbackRequest.Header.Get("HTTP-Referer"))
		require.Equal(t, "SafeAgent", fallbackRequest.Header.Get("X-Title"))
		require.Equal(t, "yes", fallbackRequest.Header.Get("X-OpenRouter-Test"))
		require.Equal(t, "openrouter", fallbackRequest.Query.Get("route"))
		require.NotContains(t, fallbackRequest.JSONBody, "thinking_token_budget")
		provider, ok := fallbackRequest.JSONBody["provider"].(map[string]any)
		require.True(t, ok)
		require.Equal(t, true, provider["zdr"])
	})

	t.Run("zero reasoning budget disables vllm thinking", func(t *testing.T) {
		request := &capturedRequest{}
		server := chatCompletionServer(t, http.StatusOK, `{"ok":true}`, request, nil)
		t.Cleanup(server.Close)

		budget := 0
		client := llm.NewVLLM(llm.VLLMConfig{
			APIKey:               "local-key",
			ChatBaseURL:          server.URL,
			ChatModel:            "vllm-chat",
			ReasoningTokenBudget: &budget,
			Headers:              map[string]string{"X-VLLM-Header": "present"},
			QueryParams:          map[string]string{"backend": "vllm"},
		})

		_, err := client.Complete(t.Context(), llm.ChatRequest{Messages: []llm.ChatMessage{{Role: "user", Content: "hello"}}})

		require.NoError(t, err)
		require.Equal(t, "Bearer local-key", request.Header.Get("Authorization"))
		require.Equal(t, "present", request.Header.Get("X-VLLM-Header"))
		require.Equal(t, "vllm", request.Query.Get("backend"))
		require.InDelta(t, 0, request.JSONBody["thinking_token_budget"], 0.0001)
		chatTemplateKwargs, ok := request.JSONBody["chat_template_kwargs"].(map[string]any)
		require.True(t, ok)
		require.Equal(t, false, chatTemplateKwargs["enable_thinking"])
		reasoning, ok := request.JSONBody["reasoning"].(map[string]any)
		require.True(t, ok)
		require.Equal(t, "none", reasoning["effort"])
	})

	t.Run("non-trigger status error does not retry", func(t *testing.T) {
		var attempts atomic.Int32
		server := chatCompletionServer(t, http.StatusBadRequest, "", nil, &attempts)
		t.Cleanup(server.Close)

		client := llm.NewClient("gpt-test", llm.WithBaseURL(server.URL), llm.WithRetryDelay(time.Millisecond))

		_, err := client.Complete(t.Context(), llm.ChatRequest{Messages: []llm.ChatMessage{{Role: "user", Content: "hi"}}})

		require.EqualError(t, err, "chat completions returned status 400")
		require.Equal(t, int32(1), attempts.Load())
		var statusErr *llm.StatusError
		require.ErrorAs(t, err, &statusErr)
		require.Equal(t, http.StatusBadRequest, statusErr.StatusCode)
	})

	t.Run("retries are disabled by default", func(t *testing.T) {
		var attempts atomic.Int32
		server := chatCompletionServer(t, http.StatusInternalServerError, "", nil, &attempts)
		t.Cleanup(server.Close)

		client := llm.NewClient("gpt-test", llm.WithBaseURL(server.URL), llm.WithRetryDelay(time.Millisecond))

		_, err := client.Complete(t.Context(), llm.ChatRequest{Messages: []llm.ChatMessage{{Role: "user", Content: "hi"}}})

		require.EqualError(t, err, "chat completions returned status 500")
		require.Equal(t, int32(1), attempts.Load())
	})

	t.Run("configured retries then succeeds", func(t *testing.T) {
		var attempts atomic.Int32
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			if attempts.Add(1) <= 2 {
				w.WriteHeader(http.StatusInternalServerError)
				_, _ = w.Write([]byte("boom"))
				return
			}
			writeChatCompletion(w, `{"ok":true}`)
		}))
		t.Cleanup(server.Close)

		client := llm.NewClient("gpt-test",
			llm.WithBaseURL(server.URL),
			llm.WithMaxRetries(2),
			llm.WithRetryDelay(time.Millisecond),
		)
		resp, err := client.Complete(t.Context(), llm.ChatRequest{Messages: []llm.ChatMessage{{Role: "user", Content: "hi"}}})

		require.NoError(t, err)
		require.Equal(t, "chatcmpl-test", resp.ID)
		require.Equal(t, int32(3), attempts.Load())
	})

	t.Run("configured retries exhausted", func(t *testing.T) {
		var attempts atomic.Int32
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			attempts.Add(1)
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte("boom"))
		}))
		t.Cleanup(server.Close)

		client := llm.NewClient("gpt-test",
			llm.WithBaseURL(server.URL),
			llm.WithMaxRetries(1),
			llm.WithRetryDelay(time.Millisecond),
		)
		_, err := client.Complete(t.Context(), llm.ChatRequest{Messages: []llm.ChatMessage{{Role: "user", Content: "hi"}}})

		require.EqualError(t, err, "chat completions failed after 2 attempts: chat completions returned status 500: boom")
		require.Equal(t, int32(2), attempts.Load())
	})

	t.Run("response body exceeding limit returns typed error", func(t *testing.T) {
		server := chatCompletionServer(t, http.StatusOK, "too large", nil, nil)
		t.Cleanup(server.Close)

		client := llm.NewClient("gpt-test", llm.WithBaseURL(server.URL), llm.WithMaxResponseBytes(8))

		_, err := client.Complete(t.Context(), llm.ChatRequest{Messages: []llm.ChatMessage{{Role: "user", Content: "hi"}}})

		require.EqualError(t, err, "read chat completions response: chat completions response exceeded 8 bytes")
		var tooLarge *llm.ResponseTooLargeError
		require.ErrorAs(t, err, &tooLarge)
		require.Equal(t, int64(8), tooLarge.Limit)
	})

	t.Run("response body close error is ignored", func(t *testing.T) {
		closeErr := errors.New("close boom")
		client := llm.NewClient("gpt-test", llm.WithHTTPClient(&http.Client{
			Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return &http.Response{
					StatusCode: http.StatusOK,
					Body:       errorOnCloseBody{Reader: strings.NewReader(chatCompletionJSON("ok")), err: closeErr},
				}, nil
			}),
		}))

		resp, err := client.Complete(t.Context(), llm.ChatRequest{Messages: []llm.ChatMessage{{Role: "user", Content: "hi"}}})

		require.NoError(t, err)
		require.Equal(t, "ok", resp.Choices[0].Message.Content)
	})
}

func TestClient_Models(t *testing.T) {
	t.Parallel()

	t.Run("happy path", func(t *testing.T) {
		request := &capturedRequest{}
		server := modelsServer(t, http.StatusOK, `{
			"object":"list",
			"data":[{"id":"qwen3.6-35b-a3b-nvfp4","object":"model","created":123,"owned_by":"vllm"}]
		}`, request)
		t.Cleanup(server.Close)

		client := llm.NewClient("ignored",
			llm.WithBaseURL(server.URL),
			llm.WithAPIKey("test-key"),
			llm.WithHeaders(map[string]string{"X-Test-Header": "yes"}),
			llm.WithQueryParams(map[string]string{"backend": "vllm"}),
		)

		resp, err := client.Models(t.Context())

		require.NoError(t, err)
		require.Equal(t, http.MethodGet, request.Method)
		require.Equal(t, "/models", request.Path)
		require.Equal(t, "Bearer test-key", request.Header.Get("Authorization"))
		require.Equal(t, "yes", request.Header.Get("X-Test-Header"))
		require.Equal(t, "vllm", request.Query.Get("backend"))
		require.Equal(t, &llm.ModelsResponse{
			Object: "list",
			Data: []llm.ModelInfo{{
				ID:      "qwen3.6-35b-a3b-nvfp4",
				Object:  "model",
				Created: 123,
				OwnedBy: "vllm",
			}},
		}, resp)
	})

	t.Run("status error", func(t *testing.T) {
		server := modelsServer(t, http.StatusServiceUnavailable, "down", nil)
		t.Cleanup(server.Close)

		client := llm.NewClient("ignored", llm.WithBaseURL(server.URL))

		_, err := client.Models(t.Context())

		require.EqualError(t, err, "models returned status 503: down")
	})
}

type capturedRequest struct {
	Method   string
	Path     string
	Query    url.Values
	Header   http.Header
	Body     []byte
	JSONBody map[string]any
}

func chatCompletionServer(t *testing.T, status int, content string, captured *capturedRequest, hits *atomic.Int32) *httptest.Server {
	t.Helper()

	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if hits != nil {
			hits.Add(1)
		}
		captureJSONRequest(t, r, captured)
		if status >= http.StatusBadRequest {
			w.WriteHeader(status)
			return
		}
		writeChatCompletion(w, content)
	}))
}

func modelsServer(t *testing.T, status int, body string, captured *capturedRequest) *httptest.Server {
	t.Helper()

	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		bodyBytes, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		if captured != nil {
			captured.Method = r.Method
			captured.Path = r.URL.Path
			captured.Query = r.URL.Query()
			captured.Header = r.Header.Clone()
			captured.Body = bodyBytes
		}
		if status >= http.StatusBadRequest {
			w.WriteHeader(status)
			_, _ = w.Write([]byte(body))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	}))
}

func rawJSONServer(t *testing.T, status int, body string) *httptest.Server {
	t.Helper()

	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captureJSONRequest(t, r, nil)
		if status >= http.StatusBadRequest {
			w.WriteHeader(status)
			_, _ = w.Write([]byte(body))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	}))
}

func captureJSONRequest(t *testing.T, r *http.Request, captured *capturedRequest) {
	t.Helper()

	body, err := io.ReadAll(r.Body)
	require.NoError(t, err)
	var jsonBody map[string]any
	require.NoError(t, json.Unmarshal(body, &jsonBody))
	if captured == nil {
		return
	}
	captured.Method = r.Method
	captured.Path = r.URL.Path
	captured.Query = r.URL.Query()
	captured.Header = r.Header.Clone()
	captured.Body = body
	captured.JSONBody = jsonBody
}

func writeChatCompletion(w http.ResponseWriter, content string) {
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write([]byte(chatCompletionJSON(content)))
}

func chatCompletionJSON(content string) string {
	body, err := json.Marshal(map[string]any{
		"id":     "chatcmpl-test",
		"object": "chat.completion",
		"model":  "test-chat-model",
		"choices": []map[string]any{
			{
				"index": 0,
				"message": map[string]any{
					"role":    "assistant",
					"content": content,
					"tool_calls": []map[string]any{
						{"id": "call_2", "type": "function", "function": map[string]any{"name": "get_weather", "arguments": "{}"}},
					},
				},
				"finish_reason": "stop",
			},
		},
		"usage": map[string]any{
			"prompt_tokens":     10,
			"completion_tokens": 5,
			"total_tokens":      15,
		},
	})
	if err != nil {
		panic(err)
	}
	return string(body)
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

type errorOnCloseBody struct {
	*strings.Reader
	err error
}

func (b errorOnCloseBody) Close() error {
	return b.err
}
