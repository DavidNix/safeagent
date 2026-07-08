package client

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync/atomic"
	"testing"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/require"
)

func TestCompletionClient_New(t *testing.T) {
	t.Parallel()

	t.Run("happy path uses VLLM config and mirrors OpenAI call shape", func(t *testing.T) {
		request := &capturedRequest{}
		server := chatCompletionServer(t, http.StatusOK, `{"ok":true}`, request, nil)
		t.Cleanup(server.Close)

		budget := 0
		client, err := NewCompletionClient(Options{
			Primary: VLLMConfig{
				APIKey:               "local-key",
				ChatBaseURL:          server.URL,
				EmbeddingsBaseURL:    server.URL,
				ChatModel:            "vllm-chat",
				EmbeddingModel:       "vllm-embedding",
				ReasoningTokenBudget: &budget,
				Headers:              map[string]string{"X-VLLM-Header": "present"},
				QueryParams:          map[string]string{"backend": "vllm"},
			},
			HTTPClient: headerHTTPClient("X-Custom-HTTP-Client", "used"),
		})
		require.NoError(t, err)

		completion, err := client.New(t.Context(), openai.ChatCompletionNewParams{
			Model: "caller-model",
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("hello"),
			},
		})
		require.NoError(t, err)
		require.NotNil(t, completion)
		require.Equal(t, "/chat/completions", request.Path)
		require.Equal(t, "Bearer local-key", request.Header.Get("Authorization"))
		require.Equal(t, "used", request.Header.Get("X-Custom-HTTP-Client"))
		require.Equal(t, "present", request.Header.Get("X-VLLM-Header"))
		require.Equal(t, "vllm", request.Query.Get("backend"))
		require.Equal(t, "vllm-chat", request.Body["model"])
		require.InDelta(t, 0, request.Body["thinking_token_budget"], 0.0001)

		chatTemplateKwargs, ok := request.Body["chat_template_kwargs"].(map[string]any)
		require.True(t, ok)
		require.Equal(t, false, chatTemplateKwargs["enable_thinking"])

		reasoning, ok := request.Body["reasoning"].(map[string]any)
		require.True(t, ok)
		require.Equal(t, "none", reasoning["effort"])
	})

	t.Run("trigger failure falls back to OpenRouter with OpenRouter fields only", func(t *testing.T) {
		primaryRequest := &capturedRequest{}
		primaryServer := chatCompletionServer(t, http.StatusServiceUnavailable, "", primaryRequest, nil)
		t.Cleanup(primaryServer.Close)

		fallbackRequest := &capturedRequest{}
		fallbackServer := chatCompletionServer(t, http.StatusOK, `{"fallback":true}`, fallbackRequest, nil)
		t.Cleanup(fallbackServer.Close)

		budget := 128
		client, err := NewCompletionClient(Options{
			Primary: VLLMConfig{
				ChatBaseURL:          primaryServer.URL,
				EmbeddingsBaseURL:    primaryServer.URL,
				ChatModel:            "vllm-chat",
				EmbeddingModel:       "vllm-embedding",
				ReasoningTokenBudget: &budget,
			},
			Fallbacks: []Provider{OpenRouterConfig{
				APIKey:                   "openrouter-key",
				BaseURL:                  fallbackServer.URL,
				ChatModel:                "openrouter-chat",
				EmbeddingModel:           "openrouter-embedding",
				SiteURL:                  "https://safeagent.example",
				AppTitle:                 "SafeAgent",
				RequireZeroDataRetention: true,
				Headers:                  map[string]string{"X-OpenRouter-Test": "yes"},
				QueryParams:              map[string]string{"route": "openrouter"},
			}},
		})
		require.NoError(t, err)

		completion, err := client.New(t.Context(), openai.ChatCompletionNewParams{
			Model: "caller-model",
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("hello"),
			},
		})
		require.NoError(t, err)
		require.NotNil(t, completion)
		require.Equal(t, "vllm-chat", primaryRequest.Body["model"])
		require.InDelta(t, 128, primaryRequest.Body["thinking_token_budget"], 0.0001)
		require.Equal(t, "openrouter-chat", fallbackRequest.Body["model"])
		require.Equal(t, "Bearer openrouter-key", fallbackRequest.Header.Get("Authorization"))
		require.Equal(t, "https://safeagent.example", fallbackRequest.Header.Get("HTTP-Referer"))
		require.Equal(t, "SafeAgent", fallbackRequest.Header.Get("X-Title"))
		require.Equal(t, "yes", fallbackRequest.Header.Get("X-OpenRouter-Test"))
		require.Equal(t, "openrouter", fallbackRequest.Query.Get("route"))
		require.NotContains(t, fallbackRequest.Body, "thinking_token_budget")
		require.NotContains(t, fallbackRequest.Body, "chat_template_kwargs")

		provider, ok := fallbackRequest.Body["provider"].(map[string]any)
		require.True(t, ok)
		require.Equal(t, true, provider["zdr"])
	})

	t.Run("non-trigger primary error does not fall back", func(t *testing.T) {
		primaryServer := chatCompletionServer(t, http.StatusBadRequest, "", nil, nil)
		t.Cleanup(primaryServer.Close)

		var fallbackHits atomic.Int32
		fallbackServer := chatCompletionServer(t, http.StatusOK, `{"fallback":true}`, nil, &fallbackHits)
		t.Cleanup(fallbackServer.Close)

		client, err := NewCompletionClient(Options{
			Primary: VLLMConfig{
				ChatBaseURL:       primaryServer.URL,
				EmbeddingsBaseURL: primaryServer.URL,
				ChatModel:         "vllm-chat",
				EmbeddingModel:    "vllm-embedding",
			},
			Fallbacks: []Provider{OpenRouterConfig{
				APIKey:         "openrouter-key",
				BaseURL:        fallbackServer.URL,
				ChatModel:      "openrouter-chat",
				EmbeddingModel: "openrouter-embedding",
			}},
		})
		require.NoError(t, err)

		_, err = client.New(t.Context(), openai.ChatCompletionNewParams{
			Model: "caller-model",
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("hello"),
			},
		})
		require.Error(t, err)
		require.Equal(t, int32(0), fallbackHits.Load())
	})

	t.Run("circuit opens after threshold and skips primary until probe succeeds", func(t *testing.T) {
		clock := &fakeClock{now: time.Unix(1000, 0)}
		var primaryHits atomic.Int32
		var primaryHealthy atomic.Bool
		primaryServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			primaryHits.Add(1)
			captureJSONRequest(t, r, nil)
			if !primaryHealthy.Load() {
				w.WriteHeader(http.StatusServiceUnavailable)
				return
			}
			writeChatCompletion(w, `{"primary":true}`)
		}))
		t.Cleanup(primaryServer.Close)

		fallbackServer := chatCompletionServer(t, http.StatusOK, `{"fallback":true}`, nil, nil)
		t.Cleanup(fallbackServer.Close)

		client, err := NewCompletionClient(Options{
			Primary: VLLMConfig{
				ChatBaseURL:       primaryServer.URL,
				EmbeddingsBaseURL: primaryServer.URL,
				ChatModel:         "vllm-chat",
				EmbeddingModel:    "vllm-embedding",
			},
			Fallbacks: []Provider{OpenRouterConfig{
				APIKey:         "openrouter-key",
				BaseURL:        fallbackServer.URL,
				ChatModel:      "openrouter-chat",
				EmbeddingModel: "openrouter-embedding",
			}},
			Breaker: BreakerConfig{
				FailureThreshold:         2,
				OpenDuration:             time.Minute,
				ProbeInterval:            10 * time.Second,
				SuccessfulProbeThreshold: 1,
				Now:                      clock.Now,
			},
		})
		require.NoError(t, err)

		for range 2 {
			_, err = client.New(t.Context(), openai.ChatCompletionNewParams{
				Model: "caller-model",
				Messages: []openai.ChatCompletionMessageParamUnion{
					openai.UserMessage("hello"),
				},
			})
			require.NoError(t, err)
		}

		primaryBefore := primaryHits.Load()
		_, err = client.New(t.Context(), openai.ChatCompletionNewParams{
			Model: "caller-model",
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("hello"),
			},
		})
		require.NoError(t, err)
		require.Equal(t, primaryBefore, primaryHits.Load())

		primaryHealthy.Store(true)
		clock.Advance(time.Minute)
		_, err = client.New(t.Context(), openai.ChatCompletionNewParams{
			Model: "caller-model",
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("hello"),
			},
		})
		require.NoError(t, err)
		require.Equal(t, primaryBefore+1, primaryHits.Load())

		_, err = client.New(t.Context(), openai.ChatCompletionNewParams{
			Model: "caller-model",
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("hello"),
			},
		})
		require.NoError(t, err)
		require.Equal(t, primaryBefore+2, primaryHits.Load())
	})

	t.Run("failed probe reopens circuit and skips primary again", func(t *testing.T) {
		clock := &fakeClock{now: time.Unix(2000, 0)}
		var primaryHits atomic.Int32
		primaryServer := chatCompletionServer(t, http.StatusServiceUnavailable, "", nil, &primaryHits)
		t.Cleanup(primaryServer.Close)

		fallbackServer := chatCompletionServer(t, http.StatusOK, `{"fallback":true}`, nil, nil)
		t.Cleanup(fallbackServer.Close)

		client, err := NewCompletionClient(Options{
			Primary: VLLMConfig{
				ChatBaseURL:       primaryServer.URL,
				EmbeddingsBaseURL: primaryServer.URL,
				ChatModel:         "vllm-chat",
				EmbeddingModel:    "vllm-embedding",
			},
			Fallbacks: []Provider{OpenRouterConfig{
				APIKey:         "openrouter-key",
				BaseURL:        fallbackServer.URL,
				ChatModel:      "openrouter-chat",
				EmbeddingModel: "openrouter-embedding",
			}},
			Breaker: BreakerConfig{
				FailureThreshold:         1,
				OpenDuration:             time.Minute,
				SuccessfulProbeThreshold: 1,
				Now:                      clock.Now,
			},
		})
		require.NoError(t, err)

		_, err = client.New(t.Context(), openai.ChatCompletionNewParams{
			Model: "caller-model",
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("hello"),
			},
		})
		require.NoError(t, err)
		primaryBeforeProbe := primaryHits.Load()

		clock.Advance(time.Minute)
		_, err = client.New(t.Context(), openai.ChatCompletionNewParams{
			Model: "caller-model",
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("hello"),
			},
		})
		require.NoError(t, err)
		require.Equal(t, primaryBeforeProbe+1, primaryHits.Load())

		primaryAfterProbe := primaryHits.Load()
		_, err = client.New(t.Context(), openai.ChatCompletionNewParams{
			Model: "caller-model",
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("hello"),
			},
		})
		require.NoError(t, err)
		require.Equal(t, primaryAfterProbe, primaryHits.Load())
	})

	t.Run("requires configured successful probes before closing", func(t *testing.T) {
		clock := &fakeClock{now: time.Unix(3000, 0)}
		var primaryHits atomic.Int32
		var primaryHealthy atomic.Bool
		primaryServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			primaryHits.Add(1)
			captureJSONRequest(t, r, nil)
			if !primaryHealthy.Load() {
				w.WriteHeader(http.StatusServiceUnavailable)
				return
			}
			writeChatCompletion(w, `{"primary":true}`)
		}))
		t.Cleanup(primaryServer.Close)

		fallbackServer := chatCompletionServer(t, http.StatusOK, `{"fallback":true}`, nil, nil)
		t.Cleanup(fallbackServer.Close)

		client, err := NewCompletionClient(Options{
			Primary: VLLMConfig{
				ChatBaseURL:       primaryServer.URL,
				EmbeddingsBaseURL: primaryServer.URL,
				ChatModel:         "vllm-chat",
				EmbeddingModel:    "vllm-embedding",
			},
			Fallbacks: []Provider{OpenRouterConfig{
				APIKey:         "openrouter-key",
				BaseURL:        fallbackServer.URL,
				ChatModel:      "openrouter-chat",
				EmbeddingModel: "openrouter-embedding",
			}},
			Breaker: BreakerConfig{
				FailureThreshold:         1,
				OpenDuration:             time.Minute,
				ProbeInterval:            10 * time.Second,
				SuccessfulProbeThreshold: 2,
				Now:                      clock.Now,
			},
		})
		require.NoError(t, err)

		_, err = client.New(t.Context(), openai.ChatCompletionNewParams{
			Model: "caller-model",
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("hello"),
			},
		})
		require.NoError(t, err)

		primaryHealthy.Store(true)
		clock.Advance(time.Minute)
		primaryBeforeProbe := primaryHits.Load()
		_, err = client.New(t.Context(), openai.ChatCompletionNewParams{
			Model: "caller-model",
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("hello"),
			},
		})
		require.NoError(t, err)
		require.Equal(t, primaryBeforeProbe+1, primaryHits.Load())

		primaryAfterFirstProbe := primaryHits.Load()
		_, err = client.New(t.Context(), openai.ChatCompletionNewParams{
			Model: "caller-model",
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("hello"),
			},
		})
		require.NoError(t, err)
		require.Equal(t, primaryAfterFirstProbe, primaryHits.Load())

		clock.Advance(10 * time.Second)
		_, err = client.New(t.Context(), openai.ChatCompletionNewParams{
			Model: "caller-model",
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("hello"),
			},
		})
		require.NoError(t, err)
		require.Equal(t, primaryAfterFirstProbe+1, primaryHits.Load())

		_, err = client.New(t.Context(), openai.ChatCompletionNewParams{
			Model: "caller-model",
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("hello"),
			},
		})
		require.NoError(t, err)
		require.Equal(t, primaryAfterFirstProbe+2, primaryHits.Load())
	})
}

func TestEmbeddingClient_New(t *testing.T) {
	t.Parallel()

	t.Run("happy path uses embedding model and mirrors OpenAI call shape", func(t *testing.T) {
		request := &capturedRequest{}
		server := embeddingServer(t, http.StatusOK, request, nil)
		t.Cleanup(server.Close)

		client, err := NewEmbeddingClient(Options{
			Primary: VLLMConfig{
				APIKey:            "local-key",
				ChatBaseURL:       server.URL,
				EmbeddingsBaseURL: server.URL,
				ChatModel:         "vllm-chat",
				EmbeddingModel:    "vllm-embedding",
				Headers:           map[string]string{"X-VLLM-Header": "present"},
				QueryParams:       map[string]string{"backend": "vllm"},
			},
		})
		require.NoError(t, err)

		response, err := client.New(t.Context(), openai.EmbeddingNewParams{
			Model: "caller-model",
			Input: openai.EmbeddingNewParamsInputUnion{
				OfString: openai.String("hello"),
			},
		})
		require.NoError(t, err)
		require.NotNil(t, response)
		require.Equal(t, "/embeddings", request.Path)
		require.Equal(t, "Bearer local-key", request.Header.Get("Authorization"))
		require.Equal(t, "present", request.Header.Get("X-VLLM-Header"))
		require.Equal(t, "vllm", request.Query.Get("backend"))
		require.Equal(t, "vllm-embedding", request.Body["model"])
	})

	t.Run("trigger failure falls back with fallback embedding model", func(t *testing.T) {
		primaryServer := embeddingServer(t, http.StatusServiceUnavailable, nil, nil)
		t.Cleanup(primaryServer.Close)

		fallbackRequest := &capturedRequest{}
		fallbackServer := embeddingServer(t, http.StatusOK, fallbackRequest, nil)
		t.Cleanup(fallbackServer.Close)

		client, err := NewEmbeddingClient(Options{
			Primary: VLLMConfig{
				ChatBaseURL:       primaryServer.URL,
				EmbeddingsBaseURL: primaryServer.URL,
				ChatModel:         "vllm-chat",
				EmbeddingModel:    "vllm-embedding",
			},
			Fallbacks: []Provider{OpenRouterConfig{
				APIKey:         "openrouter-key",
				BaseURL:        fallbackServer.URL,
				ChatModel:      "openrouter-chat",
				EmbeddingModel: "openrouter-embedding",
			}},
		})
		require.NoError(t, err)

		response, err := client.New(t.Context(), openai.EmbeddingNewParams{
			Model: "caller-model",
			Input: openai.EmbeddingNewParamsInputUnion{
				OfString: openai.String("hello"),
			},
		})
		require.NoError(t, err)
		require.NotNil(t, response)
		require.Equal(t, "openrouter-embedding", fallbackRequest.Body["model"])
		require.Equal(t, "Bearer openrouter-key", fallbackRequest.Header.Get("Authorization"))
	})
}

type capturedRequest struct {
	Path   string
	Query  url.Values
	Header http.Header
	Body   map[string]any
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

func embeddingServer(t *testing.T, status int, captured *capturedRequest, hits *atomic.Int32) *httptest.Server {
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
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"object": "list",
			"model":  "test-embedding-model",
			"data": []map[string]any{
				{
					"object":    "embedding",
					"index":     0,
					"embedding": []float64{0.1, 0.2, 0.3},
				},
			},
			"usage": map[string]any{
				"prompt_tokens": 1,
				"total_tokens":  1,
			},
		})
	}))
}

func captureJSONRequest(t *testing.T, r *http.Request, captured *capturedRequest) {
	t.Helper()

	var body map[string]any
	require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
	if captured == nil {
		return
	}
	captured.Path = r.URL.Path
	captured.Query = r.URL.Query()
	captured.Header = r.Header.Clone()
	captured.Body = body
}

func writeChatCompletion(w http.ResponseWriter, content string) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"id":     "chatcmpl-test",
		"object": "chat.completion",
		"model":  "test-chat-model",
		"choices": []map[string]any{
			{
				"index": 0,
				"message": map[string]any{
					"role":    "assistant",
					"content": content,
				},
				"finish_reason": "stop",
			},
		},
		"usage": map[string]any{
			"prompt_tokens":     1,
			"completion_tokens": 1,
			"total_tokens":      2,
		},
	})
}

func headerHTTPClient(header string, value string) *http.Client {
	return &http.Client{
		Transport: headerRoundTripper{
			base:   http.DefaultTransport,
			header: header,
			value:  value,
		},
		Timeout: 5 * time.Second,
	}
}

type headerRoundTripper struct {
	base   http.RoundTripper
	header string
	value  string
}

func (t headerRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	req.Header.Set(t.header, t.value)
	return t.base.RoundTrip(req)
}

type fakeClock struct {
	now time.Time
}

func (c *fakeClock) Now() time.Time {
	return c.now
}

func (c *fakeClock) Advance(d time.Duration) {
	c.now = c.now.Add(d)
}
