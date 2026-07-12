package llm_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	"github.com/DavidNix/safeagent/llm"
	"github.com/stretchr/testify/require"
)

func TestEmbeddingClient_Embed(t *testing.T) {
	t.Parallel()

	t.Run("happy path sends batch request and decodes response", func(t *testing.T) {
		request := &capturedRequest{}
		server := embeddingServer(t, http.StatusOK, request, nil)
		t.Cleanup(server.Close)
		dimensions := 2
		client := llm.NewEmbeddingClient(llm.EmbeddingConfig{Model: "embed-test"},
			llm.WithBaseURL(server.URL),
			llm.WithAPIKey("test-key"),
			llm.WithHeaders(map[string]string{"X-Test": "yes"}),
			llm.WithQueryParams(map[string]string{"route": "local"}),
			llm.WithExtraFields(map[string]any{"provider.zdr": true}),
		)

		resp, err := client.Embed(t.Context(), llm.EmbeddingRequest{
			Input:      []string{"first", "second"},
			Dimensions: &dimensions,
			User:       "user-1",
		})

		require.NoError(t, err)
		require.Equal(t, "/embeddings", request.Path)
		require.Equal(t, "Bearer test-key", request.Header.Get("Authorization"))
		require.Equal(t, "yes", request.Header.Get("X-Test"))
		require.Equal(t, "local", request.Query.Get("route"))
		require.JSONEq(t, `{
			"model":"embed-test",
			"input":["first","second"],
			"dimensions":2,
			"user":"user-1",
			"encoding_format":"float",
			"provider":{"zdr":true}
		}`, string(request.Body))
		require.Equal(t, "embed-test", resp.Model)
		require.Equal(t, []llm.Embedding{
			{Object: "embedding", Index: 0, Embedding: []float32{0.25, -0.5}},
			{Object: "embedding", Index: 1, Embedding: []float32{0.75, 1}},
		}, resp.Data)
		require.Equal(t, llm.EmbeddingUsage{PromptTokens: 4, TotalTokens: 4}, resp.Usage)
	})

	t.Run("provider constructors apply vllm and openrouter configuration", func(t *testing.T) {
		primaryServer := embeddingServer(t, http.StatusServiceUnavailable, nil, nil)
		t.Cleanup(primaryServer.Close)
		fallbackRequest := &capturedRequest{}
		fallbackServer := embeddingServer(t, http.StatusOK, fallbackRequest, nil)
		t.Cleanup(fallbackServer.Close)
		primary := llm.NewVLLMEmbedding(llm.VLLMEmbeddingConfig{
			ProviderID: "vllm-primary",
			BaseURL:    primaryServer.URL,
			Model:      "vllm-embed",
		})
		fallback := llm.NewOpenRouterEmbedding(llm.OpenRouterEmbeddingConfig{
			ProviderID:               "openrouter-fallback",
			APIKey:                   "openrouter-key",
			BaseURL:                  fallbackServer.URL,
			Model:                    "openrouter-embed",
			SiteURL:                  "https://safeagent.example",
			AppTitle:                 "SafeAgent",
			RequireZeroDataRetention: true,
			Headers:                  map[string]string{"X-Test": "yes"},
			QueryParams:              map[string]string{"route": "openrouter"},
		})
		events := make(chan llm.FailoverEvent, 1)
		breaker := llm.NewEmbeddingCircuitBreakerWithConfig(llm.BreakerConfig{
			OnFailover: func(_ context.Context, event llm.FailoverEvent) {
				events <- event
			},
		}, primary, fallback)

		_, err := breaker.Embed(t.Context(), llm.EmbeddingRequest{Input: []string{"first", "second"}})

		require.NoError(t, err)
		require.Equal(t, "openrouter-embed", fallbackRequest.JSONBody["model"])
		require.Equal(t, "Bearer openrouter-key", fallbackRequest.Header.Get("Authorization"))
		require.Equal(t, "https://safeagent.example", fallbackRequest.Header.Get("HTTP-Referer"))
		require.Equal(t, "SafeAgent", fallbackRequest.Header.Get("X-Title"))
		require.Equal(t, "yes", fallbackRequest.Header.Get("X-Test"))
		require.Equal(t, "openrouter", fallbackRequest.Query.Get("route"))
		provider := fallbackRequest.JSONBody["provider"].(map[string]any)
		require.Equal(t, true, provider["zdr"])
		event := <-events
		require.Equal(t, "vllm-primary", event.FromProvider)
		require.Equal(t, "openrouter-fallback", event.ToProvider)
	})

	t.Run("e2e vllm embedding client", func(t *testing.T) {
		skipE2EInShortMode(t)
		baseURL := requireE2EEnv(t, vllmEmbeddingBaseURLEnv)
		ctx, cancel := e2eContext(t)
		defer cancel()
		client := newVLLMEmbeddingE2EClient(t, ctx, baseURL)

		resp, err := client.Embed(ctx, llm.EmbeddingRequest{Input: []string{"SafeAgent embedding end-to-end test"}})

		require.NoError(t, err)
		require.NotNil(t, resp)
		require.Len(t, resp.Data, 1)
		require.NotEmpty(t, resp.Data[0].Embedding)
	})

	t.Run("status error makes one request", func(t *testing.T) {
		var hits atomic.Int32
		server := embeddingServer(t, http.StatusBadRequest, nil, &hits)
		t.Cleanup(server.Close)
		client := llm.NewEmbeddingClient(llm.EmbeddingConfig{Model: "embed-test"}, llm.WithBaseURL(server.URL))

		_, err := client.Embed(t.Context(), llm.EmbeddingRequest{Input: []string{"hello"}})

		require.EqualError(t, err, "embeddings returned status 400: bad embeddings request")
		require.Equal(t, int32(1), hits.Load())
		var statusErr *llm.StatusError
		require.ErrorAs(t, err, &statusErr)
	})

	t.Run("wrong response dimensions are rejected", func(t *testing.T) {
		server := rawJSONServer(t, http.StatusOK, `{
			"object":"list",
			"data":[{"object":"embedding","index":0,"embedding":[0.25]}],
			"model":"embed-test",
			"usage":{"prompt_tokens":1,"total_tokens":1}
		}`)
		t.Cleanup(server.Close)
		client := llm.NewEmbeddingClient(llm.EmbeddingConfig{Model: "embed-test"}, llm.WithBaseURL(server.URL))
		dimensions := 2

		_, err := client.Embed(t.Context(), llm.EmbeddingRequest{Input: []string{"hello"}, Dimensions: &dimensions})

		require.EqualError(t, err, "validate embeddings response: received 1 dimensions; requested 2")
	})

	t.Run("response body exceeding limit returns embedding error", func(t *testing.T) {
		server := embeddingServer(t, http.StatusOK, nil, nil)
		t.Cleanup(server.Close)
		client := llm.NewEmbeddingClient(llm.EmbeddingConfig{Model: "embed-test"},
			llm.WithBaseURL(server.URL), llm.WithMaxResponseBytes(8))

		_, err := client.Embed(t.Context(), llm.EmbeddingRequest{Input: []string{"hello"}})

		require.EqualError(t, err, "read embeddings response: embeddings response exceeded 8 bytes")
		var tooLarge *llm.ResponseTooLargeError
		require.ErrorAs(t, err, &tooLarge)
	})

	t.Run("request timeout", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			client := llm.NewEmbeddingClient(llm.EmbeddingConfig{Model: "embed-test"},
				llm.WithRequestTimeout(time.Second),
				llm.WithHTTPClient(&http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
					<-req.Context().Done()
					return nil, req.Context().Err()
				})}),
			)

			_, err := client.Embed(t.Context(), llm.EmbeddingRequest{Input: []string{"hello"}})

			require.ErrorIs(t, err, context.DeadlineExceeded)
		})
	})

	t.Run("caller cancellation takes precedence over serialization", func(t *testing.T) {
		client := llm.NewEmbeddingClient(llm.EmbeddingConfig{Model: "embed-test"},
			llm.WithExtraFields(map[string]any{"invalid": func() {}}))
		ctx, cancel := context.WithCancel(t.Context())
		cancel()

		_, err := client.Embed(ctx, llm.EmbeddingRequest{Input: []string{"hello"}})

		require.ErrorIs(t, err, context.Canceled)
	})
}

func TestEmbeddingCircuitBreaker_Embed(t *testing.T) {
	t.Parallel()

	t.Run("serialization error does not fall back", func(t *testing.T) {
		var fallbackHits atomic.Int32
		fallbackServer := embeddingServer(t, http.StatusOK, nil, &fallbackHits)
		t.Cleanup(fallbackServer.Close)
		primary := llm.NewEmbeddingClient(llm.EmbeddingConfig{Model: "primary"},
			llm.WithExtraFields(map[string]any{"invalid": func() {}}))
		fallback := llm.NewEmbeddingClient(llm.EmbeddingConfig{Model: "fallback"}, llm.WithBaseURL(fallbackServer.URL))
		breaker := llm.NewEmbeddingCircuitBreaker(primary, fallback)

		_, err := breaker.Embed(t.Context(), llm.EmbeddingRequest{Input: []string{"hello", "world"}})

		require.EqualError(t, err, "marshal embeddings request: json: unsupported type: func()")
		require.Equal(t, int32(0), fallbackHits.Load())
	})

	t.Run("fallback serialization error stops chain", func(t *testing.T) {
		primaryServer := embeddingServer(t, http.StatusInternalServerError, nil, nil)
		t.Cleanup(primaryServer.Close)
		var finalHits atomic.Int32
		finalServer := embeddingServer(t, http.StatusOK, nil, &finalHits)
		t.Cleanup(finalServer.Close)
		primary := llm.NewEmbeddingClient(llm.EmbeddingConfig{Model: "primary"}, llm.WithBaseURL(primaryServer.URL))
		invalid := llm.NewEmbeddingClient(llm.EmbeddingConfig{Model: "invalid"},
			llm.WithExtraFields(map[string]any{"invalid": func() {}}))
		final := llm.NewEmbeddingClient(llm.EmbeddingConfig{Model: "final"}, llm.WithBaseURL(finalServer.URL))
		breaker := llm.NewEmbeddingCircuitBreaker(primary, invalid, final)

		_, err := breaker.Embed(t.Context(), llm.EmbeddingRequest{Input: []string{"hello", "world"}})

		require.EqualError(t, err, "marshal embeddings request: json: unsupported type: func()")
		require.Equal(t, int32(0), finalHits.Load())
	})

	t.Run("bad request falls back without opening circuit", func(t *testing.T) {
		var primaryHits atomic.Int32
		primaryServer := embeddingServer(t, http.StatusBadRequest, nil, &primaryHits)
		t.Cleanup(primaryServer.Close)
		fallbackServer := embeddingServer(t, http.StatusOK, nil, nil)
		t.Cleanup(fallbackServer.Close)
		primary := llm.NewEmbeddingClient(llm.EmbeddingConfig{Model: "shared"}, llm.WithBaseURL(primaryServer.URL))
		fallback := llm.NewEmbeddingClient(llm.EmbeddingConfig{Model: "shared"}, llm.WithBaseURL(fallbackServer.URL))
		breaker := llm.NewEmbeddingCircuitBreakerWithConfig(llm.BreakerConfig{FailureThreshold: 1}, primary, fallback)

		for range 2 {
			_, err := breaker.Embed(t.Context(), llm.EmbeddingRequest{Input: []string{"hello", "world"}})
			require.NoError(t, err)
		}

		require.Equal(t, int32(2), primaryHits.Load())
	})

	t.Run("local response limit falls back without opening circuit", func(t *testing.T) {
		var primaryHits atomic.Int32
		primaryServer := embeddingServer(t, http.StatusOK, nil, &primaryHits)
		t.Cleanup(primaryServer.Close)
		fallbackServer := embeddingServer(t, http.StatusOK, nil, nil)
		t.Cleanup(fallbackServer.Close)
		primary := llm.NewEmbeddingClient(llm.EmbeddingConfig{Model: "shared"},
			llm.WithBaseURL(primaryServer.URL), llm.WithMaxResponseBytes(8))
		fallback := llm.NewEmbeddingClient(llm.EmbeddingConfig{Model: "shared"}, llm.WithBaseURL(fallbackServer.URL))
		breaker := llm.NewEmbeddingCircuitBreakerWithConfig(llm.BreakerConfig{FailureThreshold: 1}, primary, fallback)

		for range 2 {
			_, err := breaker.Embed(t.Context(), llm.EmbeddingRequest{Input: []string{"hello", "world"}})
			require.NoError(t, err)
		}

		require.Equal(t, int32(2), primaryHits.Load())
	})

	t.Run("invalid successful response falls back and opens circuit", func(t *testing.T) {
		var primaryHits atomic.Int32
		primary := llm.NewEmbeddingClient(llm.EmbeddingConfig{Model: "shared"},
			llm.WithHTTPClient(&http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				primaryHits.Add(1)
				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     make(http.Header),
					Body:       io.NopCloser(strings.NewReader(`{"data":[]}`)),
				}, nil
			})}))
		fallbackServer := embeddingServer(t, http.StatusOK, nil, nil)
		t.Cleanup(fallbackServer.Close)
		fallback := llm.NewEmbeddingClient(llm.EmbeddingConfig{Model: "shared"}, llm.WithBaseURL(fallbackServer.URL))
		breaker := llm.NewEmbeddingCircuitBreakerWithConfig(llm.BreakerConfig{FailureThreshold: 1}, primary, fallback)

		for range 2 {
			_, err := breaker.Embed(t.Context(), llm.EmbeddingRequest{Input: []string{"hello", "world"}})
			require.NoError(t, err)
		}

		require.Equal(t, int32(1), primaryHits.Load())
	})

	t.Run("caller cancellation during primary does not fall back", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		primary := llm.NewEmbeddingClient(llm.EmbeddingConfig{Model: "shared"},
			llm.WithHTTPClient(&http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				cancel()
				return nil, context.Canceled
			})}))
		var fallbackHits atomic.Int32
		fallbackServer := embeddingServer(t, http.StatusOK, nil, &fallbackHits)
		t.Cleanup(fallbackServer.Close)
		fallback := llm.NewEmbeddingClient(llm.EmbeddingConfig{Model: "shared"}, llm.WithBaseURL(fallbackServer.URL))
		breaker := llm.NewEmbeddingCircuitBreaker(primary, fallback)

		_, err := breaker.Embed(ctx, llm.EmbeddingRequest{Input: []string{"hello", "world"}})

		require.ErrorIs(t, err, context.Canceled)
		require.Equal(t, int32(0), fallbackHits.Load())
	})

	t.Run("caller cancellation releases recovery probe", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			var mode atomic.Int32
			var primaryHits atomic.Int32
			cancelProbe := func() {}
			primary := llm.NewEmbeddingClient(llm.EmbeddingConfig{Model: "shared"},
				llm.WithHTTPClient(&http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
					primaryHits.Add(1)
					switch mode.Load() {
					case 0:
						return testHTTPResponse(http.StatusForbidden, ""), nil
					case 1:
						cancelProbe()
						return nil, context.Canceled
					default:
						return testEmbeddingHTTPResponse(), nil
					}
				})}))
			fallback := llm.NewEmbeddingClient(llm.EmbeddingConfig{Model: "shared"},
				llm.WithHTTPClient(&http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
					return testEmbeddingHTTPResponse(), nil
				})}))
			breaker := llm.NewEmbeddingCircuitBreakerWithConfig(llm.BreakerConfig{
				FailureThreshold: 1,
				OpenDuration:     time.Minute,
			}, primary, fallback)

			_, err := breaker.Embed(t.Context(), llm.EmbeddingRequest{Input: []string{"hello", "world"}})
			require.NoError(t, err)
			time.Sleep(time.Minute)

			mode.Store(1)
			probeCtx, cancel := context.WithCancel(t.Context())
			cancelProbe = cancel
			_, err = breaker.Embed(probeCtx, llm.EmbeddingRequest{Input: []string{"hello", "world"}})
			require.ErrorIs(t, err, context.Canceled)

			mode.Store(2)
			resp, err := breaker.Embed(t.Context(), llm.EmbeddingRequest{Input: []string{"hello", "world"}})
			require.NoError(t, err)
			require.Len(t, resp.Data, 2)
			require.Equal(t, int32(3), primaryHits.Load())
		})
	})

	t.Run("failover events identify each failed provider", func(t *testing.T) {
		primaryServer := embeddingServer(t, http.StatusBadRequest, nil, nil)
		t.Cleanup(primaryServer.Close)
		firstFallbackServer := embeddingServer(t, http.StatusInternalServerError, nil, nil)
		t.Cleanup(firstFallbackServer.Close)
		finalServer := embeddingServer(t, http.StatusOK, nil, nil)
		t.Cleanup(finalServer.Close)
		primary := llm.NewEmbeddingClient(llm.EmbeddingConfig{Model: "shared"},
			llm.WithProviderID("primary"), llm.WithBaseURL(primaryServer.URL))
		firstFallback := llm.NewEmbeddingClient(llm.EmbeddingConfig{Model: "shared"},
			llm.WithProviderID("fallback-1"), llm.WithBaseURL(firstFallbackServer.URL))
		final := llm.NewEmbeddingClient(llm.EmbeddingConfig{Model: "shared"},
			llm.WithProviderID("fallback-2"), llm.WithBaseURL(finalServer.URL))
		events := make(chan llm.FailoverEvent, 2)
		breaker := llm.NewEmbeddingCircuitBreakerWithConfig(llm.BreakerConfig{
			FailureThreshold: 1,
			OnFailover: func(_ context.Context, event llm.FailoverEvent) {
				events <- event
			},
		}, primary, firstFallback, final)

		resp, err := breaker.Embed(t.Context(), llm.EmbeddingRequest{Input: []string{"hello", "world"}})

		require.NoError(t, err)
		require.Len(t, resp.Data, 2)
		require.Len(t, events, 2)
		first := <-events
		require.Equal(t, "primary", first.FromProvider)
		require.Equal(t, 0, first.FromProviderIndex)
		require.Equal(t, "fallback-1", first.ToProvider)
		require.False(t, first.CountedAgainstBreaker)
		second := <-events
		require.Equal(t, "fallback-1", second.FromProvider)
		require.Equal(t, 1, second.FromProviderIndex)
		require.Equal(t, "fallback-2", second.ToProvider)
		require.False(t, second.CountedAgainstBreaker)
	})

	t.Run("trigger failure falls back with the same request", func(t *testing.T) {
		primaryServer := embeddingServer(t, http.StatusServiceUnavailable, nil, nil)
		t.Cleanup(primaryServer.Close)
		fallbackRequest := &capturedRequest{}
		fallbackServer := embeddingServer(t, http.StatusOK, fallbackRequest, nil)
		t.Cleanup(fallbackServer.Close)
		primary := llm.NewEmbeddingClient(llm.EmbeddingConfig{Model: "primary"}, llm.WithBaseURL(primaryServer.URL))
		fallback := llm.NewEmbeddingClient(llm.EmbeddingConfig{Model: "fallback"}, llm.WithBaseURL(fallbackServer.URL))
		breaker := llm.NewEmbeddingCircuitBreaker(primary, fallback)
		dimensions := 2

		resp, err := breaker.Embed(t.Context(), llm.EmbeddingRequest{Input: []string{"hello", "world"}, Dimensions: &dimensions})

		require.NoError(t, err)
		require.Len(t, resp.Data, 2)
		require.Equal(t, "fallback", fallbackRequest.JSONBody["model"])
		require.Equal(t, []any{"hello", "world"}, fallbackRequest.JSONBody["input"])
		require.InDelta(t, 2, fallbackRequest.JSONBody["dimensions"], 0.0001)
	})

	t.Run("bad request falls back without same-client retry", func(t *testing.T) {
		var primaryHits atomic.Int32
		primaryServer := embeddingServer(t, http.StatusBadRequest, nil, &primaryHits)
		t.Cleanup(primaryServer.Close)
		var fallbackHits atomic.Int32
		fallbackServer := embeddingServer(t, http.StatusOK, nil, &fallbackHits)
		t.Cleanup(fallbackServer.Close)
		primary := llm.NewEmbeddingClient(llm.EmbeddingConfig{Model: "shared"}, llm.WithBaseURL(primaryServer.URL))
		fallback := llm.NewEmbeddingClient(llm.EmbeddingConfig{Model: "shared"}, llm.WithBaseURL(fallbackServer.URL))
		breaker := llm.NewEmbeddingCircuitBreaker(primary, fallback)

		resp, err := breaker.Embed(t.Context(), llm.EmbeddingRequest{Input: []string{"hello", "world"}})

		require.NoError(t, err)
		require.Len(t, resp.Data, 2)
		require.Equal(t, int32(1), primaryHits.Load())
		require.Equal(t, int32(1), fallbackHits.Load())
	})

	t.Run("e2e primary failure falls back to vllm", func(t *testing.T) {
		skipE2EInShortMode(t)
		baseURL := requireE2EEnv(t, vllmEmbeddingBaseURLEnv)
		ctx, cancel := e2eContext(t)
		defer cancel()
		fallback := newVLLMEmbeddingE2EClient(t, ctx, baseURL)
		primaryServer := embeddingServer(t, http.StatusServiceUnavailable, nil, nil)
		t.Cleanup(primaryServer.Close)
		primary := llm.NewEmbeddingClient(llm.EmbeddingConfig{
			Model: "unavailable",
		}, llm.WithBaseURL(primaryServer.URL))
		breaker := llm.NewEmbeddingCircuitBreaker(primary, fallback)

		resp, err := breaker.Embed(ctx, llm.EmbeddingRequest{Input: []string{"SafeAgent embedding fallback end-to-end test"}})

		require.NoError(t, err)
		require.Len(t, resp.Data, 1)
		require.NotEmpty(t, resp.Data[0].Embedding)
	})
}

func newVLLMEmbeddingE2EClient(t *testing.T, ctx context.Context, baseURL string) *llm.EmbeddingClient {
	t.Helper()
	model := vllmEmbeddingModel(t, ctx, baseURL)
	return llm.NewVLLMEmbedding(llm.VLLMEmbeddingConfig{BaseURL: baseURL, Model: model})
}

func vllmEmbeddingModel(t *testing.T, ctx context.Context, baseURL string) string {
	t.Helper()
	modelsClient := llm.NewClient("ignored", llm.WithBaseURL(baseURL))
	models, err := modelsClient.Models(ctx)
	require.NoError(t, err)
	require.NotEmpty(t, models.Data, "vLLM embeddings models response returned no models")
	require.NotEmpty(t, models.Data[0].ID, "first vLLM embedding model ID is empty")
	return models.Data[0].ID
}

func embeddingServer(t *testing.T, status int, captured *capturedRequest, hits *atomic.Int32) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if hits != nil {
			hits.Add(1)
		}
		if err := captureJSONRequest(r, captured); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if status >= http.StatusBadRequest {
			w.WriteHeader(status)
			_, _ = w.Write([]byte("bad embeddings request"))
			return
		}
		writeEmbeddingResponse(w)
	}))
}

func writeEmbeddingResponse(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write([]byte(`{
		"object":"list",
		"data":[
			{"object":"embedding","index":0,"embedding":[0.25,-0.5]},
			{"object":"embedding","index":1,"embedding":[0.75,1.0]}
		],
		"model":"embed-test",
		"usage":{"prompt_tokens":4,"total_tokens":4}
	}`))
}

func testEmbeddingHTTPResponse() *http.Response {
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     make(http.Header),
		Body: io.NopCloser(strings.NewReader(`{
			"data":[
				{"index":0,"embedding":[0.25,-0.5]},
				{"index":1,"embedding":[0.75,1.0]}
			]
		}`)),
	}
}
