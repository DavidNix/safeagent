package llm_test

import (
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/DavidNix/safeagent/llm"
	"github.com/stretchr/testify/require"
)

func TestCircuitBreaker_Complete(t *testing.T) {
	t.Parallel()

	t.Run("trigger failure falls back and non-trigger does not", func(t *testing.T) {
		primaryServer := chatCompletionServer(t, http.StatusServiceUnavailable, "", nil, nil)
		t.Cleanup(primaryServer.Close)

		fallbackRequest := &capturedRequest{}
		fallbackServer := chatCompletionServer(t, http.StatusOK, `{"fallback":true}`, fallbackRequest, nil)
		t.Cleanup(fallbackServer.Close)

		primary := llm.NewClient("primary", llm.WithBaseURL(primaryServer.URL))
		fallback := llm.NewClient("fallback", llm.WithBaseURL(fallbackServer.URL))
		breaker := llm.NewCircuitBreaker(primary, fallback)

		resp, err := breaker.Complete(t.Context(), llm.ChatRequest{Messages: []llm.ChatMessage{{Role: "user", Content: "hello"}}})

		require.NoError(t, err)
		require.NotNil(t, resp)
		require.Equal(t, "fallback", fallbackRequest.JSONBody["model"])
	})

	t.Run("e2e primary failure falls back to vllm", func(t *testing.T) {
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

		primary := newOpenRouterE2EClient(apiKey, primaryServer.URL)
		fallback := newVLLME2EClient(t, ctx, baseURL)
		breaker := llm.NewCircuitBreakerWithConfig(llm.BreakerConfig{FailureThreshold: 1}, primary, fallback)

		runLLME2ECompletion(t, ctx, breaker, "SAFEAGENT_LLM_CIRCUIT_BREAKER_E2E")
		require.Equal(t, int32(1), primaryHits.Load())
	})

	t.Run("non-trigger primary error does not fall back", func(t *testing.T) {
		primaryServer := chatCompletionServer(t, http.StatusBadRequest, "", nil, nil)
		t.Cleanup(primaryServer.Close)

		var fallbackHits atomic.Int32
		fallbackServer := chatCompletionServer(t, http.StatusOK, `{"fallback":true}`, nil, &fallbackHits)
		t.Cleanup(fallbackServer.Close)

		primary := llm.NewClient("primary", llm.WithBaseURL(primaryServer.URL))
		fallback := llm.NewClient("fallback", llm.WithBaseURL(fallbackServer.URL))
		breaker := llm.NewCircuitBreaker(primary, fallback)

		_, err := breaker.Complete(t.Context(), llm.ChatRequest{Messages: []llm.ChatMessage{{Role: "user", Content: "hello"}}})

		require.EqualError(t, err, "chat completions returned status 400")
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

		primary := llm.NewClient("primary", llm.WithBaseURL(primaryServer.URL))
		fallback := llm.NewClient("fallback", llm.WithBaseURL(fallbackServer.URL))
		breaker := llm.NewCircuitBreakerWithConfig(llm.BreakerConfig{
			FailureThreshold:         2,
			OpenDuration:             time.Minute,
			ProbeInterval:            10 * time.Second,
			SuccessfulProbeThreshold: 1,
			Now:                      clock.Now,
		}, primary, fallback)

		for range 2 {
			_, err := breaker.Complete(t.Context(), llm.ChatRequest{Messages: []llm.ChatMessage{{Role: "user", Content: "hello"}}})
			require.NoError(t, err)
		}

		primaryBefore := primaryHits.Load()
		_, err := breaker.Complete(t.Context(), llm.ChatRequest{Messages: []llm.ChatMessage{{Role: "user", Content: "hello"}}})
		require.NoError(t, err)
		require.Equal(t, primaryBefore, primaryHits.Load())

		primaryHealthy.Store(true)
		clock.Advance(time.Minute)
		_, err = breaker.Complete(t.Context(), llm.ChatRequest{Messages: []llm.ChatMessage{{Role: "user", Content: "hello"}}})
		require.NoError(t, err)
		require.Equal(t, primaryBefore+1, primaryHits.Load())

		_, err = breaker.Complete(t.Context(), llm.ChatRequest{Messages: []llm.ChatMessage{{Role: "user", Content: "hello"}}})
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

		primary := llm.NewClient("primary", llm.WithBaseURL(primaryServer.URL))
		fallback := llm.NewClient("fallback", llm.WithBaseURL(fallbackServer.URL))
		breaker := llm.NewCircuitBreakerWithConfig(llm.BreakerConfig{
			FailureThreshold:         1,
			OpenDuration:             time.Minute,
			SuccessfulProbeThreshold: 1,
			Now:                      clock.Now,
		}, primary, fallback)

		_, err := breaker.Complete(t.Context(), llm.ChatRequest{Messages: []llm.ChatMessage{{Role: "user", Content: "hello"}}})
		require.NoError(t, err)
		primaryBeforeProbe := primaryHits.Load()

		clock.Advance(time.Minute)
		_, err = breaker.Complete(t.Context(), llm.ChatRequest{Messages: []llm.ChatMessage{{Role: "user", Content: "hello"}}})
		require.NoError(t, err)
		require.Equal(t, primaryBeforeProbe+1, primaryHits.Load())

		primaryAfterProbe := primaryHits.Load()
		_, err = breaker.Complete(t.Context(), llm.ChatRequest{Messages: []llm.ChatMessage{{Role: "user", Content: "hello"}}})
		require.NoError(t, err)
		require.Equal(t, primaryAfterProbe, primaryHits.Load())
	})

	t.Run("pass through without fallbacks", func(t *testing.T) {
		request := &capturedRequest{}
		server := chatCompletionServer(t, http.StatusOK, `{"primary":true}`, request, nil)
		t.Cleanup(server.Close)

		primary := llm.NewClient("primary", llm.WithBaseURL(server.URL))
		breaker := llm.NewCircuitBreaker(primary)

		resp, err := breaker.Complete(t.Context(), llm.ChatRequest{Messages: []llm.ChatMessage{{Role: "user", Content: "hello"}}})

		require.NoError(t, err)
		require.NotNil(t, resp)
		require.Equal(t, "primary", request.JSONBody["model"])
	})
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
