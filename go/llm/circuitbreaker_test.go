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

func TestCircuitBreaker_Complete(t *testing.T) {
	t.Parallel()

	t.Run("trigger failure falls back", func(t *testing.T) {
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

	t.Run("forbidden primary falls back and opens circuit", func(t *testing.T) {
		var primaryHits atomic.Int32
		primaryServer := chatCompletionServer(t, http.StatusForbidden, "", nil, &primaryHits)
		t.Cleanup(primaryServer.Close)

		var fallbackHits atomic.Int32
		fallbackServer := chatCompletionServer(t, http.StatusOK, `{"fallback":true}`, nil, &fallbackHits)
		t.Cleanup(fallbackServer.Close)

		primary := llm.NewClient("primary", llm.WithBaseURL(primaryServer.URL))
		fallback := llm.NewClient("fallback", llm.WithBaseURL(fallbackServer.URL))
		breaker := llm.NewCircuitBreaker(primary, fallback)

		for range 4 {
			resp, err := breaker.Complete(t.Context(), llm.ChatRequest{Messages: []llm.ChatMessage{{Role: "user", Content: "hello"}}})
			require.NoError(t, err)
			require.Equal(t, `{"fallback":true}`, resp.Choices[0].Message.Content)
		}
		require.Equal(t, int32(3), primaryHits.Load())
		require.Equal(t, int32(4), fallbackHits.Load())
	})

	t.Run("opening circuit emits triggering error", func(t *testing.T) {
		primaryServer := chatCompletionServer(t, http.StatusForbidden, "", nil, nil)
		t.Cleanup(primaryServer.Close)
		fallbackServer := chatCompletionServer(t, http.StatusOK, "fallback", nil, nil)
		t.Cleanup(fallbackServer.Close)
		primary := llm.NewClient("primary", llm.WithBaseURL(primaryServer.URL))
		fallback := llm.NewClient("fallback", llm.WithBaseURL(fallbackServer.URL))
		events := make(chan llm.BreakerOpenEvent, 1)
		breaker := llm.NewCircuitBreakerWithConfig(llm.BreakerConfig{
			Name:             "chat-primary",
			FailureThreshold: 2,
			OpenDuration:     time.Minute,
			OnOpen: func(_ context.Context, event llm.BreakerOpenEvent) {
				events <- event
			},
		}, primary, fallback)

		_, err := breaker.Complete(t.Context(), llm.ChatRequest{Messages: []llm.ChatMessage{{Role: "user", Content: "hello"}}})
		require.NoError(t, err)
		require.Empty(t, events)

		beforeOpen := time.Now()
		_, err = breaker.Complete(t.Context(), llm.ChatRequest{Messages: []llm.ChatMessage{{Role: "user", Content: "hello"}}})
		afterOpen := time.Now()
		require.NoError(t, err)
		event := <-events
		require.Equal(t, "chat-primary", event.Name)
		require.False(t, event.OpenUntil.Before(beforeOpen.Add(time.Minute)))
		require.False(t, event.OpenUntil.After(afterOpen.Add(time.Minute)))
		require.False(t, event.WasProbe)
		var statusErr *llm.StatusError
		require.ErrorAs(t, event.Error, &statusErr)
		require.Equal(t, http.StatusForbidden, statusErr.StatusCode)
	})

	t.Run("provider deadline falls back while caller context is active", func(t *testing.T) {
		primary := llm.NewClient("primary", llm.WithHTTPClient(&http.Client{
			Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return nil, context.DeadlineExceeded
			}),
		}))
		fallbackServer := chatCompletionServer(t, http.StatusOK, "fallback", nil, nil)
		t.Cleanup(fallbackServer.Close)
		fallback := llm.NewClient("fallback", llm.WithBaseURL(fallbackServer.URL))
		breaker := llm.NewCircuitBreaker(primary, fallback)

		resp, err := breaker.Complete(t.Context(), llm.ChatRequest{Messages: []llm.ChatMessage{{Role: "user", Content: "hello"}}})

		require.NoError(t, err)
		require.Equal(t, "fallback", resp.Choices[0].Message.Content)
	})

	t.Run("caller cancellation does not fall back", func(t *testing.T) {
		var primaryHits atomic.Int32
		primaryServer := chatCompletionServer(t, http.StatusOK, "primary", nil, &primaryHits)
		t.Cleanup(primaryServer.Close)
		var fallbackHits atomic.Int32
		fallbackServer := chatCompletionServer(t, http.StatusOK, "fallback", nil, &fallbackHits)
		t.Cleanup(fallbackServer.Close)
		primary := llm.NewClient("primary", llm.WithBaseURL(primaryServer.URL))
		fallback := llm.NewClient("fallback", llm.WithBaseURL(fallbackServer.URL))
		breaker := llm.NewCircuitBreaker(primary, fallback)
		ctx, cancel := context.WithCancel(t.Context())
		cancel()

		_, err := breaker.Complete(ctx, llm.ChatRequest{Messages: []llm.ChatMessage{{Role: "user", Content: "hello"}}})

		require.ErrorIs(t, err, context.Canceled)
		require.Equal(t, int32(0), primaryHits.Load())
		require.Equal(t, int32(0), fallbackHits.Load())
	})

	t.Run("client timeout falls back with a fresh timeout", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			primary := llm.NewClient("primary",
				llm.WithRequestTimeout(time.Second),
				llm.WithHTTPClient(&http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
					<-req.Context().Done()
					return nil, req.Context().Err()
				})}),
			)
			fallback := llm.NewClient("fallback",
				llm.WithRequestTimeout(2*time.Second),
				llm.WithHTTPClient(&http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
					deadline, ok := req.Context().Deadline()
					require.True(t, ok)
					require.Equal(t, 2*time.Second, time.Until(deadline))
					return &http.Response{
						StatusCode: http.StatusOK,
						Header:     make(http.Header),
						Body:       io.NopCloser(strings.NewReader(chatCompletionJSON("fallback"))),
					}, nil
				})}),
			)
			breaker := llm.NewCircuitBreaker(primary, fallback)

			resp, err := breaker.Complete(t.Context(), llm.ChatRequest{Messages: []llm.ChatMessage{{Role: "user", Content: "hello"}}})

			require.NoError(t, err)
			require.Equal(t, "fallback", resp.Choices[0].Message.Content)
		})
	})

	t.Run("circuit opens after threshold and skips primary until probe succeeds", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			var primaryHits atomic.Int32
			var primaryHealthy atomic.Bool
			primary := llm.NewClient("primary", llm.WithHTTPClient(&http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				primaryHits.Add(1)
				if !primaryHealthy.Load() {
					return testHTTPResponse(http.StatusServiceUnavailable, ""), nil
				}
				return testHTTPResponse(http.StatusOK, `{"primary":true}`), nil
			})}))
			fallback := llm.NewClient("fallback", llm.WithHTTPClient(&http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return testHTTPResponse(http.StatusOK, `{"fallback":true}`), nil
			})}))
			breaker := llm.NewCircuitBreakerWithConfig(llm.BreakerConfig{
				FailureThreshold:         2,
				OpenDuration:             time.Minute,
				ProbeInterval:            10 * time.Second,
				SuccessfulProbeThreshold: 1,
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
			time.Sleep(time.Minute)
			_, err = breaker.Complete(t.Context(), llm.ChatRequest{Messages: []llm.ChatMessage{{Role: "user", Content: "hello"}}})
			require.NoError(t, err)
			require.Equal(t, primaryBefore+1, primaryHits.Load())

			_, err = breaker.Complete(t.Context(), llm.ChatRequest{Messages: []llm.ChatMessage{{Role: "user", Content: "hello"}}})
			require.NoError(t, err)
			require.Equal(t, primaryBefore+2, primaryHits.Load())
		})
	})

	t.Run("failed probe reopens circuit and skips primary again", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			var primaryHits atomic.Int32
			var primaryStatus atomic.Int32
			primaryStatus.Store(http.StatusServiceUnavailable)
			primary := llm.NewClient("primary", llm.WithHTTPClient(&http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				primaryHits.Add(1)
				return testHTTPResponse(int(primaryStatus.Load()), ""), nil
			})}))
			fallback := llm.NewClient("fallback", llm.WithHTTPClient(&http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return testHTTPResponse(http.StatusOK, `{"fallback":true}`), nil
			})}))
			events := make(chan llm.BreakerOpenEvent, 2)
			breaker := llm.NewCircuitBreakerWithConfig(llm.BreakerConfig{
				FailureThreshold:         1,
				OpenDuration:             time.Minute,
				SuccessfulProbeThreshold: 1,
				OnOpen: func(_ context.Context, event llm.BreakerOpenEvent) {
					events <- event
				},
			}, primary, fallback)

			_, err := breaker.Complete(t.Context(), llm.ChatRequest{Messages: []llm.ChatMessage{{Role: "user", Content: "hello"}}})
			require.NoError(t, err)
			require.False(t, (<-events).WasProbe)
			primaryBeforeProbe := primaryHits.Load()

			primaryStatus.Store(http.StatusBadRequest)
			time.Sleep(time.Minute)
			_, err = breaker.Complete(t.Context(), llm.ChatRequest{Messages: []llm.ChatMessage{{Role: "user", Content: "hello"}}})
			require.NoError(t, err)
			require.Equal(t, primaryBeforeProbe+1, primaryHits.Load())
			require.True(t, (<-events).WasProbe)

			primaryAfterProbe := primaryHits.Load()
			_, err = breaker.Complete(t.Context(), llm.ChatRequest{Messages: []llm.ChatMessage{{Role: "user", Content: "hello"}}})
			require.NoError(t, err)
			require.Equal(t, primaryAfterProbe, primaryHits.Load())
		})
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

func testHTTPResponse(status int, content string) *http.Response {
	body := content
	if status < http.StatusBadRequest {
		body = chatCompletionJSON(content)
	}
	return &http.Response{
		StatusCode: status,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}
