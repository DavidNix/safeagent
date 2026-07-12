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

	t.Run("serialization error does not fall back", func(t *testing.T) {
		var fallbackHits atomic.Int32
		fallbackServer := chatCompletionServer(t, http.StatusOK, "fallback", nil, &fallbackHits)
		t.Cleanup(fallbackServer.Close)
		primary := llm.NewClient("primary")
		fallback := llm.NewClient("fallback", llm.WithBaseURL(fallbackServer.URL))
		breaker := llm.NewCircuitBreaker(primary, fallback)

		_, err := breaker.Complete(t.Context(), llm.ChatRequest{ToolChoice: func() {}})

		require.EqualError(t, err, "marshal chat completions request: json: unsupported type: func()")
		require.Equal(t, int32(0), fallbackHits.Load())
	})

	t.Run("fallback serialization error stops chain", func(t *testing.T) {
		primaryServer := chatCompletionServer(t, http.StatusInternalServerError, "", nil, nil)
		t.Cleanup(primaryServer.Close)
		var finalHits atomic.Int32
		finalServer := chatCompletionServer(t, http.StatusOK, "final", nil, &finalHits)
		t.Cleanup(finalServer.Close)
		primary := llm.NewClient("primary", llm.WithBaseURL(primaryServer.URL))
		invalid := llm.NewClient("invalid", llm.WithExtraFields(map[string]any{"invalid": func() {}}))
		final := llm.NewClient("final", llm.WithBaseURL(finalServer.URL))
		breaker := llm.NewCircuitBreaker(primary, invalid, final)

		_, err := breaker.Complete(t.Context(), llm.ChatRequest{})

		require.EqualError(t, err, "marshal chat completions request: json: unsupported type: func()")
		require.Equal(t, int32(0), finalHits.Load())
	})

	for _, status := range []int{http.StatusBadRequest, http.StatusPreconditionFailed, http.StatusUnprocessableEntity, http.StatusConflict} {
		t.Run(http.StatusText(status)+" falls back without opening circuit", func(t *testing.T) {
			var primaryHits atomic.Int32
			primaryServer := chatCompletionServer(t, status, "", nil, &primaryHits)
			t.Cleanup(primaryServer.Close)
			fallbackServer := chatCompletionServer(t, http.StatusOK, "fallback", nil, nil)
			t.Cleanup(fallbackServer.Close)
			primary := llm.NewClient("primary", llm.WithBaseURL(primaryServer.URL))
			fallback := llm.NewClient("fallback", llm.WithBaseURL(fallbackServer.URL))
			breaker := llm.NewCircuitBreakerWithConfig(llm.BreakerConfig{FailureThreshold: 1}, primary, fallback)

			for range 2 {
				_, err := breaker.Complete(t.Context(), llm.ChatRequest{})
				require.NoError(t, err)
			}

			require.Equal(t, int32(2), primaryHits.Load())
		})
	}

	for _, status := range []int{
		http.StatusUnauthorized,
		http.StatusForbidden,
		http.StatusNotFound,
		http.StatusRequestTimeout,
		http.StatusRequestEntityTooLarge,
		http.StatusTooManyRequests,
		http.StatusInternalServerError,
	} {
		t.Run(http.StatusText(status)+" falls back and opens circuit", func(t *testing.T) {
			var primaryHits atomic.Int32
			primaryServer := chatCompletionServer(t, status, "", nil, &primaryHits)
			t.Cleanup(primaryServer.Close)
			fallbackServer := chatCompletionServer(t, http.StatusOK, "fallback", nil, nil)
			t.Cleanup(fallbackServer.Close)
			primary := llm.NewClient("primary", llm.WithBaseURL(primaryServer.URL))
			fallback := llm.NewClient("fallback", llm.WithBaseURL(fallbackServer.URL))
			breaker := llm.NewCircuitBreakerWithConfig(llm.BreakerConfig{FailureThreshold: 1}, primary, fallback)

			for range 2 {
				_, err := breaker.Complete(t.Context(), llm.ChatRequest{})
				require.NoError(t, err)
			}

			require.Equal(t, int32(1), primaryHits.Load())
		})
	}

	t.Run("failover events identify each failed provider", func(t *testing.T) {
		primaryServer := chatCompletionServer(t, http.StatusBadRequest, "", nil, nil)
		t.Cleanup(primaryServer.Close)
		firstFallbackServer := chatCompletionServer(t, http.StatusInternalServerError, "", nil, nil)
		t.Cleanup(firstFallbackServer.Close)
		finalServer := chatCompletionServer(t, http.StatusOK, "final", nil, nil)
		t.Cleanup(finalServer.Close)
		primary := llm.NewClient("shared", llm.WithProviderID("primary"), llm.WithBaseURL(primaryServer.URL))
		firstFallback := llm.NewClient("shared", llm.WithProviderID("fallback-1"), llm.WithBaseURL(firstFallbackServer.URL))
		final := llm.NewClient("shared", llm.WithProviderID("fallback-2"), llm.WithBaseURL(finalServer.URL))
		events := make(chan llm.FailoverEvent, 2)
		breaker := llm.NewCircuitBreakerWithConfig(llm.BreakerConfig{
			FailureThreshold: 1,
			OnFailover: func(_ context.Context, event llm.FailoverEvent) {
				events <- event
			},
		}, primary, firstFallback, final)

		resp, err := breaker.Complete(t.Context(), llm.ChatRequest{})

		require.NoError(t, err)
		require.Equal(t, "final", resp.Choices[0].Message.Content)
		require.Len(t, events, 2)
		first := <-events
		require.Equal(t, "primary", first.FromProvider)
		require.Equal(t, 0, first.FromProviderIndex)
		require.Equal(t, "fallback-1", first.ToProvider)
		require.False(t, first.CountedAgainstBreaker)
		var statusErr *llm.StatusError
		require.ErrorAs(t, first.Error, &statusErr)
		require.Equal(t, http.StatusBadRequest, statusErr.StatusCode)
		second := <-events
		require.Equal(t, "fallback-1", second.FromProvider)
		require.Equal(t, 1, second.FromProviderIndex)
		require.Equal(t, "fallback-2", second.ToProvider)
		require.False(t, second.CountedAgainstBreaker)
	})

	t.Run("counted primary error emits failover only for error transition", func(t *testing.T) {
		primaryServer := chatCompletionServer(t, http.StatusInternalServerError, "", nil, nil)
		t.Cleanup(primaryServer.Close)
		fallbackServer := chatCompletionServer(t, http.StatusOK, "fallback", nil, nil)
		t.Cleanup(fallbackServer.Close)
		primary := llm.NewClient("shared", llm.WithProviderID("primary"), llm.WithBaseURL(primaryServer.URL))
		fallback := llm.NewClient("shared", llm.WithProviderID("fallback"), llm.WithBaseURL(fallbackServer.URL))
		events := make(chan llm.FailoverEvent, 2)
		breaker := llm.NewCircuitBreakerWithConfig(llm.BreakerConfig{
			FailureThreshold: 1,
			OnFailover: func(_ context.Context, event llm.FailoverEvent) {
				events <- event
			},
		}, primary, fallback)

		for range 2 {
			_, err := breaker.Complete(t.Context(), llm.ChatRequest{})
			require.NoError(t, err)
		}

		require.Len(t, events, 1)
		event := <-events
		require.True(t, event.CountedAgainstBreaker)
		require.Equal(t, "primary", event.FromProvider)
		require.Equal(t, "fallback", event.ToProvider)
	})

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

	t.Run("caller cancellation during primary does not fall back", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		primary := llm.NewClient("primary", llm.WithHTTPClient(&http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			cancel()
			return nil, context.Canceled
		})}))
		var fallbackHits atomic.Int32
		fallbackServer := chatCompletionServer(t, http.StatusOK, "fallback", nil, &fallbackHits)
		t.Cleanup(fallbackServer.Close)
		fallback := llm.NewClient("fallback", llm.WithBaseURL(fallbackServer.URL))
		breaker := llm.NewCircuitBreaker(primary, fallback)

		_, err := breaker.Complete(ctx, llm.ChatRequest{})

		require.ErrorIs(t, err, context.Canceled)
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
			breaker := llm.NewCircuitBreakerWithConfig(llm.BreakerConfig{
				FailureThreshold:         1,
				OpenDuration:             time.Minute,
				SuccessfulProbeThreshold: 1,
			}, primary, fallback)

			_, err := breaker.Complete(t.Context(), llm.ChatRequest{Messages: []llm.ChatMessage{{Role: "user", Content: "hello"}}})
			require.NoError(t, err)
			primaryBeforeProbe := primaryHits.Load()

			primaryStatus.Store(http.StatusForbidden)
			time.Sleep(time.Minute)
			_, err = breaker.Complete(t.Context(), llm.ChatRequest{Messages: []llm.ChatMessage{{Role: "user", Content: "hello"}}})
			require.NoError(t, err)
			require.Equal(t, primaryBeforeProbe+1, primaryHits.Load())

			primaryAfterProbe := primaryHits.Load()
			_, err = breaker.Complete(t.Context(), llm.ChatRequest{Messages: []llm.ChatMessage{{Role: "user", Content: "hello"}}})
			require.NoError(t, err)
			require.Equal(t, primaryAfterProbe, primaryHits.Load())
		})
	})

	t.Run("neutral probe schedules another probe without reopening", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			var primaryStatus atomic.Int32
			primaryStatus.Store(http.StatusForbidden)
			primary := llm.NewClient("primary", llm.WithHTTPClient(&http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return testHTTPResponse(int(primaryStatus.Load()), "primary"), nil
			})}))
			fallback := llm.NewClient("fallback", llm.WithHTTPClient(&http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return testHTTPResponse(http.StatusOK, "fallback"), nil
			})}))
			breaker := llm.NewCircuitBreakerWithConfig(llm.BreakerConfig{
				FailureThreshold: 1,
				OpenDuration:     time.Minute,
				ProbeInterval:    10 * time.Second,
			}, primary, fallback)

			_, err := breaker.Complete(t.Context(), llm.ChatRequest{})
			require.NoError(t, err)

			primaryStatus.Store(http.StatusBadRequest)
			time.Sleep(time.Minute)
			_, err = breaker.Complete(t.Context(), llm.ChatRequest{})
			require.NoError(t, err)

			primaryStatus.Store(http.StatusOK)
			time.Sleep(10 * time.Second)
			resp, err := breaker.Complete(t.Context(), llm.ChatRequest{})
			require.NoError(t, err)
			require.Equal(t, "primary", resp.Choices[0].Message.Content)
		})
	})

	t.Run("caller cancellation releases recovery probe", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			var mode atomic.Int32
			cancelProbe := func() {}
			primary := llm.NewClient("primary", llm.WithHTTPClient(&http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				switch mode.Load() {
				case 0:
					return testHTTPResponse(http.StatusForbidden, ""), nil
				case 1:
					cancelProbe()
					return nil, context.Canceled
				default:
					return testHTTPResponse(http.StatusOK, "primary"), nil
				}
			})}))
			fallback := llm.NewClient("fallback", llm.WithHTTPClient(&http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return testHTTPResponse(http.StatusOK, "fallback"), nil
			})}))
			breaker := llm.NewCircuitBreakerWithConfig(llm.BreakerConfig{
				FailureThreshold: 1,
				OpenDuration:     time.Minute,
			}, primary, fallback)

			_, err := breaker.Complete(t.Context(), llm.ChatRequest{})
			require.NoError(t, err)
			time.Sleep(time.Minute)

			mode.Store(1)
			probeCtx, cancel := context.WithCancel(t.Context())
			cancelProbe = cancel
			_, err = breaker.Complete(probeCtx, llm.ChatRequest{})
			require.ErrorIs(t, err, context.Canceled)

			mode.Store(2)
			resp, err := breaker.Complete(t.Context(), llm.ChatRequest{})
			require.NoError(t, err)
			require.Equal(t, "primary", resp.Choices[0].Message.Content)
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
