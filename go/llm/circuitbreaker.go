package llm

import (
	"context"
	"errors"
	"net/http"
	"sync"
	"time"
)

// BreakerConfig configures primary-provider circuit breaking. Callbacks run
// synchronously outside the breaker lock and must be safe for concurrent use.
type BreakerConfig struct {
	FailureThreshold         int
	OpenDuration             time.Duration
	ProbeInterval            time.Duration
	SuccessfulProbeThreshold int
	OnFailover               func(context.Context, FailoverEvent)
}

// FailoverEvent describes an error-triggered transition to another provider.
type FailoverEvent struct {
	FromProvider          string
	FromProviderIndex     int
	ToProvider            string
	Error                 error
	CountedAgainstBreaker bool
}

// CircuitBreaker routes Chat Completions requests through a primary client and
// optional fallbacks. With no fallbacks it is a pass-through wrapper. The
// breaker records errors that indicate the primary provider is unavailable.
type CircuitBreaker struct {
	primary   *Client
	fallbacks []*Client
	breaker   *circuitBreaker
}

// NewCircuitBreaker builds a CircuitBreaker around primary and optional
// fallbacks. The primary client is assumed to be non-nil.
func NewCircuitBreaker(primary *Client, fallbacks ...*Client) *CircuitBreaker {
	return NewCircuitBreakerWithConfig(BreakerConfig{}, primary, fallbacks...)
}

// NewCircuitBreakerWithConfig builds a CircuitBreaker with custom breaker
// settings. It is primarily useful for tests and tuned production failover.
func NewCircuitBreakerWithConfig(cfg BreakerConfig, primary *Client, fallbacks ...*Client) *CircuitBreaker {
	return &CircuitBreaker{
		primary:   primary,
		fallbacks: append([]*Client(nil), fallbacks...),
		breaker:   newCircuitBreaker(cfg, len(fallbacks) > 0),
	}
}

// Complete sends a Chat Completions request. Provider failures immediately
// fall through to configured fallbacks; each client is attempted once.
func (b *CircuitBreaker) Complete(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	body, err := b.primary.prepareChatRequest(ctx, req)
	if err != nil {
		return nil, err
	}
	decision := b.breaker.decide()
	if !decision.routePrimary {
		return b.fallback(ctx, req, nil)
	}

	resp, err := b.primary.complete(ctx, body)
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			b.breaker.releaseProbe(decision.wasProbe)
			return nil, ctxErr
		}
		disposition := classifyProviderError(err)
		if !disposition.fallback {
			return nil, err
		}
		b.breaker.recordResult(decision.wasProbe, disposition.counted)
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, ctxErr
		}
		if len(b.fallbacks) == 0 {
			return nil, err
		}
		b.breaker.notifyFailover(ctx, b.primary.providerID, 0, b.fallbacks[0].providerID, err, disposition.counted)
		return b.fallback(ctx, req, []error{err})
	}

	b.breaker.recordSuccess(decision.wasProbe)
	return resp, nil
}

func (b *CircuitBreaker) fallback(ctx context.Context, req ChatRequest, errs []error) (*ChatResponse, error) {
	for i, fallback := range b.fallbacks {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		resp, err := fallback.Complete(ctx, req)
		if err != nil {
			if ctxErr := ctx.Err(); ctxErr != nil {
				return nil, ctxErr
			}
			disposition := classifyProviderError(err)
			if !disposition.fallback {
				return nil, err
			}
			errs = append(errs, err)
			if i+1 < len(b.fallbacks) {
				b.breaker.notifyFailover(ctx, fallback.providerID, i+1, b.fallbacks[i+1].providerID, err, false)
			}
			continue
		}
		return resp, nil
	}
	return nil, errors.Join(errs...)
}

type circuitBreaker struct {
	mu                        sync.Mutex
	hasFallbacks              bool
	failureThreshold          int
	openDuration              time.Duration
	probeInterval             time.Duration
	successfulProbeThreshold  int
	consecutiveFailures       int
	consecutiveProbeSuccesses int
	openUntil                 time.Time
	probeInFlight             bool
	onFailover                func(context.Context, FailoverEvent)
}

type breakerDecision struct {
	routePrimary bool
	wasProbe     bool
}

func newCircuitBreaker(cfg BreakerConfig, hasFallbacks bool) *circuitBreaker {
	failureThreshold := cfg.FailureThreshold
	if failureThreshold <= 0 {
		failureThreshold = defaultFailureThreshold
	}
	openDuration := cfg.OpenDuration
	if openDuration <= 0 {
		openDuration = defaultOpenDuration
	}
	probeInterval := cfg.ProbeInterval
	if probeInterval <= 0 {
		probeInterval = openDuration
	}
	successfulProbeThreshold := cfg.SuccessfulProbeThreshold
	if successfulProbeThreshold <= 0 {
		successfulProbeThreshold = defaultSuccessfulProbePass
	}
	return &circuitBreaker{
		onFailover:               cfg.OnFailover,
		hasFallbacks:             hasFallbacks,
		failureThreshold:         failureThreshold,
		openDuration:             openDuration,
		probeInterval:            probeInterval,
		successfulProbeThreshold: successfulProbeThreshold,
	}
}

func (b *circuitBreaker) decide() breakerDecision {
	b.mu.Lock()
	defer b.mu.Unlock()

	if !b.hasFallbacks || b.openUntil.IsZero() {
		return breakerDecision{routePrimary: true}
	}

	now := time.Now()
	if now.Before(b.openUntil) {
		return breakerDecision{routePrimary: false}
	}
	if b.probeInFlight {
		return breakerDecision{routePrimary: false}
	}

	b.probeInFlight = true
	return breakerDecision{routePrimary: true, wasProbe: true}
}

func (b *circuitBreaker) recordSuccess(wasProbe bool) {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.consecutiveFailures = 0
	if !wasProbe {
		return
	}

	b.probeInFlight = false
	b.consecutiveProbeSuccesses++
	if b.consecutiveProbeSuccesses >= b.successfulProbeThreshold {
		b.openUntil = time.Time{}
		b.consecutiveProbeSuccesses = 0
		return
	}
	b.openUntil = time.Now().Add(b.probeInterval)
}

func (b *circuitBreaker) recordFailure(wasProbe bool) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if !b.hasFallbacks {
		return
	}

	if wasProbe {
		b.probeInFlight = false
		b.consecutiveProbeSuccesses = 0
		b.openUntil = time.Now().Add(b.openDuration)
		return
	}

	b.consecutiveFailures++
	if b.consecutiveFailures < b.failureThreshold {
		return
	}
	b.consecutiveProbeSuccesses = 0
	b.openUntil = time.Now().Add(b.openDuration)
}

func (b *circuitBreaker) recordResult(wasProbe, counted bool) {
	if counted {
		b.recordFailure(wasProbe)
		return
	}
	b.recordNeutral(wasProbe)
}

func (b *circuitBreaker) recordNeutral(wasProbe bool) {
	if !wasProbe {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.probeInFlight = false
	b.openUntil = time.Now().Add(b.probeInterval)
}

func (b *circuitBreaker) releaseProbe(wasProbe bool) {
	if !wasProbe {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.probeInFlight = false
}

func (b *circuitBreaker) notifyFailover(ctx context.Context, fromProvider string, fromProviderIndex int, toProvider string, err error, counted bool) {
	if b.onFailover != nil {
		b.onFailover(ctx, FailoverEvent{
			FromProvider:          fromProvider,
			FromProviderIndex:     fromProviderIndex,
			ToProvider:            toProvider,
			Error:                 err,
			CountedAgainstBreaker: counted,
		})
	}
}

type errorDisposition struct {
	fallback bool
	counted  bool
}

func classifyProviderError(err error) errorDisposition {
	if _, ok := errors.AsType[*requestSerializationError](err); ok {
		return errorDisposition{}
	}
	if _, ok := errors.AsType[*ResponseTooLargeError](err); ok {
		return errorDisposition{fallback: true}
	}
	if statusErr, ok := errors.AsType[*StatusError](err); ok {
		switch statusErr.StatusCode {
		case http.StatusBadRequest, http.StatusPreconditionFailed, http.StatusUnprocessableEntity:
			return errorDisposition{fallback: true}
		case http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound,
			http.StatusRequestTimeout, http.StatusRequestEntityTooLarge, http.StatusTooManyRequests:
			return errorDisposition{fallback: true, counted: true}
		default:
			return errorDisposition{
				fallback: true,
				counted:  statusErr.StatusCode < http.StatusBadRequest || statusErr.StatusCode >= http.StatusInternalServerError,
			}
		}
	}
	return errorDisposition{fallback: true, counted: true}
}
