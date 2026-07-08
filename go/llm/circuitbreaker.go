package llm

import (
	"context"
	"errors"
	"sync"
	"time"
)

// BreakerConfig configures primary-provider circuit breaking.
type BreakerConfig struct {
	FailureThreshold         int
	OpenDuration             time.Duration
	ProbeInterval            time.Duration
	SuccessfulProbeThreshold int
	Now                      func() time.Time
}

// CircuitBreaker routes Chat Completions requests through a primary client and
// optional fallbacks. With no fallbacks it is a pass-through wrapper. The
// breaker records retryable provider and transport failures using the same
// rules as Client retries.
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

// Complete sends a Chat Completions request. A retryable primary failure is
// recorded against the breaker and immediately falls through to configured
// fallbacks; Client retries are zero by default.
func (b *CircuitBreaker) Complete(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	decision := b.breaker.decide()
	if decision.routePrimary {
		resp, err := b.primary.Complete(ctx, req)
		if err != nil {
			if !isTriggerError(ctx, err) {
				return nil, err
			}
			b.breaker.recordFailure(decision.wasProbe)
			if len(b.fallbacks) == 0 {
				return nil, err
			}
			return b.fallback(ctx, req, []error{err})
		}

		b.breaker.recordSuccess(decision.wasProbe)
		return resp, nil
	}

	return b.fallback(ctx, req, nil)
}

func (b *CircuitBreaker) fallback(ctx context.Context, req ChatRequest, errs []error) (*ChatResponse, error) {
	for _, fallback := range b.fallbacks {
		resp, err := fallback.Complete(ctx, req)
		if err != nil {
			if !isTriggerError(ctx, err) {
				return nil, err
			}
			errs = append(errs, err)
			continue
		}
		return resp, nil
	}
	return nil, errors.Join(errs...)
}

type circuitBreaker struct {
	mu                        sync.Mutex
	now                       func() time.Time
	hasFallbacks              bool
	failureThreshold          int
	openDuration              time.Duration
	probeInterval             time.Duration
	successfulProbeThreshold  int
	consecutiveFailures       int
	consecutiveProbeSuccesses int
	openUntil                 time.Time
	probeInFlight             bool
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
	now := cfg.Now
	if now == nil {
		now = time.Now
	}

	return &circuitBreaker{
		now:                      now,
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

	now := b.now()
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
	b.openUntil = b.now().Add(b.probeInterval)
}

func (b *circuitBreaker) recordFailure(wasProbe bool) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if wasProbe {
		b.probeInFlight = false
		b.consecutiveProbeSuccesses = 0
		b.openUntil = b.now().Add(b.openDuration)
		return
	}

	b.consecutiveFailures++
	if b.consecutiveFailures >= b.failureThreshold {
		b.consecutiveProbeSuccesses = 0
		b.openUntil = b.now().Add(b.openDuration)
	}
}
