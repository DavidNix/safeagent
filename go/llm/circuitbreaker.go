package llm

import (
	"context"
	"errors"
	"sync"
	"time"
)

// BreakerConfig configures primary-provider circuit breaking. OnOpen runs
// synchronously outside the breaker lock and must be safe for concurrent use.
type BreakerConfig struct {
	Name                     string
	FailureThreshold         int
	OpenDuration             time.Duration
	ProbeInterval            time.Duration
	SuccessfulProbeThreshold int
	OnOpen                   func(context.Context, BreakerOpenEvent)
}

// BreakerOpenEvent describes a circuit opening after failures or a failed
// recovery probe. Error is the primary request error that opened the circuit.
type BreakerOpenEvent struct {
	Name      string
	Error     error
	OpenUntil time.Time
	WasProbe  bool
}

// CircuitBreaker routes Chat Completions requests through a primary client and
// optional fallbacks. With no fallbacks it is a pass-through wrapper. The
// breaker records every primary error while the caller context remains active.
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

// Complete sends a Chat Completions request. A primary failure is recorded
// against the breaker and immediately falls through to configured fallbacks;
// each client is attempted once.
func (b *CircuitBreaker) Complete(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	decision := b.breaker.decide()
	if !decision.routePrimary {
		return b.fallback(ctx, req, nil)
	}

	resp, err := b.primary.Complete(ctx, req)
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, ctxErr
		}
		b.breaker.recordFailure(ctx, err, decision.wasProbe)
		if len(b.fallbacks) == 0 {
			return nil, err
		}
		return b.fallback(ctx, req, []error{err})
	}

	b.breaker.recordSuccess(decision.wasProbe)
	return resp, nil
}

func (b *CircuitBreaker) fallback(ctx context.Context, req ChatRequest, errs []error) (*ChatResponse, error) {
	for _, fallback := range b.fallbacks {
		resp, err := fallback.Complete(ctx, req)
		if err != nil {
			if ctxErr := ctx.Err(); ctxErr != nil {
				return nil, ctxErr
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
	hasFallbacks              bool
	failureThreshold          int
	openDuration              time.Duration
	probeInterval             time.Duration
	successfulProbeThreshold  int
	consecutiveFailures       int
	consecutiveProbeSuccesses int
	openUntil                 time.Time
	probeInFlight             bool
	name                      string
	onOpen                    func(context.Context, BreakerOpenEvent)
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
		name:                     cfg.Name,
		onOpen:                   cfg.OnOpen,
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

func (b *circuitBreaker) recordFailure(ctx context.Context, err error, wasProbe bool) {
	b.mu.Lock()

	if !b.hasFallbacks {
		b.mu.Unlock()
		return
	}

	if wasProbe {
		b.probeInFlight = false
		b.consecutiveProbeSuccesses = 0
		b.openUntil = time.Now().Add(b.openDuration)
		event := b.openEvent(err, true)
		b.mu.Unlock()
		b.notifyOpen(ctx, event)
		return
	}

	b.consecutiveFailures++
	if b.consecutiveFailures < b.failureThreshold {
		b.mu.Unlock()
		return
	}
	b.consecutiveProbeSuccesses = 0
	b.openUntil = time.Now().Add(b.openDuration)
	event := b.openEvent(err, false)
	b.mu.Unlock()
	b.notifyOpen(ctx, event)
}

func (b *circuitBreaker) openEvent(err error, wasProbe bool) BreakerOpenEvent {
	return BreakerOpenEvent{
		Name:      b.name,
		Error:     err,
		OpenUntil: b.openUntil,
		WasProbe:  wasProbe,
	}
}

func (b *circuitBreaker) notifyOpen(ctx context.Context, event BreakerOpenEvent) {
	if b.onOpen != nil {
		b.onOpen(ctx, event)
	}
}
