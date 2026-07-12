package llm

import (
	"context"
	"errors"
)

// EmbeddingCircuitBreaker routes embedding requests through a primary client
// and optional fallbacks.
type EmbeddingCircuitBreaker struct {
	primary   *EmbeddingClient
	fallbacks []*EmbeddingClient
	breaker   *circuitBreaker
}

// NewEmbeddingCircuitBreaker builds an embedding breaker around primary and
// optional fallbacks.
func NewEmbeddingCircuitBreaker(primary *EmbeddingClient, fallbacks ...*EmbeddingClient) *EmbeddingCircuitBreaker {
	return NewEmbeddingCircuitBreakerWithConfig(BreakerConfig{}, primary, fallbacks...)
}

// NewEmbeddingCircuitBreakerWithConfig builds an embedding breaker with custom
// breaker settings.
func NewEmbeddingCircuitBreakerWithConfig(cfg BreakerConfig, primary *EmbeddingClient, fallbacks ...*EmbeddingClient) *EmbeddingCircuitBreaker {
	return &EmbeddingCircuitBreaker{
		primary:   primary,
		fallbacks: append([]*EmbeddingClient(nil), fallbacks...),
		breaker:   newCircuitBreaker(cfg, len(fallbacks) > 0),
	}
}

// Embed creates embeddings using the primary client and optional fallbacks.
func (b *EmbeddingCircuitBreaker) Embed(ctx context.Context, req EmbeddingRequest) (*EmbeddingResponse, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	decision := b.breaker.decide()
	if !decision.routePrimary {
		return b.fallback(ctx, req, nil)
	}

	resp, err := b.primary.Embed(ctx, req)
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

func (b *EmbeddingCircuitBreaker) fallback(ctx context.Context, req EmbeddingRequest, errs []error) (*EmbeddingResponse, error) {
	for _, fallback := range b.fallbacks {
		resp, err := fallback.Embed(ctx, req)
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
