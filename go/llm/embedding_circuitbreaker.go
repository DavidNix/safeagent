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
	body, err := b.primary.prepareEmbeddingRequest(ctx, req)
	if err != nil {
		return nil, err
	}
	decision := b.breaker.decide()
	if !decision.routePrimary {
		return b.fallback(ctx, req, nil)
	}

	resp, err := b.primary.embed(ctx, body, req)
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
		b.breaker.notifyFailover(ctx, b.primary.client.providerID, 0, b.fallbacks[0].client.providerID, err, disposition.counted)
		return b.fallback(ctx, req, []error{err})
	}

	b.breaker.recordSuccess(decision.wasProbe)
	return resp, nil
}

func (b *EmbeddingCircuitBreaker) fallback(ctx context.Context, req EmbeddingRequest, errs []error) (*EmbeddingResponse, error) {
	for i, fallback := range b.fallbacks {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		resp, err := fallback.Embed(ctx, req)
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
				b.breaker.notifyFailover(ctx, fallback.client.providerID, i+1, b.fallbacks[i+1].client.providerID, err, false)
			}
			continue
		}
		return resp, nil
	}
	return nil, errors.Join(errs...)
}
