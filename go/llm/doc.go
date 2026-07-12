// Package llm provides plain-HTTP clients for OpenAI-compatible Chat
// Completions and Embeddings APIs.
//
// Client and EmbeddingClient own the wire-level request and response types and
// can be configured for any compatible endpoint. NewVLLM and NewOpenRouter,
// together with their embedding counterparts, provide convenient provider
// defaults while still accepting custom headers, query parameters, and body
// fields.
//
// CircuitBreaker and EmbeddingCircuitBreaker add ordered provider failover.
// They classify transport and provider errors, temporarily bypass an
// unavailable primary, and periodically probe it for recovery.
//
// This package does not depend on the agent runtime. Client and CircuitBreaker
// satisfy agent.Model without requiring an adapter.
package llm
