package llm

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"net/http"
)

// EmbeddingConfig identifies an embedding model.
type EmbeddingConfig struct {
	Model string
}

// EmbeddingClient calls an OpenAI-compatible Embeddings API.
type EmbeddingClient struct {
	client *Client
}

// VLLMEmbeddingConfig configures a vLLM OpenAI-compatible Embeddings client.
type VLLMEmbeddingConfig struct {
	ProviderID  string
	APIKey      string
	BaseURL     string
	Model       string
	Headers     map[string]string
	QueryParams map[string]string
	ExtraFields map[string]any
}

// NewVLLMEmbedding builds a vLLM Embeddings client. If APIKey is empty, the
// client uses "local" because local vLLM deployments commonly ignore auth.
func NewVLLMEmbedding(cfg VLLMEmbeddingConfig) *EmbeddingClient {
	apiKey := cmp.Or(cfg.APIKey, defaultLocalAPIKey)
	return NewEmbeddingClient(EmbeddingConfig{Model: cfg.Model},
		WithProviderID(cfg.ProviderID),
		WithAPIKey(apiKey),
		WithBaseURL(cfg.BaseURL),
		WithHeaders(cfg.Headers),
		WithQueryParams(cfg.QueryParams),
		WithExtraFields(cfg.ExtraFields),
	)
}

// OpenRouterEmbeddingConfig configures OpenRouter's Embeddings API.
type OpenRouterEmbeddingConfig struct {
	ProviderID               string
	APIKey                   string
	BaseURL                  string
	Model                    string
	SiteURL                  string
	AppTitle                 string
	RequireZeroDataRetention bool
	Headers                  map[string]string
	QueryParams              map[string]string
	ExtraFields              map[string]any
}

// NewOpenRouterEmbedding builds an OpenRouter Embeddings client. If BaseURL
// is empty, the OpenRouter API base URL is used.
func NewOpenRouterEmbedding(cfg OpenRouterEmbeddingConfig) *EmbeddingClient {
	baseURL := cmp.Or(cfg.BaseURL, defaultOpenRouterBaseURL)
	headers := map[string]string{}
	if cfg.SiteURL != "" {
		headers["HTTP-Referer"] = cfg.SiteURL
	}
	if cfg.AppTitle != "" {
		headers["X-Title"] = cfg.AppTitle
	}
	maps.Copy(headers, cfg.Headers)

	extraFields := maps.Clone(cfg.ExtraFields)
	if extraFields == nil {
		extraFields = map[string]any{}
	}
	if cfg.RequireZeroDataRetention {
		extraFields["provider.zdr"] = true
	}

	return NewEmbeddingClient(EmbeddingConfig{Model: cfg.Model},
		WithProviderID(cfg.ProviderID),
		WithAPIKey(cfg.APIKey),
		WithBaseURL(baseURL),
		WithHeaders(headers),
		WithQueryParams(cfg.QueryParams),
		WithExtraFields(extraFields),
	)
}

// NewEmbeddingClient builds an Embeddings client.
func NewEmbeddingClient(cfg EmbeddingConfig, opts ...Option) *EmbeddingClient {
	return &EmbeddingClient{
		client: NewClient(cfg.Model, opts...),
	}
}

// EmbeddingRequest is an OpenAI-compatible text embeddings request without
// the model field. EmbeddingClient injects its configured model.
type EmbeddingRequest struct {
	Input      []string `json:"input"`
	Dimensions *int     `json:"dimensions,omitempty"`
	User       string   `json:"user,omitempty"`
}

// Embedding is one vector returned by the Embeddings API.
type Embedding struct {
	Object    string    `json:"object"`
	Index     int       `json:"index"`
	Embedding []float32 `json:"embedding"`
}

// EmbeddingUsage is token usage returned by the Embeddings API.
type EmbeddingUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// EmbeddingResponse is an OpenAI-compatible Embeddings response.
type EmbeddingResponse struct {
	Object string         `json:"object"`
	Data   []Embedding    `json:"data"`
	Model  string         `json:"model"`
	Usage  EmbeddingUsage `json:"usage"`
}

type wireEmbeddingRequest struct {
	Model          string `json:"model"`
	EncodingFormat string `json:"encoding_format"`
	EmbeddingRequest
}

// Embed creates float-encoded embeddings with one provider request.
func (c *EmbeddingClient) Embed(ctx context.Context, req EmbeddingRequest) (*EmbeddingResponse, error) {
	body, err := c.prepareEmbeddingRequest(ctx, req)
	if err != nil {
		return nil, err
	}
	return c.embed(ctx, body, req)
}

func (c *EmbeddingClient) prepareEmbeddingRequest(ctx context.Context, req EmbeddingRequest) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	body, err := c.marshalRequest(req)
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, ctxErr
		}
		return nil, &requestSerializationError{operation: "embeddings", err: err}
	}
	return body, nil
}

func (c *EmbeddingClient) embed(ctx context.Context, body []byte, req EmbeddingRequest) (*EmbeddingResponse, error) {
	ctx, cancel := c.client.requestContext(ctx)
	defer cancel()

	resp, err := c.roundTrip(ctx, body)
	if err != nil {
		return nil, err
	}
	if err := validateEmbeddingResponse(*resp, len(req.Input), req.Dimensions); err != nil {
		return nil, fmt.Errorf("validate embeddings response: %w", err)
	}
	return resp, nil
}

func (c *EmbeddingClient) marshalRequest(req EmbeddingRequest) ([]byte, error) {
	body, err := json.Marshal(wireEmbeddingRequest{
		Model:            c.client.model,
		EncodingFormat:   "float",
		EmbeddingRequest: req,
	})
	if err != nil {
		return nil, err
	}
	if len(c.client.extraFields) == 0 {
		return body, nil
	}
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, err
	}
	applyExtraFields(payload, c.client.extraFields)
	return json.Marshal(payload)
}

func (c *EmbeddingClient) roundTrip(ctx context.Context, body []byte) (*EmbeddingResponse, error) {
	endpoint, err := c.client.endpoint("embeddings")
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build embeddings request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+c.client.apiKey)
	for key, value := range c.client.headers {
		httpReq.Header.Set(key, value)
	}

	resp, err := c.client.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("call embeddings: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := readResponseBody(resp.Body, c.client.maxRespBody)
	if resp.StatusCode != http.StatusOK {
		var tooLarge *ResponseTooLargeError
		if err != nil && !errors.As(err, &tooLarge) {
			return nil, fmt.Errorf("read embeddings response: %w", err)
		}
		return nil, &StatusError{
			StatusCode: resp.StatusCode,
			Body:       string(respBody),
			operation:  "embeddings",
		}
	}
	if err != nil {
		if tooLarge, ok := errors.AsType[*ResponseTooLargeError](err); ok {
			tooLarge.operation = "embeddings"
		}
		return nil, fmt.Errorf("read embeddings response: %w", err)
	}

	var parsed EmbeddingResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return nil, fmt.Errorf("decode embeddings response: %w", err)
	}
	return &parsed, nil
}

func validateEmbeddingResponse(resp EmbeddingResponse, inputCount int, requestedDimensions *int) error {
	if len(resp.Data) != inputCount {
		return fmt.Errorf("received %d embeddings for %d inputs", len(resp.Data), inputCount)
	}
	seen := make(map[int]struct{}, len(resp.Data))
	dimensions := 0
	for _, item := range resp.Data {
		if item.Index < 0 || item.Index >= inputCount {
			return fmt.Errorf("embedding index %d is outside input range", item.Index)
		}
		if _, exists := seen[item.Index]; exists {
			return fmt.Errorf("embedding index %d is duplicated", item.Index)
		}
		seen[item.Index] = struct{}{}
		if len(item.Embedding) == 0 {
			return fmt.Errorf("embedding index %d has an empty vector", item.Index)
		}
		if dimensions == 0 {
			dimensions = len(item.Embedding)
		}
		if len(item.Embedding) != dimensions {
			return fmt.Errorf("embedding index %d has %d dimensions; expected %d", item.Index, len(item.Embedding), dimensions)
		}
	}
	if requestedDimensions != nil && dimensions != *requestedDimensions {
		return fmt.Errorf("received %d dimensions; requested %d", dimensions, *requestedDimensions)
	}
	return nil
}
