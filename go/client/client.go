package client

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"maps"
	"net"
	"net/http"
	"sort"
	"sync"
	"syscall"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

const (
	defaultHTTPTimeout         = 60 * time.Second
	defaultDialTimeout         = 5 * time.Second
	defaultMaxRetries          = 0
	defaultFailureThreshold    = 3
	defaultOpenDuration        = 60 * time.Second
	defaultSuccessfulProbePass = 1
	defaultLocalAPIKey         = "local"
	defaultOpenRouterBaseURL   = "https://openrouter.ai/api/v1"
)

// Provider is a concrete OpenAI-compatible provider configuration.
type Provider interface {
	providerKind() string
}

// VLLMConfig configures a vLLM OpenAI-compatible provider.
type VLLMConfig struct {
	APIKey               string
	ChatBaseURL          string
	EmbeddingsBaseURL    string
	ChatModel            string
	EmbeddingModel       string
	ReasoningTokenBudget *int
	Headers              map[string]string
	QueryParams          map[string]string
	ExtraFields          map[string]any
}

func (VLLMConfig) providerKind() string { return "vllm" }

// OpenRouterConfig configures OpenRouter's OpenAI-compatible API.
type OpenRouterConfig struct {
	APIKey                   string
	BaseURL                  string
	ChatModel                string
	EmbeddingModel           string
	SiteURL                  string
	AppTitle                 string
	RequireZeroDataRetention bool
	Headers                  map[string]string
	QueryParams              map[string]string
	ExtraFields              map[string]any
}

func (OpenRouterConfig) providerKind() string { return "openrouter" }

// BreakerConfig configures primary-provider circuit breaking.
type BreakerConfig struct {
	FailureThreshold         int
	OpenDuration             time.Duration
	ProbeInterval            time.Duration
	SuccessfulProbeThreshold int
	Now                      func() time.Time
}

// Options configures a circuit-breaker OpenAI-compatible client.
type Options struct {
	Primary    Provider
	Fallbacks  []Provider
	Breaker    BreakerConfig
	HTTPClient *http.Client
}

// CBClient mirrors the OpenAI client service shape with circuit-breaker routing.
type CBClient struct {
	Chat       ChatService
	Embeddings EmbeddingService

	primary   providerRuntime
	fallbacks []providerRuntime
	breaker   *circuitBreaker
	initErr   error
}

// ChatService mirrors openai.Client.Chat.
type ChatService struct {
	Completions ChatCompletionService
}

// ChatCompletionService mirrors openai.Client.Chat.Completions.
type ChatCompletionService struct {
	client *CBClient
}

// EmbeddingService mirrors openai.Client.Embeddings.
type EmbeddingService struct {
	client *CBClient
}

func NewClient(opts Options) *CBClient {
	httpClient := httpClientOrDefault(opts.HTTPClient)
	primary, err := compileProvider(opts.Primary, httpClient)
	var fallbacks []providerRuntime
	if err == nil {
		fallbacks, err = compileFallbacks(opts.Fallbacks, httpClient)
	}

	cb := &CBClient{
		primary:   primary,
		fallbacks: fallbacks,
		breaker:   newCircuitBreaker(opts.Breaker, len(fallbacks) > 0),
		initErr:   err,
	}
	cb.Chat = ChatService{Completions: ChatCompletionService{client: cb}}
	cb.Embeddings = EmbeddingService{client: cb}
	return cb
}

// New creates a chat completion through the configured provider chain.
func (s ChatCompletionService) New(ctx context.Context, params openai.ChatCompletionNewParams, opts ...option.RequestOption) (*openai.ChatCompletion, error) {
	if s.client == nil {
		return nil, errors.New("client not initialized")
	}
	return s.client.newChatCompletion(ctx, params, opts...)
}

// New creates embeddings through the configured provider chain.
func (s EmbeddingService) New(ctx context.Context, params openai.EmbeddingNewParams, opts ...option.RequestOption) (*openai.CreateEmbeddingResponse, error) {
	if s.client == nil {
		return nil, errors.New("client not initialized")
	}
	return s.client.newEmbedding(ctx, params, opts...)
}

type providerRuntime struct {
	name           string
	chat           openai.Client
	embeddings     openai.Client
	chatModel      string
	embeddingModel string
	requestOptions []option.RequestOption
}

func compileFallbacks(providers []Provider, httpClient *http.Client) ([]providerRuntime, error) {
	fallbacks := make([]providerRuntime, 0, len(providers))
	for _, provider := range providers {
		runtime, err := compileProvider(provider, httpClient)
		if err != nil {
			return nil, err
		}
		fallbacks = append(fallbacks, runtime)
	}
	return fallbacks, nil
}

func compileProvider(provider Provider, httpClient *http.Client) (providerRuntime, error) {
	switch p := provider.(type) {
	case VLLMConfig:
		return compileVLLM(p, httpClient)
	case *VLLMConfig:
		if p == nil {
			return providerRuntime{}, errors.New("nil vllm provider")
		}
		return compileVLLM(*p, httpClient)
	case OpenRouterConfig:
		return compileOpenRouter(p, httpClient)
	case *OpenRouterConfig:
		if p == nil {
			return providerRuntime{}, errors.New("nil openrouter provider")
		}
		return compileOpenRouter(*p, httpClient)
	default:
		return providerRuntime{}, errors.New("unsupported provider")
	}
}

func compileVLLM(cfg VLLMConfig, httpClient *http.Client) (providerRuntime, error) {
	if cfg.ChatBaseURL == "" {
		return providerRuntime{}, errors.New("vllm chat base url is required")
	}
	embeddingsBaseURL := cfg.EmbeddingsBaseURL
	if embeddingsBaseURL == "" {
		embeddingsBaseURL = cfg.ChatBaseURL
	}

	apiKey := cfg.APIKey
	if apiKey == "" {
		apiKey = defaultLocalAPIKey
	}
	requestOptions := requestOptionsFromMaps(cfg.Headers, cfg.QueryParams, cfg.ExtraFields)
	requestOptions = append(requestOptions, reasoningOptions(cfg.ReasoningTokenBudget)...)

	return providerRuntime{
		name:           "vllm",
		chat:           newOpenAIClient(apiKey, cfg.ChatBaseURL, httpClient),
		embeddings:     newOpenAIClient(apiKey, embeddingsBaseURL, httpClient),
		chatModel:      cfg.ChatModel,
		embeddingModel: cfg.EmbeddingModel,
		requestOptions: requestOptions,
	}, nil
}

func compileOpenRouter(cfg OpenRouterConfig, httpClient *http.Client) (providerRuntime, error) {
	if cfg.APIKey == "" {
		return providerRuntime{}, errors.New("openrouter api key is required")
	}
	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = defaultOpenRouterBaseURL
	}
	headers := map[string]string{}
	if cfg.SiteURL != "" {
		headers["HTTP-Referer"] = cfg.SiteURL
	}
	if cfg.AppTitle != "" {
		headers["X-Title"] = cfg.AppTitle
	}
	maps.Copy(headers, cfg.Headers)

	extraFields := map[string]any{}
	maps.Copy(extraFields, cfg.ExtraFields)
	if cfg.RequireZeroDataRetention {
		extraFields["provider.zdr"] = true
	}

	requestOptions := requestOptionsFromMaps(headers, cfg.QueryParams, extraFields)
	apiClient := newOpenAIClient(cfg.APIKey, baseURL, httpClient)
	return providerRuntime{
		name:           "openrouter",
		chat:           apiClient,
		embeddings:     apiClient,
		chatModel:      cfg.ChatModel,
		embeddingModel: cfg.EmbeddingModel,
		requestOptions: requestOptions,
	}, nil
}

func newOpenAIClient(apiKey string, baseURL string, httpClient *http.Client) openai.Client {
	return openai.NewClient(
		option.WithAPIKey(apiKey),
		option.WithBaseURL(baseURL),
		option.WithHTTPClient(httpClient),
		option.WithMaxRetries(defaultMaxRetries),
	)
}

func requestOptionsFromMaps(headers map[string]string, queryParams map[string]string, extraFields map[string]any) []option.RequestOption {
	requestOptions := []option.RequestOption{}

	for _, key := range sortedStringKeys(headers) {
		requestOptions = append(requestOptions, option.WithHeader(key, headers[key]))
	}
	for _, key := range sortedStringKeys(queryParams) {
		requestOptions = append(requestOptions, option.WithQuery(key, queryParams[key]))
	}
	for _, key := range sortedAnyKeys(extraFields) {
		requestOptions = append(requestOptions, option.WithJSONSet(key, extraFields[key]))
	}

	return requestOptions
}

func reasoningOptions(budget *int) []option.RequestOption {
	if budget == nil || *budget < 0 {
		return nil
	}

	requestOptions := []option.RequestOption{option.WithJSONSet("thinking_token_budget", *budget)}
	switch *budget {
	case 0:
		requestOptions = append(requestOptions,
			option.WithJSONSet("chat_template_kwargs.enable_thinking", false),
			option.WithJSONSet("reasoning.effort", "none"),
		)
	default:
		requestOptions = append(requestOptions, option.WithJSONSet("reasoning.max_tokens", *budget))
	}
	return requestOptions
}

func sortedStringKeys(values map[string]string) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func sortedAnyKeys(values map[string]any) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func httpClientOrDefault(httpClient *http.Client) *http.Client {
	if httpClient != nil {
		return httpClient
	}
	return newDefaultHTTPClient()
}

func newDefaultHTTPClient() *http.Client {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.DialContext = (&net.Dialer{Timeout: defaultDialTimeout}).DialContext
	return &http.Client{
		Transport: transport,
		Timeout:   defaultHTTPTimeout,
	}
}

func (c *CBClient) newChatCompletion(ctx context.Context, params openai.ChatCompletionNewParams, opts ...option.RequestOption) (*openai.ChatCompletion, error) {
	if c.initErr != nil {
		return nil, c.initErr
	}

	decision := c.breaker.decide()
	if decision.routePrimary {
		result, err := c.primary.chat.Chat.Completions.New(ctx, chatParamsForProvider(params, c.primary), requestOptions(c.primary, opts)...)
		if err != nil {
			if !isTriggerError(ctx, err) {
				return nil, err
			}
			c.breaker.recordFailure(decision.wasProbe)
			if len(c.fallbacks) == 0 {
				return nil, err
			}
			slog.Warn("Primary chat provider failed, falling back", "provider", c.primary.name, "error", err)
			return c.fallbackChat(ctx, params, []error{fmt.Errorf("%s: %w", c.primary.name, err)}, opts...)
		}

		c.breaker.recordSuccess(decision.wasProbe)
		return result, nil
	}

	return c.fallbackChat(ctx, params, nil, opts...)
}

func (c *CBClient) fallbackChat(ctx context.Context, params openai.ChatCompletionNewParams, errs []error, opts ...option.RequestOption) (*openai.ChatCompletion, error) {
	for _, fallback := range c.fallbacks {
		result, err := fallback.chat.Chat.Completions.New(ctx, chatParamsForProvider(params, fallback), requestOptions(fallback, opts)...)
		if err != nil {
			if !isTriggerError(ctx, err) {
				return nil, err
			}
			slog.Warn("Fallback chat provider failed", "provider", fallback.name, "error", err)
			errs = append(errs, fmt.Errorf("%s: %w", fallback.name, err))
			continue
		}
		return result, nil
	}
	return nil, errors.Join(errs...)
}

func (c *CBClient) newEmbedding(ctx context.Context, params openai.EmbeddingNewParams, opts ...option.RequestOption) (*openai.CreateEmbeddingResponse, error) {
	if c.initErr != nil {
		return nil, c.initErr
	}

	decision := c.breaker.decide()
	if decision.routePrimary {
		result, err := c.primary.embeddings.Embeddings.New(ctx, embeddingParamsForProvider(params, c.primary), requestOptions(c.primary, opts)...)
		if err != nil {
			if !isTriggerError(ctx, err) {
				return nil, err
			}
			c.breaker.recordFailure(decision.wasProbe)
			if len(c.fallbacks) == 0 {
				return nil, err
			}
			slog.Warn("Primary embeddings provider failed, falling back", "provider", c.primary.name, "error", err)
			return c.fallbackEmbedding(ctx, params, []error{fmt.Errorf("%s: %w", c.primary.name, err)}, opts...)
		}

		c.breaker.recordSuccess(decision.wasProbe)
		return result, nil
	}

	return c.fallbackEmbedding(ctx, params, nil, opts...)
}

func (c *CBClient) fallbackEmbedding(ctx context.Context, params openai.EmbeddingNewParams, errs []error, opts ...option.RequestOption) (*openai.CreateEmbeddingResponse, error) {
	for _, fallback := range c.fallbacks {
		result, err := fallback.embeddings.Embeddings.New(ctx, embeddingParamsForProvider(params, fallback), requestOptions(fallback, opts)...)
		if err != nil {
			if !isTriggerError(ctx, err) {
				return nil, err
			}
			slog.Warn("Fallback embeddings provider failed", "provider", fallback.name, "error", err)
			errs = append(errs, fmt.Errorf("%s: %w", fallback.name, err))
			continue
		}
		return result, nil
	}
	return nil, errors.Join(errs...)
}

func requestOptions(provider providerRuntime, opts []option.RequestOption) []option.RequestOption {
	requestOptions := make([]option.RequestOption, 0, len(provider.requestOptions)+len(opts))
	requestOptions = append(requestOptions, provider.requestOptions...)
	requestOptions = append(requestOptions, opts...)
	return requestOptions
}

func chatParamsForProvider(params openai.ChatCompletionNewParams, provider providerRuntime) openai.ChatCompletionNewParams {
	if provider.chatModel != "" {
		params.Model = provider.chatModel
	}
	return params
}

func embeddingParamsForProvider(params openai.EmbeddingNewParams, provider providerRuntime) openai.EmbeddingNewParams {
	if provider.embeddingModel != "" {
		params.Model = provider.embeddingModel
	}
	return params
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

func isTriggerError(ctx context.Context, err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, context.Canceled) {
		return false
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return ctx.Err() == nil
	}

	if apiErr, ok := errors.AsType[*openai.Error](err); ok {
		switch {
		case apiErr.StatusCode >= http.StatusInternalServerError:
			return true
		case apiErr.StatusCode == http.StatusTooManyRequests:
			return true
		default:
			return false
		}
	}

	if _, ok := errors.AsType[net.Error](err); ok {
		return true
	}

	if errors.Is(err, syscall.ECONNREFUSED) || errors.Is(err, io.EOF) {
		return true
	}

	return false
}
