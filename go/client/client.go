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

// CompletionClient creates chat completions through a circuit-breaker provider chain.
type CompletionClient struct {
	primary   completionProvider
	fallbacks []completionProvider
	breaker   *circuitBreaker
}

// EmbeddingClient creates embeddings through a circuit-breaker provider chain.
type EmbeddingClient struct {
	primary   embeddingProvider
	fallbacks []embeddingProvider
	breaker   *circuitBreaker
}

func NewCompletionClient(opts Options) (*CompletionClient, error) {
	httpClient := httpClientOrDefault(opts.HTTPClient)
	primary, err := compileCompletionProvider(opts.Primary, httpClient)
	if err != nil {
		return nil, err
	}
	fallbacks, err := compileCompletionFallbacks(opts.Fallbacks, httpClient)
	if err != nil {
		return nil, err
	}

	return &CompletionClient{
		primary:   primary,
		fallbacks: fallbacks,
		breaker:   newCircuitBreaker(opts.Breaker, len(fallbacks) > 0),
	}, nil
}

func NewEmbeddingClient(opts Options) (*EmbeddingClient, error) {
	httpClient := httpClientOrDefault(opts.HTTPClient)
	primary, err := compileEmbeddingProvider(opts.Primary, httpClient)
	if err != nil {
		return nil, err
	}
	fallbacks, err := compileEmbeddingFallbacks(opts.Fallbacks, httpClient)
	if err != nil {
		return nil, err
	}

	return &EmbeddingClient{
		primary:   primary,
		fallbacks: fallbacks,
		breaker:   newCircuitBreaker(opts.Breaker, len(fallbacks) > 0),
	}, nil
}

type providerSettings struct {
	name                 string
	apiKey               string
	chatBaseURL          string
	embeddingsBaseURL    string
	chatModel            string
	embeddingModel       string
	requestOptions       []option.RequestOption
	reasoningTokenBudget *int
}

type completionProvider struct {
	name           string
	client         openai.Client
	model          string
	requestOptions []option.RequestOption
}

type embeddingProvider struct {
	name           string
	client         openai.Client
	model          string
	requestOptions []option.RequestOption
}

func compileCompletionFallbacks(providers []Provider, httpClient *http.Client) ([]completionProvider, error) {
	fallbacks := make([]completionProvider, 0, len(providers))
	for _, provider := range providers {
		runtime, err := compileCompletionProvider(provider, httpClient)
		if err != nil {
			return nil, err
		}
		fallbacks = append(fallbacks, runtime)
	}
	return fallbacks, nil
}

func compileEmbeddingFallbacks(providers []Provider, httpClient *http.Client) ([]embeddingProvider, error) {
	fallbacks := make([]embeddingProvider, 0, len(providers))
	for _, provider := range providers {
		runtime, err := compileEmbeddingProvider(provider, httpClient)
		if err != nil {
			return nil, err
		}
		fallbacks = append(fallbacks, runtime)
	}
	return fallbacks, nil
}

func compileCompletionProvider(provider Provider, httpClient *http.Client) (completionProvider, error) {
	settings, err := compileProviderSettings(provider)
	if err != nil {
		return completionProvider{}, err
	}
	if settings.chatBaseURL == "" {
		return completionProvider{}, fmt.Errorf("%s chat base url is required", settings.name)
	}

	requestOptions := append([]option.RequestOption{}, settings.requestOptions...)
	requestOptions = append(requestOptions, reasoningOptions(settings.reasoningTokenBudget)...)
	return completionProvider{
		name:           settings.name,
		client:         newOpenAIClient(settings.apiKey, settings.chatBaseURL, httpClient),
		model:          settings.chatModel,
		requestOptions: requestOptions,
	}, nil
}

func compileEmbeddingProvider(provider Provider, httpClient *http.Client) (embeddingProvider, error) {
	settings, err := compileProviderSettings(provider)
	if err != nil {
		return embeddingProvider{}, err
	}
	embeddingsBaseURL := settings.embeddingsBaseURL
	if embeddingsBaseURL == "" {
		embeddingsBaseURL = settings.chatBaseURL
	}
	if embeddingsBaseURL == "" {
		return embeddingProvider{}, fmt.Errorf("%s embeddings base url is required", settings.name)
	}

	return embeddingProvider{
		name:           settings.name,
		client:         newOpenAIClient(settings.apiKey, embeddingsBaseURL, httpClient),
		model:          settings.embeddingModel,
		requestOptions: settings.requestOptions,
	}, nil
}

func compileProviderSettings(provider Provider) (providerSettings, error) {
	switch p := provider.(type) {
	case VLLMConfig:
		return compileVLLMSettings(p)
	case *VLLMConfig:
		if p == nil {
			return providerSettings{}, errors.New("nil vllm provider")
		}
		return compileVLLMSettings(*p)
	case OpenRouterConfig:
		return compileOpenRouterSettings(p)
	case *OpenRouterConfig:
		if p == nil {
			return providerSettings{}, errors.New("nil openrouter provider")
		}
		return compileOpenRouterSettings(*p)
	default:
		return providerSettings{}, errors.New("unsupported provider")
	}
}

func compileVLLMSettings(cfg VLLMConfig) (providerSettings, error) {
	apiKey := cfg.APIKey
	if apiKey == "" {
		apiKey = defaultLocalAPIKey
	}
	return providerSettings{
		name:                 "vllm",
		apiKey:               apiKey,
		chatBaseURL:          cfg.ChatBaseURL,
		embeddingsBaseURL:    cfg.EmbeddingsBaseURL,
		chatModel:            cfg.ChatModel,
		embeddingModel:       cfg.EmbeddingModel,
		requestOptions:       requestOptionsFromMaps(cfg.Headers, cfg.QueryParams, cfg.ExtraFields),
		reasoningTokenBudget: cfg.ReasoningTokenBudget,
	}, nil
}

func compileOpenRouterSettings(cfg OpenRouterConfig) (providerSettings, error) {
	if cfg.APIKey == "" {
		return providerSettings{}, errors.New("openrouter api key is required")
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

	return providerSettings{
		name:              "openrouter",
		apiKey:            cfg.APIKey,
		chatBaseURL:       baseURL,
		embeddingsBaseURL: baseURL,
		chatModel:         cfg.ChatModel,
		embeddingModel:    cfg.EmbeddingModel,
		requestOptions:    requestOptionsFromMaps(headers, cfg.QueryParams, extraFields),
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

// New creates a chat completion through the configured provider chain.
func (c *CompletionClient) New(ctx context.Context, params openai.ChatCompletionNewParams, opts ...option.RequestOption) (*openai.ChatCompletion, error) {
	decision := c.breaker.decide()
	if decision.routePrimary {
		result, err := c.primary.client.Chat.Completions.New(ctx, chatParamsForProvider(params, c.primary), requestOptions(c.primary.requestOptions, opts)...)
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

func (c *CompletionClient) fallbackChat(ctx context.Context, params openai.ChatCompletionNewParams, errs []error, opts ...option.RequestOption) (*openai.ChatCompletion, error) {
	for _, fallback := range c.fallbacks {
		result, err := fallback.client.Chat.Completions.New(ctx, chatParamsForProvider(params, fallback), requestOptions(fallback.requestOptions, opts)...)
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

// New creates embeddings through the configured provider chain.
func (c *EmbeddingClient) New(ctx context.Context, params openai.EmbeddingNewParams, opts ...option.RequestOption) (*openai.CreateEmbeddingResponse, error) {
	decision := c.breaker.decide()
	if decision.routePrimary {
		result, err := c.primary.client.Embeddings.New(ctx, embeddingParamsForProvider(params, c.primary), requestOptions(c.primary.requestOptions, opts)...)
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

func (c *EmbeddingClient) fallbackEmbedding(ctx context.Context, params openai.EmbeddingNewParams, errs []error, opts ...option.RequestOption) (*openai.CreateEmbeddingResponse, error) {
	for _, fallback := range c.fallbacks {
		result, err := fallback.client.Embeddings.New(ctx, embeddingParamsForProvider(params, fallback), requestOptions(fallback.requestOptions, opts)...)
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

func requestOptions(providerOptions []option.RequestOption, opts []option.RequestOption) []option.RequestOption {
	requestOptions := make([]option.RequestOption, 0, len(providerOptions)+len(opts))
	requestOptions = append(requestOptions, providerOptions...)
	requestOptions = append(requestOptions, opts...)
	return requestOptions
}

func chatParamsForProvider(params openai.ChatCompletionNewParams, provider completionProvider) openai.ChatCompletionNewParams {
	if provider.model != "" {
		params.Model = provider.model
	}
	return params
}

func embeddingParamsForProvider(params openai.EmbeddingNewParams, provider embeddingProvider) openai.EmbeddingNewParams {
	if provider.model != "" {
		params.Model = provider.model
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
