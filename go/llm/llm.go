// Package llm provides plain-HTTP clients for OpenAI-compatible Chat
// Completions APIs. It owns the wire-level request and response types and does
// not depend on the agent runtime.
package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"math/rand/v2"
	"net"
	"net/http"
	"strconv"
	"strings"
	"syscall"
	"time"
)

const (
	// DefaultBaseURL is the OpenRouter OpenAI-compatible API base URL.
	DefaultBaseURL = "https://openrouter.ai/api/v1"
	// DefaultMaxRetries is the default number of retries after a retryable
	// failure. The default is zero so a CircuitBreaker can fail over after one
	// trigger failure instead of waiting on local retries.
	DefaultMaxRetries = 0
	// DefaultMaxResponseBytes caps response bodies at 64 MiB. This is generous
	// for current OpenAI-compatible max completion sizes while still preventing
	// accidental unbounded reads. Use WithMaxResponseBytes(0) to disable it.
	DefaultMaxResponseBytes = 64 << 20

	defaultHTTPTimeout         = 60 * time.Second
	defaultDialTimeout         = 5 * time.Second
	defaultRetryDelay          = 500 * time.Millisecond
	defaultFailureThreshold    = 3
	defaultOpenDuration        = 60 * time.Second
	defaultSuccessfulProbePass = 1
	defaultLocalAPIKey         = "local"
	defaultOpenRouterBaseURL   = DefaultBaseURL
)

// Client calls an OpenAI-compatible Chat Completions API.
type Client struct {
	model       string
	apiKey      string
	baseURL     string
	httpClient  *http.Client
	headers     map[string]string
	queryParams map[string]string
	extraFields map[string]any
	maxRetries  int
	maxRespBody int64
	retryDelay  time.Duration
}

// Option customizes a Client.
type Option func(*Client)

// WithAPIKey overrides the bearer token used for API requests.
func WithAPIKey(key string) Option {
	return func(c *Client) { c.apiKey = key }
}

// WithBaseURL overrides the API base URL, for example to point at vLLM or
// another OpenAI-compatible server.
func WithBaseURL(baseURL string) Option {
	return func(c *Client) { c.baseURL = strings.TrimSuffix(baseURL, "/") }
}

// WithHTTPClient overrides the HTTP client used to execute requests.
func WithHTTPClient(httpClient *http.Client) Option {
	return func(c *Client) { c.httpClient = httpClient }
}

// WithHeaders adds static headers to every request made by the client.
func WithHeaders(headers map[string]string) Option {
	return func(c *Client) { c.headers = maps.Clone(headers) }
}

// WithQueryParams adds static query parameters to every request made by the
// client.
func WithQueryParams(queryParams map[string]string) Option {
	return func(c *Client) { c.queryParams = maps.Clone(queryParams) }
}

// WithExtraFields adds static JSON body fields to every request made by the
// client. Dotted keys create nested objects, for example "provider.zdr".
func WithExtraFields(extraFields map[string]any) Option {
	return func(c *Client) { c.extraFields = maps.Clone(extraFields) }
}

// WithMaxRetries overrides how many times the client retries retryable
// failures. The default is zero; use a CircuitBreaker for immediate failover.
func WithMaxRetries(n int) Option {
	return func(c *Client) { c.maxRetries = n }
}

// WithMaxResponseBytes overrides the maximum response body size read by the
// client. Zero or a negative value disables the limit.
func WithMaxResponseBytes(n int64) Option {
	return func(c *Client) { c.maxRespBody = n }
}

// WithRetryDelay overrides the base delay of the exponential backoff between
// retries. A Retry-After response header takes precedence.
func WithRetryDelay(d time.Duration) Option {
	return func(c *Client) { c.retryDelay = d }
}

// NewClient builds a Chat Completions client for model. By default it uses
// DefaultBaseURL and zero retries.
func NewClient(model string, opts ...Option) *Client {
	c := &Client{
		model:       model,
		baseURL:     DefaultBaseURL,
		httpClient:  newDefaultHTTPClient(),
		headers:     map[string]string{},
		queryParams: map[string]string{},
		extraFields: map[string]any{},
		maxRetries:  DefaultMaxRetries,
		maxRespBody: DefaultMaxResponseBytes,
		retryDelay:  defaultRetryDelay,
	}
	for _, opt := range opts {
		opt(c)
	}
	if c.httpClient == nil {
		c.httpClient = newDefaultHTTPClient()
	}
	if c.headers == nil {
		c.headers = map[string]string{}
	}
	if c.queryParams == nil {
		c.queryParams = map[string]string{}
	}
	if c.extraFields == nil {
		c.extraFields = map[string]any{}
	}
	return c
}

// VLLMConfig configures a vLLM OpenAI-compatible Chat Completions client.
type VLLMConfig struct {
	APIKey      string
	ChatBaseURL string
	ChatModel   string
	// ReasoningTokenBudget emits vLLM-specific JSON fields for reasoning
	// models served with thinking/reasoning chat templates. Nil or negative
	// leaves reasoning configuration unchanged; zero disables thinking.
	ReasoningTokenBudget *int
	Headers              map[string]string
	QueryParams          map[string]string
	ExtraFields          map[string]any
}

// NewVLLM builds a vLLM Chat Completions client. If APIKey is empty, the
// client uses "local" because local vLLM deployments commonly ignore auth.
func NewVLLM(cfg VLLMConfig) *Client {
	apiKey := cfg.APIKey
	if apiKey == "" {
		apiKey = defaultLocalAPIKey
	}
	extraFields := maps.Clone(cfg.ExtraFields)
	if extraFields == nil {
		extraFields = map[string]any{}
	}
	maps.Copy(extraFields, vllmReasoningFields(cfg.ReasoningTokenBudget))

	return NewClient(cfg.ChatModel,
		WithAPIKey(apiKey),
		WithBaseURL(cfg.ChatBaseURL),
		WithHeaders(cfg.Headers),
		WithQueryParams(cfg.QueryParams),
		WithExtraFields(extraFields),
	)
}

// OpenRouterConfig configures OpenRouter's OpenAI-compatible Chat Completions
// API.
type OpenRouterConfig struct {
	APIKey                   string
	BaseURL                  string
	ChatModel                string
	SiteURL                  string
	AppTitle                 string
	RequireZeroDataRetention bool
	Headers                  map[string]string
	QueryParams              map[string]string
	ExtraFields              map[string]any
}

// NewOpenRouter builds an OpenRouter Chat Completions client. If BaseURL is
// empty, the OpenRouter API base URL is used.
func NewOpenRouter(cfg OpenRouterConfig) *Client {
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

	extraFields := maps.Clone(cfg.ExtraFields)
	if extraFields == nil {
		extraFields = map[string]any{}
	}
	if cfg.RequireZeroDataRetention {
		extraFields["provider.zdr"] = true
	}

	return NewClient(cfg.ChatModel,
		WithAPIKey(cfg.APIKey),
		WithBaseURL(baseURL),
		WithHeaders(headers),
		WithQueryParams(cfg.QueryParams),
		WithExtraFields(extraFields),
	)
}

// ChatRequest is an OpenAI-compatible Chat Completions request without the
// model field. Client injects its configured model into each request.
type ChatRequest struct {
	Messages          []ChatMessage `json:"messages"`
	Tools             []ChatTool    `json:"tools,omitempty"`
	Temperature       *float64      `json:"temperature,omitempty"`
	TopP              *float64      `json:"top_p,omitempty"`
	MaxTokens         *int          `json:"max_completion_tokens,omitempty"`
	ParallelToolCalls *bool         `json:"parallel_tool_calls,omitempty"`
	ToolChoice        any           `json:"tool_choice,omitempty"`
}

// ChatFunctionCall is a tool/function call payload in a chat message.
type ChatFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ChatToolCall is an OpenAI-compatible tool call.
type ChatToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function ChatFunctionCall `json:"function"`
}

// ChatMessage is an OpenAI-compatible chat message. ReasoningContent and
// Refusal are response fields exposed by compatible providers.
type ChatMessage struct {
	Role             string         `json:"role"`
	Content          string         `json:"content,omitempty"`
	ReasoningContent string         `json:"reasoning_content,omitempty"`
	Refusal          string         `json:"refusal,omitempty"`
	ToolCalls        []ChatToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string         `json:"tool_call_id,omitempty"`
}

// ChatFunctionDefinition describes a function tool.
type ChatFunctionDefinition struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
	Strict      bool            `json:"strict,omitempty"`
}

// ChatTool is an OpenAI-compatible tool definition.
type ChatTool struct {
	Type     string                 `json:"type"`
	Function ChatFunctionDefinition `json:"function"`
}

// ChatChoice is a single Chat Completions choice.
type ChatChoice struct {
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

// ChatUsage is token usage returned by the Chat Completions API.
type ChatUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ChatResponse is an OpenAI-compatible Chat Completions response.
type ChatResponse struct {
	ID      string       `json:"id"`
	Choices []ChatChoice `json:"choices"`
	Usage   ChatUsage    `json:"usage"`
}

// StatusError reports a non-200 response from the Chat Completions API.
type StatusError struct {
	StatusCode int
	Body       string
	RetryAfter string
}

func (e *StatusError) Error() string {
	body := strings.TrimSpace(e.Body)
	if body == "" {
		return fmt.Sprintf("chat completions returned status %d", e.StatusCode)
	}
	return fmt.Sprintf("chat completions returned status %d: %s", e.StatusCode, body)
}

// ResponseTooLargeError reports a response body that exceeded the configured
// maximum response size.
type ResponseTooLargeError struct {
	Limit int64
}

func (e *ResponseTooLargeError) Error() string {
	return fmt.Sprintf("chat completions response exceeded %d bytes", e.Limit)
}

type wireChatRequest struct {
	Model string `json:"model"`
	ChatRequest
}

// Complete sends a Chat Completions request. The client defaults to zero
// retries; configure retries explicitly or wrap clients in CircuitBreaker for
// immediate failover on retryable provider and transport failures.
func (c *Client) Complete(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	body, err := c.marshalRequest(req)
	if err != nil {
		return nil, fmt.Errorf("marshal chat completions request: %w", err)
	}

	var lastErr error
	for attempt := 0; ; attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(c.backoff(attempt, lastErr)):
			}
		}
		resp, err := c.roundTrip(ctx, body)
		if err == nil {
			return resp, nil
		}
		if !isTriggerError(ctx, err) {
			return nil, err
		}
		lastErr = err
		if attempt >= c.maxRetries {
			if c.maxRetries == 0 {
				return nil, err
			}
			return nil, fmt.Errorf("chat completions failed after %d attempts: %w", attempt+1, lastErr)
		}
	}
}

func (c *Client) marshalRequest(req ChatRequest) ([]byte, error) {
	body, err := json.Marshal(wireChatRequest{Model: c.model, ChatRequest: req})
	if err != nil {
		return nil, err
	}
	if len(c.extraFields) == 0 {
		return body, nil
	}
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, err
	}
	applyExtraFields(payload, c.extraFields)
	return json.Marshal(payload)
}

func (c *Client) backoff(attempt int, lastErr error) time.Duration {
	var statusErr *StatusError
	if errors.As(lastErr, &statusErr) && statusErr.RetryAfter != "" {
		secs, err := strconv.Atoi(statusErr.RetryAfter)
		if err == nil && secs >= 0 {
			return time.Duration(secs) * time.Second
		}
	}
	delay := c.retryDelay << (attempt - 1)
	return delay + rand.N(delay/4+1)
}

func (c *Client) roundTrip(ctx context.Context, body []byte) (*ChatResponse, error) {
	endpoint, err := c.endpoint()
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build chat completions request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	for key, value := range c.headers {
		httpReq.Header.Set(key, value)
	}

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("call chat completions: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := readResponseBody(resp.Body, c.maxRespBody)
	if err != nil {
		return nil, fmt.Errorf("read chat completions response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, &StatusError{
			StatusCode: resp.StatusCode,
			Body:       string(respBody),
			RetryAfter: resp.Header.Get("Retry-After"),
		}
	}

	var parsed ChatResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return nil, fmt.Errorf("decode chat completions response: %w", err)
	}
	return &parsed, nil
}

func (c *Client) endpoint() (string, error) {
	endpoint := c.baseURL + "/chat/completions"
	if len(c.queryParams) == 0 {
		return endpoint, nil
	}
	req, err := http.NewRequest(http.MethodPost, endpoint, nil)
	if err != nil {
		return "", fmt.Errorf("build chat completions endpoint: %w", err)
	}
	q := req.URL.Query()
	for key, value := range c.queryParams {
		q.Set(key, value)
	}
	req.URL.RawQuery = q.Encode()
	return req.URL.String(), nil
}

func readResponseBody(r io.Reader, limit int64) ([]byte, error) {
	if limit <= 0 {
		return io.ReadAll(r)
	}
	body, err := io.ReadAll(io.LimitReader(r, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(body)) > limit {
		return nil, &ResponseTooLargeError{Limit: limit}
	}
	return body, nil
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

	if statusErr, ok := errors.AsType[*StatusError](err); ok {
		switch {
		case statusErr.StatusCode >= http.StatusInternalServerError:
			return true
		case statusErr.StatusCode == http.StatusTooManyRequests:
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

func newDefaultHTTPClient() *http.Client {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.DialContext = (&net.Dialer{Timeout: defaultDialTimeout}).DialContext
	return &http.Client{
		Transport: transport,
		Timeout:   defaultHTTPTimeout,
	}
}

func applyExtraFields(payload map[string]any, extraFields map[string]any) {
	for key, value := range extraFields {
		setJSONPath(payload, strings.Split(key, "."), value)
	}
}

func setJSONPath(payload map[string]any, path []string, value any) {
	if len(path) == 1 {
		payload[path[0]] = value
		return
	}
	next, ok := payload[path[0]].(map[string]any)
	if !ok {
		next = map[string]any{}
		payload[path[0]] = next
	}
	setJSONPath(next, path[1:], value)
}

// vllmReasoningFields returns vLLM-specific request body extensions for
// reasoning-capable local models. These fields are not part of the OpenAI Chat
// Completions spec; they target vLLM deployments and compatible chat templates
// that understand thinking_token_budget, chat_template_kwargs, and reasoning.
func vllmReasoningFields(budget *int) map[string]any {
	if budget == nil || *budget < 0 {
		return nil
	}
	extraFields := map[string]any{"thinking_token_budget": *budget}
	switch *budget {
	case 0:
		extraFields["chat_template_kwargs.enable_thinking"] = false
		extraFields["reasoning.effort"] = "none"
	default:
		extraFields["reasoning.max_tokens"] = *budget
	}
	return extraFields
}
