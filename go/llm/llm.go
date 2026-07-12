package llm

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"net"
	"net/http"
	"sort"
	"strings"
	"time"
)

const (
	// DefaultBaseURL is the OpenRouter OpenAI-compatible API base URL.
	DefaultBaseURL = "https://openrouter.ai/api/v1"
	// DefaultRequestTimeout is the maximum duration of one provider request.
	DefaultRequestTimeout = 60 * time.Second
	// DefaultMaxResponseBytes caps response bodies at 64 MiB. This is generous
	// for current OpenAI-compatible max completion sizes while still preventing
	// accidental unbounded reads. Use WithMaxResponseBytes(0) to disable it.
	DefaultMaxResponseBytes = 64 << 20

	defaultDialTimeout         = 5 * time.Second
	defaultFailureThreshold    = 3
	defaultOpenDuration        = 60 * time.Second
	defaultSuccessfulProbePass = 1
	defaultLocalAPIKey         = "local"
	defaultOpenRouterBaseURL   = DefaultBaseURL
)

// Client calls an OpenAI-compatible Chat Completions API.
type Client struct {
	providerID     string
	model          string
	apiKey         string
	baseURL        string
	httpClient     *http.Client
	headers        map[string]string
	queryParams    map[string]string
	extraFields    map[string]any
	maxRespBody    int64
	requestTimeout time.Duration
}

// Option customizes a Client.
type Option func(*Client)

// WithAPIKey overrides the bearer token used for API requests.
func WithAPIKey(key string) Option {
	return func(c *Client) { c.apiKey = key }
}

// WithProviderID identifies the provider in circuit-breaker failover events.
func WithProviderID(providerID string) Option {
	return func(c *Client) { c.providerID = providerID }
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

// WithMaxResponseBytes overrides the maximum response body size read by the
// client. Zero or a negative value disables the limit.
func WithMaxResponseBytes(n int64) Option {
	return func(c *Client) { c.maxRespBody = n }
}

// WithRequestTimeout overrides the maximum duration of one provider request.
// Zero or a negative duration disables the client-level timeout.
func WithRequestTimeout(timeout time.Duration) Option {
	return func(c *Client) { c.requestTimeout = timeout }
}

// NewClient builds a Chat Completions client for model. By default it uses
// DefaultBaseURL.
func NewClient(model string, opts ...Option) *Client {
	c := &Client{
		model:          model,
		baseURL:        DefaultBaseURL,
		httpClient:     newDefaultHTTPClient(),
		headers:        map[string]string{},
		queryParams:    map[string]string{},
		extraFields:    map[string]any{},
		maxRespBody:    DefaultMaxResponseBytes,
		requestTimeout: DefaultRequestTimeout,
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
	ProviderID  string
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
	apiKey := cmp.Or(cfg.APIKey, defaultLocalAPIKey)
	extraFields := maps.Clone(cfg.ExtraFields)
	if extraFields == nil {
		extraFields = map[string]any{}
	}
	maps.Copy(extraFields, vllmReasoningFields(cfg.ReasoningTokenBudget))

	return NewClient(cfg.ChatModel,
		WithProviderID(cfg.ProviderID),
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
	ProviderID               string
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

	return NewClient(cfg.ChatModel,
		WithProviderID(cfg.ProviderID),
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
	Messages          []ChatMessage     `json:"messages"`
	Tools             []ChatTool        `json:"tools,omitempty"`
	Temperature       *float64          `json:"temperature,omitempty"`
	TopP              *float64          `json:"top_p,omitempty"`
	MaxTokens         *int              `json:"max_completion_tokens,omitempty"`
	ParallelToolCalls *bool             `json:"parallel_tool_calls,omitempty"`
	ToolChoice        any               `json:"tool_choice,omitempty"`
	StructuredOutput  *StructuredOutput `json:"-"`
}

// StructuredOutput constrains a chat completion to a JSON Schema. Providers
// must support OpenAI-compatible strict structured outputs.
//
// Example:
//
//	request := llm.ChatRequest{
//		Messages: []llm.ChatMessage{{Role: "user", Content: "Give me an answer."}},
//		StructuredOutput: &llm.StructuredOutput{
//			Name:   "answer",
//			Schema: json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}`),
//			Strict: true,
//		},
//	}
type StructuredOutput struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Schema      json.RawMessage `json:"schema"`
	Strict      bool            `json:"strict,omitempty"`
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

func (m ChatMessage) MarshalJSON() ([]byte, error) {
	var content *string
	if m.Role != "assistant" || len(m.ToolCalls) == 0 || m.Content != "" {
		content = &m.Content
	}
	return json.Marshal(struct {
		Role             string         `json:"role"`
		Content          *string        `json:"content,omitempty"`
		ReasoningContent string         `json:"reasoning_content,omitempty"`
		Refusal          string         `json:"refusal,omitempty"`
		ToolCalls        []ChatToolCall `json:"tool_calls,omitempty"`
		ToolCallID       string         `json:"tool_call_id,omitempty"`
	}{
		Role:             m.Role,
		Content:          content,
		ReasoningContent: m.ReasoningContent,
		Refusal:          m.Refusal,
		ToolCalls:        m.ToolCalls,
		ToolCallID:       m.ToolCallID,
	})
}

func (m *ChatMessage) UnmarshalJSON(data []byte) error {
	type chatMessage ChatMessage
	var raw struct {
		chatMessage
		Reasoning string `json:"reasoning,omitempty"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	*m = ChatMessage(raw.chatMessage)
	if m.ReasoningContent == "" {
		m.ReasoningContent = raw.Reasoning
	}
	return nil
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

// ModelInfo describes one model returned by an OpenAI-compatible Models API.
type ModelInfo struct {
	ID      string `json:"id"`
	Object  string `json:"object,omitempty"`
	Created int64  `json:"created,omitempty"`
	OwnedBy string `json:"owned_by,omitempty"`
}

// ModelsResponse is the OpenAI-compatible Models API response.
type ModelsResponse struct {
	Object string      `json:"object"`
	Data   []ModelInfo `json:"data"`
}

// StatusError reports a non-200 response from a provider API.
type StatusError struct {
	StatusCode int
	Body       string
	operation  string
}

func (e *StatusError) Error() string {
	operation := cmp.Or(e.operation, "chat completions")
	body := strings.TrimSpace(e.Body)
	if body == "" {
		return fmt.Sprintf("%s returned status %d", operation, e.StatusCode)
	}
	return fmt.Sprintf("%s returned status %d: %s", operation, e.StatusCode, body)
}

// ResponseTooLargeError reports a response body that exceeded the configured
// maximum response size.
type ResponseTooLargeError struct {
	Limit     int64
	operation string
}

func (e *ResponseTooLargeError) Error() string {
	operation := cmp.Or(e.operation, "chat completions")
	return fmt.Sprintf("%s response exceeded %d bytes", operation, e.Limit)
}

type wireChatRequest struct {
	Model          string              `json:"model"`
	ResponseFormat *wireResponseFormat `json:"response_format,omitempty"`
	ChatRequest
}

type wireResponseFormat struct {
	Type       string            `json:"type"`
	JSONSchema *StructuredOutput `json:"json_schema"`
}

type requestSerializationError struct {
	operation string
	err       error
}

func (e *requestSerializationError) Error() string {
	return fmt.Sprintf("marshal %s request: %v", e.operation, e.err)
}

func (e *requestSerializationError) Unwrap() error {
	return e.err
}

// Complete sends one Chat Completions request.
func (c *Client) Complete(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	body, err := c.prepareChatRequest(ctx, req)
	if err != nil {
		return nil, err
	}
	return c.complete(ctx, body)
}

func (c *Client) prepareChatRequest(ctx context.Context, req ChatRequest) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	body, err := c.marshalRequest(req)
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, ctxErr
		}
		return nil, &requestSerializationError{operation: "chat completions", err: err}
	}
	return body, nil
}

func (c *Client) complete(ctx context.Context, body []byte) (*ChatResponse, error) {
	ctx, cancel := c.requestContext(ctx)
	defer cancel()
	return c.roundTrip(ctx, body)
}

// Models returns the models advertised by the configured OpenAI-compatible API.
func (c *Client) Models(ctx context.Context) (*ModelsResponse, error) {
	ctx, cancel := c.requestContext(ctx)
	defer cancel()

	endpoint, err := c.endpoint("models")
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("build models request: %w", err)
	}
	httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	for key, value := range c.headers {
		httpReq.Header.Set(key, value)
	}

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("call models: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := readResponseBody(resp.Body, c.maxRespBody)
	if err != nil {
		return nil, fmt.Errorf("read models response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, &modelsStatusError{StatusCode: resp.StatusCode, Body: string(respBody)}
	}

	var parsed ModelsResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return nil, fmt.Errorf("decode models response: %w", err)
	}
	return &parsed, nil
}

func (c *Client) marshalRequest(req ChatRequest) ([]byte, error) {
	wireReq := wireChatRequest{Model: c.model, ChatRequest: req}
	if req.StructuredOutput != nil {
		wireReq.ResponseFormat = &wireResponseFormat{Type: "json_schema", JSONSchema: req.StructuredOutput}
	}
	body, err := json.Marshal(wireReq)
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
	responseFormat := payload["response_format"]
	applyExtraFields(payload, c.extraFields)
	if responseFormat != nil {
		payload["response_format"] = responseFormat
	}
	return json.Marshal(payload)
}

func (c *Client) roundTrip(ctx context.Context, body []byte) (*ChatResponse, error) {
	endpoint, err := c.endpoint("chat/completions")
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
	if resp.StatusCode != http.StatusOK {
		var tooLarge *ResponseTooLargeError
		if err != nil && !errors.As(err, &tooLarge) {
			return nil, fmt.Errorf("read chat completions response: %w", err)
		}
		return nil, &StatusError{
			StatusCode: resp.StatusCode,
			Body:       string(respBody),
		}
	}
	if err != nil {
		return nil, fmt.Errorf("read chat completions response: %w", err)
	}

	var parsed ChatResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return nil, fmt.Errorf("decode chat completions response: %w", err)
	}
	return &parsed, nil
}

func (c *Client) endpoint(path string) (string, error) {
	endpoint := c.baseURL + "/" + strings.TrimPrefix(path, "/")
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

type modelsStatusError struct {
	StatusCode int
	Body       string
}

func (e *modelsStatusError) Error() string {
	body := strings.TrimSpace(e.Body)
	if body == "" {
		return fmt.Sprintf("models returned status %d", e.StatusCode)
	}
	return fmt.Sprintf("models returned status %d: %s", e.StatusCode, body)
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
		return body[:limit], &ResponseTooLargeError{Limit: limit}
	}
	return body, nil
}

func (c *Client) requestContext(ctx context.Context) (context.Context, context.CancelFunc) {
	if c.requestTimeout <= 0 {
		return ctx, func() {}
	}
	return context.WithTimeout(ctx, c.requestTimeout)
}

func newDefaultHTTPClient() *http.Client {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.DialContext = (&net.Dialer{Timeout: defaultDialTimeout}).DialContext
	return &http.Client{
		Transport: transport,
	}
}

func applyExtraFields(payload map[string]any, extraFields map[string]any) {
	keys := make([]string, 0, len(extraFields))
	for key := range extraFields {
		keys = append(keys, key)
	}
	sort.Slice(keys, func(i, j int) bool {
		leftDepth := strings.Count(keys[i], ".")
		rightDepth := strings.Count(keys[j], ".")
		if leftDepth != rightDepth {
			return leftDepth < rightDepth
		}
		return keys[i] < keys[j]
	})
	for _, key := range keys {
		value := extraFields[key]
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
