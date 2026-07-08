// Package llm provides an agent.Model that speaks the OpenAI-compatible
// Chat Completions wire format over plain HTTP (OpenAI, OpenRouter, Ollama,
// vLLM, and other compatible servers). It owns its request and response
// types, so provider extensions such as vLLM's reasoning_content are
// first-class fields rather than hidden extras.
package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math/rand/v2"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/DavidNix/safeagent/agent"
)

// DefaultBaseURL is the OpenAI API base URL.
const DefaultBaseURL = "https://api.openai.com/v1"

// DefaultMaxRetries is the number of retries after a retryable failure
// (HTTP 429, 5xx, or a transport error).
const DefaultMaxRetries = 2

const defaultRetryDelay = 500 * time.Millisecond

// Doer executes a single HTTP request. *http.Client satisfies it; wrap one
// to add middleware such as logging or custom auth.
type Doer interface {
	Do(req *http.Request) (*http.Response, error)
}

// Model calls an OpenAI-compatible Chat Completions API.
type Model struct {
	name       string
	apiKey     string
	baseURL    string
	client     Doer
	maxRetries int
	retryDelay time.Duration
}

// Option customizes a Model.
type Option func(*Model)

// WithAPIKey overrides the API key read from the OPENAI_API_KEY environment
// variable.
func WithAPIKey(key string) Option {
	return func(m *Model) { m.apiKey = key }
}

// WithBaseURL overrides the API base URL, for example to point at a vLLM or
// other OpenAI-compatible server.
func WithBaseURL(url string) Option {
	return func(m *Model) { m.baseURL = strings.TrimSuffix(url, "/") }
}

// WithClient overrides the HTTP client used to execute requests.
func WithClient(client Doer) Option {
	return func(m *Model) { m.client = client }
}

// WithMaxRetries overrides how many times a retryable failure is retried.
// Zero disables retries.
func WithMaxRetries(n int) Option {
	return func(m *Model) { m.maxRetries = n }
}

// WithRetryDelay overrides the base delay of the exponential backoff between
// retries. A Retry-After response header takes precedence.
func WithRetryDelay(d time.Duration) Option {
	return func(m *Model) { m.retryDelay = d }
}

// NewModel builds a Model for the named model, for example "gpt-4.1".
func NewModel(name string, opts ...Option) *Model {
	m := &Model{
		name:       name,
		apiKey:     os.Getenv("OPENAI_API_KEY"),
		baseURL:    DefaultBaseURL,
		client:     &http.Client{Timeout: 120 * time.Second},
		maxRetries: DefaultMaxRetries,
		retryDelay: defaultRetryDelay,
	}
	for _, opt := range opts {
		opt(m)
	}
	return m
}

type chatFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type chatToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function chatFunctionCall `json:"function"`
}

type chatMessage struct {
	Role       string         `json:"role"`
	Content    string         `json:"content,omitempty"`
	ToolCalls  []chatToolCall `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
}

type chatFunctionDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
	Strict      bool            `json:"strict,omitempty"`
}

type chatTool struct {
	Type     string          `json:"type"`
	Function chatFunctionDef `json:"function"`
}

type chatRequest struct {
	Model             string        `json:"model"`
	Messages          []chatMessage `json:"messages"`
	Tools             []chatTool    `json:"tools,omitempty"`
	Temperature       *float64      `json:"temperature,omitempty"`
	TopP              *float64      `json:"top_p,omitempty"`
	MaxTokens         *int          `json:"max_completion_tokens,omitempty"`
	ParallelToolCalls *bool         `json:"parallel_tool_calls,omitempty"`
	ToolChoice        any           `json:"tool_choice,omitempty"`
}

type chatResponse struct {
	ID      string `json:"id"`
	Choices []struct {
		Message struct {
			Content          string         `json:"content"`
			ReasoningContent string         `json:"reasoning_content"`
			Refusal          string         `json:"refusal"`
			ToolCalls        []chatToolCall `json:"tool_calls"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// GetResponse sends the request to the Chat Completions API, retrying
// retryable failures, and converts the first choice into agent items.
func (m *Model) GetResponse(ctx context.Context, req agent.ModelRequest) (*agent.ModelResponse, error) {
	body, err := json.Marshal(m.buildRequest(req))
	if err != nil {
		return nil, fmt.Errorf("marshal chat completions request: %w", err)
	}

	var lastErr error
	for attempt := 0; attempt <= m.maxRetries; attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(m.backoff(attempt, lastErr)):
			}
		}
		resp, err := m.roundTrip(ctx, body)
		if err == nil {
			return resp, nil
		}
		if _, ok := errors.AsType[*retryableError](err); !ok {
			return nil, err
		}
		lastErr = err
	}
	return nil, fmt.Errorf("chat completions failed after %d attempts: %w", m.maxRetries+1, lastErr)
}

// retryableError marks a failure worth retrying and carries the server's
// Retry-After hint, if any.
type retryableError struct {
	err        error
	retryAfter string
}

func (e *retryableError) Error() string {
	return e.err.Error()
}

func (e *retryableError) Unwrap() error {
	return e.err
}

func (m *Model) backoff(attempt int, lastErr error) time.Duration {
	var retryable *retryableError
	if errors.As(lastErr, &retryable) && retryable.retryAfter != "" {
		if secs, err := strconv.Atoi(retryable.retryAfter); err == nil && secs >= 0 {
			return time.Duration(secs) * time.Second
		}
	}
	delay := m.retryDelay << (attempt - 1)
	return delay + rand.N(delay/4+1)
}

func (m *Model) roundTrip(ctx context.Context, body []byte) (*agent.ModelResponse, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, m.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build chat completions request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+m.apiKey)

	resp, err := m.client.Do(httpReq)
	if err != nil {
		if ctx.Err() != nil {
			return nil, fmt.Errorf("call chat completions: %w", err)
		}
		return nil, &retryableError{err: fmt.Errorf("call chat completions: %w", err)}
	}
	defer func() {
		if cerr := resp.Body.Close(); cerr != nil {
			slog.Warn("Failed to close chat completions response body", "error", cerr)
		}
	}()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, &retryableError{err: fmt.Errorf("read chat completions response: %w", err)}
	}
	if resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= http.StatusInternalServerError {
		return nil, &retryableError{
			err:        fmt.Errorf("chat completions returned status %d: %s", resp.StatusCode, strings.TrimSpace(string(respBody))),
			retryAfter: resp.Header.Get("Retry-After"),
		}
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("chat completions returned status %d: %s", resp.StatusCode, strings.TrimSpace(string(respBody)))
	}

	var parsed chatResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return nil, fmt.Errorf("decode chat completions response: %w", err)
	}
	if len(parsed.Choices) == 0 {
		return nil, fmt.Errorf("chat completions returned no choices")
	}

	choice := parsed.Choices[0]
	if choice.Message.Refusal != "" {
		return nil, &agent.ModelBehaviorError{Message: "model refused to produce output: " + choice.Message.Refusal}
	}

	var output []agent.Item
	if choice.Message.ReasoningContent != "" {
		output = append(output, agent.Reasoning{Content: choice.Message.ReasoningContent})
	}
	if choice.Message.Content != "" {
		output = append(output, agent.AssistantMessage(choice.Message.Content))
	}
	for _, call := range choice.Message.ToolCalls {
		output = append(output, agent.ToolCall{
			ID:        call.ID,
			Name:      call.Function.Name,
			Arguments: call.Function.Arguments,
		})
	}

	return &agent.ModelResponse{
		Output: output,
		Usage: agent.Usage{
			Requests:     1,
			InputTokens:  parsed.Usage.PromptTokens,
			OutputTokens: parsed.Usage.CompletionTokens,
			TotalTokens:  parsed.Usage.TotalTokens,
		},
		ResponseID: parsed.ID,
	}, nil
}

func (m *Model) buildRequest(req agent.ModelRequest) chatRequest {
	settings := req.ModelSettings
	return chatRequest{
		Model:             m.name,
		Messages:          buildMessages(req),
		Tools:             buildTools(req),
		Temperature:       settings.Temperature,
		TopP:              settings.TopP,
		MaxTokens:         settings.MaxTokens,
		ParallelToolCalls: settings.ParallelToolCalls,
		ToolChoice:        toolChoiceValue(settings.ToolChoice),
	}
}

func buildMessages(req agent.ModelRequest) []chatMessage {
	var messages []chatMessage
	if req.SystemInstructions != "" {
		messages = append(messages, chatMessage{Role: "system", Content: req.SystemInstructions})
	}
	var pendingCalls []chatToolCall
	flush := func() {
		if len(pendingCalls) > 0 {
			messages = append(messages, chatMessage{Role: "assistant", ToolCalls: pendingCalls})
			pendingCalls = nil
		}
	}
	for _, item := range req.Input {
		switch v := item.(type) {
		case agent.Message:
			flush()
			messages = append(messages, chatMessage{Role: string(v.Role), Content: v.Content})
		case agent.ToolCall:
			pendingCalls = append(pendingCalls, chatToolCall{
				ID:       v.ID,
				Type:     "function",
				Function: chatFunctionCall{Name: v.Name, Arguments: v.Arguments},
			})
		case agent.ToolOutput:
			flush()
			messages = append(messages, chatMessage{Role: "tool", Content: v.Output, ToolCallID: v.CallID})
		case agent.Reasoning:
			// Reasoning traces are output-only and never replayed to the model.
		}
	}
	flush()
	return messages
}

func buildTools(req agent.ModelRequest) []chatTool {
	tools := make([]chatTool, 0, len(req.Tools)+len(req.Handoffs))
	for _, tool := range req.Tools {
		tools = append(tools, chatTool{
			Type: "function",
			Function: chatFunctionDef{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  tool.Parameters,
				Strict:      tool.Strict,
			},
		})
	}
	for _, handoff := range req.Handoffs {
		tools = append(tools, chatTool{
			Type: "function",
			Function: chatFunctionDef{
				Name:        handoff.ToolName,
				Description: handoff.ToolDescription,
				Parameters:  handoff.InputSchema,
			},
		})
	}
	if len(tools) == 0 {
		return nil
	}
	return tools
}

func toolChoiceValue(choice string) any {
	switch choice {
	case "":
		return nil
	case "auto", "required", "none":
		return choice
	default:
		return map[string]any{
			"type":     "function",
			"function": map[string]any{"name": choice},
		}
	}
}
