package llm_test

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/DavidNix/safeagent/go/llm"
	"github.com/stretchr/testify/require"
)

const (
	openRouterAPIKeyEnv              = "OPENROUTER_API_KEY"
	openRouterE2EModel               = "qwen/qwen3.6-35b-a3b"
	vllmBaseURLEnv                   = "VLLM_BASE_URL"
	vllmEmbeddingBaseURLEnv          = "VLLM_EMBEDDING_BASE_URL"
	e2eTimeout                       = 2 * time.Minute
	e2eMaxCompletionTokens           = 32
	e2eReasoningTokenBudget          = 256
	e2eReasoningMaxCompletionTokens  = 768
	e2eStructuredMaxCompletionTokens = 128
)

type e2eChatCompleter interface {
	Complete(ctx context.Context, req llm.ChatRequest) (*llm.ChatResponse, error)
}

func skipE2EInShortMode(t *testing.T) {
	t.Helper()
	if testing.Short() {
		t.Skip("skipping e2e test in short mode")
	}
}

func e2eContext(t *testing.T) (context.Context, context.CancelFunc) {
	t.Helper()
	return context.WithTimeout(t.Context(), e2eTimeout)
}

func requireE2EEnv(t *testing.T, name string) string {
	t.Helper()
	value := strings.TrimSpace(os.Getenv(name))
	require.NotEmpty(t, value, "%s must be set for e2e tests", name)
	return value
}

func newOpenRouterE2EClient(apiKey, baseURL string) *llm.Client {
	cfg := llm.OpenRouterConfig{
		APIKey:    apiKey,
		ChatModel: openRouterE2EModel,
		SiteURL:   "https://github.com/DavidNix/safeagent",
		AppTitle:  "SafeAgent E2E",
		ExtraFields: map[string]any{
			"reasoning.effort": "none",
		},
	}
	if baseURL != "" {
		cfg.BaseURL = baseURL
	}
	return llm.NewOpenRouter(cfg)
}

func newOpenRouterReasoningE2EClient(apiKey, baseURL string) *llm.Client {
	cfg := llm.OpenRouterConfig{
		APIKey:    apiKey,
		ChatModel: openRouterE2EModel,
		SiteURL:   "https://github.com/DavidNix/safeagent",
		AppTitle:  "SafeAgent E2E",
		ExtraFields: map[string]any{
			"reasoning.max_tokens": e2eReasoningTokenBudget,
		},
	}
	if baseURL != "" {
		cfg.BaseURL = baseURL
	}
	return llm.NewOpenRouter(cfg)
}

func newVLLME2EClient(t *testing.T, ctx context.Context, baseURL string) *llm.Client {
	t.Helper()
	model := requireFirstVLLME2EModel(t, ctx, baseURL)
	reasoningBudget := 0

	return llm.NewVLLM(llm.VLLMConfig{
		ChatBaseURL:          baseURL,
		ChatModel:            model,
		ReasoningTokenBudget: &reasoningBudget,
	})
}

func newVLLMReasoningE2EClient(t *testing.T, ctx context.Context, baseURL string) *llm.Client {
	t.Helper()
	model := requireFirstVLLME2EModel(t, ctx, baseURL)
	reasoningBudget := e2eReasoningTokenBudget

	return llm.NewVLLM(llm.VLLMConfig{
		ChatBaseURL:          baseURL,
		ChatModel:            model,
		ReasoningTokenBudget: &reasoningBudget,
	})
}

func requireFirstVLLME2EModel(t *testing.T, ctx context.Context, baseURL string) string {
	t.Helper()
	modelsClient := llm.NewVLLM(llm.VLLMConfig{ChatBaseURL: baseURL})
	models, err := modelsClient.Models(ctx)
	require.NoError(t, err)
	require.NotEmpty(t, models.Data, "vLLM models response returned no models")
	model := strings.TrimSpace(models.Data[0].ID)
	require.NotEmpty(t, model, "first vLLM model ID is empty")
	return model
}

func runLLME2ECompletion(t *testing.T, ctx context.Context, client e2eChatCompleter, marker string) *llm.ChatResponse {
	t.Helper()
	temperature := 0.0
	maxTokens := e2eMaxCompletionTokens
	resp, err := client.Complete(ctx, llm.ChatRequest{
		Messages: []llm.ChatMessage{
			{Role: "system", Content: "You are running a SafeAgent end-to-end test. Reply with exactly the requested marker and no other text."},
			{Role: "user", Content: "Return exactly this marker: " + marker},
		},
		Temperature: &temperature,
		MaxTokens:   &maxTokens,
	})

	require.NoError(t, err)
	require.NotNil(t, resp)
	require.NotEmpty(t, resp.Choices)
	require.Contains(t, resp.Choices[0].Message.Content, marker)
	return resp
}

func runLLMReasoningE2ECompletion(t *testing.T, ctx context.Context, client e2eChatCompleter, marker string) *llm.ChatResponse {
	t.Helper()
	temperature := 0.0
	maxTokens := e2eReasoningMaxCompletionTokens
	resp, err := client.Complete(ctx, llm.ChatRequest{
		Messages: []llm.ChatMessage{
			{Role: "system", Content: "You are running a SafeAgent reasoning end-to-end test. Think briefly, then include the requested marker in the final answer."},
			{Role: "user", Content: "Think briefly about why 2+2=4, then include this marker in the final answer: " + marker},
		},
		Temperature: &temperature,
		MaxTokens:   &maxTokens,
	})

	require.NoError(t, err)
	require.NotNil(t, resp)
	require.NotEmpty(t, resp.Choices)
	require.NotEmpty(t, resp.Choices[0].Message.ReasoningContent)
	require.Contains(t, resp.Choices[0].Message.Content, marker)
	return resp
}

func runLLMStructuredOutputE2E(t *testing.T, ctx context.Context, client e2eChatCompleter, marker string) *llm.ChatResponse {
	t.Helper()
	temperature := 0.0
	maxTokens := e2eStructuredMaxCompletionTokens
	resp, err := client.Complete(ctx, llm.ChatRequest{
		Messages: []llm.ChatMessage{
			{Role: "system", Content: "You are running a SafeAgent structured-output end-to-end test. Follow the supplied JSON Schema exactly."},
			{Role: "user", Content: "Set marker to exactly " + marker + " and passed to true."},
		},
		Temperature: &temperature,
		MaxTokens:   &maxTokens,
		StructuredOutput: &llm.StructuredOutput{
			Name:        "safeagent_e2e_result",
			Description: "The result of a SafeAgent structured-output end-to-end test.",
			Schema: json.RawMessage(`{
				"type":"object",
				"properties":{
					"marker":{"type":"string"},
					"passed":{"type":"boolean"}
				},
				"required":["marker","passed"],
				"additionalProperties":false
			}`),
			Strict: true,
		},
	})

	require.NoError(t, err)
	require.NotNil(t, resp)
	require.NotEmpty(t, resp.Choices)
	var output struct {
		Marker string `json:"marker"`
		Passed bool   `json:"passed"`
	}
	require.NoError(t, json.Unmarshal([]byte(resp.Choices[0].Message.Content), &output))
	require.Equal(t, marker, output.Marker)
	require.True(t, output.Passed)
	return resp
}
