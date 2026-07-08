package client

import (
	"os"
	"testing"

	"github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/require"
)

func TestClient_E2E(t *testing.T) {
	t.Parallel()

	if testing.Short() {
		t.Skip("skipping e2e tests in short mode")
	}

	t.Run("openrouter", func(t *testing.T) {
		apiKey := os.Getenv("SAFEAGENT_OPENROUTER_API_KEY")
		chatModel := os.Getenv("SAFEAGENT_OPENROUTER_CHAT_MODEL")
		embeddingModel := os.Getenv("SAFEAGENT_OPENROUTER_EMBEDDING_MODEL")
		if apiKey == "" || chatModel == "" || embeddingModel == "" {
			t.Skip("SAFEAGENT_OPENROUTER_API_KEY, SAFEAGENT_OPENROUTER_CHAT_MODEL, and SAFEAGENT_OPENROUTER_EMBEDDING_MODEL must be set")
		}

		completionClient, err := NewCompletionClient(Options{
			Primary: OpenRouterConfig{
				APIKey:                   apiKey,
				ChatModel:                chatModel,
				EmbeddingModel:           embeddingModel,
				RequireZeroDataRetention: true,
			},
		})
		require.NoError(t, err)
		embeddingClient, err := NewEmbeddingClient(Options{
			Primary: OpenRouterConfig{
				APIKey:                   apiKey,
				ChatModel:                chatModel,
				EmbeddingModel:           embeddingModel,
				RequireZeroDataRetention: true,
			},
		})
		require.NoError(t, err)

		completion, err := completionClient.New(t.Context(), openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Reply with exactly: safeagent"),
			},
		})
		require.NoError(t, err)
		require.NotEmpty(t, completion.Choices)

		embeddings, err := embeddingClient.New(t.Context(), openai.EmbeddingNewParams{
			Input: openai.EmbeddingNewParamsInputUnion{OfString: openai.String("safeagent")},
		})
		require.NoError(t, err)
		require.NotEmpty(t, embeddings.Data)
		require.NotEmpty(t, embeddings.Data[0].Embedding)
	})

	t.Run("vllm", func(t *testing.T) {
		chatBaseURL := os.Getenv("SAFEAGENT_VLLM_CHAT_BASE_URL")
		embeddingsBaseURL := os.Getenv("SAFEAGENT_VLLM_EMBEDDINGS_BASE_URL")
		chatModel := os.Getenv("SAFEAGENT_VLLM_CHAT_MODEL")
		embeddingModel := os.Getenv("SAFEAGENT_VLLM_EMBEDDING_MODEL")
		if chatBaseURL == "" || embeddingsBaseURL == "" || chatModel == "" || embeddingModel == "" {
			t.Skip("SAFEAGENT_VLLM_CHAT_BASE_URL, SAFEAGENT_VLLM_EMBEDDINGS_BASE_URL, SAFEAGENT_VLLM_CHAT_MODEL, and SAFEAGENT_VLLM_EMBEDDING_MODEL must be set")
		}

		completionClient, err := NewCompletionClient(Options{
			Primary: VLLMConfig{
				ChatBaseURL:       chatBaseURL,
				EmbeddingsBaseURL: embeddingsBaseURL,
				ChatModel:         chatModel,
				EmbeddingModel:    embeddingModel,
			},
		})
		require.NoError(t, err)
		embeddingClient, err := NewEmbeddingClient(Options{
			Primary: VLLMConfig{
				ChatBaseURL:       chatBaseURL,
				EmbeddingsBaseURL: embeddingsBaseURL,
				ChatModel:         chatModel,
				EmbeddingModel:    embeddingModel,
			},
		})
		require.NoError(t, err)

		completion, err := completionClient.New(t.Context(), openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Reply with exactly: safeagent"),
			},
		})
		require.NoError(t, err)
		require.NotEmpty(t, completion.Choices)

		embeddings, err := embeddingClient.New(t.Context(), openai.EmbeddingNewParams{
			Input: openai.EmbeddingNewParamsInputUnion{OfString: openai.String("safeagent")},
		})
		require.NoError(t, err)
		require.NotEmpty(t, embeddings.Data)
		require.NotEmpty(t, embeddings.Data[0].Embedding)
	})
}
