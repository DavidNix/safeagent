package llm_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"

	"github.com/DavidNix/safeagent/llm"
)

func writeJSON(w http.ResponseWriter, value any) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(value); err != nil {
		panic(err)
	}
}

func ExampleClient_Complete() {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Model    string            `json:"model"`
			Messages []llm.ChatMessage `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			panic(err)
		}
		fmt.Printf("%s: %s\n", request.Model, request.Messages[0].Content)
		writeJSON(w, llm.ChatResponse{
			Choices: []llm.ChatChoice{{
				Message:      llm.ChatMessage{Role: "assistant", Content: "Hello!"},
				FinishReason: "stop",
			}},
		})
	}))
	defer server.Close()

	client := llm.NewClient("example-model",
		llm.WithBaseURL(server.URL),
		llm.WithAPIKey("example-key"),
	)
	response, err := client.Complete(context.Background(), llm.ChatRequest{
		Messages: []llm.ChatMessage{{Role: "user", Content: "Say hello."}},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(response.Choices[0].Message.Content)

	// Output:
	// example-model: Say hello.
	// Hello!
}

func ExampleStructuredOutput() {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			ResponseFormat struct {
				Type       string               `json:"type"`
				JSONSchema llm.StructuredOutput `json:"json_schema"`
			} `json:"response_format"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			panic(err)
		}
		fmt.Println(request.ResponseFormat.Type)
		fmt.Println(request.ResponseFormat.JSONSchema.Name)
		writeJSON(w, llm.ChatResponse{
			Choices: []llm.ChatChoice{{
				Message:      llm.ChatMessage{Role: "assistant", Content: `{"answer":"42"}`},
				FinishReason: "stop",
			}},
		})
	}))
	defer server.Close()

	client := llm.NewClient("example-model", llm.WithBaseURL(server.URL))
	response, err := client.Complete(context.Background(), llm.ChatRequest{
		Messages: []llm.ChatMessage{{Role: "user", Content: "What is the answer?"}},
		StructuredOutput: &llm.StructuredOutput{
			Name:   "answer",
			Schema: json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}`),
			Strict: true,
		},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(response.Choices[0].Message.Content)

	// Output:
	// json_schema
	// answer
	// {"answer":"42"}
}

func ExampleEmbeddingClient_Embed() {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Model string   `json:"model"`
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			panic(err)
		}
		fmt.Printf("%s: %d input\n", request.Model, len(request.Input))
		writeJSON(w, llm.EmbeddingResponse{
			Object: "list",
			Model:  request.Model,
			Data: []llm.Embedding{{
				Object:    "embedding",
				Index:     0,
				Embedding: []float32{0.25, 0.75},
			}},
		})
	}))
	defer server.Close()

	client := llm.NewEmbeddingClient(
		llm.EmbeddingConfig{Model: "embedding-model"},
		llm.WithBaseURL(server.URL),
	)
	response, err := client.Embed(context.Background(), llm.EmbeddingRequest{
		Input: []string{"embed this"},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(response.Data[0].Embedding)

	// Output:
	// embedding-model: 1 input
	// [0.25 0.75]
}

func ExampleCircuitBreaker_Complete() {
	primary := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "temporarily unavailable", http.StatusServiceUnavailable)
	}))
	defer primary.Close()
	fallback := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, llm.ChatResponse{
			Choices: []llm.ChatChoice{{
				Message:      llm.ChatMessage{Role: "assistant", Content: "served by fallback"},
				FinishReason: "stop",
			}},
		})
	}))
	defer fallback.Close()

	breaker := llm.NewCircuitBreaker(
		llm.NewClient("model", llm.WithProviderID("primary"), llm.WithBaseURL(primary.URL)),
		llm.NewClient("model", llm.WithProviderID("fallback"), llm.WithBaseURL(fallback.URL)),
	)
	response, err := breaker.Complete(context.Background(), llm.ChatRequest{
		Messages: []llm.ChatMessage{{Role: "user", Content: "Hello"}},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(response.Choices[0].Message.Content)

	// Output:
	// served by fallback
}

func ExampleClient_Models() {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, llm.ModelsResponse{
			Object: "list",
			Data: []llm.ModelInfo{
				{ID: "small-model"},
				{ID: "large-model"},
			},
		})
	}))
	defer server.Close()

	client := llm.NewClient("unused", llm.WithBaseURL(server.URL))
	models, err := client.Models(context.Background())
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, model := range models.Data {
		fmt.Println(model.ID)
	}

	// Output:
	// small-model
	// large-model
}

func ExampleNewOpenRouter() {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request map[string]any
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			panic(err)
		}
		provider := request["provider"].(map[string]any)
		fmt.Println(r.Header.Get("HTTP-Referer"))
		fmt.Println(r.Header.Get("X-Title"))
		fmt.Println(provider["zdr"])
		writeJSON(w, llm.ChatResponse{
			Choices: []llm.ChatChoice{{
				Message:      llm.ChatMessage{Role: "assistant", Content: "ok"},
				FinishReason: "stop",
			}},
		})
	}))
	defer server.Close()

	client := llm.NewOpenRouter(llm.OpenRouterConfig{
		APIKey:                   "example-key",
		BaseURL:                  server.URL,
		ChatModel:                "openrouter/model",
		SiteURL:                  "https://example.com",
		AppTitle:                 "Example App",
		RequireZeroDataRetention: true,
	})
	_, err := client.Complete(context.Background(), llm.ChatRequest{
		Messages: []llm.ChatMessage{{Role: "user", Content: "Hello"}},
	})
	if err != nil {
		fmt.Println(err)
	}

	// Output:
	// https://example.com
	// Example App
	// true
}
