// Package llm provides an agent.Model backed by any OpenAI-compatible
// Chat Completions API. Callers construct and configure their own client
// (the official openai-go SDK, an Azure variant, or a test fake) and inject
// it via the Client interface.
package llm

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/shared"

	"github.com/DavidNix/safeagent/agent"
)

// Client is the subset of the Chat Completions API that Model uses.
// *openai.ChatCompletionService satisfies it: pass &client.Chat.Completions
// from an openai.Client configured however you need (API key, base URL,
// HTTP client, retries).
type Client interface {
	New(ctx context.Context, params openai.ChatCompletionNewParams, opts ...option.RequestOption) (*openai.ChatCompletion, error)
}

// Model calls an OpenAI-compatible Chat Completions API through an injected
// Client.
type Model struct {
	name   string
	client Client
}

// NewModel builds a Model for the named model, for example "gpt-4.1".
func NewModel(name string, client Client) *Model {
	return &Model{name: name, client: client}
}

// GetResponse sends the request to the Chat Completions API and converts the
// first choice into agent items.
func (m *Model) GetResponse(ctx context.Context, req agent.ModelRequest) (*agent.ModelResponse, error) {
	params, err := m.buildParams(req)
	if err != nil {
		return nil, err
	}
	completion, err := m.client.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("chat completion: %w", err)
	}
	if len(completion.Choices) == 0 {
		return nil, fmt.Errorf("chat completions returned no choices")
	}

	msg := completion.Choices[0].Message
	if msg.Refusal != "" {
		return nil, &agent.ModelBehaviorError{Message: "model refused to produce output: " + msg.Refusal}
	}

	var output []agent.Item
	if msg.Content != "" {
		output = append(output, agent.AssistantMessage(msg.Content))
	}
	for _, call := range msg.ToolCalls {
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
			InputTokens:  int(completion.Usage.PromptTokens),
			OutputTokens: int(completion.Usage.CompletionTokens),
			TotalTokens:  int(completion.Usage.TotalTokens),
		},
		ResponseID: completion.ID,
	}, nil
}

func (m *Model) buildParams(req agent.ModelRequest) (openai.ChatCompletionNewParams, error) {
	params := openai.ChatCompletionNewParams{
		Model:    shared.ChatModel(m.name),
		Messages: buildMessages(req),
	}
	tools, err := buildTools(req)
	if err != nil {
		return openai.ChatCompletionNewParams{}, err
	}
	params.Tools = tools

	settings := req.ModelSettings
	if settings.Temperature != nil {
		params.Temperature = openai.Float(*settings.Temperature)
	}
	if settings.TopP != nil {
		params.TopP = openai.Float(*settings.TopP)
	}
	if settings.MaxTokens != nil {
		params.MaxCompletionTokens = openai.Int(int64(*settings.MaxTokens))
	}
	if settings.ParallelToolCalls != nil {
		params.ParallelToolCalls = openai.Bool(*settings.ParallelToolCalls)
	}
	switch settings.ToolChoice {
	case "":
	case "auto", "required", "none":
		params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{
			OfAuto: openai.String(settings.ToolChoice),
		}
	default:
		params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{
			OfFunctionToolChoice: &openai.ChatCompletionNamedToolChoiceParam{
				Function: openai.ChatCompletionNamedToolChoiceFunctionParam{Name: settings.ToolChoice},
			},
		}
	}
	return params, nil
}

func buildMessages(req agent.ModelRequest) []openai.ChatCompletionMessageParamUnion {
	var messages []openai.ChatCompletionMessageParamUnion
	if req.SystemInstructions != "" {
		messages = append(messages, openai.SystemMessage(req.SystemInstructions))
	}
	var pendingCalls []openai.ChatCompletionMessageToolCallUnionParam
	flush := func() {
		if len(pendingCalls) > 0 {
			messages = append(messages, openai.ChatCompletionMessageParamUnion{
				OfAssistant: &openai.ChatCompletionAssistantMessageParam{ToolCalls: pendingCalls},
			})
			pendingCalls = nil
		}
	}
	for _, item := range req.Input {
		switch v := item.(type) {
		case agent.Message:
			flush()
			messages = append(messages, roleMessage(v))
		case agent.ToolCall:
			pendingCalls = append(pendingCalls, openai.ChatCompletionMessageToolCallUnionParam{
				OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
					ID: v.ID,
					Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
						Name:      v.Name,
						Arguments: v.Arguments,
					},
				},
			})
		case agent.ToolOutput:
			flush()
			messages = append(messages, openai.ToolMessage(v.Output, v.CallID))
		}
	}
	flush()
	return messages
}

func roleMessage(msg agent.Message) openai.ChatCompletionMessageParamUnion {
	switch msg.Role {
	case agent.RoleSystem:
		return openai.SystemMessage(msg.Content)
	case agent.RoleAssistant:
		return openai.AssistantMessage(msg.Content)
	default:
		return openai.UserMessage(msg.Content)
	}
}

func buildTools(req agent.ModelRequest) ([]openai.ChatCompletionToolUnionParam, error) {
	tools := make([]openai.ChatCompletionToolUnionParam, 0, len(req.Tools)+len(req.Handoffs))
	for _, tool := range req.Tools {
		def := shared.FunctionDefinitionParam{Name: tool.Name}
		if tool.Description != "" {
			def.Description = openai.String(tool.Description)
		}
		params, err := functionParameters(tool.Parameters)
		if err != nil {
			return nil, fmt.Errorf("tool %q: %w", tool.Name, err)
		}
		def.Parameters = params
		if tool.Strict {
			def.Strict = openai.Bool(true)
		}
		tools = append(tools, openai.ChatCompletionFunctionTool(def))
	}
	for _, handoff := range req.Handoffs {
		def := shared.FunctionDefinitionParam{Name: handoff.ToolName}
		if handoff.ToolDescription != "" {
			def.Description = openai.String(handoff.ToolDescription)
		}
		params, err := functionParameters(handoff.InputSchema)
		if err != nil {
			return nil, fmt.Errorf("handoff %q: %w", handoff.ToolName, err)
		}
		def.Parameters = params
		tools = append(tools, openai.ChatCompletionFunctionTool(def))
	}
	if len(tools) == 0 {
		return nil, nil
	}
	return tools, nil
}

func functionParameters(schema json.RawMessage) (shared.FunctionParameters, error) {
	if len(schema) == 0 {
		return nil, nil
	}
	var params shared.FunctionParameters
	if err := json.Unmarshal(schema, &params); err != nil {
		return nil, fmt.Errorf("parse json schema: %w", err)
	}
	return params, nil
}
