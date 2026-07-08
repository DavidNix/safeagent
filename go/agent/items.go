// Package agent is a Go port of the core runtime of the OpenAI Agents SDK
// (openai-agents-js): agents with instructions, function tools, handoffs,
// guardrails, and a run loop that drives a Model until a final output.
package agent

// Role identifies the author of a message.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

// Item is a single entry in a conversation history: a message, a tool call,
// or a tool output.
type Item interface {
	isItem()
}

// Message is a plain text message in the conversation.
type Message struct {
	Role    Role
	Content string
}

func (Message) isItem() {}

// ToolCall is a request from the model to invoke a tool by name.
type ToolCall struct {
	ID        string
	Name      string
	Arguments string
}

func (ToolCall) isItem() {}

// ToolOutput is the result of a tool invocation, correlated by call ID.
type ToolOutput struct {
	CallID string
	Name   string
	Output string
}

func (ToolOutput) isItem() {}

// UserMessage builds a user role message item.
func UserMessage(content string) Message {
	return Message{Role: RoleUser, Content: content}
}

// AssistantMessage builds an assistant role message item.
func AssistantMessage(content string) Message {
	return Message{Role: RoleAssistant, Content: content}
}

// SystemMessage builds a system role message item.
func SystemMessage(content string) Message {
	return Message{Role: RoleSystem, Content: content}
}
