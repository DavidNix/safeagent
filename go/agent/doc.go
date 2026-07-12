// Package agent provides a runtime for building and running tool-using LLM
// agents.
//
// The package was originally ported from OpenAI's JavaScript Agents SDK and
// has since evolved as a Go-native runtime.
//
// An Agent combines instructions with a Model and may expose function tools,
// delegate to other agents through handoffs, and validate inputs and outputs
// with guardrails. Run and Runner execute the resulting turn loop until the
// active agent returns a final assistant message.
//
// Models implement a small call-site interface based on the OpenAI-compatible
// Chat Completions types in package llm. This keeps the runtime independent of
// any particular provider and makes local models and test doubles easy to use.
//
// Runner can carry application state to tools and guardrails through
// RunContext, collect usage across nested runs, and report lifecycle events to
// a Tracer.
package agent
