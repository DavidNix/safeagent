package agent

import "sync"

// RunContext carries the caller-provided context value and accumulated model
// usage through a single agent run. It is passed to tools and guardrails.
type RunContext struct {
	// Context is the value supplied via Runner.Context.
	Context any

	mu    sync.Mutex
	usage Usage
}

// AddUsage accumulates model usage into the run total. Safe for concurrent
// use by tools running in parallel.
func (rc *RunContext) AddUsage(u Usage) {
	rc.mu.Lock()
	defer rc.mu.Unlock()
	rc.usage.Add(u)
}

// Usage returns a snapshot of the accumulated usage for the run so far.
func (rc *RunContext) Usage() Usage {
	rc.mu.Lock()
	defer rc.mu.Unlock()
	return rc.usage
}
