package agent

// Usage tracks token usage and request counts for an agent run.
type Usage struct {
	Requests     int
	InputTokens  int
	OutputTokens int
	TotalTokens  int
}

// Add accumulates another usage entry into this one.
func (u *Usage) Add(other Usage) {
	u.Requests += other.Requests
	u.InputTokens += other.InputTokens
	u.OutputTokens += other.OutputTokens
	u.TotalTokens += other.TotalTokens
}
