package agent_test

import (
	"testing"

	"github.com/DavidNix/safeagent/agent"
	"github.com/stretchr/testify/require"
)

func TestUsage_Add(t *testing.T) {
	t.Parallel()

	t.Run("happy path", func(t *testing.T) {
		usage := agent.Usage{Requests: 1, InputTokens: 10, OutputTokens: 5, TotalTokens: 15}
		usage.Add(agent.Usage{Requests: 2, InputTokens: 20, OutputTokens: 10, TotalTokens: 30})

		require.Equal(t, agent.Usage{Requests: 3, InputTokens: 30, OutputTokens: 15, TotalTokens: 45}, usage)
	})
}

func TestRunContext_AddUsage(t *testing.T) {
	t.Parallel()

	t.Run("happy path", func(t *testing.T) {
		rc := &agent.RunContext{}
		rc.AddUsage(agent.Usage{Requests: 1, TotalTokens: 5})
		rc.AddUsage(agent.Usage{Requests: 1, TotalTokens: 7})

		require.Equal(t, agent.Usage{Requests: 2, TotalTokens: 12}, rc.Usage())
	})
}
