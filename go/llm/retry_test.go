package llm_test

import (
	"errors"
	"testing"
	"time"

	"github.com/DavidNix/safeagent/go/llm"
	"github.com/stretchr/testify/require"
)

func TestBackOffDelay(t *testing.T) {
	t.Parallel()

	config := testDelayContext{delay: 100 * time.Millisecond}

	t.Run("first retry uses base delay", func(t *testing.T) {
		require.Equal(t, 100*time.Millisecond, llm.BackOffDelay(1, errors.New("failed"), config))
	})

	t.Run("later retries double delay", func(t *testing.T) {
		require.Equal(t, 400*time.Millisecond, llm.BackOffDelay(3, errors.New("failed"), config))
	})
}

func TestFixedDelay(t *testing.T) {
	t.Parallel()

	t.Run("uses base delay", func(t *testing.T) {
		config := testDelayContext{delay: 250 * time.Millisecond}
		require.Equal(t, 250*time.Millisecond, llm.FixedDelay(4, errors.New("failed"), config))
	})
}

func TestRandomDelay(t *testing.T) {
	t.Parallel()

	t.Run("stays below maximum jitter", func(t *testing.T) {
		config := testDelayContext{maxJitter: time.Second}
		for range 100 {
			delay := llm.RandomDelay(1, errors.New("failed"), config)
			require.GreaterOrEqual(t, delay, time.Duration(0))
			require.Less(t, delay, time.Second)
		}
	})

	t.Run("zero jitter has no delay", func(t *testing.T) {
		require.Zero(t, llm.RandomDelay(1, errors.New("failed"), testDelayContext{}))
	})
}

func TestFullJitterBackoffDelay(t *testing.T) {
	t.Parallel()

	t.Run("stays below capped exponential ceiling", func(t *testing.T) {
		config := testDelayContext{delay: 100 * time.Millisecond, maxDelay: 250 * time.Millisecond}
		for range 100 {
			delay := llm.FullJitterBackoffDelay(3, errors.New("failed"), config)
			require.GreaterOrEqual(t, delay, time.Duration(0))
			require.Less(t, delay, 250*time.Millisecond)
		}
	})
}

func TestCombineDelay(t *testing.T) {
	t.Parallel()

	t.Run("adds strategies", func(t *testing.T) {
		combined := llm.CombineDelay(
			func(uint, error, llm.DelayContext) time.Duration { return 100 * time.Millisecond },
			func(uint, error, llm.DelayContext) time.Duration { return 250 * time.Millisecond },
		)

		require.Equal(t, 350*time.Millisecond, combined(1, errors.New("failed"), testDelayContext{}))
	})
}

type testDelayContext struct {
	delay     time.Duration
	maxDelay  time.Duration
	maxJitter time.Duration
}

func (c testDelayContext) Delay() time.Duration {
	return c.delay
}

func (c testDelayContext) MaxDelay() time.Duration {
	return c.maxDelay
}

func (c testDelayContext) MaxJitter() time.Duration {
	return c.maxJitter
}
