package llm

import (
	"context"
	"errors"
	"math"
	"math/rand/v2"
	"net/http"
	"time"
)

const (
	// DefaultRetryDelay is the base delay used when retries are enabled.
	DefaultRetryDelay = 100 * time.Millisecond
	// DefaultRetryMaxJitter is the maximum random delay added by the default strategy.
	DefaultRetryMaxJitter = 100 * time.Millisecond
)

// RetryConfig configures optional retries for one client. Attempts counts the
// initial request; values below two disable retries. A zero Delay or MaxJitter
// uses its package default, while a negative value disables it. A non-positive
// MaxDelay leaves delays uncapped.
type RetryConfig struct {
	Attempts  uint
	Delay     time.Duration
	MaxDelay  time.Duration
	MaxJitter time.Duration
	DelayType DelayTypeFunc
}

// DelayContext exposes retry settings to a delay strategy.
type DelayContext interface {
	Delay() time.Duration
	MaxDelay() time.Duration
	MaxJitter() time.Duration
}

// DelayTypeFunc computes the wait before the next request. Attempt starts at
// one after the initial request fails.
type DelayTypeFunc func(attempt uint, err error, config DelayContext) time.Duration

// BackOffDelay returns an exponentially increasing delay.
func BackOffDelay(attempt uint, _ error, config DelayContext) time.Duration {
	if attempt == 0 {
		attempt = 1
	}
	return exponentialDelay(config.Delay(), attempt-1)
}

// FixedDelay returns the configured base delay.
func FixedDelay(_ uint, _ error, config DelayContext) time.Duration {
	return max(config.Delay(), 0)
}

// RandomDelay returns a random delay below the configured maximum jitter.
func RandomDelay(_ uint, _ error, config DelayContext) time.Duration {
	maxJitter := config.MaxJitter()
	if maxJitter <= 0 {
		return 0
	}
	return time.Duration(rand.Int64N(int64(maxJitter)))
}

// FullJitterBackoffDelay returns a random delay below an exponential ceiling.
func FullJitterBackoffDelay(attempt uint, _ error, config DelayContext) time.Duration {
	ceiling := exponentialDelay(config.Delay(), attempt)
	if config.MaxDelay() > 0 {
		ceiling = min(ceiling, config.MaxDelay())
	}
	if ceiling <= 0 {
		return 0
	}
	return time.Duration(rand.Int64N(int64(ceiling)))
}

// CombineDelay adds the results of multiple delay strategies.
func CombineDelay(delayTypes ...DelayTypeFunc) DelayTypeFunc {
	return func(attempt uint, err error, config DelayContext) time.Duration {
		var total time.Duration
		for _, delayType := range delayTypes {
			if delayType == nil {
				continue
			}
			delay := max(delayType(attempt, err, config), 0)
			if delay > time.Duration(math.MaxInt64)-total {
				return time.Duration(math.MaxInt64)
			}
			total += delay
		}
		return total
	}
}

// WithRetry configures optional retries for a client.
func WithRetry(config RetryConfig) Option {
	return func(c *Client) { c.retry = newRetryConfig(config) }
}

type retryConfig struct {
	attempts  uint
	delay     time.Duration
	maxDelay  time.Duration
	maxJitter time.Duration
	delayType DelayTypeFunc
}

func newRetryConfig(config RetryConfig) *retryConfig {
	if config.Attempts < 2 {
		return nil
	}
	delay := config.Delay
	if delay == 0 {
		delay = DefaultRetryDelay
	}
	maxJitter := config.MaxJitter
	if maxJitter == 0 {
		maxJitter = DefaultRetryMaxJitter
	}
	delayType := config.DelayType
	if delayType == nil {
		delayType = CombineDelay(BackOffDelay, RandomDelay)
	}
	return &retryConfig{
		attempts:  config.Attempts,
		delay:     max(delay, 0),
		maxDelay:  max(config.MaxDelay, 0),
		maxJitter: max(maxJitter, 0),
		delayType: delayType,
	}
}

func (c *retryConfig) Delay() time.Duration {
	return c.delay
}

func (c *retryConfig) MaxDelay() time.Duration {
	return c.maxDelay
}

func (c *retryConfig) MaxJitter() time.Duration {
	return c.maxJitter
}

func (c *retryConfig) retryDelay(attempt uint, err error) time.Duration {
	delay := max(c.delayType(attempt, err, c), 0)
	if c.maxDelay > 0 {
		delay = min(delay, c.maxDelay)
	}
	return delay
}

func doWithRetry[T any](ctx context.Context, config *retryConfig, call func() (T, error)) (T, error) {
	var zero T
	if err := ctx.Err(); err != nil {
		return zero, err
	}
	attempts := uint(1)
	if config != nil {
		attempts = config.attempts
	}
	for attempt := uint(1); attempt <= attempts; attempt++ {
		value, err := call()
		if err == nil {
			return value, nil
		}
		if ctxErr := ctx.Err(); ctxErr != nil {
			return zero, ctxErr
		}
		if config == nil || attempt == attempts || !isTransientError(err) {
			return zero, unpackTransientError(err)
		}
		if err := waitForRetry(ctx, config.retryDelay(attempt, unpackTransientError(err))); err != nil {
			return zero, err
		}
	}
	return zero, nil
}

func waitForRetry(ctx context.Context, delay time.Duration) error {
	if delay <= 0 {
		return ctx.Err()
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

type transientError struct {
	err error
}

func (e *transientError) Error() string {
	return e.err.Error()
}

func (e *transientError) Unwrap() error {
	return e.err
}

func transient(err error) error {
	return &transientError{err: err}
}

func unpackTransientError(err error) error {
	if transientErr, ok := errors.AsType[*transientError](err); ok {
		return transientErr.err
	}
	return err
}

func isTransientError(err error) bool {
	if _, ok := errors.AsType[*transientError](err); ok {
		return true
	}
	if statusErr, ok := errors.AsType[*StatusError](err); ok {
		return isTransientStatus(statusErr.StatusCode)
	}
	if statusErr, ok := errors.AsType[*modelsStatusError](err); ok {
		return isTransientStatus(statusErr.StatusCode)
	}
	return false
}

func isTransientStatus(status int) bool {
	return status == http.StatusRequestTimeout ||
		status == http.StatusTooManyRequests ||
		status >= http.StatusInternalServerError
}

func exponentialDelay(base time.Duration, exponent uint) time.Duration {
	if base <= 0 {
		return 0
	}
	if exponent >= 63 || base > time.Duration(math.MaxInt64)>>exponent {
		return time.Duration(math.MaxInt64)
	}
	return base << exponent
}
