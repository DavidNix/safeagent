# SafeAgent Monorepo

## Project Overview

SafeAgent is a monorepo with language-specific workspaces. Go code lives in `go/` as module `github.com/DavidNix/safeagent`. Other languages to be added later.

## Quick Start

```bash
make vet          # Run all static checks
make test         # Run all tests
```

Each language sufolder has their own Makefile.

## Hard Constraints

- Bug fixes require TDD RED/GREEN: write a failing test first, confirm it fails, then write the fix and confirm tests pass.
- Feature work with new behavior requires tests and a verification pass that proves the tests fail without the implementation.
- If a subtree has its own `AGENTS.md`, read and follow the closest applicable file before editing there.

## Structure

- `go/` - Go module rooted at `github.com/DavidNix/safeagent`.
