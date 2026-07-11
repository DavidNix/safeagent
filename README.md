# SafeAgent

## Status

> **Warning: Nothing in this repository is stable at this time.** APIs, file layouts, and behavior may change or break without notice. Do not depend on it for anything.

SafeAgent builds agents and agentic workflows with a security-first mindset.

## The Lethal Trifecta

You are **not** safe from prompt injection if you allow an agent to combine all three:

1. **Private data** — secrets, credentials, personal or privileged information
2. **Untrusted content** — emails, web pages, documents, tool output, anything an attacker can influence
3. **External communication** — outbound calls to the internet or other systems

Any two are a risk; all three together is the lethal trifecta. An attacker who can plant instructions in untrusted content can exfiltrate private data through external communication.

SafeAgent exists to minimize that triangle by design, not by hope.

## Submodules

- `go/` - Native Go code
- `openclaw` - https://github.com/openclaw/openclaw
- `claude-code` - https://github.com/DavidNix/claude-code
- `pi-mono` - https://github.com/badlogic/pi-mono
- `hermes-agent` - https://github.com/nousresearch/hermes-agent
- `zeroclaw` - https://github.com/zeroclaw-labs/zeroclaw
- `opencode` - https://github.com/anomalyco/opencode
- `openai-agents-js` - https://github.com/openai/openai-agents-js

## Quick Start

```bash
git clone --recursive https://github.com/DavidNix/safeagent
make vet    # Run all static checks
make test   # Run all tests
```
