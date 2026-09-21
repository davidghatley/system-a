# Subagent Model Capability Probe

Date: 2026-09-21

## Purpose

Test whether OpenCode can assign the `general` task subagent to smaller models before resuming System-A Iteration 1 delegation.

## Commands

The tests used an ephemeral `OPENCODE_CONFIG_CONTENT` override. No persistent OpenCode configuration was changed.

```bash
OPENCODE_CONFIG_CONTENT='{"agent":{"general":{"model":"openai/gpt-5.6-luna"}}}' \
  opencode run --model openai/gpt-5.6-luna --agent build --format json \
  "Use the task tool exactly once to call the general subagent. Tell it: capability probe only, do not use tools or modify files, return exactly LUNA_TRUE_SUBAGENT_OK. After it returns, output only its returned token."
```

```bash
OPENCODE_CONFIG_CONTENT='{"agent":{"general":{"model":"opencode/nemotron-3-ultra-free"}}}' \
  opencode run --model openai/gpt-5.6-luna --agent build --format json \
  "Use the task tool exactly once to call the general subagent. Tell it: capability probe only, do not use tools or modify files, return exactly NEMOTRON_TRUE_SUBAGENT_OK. After it returns, output only its returned token."
```

## Results

| Requested child model | Task metadata child model | Result | Approximate task duration | Reported cost |
|---|---|---|---:|---:|
| `openai/gpt-5.6-luna` | `openai/gpt-5.6-luna` | `LUNA_TRUE_SUBAGENT_OK` | 2.0 s | $0 |
| `opencode/nemotron-3-ultra-free` | `opencode/nemotron-3-ultra-free` | `NEMOTRON_TRUE_SUBAGENT_OK` | 40.2 s | $0 |

## Interpretation

- **VERIFIED:** Both requested models can execute a true OpenCode `general` task subagent in the current authentication context.
- **VERIFIED:** Per-agent model selection works through OpenCode configuration.
- **VERIFIED:** The current parent `task` tool schema exposes `subagent_type` but no per-call `model` field.
- **INFERRED:** Persistent named agents are the cleanest way to select models from the parent session; direct per-call selection is unavailable here.
- **UNKNOWN:** The one-token probe does not establish either model's quality on long code review, dataset audit, or GPU-training tasks.
