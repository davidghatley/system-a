# Experiment 001 Kimi Data Audit

Audit date: 2026-09-20

## Decision

**Stop before representation freeze or training.** The pinned release passes size, schema, target-index, and split-isolation checks, but it contains no actual tool-result messages. Observable execution history is therefore absent for every prefix with a prior tool call, violating the protocol's viability gate.

## Pin And Reproduction

- Dataset: `greghavens/kimi-k3-coding-and-debugging-traces`
- Commit: `33a874c3affbdb97e142752a9144e6624ef5bd07`
- Scope: all 50 active Parquet shards listed by the pinned `dataset-manifest.json`, totaling 3,503,519 bytes
- Local data: `data/exp_001_dataset/<commit>/`, ignored by Git
- Per-file SHA-256 values: `artifacts/exp_001_dataset_checksums.json`

Rerun:

```bash
source .venv/bin/activate
python scripts/preflight_exp001_data.py \
  --dataset greghavens/kimi-k3-coding-and-debugging-traces \
  --revision 33a874c3affbdb97e142752a9144e6624ef5bd07 \
  --output artifacts/exp_001_data_profile.json
```

Two complete runs produced byte-identical profile and checksum documents. Their SHA-256 values are `884e572aae43395a738003ac8f2db7dabe5bfe5030048ac53f03986b1754599d` and `e08f7a2abd3408e2816d05821d33c9df1cafd4617cdc912223936a6e6f9d258f`, respectively. The script requires a full 40-character commit, rejects any other dataset, confines all user-selectable paths to the repository, downloads only the pinned manifest and its safe relative active-shard paths, validates the required schema and integrity assertions, and recomputes all measurements. Commands and outcomes are retained in `artifacts/exp_001_data_commands.log`.

## Exact Findings

- 3,956 rows from 582 trajectories: train 3,651 rows/533 trajectories; val 305 rows/49 trajectories.
- All 3,956 target indexes are in bounds, point to an assistant message, and identify the final message.
- No train/val overlap exists by trajectory SHA-256, exact rendered-state hash, or normalized task text.
- Targets contain 4,201 calls: `bash` 2,456, `read` 947, `write` 479, and `edit` 319.
- 584 turns are response-only; no target is empty. Call multiplicity reaches eight calls in one target turn.
- 1,415 calls have empty arguments and five have visibly redacted arguments. There are 431 turns with repeated tool names and 103 with exact repeated name/argument calls.
- Prefixes contain 18,291 prior tool calls across 3,372 rows, but zero `role=tool` result messages, zero matched call/result pairs, and zero rows with an actual result observation. Historical roles are exactly 14,448 assistant, 5,347 user, and 1,391 system messages.
- Exact duplication: 470 excess rendered states in 470 duplicate groups and 38 excess complete trajectories in 38 duplicate groups. There are 438 normalized task texts; normalized tasks do not cross the source split boundary.
- Rendered-state length is 8,490 characters on average (p95 19,565; max 47,132). These are tokenizer-independent character counts, not model-token estimates.

## Renderer And Tool Families

The script implements only a candidate renderer, `exp001-observable-v1-candidate`. It structurally takes messages strictly before `target_message_index`, recursively removes every `reasoning_content` field, and asserts that the target assistant message and reasoning are excluded. It is explicitly marked `PROPOSED_NOT_FROZEN`.

Two maps are reported and neither is selected:

- `identity-minimal`: preserve `bash`, `read`, `write`, and `edit` as separate families.
- `lexical-coarse`: map `bash` to `shell` and `read`/`write`/`edit` to `file`.

The identity map is the minimal behavior-preserving candidate. Choosing either map is a protocol-freeze decision and must not occur silently in preprocessing.

## Manual Raw Checks

Manual checks were made against records loaded directly from the pinned Parquet files, without scoring validation outcomes or using them for tuning:

- `bash-photosort`, assistant step 1: the target has parallel `bash` and `read` calls, both with literal `{}` arguments.
- `java-lease-fencing`, val assistant step 1: the target retains a concrete `bash` command, confirming that arguments are sometimes present.
- `behavior-dependency-planning-0001`, assistant step 5: consecutive assistant tool-call messages appear with no intervening tool-result message, confirming the observation deficit is serialization rather than an ID-matching bug.
- `py-mvcc-store`, assistant step 3: prior `bash` and `read` calls are followed directly by another assistant call, again with no environment result.
- `vcf91-0105`, assistant step 7: later records retain large concrete call arguments and visibly redacted content, while the sequence still contains only user/assistant roles.

## Blocker

The advertised cumulative traces preserve actions but not the actual environment observations needed for `(state, action, observation)` control-policy distillation. Assistant prose sometimes summarizes presumed results, but it is generated text and cannot substitute for tool output. Since the observable-history ratio among rows with prior calls is exactly 0.0, the preregistered stop condition is met. Experiment 001 is not viable on this pinned release unless a separately pinned source artifact with authentic tool-result messages is obtained and re-audited.
