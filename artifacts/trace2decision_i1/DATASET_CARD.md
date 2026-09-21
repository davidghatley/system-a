# GLM-5.2 Trace2Decision Dataset Card

## Summary

This is a deterministic typed-decision derivative of
`11-47/glm-5.2-coding-and-debugging-traces`, pinned at
`1371ed38f8890d0520a53bc7ad850308eb4d7a22`. It contains 1,544 decisions from
207 coding/tool-use trajectories.

These labels are observed GLM-5.2 behavior for behavioral cloning. They are not
claims of optimal, uniquely correct, or causally successful action selection.

## Native Contract

Every JSONL row has exactly:

```json
{
  "state": "pre-target text",
  "questions": {"next_action": {"type": "choice", "instructions": "Choose the next observable agent action type.", "criteria": {"read": "...", "search": "...", "edit": "...", "execute": "...", "other_tool": "...", "respond_or_finish": "..."}}},
  "gold": {"next_action": {"type": "choice", "label": "execute", "probabilities": {"read": 0.0, "search": 0.0, "edit": 0.0, "execute": 1.0, "other_tool": 0.0, "respond_or_finish": 0.0}}}
}
```

Gold distributions are one-hot and validate with `shared.typed_decisions`.
One question and one action are emitted per
row. For a parallel tool-call target, the first serialized executable call is
the single observed next action. Raw bundle names remain in audit samples, not
in model input.

## Action Mapping

| Action | Deterministic rule |
|---|---|
| `read` | exact read/view/cat family names or `read_` prefix |
| `search` | ls/list/glob/grep/find/search families |
| `edit` | edit/write/patch/replace families |
| `execute` | bash/shell/terminal/exec/run/process families |
| `other_tool` | any structured call not covered above |
| `respond_or_finish` | assistant target with no tool calls |

All observed source tool names map to read/search/edit/execute in this release;
`other_tool` has zero observed positives but remains the deterministic catchall.

## State And Leakage Policy

State includes only the initial user task, bounded initial system constraints,
the canonical action-type list, and observable messages strictly before the
target. Prior assistant reasoning is never included. The source `tools_used`
field is excluded because it summarizes the complete trajectory and is future
derived. Source IDs, teacher metadata, source split, category, and target
arguments are also excluded from model input.

Target-turn exclusion is structural (`messages[:-1]`) and was checked by target
canary mutation for every row. Observed failures: zero. Rows with any unresolved
prior call ID are excluded.

## Truncation

- Initial task: 6,000 characters.
- Initial system constraints: 2,500 characters.
- Each rendered prior message: 4,000 characters.
- Recent history: 14,000 characters.
- Oversize fields retain deterministic head and tail around an explicit marker.
- History retains a contiguous newest complete-message window and reports the
  number of omitted older messages in the state.

Character limits are explicit and tokenizer-independent. State length is
minimum 283, median 11,628, p95 17,204, maximum 19,164 characters.

## Splits And Duplicates

Normalized initial task text is SHA-256 grouped. The split is derived from the
first 64 hash bits of `trace2decision-i1-20260921\n<group>` modulo 10,000:
80% train, 10% dev, 10% test.

| Split | Rows | Trajectories/task groups |
|---|---:|---:|
| train | 1,235 | 164 |
| dev | 159 | 20 |
| test | 150 | 23 |

Trajectory crossings, normalized-task-group crossings, rendered-state hash
crossings, normalized-task duplicate groups, exact complete-trajectory
duplicate groups, and exact rendered-state duplicates are all zero.

## Label Distribution

| Label | Rows |
|---|---:|
| execute | 670 |
| read | 289 |
| edit | 237 |
| search | 182 |
| respond_or_finish | 166 |
| other_tool | 0 |

## Files

- `output/train.jsonl`, `output/dev.jsonl`, `output/test.jsonl`: model records.
- `output/report.json`: distributions, filtering, splits, lengths, and hashes.
- `output/samples.jsonl`: 12 deterministic hash-selected audit examples with
  source identities outside the model record.
- `rerun/`: independently regenerated copy.
- `rerun_verification.json`: exact hash/statistics comparison.

## License And Limitations

The source repository declares CC-BY-4.0. Attribution and the pinned source
README are preserved. The source contains generated code tasks and repository
content; dataset-level metadata does not independently resolve every embedded
content right. Teacher identity and physical execution are publisher-level
claims corroborated by row structure, not independent replay or cryptographic
attestation. Coding-only behavior may not transfer to other agent domains.
