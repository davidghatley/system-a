# Trace2Decision v2 dataset card

## Scope

This is a deterministic v2 derivative of the pinned local source
`11-47/glm-5.2-coding-and-debugging-traces` at revision
`1371ed38f8890d0520a53bc7ad850308eb4d7a22`. It preserves the v1 linkage
filter, labels, task-group split seed, and 1,544 retained decisions. It is a
behavioral-cloning dataset of observed next actions, not an optimality or task
success benchmark.

## Format

Each JSONL row has native `state`, `questions`, and `gold`, plus optional
`metadata`. `id`, `trajectory_id`, `task_group`, and `source_step` are metadata
and are not embedded in `state`. The shared typed-decision validator accepts
the rows.

## Context policy

The state reserves TASK, SYSTEM CONSTRAINTS, AVAILABLE ACTIONS, and the latest
relevant observation. Remaining room is filled greedily with newest complete
history messages. No generated summaries are used. Older optional history is
the only message class intentionally omitted; the report records its count and
source indexes per row. A bounded head/tail marker can occur within protected
text when necessary to make the actual pinned Laya input fit.

## Pins and limitations

The actual Laya source is local commit `d113dca2512fb3eaca313534bc54c7162d87c1d4`;
the tokenizer snapshot is `1c5edc17a7acd8701df6fc341c0d179f1c62c982`.
The source is machine-generated and publisher-attributed. Physical execution,
teacher identity, rights of embedded content, and downstream learnability are
not independently authenticated here. No training was run.

## Repair-cycle-1 audit

The side-effect-free pre-policy audit measures the full bounded source
candidate before 512 selection: 187/4244.48/16656 tokens (min/mean/max), with
97.41% over 512, 90.8% over 768, and 83.23% over 1024. Selected history is
restored to chronological source order before rendering. Full pre/post
distributions are in `output/report.json`.
