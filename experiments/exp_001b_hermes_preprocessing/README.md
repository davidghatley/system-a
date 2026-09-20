# Experiment 001b: Hermes Preprocessing Prototype

Status: preprocessing only. Training and official evaluation are prohibited.

## Objective

Determine whether the reconciled Hermes Kimi six-tool subset can be converted into deterministic, leakage-resistant, Laya-compatible examples while preserving complete observed call/result state under 512 tokens.

## Inputs

- Dataset revision: `lambda/hermes-agent-reasoning-traces@b92885e4f0161d4b2536512710e004d4892cac6e`
- Local Kimi shard checksum: `d4a53d84935d0bffe7054b12c0591c34fe31fc96115602394b3f517b4631dd02`
- Laya tokenizer revision: `1c5edc17a7acd8701df6fc341c0d179f1c62c982`
- Reconciliation evidence: `research/18_hermes_reconciliation.md`

## Authorized Work

- Emit candidate rows, exclusion ledger, split manifests, checksums, and bias/support tables.
- Run structural reasoning-removal, target/metadata canary, linkage, token-budget, deduplication, and split-isolation assertions.
- Produce no model outputs.

## Stop Condition

Stop on any failed integrity assertion or if selection-bias/support analysis violates the frozen gates in `protocol.json`. Completion authorizes protocol review only, not training.

## Exact Rerun

Run from the repository root with the existing environment and local tokenizer cache:

```bash
.venv/bin/python experiments/exp_001b_hermes_preprocessing/scripts/build_dataset.py
```

The command performs preprocessing only. It loads the tokenizer but no model, performs no inference or training, and emits no sealed-test prediction path. Model-ready rows are compressed under ignored `data/exp_001b/`; their checksums are recorded in `results/checksums.json`.
