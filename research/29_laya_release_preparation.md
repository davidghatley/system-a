# Laya next-action release preparation

Date: 2026-09-23. Starting revision: `653dbef4ed6aa1e7d2d2b232d131a1d72ec96cff` with substantial pre-existing uncommitted Iteration 3 repairs and results. Work stays local and offline; public upload awaits review.

## Decision and bounded work plan

The original Iteration 3 test attempt has already been consumed. `artifacts/experiment_i3/preflight/runs/final_test.attempt.json` records a single attempt, and `artifacts/experiment_i3/final_test_run.json` reports completion. `PROJECT_STATE.md` and `artifacts/experiment_i3/FINAL_REVIEW.md` additionally record prior inspection and a non-hash-bound executed loader. Do not reopen test records, reset the marker, or describe any repeat as fresh confirmation. The historical result is exploratory and its latency value is invalid for the claimed boundary. No new holdout is silently substituted.

1. Audit the two existing checkpoints and saved dev predictions, reconcile the storage incident, and preserve reproducible CPU evidence.
2. Integrate and test the existing orchestration/evaluation repairs without executing the consumed final-test route.
3. Diagnose probability quality on train/dev, assess grouped cross-validated temperature scaling, and select the release behavior before packaging.
4. Prepare clean-directory inference weights, tokenizer, code, pinned dependencies, provenance, rights assessment, examples, and model card; verify parity and quickstart if local rights and runtime permit.
5. Assign an independent reviewer after integration; resolve findings and deliver the concrete package for approval before public upload.

One coordinator owns any GPU work. The new allowance is at most 7,200 GPU-worker seconds and 11.5 GiB peak reserved, with serialized workers and no retries after OOM/budget exhaustion. This allowance does not validate historical compute. Repository instructions limit further training to explicitly bounded smoke tests; existing checkpoints are the preferred release basis. Current free disk at start: 93 GiB on `/home` (`df -h .`); each recorded checkpoint has approximately 1.69 GB weights plus 3.37 GB training state. No additional training is currently justified.

## Finding-to-evidence checklist

| Finding / gate | Evidence and verification needed | Status at start |
|---|---|---|
| Held-out status | one-shot marker, final-result status, historical review; never open test rows | Consumed; verified from metadata |
| Storage discontinuity | storage-failure report versus checkpoint bytes, hashes, recovery command/log and revision | Audit pending |
| Dev discrimination and probabilities | recompute saved predictions, ID/label/ordering checks, grouped calibration diagnostics | Audit pending |
| Runner defects | composed tests for two-seed process boundary, atomic partial results, fallback, byte hash, timing and deadline | Existing repairs under review |
| Inference provenance and license | pinned model/tokenizer and dataset/code terms, self-contained package and local parity | Pending |
| Public release | independent review and approval of actual package | Pending |

## Completion update

The scoped audits and repairs are complete locally. Dev IDs/gold match the accepted
dev bytes; both checkpoints and TF-IDF metrics recompute. The storage incident
and recovered checkpoint are separate events, but the exact executed recovery
command/time and capacity-restoration step remain unknown. Prospective runner
repairs pass composed CPU tests, including a synthetic test-byte mismatch after
the mocked marker and a failed second seed after a durable first result.

The frozen local package (`artifacts/release_i3/RELEASE_FREEZE.md`) loads in an
independent repository-local directory; seven dev records match saved raw
probabilities exactly, and raw plus exploratory dev-temperature-scaled outputs
are available. The independent Luna reviewer found no remaining concrete
implementation defect after the full-distribution integrity gate and exact
fixed-question contract were integrated. CPU timing, rights evidence, group-CV
limitations, source hashes, and the 62-test integration result are in
`artifacts/release_i3/`. GPU-worker time spent is zero; no new training or test
access occurred. Public upload awaits owner review of publisher-declared terms,
embedded-content uncertainty, and a code-license decision.

Verified versus reported versus unknown values will be maintained in the task-specific audit reports. Historical `PREDECLARATION.md`, freeze and final marker remain unchanged. Any future independent evaluation requires a genuinely new, separately declared holdout and protocol; this task does not confer fresh held-out confirmation.

## Reproduction context

`git status --short --branch`; `git log -6 --oneline`; `df -h .`; `opencode auth list` (reported OpenAI OAuth stored); OpenCode catalog identified `openai/gpt-6-luna` active. No API-key or paid provider is used for delegation.
