# Independent review 1 of the four-finding repair

Date: 2026-09-23. Reviewer: fresh `luna-worker` session using the verified OpenAI `openai/gpt-6-luna` model before the owner changed the requested orchestration model. The reviewer did not implement any reviewed changes. It did not open a historical held-out test file/result or execute a model/GPU workload.

## Disposition

**Not ready to accept all four repairs.** The reviewer found P1 gaps in private namespace reuse, isolated-suite evidence, pre-compute immutable admission, and truthful aggregate-budget semantics.

## Verified findings

- `release/i3/inference.py` used `sys.modules.get` after `setdefault` for a deterministic private namespace. A preexisting unrelated module at that exact private name could be returned without source-origin validation.
- The evidence report still said the full isolated suite was pending, even though the parent later ran it twice. The report needed a final coherent rewrite.
- Pilot decisions, result files, and freezes were create-once, but pilot/seed compute admission happened before the exclusive output was claimed. Directory-entry durability and recovery-route marker coverage also needed work.
- The aggregate ledger durably charged unresolved reservations and recorded overruns, but a final partial grant could start a worker with less than its requested allowance. A single blocking operation could overshoot its deadline. This is conservative pre-admission accounting, not a mathematically hard wall-clock cap.

## Required follow-up

1. Fail closed on preexisting private modules unless their origin and manifest digest are verified; hash-check relative bundle imports during execution.
2. Rebuild and rerun the integrated suite from a commit-derived isolated source tree that deliberately omits the historical test JSONL, run directories, and model weights; retain the input inventory and both logs.
3. Claim pilot/seed attempts before entering a worker, block automatic repeat after unresolved admission, fsync decision/freeze directory entries, and apply historical-root/consumed-marker rules without treating a separate future root as the consumed experiment.
4. Reject a worker when the remaining aggregate allowance is smaller than its full request; document bounded overshoot from already-running blocking operations. Bind identity to the actual supplied config path and test the real runner composition.

## Reviewer checks

The reviewer independently ran the then-current CPU suite once in the working tree: 57 tests, `OK`. It explicitly did not claim isolated twice-run evidence, actual GPU timing, historical checkpoint reconstruction, or fresh held-out evaluation. Final evidence must record which of these follow-ups were subsequently verified.
