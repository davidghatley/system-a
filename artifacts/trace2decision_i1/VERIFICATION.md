# Trace2Decision Iteration 1 Final Verification

Date: 2026-09-21

## Status

**ACCEPTED.** The Trace2Decision acceptance criteria are independently
reproduced. No implementation files were modified; no training or Iteration 2
work was performed.

## Commands and results

All commands were run from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s pipelines/trace2decision -p 'test_*.py' -v
```

**VERIFIED:** 3/3 tests passed (`test_mapping`,
`test_target_is_not_rendered`, and `test_contract_and_group_split`).

```bash
sha256sum -c artifacts/trace2decision_i1/checksums.sha256
```

**VERIFIED:** all 12 checks passed, including converter, verifier, tests,
source files, output files, and prior verification evidence.

The verifier code path was executed with an in-memory output sink so that the
explicit instruction to write only this file was honored. It is equivalent to
the following verifier invocation, except that `--rerun-dir` and `--evidence`
were memory-backed rather than filesystem paths:

```bash
python3 pipelines/trace2decision/verify.py \
  --source artifacts/trace2decision_i1/source/traces.jsonl \
  --expected-dir artifacts/trace2decision_i1/output \
  --rerun-dir <in-memory> --evidence <in-memory>
```

**VERIFIED:** verifier result `pass=true`; expected and rerun validation each
had 1,544 rows and no failures; report equality was true; all four compared
file hashes were equal.

## Reproduced measurements

Source: dataset `11-47/glm-5.2-coding-and-debugging-traces`, revision
`1371ed38f8890d0520a53bc7ad850308eb4d7a22`, source SHA-256
`4e6b11f5b60976368f2a76c252808eb766af2ddcf5958d93f25160e45737d3aa`.

**VERIFIED:** 1,821 structurally valid source rows; 1,544 retained decisions;
207 trajectories and task groups; 277 rows excluded for unresolved prior call
IDs; 1,337 retained rows with prior tool results; 395 missing prior results in
the full trajectory accounting; 1,668 linked results; zero malformed rows,
orphan results, target-mutation failures, duplicate rendered states, or
conflicting duplicate labels.

**VERIFIED label distribution:** execute 670, read 289, edit 237, search 182,
respond_or_finish 166, other_tool 0.

**VERIFIED splits:** train 1,235 rows/164 trajectories, dev 159/20, test
150/23. Trajectory, task-group, and state-hash crossings were all zero;
normalized-task and exact complete-trajectory duplicate groups were zero.

**VERIFIED output SHA-256:**

| File | SHA-256 |
|---|---|
| train.jsonl | `d1441e6184d5ef843192f19ed10adaf6c2065fb4971ace478de50d4084240818` |
| dev.jsonl | `202338e20b5a02cc98938b13332da3484d34137c32a4ac5ecf8f20f72b297e8f` |
| test.jsonl | `616d97e562b04723e8566028cb1a9117868580c94680db31d1f270ac334aa5b4` |
| samples.jsonl | `7f4bf7528993c68dd34a3f68dccb9a5e7afed07a97a96d146f1122ad6de97af1` |

The in-memory converter rerun reproduced the complete report byte-for-byte
under canonical JSON comparison and reproduced each hash above.

## Direct pair inspection

**VERIFIED:** five deterministic sample pairs were inspected directly against
the source. Samples included steps 2, 6, 3, 5, and 3 with target calls
`bash/read`, `bash`, `bash`, `write`, and `write`; their emitted labels were
respectively execute, execute, execute, edit, and edit. Each inspected state
contained only the pre-target prefix and had linked prior tool results.

An independent source scan found 1,821 rows, 277 unresolved-prefix rows, zero
orphan results, and observed target names `bash`, `edit`, `find`, `grep`, `ls`,
`read`, and `write`. Their mappings agree with the documented ontology; the
first serialized call rule explains the parallel `bash/read` sample.

## Limitations

**INFERRED:** normalized initial task text is an appropriate split boundary for
this release, but it is not an authenticated repository-family boundary.

**UNKNOWN:** physical tool execution, teacher identity attestation, and rights
for every embedded task or repository snippet were not independently
replayed/authenticated. Labels remain observed behavioral-cloning actions, not
claims of optimality or downstream success. The filesystem CLI rerun was not
used because it would violate the instruction allowing writes only to this
verification file; the converter and verifier were instead rerun in memory,
with the existing checksum-backed `rerun_verification2.json` retained as
additional evidence.
