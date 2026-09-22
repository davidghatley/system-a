# Integrity repair 2

Date: 2026-09-22

## Outcome

The stale checksum/provenance evidence identified by `FINAL_VERIFICATION.md` was repaired without changing converter logic, verifier logic, tests, data semantics, or the independent final-verification record. The estimand remains completed tool-bearing target turns only: no-call targets and mixed-category targets are excluded.

The existing corrected outputs remain train/dev/test **1051/140/122** rows (1,313 retained total), with 508 exclusions. Final test records were not opened or summarized during this repair; the test artifact was checked only by line count and SHA-256.

## Verified facts

- `i3_tests.py` passed 7/7 under offline CPU settings.
- Two completed verifier runs returned `pass: true`, with valid expected and rerun datasets, equal reports, and equal hashes for all five generated files.
- The completed verifier runs left identical rerun/evidence hashes. One earlier invocation was terminated by the command runner at its 120-second limit; rerunning the same deterministic command with a 600-second limit completed successfully twice.
- Output and rerun hashes agree: train `86442271b3ddb74ce517ee335f6d89fbc80f94441ff1198250da5bf9ab4dda3c`, dev `ade8262625a46eef78dcb2b11b42eaced444e1604bab1af5423412a67ae92783`, test `3a22cfaa044baaf5cc40343fdf983d45ee80a065a2397266afacb157035c6fc8`, exclusions `0a26fff0a2ee13a6da0e5fe071fc21443a385bf92937e0c14e7b8462b097ef49`, and sample review `5c64a88bd132801faeabb9d4bd46569ad93fc01a58adf3793545a08c98afa7ee`.
- Output and rerun report hashes agree at `4530d90d9f53c88cc290bffff9194f3f4465146e65e1e20deab93e599379df21`.
- Verification evidence hash is `70ff4ca75d7e682fd2911b081b125b5d0753a914572327843a6a9a6d8db97378`.
- The refreshed manifest passes `sha256sum -c` for every listed file.

## Commands

```text
env PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=pipelines/trace2decision .venv/bin/python -B -m unittest pipelines/trace2decision/i3_tests.py -v
env PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=pipelines/trace2decision .venv/bin/python -B pipelines/trace2decision/i3_verify.py --source artifacts/trace2decision_i1/source/traces.jsonl --expected-dir artifacts/trace2decision_i3/output --rerun-dir artifacts/trace2decision_i3/rerun --evidence artifacts/trace2decision_i3/verification.json --v2-dir artifacts/trace2decision_i2/output
sha256sum artifacts/trace2decision_i3/output/{train,dev,test,exclusions,sample_review}.jsonl artifacts/trace2decision_i3/output/report.json artifacts/trace2decision_i3/rerun/{train,dev,test,exclusions,sample_review}.jsonl artifacts/trace2decision_i3/rerun/report.json artifacts/trace2decision_i3/verification.json pipelines/trace2decision/i3_{convert,verify,tests}.py
wc -l artifacts/trace2decision_i3/output/{train,dev,test,exclusions,sample_review}.jsonl
sha256sum -c artifacts/trace2decision_i3/checksums.sha256
```

The verifier command was completed twice after the timed-out attempt. Environment flags prohibited Hub access and CUDA visibility; execution used four CPU threads.

## Limitations

- This repair resolves artifact integrity only. It does not expand claims about semantic command/family leakage, semantic intent, or qualitative adjudication.
- The Transformers tokenizer emitted warnings for pre-truncation sequences longer than its nominal model maximum; the established protected-state construction and final 512-token checks still passed.
- `FINAL_VERIFICATION.md` remains unchanged and records the pre-repair independent decision. Independent acceptance must re-check the refreshed manifest and evidence.

## Changed files

- `artifacts/trace2decision_i3/checksums.sha256`
- `artifacts/trace2decision_i3/provenance.json`
- `artifacts/trace2decision_i3/commands.log`
- `artifacts/trace2decision_i3/verification.json` (deterministically regenerated; content hash unchanged)
- `artifacts/trace2decision_i3/rerun/` generated files (deterministically regenerated; content hashes unchanged)
- `artifacts/trace2decision_i3/REPAIR_2.md`

No pipeline source or corrected output file content changed.
