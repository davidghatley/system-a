# Trace2Decision v3 final verification

Date: 2026-09-22. Independent CPU/offline review after `REPAIR_2.md`. Only this report was written. `i3_verify.py` was not rerun because it rewrites protected rerun/evidence files; its existing evidence was checked non-mutatingly via the manifest and provenance.

## Verified

- `sha256sum -c artifacts/trace2decision_i3/checksums.sha256`: exit 0; all 25 entries pass, including source/readme, v2 inputs, implementation, outputs, rerun, evidence, provenance, command log, repair report, and prior report.
- Unit command with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`: exit 0; `Ran 7 tests ... OK`.
- `wc -l ...`: source **1,821**; train/dev/test **1,051/140/122**; exclusions **508**; sample ledger **8**. Thus 1,313 retained plus 508 excluded equals 1,821.
- Independent ledger command: exit 0; exclusions are mixed **76**, no-call **208**, unresolved-prefix **224**; split ledgers match `report.json`; provenance rows match; `verification.json` pass/equality checks are true; v2 assignment changes are `[]`.
- `sha256sum output/test.jsonl rerun/test.jsonl && cmp -s output/report.json rerun/report.json`: exit 0. Both test hashes are `3a22cfaa044baaf5cc40343fdf983d45ee80a065a2397266afacb157035c6fc8`; reports are byte-equal. Final-test records were not opened.
- Provenance is consistent with current bytes: source SHA-256 `4e6b11f5b60976368f2a76c252808eb766af2ddcf5958d93f25160e45737d3aa`, verification SHA-256 `70ff4ca75d7e682fd2911b081b125b5d0753a914572327843a6a9a6d8db97378`, starting commit and HEAD `2d4408de85587a046c3f14fec1835dd9e69940c9`, Laya `d113dca2512fb3eaca313534bc54c7162d87c1d4`, tokenizer snapshot present.
- Authenticated prior structural results: 1,313 stable IDs; 206 exact task groups/trajectories split 163/20/23; zero exact or cross-split rendered-state duplicates; Laya lengths 187-512, mean 473.6.

## Inferred / Unknown

- Hub revision `1371ed38f8890d0520a53bc7ad850308eb4d7a22` was not queried offline; local source identity is established by the verified hash and README hash.
- Categories are observed tool-interface labels, not intent. Sample review is structural, not semantic adjudication.
- Exact normalized grouping passes, but one bounded lexical cross-split pair remains (source lines 1700/1707, Jaccard 0.8794). Semantic command leakage and repository/recipe-family independence are **UNKNOWN**.
- All 208 no-call targets are conservatively excluded for lacking a positive completion marker. Completion semantics are **UNKNOWN**; acceptance is limited to completed tool-bearing turns with structurally complete prefixes and one mapped category.

## Exact commands and results

```text
env PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=pipelines/trace2decision .venv/bin/python -B -m unittest pipelines/trace2decision/i3_tests.py -v
Result: exit 0; Ran 7 tests in 0.260s; OK.

sha256sum -c artifacts/trace2decision_i3/checksums.sha256
Result: exit 0; all 25 listed paths reported OK.

wc -l artifacts/trace2decision_i1/source/traces.jsonl artifacts/trace2decision_i3/output/{train,dev,test,exclusions,sample_review}.jsonl
Result: 1821, 1051, 140, 122, 508, and 8 lines respectively.

sha256sum artifacts/trace2decision_i3/output/test.jsonl artifacts/trace2decision_i3/rerun/test.jsonl && cmp -s artifacts/trace2decision_i3/output/report.json artifacts/trace2decision_i3/rerun/report.json
Result: exit 0; both test hashes were 3a22cfaa044baaf5cc40343fdf983d45ee80a065a2397266afacb157035c6fc8; reports were byte-equal.

env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -c 'import collections,json,pathlib; b=pathlib.Path("artifacts/trace2decision_i3"); r=json.loads((b/"output/report.json").read_text()); p=json.loads((b/"provenance.json").read_text()); v=json.loads((b/"verification.json").read_text()); e=[json.loads(x) for x in (b/"output/exclusions.jsonl").read_text().splitlines()]; c=dict(sorted(collections.Counter(x["reason"] for x in e).items())); s={k:dict(sorted(collections.Counter(x["reason"] for x in e if x["split"]==k).items())) for k in ("train","dev","test")}; q={"rows":len(e),"reasons":c,"splits":s,"balance":r["conversion"]["retained_rows"]+len(e)==r["conversion"]["source_rows"],"ledger_match":c==r["conversion"]["exclusion_reasons"] and s==r["conversion"]["exclusions_by_split_reason"],"provenance_rows_match":p["corrected_v3"]["rows"]=={"train":1051,"dev":140,"test":122,"retained_total":1313,"excluded":508},"verification_pass":v["pass"] and v["expected"]["valid"] and v["rerun"]["valid"] and v["report_equal"] and all(x["equal"] for x in v["file_comparison"].values()),"v2_changes":r["split"]["v2_assignment_changes"],"near_pairs":len(r["near_duplicate_audit"]["flagged_pairs"]),"semantic_leakage":r["near_duplicate_audit"]["semantic_leakage"]}; assert all((q["balance"],q["ledger_match"],q["provenance_rows_match"],q["verification_pass"])); print(json.dumps(q,sort_keys=True))'
Result: exit 0; rows=508; reasons={mixed_target_categories:76, no_call_without_positive_completion:208, unresolved_prefix_tool_call:224}; balance=true; ledger_match=true; provenance_rows_match=true; verification_pass=true; v2_changes=[]; near_pairs=1; semantic_leakage=UNKNOWN.
```

ACCEPTED_FOR_TRAINING
