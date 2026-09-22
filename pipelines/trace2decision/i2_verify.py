#!/usr/bin/env python3
"""Independent v2 contract, Laya-input, leakage, and deterministic-rerun verifier."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from i2_convert import LayaQ, MAX_LEN, HEAD_MAX_LEN, AutoTokenizer, TOKENIZER_DEFAULT, build_sequence, file_sha, convert
from shared.typed_decisions import validate_record, ValidationError

def validate(directory: Path, tokenizer_path: Path) -> dict:
    report = json.loads((directory / "report.json").read_text())
    tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    failures=[]; ids=set(); trajectories={}; groups={}; rows=0; lengths=[]; at768=[]; at1024=[]
    for split in ("train","dev","test"):
        path=directory/f"{split}.jsonl"; count=0
        if file_sha(path) != report["files"][path.name]["sha256"]: failures.append(f"{path.name}: hash mismatch")
        with path.open(encoding="utf-8") as f:
            for line, text in enumerate(f,1):
                count+=1; rows+=1; raw=json.loads(text); meta=raw.get("metadata",{})
                try: validate_record(raw)
                except ValidationError as e: failures.append(f"{path.name}:{line}: shared validator: {e}")
                required={"id","trajectory_id","task_group","source_step"}
                if not required <= set(meta): failures.append(f"{path.name}:{line}: metadata identifiers missing")
                if meta.get("id") in ids: failures.append(f"{path.name}:{line}: duplicate stable id")
                ids.add(meta.get("id")); trajectories.setdefault(meta.get("trajectory_id"),set()).add(split); groups.setdefault(meta.get("task_group"),set()).add(split)
                if any(str(meta.get(k,"")) in raw.get("state","") for k in ("id","trajectory_id","task_group")): failures.append(f"{path.name}:{line}: identifier leaked into state")
                if "TARGET_CANARY_9a31" in raw.get("state",""): failures.append(f"{path.name}:{line}: target canary leaked")
                state=raw["state"]; empty, markers=build_sequence(tok,"",LayaQ,MAX_LEN,HEAD_MAX_LEN); room=MAX_LEN-len(empty); full=tok(state.replace(tok.mask_token," "),add_special_tokens=False,truncation=False)["input_ids"]; seq, got=build_sequence(tok,state,LayaQ,MAX_LEN,HEAD_MAX_LEN)
                if len(full)>room or seq != empty[:-1]+full+[tok.sep_token_id] or got != markers or len(seq)>MAX_LEN: failures.append(f"{path.name}:{line}: Laya 512 construction")
                lengths.append(len(seq)); at768.append(len(build_sequence(tok,state,LayaQ,768,HEAD_MAX_LEN)[0])); at1024.append(len(build_sequence(tok,state,LayaQ,1024,HEAD_MAX_LEN)[0]))
        if count != report["files"][path.name]["rows"]: failures.append(f"{path.name}: row count mismatch")
    if rows != report["conversion"]["retained_rows"]: failures.append("total row count mismatch")
    if any(len(v)>1 for v in trajectories.values()): failures.append("trajectory crosses split")
    if any(len(v)>1 for v in groups.values()): failures.append("task group crosses split")
    if rows < 1544: failures.append("fewer than 1544 retained rows")
    return {"rows":rows,"stable_ids":len(ids),"length_512":{"min":min(lengths),"max":max(lengths),"mean":round(sum(lengths)/len(lengths),2)},"length_768":{"min":min(at768),"max":max(at768),"mean":round(sum(at768)/len(at768),2)},"length_1024":{"min":min(at1024),"max":max(at1024),"mean":round(sum(at1024)/len(at1024),2)},"failures":failures,"valid":not failures}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--source",type=Path,required=True); p.add_argument("--expected-dir",type=Path,required=True); p.add_argument("--rerun-dir",type=Path,required=True); p.add_argument("--evidence",type=Path,required=True); p.add_argument("--tokenizer",type=Path,default=TOKENIZER_DEFAULT); a=p.parse_args()
    expected=json.loads((a.expected_dir/"report.json").read_text()); rerun=convert(a.source,a.rerun_dir,a.tokenizer); ev={"expected":validate(a.expected_dir,a.tokenizer),"rerun":validate(a.rerun_dir,a.tokenizer),"report_equal":expected==rerun,"file_comparison":{n:{"expected":expected["files"][n]["sha256"],"rerun":rerun["files"][n]["sha256"],"equal":expected["files"][n]["sha256"]==rerun["files"][n]["sha256"]} for n in ("train.jsonl","dev.jsonl","test.jsonl","samples.jsonl")}}
    ev["pass"]=ev["expected"]["valid"] and ev["rerun"]["valid"] and ev["report_equal"] and all(x["equal"] for x in ev["file_comparison"].values()); a.evidence.write_text(json.dumps(ev,indent=2,sort_keys=True)+"\n")
    print(json.dumps(ev,sort_keys=True)); raise SystemExit(0 if ev["pass"] else 1)
if __name__=="__main__": main()
