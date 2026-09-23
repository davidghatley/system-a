#!/usr/bin/env python3
"""CPU/offline diagnostics on saved dev probabilities only. Never opens test data."""
import json, math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'artifacts/release_i3';OUT.mkdir(parents=True,exist_ok=True)
L=['read','search','edit','execute','other_tool','respond_or_finish']; FLOOR=1e-12

def score(name,rows,groups,probs):
 y=np.array([L.index(r['gold_label']) for r in rows]); raw=np.asarray(probs,float); sums=raw.sum(1); normalized=raw/np.maximum(sums[:,None],1e-300); p=np.clip(normalized,1e-300,1); p/=p.sum(1,keepdims=True)
 pred=p.argmax(1); conf=p.max(1); good=pred==y; nll=-np.log(np.maximum(p[np.arange(len(y)),y],FLOOR)); br=((p-np.eye(6)[y])**2).sum(1)
 cls=[]
 for i,l in enumerate(L):
  sel=y==i; cls.append({'label':l,'support':int(sel.sum()),'errors':int((sel&~good).sum()),'nll_sum':float(nll[sel].sum()),'nll_contribution_to_mean':float(nll[sel].sum()/len(y)),'mean_nll':float(nll[sel].mean()) if sel.any() else None})
 mistakes=[]
 for j in np.flatnonzero(~good): mistakes.append({'record_id':rows[j].get('record_id'),'gold':L[y[j]],'prediction':L[pred[j]],'confidence':float(conf[j]),'gold_probability':float(p[j,y[j]]),'group':groups[j]})
 return {'name':name,'n':len(y),'accuracy':float(good.mean()),'macro_f1':float(np.mean([2*((pred==i)&(y==i)).sum()/max(1,((pred==i).sum()+(y==i).sum())) for i in range(6)])),'nll':float(nll.mean()),'nll_probability_floor':FLOOR,'brier':float(br.mean()),'correct_confidence_mean':float(conf[good].mean()) if good.any() else None,'incorrect_confidence_mean':float(conf[~good].mean()) if (~good).any() else None,'per_class':cls,'zero_probability_count_input':int((raw==0).sum()),'input_row_sum_min':float(sums.min()),'input_row_sum_max':float(sums.max()),'normalization_max_abs_deviation':float(np.abs(sums-1).max()),'confident_mistakes_top_20':sorted(mistakes,key=lambda x:-x['confidence'])[:20]},(y,p,pred,conf,good)

def temperature(p,y):
 # optimize log(T) over broad bounded range, with golden-section search, NLL objective
 lp=np.log(np.clip(p,1e-300,1)); lo,hi=math.log(.01),math.log(100.)
 def obj(z):
  q=lp/math.exp(z);q-=q.max(axis=1,keepdims=True);q=np.exp(q);q/=q.sum(axis=1,keepdims=True);return float(-np.log(np.maximum(q[np.arange(len(y)),y],FLOOR)).mean())
 gr=(math.sqrt(5)-1)/2;a=hi-gr*(hi-lo);b=lo+gr*(hi-lo)
 for _ in range(100):
  if obj(a)<obj(b): hi,b=b,a;a=hi-gr*(hi-lo)
  else: lo,a=a,b;b=lo+gr*(hi-lo)
 return math.exp((lo+hi)/2)
def scaled(p,t):
 z=np.log(np.clip(p,1e-300,1))/t;z-=z.max(1,keepdims=True);q=np.exp(z);return q/q.sum(1,keepdims=True)
def metrics(y,p):
 pr=p.argmax(1);return {'nll':float(-np.log(np.maximum(p[np.arange(len(y)),y],FLOOR)).mean()),'brier':float(((p-np.eye(6)[y])**2).sum(1).mean()),'accuracy':float((pr==y).mean()),'macro_f1':float(np.mean([2*((pr==i)&(y==i)).sum()/max(1,((pr==i).sum()+(y==i).sum())) for i in range(6)]))}
def grouped_cv(rows,y,p,groups):
 # deterministic greedy assignment of entire task groups to five folds balancing size; trajectory identity asserted intact
 traj={}; gidx={}
 for i,r in enumerate(rows):
  meta=r.get('metadata',{});g=meta.get('task_group') or meta.get('trajectory_id') or r.get('record_id'); tr=meta.get('trajectory_id');
  if tr and tr in traj and traj[tr]!=g: raise ValueError('trajectory split across task_group')
  if tr:traj[tr]=g
  gidx.setdefault(g,[]).append(i)
 folds=[[] for _ in range(min(5,len(gidx)))]; counts=[0]*len(folds)
 for g,inds in sorted(gidx.items(),key=lambda kv:(-len(kv[1]),str(kv[0]))):
  k=min(range(len(folds)),key=lambda z:(counts[z],z));folds[k].extend(inds);counts[k]+=len(inds)
 out=[]; calibrated=np.full_like(p,np.nan); raw_recovered=np.full_like(p,np.nan); seen=[]
 for k,held in enumerate(folds):
  train=np.array([i for f,ids in enumerate(folds) if f!=k for i in ids]); held=np.array(held)
  if not len(train) or not len(held):continue
  t=temperature(p[train],y[train]); fold_scaled=scaled(p[held],t)
  # Preserve original row alignment: metrics() always receives y and p in
  # the same record order, regardless of deterministic fold ordering.
  calibrated[held]=fold_scaled; raw_recovered[held]=p[held]; seen.extend(held.tolist())
  out.append({'fold':k,'train_rows':len(train),'held_rows':len(held),'train_groups':len({groups[i] for i in train}),'held_groups':len({groups[i] for i in held}),'temperature_fit_train_only':t,'heldout_raw':metrics(y[held],p[held]),'heldout_scaled':metrics(y[held],fold_scaled)})
 if sorted(seen)!=list(range(len(y))) or len(seen)!=len(set(seen)):
  raise AssertionError('held-out folds must partition every input row exactly once')
 raw_pooled=metrics(y,p)
 if raw_pooled!=metrics(y,raw_recovered): raise AssertionError('fold-reconstructed pooled raw metrics differ from direct metrics')
 if not np.array_equal(calibrated.argmax(1),p.argmax(1)):
  raise AssertionError('positive scalar temperature changed predicted argmax')
 return {'scheme':'deterministic greedy size-balanced 5-fold by task_group; groups indivisible; trajectories checked to map to one task_group','folds':out,'pooled_raw':raw_pooled,'pooled_scaled_foldwise':metrics(y,calibrated),'group_count':len(gidx),'trajectory_count':len(traj)}

def main():
 sources=[]
 for name,path in [('seed42','artifacts/experiment_i3/seed42_recovery.json'),('seed314159','artifacts/experiment_i3/seed314159_run.json')]:
  d=json.loads((ROOT/path).read_text());dev=d.get('dev') or d['reports'][0]['dev']; rows=dev['predictions']; sources.append((name,rows,lambda r: [r['probabilities'].get(l,0) for l in L]))
 rows=[json.loads(x) for x in (ROOT/'artifacts/baseline_i3/preflight/corrected_v3/dev_predictions.jsonl').read_text().splitlines()]
 adapted=[{'gold_label':r['gold'],'record_id':r['id'],'metadata':r.get('metadata',{}),'probabilities':r['methods']['tfidf_logistic']['probabilities']} for r in rows]
 sources.append(('tfidf_logistic',adapted,lambda r: [r['probabilities'].get(l,0) for l in L]))
 results=[];plot=[]
 for name,rs,get in sources:
  # baseline file has no group metadata in prediction records; recover only via same-ID dev artifact (dev-only)
  if name=='tfidf_logistic':
   lookup={r['record_id']:r for r in json.loads((ROOT/'artifacts/experiment_i3/seed42_recovery.json').read_text())['dev']['predictions']}
   for r in rs:r['metadata']=lookup[r['record_id']]['metadata']
  groups=[r.get('metadata',{}).get('task_group') or r.get('metadata',{}).get('trajectory_id') or r['record_id'] for r in rs]
  result,(y,p,pr,c,ok)=score(name,rs,groups,[get(r) for r in rs]); result['grouped_temperature_cv']=grouped_cv(rs,y,p,groups)
  result['full_dev_temperature_in_sample_exploratory']=temperature(p,y);result['full_dev_scaled_in_sample_exploratory']=metrics(y,scaled(p,result['full_dev_temperature_in_sample_exploratory']))
  results.append(result);plot.append((name,c,ok))
 (OUT/'metrics.json').write_text(json.dumps(results,indent=2)+'\n')
 # SVG reliability points for raw saved probabilities.
 colors=['#1769aa','#d1495b','#2a9d8f']; parts=['<svg xmlns="http://www.w3.org/2000/svg" width="640" height="540" viewBox="0 0 640 540"><rect width="640" height="540" fill="white"/><path d="M70 40V450H590M70 450L590 40" fill="none" stroke="#777"/><text x="200" y="515" font-family="sans-serif">Mean confidence</text><text x="10" y="25" font-family="sans-serif">Accuracy</text>']
 for j,(name,c,ok) in enumerate(plot):
  points=[]
  for lo in np.arange(0,1,.1):
   m=(c>=lo)&(c<lo+.1+1e-12)
   if m.any():points.append((c[m].mean(),ok[m].mean()))
  parts.append(f'<polyline points="{" ".join(f"{70+520*x:.1f},{450-410*z:.1f}" for x,z in points)}" fill="none" stroke="{colors[j]}" stroke-width="3"/>');parts.append(f'<text x="{80+j*175}" y="480" fill="{colors[j]}" font-family="sans-serif">{name}</text>')
 parts.append('</svg>');(OUT/'reliability.svg').write_text(''.join(parts))
 print('wrote metrics.json and reliability.svg')
if __name__=='__main__':main()
