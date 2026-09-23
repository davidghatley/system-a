"""Targeted alignment regression test; no external data/model access."""
import numpy as np
from i3_dev_diagnostics import grouped_cv, metrics

def test_foldwise_calibration_restores_original_row_order():
    # Interleaved group IDs force fold assignment order to differ from source
    # row order. Distinct rows make an accidental concatenation observable.
    rows=[]; groups=[]; y=[]; p=[]
    for i in range(15):
        g=f'g{(i * 7) % 15:02d}'
        groups.append(g); rows.append({'record_id':str(i),'metadata':{'task_group':g}})
        y.append(i % 3)
        q=np.array([.72,.18,.10]) if i%3==0 else (np.array([.12,.70,.18]) if i%3==1 else np.array([.15,.20,.65]))
        # Expand to fixed six labels.
        p.append(np.r_[q, [1e-8,1e-8,1e-8]])
    y=np.asarray(y);p=np.asarray(p); result=grouped_cv(rows,y,p,groups)
    assert result['pooled_raw']==metrics(y,p)
    assert result['pooled_scaled_foldwise']['accuracy']==result['pooled_raw']['accuracy']
    assert result['pooled_scaled_foldwise']['macro_f1']==result['pooled_raw']['macro_f1']
    # Reconstruct the expected heldout-scaled metric in original source order.
    reconstructed=np.empty_like(p)
    # Each group maps to one fold; reproduce fold temperatures and place values
    # by source indices (the behavior under test).
    from i3_dev_diagnostics import temperature, scaled
    # Derive held groups from fold index using the same deterministic assignment.
    unique=sorted(set(groups),key=lambda g:(-groups.count(g),str(g)))
    bins=[[] for _ in range(5)];counts=[0]*5
    for g in unique:
        k=min(range(5),key=lambda z:(counts[z],z)); bins[k].append(g);counts[k]+=groups.count(g)
    for k,gs in enumerate(bins):
        held=np.array([i for i,g in enumerate(groups) if g in gs]);train=np.array([i for i,g in enumerate(groups) if g not in gs])
        reconstructed[held]=scaled(p[held],temperature(p[train],y[train]))
    assert result['pooled_scaled_foldwise']==metrics(y,reconstructed)

if __name__ == '__main__':
    test_foldwise_calibration_restores_original_row_order()
    print('alignment regression test passed')
