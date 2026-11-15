"""Quantify the non-standard plots directly from saved client-update records."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def window_summary(records,first,last):
    selected=[r for r in records if first<=r['round']<=last]
    rounds=sorted(set(r['round'] for r in selected))
    assert rounds==list(range(first,last+1)), 'Missing diagnostic rounds'
    for round_id in rounds:
        rows=[r for r in selected if r['round']==round_id]
        assert len({r['client'] for r in rows})==len(rows)
        assert abs(sum(r['weight'] for r in rows)-1)<1e-10
    align=np.asarray([r['alignment'] for r in selected],dtype=float)
    norm=np.asarray([r['delta_norm'] for r in selected],dtype=float)
    weights=np.asarray([r['weight'] for r in selected],dtype=float)
    assert np.isfinite(align).all() and np.isfinite(norm).all()
    return {'first_round':first,'last_round':last,'client_updates':len(selected),
            'sample_weighted_mean_alignment':float((weights*align).sum()/len(rounds)),
            'negative_update_fraction':float((align<0).mean()),
            'sample_weighted_negative_fraction':float(weights[align<0].sum()/len(rounds)),
            'median_update_norm':float(np.median(norm)),
            'p90_update_norm':float(np.quantile(norm,.9)),
            'zero_norm_updates':int((norm==0).sum())}


def summarize(results):
    root=Path(results);runs={};csv_rows=[]
    for parent in ['course','course_controls']:
        for path in sorted((root/parent).glob('*/client_updates.jsonl')):
            records=[json.loads(line) for line in path.read_text().splitlines()]
            if not records:continue
            final=max(r['round'] for r in records);width=min(10,final)
            windows={'first_10':(1,width),'last_10':(final-width+1,final),'all':(1,final)}
            if final>=40:windows['first_40']=(1,40)
            key=f'{parent}/{path.parent.name}'
            runs[key]={name:window_summary(records,*bounds) for name,bounds in windows.items()}
            for name,metrics in runs[key].items():csv_rows.append({'run':key,'window':name}|metrics)
    output={'definitions':{
        'sample_weighted_mean_alignment':'Mean over rounds of sum_k(n_k/sum_j n_j)*cos(delta_k,delta_global).',
        'negative_update_fraction':'Unweighted fraction of selected client updates with recorded cosine < 0.',
        'sample_weighted_negative_fraction':'Mean over rounds of the selected-sample weight carried by negative-cosine updates.',
        'median_update_norm':'Median L2 norm of local parameter update vectors; no normalization by local steps.',
        'windows':'First rounds 1-10 and last ten rounds of each completed run; first_40 supports matched E stress.',
        'limits':'Descriptive, same-run summaries. Global update contains each local update; cosine depends on client count. Norms depend on local steps. No causality/convergence/privacy claim.'},
        'source':'Existing training JSONL only; no new training, test evaluation, or hyperparameter selection.',
        'runs':runs}
    (root/'nonstandard_diagnostics.json').write_text(json.dumps(output,indent=2)+'\n')
    with (root/'nonstandard_diagnostics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    print(f'Quantified {len(runs)} runs and {len(csv_rows)} diagnostic windows.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',default='results')
    summarize(p.parse_args().results)
