"""Summarize the required offline helper output without changing source results."""
from pathlib import Path
import json, csv
import pandas as pd

out=Path(__file__).resolve().parents[1]
trials=pd.read_csv(out/'offline_clearance/clearance_trials.csv')
summary=pd.read_csv(out/'offline_clearance/clearance_summary.csv')
manifest=json.loads((out/'offline_clearance/analysis_manifest.json').read_text())
valid=trials[(trials.included_in_primary_pool==1)&(trials.available==1)].copy()
valid['has_zero']=(valid.zero_clearance_samples>0)
valid['has_negative']=(valid.negative_boundary_samples>0)
valid['has_nonpositive']=valid.has_zero|valid.has_negative
valid['nonpositive_samples']=valid.zero_clearance_samples+valid.negative_boundary_samples
records=[]
keys=['suite_root','experiment','width_m','nominal_don','optimizer_type']
for key,group in valid.groupby(keys,dropna=False):
 row=dict(zip(keys,key)); n=int(group.pose_samples.sum())
 row.update(available_completed_trials=len(group),pooled_pose_samples=n,zero_clearance_samples=int(group.zero_clearance_samples.sum()),negative_boundary_samples=int(group.negative_boundary_samples.sum()),nonpositive_clearance_samples=int(group.nonpositive_samples.sum()),trials_with_zero=int(group.has_zero.sum()),trials_with_negative=int(group.has_negative.sum()),trials_with_nonpositive=int(group.has_nonpositive.sum()),success_trials_with_nonpositive=int((group.has_nonpositive&(group.status=='success')).sum()),success_trials=int((group.status=='success').sum()))
 row.update(zero_sample_rate=row['zero_clearance_samples']/n if n else None,negative_sample_rate=row['negative_boundary_samples']/n if n else None,nonpositive_sample_rate=row['nonpositive_clearance_samples']/n if n else None)
 records.append(row)
pd.DataFrame(records).to_csv(out/'offline_clearance/contact_summary.csv',index=False)
valid[valid.has_nonpositive].to_csv(out/'offline_clearance/nonpositive_clearance_trials.csv',index=False)
notes={
 'analysis_manifest_summary':{k:v for k,v in manifest.items() if k!='sources'},
 'available_pose_samples_by_experiment':valid.groupby('experiment').pose_samples.sum().astype(int).to_dict(),
 'thresholds':{'zero_clearance':'exact value == 0.0 m, no epsilon','negative_boundary':'value < 0.0 m, no epsilon','nonpositive':'value <= 0.0 m, no epsilon'},
 'coverage_complete_groups':int((summary.coverage_complete==1).sum()),
 'total_groups':len(summary),
 'completed_unavailable_trial_count':int(((trials.included_in_primary_pool==1)&(trials.available==0)).sum()),
 'metric_definition':'Minimum over Euclidean convex polygon distance and signed map-boundary margin. Initial pose and every post-integration true pose pooled over completed success/failure/timeout/error trials; interrupted excluded. Mean and P05 are sample weighted, not averages of trial statistics.',
 'interpretation':'Zero includes obstacle contact/overlap or exact map-boundary contact; negative means footprint crosses map bounds. These are discrete dt=0.1 s observations, not continuous-time collision-free or probabilistic safety guarantees. A and B are separate cohorts; B reused 0.75 records are not independent runs.',
 'scope':'Only existing raw traces read; no solver or simulator run.',
}
(out/'audit/clearance_summary_audit.json').write_text(json.dumps(notes,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
print(summary[summary.experiment=='don'].to_string(index=False))
print(json.dumps({k:v for k,v in notes.items() if k!='analysis_manifest_summary'},ensure_ascii=False,indent=2))
