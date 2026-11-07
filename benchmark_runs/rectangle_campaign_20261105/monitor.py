import csv,json,time
from pathlib import Path
base=Path('/home/taekwon/projects/psdf_mpcc/benchmark_runs')
r=base/'rectangle_don_sweep_50_20261105_current_sf'
while True:
    counts={}
    errors=0
    for p in r.glob('widths/*/trial_results.csv'):
        rows=list(csv.DictReader(p.open()))
        counts[p.parent.name]=len(rows)
        errors+=sum(bool(x.get('raw_trace_error')) for x in rows)
    state=json.loads((r/'suite_state.json').read_text()) if (r/'suite_state.json').exists() else {}
    print(json.dumps({'state':state.get('status'),'current':state.get('current_width'),'rows':sum(counts.values()),'completed_widths':len(state.get('completed_widths',[])),'trace_errors':errors}),flush=True)
    if state.get('status') in ('complete','error','incomplete','interrupted'): break
    time.sleep(60)
