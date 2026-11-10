"""Run the required unmodified offline helper and retain execution receipt."""
from pathlib import Path
import subprocess, sys, time, json, datetime, hashlib
root=Path(__file__).resolve().parents[3]
out=Path(__file__).resolve().parents[1]
command=[str(root/'.venv/bin/python'), str(root/'scripts/analyze_rectangle_clearance.py'), str(root/'benchmark_runs/rectangle_analysis_20261105_current_sf/rectangle_don_100_20261105_current_sf'), str(root/'benchmark_runs/rectangle_constraint_reliability_20261107/rectangle_don_sweep_50_20261105_current_sf'), '--output-dir', str(out/'offline_clearance'), '--strict']
receipt={'command':command,'cwd':str(root),'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'running','helper_sha256':hashlib.sha256((root/'scripts/analyze_rectangle_clearance.py').read_bytes()).hexdigest()}
receipt_path=out/'audit/clearance_execution.json'
receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
start=time.perf_counter()
with (out/'audit/clearance_execution.log').open('w') as log:
    process=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
receipt.update(status='completed',exit_code=process.returncode,elapsed_seconds=time.perf_counter()-start,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
sys.exit(process.returncode)
