"""Limit one actual run and recover its recorded trajectory for animation."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('method');p.add_argument('--budget',type=int,default=900)
a=p.parse_args();out=Path(__file__).resolve().parent;label=a.method+'_pp'
cmd=[sys.executable,'-u',str(out/'run_one.py'),a.method,'--run-label',label,'--wall-limit',str(a.budget+30)]
started=time.perf_counter()
with (out/(label+'.log')).open('w') as log:
    process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        status=process.wait(timeout=a.budget)
    except subprocess.TimeoutExpired:
        live=out/label/'checkpoint.json'
        if live.exists():
            (out/label/'budget_checkpoint.json').write_bytes(live.read_bytes())
        os.killpg(process.pid,signal.SIGKILL)
        process.wait()
        with (out/(label+'_recovery.log')).open('w') as recovery:
            status=subprocess.call(cmd+['--recover-from-budget'],stdout=recovery,stderr=subprocess.STDOUT)
        result_path=out/label/'result.json'
        if result_path.exists():
            result=json.loads(result_path.read_text())
            result.update(wall_budget_seconds=a.budget,wall_seconds=time.perf_counter()-started)
            result_path.write_text(json.dumps(result,indent=2))
print(a.method,'exit',status,'wall',time.perf_counter()-started,flush=True)
