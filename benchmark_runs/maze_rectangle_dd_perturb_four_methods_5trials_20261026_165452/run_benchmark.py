"""Run the requested batch, retaining configuration and source provenance."""
import contextlib
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import runpy
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path('/home/taek111/projects/psdf_mpcc')
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ['ACADOS_SOURCE_DIR'] = '/home/taek111/projects/acados'
os.environ['LD_LIBRARY_PATH'] = '/home/taek111/projects/acados/lib'
os.environ['MPLBACKEND'] = 'Agg'
stamp = dt.datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S')
output = ROOT / 'benchmark_runs' / f'maze_rectangle_dd_perturb_four_methods_5trials_{stamp}'
output.mkdir(parents=True, exist_ok=False)
Path('/tmp/psdf_benchmark_current_path.txt').write_text(str(output))
print(f'RESULTS_DIR={output}', flush=True)
shutil.copy2(__file__, output / 'run_benchmark.py')
shutil.copy2(ROOT / 'config/config_benchmark.yaml', output / 'config.yaml')
source_paths = [ROOT / 'test_nmpc.py']
for directory in ('control', 'models', 'planning', 'sim', 'utils'):
    source_paths.extend(sorted((ROOT / directory).rglob('*.py')))
hashes = {}
for source in source_paths:
    relative = source.relative_to(ROOT)
    hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    target = output / 'source_snapshot' / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)

started = time.perf_counter()
metadata = {
    'created_at_kst': dt.datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
    'python': sys.executable, 'python_version': sys.version,
    'platform': platform.platform(),
    'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'source_sha256': hashes,
    'thread_environment': {key: os.environ.get(key) for key in (
        'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
}
(output / 'environment.json').write_text(json.dumps(metadata, indent=2))
with (output / 'run.log').open('w', buffering=1) as log:
    # dup2 also captures acados compiler and C-library output in the same log.
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    import numpy as np
    import torch
    import casadi
    from sim.simulation_mpc import simulation_mpc
    from models.dd import DifferentialDriveRectangleGeometry
    from control.mpcc_optimizer import MPCCOptimizerParam
    from control.cc_sf_optimizer import CCSFOptimizerParam
    from control.cc_mf_optimizer import CCMFOptimizerParam
    from control.cc_ecs_optimizer import CCECSOptimizerParam

    def serialize(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f'Cannot serialize {type(value)}')

    metadata.update({
        'torch': torch.__version__, 'casadi': casadi.__version__,
        'numpy': np.__version__, 'cuda_available': torch.cuda.is_available(),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'torch_num_threads': torch.get_num_threads(),
    })
    (output / 'environment.json').write_text(json.dumps(metadata, indent=2))
    params = {name: vars(cls()) for name, cls in (
        ('mpcc', MPCCOptimizerParam), ('cc_sf', CCSFOptimizerParam),
        ('cc_mf', CCMFOptimizerParam), ('cc_ecs', CCECSOptimizerParam))}
    (output / 'optimizer_defaults.json').write_text(json.dumps(params, indent=2, default=serialize))
    start, goal, grid, obstacles = simulation_mpc().create_env('maze')
    footprint = DifferentialDriveRectangleGeometry(length=0.15, width=0.10, rear_dist=0.0)
    geometry = {
        'bounds': grid[0], 'nominal_start': start, 'goal': goal, 'dt_s': 0.1,
        'footprint': footprint.equiv_rep()[0].get_ccw_vertices(),
        'obstacles': [obstacle.get_ccw_vertices() for obstacle in obstacles],
    }
    (output / 'geometry.json').write_text(json.dumps(geometry, indent=2, default=serialize))
    sys.argv = ['test_nmpc.py', '-c', str(output / 'config.yaml'),
                '--trials', '5', '--results-dir', str(output), '--no-animation', '--no-plots']
    print('Command:', ' '.join([sys.executable] + sys.argv), flush=True)
    try:
        runpy.run_path(str(ROOT / 'test_nmpc.py'), run_name='__main__')
    finally:
        metadata['finished_at_kst'] = dt.datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
        metadata['wall_time_s'] = time.perf_counter() - started
        metadata['sources_changed_during_run'] = [
            name for name, digest in hashes.items()
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
        (output / 'environment.json').write_text(json.dumps(metadata, indent=2))
        print('Batch finished; results:', output, flush=True)
