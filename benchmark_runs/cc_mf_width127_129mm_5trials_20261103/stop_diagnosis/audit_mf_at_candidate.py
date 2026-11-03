"""Evaluate recorded MF affine rows and frozen nonlinear residuals offline."""
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
RUN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUN/'source_snapshot'))
import numpy as np
import torch
from models.augmented_psdf_wrapper import AugmentedPSDFWrapper
from models.dd import DifferentialDriveRectangleGeometry
from models.geometry_utils import polygon_to_edges
from control.cc_covariance import initial_local_covariance, log_cholesky_from_covariance, propagate_log_cholesky
from sim.simulation_mpc import simulation_mpc

torch.set_num_threads(6)
_,_,_,obstacles=simulation_mpc().create_env('maze')
A,B=zip(*(polygon_to_edges(o) for o in obstacles))
rows=[]
for w,s in ((128,3),(129,1),(129,3),(129,4)):
    run=RUN/f'width_{w}mm/cc_mf/seed_{s}'
    par=SimpleNamespace(**json.loads((run/'optimizer_parameters.json').read_text()))
    cyc=list(csv.DictReader((run/'cycles.csv').open()))
    k=next(i for i,r in enumerate(cyc) if r['solve_mode']=='safe_stop_stage0')
    log=list(csv.DictReader(next((run/'data').glob('cc_mf_cycle_*.csv')).open()))
    cp=[json.loads(l) for l in (run/'animation_checkpoint.jsonl').open()]
    nominal=np.array(cp[k-1]['optimized_traj'][1][:8])
    candidate=np.array(cp[k]['optimized_traj'][1][:8])
    xi0=log_cholesky_from_covariance(initial_local_covariance(par))
    ubar=json.loads(log[k-2]['u0'])
    nominal[4:8]=propagate_log_cholesky(xi0,ubar[0],ubar[1],0,.1,par)
    geom=DifferentialDriveRectangleGeometry(.15,w/1000,0)
    verts=torch.tensor(geom._region.get_ccw_vertices(),dtype=torch.float32,device='cuda')
    wrapper=AugmentedPSDFWrapper(verts,K_max=20,E_max=100,device='cuda')
    wrapper.update_edge_clusters(list(A),list(B))
    nominal=nominal[None]
    candidate=candidate[None]
    res=wrapper.forward_mf_icdf(nominal,par.chance_epsilon,par.d_min,return_frozen_features=True)
    psi=float(res.residual[0].cpu())
    logged_psi=json.loads(log[k-1]['mf_icdf_residual_nominal'])[1]
    assert abs(psi-logged_psi)<1e-7,(w,s,psi,logged_psi)
    coefficients=res.A.cpu().numpy()
    c=res.c.cpu().numpy()
    affine=float((coefficients*candidate).sum()+c[0])
    nonlinear=wrapper.evaluate_mf_icdf_frozen(res.frozen_features,candidate,par.chance_epsilon,par.d_min)
    row={'width_m':w/1000,'seed':s,'nominal_probability_sum':float(res.probability_sum[0].cpu()),
         'nominal_psi':psi,'candidate_affine_psi':affine,
         'candidate_frozen_nonlinear_probability_sum':float(nonlinear.probability_sum[0].cpu()),
         'candidate_frozen_nonlinear_psi':float(nonlinear.residual[0].cpu()),
         'stage1_slack':json.loads(log[k-1]['mf_slack'])[1]}
    rows.append(row)
    print(json.dumps(row),flush=True)
(RUN/'stop_diagnosis/mf_candidate_residuals.json').write_text(json.dumps(rows,indent=2)+'\n')
