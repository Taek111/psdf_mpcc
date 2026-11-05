"""Validate the complete offline cache; does not replay geometry or alter inputs."""
import hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent
cache=root/'cycle_cache'
expected={f'width_{w:03d}mm_{m}_seed_{s}_stage_{j}.npz' for w in range(125,131) for m in ['cc_sf','cc_mf'] for s in range(10) for j in [1,10]}
actual={p.name for p in cache.glob('*.npz')}
assert actual==expected,dict(missing=sorted(expected-actual),extra=sorted(actual-expected))
groups={}; caps=[]; hashes={}; pipeline_hashes=set(); source_hashes=set(); agreements={}; maxima={}
for name in sorted(expected):
 p=cache/name
 hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
 with np.load(p,allow_pickle=False) as z:
  meta=json.loads(str(z['metadata_json'])); method=str(z['method']); stage=int(z['stage']); n=len(z['cycle'])
  assert int(z['schema_version'])==2
  assert z['nominal_states'].shape==(n,8) and z['post_states'].shape==(n,8)
  assert np.array_equal(z['cycle'],np.arange(n))
  assert np.allclose(np.diff(z['time']),.1,rtol=0,atol=1e-8)
  assert np.all(~z['post_valid']|z['solver_success'])
  assert np.all(~z['nominal_valid']|z['post_valid'])
  assert np.all(~z['frozenpost_probability_valid']|(z['nominal_probability_valid']&z['post_valid']))
  assert np.array_equal(z['frozenpost_valid'],z['frozenpost_probability_valid'])
  assert meta['epsilon']==.1 and meta['d_min_m']==.0001 and meta['d_col_m']==.00005
  assert float(z['sf_reference_cap_m'])==2.
  pipeline_hashes.add(meta['extraction_pipeline_sha256'])
  source_hashes.add(meta['sources']['models/augmented_psdf.py']['sha256'])
  key=f'{method}_stage{stage}'
  g=groups.setdefault(key,defaultdict(int));g['trials']+=1;g['cycles']+=n
  for flag in ['solver_success','post_valid','nominal_valid','replay_valid','post_probability_valid','nominal_probability_valid','frozenpost_probability_valid','safe_stop','native_replay_mask_disagreement']:
   g[flag]+=int(np.sum(z[flag]))
  g['mf_native_replay_rejected_reported']+=meta.get('mf_nominal_replay_rejected_cycles',0)
  for prefix in ['nominal','post']:
   valid=z[prefix+'_probability_valid']
   for field in ['p_selected','R','p_max','p_other_max']:
    a=z[prefix+'_'+field]
    assert np.isnan(a[~valid]).all(),(name,prefix,field,'invalid domain populated')
    assert np.isfinite(a[valid]).all(),(name,prefix,field,'valid domain missing')
   assert np.all(z[prefix+'_R'][valid]>=z[prefix+'_p_max'][valid]-1e-12)
   assert np.all(z[prefix+'_p_max'][valid]<=1.)
   assert np.all(z[prefix+'_p_selected'][valid]>=0.) and np.all(z[prefix+'_p_selected'][valid]<=1.)
   available=z[prefix+'_valid']
   assert np.array_equal(z[prefix+'_mf_required'][available]==1,z[prefix+'_phi_m'][available]>.0001)
   for branch in [0,1,2,3,4]:g[f'{prefix}_mf_branch_{branch}']+=int(np.sum(z[prefix+'_mf_branch'][available]==branch))
   sigma=z[prefix+'_sigma_selected_m'];f=np.isfinite(sigma)
   k=int(np.nanargmax(sigma)) if f.any() else None
   maxkey=key+'_'+prefix+'_sigma'
   if k is not None and (maxkey not in maxima or float(sigma[k])>maxima[maxkey]['sigma_m']):
    maxima[maxkey]=dict(cache=name,cycle=k,time_s=float(z['time'][k]),sigma_m=float(sigma[k]))
   active=np.flatnonzero(z[prefix+'_sf_cap_active']==1)
   g[prefix+'_sf_cap_active']+=len(active)
   for k in active:caps.append(dict(cache=name,phase=prefix,cycle=int(k),time_s=float(z['time'][k]),sigma_m=float(sigma[k]),phi_m=float(z[prefix+'_phi_m'][k])))
  active=np.flatnonzero(z['frozenpost_sf_cap_active']==1)
  g['frozenpost_sf_cap_active']+=len(active)
  for k in active:caps.append(dict(cache=name,phase='frozenpost',cycle=int(k),time_s=float(z['time'][k]),sigma_m=float(z['frozenpost_sigma_selected_m'][k])))
  if method=='cc_sf':pairs=[('SF_affine_mm',z['native_affine']*1000,z['frozenpost_sf_affine_margin_mm']),('SF_fresh_mm',z['native_fresh']*1000,z['post_sf_native_margin_mm'])]
  else:pairs=[('MF_affine_quantile',z['native_affine'],z['frozenpost_mf_affine_quantile']),('MF_fresh_quantile',z['native_fresh'],z['post_mf_quantile'])]
  for label,a,b in pairs:
   finite=np.isfinite(a)&np.isfinite(b);error=np.abs(a-b)
   if finite.any():
    k=int(np.argmax(np.where(finite,error,-1)))
    if label not in agreements or float(error[k])>agreements[label]['max_abs_error']:
     agreements[label]=dict(max_abs_error=float(error[k]),cache=name,cycle=k)
  for field in ['nominal_selected_q_match_max_abs_error','post_selected_q_match_max_abs_error','nominal_selected_distance_match_error_m','post_selected_distance_match_error_m']:
   a=np.abs(z[field]);f=np.isfinite(a)
   if f.any() and (field not in maxima or float(np.max(a[f]))>maxima[field]['error']):
    maxima[field]=dict(error=float(np.max(a[f])),cache=name)
assert len(pipeline_hashes)==1,pipeline_hashes
assert len(source_hashes)==1,source_hashes
result=dict(complete=True,caches=len(expected),groups=groups,cap_cases=caps,maxima=maxima,native_replay_agreement=agreements,pipeline_sha256=list(pipeline_hashes),archived_geometry_sha256=list(source_hashes),notes=['Covariance and probabilities are saved planning-model diagnostics, not measured uncertainty or empirical collision probabilities.','Native SF risk cap is fixed at2m on both controller trajectories; sf_quantile remains unclipped.','Independent nominal domain and fallback masks are mandatory.','Rows/stages within trials are correlated, so use trial-level summaries.'])
(root/'cache_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
(root/'cache_sha256.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['maxima']},indent=2))
