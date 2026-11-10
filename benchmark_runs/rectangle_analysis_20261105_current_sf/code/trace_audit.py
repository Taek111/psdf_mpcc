"""Read-only independent rectangle trial/trace/step audit; never imports a solver.

Run with the project .venv/bin/python. Source geometry/dynamics evidence is in
A/source_snapshot/{models/dd.py,models/dd_dynamics.py,sim/simulation_mpc.py}.
Initial clearance imports only the snapshot's pure numpy geometry helper.
Tolerance: absolute 1e-12 for pose/settings; 1e-9 for final accumulated clock;
1e-12 absolute + 1e-10 relative for serialized timing comparisons.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import yaml

sys.dont_write_bytecode = True

COMPLETE = {'success', 'failure', 'timeout', 'error'}
METHODS = ['mpcc', 'cc_ecs', 'cc_sf', 'cc_mf']
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_A = ROOT / 'benchmark_runs/rectangle_analysis_20261105_current_sf/rectangle_don_100_20261105_current_sf'
DEFAULT_B = ROOT / 'benchmark_runs/rectangle_constraint_reliability_20261107/rectangle_don_sweep_50_20261105_current_sf'

def read_csv(p):
    with p.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def dump_csv(path, rows, default_fields=()):
    fields = list(dict.fromkeys(k for row in rows for k in row)) or list(default_fields)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def close(a, b, atol=1e-12):
    try:
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        return a.shape == b.shape and bool(np.allclose(a, b, rtol=0, atol=atol, equal_nan=False))
    except (TypeError, ValueError):
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', type=Path, default=DEFAULT_A)
    parser.add_argument('--b', type=Path, default=DEFAULT_B)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'audit')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trials, groups, pose_matches, issues = [], [], [], []
    status_signatures = defaultdict(Counter)
    array_relations = Counter()
    maximum_integration_error = 0.0
    source_evidence = {}
    geometry_hashes = defaultdict(set)
    # Exact rectangle map, transcribed from the recorded create_env('maze').
    rectangles = [(0,4,0,3),(1,3,4,6),(3,6,5,6),(6,7,4,6),(4,5,0,4),
                  (5,8,2,3),(5,9,0,2),(8,9,2,4),(9,12,3,4),(11,12,4,5),
                  (7,10,5,6),(9,11,0,3),(12,13,1,2),(0,13,6,7),(-1,0,-1,7),
                  (0,13,-1,0),(13,14,-1,7)]
    obstacles_expected = [np.array([[a,c],[b,c],[b,d],[a,d]], dtype=float)*.15 for a,b,c,d in rectangles]
    for suite_label, root in [('A',args.a.resolve()),('B',args.b.resolve())]:
        manifest = json.loads((root/'suite_manifest.json').read_text())
        params = json.loads((root/'optimizer_parameters.json').read_text())
        helper_path = root/'source_snapshot/sim/start_perturbation.py'
        spec = importlib.util.spec_from_file_location(f'recorded_geometry_{suite_label}',helper_path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        source_evidence[suite_label] = {'helper':str(helper_path),'sha256':hashlib.sha256(helper_path.read_bytes()).hexdigest()}
        for width in manifest['widths']:
            directory = root / width['results_dir']
            run_manifest = yaml.safe_load((directory/'run_manifest.yaml').read_text())
            config_map = {r['configuration_id']:r['optimizer_type'] for r in run_manifest['configurations']}
            rows = read_csv(directory/'trial_results.csv')
            row_keys = Counter((r['optimizer_type'], int(r['start_seed'])) for r in rows)
            seed_rows = defaultdict(list)
            step_rows = defaultdict(list)
            for f in sorted(directory.glob('computation_times_*.csv')):
                for step in read_csv(f):
                    step_rows[(step['optimizer_type'], int(step['trial_number']))].append(step)
            summary_rows = {r['optimizer_type']:r for r in read_csv(directory/'summary.csv')}
            expected_seeds = set(manifest['seeds'])
            expected_keys = {(method,seed) for method in manifest['methods'] for seed in expected_seeds}
            missing_keys = expected_keys-set(row_keys)
            unexpected_keys = set(row_keys)-expected_keys
            for method, seed in sorted(missing_keys):
                issues.append({'suite':suite_label,'width_id':width['width_id'],'method':method,'seed':seed,'issue':'unrecorded_trial'})
            for method, seed in sorted(unexpected_keys):
                issues.append({'suite':suite_label,'width_id':width['width_id'],'method':method,'seed':seed,'issue':'unexpected_combination'})
            for key,count in row_keys.items():
                if count>1:
                    issues.append({'suite':suite_label,'width_id':width['width_id'],'method':key[0],'seed':key[1],'issue':f'duplicate_combination:{count}'})
            for row in rows:
                method, seed = row['optimizer_type'],int(row['start_seed'])
                seed_rows[seed].append(row)
                key = {'suite':suite_label,'width_id':width['width_id'],'width_m':width['width_m'],'don':width['nominal_don'],'method':method,'seed':seed,'trial_number':int(row['trial_number'])}
                found = []
                def check(ok,label):
                    if not ok:
                        found.append(label)
                out = {**key,'status':row['status'],'raw_trace_error':row.get('raw_trace_error','')}
                check(row['status'] in COMPLETE|{'interrupted'},'unknown_outcome_status')
                check(int(row['success']) == int(row['status']=='success'),'csv_success_status')
                check((row['goal_reached'].lower()=='true') == (row['status']=='success'),'csv_goal_status')
                check(config_map.get(row['configuration_id']) == method,'run_manifest_config_mapping')
                check(not row.get('raw_trace_error'),'csv_raw_trace_error')
                data = directory/'runs'/row['configuration_id']/f"trial_{int(row['trial_number']):04d}"/'data'
                npz_path, meta_path = data/'raw_trace.npz', data/'raw_trace_metadata.json'
                out['canonical_trace_path'] = str(npz_path)
                out['canonical_metadata_path'] = str(meta_path)
                try:
                    metadata = json.loads(meta_path.read_text())
                    with np.load(npz_path, allow_pickle=False) as archive:
                        arrays = {k:archive[k] for k in archive.files}
                    poses, pose_times, inputs, solve_times = [arrays[k] for k in ('poses','pose_times_s','applied_inputs','solve_nlp_times_s')]
                    statuses = metadata['solver_status_infos']
                    npose,ninput,ntime,nstatus = len(poses),len(inputs),len(solve_times),len(statuses)
                    out.update({'trace_loaded':True,'pose_samples':npose,'plant_integrations':ninput,'raw_time_samples':ntime,'status_samples':nstatus,'solve_minus_integration':ntime-ninput})
                    array_relations[f'solve_minus_integration={ntime-ninput}'] += 1
                    check(metadata.get('schema_version')==1,'schema_version')
                    check(set(arrays)=={'poses','pose_times_s','applied_inputs','solve_nlp_times_s'},'npz_array_names')
                    check(poses.shape==(npose,3) and npose>=1,'poses_shape')
                    check(pose_times.shape==(npose,),'pose_times_shape')
                    check(inputs.shape==(ninput,3),'applied_inputs_shape')
                    check(solve_times.shape==(ntime,),'solve_times_shape')
                    check(npose==ninput+1,'initial_plus_plant_integrations')
                    check(ntime==nstatus,'time_status_alignment')
                    for name,values in arrays.items():
                        check(bool(np.isfinite(values).all()),f'finite_{name}')
                    check(bool((solve_times>=0).all()),'nonnegative_solve_times')
                    for name, count in [('poses',npose),('applied_inputs',ninput),('solve_nlp_times_s',ntime),('solver_status_infos',nstatus)]:
                        check(metadata['array_counts'].get(name)==count,f'metadata_count_{name}')
                    check(close(metadata['dt_s'],.1),'dt_s')
                    check(close(pose_times,np.arange(npose)*metadata['dt_s']),'pose_times_schedule')
                    check(close(float(row['final_time']),pose_times[-1],1e-9),'final_time_pose_times')
                    csv_initial = [float(row[c]) for c in ['initial_x','initial_y','initial_theta']]
                    check(close(poses[0],csv_initial),'csv_trace_initial_pose')
                    check(close(metadata['initial_pose'],csv_initial),'csv_metadata_initial_pose')
                    check(metadata.get('raw_trace_sha256')==hashlib.sha256(npz_path.read_bytes()).hexdigest(),'npz_hash')
                    geometry = metadata['geometry']
                    foot = np.asarray(geometry['footprint_polygons'],dtype=float)
                    expected_foot = [[[-.075,-width['width_m']/2],[.075,-width['width_m']/2],[.075,width['width_m']/2],[-.075,width['width_m']/2]]]
                    check(close(foot,expected_foot),'footprint_length_width_rear_dist')
                    check(close(geometry['bounds'],[[0,0],[1.95,.9]]),'maze_bounds')
                    check(close(geometry['obstacle_polygons'],obstacles_expected),'maze_obstacles')
                    robot = metadata['robot']
                    check(robot.get('shape')=='rectangle' and close(robot.get('length'),.15) and close(robot.get('width'),width['width_m']),'robot_geometry')
                    geometry_hashes[f"{width['width_m']:.10f}"].add(hashlib.sha256(json.dumps(geometry,sort_keys=True).encode()).hexdigest())
                    for field in ['maze_type','optimizer_type','dynamics_type','path_planner']:
                        check(metadata['scenario'].get(field)==row[field],f'scenario_{field}')
                    check(metadata['optimizer_parameters']==params[method],'metadata_resolved_parameters')
                    perturb=metadata['start_perturbation']
                    check(perturb['seed']==seed,'perturbation_seed')
                    check(perturb.get('enabled') is True,'perturbation_enabled')
                    for field,expected in [('position_std',.02),('heading_std_deg',float(np.rad2deg(.05))),('max_sigma',3),('min_clearance',.001),('max_attempts',1000)]:
                        check(close(perturb.get(field),expected),f'perturbation_{field}')
                    check(1<=perturb['attempts']<=1000,'rejection_attempt_range')
                    check(close(perturb['initial_pose'],poses[0]),'perturbation_initial_pose')
                    check(close(np.asarray(perturb['nominal_pose'])+np.asarray(perturb['delta_pose']),poses[0]),'perturbation_delta')
                    check(bool((np.abs(perturb['delta_pose'])<=np.array([.06,.06,.15])+1e-12).all()),'perturbation_3sigma')
                    initial_clearance = helper.pose_clearance(poses[0],[p for p in foot],[np.asarray(p) for p in geometry['obstacle_polygons']],geometry['bounds'])
                    out['initial_clearance_m'] = initial_clearance
                    out['initial_clearance_metadata_error_m'] = initial_clearance-perturb['initial_clearance']
                    out['rejection_attempts'] = perturb['attempts']
                    check(initial_clearance>=.001,'initial_clearance_acceptance')
                    check(close(initial_clearance,perturb['initial_clearance']),'initial_clearance_metadata')
                    # Reconstruct the recorded RK4 plant map without creating a simulation.
                    theta=poses[:-1,2];v=inputs[:,0];omega=inputs[:,1];dt=metadata['dt_s']
                    half=theta+.5*dt*omega;end=theta+dt*omega
                    predicted=poses[:-1].copy()
                    predicted[:,0]+=dt*v*(np.cos(theta)+4*np.cos(half)+np.cos(end))/6
                    predicted[:,1]+=dt*v*(np.sin(theta)+4*np.sin(half)+np.sin(end))/6
                    predicted[:,2]=end
                    integration_error=float(np.max(np.abs(predicted-poses[1:]))) if ninput else 0.
                    maximum_integration_error=max(maximum_integration_error,integration_error)
                    out['rk4_max_abs_error'] = integration_error
                    check(integration_error<=1e-12,'rk4_plant_integration')
                    outcome = metadata['outcome']
                    for field in ['status','failure_reason','error','last_solve_mode']:
                        check((outcome.get(field) or '')==(row.get(field) or ''),f'outcome_{field}')
                    for field in ['final_time','distance_to_goal']:
                        check(close(outcome.get(field),float(row[field]),1e-9),f'outcome_{field}')
                    check(bool(outcome['goal_reached'])==(row['goal_reached'].lower()=='true'),'outcome_goal_reached')
                    # Recorded A* returns the goal grid cell center, and Robot uses the
                    # final global-path point for the L1 goal check (not nominal goal).
                    cell_size=.25*.15
                    goal_center=(np.floor(np.array([12*.15,.6*.15])/cell_size)+.5)*cell_size
                    goal_distance=float(np.abs(poses[-1,:2]-goal_center).sum())
                    check(close(goal_distance,float(row['distance_to_goal']),1e-12),'final_goal_distance')
                    check((goal_distance<=.02)==bool(outcome['goal_reached']),'final_goal_criterion')
                    stats=outcome.get('optimizer_runtime_stats')
                    out['runtime_stats_available']=stats is not None
                    for field in ['safe_stop_count','backup_feasible_qp_success_count','plant_input_apply_count']:
                        val=outcome.get(field)
                        check((val is None and not row[field]) or (val is not None and str(val)==row[field]),f'csv_counter_{field}')
                        if stats is not None and field in stats:
                            check(stats[field]==val,f'runtime_counter_{field}')
                    if stats is not None:
                        check(stats.get('plant_input_apply_count')==ninput,'runtime_plant_integrations')
                        check(stats.get('last_solve_mode')==outcome.get('last_solve_mode'),'runtime_last_solve_mode')
                    # Never align solve calls to plant records by truncation. Only status/time
                    # arrays share a verified solve-call index; plant counts are independent.
                    step_key=(method,int(row['trial_number']))
                    selected_steps=step_rows.pop(step_key,[])
                    expected_indices=[]
                    invalid=unknown=failed=skipped=0
                    for i,status in enumerate(statuses):
                        status_signatures[method][json.dumps(status,sort_keys=True)] += 1
                        if status.get('solver_called') is False:
                            skipped+=1
                        if status.get('success') is True and i<ntime and np.isfinite(solve_times[i]) and solve_times[i]>=0 and status.get('solver_called') is not False:
                            expected_indices.append(i)
                        elif status.get('success') is False:
                            failed+=1
                        elif status.get('success') is not True:
                            unknown+=1
                        else:
                            invalid+=1
                    actual_indices=[int(x['timestep']) for x in selected_steps]
                    check(actual_indices==expected_indices,'step_csv_success_index_vector')
                    check(len(actual_indices)==len(set(actual_indices)),'step_csv_duplicate_indices')
                    check(int(row['successful_solver_steps'])==len(expected_indices),'csv_successful_solver_steps')
                    for step in selected_steps:
                        index=int(step['timestep'])
                        check(step['trial_status']==row['status'] and step['trial_success']==row['success'],'step_trial_outcome')
                        check(int(step['start_seed'])==seed,'step_seed')
                        if index>=ntime or index>=nstatus:
                            check(False,'step_index_range')
                            continue
                        check(np.isclose(float(step['solver_computation_time_s']),solve_times[index],rtol=1e-10,atol=1e-12),'step_time_value')
                        check(close(float(step['simulation_time_s']),(index+1)*dt),'step_simulation_time')
                        check(step['solver_raw_status']==str(statuses[index].get('raw_status') or ''),'step_raw_status')
                        check(step['solver_status_code']==str(statuses[index].get('status_code')),'step_status_code')
                    selected_times=solve_times[expected_indices]
                    for field,func in [('solver_time_mean_s',np.mean),('solver_time_p95_s',lambda x:np.percentile(x,95))]:
                        check((not len(selected_times) and not row[field]) or (len(selected_times)>0 and np.isclose(float(row[field]),func(selected_times),rtol=1e-10,atol=1e-12)),f'trial_{field}')
                    stage0=sum(x.get('raw_status')=='safe_stop_stage0' for x in statuses)
                    if stats is not None and 'stage0_guard_stop_count' in stats:
                        check(stats['stage0_guard_stop_count']==stage0,'stage0_counter_vs_status')
                    out.update({'successful_time_samples':len(expected_indices),'failed_status_samples':failed,'unknown_status_samples':unknown,'invalid_success_time_samples':invalid,'skipped_solver_samples':skipped,'stage0_status_samples':stage0,'step_csv_rows':len(selected_steps)})
                except Exception as exc:
                    out.setdefault('trace_loaded',False)
                    found.append(f'trace_exception:{type(exc).__name__}:{exc}')
                out['audit_passed']=not found
                out['issue_count']=len(found)
                out['issues']=' | '.join(sorted(set(found)))
                trials.append(out)
                for issue in sorted(set(found)):
                    issues.append({**key,'issue':issue})
            for key,leftover in step_rows.items():
                issues.append({'suite':suite_label,'width_id':width['width_id'],'method':key[0],'trial_number':key[1],'issue':f'orphan_step_rows:{len(leftover)}'})
            for seed in sorted(expected_seeds|set(seed_rows)):
                sr=seed_rows[seed]
                poses=np.asarray([[float(r[k]) for k in ['initial_x','initial_y','initial_theta']] for r in sr])
                matched=len(sr)==4 and set(r['optimizer_type'] for r in sr)==set(METHODS) and close(poses,np.repeat(poses[:1],4,axis=0))
                pose_matches.append({'suite':suite_label,'width_id':width['width_id'],'width_m':width['width_m'],'don':width['nominal_don'],'seed':seed,'method_rows':len(sr),'methods':'|'.join(r['optimizer_type'] for r in sr),'matched':matched,'max_pose_abs_difference':float(np.ptp(poses,axis=0).max()) if len(poses) else None,'initial_x':poses[0,0] if len(poses) else None,'initial_y':poses[0,1] if len(poses) else None,'initial_theta':poses[0,2] if len(poses) else None})
                if not matched:
                    issues.append({'suite':suite_label,'width_id':width['width_id'],'seed':seed,'issue':'four_method_initial_pose_match'})
            # Compare supplied summary to independent aggregation from authoritative step CSV.
            for method in METHODS:
                selected=[r for r in rows if r['optimizer_type']==method]
                status_counts=Counter(r['status'] for r in selected)
                complete=sum(status_counts[s] for s in COMPLETE)
                expected=len(expected_seeds)
                summary=summary_rows.get(method,{})
                group={'suite':suite_label,'width_id':width['width_id'],'width_m':width['width_m'],'don':width['nominal_don'],'method':method,'expected_trials':expected,'recorded_trials':len(selected),'completed_trials':complete,'success_trials':status_counts['success'],'failure_trials':status_counts['failure'],'timeout_trials':status_counts['timeout'],'error_trials':status_counts['error'],'interrupted_trials':status_counts['interrupted'],'unrecorded_trials':len([k for k in missing_keys if k[0]==method]),'duplicate_extra_rows':sum(v-1 for k,v in row_keys.items() if k[0]==method and v>1),'unexpected_combinations':len([k for k in unexpected_keys if k[0]==method])}
                fs=list(directory.glob(f'computation_times_*_{method}_*.csv'))
                times=np.array([float(s['solver_computation_time_s']) for f in fs for s in read_csv(f)],dtype=float)
                gissues=[]
                comparison={'recorded_trials':len(selected),'completed_trials':complete,'successful_trials':status_counts['success'],'failed_trials':complete-status_counts['success'],'interrupted_trials':status_counts['interrupted'],'successful_solver_steps':len(times)}
                for field,value in comparison.items():
                    if field not in summary or int(summary[field])!=value:
                        gissues.append('summary_'+field)
                for field,value in [('success_rate',status_counts['success']/complete if complete else None),('solver_time_mean_s',float(np.mean(times)) if len(times) else None),('solver_time_p95_s',float(np.percentile(times,95)) if len(times) else None)]:
                    if value is None:
                        if summary.get(field):gissues.append('summary_'+field)
                    elif field not in summary or not np.isclose(float(summary[field]),value,rtol=1e-10,atol=1e-12):
                        gissues.append('summary_'+field)
                group['summary_matches']=not gissues
                group['summary_issues']='|'.join(gissues)
                groups.append(group)
                for issue in gissues:
                    issues.append({'suite':suite_label,'width_id':width['width_id'],'method':method,'issue':issue})
            print(f'{suite_label} {width["width_id"]}: audited {len(rows)} rows',flush=True)
    for width,hashes in geometry_hashes.items():
        if len(hashes)!=1:
            issues.append({'width_m':width,'issue':'geometry_differs_across_methods_or_cohorts'})
    report={
        'schema_version':1,
        'script':str(Path(__file__).resolve()),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'roots':{'A':str(args.a.resolve()),'B':str(args.b.resolve())},
        'audit_passed':not issues,
        'trial_memberships':len(trials),
        'trace_loaded':sum(r['trace_loaded'] for r in trials),
        'trace_trials_passed':sum(r['audit_passed'] for r in trials),
        'issue_count':len(issues),
        'issue_counts':dict(Counter(r['issue'] for r in issues)),
        'suites':{label:{field:sum(g[field] for g in groups if g['suite']==label) for field in ['expected_trials','recorded_trials','completed_trials','success_trials','failure_trials','timeout_trials','error_trials','interrupted_trials','unrecorded_trials','duplicate_extra_rows','unexpected_combinations']} for label in ['A','B']},
        'four_method_pose_groups':len(pose_matches),
        'four_method_pose_groups_matched':sum(r['matched'] for r in pose_matches),
        'all_summary_rows_match':all(g['summary_matches'] for g in groups),
        'array_length_relations':dict(array_relations),
        'max_rk4_integration_abs_error':maximum_integration_error,
        'goal_endpoint_m':goal_center.tolist(),
        'initial_clearance_min_m':min(r['initial_clearance_m'] for r in trials if 'initial_clearance_m' in r),
        'status_signature_counts':{m:dict(v) for m,v in status_signatures.items()},
        'geometry_hashes_by_width':{w:sorted(v) for w,v in geometry_hashes.items()},
        'recorded_source_geometry_helpers':source_evidence,
        'interpretation':[
            'A/B memberships are distinct analytical cohorts; 200 reused B memberships are not independent reruns.',
            'No status/time/plant array is truncated to force alignment; solve-call and plant counts are independently audited.',
            'Runtime counters missing for MPCC remain missing; source evidence is needed to describe structural zero.',
            'CC-ECS inherits main_solver_failure_count but its override solve_nlp never increments it; stored zero is not evidence of no main-solver failure. Safe-stop and recovery counters are incremented by this override.',
            'CC-ECS stage0_guard_stop_count is inherited zero; its solve_nlp has no stage-0 guard-stop path.',
            'Goal checks use the A* goal cell center [1.81875, 0.09375], derived from recorded GridMap and Robot._get_goal_position, rather than nominal [1.8,0.09].',
            'Stage-0 stop raw_status is available per call. Overall stop episodes and recovery episodes are not fully measured: ordinary failure/success status omits per-call solve_mode.',
            'Initial clearance is independently recomputed using each recorded snapshot helper, without simulation/solver calls.'
        ],
    }
    dump_csv(args.output_dir/'trace_trials.csv',trials)
    dump_csv(args.output_dir/'trace_groups.csv',groups)
    dump_csv(args.output_dir/'trace_pose_matching.csv',pose_matches)
    dump_csv(args.output_dir/'trace_issues.csv',issues,('suite','width_id','method','seed','issue'))
    (args.output_dir/'trace_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['audit_passed','trial_memberships','trace_loaded','trace_trials_passed','issue_count','issue_counts','max_rk4_integration_abs_error','initial_clearance_min_m']},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
