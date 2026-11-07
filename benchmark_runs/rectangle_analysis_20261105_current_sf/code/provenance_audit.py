#!/usr/bin/env python3
"""Read-only provenance and fixed-condition audit for completed Rectangle cohorts.

Run with project .venv/bin/python. No solver imports and no simulation calls.
Inputs are resolved from stored suite manifests; outputs only go to OUT.
"""
from __future__ import annotations

import argparse
import ast
import csv
import difflib
import hashlib
import json
import math
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[3]
DEFAULT_A = PROJECT / 'benchmark_runs/rectangle_don_100_20261105_current_sf'
DEFAULT_B = PROJECT / 'benchmark_runs/rectangle_don_sweep_50_20261105_current_sf'
DEFAULT_OUT = PROJECT / 'benchmark_runs/rectangle_analysis_20261105_current_sf'
METHODS = ['mpcc', 'cc_ecs', 'cc_sf', 'cc_mf']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames=None):
    names = fieldnames or list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def numerical(values):
    return {k: v for k, v in values.items() if k != 'use_debug' and not k.startswith('debug')
            and k not in ('constraint_debug_tolerance', 'first_interval_substeps')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a', type=Path, default=DEFAULT_A)
    parser.add_argument('--b', type=Path, default=DEFAULT_B)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve() / 'audit'
    out.mkdir(parents=True, exist_ok=True)
    roots = {'A': args.a.resolve(), 'B': args.b.resolve()}
    checks, hash_rows, config_rows, metadata_rows, reuse_rows, step_rows = [], [], [], [], [], []

    def check(scope, name, observed, expected):
        checks.append({'scope': scope, 'check': name, 'passed': observed == expected,
                       'observed': encoded(observed), 'expected': encoded(expected)})

    manifests = {s: load(r / 'suite_manifest.json') for s, r in roots.items()}
    params = {s: load(r / 'optimizer_parameters.json') for s, r in roots.items()}
    environments = {s: load(r / 'environment.json') for s, r in roots.items()}
    hashes = {s: load(r / 'source_hashes.json') for s, r in roots.items()}
    cfgs, rms = {}, {}
    for suite, root in roots.items():
        manifest = manifests[suite]
        snapshot = root / manifest['source_snapshot']
        recorded = hashes[suite]
        # Recreate the snapshot producer's file-selection rule, without import.
        selected_paths = [snapshot / name for name in
                          ('test_nmpc.py', 'AGENTS.md', 'benchmark_results_old.md',
                           'scripts/rectangle_benchmark_baseline.json')]
        for directory in ('control', 'models', 'planning', 'sim', 'utils', 'scripts'):
            for suffix in ('*.py', '*.sh'):
                selected_paths.extend((snapshot / directory).rglob(suffix))
        selected_paths.extend((snapshot / 'config').glob('*.yaml'))
        selected_paths.extend((snapshot / 'docs').glob('rectangle_benchmark*.md'))
        selected_names = sorted({str(p.relative_to(snapshot)) for p in selected_paths if p.is_file()})
        check(suite, 'source_file_inventory_complete', selected_names, sorted(recorded['files']))
        actual = {}
        for rel, expected_hash in recorded['files'].items():
            path = snapshot / rel
            actual_hash = sha(path) if path.is_file() else None
            actual[rel] = actual_hash
            hash_rows.append({'suite': suite, 'kind': 'source_snapshot', 'path': str(path),
                              'recorded_sha256': expected_hash, 'actual_sha256': actual_hash,
                              'matched': actual_hash == expected_hash})
            check(suite, f'source_hash:{rel}', actual_hash, expected_hash)
        aggregate = hashlib.sha256(''.join(f'{n}\0{d}\n' for n, d in sorted(actual.items())).encode()).hexdigest()
        check(suite, 'source_aggregate_actual_vs_recorded', aggregate, recorded['aggregate_sha256'])
        check(suite, 'source_aggregate_actual_vs_manifest', aggregate, manifest['source_aggregate_sha256'])
        baseline = load(root / manifest['baseline_parameters'])
        check(suite, 'baseline_validation', load(root / 'baseline_validation.json'), {'matched': True, 'differences': {}})
        for method in METHODS:
            check(suite, f'amended_baseline_numerical_parameters:{method}', numerical(params[suite][method]), numerical(baseline['parameters'][method]))
            for key, value in {'use_debug': False, 'horizon': 20, 'tf': 2.0, 'd_min': .0001, 'a_max': .3, 'r_delta_omega': .1}.items():
                check(suite, f'parameter:{method}:{key}', params[suite][method].get(key), value)
            if method != 'mpcc':
                for key, value in {'sigma_f0': .0002, 'sigma_l0': .00025, 'sigma_psi0': .00025}.items():
                    check(suite, f'parameter:{method}:{key}', params[suite][method].get(key), value)
        for key, value in {'plan_only': False, 'methods': METHODS, 'computation_time_scope': 'solve_nlp',
                           'computation_time_unit': 's', 'successful_solver_steps_only': True,
                           'include_first_step': True, 'use_debug': False, 'clearance_evaluation': 'offline_only',
                           'benchmark_num_threads': 6}.items():
            check(suite, f'suite_manifest:{key}', manifest.get(key), value)
        seeds = list(range(100 if suite == 'A' else 50))
        check(suite, 'suite_manifest:seeds', manifest['seeds'], seeds)
        widths = [.075, .1125] if suite == 'A' else [round(.15 * d / 100, 7) for d in range(70, 88)]
        check(suite, 'width_schedule', [w['width_m'] for w in manifest['widths']], widths)
        for width in manifest['widths']:
            width_id = width['width_id']
            cfgpath = root / width['config_path']
            resultdir = root / width['results_dir']
            rmpath = resultdir / 'run_manifest.yaml'
            cfg, rm = yaml.safe_load(cfgpath.read_text()), yaml.safe_load(rmpath.read_text())
            cfgs[suite, width_id], rms[suite, width_id] = cfg, rm
            scope = f'{suite}/{width_id}'
            expected_cfg = {'trials': len(seeds), 'rectangle_geometry': {'length': .15, 'width': width['width_m']},
                'mpcc': {'use_debug': False}, 'benchmark_logging': {'enabled': True},
                'start_perturbation': {'enabled': True, 'seed': 0, 'position_std': .02,
                                      'heading_std_deg': math.degrees(.05), 'max_sigma': 3.0,
                                      'min_clearance': .001, 'max_attempts': 1000},
                'localization_error': {'enabled': False},
                'defaults': {'simulation_time': 80.0, 'generate_animation': False, 'generate_plots': False},
                'success_criteria': {'position_tolerance': .02},
                'trial_failure_criteria': {'enabled': True, 'movement_window_sec': 10.0,
                                           'movement_threshold': .03, 'max_consecutive_solver_failures': 20},
                'test_configs': [{'maze_type': 'maze', 'robot_shape': 'rectangle', 'optimizer_type': method,
                                  'dynamics_type': 'differential_drive', 'path_planner': 'astar'} for method in METHODS]}
            check(scope, 'generated_config_all_fields', cfg, expected_cfg)
            check(scope, 'run_manifest_config_matches_yaml', rm['config'], cfg)
            check(scope, 'run_manifest_trial_count', rm['trials_per_configuration'], len(seeds))
            check(scope, 'run_manifest_timing', rm['computation_time'], {
                'clock': 'time.perf_counter', 'unit': 's', 'scope_by_optimizer': {m: 'solve_nlp' for m in METHODS},
                'successful_steps_only': True, 'debug_work_in_solve_nlp': 'included_when_enabled'})
            expected_configurations = [{**c, 'configuration_id': f'config{i:02d}',
                                        'start_perturbation': cfg['start_perturbation']}
                                       for i, c in enumerate(cfg['test_configs'], 1)]
            check(scope, 'configuration_identity_mapping', rm['configurations'], expected_configurations)
            config_rows.append({'suite': suite, 'width_id': width_id, 'width_m': width['width_m'],
                'nominal_don': width['nominal_don'], 'generated_yaml': str(cfgpath), 'yaml_sha256': sha(cfgpath),
                'run_manifest': str(rmpath), 'run_manifest_sha256': sha(rmpath),
                'yaml_and_manifest_config_equal': cfg == rm['config'],
                'fixed_conditions_passed': cfg == expected_cfg})
            for row in read_csv(resultdir / 'trial_results.csv'):
                method = row['optimizer_type']
                canonical = resultdir / 'runs' / row['configuration_id'] / f"trial_{int(row['trial_number']):04d}" / 'data'
                metapath = canonical / 'raw_trace_metadata.json'
                errors = []
                if not metapath.exists():
                    errors.append('metadata_missing')
                else:
                    meta = load(metapath)
                    conditions = {'optimizer_parameters': params[suite][method], 'dt_s': .1,
                        'robot': {'shape': 'rectangle', 'length': .15, 'width': width['width_m']},
                        'scenario': {'maze_type': 'maze', 'optimizer_type': method, 'dynamics_type': 'differential_drive', 'path_planner': 'astar'},
                        'computation_time': {'scope': 'solve_nlp', 'clock': 'time.perf_counter', 'unit': 's',
                            'raw_samples': 'all_attempted_calls_including_failed_or_skipped_solves',
                            'csv_summary_samples': 'successful_solver_steps_including_first_step'},
                        'clearance': {'computed_online': False, 'sampling': 'initial_true_pose_and_each_completed_plant_integration'}}
                    for k, v in conditions.items():
                        if meta.get(k) != v:
                            errors.append(k)
                    half_width = width['width_m'] / 2
                    footprint = [[[-.075, -half_width], [.075, -half_width],
                                  [.075, half_width], [-.075, half_width]]]
                    if meta.get('geometry', {}).get('footprint_polygons') != footprint:
                        errors.append('footprint_centered_rectangle_rear_dist_zero')
                    perturb = meta.get('start_perturbation', {})
                    for k, v in cfg['start_perturbation'].items():
                        expected = int(row['start_seed']) if k == 'seed' else v
                        if perturb.get(k) != expected:
                            errors.append('start_perturbation.' + k)
                    if not (.001 <= perturb.get('initial_clearance', -1)):
                        errors.append('initial_clearance_below_acceptance')
                    if not (1 <= perturb.get('attempts', 0) <= 1000):
                        errors.append('rejection_attempts')
                    delta = perturb.get('delta_pose', [])
                    if len(delta) != 3 or any(abs(v) > bound + 1e-12 for v, bound in zip(delta, [.06, .06, .15])):
                        errors.append('perturbation_outside_three_sigma')
                metadata_rows.append({'suite': suite, 'width_id': width_id, 'optimizer_type': method,
                    'start_seed': row['start_seed'], 'metadata_path': str(metapath),
                    'fixed_conditions_passed': not errors, 'errors': ';'.join(errors)})
        check(suite, 'all_trial_metadata_fixed_conditions', sum(not r['fixed_conditions_passed'] for r in metadata_rows if r['suite'] == suite), 0)
    for name, values in [('environment', environments), ('optimizer_parameters', params), ('source_hashes', hashes)]:
        check('A_vs_B', name, values['A'], values['B'])
    for field in ['computation_time_scope', 'computation_time_unit', 'successful_solver_steps_only', 'include_first_step',
                  'use_debug', 'clearance_evaluation', 'benchmark_num_threads', 'python_executable', 'host_platform']:
        check('A_vs_B', field, manifests['A'][field], manifests['B'][field])

    # Read the initial rejected preflight; it is not a benchmark cohort.
    rejected = PROJECT / 'benchmark_runs/rectangle_don_100_20261105'
    rejected_validation = load(rejected / 'baseline_validation.json')
    expected_rejection = {'matched': False, 'differences': {'cc_sf': {'d_col': {'baseline': None, 'current': 5e-5}}}}
    check('initial_preflight', 'rejected_difference', rejected_validation, expected_rejection)
    check('initial_preflight', 'state_status', load(rejected / 'suite_state.json')['status'], 'error')
    check('initial_preflight', 'trial_results_files', len(list((rejected / 'widths').glob('**/trial_results.csv'))), 0)
    check('initial_preflight', 'resolved_parameters_retained', load(rejected / 'optimizer_parameters.json'), params['A'])
    rejected_hashes = load(rejected / 'source_hashes.json')
    changed_after_rejection = [k for k in hashes['A']['files'] if hashes['A']['files'].get(k) != rejected_hashes['files'].get(k)]
    check('initial_preflight', 'only_baseline_file_amended_after_rejection', changed_after_rejection, ['scripts/rectangle_benchmark_baseline.json'])
    baseline = load(roots['A'] / manifests['A']['baseline_parameters'])
    old_baseline = load(rejected / manifests['A']['baseline_parameters'])
    expected_amended_parameters = json.loads(encoded(old_baseline['parameters']))
    expected_amended_parameters['cc_sf']['d_col'] = 5e-5
    check('initial_preflight', 'baseline_parameter_change_only_d_col', baseline['parameters'], expected_amended_parameters)
    check('initial_preflight', 'historical_parameter_hashes_preserved', baseline['sources'], old_baseline['sources'])
    check('initial_preflight', 'approved_amendment_present', len(baseline.get('approved_amendments', [])), 1)
    historical_params = []
    for method, source in baseline['sources'].items():
        path = PROJECT / source['path']
        h = sha(path) if path.is_file() else None
        historical_params.append({'method': method, 'path': str(path), 'recorded_sha256': source['sha256'], 'actual_sha256': h, 'matched': h == source['sha256']})
        check('historical_baseline', f'parameter_source_hash:{method}', h, source['sha256'])
        if path.is_file():
            check('historical_baseline', f'original_baseline_values:{method}', numerical(load(path)), numerical(old_baseline['parameters'][method]))
    historical_root = PROJECT / 'benchmark_runs/rectangle_width070_130mm_step005_10trials_20261103'
    historical_sf = historical_root / 'source_snapshot/control/cc_sf_optimizer.py'
    current_sf = roots['A'] / 'source_snapshot/control/cc_sf_optimizer.py'
    historical_hash = load(historical_root / 'source_hashes.json')['control/cc_sf_optimizer.py']
    check('historical_source', 'cc_sf_snapshot_hash', sha(historical_sf), historical_hash)
    (out / 'provenance_cc_sf_historical_to_current.diff').write_text(''.join(difflib.unified_diff(
        historical_sf.read_text().splitlines(True), current_sf.read_text().splitlines(True),
        fromfile=str(historical_sf), tofile=str(current_sf))))

    # Check every reused origin, canonical NPZ + metadata bytes and filtered steps.
    for width in manifests['B']['widths']:
        reuse = width.get('reuse')
        if not reuse:
            continue
        a_dir = roots['A'] / reuse['source_results_dir']
        b_dir = roots['B'] / width['results_dir']
        check('reuse', 'source_suite_root', reuse['source_suite_root'], str(roots['A']))
        check('reuse', 'source_suite_manifest_sha256', reuse['source_suite_manifest_sha256'], sha(roots['A'] / 'suite_manifest.json'))
        check('reuse', 'source_trial_results_sha256', reuse['source_trial_results_sha256'], sha(a_dir / 'trial_results.csv'))
        check('reuse', 'source_aggregate_sha256', reuse['source_aggregate_sha256'], hashes['A']['aggregate_sha256'])
        check('reuse', 'seeds', reuse['seeds'], list(range(50)))
        a_cfg = dict(cfgs['A', reuse['source_width_id']]); b_cfg = dict(cfgs['B', width['width_id']])
        a_cfg.pop('trials'); b_cfg.pop('trials')
        check('reuse', 'source_target_config_except_trial_count', a_cfg, b_cfg)
        check('reuse', 'timing_scope', rms['A', reuse['source_width_id']]['computation_time'], rms['B', width['width_id']]['computation_time'])
        a_rows = {(r['optimizer_type'], r['start_seed']): r for r in read_csv(a_dir / 'trial_results.csv')}
        for b_row in read_csv(b_dir / 'trial_results.csv'):
            key = (b_row['optimizer_type'], b_row['start_seed'])
            a_row = a_rows.get(key)
            issues = []
            if a_row is None:
                issues.append('source_trial_missing')
                continue
            ignored = {'output_root_dir', 'raw_trace_path', 'raw_trace_metadata_path'}
            different = [k for k, v in a_row.items() if k not in ignored and b_row.get(k) != v]
            if different:
                issues.extend('trial_value:' + k for k in different)
            for key_name, expected in {'reused_from_suite': str(roots['A']), 'reused_from_trial_results': str(a_dir / 'trial_results.csv'),
                                       'reused_from_trial_number': a_row['trial_number']}.items():
                if b_row[key_name] != expected:
                    issues.append(key_name)
            if int(b_row['start_seed']) not in reuse['seeds']:
                issues.append('seed_out_of_range')
            rec = {'optimizer_type': key[0], 'start_seed': key[1], 'source_trial_number': a_row['trial_number'],
                   'target_trial_number': b_row['trial_number']}
            for name, col in [('npz', 'raw_trace_path'), ('metadata', 'raw_trace_metadata_path')]:
                source_path, target_path = Path(a_row[col]), Path(b_row[col])
                hs = sha(source_path) if source_path.is_file() else None
                ht = sha(target_path) if target_path.is_file() else None
                match = hs is not None and hs == ht
                if not match:
                    issues.append(name + '_hash')
                if not target_path.is_relative_to(roots['B']):
                    issues.append(name + '_outside_B')
                rec.update({name + '_source_path': str(source_path), name + '_target_path': str(target_path),
                            name + '_source_sha256': hs, name + '_target_sha256': ht, name + '_identical': match})
            rec.update({'passed': not issues, 'errors': ';'.join(issues)})
            reuse_rows.append(rec)
        for bpath in sorted(b_dir.glob('computation_times_*.csv')):
            a_steps = [r for r in read_csv(a_dir / bpath.name) if int(r['start_seed']) in reuse['seeds']]
            b_steps = read_csv(bpath)
            cleaned = [{k: v for k, v in r.items() if not k.startswith('reused_from_')} for r in b_steps]
            match = a_steps == cleaned
            step_rows.append({'source_path': str(a_dir / bpath.name), 'target_path': str(bpath),
                'source_file_sha256': sha(a_dir / bpath.name), 'target_file_sha256': sha(bpath),
                'selected_source_steps': len(a_steps), 'target_steps': len(b_steps), 'filtered_rows_identical': match})
            check('reuse', 'filtered_step_rows:' + bpath.name, match, True)
    check('reuse', 'reused_trial_count', len(reuse_rows), 200)
    check('reuse', 'reused_trial_mismatches', sum(not r['passed'] for r in reuse_rows), 0)

    mpcc_path = roots['A'] / 'source_snapshot/control/mpcc_optimizer.py'
    mpcc_tree = ast.parse(mpcc_path.read_text())
    mpcc_class = next(n for n in mpcc_tree.body if isinstance(n, ast.ClassDef) and n.name == 'MPCCOptimizer')
    solve = next(n for n in mpcc_class.body if isinstance(n, ast.FunctionDef) and n.name == 'solve_nlp')
    solve_source = ast.get_source_segment(mpcc_path.read_text(), solve)
    for method in METHODS:
        path = roots['A'] / f'source_snapshot/control/{method}_optimizer.py'
        tree = ast.parse(path.read_text())
        methods = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'solve_nlp']
        check('timing_source', method + ':solve_nlp_decorator',
              [[ast.unparse(d) for d in n.decorator_list] for n in methods], [['record_solve_nlp_time']])
    evidence = {
        'mpcc_source': str(mpcc_path), 'mpcc_source_sha256': sha(mpcc_path),
        'mpcc_solve_nlp_lines': [solve.lineno, solve.end_lineno],
        'mpcc_solve_nlp_source': solve_source,
        'mpcc_class_bases': [ast.unparse(n) for n in mpcc_class.bases],
        'mpcc_zero_input_safe_stop_or_backup_structure': 'absent; failure status updates warm-start state and returns AcadosSolution from same solver; controller returns solution input',
        'mpcc_counters': 'Metadata runtime statistics and CSV counters are null. Safe-stop/recovery are structural 0, not measured 0; no conclusion that actual inputs are nonzero or safe.',
        'cc_safe_stop_definition': 'Aggregate zero-input fallback control-step count; stage0_guard_stop_count subset where available; safe_stop_count-stage0_guard_stop_count identifies non-stage0 fallback count.',
        'cc_recovery_definition': 'backup_feasible_qp_success_count counts successful recovery; configured enable_backup_solver=false does not erase actual recovery counters.',
        'episode_limit': 'Aggregate counters and last_solve_mode do not identify episodes, longest run, or exact timing. Minimal solver_status_infos do not carry the complete per-step solve mode.',
        'historical_cc_sf_path': str(historical_sf), 'historical_cc_sf_sha256': sha(historical_sf),
        'current_cc_sf_sha256': sha(current_sf),
        'historical_cc_sf_constraints': 'One projected-variance row at stages 0..N-1; no terminal row.',
        'current_cc_sf_constraints': 'Two hard rows (PSDF-d_col and PSDF-risk_margin-d_min) at stages 1..N including terminal; OCP stage0 rows inactive; exact measured-pose PSDF guard before solve.',
        'baseline_interpretation': 'matched=true verifies all nonexcluded parameters against the approved amended baseline, never historical source/formula equivalence.',
        'cli_default_interpretation': 'run_manifest.cli contains parser defaults (e.g. simulation_time=60, maze_type=s_path, optimizer_type=psdf). Stored generated config and configurations override in batch mode: test_nmpc.py lines 499-527; effective simulation_time is 80 s.',
        'sequential_execution_evidence': 'Suite source executes widths via blocking subprocess.run and test_nmpc iterates configurations; A is the completed source for B reuse. External-process load/temperature/power are unmeasured.',
        'timing_limit': 'Same recorded environment and source are verified. CPU model, GPU driver version, acados version/commit and thermal/power/concurrent process telemetry are absent; interop threads recorded as 24 while compute thread settings are 6.',
    }
    (out / 'provenance_source_evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
    write_csv(out / 'provenance_checks.csv', checks)
    write_csv(out / 'provenance_source_hashes.csv', hash_rows)
    write_csv(out / 'provenance_configs.csv', config_rows)
    write_csv(out / 'provenance_trial_conditions.csv', metadata_rows)
    write_csv(out / 'provenance_reuse_trials.csv', reuse_rows)
    write_csv(out / 'provenance_reuse_steps.csv', step_rows)
    write_csv(out / 'provenance_historical_parameter_sources.csv', historical_params)
    failures = [r for r in checks if not r['passed']]
    write_csv(out / 'provenance_mismatches.csv', failures, ['scope', 'check', 'passed', 'observed', 'expected'])
    summary = {'inputs': {s: str(r) for s, r in roots.items()}, 'analysis_code': str(Path(__file__).resolve()),
               'analysis_code_sha256': sha(__file__), 'source_aggregate_sha256': hashes['A']['aggregate_sha256'],
               'created_at_kst': {s: m['created_at_kst'] for s, m in manifests.items()},
               'source_file_count_per_suite': {s: len(h['files']) for s, h in hashes.items()},
               'checks': len(checks), 'failed_checks': len(failures),
               'fixed_condition_metadata_trials': len(metadata_rows),
               'fixed_condition_metadata_failures': sum(not r['fixed_conditions_passed'] for r in metadata_rows),
               'reused_trial_count': len(reuse_rows), 'reused_trial_failures': sum(not r['passed'] for r in reuse_rows),
               'reused_success_step_rows': sum(r['target_steps'] for r in step_rows),
               'approved_amendments': baseline['approved_amendments'],
               'initial_rejected_preflight': {'root': str(rejected), 'validation': rejected_validation,
                                             'changed_snapshot_files_after_rejection': changed_after_rejection},
               'historical_source_equivalence': False, 'environment': environments['A'], 'failures': failures}
    (out / 'provenance_audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    notes = f'''# Provenance 및 고정 조건 audit\n\nA/B는 각 {len(hashes['A']['files'])}개 snapshot 파일의 실제 SHA-256을 재계산했으며, 저장된 per-file/aggregate/manifest hash가 일치한다. 공통 aggregate는 `{hashes['A']['aggregate_sha256']}`이다. 환경·resolved optimizer parameters·timing scope가 A/B에서 정확히 일치한다. {len(config_rows)}개 생성 YAML/run manifest와 {len(metadata_rows):,}개 trial metadata의 고정 조건을 확인했다. Audit {len(checks):,}건 중 불일치 {len(failures)}건, trial 조건 불일치 {summary['fixed_condition_metadata_failures']}건이다.\n\nB DoN .75의 200건은 A seed 0–49에서 재사용됐다. 원본 suite/trial CSV hash와 모든 NPZ/metadata 쌍의 바이트 SHA-256, 결과·pose·seed·geometry·parameters·timing·환경이 동일하고 B 내부 canonical 경로에 존재한다. 필터링된 성공 step {summary['reused_success_step_rows']:,}행도 원본과 동일하다. 이들은 200개의 새로운 독립 관측이 아니다.\n\n최초 디렉토리 `rectangle_don_100_20261105`는 preflight에서 CC-SF `d_col`이 과거 baseline에 없고 현재 0.00005여서 중단됐다. 실제 trial CSV는 0개다. 이후 snapshot 변경 파일은 baseline JSON 한 개뿐이며, resolved parameters는 최초와 현재가 동일하다. `approved_amendments`의 사용자 승인은 현재 CC-SF 유지와 이 기준 변경을 기록한다. 그 외 비진단 파라미터 검사는 유지됐고 amended baseline과 일치한다. 과거 네 parameter 원본 파일 SHA-256도 기록과 일치한다.\n\n과거 CC-SF source와 현재 source는 다르다. 2026-11-03은 stage 0..N-1에 projected-variance 행 한 개와 terminal 제약 없음이었고, 현재는 stage 1..N에 hard PSDF guard 및 projected-variance 행 두 개, terminal 제약, measured-pose의 별도 stage-0 guard를 사용한다. 따라서 baseline 일치는 과거 제어 수식·source 일치가 아니다. 실제 unified diff를 보존했다.\n\nMPCC `solve_nlp()`는 주 solver 실패 시 warm-start 상태를 갱신하고 같은 solver의 `AcadosSolution`을 반환하며 CC의 zero-input safe-stop/backup 구현이 없다. Runtime 카운터는 null이므로 safe-stop/recovery를 구조적 0으로만 설명할 수 있다. CC safe-stop은 fallback 제어 step 수이고 독립 위험 episode 수가 아니다. Aggregate 카운터 및 마지막 모드만으로 episode·최장 연속 길이·정확한 시점은 복원할 수 없다.\n\nRun manifest `cli`의 60 s, s_path, psdf는 batch 실행에 사용되지 않은 parser 기본값이다. 저장 config/configurations와 `test_nmpc.py` override 경로에서 실제 80 s, maze 및 각 optimizer가 확인된다. Debug risk 보조 flag가 일부 true여도 master `use_debug=false`이며 metadata 전수 일치한다.\n\n실행일은 A {manifests['A']['created_at_kst']}, B {manifests['B']['created_at_kst']}이다. 기록 환경은 Python 3.10.12, NumPy 1.24.4, Torch 2.4.1+cu121, CUDA 12.1, RTX 4090, CasADi 3.7.1, compute thread 6, torch interop thread 24다. 동일 recorded 환경은 확인되지만 CPU 모델·GPU driver·acados revision과 온도·전력·다른 프로세스 부하는 기록되지 않아 wall-time 변동을 배제할 수 없다. 주행/solver를 재실행하지 않았고 원본 파일을 변경하지 않았다.\n'''
    (out / 'provenance_notes.md').write_text(notes)
    print(json.dumps({k: summary[k] for k in ['checks', 'failed_checks', 'fixed_condition_metadata_trials', 'fixed_condition_metadata_failures', 'reused_trial_count', 'reused_trial_failures', 'reused_success_step_rows']}, indent=2))
    if failures:
        print('See provenance_mismatches.csv')


if __name__ == '__main__':
    main()
