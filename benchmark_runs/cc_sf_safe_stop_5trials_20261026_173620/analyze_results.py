"""Cross-check stop events against plant histories and validate all five videos."""
import csv
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


summaries = read_csv(ROOT / 'safe_stop_summary.csv')
cycles = read_csv(ROOT / 'solver_cycles.csv')
trials = read_csv(ROOT / 'trial_results.csv')
assert len(summaries) == len(trials) == 5
checks = []
episodes = []
for summary, trial in zip(summaries, trials):
    number = int(summary['trial_number'])
    rows = [row for row in cycles if int(row['trial_number']) == number]
    histories = list((Path(trial['output_root_dir']) / 'data').glob('trial_history_*.csv'))
    assert len(histories) == 1
    history = read_csv(histories[0])
    assert len(rows) == len(history) == int(summary['total_cycles'])
    assert int(summary['start_seed']) == number - 1
    assert summary['counter_matches'] == 'True'
    assert summary['safe_stop_all_inputs_zero'] == 'True'
    for row, plant in zip(rows, history):
        assert row['timestep'] == plant['timestep']
        assert row['final_status'] == plant['solver_status_code']
        assert abs(float(row['time_after_s']) - float(plant['time'])) < 1e-8
        assert abs(float(row['v']) - float(plant['v'])) < 1e-12
        assert abs(float(row['omega']) - float(plant['omega'])) < 1e-12
        assert bool(int(row['safe_stop'])) == (row['solve_mode'] == 'safe_stop')
    stops = [row for row in rows if row['safe_stop'] == '1']
    assert len(stops) == int(summary['safe_stop_count'])
    for row in stops:
        assert float(row['v']) == float(row['omega']) == float(row['virtual_input']) == 0
    groups = []
    for row in stops:
        if not groups or int(row['timestep']) != int(groups[-1][-1]['timestep']) + 1:
            groups.append([])
        groups[-1].append(row)
    assert len(groups) == int(summary['safe_stop_episodes'])
    for index, group in enumerate(groups, 1):
        episodes.append({
            'trial_number': number, 'start_seed': summary['start_seed'],
            'episode': index, 'first_timestep': group[0]['timestep'],
            'last_timestep': group[-1]['timestep'],
            'start_time_s': float(group[0]['time_before_s']),
            'end_time_s': float(group[-1]['time_after_s']),
            'safe_stop_steps': len(group),
            'main_status_codes': json.dumps(sorted({int(row['main_status']) for row in group})),
            'backup_status_codes': json.dumps(sorted({int(row['backup_status']) for row in group if row['backup_status']})),
            'final_status_codes': json.dumps(sorted({int(row['final_status']) for row in group})),
        })
    video = Path(summary['animation_path'])
    assert video.is_file() and video.stat().st_size > 0
    info = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,width,height,nb_frames,r_frame_rate:format=duration',
        '-of', 'json', str(video),
    ], text=True))
    assert int(info['streams'][0]['nb_frames']) == len(rows)
    assert abs(float(info['format']['duration']) - len(rows) / 10) < 0.05
    decoded = subprocess.run([
        'ffmpeg', '-v', 'error', '-xerror', '-i', str(video), '-f', 'null', '-'
    ], capture_output=True, text=True)
    assert decoded.returncode == 0, decoded.stderr
    checks.append({
        'trial_number': number, 'history_matches_every_cycle': True,
        'runtime_counter_matches': True, 'zero_control_at_every_safe_stop': True,
        'video_decoded': True, 'video': str(video), 'video_info': info,
    })

event_fields = [
    'trial_number', 'start_seed', 'episode', 'first_timestep', 'last_timestep',
    'start_time_s', 'end_time_s', 'safe_stop_steps', 'main_status_codes',
    'backup_status_codes', 'final_status_codes',
]
with (ROOT / 'safe_stop_episodes.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=event_fields)
    writer.writeheader()
    writer.writerows(episodes)
with (ROOT / 'safe_stop_events.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(cycles[0]))
    writer.writeheader()
    writer.writerows(row for row in cycles if row['safe_stop'] == '1')
totals = {
    key: sum(int(row[key]) for row in summaries) for key in (
        'total_cycles', 'main_nonzero_status_count', 'backup_recovery_count',
        'safe_stop_count', 'safe_stop_episodes', 'safe_stop_status_zero_count',
        'final_status_zero_count',
    )
}
totals['trials_with_safe_stop'] = sum(int(row['safe_stop_count']) > 0 for row in summaries)
totals['successful_trials'] = sum(row['status'] == 'success' for row in summaries)
totals['safe_stop_duration_s'] = sum(float(row['safe_stop_duration_s']) for row in summaries)
environment = json.loads((ROOT / 'environment.json').read_text())
assert environment['sources_changed_during_run'] == []
validation = {'all_checks_passed': True, 'totals': totals, 'trial_checks': checks}
(ROOT / 'validation.json').write_text(json.dumps(validation, indent=2))
print(json.dumps({'totals': totals, 'trials': summaries}, indent=2))
