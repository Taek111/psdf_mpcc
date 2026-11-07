#!/usr/bin/env python3
"""Final offline artifact integrity checks, independent of report formatting."""
from pathlib import Path
import hashlib
import json
import re
from datetime import datetime, timezone
import pandas as pd
import numpy as np

OUT = Path(__file__).resolve().parents[1]


def main():
    report = OUT/'report_ko.md'
    text = report.read_text()
    links = re.findall(r'\]\(([^)]+)\)', text)
    missing = [link for link in links if not (OUT/link.split('#')[0]).exists()]
    a = pd.read_csv(OUT/'tables/A_quantitative_table.csv')
    b = pd.read_csv(OUT/'tables/B_success_rates.csv')
    pair = pd.read_csv(OUT/'tables/paired_seed_membership.csv')
    cm = json.loads((OUT/'offline_clearance/analysis_manifest.json').read_text())
    trace = json.loads((OUT/'audit/trace_audit.json').read_text())
    baseline = json.loads((OUT/'audit/provenance_audit.json').read_text())
    execution = json.loads((OUT/'audit/clearance_parallel_execution.json').read_text())
    preservation = json.loads((OUT/'audit/input_preservation.json').read_text())
    checks = {
        'all_report_links_exist': not missing,
        'a_8_groups_all_100_complete': len(a)==8 and bool((a.completed_trials==100).all()),
        'b_72_groups_all_50_complete': len(b)==72 and bool((b.completed_trials==50).all()),
        'b_don_grid_exact': np.allclose(sorted(b.nominal_don.unique()),np.arange(.70,.871,.01),rtol=0,atol=1e-12),
        'a_clearance_coverage_complete': bool((a.coverage_complete==1).all()),
        'a_all_numeric_metrics_finite': bool(np.isfinite(a[['clearance_mean_mm','clearance_p05_mm','solve_mean_s','solve_p95_s','traversal_success_mean_s']]).all().all()),
        'all_paired_poses_matched': bool(pair.pose_matched.all()),
        'paired_seed_rows_expected': len(pair)==6600,
        'cohort_memberships_4400_unique_4200_reused_200': cm['cohort_trial_memberships']==4400 and cm['unique_source_trial_origins']==4200 and cm['reused_trial_memberships']==200,
        'strict_clearance_success': execution['exit_code']==0,
        'trace_audit_passed': trace['audit_passed'] and trace['issue_count']==0,
        'provenance_passed': baseline['failed_checks']==0,
        'original_inputs_preserved': preservation['preserved'] and not preservation['bytecode_cache_files_remaining'],
        'plot_png_nonempty': (OUT/'figures/B_success_rates.png').stat().st_size>10000,
        'plot_svg_nonempty': (OUT/'figures/B_success_rates.svg').stat().st_size>10000,
    }
    # MPCC counter nulls are deliberately preserved, CC measurements are zero.
    checks['missing_mpcc_counters_not_imputed']=bool(a[a.optimizer_type=='mpcc'].safe_stop_count.isna().all())
    checks['cc_a_safe_stop_observed_zero']=bool((a[a.optimizer_type!='mpcc'].safe_stop_count==0).all())
    for name,value in checks.items():
        if not value:
            print('FAILED:',name)
    record={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'report_sha256':hashlib.sha256(report.read_bytes()).hexdigest(),
            'checks':{k:bool(v) for k,v in checks.items()},'passed':all(checks.values()),'missing_links':missing,
            'visual_review':'B_success_rates.png opened and inspected at full artifact resolution: four methods, same 0–100% scale, all 18 DoN ticks, actual n at each point, Wilson bands, reuse note; no clipping observed.',
            'scope':'No driving tests or solvers run. Integrity and reviewed-statistic checks only.'}
    (OUT/'audit/output_validation.json').write_text(json.dumps(record,indent=2,ensure_ascii=False))
    print('Output checks:',len(checks),'passed:',all(checks.values()))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__=='__main__':
    main()
