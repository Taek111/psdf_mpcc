"""Read existing diagnostics to audit margin changes after a zero-input stop.

No solver or navigation run is performed. The covariance map is evaluated
numerically using the frozen experimental source.
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source_snapshot'))
from control.cc_covariance import covariance_from_log_cholesky, propagate_log_cholesky
from control.cc_sf_optimizer import CCSFOptimizer, CCSFOptimizerParam


def main():
    param = CCSFOptimizerParam()
    optimizer = CCSFOptimizer()
    optimizer.param = param
    optimizer.N = 20
    with (ROOT / 'width_110mm/nominal_risk.jsonl').open() as stream:
        records = [json.loads(line) for line in stream]
    results = []
    for seed in range(3):
        first = min(row['cycle'] for row in records if row['seed'] == seed
                    and row['kind'] == 'backup')
        attempts = [row for row in records if row['seed'] == seed and row['cycle'] >= first]
        first_fast = next(row for row in attempts if row['cycle'] == first and row['kind'] == 'fast')
        zero_fast = next(row for row in attempts if row['cycle'] == first + 1 and row['kind'] == 'fast')
        xi0 = np.array(first_fast['states'][0][4:8])
        xi_after_zero = propagate_log_cholesky(xi0, 0., 0., 0., .1, param)
        np.testing.assert_array_equal(xi0, xi_after_zero)
        np.testing.assert_array_equal(zero_fast['inputs'], np.zeros((20, 3)))
        states = np.asarray(zero_fast['states'])
        physical_and_covariance = [0, 1, 2, 4, 5, 6, 7]
        np.testing.assert_array_equal(states[:, physical_and_covariance],
                                      np.tile(states[0, physical_and_covariance], (21, 1)))
        assert all(row['phi_m'][0] == first_fast['phi_m'][0] for row in attempts)
        assert all(row['risk_margin_m'][0] == first_fast['risk_margin_m'][0] for row in attempts)
        assert all(row['nominal_residual_m'][0] == first_fast['nominal_residual_m'][0] for row in attempts)
        assert all(row['status'] == (4 if row['kind'] == 'fast' else 2) for row in attempts)
        results.append({
            'seed': seed, 'first_stop_cycle': first, 'first_stop_time_s': first * .1,
            'phi_current_mm': first_fast['phi_m'][0] * 1000,
            'current_margin_before_zero_mm': first_fast['risk_margin_m'][0] * 1000,
            'current_margin_after_zero_mm': zero_fast['risk_margin_m'][0] * 1000,
            'stage1_nominal_margin_before_zero_mm': first_fast['risk_margin_m'][1] * 1000,
            'stage1_nominal_margin_after_zero_mm': zero_fast['risk_margin_m'][1] * 1000,
            'stage10_nominal_margin_before_zero_mm': first_fast['risk_margin_m'][10] * 1000,
            'stage10_nominal_margin_after_zero_mm': zero_fast['risk_margin_m'][10] * 1000,
            'current_residual_mm': first_fast['nominal_residual_m'][0] * 1000,
            'available_margin_phi_minus_dmin_mm': (first_fast['phi_m'][0] - param.d_min) * 1000,
            'initial_covariance_body_frame': covariance_from_log_cholesky(xi0).tolist(),
            'zero_input_covariance_map_difference': (xi_after_zero - xi0).tolist(),
            'current_phi_and_margin_unchanged_through_all_stop_attempts': True,
            'stop_fast_attempts': sum(row['kind'] == 'fast' for row in attempts),
            'stop_backup_attempts': sum(row['kind'] == 'backup' for row in attempts),
            'fast_status': 4, 'backup_status': 2,
        })
    output = {
        'scope': '0.110 m seeds 0–2; existing saved nominal/returned diagnostics only; no new solver run',
        'configured_sigmas_m_m_rad': [param.sigma_f0, param.sigma_l0, param.sigma_psi0],
        'configured_stationary_process_noise': [param.q_f0, param.q_l0, param.q_psi0],
        'risk_margin_floor_parameter_m': param.risk_margin_floor,
        'results': results,
        'interpretation': 'Zero input removes future control-dependent covariance growth. It does not remove the nonzero configured initial covariance fixed at stage 0 each cycle. The hard initial safety row therefore remains negative and no future control can repair it.',
    }
    (ROOT / 'stop_margin_audit.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
