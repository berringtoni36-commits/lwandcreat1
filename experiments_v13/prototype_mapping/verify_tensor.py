"""Audit development provenance, map tensors, costs, and saved conclusions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import tensor_fourier_sweep as tf
import full_tensor_check as fc

HERE=Path(__file__).resolve().parent


def main():
    tf.check_batched_controllers()
    fc.check_batched_full_update()
    output=json.loads((HERE/'tensor_fourier_results.json').read_text(encoding='utf-8'))
    full=json.loads((HERE/'full_tensor_results.json').read_text(encoding='utf-8'))
    np.testing.assert_array_equal(output['cost_nominal'],
                                  [2985+1655*r for r in range(1,9)])
    assert len(output['analyses'])==len(full['rows'])==2
    for case,analysis,full_row,map_record in zip(tf.CASES,output['analyses'],
                                                   full['rows'],output['map_parameter_files']):
        assert analysis['case']==full_row['case']==case
        assert map_record==tf.save_map_parameters(case)
        row=json.loads((HERE/f'tensor_{case}_run00.json').read_text(encoding='utf-8'))
        assert np.asarray(row['segment_anr']).shape==(4,8,5)
        assert np.isfinite(row['segment_anr']).all()
        known=tf.provenance(case)
        assert all(row['provenance'][key]==known[key] for key in known)
        assert len(analysis['zero_overhead_mean_loss_pareto_frontier'])>0
        assert analysis['oracle_joint_feasible']
        quality=analysis['oracle_lowest_cost_meeting_margin']
        assert quality['worst_segment_delta_db']<=.5
        assert quality['cost_ratio_nominal']<=.9
        assert abs(analysis['nominal_budget_headroom_after_quality_oracle']
                   -(.9*tf.COST_NOMINAL[3]-quality['mean_mults_nominal']))<1e-9
        old_r4=np.mean(analysis['old_mapping_R4_context']['segment_anr'])
        assert abs(full_row['full_minus_old_R4_db']
                   -(full_row['mean_anr']-old_r4))<1e-12
        assert full_row['full_minus_old_R4_db']>3
        print(f"{case}: quality-cost={quality['cost_ratio_nominal']:.4f}, "
              f"full-vs-oldR4={full_row['full_minus_old_R4_db']:+.3f} dB")
    assert '当前候选映射不宜推进' in (HERE/'tensor_fourier_report.md').read_text(encoding='utf-8')
    print('development-only tensor mapping verification passed')


if __name__=='__main__':
    main()
