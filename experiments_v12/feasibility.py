"""Describe the fixed-rank capacity/cost diagnostic from saved development sweeps."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE / 'outputs' / 'main_dev_round1'


def main():
    rows = []
    segment_rows = []
    for case in ('E2', 'E3'):
        for run in range(2):
            path = ROOT / f'case{case}_run{run:02d}_fixed_sweep.json'
            data = json.loads(path.read_text(encoding='utf-8'))
            means = np.asarray(data['segment_anr']).mean(axis=0)
            best = {}
            for rank in (2, 3, 4):
                choices = [(float(means[j]), data['names'][j])
                           for j in range(len(means))
                           if data['names'][j].startswith(f'R{rank}_')]
                best[rank] = min(choices)
            rows.append((case, run, best))
            for segment, values in enumerate(np.asarray(data['segment_anr']), 1):
                segment_best = {}
                for rank in (2, 3, 4):
                    segment_best[rank] = min(float(values[j]) for j, name in enumerate(data['names'])
                                             if name.startswith(f'R{rank}_'))
                segment_rows.append((case, run, segment,
                                     segment_best[2]-segment_best[4],
                                     segment_best[3]-segment_best[4]))
    lines = [
        '# 固定项数的容量—成本可行性诊断', '',
        '以下仅使用已查看的开发集前两次运行，不用于独立显著性判断。',
        '每个 R 在原步长集合中选该运行四段等权 ANR 最佳值，因此属于乐观的逐运行选参诊断。', '',
        '固定 R=4 的完整成本为 19132 次乘法/采样；10% 节省门槛为 17218.8。',
        '固定 R=3 的成本为 17477，单靠长时间停留在 R=3 仍不足以达到成本门槛；',
        '动态方法还需承担候选和管理费用。', '',
        '| 工况 | 运行 | 最佳 R2 ANR | 最佳 R3 ANR | 最佳 R4 ANR | R2−R4 | R3−R4 |',
        '|---|---:|---:|---:|---:|---:|---:|'
    ]
    for case, run, best in rows:
        lines.append(f'| {case} | {run} | {best[2][0]:.3f} ({best[2][1]}) | '
                     f'{best[3][0]:.3f} ({best[3][1]}) | '
                     f'{best[4][0]:.3f} ({best[4][1]}) | '
                     f'{best[2][0]-best[4][0]:+.3f} | '
                     f'{best[3][0]-best[4][0]:+.3f} |')
    lines += ['', '逐段乐观选参后的容量差（每段分别选择步长，不是可实施策略）：', '',
              '| 工况 | 运行 | 段 | R2−R4 dB | R3−R4 dB |',
              '|---|---:|---:|---:|---:|']
    for case, run, segment, gap2, gap3 in segment_rows:
        lines.append(f'| {case} | {run} | {segment} | {gap2:+.3f} | {gap3:+.3f} |')
    lines += ['', '这些数据说明现有 E2/E3 设计中低项数与高项数存在明显质量差，'
               '但不构成目标不可达的数学证明。后续必须用独立选择集及完整成本检验候选。']
    (HERE / 'FEASIBILITY_DIAGNOSTIC.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
