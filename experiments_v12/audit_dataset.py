"""Audit selected saved runs without changing experiment records."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


HERE=Path(__file__).resolve().parent
OUTPUTS=HERE/'outputs'
GROUPS={
    'E1': ('selection_e1_v500','selection_e1_t1000_v500','selection_e1_growth'),
    'E2': ('main_dev_round1','main_dev_aggressive_growth','main_dev_continue'),
    'E3': ('main_dev_round1','main_dev_aggressive_growth','main_dev_continue')
}


def audit():
    problems=[]
    different_length=[]
    checked=0
    max_ledger=0.
    max_physical=0.
    for case, folders in GROUPS.items():
        base=None
        for folder_name in folders:
            folder=OUTPUTS/folder_name
            protocol=json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
            for rel,wanted in protocol['sources_sha256'].items():
                saved=folder/'sources'/rel
                if not saved.is_file() or hashlib.sha256(saved.read_bytes()).hexdigest()!=wanted:
                    problems.append(f'{folder_name}: source snapshot mismatch {rel}')
            n=protocol['runs']
            for run in range(n):
                path=folder/f'case{case}_run{run:02d}.npz'
                with np.load(path) as f:
                    paired=tuple(f[k].copy() for k in
                                 ('x','d','v','Om','ph','initial_A','initial_B'))
                    ledger=np.max(np.abs(f['total_mults']-f['core_mults']-
                                         f['candidate_mults']-f['management_mults']))
                    max_ledger=max(max_ledger,float(ledger))
                    y=f['actuator_output']
                    physical=np.zeros_like(y)
                    physical[2:]=y[:-2]
                    physical[3:]+=.5*y[:-3]
                    method=json.loads(str(f['meta']))['controllers'].index('adaptive')
                    physical_error=np.max(np.abs(physical-f['secondary_output'][method]))
                    max_physical=max(max_physical,float(physical_error))
                    if not np.isfinite(f['anr']).all():
                        problems.append(f'{folder_name}: nonfinite ANR run {run}')
                    if ledger!=0 or physical_error>1e-5:
                        problems.append(f'{folder_name}: ledger/FIR run {run}')
                if folder_name==folders[0]:
                    if base is None:
                        base=[]
                    base.append(paired)
                else:
                    if len(paired[0])!=len(base[run][0]):
                        different_length.append(f'{case} run {run}: {folder_name}')
                    elif not all(np.array_equal(a,b) for a,b in zip(paired,base[run])):
                        problems.append(f'{case} run {run}: paired inputs differ in {folder_name}')
                checked+=1
    result={'checked_run_files':checked,'paired_groups':GROUPS,
            'max_ledger_difference':max_ledger,'max_physical_fir_difference':max_physical,
            'different_length_not_paired':different_length,
            'problems':problems,'passed':not problems}
    (HERE/'DATA_AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    if problems:
        raise SystemExit(1)


if __name__=='__main__':
    audit()
