"""Seven-repeat single-process Python timing on one saved input prefix."""
from __future__ import annotations

import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'

import ctypes
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from adaptive_core import V12Config
from run import make_controllers


HERE=Path(__file__).resolve().parent
FOLDER=HERE/'outputs'/'selection_e1_v500'
SOURCE=FOLDER/'caseE1_run00.npz'
SAMPLES=20000


class MemoryCounters(ctypes.Structure):
    _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong),
              ('PeakWorkingSetSize',ctypes.c_size_t),('WorkingSetSize',ctypes.c_size_t),
              ('QuotaPeakPagedPoolUsage',ctypes.c_size_t),
              ('QuotaPagedPoolUsage',ctypes.c_size_t),
              ('QuotaPeakNonPagedPoolUsage',ctypes.c_size_t),
              ('QuotaNonPagedPoolUsage',ctypes.c_size_t),
              ('PagefileUsage',ctypes.c_size_t),('PeakPagefileUsage',ctypes.c_size_t)]


def memory_bytes():
    counter=MemoryCounters()
    counter.cb=ctypes.sizeof(counter)
    get_process=ctypes.windll.kernel32.GetCurrentProcess
    get_process.restype=ctypes.c_void_p
    query=ctypes.windll.psapi.GetProcessMemoryInfo
    query.argtypes=[ctypes.c_void_p,ctypes.POINTER(MemoryCounters),ctypes.c_ulong]
    query.restype=ctypes.c_int
    if not query(get_process(),ctypes.byref(counter),counter.cb):
        raise OSError('GetProcessMemoryInfo failed')
    return int(counter.WorkingSetSize),int(counter.PeakWorkingSetSize)


def main():
    with np.load(SOURCE) as f:
        x=f['x'][:SAMPLES].copy();d=f['d'][:SAMPLES].copy()
        v=f['v'][:SAMPLES].copy();Om=f['Om'].copy();ph=f['ph'].copy()
    protocol=json.loads((FOLDER/'protocol.json').read_text(encoding='utf-8'))
    cfg=V12Config(**protocol['config'])
    seed=protocol['seed']
    methods=('fixed_R4','adaptive')
    all_times={name:[] for name in methods}
    # First pass warms Python and numerical libraries. The next seven passes
    # use fresh controllers and the same already-saved input, in one process.
    for repeat in range(8):
        for name in methods:
            controllers,_,_,_=make_controllers('E1',0,seed,cfg,'essential')
            ctrl=controllers[name]
            xb=np.zeros((1,20));zh=np.zeros((4,1,500))
            t0=perf_counter()
            for n in range(SAMPLES):
                xb[:,1:]=xb[:,:-1].copy();xb[0,0]=x[n]
                z=np.sqrt(2/500)*np.cos(np.einsum('rdm,rm->rd',Om,xb)+ph)
                zh[1:]=zh[:-1].copy();zh[0]=z
                q=zh[2]+.5*zh[3]
                ctrl.step(z,q,np.array([d[n]+v[n]]))
            elapsed=perf_counter()-t0
            if repeat:
                all_times[name].append(elapsed)
    working,peak=memory_bytes()
    result={'phase':'dev','source':str(SOURCE),'samples_per_repeat':SAMPLES,
            'warmup_repeats':1,'measured_repeats':7,
            'single_process':True,'numeric_threads_requested':1,
            'timing_includes_shared_rff_feature_generation':True,
            'does_not_measure_structural_transition_peak':True,
            'times_seconds':all_times,
            'median_us_per_sample':{n:float(np.median(t)*1e6/SAMPLES)
                                    for n,t in all_times.items()},
            'iqr_us_per_sample':{n:float((np.quantile(t,.75)-np.quantile(t,.25))*1e6/SAMPLES)
                                 for n,t in all_times.items()},
            'process_working_set_bytes_at_end':working,
            'process_peak_working_set_bytes':peak}
    (HERE/'RUNTIME_DEV.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('median_us_per_sample','iqr_us_per_sample',
                                            'process_peak_working_set_bytes')},indent=2))


if __name__=='__main__':
    main()
