"""P058 helper: compute WimPyDD (E, v) kernels for given (tag, m, delta) configurations into output/work/P058/cache.
Same grids and Hamiltonians as P058_exothermic.py; writes are atomic (temp file + os.replace) so several workers may run
in parallel with the main script.   Usage:  .venv/bin/python output/code/P058_kernels_worker.py iso:4000:300 iso:4000:350 ..."""
import sys, os, math, time
sys.path.insert(0, 'output/code')
import numpy as np
from common import lzcommon as lz

CACHE = 'output/work/P058/cache'; os.makedirs(CACHE, exist_ok=True)
MV = lz.M_V_GEV; C_UNIT = 1.0 / MV ** 2
GF = 1.1663788e-5; SW2 = 0.2312
C_P_HIG = (GF / math.sqrt(2)) * (1 - 4 * SW2); C_N_HIG = -(GF / math.sqrt(2))
WD = lz.wd()
VGRID = np.linspace(0.0, 844.0, 1200); ONES = np.ones_like(VGRID)
HAMS = {'iso': lz.wd_hamiltonian('P058w_iso', {1: (2 * C_UNIT, 0.0)}), 'p': lz.wd_hamiltonian('P058w_p', {1: (C_UNIT, C_UNIT)}),
        'hig': lz.wd_hamiltonian('P058w_hig', {1: (C_P_HIG + C_N_HIG, C_P_HIG - C_N_HIG)})}
E_EXO = np.concatenate([np.arange(1.0, 700.0, 2.5), np.arange(700.0, 2505.0, 5.0)])
E_ENDO = np.arange(1.0, 452.0, 2.5)

def compute(tag, m, delta, E):
    fn = f'{CACHE}/K_{tag}_{int(m)}_{"m" if delta < 0 else "p"}{int(round(abs(delta)))}.npz'
    if os.path.exists(fn):
        print('  exists:', fn, flush=True); return
    t = time.time()
    K = np.array([WD.diff_rate(WD.Xe, HAMS[tag], m, float(e), VGRID, ONES, j_chi=0.5, delta=float(delta), sum_over_streams=False)
                  for e in E]) * 1000.0 * 365.25
    tmp = fn + f'.tmp{os.getpid()}.npz'
    np.savez_compressed(tmp, E=E, K=K); os.replace(tmp, fn)
    print(f'  wrote {fn} ({time.time() - t:.0f} s)', flush=True)

for item in sys.argv[1:]:
    tag, m, d = item.split(':'); m = float(m); d = float(d)
    compute(tag, m, -d, E_EXO); compute(tag, m, +d, E_ENDO)
print('worker done', flush=True)
