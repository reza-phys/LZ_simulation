#!/usr/bin/env python
"""P075 helper: live WimPyDD grids (annual-mean Baxter halo, LZ efficiency, 2.84 t yr) cached to output/work/P075/.
  N_unit_grid_live.csv     isoscalar (c_p = c_n = 1/m_v^2) and proton-only (c_p = 1/m_v^2) unit-coupling counts for
                           m = 500, 2000, 5000, 10000 GeV, delta = 250-390 keV (P011 covers 300/1000/3000 GeV)
  higgsino_grid_live.csv   pure-Higgsino Z-exchange counts (P007 couplings c0_wd = -7.618e-6, c1_wd = 8.871e-6 GeV^-2)
                           for m = 5000, 10000 GeV (P007 covers 300-4000 GeV), plus 1000 GeV check points.
Recipe identical to P011/P054 (validated: reproduces the P011 grid to 1e-4 at 1 TeV).  Run from the simulation root.
"""
import os, sys, math, time, json
import numpy as np, pandas as pd
from scipy.special import erf, erfc
sys.path.insert(0, 'output/code')
from common import lzcommon as lz

OUT = 'output/work/P075'; os.makedirs(OUT, exist_ok=True)
MV = lz.M_V_GEV
VGRID = np.linspace(0.0, 844.0, 1200)
days12 = 15.0 + 365.25 / 12 * np.arange(12)
deta_annual = np.mean([lz.wd_halo(day_of_year=d, vmin=VGRID)[1] for d in days12], axis=0)
E50_HI, E50_LO, PLATEAU = lz.LZ['E_50pct_high_keV'], lz.LZ['E_50pct_low_keV'], lz.LZ['eff_plateau']
def efficiency(E, sig_hi=11.5, sig_lo=2.5):
    return PLATEAU * 0.5 * (1 + erf((E - E50_LO) / (math.sqrt(2) * sig_lo))) * 0.5 * erfc((E - E50_HI) / (math.sqrt(2) * sig_hi))
Ef = np.linspace(1.0, 330.0, 3001)
def count(ham, m, d):
    lo = lz.E_R_range_keV(m, VGRID[-1], A=124.0, delta_kev=d)[0]
    if not np.isfinite(lo) or lo > 330.0:
        return 0.0
    E = np.arange(max(1.0, math.floor(lo) - 4.0), 330.0 + 1e-9, 3.0)
    r = lz.wd_rate(ham, m, E, halo=(VGRID, deta_annual), delta_kev=d)
    y = np.interp(Ef, E, r, left=0.0, right=0.0) * efficiency(Ef)
    return float(np.trapezoid(y, Ef)) * lz.LZ['exposure_tyr']

t0 = time.time()
DELTAS = list(np.arange(250.0, 391.0, 10.0))
f1 = f'{OUT}/N_unit_grid_live.csv'
if not os.path.exists(f1):
    ham_iso = lz.wd_hamiltonian('P075_iso_unit', {1: (2.0 / MV**2, 0.0)})
    ham_p = lz.wd_hamiltonian('P075_p_unit', {1: (1.0 / MV**2, 1.0 / MV**2)})
    rows = []
    for m in (500.0, 2000.0, 5000.0, 10000.0, 1000.0):
        for d in (DELTAS if m != 1000.0 else [300.0, 365.0]):
            rows.append(dict(m_GeV=m, delta_keV=d, N_iso_unit=count(ham_iso, m, d), N_p_unit=count(ham_p, m, d), source='WimPyDD-live'))
            print('unit', m, d, rows[-1]['N_iso_unit'], rows[-1]['N_p_unit'], '%.0f s' % (time.time() - t0), flush=True)
    pd.DataFrame(rows).to_csv(f1, index=False)
f2 = f'{OUT}/higgsino_grid_live.csv'
if not os.path.exists(f2):
    hc = json.load(open('output/work/P007/higgsino_couplings.json'))
    ham_h = lz.wd_hamiltonian('P075_higgsino_Z', {1: (hc['c0_wimpydd'], hc['c1_wimpydd'])})
    rows = []
    for m in (5000.0, 10000.0, 1000.0):
        for d in (DELTAS if m != 1000.0 else [300.0, 350.0, 365.0, 375.0]):
            rows.append(dict(m_GeV=m, delta_keV=d, N_events=count(ham_h, m, d), source='WimPyDD-live'))
            print('higgsino', m, d, rows[-1]['N_events'], '%.0f s' % (time.time() - t0), flush=True)
    pd.DataFrame(rows).to_csv(f2, index=False)
print('done %.0f s' % (time.time() - t0))
