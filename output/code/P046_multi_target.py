"""
P046 -- A multi-target confirmation programme for a 300-390 keV inelastic splitting:
        xenon, iodine and tungsten as a delta-meter.

Run from the simulation root:   .venv/bin/python output/code/P046_multi_target.py

Method (all rates from WimPyDD 2.0.4 via lzcommon):
  For velocity-independent operators (O1; L10 = q^2 O4 - m_N^2 O6) WimPyDD's diff_rate is
      dR/dE (E, delta) = sum_isotopes K_i(E) * eta(v_min,i(E, delta)),   eta(v) = sum_{v_j >= v} delta_eta_j ,
  so the per-isotope kernels K_i(E) are obtained once from a single high-velocity stream with
  delta_eta = 1 (km/s)^-1 and then combined with any Earth-frame halo function (annual mean, 16 June,
  16 Dec, v_esc variants).  Validated against direct lz.wd_rate calls (section V below).

Parts
  1. window rates per t yr of element for O1 inelastic (1 TeV, delta = 250..395 keV, 1 keV steps) and
     L10 elastic on Xe, I, W, Ge (Cs if available); realistic windows; ratios R_WX, R_IX, R_GeX.
  2. delta-meter: Fisher information on delta from the W/Xe (and I/Xe) count ratio with the coupling
     profiled (binomial/multinomial), sigma_delta on exposure grids; Asimov 3 sigma separation exposures
     (delta pairs; inelastic vs L10); halo and form-factor systematics as delta biases.
     Comparison with the spectral-shape information (Fisher per event) in Xe and W.
  3. endpoint: W spectrum at delta = 366 keV, E+(delta, m, v_max); Fisher and order-statistic estimates of
     the events needed for +-50 keV; toy MLE validation; halo (v_esc +- 16 km/s) and mass systematics.
  4. background arithmetic for a CaWO4 W-band search (neutron kinematics, 180W alpha rate, 206Pb recoil).
  5. programme table.
Outputs: output/work/P046/*.csv|json, figures in output/work/P046/figures/
"""
import sys, os, math, json, time, importlib
import numpy as np
from scipy import special, optimize, stats
sys.path.insert(0, 'output/code')
from common import lzcommon as lz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'output/work/P046'; FIG = OUT + '/figures'
os.makedirs(FIG, exist_ok=True)
T0 = time.time()
WD = lz.wd()
# runtime fix from P015 (no file edited): the 18xW response files do `from numpy import *` but call `np.`
for a in (180, 182, 183, 184, 186):
    importlib.import_module(f'WimPyDD.Targets.Nuclear_response_functions.{a}W_func_w').np = np

C = lz.C_KMS; MV = lz.M_V_GEV; MN = lz.M_NUCLEON_GEV
M_CHI = 1000.0
VESC = lz.VESC_KMS
EXPO_LZ = lz.LZ['exposure_tyr']
LOG = open(f'{OUT}/run_log.txt', 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n'); LOG.flush()

# ------------------------------------------------------------------------------------------------
# targets
# ------------------------------------------------------------------------------------------------
TARGETS = {'Xe': WD.Xe, 'I': WD.I, 'W': WD.W, 'Ge': WD.Ge}
HAS_CS = hasattr(WD, 'Cs')
if HAS_CS:
    TARGETS['Cs'] = WD.Cs
for k, T in TARGETS.items():
    say(f'target {k}: A={list(T.a)} mass[GeV]={[round(m,2) for m in T.mass]} abund={[round(x,4) for x in T.abundance]} spin={list(T.spin)}')

# ------------------------------------------------------------------------------------------------
# hamiltonians (unit couplings; only ratios and LZ-normalised counts are used)
# ------------------------------------------------------------------------------------------------
H_O1 = lz.wd_hamiltonian('P046_O1_iso_unit', {1: (2.0 / MV ** 2, 0.0)})          # LZ/Anand c1^s = 1/m_v^2 (P003)
c4_wd, c6_wd = 8.0 / MV ** 2, -8.0 / MV ** 2                                        # P012: d10 = 1, m_M = m_N
H_L10 = WD.eft_hamiltonian('P046_L10_dipole_dipole', {(4, 'q2'): lambda q, A=c4_wd: [A * q ** 2 / MN ** 2, 0.0],
                                                      6: lambda A6=c6_wd: [A6, 0.0]})

# ------------------------------------------------------------------------------------------------
# Part 0: per-isotope kernels K_i(E) [events/(t yr keV) per unit eta (km/s)^-1]
#   1 TeV: evaluated directly on the 1 keV analysis grid (no interpolation); 400 and 4000 GeV (Xe, W, O1 only,
#   for the mass degeneracy of the ratio): 300-point log grid, interpolated.
# ------------------------------------------------------------------------------------------------
E_FINE = np.arange(1.0, 1500.0 + 0.5, 1.0)          # 1 keV bins, densities at bin centres (integers)
E_K300 = np.geomspace(1.0, 1500.0, 300)
STREAM = (np.array([3000.0]), np.array([1.0]))
KERN = {}      # (target, op, isotope, m_chi) -> (E grid, K)
t1 = time.time()
CACHE = f'{OUT}/kernels.npz'
if os.path.exists(CACHE) and '--recompute' not in sys.argv:
    z = np.load(CACHE)
    for key in z.files:
        if key in ('E_FINE', 'E_K300'):
            continue
        n, o, i, m = key.split('_'); m = float(m)
        KERN[(n, o, int(i), m)] = (E_FINE if m == M_CHI else E_K300, z[key])
    say(f'kernels loaded from cache {CACHE} ({len(KERN)} arrays); run with --recompute to regenerate')
else:
    for name, T in TARGETS.items():
        for op, H in (('O1', H_O1), ('L10', H_L10)):
            for i in range(T.n_isotopes):
                if op == 'L10' and T.spin[i] == 0:
                    KERN[(name, op, i, M_CHI)] = (E_FINE, np.zeros_like(E_FINE)); continue
                k = lz.wd_rate(H, M_CHI, E_FINE, halo=STREAM, delta_kev=0.0, target=T, isotopes_list={0: [i]})
                KERN[(name, op, i, M_CHI)] = (E_FINE, np.clip(np.atleast_1d(k), 0.0, None))
        say(f'kernels {name} done ({time.time()-t1:.0f} s)')
    for m in (400.0, 4000.0):
        for name in ('Xe', 'W'):
            T = TARGETS[name]
            for i in range(T.n_isotopes):
                k = lz.wd_rate(H_O1, m, E_K300, halo=STREAM, delta_kev=0.0, target=T, isotopes_list={0: [i]})
                KERN[(name, 'O1', i, m)] = (E_K300, np.clip(np.atleast_1d(k), 0.0, None))
        say(f'kernels m={m:.0f} GeV (Xe, W) done ({time.time()-t1:.0f} s)')
    np.savez(CACHE, **{f'{n}_{o}_{i}_{int(m)}': v[1] for (n, o, i, m), v in KERN.items()}, E_FINE=E_FINE, E_K300=E_K300)

def kern_interp(name, op, i, E, m_chi=M_CHI):
    Eg, K = KERN[(name, op, i, m_chi)]
    if Eg is E_FINE and E is E_FINE:
        return K
    return np.interp(np.log(E), np.log(Eg), K, left=0.0, right=0.0)

# ------------------------------------------------------------------------------------------------
# halos: eta_step(v) from WimPyDD streamed halo functions
# ------------------------------------------------------------------------------------------------
class Halo:
    def __init__(self, vmin, deta):
        self.vmin = np.asarray(vmin); self.cum = np.cumsum(np.asarray(deta)[::-1])[::-1]
        self.vmax = float(self.vmin[np.nonzero(np.asarray(deta) > 0)[0].max()]) if np.any(np.asarray(deta) > 0) else 0.0
    def eta(self, v):
        # WimPyDD dsigma_der keeps streams with v_i > v_min (strict): sum of delta_eta over v_i > v
        idx = np.searchsorted(self.vmin, v, side='right')
        out = np.zeros(np.shape(v)); ok = idx < len(self.cum); out[ok] = self.cum[idx[ok]]; return out

def make_halo(vesc=VESC, day=None, annual=False, v0=lz.V0_KMS):
    grid = np.linspace(0.0, vesc + 300.0, 1200)
    if annual:
        days12 = 15.0 + 365.25 / 12 * np.arange(12)
        deta = np.mean([lz.wd_halo(day_of_year=d, vesc=vesc, v0=v0, vmin=grid)[1] for d in days12], axis=0)
    else:
        _, deta = lz.wd_halo(day_of_year=day, vesc=vesc, v0=v0, vmin=grid)
    return Halo(grid, deta)

HALOS = {'annual': make_halo(annual=True), 'june16': make_halo(day=167), 'dec16': make_halo(day=350),
         'annual_vesc528': make_halo(vesc=528.0, annual=True), 'annual_vesc560': make_halo(vesc=560.0, annual=True),
         'annual_v0220': make_halo(v0=220.0, annual=True), 'annual_v0250': make_halo(v0=250.0, annual=True)}
say('halos done (%.0f s); v_max annual/june/dec = %.1f / %.1f / %.1f' % (time.time() - T0, HALOS['annual'].vmax, HALOS['june16'].vmax, HALOS['dec16'].vmax))

C_WD = 3.0e5   # WimPyDD's get_vmin uses c = 3e5 km/s; matching it removes a 0.5 km/s v_min offset that matters near the ceiling
def vmin_iso(E, mN, m_chi, delta):
    mu = m_chi * mN / (m_chi + mN)
    return (mN * E * 1e-6 / mu + delta * 1e-6) / np.sqrt(2 * mN * E * 1e-6) * C_WD

def spectrum(name, op, E, delta, halo, m_chi=M_CHI):
    """dR/dE in events/(t yr keV) of element, unit coupling."""
    T = TARGETS[name]; r = np.zeros(np.shape(E))
    for i in range(T.n_isotopes):
        if not np.any(KERN[(name, op, i, m_chi)][1]):
            continue
        r += kern_interp(name, op, i, E, m_chi) * halo.eta(vmin_iso(E, T.mass[i], m_chi, delta))
    return r

# ---- validation against direct WimPyDD calls -------------------------------------------------------
val = []
for (name, op, d, Es) in [('Xe', 'O1', 300.0, [100.0, 200.0, 400.0]), ('W', 'O1', 300.0, [60.0, 200.0, 800.0]),
                          ('I', 'O1', 350.0, [200.0, 400.0]), ('Xe', 'O1', 380.0, [250.0, 350.0]),
                          ('Xe', 'L10', 0.0, [50.0, 200.0]), ('W', 'O1', 366.0, [100.0, 500.0, 1000.0])]:
    h = HALOS['annual']; H = H_O1 if op == 'O1' else H_L10
    direct = lz.wd_rate(H, M_CHI, Es, halo=(h.vmin, np.diff(np.append(h.cum, 0.0)) * -1), delta_kev=d, target=TARGETS[name])
    fact = spectrum(name, op, np.array(Es), d, h)
    for e, a, b in zip(Es, np.atleast_1d(direct), fact):
        val.append(dict(target=name, op=op, delta_keV=d, E_keV=e, direct=float(a), factorised=float(b), ratio=float(b / a) if a else None))
say('validation factorised/direct:', [(v['target'], v['delta_keV'], v['E_keV'], None if v['ratio'] is None else round(v['ratio'], 4)) for v in val])

# ------------------------------------------------------------------------------------------------
# Part 1: window rates, LZ normalisation, ratios
# ------------------------------------------------------------------------------------------------
def eff_LZ(E, sig_hi=11.5, sig_lo=2.5):
    E = np.asarray(E, float)
    return lz.LZ['eff_plateau'] * 0.5 * (1 + special.erf((E - lz.LZ['E_50pct_low_keV']) / (math.sqrt(2) * sig_lo))) \
        * 0.5 * special.erfc((E - lz.LZ['E_50pct_high_keV']) / (math.sqrt(2) * sig_hi))
def box(E, lo, hi, eff=1.0):
    return eff * ((E >= lo) & (E <= hi))
WINDOWS = {   # name: (target, weight function)
    'Xe_full': ('Xe', lambda E: np.ones_like(E)), 'Xe_LZ270': ('Xe', eff_LZ), 'Xe_5_1000': ('Xe', lambda E: box(E, 5.0, 1000.0, 0.96)),
    'I_full': ('I', lambda E: np.ones_like(E)), 'I_5_1000': ('I', lambda E: box(E, 5.0, 1000.0)),
    'W_full': ('W', lambda E: np.ones_like(E)), 'W_10_1300': ('W', lambda E: box(E, 10.0, 1300.0)), 'W_110_1300': ('W', lambda E: box(E, 110.0, 1300.0)),
    'W_220_1300': ('W', lambda E: box(E, 220.0, 1300.0)),
    'Ge_full': ('Ge', lambda E: np.ones_like(E)), 'Ge_1_1000': ('Ge', lambda E: box(E, 1.0, 1000.0)),
}
if HAS_CS:
    WINDOWS.update({'Cs_full': ('Cs', lambda E: np.ones_like(E)), 'Cs_5_1000': ('Cs', lambda E: box(E, 5.0, 1000.0))})
DELTAS = np.arange(250.0, 395.0 + 0.5, 1.0)
RATE = {}      # (window, halo) -> array over DELTAS  (O1 inelastic, unit coupling, per t yr of element)
RATE_L10 = {}  # (window, halo) -> scalar
SPEC = {}      # (target, delta, halo) -> spectrum on E_FINE for selected deltas
for hk, h in HALOS.items():
    for wname, (tg, wf) in WINDOWS.items():
        w = wf(E_FINE)
        RATE[(wname, hk)] = np.array([float(np.sum(spectrum(tg, 'O1', E_FINE, d, h) * w)) for d in DELTAS])
        RATE_L10[(wname, hk)] = float(np.sum(spectrum(tg, 'L10', E_FINE, 0.0, h) * w))
    say(f'window rates for halo {hk} done ({time.time()-T0:.0f} s)')
for tg in TARGETS:
    for d in (300.0, 350.0, 366.0, 380.0):
        for hk in ('annual', 'june16', 'dec16'):
            SPEC[(tg, d, hk)] = spectrum(tg, 'O1', E_FINE, d, HALOS[hk])
    SPEC[(tg, 'L10', 'annual')] = spectrum(tg, 'L10', E_FINE, 0.0, HALOS['annual'])

def at(arr, d):
    return float(np.interp(d, DELTAS, arr))

# LZ normalisation: 1.0 event in 2.84 t yr with the LZ efficiency (annual halo)  ->  kappa_hat(delta) = (c1 m_v^2)^2
N_LZ_unit = EXPO_LZ * RATE[('Xe_LZ270', 'annual')]
KAPPA = 1.0 / N_LZ_unit
N_LZ_unit_L10 = EXPO_LZ * RATE_L10[('Xe_LZ270', 'annual')]
KAPPA_L10 = 1.0 / N_LZ_unit_L10
say('LZ unit-coupling events (P015: 13565/264.6/21.7/0.684 at 300/350/366/380):',
    {int(d): round(at(N_LZ_unit, d), 4) for d in (250, 300, 350, 366, 380, 390)}, ' L10 d10=1: %.3f (P012 3.34)' % N_LZ_unit_L10)

# ratios
def ratio(wA, wB, hk='annual'):
    return RATE[(wA, hk)] / np.where(RATE[(wB, hk)] > 0, RATE[(wB, hk)], np.nan)
R = {'W/Xe full': ratio('W_full', 'Xe_full'), 'I/Xe full': ratio('I_full', 'Xe_full'), 'Ge/Xe full': ratio('Ge_full', 'Xe_full'),
     'W(10-1300)/Xe(LZ270)': ratio('W_10_1300', 'Xe_LZ270'), 'W(10-1300)/Xe(5-1000)': ratio('W_10_1300', 'Xe_5_1000'),
     'W(110-1300)/Xe(5-1000)': ratio('W_110_1300', 'Xe_5_1000'), 'I(5-1000)/Xe(5-1000)': ratio('I_5_1000', 'Xe_5_1000'),
     'W/Xe full june16': ratio('W_full', 'Xe_full', 'june16'), 'W/Xe full vesc528': ratio('W_full', 'Xe_full', 'annual_vesc528'),
     'W/Xe full vesc560': ratio('W_full', 'Xe_full', 'annual_vesc560'), 'W/Xe full v0=220': ratio('W_full', 'Xe_full', 'annual_v0220'),
     'W/Xe full v0=250': ratio('W_full', 'Xe_full', 'annual_v0250')}
if HAS_CS:
    R['Cs/Xe full'] = ratio('Cs_full', 'Xe_full')
R_L10 = {k: RATE_L10[(a, 'annual')] / RATE_L10[(b, 'annual')] for k, a, b in
         [('W/Xe full', 'W_full', 'Xe_full'), ('I/Xe full', 'I_full', 'Xe_full'), ('Ge/Xe full', 'Ge_full', 'Xe_full'),
          ('W(10-1300)/Xe(5-1000)', 'W_10_1300', 'Xe_5_1000'), ('I(5-1000)/Xe(5-1000)', 'I_5_1000', 'Xe_5_1000')]}
# zero-momentum-transfer spin estimate of the L10 ratios: isoscalar Sigma' ~ (J+1)/J (<S_p>+<S_n>)^2 per isotope, weighted by
# nuclei per kg (abundance).  <S> values recalled (Menendez/Klos-type shell model for Xe, I, Ge; WimPyDD's own s_n = -0.17 for 183W):
# 129Xe (0.01, 0.33), 131Xe (-0.01, -0.27), 127I (0.31, 0.07), 73Ge (0.03, 0.44), 183W (0, -0.17)  [reliability: likely/uncertain]
SPIN_EST = {'Xe': [(0.264, 0.5, 0.34), (0.2123, 1.5, -0.28)], 'W': [(0.1431, 0.5, -0.17)], 'I': [(1.0, 2.5, 0.38)], 'Ge': [(0.077, 4.5, 0.47)]}
def spin_weight(name):
    T = TARGETS[name]; nt = np.array(T.nt_kg).sum()   # nuclei per kg of element (all isotopes)
    return sum(f * (J + 1) / J * S ** 2 for f, J, S in SPIN_EST[name]) * nt
L10_SPIN_EST = {k: spin_weight(k) / spin_weight('Xe') for k in ('W', 'I', 'Ge')}
say('L10 ratios (WimPyDD):', {k: '%.3g' % v for k, v in R_L10.items()}, ' zero-q spin estimate W/Xe, I/Xe, Ge/Xe:', {k: '%.3g' % v for k, v in L10_SPIN_EST.items()})

with open(f'{OUT}/rates_vs_delta.csv', 'w') as f:
    cols = ['delta_keV', 'kappa_hat_LZ'] + [f'{w}_{h}' for h in ('annual', 'june16') for w in WINDOWS] + list(R.keys())
    f.write(','.join(c.replace(',', ';') for c in cols) + '\n')
    for j, d in enumerate(DELTAS):
        if d % 5 == 0:
            row = [d, KAPPA[j]] + [RATE[(w, h)][j] for h in ('annual', 'june16') for w in WINDOWS] + [R[k][j] for k in R]
            f.write(','.join('%.6g' % x for x in row) + '\n')
with open(f'{OUT}/rates_L10.csv', 'w') as f:
    f.write('window,halo,rate_unit_d10_per_tyr_element\n')
    for (w, h), v in RATE_L10.items():
        f.write(f'{w},{h},{v:.6g}\n')
    for k, v in R_L10.items():
        f.write(f'ratio {k},annual,{v:.6g}\n')
    for k, v in L10_SPIN_EST.items():
        f.write(f'ratio {k}/Xe zero-q spin estimate,-,{v:.6g}\n')
say('ratios W/Xe full at 300/350/366/380:', [round(at(R['W/Xe full'], d), 2) for d in (300, 350, 366, 380)],
    ' I/Xe:', [round(at(R['I/Xe full'], d), 4) for d in (300, 350, 366, 380)])

# ------------------------------------------------------------------------------------------------
# Part 2: delta-meter
# ------------------------------------------------------------------------------------------------
F_W = 0.6385      # W mass fraction of CaWO4 (P015)
F_I_NaI = 0.8466  # I mass fraction of NaI (P015)
M_CS, M_I_U = 132.905, 126.904
F_I_CsI, F_CS_CsI = M_I_U / (M_CS + M_I_U), M_CS / (M_CS + M_I_U)
XE_WIN, W_WIN, I_WIN = 'Xe_5_1000', 'W_10_1300', 'I_5_1000'

def fisher_ratio(delta, E_Xe, E_CaWO4, xe_win=XE_WIN, w_win=W_WIN, E_I_comp=0.0, i_frac=F_I_NaI, hk='annual', kappa=None):
    """Fisher information on delta from the count fractions among targets, coupling profiled (multinomial)."""
    if kappa is None:
        kappa = at(KAPPA, delta)
    mus, dmus = [], []
    for win, expo in ((xe_win, E_Xe), (w_win, E_CaWO4 * F_W), (I_WIN, E_I_comp * i_frac)):
        if expo <= 0:
            continue
        arr = RATE[(win, hk)] * expo
        mus.append(at(arr, delta)); dmus.append(float(np.interp(delta, DELTAS, np.gradient(arr, DELTAS))))
    mus, dmus = np.array(mus), np.array(dmus)
    N = mus.sum(); p = mus / N; dp = (dmus * N - mus * dmus.sum()) / N ** 2
    info = kappa * N * np.sum(dp ** 2 / p)
    return dict(N_tot=kappa * N, p=p.tolist(), dp_ddelta=dp.tolist(), fisher=info, sigma_delta=1 / math.sqrt(info) if info > 0 else float('inf'),
                mu=(kappa * mus).tolist())

GRID_XE = [10.0, 20.0, 30.0, 50.0]
GRID_W = [0.01, 0.03, 0.1, 0.3, 1.0]
DM_DELTAS = [300.0, 330.0, 350.0, 366.0, 380.0]
dm_rows = []
for d in DM_DELTAS:
    for ex in GRID_XE:
        for ew in GRID_W:
            for xw in ('Xe_LZ270', 'Xe_5_1000'):
                r = fisher_ratio(d, ex, ew, xe_win=xw)
                dm_rows.append(dict(delta_keV=d, expo_Xe_tyr=ex, expo_CaWO4_tyr=ew, xe_window=xw, mu_Xe=r['mu'][0], mu_W=r['mu'][1],
                                    p_W=r['p'][1], sigma_delta_keV=r['sigma_delta']))
with open(f'{OUT}/delta_meter_sigma.csv', 'w') as f:
    keys = list(dm_rows[0].keys()); f.write(','.join(keys) + '\n')
    for r in dm_rows:
        f.write(','.join(('%.5g' % r[k]) if isinstance(r[k], float) else str(r[k]) for k in keys) + '\n')
say('sigma_delta [keV] (Xe 20 t yr 5-1000 keV; CaWO4 0.01/0.1/1 t yr):',
    {int(d): [round(fisher_ratio(d, 20.0, ew)['sigma_delta'], 1) for ew in (0.01, 0.1, 1.0)] for d in DM_DELTAS})

# iodine as a delta-meter (NaI or CsI cryogenic, 10-30 t yr, zero background) and three-target combination
io_rows = []
for d in DM_DELTAS:
    for ex in (20.0,):
        for ei in (10.0, 30.0):
            rI = fisher_ratio(d, ex, 0.0, E_I_comp=ei)
            r3 = fisher_ratio(d, ex, 0.1, E_I_comp=ei)
            io_rows.append(dict(delta_keV=d, expo_Xe_tyr=ex, expo_NaI_tyr=ei, mu_I=rI['mu'][-1], sigma_delta_IXe_keV=rI['sigma_delta'],
                                sigma_delta_three_target_W0p1_keV=r3['sigma_delta']))
with open(f'{OUT}/delta_meter_iodine.csv', 'w') as f:
    keys = list(io_rows[0].keys()); f.write(','.join(keys) + '\n')
    for r in io_rows:
        f.write(','.join('%.5g' % r[k] for k in keys) + '\n')

# spectral-shape Fisher information per event (Xe with LZ-like resolution, W with 1% resolution) -- for comparison
def smear_matrix(E, sig_of_E):
    sig = sig_of_E(E)
    M = np.exp(-0.5 * ((E[None, :] - E[:, None]) / sig[None, :]) ** 2) / (math.sqrt(2 * math.pi) * sig[None, :])   # M[obs, true]
    return M / M.sum(axis=0, keepdims=True)
SM_XE = smear_matrix(E_FINE, lambda E: 11.0 * np.sqrt(E / 248.0))       # P009/P021 resolution model
SM_W = smear_matrix(E_FINE, lambda E: np.maximum(0.01 * E, 2.0))        # recalled/assumed 1 % phonon resolution (uncertain)
SM_I = smear_matrix(E_FINE, lambda E: np.maximum(0.01 * E, 2.0))
def pdf_obs(name, d, hk='annual', SM=None, wf=None, m_chi=M_CHI):
    s = spectrum(name, 'O1', E_FINE, d, HALOS[hk], m_chi=m_chi)
    if SM is not None:
        s = SM @ s
    if wf is not None:
        s = s * wf(E_FINE)
    n = s.sum()
    return s / n if n > 0 else s, n
def fisher_shape(name, d, SM, wf, hk='annual', h=1.0):
    fp, _ = pdf_obs(name, d + h, hk, SM, wf); fm, _ = pdf_obs(name, d - h, hk, SM, wf); f0, _ = pdf_obs(name, d, hk, SM, wf)
    df = (fp - fm) / (2 * h); ok = f0 > 1e-12 * f0.max()
    return float(np.sum(df[ok] ** 2 / f0[ok]))
shape_rows = []
for d in DM_DELTAS:
    iXe270 = fisher_shape('Xe', d, SM_XE, eff_LZ); iXe1000 = fisher_shape('Xe', d, SM_XE, WINDOWS['Xe_5_1000'][1])
    iW = fisher_shape('W', d, SM_W, WINDOWS['W_10_1300'][1]); iI = fisher_shape('I', d, SM_I, WINDOWS['I_5_1000'][1])
    shape_rows.append(dict(delta_keV=d, I1_Xe_LZ270=iXe270, I1_Xe_5_1000=iXe1000, I1_W_10_1300=iW, I1_I_5_1000=iI,
                           sigma_1event_Xe270_keV=iXe270 ** -0.5, sigma_1event_Xe1000_keV=iXe1000 ** -0.5, sigma_1event_W_keV=iW ** -0.5, sigma_1event_I_keV=iI ** -0.5))
with open(f'{OUT}/shape_fisher.csv', 'w') as f:
    keys = list(shape_rows[0].keys()); f.write(','.join(keys) + '\n')
    for r in shape_rows:
        f.write(','.join('%.5g' % r[k] for k in keys) + '\n')
say('shape sigma_delta per single event [keV] (Xe270 / Xe1000 / W / I):', [(int(r['delta_keV']), round(r['sigma_1event_Xe270_keV'], 1), round(r['sigma_1event_Xe1000_keV'], 1), round(r['sigma_1event_W_keV'], 1), round(r['sigma_1event_I_keV'], 1)) for r in shape_rows])

# Asimov separation with the coupling profiled
def asimov_Z(mu1, r0, expos):
    """mu1: expected counts under the true hypothesis per target; r0: unit-coupling rates under H0; expos: exposures (element t yr)."""
    mu1 = np.asarray(mu1, float); r0e = np.asarray(r0, float) * np.asarray(expos, float)
    if r0e.sum() <= 0:
        return float('inf') if mu1.sum() > 0 else 0.0
    k0 = mu1.sum() / r0e.sum(); mu0 = k0 * r0e
    q = 0.0
    for a, b in zip(mu1, mu0):
        if a > 0 and b > 0:
            q += 2 * (b - a + a * math.log(a / b))
        elif a > 0 and b == 0:
            return float('inf')
        else:
            q += 2 * b
    return math.sqrt(max(q, 0.0))

def counts_inel(d, E_Xe, E_CaWO4, xe_win=XE_WIN, hk='annual'):
    return np.array([at(KAPPA, d) * at(RATE[(xe_win, hk)], d) * E_Xe, at(KAPPA, d) * at(RATE[(W_WIN, hk)], d) * E_CaWO4 * F_W])
def rates_inel(d, xe_win=XE_WIN, hk='annual'):
    return np.array([at(RATE[(xe_win, hk)], d), at(RATE[(W_WIN, hk)], d)])
def rates_L10(xe_win=XE_WIN, wxe_ratio=None):
    rx = RATE_L10[(xe_win, 'annual')]
    rw = RATE_L10[(W_WIN, 'annual')] if wxe_ratio is None else wxe_ratio * RATE_L10[('Xe_full', 'annual')] * RATE_L10[(W_WIN, 'annual')] / max(RATE_L10[('W_full', 'annual')], 1e-300)
    return np.array([rx, rw])
def counts_L10(E_Xe, E_CaWO4, xe_win=XE_WIN):
    r = rates_L10(xe_win)
    return KAPPA_L10 * r * np.array([E_Xe, E_CaWO4 * F_W])

def expo_W_for_Z(Zt, fn):
    """smallest CaWO4 exposure [t yr] with Z >= Zt (fn: E_CaWO4 -> Z), or inf."""
    lo, hi = 1e-5, 1e3
    if fn(hi) < Zt:
        return float('inf')
    if fn(lo) >= Zt:
        return lo
    return math.exp(optimize.brentq(lambda x: fn(math.exp(x)) - Zt, math.log(lo), math.log(hi), xtol=1e-3))

sep_rows = []
PAIRS = [(300.0, 380.0), (380.0, 300.0), (300.0, 350.0), (350.0, 300.0), (350.0, 380.0), (380.0, 350.0), (366.0, 380.0), (380.0, 366.0), (330.0, 366.0), (366.0, 330.0)]
for xw in ('Xe_LZ270', 'Xe_5_1000'):
    for ex in GRID_XE:
        for (dt, d0) in PAIRS:
            fn = lambda ew, dt=dt, d0=d0, ex=ex, xw=xw: asimov_Z(counts_inel(dt, ex, ew, xw), rates_inel(d0, xw), [ex, ew * F_W])
            e3 = expo_W_for_Z(3.0, fn); e5 = expo_W_for_Z(5.0, fn)
            sep_rows.append(dict(kind='inelastic delta pair', true=f'delta={int(dt)}', test=f'delta={int(d0)}', xe_window=xw, expo_Xe_tyr=ex,
                                 Z_at_CaWO4_0p01=fn(0.01), Z_at_0p1=fn(0.1), Z_at_1=fn(1.0), CaWO4_tyr_for_3sigma=e3, CaWO4_tyr_for_5sigma=e5,
                                 N_Xe_true=counts_inel(dt, ex, 0.0, xw)[0]))
        for dt in (300.0, 350.0, 366.0, 380.0):
            # true inelastic, test L10 (WimPyDD W spin response)
            fn = lambda ew, dt=dt, ex=ex, xw=xw: asimov_Z(counts_inel(dt, ex, ew, xw), rates_L10(xw), [ex, ew * F_W])
            sep_rows.append(dict(kind='true inelastic vs L10 (WimPyDD W Sigma-prime)', true=f'delta={int(dt)}', test='L10', xe_window=xw, expo_Xe_tyr=ex,
                                 Z_at_CaWO4_0p01=fn(0.01), Z_at_0p1=fn(0.1), Z_at_1=fn(1.0), CaWO4_tyr_for_3sigma=expo_W_for_Z(3.0, fn), CaWO4_tyr_for_5sigma=expo_W_for_Z(5.0, fn),
                                 N_Xe_true=counts_inel(dt, ex, 0.0, xw)[0]))
            # true L10, test inelastic at delta
            fn = lambda ew, dt=dt, ex=ex, xw=xw: asimov_Z(counts_L10(ex, ew, xw), rates_inel(dt, xw), [ex, ew * F_W])
            sep_rows.append(dict(kind='true L10 vs inelastic', true='L10', test=f'delta={int(dt)}', xe_window=xw, expo_Xe_tyr=ex,
                                 Z_at_CaWO4_0p01=fn(0.01), Z_at_0p1=fn(0.1), Z_at_1=fn(1.0), CaWO4_tyr_for_3sigma=expo_W_for_Z(3.0, fn), CaWO4_tyr_for_5sigma=expo_W_for_Z(5.0, fn),
                                 N_Xe_true=counts_L10(ex, 0.0, xw)[0]))
# L10 vs inelastic with the alternative (zero-q spin estimate) W/Xe ratio, Xe 20 t yr
for dt in (300.0, 350.0, 366.0, 380.0):
    rL = np.array([RATE_L10[(XE_WIN, 'annual')], L10_SPIN_EST['W'] * RATE_L10[(XE_WIN, 'annual')]])   # W rate := ratio x Xe rate (same window class)
    fn = lambda ew, dt=dt: asimov_Z(counts_inel(dt, 20.0, ew), rL, [20.0, ew * F_W])
    sep_rows.append(dict(kind='true inelastic vs L10 (zero-q spin estimate W/Xe)', true=f'delta={int(dt)}', test='L10', xe_window=XE_WIN, expo_Xe_tyr=20.0,
                         Z_at_CaWO4_0p01=fn(0.01), Z_at_0p1=fn(0.1), Z_at_1=fn(1.0), CaWO4_tyr_for_3sigma=expo_W_for_Z(3.0, fn), CaWO4_tyr_for_5sigma=expo_W_for_Z(5.0, fn),
                         N_Xe_true=counts_inel(dt, 20.0, 0.0)[0]))
with open(f'{OUT}/separation_exposures.csv', 'w') as f:
    keys = list(sep_rows[0].keys()); f.write(','.join(keys) + '\n')
    for r in sep_rows:
        f.write(','.join(('%.5g' % r[k]) if isinstance(r[k], float) else str(r[k]) for k in keys) + '\n')
say('3-sigma CaWO4 exposures (Xe 20 t yr, 5-1000 keV):', [(r['true'], r['test'], '%.3g' % r['CaWO4_tyr_for_3sigma']) for r in sep_rows if r['expo_Xe_tyr'] == 20.0 and r['xe_window'] == XE_WIN and r['kind'].startswith('inelastic')])
say('inel vs L10 (Xe 20):', [(r['kind'][:22], r['true'], r['test'], '%.3g' % r['CaWO4_tyr_for_3sigma']) for r in sep_rows if r['expo_Xe_tyr'] == 20.0 and r['xe_window'] == XE_WIN and 'L10' in r['kind']])

# systematics of the delta-meter reading: bias from halo variants and W form factor
def delta_from_ratio(Rval, key='W/Xe full'):
    arr = R[key]; ok = np.isfinite(arr)
    lg = np.log(arr[ok]); dd = DELTAS[ok]
    if not (lg.min() <= math.log(Rval) <= lg.max()):
        return float('nan')
    return float(np.interp(math.log(Rval), lg, dd))    # W/Xe is monotonic increasing in delta
sys_rows = []
for d in (300.0, 330.0, 350.0, 366.0, 380.0):
    row = dict(delta_true_keV=d, R_WXe_SHM=at(R['W/Xe full'], d))
    for var in ('june16', 'vesc528', 'vesc560', 'v0=220', 'v0=250'):
        rv = at(R[f'W/Xe full {var}'], d); row[f'R_{var}'] = rv; row[f'delta_read_{var}'] = delta_from_ratio(rv)
    for fac, lab in ((2.0, 'FFx2'), (0.5, 'FFx0.5')):
        row[f'delta_read_{lab}'] = delta_from_ratio(fac * row['R_WXe_SHM'])
    row['delta_read_isovector_x1.04'] = delta_from_ratio(1.04 / 1.0 * row['R_WXe_SHM'])   # P015 isovector weights W 1.039 / Xe 1
    row['dlnR_ddelta_per_keV'] = float(np.interp(d, DELTAS, np.gradient(np.log(R['W/Xe full']), DELTAS)))
    sys_rows.append(row)
# mass degeneracy of the ratio: R_WX(delta; m) for m = 400 / 4000 GeV (kernels above), read off with the 1 TeV calibration
R_MASS = {}
for m in (400.0, 4000.0):
    rw = np.array([float(np.sum(spectrum('W', 'O1', E_FINE, d, HALOS['annual'], m_chi=m))) for d in DELTAS])
    rx = np.array([float(np.sum(spectrum('Xe', 'O1', E_FINE, d, HALOS['annual'], m_chi=m))) for d in DELTAS])
    R_MASS[m] = rw / np.where(rx > 0, rx, np.nan)
for row in sys_rows:
    d = row['delta_true_keV']
    for m in (400.0, 4000.0):
        rv = at(R_MASS[m], d); row[f'R_m{int(m)}'] = rv; row[f'delta_read_m{int(m)}'] = delta_from_ratio(rv) if np.isfinite(rv) else float('nan')
with open(f'{OUT}/delta_meter_systematics.csv', 'w') as f:
    keys = list(sys_rows[0].keys()); f.write(','.join(keys) + '\n')
    for r in sys_rows:
        f.write(','.join('%.5g' % r[k] for k in keys) + '\n')
say('mass degeneracy: R_WX at delta=300/350 for m=400/1000/4000:', [(int(d), round(at(R_MASS[400.0], d), 1), round(at(R['W/Xe full'], d), 1), round(at(R_MASS[4000.0], d), 1)) for d in (300, 350)],
    ' delta read at true 350 if m=400/4000: %.0f / %.0f' % (sys_rows[2]['delta_read_m400'], sys_rows[2]['delta_read_m4000']))
say('delta-meter bias (true 350): vesc528 -> %.1f, vesc560 -> %.1f, FFx2 -> %.1f, june -> %.1f; dlnR/ddelta = %.4f /keV' % (
    sys_rows[2]['delta_read_vesc528'], sys_rows[2]['delta_read_vesc560'], sys_rows[2]['delta_read_FFx2'], sys_rows[2]['delta_read_june16'], sys_rows[2]['dlnR_ddelta_per_keV']))

# ------------------------------------------------------------------------------------------------
# Part 3: endpoint in tungsten (and iodine)
# ------------------------------------------------------------------------------------------------
def Eplus(mN, m_chi, v_kms, delta):
    mu = m_chi * mN / (m_chi + mN); b = v_kms / C; d = delta * 1e-6
    disc = 1 - 2 * d / (mu * b ** 2)
    if disc < 0:
        return float('nan')
    return mu ** 2 * b ** 2 / mN * (1 - d / (mu * b ** 2) + math.sqrt(disc)) * 1e6
D_END = 366.0
VJ = HALOS['june16'].vmax; VA = HALOS['annual'].vmax; VD = HALOS['dec16'].vmax
end = dict(delta_keV=D_END, v_max_june=VJ, v_max_annual=VA, v_max_dec=VD)
for name, A_list in (('W', TARGETS['W']), ('I', TARGETS['I']), ('Xe', TARGETS['Xe'])):
    end[f'Eplus_{name}_june_per_isotope'] = {int(a): Eplus(m, M_CHI, VJ, D_END) for a, m in zip(A_list.a, A_list.mass)}
    end[f'Eplus_{name}_dec_per_isotope'] = {int(a): Eplus(m, M_CHI, VD, D_END) for a, m in zip(A_list.a, A_list.mass)}
mW = TARGETS['W'].mass[-1]   # 186W: heaviest isotope defines the natural-W endpoint
EJ = Eplus(mW, M_CHI, VJ, D_END)
dEp_ddelta = (Eplus(mW, M_CHI, VJ, D_END + 1) - Eplus(mW, M_CHI, VJ, D_END - 1)) / 2
dEp_dv = (Eplus(mW, M_CHI, VJ + 1, D_END) - Eplus(mW, M_CHI, VJ - 1, D_END)) / 2
end.update(Eplus_W186_june=EJ, dEplus_ddelta_keV_per_keV=dEp_ddelta, dEplus_dvmax_keV_per_kms=dEp_dv,
           Eplus_shift_vesc_pm16=[Eplus(mW, M_CHI, VJ - 16, D_END) - EJ, Eplus(mW, M_CHI, VJ + 16, D_END) - EJ],
           delta_bias_vesc_pm16_keV=[(Eplus(mW, M_CHI, VJ - 16, D_END) - EJ) / dEp_ddelta, (Eplus(mW, M_CHI, VJ + 16, D_END) - EJ) / dEp_ddelta],
           Eplus_mass_400_4000=[Eplus(mW, 400.0, VJ, D_END), Eplus(mW, 4000.0, VJ, D_END)],
           delta_bias_mass_400_4000_keV=[(Eplus(mW, 400.0, VJ, D_END) - EJ) / dEp_ddelta, (Eplus(mW, 4000.0, VJ, D_END) - EJ) / dEp_ddelta],
           sigma_Eplus_50keV_as_delta_keV=50.0 / abs(dEp_ddelta))
# Xe for comparison
mXe = TARGETS['Xe'].mass[-1]
end['Xe136_dEplus_ddelta'] = (Eplus(mXe, M_CHI, VJ, D_END + 1) - Eplus(mXe, M_CHI, VJ, D_END - 1)) / 2
end['Xe136_Eplus_june'] = Eplus(mXe, M_CHI, VJ, D_END)
say('W endpoint (186W, June) at delta=366: %.1f keV; dE+/ddelta = %.2f; dE+/dv = %.2f keV per km/s; vesc +-16 -> %.0f/%.0f keV' % (EJ, dEp_ddelta, dEp_dv, *end['Eplus_shift_vesc_pm16']))

# W spectrum shape at 366 keV (annual-averaged, 1 % resolution, 10-1300 keV): Fisher and toys
wfW = WINDOWS['W_10_1300'][1]
fW366, _ = pdf_obs('W', D_END, 'annual', SM_W, wfW)
I1_W = fisher_shape('W', D_END, SM_W, wfW)
I1_W_nores = fisher_shape('W', D_END, None, wfW)
I1_I = fisher_shape('I', D_END, SM_I, WINDOWS['I_5_1000'][1])
N_for_sigma = lambda I1, s_delta: 1.0 / (I1 * s_delta ** 2)
end.update(I1_W_shape=I1_W, I1_W_shape_no_resolution=I1_W_nores, I1_I_shape=I1_I,
           N_W_for_sigma_delta_10keV=N_for_sigma(I1_W, 10.0), N_W_for_sigma_delta_5keV=N_for_sigma(I1_W, 5.0),
           N_W_for_sigma_Eplus_50keV=N_for_sigma(I1_W, 50.0 / abs(dEp_ddelta)),
           N_I_for_sigma_Eplus_50keV=N_for_sigma(I1_I, 50.0 / abs(end['Xe136_dEplus_ddelta'])))
# order statistic: E_max of N events
cdf = np.cumsum(fW366); cdf /= cdf[-1]
def q_Emax(N, u):
    return float(np.interp(u ** (1.0 / N), cdf, E_FINE))
os_rows = []
for N in (3, 10, 30, 100, 300, 1000, 3000):
    q16, q50, q84 = (q_Emax(N, u) for u in (0.16, 0.5, 0.84))
    os_rows.append(dict(N=N, Emax_q16=q16, Emax_q50=q50, Emax_q84=q84, gap_median_keV=EJ - q50, gap_q84_keV=EJ - q16, gap_q16_keV=EJ - q84))
N_os = None
for N in range(3, 20001):
    if EJ - q_Emax(N, 0.16) <= 50.0:
        N_os = N; break
end['N_for_Emax_within_50keV_84pct'] = N_os
# spectrum fractions and observable edges (E_90, E_99 of the resolution-smeared W spectrum) vs delta, halo and mass
sW_ann = SPEC[('W', D_END, 'annual')]
end['W_fraction_above_keV'] = {int(x): float(sW_ann[E_FINE > x].sum() / sW_ann.sum()) for x in (110, 220, 500, 800, 1000)}
end['W_fraction_last_100keV_of_window'] = float(sW_ann[E_FINE > EJ - 100].sum() / sW_ann.sum())
def edges(name, d, hk='annual', SM=SM_W, wf=wfW, m_chi=M_CHI):
    f, _ = pdf_obs(name, d, hk, SM, wf, m_chi=m_chi); c = np.cumsum(f); c /= c[-1]
    return float(np.interp(0.90, c, E_FINE)), float(np.interp(0.99, c, E_FINE)), float(E_FINE[np.argmax(f)])
edge_rows = []
for d in (300.0, 330.0, 350.0, 366.0, 380.0):
    sp = SPEC.get(('W', d, 'annual'), spectrum('W', 'O1', E_FINE, d, HALOS['annual']))
    row = dict(delta_keV=d, Eplus_W186_june=Eplus(mW, M_CHI, VJ, d), Eminus_W186_june=lz.E_R_range_keV(M_CHI, VJ, A=186, delta_kev=d)[0],
               frac_above_500=float(sp[E_FINE > 500].sum() / sp.sum()), frac_above_700=float(sp[E_FINE > 700].sum() / sp.sum()),
               frac_above_1000=float(sp[E_FINE > 1000].sum() / sp.sum()), frac_above_1300=float(sp[E_FINE > 1300].sum() / sp.sum()))
    e90, e99, pk = edges('W', d); row.update(E90=e90, E99=e99, E_peak=pk)
    e90p, e99p, _ = edges('W', d + 5); e90m, e99m, _ = edges('W', d - 5)
    row.update(dE90_ddelta=(e90p - e90m) / 10, dE99_ddelta=(e99p - e99m) / 10)
    for hk in ('annual_vesc528', 'annual_vesc560', 'june16'):
        a, b, _ = edges('W', d, hk); row[f'E90_{hk}'] = a; row[f'E99_{hk}'] = b
    for m in (400.0, 4000.0):
        a, b, pk2 = edges('W', d, m_chi=m); row[f'E90_m{int(m)}'] = a; row[f'E99_m{int(m)}'] = b; row[f'Epeak_m{int(m)}'] = pk2
    # falling-edge information only: Fisher of the conditional pdf above 300 keV
    row['I1_shape_full'] = fisher_shape('W', d, SM_W, wfW)
    row['I1_shape_above300'] = fisher_shape('W', d, SM_W, lambda E: box(E, 300.0, 1300.0))
    row['I1_shape_Xe_5_1000'] = fisher_shape('Xe', d, SM_XE, WINDOWS['Xe_5_1000'][1])
    edge_rows.append(row)
with open(f'{OUT}/W_edges.csv', 'w') as f:
    keys = list(edge_rows[0].keys()); f.write(','.join(keys) + '\n')
    for r in edge_rows:
        f.write(','.join('%.5g' % r[k] for k in keys) + '\n')
end['edges'] = edge_rows

# halo / mass bias of the *shape* delta-meter (KL projection): true pdf under a variant, fitted with SHM 1 TeV templates
DG2 = np.arange(280.0, 392.0, 1.0)
TEMPL = {name: np.array([np.log(np.maximum(pdf_obs(name, d, 'annual', SM, wf)[0], 1e-300)) for d in DG2])
         for name, SM, wf in (('W', SM_W, wfW), ('Xe', SM_XE, WINDOWS['Xe_5_1000'][1]))}
def kl_fit(name, f_true):
    lf = TEMPL[name]; s = lf @ f_true; k = int(np.argmax(s))
    if 0 < k < len(DG2) - 1:
        a, b, c = s[k - 1], s[k], s[k + 1]; k = k + 0.5 * (a - c) / (a - 2 * b + c)
    return float(DG2[0] + k)
shape_bias = []
for d in (330.0, 350.0, 366.0):
    for name, SM, wf in (('W', SM_W, wfW), ('Xe', SM_XE, WINDOWS['Xe_5_1000'][1])):
        row = dict(target=name, delta_true=d)
        for hk in ('annual_vesc528', 'annual_vesc560', 'annual_v0220', 'annual_v0250'):
            row[f'read_{hk}'] = kl_fit(name, pdf_obs(name, d, hk, SM, wf)[0])
        for m in (400.0, 4000.0):
            if name in ('W', 'Xe') and (m == 4000.0 or d <= 340.0):
                f_true, n = pdf_obs(name, d, 'annual', SM, wf, m_chi=m)
                row[f'read_m{int(m)}'] = kl_fit(name, f_true) if n > 0 else float('nan')
            else:
                row[f'read_m{int(m)}'] = float('nan')
        row['read_SHM_check'] = kl_fit(name, pdf_obs(name, d, 'annual', SM, wf)[0])
        shape_bias.append(row)
with open(f'{OUT}/shape_delta_meter_bias.csv', 'w') as f:
    keys = list(shape_bias[0].keys()); f.write(','.join(keys) + '\n')
    for r in shape_bias:
        f.write(','.join(('%.4g' % r[k]) if isinstance(r[k], float) else str(r[k]) for k in keys) + '\n')
end['shape_bias'] = shape_bias
say('shape delta-meter bias (true 366): ', [(r['target'], round(r['read_annual_vesc528'], 1), round(r['read_annual_vesc560'], 1), round(r['read_m4000'], 1), round(r['read_SHM_check'], 1)) for r in shape_bias if r['delta_true'] == 366.0])
say('W observable edges E90/E99 [keV] and dE99/ddelta:', [(int(r['delta_keV']), round(r['E90']), round(r['E99']), round(r['dE99_ddelta'], 2), 'frac>700: %.3g' % r['frac_above_700']) for r in edge_rows])
# toy MLE of delta from the W shape
rng = np.random.default_rng(46)
DGRID = np.arange(330.0, 386.0, 1.0)
LOGF = np.array([np.log(np.maximum(pdf_obs('W', d, 'annual', SM_W, wfW)[0], 1e-300)) for d in DGRID])
toy_rows = []
for N in (30, 100, 300):
    ests = []
    for _ in range(300):
        u = rng.random(N); Ei = np.interp(u, cdf, E_FINE); idx = np.clip(np.rint(Ei - 1.0).astype(int), 0, len(E_FINE) - 1)
        ll = LOGF[:, idx].sum(axis=1); k = int(np.argmax(ll))
        if 0 < k < len(DGRID) - 1:
            a, b, c = ll[k - 1], ll[k], ll[k + 1]; k_ref = k + 0.5 * (a - c) / (a - 2 * b + c)
        else:
            k_ref = k
        ests.append(DGRID[0] + k_ref)
    ests = np.array(ests)
    toy_rows.append(dict(N=N, mean=float(ests.mean()), std=float(ests.std()), fisher_sigma=1 / math.sqrt(N * I1_W), q16=float(np.percentile(ests, 16)), q84=float(np.percentile(ests, 84))))
end['toys'] = toy_rows
end['order_statistic'] = os_rows
say('W shape: I1 = %.4g per event -> sigma_delta(N=30) = %.1f keV (toys %.1f); N for E+ +-50 keV: Fisher %.1f, order-statistic %s' % (
    I1_W, 1 / math.sqrt(30 * I1_W), toy_rows[0]['std'], end['N_W_for_sigma_Eplus_50keV'], N_os))
json.dump(end, open(f'{OUT}/endpoint.json', 'w'), indent=1, default=float)

# ------------------------------------------------------------------------------------------------
# Part 4: background arithmetic for CaWO4 (recalled inputs flagged in details.md)
# ------------------------------------------------------------------------------------------------
MN_MEV, M_W_MEV, M_O_MEV, M_CA_MEV = 939.57, 183.84 * 931.49, 15.999 * 931.49, 40.078 * 931.49
def ER_max_keV(En_MeV, M_MeV):
    return 4 * MN_MEV * M_MeV / (MN_MEV + M_MeV) ** 2 * En_MeV * 1e3
bkg = dict(ER_max_W_keV={En: ER_max_keV(En, M_W_MEV) for En in (1, 2, 5, 10, 20, 50, 100)},
           ER_max_O_keV={En: ER_max_keV(En, M_O_MEV) for En in (1, 2, 5, 10, 20, 50, 100)},
           En_min_MeV_for_W_recoil={E: E / ER_max_keV(1.0, M_W_MEV) for E in (100.0, 220.0, 500.0, 1000.0)},
           Pb206_recoil_keV=103.0, W180_halflife_yr=1.8e18, W180_abundance=0.0012)
N_W_per_kg_CaWO4 = F_W / 0.18384 * 6.02214e23
lam = math.log(2) / (1.8e18 * 3.15576e7)
bkg['W180_alpha_rate_per_kg_day'] = N_W_per_kg_CaWO4 * 0.0012 * lam * 86400
bkg['W180_alpha_rate_per_kg_yr'] = bkg['W180_alpha_rate_per_kg_day'] * 365.25
bkg['W180_Q_MeV'] = 2.516; bkg['Hf176_recoil_keV'] = 2516.0 * 4.0 / (176 + 4) * (176 / 180)   # ~ Q*m_alpha/m_Hf ... approximate
bkg['Hf176_recoil_keV'] = 2516.0 * 4.0 / 180.0
# order-of-magnitude muon-induced neutron estimate (recalled flux; uncertain)
phi_n = 1e-10   # /cm^2/s, E_n > 10 MeV at ~3600 m.w.e. (recalled, uncertain, factor 3)
sig_el = 3e-24  # cm^2, n-W elastic at 10-50 MeV (recalled, uncertain, factor 2)
N_W_per_t = N_W_per_kg_CaWO4 * 1000
bkg['muon_neutron_W_scatters_per_tyr_before_veto'] = phi_n * sig_el * N_W_per_t * 3.15576e7
bkg['fraction_above_100keV_for_20MeV_neutron_flat'] = 1 - 100.0 / ER_max_keV(20.0, M_W_MEV)
bkg['note'] = 'recalled inputs, order of magnitude only; see details.md section 6'
# signal loss from a 110 keV (Pb-recoil-safe) or 220 keV (radiogenic-neutron-safe) W threshold
bkg['W_threshold_acceptance'] = {int(d): dict(thr110=at(RATE[('W_110_1300', 'annual')], d) / at(RATE[('W_10_1300', 'annual')], d),
                                              thr220=at(RATE[('W_220_1300', 'annual')], d) / at(RATE[('W_10_1300', 'annual')], d)) for d in (300, 350, 366, 380)}
json.dump(bkg, open(f'{OUT}/backgrounds.json', 'w'), indent=1, default=float)
say('180W alpha rate %.2f /kg/day; E_R,max(W) for 10/50 MeV n: %.0f/%.0f keV; muon-n W scatters/t yr before veto ~ %.0f' % (
    bkg['W180_alpha_rate_per_kg_day'], bkg['ER_max_W_keV'][10], bkg['ER_max_W_keV'][50], bkg['muon_neutron_W_scatters_per_tyr_before_veto']))

# ------------------------------------------------------------------------------------------------
# Part 5: programme table (expected events at the LZ best fit, kappa_hat(delta))
# ------------------------------------------------------------------------------------------------
PROG = [
    dict(entry='Xenon world, existing 7.5 t yr, 270 keV ROI (LZ 2.84 + XENONnT 3.1 + PandaX-4T 1.54; P005/P035)', win='Xe_LZ270', expo=7.48, frac=1.0,
         decides='whether the LZ rate is real (P(0) ~ 20 %); no delta information beyond event energies', when='now (re-analysis)'),
    dict(entry='Xenon 20 t yr, 270 keV ROI', win='Xe_LZ270', expo=20.0, frac=1.0, decides='rate x10 test; delta only via energies (P035: +-15 keV per event)', when='~2030'),
    dict(entry='Xenon 20 t yr, 5-1000 keV window (P038 extension)', win='Xe_5_1000', expo=20.0, frac=1.0, decides='captures the >270 keV lobe: x2-30 more events at delta >= 350; spectral delta-meter', when='~2030'),
    dict(entry='Xenon 50 t yr, 5-1000 keV window', win='Xe_5_1000', expo=50.0, frac=1.0, decides='anchor for the ratio: N_Xe sets the coupling', when='2030s'),
    dict(entry='CaWO4 10 kg yr, W band 10-1300 keV', win='W_10_1300', expo=0.010, frac=F_W, decides='delta >= 350: tens of events or exclusion; blind to delta = 300 (0.04 ev)', when='2-3 yr (CRESST-scale array)'),
    dict(entry='CaWO4 100 kg yr, W band 10-1300 keV', win='W_10_1300', expo=0.10, frac=F_W, decides='delta >= 330 decided; 3 sigma vs L10 for delta >= 330; W/Xe delta-meter to ~10 keV at delta >= 350', when='~5 yr'),
    dict(entry='CaWO4 1 t yr, W band 10-1300 keV', win='W_10_1300', expo=1.0, frac=F_W, decides='delta = 300 confirmed (4 ev); ratio delta-meter over the whole window; endpoint statistics at delta >= 350', when='~10 yr (dedicated)'),
    dict(entry='CaWO4 1 t yr, W band 110-1300 keV (206Pb-recoil-safe)', win='W_110_1300', expo=1.0, frac=F_W, decides='same with a hardware-simple threshold', when='~10 yr'),
    dict(entry='NaI(Tl) 30 t yr (no NR discrimination; P028)', win='I_5_1000', expo=30.0, frac=F_I_NaI, decides='nothing: signal buried in ~1e9 undiscriminated counts', when='-'),
    dict(entry='Cryogenic discriminating iodine (NaI/CsI bolometer) 10 t yr, zero background', win='I_5_1000', expo=10.0, frac=F_I_NaI, decides='I/Xe ratio: second delta-meter, weak (I/Xe < 0.6 and falling)', when='unrealistic before 2040'),
    dict(entry='Germanium 1 t yr (any window)', win='Ge_1_1000', expo=1.0, frac=1.0, decides='null control: any NR population > 100 keV falsifies inelastic DM with delta > 232 keV', when='LEGEND-class exposures exist'),
]
prog_rows = []
for p in PROG:
    row = dict(entry=p['entry'], window=p['win'], exposure_tyr_compound=p['expo'], element_fraction=p['frac'], decides=p['decides'], timescale=p['when'])
    for d in (300.0, 350.0, 366.0, 380.0):
        row[f'N_delta{int(d)}'] = at(KAPPA, d) * at(RATE[(p['win'], 'annual')], d) * p['expo'] * p['frac']
    row['N_L10'] = KAPPA_L10 * RATE_L10[(p['win'], 'annual')] * p['expo'] * p['frac']
    prog_rows.append(row)
with open(f'{OUT}/programme_table.csv', 'w') as f:
    keys = list(prog_rows[0].keys()); f.write(','.join(keys) + '\n')
    for r in prog_rows:
        f.write(','.join(('"%s"' % r[k]) if isinstance(r[k], str) else '%.4g' % r[k] for k in keys) + '\n')
say('programme counts (300/350/366/380 | L10):', [(r['entry'][:28], ['%.3g' % r[f'N_delta{d}'] for d in (300, 350, 366, 380)], '%.3g' % r['N_L10']) for r in prog_rows])

# ------------------------------------------------------------------------------------------------
# summary JSON
# ------------------------------------------------------------------------------------------------
summary = dict(
    m_chi_GeV=M_CHI, v_max_kms=dict(annual=VA, june16=VJ, dec16=VD), validation=val,
    N_LZ_unit={int(d): at(N_LZ_unit, d) for d in (250, 300, 350, 366, 380, 390)}, N_LZ_unit_L10=N_LZ_unit_L10,
    kappa_hat={int(d): at(KAPPA, d) for d in (250, 300, 350, 366, 380, 390)},
    ratios={k: {int(d): at(v, d) for d in (250, 300, 330, 350, 366, 380, 390)} for k, v in R.items()},
    ratios_L10=R_L10, L10_spin_estimate=L10_SPIN_EST,
    rates_unit_annual={w: {int(d): at(RATE[(w, 'annual')], d) for d in (300, 350, 366, 380)} for w in WINDOWS},
    rates_L10_unit_annual={w: RATE_L10[(w, 'annual')] for w in WINDOWS},
    delta_meter_examples={f'delta{int(d)}_Xe20_W{ew}': fisher_ratio(d, 20.0, ew) for d in DM_DELTAS for ew in (0.01, 0.1, 1.0)},
    shape_fisher=shape_rows, systematics=sys_rows, endpoint=end, backgrounds=bkg, programme=prog_rows,
    runtime_s=time.time() - T0)
json.dump(summary, open(f'{OUT}/P046_summary.json', 'w'), indent=1, default=float)

# ------------------------------------------------------------------------------------------------
# Figures (dataviz reference palette: blue, orange, aqua, yellow; text in ink tokens)
# ------------------------------------------------------------------------------------------------
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#898781', '#e6e5e1', '#fcfcfb'
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#c43fa8'
plt.rcParams.update({'font.size': 9.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': SURF, 'axes.facecolor': SURF})

# Fig 1: ratios vs delta
fig, ax = plt.subplots(figsize=(7.4, 4.6))
lo = np.minimum(R['W/Xe full vesc528'], R['W/Xe full vesc560']); hi = np.maximum(R['W/Xe full vesc528'], R['W/Xe full vesc560'])
ax.fill_between(DELTAS, lo, hi, color=BLUE, alpha=0.15, lw=0, label=r'W/Xe, $v_{esc}$ = 528-560 km/s')
ax.plot(DELTAS, R['W/Xe full'], color=BLUE, lw=2, label='W/Xe (full windows, annual halo)')
ax.plot(DELTAS, R['W(10-1300)/Xe(LZ270)'], color=BLUE, lw=1.2, ls='--', label='W(10-1300 keV)/Xe(LZ 270 keV ROI)')
ax.plot(DELTAS, R['I/Xe full'], color=ORANGE, lw=2, label='I/Xe (full windows)')
ax.axhline(R_L10['W/Xe full'], color=BLUE, lw=1, ls=':'); ax.text(251, R_L10['W/Xe full'] * 1.3, 'L10 W/Xe (WimPyDD, Gaussian spin FF)', color=INK2, fontsize=8)
ax.axhline(L10_SPIN_EST['W'], color=AQUA, lw=1, ls=':'); ax.text(251, L10_SPIN_EST['W'] * 1.3, 'L10 W/Xe (zero-q spin estimate)', color=INK2, fontsize=8)
ax.axhline(R_L10['I/Xe full'], color=ORANGE, lw=1, ls=':'); ax.text(300, R_L10['I/Xe full'] * 0.55, 'L10 I/Xe (WimPyDD shell-model 127I)', color=INK2, fontsize=8)
ax.axhline(1.0, color=MUTED, lw=0.6)
for d in (300, 350, 366, 380):
    ax.axvline(d, color=GRID, lw=0.8)
ax.set_yscale('log'); ax.set_xlim(250, 395); ax.set_ylim(1e-3, 3e4)
ax.set_xlabel(r'mass splitting $\delta$ [keV]  ($m_\chi$ = 1 TeV)'); ax.set_ylabel('rate ratio per tonne of element (coupling cancels)')
ax.set_title(r'The $\delta$-meter: W/Xe rises $\times$40 from 300 to 380 keV; L10 ratios are flat', fontsize=10, loc='left', color=INK)
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=8, loc='lower right')
fig.tight_layout(); fig.savefig(f'{FIG}/P046_fig1_ratios_vs_delta.png', dpi=170); plt.close(fig)

# Fig 2: sigma_delta vs CaWO4 exposure (Xe 20 t yr) and vs Xe exposure (CaWO4 0.1 t yr)
fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True)
EWs = np.geomspace(0.003, 3.0, 60); EXs = np.geomspace(5.0, 100.0, 60)
cols = {300.0: BLUE, 330.0: ORANGE, 350.0: AQUA, 366.0: MAGENTA, 380.0: YELLOW}   # order validated with dataviz validate_palette.js (light mode: all checks pass)
for d in DM_DELTAS:
    axes[0].plot(EWs, [fisher_ratio(d, 20.0, ew)['sigma_delta'] for ew in EWs], color=cols[d], lw=2, label=fr'$\delta$ = {int(d)} keV')
    axes[1].plot(EXs, [fisher_ratio(d, ex, 0.1)['sigma_delta'] for ex in EXs], color=cols[d], lw=2, label=fr'$\delta$ = {int(d)} keV')
for ax in axes:
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(0.5, 300); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.axhspan(12, 18, color='#f0efec', lw=0, zorder=0)
axes[0].text(0.0032, 13.5, 'halo + form-factor floor (~15 keV)', color=INK2, fontsize=8)
axes[0].set_xlabel(r'CaWO$_4$ exposure [t yr]  (xenon fixed at 20 t yr, 5-1000 keV)'); axes[0].set_ylabel(r'$\sigma_\delta$ from the W/Xe count ratio [keV]')
axes[1].set_xlabel(r'xenon exposure [t yr]  (CaWO$_4$ fixed at 0.1 t yr)')
axes[0].set_title(r'Statistical precision of the ratio $\delta$-meter (coupling profiled)', fontsize=10, loc='left', color=INK)
axes[0].legend(frameon=False, fontsize=8, loc='upper right')
fig.tight_layout(); fig.savefig(f'{FIG}/P046_fig2_sigma_delta_vs_exposure.png', dpi=170); plt.close(fig)

# Fig 3: W spectrum at 366 keV with endpoints, Xe for comparison, LZ best-fit normalisation
fig, ax = plt.subplots(figsize=(7.4, 4.4))
kap = at(KAPPA, D_END)
for hk, ls, lab in (('june16', '-', '16 June'), ('annual', '--', 'annual mean'), ('dec16', ':', '16 December')):
    ax.plot(E_FINE, SPEC[('W', D_END, hk)] * kap * F_W, color=BLUE, lw=1.8 if hk == 'annual' else 1.2, ls=ls, label=f'W in CaWO$_4$, {lab}')
ax.plot(E_FINE, SPEC[('Xe', D_END, 'annual')] * kap, color=ORANGE, lw=1.8, ls='--', label='Xe, annual mean')
ax.plot(E_FINE, (SM_W @ SPEC[('W', D_END, 'annual')]) * kap * F_W, color=AQUA, lw=1.0, label='W annual, 1 % resolution')
ax.axvline(EJ, color=INK2, lw=0.8); ax.text(EJ + 10, 3e-1, r'$E_+$(June) = %.0f keV' % EJ, color=INK2, fontsize=8, rotation=90, va='top')
ax.axvline(269.9, color=MUTED, lw=0.8, ls='--'); ax.text(275, 3e-5, 'LZ 270 keV edge', color=INK2, fontsize=8, rotation=90, va='bottom')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(30, 1500); ax.set_ylim(1e-5, 3)
ax.set_xlabel('nuclear recoil energy [keV]'); ax.set_ylabel(r'dR/dE at the LZ best fit [events / (t yr keV) of compound]')
ax.set_title(r'$\delta$ = 366 keV, 1 TeV: the tungsten spectrum and its endpoint', fontsize=10, loc='left', color=INK)
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=8, loc='lower left')
fig.tight_layout(); fig.savefig(f'{FIG}/P046_fig3_W_spectrum_endpoint.png', dpi=170); plt.close(fig)
say('all done in %.0f s' % (time.time() - T0))
LOG.close()
