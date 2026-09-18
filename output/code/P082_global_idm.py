#!/usr/bin/env python
"""
P082 -- A global direct-detection combination for inelastic dark matter after the LZ event:
        the surviving (m_chi, delta, sigma_n) region.

Joint likelihood = LZ extended-window likelihood (P021 model: observed-energy extended likelihood with P016's
Fig.-5 low-energy bins, the empty 125-200 keV bin, the single 248 keV event over b_H = 5.7e-4 flat in 200-270 keV;
efficiency edge = P038's NEST-LZ Monte-Carlo roll-off, E50 = 271.6 keV, plateau 0.955)
                  x  Poisson(0 | kappa N_j(m, delta)) for every zero-event archival xenon exposure
                     (LZ SR1, XENON1T, LUX 2021, PandaX-II, XENONnT 3.1 t yr, PandaX-4T 1.54 t yr; recalled ROIs, flagged)
                  [+ Fig. S7 non-xenon limits (CRESST-II, PICO-60 CF3I) compared in the (delta, sigma) plane].
Signal: WimPyDD 2.0.4 isoscalar O1, unit LZ/Anand coupling kappa = (c_1^s m_v^2)^2 = 1  <=>  WimPyDD c^0 = 2/m_v^2
(lz.wd_c_from_anand), Baxter-2021 SHM, annual-average halo (mean of 12 monthly days; June and Sun-frame variants).
Masses 200-10000 GeV: 400/1000/4000 GeV spectra reused from P021's cache, the others computed here (per-stream kernels,
cached in output/work/P082/cache/).

Run from the simulation root:
  .venv/bin/python output/code/P082_global_idm.py --spectra-only 200 300          (build + cache spectra, ~1.5 min each)
  .venv/bin/python output/code/P082_global_idm.py                                  (full analysis; ~1-2 min with caches)
"""
import sys, os, json, math, time, argparse
import numpy as np
import pandas as pd
sys.path.insert(0, 'output/code')
from common import lzcommon as lz
from scipy import stats, special, optimize

T0 = time.time()
OUT = 'output/work/P082'; FIG = OUT + '/figures'; CACHE = OUT + '/cache'
for d in (OUT, FIG, CACHE):
    os.makedirs(d, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument('--spectra-only', nargs='*', type=float, default=None)
ARGS = ap.parse_args()
LOG = open(f'{OUT}/run_log.txt', 'a')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n'); LOG.flush()
say(f'\n===== P082 run {time.strftime("%Y-%m-%d %H:%M:%S")} args={sys.argv[1:]} =====')

MV = lz.M_V_GEV; MN = lz.M_NUCLEON_GEV; EXPO = lz.LZ['exposure_tyr']
HBARC2_CM2 = 0.3894e-27                       # (hbar c)^2 [GeV^2 cm^2], recalled (certain)
MASSES = [200.0, 300.0, 400.0, 600.0, 1000.0, 2000.0, 4000.0, 10000.0]
P021_MASSES = {400.0, 1000.0, 4000.0}
VGRID = np.linspace(0.0, 830.0, 1661); ONES = np.ones_like(VGRID)
E_T = np.arange(1.5, 340.0, 3.0); DE_T = 3.0          # true recoil grid (P021)
E_O = np.arange(0.5, 340.0, 1.0)                        # observed grid, 1 keV
DOY_JUNE = 167
V_MAX_JUNE = lz.vmax_kms(lz.v_earth_kms(DOY_JUNE))
KAP = np.concatenate([[0.0], np.logspace(-7.5, 3.5, 551)])   # kappa grid (0 first)

def sigma_n_cm2(kappa, m):
    """LZ supplement: (c_1^s m_v^2)^2 = sigma_SI pi m_v^4 / mu_N^2."""
    mu = lz.mu_red(m, MN); return kappa * mu**2 / (math.pi * MV**4) * HBARC2_CM2

def ceiling_kev(m, v=V_MAX_JUNE):
    return 0.5 * lz.mu_red(m, lz.m_nucleus_gev(lz.A_XE_MEAN)) * (v / lz.C_KMS) ** 2 * 1e6

def delta_grid(m):
    c = ceiling_kev(m)
    if m in P021_MASSES:
        return np.concatenate([np.arange(100.0, 300.0, 10.0), np.arange(300.0, c, 5.0)])
    return np.concatenate([np.arange(200.0, 300.0, 10.0), np.arange(300.0, c, 5.0)])

# ------------------------------------------------------------------------------------------------
# 1. halos and spectra (per-stream WimPyDD kernels contracted with three halo functions)
# ------------------------------------------------------------------------------------------------
days12 = 15.0 + 365.25 / 12 * np.arange(12)
HALOS = {'annual': np.mean([lz.wd_halo(day_of_year=d, vmin=VGRID)[1] for d in days12], axis=0),
         'june': lz.wd_halo(day_of_year=DOY_JUNE, vmin=VGRID)[1],
         'sun': lz.wd_halo(vmin=VGRID)[1]}
COUPLING = lz.wd_c_from_anand(1.0 / MV**2, 0.0)          # WimPyDD (c0, c1) = (2/m_v^2, 0)

def spectra(m):
    if m in P021_MASSES:
        return dict(np.load(f'output/work/P021/spectra_s_{int(m)}.npz'))
    path = f'{CACHE}/spectra_s_{int(m)}.npz'
    if os.path.exists(path):
        return dict(np.load(path))
    WD = lz.wd(); ham = lz.wd_hamiltonian('O1_s_P082', {1: COUPLING})
    deltas = delta_grid(m); R = {h: np.zeros((len(deltas), len(E_T))) for h in HALOS}; ncalls = 0
    for i, d in enumerate(deltas):
        lo, hi = lz.E_R_range_keV(m, VGRID[-1], A=124.0, delta_kev=float(d))
        if math.isnan(lo):
            continue
        for j in np.where((E_T >= lo - DE_T) & (E_T <= hi + DE_T))[0]:
            K = WD.diff_rate(WD.Xe, ham, m, float(E_T[j]), VGRID, ONES, j_chi=0.5, delta=float(d),
                             sum_over_streams=False) * 1000.0 * 365.25
            ncalls += 1
            for h, deta in HALOS.items():
                R[h][i, j] = max(0.0, float(K @ deta))
    np.savez(path, deltas=deltas, E=E_T, **R)
    say(f'  spectra m={m:.0f}: {len(deltas)} deltas, {ncalls} kernel calls, t={time.time() - T0:.0f}s')
    return dict(np.load(path))

if ARGS.spectra_only is not None:
    for m in ARGS.spectra_only:
        spectra(float(m))
    say(f'spectra-only done, t={time.time() - T0:.0f}s'); sys.exit(0)

SPEC = {m: spectra(m) for m in MASSES}
HIG = dict(np.load('output/work/P021/spectra_hig_1000.npz'))       # P021: pure-Higgsino Hamiltonian, 1000 GeV
hig = json.load(open('output/work/P007/higgsino_couplings.json'))
KAPPA_HIG = {m: hig['sigma_SI_equiv_cm2_by_mass'][str(k)] / sigma_n_cm2(1.0, m)
             for m, k in zip(MASSES, [300, 300, 500, 500, 1000, 2000, 4000, 4000])}   # isoscalar-equivalent (c m_v^2)^2

# ------------------------------------------------------------------------------------------------
# 2. detector models
# ------------------------------------------------------------------------------------------------
p038 = pd.read_csv('output/work/P038/P038_mc_efficiency.csv').sort_values('E_keV')
def p_pass(E, edge=600):
    """P038 NEST-LZ Monte-Carlo pass probability for S1c < edge phd and log10 S2c < 4.15 (E >= 200 keV; 1 below)."""
    col = f'P_pass_{edge}'
    return np.where(np.asarray(E) < p038.E_keV.min(), 1.0, np.interp(E, p038.E_keV, p038[col], left=1.0, right=0.0))

def erf_up(E, e50, sig):
    return 0.5 * (1.0 + special.erf((np.asarray(E, float) - e50) / (math.sqrt(2) * sig)))

def eff_lz(E, model='p038', scale=1.0, edge=600):
    """LZ NR efficiency in true recoil energy.  model 'p038': 0.955 x MC roll-off (E50 = 271.6 keV for 600 phd);
    'erf': P021/P007 erf model (0.96, 50% at 269.9 keV, sigma 11.5).  scale rescales the energy axis (g1 change: SR1)."""
    E = np.asarray(E, float) * scale
    lo = erf_up(E, 5.4, 2.5)
    if model == 'p038':
        return 0.955 * lo * p_pass(E, edge)
    return 0.96 * lo * (1.0 - erf_up(E, 269.9, 11.5))

def eff_hard(E, e_lo, e_hi, plateau):
    return plateau * erf_up(E, e_lo, 1.5) * (np.asarray(E) < e_hi)

def sigma_E(E, s248=11.0):
    return s248 * np.sqrt(np.asarray(E) / 248.0)

SMEAR = stats.norm.pdf((E_O[:, None] - E_T[None, :]) / sigma_E(E_T)[None, :]) / sigma_E(E_T)[None, :] * DE_T
E_EV = 248.0
KERN_EV = stats.norm.pdf((E_EV - E_T) / sigma_E(E_T)) / sigma_E(E_T) * DE_T

def observe(r, model='p038'):
    """r: dR/dE_true (/t/yr/keV) at kappa = 1.  Observed-space LZ quantities (events in 2.84 t yr per unit kappa)."""
    w = r * eff_lz(E_T, model) * EXPO
    d = SMEAR @ w
    I = lambda a, b: float(np.sum(d[(E_O > a) & (E_O < b)]))
    S_L, S_M, S_H = I(5.4, 125.0), I(125.0, 200.0), I(200.0, 290.0)
    return dict(S_L=S_L, S_M=S_M, S_H=S_H, S_tot=S_L + S_M + S_H, f_ev=float(KERN_EV @ w), d=d)

# ------------------------------------------------------------------------------------------------
# 3. archival zero-event exposures (recalled; every entry flagged) and future datasets
# ------------------------------------------------------------------------------------------------
ARCH = {   # name: (exposure t yr, acceptance function in true energy, flag)
    'LZ SR1 2022 (NREFT 2023)': (60.0 * 5.5 / 365.25, lambda E: eff_lz(E, 'p038', scale=0.114 / 0.110),
        '60 live d x 5.5 t = 0.90 t yr (likely; P069 uses 0.77 with the 4.71 t FV); S1c 3-600 phd (paper l.116, certain); '
        'g1 = 0.114 vs 0.110 -> edge at 600 phd x 0.110/0.114 (likely); zero high-S1c NR-band events (uncertain)'),
    'XENON1T 2018': (1.0, lambda E: eff_hard(E, 4.9, 40.9, 0.85), '1.0 t yr (certain); ROI 4.9-40.9 keV (likely); plateau 0.85 (likely)'),
    'LUX 2021 EFT': (3.35e4 / 1000.0 / 365.25, lambda E: eff_hard(E, 3.0, 150.0, 0.8),
        '311.2 live d (certain), 3.35e4 kg d (likely); ROI edge 150 keV (uncertain; 100-250 scanned); plateau 0.8 (uncertain)'),
    'PandaX-II 2019 SD-EFT': (54.0 / 365.25, lambda E: eff_hard(E, 3.0, 100.0, 0.8),
        '54 t d (likely); ROI edge 100 keV (uncertain; 50-150 scanned); plateau 0.8 (uncertain)'),
    'XENONnT 2025 (3.1 t yr)': (3.1, lambda E: eff_hard(E, 3.0, 60.0, 0.9), '3.1 t yr (certain, cited by LZ); ROI cS1 < 100 PE ~ 60 keV (uncertain); plateau 0.9 (assumed)'),
    'PandaX-4T 2025 (1.54 t yr)': (1.54, lambda E: eff_hard(E, 3.0, 100.0, 0.9), '1.54 t yr (certain, cited by LZ); ROI edge ~100 keV (uncertain); plateau 0.9 (assumed)'),
}
ARCH_VARIANTS = {'LUX 2021 EFT': [('E_hi=100', lambda E: eff_hard(E, 3.0, 100.0, 0.8)), ('E_hi=250', lambda E: eff_hard(E, 3.0, 250.0, 0.8))],
                 'PandaX-II 2019 SD-EFT': [('E_hi=50', lambda E: eff_hard(E, 3.0, 50.0, 0.8)), ('E_hi=150', lambda E: eff_hard(E, 3.0, 150.0, 0.8))],
                 'LZ SR1 2022 (NREFT 2023)': [('erf edge 260.4', lambda E: eff_lz(E, 'erf', scale=0.114 / 0.110)), ('0.77 t yr', None), ('no 100 keV floor', 'nofloor')]}
for k, (e, _, fl) in ARCH.items():
    say(f'  archival {k:28s} {e:.4f} t yr : {fl}')

E_FLOOR = 100.0   # keV: the archival exposures are treated as zero-event data only above this true recoil energy; below it
                  # they were background-limited (P059 Sec. 4.4: SR1 limits at delta <= 150 keV imply 5-9 events; P035: XENON1T
                  # recast 11-1e5 x above LZ's edge), so their information is the published limits, which do not reach the region.
def n_true(r, expo, eff, floor=E_FLOOR):
    """expected events at kappa = 1 in true energy above the floor: expo x int_{E>floor} r eff dE"""
    return expo * float(np.sum(r * eff(E_T) * (E_T > floor)) * DE_T)

# future datasets (LZ-like acceptance assumed for the other TPCs, P035/P069; exposures recalled/flagged)
p046 = pd.read_csv('output/work/P046/rates_vs_delta.csv')          # unit-coupling rates per t yr of element, annual halo, 1000 GeV
p038full = dict(np.load('output/work/P038/spectra_s_1000_full.npz'))   # 1000 GeV O1s, E to 800 keV, 16 deltas 250-390
FUTURE = {'LZ +6.76 t yr, 600 phd edge': dict(expo=6.76, kind='lz600', flag='LZ data since Apr 2024 (P069/P020: 884 cal d x 0.593 live x 4.71 t; +-25%)'),
          'LZ +6.76 t yr, 1000 phd edge': dict(expo=6.76, kind='lz1000', flag='same exposure, P038 ROI extension (E50 = 423 keV)'),
          'XENONnT 3.1 t yr reanalysed to 270 keV': dict(expo=3.1, kind='lz600', flag='LZ-like acceptance assumed (P035/P069)'),
          'XENONnT 3.1 + 3.0 t yr (incl. untouched)': dict(expo=6.1, kind='lz600', flag='untouched 3.0 t yr recalled +-50% (P069)'),
          'PandaX-4T 1.54 t yr reanalysed to 270 keV': dict(expo=1.54, kind='lz600', flag='LZ-like acceptance assumed'),
          'PandaX-4T 1.54 + 2.5 t yr (incl. untouched)': dict(expo=4.04, kind='lz600', flag='untouched 2.5 t yr recalled +-50% (P069)'),
          'CaWO4 100 kg yr, W band 10-1300 keV': dict(expo=0.1 * 0.6385, kind='W', flag='P046 programme; tungsten mass fraction 0.6385; zero background assumed'),
          'CaWO4 10 kg yr': dict(expo=0.01 * 0.6385, kind='W', flag='P046 first stage')}

# ------------------------------------------------------------------------------------------------
# 4. LZ likelihood (P021) and the joint likelihood
# ------------------------------------------------------------------------------------------------
P16 = json.load(open('output/work/P016/P016_results.json'))['fig5_top_digitised']
lo_edges = np.array(P16['sigma_bin_low']); sel = lo_edges < 2.0 - 1e-9
B_BINS = np.array(P16['total'])[sel]; N_BINS = np.array(P16['data_counts'])[sel]
G_BINS = stats.norm.cdf(lo_edges[sel] + 0.5) - stats.norm.cdf(lo_edges[sel])
B_M, N_M, W_H, SIG_TH, B_H = 0.02, 0, 70.0, 0.3, 5.7e-4

def theta_hat(C):
    lo = np.full(C.shape[0], 1e-3); hi = np.full(C.shape[0], 30.0)
    for _ in range(60):
        th = 0.5 * (lo + hi)
        dl = np.sum(N_BINS * B_BINS / (th[:, None] * B_BINS + C), axis=1) - B_BINS.sum() - (th - 1) / SIG_TH**2
        lo = np.where(dl > 0, th, lo); hi = np.where(dl > 0, hi, th)
    return 0.5 * (lo + hi)

def lnL_LZ(k, sig, event=True):
    """Profile (over theta) ln L of the LZ data at coupling array k.  event=False: the 248 keV event is attributed to
    background (exclusion mode): its density term becomes ln(b_H/70) and only the Poisson exp(-kappa S) terms remain."""
    k = np.atleast_1d(k).astype(float)
    C = k[:, None] * sig['S_L'] * G_BINS[None, :]
    th = theta_hat(C); m_i = th[:, None] * B_BINS + C
    ll = np.sum(N_BINS * np.log(m_i) - m_i, axis=1) - 0.5 * ((th - 1) / SIG_TH) ** 2
    mM = B_M + k * sig['S_M']; ll = ll + N_M * np.log(mM) - mM
    ll = ll + (np.log(B_H / W_H + k * sig['f_ev']) if event else np.log(B_H / W_H)) - (B_H + k * sig['S_H'])
    return ll

def crossings(x, y, target, i_max):
    def cross(idx_range):
        for a, b in idx_range:
            if (y[a] - target) * (y[b] - target) <= 0 and y[a] != y[b]:
                if x[a] <= 0:
                    return 0.0
                la, lb = np.log(x[a]), np.log(x[b]); return float(np.exp(la + (target - y[a]) * (lb - la) / (y[b] - y[a])))
        return None
    lo = cross([(i - 1, i) for i in range(i_max, 0, -1)]); hi = cross([(i, i + 1) for i in range(i_max, len(x) - 1)])
    return (0.0 if lo is None else lo), (float('inf') if hi is None else hi)

def profile(ll, sig):
    """maximum, q0 vs kappa = 0, 68/90% 1-dof intervals in kappa"""
    i = int(np.argmax(ll)); q0 = max(0.0, 2 * (ll[i] - ll[0])) if i > 0 else 0.0
    k68 = crossings(KAP, ll, ll[i] - 0.5, i) if i > 0 else (0.0, crossings(KAP, ll, ll[0] - 0.5, 0)[1])
    k90 = crossings(KAP, ll, ll[i] - 1.353, i) if i > 0 else (0.0, crossings(KAP, ll, ll[0] - 1.353, 0)[1])
    return dict(llmax=float(ll[i]), kappa_hat=float(KAP[i]), q0=q0, Z=math.sqrt(q0), mu_hat=float(KAP[i] * sig['S_tot']),
                k68lo=k68[0], k68hi=k68[1], k90lo=k90[0], k90hi=k90[1])

def upper_limit(ll, dchi2):
    """kappa where -2[lnL(kappa) - max lnL] = dchi2 on the upper side (ll is unimodal on KAP)."""
    i = int(np.argmax(ll)); return crossings(KAP, ll, ll[i] - dchi2 / 2, i)[1]

# ------------------------------------------------------------------------------------------------
# 5. main scan: for every (m, delta): LZ-only and joint surfaces over kappa
# ------------------------------------------------------------------------------------------------
rows, SURF = [], {}
for m in MASSES:
    S = SPEC[m]; deltas = S['deltas']
    LL_lz = np.full((len(deltas), len(KAP)), np.nan); LL_j = LL_lz.copy(); LL_lz_ex = LL_lz.copy(); LL_j_ex = LL_lz.copy()
    N_ARCH = np.zeros((len(deltas), len(ARCH))); NFUT = {}
    for i, d in enumerate(deltas):
        r = S['annual'][i]
        sig = observe(r, 'p038')
        if sig['S_tot'] <= 0:
            continue
        Nj = np.array([n_true(r, e, f) for (e, f, _) in ARCH.values()]); N_ARCH[i] = Nj
        LL_lz[i] = lnL_LZ(KAP, sig); LL_j[i] = LL_lz[i] - KAP * Nj.sum()
        LL_lz_ex[i] = lnL_LZ(KAP, sig, event=False); LL_j_ex[i] = LL_lz_ex[i] - KAP * Nj.sum()
        f_lz, f_j = profile(LL_lz[i], sig), profile(LL_j[i], sig)
        row = dict(m_GeV=m, delta_keV=float(d), S_tot_unit=sig['S_tot'], S_H_unit=sig['S_H'], S_L_unit=sig['S_L'], f_ev_unit=sig['f_ev'],
                   N_arch_unit=float(Nj.sum()), N_SR1_unit=float(Nj[0]), N_arch_nonSR1_unit=float(Nj[1:].sum()),
                   Z_LZ=f_lz['Z'], Z_joint=f_j['Z'], q0_LZ=f_lz['q0'], q0_joint=f_j['q0'], llmax_LZ=f_lz['llmax'], llmax_joint=f_j['llmax'],
                   kappa_hat_LZ=f_lz['kappa_hat'], kappa_hat_joint=f_j['kappa_hat'], mu_hat_LZ=f_lz['mu_hat'], mu_hat_joint=f_j['mu_hat'],
                   k68lo_LZ=f_lz['k68lo'], k68hi_LZ=f_lz['k68hi'], k90lo_LZ=f_lz['k90lo'], k90hi_LZ=f_lz['k90hi'],
                   k68lo_joint=f_j['k68lo'], k68hi_joint=f_j['k68hi'], k90lo_joint=f_j['k90lo'], k90hi_joint=f_j['k90hi'],
                   N_arch_at_LZfit=f_lz['kappa_hat'] * float(Nj.sum()), N_SR1_at_LZfit=f_lz['kappa_hat'] * float(Nj[0]),
                   # exclusion mode: event treated as background (n_sig = 0 -> exact Poisson 2.303 events: dchi2 = 4.606)
                   UL_excl_LZ=upper_limit(LL_lz_ex[i], 4.606), UL_excl_joint=upper_limit(LL_j_ex[i], 4.606),
                   # event of unknown origin (n = 1): Poisson 90% UL 3.89 events <=> dchi2 = 2 (3.89 - 1 - ln 3.89) = 3.063 from the maximum
                   UL_n1_LZ=upper_limit(LL_lz[i], 3.063), UL_n1_joint=upper_limit(LL_j[i], 3.063))
        for key in ('kappa_hat_LZ', 'kappa_hat_joint', 'k90hi_joint', 'k90lo_joint', 'UL_excl_LZ', 'UL_excl_joint', 'UL_n1_LZ', 'UL_n1_joint'):
            row['sigma_' + key] = sigma_n_cm2(row[key], m) if np.isfinite(row[key]) else np.nan
        rows.append(row)
    SURF[m] = dict(deltas=deltas, LL_lz=LL_lz, LL_j=LL_j, LL_lz_ex=LL_lz_ex, LL_j_ex=LL_j_ex, N_arch=N_ARCH)
    say(f'  scan m={m:.0f}: {len(deltas)} deltas, t={time.time() - T0:.0f}s')
df = pd.DataFrame(rows); df.to_csv(f'{OUT}/P082_scan.csv', index=False)
LL0 = float(SURF[1000.0]['LL_lz'][0, 0])                   # ln L at kappa = 0 (identical for all m, delta, and for joint)

# ------------------------------------------------------------------------------------------------
# 6. regions
# ------------------------------------------------------------------------------------------------
RES = dict(settings=dict(exposure_tyr=EXPO, b_H=B_H, halo='annual mean of 12 monthly days (Baxter SHM)', edge='P038 MC roll-off, E50 271.6 keV, plateau 0.955',
                         sigma_E='11 sqrt(E/248) keV', masses=MASSES, kappa_grid='0 + logspace(-7.5, 3.5, 551)', E_true_step=DE_T))
# 6a. (m, delta) profiled over kappa: 2-dof regions relative to the global maximum
prof = df.groupby(['m_GeV', 'delta_keV'])[['llmax_LZ', 'llmax_joint']].first().reset_index()
reg_rows = []
for tag in ('LZ', 'joint'):
    col = f'llmax_{tag}'; gmax = prof[col].max(); gbest = prof.loc[prof[col].idxmax()]
    RES[f'global_best_{tag}'] = dict(m_GeV=float(gbest.m_GeV), delta_keV=float(gbest.delta_keV), lnLmax=float(gmax), q0=2 * (float(gmax) - LL0), Z=math.sqrt(2 * (float(gmax) - LL0)))
    for m in MASSES:
        sub = prof[prof.m_GeV == m].sort_values('delta_keV'); dm2 = 2 * (gmax - sub[col].values); dd = sub.delta_keV.values
        r = dict(m_GeV=m, region=tag, delta_best_at_m=float(dd[np.argmin(dm2)]), dm2_min_at_m=float(dm2.min()),
                 ceiling_keV=ceiling_kev(m), dmax248_keV=lz.delta_max_kev(248, m, v_kms=V_MAX_JUNE))
        for lev, nm in ((2.30, '68'), (4.61, '90'), (1.0, '68_1dof'), (2.706, '90_1dof')):
            ok = dm2 <= lev; r[f'delta_lo_{nm}'] = float(dd[ok].min()) if ok.any() else np.nan; r[f'delta_hi_{nm}'] = float(dd[ok].max()) if ok.any() else np.nan
            r[f'n_pts_{nm}'] = int(ok.sum())
        reg_rows.append(r)
REG_MD = pd.DataFrame(reg_rows); REG_MD.to_csv(f'{OUT}/P082_region_m_delta.csv', index=False)
say('(m, delta) regions:\n' + REG_MD[['m_GeV', 'region', 'delta_best_at_m', 'dm2_min_at_m', 'delta_lo_68', 'delta_hi_68', 'delta_lo_90', 'delta_hi_90', 'ceiling_keV']].to_string(index=False))

# 6b. (delta, kappa) at fixed mass (2-dof relative to the fixed-mass maximum), boundaries per delta
def region_dk(m, tag, LL=None):
    S = SURF[m]; LL = S[f'LL_{"lz" if tag == "LZ" else "j"}'] if LL is None else LL
    mx = np.nanmax(LL); dm2 = 2 * (mx - LL); out = []
    for i, d in enumerate(S['deltas']):
        row = dict(m_GeV=m, region=tag, delta_keV=float(d))
        for lev, nm in ((2.30, '68'), (4.61, '90')):
            ok = np.where(dm2[i] <= lev)[0]
            if len(ok):
                row[f'kappa_lo_{nm}'] = float(KAP[ok.min()]); row[f'kappa_hi_{nm}'] = float(KAP[ok.max()])
                row[f'sigma_lo_{nm}_cm2'] = sigma_n_cm2(row[f'kappa_lo_{nm}'], m); row[f'sigma_hi_{nm}_cm2'] = sigma_n_cm2(row[f'kappa_hi_{nm}'], m)
            else:
                row[f'kappa_lo_{nm}'] = row[f'kappa_hi_{nm}'] = row[f'sigma_lo_{nm}_cm2'] = row[f'sigma_hi_{nm}_cm2'] = np.nan
        out.append(row)
    return pd.DataFrame(out), dm2

DK = []; DM2 = {}
for m in (1000.0, 4000.0):
    for tag in ('LZ', 'joint'):
        t, dm2 = region_dk(m, tag); DK.append(t); DM2[(m, tag)] = dm2
DK = pd.concat(DK); DK.to_csv(f'{OUT}/P082_region_delta_sigma.csv', index=False)
def area(dm2, deltas, lev=4.61):
    w = np.gradient(deltas); dlk = np.log10(KAP[2] / KAP[1]); return float(np.nansum((dm2 <= lev).sum(axis=1) * w) * dlk)
for m in (1000.0, 4000.0):
    for tag in ('LZ', 'joint'):
        sub = DK[(DK.m_GeV == m) & (DK.region == tag)]
        in68 = sub.dropna(subset=['kappa_lo_68']); in90 = sub.dropna(subset=['kappa_lo_90'])
        RES[f'region_delta_sigma_{int(m)}_{tag}'] = dict(delta68=[float(in68.delta_keV.min()), float(in68.delta_keV.max())], delta90=[float(in90.delta_keV.min()), float(in90.delta_keV.max())],
            kappa68=[float(in68.kappa_lo_68.min()), float(in68.kappa_hi_68.max())], kappa90=[float(in90.kappa_lo_90.min()), float(in90.kappa_hi_90.max())],
            sigma68_cm2=[sigma_n_cm2(float(in68.kappa_lo_68.min()), m), sigma_n_cm2(float(in68.kappa_hi_68.max()), m)],
            sigma90_cm2=[sigma_n_cm2(float(in90.kappa_lo_90.min()), m), sigma_n_cm2(float(in90.kappa_hi_90.max()), m)],
            area90_keV_dex=area(DM2[(m, tag)], SURF[m]['deltas']), area68_keV_dex=area(DM2[(m, tag)], SURF[m]['deltas'], 2.30))
        say(f'  (delta, sigma) region m={m:.0f} {tag}: ' + json.dumps({k: (np.round(v, 4).tolist() if isinstance(v, list) else round(v, 3)) for k, v in RES[f'region_delta_sigma_{int(m)}_{tag}'].items()}))

# 6c. best-fit shift (P059 comparison) at selected points and at every mass' best delta
sel = df[df.delta_keV.isin([300, 350, 366, 370, 380, 385, 390]) | df.apply(lambda r: np.isclose(r.delta_keV, REG_MD[(REG_MD.m_GeV == r.m_GeV) & (REG_MD.region == 'joint')].delta_best_at_m.iloc[0]), axis=1)].copy()
sel['kappa_ratio'] = sel.kappa_hat_joint / sel.kappa_hat_LZ; sel['dZ'] = sel.Z_joint - sel.Z_LZ; sel['k90hi_ratio'] = sel.k90hi_joint / sel.k90hi_LZ
sel['UL_excl_ratio'] = sel.UL_excl_joint / sel.UL_excl_LZ; sel['UL_n1_ratio'] = sel.UL_n1_joint / sel.UL_n1_LZ
sel[['m_GeV', 'delta_keV', 'Z_LZ', 'Z_joint', 'dZ', 'kappa_hat_LZ', 'kappa_hat_joint', 'kappa_ratio', 'mu_hat_LZ', 'mu_hat_joint', 'N_arch_at_LZfit', 'N_SR1_at_LZfit',
     'k90hi_ratio', 'UL_excl_LZ', 'UL_excl_joint', 'UL_excl_ratio', 'UL_n1_ratio', 'sigma_kappa_hat_joint', 'sigma_UL_excl_LZ', 'sigma_UL_excl_joint']].to_csv(f'{OUT}/P082_bestfit_shift.csv', index=False, float_format='%.5g')
s1k = sel[sel.m_GeV == 1000]
say('best-fit shift, 1000 GeV:\n' + s1k[['delta_keV', 'Z_LZ', 'Z_joint', 'dZ', 'kappa_hat_LZ', 'kappa_hat_joint', 'kappa_ratio', 'mu_hat_joint', 'N_arch_at_LZfit', 'N_SR1_at_LZfit', 'k90hi_ratio', 'UL_excl_ratio']].to_string(index=False))
hi = df[(df.delta_keV >= 300) & (df.kappa_hat_LZ > 0)]; lo = df[(df.delta_keV < 300) & (df.kappa_hat_LZ > 0)]
RES['shift_summary'] = dict(delta_ge_300=dict(kappa_ratio_range=[float((hi.kappa_hat_joint / hi.kappa_hat_LZ).min()), float((hi.kappa_hat_joint / hi.kappa_hat_LZ).max())],
                                              dZ_range=[float((hi.Z_joint - hi.Z_LZ).min()), float((hi.Z_joint - hi.Z_LZ).max())],
                                              N_arch_at_LZfit_range=[float(hi.N_arch_at_LZfit.min()), float(hi.N_arch_at_LZfit.max())],
                                              N_SR1_fraction_of_arch=float((hi.N_SR1_at_LZfit / hi.N_arch_at_LZfit).mean())),
                            delta_lt_300=dict(kappa_ratio_range=[float((lo.kappa_hat_joint / lo.kappa_hat_LZ).min()), float((lo.kappa_hat_joint / lo.kappa_hat_LZ).max())],
                                              dZ_range=[float((lo.Z_joint - lo.Z_LZ).min()), float((lo.Z_joint - lo.Z_LZ).max())],
                                              N_arch_at_LZfit_range=[float(lo.N_arch_at_LZfit.min()), float(lo.N_arch_at_LZfit.max())]))
say('shift summary:', json.dumps(RES['shift_summary']))
say('low-delta rows (1000 GeV):\n' + df[(df.m_GeV == 1000) & (df.delta_keV <= 250)][['delta_keV', 'Z_LZ', 'Z_joint', 'kappa_hat_LZ', 'kappa_hat_joint', 'N_arch_at_LZfit', 'N_SR1_at_LZfit', 'N_arch_nonSR1_unit']].to_string(index=False))
# archival variants (1000 GeV, delta = 300/350/366/380): other ROI edges, 0.77 t yr SR1, erf SR1 edge
var_rows = []
S1 = SPEC[1000.0]
for d in (250.0, 300.0, 350.0, 365.0, 380.0):
    i = int(np.argmin(np.abs(S1['deltas'] - d))); r = S1['annual'][i]; kh = float(df[(df.m_GeV == 1000) & np.isclose(df.delta_keV, S1['deltas'][i])].kappa_hat_LZ.iloc[0])
    for k, vs in ARCH_VARIANTS.items():
        base_n = n_true(r, ARCH[k][0], ARCH[k][1])
        for nm, f in vs:
            n = n_true(r, 0.77, ARCH[k][1]) if f is None else (n_true(r, ARCH[k][0], ARCH[k][1], floor=0.0) if f == 'nofloor' else n_true(r, ARCH[k][0], f))
            var_rows.append(dict(delta_keV=float(S1['deltas'][i]), experiment=k, variant=nm, N_unit_base=base_n, N_unit_variant=n, N_at_LZfit_base=kh * base_n, N_at_LZfit_variant=kh * n))
VAR = pd.DataFrame(var_rows); VAR.to_csv(f'{OUT}/P082_archival_variants.csv', index=False, float_format='%.4g')
say('archival variants:\n' + VAR.to_string(index=False))
# per-experiment counts at the LZ best fit, 1000 GeV
cnt = []
for d in (250.0, 300.0, 350.0, 365.0, 380.0, 390.0):
    i = int(np.argmin(np.abs(S1['deltas'] - d))); kh = float(df[(df.m_GeV == 1000) & np.isclose(df.delta_keV, S1['deltas'][i])].kappa_hat_LZ.iloc[0])
    cnt.append(dict(delta_keV=float(S1['deltas'][i]), kappa_hat_LZ=kh, **{k: kh * SURF[1000.0]['N_arch'][i, j] for j, k in enumerate(ARCH)}))
CNT = pd.DataFrame(cnt); CNT['total'] = CNT[list(ARCH)].sum(axis=1); CNT['P0'] = np.exp(-CNT.total)
CNT.to_csv(f'{OUT}/P082_archival_counts_1000GeV.csv', index=False, float_format='%.4g'); say('archival counts at LZ fit (1000 GeV):\n' + CNT.to_string(index=False))

# ------------------------------------------------------------------------------------------------
# 7. Higgsino: fixed physical coupling (P007/P021 Hamiltonian at 1000 GeV) + archival zeros; kappa_H line at other masses
# ------------------------------------------------------------------------------------------------
hrows = []
for i, d in enumerate(HIG['deltas']):
    r = HIG['annual'][i]; sig = observe(r, 'p038')
    Nj = np.array([n_true(r, e, f) for (e, f, _) in ARCH.values()])
    ll_lz = float(lnL_LZ(np.array([1.0]), sig)[0]) if sig['S_tot'] > 0 else float(lnL_LZ(np.array([0.0]), sig)[0])
    hrows.append(dict(delta_keV=float(d), N_LZ=sig['S_tot'], N_arch=float(Nj.sum()), N_SR1=float(Nj[0]), lnL_LZ=ll_lz, lnL_joint=ll_lz - float(Nj.sum())))
HD = pd.DataFrame(hrows)
hsum = {}
for tag in ('LZ', 'joint'):
    c = f'lnL_{tag}'; HD[f'dm2_{tag}'] = 2 * (HD[c].max() - HD[c]); b = HD.loc[HD[c].idxmax()]
    rng = lambda lev: [float(HD[HD[f'dm2_{tag}'] <= lev].delta_keV.min()), float(HD[HD[f'dm2_{tag}'] <= lev].delta_keV.max())]
    hsum[tag] = dict(delta_best=float(b.delta_keV), N_LZ_at_best=float(b.N_LZ), N_arch_at_best=float(b.N_arch), range68=rng(1.0), range90=rng(2.706),
                     q_vs_bkg_at_best=2 * (float(b[c]) - LL0), Z_at_best=math.sqrt(max(0.0, 2 * (float(b[c]) - LL0))))
HD.to_csv(f'{OUT}/P082_higgsino_1000GeV.csv', index=False, float_format='%.5g')
RES['higgsino_fixed_1000'] = hsum; say('Higgsino fixed coupling (1000 GeV):', json.dumps(hsum))
# kappa_H line inside the joint 68/90% (delta, kappa) regions at every mass
hl = {}
for m in MASSES:
    t, dm2 = region_dk(m, 'joint'); kH = KAPPA_HIG[m]
    for nm in ('68', '90'):
        ins = t[(t[f'kappa_lo_{nm}'] <= kH) & (kH <= t[f'kappa_hi_{nm}'])]
        hl[f'{int(m)}_{nm}'] = [float(ins.delta_keV.min()), float(ins.delta_keV.max())] if len(ins) else None
    hl[f'{int(m)}_kappaH'] = kH
RES['higgsino_line_joint'] = hl; say('Higgsino line inside joint regions:', json.dumps(hl))

# ------------------------------------------------------------------------------------------------
# 8. non-xenon published limits (LZ Fig. S7 digitised by P015): CRESST-II (W) and PICO-60 CF3I (I) vs the region at 1 TeV
# ------------------------------------------------------------------------------------------------
fs7 = json.load(open('output/work/P015/figS7_curves.json'))['curves_points']
SIG1 = sigma_n_cm2(1.0, 1000.0)
nonxe = {}
for nm, key in (('CRESST-II 2016 (orange)', 'CRESST'), ('PICO 2023 (red)', 'PICO')):
    pts = np.array(fs7[nm]); dd, ss = pts[:, 0], pts[:, 1]
    ok = dd >= 300; slope = np.polyfit(dd[ok], np.log10(ss[ok]), 1)[0]            # empirical dex/keV above 300 keV
    f_lim = lambda x, dd=dd, ss=ss, slope=slope: np.where(x <= dd.max(), 10 ** np.interp(x, dd, np.log10(ss)), ss[-1] * 10 ** (slope * (x - dd.max())))
    t = DK[(DK.m_GeV == 1000) & (DK.region == 'joint')].dropna(subset=['kappa_hi_90'])
    lim_k = f_lim(t.delta_keV.values) / SIG1
    ratio90 = lim_k / t.kappa_hi_90.values; ratio_hat = lim_k / df[(df.m_GeV == 1000) & df.delta_keV.isin(t.delta_keV)].kappa_hat_joint.values
    nonxe[key] = dict(curve_end_keV=float(dd.max()), sigma_at_end=float(ss[-1]), slope_dex_per_keV=float(slope),
                      min_ratio_limit_over_90hi=float(np.nanmin(ratio90)), delta_of_min=float(t.delta_keV.values[np.nanargmin(ratio90)]),
                      ratio_limit_over_90hi_by_delta={str(int(d)): float(x) for d, x in zip(t.delta_keV.values, ratio90) if d >= 330},
                      ratio_limit_over_kappahat_by_delta={str(int(d)): float(x) for d, x in zip(t.delta_keV.values, ratio_hat) if d >= 330},
                      extrapolated_beyond_end=True)
# implied PICO-60 CF3I iodine counts (estimate): recalled exposure 1335 kg d (likely), iodine mass fraction 127/196.9
E_I_PICO = 1335.0 / 1000.0 / 365.25 * 127.0 / 196.9
imp = {}
for d in (250, 300, 350):
    sP = float(json.load(open('output/work/P015/figS7_curves.json'))['curves_summary']['PICO 2023 (red)']['sigma_at'][str(d)])
    rI = float(p046[p046.delta_keV == d].I_5_1000_annual.iloc[0])
    imp[str(d)] = dict(sigma_lim=sP, kappa_lim=sP / SIG1, implied_events=sP / SIG1 * rI * E_I_PICO)
nonxe['PICO_implied_iodine_events'] = dict(iodine_exposure_tyr=E_I_PICO, by_delta=imp, iodine_ceiling_1TeV_keV=0.5 * lz.mu_red(1000.0, 127 * lz.AMU_GEV) * (V_MAX_JUNE / lz.C_KMS) ** 2 * 1e6)
nonxe['Ge_ceiling_keV_infinite_mass'] = 0.5 * 73 * lz.AMU_GEV * (V_MAX_JUNE / lz.C_KMS) ** 2 * 1e6
RES['non_xenon'] = nonxe; say('non-xenon:', json.dumps(nonxe, default=float)[:1500])

# ------------------------------------------------------------------------------------------------
# 9. future exposures at 1000 GeV: Asimov (best fit true) and zero-event outcomes; 90% region area shrinkage
# ------------------------------------------------------------------------------------------------
S1 = SPEC[1000.0]; d1 = S1['deltas']; LLj1 = SURF[1000.0]['LL_j']
# unit-kappa expected counts N_X(delta) for each future dataset
n_lz600 = np.array([observe(S1['annual'][i], 'p038')['S_tot'] / EXPO for i in range(len(d1))])       # per t yr, LZ-like 600 phd
E_F = p038full['E']; dEF = float(E_F[1] - E_F[0]); dF = p038full['deltas']
n1000_16 = np.array([float(np.sum(p038full['annual'][i] * 0.955 * erf_up(E_F, 5.4, 2.5) * p_pass(E_F, 1000)) * dEF) for i in range(len(dF))])
n600_16 = np.array([float(np.sum(p038full['annual'][i] * 0.955 * erf_up(E_F, 5.4, 2.5) * p_pass(E_F, 600)) * dEF) for i in range(len(dF))])
gain1000 = np.interp(d1, dF, n1000_16 / n600_16, left=1.0, right=(n1000_16 / n600_16)[-1])
n_lz1000 = n_lz600 * gain1000
n_W = np.interp(d1, p046.delta_keV, p046.W_10_1300_annual, left=0.0, right=0.0)      # per t yr of tungsten, unit kappa
NX = {'lz600': n_lz600, 'lz1000': n_lz1000, 'W': n_W}
# two "truths" for the Asimov projections: (A) the 1 TeV joint maximum; (B) the Higgsino point (delta = 370 keV, kappa = kappa_H)
i_b1 = np.unravel_index(np.nanargmax(LLj1), LLj1.shape); ib, kb = int(i_b1[0]), float(KAP[i_b1[1]])
iH = int(np.argmin(np.abs(d1 - 370.0))); kH = KAPPA_HIG[1000.0]
A0 = area(2 * (np.nanmax(LLj1) - LLj1), d1)
def region_stats(LL, lab):
    mx = np.nanmax(LL); dm2 = 2 * (mx - LL); ok = dm2 <= 4.61
    prof_d = np.nanmax(LL, axis=1); ok_d = 2 * (mx - prof_d) <= 4.61
    i_b = np.unravel_index(np.nanargmax(LL), LL.shape)
    return dict(area90=area(dm2, d1), delta90=[float(d1[ok_d].min()), float(d1[ok_d].max())] if ok_d.any() else None,
                delta_best=float(d1[i_b[0]]), kappa_best=float(KAP[i_b[1]]),
                kappa90_at_best_delta=[float(KAP[ok[i_b[0]]].min()), float(KAP[ok[i_b[0]]].max())] if ok[i_b[0]].any() else None,
                Z=math.sqrt(max(0.0, 2 * (mx - LL0))) if lab == 'zero' else None)
def asimov_LL(NXd, nA):
    with np.errstate(divide='ignore', invalid='ignore'):
        mu = KAP[None, :] * NXd[:, None]
        return LLj1 + np.where(mu > 0, nA * np.log(np.where(mu > 0, mu, 1.0)), (-np.inf if nA > 0 else 0.0)) - mu
fut_rows = []
for nm, cfg in FUTURE.items():
    NXd = NX[cfg['kind']] * cfg['expo']
    exp_events = {str(int(d)): kb_d * NXd[int(np.argmin(np.abs(d1 - d)))] for d, kb_d in
                  [(d, float(df[(df.m_GeV == 1000) & np.isclose(df.delta_keV, d)].kappa_hat_joint.iloc[0])) for d in (300.0, 350.0, 365.0, 380.0)]}
    nA, nH = kb * NXd[ib], kH * NXd[iH]
    outA = region_stats(asimov_LL(NXd, nA), 'asimov'); outH = region_stats(asimov_LL(NXd, nH), 'asimovH'); out0 = region_stats(LLj1 - KAP[None, :] * NXd[:, None], 'zero')
    fut_rows.append(dict(dataset=nm, exposure=cfg['expo'], kind=cfg['kind'], flag=cfg['flag'], N_at_joint_best=nA, N_at_higgsino=nH, **{f'N_at_fit_d{k}': v for k, v in exp_events.items()},
                         P0_at_best=math.exp(-nA), area90_now=A0, area90_asimov=outA['area90'], frac_asimov=outA['area90'] / A0, delta90_asimov=outA['delta90'], kappa90_asimov=outA['kappa90_at_best_delta'],
                         area90_asimovH=outH['area90'], frac_asimovH=outH['area90'] / A0, delta90_asimovH=outH['delta90'], kappa90_asimovH=outH['kappa90_at_best_delta'],
                         area90_zero=out0['area90'], frac_zero=out0['area90'] / A0, delta90_zero=out0['delta90'], kappa90_zero=out0['kappa90_at_best_delta'], Z_if_zero=out0['Z'], delta_best_zero=out0['delta_best']))
FUT = pd.DataFrame(fut_rows); FUT.to_csv(f'{OUT}/P082_future_exposures_1000GeV.csv', index=False, float_format='%.4g')
say('future (1000 GeV; truth A: joint best fit delta=%.0f kappa=%.3g; truth B: Higgsino delta=370 kappa=%.4f; area90 now %.2f keV dex):\n' % (d1[ib], kb, kH, A0) +
    FUT[['dataset', 'N_at_joint_best', 'N_at_higgsino', 'N_at_fit_d350', 'N_at_fit_d365', 'frac_asimov', 'delta90_asimov', 'frac_asimovH', 'delta90_asimovH', 'frac_zero', 'delta90_zero', 'kappa90_zero', 'Z_if_zero']].to_string(index=False))
RES['future'] = dict(truthA=dict(delta=float(d1[ib]), kappa=kb), truthB=dict(delta=float(d1[iH]), kappa=kH), area90_now_keV_dex=A0,
                     gain1000_at_350_380=[float(np.interp(350, d1, gain1000)), float(np.interp(380, d1, gain1000))])
say('exclusion-mode 90% UL on sigma_n (cm^2), event treated as background, 1000 GeV:\n' +
    df[(df.m_GeV == 1000) & df.delta_keV.isin([200, 250, 300, 330, 350, 365, 370, 380, 390])][['delta_keV', 'UL_excl_LZ', 'UL_excl_joint', 'sigma_UL_excl_LZ', 'sigma_UL_excl_joint', 'UL_n1_joint', 'sigma_UL_n1_joint', 'k90hi_joint', 'sigma_k90hi_joint']].to_string(index=False))
say('4000 GeV:\n' + df[(df.m_GeV == 4000) & df.delta_keV.isin([300, 350, 380, 400, 410])][['delta_keV', 'Z_LZ', 'Z_joint', 'kappa_hat_joint', 'sigma_kappa_hat_joint', 'UL_excl_joint', 'sigma_UL_excl_joint']].to_string(index=False))

# ------------------------------------------------------------------------------------------------
# 10. robustness at 1000 GeV: P021 erf edge; June and Sun-frame halos (joint)
# ------------------------------------------------------------------------------------------------
rob = {}
for nm, model, halo in (('erf_edge', 'erf', 'annual'), ('june', 'p038', 'june'), ('sun', 'p038', 'sun')):
    LLr = np.full((len(d1), len(KAP)), np.nan); zz = []
    for i, d in enumerate(d1):
        r = S1[halo][i]; sig = observe(r, model)
        if sig['S_tot'] <= 0:
            zz.append(0.0); continue
        Nj = sum(n_true(r, e, f) for (e, f, _) in ARCH.values()); LLr[i] = lnL_LZ(KAP, sig) - KAP * Nj; zz.append(profile(LLr[i], sig)['Z'])
    mx = np.nanmax(LLr); dm2 = 2 * (mx - LLr); ok_d = (2 * (mx - np.nanmax(LLr, axis=1))) <= 4.61; ok68 = (2 * (mx - np.nanmax(LLr, axis=1))) <= 2.30
    ib_ = np.unravel_index(np.nanargmax(LLr), LLr.shape)
    rob[nm] = dict(delta_best=float(d1[ib_[0]]), kappa_best=float(KAP[ib_[1]]), Z_best=float(np.nanmax(zz)), delta68=[float(d1[ok68].min()), float(d1[ok68].max())],
                   delta90=[float(d1[ok_d].min()), float(d1[ok_d].max())], area90=area(dm2, d1), Z_300=float(zz[int(np.argmin(np.abs(d1 - 300)))]), Z_350=float(zz[int(np.argmin(np.abs(d1 - 350)))]))
RES['robustness_1000_joint'] = rob; say('robustness:', json.dumps(rob))

# selected 1000 GeV rows for the record
RES['selected_1000'] = df[(df.m_GeV == 1000) & df.delta_keV.isin([300, 330, 350, 360, 365, 370, 375, 380, 385, 390, 395])][
    ['delta_keV', 'Z_LZ', 'Z_joint', 'kappa_hat_LZ', 'kappa_hat_joint', 'k90lo_joint', 'k90hi_joint', 'sigma_kappa_hat_joint', 'N_arch_at_LZfit', 'UL_excl_LZ', 'UL_excl_joint', 'sigma_UL_excl_joint', 'UL_n1_joint']].to_dict('records')
RES['selected_4000'] = df[(df.m_GeV == 4000) & df.delta_keV.isin([300, 350, 380, 400, 410, 420])][
    ['delta_keV', 'Z_LZ', 'Z_joint', 'kappa_hat_LZ', 'kappa_hat_joint', 'k90lo_joint', 'k90hi_joint', 'sigma_kappa_hat_joint', 'UL_excl_joint', 'sigma_UL_excl_joint']].to_dict('records')
RES['runtime_s'] = time.time() - T0
json.dump(RES, open(f'{OUT}/P082_results.json', 'w'), indent=1, default=float)
np.savez(f'{OUT}/P082_surfaces.npz', kappa=KAP, **{f'deltas_{int(m)}': SURF[m]['deltas'] for m in MASSES}, **{f'LLj_{int(m)}': SURF[m]['LL_j'] for m in MASSES},
         **{f'LLlz_{int(m)}': SURF[m]['LL_lz'] for m in MASSES}, ll0=LL0)

# ------------------------------------------------------------------------------------------------
# 11. figures (Okabe-Ito, fixed assignment: LZ-only = blue #0072B2, joint = vermilion #D55E00, Higgsino = green #009E73,
#     non-xenon = purple #CC79A7 / orange #E69F00; recessive grid)
# ------------------------------------------------------------------------------------------------
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
C_LZ, C_J, C_H, C_P, C_C, C_G = '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#666666'
plt.rcParams.update({'font.size': 9.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.2, 'grid.linewidth': 0.6})

# Fig 1: (m, delta) regions
fig, ax = plt.subplots(figsize=(7.4, 4.6))
for tag, col, lab in (('LZ', C_LZ, 'LZ only'), ('joint', C_J, 'LZ + archival zeros (joint)')):
    sub = REG_MD[REG_MD.region == tag].sort_values('m_GeV')
    for nm, alpha in (('90', 0.25), ('68', 0.55)):
        ok = sub.dropna(subset=[f'delta_lo_{nm}'])
        ax.fill_between(ok.m_GeV, ok[f'delta_lo_{nm}'], ok[f'delta_hi_{nm}'], color=col, alpha=alpha, lw=0, label=f'{lab}: {nm}% (2 dof)' if True else None)
    ax.plot(sub.m_GeV, sub.delta_best_at_m, '-', color=col, lw=1.5)
ax.plot(REG_MD[REG_MD.region == 'joint'].m_GeV, REG_MD[REG_MD.region == 'joint'].ceiling_keV, ':', color=C_G, lw=1.2, label='kinematic ceiling μv²_max/2 (June)')
ax.plot(REG_MD[REG_MD.region == 'joint'].m_GeV, REG_MD[REG_MD.region == 'joint'].dmax248_keV, '--', color=C_G, lw=1.0, label='δ_max(248 keV)')
gb = RES['global_best_joint']; ax.plot([gb['m_GeV']], [gb['delta_keV']], '*', color='#222', ms=11, label=f'joint maximum ({gb["m_GeV"]:.0f} GeV, {gb["delta_keV"]:.0f} keV)')
ax.set_xscale('log'); ax.set_xlim(180, 11000); ax.set_ylim(250, 450)
ax.set_xlabel('WIMP mass m_χ (GeV)'); ax.set_ylabel('mass splitting δ (keV)')
ax.set_title('Isoscalar O₁: (m_χ, δ) profiled over σ_n — LZ event vs. global xenon combination', fontsize=10)
ax.legend(fontsize=7.5, loc='lower right', frameon=False, ncol=2)
fig.tight_layout(); fig.savefig(f'{FIG}/P082_fig1_m_delta_region.png', dpi=160); plt.close(fig)

# Fig 2: (delta, sigma) at 1 and 4 TeV
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=False)
for ax, m in zip(axs, (1000.0, 4000.0)):
    S = SURF[m]; Dg, Kg = np.meshgrid(S['deltas'], KAP[1:], indexing='ij'); conv = sigma_n_cm2(1.0, m)
    for tag, col in (('LZ', C_LZ), ('joint', C_J)):
        dm2 = DM2[(m, tag)][:, 1:]
        ax.contour(Dg, Kg * conv, np.nan_to_num(dm2, nan=1e9), levels=[4.61], colors=[col], linewidths=1.2, linestyles='--')
        ax.contourf(Dg, Kg * conv, np.nan_to_num(dm2, nan=1e9), levels=[0, 2.30], colors=[col], alpha=0.45 if tag == 'joint' else 0.25, antialiased=True)
    sub = df[df.m_GeV == m].sort_values('delta_keV')
    ax.plot(sub.delta_keV, sub.sigma_kappa_hat_joint, '-', color=C_J, lw=1.6, label='joint best fit σ̂_n(δ)')
    ax.plot(sub.delta_keV, sub.sigma_UL_excl_joint, '-', color='#222', lw=1.3, label='exclusion mode: joint 90% UL (event = background)')
    ax.plot(sub.delta_keV, sub.sigma_UL_excl_LZ, ':', color='#222', lw=1.1, label='exclusion mode: LZ only')
    ax.axhline(sigma_n_cm2(KAPPA_HIG[m], m), color=C_H, ls='--', lw=1.6, label='pure Higgsino σ_SI,eq (P007)')
    if m == 1000.0:
        for nm, col, lab in (('CRESST-II 2016 (orange)', C_C, 'CRESST-II (W) limit, LZ Fig. S7'), ('PICO 2023 (red)', C_P, 'PICO-60 CF3I (I) limit, LZ Fig. S7')):
            pts = np.array(fs7[nm]); ok = pts[:, 0] >= 100
            ax.plot(pts[ok, 0], pts[ok, 1], '-', color=col, lw=1.4, label=lab)
            xx = np.arange(pts[-1, 0], 396, 1.0); sl = nonxe['CRESST' if 'CRESST' in nm else 'PICO']['slope_dex_per_keV']
            ax.plot(xx, pts[-1, 1] * 10 ** (sl * (xx - pts[-1, 0])), ':', color=col, lw=1.2)
    ax.plot([], [], 's', color=C_LZ, alpha=0.3, label='LZ only 68% (fill) / 90% (dashed)'); ax.plot([], [], 's', color=C_J, alpha=0.5, label='joint 68% / 90%')
    ax.set_yscale('log'); ax.set_ylim(1e-46, 1e-34); ax.set_xlim(200, S['deltas'].max() + 2)
    ax.set_xlabel('mass splitting δ (keV)'); ax.set_ylabel('σ_n (cm²)'); ax.set_title(f'm_χ = {m:.0f} GeV', fontsize=10)
    ax.legend(fontsize=6.8, loc='lower right', frameon=False)
fig.suptitle('(δ, σ_n) regions: LZ event alone vs. joint with archival zero-event xenon exposures; exclusion-mode limits', fontsize=10)
fig.tight_layout(); fig.savefig(f'{FIG}/P082_fig2_delta_sigma.png', dpi=160); plt.close(fig)

# Fig 3: future exposures - 90% region area fraction (Asimov and zero) and expected counts
SHORT = ['LZ +6.76 t·yr, 600 phd', 'LZ +6.76 t·yr, 1000 phd', 'XENONnT 3.1 t·yr →270 keV', 'XENONnT 6.1 t·yr (+untouched)', 'PandaX-4T 1.54 t·yr →270 keV',
         'PandaX-4T 4.0 t·yr (+untouched)', 'CaWO₄ 100 kg·yr (W 10–1300 keV)', 'CaWO₄ 10 kg·yr']
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw=dict(width_ratios=[1.45, 1]))
y = np.arange(len(FUT))[::-1]; h = 0.27
a1.barh(y + h, FUT.frac_asimov, h, color=C_J, label='expected if the 1 TeV best fit (δ = 380 keV) is true')
a1.barh(y, FUT.frac_asimovH, h, color=C_H, label='expected if the Higgsino point (δ = 370 keV) is true')
a1.barh(y - h, np.minimum(FUT.frac_zero, 1.6), h, color=C_LZ, label='if zero events are found (region re-centres at lower δ)')
for yy, fa, fh, fz in zip(y, FUT.frac_asimov, FUT.frac_asimovH, FUT.frac_zero):
    a1.text(fa + 0.01, yy + h, f'{fa:.2f}', va='center', fontsize=7, color='#333'); a1.text(fh + 0.01, yy, f'{fh:.2f}', va='center', fontsize=7, color='#333')
    a1.text(min(fz, 1.6) + 0.01, yy - h, f'{fz:.2f}', va='center', fontsize=7, color='#333')
a1.set_yticks(y); a1.set_yticklabels(SHORT, fontsize=8); a1.set_xlim(0, 2.6); a1.set_xlabel('90% (δ, log σ_n) region area at 1 TeV, relative to today (joint); zero-event bars clipped at 1.6')
a1.legend(fontsize=7.2, frameon=False, loc='upper right'); a1.set_title('Which exposure shrinks the surviving region most?', fontsize=10)
xs = np.arange(len(FUT)); w = 0.2
for j, (dcol, lab, col) in enumerate((('N_at_fit_d300', 'δ = 300', '#56B4E9'), ('N_at_fit_d350', '350', C_LZ), ('N_at_fit_d365', '365', C_J), ('N_at_fit_d380', '380 keV', '#222'))):
    a2.bar(xs + (j - 1.5) * w, np.maximum(FUT[dcol], 1e-3), w, color=col, label=lab)
a2.set_yscale('log'); a2.set_ylim(1e-3, 3e3); a2.set_xticks(xs); a2.set_xticklabels(SHORT, rotation=35, ha='right', fontsize=7)
a2.axhline(2.303, color=C_G, ls=':', lw=1); a2.text(len(FUT) - 0.5, 2.7, '2.3 events (zero-event 90% ceiling)', fontsize=7, color=C_G, ha='right')
a2.set_ylabel('expected events at the joint best fit'); a2.legend(fontsize=7.5, frameon=False, title='true δ', title_fontsize=7.5); a2.set_title('Expected counts (1 TeV)', fontsize=10)
fig.tight_layout(); fig.savefig(f'{FIG}/P082_fig3_future.png', dpi=160); plt.close(fig)
say(f'figures written; runtime {time.time() - T0:.0f}s')
LOG.close()
