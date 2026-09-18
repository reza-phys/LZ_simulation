"""
P050 -- How quickly would a 60-tonne xenon detector confirm or exclude the LZ high-energy hint?

Run from the simulation root:   .venv/bin/python output/code/P050_next_generation.py

Steps
  1. Signal spectra (WimPyDD, 1000 GeV, Baxter-2021 halo averaged over 12 days of the year):
     L10 (dipole-dipole reduction of P012) and inelastic O1 isoscalar at delta = 300/350/366/380 keV
     (plus +-2 keV neighbours for the Fisher information on delta).  Each model is normalised to
     1.0 event per 2.84 t yr inside LZ's ROI (efficiency 0.955, erf edges at 5.4 and 269.9 keV).
  2. Observed-energy spectra: efficiency x resolution (sigma_E = 11 sqrt(E/248) keV) for three ROIs:
     LZ-like (E50 = 269.9 keV), extended to E50 = 300 keV and E50 = 400 keV.
  3. Background model per 2.84 t yr in the NR band (+-2 sigma) binned in observed energy, split into a
     per-tonne-constant part (ER leakage, accidentals, atmospheric nu) and a surface part (MSSI, ~1/R).
  4. Asimov discovery significance vs exposure; exposures for 3/5 sigma and for N = 3 events;
     calendar timelines for LZ, the present generation and a 40/60 t detector.
  5. Zero-event exclusion of the LZ 90% lower edge (0.105 and 0.38 events per 2.84 t yr in LZ's ROI).
  6. Shape discrimination L10 vs inelastic delta = 366 keV (threshold-counting test, KL/LLR with toys).
  7. Fisher information on delta from the xenon spectrum -> events/exposure for +-10 keV.
  8. Figures.
Outputs: output/work/P050/*.csv, *.json, spectra cache, figures/*.png, run_log.txt
"""
from __future__ import annotations
import sys, os, json, math, time
import numpy as np
import pandas as pd
from scipy import special, optimize, stats

sys.path.insert(0, 'output/code')
from common import lzcommon as lz

T0 = time.time()
WORK = 'output/work/P050'; FIG = WORK + '/figures'
os.makedirs(FIG, exist_ok=True)
LOG = open(WORK + '/run_log.txt', 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.write(s + '\n'); LOG.flush()

RES = {}
EXPO_LZ = lz.LZ['exposure_tyr']           # 2.84 t yr
MV2 = lz.M_V_GEV ** 2; MN = lz.M_NUCLEON_GEV
M_CHI = 1000.0
DELTAS_MAIN = [300.0, 350.0, 366.0, 380.0]
DELTAS_ALL = sorted(set(DELTAS_MAIN + [348.0, 352.0, 364.0, 368.0, 378.0, 382.0]))
RNG = np.random.default_rng(50)

# ----------------------------------------------------------------------------------------------
# 1. Efficiency, resolution, geometry
# ----------------------------------------------------------------------------------------------
PLATEAU = 0.955
def eff(E, E50_hi, sig_hi, E50_lo=5.4, sig_lo=2.5):
    E = np.asarray(E, float)
    lo = 0.5 * (1 + special.erf((E - E50_lo) / (math.sqrt(2) * sig_lo)))
    hi = 0.5 * (1 - special.erf((E - E50_hi) / (math.sqrt(2) * sig_hi)))
    return PLATEAU * lo * hi

# windows: LZ (Fig. S2: 50% at 269.9 keV, erf sigma 11.5 keV per P021); extended edges with P038's
# erf widths (10.9 keV at 600 phd -> 18 keV at 1000 phd)
WINDOWS = {'LZ': dict(E50=269.9, sig=11.5, label='LZ-like ROI (E50 = 270 keV)'),
           'ext300': dict(E50=300.0, sig=13.0, label='extended ROI, E50 = 300 keV'),
           'ext400': dict(E50=400.0, sig=17.0, label='extended ROI, E50 = 400 keV'),
           'ext600': dict(E50=600.0, sig=22.0, label='extended ROI, E50 = 600 keV (captures E+)')}
def eff_win(E, w): return eff(E, WINDOWS[w]['E50'], WINDOWS[w]['sig'])

def sigma_E(E): return 11.0 * np.sqrt(np.asarray(E, float) / 248.0)   # P009/P021 resolution model

# P038 edge table: S1c edge (phd) -> E50 (keV); used to map an energy edge to S1c for the MSSI model
P038_EDGE = np.array([[600, 271.6], [700, 310.5], [800, 348.7], [1000, 423.2], [1200, 495.1]])
def s1c_of_E(E):
    return np.interp(E, P038_EDGE[:, 1], P038_EDGE[:, 0], left=None, right=None) if 271.6 <= E <= 495.1 \
        else float(np.polyval(np.polyfit(P038_EDGE[:, 1], P038_EDGE[:, 0], 1), E))

# geometry: LZ TPC radius 72.8 cm, drift 145.6 cm (recalled, likely; P004/P022); FV treated as a cylinder
RHO_LXE = 2.86   # t/m^3 (recalled, likely)
R_LZ = 0.728
def R_fv(mass_t):        # cylinder with H = 2R
    return (mass_t / RHO_LXE / (2 * math.pi)) ** (1 / 3)
DETECTORS = {'LZ-like': dict(M=4.71, R=R_LZ), '40 t': dict(M=40.0, R=R_fv(40.0)), '60 t': dict(M=60.0, R=R_fv(60.0))}
for k, d in DETECTORS.items():
    d['surf_scale'] = R_LZ / d['R']     # per-tonne MSSI / neutron scaling (wall 2/R, cathode 1/H, H = 2R)
say('FV radii (m) and surface scaling:', {k: (round(d['R'], 3), round(d['surf_scale'], 3)) for k, d in DETECTORS.items()})
RES['geometry'] = {k: dict(M_t=d['M'], R_m=d['R'], per_tonne_surface_scale=d['surf_scale']) for k, d in DETECTORS.items()}

# ----------------------------------------------------------------------------------------------
# 2. Signal spectra (cached)
# ----------------------------------------------------------------------------------------------
E_IN = np.arange(1.5, 800.0, 3.0)        # inelastic window ends below 800 keV at 1 TeV
E_L10 = np.arange(1.5, 1500.0, 3.0)      # elastic L10 extends to ~1.4 MeV
CACHE = WORK + '/P050_spectra_cache.npz'
def compute_spectra():
    WD = lz.wd()
    VG = np.linspace(0.0, 844.0, 1200)
    days12 = [15 + 30 * i for i in range(12)]
    halos = [lz.wd_halo(day_of_year=d, vmin=VG) for d in days12]
    vmin = halos[0][0]; deta = np.mean([h[1] for h in halos], axis=0)
    c0, c1 = lz.wd_c_from_anand(1.0 / MV2, 0.0)
    hO1 = lz.wd_hamiltonian('P050_O1s', {1: (c0, c1)})
    out = {'E_in': E_IN, 'E_l10': E_L10, 'deltas': np.array(DELTAS_ALL)}
    inel = []
    for d in DELTAS_ALL:
        t = time.time()
        r = lz.wd_rate(hO1, M_CHI, E_IN, halo=(vmin, deta), delta_kev=d)
        inel.append(np.clip(r, 0, None)); say(f'  inelastic delta={d:.0f} keV: {time.time()-t:.0f} s, full-window {EXPO_LZ*np.trapezoid(inel[-1], E_IN):.4g} ev/2.84 t yr at unit coupling')
    out['inel'] = np.array(inel)
    c4 = lz.wd_c_from_anand(4.0 / MV2)[0]; c6 = lz.wd_c_from_anand(-4.0 / MV2)[0]
    hL10 = WD.eft_hamiltonian('P050_L10', {(4, 'q2'): lambda q, A=c4: [A * q ** 2 / MN ** 2, 0.0], 6: lambda A6=c6: [A6, 0.0]})
    t = time.time(); out['l10'] = np.clip(lz.wd_rate(hL10, M_CHI, E_L10, halo=(vmin, deta)), 0, None)
    say(f'  L10 (d10 = 1, WimPyDD convention): {time.time()-t:.0f} s')
    np.savez(CACHE, **out); return out
if os.path.exists(CACHE):
    SP = dict(np.load(CACHE)); say('loaded spectra cache', CACHE)
else:
    say('computing WimPyDD spectra ...'); SP = compute_spectra()
INEL = {float(d): SP['inel'][i] for i, d in enumerate(SP['deltas'])}
L10 = SP['l10']

# ----------------------------------------------------------------------------------------------
# 3. Observed-energy machinery
# ----------------------------------------------------------------------------------------------
E_OBS = np.arange(0.5, 1500.0, 1.0)
def smear_matrix(E_true):
    dE = E_true[1] - E_true[0]
    s = sigma_E(E_true)
    K = stats.norm.pdf(E_OBS[:, None], loc=E_true[None, :], scale=s[None, :]) * dE
    return K
K_IN = smear_matrix(E_IN); K_L10 = smear_matrix(E_L10)

MODELS = {'L10': (E_L10, L10, K_L10)}
for d in DELTAS_ALL:
    MODELS[f'inel{d:.0f}'] = (E_IN, INEL[d], K_IN)
MAIN_MODELS = ['L10'] + [f'inel{d:.0f}' for d in DELTAS_MAIN]
LABELS = {'L10': 'L10 (1 TeV)', 'inel300': 'inelastic 300 keV', 'inel350': 'inelastic 350 keV',
          'inel366': 'inelastic 366 keV', 'inel380': 'inelastic 380 keV'}

def obs_spectrum(model, w):
    """dN/dE_obs per t yr at unit coupling for window w."""
    E, r, K = MODELS[model]
    return K @ (r * eff_win(E, w))

NORM = {}      # per model: unit-coupling counts per 2.84 t yr in LZ ROI, full window, each window
for m in MODELS:
    E, r, K = MODELS[m]
    N_LZ = EXPO_LZ * np.trapezoid(r * eff_win(E, 'LZ'), E)
    N_full = EXPO_LZ * np.trapezoid(r, E)
    d = dict(N_LZ_unit=N_LZ, N_full_unit=N_full, A_LZ=N_LZ / N_full, kappa=1.0 / N_LZ,
             rate_full_per_tyr=N_full / N_LZ / EXPO_LZ)
    for w in WINDOWS:
        Nw = EXPO_LZ * np.trapezoid(r * eff_win(E, w), E)
        d[f'N_{w}_unit'] = Nw; d[f'A_{w}'] = Nw / N_full; d[f'rate_{w}_per_tyr'] = Nw / N_LZ / EXPO_LZ
    ob = obs_spectrum(m, 'LZ') / N_LZ * EXPO_LZ           # per 2.84 t yr at best fit
    d['E_obs_median_LZ'] = float(np.interp(0.5, np.cumsum(ob) / ob.sum(), E_OBS))
    d['frac_true_gt_270'] = float(np.trapezoid(r[E > 270], E[E > 270]) / np.trapezoid(r, E))
    NORM[m] = d
say('\nNormalisation (1 TeV; 1 event per 2.84 t yr in LZ ROI):')
tab = pd.DataFrame(NORM).T
say(tab[['N_LZ_unit', 'A_LZ', 'A_ext300', 'A_ext400', 'rate_full_per_tyr', 'rate_ext300_per_tyr', 'rate_ext400_per_tyr', 'frac_true_gt_270']].loc[MAIN_MODELS].round(4).to_string())
say('P038 A_600 for 300/350/370/380: 0.81/0.44/0.12/0.029; ours at 300/350/366/380:',
    [round(NORM[m]['A_LZ'], 3) for m in MAIN_MODELS[1:]])
tab.to_csv(WORK + '/P050_normalisation.csv')
RES['normalisation'] = {m: {k: float(v) for k, v in NORM[m].items()} for m in MODELS}

# ----------------------------------------------------------------------------------------------
# 4. Background model (NR band +-2 sigma) per 2.84 t yr in the 4.7 t LZ FV, binned in E_obs
# ----------------------------------------------------------------------------------------------
BIN_EDGES = np.array([100, 125, 200, 270, 300, 350, 400, 450, 500, 600], float)
MSSI_TAB = pd.read_csv('output/work/P038/P038_mssi_vs_edge.csv')
F_NRBAND = 0.0027 / 0.018        # P038: NR-band (+-2 sigma) share of the MSSI added by extending to 1000 phd
def mssi_mid(s1c): return float(np.interp(s1c, MSSI_TAB.S1c_edge, MSSI_TAB.MSSI_mid))
ACC_PER_100KEV = 1.0e-4          # accidentals entering the NR band per 100 keV of extension (estimate; P022 spectrum falls x9 across the ROI)
NU_ABOVE_270 = 1.0e-5            # atmospheric-nu CEvNS above 270 keV (P019: 6.5e-5 above 200 keV)
def background_bins():
    rows = []
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        if hi <= 125:   const, surf, src = 0.10, 0.0, 'ER leakage into NR band (estimate, x3 bracket)'
        elif hi <= 200: const, surf, src = 0.02, 0.0, 'P016 b_M (125-200 keV NR-band bin)'
        elif hi <= 270: const, surf, src = 1.5e-4 + 3.4e-5 + 2.2e-4, 1.7e-4, 'P016 b_H = 5.7e-4: accidentals 1.5e-4 (P022) + nu 3.4e-5 (P019) + ER/other 2.2e-4 | MSSI 1.7e-4 (P004)'
        else:
            surf = F_NRBAND * (mssi_mid(s1c_of_E(hi)) - mssi_mid(s1c_of_E(lo)))
            const = ACC_PER_100KEV * (hi - lo) / 100.0 + NU_ABOVE_270 * (hi - lo) / 330.0
            src = 'P038 MSSI interpolation x NR-band share 0.15 | accidentals 1e-4 per 100 keV + nu'
        rows.append(dict(E_lo=lo, E_hi=hi, b_const_per284=const, b_surf_per284=surf, b_total_LZ_per284=const + surf, source=src))
    return pd.DataFrame(rows)
BKG = background_bins()
say('\nBackground per 2.84 t yr in the NR band, LZ-like detector, by observed-energy bin:')
say(BKG.drop(columns='source').to_string(index=False))
BKG.to_csv(WORK + '/P050_background_bins.csv', index=False)

def active_bins(w):
    """bins inside the analysis window: everything up to 270 keV plus extension bins whose acceptance exceeds 1 %."""
    ctr = 0.5 * (BKG.E_lo + BKG.E_hi)
    return ((BKG.E_hi <= 270) | (eff_win(ctr, w) > 0.01 * PLATEAU)).values

def bkg_rate_bins(w, det):
    """background per t yr per bin for window w in detector det (extension bins weighted by the window acceptance)."""
    sc = DETECTORS[det]['surf_scale']
    ctr = 0.5 * (BKG.E_lo + BKG.E_hi)
    accept = np.where(BKG.E_hi <= 270, 1.0, eff_win(ctr, w) / PLATEAU)
    return ((BKG.b_const_per284 + sc * BKG.b_surf_per284) * accept / EXPO_LZ).values * active_bins(w)

def sig_rate_bins(m, w):
    """signal per t yr per observed-energy bin at the best fit (1 event per 2.84 t yr in LZ's ROI)."""
    ob = obs_spectrum(m, w) / NORM[m]['N_LZ_unit']
    idx = np.digitize(E_OBS, BIN_EDGES) - 1
    out = np.zeros(len(BIN_EDGES) - 1)
    for i in range(len(out)):
        out[i] = ob[idx == i].sum()
    return out * active_bins(w)     # per t yr

btab = []
for w in WINDOWS:
    for det in DETECTORS:
        b = bkg_rate_bins(w, det)
        btab.append(dict(window=w, detector=det, b_100_200_per_tyr=b[:2].sum(), b_200_270_per_tyr=b[2], b_ext_per_tyr=b[3:].sum(),
                         b_ge200_per_tyr=b[2:].sum(), b_ge200_per_284=b[2:].sum() * EXPO_LZ, b_ge100_per_tyr=b.sum()))
BT = pd.DataFrame(btab); say('\nBackground rates per t yr (NR band) by window and detector:'); say(BT.round(6).to_string(index=False))
BT.to_csv(WORK + '/P050_background_rates.csv', index=False)
RES['background_rates'] = BT.to_dict('records')

# ----------------------------------------------------------------------------------------------
# 5. Discovery: Asimov Z vs exposure
# ----------------------------------------------------------------------------------------------
def Z_asimov(k, s, b):
    S = k * s; B = k * b
    m = (S > 0) & (B > 0)
    q = 2 * ((S[m] + B[m]) * np.log1p(S[m] / B[m]) - S[m])
    return math.sqrt(max(q.sum(), 0.0))

def exposure_for_Z(Ztarget, s, b):
    f = lambda k: Z_asimov(k, s, b) - Ztarget
    return optimize.brentq(f, 1e-3, 1e7)

RATE_BAND = {'best': 1.0, 'lo68': 0.30, 'hi68': 2.36}     # LZ Table I interval on the L10 count
LIVE = {'LZ alone': 4.71 * 220 / 371, 'present generation': (4.71 + 4.0 + 2.7) * 0.6, '40 t': 40 * 0.8, '60 t': 60 * 0.8}
SCEN = [('LZ', 'LZ-like'), ('ext300', 'LZ-like'), ('ext400', 'LZ-like'), ('ext400', '40 t'), ('ext400', '60 t'), ('ext600', '60 t')]
drows = []
for w, det in SCEN:
    b = bkg_rate_bins(w, det)
    for m in MAIN_MODELS:
        s = sig_rate_bins(m, w)
        r_win = s.sum()
        row = dict(window=w, detector=det, model=m, rate_per_tyr=r_win, b_per_tyr=b[s > 1e-9 * s.max()].sum(),
                   E_3sigma=exposure_for_Z(3, s, b), E_5sigma=exposure_for_Z(5, s, b),
                   E_5sigma_lo68=exposure_for_Z(5, s * 0.30, b), E_5sigma_hi68=exposure_for_Z(5, s * 2.36, b),
                   E_N3=3.0 / r_win, E_P90_N3=stats.gamma.ppf(0.9, 3) / r_win if False else optimize.brentq(lambda k: stats.poisson.sf(2, k * r_win) - 0.9, 1e-3, 1e5),
                   s_at_5sigma=exposure_for_Z(5, s, b) * r_win)
        for prog, rate in LIVE.items():
            if (det == 'LZ-like' and prog in ('LZ alone', 'present generation')) or (det == prog):
                row[f'yr_5sigma_{prog}'] = row['E_5sigma'] / rate
                row[f'yr_3sigma_{prog}'] = row['E_3sigma'] / rate
        drows.append(row)
DISC = pd.DataFrame(drows)
say('\nDiscovery exposures (t yr) at the LZ best fit (1 event / 2.84 t yr in LZ ROI):')
say(DISC[['window', 'detector', 'model', 'rate_per_tyr', 'b_per_tyr', 'E_3sigma', 'E_5sigma', 'E_5sigma_lo68', 'E_5sigma_hi68', 'E_N3', 'E_P90_N3', 's_at_5sigma']].round(3).to_string(index=False))
DISC.to_csv(WORK + '/P050_discovery.csv', index=False)
RES['discovery'] = DISC.to_dict('records')

# June-clustering bonus (P034): against background, timing shortens the 5 sigma exposure by at most x1.36
RES['timing_gain_P034'] = 'exposures above may be divided by <= 1.36 for delta >= 350 keV if the time information is used (P034); not applied'

# ----------------------------------------------------------------------------------------------
# 6. Exclusion with zero events
# ----------------------------------------------------------------------------------------------
xrows = []
for w in WINDOWS:
    for m in MAIN_MODELS:
        A_ratio = NORM[m]['N_LZ_unit'] / NORM[m][f'N_{w}_unit']       # window rate = LZ-ROI rate / A_ratio
        for s_low in (0.105, 0.30, 0.38):
            r_low = s_low / EXPO_LZ / A_ratio      # per t yr in this window
            row = dict(window=w, model=m, s_low_per284_LZROI=s_low, rate_in_window_per_tyr=r_low,
                       E_90_CLs=-math.log(0.10) / r_low, E_95_CLs=-math.log(0.05) / r_low)
            # with LZ-like background, P(0 | s+b) <= 0.1 (slightly less exposure)
            b = bkg_rate_bins(w, 'LZ-like'); s = sig_rate_bins(m, w); bw = b[s > 1e-9 * s.max()].sum()
            row['E_90_P0'] = -math.log(0.10) / (r_low + bw)
            for prog, rate in LIVE.items():
                if (w == 'LZ' and prog in ('LZ alone', 'present generation')) or (w == 'ext400' and prog in ('40 t', '60 t')):
                    row[f'yr_90_{prog}'] = row['E_90_CLs'] / rate
            xrows.append(row)
EXCL = pd.DataFrame(xrows)
say('\nZero-event exclusion exposures (t yr):')
say(EXCL[EXCL.s_low_per284_LZROI.isin([0.105, 0.38])][['window', 'model', 's_low_per284_LZROI', 'rate_in_window_per_tyr', 'E_90_CLs', 'E_95_CLs', 'E_90_P0']].round(3).to_string(index=False))
EXCL.to_csv(WORK + '/P050_exclusion.csv', index=False)
RES['exclusion'] = EXCL.to_dict('records')

# ----------------------------------------------------------------------------------------------
# 7. Shape discrimination: L10 vs inelastic 366 keV
# ----------------------------------------------------------------------------------------------
E_LO_SHAPE = 100.0
def pdf_obs(m, w, floor=0.0):
    ob = obs_spectrum(m, w).copy(); ob[E_OBS < E_LO_SHAPE] = 0.0
    p = ob / ob.sum()
    if floor > 0:
        u = np.zeros_like(p); sel = (E_OBS >= E_LO_SHAPE) & (eff_win(E_OBS, w) > 0.01 * PLATEAU); u[sel] = 1.0 / sel.sum()
        p = (1 - floor) * p + floor * u
    return p

def llr_stats(p, q):
    m = (p > 0) & (q > 0)
    l = np.log(p[m] / q[m])
    KL_pq = float((p[m] * l).sum()); KL_qp = float(-(q[m] * l).sum())
    var_p = float((p[m] * l ** 2).sum() - KL_pq ** 2); var_q = float((q[m] * l ** 2).sum() - KL_qp ** 2)
    return KL_pq, KL_qp, math.sqrt(var_p), math.sqrt(var_q)

def toy_N_for_3sigma(p, q, Nmax=60, ntoy=20000):
    """smallest N such that the median LLR under p is a >= 3 sigma exclusion of q (and vice versa)."""
    m = (p > 0) & (q > 0); l = np.log(np.where(m, p / np.where(q > 0, q, 1), 0.0))
    cp = np.cumsum(p); cq = np.cumsum(q)
    def draw(c, N): return np.searchsorted(c, RNG.random((ntoy, N)))
    out = {}
    for direction, (pt, ct, ca, sign) in {'reject_inel366_if_L10_true': (p, cp, cq, +1), 'reject_L10_if_inel366_true': (q, cq, cp, -1)}.items():
        found = None; zs = []
        for N in range(1, Nmax + 1):
            t_true = sign * l[draw(ct, N)].sum(axis=1); t_alt = sign * l[draw(ca, N)].sum(axis=1)
            med = np.median(t_true); pval = max((t_alt >= med).mean(), 0.5 / ntoy)
            z = stats.norm.isf(pval); zs.append((N, z))
            if z >= 3 and found is None: found = N
            if found is not None and N >= found + 2: break
        out[direction] = dict(N_3sigma=found, Z_vs_N=zs)
    return out

srows = []; RES['shape'] = {}
for w in WINDOWS:
    p = pdf_obs('L10', w); q = pdf_obs('inel366', w)
    cq = np.cumsum(q); E_thr = float(np.interp(1e-3, cq, E_OBS))      # inelastic-366 spectrum starts here (0.1 % quantile)
    E_top = float(np.interp(1 - 1e-3, cq, E_OBS))
    f_out = float(p[(E_OBS < E_thr) | (E_OBS > E_top)].sum())          # L10 events outside the inelastic support
    N_count = math.log(0.00135) / math.log(1 - f_out)
    row = dict(window=w, E_thr_inel366=E_thr, E_top_inel366=E_top, f_L10_outside=f_out, N_3sigma_counting=N_count)
    for floor in (1e-3, 1e-2):
        pf, qf = pdf_obs('L10', w, floor), pdf_obs('inel366', w, floor)
        KL_pq, KL_qp, sp, sq = llr_stats(pf, qf)
        row[f'KL_L10_vs_inel_floor{floor:g}'] = KL_pq; row[f'KL_inel_vs_L10_floor{floor:g}'] = KL_qp
        row[f'N_gauss_reject_inel_floor{floor:g}'] = 9 * sq ** 2 / (KL_pq + KL_qp) ** 2
        row[f'N_gauss_reject_L10_floor{floor:g}'] = 9 * sp ** 2 / (KL_pq + KL_qp) ** 2
        toys = toy_N_for_3sigma(pf, qf)
        row[f'N_toy_reject_inel_floor{floor:g}'] = toys['reject_inel366_if_L10_true']['N_3sigma']
        row[f'N_toy_reject_L10_floor{floor:g}'] = toys['reject_L10_if_inel366_true']['N_3sigma']
        RES['shape'][f'{w}_floor{floor:g}_toys'] = {k: v['Z_vs_N'] for k, v in toys.items()}
    # exposures at the best fit (this window) for the toy-based N with the 1 % floor, and for the counting N
    row['E_reject_inel_if_L10_true_tyr'] = row['N_toy_reject_inel_floor0.01'] / NORM['L10'][f'rate_{w}_per_tyr']
    row['E_reject_L10_if_inel_true_tyr'] = row['N_toy_reject_L10_floor0.01'] / NORM['inel366'][f'rate_{w}_per_tyr']
    row['E_counting_if_inel_true_tyr'] = N_count / NORM['inel366'][f'rate_{w}_per_tyr']
    srows.append(row)
SHAPE = pd.DataFrame(srows); say('\nShape discrimination L10 vs inelastic 366 keV:'); say(SHAPE.round(4).T.to_string())
SHAPE.to_csv(WORK + '/P050_shape_discrimination.csv', index=False)
RES['shape']['table'] = SHAPE.to_dict('records')

# ----------------------------------------------------------------------------------------------
# 8. Fisher information on delta from the xenon spectrum
# ----------------------------------------------------------------------------------------------
frows = []
for d0, dm, dp in ((350.0, 348.0, 352.0), (366.0, 364.0, 368.0), (380.0, 378.0, 382.0)):
    for w in WINDOWS:
        p0 = pdf_obs(f'inel{d0:.0f}', w); pm = pdf_obs(f'inel{dm:.0f}', w); pp = pdf_obs(f'inel{dp:.0f}', w)
        dp_dd = (pp - pm) / (dp - dm)
        m = p0 > 1e-9 * p0.max()
        I = float((dp_dd[m] ** 2 / p0[m]).sum())          # per event, keV^-2
        # endpoint-only information proxy: information carried by the upper 10 % of the spectrum
        c0 = np.cumsum(p0); top = m & (c0 > 0.9)
        I_top = float((dp_dd[top] ** 2 / p0[top]).sum())
        N10 = 1.0 / (100.0 * I)
        row = dict(delta=d0, window=w, fisher_per_event_keV2=I, sigma_delta_1event=1 / math.sqrt(I), N_for_10keV=N10,
                   frac_info_top10pct=I_top / I, E_median_obs=float(np.interp(0.5, c0, E_OBS)),
                   E_for_10keV_60t_tyr=N10 / NORM[f'inel{d0:.0f}'][f'rate_{w}_per_tyr'],
                   yr_60t=N10 / NORM[f'inel{d0:.0f}'][f'rate_{w}_per_tyr'] / LIVE['60 t'])
        frows.append(row)
FISH = pd.DataFrame(frows); say('\nFisher information on delta (shape only, coupling free):'); say(FISH.round(4).to_string(index=False))
FISH.to_csv(WORK + '/P050_delta_fisher.csv', index=False)
RES['delta_fisher'] = FISH.to_dict('records')

# tungsten ratio method from P015's W/Xe rate ratios (per tonne, equal coupling): 82/222/699 at 350/366/380 keV
WXE = {350: 82.0, 366: 222.0, 380: 699.0}
slope = (math.log(WXE[380]) - math.log(WXE[350])) / 30.0        # d ln(W/Xe) / d delta per keV
frac_needed = slope * 10.0                                        # fractional ratio precision for +-10 keV
N_each = 2.0 / frac_needed ** 2                                   # equal counts in W and Xe: sigma_ratio/ratio = sqrt(2/N)
RES['tungsten_ratio_method'] = dict(dln_ratio_ddelta_per_keV=slope, frac_precision_for_10keV=frac_needed, N_each_target=N_each,
                                     note='P015 rate ratios; P046 (planned) not yet in corpus')
say(f'\nTungsten/xenon ratio method (P015 ratios): d ln(W/Xe)/d delta = {slope:.4f}/keV -> +-10 keV needs ratio to {frac_needed*100:.0f} %, i.e. ~{N_each:.1f} events in each target')
# halo systematic on delta from the endpoint: P018 slope 0.82 keV per km/s of v_max
RES['halo_systematic'] = dict(keV_per_kms=0.82, vesc_pm20_kms_keV=0.82 * 20, note='P018: delta_max moves 0.82 keV per km/s; v_esc +-20 km/s -> +-16 keV on an endpoint-based delta')

# ----------------------------------------------------------------------------------------------
# 9. Timelines and figures
# ----------------------------------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
COL = {'L10': '#000000', 'inel300': '#E69F00', 'inel350': '#56B4E9', 'inel366': '#009E73', 'inel380': '#CC79A7'}   # Okabe-Ito, fixed order

years = np.arange(2024.25, 2040.01, 0.05)
years60 = np.arange(2032.0, 2033.5001, 0.0025)
def expo_LZ(t):
    return 2.84 + LIVE['LZ alone'] * np.clip(np.minimum(t, 2029.0) - 2024.25, 0, None)
def expo_present(t):
    other = np.where(t >= 2026.67, 4.64 + (4.0 + 2.7) * 0.6 * np.clip(np.minimum(t, 2030.0) - 2026.67, 0, None), 0.0)
    return expo_LZ(t) + other
def expo_60(t, M=60.0): return M * 0.8 * np.clip(t - 2032.0, 0, None)

PROG = [('LZ alone', expo_LZ, 'LZ', 'LZ-like'), ('present generation (LZ + XENONnT + PandaX-4T)', expo_present, 'LZ', 'LZ-like'),
        ('60 t detector from 2032, ROI to 400 keV', expo_60, 'ext400', '60 t')]
trows = []
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
for ax, (name, fexp, w, det) in zip(axes, PROG):
    b = bkg_rate_bins(w, det)
    tt = years60 if det == '60 t' else years
    for m in MAIN_MODELS:
        s = sig_rate_bins(m, w)
        Z = np.array([Z_asimov(k, s, b) if k > 0 else 0.0 for k in fexp(tt)])
        ax.plot(tt, Z, color=COL[m], lw=2, label=LABELS[m] + (' (best fit)' if m == 'L10' else ''))
        Zlo = np.array([Z_asimov(k, 0.30 * s, b) if k > 0 else 0.0 for k in fexp(tt)])
        ax.plot(tt, Zlo, color=COL[m], lw=1.1, ls='--', label='lower 68 % rate (x 0.30)' if m == 'L10' else None)
        for zt in (3, 5):
            for tag, ZZ in (('best', Z), ('lo68', Zlo)):
                i = int(np.argmax(ZZ >= zt)) if (ZZ >= zt).any() else None
                trows.append(dict(programme=name, model=m, rate=tag, Z=zt, year=float(tt[i]) if i is not None else np.nan,
                                  exposure_tyr=float(fexp(tt[i])) if i is not None else np.nan))
    for zt, lab in ((3, '3 sigma'), (5, '5 sigma')):
        ax.axhline(zt, color='#888888', lw=0.8, ls=':'); ax.text(tt[0] + 0.02 * (tt[-1] - tt[0]), zt + 0.15, lab, fontsize=8, color='#555555')
    ax.set_title(name, fontsize=10); ax.set_xlabel('calendar year'); ax.grid(alpha=0.25, lw=0.5)
    ax.set_xlim(tt[0] - 0.02 * (tt[-1] - tt[0]), tt[-1]); ax.set_ylim(0, 12)
axes[0].set_ylabel('Asimov significance Z')
axes[0].legend(fontsize=7.5, frameon=False, loc='upper left')
fig.suptitle('P050: projected significance of the LZ high-energy hint (solid: LZ best-fit rate, 1 event per 2.84 t yr in LZ ROI; dashed: x 0.30)', fontsize=10)
fig.tight_layout(); fig.savefig(FIG + '/P050_fig2_timeline.png', dpi=160); plt.close(fig)
TL = pd.DataFrame(trows); say('\nTimeline milestones:'); say(TL.round(2).to_string(index=False))
TL.to_csv(WORK + '/P050_timeline_milestones.csv', index=False)
RES['timeline'] = TL.to_dict('records')

# Fig 1: observed-energy spectra at the best fit and the three windows
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for m in MAIN_MODELS:
    E, r, K = MODELS[m]
    ob = (K @ r) / NORM[m]['N_LZ_unit']          # per t yr per keV, no efficiency, best fit
    ax.plot(E_OBS, ob, color=COL[m], lw=2, label=LABELS[m])
for w, ls in (('LZ', '-'), ('ext300', '--'), ('ext400', ':'), ('ext600', '-.')):
    ax.plot(E_OBS, eff_win(E_OBS, w) * 2.0e-3, color='#666666', lw=1, ls=ls, label=f'{WINDOWS[w]["label"]} (efficiency x 2e-3)')
ax.set_xlim(50, 620); ax.set_yscale('log'); ax.set_ylim(1e-6, 5e-2)
ax.set_xlabel('observed recoil energy (keV, resolution 11 sqrt(E/248) keV)'); ax.set_ylabel('events / (t yr keV) at the LZ best fit')
ax.legend(fontsize=7.5, frameon=False, ncol=2); ax.grid(alpha=0.25, lw=0.5)
ax.set_title('P050: best-fit spectra (1 TeV) beyond LZ\'s 270 keV edge', fontsize=10)
fig.tight_layout(); fig.savefig(FIG + '/P050_fig1_spectra.png', dpi=160); plt.close(fig)

# Fig 3: Z vs exposure for the five scenarios (best-fit rate), one panel per model
fig, axes = plt.subplots(1, 5, figsize=(15, 3.6), sharey=True)
kk = np.logspace(-1, 3, 200)
SC_COL = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#CC79A7', '#D55E00']
for ax, m in zip(axes, MAIN_MODELS):
    for (w, det), c in zip(SCEN, SC_COL):
        s = sig_rate_bins(m, w); b = bkg_rate_bins(w, det)
        ax.plot(kk, [Z_asimov(k, s, b) for k in kk], color=c, lw=1.8, label=f'{w}, {det} bkg')
    ax.axhline(5, color='#888888', lw=0.8, ls=':'); ax.set_xscale('log'); ax.set_title(LABELS[m], fontsize=10)
    ax.set_xlabel('exposure (t yr)'); ax.grid(alpha=0.25, lw=0.5); ax.set_ylim(0, 10); ax.set_xlim(0.3, 300)
axes[0].set_ylabel('Asimov Z'); axes[0].legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(FIG + '/P050_fig3_Z_vs_exposure.png', dpi=160); plt.close(fig)

RES['runtime_s'] = time.time() - T0
RES['settings'] = dict(m_chi_GeV=M_CHI, halo='Baxter-2021 SHM, 12-day annual average (lz.wd_halo)', resolution='11 sqrt(E/248) keV',
                       windows=WINDOWS, bins_keV=BIN_EDGES.tolist(), live_rates_tyr_per_yr=LIVE, rate_band=RATE_BAND,
                       E_lo_shape_keV=E_LO_SHAPE, present_generation='LZ 4.71 t (0.593 live) + XENONnT 4.0 t + PandaX-4T 2.7 t (0.6 live; recalled masses), existing 3.1 + 1.54 t yr reanalysed from Sep 2026, ends 2030',
                       next_generation='40-60 t fiducial, 80 % live, data from 2032 (recalled/uncertain)')
def _default(o):
    if isinstance(o, (np.floating, np.integer)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, float) and math.isnan(o): return None
    return str(o)
json.dump(RES, open(WORK + '/P050_results.json', 'w'), indent=1, default=_default)
say(f'\ndone in {RES["runtime_s"]:.0f} s')
LOG.close()
