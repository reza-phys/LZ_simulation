"""
P016 -- Why does LZ's likelihood still give 2.7 sigma to spectra with tens of low-energy events?
A joint low/high-energy reassessment (few-bin approximation to the LZ unbinned extended likelihood).

Run from the simulation root:   .venv/bin/python output/code/P016_joint_constraint.py

Steps
 1. Digitise Fig. 5 top panel (S1c < 250 phd): background-model total and data per 0.5 sigma_NR bin.
 2. Few-bin extended likelihood: top-panel bins (5.4-125 keV NR band, ER-leakage dominated),
    a middle-panel bin (125-200 keV, ~0.02 background, 0 events), and the high-energy bin H
    (200-270 keV NR band, n_H = 1, b_H anchored so that a companion-free spectrum gives 3.4 sigma).
    Signal per unit s (s = expected 200-270 keV events): N_lo (5.4-55, from P003), N_L2 (55-125),
    N_M (125-200) companions, Gaussian in sigma_NR units.  ER-leakage scale theta profiled (Gaussian constraint).
 3. Z_joint(N_lo) curve; calibration against LZ Tables S6/S7 (L1, L2, L3, L4, L10, L15 <-> O1, O10, O11, O6, L10, O4).
 4. Explicit 2024 low-energy null as a Gaussian constraint with 90% UL N_max in {3,5,10,20} events.
 5. Bayes factors per operator (P001 single-event formula) with the low-energy likelihood folded in.
 6. Sensitivity to the low-energy background level (x0.5 ... x3), exposure fraction, signal centroid, theta prior.
 7. Toy-MC check of the Wilks mapping q0 -> Z for the few-bin model.
"""
import os, sys, json, time
sys.path.insert(0, 'output/code')
from common import lzcommon as lz
import numpy as np
import pandas as pd
from scipy import stats, optimize, special, ndimage, integrate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

t0 = time.time()
OUT = 'output/work/P016'
FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)
R = {}   # results dictionary -> P016_results.json

# ----------------------------------------------------------------------------------------------
# 1. Digitise Fig. 5 top panel (S1c < 250 phd)
# ----------------------------------------------------------------------------------------------
PNG = 'inputs/figures_png/Fig5_NR_distance_3panels_no_sig.png'
# frame rows detected (long dark horizontal runs): 496/497 (top) and 1299/1300 (bottom); major y ticks at
# rows 594.5 (10^2), 735.5, 876.5, 1017.5, 1158.5, 1299.5 (10^-3): 141.0 px per decade.  x frame as P004: 219..2409.
Y_TOP, Y_BOT, X_L, X_R = 497, 1299, 219, 2409
Y_TICK_2, PX_DEC = 594.5, 141.0
EDGES = np.arange(-8, 8.01, 0.5)
CENTRES = 0.5 * (EDGES[1:] + EDGES[:-1])


def y_to_log10(y):
    return 2.0 - (y - Y_TICK_2) / PX_DEC


def x_to_sig(x):
    return -8.0 + 16.0 * (x - X_L) / (X_R - X_L)


def sig_to_x(s):
    return X_L + (s + 8.0) / 16.0 * (X_R - X_L)


def digitise_top_panel():
    im = np.asarray(Image.open(PNG).convert('RGB')).astype(int)
    colours = {'Total': (0, 0, 255), 'Accidentals': (255, 220, 61), 'NRs': (124, 174, 0),
               'ERs': (0, 194, 249), 'Internal': (255, 0, 255), 'MSSI': (2, 81, 128), 'L10': (165, 42, 42)}
    sub = im[Y_TOP + 3:Y_BOT - 2, X_L + 3:X_R - 2]
    hist = {}
    for name, col in colours.items():
        mask = (np.abs(sub - np.array(col)).sum(axis=2) < 60)
        vals = np.full(len(EDGES) - 1, np.nan)
        for i in range(len(EDGES) - 1):
            xa = int(sig_to_x(EDGES[i] + 0.1)) - (X_L + 3)
            xb = int(sig_to_x(EDGES[i] + 0.4)) - (X_L + 3)
            ys = []
            for x in range(xa, xb):
                yy = np.where(mask[:, x])[0]
                if len(yy):
                    ys.append(yy.min() + Y_TOP + 3)   # top-most pixel of the (step) line
            if ys:
                vals[i] = 10 ** y_to_log10(np.median(ys))
        hist[name] = vals
    # data points: black filled circles centred on bin centres
    dark = (sub.sum(axis=2) < 150)
    lab, nlab = ndimage.label(dark)
    objs = ndimage.find_objects(lab)
    data = np.zeros(len(EDGES) - 1)
    data_val = np.full(len(EDGES) - 1, np.nan)
    pts = []
    for k, sl in enumerate(objs):
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        if not (14 <= h <= 60 and 14 <= w <= 60 and abs(h - w) <= 10):
            continue
        blob = (lab[sl] == k + 1)
        if blob.mean() < 0.6:
            continue
        cy = sl[0].start + h / 2 + Y_TOP + 3
        cx = sl[1].start + w / 2 + X_L + 3
        sg = x_to_sig(cx)
        i = int(np.floor((sg + 8) / 0.5))
        if abs(sg - CENTRES[i]) > 0.08:
            continue
        v = 10 ** y_to_log10(cy)
        pts.append((round(sg, 2), v))
        data_val[i] = v
        data[i] = round(v) if v < 200 else v
    return hist, data, data_val, pts


hist_top, n_top_raw, n_top_val, pts = digitise_top_panel()
b_top_raw = np.nan_to_num(hist_top['Total'])
print('[1] Fig.5 top panel digitised.  bin_low  total_model  data')
for e, b, n, v in zip(EDGES[:-1], b_top_raw, n_top_raw, n_top_val):
    print(f'     {e:+5.1f}  {b:9.3g}  {n:6.0f}  ({"" if np.isnan(v) else f"{v:.2f}"})')
R['fig5_top_digitised'] = {'sigma_bin_low': EDGES[:-1].tolist(), 'total': b_top_raw.tolist(),
                           'data_counts': n_top_raw.tolist(),
                           'data_readoff': [None if np.isnan(v) else float(v) for v in n_top_val],
                           'components': {k: np.nan_to_num(v).tolist() for k, v in hist_top.items()}}
# consistency: total data in panel vs science sample (1710 total, minus middle/bottom-panel events)
R['fig5_top_sum_data'] = float(n_top_raw.sum())
R['fig5_top_sum_model'] = float(b_top_raw.sum())
print(f'     sum data {n_top_raw.sum():.0f}, sum model {b_top_raw.sum():.0f} (science sample: 1710 obs / 1713 fit)')

# ----------------------------------------------------------------------------------------------
# 2. Spectral companions per 200-270 keV event: N_lo (5.4-55), N_L2 (55-125), N_M (125-200)  [P003 spectra]
# ----------------------------------------------------------------------------------------------
SIG_LO, SIG_HI, EFF0 = 3.4, 8.0, 0.96      # P003 efficiency model (details.md sec. 3)


def efficiency(E):
    s_lo = 0.5 * (1 + special.erf((E - 5.4) / (np.sqrt(2) * SIG_LO)))
    s_hi = 0.5 * (1 - special.erf((E - 269.9) / (np.sqrt(2) * SIG_HI)))
    return EFF0 * s_lo * s_hi


spec = np.load('output/work/P003/P003_spectra.npz')
E_grid = spec['E']
E_fine = np.linspace(1.0, 300.0, 3000)


def window(E, dR, e1, e2):
    f = np.interp(E_fine, E, dR)
    m = (E_fine >= e1) & (E_fine <= e2)
    return np.trapezoid(f[m] * efficiency(E_fine[m]), E_fine[m])


p003 = pd.read_csv('output/work/P003/P003_operator_table.csv')
rows = []
for _, r in p003.iterrows():
    key = f"{r.operator}{r.isospin}_{int(r.m_chi_GeV)}"
    dR = spec[key]
    R_lo = window(E_grid, dR, 5.4, 55); R_L2 = window(E_grid, dR, 55, 125)
    R_M = window(E_grid, dR, 125, 200); R_hi = window(E_grid, dR, 200, 270)
    rows.append(dict(model=key, operator=r.operator, isospin=r.isospin, m_chi_GeV=int(r.m_chi_GeV),
                     N_lo_P003=r.N_lo_per_hi, N_lo=R_lo / R_hi, N_L2=R_L2 / R_hi, N_M=R_M / R_hi))
ops = pd.DataFrame(rows)
# L10-like transverse-spin combination: N_lo = 0.20 (P003); its 55-200 keV shape follows O6 (same q^4 spin response)
o6 = ops[ops.model == 'O6s_1000'].iloc[0]
ops = pd.concat([ops, pd.DataFrame([dict(model='L10_1000', operator='L10', isospin='s', m_chi_GeV=1000, N_lo_P003=0.20,
                                         N_lo=0.20, N_L2=0.20 * o6.N_L2 / o6.N_lo, N_M=0.20 * o6.N_M / o6.N_lo)])],
                ignore_index=True)
print('[2] companions per 200-270 keV event (1000 GeV):')
for _, r in ops[ops.m_chi_GeV == 1000].iterrows():
    print(f"     {r.model:10s} N_lo {r.N_lo:8.2f} (P003 {r.N_lo_P003:8.2f})  N_55-125 {r.N_L2:7.2f}  N_125-200 {r.N_M:6.2f}")
R['N_lo_recomputed_vs_P003_max_rel_dev'] = float(np.nanmax(np.abs(ops.N_lo / ops.N_lo_P003 - 1)))

# ----------------------------------------------------------------------------------------------
# 3. Few-bin extended likelihood
# ----------------------------------------------------------------------------------------------
WIN_HI = 2.0     # top-panel bins used: -8 .. +2 sigma_NR (signal beyond +2 sigma: 2.3%)
sel = EDGES[:-1] < WIN_HI - 1e-9
E_LO, E_HI = EDGES[:-1][sel], EDGES[1:][sel]
B_M, N_M_OBS = 0.02, 0           # middle panel (250-500 phd) within +-1.5 sigma: read off Fig.5 (3e-4 .. 1e-2 per bin)


def anchor_bH(Z=3.4):
    """b_H such that a companion-free spectrum with n_H = 1 gives Z (asymptotic q0 = 2[ln(1/b) - 1 + b])."""
    return optimize.brentq(lambda b: 2 * (np.log(1 / b) - 1 + b) - Z ** 2, 1e-6, 0.5)


B_H_ASYM = anchor_bH(3.4)
B_H_POIS = stats.norm.sf(3.4)   # exact-Poisson anchor: P(>=1 | b) = p  ->  b = p
R['b_H_anchor_asymptotic'] = B_H_ASYM
R['b_H_anchor_exact_poisson'] = B_H_POIS
print(f'[3] b_H anchors: asymptotic {B_H_ASYM:.3e}, exact-Poisson {B_H_POIS:.3e}')


class FewBin:
    def __init__(self, b_top, n_top, f_exp=1.0, kappa=1.0, mu_sig=0.0, sig_theta=0.3, b_H=None,
                 n_H=1, b_M=B_M, n_M=N_M_OBS, N_max=None, use_L=True):
        # f_exp: fraction of the near-median top-panel background (and data, Asimov-like) that the signal
        #        actually overlaps in S1c;  kappa: scale of the background model only (mis-modelling test)
        self.b = kappa * f_exp * b_top[sel]
        self.n = f_exp * n_top[sel]
        self.g = stats.norm.cdf(E_HI - mu_sig) - stats.norm.cdf(E_LO - mu_sig)
        self.sig_theta, self.n_H, self.b_M, self.n_M = sig_theta, n_H, b_M, n_M
        self.b_H = B_H_DEFAULT if b_H is None else b_H
        self.N_max, self.use_L = N_max, use_L

    def lnL(self, s, theta, N_top, N_M, N_lo):
        mu = theta * self.b + s * N_top * self.g
        ll = 0.0
        if self.use_L:
            ll += np.sum(self.n * np.log(mu) - mu) - 0.5 * ((theta - 1) / self.sig_theta) ** 2
            muM = self.b_M + s * N_M
            ll += self.n_M * np.log(muM) - muM
        muH = self.b_H + s
        ll += self.n_H * np.log(muH) - muH
        if self.N_max is not None:     # explicit 2024 null: best fit ~0, 90% UL N_max on 5.4-55 keV signal
            ll += -0.5 * (s * N_lo / (self.N_max / 1.2816)) ** 2
        return ll

    def prof_theta(self, s, N_top, N_M, N_lo):
        if not self.use_L:
            return self.lnL(s, 1.0, N_top, N_M, N_lo), 1.0
        f = lambda th: -self.lnL(s, th, N_top, N_M, N_lo)
        r = optimize.minimize_scalar(f, bounds=(0.02, 10.0), method='bounded', options={'xatol': 1e-5})
        return -r.fun, r.x

    def q0(self, N_lo, N_L2=0.0, N_M=0.0, return_all=False):
        N_top = N_lo + N_L2
        l0, th0 = self.prof_theta(0.0, N_top, N_M, N_lo)
        grid = np.concatenate([[0.0], np.logspace(-5, 1.3, 90)])
        vals = np.array([self.prof_theta(s, N_top, N_M, N_lo)[0] for s in grid])
        i = int(np.argmax(vals))
        if i == 0:
            s_hat, lmax = 0.0, vals[0]
        else:
            lo, hi = grid[max(i - 1, 1)], grid[min(i + 1, len(grid) - 1)]
            r = optimize.minimize_scalar(lambda ls: -self.prof_theta(np.exp(ls), N_top, N_M, N_lo)[0],
                                         bounds=(np.log(lo), np.log(hi)), method='bounded', options={'xatol': 1e-4})
            s_hat, lmax = float(np.exp(r.x)), -r.fun
            if lmax < vals[i]:
                s_hat, lmax = grid[i], vals[i]
        q = max(0.0, 2 * (lmax - l0))
        Z = np.sqrt(q) if s_hat > 0 else 0.0
        if return_all:
            return dict(q0=q, Z=Z, s_hat=s_hat, theta0=th0, theta_hat=self.prof_theta(s_hat, N_top, N_M, N_lo)[1],
                        lo_events_hat=s_hat * N_lo)
        return Z


def Z_curve(model, Nlo_grid, ratio_L2=0.0, ratio_M=0.0):
    return np.array([model.q0(N, N * ratio_L2, N * ratio_M) for N in Nlo_grid])


# calibration points: (label, N_lo from P003 [1000 GeV], LZ local Z at 1000 GeV, model key for N_L2/N_M ratios)
CAL = [('L10s', 0.20, lz.LSIG['L10s'][11], 'L10_1000'), ('L4s=O6s', 0.43, lz.LSIG['L4s'][11], 'O6s_1000'),
       ('L4v=O6v', 0.45, lz.LSIG['L4v'][11], 'O6v_1000'), ('L2s=O10s', 3.08, lz.LSIG['L2s'][11], 'O10s_1000'),
       ('L2v=O10v', 3.15, lz.LSIG['L2v'][11], 'O10v_1000'), ('L15s=O4s', 28.1, lz.LSIG['L15s'][11], 'O4s_1000'),
       ('L15v=O4v', 29.4, lz.LSIG['L15v'][11], 'O4v_1000'), ('L3v=O11v', 34.0, lz.LSIG['L3v'][11], 'O11v_1000'),
       ('L3s=O11s', 258.3, lz.LSIG['L3s'][11], 'O11s_1000'), ('L1v=O1v', 476.6, lz.LSIG['L1v'][11], 'O1v_1000'),
       ('L1s=O1s', 2752.1, lz.LSIG['L1s'][11], 'O1s_1000')]


def eval_cal(model, with_mid=True):
    out = []
    for lab, Nlo, Zlz, key in CAL:
        r = ops[ops.model == key].iloc[0]
        rl2, rm = (r.N_L2 / r.N_lo, r.N_M / r.N_lo) if with_mid else (0.0, 0.0)
        res = model.q0(Nlo, Nlo * rl2, Nlo * rm, return_all=True)
        out.append(dict(label=lab, N_lo=Nlo, Z_LZ=Zlz, Z_model=res['Z'], s_hat=res['s_hat'],
                        lo_events_hat=res['lo_events_hat'], theta_hat=res['theta_hat']))
    df = pd.DataFrame(out)
    df['resid'] = df.Z_model - df.Z_LZ
    return df


# --- anchor b_H: the maximum-significance LZ model (L10, 1000 GeV) with its own 55-200 keV companions gives 3.4 sigma
BASE = dict(f_exp=1.0, sig_theta=0.3, mu_sig=0.0)
B_H_DEFAULT = B_H_ASYM
l10 = ops[ops.model == 'L10_1000'].iloc[0]


def _Z_L10(bh):
    return FewBin(b_top_raw, n_top_raw, b_H=bh, **BASE).q0(l10.N_lo, l10.N_L2, l10.N_M) - 3.4


B_H_L10 = optimize.brentq(_Z_L10, 1e-4, 5e-3, xtol=1e-7)
B_H_DEFAULT = B_H_L10
R['b_H_anchor_L10_with_companions'] = float(B_H_L10)
R['Z_L10_zero_companion_anchor'] = float(FewBin(b_top_raw, n_top_raw, b_H=B_H_ASYM, **BASE).q0(l10.N_lo, l10.N_L2, l10.N_M))
print(f'     b_H anchored on L10 (N_lo 0.20, N_55-125 {l10.N_L2:.2f}, N_125-200 {l10.N_M:.2f}) -> 3.4 sigma: b_H = {B_H_L10:.3e}'
      f'  (with the zero-companion anchor L10 would give {R["Z_L10_zero_companion_anchor"]:.2f} sigma)')

# --- 3a. the mechanism: analytic two-bin illustration (zero-companion anchor b_H = 1.14e-3) --------------
print('[3a] two-bin analytic illustration (n_L = b_L, no nuisance):')
two_bin = []
for bL in [5, 15, 45]:
    for Nlo in [0.2, 3, 28, 258, 2752]:
        f = lambda s: -(np.log(1 + s / B_H_ASYM) - s + bL * np.log(1 + s * Nlo / bL) - s * Nlo)
        r = optimize.minimize_scalar(f, bounds=(1e-8, 5), method='bounded')
        q = max(0, -2 * r.fun); two_bin.append(dict(b_L=bL, N_lo=Nlo, s_hat=r.x, Z=np.sqrt(q), lo_events=r.x * Nlo))
two_bin = pd.DataFrame(two_bin)
two_bin.to_csv(os.path.join(OUT, 'P016_two_bin_illustration.csv'), index=False)
print(two_bin.pivot(index='N_lo', columns='b_L', values='Z').round(2))
# closed form: u = s N_lo / b_L solves b_L u^2 = 1 + u  (large-b_H^-1 limit)
R['two_bin_closed_form_u'] = {str(bL): float((1 + np.sqrt(1 + 4 * bL)) / (2 * bL)) for bL in [5, 15, 45]}

# --- 3b. baseline few-bin model and calibration scan ------------------------------------------------
Nlo_grid = np.logspace(-1.5, 3.7, 36)
scan = []
print('[3b] calibration scan over (f_exp, sig_theta, mu_sig, mid-energy companions):')
for f_exp in [0.5, 1.0]:
    for sig_theta in [0.3, 1.0]:
        for mu_sig in [0.0, -0.3]:
            for with_mid in [True, False]:
                m = FewBin(b_top_raw, n_top_raw, f_exp=f_exp, mu_sig=mu_sig, sig_theta=sig_theta)
                df = eval_cal(m, with_mid)
                chi2 = float(np.sum((df.resid / 0.15) ** 2))
                scan.append(dict(f_exp=f_exp, sig_theta=sig_theta, mu_sig=mu_sig, with_mid=with_mid, chi2=chi2,
                                 rms=float(np.sqrt(np.mean(df.resid ** 2))), max_abs=float(df.resid.abs().max()),
                                 **{f"Z_{lab}": z for lab, z in zip(df.label, df.Z_model)}))
                print(f"     f_exp {f_exp:4.2f} sig_th {sig_theta:3.1f} mu {mu_sig:+4.1f} mid {int(with_mid)}  rms {scan[-1]['rms']:.2f}"
                      f"  Z: " + ' '.join(f"{z:.2f}" for z in df.Z_model))
scan = pd.DataFrame(scan)
scan.to_csv(os.path.join(OUT, 'P016_calibration_scan.csv'), index=False)
best = scan.sort_values('rms').iloc[0]
print('     best:', best[['f_exp', 'sig_theta', 'mu_sig', 'with_mid', 'rms', 'max_abs']].to_dict())
R['calibration_best'] = {k: (float(v) if not isinstance(v, (bool, np.bool_)) else bool(v)) for k, v in
                         best[['f_exp', 'sig_theta', 'mu_sig', 'with_mid', 'chi2', 'rms', 'max_abs']].items()}

# Baseline: the full top panel (f_exp = 1), sig_theta = 0.3, mu = 0, with mid-energy companions
base = FewBin(b_top_raw, n_top_raw, **BASE)
cal_base = eval_cal(base, True)
cal_base.to_csv(os.path.join(OUT, 'P016_calibration_baseline.csv'), index=False)
print('[3c] baseline calibration table:')
print(cal_base.round(3).to_string(index=False))
R['calibration_baseline_rms'] = float(np.sqrt(np.mean(cal_base.resid ** 2)))
R['baseline_settings'] = dict(BASE, b_H=B_H_DEFAULT, window_hi_sigma=WIN_HI, b_M=B_M, n_M=N_M_OBS)
# background and data within +-1.5 sigma of the NR median (top panel)
w15 = (EDGES[:-1] >= -1.5 - 1e-9) & (EDGES[1:] <= 1.5 + 1e-9)
wm = (EDGES[:-1] >= -1.5 - 1e-9) & (EDGES[1:] <= 0 + 1e-9)
R['top_panel_pm1p5'] = dict(b=float(b_top_raw[w15].sum()), n=float(n_top_raw[w15].sum()),
                            b_below_median=float(b_top_raw[wm].sum()), n_below_median=float(n_top_raw[wm].sum()),
                            accidentals=float(np.nan_to_num(hist_top['Accidentals'])[w15].sum()),
                            NRs=float(np.nan_to_num(hist_top['NRs'])[w15].sum()))
# Fisher-information effective background for a unit-normal signal: b_eff = 1 / sum(g_i^2 / b_i)
g_all = stats.norm.cdf(EDGES[1:]) - stats.norm.cdf(EDGES[:-1])
ok = b_top_raw > 0
R['b_eff_fisher_top_panel'] = float(1.0 / np.sum(g_all[ok] ** 2 / b_top_raw[ok]))
print(f"     +-1.5 sigma: b={R['top_panel_pm1p5']['b']:.1f}, n={R['top_panel_pm1p5']['n']:.0f}; below median b={R['top_panel_pm1p5']['b_below_median']:.2f}, "
      f"n={R['top_panel_pm1p5']['n_below_median']:.0f}; Fisher b_eff = {R['b_eff_fisher_top_panel']:.1f}")

# Z vs N_lo curves (baseline shapes: O4-like ratios for the mid-energy companions; also without)
r_o4 = ops[ops.model == 'O4s_1000'].iloc[0]
curves = {'baseline (O4-like 55-200 keV companions)': Z_curve(base, Nlo_grid, r_o4.N_L2 / r_o4.N_lo, r_o4.N_M / r_o4.N_lo),
          'no 55-200 keV companions': Z_curve(base, Nlo_grid),
          'sig_theta = 1.0': Z_curve(FewBin(b_top_raw, n_top_raw, f_exp=1, sig_theta=1.0), Nlo_grid, r_o4.N_L2 / r_o4.N_lo, r_o4.N_M / r_o4.N_lo),
          'f_exp = 0.5': Z_curve(FewBin(b_top_raw, n_top_raw, f_exp=0.5, sig_theta=0.3), Nlo_grid, r_o4.N_L2 / r_o4.N_lo, r_o4.N_M / r_o4.N_lo),
          'kappa = 3 (bkg x3, data fixed)': Z_curve(FewBin(b_top_raw, n_top_raw, kappa=3.0, sig_theta=0.3), Nlo_grid, r_o4.N_L2 / r_o4.N_lo, r_o4.N_M / r_o4.N_lo)}
cdf = pd.DataFrame({'N_lo': Nlo_grid, **curves})
cdf.to_csv(os.path.join(OUT, 'P016_Z_vs_Nlo_curves.csv'), index=False)


def crossing(Nlo, Z, level):
    idx = np.where((Z[:-1] >= level) & (Z[1:] < level))[0]
    if len(idx) == 0:
        return float('nan')
    i = idx[0]
    return float(np.exp(np.interp(level, [Z[i + 1], Z[i]], [np.log(Nlo[i + 1]), np.log(Nlo[i])])))


R['Nlo_crossings'] = {k: {'Z<3': crossing(Nlo_grid, v, 3.0), 'Z<2': crossing(Nlo_grid, v, 2.0), 'Z<1': crossing(Nlo_grid, v, 1.0)}
                      for k, v in curves.items()}
print('[3d] N_lo at which Z drops below 3/2/1 sigma:')
for k, v in R['Nlo_crossings'].items():
    print(f"     {k:45s} {v['Z<3']:8.1f} {v['Z<2']:8.1f} {v['Z<1']:8.1f}")

# ----------------------------------------------------------------------------------------------
# 4. Per-operator table: joint Z (baseline), Z with explicit 2024 constraints, and Bayes factors
# ----------------------------------------------------------------------------------------------
NMAX_LIST = [3, 5, 10, 20]
S_MAX = 10.0     # P001 flat prior s in [0, 10]


def bayes_factor(model, N_lo, N_L2, N_M, prior='flat', penalty='likelihood'):
    """B10 = 1 + (1/b_H) int pi(s) s e^{-s} W(s) ds, with W the low-energy likelihood ratio (theta marginalised)."""
    N_top = N_lo + N_L2
    th = np.linspace(max(0.02, 1 - 4 * model.sig_theta), 1 + 4 * model.sig_theta, 41)
    wth = stats.norm.pdf(th, 1, model.sig_theta); wth /= wth.sum()

    ref = model.b  # reference for the log-likelihood (theta = 1, s = 0): keeps exponents O(1)
    lnL0 = np.sum(model.n[None, :] * np.log(th[:, None] * model.b[None, :]) - th[:, None] * model.b[None, :], axis=1) \
        - np.sum(model.n * np.log(ref) - ref)
    Z0 = float(np.sum(wth * np.exp(lnL0)))

    def W(s):
        """marginal likelihood ratio  [int dtheta p(theta) L(s,theta)] / [int dtheta p(theta) L(0,theta)]"""
        if penalty == 'exp':
            return np.exp(-s * N_lo)
        mu = th[:, None] * model.b[None, :] + s * N_top * model.g[None, :]
        lnLs = np.sum(model.n[None, :] * np.log(mu) - mu, axis=1) - np.sum(model.n * np.log(ref) - ref) - s * N_M
        return float(np.sum(wth * np.exp(lnLs)) / Z0)

    if prior == 'flat':
        val, _ = integrate.quad(lambda s: (1 / S_MAX) * s * np.exp(-s) * W(s), 0, S_MAX, limit=200)
    else:  # log-uniform on [0.01, 30] (P001 alternative)
        val, _ = integrate.quad(lambda s: s * np.exp(-s) * W(s) / (s * np.log(30 / 0.01)), 0.01, 30, limit=200)
    return 1 + val / model.b_H


table = []
for _, r in ops.iterrows():
    res = base.q0(r.N_lo, r.N_L2, r.N_M, return_all=True)
    row = dict(model=r.model, operator=r.operator, isospin=r.isospin, m_chi_GeV=r.m_chi_GeV, N_lo=r.N_lo, N_L2=r.N_L2, N_M=r.N_M,
               Z_joint=res['Z'], s_hat=res['s_hat'], lo_events_hat=res['lo_events_hat'], theta_hat=res['theta_hat'],
               P_ge1_event_at_shat=1 - np.exp(-res['s_hat']),                      # does the best-fit model predict the event?
               s_total_hat=res['s_hat'] * (1 + r.N_lo + r.N_L2 + r.N_M),           # total fitted signal events, all energies
               # large-N_lo analytic guide (n = b, Gaussian signal): Z^2 ~ 2[ln(sqrt(b_eff)/(N_top b_H)) - 1/2 - s N_M]
               Z_analytic=np.sqrt(max(0.0, 2 * (np.log(np.sqrt(R['b_eff_fisher_top_panel']) / ((r.N_lo + r.N_L2) * B_H_DEFAULT)) - 0.5
                                                - np.sqrt(R['b_eff_fisher_top_panel']) / (r.N_lo + r.N_L2) * (1 + r.N_M)))))
    # LZ table value where a mapping exists (1000 GeV / 4000 GeV / 200 GeV columns)
    LMAP = {'O1s': 'L1s', 'O1v': 'L1v', 'O10s': 'L2s', 'O10v': 'L2v', 'O11s': 'L3s', 'O11v': 'L3v', 'O6s': 'L4s', 'O6v': 'L4v',
            'O4s': 'L15s', 'O4v': 'L15v', 'L10s': 'L10s'}
    key = f"{r.operator}{r.isospin}"
    mi = {200: 9, 1000: 11, 4000: 12}[int(r.m_chi_GeV)]
    row['Z_LZ_table'] = lz.LSIG[LMAP[key]][mi] if key in LMAP else np.nan
    for Nm in NMAX_LIST:
        m2 = FewBin(b_top_raw, n_top_raw, N_max=Nm, **BASE)           # Fig.5 bins + explicit 2024 constraint
        row[f'Z_Nmax{Nm}'] = m2.q0(r.N_lo, r.N_L2, r.N_M)
        m3 = FewBin(b_top_raw, n_top_raw, N_max=Nm, use_L=False, **BASE)   # 2024 constraint alone (no Fig.5 bins)
        row[f'Z_only2024_Nmax{Nm}'] = m3.q0(r.N_lo, r.N_L2, r.N_M)
    row['B10_flat'] = bayes_factor(base, r.N_lo, r.N_L2, r.N_M)
    row['B10_exp_penalty'] = bayes_factor(base, r.N_lo, r.N_L2, r.N_M, penalty='exp')
    row['B10_loguniform'] = bayes_factor(base, r.N_lo, r.N_L2, r.N_M, prior='log')
    table.append(row)
    print(f"[4] {r.model:10s} N_lo {r.N_lo:8.2f}  Z_joint {res['Z']:.2f} (LZ {row['Z_LZ_table']}; analytic {row['Z_analytic']:.2f})  s_hat {res['s_hat']:.3f}"
          f"  P(>=1|s_hat) {row['P_ge1_event_at_shat']:.3f}  s_tot {row['s_total_hat']:.2f}"
          f"  Nmax3/5/10/20: {row['Z_Nmax3']:.2f}/{row['Z_Nmax5']:.2f}/{row['Z_Nmax10']:.2f}/{row['Z_Nmax20']:.2f}"
          f"  B10 {row['B10_flat']:.1f} (exp {row['B10_exp_penalty']:.1f})")
table = pd.DataFrame(table)
B10_nolo = 1 + (1 / B_H_DEFAULT) * integrate.quad(lambda s: (1 / S_MAX) * s * np.exp(-s), 0, S_MAX)[0]
R['B10_no_low_energy_penalty'] = float(B10_nolo)
# posterior weights over 1000 GeV operator classes (equal prior weight per model; s and v listed separately)
t1000 = table[table.m_chi_GeV == 1000].copy()
for col in ['B10_flat', 'B10_exp_penalty']:
    t1000[f'w_{col}'] = t1000[col] / t1000[col].sum()
t1000 = t1000.sort_values('B10_flat', ascending=False)
table = table.merge(t1000[['model', 'w_B10_flat', 'w_B10_exp_penalty']], on='model', how='left')
table.to_csv(os.path.join(OUT, 'P016_operator_table.csv'), index=False)
classes = {}
for Nm in [None] + NMAX_LIST:
    col = 'Z_joint' if Nm is None else f'Z_Nmax{Nm}'
    classes[str(Nm)] = {'>=3': t1000[t1000[col] >= 3.0].model.tolist(),
                        '2-3': t1000[(t1000[col] >= 2.0) & (t1000[col] < 3.0)].model.tolist(),
                        '<2': t1000[t1000[col] < 2.0].model.tolist()}
R['classes_1000GeV'] = classes
R['posterior_weights_1000GeV_flat'] = {m: float(w) for m, w in zip(t1000.model, t1000.w_B10_flat)}
R['posterior_weights_1000GeV_exp'] = {m: float(w) for m, w in zip(t1000.model, t1000.w_B10_exp_penalty)}
R['B10_range_1000GeV'] = dict(max=float(t1000.B10_flat.max()), min=float(t1000.B10_flat.min()),
                              top=t1000.model.iloc[0], bottom=t1000.model.iloc[-1])
R['best_fit_diagnostics_1000GeV'] = {m: dict(s_hat=float(a), P_ge1=float(b), s_total=float(c), lo_events=float(d))
                                     for m, a, b, c, d in zip(t1000.model, t1000.s_hat, t1000.P_ge1_event_at_shat, t1000.s_total_hat, t1000.lo_events_hat)}
R['max_Z_shift_from_explicit_2024_constraint_Nmax3'] = float((t1000.Z_joint - t1000.Z_Nmax3).abs().max())
R['max_Z_shift_from_explicit_2024_constraint_Nmax5'] = float((t1000.Z_joint - t1000.Z_Nmax5).abs().max())
R['Z_only2024_Nmax3_min_over_Nlo_le_50'] = float(t1000[t1000.N_lo <= 50].Z_only2024_Nmax3.min())
top5 = t1000.head(5)
print('[4b] posterior weights (1000 GeV, equal prior per model):', {m: round(w, 3) for m, w in zip(t1000.model, t1000.w_B10_flat)})
R['cumulative_weight_top5_flat'] = float(top5.w_B10_flat.sum())
R['cumulative_weight_Nlo_le_1_flat'] = float(t1000[t1000.N_lo <= 1].w_B10_flat.sum())
R['cumulative_weight_Nlo_le_5_flat'] = float(t1000[t1000.N_lo <= 5].w_B10_flat.sum())
R['cumulative_weight_Nlo_gt_20_flat'] = float(t1000[t1000.N_lo > 20].w_B10_flat.sum())

# ----------------------------------------------------------------------------------------------
# 5. Sensitivity to the low-energy background level near the NR median
# ----------------------------------------------------------------------------------------------
sens = []
sel_models = ['L10_1000', 'O6s_1000', 'O10s_1000', 'O9s_1000', 'O4s_1000', 'O11v_1000', 'O11s_1000', 'O1v_1000', 'O1s_1000']
for kappa in [0.5, 1.0, 2.0, 3.0]:
    for mode in ['model_only', 'model_and_data']:
        # model_only: background model scaled, data as observed (a mis-modelled background);
        # model_and_data: both scaled (a genuinely larger background, Asimov data)
        if mode == 'model_only':
            m = FewBin(b_top_raw, n_top_raw, kappa=kappa, **BASE)
        else:
            m = FewBin(kappa * b_top_raw, kappa * n_top_raw, **BASE)
        for key in sel_models:
            r = ops[ops.model == key].iloc[0]
            sens.append(dict(kappa=kappa, mode=mode, model=key, N_lo=r.N_lo, Z=m.q0(r.N_lo, r.N_L2, r.N_M)))
sens = pd.DataFrame(sens)
sens.to_csv(os.path.join(OUT, 'P016_sensitivity_background.csv'), index=False)
print('[5] sensitivity (Z) to the low-energy background scale kappa:')
print(sens.pivot_table(index='model', columns=['mode', 'kappa'], values='Z').round(2).loc[sel_models].to_string())
R['sensitivity_O4s_1000'] = {f"{row['mode']}_k{row['kappa']}": float(row.Z) for _, row in sens[sens.model == 'O4s_1000'].iterrows()}
R['sensitivity_O1v_1000'] = {f"{row['mode']}_k{row['kappa']}": float(row.Z) for _, row in sens[sens.model == 'O1v_1000'].iterrows()}

# ----------------------------------------------------------------------------------------------
# 6. Toy-MC check of the asymptotic q0 -> Z mapping for the few-bin model (theta fixed at 1)
# ----------------------------------------------------------------------------------------------
rng = np.random.default_rng(16)
NTOY = 200_000
b_vec = np.concatenate([b_top_raw[sel], [B_M, B_H_DEFAULT]])
toy_check = []
for key in ['L10_1000', 'O10s_1000', 'O4s_1000', 'O11s_1000']:
    r = ops[ops.model == key].iloc[0]
    f_vec = np.concatenate([(r.N_lo + r.N_L2) * base.g, [r.N_M, 1.0]])    # signal per unit s
    n_toy = rng.poisson(b_vec[None, :], size=(NTOY, len(b_vec)))
    # s_hat: root of sum n_j f_j / (b_j + s f_j) = 1 ... solve by bisection on [0, 20] (vectorised)
    grad0 = (n_toy * f_vec / b_vec).sum(axis=1) - f_vec.sum()
    lo = np.zeros(NTOY); hi = np.full(NTOY, 20.0)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        g = (n_toy * f_vec / (b_vec + mid[:, None] * f_vec)).sum(axis=1) - f_vec.sum()
        pos = g > 0
        lo = np.where(pos, mid, lo); hi = np.where(pos, hi, mid)
    s_hat = np.where(grad0 > 0, 0.5 * (lo + hi), 0.0)
    mu_hat = b_vec + s_hat[:, None] * f_vec
    q0_toys = 2 * ((n_toy * (np.log(mu_hat) - np.log(b_vec)) - (mu_hat - b_vec)).sum(axis=1))
    q0_toys = np.where(s_hat > 0, np.maximum(q0_toys, 0), 0.0)
    # observed q0 with theta fixed (for a like-for-like comparison)
    m_fix = FewBin(b_top_raw, n_top_raw, f_exp=1.0, mu_sig=0.0, sig_theta=1e-3)
    q_obs = m_fix.q0(r.N_lo, r.N_L2, r.N_M, return_all=True)['q0']
    p_toy = float(np.mean(q0_toys >= q_obs))
    toy_check.append(dict(model=key, N_lo=r.N_lo, q0_obs=q_obs, Z_wilks=np.sqrt(q_obs), p_toy=p_toy,
                          Z_toy=float(stats.norm.isf(p_toy)) if p_toy > 0 else np.inf,
                          frac_toys_q0_gt0=float(np.mean(q0_toys > 0)), n_toys=NTOY))
    print(f"[6] toys {key:10s} q0_obs {q_obs:.2f} Z_wilks {np.sqrt(q_obs):.2f}  p_toy {p_toy:.2e} Z_toy {toy_check[-1]['Z_toy']:.2f}"
          f"  P(q0>0)={toy_check[-1]['frac_toys_q0_gt0']:.3f}")
toy_check = pd.DataFrame(toy_check)
toy_check.to_csv(os.path.join(OUT, 'P016_toy_check.csv'), index=False)
R['toy_check'] = toy_check.replace([np.inf], 9.99).to_dict(orient='records')

# ----------------------------------------------------------------------------------------------
# 7. Figures
# ----------------------------------------------------------------------------------------------
# Fig 1: digitised top panel with data, model, and the shape of s*N_lo companions for O4 (s_hat) and s = 1
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.step(EDGES, np.append(b_top_raw, b_top_raw[-1]), where='post', color='#1f4e9c', lw=1.8, label='background model total (digitised Fig. 5 top)')
for nm, c in [('ERs', '#00c2f9'), ('Internal', '#d600d6'), ('Accidentals', '#c9a800'), ('NRs', '#7cae00')]:
    v = np.nan_to_num(hist_top[nm]); v[v == 0] = np.nan
    ax.step(EDGES, np.append(v, v[-1]), where='post', color=c, lw=1, alpha=0.8, label=nm.lower())
m = n_top_raw > 0
ax.plot(CENTRES[m], n_top_raw[m], 'ko', ms=4, label='data (digitised)')
res_o4 = base.q0(r_o4.N_lo, r_o4.N_L2, r_o4.N_M, return_all=True)
gs = stats.norm.cdf(EDGES[1:]) - stats.norm.cdf(EDGES[:-1])
ax.step(EDGES, np.append(gs * (r_o4.N_lo + r_o4.N_L2), (gs * (r_o4.N_lo + r_o4.N_L2))[-1]), where='post', color='#b03a2e', ls='--', lw=1.4,
        label=f'O4 companions, s = 1 (N = {r_o4.N_lo + r_o4.N_L2:.0f})')
ax.step(EDGES, np.append(gs * res_o4['s_hat'] * (r_o4.N_lo + r_o4.N_L2), (gs * res_o4['s_hat'] * (r_o4.N_lo + r_o4.N_L2))[-1]), where='post',
        color='#b03a2e', lw=1.4, label=f"O4 companions at s_hat = {res_o4['s_hat']:.2f}")
ax.set_yscale('log'); ax.set_ylim(1e-3, 3e2); ax.set_xlim(-8, 8)
ax.set_xlabel(r'(log$_{10}$S2c $-$ $\mu_{NR}$)/$\sigma_{NR}$   (S1c < 250 phd)'); ax.set_ylabel('events / 0.5$\\sigma$ bin')
ax.legend(fontsize=7, loc='upper left', ncol=2); ax.grid(alpha=0.3, which='both')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig1_top_panel_digitised.png'), dpi=160); plt.close(fig)

# Fig 2: Z vs N_lo with LZ calibration points
fig, ax = plt.subplots(figsize=(7, 4.2))
cols = ['#1f4e9c', '#7f7f7f', '#2a9d8f', '#e07b00', '#b03a2e']
for (k, v), c in zip(curves.items(), cols):
    ax.plot(Nlo_grid, v, color=c, lw=1.8 if 'baseline' in k else 1.2, label=k)
ax.plot(cal_base.N_lo, cal_base.Z_LZ, 'k*', ms=9, label='LZ Tables S6/S7 (1000 GeV)')
for _, rr in cal_base.iterrows():
    ax.annotate(rr.label.split('=')[-1], (rr.N_lo, rr.Z_LZ), textcoords='offset points', xytext=(4, -10), fontsize=7)
ax.axhline(3, color='k', ls=':', lw=0.8); ax.axhline(2, color='k', ls=':', lw=0.8)
ax.set_xscale('log'); ax.set_xlabel(r'$N_{lo}$ = 5.4-55 keV signal events per 200-270 keV event'); ax.set_ylabel('local significance Z')
ax.set_ylim(0, 3.8); ax.legend(fontsize=7, loc='lower left'); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig2_Z_vs_Nlo.png'), dpi=160); plt.close(fig)

# Fig 3: per-operator Z (1000 GeV) baseline vs explicit 2024 constraints
tt = t1000.sort_values('N_lo')
fig, ax = plt.subplots(figsize=(8, 4.2))
x = np.arange(len(tt))
wd = 0.16
ax.bar(x - 2 * wd, tt.Z_joint, wd, color='#1f4e9c', label='Fig. 5 bins only (baseline)')
for i, Nm in enumerate(NMAX_LIST):
    ax.bar(x + (i - 1) * wd, tt[f'Z_Nmax{Nm}'], wd, color=['#2a9d8f', '#e07b00', '#b03a2e', '#7f7f7f'][i], label=f'+ 2024 null, N_max = {Nm}')
ax.plot(x, tt.Z_LZ_table, 'k*', ms=8, label='LZ table')
ax.axhline(3, color='k', ls=':', lw=0.8); ax.axhline(2, color='k', ls=':', lw=0.8)
fmtN = lambda n: f"{n:.2g}" if n < 10 else f"{n:.0f}"
ax.set_xticks(x); ax.set_xticklabels([f"{m.split('_')[0]}\n{fmtN(n)}" for m, n in zip(tt.model, tt.N_lo)], fontsize=6.5)
ax.set_xlabel('operator (1000 GeV) and N_lo'); ax.set_ylabel('local significance Z'); ax.set_ylim(0, 3.8)
ax.legend(fontsize=7, ncol=3, loc='upper right'); ax.grid(alpha=0.3, axis='y')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig3_operator_Z_Nmax.png'), dpi=160); plt.close(fig)

# Fig 4: posterior weights
fig, ax = plt.subplots(figsize=(8, 3.8))
tt = t1000.sort_values('N_lo')
ax.bar(x - 0.2, tt.w_B10_flat, 0.4, color='#1f4e9c', label='Fig. 5 low-energy likelihood folded in')
ax.bar(x + 0.2, tt.w_B10_exp_penalty, 0.4, color='#b03a2e', label='exp(-s N_lo) penalty (background-free low-energy region)')
ax.set_xticks(x); ax.set_xticklabels([f"{m.split('_')[0]}\n{fmtN(n)}" for m, n in zip(tt.model, tt.N_lo)], fontsize=6.5)
ax.set_ylabel('relative posterior weight (equal priors)'); ax.set_yscale('log'); ax.set_ylim(1e-4, 1)
ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='y', which='both')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig4_posterior_weights.png'), dpi=160); plt.close(fig)

R['runtime_s'] = time.time() - t0
with open(os.path.join(OUT, 'P016_results.json'), 'w') as f:
    json.dump(R, f, indent=1, default=float)
print(f'done in {R["runtime_s"]:.0f} s')
