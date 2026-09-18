"""
P020 -- What LZ's data since April 2024 should show: falsifiable predictions for the untouched exposure.

Run from the simulation root:   .venv/bin/python output/code/P020_future_lz.py

Sections
  1. Exposure of the post-1-April-2024 data (calendar days, live fraction, fiducial mass scenarios)
  2. Background expectations in the new exposure (Fig. 5 bottom panel 0.0106; +-2 sigma neighbourhood 2e-4)
  3. Signal predictions: L10 best fit (plug-in and Poisson-Gamma predictive); inelastic O1 (1000 GeV) at
     delta = 300/350/366/380 keV with seasonal PDFs from WimPyDD; seasonal visibility for N = 2, 3, 5
  4. Decision table: posterior (P001 three-hypothesis framework), frequentist local Z of the combined sample,
     N = 0 upper limits
  5. P001 corpus-prior mixture predictive; crisp forecast numbers
Outputs -> output/work/P020/  (CSV/JSON tables, figures/*.png)
"""
from __future__ import annotations
import sys, json, math, datetime as dt
sys.path.insert(0, 'output/code')
import numpy as np
from scipy import stats, integrate, special, optimize
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import lzcommon as lz

OUT = 'output/work/P020'
FIG = f'{OUT}/figures'
import os
os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(20260905)
RES = {}

# ----------------------------------------------------------------------------------------------
# 1. Exposure
# ----------------------------------------------------------------------------------------------
D_FIRST_START, D_FIRST_END = dt.date(2023, 3, 27), dt.date(2024, 4, 1)   # LZ Data Analysis paragraph
first_cal_days = (D_FIRST_END - D_FIRST_START).days                        # 371
LIVE_FRAC = lz.LZ['live_days'] / first_cal_days                             # 220/371 = 0.593
LIVE_RANGE = (0.50, 0.75)
M_FID, M_FID_ALT = lz.LZ['fiducial_mass_t'], 5.5                            # 4.71 t (same cuts) / 5.4-5.5 t if MSSI controlled
E0 = lz.LZ['exposure_tyr']                                                  # 2.84 t yr
D_NEW_START = dt.date(2024, 4, 1)
END_DATES = {'2026-09-02': dt.date(2026, 9, 2), '2026-12-31': dt.date(2026, 12, 31), '2027-12-31': dt.date(2027, 12, 31)}

exp_rows = []
for tag, dend in END_DATES.items():
    cal = (dend - D_NEW_START).days
    for M in (M_FID, M_FID_ALT):
        for f in (LIVE_FRAC, *LIVE_RANGE):
            live = cal * f
            E = live / 365.25 * M
            exp_rows.append(dict(end=tag, cal_days=cal, live_frac=round(f, 4), live_days=round(live, 1), M_t=M,
                                 exposure_tyr=round(E, 3), k=round(E / E0, 3)))
import csv
with open(f'{OUT}/exposure_scenarios.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(exp_rows[0].keys())); w.writeheader(); w.writerows(exp_rows)

def exposure(dend, f=LIVE_FRAC, M=M_FID):
    return (dend - D_NEW_START).days * f / 365.25 * M

E_NEW = exposure(END_DATES['2026-09-02']); K = E_NEW / E0
E_DEC26 = exposure(END_DATES['2026-12-31']); K_DEC26 = E_DEC26 / E0
E_DEC27 = exposure(END_DATES['2027-12-31']); K_DEC27 = E_DEC27 / E0
RES['exposure'] = dict(first_cal_days=first_cal_days, live_frac=LIVE_FRAC,
                       cal_days_to_2026_09_02=(END_DATES['2026-09-02'] - D_NEW_START).days,
                       E_new_tyr=E_NEW, k_new=K, E_dec26_tyr=E_DEC26, k_dec26=K_DEC26, E_dec27_tyr=E_DEC27, k_dec27=K_DEC27,
                       E_new_range_livefrac=[exposure(END_DATES['2026-09-02'], f) for f in LIVE_RANGE],
                       E_new_5p5t=exposure(END_DATES['2026-09-02'], M=M_FID_ALT))
print(f"[1] first run: {first_cal_days} calendar d, live fraction {LIVE_FRAC:.3f}")
print(f"    1 Apr 2024 -> 2 Sep 2026: {RES['exposure']['cal_days_to_2026_09_02']} d  ->  E = {E_NEW:.2f} t yr = {K:.2f} x 2.84"
      f"  (live 0.50-0.75: {RES['exposure']['E_new_range_livefrac'][0]:.2f}-{RES['exposure']['E_new_range_livefrac'][1]:.2f}; 5.5 t: {RES['exposure']['E_new_5p5t']:.2f})")
print(f"    -> 31 Dec 2026: {E_DEC26:.2f} t yr (k={K_DEC26:.2f});  -> 31 Dec 2027: {E_DEC27:.2f} t yr (k={K_DEC27:.2f})")

# ----------------------------------------------------------------------------------------------
# 2. Background
# ----------------------------------------------------------------------------------------------
B_PANEL = lz.LZ['bkg_highS1_panel'][0]     # 0.0106 whole S1c>500 phd panel (Fig. 5 caption)
B_NB = 2e-4                                 # within +-2 sigma of NR median at S1c>500 (dossier / P001)
B_EFF = 1e-3                                # LZ-equivalent effective single-event background (P001)
bk = {}
for name, b in (('panel_0.0106', B_PANEL), ('eff_1e-3', B_EFF), ('nb_2e-4', B_NB)):
    for tag, k in (('2026-09-02', K), ('2026-12-31', K_DEC26), ('2027-12-31', K_DEC27)):
        mu = b * k
        bk[f'{name}|{tag}'] = dict(mu=mu, P_ge1=1 - math.exp(-mu), P_ge2=float(stats.poisson.sf(1, mu)))
RES['background'] = bk
print(f"[2] background in new exposure (k={K:.2f}): panel {B_PANEL*K:.4f} (P>=1 = {bk['panel_0.0106|2026-09-02']['P_ge1']:.4f});"
      f" nb {B_NB*K:.2e} (P>=1 = {bk['nb_2e-4|2026-09-02']['P_ge1']:.2e})")

# ----------------------------------------------------------------------------------------------
# 3a. L10 best fit: plug-in Poisson and Poisson-Gamma predictive
# ----------------------------------------------------------------------------------------------
S_BF, S_UP, S_DN = 1.0, 1.4, 0.7          # Table I: 1.0 +1.4 -0.7 events in 2.84 t yr
# Check: the single-event Poisson likelihood s e^{-s} has -2 dlnL = 1 at s = 0.30 and 2.36 -> matches Table I
f_int = lambda s: math.log(s) - s + 1.0 + 0.5
s_lo = optimize.brentq(f_int, 1e-3, 1.0); s_hi = optimize.brentq(f_int, 1.0, 10.0)
RES['L10_interval_check'] = dict(s_lo=s_lo, s_hi=s_hi, table_I=[S_BF - S_DN, S_BF + S_UP])
print(f"[3a] Poisson n=1 likelihood interval (-2dlnL=1): [{s_lo:.2f}, {s_hi:.2f}] vs Table I [0.3, 2.4] -> L(s) ~ s e^-s")

def predictive_rows(k, b=B_NB):
    """P(N) for N = 0..3 and >=4 under: background only; plug-in s=1.0 (+b); s = 0.3 and 2.4 (interval ends);
    Poisson-Gamma with flat prior (posterior Gamma(2,1)); Jeffreys prior (Gamma(1.5,1))."""
    n = np.arange(0, 4)
    rows = {}
    def poi(mu): p = stats.poisson.pmf(n, mu); return list(p) + [1 - p.sum()], mu
    rows['bkg_only_nb'], _ = poi(b * k)
    rows['bkg_only_panel'], _ = poi(B_PANEL * k)
    rows['s=1.0 plug-in'], _ = poi((S_BF + b) * k)
    rows['s=0.3'], _ = poi((S_BF - S_DN + b) * k)
    rows['s=2.4'], _ = poi((S_BF + S_UP + b) * k)
    for name, alpha in (('Poisson-Gamma flat prior (Gamma(2,1))', 2.0), ('Poisson-Gamma Jeffreys (Gamma(1.5,1))', 1.5)):
        # N | s ~ Poisson(k s); s ~ Gamma(alpha, rate 1)  ->  NB(alpha, p = 1/(1+k))
        p = stats.nbinom.pmf(n, alpha, 1.0 / (1.0 + k))
        rows[name] = list(p) + [1 - p.sum()]
    means = {'s=1.0 plug-in': (S_BF + b) * k, 'Poisson-Gamma flat prior (Gamma(2,1))': 2.0 * k,
             'Poisson-Gamma Jeffreys (Gamma(1.5,1))': 1.5 * k, 'bkg_only_nb': b * k, 'bkg_only_panel': B_PANEL * k,
             's=0.3': 0.3 * k, 's=2.4': 2.4 * k}
    return rows, means

pred_rows, pred_means = predictive_rows(K)
with open(f'{OUT}/predictive_L10_2026-09-02.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['hypothesis', 'mean', 'P0', 'P1', 'P2', 'P3', 'P_ge4'])
    for kname, p in pred_rows.items():
        w.writerow([kname, f'{pred_means[kname]:.4f}'] + [f'{x:.4f}' for x in p])
RES['predictive_L10'] = {kname: dict(mean=pred_means[kname], P=p) for kname, p in pred_rows.items()}
for kname in ('s=1.0 plug-in', 'Poisson-Gamma flat prior (Gamma(2,1))'):
    p = pred_rows[kname]
    print(f"     {kname:40s} mean {pred_means[kname]:.2f}: P0={p[0]:.3f} P1={p[1]:.3f} P2={p[2]:.3f} P3={p[3]:.3f} P>=4={p[4]:.3f}")
# also for the Dec 2026 / Dec 2027 exposures
RES['predictive_L10_other_ends'] = {}
for tag, k in (('2026-12-31', K_DEC26), ('2027-12-31', K_DEC27)):
    r, m = predictive_rows(k)
    RES['predictive_L10_other_ends'][tag] = {kk: dict(mean=m[kk], P=r[kk]) for kk in ('s=1.0 plug-in', 'Poisson-Gamma flat prior (Gamma(2,1))')}

# ----------------------------------------------------------------------------------------------
# 3b. Inelastic seasonal PDFs from WimPyDD (O1 isoscalar, m = 1000 GeV), delta = 300/350/366/380 keV
# ----------------------------------------------------------------------------------------------
def eff(E):
    """LZ NR efficiency (P006 parametrisation of Fig. S2): 50% at 5.4 keV, plateau 0.96, roll-off Phi((269.9-E)/15)."""
    E = np.asarray(E, float)
    lo = 1.0 / (1.0 + np.exp(-(E - lz.LZ['E_50pct_low_keV']) / 1.0))
    hi = stats.norm.cdf((lz.LZ['E_50pct_high_keV'] - E) / 15.0)
    return lz.LZ['eff_plateau'] * lo * hi

E_GRID = np.linspace(60.0, 330.0, 91)            # inelastic onsets are >= 85 keV for delta >= 300 keV (P002)
M_CHI = 1000.0
DELTAS = [300.0, 350.0, 366.0, 380.0]
DAY_GRID = np.unique(np.concatenate([np.arange(1.0, 366.0, 8.0), [152.0, 152.5, 153.0, 156.0, 160.0, 167.9, 335.0, 336.0]]))
print(f"[3b] WimPyDD seasonal rates: {len(DELTAS)} deltas x {len(DAY_GRID)} days x {len(E_GRID)} energies")
HAM = lz.wd_hamiltonian('O1s', {1: (1.0, 0.0)})
halos = {d: lz.wd_halo(day_of_year=float(d)) for d in DAY_GRID}
halo_avg = lz.wd_halo()

season = {}
for delta in DELTAS:
    R_roi, R_noeff, R_gt270 = [], [], []
    for d in DAY_GRID:
        r = np.clip(lz.wd_rate(HAM, M_CHI, E_GRID, halo=halos[d], delta_kev=delta), 0, None)
        R_roi.append(np.trapezoid(r * eff(E_GRID), E_GRID))
        R_noeff.append(np.trapezoid(r, E_GRID))
        R_gt270.append(np.trapezoid(np.where(E_GRID > 270.0, r, 0.0), E_GRID))
    R_roi, R_noeff, R_gt270 = map(np.array, (R_roi, R_noeff, R_gt270))
    r_avg = np.clip(lz.wd_rate(HAM, M_CHI, E_GRID, halo=halo_avg, delta_kev=delta), 0, None)
    # periodic cubic spline in day-of-year (period 365.25), clipped at zero
    dd = np.concatenate([DAY_GRID - 365.25, DAY_GRID, DAY_GRID + 365.25])
    rr = np.concatenate([R_roi, R_roi, R_roi])
    spl = CubicSpline(dd, rr)
    Rfun = lambda doy, spl=spl: np.clip(spl(np.mod(doy - 1.0, 365.25) + 1.0), 0, None)
    doy_fine = np.linspace(1.0, 366.0, 3651)
    Rf = Rfun(doy_fine)
    Rmean = np.trapezoid(Rf, doy_fine) / (doy_fine[-1] - doy_fine[0])
    i_jun = np.argmin(abs(DAY_GRID - 167.9)); i_dec = np.argmin(abs(DAY_GRID - 335.0))
    imax = int(np.argmax(R_roi))
    season[delta] = dict(days=DAY_GRID, R_roi=R_roi, Rfun=Rfun, Rmean=Rmean,
                         mod_fraction=(Rf.max() - Rf.min()) / (Rf.max() + Rf.min()),
                         june_dec_ratio=R_roi[i_jun] / max(R_roi[i_dec], 1e-300),
                         peak_day=float(DAY_GRID[imax]),
                         frac_year_nonzero=float(np.mean(Rf > 1e-3 * Rf.max())),
                         eff_survival_peak=float(R_roi[imax] / R_noeff[imax]),
                         frac_gt270_peak=float(R_gt270[imax] / R_noeff[imax]),
                         R_avg_halo=float(np.trapezoid(r_avg * eff(E_GRID), E_GRID)))
    s = season[delta]
    print(f"     delta={delta:5.0f}: June/Dec={s['june_dec_ratio']:8.2f}  mod.frac={s['mod_fraction']:.3f}  peak doy={s['peak_day']:.0f}"
          f"  nonzero frac of year={s['frac_year_nonzero']:.2f}  eff survival(peak)={s['eff_survival_peak']:.3f}  frac>270keV={s['frac_gt270_peak']:.3f}")

# Calendar-time PDFs over the first run and the new window (uniform livetime assumed)
def doy_of(date):
    return (date - dt.date(date.year, 1, 1)).days + 1

def window_grid(d0, d1, step_days=0.125):
    t = np.arange(0.0, (d1 - d0).days + 1e-9, step_days)
    dates_doy = np.array([doy_of(d0 + dt.timedelta(days=float(x))) + (x % 1) for x in t])
    return t, dates_doy

t_first, doy_first = window_grid(D_FIRST_START, D_FIRST_END)
t_new, doy_new = window_grid(D_NEW_START, END_DATES['2026-09-02'])
t_dec26, doy_dec26 = window_grid(D_NEW_START, END_DATES['2026-12-31'])
month_new = np.array([(D_NEW_START + dt.timedelta(days=float(x))).month for x in t_new])

inel_rows = []
for delta in DELTAS:
    s = season[delta]
    R1 = s['Rfun'](doy_first); Rn = s['Rfun'](doy_new); Rd = s['Rfun'](doy_dec26)
    # expected events in the new window if the first window (uniform livetime) yielded exactly 1.0 expected event
    ratio_new = np.trapezoid(Rn, t_new) / np.trapezoid(R1, t_first) * (M_FID / M_FID)   # same live fraction & mass
    ratio_dec26 = np.trapezoid(Rd, t_dec26) / np.trapezoid(R1, t_first)
    p_new = Rn / np.trapezoid(Rn, t_new)                                       # normalised time PDF over new window
    f_MJJ = np.trapezoid(np.where(np.isin(month_new, [5, 6, 7]), p_new, 0), t_new)
    f_NDJ = np.trapezoid(np.where(np.isin(month_new, [11, 12, 1]), p_new, 0), t_new)
    f_MJJA = np.trapezoid(np.where(np.isin(month_new, [5, 6, 7, 8]), p_new, 0), t_new)
    # flat reference fractions in the same window
    pf = np.ones_like(t_new) / (t_new[-1] - t_new[0])
    f_MJJ_flat = np.trapezoid(np.where(np.isin(month_new, [5, 6, 7]), pf, 0), t_new)
    f_NDJ_flat = np.trapezoid(np.where(np.isin(month_new, [11, 12, 1]), pf, 0), t_new)
    # KL(mod || flat) per event, and its sd
    with np.errstate(divide='ignore', invalid='ignore'):
        lr = np.where(p_new > 0, np.log(p_new / pf), 0.0)
    KL = np.trapezoid(p_new * lr, t_new); var = np.trapezoid(p_new * lr**2, t_new) - KL**2
    season[delta].update(dict(N_ratio_new=ratio_new, N_ratio_dec26=ratio_dec26, p_new=p_new, lr_new=lr,
                              f_MJJ=f_MJJ, f_NDJ=f_NDJ, f_MJJA=f_MJJA, f_MJJ_flat=f_MJJ_flat, f_NDJ_flat=f_NDJ_flat,
                              KL=KL, KL_sd=math.sqrt(max(var, 0))))
    inel_rows.append(dict(delta_keV=delta, june_dec_ratio=s['june_dec_ratio'], mod_fraction=s['mod_fraction'],
                          peak_doy=s['peak_day'], frac_year_nonzero=s['frac_year_nonzero'],
                          eff_survival_peak=s['eff_survival_peak'], frac_gt270_peak=s['frac_gt270_peak'],
                          N_new_per_first_event=ratio_new, N_new_over_k=ratio_new / K, N_dec26_per_first_event=ratio_dec26,
                          f_MayJul=f_MJJ, f_NovJan=f_NDJ, f_MayAug=f_MJJA, f_MayJul_flat=f_MJJ_flat, f_NovJan_flat=f_NDJ_flat,
                          KL_nats_per_event=KL, KL_sd=math.sqrt(max(var, 0))))
    print(f"     delta={delta:5.0f}: N_new/first-event = {ratio_new:.3f} (= {ratio_new/K:.3f} x k); May-Jul {f_MJJ:.3f} (flat {f_MJJ_flat:.3f}),"
          f" Nov-Jan {f_NDJ:.3f} (flat {f_NDJ_flat:.3f}), May-Aug {f_MJJA:.3f}; KL {KL:.3f} nats/event")
with open(f'{OUT}/inelastic_seasonal.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(inel_rows[0].keys())); w.writeheader()
    for r in inel_rows: w.writerow({k: (f'{v:.5g}' if isinstance(v, float) else v) for k, v in r.items()})
# comparison with P006 (rate_vs_doy CSVs, 1000 GeV O1, deltas 300/350/380)
cmp = {}
for delta in (300.0, 350.0, 380.0):
    try:
        arr = np.loadtxt(f'output/work/P006/rate_vs_doy_O1_d{int(delta)}_m1000.csv', delimiter=',', skiprows=1)
        d6, r6 = arr[:, 0], arr[:, 1]
        r20 = np.array([season[delta]['Rfun'](d) for d in d6])
        sc = np.trapezoid(r6, d6) / np.trapezoid(r20, d6)
        cmp[delta] = dict(max_rel_dev=float(np.max(np.abs(r20 * sc - r6) / max(r6.max(), 1e-300))),
                          june_dec_P006=float(r6[np.argmin(abs(d6 - 167.89))] / max(r6[np.argmin(abs(d6 - 335))], 1e-300)))
    except Exception as e:
        cmp[delta] = str(e)
RES['P006_comparison'] = cmp
print("     shape check vs P006 rate_vs_doy:", {k: (v if isinstance(v, str) else f"maxdev {v['max_rel_dev']:.3f}, J/D P006 {v['june_dec_P006']:.1f}") for k, v in cmp.items()})

# Seasonal visibility: toy MC for N events; LR = prod p_mod(t_i)/p_flat(t_i) over the new window
def sample_times(p, t, n, size):
    cdf = np.cumsum(np.concatenate([[0], 0.5 * (p[1:] + p[:-1]) * np.diff(t)])); cdf /= cdf[-1]
    u = rng.random((size, n))
    return np.interp(u, cdf, t)
NTOY = 40000
vis_rows = []
for delta in DELTAS:
    s = season[delta]; lr_t = s['lr_new']; p = s['p_new']
    for n in (1, 2, 3, 5):
        tm = sample_times(p, t_new, n, NTOY); tf = rng.random((NTOY, n)) * t_new[-1]
        lnLR_mod = np.interp(tm, t_new, lr_t).sum(axis=1)
        lnLR_flat = np.interp(tf, t_new, lr_t).sum(axis=1)
        # zero-rate seasons give lnLR -> -inf under flat draws (interp of large negative); treat as -inf
        row = dict(delta_keV=delta, N=n, E_lnLR_mod=float(np.mean(lnLR_mod)),
                   P_LR_gt10_mod=float(np.mean(lnLR_mod > math.log(10))), P_LR_gt10_flat=float(np.mean(lnLR_flat > math.log(10))),
                   P_LR_gt3_mod=float(np.mean(lnLR_mod > math.log(3))), P_LR_gt3_flat=float(np.mean(lnLR_flat > math.log(3))),
                   P_all_MayAug_mod=float(np.mean(np.isin(np.array([(D_NEW_START + dt.timedelta(days=float(x))).month for x in tm.ravel()]).reshape(tm.shape), [5, 6, 7, 8]).all(axis=1))),
                   P_all_MayAug_flat=float(np.mean(np.isin(np.array([(D_NEW_START + dt.timedelta(days=float(x))).month for x in tf.ravel()]).reshape(tf.shape), [5, 6, 7, 8]).all(axis=1))))
        # median LR observed under the flat hypothesis when all N happen to be in May-Aug etc. not needed
        vis_rows.append(row)
with open(f'{OUT}/seasonal_visibility.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(vis_rows[0].keys())); w.writeheader()
    for r in vis_rows: w.writerow({k: (f'{v:.4g}' if isinstance(v, float) else v) for k, v in r.items()})
for r in vis_rows:
    if r['N'] in (2, 3, 5):
        print(f"     visibility delta={r['delta_keV']:.0f} N={r['N']}: E[lnLR]={r['E_lnLR_mod']:.2f}; P(LR>10|mod)={r['P_LR_gt10_mod']:.3f}, "
              f"P(LR>10|flat)={r['P_LR_gt10_flat']:.3f}; P(all May-Aug|mod)={r['P_all_MayAug_mod']:.3f} vs flat {r['P_all_MayAug_flat']:.3f}")
RES['seasonal_visibility'] = vis_rows

# ----------------------------------------------------------------------------------------------
# 4. Decision table (P001 framework) and frequentist significance of the combined sample
# ----------------------------------------------------------------------------------------------
PI_DM, PI_U, L_U = 0.01, 0.10, 0.10
MU_U = -math.log(1 - L_U)               # 0.1054: P(>=1 | mu_U) = L_U in the first exposure (P001 Sec. 2.5)
S_MAX = 10.0                             # flat signal prior on [0, 10] (P001 central)

def L_DM(n2, k, b):
    f = lambda s: stats.poisson.pmf(1, s + b) * stats.poisson.pmf(n2, k * (s + b)) / S_MAX
    return integrate.quad(f, 0, S_MAX, limit=200)[0]
def L_K(n2, k, b): return stats.poisson.pmf(1, b) * stats.poisson.pmf(n2, k * b)
def L_Ubk(n2, k, b): return stats.poisson.pmf(1, b + MU_U) * stats.poisson.pmf(n2, k * (b + MU_U))

def posterior(n2, k, b, pi_dm=PI_DM, pi_u=PI_U):
    w = np.array([pi_dm * L_DM(n2, k, b), pi_u * L_Ubk(n2, k, b), (1 - pi_dm - pi_u) * L_K(n2, k, b)])
    return w / w.sum(), w

# sanity: reproduce P001's "now" posterior (n2 with k -> 0: P(n2=0|0)=1)
post_now, _ = posterior(0, 1e-9, B_NB)
print(f"[4] P001 check: P(DM|event) now = {post_now[0]:.3f} (P001: 0.090); P(U)={post_now[1]:.3f}, P(K)={post_now[2]:.3f}")
p1_check = [posterior(n, 1.0, B_NB)[0][0] for n in (0, 1, 2)]
print(f"    P001 check next equal exposure 0/1/2 events: {p1_check[0]:.3f}/{p1_check[1]:.3f}/{p1_check[2]:.3f} (P001: 0.028/0.217/0.797)")
RES['P001_checks'] = dict(now=list(post_now), k1=p1_check)

# Bayes factor DM vs K for the second exposure alone (given the first): ratio of predictive probabilities
def bf_second(n2, k, b):
    # posterior over s from event 1 (flat prior on [0,S_MAX]) -> predictive for n2; vs K predictive
    num = L_DM(n2, k, b) / L_DM(0, 0.0, b)      # = E_post[P(n2 | k(s+b))]
    den = stats.poisson.pmf(n2, k * b)
    return num / den

def Z_combined(n_tot, mu_b):
    p = stats.poisson.sf(n_tot - 1, mu_b)
    return p, float(stats.norm.isf(p)) if p > 0 else float('inf')

dec_rows = []
for tag, k in (('2026-09-02', K), ('2026-12-31', K_DEC26), ('2027-12-31', K_DEC27)):
    for n2 in (0, 1, 2, 3, 4):
        row = dict(end=tag, k=round(k, 3), N_new=n2)
        for bname, b in (('2e-4', B_NB), ('1e-3', B_EFF), ('0.0106', B_PANEL)):
            post, _ = posterior(n2, k, b)
            row[f'P_DM|b={bname}'] = post[0]; row[f'P_U|b={bname}'] = post[1]; row[f'P_K|b={bname}'] = post[2]
            row[f'BF2_DMvsK|b={bname}'] = bf_second(n2, k, b)
            p, Z = Z_combined(1 + n2, b * (1 + k))
            row[f'p_comb|b={bname}'] = p; row[f'Z_comb|b={bname}'] = Z
            # new exposure alone (no LEE: pre-registered region)
            p2, Z2 = Z_combined(n2, b * k) if n2 > 0 else (1.0, 0.0)
            row[f'Z_new_only|b={bname}'] = Z2
        # posterior for the pi_DM = 0.05 variant (b=2e-4)
        row['P_DM|b=2e-4,piDM=0.05'] = posterior(n2, k, B_NB, pi_dm=0.05)[0][0]
        dec_rows.append(row)
with open(f'{OUT}/decision_table.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(dec_rows[0].keys())); w.writeheader()
    for r in dec_rows: w.writerow({kk: (f'{v:.4g}' if isinstance(v, float) else v) for kk, v in r.items()})
print("    decision table (end 2026-09-02):")
for r in dec_rows:
    if r['end'] == '2026-09-02':
        print(f"     N={r['N_new']}: P(DM)={r['P_DM|b=2e-4']:.3f}/{r['P_DM|b=1e-3']:.3f} (b=2e-4/1e-3), P(U)={r['P_U|b=2e-4']:.3f}, "
              f"BF2(DM:K)={r['BF2_DMvsK|b=2e-4']:.3g}; Z_comb = {r['Z_comb|b=2e-4']:.2f}/{r['Z_comb|b=1e-3']:.2f}/{r['Z_comb|b=0.0106']:.2f} sigma "
              f"(b=2e-4/1e-3/0.0106); Z_new_only(1e-3) = {r['Z_new_only|b=1e-3']:.2f}")
RES['decision_table'] = dec_rows

# smallest N_new giving >= 5 sigma local for the combined sample
five = {}
for bname, b in (('2e-4', B_NB), ('1e-3', B_EFF), ('0.0106', B_PANEL)):
    for tag, k in (('2026-09-02', K), ('2027-12-31', K_DEC27)):
        n = 0
        while Z_combined(1 + n, b * (1 + k))[1] < 5.0: n += 1
        n_alone = 1
        while Z_combined(n_alone, b * k)[1] < 5.0: n_alone += 1
        five[f'b={bname}|{tag}'] = dict(N_new_for_5sigma_combined=n, N_new_for_5sigma_new_only=n_alone)
RES['five_sigma'] = five
print("    N_new for 5 sigma local:", five)

# N = 0: upper limits on s (per 2.84 t yr) from the new exposure alone and combined with the first event
ul_rows = []
for tag, k in (('2026-09-02', K), ('2026-12-31', K_DEC26), ('2027-12-31', K_DEC27)):
    ul_new = -math.log(0.10) / k                        # classical 90% CL UL, N=0, b ~ 0
    # combined likelihood: L(s) ~ (s+b) e^{-(s+b)(1+k)}; flat prior on s -> posterior; 90% Bayesian UL
    lam = 1.0 + k
    post_cdf = lambda s: special.gammainc(2, lam * (s + B_NB)) - special.gammainc(2, lam * B_NB)
    norm = 1 - special.gammainc(2, lam * B_NB)
    ul_comb = optimize.brentq(lambda s: post_cdf(s) / norm - 0.90, 1e-6, 50.0)
    # profile-likelihood combined: -2 dlnL = 2.71 one-sided (Wilks) relative to s_hat = 1/lam - b
    s_hat = max(1.0 / lam - B_NB, 0.0)
    lnL = lambda s: math.log(s + B_NB) - (s + B_NB) * lam
    ul_plr = optimize.brentq(lambda s: 2 * (lnL(s_hat) - lnL(s)) - 2.706, s_hat + 1e-9, 50.0)
    frac_excl = np.clip((S_BF + S_UP - ul_new) / (S_UP + S_DN), 0, 1)
    ul_rows.append(dict(end=tag, k=k, UL90_new_only=ul_new, UL90_combined_bayes=ul_comb, UL90_combined_PLR=ul_plr,
                        s_hat_combined=s_hat, frac_of_TableI_interval_excluded_new_only=float(frac_excl),
                        best_fit_excluded_new_only=bool(ul_new < S_BF), P_N0_given_s1=math.exp(-(S_BF + B_NB) * k),
                        P_N0_PoissonGamma=float(stats.nbinom.pmf(0, 2.0, 1.0 / (1.0 + k)))))
    print(f"    N=0 by {tag} (k={k:.2f}): UL90 new-only s<{ul_new:.2f}; combined Bayes {ul_comb:.2f}, PLR {ul_plr:.2f} (s_hat={s_hat:.2f});"
          f" fraction of [0.3,2.4] excluded {frac_excl:.2f}; P(N=0|s=1)={ul_rows[-1]['P_N0_given_s1']:.3f}")
with open(f'{OUT}/N0_upper_limits.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(ul_rows[0].keys())); w.writeheader()
    for r in ul_rows: w.writerow({kk: (f'{v:.4g}' if isinstance(v, float) else v) for kk, v in r.items()})
RES['N0_upper_limits'] = ul_rows

# ----------------------------------------------------------------------------------------------
# 5. P001 corpus-prior mixture predictive and forecast summary
# ----------------------------------------------------------------------------------------------
def mixture_predictive(k, b=B_NB, nmax=6):
    w = posterior(0, 1e-9, b)[0]                       # posterior weights after event 1: DM / U / K
    n = np.arange(0, nmax + 1)
    pDM = np.array([L_DM(int(nn), k, b) / L_DM(0, 0.0, b) for nn in n])
    pU = stats.poisson.pmf(n, k * (b + MU_U)); pK = stats.poisson.pmf(n, k * b)
    p = w[0] * pDM + w[1] * pU + w[2] * pK
    return w, p, pDM, pU, pK
w1, pmix1, *_ = mixture_predictive(1.0)
print(f"[5] mixture check k=1: weights DM/U/K = {w1[0]:.3f}/{w1[1]:.3f}/{w1[2]:.3f}; P0={pmix1[0]:.3f} P1={pmix1[1]:.3f} P>=2={1-pmix1[:2].sum():.3f} (P001: 0.84/0.11/0.05)")
mix = {}
for tag, k in (('2026-09-02', K), ('2026-12-31', K_DEC26), ('2027-12-31', K_DEC27)):
    w, p, pDM, pU, pK = mixture_predictive(k)
    mix[tag] = dict(k=k, weights=list(w), P=list(p), P_ge2=float(1 - p[:2].sum()), P_ge4=float(1 - p[:4].sum()),
                    P_DM_component=list(pDM), P_U_component=list(pU), mean=float(np.sum(np.arange(len(p)) * p) + (1 - p.sum()) * (len(p))))
    print(f"    mixture k={k:.2f} ({tag}): P0={p[0]:.3f} P1={p[1]:.3f} P2={p[2]:.3f} P3={p[3]:.3f} P>=4={1-p[:4].sum():.3f}")
RES['mixture_predictive'] = dict(k1_check=dict(weights=list(w1), P=list(pmix1)), **mix)

# Higgsino (P007) cross-check: 4-5 events in the next 1000 live days at delta(N=1) -> per 2.84 t yr basis
live_new = RES['exposure']['cal_days_to_2026_09_02'] * LIVE_FRAC
RES['higgsino_P007_scaled'] = dict(live_days_new=live_new, N_from_P007_1000d=[4 * live_new / 1000, 5.2 * live_new / 1000],
                                    N_from_our_366=season[366.0]['N_ratio_new'])
print(f"    Higgsino: P007's 4-5.2 events/1000 live d -> {4*live_new/1000:.1f}-{5.2*live_new/1000:.1f} in {live_new:.0f} live d; our delta=366 seasonal ratio gives {season[366.0]['N_ratio_new']:.2f}")

forecast = dict(
    k=K, E_new_tyr=E_NEW,
    N_bestfit=(S_BF) * K, N_interval=[(S_BF - S_DN) * K, (S_BF + S_UP) * K],
    P_N0_bestfit=math.exp(-S_BF * K), P_N0_PG=pred_rows['Poisson-Gamma flat prior (Gamma(2,1))'][0],
    P_Nge1_bkg_panel=bk['panel_0.0106|2026-09-02']['P_ge1'], P_Nge1_bkg_nb=bk['nb_2e-4|2026-09-02']['P_ge1'],
    P_Nge2_bestfit=float(stats.poisson.sf(1, S_BF * K)),
    mixture_P0=mix['2026-09-02']['P'][0], mixture_P1=mix['2026-09-02']['P'][1], mixture_Pge2=mix['2026-09-02']['P_ge2'])
RES['forecast'] = forecast
print(f"    FORECAST (to 2 Sep 2026, k={K:.2f}): N = {forecast['N_bestfit']:.2f} [{forecast['N_interval'][0]:.2f}, {forecast['N_interval'][1]:.2f}];"
      f" P(N=0|s=1) = {forecast['P_N0_bestfit']:.3f}; P(N=0|PG) = {forecast['P_N0_PG']:.3f}; P(N>=1|bkg panel) = {forecast['P_Nge1_bkg_panel']:.4f};"
      f" mixture P0/P1/P>=2 = {forecast['mixture_P0']:.2f}/{forecast['mixture_P1']:.2f}/{forecast['mixture_Pge2']:.2f}")

# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#4a3aa7']   # fixed categorical order
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})

# Fig 1: predictive distributions
fig, ax = plt.subplots(figsize=(7.2, 3.8))
labels = ['0', '1', '2', '3', '>=4']
series = [('background, whole S1c>500 panel (0.0106 x k)', pred_rows['bkg_only_panel'], C[3]),
          ('L10 best fit s = 1.0 (Poisson, mean %.2f)' % (S_BF * K), pred_rows['s=1.0 plug-in'], C[0]),
          ('L10 with uncertainty (Poisson-Gamma, mean %.2f)' % (2 * K), pred_rows['Poisson-Gamma flat prior (Gamma(2,1))'], C[1]),
          ('P001 corpus-prior mixture', list(mix['2026-09-02']['P'][:4]) + [mix['2026-09-02']['P_ge4']], C[2])]
x = np.arange(5); wbar = 0.2
for i, (lab, p, col) in enumerate(series):
    ax.bar(x + (i - 1.5) * wbar, p, width=wbar - 0.02, color=col, label=lab, edgecolor='white', linewidth=1)
    for xi, pi in zip(x, p):
        if pi > 0.02: ax.text(xi + (i - 1.5) * wbar, pi + 0.01, f'{pi:.2f}', ha='center', va='bottom', fontsize=6.5, color='#52514e')
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_xlabel('N high-energy NR-band events, 1 Apr 2024 - 2 Sep 2026 (%.2f t yr, k = %.2f)' % (E_NEW, K))
ax.set_ylabel('probability'); ax.set_ylim(0, 1.0); ax.grid(axis='y', color='#e5e5e5', lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=7.5, loc='upper right')
ax.set_title('Predicted event count in the untouched LZ exposure', loc='left', fontsize=10)
fig.tight_layout(); fig.savefig(f'{FIG}/predictive_distributions.png', dpi=180); plt.close(fig)

# Fig 2: seasonal PDF over calendar time (new window to Dec 2026)
fig, ax = plt.subplots(figsize=(7.6, 3.6))
dates = [D_NEW_START + dt.timedelta(days=float(x)) for x in t_dec26]
for i, delta in enumerate(DELTAS):
    Rd = season[delta]['Rfun'](doy_dec26); pd_ = Rd / np.trapezoid(Rd, t_dec26) * (t_dec26[-1] - t_dec26[0])   # p_mod(t)/p_flat(t)
    season[delta]['peak_over_flat_dec26'] = float(pd_.max())
    jd = season[delta]["june_dec_ratio"]
    jd_lab = f'June/Dec = {jd:.0f}' if jd < 1e3 else 'zero rate Oct-Mar'
    ax.plot(dates, pd_, color=C[i], lw=2, label=f'delta = {delta:.0f} keV ({jd_lab})')
ax.axhline(1.0, color='#9a9a9a', lw=1, ls='--'); ax.text(dates[5], 1.05, 'flat (background)', fontsize=7.5, color='#52514e')
for yr in (2024, 2025, 2026):
    ax.axvspan(dt.date(yr, 5, 1), dt.date(yr, 8, 1), color='#f0f0ee', zorder=0)
ax.axvline(END_DATES['2026-09-02'], color='#0b0b0b', lw=0.8); ax.text(END_DATES['2026-09-02'], ax.get_ylim()[1] * 0.02, ' 2 Sep 2026', fontsize=7.5)
ax.set_ylabel('time PDF (relative to flat)'); ax.set_xlabel('calendar date (shaded: May-July)')
ax.set_title('Expected arrival-time density of inelastic-DM events, O1, m = 1000 GeV (WimPyDD, LZ efficiency)', loc='left', fontsize=10)
ax.legend(frameon=False, fontsize=7.5, ncol=2, loc='upper left'); ax.grid(axis='y', color='#e5e5e5', lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f'{FIG}/seasonal_pdf.png', dpi=180); plt.close(fig)

# Fig 3: posterior P(DM) vs N_new
fig, ax = plt.subplots(figsize=(5.2, 3.4))
Ns = [0, 1, 2, 3, 4]
for i, (bname, col) in enumerate((('2e-4', C[0]), ('1e-3', C[1]), ('0.0106', C[2]))):
    y = [r[f'P_DM|b={bname}'] for r in dec_rows if r['end'] == '2026-09-02']
    ax.plot(Ns, y, '-o', color=col, ms=5, lw=2, label=f'b = {bname} per 2.84 t yr')
ax.axhline(0.5, color='#9a9a9a', lw=1, ls='--'); ax.set_ylim(0, 1)
ax.set_xlabel('N new events by 2 Sep 2026'); ax.set_ylabel('P(DM | 1 + N events)'); ax.set_xticks(Ns)
ax.set_title('P001 framework: pi_DM = 0.01, pi_U = 0.1, L_U = 0.1', loc='left', fontsize=10)
ax.legend(frameon=False, fontsize=7.5); ax.grid(axis='y', color='#e5e5e5', lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f'{FIG}/posterior_vs_N.png', dpi=180); plt.close(fig)

# save JSON (strip non-serialisable)
def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items() if not callable(v) and k not in ('Rfun', 'p_new', 'lr_new', 'days', 'R_roi')}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.floating, np.integer)): return o.item()
    return o
RES['seasonal'] = {str(int(d)): clean({k: v for k, v in season[d].items() if k not in ('Rfun', 'p_new', 'lr_new', 'days', 'R_roi')}) for d in DELTAS}
with open(f'{OUT}/P020_results.json', 'w') as fh:
    json.dump(clean(RES), fh, indent=1, default=str)
print("done; results in", OUT)
