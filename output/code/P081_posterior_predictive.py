"""
P081 - Posterior predictive counts for LZ's next release: P(0), P(1), P(>=2) high-energy events
       under the corpus posterior.

Run from the simulation root:  .venv/bin/python output/code/P081_posterior_predictive.py

Hypotheses (mixture weights from P061's hyper-prior Monte Carlo):
  DM : P027's model-marginalised posterior. For every distinguishable spectrum m with posterior weight
       w_m (P027 'post_uniform'; variant 'post_classes'), the rate posterior p(s | m, data) is rebuilt with
       P027's exact joint likelihood  L(s)/L0 = e^{-s} (1 + rho_m s/b_H) W_m(s/f_hi)  (one 200-270 keV
       event on b_H = 5.695e-4, the empty 125-200 keV bin, the digitised low-energy Fig. 5 bins with the
       theta nuisance) under log-uniform (baseline), flat and Jeffreys priors on s = expected 200-270 keV
       NR-band events per 2.84 t yr.  Region ratios (55-200 keV, 270-420 keV with a 1000 phd edge) come from
       P027's spectra cache and P038's edge parameters; astrophysical spread from P068 (a1 bands, delta_max
       band as an effective delta shift); seasonal factor of the release window from P034's time PDFs.
  B  : modelled background (P016 b_H; P038 extension MSSI; P016 b_M; estimated 55-125 keV leakage).
  U  : unmodelled background: one-off (predicts B only) or steady (log-uniform rate -> Exponential(1)
       posterior per 2.84 t yr, localised in the 200-270 keV band), split f_trans = 0.5 (P061).
Outputs: output/work/P081/*.csv, *.json, figures/*.png, run_log.txt
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from scipy import stats, special
from scipy.special import logsumexp, gammaln
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, 'output/code')
from common import lzcommon as lz  # noqa: F401  (LZ dict used for exposure)

T0 = time.time()
OUT = 'output/work/P081'; FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)
LOG = open(os.path.join(OUT, 'run_log.txt'), 'w')
R = {}
rng = np.random.default_rng(81)


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s); LOG.write(s + '\n'); LOG.flush()


# Okabe-Ito palette (colour-blind safe; corpus convention)
C = dict(dm='#0072B2', mix='#E69F00', bkg='#009E73', u='#CC79A7', grey='#7F7F7F', black='#000000', v='#D55E00', sky='#56B4E9')

EXPO_FIRST = 2.84                      # t yr, LZ WS2024 (paper)
EXPOSURES = [1.0, 2.8, 5.6, 10.0]      # t yr, assumed next-release exposures
KS = [E / EXPO_FIRST for E in EXPOSURES]
K_P027 = 6.76 / EXPO_FIRST             # P020/P027/P069 'untouched' exposure, for cross-checks

# ------------------------------------------------------------------------------------------------
# 0. Inputs
# ------------------------------------------------------------------------------------------------
P016 = json.load(open('output/work/P016/P016_results.json'))
P027 = json.load(open('output/work/P027/P027_results.json'))
M = pd.read_csv('output/work/P027/P027_models.csv')
SPEC = np.load('output/work/P027/P027_spectra.npz', allow_pickle=True)
SPEC_NAMES = list(SPEC['names']); SPEC_RATES = SPEC['rates']; E_WD = SPEC['E_WD']
P008S = np.load('output/work/P008/spectra_cache.npz', allow_pickle=True)          # elastic spectra P027 took from P008 (2-400 keV)
P8_NAMES = [str(n) for n in P008S['names']]; P8_RATES = P008S['rates']; E_WD8 = P008S['E_WD']


def get_spectrum(name):
    if name in SPEC_NAMES:
        return E_WD, SPEC_RATES[SPEC_NAMES.index(name)], 'P027'
    if name in P8_NAMES:
        return E_WD8, P8_RATES[P8_NAMES.index(name)], 'P008'
    return None, None, None
P038 = json.load(open('output/work/P038/P038_results.json'))
P038_ACC = pd.read_csv('output/work/P038/P038_acceptance_vs_delta.csv')
P038_FULL = np.load('output/work/P038/spectra_s_1000_full.npz', allow_pickle=True)
P034_T = pd.read_csv('output/work/P034/time_pdfs.csv')
P034_S = pd.read_csv('output/work/P034/time_pdf_summary.csv')
P069_H = pd.read_csv('output/work/P069/P069_hidden_events.csv')
P027_PRED = pd.read_csv('output/work/P027/P027_posterior_predictive.csv')

# --- P027 likelihood ingredients (identical construction to P027_bayes_models.py) ---
dig = P016['fig5_top_digitised']
EDGES = np.array(dig['sigma_bin_low'] + [8.0])
b_top = np.array(dig['total']); n_top = np.array(dig['data_counts'])
WIN_HI = 2.0
sel = EDGES[:-1] < WIN_HI - 1e-9
B_LO, N_LO = b_top[sel], n_top[sel]
G = stats.norm.cdf(EDGES[1:][sel]) - stats.norm.cdf(EDGES[:-1][sel])
B_H = P016['baseline_settings']['b_H']          # 5.695e-4 per 2.84 t yr, 200-270 keV NR band
B_M, N_M_OBS = 0.02, 0
SIG_THETA = 0.3
TH = np.linspace(max(0.02, 1 - 4 * SIG_THETA), 1 + 4 * SIG_THETA, 41)
LNW_TH = stats.norm.logpdf(TH, 1, SIG_THETA); LNW_TH -= logsumexp(LNW_TH)
LN_L0 = logsumexp(LNW_TH + np.sum(N_LO[None, :] * np.log(TH[:, None] * B_LO[None, :]) - TH[:, None] * B_LO[None, :], axis=1)) - B_M
assert abs(B_H - P027['likelihood_inputs']['b_H']) < 1e-12
log(f'[0] P027 likelihood rebuilt: b_H={B_H:.4e}, {sel.sum()} low-E bins, b_M={B_M}')


def lnLR_mu(mu, f_lo, f_L2, f_M, f_hi, rho):
    """ln L(mu)/L(0) for total-ROI signal mu (P027 eq.); returns array over mu."""
    mu = np.atleast_1d(np.asarray(mu, float))
    f_top = f_lo + f_L2
    lam = TH[None, :, None] * B_LO[None, None, :] + mu[:, None, None] * f_top * G[None, None, :]
    ll = np.sum(N_LO * np.log(lam) - lam, axis=2) - (B_M + mu[:, None] * f_M)
    lnW = logsumexp(LNW_TH[None, :] + ll, axis=1) - LN_L0
    s = mu * f_hi
    return -s + np.log1p(rho * s / B_H) + lnW


# --- efficiencies (P027/P003 600 phd edge; P038 1000 phd edge) ---
SIG_LO, SIG_HI, EFF0 = 3.4, 8.0, 0.96
E50_1000, SIG_1000 = P038['edges']['1000']['E50_keV'], P038['edges']['1000']['sigma_erf_keV']   # 423.2, 14.5 keV
E50_600_P038 = P038['edges']['600']['E50_keV']


def eff600(E):
    return EFF0 * 0.5 * (1 + special.erf((E - 5.4) / (np.sqrt(2) * SIG_LO))) * 0.5 * (1 - special.erf((E - 269.9) / (np.sqrt(2) * SIG_HI)))


def eff1000(E):
    return EFF0 * 0.5 * (1 + special.erf((E - 5.4) / (np.sqrt(2) * SIG_LO))) * 0.5 * (1 - special.erf((E - E50_1000) / (np.sqrt(2) * SIG_1000)))


E_FINE = np.arange(1.0, 400.001, 0.5)


def interp_spectrum(E, dR, Efine=E_FINE):
    dR = np.asarray(dR, float); ok = dR > 0
    if ok.sum() < 2:
        return np.zeros_like(Efine)
    f = np.interp(Efine, E[ok], np.log(dR[ok]), left=-np.inf, right=-np.inf)
    out = np.exp(f)
    inside = (Efine >= E[ok].min()) & (Efine <= E[ok].max())
    return np.where(inside, out, 0.0)


def region_ratios(E, dR):
    """per 200-270 keV (600 phd) event: extension X = accepted by 1000 phd edge but not by 600 phd edge (all E)."""
    Ef = np.arange(1.0, E.max() + 1e-9, 0.5)
    d = interp_spectrum(E, dR, Ef)
    m_hi = (Ef >= 200) & (Ef <= 270)
    hi = np.trapezoid((d * eff600(Ef))[m_hi], Ef[m_hi])
    ext = np.trapezoid(d * np.clip(eff1000(Ef) - eff600(Ef), 0, None), Ef)
    return hi, ext


# ------------------------------------------------------------------------------------------------
# 1. Per-model rate posteriors p(s | m, data) for three priors on s (200-270 keV events per 2.84 t yr)
# ------------------------------------------------------------------------------------------------
S = np.geomspace(1e-3, 30.0, 700)
LNS = np.log(S)
MM = M[(M.post_uniform > 0) | (M.post_classes > 0)].copy().reset_index(drop=True)
log(f'[1] {len(MM)} models with non-zero posterior weight (uniform or class prior); weight sums '
    f'{MM.post_uniform.sum():.4f} / {MM.post_classes.sum():.4f}')

post = {}                # prior -> (n_models, len(S)) normalised posterior densities in ln s
smean, shat, s_med = {}, {}, {}
lnLR_all = np.zeros((len(MM), len(S)))
B_check = np.zeros(len(MM))
for j, r in MM.iterrows():
    f = (r.f_lo, r.f_L2, r.f_M, r.f_hi)
    lnLR_all[j] = lnLR_mu(S / r.f_hi, *f, r.rho)
    # P027's log-uniform marginal (check)
    B_check[j] = np.trapezoid(np.exp(lnLR_all[j]) / np.log(S[-1] / S[0]), LNS)
ratio = B_check / MM.B_logu_rho.values
log(f'    marginal-likelihood check vs P027 B_logu_rho: ratio median {np.median(ratio):.4f}, '
    f'range {ratio.min():.4f}-{ratio.max():.4f}')
R['validation_B_vs_P027'] = dict(median=float(np.median(ratio)), min=float(ratio.min()), max=float(ratio.max()))

# NOTE: P027's posterior weights post_uniform use B_logu (rho=1?); check which column matches
ratio2 = B_check / MM.B_logu.values
log(f'    (vs B_logu without rho: median {np.median(ratio2):.4f}; the table weights follow the column that matches 1.000)')
R['validation_B_vs_P027_norho'] = dict(median=float(np.median(ratio2)))

PRIORS = {'logu': np.zeros_like(S),                               # d(ln s): flat in ln s
          'flat': LNS + np.where(S <= 10.0, 0.0, -np.inf),        # ds = s d(ln s), uniform on [0,10]
          'jeff': 0.5 * LNS + np.where(S <= 10.0, 0.0, -np.inf)}  # ds/sqrt(s)
for name, lnpi in PRIORS.items():
    lp = lnLR_all + lnpi[None, :]
    lp -= logsumexp(lp + np.log(np.gradient(LNS))[None, :], axis=1, keepdims=True)
    dens = np.exp(lp)                                              # density in ln s, integrates to 1 with dlnS
    post[name] = dens
    w = dens * np.gradient(LNS)[None, :]
    smean[name] = (w * S[None, :]).sum(axis=1)
    cdf = np.cumsum(w, axis=1)
    s_med[name] = np.array([np.interp(0.5, cdf[j], S) for j in range(len(MM))])
shat['mle'] = S[np.argmax(lnLR_all, axis=1)]                      # maximum-likelihood s per model
MM['s_hat'] = shat['mle']
for name in PRIORS:
    MM[f's_mean_{name}'] = smean[name]; MM[f's_med_{name}'] = s_med[name]
# companion-free reference: logu -> Exponential(1): mean 1; flat -> Gamma(2,1) truncated at 10: mean ~2
ref = {}
for name in PRIORS:
    lp = -S + np.log1p(S / B_H) + PRIORS[name]
    w = np.exp(lp - logsumexp(lp + np.log(np.gradient(LNS)))) * np.gradient(LNS)
    ref[name] = float((w * S).sum())
R['companion_free_posterior_mean_s'] = ref
log(f'    companion-free posterior mean s (logu/flat/jeff): {ref["logu"]:.3f}/{ref["flat"]:.3f}/{ref["jeff"]:.3f} (MLE 1.0)')
for nm in ['L10s_m1000', 'O1v_m1000_d350', 'O4s_m1000_d350', 'O1s_m1000_d350', 'O1s_m1000_d300', 'O6s_m1000']:
    if nm in MM.name.values:
        r = MM[MM.name == nm].iloc[0]
        log(f'    {nm:16s} f_hi={r.f_hi:.3f} N_M={r.N_M:.3f} s_hat={r.s_hat:.3f} <s>_logu={r.s_mean_logu:.3f} '
            f'<s>_flat={r.s_mean_flat:.3f} w_unif={r.post_uniform:.4f}')

# ------------------------------------------------------------------------------------------------
# 2. Region ratios per model: g_L2M = (f_L2+f_M)/f_hi (55-200 keV), g_X (270-420 keV, 1000 phd edge)
# ------------------------------------------------------------------------------------------------
gX = np.zeros(len(MM)); src = []
for j, r in MM.iterrows():
    E_, dR_, s_ = get_spectrum(r['name'])
    src.append(s_)
    if E_ is not None:
        hi, ext = region_ratios(E_, dR_)
        gX[j] = ext / hi if hi > 0 else 0.0
    else:
        gX[j] = np.nan
missing = np.isnan(gX)
log(f'[2] extension ratios computed for {(~missing).sum()} models ({src.count("P027")} from P027 cache, {src.count("P008")} from P008 cache); '
    f'{missing.sum()} without cached spectrum (weight {MM.post_uniform[missing].sum():.4f})')
# elastic truncation (caches end at 400 keV): L10 from P050's cache to 1498 keV; factor applied to all elastic models (approximation)
P050 = np.load('output/work/P050/P050_spectra_cache.npz', allow_pickle=True)
hi_f, ext_f = region_ratios(P050['E_l10'], P050['l10'])
m400 = P050['E_l10'] <= 400.0
hi_t, ext_t = region_ratios(P050['E_l10'][m400], P050['l10'][m400])
ELASTIC_TRUNC = (ext_f / hi_f) / (ext_t / hi_t)
R['L10_truncation_check_P050'] = dict(E_max=float(P050['E_l10'].max()), gX_full=float(ext_f / hi_f), gX_trunc400=float(ext_t / hi_t), factor=float(ELASTIC_TRUNC))
log(f'    L10 (P050 cache to {P050["E_l10"].max():.0f} keV): g_X full {ext_f/hi_f:.4f} vs truncated-400 {ext_t/hi_t:.4f} -> elastic truncation factor {ELASTIC_TRUNC:.3f}')

# truncation correction (P027 spectra end at 400 keV): P038 1 TeV O1s full spectra to 800 keV
P38_D = P038_FULL['deltas']; P38_E = P038_FULL['E']
corr, hX_p38, hX_p38_trunc = {}, {}, {}
for i, d in enumerate(P38_D):
    hi_f, ext_f = region_ratios(P38_E, P038_FULL['annual'][i])
    m400 = P38_E <= 400.0
    hi_t, ext_t = region_ratios(P38_E[m400], P038_FULL['annual'][i][m400])
    hX_p38[d] = ext_f / hi_f; hX_p38_trunc[d] = ext_t / hi_t; corr[d] = (ext_f / hi_f) / (ext_t / hi_t)
log('    P038-grid extension per 200-270 event h(delta) [full / truncated-400]: ' +
    ', '.join(f'{d:.0f}:{hX_p38[d]:.3f}/{hX_p38_trunc[d]:.3f}' for d in P38_D))
# cross-check with P038's own table (extra events in 600-1000 phd per ROI event / f_hi(ROI) -> per 200-270 event)
p38_ann = P038_ACC[P038_ACC.halo == 'annual'].set_index('delta_keV')
xchk = {}
for d in [300.0, 350.0, 370.0, 380.0]:
    row = p38_ann.loc[d]
    # per-ROI extra -> per accepted 200-270 keV event: P038 S_acc_600 counts the whole accepted ROI; use our spectra directly
    xchk[d] = dict(P038_extra_per_ROI_event=float(row.extra_events_600_1000_per_ROI_event), ours_per_200_270_event=float(hX_p38[d]))
R['extension_crosscheck_P038'] = xchk
log('    P038 extra events 600-1000 phd per ROI event: ' + ', '.join(f'{d:.0f}:{v["P038_extra_per_ROI_event"]:.3f}' for d, v in xchk.items()) +
    '  (ours are per 200-270 keV event; ROI->200-270 conversion multiplies by 1/f_hi)')
Dg = np.array(sorted(corr)); Cg = np.array([corr[d] for d in Dg])


def trunc_corr(delta):
    if delta <= 0:
        return 1.0
    return float(np.interp(np.clip(delta, Dg.min(), Dg.max()), Dg, Cg))


gX_corr = np.array([gX[j] * (trunc_corr(MM.delta[j]) if MM.cls[j] == 'inelastic' else ELASTIC_TRUNC) for j in range(len(MM))])
MM['g_L2M'] = (MM.f_L2 + MM.f_M) / MM.f_hi
MM['g_lo'] = MM.f_lo / MM.f_hi
MM['g_X_raw'] = gX; MM['g_X'] = gX_corr
MM['spectrum_source'] = src

# weighted summaries of the ratios
for wcol in ['post_uniform', 'post_classes']:
    w = MM[wcol].values / MM[wcol].sum()
    log(f'    <g_L2M>_{wcol}={np.nansum(w*MM.g_L2M):.3f}, <g_X>_{wcol}={np.nansum(w*np.nan_to_num(MM.g_X)):.3f}, '
        f'<f_hi>={np.sum(w*MM.f_hi):.3f}')

# ------------------------------------------------------------------------------------------------
# 3. Seasonal factor of the release window (P034 time PDFs; window starts 1 April 2024, live fraction 220/371)
# ------------------------------------------------------------------------------------------------
DOY = P034_T.doy.values
PD = {300: P034_T.p_delta300.values, 330: P034_T.p_delta330.values, 350: P034_T.p_delta350.values,
      366: P034_T.p_delta366.values, 380: P034_T.p_delta380.values}
P_FLAT = P034_T.p_flat.values[0]
DOY_START = 92            # 1 April 2024 (leap year)
CAL_PER_TYR = 371.0 / EXPO_FIRST   # 27 Mar 2023 - 1 Apr 2024 = 371 calendar days for 2.84 t yr (220 live d)


def window_factor(p, start_doy, n_days):
    idx = (start_doy - 1 + np.arange(int(round(n_days)))) % len(DOY)
    return float(p[idx].mean() / P_FLAT)


F_first = {d: window_factor(PD[d], 86, 371) for d in PD}       # 27 March 2023 = doy 86 (non-leap)
log(f'[3] first-run window (27 Mar 2023 - 1 Apr 2024) seasonal factor: ' + ', '.join(f'd{d}:{F_first[d]:.3f}' for d in PD))
F_win = {}
for E in EXPOSURES:
    nd = E * CAL_PER_TYR
    F_win[E] = {d: window_factor(PD[d], DOY_START, nd) / F_first[d] for d in PD}
    log(f'    release window {E:>4} t yr = {nd:.0f} calendar d from 1 Apr 2024: F(delta)/F_first = ' +
        ', '.join(f'd{d}:{F_win[E][d]:.3f}' for d in PD))
# P006 ROI modulation fractions for delta < 300 (scale of (F-1)); elastic a1 = 0.025 (P068 a1_L10) vs a1(300) = 0.428
MODFRAC = {0: 0.024, 100: 0.10, 200: 0.24, 300: 0.41855}


def season_factor(E, delta, cls):
    if cls != 'inelastic':
        return 1.0 + (F_win[E][300] - 1.0) * 0.0248 / 0.4284
    if delta >= 300:
        ds = np.array(sorted(F_win[E])); fs = np.array([F_win[E][d] for d in ds])
        return float(np.interp(min(delta, 380), ds, fs))
    r = np.interp(delta, list(MODFRAC), list(MODFRAC.values())) / MODFRAC[300]
    return 1.0 + (F_win[E][300] - 1.0) * r


# P068 a1 bands (68 %) -> log-normal spread factor xi on (F-1); median-normalised
A1_BAND = {300: (0.3757, 0.553, 0.8769), 350: (0.7809, 1.272, 1.737), 366: (0.9979, 1.49, 1.789), 380: (1.199, 1.602, 1.812)}


def a1_sigma(delta):
    ds = np.array(sorted(A1_BAND)); sg = np.array([0.5 * np.log(A1_BAND[d][2] / A1_BAND[d][0]) for d in ds])
    return float(np.interp(np.clip(delta, ds.min(), ds.max()), ds, sg))


# ------------------------------------------------------------------------------------------------
# 4. Monte Carlo over (model, s, astro nuisances) -> per-unit-exposure region means
# ------------------------------------------------------------------------------------------------
NMC = 400_000
SIG_DELTA_ASTRO = 25.0     # keV; P068 delta_max 68 % band +-31 keV, v_esc-dominated; used as an effective delta shift for g_X
P38_grid_d = np.array(sorted(hX_p38)); P38_grid_h = np.log(np.array([hX_p38[d] for d in P38_grid_d]))


def h_p38(delta):
    return np.exp(np.interp(np.clip(delta, P38_grid_d.min(), P38_grid_d.max()), P38_grid_d, P38_grid_h))


def draw_dm(weight_col, prior, astro=True, seed=1):
    rg = np.random.default_rng(seed)
    w = MM[weight_col].values / MM[weight_col].sum()
    j = rg.choice(len(MM), size=NMC, p=w)
    dens = post[prior]
    cdf = np.cumsum(dens * np.gradient(LNS)[None, :], axis=1); cdf /= cdf[:, -1:]
    u = rg.random(NMC)
    s = np.empty(NMC)
    for jj in np.unique(j):
        mj = j == jj
        s[mj] = np.exp(np.interp(u[mj], cdf[jj], LNS))
    delta = MM.delta.values[j].astype(float); cls = MM.cls.values[j]
    gL = MM.g_L2M.values[j]; gXm = np.nan_to_num(MM.g_X.values[j])
    inel = cls == 'inelastic'
    if astro:
        xi = np.exp(rg.normal(0, 1, NMC) * np.array([a1_sigma(d) for d in delta]))
        dsh = rg.normal(0, SIG_DELTA_ASTRO, NMC)
        big = inel & (delta >= 250)
        gXm = gXm.copy()
        gXm[big] *= h_p38(delta[big] + dsh[big]) / h_p38(delta[big])
    else:
        xi = np.ones(NMC)
    return dict(j=j, s=s, delta=delta, cls=cls, gL=gL, gX=gXm, xi=xi)


def region_means(D, E, season=True):
    """expected counts (per 2.84 t yr unit) in H (200-270), L2M (55-200), X (270-420, 1000 phd) incl. season."""
    if season:
        F = np.array([season_factor(E, d, c) for d, c in zip(D['delta'], D['cls'])])
    else:
        F = np.ones(len(D['s']))
    F = 1.0 + (F - 1.0) * D['xi']
    F = np.clip(F, 0.0, None)
    muH = D['s'] * F
    return muH, muH * D['gL'], muH * D['gX']


def pois_pmf(n, mu):
    return np.exp(n * np.log(np.clip(mu, 1e-300, None)) - mu - gammaln(n + 1))


def predictive(mu, k, nmax=6, wts=None):
    """mixture Poisson predictive over MC samples: P(N=0..nmax-1), P(N>=nmax-1)."""
    if wts is None:
        wts = np.ones_like(mu) / len(mu)
    P = np.array([(wts * pois_pmf(n, k * mu)).sum() for n in range(nmax)])
    return P


D0 = draw_dm('post_uniform', 'logu', astro=True, seed=11)
VARIANTS = {'baseline: uniform model prior, log-uniform s, astro on': D0,
            'flat prior on s': draw_dm('post_uniform', 'flat', astro=True, seed=12),
            'Jeffreys prior on s': draw_dm('post_uniform', 'jeff', astro=True, seed=13),
            'class model prior (P027)': draw_dm('post_classes', 'logu', astro=True, seed=14),
            'no astro spread': draw_dm('post_uniform', 'logu', astro=False, seed=15)}

# P038 sideband variant: reweight by probability of zero events in the 600-1000 phd NR band in the first run (importance weights)
W_P038 = np.exp(-D0['s'] * D0['gX']); W_P038 /= W_P038.sum()

# ------------------------------------------------------------------------------------------------
# 5. Backgrounds
# ------------------------------------------------------------------------------------------------
B_L2 = 0.10                 # 55-125 keV NR band +-2 sigma per 2.84 t yr: ESTIMATE (bracket 0.03-0.3), see details
B_X = 0.0027 + 1e-4          # P038 NR-band MSSI added to 1000 phd (0.0027, bracket 0.0008-0.0034) + ~1e-4 accidentals (P022)
BKG = dict(H=B_H, L2M=B_L2 + B_M, X=B_X, panel=0.0106)
BKG_H_BRACKET = (3.0e-4, 3.5e-3)   # corpus-revised: P070/P004 (wall MSSI k=0.62) low; P010 flat ER-tail extrapolation high
R['backgrounds_per_2p84_tyr'] = dict(BKG, H_bracket=BKG_H_BRACKET, L2_estimate=B_L2, M=B_M)
log(f'[5] backgrounds per 2.84 t yr: H {B_H:.2e} (bracket {BKG_H_BRACKET}), 125-200 {B_M}, 55-125 {B_L2} (estimate), X {B_X:.4f}, panel 0.0106')

# ------------------------------------------------------------------------------------------------
# 6. Unknown steady background: log-uniform rate -> Exponential(1) posterior per 2.84 t yr (Jeffreys variant Gamma(1.5))
# ------------------------------------------------------------------------------------------------


def nb_pmf(alpha, k, nmax=6):
    """Poisson-Gamma: rate ~ Gamma(alpha, 1) per unit exposure, exposure k."""
    n = np.arange(nmax)
    return np.exp(gammaln(n + alpha) - gammaln(alpha) - gammaln(n + 1) + alpha * np.log(1 / (1 + k)) + n * np.log(k / (1 + k)))


# ------------------------------------------------------------------------------------------------
# 7. P061 hyper-prior Monte Carlo for the hypothesis weights
# ------------------------------------------------------------------------------------------------
NH = 20000
HYP = dict(pi_DM=(1e-3, 0.3), pi_U=(0.01, 0.5), L_rest=(1e-3, 0.3), f_FP=(1.0, 10.0), B_DM=(6.6, 45.0))
L_ACC, B_P061 = 9.3e-3, 5.7e-4
hs = {k: np.exp(rng.uniform(np.log(a), np.log(b), NH)) for k, (a, b) in HYP.items()}
L_DM = hs['B_DM'] / hs['f_FP'] * B_P061
pi_B = 1 - hs['pi_DM'] - hs['pi_U']
wDM = hs['pi_DM'] * L_DM; wB = pi_B * B_P061; wU = hs['pi_U'] * (L_ACC + hs['L_rest'])
Z = wDM + wB + wU
P_DM, P_B, P_U = wDM / Z, wB / Z, wU / Z
q = np.quantile(P_DM, [0.16, 0.5, 0.84])
R['P061_reproduction'] = dict(median=float(q[1]), q16=float(q[0]), q84=float(q[2]), frac_gt_0p5=float((P_DM > 0.5).mean()),
                              frac_lt_0p1=float((P_DM < 0.1).mean()), P_B_median=float(np.median(P_B)), P_U_median=float(np.median(P_U)))
log(f'[7] P061 hyper-prior reproduced: P(DM) median {q[1]:.4f}, 68% {q[0]:.4f}-{q[2]:.4f}; >0.5: {(P_DM>0.5).mean():.3f}; '
    f'P(B) median {np.median(P_B):.3f}, P(U) median {np.median(P_U):.3f}   (P061: 0.015, 0.0016-0.137, 0.024)')
F_TRANS = 0.5

# ------------------------------------------------------------------------------------------------
# 8. Predictive tables
# ------------------------------------------------------------------------------------------------
NMAX = 7
rows = []
pred_store = {}
for E, k in zip(EXPOSURES, KS):
    muH, muL, muX = region_means(D0, E)
    for reg, mu, b in [('200-270 keV (600 phd ROI)', muH, BKG['H']), ('55-270 keV (above old edge)', muH + muL, BKG['H'] + BKG['L2M']),
                       ('55-420 keV (1000 phd edge)', muH + muL + muX, BKG['H'] + BKG['L2M'] + BKG['X']),
                       ('270-420 keV extension only', muX, BKG['X'])]:
        P_dm = predictive(mu, k, NMAX)
        P_dm38 = predictive(mu, k, NMAX, W_P038)
        P_b = pois_pmf(np.arange(NMAX), k * b)
        P_us = nb_pmf(1.0, k, NMAX) if reg.startswith('200-270') else None      # steady unknown localised in H
        # mixture over the hyper-prior samples
        if P_us is None:
            P_us_eff = P_b   # steady U adds nothing outside H in this bookkeeping (localised); regions containing H handled below
        if reg.startswith('200-270') or reg.startswith('55-') :
            # for regions containing H: U_steady count = NB(1,k) in H plus Poisson background elsewhere -> convolve
            b_other = b - BKG['H']
            nb = nb_pmf(1.0, k, NMAX); pb_other = pois_pmf(np.arange(NMAX), k * b_other)
            P_us_eff = np.array([sum(nb[i] * pb_other[n - i] for i in range(n + 1)) for n in range(NMAX)])
        P_mix = (P_DM[:, None] * P_dm[None, :] + (P_B + P_U * F_TRANS)[:, None] * P_b[None, :] + (P_U * (1 - F_TRANS))[:, None] * P_us_eff[None, :])
        qm = np.quantile(P_mix, [0.16, 0.5, 0.84], axis=0)
        q2 = np.quantile(1 - P_mix[:, 0] - P_mix[:, 1], [0.16, 0.5, 0.84])       # quantiles of P(>=2) itself
        P_mix_med_point = float(q[1]) * P_dm + (1 - float(q[1])) * (np.median(P_B) / (np.median(P_B) + np.median(P_U)) * P_b + np.median(P_U) / (np.median(P_B) + np.median(P_U)) * (F_TRANS * P_b + (1 - F_TRANS) * P_us_eff))
        rows.append(dict(exposure_tyr=E, k=k, region=reg, mean_DM=float((k * mu).mean()), mean_DM_median=float(np.median(k * mu)),
                         mean_B=k * b, P0_DM=P_dm[0], P1_DM=P_dm[1], Pge2_DM=1 - P_dm[0] - P_dm[1], Pge1_DM=1 - P_dm[0],
                         P0_DM_P038wt=P_dm38[0], P1_DM_P038wt=P_dm38[1], Pge2_DM_P038wt=1 - P_dm38[0] - P_dm38[1],
                         P0_B=P_b[0], P1_B=P_b[1], Pge2_B=1 - P_b[0] - P_b[1],
                         P0_Usteady=P_us_eff[0], P1_Usteady=P_us_eff[1], Pge2_Usteady=1 - P_us_eff[0] - P_us_eff[1],
                         P0_mix_med=qm[1, 0], P0_mix_16=qm[0, 0], P0_mix_84=qm[2, 0],
                         P1_mix_med=qm[1, 1], P1_mix_16=qm[0, 1], P1_mix_84=qm[2, 1],
                         Pge2_mix_med=q2[1], Pge2_mix_lo=q2[0], Pge2_mix_hi=q2[2]))
        pred_store[(E, reg)] = dict(P_dm=P_dm, P_b=P_b, P_us=P_us_eff, P_mix=P_mix)
PRED = pd.DataFrame(rows)
PRED.to_csv(os.path.join(OUT, 'P081_predictive_counts.csv'), index=False)
log('[8] predictive counts (baseline DM branch / background / P061 mixture median):')
for _, r in PRED.iterrows():
    log(f'    {r.exposure_tyr:>4} t yr  {r.region:28s} <N>_DM={r.mean_DM:.3f}  DM P0/P1/P>=2 = {r.P0_DM:.3f}/{r.P1_DM:.3f}/{r.Pge2_DM:.3f}'
        f'  B P0/P1/P>=2 = {r.P0_B:.4f}/{r.P1_B:.2e}/{r.Pge2_B:.1e}  mix P0/P1/P>=2 = {r.P0_mix_med:.3f}/{r.P1_mix_med:.3f}/{r.Pge2_mix_med:.3f}'
        f'  [P038wt DM P0={r.P0_DM_P038wt:.3f}]')

# variants (200-270 keV region)
vrows = []
for name, D in VARIANTS.items():
    for E, k in zip(EXPOSURES, KS):
        muH, muL, muX = region_means(D, E)
        P_dm = predictive(muH, k, NMAX)
        vrows.append(dict(variant=name, exposure_tyr=E, mean_H=float((k * muH).mean()), P0=P_dm[0], P1=P_dm[1], Pge2=1 - P_dm[0] - P_dm[1],
                          mean_L2M=float((k * muL).mean()), mean_X=float((k * muX).mean())))
for E, k in zip(EXPOSURES, KS):
    muH, muL, muX = region_means(D0, E)
    P_dm = predictive(muH, k, NMAX, W_P038)
    vrows.append(dict(variant='P038 empty 600-1000 phd sideband reweighting', exposure_tyr=E, mean_H=float((k * muH * W_P038).sum()), P0=P_dm[0], P1=P_dm[1],
                      Pge2=1 - P_dm[0] - P_dm[1], mean_L2M=float((k * muL * W_P038).sum()), mean_X=float((k * muX * W_P038).sum())))
    muH, muL, muX = region_means(D0, E, season=False)
    P_dm = predictive(muH, k, NMAX)
    vrows.append(dict(variant='annual-average window (no seasonal factor)', exposure_tyr=E, mean_H=float((k * muH).mean()), P0=P_dm[0], P1=P_dm[1],
                      Pge2=1 - P_dm[0] - P_dm[1], mean_L2M=float((k * muL).mean()), mean_X=float((k * muX).mean())))
    P_dmX = predictive(muX, k, NMAX); P_dmX38 = predictive(muX, k, NMAX, W_P038)
    vrows.append(dict(variant='extension 270-420 keV only, annual window', exposure_tyr=E, mean_H=float(np.median(k * muX)), P0=P_dmX[0], P1=P_dmX[1],
                      Pge2=1 - P_dmX[0] - P_dmX[1], mean_L2M=P_dmX38[0], mean_X=float((k * muX * W_P038).sum())))
VAR = pd.DataFrame(vrows); VAR.to_csv(os.path.join(OUT, 'P081_variants.csv'), index=False)
log('    variants at 2.8 t yr (200-270 keV): ' + '; '.join(f'{r.variant.split(":")[0][:28]}: <N>={r.mean_H:.3f} P0={r.P0:.3f}' for _, r in VAR[VAR.exposure_tyr == 2.8].iterrows()))

# best fit vs posterior mean
bf_rows = []
for E, k in zip(EXPOSURES, KS):
    for wcol in ['post_uniform', 'post_classes']:
        w = MM[wcol].values / MM[wcol].sum()
        Fm = np.array([season_factor(E, d, c) for d, c in zip(MM.delta, MM.cls)])
        d = dict(exposure_tyr=E, weights=wcol, N_bestfit_H=k * float(np.sum(w * MM.s_hat * Fm)))
        for p in PRIORS:
            d[f'N_postmean_H_{p}'] = k * float(np.sum(w * MM[f's_mean_{p}'] * Fm))
        d['N_bestfit_55_270'] = k * float(np.sum(w * MM.s_hat * Fm * (1 + MM.g_L2M)))
        d['N_postmean_55_270_logu'] = k * float(np.sum(w * MM.s_mean_logu * Fm * (1 + MM.g_L2M)))
        bf_rows.append(d)
BF = pd.DataFrame(bf_rows); BF.to_csv(os.path.join(OUT, 'P081_bestfit_vs_posterior_mean.csv'), index=False)
r28 = BF[(BF.exposure_tyr == 2.8) & (BF.weights == 'post_uniform')].iloc[0]
log(f'    2.8 t yr, 200-270 keV: N at best fit {r28.N_bestfit_H:.3f} vs posterior mean logu/flat/jeff '
    f'{r28.N_postmean_H_logu:.3f}/{r28.N_postmean_H_flat:.3f}/{r28.N_postmean_H_jeff:.3f}; 55-270: {r28.N_bestfit_55_270:.3f} vs {r28.N_postmean_55_270_logu:.3f}')

# ------------------------------------------------------------------------------------------------
# 9. Exposure at which P(>=1 | DM) exceeds 0.9 (and 0.5, 0.95); annual-mean windows (season factor of a long run -> 1)
# ------------------------------------------------------------------------------------------------
kgrid = np.geomspace(0.1, 60, 200)
muH0 = D0['s']; muL0 = muH0 * D0['gL']; muX0 = muH0 * D0['gX']
PGE1 = {}
for reg, mu in [('200-270 keV', muH0), ('55-270 keV', muH0 + muL0), ('55-420 keV (1000 phd)', muH0 + muL0 + muX0)]:
    PGE1[reg] = np.array([1 - np.mean(np.exp(-k * mu)) for k in kgrid])
PGE1_P038 = np.array([1 - np.sum(W_P038 * np.exp(-k * muH0)) for k in kgrid])
PGE1_flat = np.array([1 - np.mean(np.exp(-k * VARIANTS['flat prior on s']['s'])) for k in kgrid])
PGE1_bestfit = None
# best-fit (plug-in) reference: mixture over models of Poisson at s_hat
w_u = MM.post_uniform.values / MM.post_uniform.sum()
PGE1_plug = np.array([1 - np.sum(w_u * np.exp(-k * MM.s_hat.values)) for k in kgrid])
thr = {}
for reg, arr in list(PGE1.items()) + [('200-270 keV, P038 reweighted', PGE1_P038), ('200-270 keV, flat prior', PGE1_flat), ('200-270 keV, plug-in best fit', PGE1_plug)]:
    thr[reg] = {}
    for p in [0.5, 0.9, 0.95]:
        if arr.max() > p:
            kk = float(np.interp(p, arr, kgrid)); thr[reg][f'E_tyr_P{p}'] = kk * EXPO_FIRST
        else:
            thr[reg][f'E_tyr_P{p}'] = float('inf')
R['exposure_for_Pge1'] = thr
log('[9] exposure (t yr) for P(>=1 | DM) = 0.5 / 0.9 / 0.95: ' +
    '; '.join(f'{k}: {v["E_tyr_P0.5"]:.1f}/{v["E_tyr_P0.9"]:.1f}/{v["E_tyr_P0.95"]:.1f}' for k, v in thr.items()))
pd.DataFrame(dict(k=kgrid, E_tyr=kgrid * EXPO_FIRST, **{f'Pge1_{k}': v for k, v in PGE1.items()}, Pge1_P038=PGE1_P038, Pge1_flat=PGE1_flat,
                  Pge1_plugin=PGE1_plug)).to_csv(os.path.join(OUT, 'P081_Pge1_vs_exposure.csv'), index=False)

# ------------------------------------------------------------------------------------------------
# 10. Decision table: Bayes factors and posterior P(DM | event, N) for outcomes in 200-270 keV, plus energy placements
# ------------------------------------------------------------------------------------------------
dec = []
for E, k in zip(EXPOSURES, KS):
    st = pred_store[(E, '200-270 keV (600 phd ROI)')]
    P_dm, P_b, P_us = st['P_dm'], st['P_b'], st['P_us']
    outcomes = {'N=0': (P_dm[0], P_b[0], P_us[0]), 'N=1': (P_dm[1], P_b[1], P_us[1]), 'N=2': (P_dm[2], P_b[2], P_us[2]),
                'N>=2': (1 - P_dm[0] - P_dm[1], 1 - P_b[0] - P_b[1], 1 - P_us[0] - P_us[1]),
                'N>=3': (1 - P_dm[:3].sum(), 1 - P_b[:3].sum(), 1 - P_us[:3].sum())}
    for name, (pd_, pb_, pu_) in outcomes.items():
        # posterior over the hyper-prior samples
        num = P_DM * pd_; den = num + (P_B + P_U * F_TRANS) * pb_ + P_U * (1 - F_TRANS) * pu_
        pp = num / den; qq = np.quantile(pp, [0.16, 0.5, 0.84])
        # with P(DM) fixed at P061's median and optimistic 68 % edge
        def post_fixed(pdm):
            pb = (1 - pdm) * np.median(P_B) / (np.median(P_B) + np.median(P_U)); pu = 1 - pdm - pb
            return pdm * pd_ / (pdm * pd_ + (pb + pu * F_TRANS) * pb_ + pu * (1 - F_TRANS) * pu_)
        dec.append(dict(exposure_tyr=E, outcome=name, P_DM=pd_, P_B=pb_, P_Usteady=pu_, BF_DM_vs_B=pd_ / pb_, BF_DM_vs_Usteady=pd_ / pu_,
                        BF_DM_vs_U_mixed=pd_ / (F_TRANS * pb_ + (1 - F_TRANS) * pu_),
                        postDM_med=qq[1], postDM_16=qq[0], postDM_84=qq[2], postDM_at_0p015=post_fixed(0.015), postDM_at_0p137=post_fixed(0.137),
                        postDM_at_0p5=post_fixed(0.5), P_outcome_mix_med=float(np.median(den))))
DEC = pd.DataFrame(dec); DEC.to_csv(os.path.join(OUT, 'P081_decision_table.csv'), index=False)
log('[10] decision table (200-270 keV):')
for _, r in DEC.iterrows():
    log(f'    {r.exposure_tyr:>4} t yr {r.outcome:5s} P(DM/B/Us)={r.P_DM:.3f}/{r.P_B:.2e}/{r.P_Usteady:.3f} BF_DM:B={r.BF_DM_vs_B:.3g} BF_DM:Us={r.BF_DM_vs_Usteady:.2f}'
        f'  P(DM|event,N) median {r.postDM_med:.3f} (68% {r.postDM_16:.4f}-{r.postDM_84:.3f}); at P061 median/68%-edge/0.5: {r.postDM_at_0p015:.3f}/{r.postDM_at_0p137:.3f}/{r.postDM_at_0p5:.3f}')

# energy placements for exactly two events (regions L2M = 55-200, H = 200-270, X = 270-420 [1000 phd edge])
place = []
for E, k in zip(EXPOSURES, KS):
    muH, muL, muX = region_means(D0, E)
    mus = dict(L2M=muL, H=muH, X=muX); bs = dict(L2M=BKG['L2M'], H=BKG['H'], X=BKG['X'])
    for label, (nL, nH, nX) in {'both 200-270': (0, 2, 0), '200-270 + 55-200': (1, 1, 0), 'both 55-200': (2, 0, 0),
                                '200-270 + 270-420': (0, 1, 1), 'both 270-420': (0, 0, 2), '55-200 + 270-420': (1, 0, 1),
                                'one 200-270 only (N=1, nothing elsewhere)': (0, 1, 0), 'nothing anywhere 55-420': (0, 0, 0)}.items():
        nv = dict(L2M=nL, H=nH, X=nX)
        with_ext = 'X' in label or '270-420' in label or 'anywhere' in label
        regs = ['L2M', 'H', 'X'] if with_ext else ['L2M', 'H']
        pdm = np.ones(NMC); pb = 1.0
        for rg_ in regs:
            pdm *= pois_pmf(nv[rg_], k * mus[rg_]); pb *= pois_pmf(nv[rg_], k * bs[rg_])
        p_dm = float(pdm.mean()); p_dm38 = float((pdm * W_P038).sum())
        # steady unknown localised in H: NB in H, Poisson elsewhere
        pu = nb_pmf(1.0, k, 8)[nv['H']] * np.prod([pois_pmf(nv[rg_], k * bs[rg_]) for rg_ in regs if rg_ != 'H'])
        num = P_DM * p_dm; den = num + (P_B + P_U * F_TRANS) * pb + P_U * (1 - F_TRANS) * pu
        place.append(dict(exposure_tyr=E, placement=label, regions_examined=' + '.join(regs), P_DM=p_dm, P_DM_P038wt=p_dm38, P_B=pb, P_Usteady=float(pu),
                          BF_DM_vs_B=p_dm / pb, BF_DM_vs_Usteady=p_dm / pu, postDM_med=float(np.median(num / den)),
                          postDM_16=float(np.quantile(num / den, 0.16)), postDM_84=float(np.quantile(num / den, 0.84))))
PL = pd.DataFrame(place); PL.to_csv(os.path.join(OUT, 'P081_energy_placement.csv'), index=False)
log('    two-event energy placements at 2.8 t yr:')
for _, r in PL[PL.exposure_tyr == 2.8].iterrows():
    log(f'      {r.placement:42s} P(DM)={r.P_DM:.4f} [P038wt {r.P_DM_P038wt:.4f}] P(B)={r.P_B:.2e} P(Us)={r.P_Usteady:.2e} BF_DM:B={r.BF_DM_vs_B:.3g} BF_DM:Us={r.BF_DM_vs_Usteady:.3g} P(DM|..)={r.postDM_med:.3f}')

# ------------------------------------------------------------------------------------------------
# 11. Cross-checks: P027 predictive (6.76 t yr), P069 per-model LZ-untouched counts, P020 L10, P034 a1
# ------------------------------------------------------------------------------------------------
xc = {}
D_noast = VARIANTS['no astro spread']
muH_x = D_noast['s']
P27 = predictive(muH_x, K_P027, NMAX)
xc['P027_6p76_tyr_200_270_no_season'] = dict(ours_mean=float(K_P027 * muH_x.mean()), ours_P0=float(P27[0]), ours_Pge2=float(1 - P27[0] - P27[1]),
                                            P027_mean=0.92, P027_P0=0.58, P027_Pge2=0.20)
# P027's own predictive table (uniform weights, logu, rho): weighted sum
pp = P027_PRED[(P027_PRED.prior == 'logu') & (np.abs(P027_PRED.rho - 1.0) > 1e-6)]
if len(pp):
    xc['P027_table_recomputed'] = dict(mean=float((pp.posterior_weight_uniform * pp.N_mean).sum() / pp.posterior_weight_uniform.sum()),
                                       P0=float((pp.posterior_weight_uniform * pp.P0).sum() / pp.posterior_weight_uniform.sum()))
log(f'[11] cross-check P027 (6.76 t yr, 200-270 keV, no season/astro): ours <N>={xc["P027_6p76_tyr_200_270_no_season"]["ours_mean"]:.3f} '
    f'P0={P27[0]:.3f} P>=2={1-P27[0]-P27[1]:.3f}  vs P027 0.92/0.58/0.20')
# P069: LZ untouched 6.76 t yr best-fit 200-270 keV counts per model (true energy, LZ-like roll-off), our s_hat*k*f-based equivalents
p69 = P069_H[P069_H.dataset == 'LZ untouched since Apr 2024'].set_index('model')
cmp = {}
for nm, key in [('L10s_m1000', 'L10'), ('O1s_m1000_d300', 'd300'), ('O1s_m1000_d350', 'd350')]:
    r = MM[MM.name == nm].iloc[0]
    # P069 normalises 1.0 event in the whole ROI; ours: s_hat (200-270 keV) with f_hi -> ROI-normalised 200-270 count = f_hi * k
    cmp[nm] = dict(P069_200_270=float(p69.loc[key].hidden_200_270), ours_fhi_times_k=float(r.f_hi * K_P027), ours_shat_times_k=float(r.s_hat * K_P027),
                   ours_postmean_logu_times_k=float(r.s_mean_logu * K_P027), P069_ext400_gt200=float(p69.loc[key].hidden_ext400_gt200),
                   ours_gX_plus1_times_fhi_k=float((1 + r.g_X) * r.f_hi * K_P027))
    log(f'    P069 {key:5s}: 200-270 keV in 6.76 t yr {cmp[nm]["P069_200_270"]:.3f} vs ours f_hi*k {cmp[nm]["ours_fhi_times_k"]:.3f} (s_hat*k {cmp[nm]["ours_shat_times_k"]:.3f}, '
        f'<s>k {cmp[nm]["ours_postmean_logu_times_k"]:.3f}); >200 keV with 400 keV edge {cmp[nm]["P069_ext400_gt200"]:.3f} vs ours {cmp[nm]["ours_gX_plus1_times_fhi_k"]:.3f}')
xc['P069_per_model'] = cmp
# P020: L10 total-ROI plug-in 2.38 events (whole ROI) -> our L10 total ROI = s_hat/f_hi * k
rL = MM[MM.name == 'L10s_m1000'].iloc[0]
xc['P020_L10_total_ROI_6p76'] = dict(P020_plugin=2.38, ours_shat_over_fhi_times_k=float(rL.s_hat / rL.f_hi * K_P027), ours_postmean_over_fhi_times_k=float(rL.s_mean_logu / rL.f_hi * K_P027))
log(f'    P020 L10 whole-ROI 6.76 t yr: 2.38 plug-in vs ours s_hat/f_hi*k={xc["P020_L10_total_ROI_6p76"]["ours_shat_over_fhi_times_k"]:.2f} '
    f'(joint likelihood with the empty 125-200 keV bin pulls L10 down)')
xc['P034_a1'] = {int(r.delta): float(r.a1) for _, r in P034_S.iterrows()}
xc['seasonal_factors_first_run'] = F_first
xc['seasonal_factors_release_windows'] = {str(E): F_win[E] for E in EXPOSURES}
R['crosschecks'] = xc

# ------------------------------------------------------------------------------------------------
# 12. Falsifiable statement numbers and class decomposition
# ------------------------------------------------------------------------------------------------
cls_rows = []
for E, k in zip(EXPOSURES, KS):
    muH, muL, muX = region_means(D0, E)
    for c4 in ['elastic-isoscalar', 'elastic-isovector', 'inelastic-O1', 'inelastic-O4']:
        mk = MM.class4.values[D0['j']] == c4
        if mk.sum() == 0:
            continue
        cls_rows.append(dict(exposure_tyr=E, class4=c4, weight=float(mk.mean()), mean_H=float((k * muH[mk]).mean()), P0_H=float(np.mean(np.exp(-k * muH[mk]))),
                             mean_L2M=float((k * muL[mk]).mean()), mean_X=float((k * muX[mk]).mean())))
CLS = pd.DataFrame(cls_rows); CLS.to_csv(os.path.join(OUT, 'P081_class_decomposition.csv'), index=False)
log('[12] class decomposition at 2.8 t yr: ' + '; '.join(f'{r.class4} w={r.weight:.2f} <N_H>={r.mean_H:.2f} P0={r.P0_H:.2f} <N_X>={r.mean_X:.2f}' for _, r in CLS[CLS.exposure_tyr == 2.8].iterrows()))

# s posterior summary (mixture)
w_all = np.ones(NMC) / NMC
qs = np.quantile(D0['s'], [0.05, 0.16, 0.5, 0.84, 0.95])
R['mixture_s_posterior_200_270_per_2p84'] = dict(mean=float(D0['s'].mean()), q05=float(qs[0]), q16=float(qs[1]), median=float(qs[2]), q84=float(qs[3]), q95=float(qs[4]),
                                                  flat_prior_mean=float(VARIANTS['flat prior on s']['s'].mean()), jeff_prior_mean=float(VARIANTS['Jeffreys prior on s']['s'].mean()),
                                                  P038_reweighted_mean=float((W_P038 * D0['s']).sum()))
log(f'    mixture rate posterior s (200-270 keV per 2.84 t yr): mean {D0["s"].mean():.3f}, median {qs[2]:.3f}, 68% {qs[1]:.3f}-{qs[3]:.3f}, 90% {qs[0]:.3f}-{qs[4]:.3f}; '
    f'flat-prior mean {R["mixture_s_posterior_200_270_per_2p84"]["flat_prior_mean"]:.3f}')

# ------------------------------------------------------------------------------------------------
# 13. Figures
# ------------------------------------------------------------------------------------------------
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 4, figsize=(11, 3.2), sharey=True)
for ax, E in zip(axes, EXPOSURES):
    st = pred_store[(E, '200-270 keV (600 phd ROI)')]
    x = np.arange(3); wd = 0.26
    dm = [st['P_dm'][0], st['P_dm'][1], 1 - st['P_dm'][0] - st['P_dm'][1]]
    bk = [st['P_b'][0], st['P_b'][1], 1 - st['P_b'][0] - st['P_b'][1]]
    mx_q = np.quantile(st['P_mix'], [0.16, 0.5, 0.84], axis=0)
    q2 = np.quantile(1 - st['P_mix'][:, 0] - st['P_mix'][:, 1], [0.16, 0.5, 0.84])
    mx = [mx_q[1, 0], mx_q[1, 1], q2[1]]
    mx_lo = [mx_q[0, 0], mx_q[0, 1], q2[0]]; mx_hi = [mx_q[2, 0], mx_q[2, 1], q2[2]]
    ax.bar(x - wd, dm, wd, color=C['dm'], label='DM (P027 posterior)')
    ax.bar(x, mx, wd, color=C['mix'], label='corpus mixture (P061)', yerr=[np.array(mx) - np.array(mx_lo), np.array(mx_hi) - np.array(mx)], capsize=2, ecolor=C['grey'])
    ax.bar(x + wd, bk, wd, color=C['bkg'], label='modelled background')
    for xi_, v in zip(x - wd, dm):
        ax.text(xi_, v + 0.02, f'{v:.2f}', ha='center', fontsize=7, color='#333333')
    for xi_, v in zip(x + wd, bk):
        ax.text(xi_, v + 0.02, f'{v:.3f}' if v < 0.01 else f'{v:.2f}', ha='center', fontsize=7, color='#333333')
    ax.set_xticks(x); ax.set_xticklabels(['N = 0', 'N = 1', 'N $\\geq$ 2']); ax.set_title(f'{E:g} t·yr (k = {E/EXPO_FIRST:.2f})', fontsize=9)
    ax.set_ylim(0, 1.12); ax.grid(axis='y', alpha=0.25)
axes[0].set_ylabel('probability of N events, 200–270 keV NR band')
hnd, lab = axes[0].get_legend_handles_labels()
fig.legend(hnd, lab, loc='upper center', ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=[0, 0, 1, 0.9]); fig.savefig(os.path.join(FIG, 'P081_fig1_predictive_counts.png'), dpi=180); plt.close(fig)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
Et = kgrid * EXPO_FIRST
ax1.plot(Et, PGE1['200-270 keV'], color=C['dm'], lw=2, label='200–270 keV (600 phd ROI)')
ax1.plot(Et, PGE1['55-270 keV'], color=C['v'], lw=2, label='55–270 keV')
ax1.plot(Et, PGE1['55-420 keV (1000 phd)'], color=C['sky'], lw=2, label='55–420 keV (1000 phd edge)')
ax1.plot(Et, PGE1_plug, color=C['grey'], lw=1.5, ls='--', label='200–270 keV, plug-in best fit')
ax1.plot(Et, PGE1_P038, color=C['dm'], lw=1.2, ls=':', label='200–270 keV, P038 sideband reweighted')
ax1.axhline(0.9, color=C['black'], lw=0.8, ls='-.'); ax1.text(0.11, 0.905, 'P = 0.9', fontsize=7.5)
for E in EXPOSURES:
    ax1.axvline(E, color=C['grey'], lw=0.5, alpha=0.5)
ax1.set_xscale('log'); ax1.set_xlabel('additional LZ exposure (t·yr)'); ax1.set_ylabel('P(≥ 1 event | DM)'); ax1.set_ylim(0, 1)
ax1.legend(fontsize=7, frameon=False, loc='lower right'); ax1.grid(alpha=0.25)
sub = DEC[DEC.outcome.isin(['N=0', 'N=1', 'N=2', 'N>=3'])]
for outc, col in zip(['N=0', 'N=1', 'N=2', 'N>=3'], [C['bkg'], C['mix'], C['dm'], C['v']]):
    d = sub[sub.outcome == outc]
    ax2.plot(d.exposure_tyr, d.postDM_med, 'o-', color=col, lw=1.8, ms=5, label=outc.replace('>=', '≥'))
    if outc == 'N=1':
        ax2.fill_between(d.exposure_tyr, d.postDM_16, d.postDM_84, color=col, alpha=0.15, lw=0, label='68 % of P061 hyper-prior (N = 1)')
ax2.axhline(np.median(P_DM), color=C['grey'], lw=0.8, ls='--'); ax2.text(5.0, np.median(P_DM) * 0.55, 'today (P061 median 0.015)', fontsize=7, color=C['grey'])
ax2.set_xscale('log'); ax2.set_yscale('log'); ax2.set_ylim(1e-4, 1); ax2.set_xlabel('additional LZ exposure (t·yr)'); ax2.set_ylabel('P(DM | first event, N new events)')
ax2.set_xticks(EXPOSURES); ax2.set_xticklabels([f'{E:g}' for E in EXPOSURES]); ax2.legend(fontsize=7.5, frameon=False, title='N in 200–270 keV', title_fontsize=7.5); ax2.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P081_fig2_exposure_and_decision.png'), dpi=180); plt.close(fig)

# ------------------------------------------------------------------------------------------------
# 14. Save
# ------------------------------------------------------------------------------------------------
MM[['name', 'class4', 'cls', 'op', 'tau', 'mass', 'delta', 'f_lo', 'f_L2', 'f_M', 'f_hi', 'N_lo', 'N_M', 'rho', 'B_logu', 'B_logu_rho', 'post_uniform', 'post_classes',
    's_hat', 's_mean_logu', 's_med_logu', 's_mean_flat', 's_mean_jeff', 'g_lo', 'g_L2M', 'g_X_raw', 'g_X']].to_csv(os.path.join(OUT, 'P081_model_table.csv'), index=False)
R['settings'] = dict(exposure_first_tyr=EXPO_FIRST, exposures_tyr=EXPOSURES, NMC=NMC, NH=NH, f_trans=F_TRANS, sigma_delta_astro_keV=SIG_DELTA_ASTRO,
                     release_window_start='2024-04-01 (doy 92)', calendar_days_per_tyr=CAL_PER_TYR, s_grid=[1e-3, 30.0, len(S)],
                     edge_600=dict(E50=269.9, sigma=SIG_HI), edge_1000=dict(E50=E50_1000, sigma=SIG_1000), P038_E50_600=E50_600_P038)
R['predictive_table'] = PRED.to_dict(orient='records')
R['decision_table'] = DEC.to_dict(orient='records')
R['runtime_s'] = time.time() - T0
json.dump(R, open(os.path.join(OUT, 'P081_results.json'), 'w'), indent=1, default=float)
log(f'done in {time.time()-T0:.1f} s')
