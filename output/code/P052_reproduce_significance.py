"""
P052 -- Reproducing LZ's 3.4 sigma local significance from the published summaries:
        a three-sample two-dimensional toy likelihood.

Run from the simulation root (stages keep every call under 10 minutes):
  .venv/bin/python output/code/P052_reproduce_significance.py --stage fit
  .venv/bin/python output/code/P052_reproduce_significance.py --stage sa
  .venv/bin/python output/code/P052_reproduce_significance.py --stage toys --model L10_1000 --config lzlike --toys 10000 --seed 1
  .venv/bin/python output/code/P052_reproduce_significance.py --stage final

Model (every equation and input in work/P052/details.md):
  * Observable space (S1c, d), d = (log10 S2c - mu_NR(S1c))/sigma_NR(S1c) as in LZ Fig. 5.
    S1c bins: [3,250) as ONE bin (the Fig. 5 top panel is only published panel-wide), then 25 phd bins over
    250-500 (middle panel) and 500-600 (bottom panel).  d bins: 0.5 sigma from -8 to +8.
  * Background cells: digitised Fig. 5 d-histograms per panel (P016 top; P022 middle/bottom accidentals and totals;
    P004 bottom MSSI/total) with stated within-panel S1c shapes; MSSI/NR modelled where the digitisation cannot
    resolve them.  Normalisations and Gaussian constraints from Table I; prompt/delayed samples as Poisson totals
    (Tables S1/S2) coupled through lambda_PN, lambda_DN, lambda_MSSI, lambda_PG; signal split 1 : 1e-4 : 3.1e-2.
  * Signal cells: WimPyDD spectra (P003/P012/P021 caches) -> S1c with NEST-LZ yields on LZ's contour energy scale
    (P009); S1c resolution 'nest' (1.13 sqrt(S1c), P038 MC) or 'paper' (9.3%, the quoted +-23 keV stat);
    d-PDF 'gauss' N(0,1) or 'fig5' (the digitised L10 curve of Fig. 5, per panel); ROI cuts S1c in [3,600],
    log10 S2c in [2.75, 4.15].
  * Extended binned Poisson likelihood (fine cells ~ unbinned) profiled with iminuit; q0 with s >= 0;
    asymptotic Z = sqrt(q0); toy-calibrated Z from background-only toys (Baxter et al. 2021 Eq. 6, as LZ).
"""
import os, sys, json, time, argparse, glob
import numpy as np
import pandas as pd
from scipy import stats, special
from scipy.optimize import brentq
from iminuit import Minuit
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, 'output/code')
from common import lzcommon as lz

ap = argparse.ArgumentParser()
ap.add_argument('--stage', default='fit', choices=['fit', 'sa', 'toys', 'final'])
ap.add_argument('--model', default='L10_1000')
ap.add_argument('--config', default='lzlike', choices=['base', 'lzlike', 'fig5only', 'resonly'])
ap.add_argument('--toys', type=int, default=10000)
ap.add_argument('--seed', type=int, default=1)
ARGS = ap.parse_args()

T0 = time.time()
OUT = 'output/work/P052'; FIG = os.path.join(OUT, 'figures'); TOYDIR = os.path.join(OUT, 'toys')
os.makedirs(FIG, exist_ok=True); os.makedirs(TOYDIR, exist_ok=True)
LOG = open(os.path.join(OUT, f'run_log_{ARGS.stage}.txt'), 'a')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.write(s + '\n'); LOG.flush()
say(f'\n##### stage {ARGS.stage}  args {vars(ARGS)}  {time.strftime("%Y-%m-%d %H:%M:%S")}')
R = {}

CONFIGS = {'base': dict(dshape='gauss', res='nest'), 'fig5only': dict(dshape='fig5', res='nest'),
           'resonly': dict(dshape='gauss', res='paper'), 'lzlike': dict(dshape='fig5', res='paper')}

# ----------------------------------------------------------------------------------------------
# 1. Binning
# ----------------------------------------------------------------------------------------------
S1_EDGES = np.concatenate([[3.0, 250.0], np.arange(275.0, 600.1, 25.0)])   # 1 + 14 bins
NS1 = len(S1_EDGES) - 1
S1_MID = 0.5 * (S1_EDGES[1:] + S1_EDGES[:-1])
PANEL = np.array([0] + [1] * 10 + [2] * 4)
D_EDGES = np.arange(-8.0, 8.01, 0.5); ND = len(D_EDGES) - 1
D_MID = 0.5 * (D_EDGES[1:] + D_EDGES[:-1])
D_TAIL_MAX = 2.5

# ----------------------------------------------------------------------------------------------
# 2. Detector response for the signal
# ----------------------------------------------------------------------------------------------
g1, g2 = lz.LZ['g1'], lz.LZ['g2']
E_GRID = np.arange(1.0, 400.01, 1.0)
_yc = os.path.join(OUT, 'P052_yield_cache.npz')
if os.path.exists(_yc):
    _z = np.load(_yc); Nph_tab, Ne_tab = _z['Nph'], _z['Ne']
else:
    Nph_tab = np.zeros_like(E_GRID); Ne_tab = np.zeros_like(E_GRID)
    for i, e in enumerate(E_GRID):
        Nph_tab[i], Ne_tab[i] = lz.nest_nr_yields(e)          # Table S5 with the p(E) break (nestpy 2.1.1)
    np.savez(_yc, Nph=Nph_tab, Ne=Ne_tab)
Nq_paper = 11.32 * E_GRID ** 1.112                          # LZ's own contour energy scale (P009)
ratio = np.clip((Nq_paper - Ne_tab) / Nph_tab, 1.0, 1.12)
Nph_sig = Nph_tab * ratio
S1_MEAN = g1 * Nph_sig
LOG10S2_MEAN = np.log10(g2 * Ne_tab)
def mu_NR(s1):  return np.interp(s1, S1_MEAN, LOG10S2_MEAN)
def sig_NR(s1): return 0.033 * np.sqrt(540.0 / np.clip(s1, 3.0, None))
def s1_sigma(res):
    return 1.13 * np.sqrt(S1_MEAN) if res == 'nest' else 0.093 * S1_MEAN     # 'paper': 23/248 keV
EV_S1C, EV_LOG10S2C = lz.LZ['ev_S1c'], lz.LZ['ev_log10S2c']
EV_D_NOMINAL = -1.54
R['response'] = dict(S1c_at_248keV=float(np.interp(248, E_GRID, S1_MEAN)), S1c_at_270keV=float(np.interp(270, E_GRID, S1_MEAN)),
                     E_at_S1c540=float(np.interp(540, S1_MEAN, E_GRID)), mu_NR_at_540=float(mu_NR(540.0)),
                     ratio_at_250=float(np.interp(250, E_GRID, ratio)), ratio_at_20=float(np.interp(20, E_GRID, ratio)),
                     event_d_from_our_band=float((EV_LOG10S2C - mu_NR(EV_S1C)) / 0.033),
                     sigma_S1c_at_540_nest=float(np.interp(540, S1_MEAN, s1_sigma('nest'))), sigma_S1c_at_540_paper=float(np.interp(540, S1_MEAN, s1_sigma('paper'))))
say('response:', R['response'])

def eff_low(E):
    return 0.96 * 0.5 * (1 + special.erf((E - 5.4) / (np.sqrt(2) * 3.4)))

# ----------------------------------------------------------------------------------------------
# 3. Fig. 5 digitisation of the L10 (1000 GeV) best-fit curve -> signal d-PDF per panel
# ----------------------------------------------------------------------------------------------
X_L, PX_PER_SIG = 218.7, (2135.5 - 492.5) / 12.0                    # P022 tick calibration
FIG5_PANELS = {0: (496, 1299, 594.5, 2.0, (1158.5 - 594.5) / 4.0), 1: (1299, 2102, 1383.0, 1.0, (1982.5 - 1383.0) / 5.0),
               2: (2102, 2905, 2131.5, 2.0, (2711.5 - 2131.5) / 6.0)}
def digitise_L10():
    im = np.asarray(Image.open('inputs/figures_png/Fig5_NR_distance_3panels_no_sig.png').convert('RGB')).astype(int)
    col = np.array([165, 42, 42]); out = {}
    for p, (yt, yb, yref, lref, ppd) in FIG5_PANELS.items():
        sub = im[yt + 3:yb - 2, :]; mask = np.abs(sub - col).sum(axis=2) < 60
        vals = np.zeros(ND)
        for i in range(ND):
            xa = int(X_L + (D_EDGES[i] + 8 + 0.10) * PX_PER_SIG); xb = int(X_L + (D_EDGES[i] + 8 + 0.40) * PX_PER_SIG)
            ys = [np.where(mask[:, x])[0].min() + yt + 3 for x in range(xa, xb) if mask[:, x].any()]
            if ys: vals[i] = 10 ** (lref - (np.median(ys) - yref) / ppd)
        out[p] = vals
    return out
L10_FIG5 = digitise_L10()
L10_FIG5_SUM = {p: float(v.sum()) for p, v in L10_FIG5.items()}
DSHAPE_FIG5 = {p: v / v.sum() for p, v in L10_FIG5.items()}
GAUSS = np.diff(stats.norm.cdf(D_EDGES))
R['fig5_L10_digitised'] = dict(sums=L10_FIG5_SUM, total=float(sum(L10_FIG5_SUM.values())),
                               frac_event_bin={p: float(DSHAPE_FIG5[p][int((EV_D_NOMINAL + 8) / 0.5)]) for p in DSHAPE_FIG5},
                               gauss_event_bin=float(GAUSS[int((EV_D_NOMINAL + 8) / 0.5)]),
                               mean_d={p: float(np.sum(D_MID * DSHAPE_FIG5[p])) for p in DSHAPE_FIG5},
                               sd_d={p: float(np.sqrt(np.sum(D_MID ** 2 * DSHAPE_FIG5[p]) - np.sum(D_MID * DSHAPE_FIG5[p]) ** 2)) for p in DSHAPE_FIG5})
say('Fig.5 L10 digitised:', json.dumps(R['fig5_L10_digitised'], default=float))
pd.DataFrame(dict(d_low=D_EDGES[:-1], d_high=D_EDGES[1:], L10_top=L10_FIG5[0], L10_mid=L10_FIG5[1], L10_bot=L10_FIG5[2], gauss=GAUSS)).to_csv(
    os.path.join(OUT, 'P052_fig5_L10_digitised.csv'), index=False)

def signal_cells(E, dRdE, dshape='gauss', res='nest', mu_d=0.0, sig_d=1.0):
    """Expected events per cell for unit in-ROI signal: (matrix NS1 x ND normalised to 1, in-ROI fraction, S1c-bin weights)."""
    r = np.interp(E_GRID, E, dRdE, left=0.0, right=0.0) * eff_low(E_GRID)
    S1S = s1_sigma(res)
    cdf = stats.norm.cdf((S1_EDGES[None, :] - S1_MEAN[:, None]) / S1S[:, None])
    p_s1 = np.diff(cdf, axis=1)
    w_s1 = (r[:, None] * p_s1).sum(axis=0)
    M = np.zeros((NS1, ND))
    for k in range(NS1):
        dmax = (4.15 - mu_NR(S1_MID[k])) / sig_NR(S1_MID[k]); dmin = (2.75 - mu_NR(S1_MID[k])) / sig_NR(S1_MID[k])
        if dshape == 'gauss':
            lo = np.clip(D_EDGES[:-1], dmin, dmax); hi = np.clip(D_EDGES[1:], dmin, dmax)
            pd_ = stats.norm.cdf((hi - mu_d) / sig_d) - stats.norm.cdf((lo - mu_d) / sig_d)
        else:
            pd_ = DSHAPE_FIG5[PANEL[k]] * ((D_EDGES[1:] <= dmax) & (D_EDGES[:-1] >= dmin))
        M[k] = w_s1[k] * pd_
    tot = M.sum()
    return M / tot, float(tot / max(r.sum(), 1e-300)), w_s1 / tot

# ----------------------------------------------------------------------------------------------
# 4. Signal spectra (WimPyDD caches from P003, P012, P021)
# ----------------------------------------------------------------------------------------------
P003 = np.load('output/work/P003/P003_spectra.npz')
P012 = np.load('output/work/P012/P012_L10_spectra_d10_1.npz')
P021s = {m: np.load(f'output/work/P021/spectra_s_{m}.npz') for m in (400, 1000, 4000)}
P021v = {m: np.load(f'output/work/P021/spectra_v_{m}.npz') for m in (400, 1000, 4000)}
MODELS = []
for m, z in zip([100, 200, 400, 1000, 4000], [2.9, 3.1, 3.4, 3.4, 3.4]):
    MODELS.append((f'L10_{m}', 'L10 (m scan)', P012['E'], P012[f'm{m}'], z))
ELASTIC = {('O1s', 1000): 0.0, ('O1v', 1000): 1.3, ('O4s', 1000): 2.7, ('O4v', 1000): 2.7, ('O6s', 1000): 3.1, ('O6v', 1000): 3.3,
           ('O10s', 1000): 3.0, ('O10v', 1000): 3.1, ('O11s', 1000): 1.7, ('O11v', 1000): 2.6,
           ('O4s', 200): 2.3, ('O6s', 200): 2.2, ('O10s', 200): 2.8, ('O11s', 200): 1.1, ('O1v', 200): 0.0,
           ('O4s', 4000): 2.7, ('O6s', 4000): 2.6, ('O10s', 4000): 3.1, ('O11s', 4000): 1.8, ('O1v', 4000): 1.2}
for (op, m), z in ELASTIC.items():
    MODELS.append((f'{op}_{m}', 'elastic operators', P003['E'], P003[f'{op}_{m}'], z))
def inel(tag, m, delta):
    S = (P021s if tag == 's' else P021v)[m]
    i = int(np.argmin(np.abs(S['deltas'] - delta))); assert abs(S['deltas'][i] - delta) < 1e-6
    return S['E'], S['annual'][i]
for m in (400, 1000, 4000):
    for j, delta in enumerate(lz.OSIG_DELTAS):
        if delta < 100: continue
        for tag, key in (('s', 'O1s'), ('v', 'O1v')):
            z = lz.OSIG[key][m][j]
            if z is None: continue
            E, r = inel(tag, m, delta)
            if r.sum() <= 0: continue
            MODELS.append((f'{key}_{m}_d{delta}', 'inelastic O1 (s/v)', E, r, z))
# diagnostic sets (excluded from the rms): LZ's own Fig. 1 spectra (P003 vector digitisation) and Helm-form-factor inelastic O1s
F1B = pd.read_csv('output/work/P003/fig1_digitised_bottom.csv'); F1T = pd.read_csv('output/work/P003/fig1_digitised_top.csv')
def _clean(E, y):
    ok = np.isfinite(y) & (y > 0); return E[ok], y[ok]
MODELS.append(('L10_1000_fig1', 'check: LZ Fig. 1 spectrum', *_clean(F1B.E_keV.values.astype(float), F1B.L10s_1000GeV.values), 3.4))
MODELS.append(('L10_200_fig1', 'check: LZ Fig. 1 spectrum', *_clean(F1B.E_keV.values.astype(float), F1B.L10s_200GeV.values), 3.1))
MODELS.append(('O1s_1000_fig1', 'check: LZ Fig. 1 spectrum', *_clean(F1T.E_keV.values.astype(float), F1T.O1s_delta0_1000GeV.values), 0.0))
MODELS.append(('O1s_1000_d200_fig1', 'check: LZ Fig. 1 spectrum', *_clean(F1T.E_keV.values.astype(float), F1T.O1s_delta200_1000GeV.values), 2.7))
MODELS.append(('O1s_1000_d300_fig1', 'check: LZ Fig. 1 spectrum', *_clean(F1T.E_keV.values.astype(float), F1T.O1s_delta300_1000GeV.values), 3.0))
E_H = np.arange(1.0, 400.0, 1.0)
for delta, z in ((200, 2.7), (250, 2.9), (300, 3.0), (350, 3.3)):
    rH = np.array([lz.dRdE_SI(e, 1000.0, 1e-45, delta_kev=float(delta)) for e in E_H])
    MODELS.append((f'O1s_1000_d{delta}_helm', 'check: Helm form factor', E_H, rH, z))
MODEL_BY_NAME = {mm[0]: mm for mm in MODELS}
say(f'{len(MODELS)} models')

# ----------------------------------------------------------------------------------------------
# 5. Background cells
# ----------------------------------------------------------------------------------------------
P016 = json.load(open('output/work/P016/P016_results.json'))['fig5_top_digitised']
P004 = pd.read_csv('output/work/P004/P004_fig5_digitised.csv')
P022 = pd.read_csv('output/work/P022/P022_fig5_accidentals_digitised.csv')
def gauss_hist(mu=0.0, sig=1.0): return np.diff(stats.norm.cdf((D_EDGES - mu) / sig))
mssi_shape = P004.mssi.values / P004.mssi.values.sum()
MSSI_TOT = 4.9e-3; MSSI_BOT = float(P004.mssi.values.sum())
MSSI_PANEL = np.array([0.5 * (MSSI_TOT - MSSI_BOT), 0.5 * (MSSI_TOT - MSSI_BOT), MSSI_BOT])
NU_TOT, B8_TOT = 0.11, 0.057
LAM_NU = 58.0
def expo_bin_weights(lam, edges): return np.exp(-edges[:-1] / lam) - np.exp(-edges[1:] / lam)
nu_s1 = expo_bin_weights(LAM_NU, S1_EDGES); nu_s1 /= (np.exp(-3.0 / LAM_NU) - np.exp(-600.0 / LAM_NU))
LAM_N = 35.0
n_s1 = expo_bin_weights(LAM_N, S1_EDGES); n_s1 /= n_s1.sum()
acc_top = np.array(P016['components']['Accidentals'])
acc_mid = P022['250<S1c<500_acc_filled'].values
acc_bot = P022['S1c>500_acc_filled'].values
LAM_ACC = 175.0 / np.log((acc_mid.sum() / 250.0) / (acc_bot.sum() / 100.0))
tot_top = np.array(P016['components']['Total']); int_top = np.array(P016['components']['Internal']); er_top = np.array(P016['components']['ERs'])
nr_top = gauss_hist() * (NU_TOT * nu_s1[0] + B8_TOT); mssi_top = mssi_shape * MSSI_PANEL[0]
er_top_filled = np.where(er_top > 0, er_top, np.clip(tot_top - int_top - acc_top - nr_top - mssi_top, 0, None))
ER_TOP = er_top_filled + int_top
tot_mid = np.nan_to_num(P022['250<S1c<500_total'].values)
nr_mid = gauss_hist() * NU_TOT * nu_s1[1:11].sum(); mssi_mid = mssi_shape * MSSI_PANEL[1]
ER_MID = np.clip(tot_mid - acc_mid - nr_mid - mssi_mid, 0, None)
tot_bot = P004.total.values
nr_bot = gauss_hist() * NU_TOT * nu_s1[11:].sum(); mssi_bot = P004.mssi.values
ER_BOT = np.clip(tot_bot - acc_bot - nr_bot - mssi_bot, 0, None)
def spread(h, w): return w[:, None] * h[None, :]
w_er_mid = expo_bin_weights(30.0, S1_EDGES[1:12] - 250.0); w_er_mid /= w_er_mid.sum()
w_flat_mid = np.full(10, 0.1); w_flat_bot = np.full(4, 0.25)
w_acc_mid = expo_bin_weights(LAM_ACC, S1_EDGES[1:12]); w_acc_mid /= w_acc_mid.sum()
w_acc_bot = expo_bin_weights(LAM_ACC, S1_EDGES[11:]); w_acc_bot /= w_acc_bot.sum()
w_nu_mid = nu_s1[1:11] / nu_s1[1:11].sum(); w_nu_bot = nu_s1[11:] / nu_s1[11:].sum()
def assemble(top, mid, bot, w_mid, w_bot):
    M = np.zeros((NS1, ND)); M[0] = top; M[1:11] = spread(mid, w_mid); M[11:] = spread(bot, w_bot); return M
ER = assemble(ER_TOP, ER_MID, ER_BOT, w_er_mid, w_flat_bot)
ACC = assemble(acc_top, acc_mid, acc_bot, w_acc_mid, w_acc_bot)
MSSI = assemble(mssi_top, mssi_mid, mssi_bot, w_flat_mid, w_flat_bot)
NU = assemble(gauss_hist() * (NU_TOT * nu_s1[0] + B8_TOT), nr_mid, nr_bot, w_nu_mid, w_nu_bot)
NEUT = n_s1[:, None] * gauss_hist()[None, :]
TAILMASK = (D_MID < D_TAIL_MAX)[None, :].repeat(NS1, axis=0)
ER_BULK = ER * (~TAILMASK); ER_TAIL = ER * TAILMASK
PLATEAU = ER_BULK / ER_BULK.sum()
ER_TOTAL_TABLE = 1341 + 140.6 + 110.0 + 55.3 + 21.0 + 17.1 + 8.9 + 8.5 + 1.5
F_DET = (8.5 + 1.5) / ER_TOTAL_TABLE
R['background'] = dict(ER_top=float(ER_TOP.sum()), ER_top_tail=float(ER_TOP[D_MID < D_TAIL_MAX].sum()), ER_mid=float(ER_MID.sum()), ER_bot=float(ER_BOT.sum()),
                       ACC=float(ACC.sum()), ACC_by_panel=[float(acc_top.sum()), float(acc_mid.sum()), float(acc_bot.sum())], MSSI=float(MSSI.sum()),
                       NU=float(NU.sum()), NU_bot=float(nr_bot.sum()), bottom_total=float(tot_bot.sum()), lambda_acc_phd=float(LAM_ACC),
                       total_visible_model=float(ER.sum() + ACC.sum() + MSSI.sum() + NU.sum()))
say('background totals:', R['background'])

# ----------------------------------------------------------------------------------------------
# 6. Data cells
# ----------------------------------------------------------------------------------------------
data_top = np.array(P016['data_counts'], dtype=float)
low_mask = D_MID < D_TAIL_MAX
N_MID_DATA = {4.5: 1, 5.0: 1, 5.5: 3, 6.0: 4, 6.5: 10, 7.0: 4}
n_mid_total = sum(N_MID_DATA.values())
plateau_scale = (1710 - 1 - n_mid_total - data_top[low_mask].sum()) / data_top[~low_mask].sum()
data_top_scaled = data_top.copy(); data_top_scaled[~low_mask] *= plateau_scale
DATA = np.zeros((NS1, ND)); DATA[0] = data_top_scaled
for dlo, n in N_MID_DATA.items(): DATA[1, int(round((dlo + 8.0) / 0.5))] += n
def event_cell(d_ev):
    return int(np.searchsorted(S1_EDGES, EV_S1C) - 1), int(np.floor((d_ev + 8.0) / 0.5))
K_EV, J_EV = event_cell(EV_D_NOMINAL)
DATA[K_EV, J_EV] += 1
N_PROMPT, N_DELAYED = 66, 55
PROMPT_SMALL_ER = 0.13 + 0.0141 + 0.011 + 0.0055 + 0.0021 + 0.0017 + 0.00089 + 0.00027
DELAYED_BKG = 39.5 + 4.1 + 3.2 + 1.6 + 0.62 + 0.50 + 0.26 + 0.25 + 0.080 + 0.045 + 0.0033 + 0.0017 + 0.00014
DELAYED_SIG = np.sqrt(4.7 ** 2 + 0.2 ** 2 + 0.5 ** 2 + 0.1 ** 2 + 0.19 ** 2 + 0.15 ** 2 + 0.08 ** 2 + 0.10 ** 2 + 0.019 ** 2 + 0.010 ** 2) / DELAYED_BKG
ER_SIG = np.sqrt(160 ** 2 + 8.4 ** 2 + 16.5 ** 2 + 3.1 ** 2 + 6.3 ** 2 + 5.1 ** 2 + 2.7 ** 2) / (ER_TOTAL_TABLE - 10.0)
R_DET = 8.5 / 0.12; R_XE = 1.5 / 0.12; R_MSSI = MSSI_TOT / 0.06
SIG_SPLIT = dict(prompt=1.0e-4, delayed=3.1e-2)
R['data'] = dict(top_digitised=float(data_top.sum()), top_tail=float(data_top[low_mask].sum()), plateau_scale=float(plateau_scale), mid=n_mid_total,
                 science_total=float(DATA.sum()), event_cell=[int(K_EV), int(J_EV)], event_cell_edges=[float(S1_EDGES[K_EV]), float(S1_EDGES[K_EV + 1]), float(D_EDGES[J_EV]), float(D_EDGES[J_EV + 1])])
say('data:', R['data'])

# ----------------------------------------------------------------------------------------------
# 7. Likelihood
# ----------------------------------------------------------------------------------------------
PNAMES = ['s', 'tER', 'tTail', 'tAcc', 'tMSSI', 'tNu', 'Nn', 'tDet', 'tXe', 'tD', 'lPG', 'lMSSI', 'lPN', 'lDN']
NOM = dict(s=0.0, tER=1.0, tTail=1.0, tAcc=1.0, tMSSI=1.0, tNu=1.0, Nn=0.0, tDet=1.0, tXe=1.0, tD=1.0, lPG=0.88, lMSSI=0.94, lPN=0.05, lDN=0.87)
CON = dict(tER=ER_SIG, tTail=0.30, tAcc=0.6 / 2.7, tMSSI=1.0, tNu=0.2 / 1.1, tDet=3.4 / 8.5, tXe=0.3 / 1.5, tD=DELAYED_SIG,
           lPG=0.02, lMSSI=0.02, lPN=0.01, lDN=0.02)
LIMITS = dict(s=(0, 60), tER=(0.3, 3), tTail=(0, 5), tAcc=(0, 5), tMSSI=(0, 8), tNu=(0, 5), Nn=(0, 200), tDet=(0, 5), tXe=(0, 5),
              tD=(0.3, 3), lPG=(0.5, 1.0), lMSSI=(0.5, 1.0), lPN=(0.0, 0.3), lDN=(0.5, 1.0))

class Likelihood:
    def __init__(self, S, use_veto=True, merge_1d=False, fix_nuis=False, aux=None, fix_lambdas=False):
        self.S = S; self.use_veto = use_veto; self.fix_nuis = fix_nuis; self.fix_lambdas = fix_lambdas
        self.aux = dict(NOM) if aux is None else aux
        comps = dict(S=S, ERB=ER_BULK, ERT=ER_TAIL, ACC=ACC, MSSI=MSSI, NU=NU, NEUT=NEUT, PLAT=PLATEAU)
        self.merge = None
        if merge_1d:
            core = (D_MID > -3) & (D_MID < 3); idx = np.where(core)[0]
            def mg(M): return np.concatenate([M[:, :idx[0]], M[:, core].sum(axis=1, keepdims=True), M[:, idx[-1] + 1:]], axis=1)
            comps = {k: mg(v) for k, v in comps.items()}; self.merge = mg
        self.c = {k: v.ravel() for k, v in comps.items()}
        self.set_data(DATA, N_PROMPT, N_DELAYED)
    def set_data(self, D, n_p, n_d):
        self.n = (self.merge(D) if self.merge else D).ravel(); self.n_p = n_p; self.n_d = n_d
    def mu_cells(self, p):
        c = self.c; sci_mssi = p['tMSSI'] * R_MSSI * (1 - p['lMSSI']) / MSSI_TOT
        return (p['s'] * c['S'] + p['tER'] * (1 - F_DET) * (c['ERB'] + p['tTail'] * c['ERT']) + p['tAcc'] * c['ACC']
                + sci_mssi * c['MSSI'] + p['tNu'] * c['NU'] + p['Nn'] * c['NEUT']
                + (1 - p['lPG']) * (R_DET * p['tDet'] + R_XE * p['tXe']) * c['PLAT'])
    def mu_veto(self, p):
        f_sci = max(1 - p['lPN'] - p['lDN'], 1e-3)
        mu_p = (PROMPT_SMALL_ER * p['tER'] + p['lPG'] * (R_DET * p['tDet'] + R_XE * p['tXe']) + p['lMSSI'] * R_MSSI * p['tMSSI']
                + p['lPN'] / f_sci * p['Nn'] + SIG_SPLIT['prompt'] * p['s'])
        mu_d = DELAYED_BKG * p['tD'] + p['lDN'] / f_sci * p['Nn'] + SIG_SPLIT['delayed'] * p['s']
        return mu_p, mu_d
    def nll(self, *args):
        p = dict(zip(PNAMES, args))
        mu = np.clip(self.mu_cells(p), 1e-300, None)
        val = np.sum(mu - self.n * np.log(mu))
        if self.use_veto:
            mp, md = self.mu_veto(p); mp = max(mp, 1e-300); md = max(md, 1e-300)
            val += mp - self.n_p * np.log(mp) + md - self.n_d * np.log(md)
        for k, sg in CON.items(): val += 0.5 * ((p[k] - self.aux[k]) / sg) ** 2
        return val
    def fit(self, fix_s=None, start=None):
        st = dict(NOM) if start is None else dict(start)
        if fix_s is not None: st['s'] = fix_s
        m = Minuit(self.nll, *[st[k] for k in PNAMES], name=PNAMES)
        for k in PNAMES: m.limits[k] = LIMITS[k]
        m.errordef = 0.5; m.strategy = 0; m.print_level = 0
        if fix_s is not None: m.fixed['s'] = True
        if not self.use_veto:
            for k in ('tD', 'lPN', 'lDN', 'lPG', 'lMSSI', 'tDet', 'tXe'): m.fixed[k] = True
        if self.fix_lambdas:
            for k in ('lPG', 'lMSSI', 'lPN', 'lDN'): m.fixed[k] = True
        if self.fix_nuis:
            for k in PNAMES:
                if k != 's': m.fixed[k] = True
        m.migrad()
        return m
    def q0(self, return_fits=False):
        m0 = self.fit(fix_s=0.0)
        p0 = dict(zip(PNAMES, m0.values)); mu0 = np.clip(self.mu_cells(p0), 1e-300, None)
        score = np.sum(self.n * self.c['S'] / mu0) - self.c['S'].sum()
        if self.use_veto:
            mp, md = self.mu_veto(p0)
            score += self.n_p * SIG_SPLIT['prompt'] / mp - SIG_SPLIT['prompt'] + self.n_d * SIG_SPLIT['delayed'] / md - SIG_SPLIT['delayed']
        if score <= 0:
            return (0.0, 0.0, m0, None) if return_fits else (0.0, 0.0)
        st = dict(zip(PNAMES, m0.values)); st['s'] = 1.0
        m1 = self.fit(start=st)
        if m1.fval > m0.fval:
            st['s'] = 0.3; m1b = self.fit(start=st)
            if m1b.fval < m1.fval: m1 = m1b
        q = max(0.0, 2 * (m0.fval - m1.fval))
        return (q, float(m1.values['s']), m0, m1) if return_fits else (q, float(m1.values['s']))

def Zof(q): return float(np.sqrt(max(q, 0.0)))
def summarize(df, col='Z'):
    d = df[col] - df.Z_LZ
    return dict(rms=float(np.sqrt(np.mean(d ** 2))), mean=float(d.mean()), max_abs=float(d.abs().max()), n=int(len(d)))
def sig_for(name, cfg, **kw):
    _, cls, E, r, z = MODEL_BY_NAME[name]; return signal_cells(E, r, dshape=CONFIGS[cfg]['dshape'], res=CONFIGS[cfg]['res'], **kw)

# ==============================================================================================
if ARGS.stage == 'fit':
    # -- 8. data fits, all models, four configurations
    allrows = []
    for cfg in ['base', 'fig5only', 'resonly', 'lzlike']:
        say(f'\n=== data fits: config {cfg} {CONFIGS[cfg]} ===')
        for name, cls, E, r, zlz in MODELS:
            S, f_roi, w_s1 = signal_cells(E, r, **CONFIGS[cfg])
            L = Likelihood(S); q, sh, m0, m1 = L.q0(return_fits=True)
            p0 = dict(zip(PNAMES, m0.values)); b_ev = float(L.mu_cells(p0).reshape(NS1, ND)[K_EV, J_EV]); s_ev = float(S[K_EV, J_EV])
            allrows.append(dict(config=cfg, model=name, cls=cls, Z_LZ=zlz, q0=q, Z=Zof(q), s_hat=sh, frac_top=float(w_s1[0]), frac_mid=float(w_s1[1:11].sum()),
                                frac_bot=float(w_s1[11:].sum()), f_in_roi=f_roi, b_event_cell=b_ev, s_event_cell_per_unit=s_ev, LR_event_per_unit=s_ev / b_ev))
            say(f'{name:16s} LZ {zlz:3.1f}  Z {Zof(q):4.2f}  q0 {q:6.2f}  s_hat {sh:5.2f}  split {w_s1[0]:.2f}/{w_s1[1:11].sum():.2f}/{w_s1[11:].sum():.2f}  LR_ev {s_ev/b_ev:7.1f}')
    df = pd.DataFrame(allrows); df['diff'] = df.Z - df.Z_LZ
    df.to_csv(os.path.join(OUT, 'P052_data_fits_all_configs.csv'), index=False)
    R['fits_vs_LZ'] = {}
    for cfg in CONFIGS:
        sub_all = df[df.config == cfg]; sub = sub_all[~sub_all.cls.str.startswith('check')]
        R['fits_vs_LZ'][cfg] = dict(all=summarize(sub), excl_LZ_zero=summarize(sub[sub.Z_LZ > 0.05]),
                                    checks={r.model: dict(Z=float(r.Z), Z_LZ=float(r.Z_LZ), frac_bot=float(r.frac_bot), frac_top=float(r.frac_top), LR_ev=float(r.LR_event_per_unit))
                                            for r in sub_all[sub_all.cls.str.startswith('check')].itertuples()},
                                    by_class={c: summarize(sub[sub.cls == c]) for c in sub.cls.unique()},
                                    isoscalar_O1_inel=summarize(sub[sub.model.str.startswith('O1s_') & sub.model.str.contains('_d')]),
                                    isovector_O1_inel=summarize(sub[sub.model.str.startswith('O1v_') & sub.model.str.contains('_d')]),
                                    headline={n: float(sub.set_index('model').loc[n, 'Z']) for n in ['L10_1000', 'O1s_1000_d350', 'O1s_1000_d300', 'O1s_1000', 'O4s_1000', 'O1v_1000', 'O10s_1000', 'O11s_1000']})
        say(f'{cfg:9s} vs LZ: all rms {R["fits_vs_LZ"][cfg]["all"]["rms"]:.2f} mean {R["fits_vs_LZ"][cfg]["all"]["mean"]:+.2f}; '
            f'O1s inel rms {R["fits_vs_LZ"][cfg]["isoscalar_O1_inel"]["rms"]:.2f} mean {R["fits_vs_LZ"][cfg]["isoscalar_O1_inel"]["mean"]:+.2f}; '
            f'O1v inel mean {R["fits_vs_LZ"][cfg]["isovector_O1_inel"]["mean"]:+.2f}; L10 {R["fits_vs_LZ"][cfg]["headline"]["L10_1000"]:.2f}')

    # -- 9. L10 (1000 GeV) detail in base and lzlike
    R['L10_1000'] = {}
    for cfg in ['base', 'lzlike']:
        S_L10, _, _ = sig_for('L10_1000', cfg); L = Likelihood(S_L10)
        q, sh, m0, m1 = L.q0(return_fits=True); m1.minos('s'); me = m1.merrors['s']
        pulls = {k: float((m1.values[k] - NOM[k]) / CON[k]) for k in CON}
        p1 = dict(zip(PNAMES, m1.values)); mp, md = L.mu_veto(p1); mu1 = L.mu_cells(p1).reshape(NS1, ND)
        def q_at(sfix): return 2 * (L.fit(fix_s=sfix, start=dict(zip(PNAMES, m1.values))).fval - m1.fval)
        ul = float(brentq(lambda x: q_at(x) - 2.706, sh + 0.01, 20.0)); ll = float(brentq(lambda x: q_at(x) - 2.706, 1e-4, sh)) if q > 2.706 else 0.0
        R['L10_1000'][cfg] = dict(q0=q, Z_asym=Zof(q), s_hat=sh, s_minos=[float(me.lower), float(me.upper)], s_68_interval=[sh + me.lower, sh + me.upper],
                                  s_90_interval_asym=[ll, ul], pulls=pulls, max_abs_pull=float(max(abs(v) for v in pulls.values())), Nn_hat=float(m1.values['Nn']),
                                  mu_prompt_fit=float(mp), mu_delayed_fit=float(md), science_total_fit=float(mu1.sum()),
                                  fitted_counts=dict(ER=float(p1['tER'] * (1 - F_DET) * (ER_BULK.sum() + p1['tTail'] * ER_TAIL.sum())), accidentals=float(p1['tAcc'] * ACC.sum()),
                                                     MSSI=float(p1['tMSSI'] * R_MSSI * (1 - p1['lMSSI'])), atm_nu_B8=float(p1['tNu'] * NU.sum()), detNR_science=float(p1['Nn'])),
                                  b_event_cell_H0=float(L.mu_cells(dict(zip(PNAMES, m0.values))).reshape(NS1, ND)[K_EV, J_EV]), s_event_cell=float(S_L10[K_EV, J_EV] * sh),
                                  signal_split=dict(top=float(S_L10[0].sum()), mid=float(S_L10[1:11].sum()), bot=float(S_L10[11:].sum())),
                                  Fig5_split_digitised=dict(top=L10_FIG5_SUM[0] / sum(L10_FIG5_SUM.values()), mid=L10_FIG5_SUM[1] / sum(L10_FIG5_SUM.values()), bot=L10_FIG5_SUM[2] / sum(L10_FIG5_SUM.values())))
        say(f'L10 detail {cfg}:', json.dumps(R['L10_1000'][cfg], default=float))

    # -- 10. variants for the headline models (both configs)
    HEAD = ['L10_1000', 'O1s_1000_d350', 'O1s_1000_d300', 'O1s_1000', 'O1v_1000', 'O4s_1000', 'O10s_1000', 'O11s_1000']
    R['variants'] = {}
    for cfg in ['base', 'lzlike']:
        SIG = {n: sig_for(n, cfg)[0] for n in HEAD}
        VAR = {}
        def run_variant(label, builder):
            out = {}
            for n in HEAD:
                L = builder(n); q, sh = L.q0(); out[n] = dict(q0=q, Z=Zof(q), s_hat=sh)
            say(f'[{cfg}] {label:36s} ' + '  '.join(f'{n}:{out[n]["Z"]:4.2f}' for n in HEAD)); return out
        VAR['baseline'] = run_variant('baseline (2D, 3 samples, profiled)', lambda n: Likelihood(SIG[n]))
        VAR['no veto samples'] = run_variant('no veto samples', lambda n: Likelihood(SIG[n], use_veto=False))
        VAR['nuisances fixed'] = run_variant('nuisances fixed (s only)', lambda n: Likelihood(SIG[n], fix_nuis=True))
        VAR['1D in S1c (|d|<3 merged)'] = run_variant('1D in S1c (|d|<3 merged)', lambda n: Likelihood(SIG[n], merge_1d=True))
        for tsig in (0.15, 1.0):
            CON['tTail'] = tsig; VAR[f'ER-tail constraint {int(100*tsig)}%'] = run_variant(f'ER-tail constraint {int(100*tsig)}%', lambda n: Likelihood(SIG[n]))
        CON['tTail'] = 0.30
        def scaled(fm, fa):
            def b(n):
                L = Likelihood(SIG[n]); L.c['MSSI'] = L.c['MSSI'] * fm; L.c['ACC'] = L.c['ACC'] * fa; return L
            return b
        VAR['MSSI x2'] = run_variant('MSSI x2', scaled(2.0, 1.0)); VAR['accidentals+MSSI x2'] = run_variant('accidentals and MSSI x2', scaled(2.0, 2.0))
        VAR['accidentals+MSSI x0.5'] = run_variant('accidentals and MSSI x0.5', scaled(0.5, 0.5))
        def build_band(n):
            S, _, _ = sig_for(n, cfg, sig_d=0.0212 / 0.0313) if CONFIGS[cfg]['dshape'] == 'gauss' else sig_for(n, cfg)
            L = Likelihood(S); D2 = DATA.copy(); D2[K_EV, J_EV] -= 1; k, j = event_cell(-2.33); D2[k, j] += 1; L.set_data(D2, N_PROMPT, N_DELAYED); return L
        VAR['band 0.0212 dex (event at -2.33)'] = run_variant('band 0.0212 dex, event at -2.33', build_band)
        def build_ev(n):
            L = Likelihood(SIG[n]); D2 = DATA.copy(); D2[K_EV, J_EV] -= 1; k, j = event_cell(-2.01); D2[k, j] += 1; L.set_data(D2, N_PROMPT, N_DELAYED); return L
        VAR['event one d-bin lower'] = run_variant('event moved one d-bin lower', build_ev)
        def build_ev_up(n):
            L = Likelihood(SIG[n]); D2 = DATA.copy(); D2[K_EV, J_EV] -= 1; k, j = event_cell(-1.2); D2[k, j] += 1; L.set_data(D2, N_PROMPT, N_DELAYED); return L
        VAR['event one d-bin higher'] = run_variant('event moved one d-bin higher', build_ev_up)
        if CONFIGS[cfg]['dshape'] == 'gauss':
            VAR['signal d-mean -0.25'] = run_variant('signal d-mean -0.25', lambda n: Likelihood(sig_for(n, cfg, mu_d=-0.25)[0]))
        R['variants'][cfg] = VAR
        pd.DataFrame({lab: {n: v[n]['Z'] for n in HEAD} for lab, v in VAR.items()}).T.to_csv(os.path.join(OUT, f'P052_variants_Z_{cfg}.csv'))
    R['runtime_s'] = time.time() - T0
    json.dump(R, open(os.path.join(OUT, 'P052_results_fit.json'), 'w'), indent=1, default=float)

# ==============================================================================================
elif ARGS.stage == 'sa':
    # semi-analytic single-event calibration: p0 ~ sum_c mu_c 1[q0(one event in cell c, rest = observed minus event) >= q0_obs]
    SA = {}
    for cfg in ['base', 'lzlike']:
        for n_model in ['L10_1000', 'O1s_1000_d350', 'O1s_1000_d300', 'O4s_1000', 'O1v_1000']:
            S, _, _ = sig_for(n_model, cfg); L = Likelihood(S, fix_lambdas=True); q_obs, _ = L.q0()
            D0 = DATA.copy(); D0[K_EV, J_EV] -= 1; mu_nom = L.mu_cells(dict(NOM)).reshape(NS1, ND)
            p_sa = 0.0; p_pos = 0.0; cells = []
            for k in range(NS1):
                for j in range(ND):
                    if S[k, j] <= 1e-9 or mu_nom[k, j] > 2.0: continue
                    D1 = D0.copy(); D1[k, j] += 1; L.set_data(D1, N_PROMPT, N_DELAYED); q, sh = L.q0()
                    cells.append((k, j, q, mu_nom[k, j], S[k, j]))
                    if q >= q_obs - 1e-9: p_sa += mu_nom[k, j]
                    if q > 0: p_pos += mu_nom[k, j]
            L.set_data(DATA, N_PROMPT, N_DELAYED)
            SA[f'{cfg}:{n_model}'] = dict(config=cfg, model=n_model, q0_obs=q_obs, Z_asym=Zof(q_obs), p0_semi_analytic=p_sa,
                                          Z_semi_analytic=float(stats.norm.isf(p_sa)) if 0 < p_sa < 1 else None, bkg_in_q0gt0_cells=p_pos, n_cells=len(cells))
            say(f'[{cfg}] {n_model:15s} q0_obs {q_obs:5.2f} Z_asym {Zof(q_obs):4.2f}  p0_SA {p_sa:.2e} -> Z_SA {SA[f"{cfg}:{n_model}"]["Z_semi_analytic"]}; bkg in q0>0 cells {p_pos:.3e}')
            pd.DataFrame(cells, columns=['k', 'j', 'q0', 'mu_bkg', 'sig_frac']).to_csv(os.path.join(OUT, f'P052_single_event_cells_{cfg}_{n_model}.csv'), index=False)
    json.dump(SA, open(os.path.join(OUT, 'P052_results_sa.json'), 'w'), indent=1, default=float)

# ==============================================================================================
elif ARGS.stage == 'toys':
    rng = np.random.default_rng(ARGS.seed)
    S, _, _ = sig_for(ARGS.model, ARGS.config); Lref = Likelihood(S, fix_lambdas=True)
    q_obs, sh_obs = Lref.q0()
    def gen_toy():
        p = dict(NOM); mu = Lref.mu_cells(p); n = rng.poisson(np.clip(mu, 0, None)); mp, md = Lref.mu_veto(p)
        aux = dict(NOM)
        for k, sg in CON.items(): aux[k] = NOM[k] + sg * rng.standard_normal()
        return n, rng.poisson(mp), rng.poisson(md), aux
    qs = np.zeros(ARGS.toys); shs = np.zeros(ARGS.toys); t0 = time.time()
    for i in range(ARGS.toys):
        n, n_p, n_d, aux = gen_toy()
        Lt = Likelihood(S, fix_lambdas=True, aux=aux); Lt.set_data(n.reshape(NS1, ND), n_p, n_d)
        qs[i], shs[i] = Lt.q0()
        if time.time() - t0 > 540: qs = qs[:i + 1]; shs = shs[:i + 1]; say(f'time cap reached after {i+1} toys'); break
    np.savez(os.path.join(TOYDIR, f'toys_{ARGS.config}_{ARGS.model}_seed{ARGS.seed}.npz'), q0=qs, s_hat=shs, q0_obs=q_obs, s_hat_obs=sh_obs)
    say(f'{ARGS.config} {ARGS.model}: q0_obs {q_obs:.3f}, {len(qs)} toys, exceed {int(np.sum(qs >= q_obs))}, P(q0>0) {np.mean(qs > 0):.4f}, {time.time()-t0:.0f}s')

# ==============================================================================================
elif ARGS.stage == 'final':
    Rf = json.load(open(os.path.join(OUT, 'P052_results_fit.json')))
    SA = json.load(open(os.path.join(OUT, 'P052_results_sa.json')))
    df = pd.read_csv(os.path.join(OUT, 'P052_data_fits_all_configs.csv'))
    TOYS = {}
    for f in sorted(glob.glob(os.path.join(TOYDIR, 'toys_*.npz'))):
        z = np.load(f); key = os.path.basename(f)[5:-4].rsplit('_seed', 1)[0]
        T = TOYS.setdefault(key, dict(q0=[], q0_obs=float(z['q0_obs']), files=[]))
        T['q0'].append(z['q0']); T['files'].append(os.path.basename(f))
    for key, T in TOYS.items():
        qs = np.concatenate(T['q0']); q_obs = T['q0_obs']; n = len(qs); n_exc = int(np.sum(qs >= q_obs))
        p_toy = (n_exc + 0.5) / (n + 1); p_lo, p_hi = stats.beta.ppf([0.16, 0.84], n_exc + 0.5, n - n_exc + 0.5)
        cfg, model = key.split('_', 1)
        T.update(n_toys=n, n_exceed=n_exc, p_toy=float(p_toy), p_toy_68=[float(p_lo), float(p_hi)], Z_toy=float(stats.norm.isf(p_toy)),
                 Z_toy_68=[float(stats.norm.isf(p_hi)), float(stats.norm.isf(p_lo))], Z_asym=Zof(q_obs), p_chernoff=float(0.5 * stats.chi2.sf(q_obs, 1)),
                 frac_q0_gt0=float(np.mean(qs > 0)), q0_quantiles={q: float(np.quantile(qs, float(q))) for q in ('0.9', '0.99', '0.999')},
                 Z_LZ=float(MODEL_BY_NAME[model][4]), config=cfg, model=model)
        # effective 'p(q0>0)-corrected Chernoff': p = P(q0>0) * P(chi2_1 >= q | q>0)  -- test of the mixture form
        pos = qs[qs > 0]; T['p_conditional_tail_toys'] = float(np.mean(pos >= q_obs)) if len(pos) else None
        T['Z_upper_bound_from_frac_pos'] = float(stats.norm.isf(T['frac_q0_gt0']))
        del T['q0']
        say(f'{key:24s} n {n:6d} exceed {n_exc:3d} p {p_toy:.2e} Z_toy {T["Z_toy"]:.2f} [{T["Z_toy_68"][0]:.2f},{T["Z_toy_68"][1]:.2f}]  Z_asym {T["Z_asym"]:.2f}  LZ {T["Z_LZ"]}  P(q0>0) {T["frac_q0_gt0"]:.4f}')
    Rf['toys'] = TOYS; Rf['semi_analytic'] = SA
    Rf['inputs'] = dict(S1_edges=S1_EDGES.tolist(), d_edges=[-8, 8, 0.5], event_d=EV_D_NOMINAL, mid_panel_data=N_MID_DATA, constraints=CON, nominal=NOM,
                        R_DET=R_DET, R_XE=R_XE, R_MSSI=R_MSSI, PROMPT_SMALL_ER=PROMPT_SMALL_ER, DELAYED_BKG=DELAYED_BKG, DELAYED_SIG=DELAYED_SIG, ER_SIG=ER_SIG,
                        lambda_nu_phd=LAM_NU, lambda_neutron_phd=LAM_N, lambda_acc_phd=LAM_ACC, MSSI_panel_split=MSSI_PANEL.tolist(), sig_split=SIG_SPLIT, configs=CONFIGS)
    # ---------------- figures
    C = {'L10 (m scan)': '#2a78d6', 'elastic operators': '#eb6834', 'inelastic O1 (s/v)': '#1baf7a'}
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.6), dpi=160, sharey=True)
    for ax, cfg, ttl in ((axs[0], 'base', 'NEST-only signal model (Gaussian d, σ_S1c = 1.13√S1c)'), (axs[1], 'lzlike', 'LZ-like signal model (Fig. 5 d-shape, σ_E = 23 keV)')):
        sub = df[df.config == cfg]
        ax.fill_between([0, 3.8], [-0.3, 3.5], [0.3, 4.1], color='#f0efec', zorder=0)
        ax.plot([0, 3.8], [0, 3.8], color='#898781', lw=1, ls='--', zorder=1)
        for cls, col in C.items():
            s2 = sub[sub.cls == cls]; ax.scatter(s2.Z_LZ, s2.Z, s=26, color=col, label=f'{cls} (n={len(s2)})', zorder=3, edgecolor='white', linewidth=0.6)
        for key, T in TOYS.items():
            if T['config'] != cfg: continue
            ax.errorbar(T['Z_LZ'], T['Z_toy'], yerr=[[T['Z_toy'] - T['Z_toy_68'][0]], [T['Z_toy_68'][1] - T['Z_toy']]], fmt='D', ms=6, color='#0b0b0b', mfc='none', zorder=4, lw=1)
        ax.errorbar([], [], fmt='D', color='#0b0b0b', mfc='none', label='toy-calibrated Z')
        st = Rf['fits_vs_LZ'][cfg]['all']
        ax.set_title(f'{ttl}\nrms {st["rms"]:.2f}σ, mean {st["mean"]:+.2f}σ (asymptotic)', fontsize=8.5)
        ax.set_xlabel('LZ local significance, Tables S6/S7 [σ]'); ax.set_xlim(-0.1, 3.8); ax.set_ylim(-0.1, 4.1)
        for s in ('top', 'right'): ax.spines[s].set_visible(False)
    axs[0].set_ylabel('this work, Z [σ]'); axs[0].legend(fontsize=7.5, loc='upper left', frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P052_fig1_Z_scatter.png')); plt.close(fig)

    key = 'lzlike_L10_1000' if 'lzlike_L10_1000' in TOYS else list(TOYS)[0]
    qs = np.concatenate([np.load(os.path.join(TOYDIR, f))['q0'] for f in TOYS[key]['files']]); T = TOYS[key]
    fig, ax = plt.subplots(figsize=(5.6, 4.0), dpi=160)
    x = np.linspace(0, 16, 400); srt = np.sort(qs); sf = 1.0 - np.arange(1, len(srt) + 1) / len(srt)
    ax.step(srt, np.clip(sf, 1.0 / len(srt), None), where='post', color='#2a78d6', lw=2, label=f'background-only toys, L10 1000 GeV (n = {len(qs)})')
    ax.plot(x, 0.5 * stats.chi2.sf(x, 1), color='#eb6834', lw=2, ls='--', label='Chernoff: ½ χ²₁ survival')
    ax.plot(x, T['frac_q0_gt0'] * stats.chi2.sf(x, 1), color='#1baf7a', lw=1.5, ls='-.', label=f'P(q₀>0)={T["frac_q0_gt0"]:.3f} × χ²₁ survival')
    ax.axvline(T['q0_obs'], color='#0b0b0b', lw=1, ls=':'); ax.text(T['q0_obs'] + 0.2, 0.25, f'q₀(data) = {T["q0_obs"]:.1f}', fontsize=8)
    ax.set_yscale('log'); ax.set_ylim(2e-5, 1.2); ax.set_xlim(0, 16); ax.set_xlabel('q₀'); ax.set_ylabel('P(q₀ ≥ x | background only)')
    ax.set_title(f'p₀: toys {T["p_toy"]:.1e} (Z = {T["Z_toy"]:.2f}), asymptotic {T["p_chernoff"]:.1e} (Z = {T["Z_asym"]:.2f}); LZ 3.4σ', fontsize=8.5)
    ax.legend(fontsize=7.5, frameon=False, loc='lower left')
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P052_fig2_q0_distribution.png')); plt.close(fig)

    # figure 3: model map at S1c > 250 (background vs L10 signal), event marked
    S_L10, _, _ = sig_for('L10_1000', 'lzlike'); sh = Rf['L10_1000']['lzlike']['s_hat']
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.9), dpi=160, sharey=True)
    mu_b = Likelihood(S_L10).mu_cells(dict(NOM)).reshape(NS1, ND)
    for ax, M, ttl in ((axs[0], mu_b, 'background model (nominal)'), (axs[1], S_L10 * sh, f'L10 1000 GeV signal, ŝ = {sh:.2f} events')):
        im = ax.pcolormesh(S1_EDGES[1:], D_EDGES, np.log10(np.clip(M[1:], 1e-7, None)).T, cmap='Blues', vmin=-7, vmax=0)
        ax.plot([EV_S1C], [EV_D_NOMINAL], marker='*', color='#e34948', ms=13, mec='#0b0b0b'); ax.set_xlabel('S1c [phd]'); ax.set_title(ttl, fontsize=9)
    axs[0].set_ylabel('d = (log₁₀S2c − μ_NR)/σ_NR'); cb = fig.colorbar(im, ax=axs, fraction=0.03, pad=0.02); cb.set_label('log₁₀ events per 25 phd × 0.5σ cell')
    fig.savefig(os.path.join(FIG, 'P052_fig3_model_map.png'), bbox_inches='tight'); plt.close(fig)

    # figure 4: signal d-shape, Fig. 5 digitised vs Gaussian
    fig, ax = plt.subplots(figsize=(5.4, 3.8), dpi=160)
    ax.step(D_EDGES, np.append(GAUSS, GAUSS[-1]), where='post', color='#898781', lw=1.5, label='N(0,1) per 0.5σ bin')
    for p, col, lab in ((1, '#2a78d6', 'Fig. 5 L10 curve, 250<S1c<500'), (2, '#eb6834', 'Fig. 5 L10 curve, S1c>500')):
        v = DSHAPE_FIG5[p]; ax.step(D_EDGES, np.append(v, v[-1]), where='post', color=col, lw=2, label=lab)
    ax.axvline(EV_D_NOMINAL, color='#0b0b0b', ls=':', lw=1); ax.text(EV_D_NOMINAL - 0.1, 0.12, 'event', fontsize=8, ha='right')
    ax.set_yscale('log'); ax.set_ylim(1e-4, 0.4); ax.set_xlim(-6, 5); ax.set_xlabel('d'); ax.set_ylabel('signal fraction per 0.5σ bin'); ax.legend(fontsize=7.5, frameon=False)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P052_fig4_signal_dshape.png')); plt.close(fig)
    Rf['runtime_final_s'] = time.time() - T0
    json.dump(Rf, open(os.path.join(OUT, 'P052_results.json'), 'w'), indent=1, default=float)
    say('final results written')
say(f'stage {ARGS.stage} done in {time.time() - T0:.0f} s')
