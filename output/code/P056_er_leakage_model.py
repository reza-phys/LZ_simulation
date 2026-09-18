"""
P056 -- ER/NR discrimination at 250 keV: the leakage predicted by the double-skew-Gaussian recombination model
        and how far the tails must be trusted.

Run from the simulation root:  .venv/bin/python output/code/P056_er_leakage_model.py     (~2-4 min)
Outputs: output/work/P056/*.csv, results.json, figures/*.png

Model (all stated in details.md):
  * ER mean yields: NEST v2 beta model with LZ Table S3 parameters via lz.nest_er_yields (nestpy 2.1.1);
    exciton/ion ratio alpha_ex(E) from the same nestpy YieldResult (0.182 above ~40 keV).
  * Quanta: Nq ~ N(mu_q, sqrt(mu_q)) (Fano 1.0, measured from nestpy GetQuanta); Ni | Nq ~ N(Nq/(1+a), sqrt(Nq a)/(1+a)).
  * Recombination fluctuation: Var(Ne | Ni) = r(1-r) Ni + (k sigma_p(Nq) Ni)^2 with sigma_p the Table S4 double
    skew-Gaussian in x = log10 Nq; k is a calibration factor (k = 1 is the literal NEST GetQuanta combination
    sigma_p * Ni; the drawn band and the 212Pb scatter require k ~ 0.5, cf. P010).
  * Shape of Ne | Ni: skew-normal with shape parameter a = s * a_NEST(E), where a_NEST(E) is the NEST v2.4.0 ER
    skewness measured from nestpy GetQuanta samples (positive: tail towards MORE electrons); s = 0 Gaussian,
    s = 1 NEST, s = 0.5 / 1.5 sensitivity, s = -1 mirrored (heavy low-Ne tail toy).
  * Detector (as P024): S1c = Binomial(Nph, g1) with SPE resolution 0.338 and a 2 % position residual;
    S2c = Binomial(Ne, eps = 0.80) * g2/eps with SE Fano 4 and a 2 % position residual (Gaussian approximations).
  * Everything is integrated deterministically on a (S1c, log10 S2c) grid, so tail probabilities down to 1e-15 are
    numerically exact for the stated model; a Monte Carlo cross-checks the band percentiles.
"""
from __future__ import annotations
import sys, os, json, math, time
sys.path.insert(0, 'output/code')
import numpy as np
import pandas as pd
from scipy import stats, special, optimize, ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from common import lzcommon as lz
import nestpy

T0 = time.time()
OUT = 'output/work/P056'; FIG = f'{OUT}/figures'
os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(56)
nestpy.RandomGen.rndm().set_seed(56)
R = {}
def log(*a):
    print(*a, flush=True)

G1, G2 = lz.LZ['g1'], lz.LZ['g2']
S1_EV, S2_EV = lz.LZ['ev_S1c'], lz.LZ['ev_S2c']; L2_EV = math.log10(S2_EV)
ROI_TOP = lz.LZ['log10S2c_max']           # 4.15
DEX = 1 / math.log(10)
SPE_RES, EXT_EFF, S2_FANO, POS_RES = 0.338, 0.80, 4.0, 0.02   # P024 detector model (SPE res from nestpy LZ_WS2024)
P_S4 = dict(lz.NEST_ER_FLUCT_LZ)

# =====================================================================================================
# 1. NEST ER means (Table S3) and the NEST ER skewness, from nestpy
# =====================================================================================================
DET = nestpy.detectors.LZ_WS2024(); NC = nestpy.NESTcalc(DET)
ER_PARS = [lz.NEST_ER_LZ[f'm{i}'] for i in range(1, 11)]
W_LZ = list(DET.nr_er_width_parameters)

def er_yield(E):
    return NC.GetYields(nestpy.interactions.beta, float(E), 2.9, lz.DRIFT_FIELD_VCM, 131.293, 54,
                        list(nestpy.default_nr_parameters), ER_PARS)

E_TAB = np.arange(1.0, 201.0, 1.0)
_y = [er_yield(E) for E in E_TAB]
NPH_TAB = np.array([y.PhotonYield for y in _y]); NE_TAB = np.array([y.ElectronYield for y in _y])
NQ_TAB = NPH_TAB + NE_TAB; ALPHA_TAB = np.array([y.ExcitonRatio for y in _y])
W_NEST = 1e3 * E_TAB[63] / NQ_TAB[63]        # eV per quantum at 64 keV
def er_means(E):
    E = np.atleast_1d(E).astype(float)
    return (np.interp(E, E_TAB, NPH_TAB), np.interp(E, E_TAB, NE_TAB), np.interp(E, E_TAB, NQ_TAB), np.interp(E, E_TAB, ALPHA_TAB))
def er_means_s(E):
    return tuple(float(v[0]) for v in er_means(E))

# NEST ER skewness (nestpy GetQuanta, LZ_WS2024 widths, default SkewnessER formula): sample skewness -> shape a
def skew_to_shape(g):
    g = float(np.clip(g, -0.99, 0.99)); s = np.sign(g); g = abs(g)
    if g < 1e-4: return 0.0
    c = (4 - math.pi) / 2
    fn = lambda d: c * (d * math.sqrt(2 / math.pi)) ** 3 / (1 - 2 * d * d / math.pi) ** 1.5 - g
    d = optimize.brentq(fn, 1e-6, 0.99999)
    return s * d / math.sqrt(1 - d * d)

def nestpy_samples(E, n=40000, W=None, sk=-999.0):
    y = er_yield(E); W = W_LZ if W is None else W
    q = [NC.GetQuanta(y, 2.9, W, sk) for _ in range(n)]
    return np.array([x.electrons for x in q], float), np.array([x.ions for x in q], float), np.array([x.photons for x in q], float)

E_SK = np.array([3, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 160])
sk_rows = []
for E in E_SK:
    ne, ni, nph = nestpy_samples(E)
    W0 = list(W_LZ); W0[7] = 0.0                       # ER omega amplitude -> 0: binomial + ion-Fano only
    ne0, _, _ = nestpy_samples(E, n=20000, W=W0)
    g = float(stats.skew(ne)); nq = ne + nph
    yy = er_yield(E); r = 1 - yy.ElectronYield / (yy.ElectronYield + yy.PhotonYield) * (1 + yy.ExcitonRatio)
    sk_rows.append(dict(E_keV=E, Nq=nq.mean(), Ni=ni.mean(), sd_Ni=ni.std(), fano_Nq=nq.var() / nq.mean(), r=r,
                        sd_Ne_nestpy_LZWS2024=ne.std(), sd_Ne_omega0=ne0.std(), sample_skew=g, shape_a=skew_to_shape(g),
                        eff_sigma_p_nestpy=math.sqrt(max(ne.std() ** 2 - ne0.std() ** 2, 0)) / ni.mean(),
                        sigma_p_TableS4=0.0))
SK = pd.DataFrame(sk_rows)
def sigma_p(nq, P=P_S4):
    x = np.log10(np.asarray(nq, float)); tot = 0.0
    for i in (1, 2):
        A, mu, sg, al = P[f'A{i}'], P[f'mu{i}'], P[f'sigma{i}'], P[f'alpha{i}']
        tot = tot + A * (1 + special.erf(al * (x - mu) / (math.sqrt(2) * sg))) * np.exp(-(x - mu) ** 2 / (2 * sg ** 2))
    return tot
SK['sigma_p_TableS4'] = sigma_p(SK['Nq'].values)
SK['sd_Ne_TableS4_k1'] = np.sqrt(SK['r'] * (1 - SK['r']) * SK['Ni'] + (SK['sigma_p_TableS4'] * SK['Ni']) ** 2)
SK.to_csv(f'{OUT}/nest_skewness_and_widths.csv', index=False)
log('[1] nestpy ER fluctuation table:\n', SK[['E_keV', 'Nq', 'Ni', 'r', 'sd_Ne_nestpy_LZWS2024', 'sd_Ne_omega0', 'sample_skew', 'shape_a', 'eff_sigma_p_nestpy', 'sigma_p_TableS4', 'sd_Ne_TableS4_k1']].round(4).to_string(index=False))
# shape a(E): interpolate; NEST sets skewness = 0 when Nq > 1e4 (seen as ~0 above 130 keV)
def a_nest(E):
    a = np.interp(np.atleast_1d(E), SK['E_keV'].values, SK['shape_a'].values)
    nq = er_means(E)[2]
    return np.where(nq > 1e4, 0.0, a)      # NEST: skewness set to 0 for Nq > 1e4 (nestpy shows ~0 above 130 keV)
R['nest_fluct_table'] = SK.to_dict(orient='records')
R['W_eV_nestpy'] = W_NEST

# =====================================================================================================
# 2. Deterministic response engine on a (S1c, log10 S2c) grid
# =====================================================================================================
S1_EDGES = np.arange(240.0, 660.0 + 0.1, 10.0)          # 10-phd columns 240-660
S1_C = 0.5 * (S1_EDGES[1:] + S1_EDGES[:-1])
L2_EDGES = np.arange(3.60, 5.20 + 1e-9, 0.0025)         # 0.0025-dex rows (covers the whole ER band and the NR band)
L2_C = 0.5 * (L2_EDGES[1:] + L2_EDGES[:-1])
E_GRID = np.arange(25.0, 160.0 + 0.1, 2.0)               # ER energies feeding S1c 240-660 (2-keV steps; sums x DE)
DE = float(E_GRID[1] - E_GRID[0])
NDTR = special.ndtr
def npdf(z): return np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)

def skewnorm_pdf_grid(ne, mean, sd, a):
    """skew-normal pdf with given mean, sd, shape a (a = 0 -> Gaussian)."""
    if abs(a) < 1e-9:
        return npdf((ne - mean) / sd) / sd
    d = a / math.sqrt(1 + a * a)
    om = sd / math.sqrt(1 - 2 * d * d / math.pi)
    xi = mean - om * d * math.sqrt(2 / math.pi)
    z = (ne - xi) / om
    return 2 / om * npdf(z) * NDTR(a * z)

GH_X, GH_W = np.polynomial.hermite_e.hermegauss(3)      # probabilists' Hermite: 3 nodes for N(0,1) (Nq and Ni fluctuations
GH_W = GH_W / GH_W.sum()                                  # are small compared with the recombination term; checked vs MC)

def response_at_E(E, k=0.5, s=1.0, f=0.0, P=P_S4, s1_edges=S1_EDGES, l2_edges=L2_EDGES, n_ne=200, alpha_fix=None, a_fix=None):
    """P(S1c bin, log10 S2c bin | ER of energy E) for the model (k, s, sigma_p params P, charge suppression f).
    Returns array [n_S1 bins, n_L2 bins]."""
    nph_m, ne_m, nq_m, al = er_means_s(E)
    if alpha_fix is not None: al = alpha_fix
    a = float(a_nest(E)[0]) * s if a_fix is None else a_fix
    ni_m = nq_m / (1 + al); r = 1 - ne_m / ni_m
    H = np.zeros((len(s1_edges) - 1, len(l2_edges) - 1))
    for xq, wq in zip(GH_X, GH_W):
        nq = nq_m + xq * math.sqrt(nq_m)
        sp = float(sigma_p(nq, P))
        ni_c = nq / (1 + al); sd_ni = math.sqrt(nq * al) / (1 + al)
        for xi_, wi in zip(GH_X, GH_W):
            ni = ni_c + xi_ * sd_ni
            ne_mean = (1 - f) * (1 - r) * ni
            sd = math.sqrt(max(r * (1 - r) * ni, 0) + (k * sp * ni) ** 2)
            ne = np.linspace(max(1.0, ne_mean - 14 * sd), min(ni - 1, ne_mean + 14 * sd), n_ne)
            pdf = skewnorm_pdf_grid(ne, ne_mean, sd, a); dne = ne[1] - ne[0]
            w = pdf * dne
            nph = nq - ne
            mu1 = G1 * nph; sd1 = np.sqrt(G1 * nph * (1 - G1 + SPE_RES ** 2) + (POS_RES * mu1) ** 2)
            P1 = np.diff(NDTR((s1_edges[None, :] - mu1[:, None]) / sd1[:, None]), axis=1)        # [n_ne, nS1]
            mu2 = np.log10(G2 * ne); sd2 = DEX * np.sqrt((1 - EXT_EFF) / (EXT_EFF * ne) + S2_FANO / (G2 * ne) + POS_RES ** 2)
            P2 = np.diff(NDTR((l2_edges[None, :] - mu2[:, None]) / sd2[:, None]), axis=1)        # [n_ne, nL2]
            H += wq * wi * (P1 * w[:, None]).T @ P2
    return H

def response_flat(E_grid, **kw):
    """Sum over a flat spectrum (per keV): returns array [nE, nS1, nL2]."""
    return np.array([response_at_E(E, **kw) for E in E_grid])

# ---- Monte Carlo cross-check of the engine (band percentiles at fixed S1c) ----
def band_mc(k=0.5, s=1.0, n=1_500_000, E_lo=25.0, E_hi=160.0):
    E = rng.uniform(E_lo, E_hi, n)
    nph_m, ne_m, nq_m, al = er_means(E)
    nq = rng.normal(nq_m, np.sqrt(nq_m))
    ni = rng.normal(nq / (1 + al), np.sqrt(nq * al) / (1 + al))
    r = 1 - ne_m / (nq_m / (1 + al))
    sd = np.sqrt(np.clip(r * (1 - r) * ni, 0, None) + (k * sigma_p(nq) * ni) ** 2)
    a = a_nest(E) * s
    d = a / np.sqrt(1 + a * a); om = sd / np.sqrt(1 - 2 * d * d / np.pi); xi = (1 - r) * ni - om * d * np.sqrt(2 / np.pi)
    u0 = rng.normal(size=n); u1 = rng.normal(size=n)
    z = np.where(u1 < a * u0, u0, -u0)                     # skew-normal sampling (Azzalini): u0 | u1 < a u0
    z = np.where(np.abs(a) < 1e-9, u0, z)
    ne = xi + om * z
    ne = np.clip(ne, 1, ni - 1); nph = nq - ne
    s1 = rng.binomial(np.round(nph).astype(int), G1).astype(float)
    s1 = s1 * (1 + rng.normal(0, SPE_RES / np.sqrt(np.clip(s1, 1, None)))) * (1 + rng.normal(0, POS_RES, n))
    next_ = rng.binomial(np.round(ne).astype(int), EXT_EFF).astype(float)
    s2 = next_ * G2 / EXT_EFF; s2 = s2 * (1 + rng.normal(0, np.sqrt(S2_FANO / np.clip(s2, 1, None)))) * (1 + rng.normal(0, POS_RES, n))
    return E, s1, np.log10(np.clip(s2, 1, None))

def percentiles_from_H(Hsum, s1c0, half=10.0):
    cols = np.where((S1_C > s1c0 - half) & (S1_C < s1c0 + half))[0]
    p = Hsum[cols].sum(axis=0); c = np.cumsum(p) / p.sum()
    q = np.interp([0.10, 0.50, 0.90], c, L2_EDGES[1:])
    return dict(p10=q[0], p50=q[1], p90=q[2], halfwidth=(q[2] - q[0]) / 2, upper=(q[2] - q[1]), lower=(q[1] - q[0]))

# =====================================================================================================
# 3. Digitisation: Fig. 4 ER/NR band lines; Fig. 2 212Pb points; Fig. 5 histograms
# =====================================================================================================
def load_png(name):
    return (mpimg.imread(f'inputs/figures_png/{name}.png')[:, :, :3] * 255).astype(int)

def frame(rgb):
    dark = rgb.sum(axis=2) < 150; H_, W_ = dark.shape
    cols = np.where(dark.sum(axis=0) > 0.5 * H_)[0]; rows = np.where(dark.sum(axis=1) > 0.5 * W_)[0]
    return cols.min(), cols.max(), rows.min(), rows.max()

def major_yticks(rgb, xr, yt, yb):
    dark = rgb.sum(axis=2) < 200; runs = np.zeros(rgb.shape[0], int)
    for y in range(yt + 1, yb):
        x = xr - 1
        while x > xr - 40 and dark[y, x]: x -= 1
        runs[y] = (xr - 1) - x
    rows = np.where(runs >= 18)[0]; rows = rows[(rows > yt + 3) & (rows < yb - 3)]
    groups, cur = [], [rows[0]]
    for yy in rows[1:]:
        if yy - cur[-1] <= 2: cur.append(yy)
        else: groups.append(cur); cur = [yy]
    groups.append(cur)
    return np.array([np.mean(g) for g in groups])

def calib_s1l2(name):
    """axis calibration of Figs 2/4 (P024 method): x 0-800 phd across the frame, y from major ticks + frame top = 5.0."""
    rgb = load_png(name); xl, xr, yt, yb = frame(rgb)
    ticks = major_yticks(rgb, xr, yt, yb); guess = (yb - yt) / 2.25
    vals = 5.0 - np.round((ticks - yt) / guess * 2) / 2
    ticks = np.append(ticks, yt); vals = np.append(vals, 5.0)
    slope, inter = np.polyfit(ticks, vals, 1)
    x_of = lambda s1: xl + s1 / 800.0 * (xr - xl); s1_of = lambda x: (x - xl) / (xr - xl) * 800.0
    l_of = lambda y: slope * y + inter
    return rgb, dict(xl=int(xl), xr=int(xr), yt=int(yt), yb=int(yb), px_per_dex=float(-1 / slope), resid=float(np.abs(vals - (slope * ticks + inter)).max())), x_of, s1_of, l_of

def line_clusters(mask, x_of, l_of, s1, half=5, min_px=3):
    x0 = int(round(x_of(s1))); ys = []
    for x in range(x0 - half, x0 + half + 1):
        ys.extend(np.where(mask[:, x])[0].tolist())
    ys = np.sort(np.array(ys))
    if ys.size == 0: return []
    groups, cur = [], [ys[0]]
    for yy in ys[1:]:
        if yy - cur[-1] <= 2: cur.append(yy)
        else: groups.append(cur); cur = [yy]
    groups.append(cur)
    return sorted([float(l_of(np.mean(g))) for g in groups if len(g) >= min_px])

rgb4, cal4, x4, s1_4, l4 = calib_s1l2('Fig4_science_sample_wbands')
blue4 = (rgb4[:, :, 2] > 90) & (rgb4[:, :, 0] < 60) & (rgb4[:, :, 1] < 60)
red4 = (rgb4[:, :, 0] > 170) & (rgb4[:, :, 1] < 90) & (rgb4[:, :, 2] < 120)
rgb2, cal2, x2, s1_2, l2_ = calib_s1l2('Fig2_Calibrations')
blue2 = (rgb2[:, :, 2] > 90) & (rgb2[:, :, 0] < 60) & (rgb2[:, :, 1] < 60)
red2 = (rgb2[:, :, 0] > 170) & (rgb2[:, :, 1] < 90) & (rgb2[:, :, 2] < 120)
dig_rows = []
for s1 in np.arange(250, 801, 10):
    for fig, msk_b, msk_r, xo, lo in (('Fig4', blue4, red4, x4, l4), ('Fig2', blue2, red2, x2, l2_)):
        b = line_clusters(msk_b, xo, lo, s1); r_ = line_clusters(msk_r, xo, lo, s1)
        row = dict(fig=fig, S1c=s1)
        if len(b) == 3: row.update(ER_p10=b[0], ER_p50=b[1], ER_p90=b[2])
        elif len(b) == 2 and fig == 'Fig2' and s1 > 780: pass
        if len(r_) == 3: row.update(NR_p10=r_[0], NR_p50=r_[1], NR_p90=r_[2])
        elif len(r_) == 2 and abs(s1 - 540) <= 20 and fig == 'Fig4':   # lower NR line hidden by the event marker
            row.update(NR_p50=r_[0], NR_p90=r_[1])
        dig_rows.append(row)
DIG = pd.DataFrame(dig_rows)
DIG['ER_halfwidth'] = (DIG['ER_p90'] - DIG['ER_p10']) / 2; DIG['ER_upper'] = DIG['ER_p90'] - DIG['ER_p50']; DIG['ER_lower'] = DIG['ER_p50'] - DIG['ER_p10']
DIG['NR_halfwidth'] = (DIG['NR_p90'] - DIG['NR_p10']) / 2
DIG.to_csv(f'{OUT}/digitised_bands_fig2_fig4.csv', index=False)
D4 = DIG[DIG.fig == 'Fig4'].set_index('S1c'); D2 = DIG[DIG.fig == 'Fig2'].set_index('S1c')
log('[3] Fig4 calibration', cal4, '\n    Fig2 calibration', cal2)
log(D4.loc[[300, 400, 500, 540, 600, 700, 800], ['ER_p10', 'ER_p50', 'ER_p90', 'ER_halfwidth', 'ER_upper', 'ER_lower', 'NR_p10', 'NR_p50', 'NR_p90']].round(4).to_string())
# NR median and Fig.-5 sigma unit (drawn 10-90 % half-width) as functions of S1c (Fig. 4 digitisation, gaps interpolated)
ok = D4['NR_p50'].notna()
MU_NR = lambda s1: np.interp(s1, D4.index[ok].values.astype(float), D4['NR_p50'][ok].values)
okh = D4['NR_halfwidth'].notna()
HW_NR = lambda s1: np.interp(s1, D4.index[okh].values.astype(float), D4['NR_halfwidth'][okh].values)
ev_hw = (MU_NR(S1_EV) - L2_EV) / HW_NR(S1_EV)
R['fig4_digitised'] = dict(calibration=cal4, NR_median_540=float(MU_NR(S1_EV)), NR_halfwidth_540=float(HW_NR(S1_EV)),
                           event_depth_dex=float(MU_NR(S1_EV) - L2_EV), event_in_halfwidths=float(ev_hw),
                           ER_median_540=float(np.interp(540, D4.index[D4.ER_p50.notna()].astype(float), D4.ER_p50.dropna())),
                           ER_halfwidth_540=float(np.interp(540, D4.index[D4.ER_halfwidth.notna()].astype(float), D4.ER_halfwidth.dropna())))
log(f"    NR median at 540: {MU_NR(S1_EV):.4f}, drawn half-width {HW_NR(S1_EV):.4f} dex; event {MU_NR(S1_EV)-L2_EV:.4f} dex below = {ev_hw:.2f} half-widths (Fig. 5 shows it in the -2..-1.5 bin)")
EV_DEPTH = float(MU_NR(S1_EV) - L2_EV)   # the event's depth below the NR median; LZ calls this 1.5 sigma_NR

# ---- Fig. 2 212Pb (green) points at S1c 400-700: residuals about the drawn ER median ----
green = (np.abs(rgb2[:, :, 0] - 46) < 25) & (np.abs(rgb2[:, :, 1] - 139) < 25) & (np.abs(rgb2[:, :, 2] - 87) < 25)
grey = (np.abs(rgb2[:, :, 0] - 150) < 12) & (np.abs(rgb2[:, :, 1] - 166) < 12) & (np.abs(rgb2[:, :, 2] - 166) < 12)
lab, nlab = ndimage.label(green)
cent = np.array(ndimage.center_of_mass(green, lab, range(1, nlab + 1))); sizes = np.array(ndimage.sum(green, lab, range(1, nlab + 1)))
s1g = s1_2(cent[:, 1]); l2g = l2_(cent[:, 0])
sparse = (s1g > 550) & (s1g < 800)
single_px = float(np.median(sizes[sparse])) if sparse.sum() > 5 else float(np.median(sizes))
n_est = np.maximum(1, np.round(sizes / single_px))
er50_2 = D2['ER_p50'].dropna(); er_hw_2 = D2['ER_halfwidth'].dropna()
med_of = lambda s: np.interp(s, er50_2.index.values.astype(float), er50_2.values)
hw_of = lambda s: np.interp(s, er_hw_2.index.values.astype(float), er_hw_2.values)
resid = l2g - med_of(s1g); zres = resid / hw_of(s1g)
def wquant(x, w, q):
    o = np.argsort(x); c = np.cumsum(w[o]) / w[o].sum(); return float(np.interp(q, c, x[o]))
PB = {}
# multiplicity-weighted statistics (overlapping markers merge into large blobs: each blob is counted n_est times at its
# centroid; singles dominate the tails, so tail counts are exact while the core is approximate).  The plot top
# (log10 S2c = 5.0) clips the upper side above S1c ~ 560, hence the 450-560 window is the unbiased one.
for lo, hi in ((450, 560), (450, 550), (500, 600), (600, 800)):
    mm = (s1g >= lo) & (s1g < hi) & (l2g > 4.0); w = n_est[mm]; z = zres[mm]; rr = resid[mm]
    mean_w = float(np.average(rr, weights=w)); sd_w = float(np.sqrt(np.average((rr - mean_w) ** 2, weights=w)))
    PB[f'{lo}_{hi}'] = dict(n_blobs=int(mm.sum()), n_est_all=int(w.sum()), n_singles=int((n_est[mm] == 1).sum()),
                            mean_resid_dex=mean_w, sd_dex_weighted=sd_w,
                            halfwidth_1090_weighted=0.5 * (wquant(rr, w, 0.9) - wquant(rr, w, 0.1)),
                            p10_p90_weighted=[wquant(rr, w, 0.1), wquant(rr, w, 0.9)],
                            min_z=float(z.min()), max_z=float(z.max()),
                            n_below_m2hw=int(w[z < -2].sum()), n_below_m3hw=int(w[z < -3].sum()), n_above_p2hw=int(w[z > 2].sum()),
                            frac_below_m2hw=float(w[z < -2].sum() / w.sum()), frac_below_m3hw=float(w[z < -3].sum() / w.sum()),
                            gauss_expect_below_m2hw=float(stats.norm.cdf(-2 * 1.2816)), gauss_expect_below_m3hw=float(stats.norm.cdf(-3 * 1.2816)),
                            drawn_halfwidth_mean=float(hw_of(0.5 * (lo + hi))), single_marker_px=single_px)
# grey points (LZ: > 4 sigma from the band mean, excluded as wall leakage) below the ER band at S1c 450-650
labg, nlg = ndimage.label(grey)
cg = np.array(ndimage.center_of_mass(grey, labg, range(1, nlg + 1))) if nlg else np.zeros((0, 2))
s1gr = s1_2(cg[:, 1]); l2gr = l2_(cg[:, 0])
mg = (s1gr >= 450) & (s1gr < 650)
PB['grey_450_650'] = dict(n=int(mg.sum()), n_between_NR_and_ER=int((mg & (l2gr > MU_NR(np.clip(s1gr, 250, 800)) + 0.1) & (l2gr < med_of(s1gr) - 4 * hw_of(s1gr))).sum()),
                          n_in_NR_pm3hw=int((mg & (np.abs(l2gr - MU_NR(np.clip(s1gr, 250, 800))) < 3 * HW_NR(np.clip(s1gr, 250, 800)))).sum()))
R['pb212_fig2'] = PB
pd.DataFrame(dict(S1c=s1g, log10S2c=l2g, resid_dex=resid, z_halfwidths=zres, blob_px=sizes, n_est=n_est)).to_csv(f'{OUT}/pb212_points_fig2.csv', index=False)
log('    212Pb points (Fig. 2):', json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in PB.items()}, indent=0))

# ---- Fig. 5: light-blue (continuous ERs), magenta (internal gamma+IC+EC), total, in all three panels ----
def digitise_fig5():
    im = load_png('Fig5_NR_distance_3panels_no_sig')
    dark = im.sum(axis=2) < 150
    rows_dark = np.where(dark[:, 300:2300].mean(axis=1) > 0.9)[0]
    # group frame lines
    groups, cur = [], [rows_dark[0]]
    for yy in rows_dark[1:]:
        if yy - cur[-1] <= 3: cur.append(yy)
        else: groups.append(cur); cur = [yy]
    groups.append(cur)
    lines = [int(np.mean(g)) for g in groups]
    panels = [(lines[i], lines[i + 1]) for i in range(len(lines) - 1) if lines[i + 1] - lines[i] > 400]
    x_l, x_r = 219, 2409
    colours = {'ERs': (0, 194, 249), 'Internal': (255, 0, 255), 'Total': (0, 0, 255), 'MSSI': (2, 81, 128), 'Accidentals': (255, 220, 61), 'NRs': (113, 170, 52)}
    # bottom-of-frame values and major-tick spacing (decades) per panel, from the axis labels of Fig. 5
    # frame-bottom values from the axis labels of Fig. 5: top panel 1e-3, middle 1e-5, bottom 1e-6.  Major ticks are
    # one decade apart in the top/middle panels (labels every other tick) and two decades apart in the bottom panel;
    # the minor-tick pattern (8 vs 17 minors between majors) fixes this: spacing < 150 px -> 1 decade, else 2.
    bottoms = {0: -3.0, 1: -5.0, 2: -6.0}
    edges = np.arange(-8, 8.01, 0.5); out = {}
    for ip, (yt, yb) in enumerate(panels):
        tk = major_yticks(im, x_r, yt, yb)             # major ticks on the RIGHT frame edge (histograms start on the left one)
        spacing = float(np.median(np.diff(tk))); dec_per_tick = 1.0 if spacing < 150 else 2.0; px_per_dec = spacing / dec_per_tick
        y_to_log = lambda y, yb=yb, p=px_per_dec, b=bottoms[ip]: b + (yb - y) / p
        pan = dict(frame=(yt, yb), tick_rows=tk.tolist(), px_per_decade=float(px_per_dec), log_top=float(y_to_log(yt)),
                   tick_check_first_tick_log=float(y_to_log(tk[0])))
        for name, col in colours.items():
            col = np.array(col); sub = im[yt + 3:yb - 2, x_l + 3:x_r - 2]
            mask = (np.abs(sub - col).sum(axis=2) < 60)
            vals = np.full(len(edges) - 1, np.nan)
            for i in range(len(edges) - 1):
                xa = int(x_l + (edges[i] + 8 + 0.1) / 16 * (x_r - x_l)); xb = int(x_l + (edges[i] + 8 + 0.4) / 16 * (x_r - x_l))
                ys = []
                for x in range(xa, xb):
                    yy = np.where(mask[:, x - (x_l + 3)])[0]
                    if len(yy): ys.append(yy.min() + yt + 3)
                if ys: vals[i] = 10 ** y_to_log(np.median(ys))
            pan[name] = vals
        # black data points
        blk = (im[yt + 3:yb - 2, x_l + 3:x_r - 2].sum(axis=2) < 60)
        labb, nb = ndimage.label(blk); sz = ndimage.sum(blk, labb, range(1, nb + 1)); cm = ndimage.center_of_mass(blk, labb, range(1, nb + 1))
        pts = [(float(-8 + 16 * (c[1] + x_l + 3 - x_l) / (x_r - x_l)), float(10 ** y_to_log(c[0] + yt + 3))) for c, s_ in zip(cm, sz) if 150 < s_ < 3000]
        pan['data_points'] = pts
        out[ip] = pan
    return edges, out

F5_EDGES, F5 = digitise_fig5()
F5_C = 0.5 * (F5_EDGES[1:] + F5_EDGES[:-1])
f5_tab = []
for ip, lab_ in ((0, 'S1c<250'), (1, '250<S1c<500'), (2, 'S1c>500')):
    for i in range(len(F5_C)):
        f5_tab.append(dict(panel=lab_, x_lo=F5_EDGES[i], x_hi=F5_EDGES[i + 1], continuous_ERs=F5[ip]['ERs'][i], internal_gamma_IC_EC=F5[ip]['Internal'][i],
                           total=F5[ip]['Total'][i], MSSI=F5[ip]['MSSI'][i], accidentals=F5[ip]['Accidentals'][i], NRs=F5[ip]['NRs'][i]))
F5T = pd.DataFrame(f5_tab); F5T.to_csv(f'{OUT}/fig5_digitised_all_panels.csv', index=False)
tot_bottom = np.nansum(F5[2]['Total']); er_bottom = np.nansum(F5[2]['ERs']); mag_bottom = np.nansum(F5[2]['Internal'])
er_mid = np.nansum(F5[1]['ERs']); mag_mid = np.nansum(F5[1]['Internal']); tot_mid = np.nansum(F5[1]['Total'])
data_mid = sum(v for x, v in F5[1]['data_points']); data_top = sum(v for x, v in F5[0]['data_points'])
R['fig5_digitised'] = dict(bottom_total=float(tot_bottom), bottom_continuous_ERs=float(er_bottom), bottom_internal=float(mag_bottom),
                           middle_total=float(tot_mid), middle_continuous_ERs=float(er_mid), middle_internal=float(mag_mid),
                           middle_data_points_sum=float(data_mid), top_data_points_sum=float(data_top),
                           panels={ip: dict(frame=F5[ip]['frame'], px_per_decade=F5[ip]['px_per_decade'], log_top=F5[ip]['log_top']) for ip in F5},
                           bottom_ERs_by_bin={f'{F5_EDGES[i]:+.1f}': float(F5[2]['ERs'][i]) for i in range(len(F5_C)) if np.isfinite(F5[2]['ERs'][i])},
                           bottom_internal_by_bin={f'{F5_EDGES[i]:+.1f}': float(F5[2]['Internal'][i]) for i in range(len(F5_C)) if np.isfinite(F5[2]['Internal'][i])})
log(f"    Fig.5 digitised: bottom total {tot_bottom:.4f} (caption 0.0106); bottom continuous ERs {er_bottom:.4f}, internal {mag_bottom:.4f}; "
    f"middle: total {tot_mid:.2f}, cont. ERs {er_mid:.2f}, internal {mag_mid:.2f}, data points {data_mid:.0f}; top data {data_top:.0f} (1710 total obs.)")
log('    bottom ERs by bin:', {k: f'{v:.1e}' for k, v in R['fig5_digitised']['bottom_ERs_by_bin'].items()})
log('    bottom internal by bin:', {k: f'{v:.1e}' for k, v in R['fig5_digitised']['bottom_internal_by_bin'].items()})

# =====================================================================================================
# 4. Band calibration: k from the drawn ER lines (Fig. 4) and the 212Pb scatter; asymmetry test of the skew
# =====================================================================================================
CAL_S1 = [300, 350, 400, 450, 500, 540, 600]
def band_table(Hsum):
    return {s: percentiles_from_H(Hsum, s) for s in CAL_S1}
def mismatch(k, s):
    H = response_flat(E_GRID, k=k, s=s).sum(axis=0); bt = band_table(H)
    return sum((bt[s1]['halfwidth'] / D4.loc[s1, 'ER_halfwidth'] - 1) ** 2 for s1 in CAL_S1 if np.isfinite(D4.loc[s1, 'ER_halfwidth'])), bt
KCAL = {}
KGRID = np.array([0.35, 0.50, 0.65, 0.80, 1.00])
for s in (1.0, 0.0):
    ratios = []
    for k in (KGRID if s == 1.0 else KGRID[:3]):
        _, bt = mismatch(k, s)
        ratios.append(np.mean([bt[s1]['halfwidth'] / D4.loc[s1, 'ER_halfwidth'] for s1 in CAL_S1]))
    ratios = np.array(ratios); kg = KGRID[:len(ratios)]   # mean model/drawn half-width ratio, monotonic in k
    kbest = float(np.interp(1.0, ratios, kg)); _, bt = mismatch(kbest, s)
    KCAL[s] = dict(k=kbest, band=bt, k_grid=kg.tolist(), ratio_grid=ratios.tolist())
    log(f"[4] k calibration (s={s}): k = {kbest:.3f} (ratios {np.round(ratios, 3).tolist()}); model half-widths vs drawn:", {s1: (round(bt[s1]['halfwidth'], 4), round(float(D4.loc[s1, 'ER_halfwidth']), 4)) for s1 in CAL_S1}, f'({time.time()-T0:.0f} s)')
K_BASE = round(KCAL[1.0]['k'], 2)
# fixed-k variants and their widths / asymmetry
CAL_ROWS = []
for k, s, tag in ((1.0, 1.0, 'k=1 literal sigma_p*Ni, NEST skew'), (K_BASE, 1.0, f'k={K_BASE} NEST skew (baseline)'), (K_BASE, 0.0, f'k={K_BASE} Gaussian'),
                  (K_BASE, -1.0, f'k={K_BASE} mirrored skew'), (K_BASE, 1.5, f'k={K_BASE} skew x1.5'), (K_BASE, 0.5, f'k={K_BASE} skew x0.5'),
                  (0.35, 1.0, 'k=0.35 NEST skew'), (0.75, 1.0, 'k=0.75 NEST skew')):
    H = response_flat(E_GRID, k=k, s=s).sum(axis=0); bt = band_table(H)
    for s1 in CAL_S1:
        CAL_ROWS.append(dict(variant=tag, k=k, s=s, S1c=s1, **bt[s1], drawn_p10=D4.loc[s1, 'ER_p10'], drawn_p50=D4.loc[s1, 'ER_p50'], drawn_p90=D4.loc[s1, 'ER_p90'],
                             drawn_halfwidth=D4.loc[s1, 'ER_halfwidth'], drawn_upper=D4.loc[s1, 'ER_upper'], drawn_lower=D4.loc[s1, 'ER_lower']))
CAL = pd.DataFrame(CAL_ROWS); CAL['asym_model'] = CAL['upper'] / CAL['lower']; CAL['asym_drawn'] = CAL['drawn_upper'] / CAL['drawn_lower']
CAL.to_csv(f'{OUT}/band_calibration.csv', index=False)
log(CAL[CAL.S1c.isin([300, 400, 540, 600])][['variant', 'S1c', 'p10', 'p50', 'p90', 'halfwidth', 'drawn_halfwidth', 'asym_model', 'asym_drawn']].round(4).to_string(index=False))
# MC cross-check of the engine at the baseline
E_mc, s1_mc, l2_mc = band_mc(k=K_BASE, s=1.0)
MCCHK = {}
for s1 in (300, 400, 540, 600):
    m = np.abs(s1_mc - s1) < 10; q = np.percentile(l2_mc[m], [10, 50, 90])
    MCCHK[s1] = dict(mc_p10=q[0], mc_p50=q[1], mc_p90=q[2], mc_n=int(m.sum()), engine=KCAL[1.0]['band'][s1] if abs(KCAL[1.0]['k'] - K_BASE) < 0.02 else None,
                     mc_frac_below_ROI=float((l2_mc[m] < ROI_TOP).mean()), E_p10_p90=list(np.percentile(E_mc[m], [10, 90])))
R['mc_crosscheck'] = MCCHK
log('    MC cross-check (baseline):', {s: (round(v['mc_p10'], 4), round(v['mc_p50'], 4), round(v['mc_p90'], 4), v['E_p10_p90']) for s, v in MCCHK.items()})
R['k_calibration'] = dict(k_nest_skew=KCAL[1.0]['k'], k_gauss=KCAL[0.0]['k'], k_base=K_BASE,
                          sigma_Ne_70keV=dict(k1=float(np.sqrt(0.72 * 0.28 * 4405 + (sigma_p(5208) * 4405) ** 2)),
                                              k_base=float(np.sqrt(0.72 * 0.28 * 4405 + (K_BASE * sigma_p(5208) * 4405) ** 2)),
                                              nestpy_LZWS2024=float(SK.loc[SK.E_keV == 70, 'sd_Ne_nestpy_LZWS2024'].iloc[0])),
                          sigma_p_at_event_Nq=float(sigma_p(5179)))

# =====================================================================================================
# 5. Leakage of the beta continuum: fractions per S1c bin, expected events, Fig. 5 projection
# =====================================================================================================
# ER continuum rate per keV_ee from the ROI counts (Table I continuum components) and the model's ROI acceptance
S1_LO_EDGES = np.arange(0.0, 640.0 + 0.1, 5.0); L2_LO_EDGES = np.arange(2.5, 4.8 + 1e-9, 0.005)
E_LO_GRID = np.arange(1.0, 40.0 + 0.1, 0.5)
acc = []
for E in E_LO_GRID:
    H = response_at_E(E, k=K_BASE, s=1.0, s1_edges=S1_LO_EDGES, l2_edges=L2_LO_EDGES, n_ne=300)
    c1 = (S1_LO_EDGES[:-1] >= 3.0) & (S1_LO_EDGES[1:] <= 600.0); c2 = (L2_LO_EDGES[:-1] >= 2.75) & (L2_LO_EDGES[1:] <= ROI_TOP)
    acc.append(H[np.ix_(c1, c2)].sum())
acc = np.array(acc); A_ROI = float(np.trapezoid(acc, E_LO_GRID))
N_CONT_ROI = 1341 + 140.6 + 110.0 + 55.3 + 8.5
RATE = N_CONT_ROI / A_ROI
R['er_rate'] = dict(ROI_acceptance_integral_keV=A_ROI, N_continuum_ROI_TableI=N_CONT_ROI, rate_per_keV=RATE,
                    acceptance_curve={float(E): float(a) for E, a in zip(E_LO_GRID[::4], acc[::4])}, P010_rate_per_keV=85.0,
                    note='flat extrapolation of the 2-15 keV_ee continuum rate to 40-160 keV_ee; +-50 % systematic (214Pb/212Pb beta shapes, 136Xe rise, tritium absent above 18.6 keV)')
log(f"[5] ROI acceptance integral {A_ROI:.2f} keV -> continuum rate {RATE:.1f} events/keV_ee (P010: 85)")

# variants for the leakage study
def s4_variant(**changes):
    P = dict(P_S4); P.update(changes); return P
VARIANTS = {
    'baseline: k=%.2f, NEST skew' % K_BASE: dict(k=K_BASE, s=1.0),
    'Gaussian (s=0)': dict(k=K_BASE, s=0.0),
    'skew x0.5': dict(k=K_BASE, s=0.5),
    'skew x1.5': dict(k=K_BASE, s=1.5),
    'mirrored skew (s=-1)': dict(k=K_BASE, s=-1.0),
    'k=0.35 NEST skew': dict(k=0.35, s=1.0),
    'k=0.75 NEST skew': dict(k=0.75, s=1.0),
    'k=1.0 literal NEST skew': dict(k=1.0, s=1.0),
    'k=1.0 literal Gaussian': dict(k=1.0, s=0.0),
    'alpha1 x0.5': dict(k=K_BASE, s=1.0, P=s4_variant(alpha1=P_S4['alpha1'] * 0.5)),
    'alpha1 x1.5': dict(k=K_BASE, s=1.0, P=s4_variant(alpha1=P_S4['alpha1'] * 1.5)),
    'alpha2 x0.5': dict(k=K_BASE, s=1.0, P=s4_variant(alpha2=P_S4['alpha2'] * 0.5)),
    'alpha2 x1.5': dict(k=K_BASE, s=1.0, P=s4_variant(alpha2=P_S4['alpha2'] * 1.5)),
    'alpha1,alpha2 both x0.5': dict(k=K_BASE, s=1.0, P=s4_variant(alpha1=P_S4['alpha1'] * 0.5, alpha2=P_S4['alpha2'] * 0.5)),
    'alpha2 sign flipped': dict(k=K_BASE, s=1.0, P=s4_variant(alpha2=-P_S4['alpha2'])),
    'Gaussian, k=0.75': dict(k=0.75, s=0.0),
    'mirrored skew, k=0.35': dict(k=0.35, s=-1.0),
}
S1_BINS = [(300, 350), (350, 400), (400, 450), (450, 500), (500, 550), (550, 600), (500, 600), (480, 600)]
def region_sum(H3, s1_lo, s1_hi, l2_lo_fn, l2_hi_fn):
    """N per keV-flat unit: sum over E of H over cells with S1 in [lo,hi) and l2 in [lo_fn(S1c), hi_fn(S1c))."""
    Hs = H3.sum(axis=0) if H3.ndim == 3 else H3
    cols = np.where((S1_C > s1_lo) & (S1_C < s1_hi))[0]; tot = 0.0
    for c in cols:
        lo = l2_lo_fn(S1_C[c]); hi = l2_hi_fn(S1_C[c])
        rows = (L2_C > lo) & (L2_C < hi)
        tot += Hs[c, rows].sum()
    return float(tot)
REG = {   # name: (l2_lo(S1c), l2_hi(S1c))
    'below_NR_median': (lambda s: -9, lambda s: MU_NR(s)),
    'below_event_depth(-1.5sig_LZ)': (lambda s: -9, lambda s: MU_NR(s) - EV_DEPTH),
    'in_ROI(<4.15)': (lambda s: -9, lambda s: ROI_TOP),
    'NR_pm1hw': (lambda s: MU_NR(s) - HW_NR(s), lambda s: MU_NR(s) + HW_NR(s)),
    'event_region_V(3.90-4.00)': (lambda s: 3.90, lambda s: 4.00),
    'gap_G(-1hw..ROI_top)': (lambda s: MU_NR(s) - HW_NR(s), lambda s: ROI_TOP),
    'tail_+1..+5hw': (lambda s: MU_NR(s) + HW_NR(s), lambda s: min(MU_NR(s) + 5 * HW_NR(s), ROI_TOP)),
}
LEAK_ROWS = []; H3_STORE = {}
for vname, kw in VARIANTS.items():
    H3 = response_flat(E_GRID, **kw); H3_STORE[vname] = H3
    for (lo, hi) in S1_BINS:
        n_tot = region_sum(H3, lo, hi, lambda s: -9, lambda s: 9)          # all ERs in the S1c bin (flat spectrum, per DE keV)
        row = dict(variant=vname, S1c_lo=lo, S1c_hi=hi, N_ER_in_bin=n_tot * RATE * DE)
        for rn, (flo, fhi) in REG.items():
            v = region_sum(H3, lo, hi, flo, fhi)
            row[f'frac_{rn}'] = v / n_tot if n_tot > 0 else np.nan; row[f'N_{rn}'] = v * RATE * DE
        LEAK_ROWS.append(row)
    log(f"    variant done: {vname} ({time.time()-T0:.0f} s)")
LEAK = pd.DataFrame(LEAK_ROWS); LEAK.to_csv(f'{OUT}/beta_leakage_by_S1c_bin.csv', index=False)
np.savez_compressed(f'{OUT}/H3_store.npz', E_GRID=E_GRID, S1_EDGES=S1_EDGES, L2_EDGES=L2_EDGES, names=np.array(list(H3_STORE)), **{f'H{i}': v for i, v in enumerate(H3_STORE.values())})
show = LEAK[LEAK.S1c_lo.isin([300, 400, 500, 550]) & (LEAK.S1c_hi - LEAK.S1c_lo == 50)]
log('\n[5] beta-continuum leakage fractions (per S1c bin) and expected events in 2.84 t yr:')
log(show[['variant', 'S1c_lo', 'N_ER_in_bin', 'frac_in_ROI(<4.15)', 'frac_below_NR_median', 'frac_below_event_depth(-1.5sig_LZ)', 'N_in_ROI(<4.15)', 'N_gap_G(-1hw..ROI_top)', 'N_event_region_V(3.90-4.00)', 'N_tail_+1..+5hw']].to_string(index=False, float_format=lambda x: f'{x:.3g}'))

# Fig.-5 projection for S1c > 500 (500-600): expected counts per 0.5-halfwidth bin in x = (l2 - mu_NR)/hw
def fig5_projection(H3, s1_lo=500, s1_hi=600, f_scale=RATE * DE):
    Hs = H3.sum(axis=0) if H3.ndim == 3 else H3
    cols = np.where((S1_C > s1_lo) & (S1_C < s1_hi))[0]
    out = np.zeros(len(F5_C))
    for c in cols:
        x = (L2_C - MU_NR(S1_C[c])) / HW_NR(S1_C[c])
        idx = np.digitize(x, F5_EDGES) - 1; ok_ = (idx >= 0) & (idx < len(F5_C)) & (L2_C < ROI_TOP)
        np.add.at(out, idx[ok_], Hs[c, ok_])
    return out * f_scale
PROJ = {v: fig5_projection(H3_STORE[v]) for v in ('baseline: k=%.2f, NEST skew' % K_BASE, 'Gaussian (s=0)', 'mirrored skew (s=-1)', 'k=1.0 literal Gaussian', 'k=0.75 NEST skew', 'Gaussian, k=0.75')}
pd.DataFrame(dict(x_lo=F5_EDGES[:-1], x_hi=F5_EDGES[1:], LZ_continuous_ERs=F5[2]['ERs'], LZ_internal=F5[2]['Internal'], **{k: v for k, v in PROJ.items()})).to_csv(f'{OUT}/fig5_bottom_projection.csv', index=False)
for v, arr in PROJ.items():
    sel = (F5_C > 1) & (F5_C < 5.5)
    log(f"    Fig.5 projection S1c>500 [{v}]: sum(+1..+5.5) = {arr[sel].sum():.2e} (LZ light-blue {np.nansum(F5[2]['ERs'][sel]):.2e}); sum(-2..+1) = {arr[(F5_C>-2)&(F5_C<1)].sum():.2e}; bin(-2,-1.5) = {arr[np.argmin(np.abs(F5_C+1.75))]:.2e}")
# middle-panel cross-check of the rate: continuous ERs in the ROI at 250 < S1c < 500 (engine columns cover 240-500)
Hb = H3_STORE['baseline: k=%.2f, NEST skew' % K_BASE]
n_mid = region_sum(Hb, 250, 500, lambda s: -9, lambda s: ROI_TOP) * RATE * DE
R['middle_panel_check'] = dict(model_continuous_ERs_250_500_in_ROI=n_mid, fig5_light_blue_middle_sum=float(er_mid), fig5_data_points_middle=float(data_mid))
log(f"    middle-panel check: model continuous ERs in ROI at 250<S1c<500 = {n_mid:.2f} vs Fig. 5 light-blue sum {er_mid:.2f}")

# =====================================================================================================
# 6. EC lines: 124Xe KK 64.3 keV (564 decays) and 125I K 67.3 keV (1e3, 1e4 decays), charge suppression f
# =====================================================================================================
N_KK = 564.0; N_I125 = (1e3, 1e4)
EC_ROWS = []
for line, E, Ns in (('Xe124_KK_64.3', 64.3, (N_KK,)), ('I125_K_67.3', 67.3, N_I125)):
    for vname in ('baseline: k=%.2f, NEST skew' % K_BASE, 'Gaussian (s=0)', 'mirrored skew (s=-1)', 'k=0.75 NEST skew', 'Gaussian, k=0.75', 'k=1.0 literal Gaussian', 'k=1.0 literal NEST skew'):
        kw = VARIANTS[vname]
        for f in (0.0, 0.1, 0.2, 0.3):
            H = response_at_E(E, f=f, **kw)
            nph_m, ne_m, nq_m, al = er_means_s(E)
            for Nd in Ns:
                row = dict(line=line, E_keV=E, variant=vname, f=f, N_decays=Nd, Ne_mean=(1 - f) * ne_m, S1c_mean=G1 * (nq_m - (1 - f) * ne_m),
                           log10S2c_mean=math.log10(G2 * (1 - f) * ne_m))
                for rn, (flo, fhi) in REG.items():
                    row[f'P_{rn}_S1c480_600'] = region_sum(H, 480, 600, flo, fhi)
                    row[f'N_{rn}_S1c480_600'] = row[f'P_{rn}_S1c480_600'] * Nd
                row['P_in_ROI_anyS1c'] = region_sum(H, 240, 660, lambda s: -9, lambda s: ROI_TOP)
                row['N_in_ROI_anyS1c'] = row['P_in_ROI_anyS1c'] * Nd
                EC_ROWS.append(row)
EC = pd.DataFrame(EC_ROWS); EC.to_csv(f'{OUT}/ec_lines_leakage.csv', index=False)
log('\n[6] EC lines (S1c 480-600):')
log(EC[EC.variant.isin(['baseline: k=%.2f, NEST skew' % K_BASE, 'Gaussian (s=0)', 'mirrored skew (s=-1)']) & EC.f.isin([0.0, 0.2, 0.3])][
    ['line', 'variant', 'f', 'N_decays', 'S1c_mean', 'log10S2c_mean', 'N_in_ROI_anyS1c', 'N_gap_G(-1hw..ROI_top)_S1c480_600', 'N_event_region_V(3.90-4.00)_S1c480_600', 'N_below_NR_median_S1c480_600']].to_string(index=False, float_format=lambda x: f'{x:.3g}'))

# =====================================================================================================
# 7. Sensitivity summary (orders of magnitude) and the calibration needed
# =====================================================================================================
base_name = 'baseline: k=%.2f, NEST skew' % K_BASE
def pick(v, lo, hi, col):
    return float(LEAK[(LEAK.variant == v) & (LEAK.S1c_lo == lo) & (LEAK.S1c_hi == hi)][col].iloc[0])
SENS = []
for v in VARIANTS:
    d = dict(variant=v)
    for col in ('N_in_ROI(<4.15)', 'N_gap_G(-1hw..ROI_top)', 'N_below_NR_median', 'N_event_region_V(3.90-4.00)'):
        b = pick(base_name, 500, 600, col); x = pick(v, 500, 600, col)
        d[col + '_500_600'] = x; d['log10_ratio_' + col] = math.log10(x / b) if x > 0 and b > 0 else np.nan
    d['sigma_p_at_Nq5179'] = float(sigma_p(5179, VARIANTS[v].get('P', P_S4)))
    d['ER_halfwidth_540_model'] = float(percentiles_from_H(H3_STORE[v].sum(axis=0), 540)['halfwidth'])
    SENS.append(d)
SENS = pd.DataFrame(SENS); SENS.to_csv(f'{OUT}/sensitivity_summary.csv', index=False)
log('\n[7] sensitivity (S1c 500-600):')
log(SENS[['variant', 'sigma_p_at_Nq5179', 'ER_halfwidth_540_model', 'N_in_ROI(<4.15)_500_600', 'N_gap_G(-1hw..ROI_top)_500_600', 'N_event_region_V(3.90-4.00)_500_600', 'log10_ratio_N_in_ROI(<4.15)', 'log10_ratio_N_event_region_V(3.90-4.00)']].to_string(index=False, float_format=lambda x: f'{x:.3g}'))

# calibration needed: number of ER calibration events at S1c 500-600 to bound a leakage fraction p at 90 % CL (0 observed): N = 2.303/p
# fraction of 212Pb / 14C beta decays that land at S1c 500-600 (energies ~ 80-115 keV): allowed-shape beta spectra (recalled: certain)
def beta_spectrum(E, Q, Z=83):
    E = np.asarray(E, float); me = 511.0
    W = E / me + 1; p = np.sqrt(np.clip(W * W - 1, 0, None)); eta = 1 / 137.036 * Z * W / np.clip(p, 1e-9, None)
    F = 2 * np.pi * eta / (1 - np.exp(-2 * np.pi * eta))
    return np.where((E > 0) & (E < Q), F * p * W * (Q - E) ** 2, 0.0)
Eb = np.arange(0.5, 600, 0.5)
frac_500_600 = {}
Hcols = np.where((S1_C > 500) & (S1_C < 600))[0]
P_S1_given_E = np.array([H3_STORE[base_name][i][Hcols].sum() for i in range(len(E_GRID))])   # P(S1c in 500-600 | E) (per energy; no DE)
for src, Q, Z in (('Pb212', 570.0, 83), ('C14', 156.5, 7), ('Pb214', 1019.0, 83)):
    sp = beta_spectrum(Eb, Q, Z); sp /= np.trapezoid(sp, Eb)
    f = float(np.trapezoid(np.interp(Eb, E_GRID, P_S1_given_E, left=0, right=0) * sp, Eb))
    frac_500_600[src] = f
p_targets = {'ROI leakage fraction (baseline, 500-600)': pick(base_name, 500, 600, 'frac_in_ROI(<4.15)'),
             'ROI leakage fraction (Gaussian, 500-600)': pick('Gaussian (s=0)', 500, 600, 'frac_in_ROI(<4.15)'),
             'ROI leakage fraction (k=1 Gaussian)': pick('k=1.0 literal Gaussian', 500, 600, 'frac_in_ROI(<4.15)'),
             'gap fraction (mirrored skew)': pick('mirrored skew (s=-1)', 500, 600, 'frac_gap_G(-1hw..ROI_top)'),
             '1e-3': 1e-3, '1e-4': 1e-4, '1e-5': 1e-5}
CALNEED = {name: dict(p=p, N_events_500_600_for_90CL=(2.303 / p if p > 0 else np.inf),
                      decays_needed={src: (2.303 / p / f if p > 0 else np.inf) for src, f in frac_500_600.items()}) for name, p in p_targets.items()}
R['calibration_needed'] = dict(frac_of_beta_decays_at_S1c_500_600=frac_500_600, targets=CALNEED,
                               current_Pb212_points_500_600=PB['500_600']['n_est_all'],
                               current_bound_90CL=2.303 / max(PB['500_600']['n_est_all'], 1))
log('    fraction of beta decays landing at S1c 500-600:', {k: round(v, 4) for k, v in frac_500_600.items()})
log('    calibration needed:', json.dumps({k: dict(p=f"{v['p']:.2e}", N=f"{v['N_events_500_600_for_90CL']:.2e}", decays={s: f'{d:.1e}' for s, d in v['decays_needed'].items()}) for k, v in CALNEED.items()}, indent=0))

# =====================================================================================================
# 8. Figures
# =====================================================================================================
C1, C2, C3, C4, C5, INK, GRID = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#52514e', '#e1e0d9'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK})
# Fig 1: ER band (model variants vs drawn) + 212Pb points + NR band + event + regions
fig, ax = plt.subplots(figsize=(8.2, 5.6))
s1_plot = np.arange(250, 651, 10)
for vname, col, ls in ((base_name, C1, '-'), ('Gaussian (s=0)', C3, '--'), ('mirrored skew (s=-1)', C2, ':'), ('k=1.0 literal NEST skew', C4, '-.')):
    Hs = H3_STORE[vname].sum(axis=0)
    p10 = []; p50 = []; p90 = []
    for s1 in s1_plot:
        q = percentiles_from_H(Hs, s1, half=6); p10.append(q['p10']); p50.append(q['p50']); p90.append(q['p90'])   # +-6: the two 10-phd columns around s1
    ax.plot(s1_plot, p50, color=col, ls=ls, lw=1.4, label=f'model median/10-90 %: {vname}')
    ax.plot(s1_plot, p10, color=col, ls=ls, lw=0.9); ax.plot(s1_plot, p90, color=col, ls=ls, lw=0.9)
d4 = D4.dropna(subset=['ER_p50'])
ax.plot(d4.index, d4.ER_p50, 'k-', lw=2, alpha=0.6, label='LZ Fig. 4 ER band (digitised: median, 10-90 %)')
ax.plot(d4.index, d4.ER_p10, 'k--', lw=1.2, alpha=0.6); ax.plot(d4.index, d4.ER_p90, 'k--', lw=1.2, alpha=0.6)
msel = (s1g > 250) & (s1g < 800)
ax.scatter(s1g[msel], l2g[msel], s=6, color=C5, alpha=0.6, label=r'LZ Fig. 2 $^{212}$Pb calibration points (digitised)')
ax.plot(s1_plot, MU_NR(s1_plot), color='#c0392b', lw=1.5, label='NR median (Fig. 4, digitised)')
ax.plot(s1_plot, MU_NR(s1_plot) - HW_NR(s1_plot), color='#c0392b', lw=0.8, ls='--'); ax.plot(s1_plot, MU_NR(s1_plot) + HW_NR(s1_plot), color='#c0392b', lw=0.8, ls='--')
ax.axhline(ROI_TOP, color=INK, ls='-.', lw=0.8, label='WS ROI top (S2c = 10$^{4.15}$)')
ax.fill_between([480, 600], [MU_NR(480) - HW_NR(480), MU_NR(600) - HW_NR(600)], [ROI_TOP, ROI_TOP], color=C4, alpha=0.2, label='gap G (0 events observed)')
ax.fill_between([500, 600], [3.90, 3.90], [4.00, 4.00], color=C3, alpha=0.25, label='event region V')
ax.plot(S1_EV, L2_EV, marker='*', color='k', ms=15, ls='none', label='event (540.1 phd, 10$^{3.97}$ phd)')
ax.set_xlim(250, 650); ax.set_ylim(3.75, 5.1); ax.set_xlabel('S1c [phd]'); ax.set_ylabel(r'$\log_{10}$(S2c [phd])')
ax.set_title(f'ER band from Table S3 + Table S4 $\\sigma_p$ (k = {K_BASE}) with NEST / Gaussian / mirrored recombination skew, vs LZ drawn band and $^{{212}}$Pb data', fontsize=9)
ax.legend(fontsize=6.5, loc='upper left', ncol=2); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f'{FIG}/P056_fig1_er_band_variants.png', dpi=160); plt.close(fig)

# Fig 2: leakage vs S1c (fraction in ROI, below NR median, below event depth) for the variants
fig, axs = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
bins50 = [(a, b) for (a, b) in S1_BINS if b - a == 50]
for ax, col, ttl in zip(axs, ('frac_in_ROI(<4.15)', 'frac_below_NR_median', 'frac_below_event_depth(-1.5sig_LZ)'),
                        ('fraction of ERs entering the ROI (S2c < 10$^{4.15}$)', 'fraction below the NR median', "fraction below the event's depth (LZ: $-1.5\\sigma_{NR}$)")):
    for vname, c, ls in ((base_name, C1, '-'), ('Gaussian (s=0)', C3, '--'), ('mirrored skew (s=-1)', C2, ':'), ('k=1.0 literal NEST skew', C4, '-.'), ('k=1.0 literal Gaussian', C5, '-.'), ('k=0.75 NEST skew', INK, '-')):
        y = [pick(vname, a, b, col) for (a, b) in bins50]
        ax.plot([0.5 * (a + b) for (a, b) in bins50], np.clip(y, 1e-30, None), color=c, ls=ls, marker='o', ms=3, label=vname)
    ax.set_yscale('log'); ax.set_ylim(1e-20, 1); ax.set_xlabel('S1c [phd] (50-phd bins)'); ax.set_title(ttl, fontsize=9); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
axs[0].set_ylabel('fraction of flat-spectrum ERs at that S1c'); axs[0].legend(fontsize=6.5, loc='lower left')
fig.suptitle('Leakage of beta-like ERs versus S1c for the recombination-fluctuation variants (Table S4 $\\sigma_p$, calibrated k)', fontsize=10)
fig.tight_layout(); fig.savefig(f'{FIG}/P056_fig2_leakage_vs_S1c.png', dpi=160); plt.close(fig)

# Fig 3: Fig. 5 bottom-panel comparison (S1c > 500): LZ light-blue / magenta vs our projections, plus EC lines
fig, ax = plt.subplots(figsize=(8.5, 4.6))
ax.step(F5_EDGES[:-1], np.where(np.isfinite(F5[2]['ERs']), F5[2]['ERs'], np.nan), where='post', color='#00c2f9', lw=2, label='LZ Fig. 5 bottom: Continuous ERs (digitised)')
ax.step(F5_EDGES[:-1], np.where(np.isfinite(F5[2]['Internal']), F5[2]['Internal'], np.nan), where='post', color='magenta', lw=2, label='LZ Fig. 5 bottom: Internal $\\gamma$ + ICs + ECs (digitised)')
ax.step(F5_EDGES[:-1], np.where(np.isfinite(F5[2]['Total']), F5[2]['Total'], np.nan), where='post', color='blue', lw=1, alpha=0.5, label='LZ total model (digitised)')
for vname, c, ls in ((base_name, C1, '-'), ('Gaussian (s=0)', C3, '--'), ('mirrored skew (s=-1)', C2, ':'), ('k=1.0 literal Gaussian', C5, '-.'), ('Gaussian, k=0.75', C4, '--')):
    ax.step(F5_EDGES[:-1], np.clip(PROJ[vname], 1e-12, None), where='post', color=c, ls=ls, lw=1.3, label=f'this work, beta continuum: {vname}')
# EC lines at f = 0.2, baseline and Gaussian, projected
for line, E, Nd, c in (('Xe124_KK_64.3', 64.3, N_KK, '#8e44ad'), ('I125_K_67.3', 67.3, 1e4, '#16a085')):
    for vname, ls in (('Gaussian (s=0)', '--'),):
        H = response_at_E(E, f=0.2, **VARIANTS[vname]); pr = fig5_projection(H, 500, 600, f_scale=Nd)
        ax.step(F5_EDGES[:-1], np.clip(pr, 1e-12, None), where='post', color=c, ls=ls, lw=1.2, label=f'{line}, f=0.2, {Nd:.0f} decays, Gaussian')
ax.plot(-1.75, 1.0, 'ko', ms=6, label='the event (data)')
ax.set_yscale('log'); ax.set_ylim(1e-9, 3); ax.set_xlim(-4, 6)
ax.set_xlabel(r'$(\log_{10}\mathrm{S2c} - \mu_{NR})/\sigma_{NR}$  ($\sigma_{NR}$ = drawn 10-90 %% half-width, %.3f dex)' % float(HW_NR(540))); ax.set_ylabel('events per 0.5 bin, S1c 500-600 phd, 2.84 t yr')
ax.legend(fontsize=6, loc='upper left', ncol=2); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_title('S1c > 500 phd: LZ background-model ER components versus this work\'s recombination-fluctuation predictions', fontsize=9)
fig.tight_layout(); fig.savefig(f'{FIG}/P056_fig3_fig5_projection.png', dpi=160); plt.close(fig)

# Fig 4: sigma_p(x) with alpha variants, and the Ne pdf shapes at 70 keV
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
xg = np.linspace(2.0, 4.6, 400); nqg = 10 ** xg
axs[0].plot(xg, sigma_p(nqg), color=INK, lw=2, label='Table S4 (LZ)')
for lab_, P, c in (('alpha1 x0.5', s4_variant(alpha1=P_S4['alpha1'] * 0.5), C1), ('alpha1 x1.5', s4_variant(alpha1=P_S4['alpha1'] * 1.5), C1), ('alpha2 x0.5', s4_variant(alpha2=P_S4['alpha2'] * 0.5), C2), ('alpha2 x1.5', s4_variant(alpha2=P_S4['alpha2'] * 1.5), C2), ('alpha2 sign flipped', s4_variant(alpha2=-P_S4['alpha2']), C4)):
    axs[0].plot(xg, sigma_p(nqg, P), color=c, lw=1, ls='--' if 'x0.5' in lab_ else ':', label=lab_)
axs[0].plot(np.log10(SK.Nq), SK.eff_sigma_p_nestpy, 's', color=C3, ms=4, label='nestpy 2.1.1 LZ_WS2024 effective $\\sigma_p$ (v2.4.0 form)')
axs[0].plot(np.log10(SK.Nq), K_BASE * sigma_p(SK.Nq), 'o', color=C5, ms=4, label=f'k = {K_BASE} x Table S4 (band-calibrated)')
for E, lab_ in ((5, 'tritium 5 keV'), (18.6, 'T endpoint'), (69.6, 'event E$_{ee}$'), (156, r'$^{14}$C endpoint')):
    x_ = math.log10(er_means_s(E)[2]); axs[0].axvline(x_, color=GRID, lw=0.8); axs[0].text(x_, 0.095, lab_, rotation=90, fontsize=6, va='top')
axs[0].set_xlabel(r'$x = \log_{10} N_q$'); axs[0].set_ylabel(r'$\sigma_p$'); axs[0].set_ylim(0, 0.1); axs[0].legend(fontsize=6.5); axs[0].grid(color=GRID, lw=0.6); axs[0].set_axisbelow(True)
axs[0].set_title('Table S4 double skew-Gaussian $\\sigma_p(x)$ and its $\\alpha_i$ sensitivity', fontsize=9)
ne_g = np.linspace(0, 1900, 800); nph_m, ne_m, nq_m, al = er_means_s(69.6); ni_m = nq_m / (1 + al); r_ = 1 - ne_m / ni_m
sd_b = math.sqrt(r_ * (1 - r_) * ni_m + (K_BASE * float(sigma_p(nq_m)) * ni_m) ** 2); a70 = float(a_nest(69.6)[0])
for a, lab_, c in ((a70, f'NEST skew a = {a70:.2f}', C1), (0.0, 'Gaussian', C3), (-a70, 'mirrored skew', C2), (1.5 * a70, 'skew x1.5', C4), (0.5 * a70, 'skew x0.5', C5)):
    axs[1].plot(ne_g, skewnorm_pdf_grid(ne_g, float(ne_m), sd_b, a), color=c, label=lab_)
axs[1].axvline(S2_EV / G2, color='k', lw=1.2, label=f'event N$_e$ = {S2_EV/G2:.0f}'); axs[1].axvline(10 ** ROI_TOP / G2, color=INK, ls='-.', lw=0.8, label='ROI top (N$_e$ = 409)')
axs[1].set_yscale('log'); axs[1].set_ylim(1e-16, 1e-2); axs[1].set_xlabel(r'$N_e$ for a 69.6 keV ER (mean %.0f, $\sigma$ = %.0f e)' % (ne_m, sd_b)); axs[1].set_ylabel('pdf'); axs[1].legend(fontsize=6.5); axs[1].grid(color=GRID, lw=0.6); axs[1].set_axisbelow(True)
axs[1].set_title('Recombination-fluctuation shape: the NR-ward tail is set by the sign of the skew', fontsize=9)
fig.tight_layout(); fig.savefig(f'{FIG}/P056_fig4_sigma_p_and_shapes.png', dpi=160); plt.close(fig)

# =====================================================================================================
# 9. Save results
# =====================================================================================================
R['event'] = dict(S1c=S1_EV, S2c=S2_EV, log10S2c=L2_EV, Ne=S2_EV / G2, Nph=S1_EV / G1, Nq=S1_EV / G1 + S2_EV / G2, E_ee_keV_nestW=W_NEST * 1e-3 * (S1_EV / G1 + S2_EV / G2))
R['variants'] = {k: {kk: (vv if kk != 'P' else vv) for kk, vv in v.items()} for k, v in VARIANTS.items()}
R['leakage_500_600'] = {v: {c: pick(v, 500, 600, c) for c in LEAK.columns if c.startswith(('N_', 'frac_'))} for v in VARIANTS}
R['leakage_bins_baseline'] = LEAK[LEAK.variant == base_name].to_dict(orient='records')
R['fig5_projection'] = {v: arr.tolist() for v, arr in PROJ.items()}
R['ec_selected'] = EC[EC.f.isin([0.0, 0.2, 0.3])].to_dict(orient='records')
R['runtime_s'] = time.time() - T0
def _conv(o):
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, dict): return {str(k): _conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_conv(v) for v in o]
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)): return None
    return o
with open(f'{OUT}/results.json', 'w') as fh:
    json.dump(_conv(R), fh, indent=1)
log(f'saved {OUT}/results.json  ({time.time()-T0:.0f} s)')
