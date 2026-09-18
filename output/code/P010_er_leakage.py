"""
P010 -- ER leakage: the recombination fraction required and the 124Xe/125I double-vacancy hypothesis.

Run from the simulation root:  .venv/bin/python output/code/P010_er_leakage.py

Sections
  1. Event quanta, ER-equivalent energy, required recombination fraction
  2. Recombination-fluctuation model (Table S4 sigma_p); digitisation of the Fig. 4 bands; calibration factor k
     for the fixed-S1c band width; the meaning of the paper's "6.7 sigma" / "1.5 sigma"
  3. Fixed-energy line model (64.3 keV 124Xe KK, 67.3 keV 125I K) with recombination enhancement f and four
     tail shapes; probabilities to reach the event region V and the empty gap region G; continuum ER leakage
  4. Candidate populations (124Xe 2nuECEC decays by mode, 125I from the 3.6 d effective half-life, activation timing)
  5. The gap argument: Fig. 5 bottom-panel plateau; likelihood of "1 event in V, 0 in G"
  6. Figures and tables -> output/work/P010/

All LZ numbers from arXiv:2609.02823 (lzcommon.LZ, Tables I, S3, S4; Figs 4, 5).  Recalled inputs are
flagged in details.md / P010.json.
"""
from __future__ import annotations
import sys, json, math, os
sys.path.insert(0, 'output/code')
import numpy as np
import pandas as pd
from scipy import stats, special, optimize, integrate, ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from common import lzcommon as lz

OUT = 'output/work/P010'
FIG = f'{OUT}/figures'
os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(10)
R = {}   # results dictionary -> JSON

g1, g2 = lz.LZ['g1'], lz.LZ['g2']
S1c, S2c = lz.LZ['ev_S1c'], lz.LZ['ev_S2c']
LS2_EV = math.log10(S2c)
ROI_LOGS2_MAX = lz.LZ['log10S2c_max']
NE_ROI_MAX = 10**ROI_LOGS2_MAX / g2            # 409 electrons: ROI upper S2c edge in electrons

# ============================================================================================
# 1. Event quanta and ER-equivalent energy
# ============================================================================================
Nph = S1c / g1
Ne = S2c / g2
Nq = Nph + Ne
_yy = lz.nest_er_yields(64.3, params=lz.NEST_ER_LZ)
W_NEST = 1e3 * 64.3 / (_yy[0] + _yy[1])          # nestpy's own W at 2.9 g/cm3 (13.44 eV)
E_ee = {13.7: 13.7e-3 * Nq, 13.5: 13.5e-3 * Nq, round(W_NEST, 2): W_NEST * 1e-3 * Nq}
W_for = {E: E * 1e3 / Nq for E in (64.3, 67.3)}
dNq = math.hypot(S1c / g1**2 * lz.LZ['g1_err'], S2c / g2**2 * lz.LZ['g2_err'])
R['event'] = dict(Nph=Nph, Ne=Ne, Nq=Nq, dNq_from_g1g2=dNq, W_NEST_eV=W_NEST, E_ee_keV=E_ee, W_needed_eV=W_for)
print(f"[1] Nph={Nph:.0f} Ne={Ne:.1f} Nq={Nq:.0f} (+-{dNq:.0f} from g1,g2); W_NEST={W_NEST:.2f} eV")
print("    E_ee(W):", {k: round(v, 2) for k, v in E_ee.items()}, " W needed for 64.3/67.3 keV:", {k: round(v, 2) for k, v in W_for.items()})

ALPHA_NEST = 0.067366 + 0.039693 * 2.9       # NEST v2 ER exciton/ion ratio at 2.9 g/cm3 (recalled, likely) = 0.182
def N_ions(nq, alpha=ALPHA_NEST):
    return nq / (1 + alpha)

_ER_CACHE = {}
def er_mean(E):
    E = float(E)
    if E not in _ER_CACHE:
        nph, ne = lz.nest_er_yields(E, params=lz.NEST_ER_LZ)
        _ER_CACHE[E] = (nph, ne, nph + ne)
    return _ER_CACHE[E]

E_EV = round(W_NEST * 1e-3 * Nq, 1)   # 69.6 keV: ER-equivalent energy on NEST's own W
rows = []
for E in (64.3, 67.3, E_EV, 70.9):
    nph, ne, nq = er_mean(E)
    ni = N_ions(nq)
    rows.append(dict(E_keV=E, Nph_mean=nph, Ne_mean=ne, Nq_mean=nq, Ni=ni, r_mean=1 - ne / ni,
                     S1c_mean=nph * g1, log10S2c_mean=math.log10(ne * g2)))
R['er_means'] = rows
req = {}
for alpha in (0.06, 0.1, ALPHA_NEST, 0.2):
    ni = N_ions(Nq, alpha); req[round(alpha, 4)] = dict(Ni=ni, r_req=1 - Ne / ni)
R['r_required'] = req
ni_ev = N_ions(Nq)
r_req = 1 - Ne / ni_ev
_, ne70, nq70 = er_mean(E_EV)
r_mean70 = 1 - ne70 / N_ions(nq70)
R['delta_r'] = dict(E_used=E_EV, r_req=r_req, r_mean_beta_same_E=r_mean70, delta_r=r_req - r_mean70, Ne_mean_beta=ne70,
                    r_req_range_alpha_006_020=(req[0.06]['r_req'], req[0.2]['r_req']))
print(f"    alpha={ALPHA_NEST:.3f}: Ni={ni_ev:.0f}, r_req={r_req:.3f} (alpha 0.06-0.2: {req[0.06]['r_req']:.3f}-{req[0.2]['r_req']:.3f}); beta mean at {E_EV} keV: Ne={ne70:.0f}, r={r_mean70:.3f}, delta r={r_req-r_mean70:.3f}")

# ============================================================================================
# 2. Fluctuation model and band calibration
# ============================================================================================
P = lz.NEST_ER_FLUCT_LZ
def sigma_p(nq):
    """Table S4 double skew-Gaussian in x = log10(Nq)."""
    x = np.log10(nq); tot = 0.0
    for i in (1, 2):
        A, mu, sg, al = P[f'A{i}'], P[f'mu{i}'], P[f'sigma{i}'], P[f'alpha{i}']
        tot = tot + A * (1 + special.erf(al * (x - mu) / (math.sqrt(2) * sg))) * np.exp(-(x - mu)**2 / (2 * sg**2))
    return tot

K_CAL = 1.0   # set below after calibration; sigma_Ne = k * sqrt(r(1-r)Ni + sigma_p^2 Ni^2)
def sigma_Ne(ne_mean, ni, nq, k=None):
    k = K_CAL if k is None else k
    r = 1 - ne_mean / ni
    return k * np.sqrt(np.clip(r * (1 - r) * ni, 0, None) + sigma_p(nq)**2 * ni**2)

sp_ev = float(sigma_p(Nq))
sig_raw_70 = float(sigma_Ne(ne70, N_ions(nq70), nq70, k=1.0))
R['fluct_raw'] = dict(sigma_p_at_event_Nq=sp_ev, sigma_Ne_beta_raw=sig_raw_70, deficit_electrons=ne70 - Ne,
                      deficit_sigma_linear_raw=(ne70 - Ne) / sig_raw_70,
                      binomial_part=math.sqrt(r_mean70 * (1 - r_mean70) * N_ions(nq70)), sigma_p_part=sp_ev * N_ions(nq70))
print(f"[2] sigma_p(Nq={Nq:.0f}) = {sp_ev:.4f}; raw sigma_Ne(beta, {E_EV} keV) = {sig_raw_70:.0f} e -> deficit {ne70-Ne:.0f} e = {(ne70-Ne)/sig_raw_70:.2f} sigma (linear, raw width)")

# nestpy's own (default-width) fluctuation model: measure the skewness shape and the Nq Fano factor
import nestpy
det = nestpy.detectors.LZ_WS2024(); nc = nestpy.NESTcalc(det)
nestpy.RandomGen.rndm().set_seed(10)   # reproducible nestpy sampling
def nestpy_sample(E, n=40000):
    y = nc.GetYields(nestpy.interactions.beta, float(E), 2.9, lz.DRIFT_FIELD_VCM, 131.293, 54,
                     list(nestpy.default_nr_parameters), [lz.NEST_ER_LZ[f"m{i}"] for i in range(1, 11)])
    ne_s = np.empty(n); nq_s = np.empty(n)
    for i in range(n):
        q = nc.GetQuanta(y, 2.9); ne_s[i] = q.electrons; nq_s[i] = q.electrons + q.photons
    return ne_s, nq_s
ne_s, nq_s = nestpy_sample(64.3)
a_fit, loc_fit, sc_fit = stats.skewnorm.fit(ne_s)
FANO = float(nq_s.var() / nq_s.mean())
A_NEST = float(a_fit)
R['nestpy_default_fluct_64keV'] = dict(mean_Ne=ne_s.mean(), std_Ne=ne_s.std(), sample_skew=float(stats.skew(ne_s)),
                                       skewnorm_shape_a=A_NEST, fano_Nq=FANO, min_Ne=float(ne_s.min()),
                                       frac_below_ROI_edge=float((ne_s < NE_ROI_MAX).mean()),
                                       sigma_log10_fixedE=ne_s.std() / (ne_s.mean() * math.log(10)))
print(f"    nestpy default fluct @64.3 keV: std(Ne)={ne_s.std():.0f}, skew={stats.skew(ne_s):.2f} -> skewnorm a={A_NEST:.2f}; Fano(Nq)={FANO:.2f}")

# ---- Digitise Fig. 4 (axis calibration from frame and tick pixels; see details.md) ----
img = np.asarray(Image.open('inputs/figures_png/Fig4_science_sample_wbands.png').convert('RGB')).astype(int)
X0, PX_PER_PHD = 143.5, 1.30
Y0, PX_PER_DEX = 35.0, 383.0
def to_data(col, row): return (col - X0) / PX_PER_PHD, 5.0 - (row - Y0) / PX_PER_DEX
def to_px(s1, ls2): return X0 + PX_PER_PHD * s1, Y0 + PX_PER_DEX * (5.0 - ls2)
r_, g_, b_ = img[..., 0], img[..., 1], img[..., 2]
blue = (b_ > 90) & (r_ < 60) & (g_ < 60)
red = (r_ > 150) & (g_ < 80) & (b_ < 100)
black = (r_ < 50) & (g_ < 50) & (b_ < 50)
def band_lines(s1c0, mask):
    col = int(round(to_px(s1c0, 4.0)[0]))
    rows_ = np.where(mask[:, col - 2:col + 3].any(axis=1))[0]
    rows_ = rows_[(rows_ > 40) & (rows_ < 890)]
    groups = np.split(rows_, np.where(np.diff(rows_) > 2)[0] + 1) if rows_.size else []
    return sorted([to_data(col, g.mean())[1] for g in groups if g.size], reverse=True)
dig = {}
for s1c0 in (200, 300, 400, 540):
    er_l = band_lines(s1c0, blue); nr_l = band_lines(s1c0, red)
    d = dict(ER=er_l, NR=nr_l)
    if len(er_l) >= 3:
        d['ER_median'] = er_l[1]; d['ER_halfwidth_1090'] = (er_l[0] - er_l[2]) / 2; d['ER_sigma_gauss'] = d['ER_halfwidth_1090'] / 1.2816
    dig[s1c0] = d
# NR at 540: the lower dashed line is hidden by the event marker; use the two upper lines
nr540 = dig[540]['NR']
dig[540]['NR_median'] = nr540[1] if len(nr540) >= 2 else np.nan
dig[540]['NR_halfwidth_1090'] = (nr540[0] - nr540[1]) if len(nr540) >= 2 else np.nan
er_med_540 = dig[540]['ER_median']; er_hw_540 = dig[540]['ER_halfwidth_1090']
sigma_units = dict(ER_median_540=er_med_540, ER_halfwidth_1090_540=er_hw_540, ER_sigma_gauss_540=er_hw_540 / 1.2816,
                   dist_event_dex=er_med_540 - LS2_EV,
                   n_halfwidths=(er_med_540 - LS2_EV) / er_hw_540, n_gauss_sigma=(er_med_540 - LS2_EV) / (er_hw_540 / 1.2816),
                   paper_quoted=6.7,
                   NR_median_540=dig[540]['NR_median'], NR_halfwidth_540=dig[540]['NR_halfwidth_1090'],
                   NR_n_halfwidths=(dig[540]['NR_median'] - LS2_EV) / dig[540]['NR_halfwidth_1090'],
                   NR_n_gauss_sigma=(dig[540]['NR_median'] - LS2_EV) / (dig[540]['NR_halfwidth_1090'] / 1.2816), paper_quoted_NR=1.5)
R['fig4_digitised'] = dict(lines=dig, sigma_units=sigma_units)
print(f"    Fig4 digitised @540: ER median {er_med_540:.3f}, 10-90 half-width {er_hw_540:.3f} dex -> event is {sigma_units['n_halfwidths']:.2f} half-widths = {sigma_units['n_gauss_sigma']:.1f} Gaussian sigma below (paper: 6.7)")
print(f"    NR @540: median {dig[540]['NR_median']:.3f}, half-width {dig[540]['NR_halfwidth_1090']:.3f} -> {sigma_units['NR_n_halfwidths']:.2f} half-widths = {sigma_units['NR_n_gauss_sigma']:.1f} Gaussian sigma (paper: 1.5)")

# black points in Fig 4: count ROI events per S1c bin (S1c > 150; blobs)
lab, nlab = ndimage.label(black)
sizes = ndimage.sum(black, lab, range(1, nlab + 1)); coms = ndimage.center_of_mass(black, lab, range(1, nlab + 1))
pts = np.array([to_data(cc, rr) for sz, (rr, cc) in zip(sizes, coms) if 15 < sz < 400 and 40 < rr < 890 and 150 < cc < 1180])
in_roi = pts[(pts[:, 1] < ROI_LOGS2_MAX + 0.01) & (pts[:, 1] > 2.75)]
bins = [150, 200, 250, 300, 350, 400, 450, 500, 600]
counts, _ = np.histogram(in_roi[:, 0], bins=bins)
R['fig4_black_points'] = dict(bins_S1c=bins, counts_in_ROI=counts.tolist(),
                              points_S1c_gt_150=[(round(p[0], 1), round(p[1], 3)) for p in in_roi[in_roi[:, 0] > 150]])
print("    Fig4 black points in ROI per S1c bin", bins, counts.tolist())

# ---- Band MC at fixed S1c (flat ER spectrum) for a given width factor k and tail shape ----
Eg = np.linspace(1.0, 130.0, 259)
MEANS = np.array([er_mean(e) for e in Eg])
S1_EXTRA = 0.03   # fractional S1c spread beyond photon counting (assumption)
def band_mc(tail, k, n=600000, f=0.0):
    E = rng.uniform(1.0, 130.0, n)
    nq_m = np.interp(E, Eg, MEANS[:, 2]); ne_m = np.interp(E, Eg, MEANS[:, 1]) * (1 - f)
    nq = rng.normal(nq_m, np.sqrt(FANO * nq_m))
    ni = nq / (1 + ALPHA_NEST)
    sd = sigma_Ne(ne_m, ni, nq, k=k)
    if tail == 'gauss':
        ne = rng.normal(ne_m, sd)
    else:
        a = A_NEST if tail == 'nest_skew' else -2.599
        d = a / math.sqrt(1 + a * a); om = sd / math.sqrt(1 - 2 * d * d / math.pi); xi = ne_m - om * d * math.sqrt(2 / math.pi)
        ne = xi + om * stats.skewnorm.rvs(a, size=n, random_state=rng)
    ne = np.clip(ne, 1, nq - 1); nph = nq - ne
    s1 = rng.binomial(np.round(nph).astype(int), g1) * rng.normal(1, S1_EXTRA, n)
    s2 = ne * g2 * rng.normal(1, np.sqrt(1 / ne + 0.03**2), n)
    return E, s1, np.log10(s2)

def band_stats(tail, k, s1_list=(200, 300, 400, 540), n=600000):
    E, s1, ls2 = band_mc(tail, k, n)
    out = {}
    for s1c0 in s1_list:
        m = (s1 > s1c0 - 10) & (s1 < s1c0 + 10)
        q = np.percentile(ls2[m], [10, 50, 90])
        out[s1c0] = dict(n=int(m.sum()), p10=q[0], median=q[1], p90=q[2], halfwidth_1090=(q[2] - q[0]) / 2,
                         E_p10_p90=list(np.percentile(E[m], [10, 90])),
                         frac_below_ROI=float((ls2[m] < ROI_LOGS2_MAX).mean()), frac_below_event=float((ls2[m] < LS2_EV).mean()))
    return out

# calibrate k on the digitised half-widths at 300, 400, 540 (nest_skew shape)
targets = {s: dig[s]['ER_halfwidth_1090'] for s in (300, 400, 540)}
def mismatch(k):
    bs = band_stats('nest_skew', k, s1_list=(300, 400, 540), n=300000)
    return sum((bs[s]['halfwidth_1090'] / targets[s] - 1)**2 for s in targets)
kgrid = np.arange(0.40, 1.01, 0.05)
mis = [mismatch(k) for k in kgrid]
K_CAL = float(kgrid[int(np.argmin(mis))])
# refine
kfine = np.arange(K_CAL - 0.05, K_CAL + 0.051, 0.01)
misf = [mismatch(k) for k in kfine]
K_CAL = float(round(kfine[int(np.argmin(misf))], 2))
band_cal = band_stats('nest_skew', K_CAL)
band_raw = band_stats('nest_skew', 1.0)
band_gauss_cal = band_stats('gauss', K_CAL)
R['band_calibration'] = dict(k_cal=K_CAL, targets_halfwidth_1090=targets, mismatch_grid=dict(zip([round(k, 2) for k in kgrid], mis)),
                             band_cal=band_cal, band_raw=band_raw, band_gauss_cal=band_gauss_cal,
                             fixedS1c_over_fixedE_factor_88keV=float(1 + np.gradient(MEANS[:, 1], Eg)[np.argmin(abs(Eg - 88))] / np.gradient(MEANS[:, 0], Eg)[np.argmin(abs(Eg - 88))]))
print(f"    width calibration: k = {K_CAL:.2f}; MC half-widths (cal) at 300/400/540: {[round(band_cal[s]['halfwidth_1090'],3) for s in (300,400,540)]} vs digitised {[round(targets[s],3) for s in (300,400,540)]}; raw k=1: {[round(band_raw[s]['halfwidth_1090'],3) for s in (300,400,540)]}")
print(f"    MC ER median at 540: {band_cal[540]['median']:.3f} (digitised {er_med_540:.3f}); energies contributing at S1c=540: {[round(x,1) for x in band_cal[540]['E_p10_p90']]} keV")
sig_cal_70 = float(sigma_Ne(ne70, N_ions(nq70), nq70))
R['fluct_cal'] = dict(k=K_CAL, sigma_Ne_beta_cal=sig_cal_70, deficit_sigma_linear_cal=(ne70 - Ne) / sig_cal_70,
                      sigma_log10_fixedE_cal=sig_cal_70 / (ne70 * math.log(10)))
print(f"    calibrated sigma_Ne(beta, {E_EV} keV) = {sig_cal_70:.0f} e -> deficit = {(ne70-Ne)/sig_cal_70:.2f} sigma (linear Ne, fixed E)")

# ============================================================================================
# 3. Fixed-energy line model with enhancement f and tail shapes
# ============================================================================================
def skewnorm_params(mean, sd, a):
    d = a / math.sqrt(1 + a * a)
    omega = sd / math.sqrt(1 - 2 * d * d / math.pi)
    return mean - omega * d * math.sqrt(2 / math.pi), omega

class ExpTail:
    """Gaussian core with an exponential lower tail matched in value and slope at z = -z0 (heavier than Gaussian)."""
    def __init__(self, mean, sd, z0=2.0):
        self.m, self.s, self.z0 = mean, sd, z0
        self.norm = (1 - stats.norm.cdf(-z0)) + stats.norm.pdf(z0) / z0
    def cdf(self, x):
        z = np.asarray((x - self.m) / self.s, dtype=float)
        out = np.where(z < -self.z0, stats.norm.pdf(self.z0) / self.z0 * np.exp(self.z0 * (z + self.z0)),
                       stats.norm.pdf(self.z0) / self.z0 + stats.norm.cdf(z) - stats.norm.cdf(-self.z0))
        return out / self.norm
    def pdf(self, x):
        z = np.asarray((x - self.m) / self.s, dtype=float)
        out = np.where(z < -self.z0, stats.norm.pdf(self.z0) * np.exp(self.z0 * (z + self.z0)), stats.norm.pdf(z))
        return out / self.norm / self.s

def ne_dist(E, f, tail, k=None):
    nph, ne_m, nq = er_mean(E)
    ni = N_ions(nq)
    ne_m = ne_m * (1 - f)
    sd = float(sigma_Ne(ne_m, ni, nq, k=k))
    if tail == 'nest_skew':
        xi, om = skewnorm_params(ne_m, sd, A_NEST); return stats.skewnorm(A_NEST, xi, om), nq, ne_m, sd
    if tail == 'gauss':
        return stats.norm(ne_m, sd), nq, ne_m, sd
    if tail == 'neg_skew':
        xi, om = skewnorm_params(ne_m, sd, -2.599); return stats.skewnorm(-2.599, xi, om), nq, ne_m, sd
    if tail == 'exp_tail':
        return ExpTail(ne_m, sd), nq, ne_m, sd
    raise ValueError(tail)

def p_s1_in(ne, nq, lo, hi):
    nph = nq - ne
    mu = g1 * nph
    sd = np.sqrt(g1 * (1 - g1) * nph + (S1_EXTRA * mu)**2 + (g1**2) * FANO * nq)
    return stats.norm.cdf(hi, mu, sd) - stats.norm.cdf(lo, mu, sd)

# regions in (Ne, S1c):  V = event-like; G = the empty gap between V and the ROI edge; B = below V
V = dict(ne=(230.0, 290.0), s1=(500.0, 600.0))       # log10 S2c 3.90-4.00 (event 3.967 +- ~1.5 sigma_NR), S1c 500-600
G = dict(ne=(290.0, NE_ROI_MAX), s1=(475.0, 600.0))   # log10 S2c 4.00-4.15, S1c 475-600  (0 events observed)
B = dict(ne=(0.0, 230.0), s1=(475.0, 600.0))
MID = dict(ne=(0.0, NE_ROI_MAX), s1=(250.0, 500.0))   # Fig.5 middle panel ROI region
def region_prob(dist, nq, reg, n=600):
    ne_grid = np.linspace(reg['ne'][0], reg['ne'][1], n)
    return float(np.trapezoid(dist.pdf(ne_grid) * p_s1_in(ne_grid, nq, *reg['s1']), ne_grid))

TAILS = ['nest_skew', 'gauss', 'exp_tail', 'neg_skew']
FS = [0.0, 0.1, 0.2, 0.3, 0.4]
LINES = ((64.3, 'Xe124_KK'), (67.3, 'I125_K'), (E_EV, 'beta_at_Eee'))
line_tab = []
for wname, k in (('cal', K_CAL), ('raw', 1.0)):
    for E, name in LINES:
        for tail in TAILS:
            for f in FS:
                d, nq, ne_m, sd = ne_dist(E, f, tail, k=k)
                row = dict(width=wname, k=k, line=name, E_keV=E, tail=tail, f=f, Ne_mean=ne_m, sigma_Ne=sd,
                           z_event=(ne_m - Ne) / sd, z_ROI_edge=(ne_m - NE_ROI_MAX) / sd,
                           P_Ne_le_event=float(d.cdf(Ne)), P_Ne_le_ROI=float(d.cdf(NE_ROI_MAX)),
                           P_V=region_prob(d, nq, V), P_G=region_prob(d, nq, G), P_B=region_prob(d, nq, B),
                           S1c_if_Ne_event=g1 * (nq - Ne))
                row['P_G_over_P_V'] = row['P_G'] / row['P_V'] if row['P_V'] > 0 else np.inf
                line_tab.append(row)
df_lines = pd.DataFrame(line_tab)
df_lines.to_csv(f'{OUT}/line_tail_probabilities.csv', index=False)
print("[3] fixed-energy line tail probabilities (calibrated width, selected):")
sel = df_lines[(df_lines['width'] == 'cal') & (df_lines['f'].isin([0.0, 0.2, 0.4])) & (df_lines['tail'].isin(['nest_skew', 'gauss'])) & (df_lines['line'] != 'beta_at_Eee')]
print(sel[['line', 'tail', 'f', 'Ne_mean', 'sigma_Ne', 'z_event', 'P_Ne_le_event', 'P_V', 'P_G', 'S1c_if_Ne_event']].to_string(index=False, float_format=lambda x: f'{x:.3g}'))

# Energy consistency: can a 64.3 / 67.3 keV deposit give Nq = 5179 (given Ne = 269)?
def nq_resolution(E):
    nph, ne_m, nq = er_mean(E)
    nph_ev = nq - Ne
    var = (g1 * (1 - g1) * nph_ev + (S1_EXTRA * g1 * nph_ev)**2) / g1**2 + Ne * (1 + 0.03**2 * Ne) + FANO * nq
    return nq, math.sqrt(var)
en_rows = []
for E, name in LINES[:2]:
    nq_l, s = nq_resolution(E)
    en_rows.append(dict(line=name, E_keV=E, Nq_line=nq_l, sigma_Nq=s, Nq_event=Nq, pull=(Nq - nq_l) / s,
                        pull_incl_g_unc=(Nq - nq_l) / math.hypot(s, dNq), frac_diff=(Nq - nq_l) / nq_l))
R['energy_consistency'] = en_rows
print("    energy pulls (Nq_event vs line, S1-resolution / incl. g1,g2):", [(r['line'], round(r['pull'], 2), round(r['pull_incl_g_unc'], 2)) for r in en_rows])

# Continuum ER: rate per keV from the ROI count; leakage into V, G and the middle panel
acc = np.array([float(ne_dist(E, 0.0, 'nest_skew')[0].cdf(NE_ROI_MAX)) if E > 1.5 else 1.0 for E in np.linspace(1, 30, 59)])
int_acc = float(np.trapezoid(acc, np.linspace(1, 30, 59)))
N_cont_ROI = 1341 + 140.6 + 110.0 + 55.3 + 8.5
R_cont = N_cont_ROI / int_acc
Ecs = np.linspace(56, 84, 29); Emid = np.linspace(30, 100, 36)
cont = {}
for wname, k in (('cal', K_CAL), ('raw', 1.0)):
    for tail in TAILS:
        pv = float(np.trapezoid([region_prob(*ne_dist(E, 0.0, tail, k=k)[:2], V) for E in Ecs], Ecs))
        pg = float(np.trapezoid([region_prob(*ne_dist(E, 0.0, tail, k=k)[:2], G) for E in Ecs], Ecs))
        pm = float(np.trapezoid([region_prob(*ne_dist(E, 0.0, tail, k=k)[:2], MID) for E in Emid], Emid))
        cont[f'{wname}_{tail}'] = dict(N_V=pv * R_cont, N_G=pg * R_cont, N_middle_panel_ROI=pm * R_cont)
R['continuum'] = dict(int_acc_keV=int_acc, N_continuum_ROI=N_cont_ROI, rate_per_keV=R_cont, leakage=cont,
                      fig5_middle_continuous_ER_model_sum_readoff=2.5,
                      note='flat continuum with the same per-keV rate at 30-100 keV as at 2-15 keV (+-50%)')
print(f"    continuum ER: ROI acceptance integral {int_acc:.1f} keV -> {R_cont:.0f} events/keV")
for kk, v in cont.items(): print(f"      {kk:16s} N_V={v['N_V']:.2e}  N_G={v['N_G']:.2e}  N(250<S1c<500, ROI)={v['N_middle_panel_ROI']:.2f}  [Fig.5 model ~2.5, data 23 total mostly EC/Kr]")

# ============================================================================================
# 4. Candidate populations
# ============================================================================================
import periodictable as pt
import radioactivedecay as rd
ab124 = pt.Xe[124].abundance / 100.0
M_Xe = pt.Xe.mass
N_A = 6.02214076e23
mass_g = lz.LZ['fiducial_mass_t'] * 1e6
live_yr = lz.LZ['live_days'] / 365.25
T12_124 = 1.1e22   # yr, recalled (XENONnT 2022, LZ 2024 ~1.1e22; XENON1T 2019 1.8e22), reliability: likely
N124 = mass_g / M_Xe * N_A * ab124
lam124 = math.log(2) / T12_124
Ndec124 = N124 * lam124 * live_yr
BR = dict(KK=0.724, KL=0.200, KM_KN=0.056, LL_LM_MM=0.020)   # recalled (XENON1T/XENONnT 2nuECEC analyses), reliability: uncertain (+-0.03)
mode_E = dict(KK=64.3, KL=36.7, KM_KN=33.2, LL_LM_MM=9.8)    # Te binding K 31.8, L 4.9, M 1.0 keV (recalled, likely)
cand = dict(N_Xe124_atoms=N124, lambda_per_yr=lam124, decays_in_exposure=Ndec124, by_mode={k: Ndec124 * v for k, v in BR.items()},
            branching=BR, T12_yr=T12_124, abundance=ab124, decays_if_T12_1p8e22=Ndec124 * 1.1 / 1.8)
print(f"[4] 124Xe: N={N124:.3e} atoms, decays in 220 d x 4.71 t = {Ndec124:.0f} (KK {Ndec124*BR['KK']:.0f}, KL {Ndec124*BR['KL']:.0f}, KM/KN {Ndec124*BR['KM_KN']:.0f}, LL/LM/MM {Ndec124*BR['LL_LM_MM']:.0f})")
mode_rows = []
for wname, k in (('cal', K_CAL), ('raw', 1.0)):
    for kmode, E in mode_E.items():
        for f in (0.0, 0.1, 0.2, 0.3):
            d, nq, ne_m, sd = ne_dist(E, f, 'nest_skew', k=k)
            p_roi = float(d.cdf(NE_ROI_MAX))
            mode_rows.append(dict(width=wname, mode=kmode, E_keV=E, f=f, BR=BR[kmode], decays=Ndec124 * BR[kmode], Ne_mean=ne_m, sigma_Ne=sd,
                                  log10S2c_mean=math.log10(ne_m * g2), S1c_mean=g1 * (nq - ne_m), S1c_min_in_ROI=g1 * (nq - NE_ROI_MAX),
                                  P_in_ROI=p_roi, N_in_ROI=Ndec124 * BR[kmode] * p_roi))
df_modes = pd.DataFrame(mode_rows); df_modes.to_csv(f'{OUT}/xe124_modes_in_ROI.csv', index=False)
print(df_modes[df_modes['width'] == 'cal'][['mode', 'E_keV', 'f', 'decays', 'Ne_mean', 'sigma_Ne', 'S1c_mean', 'S1c_min_in_ROI', 'P_in_ROI', 'N_in_ROI']].to_string(index=False, float_format=lambda x: f'{x:.3g}'))
cand['modes'] = mode_rows
# Consistency of LZ's 21.0 +- 6.3 ROI events with a common enhancement f applied to all 124Xe modes (calibrated width)
fscan = np.linspace(0.0, 0.35, 36)
n_roi_f = []
for f in fscan:
    tot = 0.0
    for kmode, E in mode_E.items():
        d, nq, ne_m, sd = ne_dist(E, f, 'nest_skew'); tot += Ndec124 * BR[kmode] * float(d.cdf(NE_ROI_MAX))
    n_roi_f.append(tot)
n_roi_f = np.array(n_roi_f)
chi2 = ((n_roi_f - 21.0) / 6.3)**2
ok = fscan[chi2 < 1.0]
cand['f_from_21_ROI_events_all_modes'] = dict(f_grid=fscan.tolist(), N_ROI_predicted=n_roi_f.tolist(), f_best=float(fscan[np.argmin(chi2)]),
                                              f_1sigma_range=(float(ok.min()), float(ok.max())) if ok.size else None,
                                              N_ROI_at_f_0_0p1_0p2_0p3=[float(np.interp(x, fscan, n_roi_f)) for x in (0.0, 0.1, 0.2, 0.3)])
print(f"    124Xe ROI count vs common f (all modes, cal width): N(f=0,0.1,0.2,0.3) = {[round(x,1) for x in cand['f_from_21_ROI_events_all_modes']['N_ROI_at_f_0_0p1_0p2_0p3']]}; LZ 21.0+-6.3 -> f_best={cand['f_from_21_ROI_events_all_modes']['f_best']:.2f}, 1-sigma {cand['f_from_21_ROI_events_all_modes']['f_1sigma_range']}")
p_needed = 21.0 / (Ndec124 * BR['KK'])
def f_for_P(E, p, k):
    fn = lambda f: ne_dist(E, f, 'nest_skew', k=k)[0].cdf(NE_ROI_MAX) - p
    return optimize.brentq(fn, 0.0, 0.8)
cand['if_21_ROI_events_were_KK'] = {w: dict(P_needed=p_needed, f_needed=f_for_P(64.3, p_needed, k), Ne_mean_needed=er_mean(64.3)[1] * (1 - f_for_P(64.3, p_needed, k)),
                                            S1c_min=g1 * (er_mean(64.3)[2] - NE_ROI_MAX)) for w, k in (('cal', K_CAL), ('raw', 1.0))}
print(f"    if all 21 ROI 124Xe events were KK-line leakage: P(Ne<409)={p_needed:.3f} -> f = {cand['if_21_ROI_events_were_KK']['cal']['f_needed']:.2f} (cal) / {cand['if_21_ROI_events_were_KK']['raw']['f_needed']:.2f} (raw); they would sit at S1c > {g1*(er_mean(64.3)[2]-NE_ROI_MAX):.0f} phd where Fig.4 shows 0 events")

# 125I timing and population
hl = {n: rd.Nuclide(n).half_life('d') for n in ['I-125', 'Xe-125', 'Xe-127', 'Xe-131m', 'Xe-129m', 'Xe-133', 'Kr-83m']}
t_eff = lz.LZ['I125_eff_halflife_d'][0]
lam_eff = math.log(2) / t_eff; lam_dec = math.log(2) / hl['I-125']; lam_x = math.log(2) / hl['Xe-125']
frac_decay_in_LXe = lam_dec / lam_eff
def I125_rate(t):   # normalised 125I decay-rate profile after prompt 125Xe production (Bateman: 125Xe -> 125I, removal+decay)
    return lam_dec * lam_x / (lam_eff - lam_x) * (np.exp(-lam_x * t) - np.exp(-lam_eff * t)) / frac_decay_in_LXe
tpeak = math.log(lam_eff / lam_x) / (lam_eff - lam_x)
I125 = dict(half_lives_d=hl, t_eff_d=t_eff, frac_atoms_decaying_in_LXe=frac_decay_in_LXe, t_peak_d=tpeak,
            rate_8d_over_peak=float(I125_rate(8.0) / I125_rate(tpeak)), frac_decays_after_8d=integrate.quad(I125_rate, 8, 300)[0],
            frac_decays_day7to9=integrate.quad(I125_rate, 7, 9)[0], frac_decays_within_20d=integrate.quad(I125_rate, 0, 20)[0],
            Xe125_remaining_8d=2**(-8.0 / hl['Xe-125']), Xe127_remaining_8d=2**(-8.0 / hl['Xe-127']),
            frac_livetime_within_20d_of_3_cals=3 * 20 / lz.LZ['live_days'])
I125_L_E = 40.4; BR_L = 0.20   # 125I L/M-capture branch: 35.5 keV level + L binding 4.9 keV; K:L ~ 80:20 (recalled, likely)
I125_rows = []
for wname, k in (('cal', K_CAL), ('raw', 1.0)):
    for f in (0.0, 0.1, 0.2, 0.3):
        d, nq, ne_m, sd = ne_dist(I125_L_E, f, 'nest_skew', k=k)
        pL = float(d.cdf(NE_ROI_MAX)); N_tot = 8.9 / (BR_L * pL) if pL > 0 else np.inf
        I125_rows.append(dict(width=wname, f=f, P_L_in_ROI=pL, N_I125_total=N_tot, N_I125_K67=N_tot * (1 - BR_L), S1c_min_L_in_ROI=g1 * (nq - NE_ROI_MAX)))
I125['population_from_8p9_ROI_events'] = I125_rows
cand['I125'] = I125
R['candidates'] = cand
print(f"    125I: T1/2={hl['I-125']:.1f} d, eff {t_eff} d -> {frac_decay_in_LXe*100:.1f}% of 125I atoms decay in the LXe; rate at +8 d = {I125['rate_8d_over_peak']:.2f} of peak (peak {tpeak:.1f} d); {I125['frac_decays_after_8d']*100:.0f}% of 125I decays occur after day 8; 125Xe left after 8 d: {I125['Xe125_remaining_8d']:.1e}")
print("    125I decays implied by 8.9 ROI events via L-capture leakage (cal):", [(r['f'], round(r['P_L_in_ROI'], 3), round(r['N_I125_total'])) for r in I125_rows if r['width'] == 'cal'])

# ============================================================================================
# 5. Gap argument
# ============================================================================================
fig5_bottom = dict(bins_sigmaNR=[(1, 1.5), (1.5, 2), (2, 2.5), (2.5, 3), (3, 3.5), (3.5, 4), (4, 4.5), (4.5, 5), (5, 5.5)],
                   total=[3e-4, 8e-4, 1e-3, 5e-4, 5e-4, 1e-3, 2.2e-3, 2.3e-3, 3e-4],
                   magenta_EC_IC=[0, 1.3e-4, 2.5e-4, 3e-4, 3e-4, 3e-4, 4e-4, 4.5e-4, 1e-4],
                   floor_total_per_bin=4e-5, floor_components=dict(accidentals=2.5e-5, MSSI=1.5e-5, NR=5e-6), event_bin=(-2, -1.5),
                   note='read by eye from the PNG crop; 0.5 sigma_NR bins; ER components drop off-scale below +1 sigma_NR')
plateau = np.array(fig5_bottom['total']); mag = np.array(fig5_bottom['magenta_EC_IC'])
er_sum = plateau.sum(); mag_sum = mag.sum()
tot_check = er_sum + fig5_bottom['floor_total_per_bin'] * 32
plateau_low = plateau[:3].mean()
mu_flat_V = plateau_low * 4          # +-1 sigma_NR around the event (4 bins of 0.5)
mu_flat_gap = plateau_low * 7        # -2.5 .. +1
mu_bkg_V = fig5_bottom['floor_total_per_bin'] * 4
gap = dict(fig5_bottom_read=fig5_bottom, ER_plateau_sum=er_sum, EC_IC_sum=mag_sum, total_check=tot_check, paper_total=lz.LZ['bkg_highS1_panel'][0],
           plateau_low_edge_per_bin=plateau_low, flat_extrapolation_mu_V=mu_flat_V, flat_extrapolation_mu_minus2p5_to_plus1=mu_flat_gap,
           modelled_floor_V=mu_bkg_V, ratio_flat_to_floor=mu_flat_V / mu_bkg_V, P_ge1_flat=1 - math.exp(-mu_flat_V),
           sigmaNR_bin_dex=0.5 * dig[540]['NR_halfwidth_1090'], gap_plus1_to_event_dex=2.5 * dig[540]['NR_halfwidth_1090'])
print(f"[5] Fig5 bottom: ER plateau sum {er_sum:.4f} (EC/IC {mag_sum:.4f}); +floor -> {tot_check:.4f} vs paper 0.0106; plateau at +1..+2.5: {plateau_low:.1e}/bin")
print(f"    flat extrapolation to +-1 sigma_NR of the event: mu={mu_flat_V:.1e} (x{gap['ratio_flat_to_floor']:.0f} the modelled floor {mu_bkg_V:.1e}); P(>=1)={gap['P_ge1_flat']:.2%}")

def cond_tail(zedge, zev, shape, scale=1.0):
    if shape == 'gauss': return stats.norm.sf(zev) / stats.norm.sf(zedge)
    if shape == 'exp': return math.exp(-(zev - zedge) / scale)
    if shape == 'flat': return 1.0 / (zev + 1.0 - zedge)
z_edge = (er_med_540 - ROI_LOGS2_MAX) / (er_hw_540 / 1.2816); z_ev = sigma_units['n_gauss_sigma']
gap['cond_tail'] = dict(z_edge_gauss=z_edge, z_event_gauss=z_ev, gauss=cond_tail(z_edge, z_ev, 'gauss'), exp_1sigma=cond_tail(z_edge, z_ev, 'exp', 1.0),
                        exp_0p5sigma=cond_tail(z_edge, z_ev, 'exp', 0.5), flat=cond_tail(z_edge, z_ev, 'flat'))
print(f"    ROI edge is {z_edge:.1f} Gaussian sigma below the ER median at S1c=540, the event {z_ev:.1f}; P(>= event | beyond edge):", {k: f'{v:.1e}' for k, v in gap['cond_tail'].items() if k in ('gauss', 'exp_1sigma', 'exp_0p5sigma', 'flat')})

# Likelihood of "1 in V, 0 in G" for a leaking line with shape ratio rho = P_G/P_V, maximised over the normalisation
def L_obs(muV, rho): return muV * math.exp(-muV * (1 + rho))
lik_rows = []
for wname in ('cal', 'raw'):
    for tail in TAILS:
        for name in ('Xe124_KK', 'I125_K'):
            for f in (0.2, 0.3, 0.4):
                sub = df_lines[(df_lines['width'] == wname) & (df_lines['line'] == name) & (df_lines['tail'] == tail) & (df_lines['f'] == f)].iloc[0]
                rho = sub['P_G_over_P_V']; muV_best = 1.0 / (1.0 + rho)
                Nline = Ndec124 * BR['KK'] if name == 'Xe124_KK' else 500.0
                muV_pred = Nline * sub['P_V']; muG_pred = Nline * sub['P_G']
                lik_rows.append(dict(width=wname, tail=tail, line=name, f=f, rho_G_over_V=rho, muV_best=muV_best, L_best=L_obs(muV_best, rho),
                                     L_bkg_floor=mu_bkg_V * math.exp(-mu_bkg_V), L_ratio_best_over_floor=L_obs(muV_best, rho) / (mu_bkg_V * math.exp(-mu_bkg_V)),
                                     N_line_assumed=Nline, muV_pred=muV_pred, muG_pred=muG_pred, P_0_in_G_pred=math.exp(-muG_pred),
                                     L_pred=L_obs(muV_pred, rho) if muV_pred > 0 else 0.0))
df_lik = pd.DataFrame(lik_rows); df_lik.to_csv(f'{OUT}/gap_likelihood.csv', index=False)
# (f, N_I125) scan for the 125I K-line: expected event-like and gap counts, and P(0 in G)
scan_rows = []
for tail in ('nest_skew', 'gauss', 'exp_tail'):
    for f in (0.1, 0.2, 0.25, 0.3, 0.4):
        d, nq, ne_m, sd = ne_dist(67.3, f, tail)
        pv, pg = region_prob(d, nq, V), region_prob(d, nq, G)
        for N in (100, 1000, 10000):
            scan_rows.append(dict(tail=tail, f=f, N_I125_K=N, N_V=N * pv, N_G=N * pg, P_0_in_G=math.exp(-N * pg), P_ge1_in_V=1 - math.exp(-N * pv)))
df_scan = pd.DataFrame(scan_rows); df_scan.to_csv(f'{OUT}/I125_f_N_scan.csv', index=False)
print("    125I K-line (f, N) scan, Gaussian tail:")
print(df_scan[df_scan['tail'] == 'gauss'].pivot(index='f', columns='N_I125_K', values='N_V').to_string(float_format=lambda x: f'{x:.2e}'))
print("    P(0 in gap G):"); print(df_scan[df_scan['tail'] == 'gauss'].pivot(index='f', columns='N_I125_K', values='P_0_in_G').to_string(float_format=lambda x: f'{x:.3f}'))
gap['mu_bkg_V_modelled'] = mu_bkg_V
R['gap'] = gap
print(df_lik[(df_lik['width'] == 'cal') & (df_lik['tail'].isin(['nest_skew', 'gauss']))][['tail', 'line', 'f', 'rho_G_over_V', 'muV_best', 'L_ratio_best_over_floor', 'muV_pred', 'muG_pred', 'P_0_in_G_pred']].to_string(index=False, float_format=lambda x: f'{x:.3g}'))

# ============================================================================================
# 6. Figures
# ============================================================================================
# Fig 1: (S1c, log S2c) plane
nr_E = np.linspace(5, 300, 120); nr = np.array([lz.nest_nr_yields(e) for e in nr_E])
fig, ax = plt.subplots(figsize=(7.4, 5.4))
ax.plot(MEANS[:, 0] * g1, np.log10(MEANS[:, 1] * g2), color='tab:blue', lw=2, label='ER mean at fixed $E$ (NEST, Table S3)')
sdv = sigma_Ne(MEANS[:, 1], N_ions(MEANS[:, 2]), MEANS[:, 2])
ax.fill_between(MEANS[:, 0] * g1, np.log10((MEANS[:, 1] - sdv) * g2), np.log10((MEANS[:, 1] + sdv) * g2), color='tab:blue', alpha=0.15,
                label=fr'$\pm1\sigma_{{N_e}}$ at fixed $E$ (k={K_CAL:.2f} calibrated)')
# digitised band points
for s in (200, 300, 400, 540):
    if 'ER_median' in dig[s]:
        ax.errorbar(s, dig[s]['ER_median'], yerr=dig[s]['ER_halfwidth_1090'], fmt='s', color='navy', ms=4, capsize=3, label='Fig. 4 ER band (digitised, 10-90%)' if s == 200 else None)
ax.plot(nr[:, 0] * g1, np.log10(nr[:, 1] * g2), color='tab:red', lw=2, label='NR mean (NEST, Table S5)')
ax.axhline(ROI_LOGS2_MAX, color='gray', ls='--', lw=1); ax.axvline(600, color='gray', ls='--', lw=1)
ax.axvspan(475, 600, ymin=(4.0 - 3.4) / 1.6, ymax=(4.15 - 3.4) / 1.6, color='orange', alpha=0.2, label='gap region G (0 events)')
ax.axvspan(500, 600, ymin=(3.9 - 3.4) / 1.6, ymax=(4.0 - 3.4) / 1.6, color='green', alpha=0.2, label='event region V')
for E, c, lab_ in ((64.3, 'tab:purple', r'$^{124}$Xe KK 64.3 keV'), (67.3, 'tab:green', r'$^{125}$I K 67.3 keV')):
    nq_l = er_mean(E)[2]; ne_l = np.linspace(50, nq_l * 0.35, 100)
    ax.plot(g1 * (nq_l - ne_l), np.log10(ne_l * g2), color=c, lw=1.2, ls=':', label=f'{lab_}: constant-$N_q$ locus')
    for f, mk in ((0.0, 'o'), (0.1, 's'), (0.2, 'D'), (0.3, '^'), (0.4, 'v')):
        ne_m = er_mean(E)[1] * (1 - f); ax.plot(g1 * (nq_l - ne_m), math.log10(ne_m * g2), marker=mk, color=c, ms=6, ls='none')
ax.plot(S1c, LS2_EV, marker='*', color='k', ms=16, ls='none', label='event (540.1 phd, 10$^{3.97}$ phd)')
ax.annotate(f'$r_{{req}}$ = {r_req:.3f}\n(β mean {r_mean70:.3f})', (S1c, LS2_EV), xytext=(600, 3.75), fontsize=9, arrowprops=dict(arrowstyle='->', color='k'))
ax.set_xlim(0, 700); ax.set_ylim(3.4, 5.0); ax.set_xlabel('S1c [phd]'); ax.set_ylabel(r'$\log_{10}$(S2c [phd])')
ax.set_title('Enhanced-recombination lines: markers $f$ = 0, 0.1, 0.2, 0.3, 0.4 along constant-$N_q$ loci', fontsize=10)
ax.legend(fontsize=7, loc='lower right', ncol=2); fig.tight_layout(); fig.savefig(f'{FIG}/P010_fig1_bands_event_lines.png', dpi=160); plt.close(fig)

# Fig 2: N_V and N_G vs f (calibrated width), both lines, four tails
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
fgrid = np.linspace(0, 0.5, 51); NKK = Ndec124 * BR['KK']
for ax, (name, E, Nline, ttl) in zip(axs, (('Xe124_KK', 64.3, NKK, r'$^{124}$Xe KK, %d decays' % NKK), ('I125_K', 67.3, 500.0, r'$^{125}$I K-capture, 500 decays (order of magnitude)'))):
    for tail, c in zip(TAILS, ('tab:blue', 'k', 'tab:orange', 'tab:red')):
        nv, ng = [], []
        for f in fgrid:
            d, nq, ne_m, sd = ne_dist(E, f, tail); nv.append(Nline * region_prob(d, nq, V)); ng.append(Nline * region_prob(d, nq, G))
        ax.plot(fgrid, nv, color=c, lw=2, label=f'{tail}: $N_V$ (event-like)'); ax.plot(fgrid, ng, color=c, lw=1, ls='--', label=f'{tail}: $N_G$ (gap; 0 observed)')
    ax.axhline(mu_bkg_V, color='gray', ls=':', label='modelled non-ER floor in V'); ax.axhline(1, color='gray', lw=0.5)
    ax.set_yscale('log'); ax.set_ylim(1e-7, 100); ax.set_xlabel('recombination enhancement $f$ (mean $N_e$ reduced by $f$)'); ax.set_title(ttl, fontsize=10); ax.grid(alpha=0.3)
axs[0].set_ylabel(f'expected events in 2.84 t yr (width k={K_CAL:.2f})'); axs[0].legend(fontsize=6.5, ncol=2, loc='upper left')
fig.tight_layout(); fig.savefig(f'{FIG}/P010_fig2_NV_NG_vs_enhancement.png', dpi=160); plt.close(fig)

# Fig 3: 125I / 125Xe time profile
t = np.linspace(0, 30, 600)
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(t, I125_rate(t) / I125_rate(tpeak), label=r'$^{125}$I decay rate (T$_{1/2}^{eff}$ = 3.6 d, fed by $^{125}$Xe 16.9 h)')
ax.plot(t, 2**(-t / hl['Xe-125']), label=r'$^{125}$Xe (16.9 h)'); ax.plot(t, 2**(-t / hl['Xe-127']), label=r'$^{127}$Xe (36.4 d)')
ax.axvline(8, color='k', ls='--', label='event: +8 d after AmBe')
ax.set_yscale('log'); ax.set_ylim(1e-4, 1.5); ax.set_xlabel('days after AmBe calibration'); ax.set_ylabel('relative activity'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f'{FIG}/P010_fig3_I125_timing.png', dpi=160); plt.close(fig)

def _conv(o):
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, dict): return {str(k): _conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_conv(v) for v in o]
    return o
with open(f'{OUT}/P010_results.json', 'w') as fh:
    json.dump(_conv(R), fh, indent=1)
print("saved", f'{OUT}/P010_results.json')
