"""
P024_nr_band_width.py -- How wide is the NR band at 250 keV?  Sensitivity of LZ's "1.5 sigma below the NR median"
statement (arXiv:2609.02823) to the NEST fluctuation model, the p(E) yield break, g1/g2, the drift field and the
detector terms; AmBe calibration statistics near 250 keV; NR-PDF overlap with the MSSI contours of Fig. S1a.

Run from the simulation root:  .venv/bin/python output/code/P024_nr_band_width.py
Outputs: output/work/P024/*.csv, results.json, figures/*.png

Sections
 1. fixed-S1c NR band (flat spectrum 200-320 keV, |S1c - 540.1| < 10 phd) with nestpy GetQuanta fluctuations
    (LZ_WS2024 nr_er_width_parameters) + detector model; ~45 variants (width parameters +-30 %, yield scale, break
    on/off, g1/g2, position smearing, extraction efficiency, S2 Fano, field); band sigma, 10-90 %, event distance,
    P(log10 S2c <= observed)
 2. analytical variance decomposition of log10 S2c at 248 keV
 3. digitisation of the drawn 10-90 % NR lines (Figs. 2 and 4) at S1c = 480-600 phd; the k needed to reproduce them
 4. AmBe statistics: count of purple AmBe points in Fig. 2 per S1c bin; recoil-spectrum estimate of the 230-270 keV
    fraction (isotropic vs black-disk angular distribution); precision on the band median and width
 5. MSSI 68 %/95 % contour edges at S1c = 540 from Fig. S1a; fraction of the NR PDF inside them
"""
from __future__ import annotations
import json, math, os, sys, time
import numpy as np
import pandas as pd
from scipy import stats, special, ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import image as mpimg

ROOT = "/Users/reza/LZ_simulation"
os.chdir(ROOT)
sys.path.insert(0, "output/code")
from common import lzcommon as lz
import nestpy

OUT = "output/work/P024"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(20260908)
nestpy.RandomGen.rndm().set_seed(20260908)

G1, G1E = lz.LZ["g1"], lz.LZ["g1_err"]          # 0.110 +- 0.002 phd/photon (paper)
G2, G2E = lz.LZ["g2"], lz.LZ["g2_err"]          # 34.5 +- 1.1 phd/electron (paper)
S1, S2 = lz.LZ["ev_S1c"], lz.LZ["ev_S2c"]       # 540.1 phd, 9268 phd
L2 = math.log10(S2)                              # 3.9670
FIELD = lz.DRIFT_FIELD_VCM                       # 96.5 V/cm (nestpy LZ_WS2024 central field)
DENSITY = 2.9
DEX = 1.0 / math.log(10.0)                       # 0.4343: relative sigma -> dex

DET = nestpy.detectors.LZ_WS2024()
NC = nestpy.NESTcalc(DET)
W_LZ = list(DET.nr_er_width_parameters)          # 13 entries; NR ones identified empirically below
S2_FANO = DET.get_s2Fano()                       # 4.0
SPE_RES = DET.get_sPEres()                       # 0.338
P_DPHE = DET.get_P_dphe()                        # 0.214 (double photoelectron emission probability)
EXT_EFF = 0.80        # recalled: LZ SR1 electron extraction efficiency 80.5 % (uncertain); nestpy CalculateG2 gives 0.726
POS_RES = 0.02        # assumed residual of position corrections (relative), on S1c and S2c (as P009)
# P009's paper-contour energy scale: Nq = 11.32 E^1.1117 (from the Fig. 2/4 constant-energy contours)
P9 = json.load(open("output/work/P009/results.json"))
ALPHA_P, BETA_P = P9["digitised"]["alpha_paper"], P9["digitised"]["beta_paper"]
S1_HALF = 10.0        # half-width of the fixed-S1c slice [phd]

# ----------------------------------------------------------------------------------------------------
# yields
# ----------------------------------------------------------------------------------------------------
NR_NOBREAK = {k: v for k, v in lz.NEST_NR_LZ.items() if k not in ("a", "b", "E0")}
PARAMS = {"paper": lz.NEST_NR_LZ, "LZtab": lz.NEST_NR_LZ, "nobreak": NR_NOBREAK, "default": lz.NEST_NR_DEFAULT}


def yield_result(E, scale="paper", field=FIELD):
    """nestpy YieldResult at energy E for a yield variant.  'paper' = LZ Table S5 charge yield with the total
    quanta rescaled to the paper's own contour scale (P009): photons shifted by the mean difference."""
    y = NC.GetYields(nestpy.interactions.NR, float(E), DENSITY, field, 131.293, 54,
                     lz.nest_nr_params_vector(E, PARAMS[scale]))
    shift = 0.0
    if scale == "paper":
        shift = ALPHA_P * E ** BETA_P - (y.PhotonYield + y.ElectronYield)
    return y, shift


def quanta(E, n, scale="paper", W=W_LZ, field=FIELD):
    """n GetQuanta draws (photons, electrons, ions) at energy E."""
    y, shift = yield_result(E, scale, field)
    out = np.empty((n, 3))
    for i in range(n):
        q = NC.GetQuanta(y, DENSITY, W)
        out[i] = q.photons, q.electrons, q.ions
    out[:, 0] += shift
    return out, y


def detect(q, g1=G1, g2=G2, pos_res=POS_RES, ext=EXT_EFF, s2fano=S2_FANO, perfect=False, k_e=1.0):
    """Detector model.  S1c: binomial(Nph, g1) photons detected, double-PE emission P_dphe (phd = phe*(1+P) on
    average, absorbed into g1 which is in phd/photon: we draw the binomial in phd directly), PMT single-phe
    resolution, position-correction residual.  S2c: binomial extraction, SE-size Fano, position residual.
    k_e rescales the electron fluctuation about its mean at the quanta level (P010-style calibration factor)."""
    nph = np.clip(np.round(q[:, 0]).astype(int), 0, None)
    ne = q[:, 1].copy()
    if k_e != 1.0:
        ne = ne.mean() + k_e * (ne - ne.mean())
    ne = np.clip(np.round(ne).astype(int), 0, None)
    if perfect:
        return g1 * nph.astype(float), np.log10(np.clip(g2 * ne, 1, None))
    n1 = rng.binomial(nph, g1).astype(float)
    n1 = n1 * (1 + rng.normal(0, SPE_RES / np.sqrt(np.clip(n1, 1, None))))
    s1c = n1 * (1 + rng.normal(0, pos_res, n1.size)) if pos_res > 0 else n1
    next_ = rng.binomial(ne, ext).astype(float)
    se = g2 / ext
    s2 = next_ * se
    if s2fano > 0:
        s2 = s2 * (1 + rng.normal(0, np.sqrt(s2fano / np.clip(s2, 1, None))))
    if pos_res > 0:
        s2 = s2 * (1 + rng.normal(0, pos_res, s2.size))
    return s1c, np.log10(np.clip(s2, 1, None))


def band_stats(l2, s1c=None, E=None):
    p10, p16, p50, p84, p90 = np.percentile(l2, [10, 16, 50, 84, 90])
    sd = l2.std()
    d = dict(n=int(l2.size), p10=p10, p50=p50, p90=p90, sd=sd, sd_1684=0.5 * (p84 - p16),
             halfwidth_1090=0.5 * (p90 - p10), sigma_eq_1090=0.5 * (p90 - p10) / 1.2816,
             skew=float(stats.skew(l2)), offset=p50 - L2, nsig=(p50 - L2) / sd,
             nsig_1090=(p50 - L2) / (0.5 * (p90 - p10) / 1.2816),
             P_below=float((l2 <= L2).mean()), P_below_gauss=float(stats.norm.cdf((L2 - p50) / sd)))
    if E is not None:
        d["mean_E"] = float(np.mean(E)); d["sd_E"] = float(np.std(E))
    return d


# ----------------------------------------------------------------------------------------------------
# 1. fixed-S1c band, variants
# ----------------------------------------------------------------------------------------------------
E_LO, E_HI, N_BAND = 200.0, 320.0, 300000
E_edges = np.arange(E_LO, E_HI + 0.5, 1.0)


def fixed_s1c_band(scale="paper", W=W_LZ, field=FIELD, det_kw=None, n=N_BAND, s1_half=S1_HALF):
    det_kw = det_kw or {}
    Eflat = rng.uniform(E_LO, E_HI, n)
    idx = np.digitize(Eflat, E_edges) - 1
    s1_all = np.empty(n); l2_all = np.empty(n)
    for k in range(E_edges.size - 1):
        m = idx == k
        if not m.any():
            continue
        q, _ = quanta(0.5 * (E_edges[k] + E_edges[k + 1]), int(m.sum()), scale, W, field)
        s1c, l2 = detect(q, **det_kw)
        s1_all[m] = s1c; l2_all[m] = l2
    sel = np.abs(s1_all - S1) < s1_half
    return band_stats(l2_all[sel], E=Eflat[sel]), (s1_all[sel], l2_all[sel], Eflat[sel])


t0 = time.time()
VAR = {}
SAMPLES = {}
# baseline and yield-scale variants
for name, kw in [("baseline (paper scale, LZ widths)", dict()),
                 ("Table S5 as printed", dict(scale="LZtab")),
                 ("no p(E) break", dict(scale="nobreak")),
                 ("NEST default yields", dict(scale="default")),
                 ("field 90 V/cm", dict(field=90.0)),
                 ("field 100 V/cm", dict(field=100.0))]:
    VAR[name], SAMPLES[name] = fixed_s1c_band(**kw)
    print("%-40s sd=%.4f  10-90 half=%.4f  nsig=%.2f  (%.0f s)" % (name, VAR[name]["sd"], VAR[name]["halfwidth_1090"],
                                                                 VAR[name]["nsig"], time.time() - t0))
N_SMALL = 150000
# width parameters +-30 % (entries 11, 12 are zero: test +-0.3 absolute instead)
for i in range(13):
    for f, tag in ((0.7, "x0.7"), (1.3, "x1.3")):
        W = list(W_LZ)
        if W[i] == 0.0:
            W[i] = 0.3 if f > 1 else -0.3
            tag = "=+0.3" if f > 1 else "=-0.3"
        else:
            W[i] = W[i] * f
        name = f"width[{i}] {tag}"
        VAR[name], _ = fixed_s1c_band(W=W, n=N_SMALL)
        print("%-40s sd=%.4f  nsig=%.2f" % (name, VAR[name]["sd"], VAR[name]["nsig"]))
# detector variants
for name, kw in [("g1 -2%", dict(det_kw=dict(g1=G1 - G1E))), ("g1 +2%", dict(det_kw=dict(g1=G1 + G1E))),
                 ("g2 -3%", dict(det_kw=dict(g2=G2 - G2E))), ("g2 +3%", dict(det_kw=dict(g2=G2 + G2E))),
                 ("position residual 0%", dict(det_kw=dict(pos_res=0.0))),
                 ("position residual 3%", dict(det_kw=dict(pos_res=0.03))),
                 ("position residual 6%", dict(det_kw=dict(pos_res=0.06))),
                 ("extraction eff. 0.726", dict(det_kw=dict(ext=0.726))),
                 ("extraction eff. 1.0", dict(det_kw=dict(ext=1.0))),
                 ("S2 Fano 0", dict(det_kw=dict(s2fano=0.0))),
                 ("quanta only (perfect detector)", dict(det_kw=dict(perfect=True))),
                 ("binomial floor: omega=0, Fano->0, perfect", dict(W=[1e-6, 1e-6, 0.0] + W_LZ[3:], det_kw=dict(perfect=True))),
                 ("omega = 0 (full detector)", dict(W=W_LZ[:2] + [0.0] + W_LZ[3:])),
                 ("k_e = 0.5 (electron fluct. halved)", dict(det_kw=dict(k_e=0.5))),
                 ("k_e = 0 (no electron fluct.)", dict(det_kw=dict(k_e=0.0)))]:
    VAR[name], SAMPLES[name] = fixed_s1c_band(n=N_SMALL, **kw)
    print("%-40s sd=%.4f  nsig=%.2f  (%.0f s)" % (name, VAR[name]["sd"], VAR[name]["nsig"], time.time() - t0))
VARDF = pd.DataFrame(VAR).T
VARDF.index.name = "variant"
VARDF.to_csv(os.path.join(OUT, "band_variants.csv"))
base = VAR["baseline (paper scale, LZ widths)"]

# fixed-energy (248 keV) distributions: P(log10 S2c <= observed | genuine 248 keV NR), unconditional and given the S1c slice
FIXE = {}
for E in (248.0, 240.0, 256.0, 265.0):
    q, y = quanta(E, 200000)
    s1c, l2 = detect(q)
    sel = np.abs(s1c - S1) < S1_HALF
    FIXE[str(E)] = dict(all=band_stats(l2), in_slice=band_stats(l2[sel]),
                        mean_S1c=float(s1c.mean()), sd_S1c=float(s1c.std()),
                        P_below_uncond=float((l2 <= L2).mean()), P_below_in_slice=float((l2[sel] <= L2).mean()))
    print("E=%.0f: S1c %.0f+-%.0f, log S2c %.4f sd %.4f (slice %.4f), P(<=obs) %.4f / %.4f" % (
        E, s1c.mean(), s1c.std(), l2.mean(), l2.std(), l2[sel].std(), FIXE[str(E)]["P_below_uncond"], FIXE[str(E)]["P_below_in_slice"]))

# ----------------------------------------------------------------------------------------------------
# 2. analytical variance decomposition at 248 keV (paper scale)
# ----------------------------------------------------------------------------------------------------
q248, y248 = quanta(248.0, 100000)
Nph_m, Ne_m = q248[:, 0].mean(), q248[:, 1].mean()
Ni_m = q248[:, 2].mean()
Nq_m = Nph_m + Ne_m
NexONi = y248.ExcitonRatio
r = 1.0 - y248.ElectronYield / (y248.ElectronYield + y248.PhotonYield) * (1 + NexONi)   # NEST: r = 1 - (1+Nex/Ni) elecFrac
elecFrac = y248.ElectronYield / (y248.ElectronYield + y248.PhotonYield)
omega = W_LZ[2] * math.exp(-0.5 * ((elecFrac - W_LZ[3]) / W_LZ[4]) ** 2)   # NEST NR form (recalled: likely); verified vs MC below
V_binom = r * (1 - r) * Ni_m
V_omega = (omega * Ni_m) ** 2
V_ionfano = ((1 - r) ** 2) * W_LZ[0] * Ni_m         # ion-number fluctuation propagated to Ne at fixed r
sd_Ne_mc = q248[:, 1].std()
sd_Ni_mc = q248[:, 2].std()
rel = lambda V: V / Ne_m ** 2
ext_rel = (1 - EXT_EFF) / (EXT_EFF * Ne_m)
se_rel = S2_FANO / (EXT_EFF * Ne_m * G2 / EXT_EFF)
pos_rel = POS_RES ** 2
# energy mixing inside the S1c slice: dlog10S2c/dE x sigma_E, sigma_E = sd(S1c at fixed E)/(dS1c/dE)
yA, shA = yield_result(243.0); yB, shB = yield_result(253.0)
dS1_dE = G1 * ((yB.PhotonYield + shB) - (yA.PhotonYield + shA)) / 10.0
dl2_dE = (math.log10(G2 * yB.ElectronYield) - math.log10(G2 * yA.ElectronYield)) / 10.0
sd_S1c_248 = FIXE["248.0"]["sd_S1c"]
sigE_slice = math.sqrt(max(S1_HALF ** 2 / 3 + sd_S1c_248 ** 2, 0)) / dS1_dE   # slice half-width (uniform) + S1 resolution
mix_rel = (dl2_dE * sigE_slice / DEX) ** 2
DECOMP = dict(E=248.0, Nph=Nph_m, Ne=Ne_m, Ni=Ni_m, Nex=Ni_m * NexONi, Nq=Nq_m, NexONi=NexONi, r=r,
              elecFrac=elecFrac, omega=omega, sd_Ne_MC=sd_Ne_mc, sd_Ni_MC=sd_Ni_mc, sd_Ni_pred=math.sqrt(W_LZ[0] * Ni_m),
              sd_Ne_pred_binom_omega_fano=math.sqrt(V_binom + V_omega + V_ionfano),
              terms_dex={"recombination binomial r(1-r)Ni": DEX * math.sqrt(rel(V_binom)),
                         "recombination omega^2 Ni^2": DEX * math.sqrt(rel(V_omega)),
                         "ion Fano (W[0]) -> Ne": DEX * math.sqrt(rel(V_ionfano)),
                         "extraction binomial (eps=%.2f)" % EXT_EFF: DEX * math.sqrt(ext_rel),
                         "SE-size Fano (F=%.0f)" % S2_FANO: DEX * math.sqrt(se_rel),
                         "position residual (%.0f%%)" % (100 * POS_RES): DEX * math.sqrt(pos_rel),
                         "energy mixing in S1c slice": DEX * math.sqrt(mix_rel)},
              dS1c_dE=dS1_dE, dlog10S2c_dE=dl2_dE, sigma_E_slice=sigE_slice)
DECOMP["total_dex_analytic"] = math.sqrt(sum(v ** 2 for v in DECOMP["terms_dex"].values()))
DECOMP["quanta_only_dex_analytic"] = math.sqrt(sum(DECOMP["terms_dex"][k] ** 2 for k in list(DECOMP["terms_dex"])[:3]))
DECOMP["MC_fixedE_quanta_only_dex"] = DEX * sd_Ne_mc / Ne_m
DECOMP["MC_fixedS1c_baseline_dex"] = VAR["baseline (paper scale, LZ widths)"]["sd"]
print("DECOMP", json.dumps(DECOMP, indent=1, default=float))

# ----------------------------------------------------------------------------------------------------
# 3. digitisation of the drawn NR lines (Figs. 2 and 4) and the k needed to reproduce them
# ----------------------------------------------------------------------------------------------------
def load_png(name):
    return (mpimg.imread(f"inputs/figures_png/{name}.png")[:, :, :3] * 255).astype(int)


def frame(rgb):
    dark = rgb.sum(axis=2) < 150
    H, W_ = dark.shape
    cols = np.where(dark.sum(axis=0) > 0.5 * H)[0]; rows = np.where(dark.sum(axis=1) > 0.5 * W_)[0]
    return cols.min(), cols.max(), rows.min(), rows.max()


def roi_top_row(rgb, xl, xr, yt, yb):
    """row of the dashed grey ROI top edge (log10 S2c = 4.15)"""
    r_, g_, b_ = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    grey = (np.abs(r_ - g_) < 6) & (np.abs(g_ - b_) < 6) & (r_ > 110) & (r_ < 175)
    rowsum = grey[yt + 5:yb - 5, xl + 5:xr - 5].sum(axis=1)
    order = np.argsort(-rowsum)
    y1 = order[0] + yt + 5
    y2 = next(o for o in order if abs(o + yt + 5 - y1) > 50) + yt + 5
    return min(y1, y2), max(y1, y2)


def major_yticks(rgb, xr, yt, yb):
    """rows of the major y-axis ticks: 22-px dark runs entering the frame from the RIGHT edge (the left edge is
    covered by black data points in Figs. 4/S1a; minor ticks are 11 px)"""
    dark = rgb.sum(axis=2) < 200
    runs = np.zeros(rgb.shape[0], dtype=int)
    for y in range(yt + 1, yb):
        x = xr - 1
        while x > xr - 40 and dark[y, x]:
            x -= 1
        runs[y] = (xr - 1) - x
    rows = np.where(runs >= 18)[0]
    rows = rows[(rows > yt + 3) & (rows < yb - 3)]
    groups, cur = [], [rows[0]]
    for yy in rows[1:]:
        if yy - cur[-1] <= 2:
            cur.append(yy)
        else:
            groups.append(cur); cur = [yy]
    groups.append(cur)
    return np.array([np.mean(g) for g in groups])


def calib(name):
    """axis calibration: x from the frame (0-800 phd at the frame edges, verified by P009 with the ROI edge at
    600 phd); y from the major ticks (half-dex spacing) plus the frame top at 5.0 -- robust against the ER cloud"""
    rgb = load_png(name)
    xl, xr, yt, yb = frame(rgb)
    ticks = major_yticks(rgb, xr, yt, yb)
    guess = (yb - yt) / 2.25                        # px per dex if the frame spans 2.75-5.0
    vals = 5.0 - np.round((ticks - yt) / guess * 2) / 2   # nearest half-dex value for each tick
    ticks = np.append(ticks, yt); vals = np.append(vals, 5.0)
    slope, inter = np.polyfit(ticks, vals, 1)       # dex per pixel (negative), intercept
    resid = vals - (slope * ticks + inter)
    x_of = lambda s1: xl + s1 / 800.0 * (xr - xl)
    s1_of = lambda x: (x - xl) / (xr - xl) * 800.0
    l_of = lambda y: slope * y + inter
    y_of = lambda l: (l - inter) / slope
    return rgb, dict(xl=int(xl), xr=int(xr), yt=int(yt), yb=int(yb), tick_rows=ticks.tolist(),
                     tick_fit_max_resid_dex=float(np.abs(resid).max()), px_per_dex=float(-1 / slope),
                     frame_top_check=float(l_of(yt)), frame_bottom_check=float(l_of(yb))), x_of, s1_of, l_of, y_of


def red_lines(rgb, x_of, l_of, s1_vals, half=6):
    """centroids (log10 S2c) of the three red NR lines (p10, median, p90) in a +-half px column around each S1c"""
    red = (rgb[:, :, 0] > 170) & (rgb[:, :, 1] < 90) & (rgb[:, :, 2] < 120)
    out = {}
    for s1 in s1_vals:
        x0 = int(round(x_of(s1)))
        ys = []
        for x in range(x0 - half, x0 + half + 1):
            rows = np.where(red[:, x])[0]
            ys.extend(rows.tolist())
        ys = np.sort(np.array(ys))
        if ys.size == 0:
            continue
        groups, cur = [], [ys[0]]
        for yy in ys[1:]:
            if yy - cur[-1] <= 2:
                cur.append(yy)
            else:
                groups.append(cur); cur = [yy]
        groups.append(cur)
        groups = sorted(groups, key=np.mean)
        vals = sorted([float(l_of(np.mean(g))) for g in groups if len(g) >= 3])
        out[s1] = vals
    return out


DIG = {}
for name in ["Fig4_science_sample_wbands", "Fig2_Calibrations"]:
    rgb, cal, x_of, s1_of, l_of, y_of = calib(name)
    lines = red_lines(rgb, x_of, l_of, [480, 500, 520, 540.1, 560, 580, 600])
    rows = []
    for s1, vals in lines.items():
        if len(vals) == 3:
            rows.append(dict(S1c=s1, p10=vals[0], p50=vals[1], p90=vals[2], halfwidth=0.5 * (vals[2] - vals[0]),
                             lower_half=vals[1] - vals[0], upper_half=vals[2] - vals[1]))
    df = pd.DataFrame(rows)
    DIG[name] = dict(calibration=cal, lines=df.to_dict(orient="records"),
                     mean_halfwidth=float(df.halfwidth.mean()), sd_halfwidth=float(df.halfwidth.std()),
                     mean_lower_half=float(df.lower_half.mean()), mean_upper_half=float(df.upper_half.mean()),
                     px_per_0p01dex=cal["px_per_dex"] / 100)
    df.to_csv(os.path.join(OUT, f"digitised_NR_lines_{name}.csv"), index=False)
    print(name, cal, "\n", df.round(4).to_string())
hw_drawn = 0.5 * (DIG["Fig4_science_sample_wbands"]["mean_halfwidth"] + DIG["Fig2_Calibrations"]["mean_halfwidth"])
sig_drawn = hw_drawn / 1.2816
med_drawn_540 = [r_["p50"] for r_ in DIG["Fig4_science_sample_wbands"]["lines"] if abs(r_["S1c"] - 540.1) < 0.5][0]
p10_drawn_540 = [r_["p10"] for r_ in DIG["Fig4_science_sample_wbands"]["lines"] if abs(r_["S1c"] - 540.1) < 0.5][0]
DRAWN = dict(halfwidth_1090_dex=hw_drawn, sigma_eq_dex=sig_drawn, median_at_540=med_drawn_540, p10_at_540=p10_drawn_540,
             event_offset=med_drawn_540 - L2, nsig_gauss=(med_drawn_540 - L2) / sig_drawn,
             n_halfwidths=(med_drawn_540 - L2) / hw_drawn,
             sigma_implied_by_paper_1p5=(med_drawn_540 - L2) / 1.5,
             P_below_gauss=float(stats.norm.cdf(-(med_drawn_540 - L2) / sig_drawn)),
             P_below_if_sigma_paper=float(stats.norm.cdf(-1.5)))
# what electron-fluctuation scale k_e reproduces the drawn sigma?  variance budget: detector terms fixed
det_terms = sum(DECOMP["terms_dex"][k] ** 2 for k in list(DECOMP["terms_dex"])[3:])
quanta_needed = sig_drawn ** 2 - det_terms
DRAWN["detector_terms_dex"] = math.sqrt(det_terms)
DRAWN["quanta_sigma_needed_dex"] = math.sqrt(quanta_needed) if quanta_needed > 0 else float("nan")
DRAWN["k_e_needed_full_detector"] = (math.sqrt(quanta_needed) / DECOMP["quanta_only_dex_analytic"]) if quanta_needed > 0 else 0.0
DRAWN["k_e_needed_perfect_detector"] = sig_drawn / DECOMP["quanta_only_dex_analytic"]
DRAWN["binomial_floor_dex"] = DECOMP["terms_dex"]["recombination binomial r(1-r)Ni"]
DRAWN["poisson_on_Ne_model_dex"] = DEX / math.sqrt(10 ** med_drawn_540 / G2)
DRAWN["poisson_on_Ne_model_halfwidth"] = 1.2816 * DRAWN["poisson_on_Ne_model_dex"]
print("DRAWN", json.dumps(DRAWN, indent=1, default=float))

# ----------------------------------------------------------------------------------------------------
# 4. AmBe statistics near 250 keV
# ----------------------------------------------------------------------------------------------------
rgb2, cal2, x_of2, s1_of2, l_of2, y_of2 = calib("Fig2_Calibrations")
purple = (np.abs(rgb2[:, :, 0] - 144) < 40) & (rgb2[:, :, 1] < 80) & (np.abs(rgb2[:, :, 2] - 208) < 45)
lab, nlab = ndimage.label(purple)
objs = ndimage.find_objects(lab)
cent = ndimage.center_of_mass(purple, lab, range(1, nlab + 1))
sizes = ndimage.sum(purple, lab, range(1, nlab + 1))
cent = np.array(cent); sizes = np.array(sizes)
s1_pts = s1_of2(cent[:, 1]); l2_pts = l_of2(cent[:, 0])
single_size = np.median(sizes[(s1_pts > 450)])          # typical single-marker blob size from the sparse region
n_est = np.maximum(1, np.round(sizes / single_size))    # blobs larger than one marker: count as several
bins = np.arange(0, 820, 20)
h_blob, _ = np.histogram(s1_pts, bins)
h_est, _ = np.histogram(s1_pts, bins, weights=n_est)
AMBE = dict(n_blobs_total=int(nlab), n_est_total_lower_bound=int(n_est.sum()), single_marker_px=float(single_size),
            per_20phd=[dict(S1c_lo=float(bins[i]), S1c_hi=float(bins[i + 1]), blobs=int(h_blob[i]), est=int(h_est[i]))
                       for i in range(bins.size - 1)],
            n_500_580=int(n_est[(s1_pts >= 500) & (s1_pts < 580)].sum()),
            n_520_560=int(n_est[(s1_pts >= 520) & (s1_pts < 560)].sum()),
            n_gt_500=int(n_est[s1_pts >= 500].sum()), n_gt_450=int(n_est[s1_pts >= 450].sum()),
            n_450_500=int(n_est[(s1_pts >= 450) & (s1_pts < 500)].sum()),
            n_gt_580=int(n_est[s1_pts >= 580].sum()), S1c_max=float(s1_pts.max()),
            in_ROI_gt_500=int(n_est[(s1_pts >= 500) & (l2_pts < 4.15)].sum()))
# energy equivalents of the S1c bins on the paper scale
Egrid = np.arange(100, 400, 1.0)
S1grid = np.array([G1 * (yield_result(E)[0].PhotonYield + yield_result(E)[1]) for E in Egrid])
E_of_S1 = lambda s: float(np.interp(s, S1grid, Egrid))
AMBE["E_of_S1c"] = {str(s): E_of_S1(s) for s in (450, 500, 520, 540, 560, 580, 600)}
# purple points as a fraction of the AmBe NR population: the dense region undercounts, so we normalise the
# spectrum model to the count in a sparse-but-populated bin (S1c 400-500) instead of the total
pd.DataFrame(dict(S1c=s1_pts, log10S2c=l2_pts, blob_px=sizes, n_est=n_est)).to_csv(os.path.join(OUT, "ambe_points_fig2.csv"), index=False)

# recoil-spectrum model.  AmBe neutron spectrum (ISO 8529-1 shape, recalled/uncertain) as a 1-MeV histogram;
# elastic n-Xe: E_R,max = 4 m_n M/(m_n+M)^2 E_n = 0.0301 E_n (certain); sigma_el(E_n) ~ 4 b flat 2-11 MeV (likely);
# angular distribution isotropic (upper) or black-disk |2 J1(qR)/qR|^2 with R = 6.35 fm (lower; P013)
AMBE_SPEC = np.array([[0.5, 0.06], [1.5, 0.10], [2.5, 0.15], [3.5, 0.18], [4.5, 0.15], [5.5, 0.11], [6.5, 0.08],
                      [7.5, 0.07], [8.5, 0.05], [9.5, 0.035], [10.5, 0.015]])
KIN = 4 * 1.008665 * 131.29 / (1.008665 + 131.29) ** 2           # E_R,max / E_n = 0.0301
M_XE_MEV = 131.29 * 931.494; HBARC = 197.327; R_BD = 6.35


def recoil_pdf(E_R_keV, iso=True):
    """dN/dE_R (per keV) for the AmBe spectrum, single elastic scatter, normalised per neutron scatter."""
    E_R = np.atleast_1d(E_R_keV).astype(float)
    out = np.zeros_like(E_R)
    for En, f in AMBE_SPEC:
        Emax = KIN * En * 1e3
        m = E_R < Emax
        if iso:
            out[m] += f / Emax
        else:
            q = np.sqrt(2 * M_XE_MEV * E_R[m] * 1e-3) / HBARC        # fm^-1
            x = q * R_BD
            ang = (2 * special.j1(x) / x) ** 2
            # normalisation over 0..Emax
            Eg = np.linspace(1e-3, Emax, 4000); qg = np.sqrt(2 * M_XE_MEV * Eg * 1e-3) / HBARC; xg = qg * R_BD
            norm = np.trapezoid((2 * special.j1(xg) / xg) ** 2, Eg)
            out[m] += f * ang / norm
    return out


Eg = np.linspace(0.5, 335, 3000)
SPEC = {}
for iso, tag in ((True, "isotropic"), (False, "black_disk")):
    pdf = recoil_pdf(Eg, iso); pdf /= np.trapezoid(pdf, Eg)
    cdf = np.concatenate([[0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(Eg))])
    F = lambda a, b: float(np.interp(b, Eg, cdf) - np.interp(a, Eg, cdf))
    SPEC[tag] = dict(f_230_270=F(230, 270), f_gt_200=F(200, 335), f_gt_100=F(100, 335), f_50_150=F(50, 150),
                     f_E450_E500=F(E_of_S1(450), E_of_S1(500)), f_E500_E580=F(E_of_S1(500), E_of_S1(580)),
                     f_20_74=F(20, 74), mean_keV=float(np.trapezoid(Eg * pdf, Eg)))
    # implied total sample size from the Fig. 2 count in S1c 450-500 (E 210-232 keV) and 500-580
    SPEC[tag]["N_total_from_450_500"] = AMBE["n_450_500"] / SPEC[tag]["f_E450_E500"]
    SPEC[tag]["N_total_from_500_580"] = AMBE["n_500_580"] / SPEC[tag]["f_E500_E580"]
    SPEC[tag]["N_230_270_per_1000_events"] = 1000 * SPEC[tag]["f_230_270"]
    SPEC[tag]["N_230_270_per_10000_events"] = 10000 * SPEC[tag]["f_230_270"]
# multiple-scatter/geometry effects are not modelled: state.  Precision on the band median and width:
N_obs = AMBE["n_500_580"]
for nm, sig in (("MC", VAR["baseline (paper scale, LZ widths)"]["sd"]), ("drawn", sig_drawn)):
    AMBE[f"sigma_median_{nm}_N{N_obs}"] = sig / math.sqrt(N_obs)
    AMBE[f"sigma_width_{nm}_N{N_obs}"] = sig / math.sqrt(2 * N_obs)
    AMBE[f"sigma_median_{nm}_frac_of_offset_N{N_obs}"] = sig / math.sqrt(N_obs) / (med_drawn_540 - L2)
# band width measured from LZ's own AmBe points (Fig. 2): residuals of single-marker purple blobs about the
# digitised NR median line (solid red), S1c 450-600
red2 = (rgb2[:, :, 0] > 170) & (rgb2[:, :, 1] < 90) & (rgb2[:, :, 2] < 120)
med_x, med_l = [], []
for x in range(int(x_of2(280)), int(x_of2(640))):
    rows = np.where(red2[:, x])[0]
    if rows.size == 0:
        continue
    groups, cur = [], [rows[0]]
    for yy in rows[1:]:
        if yy - cur[-1] <= 2:
            cur.append(yy)
        else:
            groups.append(cur); cur = [yy]
    groups.append(cur)
    if len(groups) == 3:                       # p90 / median / p10 all present: middle cluster is the median
        med_x.append(s1_of2(x)); med_l.append(l_of2(np.mean(groups[1])))
    elif len(groups) == 1 and len(groups[0]) >= 4:   # only the solid line present (dash gaps)
        med_x.append(s1_of2(x)); med_l.append(l_of2(np.mean(groups[0])))
med_x, med_l = np.array(med_x), np.array(med_l)
single = n_est == 1
RESID = {}
for lo, hi, tag in ((450, 600, "450_600"), (500, 580, "500_580"), (300, 450, "300_450")):
    m = single & (s1_pts >= lo) & (s1_pts < hi) & (l2_pts > 3.6)
    res = l2_pts[m] - np.interp(s1_pts[m], med_x, med_l)
    n = int(m.sum())
    sd = float(res.std(ddof=1)); mad = float(1.4826 * np.median(np.abs(res - np.median(res))))
    lnL = lambda s: float(-0.5 * np.sum((res / s) ** 2) - n * math.log(s))
    RESID[tag] = dict(n=n, mean=float(res.mean()), sd=sd, sd_err=sd / math.sqrt(2 * n), robust_sd=mad,
                      p10=float(np.percentile(res, 10)), p90=float(np.percentile(res, 90)),
                      halfwidth_1090=float(0.5 * (np.percentile(res, 90) - np.percentile(res, 10))),
                      lnL_MC_over_drawn=lnL(base["sd"]) - lnL(sig_drawn),
                      n_below_1p5sig_MC=int((res < -1.5 * base["sd"]).sum()), n_below_2p3sig_drawn=int((res < -2.33 * sig_drawn).sum()),
                      residuals=res.tolist())
    print("AmBe residuals", tag, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in RESID[tag].items() if k != "residuals"})
AMBE["residuals_about_median"] = RESID
AMBE["median_curve_points"] = int(med_x.size)
AMBE["N_for_10pct_width"] = 50            # sigma_w/w = 1/sqrt(2N) = 0.1 -> N = 50
AMBE["N_for_5pct_width"] = 200
AMBE["spectrum_models"] = SPEC
print("AMBE", json.dumps({k: v for k, v in AMBE.items() if k != "per_20phd"}, indent=1, default=float))
print(pd.DataFrame(AMBE["per_20phd"]).query("S1c_lo >= 300").to_string())

# ----------------------------------------------------------------------------------------------------
# 5. MSSI contours in Fig. S1a at S1c = 540
# ----------------------------------------------------------------------------------------------------
rgbS, calS, x_ofS, s1_ofS, l_ofS, y_ofS = calib("FigS1a_science_sample_marked")
blue = (rgbS[:, :, 0] < 40) & (np.abs(rgbS[:, :, 1] - 80) < 30) & (np.abs(rgbS[:, :, 2] - 128) < 30)
brown = (np.abs(rgbS[:, :, 0] - 160) < 25) & (np.abs(rgbS[:, :, 1] - 40) < 25) & (np.abs(rgbS[:, :, 2] - 40) < 25)


def contour_rows(mask, s1, half=12):
    x0 = int(round(x_ofS(s1)))
    ys = np.sort(np.concatenate([np.where(mask[:, x])[0] for x in range(x0 - half, x0 + half + 1)]))
    if ys.size == 0:
        return []
    groups, cur = [], [ys[0]]
    for yy in ys[1:]:
        if yy - cur[-1] <= 3:
            cur.append(yy)
        else:
            groups.append(cur); cur = [yy]
    groups.append(cur)
    return sorted([dict(log10S2c=float(l_ofS(np.mean(g))), n_px=len(g)) for g in groups], key=lambda d: d["log10S2c"])


MSSI = {}
for s1 in (500, 520, 540.1, 560, 580):
    MSSI[str(s1)] = dict(mssi_blue=contour_rows(blue, s1), L10_brown=contour_rows(brown, s1))
    print(s1, MSSI[str(s1)])
# interpretation at 540: lowest blue cluster = 95 % lower edge (dashed); next = 68 % lower edge (solid);
# upper edges lie above the ROI top (4.15) where the contours are clipped
bl = [d["log10S2c"] for d in MSSI["540.1"]["mssi_blue"] if d["log10S2c"] < 4.14]
mssi95_lo, mssi68_lo = bl[0], bl[1] if len(bl) > 1 else float("nan")
ROI_TOP = 4.15
base = VAR["baseline (paper scale, LZ widths)"]
l2_base = SAMPLES["baseline (paper scale, LZ widths)"][1]
OVER = dict(mssi95_lower_edge=mssi95_lo, mssi68_lower_edge=mssi68_lo, roi_top=ROI_TOP,
            event_minus_68edge_dex=L2 - mssi68_lo, event_minus_95edge_dex=L2 - mssi95_lo,
            frac_NR_MC_in_68=float(((l2_base >= mssi68_lo) & (l2_base <= ROI_TOP)).mean()),
            frac_NR_MC_in_95=float(((l2_base >= mssi95_lo) & (l2_base <= ROI_TOP)).mean()),
            frac_NR_MC_above_ROI=float((l2_base > ROI_TOP).mean()),
            frac_NR_gauss033_in_68=float(stats.norm.cdf((ROI_TOP - base["p50"]) / base["sd"]) - stats.norm.cdf((mssi68_lo - base["p50"]) / base["sd"])),
            frac_NR_gauss033_in_95=float(stats.norm.cdf((ROI_TOP - base["p50"]) / base["sd"]) - stats.norm.cdf((mssi95_lo - base["p50"]) / base["sd"])),
            frac_NR_drawn_in_68=float(stats.norm.cdf((ROI_TOP - med_drawn_540) / sig_drawn) - stats.norm.cdf((mssi68_lo - med_drawn_540) / sig_drawn)),
            frac_NR_drawn_in_95=float(stats.norm.cdf((ROI_TOP - med_drawn_540) / sig_drawn) - stats.norm.cdf((mssi95_lo - med_drawn_540) / sig_drawn)),
            frac_NR_MC_below_event=base["P_below"],
            frac_NR_MC_between_68edge_and_event=float(((l2_base >= mssi68_lo) & (l2_base <= L2)).mean()))
print("OVER", json.dumps(OVER, indent=1))

# ----------------------------------------------------------------------------------------------------
# results
# ----------------------------------------------------------------------------------------------------
RES = dict(inputs=dict(g1=G1, g2=G2, S1c=S1, S2c=S2, log10S2c=L2, field=FIELD, width_parameters=W_LZ, ext_eff=EXT_EFF,
                       pos_res=POS_RES, s2Fano=S2_FANO, sPEres=SPE_RES, P_dphe=P_DPHE, alpha_paper=ALPHA_P, beta_paper=BETA_P,
                       E_range=[E_LO, E_HI], S1_half=S1_HALF, N_band=N_BAND, N_small=N_SMALL, nest_version=nestpy.__nest_version__),
           variants=VAR, fixed_energy=FIXE, decomposition=DECOMP, digitised=DIG, drawn=DRAWN, ambe=AMBE, mssi_contours=MSSI,
           overlap=OVER, runtime_s=time.time() - t0)
with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(RES, f, indent=1, default=float)

# ----------------------------------------------------------------------------------------------------
# figures (dataviz palette: series blue #2a78d6, orange #eb6834, aqua #1baf7a, yellow #eda100, magenta #e87ba4; ink #52514e)
# ----------------------------------------------------------------------------------------------------
C1, C2, C3, C4, C5, INK, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#52514e", "#e1e0d9"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
order = [k for k in VAR if not k.startswith("width[")] + [k for k in VAR if k.startswith("width[") and
                                                          abs(VAR[k]["sd"] - base["sd"]) > 0.0008]
fig, axs = plt.subplots(1, 2, figsize=(11, 0.28 * len(order) + 1.6), sharey=True)
ypos = np.arange(len(order))[::-1]
sd = [VAR[k]["sd"] for k in order]; ns = [VAR[k]["nsig"] for k in order]
axs[0].barh(ypos, sd, color=C1, height=0.6)
axs[0].axvline(base["sd"], color=INK, lw=1, ls="-", label=f"MC baseline {base['sd']:.4f} dex")
axs[0].axvline(sig_drawn, color=C2, lw=1.5, ls="--", label=f"drawn 10-90 % lines: sigma_eq {sig_drawn:.4f}")
axs[0].axvline(DRAWN["sigma_implied_by_paper_1p5"], color=C3, lw=1.5, ls=":", label=f"offset/1.5 = {DRAWN['sigma_implied_by_paper_1p5']:.4f} (paper's 1.5 sigma)")
axs[0].axvline(DRAWN["binomial_floor_dex"], color=C4, lw=1.5, ls="-.", label=f"recombination-binomial floor {DRAWN['binomial_floor_dex']:.4f}")
axs[0].set_yticks(ypos); axs[0].set_yticklabels(order, fontsize=7.5)
axs[0].set_xlabel("sigma(log10 S2c) at S1c = 540 +- 10 phd [dex]"); axs[0].set_xlim(0, 0.05)
axs[0].legend(fontsize=7, loc="upper right"); axs[0].grid(axis="x", color=GRID, lw=0.6); axs[0].set_axisbelow(True)
axs[1].barh(ypos, ns, color=C5, height=0.6)
axs[1].axvline(1.5, color=INK, lw=1, label="paper: 1.5 sigma")
axs[1].axvline(base["nsig"], color=C1, lw=1, ls="--", label=f"MC baseline {base['nsig']:.2f} sigma")
axs[1].axvline(DRAWN["nsig_gauss"], color=C2, lw=1.5, ls="--", label=f"drawn lines {DRAWN['nsig_gauss']:.2f} sigma")
axs[1].set_xlabel("event distance below the band median [sigma = MC sd]"); axs[1].set_xlim(0, 5)
axs[1].legend(fontsize=7, loc="upper right"); axs[1].grid(axis="x", color=GRID, lw=0.6); axs[1].set_axisbelow(True)
fig.suptitle("NR band width at S1c = 540 phd: NEST/detector variants (left) and the event's sigma-distance (right)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "P024_band_width_variants.png"), dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.8))
xx = np.linspace(3.85, 4.15, 600)
ax.hist(l2_base, bins=np.linspace(3.85, 4.15, 76), density=True, color=C1, alpha=0.35, label=f"NEST MC, flat spectrum, S1c = 540 +- 10 (sd {base['sd']:.3f} dex, skew {base['skew']:+.2f})")
ax.plot(xx, stats.norm.pdf(xx, med_drawn_540, sig_drawn), color=C2, lw=2, ls="--", label=f"Gaussian from drawn 10-90 % lines (median {med_drawn_540:.3f}, sigma {sig_drawn:.4f})")
ax.plot(xx, stats.norm.pdf(xx, med_drawn_540, DRAWN["sigma_implied_by_paper_1p5"]), color=C3, lw=2, ls=":", label=f"Gaussian with sigma = offset/1.5 = {DRAWN['sigma_implied_by_paper_1p5']:.4f} (paper's statement)")
ax.axvspan(mssi68_lo, ROI_TOP, color=C4, alpha=0.18, label=f"MSSI 68 % region at 540 (Fig. S1a): >= {mssi68_lo:.3f}")
ax.axvspan(mssi95_lo, mssi68_lo, color=C4, alpha=0.08, label=f"MSSI 95 % region: >= {mssi95_lo:.3f}")
ax.axvline(L2, color="k", lw=1.5, label="event log10 S2c = 3.967")
ax.axvline(ROI_TOP, color=INK, lw=0.8, ls="-.", label="ROI top 4.15")
ax.set_xlabel("log10(S2c [phd]) at S1c = 540 phd"); ax.set_ylabel("probability density [1/dex]")
ax.set_xlim(3.85, 4.15); ax.set_ylim(0, 26); ax.legend(fontsize=7, loc="upper right"); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_title("NR PDF at the event's S1c: MC vs drawn band vs paper's sigma, with the MSSI containment regions", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "P024_pdf_at_540.png"), dpi=150); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
terms = DECOMP["terms_dex"]
axs[0].barh(list(terms.keys())[::-1], [terms[k] * 1e3 for k in list(terms)[::-1]], color=C1, height=0.6)
axs[0].axvline(DECOMP["total_dex_analytic"] * 1e3, color=INK, lw=1, label=f"quadrature total {DECOMP['total_dex_analytic']:.4f} dex")
axs[0].axvline(base["sd"] * 1e3, color=C2, lw=1, ls="--", label=f"MC sd {base['sd']:.4f} dex")
axs[0].set_xlabel("contribution to sigma(log10 S2c) [milli-dex]"); axs[0].legend(fontsize=7); axs[0].grid(axis="x", color=GRID, lw=0.6); axs[0].set_axisbelow(True)
axs[0].set_title("Variance decomposition at 248 keV (Ne = %.0f, Ni = %.0f, r = %.3f)" % (Ne_m, Ni_m, r), fontsize=9)
dfp = pd.DataFrame(AMBE["per_20phd"]); dfp = dfp[dfp.S1c_lo >= 200]
axs[1].bar(dfp.S1c_lo + 10, dfp.est, width=18, color=C5, label="AmBe points in Fig. 2 (blob count, overlaps corrected)")
axs[1].axvspan(500, 580, color=C4, alpha=0.2, label="S1c 500-580 (~230-270 keV): N = %d" % AMBE["n_500_580"])
axs[1].axvline(S1, color="k", lw=1.2, label="event S1c")
axs[1].set_xlabel("S1c [phd]"); axs[1].set_ylabel("AmBe calibration points per 20 phd"); axs[1].legend(fontsize=7)
axs[1].grid(axis="y", color=GRID, lw=0.6); axs[1].set_axisbelow(True); axs[1].set_title("AmBe calibration statistics at high S1c (Fig. 2)", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "P024_decomposition_and_ambe.png"), dpi=150); plt.close(fig)
fig, ax = plt.subplots(figsize=(7.5, 4.2))
res = np.array(RESID["450_600"]["residuals"])
ax.hist(res, bins=np.arange(-0.1, 0.101, 0.01), color=C5, alpha=0.6, label=f"AmBe points, S1c 450-600 (N = {res.size}; sd {res.std(ddof=1):.4f} dex)")
xx = np.linspace(-0.1, 0.1, 400)
ax.plot(xx, res.size * 0.01 * stats.norm.pdf(xx, 0, base["sd"]), color=C1, lw=2, label=f"NEST MC width {base['sd']:.4f} dex")
ax.plot(xx, res.size * 0.01 * stats.norm.pdf(xx, 0, sig_drawn), color=C2, lw=2, ls="--", label=f"drawn-line width {sig_drawn:.4f} dex")
ax.axvline(L2 - med_drawn_540, color="k", lw=1.5, label="event offset from the drawn median")
ax.set_xlabel("log10 S2c - NR median (digitised) [dex]"); ax.set_ylabel("AmBe calibration points per 0.01 dex")
ax.legend(fontsize=7); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_title("Band width measured from LZ's AmBe points at S1c = 450-600 phd (Fig. 2)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "P024_ambe_residuals.png"), dpi=150); plt.close(fig)
print("done in %.0f s" % (time.time() - t0))
