#!/usr/bin/env python
"""P088 -- Decisive-test timeline for the LZ 248 keV event.

Synthesis / projection paper. All physics inputs (signal rates per t.yr in each window, acceptances,
backgrounds, shape/modulation/target event requirements, the P(DM) posterior) are taken from the corpus
(P016, P020, P034, P038, P046, P047, P050, P057, P061, P069). Only the counting statistics are recomputed
here: Asimov significances, exact-Poisson first-crossing toys, Gamma-Poisson Bayes factors and the
posterior update of P(DM) on continued nulls.

Run from the simulation root:  .venv/bin/python output/code/P088_timeline.py
Writes: output/work/P088/*.csv, *.json, details tables; output/work/P088/figures/*.png
"""
import json, os, sys, time
import numpy as np
import pandas as pd
from scipy import stats, special
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, "output/code")
from common import lzcommon as lz  # noqa: E402  (LZ constants only)

T0 = time.time()
OUT = "output/work/P088"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(88)

# ----------------------------------------------------------------------------------------------
# 1. CORPUS INPUTS (every number carries its source)
# ----------------------------------------------------------------------------------------------
NOW = 2026 + (16 + 31 + 31 + 30 + 31 + 30 + 31 + 28 + 31 + 30 + 31) / 365.0  # 2026-09-16 -> 2026.71
NOW = 2026.71
MODELS = ["L10", "d300", "d350", "d366", "d380"]
LABEL = {"L10": "B: L10 (elastic q^4 spin)", "d300": "A: inel. 300 keV", "d350": "A: inel. 350 keV",
         "d366": "A: inel. 366 keV", "d380": "A: inel. 380 keV"}
SHORT = {"L10": "L10", "d300": "δ=300", "d350": "δ=350", "d366": "δ=366", "d380": "δ=380"}

# Signal rates per t.yr at the LZ best fit (1.0 event per 2.84 t.yr in LZ's ROI) -- P069 normalisation
# table (reproduces P050 to <=0.3 %). Windows: LZ-like 600 phd edge (E50 = 270 keV) and a 1000 phd-like
# edge (P050's E50 = 400 keV proxy; P038's 1000 phd is E50 = 423 keV).
RATE = {
    # per t.yr, LZ best fit
    "roi270":  {"L10": 0.352113, "d300": 0.352113, "d350": 0.352113, "d366": 0.352113, "d380": 0.352113},
    "c200_270": {"L10": 0.120513, "d300": 0.062824, "d350": 0.178018, "d366": 0.273672, "d380": 0.278338},
    "roi400":  {"L10": 0.446845, "d300": 0.401243, "d350": 0.687498, "d366": 1.702909, "d380": 10.496775},
    "c200_400": {"L10": 0.219641, "d300": 0.112330, "d350": 0.515900, "d366": 1.634376, "d380": 10.493725},
}
# Backgrounds per t.yr (NR band). P016 b_H = 5.7e-4 per 2.84 t.yr in 200-270 keV (baseline);
# P001/P061 brackets 2e-4 and the whole S1c>500 phd panel 0.0106 (LZ Table); P050 ext400 >200 keV
# 2.74e-3 per 2.84 t.yr (LZ-like), 1.6e-3 (60 t); P061 L_acc = 9.3e-3 per 2.84 t.yr summed residual
# unmodelled channels (artefacts 0.006, ER 2.8e-3, ...) -- LZ-specific, treated as a steady LZ-only rate.
B = {
    "c200_270": {"model": 5.7e-4 / 2.84, "lo": 2.0e-4 / 2.84, "panel": 0.0106 / 2.84},
    "c200_400": {"model": 2.74e-3 / 2.84, "lo": 2.74e-3 / 2.84 * (2.0 / 5.7), "panel": 0.0106 / 2.84 + 2.17e-3 / 2.84},
    "c200_400_60t": {"model": 1.6e-3 / 2.84},
    "L_acc_LZ_only": 9.3e-3 / 2.84,   # P061 (upper bracket 0.066 / 2.84)
}
# P038: events in the 600-1000 phd sideband per LZ-ROI event at the best fit (annual halo), b = 0.0027/2.84 t.yr
P038_EXTRA = {"d300": 0.155, "d350": 1.062, "d366": 3.81, "d380": 30.6, "L10": (0.219641 - 0.120513) / 0.352113}
P038_B_EXT = 0.0027  # NR-band MSSI added by 600->1000 phd per 2.84 t.yr
# P057 (toys) / P050: events separating L10 from inelastic-366 at 3 sigma (whole-window events)
N_SHAPE = {"roi270": {"L10_true": 2, "inel_true": 7}, "roi400": {"L10_true": 2, "inel_true": 5},
           "P050_400keV": {"L10_true": 4, "inel_true": 6}}
# P034: modulation events vs a time-flat population of equal rate (LZ-like window, 1 TeV, best fit)
N_MOD = {"d300": (96.5, 268.0), "d350": (11.5, 32.0), "d366": (6.5, 17.6), "d380": (5.3, 14.1)}
MOD_GAIN_400 = {"d300": 1.0, "d350": 1.0, "d366": 2.8, "d380": 21.0}  # P034: exposure gain with a 400 keV ROI
# P046: CaWO4 counts per t.yr(compound) in the 206Pb-safe W band 110-1300 keV at LZ's fit; L10 6.3e-5
CAWO4_RATE = {"d300": 2.602, "d350": 36.95, "d366": 279.0, "d380": 5622.0, "L10": 6.3e-5}
CAWO4_N3_VS_L10_kgyr = {"d300": 215, "d350": 17, "d366": 2.1, "d380": 0.09}  # P046
# P047: 136Xe-enriched / natural pair: 9 window events total (12.8 t.yr per detector at LZ's rate) for d366 vs L10
P047_N = 9; P047_E_PER_DET = 12.8
# P061: P(DM | event) community-prior median and 68 % range; P001/P069 free-rate-unknown cap 0.091
PI_DM = {"median": 0.015, "lo68": 0.0016, "hi68": 0.137}
# LZ rate band (Table I 68 %): x0.30 - x2.36 of the best fit (P050/P069 convention)
RATE_LO, RATE_HI = 0.30, 2.36

# Exposure ledger (P069 Table; P020 for LZ; recalled items flagged in details/JSON)
# each segment: (t_start, t_end, exposure t.yr) ; continuing slopes in t.yr per calendar year
LEDGER = {
    "LZ":        {"seg": [(2021.98, 2022.35, 0.77), (2023.23, 2024.25, 2.84), (2024.25, 2026.67, 6.76)], "slope": 4.71 * 0.593, "run_to": 2032.0},
    "XENONnT":   {"seg": [(2021.5, 2023.6, 3.1), (2023.6, 2026.67, 3.0)], "slope": 4.0 * 0.60, "run_to": 2032.0},
    "PandaX-4T": {"seg": [(2020.9, 2022.4, 1.54), (2023.9, 2026.67, 2.5)], "slope": 2.7 * 0.60, "run_to": 2032.0},
    "XLZD":      {"seg": [], "slope": 48.0, "run_from": 2032.0, "run_to": 2040.0},
}
TGRID = np.round(np.arange(2021.0, 2036.0001, 0.01), 2)


def cum_exposure(name, t=TGRID):
    L = LEDGER[name]
    E = np.zeros_like(t)
    for (a, b, x) in L["seg"]:
        E += x * np.clip((t - a) / (b - a), 0, 1)
    if L["seg"]:
        start = L["seg"][-1][1]
    else:
        start = L["run_from"]
    E += L["slope"] * np.clip(t - start, 0, L["run_to"] - start)
    return E


EXP = {k: cum_exposure(k) for k in LEDGER}
EXP["present"] = EXP["LZ"] + EXP["XENONnT"] + EXP["PandaX-4T"]
EXP["world"] = EXP["present"] + EXP["XLZD"]
EXP_HE = {k: v.copy() for k, v in EXP.items()}
# LZ's SR3 2.84 t.yr is already analysed to 270 keV: the *new* high-energy data excludes it in the
# 200-270 window (event already counted); for the 200-400 window the SR3 sideband 600-1000 phd is new.
i_now = int(np.argmin(np.abs(TGRID - NOW)))


def qinf(x, qs):
    """quantiles of first-crossing dates with 'never' (inf) handled as a very late date."""
    xx = np.where(np.isfinite(x), x, 9999.0)
    q = np.quantile(xx, qs)
    return np.where(q > 2100, np.inf, q)


def ym(t):
    """calendar label; crossings earlier than today mean the needed exposure is already on disk."""
    if not np.isfinite(t):
        return ">2035"
    if t <= NOW:
        return "on disk"
    y = int(np.floor(t)); m = int(np.floor((t - y) * 12)) + 1
    return f"{['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][m-1]} {y}"


# ----------------------------------------------------------------------------------------------
# 2. Counting statistics
# ----------------------------------------------------------------------------------------------
def z_asimov(s, b):
    s = np.asarray(s, float); b = np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = 2 * ((s + b) * np.log1p(s / b) - s)
    return np.sqrt(np.clip(q, 0, None))


def z_count(n, b):
    """exact one-sided Poisson significance of observing >= n given b."""
    p = stats.poisson.sf(n - 1, b)
    return stats.norm.isf(np.clip(p, 1e-300, 1))


P3, P5 = stats.norm.sf(3), stats.norm.sf(5)


def first_cross(t, mu_sig, bkg, p_thr, ntoy, n_prior=0, b_prior=0.0, rate_scale=1.0, gamma_alpha=None):
    """Toy first-crossing calendar dates of a p-value threshold.
    Events arrive as an inhomogeneous Poisson process with cumulative mean mu_sig(t)*scale + bkg(t).
    n_prior/b_prior: events and background already in hand (LZ's one event) added to every toy.
    gamma_alpha: if set, the rate scale is drawn from Gamma(alpha, 1) (Poisson-Gamma, LZ rate posterior).
    Returns array of crossing dates (inf if never before t[-1])."""
    NMAX = 30  # only the first 30 arrivals can matter for a first crossing (b << 1 everywhere here)
    if gamma_alpha is None:
        scale = np.full(ntoy, rate_scale)
    else:
        scale = rng.gamma(gamma_alpha, 1.0, ntoy) * rate_scale
    # arrival positions of a unit-rate Poisson process in cumulative-mean space: cumsum of Exp(1)
    A = np.cumsum(rng.exponential(1.0, (ntoy, NMAX)), axis=1)
    idx = np.arange(1, NMAX + 1)[None, :]
    dates = np.full((ntoy, NMAX), np.inf)
    if gamma_alpha is None:
        cum = scale[0] * mu_sig + bkg
        pos = np.searchsorted(cum, A, side="left")
        valid = pos < len(t)
        dates[valid] = t[np.clip(pos, 0, len(t) - 1)][valid]
    else:
        # group toys into 60 quantile bins of the rate scale (cumulative curve depends on the scale)
        qs = np.quantile(scale, np.linspace(0, 1, 61))
        bins = np.clip(np.searchsorted(qs, scale, side="right") - 1, 0, 59)
        for bi in range(60):
            m = bins == bi
            if not m.any():
                continue
            sc = scale[m].mean()
            cum = sc * mu_sig + bkg
            pos = np.searchsorted(cum, A[m], side="left")
            valid = pos < len(t)
            dd = np.full(pos.shape, np.inf)
            dd[valid] = t[np.clip(pos, 0, len(t) - 1)][valid]
            dates[m] = dd
    valid = np.isfinite(dates)
    b_at = np.interp(np.where(valid, dates, t[-1]), t, bkg) + b_prior
    n_at = idx + n_prior
    ok = valid & (stats.poisson.sf(n_at - 1, b_at) < p_thr)
    dates_ok = np.where(ok, dates, np.inf)
    out = dates_ok.min(axis=1)
    # prior events alone may already cross (at t[0])
    if n_prior > 0:
        pre = stats.poisson.sf(n_prior - 1, b_prior + bkg[0]) < p_thr
        if pre:
            out[:] = 2024.25  # LZ's own event already crosses (3.25 sigma exact Poisson for b = 5.7e-4): end of SR3 data
    return out


def cross_date(t, z, thr):
    i = np.where(z >= thr)[0]
    return t[i[0]] if len(i) else np.inf


# ----------------------------------------------------------------------------------------------
# 3. Timeline: experiment x window x model -> expected events, Asimov Z, toy crossing dates
# ----------------------------------------------------------------------------------------------
WINDOWS = {"600phd": ("c200_270", "roi270"), "1000phd": ("c200_400", "roi400")}
EXPTS = ["LZ", "XENONnT", "PandaX-4T", "present", "XLZD", "world"]
NTOY = 20000
rows = []
curves = {}
t_new = TGRID[i_now:]  # from now
for ename in EXPTS:
    for wname, (cw, roiw) in WINDOWS.items():
        for model in MODELS:
            r = RATE[cw][model]
            # exposure counted as "new high-energy data": everything not yet examined above 200 keV.
            E_all = EXP[ename].copy()
            if ename in ("LZ", "present", "world") and wname == "600phd":
                E_new = E_all - 2.84 * (TGRID >= 2024.25) - 2.84 * np.clip((TGRID - 2023.23) / (2024.25 - 2023.23), 0, 1) * (TGRID < 2024.25)
            else:
                E_new = E_all  # for the 1000 phd window LZ's SR3 sideband is also new
            E_new = np.clip(E_new, 0, None)
            if ename == "XLZD" or (ename == "world"):
                b_rate = B["c200_400_60t"]["model"] if cw == "c200_400" else B["c200_270"]["model"]
            else:
                b_rate = B[cw]["model"]
            if ename == "world":
                # present generation in the chosen window + XLZD always with its 400 keV ROI (P050)
                mu_sig = r * (EXP["present"] - (E_all - E_new)) + RATE["c200_400"][model] * EXP["XLZD"]
                bkg = B[cw]["model"] * (EXP["present"] - (E_all - E_new)) + B["c200_400_60t"]["model"] * EXP["XLZD"]
            else:
                mu_sig = r * E_new
                bkg = b_rate * E_new
            # data on disk today
            mu_now = mu_sig[i_now]
            # Asimov (new-only, best fit) and combined with LZ's one event (s=1, b=5.7e-4)
            zA_new = z_asimov(mu_sig, bkg)
            zA_comb = z_asimov(mu_sig + 1.0, bkg + 5.7e-4)
            zA_new_lo = z_asimov(RATE_LO * mu_sig, bkg)
            d = dict(experiment=ename, window=wname, model=model, rate_per_tyr=r, b_per_tyr=b_rate,
                     E_new_now=E_new[i_now], mu_now=mu_now, P0_now=np.exp(-mu_now),
                     mu_end2027=np.interp(2028.0, TGRID, mu_sig), mu_end2028=np.interp(2029.0, TGRID, mu_sig),
                     mu_2030=np.interp(2031.0, TGRID, mu_sig),
                     asimov3_new=cross_date(TGRID, zA_new, 3), asimov5_new=cross_date(TGRID, zA_new, 5),
                     asimov3_comb=cross_date(TGRID, zA_comb, 3), asimov5_comb=cross_date(TGRID, zA_comb, 5),
                     asimov5_new_lo68=cross_date(TGRID, zA_new_lo, 5))
            # toys (exact Poisson counts), new-only, best fit; from the start of the new data
            for tag, kw in [("best", dict(rate_scale=1.0)), ("lo68", dict(rate_scale=RATE_LO)),
                            ("PG", dict(gamma_alpha=2.0))]:
                for zthr, pthr in [(3, P3), (5, P5)]:
                    fc = first_cross(TGRID, mu_sig, bkg, pthr, NTOY, **kw)
                    q = qinf(fc, [0.16, 0.5, 0.84])
                    d[f"toy{zthr}_{tag}_q16"], d[f"toy{zthr}_{tag}_med"], d[f"toy{zthr}_{tag}_q84"] = q
                    for yr in (2028.0, 2029.0, 2031.0):
                        d[f"P{zthr}_{tag}_by{int(yr)-1}"] = float(np.mean(fc <= yr))
            # combined with LZ's event (n_prior=1, b_prior=5.7e-4)
            for zthr, pthr in [(3, P3), (5, P5)]:
                fc = first_cross(TGRID, mu_sig, bkg, pthr, NTOY, n_prior=1, b_prior=5.7e-4)
                d[f"toy{zthr}_comb_med"] = float(qinf(fc, [0.5])[0])
                d[f"P{zthr}_comb_by2027"] = float(np.mean(fc <= 2028.0))
                d[f"P{zthr}_comb_by2028"] = float(np.mean(fc <= 2029.0))
            rows.append(d)
            curves[(ename, wname, model)] = dict(mu=mu_sig, b=bkg, zA=zA_new, zA_comb=zA_comb)
TL = pd.DataFrame(rows)
TL.to_csv(os.path.join(OUT, "P088_timeline_table.csv"), index=False)
print(f"[{time.time()-T0:5.1f}s] timeline table done ({len(TL)} rows)")

# ----------------------------------------------------------------------------------------------
# 4. LZ's own 600-1000 phd sideband: the test already on disk (P038 + P020 exposure)
# ----------------------------------------------------------------------------------------------
sb = []
for model in MODELS:
    for E, lab in [(2.84, "SR3 (analysed, sideband empty)"), (6.76, "untouched Apr 2024-Sep 2026"), (2.84 + 6.76, "SR3+untouched")]:
        mu = P038_EXTRA[model] * E / 2.84
        b = P038_B_EXT * E / 2.84
        sb.append(dict(model=model, dataset=lab, exposure_tyr=E, mu_sideband=mu, b_sideband=b,
                       P0_if_true=np.exp(-mu), Z_if_zero_excl=np.sqrt(2 * mu) if mu > 0 else 0,  # Asimov exclusion proxy
                       Z_asimov_disc=z_asimov(mu, b)))
SB = pd.DataFrame(sb); SB.to_csv(os.path.join(OUT, "P088_LZ_sideband.csv"), index=False)

# ----------------------------------------------------------------------------------------------
# 5. Bayes factors (DM-with-continuing-rate vs one-off) and P(settled) by date; null branch
# ----------------------------------------------------------------------------------------------
S_GRID = np.linspace(0, 40, 8001)


def bf_dm_vs_oneoff(n, k, b, alpha=2.0):
    """BF = int Gamma(s; alpha, 1) Pois(n | k s + b) ds / Pois(n | b); s in LZ-ROI units (events per 2.84 t.yr)."""
    n = np.atleast_1d(n)
    prior = stats.gamma.pdf(S_GRID, alpha, scale=1.0)
    lam = k * S_GRID[None, :] + b
    like = stats.poisson.pmf(n[:, None], lam)
    num = np.trapezoid(like * prior[None, :], S_GRID, axis=1)
    den = stats.poisson.pmf(n, b)
    return num / np.clip(den, 1e-300, None)


settled = []
for ename in ["LZ", "present", "world"]:
    for wname in WINDOWS:
        for model in MODELS:
            c = curves[(ename, wname, model)]
            for yr, ylab in [(2028.0, "end-2027"), (2029.0, "end-2028"), (2031.0, "end-2030"), (2033.0, "end-2032")]:
                k = float(np.interp(yr, TGRID, c["mu"]))   # expected new events for s = 1
                b = float(np.interp(yr, TGRID, c["b"]))
                nn = np.arange(0, max(400, int(5 * k + 50)))
                bf = bf_dm_vs_oneoff(nn, k, b)
                dec_dm = bf > 100; dec_off = bf < 1 / 100
                for truth, lam in [("A/B best fit", k + b), ("A/B x0.30", RATE_LO * k + b), ("C one-off", b)]:
                    pn = stats.poisson.pmf(nn, lam)
                    settled.append(dict(experiment=ename, window=wname, model=model, date=ylab, k_expected=k, b=b, truth=truth,
                                        P_BF_gt100=float(pn[dec_dm].sum()), P_BF_lt0p01=float(pn[dec_off].sum()),
                                        P_settled=float(pn[dec_dm | dec_off].sum()), BF_if_zero=float(bf[0]),
                                        N_min_for_BF100=int(nn[dec_dm][0]) if dec_dm.any() else -1))
SET = pd.DataFrame(settled); SET.to_csv(os.path.join(OUT, "P088_settled.csv"), index=False)
print(f"[{time.time()-T0:5.1f}s] settled table done")

# null branch: P(DM | 0 new events) from P061's prior, versus date; alpha = 2 (flat) and 1.5 (Jeffreys)
null_rows = []
for ename in ["LZ", "present", "world"]:
    for wname in WINDOWS:
        for model in MODELS:
            c = curves[(ename, wname, model)]
            for alpha in (2.0, 1.5):
                L_dm = (1 + c["mu"]) ** (-alpha) * np.exp(-c["b"]) / np.exp(-c["b"])  # background cancels for n=0
                for plab, pi in PI_DM.items():
                    odds = pi / (1 - pi) * L_dm
                    P = odds / (1 + odds)
                    d = dict(experiment=ename, window=wname, model=model, alpha=alpha, prior=plab, pi=pi,
                             P_now=float(P[i_now]))
                    for thr in (1e-2, 1e-3, 1e-4):
                        d[f"date_P_lt_{thr:g}"] = cross_date(TGRID, -P, -thr)
                    null_rows.append(d)
NULL = pd.DataFrame(null_rows); NULL.to_csv(os.path.join(OUT, "P088_null_branch.csv"), index=False)

# ----------------------------------------------------------------------------------------------
# 6. A vs B: spectral shape (P057/P050), modulation (P034), tungsten (P046), isotopes (P047)
# ----------------------------------------------------------------------------------------------
def nth_event_dates(mu_curve, n, qs=(0.16, 0.5, 0.84)):
    """dates at which the n-th event arrives (Gamma quantiles in cumulative-mean space)."""
    g = stats.gamma.ppf(qs, n, scale=1.0)
    return [cross_date(TGRID, mu_curve, gq) for gq in g]


ab = []
for ename in ["LZ", "present", "world"]:
    for wname, (cw, roiw) in WINDOWS.items():
        # whole-window pooled events (shape and timing use every event in the ROI)
        for truth in ["L10", "d300", "d350", "d366", "d380"]:
            r = RATE[roiw][truth]
            E_all = EXP[ename].copy()
            if wname == "600phd" and ename in ("LZ", "present", "world"):
                E_new = E_all - 2.84 * np.clip((TGRID - 2023.23) / (2024.25 - 2023.23), 0, 1)
            else:
                E_new = E_all
            mu = r * np.clip(E_new, 0, None) + 1.0 * (TGRID >= 2023.46)  # LZ's event itself counts for shape
            # shape (L10 vs inelastic-366 requirement; P057 toys)
            key = "L10_true" if truth == "L10" else "inel_true"
            n_shape = N_SHAPE[roiw][key]
            n_shape_p050 = N_SHAPE["P050_400keV"][key]
            d = dict(experiment=ename, window=wname, truth=truth, roi_rate_per_tyr=r, N_shape_P057=n_shape, N_shape_P050=n_shape_p050)
            d["shape_q16"], d["shape_med"], d["shape_q84"] = nth_event_dates(mu, n_shape)
            d["shape_P050_med"] = nth_event_dates(mu, n_shape_p050)[1]
            # modulation vs time-flat alternative (P034), only for inelastic truths
            if truth in N_MOD:
                n3, n5 = N_MOD[truth]
                gain = MOD_GAIN_400[truth] if wname == "1000phd" else 1.0
                # P034 counts are for the LZ-like window; with the 400 keV ROI P034 quotes exposure gains
                mu_mod = mu if wname == "600phd" else (RATE["roi270"][truth] * gain * np.clip(E_new, 0, None) + 1.0 * (TGRID >= 2023.46))
                d["mod3_q16"], d["mod3_med"], d["mod3_q84"] = nth_event_dates(mu_mod, n3)
                d["mod5_med"] = nth_event_dates(mu_mod, n5)[1]
                d["N_mod3"], d["N_mod5"] = n3, n5
            ab.append(d)
AB = pd.DataFrame(ab); AB.to_csv(os.path.join(OUT, "P088_AvsB_shape_modulation.csv"), index=False)

# CaWO4 scenarios: exposure schedules (recalled/assumed) and counts
CAWO4 = {"10 kg.yr (CRESST-scale, 2027-2030)": (2027.0, 2030.0, 0.010),
         "100 kg.yr (hypothetical array, 2029-2034)": (2029.0, 2034.0, 0.100),
         "1 t.yr (hypothetical, 2030-2035)": (2030.0, 2035.0, 1.0)}
B_W = {"b=0.01/kg.yr": 10.0, "b=0.1/kg.yr": 100.0}  # per t.yr in the W band > 110 keV; recalled placeholder
caw = []
for lab, (a, bb, Etot) in CAWO4.items():
    E = Etot * np.clip((TGRID - a) / (bb - a), 0, 1)
    for model in MODELS:
        mu = CAWO4_RATE[model] * E
        for blab, brate in B_W.items():
            bk = brate * E
            zA = z_asimov(mu, bk)
            fc3 = first_cross(TGRID, mu, bk, P3, 5000); fc5 = first_cross(TGRID, mu, bk, P5, 5000)
            caw.append(dict(scenario=lab, model=model, bkg=blab, exposure_tyr=Etot, mu_total=mu[-1], b_total=bk[-1],
                            asimov3=cross_date(TGRID, zA, 3), asimov5=cross_date(TGRID, zA, 5),
                            toy3_med=float(qinf(fc3, [0.5])[0]), toy5_med=float(qinf(fc5, [0.5])[0]),
                            P3_by_end=float(np.isfinite(fc3).mean()), P5_by_end=float(np.isfinite(fc5).mean()),
                            kgyr_for_3sigma_vs_L10_P046=CAWO4_N3_VS_L10_kgyr.get(model, np.nan)))
CAW = pd.DataFrame(caw); CAW.to_csv(os.path.join(OUT, "P088_cawo4.csv"), index=False)

# 136Xe pair (P047): years needed for a hypothetical enriched TPC of mass M at 60 % live
iso = []
for M in (1.0, 5.0):
    yrs = P047_E_PER_DET / (M * 0.6)
    iso.append(dict(enriched_mass_t=M, live_frac=0.6, tyr_per_yr=M * 0.6, years_to_12p8_tyr=yrs, N_events_total=P047_N))
ISO = pd.DataFrame(iso); ISO.to_csv(os.path.join(OUT, "P088_isotope_pair.csv"), index=False)
print(f"[{time.time()-T0:5.1f}s] A-vs-B, CaWO4, isotopes done")

# ----------------------------------------------------------------------------------------------
# 7. Summary table  experiment x model -> date   (median toy 5 sigma, new data only, best fit; and 3 sigma)
# ----------------------------------------------------------------------------------------------
def fmt(t):
    return ym(t)


summary = []
for ename in EXPTS:
    for wname in WINDOWS:
        d = dict(experiment=ename, window=wname)
        for model in MODELS:
            r = TL[(TL.experiment == ename) & (TL.window == wname) & (TL.model == model)].iloc[0]
            d[f"{SHORT[model]} 3σ"] = fmt(r["toy3_best_med"])
            d[f"{SHORT[model]} 5σ"] = fmt(r["toy5_best_med"])
            d[f"{SHORT[model]} 5σ(×0.30)"] = fmt(r["toy5_lo68_med"])
        summary.append(d)
SUM = pd.DataFrame(summary); SUM.to_csv(os.path.join(OUT, "P088_table_experiment_x_model.csv"), index=False)
with open(os.path.join(OUT, "P088_table_experiment_x_model.md"), "w") as f:
    f.write(SUM.to_markdown(index=False))

# ----------------------------------------------------------------------------------------------
# 8. Figures
# ----------------------------------------------------------------------------------------------
COL = {"L10": "#2a78d6", "d300": "#eb6834", "d350": "#1baf7a", "d366": "#eda100", "d380": "#e87ba4"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# Fig 1: expected new events and median significance vs date, present generation, both windows
fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), sharex=True)
for j, wname in enumerate(WINDOWS):
    ax = axes[0, j]; ax2 = axes[1, j]
    for model in MODELS:
        c = curves[("present", wname, model)]
        m = TGRID >= 2023.0
        ax.plot(TGRID[m], c["mu"][m], color=COL[model], lw=2, label=LABEL[model])
        ax2.plot(TGRID[m], c["zA"][m], color=COL[model], lw=2)
        r = TL[(TL.experiment == "present") & (TL.window == wname) & (TL.model == model)].iloc[0]
        for zthr, mk in [(3, "o"), (5, "s")]:
            x = max(r[f"toy{zthr}_best_med"], NOW)  # crossings before today = exposure already on disk
            if np.isfinite(x) and x < 2036:
                ax2.plot([max(r[f"toy{zthr}_best_q16"], NOW), min(max(r[f"toy{zthr}_best_q84"], NOW), 2035.9)], [zthr, zthr], color=COL[model], lw=1, alpha=0.6)
                ax2.plot(x, zthr, mk, color=COL[model], ms=6, mec="white")
    c = curves[("present", wname, "L10")]
    ax.plot(TGRID[m], c["b"][m], color="0.4", ls=":", lw=1.5, label="background (model)")
    ax.set_yscale("log"); ax.set_ylim(3e-3, 300)
    ax.axvline(NOW, color="0.6", lw=0.8, ls="--"); ax2.axvline(NOW, color="0.6", lw=0.8, ls="--")
    ax.set_title({"600phd": "600 phd edge: 200–270 keV counting", "1000phd": "1000 phd edge: 200–400 keV counting"}[wname])
    ax.set_ylabel("expected new events (LZ+XENONnT+PandaX-4T)" if j == 0 else "")
    ax2.set_ylabel("median significance (σ), new data only" if j == 0 else "")
    ax2.axhline(3, color="0.7", lw=0.8); ax2.axhline(5, color="0.7", lw=0.8)
    ax2.set_ylim(0, 10); ax2.set_xlim(2023, 2032)
    ax2.set_xlabel("calendar year (data on disk)")
    ax2.text(2023.1, 3.15, "3σ", color="0.4"); ax2.text(2023.1, 5.15, "5σ", color="0.4")
axes[0, 0].legend(loc="upper left", fontsize=7.5, frameon=False)
axes[1, 0].text(2029.9, 0.4, "markers: toy median first crossing\nbars: 16–84 %", fontsize=7, color="0.3")
fig.suptitle("P088 Fig. 1 — Present-generation xenon: expected high-energy events and discovery significance vs date (LZ best-fit rate)", fontsize=9.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "P088_fig1_timeline.png"), dpi=160); plt.close(fig)

# Fig 3: P(settled) and null branch
fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
ax = axes[0]
for model in MODELS:
    s = SET[(SET.experiment == "present") & (SET.window == "600phd") & (SET.model == model) & (SET.truth == "A/B best fit")]
    s2 = SET[(SET.experiment == "present") & (SET.window == "600phd") & (SET.model == model) & (SET.truth == "A/B x0.30")]
    xs = [2027.99, 2028.99, 2030.99, 2032.99]
    ax.plot(xs, s.P_settled.values, "-o", color=COL[model], lw=2, ms=5, label=SHORT[model] + " true (best fit)")
    ax.plot(xs, s2.P_settled.values, "--", color=COL[model], lw=1.2)
s0 = SET[(SET.experiment == "present") & (SET.window == "600phd") & (SET.model == "L10") & (SET.truth == "C one-off")]
ax.plot(xs, s0.P_settled.values, "-", color="0.2", lw=2, label="one-off true (BF<1/100), L10 k")
ax.set_ylim(0, 1.02); ax.set_ylabel("P(Bayes factor > 100 either way)"); ax.set_xlabel("data on disk by end of year")
ax.set_title("Present generation, 200–270 keV (solid best fit; dashed ×0.30)", fontsize=9)
ax.legend(fontsize=7, frameon=False, loc="lower right")
ax = axes[1]
for model in MODELS:
    c = curves[("present", "600phd", model)]
    for alpha, ls in [(2.0, "-"), (1.5, "--")]:
        P = PI_DM["median"] / (1 - PI_DM["median"]) * (1 + c["mu"]) ** (-alpha)
        P = P / (1 + P)
        m = TGRID >= NOW - 0.05
        ax.plot(TGRID[m], P[m], ls, color=COL[model], lw=2 if alpha == 2 else 1.2,
                label=SHORT[model] + (" (flat)" if alpha == 2 else " (Jeffreys)") if model in ("L10", "d366") else None)
ax.axhline(1e-3, color="0.5", lw=0.8); ax.text(2031.3, 1.15e-3, "P(DM) = 10⁻³", fontsize=7, color="0.4")
ax.set_yscale("log"); ax.set_ylim(1e-5, 3e-2); ax.set_xlim(NOW - 0.05, 2032)
ax.set_ylabel("P(DM | no new event), prior 0.015 (P061)"); ax.set_xlabel("calendar year (data on disk)")
ax.set_title("Null branch: zero events in the present generation", fontsize=9)
ax.legend(fontsize=7, frameon=False)
fig.suptitle("P088 Fig. 3 — When is the question settled? Gamma–Poisson Bayes factors and the null-branch posterior", fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "P088_fig3_settled_null.png"), dpi=160); plt.close(fig)

# Fig 2: decision tree (drawn from the computed numbers)
def g(ename, wname, model, col):
    return TL[(TL.experiment == ename) & (TL.window == wname) & (TL.model == model)].iloc[0][col]


def box(ax, x, y, w, h, text, fc="#f4f4f2", ec="0.3", fs=7.6, bold=False):
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.02", fc=fc, ec=ec, lw=1)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x0, y0, x1, y1, text="", col="0.3"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.1))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, fontsize=7, ha="center", va="center",
                bbox=dict(fc="white", ec="none", pad=0.5))


sbz = {m: SB[(SB.model == m) & (SB.dataset == "SR3+untouched")].iloc[0] for m in MODELS}
pres_mu_now = {m: g("present", "600phd", m, "mu_now") for m in MODELS}
p5_27 = {m: g("present", "600phd", m, "P5_best_by2027") for m in MODELS}
p5_27c = {m: g("present", "600phd", m, "P5_comb_by2027") for m in MODELS}
null_med = NULL[(NULL.experiment == "present") & (NULL.window == "600phd") & (NULL.alpha == 2.0) & (NULL.prior == "median")]
null_L10 = null_med[null_med.model == "L10"].iloc[0]["date_P_lt_0.001"]
null_366 = null_med[null_med.model == "d366"].iloc[0]["date_P_lt_0.001"]
ab_pres = AB[(AB.experiment == "present")]
sh_L10 = ab_pres[(ab_pres.window == "1000phd") & (ab_pres.truth == "L10")].iloc[0]["shape_med"]
sh_366 = ab_pres[(ab_pres.window == "1000phd") & (ab_pres.truth == "d366")].iloc[0]["shape_med"]
sh_366_600 = ab_pres[(ab_pres.window == "600phd") & (ab_pres.truth == "d366")].iloc[0]["shape_med"]
mod_366 = ab_pres[(ab_pres.window == "600phd") & (ab_pres.truth == "d366")].iloc[0]["mod3_med"]
mod_350 = ab_pres[(ab_pres.window == "600phd") & (ab_pres.truth == "d350")].iloc[0]["mod3_med"]
mod_300 = ab_pres[(ab_pres.window == "600phd") & (ab_pres.truth == "d300")].iloc[0]["mod3_med"]
mod_366_1000 = ab_pres[(ab_pres.window == "1000phd") & (ab_pres.truth == "d366")].iloc[0]["mod3_med"]
xl = {m: g("XLZD", "1000phd", m, "asimov5_new") for m in MODELS}
caw10 = CAW[(CAW.scenario.str.startswith("10 kg")) & (CAW.bkg == "b=0.01/kg.yr")].set_index("model")
caw100 = CAW[(CAW.scenario.str.startswith("100 kg")) & (CAW.bkg == "b=0.01/kg.yr")].set_index("model")

def od(t):  # short label for the tree
    s = ym(t)
    return "already on disk" if s == "on disk" else s


fig, ax = plt.subplots(figsize=(13.5, 10)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
FS = 7.2
box(ax, 50, 95.5, 96, 6.5,
    f"NOW (Sep 2026): 18 t·yr of xenon data above 200 keV never examined (P069; LZ 7.6, XENONnT 6.2, PandaX-4T 4.1 t·yr).\n"
    f"Expected new 200–270 keV events at LZ's best fit: L10 {pres_mu_now['L10']:.1f}, δ=300 {pres_mu_now['d300']:.1f}, δ=350 {pres_mu_now['d350']:.1f}, "
    f"δ=366 {pres_mu_now['d366']:.1f}; modelled background 0.004 (P016)",
    fc="#e3edf9", bold=True, fs=FS)
box(ax, 25, 84, 46, 9,
    f"STEP 1a — LZ opens its own 600–1000 phd sideband, 9.6 t·yr (P038, P020)\n"
    f"expected: L10 {sbz['L10'].mu_sideband:.1f}, δ=350 {sbz['d350'].mu_sideband:.1f}, δ=366 {sbz['d366'].mu_sideband:.0f}, δ=380 {sbz['d380'].mu_sideband:.0f} events; bkg {sbz['d366'].b_sideband:.3f}\n"
    f"zero events: δ=366 excluded (P(0) = {sbz['d366'].P0_if_true:.0e}), δ=350 disfavoured (P(0) = {sbz['d350'].P0_if_true:.2f});\n"
    f"the SR3 sideband alone already gives P(0) = 0.02 (δ=366), 5e-14 (δ=380)",
    fc="#fff4e5", fs=FS)
box(ax, 75, 84, 46, 9,
    f"STEP 1b — pre-registered 200–270 keV reanalysis: XENONnT + PandaX-4T + LZ untouched (P069)\n"
    f"P(≥2 new events | best fit): L10 {1-stats.poisson.cdf(1,pres_mu_now['L10']):.2f}, δ=350 {1-stats.poisson.cdf(1,pres_mu_now['d350']):.2f}, δ=366 {1-stats.poisson.cdf(1,pres_mu_now['d366']):.2f}\n"
    f"2 events = 5.5σ with LZ's event, 4.3σ new-only (b = 0.004); P(0 | L10) = {np.exp(-pres_mu_now['L10']):.2f}\n"
    f"earliest publication mid-2027 (P069); every date below is 'data on disk' + analysis latency",
    fc="#fff4e5", fs=FS)
arrow(ax, 50, 92.2, 25, 88.6); arrow(ax, 50, 92.2, 75, 88.6)
# branches
box(ax, 17, 67, 32, 12,
    f"NULL BRANCH — 0 new events\n"
    f"best fit excluded at 90 % (UL 0.89× L10, 0.39× δ366; P069)\n"
    f"P(DM): 0.015 (P061) → 10⁻³ by {od(null_L10)} (L10),\n"
    f"{od(null_366)} (δ=366); 10⁻⁴ (L10) not before 2032\n"
    f"Bayes factor DM:one-off < 1/100 needs 75 t·yr (L10)",
    fc="#eeeeee", fs=FS)
box(ax, 50, 67, 30, 12,
    f"ONE new event (P = {stats.poisson.pmf(1,pres_mu_now['L10']):.2f} | L10)\n"
    f"4.3σ with LZ's event (P069)\n"
    f"BF(DM : one-off) = {bf_dm_vs_oneoff(1, pres_mu_now['L10'], 0.004)[0]:.0f} — not settled\n"
    f"its energy decides: < 55 keV kills inelastic,\n"
    f"> 300 keV favours δ ≥ 366 by 6–7 (P057)",
    fc="#fff9db", fs=FS)
box(ax, 83, 67, 32, 12,
    f"≥ 2 new events — 5σ, BF > 100: a population exists\n"
    f"P(5σ new-only by end-2027 | best fit):\n"
    f"L10 {p5_27['L10']:.2f}, δ=300 {p5_27['d300']:.2f}, δ=350 {p5_27['d350']:.2f}, δ=366 {p5_27['d366']:.2f}\n"
    f"(with LZ's event {p5_27c['L10']:.2f}/{p5_27c['d300']:.2f}/{p5_27c['d350']:.2f}/{p5_27c['d366']:.2f})\n"
    f"a free-rate unknown still caps P(DM) at 0.09 (P001)",
    fc="#e6f5ec", fs=FS)
arrow(ax, 38, 79.4, 20, 73.2, "N = 0"); arrow(ax, 55, 79.4, 50, 73.2, "N = 1"); arrow(ax, 70, 79.4, 80, 73.2, "N ≥ 2")
# A vs B stage
box(ax, 50, 47, 96, 14,
    "STEP 2 — A (inelastic) vs B (L10) vs a steady unknown: counting cannot do it (P001); shape, timing, target\n"
    f"• spectral shape (P057 toys; P050 4–6): 2 events if L10 is true, 5–7 if inelastic — {od(sh_L10)} (L10) / {od(sh_366_600)} (δ=366, 600 phd edge)\n"
    f"• June clustering (P034): 3σ needs 6.5 / 11.5 / 97 events at δ = 366 / 350 / 300 keV → {od(mod_366)} / {od(mod_350)} / {od(mod_300)} (600 phd); 5σ at δ=366: {od(ab_pres[(ab_pres.window=='600phd')&(ab_pres.truth=='d366')].iloc[0]['mod5_med'])}\n"
    f"• tungsten (P046): CRESST-scale 10 kg·yr CaWO₄ (2027–30) holds {caw10.loc['d366','mu_total']:.1f} / {caw10.loc['d350','mu_total']:.2f} / {caw10.loc['d300','mu_total']:.2f} events at δ = 366 / 350 / 300 — blind except δ ≥ 366; "
    f"100 kg·yr (2029–34): {caw100.loc['d350','mu_total']:.1f} events at δ=350, 3σ vs L10 only for δ ≥ 350\n"
    f"• ¹³⁶Xe-enriched/natural pair (P047): 9 events, 12.8 t·yr per detector — a 5 t enriched TPC needs {ISO.iloc[1].years_to_12p8_tyr:.1f} yr of running; nothing before the mid-2030s",
    fc="#f4f4f2", fs=FS)
arrow(ax, 83, 61, 62, 54.1); arrow(ax, 50, 61, 50, 54.1)
box(ax, 50, 27, 96, 10,
    f"STEP 3 — XLZD 60 t, 400 keV ROI, 48 t·yr/yr from 2032 (recalled, uncertain; P050): new-only 5σ by {ym(xl['L10'])} (L10) / {ym(xl['d300'])} (δ=300) / {ym(xl['d350'])} (350) / {ym(xl['d366'])} (366)\n"
    "at the best fit; 5σ within 2033 even at ×0.30; shape (4–6 events) and June clustering inside its first year (P050, P034);\n"
    "zero events exclude LZ's lower 68 % edge within 2032 (P050). If nothing has appeared by then, P(DM) ≲ 10⁻⁴ for every prior in P061's 68 % range.",
    fc="#e3edf9", fs=FS)
arrow(ax, 50, 39.9, 50, 32.1)
box(ax, 50, 11, 96, 9,
    "VERDICTS (median, data on disk; add ≈ 1 yr for analysis).  A/B vs one-off, 5σ new-only, present generation, 600 phd edge: "
    f"{ym(g('present','600phd','L10','toy5_best_med'))} (L10), {ym(g('present','600phd','d300','toy5_best_med'))} (δ=300), on disk (δ ≥ 350);\n"
    f"at ×0.30: {ym(g('present','600phd','L10','toy5_lo68_med'))} (L10), {ym(g('present','600phd','d366','toy5_lo68_med'))} (δ=366) — XLZD 2033.  A vs B: shape — events already on disk; timing — {od(mod_366)} (δ=366), {od(mod_350)} (δ=350), never (δ=300) before XLZD.\n"
    f"Null: P(DM) < 10⁻³ by {od(null_L10)} (L10) with zero events; δ ≥ 350 with a 1000 phd edge decided by LZ's own sideband on reanalysis.",
    fc="#fdf0e6", bold=True, fs=FS)
arrow(ax, 50, 21.9, 50, 15.6)
ax.set_title("P088 Fig. 2 — Decision tree for the LZ 248 keV event (numbers from this paper's tables; corpus sources in brackets)", fontsize=10)
fig.savefig(os.path.join(FIG, "P088_fig2_decision_tree.png"), dpi=160, bbox_inches="tight"); plt.close(fig)
print(f"[{time.time()-T0:5.1f}s] figures done")

# ----------------------------------------------------------------------------------------------
# 9. Results JSON (headline numbers)
# ----------------------------------------------------------------------------------------------
def rowd(ename, wname, model):
    r = TL[(TL.experiment == ename) & (TL.window == wname) & (TL.model == model)].iloc[0]
    keys = ["E_new_now", "mu_now", "P0_now", "mu_end2027", "mu_end2028", "mu_2030", "asimov3_new", "asimov5_new", "asimov3_comb",
            "asimov5_comb", "asimov5_new_lo68", "toy3_best_med", "toy3_best_q16", "toy3_best_q84", "toy5_best_med", "toy5_best_q16", "toy5_best_q84",
            "toy5_lo68_med", "toy5_PG_med", "toy5_comb_med", "P3_best_by2027", "P5_best_by2027", "P5_best_by2028", "P5_best_by2030",
            "P5_lo68_by2028", "P5_PG_by2028", "P5_comb_by2027", "P5_comb_by2028"]
    return {k: (float(r[k]) if np.isfinite(r[k]) else None) for k in keys}


res = {
    "date_now": NOW,
    "exposure_on_disk_now_tyr": {k: float(EXP[k][i_now]) for k in EXP},
    "exposure_by_year_tyr": {str(y): {k: float(np.interp(y, TGRID, EXP[k])) for k in EXP} for y in (2027, 2028, 2029, 2031, 2033)},
    "timeline": {f"{e}|{w}|{m}": rowd(e, w, m) for e in EXPTS for w in WINDOWS for m in MODELS},
    "LZ_sideband": SB.to_dict(orient="records"),
    "settled_present_600phd": SET[(SET.experiment == "present") & (SET.window == "600phd")].to_dict(orient="records"),
    "settled_present_1000phd": SET[(SET.experiment == "present") & (SET.window == "1000phd")].to_dict(orient="records"),
    "null_branch_present": NULL[NULL.experiment == "present"].to_dict(orient="records"),
    "AvsB_present": AB[AB.experiment == "present"].to_dict(orient="records"),
    "cawo4": CAW.to_dict(orient="records"),
    "isotope_pair": ISO.to_dict(orient="records"),
    "inputs": {"RATE": RATE, "B": B, "P038_EXTRA": P038_EXTRA, "N_SHAPE": N_SHAPE, "N_MOD": N_MOD, "MOD_GAIN_400": MOD_GAIN_400,
               "CAWO4_RATE": CAWO4_RATE, "PI_DM": PI_DM, "LEDGER": {k: {kk: (vv if not isinstance(vv, list) else [list(x) for x in vv]) for kk, vv in v.items()} for k, v in LEDGER.items()},
               "NTOY": NTOY, "rate_band": [RATE_LO, RATE_HI]},
    "runtime_s": time.time() - T0,
}


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


with open(os.path.join(OUT, "P088_results.json"), "w") as f:
    json.dump(_clean(res), f, indent=1)

# console digest
print("\n=== exposures on disk now (t.yr):", {k: round(float(EXP[k][i_now]), 2) for k in ["LZ", "XENONnT", "PandaX-4T", "present"]})
print("=== present generation, 600 phd (200-270 keV), new-only, best fit")
print(TL[(TL.experiment == "present") & (TL.window == "600phd")][["model", "mu_now", "P0_now", "mu_end2027", "mu_end2028", "asimov3_new", "asimov5_new", "toy3_best_med", "toy5_best_med", "toy5_best_q16", "toy5_best_q84", "toy5_lo68_med", "toy5_PG_med", "toy5_comb_med", "P5_best_by2027", "P5_best_by2028", "P5_lo68_by2028", "P5_comb_by2027"]].round(3).to_string())
print("=== present generation, 1000 phd (200-400 keV)")
print(TL[(TL.experiment == "present") & (TL.window == "1000phd")][["model", "mu_now", "P0_now", "asimov5_new", "toy3_best_med", "toy5_best_med", "toy5_lo68_med", "P5_best_by2027"]].round(3).to_string())
print("=== LZ alone / XENONnT / PandaX / XLZD (600 & 1000), toy5 median")
print(TL[TL.experiment.isin(["LZ", "XENONnT", "PandaX-4T", "XLZD"])][["experiment", "window", "model", "mu_now", "toy3_best_med", "toy5_best_med", "toy5_lo68_med", "asimov5_new"]].round(2).to_string())
print("=== LZ sideband"); print(SB.round(4).to_string())
print("=== settled (present, 600phd, best fit & one-off)")
print(SET[(SET.experiment == "present") & (SET.truth != "A/B x0.30")][["window", "model", "date", "k_expected", "truth", "P_BF_gt100", "P_BF_lt0p01", "P_settled", "BF_if_zero", "N_min_for_BF100"]].round(3).to_string())
print("=== null branch (present)"); print(NULL[(NULL.experiment == "present")].round(4).to_string())
print("=== A vs B (present)"); print(AB[AB.experiment == "present"].round(2).to_string())
print("=== CaWO4"); print(CAW.round(3).to_string())
print(ISO)
print(SUM.to_string())
print(f"done in {time.time()-T0:.1f} s")
