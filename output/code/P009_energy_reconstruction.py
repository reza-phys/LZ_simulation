"""
P009_energy_reconstruction.py -- recoil-energy reconstruction of the LZ event (S1c = 540.1 phd, S2c = 9268 phd)
with the LZ-tuned NEST model (arXiv:2609.02823, Supplemental Table S5), nestpy 2.1.1.

Run from the simulation root:   .venv/bin/python output/code/P009_energy_reconstruction.py
(after output/code/P009_digitise_bands.py, which produces output/work/P009/digitised_fig.json)

Sections
 1. mean-yield curves Nph(E), Ne(E) 100-350 keV for five yield variants -> S1c = g1 Nph, S2c = g2 Ne
 2. one-dimensional estimators E_S1, E_S2, E_comb (Nq inversion), alpha E^beta inversion; g1/g2 propagation
 3. two-dimensional Monte-Carlo likelihood in (S1c, log10 S2c) with nestpy GetQuanta fluctuations plus a
    simple detector model (binomial S1 detection, PMT resolution, binomial extraction, SE-size Fano, 2 %
    position-correction residual); NR band at fixed S1c for a flat spectrum; E_ML, 68 % interval;
    sigma below the median
 4. efficiency roll-off (Fig. S2 inset), epsilon(E_ML), P(S1c > 600 | E_ML), MC E50 per variant
 5. delta_max for inelastic DM at 248 / 250 / 265 keV; ER-equivalent energy
Outputs: output/work/P009/*.csv, results.json, figures/*.png
"""
from __future__ import annotations
import json, math, os, sys, time
import numpy as np
import pandas as pd
from scipy import stats, optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/reza/LZ_simulation"
os.chdir(ROOT)
sys.path.insert(0, "output/code")
from common import lzcommon as lz
import nestpy

OUT = "output/work/P009"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(20260904)

G1, G1E = lz.LZ["g1"], lz.LZ["g1_err"]          # 0.110 +- 0.002 phd/photon   (paper, detector paragraph)
G2, G2E = lz.LZ["g2"], lz.LZ["g2_err"]          # 34.5 +- 1.1 phd/electron
S1, S2 = lz.LZ["ev_S1c"], lz.LZ["ev_S2c"]       # 540.1 phd, 9268 phd
L2 = math.log10(S2)                              # 3.9670
FIELD = lz.DRIFT_FIELD_VCM                       # 96.5 V/cm (nestpy LZ_WS2024 central field)
DENSITY = 2.9

DET = nestpy.detectors.LZ_WS2024()
NC = nestpy.NESTcalc(DET)
WIDTH_LZ = list(DET.nr_er_width_parameters)      # NR/ER fluctuation parameters stored in the nestpy LZ detector
W_NEST = NC.WorkFunction(DENSITY).Wq_eV          # 13.44 eV (nestpy default work function)

# ----------------------------------------------------------------------------------------------------
# digitised figure information (P009_digitise_bands.py)
# ----------------------------------------------------------------------------------------------------
DIG = json.load(open(os.path.join(OUT, "digitised_fig.json")))
CONT = DIG["Fig4_science_sample_wbands"]["contours"]          # 150,200,250,300 keV crossings with NR median
BAND_DIG = DIG["Fig2_Calibrations"]["band_at_S1c"]["540.1"]["log10S2c_groups_low_to_high"]  # [p10, median, p90]
EV_DIG = DIG["Fig4_science_sample_wbands"]["event"]
# implied W from each contour crossing (contours are lines of constant S1c/g1 + S2c/g2)
W_impl = [c["E_ee_keVee"] * 1e3 / (c["S1c_at_NR_median"] / G1 + 10 ** c["log10S2c_at_NR_median"] / G2) for c in CONT]
W_PAPER = float(np.mean(W_impl))
# Nq(E) implied by the six printed contour labels with that W:  Nq = 1e3 E_ee / W ; fit alpha E^beta
LABELS = [(50.0, 11.8), (100.0, 25.5), (150.0, 40.0), (200.0, 55.1), (250.0, 70.6), (300.0, 86.5)]
_E = np.array([e for e, _ in LABELS]); _Q = np.array([q * 1e3 / W_PAPER for _, q in LABELS])
_b, _la = np.polyfit(np.log(_E), np.log(_Q), 1)
ALPHA_PAPER, BETA_PAPER = math.exp(_la), _b

# ----------------------------------------------------------------------------------------------------
# 1. yield variants
# ----------------------------------------------------------------------------------------------------
NR_LZ_NOBREAK = {k: v for k, v in lz.NEST_NR_LZ.items() if k not in ("a", "b", "E0")}
NR_LZ_B111 = dict(lz.NEST_NR_LZ, beta=1.11)      # beta as stored in nestpy's LZ_WS2024 detector (Table S5 prints 1.1)


def yields(E, variant):
    """Mean (Nph, Ne) for an NR of energy E [keV]."""
    if variant == "LZ-tab":
        return lz.nest_nr_yields(E, params=lz.NEST_NR_LZ)
    if variant == "LZ-nobreak":
        return lz.nest_nr_yields(E, params=NR_LZ_NOBREAK)
    if variant == "NEST-default":
        return lz.nest_nr_yields(E, params=lz.NEST_NR_DEFAULT)
    if variant == "LZ-beta1.11":
        return lz.nest_nr_yields(E, params=NR_LZ_B111)
    if variant == "paper-contour":
        # total quanta from the paper's own constant-energy contours; charge yield from the LZ-tuned Qy
        nph, ne = lz.nest_nr_yields(E, params=lz.NEST_NR_LZ)
        nq = ALPHA_PAPER * E ** BETA_PAPER
        return nq - ne, ne
    raise ValueError(variant)


VARIANTS = ["LZ-tab", "LZ-nobreak", "NEST-default", "LZ-beta1.11", "paper-contour"]
Egrid = np.arange(100.0, 350.01, 1.0)
curves = {}
rows = []
for v in VARIANTS:
    arr = np.array([yields(E, v) for E in Egrid])
    curves[v] = dict(E=Egrid, Nph=arr[:, 0], Ne=arr[:, 1], S1c=G1 * arr[:, 0], log10S2c=np.log10(G2 * arr[:, 1]))
    for E, (nph, ne) in zip(Egrid, arr):
        rows.append(dict(variant=v, E_keV=E, Nph=nph, Ne=ne, Nq=nph + ne, S1c=G1 * nph, S2c=G2 * ne,
                         log10S2c=math.log10(G2 * ne)))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "yield_curves.csv"), index=False)


def invert(curve_x, curve_E, target):
    """Invert a monotonic curve x(E) at x = target by linear interpolation."""
    return float(np.interp(target, curve_x, curve_E))


# ----------------------------------------------------------------------------------------------------
# 2. one-dimensional estimators
# ----------------------------------------------------------------------------------------------------
est_rows = []
for v in VARIANTS:
    c = curves[v]
    Nq_ev = S1 / G1 + S2 / G2
    E_s1 = invert(c["S1c"], Egrid, S1)
    E_s2 = invert(c["log10S2c"], Egrid, L2)
    E_cb = invert(c["Nph"] + c["Ne"], Egrid, Nq_ev)
    # g1, g2 propagation (one at a time, +-1 sigma)
    E_s1_g1 = [invert(G1 * c["Nph"] / G1 * g, Egrid, S1) for g in (G1 - G1E, G1 + G1E)]
    E_s2_g2 = [invert(np.log10(g * c["Ne"]), Egrid, L2) for g in (G2 - G2E, G2 + G2E)]
    E_cb_g = [invert(c["Nph"] + c["Ne"], Egrid, S1 / g1 + S2 / g2) for g1, g2 in
              ((G1 + G1E, G2 + G2E), (G1 - G1E, G2 - G2E))]
    # local slopes (for the statistical error later)
    dlnS1_dlnE = np.gradient(np.log(c["S1c"]), np.log(Egrid))[np.searchsorted(Egrid, E_s1)]
    dlog10S2_dE = np.gradient(c["log10S2c"], Egrid)[np.searchsorted(Egrid, min(E_s2, 349))]
    est_rows.append(dict(variant=v, E_S1=E_s1, E_S1_g1lo=E_s1_g1[0], E_S1_g1hi=E_s1_g1[1],
                         E_S2=E_s2, E_S2_g2lo=E_s2_g2[0], E_S2_g2hi=E_s2_g2[1],
                         E_comb=E_cb, E_comb_glo=E_cb_g[0], E_comb_ghi=E_cb_g[1],
                         S1c_at_248=float(np.interp(248, Egrid, c["S1c"])),
                         log10S2c_at_248=float(np.interp(248, Egrid, c["log10S2c"])),
                         S1c_at_270=float(np.interp(270, Egrid, c["S1c"])),
                         Nq_at_250=float(np.interp(250, Egrid, c["Nph"] + c["Ne"])),
                         dlnS1_dlnE=float(dlnS1_dlnE), dlog10S2_dE=float(dlog10S2_dE)))
EST = pd.DataFrame(est_rows)
EST.to_csv(os.path.join(OUT, "estimators_1D.csv"), index=False)
# paper-style alpha E^beta inversion of the total quanta
Nq_ev = S1 / G1 + S2 / G2
E_alpha_beta = (Nq_ev / 11.2) ** (1 / 1.1)
E_alpha_beta_111 = (Nq_ev / 11.2) ** (1 / 1.11)
E_alpha_beta_paper = (Nq_ev / ALPHA_PAPER) ** (1 / BETA_PAPER)
# paper-contour interpolation of the event S1c between the digitised crossings
cE = np.array([c["E_nr_keV"] for c in CONT]); cS = np.array([c["S1c_at_NR_median"] for c in CONT])
E_contour_S1 = float(np.interp(S1, cS, cE))
# and via the E_ee labels (combined quanta with W_PAPER)
lab_E = np.array([e for e, _ in LABELS]); lab_Q = np.array([q for _, q in LABELS])
E_contour_Nq = float(np.interp(Nq_ev * W_PAPER / 1e3, lab_Q, lab_E))

# ----------------------------------------------------------------------------------------------------
# 3. Monte-Carlo detector model and 2D likelihood
# ----------------------------------------------------------------------------------------------------
EXT_EFF = 0.726          # LZ_WS2024 extraction efficiency from nestpy CalculateG2 (documented in details.md)
SE_SIZE = G2 / EXT_EFF   # phd per extracted electron so that <S2c> = g2 Ne
S2_FANO = DET.get_s2Fano()   # 4.0
SPE_RES = DET.get_sPEres()   # 0.338
POS_RES = 0.02           # assumed residual of position corrections (relative), applied to S1c and S2c


def quanta_samples(E, variant, n):
    """n samples of (Nph, Ne) from nestpy GetQuanta with the LZ width parameters.  For the variants that
    only rescale the mean total quanta, the YieldResult is built from the LZ-tab yields and shifted."""
    nph_m, ne_m = yields(E, variant)
    if variant in ("LZ-tab", "LZ-nobreak", "NEST-default", "LZ-beta1.11"):
        params = {"LZ-tab": lz.NEST_NR_LZ, "LZ-nobreak": NR_LZ_NOBREAK, "NEST-default": lz.NEST_NR_DEFAULT,
                  "LZ-beta1.11": NR_LZ_B111}[variant]
        y = NC.GetYields(nestpy.interactions.NR, float(E), DENSITY, FIELD, 131.293, 54,
                         lz.nest_nr_params_vector(E, params))
        out = np.empty((n, 2))
        for i in range(n):
            q = NC.GetQuanta(y, DENSITY, WIDTH_LZ)
            out[i] = q.photons, q.electrons
        return out
    # paper-contour: same fluctuations as LZ-tab, photons shifted by the difference of the means
    base = quanta_samples(E, "LZ-tab", n)
    nph0, _ = yields(E, "LZ-tab")
    base[:, 0] += nph_m - nph0
    return base


def detect(q, g1=G1, g2=G2, pos_res=POS_RES, s2_scale=1.0):
    """Detector response: S1c and log10 S2c from quanta samples q[:, (Nph, Ne)]."""
    nph = np.clip(np.round(q[:, 0]).astype(int), 0, None)
    ne = np.clip(np.round(q[:, 1]).astype(int), 0, None)
    n1 = rng.binomial(nph, g1).astype(float)
    n1 = n1 * (1 + rng.normal(0, SPE_RES / np.sqrt(np.clip(n1, 1, None))))     # PMT single-phe resolution
    s1c = n1 * (1 + rng.normal(0, pos_res, n1.size))
    next_ = rng.binomial(ne, EXT_EFF).astype(float)
    s2 = next_ * (g2 / EXT_EFF) * (1 + rng.normal(0, np.sqrt(S2_FANO / np.clip(next_ * (g2 / EXT_EFF), 1, None))))
    s2 = s2 * (1 + rng.normal(0, pos_res, s2.size))
    l2 = np.log10(np.clip(s2, 1, None))
    if s2_scale != 1.0:      # optional rescaling of the S2 spread about its mean (band-width calibration)
        l2 = l2.mean() + s2_scale * (l2 - l2.mean())
    return s1c, l2


def gauss2_loglike(s1c, l2, x, y):
    """log of a bivariate-normal density fitted to the samples, evaluated at (x, y)."""
    mu = np.array([s1c.mean(), l2.mean()])
    cov = np.cov(np.vstack([s1c, l2]))
    return stats.multivariate_normal(mu, cov).logpdf([x, y]), mu, cov


E_scan = np.arange(180.0, 320.01, 2.0)
N_MC = 20000
scan_rows = []
like = {}
t0 = time.time()
for v in VARIANTS:
    ll = np.empty(E_scan.size); ll_kde = np.full(E_scan.size, np.nan)
    for i, E in enumerate(E_scan):
        q = quanta_samples(E, v, N_MC)
        s1c, l2 = detect(q)
        ll[i], mu, cov = gauss2_loglike(s1c, l2, S1, L2)
        if v in ("LZ-tab", "paper-contour") and int(E) % 10 == 0:
            kde = stats.gaussian_kde(np.vstack([s1c, l2]))
            ll_kde[i] = math.log(max(kde([S1, L2])[0], 1e-300))
        scan_rows.append(dict(variant=v, E_keV=E, lnL=ll[i], lnL_kde=ll_kde[i], mean_S1c=mu[0], mean_log10S2c=mu[1],
                              sd_S1c=math.sqrt(cov[0, 0]), sd_log10S2c=math.sqrt(cov[1, 1]),
                              corr=cov[0, 1] / math.sqrt(cov[0, 0] * cov[1, 1]),
                              P_S1c_gt_600=float((s1c > 600).mean())))
    like[v] = ll
print("MC scan time %.1f s" % (time.time() - t0))
SCAN = pd.DataFrame(scan_rows)
SCAN.to_csv(os.path.join(OUT, "likelihood_scan.csv"), index=False)


def ml_interval(E, ll):
    """Maximum-likelihood energy and 68 % interval (delta lnL = 0.5) on a fine spline of the scan."""
    fine = np.arange(E[0], E[-1] + 0.01, 0.1)
    llf = np.interp(fine, E, ll)
    # smooth slightly (MC noise) with a quadratic fit around the maximum
    i = int(np.argmax(llf)); sel = (fine > fine[i] - 25) & (fine < fine[i] + 25)
    p = np.polyfit(fine[sel], llf[sel], 2)
    Eml = -p[1] / (2 * p[0])
    sig = math.sqrt(-0.5 / p[0]) if p[0] < 0 else float("nan")
    # numerical interval from the (unsmoothed) profile
    lmax = np.polyval(p, Eml)
    inside = fine[llf > lmax - 0.5]
    return float(Eml), float(sig), float(inside.min()), float(inside.max())


ML = {}
for v in VARIANTS:
    Eml, sig, lo, hi = ml_interval(E_scan, like[v])
    ML[v] = dict(E_ML=Eml, sigma_quadratic=sig, lo68=lo, hi68=hi)
# systematic variations for the two main variants: g1 +-2 %, g2 +-3 %, POS_RES 0 / 0.04
SYS = {}
for v in ("LZ-tab", "paper-contour"):
    SYS[v] = {}
    for tag, kw in [("g1-", dict(g1=G1 - G1E)), ("g1+", dict(g1=G1 + G1E)), ("g2-", dict(g2=G2 - G2E)),
                    ("g2+", dict(g2=G2 + G2E)), ("pos0", dict(pos_res=0.0)), ("pos4", dict(pos_res=0.04))]:
        ll = np.empty(E_scan.size)
        for i, E in enumerate(E_scan):
            q = quanta_samples(E, v, 8000)
            s1c, l2 = detect(q, **kw)
            ll[i] = gauss2_loglike(s1c, l2, S1, L2)[0]
        Eml, sig, lo, hi = ml_interval(E_scan, ll)
        SYS[v][tag] = dict(E_ML=Eml, sigma=sig, lo68=lo, hi68=hi)

# NR band at fixed S1c for a flat energy spectrum (as in the paper's Fig. 2 caption)
BAND = {}
for v in ("LZ-tab", "paper-contour"):
    Eflat = rng.uniform(150, 400, 400000)
    s1_all = np.empty(Eflat.size); l2_all = np.empty(Eflat.size)
    # evaluate in 2-keV bins for speed
    bins = np.arange(150, 402, 2.0)
    idx = np.digitize(Eflat, bins) - 1
    for k in range(bins.size - 1):
        m = idx == k
        if not m.any():
            continue
        q = quanta_samples(0.5 * (bins[k] + bins[k + 1]), v, int(m.sum()))
        s1c, l2 = detect(q)
        s1_all[m] = s1c; l2_all[m] = l2
    sel = np.abs(s1_all - S1) < 5.0
    p10, p50, p90 = np.percentile(l2_all[sel], [10, 50, 90])
    sd = l2_all[sel].std()
    BAND[v] = dict(n=int(sel.sum()), p10=float(p10), p50=float(p50), p90=float(p90), sd=float(sd),
                   halfwidth_10_90=float(0.5 * (p90 - p10)), event_offset_dex=float(p50 - L2),
                   sigma_below_median_MC=float((p50 - L2) / sd),
                   sigma_below_median_gauss_equiv_from_1090=float((p50 - L2) / (0.5 * (p90 - p10) / 1.2816)),
                   mean_E_in_slice=float(Eflat[sel].mean()))
# digitised band comparison
dig_half = 0.5 * (BAND_DIG[2] - BAND_DIG[0]); dig_med = BAND_DIG[1]
DIGBAND = dict(p10=BAND_DIG[0], p50=dig_med, p90=BAND_DIG[2], halfwidth_10_90=dig_half,
               sigma_gauss_equiv=dig_half / 1.2816, event_offset_dex=dig_med - L2,
               sigma_below_median=(dig_med - L2) / (dig_half / 1.2816),
               sigma_implied_by_paper_1p5=(dig_med - L2) / 1.5)

# ----------------------------------------------------------------------------------------------------
# 4. efficiency roll-off
# ----------------------------------------------------------------------------------------------------
# readings of the Fig. S2 inset (black "+ROI" curve), by eye from the PNG, +-0.03 in efficiency
INSET = np.array([[250, 0.93], [255, 0.88], [260, 0.79], [265, 0.66], [270, 0.50], [275, 0.35], [280, 0.22],
                  [285, 0.12], [290, 0.05], [295, 0.02], [300, 0.005]])
PLATEAU = 0.955   # green "+SS & cuts" level in Fig. S2 (paper: 96 % average 14-250 keV)
E50 = lz.LZ["E_50pct_high_keV"]   # 269.9


def eff_model(E, sigE, plateau=PLATEAU, e50=E50):
    return plateau * stats.norm.cdf((e50 - E) / sigE)


sigE_fit, _ = optimize.curve_fit(lambda E, s: eff_model(E, s), INSET[:, 0], INSET[:, 1], p0=[12.0])
SIG_E = float(sigE_fit[0])
resid = INSET[:, 1] - eff_model(INSET[:, 0], SIG_E)
EFF = dict(sigma_E_keV=SIG_E, max_abs_residual=float(np.abs(resid).max()), plateau=PLATEAU, E50=E50)
for v in VARIANTS:
    ML[v]["eff_at_EML"] = float(eff_model(ML[v]["E_ML"], SIG_E))
    ML[v]["eff_at_lo68"] = float(eff_model(ML[v]["lo68"], SIG_E))
    ML[v]["eff_at_hi68"] = float(eff_model(ML[v]["hi68"], SIG_E))
    # MC probability of failing the S1c < 600 cut at E_ML, and the MC E50 (plateau x P(S1c<600) = 0.5)
    sub = SCAN[SCAN.variant == v]
    pf = np.interp(ML[v]["E_ML"], sub.E_keV, sub.P_S1c_gt_600)
    ML[v]["P_S1c_gt_600_at_EML"] = float(pf)
    effmc = PLATEAU * (1 - sub.P_S1c_gt_600.values)
    ML[v]["E50_MC"] = float(np.interp(0.5, effmc[::-1], sub.E_keV.values[::-1]))
    ML[v]["S1c_median_at_269.9"] = float(np.interp(E50, sub.E_keV, sub.mean_S1c))
ML_eff_paper = dict(eff_at_248=float(eff_model(248, SIG_E)), eff_at_265=float(eff_model(265, SIG_E)),
                    eff_at_271=float(eff_model(271, SIG_E)))

# ----------------------------------------------------------------------------------------------------
# 5. kinematics and ER-equivalent energy
# ----------------------------------------------------------------------------------------------------
vmax_june = lz.vmax_kms(lz.v_earth_kms(167))
DELTA = {str(E): dict(m1000_June=lz.delta_max_kev(E, 1000, v_kms=vmax_june), m1000_avg=lz.delta_max_kev(E, 1000),
                      m400_June=lz.delta_max_kev(E, 400, v_kms=vmax_june), m4000_June=lz.delta_max_kev(E, 4000, v_kms=vmax_june),
                      m_min_elastic_June=lz.m_chi_min_gev(E, v_kms=vmax_june))
         for E in (248.0, 250.0, 265.0)}
EEE = dict(W13p7=lz.combined_energy_keV(S1, S2, W_eV=13.7), W13p5=lz.combined_energy_keV(S1, S2, W_eV=13.5),
           W_nestpy=lz.combined_energy_keV(S1, S2, W_eV=W_NEST), W_paper=lz.combined_energy_keV(S1, S2, W_eV=W_PAPER),
           Nph=S1 / G1, Ne=S2 / G2, Nq=Nq_ev, W_nestpy_eV=W_NEST, W_paper_eV=W_PAPER)
# ER band position of a beta ER with the same total quanta (LZ-tuned ER model) and the recombination needed
er_nph, er_ne = lz.nest_er_yields(EEE["W_paper"], params=lz.NEST_ER_LZ)
EEE.update(ER_model_S1c=G1 * er_nph, ER_model_log10S2c=math.log10(G2 * er_ne),
           ER_model_electron_fraction=er_ne / (er_nph + er_ne), event_electron_fraction=EEE["Ne"] / Nq_ev)

# ----------------------------------------------------------------------------------------------------
# collect results
# ----------------------------------------------------------------------------------------------------
RES = dict(
    inputs=dict(g1=G1, g1_err=G1E, g2=G2, g2_err=G2E, S1c=S1, S2c=S2, log10S2c=L2, field_Vcm=FIELD,
                nestpy_detector="LZ_WS2024 (g1=0.1122 in object; we use the paper's g1 = 0.110 and g2 = 34.5 externally)",
                width_parameters=WIDTH_LZ, ext_eff=EXT_EFF, SE_size_phd=SE_SIZE, s2Fano=S2_FANO, sPEres=SPE_RES,
                pos_res=POS_RES, N_MC_per_energy=N_MC, E_scan=[float(E_scan[0]), float(E_scan[-1]), 2.0]),
    digitised=dict(W_implied_per_contour_eV=W_impl, W_paper_eV=W_PAPER, alpha_paper=ALPHA_PAPER, beta_paper=BETA_PAPER,
                   contours=CONT, event_marker=EV_DIG, band_at_540=DIGBAND),
    estimators_1D=EST.to_dict(orient="records"),
    E_alpha_beta=dict(beta1p1=E_alpha_beta, beta1p11=E_alpha_beta_111, paper_fit=E_alpha_beta_paper),
    E_from_paper_contours=dict(S1c_interpolation=E_contour_S1, Nq_interpolation=E_contour_Nq),
    likelihood_2D=ML, systematics_2D=SYS, band_fixed_S1c=BAND,
    efficiency=EFF, efficiency_at_paper_energies=ML_eff_paper, delta_max=DELTA, ER_equivalent=EEE,
    paper=dict(E=248, stat=23, sys=23, sigma_below_NR=1.5, E50=E50),
)
with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(RES, f, indent=1, default=float)

# ----------------------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 5.2))
cols = dict(zip(VARIANTS, ["C3", "C1", "C0", "C2", "k"]))
for v in VARIANTS:
    c = curves[v]
    ax.plot(c["S1c"], c["log10S2c"], color=cols[v], lw=1.5, label=f"{v} mean (100-350 keV)")
    for E in (200, 250, 300):
        i = np.searchsorted(Egrid, E)
        ax.plot(c["S1c"][i], c["log10S2c"][i], "o", color=cols[v], ms=4)
# MC scatter at E_ML for LZ-tab and paper-contour
for v, mk in (("LZ-tab", "."), ("paper-contour", ".")):
    q = quanta_samples(ML[v]["E_ML"], v, 3000); s1c, l2 = detect(q)
    ax.scatter(s1c, l2, s=2, color=cols[v], alpha=0.15, label=f"{v}: MC at E_ML = {ML[v]['E_ML']:.0f} keV")
ax.errorbar([S1], [L2], fmt="*", color="gold", mec="k", ms=16, label="event (540.1, 3.967)", zorder=10)
ax.plot([c["S1c_at_NR_median"] for c in CONT], [c["log10S2c_at_NR_median"] for c in CONT], "s", mfc="none", mec="k",
        ms=9, label="paper contours x NR median (digitised Fig. 4)")
for c in CONT:
    ax.annotate(f"{c['E_nr_keV']:.0f}", (c["S1c_at_NR_median"], c["log10S2c_at_NR_median"] + 0.012), ha="center", fontsize=8)
ax.plot([540.1] * 3, BAND_DIG, "_", color="0.4", ms=14, label="digitised band p10/p50/p90 at 540 phd")
ax.axvline(600, ls="--", color="0.5"); ax.text(603, 3.88, "ROI S1c = 600", rotation=90, fontsize=8)
ax.set_xlim(250, 720); ax.set_ylim(3.85, 4.12)
ax.set_xlabel("S1c [phd]"); ax.set_ylabel("log10(S2c [phd])")
ax.set_title("NR band mean curves, NEST-fluctuation MC and the LZ event")
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "P009_band_and_event.png"), dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 5.0))
for v in VARIANTS:
    ll = like[v]; ax.plot(E_scan, ll - ll.max(), color=cols[v], lw=1.6,
                          label=f"{v}: E_ML = {ML[v]['E_ML']:.1f} [{ML[v]['lo68']:.0f}, {ML[v]['hi68']:.0f}] keV")
sub = SCAN[(SCAN.variant == "LZ-tab") & np.isfinite(SCAN.lnL_kde)]
ax.plot(sub.E_keV, sub.lnL_kde - sub.lnL_kde.max(), "x", color="C3", ms=6, label="LZ-tab, KDE check")
ax.axhline(-0.5, ls=":", color="0.5"); ax.axvspan(248 - 23, 248 + 23, color="0.85", label="paper 248 +- 23 (stat)")
ax.axvline(E50, ls="--", color="0.3"); ax.text(E50 + 1, -7.5, "50 % eff. 269.9 keV", rotation=90, fontsize=8)
ax2 = ax.twinx(); Ef = np.linspace(180, 320, 300); ax2.plot(Ef, eff_model(Ef, SIG_E), color="0.5", lw=1, ls="-.")
ax2.set_ylabel("efficiency model (dash-dot)"); ax2.set_ylim(0, 1.05)
ax.set_ylim(-8, 0.5); ax.set_xlim(180, 320)
ax.set_xlabel("nuclear recoil energy E [keV]"); ax.set_ylabel("ln L(E) - max")
ax.set_title("2D (S1c, log10 S2c) likelihood of the event vs recoil energy")
ax.legend(fontsize=7, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "P009_likelihood_vs_E.png"), dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(INSET[:, 0], INSET[:, 1], "ko", label="Fig. S2 inset (read by eye)")
ax.plot(Ef, eff_model(Ef, SIG_E), "k-", label=f"0.955 Phi((269.9-E)/{SIG_E:.1f} keV)")
for v in ("LZ-tab", "paper-contour", "LZ-beta1.11"):
    sub = SCAN[SCAN.variant == v]
    ax.plot(sub.E_keV, PLATEAU * (1 - sub.P_S1c_gt_600), "--", color=cols[v], label=f"MC 0.955 P(S1c<600|E), {v}")
    ax.axvline(ML[v]["E_ML"], color=cols[v], lw=0.8)
ax.set_xlim(220, 320); ax.set_ylim(0, 1); ax.set_xlabel("E [keV]"); ax.set_ylabel("efficiency")
ax.legend(fontsize=7); ax.set_title("High-energy efficiency roll-off: paper inset vs S1c < 600 phd cut in the MC")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "P009_efficiency.png"), dpi=150); plt.close(fig)

# ----------------------------------------------------------------------------------------------------
# console summary
# ----------------------------------------------------------------------------------------------------
print("W implied by contours [eV]:", np.round(W_impl, 3), " mean", round(W_PAPER, 3))
print("paper Nq(E) fit: alpha = %.2f, beta = %.3f (W = %.2f eV)" % (ALPHA_PAPER, BETA_PAPER, W_PAPER))
print(EST[["variant", "E_S1", "E_S2", "E_comb", "S1c_at_248", "log10S2c_at_248", "S1c_at_270", "Nq_at_250"]].round(3).to_string())
print("alpha E^beta inversions:", RES["E_alpha_beta"])
print("paper-contour interpolation:", RES["E_from_paper_contours"])
for v in VARIANTS:
    print(v, {k: round(x, 3) for k, x in ML[v].items()})
print("SYS", json.dumps(SYS, indent=0, default=lambda x: round(float(x), 2)))
print("BAND", json.dumps(BAND, indent=0, default=lambda x: round(float(x), 4)))
print("DIGBAND", {k: round(x, 4) for k, x in DIGBAND.items()})
print("EFF", EFF, ML_eff_paper)
print("DELTA", json.dumps(DELTA, indent=0, default=lambda x: round(float(x), 1)))
print("EEE", {k: round(x, 4) for k, x in EEE.items()})
