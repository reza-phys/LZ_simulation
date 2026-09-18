"""
P043_gain_systematics.py -- propagating the LZ gain (g1, g2), NR-yield-scale, W and drift-field
uncertainties to the recoil energy of the 248 keV event, its NR-band position, the efficiency,
delta_max, the inelastic likelihood peak and the fitted couplings.

Run from the simulation root:   .venv/bin/python output/code/P043_gain_systematics.py

Inputs (paper, arXiv:2609.02823): g1 = 0.110 +- 0.002 phd/photon, g2 = 34.5 +- 1.1 phd/electron
(detector paragraph); S1c = 540.1 phd, S2c = 9268 phd, E = 248 +- 23 (stat) +- 23 (sys) keV, 1.5 sigma
below the NR median (Results); Table S5 NR yield parameters incl. the p(E) break (supplement).
Corpus: P009 paper-contour energy scale Nq = alpha_p E^beta_p (alpha_p = 11.32, beta_p = 1.112), P024 band
width 0.0313 dex and baseline 1.54 sigma, P038 E50 = 271.6 keV / erf sigma 10.9 keV, P012 L10 spectra,
P038 full-window inelastic O1 spectra, P021 peak-vs-E_obs rule.

Sections
 1. yield model on a (E, field) grid; three energy estimators (S1-only, combined quanta, 2D-Gaussian ML proxy)
    -> linear error propagation with `uncertainties` (numerical derivatives via uncertainties.wrap)
    -> full Monte Carlo (1e5 draws) per source and all sources together; budget closure vs +-23 keV
 2. NR-band position: sigma-distance of the event below the NR median as a function of the same nuisances
    plus the band-width uncertainty; P(> 2 sigma), P(< 1 sigma)
 3. efficiency at the event energy: naive (fixed E = 248) vs correlated (E_ML and E50 move together);
    posterior-averaged acceptance correction 1/eps
 4. kinematics: delta_max(E) for 400/1000/4000 GeV (June v_max) with stat and sys errors; P021 peak rule
 5. couplings: L10 d10 (P012) and inelastic kappa (P021/P038) fractional systematics from the acceptance
 6. table of corpus numbers with systematics; two figures
"""
from __future__ import annotations
import os, sys, json, math, time
import numpy as np
import pandas as pd
from scipy import stats, optimize, special
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uncertainties as unc
from uncertainties import ufloat

sys.path.insert(0, "output/code")
from common import lzcommon as lz

OUT = "output/work/P043"; FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(43)
T0 = time.time()

# ------------------------------------------------------------------------------------------------
# 0. inputs
# ------------------------------------------------------------------------------------------------
S1C, S2C = lz.LZ["ev_S1c"], lz.LZ["ev_S2c"]              # 540.1 phd, 9268 phd  [paper]
L2 = math.log10(S2C)                                      # 3.9670
G1, G1E = lz.LZ["g1"], lz.LZ["g1_err"]                    # 0.110 +- 0.002       [paper]
G2, G2E = lz.LZ["g2"], lz.LZ["g2_err"]                    # 34.5 +- 1.1          [paper]
F0, FE = lz.DRIFT_FIELD_VCM, 5.0                          # 96.5 V/cm; +-5 V/cm assumed bracket (P024 scanned 90/100)
S0, SE_ = 1.0, 0.04                                       # NR total-quanta scale: half of Table-S5-vs-contour offset (7.7 %)
W0, WE = 13.5, 0.2                                        # eV per quantum [recalled, likely]; 13.44 (nestpy) .. 13.7 (NEST v2)
P009 = json.load(open("output/work/P009/results.json"))
ALPHA_P, BETA_P = P009["digitised"]["alpha_paper"], P009["digitised"]["beta_paper"]   # 11.3245, 1.11167
SIG_B0, SIG_BE = 0.0313, 0.004                            # P024 band width (dex) and its uncertainty (model 0.031 vs AmBe 0.036 +- 0.005)
NSIG_P024 = 1.5407                                        # P024 baseline sigma-distance (anchor)
PLATEAU, PLATEAU_E = 0.955, 0.010                         # Fig. S2 plateau (P009 reading) and assumed +-0.01
E50_P038, SIG_ERF = 271.6, 10.93                          # P038 roll-off for the 600 phd edge
SIG_S1C_248 = 26.15                                       # P038 MC sd of S1c at 248 keV (phd)
SIG_STAT = 9.84                                           # P009 statistical sigma of E_ML on the paper scale (keV)
E_PAPER, STAT_PAPER, SYS_PAPER = 248.0, 23.0, 23.0

# ------------------------------------------------------------------------------------------------
# 1a. yield model: Ne(E, F) from LZ-tuned NEST (Table S5 + p(E) break); Nq = s alpha_p E^beta_p (paper scale)
# ------------------------------------------------------------------------------------------------
EG = np.arange(120.0, 400.01, 0.5)
FG = np.array([86.5, 91.5, 96.5, 101.5, 106.5])
NE_TAB = np.empty((FG.size, EG.size))
for j, F in enumerate(FG):
    for i, E in enumerate(EG):
        NE_TAB[j, i] = lz.nest_nr_yields(E, field=F, params=lz.NEST_NR_LZ)[1]
print("yield grid built in %.1f s" % (time.time() - T0))


def Ne_of(E, F):
    """Ne(E, F) by bilinear interpolation (E and F may be arrays of equal shape or scalars)."""
    E = np.asarray(E, float); F = np.asarray(F, float)
    jf = np.clip(np.searchsorted(FG, F) - 1, 0, FG.size - 2)
    wf = (F - FG[jf]) / (FG[jf + 1] - FG[jf])
    ie = np.clip(np.searchsorted(EG, E) - 1, 0, EG.size - 2)
    we = (E - EG[ie]) / (EG[ie + 1] - EG[ie])
    lo = NE_TAB[jf, ie] * (1 - we) + NE_TAB[jf, ie + 1] * we
    hi = NE_TAB[jf + 1, ie] * (1 - we) + NE_TAB[jf + 1, ie + 1] * we
    return lo * (1 - wf) + hi * wf


def Nq_of(E, s):
    return s * ALPHA_P * np.asarray(E, float) ** BETA_P


def Nph_of(E, s, F):
    return Nq_of(E, s) - Ne_of(E, F)


def sig_s1c(E, g1, s, F):
    """S1c resolution (phd) ~ sqrt(mean S1c), anchored to P038's 26.15 phd at 248 keV."""
    return SIG_S1C_248 * np.sqrt(g1 * Nph_of(E, s, F) / (G1 * Nph_of(248.0, 1.0, F0)))


def bisect_vec(f, lo, hi, n=45):
    lo = np.full_like(f(hi) * 0.0, lo) if np.ndim(lo) == 0 else lo
    hi = np.full_like(lo, hi) if np.ndim(hi) == 0 else hi
    flo = f(lo)
    for _ in range(n):
        mid = 0.5 * (lo + hi); fm = f(mid)
        same = np.sign(fm) == np.sign(flo)
        lo = np.where(same, mid, lo); flo = np.where(same, fm, flo); hi = np.where(same, hi, mid)
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------------------------------------
# 1b. estimators (vectorised over draws)
# ------------------------------------------------------------------------------------------------
def E_S1(g1, s, F):
    g1, s, F = np.broadcast_arrays(np.asarray(g1, float), np.asarray(s, float), np.asarray(F, float))
    return bisect_vec(lambda E: g1 * Nph_of(E, s, F) - S1C, 120.0 + 0 * g1, 400.0 + 0 * g1)


def E_comb(g1, g2, s):
    nq = S1C / np.asarray(g1, float) + S2C / np.asarray(g2, float)
    return (nq / (np.asarray(s, float) * ALPHA_P)) ** (1.0 / BETA_P)


def E_ML(g1, g2, s, F, sig_b=SIG_B0, chunk=2000):
    """2D Gaussian likelihood proxy in (S1c, log10 S2c): maximise over E on a 0.5 keV grid + parabolic refinement."""
    g1, g2, s, F = np.broadcast_arrays(*[np.atleast_1d(np.asarray(x, float)) for x in (g1, g2, s, F)])
    out = np.empty(g1.size)
    Eg = EG[(EG >= 150) & (EG <= 350)]
    for a in range(0, g1.size, chunk):
        sl = slice(a, a + chunk)
        g1c, g2c, sc, Fc = g1[sl, None], g2[sl, None], s[sl, None], F[sl, None]
        Eb = Eg[None, :]
        mu1 = g1c * Nph_of(Eb, sc, Fc)
        s1 = sig_s1c(Eb, g1c, sc, Fc)
        mu2 = np.log10(g2c * Ne_of(Eb, Fc))
        chi2 = ((S1C - mu1) / s1) ** 2 + ((L2 - mu2) / sig_b) ** 2 + 2 * np.log(s1)   # -2 lnL up to const
        i = np.argmin(chi2, axis=1)
        i = np.clip(i, 1, Eg.size - 2)
        rows = np.arange(i.size)
        y0, y1, y2 = chi2[rows, i - 1], chi2[rows, i], chi2[rows, i + 1]
        denom = (y0 - 2 * y1 + y2)
        shift = np.where(denom > 0, 0.5 * (y0 - y2) / denom, 0.0)
        out[sl] = Eg[i] + shift * (Eg[1] - Eg[0])
    return out


def E_ee(g1, g2, W):
    return W * 1e-3 * (S1C / g1 + S2C / g2)


# central values
E_S1_0 = float(E_S1(G1, S0, F0)); E_CB_0 = float(E_comb(G1, G2, S0)); E_ML_0 = float(E_ML(G1, G2, S0, F0)[0])
E_EE_0 = E_ee(G1, G2, W0)
print("central: E_S1 %.2f  E_comb %.2f  E_ML %.2f  E_ee %.2f keVee" % (E_S1_0, E_CB_0, E_ML_0, E_EE_0))
print("  (P009: 248.56 / 247.16 / 245.82 ; 69.9 at W = 13.5)")

# ------------------------------------------------------------------------------------------------
# 1c. linear propagation with `uncertainties` (numerical derivatives through uncertainties.wrap)
# ------------------------------------------------------------------------------------------------
ug1 = ufloat(G1, G1E, "g1"); ug2 = ufloat(G2, G2E, "g2"); us = ufloat(S0, SE_, "Nq-scale")
uF = ufloat(F0, FE, "field"); uW = ufloat(W0, WE, "W")
wE_S1 = unc.wrap(lambda g1, s, F: float(E_S1(g1, s, F)))
wE_CB = unc.wrap(lambda g1, g2, s: float(E_comb(g1, g2, s)))
wE_ML = unc.wrap(lambda g1, g2, s, F: float(E_ML(g1, g2, s, F)[0]))
wE_EE = unc.wrap(lambda g1, g2, W: float(E_ee(g1, g2, W)))
LIN = {}
for name, u in (("E_S1", wE_S1(ug1, us, uF)), ("E_comb", wE_CB(ug1, ug2, us)), ("E_ML", wE_ML(ug1, ug2, us, uF)),
                ("E_ee", wE_EE(ug1, ug2, uW))):
    comps = {k.tag: float(v) for k, v in u.error_components().items()}
    LIN[name] = dict(value=u.n, sigma_total=u.s, **comps)
    print("linear %-7s = %.2f +- %.2f  |" % (name, u.n, u.s), "  ".join("%s %.2f" % kv for kv in comps.items()))
# elasticities (dlnE/dln x) for the record
ELAST = dict(E_S1_g1=LIN["E_S1"]["g1"] / E_S1_0 / (G1E / G1), E_S1_s=LIN["E_S1"]["Nq-scale"] / E_S1_0 / SE_,
             E_comb_g1=LIN["E_comb"]["g1"] / E_CB_0 / (G1E / G1), E_comb_g2=LIN["E_comb"]["g2"] / E_CB_0 / (G2E / G2),
             E_ML_g1=LIN["E_ML"]["g1"] / E_ML_0 / (G1E / G1), E_ML_g2=LIN["E_ML"]["g2"] / E_ML_0 / (G2E / G2),
             E_ML_s=LIN["E_ML"]["Nq-scale"] / E_ML_0 / SE_, dlnS1c_dlnE=BETA_P * Nq_of(248, 1) / Nph_of(248, 1, F0))
# analytic cross-check: dE/E = -(1/beta_eff) dg1/g1 with beta_eff = dlnS1c/dlnE
beta_eff = float(np.gradient(np.log(G1 * Nph_of(EG, 1.0, F0)), np.log(EG))[np.searchsorted(EG, E_S1_0)])
ELAST["beta_eff_S1"] = beta_eff
print("beta_eff(S1) = %.3f -> analytic g1 +-%.2f%% => -/+%.2f keV; Nq-scale +-4%% => -/+%.2f keV" %
      (beta_eff, 100 * G1E / G1, E_S1_0 * (G1E / G1) / beta_eff, E_S1_0 * SE_ / beta_eff))

# ------------------------------------------------------------------------------------------------
# 1d. Monte Carlo: 1e5 draws, per source and total
# ------------------------------------------------------------------------------------------------
N = 100_000
D = dict(g1=rng.normal(G1, G1E, N), g2=rng.normal(G2, G2E, N), s=rng.normal(S0, SE_, N),
         F=np.clip(rng.normal(F0, FE, N), FG[0] + 0.01, FG[-1] - 0.01), W=rng.normal(W0, WE, N),
         sig_b=np.clip(rng.normal(SIG_B0, SIG_BE, N), 0.015, None), plateau=rng.normal(PLATEAU, PLATEAU_E, N))
SOURCES = ["g1", "g2", "Nq-scale", "field", "W"]


def draws(active):
    """nuisance arrays with only the sources in `active` varied."""
    return dict(g1=D["g1"] if "g1" in active else np.full(N, G1), g2=D["g2"] if "g2" in active else np.full(N, G2),
                s=D["s"] if "Nq-scale" in active else np.full(N, S0), F=D["F"] if "field" in active else np.full(N, F0),
                W=D["W"] if "W" in active else np.full(N, W0))


def summarize(x):
    q = np.percentile(x, [2.275, 15.865, 50, 84.135, 97.725])
    return dict(mean=float(np.mean(x)), sd=float(np.std(x)), median=float(q[2]), lo68=float(q[1]), hi68=float(q[3]),
                lo95=float(q[0]), hi95=float(q[4]), skew=float(stats.skew(x)))


MC = {}
MC_SAMPLES = {}
for act in [[s] for s in SOURCES] + [SOURCES]:
    tag = act[0] if len(act) == 1 else "all"
    d = draws(act)
    es1 = E_S1(d["g1"], d["s"], d["F"]); ecb = E_comb(d["g1"], d["g2"], d["s"])
    eml = E_ML(d["g1"], d["g2"], d["s"], d["F"]); eee = E_ee(d["g1"], d["g2"], d["W"])
    MC[tag] = dict(E_S1=summarize(es1), E_comb=summarize(ecb), E_ML=summarize(eml), E_ee=summarize(eee))
    MC_SAMPLES[tag] = dict(E_S1=es1, E_comb=ecb, E_ML=eml, E_ee=eee)
    print("MC %-9s sd: E_S1 %.2f  E_comb %.2f  E_ML %.2f  E_ee %.2f   (E_ML mean %.2f, 68%% %.1f-%.1f)" %
          (tag, MC[tag]["E_S1"]["sd"], MC[tag]["E_comb"]["sd"], MC[tag]["E_ML"]["sd"], MC[tag]["E_ee"]["sd"],
           MC[tag]["E_ML"]["mean"], MC[tag]["E_ML"]["lo68"], MC[tag]["E_ML"]["hi68"]))
print("MC time %.1f s" % (time.time() - T0))

# budget closure: which Nq-scale sigma would reproduce the paper's +-23 keV systematic?
sys_ML_total = MC["all"]["E_ML"]["sd"]
other2 = sum(LIN["E_ML"][k] ** 2 for k in ("g1", "g2", "field"))
s_needed = math.sqrt(max(SYS_PAPER ** 2 - other2, 0.0)) / (LIN["E_ML"]["Nq-scale"] / SE_)
CLOSURE = dict(sys_total_MC_E_ML=sys_ML_total, sys_total_linear_E_ML=LIN["E_ML"]["sigma_total"], paper_sys=SYS_PAPER,
               ratio_paper_over_ours=SYS_PAPER / sys_ML_total, Nq_scale_sigma_needed=s_needed,
               Nq_scale_sigma_needed_pct=100 * s_needed,
               stat_sys_paper_quadrature=math.hypot(STAT_PAPER, SYS_PAPER), stat_sys_ours_quadrature=math.hypot(SIG_STAT, sys_ML_total),
               tableS5_vs_contour_E_ML_shift_keV=262.3 - 245.8)
print("budget: ours sigma_sys(E_ML) = %.1f keV vs paper 23 keV; Nq-scale sigma needed to close = %.1f%%" %
      (sys_ML_total, 100 * s_needed))

# ------------------------------------------------------------------------------------------------
# 2. band position: sigma-distance of the event below the NR median at S1c = 540.1
# ------------------------------------------------------------------------------------------------
def nsig_raw(g1, g2, s, F, sig_b):
    Es = E_S1(g1, s, F)                         # energy of the NR median at the event's S1c
    med = np.log10(g2 * Ne_of(Es, F))           # median log10 S2c of the NR band there
    return (med - L2) / sig_b


NS0 = float(nsig_raw(G1, G2, S0, F0, SIG_B0))
ANCHOR = NSIG_P024 - NS0                         # P024 full-MC baseline minus our mean-yield estimate (skew/median offset)
print("raw sigma-distance %.3f (P024 MC 1.541; anchor offset %.3f applied to shifts)" % (NS0, ANCHOR))
BAND = {}
BAND_SAMPLES = {}
band_sources = SOURCES[:4] + ["band-width"]
for act in [[s] for s in band_sources] + [band_sources]:
    tag = act[0] if len(act) == 1 else "all"
    d = draws(act); sb = D["sig_b"] if "band-width" in act else np.full(N, SIG_B0)
    ns = nsig_raw(d["g1"], d["g2"], d["s"], d["F"], sb) + ANCHOR
    BAND[tag] = summarize(ns); BAND[tag].update(P_gt2=float(np.mean(ns > 2)), P_lt1=float(np.mean(ns < 1)),
                                                P_gt2p5=float(np.mean(ns > 2.5)))
    BAND_SAMPLES[tag] = ns
    print("band %-10s sigma-distance %.3f +- %.3f  68%% [%.2f, %.2f]  P(>2)=%.3f P(<1)=%.3f" %
          (tag, BAND[tag]["mean"], BAND[tag]["sd"], BAND[tag]["lo68"], BAND[tag]["hi68"], BAND[tag]["P_gt2"], BAND[tag]["P_lt1"]))
# one-sided +-1 sigma table for the record (g2 +-3.2 %, etc.)
BAND_1SIG = {}
for tag, kw in (("g2-", dict(g2=G2 - G2E)), ("g2+", dict(g2=G2 + G2E)), ("g1-", dict(g1=G1 - G1E)), ("g1+", dict(g1=G1 + G1E)),
                ("s-", dict(s=S0 - SE_)), ("s+", dict(s=S0 + SE_)), ("F-", dict(F=F0 - FE)), ("F+", dict(F=F0 + FE)),
                ("Table S5 scale (s=1/1.077)", dict(s=1 / 1.077)), ("sig_b 0.036", dict(sig_b=0.036)), ("sig_b 0.0212 drawn", dict(sig_b=0.0212))):
    a = dict(g1=G1, g2=G2, s=S0, F=F0, sig_b=SIG_B0); a.update(kw)
    BAND_1SIG[tag] = float(nsig_raw(**a)) + ANCHOR
print("band one-at-a-time:", {k: round(v, 3) for k, v in BAND_1SIG.items()})
# P(<= event | NR) implied
BAND["P_below_event_central"] = float(stats.norm.sf(NSIG_P024))
BAND["P_below_event_marginal"] = float(np.mean(stats.norm.sf(BAND_SAMPLES["all"])))

# ------------------------------------------------------------------------------------------------
# 3. efficiency at the event energy
# ------------------------------------------------------------------------------------------------
def eps(E, g1, s, F, plateau=PLATEAU):
    """epsilon(E) = plateau x P(S1c < 600 | E) with Gaussian S1c of sd sig_s1c (S2c ceiling irrelevant here, P038)."""
    mu = g1 * Nph_of(E, s, F)
    return plateau * stats.norm.cdf((600.0 - mu) / sig_s1c(E, g1, s, F))


def E50(g1, s, F):
    return bisect_vec(lambda E: g1 * Nph_of(E, s, F) - 600.0, 120.0 + 0 * np.asarray(g1, float), 400.0 + 0 * np.asarray(g1, float))


EFF = {}
EFF["E50_central"] = float(E50(G1, S0, F0))
EFF["E50_g1pm"] = [float(E50(G1 - G1E, S0, F0)), float(E50(G1 + G1E, S0, F0))]
EFF["E50_spm"] = [float(E50(G1, S0 - SE_, F0)), float(E50(G1, S0 + SE_, F0))]
EFF["E50_Fpm"] = [float(E50(G1, S0, F0 - FE)), float(E50(G1, S0, F0 + FE))]
EFF["E50_tableS5_scale"] = float(E50(G1, 1 / 1.077, F0))
print("E50 = %.1f keV (P038 271.6); g1 -/+: %s ; s -/+: %s ; F -/+: %s" % (EFF["E50_central"], np.round(EFF["E50_g1pm"], 1),
      np.round(EFF["E50_spm"], 1), np.round(EFF["E50_Fpm"], 1)))
# (a) naive: E fixed at 248 keV, curve moves
EFF["naive_248_central"] = float(eps(248.0, G1, S0, F0))
EFF["naive_248_g1pm"] = [float(eps(248.0, G1 - G1E, S0, F0)), float(eps(248.0, G1 + G1E, S0, F0))]
EFF["naive_248_spm"] = [float(eps(248.0, G1, S0 - SE_, F0)), float(eps(248.0, G1, S0 + SE_, F0))]
d = draws(SOURCES)
eps_naive = eps(248.0, d["g1"], d["s"], d["F"], D["plateau"])
EFF["naive_248_MC"] = summarize(eps_naive)
# (b) correlated: evaluate at the reconstructed energy of the same draw
eml_all = MC_SAMPLES["all"]["E_ML"]
eps_corr = eps(eml_all, d["g1"], d["s"], d["F"], D["plateau"])
EFF["at_EML_central"] = float(eps(E_ML_0, G1, S0, F0))
EFF["at_EML_MC"] = summarize(eps_corr)
EFF["inv_eps_at_EML_MC"] = summarize(1 / eps_corr)
# (c) posterior-averaged acceptance correction: E_true ~ N(E_ML, sig_stat) within each draw
Et = eml_all + rng.normal(0, SIG_STAT, N)
eps_post = eps(Et, d["g1"], d["s"], d["F"], D["plateau"])
EFF["inv_eps_posterior_MC"] = summarize(1 / eps_post)
EFF["eps_posterior_MC"] = summarize(eps_post)
# stat-only version (no systematics)
Et0 = E_ML_0 + rng.normal(0, SIG_STAT, N)
EFF["inv_eps_posterior_stat_only"] = summarize(1 / eps(Et0, G1, S0, F0))
EFF["eps_262"] = float(eps(262.0, G1, S0, F0)); EFF["eps_248"] = float(eps(248.0, G1, S0, F0)); EFF["eps_246"] = float(eps(246.0, G1, S0, F0))
print("eps(246/248/262) = %.3f/%.3f/%.3f (P038 0.950/0.945/0.792)" % (EFF["eps_246"], EFF["eps_248"], EFF["eps_262"]))
print("naive eps(248): %.3f, g1 -/+ %s, MC sd %.3f | correlated eps(E_ML): %.3f, MC sd %.3f | <1/eps> posterior %.3f +- %.3f" %
      (EFF["naive_248_central"], np.round(EFF["naive_248_g1pm"], 3), EFF["naive_248_MC"]["sd"], EFF["at_EML_central"],
       EFF["at_EML_MC"]["sd"], EFF["inv_eps_posterior_MC"]["mean"], EFF["inv_eps_posterior_MC"]["sd"]))

# ------------------------------------------------------------------------------------------------
# 4. kinematics: delta_max and the P021 peak
# ------------------------------------------------------------------------------------------------
VMAX_JUNE = lz.vmax_kms(lz.v_earth_kms(167))            # 809.1 km/s (16 June = day 167)
KIN = dict(vmax_june_kms=VMAX_JUNE)
for m in (400, 1000, 4000):
    dm = lambda E: lz.delta_max_kev(E, m, v_kms=VMAX_JUNE)
    d0 = dm(E_PAPER); slope = (dm(E_PAPER + 1) - dm(E_PAPER - 1)) / 2
    dm_ml = np.array([dm(e) for e in eml_all[:20000]])
    dm_tot = np.array([dm(e) for e in (eml_all[:20000] + rng.normal(0, SIG_STAT, 20000))])
    KIN[str(m)] = dict(delta_max_248=d0, delta_max_246=dm(E_ML_0), slope_keV_per_keV=slope,
                       sigma_stat=abs(slope) * SIG_STAT, sigma_sys=float(np.std(dm_ml)), sigma_tot=float(np.std(dm_tot)),
                       delta_max_at_pm_sys=[dm(E_ML_0 - sys_ML_total), dm(E_ML_0 + sys_ML_total)],
                       delta_max_at_paper_pm32=[dm(E_PAPER - 32.5), dm(E_PAPER + 32.5)],
                       mc_mean=float(np.mean(dm_tot)), mc_lo68=float(np.percentile(dm_tot, 15.865)), mc_hi68=float(np.percentile(dm_tot, 84.135)))
    print("delta_max(m=%d): %.1f keV at 248, slope %.3f keV/keV, sigma stat %.1f sys %.1f tot %.1f" %
          (m, d0, slope, KIN[str(m)]["sigma_stat"], KIN[str(m)]["sigma_sys"], KIN[str(m)]["sigma_tot"]))
# P021 rule of thumb: peak delta vs E_obs, (246 -> 380), (262 -> 385) at 1000 GeV isoscalar; 5 keV grid
peak_slope = (385.0 - 380.0) / (262.0 - 246.0)
KIN["P021_peak"] = dict(slope_keV_per_keV=peak_slope, grid_step_keV=5.0, sigma_sys=peak_slope * sys_ML_total,
                        sigma_stat=peak_slope * SIG_STAT, sigma_tot=peak_slope * math.hypot(SIG_STAT, sys_ML_total),
                        note="P021 grid is 5 keV, so the slope is coarse (0.2-0.5 keV/keV bracket); Higgsino delta(N=1) = 366 keV (P007) is rate-set and independent of E_obs")
print("P021 peak: slope %.2f -> sigma_sys %.1f, sigma_tot %.1f keV" % (peak_slope, KIN["P021_peak"]["sigma_sys"], KIN["P021_peak"]["sigma_tot"]))

# ------------------------------------------------------------------------------------------------
# 5. couplings: acceptance sensitivity to the S1c edge (E50) and hence to g1 / Nq-scale
# ------------------------------------------------------------------------------------------------
def eff_curve(E, e50, plateau=PLATEAU, sig=SIG_ERF):
    lo = 0.5 * (1 + special.erf((E - 5.4) / (math.sqrt(2) * 2.5)))
    return plateau * lo * 0.5 * (1 + special.erf((e50 - E) / (math.sqrt(2) * sig)))


L10 = np.load("output/work/P012/P012_L10_spectra_d10_1.npz")
INEL = np.load("output/work/P038/spectra_s_1000_full.npz")
COUP = {}
E50s = dict(central=EFF["E50_central"], g1m=EFF["E50_g1pm"][0], g1p=EFF["E50_g1pm"][1], sm=EFF["E50_spm"][0], sp=EFF["E50_spm"][1],
            tabS5=EFF["E50_tableS5_scale"], P038=E50_P038)
for m in (200, 1000, 4000):
    E = L10["E"]; r = L10["m%d" % m]
    Nu = {k: float(np.trapezoid(r * eff_curve(E, e), E)) for k, e in E50s.items()}
    COUP["L10_m%d" % m] = dict(N_unit_rel={k: v / Nu["central"] for k, v in Nu.items()},
                              d10_rel={k: math.sqrt(Nu["central"] / v) for k, v in Nu.items()},
                              frac_rate_above_250=float(np.trapezoid(r[E > 250], E[E > 250]) / np.trapezoid(r, E)))
    print("L10 m=%d: N_unit rel g1-/+ %.4f/%.4f, s-/+ %.4f/%.4f -> d10 rel g1 %.4f/%.4f, s %.4f/%.4f" %
          (m, COUP["L10_m%d" % m]["N_unit_rel"]["g1m"], COUP["L10_m%d" % m]["N_unit_rel"]["g1p"],
           COUP["L10_m%d" % m]["N_unit_rel"]["sm"], COUP["L10_m%d" % m]["N_unit_rel"]["sp"],
           COUP["L10_m%d" % m]["d10_rel"]["g1m"], COUP["L10_m%d" % m]["d10_rel"]["g1p"],
           COUP["L10_m%d" % m]["d10_rel"]["sm"], COUP["L10_m%d" % m]["d10_rel"]["sp"]))
# d10 with systematics (P012: 0.279 at 1000 GeV, stat 0.15-0.43 for 0.3-2.4 events)
d10_0 = 0.279
rel = COUP["L10_m1000"]["d10_rel"]
COUP["d10_1000"] = dict(central=d10_0, g1_pm=[d10_0 * rel["g1m"], d10_0 * rel["g1p"]], s_pm=[d10_0 * rel["sm"], d10_0 * rel["sp"]],
                        frac_sys=math.hypot(0.5 * abs(rel["g1m"] - rel["g1p"]), 0.5 * abs(rel["sm"] - rel["sp"])),
                        stat_interval=[0.15, 0.43])
# inelastic: acceptance A(E50) for 1000 GeV isoscalar O1 (annual halo), kappa ∝ 1/A
Ei = INEL["E"]; dl = INEL["deltas"]
for dd in (300.0, 350.0, 370.0, 380.0):
    r = INEL["annual"][np.argmin(np.abs(dl - dd))]
    tot = float(np.trapezoid(r, Ei))
    A = {k: float(np.trapezoid(r * eff_curve(Ei, e, plateau=1.0), Ei) / tot) for k, e in E50s.items()}
    COUP["inel_d%d" % dd] = dict(A=A, kappa_rel={k: A["central"] / v for k, v in A.items()},
                                 kappa_rel_g1_pm=[A["central"] / A["g1m"], A["central"] / A["g1p"]],
                                 kappa_rel_s_pm=[A["central"] / A["sm"], A["central"] / A["sp"]],
                                 A_P038=[0.81, 0.44, 0.12, 0.029][[300., 350., 370., 380.].index(dd)])
    print("inelastic delta=%d: A_600 = %.3f (P038 %.3f); g1 -/+ -> kappa x %.2f/%.2f; s -/+ -> x %.2f/%.2f; Table-S5 scale x %.2f" %
          (dd, A["central"], COUP["inel_d%d" % dd]["A_P038"], A["central"] / A["g1m"], A["central"] / A["g1p"],
           A["central"] / A["sm"], A["central"] / A["sp"], A["central"] / A["tabS5"]))

# ------------------------------------------------------------------------------------------------
# 6. corpus table with systematics
# ------------------------------------------------------------------------------------------------
kap380 = COUP["inel_d380"]
TABLE = [
    dict(quantity="E_R (2D-ML, paper scale) [keV]", central=round(E_ML_0, 1), stat=SIG_STAT, sys=round(sys_ML_total, 1),
         sys_breakdown="g1 %.1f, g2 %.1f, Nq-scale %.1f, field %.1f" % tuple(LIN["E_ML"][k] for k in ("g1", "g2", "Nq-scale", "field")),
         note="paper: 248 +- 23 +- 23; our sys is the +-4%% yield-scale proxy; %.0f%% Nq-scale would be needed for 23 keV" % (100 * s_needed)),
    dict(quantity="E_S1 (S1-only) [keV]", central=round(E_S1_0, 1), stat=10.4, sys=round(MC["all"]["E_S1"]["sd"], 1),
         sys_breakdown="g1 %.1f, Nq-scale %.1f, field %.1f" % tuple(LIN["E_S1"][k] for k in ("g1", "Nq-scale", "field")), note=""),
    dict(quantity="E_ee (keVee, W(S1c/g1+S2c/g2))", central=round(E_EE_0, 1), stat=1.2, sys=round(MC["all"]["E_ee"]["sd"], 1),
         sys_breakdown="g1 %.1f, g2 %.1f, W %.1f" % tuple(LIN["E_ee"][k] for k in ("g1", "g2", "W")), note="stat = Nq +-90 (P010)"),
    dict(quantity="sigma-distance below NR median", central=round(NSIG_P024, 2), stat="n/a", sys=round(BAND["all"]["sd"], 2),
         sys_breakdown="g2 %.2f, band-width %.2f, Nq-scale %.2f, g1 %.2f, field %.2f" % tuple(BAND[k]["sd"] for k in ("g2", "band-width", "Nq-scale", "g1", "field")),
         note="P(>2 sigma) = %.2f, P(<1 sigma) = %.2f" % (BAND["all"]["P_gt2"], BAND["all"]["P_lt1"])),
    dict(quantity="epsilon at reconstructed energy", central=round(EFF["at_EML_central"], 3), stat="n/a", sys=round(EFF["at_EML_MC"]["sd"], 3),
         sys_breakdown="plateau 0.010 dominates; g1/Nq-scale cancel (edge and event share S1c)", note="naive eps(248) with moving curve: %.3f-%.3f" % tuple(sorted(EFF["naive_248_g1pm"]))),
    dict(quantity="1/epsilon acceptance correction", central=round(EFF["inv_eps_posterior_MC"]["mean"], 3), stat=round(EFF["inv_eps_posterior_stat_only"]["sd"], 3),
         sys=round(math.sqrt(max(EFF["inv_eps_posterior_MC"]["sd"] ** 2 - EFF["inv_eps_posterior_stat_only"]["sd"] ** 2, 0)), 3),
         sys_breakdown="posterior-averaged over E_true ~ N(E_ML, 9.8 keV)", note="1/eps(262) = %.2f" % (1 / EFF["eps_262"])),
    dict(quantity="delta_max(1 TeV, June) [keV]", central=round(KIN["1000"]["delta_max_248"], 1), stat=round(KIN["1000"]["sigma_stat"], 1),
         sys=round(KIN["1000"]["sigma_sys"], 1), sys_breakdown="slope %.3f keV/keV x sigma_E" % KIN["1000"]["slope_keV_per_keV"],
         note="400 GeV: %.1f +- %.1f +- %.1f; 4000 GeV: %.1f +- %.1f +- %.1f" % (KIN["400"]["delta_max_248"], KIN["400"]["sigma_stat"], KIN["400"]["sigma_sys"],
                                                                           KIN["4000"]["delta_max_248"], KIN["4000"]["sigma_stat"], KIN["4000"]["sigma_sys"])),
    dict(quantity="P021 likelihood peak delta (1 TeV) [keV]", central=380, stat=round(KIN["P021_peak"]["sigma_stat"], 1), sys=round(KIN["P021_peak"]["sigma_sys"], 1),
         sys_breakdown="0.31 keV/keV x sigma_E (5 keV grid)", note="Higgsino delta(N=1)=366 keV unaffected (rate-set)"),
    dict(quantity="d10^s (1 TeV, L10)", central=d10_0, stat="0.15-0.43", sys=round(d10_0 * COUP["d10_1000"]["frac_sys"], 3),
         sys_breakdown="g1 %.1f%%, Nq-scale %.1f%% (via acceptance)" % (100 * 0.5 * abs(rel["g1m"] - rel["g1p"]), 100 * 0.5 * abs(rel["sm"] - rel["sp"])), note="convention factor 2 (DR-001) separate"),
    dict(quantity="kappa(delta=380, 1 TeV) = 2.19", central=2.19, stat="0.66-5.2", sys="x/÷ %.2f (g1), x/÷ %.2f (Nq-scale)" % (
         max(kap380["kappa_rel_g1_pm"]) if max(kap380["kappa_rel_g1_pm"]) > 1 else 1 / min(kap380["kappa_rel_g1_pm"]),
         max(kap380["kappa_rel_s_pm"]) if max(kap380["kappa_rel_s_pm"]) > 1 else 1 / min(kap380["kappa_rel_s_pm"])),
         sys_breakdown="A_600(380) = %.3f moves to %.3f-%.3f (g1), %.3f-%.3f (Nq-scale)" % (kap380["A"]["central"], kap380["A"]["g1m"], kap380["A"]["g1p"], kap380["A"]["sm"], kap380["A"]["sp"]),
         note="delta=300: x/÷ %.2f (g1); delta=350: x/÷ %.2f" % (max(COUP["inel_d300"]["kappa_rel_g1_pm"]), max(COUP["inel_d350"]["kappa_rel_g1_pm"]))),
]
pd.DataFrame(TABLE).to_csv(os.path.join(OUT, "corpus_numbers_with_systematics.csv"), index=False)

# budget CSV
rows = []
for src in SOURCES + ["all"]:
    rows.append(dict(source=src, sigma_E_S1_MC=MC[src]["E_S1"]["sd"], sigma_E_comb_MC=MC[src]["E_comb"]["sd"], sigma_E_ML_MC=MC[src]["E_ML"]["sd"],
                     sigma_E_ee_MC=MC[src]["E_ee"]["sd"],
                     sigma_E_S1_lin=LIN["E_S1"].get(src, 0.0) if src != "all" else LIN["E_S1"]["sigma_total"],
                     sigma_E_comb_lin=LIN["E_comb"].get(src, 0.0) if src != "all" else LIN["E_comb"]["sigma_total"],
                     sigma_E_ML_lin=LIN["E_ML"].get(src, 0.0) if src != "all" else LIN["E_ML"]["sigma_total"],
                     sigma_nsig_band=BAND.get(src, {}).get("sd", np.nan) if src != "all" else BAND["all"]["sd"]))
BUDGET = pd.DataFrame(rows); BUDGET.to_csv(os.path.join(OUT, "systematic_budget.csv"), index=False)
print(BUDGET.round(3).to_string())

RES = dict(inputs=dict(S1c=S1C, S2c=S2C, g1=[G1, G1E], g2=[G2, G2E], Nq_scale=[S0, SE_], field=[F0, FE], W=[W0, WE],
                       alpha_paper=ALPHA_P, beta_paper=BETA_P, sig_band=[SIG_B0, SIG_BE], nsig_P024=NSIG_P024, plateau=[PLATEAU, PLATEAU_E],
                       sig_S1c_248=SIG_S1C_248, sig_stat=SIG_STAT, N_MC=N, seed=43, E_grid=[120, 400, 0.5], F_grid=FG.tolist()),
           central=dict(E_S1=E_S1_0, E_comb=E_CB_0, E_ML=E_ML_0, E_ee=E_EE_0), linear=LIN, elasticities=ELAST, MC=MC, closure=CLOSURE,
           band=BAND, band_one_at_a_time=BAND_1SIG, band_anchor_offset=ANCHOR, band_raw_central=NS0, efficiency=EFF, kinematics=KIN, couplings=COUP,
           table=TABLE, runtime_s=time.time() - T0)
json.dump(RES, open(os.path.join(OUT, "P043_results.json"), "w"), indent=1, default=float)

# ------------------------------------------------------------------------------------------------
# figures (Okabe-Ito colour-blind-safe palette; neutral ink for text)
# ------------------------------------------------------------------------------------------------
OI = dict(orange="#E69F00", sky="#56B4E9", green="#009E73", blue="#0072B2", verm="#D55E00", purple="#CC79A7", grey="#8a8a8a")
INK, MUTED = "#222222", "#666666"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False})

# Fig 1: systematic budget, four panels (E_ML, band sigma-distance, delta_max(1 TeV), kappa(380)/d10), horizontal bars
fig, axs = plt.subplots(1, 4, figsize=(13.5, 3.7))
src_lab = ["g1 ±2%", "g2 ±3%", "Nq-scale ±4%", "field ±5 V/cm", "W ±0.2 eV", "band width ±13%"]
def barh(ax, vals, labels, title, xlabel, ref=None, reflab=None, total=None):
    y = np.arange(len(vals))
    ax.barh(y, vals, height=0.55, color=OI["blue"], edgecolor="white", linewidth=2)
    if total is not None:
        ax.barh(len(vals), total, height=0.55, color=OI["orange"], edgecolor="white", linewidth=2)
        labels = labels + ["total (MC)"]; y = np.arange(len(vals) + 1)
        ax.text(total * 1.02, len(vals), "%.2g" % total, va="center", color=INK, fontsize=8)
    for yi, v in zip(y, vals):
        ax.text(v * 1.02 + 1e-9, yi, "%.2g" % v, va="center", color=INK, fontsize=8)
    if ref is not None:
        ax.axvline(ref, color=OI["verm"], lw=1.5, ls="--"); ax.text(ref, len(labels) - 0.4, reflab, color=OI["verm"], fontsize=8, ha="right", va="bottom")
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis(); ax.set_title(title, color=INK, fontsize=9.5, loc="left")
    ax.set_xlabel(xlabel); ax.grid(axis="x", color="#dddddd", lw=0.6); ax.set_axisbelow(True)
barh(axs[0], [MC[s]["E_ML"]["sd"] for s in SOURCES], src_lab[:5], "σ_sys on E_R (2D-ML proxy)", "keV", ref=SYS_PAPER, reflab="paper ±23", total=MC["all"]["E_ML"]["sd"])
axs[0].set_xlim(0, 26)
barh(axs[1], [BAND[s]["sd"] for s in ["g1", "g2", "Nq-scale", "field", "band-width"]], [src_lab[i] for i in (0, 1, 2, 3, 5)],
     "σ on band distance (1.54σ)", "σ units", total=BAND["all"]["sd"])
barh(axs[2], [np.std([lz.delta_max_kev(e, 1000, v_kms=VMAX_JUNE) for e in MC_SAMPLES[s]["E_ML"][:5000]]) for s in SOURCES], src_lab[:5],
     "σ on δ_max(1 TeV, June)", "keV", ref=KIN["1000"]["sigma_stat"], reflab="stat", total=KIN["1000"]["sigma_sys"])
kk = [100 * 0.5 * abs(COUP["inel_d%d" % d]["kappa_rel_g1_pm"][0] - COUP["inel_d%d" % d]["kappa_rel_g1_pm"][1]) for d in (300, 350, 370, 380)]
kk_s = [100 * 0.5 * abs(COUP["inel_d%d" % d]["kappa_rel_s_pm"][0] - COUP["inel_d%d" % d]["kappa_rel_s_pm"][1]) for d in (300, 350, 370, 380)]
y = np.arange(5)
axs[3].barh(y - 0.18, [100 * 0.5 * abs(rel["g1m"] - rel["g1p"])] + kk, height=0.34, color=OI["blue"], edgecolor="white", linewidth=2, label="g1 ±2%")
axs[3].barh(y + 0.18, [100 * 0.5 * abs(rel["sm"] - rel["sp"])] + kk_s, height=0.34, color=OI["green"], edgecolor="white", linewidth=2, label="Nq-scale ±4%")
axs[3].set_yticks(y); axs[3].set_yticklabels(["d10 (L10, 1 TeV)", "κ δ=300", "κ δ=350", "κ δ=370", "κ δ=380"]); axs[3].invert_yaxis()
axs[3].set_xscale("log"); axs[3].set_xlim(0.05, 300); axs[3].set_xlabel("half-range of coupling shift [%]"); axs[3].set_title("couplings (via acceptance)", color=INK, fontsize=9.5, loc="left")
axs[3].legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.02, 0.62)); axs[3].grid(axis="x", color="#dddddd", lw=0.6); axs[3].set_axisbelow(True)
for yi, v in zip(y, [100 * 0.5 * abs(rel["g1m"] - rel["g1p"])] + kk):
    axs[3].text(v * 1.1, yi - 0.18, "%.2g" % v, va="center", color=INK, fontsize=7.5)
fig.suptitle("P043: systematic budget of the LZ 248 keV event (g1 ±2 %, g2 ±3 %, NR-yield scale ±4 %, field, W, band width)", color=INK, fontsize=10, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(os.path.join(FIG, "P043_fig1_budget.png"), dpi=170); plt.close(fig)

# Fig 2: E_R distributions: sys-only (MC all), stat-only, stat+sys, paper's 248 +- 23 +- 23
fig, ax = plt.subplots(figsize=(7.2, 4.0))
xs = np.linspace(190, 310, 600)
E_sys = MC_SAMPLES["all"]["E_ML"]; E_tot = E_sys + rng.normal(0, SIG_STAT, N)
ax.hist(E_sys, bins=120, range=(190, 310), density=True, histtype="stepfilled", color=OI["sky"], alpha=0.45, label="systematics only (MC, σ = %.1f keV)" % np.std(E_sys))
ax.hist(E_tot, bins=120, range=(190, 310), density=True, histtype="step", color=OI["blue"], lw=2, label="stat ⊕ sys (σ = %.1f keV)" % np.std(E_tot))
ax.plot(xs, stats.norm.pdf(xs, E_ML_0, SIG_STAT), color=OI["green"], lw=2, ls="--", label="statistical only (P009, σ = 9.8 keV)")
ax.plot(xs, stats.norm.pdf(xs, E_PAPER, math.hypot(STAT_PAPER, SYS_PAPER)), color=OI["verm"], lw=2, label="paper 248 ± 23 ± 23 (σ = 32.5 keV)")
ax.axvline(269.9, color=MUTED, lw=1, ls=":"); ax.text(270.5, ax.get_ylim()[1] * 0.92, "50 % efficiency\n269.9 keV", color=MUTED, fontsize=8)
ax.axvline(262.3, color=OI["purple"], lw=1, ls=":"); ax.text(262.8, ax.get_ylim()[1] * 0.70, "Table S5\nas printed", color=OI["purple"], fontsize=8)
ax.set_xlabel("reconstructed recoil energy E_R [keV]"); ax.set_ylabel("probability density [keV$^{-1}$]")
ax.set_title("P043: energy of the LZ event with propagated gain, yield-scale, field and W uncertainties", color=INK, fontsize=9.5, loc="left")
ax.legend(frameon=False, fontsize=8, loc="upper left"); ax.grid(color="#e5e5e5", lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "P043_fig2_energy_distribution.png"), dpi=170); plt.close(fig)

# Fig 3: band sigma-distance distribution
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ns = BAND_SAMPLES["all"]
ax.hist(ns, bins=120, range=(0.4, 2.8), density=True, histtype="stepfilled", color=OI["sky"], alpha=0.5, label="all sources (σ = %.2f)" % np.std(ns))
ax.hist(BAND_SAMPLES["g2"], bins=120, range=(0.4, 2.8), density=True, histtype="step", color=OI["blue"], lw=1.8, label="g2 only (σ = %.2f)" % BAND["g2"]["sd"])
ax.hist(BAND_SAMPLES["band-width"], bins=120, range=(0.4, 2.8), density=True, histtype="step", color=OI["green"], lw=1.8, label="band width only (σ = %.2f)" % BAND["band-width"]["sd"])
ax.axvline(1.5, color=OI["verm"], lw=1.5, ls="--"); ax.text(1.52, ax.get_ylim()[1] * 0.9, "paper 1.5σ", color=OI["verm"], fontsize=8)
for xv, lab in ((1.0, "1σ"), (2.0, "2σ")):
    ax.axvline(xv, color=MUTED, lw=0.8, ls=":"); ax.text(xv + 0.02, ax.get_ylim()[1] * 0.6, lab, color=MUTED, fontsize=8)
ax.set_xlabel("event distance below the NR median [σ]"); ax.set_ylabel("density")
ax.set_title("P043: NR-band position with systematics  (P(>2σ) = %.2f, P(<1σ) = %.2f)" % (BAND["all"]["P_gt2"], BAND["all"]["P_lt1"]), color=INK, fontsize=9.5, loc="left")
ax.legend(frameon=False, fontsize=8); ax.grid(color="#e5e5e5", lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "P043_fig3_band_distance.png"), dpi=170); plt.close(fig)
print("done in %.1f s" % (time.time() - T0))
