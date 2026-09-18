#!/usr/bin/env python
"""P099 -- Decisive tests and a falsifiable forecast for LZ's next release.

Synthesis paper. No physics is recomputed here: every rate, acceptance and
background is taken from a corpus paper (cited in the RATES/BKG dictionaries
below). This script recomputes only the *counting and decision statistics*
needed to state the forecast crisply (Poisson, Gamma-Poisson, mixture
posteriors, exact-Poisson discovery probabilities, null-branch dates), and
validates each against the number the source paper quotes.

Run from the simulation root:
    .venv/bin/python output/code/P099_decisive_tests.py
Outputs: output/work/P099/tables/*.csv, output/work/P099/P099_results.json,
         output/work/P099/figures/P099_fig1_forecast.png
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import optimize, stats

sys.path.insert(0, "output/code")
from common import lzcommon as lz  # noqa: E402  (LZ constants only)

OUT = "output/work/P099"
os.makedirs(f"{OUT}/tables", exist_ok=True)
os.makedirs(f"{OUT}/figures", exist_ok=True)

E_LZ = float(lz.LZ.get("exposure_tyr", 2.84))  # LZ WS2024 exposure, t.yr
NOW = 2026.71  # 16 Sep 2026 (simulated date of this paper)

# ----------------------------------------------------------------------------
# 1. Corpus inputs (all cited)
# ----------------------------------------------------------------------------
# Best-fit rates per t.yr in 200-270 keV (600 phd edge) and 200-400 keV
# (1000 phd edge), LZ's whole-ROI best fit normalised to one event in 2.84 t.yr
# (P069 normalisation table, reproduced by P088 Sec. 2.2; P050 <= 0.3 %).
RATES_BAND = {"L10": 0.121, "d300": 0.063, "d350": 0.178, "d366": 0.274, "d380": 0.278}
RATES_400 = {"L10": 0.220, "d300": 0.112, "d350": 0.516, "d366": 1.63, "d380": 10.5}
# P079/P020 anchoring: one *band* event per 2.84 t.yr (2.7x the ROI anchoring
# for L10; P079 caveat). Used as the upper bracket of the plug-in forecast.
RATE_BAND_ANCHORED = 1.0 / E_LZ  # per t.yr
# Corpus posterior for the 200-270 keV rate per 2.84 t.yr (P081): mean 0.385,
# median 0.19, 68 % 0.036-0.69.
P081_MEAN, P081_MEDIAN, P081_68 = 0.385, 0.19, (0.036, 0.69)
# Backgrounds per 2.84 t.yr: NR-band 200-270 keV 5.7e-4 (P016, adopted by
# P069/P081/P088); 270-420 keV extension 0.0028 (P038 NR-band MSSI at 1000 phd);
# 55-200 keV 0.12 (P081: 0.10 estimate for 55-125 + 0.02 P016 for 125-200);
# corpus-revised neighbourhood background 9.4e-4 (P093, artefact class 64 %).
BKG = {"band": 5.7e-4, "ext": 0.0028, "low": 0.12, "corpus_nb": 9.4e-4}
# Hypothesis weights (P061 community-prior medians as propagated by P081):
PI_DM, PI_B, PI_U = 0.015, 0.148, 0.789
F_TRANS = 0.5  # P061/P081: half of the unknown's mass is a one-off transient
# Exposure ledger (P069 Table 1 / P088 Sec. 2.1), t.yr
LEDGER = {
    "LZ_untouched": 6.76, "XENONnT_analysed_lowE": 3.1, "XENONnT_untouched": 3.0,
    "PandaX4T_analysed_lowE": 1.54, "PandaX4T_untouched": 2.5,
}
UNEXAMINED = 17.9  # t.yr above 200 keV (P088; P069 quotes 19.1 incl. archival)
ACCRUAL = {"LZ": 2.79, "XENONnT": 2.4, "PandaX4T": 1.62}  # t.yr per yr (P069/P088)
# P079: LZ's untouched 524 live days, band-anchored L10 (2.38 events at 600 phd),
# 1000 phd edge signal x1.89/3.0/6.3 (L10/d350/d366), b 8.9e-4 -> 8.0e-3.
P079 = {"mu600_L10": 2.38, "gain1000": {"L10": 1.89, "d350": 3.0, "d366": 6.3},
        "b600": 8.9e-4, "b1000": 8.0e-3}
# P038: extra 600-1000 phd events per ROI event at LZ's raster-scan couplings
P038_EXTRA = {"d300": 0.155, "d350": 1.06, "d366": 3.81, "d370": 6.84, "d380": 30.6}
# P088 sideband table: SR3 (2.84 t.yr) mu = 0.28/0.155/1.06/3.81/30.6 for
# L10/d300/d350/d366/d380; untouched 6.76 t.yr adds x2.38.
SIDEBAND_SR3 = {"L10": 0.2815, "d300": 0.155, "d350": 1.062, "d366": 3.81, "d380": 30.6}

results = {"inputs": {"E_LZ": E_LZ, "rates_band": RATES_BAND, "rates_400": RATES_400,
                      "bkg_per_2p84": BKG, "weights": [PI_DM, PI_B, PI_U],
                      "ledger": LEDGER, "accrual": ACCRUAL}}

# ----------------------------------------------------------------------------
# 2. Gamma approximation to the P081 band-rate posterior
# ----------------------------------------------------------------------------
def gamma_from_mean_median(mean, median):
    """Shape a and scale th of a Gamma with the given mean and median."""
    def f(loga):
        a = np.exp(loga)
        return stats.gamma.median(a, scale=mean / a) - median
    a = np.exp(optimize.brentq(f, np.log(0.05), np.log(50)))
    return a, mean / a

A_POST, TH_POST = gamma_from_mean_median(P081_MEAN, P081_MEDIAN)
q16, q84 = stats.gamma.ppf([0.16, 0.84], A_POST, scale=TH_POST)
results["posterior_gamma"] = {"shape": A_POST, "scale_per_2p84": TH_POST,
                              "q16": q16, "q84": q84, "P081_68": P081_68}
print(f"Gamma fit to P081 posterior: a={A_POST:.3f}, theta={TH_POST:.3f}; "
      f"68% [{q16:.3f},{q84:.3f}] vs P081 {P081_68}")


def nb_pmf(n, k, a, th):
    """Gamma-Poisson (negative binomial) pmf for n counts when the mean is
    k*s with s ~ Gamma(a, th)."""
    p = 1.0 / (1.0 + k * th)
    return stats.nbinom.pmf(n, a, p)


def nb_sf(n, k, a, th):
    p = 1.0 / (1.0 + k * th)
    return stats.nbinom.sf(n, a, p)  # P(N > n)


# ----------------------------------------------------------------------------
# 3. Forecast table: P(0), P(1), P(>=2) per window and exposure
# ----------------------------------------------------------------------------
# Window ratios relative to the 200-270 keV band, from P081's predictive table
# (means at 2.8 t.yr: band 0.378, 55-270 1.028, 55-420 5.51, 270-420 4.48;
# the 1000 phd numbers are tail-dominated by delta >= 366 keV models, so P081
# quotes medians; we keep them only as P081 quotes them, not via ratios).
EXPOSURES = [1.0, 2.8, 5.6, 6.76, 10.0]
rows = []
for E in EXPOSURES:
    k = E / E_LZ
    # (i) corpus posterior (Gamma-Poisson)
    p0 = nb_pmf(0, k, A_POST, TH_POST); p1 = nb_pmf(1, k, A_POST, TH_POST)
    pge2 = nb_sf(1, k, A_POST, TH_POST)
    # (ii) backgrounds
    b = BKG["band"] * k
    rows.append(dict(exposure_tyr=E, window="200-270 keV", hypothesis="DM corpus posterior (Gamma-Poisson)",
                     mean=P081_MEAN * k, P0=p0, P1=p1, Pge2=pge2, Pge1=1 - p0))
    rows.append(dict(exposure_tyr=E, window="200-270 keV", hypothesis="modelled background (P016)",
                     mean=b, P0=np.exp(-b), P1=b * np.exp(-b), Pge2=stats.poisson.sf(1, b), Pge1=1 - np.exp(-b)))
    bc = BKG["corpus_nb"] * k
    rows.append(dict(exposure_tyr=E, window="200-270 keV", hypothesis="corpus-revised background (P093)",
                     mean=bc, P0=np.exp(-bc), P1=bc * np.exp(-bc), Pge2=stats.poisson.sf(1, bc), Pge1=1 - np.exp(-bc)))
    # steady unknown, Exponential(1) rate per 2.84 t.yr (P081): P(N) = k^N/(1+k)^(N+1)
    rows.append(dict(exposure_tyr=E, window="200-270 keV", hypothesis="steady unknown (Exp(1) rate, P081)",
                     mean=k, P0=1 / (1 + k), P1=k / (1 + k) ** 2, Pge2=1 - 1 / (1 + k) - k / (1 + k) ** 2,
                     Pge1=k / (1 + k)))
    # (iii) plug-in best fits, ROI-anchored (P069/P088)
    for m, r in RATES_BAND.items():
        mu = r * E
        rows.append(dict(exposure_tyr=E, window="200-270 keV", hypothesis=f"plug-in best fit {m} (P069 ROI-anchored)",
                         mean=mu, P0=np.exp(-mu), P1=mu * np.exp(-mu), Pge2=stats.poisson.sf(1, mu), Pge1=1 - np.exp(-mu)))
    mu = RATE_BAND_ANCHORED * E
    rows.append(dict(exposure_tyr=E, window="200-270 keV", hypothesis="plug-in, one band event per 2.84 t.yr (P020/P079 anchoring)",
                     mean=mu, P0=np.exp(-mu), P1=mu * np.exp(-mu), Pge2=stats.poisson.sf(1, mu), Pge1=1 - np.exp(-mu)))
    # extension 270-420 keV, background only (DM side is P081's, tail-dominated)
    bx = BKG["ext"] * k
    rows.append(dict(exposure_tyr=E, window="270-420 keV (1000 phd)", hypothesis="modelled background (P038 MSSI)",
                     mean=bx, P0=np.exp(-bx), P1=bx * np.exp(-bx), Pge2=stats.poisson.sf(1, bx), Pge1=1 - np.exp(-bx)))
    bl = BKG["low"] * k
    rows.append(dict(exposure_tyr=E, window="55-200 keV", hypothesis="modelled background (P081/P016)",
                     mean=bl, P0=np.exp(-bl), P1=bl * np.exp(-bl), Pge2=stats.poisson.sf(1, bl), Pge1=1 - np.exp(-bl)))
    for m, r in RATES_400.items():
        mu = r * E
        rows.append(dict(exposure_tyr=E, window="200-400 keV (1000 phd)", hypothesis=f"plug-in best fit {m} (P069/P050)",
                         mean=mu, P0=np.exp(-mu), P1=mu * np.exp(-mu), Pge2=stats.poisson.sf(1, mu), Pge1=1 - np.exp(-mu)))
fc = pd.DataFrame(rows)
fc.to_csv(f"{OUT}/tables/P099_forecast_counts.csv", index=False)

# Validation against P081 (2.8 t.yr, band): DM P0/P1/P>=2 = 0.752/0.169/0.079; B P1 = 5.6e-4
v = fc[(fc.exposure_tyr == 2.8) & (fc.window == "200-270 keV")]
dm = v[v.hypothesis.str.startswith("DM")].iloc[0]
results["validation_P081_2p8_band"] = {
    "ours_P0_P1_Pge2": [dm.P0, dm.P1, dm.Pge2], "P081": [0.752, 0.169, 0.079],
    "ours_P1_bkg": float(v[v.hypothesis.str.startswith("modelled")].iloc[0].P1), "P081_P1_bkg": 5.6e-4}
print("2.8 t.yr band: DM P0/P1/P>=2 =", np.round([dm.P0, dm.P1, dm.Pge2], 3), "(P081 0.752/0.169/0.079)")
dm10 = fc[(fc.exposure_tyr == 10.0) & (fc.window == "200-270 keV") & fc.hypothesis.str.startswith("DM")].iloc[0]
results["validation_P081_10_band"] = {"ours": [dm10.P0, dm10.P1, dm10.Pge2], "P081": [0.492, 0.215, 0.293]}
print("10 t.yr band: DM P0/P1/P>=2 =", np.round([dm10.P0, dm10.P1, dm10.Pge2], 3), "(P081 0.492/0.215/0.293)")

# Exposure for P(>=1 | DM) = 0.5 and 0.9 in the band (P081: 9.8 and 123 t.yr)
def E_for(p):
    f = lambda E: (1 - nb_pmf(0, E / E_LZ, A_POST, TH_POST)) - p
    return optimize.brentq(f, 0.01, 1e5)
results["exposure_for_Pge1_band"] = {"P0.5": E_for(0.5), "P0.9": E_for(0.9), "P081": [9.85, 123.3]}
print("E(P>=1 = 0.5/0.9) =", round(E_for(0.5), 1), round(E_for(0.9), 0), "t.yr (P081 9.8/123)")

# ----------------------------------------------------------------------------
# 4. Decision rule: P(DM | N) at 2.8 t.yr under the P061/P081 mixture
# ----------------------------------------------------------------------------
def mixture_post(E, N, band_only=True):
    k = E / E_LZ
    L_dm = nb_pmf(N, k, A_POST, TH_POST)
    b = BKG["band"] * k
    L_b = stats.poisson.pmf(N, b)
    L_u1 = 1.0 if N == 0 else 0.0  # one-off transient predicts nothing new
    L_us = k ** N / (1 + k) ** (N + 1)  # steady Exp(1) unknown
    L_u = F_TRANS * L_u1 + (1 - F_TRANS) * L_us
    num = PI_DM * L_dm
    den = num + PI_B * L_b + PI_U * L_u
    return dict(P_DM=num / den, BF_DM_B=L_dm / L_b, BF_DM_Usteady=L_dm / L_us, BF_DM_Umix=L_dm / L_u,
                L_dm=L_dm, L_b=L_b, L_us=L_us)

dec = []
for E in [1.0, 2.8, 5.6, 10.0]:
    for N in [0, 1, 2, 3]:
        r = mixture_post(E, N); r.update(exposure_tyr=E, N=N); dec.append(r)
dec = pd.DataFrame(dec)
dec.to_csv(f"{OUT}/tables/P099_decision_rule.csv", index=False)
d28 = dec[dec.exposure_tyr == 2.8].set_index("N")
results["validation_P081_decision_2p8"] = {
    "ours_PDM_N0_N1_N2": [d28.loc[0, "P_DM"], d28.loc[1, "P_DM"], d28.loc[2, "P_DM"]],
    "P081_PDM_N0_N1_N2": [0.0143, 0.0290, 0.0176],
    "ours_BF_DM_B_N1": d28.loc[1, "BF_DM_B"], "P081_BF_DM_B_N1": 300.3,
    "ours_BF_DM_Usteady_N1": d28.loc[1, "BF_DM_Usteady"], "P081": 0.674}
print("P(DM|N=0,1,2; 2.8 t.yr) =", np.round(d28.P_DM.values[:3], 4), "(P081 0.0143/0.0290/0.0176)")
print("BF DM:B (N=1) =", round(d28.loc[1, "BF_DM_B"], 0), "(P081 300); DM:U_steady =", round(d28.loc[1, "BF_DM_Usteady"], 2), "(P081 0.67)")

# Energy placement (P081 table, cited): recompute only the background side.
k28 = 2.8 / E_LZ
results["energy_placement_bkg_side"] = {
    "P(2 in 270-420 | B)": float(stats.poisson.pmf(2, BKG["ext"] * k28)),
    "P081_P_B_both_270_420": 3.4e-6, "P081_BF_DM_B": 3793, "P081_BF_DM_U": 7529, "P081_postDM": 0.9865,
    "P081_one_band_plus_one_ext_postDM": 0.571, "P081_both_band_postDM": 0.0109,
    "P081_band_plus_55_200_postDM": 0.061, "P081_both_55_200_postDM": 0.153}

# ----------------------------------------------------------------------------
# 5. Null branch: when does zero push P(DM) below 1e-3?
# ----------------------------------------------------------------------------
# P088 recipe: Gamma(2,1) rate posterior after LZ's one event (flat prior);
# P(0 new | DM) = (1+k)^-2 with k = expected best-fit events in the new data;
# alternatives predict zero with probability ~1; prior P(DM) = 0.015.
def p_dm_null(k, pi=PI_DM, alpha=2.0):
    L = (1 + k) ** (-alpha)
    return pi * L / (pi * L + (1 - pi))

def k_for_P(Ptarget, pi=PI_DM, alpha=2.0):
    return optimize.brentq(lambda k: p_dm_null(k, pi, alpha) - Ptarget, 0, 1e4)

K_1e3 = k_for_P(1e-3)
present_rate = sum(ACCRUAL.values())  # t.yr per yr
null_rows = []
for m, r in RATES_BAND.items():
    mu_now = r * UNEXAMINED
    need = max(K_1e3 - mu_now, 0.0)
    date = NOW + need / (r * present_rate) if need > 0 else NOW
    # LZ alone, 600 phd, untouched 6.76 t.yr then 2.79 t.yr/yr
    mu_lz = r * LEDGER["LZ_untouched"]
    need_lz = max(K_1e3 - mu_lz, 0.0)
    date_lz = NOW + need_lz / (r * ACCRUAL["LZ"])
    # LZ alone, 1000 phd: 200-400 keV rate on untouched data + SR3 sideband already empty
    r4 = RATES_400[m]
    mu_lz4 = r4 * LEDGER["LZ_untouched"] + (r4 - r) * E_LZ
    need4 = max(K_1e3 - mu_lz4, 0.0)
    date_lz4 = NOW + need4 / (r4 * ACCRUAL["LZ"])
    null_rows.append(dict(model=m, k_needed=K_1e3, mu_on_disk_present=mu_now, date_present_600phd=date,
                          mu_LZ_untouched_600=mu_lz, date_LZ_600phd=date_lz,
                          mu_LZ_1000=mu_lz4, date_LZ_1000phd=date_lz4))
nb = pd.DataFrame(null_rows)
nb.to_csv(f"{OUT}/tables/P099_null_branch.csv", index=False)
results["null_branch"] = {"k_for_P_lt_1e-3": K_1e3,
                          "present_600phd_dates": dict(zip(nb.model, nb.date_present_600phd.round(2))),
                          "P088_present_600phd": {"L10": 2027.61, "d300": 2030.86, "d350": 2026.38, "d366": 2025.16},
                          "LZ_600phd_dates": dict(zip(nb.model, nb.date_LZ_600phd.round(2))),
                          "LZ_1000phd_dates": dict(zip(nb.model, nb.date_LZ_1000phd.round(2))),
                          "P088_LZ_1000phd": {"L10": 2027.69, "d300": "never<2035", "d350": 2024.98, "d366": 2023.60}}
print("Null branch k for P(DM)<1e-3:", round(K_1e3, 2), "expected events")
print(nb.round(2).to_string())

# ----------------------------------------------------------------------------
# 6. Reanalysis on disk: hidden events, P(>=1), P(0); Z for two events
# ----------------------------------------------------------------------------
re_rows = []
E_XP = LEDGER["XENONnT_analysed_lowE"] + LEDGER["XENONnT_untouched"] + LEDGER["PandaX4T_analysed_lowE"] + LEDGER["PandaX4T_untouched"]
for m, r in RATES_BAND.items():
    mu_all = r * UNEXAMINED; mu_xp = r * E_XP; mu_lz = r * LEDGER["LZ_untouched"]
    re_rows.append(dict(model=m, mu_unexamined_17p9=mu_all, P0_unexamined=np.exp(-mu_all),
                        mu_XENONnT_PandaX_10p1=mu_xp, Pge1_XP=1 - np.exp(-mu_xp),
                        mu_LZ_untouched=mu_lz, Pge1_LZ=1 - np.exp(-mu_lz)))
# corpus-posterior version (Gamma-Poisson) for the same exposures
for lab, E in [("posterior_unexamined_17.9", UNEXAMINED), ("posterior_XP_10.1", E_XP), ("posterior_LZ_6.76", LEDGER["LZ_untouched"])]:
    k = E / E_LZ
    re_rows.append(dict(model=lab, mu_unexamined_17p9=P081_MEAN * k, P0_unexamined=nb_pmf(0, k, A_POST, TH_POST),
                        mu_XENONnT_PandaX_10p1=np.nan, Pge1_XP=np.nan, mu_LZ_untouched=np.nan, Pge1_LZ=np.nan))
re = pd.DataFrame(re_rows)
re.to_csv(f"{OUT}/tables/P099_reanalysis_on_disk.csv", index=False)
b_un = BKG["band"] * UNEXAMINED / E_LZ
Z2 = stats.norm.isf(stats.poisson.sf(1, b_un))
Z2c = stats.norm.isf(stats.poisson.sf(2, b_un + BKG["band"]))
results["reanalysis"] = {"mu_unexamined": dict(zip(re.model, re.mu_unexamined_17p9.round(3))),
                         "P088_mu_unexamined": {"L10": 2.2, "d300": 1.1, "d350": 3.2, "d366": 4.9},
                         "P0_unexamined": dict(zip(re.model, re.P0_unexamined.round(4))),
                         "P088_P0": {"L10": 0.12, "d300": 0.32, "d350": 0.04, "d366": 0.007},
                         "b_unexamined": b_un, "Z_two_new_events": Z2, "P088_Z": 4.3,
                         "Z_two_new_plus_LZ": Z2c, "P088_Z_with_LZ": 5.5}
print(re.round(3).to_string())
print(f"Two new events in 17.9 t.yr: Z = {Z2:.2f} (P088 4.3); with LZ's event {Z2c:.2f} (P088 5.5)")

# ----------------------------------------------------------------------------
# 7. LZ's own untouched exposure: P(5 sigma) with 600 vs 1000 phd (P079 check)
# ----------------------------------------------------------------------------
def n_for_5sigma(b):
    n = 1
    while stats.poisson.sf(n - 1, b) > stats.norm.sf(5):
        n += 1
    return n

p5 = []
for m in ["L10", "d350", "d366"]:
    mu6 = P079["mu600_L10"] if m == "L10" else None
    # P079 quotes P(5sigma) only for its own band-anchored L10 and delta grid via
    # signal gains; we check L10 (600 and 1000 phd) and d366 (1000 phd).
    if m == "L10":
        n6 = n_for_5sigma(P079["b600"]); P6 = stats.poisson.sf(n6 - 1, mu6)
    else:
        n6, P6 = np.nan, np.nan
    mu10 = P079["mu600_L10"] * P079["gain1000"][m]
    n10 = n_for_5sigma(P079["b1000"]); P10 = stats.poisson.sf(n10 - 1, mu10)
    p5.append(dict(model=m, mu_600=mu6, n_5sigma_600=n6, P5sigma_600=P6, mu_1000=mu10, n_5sigma_1000=n10, P5sigma_1000=P10))
p5 = pd.DataFrame(p5)
p5.to_csv(f"{OUT}/tables/P099_LZ_untouched_5sigma.csv", index=False)
results["validation_P079"] = {"ours": p5.to_dict("records"), "P079": {"L10_600": 0.43, "L10_1000": 0.83, "d366_1000": 1.00}}
print(p5.round(3).to_string())

# ----------------------------------------------------------------------------
# 8. Sideband already on disk (600-1000 phd): P(0 | delta)
# ----------------------------------------------------------------------------
sb = []
for m, mu in SIDEBAND_SR3.items():
    mu_un = mu * LEDGER["LZ_untouched"] / E_LZ
    sb.append(dict(model=m, mu_SR3=mu, P0_SR3=np.exp(-mu), mu_untouched=mu_un, P0_untouched=np.exp(-mu_un),
                   P0_SR3_plus_untouched=np.exp(-mu - mu_un)))
sb = pd.DataFrame(sb)
sb.to_csv(f"{OUT}/tables/P099_sideband.csv", index=False)
results["sideband"] = {"P0_SR3": dict(zip(sb.model, sb.P0_SR3.round(4))), "P088_P0_SR3_d366": 0.022,
                       "P0_untouched": dict(zip(sb.model, sb.P0_untouched.round(5)))}
print(sb.round(4).to_string())

# ----------------------------------------------------------------------------
# 9. Exothermic companions: P(0 in 125-200 keV | N band events)
# ----------------------------------------------------------------------------
# P072: N(125-200)/N(200-270) = 5.7/5.4/4.5/2.1 at |delta| = 278/300/350/500 keV
EXO_RATIO = {"278": 5.7, "300": 5.4, "350": 4.5, "500": 2.1}
exo = pd.DataFrame([dict(abs_delta_keV=d, companions_per_band_event=r,
                         P0_companions_given_1_band=np.exp(-r), P0_companions_given_2_band=np.exp(-2 * r))
                    for d, r in EXO_RATIO.items()])
exo.to_csv(f"{OUT}/tables/P099_exothermic_companions.csv", index=False)
results["exothermic"] = exo.to_dict("records")

# ----------------------------------------------------------------------------
# 10. Test x interpretation table
# ----------------------------------------------------------------------------
tests = [
 # test, interpretation, expected signature, P(outcome), date, source
 ("A. LZ untouched 6.76 t.yr, 200-270 keV (600 phd)", "inelastic O1 d335-390", "1.8-1.9 band events (d366/380); 0.4 (d300)", "P(>=1)=0.84 (d366), 0.35 (d300)", "next LZ release (2027)", "P069/P088"),
 ("A. LZ untouched 6.76 t.yr, 200-270 keV (600 phd)", "elastic L10/O6", "0.82 (ROI-anchored) to 2.38 (band-anchored) events", "P(>=1)=0.56-0.91; P(5sigma)=0.43 (band-anchored)", "2027", "P069/P020/P079"),
 ("A. LZ untouched 6.76 t.yr, 200-270 keV (600 phd)", "corpus DM posterior", "0.92 events (mean)", "P(>=1)=0.42", "2027", "P081/this work"),
 ("A. LZ untouched 6.76 t.yr, 200-270 keV (600 phd)", "background one-off", "0 events", "P(>=1)=1.4e-3 (modelled), 2.2e-3 (corpus)", "2027", "P016/P093"),
 ("A. LZ untouched 6.76 t.yr, 200-270 keV (600 phd)", "steady unknown", "2.4 events (Exp(1) rate)", "P(>=1)=0.70", "2027", "P081"),
 ("A'. LZ 1000 phd edge on the same data", "inelastic d366", "+9.1 events in 600-1000 phd", "P(0)=1.2e-4 -> killed if empty", "2027", "P038/P088"),
 ("A'. LZ 1000 phd edge on the same data", "elastic L10", "x1.89 signal, b 8e-3", "P(5sigma)=0.83 (band-anchored)", "2027", "P079"),
 ("B. XENONnT+PandaX-4T high-energy reanalysis (10.1 t.yr)", "any DM class", "1.2 (L10) - 2.8 (d366) events", "P(>=1)=0.71-0.94 best fit; 0.55 posterior", "mid-2027 earliest", "P069/P088"),
 ("B. XENONnT+PandaX-4T high-energy reanalysis (10.1 t.yr)", "steady LZ-specific unknown", "0 events", "kill if >=1 elsewhere; BF>100 for 2 events", "mid-2027", "P061/P069"),
 ("C. 600-1000 phd sideband already on disk (SR3, 2.84 t.yr)", "inelastic d>=366", "3.8-31 events expected, 0 seen", "P(0)=0.022 (366), 5e-14 (380)", "now", "P038/P088"),
 ("D. Annual modulation", "inelastic d>=350", "June clustering, a1=1.2-1.7", "3sigma needs 5-12 events = 15-33 t.yr", "2029-2035 (LZ); 2033 XLZD", "P034"),
 ("E. Spectral shape (second event energy)", "elastic vs inelastic", "E2<125 keV kills inelastic (LR<=0.04); E2>300 keV favours d366/380 x6-7", "2-7 events for 3sigma", "with the 2nd-3rd event", "P057"),
 ("F. CaWO4 10 kg.yr", "inelastic d366 / d350 / d300", "2.8 / 0.37 / 0.03 events", "P(3sigma)=0.62/0.03/0.004", ">=2030", "P046/P088"),
 ("F. 136Xe-enriched vs natural pair", "inelastic vs spin", "9 events shared, 12.8 t.yr each", "3sigma", "mid-2030s", "P047/P088"),
 ("F. Directional detectors", "inelastic near ceiling", "recoils within 21 deg of anti-apex", "<1 event/yr per 1000 m3", "decades", "P067"),
 ("G. Waveform/topology release (DR-003)", "artefact / charge-poor ER", "S1 PSD, S2 width, TBA, per-PMT", "odds shift +-0.45 dex (PSD), +-0.2 (audit)", "at LZ's discretion", "P093/P041/P077"),
 ("G. Data release DR-002 (event list, PDFs)", "statistical reassessment", "closes the 0.3 sigma residual of P052", "Z_L10 3.0-3.2 vs 3.4", "at LZ's discretion", "P052"),
 ("G. Data release DR-001 (L10 normalisation)", "L10 coupling convention", "factor-2 in d10", "P012/P045 provisional", "at LZ's discretion", "P012/P045"),
]
tt = pd.DataFrame(tests, columns=["test", "interpretation", "expected_signature", "P_outcome", "date", "source"])
tt.to_csv(f"{OUT}/tables/P099_test_x_interpretation.csv", index=False)
with open(f"{OUT}/tables/P099_test_x_interpretation.md", "w") as f:
    f.write(tt.to_markdown(index=False))

# Kill conditions
kills = [
 ("inelastic O1, delta 335-390 keV (P082)", "empty 600-1000 phd in LZ's untouched 6.76 t.yr: P(0|d366)=1.2e-4, P(0|d350)=0.08; any event <125 keV (LR<=0.04); Higgsino sub-case already excluded by solar capture (P076)", "2027 (next LZ release)"),
 ("elastic q^4-spin (L10/O6/O9xq^2)", "zero band events in 17.9 t.yr unexamined (P(0)=0.12 best fit; UL90 -> rate <1.0x best fit); a second event above 300 keV with a 1000 phd edge (LR 6-7 for inelastic); definitive exclusion of the 0.105 lower edge needs 49-62 t.yr", "2027 (weak), 2033 XLZD (strong)"),
 ("exothermic down-scatter (P072, B<=2)", "already P(0 companions | 1 band event)=exp(-5.4)=0.005 at |delta|=300 keV; any confirmed band event without a 125-200 keV companion", "now / with the next event"),
 ("background one-off (modelled or transient)", "any new NR-band event above 200 keV in LZ: BF>=300 (P081); expected with P=0.25 per 2.8 t.yr under the DM posterior", "2027"),
 ("steady LZ-specific unknown", "an event in XENONnT/PandaX-4T (P(>=1)=0.71-0.94 at best fit); or LZ events in 270-420 or 55-200 keV (two extension events: P(DM)=0.99)", "mid-2027 (reanalysis) / 2027 (LZ 1000 phd)"),
 ("instrumental artefact / charge-poor ER", "waveform-level quantities (DR-003): S1 prompt fraction, S2 width, TBA; P041's predicted 65-101 charge-deficient ERs in the empty strip and 83mKr map hole", "at LZ's discretion; no exposure needed"),
]
kd = pd.DataFrame(kills, columns=["interpretation", "kill_condition", "expected_date"])
kd.to_csv(f"{OUT}/tables/P099_kill_conditions.csv", index=False)

# Pre-registered predictions (numbers pulled from the tables above)
fc28 = fc[(fc.exposure_tyr == 2.8) & (fc.window == "200-270 keV")].set_index("hypothesis")
pred = [
 ("F1", "LZ next release, 2.8 t.yr new, 200-270 keV, 600 phd edge", "N=0", fc28.loc["DM corpus posterior (Gamma-Poisson)", "P0"], np.exp(-BKG["band"] * k28), "P081: 0.75 / 0.9994"),
 ("F2", "same", "N>=1", fc28.loc["DM corpus posterior (Gamma-Poisson)", "Pge1"], 1 - np.exp(-BKG["band"] * k28), "P081: 0.25 / 5.6e-4"),
 ("F3", "same", "N>=2", fc28.loc["DM corpus posterior (Gamma-Poisson)", "Pge2"], stats.poisson.sf(1, BKG["band"] * k28), "P081: 0.08 / 1.6e-7"),
 ("F4", "same, 55-270 keV", "N>=1", 0.50, 1 - np.exp(-(BKG["band"] + BKG["low"]) * k28), "P081: 0.50 (DM); background dominated by 55-125 keV"),
 ("F5", "same, 1000 phd edge, 270-420 keV only", "N>=1", 0.26, 1 - np.exp(-BKG["ext"] * k28), "P081 median: 0.26 (DM) / 0.0028 (B)"),
 ("F6", "LZ full untouched 6.76 t.yr, 200-270 keV", "N>=1", 1 - nb_pmf(0, LEDGER["LZ_untouched"] / E_LZ, A_POST, TH_POST), 1 - np.exp(-BKG["band"] * LEDGER["LZ_untouched"] / E_LZ), "P020 plug-in (band-anchored): 0.91"),
 ("F7", "XENONnT+PandaX-4T 10.1 t.yr reanalysed above 200 keV", "N>=1", 1 - nb_pmf(0, E_XP / E_LZ, A_POST, TH_POST), 1 - np.exp(-BKG["band"] * E_XP / E_LZ), "P069 best fit: 0.71 (L10) - 0.94 (d366)"),
]
pr = pd.DataFrame(pred, columns=["id", "dataset", "outcome", "P_DM_posterior", "P_background", "cross_reference"])
pr.to_csv(f"{OUT}/tables/P099_preregistered_predictions.csv", index=False)
results["preregistered"] = pr.to_dict("records")
print(pr.round(4).to_string())

# ----------------------------------------------------------------------------
# 11. Figure
# ----------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = {"blue": "#2a78d6", "orange": "#eb6834", "violet": "#4a3aa7", "red": "#e34948", "ink": "#0b0b0b",
     "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "surface": "#fcfcfb"}
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor=C["surface"])
# Left: P(>=1) vs exposure
Eg = np.logspace(np.log10(0.3), np.log10(60), 200)
kg = Eg / E_LZ
ax = axes[0]
ax.plot(Eg, 1 - nb_pmf(0, kg, A_POST, TH_POST), color=C["blue"], lw=2, label="DM, corpus posterior (P081-matched)")
ax.plot(Eg, 1 - np.exp(-RATES_BAND["L10"] * Eg), color=C["blue"], lw=1.4, ls="--", label="DM, L10 best fit, ROI-anchored (P069)")
ax.plot(Eg, 1 - np.exp(-RATE_BAND_ANCHORED * Eg), color=C["blue"], lw=1.4, ls=":", label="DM, one band event / 2.84 t·yr (P020/P079)")
ax.plot(Eg, kg / (1 + kg), color=C["violet"], lw=2, label="steady unknown, Exp(1) rate (P081)")
ax.plot(Eg, 1 - np.exp(-BKG["band"] * kg), color=C["orange"], lw=2, label="modelled background (P016)")
ax.plot(Eg, 1 - np.exp(-BKG["corpus_nb"] * kg), color=C["orange"], lw=1.4, ls="--", label="corpus-revised background (P093)")
for x, lab in [(2.8, "2.8"), (6.76, "LZ untouched 6.76"), (17.9, "unexamined 17.9")]:
    ax.axvline(x, color=C["muted"], lw=0.8, ls="-", alpha=0.6)
    ax.text(x * 1.04, 1.6e-4, lab, rotation=90, va="bottom", ha="left", fontsize=7.5, color=C["ink2"])
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-4, 1.05); ax.set_xlim(0.3, 60)
ax.set_xlabel("additional exposure above 200 keV (t·yr)", color=C["ink"])
ax.set_ylabel("P(≥1 event in 200–270 keV)", color=C["ink"])
ax.set_title("What LZ (and the xenon world) should see", fontsize=10.5, loc="left", color=C["ink"])
ax.grid(True, which="major", color=C["grid"], lw=0.6); ax.set_facecolor(C["surface"])
ax.legend(fontsize=7.2, frameon=False, loc="center left", bbox_to_anchor=(0.01, 0.40))
# Right: P(DM | outcome) at 2.8 t.yr
ax = axes[1]
outcomes = [
 ("nothing anywhere (55–420 keV)", 0.0089, C["orange"]),
 ("N = 0 in band", d28.loc[0, "P_DM"], C["orange"]),
 ("N = 1 in band only", d28.loc[1, "P_DM"], C["violet"]),
 ("two events, both in band", 0.0109, C["violet"]),
 ("band + 55–200 keV", 0.061, C["blue"]),
 ("both 55–200 keV", 0.153, C["blue"]),
 ("band + 270–420 keV", 0.571, C["blue"]),
 ("both 270–420 keV", 0.9865, C["blue"]),
]
y = np.arange(len(outcomes))[::-1]
for yi, (lab, p, col) in zip(y, outcomes):
    ax.barh(yi, p, color=col, height=0.62)
    ax.text(p * 1.15, yi, f"{p:.3g}", va="center", fontsize=8, color=C["ink2"])
ax.set_yticks(y); ax.set_yticklabels([o[0] for o in outcomes], fontsize=8.2)
ax.set_xscale("log"); ax.set_xlim(3e-3, 3)
ax.axvline(PI_DM, color=C["muted"], lw=0.9, ls="--"); ax.text(PI_DM * 1.05, y[0] + 0.55, "prior 0.015 (P061)", fontsize=7.5, color=C["ink2"])
ax.axvline(0.5, color=C["muted"], lw=0.9, ls=":"); ax.text(0.52, y[0] + 0.55, "0.5", fontsize=7.5, color=C["ink2"])
ax.set_xlabel("P(DM | outcome of the next 2.8 t·yr)", color=C["ink"])
ax.set_title("Decision rule: P(DM) after the next 2.8 t·yr", fontsize=10.5, loc="left", color=C["ink"])
ax.grid(True, axis="x", color=C["grid"], lw=0.6); ax.set_facecolor(C["surface"])
for a in axes:
    for s in ["top", "right"]:
        a.spines[s].set_visible(False)
    a.tick_params(colors=C["ink2"], labelsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/figures/P099_fig1_forecast.png", dpi=170, facecolor=C["surface"])
print("figure written")

def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    return o

with open(f"{OUT}/P099_results.json", "w") as f:
    json.dump(_clean(results), f, indent=1)
print("done")
