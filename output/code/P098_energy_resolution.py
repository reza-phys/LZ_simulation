"""
P098 -- Energy resolution at 250 keV from quanta statistics.

Does LZ's 248 +- 23 (stat) keV follow from N_q fluctuations, photon/electron counting statistics,
recombination and the NR-band width?  First-principles variance budget of the reconstructed energy of a
248 keV NR (S1c = 540.1 phd, S2c = 9268 phd) with nestpy 2.1.1 (LZ_WS2024 widths, LZ Table S5 mean yields
with the p(E) break, P009's paper-contour N_q scale), a layered detector model, three estimators
(S1-only, combined quanta N_q, 2D Gaussian-ML), the band-width <-> energy-resolution relation through the
S1-S2 anticorrelation, the ER comparison, the S1-only / S2-only event energies, and the propagation to
delta_max.  Run from the simulation root:   .venv/bin/python output/code/P098_energy_resolution.py
"""
import sys, json, math, time
sys.path.insert(0, 'output/code')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nestpy
from common import lzcommon as lz

t0 = time.time()
OUT = "output/work/P098"
FIG = OUT + "/figures"
import os
os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(98)
nestpy.RandomGen.rndm().set_seed(98)

# ------------------------------------------------------------------------------------------------
# 0. Inputs
# ------------------------------------------------------------------------------------------------
G1, G1E = lz.LZ["g1"], lz.LZ["g1_err"]            # 0.110 +- 0.002 phd/photon   [paper, detector paragraph]
G2, G2E = lz.LZ["g2"], lz.LZ["g2_err"]            # 34.5 +- 1.1 phd/electron
S1_EV, S2_EV = lz.LZ["ev_S1c"], lz.LZ["ev_S2c"]   # 540.1 phd, 9268 phd          [paper, Results]
E_PAPER, E_STAT, E_SYS = 248.0, 23.0, 23.0        # paper: 248 +- 23 (stat) +- 23 (sys) keV (abstract, Results, Summary)
FIELD, DENSITY = lz.DRIFT_FIELD_VCM, 2.9
DET = nestpy.detectors.LZ_WS2024()
NC = nestpy.NESTcalc(DET)
WIDTH_LZ = list(DET.nr_er_width_parameters)   # LZ_WS2024 NR/ER width vector (13 entries)
P_DPHE = DET.get_P_dphe()                      # 0.214 in the nestpy object (P009 details)
SPE_RES = DET.get_sPEres()          # 0.338
S2_FANO = DET.get_s2Fano()          # 4.0
EXT_EFF = 0.80                      # recalled (LZ SR1 80.5 %; uncertain); P024 baseline. nestpy CalculateG2 -> 0.726 (variant)
POS_RES = 0.02                      # assumed residual of position corrections (P009/P024 baseline); 0.03 variant (P043)
ELIFE_REL = 0.015                   # S2c electron-lifetime correction uncertainty: drift 870 us, tau_e ~ 6 ms (recalled, uncertain),
                                    # 10 % on tau -> 0.87/6 * 0.10 = 1.5 % on S2c.  Enters only through S2.
try:
    W_NEST = NC.WorkFunction(DENSITY).Wq_eV      # NEST v2 density-dependent work function
except Exception:
    W_NEST = 13.44                               # value P009 obtained from nestpy (eV per quantum)
W_PAPER = 13.4628                   # P009: implied by the paper's keVee contour labels
ALPHA_P, BETA_P = 11.3245, 1.11167  # P009: paper-contour scale N_q = alpha E^beta (results.json)

def yields_tab(E):
    """LZ Table S5 (with the p(E) break) mean yields via lzcommon/nestpy."""
    return lz.nest_nr_yields(E)

def yields_contour(E):
    """P009 paper-contour scale: N_q = 11.32 E^1.112, N_e from Table S5, N_ph = N_q - N_e."""
    nph, ne = yields_tab(E)
    nq = ALPHA_P * E ** BETA_P
    return nq - ne, ne

SCALES = {"TableS5": yields_tab, "contour": yields_contour}

# ------------------------------------------------------------------------------------------------
# 1. Mean quanta at 248 keV, combined energy and its inversion
# ------------------------------------------------------------------------------------------------
E_grid = np.arange(20.0, 500.01, 0.5)     # wide enough that no 100-330 keV fluctuation falls off the grid
curves = {}
for name, f in SCALES.items():
    arr = np.array([f(E) for E in E_grid])
    curves[name] = dict(E=E_grid, nph=arr[:, 0], ne=arr[:, 1], nq=arr.sum(axis=1))

def invert(curve_key, scale, value):
    c = curves[scale]
    y = c[curve_key]
    if not np.all(np.diff(y) > 0):
        # S2 (N_e) curve is monotone but very flat; use monotone interpolation on the increasing part
        idx = np.argmax(y)
        return float(np.interp(value, y[:idx + 1], c["E"][:idx + 1], left=np.nan, right=np.nan))
    return float(np.interp(value, y, c["E"], left=np.nan, right=np.nan))

nq_ev = S1_EV / G1 + S2_EV / G2
nph_ev, ne_ev = S1_EV / G1, S2_EV / G2
part1 = dict(event=dict(Nph_est=nph_ev, Ne_est=ne_ev, Nq_est=nq_ev,
                        E_ee_keV_W13p7=lz.combined_energy_keV(S1_EV, S2_EV),
                        E_ee_keV_Wnest=lz.combined_energy_keV(S1_EV, S2_EV, W_eV=W_NEST),
                        E_ee_keV_Wpaper=lz.combined_energy_keV(S1_EV, S2_EV, W_eV=W_PAPER)))
for name in SCALES:
    nph, ne = SCALES[name](E_PAPER)
    part1[name] = dict(Nph_248=nph, Ne_248=ne, Nq_248=nph + ne, S1c_248=G1 * nph, S2c_248=G2 * ne,
                       log10S2c_248=math.log10(G2 * ne),
                       E_S1_only=invert("nph", name, nph_ev), E_Nq=invert("nq", name, nq_ev),
                       E_S2_only=invert("ne", name, ne_ev),
                       dNq_dE=float(np.gradient(curves[name]["nq"], E_grid)[np.searchsorted(E_grid, E_PAPER)]),
                       dNph_dE=float(np.gradient(curves[name]["nph"], E_grid)[np.searchsorted(E_grid, E_PAPER)]),
                       dNe_dE=float(np.gradient(curves[name]["ne"], E_grid)[np.searchsorted(E_grid, E_PAPER)]))
    c = curves[name]
    i = np.searchsorted(E_grid, E_PAPER)
    part1[name]["beta_eff_Nq"] = float(np.gradient(np.log(c["nq"]), np.log(E_grid))[i])
    part1[name]["beta_eff_S1"] = float(np.gradient(np.log(c["nph"]), np.log(E_grid))[i])
    part1[name]["beta_eff_S2"] = float(np.gradient(np.log(c["ne"]), np.log(E_grid))[i])
print("Part 1:", json.dumps(part1, indent=1, default=float))

# ------------------------------------------------------------------------------------------------
# 2. Quanta fluctuations at fixed E from nestpy GetQuanta
# ------------------------------------------------------------------------------------------------
def quanta_samples(E, n, scale="contour", kind="NR"):
    if kind == "NR":
        y = NC.GetYields(nestpy.interactions.NR, float(E), DENSITY, FIELD, 131.293, 54,
                         lz.nest_nr_params_vector(E, lz.NEST_NR_LZ))
    else:
        er = [lz.NEST_ER_LZ[f"m{i}"] for i in range(1, 11)]
        y = NC.GetYields(nestpy.interactions.beta, float(E), DENSITY, FIELD, 131.293, 54,
                         list(nestpy.default_nr_parameters), er)
    out = np.empty((n, 2))
    for i in range(n):
        q = NC.GetQuanta(y, DENSITY, WIDTH_LZ)
        out[i] = q.photons, q.electrons
    if kind == "NR" and scale == "contour":       # shift photons by the mean difference (P009/P024 convention)
        out[:, 0] += yields_contour(E)[0] - yields_tab(E)[0]
    return out

LAYERS = ["quanta", "+S1 binomial (g1)", "+double phe", "+SPE resolution", "+extraction binomial",
          "+SE Fano", "+position residual", "+e-lifetime corr."]

def detect(q, upto, pos_res=POS_RES, ext=EXT_EFF, elife=ELIFE_REL, dphe=P_DPHE):
    """S1c, S2c from quanta with detector layers switched on cumulatively up to index `upto`."""
    nph = np.clip(np.round(q[:, 0]).astype(int), 0, None)
    ne = np.clip(np.round(q[:, 1]).astype(int), 0, None)
    # S1
    if upto >= 1:
        if upto >= 2:
            ndet = rng.binomial(nph, G1 / (1 + dphe))
            phe = ndet + rng.binomial(ndet, dphe)          # phd = detected photons + double-phe extras
        else:
            ndet = rng.binomial(nph, G1)
            phe = ndet.astype(float)
        phe = phe.astype(float)
        if upto >= 3:
            phe = phe + rng.normal(0, SPE_RES * np.sqrt(np.clip(ndet, 1, None)))
        s1 = phe
    else:
        s1 = G1 * nph.astype(float)
    # S2
    if upto >= 4:
        nx = rng.binomial(ne, ext).astype(float)
        s2 = nx * (G2 / ext)
    else:
        s2 = G2 * ne.astype(float)
    if upto >= 5:
        s2 = s2 + rng.normal(0, np.sqrt(S2_FANO * np.clip(s2, 1, None)))
    if upto >= 6:
        s1 = s1 * (1 + rng.normal(0, pos_res, s1.size))
        s2 = s2 * (1 + rng.normal(0, pos_res, s2.size))
    if upto >= 7:
        s2 = s2 * (1 + rng.normal(0, elife, s2.size))
    return s1, s2

def estimators(s1, s2, scale):
    nq = s1 / G1 + s2 / G2
    c = curves[scale]
    E_nq = np.interp(nq, c["nq"], c["E"], left=np.nan, right=np.nan)
    E_s1 = np.interp(s1 / G1, c["nph"], c["E"], left=np.nan, right=np.nan)
    return E_nq, E_s1, nq

def stats(x):
    x = x[np.isfinite(x)]
    p16, p50, p84 = np.percentile(x, [16, 50, 84])
    return dict(mean=float(x.mean()), sd=float(x.std()), p16=float(p16), p50=float(p50), p84=float(p84),
                hw68=float((p84 - p16) / 2))

N_MAIN = 120_000
print(f"[{time.time()-t0:.0f}s] sampling {N_MAIN} quanta at 248 keV ...")
Q248 = quanta_samples(E_PAPER, N_MAIN, "contour")
Q248_tab = Q248.copy(); Q248_tab[:, 0] -= yields_contour(E_PAPER)[0] - yields_tab(E_PAPER)[0]
print(f"[{time.time()-t0:.0f}s] done")

nph_s, ne_s = Q248[:, 0], Q248[:, 1]
nq_s = nph_s + ne_s
cov = np.cov(nph_s, ne_s)
quanta = dict(mean_Nph=float(nph_s.mean()), mean_Ne=float(ne_s.mean()), mean_Nq=float(nq_s.mean()),
              sd_Nph=float(nph_s.std()), sd_Ne=float(ne_s.std()), sd_Nq=float(nq_s.std()),
              cov_Nph_Ne=float(cov[0, 1]), corr_Nph_Ne=float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])),
              check_VarNq=float(cov[0, 0] + cov[1, 1] + 2 * cov[0, 1]))
print("quanta fluctuations:", quanta)

# variance budget, layer by layer, for the N_q and S1-only estimators, both scales
budget_rows = []
prev = {}
for scale in ("contour", "TableS5"):
    Q = Q248 if scale == "contour" else Q248_tab
    dNqdE, dNphdE = part1[scale]["dNq_dE"], part1[scale]["dNph_dE"]
    prev_var = dict(Nq=0.0, S1=0.0, ENq=0.0, ES1=0.0)
    for k, layer in enumerate(LAYERS):
        s1, s2 = detect(Q, k)
        E_nq, E_s1, nq = estimators(s1, s2, scale)
        v = dict(Nq=float(np.var(nq)), S1=float(np.var(s1 / G1)),
                 ENq=float(np.nanvar(E_nq)), ES1=float(np.nanvar(E_s1)))
        row = dict(scale=scale, layer=layer,
                   sd_Nq_cum=math.sqrt(v["Nq"]), sd_Nq_marg=math.sqrt(max(v["Nq"] - prev_var["Nq"], 0)),
                   sd_ENq_cum=math.sqrt(v["ENq"]), sd_ENq_marg=math.sqrt(max(v["ENq"] - prev_var["ENq"], 0)),
                   sd_ES1_cum=math.sqrt(v["ES1"]), sd_ES1_marg=math.sqrt(max(v["ES1"] - prev_var["ES1"], 0)),
                   sd_S1c_cum_phd=G1 * math.sqrt(v["S1"]),
                   sd_log10S2c_cum=float(np.std(np.log10(np.clip(s2, 1, None)))),
                   hw68_ENq=stats(E_nq)["hw68"], hw68_ES1=stats(E_s1)["hw68"])
        row["pct_ENq_marg"] = 100 * row["sd_ENq_marg"] / E_PAPER
        row["pct_ENq_cum"] = 100 * row["sd_ENq_cum"] / E_PAPER
        budget_rows.append(row)
        prev_var = v
budget = pd.DataFrame(budget_rows)
budget.to_csv(OUT + "/variance_budget.csv", index=False)
print(budget.round(3).to_string())

# analytic expectations for the marginal terms (contour scale, in N_q quanta units)
nph_m, ne_m = yields_contour(E_PAPER)
nq_m = nph_m + ne_m
n_det = nph_m * G1 / (1 + P_DPHE)
ana = {
    "S1 binomial (g1)": math.sqrt(nph_m * (1 - G1) / G1),
    "double phe (extra)": math.sqrt((n_det * P_DPHE * (1 - P_DPHE) + (1 + P_DPHE) ** 2 * n_det * (1 - G1 / (1 + P_DPHE))) / G1 ** 2
                                    - nph_m * (1 - G1) / G1),
    "SPE resolution": SPE_RES * math.sqrt(n_det) / G1,
    "extraction binomial": math.sqrt(ne_m * (1 - EXT_EFF) / EXT_EFF),
    "SE Fano": math.sqrt(S2_FANO * G2 * ne_m) / G2,
    "position residual": POS_RES * math.sqrt(nph_m ** 2 + ne_m ** 2),
    "e-lifetime": ELIFE_REL * ne_m,
}
ana_keV = {k: v / part1["contour"]["dNq_dE"] for k, v in ana.items()}
print("analytic marginal sd in quanta:", {k: round(v, 1) for k, v in ana.items()})
print("analytic marginal sd in keV   :", {k: round(v, 2) for k, v in ana_keV.items()})

# 2D Gaussian-ML estimator: Fisher information sigma_E^-2 = m'(E)^T C^-1 m'(E) with C from full-detector MC
def mean_vec(E, scale, n=20000):
    q = quanta_samples(E, n, scale)
    s1, s2 = detect(q, len(LAYERS) - 1)
    return np.array([s1.mean(), np.log10(s2).mean()]), np.cov(s1, np.log10(s2))

twoD = {}
for scale in ("contour", "TableS5"):
    print(f"[{time.time()-t0:.0f}s] 2D estimator, {scale}")
    ne_m_local = SCALES[scale](E_PAPER)[1]
    m_lo, _ = mean_vec(E_PAPER - 4, scale)
    m_hi, _ = mean_vec(E_PAPER + 4, scale)
    m0, C = mean_vec(E_PAPER, scale, 40000)
    dm = (m_hi - m_lo) / 8.0
    Cinv = np.linalg.inv(C)
    fisher = dm @ Cinv @ dm
    sigE = 1 / math.sqrt(fisher)
    # weights of the optimal linear estimator: w = C^-1 dm / fisher
    w = Cinv @ dm / fisher
    twoD[scale] = dict(mean_S1c=float(m0[0]), mean_log10S2c=float(m0[1]), sd_S1c=float(math.sqrt(C[0, 0])),
                       sd_log10S2c=float(math.sqrt(C[1, 1])), corr=float(C[0, 1] / math.sqrt(C[0, 0] * C[1, 1])),
                       dS1c_dE=float(dm[0]), dlog10S2c_dE=float(dm[1]), sigma_E_2D=sigE,
                       sigma_E_S1only_gauss=float(math.sqrt(C[0, 0]) / abs(dm[0])),
                       sigma_E_S2only_gauss=float(math.sqrt(C[1, 1]) / abs(dm[1])),
                       weight_S1_keV_per_phd=float(w[0]), weight_log10S2c_keV_per_dex=float(w[1]),
                       # energy-equivalent of the event's charge deficit through the 2D estimator
                       event_nsig_band=float((m0[1] - math.log10(S2_EV)) / math.sqrt(C[1, 1])),
                       event_E_2D_gauss=float(E_PAPER + w[0] * (S1_EV - m0[0]) + w[1] * (math.log10(S2_EV) - m0[1])),
                       event_dE_from_S2_deficit=float(w[1] * (math.log10(S2_EV) - m0[1])),
                       Nq_estimator_weight_keV_per_dex=float(math.log(10) * ne_m_local / part1[scale]["dNq_dE"]))
print("2D:", json.dumps(twoD, indent=1))

# ------------------------------------------------------------------------------------------------
# 3. Band width <-> energy resolution through the S1-S2 anticorrelation (contour scale, full detector)
# ------------------------------------------------------------------------------------------------
s1f, s2f = detect(Q248, len(LAYERS) - 1)
l2f = np.log10(s2f)
# conditional band width at fixed E (no energy mixing) and residual after regressing on S1c
slope = np.cov(s1f, l2f)[0, 1] / np.var(s1f)
resid = l2f - slope * (s1f - s1f.mean())
band = dict(sd_log10S2c_fixedE=float(l2f.std()), sd_log10S2c_given_S1c=float(resid.std()),
            slope_dlog10S2c_dS1c=float(slope),
            electrons_per_sigma_band=float(l2f.std() * math.log(10) * ne_m),
            keV_per_sigma_band_Nq=float(l2f.std() * math.log(10) * ne_m / part1["contour"]["dNq_dE"]),
            # recombination anticorrelation: Var_rec(Ne) cancels in Nq
            var_Nph_quanta=float(cov[0, 0]), var_Ne_quanta=float(cov[1, 1]), cov_quanta=float(cov[0, 1]),
            var_Nq_quanta=float(np.var(nq_s)),
            frac_of_VarNe_cancelled=float(-cov[0, 1] / cov[1, 1]),
            P024_sd_band=0.0313, P024_recomb_term=0.0231)
# event charge deficit in electrons and keV
ne_med_at_event = 10 ** twoD["contour"]["mean_log10S2c"] / G2
band["event_Ne_obs"] = ne_ev
band["event_Ne_model_248"] = float(ne_med_at_event)
band["event_deficit_e"] = float(ne_med_at_event - ne_ev)
band["event_deficit_keV_in_Nq"] = band["event_deficit_e"] / part1["contour"]["dNq_dE"]
band["event_nsig"] = twoD["contour"]["event_nsig_band"]
print("band:", json.dumps(band, indent=1))

# ------------------------------------------------------------------------------------------------
# 4. Energy dependence of sigma_E (N_q and S1 estimators, full detector), fit sigma = A (E/248)^gamma
# ------------------------------------------------------------------------------------------------
E_list = [100.0, 150.0, 200.0, 248.0, 300.0, 330.0]
edep = []
for E in E_list:
    n = 25000
    q = quanta_samples(E, n, "contour")
    s1, s2 = detect(q, len(LAYERS) - 1)
    E_nq, E_s1, _ = estimators(s1, s2, "contour")
    st_nq, st_s1 = stats(E_nq), stats(E_s1)
    edep.append(dict(E=E, sd_ENq=st_nq["sd"], hw68_ENq=st_nq["hw68"], sd_ES1=st_s1["sd"], hw68_ES1=st_s1["hw68"],
                     pct_ENq=100 * st_nq["sd"] / E, sd_log10S2c=float(np.std(np.log10(s2))),
                     sd_S1c_rel=float(s1.std() / s1.mean())))
edep = pd.DataFrame(edep)
edep.to_csv(OUT + "/sigma_E_vs_E.csv", index=False)
fit_nq = np.polyfit(np.log(edep.E / 248.0), np.log(edep.sd_ENq), 1)
fit_s1 = np.polyfit(np.log(edep.E / 248.0), np.log(edep.sd_ES1), 1)
edep_fit = dict(A_Nq=float(math.exp(fit_nq[1])), gamma_Nq=float(fit_nq[0]),
                A_S1=float(math.exp(fit_s1[1])), gamma_S1=float(fit_s1[0]),
                sqrt_law_11_at_E={str(E): 11 * math.sqrt(E / 248) for E in E_list})
print(edep.round(3).to_string()); print(edep_fit)

# ------------------------------------------------------------------------------------------------
# 5. Detector-parameter variants at 248 keV (N_q estimator, full detector)
# ------------------------------------------------------------------------------------------------
variants = {}
for label, kw in [("baseline", {}), ("pos 0 %", dict(pos_res=0.0)), ("pos 3 %", dict(pos_res=0.03)),
                  ("pos 4 %", dict(pos_res=0.04)), ("ext 0.726", dict(ext=0.726)), ("no dphe", dict(dphe=0.0)),
                  ("elife 0", dict(elife=0.0)), ("elife 5 %", dict(elife=0.05))]:
    s1, s2 = detect(Q248, len(LAYERS) - 1, **kw)
    E_nq, E_s1, _ = estimators(s1, s2, "contour")
    variants[label] = dict(sd_ENq=stats(E_nq)["sd"], hw68_ENq=stats(E_nq)["hw68"], sd_ES1=stats(E_s1)["sd"],
                           sd_S1c_rel=float(s1.std() / s1.mean()))
# what S1 spread would be needed for sigma_E = 23 keV?
need = dict(sd_Nq_for_23keV=23.0 * part1["contour"]["dNq_dE"],
            rel_S1_for_23keV=23.0 * part1["contour"]["dNq_dE"] / nph_m,
            variance_scale_needed=(23.0 / variants["baseline"]["sd_ENq"]) ** 2)
print("variants:", json.dumps(variants, indent=1)); print("needed for 23 keV:", need)

# ------------------------------------------------------------------------------------------------
# 6. ER at 248 keV (beta, LZ Table S3 means) with the same machinery; E = W N_q
# ------------------------------------------------------------------------------------------------
print(f"[{time.time()-t0:.0f}s] ER sampling")
nph_er, ne_er = lz.nest_er_yields(E_PAPER, params=lz.NEST_ER_LZ)
Q_er = quanta_samples(E_PAPER, 40000, kind="ER")
er_rows = []
prev = 0.0
for k, layer in enumerate(LAYERS):
    s1, s2 = detect(Q_er, k)
    nq = s1 / G1 + s2 / G2
    E_er = nq * W_NEST * 1e-3
    v = float(np.var(E_er))
    er_rows.append(dict(layer=layer, sd_E_cum=math.sqrt(v), sd_E_marg=math.sqrt(max(v - prev, 0)),
                        pct_cum=100 * math.sqrt(v) / E_er.mean(), mean_E=float(E_er.mean())))
    prev = v
er = pd.DataFrame(er_rows)
er.to_csv(OUT + "/er_budget_248keV.csv", index=False)
er_summary = dict(Nph=nph_er, Ne=ne_er, Nq=nph_er + ne_er, E_W_Nq=(nph_er + ne_er) * W_NEST * 1e-3,
                  sd_Nph_q=float(Q_er[:, 0].std()), sd_Ne_q=float(Q_er[:, 1].std()),
                  corr_q=float(np.corrcoef(Q_er[:, 0], Q_er[:, 1])[0, 1]), sd_Nq_q=float(Q_er.sum(axis=1).std()),
                  sigma_E_pct_full=er_rows[-1]["pct_cum"], sigma_E_keV_full=er_rows[-1]["sd_E_cum"],
                  recalled_XENON1T_formula_pct=100 * (0.317 / math.sqrt(E_PAPER) + 0.0015),
                  recalled_XENON1T_164_pct=100 * (0.317 / math.sqrt(164) + 0.0015),
                  recalled_XENON1T_236_pct=100 * (0.317 / math.sqrt(236) + 0.0015))
print(er.round(3).to_string()); print("ER:", er_summary)

# ------------------------------------------------------------------------------------------------
# 7. S1-only / S2-only event energies with their resolutions (MC intervals)
# ------------------------------------------------------------------------------------------------
s1f, s2f = detect(Q248, len(LAYERS) - 1)
E_nq_f, E_s1_f, _ = estimators(s1f, s2f, "contour")
c = curves["contour"]
imax = int(np.argmax(c["ne"]))
E_s2_f = np.interp(s2f / G2, c["ne"][:imax + 1], c["E"][:imax + 1], left=np.nan, right=np.nan)
evt = {}
for scale in ("contour", "TableS5"):
    evt[scale] = dict(E_S1_only=part1[scale]["E_S1_only"], E_Nq=part1[scale]["E_Nq"], E_S2_only=part1[scale]["E_S2_only"])
evt["MC_248_contour"] = dict(E_S1=stats(E_s1_f), E_Nq=stats(E_nq_f), E_S2=stats(E_s2_f),
                             frac_S2_unresolved=float(np.mean(~np.isfinite(E_s2_f))))
# S2-only energy for the event: propagate sd(N_e) through the flat N_e(E)
sd_ne_full = float(np.std(s2f / G2))
evt["S2_only_sigma_keV_linear"] = sd_ne_full / part1["contour"]["dNe_dE"]
evt["S1_minus_S2_energy_keV"] = evt["contour"]["E_S1_only"] - evt["contour"]["E_S2_only"]
evt["S1_minus_S2_in_sigma"] = evt["S1_minus_S2_energy_keV"] / math.hypot(
    twoD["contour"]["sigma_E_S1only_gauss"], evt["S2_only_sigma_keV_linear"])
print("event:", json.dumps(evt, indent=1, default=float))

# ------------------------------------------------------------------------------------------------
# 8. Propagation to delta_max and to the inelastic-likelihood peak
# ------------------------------------------------------------------------------------------------
v_june = lz.vmax_kms(lz.v_earth_kms(day_of_year=167))
prop = dict(v_max_june_kms=float(v_june))
sig_options = dict(sigma_Nq_MC=variants["baseline"]["sd_ENq"], sigma_2D=twoD["contour"]["sigma_E_2D"], sigma_23=23.0,
                   sigma_total_with_P043_sys=math.hypot(variants["baseline"]["sd_ENq"], 9.6))
for m in (400, 1000, 4000):
    d0 = lz.delta_max_kev(E_PAPER, m, v_kms=v_june)
    slope_d = (lz.delta_max_kev(E_PAPER + 5, m, v_kms=v_june) - lz.delta_max_kev(E_PAPER - 5, m, v_kms=v_june)) / 10
    prop[f"m{m}"] = dict(delta_max_248=float(d0), ddelta_dE=float(slope_d),
                         **{f"sigma_delta_{k}": float(abs(slope_d) * s) for k, s in sig_options.items()})
prop["P021_peak_slope_keV_per_keV"] = 0.31
prop["P021_peak_shift"] = {k: 0.31 * s for k, s in sig_options.items()}
# smeared density at 248 keV for a spectrum with a sharp kinematic edge at E_max (flat below): ratio 23 vs 11
from scipy.stats import norm
edge = {}
for Emax in (248, 240, 230, 220):
    f11 = norm.cdf((Emax - 248) / 11.0); f23 = norm.cdf((Emax - 248) / 23.0)
    edge[str(Emax)] = dict(f_sigma11=float(f11), f_sigma23=float(f23), ratio=float(f23 / f11))
prop["edge_density_ratio_23_vs_11"] = edge
print("propagation:", json.dumps(prop, indent=1))

# ------------------------------------------------------------------------------------------------
# 9. Figures
# ------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
b = budget[budget.scale == "contour"]
ax = axes[0]
ax.barh(b.layer, b.sd_ENq_marg, color="#4C72B0")
for i, (m_, cval) in enumerate(zip(b.sd_ENq_marg, b.sd_ENq_cum)):
    ax.text(m_ + 0.15, i, f"{m_:.1f} (cum {cval:.1f})", va="center", fontsize=8)
ax.axvline(23, color="0.4", ls="--"); ax.text(23.3, len(b) - 0.7, "LZ ±23 (stat)", rotation=90, fontsize=8, color="0.4", va="bottom")
ax.set_xlabel("marginal σ_E of the N_q estimator [keV] (cumulative in brackets)")
ax.set_title("248 keV NR: variance budget (contour scale)")
ax.set_xlim(0, 26); ax.invert_yaxis()
ax = axes[1]
ax.errorbar(edep.E, edep.sd_ENq, fmt="o-", label="N_q estimator (MC, full detector)")
ax.plot(edep.E, edep.sd_ES1, "s--", label="S1-only estimator")
Ef = np.linspace(100, 330, 50)
ax.plot(Ef, 11 * np.sqrt(Ef / 248), ":", color="0.3", label="11 √(E/248) (P021/P057)")
ax.plot(Ef, edep_fit["A_Nq"] * (Ef / 248) ** edep_fit["gamma_Nq"], "-", color="0.6", lw=0.8,
        label=f"fit {edep_fit['A_Nq']:.1f} (E/248)^{edep_fit['gamma_Nq']:.2f}")
ax.axhline(23, color="0.4", ls="--"); ax.text(105, 23.5, "±23 keV", fontsize=8, color="0.4")
ax.set_xlabel("recoil energy [keV]"); ax.set_ylabel("σ_E [keV]"); ax.legend(fontsize=8)
ax.set_title("energy dependence"); ax.set_ylim(0, 26)
fig.tight_layout(); fig.savefig(FIG + "/P098_budget_and_energy_dependence.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.5, 4.2))
sel = rng.choice(len(s1f), 6000, replace=False)
ax.scatter(s1f[sel], np.log10(s2f[sel]), s=2, alpha=0.3, color="#4C72B0", label="MC, 248 keV NR (full detector)")
ax.plot(S1_EV, math.log10(S2_EV), "r*", ms=12, label="event (540.1, 3.967)")
xs = np.linspace(430, 660, 50)
for name, f in SCALES.items():
    pts = np.array([[G1 * f(E)[0], math.log10(G2 * f(E)[1])] for E in (200, 220, 248, 270, 300)])
    ax.plot(pts[:, 0], pts[:, 1], "k-" if name == "contour" else "k--", lw=0.8, label=f"NR mean, {name} scale")
ax.set_xlabel("S1c [phd]"); ax.set_ylabel("log10 S2c"); ax.legend(fontsize=7, loc="lower right")
ax.set_title("fixed-energy scatter: S1 (energy) vs S2 (band) axes")
fig.tight_layout(); fig.savefig(FIG + "/P098_fixed_energy_scatter.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------------------------------------------
# 10. Save
# ------------------------------------------------------------------------------------------------
res = dict(inputs=dict(g1=G1, g2=G2, S1c=S1_EV, S2c=S2_EV, field=FIELD, width_parameters=WIDTH_LZ, P_dphe=P_DPHE,
                       sPEres=SPE_RES, s2Fano=S2_FANO, ext_eff=EXT_EFF, pos_res=POS_RES, elife_rel=ELIFE_REL,
                       W_nest_eV=W_NEST, W_paper_eV=W_PAPER, alpha_paper=ALPHA_P, beta_paper=BETA_P, N_main=N_MAIN,
                       paper_energy=dict(E=E_PAPER, stat=E_STAT, sys=E_SYS,
                                         source="fulltext.tex lines 15, 165, 332: 248 +- 23 (stat) +- 23 (sys) keV")),
           part1=part1, quanta_fluct=quanta, analytic_marginals_quanta=ana, analytic_marginals_keV=ana_keV,
           twoD=twoD, band=band, energy_dependence_fit=edep_fit, variants=variants, needed_for_23keV=need,
           ER=er_summary, event=evt, propagation=prop, runtime_s=time.time() - t0)
with open(OUT + "/results.json", "w") as f:
    json.dump(res, f, indent=1, default=float)
print(f"done in {time.time()-t0:.0f}s")
