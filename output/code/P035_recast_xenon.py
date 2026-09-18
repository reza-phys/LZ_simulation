"""
P035 -- Recasting XENONnT and PandaX-4T to inelastic dark matter: the exclusion they could publish tomorrow.

Run from the simulation root:  .venv/bin/python output/code/P035_recast_xenon.py

What it does
  1. Detector models.  XENONnT (3.1 t yr) and PandaX-4T (1.54 t yr) are treated as natural-xenon TPCs with the
     same dR/dE_R per tonne-year as LZ (target universality) and their own exposures.  Efficiencies:
        eff_LZ  : 0.96 plateau, erf turn-on 50% at 5.4 keV (sigma 2.5), erf roll-off 50% at 269.9 keV (sigma 12)  [LZ Fig. S2, P005 model]
        eff_hi  : 0.85 plateau, 50% at 300 keV (sigma 12)                                                              [variant]
        eff_edge(E_max): 0.96 plateau with a hard upper edge (E_max = 150, 200, 250, 270, 300 keV)                    [ROI-edge scan]
     Background per 2.84 t yr in the high-energy NR band: b = 2e-4 (LZ Fig. 5 within +-2 sigma of NR median at S1c>500),
     5.7e-4 (P016/P021 anchor for 200-270 keV), 1e-3 (round), 0.0106 (whole S1c>500 panel), scaled by exposure/2.84.
  2. Expected counts N(delta, kappa) for O1 isoscalar inelastic DM (WimPyDD, LZ/Anand unit coupling kappa = (c_1^s m_v^2)^2
     via lz.wd_c_from_anand), m = 400/1000/4000 GeV, delta = 100 keV .. kinematic ceiling; annual-average Baxter-2021 halo.
     90% CL upper limits on kappa (hence sigma_n) for n = 0 (Poisson N <= 2.303; Feldman-Cousins with b) and n = 1
     (Poisson-with-background; FC).  sigma_SI = kappa mu_N^2 / (pi m_v^4) (LZ supplement Eq., 'scalar normalisation').
  3. Discovery case: XENONnT sees one event at E2 = 200/230/260 keV.  Counting significance (Poisson with the tiny
     background) for XENONnT alone and LZ+XENONnT; joint (delta, kappa) extended likelihood in observed energy with
     the P021 resolution model, 68/90% 2-dof regions and the profile Z(delta).
  4. Existing XENON1T 2018 inelastic limit (Fig. S7 green curve, digitised by P015) vs LZ's 90% interval (P007 digitised
     Fig. 6/S7) at delta = 100-231 keV; recast check of XENON1T with a 40.9 keV ROI edge; ROI edges implied by the
     end-points of the XENON1T (230.8 keV) and PandaX-4T 2021 (303.3 keV) curves.
  5. Sensitivity to the ROI upper edge: N in XENONnT/PandaX-4T at LZ's one-event coupling for edges 150-300 keV.

Outputs: output/work/P035/*.csv, *.json, spectra_m*.npz (cache), figures/*.png
"""
import os, sys, json, math, time
import numpy as np
import pandas as pd
from scipy import special, stats, optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "output/code")
from common import lzcommon as lz

OUT = "output/work/P035"; FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
LOG = open(os.path.join(OUT, "run_log.txt"), "w")
def say(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.write(s + "\n"); LOG.flush()

# ------------------------------------------------------------------------------------------------
# 0. Inputs and constants
# ------------------------------------------------------------------------------------------------
MV = lz.M_V_GEV                      # 246.2 GeV  (Higgs vev, LZ/Anand unit coupling)
HBARC2_CM2 = 0.3894e-27              # (hbar c)^2 GeV^2 cm^2   [recalled, certain]
MN = lz.M_NUCLEON_GEV
LZ_EXPO = lz.LZ["exposure_tyr"]      # 2.84 t yr  [paper]
EXPO = {"LZ": LZ_EXPO, "XENONnT": 3.1, "PandaX-4T": 1.54}     # t yr  [XENONnT/PandaX exposures: recalled, certain; cited in LZ Intro]
E50_LO, E50_HI, PLATEAU = lz.LZ["E_50pct_low_keV"], lz.LZ["E_50pct_high_keV"], lz.LZ["eff_plateau"]   # 5.4, 269.9 keV, 0.96 [paper]
SIG_LO, SIG_HI = 2.5, 12.0           # keV, erf widths read from Fig. S2 (P005/P021)
B_PER_LZ = {"nrband_2sig": 2e-4, "band_200_270": 5.7e-4, "round_1e-3": 1e-3, "whole_panel": 0.0106}   # per 2.84 t yr
MASSES = [400.0, 1000.0, 4000.0]
E_GRID = np.arange(1.0, 451.0, 1.5)  # keV, true recoil energy
V_MAX_JUNE = lz.vmax_kms(v_e=lz.v_earth_kms(167))    # 809 km/s (16 June)
V_MAX_ANN = lz.vmax_kms()                            # 794.6 km/s (annual-mean Earth speed)

def erf_up(E, e50, sig):
    return 0.5 * (1.0 + special.erf((E - e50) / (math.sqrt(2.0) * sig)))

def eff_LZ(E):
    return PLATEAU * erf_up(E, E50_LO, SIG_LO) * (1.0 - erf_up(E, E50_HI, SIG_HI))

def eff_hi(E):
    return 0.85 * erf_up(E, 5.0, 2.0) * (1.0 - erf_up(E, 300.0, SIG_HI))

def eff_edge(E, emax, plateau=PLATEAU):
    return plateau * erf_up(E, E50_LO, SIG_LO) * (E <= emax)

EFFS = {"LZlike": eff_LZ, "hiedge": eff_hi}
EDGES = [150.0, 200.0, 250.0, 270.0, 300.0]

def ceiling_kev(m, v=V_MAX_JUNE):
    mu = lz.mu_red(m, lz.m_nucleus_gev(lz.A_XE_MEAN))
    return 0.5 * mu * (v / lz.C_KMS) ** 2 * 1e6

def delta_grid(m):
    c = ceiling_kev(m)
    d = list(np.arange(100.0, 300.0, 10.0)) + list(np.arange(300.0, c + 1e-9, 5.0))
    return np.array(d)

def sigma_unit_cm2(m):
    """sigma_SI (cm^2) for kappa = (c_1^s m_v^2)^2 = 1: sigma = mu_N^2/(pi m_v^4) (hbar c)^2  [LZ supplement]."""
    mu = m * MN / (m + MN)
    return mu ** 2 / (math.pi * MV ** 4) * HBARC2_CM2

# ------------------------------------------------------------------------------------------------
# 1. WimPyDD spectra (cached)
# ------------------------------------------------------------------------------------------------
# Halo.  lz.wd_halo() with no day returns WimPyDD's default single-day halo (day = modulation phase - T/4, i.e. the
# mean-Earth-speed / 'Sun-frame' halo, v_max = 794.6 km/s).  P021 calls this configuration 'sun' and finds that it is the
# one reproducing LZ's Fig. 6 upper edge at delta = 350 keV; we adopt it as baseline and use a 12-monthly-day annual
# average (P021's 'baseline'/'annual', which includes the June tail to 809 km/s) as the halo-tail bracket at 1000 GeV.
HAM = lz.wd_hamiltonian("O1s_unit", {1: lz.wd_c_from_anand(1.0 / MV ** 2, 0.0)})
HALO_SUN = lz.wd_halo()
def halo_annual12():
    days = np.arange(15.0, 365.0, 30.4375)
    hs = [lz.wd_halo(day_of_year=float(d)) for d in days]
    return hs[0][0], np.mean([h[1] for h in hs], axis=0)

def spectra(m, halo="sun", deltas=None, tag=""):
    fn = os.path.join(OUT, f"spectra_m{int(m)}{tag}.npz")
    if os.path.exists(fn):
        z = np.load(fn); return z["deltas"], z["E"], z["rate"]
    deltas = delta_grid(m) if deltas is None else np.asarray(deltas, float)
    hl = HALO_SUN if halo == "sun" else halo_annual12()
    rate = np.zeros((len(deltas), len(E_GRID)))
    t0 = time.time()
    part = fn.replace(".npz", "_partial.npz")
    done = 0
    if os.path.exists(part):
        zp = np.load(part)
        if len(zp["deltas"]) == len(deltas) and np.allclose(zp["deltas"], deltas):
            rate[:] = zp["rate"]; done = int(zp["done"])
    for i, d in enumerate(deltas):
        if i < done: continue
        r = lz.wd_rate(HAM, m, E_GRID, halo=hl, delta_kev=float(d))
        rate[i] = np.clip(np.nan_to_num(r), 0.0, None)
        say(f"  m={m:.0f}{tag} delta={d:.0f}: integral {np.trapezoid(rate[i], E_GRID):.4g} /t/yr  ({time.time()-t0:.0f}s)")
        np.savez(part, deltas=deltas, rate=rate, done=i + 1)          # incremental cache
    np.savez(fn, deltas=deltas, E=E_GRID, rate=rate)
    return deltas, E_GRID, rate

def integ(y, w=None):
    return float(np.trapezoid(y if w is None else y * w, E_GRID))

SPEC = {m: spectra(m) for m in MASSES}
ANN_DELTAS = [300.0, 320.0, 340.0, 350.0, 360.0, 370.0, 375.0, 380.0, 385.0]       # reduced grid: the 12-day halo is ~10x slower per point
SPEC_ANN = spectra(1000.0, halo="annual12", deltas=ANN_DELTAS, tag="_annual12")     # halo-tail bracket
ANN = {float(d): SPEC_ANN[2][i] for i, d in enumerate(SPEC_ANN[0])}
SPEC_LOWD = spectra(1000.0, deltas=[0.0, 50.0], tag="_lowdelta")                    # for the XENON1T elastic check

# validation against P021's S_tot_unit (accepted events per unit kappa in LZ, 2.84 t yr, LZ-like efficiency)
p21 = pd.read_csv("output/work/P021/P021_scan_baseline.csv")
p21s = p21[(p21.tag == "s") & (p21.m_GeV == 1000)].set_index("delta_keV")          # P021 baseline = 12-day annual halo
p21all = pd.read_csv("output/work/P021/P021_scan_all.csv")
p21sun = p21all[(p21all.tag == "s") & (p21all.m_GeV == 1000) & (p21all.cfg == "sun")].set_index("delta_keV")
val = []
d1000, _, r1000 = SPEC[1000.0]
for i, d in enumerate(d1000):
    if d in p21s.index:
        mine = LZ_EXPO * integ(r1000[i], eff_LZ(E_GRID))
        mine_ann = LZ_EXPO * integ(ANN[float(d)], eff_LZ(E_GRID)) if float(d) in ANN else np.nan
        val.append(dict(delta_keV=d, N_unit_LZ_sun_this=mine, N_unit_LZ_P021_sun=p21sun.loc[d, "S_tot_unit"], ratio_sun=mine / p21sun.loc[d, "S_tot_unit"],
                        N_unit_LZ_annual12_this=mine_ann, N_unit_LZ_P021_annual=p21s.loc[d, "S_tot_unit"], ratio_annual=mine_ann / p21s.loc[d, "S_tot_unit"],
                        annual_over_sun_this=mine_ann / mine if mine > 0 else np.nan))
val = pd.DataFrame(val); val.to_csv(os.path.join(OUT, "validation_vs_P021.csv"), index=False)
say("validation vs P021 S_tot_unit (1000 GeV):"); say(val.to_string(index=False))

# ------------------------------------------------------------------------------------------------
# 2. Expected counts per unit kappa, limits
# ------------------------------------------------------------------------------------------------
def fc_upper(n_obs, b, cl=0.90, s_max=20.0, ds=0.01, n_max=60):
    """Feldman-Cousins unified 90% interval for Poisson n with known background b; returns (lower, upper) on s."""
    s_grid = np.arange(0.0, s_max + ds / 2, ds)
    ns = np.arange(0, n_max + 1)
    accepted = np.zeros((len(s_grid), len(ns)), bool)
    for i, s in enumerate(s_grid):
        p = stats.poisson.pmf(ns, s + b)
        s_best = np.maximum(0.0, ns - b)
        p_best = stats.poisson.pmf(ns, s_best + b)
        R = p / np.where(p_best > 0, p_best, 1e-300)
        order = np.argsort(-R)
        cum = np.cumsum(p[order])
        k = np.searchsorted(cum, cl) + 1
        accepted[i, order[:k]] = True
    ok = s_grid[accepted[:, n_obs]]
    return (float(ok.min()), float(ok.max())) if len(ok) else (np.nan, np.nan)

def pois_upper(n_obs, b, cl=0.90):
    """Classical Poisson-with-background upper limit: sum_{k<=n} P(k | s+b) = 1-cl  (2.303 for n=0, any b)."""
    f = lambda s: stats.poisson.cdf(n_obs, s + b) - (1 - cl)
    return float(optimize.brentq(f, 0.0, 50.0))

rows = []
NUNIT = {}     # (m, exp, effname) -> array over deltas
for m in MASSES:
    deltas, E, rate = SPEC[m]
    su = sigma_unit_cm2(m)
    for effname, eff in EFFS.items():
        w = eff(E)
        per_tyr = np.array([integ(rate[i], w) for i in range(len(deltas))])
        for exp, X in EXPO.items():
            NUNIT[(m, exp, effname)] = X * per_tyr
    for i, d in enumerate(deltas):
        row = dict(m_GeV=m, delta_keV=d, sigma_unit_cm2=su,
                   E_lo_keV=lz.E_R_range_keV(m, V_MAX_ANN, delta_kev=d)[0] if d < ceiling_kev(m, V_MAX_ANN) else np.nan)
        for effname in EFFS:
            for exp in EXPO:
                row[f"Nunit_{exp}_{effname}"] = NUNIT[(m, exp, effname)][i]
        rows.append(row)
tab = pd.DataFrame(rows)

# limits at 1000 GeV (and 400/4000) for XENONnT, PandaX-4T and combined, both efficiencies
lim_rows = []
for m in MASSES:
    deltas = SPEC[m][0]; su = sigma_unit_cm2(m)
    for effname in EFFS:
        NX = NUNIT[(m, "XENONnT", effname)]; NP = NUNIT[(m, "PandaX-4T", effname)]; NL = NUNIT[(m, "LZ", effname)]
        for i, d in enumerate(deltas):
            r = dict(m_GeV=m, eff=effname, delta_keV=d, Nunit_X=NX[i], Nunit_P=NP[i], Nunit_LZ=NL[i])
            for tag, N in [("X", NX[i]), ("P", NP[i]), ("XP", NX[i] + NP[i])]:
                r[f"kappa90_n0_{tag}"] = 2.302585 / N if N > 0 else np.inf
                r[f"sigma90_n0_{tag}_cm2"] = r[f"kappa90_n0_{tag}"] * su
                r[f"kappa90_n1_{tag}"] = 3.889720 / N if N > 0 else np.inf      # Poisson, b -> 0
                r[f"sigma90_n1_{tag}_cm2"] = r[f"kappa90_n1_{tag}"] * su
            lim_rows.append(r)
lim = pd.DataFrame(lim_rows)
lim.to_csv(os.path.join(OUT, "projected_limits.csv"), index=False)
tab.to_csv(os.path.join(OUT, "Nunit_table.csv"), index=False)

# FC and Poisson-with-background event-count limits (exposure-scaled backgrounds)
cnt = []
for exp in ["XENONnT", "PandaX-4T", "XENONnT+PandaX-4T"]:
    X = EXPO["XENONnT"] + EXPO["PandaX-4T"] if "+" in exp else EXPO[exp]
    for bname, b0 in B_PER_LZ.items():
        b = b0 * X / LZ_EXPO
        for n in [0, 1]:
            fc = fc_upper(n, b)
            cnt.append(dict(experiment=exp, exposure_tyr=X, bkg_model=bname, b=b, n_obs=n, poisson_UL90=pois_upper(n, b), FC90_lo=fc[0], FC90_hi=fc[1]))
cnt = pd.DataFrame(cnt); cnt.to_csv(os.path.join(OUT, "count_limits_FC.csv"), index=False)
say("count limits:"); say(cnt.to_string(index=False))

# ------------------------------------------------------------------------------------------------
# 3. Comparison with LZ: P007 digitised intervals and P021 kappa_hat(delta), 1000 GeV
# ------------------------------------------------------------------------------------------------
lzint = pd.read_csv("output/work/P007/lz_intervals_digitised.csv")
def lz_edge(d, col):
    sub = lzint.dropna(subset=[col]); sub = sub[sub[col] > 0]
    return float(np.exp(np.interp(d, sub.delta_keV, np.log(sub[col]))))

d1000 = SPEC[1000.0][0]; su1000 = sigma_unit_cm2(1000.0)
cmp_rows = []
for i, d in enumerate(d1000):
    NL = NUNIT[(1000.0, "LZ", "LZlike")][i]
    kap1 = 1.0 / NL if NL > 0 else np.nan          # one-event coupling (mu_hat = 1.0), this work
    r = dict(delta_keV=d, Nunit_LZ=NL, kappa_1event=kap1, sigma_1event_cm2=kap1 * su1000)
    if d in p21s.index:
        r.update(kappa_hat_P021=p21s.loc[d, "kappa_hat"], kappa_90lo_P021=p21s.loc[d, "kappa_90lo"], kappa_90hi_P021=p21s.loc[d, "kappa_90hi"],
                 kappa_68lo_P021=p21s.loc[d, "kappa_68lo"], kappa_68hi_P021=p21s.loc[d, "kappa_68hi"], Z_P021=p21s.loc[d, "Z"],
                 kappa_hat_P021sun=p21sun.loc[d, "kappa_hat"], kappa_90lo_P021sun=p21sun.loc[d, "kappa_90lo"], kappa_90hi_P021sun=p21sun.loc[d, "kappa_90hi"])
    if 100 <= d <= 350:
        r.update(LZ_fig6_upper=lz_edge(d, "c1s_mv2_sq_upper"), LZ_fig6_lower=lz_edge(d, "c1s_mv2_sq_lower") if d >= 100 else np.nan)
    for effname in EFFS:
        NX = NUNIT[(1000.0, "XENONnT", effname)][i]; NP = NUNIT[(1000.0, "PandaX-4T", effname)][i]
        r[f"N_X_at_1event_{effname}"] = kap1 * NX; r[f"N_P_at_1event_{effname}"] = kap1 * NP
        r[f"N_XP_at_1event_{effname}"] = kap1 * (NX + NP)
        r[f"P0_XP_{effname}"] = math.exp(-kap1 * (NX + NP))
        for tag, N in [("X", NX), ("P", NP), ("XP", NX + NP)]:
            klim = 2.302585 / N if N > 0 else np.inf
            r[f"kappa90_n0_{tag}_{effname}"] = klim
            ratio = klim / kap1 if kap1 > 0 else np.nan          # limit / one-event coupling: exposure x efficiency ratio only
            r[f"lim_over_kappa1event_{tag}_{effname}"] = ratio
            # fraction of LZ's 90% interval (in log kappa) excluded, computed normalisation-consistently: the projected
            # limit sits at ratio x kappa_hat in whichever normalisation kappa_hat and the interval share (P021 sun / annual)
            for src, P in [("sun", p21sun), ("ann", p21s)]:
                if d in P.index and P.loc[d, "kappa_90hi"] > 0 and P.loc[d, "kappa_90lo"] > 0 and P.loc[d, "kappa_hat"] > 0 and np.isfinite(ratio):
                    lo, hi, kh = math.log(P.loc[d, "kappa_90lo"]), math.log(P.loc[d, "kappa_90hi"]), math.log(P.loc[d, "kappa_hat"] * ratio)
                    r[f"frac_LZ90_excluded_{tag}_{effname}_{src}"] = float(np.clip((hi - kh) / (hi - lo), 0, 1))
                    r[f"lim_over_kappahat_{tag}_{effname}_{src}"] = ratio
                    r[f"LZ90hi_over_kappahat_{src}"] = math.exp(hi) / P.loc[d, "kappa_hat"]
            if 100 <= d <= 350:
                r[f"lim_over_LZupper_{tag}_{effname}"] = klim / r["LZ_fig6_upper"]
    # halo-tail bracket (12-monthly-day annual average), LZ-like efficiency
    ia = integ(ANN[float(d)], eff_LZ(E_GRID)) if float(d) in ANN else np.nan
    NLa = LZ_EXPO * ia; NXa = EXPO["XENONnT"] * ia; NPa = EXPO["PandaX-4T"] * ia
    r.update(Nunit_LZ_annual12=NLa, kappa_1event_annual12=1.0 / NLa if NLa > 0 else np.nan, sigma_1event_annual12_cm2=su1000 / NLa if NLa > 0 else np.nan,
             sigma90_n0_XP_annual12_cm2=2.302585 / (NXa + NPa) * su1000 if NXa + NPa > 0 else np.nan,
             sigma90_n0_XP_sun_cm2=2.302585 / (NUNIT[(1000.0, "XENONnT", "LZlike")][i] + NUNIT[(1000.0, "PandaX-4T", "LZlike")][i]) * su1000)
    cmp_rows.append(r)
cmp = pd.DataFrame(cmp_rows); cmp.to_csv(os.path.join(OUT, "comparison_with_LZ_1000GeV.csv"), index=False)
say("comparison with LZ (1000 GeV, LZ-like eff):")
say(cmp[["delta_keV", "Nunit_LZ", "kappa_1event", "kappa_hat_P021sun", "kappa_hat_P021", "N_XP_at_1event_LZlike", "P0_XP_LZlike",
         "lim_over_kappa1event_X_LZlike", "lim_over_kappa1event_XP_LZlike", "LZ90hi_over_kappahat_sun", "frac_LZ90_excluded_X_LZlike_sun", "frac_LZ90_excluded_XP_LZlike_sun",
         "frac_LZ90_excluded_XP_LZlike_ann", "frac_LZ90_excluded_XP_hiedge_sun", "lim_over_LZupper_XP_LZlike", "Nunit_LZ_annual12", "sigma90_n0_XP_sun_cm2", "sigma90_n0_XP_annual12_cm2"]].to_string(index=False))

# Reduction of LZ's local significance if XENONnT and PandaX-4T (LZ-like eff) see nothing:
# P021: q0 = 2[ln(f/b) - 1 + b/f] with f = accepted-spectrum density at 248 keV per unit LZ signal.  Adding zero-count
# exposures multiplies the total expectation per unit kappa by R = N_tot/N_LZ; the event density is unchanged, so
# q0' = 2[ln(f/(R b)) - 1 + R b/f] (asymptotic, single event, b << f).
def q0_single(fb):
    return 2.0 * (math.log(fb) - 1.0 + 1.0 / fb)
zred = []
for d in [300.0, 350.0, 370.0, 380.0, 385.0]:
    Zl = p21s.loc[d, "Z"]; q0 = Zl ** 2
    fb = optimize.brentq(lambda x: q0_single(x) - q0, 1.0 + 1e-9, 1e12)
    i = list(d1000).index(d)
    for effname in EFFS:
        R = (NUNIT[(1000.0, "LZ", effname)][i] + NUNIT[(1000.0, "XENONnT", effname)][i] + NUNIT[(1000.0, "PandaX-4T", effname)][i]) / NUNIT[(1000.0, "LZ", "LZlike")][i]
        q0n = q0_single(fb / R) if fb / R > 1 else 0.0
        zred.append(dict(delta_keV=d, eff=effname, Z_LZ_P021=Zl, R_total_over_LZ=R, Z_combined_null=math.sqrt(max(q0n, 0)), kappa_hat_combined=p21s.loc[d, "kappa_hat"] / R))
zred = pd.DataFrame(zred); zred.to_csv(os.path.join(OUT, "Z_reduction_if_null.csv"), index=False)
say("Z reduction if others null:"); say(zred.to_string(index=False))

# ------------------------------------------------------------------------------------------------
# 4. Discovery case: XENONnT sees one event at E2 in 200-270 keV
# ------------------------------------------------------------------------------------------------
disc = {}
for bname, b0 in B_PER_LZ.items():
    bL = b0; bX = b0 * EXPO["XENONnT"] / LZ_EXPO
    pX = 1 - stats.poisson.cdf(0, bX)                    # P(>=1 | bX) XENONnT alone
    pC = 1 - stats.poisson.cdf(1, bL + bX)               # P(>=2 | bL+bX)
    disc[bname] = dict(b_LZ=bL, b_X=bX, p_X_alone=pX, Z_X_alone=float(stats.norm.isf(pX)), p_comb_2events=pC, Z_comb_2events=float(stats.norm.isf(pC)))
say("discovery counting:"); say(json.dumps(disc, indent=1))

# joint (delta, kappa) extended likelihood in observed energy: events at 248 keV (LZ) and E2 (XENONnT), zero counts
# in 125-200 keV in both; resolution sigma_E = 11 sqrt(E/248) keV (P021/P009); flat b_H = 5.7e-4 per 2.84 t yr in 200-270 keV.
def smear(a, E):
    sig = 11.0 * np.sqrt(np.maximum(E, 1.0) / 248.0)
    K = np.exp(-0.5 * ((E[:, None] - E[None, :]) / sig[None, :]) ** 2) / (math.sqrt(2 * math.pi) * sig[None, :])
    dE = E[1] - E[0]
    return K @ (a * dE)

B_H = 5.7e-4
def joint_scan(E2_list, m=1000.0):
    deltas, E, rate = SPEC[m]
    wL = eff_LZ(E)
    out = {}
    dens = []     # per delta: (S_LZ_total, S_X_total, f_LZ(248), f_X(E2), S_LZ_125_200, S_X_125_200) per unit kappa
    for i, d in enumerate(deltas):
        aL = rate[i] * wL * LZ_EXPO; aX = rate[i] * wL * EXPO["XENONnT"]
        oL = smear(aL, E); oX = smear(aX, E)
        m1 = (E >= 125) & (E < 200); m2 = (E >= 200) & (E <= 270)
        dens.append(dict(SL=integ(oL, m2), SX=integ(oX, m2), fL=float(np.interp(248.0, E, oL)),
                         fX={E2: float(np.interp(E2, E, oX)) for E2 in E2_list}, ZL=integ(oL, m1), ZX=integ(oX, m1)))
    kap = np.logspace(-9, 3, 1201)
    bL = B_H / 70.0; bX = B_H * EXPO["XENONnT"] / LZ_EXPO / 70.0    # per keV densities
    BL = B_H; BX = B_H * EXPO["XENONnT"] / LZ_EXPO
    for case in ["LZ only"] + [f"LZ+X({E2:.0f})" for E2 in E2_list]:
        LL = np.full((len(deltas), len(kap)), -np.inf)
        for i, dd in enumerate(dens):
            if case == "LZ only":
                ll = np.log(kap * dd["fL"] + bL) - kap * (dd["SL"] + dd["ZL"]) - BL
            else:
                E2 = float(case.split("(")[1][:-1])
                ll = (np.log(kap * dd["fL"] + bL) - kap * (dd["SL"] + dd["ZL"]) - BL
                      + np.log(kap * dd["fX"][E2] + bX) - kap * (dd["SX"] + dd["ZX"]) - BX)
            LL[i] = ll
        ll0 = (math.log(bL) - BL) + ((math.log(bX) - BX) if case != "LZ only" else 0.0)
        imax = np.unravel_index(np.nanargmax(LL), LL.shape)
        q0 = 2 * (LL[imax] - ll0)
        dm2 = 2 * (LL[imax] - LL)
        prof = LL.max(axis=1); Zprof = np.sqrt(np.maximum(2 * (prof - ll0), 0))
        in68 = dm2 <= 2.30; in90 = dm2 <= 4.61
        d68 = deltas[in68.any(axis=1)]; d90 = deltas[in90.any(axis=1)]
        k68 = kap[in68.any(axis=0)]; k90 = kap[in90.any(axis=0)]
        out[case] = dict(delta_best=float(deltas[imax[0]]), kappa_best=float(kap[imax[1]]), Z_best=float(math.sqrt(max(q0, 0))),
                         delta68=[float(d68.min()), float(d68.max())] if len(d68) else None, delta90=[float(d90.min()), float(d90.max())] if len(d90) else None,
                         kappa68=[float(k68.min()), float(k68.max())] if len(k68) else None, kappa90=[float(k90.min()), float(k90.max())] if len(k90) else None,
                         Zprof={float(d): float(z) for d, z in zip(deltas, Zprof)},
                         kappahat_prof={float(d): float(kap[j]) for d, j in zip(deltas, LL.argmax(axis=1))})
        np.savez(os.path.join(OUT, f"joint_surface_{case.replace('+','_').replace('(','').replace(')','').replace(' ','_')}.npz"), deltas=deltas, kappa=kap, dm2=dm2)
    return out
JOINT = joint_scan([200.0, 230.0, 260.0])
for c, v in JOINT.items():
    say(f"joint {c}: best delta {v['delta_best']}, kappa {v['kappa_best']:.3g}, Z {v['Z_best']:.2f}, delta68 {v['delta68']}, delta90 {v['delta90']}, kappa68 {v['kappa68']}")
    say("   Zprof at 300/350/380:", {d: round(v['Zprof'][d], 2) for d in [300.0, 350.0, 380.0]})
json.dump(JOINT, open(os.path.join(OUT, "joint_fit_results.json"), "w"), indent=1)

# ------------------------------------------------------------------------------------------------
# 5. Existing XENON1T 2018 (and PandaX-4T 2021) inelastic limits vs LZ
# ------------------------------------------------------------------------------------------------
s7 = json.load(open("output/work/P015/figS7_curves.json"))["curves_points"]
x1t = np.array(s7["XENON1T 2018 (green)"]); px = np.array(s7["PandaX-4T 2021 (blue)"])
def curve(pts, d):
    return float(np.exp(np.interp(d, pts[:, 0], np.log(pts[:, 1]))))
x1t_rows = []
for d in [100.0, 150.0, 200.0, 220.0, 230.8]:
    s_x = curve(x1t, d); up = lz_edge(d, "sigmaSI_upper_cm2"); lo = lz_edge(d, "sigmaSI_lower_cm2")
    i = int(np.argmin(np.abs(d1000 - d)))
    kap_x = s_x / su1000
    NL = kap_x * np.interp(d, d1000, NUNIT[(1000.0, "LZ", "LZlike")])
    x1t_rows.append(dict(delta_keV=d, sigma_X1T_cm2=s_x, LZ_upper_cm2=up, LZ_lower_cm2=lo, X1T_over_LZ_upper=s_x / up, X1T_over_LZ_lower=s_x / lo,
                         N_LZ_events_at_X1T_limit=NL))
x1t_df = pd.DataFrame(x1t_rows); x1t_df.to_csv(os.path.join(OUT, "xenon1t_vs_LZ.csv"), index=False)
say("XENON1T 2018 vs LZ:"); say(x1t_df.to_string(index=False))

# recast check: 1.0 t yr, 0.9 plateau, hard edge 40.9 keV (recalled XENON1T 2018 NR ROI 4.9-40.9 keV; likely), n = 0 -> 2.30 events
X1T_EDGE = 40.9; X1T_EXPO = 1.0
rec = []
deltas, E, rate = SPEC[1000.0]
pairs = [(0.0, SPEC_LOWD[2][0]), (50.0, SPEC_LOWD[2][1])] + [(d, rate[i]) for i, d in enumerate(deltas) if d <= 240]
for d, rr in pairs:
    w = 0.9 * erf_up(E, 4.9, 1.0) * (E <= X1T_EDGE)
    N = X1T_EXPO * integ(rr, w)
    rec.append(dict(delta_keV=d, Nunit_X1T_recast=N, sigma90_recast_cm2=(2.302585 / N * su1000) if N > 0 else np.inf,
                    sigma_X1T_digitised_cm2=curve(x1t, max(d, x1t[0, 0])) if d <= 230.8 else np.nan))
rec = pd.DataFrame(rec); rec["ratio_recast_over_digitised"] = rec.sigma90_recast_cm2 / rec.sigma_X1T_digitised_cm2
# shape check: rescale the recast to the digitised curve at delta = 0 (elastic; the published XENON1T 1 TeV SI limit) and
# compare the delta-dependence only
norm0 = float(rec.loc[rec.delta_keV == 0.0, "ratio_recast_over_digitised"].iloc[0])
rec["shape_ratio_normalised_at_0"] = rec.ratio_recast_over_digitised / norm0
rec.to_csv(os.path.join(OUT, "xenon1t_recast_check.csv"), index=False)
say("XENON1T recast check:"); say(rec.to_string(index=False))

# ROI edges implied by the curve end-points: E_-(delta_end) for several v_max choices (1 TeV)
impl = []
for name, dend in [("XENON1T 2018", 230.8), ("PandaX-4T 2021", 303.3)]:
    for vname, v in [("annual 794.6", V_MAX_ANN), ("June 809.1", V_MAX_JUNE), ("v0+vesc 776", 232.0 + 544.0), ("vE=232 + vesc=544 + 15", 791.0)]:
        impl.append(dict(curve=name, delta_end_keV=dend, vmax_label=vname, vmax_kms=v, E_minus_keV=lz.E_R_range_keV(1000.0, v, delta_kev=dend)[0]))
impl = pd.DataFrame(impl); impl.to_csv(os.path.join(OUT, "implied_roi_edges.csv"), index=False)
say("implied ROI edges:"); say(impl.to_string(index=False))

# ------------------------------------------------------------------------------------------------
# 6. ROI-edge sensitivity (P005 point): counts at LZ's one-event coupling vs hard upper edge
# ------------------------------------------------------------------------------------------------
edge_rows = []
for m in MASSES:
    deltas, E, rate = SPEC[m]
    for d in [250.0, 300.0, 350.0, 380.0]:
        if d > deltas.max(): continue
        i = int(np.argmin(np.abs(deltas - d)))
        NL = LZ_EXPO * integ(rate[i], eff_LZ(E))
        if NL <= 0: continue
        kap1 = 1.0 / NL
        r = dict(m_GeV=m, delta_keV=deltas[i], kappa_1event=kap1)
        for e in EDGES:
            w = eff_edge(E, e)
            r[f"N_X_edge{int(e)}"] = kap1 * EXPO["XENONnT"] * integ(rate[i], w)
            r[f"N_P_edge{int(e)}"] = kap1 * EXPO["PandaX-4T"] * integ(rate[i], w)
        r["N_X_LZlike"] = kap1 * EXPO["XENONnT"] * integ(rate[i], eff_LZ(E)); r["N_X_hiedge"] = kap1 * EXPO["XENONnT"] * integ(rate[i], eff_hi(E))
        r["N_P_LZlike"] = kap1 * EXPO["PandaX-4T"] * integ(rate[i], eff_LZ(E)); r["N_P_hiedge"] = kap1 * EXPO["PandaX-4T"] * integ(rate[i], eff_hi(E))
        edge_rows.append(r)
edge = pd.DataFrame(edge_rows); edge.to_csv(os.path.join(OUT, "roi_edge_scan.csv"), index=False)
say("ROI edge scan:"); say(edge.to_string(index=False))

# ------------------------------------------------------------------------------------------------
# 7. Summary JSON
# ------------------------------------------------------------------------------------------------
def at(df, d, col, key="delta_keV"):
    return float(df.loc[np.argmin(np.abs(df[key].values - d)), col])
c1000 = cmp
summary = dict(
    inputs=dict(exposures_tyr=EXPO, eff_LZlike=dict(plateau=PLATEAU, E50=[E50_LO, E50_HI], sig=[SIG_LO, SIG_HI]),
                eff_hiedge=dict(plateau=0.85, E50=[5.0, 300.0], sig=[2.0, SIG_HI]), bkg_per_2p84_tyr=B_PER_LZ, halo="Baxter-2021 SHM annual average (lz.wd_halo)",
                coupling="O1 isoscalar, kappa=(c_1^s m_v^2)^2, WimPyDD c0 = 2/m_v^2 (P003)", sigma_unit_cm2_1000GeV=su1000,
                E_grid_keV=[float(E_GRID[0]), float(E_GRID[-1]), float(E_GRID[1] - E_GRID[0])]),
    validation_vs_P021=dict(sun_ratio_le350=[float(val[val.delta_keV <= 350].ratio_sun.min()), float(val[val.delta_keV <= 350].ratio_sun.max())],
                            sun_ratio_370_385=[float(val[(val.delta_keV >= 370) & (val.delta_keV <= 385)].ratio_sun.min()), float(val[(val.delta_keV >= 370) & (val.delta_keV <= 385)].ratio_sun.max())],
                            annual_ratio_le350=[float(val[val.delta_keV <= 350].ratio_annual.min()), float(val[val.delta_keV <= 350].ratio_annual.max())],
                            annual_ratio_370_385=[float(val[(val.delta_keV >= 370) & (val.delta_keV <= 385)].ratio_annual.min()), float(val[(val.delta_keV >= 370) & (val.delta_keV <= 385)].ratio_annual.max())],
                            annual_over_sun={str(int(d)): float(val.loc[val.delta_keV == d, "annual_over_sun_this"].iloc[0]) for d in ANN_DELTAS}),
    halo_bracket_1000GeV={str(int(d)): dict(sigma_1event_sun=at(c1000, d, "sigma_1event_cm2"), sigma_1event_annual12=at(c1000, d, "sigma_1event_annual12_cm2"),
                                            sigma90_XP_sun=at(c1000, d, "sigma90_n0_XP_sun_cm2"), sigma90_XP_annual12=at(c1000, d, "sigma90_n0_XP_annual12_cm2")) for d in [300, 350, 370, 380]},
    frac_LZ90_excluded_1000GeV={str(int(d)): {k: at(c1000, d, k) for k in ["frac_LZ90_excluded_X_LZlike_sun", "frac_LZ90_excluded_XP_LZlike_sun", "frac_LZ90_excluded_XP_LZlike_ann",
                                                                          "frac_LZ90_excluded_XP_hiedge_sun", "LZ90hi_over_kappahat_sun", "LZ90hi_over_kappahat_ann"]} for d in [250, 300, 350, 370, 380]},
    N_at_LZ_one_event_1000GeV={str(int(d)): dict(X_LZlike=at(c1000, d, "N_X_at_1event_LZlike"), P_LZlike=at(c1000, d, "N_P_at_1event_LZlike"),
                                                 XP_LZlike=at(c1000, d, "N_XP_at_1event_LZlike"), P0_XP_LZlike=at(c1000, d, "P0_XP_LZlike"),
                                                 XP_hiedge=at(c1000, d, "N_XP_at_1event_hiedge"), P0_XP_hiedge=at(c1000, d, "P0_XP_hiedge")) for d in [250, 300, 350, 366, 380]},
    zero_event_limits_1000GeV={str(int(d)): dict(sigma90_X_LZlike=at(lim[(lim.m_GeV == 1000) & (lim.eff == "LZlike")].reset_index(), d, "sigma90_n0_X_cm2"),
                                                 sigma90_XP_LZlike=at(lim[(lim.m_GeV == 1000) & (lim.eff == "LZlike")].reset_index(), d, "sigma90_n0_XP_cm2"),
                                                 sigma90_XP_hiedge=at(lim[(lim.m_GeV == 1000) & (lim.eff == "hiedge")].reset_index(), d, "sigma90_n0_XP_cm2"),
                                                 lim_over_kappa1event_X=at(c1000, d, "lim_over_kappa1event_X_LZlike"), lim_over_kappa1event_XP=at(c1000, d, "lim_over_kappa1event_XP_LZlike"),
                                                 frac_LZ90_excluded_X_sun=at(c1000, d, "frac_LZ90_excluded_X_LZlike_sun") if d in p21s.index else None,
                                                 frac_LZ90_excluded_XP_sun=at(c1000, d, "frac_LZ90_excluded_XP_LZlike_sun") if d in p21s.index else None,
                                                 lim_over_LZupper_XP=at(c1000, d, "lim_over_LZupper_XP_LZlike") if d <= 350 else None,
                                                 LZ_upper_sigma=lz_edge(d, "sigmaSI_upper_cm2") if d <= 350 else None) for d in [250, 300, 350, 380]},
    count_limits=cnt.to_dict(orient="records"), Z_reduction_if_null=zred.to_dict(orient="records"), discovery_counting=disc,
    joint_fit={k: {kk: vv for kk, vv in v.items() if kk not in ("Zprof", "kappahat_prof")} for k, v in JOINT.items()},
    joint_Zprof_300_350_380={k: [v["Zprof"][300.0], v["Zprof"][350.0], v["Zprof"][380.0]] for k, v in JOINT.items()},
    xenon1t_vs_LZ=x1t_df.to_dict(orient="records"), xenon1t_recast_check=dict(ratio_range=[float(rec.ratio_recast_over_digitised.min()), float(rec.ratio_recast_over_digitised.max())],
                                                                              rows=rec.to_dict(orient="records")),
    implied_roi_edges=impl.to_dict(orient="records"), roi_edge_scan=edge.to_dict(orient="records"),
    ceilings_keV={str(int(m)): ceiling_kev(m) for m in MASSES}, vmax_kms=dict(annual=V_MAX_ANN, june=V_MAX_JUNE))
json.dump(summary, open(os.path.join(OUT, "P035_summary.json"), "w"), indent=1, default=float)

# ------------------------------------------------------------------------------------------------
# 8. Figures (Okabe-Ito, fixed order; one axis per panel)
# ------------------------------------------------------------------------------------------------
OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442", blue="#0072B2", verm="#D55E00", purple="#CC79A7", grey="#7f7f7f")
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False})

# Fig 1: (delta, sigma) plane
fig, ax = plt.subplots(figsize=(7.2, 5.2))
dd = np.linspace(100, 350, 200)
ax.fill_between(dd, [lz_edge(x, "sigmaSI_lower_cm2") for x in dd], [lz_edge(x, "sigmaSI_upper_cm2") for x in dd], color=OI["grey"], alpha=0.3, lw=0, label="LZ 2026 two-sided 90% (Fig. S7, digitised P007)")
sel = cmp.dropna(subset=["kappa_hat_P021"]); sel = sel[sel.kappa_hat_P021 > 0]
ax.plot(sel.delta_keV, sel.kappa_hat_P021 * su1000, color=OI["black"], lw=1.6, label="LZ best fit κ̂(δ), P021 (μ̂ = 1 event)")
ax.fill_between(sel.delta_keV, sel.kappa_68lo_P021.clip(1e-12) * su1000, sel.kappa_68hi_P021 * su1000, color=OI["black"], alpha=0.12, lw=0, label="P021 68% band")
L = lim[(lim.m_GeV == 1000) & (lim.eff == "LZlike")]; Lh = lim[(lim.m_GeV == 1000) & (lim.eff == "hiedge")]
ax.plot(L.delta_keV, L.sigma90_n0_X_cm2, color=OI["blue"], lw=2, label="XENONnT 3.1 t·yr, 0 events, LZ-like ε (this work)")
ax.plot(L.delta_keV, L.sigma90_n0_P_cm2, color=OI["orange"], lw=2, label="PandaX-4T 1.54 t·yr, 0 events, LZ-like ε")
ax.plot(L.delta_keV, L.sigma90_n0_XP_cm2, color=OI["verm"], lw=2, ls="--", label="XENONnT+PandaX-4T, 0 events")
ax.plot(Lh.delta_keV, Lh.sigma90_n0_XP_cm2, color=OI["verm"], lw=1.2, ls=":", label="XENONnT+PandaX-4T, 0 events, ε: 0.85 / 50% at 300 keV")
ann = cmp.dropna(subset=["sigma90_n0_XP_annual12_cm2"])
ax.plot(ann.delta_keV, ann.sigma90_n0_XP_annual12_cm2, color=OI["yellow"], lw=1.2, ls="--", label="XENONnT+PandaX-4T, 0 events, 12-day annual halo (tail bracket)")
ax.plot(L.delta_keV, L.sigma90_n1_X_cm2, color=OI["sky"], lw=1.2, ls="-.", label="XENONnT, 1 event (N ≤ 3.89)")
ax.plot(x1t[:, 0], x1t[:, 1], color=OI["green"], lw=1.6, label="XENON1T 2018 (Fig. S7 green, PICO recast; ends 231 keV)")
ax.plot(px[:, 0], px[:, 1], color=OI["purple"], lw=1.2, label="PandaX-4T 2021 (Fig. S7 blue; ends 303 keV)")
ax.set_yscale("log"); ax.set_xlim(100, 400); ax.set_ylim(1e-47, 1e-34)
ax.set_xlabel("mass splitting δ [keV]"); ax.set_ylabel("σ$_{SI}$-equivalent (scalar normalisation) [cm²]")
ax.set_title("Inelastic O$_1$ isoscalar, m$_χ$ = 1 TeV: what XENONnT and PandaX-4T could publish tomorrow", fontsize=9.5)
ax.axvline(ceiling_kev(1000.0), color=OI["grey"], lw=0.8, ls=":"); ax.text(ceiling_kev(1000.0) - 2, 3e-47, "June ceiling δ$_{max}$(1 TeV)", rotation=90, fontsize=7, ha="right", va="bottom", color=OI["grey"])
ax.legend(fontsize=6.6, loc="upper left", frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "delta_sigma_plane_1TeV.png"), dpi=170); plt.close(fig)

# Fig 2: expected counts at LZ's one-event coupling vs ROI edge
fig, ax = plt.subplots(figsize=(6.4, 4.2))
e1000 = edge[edge.m_GeV == 1000]
cols = [OI["blue"], OI["orange"], OI["green"], OI["verm"]]
for (_, r), c in zip(e1000.iterrows(), cols):
    y = [r[f"N_X_edge{int(e)}"] + r[f"N_P_edge{int(e)}"] for e in EDGES]
    ax.plot(EDGES, y, "o-", color=c, lw=1.6, ms=5, label=f"δ = {r.delta_keV:.0f} keV")
    ax.annotate(f"{y[-1]:.2f}", (EDGES[-1], y[-1]), textcoords="offset points", xytext=(6, 0), fontsize=7, color=c, va="center")
ax.axhline(2.303, color=OI["grey"], lw=0.8, ls="--"); ax.text(151, 2.4, "2.30: zero events exclude LZ's best fit at 90%", fontsize=7, color=OI["grey"])
ax.set_xlabel("hard ROI upper edge [keV] (plateau 0.96)"); ax.set_ylabel("expected events, XENONnT + PandaX-4T (4.64 t·yr)\nat LZ's one-event coupling")
ax.set_title("Sensitivity to the ROI upper edge (1 TeV, O$_1$ inelastic)", fontsize=9.5); ax.set_ylim(0, 3.2); ax.set_xlim(140, 320)
ax.legend(fontsize=7.5, frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "roi_edge_scan.png"), dpi=170); plt.close(fig)

# Fig 3: joint (delta, kappa) regions for the discovery case
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
ax = axes[0]
for case, c in zip(JOINT.keys(), [OI["black"], OI["blue"], OI["orange"], OI["green"]]):
    z = np.load(os.path.join(OUT, f"joint_surface_{case.replace('+','_').replace('(','').replace(')','').replace(' ','_')}.npz"))
    ax.contour(z["deltas"], z["kappa"] * su1000, z["dm2"].T, levels=[2.30], colors=[c], linewidths=[1.6])
    ax.plot([], [], color=c, lw=1.6, label=f"{case} 68% (2 dof)")
ax.set_yscale("log"); ax.set_xlim(200, 400); ax.set_ylim(1e-45, 1e-35)
ax.set_xlabel("δ [keV]"); ax.set_ylabel("σ$_{SI}$-equivalent [cm²]"); ax.set_title("Joint 68% regions: LZ event + one XENONnT event at E₂", fontsize=9)
ax.legend(fontsize=7, frameon=False, loc="upper left")
ax = axes[1]
for case, c in zip(JOINT.keys(), [OI["black"], OI["blue"], OI["orange"], OI["green"]]):
    zp = JOINT[case]["Zprof"]; ds = sorted(zp); ax.plot(ds, [zp[d] for d in ds], color=c, lw=1.6, label=case)
ax.set_xlim(200, 400); ax.set_ylim(0, 6); ax.set_xlabel("δ [keV]"); ax.set_ylabel("profile Z = √q₀ (asymptotic)"); ax.set_title("Profile significance vs δ (1 TeV)", fontsize=9)
ax.legend(fontsize=7, frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "joint_discovery_case.png"), dpi=170); plt.close(fig)

# Fig 4: efficiency models and the delta=300/380 spectra
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(E_GRID, eff_LZ(E_GRID), color=OI["black"], lw=1.6, label="LZ-like: 0.96, 50% at 269.9 keV")
ax.plot(E_GRID, eff_hi(E_GRID), color=OI["verm"], lw=1.4, ls="--", label="variant: 0.85, 50% at 300 keV")
for e, c in zip([150.0, 200.0, 250.0], [OI["blue"], OI["orange"], OI["green"]]):
    ax.plot(E_GRID, eff_edge(E_GRID, e), color=c, lw=1.0, ls=":", label=f"hard edge {e:.0f} keV")
d, E, r = SPEC[1000.0]
for dl, c in [(300.0, OI["sky"]), (380.0, OI["purple"])]:
    i = int(np.argmin(np.abs(d - dl))); y = r[i] / r[i].max()
    ax.plot(E, y, color=c, lw=1.2, label=f"O$_1$ spectrum δ = {dl:.0f} keV (scaled)")
ax.set_xlim(0, 450); ax.set_ylim(0, 1.05); ax.set_xlabel("true recoil energy [keV]"); ax.set_ylabel("efficiency / scaled rate")
ax.legend(fontsize=6.8, frameon=False, ncol=2); fig.tight_layout(); fig.savefig(os.path.join(FIG, "efficiency_models.png"), dpi=170); plt.close(fig)

say("done."); LOG.close()
