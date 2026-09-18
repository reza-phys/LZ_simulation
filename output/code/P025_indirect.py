"""
P025 -- Indirect-detection constraints on TeV inelastic dark matter after LZ:
gamma-rays, antiprotons and the CMB.

Run from the simulation root:  .venv/bin/python output/code/P025_indirect.py

Three corpus-favoured interpretations of the LZ 248 keV event are confronted with
recalled indirect-detection limits:
  (a) pure Higgsino doublet, m = 0.3-4 TeV, delta ~ 366 keV (P007, P014);
  (b) pseudo-Dirac fermion + dark photon, alpha_D from the relic density (P011);
  (c) contact magnetic-dipole (L10) WIMP, d10 = 0.28 at 1 TeV (P012).
Every experimental limit is RECALLED (no data access) and carried with a reliability
flag; the flags are written to the output JSON so the paper can quote them.

Outputs: output/work/P025/*.csv, *.json, figures/*.png
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "output/code")
from common import lzcommon as lz  # noqa: E402  (constants only)

OUT = "output/work/P025"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

# ----------------------------------------------------------------------------
# 0. Constants (recalled, certain unless flagged)
# ----------------------------------------------------------------------------
HBARC_GEV_CM = 1.973269804e-14           # GeV cm
C_CM_S = 2.99792458e10                   # cm/s
GEV2_TO_CM3S = HBARC_GEV_CM**2 * C_CM_S  # 1 GeV^-2 = 1.167e-17 cm^3/s
ALPHA_EM = 1 / 137.036                   # alpha(0); alpha(m_Z)=1/128 changes lines by ~15%
S2W = 0.2312                             # sin^2 theta_W (MSbar at m_Z)
C2W = 1 - S2W
ALPHA2 = 1 / 29.6                        # alpha_2 = alpha/sin^2 theta_W ~ 0.0338 (recalled, certain)
G2 = np.sqrt(4 * np.pi * ALPHA2)         # SU(2) coupling g ~ 0.652
M_W = 80.377
M_Z = 91.1876
SIGV_THERMAL = 2.2e-26                   # cm^3/s, Majorana canonical (Steigman-Dasgupta-Beacom 2012)
OMEGA_H2 = 0.120                         # Planck 2018

# Kinematic anchors from the corpus (P007/P002)
DELTA_KEV = 366.0
M_HIGGSINO_THERMAL = 1100.0              # GeV (recalled, certain)

log = []
def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log.append(s)

P("P025 indirect detection -- run log")
P(f"1 GeV^-2 = {GEV2_TO_CM3S:.4e} cm^3/s ; g2 = {G2:.4f}; tan^2 thetaW = {S2W/C2W:.4f}")

# ----------------------------------------------------------------------------
# 1. Recalled indirect-detection limits (anchors at 1 TeV + approximate mass scaling)
#    Each entry: value at 1 TeV [cm^3/s], power-law index in m for 0.3-4 TeV, reliability.
#    The mass scaling is OUR approximation (continuum gamma-ray limits rise roughly
#    linearly with m above a few hundred GeV); only the 1 TeV anchors are recalled.
# ----------------------------------------------------------------------------
LIMITS = {
    # continuum gamma rays, WW-like final states
    "Fermi-LAT dSph (15 dSph, 6 yr; Ackermann+2015 / Albert+2017), WW~bb":
        dict(at1TeV=1.0e-25, index=1.0, band=2.0, reliability="likely (factor ~2)",
             channel="WW/bb"),
    "H.E.S.S. Galactic Centre (Einasto; 254 h 2016 / 546 h 2022), WW":
        dict(at1TeV=4.0e-26, index=0.5, band=3.0, reliability="likely (Einasto); x10 weaker for cored profiles",
             channel="WW"),
    "AMS-02 antiprotons (Cuoco+2017, Cui+2017 style analyses), WW~bb":
        dict(at1TeV=1.0e-25, index=1.0, band=5.0, reliability="uncertain (propagation, x3-10)",
             channel="WW/bb"),
    # leptonic
    "Fermi-LAT dSph, tau tau":
        dict(at1TeV=1.0e-25, index=1.0, band=2.0, reliability="likely", channel="tautau"),
    "Fermi-LAT dSph, mu mu":
        dict(at1TeV=1.0e-24, index=1.0, band=3.0, reliability="uncertain", channel="mumu"),
    "H.E.S.S. Galactic Centre (Einasto), tau tau":
        dict(at1TeV=1.0e-26, index=0.5, band=3.0, reliability="likely (Einasto)", channel="tautau"),
    # gamma-ray lines
    "H.E.S.S. inner-Galaxy line search (Abdallah+2018), gamma gamma":
        dict(at1TeV=4.0e-28, index=1.0, band=3.0, reliability="likely (order of magnitude; Einasto)",
             channel="gammagamma"),
}
# Planck p_ann bound (Planck 2018 TT,TE,EE+lowE+lensing; 3.2e-28 in 2018, 3.5e-28 in the 2015-style number)
PLANCK_PANN = 3.5e-28    # cm^3 s^-1 GeV^-1  (recalled, likely; 3.2e-28 also quoted)
F_EFF = {"WW": 0.20, "ee": 0.45, "mumu": 0.20, "hadrons(pi)": 0.20, "cascade_mix": 0.35}  # Slatyer 2016 (recalled, likely)

def limit_at(name, m_gev):
    d = LIMITS[name]
    return d["at1TeV"] * (m_gev / 1000.0) ** d["index"]

# ----------------------------------------------------------------------------
# 2. HIGGSINO
# ----------------------------------------------------------------------------
# Effective (coannihilating, freeze-out) cross-section of a pure Higgsino doublet
# (Arkani-Hamed, Delgado, Giudice 2006; recalled/likely):
#   <sigma v>_eff = g^4 (21 + 3 tan^2 tW + 11 tan^4 tW) / (512 pi mu^2)
# Cross-check: Cirelli-Fornengo-Strumia (2006) fermion n-plet formula with n=2, Y=1/2 gives
#   g^4 [81 + 12 tan^2 + 10.75 tan^4] / (4 * 512 pi mu^2)  (our recollection), i.e. 20.25 vs 21
#   in the leading term.
T2W = S2W / C2W
def sigv_eff_higgsino_gev2(mu):
    return G2**4 * (21 + 3 * T2W + 11 * T2W**2) / (512 * np.pi * mu**2)

def sigv_eff_cfs_gev2(mu):
    return G2**4 * (81 + 12 * T2W + 10.75 * T2W**2) / (4 * 512 * np.pi * mu**2)

# Today: only chi_1 remains. Tree-level s-wave annihilation of the lightest Majorana state.
# Derivation (details.md sec. 2): a Majorana pair with a vertex of strength g_eff to a
# (fermion, vector) pair annihilates via t- and u-channel exchange of the (degenerate)
# partner fermion. Calibrating the spin-averaged threshold formula on para-positronium
# (sigma v(e+e- -> gamma gamma) = pi alpha^2 / m^2) and on the wino (g_eff = g,
# <sigma v>(WW) = 2 pi alpha_2^2 / m^2, recalled/likely):
#   <sigma v>(chi chi -> V V') = g_eff^4 / (8 pi m^2)  x  (1/2 if V = V' identical).
# Pure Higgsino: chi_1-chi^+-W vertex g/2 (vector-like), chi_1-chi_2-Z vertex g/(2 c_W).
def sigv_higgsino_today_gev2(m):
    ww = (G2 / 2) ** 4 / (8 * np.pi * m**2) * (1 - M_W**2 / m**2) ** 1.5
    zz = 0.5 * (G2 / (2 * np.sqrt(C2W))) ** 4 / (8 * np.pi * m**2) * (1 - M_Z**2 / m**2) ** 1.5
    return ww, zz

def sigv_wino_today_gev2(m):
    return G2**4 / (8 * np.pi * m**2)    # = 2 pi alpha_2^2 / m^2

# gamma-ray line (one loop, heavy limit): wino  <sigma v>_gg -> 4 pi alpha^2 alpha_2^2 / m_W^2
# (Bergstrom-Ullio 1997; recalled/likely, mass-independent). Higgsino: each of the four
# gauge vertices carries g/2 instead of g -> factor 1/16 (our estimate, uncertain x3).
def sigv_line_higgsino_gev2():
    gg = (1 / 16) * 4 * np.pi * ALPHA_EM**2 * ALPHA2**2 / M_W**2
    # gamma Z from the chargino loop: Z coupling g(1/2 - s_W^2)/c_W vs photon e = g s_W; x2 non-identical
    gZ_over_gg = 2 * ((0.5 - S2W) / np.sqrt(C2W)) ** 2 / S2W
    return gg, gg * gZ_over_gg

P("\n=== HIGGSINO ===")
mu_grid = np.array([300, 500, 800, 1000, 1100, 1500, 2000, 3000, 4000.0])
rows = []
S_LO, S_HI = 1.0, 3.0   # Sommerfeld factor band for a doublet below ~3 TeV (recalled/likely; P084 will refine)
for mu in mu_grid:
    eff = sigv_eff_higgsino_gev2(mu) * GEV2_TO_CM3S
    eff_cfs = sigv_eff_cfs_gev2(mu) * GEV2_TO_CM3S
    ww, zz = sigv_higgsino_today_gev2(mu)
    ww *= GEV2_TO_CM3S; zz *= GEV2_TO_CM3S
    tot = ww + zz
    omega = OMEGA_H2 * SIGV_THERMAL / eff
    rows.append(dict(m_GeV=mu, sigv_eff_ADG=eff, sigv_eff_CFS=eff_cfs, Omega_h2_thermal=omega,
                     sigv_WW=ww, sigv_ZZ=zz, sigv_today_S1=tot, sigv_today_S3=S_HI * tot,
                     fermi_dsph_WW=limit_at("Fermi-LAT dSph (15 dSph, 6 yr; Ackermann+2015 / Albert+2017), WW~bb", mu),
                     hess_gc_WW=limit_at("H.E.S.S. Galactic Centre (Einasto; 254 h 2016 / 546 h 2022), WW", mu),
                     ams_pbar=limit_at("AMS-02 antiprotons (Cuoco+2017, Cui+2017 style analyses), WW~bb", mu),
                     planck_WW=PLANCK_PANN * mu / F_EFF["WW"]))
hig = pd.DataFrame(rows)
hig.to_csv(os.path.join(OUT, "higgsino_sigv_vs_mass.csv"), index=False)
for r in rows:
    P(f"m={r['m_GeV']:6.0f} GeV: sigv_eff(ADG)={r['sigv_eff_ADG']:.2e} (CFS {r['sigv_eff_CFS']:.2e}) "
      f"Omega h2={r['Omega_h2_thermal']:.3f} | today WW={r['sigv_WW']:.2e} ZZ={r['sigv_ZZ']:.2e} "
      f"tot={r['sigv_today_S1']:.2e} (x3: {r['sigv_today_S3']:.2e}) | Fermi {r['fermi_dsph_WW']:.1e} "
      f"HESS {r['hess_gc_WW']:.1e} AMS {r['ams_pbar']:.1e} Planck {r['planck_WW']:.1e}")

gg, gZ = sigv_line_higgsino_gev2()
gg *= GEV2_TO_CM3S; gZ *= GEV2_TO_CM3S
line_photon_equiv = gg + 0.5 * gZ
P(f"Higgsino line (one loop, heavy limit, our estimate): gg={gg:.2e}, gZ={gZ:.2e}, "
  f"gg+gZ/2={line_photon_equiv:.2e} cm^3/s (S=1); x3 -> {3*line_photon_equiv:.2e}; "
  f"H.E.S.S. line limit ~{LIMITS['H.E.S.S. inner-Galaxy line search (Abdallah+2018), gamma gamma']['at1TeV']:.0e} at 1 TeV")
wino_check = sigv_wino_today_gev2(3000.0) * GEV2_TO_CM3S
P(f"Wino check: <sigma v>(WW) at 3 TeV = {wino_check:.2e} cm^3/s (literature tree level ~1e-26, recalled)")
P(f"Higgsino ZZ/WW ratio = {1/(2*C2W**2):.3f}; today/eff ratio at 1 TeV = "
  f"{hig.loc[hig.m_GeV==1000,'sigv_today_S1'].values[0]/hig.loc[hig.m_GeV==1000,'sigv_eff_ADG'].values[0]:.3f}")

# Mass at which Omega h^2 = 0.12 with the ADG formula
from scipy.optimize import brentq
m_th = brentq(lambda m: OMEGA_H2 * SIGV_THERMAL / (sigv_eff_higgsino_gev2(m) * GEV2_TO_CM3S) - OMEGA_H2, 300, 4000)
P(f"Thermal Higgsino mass from ADG formula (no Sommerfeld): {m_th:.0f} GeV (literature 1.1 TeV)")

# Ratios to limits at the thermal mass and at 1 TeV
hig_summary = {}
for mref in (1000.0, 1100.0):
    r = hig[hig.m_GeV == mref].iloc[0]
    hig_summary[str(int(mref))] = dict(
        sigv_today_S1=r.sigv_today_S1, sigv_today_S3=r.sigv_today_S3,
        ratio_fermi_S1=r.sigv_today_S1 / r.fermi_dsph_WW, ratio_fermi_S3=r.sigv_today_S3 / r.fermi_dsph_WW,
        ratio_hess_S1=r.sigv_today_S1 / r.hess_gc_WW, ratio_hess_S3=r.sigv_today_S3 / r.hess_gc_WW,
        ratio_ams_S1=r.sigv_today_S1 / r.ams_pbar, ratio_planck_S3=r.sigv_today_S3 / r.planck_WW,
        line_gg=gg, line_gg_plus_half_gZ=line_photon_equiv,
        ratio_line_S1=line_photon_equiv / limit_at("H.E.S.S. inner-Galaxy line search (Abdallah+2018), gamma gamma", mref),
        ratio_line_S3=3 * line_photon_equiv / limit_at("H.E.S.S. inner-Galaxy line search (Abdallah+2018), gamma gamma", mref))
    P(f"Higgsino at {mref:.0f} GeV: sigv/limit  Fermi {hig_summary[str(int(mref))]['ratio_fermi_S1']:.2f}-"
      f"{hig_summary[str(int(mref))]['ratio_fermi_S3']:.2f}, HESS {hig_summary[str(int(mref))]['ratio_hess_S1']:.2f}-"
      f"{hig_summary[str(int(mref))]['ratio_hess_S3']:.2f}, AMS {hig_summary[str(int(mref))]['ratio_ams_S1']:.2f}, "
      f"Planck(S=3) {hig_summary[str(int(mref))]['ratio_planck_S3']:.1e}, line {hig_summary[str(int(mref))]['ratio_line_S1']:.2f}-"
      f"{hig_summary[str(int(mref))]['ratio_line_S3']:.2f}")

# ----------------------------------------------------------------------------
# 3. DARK-PHOTON PSEUDO-DIRAC MODEL
# ----------------------------------------------------------------------------
P("\n=== DARK PHOTON ===")
# Relic density: at freeze-out chi_1, chi_2 are equally populated (T_f >> delta). Tree-level
# annihilations to A'A': chi_1 chi_1 (via virtual chi_2) and chi_2 chi_2 (via virtual chi_1),
# each with <sigma v> = pi alpha_D^2 / m^2 (Majorana formula, g_eff = g_D, identical bosons);
# chi_1 chi_2 -> A'A' vanishes at tree level (the A' current is purely off-diagonal), and
# chi_1 chi_2 -> A'* -> f fbar is eps^2-suppressed. Coannihilation-weighted:
#   sigma_eff = (sigma_11 + sigma_22 + 2 sigma_12)/4 = pi alpha_D^2 / (2 m^2).
# P011 set pi alpha_D^2/m^2 = 2.2e-26 (alpha_D = 0.0245 at 1 TeV); the counting above gives
# alpha_D larger by sqrt(2). Both are carried.
def alpha_D_thermal(m, convention="eff"):
    if convention == "eff":     # sigma_eff = 2.2e-26
        return np.sqrt(2 * SIGV_THERMAL / GEV2_TO_CM3S * m**2 / np.pi)
    return np.sqrt(SIGV_THERMAL / GEV2_TO_CM3S * m**2 / np.pi)   # P011

def sigv_today_dp_cm3s(m, aD):
    return np.pi * aD**2 / m**2 * GEV2_TO_CM3S    # chi_1 chi_1 -> A' A'

# Hulthen-potential Sommerfeld factor (Cassel 2010; Slatyer 2010; Feng-Kaplinghat-Yu 2010):
#   eps_v = v/alpha_D (v = single-particle CM velocity = v_rel/2), eps_phi = m_A'/(alpha_D m_chi)
#   S = (pi/eps_v) sinh(2 pi eps_v/e*) / [cosh(2 pi eps_v/e*) - cos(2 pi sqrt(1/e* - eps_v^2/e*^2))],
#   e* = pi^2 eps_phi / 6.   Coulomb limit S -> pi/eps_v; v->0 saturation S_sat = 2 pi^2/(e*(1-cos(2 pi/sqrt(e*))))
def S_hulthen(eps_v, eps_phi):
    """Valid for eps_v <~ 1 (it lacks the 1/(1-exp(-pi/eps_v)) of the exact Coulomb result)."""
    eps_phi = np.atleast_1d(np.asarray(eps_phi, dtype=float))
    es = np.pi**2 * eps_phi / 6.0
    x = 2 * np.pi * eps_v / es
    arg = 1.0 / es - eps_v**2 / es**2
    y = 2 * np.pi * np.sqrt(np.abs(arg))
    ratio = np.empty_like(es)
    small = x < 30
    with np.errstate(over="ignore", invalid="ignore"):
        cosarg = np.where(arg >= 0, np.cos(y), np.cosh(np.minimum(y, 700)))
        ratio[small] = (np.sinh(x[small]) / (np.cosh(x[small]) - cosarg[small]))
    big = ~small          # sinh x/(cosh x - cosh y) -> 1/(1 - exp(y - x)) for x, y >> 1 (y < x always)
    ratio[big] = np.where(arg[big] >= 0, 1.0, 1.0 / (1.0 - np.exp(np.minimum(y[big] - x[big], -1e-12))))
    return (np.pi / eps_v) * ratio

def S_coulomb_exact(eps_v):
    return (np.pi / eps_v) / (1 - np.exp(-np.pi / eps_v))

def S_sat(eps_phi):
    es = np.pi**2 * eps_phi / 6.0
    return 2 * np.pi**2 / (es * (1 - np.cos(2 * np.pi / np.sqrt(es))))

# check limits
P(f"Hulthen check: eps_phi->0.001, eps_v=0.1: S={S_hulthen(0.1, 1e-3)[0]:.2f} vs Coulomb pi/eps_v={np.pi/0.1:.2f}; "
  f"S_sat(eps_phi=10)={S_sat(10.0):.3f} (->1); large-eps_v branch S(eps_v=5, eps_phi=0.03)={S_hulthen(5.0, 0.03)[0]:.3f} "
  f"(pi/eps_v={np.pi/5:.3f}); exact Coulomb S(eps_v=5)={S_coulomb_exact(5.0):.3f}")

V_REC = 1e-8      # single-particle velocity at recombination for TeV DM (order of magnitude; deep saturation)
V_DSPH = 3e-5     # dwarf spheroidal (sigma ~ 10 km/s)
V_GC = 5e-4       # Galactic centre / halo (v_rel ~ 300 km/s -> v ~ 150 km/s)

mA_grid = np.logspace(-1.3, 2, 800)   # 0.05 - 100 GeV
dp_rows = []
dp_windows = {}
for m in (300.0, 1000.0, 3000.0):
    for conv in ("eff", "P011"):
        aD = alpha_D_thermal(m, conv)
        sv0 = sigv_today_dp_cm3s(m, aD)
        eps_phi = mA_grid / (aD * m)
        Ssat = S_sat(eps_phi)
        S_gc = S_hulthen(V_GC / aD, eps_phi)
        S_dsph = S_hulthen(V_DSPH / aD, eps_phi)
        # CMB: f_eff <sigma v>_rec / m < p_ann
        feff = np.where(mA_grid < 0.211, F_EFF["ee"], F_EFF["cascade_mix"])
        S_allowed_cmb = PLANCK_PANN * m / (feff * sv0)
        cmb_ok = Ssat < S_allowed_cmb
        # Fermi dSph cascade limits (recalled/uncertain): 4e ~ FSR only (>1e-23), 4mu ~ 1e-24, mixed hadronic/lept ~3e-25 at 1 TeV
        dsph_lim = np.where(mA_grid < 0.211, 1e-23, np.where(mA_grid < 0.6, 1.0e-24, 3.0e-25)) * (m / 1000.0)
        dsph_ok = S_dsph * sv0 < dsph_lim
        if conv == "eff":
            for mA, ep, ss, sg, sd, sa, ok1, ok2 in zip(mA_grid, eps_phi, Ssat, S_gc, S_dsph, S_allowed_cmb, cmb_ok, dsph_ok):
                dp_rows.append(dict(m_GeV=m, alpha_D=aD, mA_GeV=mA, eps_phi=ep, S_sat=ss, S_GC=sg, S_dSph=sd,
                                    S_allowed_CMB=sa, CMB_ok=bool(ok1), dSph_ok=bool(ok2)))
        # summary: smallest m_A' above which CMB is satisfied for ALL heavier m_A' (ignoring resonance spikes)
        ok = cmb_ok.copy()
        idx_all_ok = None
        for i in range(len(mA_grid)):
            if ok[i:].all():
                idx_all_ok = i
                break
        mA_min_all = mA_grid[idx_all_ok] if idx_all_ok is not None else np.nan
        # smallest m_A' with any CMB-allowed point (anti-resonance valleys)
        mA_min_any = mA_grid[np.argmax(ok)] if ok.any() else np.nan
        frac_allowed_sub_GeV = ok[mA_grid < 1.0].mean()
        # Sommerfeld at freeze-out (v ~ 0.3): exact Coulomb value is an upper bound (Yukawa screening only lowers it)
        S_fo = S_coulomb_exact(0.3 / aD)
        # dwarf-based threshold (recalled/uncertain cascade limit): smallest m_A' above which all heavier m_A' pass
        idx_d = None
        for i in range(len(mA_grid)):
            if dsph_ok[i:].all():
                idx_d = i
                break
        mA_min_dsph = mA_grid[idx_d] if idx_d is not None else np.nan
        S_allowed_dsph = float(3.0e-25 * (m / 1000.0) / sv0)
        dp_windows[f"{int(m)}_{conv}"] = dict(
            m_GeV=m, convention=conv, alpha_D=aD, sigv_today=sv0, aD_m_GeV=aD * m,
            S_allowed_CMB_mix=float(PLANCK_PANN * m / (F_EFF["cascade_mix"] * sv0)),
            S_allowed_CMB_ee=float(PLANCK_PANN * m / (F_EFF["ee"] * sv0)),
            S_sat_at_mA=dict(zip(["0.1", "0.3", "1", "3", "10", "30"],
                                 [float(S_sat(x / (aD * m))) for x in (0.1, 0.3, 1, 3, 10, 30)])),
            S_GC_at_mA=dict(zip(["0.1", "1", "10"], [float(S_hulthen(V_GC / aD, x / (aD * m))[0]) for x in (0.1, 1, 10)])),
            S_freezeout_hulthen_1GeV=float(S_hulthen(0.3 / aD, 1.0 / (aD * m))[0]),
            S_coulomb_rec_unsaturated=float(np.pi * aD / V_REC),
            S_coulomb_GC=float(np.pi * aD / V_GC),
            resonances_mA_GeV=[float(6 / np.pi**2 * aD * m / n**2) for n in range(1, 6)],
            mA_min_CMB_all_heavier_ok=float(mA_min_all), mA_min_CMB_any_ok=float(mA_min_any),
            frac_subGeV_CMB_allowed=float(frac_allowed_sub_GeV), S_freezeout_coulomb=float(S_fo),
            mA_min_dSph_all_heavier_ok=float(mA_min_dsph), S_allowed_dSph_cascade=S_allowed_dsph,
            S_sat_valley_min_formula_6_over_epsphi_at_1GeV=float(6.0 / (1.0 / (aD * m))),
            delta_check_alphaD_mA_over_2delta_at_1GeV=float(aD * 1.0 / (2 * DELTA_KEV * 1e-6)),
            delta_check_alphaD2_m_over_4delta=float(aD**2 * m / 4 / (2 * DELTA_KEV * 1e-6)))
        w = dp_windows[f"{int(m)}_{conv}"]
        P(f"m={m:.0f} [{conv}]: alpha_D={aD:.4f}, sigv_today={sv0:.2e}, alpha_D m={aD*m:.1f} GeV, "
          f"S_allowed(CMB, mix)={w['S_allowed_CMB_mix']:.1f} | S_sat at mA'=0.1/1/10 GeV: "
          f"{w['S_sat_at_mA']['0.1']:.0f}/{w['S_sat_at_mA']['1']:.0f}/{w['S_sat_at_mA']['10']:.1f} | "
          f"S_GC at 0.1/1/10: {w['S_GC_at_mA']['0.1']:.0f}/{w['S_GC_at_mA']['1']:.0f}/{w['S_GC_at_mA']['10']:.1f} | "
          f"resonances {['%.2f'%x for x in w['resonances_mA_GeV']]} GeV | CMB ok for all mA'>={mA_min_all:.2f} GeV, "
          f"any ok above {mA_min_any:.2f} GeV, sub-GeV allowed fraction {frac_allowed_sub_GeV:.2f} | "
          f"dSph(cascade ~3e-25 m/TeV, uncertain): S_allowed={S_allowed_dsph:.1f}, all ok for mA'>={mA_min_dsph:.1f} GeV | "
          f"S_fo(Coulomb, v=0.3)={S_fo:.2f}; Coulomb unsaturated S(v=1e-8)={w['S_coulomb_rec_unsaturated']:.1e}")
pd.DataFrame(dp_rows).to_csv(os.path.join(OUT, "darkphoton_sommerfeld_grid.csv"), index=False)

# ----------------------------------------------------------------------------
# 4. CONTACT DIPOLE (L10) WIMP
# ----------------------------------------------------------------------------
P("\n=== L10 CONTACT DIPOLE ===")
D10 = 0.28                        # P012 (LZ normalisation; factor-2 convention ambiguity, DR-001)
M_V = lz.M_V_GEV                  # 246.2 GeV
M_N = lz.M_NUCLEON_GEV
G_rel = D10 / M_V**2              # dimension-6 tensor-tensor coefficient, GeV^-2
Lambda6 = G_rel ** -0.5
G8 = G_rel / M_N**2               # P012's "dimension-8" reading, GeV^-4
Lambda8 = G8 ** -0.25
P(f"G_rel = d10/m_v^2 = {G_rel:.3e} GeV^-2 = (1/{Lambda6:.0f} GeV)^2 ; G8 = {G8:.3e} GeV^-4 = (1/{Lambda8:.1f} GeV)^4")
# Nucleon tensor charges (recalled, likely): delta_u = 0.8, delta_d = -0.2; isoscalar d10^s couples p and n equally
DELTA_U, DELTA_D = 0.8, -0.2
# isoscalar nucleon coupling from quark couplings C_u, C_d: c_p = C_u du + C_d dd ; c_n = C_u dd + C_d du ; c_p = c_n -> C_u = C_d
C_q = G_rel / (DELTA_U + DELTA_D)   # equal quark couplings
P(f"Equal quark tensor couplings C_q = {C_q:.3e} GeV^-2 (isoscalar nucleon coupling / (du+dd)={DELTA_U+DELTA_D})")
m_chi = 1000.0
N_c = 3
n_flavours = 5                     # u,d,s,c,b open at sqrt(s)=2 TeV (t as well: 6)
KAPPA_LO, KAPPA_HI = 1.0, 6.0      # O(1) operator-dependent factor for s-wave tensor annihilation (uncertain)
sigv_dim6 = N_c * n_flavours * C_q**2 * m_chi**2 / np.pi * GEV2_TO_CM3S
sigv_dim8 = m_chi**2 / (np.pi * Lambda8**4) * GEV2_TO_CM3S       # the literal (20.9 GeV)^-4 reading
v_fo, v_gal = 0.3, 1e-3
unitarity_fo = 4 * np.pi / (m_chi**2 * v_fo) * GEV2_TO_CM3S       # s-wave (J=0) unitarity, sigma v <= 4 pi/(m^2 v_rel)
unitarity_gal = 4 * np.pi / (m_chi**2 * v_gal) * GEV2_TO_CM3S
P(f"L10 annihilation, dim-6 tensor reading (kappa=1..6): {sigv_dim6:.2e} - {KAPPA_HI*sigv_dim6:.2e} cm^3/s")
P(f"L10 annihilation, literal dim-8 reading Lambda=20.9 GeV: {sigv_dim8:.2e} cm^3/s")
P(f"Unitarity bound sigma v <= 4 pi/(m^2 v): {unitarity_fo:.2e} (v=0.3) ; {unitarity_gal:.2e} (v=1e-3) cm^3/s")
omega_frac = SIGV_THERMAL / sigv_dim6
P(f"Thermal symmetric relic fraction Omega/Omega_DM ~ 2.2e-26/sigv = {omega_frac:.1e} (kappa=1); "
  f"needed rescaling of d10 to keep 1 LZ event: x{1/np.sqrt(omega_frac):.0f}, which raises sigv by x{1/omega_frac:.0e} -> runaway")
qq_limit_1TeV = 1.0e-25
P(f"Ratio to Fermi/H.E.S.S. qqbar limit (~{qq_limit_1TeV:.0e} at 1 TeV): {sigv_dim6/qq_limit_1TeV:.1e} (dim-6), "
  f"{sigv_dim8/qq_limit_1TeV:.1e} (dim-8)")
# Mediator mass for which the EFT is valid at annihilation: M > 2 m_chi; then couplings g_chi g_N / M^2 = G_rel
M_med = 2000.0
g_prod = G_rel * M_med**2
P(f"For a mediator of mass {M_med:.0f} GeV the product of tensor couplings g_chi g_N = {g_prod:.1f} "
  f"(non-perturbative: sqrt = {np.sqrt(g_prod):.1f} each); for M = 1 TeV: {G_rel*1e6:.1f}")
l10 = dict(d10=D10, G_rel_GeV2=G_rel, Lambda6_GeV=Lambda6, G8_GeV4=G8, Lambda8_GeV=Lambda8, C_q_GeV2=C_q,
           sigv_dim6_kappa1=sigv_dim6, sigv_dim6_kappa6=KAPPA_HI * sigv_dim6, sigv_dim8_literal=sigv_dim8,
           unitarity_v03=unitarity_fo, unitarity_v1e3=unitarity_gal, omega_fraction_symmetric=omega_frac,
           ratio_to_qq_limit_dim6=sigv_dim6 / qq_limit_1TeV, ratio_to_qq_limit_dim8=sigv_dim8 / qq_limit_1TeV,
           g_chi_g_N_for_M2TeV=g_prod)

# ----------------------------------------------------------------------------
# 5. SUMMARY TABLE
# ----------------------------------------------------------------------------
h1 = hig[hig.m_GeV == 1000].iloc[0]
dp1 = dp_windows["1000_eff"]
summary = [
    dict(model="Higgsino doublet, 1 TeV, delta=366 keV", sigv_today=f"{h1.sigv_today_S1:.1e} (S=1) - {h1.sigv_today_S3:.1e} (S=3)",
         channel="W+W- (54%), ZZ (46%)", strongest_bound="H.E.S.S. GC Einasto ~4e-26 (likely, profile x10)",
         ratio="0.24-0.7", verdict="allowed; marginal only for S~3 and a cuspy profile; CTA-testable"),
    dict(model="Higgsino line", sigv_today=f"{line_photon_equiv:.0e} (S=1)", channel="gamma gamma + gamma Z/2",
         strongest_bound="H.E.S.S. line ~4e-28 (likely)", ratio=f"{hig_summary['1000']['ratio_line_S1']:.2f}-{hig_summary['1000']['ratio_line_S3']:.2f}",
         verdict="allowed (x3-10 below)"),
    dict(model=f"Dark photon, 1 TeV, alpha_D=0.035 (relic), m_A' < {dp1['mA_min_CMB_any_ok']:.0f} GeV", sigv_today=f"{dp1['sigv_today']:.1e} x S_sat (S_sat={dp1['S_sat_at_mA']['1']:.0f} at 1 GeV)",
         channel="A'A' -> 4l / 2l+hadrons", strongest_bound=f"Planck p_ann (S_allowed={dp1['S_allowed_CMB_mix']:.0f})",
         ratio=f"{dp1['S_sat_at_mA']['1']/dp1['S_allowed_CMB_mix']:.0f} at m_A'=1 GeV", verdict="excluded at every sub-GeV and few-GeV mass (valleys included)"),
    dict(model=f"Dark photon, 1 TeV, m_A' = {dp1['mA_min_CMB_any_ok']:.0f}-{dp1['mA_min_CMB_all_heavier_ok']:.0f} GeV (valleys) or >= {dp1['mA_min_CMB_all_heavier_ok']:.0f} GeV",
         sigv_today=f"{dp1['sigv_today']:.1e} x S<{dp1['S_allowed_CMB_mix']:.0f}",
         channel="A'A' -> leptons+hadrons (soft gammas)", strongest_bound=f"Planck; Fermi dSph cascades ~3e-25 (uncertain) need m_A' >= {dp1['mA_min_dSph_all_heavier_ok']:.0f} GeV",
         ratio="<1", verdict="allowed (CMB); dwarfs marginal below ~50 GeV"),
    dict(model="L10 contact dipole, 1 TeV, d10=0.28 (symmetric Dirac)", sigv_today=f"{sigv_dim6:.0e}-{KAPPA_HI*sigv_dim6:.0e} (dim-6); {sigv_dim8:.0e} (literal dim-8)",
         channel="q qbar (tensor)", strongest_bound="Fermi/H.E.S.S. qq ~1e-25; unitarity",
         ratio=f"{sigv_dim6/qq_limit_1TeV:.0e}", verdict="excluded as a symmetric thermal WIMP; EFT invalid; viable only as asymmetric Dirac DM (no annihilation today)"),
]
pd.DataFrame(summary).to_csv(os.path.join(OUT, "summary_table.csv"), index=False)
P("\nSUMMARY TABLE")
for s in summary:
    P("  " + " | ".join(f"{k}={v}" for k, v in s.items()))

results = dict(constants=dict(GEV2_TO_CM3S=GEV2_TO_CM3S, g2=G2, alpha2=ALPHA2, s2w=S2W, planck_pann=PLANCK_PANN, f_eff=F_EFF),
               recalled_limits=LIMITS, higgsino=hig_summary, higgsino_thermal_mass_ADG=m_th,
               higgsino_line=dict(gg=gg, gZ=gZ, gg_plus_half_gZ=line_photon_equiv), wino_WW_3TeV_check=wino_check,
               darkphoton=dp_windows, L10=l10, summary=summary)
with open(os.path.join(OUT, "P025_results.json"), "w") as f:
    json.dump(results, f, indent=1, default=float)

# ----------------------------------------------------------------------------
# 6. FIGURES
# ----------------------------------------------------------------------------
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED = "#222222", "#6b6b6b"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})

# Fig 1: Higgsino <sigma v> vs mass with recalled limit bands
mm = np.logspace(np.log10(300), np.log10(4000), 200)
ww, zz = sigv_higgsino_today_gev2(mm)
tot = (ww + zz) * GEV2_TO_CM3S
fig, ax = plt.subplots(figsize=(6.4, 4.4))
def band(name, color, label, dy=4):
    d = LIMITS[name]
    c = d["at1TeV"] * (mm / 1000) ** d["index"]
    ax.fill_between(mm, c / np.sqrt(d["band"]), c * np.sqrt(d["band"]), color=color, alpha=0.18, lw=0)
    ax.plot(mm, c, color=color, lw=1.5, ls="--")
    ax.annotate(label, (mm[-1], c[-1]), xytext=(-4, dy), textcoords="offset points", ha="right", color=INK, fontsize=8.5)
band("Fermi-LAT dSph (15 dSph, 6 yr; Ackermann+2015 / Albert+2017), WW~bb", C2, "Fermi-LAT dSphs (WW, recalled)", dy=5)
band("H.E.S.S. Galactic Centre (Einasto; 254 h 2016 / 546 h 2022), WW", C3, "H.E.S.S. GC, Einasto (WW, recalled)")
band("AMS-02 antiprotons (Cuoco+2017, Cui+2017 style analyses), WW~bb", C4, "AMS-02 antiprotons (recalled, same anchor)", dy=-12)
ax.fill_between(mm, tot, S_HI * tot, color=C1, alpha=0.35, lw=0)
ax.plot(mm, tot, color=C1, lw=2)
ax.annotate("pure Higgsino today, WW+ZZ\n(tree x Sommerfeld S = 1 to 3)", (320, 3.0e-27), color=INK, fontsize=8.5)
ax.plot(mm, sigv_eff_higgsino_gev2(mm) * GEV2_TO_CM3S, color=C1, lw=1, ls=":")
ax.annotate("freeze-out effective\n(coannihilation)", (1900, 5.0e-27), color=MUTED, fontsize=8)
ax.axhline(SIGV_THERMAL, color=MUTED, lw=0.8, ls="-.")
ax.annotate("2.2e-26 (canonical thermal)", (2200, SIGV_THERMAL * 1.1), color=MUTED, fontsize=8)
ax.axvline(M_HIGGSINO_THERMAL, color=MUTED, lw=0.8)
ax.annotate("thermal 1.1 TeV", (M_HIGGSINO_THERMAL * 1.03, 2.5e-27), rotation=90, color=MUTED, fontsize=8, va="bottom")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(300, 4000); ax.set_ylim(1e-27, 1e-24)
ax.set_xlabel("Higgsino mass [GeV]"); ax.set_ylabel(r"$\langle\sigma v\rangle$ today [cm$^3$ s$^{-1}$]")
ax.set_title("Higgsino annihilation vs recalled indirect limits", fontsize=11, loc="left")
ax.grid(True, which="major", color="#e6e6e6", lw=0.6)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "P025_fig1_higgsino_sigv_vs_mass.png"), dpi=180)
plt.close(fig)

# Fig 2: dark photon S_sat vs m_A' for three masses, with CMB-allowed S
fig, ax = plt.subplots(figsize=(6.4, 4.4))
for m, color, label in ((300.0, C3, "300 GeV"), (1000.0, C1, "1 TeV"), (3000.0, C2, "3 TeV")):
    aD = alpha_D_thermal(m, "eff")
    ep = mA_grid / (aD * m)
    Ss = np.clip(S_sat(ep), 1, 1e6)
    ax.plot(mA_grid, Ss, color=color, lw=1.8, label=f"m = {label}, " + r"$\alpha_D$" + f" = {aD:.3f}")
    sv0 = sigv_today_dp_cm3s(m, aD)
    Sal = PLANCK_PANN * m / (F_EFF["cascade_mix"] * sv0)
    ax.axhline(Sal, color=color, lw=1, ls="--")
    ax.annotate(f"Planck-allowed S ({label})", (0.055, Sal * 1.15), color=INK, fontsize=8)
ax.axvspan(0.05, 0.211, color="#dddddd", alpha=0.5, lw=0)
ax.annotate("A'→e⁺e⁻ only", (0.06, 2.2e5), fontsize=8, color=MUTED)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(0.05, 100); ax.set_ylim(1, 1e6)
ax.set_xlabel(r"dark-photon mass $m_{A'}$ [GeV]")
ax.set_ylabel(r"saturated Sommerfeld factor $S_{\rm sat}$ (Hulthén)")
ax.set_title("Secluded pseudo-Dirac DM: CMB-era enhancement vs Planck", fontsize=11, loc="left")
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
ax.grid(True, which="major", color="#e6e6e6", lw=0.6)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "P025_fig2_darkphoton_sommerfeld_cmb.png"), dpi=180)
plt.close(fig)

with open(os.path.join(OUT, "run_log.txt"), "w") as f:
    f.write("\n".join(log) + "\n")
P("\nWritten:", OUT)
