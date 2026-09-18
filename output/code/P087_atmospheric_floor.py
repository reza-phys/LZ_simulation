"""
P087 -- The atmospheric-neutrino floor in the extended window: CEvNS rates, spectral shape and
uncertainty at 55-420 keV in xenon, and whether it limits the decisive tests.

Run from the simulation root:   .venv/bin/python output/code/P087_atmospheric_floor.py
Outputs -> output/work/P087/  (CSV tables, P087_results.json, run_log.txt, figures/*.png)

Sections
  A. Atmospheric-neutrino flux parametrisation (recalled Honda/Battistoni-like; bracketed shapes; +-25% norm)
  B. CEvNS machinery (vectorised): Helm, Helm+neutron-skin and WimPyDD shell-model weak form factors
  C. Differential spectra 1-600 keV and counts in 5.4-55 / 55-125 / 125-200 / 200-270 / 270-420 keV,
     LZ (600 phd) and extended (1000 phd) efficiencies; validation against LZ Table I (0.11 +- 0.02)
  D. DSNB, 8B and hep in the same bins (kinematics)
  E. NR-band location of a 248 keV CEvNS recoil (nestpy, LZ-tuned)
  F. Neutrino-floor implications: 1-event exposures, discovery limits for O1 inelastic (1 TeV, delta =
     300/350/366 keV) vs exposure with/without the atmospheric background, systematic floor
  G. Spectral separation atmospheric CEvNS vs inelastic signal (KL, Gaussian N_3sigma)
  H. Annual modulation of the atmospheric flux vs the DM modulation of P034
"""
from __future__ import annotations
import json, math, os, sys, time
import numpy as np
import pandas as pd
from scipy import integrate, optimize, special, stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "output/code")
from common import lzcommon as lz

OUT = "output/work/P087"; FIG = f"{OUT}/figures"
os.makedirs(FIG, exist_ok=True)
R: dict = {}; LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)
T0 = time.time()

# ------------------------------------------------------------------------------------------------
# 0. Constants (recalled, certain unless flagged) and LZ numbers
# ------------------------------------------------------------------------------------------------
GF = 1.1663788e-5            # GeV^-2 (PDG; certain)
S2W = 0.23867                # low-q weak mixing angle (recalled, likely) -- same as P019/P060
GEV_TO_CM2 = lz.GEV_TO_CM2
ISO = lz.XE_ISOTOPES; Z_XE = 54
EXPOSURE = lz.LZ["exposure_tyr"]                     # 2.84 t yr
N_ATM_ROI, N_ATM_ERR = lz.LZ["bkg_expected"]["atm_nu"]   # Table I: (1.1 +- 0.2) x 10^-1
N_B8_ROI, N_B8_ERR = lz.LZ["bkg_expected"]["B8_hep"]     # Table I: (5.7 +- 0.6) x 10^-2
SEC_PER_YR = 3.15576e7
def mN_gev(A): return lz.m_nucleus_gev(A)
def n_per_tonne(A, f): return 1e6 / (A * 1.66054e-24) * f
N_T_TONNE = sum(n_per_tonne(A, f) for A, f in ISO.items())
def QW(A): return (A - Z_XE) - (1 - 4 * S2W) * Z_XE
def q_MeV(E_keV, A): return math.sqrt(2 * mN_gev(A) * E_keV * 1e-6) * 1e3
R["constants"] = dict(GF=GF, s2w=S2W, QW_131=QW(131), nuclei_per_tonne=N_T_TONNE, exposure_tyr=EXPOSURE,
                      TableI_atm=[N_ATM_ROI, N_ATM_ERR], TableI_B8hep=[N_B8_ROI, N_B8_ERR], q_MeV_248_A131=q_MeV(248, 131))
log(f"Q_W(131Xe) = {QW(131):.2f}; nuclei/t = {N_T_TONNE:.3e}; q(248 keV) = {q_MeV(248,131):.1f} MeV; Table I atm nu = {N_ATM_ROI} +- {N_ATM_ERR}, 8B+hep = {N_B8_ROI} +- {N_B8_ERR}")

# efficiencies (true recoil energy).  LZ: plateau 0.955, erf turn-on 50% at 5.4 keV (sigma 2.5), roll-off 50% at
# 269.9 keV (sigma 11.5; Fig. S2 / P021 / P050).  Extended 1000 phd: E50 = 423.2 keV, sigma 18 keV (P038).
PLATEAU = 0.955
def eff_gen(E, E50_hi, sig_hi, E50_lo=5.4, sig_lo=2.5):
    E = np.asarray(E, float)
    lo = 0.5 * (1 + special.erf((E - E50_lo) / (math.sqrt(2) * sig_lo)))
    hi = 0.5 * (1 - special.erf((E - E50_hi) / (math.sqrt(2) * sig_hi)))
    return PLATEAU * lo * hi
SIG_LO = 2.5   # provisional; re-fitted below to LZ's 8B+hep expectation (Table I: 0.057), which fixes the sub-threshold acceptance
def eff_lz(E): return eff_gen(E, 269.9, 11.5, 5.4, SIG_LO)
def eff_ext(E): return eff_gen(E, 423.2, 18.0, 5.4, SIG_LO)
def eff_none(E): return np.ones_like(np.asarray(E, float))
EFFS = {"none": eff_none, "LZ600phd": eff_lz, "ext1000phd": eff_ext}

# ------------------------------------------------------------------------------------------------
# A. Atmospheric flux (RECALLED).  All-flavour sum (nu_e, nu_e-bar, nu_mu, nu_mu-bar); CEvNS is flavour-blind.
#    Shape: smooth double-broken power law  dPhi/dE = K (E/E1)^-ga [1+(E/E1)^n]^-((gb-ga)/n) [1+(E/E2)^n]^-((gc-gb)/n)
#    ga ~ 1.0 for 10-100 MeV (Battistoni et al. 2005 FLUKA low-energy flux, E dPhi/dE roughly flat; likely),
#    gb ~ 2.0 for 0.2-1 GeV and gc ~ 2.7 above ~1.5 GeV (Honda et al. 2011; likely).
#    Normalisation: 10.5 cm^-2 s^-1 above ~13 MeV (Billard et al. 2014 Table I / Baxter et al. 2021), +-20-25 %
#    (recalled, likely).  SURF (44 N, rigidity cutoff ~ 2-3 GV) vs Kamioka/LNGS site differences ~ 10-15 % at
#    100 MeV are inside this band (recalled, uncertain).
# ------------------------------------------------------------------------------------------------
PHI_TOTAL = 10.5; PHI_SYS = 0.25; E_LO_MEV = 13.0
PHI_1GEV_RECALLED = (1.5e-4, 2.5e-4)   # all-flavour dPhi/dE at 1 GeV, cm^-2 s^-1 MeV^-1 (Honda 2011 / Gaisser-Honda 2002; recalled, likely x1.5)
ENU = np.geomspace(1.0, 1e5, 6000)                     # MeV
def shape_smooth(E, ga=1.0, E1=120.0, gb=2.3, E2=2000.0, gc=3.0, n=2.0):
    E = np.asarray(E, float)
    s = (E / E1) ** (-ga) * (1 + (E / E1) ** n) ** (-(gb - ga) / n) * (1 + (E / E2) ** n) ** (-(gc - gb) / n)
    return np.where(E < E_LO_MEV, 0.0, s)
def shape_broken(E, g1=1.0, Eb=100.0, g2=2.5):            # P019 family (continuity with the corpus)
    E = np.asarray(E, float)
    s = np.where(E < Eb, (E / Eb) ** (-g1), (E / Eb) ** (-g2))
    return np.where(E < 10.0, 0.0, s)
FLUX_VARIANTS = {
    "central (ga=1.0, E1=120, gb=2.3, E2=2000, gc=3.0)": lambda E: shape_smooth(E),
    "soft low-E (ga=1.2)": lambda E: shape_smooth(E, ga=1.2),
    "hard low-E (ga=0.8)": lambda E: shape_smooth(E, ga=0.8),
    "low break (E1=90)": lambda E: shape_smooth(E, E1=90.0),
    "high break (E1=160)": lambda E: shape_smooth(E, E1=160.0),
    "hard mid (gb=2.0)": lambda E: shape_smooth(E, gb=2.0),
    "soft mid (gb=2.6)": lambda E: shape_smooth(E, gb=2.6),
    "hard high (gc=2.7)": lambda E: shape_smooth(E, gc=2.7),
    "soft high (gc=3.3)": lambda E: shape_smooth(E, gc=3.3),
    "P019 central broken (g1=1, Eb=100, g2=2.5)": lambda E: shape_broken(E),
    "P019 hard broken (g2=2.0)": lambda E: shape_broken(E, g2=2.0),
    "P019 soft broken (g2=3.0)": lambda E: shape_broken(E, g2=3.0),
}
def normalised_flux(shape_fn, total=PHI_TOTAL):
    s = shape_fn(ENU); I = np.trapezoid(s, ENU)
    return s * total / I                                    # cm^-2 s^-1 MeV^-1 on ENU
def flux_above(phi, Emin): m = ENU >= Emin; return float(np.trapezoid(phi[m], ENU[m]))
flux_rows = []
FLUX = {}
for name, fn in FLUX_VARIANTS.items():
    phi = normalised_flux(fn); FLUX[name] = phi
    flux_rows.append(dict(variant=name, phi_total=PHI_TOTAL, phi_at_30MeV=float(np.interp(30, ENU, phi)), phi_at_100MeV=float(np.interp(100, ENU, phi)),
                          phi_at_1GeV=float(np.interp(1000, ENU, phi)), phi_above_60MeV=flux_above(phi, 60), phi_above_110MeV=flux_above(phi, 110),
                          phi_above_160MeV=flux_above(phi, 160), phi_above_1GeV=flux_above(phi, 1000)))
    log(f"[flux] {name:48s} dPhi/dE(30/100/1000 MeV) = {flux_rows[-1]['phi_at_30MeV']:.3e}/{flux_rows[-1]['phi_at_100MeV']:.3e}/{flux_rows[-1]['phi_at_1GeV']:.3e}; Phi(>110 MeV) = {flux_rows[-1]['phi_above_110MeV']:.2f} cm^-2 s^-1")
pd.DataFrame(flux_rows).to_csv(f"{OUT}/P087_flux_variants.csv", index=False)
R["flux_variants"] = flux_rows
CENTRAL = list(FLUX_VARIANTS)[0]

# ------------------------------------------------------------------------------------------------
# B. CEvNS machinery.  dsigma/dE_R = (G_F^2 m_A/4pi) Q_W^2 (1 - m_A E_R/2E^2) F^2(E_R)
#    dR/dE_R = sum_A n_A pref_A F_A^2(E_R) [I0(Emin_A) - (m_A E_R/2) I2(Emin_A)]  with I0 = int Phi dE, I2 = int Phi/E^2 dE
# ------------------------------------------------------------------------------------------------
def helm_F2(E_keV, A, dc_fm=0.0):
    """Helm F^2 (Lewin-Smith), with optional shift dc of the box radius c (neutron-skin variant)."""
    q_fm = math.sqrt(2 * mN_gev(A) * E_keV * 1e-6) / lz.HBARC_GEV_FM
    s = 0.9; c = 1.23 * A ** (1 / 3) - 0.60 + dc_fm; a = 0.52
    rn = math.sqrt(c * c + 7 / 3 * math.pi ** 2 * a * a - 5 * s * s); x = q_fm * rn
    if x < 1e-6: return 1.0
    j1 = (math.sin(x) - x * math.cos(x)) / x ** 2
    return (3 * j1 / x) ** 2 * math.exp(-(q_fm * s) ** 2)
def Emin_MeV(E_keV, A):
    ER = np.asarray(E_keV, float) * 1e-3; M = mN_gev(A) * 1e3
    return 0.5 * (ER + np.sqrt(ER * ER + 2 * M * ER))
def ERmax_keV(Enu_MeV, A=lz.A_XE_MEAN): M = mN_gev(A) * 1e3; return 2 * Enu_MeV ** 2 / (M + 2 * Enu_MeV) * 1e3
E_R = np.concatenate([np.arange(0.5, 20.0, 0.25), np.arange(20.0, 600.1, 1.0)])   # keV
E_R_MIN, E_R_MAX = float(E_R[0]), float(E_R[-1])
def cum_from_top(y):
    seg = 0.5 * (y[1:] + y[:-1]) * np.diff(ENU)
    out = np.zeros_like(y); out[:-1] = np.cumsum(seg[::-1])[::-1]; return out
def flux_integrals(phi):
    return cum_from_top(phi), cum_from_top(phi / ENU ** 2)

# --- shell-model weak form factor from WimPyDD (M response with c_p = -(1-4 s2w), c_n = 1), heavy WIMP, Sun-frame halo,
#     with the velocity integral eta(v_min) divided out explicitly (improves on P019, which left eta in the ratio)
def build_shell_ratio():
    cp, cn = -(1 - 4 * S2W), 1.0
    ham = lz.wd_hamiltonian("P087_weak_charge", {1: (cp + cn, cp - cn)})
    halo = lz.wd_halo(); M_HEAVY = 1e6
    Eg = np.concatenate([[1.0], np.arange(2.0, 30.0, 2.0), np.arange(30.0, 600.1, 2.5)])
    rw = lz.wd_rate(ham, M_HEAVY, Eg, halo=halo)
    v_e = lz.v_earth_kms()
    D = np.zeros_like(Eg)
    for A, f in ISO.items():
        nA = n_per_tonne(A, f); mA = mN_gev(A)
        eta = np.array([lz.eta0(lz.vmin_kms(e, M_HEAVY, A), v_e=v_e) for e in Eg])
        D += nA * mA * QW(A) ** 2 * np.array([helm_F2(e, A) for e in Eg]) * eta
    ratio = rw / D; ratio /= ratio[0]
    return Eg, ratio, rw
Eg_sh, SHELL_RATIO, rw_dbg = build_shell_ratio()
def shell_ratio(E): return np.interp(E, Eg_sh, SHELL_RATIO)
R["shell_ratio_check"] = {f"{e:.0f} keV": float(shell_ratio(e)) for e in (5, 10, 20, 30, 50, 100, 150, 200, 248, 300, 350, 400, 500)}
log("shell/Helm F^2 ratio (WimPyDD M response / Helm, eta divided out):", {k: f"{v:.3f}" for k, v in R["shell_ratio_check"].items()})

FF_MODELS = {"helm": lambda E, A: helm_F2(E, A), "helm_skin+0.15fm": lambda E, A: helm_F2(E, A, 0.15),
             "shell(WimPyDD)": lambda E, A: helm_F2(E, A) * float(shell_ratio(E))}
_F2CACHE = {}
def F2_table(model):
    if model not in _F2CACHE:
        _F2CACHE[model] = {A: np.array([FF_MODELS[model](e, A) for e in E_R]) for A in ISO}
    return _F2CACHE[model]
def F2_natural(model):
    """Q_W^2- and abundance-weighted natural-Xe F^2 on E_R."""
    w = {A: n_per_tonne(A, f) * QW(A) ** 2 for A, f in ISO.items()}; T = F2_table(model)
    return sum(w[A] * T[A] for A in ISO) / sum(w.values())
def cevns_spectrum(phi, model="helm"):
    """dR/dE_R in events/(t yr keV) for natural xenon on the E_R grid."""
    I0, I2 = flux_integrals(phi); T = F2_table(model); out = np.zeros_like(E_R)
    for A, f in ISO.items():
        mA = mN_gev(A); nA = n_per_tonne(A, f)
        pref = GF ** 2 * mA / (4 * math.pi) * QW(A) ** 2 * GEV_TO_CM2 * 1e-6      # cm^2/keV
        em = Emin_MeV(E_R, A); le = np.log(ENU)
        i0 = np.interp(np.log(em), le, I0); i2 = np.interp(np.log(em), le, I2)
        out += nA * pref * T[A] * (i0 - (mA * 1e3) * (E_R * 1e-3) / 2 * i2) * SEC_PER_YR
    return out
def count(spec, lo, hi, eff=eff_none):
    m = (E_R >= lo) & (E_R <= hi); Eg = np.concatenate([[lo], E_R[m], [hi]])
    return float(np.trapezoid(np.interp(Eg, E_R, spec * eff(E_R)), Eg))
BINS = [(5.4, 55.0), (55.0, 125.0), (125.0, 200.0), (200.0, 270.0), (270.0, 420.0)]
def bin_counts(spec, eff):
    return {f"{lo:g}-{hi:g}": count(spec, lo, hi, eff) for lo, hi in BINS}

# ------------------------------------------------------------------------------------------------
# B2. Sub-threshold turn-on calibrated on LZ's own 8B+hep expectation (0.057 +- 0.006 in 2.84 t yr).  All 8B/hep recoils
#     lie below 5.8 keV, so this number measures the acceptance below the 5.4 keV 50% point.  Spectra (recalled): Keil-Raffelt
#     alpha = 2.5; 8B 5.25e6 cm^-2 s^-1, <E> 6.7 MeV, end point 16.5 MeV; hep 8.0e3 cm^-2 s^-1, <E> 9.6 MeV, end point 18.77 MeV.
# ------------------------------------------------------------------------------------------------
def kr_pdf(E, Emean, alpha=2.5, Emax=None):
    E = np.asarray(E, float); f = E ** alpha * np.exp(-(alpha + 1) * E / Emean)
    if Emax is not None: f = np.where(E > Emax, 0.0, f)
    return f / np.trapezoid(f, ENU)
def kr_flux(comps):   # list of (flux, <E>, Emax)
    return sum(F * kr_pdf(ENU, Em, Emax=Emx) for F, Em, Emx in comps)
PHI_B8 = kr_flux([(5.25e6, 6.7, 16.5)]); PHI_HEP = kr_flux([(8.0e3, 9.6, 18.77)])
SPEC_B8 = cevns_spectrum(PHI_B8, "helm"); SPEC_HEP = cevns_spectrum(PHI_HEP, "helm")
def N_b8hep(sig_lo):
    e = lambda E: eff_gen(E, 269.9, 11.5, 5.4, sig_lo)
    return (count(SPEC_B8, E_R_MIN, E_R_MAX, e) + count(SPEC_HEP, E_R_MIN, E_R_MAX, e)) * EXPOSURE
turnon_scan = {s: N_b8hep(s) for s in (0.5, 0.75, 1.0, 1.5, 2.0, 2.5)}
SIG_LO_FIT = optimize.brentq(lambda s: N_b8hep(s) - N_B8_ROI, 0.2, 3.0)
SIG_LO_RANGE = [optimize.brentq(lambda s: N_b8hep(s) - (N_B8_ROI + N_B8_ERR), 0.2, 3.0), optimize.brentq(lambda s: N_b8hep(s) - (N_B8_ROI - N_B8_ERR), 0.2, 3.0)]
R["turnon_calibration"] = dict(N_b8hep_vs_sigma_lo=turnon_scan, sigma_lo_fit_keV=SIG_LO_FIT, sigma_lo_range_keV=SIG_LO_RANGE, TableI_B8hep=[N_B8_ROI, N_B8_ERR],
                               B8_raw_per_tyr=count(SPEC_B8, E_R_MIN, E_R_MAX), hep_raw_per_tyr=count(SPEC_HEP, E_R_MIN, E_R_MAX))
log(f"8B+hep ROI count vs turn-on sigma_lo: " + ", ".join(f"{s} keV: {v:.3g}" for s, v in turnon_scan.items()) + f" -> Table I 0.057 reproduced with sigma_lo = {SIG_LO_FIT:.2f} keV (range {SIG_LO_RANGE[1]:.2f}-{SIG_LO_RANGE[0]:.2f}); raw 8B {R['turnon_calibration']['B8_raw_per_tyr']:.0f}/t yr")
SIG_LO = SIG_LO_FIT

# ------------------------------------------------------------------------------------------------
# C. Atmospheric spectra, normalisation check against Table I, bin counts
# ------------------------------------------------------------------------------------------------
rows = []; SPEC = {}
for name, phi in FLUX.items():
    for model in FF_MODELS:
        sp = cevns_spectrum(phi, model); SPEC[(name, model)] = sp
        N_roi = count(sp, E_R_MIN, E_R_MAX, eff_lz) * EXPOSURE          # recalled flux, full efficiency curve -> predicted Table I number
        anchor = N_ATM_ROI / N_roi
        row = dict(variant=name, ff=model, N_ROI_recalled_2p84=N_roi, anchor_LZ_over_recalled=anchor, N_total_raw_per_tyr=count(sp, E_R_MIN, E_R_MAX),
                   N_below_5p4_LZeff_per_tyr_anchored=count(sp, E_R_MIN, 5.4, eff_lz) * anchor,
                   phi_1GeV_anchored=float(np.interp(1000, ENU, phi)) * anchor, phi_above_110_anchored=flux_above(phi, 110) * anchor,
                   dRdE_248_recalled=float(np.interp(248, E_R, sp)), dRdE_248_anchored=float(np.interp(248, E_R, sp)) * anchor)
        for effname, eff in EFFS.items():
            bc = bin_counts(sp, eff)
            for k, v in bc.items():
                row[f"{effname}|{k}|per_tyr_recalled"] = v; row[f"{effname}|{k}|per_tyr_anchored"] = v * anchor
        rows.append(row)
        if model == "helm" or name == CENTRAL:
            log(f"[atm] {name:44s} {model:16s} N_ROI(recalled)={N_roi:.4f} (LZ 0.11 -> x{anchor:.2f}); phi(1 GeV) anchored {row['phi_1GeV_anchored']:.2e} [target {PHI_1GEV_RECALLED}]; anchored per t yr: "
                + ", ".join(f"{k}: {row[f'LZ600phd|{k}|per_tyr_anchored']:.2e}" for k in ("5.4-55", "55-125", "125-200", "200-270"))
                + f", 270-420(ext): {row['ext1000phd|270-420|per_tyr_anchored']:.2e}")
DF = pd.DataFrame(rows); DF.to_csv(f"{OUT}/P087_bin_counts_all_variants.csv", index=False)
c = DF[(DF.variant == CENTRAL) & (DF.ff == "helm")].iloc[0]
ANCHOR = float(c.anchor_LZ_over_recalled)
R["normalisation"] = dict(central_variant=CENTRAL, N_ROI_recalled_2p84=float(c.N_ROI_recalled_2p84), TableI=N_ATM_ROI, anchor=ANCHOR,
                          N_ROI_recalled_range_all_helm=[float(DF[DF.ff == "helm"].N_ROI_recalled_2p84.min()), float(DF[DF.ff == "helm"].N_ROI_recalled_2p84.max())],
                          pull_sigma=(float(c.N_ROI_recalled_2p84) - N_ATM_ROI) / math.hypot(N_ATM_ERR, PHI_SYS * float(c.N_ROI_recalled_2p84)),
                          phi_above_110_recalled=flux_rows[0]["phi_above_110MeV"], phi_above_110_anchored=flux_rows[0]["phi_above_110MeV"] * ANCHOR,
                          phi_total_anchored=PHI_TOTAL * ANCHOR, phi_1GeV_anchored=float(c.phi_1GeV_anchored), phi_1GeV_recalled_target=PHI_1GEV_RECALLED,
                          N_below_5p4_LZeff_per_tyr_anchored=float(c.N_below_5p4_LZeff_per_tyr_anchored), P019_phi_above_110_implied=3.08)
log(f"NORMALISATION: recalled flux (10.5 cm^-2 s^-1) predicts N_ROI = {c.N_ROI_recalled_2p84:.4f} in 2.84 t yr vs Table I 0.11 +- 0.02 -> anchor x{ANCHOR:.3f} "
    f"(pull {R['normalisation']['pull_sigma']:+.2f} sigma incl. 25% flux sys); Helm range over shapes {R['normalisation']['N_ROI_recalled_range_all_helm']}")

# main table: central Helm anchored, per t yr and per 2.84 t yr; shell and skin variants; shape range (anchored, Helm)
main = []
for (lo, hi) in BINS:
    k = f"{lo:g}-{hi:g}"
    effname = "ext1000phd" if lo >= 270 else "LZ600phd"
    sel = lambda ff: DF[(DF.variant == CENTRAL) & (DF.ff == ff)].iloc[0]
    helm_all = DF[DF.ff == "helm"]
    d = dict(bin_keV=k, efficiency=effname,
             raw_per_tyr=float(sel("helm")[f"none|{k}|per_tyr_anchored"]),
             helm_per_tyr=float(sel("helm")[f"{effname}|{k}|per_tyr_anchored"]), helm_per_2p84=float(sel("helm")[f"{effname}|{k}|per_tyr_anchored"]) * EXPOSURE,
             helm_shape_min_per_tyr=float(helm_all[f"{effname}|{k}|per_tyr_anchored"].min()), helm_shape_max_per_tyr=float(helm_all[f"{effname}|{k}|per_tyr_anchored"].max()),
             skin_per_tyr=float(sel("helm_skin+0.15fm")[f"{effname}|{k}|per_tyr_anchored"]), shell_per_tyr=float(sel("shell(WimPyDD)")[f"{effname}|{k}|per_tyr_anchored"]),
             ext1000_helm_per_tyr=float(sel("helm")[f"ext1000phd|{k}|per_tyr_anchored"]),
             exposure_for_1_event_helm_tyr=1.0 / float(sel("helm")[f"{effname}|{k}|per_tyr_anchored"]),
             exposure_for_1_event_shell_tyr=1.0 / float(sel("shell(WimPyDD)")[f"{effname}|{k}|per_tyr_anchored"]))
    d["exposure_for_1_event_helm_hardest_tyr"] = 1.0 / d["helm_shape_max_per_tyr"]
    main.append(d)
    log(f"[bin {k:8s} {effname}] anchored Helm {d['helm_per_tyr']:.3e}/t yr ({d['helm_per_2p84']:.3e} per 2.84); shapes {d['helm_shape_min_per_tyr']:.2e}-{d['helm_shape_max_per_tyr']:.2e}; "
        f"skin {d['skin_per_tyr']:.2e}; shell {d['shell_per_tyr']:.2e}; 1 event at {d['exposure_for_1_event_helm_tyr']:.3g} (Helm) / {d['exposure_for_1_event_shell_tyr']:.3g} (shell) t yr")
pd.DataFrame(main).to_csv(f"{OUT}/P087_atm_bin_table.csv", index=False)
R["atm_bins"] = main
# comparison points: P019 (central Helm, anchored): N(200-270) = 6.5e-5 per 2.84; Fig. 5 green S1c>500: 3.4e-5; P016 b_H = 5.7e-4 total NR band
b_200_270 = [m for m in main if m["bin_keV"] == "200-270"][0]
R["compare_corpus"] = dict(P019_N_200_270_per_2p84=6.5e-5, ours_N_200_270_per_2p84=b_200_270["helm_per_2p84"], P019_fig5_green_S1c_gt_500=3.4e-5,
                           ours_N_above_230_LZeff_per_2p84=count(SPEC[(CENTRAL, "helm")], 230, 600, eff_lz) * ANCHOR * EXPOSURE,
                           P016_total_NR_band_200_270_per_2p84=5.7e-4, cevns_share_of_P016=b_200_270["helm_per_2p84"] / 5.7e-4)
log("corpus comparison:", {k: (f"{v:.3e}" if isinstance(v, float) else v) for k, v in R["compare_corpus"].items()})
# spectra table (anchored)
spec_tab = {"E_keV": E_R, "eff_LZ": eff_lz(E_R), "eff_ext1000": eff_ext(E_R)}
for model in FF_MODELS: spec_tab[f"atm_{model}_anchored_per_tyr_keV"] = SPEC[(CENTRAL, model)] * ANCHOR
spec_tab["atm_helm_shape_min"] = np.min([SPEC[(n, "helm")] * DF[(DF.variant == n) & (DF.ff == "helm")].iloc[0].anchor_LZ_over_recalled for n in FLUX], axis=0)
spec_tab["atm_helm_shape_max"] = np.max([SPEC[(n, "helm")] * DF[(DF.variant == n) & (DF.ff == "helm")].iloc[0].anchor_LZ_over_recalled for n in FLUX], axis=0)
spec_tab["F2_helm_nat"] = F2_natural("helm"); spec_tab["F2_skin_nat"] = F2_natural("helm_skin+0.15fm"); spec_tab["F2_shell_nat"] = F2_natural("shell(WimPyDD)")
# form-factor nodes
for model in FF_MODELS:
    F2n = F2_natural(model); m1 = (E_R > 60) & (E_R < 160); m2 = (E_R > 180) & (E_R < 400)
    R[f"node_{model}"] = dict(node1_keV=float(E_R[m1][np.argmin(F2n[m1])]), node2_keV=float(E_R[m2][np.argmin(F2n[m2])]), F2_at_248=float(np.interp(248, E_R, F2n)),
                              F2_at_30=float(np.interp(30, E_R, F2n)), F2_at_100=float(np.interp(100, E_R, F2n)), F2_at_200=float(np.interp(200, E_R, F2n)),
                              F2_at_350=float(np.interp(350, E_R, F2n)), F2_at_420=float(np.interp(420, E_R, F2n)))
    log(f"F^2 natural Xe [{model}]: nodes {R[f'node_{model}']['node1_keV']:.0f}/{R[f'node_{model}']['node2_keV']:.0f} keV; F^2(30/100/200/248/350/420) = {R[f'node_{model}']['F2_at_30']:.3f}/{R[f'node_{model}']['F2_at_100']:.2e}/{R[f'node_{model}']['F2_at_200']:.2e}/{R[f'node_{model}']['F2_at_248']:.2e}/{R[f'node_{model}']['F2_at_350']:.2e}/{R[f'node_{model}']['F2_at_420']:.2e}")

# ------------------------------------------------------------------------------------------------
# D. DSNB, 8B, hep (kinematics + counts).  Spectra: Keil-Raffelt pinched (alpha 2.5) as in P060 (recalled);
#    8B: total 5.25e6 cm^-2 s^-1 (SNO/B16; recalled certain to 5%), <E> = 6.7 MeV, end point 16.5 MeV (approx. shape);
#    hep: 8.0e3 cm^-2 s^-1 (recalled likely), <E> = 9.6 MeV, end point 18.77 MeV.
# ------------------------------------------------------------------------------------------------
OTHER = {
    "DSNB central (30 cm-2 s-1; <E> 9/11/13 MeV)": kr_flux([(5.0, 9.0, None), (5.0, 11.0, None), (20.0, 13.0, None)]),
    "DSNB very hard (30; <E> 12/15/25 MeV)": kr_flux([(5.0, 12.0, None), (5.0, 15.0, None), (20.0, 25.0, None)]),
    "solar 8B (5.25e6; <E> 6.7, end 16.5 MeV)": PHI_B8,
    "solar hep (8.0e3; <E> 9.6, end 18.77 MeV)": PHI_HEP,
}
R["turnon_calibration"]["B8_ERmax_keV"] = ERmax_keV(16.5); R["turnon_calibration"]["hep_ERmax_keV"] = ERmax_keV(18.77)
R["kinematics"] = dict(ERmax_keV={f"{e} MeV": ERmax_keV(e) for e in (16.5, 18.77, 30, 60, 100)},
                       Enu_min_MeV={f"{e} keV": float(Emin_MeV(e, lz.A_XE_MEAN)) for e in (55, 125, 200, 248, 270, 420)})
log("kinematics:", R["kinematics"])
other_rows = []
for name, phi in OTHER.items():
    sp = cevns_spectrum(phi, "helm"); SPEC[(name, "helm")] = sp
    row = dict(source=name, N_ROI_LZeff_per_2p84=count(sp, E_R_MIN, E_R_MAX, eff_lz) * EXPOSURE,
               N_ROI_LZeff_turnon1keV_per_2p84=count(sp, E_R_MIN, E_R_MAX, lambda E: eff_gen(E, 269.9, 11.5, 5.4, 1.0)) * EXPOSURE,
               N_all_raw_per_tyr=count(sp, E_R_MIN, E_R_MAX), N_above_5p4_raw_per_tyr=count(sp, 5.4, E_R_MAX), N_above_55_raw_per_tyr=count(sp, 55, E_R_MAX))
    for k, v in bin_counts(sp, eff_lz).items(): row[f"LZ600phd|{k}|per_tyr"] = v
    row["ext1000phd|270-420|per_tyr"] = count(sp, 270, 420, eff_ext)
    other_rows.append(row)
    log(f"[other] {name:48s} ROI(LZ eff) {row['N_ROI_LZeff_per_2p84']:.3e} per 2.84 t yr (turn-on sigma 1 keV: {row['N_ROI_LZeff_turnon1keV_per_2p84']:.3e}); raw all {row['N_all_raw_per_tyr']:.2e}, >5.4 keV {row['N_above_5p4_raw_per_tyr']:.2e}, >55 keV {row['N_above_55_raw_per_tyr']:.2e}/t yr; bins: "
        + ", ".join(f"{k}: {row[f'LZ600phd|{k}|per_tyr']:.1e}" for k in ("5.4-55", "55-125", "125-200", "200-270")) + f", 270-420: {row['ext1000phd|270-420|per_tyr']:.1e}")
pd.DataFrame(other_rows).to_csv(f"{OUT}/P087_other_neutrinos.csv", index=False)
R["other_neutrinos"] = other_rows
R["B8hep_vs_TableI"] = dict(ours_ROI_per_2p84_turnon2p5=other_rows[2]["N_ROI_LZeff_per_2p84"] + other_rows[3]["N_ROI_LZeff_per_2p84"],
                            ours_ROI_per_2p84_turnon1p0=other_rows[2]["N_ROI_LZeff_turnon1keV_per_2p84"] + other_rows[3]["N_ROI_LZeff_turnon1keV_per_2p84"], TableI=N_B8_ROI,
                            B8_raw_per_tyr=other_rows[2]["N_all_raw_per_tyr"], hep_raw_per_tyr=other_rows[3]["N_all_raw_per_tyr"],
                            note="threshold-dominated: all 8B/hep recoils lie below 5.8 keV; the erf turn-on (5.4 keV, sigma 2.5 or 1.0 keV) is only a proxy for LZ's S1c>=3 phd acceptance")
log(f"8B+hep in ROI with our turn-on: {R['B8hep_vs_TableI']['ours_ROI_per_2p84_turnon2p5']:.3f} (sigma 2.5 keV) / {R['B8hep_vs_TableI']['ours_ROI_per_2p84_turnon1p0']:.3f} (sigma 1 keV) vs Table I 0.057 (threshold-dominated; all recoils < 5.8 keV); raw 8B {R['B8hep_vs_TableI']['B8_raw_per_tyr']:.2f}/t yr")
spec_tab["DSNB_central_per_tyr_keV"] = SPEC[(list(OTHER)[0], "helm")]; spec_tab["DSNB_veryhard_per_tyr_keV"] = SPEC[(list(OTHER)[1], "helm")]
spec_tab["B8_per_tyr_keV"] = SPEC[(list(OTHER)[2], "helm")]; spec_tab["hep_per_tyr_keV"] = SPEC[(list(OTHER)[3], "helm")]

# ------------------------------------------------------------------------------------------------
# E. NR-band location of a 248 keV CEvNS recoil (identical to any 248 keV NR)
# ------------------------------------------------------------------------------------------------
nph, ne = lz.nest_nr_yields(248.0)
P019 = json.load(open("output/work/P019/P019_results.json")); k_s1 = P019["S1_scale_factor_vs_TableS5"]
R["nr_band_248"] = dict(nestpy_Nph=nph, nestpy_Ne=ne, S1c_TableS5=lz.LZ["g1"] * nph, S1c_P009_contour_scale=lz.LZ["g1"] * nph * k_s1, log10S2c=math.log10(lz.LZ["g2"] * ne),
                        event_S1c=lz.LZ["ev_S1c"], event_log10S2c=lz.LZ["ev_log10S2c"], event_sigma_below_NR_median=lz.LZ["ev_sigma_below_NR_median"],
                        note="a CEvNS recoil of 248 keV populates exactly the same {S1c, log10 S2c} distribution as a DM NR of 248 keV; no S1/S2 discrimination")
log(f"248 keV NR (nestpy LZ-tuned): Nph={nph:.0f}, Ne={ne:.0f} -> S1c={lz.LZ['g1']*nph:.0f} phd (Table S5) / {lz.LZ['g1']*nph*k_s1:.0f} phd (P009 scale), log10 S2c={math.log10(lz.LZ['g2']*ne):.3f}; event 540.1 phd, 3.967 (1.5 sigma below NR median)")

# ------------------------------------------------------------------------------------------------
# F. Signal spectra (P050 WimPyDD cache: isoscalar O1, 1 TeV, Anand unit coupling c^0 = 1/m_v^2, 12-day annual mean),
#    verification, discovery limits vs exposure with/without atmospheric background, systematic floor
# ------------------------------------------------------------------------------------------------
SP = np.load("output/work/P050/P050_spectra_cache.npz")
E_IN = SP["E_in"]; DELTAS = [float(d) for d in SP["deltas"]]
INEL = {d: np.interp(E_R, E_IN, SP["inel"][i], left=0.0, right=0.0) for i, d in enumerate(DELTAS)}
L10 = np.interp(E_R, SP["E_l10"], SP["l10"], left=0.0, right=0.0)
# verification of the cache with a fresh WimPyDD call at delta = 350 keV (annual mean over the same 12 days)
MV2 = lz.M_V_GEV ** 2; c0, c1 = lz.wd_c_from_anand(1.0 / MV2, 0.0)
hO1 = lz.wd_hamiltonian("P087_O1s", {1: (c0, c1)})
VG = np.linspace(0.0, 844.0, 1200); halos = [lz.wd_halo(day_of_year=d, vmin=VG) for d in [15 + 30 * i for i in range(12)]]
HALO_ANN = (halos[0][0], np.mean([h[1] for h in halos], axis=0))
Echk = np.array([201.0, 249.0, 300.0, 351.0])
r_fresh = lz.wd_rate(hO1, 1000.0, Echk, halo=HALO_ANN, delta_kev=350.0); r_cache = np.interp(Echk, E_IN, SP["inel"][DELTAS.index(350.0)])
R["cache_verification_delta350"] = dict(E_keV=Echk.tolist(), fresh=r_fresh.tolist(), cache=r_cache.tolist(), ratio=(r_fresh / r_cache).tolist())
log("P050 cache check (delta=350, unit coupling, per t yr keV): fresh", np.round(r_fresh, 4), "cache", np.round(r_cache, 4))
MU_N = 1000.0 * lz.M_NUCLEON_GEV / (1000.0 + lz.M_NUCLEON_GEV)
SIGMA_UNIT = MU_N ** 2 / math.pi / MV2 ** 2 * GEV_TO_CM2          # sigma_n for Anand unit coupling (c_p = c_n = 1/m_v^2)
R["sigma_n_unit_coupling_cm2"] = SIGMA_UNIT
SIGNALS = {"O1 inelastic d=300": INEL[300.0], "O1 inelastic d=350": INEL[350.0], "O1 inelastic d=366": INEL[366.0], "L10 (d10=1)": L10}
# LZ best-fit normalisation: kappa = coupling^2 giving 1.0 event per 2.84 t yr in the LZ ROI (5.4-270 keV, LZ eff)
KAPPA = {k: 1.0 / (EXPOSURE * count(s, E_R_MIN, E_R_MAX, eff_lz)) for k, s in SIGNALS.items()}
R["kappa_LZ_bestfit"] = KAPPA; R["sigma_n_LZ_bestfit_cm2"] = {k: v * SIGMA_UNIT for k, v in KAPPA.items() if k.startswith("O1")}
log("LZ best-fit coupling^2 (1 event/2.84 t yr):", {k: f"{v:.3e}" for k, v in KAPPA.items()}, "P050: 7.38e-5 / 3.75e-3 / 4.57e-2 / 0.301")
for k, s in SIGNALS.items(): spec_tab[f"signal_{k.replace(' ', '_')}_LZbestfit_per_tyr_keV"] = s * KAPPA[k]
pd.DataFrame(spec_tab).to_csv(f"{OUT}/P087_spectra.csv", index=False)

WINDOWS = {"LZ 200-270 keV (600 phd)": (200.0, 270.0, eff_lz), "ext 200-420 keV (1000 phd)": (200.0, 420.0, eff_ext),
           "ext 55-420 keV (1000 phd)": (55.0, 420.0, eff_ext), "LZ 5.4-270 keV (600 phd)": (E_R_MIN, 270.0, eff_lz)}
atm_c = SPEC[(CENTRAL, "helm")] * ANCHOR
def anchored(name, model): return SPEC[(name, model)] * float(DF[(DF.variant == name) & (DF.ff == model)].iloc[0].anchor_LZ_over_recalled)
BKG = {}
for w, (lo, hi, eff) in WINDOWS.items():
    allv = [count(anchored(n, m), lo, hi, eff) for n in FLUX for m in FF_MODELS]
    BKG[w] = dict(atm_central_helm=count(atm_c, lo, hi, eff), atm_min=min(allv), atm_max=max(allv), atm_shell_central=count(anchored(CENTRAL, "shell(WimPyDD)"), lo, hi, eff))
# all modelled NR-band backgrounds (P016 for 200-270; P050 bins for the extension; per 2.84 -> per t yr)
P050B = pd.read_csv("output/work/P050/P050_background_bins.csv")
def p050_bkg(lo, hi):
    tot = 0.0
    for _, r in P050B.iterrows():
        ov = max(0.0, min(hi, r.E_hi) - max(lo, r.E_lo)); tot += r.b_total_LZ_per284 * ov / (r.E_hi - r.E_lo)
    return tot / EXPOSURE
BKG["LZ 200-270 keV (600 phd)"]["all_NR_band"] = 5.7e-4 / EXPOSURE
BKG["ext 200-420 keV (1000 phd)"]["all_NR_band"] = p050_bkg(200, 420)
BKG["ext 55-420 keV (1000 phd)"]["all_NR_band"] = p050_bkg(100, 420) + BKG["ext 55-420 keV (1000 phd)"]["atm_central_helm"]   # P050 table starts at 100 keV
BKG["LZ 5.4-270 keV (600 phd)"]["all_NR_band"] = float("nan")
R["backgrounds_per_tyr"] = BKG
for w, b in BKG.items(): log(f"[bkg per t yr] {w:30s} atm Helm {b['atm_central_helm']:.3e} (range {b['atm_min']:.1e}-{b['atm_max']:.1e}; shell {b['atm_shell_central']:.1e}); all NR-band {b['all_NR_band']:.2e}")
SUNIT = {w: {k: count(s, lo, hi, eff) for k, s in SIGNALS.items()} for w, (lo, hi, eff) in WINDOWS.items()}
R["signal_per_unit_coupling_per_tyr"] = SUNIT

# discovery limit: Billard-style -- the coupling at which 90% of experiments reach 3 sigma (one-sided p < 1.35e-3),
# exact Poisson with a log-normal background systematic marginalised (Gauss-Hermite)
P3 = stats.norm.sf(3.0)
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(40); GH_W = GH_W / GH_W.sum()
def p_at_least(n, b, sig_ln):
    if b <= 0: return 0.0 if n >= 1 else 1.0
    if sig_ln <= 0: return float(stats.poisson.sf(n - 1, b))
    bb = b * np.exp(sig_ln * GH_X - 0.5 * sig_ln ** 2)
    return float(np.sum(GH_W * stats.poisson.sf(n - 1, bb)))
def n_crit(b, sig_ln):
    n = 1
    while p_at_least(n, b, sig_ln) > P3: n += 1
    return n
def s_discovery(b, sig_ln, power=0.9):
    n = n_crit(b, sig_ln)
    lam = optimize.brentq(lambda L: stats.poisson.sf(n - 1, L) - power, 1e-6, 1e4)
    return max(lam - b, 0.0), n
EXPO = np.geomspace(1.0, 1e5, 81)
SIG_LN_25 = math.log(1.25)
dl_rows = []; DL = {}
for w, (lo, hi, eff) in WINDOWS.items():
    if w.startswith("LZ 5.4"): continue
    for k in SIGNALS:
        su = SUNIT[w][k]
        for case, (brate, sig_ln) in {"no background": (0.0, 0.0), "atm CEvNS central (Helm, 25% sys)": (BKG[w]["atm_central_helm"], SIG_LN_25),
                                      "atm CEvNS maximal (hard shape, 25% sys)": (BKG[w]["atm_max"], SIG_LN_25),
                                      "all modelled NR-band backgrounds (25% sys)": (BKG[w]["all_NR_band"], SIG_LN_25)}.items():
            kap = []; ncs = []
            for ex in EXPO:
                s, n = s_discovery(brate * ex, sig_ln); kap.append(s / (su * ex)); ncs.append(n)
            kap = np.array(kap); DL[(w, k, case)] = kap
            for ex, kv, n in zip(EXPO, kap, ncs):
                dl_rows.append(dict(window=w, signal=k, case=case, exposure_tyr=ex, kappa_DL=kv, sigma_n_DL_cm2=kv * SIGMA_UNIT if k.startswith("O1") else float("nan"), n_crit=n))
pd.DataFrame(dl_rows).to_csv(f"{OUT}/P087_discovery_limits.csv", index=False)
# summary numbers: exposure where atm background first changes the limit by >10% and by x2; limits at 10/100 t yr; systematic floor
summ = []
for w in list(WINDOWS)[:3]:
    for k in SIGNALS:
        k0 = DL[(w, k, "no background")]; k1 = DL[(w, k, "atm CEvNS central (Helm, 25% sys)")]; k2 = DL[(w, k, "atm CEvNS maximal (hard shape, 25% sys)")]; k3 = DL[(w, k, "all modelled NR-band backgrounds (25% sys)")]
        def first_over(kk, f):
            idx = np.where(kk / k0 > f)[0]; return float(EXPO[idx[0]]) if len(idx) else float("inf")
        su = SUNIT[w][k]; b = BKG[w]["atm_central_helm"]
        floor25 = 3 * 0.25 * b / su; floor_max = 3 * (BKG[w]["atm_max"] - BKG[w]["atm_min"]) / su
        def eps_single_event(brate):   # exposure beyond which one observed event is no longer a 3 sigma discovery (25% sys)
            if brate <= 0: return float("inf")
            return optimize.brentq(lambda ex: p_at_least(1, brate * ex, SIG_LN_25) - P3, 1e-3, 1e9)
        d = dict(window=w, signal=k, s_unit_per_tyr=su, b_atm_per_tyr=b, kappa_LZ_bestfit=KAPPA[k],
                 exposure_single_event_3sigma_atm=eps_single_event(b), exposure_single_event_3sigma_atm_max=eps_single_event(BKG[w]["atm_max"]),
                 exposure_single_event_3sigma_allNR=eps_single_event(BKG[w]["all_NR_band"]),
                 kappa_DL_10tyr_nobkg=float(np.interp(10, EXPO, k0)), kappa_DL_10tyr_atm=float(np.interp(10, EXPO, k1)), kappa_DL_100tyr_nobkg=float(np.interp(100, EXPO, k0)),
                 kappa_DL_100tyr_atm=float(np.interp(100, EXPO, k1)), kappa_DL_100tyr_atm_max=float(np.interp(100, EXPO, k2)), kappa_DL_100tyr_allNR=float(np.interp(100, EXPO, k3)),
                 exposure_atm_changes_DL_10pct=first_over(k1, 1.1), exposure_atm_changes_DL_x2=first_over(k1, 2.0), exposure_atm_max_changes_DL_10pct=first_over(k2, 1.1),
                 exposure_allNR_changes_DL_10pct=first_over(k3, 1.1),
                 exposure_DL_reaches_LZ_bestfit_nobkg=float(np.interp(-math.log(KAPPA[k]), -np.log(k0), EXPO)) if k0.min() < KAPPA[k] < k0.max() else float("nan"),
                 kappa_floor_25pct_norm=floor25, kappa_floor_shape_FF_bracket=floor_max, floor_over_LZ_bestfit_25pct=floor25 / KAPPA[k], floor_over_LZ_bestfit_bracket=floor_max / KAPPA[k],
                 exposure_floor_equals_1_event=1.0 / (3 * 0.25 * b) if b > 0 else float("inf"))
        summ.append(d)
        log(f"[DL {w:28s} {k:20s}] s_unit {su:.3e}/t yr; b_atm {b:.2e}/t yr; kappa_LZ {KAPPA[k]:.2e}; DL(100 t yr) no-bkg {d['kappa_DL_100tyr_nobkg']:.2e}, atm {d['kappa_DL_100tyr_atm']:.2e}, allNR {d['kappa_DL_100tyr_allNR']:.2e}; "
            f"one event stops being 3 sigma at {d['exposure_single_event_3sigma_atm']:.3g} t yr (atm; max shape {d['exposure_single_event_3sigma_atm_max']:.3g}; all NR {d['exposure_single_event_3sigma_allNR']:.3g}); "
            f"sys floor/LZ best fit = {d['floor_over_LZ_bestfit_25pct']:.1e} (bracket {d['floor_over_LZ_bestfit_bracket']:.1e})")
pd.DataFrame(summ).to_csv(f"{OUT}/P087_discovery_summary.csv", index=False)
R["discovery_summary"] = summ

# ------------------------------------------------------------------------------------------------
# G. Spectral separation: atmospheric CEvNS vs inelastic signal (resolution sigma_E = 11 sqrt(E/248) keV, P009/P021)
# ------------------------------------------------------------------------------------------------
def smear(spec):
    sig = 11.0 * np.sqrt(E_R / 248.0); dE = np.gradient(E_R)
    K = np.exp(-0.5 * ((E_R[:, None] - E_R[None, :]) / sig[None, :]) ** 2) / (math.sqrt(2 * math.pi) * sig[None, :])
    return K @ (spec * dE)
kl_rows = []
for w, (lo, hi, eff) in WINDOWS.items():
    if w.startswith("LZ 5.4") or w.startswith("ext 55"): continue
    m = (E_R >= lo) & (E_R <= hi)
    for bname, bspec in {"atm Helm": atm_c, "atm shell": anchored(CENTRAL, "shell(WimPyDD)")}.items():
        pb = smear(bspec * eff(E_R))[m]; pb = np.clip(pb, 1e-300, None); pb /= pb.sum()
        for k, s in SIGNALS.items():
            ps = smear(s * eff(E_R))[m]; ps = np.clip(ps, 1e-300, None); ps /= ps.sum()
            l = np.log(ps / pb); Es = float(np.sum(ps * l)); Eb = float(np.sum(pb * l))
            Vb = float(np.sum(pb * l * l) - Eb ** 2); Vs = float(np.sum(ps * l * l) - Es ** 2)
            N3_b = 9 * Vb / (Es - Eb) ** 2; N3_s = 9 * Vs / (Es - Eb) ** 2
            med_s = float(E_R[m][np.searchsorted(np.cumsum(ps), 0.5)]); med_b = float(E_R[m][np.searchsorted(np.cumsum(pb), 0.5)])
            kl_rows.append(dict(window=w, background=bname, signal=k, KL_s_b=Es, KL_b_s=-Eb, N3sigma_var_null=N3_b, N3sigma_var_signal=N3_s, median_E_signal=med_s, median_E_bkg=med_b))
            log(f"[KL {w:28s} {bname:10s} vs {k:20s}] KL(s||b) = {Es:.2f}, KL(b||s) = {-Eb:.2f} nats; N_3sigma = {N3_b:.1f} (null var) / {N3_s:.1f} (signal var); medians {med_s:.0f}/{med_b:.0f} keV")
pd.DataFrame(kl_rows).to_csv(f"{OUT}/P087_shape_separation.csv", index=False)
R["shape_separation"] = kl_rows

# ------------------------------------------------------------------------------------------------
# H. Annual modulation: atmospheric flux seasonal amplitude ~2% (1-4%) (recalled, uncertain; sub-GeV flux, mid-latitude)
#    vs inelastic-DM a1 from P034 (0.43/1.18/1.52 at delta = 300/350/366 keV); Gaussian cosine test Z = a sqrt(N/2)
# ------------------------------------------------------------------------------------------------
A_ATM = 0.02; A_ATM_RANGE = (0.01, 0.04); A_DM = {"300": 0.428, "350": 1.179, "366": 1.521, "380": 1.656}
def N3(a): return 18.0 / a ** 2
roi_rate = count(atm_c, E_R_MIN, E_R_MAX, eff_lz)
R["modulation"] = dict(atm_seasonal_amplitude=A_ATM, atm_seasonal_range=A_ATM_RANGE, N3sigma_atm=N3(A_ATM), N3sigma_atm_range=[N3(A_ATM_RANGE[1]), N3(A_ATM_RANGE[0])],
                       exposure_for_N3sigma_atm_ROI_tyr=N3(A_ATM) / roi_rate, atm_ROI_rate_per_tyr=roi_rate,
                       DM_a1_P034=A_DM, ratio_a1_DM_over_atm={k: v / A_ATM for k, v in A_DM.items()}, N3sigma_gaussian_DM={k: N3(v) for k, v in A_DM.items()},
                       modulated_atm_counts_200_270_per_tyr=A_ATM * BKG["LZ 200-270 keV (600 phd)"]["atm_central_helm"],
                       solar_cycle_variation_recalled="~5-10% peak-to-peak at 100 MeV over 11 yr (recalled, uncertain); not annual")
log(f"modulation: atm seasonal a={A_ATM} -> N_3sigma = {N3(A_ATM):.0f} events = {N3(A_ATM)/roi_rate:.2e} t yr of ROI CEvNS; DM a1/atm = " + ", ".join(f"{k}: {v/A_ATM:.0f}" for k, v in A_DM.items()))

# ------------------------------------------------------------------------------------------------
# Figures (dataviz conventions: fixed categorical order, thin marks, direct labels, single axis, legend for >= 2 series)
# ------------------------------------------------------------------------------------------------
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.2})
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
a = ax[0]
a.fill_between(E_R, spec_tab["atm_helm_shape_min"], spec_tab["atm_helm_shape_max"], color=C[0], alpha=0.18, lw=0, label="atm. ν, Helm: flux-shape band")
a.plot(E_R, atm_c, color=C[0], lw=2, label="atm. ν CEνNS, Helm (LZ-anchored)")
a.plot(E_R, anchored(CENTRAL, "shell(WimPyDD)"), color=C[1], lw=1.6, ls="--", label="atm. ν, shell-model F² (WimPyDD)")
a.plot(E_R, anchored(CENTRAL, "helm_skin+0.15fm"), color=C[1], lw=1.0, ls=":", label="atm. ν, Helm + 0.15 fm skin")
a.plot(E_R, SPEC[(list(OTHER)[1], "helm")], color=C[2], lw=1.6, label="DSNB (very hard, upper bracket)")
a.plot(E_R, SPEC[(list(OTHER)[0], "helm")], color=C[2], lw=1.0, ls=":", label="DSNB (central)")
a.plot(E_R, SPEC[(list(OTHER)[2], "helm")] + SPEC[(list(OTHER)[3], "helm")], color=C[3], lw=1.6, label="solar ⁸B + hep")
for lo, hi in BINS: a.axvline(lo, color="#52514e", lw=0.6, ls="--")
a.axvline(420, color="#52514e", lw=0.6, ls="--"); a.axvspan(225, 271, color="#d9d9d6", alpha=0.6, lw=0)
a.text(230, 3e-1, "LZ event\n248 ± 23 keV", fontsize=7.5, color="#52514e")
a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(0.5, 600); a.set_ylim(1e-10, 3)
a.set_xlabel("nuclear recoil energy E_R [keV]"); a.set_ylabel("dR/dE_R [events / (t·yr·keV)]")
a.set_title("Neutrino CEνNS spectra in xenon (true energy; bin edges dashed)", loc="left", fontsize=9); a.legend(fontsize=7, frameon=False, loc="lower left")
b = ax[1]
b.plot(E_R, atm_c, color=C[0], lw=2, label="atm. ν CEνNS, Helm")
b.plot(E_R, anchored(CENTRAL, "shell(WimPyDD)"), color=C[1], lw=1.6, ls="--", label="atm. ν, shell-model F²")
for i, (k, s) in enumerate(SIGNALS.items()):
    b.plot(E_R, s * KAPPA[k], color=C[4 + i], lw=1.4, label=f"{k}, LZ best fit (1 ev/2.84 t·yr)")
b.plot(E_R, eff_lz(E_R) * 1e-9, color="#52514e", lw=1, label="efficiency ×10⁻⁹: LZ 600 phd (solid), 1000 phd (dotted)")
b.plot(E_R, eff_ext(E_R) * 1e-9, color="#52514e", lw=1, ls=":")
for lo, hi in BINS: b.axvline(lo, color="#52514e", lw=0.6, ls="--")
b.axvline(420, color="#52514e", lw=0.6, ls="--"); b.axvspan(225, 271, color="#d9d9d6", alpha=0.6, lw=0)
b.set_xscale("linear"); b.set_xlim(0, 600); b.set_xlabel("nuclear recoil energy E_R [keV]")
b.set_title("Atmospheric CEνNS vs the LZ-normalised signals, 0–600 keV", loc="left", fontsize=9); b.legend(fontsize=7, frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(f"{FIG}/P087_fig1_spectra.png", dpi=160); plt.close(fig)

fig, axs = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
cases = [("no background", C[0], "-"), ("atm CEvNS central (Helm, 25% sys)", C[1], "--"), ("atm CEvNS maximal (hard shape, 25% sys)", C[2], "-."), ("all modelled NR-band backgrounds (25% sys)", C[3], ":")]
P034_E3 = {"O1 inelastic d=300": 274, "O1 inelastic d=350": 33, "O1 inelastic d=366": 19}
for row, w in enumerate(list(WINDOWS)[:2]):
    for col, k in enumerate(["O1 inelastic d=300", "O1 inelastic d=350", "O1 inelastic d=366"]):
        a = axs[row, col]
        for case, col_, ls in cases:
            a.plot(EXPO, DL[(w, k, case)] * SIGMA_UNIT, color=col_, lw=1.8, ls=ls, label=case if (row == 0 and col == 0) else None)
        a.axhline(KAPPA[k] * SIGMA_UNIT, color="#52514e", lw=1, ls="--"); a.text(1.2, KAPPA[k] * SIGMA_UNIT * 1.25, "LZ best fit (1 event)", fontsize=7, color="#52514e")
        a.axvline(P034_E3[k], color="#52514e", lw=0.8, ls=":"); a.text(P034_E3[k] * 1.1, KAPPA[k] * SIGMA_UNIT * 3e-3, f"P034 E_3σ\n{P034_E3[k]} t·yr", fontsize=7, color="#52514e")
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(1, 1e5)
        a.set_title(f"{k.replace('O1 inelastic d=', 'δ = ')} keV — {w}", fontsize=8.5, loc="left")
        if col == 0: a.set_ylabel("3σ discovery limit σ_n [cm²]")
        if row == 1: a.set_xlabel("exposure [t·yr]")
axs[0, 0].legend(fontsize=7, frameon=False, loc="lower left")
fig.suptitle("Discovery limit (90 % of experiments at 3σ) for isoscalar O1 inelastic DM at 1 TeV, with and without the atmospheric CEνNS background", fontsize=9, x=0.02, ha="left")
fig.tight_layout(); fig.savefig(f"{FIG}/P087_fig2_discovery_limits.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for a, (w, (lo, hi, eff)) in zip(ax, list(WINDOWS.items())[:2]):
    m = (E_R >= lo) & (E_R <= hi)
    def pdf(s):
        p = smear(s * eff(E_R))[m]; return p / np.trapezoid(p, E_R[m])
    a.plot(E_R[m], pdf(atm_c), color=C[0], lw=2, label="atm. ν, Helm")
    a.plot(E_R[m], pdf(anchored(CENTRAL, "shell(WimPyDD)")), color=C[1], lw=1.6, ls="--", label="atm. ν, shell-model F²")
    for i, (k, s) in enumerate(SIGNALS.items()): a.plot(E_R[m], pdf(s), color=C[4 + i], lw=1.4, label=k)
    a.set_yscale("log"); a.set_ylim(1e-5, 0.1); a.set_xlabel("observed recoil energy [keV] (σ_E = 11√(E/248) keV)"); a.set_title(f"Normalised shapes, {w}", fontsize=9, loc="left")
ax[0].set_ylabel("probability density [keV⁻¹]"); ax[0].legend(fontsize=7, frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig(f"{FIG}/P087_fig3_shapes.png", dpi=160); plt.close(fig)

fig, a = plt.subplots(figsize=(6.4, 4.0))
a.plot(E_R, spec_tab["F2_helm_nat"], color=C[0], lw=2, label="Helm (natural Xe, Q_W²-weighted)")
a.plot(E_R, spec_tab["F2_skin_nat"], color=C[1], lw=1.4, ls=":", label="Helm + 0.15 fm neutron skin")
a.plot(E_R, spec_tab["F2_shell_nat"], color=C[1], lw=1.6, ls="--", label="shell model (WimPyDD M response)")
a.axvline(248, color="#52514e", lw=0.8, ls=":"); a.text(252, 0.3, "248 keV, q = 246 MeV", fontsize=7.5, color="#52514e")
a.set_yscale("log"); a.set_ylim(1e-7, 1.5); a.set_xlim(0, 600); a.set_xlabel("E_R [keV]"); a.set_ylabel("weak form factor F²(q)")
a.set_title("Loss of coherence: the form factor controls the 200–420 keV floor", fontsize=9, loc="left"); a.legend(fontsize=7.5, frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig(f"{FIG}/P087_fig4_formfactor.png", dpi=160); plt.close(fig)

# ------------------------------------------------------------------------------------------------
# Headline
# ------------------------------------------------------------------------------------------------
d366 = [s for s in summ if s["window"].startswith("LZ 200") and s["signal"].endswith("366")][0]
d366e = [s for s in summ if s["window"].startswith("ext 200") and s["signal"].endswith("366")][0]
R["headline"] = dict(anchor=ANCHOR, N_ROI_recalled=float(c.N_ROI_recalled_2p84), sigma_lo_fit=SIG_LO_FIT,
                     eps_single_event_3sigma={s["window"] + "|" + s["signal"]: [s["exposure_single_event_3sigma_atm"], s["exposure_single_event_3sigma_allNR"]] for s in summ if s["signal"].endswith("366")},
                     atm_200_270_per_tyr_helm=b_200_270["helm_per_tyr"], atm_200_270_per_2p84=b_200_270["helm_per_2p84"], atm_200_270_shape_range=[b_200_270["helm_shape_min_per_tyr"], b_200_270["helm_shape_max_per_tyr"]],
                     atm_200_270_shell=b_200_270["shell_per_tyr"], exposure_1_event_200_270=[b_200_270["exposure_for_1_event_helm_hardest_tyr"], b_200_270["exposure_for_1_event_helm_tyr"], b_200_270["exposure_for_1_event_shell_tyr"]],
                     bins_per_tyr_helm={m["bin_keV"]: m["helm_per_tyr"] for m in main}, exposure_atm_changes_DL_10pct_366_LZ=d366["exposure_atm_changes_DL_10pct"],
                     exposure_atm_changes_DL_10pct_366_ext=d366e["exposure_atm_changes_DL_10pct"], floor_over_bestfit_366=d366["floor_over_LZ_bestfit_25pct"],
                     N3sigma_shape_366_LZ=[r["N3sigma_var_null"] for r in kl_rows if r["window"].startswith("LZ") and r["background"] == "atm Helm" and r["signal"].endswith("366")][0],
                     modulation_N3sigma_atm=N3(A_ATM), runtime_s=time.time() - T0)
with open(f"{OUT}/P087_results.json", "w") as f: json.dump(R, f, indent=1, default=float)
with open(f"{OUT}/run_log.txt", "w") as f: f.write("\n".join(LOG))
print(f"done -> {OUT}  ({time.time()-T0:.0f} s)")
