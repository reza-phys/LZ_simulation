"""
P060 -- Diffuse supernova, Galactic supernova, solar-flare and other astrophysical neutrinos at 248 keV:
could a neutrino burst on 16 June 2023 explain the LZ event?

Run from the simulation root:   .venv/bin/python output/code/P060_astro_nu.py
Outputs -> output/work/P060/  (JSON, CSV, figures/*.png)

Sections
  A. Kinematics: E_R,max(E_nu) for xenon, E_nu,min for the 200-271 keV window.
  B. Source spectra (recalled, flagged): DSNB, Galactic core-collapse SN (10 kpc, 1 kpc, 0.2 kpc),
     SN 2023ixf (6.4 Mpc), solar flares, IceCube diffuse, reactor/geo/solar (kinematic only).
  C. Coherent CEvNS counts in LZ (4.71 t x 220 d) per source, all energies and in 200-270 / 225-271 keV.
  D. Incoherent lone-NR channel of P019 (NC quasi-elastic neutron knock-out, residual nucleus recoils
     with the hole momentum, E_max = 279 keV) with an explicit kinematic threshold; applied to the DSNB,
     a Galactic SN and (validation) the atmospheric flux of P019.
  E. High-energy (IceCube) DIS interaction count and the fraction that could deposit < 300 keV.
  F. Transient hypothesis: fluence at Earth of E_nu >= 130 MeV neutrinos needed for one 200-270 keV
     CEvNS event; energy budget vs distance; Super-K response; LZ's own low-energy companions.
  G. Summary table + figures.
"""
from __future__ import annotations
import json, math, os, sys
import numpy as np
from scipy import integrate, optimize
from scipy.stats import norm
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "output/code")
from common import lzcommon as lz

OUT = "output/work/P060"; FIG = f"{OUT}/figures"
os.makedirs(FIG, exist_ok=True)
R: dict = {}; LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)

# ------------------------------------------------------------------------------------------------
# 0. Constants (recalled, certain unless flagged) and LZ numbers (lzcommon / Table I)
# ------------------------------------------------------------------------------------------------
GF = 1.1663788e-5            # GeV^-2
S2W = 0.23867                # low-q weak mixing angle (recalled, likely) -- same as P019
GEV_TO_CM2 = lz.GEV_TO_CM2
M_NUC = 0.938272             # GeV
M_N_MEV = 939.565            # neutron mass, MeV (recalled, certain)
ISO = lz.XE_ISOTOPES; Z_XE = 54
MASS_T = lz.LZ["fiducial_mass_t"]; LIVE_D = lz.LZ["live_days"]; EXPOSURE = lz.LZ["exposure_tyr"]
LIVE_S = LIVE_D * 86400.0; SEC_PER_YR = 3.15576e7
ERG_PER_MEV = 1.602176634e-6; KPC_CM = 3.0857e21
N_ATM_ROI = lz.LZ["bkg_expected"]["atm_nu"][0]; N_B8_ROI = lz.LZ["bkg_expected"]["B8_hep"][0]
def mN_gev(A): return lz.m_nucleus_gev(A)
def n_per_tonne(A, f): return 1e6 / (A * 1.66054e-24) * f
N_T_TONNE = sum(n_per_tonne(A, f) for A, f in ISO.items())          # nuclei per tonne
def QW(A): return (A - Z_XE) - (1 - 4 * S2W) * Z_XE
R["constants"] = dict(GF=GF, s2w=S2W, mass_t=MASS_T, live_days=LIVE_D, exposure_tyr=EXPOSURE,
                      nuclei_per_tonne=N_T_TONNE, nuclei_in_FV=N_T_TONNE * MASS_T, QW_131=QW(131))
log(f"nuclei per tonne {N_T_TONNE:.3e}; in 4.71 t {N_T_TONNE*MASS_T:.3e}; Q_W(131) = {QW(131):.2f}")

# efficiency (P009 fit used by P019): plateau 0.955, 50% at 5.4 and 269.9 keV
def eff(E):
    E = np.asarray(E, float)
    return 0.955 * norm.cdf((269.9 - E) / 11.8) * norm.cdf((E - 5.4) / 1.0)

# ------------------------------------------------------------------------------------------------
# A. Kinematics
# ------------------------------------------------------------------------------------------------
def ERmax_keV(Enu_MeV, A=lz.A_XE_MEAN):
    M = mN_gev(A) * 1e3
    return 2 * Enu_MeV ** 2 / (M + 2 * Enu_MeV) * 1e3
def Enu_min_MeV(E_keV, A=lz.A_XE_MEAN):
    ER = E_keV * 1e-3; M = mN_gev(A) * 1e3
    return 0.5 * (ER + math.sqrt(ER * ER + 2 * M * ER))
kin = {}
for E in (1.8, 3.0, 10.0, 15.0, 16.3, 18.8, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 123.1, 150.0, 200.0, 300.0, 1000.0):
    kin[E] = dict(ERmax_natXe_keV=ERmax_keV(E), ERmax_131_keV=ERmax_keV(E, 131), ERmax_124_keV=ERmax_keV(E, 124))
R["ERmax_table"] = kin
R["Enu_min_MeV"] = {f"{e} keV": dict(natXe=Enu_min_MeV(e), A131=Enu_min_MeV(e, 131), A124=Enu_min_MeV(e, 124), A136=Enu_min_MeV(e, 136))
                    for e in (200, 225, 248, 270, 271)}
log("E_R,max(60 MeV) = %.1f keV (nat Xe), %.1f (124Xe); E_R,max(50) = %.1f; E_R,max(100) = %.1f keV" %
    (ERmax_keV(60), ERmax_keV(60, 124), ERmax_keV(50), ERmax_keV(100)))
log("E_nu,min: 200 keV %.1f, 225 %.1f, 248 %.1f, 271 %.1f MeV (nat Xe); 248 keV on 124Xe %.1f MeV" %
    (Enu_min_MeV(200), Enu_min_MeV(225), Enu_min_MeV(248), Enu_min_MeV(271), Enu_min_MeV(248, 124)))
pd.DataFrame([dict(Enu_MeV=E, **v) for E, v in kin.items()]).to_csv(f"{OUT}/P060_kinematics.csv", index=False)

# ------------------------------------------------------------------------------------------------
# B. Source spectra (number flux per cm^2 s MeV, or fluence per cm^2 MeV for bursts)
#    Keil-Raffelt pinched spectrum f(E) = N E^alpha exp(-(alpha+1) E/<E>), alpha = 2.5 (recalled, likely 2-3)
# ------------------------------------------------------------------------------------------------
ALPHA = 2.5
def kr_norm(Emean, alpha=ALPHA):
    I, _ = integrate.quad(lambda e: e ** alpha * math.exp(-(alpha + 1) * e / Emean), 0, 60 * Emean, limit=200)
    return 1.0 / I
def kr_pdf(E, Emean, alpha=ALPHA, N=None):
    N = kr_norm(Emean, alpha) if N is None else N
    return N * E ** alpha * np.exp(-(alpha + 1) * E / Emean)
def frac_above(Emean, Ethr, alpha=ALPHA):
    N = kr_norm(Emean, alpha)
    I, _ = integrate.quad(lambda e: kr_pdf(e, Emean, alpha, N), Ethr, 60 * Emean, limit=200)
    return I

class Source:
    """components: list of (number flux or fluence [cm^-2 (s^-1)], <E> MeV, alpha)"""
    def __init__(self, name, comps, burst, note=""):
        self.name, self.comps, self.burst, self.note = name, comps, burst, note
        self._N = [kr_norm(Em, al) for (_, Em, al) in comps]
    def phi(self, E):        # per MeV
        return sum(F * kr_pdf(E, Em, al, N) for (F, Em, al), N in zip(self.comps, self._N))
    def total(self): return sum(F for F, _, _ in self.comps)
    def above(self, Ethr): return sum(F * frac_above(Em, Ethr, al) for F, Em, al in self.comps)

# --- DSNB (recalled): all-flavour number flux ~30 cm^-2 s^-1 (uncertain x3); redshift-softened <E>.
#     Upper bracket: each flavour saturating the Super-K nu_e-bar limit 2.7 cm^-2 s^-1 above 17.3 MeV (Bays et al. 2012; likely)
DSNB_TOTAL = 30.0
def dsnb(name, Emeans, total=None, sk_saturate=False):
    comps = []
    for Em, share in zip(Emeans, (1 / 6, 1 / 6, 4 / 6)):
        if sk_saturate:
            F = 2.7 / frac_above(Em, 17.3) * (1 if share < 0.5 else 4)   # each species at the nu_e-bar limit
        else:
            F = total * share
        comps.append((F, Em, ALPHA))
    return Source(name, comps, burst=False)
SOURCES = {
    "DSNB central (<E> 9/11/13 MeV, 30 cm-2 s-1)": dsnb("DSNB central", (9.0, 11.0, 13.0), DSNB_TOTAL),
    "DSNB hard (<E> 12/15/18 MeV, 30 cm-2 s-1)": dsnb("DSNB hard", (12.0, 15.0, 18.0), DSNB_TOTAL),
    "DSNB very hard (<E> 12/15/25 MeV, 30 cm-2 s-1)": dsnb("DSNB very hard", (12.0, 15.0, 25.0), DSNB_TOTAL),
    "DSNB SK-limit-saturating (<E> 12/15/18)": dsnb("DSNB SK-sat", (12.0, 15.0, 18.0), sk_saturate=True),
    "DSNB SK-limit-saturating very hard (12/15/25)": dsnb("DSNB SK-sat very hard", (12.0, 15.0, 25.0), sk_saturate=True),
}
# --- Galactic core-collapse SN: E_tot = 3e53 erg in neutrinos (recalled, certain to x1.5), equipartition over 6 species
E_SN_ERG = 3e53
def sn_burst(name, d_kpc, Emeans=(12.0, 15.0, 18.0)):
    d = d_kpc * KPC_CM; comps = []
    for Em, mult in zip(Emeans, (1, 1, 4)):
        N_nu = E_SN_ERG / 6 / (Em * ERG_PER_MEV) * mult
        comps.append((N_nu / (4 * math.pi * d * d), Em, ALPHA))
    return Source(name, comps, burst=True)
SOURCES["Galactic SN 10 kpc (<E> 12/15/18)"] = sn_burst("SN 10 kpc", 10.0)
SOURCES["Galactic SN 10 kpc hard (<E> 15/18/25)"] = sn_burst("SN 10 kpc hard", 10.0, (15.0, 18.0, 25.0))
SOURCES["Galactic SN 1 kpc (<E> 12/15/18)"] = sn_burst("SN 1 kpc", 1.0)
SOURCES["Betelgeuse-like SN 0.2 kpc (<E> 12/15/18)"] = sn_burst("SN 0.2 kpc", 0.2)
SOURCES["SN 2023ixf, M101 6.4 Mpc, 19 May 2023 (<E> 12/15/18)"] = sn_burst("SN 2023ixf", 6400.0)
for k, s in SOURCES.items():
    log(f"{k:55s} total {'fluence' if s.burst else 'flux'} {s.total():.3e}  above 17.3 MeV {s.above(17.3):.3e}  above 30 MeV {s.above(30):.3e}  above 110 MeV {s.above(110):.3e}")
R["source_flux_summary"] = {k: dict(total=s.total(), above_17p3=s.above(17.3), above_30=s.above(30), above_60=s.above(60), above_110=s.above(110), burst=s.burst)
                            for k, s in SOURCES.items()}

# ------------------------------------------------------------------------------------------------
# C. Coherent CEvNS counts.  dsigma/dE_R = G_F^2 M/(4 pi) Q_W^2 (1 - M E_R/2E^2) F_Helm^2   (P019 convention)
# ------------------------------------------------------------------------------------------------
E_GRID = np.concatenate([np.arange(0.25, 10, 0.25), np.arange(10, 100, 1.0), np.arange(100, 330, 2.0)])
def cevns_spectrum(src: Source, E_grid=E_GRID):
    """dN/dE_R [events per keV] for the whole FV; for steady sources over the 220 live days, for bursts per burst."""
    out = np.zeros_like(E_grid, float)
    tfac = LIVE_S if not src.burst else 1.0
    for A, f in ISO.items():
        M = mN_gev(A); NT = n_per_tonne(A, f) * MASS_T
        pref = GF ** 2 * M / (4 * math.pi) * QW(A) ** 2 * GEV_TO_CM2 * 1e-6      # cm^2/keV per unit kinematic factor
        Mm = M * 1e3
        for i, E in enumerate(E_grid):
            emin = Enu_min_MeV(E, A)
            Emax = 60 * max(Em for _, Em, _ in src.comps)
            if emin >= Emax: continue
            g = lambda e: src.phi(e) * max(1 - Mm * E * 1e-3 / (2 * e * e), 0.0)
            I, _ = integrate.quad(g, emin, Emax, limit=300, points=[p for p in (2 * emin, 5 * emin) if emin < p < Emax] or None)
            out[i] += NT * pref * lz.helm_F2(E, A) * I * tfac
    return out
def window_counts(spec, E_grid=E_GRID):
    w = eff(E_grid)
    def N(a, b, weighted=True):
        m = (E_grid >= a) & (E_grid <= b); Eg = np.concatenate([[a], E_grid[m], [b]])
        sp = np.interp(Eg, E_grid, spec * (w if weighted else 1.0)); return float(np.trapezoid(sp, Eg))
    return dict(N_all_raw=N(0.25, 329, False), N_ROI_eff=N(5.4, 270), N_lo_5p4_55=N(5.4, 55), N_lo_5p4_200=N(5.4, 200),
                N_200_270=N(200, 270), N_225_271=N(225, 271), N_above_100=N(100, 329), N_above_50=N(50, 329))
coh = {}
for k, s in SOURCES.items():
    sp = cevns_spectrum(s); wc = window_counts(sp); coh[k] = wc
    log(f"[coh] {k:55s} N_all {wc['N_all_raw']:.3e}  ROI {wc['N_ROI_eff']:.3e}  >50 {wc['N_above_50']:.2e}  >100 {wc['N_above_100']:.2e}  200-270 {wc['N_200_270']:.2e}  225-271 {wc['N_225_271']:.2e}")
R["coherent_counts"] = coh
pd.DataFrame([dict(source=k, **v) for k, v in coh.items()]).to_csv(f"{OUT}/P060_coherent_counts.csv", index=False)
# sanity: Galactic SN CEvNS per tonne (recalled literature: tens of events per tonne of Xe for 10 kpc; likely)
R["SN10kpc_events_per_tonne_all_ER"] = coh["Galactic SN 10 kpc (<E> 12/15/18)"]["N_all_raw"] / MASS_T
log(f"Galactic SN 10 kpc: {R['SN10kpc_events_per_tonne_all_ER']:.1f} CEvNS recoils per tonne (all E_R), {coh['Galactic SN 10 kpc (<E> 12/15/18)']['N_ROI_eff']:.1f} in the ROI with efficiency")

# ------------------------------------------------------------------------------------------------
# D. Incoherent lone-NR channel (P019 Sec. 4b) with explicit kinematic threshold
#    NC quasi-elastic n knock-out; residual nucleus recoils with the hole momentum p: E_res = p^2/2M_{A-1}.
#    Free-nucleon Llewellyn-Smith NC elastic cross-section with Pauli factor (copied from P019 for consistency).
# ------------------------------------------------------------------------------------------------
MA, MV = 1.03, 0.84; gA = 1.267; mu_p, mu_n = 2.793, -1.913
def ff_nc(Q2, nucleon):
    GD_V = (1 + Q2 / MV ** 2) ** -2; GD_A = (1 + Q2 / MA ** 2) ** -2; tau = Q2 / (4 * M_NUC ** 2)
    GEp, GMp, GEn, GMn = GD_V, mu_p * GD_V, 0.0, mu_n * GD_V
    F1p = (GEp + tau * GMp) / (1 + tau); F2p = (GMp - GEp) / (1 + tau)
    F1n = (GEn + tau * GMn) / (1 + tau); F2n = (GMn - GEn) / (1 + tau)
    s = +1 if nucleon == "p" else -1
    F1 = s * 0.5 * (F1p - F1n) - 2 * S2W * (F1p if nucleon == "p" else F1n)
    F2 = s * 0.5 * (F2p - F2n) - 2 * S2W * (F2p if nucleon == "p" else F2n)
    return F1, F2, s * 0.5 * gA * GD_A
def dsig_dQ2_nce(Enu, Q2, nucleon, antinu=False):
    M = M_NUC; F1, F2, FA = ff_nc(Q2, nucleon); tau = Q2 / (4 * M * M)
    A = 4 * tau * ((1 + tau) * FA ** 2 - (1 - tau) * F1 ** 2 + tau * (1 - tau) * F2 ** 2 + 4 * tau * F1 * F2)
    B = 4 * tau * FA * (F1 + F2); C = 0.25 * (FA ** 2 + F1 ** 2 + tau * F2 ** 2)
    su = 4 * M * Enu - Q2; sign = -1 if antinu else +1
    return GF ** 2 * M ** 2 / (8 * math.pi * Enu ** 2) * (A + sign * B * su / M ** 2 + C * su ** 2 / M ** 4) * GEV_TO_CM2
def Q2max(Enu): return 4 * Enu ** 2 * M_NUC / (M_NUC + 2 * Enu)
KF = 0.26
def sigma_nce(Enu_GeV, nucleon="n", pauli=True):
    def pf(Q2):
        q = math.sqrt(Q2 + (Q2 / (2 * M_NUC)) ** 2); x = q / (2 * KF)
        return 1.0 if x >= 1 else 1.5 * x - 0.5 * x ** 3
    f = lambda Q2: 0.5 * (dsig_dQ2_nce(Enu_GeV, Q2, nucleon, False) + dsig_dQ2_nce(Enu_GeV, Q2, nucleon, True)) * (pf(Q2) if pauli else 1.0)
    I, _ = integrate.quad(f, 0, Q2max(Enu_GeV), limit=200); return I
_SIG_E = np.geomspace(10, 3000, 70); _SIG_N = np.array([sigma_nce(e * 1e-3, "n") for e in _SIG_E])
def sig_n(E_MeV): return np.interp(E_MeV, _SIG_E, _SIG_N)
log("sigma_NCE(n, Pauli) at 20/30/50/100/200 MeV: " + ", ".join(f"{sig_n(e):.2e}" for e in (20, 30, 50, 100, 200)) + " cm^2")

S_N = 8.5           # neutron separation energy of Xe isotopes, MeV (recalled, likely; 7.9-9.6 across isotopes)
M_RES = mN_gev(130) * 1e3
kF_MeV = 260.0
E_RES_MAX = kF_MeV ** 2 / (2 * M_RES) * 1e3
def p_of_Eres(E_keV): return math.sqrt(2 * M_RES * E_keV * 1e-3)
def p_max_reachable(Enu, E_res_keV=248.0):
    """Largest hole momentum p whose knock-out is kinematically allowed at E_nu: impulse approximation, final neutron free
    with T_f = |p+q|^2/2m_n, energy conservation E_nu = E_nu' + T_f + S_n + E_res, q <= E_nu + E_nu'.
    Maximum over E_nu' is at E_nu' -> 0: p_max = E_nu + sqrt(2 m_n (E_nu - S_n - E_res))."""
    avail = Enu - S_N - E_res_keV * 1e-3
    return 0.0 if avail <= 0 else Enu + math.sqrt(2 * M_N_MEV * avail)
def Enu_threshold(E_res_keV):
    p = p_of_Eres(E_res_keV)
    return optimize.brentq(lambda E: p_max_reachable(E, E_res_keV) - p, S_N + E_res_keV * 1e-3 + 1e-6, 1000.0)
thr = {e: Enu_threshold(e) for e in (100, 150, 200, 225, 248, 271, 279)}
R["fermi_recoil"] = dict(S_n_MeV=S_N, kF_MeV=kF_MeV, E_res_max_keV=E_RES_MAX, p_window_MeV=[p_of_Eres(225), p_of_Eres(271)],
                         Enu_threshold_MeV={f"{k} keV": v for k, v in thr.items()},
                         Enu_threshold_coherent_248=Enu_min_MeV(248))
log("Fermi-recoil channel thresholds E_nu,thr(E_res): " + ", ".join(f"{k} keV: {v:.1f} MeV" for k, v in thr.items()))
P_ESC, P_UNT, P_GS = 0.3, 0.2, 0.3        # P019 central factors (ranges 0.1-0.35, 0.08-0.3, 0.1-0.5)
LONE_LO = 0.1 * 0.08 * 0.1; LONE_C = P_ESC * P_UNT * P_GS; LONE_HI = 0.35 * 0.3 * 0.5
def f_win(Enu, lo=225.0, hi=271.0):
    plo, phi = p_of_Eres(lo), min(p_of_Eres(hi), kF_MeV)
    pm = min(p_max_reachable(Enu, lo), phi)
    return 0.0 if pm <= plo else (pm ** 3 - plo ** 3) / kF_MeV ** 3
R["fermi_recoil"]["f_win_ungated"] = (min(p_of_Eres(271), kF_MeV) ** 3 - p_of_Eres(225) ** 3) / kF_MeV ** 3
R["fermi_recoil"]["f_win_at_E"] = {E: f_win(E) for E in (25, 30, 32, 35, 40, 50, 60, 100, 200)}
def lone_NR_count(src: Source, Emax=None):
    """Expected lone 225-271 keV residual-nucleus recoils for the source (steady: over 220 d; burst: per burst)."""
    tfac = LIVE_S if not src.burst else 1.0
    Emax = Emax or 60 * max(Em for _, Em, _ in src.comps)
    N_n = sum(n_per_tonne(A, f) * (A - Z_XE) for A, f in ISO.items()) * MASS_T     # neutrons in FV
    g = lambda e: src.phi(e) * sig_n(e) * f_win(e)
    Emin = thr[225]
    if Emin >= Emax: return 0.0, 0.0
    I, _ = integrate.quad(g, Emin, Emax, limit=300, points=[p for p in (1.5 * Emin, 3 * Emin) if Emin < p < Emax] or None)
    knock = N_n * tfac * integrate.quad(lambda e: src.phi(e) * sig_n(e), 10.0, Emax, limit=300)[0]   # all n knock-outs (Pauli only)
    return N_n * tfac * I * LONE_C, knock
inc = {}
for k, s in SOURCES.items():
    N_lone, knock = lone_NR_count(s)
    inc[k] = dict(N_lone_225_271_central=N_lone, range=[N_lone / LONE_C * LONE_LO, N_lone / LONE_C * LONE_HI], n_knockouts_all=knock)
    log(f"[inc] {k:55s} n knock-outs {knock:.3e}; lone-NR 225-271 keV {N_lone:.3e} [{inc[k]['range'][0]:.1e}, {inc[k]['range'][1]:.1e}]")
R["incoherent_counts"] = inc
# validation against P019: atmospheric central shape with P019's normalisation (C_norm 0.05339 cm^-2 s^-1 MeV^-1 at E_b = 100 MeV)
P019 = json.load(open("output/work/P019/P019_results.json"))
C_atm = [s for s in P019["cevns_summary"] if s["shape"].startswith("central") and s["ffmodel"] == "helm"][0]["C_norm"]
def atm_phi(E):
    E = np.asarray(E, float); out = np.where(E < 100, (E / 100.0) ** -1.0, (E / 100.0) ** -2.5)
    return C_atm * np.where(E < 10, 0.0, out)
class AtmSrc:
    burst = False; comps = [(1.0, 1e4 / 60, 1.0)]
    def phi(self, E): return atm_phi(E)
N_lone_atm, knock_atm = lone_NR_count(AtmSrc(), Emax=1e4)
R["validation_vs_P019"] = dict(atm_lone_NR_gated=N_lone_atm, P019_lone_NR=P019["knockout"]["N_lone_NR_225_271"],
                               atm_n_knockouts=knock_atm, P019_n_knockouts=P019["knockout"]["N_inc_n_exposure"],
                               P019_coherent_225_271=P019["headline"]["N_225_271_central_helm"], P019_N_lo_per_hi=P019["headline"]["N_lo_per_hi_central"])
log(f"Validation: atmospheric lone-NR with gate {N_lone_atm:.2e} vs P019 {P019['knockout']['N_lone_NR_225_271']:.2e}; knock-outs {knock_atm:.3e} vs P019 {P019['knockout']['N_inc_n_exposure']:.3e}")

# ------------------------------------------------------------------------------------------------
# E. IceCube diffuse astrophysical flux: DIS interactions in the FV (recalled inputs, likely x2)
#    per-flavour E^2 Phi = 1.0e-8 (E/100 TeV)^(-0.5) GeV cm^-2 s^-1 sr^-1  (gamma = 2.5), 3 flavours, 4 pi
#    sigma_CC(E) = 0.68e-38 E [GeV] (1 + E/E_W)^(-0.637) cm^2 with E_W = 3.7e4 GeV (matches 5.53e-36 E^0.363 at 10 PeV); NC = 0.4 CC
# ------------------------------------------------------------------------------------------------
def phi_ic(E_GeV): return 1.0e-8 * (E_GeV / 1e5) ** -0.5 / E_GeV ** 2          # GeV^-1 cm^-2 s^-1 sr^-1 per flavour
def sig_tot(E_GeV): return 0.68e-38 * E_GeV * (1 + E_GeV / 3.7e4) ** -0.637 * 1.4
N_NUCLEONS = MASS_T * 1e6 / 1.66054e-24
I_ic, _ = integrate.quad(lambda e: phi_ic(e) * sig_tot(e), 1e3, 1e7, limit=300)
N_ic = N_NUCLEONS * 4 * math.pi * 3 * LIVE_S * I_ic
# fraction of NC events with hadronic energy < 300 keV: y < 3e-4 MeV / E; dsigma/dy ~ flat at small y -> fraction ~ y_max
E_typ = 1e5; y_max = 0.3e-3 / (E_typ * 1e3)
R["icecube"] = dict(E2Phi_100TeV_per_flavour=1e-8, gamma=2.5, sigma_tot_100TeV_cm2=sig_tot(1e5), sigma_tot_10TeV=sig_tot(1e4), sigma_tot_1PeV=sig_tot(1e6),
                    nucleons_FV=N_NUCLEONS, N_interactions_220d_1TeV_10PeV=N_ic, N_from_100TeV_up=N_NUCLEONS * 4 * math.pi * 3 * LIVE_S * integrate.quad(lambda e: phi_ic(e) * sig_tot(e), 1e5, 1e7)[0],
                    y_max_for_300keV_at_100TeV=y_max, N_with_Ehad_below_300keV=N_ic * y_max * 0.4 / 1.4)
log(f"IceCube diffuse: sigma_tot(100 TeV) = {sig_tot(1e5):.2e} cm^2; interactions in FV over 220 d (1 TeV-10 PeV) = {N_ic:.2e}; with E_had < 300 keV: {R['icecube']['N_with_Ehad_below_300keV']:.1e}")

# ------------------------------------------------------------------------------------------------
# F. Transient hypothesis: required fluence at Earth for one 200-270 keV CEvNS event in the FV
# ------------------------------------------------------------------------------------------------
def sigma_window(Enu_MeV, lo=200.0, hi=270.0, weighted=True):
    """Efficiency-weighted CEvNS cross-section into [lo, hi] keV per natural-Xe nucleus (abundance-weighted), cm^2."""
    tot = 0.0
    for A, f in ISO.items():
        M = mN_gev(A); Mm = M * 1e3
        pref = GF ** 2 * M / (4 * math.pi) * QW(A) ** 2 * GEV_TO_CM2 * 1e-6
        g = lambda E: pref * lz.helm_F2(E, A) * max(1 - Mm * E * 1e-3 / (2 * Enu_MeV ** 2), 0.0) * (float(eff(E)) if weighted else 1.0)
        I, _ = integrate.quad(g, lo, min(hi, ERmax_keV(Enu_MeV, A)) if ERmax_keV(Enu_MeV, A) > lo else lo, limit=200)
        tot += f * I
    return tot
trans = []
for Enu in (125, 130, 140, 150, 175, 200, 250, 300, 500, 1000, 3000):
    s_win = sigma_window(Enu); s_lo = sigma_window(Enu, 5.4, 200.0); s_lo55 = sigma_window(Enu, 5.4, 55.0); s_all = sigma_window(Enu, 0.01, 1e4, weighted=False)
    NT = N_T_TONNE * MASS_T
    F_req = 1 / (NT * s_win) if s_win > 0 else float("inf")
    trans.append(dict(Enu_MeV=Enu, sigma_window_cm2=s_win, sigma_total_cm2=s_all, F_req_cm2=F_req, N_lo_5p4_200_per_window_event=s_lo / s_win if s_win > 0 else float("nan"),
                      N_lo_5p4_55_per_window_event=s_lo55 / s_win if s_win > 0 else float("nan"),
                      E_budget_erg_10kpc=F_req * 4 * math.pi * (10 * KPC_CM) ** 2 * Enu * ERG_PER_MEV,
                      E_budget_erg_1kpc=F_req * 4 * math.pi * (1 * KPC_CM) ** 2 * Enu * ERG_PER_MEV,
                      E_budget_erg_100pc=F_req * 4 * math.pi * (0.1 * KPC_CM) ** 2 * Enu * ERG_PER_MEV,
                      E_budget_erg_1AU=F_req * 4 * math.pi * (1.496e13) ** 2 * Enu * ERG_PER_MEV))
    log(f"E_nu = {Enu:5d} MeV: sigma(200-270, eff) = {s_win:.2e} cm^2, F_req = {F_req:.2e} cm^-2, N_lo(5.4-200) = {trans[-1]['N_lo_5p4_200_per_window_event']:.0f}, N_lo(5.4-55) = {trans[-1]['N_lo_5p4_55_per_window_event']:.0f}, E(10 kpc) = {trans[-1]['E_budget_erg_10kpc']:.1e} erg")
pd.DataFrame(trans).to_csv(f"{OUT}/P060_transient_required_fluence.csv", index=False)
R["transient"] = trans
# spectrum-averaged: P019 central atmospheric shape restricted to E > 110 MeV, and flat E dPhi/dE 130 MeV-1 GeV
def F_req_spectrum(shape_fn, Emin, Emax):
    norm_, _ = integrate.quad(shape_fn, Emin, Emax, limit=200)
    Eg = np.geomspace(Emin, Emax, 60); w = np.array([shape_fn(e) for e in Eg]) / norm_
    s_win = np.trapezoid(w * np.array([sigma_window(e) for e in Eg]), Eg); s_lo = np.trapezoid(w * np.array([sigma_window(e, 5.4, 200.0) for e in Eg]), Eg)
    return 1 / (N_T_TONNE * MASS_T * s_win), s_lo / s_win
F_atm_shape, Nlo_atm_shape = F_req_spectrum(lambda e: (e / 100) ** -2.5, 110.0, 1e4)
F_flat, Nlo_flat = F_req_spectrum(lambda e: 1 / e, 130.0, 1000.0)
R["transient_spectra"] = dict(atm_like_above_110MeV=dict(F_req=F_atm_shape, N_lo=Nlo_atm_shape), flat_logE_130_1000=dict(F_req=F_flat, N_lo=Nlo_flat))
log(f"Spectrum-averaged F_req: atmospheric-like E^-2.5 above 110 MeV {F_atm_shape:.2e} cm^-2 (N_lo {Nlo_atm_shape:.0f}); flat in log E 130 MeV-1 GeV {F_flat:.2e} (N_lo {Nlo_flat:.0f})")

# Comparison fluences (recalled, flagged in details.md)
F_ref = dict(
    galactic_SN_10kpc_all_flavour=SOURCES["Galactic SN 10 kpc (<E> 12/15/18)"].total(),
    galactic_SN_10kpc_above_110MeV=SOURCES["Galactic SN 10 kpc hard (<E> 15/18/25)"].above(110),
    SN2023ixf_all_flavour=SOURCES["SN 2023ixf, M101 6.4 Mpc, 19 May 2023 (<E> 12/15/18)"].total(),
    magnetar_giant_flare_all_energy_into_200MeV_nu=2e46 / (200 * ERG_PER_MEV) / (4 * math.pi * (10 * KPC_CM) ** 2),   # SGR 1806-20-like 2e46 erg at 10 kpc (recalled, likely)
    GRB_neutrino_number_fluence_TeV_PeV=1e-3 / 1e5,     # E^2 F ~ 1e-3 GeV cm^-2 at 100 TeV -> ~1e-8 cm^-2 (recalled, uncertain)
    solar_flare_model_fluence_upper=1e4,                # model fluences at Earth for the largest flares, <= 1e2-1e4 cm^-2 (recalled, uncertain)
    solar_flare_SK_limit_like=1e7,                      # SK per-flare limits ~1e5-1e7 cm^-2 (recalled, uncertain)
)
F_req_150 = [t for t in trans if t["Enu_MeV"] == 150][0]["F_req_cm2"]; F_req_300 = [t for t in trans if t["Enu_MeV"] == 300][0]["F_req_cm2"]
R["reference_fluences"] = F_ref
R["expected_if_all_at_150MeV"] = {k: v / F_req_150 for k, v in F_ref.items()}
R["expected_if_all_at_300MeV"] = {k: v / F_req_300 for k, v in F_ref.items()}
for k, v in F_ref.items(): log(f"reference fluence {k:48s} {v:.2e} cm^-2 -> events if all at 150 MeV: {v/F_req_150:.1e}, at 300 MeV: {v/F_req_300:.1e}")

# Super-K response to F_req (nu_e-bar IBD only, 1/6 of the fluence): sigma_IBD ~ 9.52e-44 E_e p_e cm^2 (Vogel-Beacom; recalled, certain in form,
# overestimates by <~x2 at 150-300 MeV); free protons in 22.5 kt: 2/18 x 22.5e9 g x N_A
N_P_SK = 2 / 18 * 22.5e9 * 6.02214e23
def sig_ibd(E): Ee = E - 1.293; return 9.52e-44 * Ee * math.sqrt(max(Ee * Ee - 0.511 ** 2, 0)) if Ee > 0.511 else 0.0
R["superK"] = {f"{E} MeV": dict(sigma_IBD=sig_ibd(E), N_SK_events=N_P_SK * F / 6 * sig_ibd(E))
               for E, F in ((150, F_req_150), (300, F_req_300))}
log("Super-K IBD events for F_req (nu_e-bar = 1/6): " + ", ".join(f"{k}: {v['N_SK_events']:.1e}" for k, v in R["superK"].items()))
# LZ itself: within the burst, a CEvNS population of N_lo low-energy recoils (from trans table) and, on the same footing, the
# number of coherent recoils in the ROI: quote for 150 and 300 MeV
R["LZ_companions"] = {f"{t['Enu_MeV']} MeV": dict(N_lo_5p4_200=t["N_lo_5p4_200_per_window_event"], N_lo_5p4_55=t["N_lo_5p4_55_per_window_event"]) for t in trans}

# ------------------------------------------------------------------------------------------------
# G. Summary table
# ------------------------------------------------------------------------------------------------
c10 = coh["Galactic SN 10 kpc (<E> 12/15/18)"]; c10h = coh["Galactic SN 10 kpc hard (<E> 15/18/25)"]
dsnb_c = coh["DSNB central (<E> 9/11/13 MeV, 30 cm-2 s-1)"]; dsnb_h = coh["DSNB SK-limit-saturating very hard (12/15/25)"]
summary = [
    dict(source="reactor / geo antineutrinos", Enu_range_MeV="<= 10", ERmax_keV=f"{ERmax_keV(10):.1f}", coherent_200_270="0 (kinematic)", incoherent_lone_NR="0 (below knock-out threshold)", verdict="excluded by kinematics"),
    dict(source="solar 8B / hep", Enu_range_MeV="<= 18.8", ERmax_keV=f"{ERmax_keV(18.8):.1f}", coherent_200_270="0 (kinematic); LZ Table I 0.057 in ROI", incoherent_lone_NR="0 (E_nu < %.0f MeV)" % thr[225], verdict="excluded by kinematics (P019)"),
    dict(source="DSNB (steady)", Enu_range_MeV="~5-60, tail", ERmax_keV=f"{ERmax_keV(60):.0f} at 60 MeV",
         coherent_200_270=f"{dsnb_c['N_200_270']:.0e} (central) - {dsnb_h['N_200_270']:.0e} (SK-saturating, very hard)",
         incoherent_lone_NR=f"{inc['DSNB central (<E> 9/11/13 MeV, 30 cm-2 s-1)']['N_lone_225_271_central']:.0e} - {inc['DSNB SK-limit-saturating very hard (12/15/25)']['N_lone_225_271_central']:.0e}", verdict="negligible"),
    dict(source="Galactic core-collapse SN, 10 kpc (none occurred in 2023)", Enu_range_MeV="~5-80", ERmax_keV=f"{ERmax_keV(60):.0f} at 60 MeV",
         coherent_200_270=f"{c10['N_200_270']:.0e} - {c10h['N_200_270']:.0e} per burst (with {c10['N_ROI_eff']:.0f} ROI recoils in ~10 s)",
         incoherent_lone_NR=f"{inc['Galactic SN 10 kpc (<E> 12/15/18)']['N_lone_225_271_central']:.0e} - {inc['Galactic SN 10 kpc hard (<E> 15/18/25)']['N_lone_225_271_central']:.0e} per burst", verdict="no burst occurred; would be accompanied by ~%.0f low-energy recoils" % c10["N_ROI_eff"]),
    dict(source="SN 2023ixf (M101, 6.4 Mpc, 19 May 2023)", Enu_range_MeV="~5-60", ERmax_keV=f"{ERmax_keV(60):.0f}",
         coherent_200_270=f"{coh['SN 2023ixf, M101 6.4 Mpc, 19 May 2023 (<E> 12/15/18)']['N_200_270']:.0e}", incoherent_lone_NR=f"{inc['SN 2023ixf, M101 6.4 Mpc, 19 May 2023 (<E> 12/15/18)']['N_lone_225_271_central']:.0e}", verdict="28 d early, 1e6 too faint, kinematically blind"),
    dict(source="solar-flare neutrinos", Enu_range_MeV="~10-300 (pion decay)", ERmax_keV="reaches window only above 123 MeV",
         coherent_200_270=f"<= {F_ref['solar_flare_SK_limit_like']/F_req_150:.0e} (SK-limit fluence 1e7 all at 150 MeV); model ~{F_ref['solar_flare_model_fluence_upper']/F_req_150:.0e}", incoherent_lone_NR="smaller", verdict="negligible; flare occurrence irrelevant"),
    dict(source="atmospheric nu (steady; P019)", Enu_range_MeV="10-1e4", ERmax_keV="unbounded", coherent_200_270="6.5e-5 (1.7e-5 in 225-271)", incoherent_lone_NR="1.7e-4 (8e-6-5e-4)", verdict="dominant neutrino channel, still <= 5e-4 (P019)"),
    dict(source="IceCube diffuse astrophysical nu", Enu_range_MeV="1e6-1e10", ERmax_keV="unbounded", coherent_200_270=f"{R['icecube']['N_interactions_220d_1TeV_10PeV']:.0e} interactions total, GeV-PeV deposits; < 300 keV: {R['icecube']['N_with_Ehad_below_300keV']:.0e}", incoherent_lone_NR="-", verdict="negligible"),
    dict(source="hypothetical burst at 21:22 UTC (E_nu >= 130 MeV)", Enu_range_MeV=">= 123", ERmax_keV=">= 248",
         coherent_200_270=f"needs F = {F_req_150:.0e} ({F_req_300:.0e}) cm^-2 at 150 (300) MeV", incoherent_lone_NR="-",
         verdict=f"needs {trans[3]['E_budget_erg_10kpc']:.0e} erg at 10 kpc; ~{R['superK']['150 MeV']['N_SK_events']:.0e} Super-K events; {trans[3]['N_lo_5p4_200_per_window_event']:.0f} LZ companions"),
]
pd.DataFrame(summary).to_csv(f"{OUT}/P060_summary_table.csv", index=False)
R["summary_table"] = summary

# ------------------------------------------------------------------------------------------------
# Figures (matplotlib; dataviz conventions: fixed categorical order, thin marks, direct labels, single axis)
# ------------------------------------------------------------------------------------------------
C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
Eg = np.geomspace(1, 1e10, 500)
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.plot(Eg, [ERmax_keV(e) for e in Eg], color=C1, lw=2, label="coherent CEνNS: E_R,max = 2E²/(m_N+2E)")
ax.axhspan(225, 271, color="#d9d9d6", alpha=0.7, lw=0); ax.text(1.15, 130, "LZ event 248 ± 23 keV", color="#52514e", fontsize=8)
ax.axhline(E_RES_MAX, color=C2, lw=1.5, ls="--"); ax.text(1.15, E_RES_MAX * 2.2, f"incoherent lone-NR ceiling k_F²/2M = {E_RES_MAX:.0f} keV (P019)", color=C2, fontsize=8)
ax.axvline(thr[225], color=C2, lw=1, ls=":"); ax.text(thr[225] * 0.9, 3e6, f"lone-NR\nthreshold\n{thr[225]:.0f} MeV", color=C2, fontsize=7.5, ha="right")
ax.axvline(Enu_min_MeV(248), color=C1, lw=1, ls=":"); ax.text(Enu_min_MeV(248) * 1.1, 3e6, f"coherent\nthreshold\n{Enu_min_MeV(248):.0f} MeV", color=C1, fontsize=7.5)
bands = [("reactor/geo", 1.8, 10, 0.02), ("solar ⁸B/hep", 5, 18.8, 0.004), ("DSNB / SN", 5, 60, 0.6), ("solar flare", 10, 300, 40),
         ("atmospheric (P019)", 10, 1e4, 3000), ("IceCube diffuse", 1e6, 1e10, 1e8)]
for name, lo, hi, y in bands:
    ax.plot([lo, hi], [y, y], color="#52514e", lw=4, solid_capstyle="butt", alpha=0.35)
    ax.text(lo, y * 1.35, name, fontsize=7.5, color="#52514e")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1, 1e10); ax.set_ylim(1e-3, 1e12)
ax.set_xlabel("neutrino energy E_ν [MeV]"); ax.set_ylabel("maximum xenon recoil energy [keV]")
ax.set_title("Which neutrinos can reach 248 keV in xenon?", loc="left"); ax.grid(alpha=0.2, which="major"); ax.legend(loc="lower right", fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(f"{FIG}/P060_ERmax_vs_Enu.png", dpi=160); plt.close(fig)

# counts figure: horizontal bars (log), coherent vs incoherent per source
rows = [
    ("DSNB, central", dsnb_c["N_200_270"], inc["DSNB central (<E> 9/11/13 MeV, 30 cm-2 s-1)"]["N_lone_225_271_central"]),
    ("DSNB, SK-saturating very hard", dsnb_h["N_200_270"], inc["DSNB SK-limit-saturating very hard (12/15/25)"]["N_lone_225_271_central"]),
    ("Galactic SN 10 kpc (per burst; none in 2023)", c10["N_200_270"], inc["Galactic SN 10 kpc (<E> 12/15/18)"]["N_lone_225_271_central"]),
    ("Galactic SN 10 kpc, hard (per burst)", c10h["N_200_270"], inc["Galactic SN 10 kpc hard (<E> 15/18/25)"]["N_lone_225_271_central"]),
    ("SN 2023ixf, 6.4 Mpc (per burst)", coh["SN 2023ixf, M101 6.4 Mpc, 19 May 2023 (<E> 12/15/18)"]["N_200_270"], inc["SN 2023ixf, M101 6.4 Mpc, 19 May 2023 (<E> 12/15/18)"]["N_lone_225_271_central"]),
    ("solar flare, SK-limit fluence at 150 MeV", F_ref["solar_flare_SK_limit_like"] / F_req_150, np.nan),
    ("atmospheric, steady (P019)", P019["headline"]["N_above_200_central_helm"], P019["knockout"]["N_lone_NR_225_271"]),
    ("IceCube diffuse, E_had < 300 keV", R["icecube"]["N_with_Ehad_below_300keV"], np.nan),
]
fig, ax = plt.subplots(figsize=(7.6, 4.4))
y = np.arange(len(rows))[::-1]; floor = 1e-16
for i, (name, c, ic) in zip(y, rows):
    cc = max(c, floor); ax.barh(i + 0.18, cc, left=floor, height=0.34, color=C1, lw=0)
    ax.text(cc * 1.5, i + 0.18, f"{c:.0e}", va="center", fontsize=7.5, color="#0b0b0b")
    if np.isfinite(ic):
        icc = max(ic, floor); ax.barh(i - 0.18, icc, left=floor, height=0.34, color=C2, lw=0)
        ax.text(icc * 1.5, i - 0.18, f"{ic:.0e}", va="center", fontsize=7.5, color="#0b0b0b")
ax.axvline(1.0, color="#0b0b0b", lw=1, ls="--"); ax.text(1.0, len(rows) - 0.4, " one event", fontsize=8)
ax.set_xscale("log"); ax.set_xlim(floor, 30); ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
ax.set_xlabel("expected events in the LZ window (200–270 keV coherent; 225–271 keV lone-NR), 4.71 t × 220 d")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C1, label="coherent CEνNS, 200–270 keV"), Patch(color=C2, label="incoherent lone-NR, 225–271 keV (P019 channel)")], loc="lower right", fontsize=8, frameon=False)
ax.set_title("Expected events per source (4.71 t × 220 d; bursts per burst)", loc="left", fontsize=9)
ax.grid(axis="x", alpha=0.2); fig.tight_layout(); fig.savefig(f"{FIG}/P060_expected_counts.png", dpi=160); plt.close(fig)

# required fluence figure
fig, ax = plt.subplots(figsize=(6.4, 4.2))
d = pd.DataFrame(trans)
ax.plot(d.Enu_MeV, d.F_req_cm2, color=C1, lw=2, marker="o", ms=4, label="fluence for one 200–270 keV CEνNS event in LZ")
for name, F, col, dy in (("Galactic SN 10 kpc, all flavours (but E ≈ 15 MeV)", F_ref["galactic_SN_10kpc_all_flavour"], C3, 0.25),
                         ("Galactic SN 10 kpc, above 110 MeV (hard spectrum)", F_ref["galactic_SN_10kpc_above_110MeV"], C3, 1.4),
                         ("solar flare, SK-limit-like (any energy)", F_ref["solar_flare_SK_limit_like"], C4, 0.25),
                         ("magnetar giant flare, all 2e46 erg into 200 MeV ν", F_ref["magnetar_giant_flare_all_energy_into_200MeV_nu"], C5, 1.4)):
    ax.axhline(F, color=col, lw=1.2, ls="--"); ax.text(128, F * dy, name, color=col, fontsize=7.5)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("neutrino energy E_ν [MeV] (monoenergetic)"); ax.set_ylabel("fluence at Earth [cm⁻²]")
ax.set_ylim(1e-1, 1e15); ax.set_title("A burst explanation needs ≳10¹² ν cm⁻² above 123 MeV", loc="left"); ax.legend(fontsize=8, frameon=False, loc="upper right"); ax.grid(alpha=0.2)
fig.tight_layout(); fig.savefig(f"{FIG}/P060_required_fluence.png", dpi=160); plt.close(fig)

R["headline"] = dict(ERmax_60MeV_keV=ERmax_keV(60), Enu_min_248=Enu_min_MeV(248), lone_NR_threshold_225keV_MeV=thr[225], lone_NR_threshold_248keV_MeV=thr[248],
                     DSNB_coherent_window_range=[dsnb_c["N_200_270"], dsnb_h["N_200_270"]],
                     DSNB_lone_NR_range=[inc["DSNB central (<E> 9/11/13 MeV, 30 cm-2 s-1)"]["N_lone_225_271_central"], inc["DSNB SK-limit-saturating very hard (12/15/25)"]["N_lone_225_271_central"]],
                     SN10kpc_coherent_window=[c10["N_200_270"], c10h["N_200_270"]], SN10kpc_lone_NR=[inc["Galactic SN 10 kpc (<E> 12/15/18)"]["N_lone_225_271_central"], inc["Galactic SN 10 kpc hard (<E> 15/18/25)"]["N_lone_225_271_central"]],
                     SN10kpc_ROI_recoils=c10["N_ROI_eff"], icecube_interactions=N_ic, F_req_150=F_req_150, F_req_300=F_req_300,
                     E_budget_10kpc_150MeV_erg=[t for t in trans if t["Enu_MeV"] == 150][0]["E_budget_erg_10kpc"], SK_events_150=R["superK"]["150 MeV"]["N_SK_events"],
                     N_lo_150=[t for t in trans if t["Enu_MeV"] == 150][0]["N_lo_5p4_200_per_window_event"])
with open(f"{OUT}/P060_results.json", "w") as f: json.dump(R, f, indent=1, default=float)
with open(f"{OUT}/run_log.txt", "w") as f: f.write("\n".join(LOG))
print("done ->", OUT)
