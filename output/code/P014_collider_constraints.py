"""
P014 -- Collider constraints on electroweak-multiplet inelastic dark matter after the LZ event.

Run from the simulation root:  .venv/bin/python output/code/P014_collider_constraints.py

Parts
  A  one-loop radiative chargino-neutralino splitting for a doublet (Higgsino) and a triplet (wino)
     as a function of the multiplet mass (Feynman-parameter form of the Thomas-Wells / Cirelli et al. result)
  B  tree-level piece of the charged-neutral splitting from the P007 gaugino table; total Delta m+- at the
     P007 fit points; demonstration that Delta m+- is independent of the neutral splitting delta at the 0.1% level
  C  chargino partial widths (pi, e nu, mu nu), lifetimes and c*tau
  D  recalled 13 TeV Drell-Yan cross-sections (flagged) -> expected produced pairs for 140/300/3000 fb^-1
  E  disappearing-track survival probabilities P(r_decay > r_min) and a toy yield calibrated on the wino limit
  F  kinematic mass floors from lzcommon (cross-check of P002) and the constraint map
  G  decays of the excited neutral state chi2 (delta ~ 366 keV): channels, lifetimes
  H  figures

All experimental bounds are RECALLED and flagged in RECALLED below; nothing here is fetched.
"""
import sys, os, json
sys.path.insert(0, 'output/code')
from common import lzcommon as lz

import numpy as np
import pandas as pd
from scipy import integrate, optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'output/work/P014'
FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)

# ----------------------------------------------------------------------------------------------
# Recalled constants (reliability in brackets)
# ----------------------------------------------------------------------------------------------
ALPHA_MZ = 1 / 127.95        # alpha_em(m_Z), MSbar [certain to 0.1%]
S2W = 0.2312                 # sin^2 theta_W (MSbar, m_Z) [certain]
C2W = 1 - S2W
MZ, MW = 91.1876, 80.377     # GeV [certain]
GF = 1.1664e-5               # GeV^-2 [certain]
FPI = 0.1302                 # GeV, f_pi+- in the 130 MeV convention (<0|d-bar gamma^mu gamma5 u|pi> = i f_pi p^mu) [certain]
VUD = 0.974                  # cos theta_c [certain]
MPI, MMU, ME = 0.13957, 0.10566, 0.000511   # GeV [certain]
HBAR_GEV_S = 6.5821e-25      # [certain]
C_CM_S = 2.99792458e10       # [certain]
ALPHA_2 = ALPHA_MZ / S2W     # alpha_em = alpha_2 sin^2 theta_W

RECALLED = []
def rec(item, source, rel):
    RECALLED.append(dict(item=item, presumed_source=source, reliability=rel))

rec("alpha_em(m_Z)=1/127.95, sin^2 theta_W=0.2312, m_Z=91.1876, m_W=80.377 GeV, G_F=1.1664e-5 GeV^-2", "PDG", "certain")
rec("f_pi = 130.2 MeV (130 MeV convention), |V_ud| = 0.974, m_pi+- = 139.57, m_mu = 105.66 MeV", "PDG", "certain")
rec("One-loop EW mass splitting of a heavy SU(2) multiplet: Delta M = alpha_2 M/(4 pi) {(Q^2-Q'^2) s_W^2 f(m_Z/M) + (Q-Q')(Q+Q'-2Y)[f(m_W/M)-f(m_Z/M)]}, f(r)->2 pi r for r->0",
    "Thomas & Wells PRL 81 (1998) 34; Cirelli, Fornengo, Strumia NPB 753 (2006) 178", "likely (asymptotes reproduced here: alpha m_Z/2 and alpha_2 m_W (1-c_W)/2)")
rec("Two-loop wino splitting 164-165 MeV at TeV masses", "Ibe, Matsumoto, Sato PLB 721 (2013) 252", "likely")
rec("Chargino decay widths: Gamma(chi+- -> chi0 pi+-) = kappa^2 G_F^2 f_pi^2 |V_ud|^2 Delta^3 (1-m_pi^2/Delta^2)^1/2 / pi, Gamma(chi+- -> chi0 l nu) = kappa^2 G_F^2 Delta^5/(15 pi^3) x phase space; kappa^2 = 1 (doublet, summed over chi1,chi2), 2 (triplet)",
    "Chen, Drees, Gunion PRD 55 (1997) 330; derived here from the W-exchange four-fermion operator", "likely (prefactor derived; reproduces the standard wino c tau ~ 6 cm)")

# ----------------------------------------------------------------------------------------------
# Part A: radiative splitting
# ----------------------------------------------------------------------------------------------
def ftilde(r):
    """f(r) - f(0) with f(r) = 2 int_0^1 dx (1+x) ln[x^2 + (1-x) r^2]  (Feynman gauge, on-shell heavy fermion)."""
    if r == 0:
        return 0.0
    g = lambda x: (1 + x) * np.log1p((1 - x) * r * r / (x * x))
    val, _ = integrate.quad(g, 0, 1, limit=200, points=[min(r, 0.5)])
    return 2 * val

def dm_higgsino_rad(M):
    """Doublet, Y=1/2: charged minus neutral, GeV."""
    return ALPHA_MZ * M / (4 * np.pi) * ftilde(MZ / M)

C2W_OS = (MW / MZ) ** 2      # on-shell cos^2 theta_W: the W/Z mass-difference piece is c^2 m_Z -> m_W^2/m_Z
def dm_wino_rad(M, c2w=C2W_OS):
    """Triplet, Y=0: charged minus neutral, GeV. Default: on-shell c_W in the Z term (MSbar c_W gives +6 MeV)."""
    return ALPHA_2 * M / (4 * np.pi) * (ftilde(MW / M) - c2w * ftilde(MZ / M))

log = []
def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s); log.append(s)

P("=== Part A: radiative splitting ===")
for r in [1e-3, 1e-2, 0.1, 0.3]:
    P(f"  ftilde(r)/(2 pi r) at r={r}: {ftilde(r)/(2*np.pi*r):.4f}")
asym_H = ALPHA_MZ * MZ / 2
asym_W = ALPHA_2 * MW / 2 * (1 - MW / MZ)
P(f"  asymptotic Higgsino splitting alpha m_Z/2 = {asym_H*1e3:.1f} MeV (with alpha(0)=1/137.036: {MZ/137.036/2*1e3:.1f} MeV); "
  f"wino alpha_2 m_W (1-c_W)/2 = {asym_W*1e3:.1f} MeV (on-shell c_W; MSbar c_W: {ALPHA_2*MW/2*(1-np.sqrt(C2W))*1e3:.1f} MeV)")
P(f"  wino at 1 TeV: on-shell {dm_wino_rad(1000)*1e3:.1f} MeV vs MSbar {dm_wino_rad(1000, C2W)*1e3:.1f} MeV; literature two-loop ~164 MeV")
M_grid = np.logspace(2, np.log10(5000), 60)
dmH = np.array([dm_higgsino_rad(M) for M in M_grid])
dmW = np.array([dm_wino_rad(M) for M in M_grid])
splitting_df = pd.DataFrame(dict(M_GeV=M_grid, dm_higgsino_MeV=dmH * 1e3, dm_wino_MeV=dmW * 1e3))
splitting_df.to_csv(os.path.join(OUT, 'radiative_splitting_vs_mass.csv'), index=False)
for M in [100, 200, 300, 500, 1000, 2000, 4000]:
    P(f"  M={M:5d} GeV: Delta m_rad Higgsino = {dm_higgsino_rad(M)*1e3:6.1f} MeV, wino = {dm_wino_rad(M)*1e3:6.1f} MeV")
# scheme systematic for the Higgsino: alpha(m_Z) vs alpha(0) rescales the whole splitting
P(f"  Higgsino scheme spread at 1 TeV: alpha(m_Z) -> {dm_higgsino_rad(1000)*1e3:.1f} MeV, alpha(0) -> {dm_higgsino_rad(1000)*1e3*127.95/137.036:.1f} MeV")

# ----------------------------------------------------------------------------------------------
# Part B: tree-level piece from P007 and total Delta m+- at the fit points
# ----------------------------------------------------------------------------------------------
P("\n=== Part B: tree-level piece and total Delta m+- ===")
g = pd.read_csv('output/work/P007/gaugino_masses_for_splitting.csv')
g['tree_over_delta'] = g['tree_charged_neutral_MeV'] * 1e3 / g['delta_target_keV']   # dimensionless (keV/keV)
ratio_tab = g.groupby(['hierarchy', 'tanb'])['tree_over_delta'].mean().reset_index()
P("  tree-level (m_chi+- - m_chi1)/delta by hierarchy and tan beta (P007 4x4 diagonalisation):")
for _, row in ratio_tab.iterrows():
    P(f"    {row['hierarchy']:22s} tanb={row['tanb']:4.0f}: {row['tree_over_delta']*1e3:.3f} x 1e-3")
r_min, r_max = g['tree_over_delta'].min(), g['tree_over_delta'].max()
P(f"  range over hierarchies/tan beta: ({r_min*1e3:.2f} - {r_max*1e3:.2f}) x 1e-3 -> tree piece = ({r_min*1e3:.2f}-{r_max*1e3:.2f}) x 1e-3 delta")

# P007 fit points (mu, delta(N=1)) and 90% one-event bands
fit_points = [(300, 310, 307, 311), (500, 342, 336, 351), (1000, 366, 358, 380), (2000, 375, 365, 391), (4000, 376, 366, 394)]
rec("P007 fit points: delta(N=1) = 310/342/366/375/376 keV at mu = 300/500/1000/2000/4000 GeV; 90% bands", "corpus P007", "corpus")
rows = []
for mu, d, dlo, dhi in fit_points:
    rad = dm_higgsino_rad(mu)
    tree_lo, tree_hi = r_min * d * 1e-6, r_max * d * 1e-6   # GeV
    tree_mid = 0.5 * (tree_lo + tree_hi)
    rows.append(dict(mu_GeV=mu, delta_keV=d, delta_lo_keV=dlo, delta_hi_keV=dhi, dm_rad_MeV=rad * 1e3,
                     tree_lo_MeV=tree_lo * 1e3, tree_hi_MeV=tree_hi * 1e3,
                     dm_pm_MeV=(rad + tree_mid) * 1e3, dm_pm_lo_MeV=(rad + tree_lo) * 1e3, dm_pm_hi_MeV=(rad + tree_hi) * 1e3))
fit_df = pd.DataFrame(rows)
for _, r in fit_df.iterrows():
    P(f"  mu={r.mu_GeV:5.0f} GeV, delta={r.delta_keV:.0f} keV: Delta m_rad={r.dm_rad_MeV:.1f} MeV, tree={r.tree_lo_MeV:.2f}-{r.tree_hi_MeV:.2f} MeV, total={r.dm_pm_MeV:.1f} MeV")
# delta-independence: scan delta 100-400 keV at 1 TeV
dscan = np.array([100, 200, 300, 366, 400])
dm_scan = dm_higgsino_rad(1000) * 1e3 + r_max * dscan * 1e-3
P(f"  Delta m+- (1 TeV, max tree coefficient) for delta = {list(dscan)} keV: {np.round(dm_scan, 3)} MeV; "
  f"fractional change 100->400 keV: {(dm_scan[-1]-dm_scan[0])/dm_scan[0]*100:.3f} %")

# ----------------------------------------------------------------------------------------------
# Part C: chargino widths and lifetimes
# ----------------------------------------------------------------------------------------------
def gamma_pi(dm, kappa2):
    if dm <= MPI:
        return 0.0
    return kappa2 * GF**2 * FPI**2 * VUD**2 / np.pi * dm**3 * np.sqrt(1 - MPI**2 / dm**2)

def gamma_lep(dm, kappa2, ml):
    """Heavy-fermion limit, vector current: Gamma = kappa^2 G_F^2/(15 pi^3) * Delta^5 * I(ml/Delta),
    I = 30/Delta^5 int_0^{pmax} p^2 (Delta - sqrt(p^2+ml^2))^2 dp  (I=1 for ml=0)."""
    if dm <= ml:
        return 0.0
    pmax = np.sqrt(dm**2 - ml**2)
    integrand = lambda p: p**2 * (dm - np.sqrt(p**2 + ml**2))**2
    I = 30 * integrate.quad(integrand, 0, pmax)[0] / dm**5
    return kappa2 * GF**2 / (15 * np.pi**3) * dm**5 * I

def lifetime_table(dm, kappa2):
    gp, ge, gm = gamma_pi(dm, kappa2), gamma_lep(dm, kappa2, ME), gamma_lep(dm, kappa2, MMU)
    gt = gp + ge + gm
    tau = HBAR_GEV_S / gt
    return dict(Gamma_pi_GeV=gp, Gamma_e_GeV=ge, Gamma_mu_GeV=gm, Gamma_tot_GeV=gt,
                BR_pi=gp / gt, BR_e=ge / gt, BR_mu=gm / gt, tau_s=tau, tau_ns=tau * 1e9, ctau_cm=tau * C_CM_S)

P("\n=== Part C: chargino widths and lifetimes ===")
# wino check
wino_rows = []
for M in [200, 500, 660, 1000, 2000, 2900]:
    dm = dm_wino_rad(M)
    t = lifetime_table(dm, 2.0)
    wino_rows.append(dict(model='wino', M_GeV=M, dm_MeV=dm * 1e3, **t))
    P(f"  wino  M={M:5d}: Delta m={dm*1e3:.1f} MeV, BR(pi)={t['BR_pi']:.3f}, BR(e)={t['BR_e']:.3f}, BR(mu)={t['BR_mu']:.4f}, tau={t['tau_ns']:.3f} ns, c tau={t['ctau_cm']:.2f} cm")
t165 = lifetime_table(0.165, 2.0)
P(f"  wino at Delta m = 165 MeV exactly: tau = {t165['tau_ns']:.3f} ns, c tau = {t165['ctau_cm']:.2f} cm (literature: ~0.2 ns, ~6 cm)")
higgs_rows = []
for _, r in fit_df.iterrows():
    dm = r.dm_pm_MeV * 1e-3
    t = lifetime_table(dm, 1.0)
    t_lo = lifetime_table(dm - 0.010, 1.0); t_hi = lifetime_table(dm + 0.010, 1.0)
    higgs_rows.append(dict(model='higgsino', M_GeV=r.mu_GeV, delta_keV=r.delta_keV, dm_MeV=dm * 1e3, **t,
                           ctau_cm_dm_minus10MeV=t_lo['ctau_cm'], ctau_cm_dm_plus10MeV=t_hi['ctau_cm']))
    P(f"  Higgsino mu={r.mu_GeV:5.0f}, delta={r.delta_keV:.0f} keV: Delta m={dm*1e3:.1f} MeV, BR(pi)={t['BR_pi']:.3f}, BR(e)={t['BR_e']:.3f}, BR(mu)={t['BR_mu']:.3f}, "
      f"tau={t['tau_ns']:.4f} ns, c tau={t['ctau_cm']:.3f} cm  [Delta m -+10 MeV: {t_lo['ctau_cm']:.3f}/{t_hi['ctau_cm']:.3f} cm]")
life_df = pd.DataFrame(wino_rows + higgs_rows)
life_df.to_csv(os.path.join(OUT, 'chargino_lifetimes.csv'), index=False)
# c tau curves vs mass for the figure
ctauH = np.array([lifetime_table(dm, 1.0)['ctau_cm'] for dm in dmH])
ctauW = np.array([lifetime_table(dm, 2.0)['ctau_cm'] for dm in dmW])
splitting_df['ctau_higgsino_cm'] = ctauH
splitting_df['ctau_wino_cm'] = ctauW
splitting_df.to_csv(os.path.join(OUT, 'radiative_splitting_vs_mass.csv'), index=False)

# ----------------------------------------------------------------------------------------------
# Part D: production cross-sections (recalled) and pair counts
# ----------------------------------------------------------------------------------------------
P("\n=== Part D: production ===")
# Anchor: pure-wino chi1+- chi2^0 at 13 TeV, NLO+NLL (LHC SUSY cross-section WG) -- recalled, likely (x/ 1.5)
wino_anchor_m = np.array([100, 200, 300, 400, 500, 600, 800, 1000.])
wino_anchor_pb = np.array([12., 1.8, 0.39, 0.12, 0.046, 0.020, 0.0046, 0.0013])
rec("13 TeV NLO+NLL pure-wino chi1+- chi2^0 cross-section: 12 pb (100 GeV), 1.8 (200), 0.39 (300), 0.12 (400), 0.046 (500), 0.020 (600), 0.0046 (800), 0.0013 pb (1000 GeV)",
    "LHC SUSY Cross Section Working Group tables (as reproduced in ATLAS/CMS electroweakino papers)", "likely (factor ~1.5)")
rec("Higgsino chi+-chi0 channels = 1/2 of the wino chi1+-chi2^0 cross-section at equal mass (W coupling g/2 per Majorana state vs g); chi+chi- and chi1chi2 add ~+50%",
    "SU(2) Wigner-Eckart; derived here", "likely (channel fractions uncertain)")
rec("Alternative (assignment) parametrisation: sigma(chi+chi- + chi+-chi0) ~ 20 fb (300 GeV), 1 fb (600), 0.1 fb (1 TeV)", "assignment; order-of-magnitude", "uncertain")
coef = np.polyfit(np.log(wino_anchor_m), np.log(wino_anchor_pb), 2)
def sigma_wino_c1n2_pb(m):
    return np.exp(np.polyval(coef, np.log(m)))
KAPPA_H = 0.5          # chi+-chi0 channels only (derived); total incl. C1C1, N1N2 ~ 0.8
KAPPA_H_TOT = 0.8
def sigma_higgsino_pb(m, total=False):
    return (KAPPA_H_TOT if total else KAPPA_H) * sigma_wino_c1n2_pb(m)
# assignment's low parametrisation: log-linear through the three anchors (fit)
alt_m = np.array([300, 600, 1000.]); alt_fb = np.array([20, 1, 0.1])
alt_coef = np.polyfit(np.log(alt_m), np.log(alt_fb), 1)
def sigma_alt_fb(m):
    return np.exp(np.polyval(alt_coef, np.log(m)))
P("  quadratic-in-ln m fit to the wino anchors, residuals (%):", np.round((sigma_wino_c1n2_pb(wino_anchor_m) / wino_anchor_pb - 1) * 100, 1))
P(f"  local log-slope d ln sigma/d ln m at 300/1000/2000 GeV: "
  + ', '.join(f"{np.polyval(np.polyder(coef), np.log(m)):.2f}" for m in [300, 1000, 2000]))
lumis = {'Run2_140': 140, 'Run3_300': 300, 'HL_LHC_3000': 3000}
prod_rows = []
for m in [300, 500, 660, 1000, 1100, 2000, 2900]:
    sW = sigma_wino_c1n2_pb(m) * 1e3  # fb
    sH = sigma_higgsino_pb(m) * 1e3
    sHt = sigma_higgsino_pb(m, total=True) * 1e3
    sA = sigma_alt_fb(m)
    row = dict(m_GeV=m, sigma_wino_C1N2_fb=sW, sigma_higgsino_chargedneutral_fb=sH, sigma_higgsino_total_fb=sHt, sigma_alt_fb=sA,
               extrapolated=(m > 1000))
    for k, L in lumis.items():
        row[f'N_higgsino_{k}'] = sH * L
        row[f'N_higgsino_total_{k}'] = sHt * L
        row[f'N_wino_C1N2_{k}'] = sW * L
        row[f'N_alt_{k}'] = sA * L
    prod_rows.append(row)
    P(f"  m={m:5d}: sigma wino C1N2={sW:9.3f} fb, Higgsino chi+-chi0={sH:8.3f} fb (total ~{sHt:8.3f}), alt={sA:8.3f} fb; "
      f"Higgsino pairs 140/300/3000 fb^-1 = {sH*140:8.1f}/{sH*300:8.1f}/{sH*3000:9.1f}" + ("  [extrapolated]" if m > 1000 else ""))
prod_df = pd.DataFrame(prod_rows)
prod_df.to_csv(os.path.join(OUT, 'production_and_pair_counts.csv'), index=False)

# ----------------------------------------------------------------------------------------------
# Part E: disappearing-track survival and toy yield
# ----------------------------------------------------------------------------------------------
P("\n=== Part E: disappearing-track survival ===")
rec("ATLAS pixel-tracklet disappearing-track search (136 fb^-1, 2022): tracklet needs 4 pixel hits, i.e. decay radius >~ 12 cm; excludes pure wino to ~660 GeV and Higgsino-cross-section charginos to ~210 GeV, both for tau = 0.2 ns",
    "ATLAS, EPJC 82 (2022) 606, arXiv:2201.02472", "likely (masses); the Higgsino limit being quoted at tau=0.2 ns: likely")
rec("CMS disappearing-track (Run 2): wino ~ 880 GeV at tau = 3 ns, ~ 470 GeV at tau = 0.2 ns", "CMS, arXiv:2004.05153", "uncertain")
rec("LEP2 chargino bound 92-103.5 GeV (92 GeV for Delta m ~ 0.1-3 GeV; 103.5 GeV for large Delta m)", "LEP SUSY WG combinations", "certain")
rec("Soft-lepton compressed searches exclude Higgsinos to ~ 210-250 GeV for Delta m ~ 5-30 GeV only; monojet-type limits on degenerate electroweakinos <~ 100-150 GeV",
    "ATLAS arXiv:1911.12606 and 2021 update; CMS soft-lepton searches; monojet interpretations", "likely (soft-lepton), uncertain (monojet)")
rec("Prospects: HL-LHC disappearing tracks wino ~ 0.9-1.1 TeV, Higgsino <~ 200-300 GeV only with short tracklets; FCC-hh 100 TeV 30 ab^-1 disappearing tracks: wino discovery ~3 TeV, Higgsino ~1 TeV; 10 TeV muon collider: thermal Higgsino and wino covered",
    "CERN YR 1812.07831; Fukuda-Nagata-Otono-Shirai PLB 781 (2018) 306; Mahbubani-Schwaller-Zurita JHEP 06 (2017) 119; Saito et al. EPJC 79 (2019) 469; Capdevilla et al. JHEP 06 (2021) 133; Han et al. 1805.00015", "uncertain (numbers), likely (qualitative)")
rec("Thermal masses: Higgsino ~1.1 TeV, wino ~2.7-3.0 TeV", "standard results (Cirelli et al.; Hisano et al. with Sommerfeld)", "certain/likely")

ctau_H_fit = float(life_df[(life_df.model == 'higgsino') & (life_df.M_GeV == 1000)].ctau_cm.iloc[0])
ctau_W_660 = float(life_df[(life_df.model == 'wino') & (life_df.M_GeV == 660)].ctau_cm.iloc[0])
surv_rows = []
for rmin in [12.0, 30.0]:
    for bg in [0.5, 1.0, 2.0, 3.0]:
        pH = np.exp(-rmin / (bg * ctau_H_fit)); pW = np.exp(-rmin / (bg * ctau_W_660))
        surv_rows.append(dict(r_min_cm=rmin, betagamma=bg, P_higgsino=pH, P_wino=pW, ratio_H_over_W=pH / pW))
        P(f"  r_min={rmin:4.0f} cm, beta gamma={bg:3.1f}: P(Higgsino, c tau={ctau_H_fit:.2f} cm)={pH:.2e}, P(wino, c tau={ctau_W_660:.2f} cm)={pW:.2e}, ratio={pH/pW:.1e}")
surv_df = pd.DataFrame(surv_rows)
surv_df.to_csv(os.path.join(OUT, 'tracklet_survival.csv'), index=False)
# beta gamma needed for P = 1e-3 at r_min = 12 cm with the Higgsino lifetime
bg_needed = 12.0 / (ctau_H_fit * np.log(1e3))
P(f"  beta gamma needed for P(r>12 cm) = 1e-3 with c tau = {ctau_H_fit:.2f} cm: {bg_needed:.1f}")

# Toy calibration: the ATLAS wino exclusion at 660 GeV, 136 fb^-1. Assume the 95% exclusion corresponds to ~5 signal
# events after all selections; production = C1N2 + C1C1 (~1.5 x C1N2). eps_tot = 5 / N_prod. The lifetime part at
# beta gamma = 1 is P_W; the rest (kinematic/isolation/trigger) is eps_kin = eps_tot / P_W.
N_prod_W660 = 1.5 * sigma_wino_c1n2_pb(660) * 1e3 * 136
eps_tot_W = 5.0 / N_prod_W660
P_W_bg1 = np.exp(-12 / ctau_W_660)
eps_kin = eps_tot_W / P_W_bg1
P(f"  toy calibration: wino 660 GeV produces {N_prod_W660:.0f} events in 136 fb^-1 -> eps_tot ~ {eps_tot_W:.2e}; P_W(bg=1)={P_W_bg1:.3f} -> eps_kin ~ {eps_kin:.3f}")
toy_rows = []
for m in [200, 300, 500, 1000]:
    for bg in [1.0, 2.0]:
        pH = np.exp(-12 / (bg * ctau_H_fit))
        nH = 2 * sigma_higgsino_pb(m, total=True) * 1e3 * 3000 * eps_kin * pH    # ~2 charginos per C1C1, 1 per C1N; use 2 as upper bound
        toy_rows.append(dict(m_GeV=m, betagamma=bg, N_tracklets_HL_LHC_upper=nH))
        P(f"  toy HL-LHC Higgsino tracklets (upper bound, r_min=12 cm, beta gamma={bg}): m={m}: {nH:.2e}")
pd.DataFrame(toy_rows).to_csv(os.path.join(OUT, 'toy_tracklet_yield.csv'), index=False)

# ----------------------------------------------------------------------------------------------
# Part F: kinematic floors and the constraint map
# ----------------------------------------------------------------------------------------------
P("\n=== Part F: kinematic floors (lzcommon) and constraint map ===")
v_june = lz.vmax_kms(v_e=lz.v_earth_kms(167))
floors = {}
for d in [300, 350, 366, 380]:
    floors[d] = float(lz.m_chi_min_gev(248.0, v_kms=v_june, delta_kev=d))
    P(f"  m_min(248 keV, delta={d} keV, v_max(16 June)={v_june:.1f} km/s) = {floors[d]:.0f} GeV")
P("  (P002 quotes 259 / 453 / 821 GeV at 300 / 350 / 380 keV)")

def status_higgsino(m):
    lep = 'excluded' if m < 92 else 'allowed'
    dt = f'unprobed (c tau {ctau_H_fit:.2f} cm; P(r>12 cm)={np.exp(-12/ctau_H_fit):.0e} at beta gamma=1)' if m > 92 else '-'
    soft = 'not applicable (Delta m = 0.36 GeV)'
    mono = 'unprobed' if m > 150 else 'marginal'
    return lep, dt, soft, mono
map_rows = []
for m in [300, 500, 1000, 1100, 2000, 4000]:
    lep, dt, soft, mono = status_higgsino(m)
    lz_ok = 'yes' if m >= floors[300] else 'no'
    map_rows.append(dict(model='Higgsino (doublet)', m_GeV=m, LEP=lep, disappearing_track=dt, soft_lepton=soft, monojet=mono,
                         LZ_compatible_mass=lz_ok, overall='allowed / unprobed by all colliders'))
for m in [300, 500, 660, 1000, 2000, 2900]:
    dt = 'excluded (ATLAS wino DT <660 GeV, tau 0.2 ns)' if m <= 660 else 'allowed (beyond 660 GeV)'
    map_rows.append(dict(model='wino (triplet)', m_GeV=m, LEP='allowed', disappearing_track=dt, soft_lepton='not applicable', monojet='unprobed',
                         LZ_compatible_mass='n/a: neutral triplet (Y=0) has no Z coupling, no O(100 keV) inelastic partner',
                         overall=('excluded' if m <= 660 else 'allowed / unprobed')))
map_df = pd.DataFrame(map_rows)
map_df.to_csv(os.path.join(OUT, 'constraint_map.csv'), index=False)
P(map_df[['model', 'm_GeV', 'disappearing_track', 'overall']].to_string(index=False))

# ----------------------------------------------------------------------------------------------
# Part G: excited-state decays
# ----------------------------------------------------------------------------------------------
P("\n=== Part G: chi2 decays at the LZ-favoured splitting ===")
rec("Magnetic-dipole transition decay Gamma(chi2 -> chi1 gamma) = mu_12^2 delta^3 / pi", "inelastic-DM literature (e.g. Weiner & Yavin 2012)", "likely")
rec("Heavy-Dirac-Higgsino loop magnetic moment ~ (alpha_2/2pi) e/(2 mu) x O(1)", "order-of-magnitude estimate (Schwinger-like coefficient for M >> m_W)", "uncertain")
chi2_rows = []
for mu, d, *_ in fit_points:
    dg = d * 1e-6  # GeV
    ee_open = dg > 2 * ME
    G_nu = GF**2 * dg**5 / (20 * np.pi**3)          # 3 flavours, Z-mediated, derived (1/4 of the chargino-lepton coefficient per flavour)
    mu12 = ALPHA_2 / (2 * np.pi) * np.sqrt(4 * np.pi * ALPHA_MZ) / (2 * mu)   # GeV^-1, c_loop = 1
    G_gam = mu12**2 * dg**3 / np.pi
    chi2_rows.append(dict(mu_GeV=mu, delta_keV=d, ee_channel_open=ee_open, tau_nunu_s=HBAR_GEV_S / G_nu, mu12_GeV_inv=mu12, tau_gamma_s=HBAR_GEV_S / G_gam))
    P(f"  mu={mu:5d}, delta={d} keV: e+e- open? {ee_open} (2 m_e = {2*ME*1e6:.0f} keV); tau(chi2->chi1 nu nubar) = {HBAR_GEV_S/G_nu:.2e} s; "
      f"mu_12 ~ {mu12:.1e} GeV^-1 -> tau(chi2->chi1 gamma) ~ {HBAR_GEV_S/G_gam:.1e} s (c_loop=1, uncertain)")
pd.DataFrame(chi2_rows).to_csv(os.path.join(OUT, 'chi2_decays.csv'), index=False)

# ----------------------------------------------------------------------------------------------
# Part H: figures (dataviz palette: blue = Higgsino, orange = wino; reserved red only for exclusions)
# ----------------------------------------------------------------------------------------------
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
BLUE_STEPS = ['#86b6ef', '#2a78d6', '#104281']
RED = '#d03b3b'; MUTED = '#898781'; GRID = '#e1e0d9'; INK = '#0b0b0b'; INK2 = '#52514e'; SURF = '#fcfcfb'
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10, 'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlecolor': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF})

# Figure 1: mass vs c tau
fig, ax = plt.subplots(figsize=(8, 5.6))
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(80, 6000); ax.set_ylim(0.1, 200)
ax.grid(True, which='major', color=GRID, lw=0.6); ax.set_axisbelow(True)
# LEP
ax.axvspan(80, 92, color=RED, alpha=0.18, lw=0)
ax.text(85, 120, 'LEP', color=RED, fontsize=9, rotation=90, va='top', ha='center')
# LZ kinematic floor (Higgsino only, delta >= 300 keV)
ax.axvspan(92, floors[300], color=MUTED, alpha=0.10, lw=0)
ax.text(np.sqrt(92 * floors[300]), 0.13, f'LZ needs\nm > {floors[300]:.0f} GeV\n(δ ≥ 300 keV, P002)', color=INK2, fontsize=8, ha='center', va='bottom')
# ATLAS DT exclusions (recalled) as thick segments at c tau = 6 cm
ax.plot([92, 660], [6.0, 6.0], color=RED, lw=5, alpha=0.6, solid_capstyle='butt')
ax.text(100, 5.0, 'ATLAS disappearing track (136 fb⁻¹), wino σ, τ = 0.2 ns: m < 660 GeV', color=RED, fontsize=8, va='center', ha='left')
ax.plot([92, 210], [3.6, 3.6], color=RED, lw=3, alpha=0.45, solid_capstyle='butt')
ax.text(100, 3.1, 'same search, Higgsino σ, τ = 0.2 ns: m < 210 GeV (not the natural Higgsino lifetime)', color=RED, fontsize=8, va='top', ha='left')
ax.text(5500, 0.115, 'Prospects (recalled):\nHL-LHC tracklets: wino ≲ 1 TeV, Higgsino ≲ 0.3 TeV\nFCC-hh: Higgsino ≈ 1 TeV, wino ≈ 3 TeV; 10 TeV μ collider: both',
        color=INK2, fontsize=7.5, ha='right', va='bottom')
# theory curves
ax.plot(M_grid, ctauW, color=ORANGE, lw=2, label='pure wino (triplet): Δm± radiative')
ax.plot(M_grid, ctauH, color=BLUE, lw=2, label='pure Higgsino (doublet): Δm± radiative + O(δ)')
ax.text(4600, ctauW[-1] * 1.2, 'wino', color=INK2, fontsize=9, ha='right', va='bottom')
ax.text(1400, 0.40, 'Higgsino', color=INK2, fontsize=9, ha='center', va='top')
hp = life_df[life_df.model == 'higgsino']
ax.scatter(hp.M_GeV, hp.ctau_cm, s=55, color=BLUE, edgecolor=SURF, linewidth=1.5, zorder=5,
           label='P007 fit points (δ = 310–376 keV)')
for _, r in hp.iterrows():
    ax.annotate(f'δ={r.delta_keV:.0f} keV', (r.M_GeV, r.ctau_cm), textcoords='offset points', xytext=(0, -14), ha='center', fontsize=7.5, color=INK2)
# thermal masses
for m, lab, col in [(1100, 'thermal Higgsino\n1.1 TeV', BLUE), (2900, 'thermal wino\n2.9 TeV', ORANGE)]:
    ax.axvline(m, color=col, lw=1, ls=':', alpha=0.8)
    ax.text(m * 1.03, 140, lab, color=col, fontsize=8, va='top')
# tracklet radius line
ax.axhline(12, color=MUTED, lw=1, ls='--')
ax.text(4800, 13, 'pixel-tracklet radius 12 cm', color=MUTED, fontsize=8, ha='right', va='bottom')
ax.set_xlabel('multiplet mass  m  [GeV]')
ax.set_ylabel('chargino  cτ  [cm]')
ax.set_title('Chargino lifetime versus mass: where colliders have looked', loc='left', fontsize=11)
ax.legend(loc='upper right', fontsize=8, frameon=False, bbox_to_anchor=(1.0, 0.86))
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
fig.text(0.01, 0.005, 'Exclusion bars are recalled from published searches (see details.md); curves and points are computed here.', fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(os.path.join(FIG, 'P014_fig1_mass_vs_ctau.png'), dpi=160)
plt.close(fig)

# Figure 2: expected Higgsino pairs vs mass
mgrid = np.logspace(np.log10(150), np.log10(2500), 100)
fig, ax = plt.subplots(figsize=(8, 5.2))
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(150, 2500); ax.set_ylim(0.1, 1e6)
ax.grid(True, which='major', color=GRID, lw=0.6); ax.set_axisbelow(True)
sH_fb = sigma_higgsino_pb(mgrid) * 1e3
for (k, L), col in zip(lumis.items(), BLUE_STEPS):
    lab = {'Run2_140': 'Run 2, 140 fb⁻¹', 'Run3_300': 'Run 3, 300 fb⁻¹', 'HL_LHC_3000': 'HL-LHC, 3000 fb⁻¹'}[k]
    ax.plot(mgrid, sH_fb * L, color=col, lw=2, label=lab)
    ax.text(2400, sH_fb[-1] * L * 1.1, lab.split(',')[0], color=INK2, fontsize=8, ha='right', va='bottom')
ax.fill_between(mgrid, sH_fb * 3000 / 2, sH_fb * 3000 * 2, color=BLUE_STEPS[2], alpha=0.12, lw=0, label='×/÷ 2 cross-section uncertainty (HL-LHC)')
ax.plot(mgrid, sigma_alt_fb(mgrid) * 3000, color=MUTED, lw=1.2, ls='--', label='HL-LHC with the lower (assignment) parametrisation')
ax.axvspan(150, floors[300], color=MUTED, alpha=0.10, lw=0)
ax.text(np.sqrt(150 * floors[300]), 6e2, f'below the LZ\nmass floor\n({floors[300]:.0f} GeV, P002)', color=INK2, fontsize=8, ha='center', va='center')
ax.axvline(1100, color=BLUE, lw=1, ls=':', alpha=0.8); ax.text(1130, 3e5, 'thermal 1.1 TeV', color=BLUE, fontsize=8)
ax.axhline(1, color=MUTED, lw=1, ls='--'); ax.text(160, 1.25, '1 pair', color=MUTED, fontsize=8)
ax.text(1100, 0.5, 'extrapolated beyond 1 TeV', color=MUTED, fontsize=7.5, style='italic')
for m in [300, 500, 1000]:
    n = sigma_higgsino_pb(m) * 1e3 * 3000
    ax.scatter([m], [n], s=40, color=BLUE_STEPS[2], edgecolor=SURF, linewidth=1.2, zorder=5)
    ex = int(np.floor(np.log10(n)))
    ax.annotate(f'{n/10**ex:.1f}×10$^{ex}$', (m, n), textcoords='offset points', xytext=(6, 4), fontsize=8, color=INK2)
ax.set_xlabel('Higgsino mass  μ  [GeV]')
ax.set_ylabel('produced χ±χ⁰ pairs (13–14 TeV)')
ax.set_title('Higgsino pairs produced at the LHC versus mass (recalled cross-sections)', loc='left', fontsize=11)
ax.legend(loc='upper right', fontsize=8, frameon=False)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'P014_fig2_pairs_vs_mass.png'), dpi=160)
plt.close(fig)

# ----------------------------------------------------------------------------------------------
# Summary JSON
# ----------------------------------------------------------------------------------------------
summary = dict(
    asymptotic_splitting_MeV=dict(higgsino=asym_H * 1e3, wino=asym_W * 1e3),
    radiative_splitting_MeV={str(M): dict(higgsino=dm_higgsino_rad(M) * 1e3, wino=dm_wino_rad(M) * 1e3) for M in [100, 300, 500, 1000, 2000, 4000]},
    tree_over_delta_range=[r_min, r_max],
    fit_points=fit_df.to_dict(orient='records'),
    lifetimes=life_df.to_dict(orient='records'),
    dm_scan_1TeV=dict(delta_keV=dscan.tolist(), dm_pm_MeV=dm_scan.tolist()),
    wino_165MeV=t165,
    production=prod_df.to_dict(orient='records'),
    survival=surv_df.to_dict(orient='records'),
    betagamma_needed_for_P_1e3_at_r12cm=bg_needed,
    toy_calibration=dict(N_prod_wino660_136fb=N_prod_W660, eps_tot=eps_tot_W, eps_kin=eps_kin),
    kinematic_floors_GeV=floors,
    chi2=chi2_rows,
    recalled=RECALLED,
)
with open(os.path.join(OUT, 'P014_results.json'), 'w') as f:
    json.dump(summary, f, indent=1, default=float)
with open(os.path.join(OUT, 'recalled_knowledge.json'), 'w') as f:
    json.dump(RECALLED, f, indent=1)
with open(os.path.join(OUT, 'run_log.txt'), 'w') as f:
    f.write('\n'.join(log) + '\n')
P(f"\nwrote {OUT}/P014_results.json, CSVs, run_log.txt and figures in {FIG}")
