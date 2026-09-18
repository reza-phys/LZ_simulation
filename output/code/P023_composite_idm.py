"""
P023 -- Composite dark-baryon dark matter with a ~300 keV hyperfine splitting: a UV origin for the LZ event?

Run from the simulation root:   .venv/bin/python output/code/P023_composite_idm.py [--recompute]

Parts
  1  Hyperfine-splitting scaling relations (Coulombic heavy-heavy, HQET heavy-light, light-constituent baryon),
     calibrated/validated on recalled QCD and atomic data, solved for delta = 250-400 keV at M = 0.3-4 TeV.
  2  Inelastic (spin-flip) transition cross-section: photon-mediated magnetic-dipole transition (MiDM-like) and its
     kinetically-mixed dark-photon variant, computed with WimPyDD (P012 NREFT coefficients, delta > 0), plus a
     Z-axial spin-flip (composite of EW-doublet constituents) as O4 inelastic; mu_tr(N=1) at delta = 300, 366 keV.
  3  Excited-state lifetime chi2 -> chi1 gamma (Gamma = mu^2 delta^3/pi), single-site-NR requirement, cosmology.
  4  Self-interactions, asymmetric relic abundance, symmetric annihilation.
  5  Three benchmark scenarios (table).
All recalled inputs are collected in RECALLED and written to output/work/P023/recalled_inputs.json.
"""
from __future__ import annotations
import sys, os, math, json, time, argparse
import numpy as np
import pandas as pd
from scipy import special, optimize

sys.path.insert(0, 'output/code')
from common import lzcommon as lz

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'output/work/P023'; FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument('--recompute', action='store_true'); ARGS = ap.parse_args()
T0 = time.time()
log_lines = []
def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log_lines.append(s)

# ----------------------------------------------------------------------------------------------
# 0. Constants.  Every recalled number is listed with a reliability flag.
# ----------------------------------------------------------------------------------------------
RECALLED = []
def rec(item, value, source, reliability):
    RECALLED.append(dict(item=item, value=value, presumed_source=source, reliability=reliability)); return value

ALPHA = rec('fine-structure constant alpha = 1/137.036', 1 / 137.036, 'PDG', 'certain')
E_CH = math.sqrt(4 * math.pi * ALPHA)                      # Heaviside-Lorentz e = 0.3028
G_F = rec('G_F = 1.1664e-5 GeV^-2', 1.1664e-5, 'PDG', 'certain')
M_P = lz.M_NUCLEON_GEV                                     # 0.938272 GeV (lzcommon)
M_E = rec('electron mass 0.510999 MeV', 0.510999e-3, 'PDG', 'certain')
G_P, G_N = rec('nucleon g-factors g_p = 5.5857, g_n = -3.8261', (5.5857, -3.8261), 'PDG', 'certain')
HBAR_GEV_S = rec('hbar = 6.5821e-25 GeV s', 6.5821e-25, 'PDG', 'certain')
HBARC = lz.HBARC_GEV_FM; GEV2_CM2 = lz.GEV_TO_CM2
MU_N_GEV = E_CH / (2 * M_P)                                # nuclear magneton, natural units (GeV^-1)
MU_N_ECM = HBARC / (2 * M_P) * 1e-13                       # 1.0515e-14 e cm
C_CMS = 2.99792458e10
T_UNIVERSE_S = rec('age of the Universe 13.8 Gyr', 13.8e9 * 3.15576e7, 'Planck 2018', 'certain')
# QCD hadron masses [MeV] (PDG; certain to the quoted precision)
HAD = rec('hadron masses [MeV]: N 938.9, Delta 1232, Lambda 1115.7, Sigma0 1192.6, Sigma(1385) 1384.6, B 5279.5, B* 5324.7, '
          'D 1867.2, D* 2008.6, eta_c 2984.1, J/psi 3096.9, eta_b 9398.7, Upsilon 9460.3, Sigma_c 2453.5, Sigma_c* 2518.1, '
          'Sigma_b 5813, Sigma_b* 5833',
          dict(N=938.9, Delta=1232.0, Lam=1115.7, Sig0=1192.6, SigStar=1384.6, B=5279.5, Bstar=5324.7, D=1867.2, Dstar=2008.6,
               etac=2984.1, Jpsi=3096.9, etab=9398.7, Ups=9460.3, Sigc=2453.5, SigcStar=2518.1, Sigb=5813.0, SigbStar=5833.0),
          'PDG', 'certain (+-1-3 MeV for averages)')
M_C, M_B = rec('MSbar quark masses m_c = 1.27 GeV, m_b = 4.18 GeV', (1.27, 4.18), 'PDG', 'certain')
LAM_QCD = rec('Lambda_QCD^(3), MSbar ~ 0.33 GeV', 0.33, 'PDG QCD review', 'likely (+-0.03)')
ALPHA_S_MC, ALPHA_S_MB = rec('alpha_s(m_c) ~ 0.38, alpha_s(m_b) ~ 0.22', (0.38, 0.22), 'PDG running', 'likely')
DM_S = rec('m_s - m_ud ~ 93 MeV (MSbar 2 GeV)', 0.093, 'PDG', 'likely')
H_HFS_MHZ = rec('hydrogen 21 cm hyperfine splitting 1420.4 MHz; A_21 = 2.87e-15 s^-1', (1420.4, 2.87e-15), 'standard atomic physics', 'certain')
PS_HFS_GHZ = rec('positronium ground-state hfs 203.4 GHz; leading order (7/12) alpha^4 m_e', 203.4, 'standard QED', 'certain')
EV_HZ = 2.417989e14                                        # 1 eV in Hz (h), certain
OMEGA_DM_H2 = rec('Omega_DM h^2 = 0.120', 0.120, 'Planck 2018', 'certain')
RHO_C_H2 = rec('rho_c/h^2 = 1.0537e-5 GeV cm^-3', 1.0537e-5, 'PDG', 'certain')
S0 = rec('entropy density today s_0 = 2891 cm^-3', 2891.0, 'PDG', 'certain')
ETA_B = rec('baryon asymmetry n_B/s = 8.7e-11', 8.7e-11, 'Planck/PDG', 'certain')
SV_THERMAL = rec('thermal relic <sigma v> = 2.2e-26 cm^3/s (m >> 10 GeV)', 2.2e-26, 'Steigman-Dasgupta-Beacom 2012', 'certain')
DQ = rec('nucleon axial charges Delta u = 0.842, Delta d = -0.427, Delta s = -0.085', (0.842, -0.427, -0.085), 'HERMES/COMPASS-era values', 'likely')
LXE_ATT_300 = rec('LXe attenuation length of a 300 keV gamma ~ 2 cm (+-50%)', 2.0, 'NIST XCOM-type tables (mu/rho ~ 0.15-0.2 cm^2/g)', 'uncertain')
GAMMA_M1 = rec('MiDM de-excitation width Gamma(chi2 -> chi1 gamma) = mu_tr^2 delta^3 / pi', 'formula', 'Chang, Weiner, Yavin 2010 (MiDM)', 'likely')
ANN_COUL = rec('<sigma v>(Q Qbar -> A_D A_D) ~ pi alpha_D^2 / m_Q^2', 'formula', 'secluded-DM literature (also P011)', 'likely')
GLUEBALL = rec('lightest glueball 0++ mass ~ 1.7 GeV ~ 5-6 Lambda_QCD^(3)', 5.5, 'lattice QCD (Morningstar-Peardon) + PDG', 'likely')
V_CHI = rec('post-scatter chi2 speed ~ 750 km/s (v_min(248 keV, delta=300) ~ 700 km/s; the recoil takes <1% of the momentum)', 750.0, 'kinematics (lzcommon)', 'certain (order)')
L_TPC = rec('LZ TPC: 1.46 m drift x 1.46 m diameter; mean remaining chord ~1 m', (1.46, 1.0), 'LZ detector papers', 'likely')
# NREFT photon-dipole coefficients: identical to P012 (recalled/likely there, validated against its Fig. 1 shape)
rec('photon-mediated dipole NREFT: c1 = e mu Q/(2 m_chi), c5 = 2 e mu m_N Q/q^2, c4 = e mu g_N/m_N, c6 = -e mu g_N m_N/q^2 (Anand normalisation)',
    'formula', 'Fitzpatrick et al. 2012 / Anand et al. 2014 via P012', 'likely')
rec('Fermi contact hfs: Delta E = (8/3) alpha^4 mu_red^3/(m1 m2) (g1 g2/4); quarkonium colour factor C_F = 4/3 (SU(3)), 3/4 (SU(2))',
    'formula', 'standard QED / QCD textbooks', 'certain')
rec('HQET: heavy-light hyperfine splitting m_H*^2 - m_H^2 = const (Lambda^2), i.e. delta ~ Lambda^2/m_Q', 'formula', 'HQET (Neubert review)', 'certain')
rec('one-loop running Lambda = mu exp(-2 pi/(b0 alpha(mu))), b0 = (11 N - 2 n_f)/3', 'formula', 'standard', 'certain')

EXPOSURE = lz.LZ['exposure_tyr']
MASSES_M = [300.0, 1000.0, 4000.0]
DELTAS_KEV = [250.0, 300.0, 366.0, 400.0]

# ----------------------------------------------------------------------------------------------
# 1. Hyperfine splittings
# ----------------------------------------------------------------------------------------------
log('=== Part 1: hyperfine scaling relations ===')
def hfs_contact(alpha_eff, m1, m2, g1=2.0, g2=2.0):
    """Fermi-contact (Coulombic 1S) hyperfine splitting for two point particles [GeV]."""
    mu = m1 * m2 / (m1 + m2)
    return (8.0 / 3.0) * alpha_eff ** 4 * mu ** 3 / (m1 * m2) * (g1 * g2 / 4.0)

# validation: hydrogen and positronium
dE_H = hfs_contact(ALPHA, M_E, M_P, 2.0, G_P)
H_MHz = dE_H * 1e9 * EV_HZ / 1e6
dE_Ps = (7.0 / 12.0) * ALPHA ** 4 * M_E                    # spin-spin (1/3) + annihilation (1/4)
Ps_GHz = dE_Ps * 1e9 * EV_HZ / 1e9
log(f'  hydrogen hfs from contact formula: {H_MHz:.1f} MHz (measured 1420.4; leading order 1418.8)')
log(f'  positronium (7/12) alpha^4 m_e: {Ps_GHz:.1f} GHz (measured 203.4)')
# QCD checks of the scalings (ratios are coupling-normalisation free)
r_quarkonium_obs = (HAD['Ups'] - HAD['etab']) / (HAD['Jpsi'] - HAD['etac'])
r_quarkonium_pred = (ALPHA_S_MB / ALPHA_S_MC) ** 4 * (M_B / M_C)
r_meson_obs = (HAD['Bstar'] - HAD['B']) / (HAD['Dstar'] - HAD['D']);  r_meson_pred = M_C / M_B
r_baryon_obs = (HAD['SigbStar'] - HAD['Sigb']) / (HAD['SigcStar'] - HAD['Sigc'])
log(f'  Coulombic alpha^4 m scaling: (Ups-eta_b)/(J/psi-eta_c) obs {r_quarkonium_obs:.2f} vs (a_s(m_b)/a_s(m_c))^4 m_b/m_c = {r_quarkonium_pred:.2f}')
log(f'  HQET 1/m_Q scaling: (B*-B)/(D*-D) obs {r_meson_obs:.3f}, (Sig_b*-Sig_b)/(Sig_c*-Sig_c) obs {r_baryon_obs:.3f} vs m_c/m_b = {r_meson_pred:.3f}')
# absolute Coulombic prediction for QCD quarkonia (shows they are NOT Coulombic)
kappa_su3 = (1.0 / 3.0) * (4.0 / 3.0) ** 4                 # 256/243 for colour-singlet Q Qbar with alpha_eff = C_F alpha_s
log(f'  Coulombic SU(3) formula for J/psi-eta_c: {kappa_su3 * ALPHA_S_MC ** 4 * M_C * 1e3:.1f} MeV (obs {HAD["Jpsi"] - HAD["etac"]:.0f}); '
    f'Upsilon-eta_b: {kappa_su3 * ALPHA_S_MB ** 4 * M_B * 1e3:.1f} MeV (obs {HAD["Ups"] - HAD["etab"]:.0f}) -> QCD quarkonia are 4-6x more compact than Coulombic')

# HQET calibration constant K = delta_hf * m_Q  (GeV^2), meson and baryon
K_B = (HAD['Bstar'] - HAD['B']) * 1e-3 * M_B; K_D = (HAD['Dstar'] - HAD['D']) * 1e-3 * M_C
K_MESON = 0.5 * (K_B + K_D)
K_SB = (HAD['SigbStar'] - HAD['Sigb']) * 1e-3 * M_B; K_SC = (HAD['SigcStar'] - HAD['Sigc']) * 1e-3 * M_C
K_BARYON = 0.5 * (K_SB + K_SC)
log(f'  HQET constants K = delta*m_Q: B {K_B:.3f}, D {K_D:.3f} (mean {K_MESON:.3f} GeV^2 = {K_MESON/LAM_QCD**2:.2f} Lambda^2); '
    f'Sigma_b {K_SB:.3f}, Sigma_c {K_SC:.3f} (mean {K_BARYON:.3f} GeV^2 = {K_BARYON/LAM_QCD**2:.2f} Lambda^2)')

rows = []
for M in MASSES_M:
    for d_keV in DELTAS_KEV:
        d = d_keV * 1e-6
        # (A) Coulombic heavy-heavy, equal constituents m_Q = M/2, spin-spin only (kappa = 1/3), alpha_eff
        mQ = M / 2.0
        a_eff = (3.0 * d / mQ) ** 0.25
        E_B = a_eff ** 2 * mQ / 4.0                          # 1S binding, mu_red = m_Q/2
        a0_fm = 2.0 / (a_eff * mQ) * HBARC                   # Bohr radius = 1/(mu alpha)
        p_B = a_eff * mQ / 2.0                               # Bohr momentum
        E_2S1S = 0.75 * E_B
        # if the binding force is a dark SU(3), alpha_D = alpha_eff/C_F and the confinement scale follows from one-loop running (n_f=0 below m_Q)
        a_D3 = a_eff / (4.0 / 3.0); Lam_su3 = p_B * math.exp(-2 * math.pi / (11.0 * a_D3))
        a_D2 = a_eff / (3.0 / 4.0); Lam_su2 = p_B * math.exp(-2 * math.pi / ((22.0 / 3.0) * a_D2))
        # (B) HQET heavy-light: delta = K (Lambda_D/Lambda_QCD)^2 / m_Q, m_Q ~ M - 3.3 Lambda_D (B: m_B - m_b = 1.1 GeV = 3.3 Lambda)
        def solve_lam(K):
            f = lambda L: K * (L / LAM_QCD) ** 2 / (M - 3.3 * L) - d
            return optimize.brentq(f, 1e-3, M / 4.0)
        Lam_meson = solve_lam(K_MESON); Lam_baryon = solve_lam(K_BARYON)
        d_at_LamQCD_meson = K_MESON / (M - 3.3 * LAM_QCD); d_at_LamQCD_baryon = K_BARYON / (M - 3.3 * LAM_QCD)
        # (C) light-constituent baryon: delta/M in QCD (Delta-N) vs required; Sigma0-Lambda-type explicit breaking
        ratio_qcd = (HAD['Delta'] - HAD['N']) / HAD['N']
        supp = (d / M) / ratio_qcd
        dm_q = d / ((HAD['Sig0'] - HAD['Lam']) * 1e-3 / DM_S)   # explicit flavour-breaking mass difference needed
        rows.append(dict(M_GeV=M, delta_keV=d_keV, coul_mQ_GeV=mQ, coul_alpha_eff=a_eff, coul_E_B_MeV=E_B * 1e3, coul_a0_fm=a0_fm,
                         coul_pBohr_GeV=p_B, coul_2S1S_MeV=E_2S1S * 1e3, coul_alphaD_SU3=a_D3, coul_LambdaD_SU3_eV=Lam_su3 * 1e9,
                         coul_alphaD_SU2=a_D2, coul_LambdaD_SU2_eV=Lam_su2 * 1e9,
                         hl_LambdaD_meson_GeV=Lam_meson, hl_LambdaD_baryon_GeV=Lam_baryon,
                         hl_delta_at_LambdaQCD_meson_keV=d_at_LamQCD_meson * 1e6, hl_delta_at_LambdaQCD_baryon_keV=d_at_LamQCD_baryon * 1e6,
                         light_delta_over_M=d / M, light_suppression_vs_DeltaN=supp, light_dm_q_MeV=dm_q * 1e3))
hf = pd.DataFrame(rows); hf.to_csv(os.path.join(OUT, 'P023_hyperfine_solutions.csv'), index=False, float_format='%.5g')
r1 = hf[(hf.M_GeV == 1000) & (hf.delta_keV == 300)].iloc[0]
log(f'  M=1 TeV, delta=300 keV: Coulombic alpha_eff = {r1.coul_alpha_eff:.4f} (m_Q = 500 GeV), E_B = {r1.coul_E_B_MeV:.0f} MeV, a0 = {r1.coul_a0_fm:.4f} fm, '
    f'2S-1S = {r1.coul_2S1S_MeV:.0f} MeV; SU(3)_D alpha_D = {r1.coul_alphaD_SU3:.4f} -> Lambda_D = {r1.coul_LambdaD_SU3_eV:.2g} eV; '
    f'SU(2)_D alpha_D = {r1.coul_alphaD_SU2:.4f} -> Lambda_D = {r1.coul_LambdaD_SU2_eV:.2g} eV')
log(f'  heavy-light: Lambda_D = {r1.hl_LambdaD_meson_GeV:.3f} GeV (meson-like), {r1.hl_LambdaD_baryon_GeV:.3f} GeV (Sigma_Q-like); '
    f'at Lambda_D = Lambda_QCD: delta = {r1.hl_delta_at_LambdaQCD_meson_keV:.0f} / {r1.hl_delta_at_LambdaQCD_baryon_keV:.0f} keV')
log(f'  light baryon: delta/M = {r1.light_delta_over_M:.1e} vs (Delta-N)/N = 0.31: suppression {r1.light_suppression_vs_DeltaN:.1e}; '
    f'Sigma0-Lambda-type breaking needs dm_q = {r1.light_dm_q_MeV:.2f} MeV')
M_LamQCD_meson = K_MESON / 300e-6; M_LamQCD_baryon = K_BARYON / 300e-6
log(f'  mass at which Lambda_D = Lambda_QCD gives delta = 300 keV: {M_LamQCD_meson:.0f} GeV (meson), {M_LamQCD_baryon:.0f} GeV (baryon)')

# ----------------------------------------------------------------------------------------------
# 2. Transition cross-sections with WimPyDD
# ----------------------------------------------------------------------------------------------
log('=== Part 2: transition cross-sections (WimPyDD) ===')
WD = lz.wd()
E = np.arange(1.0, 331.0, 2.0)
def efficiency(E, sig_lo=3.4, sig_hi=8.0, eff0=lz.LZ['eff_plateau']):      # P003/P012 erf model
    lo, hi = lz.LZ['E_50pct_low_keV'], lz.LZ['E_50pct_high_keV']
    return eff0 * 0.5 * (1 + special.erf((E - lo) / (math.sqrt(2) * sig_lo))) * 0.5 * (1 - special.erf((E - hi) / (math.sqrt(2) * sig_hi)))
def counts(E, r):
    eff = efficiency(E); w = eff * r
    N_roi = np.trapezoid(w, E) * EXPOSURE
    N_hi = np.trapezoid(np.where((E >= 200) & (E <= 270), w, 0.0), E) * EXPOSURE
    N_lo = np.trapezoid(np.where((E >= 5.4) & (E <= 55), w, 0.0), E) * EXPOSURE
    return N_roi, N_hi, N_lo
VGRID = np.linspace(0.0, lz.VESC_KMS + 300.0, 1200)
days12 = np.arange(15, 366, 30.4)
HALOS = {'annual': (VGRID, np.mean([lz.wd_halo(day_of_year=d, vmin=VGRID)[1] for d in days12], axis=0)),
         'june16': lz.wd_halo(day_of_year=167, vmin=VGRID)}

MU_REF = MU_N_GEV                                     # compute at mu_tr = 1 nuclear magneton
def h_photon(name, mA_GeV=None, mchi_fixed=1000.0):
    """P012 photon-dipole NREFT (WimPyDD isospin (c_p+c_n, c_p-c_n)); with mA_GeV: kinetically-mixed massive A',
    every coefficient x q^2/(q^2 + mA^2) (the A' propagator relative to the photon; m_chi fixed inside the closure)."""
    mu = MU_REF
    if mA_GeV is None:
        return WD.eft_hamiltonian(name, {
            1: lambda mchi, mu=mu: [E_CH * mu / (2 * mchi), E_CH * mu / (2 * mchi)],
            5: lambda q, mu=mu: [2 * E_CH * mu * M_P / q ** 2, 2 * E_CH * mu * M_P / q ** 2],
            4: lambda mu=mu: [E_CH * mu * (G_P + G_N) / M_P, E_CH * mu * (G_P - G_N) / M_P],
            6: lambda q, mu=mu: [-E_CH * mu * (G_P + G_N) * M_P / q ** 2, -E_CH * mu * (G_P - G_N) * M_P / q ** 2]})
    P = lambda q, mA=mA_GeV: q ** 2 / (q ** 2 + mA ** 2)
    return WD.eft_hamiltonian(name, {
        1: lambda q, mu=mu, m=mchi_fixed: [E_CH * mu / (2 * m) * P(q)] * 2,
        5: lambda q, mu=mu: [2 * E_CH * mu * M_P / q ** 2 * P(q)] * 2,
        4: lambda q, mu=mu: [E_CH * mu * (G_P + G_N) / M_P * P(q), E_CH * mu * (G_P - G_N) / M_P * P(q)],
        6: lambda q, mu=mu: [-E_CH * mu * (G_P + G_N) * M_P / q ** 2 * P(q), -E_CH * mu * (G_P - G_N) * M_P / q ** 2 * P(q)]})
# Z-axial spin-flip: c4^N = 4 sqrt2 G_F g_A^chi a_N, g_A^chi = T3 = 1/2 (composite matrix element g_tr = 1)
A_P = 0.5 * DQ[0] - 0.5 * DQ[1] - 0.5 * DQ[2]; A_N = 0.5 * DQ[1] - 0.5 * DQ[0] - 0.5 * DQ[2]
G_A_CHI = 0.5
c4p = 4 * math.sqrt(2) * G_F * G_A_CHI * A_P; c4n = 4 * math.sqrt(2) * G_F * G_A_CHI * A_N
h_Z = lz.wd_hamiltonian('P023_Z_axial_O4', {4: (c4p + c4n, c4p - c4n)})
log(f'  Z-axial: a_p = {A_P:.3f}, a_n = {A_N:.3f}; c4_p = {c4p:.3e}, c4_n = {c4n:.3e} GeV^-2 (WimPyDD c0 = {c4p+c4n:.2e}, c1 = {c4p-c4n:.2e})')

cache = os.path.join(OUT, 'P023_spectra_cache.npz')
spectra = {}
if os.path.exists(cache) and not ARGS.recompute:
    z = np.load(cache, allow_pickle=True); spectra = z['spectra'].item(); log(f'  loaded cached spectra ({len(spectra)} configs)')
def get_spec(key, ham, m, delta, halo):
    if key not in spectra:
        spectra[key] = lz.wd_rate(ham, m, E, halo=HALOS[halo], delta_kev=delta)
        np.savez(cache, spectra=np.array(spectra, dtype=object))
        log(f'    computed {key} ({time.time()-T0:.0f} s)')
    return spectra[key]

h_ph = h_photon('P023_photon_dipole')
DSCAN = list(np.arange(200.0, 391.0, 10.0)) + [366.0]
res = []
for d in DSCAN:
    r = get_spec(f'ph|1000|{d}|annual', h_ph, 1000.0, d, 'annual'); N_roi, N_hi, N_lo = counts(E, r)
    res.append(dict(model='photon_dipole', m_GeV=1000.0, delta_keV=d, halo='annual', N_roi_unit=N_roi, N_hi_unit=N_hi, N_lo_unit=N_lo))
    rz = get_spec(f'Z|1000|{d}|annual', h_Z, 1000.0, d, 'annual'); Nz = counts(E, rz)
    res.append(dict(model='Z_axial_O4', m_GeV=1000.0, delta_keV=d, halo='annual', N_roi_unit=Nz[0], N_hi_unit=Nz[1], N_lo_unit=Nz[2]))
for d in [300.0, 366.0]:
    r = get_spec(f'ph|1000|{d}|june16', h_ph, 1000.0, d, 'june16'); N = counts(E, r)
    res.append(dict(model='photon_dipole', m_GeV=1000.0, delta_keV=d, halo='june16', N_roi_unit=N[0], N_hi_unit=N[1], N_lo_unit=N[2]))
    rz = get_spec(f'Z|1000|{d}|june16', h_Z, 1000.0, d, 'june16'); Nz = counts(E, rz)
    res.append(dict(model='Z_axial_O4', m_GeV=1000.0, delta_keV=d, halo='june16', N_roi_unit=Nz[0], N_hi_unit=Nz[1], N_lo_unit=Nz[2]))
for m in [400.0, 4000.0]:
    r = get_spec(f'ph|{m}|300.0|annual', h_ph, m, 300.0, 'annual'); N = counts(E, r)
    res.append(dict(model='photon_dipole', m_GeV=m, delta_keV=300.0, halo='annual', N_roi_unit=N[0], N_hi_unit=N[1], N_lo_unit=N[2]))
r0 = get_spec('ph|1000|0.0|annual', h_ph, 1000.0, 0.0, 'annual'); N0 = counts(E, r0)
res.append(dict(model='photon_dipole', m_GeV=1000.0, delta_keV=0.0, halo='annual', N_roi_unit=N0[0], N_hi_unit=N0[1], N_lo_unit=N0[2]))
# massive dark photon (kinetic mixing) variants at delta = 300 keV
for mA in [0.3, 1.0, 3.0]:
    hA = h_photon(f'P023_Aprime_{mA}', mA_GeV=mA)
    r = get_spec(f'Ap{mA}|1000|300.0|annual', hA, 1000.0, 300.0, 'annual'); N = counts(E, r)
    res.append(dict(model=f'Aprime_mA_{mA}GeV', m_GeV=1000.0, delta_keV=300.0, halo='annual', N_roi_unit=N[0], N_hi_unit=N[1], N_lo_unit=N[2]))
df = pd.DataFrame(res)
# required moment for N_roi = 1 (and N_hi = 1) at mu_tr = 1 mu_N reference; rates scale as mu^2
ph = df.model == 'photon_dipole'
df.loc[ph, 'mu_N1_roi_muN'] = np.sqrt(1.0 / df.loc[ph, 'N_roi_unit'].clip(lower=1e-300))
df.loc[ph, 'mu_N1_hi_muN'] = np.sqrt(1.0 / df.loc[ph, 'N_hi_unit'].clip(lower=1e-300))
df.loc[ph, 'mu_N1_roi_ecm'] = df.loc[ph, 'mu_N1_roi_muN'] * MU_N_ECM
df.loc[ph, 'mu_N1_roi_in_e_over_2mchi'] = df.loc[ph, 'mu_N1_roi_muN'] * df.loc[ph, 'm_GeV'] / M_P
df.loc[ph, 'mu_N1_roi_GeVinv'] = df.loc[ph, 'mu_N1_roi_muN'] * MU_N_GEV
df.loc[ph, 'N_lo_per_hi'] = df.loc[ph, 'N_lo_unit'] / df.loc[ph, 'N_hi_unit'].clip(lower=1e-300)
zz = df.model == 'Z_axial_O4'
df.loc[zz, 'gA_chi_for_N1'] = G_A_CHI * np.sqrt(1.0 / df.loc[zz, 'N_roi_unit'].clip(lower=1e-300))
df.to_csv(os.path.join(OUT, 'P023_transition_rates.csv'), index=False, float_format='%.5g')

def row(model, d, halo='annual', m=1000.0):
    return df[(df.model == model) & (df.delta_keV == d) & (df.halo == halo) & (df.m_GeV == m)].iloc[0]
p300, p366 = row('photon_dipole', 300.0), row('photon_dipole', 366.0)
log(f'  photon dipole 1 TeV annual: N(1 mu_N) ROI = {p300.N_roi_unit:.4g} (d=300), {p366.N_roi_unit:.4g} (d=366); elastic {N0[0]:.4g}, elastic N_lo/hi = {N0[2]/N0[1]:.0f}')
log(f'  mu_tr(N_ROI=1): d=300: {p300.mu_N1_roi_muN:.3e} mu_N = {p300.mu_N1_roi_ecm:.3e} e cm = {p300.mu_N1_roi_in_e_over_2mchi:.3f} e/(2 m_chi); '
    f'd=366: {p366.mu_N1_roi_muN:.3e} mu_N = {p366.mu_N1_roi_ecm:.3e} e cm = {p366.mu_N1_roi_in_e_over_2mchi:.3f} e/(2 m_chi)')
log(f'  mu_tr(N_hi=1): d=300 {p300.mu_N1_hi_muN:.3e} mu_N; d=366 {p366.mu_N1_hi_muN:.3e} mu_N; N_lo per hi event: {p300.N_lo_per_hi:.2g} (d=300)')
pj300, pj366 = row('photon_dipole', 300.0, 'june16'), row('photon_dipole', 366.0, 'june16')
log(f'  June-16 halo: N ratio june/annual = {pj300.N_roi_unit/p300.N_roi_unit:.2f} (d=300), {pj366.N_roi_unit/p366.N_roi_unit:.2f} (d=366)')
for m in [400.0, 4000.0]:
    pm = row('photon_dipole', 300.0, 'annual', m); log(f'  mass {m:.0f} GeV, d=300: mu(N=1) = {pm.mu_N1_roi_muN:.3e} mu_N = {pm.mu_N1_roi_in_e_over_2mchi:.3f} e/(2m_chi)')
for mA in [0.3, 1.0, 3.0]:
    rA = df[df.model == f'Aprime_mA_{mA}GeV'].iloc[0]; log(f'  massive A\' m_A = {mA} GeV: rate suppression vs photon S = {rA.N_roi_unit/p300.N_roi_unit:.3e} (d=300)')
z300, z366 = row('Z_axial_O4', 300.0), row('Z_axial_O4', 366.0)
zs = df[zz & (df.halo == 'annual')].sort_values('delta_keV')
# delta at which the Z-axial gives N = 1 (log interpolation)
def delta_for_N(sub, target=1.0):
    d = sub.delta_keV.values; n = sub.N_roi_unit.values; ok = n > 0
    lg = np.log(n[ok]); dd = d[ok]
    for i in range(len(dd) - 1):
        if (lg[i] - math.log(target)) * (lg[i + 1] - math.log(target)) <= 0:
            return dd[i] + (dd[i + 1] - dd[i]) * (lg[i] - math.log(target)) / (lg[i] - lg[i + 1])
    return float('nan')
dZ1 = delta_for_N(zs)
log(f'  Z-axial O4 (g_A^chi = 0.5, 1 TeV): N = {z300.N_roi_unit:.3g} (d=300), {z366.N_roi_unit:.3g} (d=366); N=1 at delta = {dZ1:.0f} keV; '
    f'g_A^chi for N=1 at d=300: {z300.gA_chi_for_N1:.3f}, d=366: {z366.gA_chi_for_N1:.3f}')
# comparison with P012 (elastic photon dipole mu for one hi event 2.1e-5 mu_N; L10 equivalent 2.8e-5 mu_N) and P011 (sigma_p)
P012_MU_HI_ELASTIC = 2.1041e-05; P012_MU_L10_EQUIV = 2.8e-5
P011_SIGMA_P = {250.0: 1.20e-42, 300.0: 8.59e-42, 350.0: 2.99e-40, 365.0: 2.29e-39}
P011_EPS_1GEV_300 = 8.7e-7
log(f'  vs P012: mu(N_hi=1, d=300)/mu_elastic(P012) = {p300.mu_N1_hi_muN/P012_MU_HI_ELASTIC:.2f}; vs L10-equivalent 2.8e-5 mu_N: {p300.mu_N1_roi_muN/P012_MU_L10_EQUIV:.2f}')

# translation to composite parameters: mu_tr = eps sqrt(alpha_D/alpha) g_tr e/(2 m_const)
def eps_needed(mu_muN, m_const_GeV, alpha_D, g_tr=1.0):
    return mu_muN * (m_const_GeV / M_P) * math.sqrt(ALPHA / alpha_D) / g_tr
trans = []
for d, p in [(300.0, p300), (366.0, p366)]:
    rr = hf[(hf.M_GeV == 1000) & (hf.delta_keV == d)].iloc[0] if d in DELTAS_KEV else None
    a_coul = float(rr.coul_alpha_eff) if rr is not None else (3.0 * d * 1e-6 / 500.0) ** 0.25
    Lam_hl = float(rr.hl_LambdaD_meson_GeV) if rr is not None else LAM_QCD * math.sqrt(d * 1e-6 * 1000 / K_MESON)
    trans.append(dict(delta_keV=d, mu_N1_muN=p.mu_N1_roi_muN,
                      eps_A_coulombic_massless_Ap=eps_needed(p.mu_N1_roi_muN, 500.0, a_coul),
                      eps_B_heavylight_massless_Ap_alphaD0p1=eps_needed(p.mu_N1_roi_muN, Lam_hl, 0.1),
                      eps_B_heavylight_mA1GeV_alphaD0p1=eps_needed(p.mu_N1_roi_muN, Lam_hl, 0.1) / math.sqrt(df[df.model == 'Aprime_mA_1.0GeV'].iloc[0].N_roi_unit / p300.N_roi_unit),
                      mu_natural_C_EWcharged_muN=(M_P / (1000.0 / 3.0)),   # e/(2 Lambda_D) with Lambda_D ~ M/3 (light-constituent baryon)
                      overshoot_C_events=(M_P / (1000.0 / 3.0)) ** 2 * p.N_roi_unit))
tr = pd.DataFrame(trans); tr.to_csv(os.path.join(OUT, 'P023_epsilon_translation.csv'), index=False, float_format='%.4g')
for _, t in tr.iterrows():
    log(f'  d={t.delta_keV:.0f}: eps(A, Coulombic, massless A\') = {t.eps_A_coulombic_massless_Ap:.3g}; eps(B, heavy-light, massless A\', alpha_D=0.1) = {t.eps_B_heavylight_massless_Ap_alphaD0p1:.3g}; '
        f'm_A\'=1 GeV: {t.eps_B_heavylight_mA1GeV_alphaD0p1:.3g} (P011 vector: 8.7e-7); EW-charged light baryon natural mu = {t.mu_natural_C_EWcharged_muN:.2e} mu_N -> {t.overshoot_C_events:.3g} events')

# ----------------------------------------------------------------------------------------------
# 3. Excited-state lifetime and the single-site requirement
# ----------------------------------------------------------------------------------------------
log('=== Part 3: chi2 lifetime ===')
def gamma_M1(mu_GeVinv, delta_GeV): return mu_GeVinv ** 2 * delta_GeV ** 3 / math.pi
# 21 cm validation: mu = mu_B, delta = 5.87e-6 eV
muB = E_CH / (2 * M_E); dE21 = H_HFS_MHZ[0] * 1e6 / EV_HZ * 1e-9
A21_formula = gamma_M1(muB, dE21) / HBAR_GEV_S
log(f'  formula check on H 21 cm: Gamma = mu_B^2 dE^3/pi -> {A21_formula:.2e} s^-1 vs measured {H_HFS_MHZ[1]:.2e} s^-1 (ratio {A21_formula/H_HFS_MHZ[1]:.1f}; spin-matrix-element factor)')
life = []
v_cm_s = V_CHI * 1e5
for d, p in [(300.0, p300), (366.0, p366)]:
    for label, mu_muN in [('N_ROI=1', p.mu_N1_roi_muN), ('N=0.105 (90% low)', p.mu_N1_roi_muN * math.sqrt(0.105)), ('N=3.65 (90% high)', p.mu_N1_roi_muN * math.sqrt(3.65))]:
        G = gamma_M1(mu_muN * MU_N_GEV, d * 1e-6); tau = HBAR_GEV_S / G
        lam_m = v_cm_s * tau / 100.0
        P_in_1m = 1 - math.exp(-L_TPC[1] / lam_m) if lam_m > 0 else 1.0
        P_in_full = 1 - math.exp(-L_TPC[0] / lam_m)
        life.append(dict(delta_keV=d, case=label, mu_muN=mu_muN, mu_ecm=mu_muN * MU_N_ECM, Gamma_GeV=G, tau_s=tau, decay_length_m=lam_m,
                         P_decay_in_1m=P_in_1m, P_decay_in_1p46m=P_in_full, tau_over_t_universe=tau / T_UNIVERSE_S))
lf = pd.DataFrame(life); lf.to_csv(os.path.join(OUT, 'P023_chi2_lifetime.csv'), index=False, float_format='%.4g')
tau_min_10pct = L_TPC[1] / (v_cm_s / 100.0) / math.log(1 / 0.9); t_cross = L_TPC[0] / (v_cm_s / 100.0)
for _, l in lf[lf.case == 'N_ROI=1'].iterrows():
    log(f'  d={l.delta_keV:.0f}, mu = {l.mu_muN:.2e} mu_N: tau = {l.tau_s:.3g} s, decay length {l.decay_length_m:.3g} m, P(decay inside 1 m) = {l.P_decay_in_1m:.2e}, tau/t_U = {l.tau_over_t_universe:.1e}')
log(f'  TPC crossing time {t_cross*1e6:.2f} us at {V_CHI:.0f} km/s; tau >= {tau_min_10pct*1e6:.1f} us keeps P(decay inside) <= 10%; a 300 keV gamma travels ~{LXE_ATT_300:.0f} cm in LXe')
# lifetime scaling: tau = pi hbar/(mu^2 delta^3); also the mu that would make the decay prompt (tau = 2 us)
mu_prompt = math.sqrt(math.pi * HBAR_GEV_S / (2e-6 * (300e-6) ** 3)) / MU_N_GEV
log(f'  mu giving tau = 2 us at delta = 300 keV: {mu_prompt:.2e} mu_N ({mu_prompt/p300.mu_N1_roi_muN:.0f}x the LZ value -> {(mu_prompt/p300.mu_N1_roi_muN)**2:.0f}x the rate)')
# single-site ceiling on delta for the photon-M1 model: the LZ rate fixes mu(delta), hence tau(delta) = pi hbar/(mu^2 delta^3)
ss_rows = []
scan_ph = df[ph & (df.halo == 'annual') & (df.m_GeV == 1000) & (df.delta_keV > 0)].sort_values('delta_keV')
for _, s in scan_ph.iterrows():
    d = s.delta_keV
    for label, fac in [('N=1', 1.0), ('N=3.65', 3.65), ('N=0.105', 0.105)]:
        mu = s.mu_N1_roi_muN * math.sqrt(fac)
        tau = HBAR_GEV_S / gamma_M1(mu * MU_N_GEV, d * 1e-6); lam_m = v_cm_s * tau / 100.0
        ss_rows.append(dict(delta_keV=d, case=label, mu_muN=mu, tau_s=tau, decay_length_m=lam_m, P_decay_in_1m=1 - math.exp(-L_TPC[1] / lam_m)))
ss = pd.DataFrame(ss_rows); ss.to_csv(os.path.join(OUT, 'P023_single_site_ceiling.csv'), index=False, float_format='%.4g')
def delta_ceiling(case, tau_target):
    t = ss[ss.case == case].sort_values('delta_keV'); d = t.delta_keV.values; lt = np.log(t.tau_s.values) - math.log(tau_target)
    for i in range(len(d) - 1):
        if lt[i] > 0 >= lt[i + 1]:
            return d[i] + (d[i + 1] - d[i]) * lt[i] / (lt[i] - lt[i + 1])
    return float('nan')
CEIL = {c: dict(P10=delta_ceiling(c, tau_min_10pct), P63=delta_ceiling(c, t_cross)) for c in ['N=1', 'N=3.65', 'N=0.105']}
log(f'  single-site ceiling on delta (tau >= {tau_min_10pct*1e6:.1f} us, P_in <= 10%): {CEIL["N=1"]["P10"]:.0f} keV (N=1), {CEIL["N=3.65"]["P10"]:.0f} keV (N=3.65), {CEIL["N=0.105"]["P10"]:.0f} keV (N=0.105); '
    f'tau = crossing time ({t_cross*1e6:.2f} us): {CEIL["N=1"]["P63"]:.0f} / {CEIL["N=3.65"]["P63"]:.0f} / {CEIL["N=0.105"]["P63"]:.0f} keV')

# ----------------------------------------------------------------------------------------------
# 4. Cosmology and self-interactions
# ----------------------------------------------------------------------------------------------
log('=== Part 4: cosmology / self-interactions ===')
def sigma_over_m_cm2_g(sigma_GeV2, M_GeV):
    return sigma_GeV2 * GEV2_CM2 / (M_GeV * 1.78266e-24)
cos = []
for M in MASSES_M:
    eta_D = OMEGA_DM_H2 * RHO_C_H2 / (M * S0)
    rr = hf[(hf.M_GeV == M) & (hf.delta_keV == 300.0)].iloc[0]
    # (A) Coulombic: geometric size ~ Bohr radius; symmetric annihilation of constituents pi alpha^2/m_Q^2
    a0_GeVinv = rr.coul_a0_fm / HBARC
    sA = 4 * math.pi * a0_GeVinv ** 2
    sv_A = math.pi * rr.coul_alpha_eff ** 2 / rr.coul_mQ_GeV ** 2 * GEV2_CM2 * C_CMS       # cm^3/s (v-independent s-wave)
    alpha_thermal = math.sqrt(SV_THERMAL / (GEV2_CM2 * C_CMS) * rr.coul_mQ_GeV ** 2 / math.pi)
    delta_if_thermal = (1.0 / 3.0) * alpha_thermal ** 4 * rr.coul_mQ_GeV
    # (B) heavy-light: geometric 4 pi/Lambda_D^2; strong annihilation <sigma v> ~ (4 pi/Lambda_D^2) v, v_FO ~ 0.3
    Lam = rr.hl_LambdaD_meson_GeV
    sB = 4 * math.pi / Lam ** 2
    sv_B = sB * 0.3 * GEV2_CM2 * C_CMS
    # (C) light baryon Lambda_D ~ M/3
    sC = 4 * math.pi / (M / 3.0) ** 2; sv_C = sC * 0.3 * GEV2_CM2 * C_CMS
    cos.append(dict(M_GeV=M, eta_D=eta_D, eta_D_over_eta_B=eta_D / ETA_B,
                    A_sigma_over_m=sigma_over_m_cm2_g(sA, M), A_sigmav_cm3s=sv_A, A_Omega_sym_h2=OMEGA_DM_H2 * SV_THERMAL / sv_A,
                    A_alpha_thermal=alpha_thermal, A_delta_if_thermal_keV=delta_if_thermal * 1e6,
                    B_LambdaD_GeV=Lam, B_sigma_over_m=sigma_over_m_cm2_g(sB, M), B_sigmav_cm3s=sv_B, B_sym_fraction=SV_THERMAL / sv_B,
                    B_glueball_GeV=GLUEBALL * Lam,
                    C_sigma_over_m=sigma_over_m_cm2_g(sC, M), C_sigmav_cm3s=sv_C, C_sym_fraction=SV_THERMAL / sv_C))
co = pd.DataFrame(cos); co.to_csv(os.path.join(OUT, 'P023_cosmology.csv'), index=False, float_format='%.4g')
c1 = co[co.M_GeV == 1000].iloc[0]
log(f'  1 TeV: eta_D = {c1.eta_D:.2e} = {c1.eta_D_over_eta_B:.2e} eta_B; A: sigma/m = {c1.A_sigma_over_m:.1e} cm^2/g, <sv> = {c1.A_sigmav_cm3s:.2e} -> Omega_sym h^2 = {c1.A_Omega_sym_h2:.3f}; '
    f'thermal alpha = {c1.A_alpha_thermal:.4f} -> delta = {c1.A_delta_if_thermal_keV:.1f} keV')
log(f'  B: Lambda_D = {c1.B_LambdaD_GeV:.2f} GeV, sigma/m = {c1.B_sigma_over_m:.1e} cm^2/g, <sv>_strong = {c1.B_sigmav_cm3s:.1e} cm^3/s -> symmetric fraction {c1.B_sym_fraction:.1e}; glueball ~ {c1.B_glueball_GeV:.1f} GeV')
log(f'  C: sigma/m = {c1.C_sigma_over_m:.1e} cm^2/g, symmetric fraction {c1.C_sym_fraction:.1e}')

# ----------------------------------------------------------------------------------------------
# 5. Benchmark table
# ----------------------------------------------------------------------------------------------
tA = tr[tr.delta_keV == 300].iloc[0]
bench = [
    dict(scenario='A: Coulombic dark atom / diquark (U(1)_D or weakly coupled SU(2)_D), m_Q = 500 GeV',
         parameters=f'alpha_eff = {r1.coul_alpha_eff:.3f}; E_B = {r1.coul_E_B_MeV:.0f} MeV; a0 = {r1.coul_a0_fm:.3f} fm',
         delta_keV=300.0, mu_tr_required_muN=p300.mu_N1_roi_muN, portal_parameter=f"eps = {tA.eps_A_coulombic_massless_Ap:.2g} (massless A', millicharged constituents)",
         tau_chi2_s=lf[(lf.delta_keV == 300) & (lf.case == 'N_ROI=1')].iloc[0].tau_s,
         relic=f'asymmetric, eta_D = {c1.eta_D:.1e}; thermal alpha would give delta = {c1.A_delta_if_thermal_keV:.0f} keV',
         sigma_over_m_cm2_g=c1.A_sigma_over_m,
         distinguishing=f'2S level at {r1.coul_2S1S_MeV:.0f} MeV (no tower at multiples of delta); delta proportional to M at fixed alpha; no dark glueballs (U(1)); '
                        f'SU(3)_D version confines at {r1.coul_LambdaD_SU3_eV:.1g} eV (dark radiation)'),
    dict(scenario='B: heavy-light dark hadron (SU(3)_D + U(1)_D), m_Q ~ 1 TeV, one light dark quark',
         parameters=f'Lambda_D = {r1.hl_LambdaD_meson_GeV:.2f} GeV (meson-like) / {r1.hl_LambdaD_baryon_GeV:.2f} GeV (Sigma_Q-like); B*-B scaled by m_b/m_Q',
         delta_keV=300.0, mu_tr_required_muN=p300.mu_N1_roi_muN,
         portal_parameter=f"eps = {tA.eps_B_heavylight_massless_Ap_alphaD0p1:.1g} (massless A', alpha_D = 0.1) or {tA.eps_B_heavylight_mA1GeV_alphaD0p1:.1g} (m_A' = 1 GeV)",
         tau_chi2_s=lf[(lf.delta_keV == 300) & (lf.case == 'N_ROI=1')].iloc[0].tau_s,
         relic=f'asymmetric, eta_D = {c1.eta_D:.1e}; symmetric remnant {c1.B_sym_fraction:.0e}', sigma_over_m_cm2_g=c1.B_sigma_over_m,
         distinguishing=f'dark glueballs ~{c1.B_glueball_GeV:.0f} GeV and light-quark excitations ~0.5 GeV (must decay before BBN); delta proportional to 1/M: 300 keV at 1 TeV <-> {r1.hl_delta_at_LambdaQCD_meson_keV:.0f} keV if Lambda_D = Lambda_QCD'),
    dict(scenario='C: light-constituent dark baryon (technibaryon-like), Lambda_D ~ M/3',
         parameters=f'delta/M = 3e-7 vs 0.31 (Delta-N): suppression {r1.light_suppression_vs_DeltaN:.0e}; or dm_q = {r1.light_dm_q_MeV:.2f} MeV explicit breaking',
         delta_keV=300.0, mu_tr_required_muN=p300.mu_N1_roi_muN,
         portal_parameter=f'EW-charged constituents give mu ~ e/(2 Lambda_D) = {tA.mu_natural_C_EWcharged_muN:.1e} mu_N -> {tA.overshoot_C_events:.0e} events: constituents must be EW-neutral; Z-axial (T3 = 1/2) gives N = {z300.N_roi_unit:.2g} at 300 keV, N = 1 at {dZ1:.0f} keV',
         tau_chi2_s=float('nan'), relic=f'asymmetric, eta_D = {c1.eta_D:.1e}', sigma_over_m_cm2_g=c1.C_sigma_over_m,
         distinguishing='no natural 300 keV scale; splitting tuned at 1e-6; rotational/radial tower at ~Lambda_D'),
]
bt = pd.DataFrame(bench); bt.to_csv(os.path.join(OUT, 'P023_benchmarks.csv'), index=False)

# ----------------------------------------------------------------------------------------------
# Figures (palette: blue #2a78d6, orange #eb6834, aqua #1baf7a, yellow #eda100; light surface)
# ----------------------------------------------------------------------------------------------
C1, C2, C3, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25})
# Fig 1: delta vs M for the scalings, with the LZ-allowed region
Mg = np.logspace(math.log10(200), math.log10(6000), 200)
fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.fill_between([259, 4000], 250, 400, color='#f0efec', label='LZ-compatible (P002/P011: δ 250–400 keV, m ≥ 259 GeV)')
for a, ls in [(0.030, ':'), (r1.coul_alpha_eff, '-'), (0.045, '--')]:
    ax.plot(Mg, (1.0 / 3.0) * a ** 4 * (Mg / 2) * 1e6, color=C1, ls=ls, lw=2, label=f'Coulombic, α_eff = {a:.3f}' if ls == '-' else None)
for L, ls in [(LAM_QCD, ':'), (r1.hl_LambdaD_meson_GeV, '-'), (0.45, '--')]:
    ax.plot(Mg, K_MESON * (L / LAM_QCD) ** 2 / (Mg - 3.3 * L) * 1e6, color=C2, ls=ls, lw=2, label=f'heavy–light, Λ_D = {L:.2f} GeV' if ls == '-' else None)
ax.plot(Mg, K_BARYON * (r1.hl_LambdaD_meson_GeV / LAM_QCD) ** 2 / (Mg - 3.3 * r1.hl_LambdaD_meson_GeV) * 1e6, color=C3, lw=2, label='heavy–light Σ_Q-like, same Λ_D')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(200, 6000); ax.set_ylim(20, 3000)
ax.set_xlabel('composite mass M [GeV]'); ax.set_ylabel('hyperfine splitting δ [keV]')
ax.set_title('Hyperfine splitting vs mass (dotted/dashed: ±1 step in the parameter)')
ax.legend(fontsize=8, loc='lower left'); fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P023_fig1_delta_vs_mass.png'), dpi=140); plt.close(fig)
# Fig 2: required transition moment and Z-axial counts vs delta
sub = df[ph & (df.halo == 'annual') & (df.m_GeV == 1000) & (df.delta_keV > 0)].sort_values('delta_keV')
fig, axs = plt.subplots(1, 2, figsize=(9.2, 4.0))
axs[0].plot(sub.delta_keV, sub.mu_N1_roi_muN, color=C1, lw=2, label='N_ROI = 1')
axs[0].fill_between(sub.delta_keV, sub.mu_N1_roi_muN * math.sqrt(0.105), sub.mu_N1_roi_muN * math.sqrt(3.65), color=C1, alpha=0.15, label='0.105–3.65 events')
axs[0].axhline(P012_MU_HI_ELASTIC, color=C2, lw=1.5, ls='--', label='P012 elastic photon dipole (excluded)')
axs[0].set_yscale('log'); axs[0].set_xlabel('δ [keV]'); axs[0].set_ylabel('transition moment μ_tr [μ_N]'); axs[0].set_title('Photon-mediated M1 transition, 1 TeV'); axs[0].legend(fontsize=8)
axs[1].plot(zs.delta_keV, zs.N_roi_unit, color=C3, lw=2, label='Z-axial, g_A^χ = 1/2')
axs[1].axhline(1.0, color='#52514e', lw=1, ls=':'); axs[1].axhspan(0.105, 3.65, color='#f0efec')
axs[1].set_yscale('log'); axs[1].set_xlabel('δ [keV]'); axs[1].set_ylabel('events in 2.84 t·yr'); axs[1].set_title('Z-mediated axial spin flip (O4 inelastic)'); axs[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P023_fig2_moment_and_Z.png'), dpi=140); plt.close(fig)
# Fig 3: lifetime vs moment with detector and cosmological lines
mus = np.logspace(-7, -1, 200)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for d, c in [(300.0, C1), (366.0, C2)]:
    ax.plot(mus, HBAR_GEV_S / gamma_M1(mus * MU_N_GEV, d * 1e-6), color=c, lw=2, label=f'δ = {d:.0f} keV')
ax.axvspan(p300.mu_N1_roi_muN * math.sqrt(0.105), p300.mu_N1_roi_muN * math.sqrt(3.65), color='#f0efec', label='LZ 90% (δ = 300)')
ax.axhline(tau_min_10pct, color='#52514e', ls=':', lw=1); ax.text(1.5e-7, tau_min_10pct * 1.5, f'single-site NR needs τ ≳ {tau_min_10pct*1e6:.0f} μs', fontsize=8)
ax.axhline(T_UNIVERSE_S, color='#52514e', ls='--', lw=1); ax.text(1.5e-7, T_UNIVERSE_S * 2, 'age of the Universe', fontsize=8)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('transition moment μ_tr [μ_N]'); ax.set_ylabel('τ(χ₂ → χ₁ γ) [s]')
ax.set_title('Excited-state lifetime, Γ = μ²δ³/π'); ax.legend(fontsize=8, loc='upper right')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P023_fig3_lifetime.png'), dpi=140); plt.close(fig)
# Fig 4: spectra
fig, ax = plt.subplots(figsize=(6.4, 4.0))
for d, c in [(300.0, C1), (366.0, C2)]:
    r = spectra[f'ph|1000|{d}|annual']; ax.plot(E, r / max(r.max(), 1e-300), color=c, lw=2, label=f'photon M1, δ = {d:.0f} keV')
rzp = spectra['Z|1000|300.0|annual']; ax.plot(E, rzp / rzp.max(), color=C3, lw=2, ls='--', label='Z-axial O4, δ = 300 keV')
ax.axvline(248, color='#52514e', lw=1, ls=':'); ax.text(250, 0.9, 'event', fontsize=8)
ax.set_xlabel('recoil energy [keV]'); ax.set_ylabel('dR/dE (normalised to peak)'); ax.set_title('Inelastic spin-flip spectra in xenon, 1 TeV (annual halo)'); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P023_fig4_spectra.png'), dpi=140); plt.close(fig)

# ----------------------------------------------------------------------------------------------
# Summary JSON
# ----------------------------------------------------------------------------------------------
summary = dict(
    validation=dict(H_hfs_MHz=H_MHz, Ps_hfs_GHz=Ps_GHz, quarkonium_ratio_obs=r_quarkonium_obs, quarkonium_ratio_pred=r_quarkonium_pred,
                    meson_ratio_obs=r_meson_obs, baryon_ratio_obs=r_baryon_obs, mc_over_mb=r_meson_pred,
                    coulombic_Jpsi_MeV=kappa_su3 * ALPHA_S_MC ** 4 * M_C * 1e3, coulombic_Ups_MeV=kappa_su3 * ALPHA_S_MB ** 4 * M_B * 1e3,
                    A21_formula=A21_formula, A21_measured=H_HFS_MHZ[1]),
    hqet=dict(K_B=K_B, K_D=K_D, K_meson=K_MESON, K_SigmaB=K_SB, K_SigmaC=K_SC, K_baryon=K_BARYON, M_LambdaQCD_meson_GeV=M_LamQCD_meson, M_LambdaQCD_baryon_GeV=M_LamQCD_baryon),
    hyperfine_1TeV_300keV=r1.to_dict(),
    photon_dipole_1TeV=dict(N_unit_roi_300=p300.N_roi_unit, N_unit_roi_366=p366.N_roi_unit, N_unit_elastic=N0[0], N_lo_per_hi_elastic=N0[2] / N0[1],
                            mu_N1_roi_muN_300=p300.mu_N1_roi_muN, mu_N1_roi_ecm_300=p300.mu_N1_roi_ecm, mu_N1_roi_e_over_2m_300=p300.mu_N1_roi_in_e_over_2mchi,
                            mu_N1_roi_muN_366=p366.mu_N1_roi_muN, mu_N1_roi_ecm_366=p366.mu_N1_roi_ecm, mu_N1_roi_e_over_2m_366=p366.mu_N1_roi_in_e_over_2mchi,
                            mu_N1_hi_muN_300=p300.mu_N1_hi_muN, mu_N1_hi_muN_366=p366.mu_N1_hi_muN, N_lo_per_hi_300=p300.N_lo_per_hi,
                            june_over_annual_300=pj300.N_roi_unit / p300.N_roi_unit, june_over_annual_366=pj366.N_roi_unit / p366.N_roi_unit,
                            Aprime_suppression={str(mA): float(df[df.model == f'Aprime_mA_{mA}GeV'].iloc[0].N_roi_unit / p300.N_roi_unit) for mA in [0.3, 1.0, 3.0]},
                            ratio_to_P012_elastic_hi=p300.mu_N1_hi_muN / P012_MU_HI_ELASTIC, ratio_to_P012_L10_equiv=p300.mu_N1_roi_muN / P012_MU_L10_EQUIV),
    Z_axial=dict(a_p=A_P, a_n=A_N, c4p=c4p, c4n=c4n, N_300=z300.N_roi_unit, N_366=z366.N_roi_unit, delta_N1_keV=dZ1, gA_for_N1_300=z300.gA_chi_for_N1, gA_for_N1_366=z366.gA_chi_for_N1),
    epsilon_translation=tr.to_dict(orient='records'), lifetime=lf.to_dict(orient='records'),
    tau_min_10pct_s=tau_min_10pct, t_cross_s=t_cross, mu_prompt_muN=mu_prompt, single_site_delta_ceiling_keV=CEIL,
    cosmology=co.to_dict(orient='records'), benchmarks=bench,
    settings=dict(exposure=EXPOSURE, E_grid='1-329 keV, 2 keV', efficiency='P003 erf (0.96; 5.4/269.9 keV; sig 3.4/8 keV)', halos='annual = 12-day mean of Baxter SHM; june16 = day 167',
                  photon_coefficients='P012 (Anand normalisation, WimPyDD c0=cp+cn)', gA_chi=G_A_CHI, runtime_s=time.time() - T0))
def _clean(o):
    if isinstance(o, dict): return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return float(o)
    return o
json.dump(_clean(summary), open(os.path.join(OUT, 'P023_summary.json'), 'w'), indent=1)
json.dump(RECALLED, open(os.path.join(OUT, 'recalled_inputs.json'), 'w'), indent=1)
open(os.path.join(OUT, 'P023_run.log'), 'w').write('\n'.join(log_lines))
log(f'done in {time.time()-T0:.0f} s; {len(RECALLED)} recalled inputs')
