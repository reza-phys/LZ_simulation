"""
P044_anapole_edm.py -- Anapole and electric-dipole dark matter versus the LZ 248 keV event:
contact and photon-mediated versions, and what L16 can and cannot be.

Run from the simulation root:   .venv/bin/python output/code/P044_anapole_edm.py

Parts
  A  constants, efficiency model (P003/P012 form), window integrals, P016-calibrated Z(N_lo) curve
  B  NREFT Hamiltonians in WimPyDD (c^0 = c_p + c_n, c^1 = c_p - c_n; P003 convention):
       anapole  L = A chibar g^mu g5 chi d^nu F_{mu nu}  -> e A chibar g^mu g5 chi J^EM_mu
                c8^N = 2 e A Q_N (O8, proton charge, velocity-suppressed coherent)
                c9^N = e A g_N  (O9, full nucleon magnetic moment g_N = 2(Q_N + kappa_N))
                [recalled/likely: Fitzpatrick+ 2012 A-V -> 2 O8 + 2 O9 for F1, tensor part -> 2 F2 O9;
                 numerically confirmed by directdm 2.2.2 output quoted in work/P031 (c8p 0.606 = 2e, c9p/c8p = g_p/2)]
       EDM      L = (d_E/2) chibar sigma^{mu nu} i g5 chi F_{mu nu} -> c11^N = 2 e d_E Q_N m_N / q^2  (photon, long range)
                [recalled/likely; same structure as P012's c5 = 2 e mu m_N Q_N/q^2; directdm C52 gives c11^p = 0.072 at q = 245 MeV, matches]
       contact analogues: EDM-type with heavy mediator (O11 constant, proton only), O11 x q^2/m_v^2; anapole-type A-V
                with a heavy vector (F2 = 0: c9/c8 = 1) and a q^2-weighted anapole; pure O9 (Q17/L9-like) and q^2 O9
       L10 reference (P012 implementation)
  C  spectra at 200 / 1000 / 4000 GeV, window rates, N_lo per 200-270 keV event, Z_pred (P016 curve),
     couplings for 1.0 LZ event (whole ROI; and 1 event in 200-270 keV), physical moments, 2024-null bounds
  D  distinguishability from L10: peaks, 100-200/200-270 ratios, KL-based number of events for 3 sigma separation
  E  figures, CSV/JSON
Outputs -> output/work/P044/*.csv, *.json, figures/*.png
"""
from __future__ import annotations
import sys, os, json, math, time
sys.path.insert(0, 'output/code')
import numpy as np
import pandas as pd
from scipy import special, integrate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import lzcommon as lz

OUT = 'output/work/P044'
FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)
T0 = time.time()
LOG = open(os.path.join(OUT, 'P044_run.log'), 'w')
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()

# ------------------------------------------------------------------------------------------------
# Part A: constants and helpers
# ------------------------------------------------------------------------------------------------
MN = lz.M_NUCLEON_GEV                  # 0.938272 GeV
MV2 = lz.M_V_GEV ** 2                  # (246.2 GeV)^2
HBARC_FM = lz.HBARC_GEV_FM             # 0.19733 GeV fm
HBARC_CM = HBARC_FM * 1e-13            # GeV cm
ALPHA_EM = 1.0 / 137.036               # recalled, certain
E_CH = math.sqrt(4 * math.pi * ALPHA_EM)   # 0.30282 (Heaviside-Lorentz)
G_P, G_N = 5.5857, -3.8261             # nucleon g-factors (recalled, certain)
MU_NUC_ECM = HBARC_CM / (2 * MN)       # nuclear magneton in e cm = 1.052e-14
EXPOSURE = lz.LZ['exposure_tyr']       # 2.84 t yr
EXPOSURE_2024 = 4.2                    # t yr (recalled, certain; as P012)
TOL_2024 = (3.0, 5.0)                  # tolerated 5.4-55 keV signal events in the 2024 search (recalled, uncertain; P003/P012)
MASSES = [200.0, 1000.0, 4000.0]
N_BEST, N_LO68, N_UP68 = 1.0, 0.3, 2.4  # Table I best fit 1.0 +1.4 -0.7 events (L10, 1000 GeV)

def efficiency(E, sig_lo=3.4, sig_hi=8.0, eff0=lz.LZ['eff_plateau']):
    E = np.asarray(E, float)
    lo, hi = lz.LZ['E_50pct_low_keV'], lz.LZ['E_50pct_high_keV']
    return eff0 * 0.5 * (1 + special.erf((E - lo) / (math.sqrt(2) * sig_lo))) * 0.5 * (1 - special.erf((E - hi) / (math.sqrt(2) * sig_hi)))

def interp_spec(Egrid, dRdE, Ef):
    y = np.asarray(dRdE, float)
    if np.all(y > 0):
        return 10 ** np.interp(Ef, Egrid, np.log10(y))
    return np.interp(Ef, Egrid, y)

def window_rate(Egrid, dRdE, e1, e2):
    Ef = np.linspace(e1, e2, 2001)
    return float(integrate.trapezoid(interp_spec(Egrid, dRdE, Ef) * efficiency(Ef), Ef))

def summarise(Egrid, dRdE):
    R_lo = window_rate(Egrid, dRdE, 5.4, 55.0)
    R_mid = window_rate(Egrid, dRdE, 55.0, 200.0)
    R_m2 = window_rate(Egrid, dRdE, 100.0, 200.0)
    R_hi = window_rate(Egrid, dRdE, 200.0, 270.0)
    R_roi = window_rate(Egrid, dRdE, 1.0, 330.0)
    R_edge = window_rate(Egrid, dRdE, 270.0, 330.0)          # efficiency-weighted rate above the 50% edge (roll-off region)
    R_raw_hi = float(integrate.trapezoid(interp_spec(Egrid, dRdE, np.linspace(270.0, 330.0, 601)), np.linspace(270.0, 330.0, 601)))
    R_raw_all = float(integrate.trapezoid(interp_spec(Egrid, dRdE, np.linspace(1.0, 330.0, 3291)), np.linspace(1.0, 330.0, 3291)))
    Ef = np.linspace(238.0, 258.0, 201)                        # density at 248 +- 10 keV per ROI event (in-window shape factor, P027's rho)
    rho248 = float(integrate.trapezoid(interp_spec(Egrid, dRdE, Ef) * efficiency(Ef), Ef)) / 20.0 / R_roi
    return dict(R_lo=R_lo, R_mid=R_mid, R_100_200=R_m2, R_hi=R_hi, R_roi=R_roi,
                N_lo_per_hi=R_lo / R_hi, N_mid_per_hi=R_mid / R_hi, ratio_100_200_over_hi=R_m2 / R_hi,
                f_lo=R_lo / R_roi, f_hi=R_hi / R_roi, f_edge_270_330=R_edge / R_roi, f_raw_above270=R_raw_hi / R_raw_all,
                rho_248_per_roi_event=rho248)

# P016-calibrated local significance as a function of N_lo (their Table: Z_model vs N_lo at 1000 GeV, plus the
# thresholds Z = 3, 2, 1 at N_lo = 10.9, 160, 937). Log-linear interpolation, as used by P031.
P016_NLO = np.array([0.20, 0.43, 3.1, 10.9, 28.0, 160.0, 258.0, 477.0, 937.0, 2752.0])
P016_Z = np.array([3.40, 3.26, 3.08, 3.00, 2.70, 2.00, 1.90, 1.65, 1.00, 0.48])
def z_pred(nlo):
    return float(np.interp(math.log10(max(nlo, 1e-3)), np.log10(P016_NLO), P016_Z, left=3.45, right=0.0))

# ------------------------------------------------------------------------------------------------
# Part B: Hamiltonians.  Coefficient values are captured by closure (WimPyDD treats every non-reserved
# argument, including defaults, as a shared global parameter -- P031 pitfall).  Reserved: q, mchi, delta, q0*.
# ------------------------------------------------------------------------------------------------
WD = lz.wd()
halo = lz.wd_halo()      # Sun-frame Baxter-2021 SHM with the P002 v_min grid fix

def const_fn(cp, cn):
    c0, c1 = cp + cn, cp - cn
    def f():
        return [c0, c1]
    return f
def qpow_fn(cp, cn, power, scale=1.0):
    """coefficient (cp, cn) x (q^power / scale)"""
    c0, c1 = cp + cn, cp - cn
    def f(q):
        fac = q ** power / scale
        return [c0 * fac, c1 * fac]
    return f

def ham(name, wc):
    return WD.eft_hamiltonian('P044_' + name, wc)

HAMS = {}
# (1) photon anapole, A = 1 GeV^-2: c8^p = 2e, c8^n = 0 ; c9^N = e g_N
HAMS['anapole_full'] = ham('anapole_full', {8: const_fn(2 * E_CH, 0.0), 9: const_fn(E_CH * G_P, E_CH * G_N)})
HAMS['anapole_O8only'] = ham('anapole_O8only', {8: const_fn(2 * E_CH, 0.0)})
HAMS['anapole_O9only'] = ham('anapole_O9only', {9: const_fn(E_CH * G_P, E_CH * G_N)})
HAMS['anapole_flipsign'] = ham('anapole_flipsign', {8: const_fn(2 * E_CH, 0.0), 9: const_fn(-E_CH * G_P, -E_CH * G_N)})
# (2) contact anapole analogues: heavy vector with proton vector charge only (F2 = 0): c8 = c9 = 2 e A_c (Fitzpatrick 2O8+2O9)
HAMS['AV_heavy_F1only'] = ham('AV_heavy_F1only', {8: const_fn(2 * E_CH, 0.0), 9: const_fn(2 * E_CH, 0.0)})
# q^2-weighted anapole (dimension-8 contact analogue, coefficients x q^2/m_v^2)
HAMS['anapole_q2'] = ham('anapole_q2', {8: qpow_fn(2 * E_CH, 0.0, 2, MV2), 9: qpow_fn(E_CH * G_P, E_CH * G_N, 2, MV2)})
# (3) photon EDM, d_E = 1 GeV^-1: c11^p = 2 e m_N / q^2, c11^n = 0
HAMS['edm_photon'] = ham('edm_photon', {11: qpow_fn(2 * E_CH * MN, 0.0, -2)})
# contact EDM-like: heavy mediator M -> c11^p = 2 e d_E m_N / M^2 ; computed at unit c11^p = 1 GeV^-2 (proton only)
HAMS['edm_contact_q0'] = ham('edm_contact_q0', {11: const_fn(1.0, 0.0)})
# O11 x q^2/m_v^2 (proton only), unit c11^p = q^2/m_v^2
HAMS['edm_contact_q2'] = ham('edm_contact_q2', {11: qpow_fn(1.0, 0.0, 2, MV2)})
# (4) pure O9 (Q17 / L9-like axial-tensor structure), isoscalar Anand unit c^0 = 1/m_v^2 -> WimPyDD 2/m_v^2
HAMS['O9_iso'] = ham('O9_iso', {9: const_fn(1.0 / MV2, 1.0 / MV2)})
HAMS['O9_q2_iso'] = ham('O9_q2_iso', {9: qpow_fn(1.0 / MV2, 1.0 / MV2, 2, MN ** 2)})
# O14 isoscalar (axial-DM x pseudotensor-nucleon type candidates reduce to velocity-suppressed spin operators)
HAMS['O14_iso'] = ham('O14_iso', {14: const_fn(1.0 / MV2, 1.0 / MV2)})
# (5) L10 reference (P012): 4[(q^2/m_N^2) O4 - O6] x d10/m_v^2, Anand isoscalar -> WimPyDD x2; d10 = 1
HAMS['L10_ref'] = ham('L10_ref', {4: qpow_fn(4.0 / MV2, 4.0 / MV2, 2, MN ** 2), 6: const_fn(-4.0 / MV2, -4.0 / MV2)})

ALL_MASS = ['anapole_full', 'edm_photon', 'edm_contact_q0', 'edm_contact_q2', 'O9_iso', 'O9_q2_iso', 'L10_ref']   # run at all masses
ONE_TEV_ONLY = [k for k in HAMS if k not in ALL_MASS]

E = np.union1d(np.arange(1.0, 330.0 + 1e-9, 3.0), [5.4, 55.0, 100.0, 200.0, 248.0, 269.9, 300.0])
log(f'E grid: {len(E)} points; halo v_max = {halo[0][-1]:.0f} km/s; hamiltonians: {list(HAMS)}')

spectra = {}
for name, h in HAMS.items():
    for m in (MASSES if name in ALL_MASS else [1000.0]):
        spectra[(name, m)] = lz.wd_rate(h, m, E, halo)
        log(f'  {name:18s} m={m:5.0f} GeV  dR/dE(248) = {spectra[(name, m)][np.searchsorted(E, 248.0)]:.3e} /t/yr/keV  ({time.time()-T0:.0f}s)')
np.savez(os.path.join(OUT, 'P044_spectra_unit_coupling.npz'), E=E, **{f'{k[0]}__m{int(k[1])}': v for k, v in spectra.items()})

# ------------------------------------------------------------------------------------------------
# Part C: window rates, N_lo, couplings for one LZ event, physical moments, 2024-null bounds
# ------------------------------------------------------------------------------------------------
UNIT = {  # unit coupling used in the Hamiltonian and how the rate scales
    'anapole_full': ('A', 'GeV^-2', 1.0), 'anapole_O8only': ('A', 'GeV^-2', 1.0), 'anapole_O9only': ('A', 'GeV^-2', 1.0),
    'anapole_flipsign': ('A', 'GeV^-2', 1.0), 'AV_heavy_F1only': ('A_c', 'GeV^-2', 1.0), 'anapole_q2': ('A_8', 'GeV^-2', 1.0),
    'edm_photon': ('d_E', 'GeV^-1', 1.0), 'edm_contact_q0': ('c11^p', 'GeV^-2', 1.0), 'edm_contact_q2': ('c11^p(q=m_v)', 'GeV^-2', 1.0),
    'O9_iso': ('c9^s m_v^2', '1', 1.0), 'O9_q2_iso': ('c9^s m_v^2 (q=m_N)', '1', 1.0), 'O14_iso': ('c14^s m_v^2', '1', 1.0),
    'L10_ref': ('d10 (Anand x WimPyDD norm.)', '1', 1.0)}
rows = []
for (name, m), sp in spectra.items():
    s = summarise(E, sp)
    N_roi_unit = s['R_roi'] * EXPOSURE
    N_hi_unit = s['R_hi'] * EXPOSURE
    g_roi = math.sqrt(N_BEST / N_roi_unit)        # coupling for 1.0 event in the whole ROI (LZ best-fit style)
    g_hi = math.sqrt(1.0 / N_hi_unit)             # coupling for 1 event in 200-270 keV
    N2024_lo_per_hi = s['N_lo_per_hi'] * EXPOSURE_2024 / EXPOSURE
    row = dict(structure=name, m_GeV=m, coupling=UNIT[name][0], unit=UNIT[name][1], **s,
               N_roi_unit=N_roi_unit, N_hi_unit=N_hi_unit, g_1roi=g_roi, g_1roi_lo68=math.sqrt(N_LO68 / N_roi_unit),
               g_1roi_up68=math.sqrt(N_UP68 / N_roi_unit), g_1hi=g_hi, Z_pred_P016=z_pred(s['N_lo_per_hi']),
               N2024_lo_if_1hi=N2024_lo_per_hi, N2024_lo_if_1roi=s['f_lo'] * EXPOSURE_2024 / EXPOSURE,
               g_max_2024_tol3=g_hi * math.sqrt(TOL_2024[0] / N2024_lo_per_hi), g_max_2024_tol5=g_hi * math.sqrt(TOL_2024[1] / N2024_lo_per_hi),
               dRdE_20=float(sp[np.searchsorted(E, 19.0)]), dRdE_248=float(sp[np.searchsorted(E, 248.0)]),
               E_peak_keV=float(E[np.argmax(sp * efficiency(E))]))
    # physical conversions
    if name.startswith('anapole') or name == 'AV_heavy_F1only':
        row.update(A_1roi_GeV2=g_roi, A_1roi_efm2=g_roi * HBARC_FM ** 2, Lambda_1roi_GeV=g_roi ** -0.5,
                   A_1hi_GeV2=g_hi, A_max_2024_tol3_GeV2=row['g_max_2024_tol3'], A_max_2024_tol5_GeV2=row['g_max_2024_tol5'])
    if name == 'edm_photon':
        row.update(dE_1roi_GeV1=g_roi, dE_1roi_ecm=g_roi * HBARC_CM, dE_1hi_ecm=g_hi * HBARC_CM,
                   dE_max_2024_tol3_ecm=row['g_max_2024_tol3'] * HBARC_CM, dE_max_2024_tol5_ecm=row['g_max_2024_tol5'] * HBARC_CM)
    if name == 'edm_contact_q0':
        # c11^p = 2 e d_E m_N / M^2  ->  d_E / M^2 = c11 / (2 e m_N); quote d_E for M = 1 TeV in e cm
        row.update(dE_over_M2_1roi_GeV3=g_roi / (2 * E_CH * MN), dE_for_M1TeV_1roi_ecm=g_roi / (2 * E_CH * MN) * 1e6 * HBARC_CM)
    rows.append(row)
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, 'P044_results_table.csv'), index=False, float_format='%.5g')
log('\nPart C summary (1000 GeV):')
log(res[res.m_GeV == 1000][['structure', 'N_lo_per_hi', 'N_mid_per_hi', 'ratio_100_200_over_hi', 'f_hi', 'Z_pred_P016', 'E_peak_keV', 'g_1roi', 'g_1hi', 'N2024_lo_if_1hi']].to_string(index=False))
log('\nMass dependence:')
log(res[res.structure.isin(ALL_MASS)][['structure', 'm_GeV', 'N_lo_per_hi', 'Z_pred_P016', 'g_1roi', 'g_1hi', 'g_max_2024_tol3']].to_string(index=False))
log('\nHardness vs mass (L16 pattern test: Table S6 L16^s = 2.8/3.3/3.4/3.4/3.2 at 100/200/400/1000/4000 GeV; L10^s = 2.9/3.1/3.4/3.4/3.4):')
log(res[res.structure.isin(['L10_ref', 'O9_iso', 'O9_q2_iso', 'anapole_full'])][['structure', 'm_GeV', 'E_peak_keV', 'f_hi', 'f_edge_270_330', 'f_raw_above270', 'rho_248_per_roi_event', 'N_lo_per_hi']].to_string(index=False))

# anapole decomposition and sign robustness at 1000 GeV
def pick(name, m=1000.0):
    return res[(res.structure == name) & (res.m_GeV == m)].iloc[0]
a_full, a_8, a_9, a_flip = pick('anapole_full'), pick('anapole_O8only'), pick('anapole_O9only'), pick('anapole_flipsign')
decomp = dict(
    O8_fraction_R_lo=a_8.R_lo / a_full.R_lo, O9_fraction_R_lo=a_9.R_lo / a_full.R_lo,
    O8_fraction_R_hi=a_8.R_hi / a_full.R_hi, O9_fraction_R_hi=a_9.R_hi / a_full.R_hi,
    O8_fraction_at_20keV=a_8.dRdE_20 / a_full.dRdE_20, O9_fraction_at_248keV=a_9.dRdE_248 / a_full.dRdE_248,
    interference_R_lo=(a_full.R_lo - a_8.R_lo - a_9.R_lo) / a_full.R_lo, interference_R_hi=(a_full.R_hi - a_8.R_hi - a_9.R_hi) / a_full.R_hi,
    N_lo_full=a_full.N_lo_per_hi, N_lo_O8only=a_8.N_lo_per_hi, N_lo_O9only=a_9.N_lo_per_hi, N_lo_flipsign=a_flip.N_lo_per_hi,
    R_hi_flip_over_full=a_flip.R_hi / a_full.R_hi, R_lo_flip_over_full=a_flip.R_lo / a_full.R_lo)
log('\nAnapole decomposition (1000 GeV): ' + json.dumps(decomp, default=float))

# cross-checks against P003 / P012 / P031 published numbers
xchk = dict(O9_iso_N_lo_1000=pick('O9_iso').N_lo_per_hi, P003_O9s=2.1,
            edm_contact_q0_N_lo_1000=pick('edm_contact_q0').N_lo_per_hi, P003_O11s_isoscalar=258, P031_PS_O11=260,
            anapole_full_N_lo_1000=a_full.N_lo_per_hi, P031_anapole=26.9,
            edm_photon_N_lo_1000=pick('edm_photon').N_lo_per_hi, P031_edm_photon=18200,
            L10_ref_N_lo_1000=pick('L10_ref').N_lo_per_hi, P003_P012_L10=0.20,
            L10_ref_N_unit_wd_1000=pick('L10_ref').N_roi_unit, P012_N_unit_wd_1000=3.34,
            anapole_A_1roi_1000=a_full.A_1roi_GeV2, P031_anapole_a_1event=2.73e-6)
log('Cross-checks: ' + json.dumps(xchk, default=float))

# ------------------------------------------------------------------------------------------------
# Part D: distinguishability from L10 (shape only; both normalised to unit probability in the efficiency-weighted ROI)
# ------------------------------------------------------------------------------------------------
Ef = np.linspace(1.0, 330.0, 3300)
def pdf(name, m=1000.0, smear_sigma248=None):
    y = interp_spec(E, spectra[(name, m)], Ef) * efficiency(Ef)
    if smear_sigma248:
        # Gaussian resolution sigma(E) = sigma248 * sqrt(E/248) (recalled/likely scaling; 23 keV stat at 248 keV from LZ)
        sig = smear_sigma248 * np.sqrt(np.maximum(Ef, 1.0) / 248.0)
        ker = np.exp(-0.5 * ((Ef[:, None] - Ef[None, :]) / sig[None, :]) ** 2) / (sig[None, :] * math.sqrt(2 * math.pi))
        y = ker @ y * (Ef[1] - Ef[0])
    return y / integrate.trapezoid(y, Ef)
def separation(p, q):
    """Expected per-event log-likelihood ratio (KL) and its variance under p; events for 3 sigma (median) separation."""
    ok = (p > 0) & (q > 0)
    lr = np.zeros_like(p); lr[ok] = np.log(p[ok] / q[ok])
    kl = integrate.trapezoid(p * lr, Ef)
    var = integrate.trapezoid(p * lr ** 2, Ef) - kl ** 2
    kl_q = -integrate.trapezoid(q * lr, Ef)
    var_q = integrate.trapezoid(q * lr ** 2, Ef) - kl_q ** 2
    # decide between p and q with N events: Z ~ sqrt(N) * (KL_p + KL_q) / (sqrt(Var_p) + sqrt(Var_q))  (symmetric threshold)
    Zsym = (kl + kl_q) / (math.sqrt(var) + math.sqrt(var_q))
    return dict(KL_p_q=kl, KL_q_p=kl_q, N_3sigma_sym=9.0 / Zsym ** 2, N_3sigma_from_p=9.0 * var / kl ** 2 if kl > 0 else np.inf)
p_L10 = pdf('L10_ref'); p_L10_s = pdf('L10_ref', smear_sigma248=23.0)
sep_rows = []
for name in ['anapole_full', 'anapole_O9only', 'O9_iso', 'O9_q2_iso', 'O14_iso', 'edm_contact_q2', 'edm_contact_q0', 'edm_photon', 'AV_heavy_F1only', 'anapole_q2']:
    p = pdf(name); ps = pdf(name, smear_sigma248=23.0)
    d0 = separation(p, p_L10); d1 = separation(ps, p_L10_s)
    r = res[(res.structure == name) & (res.m_GeV == 1000)].iloc[0]; rL = pick('L10_ref')
    sep_rows.append(dict(structure=name, E_peak_keV=r.E_peak_keV, E_peak_L10=rL.E_peak_keV,
                         ratio_100_200_over_hi=r.ratio_100_200_over_hi, ratio_L10=rL.ratio_100_200_over_hi,
                         ratio_mid_over_hi=r.N_mid_per_hi, ratio_mid_L10=rL.N_mid_per_hi,
                         KL_vs_L10_ideal=d0['KL_p_q'], N3sigma_ideal=d0['N_3sigma_sym'], KL_vs_L10_smeared=d1['KL_p_q'], N3sigma_smeared=d1['N_3sigma_sym']))
sep = pd.DataFrame(sep_rows)
sep.to_csv(os.path.join(OUT, 'P044_distinguishability_vs_L10.csv'), index=False, float_format='%.4g')
log('\nPart D (1000 GeV, vs L10):'); log(sep.to_string(index=False))

# ------------------------------------------------------------------------------------------------
# Part E: figures
# ------------------------------------------------------------------------------------------------
STYLE = {'anapole_full': ('C0', '-', 'photon anapole (O8+O9)'), 'anapole_O8only': ('C0', '--', '  charge part O8 only'),
         'anapole_O9only': ('C0', ':', '  magnetic part O9 only'), 'edm_photon': ('C3', '-', 'photon EDM, O11/q$^2$'),
         'edm_contact_q0': ('C3', '--', 'contact EDM-type, O11'), 'edm_contact_q2': ('C3', ':', 'O11 $\\times$ q$^2$'),
         'O9_iso': ('C2', '-', 'pure O9 (axial-tensor, L9/Q17-like)'), 'O9_q2_iso': ('C2', ':', 'O9 $\\times$ q$^2$'),
         'L10_ref': ('k', '-', 'L10 (P012 reference)'), 'AV_heavy_F1only': ('C1', '--', 'heavy-vector A-V, F$_2$=0'),
         'anapole_q2': ('C1', ':', 'anapole $\\times$ q$^2$'), 'O14_iso': ('C4', '-.', 'O14')}
fig, ax = plt.subplots(figsize=(7.4, 4.8))
for name in ['L10_ref', 'anapole_full', 'anapole_O8only', 'anapole_O9only', 'edm_photon', 'edm_contact_q0', 'edm_contact_q2', 'O9_iso']:
    sp = spectra[(name, 1000.0)]; s = summarise(E, sp)
    c, ls, lab = STYLE[name]
    ax.plot(E, sp * efficiency(E) / s['R_roi'], color=c, ls=ls, lw=1.5, label=f"{lab} (N$_{{lo}}$={s['N_lo_per_hi']:.3g})")
ax.axvspan(5.4, 55, color='C3', alpha=0.06); ax.axvspan(200, 270, color='C2', alpha=0.10); ax.axvline(248, color='k', ls=':', lw=0.8)
ax.set_yscale('log'); ax.set_xlim(0, 330); ax.set_ylim(1e-5, 3)
ax.set_xlabel('$E_R$ [keV]'); ax.set_ylabel('efficiency-weighted dR/dE per ROI event [keV$^{-1}$]')
ax.set_title('P044: anapole, EDM and axial-tensor spectra at 1 TeV, each normalised to 1.0 LZ event'); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P044_spectra_1TeV.png'), dpi=140); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.4, 4.4))
order = ['L10_ref', 'O9_iso', 'O9_q2_iso', 'anapole_O9only', 'O14_iso', 'edm_contact_q2', 'anapole_q2', 'anapole_full', 'AV_heavy_F1only', 'edm_contact_q0', 'edm_photon']
vals = [pick(n).N_lo_per_hi for n in order]
cols = ['C2' if v <= 5 else ('C1' if v <= 30 else 'C3') for v in vals]
ax.bar(range(len(order)), vals, color=cols)
ax.axhline(3, color='k', ls='--', lw=0.8); ax.axhline(5, color='k', ls=':', lw=0.8)
ax.set_yscale('log'); ax.set_xticks(range(len(order))); ax.set_xticklabels([STYLE[n][2].strip() for n in order], rotation=35, ha='right', fontsize=7.5)
ax.set_ylabel('N$_{lo}$ = events in 5.4-55 keV per event in 200-270 keV'); ax.set_title('P044: low-energy companions per 248 keV-class event (1 TeV)')
for i, (v, n) in enumerate(zip(vals, order)):
    ax.text(i, v * 1.25, f'{v:.3g}\nZ$\\approx${pick(n).Z_pred_P016:.1f}$\\sigma$', ha='center', fontsize=6.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P044_Nlo_bar.png'), dpi=140); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.2))
an = res[res.structure == 'anapole_full'].sort_values('m_GeV'); ed = res[res.structure == 'edm_photon'].sort_values('m_GeV')
axs[0].fill_between(an.m_GeV, an.g_1roi_lo68, an.g_1roi_up68, color='C0', alpha=0.25, label='1.0 (0.3-2.4) LZ events, whole ROI')
axs[0].plot(an.m_GeV, an.g_1roi, 'o-', color='C0'); axs[0].plot(an.m_GeV, an.g_1hi, 's--', color='C0', label='1 event in 200-270 keV')
axs[0].plot(an.m_GeV, an.g_max_2024_tol3, 'k-', lw=1.2, label='2024 null (5.4-55 keV, $\\leq$3 events)'); axs[0].plot(an.m_GeV, an.g_max_2024_tol5, 'k:', lw=1.2, label='$\\leq$5 events')
axs[0].set_xscale('log'); axs[0].set_yscale('log'); axs[0].set_xlabel('m$_\\chi$ [GeV]'); axs[0].set_ylabel('anapole moment $\\mathcal{A}$ [GeV$^{-2}$]'); axs[0].legend(fontsize=7); axs[0].set_title('photon anapole')
axs[1].fill_between(ed.m_GeV, ed.g_1roi_lo68 * HBARC_CM, ed.g_1roi_up68 * HBARC_CM, color='C3', alpha=0.25, label='1.0 (0.3-2.4) LZ events, whole ROI')
axs[1].plot(ed.m_GeV, ed.g_1roi * HBARC_CM, 'o-', color='C3'); axs[1].plot(ed.m_GeV, ed.g_1hi * HBARC_CM, 's--', color='C3', label='1 event in 200-270 keV')
axs[1].plot(ed.m_GeV, ed.g_max_2024_tol3 * HBARC_CM, 'k-', lw=1.2, label='2024 null ($\\leq$3 events)'); axs[1].plot(ed.m_GeV, ed.g_max_2024_tol5 * HBARC_CM, 'k:', lw=1.2, label='$\\leq$5 events')
axs[1].set_xscale('log'); axs[1].set_yscale('log'); axs[1].set_xlabel('m$_\\chi$ [GeV]'); axs[1].set_ylabel('electric dipole d$_E$ [e cm]'); axs[1].legend(fontsize=7); axs[1].set_title('photon EDM')
fig.suptitle('P044: moments giving the LZ event vs the low-energy null'); fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P044_moment_vs_mass.png'), dpi=140); plt.close(fig)

# ------------------------------------------------------------------------------------------------
# summary JSON
# ------------------------------------------------------------------------------------------------
summary = dict(
    settings=dict(masses=MASSES, exposure_tyr=EXPOSURE, exposure_2024_tyr=EXPOSURE_2024, tol_2024=TOL_2024, halo='lz.wd_halo() Sun-frame Baxter-2021 SHM',
                  vmax_kms=float(halo[0][-1]), E_grid=f'1-330 keV, 3 keV steps + reference energies ({len(E)} points)',
                  efficiency='P003 erf model (0.96; 50% at 5.4/269.9 keV; sig 3.4/8 keV)',
                  constants=dict(e=E_CH, g_p=G_P, g_n=G_N, m_N=MN, m_v2=MV2, hbarc_fm=HBARC_FM, mu_N_ecm=MU_NUC_ECM),
                  reductions=dict(anapole='c8^N = 2 e A Q_N, c9^N = e A g_N (L = A chibar g^mu g5 chi d^nu F_{mu nu}); recalled/likely, directdm-consistent (P031)',
                                  edm='c11^N = 2 e d_E Q_N m_N / q^2 (L = (d_E/2) chibar sigma^{mu nu} i g5 chi F_{mu nu}); recalled/likely, directdm-consistent (P031 c11^p = 0.072 at q = 245 MeV for C52 = 1)',
                                  L10='P012: 4[(q^2/m_N^2) O4 - O6] d10/m_v^2, Anand isoscalar x2 for WimPyDD')),
    results=res.to_dict(orient='records'), anapole_decomposition_1000GeV=decomp, cross_checks=xchk,
    distinguishability_vs_L10_1000GeV=sep.to_dict(orient='records'), runtime_s=time.time() - T0)
with open(os.path.join(OUT, 'P044_summary.json'), 'w') as f:
    json.dump(summary, f, indent=1, default=float)
log(f'done in {time.time()-T0:.0f} s')
LOG.close()
