"""
P096 -- Large liquid-scintillator detectors as high-energy recoil detectors: can Borexino, KamLAND, JUNO or
SNO+ see the LZ models through 12C, 13C, 1H (and 16O) recoils?

Run from the simulation root:   .venv/bin/python output/code/P096_scintillators.py   [--recompute]

Parts
 1. Kinematics on light nuclei (A = 1, 12, 13, 16 and a scan): elastic E_R,max, inelastic ceiling mu v_max^2/2,
    delta_max, A_min(delta) at v_max = v_esc + v_E(16 June).
 2. Elastic q^4-spin models (L10 dipole-dipole and O6) on 1H via WimPyDD (1H response functions exist, itar=21),
    validated against an analytic free-proton calculation; 13C by the single-particle estimate (WimPyDD has
    itar=0 for 13C, i.e. no response functions); normalised to LZ's 1.0 accepted event in 2.84 t yr.
    Elastic SI reference (O1) on C, H, O at sigma_n = 1e-47 cm^2.
 3. Exothermic (delta < 0) O1 down-scattering of a surviving chi_2 on C, H, O with WimPyDD kernels, coupling fixed by
    the endothermic interpretation (kappa = 1 / N_endo,ROI per unit coupling, 24-day annual mean), f_2 from P058.
 4. Detector response: Birks/Cecil quenching (recalled), 14C background, thresholds, S/B and stat-only reach.
"""
from __future__ import annotations
import sys, os, json, math, time
import numpy as np
import pandas as pd
sys.path.insert(0, 'output/code')
from common import lzcommon as lz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import integrate, special, stats

OUT = 'output/work/P096'; FIG = f'{OUT}/figures'
os.makedirs(FIG, exist_ok=True)
LOG = open(f'{OUT}/run_log.txt', 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n'); LOG.flush()

T0 = time.time()
WD = lz.wd()
C = lz.C_KMS; MV = lz.M_V_GEV; MN = lz.M_NUCLEON_GEV; AMU = lz.AMU_GEV
M_CHI = 1000.0
DOY_EVENT = 167                      # 16 June
V_E_JUNE = lz.v_earth_kms(DOY_EVENT)
V_MAX = lz.VESC_KMS + V_E_JUNE
EXPO_LZ = lz.LZ['exposure_tyr']
DELTAS = [300.0, 350.0, 366.0, 380.0]
RECOMPUTE = '--recompute' in sys.argv
say(f'v_E(16 June) = {V_E_JUNE:.1f} km/s, v_max = {V_MAX:.1f} km/s')

# =====================================================================================================
# Detectors (all recalled; see details.md for reliability flags)
# =====================================================================================================
# mass fractions from the chemical formula (certain given the formula): LAB ~ C18H30, PC = C9H12, dodecane C12H26
def mass_fractions(formula):
    """formula: dict element -> number of atoms; returns mass fractions with A(C)=12.011, A(H)=1.008, A(O)=15.999."""
    Aw = {'C': 12.011, 'H': 1.008, 'O': 15.999}
    tot = sum(Aw[e] * n for e, n in formula.items())
    return {e: Aw[e] * n / tot for e, n in formula.items()}

F_LAB = mass_fractions({'C': 18, 'H': 30})
F_PC = mass_fractions({'C': 9, 'H': 12})
F_DOD = mass_fractions({'C': 12, 'H': 26})
F_KAM = {e: 0.8 * F_DOD[e] + 0.2 * F_PC[e] for e in ('C', 'H')}   # 80% dodecane + 20% PC by volume ~ by mass (approx.)
F_H2O = mass_fractions({'H': 2, 'O': 1})
DETECTORS = {
    # name: mass [t], fractions, threshold [keV_ee] (recalled/uncertain), 14C/12C ratio (recalled), status
    'Borexino': dict(mass_t=278.0, frac=F_PC, thr_keVee=50.0, c14=2.7e-18, status='2007-2021 (ended)'),
    'KamLAND': dict(mass_t=1000.0, frac=F_KAM, thr_keVee=200.0, c14=1.0e-17, status='running'),
    'SNO+': dict(mass_t=780.0, frac=F_LAB, thr_keVee=200.0, c14=1.0e-17, status='scintillator phase running'),
    'JUNO': dict(mass_t=20000.0, frac=F_LAB, thr_keVee=100.0, c14=1.0e-17, status='commissioning 2025-26'),
    'SNO+ water': dict(mass_t=905.0, frac=F_H2O, thr_keVee=3500.0, c14=0.0, status='2017-2019 phase (Cherenkov)'),
}
say('mass fractions LAB', {k: round(v, 4) for k, v in F_LAB.items()}, 'PC', {k: round(v, 4) for k, v in F_PC.items()},
    'KamLAND', {k: round(v, 4) for k, v in F_KAM.items()})

# =====================================================================================================
# Part 1: kinematics
# =====================================================================================================
def ceiling_kev(A, m_chi=M_CHI, v=V_MAX):
    mu = lz.mu_red(m_chi, lz.m_nucleus_gev(A))
    return mu * (v / C) ** 2 / 2 * 1e6

def ermax_kev(A, m_chi=M_CHI, v=V_MAX, delta=0.0):
    return lz.E_R_range_keV(m_chi, v, A, delta)[1]

def vmin_any_E_kms(A, delta, m_chi=M_CHI):
    """Smallest speed allowing an endothermic scatter with splitting delta on nucleus A (at E_R = E*)."""
    mu = lz.mu_red(m_chi, lz.m_nucleus_gev(A))
    return math.sqrt(2 * delta * 1e-6 / mu) * C

def A_min(delta, m_chi=M_CHI, v=V_MAX):
    for A in range(1, 400):
        if ceiling_kev(A, m_chi, v) >= delta:
            return A
    return None

rows = []
NUCLEI = {'1H': 1, '12C': 12, '13C': 13, '16O': 16, '19F': 19, '127I': 127, 'Xe(131.3)': lz.A_XE_MEAN, '184W': 184}
for name, A in NUCLEI.items():
    for m in (400.0, M_CHI, 4000.0):
        r = dict(nucleus=name, A=A, m_chi_GeV=m, ceiling_keV=ceiling_kev(A, m), ER_max_elastic_keV=ermax_kev(A, m),
                 delta_max_at_ERmax_over_4=None)
        for d in DELTAS:
            r[f'vmin_kms_delta{int(d)}'] = vmin_any_E_kms(A, d, m)
            r[f'ERrange_delta{int(d)}'] = lz.E_R_range_keV(m, V_MAX, A, d)
        rows.append(r)
KIN = pd.DataFrame(rows)
KIN.to_csv(f'{OUT}/P096_kinematics.csv', index=False)
for name in ('1H', '12C', '13C', '16O', 'Xe(131.3)'):
    r = KIN[(KIN.nucleus == name) & (KIN.m_chi_GeV == M_CHI)].iloc[0]
    say(f'{name:9s} 1 TeV: ceiling {r.ceiling_keV:7.1f} keV, E_R,max(elastic) {r.ER_max_elastic_keV:7.1f} keV, '
        f'v_min(delta=300) {r.vmin_kms_delta300:6.0f} km/s, (380) {r.vmin_kms_delta380:6.0f} km/s')
AMIN = {int(d): {m: A_min(d, m) for m in (400.0, M_CHI, 4000.0)} for d in DELTAS}
say('A_min(delta) for m = 400/1000/4000 GeV:', AMIN)
# ceiling curve for the figure
A_SCAN = np.arange(1, 201)
CEIL = {m: np.array([ceiling_kev(A, m) for A in A_SCAN]) for m in (400.0, M_CHI, 4000.0)}
ERMAX = np.array([ermax_kev(A, M_CHI) for A in A_SCAN])

# =====================================================================================================
# Halo: June (day 167) for elastic rates; 24-day annual mean for the endothermic normalisation
# =====================================================================================================
VGRID = np.linspace(0.0, 844.0, 1200)
ONES = np.ones_like(VGRID)
def halo(day):
    return lz.wd_halo(day_of_year=day, vmin=VGRID)
HALO_JUNE = halo(DOY_EVENT)
DAYS = np.linspace(0.5, 364.5, 24)
HALO_DAYS = [halo(d) for d in DAYS]
DETA_ANN = np.mean([h[1] for h in HALO_DAYS], axis=0)
assert np.allclose(HALO_JUNE[0], VGRID)
say(f'halos built ({time.time()-T0:.0f} s)')

def eff_roi(E):
    """LZ WS efficiency model (P003/P021): 0.96 with erf edges at 5.4 keV (sigma 2.5) and 269.9 keV (sigma 11.5)."""
    return 0.96 * stats.norm.cdf((E - 5.4) / 2.5) * (1 - stats.norm.cdf((E - 269.9) / 11.5))

# =====================================================================================================
# Hamiltonians (WimPyDD convention c^0 = c_p + c_n; LZ/Anand unit isoscalar coupling <-> c^0 = 2/m_v^2, P003)
# =====================================================================================================
H_O1 = lz.wd_hamiltonian('P096_O1_iso', {1: (2.0 / MV ** 2, 0.0)})
c4_wd, c6_wd = 8.0 / MV ** 2, -8.0 / MV ** 2          # P012: L10 with d10 = 1, m_M = m_N
H_L10 = WD.eft_hamiltonian('P096_L10_dipole_dipole', {(4, 'q2'): lambda q, A=c4_wd: [A * q ** 2 / MN ** 2, 0.0],
                                                      6: lambda A6=c6_wd: [A6, 0.0]})
H_O6 = lz.wd_hamiltonian('P096_O6_iso', {6: (2.0 / MV ** 2, 0.0)})
SIGMA_N_REF = 1e-47   # cm^2, SI reference (recalled: LZ 2024 limit at 1 TeV ~1e-47 cm^2; likely)
H_SI = lz.wd_hamiltonian('P096_SI_ref', {1: lz.wd_c_SI_from_sigma_n(SIGMA_N_REF, M_CHI)})

def rate(ham, target, E, delta=0.0, deta=None, m=M_CHI):
    """events / (t yr keV) per tonne of the element, natural abundance, halo (VGRID, deta)."""
    d = HALO_JUNE[1] if deta is None else deta
    return np.clip(np.atleast_1d(lz.wd_rate(ham, m, E, halo=(VGRID, d), delta_kev=delta, target=target)), 0.0, None)

def kernel(ham, target, E, delta=0.0, m=M_CHI):
    """K[E_i, v_j] in events/(t yr keV) per unit delta_eta: rate(day) = K @ delta_eta(day)."""
    K = np.array([WD.diff_rate(target, ham, m, float(e), VGRID, ONES, j_chi=0.5, delta=delta, sum_over_streams=False)
                  for e in E]) * 1000.0 * 365.25
    return np.clip(K, 0.0, None)

# =====================================================================================================
# Part 2a: L10 and O6 normalised to LZ's 1.0 event (June halo; Sun-frame check)
# =====================================================================================================
E_XE = np.arange(1.0, 301.0, 2.0)
CACHE = f'{OUT}/P096_cache.npz'
cache = {} if (RECOMPUTE or not os.path.exists(CACHE)) else dict(np.load(CACHE, allow_pickle=True))
def cached(key, fn):
    if key not in cache:
        cache[key] = fn()
        np.savez(CACHE, **cache)
    return cache[key]

R_L10_XE = cached('L10_Xe_june', lambda: rate(H_L10, WD.Xe, E_XE))
R_O6_XE = cached('O6_Xe_june', lambda: rate(H_O6, WD.Xe, E_XE))
N_LZ = {}
for nm, R in (('L10', R_L10_XE), ('O6', R_O6_XE)):
    N_LZ[nm] = EXPO_LZ * np.trapezoid(R * eff_roi(E_XE), E_XE)
    hi = EXPO_LZ * np.trapezoid(R * eff_roi(E_XE) * ((E_XE >= 200) & (E_XE <= 270)), E_XE)
    say(f'{nm}: LZ accepted events at unit coupling (2.84 t yr, June halo) = {N_LZ[nm]:.3f} (200-270 keV: {hi:.3f}); '
        f'scale to 1.0 event: {1/N_LZ[nm]:.4f}  [P012 (Sun frame, all E): N(d10=1) ours = 3.34]')
SUN = lz.wd_halo(vmin=VGRID)
R_L10_XE_SUN = cached('L10_Xe_sun', lambda: rate(H_L10, WD.Xe, E_XE, deta=SUN[1]))
say(f'L10 Sun-frame check: N(unit) = {EXPO_LZ*np.trapezoid(R_L10_XE_SUN*eff_roi(E_XE), E_XE):.3f}  ({time.time()-T0:.0f} s)')
SCALE = {nm: 1.0 / N_LZ[nm] for nm in N_LZ}     # multiply unit-coupling rates by this for LZ's best fit (1.0 event)

# ---- hydrogen: WimPyDD 1H (itar 21) -----------------------------------------------------------------------------
E_H = np.arange(0.1, 16.0, 0.1)
R_L10_H = cached('L10_H_june', lambda: rate(H_L10, WD.H, E_H))
R_O6_H = cached('O6_H_june', lambda: rate(H_O6, WD.H, E_H))

# ---- analytic free-nucleon check ------------------------------------------------------------------------------
def dRdE_pointspin(E_keV, m_T, n_T_per_t, c_p, amp2_factor, S2_factor=1.0, F2=lambda q: 1.0, v_e=V_E_JUNE, m_chi=M_CHI):
    """dR/dE [events/(t yr keV)] for a J=1/2 target with spin-averaged |M|^2 = amp2_factor * c_p^2 (q/m_N)^4 * S2_factor * F2(q)
    (amp2_factor = 1/16 for O6 alone, 2 for L10 = 4[(q^2/m_N^2)O4 - O6]; S2_factor = (2<S>)^2 relative to a free nucleon),
    with dsigma/dE = (2 m_T / 4 pi v^2) |M|^2_avg (Anand et al. convention) and rho = 0.3 GeV/cm^3."""
    mu = lz.mu_red(m_chi, m_T)
    out = []
    for E in np.atleast_1d(E_keV):
        q2 = 2 * m_T * E * 1e-6
        vmin = (m_T * E * 1e-6 / mu) / math.sqrt(q2) * C
        eta = lz.eta0(vmin, v_e=v_e)
        X = (2 * m_T / (4 * math.pi)) * amp2_factor * c_p ** 2 * (q2 / MN ** 2) ** 2 * S2_factor * F2(math.sqrt(q2))   # GeV^-3
        dsig = X * lz.GEV_TO_CM2 * 1e-6                       # cm^2 / keV  (times 1/v^2)
        out.append(n_T_per_t * (lz.RHO0_GEV_CM3 / m_chi) * dsig * eta * C ** 2 * 1e5 * 3.15576e7)
    return np.array(out)

M_H = float(WD.H.mass[0]); NT_H = float(WD.H.nt_kg[0]) * 1000.0
A_O6 = dRdE_pointspin(E_H, M_H, NT_H, 1.0 / MV ** 2, 1.0 / 16.0)
A_L10 = dRdE_pointspin(E_H, M_H, NT_H, 1.0 / MV ** 2, 2.0)
sel = (E_H > 0.5) & (R_O6_H > 0)
say('analytic/WimPyDD on 1H: O6 ratio mean %.4f (rms %.4f); L10 ratio mean %.4f (rms %.4f); WimPyDD L10/O6 at 4 keV = %.2f (analytic 32)'
    % (np.mean(A_O6[sel] / R_O6_H[sel]), np.std(A_O6[sel] / R_O6_H[sel]), np.mean(A_L10[sel] / R_L10_H[sel]),
       np.std(A_L10[sel] / R_L10_H[sel]), R_L10_H[np.argmin(abs(E_H - 4))] / R_O6_H[np.argmin(abs(E_H - 4))]))

# ---- 13C: single-particle estimate (WimPyDD itar = 0 -> all responses vanish) ---------------------------------
S_N_13C = -1.0 / 6.0        # <S_n> for a p1/2 neutron (j = l - 1/2): -j/(2(j+1)) = -1/6 (recalled shell-model value ~ -0.17, likely)
S2_13C = (2 * S_N_13C) ** 2  # = 1/9 relative to the free proton (isoscalar coupling, J = 1/2 both)
B_HO = 1.64                  # fm, harmonic-oscillator length for A ~ 12-13 (recalled/likely)
def F2_p_shell(q_gev):
    x = (q_gev / lz.HBARC_GEV_FM * B_HO) ** 2
    return ((1 - x / 6.0) * math.exp(-x / 4.0)) ** 2
M_13C = 13 * AMU
E_13C = np.arange(0.5, 200.0, 0.5)
A_L10_13C = dRdE_pointspin(E_13C, M_13C, 1e6 / (13.003 * 1.66054e-24) * 1.0, 1.0 / MV ** 2, 2.0, S2_13C, F2_p_shell)   # per tonne of 13C
A_O6_13C = dRdE_pointspin(E_13C, M_13C, 1e6 / (13.003 * 1.66054e-24) * 1.0, 1.0 / MV ** 2, 1.0 / 16.0, S2_13C, F2_p_shell)
F13 = 0.0107 * 13.003 / (0.9893 * 12.0 + 0.0107 * 13.003)     # mass fraction of 13C in natural carbon (abundance 1.07 %, recalled/certain)
say(f'13C mass fraction of natural carbon {F13:.4f}; spin factor (2<S_n>)^2 = {S2_13C:.4f}; F^2(q) at E_R,max = {F2_p_shell(math.sqrt(2*M_13C*ermax_kev(13)*1e-6)):.3f}')

# ---- SI reference (O1, sigma_n = 1e-47 cm^2) on C, H, O (elastic, June) ------------------------------------------
E_SI = {'C': np.arange(0.5, 200.0, 1.0), 'H': np.arange(0.1, 16.0, 0.1), 'O': np.arange(0.5, 260.0, 1.0)}
R_SI = {k: cached(f'SI_{k}_june', lambda k=k: rate(H_SI, {'C': WD.C, 'H': WD.H, 'O': WD.O}[k], E_SI[k])) for k in E_SI}
say(f'SI reference totals per tonne of element per yr (sigma_n=1e-47, 1 TeV): ' +
    ', '.join(f'{k} {np.trapezoid(R_SI[k], E_SI[k]):.3e}' for k in R_SI) + f'  ({time.time()-T0:.0f} s)')

# =====================================================================================================
# Part 3: exothermic down-scattering kernels
# =====================================================================================================
E_ENDO = np.arange(100.0, 452.0, 2.5)
E_EXO = {'C': np.arange(2.5, 800.0, 5.0), 'H': np.arange(150.0, 452.0, 2.0), 'O': np.arange(2.5, 850.0, 5.0)}
TGT = {'C': WD.C, 'H': WD.H, 'O': WD.O}
K_ENDO, K_EXO = {}, {}
for d in DELTAS:
    K_ENDO[d] = cached(f'Kendo_Xe_{int(d)}', lambda d=d: kernel(H_O1, WD.Xe, E_ENDO, delta=+d))
    say(f'endothermic Xe kernel delta={d:.0f} done ({time.time()-T0:.0f} s)')
    for k in TGT:
        K_EXO[(k, d)] = cached(f'Kexo_{k}_{int(d)}', lambda k=k, d=d: kernel(H_O1, TGT[k], E_EXO[k], delta=-d))
    say(f'exothermic C/H/O kernels delta={d:.0f} done ({time.time()-T0:.0f} s)')
E_EXO_XE = np.arange(2.5, 800.0, 5.0)
K_EXO_XE300 = cached('Kexo_Xe_300', lambda: kernel(H_O1, WD.Xe, E_EXO_XE, delta=-300.0))
say(f'exothermic Xe kernel (300, validation) done ({time.time()-T0:.0f} s)')

def counts_endo(d, deta):
    R = K_ENDO[d] @ deta
    return EXPO_LZ * np.trapezoid(R * eff_roi(E_ENDO), E_ENDO)

EXO = []     # per-target exothermic spectra (unit coupling), annual mean
KAPPA, N_ENDO = {}, {}
for d in DELTAS:
    n_days = np.array([counts_endo(d, h[1]) for h in HALO_DAYS])
    N_ENDO[d] = float(n_days.mean()); KAPPA[d] = 1.0 / N_ENDO[d]
    say(f'delta={d:.0f}: N_endo,ROI(unit, annual 24-day) = {N_ENDO[d]:.4g} (June {counts_endo(d, HALO_JUNE[1]):.4g}, '
        f'Dec {counts_endo(d, halo(350)[1]):.4g}); kappa = 1/N = {KAPPA[d]:.4g}  [P021 annual: 1.35e4 / 258.7 / - / 0.456 at 300/350/380]')
    for k in TGT:
        Rann = K_EXO[(k, d)] @ DETA_ANN; Rjun = K_EXO[(k, d)] @ HALO_JUNE[1]
        E = E_EXO[k]; tot = np.trapezoid(Rann, E)
        cdf = integrate.cumulative_trapezoid(Rann, E, initial=0) / tot
        q16, q50, q84 = np.interp([0.16, 0.5, 0.84], cdf, E)
        mu = lz.mu_red(M_CHI, lz.m_nucleus_gev({'C': 12, 'H': 1, 'O': 16}[k]))
        Estar = d * mu / lz.m_nucleus_gev({'C': 12, 'H': 1, 'O': 16}[k])
        EXO.append(dict(target=k, delta_keV=d, Estar_keV=Estar, N_unit_per_t_yr_annual=tot, N_unit_per_t_yr_june=np.trapezoid(Rjun, E),
                        E_peak_keV=E[np.argmax(Rann)], E16=q16, E50=q50, E84=q84, kappa=KAPPA[d],
                        N_per_t_yr_f2eq1=KAPPA[d] * tot, Eplus_vmax=lz.E_R_range_keV(M_CHI, V_MAX, {'C': 12, 'H': 1, 'O': 16}[k], -d)[1]))
EXO = pd.DataFrame(EXO); EXO.to_csv(f'{OUT}/P096_exothermic_spectra.csv', index=False)
for _, r in EXO.iterrows():
    say(f"  exo {r.target} delta={r.delta_keV:.0f}: E*={r.Estar_keV:.0f} keV, peak {r.E_peak_keV:.0f}, 16/50/84% = {r.E16:.0f}/{r.E50:.0f}/{r.E84:.0f} keV, "
        f"E+(v_max)={r.Eplus_vmax:.0f}; N(unit)={r.N_unit_per_t_yr_annual:.3e} /t/yr (June x{r.N_unit_per_t_yr_june/r.N_unit_per_t_yr_annual:.3f}); "
        f"kappa*N = {r.N_per_t_yr_f2eq1:.3e} /t/yr at f2=1")
# validation vs P058: R_ROI (Xe exothermic ROI events per endothermic ROI event, delta = 300, iso, annual)
R_exo_xe = K_EXO_XE300 @ DETA_ANN
N_exo_roi_xe = EXPO_LZ * np.trapezoid(R_exo_xe * eff_roi(E_EXO_XE), E_EXO_XE)
say(f'validation: Xe exothermic ROI count per unit (annual) {N_exo_roi_xe:.4g}; R_ROI = {N_exo_roi_xe/N_ENDO[300.0]:.1f}  [P058: 916]')

# P058 joint 90% CL f2 limits (isoscalar, 1 TeV, mu_endo = 1)
F2_P058 = {300.0: 2.8e-3, 350.0: 5.7e-5, 366.0: 4.8e-6, 380.0: 1.7e-7}

# =====================================================================================================
# Part 4: detector response -- quenching, 14C, thresholds
# =====================================================================================================
# Birks quenching: dL/dE = 1 / (1 + kB rho S_e(E)), Q(E) = L(E)/E; kB = 0.0098 cm/MeV for LAB (recalled/likely),
# rho = 0.86 g/cm^3 (LAB; PC 0.88).  Proton electronic stopping power in a light hydrocarbon/water (MeV cm^2/g),
# recalled from PSTAR-like tables (likely, +-20 %): maximum ~800 near 80 keV, 260 at 1 MeV.
KB_CM_PER_MEV = 0.0098; RHO_LAB = 0.86
_SP_E = np.array([1, 3, 10, 30, 80, 100, 200, 300, 500, 1000, 3000, 10000.0])          # keV
_SP_P = np.array([170, 270, 430, 640, 810, 790, 680, 580, 420, 262, 116, 45.7])          # MeV cm^2/g
def S_e_proton(E_keV):
    return np.exp(np.interp(np.log(np.clip(E_keV, 1.0, 1e4)), np.log(_SP_E), np.log(_SP_P)))
def S_e_carbon(E_keV):
    """carbon-ion electronic stopping in hydrocarbon, LSS regime S_e ~ v: 2500 (E/300 keV)^0.5 MeV cm^2/g (recalled/uncertain, x/ 1.5)."""
    return 2500.0 * np.sqrt(np.clip(E_keV, 1.0, None) / 300.0)
F_E_CARBON = 0.6     # electronic fraction of the carbon-ion energy loss at 0.1-0.5 MeV (nuclear stopping gives no light; recalled/uncertain)
def Q_birks(E_keV, S_func, f_e=1.0, kB=KB_CM_PER_MEV, rho=RHO_LAB, n=400):
    """quenching factor Q = L/E for an ion of energy E_keV (scalar or array)."""
    out = []
    for E in np.atleast_1d(np.asarray(E_keV, float)):
        Eg = np.linspace(0.0, E, n)
        dLdE = f_e / (1 + kB * rho * S_func(np.clip(Eg, 1e-3, None)))
        out.append(np.trapezoid(dLdE, Eg) / E if E > 0 else 0.0)
    out = np.array(out)
    return out if out.size > 1 else float(out[0])
def L_cecil_MeVee(E_MeV):
    """cross-check: NE-213 proton light output, Cecil-Anderson-Madey form L = 0.83E - 2.82[1 - exp(-0.25 E^0.93)] (recalled/likely; valid above ~0.3 MeV)."""
    return 0.83 * E_MeV - 2.82 * (1 - math.exp(-0.25 * E_MeV ** 0.93))
Q_P = {E: Q_birks(E, S_e_proton) for E in (4.6, 6.0, 14.0, 100.0, 300.0, 1000.0)}
Q_P_CECIL = {E: L_cecil_MeVee(E / 1000) / (E / 1000) for E in (300.0, 500.0, 1000.0)}
Q_C_TAB = {E: Q_birks(E, S_e_carbon, F_E_CARBON) for E in (57.0, 100.0, 300.0, 380.0)}
Q_C_UNC = (0.6, 1.7)          # multiplicative uncertainty band on Q_C (recalled literature values 0.02-0.06 at 0.1-0.5 MeV; uncertain)
Q_C = float(Q_C_TAB[300.0]); Q_C_RANGE = (Q_C * Q_C_UNC[0], Q_C * Q_C_UNC[1])
Q_O = Q_C * 0.9                # oxygen ion (only relevant for SNO+ water, where a 300 keV ion gives no Cherenkov light anyway)
say('Birks proton quenching Q_p(E) =', {k: round(v, 3) for k, v in Q_P.items()}, '| Cecil cross-check', {k: round(v, 3) for k, v in Q_P_CECIL.items()})
say('Birks-LSS carbon quenching Q_C(E) =', {k: round(v, 4) for k, v in Q_C_TAB.items()}, '; adopted range at 300 keV', tuple(round(x, 3) for x in Q_C_RANGE))

# 14C beta spectrum (Q = 156.5 keV, allowed, Fermi function for Z = 7)
Q14 = 156.5; ME = 510.999
def c14_pdf(T):
    """allowed beta spectrum dN/dT (keV^-1), Fermi function in the non-relativistic Coulomb approximation."""
    T = np.atleast_1d(np.asarray(T, float)); out = np.zeros_like(T)
    m = (T > 0) & (T < Q14)
    W = 1 + T[m] / ME; p = np.sqrt(W ** 2 - 1); eta = 7 / 137.036 * W / p
    F = 2 * np.pi * eta / (1 - np.exp(-2 * np.pi * eta))
    out[m] = F * p * W * (Q14 - T[m]) ** 2
    return out
T_GRID = np.linspace(0.0, Q14, 2001)
c14_pdf_raw = c14_pdf(T_GRID)
NORM14 = np.trapezoid(c14_pdf_raw, T_GRID)
def c14_frac(lo, hi):
    m = (T_GRID >= lo) & (T_GRID <= hi)
    return np.trapezoid(c14_pdf_raw[m], T_GRID[m]) / NORM14
LAMBDA14 = math.log(2) / (5730.0 * 3.15576e7)   # s^-1 (half-life 5730 yr, recalled/certain)
def c14_rate_per_yr(mass_C_t, ratio):
    n_C = mass_C_t * 1e6 / 12.011 * 6.02214e23
    return n_C * ratio * LAMBDA14 * 3.15576e7
say(f'14C fraction of decays in 5-15 / 10-30 / 20-40 / 50-156 keVee: {c14_frac(5,15):.3f} / {c14_frac(10,30):.3f} / {c14_frac(20,40):.3f} / {c14_frac(50,Q14):.3f}')

# ---- assemble the model x detector table -----------------------------------------------------------------------
TABLE = []
def add(model, det, target, N_per_yr, ER_typ, Eee_lo, Eee_hi, note=''):
    D = DETECTORS[det]
    win_lo, win_hi = Eee_lo, Eee_hi
    B = 0.0
    if D['c14'] > 0 and det != 'SNO+ water':
        B = c14_rate_per_yr(D['mass_t'] * D['frac'].get('C', 0.0), D['c14']) * c14_frac(win_lo, win_hi)
    passes = Eee_hi >= D['thr_keVee']
    TABLE.append(dict(model=model, detector=det, target=target, events_per_yr=N_per_yr, E_R_typ_keV=ER_typ,
                      E_ee_lo=Eee_lo, E_ee_hi=Eee_hi, threshold_keVee=D['thr_keVee'], threshold_pass=passes,
                      B14C_per_yr_in_window=B, S_over_B=(N_per_yr / B if B > 0 else float('inf')),
                      S_over_sqrtB_1yr=(N_per_yr / math.sqrt(B) if B > 0 else float('inf')), note=note))

# (A) inelastic endothermic: forbidden on every scintillator nucleus
for det in DETECTORS:
    for d in DELTAS:
        add(f'A: inelastic O1 delta={d:.0f}', det, 'C/H/O', 0.0, float('nan'), 0.0, 0.0,
            note=f'kinematically forbidden: A_min = {AMIN[int(d)][M_CHI]}, ceiling(12C) = {ceiling_kev(12):.0f} keV')

# (B) elastic L10 / O6 at LZ's best fit on 1H and 13C
for nm, RH, R13 in (('L10', R_L10_H, A_L10_13C), ('O6', R_O6_H, A_O6_13C)):
    nH = SCALE[nm] * np.trapezoid(RH, E_H)            # per tonne of H per yr
    n13 = SCALE[nm] * np.trapezoid(R13, E_13C)        # per tonne of 13C per yr
    cdfH = integrate.cumulative_trapezoid(RH, E_H, initial=0) / np.trapezoid(RH, E_H)
    cdf13 = integrate.cumulative_trapezoid(R13, E_13C, initial=0) / np.trapezoid(R13, E_13C)
    EH50 = np.interp(0.5, cdfH, E_H); E1350 = np.interp(0.5, cdf13, E_13C)
    say(f'{nm} at LZ best fit: {nH:.3e} events per tonne of H per yr (median E_R {EH50:.1f} keV, E_R,max {E_H[RH>0].max():.1f}); '
        f'{n13:.3e} per tonne of 13C per yr (median {E1350:.0f} keV)')
    EH16, EH84 = np.interp([0.16, 0.84], cdfH, E_H); E1316, E1384 = np.interp([0.16, 0.84], cdf13, E_13C)
    for det, D in DETECTORS.items():
        mH = D['mass_t'] * D['frac'].get('H', 0.0); mC = D['mass_t'] * D['frac'].get('C', 0.0)
        NH = nH * mH; N13 = n13 * mC * F13
        QpH = Q_birks(EH50, S_e_proton)
        add(f'B: {nm} (LZ fit) on 1H', det, '1H', NH, EH50, QpH * EH16, QpH * EH84,
            note=f'per kt yr: {nH*D["frac"].get("H",0)*1000:.2e}; proton quenching Q~{QpH:.2f} (electronic stopping only; nuclear stopping lowers it further)')
        if mC > 0:
            Q13 = Q_birks(E1350, S_e_carbon, F_E_CARBON)
            add(f'B: {nm} (LZ fit) on 13C', det, '13C', N13, E1350, Q13 * E1316, Q13 * E1384,
                note=f'per kt yr: {n13*D["frac"]["C"]*F13*1000:.2e}; single-particle <S_n> = -1/6; Q_C~{Q13:.3f} (x0.6-1.7)')

# (C) SI reference
for det, D in DETECTORS.items():
    for k in ('C', 'H', 'O'):
        mk = D['mass_t'] * D['frac'].get(k, 0.0)
        if mk <= 0: continue
        R = R_SI[k]; E = E_SI[k]; tot = np.trapezoid(R, E)
        cdf = integrate.cumulative_trapezoid(R, E, initial=0) / tot; E16, E50, E84 = np.interp([0.16, 0.5, 0.84], cdf, E)
        Q = Q_birks(E50, S_e_proton) if k == 'H' else (Q_birks(E50, S_e_carbon, F_E_CARBON) if k == 'C' else Q_O)
        add(f'C: SI sigma_n=1e-47 on {k}', det, k, tot * mk, E50, Q * E16, Q * E84, note=f'per kt yr: {tot*D["frac"][k]*1000:.2e}; Q={Q:.3f}')

# (D) exothermic down-scattering at P058's f2 limits (and f2 = 1)
for det, D in DETECTORS.items():
    for d in DELTAS:
        for k in ('C', 'H', 'O'):
            mk = D['mass_t'] * D['frac'].get(k, 0.0)
            if mk <= 0: continue
            r = EXO[(EXO.target == k) & (EXO.delta_keV == d)].iloc[0]
            N1 = r.N_per_t_yr_f2eq1 * mk                     # f2 = 1
            Nf = N1 * F2_P058[d]
            if k == 'H':
                Q = Q_birks(r.E50, S_e_proton)
            else:
                Q = Q_birks(r.E50, S_e_carbon, F_E_CARBON) if k == 'C' else Q_O
            lo, hi = Q * r.E16, Q * r.E84
            add(f'D: exothermic delta={d:.0f} on {k} (f2 = P058 limit {F2_P058[d]:.1e})', det, k, Nf, r.E50, lo, hi,
                note=f'f2=1: {N1:.3e}/yr; per kt yr at f2=1: {r.N_per_t_yr_f2eq1*D["frac"][k]*1000:.3e}; Q={Q:.3f}'
                     + (f' (range {Q*Q_C_UNC[0]:.3f}-{Q*Q_C_UNC[1]:.3f} -> {Q*Q_C_UNC[0]*r.E50:.0f}-{Q*Q_C_UNC[1]*r.E50:.0f} keVee)' if k == 'C' else
                        f' (Cecil cross-check {L_cecil_MeVee(r.E50/1000)/(r.E50/1000):.3f})'))
TAB = pd.DataFrame(TABLE); TAB.to_csv(f'{OUT}/P096_model_detector_table.csv', index=False)

# ---- exothermic reach: f2 sensitivity (stat only, 90% CL, 1 yr) with and without ER/NR pulse-shape rejection ---
REACH = []
EPS_SYS = 1e-3     # assumed relative knowledge of the 14C spectrum shape/rate in the signal window (systematic floor; assumption)
for det in ('Borexino', 'KamLAND', 'SNO+', 'JUNO'):
    D = DETECTORS[det]
    for d in DELTAS:
        rows_ = TAB[(TAB.detector == det) & TAB.model.str.startswith(f'D: exothermic delta={d:.0f}')]
        rC = rows_[rows_.target == 'C'].iloc[0]; rH = rows_[rows_.target == 'H'].iloc[0]
        S1_C = rC.events_per_yr / F2_P058[d]; S1_H = rH.events_per_yr / F2_P058[d]       # events/yr at f2 = 1
        for rej in (1.0, 1e-2, 1e-4):
            BC = rC.B14C_per_yr_in_window * rej; BH = rH.B14C_per_yr_in_window * rej
            f2_C = 1.28 * math.sqrt(BC) / S1_C if BC > 0 else 2.3 / S1_C
            f2_H = 1.28 * math.sqrt(BH) / S1_H if BH > 0 else 2.3 / S1_H
            REACH.append(dict(detector=det, delta_keV=d, ER_rejection=rej, f2_90_C_1yr=f2_C, f2_90_H_1yr=f2_H,
                              f2_sys_C_eps1e3=EPS_SYS * BC / S1_C, f2_sys_H_eps1e3=EPS_SYS * BH / S1_H,
                              f2_P058=F2_P058[d], ratio_C_over_LZ=f2_C / F2_P058[d], ratio_H_over_LZ=f2_H / F2_P058[d],
                              S_C_f2eq1=S1_C, B_C_window=rC.B14C_per_yr_in_window, S_H_f2eq1=S1_H, B_H_window=rH.B14C_per_yr_in_window,
                              rej_needed_to_match_LZ_C=(S1_C * F2_P058[d] / 1.28) ** 2 / rC.B14C_per_yr_in_window,
                              rej_needed_to_match_LZ_H=(S1_H * F2_P058[d] / 1.28) ** 2 / rH.B14C_per_yr_in_window))
REACH = pd.DataFrame(REACH); REACH.to_csv(f'{OUT}/P096_exothermic_reach.csv', index=False)
for _, r in REACH[REACH.ER_rejection == 1.0].iterrows():
    say(f"reach {r.detector:8s} delta={r.delta_keV:.0f}: S(f2=1) C {r.S_C_f2eq1:.3e}/yr, 14C in C window {r.B_C_window:.3e}/yr -> f2_90(stat) = {r.f2_90_C_1yr:.2e} "
        f"= {r.ratio_C_over_LZ:.2g} x P058; sys floor (1e-3) {r.f2_sys_C_eps1e3:.2e} = {r.f2_sys_C_eps1e3/r.f2_P058:.2g} x P058; H line: S {r.S_H_f2eq1:.3e}, B {r.B_H_window:.3e}, "
        f"f2_90 = {r.f2_90_H_1yr:.2e} ({r.ratio_H_over_LZ:.2g} x P058); rejection needed to match LZ: C {r.rej_needed_to_match_LZ_C:.1e}, H {r.rej_needed_to_match_LZ_H:.1e}")

# =====================================================================================================
# Figures
# =====================================================================================================
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for m, ls in ((400.0, ':'), (M_CHI, '-'), (4000.0, '--')):
    ax[0].plot(A_SCAN, CEIL[m], 'k', ls=ls, lw=1.2, label=f'inelastic ceiling μv²_max/2, {m:.0f} GeV')
ax[0].plot(A_SCAN, ERMAX, color='C0', lw=1.2, label='elastic E_R,max (1 TeV)')
for i, (d, c) in enumerate(zip(DELTAS, ('C3', 'C1', 'C2', 'C4'))):
    ax[0].axhline(d, color=c, lw=0.8, alpha=0.7)
    ax[0].text(1.05, 1500 / 1.45 ** i, f'δ = {d:.0f} keV: A ≥ {AMIN[int(d)][M_CHI]}', color=c, fontsize=7)
for name, A, dx, dy in (('¹H', 1, 1.1, 0.7), ('¹²C', 12, 1.1, 0.65), ('¹⁶O', 16, 1.1, 0.65), ('¹⁹F', 19, 1.15, 1.2), ('¹²⁷I', 127, 0.55, 1.35),
                        ('Xe', 131, 0.85, 0.6), ('¹⁸⁴W', 184, 0.72, 1.3)):
    ax[0].plot([A], [ceiling_kev(A)], 'o', ms=4, color='C3' if A <= 16 else 'gray'); ax[0].text(A * dx, ceiling_kev(A) * dy, name, fontsize=8)
ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_xlabel('mass number A'); ax[0].set_ylabel('keV')
ax[0].set_ylim(1, 3000); ax[0].set_title('Kinematic ceilings at v_max = %.0f km/s (16 June)' % V_MAX, fontsize=9); ax[0].legend(fontsize=7, loc='lower right')
# right: exothermic spectra on C and H (JUNO, 1 yr, f2 = P058 limit) vs 14C, in keV_ee
for d, c in zip((300.0, 380.0), ('C3', 'C4')):
    rC = EXO[(EXO.target == 'C') & (EXO.delta_keV == d)].iloc[0]; rH = EXO[(EXO.target == 'H') & (EXO.delta_keV == d)].iloc[0]
    mC = DETECTORS['JUNO']['mass_t'] * F_LAB['C']; mH = DETECTORS['JUNO']['mass_t'] * F_LAB['H']
    RC = (K_EXO[('C', d)] @ DETA_ANN) * KAPPA[d] * F2_P058[d] * mC; RH = (K_EXO[('H', d)] @ DETA_ANN) * KAPPA[d] * F2_P058[d] * mH
    EeC = E_EXO['C'] * Q_C; ax[1].plot(EeC, RC / Q_C, color=c, lw=1.4, label=f'¹²C recoils, δ = {d:.0f} keV, f₂ = {F2_P058[d]:.1e}')
    QH = Q_birks(E_EXO['H'], S_e_proton); EeH = E_EXO['H'] * QH
    ax[1].plot(EeH, RH / np.gradient(EeH, E_EXO['H']), color=c, lw=1.4, ls='--', label=f'¹H recoils, δ = {d:.0f} keV')
B14 = c14_rate_per_yr(DETECTORS['JUNO']['mass_t'] * F_LAB['C'], DETECTORS['JUNO']['c14'])
ax[1].plot(T_GRID, B14 * c14_pdf_raw / NORM14, 'k', lw=1.2, label='¹⁴C β (¹⁴C/¹²C = 10⁻¹⁷)')
ax[1].axvline(DETECTORS['JUNO']['thr_keVee'], color='gray', ls=':', lw=1); ax[1].text(DETECTORS['JUNO']['thr_keVee'] * 1.05, 1e3, 'JUNO threshold\n(~0.1 MeV, recalled)', fontsize=7)
ax[1].set_yscale('log'); ax[1].set_xlim(0, 160); ax[1].set_ylim(1, 1e12); ax[1].set_xlabel('electron-equivalent energy (keV$_{ee}$)')
ax[1].set_ylabel('events / (keV$_{ee}$ · 20 kt · yr)'); ax[1].set_title('JUNO: exothermic χ₂ down-scattering at LZ\'s f₂ limit vs ¹⁴C', fontsize=9)
ax[1].legend(fontsize=7, loc='upper right')
plt.tight_layout(); plt.savefig(f'{FIG}/P096_fig1_ceilings_and_exothermic.png', dpi=160); plt.close()

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(E_H, SCALE['L10'] * R_L10_H * 1000, label='L10 on ¹H (WimPyDD, LZ fit)', color='C0')
ax.plot(E_H, SCALE['L10'] * A_L10 * 1000, ':', color='C0', label='analytic free proton')
ax.plot(E_13C, SCALE['L10'] * A_L10_13C * 1000 * F13, color='C2', label='L10 on ¹³C per tonne of natural C (single-particle)')
ax.plot(E_H, SCALE['O6'] * R_O6_H * 1000, '--', color='C1', label='O6 on ¹H (LZ fit)')
ax.set_yscale('log'); ax.set_xscale('log'); ax.set_xlabel('recoil energy (keV)'); ax.set_ylabel('events / (kt · yr · keV)')
ax.set_title('Elastic q⁴-spin models at LZ\'s best fit on scintillator nuclei', fontsize=9); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f'{FIG}/P096_fig2_L10_light_nuclei.png', dpi=160); plt.close()

# =====================================================================================================
# Summary JSON
# =====================================================================================================
SUM = dict(v_E_june=V_E_JUNE, v_max=V_MAX, ceilings_1TeV={n: ceiling_kev(A) for n, A in NUCLEI.items()},
           ER_max_elastic_1TeV={n: ermax_kev(A) for n, A in NUCLEI.items()}, A_min=AMIN,
           vmin_delta300={n: vmin_any_E_kms(A, 300) for n, A in NUCLEI.items()},
           N_LZ_unit=N_LZ, scale_to_1_event=SCALE,
           L10_H_per_t_H_yr=float(SCALE['L10'] * np.trapezoid(R_L10_H, E_H)), O6_H_per_t_H_yr=float(SCALE['O6'] * np.trapezoid(R_O6_H, E_H)),
           L10_13C_per_t_13C_yr=float(SCALE['L10'] * np.trapezoid(A_L10_13C, E_13C)), F13_mass_fraction=F13,
           analytic_over_wimpydd_O6=float(np.mean(A_O6[sel] / R_O6_H[sel])), analytic_over_wimpydd_L10=float(np.mean(A_L10[sel] / R_L10_H[sel])),
           SI_ref_per_t_yr={k: float(np.trapezoid(R_SI[k], E_SI[k])) for k in R_SI},
           N_endo_unit_annual=N_ENDO, kappa=KAPPA, R_ROI_300_validation=float(N_exo_roi_xe / N_ENDO[300.0]),
           f2_P058=F2_P058, Q_p=Q_P, Q_C_range=Q_C_RANGE, c14_frac_10_30=c14_frac(10, 30),
           c14_rate_JUNO_per_yr=B14, detectors={k: dict(mass_t=v['mass_t'], frac=v['frac'], thr=v['thr_keVee'], c14=v['c14']) for k, v in DETECTORS.items()},
           runtime_s=time.time() - T0)
json.dump(SUM, open(f'{OUT}/P096_summary.json', 'w'), indent=1, default=float)
say(f'done in {time.time()-T0:.0f} s')
