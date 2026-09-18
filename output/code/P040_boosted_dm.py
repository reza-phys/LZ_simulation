"""
P040 -- Cosmic-ray-boosted and otherwise fast light dark matter at the extended LZ window.

Run from the simulation root:   .venv/bin/python output/code/P040_boosted_dm.py

Computes
  1. relativistic elastic kinematics chi + Xe -> chi + Xe: E_R,max(T, m_chi), T_min(248 keV);
  2. recoil-spectrum shapes for a relativistic beam (flat in E_R for a constant amplitude,
     plus vector-/scalar-/pseudoscalar-mediator amplitude structures) folded with the Helm
     (lzcommon.helm_F2) and shell-model (P017, WimPyDD) M form factors, and the lone-event
     statistic N_lo = R(5.4-55 keV)/R(200-270 keV) for monochromatic beams, power-law fluxes
     and the self-consistent CRDM flux;
  3. CRDM flux normalisation (Bringmann-Pospelov mechanism) from the PDG local cosmic-ray proton
     spectrum, the sigma_chi p that gives one LZ event in 200-270 keV in 2.84 t yr, the implied
     5.4-55 keV population, and the Earth-attenuation ceiling in 1.5 km of rock;
  4. a heavy (TeV) fast sub-population (beta = 0.005-0.02): E_max and N_lo.
Results -> output/work/P040/*.csv, *.json ; figures -> output/work/P040/figures/*.png
"""
from __future__ import annotations
import sys, json, math, os
sys.path.insert(0, 'output/code')
import numpy as np
from scipy import integrate, special
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import lzcommon as lz

OUT = 'output/work/P040'
FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)

# ----------------------------------------------------------------------------- constants
M_P = lz.M_NUCLEON_GEV                 # 0.938272 GeV
AMU = lz.AMU_GEV
ISO = lz.XE_ISOTOPES                   # A -> number abundance
A_MEAN = lz.A_XE_MEAN                  # 131.29
M_XE = A_MEAN * AMU                    # 122.3 GeV (natural-Xe mean nucleus, used for kinematics)
KEV = 1e-6                             # GeV per keV
KPC_CM = 3.0857e21
RHO_CHI = lz.RHO0_GEV_CM3              # 0.3 GeV/cm^3 (local; Baxter 2021)
LAMBDA2_DIPOLE = 0.71                  # GeV^2, proton dipole form factor G = (1+q^2/0.71)^-2 (recalled, likely)
E_LO, E_LO_MAX = 5.4, 55.0             # 2024 low-energy ROI (keV), LZ Data Analysis paragraph
E_HI, E_HI_MAX = 200.0, 270.0          # window used by P003 and later papers
E_EVENT = 248.0

# ----------------------------------------------------------------------------- efficiency (P003 model)
SIG_LO, SIG_HI, EFF0 = 3.4, 8.0, 0.96
def efficiency(E):
    E = np.asarray(E, dtype=float)
    s_lo = 0.5 * (1 + special.erf((E - lz.LZ['E_50pct_low_keV']) / (np.sqrt(2) * SIG_LO)))
    s_hi = 0.5 * (1 - special.erf((E - lz.LZ['E_50pct_high_keV']) / (np.sqrt(2) * SIG_HI)))
    return EFF0 * s_lo * s_hi

# ----------------------------------------------------------------------------- form factors on a common grid
P17 = np.load('output/work/P017/P017_response_curves.npz', allow_pickle=True)
E_GRID = P17['E_grid']                              # 1.0 ... 420 keV, 0.1 keV steps
F2_SHELL_NAT = P17['F2_shell']                      # natural-Xe shell-model M response / M(0)
F2_HELM_P17 = P17['F2_helm']
F2_SHELL_ISO = {A: P17[f'F2_shell_{A}'] for A in (128, 129, 130, 131, 132, 134, 136)}
F2_SHELL_ISO[124] = F2_SHELL_ISO[128]; F2_SHELL_ISO[126] = F2_SHELL_ISO[128]   # 0.2 % abundance, negligible
F2_HELM_ISO = {A: np.array([lz.helm_F2(e, A) for e in E_GRID]) for A in ISO}
# rate-weighted natural Helm F^2 (weights f_i A_i^2, appropriate for coherent SI)
w = np.array([ISO[A] * A**2 for A in ISO]); w /= w.sum()
F2_HELM_NAT = sum(wi * F2_HELM_ISO[A] for wi, A in zip(w, ISO))

# spin-dependent Sigma'' (isoscalar W00) for 129Xe and 131Xe from P017 -> natural-Xe response for the q^4 strawman
E_SP = P17['E_sp']
SPP_NAT = ISO[129] * P17['129_Spp_W00'] + ISO[131] * P17['131_Spp_W00']
SPP_NAT_G = np.interp(E_GRID, E_SP, SPP_NAT / SPP_NAT[0])

E_STEP = np.diff(E_GRID).mean()
EFF_G = efficiency(E_GRID)
MASK_LO = (E_GRID >= E_LO) & (E_GRID <= E_LO_MAX)
MASK_HI = (E_GRID >= E_HI) & (E_GRID <= E_HI_MAX)

def integ(y, mask):
    return np.trapezoid(y[mask], E_GRID[mask])

def n_lo(spec):
    """spec: dR/dE_R on E_GRID (arbitrary normalisation, efficiency NOT included)."""
    lo = integ(spec * EFF_G, MASK_LO); hi = integ(spec * EFF_G, MASK_HI)
    return lo / hi if hi > 0 else np.inf

# ----------------------------------------------------------------------------- kinematics
def er_max_gev(T, m, mN=M_XE):
    """Maximum elastic recoil energy for projectile kinetic energy T (GeV), mass m, on nucleus mN.
    E_R,max = 2 mN p^2 / s,  p^2 = T^2 + 2 m T,  s = (m+mN)^2 + 2 mN T  (exact two-body kinematics)."""
    T = np.asarray(T, dtype=float)
    return 2 * mN * (T**2 + 2 * m * T) / ((m + mN)**2 + 2 * mN * T)

def t_min_gev(E_R_keV, m, mN=M_XE):
    """Minimum kinetic energy giving recoil E_R (solve E_R,max(T) = E_R; quadratic in T)."""
    ER = E_R_keV * KEV
    # 2 mN (T^2 + 2 m T) = ER ((m+mN)^2 + 2 mN T)  ->  2mN T^2 + (4 m mN - 2 mN ER) T - ER (m+mN)^2 = 0
    a = 2 * mN; b = 4 * m * mN - 2 * mN * ER; c = -ER * (m + mN)**2
    return (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)

def beta_of(T, m):
    E = T + m
    return math.sqrt(1 - (m / E)**2)

def amplitude_factor(E_R_keV, T, m, kind, mN=M_XE):
    """|M|^2(t)/|M|^2(t->0) for a relativistic Dirac fermion chi on a point nucleus (spin-averaged).
    const : Bringmann-Pospelov assumption (dsigma/dE_R flat up to E_R,max)
    vecD  : vector mediator, |M|^2 ~ 2(s-Sigma)^2 + 2 s t + t^2      (recalled, likely; derived in details.md)
    scaD  : scalar mediator, |M|^2 ~ (4 m^2 - t)(4 mN^2 - t)         (recalled, likely)
    The scalar case is normalised to its value at E_R -> 0 only when m > 0."""
    ER = np.asarray(E_R_keV, dtype=float) * KEV
    Echi = T + m
    if kind == 'const':
        return np.ones_like(ER)
    if kind == 'vecD':
        Sigma = m * m + mN * mN
        return 1 - ER * (Sigma + 2 * mN * Echi) / (2 * mN * Echi**2) + ER**2 / (2 * Echi**2)
    if kind == 'scaD':
        return (1 + mN * ER / (2 * m * m)) * (1 + ER / (2 * mN))
    raise ValueError(kind)

def beam_spectrum(T, m, kind='const', ff='helm', mN=M_XE):
    """dR/dE_R (arbitrary norm) on E_GRID for a monochromatic beam of kinetic energy T (GeV)."""
    emax = float(er_max_gev(T, m, mN)) / KEV
    F2 = {'helm': F2_HELM_NAT, 'shell': F2_SHELL_NAT, 'none': np.ones_like(E_GRID),
          'spp_q4': (E_GRID * KEV * 2 * mN) ** 2 / (2 * mN * E_EVENT * KEV) ** 2 * SPP_NAT_G}[ff]
    spec = amplitude_factor(E_GRID, T, m, kind, mN) * F2 / emax
    spec = np.where(E_GRID <= emax, spec, 0.0)
    return spec, emax

# ----------------------------------------------------------------------------- 1. kinematics table
results = {}
masses_light = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
kin_rows = []
for m in masses_light + [1000.0]:
    Tm = t_min_gev(E_EVENT, m); Tm270 = t_min_gev(E_HI_MAX, m); Tm200 = t_min_gev(E_HI, m)
    kin_rows.append(dict(m_chi_GeV=m, T_min_248_MeV=Tm * 1e3, T_min_200_MeV=Tm200 * 1e3, T_min_270_MeV=Tm270 * 1e3,
                         beta_min_248=beta_of(Tm, m), p_min_MeV=math.sqrt(Tm**2 + 2 * m * Tm) * 1e3,
                         E_R_max_at_2Tmin_keV=float(er_max_gev(2 * Tm, m)) / KEV))
import pandas as pd
kin = pd.DataFrame(kin_rows); kin.to_csv(f'{OUT}/P040_kinematics.csv', index=False)
print('\n== kinematics (E_R = 248 keV on natural Xe, m_N = %.1f GeV)' % M_XE); print(kin.to_string(index=False))
# sanity: non-relativistic limit E_R,max -> 2 mu^2 v^2 / mN and neutron check vs P013 (8.20 MeV)
T_n = t_min_gev(E_EVENT, 0.939565)
print('neutron check: T_min(248 keV) = %.3f MeV (P013: 8.20 MeV)' % (T_n * 1e3))
results['neutron_Tmin_MeV'] = T_n * 1e3

# ----------------------------------------------------------------------------- 2. N_lo for beams
print('\n== form-factor numbers used')
for e in (10, 30, 55, 200, 248, 270):
    i = np.argmin(abs(E_GRID - e))
    print(f'E={e:4d} keV  Helm F2(nat, A^2-weighted)={F2_HELM_NAT[i]:.3e}  P017 Helm={F2_HELM_P17[i]:.3e}  shell={F2_SHELL_NAT[i]:.3e}  eff={EFF_G[i]:.3f}')
results['F2'] = {str(e): dict(helm=float(F2_HELM_NAT[np.argmin(abs(E_GRID - e))]), shell=float(F2_SHELL_NAT[np.argmin(abs(E_GRID - e))]))
                 for e in (10, 30, 55, 200, 248, 270)}
# form-factor suppression ratios
i30 = np.argmin(abs(E_GRID - 30)); i248 = np.argmin(abs(E_GRID - 248))
results['F2_ratio_30_over_248'] = dict(helm=float(F2_HELM_NAT[i30] / F2_HELM_NAT[i248]), shell=float(F2_SHELL_NAT[i30] / F2_SHELL_NAT[i248]))
print('F2(30)/F2(248): Helm %.0f, shell %.0f' % (results['F2_ratio_30_over_248']['helm'], results['F2_ratio_30_over_248']['shell']))

# flat-spectrum (E_R,max -> infinity) minimum N_lo for each response
flat = {}
for ff in ('none', 'helm', 'shell', 'spp_q4'):
    spec, _ = beam_spectrum(100.0, 1.0, 'const', ff)   # T = 100 GeV -> E_max >> 270 keV
    flat[ff] = n_lo(spec)
# analytic check of the no-form-factor value: (55-5.4)/(70) with efficiency
print('\n== N_lo for a flat-in-E_R spectrum (E_R,max >> 270 keV), const amplitude:')
for k, v in flat.items(): print(f'   {k:7s} N_lo = {v:9.2f}')
results['N_lo_flat'] = flat
# q^2 and q^4 ladder with coherent form factors (E_R^n weights) for reference
ladder = {}
for n in (0, 1, 2):
    for ff, F2 in (('helm', F2_HELM_NAT), ('shell', F2_SHELL_NAT)):
        ladder[f'q^{2*n}_{ff}'] = n_lo((E_GRID ** n) * F2)
results['N_lo_qladder_flat'] = ladder
print('   q^2n ladder (const x E_R^n x F2):', {k: round(v, 2) for k, v in ladder.items()})

# monochromatic beams: N_lo vs T for each mass and amplitude, both form factors
T_scan = np.logspace(-3, 1, 400)   # GeV
beam_rows = []
for m in masses_light:
    for kind in ('const', 'vecD', 'scaD'):
        for ff in ('helm', 'shell'):
            for T in T_scan:
                spec, emax = beam_spectrum(T, m, kind, ff)
                if emax < E_EVENT: continue
                beam_rows.append(dict(m_chi_GeV=m, kind=kind, ff=ff, T_GeV=T, E_R_max_keV=emax, N_lo=n_lo(spec)))
beam = pd.DataFrame(beam_rows); beam.to_csv(f'{OUT}/P040_Nlo_beams.csv', index=False)

# minimiser: N_lo vs E_R,max between 248 and 600 keV (const amplitude), mass 0.1 GeV
mini = []
for emax_target in np.arange(248, 601, 2.0):
    # invert E_R,max(T) for T
    T = t_min_gev(emax_target, 0.1)
    for ff in ('helm', 'shell'):
        spec, emax = beam_spectrum(T, 0.1, 'const', ff)
        mini.append(dict(E_R_max_keV=emax, ff=ff, N_lo=n_lo(spec)))
mini = pd.DataFrame(mini); mini.to_csv(f'{OUT}/P040_Nlo_vs_Emax.csv', index=False)
for ff in ('helm', 'shell'):
    sub = mini[mini.ff == ff]; j = sub.N_lo.idxmin()
    results[f'N_lo_min_{ff}'] = dict(N_lo=float(sub.loc[j, 'N_lo']), E_R_max_keV=float(sub.loc[j, 'E_R_max_keV']),
                                     N_lo_at_Emax_250=float(sub.iloc[np.argmin(abs(sub.E_R_max_keV - 250))].N_lo),
                                     N_lo_at_Emax_270=float(sub.iloc[np.argmin(abs(sub.E_R_max_keV - 270))].N_lo))
    print(f'   minimum over E_R,max ({ff}): N_lo = {results[f"N_lo_min_{ff}"]["N_lo"]:.1f} at E_R,max = {results[f"N_lo_min_{ff}"]["E_R_max_keV"]:.0f} keV; '
          f'E_R,max=250: {results[f"N_lo_min_{ff}"]["N_lo_at_Emax_250"]:.1f}, 270: {results[f"N_lo_min_{ff}"]["N_lo_at_Emax_270"]:.1f}')

# summary table of beams at T = 1.2 T_min(270) and at T = 10 T_min
beam_summary = []
for m in masses_light:
    for kind in ('const', 'vecD', 'scaD'):
        for fac, lab in ((1.2, '1.2xTmin270'), (10.0, '10xTmin270')):
            T = fac * t_min_gev(E_HI_MAX, m)
            row = dict(m_chi_GeV=m, kind=kind, T_label=lab, T_MeV=T * 1e3, E_R_max_keV=float(er_max_gev(T, m)) / KEV)
            for ff in ('helm', 'shell'):
                spec, _ = beam_spectrum(T, m, kind, ff); row[f'N_lo_{ff}'] = n_lo(spec)
            beam_summary.append(row)
bs = pd.DataFrame(beam_summary); bs.to_csv(f'{OUT}/P040_Nlo_beam_summary.csv', index=False)
print('\n== monochromatic beams (N_lo):'); print(bs.to_string(index=False, float_format=lambda x: f'{x:.4g}'))

# power-law incident spectra dPhi/dT ~ T^-gamma, T in [T_lo, 100 GeV]
def powerlaw_spectrum(m, gamma, T_lo, T_hi=100.0, kind='const', ff='helm', nT=600):
    Tg = np.logspace(math.log10(T_lo), math.log10(T_hi), nT)
    wgt = Tg ** (-gamma)
    spec = np.zeros_like(E_GRID)
    for T, wt in zip(Tg, wgt):
        s, _ = beam_spectrum(T, m, kind, ff); spec += wt * s * T   # log-spaced: dT = T dlnT
    return spec
pl_rows = []
for m in masses_light:
    for gamma in (1.5, 2.0, 2.7):
        for tlo_lab, T_lo in (('Tmin270', t_min_gev(E_HI_MAX, m)), ('10MeV', 0.010)):
            if T_lo > 50: continue
            row = dict(m_chi_GeV=m, gamma=gamma, T_lo_label=tlo_lab, T_lo_MeV=T_lo * 1e3)
            for ff in ('helm', 'shell'):
                row[f'N_lo_{ff}'] = n_lo(powerlaw_spectrum(m, gamma, T_lo, ff=ff))
            pl_rows.append(row)
pl = pd.DataFrame(pl_rows); pl.to_csv(f'{OUT}/P040_Nlo_powerlaw.csv', index=False)
print('\n== power-law incident spectra (const amplitude):'); print(pl.to_string(index=False, float_format=lambda x: f'{x:.4g}'))

# ----------------------------------------------------------------------------- 3. CRDM flux and LZ rate
def phi_p(Tp):
    """Local interstellar cosmic-ray proton flux, 4pi-integrated, cm^-2 s^-1 GeV^-1.
    PDG: I_N(E) ~ 1.8e4 (E/GeV)^-2.7 nucleons m^-2 s^-1 sr^-1 GeV^-1, E = total energy per nucleon,
    valid from a few GeV to 100 TeV (recalled, likely); extended down to T_p = 0.1 GeV as an order-of-magnitude LIS."""
    E = Tp + M_P
    return 4 * math.pi * 1.8 * E ** (-2.7)

def t_chi_max(Tp, m):
    return er_max_gev(Tp, M_P, mN=m)          # roles swapped: proton projectile, chi target
def t_p_min(Tchi, m):
    return t_min_gev(Tchi / KEV, M_P, mN=m)

TCHI = np.logspace(-4, 2, 500)   # GeV
TP = np.logspace(-1, 5, 3000)
def crdm_flux(m, D_eff_kpc=1.0, sigma_p=1e-30, proton_ff=True):
    """dPhi_chi/dT_chi (cm^-2 s^-1 GeV^-1) for constant amplitude, sigma_chi p = sigma_p, D_eff."""
    n_chi = RHO_CHI / m
    out = np.zeros_like(TCHI)
    Tmax = t_chi_max(TP, m)
    dP = phi_p(TP)
    for i, Tc in enumerate(TCHI):
        ok = Tmax > Tc
        if not ok.any(): continue
        q2 = 2 * m * Tc
        G2 = (1 + q2 / LAMBDA2_DIPOLE) ** -4 if proton_ff else 1.0
        integrand = dP[ok] * sigma_p / Tmax[ok] * G2
        out[i] = np.trapezoid(integrand, TP[ok])
    return D_eff_kpc * KPC_CM * n_chi * out

flux_rows = []
FLUX = {}
for m in masses_light:
    f = crdm_flux(m); FLUX[m] = f
    sel = TCHI >= 0.010
    Phi_gt10 = np.trapezoid(f[sel], TCHI[sel]); Phi_tot = np.trapezoid(f, TCHI)
    Tm = t_min_gev(E_EVENT, m); sel2 = TCHI >= Tm
    Phi_gtTmin = np.trapezoid(f[sel2], TCHI[sel2])
    flux_rows.append(dict(m_chi_GeV=m, Phi_total=Phi_tot, Phi_T_gt_10MeV=Phi_gt10, Phi_T_gt_Tmin248=Phi_gtTmin,
                          T_min248_MeV=Tm * 1e3, recalled_scaling_1em7_GeV_over_m=1e-7 / m))
fl = pd.DataFrame(flux_rows); fl.to_csv(f'{OUT}/P040_crdm_flux.csv', index=False)
print('\n== CRDM flux at sigma_chi p = 1e-30 cm^2, D_eff = 1 kpc, rho = 0.3 GeV/cm^3 (cm^-2 s^-1):')
print(fl.to_string(index=False, float_format=lambda x: f'{x:.3g}'))

# LZ rate: R = sigma_p^2 * K
LIVE_S = lz.LZ['live_days'] * 86400.0
M_FV_G = lz.LZ['fiducial_mass_t'] * 1e6
N_NUC = {A: ISO[A] * M_FV_G / (A_MEAN * 1.66054e-24) for A in ISO}   # nuclei of each isotope in the FV
def K_factor(m, ff='helm', D_eff_kpc=1.0, window=(E_HI, E_HI_MAX), kind='const'):
    f1 = crdm_flux(m, D_eff_kpc, sigma_p=1.0)      # flux per unit sigma_p (linear)
    mask = (E_GRID >= window[0]) & (E_GRID <= window[1])
    mu_p = lz.mu_red(m, M_P)
    K = 0.0
    for A in ISO:
        mN = A * AMU; mu_N = lz.mu_red(m, mN)
        F2 = (F2_HELM_ISO if ff == 'helm' else F2_SHELL_ISO)[A]
        emax = er_max_gev(TCHI, m, mN)                                     # GeV, per T
        # inner integral over E_R (GeV) of eff*F2*amp / emax, with E_R <= emax
        ER = E_GRID * KEV
        inner = np.zeros_like(TCHI)
        for i, (T, em) in enumerate(zip(TCHI, emax)):
            ok = mask & (ER <= em)
            if not ok.any(): continue
            amp = amplitude_factor(E_GRID[ok], T, m, kind, mN)
            inner[i] = np.trapezoid(EFF_G[ok] * F2[ok] * amp, ER[ok]) / em
        sigma_N_per_sigma_p = A**2 * (mu_N / mu_p) ** 2
        K += N_NUC[A] * LIVE_S * sigma_N_per_sigma_p * np.trapezoid(f1 * inner, TCHI)
    return K

sig_rows = []
for m in masses_light:
    row = dict(m_chi_GeV=m)
    for ff in ('helm', 'shell'):
        for D in (1.0, 10.0):
            K = K_factor(m, ff, D); row[f'sigma_p_1event_{ff}_D{int(D)}kpc_cm2'] = math.sqrt(1.0 / K)
    # low-energy companions with the CRDM spectrum shape (const amplitude): N_lo in 2.84 t yr and 4.2 t yr (2024 search)
    for ff in ('helm', 'shell'):
        K_lo = K_factor(m, ff, 1.0, window=(E_LO, E_LO_MAX))
        K_hi = K_factor(m, ff, 1.0)
        row[f'N_lo_crdm_{ff}'] = K_lo / K_hi
    sig_rows.append(row)
sg = pd.DataFrame(sig_rows); sg.to_csv(f'{OUT}/P040_sigma_required.csv', index=False)
print('\n== sigma_chi p giving 1 event in 200-270 keV (2.84 t yr) and CRDM-spectrum N_lo:')
print(sg.to_string(index=False, float_format=lambda x: f'{x:.3g}'))

# recalled exclusion bands (order of magnitude; flagged uncertain in details.md)
RECALLED_LIMITS = dict(
    XENON1T_BP2019=dict(m_range_GeV=(1e-3, 1.0), sigma_lower=1e-31, sigma_upper=1e-28, reliability='uncertain',
                        note='Bringmann & Pospelov PRL 122, 171801 (2019) recast of XENON1T: excluded band roughly 1e-31 to 1e-28 cm^2 for m_chi <~ 0.1-1 GeV'),
    PandaX4T_2022=dict(m_range_GeV=(1e-3, 0.1), sigma_lower=3e-32, sigma_upper=1e-28, reliability='uncertain',
                       note='PandaX-4T CRDM search (PRL 128, 171801, 2022): lower edge a few 1e-32 cm^2 (recalled, uncertain)'),
    LZ2024_lowE_elastic=dict(note='LZ 2024 low-energy search; any sigma_p giving 1 window event also gives N_lo companions (computed here), independent of recalled limits'),
)
results['recalled_limits'] = RECALLED_LIMITS

# Earth attenuation in 1478 m of standard rock
ROCK_DEPTH_CM = 1478e2; ROCK_RHO = 2.7
ROCK = {16: 0.47, 28: 0.34, 27: 0.08, 40: 0.04, 56: 0.05, 24: 0.02}   # A -> mass fraction (crustal rock, recalled/likely)
X_COL = ROCK_DEPTH_CM * ROCK_RHO   # g/cm^2
att_rows = []
for m in masses_light + [1000.0]:
    T = 1.5 * t_min_gev(E_EVENT, m)        # a particle arriving with 1.5 T_min
    mu_p = lz.mu_red(m, M_P)
    n_sc_per_sigma = 0.0; dE_per_sigma = 0.0
    for A, wA in ROCK.items():
        mN = A * AMU; mu_N = lz.mu_red(m, mN)
        n_A = X_COL * wA / (A * 1.66054e-24)                # nuclei per cm^2
        emax = float(er_max_gev(T, m, mN))                  # GeV
        Eg = np.linspace(0, emax, 400)
        F2 = np.array([lz.helm_F2(e / KEV, A) if e > 0 else 1.0 for e in Eg])
        avgF2 = np.trapezoid(F2, Eg) / emax                 # <F^2> over flat spectrum
        avgE = np.trapezoid(F2 * Eg, Eg) / emax             # <E_R> (GeV) per unit sigma_N-weighted
        sN = A**2 * (mu_N / mu_p) ** 2
        n_sc_per_sigma += n_A * sN * avgF2
        dE_per_sigma += n_A * sN * avgE
    frac_loss_per_sigma = dE_per_sigma / T                  # fractional energy loss per unit sigma_p (cm^-2)
    att_rows.append(dict(m_chi_GeV=m, T_GeV=T, n_scatters_at_1em30=n_sc_per_sigma * 1e-30, frac_loss_at_1em30=frac_loss_per_sigma * 1e-30,
                         sigma_p_30pct_loss_cm2=0.3 / frac_loss_per_sigma, sigma_p_1_scatter_cm2=1.0 / n_sc_per_sigma))
at = pd.DataFrame(att_rows); at.to_csv(f'{OUT}/P040_attenuation.csv', index=False)
print('\n== Earth attenuation, %.0f m rock (X = %.2e g/cm^2), Helm form factors, flat spectrum:' % (ROCK_DEPTH_CM / 100, X_COL))
print(at.to_string(index=False, float_format=lambda x: f'{x:.3g}'))

# ----------------------------------------------------------------------------- 4. heavy fast component
heavy_rows = []
for m in (1000.0, 4000.0):
    for beta in (0.005, 0.01, 0.02):
        T = m * (1 / math.sqrt(1 - beta**2) - 1)
        emax = float(er_max_gev(T, m)) / KEV
        mu = lz.mu_red(m, M_XE); emax_nr = 2 * mu**2 * beta**2 / M_XE / KEV
        row = dict(m_chi_GeV=m, beta=beta, T_GeV=T, E_R_max_keV=emax, E_R_max_NR_formula_keV=emax_nr)
        for ff in ('helm', 'shell'):
            spec, _ = beam_spectrum(T, m, 'const', ff); row[f'N_lo_{ff}'] = n_lo(spec)
        heavy_rows.append(row)
hv = pd.DataFrame(heavy_rows); hv.to_csv(f'{OUT}/P040_heavy_fast.csv', index=False)
print('\n== heavy fast component:'); print(hv.to_string(index=False, float_format=lambda x: f'{x:.4g}'))
# SHM comparison: lzcommon Helm SI rate N_lo for 1000 GeV (elastic, standard halo)
E_shm = E_GRID[E_GRID <= 300]
r_shm = np.array([lz.dRdE_SI(e, 1000.0, 1e-45) for e in E_shm])
spec_shm = np.interp(E_GRID, E_shm, r_shm, right=0.0)
results['N_lo_SHM_1TeV_Helm'] = n_lo(spec_shm)
print('   SHM 1 TeV elastic SI (Helm, lzcommon): N_lo = %.0f  (P030 Helm: 353 flat; P003 shell O1: 2752)' % results['N_lo_SHM_1TeV_Helm'])

# ----------------------------------------------------------------------------- figures
plt.rcParams.update({'font.size': 9})
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
colors = {1e-3: 'C0', 1e-2: 'C1', 1e-1: 'C2', 1.0: 'C3', 10.0: 'C4'}
ls = {'const': '-', 'vecD': '--', 'scaD': ':'}
for ax, ff in zip(axes, ('helm', 'shell')):
    for m in masses_light:
        for kind in ('const', 'scaD'):
            sub = beam[(beam.m_chi_GeV == m) & (beam.kind == kind) & (beam.ff == ff)]
            ax.plot(sub.T_GeV * 1e3, sub.N_lo, ls[kind], color=colors[m], lw=1.4,
                    label=f'm = {m*1e3:g} MeV, {"constant amplitude" if kind=="const" else "scalar mediator (q^2)"}' if ff == 'helm' else None)
    ax.axhspan(3, 5, color='0.8', alpha=0.6); ax.text(1.3, 3.6, 'lone-event tolerance (P003: 3-5)', fontsize=7)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('incident kinetic energy T [MeV]')
    ax.set_title({'helm': 'Helm form factor (natural Xe)', 'shell': 'shell-model M response (P017 / WimPyDD)'}[ff])
    ax.set_ylim(1, 1e5); ax.grid(alpha=0.3, which='both')
axes[0].set_ylabel('N_lo = events in 5.4-55 keV per event in 200-270 keV')
axes[0].legend(fontsize=6.5, ncol=2, loc='upper right')
fig.suptitle('P040: monochromatic relativistic beams on xenon -- low-energy companions per 200-270 keV event', fontsize=10)
fig.tight_layout(); fig.savefig(f'{FIG}/P040_fig1_Nlo_vs_T.png', dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.2, 4.4))
mm = sg.m_chi_GeV.values
ax.plot(mm, sg['sigma_p_1event_helm_D1kpc_cm2'], 'o-', color='C3', label='1 LZ event in 200-270 keV, Helm, D_eff = 1 kpc')
ax.plot(mm, sg['sigma_p_1event_shell_D1kpc_cm2'], 's--', color='C1', label='same, shell-model M response')
ax.plot(mm, sg['sigma_p_1event_helm_D10kpc_cm2'], 'o:', color='C3', alpha=0.6, label='Helm, D_eff = 10 kpc')
ax.fill_between([1e-3, 1.0], 1e-31, 1e-28, color='0.85', label='recalled CRDM exclusions (XENON1T recast / PandaX-4T), uncertain')
ax.plot(at.m_chi_GeV[:5], at.sigma_p_30pct_loss_cm2[:5], 'k-.', label='30 % energy loss in 1.5 km rock (this work)')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('m_chi [GeV]'); ax.set_ylabel('sigma_chi p [cm^2]')
ax.set_title('P040: CRDM cross-section required for one LZ window event', fontsize=10)
ax.legend(fontsize=6.5, loc='lower left'); ax.grid(alpha=0.3, which='both')
ax.text(0.98, 0.97, 'every point also implies N_lo = %d-%d events\nin 5.4-55 keV per window event (Fig. 1, Table 4)' % (sg.N_lo_crdm_helm.min(), sg.N_lo_crdm_shell.max()),
        fontsize=7, ha='right', va='top', transform=ax.transAxes)
fig.tight_layout(); fig.savefig(f'{FIG}/P040_fig2_sigma_required.png', dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.2, 4.0))
for ff, c in (('helm', 'C3'), ('shell', 'C1')):
    sub = mini[mini.ff == ff]; ax.plot(sub.E_R_max_keV, sub.N_lo, color=c, label=f'{ff} form factor')
ax.axvline(E_EVENT, color='k', ls=':', lw=0.8); ax.text(250, 20, '248 keV', fontsize=7)
ax.axhspan(3, 5, color='0.8'); ax.set_xlabel('E_R,max of the beam [keV]'); ax.set_ylabel('N_lo'); ax.set_yscale('log')
ax.set_title('P040: N_lo of a flat recoil spectrum vs its end point', fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f'{FIG}/P040_fig3_Nlo_vs_Emax.png', dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.2, 4.0))
for m in masses_light:
    ax.plot(TCHI * 1e3, TCHI * FLUX[m], color=colors[m], label=f'm = {m*1e3:g} MeV')
    ax.axvline(t_min_gev(E_EVENT, m) * 1e3, color=colors[m], ls=':', lw=0.8)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e-13, 1e-4); ax.set_xlim(0.1, 1e5)
ax.set_xlabel('T_chi [MeV]'); ax.set_ylabel('T dPhi/dT [cm^-2 s^-1]')
ax.set_title('P040: CRDM flux, sigma_chi p = 1e-30 cm^2, D_eff = 1 kpc (dotted: T_min for 248 keV)', fontsize=9)
ax.legend(fontsize=7); ax.grid(alpha=0.3, which='both')
fig.tight_layout(); fig.savefig(f'{FIG}/P040_fig4_crdm_flux.png', dpi=160); plt.close(fig)

results['kinematics'] = kin.to_dict(orient='records')
results['sigma_required'] = sg.to_dict(orient='records')
results['attenuation'] = at.to_dict(orient='records')
results['heavy_fast'] = hv.to_dict(orient='records')
results['crdm_flux_1e-30_1kpc'] = fl.to_dict(orient='records')
results['inputs'] = dict(m_N_GeV=M_XE, efficiency='P003 erf model eff0=0.96, sig_lo=3.4, sig_hi=8.0 keV', rho_chi=RHO_CHI,
                         proton_spectrum='PDG 1.8e4 (E/GeV)^-2.7 nucleons m^-2 s^-1 sr^-1 GeV^-1 (recalled, likely)',
                         rock=dict(depth_m=1478, rho=2.7, X_g_cm2=X_COL, composition=ROCK), lambda2_dipole=LAMBDA2_DIPOLE)
with open(f'{OUT}/P040_results.json', 'w') as fh:
    json.dump(results, fh, indent=1, default=float)
print('\nwrote', OUT)
