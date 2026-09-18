#!/usr/bin/env python
"""
P094 -- Multi-component dark matter with a populated excited state: electron-recoil lines,
low-threshold recoils and the two-state phenomenology across experiments.

Run from the simulation root:   .venv/bin/python output/code/P094_multicomponent.py
Outputs: output/work/P094/*.csv, P094_results.json, figures/*.png

Parts
  A  exothermic DM-electron kinematics (exact relativistic two-body): line at T_e = delta + O(v)
  B  exothermic DM-electron rate for the P011 dark-photon coupling (sigma_e/sigma_p = mu_e^2/mu_p^2)
     versus P058's nuclear exothermic rate
  C  LZ high-energy ER background at 300-380 keV (2nbb computed, 214Pb scaled) and the line sensitivity
  D  WimPyDD exothermic nuclear spectra on Xe, Ar, Ge, Si, O, Ca, W, Na, I (O1, proton-only and isoscalar)
  E  re-population of chi2 today: chi1 chi1 -> chi2 chi2 threshold, up-scattering in Earth (Fe, WimPyDD),
     ISM heavy nuclei, Sun, cosmic rays
  F  combined (tau, f2) map: P026 freeze-out curve, P058 nuclear limits, P066 gamma line, this work
"""
import sys, os, json, math, time
sys.path.insert(0, 'output/code')
from common import lzcommon as lz
import numpy as np
import pandas as pd
from scipy import integrate, special, stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

T0 = time.time()
OUT = 'output/work/P094'
FIG = OUT + '/figures'
os.makedirs(FIG, exist_ok=True)
RES = {}

# ----------------------------------------------------------------------------- constants
C = lz.C_KMS                      # km/s
ME = 0.51099895e-3                # GeV (recalled, certain)
MP = lz.M_NUCLEON_GEV
NA = 6.02214076e23
YR = 3.15576e7                    # s
TU = 13.8e9 * YR                  # 4.35e17 s (P026/P058 convention)
RHO0 = lz.RHO0_GEV_CM3
CM_PER_S = 2.99792458e10
GEV2_TO_CM2 = lz.GEV_TO_CM2
A_XE = 131.293
Z_XE = 54
ATOMS_PER_T_XE = 1e6 / A_XE * NA          # 4.587e27
ELECTRONS_PER_T_XE = Z_XE * ATOMS_PER_T_XE

# P011 sigma_p(N=1), 1 TeV, annual halo [cm^2]; 366 keV from P058 (P011 quotes 365 keV: 2.29e-39)
SIGMA_P_N1 = {250: 1.20e-42, 300: 8.59e-42, 350: 2.99e-40, 366: 2.69e-39, 380: 5.60e-38}
# P058 proton-only exothermic ROI events per endothermic ROI event and unit f2 (2.84 t yr, annual halo)
P058_R_ROI_P = {300: 859., 350: 1.81e4, 366: 1.43e5, 380: 2.64e6}
P058_F2_P = {300: 3.1e-3, 350: 9.7e-5, 366: 1.1e-5, 380: 5.7e-7}     # joint 90% CL, proton-only
P058_F2_ISO = {300: 2.8e-3, 350: 5.7e-5, 366: 4.8e-6, 380: 1.7e-7}
P058_TAU0_1GEV = {250: 5.0e19, 278: 1.0e19, 300: 2.78e18, 350: 3.7e16, 366: 3.3e15, 380: 1.3e14}  # s at m_A'=1 GeV
F2_FO = {'dark photon': 0.50, 'Higgsino': 0.42}     # P026
EXPOSURE_LZ = lz.LZ['exposure_tyr']

def log(*a):
    print(*a, flush=True)

# =============================================================================
# PART A: exothermic DM-electron kinematics (exact relativistic two-body)
# =============================================================================
def electron_line(m_gev, delta_kev, v_kms):
    """chi2 (mass m+delta) with lab speed v hits a free electron at rest; returns dict with the
    electron kinetic energy T (keV) at cos theta* = -1, 0, +1 and the CM-isotropic mean."""
    m1 = m_gev
    m2 = m_gev + delta_kev * 1e-6
    beta = v_kms / C
    gam = 1.0 / math.sqrt(1.0 - beta * beta)
    E2 = gam * m2
    P2 = gam * m2 * beta
    s = m2 * m2 + ME * ME + 2.0 * E2 * ME
    rs = math.sqrt(s)
    Ef_star = (s + ME * ME - m1 * m1) / (2.0 * rs)
    pf_star = math.sqrt(max(Ef_star * Ef_star - ME * ME, 0.0))
    beta_cm = P2 / (E2 + ME)
    gam_cm = 1.0 / math.sqrt(1.0 - beta_cm * beta_cm)
    T = {}
    for lab, ct in (('min', -1.0), ('mid', 0.0), ('max', 1.0)):
        T[lab] = (gam_cm * (Ef_star + beta_cm * pf_star * ct) - ME) * 1e6
    T['mean'] = (gam_cm * Ef_star - ME) * 1e6
    T['pf_keV'] = pf_star * 1e6
    return T

rowsA = []
for m in (400., 1000., 4000.):
    for d in (300., 350., 380.):
        for v in (0., 250., 550., 810.):
            t = electron_line(m, d, v)
            pf = math.sqrt(d * d + 2 * ME * 1e6 * d)          # keV, m -> infinity
            rowsA.append(dict(m_GeV=m, delta_keV=d, v_kms=v, T_min_keV=t['min'], T_mean_keV=t['mean'],
                              T_max_keV=t['max'], half_width_keV=0.5 * (t['max'] - t['min']),
                              doppler_estimate_keV=pf * v / C, p_f_keV=pf,
                              mean_minus_delta_eV=(t['mean'] - d) * 1e3))
dfA = pd.DataFrame(rowsA)
dfA.to_csv(OUT + '/P094_electron_line_kinematics.csv', index=False)
log('\n[A] electron line, 1 TeV, delta=300 keV:')
log(dfA[(dfA.m_GeV == 1000) & (dfA.delta_keV == 300)].to_string(index=False))

# halo rms line-of-sight speed for the Doppler width (Maxwellian v0=238, Earth speed 250.6, cut at 544)
def shm_moments(v0=lz.V0_KMS, vE=None, vesc=lz.VESC_KMS):
    vE = lz.v_earth_kms() if vE is None else vE
    # Earth-frame speed distribution of the truncated Maxwellian, numerically
    def f_gal(u):
        return np.exp(-(u / v0) ** 2) * (u < vesc)
    v = np.linspace(0, vesc + vE, 4000)
    # f_E(v) = v^2 int dcos f_gal(|v - vE|)
    ct = np.linspace(-1, 1, 801)
    F = np.zeros_like(v)
    for i, vi in enumerate(v):
        u = np.sqrt(vi * vi + vE * vE + 2 * vi * vE * ct)
        F[i] = vi * vi * np.trapezoid(f_gal(u), ct)
    F /= np.trapezoid(F, v)
    return dict(v_mean=float(np.trapezoid(v * F, v)), v_rms=float(np.sqrt(np.trapezoid(v * v * F, v))),
                v_inv_mean=float(np.trapezoid(F / np.maximum(v, 1e-3), v)))
mom = shm_moments()
pf300 = math.sqrt(300. ** 2 + 2 * ME * 1e6 * 300.)
sigma_doppler = {d: math.sqrt(d * d + 2 * ME * 1e6 * d) * mom['v_rms'] / math.sqrt(3) / C for d in (300, 350, 380)}
def sigma_det(E):            # LXe TPC ER resolution sigma/E = 0.32/sqrt(E) + 0.0015 (recalled, likely)
    return E * (0.32 / math.sqrt(E) + 0.0015)
RES['A'] = dict(halo_v_mean_kms=mom['v_mean'], halo_v_rms_kms=mom['v_rms'],
                p_f_keV={d: math.sqrt(d * d + 2 * ME * 1e6 * d) for d in (300, 350, 380)},
                sigma_doppler_keV=sigma_doppler,
                sigma_detector_keV={d: sigma_det(d) for d in (300, 350, 380)},
                T_mean_minus_delta_eV_1TeV_250kms=float(dfA[(dfA.m_GeV == 1000) & (dfA.delta_keV == 300) & (dfA.v_kms == 250)].mean_minus_delta_eV.iloc[0]))
log('[A] halo <v>=%.0f km/s, v_rms=%.0f; Doppler sigma at 300/350/380 keV: %s keV; detector sigma: %s keV' % (
    mom['v_mean'], mom['v_rms'], {k: round(v, 2) for k, v in sigma_doppler.items()}, {k: round(sigma_det(k), 1) for k in (300, 350, 380)}))

# where a 300-380 keV ER lands in LZ's {S1c, log10 S2c}
rowsA2 = []
for E in (20., 25., 30., 40., 100., 130., 300., 350., 380.):
    nph, ne = lz.nest_er_yields(E, params=lz.NEST_ER_LZ)
    rowsA2.append(dict(E_keV=E, Nph=nph, Ne=ne, S1c_phd=lz.LZ['g1'] * nph, log10S2c=math.log10(lz.LZ['g2'] * ne),
                       in_WS_ROI=(lz.LZ['g1'] * nph <= 600) and (math.log10(lz.LZ['g2'] * ne) <= 4.15)))
dfA2 = pd.DataFrame(rowsA2)
dfA2.to_csv(OUT + '/P094_ER_in_LZ_space.csv', index=False)
log('[A] ER position (LZ-tuned NEST):\n' + dfA2.round(2).to_string(index=False))
RES['A']['ER_300keV_S1c_phd'] = float(dfA2[dfA2.E_keV == 300].S1c_phd.iloc[0])
RES['A']['ER_300keV_log10S2c'] = float(dfA2[dfA2.E_keV == 300].log10S2c.iloc[0])
RES['A']['ER_380keV_S1c_phd'] = float(dfA2[dfA2.E_keV == 380].S1c_phd.iloc[0])
RES['A']['ER_380keV_log10S2c'] = float(dfA2[dfA2.E_keV == 380].log10S2c.iloc[0])

# =============================================================================
# PART B: exothermic DM-electron rate
# =============================================================================
def mu(a, b):
    return a * b / (a + b)

def sigma_v_electron(sigma_e_cm2, delta_kev):
    """<sigma v> for chi2 e -> chi1 e on a free electron at rest, heavy DM, contact vector mediator:
       sigma = sigma_e^NR (1 + T/2m_e) p_f/p_i   ->  sigma v = sigma_e c (p_f/m_e)(1 + delta/2m_e).  [cm^3/s]"""
    d = delta_kev * 1e-6
    pf = math.sqrt(d * d + 2 * ME * d)
    return sigma_e_cm2 * CM_PER_S * (pf / ME) * (1.0 + d / (2 * ME))

def rate_electron_per_tyr(sigma_e_cm2, delta_kev, m_gev, f2=1.0, m_Aprime_gev=None):
    n_chi = RHO0 * f2 / m_gev
    sv = sigma_v_electron(sigma_e_cm2, delta_kev)
    if m_Aprime_gev is not None:
        d = delta_kev * 1e-6
        q2 = d * d + 2 * ME * d
        sv *= (m_Aprime_gev ** 2 / (m_Aprime_gev ** 2 + q2)) ** 2
    return ELECTRONS_PER_T_XE * n_chi * sv * YR

rowsB = []
for d, sp in SIGMA_P_N1.items():
    m = 1000.
    ratio_mu2 = (mu(ME, m) / mu(MP, m)) ** 2
    se = sp * ratio_mu2
    Re = rate_electron_per_tyr(se, d, m)
    Re_per_sigma = rate_electron_per_tyr(1.0, d, m)
    RN = P058_R_ROI_P.get(d, np.nan) / EXPOSURE_LZ            # nuclear exothermic ROI events per t yr per f2
    kin = sigma_v_electron(1.0, d) / CM_PER_S
    rowsB.append(dict(delta_keV=d, sigma_p_N1_cm2=sp, mu_e2_over_mu_p2=ratio_mu2, sigma_e_cm2=se,
                      sigma_v_over_sigma_c=kin, p_f_keV=math.sqrt(d * d + 2 * ME * 1e6 * d),
                      R_e_per_tyr_f2eq1=Re, R_e_per_tyr_per_f2sigma_e=Re_per_sigma,
                      R_nuc_ROI_per_tyr_f2eq1_P058=RN, ratio_nuc_over_e=RN / Re if RN == RN else np.nan,
                      prop_suppression_mA_2p4GeV=(2.4 ** 2 / (2.4 ** 2 + (math.sqrt(d * d + 2 * ME * 1e6 * d) * 1e-6) ** 2)) ** 2))
dfB = pd.DataFrame(rowsB)
dfB.to_csv(OUT + '/P094_electron_rate.csv', index=False)
log('\n[B] electron line rates (1 TeV, P011 sigma_p(N=1)):\n' + dfB.to_string(index=False, float_format=lambda x: '%.3g' % x))
RES['B'] = dfB.set_index('delta_keV').to_dict(orient='index')

# =============================================================================
# PART C: LZ ER background at 300-380 keV and line sensitivity (ESTIMATE)
# =============================================================================
Q_BB = 2457.8    # keV, 136Xe Q value (recalled, certain)
T_HALF_2NBB = 2.165e21   # yr, EXO-200 (recalled, likely)
ABUND_136 = 0.08857      # (recalled, certain to 1%)
def pr_shape(K_kev):
    """Primakoff-Rosen 2nbb summed-electron spectrum (unnormalised), K in keV."""
    K = np.asarray(K_kev, float) / (ME * 1e6)
    T0 = Q_BB / (ME * 1e6)
    x = K * (T0 - K) ** 5 * (1 + 2 * K + 4 * K ** 2 / 3 + K ** 3 / 3 + K ** 4 / 30)
    return np.where((K > 0) & (K < T0), x, 0.0)
Kgrid = np.linspace(0.01, Q_BB, 20000)
norm = np.trapezoid(pr_shape(Kgrid), Kgrid)
N136_per_t = ABUND_136 * ATOMS_PER_T_XE
decays_per_tyr = N136_per_t * math.log(2) / T_HALF_2NBB
def bb2n_per_tyr_kev(E):
    return decays_per_tyr * pr_shape(E) / norm
bb_check_roi = float(np.trapezoid(bb2n_per_tyr_kev(np.linspace(1.5, 20, 200)), np.linspace(1.5, 20, 200))) * EXPOSURE_LZ
bb_check_40 = float(np.trapezoid(bb2n_per_tyr_kev(np.linspace(1.5, 40, 200)), np.linspace(1.5, 40, 200))) * EXPOSURE_LZ

# 214Pb ground-state beta continuum shape (allowed, Fermi function approx), Q=1019 keV, Z_daughter=83
def beta_shape(T_kev, Q=1019., Z=83):
    T = np.asarray(T_kev, float) / (ME * 1e6)
    W = T + 1.0
    p = np.sqrt(np.maximum(W * W - 1, 1e-12))
    eta = (1 / 137.036) * Z * W / p
    F = 2 * math.pi * eta / (1 - np.exp(-2 * math.pi * eta))
    Q0 = Q / (ME * 1e6)
    return np.where((T > 0) & (T < Q0), F * p * W * (Q0 - T) ** 2, 0.0)
shape_ratio = {E: float(beta_shape(E) / beta_shape(20.)) for E in (300, 350, 380)}
# internal beta normalisation: 1341 events (Table I) in an effective ER window W_eff of the WS ROI
W_EFF = dict(central=25., low=15., high=40.)   # keV_ee, estimated from Fig. 4 (ER band exits ROI at ~20-25 keV_ee median, lower tail to ~40-55)
rowsC = []
for E in (300, 350, 380):
    bb = float(bb2n_per_tyr_kev(E))
    pb = {k: 1341. / (EXPOSURE_LZ * w) * shape_ratio[E] for k, w in W_EFF.items()}
    sig = sigma_det(E)
    for label, B in (('low', bb + pb['high']), ('central', bb + pb['central']), ('high', bb + pb['low'])):
        win = 3.0 * sig       # +-1.5 sigma, 86.6 % containment
        for expo, name in ((EXPOSURE_LZ, 'LZ 2.84 t yr'), (1.0, '1 t yr'), (20.0, '20 t yr')):
            Bwin = B * win * expo
            S90 = 1.64 * math.sqrt(Bwin) / 0.866
            Re_per = rate_electron_per_tyr(1.0, E, 1000.)
            f2se = S90 / (expo * Re_per)
            rowsC.append(dict(E_keV=E, bkg_case=label, bb2n_per_tyr_keV=bb, Pb214_per_tyr_keV=B - bb, B_total_per_tyr_keV=B,
                              sigma_E_keV=sig, exposure_tyr=expo, experiment=name, B_in_window=Bwin, S90_events=S90,
                              f2_sigma_e_limit_cm2=f2se,
                              f2_limit_DP_model=f2se / dfB.set_index('delta_keV').loc[E, 'sigma_e_cm2'] if E in dfB.delta_keV.values else np.nan))
dfC = pd.DataFrame(rowsC)
dfC.to_csv(OUT + '/P094_ER_line_sensitivity.csv', index=False)
log('\n[C] 2nbb: %.3g decays/(t yr); dN/dE at 300/350/380 keV = %.2f/%.2f/%.2f per (t yr keV); check 1.5-20 (1.5-40) keV x 2.84 t yr = %.0f (%.0f) vs LZ Table I 110' % (
    decays_per_tyr, bb2n_per_tyr_kev(300), bb2n_per_tyr_kev(350), bb2n_per_tyr_kev(380), bb_check_roi, bb_check_40))
log('[C] 214Pb shape ratio S(E)/S(20 keV): %s' % {k: round(v, 2) for k, v in shape_ratio.items()})
log(dfC[(dfC.bkg_case == 'central')].to_string(index=False, float_format=lambda x: '%.3g' % x))
RES['C'] = dict(decays_2nbb_per_tyr=decays_per_tyr, bb2n_at_300_350_380=[float(bb2n_per_tyr_kev(E)) for E in (300, 350, 380)],
                bb2n_check_1p5_20keV_x2p84=bb_check_roi, bb2n_check_1p5_40keV_x2p84=bb_check_40, Pb_shape_ratio=shape_ratio,
                central=dfC[(dfC.bkg_case == 'central')].to_dict(orient='records'))

# =============================================================================
# PART D: exothermic nuclear spectra on light and heavy targets (WimPyDD, O1)
# =============================================================================
WD = lz.wd()
HALO = lz.wd_halo()              # Sun-frame Baxter halo (P035 labelling); exothermic rates vary +-4 % over the year (P058)
HAM_P = lz.wd_hamiltonian('P094_p', {1: (1.0, 1.0)})       # c_p = 1 GeV^-2, c_n = 0
HAM_ISO = lz.wd_hamiltonian('P094_iso', {1: (2.0, 0.0)})   # c_p = c_n = 1 GeV^-2
MU_P_1TEV = mu(MP, 1000.)
SIGMA_P_UNIT = MU_P_1TEV ** 2 / math.pi * GEV2_TO_CM2       # sigma_p for c_p = 1 GeV^-2 [cm^2]
log('\n[D] sigma_p(c_p = 1 GeV^-2, 1 TeV) = %.3e cm^2' % SIGMA_P_UNIT)
# WimPyDD's 18xW and 4xCa response files use `np` without importing it (NameError); runtime fix as in P015/P046/P065,
# no WimPyDD file is edited.
import importlib
for mod in ('180W', '182W', '183W', '184W', '186W', '42Ca', '44Ca', '46Ca'):
    importlib.import_module('WimPyDD.Targets.Nuclear_response_functions.%s_func_w' % mod).np = np
TARGETS = {'Xe': WD.Xe, 'Ar': WD.Ar, 'Ge': WD.Ge, 'Si': WD.Si, 'O': WD.O, 'Ca': WD.Ca, 'W': WD.W, 'Na': WD.Na, 'I': WD.I}
COMPOUNDS = {'CaWO4': {'Ca': 40.078 / 287.92, 'W': 183.84 / 287.92, 'O': 63.996 / 287.92},
             'NaI': {'Na': 22.99 / 149.89, 'I': 126.90 / 149.89}}
EGRID = np.arange(2.0, 802.0, 4.0)
CACHE = OUT + '/cache'
os.makedirs(CACHE, exist_ok=True)

def spectrum(tname, ham_name, ham, m, delta):
    fn = '%s/spec_%s_%s_%d_%+d.npy' % (CACHE, tname, ham_name, int(m), int(delta))
    if os.path.exists(fn):
        return np.load(fn)
    r = lz.wd_rate(ham, m, EGRID, halo=HALO, delta_kev=delta, target=TARGETS[tname])   # events/(t yr keV) at c_p=1
    np.save(fn, r)
    return r

def mean_A(t):
    return float(np.sum(np.atleast_1d(t.mass)) / len(np.atleast_1d(t.mass)) / lz.AMU_GEV)

rowsD, spectra = [], {}
t0 = time.time()
for d in (300., 350., 380.):
    for tname in TARGETS:
        r = spectrum(tname, 'p', HAM_P, 1000., -d) * SIGMA_P_N1[int(d)] / SIGMA_P_UNIT     # per t yr keV at sigma_p(N=1), f2 = 1
        spectra[(tname, 'p', d)] = r
        A = mean_A(TARGETS[tname])
        mN = A * lz.AMU_GEV
        muN = mu(1000., mN)
        Estar = d * muN / mN
        Dw = muN * (250. / C) * math.sqrt(2 * muN * d * 1e-6) / mN * 1e6
        tot = np.trapezoid(r, EGRID)
        cum = np.cumsum(r) * 4.0 / max(tot, 1e-300)
        pct = lambda p: float(np.interp(p, cum, EGRID))
        def frac(a, b):
            mk = (EGRID >= a) & (EGRID < b)
            return float(np.trapezoid(r[mk], EGRID[mk]) / tot) if tot > 0 else 0.
        rowsD.append(dict(target=tname, coupling='p', delta_keV=d, A_mean=A, E_star_keV=Estar, half_width_250kms_keV=Dw,
                          rate_per_kgyr_f2eq1=tot / 1000., p16=pct(0.16), p50=pct(0.5), p84=pct(0.84), peak_keV=float(EGRID[np.argmax(r)]),
                          frac_below_20=frac(0, 20), frac_20_100=frac(20, 100), frac_100_200=frac(100, 200), frac_200_400=frac(200, 400), frac_above_400=frac(400, 1e9)))
    log('[D] delta=%d done, %.0f s' % (d, time.time() - t0))
for tname in ('Xe', 'Ar', 'Ge', 'Si'):
    r = spectrum(tname, 'iso', HAM_ISO, 1000., -300.) * SIGMA_P_N1[300] / SIGMA_P_UNIT
    spectra[(tname, 'iso', 300.)] = r
    tot = np.trapezoid(r, EGRID)
    rowsD.append(dict(target=tname, coupling='iso', delta_keV=300., A_mean=mean_A(TARGETS[tname]), rate_per_kgyr_f2eq1=tot / 1000.,
                      p50=float(np.interp(0.5, np.cumsum(r) * 4.0 / tot, EGRID))))
dfD = pd.DataFrame(rowsD)
# compounds
for cname, comp in COMPOUNDS.items():
    for d in (300., 350., 380.):
        r = sum(w * spectra[(el, 'p', d)] for el, w in comp.items())
        spectra[(cname, 'p', d)] = r
        tot = np.trapezoid(r, EGRID)
        cum = np.cumsum(r) * 4.0 / tot
        dfD.loc[len(dfD)] = dict(target=cname, coupling='p', delta_keV=d, A_mean=np.nan, E_star_keV=np.nan, half_width_250kms_keV=np.nan,
                                 rate_per_kgyr_f2eq1=tot / 1000., p16=float(np.interp(0.16, cum, EGRID)), p50=float(np.interp(0.5, cum, EGRID)),
                                 p84=float(np.interp(0.84, cum, EGRID)), peak_keV=float(EGRID[np.argmax(r)]),
                                 frac_below_20=float(np.trapezoid(r[EGRID < 20], EGRID[EGRID < 20]) / tot), frac_20_100=np.nan, frac_100_200=np.nan,
                                 frac_200_400=np.nan, frac_above_400=np.nan)
xe_rate = {d: float(dfD[(dfD.target == 'Xe') & (dfD.coupling == 'p') & (dfD.delta_keV == d)].rate_per_kgyr_f2eq1.iloc[0]) for d in (300., 350., 380.)}
dfD['ratio_to_Xe_per_kg'] = [row.rate_per_kgyr_f2eq1 / xe_rate[row.delta_keV] if row.coupling == 'p' else np.nan for row in dfD.itertuples()]
dfD.to_csv(OUT + '/P094_target_spectra_summary.csv', index=False)
pd.DataFrame({'E_keV': EGRID, **{'%s_p_%d' % (k[0], int(k[2])): v for k, v in spectra.items() if k[1] == 'p'}}).to_csv(OUT + '/P094_target_spectra.csv', index=False)
log('\n[D] exothermic spectra, proton-only, 1 TeV, sigma_p(N=1), f2 = 1, Sun-frame halo:')
log(dfD[dfD.coupling == 'p'].to_string(index=False, float_format=lambda x: '%.3g' % x))
log(dfD[dfD.coupling == 'iso'].to_string(index=False, float_format=lambda x: '%.3g' % x))
# cross-check with P058: Xe proton-only total per t yr at sigma_p(N=1), delta = 300 -> compare with 859/2.84 ROI events (ROI acc ~0.93 x 0.96)
xe_tot_300 = xe_rate[300.] * 1000.
log('[D] check: Xe total exothermic rate at sigma_p(N=1), delta=300 = %.0f /(t yr); P058 ROI-accepted 859/2.84 = %.0f /(t yr) (ROI x eff ~ 0.89)' % (xe_tot_300, 859 / 2.84))
RES['D'] = dict(sigma_p_unit_cm2=SIGMA_P_UNIT, Xe_total_per_tyr_300=xe_tot_300, P058_ROI_per_tyr_300=859 / 2.84,
                table=dfD.to_dict(orient='records'))

# per-experiment exothermic events per unit f2 at sigma_p(N=1) (recalled exposures, flagged in details)
EXPERIMENTS = [  # name, target, exposure (t yr), reliability, published analysis range (keV_nr) [recalled]
    ('LZ WS2024 (this paper)', 'Xe', EXPOSURE_LZ, 'certain (paper)', '5-270 (+ empty 272-670)'),
    ('XENONnT SR0+SR1', 'Xe', 3.1, 'uncertain', '3-60'),
    ('PandaX-4T Run0+Run1', 'Xe', 1.54, 'likely', '~3-100'),
    ('XENON1T', 'Xe', 1.0, 'certain', '3-70'),
    ('DEAP-3600 (2019)', 'Ar', 758 / 365.25, 'likely', '~50-100 (WIMP ROI)'),
    ('DarkSide-50 (2018)', 'Ar', 16660 / 1000 / 365.25, 'likely', '~30-200'),
    ('SuperCDMS Soudan iZIP', 'Ge', 1690 / 1000 / 365.25, 'likely', '8-115'),
    ('EDELWEISS-III', 'Ge', 496 / 1000 / 365.25, 'likely', '~10-100'),
    ('CRESST-III (2019)', 'CaWO4', 5.6 / 1000 / 365.25, 'likely', '0.03-16'),
    ('COSINE-100 (2021)', 'NaI', 172 / 1000, 'uncertain', '~1-15 keV_ee'),
]
rowsX = []
for name, tg, expo, rel, rng in EXPERIMENTS:
    for d in (300., 350., 380.):
        rate = float(dfD[(dfD.target == tg) & (dfD.coupling == 'p') & (dfD.delta_keV == d)].rate_per_kgyr_f2eq1.iloc[0]) * 1000.
        N = rate * expo
        rowsX.append(dict(experiment=name, target=tg, exposure_tyr=expo, exposure_reliability=rel, analysis_range_recalled=rng, delta_keV=d,
                          exo_events_per_f2=N, f2_sensitivity_if_zero_events_full_spectrum=2.3 / N if N > 0 else np.inf))
dfX = pd.DataFrame(rowsX)
dfX.to_csv(OUT + '/P094_experiments.csv', index=False)
log('\n[D] events per unit f2 at sigma_p(N=1), whole exothermic spectrum, no efficiency (delta = 300):')
log(dfX[dfX.delta_keV == 300].to_string(index=False, float_format=lambda x: '%.3g' % x))
RES['D']['experiments_300'] = dfX[dfX.delta_keV == 300].to_dict(orient='records')

# =============================================================================
# PART E: re-population of chi2 today
# =============================================================================
rowsE = []
for m in (400., 1000., 4000.):
    for d in (300., 350., 380.):
        v22 = math.sqrt(8 * d * 1e-6 / m) * C     # chi1 chi1 -> chi2 chi2 : mu v_rel^2/2 = m v_rel^2/4 >= 2 delta
        v12 = math.sqrt(4 * d * 1e-6 / m) * C     # chi1 chi1 -> chi1 chi2 : >= delta
        rowsE.append(dict(m_GeV=m, delta_keV=d, v_rel_threshold_chi2chi2_kms=v22, v_rel_threshold_chi1chi2_kms=v12,
                          v_rel_max_halo_kms=2 * (lz.VESC_KMS + lz.v_earth_kms(153)), ratio_to_max=v12 / (2 * (lz.VESC_KMS + lz.v_earth_kms(153)))))
dfE = pd.DataFrame(rowsE)
dfE.to_csv(OUT + '/P094_selfscatter_thresholds.csv', index=False)
log('\n[E] self-up-scattering thresholds:\n' + dfE.to_string(index=False, float_format=lambda x: '%.4g' % x))

# nuclear thresholds for up-scattering chi1 N -> chi2 N: v_min(optimal) = sqrt(delta/(2 mu))
def vmin_opt(A, m, d):
    """minimum of v_min(E) = (m_N E/mu + delta)/sqrt(2 m_N E), reached at E = mu delta/m_N: v = sqrt(2 delta/mu)"""
    mN = A * lz.AMU_GEV
    return math.sqrt(2 * d * 1e-6 / mu(m, mN)) * C
thr = {A: vmin_opt(A, 1000., 300.) for A in (1, 4, 16, 28, 56, 131, 208)}
V_MAX_HALO = lz.VESC_KMS + lz.v_earth_kms(153)      # 810 km/s (June)
log('[E] v_min,opt (km/s) for delta=300 keV, 1 TeV on A=1,4,16,28,56,131,208: %s ; halo v_max = %.0f' % ({k: round(v) for k, v in thr.items()}, V_MAX_HALO))
# check: lzcommon vmin_kms at E = mu delta/m_N for Xe
_mN = A_XE * lz.AMU_GEV; _mu = mu(1000., _mN)
log('[E] check lz.vmin_kms(E*, 1 TeV, delta=300) on Xe = %.0f km/s vs sqrt(2 delta/mu) = %.0f' % (lz.vmin_kms(300. * _mu / _mN, 1000., A_XE, 300.), thr[131]))

# Earth: up-scattering on the Fe/Ni core (WimPyDD Fe target, delta > 0, proton-only at sigma_p(N=1))
M_CORE_KG = 0.325 * 5.972e24      # Fe/Ni core mass (recalled, likely; treated as Fe)
R_EARTH_CM = 6.371e8
R_CORE_CM = 3.48e8
rowsE2 = []
for d in (300., 350., 380.):
    rFe = lz.wd_rate(HAM_P, 1000., np.arange(1., 400., 2.), halo=HALO, delta_kev=+d, target=WD.Fe) * SIGMA_P_N1[int(d)] / SIGMA_P_UNIT  # per t yr keV
    up_per_kgyr = float(np.trapezoid(rFe, np.arange(1., 400., 2.))) / 1000.
    rXe = lz.wd_rate(HAM_P, 1000., np.arange(1., 450., 2.), halo=HALO, delta_kev=+d, target=WD.Xe) * SIGMA_P_N1[int(d)] / SIGMA_P_UNIT
    up_xe_per_tyr = float(np.trapezoid(rXe, np.arange(1., 450., 2.)))
    N_up_earth_per_yr = up_per_kgyr * M_CORE_KG
    n_chi = RHO0 / 1000.
    flux_cross_per_yr = n_chi * mom['v_mean'] * 1e5 * math.pi * R_EARTH_CM ** 2 * YR     # chi1 crossings of the Earth per yr
    P_conv = N_up_earth_per_yr / flux_cross_per_yr
    f2_earth = P_conv * 0.5 * (R_CORE_CM / R_EARTH_CM) ** 2   # fraction of DM at the detector that has just crossed the core
    # ISM: heavy nuclei (A > 100) number density ~1e-9 cm^-3 x <sigma v>_up(Xe)
    sv_up_xe = up_xe_per_tyr / ATOMS_PER_T_XE / YR / n_chi     # cm^3/s per Xe atom
    Gamma_ISM = 1e-9 * sv_up_xe
    # cosmic rays: Phi_CR(>GeV) ~ 1 cm^-2 s^-1 x sigma_p
    Gamma_CR = 1.0 * SIGMA_P_N1[int(d)]
    # Sun: probability of crossing the Sun over t_U, times conversion per crossing on Fe (Fe/H = 3e-5, column 1.2e35 cm^-2, sigma_N(Fe) F^2 ~ 5e-36 -> ~2e-5)
    P_sun_cross = mom['v_mean'] * 1e5 * math.pi * (6.96e10) ** 2 * TU / (4 / 3 * math.pi * (3.1e22) ** 3)
    # Pb in the Earth (bulk 0.2 ppm by mass, recalled/uncertain): v_min = 577 km/s allowed; per-atom rate bounded by 30 x the Xe rate
    N_Pb = 0.2e-6 * 5.972e27 / (207.2 * 1.66054e-24)
    up_pb_per_atom_yr = 30. * up_xe_per_tyr / ATOMS_PER_T_XE
    P_conv_Pb = N_Pb * up_pb_per_atom_yr / flux_cross_per_yr
    rowsE2.append(dict(delta_keV=d, v_thr_Fe_kms=vmin_opt(56, 1000., d), v_thr_Pb_kms=vmin_opt(208, 1000., d), v_max_halo_kms=V_MAX_HALO,
                       Fe_upscatter_per_kgyr_sigmaN1=up_per_kgyr, N_up_Earth_core_per_yr=N_up_earth_per_yr,
                       chi1_crossings_per_yr=flux_cross_per_yr, P_conv_per_crossing=P_conv, f2_induced_Earth=f2_earth,
                       N_Pb_Earth=N_Pb, P_conv_Pb_upper=P_conv_Pb, f2_induced_Pb_upper=0.5 * P_conv_Pb,
                       Xe_upscatter_per_tyr=up_xe_per_tyr, P058_endo_ROI_check=1.0 / EXPOSURE_LZ, sigma_v_up_Xe_cm3_s=sv_up_xe,
                       Gamma_ISM_per_s=Gamma_ISM, f2_ISM_tU=Gamma_ISM * TU, Gamma_CR_per_s=Gamma_CR, f2_CR_tU=Gamma_CR * TU,
                       P_sun_crossing_tU=P_sun_cross, f2_Sun=P_sun_cross * 2e-5))
dfE2 = pd.DataFrame(rowsE2)
dfE2.to_csv(OUT + '/P094_repopulation.csv', index=False)
log('\n[E] re-population estimates:\n' + dfE2.to_string(index=False, float_format=lambda x: '%.3g' % x))
RES['E'] = dict(thresholds=dfE.to_dict(orient='records'), vmin_opt_300=thr, repop=dfE2.to_dict(orient='records'))

# ---- E2: halo self-up-scattering chi1 chi1 -> chi2 chi2 (t/u-channel A', both vertices off-diagonal) --------------
# NR contact limit: sigma_0 = c^2 mu^2/pi with c = g_D^2/m_A'^2, mu = m/2  ->  sigma_0 = 4 pi alpha_D^2 m^2 / m_A'^4;
# inelastic: sigma(v) = (sigma_0/2) int dcos (p_f/p_i) [m_A'^2/(m_A'^2 + q^2)]^2,  q^2 = p_i^2 + p_f^2 - 2 p_i p_f cos,
# p_i = mu v_rel, p_f = sqrt(p_i^2 - 2 mu (2 delta)).  Relative speeds: Maxwellian with v0_rel = sqrt(2) v0 (Galactic frame),
# truncated at 2 v_esc.  Identical-particle factors are O(1) and not included (flagged).
ALPHA_D_RELIC = {300.: 0.0073, 1000.: 0.0245, 3000.: 0.073}      # P011 (relic density), 1 TeV value used
def sigma_up_v(v_rel_kms, m, d, mA, alphaD):
    mu_ = m / 2
    pi_ = mu_ * v_rel_kms / C
    pf2 = pi_ * pi_ - 2 * mu_ * (2 * d * 1e-6)
    if pf2 <= 0:
        return 0.0
    pf = math.sqrt(pf2)
    sig0 = 4 * math.pi * alphaD ** 2 * m ** 2 / mA ** 4 * GEV2_TO_CM2
    ct = np.linspace(-1, 1, 201)
    q2 = pi_ ** 2 + pf ** 2 - 2 * pi_ * pf * ct
    prop = (mA ** 2 / (mA ** 2 + q2)) ** 2
    return sig0 * 0.5 * np.trapezoid(prop, ct) * pf / pi_
def f_rel(v, v0=lz.V0_KMS, vesc=lz.VESC_KMS):
    v0r = math.sqrt(2) * v0
    f = v * v * np.exp(-(v / v0r) ** 2) * (v < 2 * vesc)
    return f / np.trapezoid(f, v)
VREL = np.linspace(0.1, 2 * lz.VESC_KMS, 2000)
FREL = f_rel(VREL)
def gamma_up(m, d, mA, alphaD):
    sv = np.array([sigma_up_v(v, m, d, mA, alphaD) * v * 1e5 for v in VREL])   # cm^3/s
    return RHO0 / m * np.trapezoid(FREL * sv, VREL)                            # s^-1 per chi1 (f1 ~ 1)
frac_above = {d: float(np.trapezoid(FREL[VREL > math.sqrt(8 * d * 1e-6 / 1000.) * C], VREL[VREL > math.sqrt(8 * d * 1e-6 / 1000.) * C])) for d in (300., 350., 380.)}
v_rel_mean = float(np.trapezoid(FREL * VREL, VREL))
rowsE3 = []
for d in (300., 350., 380.):
    tau0 = P058_TAU0_1GEV[int(d)]
    for mA in (2.4, 2.6, 2.9, 3.0, 3.5, 4.0, 5.0, 7.0, 10.0):
        for aD in (ALPHA_D_RELIC[1000.], 0.1):
            G = gamma_up(1000., d, mA, aD)
            tau_ = tau0 * mA ** -4
            f2ss = G * tau_ * (1 - math.exp(-TU / tau_))
            f2prim = F2_FO['dark photon'] * math.exp(-TU / tau_)
            sig0 = 4 * math.pi * aD ** 2 * 1000. ** 2 / mA ** 4 * GEV2_TO_CM2
            rowsE3.append(dict(delta_keV=d, m_Aprime_GeV=mA, alpha_D=aD, sigma0_cm2=sig0, sigma_over_m_cm2_g=sig0 / (1000. * 1.7827e-24),
                               Gamma_up_per_s=G, tau_s=tau_, f2_steady_state=f2ss, f2_primordial=f2prim, f2_total=f2ss + f2prim,
                               LZ_exo_events_2p84tyr_ss=P058_R_ROI_P[int(d)] * f2ss, LZ_exo_events_2p84tyr_prim=P058_R_ROI_P[int(d)] * f2prim))
dfE3 = pd.DataFrame(rowsE3)
dfE3.to_csv(OUT + '/P094_self_upscatter.csv', index=False)
# Higgsino: Z exchange, c ~ G_F/sqrt(2) x O(1), tau_nunu = 1.2e6 s (P026)
sig_hig = (1.1664e-5 / math.sqrt(2)) ** 2 * (500.) ** 2 / math.pi * GEV2_TO_CM2
G_hig = RHO0 / 1000. * sig_hig * frac_above[300.] * 0.6 * v_rel_mean * 1e5
log('\n[E2] halo self-up-scattering: <v_rel> = %.0f km/s, fraction above threshold (1 TeV) = %s' % (v_rel_mean, {k: round(v, 3) for k, v in frac_above.items()}))
log(dfE3[(dfE3.alpha_D < 0.05)].to_string(index=False, float_format=lambda x: '%.3g' % x))
log('[E2] Higgsino: sigma_self ~ %.1e cm^2, Gamma_up ~ %.1e /s, f2_ss = Gamma tau_nunu = %.1e' % (sig_hig, G_hig, G_hig * 1.2e6))
RES['E2'] = dict(v_rel_mean=v_rel_mean, frac_above_threshold=frac_above, table=dfE3.to_dict(orient='records'),
                 higgsino=dict(sigma_cm2=sig_hig, Gamma=G_hig, f2_ss=G_hig * 1.2e6))
# crossover m_A' where the steady-state population exceeds the primordial remnant (delta = 300, relic alpha_D)
mAs = np.linspace(2.0, 6.0, 401)
ss = np.array([gamma_up(1000., 300., x, ALPHA_D_RELIC[1000.]) * P058_TAU0_1GEV[300] * x ** -4 for x in mAs])
pr = F2_FO['dark photon'] * np.exp(-TU / (P058_TAU0_1GEV[300] * mAs ** -4))
ix = int(np.argmax(ss > pr))
RES['E2']['mA_crossover_GeV_300'] = float(mAs[ix])
log('[E2] steady-state exceeds primordial for m_A\' > %.2f GeV (delta = 300 keV, alpha_D = 0.0245)' % mAs[ix])

# =============================================================================
# PART F: combined (tau, f2) map
# =============================================================================
tau = np.logspace(14, 24, 600)
f2_dp = F2_FO['dark photon'] * np.exp(-TU / tau)
f2_hig = F2_FO['Higgsino'] * np.exp(-TU / tau)
def tau_of_f2(f2, f2fo=0.5):
    return TU / math.log(f2fo / f2)
def mA_of_tau(tau_s, d):
    return (P058_TAU0_1GEV[d] / tau_s) ** 0.25
rowsF = []
for f2v in (2.8e-3, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8):
    t = tau_of_f2(f2v)
    rowsF.append(dict(f2_today=f2v, tau_s=t, tau_over_tU=t / TU, mA_GeV_300=mA_of_tau(t, 300), mA_GeV_350=mA_of_tau(t, 350), mA_GeV_380=mA_of_tau(t, 380)))
dfF = pd.DataFrame(rowsF)
dfF.to_csv(OUT + '/P094_f2_tau_map.csv', index=False)
log('\n[F] f2(today) <-> tau <-> m_A\' (dark photon, f2,fo = 0.5):\n' + dfF.to_string(index=False, float_format=lambda x: '%.3g' % x))
# electron-line f2 limit in the DP model (no constraint) and the f2 sigma_e sensitivity
c_row = dfC[(dfC.bkg_case == 'central') & (dfC.E_keV == 300) & (dfC.experiment == 'LZ 2.84 t yr')].iloc[0]
RES['F'] = dict(table=dfF.to_dict(orient='records'), f2_sigma_e_LZ_300=float(c_row.f2_sigma_e_limit_cm2),
                f2_limit_DP_300_from_ER_line=float(c_row.f2_limit_DP_model),
                DEAP_f2_sens_300=float(dfX[(dfX.experiment.str.startswith('DEAP')) & (dfX.delta_keV == 300)].f2_sensitivity_if_zero_events_full_spectrum.iloc[0]),
                DEAP_f2_sens_380=float(dfX[(dfX.experiment.str.startswith('DEAP')) & (dfX.delta_keV == 380)].f2_sensitivity_if_zero_events_full_spectrum.iloc[0]))

# =============================================================================
# FIGURES (palette from the dataviz reference: blue, orange, aqua, yellow, magenta, green, violet)
# =============================================================================
PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e',
                     'ytick.color': '#52514e', 'grid.color': '#e1e0d9', 'axes.grid': True, 'grid.linewidth': 0.6})

# Fig 1: exothermic spectra per kg yr on the targets (delta = 300), plus electron line inset-like panel
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
ax = axs[0]
for i, tname in enumerate(['Xe', 'Ar', 'Ge', 'Si', 'O', 'W']):
    r = spectra[(tname, 'p', 300.)] / 1000.
    ax.plot(EGRID, r, color=PAL[i], lw=2, label=tname)
ax.set_yscale('log')
ax.set_xlim(0, 700)
ymax = max(np.max(spectra[(t, 'p', 300.)]) for t in ['Xe', 'Ar', 'Ge', 'Si', 'O', 'W']) / 1000.
ax.set_ylim(ymax * 1e-5, ymax * 3)
ax.axvspan(0, 16, color='#e1e0d9', alpha=0.6, lw=0)
ax.axvspan(5.4, 270, color='#cde2fb', alpha=0.35, lw=0)
ax.text(140, ymax * 1.5, 'LZ WS ROI (Xe)', ha='center', color='#52514e', fontsize=8)
ax.annotate('low-threshold ROIs\nend below ~16 keV', xy=(16, ymax * 2e-4), xytext=(60, ymax * 1e-4), fontsize=7, color='#52514e',
            arrowprops=dict(arrowstyle='-', color='#898781', lw=0.8))
ax.set_xlabel('nuclear recoil energy [keV]')
ax.set_ylabel('dR/dE per kg yr per unit f$_2$  [keV$^{-1}$]')
ax.set_title('Exothermic $\\chi_2 N \\to \\chi_1 N$, $\\delta$ = 300 keV, 1 TeV, $\\sigma_p$(N=1) (P011)', fontsize=9)
ax.legend(frameon=False, ncol=2, fontsize=8)
ax = axs[1]
E = np.linspace(250, 430, 2000)
for i, d in enumerate((300, 350, 380)):
    sig = math.sqrt(sigma_det(d) ** 2 + sigma_doppler[d] ** 2)
    line = stats.norm.pdf(E, d, sig)
    ax.plot(E, line / line.max(), color=PAL[i], lw=2, label='$\\delta$ = %d keV, $\\sigma_E$ = %.1f keV' % (d, sig))
    dop = stats.norm.pdf(E, d, sigma_doppler[d])
    ax.plot(E, dop / dop.max(), color=PAL[i], lw=1, ls=':')
ax.set_xlabel('deposited energy [keV]')
ax.set_ylabel('line shape (normalised)')
ax.set_title('ER line $\\chi_2 e \\to \\chi_1 e$: Doppler width (dotted) vs LXe resolution', fontsize=9)
ax.legend(frameon=False, fontsize=8)
ax.text(255, 0.9, 'S1c = %.0f-%.0f phd\nlog$_{10}$S2c = %.2f-%.2f\n(outside WS ROI)' % (RES['A']['ER_300keV_S1c_phd'], RES['A']['ER_380keV_S1c_phd'],
        RES['A']['ER_300keV_log10S2c'], RES['A']['ER_380keV_log10S2c']), fontsize=7.5, va='top', color='#52514e')
fig.tight_layout()
fig.savefig(FIG + '/P094_fig1_spectra_and_line.png', dpi=160)
plt.close(fig)

# Fig 2: (tau, f2) map
fig, ax = plt.subplots(figsize=(6.4, 4.6))
ax.plot(tau, f2_dp, color=PAL[0], lw=2, label='P026 freeze-out: $f_2 = 0.50\\,e^{-t_U/\\tau}$ (dark photon)')
ax.plot(tau, f2_hig, color=PAL[0], lw=1.2, ls='--', label='$f_2 = 0.42\\,e^{-t_U/\\tau}$ (Higgsino)')
for i, (d, lim) in enumerate(P058_F2_P.items()):
    if d == 366:
        continue
    ax.axhspan(lim, 1, color=PAL[1], alpha=0.08, lw=0)
    ax.axhline(lim, color=PAL[1], lw=1.2)
    ax.text(9e23, lim * 1.35, 'P058 LZ nuclear, $\\delta$ = %d keV: $f_2$ < %.1e' % (d, lim), fontsize=7, color='#52514e', ha='right')
ax.plot(tau, 5e-23 * tau, color=PAL[2], lw=1.5)
ax.plot(tau, 5e-24 * tau, color=PAL[2], lw=1, ls=':')
ax.text(3e18, 1.5e-9, 'P066 SPI line ($\\gamma$ channel, BR$_\\gamma$ = 1):\nexcluded above $f_2 = 5\\times10^{-23}\\,\\tau_\\gamma$', fontsize=7, color='#52514e')
# steady-state halo re-population in the dark-photon model (tau <-> m_A' at delta = 300 keV)
tau_ss = np.logspace(14, 18.5, 60)
mA_ss = (P058_TAU0_1GEV[300] / tau_ss) ** 0.25
f2_ss_curve = np.array([gamma_up(1000., 300., x, ALPHA_D_RELIC[1000.]) for x in mA_ss]) * tau_ss * (1 - np.exp(-TU / tau_ss))
ax.plot(tau_ss, f2_ss_curve, color=PAL[4], lw=1.8, ls='-.', label='this work: halo $\\chi_1\\chi_1\\to\\chi_2\\chi_2$ steady state, $\\alpha_D$ = 0.0245')
ax.axvspan(2e12, 1e19, ymin=0, ymax=0.03, color=PAL[3], alpha=0.5, lw=0)
ax.text(1.5e14, 2e-12, 'P026 Planck ($\\gamma$ channel, $\\tau_\\gamma$ = 2e12-1e19 s)', fontsize=7, color='#52514e')
ax.axhline(RES['F']['DEAP_f2_sens_300'], color=PAL[6], lw=1.2, ls='-.')
ax.text(1.3e14, RES['F']['DEAP_f2_sens_300'] * 0.4, 'this work: Ar 2.1 t yr, 0 events, $\\delta$ = 300 keV (projection)', fontsize=7, color=PAL[6])
ax.axvline(TU, color='#898781', lw=0.8, ls=':')
ax.text(TU * 1.15, 3e-11, '$t_U$', fontsize=8, color='#52514e')
for f2v in (2.8e-3, 1e-5):
    t = tau_of_f2(f2v)
    ax.plot([t], [f2v], 'o', color=PAL[0], ms=6, mec='#fcfcfb')
ax.annotate('$f_2$ = 3e-3 to 1e-5 spans only\n$\\tau$ = 8.4e16 to 4.0e16 s\n($m_{A\\prime}$ = 2.4-2.9 GeV at 300 keV)', xy=(tau_of_f2(1e-5), 1e-5),
            xytext=(1.5e14, 1e-8), fontsize=7.5, color='#0b0b0b', arrowprops=dict(arrowstyle='-', color='#898781', lw=0.8))
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1e14, 1e24)
ax.set_ylim(1e-12, 1)
ax.set_xlabel('$\\chi_2$ lifetime $\\tau$ [s]')
ax.set_ylabel('excited fraction today $f_2$')
ax.set_title('Where a populated excited state can still hide (1 TeV)', fontsize=9)
ax.legend(frameon=False, fontsize=7.5, loc='lower right')
fig.tight_layout()
fig.savefig(FIG + '/P094_fig2_f2_tau_map.png', dpi=160)
plt.close(fig)

RES['runtime_s'] = time.time() - T0
with open(OUT + '/P094_results.json', 'w') as f:
    json.dump(RES, f, indent=1, default=float)
log('\nDone in %.0f s' % RES['runtime_s'])
