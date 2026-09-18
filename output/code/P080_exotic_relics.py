"""
P080 -- Millicharged particles, strongly interacting massive relics and other exotic heavy
particles as the source of a single 248 keV xenon recoil (LZ arXiv:2609.02823).

Run from the simulation root:  .venv/bin/python output/code/P080_exotic_relics.py

Parts
  A  millicharged particles (MCP): halo Rutherford spectrum, N_lo, epsilon for one event,
     stopping in atmosphere/rock, magnetic deflection, relativistic (cosmic-ray-produced) MCPs
  B  SIMPs / composites: mean free path, exactly-one-scatter probability (cylinder chord MC),
     spectral fractions (window / low ROI / HE sideband), flux and mass for one event,
     overburden transparency, continuous-track objects (nuclearites, Q-balls, CHAMPs, monopoles)
  C  flux ceiling from N = 1 in 2.84 t yr and the rho/m bound (superheavy / Planck-scale DM)
  D  inelastic composites (dark nuclei): delta_max vs mass and the mass ceiling
Outputs: output/work/P080/*.csv, *.json, figures/*.png
"""
from __future__ import annotations
import sys, os, json, math
sys.path.insert(0, 'output/code')
import numpy as np
import pandas as pd
from scipy import special, integrate, stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import lzcommon as lz

OUT = 'output/work/P080'
FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)
RES = {}

# ----------------------------------------------------------------------------------
# 0. Constants, detector geometry, efficiency, nuclear responses
# ----------------------------------------------------------------------------------
ALPHA = 1 / 137.035999          # fine-structure constant (recalled, certain)
HBARC2_CM2_GEV2 = lz.GEV_TO_CM2  # 3.894e-28 cm^2 GeV^2
C_CMS = lz.C_KMS * 1e5
Z_XE = 54
RHO_LXE = 2.9                    # g/cm^3 (lzcommon NEST density)
AMU_G = 1.66054e-24
N_XE = RHO_LXE / (lz.A_XE_MEAN * AMU_G)          # Xe nuclei per cm^3
LIVE_S = lz.LZ['live_days'] * 86400.0
M_FV_G = lz.LZ['fiducial_mass_t'] * 1e6
N_T_FV = M_FV_G / (lz.A_XE_MEAN * AMU_G)          # nuclei in the FV
EXPO_TYR = lz.LZ['exposure_tyr']

# TPC: 1.456 m diameter x 1.456 m drift (recalled from the LZ instrument paper, likely).
R_TPC, H_TPC = 72.8, 145.6
# FV: paper gives z from 9.0 cm above cathode to 12.8 cm below gate; radius from the 4.71 t mass
H_FV = H_TPC - 9.0 - 12.8
V_FV = M_FV_G / RHO_LXE
R_FV = math.sqrt(V_FV / (math.pi * H_FV))
S_FV = 2 * math.pi * R_FV**2 + 2 * math.pi * R_FV * H_FV
LBAR_FV = 4 * V_FV / S_FV
V_TPC = math.pi * R_TPC**2 * H_TPC
S_TPC = 2 * math.pi * R_TPC**2 + 2 * math.pi * R_TPC * H_TPC
LBAR_TPC = 4 * V_TPC / S_TPC
RES['geometry'] = dict(n_Xe_cm3=N_XE, N_T_FV=N_T_FV, R_FV_cm=R_FV, H_FV_cm=H_FV, S_FV_cm2=S_FV,
                       Lbar_FV_cm=LBAR_FV, R_TPC_cm=R_TPC, H_TPC_cm=H_TPC, S_TPC_cm2=S_TPC,
                       Lbar_TPC_cm=LBAR_TPC, live_s=LIVE_S, M_TPC_t=V_TPC * RHO_LXE / 1e6)

# Efficiency model (P003 constants: plateau 0.96, erf roll-offs to 50 % at 5.4 and 269.9 keV)
E50_LO, E50_HI = lz.LZ['E_50pct_low_keV'], lz.LZ['E_50pct_high_keV']
def eff(E, sig_lo=3.4, sig_hi=8.0, eff0=lz.LZ['eff_plateau']):
    E = np.asarray(E, float)
    s_lo = 0.5 * (1 + special.erf((E - E50_LO) / (math.sqrt(2) * sig_lo)))
    s_hi = 0.5 * (1 - special.erf((E - E50_HI) / (math.sqrt(2) * sig_hi)))
    return eff0 * s_lo * s_hi

# windows (keV)
LO = (5.4, 55.0); WIN = (200.0, 270.0)

# shell-model M response (P017 WimPyDD tables, natural Xe, normalised to 1 at q=0)
d17 = np.load('output/work/P017/P017_response_curves.npz')
E17, F2S17, F2H17 = d17['E_grid'], d17['F2_shell'], d17['F2_helm']
def F2_shell(E):
    return np.interp(E, E17, F2S17, left=1.0, right=F2S17[-1])
def F2_helm_nat(E):
    E = np.atleast_1d(np.asarray(E, float))
    w = np.array([f * A**2 for A, f in lz.XE_ISOTOPES.items()]); w /= w.sum()
    out = np.zeros_like(E)
    for (A, f), wi in zip(lz.XE_ISOTOPES.items(), w):
        out += wi * np.array([lz.helm_F2(e, A) for e in E])
    return out
RES['F2_check'] = dict(F2_30_over_248_helm=float(F2_helm_nat(30)[0] / F2_helm_nat(248)[0]),
                       F2_30_over_248_shell=float(F2_shell(30) / F2_shell(248)),
                       F2_248_helm=float(F2_helm_nat(248)[0]), F2_248_shell=float(F2_shell(248)))

def win_int(E, y, lo, hi, use_eff=True):
    m = (E >= lo) & (E <= hi)
    Ef = np.linspace(lo, hi, 2001)
    yf = np.interp(Ef, E, y)
    if use_eff:
        yf = yf * eff(Ef)
    return np.trapezoid(yf, Ef)

# ----------------------------------------------------------------------------------
# A. Millicharged particles
# ----------------------------------------------------------------------------------
# Rutherford (non-relativistic, heavy projectile, screening irrelevant for E_R > keV):
#   dsigma/dE_R = 2 pi Z^2 eps^2 alpha^2 (hbar c)^2 / (m_N v^2 E_R^2) * F^2(E_R)
# halo rate: dR/dE = N_T (rho/m) [2 pi Z^2 eps^2 alpha^2 (hbar c)^2 c^2 / (m_N E_R^2)] F^2 eta(v_min)
def dRdE_mcp(E_keV, m_gev, eps, shell=False, v_e=None):
    """events / (t yr keV) for natural Xe (Z = 54 all isotopes)."""
    E = np.atleast_1d(np.asarray(E_keV, float))
    tot = np.zeros_like(E)
    for A, f in lz.XE_ISOTOPES.items():
        mN = lz.m_nucleus_gev(A)
        vmin = np.array([lz.vmin_kms(e, m_gev, A) for e in E])
        eta = np.atleast_1d(lz.eta0(vmin, v_e=v_e)) * 1e-5        # s/cm
        F2 = F2_shell(E) if shell else np.array([lz.helm_F2(e, A) for e in E])
        n_T = 1e6 / (A * AMU_G) * f
        pref = 2 * math.pi * Z_XE**2 * eps**2 * ALPHA**2 * HBARC2_CM2_GEV2 * C_CMS**2 / (mN * (E * 1e-6)**2)  # cm^2 cm^2/s^2 /GeV
        r = n_T * (lz.RHO0_GEV_CM3 / m_gev) * pref * F2 * eta      # per s per GeV
        tot += r * 3.15576e7 * 1e-6                                # per yr per keV
    return tot

# unit check of the same machinery against lz.dRdE_SI (contact SI, Helm)
def dRdE_contact_check(E_keV, m_gev, sig_n):
    E = np.atleast_1d(np.asarray(E_keV, float)); tot = np.zeros_like(E)
    for A, f in lz.XE_ISOTOPES.items():
        mN = lz.m_nucleus_gev(A); mu_n = lz.mu_red(m_gev, lz.M_NUCLEON_GEV); mu_N = lz.mu_red(m_gev, mN)
        sigN = sig_n * (mu_N / mu_n)**2 * A**2
        vmin = np.array([lz.vmin_kms(e, m_gev, A) for e in E])
        eta = np.atleast_1d(lz.eta0(vmin)) * 1e-5
        F2 = np.array([lz.helm_F2(e, A) for e in E]); n_T = 1e6 / (A * AMU_G) * f
        tot += n_T * (lz.RHO0_GEV_CM3 / m_gev) * sigN * mN / (2 * mu_N**2) * C_CMS**2 * F2 * eta * 3.15576e7 * 1e-6
    return tot
Echk = np.array([10., 50., 100., 248.])
chk = dRdE_contact_check(Echk, 1000., 1e-45) / np.array([lz.dRdE_SI(e, 1000., 1e-45) for e in Echk])
RES['unit_check_contact_vs_lzcommon'] = chk.tolist()

Egrid = np.concatenate([np.linspace(0.5, 60, 600), np.linspace(60.5, 300, 480)])
m_min_248 = lz.m_chi_min_gev(248.0)
m_min_200 = lz.m_chi_min_gev(200.0)
RES['mcp_m_min_GeV'] = dict(E248=m_min_248, E200=m_min_200)

rows = []
for m in [60, 100, 200, 400, 1000, 4000, 10000]:
    for shell in (False, True):
        y = dRdE_mcp(Egrid, m, 1.0, shell=shell)
        R_lo = win_int(Egrid, y, *LO); R_win = win_int(Egrid, y, *WIN)
        R_all = win_int(Egrid, y, 0.5, 300, use_eff=False)
        if R_win > 0:
            eps1 = math.sqrt(1.0 / (R_win * EXPO_TYR))          # eps for one window event
        else:
            eps1 = float('nan')
        Ek_keV = 0.5 * m * 1e6 * (700 / lz.C_KMS)**2
        rows.append(dict(m_GeV=m, response='shell' if shell else 'Helm', R_lo_unit=R_lo, R_win_unit=R_win,
                         N_lo=R_lo / R_win if R_win > 0 else float('inf'), eps_one_event=eps1,
                         N_lo_events_at_eps1=R_lo * eps1**2 * EXPO_TYR, KE_700kms_keV=Ek_keV))
mcp = pd.DataFrame(rows); mcp.to_csv(f'{OUT}/P080_mcp_halo.csv', index=False)
# equivalent contact-SI cross-section: sigma_n at which O1 (Helm) gives the same 5.4-55 keV rate as the MCP at eps_1 (1 TeV)
_r = mcp[(mcp.m_GeV == 1000) & (mcp.response == 'Helm')].iloc[0]
_R_lo_O1 = win_int(Egrid, np.array([lz.dRdE_SI(e, 1000., 1e-45) for e in Egrid]), *LO)
RES['mcp_equivalent_sigma_SI_1TeV_cm2'] = float(_r.R_lo_unit * _r.eps_one_event**2 / _R_lo_O1 * 1e-45)
RES['mcp_lowE_events_at_eps1_1TeV'] = float(_r.N_lo_events_at_eps1)
# reference: contact O1 Helm N_lo with the same grids (compare P003/P030/P040)
y_o1 = np.array([lz.dRdE_SI(e, 1000., 1e-45) for e in Egrid])
RES['N_lo_O1_helm_1TeV_check'] = float(win_int(Egrid, y_o1, *LO) / win_int(Egrid, y_o1, *WIN))
# beam-like (v-independent) 1/E^2 x F^2 floor (relativistic MCPs: eta -> const)
y_flat_h = F2_helm_nat(Egrid) / Egrid**2; y_flat_s = F2_shell(Egrid) / Egrid**2
RES['N_lo_relativistic_MCP'] = dict(helm=float(win_int(Egrid, y_flat_h, *LO) / win_int(Egrid, y_flat_h, *WIN)),
                                    shell=float(win_int(Egrid, y_flat_s, *LO) / win_int(Egrid, y_flat_s, *WIN)))

# --- MCP stopping in the overburden for the eps that gives one event -------------------
# nuclear (Rutherford) stopping: dE/dx = n sum_i 2 pi Z_i^2 eps^2 alpha^2 (hbarc)^2 /(m_i v^2) ln(Emax_i/Emin_i)
# Emax = 2 mu^2 v^2/m_i ; Emin from screening q_min = hbar/a_TF, a_TF = 0.885 a0 Z^-1/3 (recalled, likely)
ROCK = dict(O=(8, 16, 0.47), Si=(14, 28, 0.28), Al=(13, 27, 0.08), Fe=(26, 56, 0.05), Ca=(20, 40, 0.04),
            K=(19, 39, 0.03), Na=(11, 23, 0.03), Mg=(12, 24, 0.02))   # crustal mass fractions (recalled, likely)
AIR = dict(N=(7, 14, 0.755), O=(8, 16, 0.232), Ar=(18, 40, 0.013))
X_ROCK = 1478e2 * 2.7          # g/cm^2 (SURF depth 1478 m, standard rock 2.7 g/cm^3; recalled, likely)
X_AIR = 1033.0                 # g/cm^2 (recalled, certain)
A0_CM = 0.529e-8
def rutherford_loss_gev_per_gcm2(m_gev, eps, v_kms, comp):
    beta = v_kms / lz.C_KMS; tot = 0.0
    for Z, A, w in comp.values():
        mA = A * lz.AMU_GEV; mu = lz.mu_red(m_gev, mA)
        Emax = 2 * mu**2 * beta**2 / mA                                  # GeV
        a_tf = 0.885 * A0_CM * Z**(-1 / 3)
        qmin = math.sqrt(HBARC2_CM2_GEV2) / a_tf                          # GeV
        Emin = qmin**2 / (2 * mA)
        L = math.log(max(Emax / Emin, 1.0 + 1e-12))
        n_per_g = w / (A * AMU_G)
        tot += n_per_g * 2 * math.pi * Z**2 * eps**2 * ALPHA**2 * HBARC2_CM2_GEV2 / (mA * beta**2) * L
    return tot   # GeV per g/cm^2
# electronic stopping (Lindhard-Scharff, slow point charge; eps^2 scaling is an estimate, uncertain):
# S_e = 8 pi e^2 a0 Z1^(7/6) Z2 /(Z1^(2/3)+Z2^(2/3))^(3/2) * v/v0 per atom, with Z1 -> eps and eps^(7/6) -> eps^2
def lindhard_electronic_gev_per_gcm2(Z1, v_kms, comp, charge_scaling='LS'):
    v0 = ALPHA * lz.C_KMS; tot = 0.0
    for Z2, A, w in comp.values():
        if charge_scaling == 'LS':
            zf = Z1**(7 / 6) * Z2 / (Z1**(2 / 3) + Z2**(2 / 3))**1.5
        else:  # point charge Z1 << 1: eps^2 times the Z1=1 value
            zf = Z1**2 * Z2 / (1 + Z2**(2 / 3))**1.5
        Se = 8 * math.pi * (1.44e-7 * 1e-9) * A0_CM * zf * (v_kms / v0)   # eV cm^2 ... e^2 = 1.44 eV nm = 1.44e-7 eV cm
        # careful with units: e^2 = 1.44 eV nm = 1.44e-7 eV cm ; a0 in cm -> e^2 a0 in eV cm^2
        Se = 8 * math.pi * 1.44e-7 * A0_CM * zf * (v_kms / v0)             # eV cm^2 per atom
        tot += Se * 1e-9 * w / (A * AMU_G)                                 # GeV per g/cm^2
    return tot

srows = []
for m in [100, 1000, 10000]:
    eps1 = float(mcp[(mcp.m_GeV == m) & (mcp.response == 'Helm')].eps_one_event.iloc[0])
    KE = 0.5 * m * (700 / lz.C_KMS)**2                                   # GeV
    dE_nuc = rutherford_loss_gev_per_gcm2(m, eps1, 700, ROCK) * X_ROCK + rutherford_loss_gev_per_gcm2(m, eps1, 700, AIR) * X_AIR
    dE_el = lindhard_electronic_gev_per_gcm2(eps1, 700, ROCK, 'point') * X_ROCK
    # Larmor radii: r = p/(eps e B); p[GeV] -> r[m] = p/(0.2998 eps B[T])
    p_gev = m * 700 / lz.C_KMS
    rL_earth_km = p_gev / (0.2998 * eps1 * 3e-5) / 1e3
    rL_helio_AU = p_gev / (0.2998 * eps1 * 5e-9) / 1.496e11
    # eps at which the heliospheric Larmor radius equals 1 AU and at which rock stops the particle (dE = KE)
    eps_helio = p_gev / (0.2998 * 5e-9 * 1.496e11)
    eps_stop = eps1 * math.sqrt(KE / (dE_nuc + dE_el))
    srows.append(dict(m_GeV=m, eps_one_event=eps1, KE_700_GeV=KE, dE_nuclear_GeV=dE_nuc, dE_electronic_est_GeV=dE_el,
                      frac_loss=(dE_nuc + dE_el) / KE, rL_Earth_km=rL_earth_km, rL_helio_AU=rL_helio_AU,
                      eps_rL_1AU=eps_helio, eps_stopped_in_overburden=eps_stop))
stop = pd.DataFrame(srows); stop.to_csv(f'{OUT}/P080_mcp_stopping.csv', index=False)

# --- relativistic MCPs (cosmic-ray / supernova accelerated): beta -> 1, heavy projectile ---------
def sigma_window_rel(eps, lo, hi, beta=1.0, shell=False, use_eff=True):
    E = np.linspace(lo, hi, 2001)
    F2 = F2_shell(E) if shell else F2_helm_nat(E)
    mN = lz.m_nucleus_gev(lz.A_XE_MEAN)
    ds = 2 * math.pi * Z_XE**2 * eps**2 * ALPHA**2 * HBARC2_CM2_GEV2 / (mN * beta**2 * (E * 1e-6)**2) * F2 * 1e-6  # cm^2/keV
    if use_eff:
        ds = ds * eff(E)
    return np.trapezoid(ds, E)
sig_win_e1 = sigma_window_rel(1.0, *WIN)          # cm^2 per eps^2
sig_lo_e1 = sigma_window_rel(1.0, *LO)
phi_eps2 = 1.0 / (N_T_FV * sig_win_e1 * LIVE_S)   # eps^2 * Phi for one event, cm^-2 s^-1
# NR/ER at the same deposited energy: dsigma_N/dsigma_e = Z^2/(Z) * m_e/m_N (Rutherford on Z electrons)
nr_over_er = Z_XE * 0.000511 / lz.m_nucleus_gev(lz.A_XE_MEAN)
# ionisation track: dE/dx_min(LXe) ~ 1.25 MeV/(g/cm^2) (recalled, likely) * eps^2 over the chord
dEdx_min = 1.25e-3  # GeV/(g/cm^2)
RES['mcp_relativistic'] = dict(sigma_win_per_eps2_cm2=sig_win_e1, sigma_lo_per_eps2_cm2=sig_lo_e1,
                               N_lo=sig_lo_e1 / sig_win_e1, Phi_eps2_one_event_cm2s=phi_eps2,
                               NR_over_ER_same_energy=nr_over_er, ER_per_NR=1 / nr_over_er,
                               track_keV_per_eps2_1p3m=dEdx_min * RHO_LXE * LBAR_TPC * 1e6,
                               eps_for_track_1keV=math.sqrt(1e-6 / (dEdx_min * RHO_LXE * LBAR_TPC)),
                               overburden_loss_GeV_per_eps2=dEdx_min * 2 * (X_ROCK + X_AIR))  # ~2x min for e-loss at high gamma

# ----------------------------------------------------------------------------------
# B. SIMPs and composite relics
# ----------------------------------------------------------------------------------
lam = lambda sig: 1.0 / (N_XE * sig)                      # cm
sig_lam_TPC = 1.0 / (N_XE * LBAR_TPC)
sig_lam_146 = 1.0 / (N_XE * H_TPC)
RES['simp_sigma_for_lambda'] = dict(sigma_lambda_eq_Lbar_TPC_cm2=sig_lam_TPC, sigma_lambda_eq_146cm_cm2=sig_lam_146,
                                    Lbar_TPC_cm=LBAR_TPC)

# chord-length MC for an isotropic flux through the TPC cylinder (scatter position uniform along the chord)
rng = np.random.default_rng(80)
def cylinder_chords(n, R, H):
    # sample entry points on the surface weighted by area and directions by cos-law
    A_side = 2 * math.pi * R * H; A_cap = math.pi * R**2
    which = rng.choice(3, size=n, p=[A_side, A_cap, A_cap] / np.array(A_side + 2 * A_cap))
    # local frame: inward normal; cos-law direction
    u1, u2 = rng.random(n), rng.random(n)
    ct = np.sqrt(u1); st = np.sqrt(1 - ct**2); ph = 2 * math.pi * u2
    # positions
    x = np.zeros(n); y = np.zeros(n); z = np.zeros(n); dx = np.zeros(n); dy = np.zeros(n); dz = np.zeros(n)
    s = which == 0
    phi_s = 2 * math.pi * rng.random(s.sum())
    x[s] = R * np.cos(phi_s); y[s] = R * np.sin(phi_s); z[s] = H * rng.random(s.sum())
    # inward normal (-cos phi, -sin phi, 0); build direction
    nx, ny = -np.cos(phi_s), -np.sin(phi_s)
    # tangent vectors: t1 = (0,0,1), t2 = n x t1 = (ny, -nx, 0)
    dx[s] = ct[s] * nx + st[s] * np.cos(ph[s]) * 0 + st[s] * np.sin(ph[s]) * ny
    dy[s] = ct[s] * ny + st[s] * np.sin(ph[s]) * (-nx)
    dz[s] = st[s] * np.cos(ph[s])
    for k, (zc, sgn) in ((1, (H, -1.0)), (2, (0.0, 1.0))):
        c = which == k; nc = c.sum()
        rr = R * np.sqrt(rng.random(nc)); pp = 2 * math.pi * rng.random(nc)
        x[c] = rr * np.cos(pp); y[c] = rr * np.sin(pp); z[c] = zc
        dz[c] = sgn * ct[c]; dx[c] = st[c] * np.cos(ph[c]); dy[c] = st[c] * np.sin(ph[c])
    # chord length: intersect with cylinder
    # radial: |(x,y) + t (dx,dy)| = R  -> a t^2 + b t + c = 0
    a = dx**2 + dy**2; b = 2 * (x * dx + y * dy); c = x**2 + y**2 - R**2
    with np.errstate(divide='ignore', invalid='ignore'):
        t_rad = np.where(a > 1e-12, (-b + np.sqrt(np.maximum(b**2 - 4 * a * c, 0))) / (2 * a), np.inf)
        t_z = np.where(dz > 0, (H - z) / dz, np.where(dz < 0, -z / dz, np.inf))
    t = np.minimum(t_rad, t_z)
    t = np.where(t < 1e-9, np.maximum(t_rad, 1e-9), t)
    return t
chords = cylinder_chords(200000, R_TPC, H_TPC)
RES['chord_MC'] = dict(mean_cm=float(chords.mean()), analytic_4V_over_S=LBAR_TPC, median_cm=float(np.median(chords)),
                       p10=float(np.percentile(chords, 10)), p90=float(np.percentile(chords, 90)))

# probability of exactly one visible scatter per traversal, averaged over chords
sig_grid = np.logspace(-27, -19, 193)
mu_grid = N_XE * sig_grid[:, None] * chords[None, :]
P1 = np.mean(mu_grid * np.exp(-mu_grid), axis=1)
P0 = np.mean(np.exp(-mu_grid), axis=1)
Pmulti = 1 - P0 - P1
imax = int(np.argmax(P1))
RES['simp_P1'] = dict(sigma_at_max_P1_cm2=float(sig_grid[imax]), P1_max=float(P1[imax]),
                      P1_at_sigma_lambda_TPC=float(np.interp(sig_lam_TPC, sig_grid, P1)),
                      Pmulti_at_sigma_lambda_TPC=float(np.interp(sig_lam_TPC, sig_grid, Pmulti)))
pd.DataFrame(dict(sigma_N_cm2=sig_grid, lambda_cm=lam(sig_grid), P0=P0, P1=P1, Pmulti=Pmulti)).to_csv(
    f'{OUT}/P080_simp_P1.csv', index=False)

# --- spectral fractions for a heavy (m >> m_N) relic at SHM velocities -------------------------
# flat-in-E_R (isotropic CM) spectrum up to Emax(v) = 2 m_N v^2 ; SHM-averaged: dR/dE ∝ ∫_{v>v_min} f(v)/v dv = eta(v_min)
# with v_min = sqrt(E/(2 m_N)) for m >> m_N ; case (i) pointlike coupling: x F^2 ; (ii) opaque composite: no F^2
mN_nat = lz.m_nucleus_gev(lz.A_XE_MEAN)
E_hi = np.linspace(0.5, 2000, 4000)
vmin_heavy = np.sqrt(E_hi * 1e-6 / (2 * mN_nat)) * lz.C_KMS          # km/s
eta_heavy = np.atleast_1d(lz.eta0(vmin_heavy))
spec_flat = eta_heavy.copy()
spec_helm = eta_heavy * F2_helm_nat(E_hi)
spec_shell = eta_heavy * F2_shell(E_hi)
Emax_700 = 2 * mN_nat * (700 / lz.C_KMS)**2 * 1e6
Emax_vmax = 2 * mN_nat * (lz.vmax_kms() / lz.C_KMS)**2 * 1e6
RES['heavy_kinematics'] = dict(Emax_700kms_keV=Emax_700, Emax_vmax_keV=Emax_vmax, vmax_kms=lz.vmax_kms(),
                               vmin_248_heavy_kms=float(np.sqrt(248e-6 / (2 * mN_nat)) * lz.C_KMS))

# HE sideband acceptance: 800 < S1c < 1700 phd -> NR energies via LZ-tuned NEST
def S1c_of_E(E):
    nph, ne = lz.nest_nr_yields(E)
    return lz.LZ['g1'] * nph
Es = np.array([100, 150, 200, 248, 300, 350, 400, 500, 600, 700, 800, 1000, 1200, 1500, 2000], float)
S1s = np.array([S1c_of_E(e) for e in Es])
E_S1_800 = float(np.interp(800, S1s, Es)); E_S1_1700 = float(np.interp(1700, S1s, Es)); E_S1_600 = float(np.interp(600, S1s, Es))
RES['HESB_energy_map'] = dict(E_at_S1c_600=E_S1_600, E_at_S1c_800=E_S1_800, E_at_S1c_1700=E_S1_1700,
                              S1c_at_248=float(np.interp(248, Es, S1s)))
def eff_hesb(E):
    return lz.LZ['eff_plateau'] * ((E >= E_S1_800) & (E <= E_S1_1700))

frows = []
for name, y in (('flat (opaque composite)', spec_flat), ('contact x Helm', spec_helm), ('contact x shell', spec_shell)):
    R_lo = win_int(E_hi, y, *LO); R_win = win_int(E_hi, y, *WIN)
    R_mid = win_int(E_hi, y, 55.0, 200.0)                             # NR band between the two windows (empty in LZ; P016 b=0.02)
    R_hesb = np.trapezoid(y * eff_hesb(E_hi), E_hi)
    R_tot_vis = np.trapezoid(y[E_hi >= 1.0], E_hi[E_hi >= 1.0])       # all scatters above 1 keV (visible S2)
    R_tot_all = np.trapezoid(y, E_hi)
    frows.append(dict(spectrum=name, f_win=R_win / R_tot_vis, f_lo=R_lo / R_tot_vis, f_mid=R_mid / R_tot_vis, f_hesb=R_hesb / R_tot_vis,
                      N_lo=R_lo / R_win, N_mid_per_win=R_mid / R_win, N_hesb_per_win=R_hesb / R_win,
                      N_above270_per_win=np.trapezoid(y[E_hi > 270], E_hi[E_hi > 270]) / R_win,
                      P0_mid_and_hesb=math.exp(-(R_mid + R_hesb) / R_win),
                      frac_below_1keV=1 - R_tot_vis / R_tot_all))
frac = pd.DataFrame(frows); frac.to_csv(f'{OUT}/P080_simp_spectral_fractions.csv', index=False)
p0_hesb = {r['spectrum']: math.exp(-r['N_hesb_per_win']) for _, r in frac.iterrows()}
RES['HESB_p0_given_one_window_event'] = p0_hesb
RES['P0_mid_and_hesb_given_one_window_event'] = {r['spectrum']: r['P0_mid_and_hesb'] for _, r in frac.iterrows()}
# multi-scatter traversals per single-scatter window event: Pmulti/(P1 f_win); for mu << 1 this -> mu/(2 f_win)
ms_rows = []
for sig in [1e-26, 3e-26, 1e-25, 3e-25, sig_lam_TPC, 3e-24, 1e-23]:
    p1 = float(np.interp(sig, sig_grid, P1)); pm = float(np.interp(sig, sig_grid, Pmulti))
    ms_rows.append(dict(sigma_N_cm2=sig, mu_mean=N_XE * sig * LBAR_TPC, P1=p1, Pmulti=pm,
                        MS_per_window_event_flat=pm / (p1 * float(frac.f_win.iloc[0])),
                        MS_per_window_event_contact_helm=pm / (p1 * float(frac.f_win.iloc[1]))))
ms = pd.DataFrame(ms_rows); ms.to_csv(f'{OUT}/P080_simp_multiscatter.csv', index=False)
# sigma below which fewer than 1 MS traversal accompanies the window event (flat spectrum)
RES['simp_sigma_MS_lt_1_flat_cm2'] = float(np.interp(1.0, ms.MS_per_window_event_flat.values, ms.sigma_N_cm2.values))

# --- flux, mass and overburden for one window event -------------------------------------------
# one event: N = Phi_4pi (S_TPC/4) T * P1(sigma) * f_win   (scatter anywhere in the TPC; FV acceptance ~ V_FV/V_TPC)
V_ACC = V_FV / V_TPC
def N_events(sig, m_gev, f_win, vbar_kms=lz.v_earth_kms() * 1.15):
    # <v> of the SHM in the lab ~ 1.15 v_E (Baxter SHM; computed below more precisely)
    n_chi = lz.RHO0_GEV_CM3 / m_gev
    Phi = n_chi * vbar_kms * 1e5                     # cm^-2 s^-1 (4pi-integrated)
    P1s = np.interp(sig, sig_grid, P1)
    return Phi * (S_TPC / 4) * LIVE_S * P1s * f_win * V_ACC
# mean lab speed of the SHM (Baxter, v_E = 250.6, v0 = 238, vesc = 544): numerical
def mean_speed_shm():
    v = np.linspace(0.1, lz.vmax_kms(), 4000)
    vE, v0, vesc = lz.v_earth_kms(), lz.V0_KMS, lz.VESC_KMS
    # lab-frame speed distribution of a truncated Maxwellian (angular integral analytic)
    f = np.zeros_like(v)
    for i, vi in enumerate(v):
        # integrate over cos(theta) between v_lab and v_E: v_gal^2 = v^2 + vE^2 + 2 v vE c
        c = np.linspace(-1, 1, 801)
        vg2 = vi**2 + vE**2 + 2 * vi * vE * c
        g = np.exp(-vg2 / v0**2) * (vg2 < vesc**2)
        f[i] = vi**2 * np.trapezoid(g, c)
    f /= np.trapezoid(f, v)
    return float(np.trapezoid(v * f, v)), float(np.trapezoid(f / v, v))
VBAR, INVV = mean_speed_shm()
RES['shm_mean_speed_kms'] = VBAR
# mass giving one event for a given sigma and spectrum (N ∝ 1/m)
mrows = []
for _, r in frac.iterrows():
    for sig in [1e-26, sig_lam_TPC, 1e-24, 1e-23]:
        N1 = N_events(sig, 1.0, r['f_win'], VBAR)          # events for m = 1 GeV
        m_one = N1                                          # N = N1/m -> m = N1 for N = 1
        # overburden: pointlike coupling scales sigma_A = sigma_N (A_rock/A_Xe)^4 (heavy DM, coherent); opaque: geometric, same sigma
        mrows.append(dict(spectrum=r['spectrum'], sigma_N_cm2=sig, lambda_cm=lam(sig), P1=float(np.interp(sig, sig_grid, P1)),
                          m_one_event_GeV=m_one, N_traversals_TPC=(lz.RHO0_GEV_CM3 / m_one) * VBAR * 1e5 * (S_TPC / 4) * LIVE_S,
                          N_lo_events=r['N_lo'], N_hesb_events=r['N_hesb_per_win']))
mass = pd.DataFrame(mrows); mass.to_csv(f'{OUT}/P080_simp_mass_for_one_event.csv', index=False)

# overburden transparency: mean number of rock scatters and fractional speed loss
def rock_scatters(sig_xe, m_gev, scaling):
    n_sc = 0.0; dE_frac = 0.0
    for Z, A, w in ROCK.values():
        mA = A * lz.AMU_GEV
        if scaling == 'coherent':      # sigma_A = sigma_n A^2 (mu_A/mu_n)^2 ; for m >> m_A: sigma_A/sigma_Xe = (A/A_Xe)^4
            sigA = sig_xe * (A / lz.A_XE_MEAN)**2 * (lz.mu_red(m_gev, mA) / lz.mu_red(m_gev, mN_nat))**2
        else:                          # geometric (opaque sphere): same cross-section on every nucleus
            sigA = sig_xe
        n_per_g = w / (A * AMU_G)
        nsc = n_per_g * X_ROCK * sigA
        n_sc += nsc
        dE_frac += nsc * 2 * lz.mu_red(m_gev, mA)**2 / (m_gev * mA)   # <E_R>/E_kin = 2 mu^2/(m mA) per isotropic scatter
    return n_sc, dE_frac
orows = []
for m in [1e3, 1e4, 1e5, 1e6, 1e8, 1e10, 1e12, 1e14, 1e16]:
    for scaling in ('coherent', 'geometric'):
        nsc, dEf = rock_scatters(sig_lam_TPC, m, scaling)
        orows.append(dict(m_GeV=m, scaling=scaling, sigma_Xe_cm2=sig_lam_TPC, N_rock_scatters=nsc, frac_energy_loss=dEf,
                          transparent_30pct=dEf < 0.3))
ob = pd.DataFrame(orows); ob.to_csv(f'{OUT}/P080_simp_overburden.csv', index=False)
# minimum mass for 30 % loss at sigma_lambda for both scalings (dE_frac ∝ 1/m for m >> m_A)
def m_transparent(scaling, frac=0.3):
    lo, hi = 1e2, 1e20
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if rock_scatters(sig_lam_TPC, mid, scaling)[1] > frac: lo = mid
        else: hi = mid
    return hi
RES['simp_overburden_min_mass_GeV'] = dict(coherent=m_transparent('coherent'), geometric=m_transparent('geometric'))

# --- continuous-track objects ---------------------------------------------------------------------
v700 = 700e5  # cm/s
track = {}
# nuclearite / strange-quark-matter nugget (De Rujula-Glashow 1984): sigma = pi (1e-8 cm)^2 for M < 1.5 ng (atomic radius; recalled likely)
for name, sig_geo in (('nuclearite_M<1.5ng (sigma=pi(1e-8cm)^2)', math.pi * 1e-16), ('nuclear-size composite R=4 fm', math.pi * (4e-13)**2),
                      ('R = 1e-10 cm composite', math.pi * 1e-20)):
    dEdx = sig_geo * RHO_LXE * v700**2        # erg/cm
    dEdx_keV_cm = dEdx / 1.602e-9
    track[name] = dict(sigma_cm2=sig_geo, dEdx_keV_per_cm=dEdx_keV_cm, E_over_chord_keV=dEdx_keV_cm * LBAR_TPC,
                       lambda_cm=lam(sig_geo))
# unit-charge CHAMP and Dirac monopole electronic stopping at 700 km/s (Lindhard-Scharff; monopole via Z_eff = g beta / e = (1/(2 alpha)) beta)
XE = dict(Xe=(54, 131.29, 1.0))
S_champ = lindhard_electronic_gev_per_gcm2(1.0, 700, XE, 'LS')          # GeV per g/cm^2
S_champ_rock = lindhard_electronic_gev_per_gcm2(1.0, 700, ROCK, 'LS')
beta700 = 700 / lz.C_KMS; zeff_mono = beta700 / (2 * ALPHA)
S_mono = lindhard_electronic_gev_per_gcm2(1.0, 700, XE, 'LS') * zeff_mono**2
track['CHAMP (charge e)'] = dict(dEdx_keV_per_cm=S_champ * RHO_LXE * 1e6, E_over_chord_keV=S_champ * RHO_LXE * 1e6 * LBAR_TPC,
                                 overburden_loss_GeV=S_champ_rock * (X_ROCK + X_AIR),
                                 m_min_survive_30pct_GeV=S_champ_rock * (X_ROCK + X_AIR) / (0.3 * 0.5 * beta700**2))
# alternative (Ahlen-Kinoshita-type, slow monopole): dE/dx ~ (g/e)^2 K (Z/A) rho beta L with K = 0.307 MeV cm^2/g, L ~ 3 (recalled, uncertain)
S_mono_AK = (1 / (2 * ALPHA))**2 * 0.307e-3 * (Z_XE / lz.A_XE_MEAN) * beta700 * 3.0     # GeV per g/cm^2
track['Dirac monopole (Z_eff = beta/2alpha)'] = dict(Z_eff=zeff_mono, dEdx_keV_per_cm=S_mono * RHO_LXE * 1e6,
                                                     E_over_chord_keV=S_mono * RHO_LXE * 1e6 * LBAR_TPC,
                                                     dEdx_keV_per_cm_AK_estimate=S_mono_AK * RHO_LXE * 1e6)
RES['continuous_tracks'] = track

# ----------------------------------------------------------------------------------
# C. Flux ceiling from N = 1 and the rho/m bound
# ----------------------------------------------------------------------------------
F_iso = 1.0 / (math.pi * S_FV * LIVE_S)           # cm^-2 s^-1 sr^-1 for expectation 1 through the FV surface
F_iso_TPC = 1.0 / (math.pi * S_TPC * LIVE_S)
Phi4pi = 4 * math.pi * F_iso
FC90_n1 = 4.36                                     # Feldman-Cousins 90 % UL for n=1, b=0 (recalled, certain)
m_max_p1 = lz.RHO0_GEV_CM3 * VBAR * 1e5 / Phi4pi   # mass for which one traversal of the FV per run at rho = 0.3
RES['flux_ceiling'] = dict(F_one_event_FV_cm2_s_sr=F_iso, F_one_event_TPC_cm2_s_sr=F_iso_TPC, F_90UL_FV=F_iso * FC90_n1,
                           Phi_4pi_FV_cm2_s=Phi4pi, m_for_one_FV_traversal_GeV=m_max_p1,
                           m_Planck_GeV=1.22e19, traversals_Planck_mass=m_max_p1 / 1.22e19,
                           Parker_bound_recalled=1e-15, MACRO_recalled=1.4e-16,
                           ratio_to_Parker=F_iso / 1e-15, ratio_to_MACRO=F_iso / 1.4e-16)
# rho/m x p for one event: (rho/m) p = Phi4pi/vbar  ; table m vs required interaction probability p
prow = []
for m in [1e6, 1e8, 1e10, 1e12, 1e14, 1e16, 1e18, 1.22e19]:
    p_req = m / m_max_p1
    sig_req = -math.log(1 - min(p_req, 1 - 1e-12)) / (N_XE * LBAR_FV) if p_req < 1 else float('nan')
    prow.append(dict(m_GeV=m, N_traversals_FV=m_max_p1 / m, p_required=p_req, sigma_N_required_cm2=sig_req))
pflux = pd.DataFrame(prow); pflux.to_csv(f'{OUT}/P080_flux_mass.csv', index=False)

# ----------------------------------------------------------------------------------
# D. Inelastic composites (dark nuclei): delta_max vs mass and the mass ceiling
# ----------------------------------------------------------------------------------
drows = []
for m in [1e2, 1e3, 1e4, 1e6, 1e9, 1e12, 1e15, 1e18]:
    drows.append(dict(m_GeV=m, delta_max_248_keV=lz.delta_max_kev(248, m), delta_max_270_136Xe_keV=lz.delta_max_kev(270, m, A=136),
                      vmin_248_elastic_kms=lz.vmin_kms(248, m)))
dm = pd.DataFrame(drows); dm.to_csv(f'{OUT}/P080_inelastic_composites.csv', index=False)
q248 = math.sqrt(2 * mN_nat * 248e-6)  # GeV
RES['inelastic'] = dict(q_248_MeV=q248 * 1e3, R_for_qR_eq_1_fm=lz.HBARC_GEV_FM / q248,
                        delta_max_heavy_limit_keV=float(dm.delta_max_248_keV.iloc[-1]),
                        mass_ceiling_p1_GeV=m_max_p1)

# ----------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------
# Fig 1: recoil spectra shapes (MCP 1/E^2, contact, flat) normalised to the window
fig, ax = plt.subplots(figsize=(6.4, 4.2))
def norm_win(E, y):
    return y / win_int(E, y, *WIN)
y_mcp = dRdE_mcp(Egrid, 1000., 1.0)
ax.plot(Egrid, norm_win(Egrid, y_mcp), label='millicharged, 1 TeV (Rutherford $\\propto F^2\\eta/E_R^2$)')
ax.plot(Egrid, norm_win(Egrid, y_o1), label='contact SI, 1 TeV (Helm)')
ax.plot(E_hi, norm_win(E_hi, spec_flat), label='opaque heavy composite (flat to $2m_Nv^2$)')
ax.plot(E_hi, norm_win(E_hi, spec_helm), '--', label='heavy pointlike SIMP (flat$\\times F^2$)')
ax.axvspan(*LO, color='C3', alpha=0.15, label='5.4-55 keV ROI'); ax.axvspan(*WIN, color='C2', alpha=0.2, label='200-270 keV')
ax.axvspan(E_S1_800, E_S1_1700, color='0.5', alpha=0.2, label=f'HE SB ({E_S1_800:.0f}-{E_S1_1700:.0f} keV)')
ax.set_yscale('log'); ax.set_xscale('log'); ax.set_xlim(1, 2000); ax.set_ylim(1e-5, 1e5)
ax.set_xlabel('$E_R$ [keV]'); ax.set_ylabel('dR/dE$_R$ per 200-270 keV event [keV$^{-1}$]')
ax.set_title('P080: recoil spectra normalised to one 200-270 keV event'); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f'{FIG}/P080_fig1_spectra.png', dpi=150); plt.close(fig)

# Fig 2: SIMP plane (m, sigma_N): N=1 contours, lambda = L, overburden, recalled exclusions (schematic)
fig, ax = plt.subplots(figsize=(6.4, 4.6))
mgrid = np.logspace(3, 19, 200)
for name, ls in (('flat (opaque composite)', '-'), ('contact x Helm', '--')):
    fw = float(frac[frac.spectrum == name].f_win.iloc[0])
    # for each m find sigma with N = 1 (N is non-monotonic in sigma: rising then falling); take the lower branch
    sig_low = []; sig_high = []
    for m in mgrid:
        N = np.array([N_events(s, m, fw, VBAR) for s in sig_grid])
        above = N >= 1
        if above.any():
            sig_low.append(sig_grid[np.argmax(above)]); sig_high.append(sig_grid[len(above) - 1 - np.argmax(above[::-1])])
        else:
            sig_low.append(np.nan); sig_high.append(np.nan)
    ax.plot(mgrid, sig_low, ls=ls, color='C0', label=f'N=1 in 2.84 t yr, {name}')
    ax.plot(mgrid, sig_high, ls=ls, color='C0')
ax.axhline(sig_lam_TPC, color='k', lw=0.8, label=f'$\\lambda$ = mean TPC chord ({LBAR_TPC:.0f} cm)')
mmin_c, mmin_g = RES['simp_overburden_min_mass_GeV']['coherent'], RES['simp_overburden_min_mass_GeV']['geometric']
ax.axvline(mmin_c, color='C1', ls='--', lw=0.8); ax.axvline(mmin_g, color='C1', ls='-', lw=0.8)
ax.text(mmin_c * 1.3, 5e-20, 'rock opaque to the left\n(coherent $A^4$ scaling)', fontsize=6, color='C1', va='top')
ax.text(mmin_g * 1.3, 3e-21, 'rock opaque to the left\n(geometric scaling)', fontsize=6, color='C1', va='top')
ax.axvline(m_max_p1, color='C3', lw=1.0); ax.text(m_max_p1 / 40, 5e-20, '< 1 traversal\nper run', fontsize=6, color='C3', va='top')
ax.text(1e11, 1e-24, 'N = 1 possible only\ninside the loops', fontsize=6.5, color='C0', ha='center')
# recalled exclusions (uncertain): schematic boxes
ax.fill_between([1e3, 1e15], 1e-31, 3e-26, color='0.85', alpha=0.6, label='recalled exclusions (uncertain): XENON1T/LZ single-scatter,\nCDMS-I/XQC, CRESST/EDELWEISS surface, MJD, DEAP-3600 multiscatter')
ax.fill_between([1e15, 1e19], 1e-31, 1e-18, color='0.85', alpha=0.6)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(1e3, 1e19); ax.set_ylim(1e-27, 1e-19)
ax.set_xlabel('$m_\\chi$ [GeV]'); ax.set_ylabel('$\\sigma_{\\chi\\mathrm{Xe}}$ (total, per nucleus) [cm$^2$]')
ax.set_title('P080: heavy-relic plane, single 200-270 keV scatter per traversal')
fig.legend(fontsize=5.5, loc='lower center', ncol=2, frameon=False); fig.subplots_adjust(bottom=0.26, top=0.93, left=0.12, right=0.97)
fig.savefig(f'{FIG}/P080_fig2_simp_plane.png', dpi=150); plt.close(fig)

# Fig 3: P(exactly one scatter) vs sigma
fig, ax = plt.subplots(figsize=(6.0, 3.8))
ax.plot(sig_grid, P1, label='exactly one scatter (single S1/S2 pair)')
ax.plot(sig_grid, Pmulti, label='$\\geq$2 scatters (multiple S2s: rejected)')
ax.plot(sig_grid, P0, label='no scatter')
ax.axvline(sig_lam_TPC, color='k', lw=0.8); ax.set_xscale('log'); ax.set_xlabel('$\\sigma_{\\chi\\mathrm{Xe}}$ [cm$^2$]')
ax.set_ylabel('probability per TPC traversal'); ax.legend(fontsize=7); ax.set_title('P080: isotropic chords through the LZ TPC')
fig.tight_layout(); fig.savefig(f'{FIG}/P080_fig3_P1.png', dpi=150); plt.close(fig)

# ----------------------------------------------------------------------------------
# Summary table (candidate / required / recalled bound / verdict)
# ----------------------------------------------------------------------------------
Nlo_mcp_h = float(mcp[(mcp.m_GeV == 1000) & (mcp.response == 'Helm')].N_lo.iloc[0])
Nlo_mcp_s = float(mcp[(mcp.m_GeV == 1000) & (mcp.response == 'shell')].N_lo.iloc[0])
eps1TeV = float(mcp[(mcp.m_GeV == 1000) & (mcp.response == 'Helm')].eps_one_event.iloc[0])
summary = pd.DataFrame([
    dict(candidate='halo millicharged, 100 GeV-10 TeV', required=f'eps = {eps1TeV:.1e} (1 TeV)',
         existing_bound='none relevant at eps~1e-7 (recalled: LZ/XENON1T MCP-DM limits eps <~ 1e-9-1e-8 for elastic; uncertain)',
         verdict=f'N_lo = {Nlo_mcp_h:.1e} (Helm) / {Nlo_mcp_s:.1e} (shell): excluded by the empty low-energy ROI'),
    dict(candidate='relativistic millicharged (CR / SN)', required=f'Phi eps^2 = {phi_eps2:.1e} cm^-2 s^-1',
         existing_bound='recalled: Plestid et al. 2020 / ArgoNeuT / SENSEI (uncertain)',
         verdict=f"N_lo = {RES['N_lo_relativistic_MCP']['helm']:.0f}/{RES['N_lo_relativistic_MCP']['shell']:.0f}, {1/nr_over_er:.0f} ERs per NR: excluded"),
    dict(candidate='pointlike SIMP, one scatter per traversal', required=f'sigma_Xe ~ {sig_lam_TPC:.1e} cm^2, m ~ {float(mass[(mass.spectrum=="contact x Helm")&(mass.sigma_N_cm2==sig_lam_TPC)].m_one_event_GeV.iloc[0]):.1e} GeV',
         existing_bound='recalled: MJD/CRESST/EDELWEISS surface, XENON1T/DEAP multiscatter (uncertain)',
         verdict=f"N_lo = {float(frac[frac.spectrum=='contact x Helm'].N_lo.iloc[0]):.0f}: excluded; 26 % of traversals multi-scatter"),
    dict(candidate='opaque heavy composite (flat spectrum)', required=f'sigma_Xe ~ {sig_lam_TPC:.1e} cm^2 (R~4 fm), m ~ {float(mass[(mass.spectrum=="flat (opaque composite)")&(mass.sigma_N_cm2==sig_lam_TPC)].m_one_event_GeV.iloc[0]):.1e} GeV',
         existing_bound='recalled: DEAP-3600 / MJD multiscatter, MICA, rocky planets (uncertain)',
         verdict=f"N_lo = {float(frac[frac.spectrum=='flat (opaque composite)'].N_lo.iloc[0]):.2f} but {float(frac[frac.spectrum=='flat (opaque composite)'].N_hesb_per_win.iloc[0]):.1f} HE-SB events per window event (0 observed, p = {p0_hesb['flat (opaque composite)']:.3f})"),
    dict(candidate='nuclearites / Q-balls (sigma >= 1e-16 cm^2)', required='none: dE/dx = sigma rho v^2',
         existing_bound='recalled: MACRO, SLIM, MICA, Ohya (uncertain)',
         verdict=f"{track['nuclearite_M<1.5ng (sigma=pi(1e-8cm)^2)']['dEdx_keV_per_cm']:.1e} keV/cm continuous track: not a point-like 248 keV NR"),
    dict(candidate='CHAMP (charge e) / monopole at 700 km/s', required='none',
         existing_bound='recalled: MACRO/Parker (monopoles); CHAMP stopping (uncertain)',
         verdict=f"{track['CHAMP (charge e)']['dEdx_keV_per_cm']:.1e} / {track['Dirac monopole (Z_eff = beta/2alpha)']['dEdx_keV_per_cm']:.1e} keV/cm electronic track; CHAMP needs m > {track['CHAMP (charge e)']['m_min_survive_30pct_GeV']:.0e} GeV to reach 1478 m"),
    dict(candidate='superheavy / Planck-scale DM (p ~ 1)', required=f'F = {F_iso:.1e} cm^-2 s^-1 sr^-1, m <= {m_max_p1:.1e} GeV',
         existing_bound='Parker 1e-15, MACRO 1.4e-16 cm^-2 s^-1 sr^-1 (recalled, monopoles)',
         verdict=f'flux ceiling: m <= {m_max_p1:.1e} GeV x p; Planck mass gives {m_max_p1/1.22e19:.2f} traversals per run'),
    dict(candidate='inelastic composite (dark nucleus)', required=f"delta <= {RES['inelastic']['delta_max_heavy_limit_keV']:.0f} keV (m -> inf), R_D < {RES['inelastic']['R_for_qR_eq_1_fm']:.2f} fm, m <= {m_max_p1:.0e} GeV",
         existing_bound='P042: tau(chi2->chi1 gamma) > 0.43 us; P030/P007 window',
         verdict='viable only as the IDM window of P007/P021 with a composite UV completion (P023)'),
])
summary.to_csv(f'{OUT}/P080_summary_table.csv', index=False)

with open(f'{OUT}/P080_results.json', 'w') as f:
    json.dump(RES, f, indent=1, default=float)

# console summary
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)
print(json.dumps(RES, indent=1, default=float))
print('\n--- MCP halo ---\n', mcp.to_string())
print('\n--- MCP stopping ---\n', stop.to_string())
print('\n--- SIMP spectral fractions ---\n', frac.to_string())
print('\n--- SIMP mass for one event ---\n', mass.to_string())
print('\n--- overburden ---\n', ob.to_string())
print('\n--- flux/mass ---\n', pflux.to_string())
print('\n--- inelastic ---\n', dm.to_string())
print('\n--- summary ---\n', summary.to_string())
