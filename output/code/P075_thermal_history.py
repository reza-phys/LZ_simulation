#!/usr/bin/env python
"""P075: Thermal history of ~1 TeV pseudo-Dirac dark matter with the LZ-fitted coupling.

Run from the simulation root:  .venv/bin/python output/code/P075_thermal_history.py

Parts
  A  unit-coupling LZ counts N_iso(m,delta), N_p(m,delta) (P011 grid for 0.3/1/3 TeV + cached live WimPyDD for 0.5/2/5/10 TeV
     from the helper P075_wimpydd_grid.py) and the Higgsino grid (P007 kernels 0.3-4 TeV + live 5/10 TeV)
  B  relic-density machinery (Boltzmann ODE with g_*(T), semi-analytic check), validation (2.2e-26 WIMP; 1.1 TeV Higgsino)
  C  coannihilation with the delta-split partner: Griest-Seckel sigma_eff(delta)/sigma_eff(0)
  D  Higgsino: Omega_th(m), thermal fraction, dilution, non-thermal T_RH, LZ counts at thermal density, delta windows
  E  heavy Z' (contact): coupling-independent N_th(m,delta) for B-L and U(1)_B, Omega at LZ coupling, dilution / NT fraction
  F  dark photon: thermal alpha_D(m), LZ epsilon, Planck floor vs epsilon ceiling window in (m,delta), dilution to reopen it,
     contact-limit fixed point (proton-only, S=8) and the s-channel fraction at realistic m_A'
  G  freeze-in / low-reheating production with the LZ coupling (UV contact operator), thermalisation test
  H  asymmetric DM: chi <-> chibar oscillation time 1/delta
  I  map figure + tables
All formulae are stated in work/P075/details.md; recalled inputs are flagged there and in provenance/P075.json.
"""
import os, sys, math, json, time
import numpy as np
import pandas as pd
from scipy import integrate, optimize
from scipy.special import kve, erf, erfc, kv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, 'output/code')
from common import lzcommon as lz

OUT = 'output/work/P075'; FIG = f'{OUT}/figures'
os.makedirs(FIG, exist_ok=True)
LOG = open(f'{OUT}/run_log.txt', 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.write(s + '\n'); LOG.flush()
T_START = time.time()

# ------------------------------------------------------------------------------------------------
# constants (recalled, certain unless flagged)
# ------------------------------------------------------------------------------------------------
MV = lz.M_V_GEV                      # 246.2 GeV, LZ/Anand unit coupling c = 1/m_v^2
M_NUC = lz.M_NUCLEON_GEV
GEV2_CM2 = lz.GEV_TO_CM2             # 3.894e-28 cm^2 per GeV^-2
C_CMS = 2.99792458e10
GEV2_TO_CM3S = GEV2_CM2 * C_CMS      # 1 GeV^-2 (x c) = 1.1674e-17 cm^3/s
HBAR_GEV_S = 6.582119569e-25
M_PL = 1.220890e19                   # GeV (non-reduced)
OMEGA_H2 = 0.120                     # Planck 2018
S0_OVER_RHOC_H2 = 2.755e8            # GeV^-1: Omega h^2 = 2.755e8 Y_inf m/GeV (s0 = 2891 cm^-3, rho_c/h^2 = 1.054e-5 GeV cm^-3) [certain]
ALPHA_EM = 1 / 137.036
SW2 = 0.2312; TW2 = SW2 / (1 - SW2)
G_WEAK = 0.6517                      # SU(2) coupling at m_Z (likely)
ALPHA2 = G_WEAK**2 / (4 * math.pi)
EPS_MAX = 1e-3                       # recalled dark-photon kinetic-mixing ceiling (BaBar/LHCb, 1-10 GeV) used by P011 [likely]
SIGV_CANON = 2.2e-26                 # cm^3/s, Steigman-Dasgupta-Beacom 2012 for m >> 10 GeV [certain]
LZ_BAND = (0.105, 3.65)              # P007/P021: LZ 90% two-sided interval in expected events for one observed event
KAPPA_HIGGSINO = 0.0777              # P007: Higgsino Xe amplitude equals an isoscalar coupling with (c_eq m_v^2)^2 = 0.0777 (used only for the freeze-in G estimate)

# g_*(T) (energy) table, standard SM values (Kolb-Turner / Husdal) [recalled, likely; +-5%]
GSTAR_T = np.array([1e-4, 5e-4, 1e-3, 1e-2, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 100.0, 150.0, 200.0, 300.0, 1e3, 1e5])
GSTAR_V = np.array([3.36, 5.0, 10.75, 10.75, 11.5, 17.25, 25.0, 60.0, 66.0, 72.0, 75.75, 79.0, 86.0, 86.25, 88.0, 91.0, 96.0, 103.0, 106.75, 106.75, 106.75, 106.75])
def gstar(T):
    return float(np.interp(math.log(T), np.log(GSTAR_T), GSTAR_V))

# ------------------------------------------------------------------------------------------------
# Part A: unit-coupling LZ counts N_iso (c_p = c_n = 1/m_v^2) and N_p (c_p = 1/m_v^2, c_n = 0)
#         P011 grid (annual-mean Baxter halo, eff sigma 11.5 keV) for 300/1000/3000 GeV; live WimPyDD for 500/2000/5000/10000 GeV
# ------------------------------------------------------------------------------------------------
say('Part A: unit-coupling LZ count grid')
p11 = pd.read_csv('output/work/P011/sigma_p_required.csv')
p11 = p11[(p11.halo == 'annual') & (p11.eff_sigma_keV == 11.5)]
rows = [dict(m_GeV=r.m_GeV, delta_keV=r.delta_keV, N_iso_unit=r.N_iso_unit, N_p_unit=r.N_p_unit, source='P011')
        for r in p11.itertuples() if 240.0 <= r.delta_keV <= 400.0]
CACHE = f'{OUT}/N_unit_grid_live.csv'; CACHE_H = f'{OUT}/higgsino_grid_live.csv'
if not (os.path.exists(CACHE) and os.path.exists(CACHE_H)):
    sys.exit('run output/code/P075_wimpydd_grid.py first (builds the cached live WimPyDD grids, ~5 min)')
live = pd.read_csv(CACHE); liveH = pd.read_csv(CACHE_H)
for r in live[live.m_GeV == 1000.0].itertuples():
    ref = p11[(p11.m_GeV == 1000.0) & (p11.delta_keV == r.delta_keV)].iloc[0]
    say('  live/P011 check 1 TeV delta=%.0f: iso %.4f, p %.4f' % (r.delta_keV, r.N_iso_unit / ref.N_iso_unit, r.N_p_unit / ref.N_p_unit))
grid = pd.concat([pd.DataFrame(rows), live[live.m_GeV != 1000.0]], ignore_index=True).sort_values(['m_GeV', 'delta_keV'])
grid.to_csv(f'{OUT}/N_unit_grid.csv', index=False)
# Higgsino grid: P007's own Z-exchange kernels (annual halo, eff 11.5) for 300-4000 GeV + live 5000/10000 GeV
p7 = pd.read_csv('output/work/P007/N_events_grid.csv'); p7 = p7[(p7.halo == 'annual') & (p7.eff_sigma_keV == 11.5) & (p7.delta_keV >= 240.0)]
hrows = [dict(m_GeV=r.m_GeV, delta_keV=r.delta_keV, N_events=r.N_events, source='P007') for r in p7.itertuples()]
gridH = pd.concat([pd.DataFrame(hrows), liveH[liveH.m_GeV != 1000.0]], ignore_index=True).sort_values(['m_GeV', 'delta_keV'])
gridH.to_csv(f'{OUT}/higgsino_grid.csv', index=False)
hchk = []
for r in liveH[liveH.m_GeV == 1000.0].itertuples():
    ref = p7[(p7.m_GeV == 1000.0) & (p7.delta_keV == r.delta_keV)].N_events.iloc[0]
    hchk.append(dict(delta_keV=r.delta_keV, N_P007=ref, N_live=r.N_events, ratio=r.N_events / ref))
hchk = pd.DataFrame(hchk); hchk.to_csv(f'{OUT}/higgsino_grid_check.csv', index=False)
say('  live Higgsino vs P007 grid at 1 TeV: ' + ', '.join('%.0f keV: %.3f' % (a, b) for a, b in zip(hchk.delta_keV, hchk.ratio)))

# kinematic ceiling used to scale the mass interpolation: delta_c(m) = mu(m, m_Xe) v_max^2/2, v_max = v_esc + v_sun + v_orb
V_MAX_KMS = lz.VESC_KMS + 250.6 + lz.V_EARTH_ORBIT
M_XE = lz.A_XE_MEAN * lz.AMU_GEV
def delta_c(m):
    mu = m * M_XE / (m + M_XE)
    return 0.5 * mu * (V_MAX_KMS / lz.C_KMS)**2 * 1e6   # keV
def make_interp(df, col):
    masses = sorted(df.m_GeV.unique())
    tabs = {}
    for mm in masses:
        g = df[df.m_GeV == mm].sort_values('delta_keV')
        tabs[mm] = (g.delta_keV.values / delta_c(mm), g[col].values)
    def at_mass(mm, xi):
        x, n = tabs[mm]
        ok = n > 0
        if ok.sum() < 2 or xi < x[ok][0] or xi > x[ok][-1]:
            # beyond the last positive tabulated point -> zero (kinematically closed / below grid resolution)
            return 0.0 if xi > x[ok][-1] else float(np.exp(np.interp(xi, x[ok], np.log(n[ok]))))
        return float(np.exp(np.interp(xi, x[ok], np.log(n[ok]))))
    lm = np.log(masses)
    def f(m, d):
        """log-linear interpolation in ln m at fixed xi = delta/delta_c(m) between the two neighbouring tabulated masses."""
        xi = d / delta_c(m)
        if m <= masses[0]: return at_mass(masses[0], xi)
        if m >= masses[-1]: return at_mass(masses[-1], xi)
        i = int(np.searchsorted(lm, math.log(m), side='right')) - 1
        i = min(max(i, 0), len(masses) - 2)
        a, b = at_mass(masses[i], xi), at_mass(masses[i + 1], xi)
        t = (math.log(m) - lm[i]) / (lm[i + 1] - lm[i])
        if a > 0 and b > 0:
            return math.exp((1 - t) * math.log(a) + t * math.log(b))
        if a > 0: return a * (1 - t) if t > 0.5 else a          # edge: one neighbour closed
        if b > 0: return b * t if t < 0.5 else b
        return 0.0
    return f, masses
N_iso, MASSES = make_interp(grid, 'N_iso_unit')
N_p, _ = make_interp(grid, 'N_p_unit')
N_higgsino, MASSES_H = make_interp(gridH, 'N_events')
say('  unit-coupling masses:', MASSES, ' Higgsino masses:', MASSES_H, ' delta_c(1 TeV) = %.1f keV' % delta_c(1000.0))
# interpolation test: predict a tabulated mass from its neighbours (leave-one-out)
itest = []
for target, (lo_m, hi_m) in ((1000.0, (500.0, 2000.0)), (3000.0, (2000.0, 5000.0))):
    sub = grid[grid.m_GeV.isin([lo_m, hi_m])]
    fpred, _ = make_interp(sub, 'N_iso_unit')
    for d in (300.0, 350.0, 366.0, 380.0):
        truth = N_iso(target, d)
        if truth > 0:
            itest.append(dict(target_m=target, delta_keV=d, N_true=truth, N_interp=fpred(target, d), ratio=fpred(target, d) / truth))
itest = pd.DataFrame(itest); itest.to_csv(f'{OUT}/mass_interpolation_test.csv', index=False)
say('  leave-one-out mass interpolation (N_iso): ratios ' + ', '.join('%.0f/%.0f: %.2f' % (r.target_m, r.delta_keV, r.ratio) for r in itest.itertuples()))

# ------------------------------------------------------------------------------------------------
# Part B: relic-density machinery
# ------------------------------------------------------------------------------------------------
say('Part B: relic machinery')
G_DOF = 4.0   # chi1 (2) + chi2 (2); delta << T_fo so both are fully populated (Part C)
def Yeq(x, m, g=G_DOF):
    T = m / x
    return 45 * g / (4 * math.pi**4 * gstar(T)) * x**2 * kve(2, x) * math.exp(-x)
def omega_ode(m, sigv_of_x, x0=3.0, x1=2000.0):
    """dY/dx = -sqrt(pi/45) M_Pl m sqrt(g*(T)) <sigma v>_eff(x) (Y^2 - Yeq^2)/x^2 ; <sigma v> in GeV^-2; g*s = g* assumed."""
    def rhs(x, y):
        T = m / x
        lam = math.sqrt(math.pi / 45) * M_PL * m * math.sqrt(gstar(T))
        return [-lam / x**2 * sigv_of_x(x) * (y[0]**2 - Yeq(x, m)**2)]
    sol = integrate.solve_ivp(rhs, [x0, x1], [Yeq(x0, m)], method='Radau', rtol=1e-8, atol=1e-32)
    return S0_OVER_RHOC_H2 * sol.y[0, -1] * m, sol
def xf_of(sol, m):
    """freeze-out point: Y = 2.5 Yeq (conventional)."""
    x = sol.t; y = sol.y[0]
    ye = np.array([Yeq(xx, m) for xx in x])
    i = np.argmax(y > 2.5 * ye)
    return float(x[i]) if i > 0 else float('nan')
def const(v): return lambda x: v

# validation 1: canonical WIMP
val = {}
for m in (100.0, 1000.0, 10000.0):
    om, sol = omega_ode(m, const(SIGV_CANON / GEV2_TO_CM3S))
    val[f'const_2.2e-26_m{int(m)}'] = dict(omega_h2=om, x_f=xf_of(sol, m))
    say('  <sigma v> = 2.2e-26 cm^3/s, m = %5.0f GeV: Omega h^2 = %.4f, x_f = %.1f' % (m, om, xf_of(sol, m)))
# required constant <sigma v>_eff(m) for Omega h^2 = 0.12 (all s-wave, velocity-independent cases below reuse this)
MGRID = np.unique(np.round(np.concatenate([np.geomspace(300.0, 10000.0, 25), [500.0, 1000.0, 1100.0, 1150.0, 2000.0, 3000.0, 5000.0]]), 1))
SV_REQ = {}
for m in MGRID:
    f = lambda lg: math.log(omega_ode(m, const(10**lg))[0] / OMEGA_H2)
    SV_REQ[float(m)] = 10**optimize.brentq(f, -11.5, -7.5, xtol=1e-5)
sv_req_of_m = lambda m: float(np.exp(np.interp(math.log(m), np.log(MGRID), np.log([SV_REQ[float(x)] for x in MGRID]))))
say('  required <sigma v>_eff [cm^3/s]: ' + ', '.join('%.0f GeV: %.3e' % (m, SV_REQ[float(m)] * GEV2_TO_CM3S) for m in (300.0, 1000.0, 3000.0, 10000.0)))

# validation 2: Higgsino (ADG 2006 effective coannihilation cross-section, P025) -> Omega h^2 = 0.12 near 1.1 TeV
def sigv_higgsino_eff(mu):
    """<sigma v>_eff = g^4 (21 + 3 tan^2 + 11 tan^4 theta_W)/(512 pi mu^2) [GeV^-2]  (Arkani-Hamed, Delgado, Giudice 2006; recalled/likely)."""
    return G_WEAK**4 * (21 + 3 * TW2 + 11 * TW2**2) / (512 * math.pi * mu**2)
def omega_higgsino(mu):
    return omega_ode(mu, const(sigv_higgsino_eff(mu)))[0]
mu_th = optimize.brentq(lambda mu: omega_higgsino(mu) - OMEGA_H2, 600.0, 2000.0, xtol=0.5)
val['higgsino_thermal_mass_GeV'] = mu_th
val['higgsino_omega_1000'] = omega_higgsino(1000.0)
say('  Higgsino: Omega h^2 = 0.12 at mu = %.0f GeV (literature 1.1 TeV; P025 semi-analytic 1167 GeV); Omega h^2(1 TeV) = %.4f' % (mu_th, val['higgsino_omega_1000']))
# constant-g* variant (P054 used g* = 86.25) for the systematic
def omega_ode_gconst(m, sv, gs=86.25):
    lam = math.sqrt(math.pi / 45) * M_PL * m * math.sqrt(gs)
    Ye = lambda x: 45 * G_DOF / (4 * math.pi**4 * gs) * x**2 * kve(2, x) * math.exp(-x)
    sol = integrate.solve_ivp(lambda x, y: [-lam / x**2 * sv * (y[0]**2 - Ye(x)**2)], [3.0, 2000.0], [Ye(3.0)], method='Radau', rtol=1e-8, atol=1e-32)
    return S0_OVER_RHOC_H2 * sol.y[0, -1] * m
val['gstar_systematic_1TeV'] = omega_ode_gconst(1000.0, SIGV_CANON / GEV2_TO_CM3S) / val['const_2.2e-26_m1000']['omega_h2']
say('  g*(T) vs constant 86.25 at 1 TeV: Omega ratio (const/varying) = %.3f' % val['gstar_systematic_1TeV'])
json.dump(val, open(f'{OUT}/relic_validation.json', 'w'), indent=1)

# ------------------------------------------------------------------------------------------------
# Part C: coannihilation with the delta-split partner (Griest-Seckel 1991)
# ------------------------------------------------------------------------------------------------
say('Part C: effect of delta on the coannihilating pair')
def sigma_eff_ratio(delta_keV, m, x, channel):
    """sigma_eff(delta)/sigma_eff(0) for two Majorana states g1 = g2 = 2, Delta_2 = delta/m, at fixed x = m/T.
    channel 'offdiag': only sigma_12 != 0 (vector-current s-channel, Z' or A'* -> ff);
            'diag'   : sigma_11 = sigma_22 != 0, sigma_12 = 0 (chi chi -> A'A' / Z'Z' via t/u exchange; P025)."""
    D = delta_keV * 1e-6 / m
    w1, w2 = 1.0, (1 + D)**1.5 * math.exp(-x * D)
    geff = 2 * w1 + 2 * w2
    if channel == 'offdiag':
        num = 2 * (2 * w1) * (2 * w2)       # sigma_12 + sigma_21
        den0 = 2 * 2 * 2
    else:
        num = (2 * w1)**2 + (2 * w2)**2
        den0 = 2 * 2**2
    return (num / geff**2) / (den0 / 16.0)
crows = []
for m in (300.0, 1000.0, 3000.0, 10000.0):
    x_f = 25.0 - math.log(m / 1000.0) * 0.0  # placeholder replaced below by the solved x_f
    om, sol = omega_ode(m, const(sv_req_of_m(m)))
    x_f = xf_of(sol, m)
    for d in (250.0, 300.0, 350.0, 380.0):
        for ch in ('offdiag', 'diag'):
            r = sigma_eff_ratio(d, m, x_f, ch)
            crows.append(dict(m_GeV=m, delta_keV=d, x_f=x_f, T_f_GeV=m / x_f, delta_over_T_f=d * 1e-6 * x_f / m, channel=ch, sigma_eff_ratio=r, one_minus_ratio=1 - r))
# chargino coannihilation for comparison (Dirac, g = 4, Delta_pm = 342 MeV/m from P014)
for m in (300.0, 1000.0, 3000.0):
    D = 0.342 / m; x_f = 25.0
    w = (1 + D)**1.5 * math.exp(-x_f * D)
    crows.append(dict(m_GeV=m, delta_keV=342e3, x_f=x_f, T_f_GeV=m / x_f, delta_over_T_f=D * x_f, channel='chargino_weight', sigma_eff_ratio=w, one_minus_ratio=1 - w))
C = pd.DataFrame(crows); C.to_csv(f'{OUT}/coannihilation_delta_effect.csv', index=False)
say(C[(C.channel != 'chargino_weight') & (C.delta_keV == 300.0)].to_string(index=False))
say('  chargino Boltzmann weight at x_f = 25: ' + ', '.join('%.0f GeV: %.4f' % (r.m_GeV, r.sigma_eff_ratio) for r in C[C.channel == 'chargino_weight'].itertuples()))

# ------------------------------------------------------------------------------------------------
# non-thermal helpers
# ------------------------------------------------------------------------------------------------
def Y_required(m): return OMEGA_H2 / (S0_OVER_RHOC_H2 * m)
def H_of_T(T): return 1.66 * math.sqrt(gstar(T)) * T**2 / M_PL
def s_of_T(T): return 2 * math.pi**2 / 45 * gstar(T) * T**3
def T_RH_nonthermal(m, sigv):
    """Reheating temperature at which the annihilation-limited non-thermal yield Y = H/(s <sigma v>) equals Y_req
    (moduli/gravitino/inflaton decay with T_RH < T_fo; Moroi-Randall 2000 / Giudice-Kolb-Riotto 2001, recalled/likely)."""
    f = lambda lT: math.log(H_of_T(10**lT) / (s_of_T(10**lT) * sigv) / Y_required(m))
    try:
        return 10**optimize.brentq(f, -3.0, math.log10(m), xtol=1e-4)
    except ValueError:
        return float('nan')

# ------------------------------------------------------------------------------------------------
# Part D: Higgsino
# ------------------------------------------------------------------------------------------------
say('Part D: Higgsino')
drows = []
for m in MGRID:
    om = omega_higgsino(m); f = om / OMEGA_H2
    sv = sigv_higgsino_eff(m)
    om_sol = omega_ode(m, const(sv))[1]; xf = xf_of(om_sol, m)
    drows.append(dict(m_GeV=m, omega_h2_thermal=om, f_thermal=f, x_f=xf, T_fo_GeV=m / xf,
                      nonthermal_fraction_needed=max(0.0, 1 - f), dilution_D_needed=max(1.0, f),
                      T_RH_nonthermal_GeV=T_RH_nonthermal(m, sv) if f < 1 else float('nan'),
                      sigv_eff_cm3s=sv * GEV2_TO_CM3S))
D = pd.DataFrame(drows); D.to_csv(f'{OUT}/higgsino_thermal_history.csv', index=False)
say(D[D.m_GeV.isin([300.0, 500.0, 1000.0, 1100.0, 1150.0, 2000.0, 3000.0, 5000.0, 10000.0])][['m_GeV', 'omega_h2_thermal', 'f_thermal', 'x_f', 'T_fo_GeV', 'nonthermal_fraction_needed', 'dilution_D_needed', 'T_RH_nonthermal_GeV']].to_string(index=False))
f_th_higgsino = lambda m: math.exp(np.interp(math.log(m), np.log(MGRID), np.log(D.f_thermal.values)))
# mass range in which the Higgsino is 'thermal' within a +-20% theory band (Sommerfeld, g*, NLO)
m_lo_th = optimize.brentq(lambda m: f_th_higgsino(m) - 0.8, 300.0, 10000.0); m_hi_th = optimize.brentq(lambda m: f_th_higgsino(m) - 1.25, 300.0, 10000.0)
say('  Higgsino thermal within +-20-25%%: m = %.0f-%.0f GeV' % (m_lo_th, m_hi_th))

def delta_for_N(fun, m, N, lo=245.0, hi=400.0):
    """largest delta at which fun(m, delta) = N (the falling branch)."""
    ds = np.arange(lo, hi + 0.01, 1.0)
    v = np.array([fun(m, d) for d in ds])
    ok = v > 0
    if not ok.any() or v[ok].max() < N:
        return float('nan')
    # scan from the top down for the first crossing
    for i in range(len(ds) - 1, 0, -1):
        if v[i] < N <= v[i - 1] or (v[i] == 0 and v[i - 1] >= N):
            if v[i] > 0:
                return float(optimize.brentq(lambda d: math.log(fun(m, d) / N), ds[i - 1], ds[i]))
            return float(ds[i - 1])
    return float('nan')

hw = []
for m in (300.0, 500.0, 700.0, 1000.0, 1100.0, 1150.0, 1300.0, 2000.0, 3000.0, 5000.0, 10000.0):
    f = min(1.0, f_th_higgsino(m))
    full = lambda mm, d: N_higgsino(mm, d)
    resc = lambda mm, d, f=f: f * N_higgsino(mm, d)
    hw.append(dict(m_GeV=m, f_thermal=f_th_higgsino(m), rho_fraction_used=f, dilution_D=max(1.0, f_th_higgsino(m)),
                   delta_N1_full_density=delta_for_N(full, m, 1.0), delta_lo_full=delta_for_N(full, m, LZ_BAND[1]), delta_hi_full=delta_for_N(full, m, LZ_BAND[0]),
                   delta_N1_thermal_density=delta_for_N(resc, m, 1.0), delta_lo_thermal=delta_for_N(resc, m, LZ_BAND[1]), delta_hi_thermal=delta_for_N(resc, m, LZ_BAND[0]),
                   N_at_300keV_thermal=resc(m, 300.0), N_at_366keV_thermal=resc(m, 366.0), N_at_366keV_full=full(m, 366.0)))
HW = pd.DataFrame(hw); HW.to_csv(f'{OUT}/higgsino_delta_windows.csv', index=False)
say(HW.to_string(index=False))

# ------------------------------------------------------------------------------------------------
# Part E: heavy Z' contact limit (P054 S3): LZ and <sigma v> both ~ G^2 -> coupling-independent N_th(m, delta)
# ------------------------------------------------------------------------------------------------
say("Part E: heavy Z' contact limit")
S_MODEL = {'B-L': 6.5, 'U(1)_B': 2.0, "A'(proton-only, S=8)": 8.0}   # sum_f N_c (g_f/g_N)^2 (chiral 1/2 for nu_L), massless f; top open for sqrt(s) = 2m >= 600 GeV
def G_relic_contact(m, S):
    """thermal G = g_chi g_N/m_med^2 [GeV^-2]: sigma_eff = 1/2 <sigma v>(chi chibar -> ff) = S G^2 m^2/(2 pi) = <sigma v>_req."""
    return math.sqrt(2 * math.pi * sv_req_of_m(m) / (S * m**2))
def G_LZ(m, d, Nfun=N_iso, N=1.0):
    nu = Nfun(m, d)
    return math.sqrt(N / nu) / MV**2 if nu > 0 else float('nan')
DGRID = np.arange(250.0, 390.01, 2.5)
erows = []
for m in MGRID:
    for d in DGRID:
        nu_iso, nu_p = N_iso(m, d), N_p(m, d)
        row = dict(m_GeV=m, delta_keV=d, N_iso_unit=nu_iso, N_p_unit=nu_p,
                   G_LZ_iso=G_LZ(m, d) if nu_iso > 0 else float('nan'), G_LZ_p=G_LZ(m, d, N_p) if nu_p > 0 else float('nan'))
        for model, S in S_MODEL.items():
            Gr = G_relic_contact(m, S)
            nu = nu_p if model.startswith("A'") else nu_iso
            Nth = nu * (Gr * MV**2)**2
            key = {'B-L': 'BL', 'U(1)_B': 'UB', "A'(proton-only, S=8)": 'Ap'}[model]
            row[f'G_relic_{key}'] = Gr
            row[f'N_th_{key}'] = Nth                       # = Omega_th(G_LZ)/Omega_DM (coupling-independent)
            row[f'f_th_{key}'] = Nth
            row[f'NT_fraction_{key}'] = max(0.0, 1 - Nth) if Nth > 0 else float('nan')
            row[f'dilution_D_{key}'] = max(1.0, Nth)
            row[f'T_RH_NT_{key}'] = T_RH_nonthermal(m, 0.5 * S * (row['G_LZ_p' if key == 'Ap' else 'G_LZ_iso'])**2 * m**2 / math.pi) if 0 < Nth < 1 else float('nan')
        erows.append(row)
E = pd.DataFrame(erows); E.to_csv(f'{OUT}/zprime_contact_map.csv', index=False)
for mshow in (300.0, 1000.0, 3000.0, 10000.0):
    e1 = E[np.isclose(E.m_GeV, mshow)]
    say('  m = %.0f GeV: ' % mshow)
    say(e1[np.isin(e1.delta_keV, [250.0, 270.0, 300.0, 350.0, 365.0, 380.0])][['delta_keV', 'G_LZ_iso', 'G_relic_BL', 'G_relic_UB', 'N_th_BL', 'N_th_UB', 'N_th_Ap', 'T_RH_NT_BL', 'T_RH_NT_UB']].to_string(index=False))
# delta at which N_th = 1, 0.105 for each m and model
zw = []
for m in (300.0, 500.0, 1000.0, 2000.0, 3000.0, 5000.0, 10000.0):
    r = dict(m_GeV=m)
    for key, S, Nfun in (('BL', 6.5, N_iso), ('UB', 2.0, N_iso), ('Ap', 8.0, N_p)):
        Gr = G_relic_contact(m, S)
        fun = lambda mm, d, Gr=Gr, Nfun=Nfun: Nfun(mm, d) * (Gr * MV**2)**2
        r[f'delta_Nth1_{key}'] = delta_for_N(fun, m, 1.0, lo=250.0)          # NaN with Nth_250 < 1 means 'below 250 keV'
        r[f'delta_Nth0p105_{key}'] = delta_for_N(fun, m, LZ_BAND[0], lo=250.0)
        r[f'Nth_250_{key}'] = fun(m, 250.0)
        r[f'Nth_300_{key}'] = fun(m, 300.0); r[f'Nth_350_{key}'] = fun(m, 350.0); r[f'Nth_366_{key}'] = fun(m, 366.0)
    zw.append(r)
ZW = pd.DataFrame(zw); ZW.to_csv(f'{OUT}/zprime_contact_windows.csv', index=False)
say(ZW.to_string(index=False))
say('  P054 cross-check (1 TeV, 300 keV; P054 used g*=86.25 and its own <sigma v>_req): N_th B-L = %.3f (P054 0.095), U(1)_B = %.3f (P054 0.31)'
    % (float(ZW[ZW.m_GeV == 1000.0].Nth_300_BL.iloc[0]), float(ZW[ZW.m_GeV == 1000.0].Nth_300_UB.iloc[0])))

# ------------------------------------------------------------------------------------------------
# Part F: dark photon (secluded A'A' freeze-out; LZ fixes eps^2 alpha_D / m_A'^4)
# ------------------------------------------------------------------------------------------------
say('Part F: dark photon')
def alpha_D_thermal(m, convention='eff'):
    """A'A' channel: sigma_eff = pi alpha_D^2/(2 m^2) (P025 'eff': sigma_11 = sigma_22 = pi alpha_D^2/m^2, sigma_12 = 0)
       or pi alpha_D^2/m^2 (P011). alpha_D such that sigma_eff = <sigma v>_req(m)."""
    fac = 2.0 if convention == 'eff' else 1.0
    return math.sqrt(fac * m**2 * sv_req_of_m(m) / math.pi)
def sigma_p_N1_cm2(m, d, N=1.0):
    nu = N_p(m, d)
    if nu <= 0: return float('nan')
    cp = math.sqrt(N / nu) / MV**2                      # GeV^-2
    mu_p = m * M_NUC / (m + M_NUC)
    return cp**2 * mu_p**2 / math.pi * GEV2_CM2         # sigma_p = c_p^2 mu_p^2/pi  (P011 convention)
def eps_required(m, d, mA, aD):
    """sigma_p = 16 pi alpha alpha_D eps^2 mu_p^2/m_A'^4 (P011)."""
    sp = sigma_p_N1_cm2(m, d) / GEV2_CM2
    mu_p = m * M_NUC / (m + M_NUC)
    return math.sqrt(sp * mA**4 / (16 * math.pi * ALPHA_EM * aD * mu_p**2))
def mA_ceiling(m, d, aD, eps_max=EPS_MAX):
    sp = sigma_p_N1_cm2(m, d) / GEV2_CM2
    mu_p = m * M_NUC / (m + M_NUC)
    return (16 * math.pi * ALPHA_EM * aD * eps_max**2 * mu_p**2 / sp)**0.25
S_ALLOWED_1TEV = 23.0     # P025: Planck-allowed saturated Sommerfeld factor at 1 TeV for the thermal A'A' cross-section (f_eff = 0.35)
def mA_floor_planck(m, aD, aD_th):
    """P025 criterion S_sat = 6 alpha_D m/m_A' <= S_allowed, S_allowed = 23 (m/TeV) (alpha_D_th/alpha_D)^2 (p_ann ~ S <sigma v>_0/m, <sigma v>_0 ~ alpha_D^2)."""
    S_allowed = S_ALLOWED_1TEV * (m / 1000.0) * (aD_th / aD)**2
    return 6 * aD * m / S_allowed
def chi2_floor_P011(d):
    """P011/P026: m_A' floor from the requirement that chi2 has decayed (exothermic bound); 2.45/0.92/0.55/0.25 GeV at 300/350/365/380 keV; 5.0 at 250 (P011 table)."""
    return float(np.exp(np.interp(d, [250.0, 300.0, 350.0, 365.0, 380.0], np.log([5.0, 2.45, 0.92, 0.55, 0.25]))))
frows = []
for m in MGRID:
    aD = alpha_D_thermal(m)
    for d in DGRID:
        sp = sigma_p_N1_cm2(m, d)
        if not np.isfinite(sp):
            frows.append(dict(m_GeV=m, delta_keV=d, alpha_D_thermal=aD)); continue
        ceil = mA_ceiling(m, d, aD); floor = max(mA_floor_planck(m, aD, aD), chi2_floor_P011(d))
        # dilution needed to reopen the window: alpha_D = k alpha_D_th, floor ~ k^3, ceiling ~ k^(1/4); D = k^-2
        k_open = min(1.0, (ceil / floor)**(4.0 / 11.0)) if ceil < floor else 1.0
        frows.append(dict(m_GeV=m, delta_keV=d, alpha_D_thermal=aD, sigma_p_N1_cm2=sp,
                          eps_mA1GeV=eps_required(m, d, 1.0, aD), eps_mA10GeV=eps_required(m, d, 10.0, aD),
                          mA_ceiling_GeV=ceil, mA_floor_planck_GeV=mA_floor_planck(m, aD, aD), mA_floor_chi2_GeV=chi2_floor_P011(d),
                          window_open=ceil > floor, window_ratio=ceil / floor, k_alphaD_to_open=k_open, dilution_D_to_open=k_open**-2,
                          alpha_D_diluted=k_open * aD))
F = pd.DataFrame(frows); F.to_csv(f'{OUT}/darkphoton_map.csv', index=False)
say('  alpha_D(thermal, eff) = %.4f (300 GeV), %.4f (1 TeV), %.4f (3 TeV), %.4f (10 TeV)' % tuple(alpha_D_thermal(x) for x in (300.0, 1000.0, 3000.0, 10000.0)))
for mshow in (1000.0, 3000.0):
    f1 = F[np.isclose(F.m_GeV, mshow)]
    say('  m = %.0f GeV:' % mshow)
    say(f1[np.isin(f1.delta_keV, [250.0, 300.0, 350.0, 365.0, 380.0])][['delta_keV', 'sigma_p_N1_cm2', 'eps_mA10GeV', 'mA_ceiling_GeV', 'mA_floor_planck_GeV', 'mA_floor_chi2_GeV', 'window_open', 'dilution_D_to_open', 'alpha_D_diluted']].to_string(index=False))
# delta at which the thermal window closes, per mass; dilution needed at 350/365/380 keV
def D_to_open(m, d):
    aD = alpha_D_thermal(m)
    sp = sigma_p_N1_cm2(m, d)
    if not np.isfinite(sp): return float('nan')
    ceil = mA_ceiling(m, d, aD); floor = max(mA_floor_planck(m, aD, aD), chi2_floor_P011(d))
    return 1.0 if ceil > floor else (ceil / floor)**(-8.0 / 11.0)
dw = []
for m in (300.0, 500.0, 1000.0, 2000.0, 3000.0, 5000.0, 10000.0):
    aD = alpha_D_thermal(m)
    g = lambda d: math.log(mA_ceiling(m, d, aD) / max(mA_floor_planck(m, aD, aD), chi2_floor_P011(d)))
    ds = [d for d in DGRID if np.isfinite(sigma_p_N1_cm2(m, d))]
    vals = [g(d) for d in ds]
    dclose = float('nan')
    if vals[0] <= 0: dclose = -1.0                        # closed already at 250 keV
    else:
        for i in range(1, len(ds)):
            if vals[i - 1] > 0 >= vals[i]:
                dclose = optimize.brentq(g, ds[i - 1], ds[i]); break
        if not np.isfinite(dclose) or dclose == float('nan'): dclose = ds[-1] if vals[-1] > 0 else dclose
    dw.append(dict(m_GeV=m, alpha_D_thermal=aD, alpha_D_P011conv=alpha_D_thermal(m, 'P011'), mA_floor_planck_GeV=mA_floor_planck(m, aD, aD),
                   ratio_250=math.exp(vals[0]), ratio_300=math.exp(g(300.0)) if np.isfinite(sigma_p_N1_cm2(m, 300.0)) else float('nan'),
                   delta_window_closes_keV=dclose, D_to_open_300=D_to_open(m, 300.0), D_to_open_350=D_to_open(m, 350.0), D_to_open_365=D_to_open(m, 365.0), D_to_open_380=D_to_open(m, 380.0)))
DW = pd.DataFrame(dw); DW.to_csv(f'{OUT}/darkphoton_windows.csv', index=False)
say(DW.to_string(index=False))
# s-channel fraction at the LZ coupling for realistic m_A' (does the P054 fixed point apply to the dark photon?)
srows = []
for m in (300.0, 1000.0, 3000.0):
    for d in (300.0, 350.0, 366.0):
        Gp = G_LZ(m, d, N_p); Gr = G_relic_contact(m, 8.0)
        ratio_contact = (Gp / Gr)**2                 # <sigma v>_s-channel(contact, LZ coupling)/<sigma v>_req = 1/N_th_Ap
        for mA in (1.0, 10.0, 100.0, 1000.0, 5000.0, 20000.0):
            prop = mA**4 / ((4 * m**2 - mA**2)**2 + 1e-300)   # (m_A'^2/(s - m_A'^2))^2 at s = 4m^2 relative to contact 1/m_A'^4
            srows.append(dict(m_GeV=m, delta_keV=d, mA_GeV=mA, s_channel_over_required=ratio_contact * prop, N_th_contact_Ap=1 / ratio_contact))
SR = pd.DataFrame(srows); SR.to_csv(f'{OUT}/darkphoton_schannel_fraction.csv', index=False)
say('  dark photon s-channel (chi1 chi2 -> A\'* -> ff) at the LZ coupling relative to the required <sigma v>, 1 TeV / 300 keV:')
say(SR[(SR.m_GeV == 1000.0) & (SR.delta_keV == 300.0)][['mA_GeV', 's_channel_over_required', 'N_th_contact_Ap']].to_string(index=False))

# ------------------------------------------------------------------------------------------------
# Part G: freeze-in / low-reheating production with the LZ coupling (UV contact operator G (chibar gamma chi)(fbar gamma f))
# ------------------------------------------------------------------------------------------------
say('Part G: freeze-in with the LZ coupling')
def R_production(T, m, G, S, mf_list=None):
    """pair-production rate density sum_f n_f n_fbar <sigma v>(ff -> chi chibar) [GeV^4], massless f, Maxwell-Boltzmann:
       R = sum_f (2N_c)^2 T/(32 pi^4) Int ds sigma_f(s) s^{3/2} K1(sqrt(s)/T),
       sigma_f(ff->chi chibar) = N_c (g_f/g_N)^2 G^2 beta_chi (s + 2m^2)/(12 pi N_c^2 ... ) -> total charge sum S:
       sum_f (2N_c)^2 sigma_f = 4 S G^2 beta_chi (s + 2m^2)/(12 pi)  (colour-averaged sigma_f carries 1/N_c; details.md S7)."""
    smin = 4 * m**2
    # s from threshold to (2m + 40 T)^2: the massless integrand u^6 K1(u) (u = sqrt(s)/T) peaks at u ~ 6.5 and is negligible beyond 40
    wmax = (2 * m + 40.0 * T)**2 / smin - 1.0
    w = np.concatenate([[0.0], np.geomspace(1e-7, wmax, 3000)])
    s = smin * (1 + w)
    beta = np.sqrt(np.maximum(1 - smin / s, 0.0))
    sig = 4 * S * G**2 * beta * (s + 2 * m**2) / (12 * math.pi)
    integrand = sig * s**1.5 * kve(1, np.sqrt(s) / T) * np.exp(-np.sqrt(s) / T)
    I = float(np.trapezoid(integrand, s))
    return T / (32 * math.pi**4) * I
def Y_freeze_in(T_RH, m, G, S):
    """Y(T -> 0) = Int_0^{T_RH} dT 2 R(T)/(s(T) H(T) T), starting from Y = 0 at T_RH (instantaneous reheating)."""
    Ts = np.geomspace(max(m / 60.0, 1e-3), T_RH, 400)
    vals = np.array([2 * R_production(T, m, G, S) / (s_of_T(T) * H_of_T(T) * T) for T in Ts])
    return float(np.trapezoid(vals, Ts))
def closed_form_UV(T_RH, m, G, S):
    """massless-chi limit: Y = (2 * 8 S G^2/pi^5) (45/(2 pi^2 g*)) (M_Pl/(1.66 sqrt g*)) T_RH^3/3."""
    g = gstar(T_RH)
    return 16 * S * G**2 / math.pi**5 * 45 / (2 * math.pi**2 * g) * M_PL / (1.66 * math.sqrt(g)) * T_RH**3 / 3
grows = []
for m in (300.0, 1000.0, 3000.0, 10000.0):
    for label, S, d, Nfun in (("Z' U(1)_B", 2.0, 300.0, N_iso), ("Z' B-L", 6.5, 300.0, N_iso), ("Z' U(1)_B", 2.0, 366.0, N_iso), ('Higgsino-equivalent isoscalar', 6.5, None, None)):
        G = math.sqrt(KAPPA_HIGGSINO) / MV**2 if d is None else G_LZ(m, d, Nfun)
        if not np.isfinite(G): continue
        Yreq = Y_required(m)
        # massless check of the numerical integrator at T_RH = 20 m
        chk = Y_freeze_in(20 * m, m, G, S) / closed_form_UV(20 * m, m, G, S)
        # yield if T_RH = 10 m (freeze-in regime would need Y << Yeq); compare with Yeq(T = m)
        Y10 = Y_freeze_in(10 * m, m, G, S)
        # T_RH giving exactly Y_req (Boltzmann-suppressed production below m)
        f = lambda lT: math.log(max(Y_freeze_in(10**lT, m, G, S), 1e-300) / Yreq)
        try:
            T_RH_req = 10**optimize.brentq(f, math.log10(m / 50.0), math.log10(10 * m), xtol=1e-3)
        except ValueError:
            T_RH_req = float('nan')
        grows.append(dict(m_GeV=m, model=label, delta_keV=d, G_GeV2=G, S=S, closed_form_check=chk, Y_required=Yreq,
                          Y_FI_TRH_10m=Y10, overproduction_TRH_10m=Y10 / Yreq, Yeq_at_T_eq_m=Yeq(1.0, m),
                          thermalises=Y10 > Yeq(1.0, m), T_RH_for_Omega_DM_GeV=T_RH_req, m_over_T_RH=m / T_RH_req if np.isfinite(T_RH_req) else float('nan')))
        say('  m=%5.0f %-30s delta=%s: G=%.2e, massless check %.3f, Y_FI(T_RH=10m)/Y_req = %.2e (thermalises: %s), T_RH(Omega_DM) = %.1f GeV = m/%.1f'
            % (m, label, d, G, chk, Y10 / Yreq, Y10 > Yeq(1.0, m), T_RH_req, m / T_RH_req))
GR = pd.DataFrame(grows); GR.to_csv(f'{OUT}/freeze_in_low_reheating.csv', index=False)

# ------------------------------------------------------------------------------------------------
# Part H: asymmetric DM -- chi <-> chibar oscillations from the Majorana splitting
# ------------------------------------------------------------------------------------------------
say('Part H: asymmetric DM')
t_osc = {d: HBAR_GEV_S / (d * 1e-6) for d in (250.0, 300.0, 380.0)}   # 1/delta in seconds
T_at_tosc = {d: math.sqrt(M_PL / (1.66 * math.sqrt(106.75) * 2 * t / HBAR_GEV_S)) for d, t in t_osc.items()}   # radiation era: t = 1/(2H)
say('  oscillation time 1/delta = %.2e s (300 keV); the radiation-era temperature at t = 1/delta is %.2e GeV' % (t_osc[300.0], T_at_tosc[300.0]))
json.dump(dict(t_osc_s=t_osc, T_radiation_at_t_osc_GeV=T_at_tosc), open(f'{OUT}/asymmetric_oscillation.json', 'w'), indent=1)

# ------------------------------------------------------------------------------------------------
# Part I: map figure and summary
# ------------------------------------------------------------------------------------------------
say('Part I: figures')
Mm, Dd = np.meshgrid(MGRID, DGRID, indexing='ij')
def field(df, col):
    Z = np.full(Mm.shape, np.nan)
    for i, m in enumerate(MGRID):
        sub = df[np.isclose(df.m_GeV, m)].set_index('delta_keV')
        for j, d in enumerate(DGRID):
            if d in sub.index:
                v = sub.loc[d, col]
                Z[i, j] = float(v) if np.isfinite(v) else np.nan
    return Z
# Higgsino field: log10 N_LZ at thermal density (capped at 1) ; f_th vs m
ZH = np.full(Mm.shape, np.nan)
for i, m in enumerate(MGRID):
    f = min(1.0, f_th_higgsino(m))
    for j, d in enumerate(DGRID):
        n = f * N_higgsino(m, d)
        ZH[i, j] = math.log10(n) if n > 0 else np.nan
with np.errstate(divide='ignore', invalid='ignore'):
    ZBL = np.log10(field(E, 'N_th_BL')); ZUB = np.log10(field(E, 'N_th_UB'))
    ZDP = np.log10(field(F, 'window_ratio'))
    ZBL[~np.isfinite(ZBL)] = np.nan; ZUB[~np.isfinite(ZUB)] = np.nan; ZDP[~np.isfinite(ZDP)] = np.nan

fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0), sharex=True, sharey=True)
cmap = plt.get_cmap('RdYlBu_r')
panels = [(axes[0, 0], ZH, 'Higgsino: log10 N_LZ (thermal density, f_th capped at 1)', (-3, 3)),
          (axes[0, 1], ZUB, "heavy Z' U(1)_B contact: log10 N_th = log10 (Omega_th/Omega_DM at LZ coupling)", (-3, 3)),
          (axes[1, 0], ZBL, "heavy Z' B-L contact: log10 N_th", (-3, 3)),
          (axes[1, 1], ZDP, "dark photon (A'A' thermal): log10 [m_A' ceiling(eps<1e-3) / floor(Planck, chi2 decay)]", (-2, 2))]
for ax, Z, title, (vmin, vmax) in panels:
    pc = ax.pcolormesh(Mm / 1000.0, Dd, Z, cmap=cmap, vmin=vmin, vmax=vmax, shading='nearest')
    ax.set_xscale('log'); ax.set_title(title, fontsize=9)
    fig.colorbar(pc, ax=ax, shrink=0.85)
    if Z is ZDP:
        cs = ax.contour(Mm / 1000.0, Dd, Z, levels=[0.0], colors='k', linewidths=1.5)
        ax.clabel(cs, fmt={0.0: 'window closes'}, fontsize=7)
    else:
        cs = ax.contour(Mm / 1000.0, Dd, Z, levels=[math.log10(LZ_BAND[0]), 0.0, math.log10(LZ_BAND[1])], colors='k', linewidths=[0.8, 1.6, 0.8], linestyles=['--', '-', '--'])
        ax.clabel(cs, fmt={math.log10(LZ_BAND[0]): '0.105', 0.0: 'N=1', math.log10(LZ_BAND[1]): '3.65'}, fontsize=7)
    if Z is ZH:
        ax.axvline(mu_th / 1000.0, color='k', ls=':', lw=1); ax.text(mu_th / 1000.0 * 1.05, 255, 'thermal\n%.0f GeV' % mu_th, fontsize=7)
for ax in axes[1]: ax.set_xlabel(r'$m_\chi$ [TeV]')
for ax in axes[:, 0]: ax.set_ylabel(r'$\delta$ [keV]')
fig.suptitle('P075: production mechanism vs the LZ-required coupling (annual-mean halo, 2.84 t yr). '
             'Red: over-abundant or too many events; blue: under-abundant at the LZ coupling.', fontsize=8)
fig.tight_layout(); fig.savefig(f'{FIG}/P075_fig1_mechanism_map.png', dpi=160); plt.close(fig)

# Fig 2: thermal fraction and required non-thermal quantities vs mass
fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
ax = axes[0]
ax.plot(D.m_GeV / 1000, D.omega_h2_thermal, label='Higgsino (ADG coannihilation)')
ax.axhline(OMEGA_H2, color='k', ls='--', lw=0.8); ax.axvline(mu_th / 1000, color='k', ls=':', lw=0.8)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'$m_\chi$ [TeV]'); ax.set_ylabel(r'thermal $\Omega h^2$'); ax.legend(fontsize=8)
ax.set_title('Higgsino relic (Boltzmann ODE, g*(T))', fontsize=9)
ax = axes[1]
for key, lab in (('BL', "Z' B-L"), ('UB', "Z' U(1)_B"), ('Ap', "A' contact (proton-only, S=8)")):
    for d, ls in ((300.0, '-'), (350.0, '--')):
        sub = E[E.delta_keV == d]
        ax.plot(sub.m_GeV / 1000, sub[f'N_th_{key}'], ls=ls, label='%s, %.0f keV' % (lab, d))
ax.axhspan(LZ_BAND[0], LZ_BAND[1], color='0.85')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'$m_\chi$ [TeV]'); ax.set_ylabel(r'$N_{th}$ = $\Omega_{th}/\Omega_{DM}$ at LZ coupling'); ax.legend(fontsize=6)
ax.set_title('contact-limit fixed point (grey: LZ 90% band)', fontsize=9)
ax = axes[2]
ax.plot(D.m_GeV / 1000, D.T_RH_nonthermal_GeV, label='Higgsino: T_RH (annihilation-limited NT)')
sub = E[E.delta_keV == 300.0]; ax.plot(sub.m_GeV / 1000, sub.T_RH_NT_UB, label="Z' U(1)_B, 300 keV: T_RH")
sub = E[E.delta_keV == 350.0]; ax.plot(sub.m_GeV / 1000, sub.T_RH_NT_UB, ls='--', label="Z' U(1)_B, 350 keV: T_RH")
ax.plot(D.m_GeV / 1000, D.T_fo_GeV, color='k', ls=':', label='T_fo = m/x_f')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'$m_\chi$ [TeV]'); ax.set_ylabel('T [GeV]'); ax.legend(fontsize=7)
ax.set_title('reheating temperature for a non-thermal top-up', fontsize=9)
fig.tight_layout(); fig.savefig(f'{FIG}/P075_fig2_thermal_quantities.png', dpi=160); plt.close(fig)

summary = dict(
    higgsino_thermal_mass_GeV=mu_th, higgsino_thermal_band_GeV=[m_lo_th, m_hi_th], relic_validation=val,
    higgsino_1TeV=dict(omega=float(D[np.isclose(D.m_GeV, 1000.0)].omega_h2_thermal.iloc[0]), f=float(D[np.isclose(D.m_GeV, 1000.0)].f_thermal.iloc[0])),
    interpolation_test=itest.to_dict('records'),
    coannihilation_delta_effect_max=float(C[C.channel != 'chargino_weight'].one_minus_ratio.abs().max()),
    zprime_windows=ZW.to_dict('records'), darkphoton_windows=DW.to_dict('records'), higgsino_windows=HW.to_dict('records'),
    freeze_in=GR.to_dict('records'), asymmetric=dict(t_osc_s=t_osc, T_GeV=T_at_tosc), runtime_s=time.time() - T_START)
json.dump(summary, open(f'{OUT}/P075_summary.json', 'w'), indent=1, default=float)
say('done in %.0f s' % (time.time() - T_START))
