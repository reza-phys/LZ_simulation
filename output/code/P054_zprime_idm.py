"""P054: Pseudo-Dirac dark matter with a heavy Z' -- LZ rate, relic density, colliders, splitting, chi2 lifetime.

Model: L = Z'_mu [ g_chi chibar gamma^mu chi + sum_q g_q qbar gamma^mu q (+ leptons for B-L) ].
In the Majorana basis chi = (chi1 + i chi2)/sqrt2 the vector current is off-diagonal, so
  * scattering is inelastic chi1 N -> chi2 N with c_p = c_n = g_chi g_N / m_Z'^2 (g_N = 3 g_q nucleon charge; isoscalar O1),
  * annihilation is chi1 chi2 -> Z'* -> f fbar (s-wave), plus chi_i chi_i -> Z' Z' if m_Z' < m_chi.
Run from the simulation root:  .venv/bin/python output/code/P054_zprime_idm.py
"""
import sys, os, math, json, time
sys.path.insert(0, 'output/code')
import numpy as np
import pandas as pd
from scipy import integrate, optimize, special
from scipy.special import kve, erf, erfc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from common import lzcommon as lz

OUT = 'output/work/P054'; FIG = f'{OUT}/figures'
os.makedirs(FIG, exist_ok=True)
LOG = open(f'{OUT}/run_log.txt', 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.write(s + '\n'); LOG.flush()

# ------------------------------------------------------------------------------------------------
# constants (recalled, certain unless noted)
# ------------------------------------------------------------------------------------------------
MV = lz.M_V_GEV                      # 246.2 GeV: LZ/Anand unit coupling c = 1/m_v^2
GEV2_CM2 = 0.3894e-27                # (hbar c)^2 in GeV^-2 -> cm^2
C_CMS = 2.99792458e10
GEV2_TO_CM3S = GEV2_CM2 * C_CMS      # 1 GeV^-2 x c = 1.1674e-17 cm^3/s
HBAR_GEV_S = 6.582119569e-25
M_PL = 1.220890e19                   # GeV
OMEGA_H2 = 0.120                     # Planck 2018 (certain)
GSTAR = 86.25                        # g_* at T ~ 20-50 GeV (likely; 10% band)
T_UNIV_S = 4.35e17                   # 13.8 Gyr
ALPHA_EM = 1 / 137.036; SW2 = 0.2312; MZ = 91.1876; G_WEAK = 0.6517  # g = e/sin(theta_W)
M_NUC = lz.M_NUCLEON_GEV
Y_ELECTRON = math.sqrt(2) * 0.000511 / MV   # 2.94e-6 (SM electron Yukawa)

# SM fermions: (name, N_c, mass, charge-ratio dict per model, chiral flag)
#   B-L: quarks 1/3, charged leptons -1, nu_L -1 (LH only); U(1)_B: quarks 1/3 only.  Nucleon charge = 1 in both.
FERMIONS = [('u', 3, 0.0022, 1/3), ('d', 3, 0.0047, 1/3), ('s', 3, 0.093, 1/3), ('c', 3, 1.27, 1/3),
            ('b', 3, 4.18, 1/3), ('t', 3, 172.7, 1/3),
            ('e', 1, 0.000511, 1.0), ('mu', 1, 0.10566, 1.0), ('tau', 1, 1.7769, 1.0),
            ('nu', 3, 0.0, 1.0)]   # nu entry: 3 flavours, LH only -> factor 1/2 applied below
def fermion_list(model):
    if model == 'B-L':
        return FERMIONS
    return [f for f in FERMIONS if f[1] == 3 and f[0] not in ('nu',)]   # U(1)_B: quarks only
def sum_ncg2(model):
    """sum_f N_c (g_f/g_N)^2 x chiral factor, massless limit."""
    tot = 0.0
    for name, nc, mf, r in fermion_list(model):
        tot += nc * r**2 * (0.5 if name == 'nu' else 1.0)
    return tot

# ------------------------------------------------------------------------------------------------
# Part A: LZ requirement  G = g_chi g_N / m_Z'^2  from P011's isoscalar unit-coupling grid
# ------------------------------------------------------------------------------------------------
p11 = pd.read_csv('output/work/P011/sigma_p_required.csv')
p11 = p11[(p11.halo == 'annual') & (p11.eff_sigma_keV == 11.5)]
GRID = {m: p11[p11.m_GeV == m].sort_values('delta_keV')[['delta_keV', 'N_iso_unit']].values for m in (300.0, 1000.0, 3000.0)}

def N_unit(delta, m=1000.0):
    """Expected LZ events (2.84 t yr, annual-mean Baxter halo, LZ efficiency) for the isoscalar unit coupling
    c_p = c_n = 1/m_v^2 (WimPyDD c^0 = 2/m_v^2), log-interpolated on P011's 5-keV grid."""
    d, n = GRID[m][:, 0], GRID[m][:, 1]
    ok = n > 0
    return float(np.exp(np.interp(delta, d[ok], np.log(n[ok]), right=-np.inf)))

def G_LZ(delta, m=1000.0, N=1.0):
    """g_chi g_N / m_Z'^2 [GeV^-2] giving N events: kappa = (G m_v^2)^2 = N / N_unit."""
    return math.sqrt(N / N_unit(delta, m)) / MV**2

def sigma_n_cm2(G, m=1000.0):
    mu = m * M_NUC / (m + M_NUC)
    return G**2 * mu**2 / math.pi * GEV2_CM2

DELTAS = [250.0, 280.0, 300.0, 320.0, 340.0, 350.0, 360.0, 366.0, 370.0, 380.0]
rowsA = []
say('Part A: LZ requirement (isoscalar O1; B-L and U(1)_B both have nucleon charge 1 for p and n)')
for m in (300.0, 1000.0, 3000.0):
    for d in DELTAS:
        nu = N_unit(d, m)
        if not np.isfinite(nu) or nu <= 0:
            continue
        G1 = G_LZ(d, m)
        rowsA.append(dict(m_chi_GeV=m, delta_keV=d, N_unit=nu, kappa_N1=1 / nu, G_N1_GeV2=G1,
                          G_N3p65=G_LZ(d, m, 3.65), G_N0p105=G_LZ(d, m, 0.105), sigma_n_N1_cm2=sigma_n_cm2(G1, m),
                          gg_at_mZp_1TeV=G1 * 1e6, gg_at_mZp_3TeV=G1 * 9e6, gg_at_mZp_10TeV=G1 * 1e8,
                          mZp_max_TeV_gg1=math.sqrt(1.0 / G1) / 1e3, mZp_max_TeV_gg4pi=math.sqrt(4 * math.pi / G1) / 1e3))
A = pd.DataFrame(rowsA); A.to_csv(f'{OUT}/lz_requirement.csv', index=False)
say(A[A.m_chi_GeV == 1000][['delta_keV', 'N_unit', 'kappa_N1', 'G_N1_GeV2', 'sigma_n_N1_cm2', 'gg_at_mZp_1TeV', 'gg_at_mZp_3TeV', 'mZp_max_TeV_gg1']].to_string(index=False))
# cross-checks against P021 (kappa_hat, shape-aware likelihood) and P007's digitised LZ Fig. 6 upper edges
P021_KAPPA = {300.0: 7.4e-5, 350.0: 3.865e-3, 365.0: 0.0402, 380.0: 2.194}
say('kappa(N=1) this work vs P021 kappa_hat:', {d: (round(1 / N_unit(d), 7), k, round((1 / N_unit(d)) / k, 2)) for d, k in P021_KAPPA.items()})
LZ_EDGE = {300.0: 2.6e-4, 350.0: 2.25e-2}   # P007 digitised 90% upper edges of LZ Fig. 6 (1 TeV), (c m_v^2)^2
say('LZ Fig.6 90% upper edge -> G_upper [GeV^-2]:', {d: '%.3e' % (math.sqrt(k) / MV**2) for d, k in LZ_EDGE.items()},
    ' (events at edge: %.2f, %.2f)' % tuple(k * N_unit(d) for d, k in LZ_EDGE.items()))
# Higgsino-equivalent isoscalar coupling (P007): (c_eq m_v^2)^2 = 0.0777
G_HIGGSINO_EQ = math.sqrt(0.0777) / MV**2
say('Higgsino-equivalent isoscalar G = %.3e GeV^-2 (G_F/sqrt2 = %.3e)' % (G_HIGGSINO_EQ, 1.1663787e-5 / math.sqrt(2)))

# ------------------------------------------------------------------------------------------------
# Part A2: live WimPyDD validation of two grid points (1000 GeV, delta = 300 and 365 keV), P011 recipe
# ------------------------------------------------------------------------------------------------
say('Part A2: WimPyDD validation of the reused grid')
VGRID = np.linspace(0.0, 844.0, 1200)
days12 = 15.0 + 365.25 / 12 * np.arange(12)
deta_annual = np.mean([lz.wd_halo(day_of_year=d, vmin=VGRID)[1] for d in days12], axis=0)
E50_HI, E50_LO, PLATEAU = lz.LZ['E_50pct_high_keV'], lz.LZ['E_50pct_low_keV'], lz.LZ['eff_plateau']
def efficiency(E, sig_hi=11.5, sig_lo=2.5):
    return PLATEAU * 0.5 * (1 + erf((E - E50_LO) / (math.sqrt(2) * sig_lo))) * 0.5 * erfc((E - E50_HI) / (math.sqrt(2) * sig_hi))
ham_iso = lz.wd_hamiltonian('P054_iso_unit', {1: (2.0 / MV**2, 0.0)})
val = {}
for d in (300.0, 365.0):
    lo = lz.E_R_range_keV(1000.0, VGRID[-1], A=124.0, delta_kev=d)[0]
    E = np.arange(max(1.0, math.floor(lo) - 4.0), 330.0 + 1e-9, 3.0)
    r = lz.wd_rate(ham_iso, 1000.0, E, halo=(VGRID, deta_annual), delta_kev=d)
    Ef = np.linspace(1.0, 330.0, 3001)      # whole ROI (the Xe inelastic window at delta = 300 keV, 1 TeV starts near 91 keV, P015)
    y = np.interp(Ef, E, r, left=0.0, right=0.0) * efficiency(Ef)
    N = float(np.trapezoid(y, Ef)) * lz.LZ['exposure_tyr']
    val[d] = dict(live=N, grid=N_unit(d), ratio=N / N_unit(d))
    say('  delta=%.0f keV: live WimPyDD N_unit = %.1f, P011 grid = %.1f, ratio %.3f' % (d, N, N_unit(d), N / N_unit(d)))
json.dump(val, open(f'{OUT}/wimpydd_validation.json', 'w'), indent=1)

# ------------------------------------------------------------------------------------------------
# Part B: relic density machinery
# ------------------------------------------------------------------------------------------------
def width_Zp(M, gN, gchi, m_chi, model):
    """Gamma(Z' -> SM ff) + Gamma(Z' -> chi chibar) [GeV], vector couplings; nu_L chiral (1/2)."""
    G = 0.0
    for name, nc, mf, r in fermion_list(model):
        if M <= 2 * mf:
            continue
        gf = r * gN
        b = math.sqrt(1 - 4 * mf**2 / M**2)
        if name == 'nu':
            G += nc * gf**2 * M / (24 * math.pi)
        else:
            G += nc * gf**2 * M * b * (1 + 2 * mf**2 / M**2) / (12 * math.pi)
    if M > 2 * m_chi:
        b = math.sqrt(1 - 4 * m_chi**2 / M**2)
        G += gchi**2 * M * b * (1 + 2 * m_chi**2 / M**2) / (12 * math.pi)
    return G

def sigma_ff_over_G2(s, m, M, Gam, model):
    """sigma(chi chibar -> Z'* -> sum_f f fbar)(s) per unit G^2 = (g_chi g_N/M^2)^2, i.e. sigma = G^2 * M^4 * [this].
    Derived (details.md S2): sigma = N_c g_chi^2 g_f^2 beta_f (s+2m^2)(s+2m_f^2) / (12 pi s beta_chi D(s)), D = (s-M^2)^2 + M^2 Gamma^2.
    Vectorised in s."""
    s = np.asarray(s, dtype=float)
    D = (s - M**2)**2 + M**2 * Gam**2
    bchi = np.sqrt(np.maximum(1 - 4 * m**2 / s, 1e-300))
    tot = np.zeros_like(s)
    for name, nc, mf, r in fermion_list(model):
        fac = 0.5 if name == 'nu' else 1.0
        open_ = s > 4 * mf**2
        bf = np.sqrt(np.maximum(1 - 4 * mf**2 / s, 0.0))
        tot += np.where(open_, nc * r**2 * fac * bf * (s + 2 * m**2) * (s + 2 * mf**2), 0.0)
    return tot / (12 * math.pi * s * bchi * D)

def sigv_ff_thermal(x, m, M, Gam, model):
    """Gondolo-Gelmini thermal average <sigma v_Mol>(x) of the s-channel process, per unit G^2:
    <sv> = x/(4 K2(x)^2) Int du sigma(s) u(4+u)(2+u)^2 K1(x(2+u)),  s = m^2 (2+u)^2  (details.md S2).
    Fixed-grid trapezoid with a dense patch of half-width 30 Gamma/(2m) around the Z' pole."""
    ures = M / m - 2.0
    umax = 60.0 / x + 1.0
    u = np.concatenate([[0.0], np.geomspace(1e-6 / x, umax, 2500)])   # log grid: the weight e^{-xu} lives at u ~ 1/x
    if ures > 0:
        w = max(Gam / (2 * m), 1e-9)
        u = np.concatenate([u, ures + np.linspace(-30, 30, 1201) * w, ures + np.linspace(-300, 300, 601) * w])
        u = np.unique(u[(u > 0) & (u <= max(umax, ures + 300 * w))])
    s = m**2 * (2 + u)**2
    f = sigma_ff_over_G2(s, m, M, Gam, model) * M**4 * u * (4 + u) * (2 + u)**2 * kve(1, x * (2 + u)) * np.exp(-x * u)
    f[0] = 0.0
    return x / (4 * kve(2, x)**2) * float(np.trapezoid(f, u))

def sigv_ZpZp(gchi, m, M):
    """s-wave chi chibar -> Z'Z' (t/u channel), Dirac: g^4/(16 pi m^2) (1-r^2)^{3/2}/(1-r^2/2)^2 [recalled, likely]."""
    if M >= m:
        return 0.0
    r2 = (M / m)**2
    return gchi**4 / (16 * math.pi * m**2) * (1 - r2)**1.5 / (1 - r2 / 2)**2

XGRID = np.logspace(math.log10(3.0), 3.0, 36)
def sigv_shape_table(m, M, Gam, model):
    return np.array([sigv_ff_thermal(x, m, M, Gam, model) for x in XGRID])

def omega_h2(m, sigv_eff_of_x):
    """Solve dY/dx = -sqrt(pi/45) M_Pl m sqrt(g*) <sigma v>_eff (Y^2 - Yeq^2)/x^2, g = 4 internal dof (chi1, chi2 x 2 spins)."""
    g = 4.0
    lam = math.sqrt(math.pi / 45) * M_PL * m * math.sqrt(GSTAR)
    def Yeq(x):
        return 45 * g / (4 * math.pi**4 * GSTAR) * x**2 * kve(2, x) * math.exp(-x)
    def rhs(x, y):
        return [-lam / x**2 * sigv_eff_of_x(x) * (y[0]**2 - Yeq(x)**2)]
    x0 = 3.0
    sol = integrate.solve_ivp(rhs, [x0, 1000.0], [Yeq(x0)], method='Radau', rtol=1e-7, atol=1e-30)
    return 2.755e8 * sol.y[0, -1] * m

def omega_semi(m, sigv_eff_of_x, cal=1.0):
    """Fast semi-analytic freeze-out (Kolb-Turner / Gondolo-Gelmini): x_f from the iterated criterion
    x_f = ln[0.038 g M_Pl m <sv>(x_f)/sqrt(g*)] - 0.5 ln x_f, then Omega h^2 = 1.07e9 GeV^-1 / (sqrt(g*) M_Pl J), J = Int_{x_f}^inf <sv>/x^2 dx.
    'cal' is the ODE/semi-analytic calibration factor measured below for a constant cross-section."""
    g = 4.0
    xf = 25.0
    for _ in range(12):
        arg = 0.038 * g * M_PL * m * max(sigv_eff_of_x(xf), 1e-300) / math.sqrt(GSTAR)
        xf = max(math.log(arg) - 0.5 * math.log(xf), 3.0)
    xs = np.geomspace(xf, 2e4, 600)
    sv = np.array([sigv_eff_of_x(x) for x in xs])
    J = float(np.trapezoid(sv / xs**2, xs)) + sv[-1] / xs[-1]
    return cal * 1.07e9 / (math.sqrt(GSTAR) * M_PL * J)

CAL = {}   # filled in the validation block: Omega_ODE / Omega_semi for a constant <sigma v>

def omega_model(m, M, gchi, gN, model, shape=None, fast=False):
    """Omega h^2 for given couplings; shape = precomputed ff thermal table per unit G^2 (recomputed if None).
    fast=True uses the calibrated semi-analytic estimate (map), else the full Boltzmann ODE."""
    Gam = width_Zp(M, gN, gchi, m, model)
    if shape is None:
        shape = sigv_shape_table(m, M, Gam, model)
    G2 = (gchi * gN / M**2)**2
    zz = sigv_ZpZp(gchi, m, M)
    logs = np.log(np.maximum(shape * G2, 1e-300))
    def sv_eff(x):
        return 0.5 * (math.exp(np.interp(math.log(x), np.log(XGRID), logs)) + zz)   # sigma_eff = sigma_{chi chibar}/2
    if fast:
        return omega_semi(m, sv_eff, CAL.get(m, 1.0))
    return omega_h2(m, sv_eff)

# validation of the relic machinery
say('Part B: relic machinery validation')
def sv_const(v):
    return lambda x: v
for m in (300.0, 1000.0, 3000.0):
    om = omega_h2(m, sv_const(2.2e-26 / GEV2_TO_CM3S))
    oms = omega_semi(m, sv_const(2.2e-26 / GEV2_TO_CM3S))
    CAL[m] = om / oms
    say('  constant <sigma v>_eff = 2.2e-26 cm^3/s, m = %.0f GeV: Omega h^2 (ODE) = %.4f, semi-analytic = %.4f -> calibration %.3f' % (m, om, oms, CAL[m]))
def solve_const_sigv(m):
    f = lambda lg: math.log(omega_h2(m, sv_const(10**lg)) / OMEGA_H2)
    return 10**optimize.brentq(f, -12, -6, xtol=1e-4)
SV_REQ = {m: solve_const_sigv(m) for m in (300.0, 1000.0, 3000.0)}
say('  required constant <sigma v>_eff for Omega h^2 = 0.12: ', {m: '%.3e cm^3/s' % (v * GEV2_TO_CM3S) for m, v in SV_REQ.items()})
# thermal-average check: contact limit must reproduce the s-wave threshold value  sum N_c g_f^2 g_chi^2 m^2/(pi (4m^2-M^2)^2)
m, M = 1000.0, 20000.0
tab = sigv_shape_table(m, M, width_Zp(M, 0.3, 1.0, m, 'U(1)_B'), 'U(1)_B')
thr = sum_ncg2('U(1)_B') * M**4 * m**2 / (math.pi * (4 * m**2 - M**2)**2)
say('  contact-limit check (M=20 TeV, U(1)_B): <sigma v>(x=1000)/threshold = %.4f, (x=20) = %.4f' % (tab[-1] / thr, np.interp(20, XGRID, tab) / thr))
# alpha_D check vs P025: pi alpha^2/m^2 = 4.4e-26 -> alpha = 0.035 at 1 TeV
def gchi_thermal_ZZ(m, M):
    f = lambda g: math.log(omega_semi(m, sv_const(0.5 * sigv_ZpZp(g, m, M)), CAL.get(m, 1.0)) / OMEGA_H2)
    return optimize.brentq(f, 0.05, 3.5, xtol=1e-4)
g_th = gchi_thermal_ZZ(1000.0, 100.0)
say('  Z\'Z\'-only thermal coupling at 1 TeV, M=100 GeV: g_chi = %.3f, alpha_chi = %.4f (P025: 0.035)' % (g_th, g_th**2 / (4 * math.pi)))
# semi-analytic vs ODE for resonant and off-resonant s-channel cases (U(1)_B, 1 TeV, g_N = 0.6)
res_val = []
for M, gchi in ((500.0, 0.3), (1900.0, 0.05), (2000.0, 0.01), (2100.0, 0.05), (3000.0, 1.0)):
    o1 = omega_model(1000.0, M, gchi, 0.6, 'U(1)_B'); o2 = omega_model(1000.0, M, gchi, 0.6, 'U(1)_B', fast=True)
    res_val.append(dict(mZp_GeV=M, gchi=gchi, Omega_ODE=o1, Omega_semi=o2, ratio=o2 / o1))
    say('  semi/ODE check M=%.0f g_chi=%.2f: Omega ODE %.4g, semi %.4g, ratio %.3f' % (M, gchi, o1, o2, o2 / o1))
pd.DataFrame(res_val).to_csv(f'{OUT}/relic_method_validation.csv', index=False)

# ------------------------------------------------------------------------------------------------
# Part C: the contact-limit tension and the thermal fixed point N_thermal(delta) (coupling independent)
# ------------------------------------------------------------------------------------------------
say('Part C: contact limit (m_Z\' >> 2 m_chi): both LZ and relic scale with G^2')
rowsC = []
for m in (300.0, 1000.0, 3000.0):
    for model in ('B-L', 'U(1)_B'):
        S = sum_ncg2(model)
        # thermal <sigma v>_{chi chibar} = 2 x required sigma_eff; contact: S G^2 m^2 / pi
        G_rel = math.sqrt(2 * SV_REQ[m] * math.pi / (S * m**2))
        for d in (250.0, 270.0, 300.0, 350.0):
            nu = N_unit(d, m)
            if nu <= 0:
                continue
            G1 = G_LZ(d, m)
            rowsC.append(dict(m_chi_GeV=m, model=model, delta_keV=d, G_LZ=G1, G_relic_contact=G_rel, ratio_G=G1 / G_rel,
                              ratio_rate=(G1 / G_rel)**2, Omega_h2_at_LZ_coupling=OMEGA_H2 / (G1 / G_rel)**2,
                              N_thermal_fixed_point=nu * (G_rel * MV**2)**2))
        # delta at which the thermal contact Z' gives exactly one event
        f = lambda d: math.log(N_unit(d, m) * (G_rel * MV**2)**2)
        try:
            d1 = optimize.brentq(f, 200.0, GRID[m][GRID[m][:, 1] > 0][-1, 0] - 1)
        except ValueError:
            d1 = float('nan')
        say('  m=%.0f %s: G_relic(contact) = %.3e GeV^-2, delta(N_thermal=1) = %.1f keV; N_thermal(300 keV) = %.3f' %
            (m, model, G_rel, d1, N_unit(300.0, m) * (G_rel * MV**2)**2 if N_unit(300.0, m) > 0 else float('nan')))
        rowsC.append(dict(m_chi_GeV=m, model=model, delta_keV=float('nan'), G_relic_contact=G_rel, delta_N_thermal_1=d1))
C = pd.DataFrame(rowsC); C.to_csv(f'{OUT}/contact_limit_tension.csv', index=False)
say(C[(C.m_chi_GeV == 1000) & C.delta_keV.notna()][['model', 'delta_keV', 'G_LZ', 'G_relic_contact', 'ratio_G', 'ratio_rate', 'Omega_h2_at_LZ_coupling', 'N_thermal_fixed_point']].to_string(index=False))

# ------------------------------------------------------------------------------------------------
# Part D: recalled collider ceilings (FLAGGED uncertain) and the maximal thermal LZ count map N_max(m_Z', delta)
# ------------------------------------------------------------------------------------------------
def gq_max_dijet(M):
    """Recalled/uncertain: ATLAS/CMS dijet + low-mass (ISR-assisted) dijet limits on a leptophobic Z' with universal g_q."""
    if M < 500: return 0.05
    if M < 1500: return 0.10
    if M < 3000: return 0.20
    if M < 5000: return 0.30
    return 1.0                                  # effectively unconstrained; perturbativity only
def gN_max(M, model):
    if model == 'U(1)_B':
        return min(3 * gq_max_dijet(M), math.sqrt(4 * math.pi))
    # B-L: g_N = g'; LHC dilepton M/g' >~ 20 TeV (uncertain), LEP-II contact M/g' >~ 7 TeV (likely)
    return min(M / 20000.0, math.sqrt(4 * math.pi))
GCHI_MAX = math.sqrt(4 * math.pi)

def best_thermal_point(m, M, model, gN=None):
    """Maximal g_chi g_N compatible with Omega h^2 = 0.12 (and g_chi <= sqrt(4 pi)) at collider-maximal g_N.
    Returns dict with gchi, gN, G, Omega, channels; feasible=False if overabundant even at g_chi max."""
    gN = gN_max(M, model) if gN is None else gN
    Gam0 = width_Zp(M, gN, 1.0, m, model)
    shape = sigv_shape_table(m, M, Gam0, model)
    def f(lg):
        return math.log(omega_model(m, M, 10**lg, gN, model, shape, fast=True) / OMEGA_H2)
    lo, hi = -3.0, math.log10(GCHI_MAX)
    fhi = f(hi)
    if fhi > 0:      # overabundant even at maximal g_chi
        return dict(feasible=False, gchi=GCHI_MAX, gN=gN, G=GCHI_MAX * gN / M**2, Omega=OMEGA_H2 * math.exp(fhi))
    flo = f(lo)
    if flo < 0:      # underabundant even at tiny g_chi (resonance): still a root below 1e-3 -> extend
        lo = -6.0
        if f(lo) < 0:
            return dict(feasible=True, gchi=1e-6, gN=gN, G=1e-6 * gN / M**2, Omega=OMEGA_H2, sigv_ff_x20_cm3s=np.nan, sigv_ZZ_cm3s=0.0, Gamma_over_M=np.nan)
    lg = optimize.brentq(f, lo, hi, xtol=1e-3)
    gchi = 10**lg
    # refine width with actual gchi near resonance
    if abs(M - 2 * m) < 0.3 * M:
        shape = sigv_shape_table(m, M, width_Zp(M, gN, gchi, m, model), model)
        try:
            lg = optimize.brentq(lambda l: math.log(omega_model(m, M, 10**l, gN, model, shape, fast=True) / OMEGA_H2), -6.0, hi, xtol=1e-3)
            gchi = 10**lg
        except ValueError:
            pass
    G2 = (gchi * gN / M**2)**2
    sv_ff0 = float(np.interp(20.0, XGRID, shape)) * G2 * GEV2_TO_CM3S
    return dict(feasible=True, gchi=gchi, gN=gN, G=gchi * gN / M**2, Omega=OMEGA_H2,
                sigv_ff_x20_cm3s=sv_ff0, sigv_ZZ_cm3s=sigv_ZpZp(gchi, m, M) * GEV2_TO_CM3S, Gamma_over_M=width_Zp(M, gN, gchi, m, model) / M)

say('Part D: maximal thermal LZ count map (this is the slow part)')
MZP = np.logspace(2, 4, 29)          # 100 GeV .. 10 TeV
rowsD = []
t0 = time.time()
for m in (300.0, 1000.0, 3000.0):
    for model in ('B-L', 'U(1)_B'):
        for M in MZP:
            bp = best_thermal_point(m, M, model)
            Gmax = bp['G']
            row = dict(m_chi_GeV=m, model=model, mZp_GeV=M, feasible=bp['feasible'], gchi=bp['gchi'], gN=bp['gN'], G_max=Gmax,
                       Omega_h2=bp['Omega'], sigv_ff_cm3s=bp.get('sigv_ff_x20_cm3s', np.nan), sigv_ZZ_cm3s=bp.get('sigv_ZZ_cm3s', np.nan),
                       Gamma_over_M=bp.get('Gamma_over_M', np.nan))
            for d in DELTAS:
                nu = N_unit(d, m)
                row[f'Nmax_{int(d)}'] = nu * (Gmax * MV**2)**2 if (bp['feasible'] and nu > 0) else (np.nan if nu <= 0 else 0.0)
            # delta at which N_max = 1
            if bp['feasible']:
                f = lambda d: math.log(max(N_unit(d, m), 1e-300) * (Gmax * MV**2)**2)
                dtop = GRID[m][GRID[m][:, 1] > 0][-1, 0] - 1
                try:
                    row['delta_Nmax1_keV'] = optimize.brentq(f, 200.0, dtop) if f(200.0) > 0 > f(dtop) else (float('nan') if f(200.0) < 0 else dtop)
                except ValueError:
                    row['delta_Nmax1_keV'] = float('nan')
            else:
                row['delta_Nmax1_keV'] = float('nan')
            rowsD.append(row)
        say('  done m=%.0f %s (%.0f s)' % (m, model, time.time() - t0))
D = pd.DataFrame(rowsD); D.to_csv(f'{OUT}/thermal_max_map.csv', index=False)
for model in ('B-L', 'U(1)_B'):
    s = D[(D.m_chi_GeV == 1000) & (D.model == model)]
    say(f'  1 TeV {model}:'); say(s[['mZp_GeV', 'feasible', 'gchi', 'gN', 'G_max', 'Nmax_300', 'Nmax_350', 'Nmax_366', 'delta_Nmax1_keV']].to_string(index=False))
summ = {}
for m in (300.0, 1000.0, 3000.0):
    for model in ('B-L', 'U(1)_B'):
        s = D[(D.m_chi_GeV == m) & (D.model == model) & D.feasible]
        summ[f'{int(m)}_{model}'] = dict(delta_max_thermal_keV=float(np.nanmax(s.delta_Nmax1_keV)) if len(s) else None,
                                         at_mZp_GeV=float(s.mZp_GeV.iloc[int(np.nanargmax(s.delta_Nmax1_keV.values))]) if len(s) else None,
                                         heavy_only_mZp_ge_2TeV_delta_max=float(np.nanmax(s[s.mZp_GeV >= 2000].delta_Nmax1_keV)) if len(s[s.mZp_GeV >= 2000]) else None)
say('  delta_max for a thermal relic + LZ + colliders:', json.dumps(summ, indent=1))

# ------------------------------------------------------------------------------------------------
# Part E: Majorana splitting from a charge-2q_chi scalar; chi2 lifetime; Sommerfeld/CMB
# ------------------------------------------------------------------------------------------------
def splitting(M, gchi, delta_keV):
    """S with U(1)' charge 2 q_chi: delta = y_S v_S/sqrt2, m_Z' = g' q_S v_S = 2 g_chi v_S."""
    vS = M / (2 * gchi)
    yS = math.sqrt(2) * delta_keV * 1e-6 / vS
    return vS, yS
def tau_chi2_nunu(Gnu, delta_keV):
    """Gamma(chi2 -> chi1 nu nubar) = N_nu G^2 delta^5/(120 pi^3) (P011 derivation, LH neutrinos), N_nu = 3."""
    dl = delta_keV * 1e-6
    Gam = 3 * Gnu**2 * dl**5 / (120 * math.pi**3)
    return HBAR_GEV_S / Gam
TAU_FLOOR = 7.2e16      # P026/P011: chi2 must have decayed (f2 < 1.2e-3 at delta = 300 keV)
def sommerfeld_S0(alpha, m, M):
    """Hulthen saturation value (v -> 0, off resonance): S0 = 2 pi^2/(eps' (1 - cos(2 pi/sqrt eps'))), eps' = pi^2/6 x M/(alpha m)."""
    ep = math.pi**2 / 6 * M / (alpha * m)
    return 2 * math.pi**2 / (ep * (1 - math.cos(2 * math.pi / math.sqrt(ep))))
PLANCK_PANN = 3.5e-28   # cm^3 s^-1 GeV^-1 (recalled, likely); f_eff = 0.35 (P025)
F_EFF = 0.35

say('Part E: splitting Yukawa, chi2 lifetime, Sommerfeld')
rowsE = []
for M in (300.0, 1000.0, 3000.0, 10000.0):
    for gchi in (0.1, 0.66, 1.0, 3.0):
        for d in (300.0, 366.0):
            vS, yS = splitting(M, gchi, d)
            rowsE.append(dict(mZp_GeV=M, gchi=gchi, delta_keV=d, vS_GeV=vS, yS=yS, yS_over_ye=yS / Y_ELECTRON))
E = pd.DataFrame(rowsE); E.to_csv(f'{OUT}/splitting_yukawa.csv', index=False)
say('  y_S for delta=300 keV: v_S = 1 TeV -> %.2e, 10 TeV -> %.2e, 100 TeV -> %.2e (y_e = %.2e)' %
    (math.sqrt(2) * 3e-4 / 1e3, math.sqrt(2) * 3e-4 / 1e4, math.sqrt(2) * 3e-4 / 1e5, Y_ELECTRON))
# chi2 lifetime B-L: G_nu = G_LZ exactly (nu charge magnitude = nucleon charge)
rowsL = []
for d in DELTAS:
    G1 = G_LZ(d, 1000.0)
    tau_BL = tau_chi2_nunu(G1, d)
    # U(1)_B: tree level absent; via Z-Z' mass mixing theta: G_nu = g_chi sin(theta) (g/2c_W)/m_Z^2 -> required sin(theta) for tau < floor
    cW = math.sqrt(1 - SW2)
    Gnu_req = math.sqrt(HBAR_GEV_S * 120 * math.pi**3 / (3 * TAU_FLOOR * (d * 1e-6)**5))
    sin_req_over_gchi = Gnu_req / (G_WEAK / (2 * cW) / MZ**2)
    rowsL.append(dict(delta_keV=d, G_LZ=G1, tau_BL_s=tau_BL, tau_BL_over_floor=tau_BL / TAU_FLOOR, Gnu_required_for_floor=Gnu_req,
                      sin_thetaZZp_x_gchi_required=sin_req_over_gchi))
L = pd.DataFrame(rowsL); L.to_csv(f'{OUT}/chi2_lifetime.csv', index=False)
say(L.to_string(index=False))
# Sommerfeld / CMB at the Z'Z'-thermal coupling (only relevant if m_Z' < m_chi; else chi1 chi1 has no tree-level annihilation)
rowsS = []
for M in (100.0, 300.0, 500.0, 900.0):
    g = gchi_thermal_ZZ(1000.0, M)
    al = g**2 / (4 * math.pi)
    S0 = sommerfeld_S0(al, 1000.0, M)
    sv = 0.5 * sigv_ZpZp(g, 1000.0, M) * GEV2_TO_CM3S
    rowsS.append(dict(mZp_GeV=M, gchi_thermal=g, alpha_chi=al, eps_phi=M / (al * 1000.0), S0_saturated=S0, sigv_eff_cm3s=sv,
                      sigv_CMB_cm3s=sv * S0, Planck_limit_cm3s=PLANCK_PANN * 1000.0 / F_EFF, ratio_to_Planck=sv * S0 / (PLANCK_PANN * 1000.0 / F_EFF)))
S = pd.DataFrame(rowsS); S.to_csv(f'{OUT}/sommerfeld_cmb.csv', index=False)
say(S.to_string(index=False))

# ------------------------------------------------------------------------------------------------
# Part F: benchmark points
# ------------------------------------------------------------------------------------------------
say('Part F: benchmarks')
def benchmark(tag, model, m, M, delta, gN=None, gchi=None):
    G1 = G_LZ(delta, m)
    if gN is None and gchi is None:
        # walk along the LZ constraint g_chi g_N = G_LZ M^2 and find g_chi with Omega h^2 = 0.12 (Z'Z' grows with g_chi)
        def f(lg):
            g = 10**lg
            return math.log(omega_model(m, M, g, G1 * M**2 / g, model, fast=True) / OMEGA_H2)
        lo, hi = math.log10(0.02), math.log10(GCHI_MAX)
        try:
            gchi = 10**optimize.brentq(f, lo, hi, xtol=1e-3)
        except ValueError:
            gchi = GCHI_MAX if f(hi) > 0 else 10**lo
        gN = G1 * M**2 / gchi
    elif gchi is None:
        gchi = G1 * M**2 / gN
    elif gN is None:
        gN = G1 * M**2 / gchi
    G = gchi * gN / M**2
    N_full = N_unit(delta, m) * (G * MV**2)**2
    om = omega_model(m, M, gchi, gN, model)
    N_scaled = N_full * min(1.0, om / OMEGA_H2)
    vS, yS = splitting(M, gchi, delta)
    if model == 'B-L':
        tau = tau_chi2_nunu(G, delta)
    else:
        tau = float('nan')   # needs Z-Z' mixing; see chi2_lifetime.csv
    coll_ok = gN <= gN_max(M, model) * 1.0001
    pert_ok = gchi <= GCHI_MAX
    row = dict(tag=tag, model=model, m_chi_GeV=m, mZp_GeV=M, delta_keV=delta, gchi=gchi, gN=gN, gq=gN / 3, G_GeV2=G,
               alpha_chi=gchi**2 / (4 * math.pi), N_LZ_full_density=N_full, Omega_h2=om, N_LZ_rescaled=N_scaled,
               sigma_n_cm2=sigma_n_cm2(G, m), vS_GeV=vS, yS=yS, tau_chi2_s=tau, collider_ok=coll_ok, perturbative=pert_ok,
               Zp_to_chichi_open=M > 2 * m, monojet_relevant=M > 2 * m)
    say('  %-3s %-6s m=%.0f M=%.0f d=%.0f: g_chi=%.3f g_N=%.4f (g_q=%.4f) G=%.2e N_full=%.2f Omega h2=%.3f N_resc=%.2f v_S=%.0f y_S=%.1e tau=%.1e s coll=%s pert=%s' %
        (tag, model, m, M, delta, gchi, gN, gN / 3, G, N_full, om, N_scaled, vS, yS, tau, coll_ok, pert_ok))
    return row
B = []
B.append(benchmark('B1', 'U(1)_B', 1000.0, 500.0, 300.0))                 # light Z', thermal via Z'Z', g_N from LZ
B.append(benchmark('B2', 'U(1)_B', 1000.0, 300.0, 340.0))                 # light Z', higher delta
B.append(benchmark('B3', 'U(1)_B', 1000.0, 3000.0, 300.0, gN=0.6))        # heavy Z' at dijet ceiling, LZ-fixed g_chi -> Omega
B.append(benchmark('B4', 'U(1)_B', 1000.0, 3000.0, 270.0, gN=0.6))        # heavy Z' at the thermal fixed point
B.append(benchmark('B5', 'B-L', 1000.0, 1000.0, 300.0, gN=0.05))          # B-L at the dilepton ceiling
B.append(benchmark('B6', 'B-L', 1000.0, 3000.0, 300.0, gN=0.15))          # B-L heavy
B.append(benchmark('B7', 'U(1)_B', 1000.0, 500.0, 366.0, gN=0.15))        # Higgsino-like delta: needs g_chi > sqrt(4pi)?
B.append(benchmark('B8', 'U(1)_B', 1000.0, 200.0, 380.0, gN=0.15))        # P021 peak
B.append(benchmark('B9', 'U(1)_B', 1000.0, 2000.0, 300.0, gN=0.6))        # resonance m_Z' = 2 m_chi
B.append(benchmark('B10', 'U(1)_B', 3000.0, 1500.0, 300.0))               # 3 TeV chi
B.append(benchmark('B11', 'U(1)_B', 300.0, 150.0, 280.0))                 # 300 GeV chi
B.append(benchmark('B12', 'B-L', 1000.0, 1000.0, 300.0))                  # B-L along the LZ constraint: thermal solution
B.append(benchmark('B13', 'B-L', 1000.0, 300.0, 300.0))                   # B-L light Z': dilepton ceiling g' <= 0.015?
pd.DataFrame(B).to_csv(f'{OUT}/benchmarks.csv', index=False)

# ------------------------------------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------------------------------------
PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
INK, INK2, SURF = '#0b0b0b', '#52514e', '#fcfcfb'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'text.color': INK, 'axes.facecolor': SURF, 'figure.facecolor': SURF, 'axes.grid': True, 'grid.color': '#e6e5e2',
                     'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw=dict(width_ratios=[1.15, 1]))
dd = np.arange(200.0, 400.1, 1.0)
for i, m in enumerate((300.0, 1000.0, 3000.0)):
    g = np.array([G_LZ(d, m) if N_unit(d, m) > 0 else np.nan for d in dd])
    ax.plot(dd, g, color=PAL[i], lw=2 if m == 1000 else 1.3, label=f'LZ N = 1, m_χ = {m/1000:g} TeV')
    if m == 1000.0:
        ax.fill_between(dd, g / math.sqrt(3.65), g / math.sqrt(0.105), color=PAL[i], alpha=0.13, lw=0, label='0.105–3.65 events (90 % band)')
ax.axhline(G_HIGGSINO_EQ, color=INK2, lw=1.0, ls=':', label='Higgsino-equivalent (P007)')
s = D[(D.m_chi_GeV == 1000) & (D.model == 'U(1)_B')]
for M, ls, lab in ((500.0, '--', "U(1)_B, m_Z′ = 0.5 TeV: thermal (Z′Z′) × dijet ceiling"), (3000.0, '-.', "U(1)_B, m_Z′ = 3 TeV: thermal ff̄ (exact)"), (10000.0, (0, (1, 1)), "U(1)_B, m_Z′ = 10 TeV: thermal ff̄ (exact)")):
    r = s.iloc[int(np.argmin(np.abs(s.mZp_GeV.values - M)))]
    ax.axhline(r.G_max, color=PAL[3], lw=1.3, ls=ls, label=lab)
sb = D[(D.m_chi_GeV == 1000) & (D.model == 'B-L')]
r = sb.iloc[int(np.argmin(np.abs(sb.mZp_GeV.values - 1000.0)))]
ax.axhline(r.G_max, color=PAL[6], lw=1.3, ls='--', label='B−L, m_Z′ = 1 TeV: thermal × dilepton ceiling')
ax.set_yscale('log'); ax.set_xlim(200, 400); ax.set_ylim(1e-9, 1e-4)
ax.set_xlabel('mass splitting δ [keV]'); ax.set_ylabel("G = g_χ g_N / m_Z′²  [GeV⁻²]")
ax.set_title("Z′-portal coupling giving one LZ event (2.84 t·yr, annual halo)", fontsize=9.5)
ax.legend(fontsize=7.2, loc='upper left')
# right: log10 N_max map for U(1)_B, 1 TeV
dgrid = np.arange(200.0, 391.0, 2.0)
Z = np.full((len(MZP), len(dgrid)), np.nan)
for i, M in enumerate(MZP):
    r = s.iloc[i]
    if not r.feasible:
        continue
    for j, d in enumerate(dgrid):
        nu = N_unit(d, 1000.0)
        Z[i, j] = math.log10(max(nu * (r.G_max * MV**2)**2, 1e-12)) if nu > 0 else np.nan
cmap = LinearSegmentedColormap.from_list('div', ['#2a78d6', '#cde2fb', '#f0efec', '#f8cbb4', '#eb6834'])
norm = TwoSlopeNorm(vmin=-6, vcenter=0, vmax=6)
pc = ax2.pcolormesh(dgrid, MZP / 1000, np.clip(Z, -6, 6), cmap=cmap, norm=norm, shading='nearest')
cs = ax2.contour(dgrid, MZP / 1000, Z, levels=[math.log10(0.105), 0.0, math.log10(3.65)], colors=[INK2, INK, INK2], linewidths=[0.8, 1.6, 0.8], linestyles=['--', '-', '--'])
ax2.clabel(cs, fmt={math.log10(0.105): '0.105', 0.0: 'N = 1', math.log10(3.65): '3.65'}, fontsize=7)
inf = s[~s.feasible]
if len(inf):
    ax2.scatter(np.full(len(inf), 205.0), inf.mZp_GeV / 1000, marker='x', color=INK, s=14, label='overabundant at any g_χ ≤ √4π')
ax2.set_yscale('log'); ax2.set_xlabel('mass splitting δ [keV]'); ax2.set_ylabel("m_Z′ [TeV]")
ax2.set_title("U(1)_B, m_χ = 1 TeV: maximal LZ count for a thermal relic\nwithin recalled dijet ceilings (log₁₀ N_max)", fontsize=9.5)
ax2.axhline(1.0, color=INK2, lw=0.6, ls=':'); ax2.axhline(2.0, color=INK2, lw=0.6, ls=':')
ax2.text(392, 1.03, 'Z′Z′ opens', fontsize=7, color=INK2, ha='right'); ax2.text(392, 2.06, 'resonance', fontsize=7, color=INK2, ha='right')
cb = fig.colorbar(pc, ax=ax2, pad=0.02); cb.set_label('log₁₀ N_max (thermal, collider-allowed)')
ax2.grid(False)
fig.tight_layout(); fig.savefig(f'{FIG}/P054_fig1_G_vs_delta_and_Nmax_map.png', dpi=170); plt.close(fig)

json.dump(dict(lz_requirement_1TeV={int(r.delta_keV): dict(N_unit=r.N_unit, G_N1=r.G_N1_GeV2, sigma_n=r.sigma_n_N1_cm2) for _, r in A[A.m_chi_GeV == 1000].iterrows()},
               higgsino_equiv_G=G_HIGGSINO_EQ, sv_required_const=SV_REQ, thermal_delta_max=summ, wimpydd_validation=val,
               constants=dict(MV=MV, GSTAR=GSTAR, OMEGA_H2=OMEGA_H2, TAU_FLOOR=TAU_FLOOR, PLANCK_PANN=PLANCK_PANN, F_EFF=F_EFF)),
          open(f'{OUT}/P054_results.json', 'w'), indent=1, default=float)
say('done in %.0f s' % (time.time() - t0))
