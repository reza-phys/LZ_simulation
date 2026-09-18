#!/usr/bin/env python
"""
P086 -- BBN and CMB constraints on the mediator required for the LZ inelastic cross-section:
dark photons, dark Higgs and light Z' in the early Universe.

Run from the simulation root:  .venv/bin/python output/code/P086_mediator_cosmology.py
Outputs -> output/work/P086/ (CSV/JSON tables, run_log.txt) and output/work/P086/figures/ (PNG).

Contents
  1. g*(T), g*s(T) from exact Bose/Fermi integrals with a smoothed QCD transition; H(T), t(T), DeltaN_eff(T_dec).
  2. Dark photon A': partial widths (e, mu, tau, hadrons via a recalled piecewise R(s)), lifetime map in (m_A', eps),
     thermalisation line (inverse decays vs H), freeze-in yield below it, BBN (recalled KKM-type envelope) and
     FIRAS mu/y tests for late decays.
  3. Dark Higgs h_D: mass relation, relic yield from a Boltzmann solve (forbidden h h -> A'A' for m_h < m_A'),
     lifetime through the A' loop (l+l-), tree-level h -> A'*A'* -> 4f, and h -> A' f fbar (m_h > m_A');
     Higgs-portal mixing needed to decay before BBN.
  4. Self-interaction sigma_T/m of chi1 via A' exchange (classical Yukawa formulae, Tulin-Yu-Zurek) vs Bullet cluster.
  5. Heavy Z' (P054): prompt decays -- one-line check.
  6. Combined (m_A', eps) map at m_chi = 1 TeV, alpha_D = 0.1 with P011 band, chi2 floor, P025 Planck bound,
     this work's BBN/CMB/thermalisation/dark-Higgs lines and recalled accelerator ceilings.
All recalled numbers are flagged in the run log and in provenance/P086.json.
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from scipy import integrate, special, optimize, interpolate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "output/code")
from common import lzcommon as lz  # noqa: E402  (GEV_TO_CM2, M_NUCLEON_GEV)

T0 = time.time()
OUT = "output/work/P086"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
LOG = open(os.path.join(OUT, "run_log.txt"), "w")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")


# ----------------------------------------------------------------------------------------------------------
# 0. Constants (recalled; reliability in brackets)
# ----------------------------------------------------------------------------------------------------------
M_PL = 1.220890e19        # GeV, Planck mass [certain]
HBAR = 6.582119569e-25    # GeV s [certain]
ALPHA = 1 / 137.035999    # [certain]
E_CH = np.sqrt(4 * np.pi * ALPHA)
M_E, M_MU, M_TAU = 0.000510999, 0.1056584, 1.77686   # GeV [certain]
M_PI, M_K, M_ETA = 0.13957, 0.4937, 0.5479           # [certain]
M_RHO, G_RHO = 0.7753, 0.1491                          # [certain]
M_U, M_D, M_S, M_C, M_B, M_T = 0.0022, 0.0047, 0.095, 1.27, 4.18, 173.0  # [likely; MSbar-ish]
M_W, M_Z, M_H = 80.38, 91.19, 125.1                    # [certain]
V_EW = 246.2
T_QCD = 0.170             # GeV, smoothed quark-hadron transition [likely]
T_NU_DEC = 2.0e-3         # GeV, instantaneous neutrino decoupling [likely]
T_CMB0 = 2.3486e-13       # GeV (2.7255 K) [certain]
H0_GEV = 67.4 * 1e3 / 3.0857e22 * HBAR   # 67.4 km/s/Mpc -> s^-1 -> GeV [likely, Planck 2018]
OMEGA_M, OMEGA_L = 0.315, 0.685          # [likely]
RHO_C_OVER_S0 = 4.4e-10 * 0.12 / 0.12    # GeV: (Omega_DM h^2 = 0.12) <-> rho_DM/s0 = 0.44 eV [certain-ish]
S0 = 2891.2               # cm^-3, entropy density today [certain]
RHO_C_H2 = 1.0537e-5      # GeV cm^-3 [certain]
T_UNIVERSE_S = 4.35e17    # s [certain]
GEV2_TO_CM2 = lz.GEV_TO_CM2
GEV_TO_G = 1.78266e-24

# ----------------------------------------------------------------------------------------------------------
# 1. g*(T), g*s(T) from exact thermal integrals
# ----------------------------------------------------------------------------------------------------------
# species: (name, g, mass, statistics, sector)  sector: 'lep','gam','qcd_q','qcd_g','had','ew'
SPECIES = [
    ("gamma", 2, 0.0, "B", "gam"),
    ("e", 4, M_E, "F", "lep"), ("mu", 4, M_MU, "F", "lep"), ("tau", 4, M_TAU, "F", "lep"),
    ("nu", 6, 0.0, "F", "nu"),
    ("u", 12, M_U, "F", "qcd_q"), ("d", 12, M_D, "F", "qcd_q"), ("s", 12, M_S, "F", "qcd_q"),
    ("c", 12, M_C, "F", "qcd_q"), ("b", 12, M_B, "F", "qcd_q"), ("t", 12, M_T, "F", "qcd_q"),
    ("gluon", 16, 0.0, "B", "qcd_g"),
    ("pi", 3, M_PI, "B", "had"), ("K", 4, M_K, "B", "had"), ("eta", 1, M_ETA, "B", "had"),
    ("W", 6, M_W, "B", "ew"), ("Z", 3, M_Z, "B", "ew"), ("h", 1, M_H, "B", "ew"),
]


def _rho_p_integrals(y, stat):
    """(rho/(g T^4), p/(g T^4)) for one species with y = m/T."""
    sgn = -1.0 if stat == "B" else 1.0
    if y > 60:
        return 0.0, 0.0
    f_rho = lambda x: x * x * np.sqrt(x * x + y * y) / (np.exp(np.sqrt(x * x + y * y)) + sgn)
    f_p = lambda x: x ** 4 / np.sqrt(x * x + y * y) / (np.exp(np.sqrt(x * x + y * y)) + sgn) / 3.0
    r = integrate.quad(f_rho, 0, 60 + y, limit=200)[0] / (2 * np.pi ** 2)
    p = integrate.quad(f_p, 0, 60 + y, limit=200)[0] / (2 * np.pi ** 2)
    return r, p


def qcd_switch(T):
    """1 -> quarks/gluons, 0 -> hadrons; smooth over ~+-25 MeV around T_QCD."""
    return 0.5 * (1 + np.tanh((T - T_QCD) / 0.025))


_Tgrid = np.logspace(-10, 4, 561)   # 0.1 eV (z ~ 400) to 10 TeV
_gs, _gss, _gs_em, _gss_em = [], [], [], []
for T in _Tgrid:
    rho, sden, rho_em, s_em = 0.0, 0.0, 0.0, 0.0
    w = qcd_switch(T)
    for name, g, m, stat, sec in SPECIES:
        if sec == "nu":
            continue
        r, p = _rho_p_integrals(m / T, stat)
        fac = 1.0
        if sec in ("qcd_q", "qcd_g"):
            fac = w
        elif sec == "had":
            fac = 1 - w
        rho += fac * g * r
        sden += fac * g * (r + p)
        if sec in ("gam", "lep", "had", "qcd_q", "qcd_g", "ew"):
            rho_em += fac * g * r
            sden_em_add = fac * g * (r + p)
            s_em += sden_em_add
    _gs_em.append(rho * 30 / np.pi ** 2)
    _gss_em.append(sden * 45 / (2 * np.pi ** 2))
_gs_em = np.array(_gs_em)
_gss_em = np.array(_gss_em)
# neutrinos: coupled (T_nu = T) above T_NU_DEC, then T_nu/T = (g*s_em(T_dec)/g*s_em(T))^(1/3)
_gss_em_dec = np.interp(np.log(T_NU_DEC), np.log(_Tgrid), _gss_em)
_ratio = np.where(_Tgrid > T_NU_DEC, 1.0, (_gss_em / _gss_em_dec) ** (1 / 3))   # T_nu/T_gamma -> (4/11)^(1/3)
GS_TAB = _gs_em + 6 * 7 / 8 * _ratio ** 4
GSS_TAB = _gss_em + 6 * 7 / 8 * _ratio ** 3


def gstar(T):
    return np.interp(np.log(T), np.log(_Tgrid), GS_TAB)


def gstar_s(T):
    return np.interp(np.log(T), np.log(_Tgrid), GSS_TAB)


def hubble(T):
    """H(T) in GeV, radiation domination."""
    return 1.66 * np.sqrt(gstar(T)) * T ** 2 / M_PL


def entropy(T):
    return 2 * np.pi ** 2 / 45 * gstar_s(T) * T ** 3


def rho_gamma(T):
    return np.pi ** 2 / 15 * T ** 4


# time-temperature relation: dt = -dT/(H T) (1 + (1/3) dln g*s/dln T); include matter+Lambda for late times
_lnT = np.log(_Tgrid)
_dlngss = np.gradient(np.log(GSS_TAB), _lnT)


def hubble_full(T):
    z1 = T / T_CMB0
    return np.sqrt(hubble(T) ** 2 + H0_GEV ** 2 * (OMEGA_M * z1 ** 3 + OMEGA_L))


_t_tab = np.zeros_like(_Tgrid)
_t_tab[-1] = 1 / (2 * hubble(_Tgrid[-1]))
for i in range(len(_Tgrid) - 2, -1, -1):
    Ta, Tb = _Tgrid[i], _Tgrid[i + 1]
    f = lambda lnT: (1 + _dlngss[i] / 3) / hubble_full(np.exp(lnT))
    _t_tab[i] = _t_tab[i + 1] + integrate.quad(f, np.log(Ta), np.log(Tb))[0]
_t_tab *= HBAR   # GeV^-1 -> s


def t_of_T(T):
    return np.interp(np.log(T), _lnT, _t_tab)          # _lnT increasing, _t_tab decreasing: fine for np.interp


_logt_rev, _lnT_rev = np.log(_t_tab[::-1]), _lnT[::-1]   # increasing log t


def T_of_t(t):
    t = np.asarray(t, dtype=float)
    early = np.sqrt(M_PL * HBAR / (2 * 1.66 * np.sqrt(106.75) * t))   # radiation era above the table (T > 10 TeV); t in s
    return np.where(t < _t_tab[-1], early, np.exp(np.interp(np.log(np.maximum(t, _t_tab[-1])), _logt_rev, _lnT_rev)))


log("=== P086 mediator cosmology ===")
log("[1] g*(T) validation (exact integrals, smoothed QCD switch at %.0f MeV):" % (T_QCD * 1e3))
for T in [1e-5, 1e-4, 1e-3, 5e-3, 0.02, 0.1, 0.15, 0.2, 0.5, 1.0, 10.0, 100.0, 1e3]:
    log("   T = %8.4g GeV : g* = %7.3f  g*s = %7.3f   t = %.3g s" % (T, gstar(T), gstar_s(T), t_of_T(T)))
log("   (textbook: g*=106.75 above ~300 GeV, ~86 at 10 GeV, 10.75 at 10 MeV, 3.363/3.909 (g*/g*s) at T<<m_e)")
log("   t(T=1 MeV) = %.3g s, t(0.1 MeV) = %.3g s (textbook ~1 s and ~100-150 s)" % (t_of_T(1e-3), t_of_T(1e-4)))
log("   T(z=1100) = %.3g GeV -> t = %.3g s (textbook 1.2e13 s)" % (T_CMB0 * 1101, t_of_T(T_CMB0 * 1101)))
pd.DataFrame({"T_GeV": _Tgrid, "gstar": GS_TAB, "gstar_s": GSS_TAB, "t_s": _t_tab}).to_csv(
    os.path.join(OUT, "gstar_table.csv"), index=False)


# DeltaN_eff for a species (g_eff dof, still relativistic at CMB) that decoupled from the SM at T_dec
def delta_neff(T_dec, g_b=0, g_f=0):
    geff = g_b + 7 / 8 * g_f
    return geff / (7 / 4) * (gstar_s(T_NU_DEC) / gstar_s(T_dec)) ** (4 / 3)


rows = []
for Tdec in [2e-3, 5e-3, 1e-2, 3e-2, 0.1, 0.15, 0.2, 0.5, 1.0, 5.0, 50.0, 1000.0]:
    rows.append(dict(T_dec_GeV=Tdec, gstar_s=gstar_s(Tdec), dNeff_scalar_1dof=delta_neff(Tdec, 1, 0),
                     dNeff_vector_3dof=delta_neff(Tdec, 3, 0), dNeff_darksector_7p5=delta_neff(Tdec, 4, 4),
                     dNeff_Majorana_pair_4f=delta_neff(Tdec, 0, 4)))
NEFF = pd.DataFrame(rows)
NEFF.to_csv(os.path.join(OUT, "delta_neff_table.csv"), index=False)
log("[1b] DeltaN_eff of a relativistic relic vs T_dec (Planck bound DeltaN_eff < 0.3, recalled likely):")
log(NEFF.to_string(index=False, float_format=lambda x: "%.3g" % x))
T_scalar_ok = optimize.brentq(lambda lT: delta_neff(np.exp(lT), 1, 0) - 0.3, np.log(2e-3), np.log(10))
T_vector_ok = optimize.brentq(lambda lT: delta_neff(np.exp(lT), 3, 0) - 0.3, np.log(2e-3), np.log(10))
log("   1-dof scalar needs T_dec > %.3g GeV; 3-dof vector T_dec > %.3g GeV to satisfy DeltaN_eff<0.3"
    % (np.exp(T_scalar_ok), np.exp(T_vector_ok)))
# Boltzmann suppression of a m>=0.3 GeV species at neutrino decoupling
for m in [0.3, 1.0]:
    log("   n/n_rel of a m=%.1f GeV species at T_nu,dec=2 MeV: exp(-m/T) ~ %.1e -> DeltaN_eff = 0 in practice"
        % (m, np.exp(-m / T_NU_DEC)))

# ----------------------------------------------------------------------------------------------------------
# 2. Dark photon widths, lifetimes, thermalisation, freeze-in, BBN/CMB
# ----------------------------------------------------------------------------------------------------------


def smoothstep(x, x0, x1):
    u = np.clip((x - x0) / (x1 - x0), 0, 1)
    return u * u * (3 - 2 * u)


def R_had(rs):
    """R(s) = sigma(e+e- -> hadrons)/sigma(e+e- -> mu+mu-) at sqrt(s)=rs [recalled, uncertain, x1.5-2 below 2 GeV].
    2pi via rho Breit-Wigner pion form factor; continuum uds (2) from 1.0-1.5 GeV, charm (+4/3) from 3.7-4.2 GeV,
    bottom (+1/3) from 10.5-11 GeV; x(1+alpha_s/pi)~1.05.  Narrow omega/phi/J/psi/Upsilon poles ignored."""
    rs = np.asarray(rs, dtype=float)
    s = rs ** 2
    R = np.zeros_like(rs)
    m2pi = rs > 2 * M_PI
    Fpi2 = M_RHO ** 4 / ((s - M_RHO ** 2) ** 2 + M_RHO ** 2 * G_RHO ** 2)
    R = np.where(m2pi, 0.25 * np.clip(1 - 4 * M_PI ** 2 / s, 0, None) ** 1.5 * Fpi2, 0.0)
    R = R * (1 - smoothstep(rs, 1.0, 1.6)) + 1.05 * (2.0 * smoothstep(rs, 1.0, 1.5) + 4 / 3 * smoothstep(rs, 3.7, 4.2)
                                                      + 1 / 3 * smoothstep(rs, 10.5, 11.0))
    return R


def gamma_ll(m, eps, ml):
    r = np.clip(4 * ml ** 2 / m ** 2, 0, 1)
    return ALPHA * eps ** 2 * m / 3 * (1 + r / 2) * np.sqrt(1 - r)


def gamma_Ap(m, eps, parts=False):
    m = np.asarray(m, dtype=float)
    ge = gamma_ll(m, eps, M_E)
    gm = gamma_ll(m, eps, M_MU)
    gt = gamma_ll(m, eps, M_TAU)
    gh = ALPHA * eps ** 2 * m / 3 * R_had(m)
    tot = ge + gm + gt + gh
    if parts:
        return tot, ge, gm, gt, gh
    return tot


def tau_Ap(m, eps):
    return HBAR / gamma_Ap(m, eps)


log("[2] Dark photon widths and lifetimes")
for m in [0.3, 0.5, 0.775, 1, 3, 10, 30]:
    tot, ge, gm, gt, gh = gamma_Ap(m, 1e-6, parts=True)
    log("   m_A'=%5.2f GeV eps=1e-6: Gamma=%.3g GeV; BR(ee,mumu,tautau,had)=%.2f,%.2f,%.2f,%.2f; R=%.2f; tau=%.3g s"
        % (m, tot, ge / tot, gm / tot, gt / tot, gh / tot, R_had(m), HBAR / tot))

# lifetime grid
MA = np.logspace(np.log10(0.05), 2, 241)
EPS = np.logspace(-14, -2, 241)
MM, EE = np.meshgrid(MA, EPS)
TAU = HBAR / gamma_Ap(MM, EE)
pd.DataFrame(TAU, index=EPS, columns=MA).to_csv(os.path.join(OUT, "tau_Aprime_grid_s.csv"))
# eps for tau = 1 s and 1e-2 s
eps_tau1 = np.sqrt(HBAR / gamma_Ap(MA, 1.0) / 1.0)
eps_tau001 = np.sqrt(HBAR / gamma_Ap(MA, 1.0) / 1e-2)
eps_tau1e4 = np.sqrt(HBAR / gamma_Ap(MA, 1.0) / 1e4)
for m in [0.3, 1, 3, 10]:
    i = np.argmin(abs(MA - m))
    log("   tau_A'=1 s at eps=%.2e; tau=1e-2 s at eps=%.2e (m_A'=%.3g GeV)" % (eps_tau1[i], eps_tau001[i], MA[i]))
# lifetimes across the P011 band
for (m, e) in [(0.3, 1.1e-7), (1, 8.7e-7), (3, 7.4e-6), (10, 8.3e-5), (30, 7.5e-4)]:
    log("   P011 band point m_A'=%g GeV, eps=%.1e (delta=300 keV): tau_A' = %.2e s, i.e. t=tau at T ~ %.3g GeV (>> m_A': decays in equilibrium)"
        % (m, e, tau_Ap(m, e), float(T_of_t(tau_Ap(m, e)))))


# thermalisation: inverse decays  Gamma_ID(T) = Gamma K1(x)/K2(x); thermalised if max_T Gamma_ID/H >= 1
def k1k2(x):
    return special.kve(1, x) / special.kve(2, x)


# freeze-in yield from inverse decays (valid while Y << Y_eq):  dY/dT = -gamma/(s H T), gamma = g m^2 T K1(m/T) Gamma/(2 pi^2)
def Y_freeze_in(m, eps):
    G = gamma_Ap(m, eps)

    def integrand(lnT):
        T = np.exp(lnT)
        gam = 3 * m ** 2 * T * special.kve(1, m / T) * np.exp(-m / T) * G / (2 * np.pi ** 2)
        return gam / (entropy(T) * hubble(T))   # dY/dlnT (positive, integrate over lnT)
    return integrate.quad(integrand, np.log(m / 60), np.log(200 * m), limit=200)[0]


def Y_eq_boson(m, T, g=3):
    x = m / T
    return 45 / (4 * np.pi ** 4) * g / gstar_s(T) * x ** 2 * special.kve(2, x) * np.exp(-x)


# Thermalisation criterion A (adopted): the freeze-in production would reach the equilibrium abundance, Y_FI(eps) = Y_eq(T=m).
# Criterion B (cross-check): Gamma_dec K1/K2 = H at T = m/3, where the inverse-decay production rate per unit comoving
# volume peaks.  (Gamma K1/K2 / H keeps growing towards T << m, but there no fermions have the energy to produce an A'.)
eps_therm = np.array([np.sqrt(Y_eq_boson(m, m) / Y_freeze_in(m, 1.0)) for m in MA])
eps_therm_B = np.array([np.sqrt(hubble(m / 3) / (gamma_Ap(m, 1.0) * k1k2(3.0))) for m in MA])
pd.DataFrame({"m_A_GeV": MA, "eps_therm_YFI_eq_Yeq": eps_therm, "eps_therm_GammaK1K2_eq_H_at_m_over_3": eps_therm_B,
              "eps_tau_1s": eps_tau1, "eps_tau_1e-2s": eps_tau001, "eps_tau_1e4s": eps_tau1e4}).to_csv(
    os.path.join(OUT, "eps_thermalisation.csv"), index=False)
log("[2b] Thermalisation with the SM plasma via f fbar <-> A' (criterion A: Y_FI = Y_eq(T=m); B: Gamma K1/K2 = H at T=m/3):")
for m in [0.1, 0.3, 1, 3, 10, 30]:
    i = np.argmin(abs(MA - m))
    log("   m_A'=%5.2f GeV: eps_therm(A) = %.2e, (B) = %.2e; tau_A' at eps_therm(A) = %.2e s; 1/H(T=m) = %.2e s"
        % (MA[i], eps_therm[i], eps_therm_B[i], tau_Ap(MA[i], eps_therm[i]), HBAR / hubble(MA[i])))

# kinetic coupling of chi (1 TeV) to SM via A' exchange: Gamma ~ n_f <sigma v>, sigma_t ~ 4 pi alpha alpha_D eps^2/max(T,m_A')^2
ALPHA_D = 0.1
M_CHI = 1000.0


def eps_kin_chi(T, mA, alphaD=ALPHA_D):
    n_f = 0.75 * 1.2 / np.pi ** 2 * T ** 3 * 20  # ~ 20 relativistic fermionic dof, n = (3/4) zeta(3)/pi^2 g T^3
    sig = 4 * np.pi * ALPHA * alphaD / max(T, mA) ** 2  # eps^2 stripped
    return np.sqrt(hubble(T) / (n_f * sig))


log("[2c] chi-SM kinetic coupling (rough, x3): eps needed for Gamma(chi f -> chi f) = H at T=1 TeV / 40 GeV / 1 GeV,"
    " m_A'=1 GeV: %.1e / %.1e / %.1e" % (eps_kin_chi(1000, 1), eps_kin_chi(40, 1), eps_kin_chi(1, 1)))


log("[2d] Freeze-in yield scaling check (Y_FI ~ eps^2) and analytic comparison Y_FI ~ 405 sqrt(10) g M_Pl Gamma/(8 pi^4 1.66 g*s sqrt(g*) m^2)... order of magnitude")
for m in [0.3, 1, 10]:
    i = np.argmin(abs(MA - m))
    yfi = Y_freeze_in(MA[i], 1e-11)
    yan = 405 * np.sqrt(10) * 3 * M_PL * gamma_Ap(MA[i], 1e-11) / (8 * np.pi ** 4 * 1.66 * gstar_s(MA[i]) * np.sqrt(gstar(MA[i])) * MA[i] ** 2)
    log("   m=%.3g eps=1e-11: Y_FI = %.3g (analytic %.3g); Y_FI(2e-11)/Y_FI(1e-11) = %.2f" % (MA[i], yfi, yan, Y_freeze_in(MA[i], 2e-11) / yfi))


# recalled BBN envelope on E_vis Y_X [GeV] vs tau [s] (Kawasaki-Kohri-Moroi(-Takaesu)-type, EM+hadronic, GeV-scale X)
_bbn_logtau = np.array([0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12])
_bbn_logxi = np.array([-5.0, -6.5, -8, -9, -10, -11.5, -12.5, -13, -12.7, -12.3, -12.0, -11.5])


def xi_bbn_max(tau):
    lt = np.log10(np.clip(tau, 1.0, 1e12))
    return 10 ** np.interp(lt, _bbn_logtau, _bbn_logxi)


MU_FIRAS, Y_FIRAS = 9e-5, 1.5e-5   # [certain]


def late_decay_excluded(m, Y, tau):
    """Return (excluded?, reason, dict) for a relic with m Y [GeV] decaying at tau [s]; all criteria recalled."""
    if tau < 1.0:
        return False, "decays before 1 s", {}
    T_dec = T_of_t(tau)
    drho = m * Y * gstar_s(T_dec) / (1.5 * T_dec)   # Delta rho / rho_gamma at decay
    d = dict(T_dec=T_dec, drho_over_rhog=drho, xi=m * Y, xi_max=xi_bbn_max(tau))
    if tau > 1e13:
        # after recombination: Planck decaying-DM.  P026 re-expression: > 0.2 eV per baryon deposited (f_eff = 0.3) excluded
        # for tau < t_U; for tau > t_U the Slatyer-Wu form f_eff (rho_X/rho_DM) Gamma < 1e-25 s^-1 [recalled, likely, x3].
        eV_per_baryon = m * Y / 8.6e-11 * 1e9  # Y_b = n_b/s = 8.6e-11 [certain]
        frac = m * Y / 4.4e-10
        d["eV_per_baryon"] = eV_per_baryon
        d["Omega_over_OmegaDM"] = frac
        if tau > T_UNIVERSE_S:
            if frac > 1.0:
                return True, "overclosure (quasi-stable)", d
            return 0.3 * frac / tau > 1e-25, "Planck decaying DM (tau > t_U)", d
        return eV_per_baryon * 0.3 > 0.2, "Planck (post-recombination injection)", d
    exc = m * Y > xi_bbn_max(tau)
    reason = "BBN envelope"
    if 1e6 < tau < 1e9:
        mu = 1.4 * drho
        d["mu"] = mu
        if mu > MU_FIRAS:
            exc, reason = True, "FIRAS mu"
    elif tau >= 1e9:
        yy = drho / 4
        d["y"] = yy
        if yy > Y_FIRAS:
            exc, reason = True, "FIRAS y"
    return exc, reason, d


log("[2e] Late-decaying A' (freeze-in population, eps < eps_therm): scan for BBN/FIRAS exclusion")
Ap_excl = np.zeros_like(MM, dtype=bool)
Ap_rows = []
for j, m in enumerate(MA[::8]):
    for i, e in enumerate(EPS[::4]):
        jj, ii = j * 8, i * 4
        if e >= eps_therm[jj]:
            continue
        tau = tau_Ap(m, e)
        if tau < 1.0:
            continue
        Y = Y_freeze_in(m, e)
        exc, reason, d = late_decay_excluded(m, Y, tau)
        Ap_rows.append(dict(m_A_GeV=m, eps=e, tau_s=tau, Y_FI=Y, mY_GeV=m * Y, excluded=exc, reason=reason, **d))
APDF = pd.DataFrame(Ap_rows)
APDF.to_csv(os.path.join(OUT, "Aprime_late_decay_scan.csv"), index=False)
if len(APDF):
    ex = APDF[APDF.excluded]
    log("   scanned %d (m,eps) points with tau>1 s below the thermalisation line; %d excluded" % (len(APDF), len(ex)))
    if len(ex):
        for m in sorted(ex.m_A_GeV.unique())[::3]:
            sub = ex[ex.m_A_GeV == m]
            log("   m_A'=%.3g GeV: excluded eps in [%.1e, %.1e] (tau %.1e-%.1e s), reasons: %s"
                % (m, sub.eps.min(), sub.eps.max(), sub.tau_s.min(), sub.tau_s.max(), ", ".join(sorted(sub.reason.unique()))))
    log("   max mY/xi_max among tau>1 s points: %.3g" % (APDF.mY_GeV / APDF.xi_max).max())
    log("   max eps ever excluded: %.2e (LZ band needs eps >= 1e-7)" % (ex.eps.max() if len(ex) else np.nan))

# ----------------------------------------------------------------------------------------------------------
# 3. Dark Higgs
# ----------------------------------------------------------------------------------------------------------
G_D = np.sqrt(4 * np.pi * ALPHA_D)
log("[3] Dark Higgs: g_D = %.3f (alpha_D = 0.1). Mass relations: charge-q_h Higgs, m_A' = q_h g_D v_D, m_h = sqrt(2 lambda) v_D"
    % G_D)
for qh in [1, 2]:
    for lam in [0.1, 0.3, 1.0, 2.0, 4.0]:
        r = np.sqrt(2 * lam) / (qh * G_D)
        log("   q_h=%d lambda=%.1f : r = m_h/m_A' = %.3f  (v_D/m_A' = %.3f)" % (qh, lam, r, 1 / (qh * G_D)))
    log("   q_h=%d: r >= 1 needs lambda >= %.2f ; r >= 2 (prompt h->A'A') needs lambda >= %.1f" % (qh, (qh * G_D) ** 2 / 2, 2 * (qh * G_D) ** 2))


def sigv_hh_AA(m_h, m_A, alphaD=ALPHA_D, kappa=1.0):
    """s-wave <sigma v>(h h -> A'A') for m_h > m_A' (O(1) normalisation kappa, flagged)."""
    beta = np.sqrt(max(1 - m_A ** 2 / m_h ** 2, 0))
    return kappa * np.pi * alphaD ** 2 / m_h ** 2 * beta


def sigv_AA_hh(m_h, m_A, alphaD=ALPHA_D, kappa=1.0):
    beta = np.sqrt(max(1 - m_h ** 2 / m_A ** 2, 0))
    return kappa * np.pi * alphaD ** 2 / m_A ** 2 * beta


def relic_yield_h(m_h, m_A, alphaD=ALPHA_D, kappa=1.0):
    """Solve dY/dx for the dark Higgs (g=1) annihilating to A'A'. Forbidden regime uses detailed balance:
    <sigma v>_{hh->AA} = <sigma v>_{AA->hh} (g_A/g_h)^2 (m_A/m_h)^3 exp(-2 Delta x), Delta=(m_A-m_h)/m_h."""
    r = m_h / m_A
    if r >= 1:
        sv = lambda x: sigv_hh_AA(m_h, m_A, alphaD, kappa)
    else:
        Delta = (m_A - m_h) / m_h
        sv0 = sigv_AA_hh(m_h, m_A, alphaD, kappa) * 9.0 * (m_A / m_h) ** 3
        sv = lambda x: sv0 * np.exp(-2 * Delta * x)

    def Yeq(x):
        T = m_h / x
        return 45 / (4 * np.pi ** 4) / gstar_s(T) * x ** 2 * special.kve(2, x) * np.exp(-x)

    def lam(x):
        T = m_h / x
        return np.sqrt(np.pi / 45) * M_PL * m_h * gstar_s(T) / np.sqrt(gstar(T)) * sv(x) / x ** 2

    def rhs(x, W):
        Y = np.exp(W[0])
        return [-lam(x) * (Y - Yeq(x) ** 2 / Y)]

    def jac(x, W):
        Y = np.exp(W[0])
        return [[-lam(x) * (Y + Yeq(x) ** 2 / Y)]]
    x0 = 2.0
    sol = integrate.solve_ivp(rhs, [x0, 2000.0], [np.log(Yeq(x0))], method="Radau", jac=jac, rtol=1e-8, atol=1e-12)
    if not sol.success:
        log("   WARNING: Boltzmann solve failed for m_h=%g m_A=%g: %s" % (m_h, m_A, sol.message))
    return float(np.exp(sol.y[0, -1]))


log("[3a] Dark-Higgs relic yield Y_h (Boltzmann, alpha_D=0.1); m Y_DM = 4.4e-10 GeV for comparison")
hrows = []
for m_A in [1.0, 3.0, 10.0]:
    for r in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.05, 1.2, 1.5, 2.5]:
        m_h = r * m_A
        Y = relic_yield_h(m_h, m_A)
        hrows.append(dict(m_A_GeV=m_A, r=r, m_h_GeV=m_h, Y_h=Y, mY_GeV=m_h * Y, Omega_h_h2=m_h * Y * S0 / RHO_C_H2,
                          Omega_over_DM=m_h * Y / 4.4e-10))
HY = pd.DataFrame(hrows)
HY.to_csv(os.path.join(OUT, "dark_higgs_relic.csv"), index=False)
log(HY.to_string(index=False, float_format=lambda x: "%.3g" % x))
# analytic check for r>1: Omega h^2 ~ 1.07e9 x_f/(sqrt(g*) M_Pl <sigma v>)
m_A, r = 1.0, 1.5
sv = sigv_hh_AA(r * m_A, m_A)
log("   analytic check r=1.5,m_A'=1: <sv>=%.3g GeV^-2 -> Omega h^2 ~ %.2g vs ODE %.2g"
    % (sv, 1.07e9 * 20 / (np.sqrt(gstar(r * m_A / 20)) * M_PL * sv), HY[(HY.m_A_GeV == 1) & (HY.r == 1.5)].Omega_h_h2.iloc[0]))
# 3->2 cannibal estimate at x=20 for m_h=1 GeV
n20 = (1 / (2 * np.pi * 20)) ** 1.5 * np.exp(-20)
G32 = n20 ** 2 * (1.0 / (4 * np.pi)) ** 3
log("   3->2 cannibal rate at x=20, m_h=1 GeV, lambda=1: Gamma_32 ~ %.1e GeV vs H = %.1e GeV -> negligible"
    % (G32, hubble(1 / 20)))


# lifetimes
def gamma_SMlike(m):
    """Gamma of an SM-Higgs-like scalar of mass m into f fbar (tree level, running masses ignored) [certain formula]."""
    tot = 0.0
    for mf, nc, thr in [(M_E, 1, 2 * M_E), (M_MU, 1, 2 * M_MU), (M_TAU, 1, 2 * M_TAU), (M_S, 3, 2 * M_K), (M_C, 3, 3.74), (M_B, 3, 10.6)]:
        if m > thr:
            tot += nc * mf ** 2 * m / (8 * np.pi * V_EW ** 2) * (1 - (thr / m) ** 2) ** 1.5
    return tot


def gamma_h_loop(m_h, m_A, eps, alphaD=ALPHA_D):
    """h_D -> f fbar through an A' loop: Gamma = alpha_D alpha^2 eps^4 sum_f N_c Q_f^4 m_f^2 m_h beta^3/(8 pi^2 m_A'^2)
    (loop function set to 1; derived here; x3 uncertainty)."""
    tot = 0.0
    for mf, nc, q, thr in [(M_E, 1, 1, 2 * M_E), (M_MU, 1, 1, 2 * M_MU), (M_TAU, 1, 1, 2 * M_TAU),
                           (M_S, 3, 1 / 3, 2 * M_K), (M_C, 3, 2 / 3, 3.74), (M_B, 3, 1 / 3, 10.6)]:
        if m_h > thr:
            tot += nc * q ** 4 * mf ** 2 * (1 - (thr / m_h) ** 2) ** 1.5
    return alphaD * ALPHA ** 2 * eps ** 4 * tot * m_h / (8 * np.pi ** 2 * m_A ** 2)


def gamma_h_4f(m_h, m_A, eps, alphaD=ALPHA_D):
    """Tree-level h_D -> A'* A'* -> 4f (eps^4). Dimensional estimate: |M|^2 ~ 4 g_D^2 e^4 eps^4 m_h^4/m_A'^6 x (2+R)^2,
    4-body massless phase space Phi_4 = m^4/(2 (4pi)^5 3! 2!), propagator enhancement 1/(1-r^2)^2 for r<1
    (x10 uncertainty)."""
    r = min(m_h / m_A, 0.97)
    Nch = (2 + R_had(m_h / 2)) ** 2
    gD2 = 4 * np.pi * alphaD
    M2 = 4 * gD2 * E_CH ** 4 * eps ** 4 * m_h ** 4 / m_A ** 6 * Nch / (1 - r ** 2) ** 2
    Phi4 = m_h ** 4 / (2 * (4 * np.pi) ** 5 * 6 * 2)
    return M2 * Phi4 / (2 * m_h) if m_h < m_A else 0.0


def gamma_h_3body(m_h, m_A, eps, alphaD=ALPHA_D):
    """h_D -> A' A'* -> A' f fbar for m_A' < m_h < 2 m_A' (eps^2). Crude: Gamma_2(on-shell-like) x alpha eps^2 (2+R)/(3 pi) x 0.1 (1-1/r)^3.
    (x10 uncertainty; only its order of magnitude matters: it is always << 1 s in the band)."""
    r = m_h / m_A
    if r <= 1:
        return 0.0
    G2 = alphaD * m_h ** 3 / (8 * m_A ** 2)
    return G2 * ALPHA * eps ** 2 * (2 + R_had(m_h / 2)) / (3 * np.pi) * 0.1 * (1 - 1 / r) ** 3


def gamma_h_AA(m_h, m_A, alphaD=ALPHA_D, qh=2):
    if m_h <= 2 * m_A:
        return 0.0
    g_hAA = 2 * qh * G_D * m_A  # hAA vertex for m_A = q_h g_D v_D  (2 m_A^2/v_D)
    x = m_A ** 2 / m_h ** 2
    return g_hAA ** 2 * m_h ** 3 / (128 * np.pi * m_A ** 4) * (1 - 4 * x + 12 * x ** 2) * np.sqrt(1 - 4 * x)


def tau_h(m_h, m_A, eps, theta=0.0):
    G = gamma_h_loop(m_h, m_A, eps) + gamma_h_4f(m_h, m_A, eps) + gamma_h_3body(m_h, m_A, eps) + gamma_h_AA(m_h, m_A) \
        + theta ** 2 * gamma_SMlike(m_h)
    return HBAR / G if G > 0 else np.inf


log("[3b] Dark-Higgs lifetimes (no Higgs portal):")
for m_A in [1.0, 3.0, 10.0]:
    for r in [0.5, 0.9, 1.2, 2.5]:
        m_h = r * m_A
        line = "   m_A'=%4.1f r=%.2f m_h=%5.2f:" % (m_A, r, m_h)
        for eps in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
            line += " tau(eps=%.0e)=%.1e s" % (eps, tau_h(m_h, m_A, eps))
        log(line)
m_h, m_A = 0.9, 1.0
log("   channel split at m_A'=1, r=0.9, eps=1e-6: loop %.2e, 4f %.2e GeV" % (gamma_h_loop(m_h, m_A, 1e-6), gamma_h_4f(m_h, m_A, 1e-6)))

# exclusion in the (m_A', eps) plane for r = 0.5, 0.9 (forbidden regime) and 1.2
log("[3c] Dark-Higgs exclusion in the (m_A', eps) plane (yield from Boltzmann, decay via loop+4f (+3-body))")
MA_H = np.logspace(np.log10(0.05), 2, 61)
EPS_H = np.logspace(-10, -2, 97)
R_LIST = [0.5, 0.7, 0.9, 1.2]
hd_excl, hd_eps_line, hd_Y = {}, {}, {}
hrows = []
for r in R_LIST:
    grid = np.zeros((len(EPS_H), len(MA_H)), dtype=bool)
    eline = np.full(len(MA_H), np.nan)
    Ys = np.zeros(len(MA_H))
    for j, m_A in enumerate(MA_H):
        m_h = r * m_A
        Y = relic_yield_h(m_h, m_A)
        Ys[j] = Y
        for i, e in enumerate(EPS_H):
            tau = tau_h(m_h, m_A, e)
            exc, reason, d = late_decay_excluded(m_h, Y, tau)
            grid[i, j] = exc
            if j % 10 == 0 and i % 12 == 0:
                hrows.append(dict(r=r, m_A_GeV=m_A, m_h_GeV=m_h, eps=e, Y_h=Y, mY_GeV=m_h * Y, tau_h_s=tau, excluded=exc, reason=reason))
        f = lambda le: np.log10(tau_h(m_h, m_A, 10 ** le))   # eps above which tau_h < 1 s
        try:
            eline[j] = 10 ** optimize.brentq(f, -12, 0)
        except ValueError:
            eline[j] = np.nan
    hd_excl[r], hd_eps_line[r], hd_Y[r] = grid, eline, Ys
    for m in [0.3, 1, 3, 10, 30]:
        j = np.argmin(abs(MA_H - m))
        col = grid[:, j]
        exc_eps = EPS_H[col]
        rng = ("eps in [%.1e, %.1e]" % (exc_eps.min(), exc_eps.max())) if col.any() else "none"
        log("   r=%.1f m_A'=%.3g GeV: mY_h = %.2e GeV (Omega_h/Omega_DM=%.2g); tau_h<1 s needs eps >= %.2e; excluded %s"
            % (r, MA_H[j], r * MA_H[j] * Ys[j], r * MA_H[j] * Ys[j] / 4.4e-10, eline[j], rng))
    pd.DataFrame({"m_A_GeV": MA_H, "Y_h": Ys, "mY_GeV": r * MA_H * Ys, "eps_tau_h_1s": eline}).to_csv(
        os.path.join(OUT, "dark_higgs_yield_line_r%.1f.csv" % r), index=False)
    pd.DataFrame(grid.astype(int), index=EPS_H, columns=MA_H).to_csv(os.path.join(OUT, "dark_higgs_excluded_grid_r%.1f.csv" % r))
pd.DataFrame(hrows).to_csv(os.path.join(OUT, "dark_higgs_exclusion_samples.csv"), index=False)
# which reasons dominate for r=0.9
sub = pd.DataFrame(hrows)
log("   exclusion reasons by r: " + "; ".join("r=%.1f: %s" % (r, dict(sub[(sub.r == r) & sub.excluded].reason.value_counts())) for r in R_LIST))

# Higgs-portal escape: theta needed for tau_h < 1 s, and implied lambda_hphi
log("[3d] Higgs-portal mixing needed for tau_h <= 1 s (0.1 s), and implied lambda_hphi ~ theta m_H^2/(v v_D), v_D = m_A'/(2 g_D)")
prow = []
for m_A in [0.3, 1.0, 3.0, 10.0, 30.0]:
    for r in [0.5, 0.9]:
        m_h = r * m_A
        v_D = m_A / (2 * G_D)
        th1 = np.sqrt(HBAR / 1.0 / gamma_SMlike(m_h)) if gamma_SMlike(m_h) > 0 else np.inf
        th01 = np.sqrt(HBAR / 0.1 / gamma_SMlike(m_h)) if gamma_SMlike(m_h) > 0 else np.inf
        prow.append(dict(m_A_GeV=m_A, r=r, m_h_GeV=m_h, Gamma_SMlike_GeV=gamma_SMlike(m_h), theta_tau1s=th1, theta_tau0p1s=th01,
                         lambda_hphi_tau1s=th1 * M_H ** 2 / (V_EW * v_D)))
        log("   m_A'=%5.1f r=%.1f m_h=%5.2f: Gamma_SM-like=%.2e GeV; theta >= %.1e (1 s), %.1e (0.1 s); lambda_hphi >= %.1e"
            % (m_A, r, m_h, gamma_SMlike(m_h), th1, th01, th1 * M_H ** 2 / (V_EW * v_D)))
pd.DataFrame(prow).to_csv(os.path.join(OUT, "dark_higgs_portal_escape.csv"), index=False)

# ----------------------------------------------------------------------------------------------------------
# 4. Self-interactions of chi1 via A' exchange (classical Yukawa transfer cross-sections, Tulin-Yu-Zurek 2013)
# ----------------------------------------------------------------------------------------------------------


def sigmaT_classical(beta, m_A, attractive=True):
    if attractive:
        if beta < 0.1:
            return 4 * np.pi / m_A ** 2 * beta ** 2 * np.log(1 + 1 / beta)
        if beta < 1e3:
            return 8 * np.pi / m_A ** 2 * beta ** 2 / (1 + 1.5 * beta ** 1.65)
        return np.pi / m_A ** 2 * (np.log(beta) + 1 - 0.5 / np.log(beta)) ** 2
    else:
        if beta < 0.1:
            return 2 * np.pi / m_A ** 2 * beta ** 2 * np.log(1 + 1 / beta ** 2)
        if beta < 1e3:
            return 7 * np.pi / m_A ** 2 * (beta ** 1.8 + 280 * (beta / 10) ** 10.3) / (1 + 1.4 * beta + 0.006 * beta ** 4 + 160 * (beta / 10) ** 10)
        return np.pi / m_A ** 2 * (np.log(2 * beta) - np.log(np.log(2 * beta))) ** 2


log("[4] chi1 self-interactions at m_chi=1 TeV, alpha_D=0.1 (attractive/repulsive average for the near-degenerate pseudo-Dirac pair)")
srows = []
for v_kms in [30.0, 300.0, 1000.0, 4000.0]:
    v = v_kms / 299792.458
    for m_A in [0.3, 1.0, 3.0, 10.0, 30.0]:
        beta = 2 * ALPHA_D * m_A / (M_CHI * v ** 2)
        sa = sigmaT_classical(beta, m_A, True)
        sr = sigmaT_classical(beta, m_A, False)
        s_avg = 0.5 * (sa + sr)
        unit = 4 * np.pi / (M_CHI * v) ** 2 * 4  # s-wave unitarity ceiling x4 (few partial waves): upper bound in the quantum regime
        classical_ok = M_CHI * v / m_A > 1
        sig_cm2 = s_avg * GEV2_TO_CM2
        adopted = sig_cm2 if classical_ok else min(sig_cm2, unit * GEV2_TO_CM2)
        srows.append(dict(v_kms=v_kms, m_A_GeV=m_A, beta=beta, mv_over_mA=M_CHI * v / m_A, sigmaT_att_GeV2=sa, sigmaT_rep_GeV2=sr,
                          sigma_over_m_classical_cm2_g=sig_cm2 / (M_CHI * GEV_TO_G),
                          sigma_over_m_adopted_cm2_g=adopted / (M_CHI * GEV_TO_G),
                          quantum_regime_unitarity_x4_cm2_g=unit * GEV2_TO_CM2 / (M_CHI * GEV_TO_G),
                          classical_regime=classical_ok))
SIDM = pd.DataFrame(srows)
SIDM.to_csv(os.path.join(OUT, "sidm_sigma_over_m.csv"), index=False)
log(SIDM.to_string(index=False, float_format=lambda x: "%.3g" % x))
log("   Bullet cluster: sigma/m < ~1 cm^2/g (recalled, likely); dwarf-core-relevant range 0.1-1 cm^2/g at v~30 km/s")
KE_cluster = 0.5 * M_CHI * (1000 / 299792.458) ** 2
log("   chi1 chi1 -> chi2 chi2 up-scattering: KE(1000 km/s) = %.1f MeV >> 2 delta = 0.6 MeV -> open at cluster speeds; at 30 km/s KE = %.3g keV -> closed"
    % (KE_cluster * 1e3, 0.5 * M_CHI * (30 / 299792.458) ** 2 * 1e6))

# ----------------------------------------------------------------------------------------------------------
# 5. Heavy Z' (P054)
# ----------------------------------------------------------------------------------------------------------
for mZp, gq in [(500.0, 0.017), (2000.0, 0.02)]:
    G = 3 * 6 * gq ** 2 * mZp / (12 * np.pi)  # 6 quark flavours, N_c=3, vector coupling g_q
    log("[5] Z' m=%.0f GeV, g_q=%.3f: Gamma(qq)=%.2e GeV, tau=%.1e s (decays at T~TeV; only standard freeze-out, P054)" % (mZp, gq, G, HBAR / G))

# ----------------------------------------------------------------------------------------------------------
# 6. Combined map at 1 TeV, alpha_D = 0.1
# ----------------------------------------------------------------------------------------------------------
P011 = pd.read_csv("output/work/P011/epsilon_required.csv")
P011 = P011[(P011.m_GeV == 1000.0) & (P011.alpha_D_label == "alpha_D=0.1")]
S_pts = {}   # propagator/rate suppression S(m_A') per delta, from P011 (0.1 GeV value 0.034 from P011 text, delta=300)
for d in [250.0, 300.0, 350.0, 365.0, 380.0]:
    sub = P011[P011.delta_keV == d]
    if len(sub) == 0:
        continue
    ms = list(sub.m_A_GeV.values)
    Ss = list(sub.S_propagator.values)
    K = sub.eps2alphaD_over_mA4_GeV_m4.iloc[0]
    S_pts[d] = (np.array([0.1] + ms), np.array([0.034] + Ss), K)


def eps_LZ(m, delta):
    ms, Ss, K = S_pts[delta]
    lS = np.interp(np.log(m), np.log(ms), np.log(Ss), left=np.log(0.034) + 4 * (np.log(m) - np.log(0.1)) if np.isscalar(m) else None)
    if not np.isscalar(m):
        lS = np.interp(np.log(m), np.log(ms), np.log(Ss))
        lo = m < 0.1
        lS = np.where(lo, np.log(0.034) + 4 * (np.log(m) - np.log(0.1)), lS)
    S = np.minimum(np.exp(lS), 1.0)
    return np.sqrt(K * m ** 4 / (ALPHA_D * S))


log("[6] LZ band reconstruction from P011 (1 TeV, alpha_D=0.1): eps(m_A'=1 GeV, delta=300) = %.3e (P011: 8.65e-7)" % eps_LZ(np.array([1.0]), 300.0)[0])
log("   eps(10 GeV, 300) = %.3e (P011 8.28e-5); eps(0.3, 300) = %.3e (P011 1.12e-7)" % (eps_LZ(np.array([10.0]), 300.0)[0], eps_LZ(np.array([0.3]), 300.0)[0]))
EPS_FLOOR = {250.0: 7.76e-6, 300.0: 5.44e-6, 350.0: 4.46e-6, 365.0: 4.39e-6, 380.0: 4.46e-6}  # P011 chi2-decay floors (alpha_D=0.1)
BABAR = 1e-3
PLANCK_MA = 9.2  # P025 (thermal alpha_D = 0.035)


def band_crossing(delta, level):
    f = lambda lm: np.log(eps_LZ(np.array([np.exp(lm)]), delta)[0]) - np.log(level)
    try:
        return np.exp(optimize.brentq(f, np.log(0.05), np.log(100)))
    except ValueError:
        return np.nan


def band_excluded_by_hd(d, r, m_lo, m_hi):
    """Fraction and sub-ranges of the LZ band (delta=d) between m_lo and m_hi that fall in the h_D exclusion grid for ratio r."""
    ms = np.logspace(np.log10(m_lo), np.log10(m_hi), 200)
    es = eps_LZ(ms, d)
    flags = []
    for m, e in zip(ms, es):
        j = np.argmin(abs(np.log(MA_H) - np.log(m)))
        i = np.argmin(abs(np.log(EPS_H) - np.log(e)))
        flags.append(bool(hd_excl[r][i, j]))
    flags = np.array(flags)
    ranges = []
    if flags.any():
        idx = np.where(np.diff(np.concatenate([[0], flags.astype(int), [0]])) != 0)[0]
        for a, b in zip(idx[::2], idx[1::2]):
            ranges.append([float(ms[a]), float(ms[min(b, len(ms) - 1)])])
    return float(flags.mean()), ranges


summary_map = {}
for d in [250.0, 300.0, 350.0, 380.0]:
    m_floor = band_crossing(d, EPS_FLOOR[d])
    m_babar = band_crossing(d, BABAR)
    m_lo = max(m_floor, PLANCK_MA)
    hd_info = {}
    for r in R_LIST:
        frac_full, rng_full = band_excluded_by_hd(d, r, max(m_floor, 0.06), m_babar)
        frac_surv, rng_surv = band_excluded_by_hd(d, r, m_lo, m_babar) if m_lo < m_babar else (np.nan, [])
        hd_info["r=%.1f" % r] = dict(frac_of_band_above_chi2_floor=frac_full, excluded_mA_ranges_GeV=rng_full,
                                     frac_of_surviving_band=frac_surv, excluded_surviving_ranges_GeV=rng_surv)
    surv = [m_lo, m_babar] if m_lo < m_babar else None
    summary_map[d] = dict(m_floor_chi2=m_floor, m_babar=m_babar, m_planck=PLANCK_MA, surviving_mA_GeV=surv,
                          eps_surviving=[float(eps_LZ(np.array([m_lo]), d)[0]), BABAR] if surv else None, dark_higgs=hd_info)
    log("   delta=%.0f keV: chi2 floor m>=%.2f GeV; Planck (P025) m>=%.1f; BaBar m<=%.1f; surviving: %s"
        % (d, m_floor, PLANCK_MA, m_babar, ("m_A' = %.1f-%.1f GeV, eps = %.1e-%.0e" % (m_lo, m_babar, eps_LZ(np.array([m_lo]), d)[0], BABAR)) if surv else "NONE (Planck floor above BaBar ceiling)"))
    for r in R_LIST:
        hi = hd_info["r=%.1f" % r]
        log("      dark Higgs r=%.1f (no portal): excludes %.0f%% of the band above the chi2 floor %s; %.0f%% of the surviving segment %s"
            % (r, 100 * hi["frac_of_band_above_chi2_floor"], hi["excluded_mA_ranges_GeV"], 100 * (hi["frac_of_surviving_band"] if np.isfinite(hi["frac_of_surviving_band"]) else 0),
               hi["excluded_surviving_ranges_GeV"]))
# in-band A' lifetimes/thermalisation ratios at the surviving edges
for d in [300.0]:
    for m in [summary_map[d]["m_floor_chi2"], PLANCK_MA, summary_map[d]["m_babar"]]:
        e = eps_LZ(np.array([m]), d)[0]
        log("   in-band (delta=300): m_A'=%.2f eps=%.2e tau_A'=%.2e s, eps/eps_therm=%.1e, ratio to tau=1 s line %.1e"
            % (m, e, tau_Ap(m, e), e / np.interp(m, MA, eps_therm), e / np.interp(m, MA, eps_tau1)))

# ------------------------------------------------ figures ------------------------------------------------
C = dict(band="#1f77b4", floor="#d62728", planck="#9467bd", babar="#7f7f7f", therm="#2ca02c", tau="#ff7f0e", hd="#8c564b", bbn="#e377c2")

# Fig 1: g*, DeltaN_eff
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].semilogx(_Tgrid, GS_TAB, label=r"$g_*$")
ax[0].semilogx(_Tgrid, GSS_TAB, "--", label=r"$g_{*s}$")
ax[0].set_xlabel("T [GeV]"); ax[0].set_ylabel("effective d.o.f."); ax[0].set_xlim(1e-4, 1e4); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[0].set_title("Exact thermal integrals, smoothed QCD transition")
Td = np.logspace(np.log10(2e-3), 3, 200)
for gb, gf, lab in [(1, 0, "scalar (1 dof)"), (3, 0, "vector (3 dof)"), (4, 4, r"$A'+h_D+\chi_{1,2}$ (7.5)")]:
    ax[1].loglog(Td, [delta_neff(t, gb, gf) for t in Td], label=lab)
ax[1].axhline(0.3, color="k", ls=":", label="Planck (recalled) 0.3")
ax[1].set_xlabel(r"$T_{\rm dec}$ [GeV]"); ax[1].set_ylabel(r"$\Delta N_{\rm eff}$ (if still relativistic)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
ax[1].set_title(r"$\Delta N_{\rm eff}$ of a decoupled relativistic relic")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "P086_fig1_gstar_neff.png"), dpi=150); plt.close(fig)

# Fig 2: tau_A' map
fig, ax = plt.subplots(figsize=(7, 5))
cs = ax.contourf(MA, EPS, np.log10(TAU), levels=np.arange(-20, 21, 2), cmap="viridis")
cb = fig.colorbar(cs, ax=ax); cb.set_label(r"$\log_{10}(\tau_{A'}/{\rm s})$")
ax.contour(MA, EPS, np.log10(TAU), levels=[0, 2, 4, 6], colors="w", linewidths=0.8)
ax.plot(MA, eps_tau1, color=C["tau"], lw=2, label=r"$\tau_{A'}=1$ s")
ax.plot(MA, eps_therm, color=C["therm"], lw=2, label=r"thermalisation $\Gamma_{f\bar f\to A'}=H$")
if len(APDF) and APDF.excluded.any():
    ex = APDF[APDF.excluded]
    ax.scatter(ex.m_A_GeV, ex.eps, s=6, color=C["bbn"], label="freeze-in $A'$: BBN/FIRAS excluded")
for d, ls in [(300.0, "-"), (350.0, "--")]:
    ax.plot(MA, eps_LZ(MA, d), color=C["band"], ls=ls, lw=2, label=r"LZ fit $\delta=%d$ keV (P011)" % d)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.05, 100); ax.set_ylim(1e-14, 1e-2)
ax.set_xlabel(r"$m_{A'}$ [GeV]"); ax.set_ylabel(r"$\epsilon$"); ax.legend(fontsize=7, loc="lower right"); ax.set_title(r"Dark-photon lifetime, $m_\chi=1$ TeV, $\alpha_D=0.1$")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "P086_fig2_tau_map.png"), dpi=150); plt.close(fig)

# Fig 3: dark Higgs
fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
for m_A in [1.0, 3.0, 10.0]:
    sub = HY[HY.m_A_GeV == m_A]
    ax[0].semilogy(sub.r, sub.Omega_over_DM, "o-", label=r"$m_{A'}=%g$ GeV" % m_A)
ax[0].axhline(1, color="k", ls=":"); ax[0].set_xlabel(r"$r=m_{h_D}/m_{A'}$"); ax[0].set_ylabel(r"$m_h Y_h / (\rho_{\rm DM}/s)$")
ax[0].set_title(r"Dark-Higgs relic (if stable), $\alpha_D=0.1$"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
eps_line = np.logspace(-8, -2, 100)
for m_A, col in [(1.0, "C0"), (3.0, "C1"), (10.0, "C2")]:
    ax[1].loglog(eps_line, [tau_h(0.9 * m_A, m_A, e) for e in eps_line], color=col, label=r"$r=0.9,\ m_{A'}=%g$" % m_A)
    ax[1].loglog(eps_line, [tau_h(0.7 * m_A, m_A, e) for e in eps_line], color=col, ls=":", label=r"$r=0.7$")
    ax[1].loglog(eps_line, [tau_h(1.2 * m_A, m_A, e) for e in eps_line], color=col, ls="--", label=r"$r=1.2$")
ax[1].axhspan(1, 1e13, color=C["bbn"], alpha=0.10); ax[1].text(2e-8, 3, "BBN/FIRAS window (excluded if $m Y_h$ exceeds bound)", fontsize=7)
ax[1].axhspan(1e13, 1e21, color=C["planck"], alpha=0.10); ax[1].text(2e-8, 3e13, "Planck window ($10^{13}$-$10^{21}$ s)", fontsize=7)
ax[1].axhline(T_UNIVERSE_S, color="k", ls=":")
for d in [300.0]:
    ax[1].axvspan(eps_LZ(np.array([2.6]), d)[0], eps_LZ(np.array([34.0]), d)[0], color=C["band"], alpha=0.15)
ax[1].text(1.2e-5, 1e-9, "LZ band\n($\\delta$=300 keV,\n2.6-34 GeV)", fontsize=8, color=C["band"])
ax[1].set_xlabel(r"$\epsilon$"); ax[1].set_ylabel(r"$\tau_{h_D}$ [s]"); ax[1].set_ylim(1e-12, 1e22); ax[1].legend(fontsize=7, loc="upper right"); ax[1].grid(alpha=0.3)
ax[1].set_title("Dark-Higgs lifetime (no Higgs portal)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "P086_fig3_dark_higgs.png"), dpi=150); plt.close(fig)

# Fig 4: combined map
fig, ax = plt.subplots(figsize=(8, 6))
m_fill = np.logspace(np.log10(0.05), 2, 300)
ax.fill_between(m_fill, eps_LZ(m_fill, 250.0), eps_LZ(m_fill, 350.0), color=C["band"], alpha=0.25, label=r"LZ fit, $\delta=250$-$350$ keV (P011)")
ax.plot(m_fill, eps_LZ(m_fill, 300.0), color=C["band"], lw=2, label=r"$\delta=300$ keV")
ax.plot(m_fill, eps_LZ(m_fill, 380.0), color=C["band"], lw=1, ls=":", label=r"$\delta=380$ keV")
ax.axhspan(1e-14, EPS_FLOOR[300.0], color=C["floor"], alpha=0.12)
ax.axhline(EPS_FLOOR[300.0], color=C["floor"], lw=1.5, label=r"$\chi_2$-decay floor $\epsilon\geq5.4\times10^{-6}$ (P011/P026/P058)")
ax.axvspan(0.05, PLANCK_MA, color=C["planck"], alpha=0.10)
ax.axvline(PLANCK_MA, color=C["planck"], lw=1.5, label=r"Planck Sommerfeld, thermal $\alpha_D$: $m_{A'}<9.2$ GeV (P025)")
ax.axhspan(BABAR, 1e-1, color=C["babar"], alpha=0.25, label=r"BaBar $\epsilon>10^{-3}$ (recalled, likely)")
ax.plot([10, 70], [3e-4, 3e-4], color=C["babar"], ls="--", lw=1, label=r"LHCb prompt $\sim3\times10^{-4}$, 10-70 GeV (recalled, uncertain)")
ax.plot(MA, eps_therm, color=C["therm"], lw=2, label=r"$A'$ thermalises with SM (this work)")
ax.plot(MA, eps_tau1, color=C["tau"], lw=2, label=r"$\tau_{A'}=1$ s (this work)")
if len(APDF) and APDF.excluded.any():
    ex = APDF[APDF.excluded]
    ax.scatter(ex.m_A_GeV, ex.eps, s=5, color=C["bbn"], label="freeze-in $A'$ BBN/FIRAS excluded (this work)")
MH, EH = np.meshgrid(MA_H, EPS_H)
plt.rcParams["hatch.color"] = C["hd"]
plt.rcParams["hatch.linewidth"] = 0.6
ax.contourf(MH, EH, hd_excl[0.7].astype(float), levels=[0.5, 1.5], colors="none", hatches=["\\\\\\"])
ax.contour(MH, EH, hd_excl[0.7].astype(float), levels=[0.5], colors=C["hd"], linewidths=1.5, linestyles="--")
ax.contourf(MH, EH, hd_excl[0.9].astype(float), levels=[0.5, 1.5], colors=[C["hd"]], alpha=0.25)
ax.contour(MH, EH, hd_excl[0.9].astype(float), levels=[0.5], colors=C["hd"], linewidths=1.5)
ax.plot([], [], color=C["hd"], lw=1.5, ls="--", label=r"$h_D$ excluded, $m_{h_D}=0.7\,m_{A'}$, no Higgs portal (hatched; this work)")
ax.plot([], [], color=C["hd"], lw=1.5, label=r"$h_D$ excluded, $m_{h_D}=0.9\,m_{A'}$ (filled; this work)")
# surviving region delta=300
m_lo, m_hi = summary_map[300.0]["surviving_mA_GeV"]
mm = np.logspace(np.log10(m_lo), np.log10(m_hi), 50)
ax.plot(mm, eps_LZ(mm, 300.0), color="k", lw=4, alpha=0.6, label=r"surviving ($\delta=300$): %.0f-%.0f GeV" % (m_lo, m_hi))
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.05, 100); ax.set_ylim(1e-12, 1e-1)
ax.set_xlabel(r"$m_{A'}$ [GeV]"); ax.set_ylabel(r"kinetic mixing $\epsilon$")
ax.set_title(r"Dark-photon mediator for the LZ event: $m_\chi=1$ TeV, $\alpha_D=0.1$")
ax.legend(fontsize=6.5, loc="lower right", ncol=1); ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "P086_fig4_combined_map.png"), dpi=150); plt.close(fig)

# summary JSON
summary = dict(
    gstar_checks={str(T): [float(gstar(T)), float(gstar_s(T))] for T in [1e-4, 1e-2, 0.1, 1.0, 10.0, 1e3]},
    t_of_T_s={"1 MeV": float(t_of_T(1e-3)), "0.1 MeV": float(t_of_T(1e-4)), "z=1100": float(t_of_T(T_CMB0 * 1101))},
    eps_therm={"%.3g" % MA[np.argmin(abs(MA - m))]: float(eps_therm[np.argmin(abs(MA - m))]) for m in [0.1, 0.3, 1, 3, 10, 30]},
    eps_tau1s={"%.3g" % MA[np.argmin(abs(MA - m))]: float(eps_tau1[np.argmin(abs(MA - m))]) for m in [0.3, 1, 3, 10, 30]},
    tau_band_points_s={"m=%g,eps=%.1e" % (m, e): float(tau_Ap(m, e)) for (m, e) in [(0.3, 1.1e-7), (1, 8.7e-7), (3, 7.4e-6), (10, 8.3e-5), (30, 7.5e-4)]},
    Aprime_late_decay_excluded_points=int(APDF.excluded.sum()) if len(APDF) else 0,
    Aprime_late_decay_max_eps_excluded=float(APDF[APDF.excluded].eps.max()) if len(APDF) and APDF.excluded.any() else None,
    dark_higgs_relic=HY.to_dict(orient="records"),
    dark_higgs_lambda_thresholds={"q_h=2: r>=1": float((2 * G_D) ** 2 / 2), "q_h=1: r>=1": float(G_D ** 2 / 2), "q_h=2: r>=2": float(2 * (2 * G_D) ** 2)},
    dark_higgs_eps_tau1s_lines={"r=%.1f" % r: {"%.3g" % MA_H[j]: float(hd_eps_line[r][j]) for j in range(0, len(MA_H), 10)} for r in R_LIST},
    dark_higgs_mY_GeV={"r=%.1f" % r: {"%.3g" % MA_H[j]: float(r * MA_H[j] * hd_Y[r][j]) for j in range(0, len(MA_H), 10)} for r in R_LIST},
    eps_therm_B_check={"%.3g" % MA[np.argmin(abs(MA - m))]: float(eps_therm_B[np.argmin(abs(MA - m))]) for m in [0.3, 1, 10]},
    portal_escape=prow,
    sidm=SIDM.to_dict(orient="records"),
    delta_neff=NEFF.to_dict(orient="records"),
    T_dec_needed_for_dNeff_below_0p3_GeV={"scalar": float(np.exp(T_scalar_ok)), "vector": float(np.exp(T_vector_ok))},
    combined_map={str(k): v for k, v in summary_map.items()},
    runtime_s=time.time() - T0,
)
with open(os.path.join(OUT, "P086_summary.json"), "w") as f:
    json.dump(summary, f, indent=1, default=float)
log("Runtime %.1f s. Figures: %s" % (time.time() - T0, ", ".join(sorted(os.listdir(FIG)))))
LOG.close()
