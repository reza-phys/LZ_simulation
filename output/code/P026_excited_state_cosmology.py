#!/usr/bin/env python
"""
P026 - Cosmological constraints on the excited state of inelastic DM (delta ~ 200-400 keV, m ~ 0.3-4 TeV).

Parts
 A. Thermodynamics of the SM bath (g_*, T_nu/T_gamma, H, t) and the chi2 <-> chi1 transition rates on the bath
    for (i) the Higgsino (Z exchange, couplings fixed by G_F; P007) and (ii) the dark-photon pseudo-Dirac model
    with the LZ-fitted sigma_p(N=1) from P011.  Boltzmann equation for the excited fraction f2(T) -> freeze-out
    value f2_fo and decoupling temperature T_*.
 B. chi2 decays: tau(chi2 -> chi1 nu nubar) via Z (P014 formula), tau(chi2 -> chi1 gamma) via loop transition
    dipole (P014 estimate, kappa band), dark-photon tau_nunu(m_A') from the P011 width with the LZ-fit coupling.
 C. Energy injection vs BBN, FIRAS (mu, y), CMB anisotropies (energy per baryon criterion), 300 keV line (SPI),
    extragalactic continuum.  tau-f2 exclusion maps for the visible (gamma) and invisible (nu nubar) channels.
 D. Surviving chi2 today, link to the exothermic LZ bound (P011) and to a Galactic-centre 300 keV line.

Run from the simulation root:  .venv/bin/python output/code/P026_excited_state_cosmology.py
All recalled inputs are collected in RECALLED (written to work/P026/recalled_knowledge.json).
"""
import json, os, sys
import numpy as np
from scipy import integrate, optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "output/code")
from common import lzcommon as lz  # noqa: E402  (RHO0_GEV_CM3, GEV_TO_CM2, HBARC)

OUT = "output/work/P026"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


# ----------------------------------------------------------------------------------------------------------
# 0. Constants (recalled; PDG / Planck 2018).  GeV units unless stated.
# ----------------------------------------------------------------------------------------------------------
GF = 1.1663788e-5          # GeV^-2
S2W = 0.2312               # sin^2 theta_W (MSbar at m_Z)
ALPHA = 1 / 137.036
ALPHA2 = 0.0338            # g^2/4pi, g = 0.652
E_CHARGE = np.sqrt(4 * np.pi * ALPHA)
M_E, M_MU = 0.000511, 0.10566
M_P = 0.938272
M_PL = 1.220890e19         # GeV
HBAR_GEV_S = 6.582119569e-25
C_CM_S = 2.99792458e10
KPC_CM = 3.0857e21
GEV_TO_CM2 = lz.GEV_TO_CM2
# cosmology (Planck 2018, recalled certain)
H0_KM_S_MPC = 67.4
H0_S = H0_KM_S_MPC / 3.0857e19            # s^-1
h = H0_KM_S_MPC / 100
OMEGA_C_H2, OMEGA_B_H2, OMEGA_G_H2 = 0.120, 0.0224, 2.47e-5
OMEGA_M = (OMEGA_C_H2 + OMEGA_B_H2) / h**2
OMEGA_R = 1.68 * OMEGA_G_H2 / h**2         # photons + 3 massless nu (approx)
OMEGA_L = 1 - OMEGA_M - OMEGA_R
T0_K = 2.7255
K_B_GEV = 8.617333e-14
T0_GEV = T0_K * K_B_GEV
RHO_CRIT_H2 = 1.05368e-5                  # GeV cm^-3
RHO_DM0 = OMEGA_C_H2 * RHO_CRIT_H2        # GeV cm^-3
RHO_G0 = OMEGA_G_H2 * RHO_CRIT_H2
N_B0 = OMEGA_B_H2 * RHO_CRIT_H2 / M_P     # cm^-3
T_U_S = 4.35e17                           # 13.8 Gyr
T_REC_S = 1.2e13                          # z ~ 1100 (t = 380 kyr)

RECALLED = [
    dict(item="G_F, sin^2 theta_W = 0.2312, alpha, alpha_2 = 0.0338, m_e, m_mu, m_p, M_Pl", presumed_source="PDG", reliability="certain"),
    dict(item="Planck 2018: H0 = 67.4, Omega_c h^2 = 0.120, Omega_b h^2 = 0.0224, T0 = 2.7255 K; Omega_gamma h^2 = 2.47e-5; t_U = 13.8 Gyr; t_rec = 3.8e5 yr", presumed_source="Planck 2018 / PDG", reliability="certain"),
    dict(item="Z couplings g_V = T3 - 2Q s_W^2, g_A = T3; Higgsino neutral Dirac state couples like nu with g_V = 1 (vector-like, T3 = 1/2 both chiralities) -> L = -sqrt2 G_F [psi-bar gamma psi][f-bar gamma (g_V - g_A gamma5) f]; consistent with P007 c_n = -G_F/sqrt2 and sigma_n = G_F^2 mu^2/2pi", presumed_source="SM; P007", reliability="certain (reproduces P007 sigma_n)"),
    dict(item="Heavy-target scattering cross-section sigma(chi2 f -> chi1 f) = (c_V^2 + c_A^2)(E+delta)^2/(2 pi) for massless f, derived here (Tr[p' g0 p g0] = 4E E'(1+cos theta))", presumed_source="derived; checks: CEvNS total sigma = G_F^2 Q_W^2 E^2/4pi reproduced", reliability="certain"),
    dict(item="Detailed balance Gamma_up = Gamma_down exp(-delta/T) for a heavy two-level system in a thermal bath", presumed_source="statistical mechanics", reliability="certain"),
    dict(item="Neutrino decoupling at T ~ 1-2 MeV; T_nu/T_gamma -> (4/11)^(1/3) after e+e- annihilation", presumed_source="standard cosmology", reliability="certain"),
    dict(item="Gamma(chi2 -> chi1 nu nubar) = G_F^2 delta^5/(20 pi^3) (Higgsino, 3 flavours); dark photon: Gamma = 3 G_eff^2 delta^5/(120 pi^3), G_eff = g_D eps tan(theta_W) g/(4 cos(theta_W) m_Z^2)", presumed_source="corpus P014 / P011 (derived there)", reliability="corpus"),
    dict(item="Gamma(chi2 -> chi1 gamma) = mu_12^2 delta^3/pi with mu_12 ~ kappa (alpha_2/2pi) e/(2 mu), kappa = 0.1-3", presumed_source="P014 estimate; heavy-neutrino transition-moment width", reliability="uncertain (order of magnitude)"),
    dict(item="Dark-photon chi2 -> chi1 gamma vanishes for an on-shell photon (kinetic mixing couples the photon to the dark current only through eps q^2/(q^2 - m_A'^2))", presumed_source="corpus P011; Holdom kinetic mixing", reliability="likely"),
    dict(item="Photodissociation thresholds: D 2.224 MeV, 7Be 1.587, 7Li 2.467, 3He 5.49, 4He 19.81 MeV", presumed_source="nuclear binding energies", reliability="certain"),
    dict(item="COBE/FIRAS: |mu| < 9e-5, |y| < 1.5e-5 (95%); mu ~ 1.4 Delta rho/rho for z = 5e4-2e6, y = Delta rho/(4 rho) for z < 5e4; thermalisation complete for z > 2e6; PIXIE-class sensitivity ~1e-8", presumed_source="Fixsen et al. 1996; Chluba & Sunyaev 2012; Kogut et al. 2011", reliability="certain (FIRAS), likely (windows), uncertain (PIXIE)"),
    dict(item="Planck bound on decaying DM (tau >> t_U): f_eff Gamma <~ 1e-25 s^-1 for decays to e+e-/gamma, i.e. ~0.2 eV of deposited energy per baryon per Hubble time at z ~ 600; used here as 'deposited EM energy per baryon between t_rec and t_U <~ 0.2 eV' (factor ~3)", presumed_source="Slatyer & Wu PRD 95, 023010 (2017); Poulin, Serpico, Lesgourgues 2016", reliability="uncertain (factor 3; exact shape of the bound vs tau not reproduced)"),
    dict(item="f_eff for 300 keV photons ~0.5 x (1 - exp(-tau_Compton per Hubble time)); Klein-Nishina cross-section formula", presumed_source="Slatyer 2016 deposition efficiencies (order of magnitude); QED", reliability="uncertain (f_eff), certain (KN)"),
    dict(item="INTEGRAL/SPI narrow-line 3 sigma sensitivity at 300 keV ~ 3e-5 ph cm^-2 s^-1 (deep GC exposure ~1e-5)", presumed_source="SPI instrument papers (Roques et al. 2003; Vedrenne et al. 2003) and 511 keV/60Fe line studies", reliability="uncertain (factor 3)"),
    dict(item="Cosmic gamma-ray background at 300 keV: E^2 dN/dE ~ 2 keV cm^-2 s^-1 sr^-1", presumed_source="HEAO-1 / SMM / COMPTEL compilations", reliability="uncertain (factor 2)"),
    dict(item="Galactic NFW: r_s = 20 kpc, R_sun = 8.2 kpc, rho_sun = 0.3 GeV/cm^3 (lzcommon RHO0)", presumed_source="standard; Baxter et al. 2021 for rho_sun", reliability="likely"),
    dict(item="Ionisation energy of H 13.6 eV; Planck tau_reio = 0.054 +- 0.007", presumed_source="atomic physics; Planck 2018", reliability="certain"),
]

# ----------------------------------------------------------------------------------------------------------
# A. Thermodynamics of the bath
# ----------------------------------------------------------------------------------------------------------
def _fd_int(kind, m, T):
    """Fermi-Dirac integrals per degree of freedom: number, energy, pressure densities (Boltzmann-exact FD)."""
    if m / T > 60:
        return 0.0
    x = m / T

    def f(u):  # u = p/T
        E = np.sqrt(u * u + x * x)
        occ = 1.0 / (np.exp(E) + 1.0)
        if kind == "n":
            return u * u * occ
        if kind == "rho":
            return u * u * E * occ
        return u**4 / (3 * E) * occ  # pressure

    val, _ = integrate.quad(f, 0, 50 + 5 * x, limit=200)
    return val / (2 * np.pi**2) * T**(4 if kind != "n" else 3)


def rho_e(T):  # e+ e- (4 dof)
    return 4 * _fd_int("rho", M_E, T)


def s_e(T):
    return 4 * (_fd_int("rho", M_E, T) + _fd_int("p", M_E, T)) / T


def n_species(T, m, g):
    return g * _fd_int("n", m, T)


S_HIGH = (2 * np.pi**2 / 45) * (2 + 7 / 8 * 4)   # (s_gamma + s_e)/T^3 at T >> m_e


def T_nu_over_T(T):
    """Neutrino-to-photon temperature after nu decoupling from (s_gamma + s_e) a^3 = const (instantaneous
    decoupling at T = 2 MeV; above that T_nu = T)."""
    if T >= 2e-3:
        return 1.0
    s_over_T3 = (2 * np.pi**2 / 45) * 2 + s_e(T) / T**3
    return (s_over_T3 / S_HIGH) ** (1 / 3)


def g_star(T):
    """Effective relativistic dof in rho at T (photons, e+-, 3 nu, mu, and quarks/gluons above 150 MeV
    (crude step), hadrons ignored)."""
    r = T_nu_over_T(T)
    g = 2 + (7 / 8) * 6 * r**4 + rho_e(T) / ((np.pi**2 / 30) * T**4)
    g += 4 * _fd_int("rho", M_MU, T) / ((np.pi**2 / 30) * T**4)
    if T > 0.15:
        g += 16 + (7 / 8) * 12 * 3 * (1 if T > 0.15 else 0)  # gluons + u,d,s
    return g


def H_early(T):
    return 1.66 * np.sqrt(g_star(T)) * T**2 / M_PL   # GeV


# late-time Friedmann: t(z), z(t)
def H_z(z):
    return H0_S * np.sqrt(OMEGA_R * (1 + z)**4 + OMEGA_M * (1 + z)**3 + OMEGA_L)


Z_RAD = 1e6   # above this the matter term is < 0.4% of radiation: use t = 1/(2H) (radiation era)
_zgrid = np.logspace(-3, np.log10(Z_RAD), 900)
_tgrid = np.array([integrate.quad(lambda u: 1 / H_z(np.exp(u) - 1), np.log1p(z), np.log1p(Z_RAD), limit=400)[0]
                   + 1 / (2 * H_z(Z_RAD)) for z in _zgrid])   # u = ln(1+z)
assert np.all(np.diff(_tgrid) < 0) and np.all(np.isfinite(_tgrid)), "t(z) grid not monotonic"


def t_of_z(z):
    if z > Z_RAD:
        return 1 / (2 * H_z(z))
    return np.interp(np.log10(z), np.log10(_zgrid), _tgrid) if z >= 1e-3 else T_U_S


def z_of_t(t):
    if t < _tgrid[-1]:   # radiation era: H = H0 sqrt(Omega_r) (1+z)^2 = 1/(2t)  (g_* changes before e+e-
        return np.sqrt(1 / (2 * t * H0_S * np.sqrt(OMEGA_R))) - 1   # annihilation ignored here: ~10% in t)
    return 10 ** np.interp(-np.log10(t), -np.log10(_tgrid), np.log10(_zgrid))


def T_of_t_keV(t):
    """Photon temperature at cosmic time t (keV). For t < 1e4 s invert H_early(T) = 1/(2t) (includes g_*(T));
    later use T0 (1+z)."""
    if t < 1e4:
        f = lambda lnT: np.log(H_early(np.exp(lnT)) / HBAR_GEV_S) - np.log(1 / (2 * t))
        return np.exp(optimize.brentq(f, np.log(1e-7), np.log(1.0))) * 1e6
    return T0_GEV * (1 + z_of_t(t)) * 1e6


# ----------------------------------------------------------------------------------------------------------
# A. Transition rates on the bath
#    L = [chi1-bar gamma^mu chi2][f-bar gamma_mu (c_V - c_A gamma5) f]  ->  sigma_down(E) = (c_V^2+c_A^2)(E+delta)^2/(2 pi)
#    Gamma_down = sum_f g_f/(2 pi^2) int dE E^2 e^{-E/T} sigma(E)  (Boltzmann statistics for the bath; FD would
#    change the E^4 moment by 3%).  Detailed balance: Gamma_up = Gamma_down exp(-delta/T).
# ----------------------------------------------------------------------------------------------------------
def gVA(T3, Q):
    return T3 - 2 * Q * S2W, T3


SPECIES_H = []   # Higgsino: (name, g, mass, c2 = c_V^2 + c_A^2), c = sqrt2 G_F g_{V,A}
for name, T3, Q, g, m in [("nu", 0.5, 0, 4, 0.0), ("e", -0.5, -1, 4, M_E), ("mu", -0.5, -1, 4, M_MU),
                          ("u", 0.5, 2 / 3, 12, 0.0025), ("d", -0.5, -1 / 3, 12, 0.005), ("s", -0.5, -1 / 3, 12, 0.095)]:
    gv, ga = gVA(T3, Q)
    if name == "nu":
        gv, ga, g = 0.5, 0.5, 4 * 3   # three flavours; 4 states each in the Dirac bookkeeping (see details)
    SPECIES_H.append((name, g, m, 2 * GF**2 * (gv**2 + ga**2)))


def species_DP(c2):
    """Dark photon: coupling c_D Q_f to charged fermions only."""
    return [("e", 4, M_E, c2), ("mu", 4, M_MU, c2), ("u", 12, 0.0025, c2 * 4 / 9), ("d", 12, 0.005, c2 / 9),
            ("s", 12, 0.095, c2 / 9)]


def K_moment(T, delta, m, sign=+1):
    """int_m^inf dE p E (E + sign*delta)^2 e^{-E/T} (massless: 24T^5 + 12 delta T^4 + 2 delta^2 T^3)."""
    if m == 0.0:
        return 24 * T**5 + sign * 12 * delta * T**4 + 2 * delta**2 * T**3
    if m / T > 60:
        return 0.0
    f = lambda E: np.sqrt(E * E - m * m) * E * (E + sign * delta)**2 * np.exp(-E / T)
    return integrate.quad(f, m, m + 60 * T, limit=200)[0]


def Gamma_down(T, delta, species, hadron_cut=0.15):
    G = 0.0
    r = T_nu_over_T(T)
    for name, g, m, c2 in species:
        if name in ("u", "d", "s") and T < hadron_cut:
            continue   # confined: no free quarks; hadron channels neglected
        Tf = T * r if name == "nu" else T
        G += g / (2 * np.pi**2) * c2 / (2 * np.pi) * K_moment(Tf, delta, m)
    return G   # GeV


def solve_f2(delta, species, T_start=0.2, T_end=2e-7, tau_s=None, label=""):
    """Integrate df2/dt = -(Gd + Gu) f2 + Gu - f2/tau from T_start (f2 = f2_eq) to T_end. Returns dict."""
    x0, x1 = np.log(T_start), np.log(T_end)

    def rhs(x, y):
        T = np.exp(x)
        Gd = Gamma_down(T, delta, species)
        Gu = Gd * np.exp(-delta / T)
        Hc = H_early(T)
        dec = 0.0 if tau_s is None else HBAR_GEV_S / tau_s
        dfdt = -(Gd + Gu + dec) * y[0] + Gu
        return [-dfdt / Hc]   # dt = -dx/H

    f_eq0 = np.exp(-delta / T_start) / (1 + np.exp(-delta / T_start))
    sol = integrate.solve_ivp(rhs, (x0, x1), [f_eq0], method="LSODA", rtol=1e-7, atol=1e-12, dense_output=True)
    Ts = np.exp(np.linspace(x0, x1, 400))
    f2 = sol.sol(np.log(Ts))[0]
    # decoupling temperature: Gamma_down (1 + e^{-delta/T}) = H
    g = lambda lnT: np.log(Gamma_down(np.exp(lnT), delta, species) * (1 + np.exp(-delta / np.exp(lnT)))) - np.log(H_early(np.exp(lnT)))
    try:
        T_star = np.exp(optimize.brentq(g, np.log(1e-5), np.log(T_start)))
    except ValueError:
        T_star = np.nan
    res = dict(label=label, delta_keV=delta * 1e6, T_star_MeV=T_star * 1e3, f2_eq_Tstar=np.exp(-delta / T_star) / (1 + np.exp(-delta / T_star)),
               f2_freeze=float(f2[-1]), T_grid_MeV=(Ts * 1e3).tolist(), f2_grid=f2.tolist())
    return res


# ----------------------------------------------------------------------------------------------------------
# B. Decays
# ----------------------------------------------------------------------------------------------------------
def tau_nunu_higgsino_s(delta):
    return HBAR_GEV_S / (GF**2 * delta**5 / (20 * np.pi**3))


def tau_gamma_higgsino_s(delta, mu_GeV, kappa=1.0):
    mu12 = kappa * (ALPHA2 / (2 * np.pi)) * E_CHARGE / (2 * mu_GeV)
    return HBAR_GEV_S / (mu12**2 * delta**3 / np.pi)


MZ, CW = 91.1876, np.sqrt(1 - S2W)
G_W = np.sqrt(4 * np.pi * ALPHA2)


def tau_nunu_darkphoton_s(delta, eps, alpha_D):
    g_D = np.sqrt(4 * np.pi * alpha_D)
    Geff = g_D * eps * (np.sqrt(S2W) / CW) * G_W / (4 * CW * MZ**2)
    return HBAR_GEV_S / (3 * Geff**2 * delta**5 / (120 * np.pi**3))


# ----------------------------------------------------------------------------------------------------------
# C. Energy injection and constraints
# ----------------------------------------------------------------------------------------------------------
def sigma_KN_cm2(E_GeV):
    x = E_GeV / M_E
    sT = 6.6524e-25
    term = ((1 + x) / x**3) * (2 * x * (1 + x) / (1 + 2 * x) - np.log(1 + 2 * x)) + np.log(1 + 2 * x) / (2 * x) - (1 + 3 * x) / (1 + 2 * x)**2
    return 0.75 * sT * term


def f_eff_photon(z, E_GeV):
    tauC = N_B0 * (1 + z)**3 * sigma_KN_cm2(E_GeV) * C_CM_S / H_z(z)   # all electrons (bound + free) per Hubble time
    return 0.5 * (1 - np.exp(-tauC))


def energy_per_baryon_eV(f2, delta, m):
    n_ratio = (OMEGA_C_H2 / OMEGA_B_H2) * M_P / m   # n_DM / n_b
    return f2 * delta * 1e9 * n_ratio


def deposited_eV_per_baryon(f2, delta, m, tau_s, n_t=60):
    """EM energy deposited per baryon between t_rec and today for decays with lifetime tau, weighting each
    epoch by f_eff(z)."""
    ts = np.logspace(np.log10(T_REC_S), np.log10(T_U_S), n_t)
    w = np.exp(-ts / tau_s) / tau_s
    feff = np.array([f_eff_photon(z_of_t(t), delta) for t in ts])
    frac = np.trapezoid(w * feff, ts)
    return energy_per_baryon_eV(f2, delta, m) * frac, frac


def drho_over_rho_gamma(f2, delta, m, tau_s):
    z = z_of_t(tau_s)
    return f2 * (delta / m) * (RHO_DM0 / RHO_G0) / (1 + z), z


# Galactic decay D-factor (NFW), cone of half-angle theta
def D_factor(theta_deg, r_s_kpc=20.0, R0_kpc=8.2, rho0=lz.RHO0_GEV_CM3, gamma=1.0):
    rho_s = rho0 * (R0_kpc / r_s_kpc) ** gamma * (1 + R0_kpc / r_s_kpc) ** (3 - gamma)

    def rho(r):
        return rho_s / ((r / r_s_kpc) ** gamma * (1 + r / r_s_kpc) ** (3 - gamma))

    def los(psi):
        f = lambda l: rho(np.sqrt(l * l + R0_kpc**2 - 2 * l * R0_kpc * np.cos(psi)))
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", integrate.IntegrationWarning)
            return integrate.quad(f, 0, 100, limit=400, points=[R0_kpc * np.cos(psi)])[0]   # GeV/cm^3 * kpc

    val = integrate.quad(lambda p: los(p) * 2 * np.pi * np.sin(p), 0, np.radians(theta_deg), limit=100)[0]
    return val * KPC_CM   # GeV cm^-2 sr


def line_flux(f2, tau_s, m, D):
    return f2 * D / (4 * np.pi * tau_s * m)   # ph cm^-2 s^-1


def egb_intensity(f2, tau_s, m):
    """Isotropic extragalactic decay intensity today at E = delta (ph cm^-2 s^-1 sr^-1), sum over z of
    (c/4pi) n_DM0 Gamma int dz/(H(1+z)) (redshifted-line integral; photons at E0=delta/(1+z) -> E^2 dN/dE at E0
    ~ delta requires the z~0 shell, so we report the intensity per ln E ~ (c/4pi) n Gamma/H0 x O(1))."""
    integrand = lambda z: 1 / (H_z(z) * (1 + z))
    I = integrate.quad(integrand, 0, 1, limit=200)[0]  # photons observed within a factor 2 of delta
    return (C_CM_S / (4 * np.pi)) * (RHO_DM0 / m) * (f2 / tau_s) * I


# ----------------------------------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    results = {}
    log("# P026 run log")
    log(f"rho_DM0 = {RHO_DM0:.3e} GeV/cm3, rho_gamma0 = {RHO_G0:.3e}, n_b0 = {N_B0:.3e} cm^-3, rho_DM/rho_gamma today = {RHO_DM0/RHO_G0:.0f}")
    log(f"t(z=1100) = {t_of_z(1100):.3e} s, t(z=2e6) = {t_of_z(2e6):.3e} s, t(z=5e4) = {t_of_z(5e4):.3e} s, t(z=0.001) = {t_of_z(1e-3):.3e} s")
    log(f"g_*(10 MeV) = {g_star(0.01):.2f}, g_*(1 MeV) = {g_star(1e-3):.2f}, g_*(0.05 MeV) = {g_star(5e-5):.3f} (3.36 expected), T_nu/T(0.01 MeV) = {T_nu_over_T(1e-5):.4f} (0.7138)")

    # ---- A. cross-section checks
    d = 300e-6
    c2_n = 2 * GF**2 * (0.5**2)   # neutron: g_V = -1/2, g_A = 0 (spin-summed nucleus)  -> sigma_n = c^2 mu^2/pi
    mu_n = M_P * 1000 / (M_P + 1000)
    sig_n = c2_n * mu_n**2 / np.pi * GEV_TO_CM2
    log(f"check: Higgsino sigma_n (NR, c_n^2 = G_F^2/2) = {sig_n:.3e} cm2 (P007: 7.40e-39)")
    E = 1e-3
    sig_nu = 2 * GF**2 * 0.5 * (E + d)**2 / (2 * np.pi) * GEV_TO_CM2  # per nu state (Dirac bookkeeping x2 states)
    log(f"check: sigma(chi2 nu -> chi1 nu) per nu state at E = 1 MeV = {sig_nu:.3e} cm2 (= G_F^2 (E+delta)^2/pi)")
    log(f"check: total CEvNS-like sigma G_F^2 E^2/(4pi) at 1 MeV = {GF**2*E**2/(4*np.pi)*GEV_TO_CM2:.3e} cm2")

    # ---- A. Higgsino f2 freeze-out
    deltas = [200e-6, 250e-6, 300e-6, 350e-6, 366e-6, 400e-6]
    higg = []
    for dl in deltas:
        r = solve_f2(dl, SPECIES_H, label="Higgsino")
        higg.append(r)
        log(f"Higgsino delta = {dl*1e6:.0f} keV: T_* = {r['T_star_MeV']:.2f} MeV, f2_eq(T_*) = {r['f2_eq_Tstar']:.3f}, f2_freeze = {r['f2_freeze']:.3f}")
    # sensitivity: without e,mu (nu only) and with FD moment factor 0.972
    r_nu = solve_f2(300e-6, [s for s in SPECIES_H if s[0] == "nu"], label="Higgsino nu-only")
    log(f"Higgsino delta=300 nu-only bath: T_* = {r_nu['T_star_MeV']:.2f} MeV, f2_freeze = {r_nu['f2_freeze']:.3f}")
    sp_half = [(n, g, m, c2 * 0.5) for n, g, m, c2 in SPECIES_H]
    r_half = solve_f2(300e-6, sp_half, label="Higgsino x0.5 rates")
    sp_dbl = [(n, g, m, c2 * 2.0) for n, g, m, c2 in SPECIES_H]
    r_dbl = solve_f2(300e-6, sp_dbl, label="Higgsino x2 rates")
    log(f"Higgsino delta=300 rate x0.5: T_* = {r_half['T_star_MeV']:.2f} MeV, f2 = {r_half['f2_freeze']:.3f}; x2: T_* = {r_dbl['T_star_MeV']:.2f}, f2 = {r_dbl['f2_freeze']:.3f}")
    # chi2 chi2 -> chi1 chi1 check at T_*
    Tst = higg[2]["T_star_MeV"] * 1e-3
    m = 1000.0
    sig_chichi = c2_n * 4 * (m / 2)**2 / np.pi  # rough: vector-vector contact between heavy fermions, mu = m/2, Q_W-like factor ~ (2 x 1/2)^2
    n_chi = (RHO_DM0 / m) * (Tst / T0_GEV)**3 * (3.91 / g_star(Tst)) ** 1  # crude entropy dilution ratio
    v_rel = np.sqrt(3 * Tst / m) * 2
    G_chichi = n_chi * sig_chichi * GEV_TO_CM2 * v_rel * C_CM_S  # s^-1
    H_star_s = H_early(Tst) / HBAR_GEV_S
    log(f"chi2 chi2 -> chi1 chi1 at T_*: n_chi = {n_chi:.2e} cm^-3, sigma = {sig_chichi*GEV_TO_CM2:.2e} cm2, Gamma/H = {G_chichi/H_star_s:.1e}")

    # ---- A. Dark photon with LZ-fit couplings (P011 sigma_p(N=1), 1 TeV, annual halo)
    sigma_p_N1 = {250: 1.1987e-42, 300: 8.5942e-42, 350: 2.9856e-40, 365: 2.2865e-39, 380: 5.5953e-38}  # P011 chi2_lifetime.csv
    N_exo_f2_1 = {250: 250.7, 300: 858.2, 350: 18134.0, 365: 122442.0, 380: 2648780.0}                  # P011
    mu_p = M_P * 1000 / (M_P + 1000)
    dp = []
    for dk, sp in sigma_p_N1.items():
        c2 = sp / GEV_TO_CM2 * np.pi / mu_p**2   # c_D^2 = sigma_p pi/mu_p^2 (GeV^-4)
        r = solve_f2(dk * 1e-6, species_DP(c2), T_start=1.0, label="dark photon")
        r["sigma_p_N1_cm2"] = sp
        r["cD2_over_GF2"] = c2 / GF**2
        dp.append(r)
        log(f"Dark photon delta = {dk} keV: c_D^2/G_F^2 = {c2/GF**2:.2e}, T_* = {r['T_star_MeV']:.1f} MeV, f2_freeze = {r['f2_freeze']:.4f}")
    results["freeze_out"] = dict(higgsino=[{k: v for k, v in r.items() if not k.endswith("grid")} for r in higg],
                                 higgsino_variants=[{k: v for k, v in r.items() if not k.endswith("grid")} for r in (r_nu, r_half, r_dbl)],
                                 dark_photon=[{k: v for k, v in r.items() if not k.endswith("grid")} for r in dp],
                                 chichi_Gamma_over_H_at_Tstar=G_chichi / H_star_s)

    # ---- B. Decays
    dec = []
    for mu_, dk in [(300, 310), (500, 342), (1000, 366), (2000, 375), (4000, 376), (1000, 300), (1000, 250), (1000, 200)]:
        dl = dk * 1e-6
        row = dict(mu_GeV=mu_, delta_keV=dk, tau_nunu_s=tau_nunu_higgsino_s(dl),
                   tau_gamma_s=tau_gamma_higgsino_s(dl, mu_), tau_gamma_lo_s=tau_gamma_higgsino_s(dl, mu_, 3.0),
                   tau_gamma_hi_s=tau_gamma_higgsino_s(dl, mu_, 0.1))
        row["T_at_tau_nunu_keV"] = T_of_t_keV(row["tau_nunu_s"])
        row["T_at_tau_gamma_MeV"] = T_of_t_keV(row["tau_gamma_s"]) * 1e-3
        row["BR_gamma_if_dipole"] = row["tau_nunu_s"] / (row["tau_nunu_s"] + row["tau_gamma_s"])
        dec.append(row)
        log(f"Higgsino mu={mu_} delta={dk}: tau_nunu = {row['tau_nunu_s']:.2e} s (T = {row['T_at_tau_nunu_keV']:.2f} keV), tau_gamma = {row['tau_gamma_s']:.2e} s [{row['tau_gamma_lo_s']:.1e}, {row['tau_gamma_hi_s']:.1e}] (T = {row['T_at_tau_gamma_MeV']:.1f} MeV)")
    results["higgsino_decays"] = dec
    # dark photon tau(m_A') for the LZ fit: eps^2 alpha_D = sigma_p m_A'^4/(16 pi alpha mu_p^2)
    dpt = []
    for dk, sp in sigma_p_N1.items():
        for mA in [0.3, 0.5, 1.0, 2.0, 2.6, 5.0, 10.0, 30.0]:
            eps2aD = (sp / GEV_TO_CM2) * mA**4 / (16 * np.pi * ALPHA * mu_p**2)
            aD = 0.0245  # thermal alpha_D at 1 TeV (P011); tau depends only on eps^2 alpha_D
            tau = tau_nunu_darkphoton_s(dk * 1e-6, np.sqrt(eps2aD / aD), aD)
            f2_today = 0.5 * np.exp(-T_U_S / tau)
            dpt.append(dict(delta_keV=dk, mA_GeV=mA, eps2_alphaD=eps2aD, eps_at_thermal_alphaD=np.sqrt(eps2aD / aD), tau_nunu_s=tau,
                            f2_today=f2_today, N_exo_LZ=N_exo_f2_1[dk] * f2_today))
    results["dark_photon_tau"] = dpt
    for r in dpt:
        if r["delta_keV"] in (300, 365):
            log(f"DP delta={r['delta_keV']} mA'={r['mA_GeV']} GeV: eps^2 alpha_D = {r['eps2_alphaD']:.2e}, tau = {r['tau_nunu_s']:.2e} s, f2_today = {r['f2_today']:.2e}, N_exo = {r['N_exo_LZ']:.2e}")
    # mA' threshold for N_exo <= 1 and for tau < t_U
    for dk, sp in sigma_p_N1.items():
        f2max = 1.0 / N_exo_f2_1[dk]
        tau_max = T_U_S / np.log(0.5 / f2max)
        K = (sp / GEV_TO_CM2) / (16 * np.pi * ALPHA * mu_p**2)   # eps^2 alpha_D = K mA^4
        aD = 0.0245
        tau_ref = tau_nunu_darkphoton_s(dk * 1e-6, np.sqrt(K / aD), aD)  # at mA = 1 GeV
        mA_min_exo = (tau_ref / tau_max) ** 0.25
        mA_tU = (tau_ref / T_U_S) ** 0.25
        log(f"DP delta={dk}: f2_max(LZ exo) = {f2max:.2e}, tau_max = {tau_max:.2e} s, m_A' >= {mA_min_exo:.2f} GeV (P011: 2.58 at 300); tau = t_U at m_A' = {mA_tU:.2f} GeV; tau(1 GeV) = {tau_ref:.2e} s")
        results.setdefault("dark_photon_thresholds", []).append(dict(delta_keV=dk, f2_max=f2max, tau_max_s=tau_max, mA_min_GeV=mA_min_exo, mA_tau_eq_tU_GeV=mA_tU, tau_at_1GeV_s=tau_ref))

    # ---- C. Energy injection
    m = 1000.0
    f2H = higg[2]["f2_freeze"]
    E_b = energy_per_baryon_eV(f2H, 300e-6, m)
    log(f"Energy budget (m = 1 TeV, delta = 300 keV, f2 = {f2H:.3f}): f2 delta/m = {f2H*300e-6/m:.2e} of rho_DM; {E_b:.0f} eV per baryon (n_DM/n_b = {(OMEGA_C_H2/OMEGA_B_H2)*M_P/m:.2e})")
    inj = []
    for tau in [0.06, 4.6e5, 1.2e6, 1e8, 1e10, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e20, 1e22]:
        dr, z = drho_over_rho_gamma(f2H, 300e-6, m, tau)
        row = dict(tau_s=tau, z_inj=z, T_inj_keV=T_of_t_keV(tau), drho_over_rho_gamma=dr)
        if z > 2e6:
            row["distortion"] = "thermalised (z > 2e6)"
            row["mu_or_y"] = 0.0
        elif z > 5e4:
            row["distortion"] = "mu"
            row["mu_or_y"] = 1.4 * dr
        elif z > 1100:
            row["distortion"] = "y"
            row["mu_or_y"] = dr / 4
        else:
            row["distortion"] = "post-recombination (anisotropies)"
            row["mu_or_y"] = dr / 4
        dep, frac = deposited_eV_per_baryon(f2H, 300e-6, m, tau)
        row["deposited_eV_per_baryon"] = dep
        row["frac_decays_in_window_x_feff"] = frac
        row["f_eff_at_z_inj"] = f_eff_photon(z, 300e-6) if z < 1e4 else None
        inj.append(row)
        log(f"tau = {tau:.1e} s: z = {z:.3e}, T = {row['T_inj_keV']:.3e} keV, Drho/rho_g = {dr:.2e} ({row['distortion']}, mu/y = {row['mu_or_y']:.1e}); deposited {dep:.2e} eV/baryon (gamma channel; window frac x f_eff = {frac:.2e})")
    results["injection_higgsino_f2"] = inj
    log(f"sigma_KN(300 keV) = {sigma_KN_cm2(300e-6):.3e} cm2; tau_Compton/Hubble at z = 1000/300/100/30: " +
        ", ".join(f"{N_B0*(1+z)**3*sigma_KN_cm2(300e-6)*C_CM_S/H_z(z):.2f}" for z in [1000, 300, 100, 30]))

    # CMB-anisotropy exclusion in (tau, f2) for the gamma channel: deposited > 0.2 eV/baryon
    E_CRIT = 0.2
    taus = np.logspace(10, 26, 161)
    f2_grid = np.logspace(-7, 0, 141)
    dep_unit = np.array([deposited_eV_per_baryon(1.0, 300e-6, m, t)[0] for t in taus])  # per unit f2
    with np.errstate(divide="ignore"):
        f2_cmb = E_CRIT / dep_unit   # excluded above this f2 (inf where nothing is deposited)
    # window for f2 = f2H
    excl = taus[f2_cmb < f2H]
    if excl.size:
        log(f"CMB anisotropies (gamma channel, f2 = {f2H:.3f}, m = 1 TeV, delta = 300 keV, E_crit = {E_CRIT} eV/baryon): excluded tau in [{excl.min():.1e}, {excl.max():.1e}] s")
        results["cmb_window_f2H"] = [float(excl.min()), float(excl.max())]
    f2_cmb_min = f2_cmb.min()
    log(f"CMB anisotropies: minimum allowed f2 over all tau (gamma channel) = {f2_cmb_min:.2e} at tau = {taus[f2_cmb.argmin()]:.1e} s; E_crit x3 -> {3*f2_cmb_min:.1e}")
    for f2v in [0.5, 0.1, 1e-2, 1e-3]:
        ex = taus[f2_cmb < f2v]
        log(f"  f2 = {f2v}: excluded tau in [{ex.min():.1e}, {ex.max():.1e}] s" if ex.size else f"  f2 = {f2v}: nothing excluded")
        results.setdefault("cmb_windows", {})[str(f2v)] = [float(ex.min()), float(ex.max())] if ex.size else None
    # FIRAS
    tau_mu = np.logspace(6, 13, 100)
    worst_mu = max(1.4 * drho_over_rho_gamma(0.5, 300e-6, m, t)[0] for t in tau_mu if 5e4 < z_of_t(t) < 2e6)
    worst_y = max(0.25 * drho_over_rho_gamma(0.5, 300e-6, m, t)[0] for t in tau_mu if 1100 < z_of_t(t) < 5e4)
    log(f"FIRAS: max mu = {worst_mu:.1e} (limit 9e-5), max y = {worst_y:.1e} (limit 1.5e-5) for f2 = 0.5, m = 1 TeV; mass-scaled: y_max = {worst_y*1000/300:.1e} at 300 GeV")
    results["firas"] = dict(mu_max_f2_half=worst_mu, y_max_f2_half=worst_y, mu_limit=9e-5, y_limit=1.5e-5)
    # BBN
    results["bbn"] = dict(thresholds_MeV=dict(D=2.224, Be7=1.587, Li7=2.467, He3=5.49, He4=19.81), photon_energy_MeV=0.3,
                          note="300 keV photons below every photodissociation threshold; e+e- closed; nu injection Delta N_eff below")
    # Delta N_eff from nu nubar at tau_nunu
    for dk, tau in [(300, 1.24e6), (366, 4.6e5)]:
        dr, z = drho_over_rho_gamma(f2H, dk * 1e-6, m, tau)
        dNeff = dr * (8 / 7) * (11 / 4) ** (4 / 3)   # rho_nu per Neff unit = (7/8)(4/11)^(4/3) rho_gamma
        log(f"nu nubar injection at tau = {tau:.1e} s (delta = {dk}): Drho/rho_gamma = {dr:.1e} -> Delta N_eff = {dNeff:.1e}")
        results.setdefault("delta_Neff", {})[str(dk)] = dNeff

    # ---- D. Today: line flux and EGB
    D10 = D_factor(10.0)
    D5 = D_factor(5.0)
    D16 = D_factor(16.0)
    log(f"NFW decay D-factor: {D5:.2e} (5 deg), {D10:.2e} (10 deg), {D16:.2e} (16 deg) GeV cm^-2 sr")
    PHI_SPI = 3e-5
    lines = []
    for f2v in [0.5, 1e-3]:
        for tau in [1e18, 1e20, 1e22, 1e24]:
            phi = line_flux(f2v, tau, m, D10)
            lines.append(dict(f2=f2v, tau_s=tau, flux_10deg=phi, detectable=phi > PHI_SPI))
            log(f"line: f2 = {f2v}, tau = {tau:.0e} s, m = 1 TeV: Phi(10 deg) = {phi:.2e} ph/cm2/s ({'above' if phi > PHI_SPI else 'below'} SPI ~{PHI_SPI:.0e})")
    ratio_max = PHI_SPI * 4 * np.pi * m / D10
    log(f"SPI bound: f2_today/tau_gamma <= {ratio_max:.1e} s^-1 (10 deg, 1 TeV); tau_gamma >= {0.5/ratio_max:.1e} s for f2 = 0.5, {1e-3/ratio_max:.1e} s for f2 = 1e-3")
    I_egb = 2.0   # keV cm^-2 s^-1 sr^-1 at 300 keV (recalled)
    tau_egb = None
    for tau in np.logspace(17, 26, 200):
        if egb_intensity(0.5, tau, m) * 300 < I_egb:
            tau_egb = tau
            break
    log(f"EGB: decay intensity x E < 2 keV/cm2/s/sr requires tau_gamma >= {tau_egb:.1e} s for f2 = 0.5 (isotropic, z<1 shell)")
    results["today"] = dict(D_factors=dict(deg5=D5, deg10=D10, deg16=D16), lines=lines, spi_ratio_max=ratio_max,
                            tau_gamma_min_spi_f2_half=0.5 / ratio_max, tau_gamma_min_egb_f2_half=tau_egb,
                            higgsino_f2_today=0.5 * np.exp(-T_U_S / 1.24e6) if T_U_S / 1.24e6 < 700 else 0.0)

    # ---- Figure 1: tau vs f2 exclusion maps
    from matplotlib.colors import to_rgba
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    TT, FF = np.meshgrid(taus, f2_grid)
    f2_today = FF * np.exp(-T_U_S / TT)
    # left: visible (gamma) channel
    ax = axes[0]
    cmb_mask = FF > f2_cmb[None, :]
    spi_mask = f2_today / TT > ratio_max
    exo_mask = f2_today > 1 / N_exo_f2_1[300]
    ax.contourf(TT, FF, cmb_mask.astype(float), levels=[0.5, 1.5], colors=[to_rgba("#c0504d", 0.55)])
    ax.contourf(TT, FF, spi_mask.astype(float), levels=[0.5, 1.5], colors=[to_rgba("#f2a93b", 0.6)])
    ax.contourf(TT, FF, exo_mask.astype(float), levels=[0.5, 1.5], colors=[to_rgba("#4f81bd", 0.5)], hatches=["//"])
    ax.axvline(T_REC_S, color="k", ls=":", lw=0.8)
    ax.axvline(T_U_S, color="k", ls="--", lw=0.8)
    ax.text(T_REC_S * 1.3, 2e-7, "recombination", rotation=90, fontsize=8, va="bottom")
    ax.text(T_U_S * 1.3, 2e-7, "t$_U$", rotation=90, fontsize=8, va="bottom")
    ax.text(3e14, 3e-3, "CMB anisotropies\n(> 0.2 eV/baryon)", fontsize=8, color="#7b1f1c")
    ax.text(2e22, 0.02, "SPI 300 keV line\n(GC, 10$^\\circ$)", fontsize=8, color="#7a4a00")
    ax.text(3e17, 1e-5, "LZ exothermic\n(P011: $f_2^{today}$ > 1.2e-3)", fontsize=8, color="#1f3f6b")
    ax.set_title("visible channel: $\\chi_2 \\to \\chi_1 \\gamma$", fontsize=10)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e10, 1e26); ax.set_ylim(1e-7, 2.0)
    ax.set_xlabel("$\\tau(\\chi_2)$ [s]"); ax.set_ylabel("$f_2$ at freeze-out")
    # right: invisible channel
    ax = axes[1]
    ax.contourf(TT, FF, exo_mask.astype(float), levels=[0.5, 1.5], colors=[to_rgba("#4f81bd", 0.5)], hatches=["//"])
    ax.axvline(T_REC_S, color="k", ls=":", lw=0.8); ax.axvline(T_U_S, color="k", ls="--", lw=0.8)
    ax.text(3e17, 1e-5, "LZ exothermic", fontsize=8, color="#1f3f6b")
    ax.text(2e11, 3e-3, "no cosmological constraint\n($\\Delta N_{eff}$ < 1e-8, $\\mu$, y < 1e-7)", fontsize=8)
    # model points: dark photon tau(m_A') at delta = 300 keV
    for r in dpt:
        if r["delta_keV"] == 300 and r["mA_GeV"] in (0.5, 1.0, 2.6, 5.0, 10.0):
            ax.plot(r["tau_nunu_s"], dp[1]["f2_freeze"], "o", color="#2e7d32", ms=6)
            ax.annotate(f"{r['mA_GeV']:g}", (r["tau_nunu_s"], dp[1]["f2_freeze"]), textcoords="offset points",
                        xytext=(0, -13), fontsize=7.5, ha="center", color="#2e7d32")
    ax.text(1.5e13, 0.03, "dark photon, LZ-fit coupling ($f_2$ = 0.50):\n$\\tau_{\\nu\\bar\\nu}$ = 2.6e18 s $(m_{A'}/\\mathrm{GeV})^{-4}$; labels = $m_{A'}$ [GeV]",
            fontsize=7.5, color="#2e7d32")
    ax.set_title("invisible channel: $\\chi_2 \\to \\chi_1 \\nu\\bar\\nu$ (dark photon, LZ fit)", fontsize=10)
    ax.set_xscale("log"); ax.set_xlim(1e10, 1e26); ax.set_xlabel("$\\tau(\\chi_2)$ [s]")
    for a in axes:
        a.grid(alpha=0.25, which="major")
    # Higgsino markers (off-scale left): annotate
    axes[0].annotate("Higgsino ($f_2$ = 0.42): $\\tau_\\gamma$ ~ 0.06 s,\n$\\tau_{\\nu\\bar\\nu}$ = 5e5-1e6 s: off scale to the left,\nno constraint", xy=(1.5e10, 3e-6), fontsize=8, color="#444")
    axes[0].text(1.5e10, 0.3, "$\\leftarrow$ Higgsino", fontsize=8, color="#444")
    fig.suptitle("P026: excited-state lifetime vs freeze-out fraction, m = 1 TeV, $\\delta$ = 300 keV", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "P026_fig1_tau_f2_map.png"), dpi=160)
    plt.close(fig)

    # ---- Figure 2: f2(T) evolution
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for r in (higg[2], higg[4]):
        ax.plot(r["T_grid_MeV"], r["f2_grid"], label=f"Higgsino, $\\delta$ = {r['delta_keV']:.0f} keV (T$_*$ = {r['T_star_MeV']:.2f} MeV)")
    ax.plot(dp[1]["T_grid_MeV"], dp[1]["f2_grid"], label=f"dark photon (LZ fit), $\\delta$ = 300 keV (T$_*$ = {dp[1]['T_star_MeV']:.0f} MeV)", color="#2e7d32")
    Tg = np.array(higg[2]["T_grid_MeV"])
    ax.plot(Tg, np.exp(-0.3 / Tg) / (1 + np.exp(-0.3 / Tg)), "k:", lw=0.9, label="equilibrium $e^{-\\delta/T}/(1+e^{-\\delta/T})$, 300 keV")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(100, 1e-3); ax.set_ylim(1e-3, 0.7)
    ax.set_xlabel("bath temperature T [MeV]"); ax.set_ylabel("excited fraction $f_2 = n_2/(n_1+n_2)$")
    ax.legend(fontsize=7.5, loc="lower left"); ax.grid(alpha=0.25)
    ax.set_title("P026: kinetic freeze-out of the $\\chi_2 \\leftrightarrow \\chi_1$ bath transitions", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "P026_fig2_f2_evolution.png"), dpi=160); plt.close(fig)

    # ---- save
    with open(os.path.join(OUT, "P026_results.json"), "w") as f:
        json.dump(results, f, indent=1, default=float)
    with open(os.path.join(OUT, "recalled_knowledge.json"), "w") as f:
        json.dump(RECALLED, f, indent=1)
    import csv
    with open(os.path.join(OUT, "freeze_out.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "delta_keV", "T_star_MeV", "f2_eq_Tstar", "f2_freeze"])
        for r in higg + dp:
            w.writerow([r["label"], r["delta_keV"], f"{r['T_star_MeV']:.4f}", f"{r['f2_eq_Tstar']:.5f}", f"{r['f2_freeze']:.5f}"])
    with open(os.path.join(OUT, "higgsino_decays.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dec[0].keys())); w.writeheader(); w.writerows(dec)
    with open(os.path.join(OUT, "dark_photon_tau.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dpt[0].keys())); w.writeheader(); w.writerows(dpt)
    with open(os.path.join(OUT, "injection_scan.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(inj[0].keys())); w.writeheader(); w.writerows(inj)
    with open(os.path.join(OUT, "cmb_exclusion_curve.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["tau_s", "f2_excluded_above_gamma_channel", "deposited_eV_per_baryon_per_unit_f2"])
        for t, fc, du in zip(taus, f2_cmb, dep_unit):
            w.writerow([f"{t:.4e}", f"{fc:.4e}", f"{du:.4e}"])
    with open(os.path.join(OUT, "run_log.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")
    log("done.")
