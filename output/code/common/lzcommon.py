"""
lzcommon.py -- shared constants and physics utilities for the LZ 248 keV event corpus.

Everything numerical about the LZ result is taken from arXiv:2609.02823 (the provided
paper, file inputs/LZ_arXiv_2609.02823_fulltext.tex).  Halo parameters follow the
Baxter et al. (2021) recommendations, which the LZ paper says it adopted (recalled
knowledge, reliability: certain).  Run with the simulation root as working directory:

    .venv/bin/python output/code/P0XX_something.py

and in scripts:  sys.path.insert(0, 'output/code'); from common import lzcommon as lz
"""
from __future__ import annotations
import math
import numpy as np
from scipy import integrate, special, stats

# ----------------------------------------------------------------------------
# 1. LZ paper numbers (arXiv:2609.02823) -- section / table given for each
# ----------------------------------------------------------------------------
LZ = dict(
    arxiv="2609.02823",
    live_days=220.0,                     # Data Analysis paragraph
    fiducial_mass_t=4.71, fiducial_mass_err_t=0.08,
    exposure_tyr=2.84,                   # abstract  (220 d x 4.71 t = 2.84 t yr)
    exposure_check_tyr=220.0/365.25*4.71,
    S1c_min=3.0, S1c_max=600.0,          # WS ROI in phd
    log10S2c_min=2.75, log10S2c_max=4.15,
    S2_raw_min=645.0,
    g1=0.110, g1_err=0.002,              # phd/photon
    g2=34.5, g2_err=1.1,                 # phd/electron
    E_50pct_low_keV=5.4, E_50pct_high_keV=269.9,
    eff_plateau=0.96,                    # average NR efficiency 14-250 keV
    # event of interest
    ev_S1c=540.1, ev_S2c=9268.0, ev_log10S2c=math.log10(9268.0),
    ev_E_keV=248.0, ev_E_stat=23.0, ev_E_sys=23.0,
    ev_time_utc="2023-06-16T21:22:39",
    ev_z_above_cathode_cm=26.4, ev_r_from_true_wall_cm=26.9, ev_r_from_reco_wall_cm=23.4,
    ev_r2_cm2=45.9**2,
    ev_sigma_below_NR_median=1.5, ev_sigma_below_ER_median=6.7,
    # background expectations in the science sample (Table I, "Expected")
    bkg_expected=dict(
        internal_beta=(1341, 160), nu_ER=(140.6, 8.4), Xe136=(110.0, 16.5), CH3T_C14=(55.3, 3.1),
        Xe124=(21.0, 6.3), Kr83m=(17.1, 5.1), I125=(8.9, 2.7), detector_ER=(8.5, 3.4),
        accidentals=(2.7, 0.6), Xe127_Xe125=(1.5, 0.3), atm_nu=(0.11, 0.02), B8_hep=(0.057, 0.006),
        MSSI=(4.9e-3, 4.9e-3)),
    bkg_total_fit=(1713, 39), n_obs_science=1710,
    n_obs_prompt=66, n_fit_prompt=(69.4, 11.7), n_obs_delayed=55, n_fit_delayed=(50.4, 2.0),
    detector_NR_fit_interval=(0.0, 0.118),
    muon_neutron_UL90=4.6e-4,
    L10s_1000_bestfit=(1.0, +1.4, -0.7),
    bkg_highS1_panel=(0.0106, 0.0008),   # Fig.5 bottom panel (S1c > 500 phd), integrated background
    MSSI_veto_eff=(0.94, 0.02), neutron_veto_eff=(0.92, 0.04),
    charge_dead_fraction=0.003,
    local_sig_max=3.4, global_sig=2.6, n_models=616, n_distinct=293,
    MSSI_first_scatter_keV=(12, 2), MSSI_first_S1c=(69, 17), MSSI_second_S1c=471,
    MSSI_wall_second_keV=(77, 7), MSSI_RFR_second_keV=(204, 65, 38),
    Xe124_DEC_keV=64.3, I125_DEC_keV=67.3,
    I125_eff_halflife_d=(3.6, 0.2),
)

# NEST v2.4.5 parameters tuned by LZ (Supplemental Tables S3-S5)
NEST_NR_LZ = dict(alpha=11.2, beta=1.1, gamma=0.052, delta=-0.0533, epsilon=10.8, zeta=0.53,
                  eta=1.4, theta=0.31, iota=2.5, p=0.50, f1=1.39, f2=1.74, a=0.0230, b=0.0289, E0=74.7)
NEST_NR_DEFAULT = dict(alpha=11.0, beta=1.1, gamma=0.048, delta=-0.0533, epsilon=12.6, zeta=0.30,
                       eta=2.0, theta=0.30, iota=2.0, p=0.50, f1=1.0, f2=1.0)
NEST_ER_LZ = dict(m1=12.31, m2=84.91, m3=0.5707, m4=2.804, m5=34.04, m6=0.0, m7=82.57, m8=4.557,
                  m9=0.2207, m10=0.1272)
NEST_ER_DEFAULT_97V = dict(m1=14.6, m2=77.3, m3=0.8, m4=2.1, m5=19.3, m6=0.0, m7=78.0, m8=4.3, m9=0.33, m10=0.08)
NEST_ER_FLUCT_LZ = dict(A1=0.03174, mu1=2.588, sigma1=0.3627, alpha1=-0.3583,
                        A2=0.06211, mu2=5.2, sigma2=1.359, alpha2=-2.599)
DRIFT_FIELD_VCM = 96.5   # nestpy LZ_WS2024 detector central field; paper quotes NEST defaults "at 97 V/cm"
W_EV = 13.7              # recalled: NEST v2 work function (eV per quantum); 13.5 eV also used in literature

# Local significance table for the Lagrangians (Table S6), masses in GeV
LSIG_MASSES = [10, 12, 14, 17, 21, 30, 40, 50, 100, 200, 400, 1000, 4000]
LSIG = {
 "L1s":[0,0,0,0,0,0,0,0,0,0,0,0,0], "L1v":[0,0,0,0,0,0,0,0,0,0,0.3,1.3,1.2],
 "L2s":[0,0,0,0,0,0,0,0,2.4,2.8,2.9,3.0,3.1], "L2v":[0,0,0,0,0,0,0,0,2.4,2.5,3.0,3.1,3.1],
 "L3s":[0,0,0,0,0,0,0,0,0,1.1,1.5,1.7,1.8], "L3v":[0,0,0,0,0,0,0,0,0.2,2.2,2.5,2.6,2.7],
 "L4s":[0,0,0,0,0,0,0,0.1,2.6,3.1,3.1,3.1,3.0], "L4v":[0,0,0,0,0,0,0,0.1,2.7,3.1,3.2,3.1,3.1],
 "L5s":[0]*13, "L5v":[0,0,0,0,0,0,0,0,0,0,0.7,1.3,1.3],
 "L6s":[0,0,0,0,0,0,0,0,1.1,2.2,2.6,2.6,2.6], "L6v":[0,0,0,0,0,0,0.1,0.1,2.9,3.1,3.2,3.3,3.3],
 "L7s":[0,0,0,0,0,0,0,0,1.6,2.5,2.7,2.7,2.7], "L7v":[0,0,0,0,0,0,0,0,1.6,2.3,2.7,2.7,2.8],
 "L8s":[0,0,0,0,0,0,0,0,2.3,2.8,2.9,3.0,3.0], "L8v":[0,0,0,0,0,0,0,0,2.4,2.9,2.9,3.0,3.0],
 "L9s":[0,0,0,0,0,0,0,0,2.1,2.9,2.9,3.0,3.1], "L9v":[0,0,0,0,0,0,0,0,2.7,3.0,3.1,3.2,3.1],
 "L10s":[0,0,0,0,0,0,0,0,2.9,3.1,3.4,3.4,3.4], "L10v":[0,0,0,0,0,0,0,0,3.0,3.2,3.3,3.4,3.4],
 "L11s":[0,0,0,0,0,0,0,0,2.5,3.0,3.1,3.2,3.2], "L11v":[0,0,0,0,0,0,0,0,2.5,2.9,3.1,3.2,3.2],
 "L12s":[0,0,0,0,0,0,0,0,2.4,3.1,3.2,3.2,3.3], "L12v":[0,0,0,0,0,0,0,0,2.5,3.0,3.2,3.2,3.2],
 "L13s":[0,0,0,0,0,0,0,0,0,2.0,2.4,2.5,2.6], "L13v":[0,0,0,0,0,0,0,0,2.0,2.8,3.0,3.0,3.0],
 "L14s":[0,0,0,0,0,0,0,0,2.5,2.9,3.0,3.1,3.1], "L14v":[0,0,0,0,0,0,0,0,2.5,2.9,3.0,3.1,3.1],
 "L15s":[0,0,0,0,0,0,0,0,1.1,2.3,2.6,2.7,2.7], "L15v":[0,0,0,0,0,0,0,0,1.0,2.4,2.6,2.7,2.7],
 "L16s":[0,0,0,0,0,0,0,0,2.8,3.3,3.4,3.4,3.2], "L16v":[0,0,0,0,0,0,0,0,2.4,3.2,3.3,3.2,3.2],
 "L17s":[0,0,0,0,0,0,0,0,0,1.1,1.6,1.8,1.8], "L17v":[0,0,0,0,0,0,0,0,0,2.1,2.5,2.6,2.7],
 "L18s":[0,0,0,0,0,0,0,0,2.2,2.9,3.1,3.1,3.1], "L18v":[0,0,0,0,0,0,0,0,2.2,2.8,2.9,2.9,2.9],
 "L19s":[0,0,0,0,0,0,0,0,2.2,2.9,3.0,3.0,3.1], "L19v":[0,0,0,0,0,0,0,0,2.2,2.9,3.0,3.0,3.1],
 "L20s":[0,0,0,0,0,0,0,0.1,2.7,3.1,3.2,3.2,3.3], "L20v":[0,0,0,0,0,0,0,0.1,2.7,3.0,3.3,3.2,3.3],
}
# Inelastic table (Table S7): sigma[model][mass][delta]; None = "not physical"
OSIG_DELTAS = [0, 50, 100, 150, 200, 250, 300, 350]
OSIG = {
 "O1s": {400:[0,0,0.8,2.2,2.6,2.9,2.9,None], 1000:[0,0,0.8,2.2,2.7,2.9,3.0,3.3], 4000:[0,0,1.0,2.3,2.7,2.9,3.0,3.3]},
 "O1v": {400:[0.8,1.4,2.6,2.8,2.8,2.8,3.1,None], 1000:[1.3,1.6,2.6,2.8,2.9,3.0,3.4,3.4], 4000:[1.1,1.7,2.6,2.9,2.9,3.1,3.4,3.4]},
 "O4s": {400:[2.6,2.7,2.8,3.1,3.2,3.2,3.3,None], 1000:[2.7,2.8,2.8,3.0,3.2,3.2,3.4,3.4], 4000:[2.8,3.0,3.0,3.1,3.2,3.3,3.4,3.4]},
 "O4v": {400:[2.6,2.7,2.8,3.1,3.2,3.2,3.3,None], 1000:[2.7,2.8,2.8,3.0,3.2,3.2,3.3,3.4], 4000:[2.8,3.0,3.0,3.1,3.2,3.3,3.4,3.4]},
}

# ----------------------------------------------------------------------------
# 2. Physical constants and target
# ----------------------------------------------------------------------------
C_KMS = 299792.458
AMU_GEV = 0.9314941        # GeV per atomic mass unit (recalled, certain)
M_NUCLEON_GEV = 0.938272   # proton mass; nucleon mass used in NREFT normalisation (recalled, certain)
HBARC_GEV_FM = 0.1973269804
GEV_TO_CM2 = (HBARC_GEV_FM * 1e-13) ** 2   # 1 GeV^-2 = 3.8938e-28 cm^2
RHO0_GEV_CM3 = 0.3
V0_KMS, VESC_KMS = 238.0, 544.0
V_SUN_PEC = np.array([11.1, 12.2, 7.3])   # (U,V,W) km/s
V_EARTH_ORBIT = 29.8                        # km/s

# xenon isotopes: (A, abundance) natural, recalled/certain; masses via A*amu (adequate at 1e-3)
XE_ISOTOPES = {124: 0.00095, 126: 0.00089, 128: 0.0191, 129: 0.2644, 130: 0.0408, 131: 0.2118,
               132: 0.2689, 134: 0.1044, 136: 0.0887}
A_XE_MEAN = sum(A * f for A, f in XE_ISOTOPES.items()) / sum(XE_ISOTOPES.values())  # 131.29

def m_nucleus_gev(A: float) -> float:
    return A * AMU_GEV

# ----------------------------------------------------------------------------
# 3. Halo model: Earth velocity and velocity integrals
# ----------------------------------------------------------------------------
def v_earth_kms(day_of_year: float | None = None) -> float:
    """Speed of the Earth relative to the halo rest frame.  If day_of_year is None
    return the time-average |v_sun| (~250.6 km/s).  Otherwise a simple cosine model:
    v_E(t) = |v_sun| + 15.0 cos(2 pi (t - t_peak)/365.25) with t_peak = 2 June (day 153).
    The 15 km/s amplitude is the projection of the 29.8 km/s orbital speed onto the
    direction of solar motion (cos ~ 60 deg) -- recalled/standard, accurate to ~1 km/s."""
    v_sun = float(np.linalg.norm(V_SUN_PEC + np.array([0.0, V0_KMS, 0.0])))
    if day_of_year is None:
        return v_sun
    return v_sun + 15.0 * math.cos(2 * math.pi * (day_of_year - 153.0) / 365.25)

def eta0(vmin_kms, v_e=None, v0=V0_KMS, vesc=VESC_KMS):
    """Mean inverse speed <1/v> (in (km/s)^-1) for a truncated Maxwellian (McCabe 2010 form),
    vectorised over vmin."""
    if v_e is None:
        v_e = v_earth_kms()
    vmin = np.atleast_1d(np.asarray(vmin_kms, dtype=float))
    x, y, z = vmin / v0, v_e / v0, vesc / v0
    N = special.erf(z) - 2 * z / math.sqrt(math.pi) * math.exp(-z * z)
    out = np.zeros_like(x)
    m1 = x < z - y
    out[m1] = (special.erf(x[m1] + y) - special.erf(x[m1] - y) - 4 * y / math.sqrt(math.pi) * math.exp(-z * z)) / (2 * y)
    m2 = (x >= z - y) & (x < z + y)
    out[m2] = (special.erf(z) - special.erf(x[m2] - y) - 2 * (y + z - x[m2]) / math.sqrt(math.pi) * math.exp(-z * z)) / (2 * y)
    out = out / (N * v0)
    return out if out.size > 1 else float(out[0])

def vmax_kms(v_e=None, vesc=VESC_KMS):
    return (v_earth_kms() if v_e is None else v_e) + vesc

# ----------------------------------------------------------------------------
# 4. Kinematics (elastic and inelastic)
# ----------------------------------------------------------------------------
def mu_red(m1, m2):
    return m1 * m2 / (m1 + m2)

def vmin_kms(E_R_keV, m_chi_gev, A=A_XE_MEAN, delta_kev=0.0):
    """Minimum DM speed to give recoil E_R with splitting delta (delta>0: endothermic)."""
    mN = m_nucleus_gev(A)
    mu = mu_red(m_chi_gev, mN)
    E_R = E_R_keV * 1e-6   # GeV
    d = delta_kev * 1e-6
    arg = (mN * E_R / mu + d) / math.sqrt(2 * mN * E_R)   # dimensionless (units of c)
    return arg * C_KMS

def E_R_range_keV(m_chi_gev, v_kms, A=A_XE_MEAN, delta_kev=0.0):
    """Allowed recoil-energy interval [E-, E+] (keV) for DM speed v and splitting delta.
    Returns (nan, nan) if kinematically forbidden."""
    mN = m_nucleus_gev(A)
    mu = mu_red(m_chi_gev, mN)
    beta = v_kms / C_KMS
    d = delta_kev * 1e-6
    disc = 1 - 2 * d / (mu * beta**2)
    if disc < 0:
        return float("nan"), float("nan")
    # E_R(+/-) = (mu^2 v^2 / m_N) [ 1 - delta/(mu v^2) -/+ sqrt(1 - 2 delta/(mu v^2)) ]
    pref = mu**2 * beta**2 / mN
    mid = 1 - d / (mu * beta**2)
    return pref * (mid - math.sqrt(disc)) * 1e6, pref * (mid + math.sqrt(disc)) * 1e6

def delta_max_kev(E_R_keV, m_chi_gev, A=A_XE_MEAN, v_kms=None):
    """Largest splitting for which a recoil of E_R is kinematically possible at speed v."""
    if v_kms is None:
        v_kms = vmax_kms()
    mN = m_nucleus_gev(A)
    mu = mu_red(m_chi_gev, mN)
    E_R = E_R_keV * 1e-6
    return (v_kms / C_KMS * math.sqrt(2 * mN * E_R) - mN * E_R / mu) * 1e6

def m_chi_min_gev(E_R_keV, A=A_XE_MEAN, v_kms=None, delta_kev=0.0):
    """Smallest DM mass able to produce E_R at speed v (elastic or inelastic), by bisection."""
    if v_kms is None:
        v_kms = vmax_kms()
    lo, hi = 1.0, 1e6
    f = lambda m: vmin_kms(E_R_keV, m, A, delta_kev) - v_kms
    if f(hi) > 0:
        return float("inf")
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return hi

# ----------------------------------------------------------------------------
# 5. Form factor and spin-independent rates
# ----------------------------------------------------------------------------
def helm_F2(E_R_keV, A):
    """Helm form factor squared (Lewin & Smith 1996 parametrisation)."""
    mN = m_nucleus_gev(A)
    q_gev = math.sqrt(2 * mN * E_R_keV * 1e-6)
    q_fm = q_gev / HBARC_GEV_FM
    s = 0.9
    c = 1.23 * A ** (1 / 3) - 0.60
    a = 0.52
    rn = math.sqrt(c * c + 7 / 3 * math.pi**2 * a * a - 5 * s * s)
    x = q_fm * rn
    if x < 1e-6:
        return 1.0
    j1 = (math.sin(x) - x * math.cos(x)) / x**2
    return (3 * j1 / x) ** 2 * math.exp(-(q_fm * s) ** 2)

def dRdE_SI(E_R_keV, m_chi_gev, sigma_n_cm2, A=None, delta_kev=0.0, v_e=None, rho=RHO0_GEV_CM3):
    """SI (isoscalar, f_n=f_p) differential rate in events / (tonne yr keV), summed over Xe
    isotopes (natural abundance) unless A is given.  Inelastic via delta (keV)."""
    isos = XE_ISOTOPES if A is None else {A: 1.0}
    tot = 0.0
    for Ai, fi in isos.items():
        mN = m_nucleus_gev(Ai)
        mu_n = mu_red(m_chi_gev, M_NUCLEON_GEV)
        mu_N = mu_red(m_chi_gev, mN)
        sigma_N = sigma_n_cm2 * (mu_N / mu_n) ** 2 * Ai**2   # cm^2
        vmin = vmin_kms(E_R_keV, m_chi_gev, Ai, delta_kev)
        eta = eta0(vmin, v_e=v_e)   # (km/s)^-1
        n_T = 1e6 / (Ai * 1.66054e-24) * fi          # nuclei of isotope i per tonne of natural Xe
        # dsigma/dE_R = m_N sigma_N F^2 / (2 mu_N^2 v^2); rate = n_T (rho/m_chi) * m_N sigma_N F^2/(2 mu_N^2) * eta
        # [1/cm^3][cm^2][1/GeV][s/km] * c^2[(km/s)^2] * 1e5 [cm/km] * 1e-6 [GeV/keV] -> 1/(s keV)
        rate_per_s_keV = (n_T * (rho / m_chi_gev) * sigma_N * helm_F2(E_R_keV, Ai) * mN / (2 * mu_N**2)
                          * eta * C_KMS**2 * 1e5 * 1e-6)
        tot += rate_per_s_keV * 3.15576e7            # per tonne per year per keV
    return tot

# ----------------------------------------------------------------------------
# 6. NEST yields with LZ tuned parameters (via nestpy), p(E) break implemented by hand
# ----------------------------------------------------------------------------
def nest_nr_params_vector(E_keV, params=NEST_NR_LZ):
    p = params["p"]
    if "E0" in params and E_keV > params["E0"]:
        p = 0.5 + params["a"] * math.log(1 + params["b"] * (E_keV - params["E0"]))
    return [params["alpha"], params["beta"], params["gamma"], params["delta"], params["epsilon"],
            params["zeta"], params["eta"], params["theta"], params["iota"], p, params["f1"], params["f2"]]

def nest_nr_yields(E_keV, field=DRIFT_FIELD_VCM, params=NEST_NR_LZ, density=2.9):
    """Return (Nph, Ne) mean yields for an NR of energy E using nestpy with the given
    parameter set (default: LZ-tuned incl. the high-energy p(E) break)."""
    import nestpy
    det = nestpy.detectors.LZ_WS2024()
    nc = nestpy.NESTcalc(det)
    y = nc.GetYields(nestpy.interactions.NR, float(E_keV), density, field, 131.293, 54, nest_nr_params_vector(E_keV, params))
    return y.PhotonYield, y.ElectronYield

def nest_er_yields(E_keV, field=DRIFT_FIELD_VCM, params=None, density=2.9):
    """(Nph, Ne) for a beta ER using nestpy.  params: dict m1..m10 or None for NEST default."""
    import nestpy
    det = nestpy.detectors.LZ_WS2024()
    nc = nestpy.NESTcalc(det)
    er = [-1.0] * 10 if params is None else [params[f"m{i}"] for i in range(1, 11)]
    y = nc.GetYields(nestpy.interactions.beta, float(E_keV), density, field, 131.293, 54, list(nestpy.default_nr_parameters), er)
    return y.PhotonYield, y.ElectronYield

def combined_energy_keV(S1c, S2c, g1=LZ["g1"], g2=LZ["g2"], W_eV=W_EV):
    """ER-equivalent energy from the doke-style combined estimator E = W (S1c/g1 + S2c/g2)."""
    return W_eV * 1e-3 * (S1c / g1 + S2c / g2)

# ----------------------------------------------------------------------------
# 7. Statistics helpers
# ----------------------------------------------------------------------------
def p_to_sigma(p, one_sided=True):
    return stats.norm.isf(p) if one_sided else stats.norm.isf(p / 2)

def sigma_to_p(sig, one_sided=True):
    return stats.norm.sf(sig) if one_sided else 2 * stats.norm.sf(sig)

def poisson_p_at_least(n, mu):
    return stats.poisson.sf(n - 1, mu)

def global_p_from_trials(p_local, n_eff):
    return 1 - (1 - p_local) ** n_eff

if __name__ == "__main__":
    print("v_sun (km/s)", v_earth_kms(), " v_E(16 June, doy 167)", v_earth_kms(167))
    print("A_Xe mean", A_XE_MEAN)
    print("vmin(248 keV, 1000 GeV, elastic)", vmin_kms(248, 1000))
    print("delta_max(248 keV, 1000 GeV) [keV]", delta_max_kev(248, 1000))
    print("SI rate 1000 GeV 1e-45 cm2 @ 50 keV [/t/yr/keV]", dRdE_SI(50, 1000, 1e-45))

# ----------------------------------------------------------------------------
# 8. WimPyDD wrapper (the code LZ used).  Must run with cwd = simulation root.
# ----------------------------------------------------------------------------
_WD = None
def wd():
    """Import WimPyDD lazily (it prints target-loading messages on import)."""
    global _WD
    if _WD is None:
        import WimPyDD as WD
        _WD = WD
    return _WD

def wd_halo(day_of_year=None, v0=V0_KMS, vesc=VESC_KMS, v_sun_pec=V_SUN_PEC, yearly_modulation=False, **kw):
    """(vmin, delta_eta) arrays for WimPyDD with the Baxter-2021 SHM parameters.
    day_of_year: if given, evaluate the halo function on that day (Earth orbital velocity included).
    NOTE (P035, step 098): with day_of_year=None WimPyDD returns the SUN-FRAME halo (no Earth orbital
    motion, v_E = |v_sun| = 250.6 km/s), NOT an average over the year.  Corpus papers that call this the
    "annual average" mean the Sun frame; the true 12-day annual mean differs by x1.08/1.7/10 at
    delta = 300/350/380 keV (1 TeV).  For a genuine annual mean, average over day_of_year.
    yearly_modulation=True returns the modulation-amplitude halo function instead of the average."""
    WD = wd()
    # P002 found that WimPyDD's default v_min grid stops at ~795 km/s, truncating the June halo
    # (true v_max ~ 810 km/s) and giving negative rates near delta_max.  Pass an explicit grid.
    if "vmin" not in kw:
        kw["vmin"] = np.linspace(0.0, vesc + 300.0, 1200)
    args = dict(v_rot_gal=np.array([0.0, v0, 0.0]), v_sun_rot=np.array(v_sun_pec, dtype=float),
                v_esc_gal=vesc, yearly_modulation=yearly_modulation)
    if day_of_year is not None:
        args.update(full_year_sampling=False, day_of_the_year=day_of_year)
    args.update(kw)
    return WD.streamed_halo_function(**args)

def wd_hamiltonian(name, couplings):
    """couplings: dict {operator_index: (c0, c1)} with c^tau in GeV^-2 (tau=0 isoscalar, 1 isovector)
    in WimPyDD's convention.  Returns an eft_hamiltonian object."""
    WD = wd()
    # Closures with NO default arguments: WimPyDD inspects the lambda signatures and treats default
    # arguments as shared, named Hamiltonian parameters, so the earlier `lambda c=v:` form silently
    # applied one operator's coupling to all operators in multi-operator Hamiltonians (found by P031;
    # verified by the coordinator at step 102: O1+O4 combined rate was x2500 too large).  Single-operator
    # Hamiltonians built with the old form were unaffected.
    def _make(c):
        c0, c1 = float(c[0]), float(c[1])
        def f():
            return [c0, c1]
        return f
    wc = {k: _make(v) for k, v in couplings.items()}
    return WD.eft_hamiltonian(name, wc)

def wd_c_SI_from_sigma_n(sigma_n_cm2, m_chi_gev, cn_over_cp=1.0):
    """WimPyDD's own mapping (test_WimPyDD_installation.py): c^0,c^1 for O1 giving WIMP-nucleon
    cross-section sigma_N = c_N^2 mu^2/pi with c_N = c_p (= c_n for r=1)."""
    hbarc2 = 0.389e-27
    mn = 0.931
    mu = m_chi_gev * mn / (m_chi_gev + mn)
    cN = math.sqrt(math.pi * sigma_n_cm2 / hbarc2) / mu
    return (cN * (1 + cn_over_cp), cN * (1 - cn_over_cp))

def wd_rate(ham, m_chi_gev, E_keV, halo=None, delta_kev=0.0, j_chi=0.5, target=None, **args):
    """Differential rate in events / (tonne year keV) for natural xenon (or another WimPyDD target),
    summed over isotopes.  halo = (vmin, delta_eta) from wd_halo()."""
    WD = wd()
    if halo is None:
        halo = wd_halo()
    vmin, deta = halo
    tgt = WD.Xe if target is None else target
    E = np.atleast_1d(np.asarray(E_keV, dtype=float))
    r = np.array([WD.diff_rate(tgt, ham, m_chi_gev, float(e), vmin, deta, j_chi=j_chi, delta=delta_kev, **args) for e in E])
    r = r * 1000.0 * 365.25   # events/kg/day/keV -> events/tonne/year/keV
    return r if r.size > 1 else float(r[0])

M_V_GEV = 246.2   # Higgs vev used by LZ/Anand for "unit coupling" c = 1/m_v^2

# Convention settled by P003 (digitisation of LZ Fig. 1) and confirmed by P007:
#   LZ/Anand "unit coupling" c_i^s = 1/m_v^2  <=>  WimPyDD c^0 = 2/m_v^2  (WimPyDD c^0 = c_p + c_n, c^1 = c_p - c_n)
def wd_c_from_anand(c0_anand, c1_anand=0.0):
    """Convert Anand-et-al isospin couplings (c^0=(c_p+c_n)/2, c^1=(c_p-c_n)/2) to WimPyDD's (c_p+c_n, c_p-c_n)."""
    return (2.0 * c0_anand, 2.0 * c1_anand)
