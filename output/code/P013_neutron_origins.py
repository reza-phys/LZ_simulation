"""
P013 -- Neutron origins of a 248 keV single scatter: spectra, multiplicity and what
the 2.84 t yr already exclude.

Run from the simulation root:   .venv/bin/python output/code/P013_neutron_origins.py

Sections (numbers in the console log and in output/work/P013/P013_results.json):
  1. elastic kinematics (E_n,min per Xe isotope; recoil distributions; fraction > 200 keV)
  2. source spectra ((alpha,n) endpoints, 238U SF Watt, muon-induced power law) and
     first-scatter fractions above 200 keV
  3. vectorised Monte Carlo neutron transport in a homogeneous LXe cylinder ->
     single-scatter (SS) probability, SS recoil spectrum, low/high companion ratios
  4. expected 200-270 keV SS counts per source from LZ's own constraints and P(>=1)
  5. (n,n'gamma) hybrid loci in (S1c, log10 S2c) with the LZ-tuned NEST yields
  6. (AmBe activation: no neutron emitters; discussed in details.md)

Every cross-section / spectrum that is not in the LZ paper is RECALLED and flagged
in the RECALLED dict below with a reliability.
"""
from __future__ import annotations
import sys, os, json, math, time
sys.path.insert(0, 'output/code')
from common import lzcommon as lz
import numpy as np
from scipy import special, integrate, stats
import pandas as pd
import periodictable as pt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'output/work/P013'; FIG = OUT + '/figures'
os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(20260905)
t0 = time.time()

# ----------------------------------------------------------------------------
# constants and recalled inputs
# ----------------------------------------------------------------------------
U_MEV = 931.49410     # MeV per u (certain)
M_N = 939.56542       # neutron mass, MeV (certain)
M_ALPHA = 3727.379    # MeV (certain)
HBARC = 197.3269804   # MeV fm (certain)
RHO_LXE = 2.9         # g/cm3 (LZ NEST density; lzcommon)
N_XE = RHO_LXE / (pt.Xe.mass * 1.66053907e-24)   # atoms / cm3
M_XE = pt.Xe.mass * U_MEV                        # mean Xe mass, MeV

RECALLED = {
 "Q_values_alpha_n": ("19F(a,n)22Na -1.951; 13C(a,n)16O +2.216; 17O(a,n)20Ne +0.587; 18O(a,n)21Ne -0.697; "
                       "10B(a,n)13N +1.059; 11B(a,n)14N +0.158; 27Al(a,n)30P -2.644; 9Be(a,n)12C +5.702; "
                       "25Mg(a,n)28Si +2.654; 29Si(a,n)32S -1.527 MeV", "likely (+-0.05 MeV)"),
 "alpha_energies": ("U chain: 218Po 6.003, 214Po 7.687, 210Po 5.304; Th chain: 220Rn 6.288, 216Po 6.778, "
                    "212Bi 6.05/6.09 (36%), 212Po 8.785 MeV (64%)", "certain"),
 "thick_target_yields": ("per 1e6 alphas at 5.3 MeV: F ~12, natural C ~0.11, natural O ~0.07, Al ~0.7; "
                         "steep rise with E_alpha", "uncertain (x2)"),
 "f_C_high_energy_fraction": ("fraction of PTFE (alpha,n) neutrons above 6.5 MeV from 13C(a,n) ground-state "
                              "forward emission, estimated 2.5e-3 (range 5e-4 .. 1e-2)", "uncertain (x3)"),
 "watt_238U_SF": ("a = 0.988 MeV, b = 2.249 MeV^-1 (SOURCES-4C convention), nu_bar = 2.0, ~7 gammas/fission",
                  "likely"),
 "P_nu_238U_SF": ("P(nu)=0.05,0.25,0.37,0.25,0.07,0.01 for nu=0..5", "uncertain"),
 "muon_neutron_spectrum": ("dN/dE ~ E^-1 (10 MeV - 1 GeV) at production, steepening towards E^-2 above ~100 MeV",
                           "likely"),
 "sigma_nXe": ("elastic/nonelastic n-Xe cross sections (b): 0.5 MeV 6.0/0.3; 1: 5.5/0.9; 2: 4.8/1.4; 3: 4.2/1.7; "
               "5: 3.7/1.9; 8: 3.4/1.9; 10: 3.5/1.8; 14: 2.9/1.8; 20: 2.6/1.7; 50: 2.3/1.6; 100: 2.0/1.6; "
               "300: 1.3/1.5; 1000: 1.0/1.6", "likely (+-30%); assignment's 4 b at 10 MeV within range"),
 "black_disk_radius": ("strong-absorption radius R = 1.25 A^(1/3) fm = 6.35 fm for Xe; sharp-cutoff (black-disk) "
                       "Fraunhofer pattern |2J1(qR)/qR|^2 for elastic scattering", "likely (R +-15%)"),
 "Xe_inelastic_lines": ("129Xe 39.58 keV (t1/2 0.97 ns, mostly converted), 131Xe 80.19 keV (0.48 ns), "
                        "132Xe 667.7 keV", "certain"),
 "OD_geometry": ("per-particle escape/untag probabilities for fission companions: p_n ~ 0.07 (0.05-0.15), "
                 "p_gamma ~ 0.15 (0.10-0.30)", "uncertain"),
 "low_energy_NR_tolerance": ("<= 3 unexplained NR-band single scatters in 5.4-55 keV in the 2024 LZ search",
                             "uncertain (P003 used 3-5)"),
}
results = {"recalled": RECALLED}

def log(*a):
    print(*a); sys.stdout.flush()

# ----------------------------------------------------------------------------
# 1. Elastic kinematics
# ----------------------------------------------------------------------------
log("=== 1. Kinematics ===")
def E_R_max(E_n, M=M_XE):
    """max recoil energy (MeV) for neutron kinetic energy E_n (MeV) on nucleus of mass M (MeV); non-rel."""
    return 4 * M_N * M / (M_N + M) ** 2 * E_n

def p_lab(E_n):
    return math.sqrt(E_n ** 2 + 2 * M_N * E_n)

def k_cm_fm(E_n, M=M_XE):
    """CM wave number in fm^-1 (heavy-target approx p_cm = p_lab M/(M+m_n))."""
    return p_lab(E_n) * M / (M + M_N) / HBARC

iso = {}
for A, frac in lz.XE_ISOTOPES.items():
    M = pt.Xe[A].mass * U_MEV
    iso[A] = dict(abundance=frac, kappa=4 * M_N * M / (M_N + M) ** 2, E_n_min_248=0.248 / (4 * M_N * M / (M_N + M) ** 2),
                  E_n_min_200=0.200 / (4 * M_N * M / (M_N + M) ** 2))
kappa_mean = 4 * M_N * M_XE / (M_N + M_XE) ** 2
E_n_min_248 = 0.248 / kappa_mean; E_n_min_200 = 0.200 / kappa_mean; E_n_min_270 = 0.270 / kappa_mean
E_n_min_225 = 0.225 / kappa_mean; E_n_min_294 = 0.294 / kappa_mean
q_248 = math.sqrt(2 * M_XE * 0.248) / HBARC   # fm^-1
for A in sorted(iso):
    log(f"  {A}Xe abund {iso[A]['abundance']:.4f}  E_max/E_n = {iso[A]['kappa']:.5f}  E_n,min(248 keV) = {iso[A]['E_n_min_248']:.3f} MeV")
log(f"  natural mix: kappa={kappa_mean:.5f}; E_n,min = {E_n_min_200:.3f} (200), {E_n_min_248:.3f} (248), {E_n_min_270:.3f} (270 keV) MeV;"
    f" 248+-46 keV -> {E_n_min_225:.2f}-{E_n_min_294:.2f} MeV; q(248 keV) = {q_248:.3f} fm^-1")
results["kinematics"] = dict(isotopes={str(A): v for A, v in iso.items()}, kappa_mean=kappa_mean,
                             E_n_min_MeV={"200": E_n_min_200, "248": E_n_min_248, "270": E_n_min_270,
                                          "225": E_n_min_225, "294": E_n_min_294}, q_248_fm=q_248)

R_BD = 1.25 * pt.Xe.mass ** (1 / 3)   # fm
def f_blackdisk(x):
    x = np.asarray(x, float); out = np.ones_like(x)
    m = x > 1e-8; out[m] = (2 * special.j1(x[m]) / x[m]) ** 2
    return out

def recoil_pdf(E_n, model, R=R_BD, n=20001):
    """dsigma/dE_R (normalised) on a grid 0..E_max for models 'iso', 'bd' (black disk), 'gauss' (exp(-a(1-cos)))."""
    Emax = E_R_max(E_n); E = np.linspace(0, Emax, n)
    q = np.sqrt(2 * M_XE * E) / HBARC
    if model == 'iso':
        w = np.ones_like(E)
    elif model == 'bd':
        w = f_blackdisk(q * R)
    elif model == 'gauss':
        # dsigma/dOmega ~ exp(-a(1-cos)) with a = (kR)^2/2 (Gaussian approx. of the diffraction peak)
        k = k_cm_fm(E_n); a = (k * R) ** 2 / 2
        w = np.exp(-2 * a * E / Emax)
    w /= np.trapezoid(w, E)
    return E, w

def frac_between(E_n, lo, hi, model, R=R_BD, n=20001):
    E, w = recoil_pdf(E_n, model, R, n=n)
    m = (E >= lo) & (E <= hi)
    return float(np.trapezoid(w[m], E[m])) if m.sum() > 1 else 0.0

kin_rows = []
for E_n in [8.5, 10, 15, 50, 200]:
    Emax = E_R_max(E_n); k = k_cm_fm(E_n)
    row = dict(E_n_MeV=E_n, E_R_max_keV=1e3 * Emax, kR=k * R_BD, a_gauss=(k * R_BD) ** 2 / 2)
    for model in ['iso', 'bd', 'gauss']:
        row[f'f_gt200_{model}'] = frac_between(E_n, 0.200, 10, model)
        row[f'f_200_270_{model}'] = frac_between(E_n, 0.200, 0.270, model)
        row[f'f_lt55_{model}'] = frac_between(E_n, 0, 0.055, model)
    for Rv in [5.6, 7.1]:
        row[f'f_gt200_bd_R{Rv}'] = frac_between(E_n, 0.200, 10, 'bd', R=Rv)
        row[f'f_200_270_bd_R{Rv}'] = frac_between(E_n, 0.200, 0.270, 'bd', R=Rv)
    kin_rows.append(row)
    log(f"  E_n={E_n:6.1f} MeV E_max={1e3*Emax:7.1f} keV kR={k*R_BD:5.2f}  f(>200): iso {row['f_gt200_iso']:.3f} "
        f"bd {row['f_gt200_bd']:.2e} (R5.6 {row['f_gt200_bd_R5.6']:.1e}, R7.1 {row['f_gt200_bd_R7.1']:.1e}) gauss {row['f_gt200_gauss']:.1e}"
        f" | f(200-270): iso {row['f_200_270_iso']:.3f} bd {row['f_200_270_bd']:.2e} (R5.6 {row['f_200_270_bd_R5.6']:.1e}, R7.1 {row['f_200_270_bd_R7.1']:.1e})")
# black-disk zeros in E_R (the 200-270 keV window sits just above the second zero)
zeros_x = special.jn_zeros(1, 4); zeros_ER_keV = 1e3 * (zeros_x / R_BD * HBARC) ** 2 / (2 * M_XE)
log(f"  black-disk zeros (R={R_BD:.2f} fm) at E_R = " + ", ".join(f"{z:.0f}" for z in zeros_ER_keV) + " keV")
results["kinematics"]["blackdisk_zeros_keV"] = zeros_ER_keV.tolist()
kin_df = pd.DataFrame(kin_rows); kin_df.to_csv(f'{OUT}/kinematics_fractions.csv', index=False)
results["kinematics"]["R_blackdisk_fm"] = R_BD
results["kinematics"]["table"] = kin_rows

# ----------------------------------------------------------------------------
# 2. Source spectra
# ----------------------------------------------------------------------------
log("=== 2. Source spectra ===")
# 2a (alpha,n) endpoints from two-body kinematics
Qv = {"19F": -1.951, "13C": 2.216, "17O": 0.587, "18O": -0.697, "10B": 1.059, "11B": 0.158, "27Al": -2.644,
      "9Be": 5.702, "25Mg": 2.654, "29Si": -1.527}
targets = {"19F": (pt.F[19].mass, pt.Na[22].mass), "13C": (pt.C[13].mass, pt.O[16].mass), "17O": (pt.O[17].mass, pt.Ne[20].mass),
           "18O": (pt.O[18].mass, pt.Ne[21].mass), "10B": (pt.B[10].mass, pt.N[13].mass), "11B": (pt.B[11].mass, pt.N[14].mass),
           "27Al": (pt.Al[27].mass, pt.P[30].mass), "9Be": (pt.Be[9].mass, pt.C[12].mass), "25Mg": (pt.Mg[25].mass, pt.Si[28].mass),
           "29Si": (pt.Si[29].mass, pt.S[32].mass)}
alphas = {"218Po(U)": 6.003, "214Po(U)": 7.687, "210Po(U)": 5.304, "220Rn(Th)": 6.288, "216Po(Th)": 6.778, "212Po(Th)": 8.785}

def En_max_alpha_n(E_a, M_A_u, M_B_u, Q):
    M_A = M_A_u * U_MEV; M_B = M_B_u * U_MEV
    E_cm = E_a * M_A / (M_ALPHA + M_A); E_av = E_cm + Q
    if E_av <= 0: return 0.0
    E_ncm = E_av * M_B / (M_N + M_B)
    v_n = math.sqrt(2 * E_ncm / M_N); V = math.sqrt(2 * M_ALPHA * E_a) / (M_ALPHA + M_A)
    return 0.5 * M_N * (v_n + V) ** 2

end_rows = []
for tg, (mA, mB) in targets.items():
    row = dict(target=tg, Q_MeV=Qv[tg])
    for an, Ea in alphas.items():
        row[an] = En_max_alpha_n(Ea, mA, mB, Qv[tg])
    end_rows.append(row)
    log(f"  {tg:5s} Q={Qv[tg]:+.3f}: E_n,max = " + ", ".join(f"{an} {row[an]:.2f}" for an in alphas))
end_df = pd.DataFrame(end_rows); end_df.to_csv(f'{OUT}/alpha_n_endpoints.csv', index=False)
# alpha energy needed for 13C(a,n) g.s. to reach E_n,min(248)
from scipy.optimize import brentq
Ea_13C = brentq(lambda Ea: En_max_alpha_n(Ea, *targets["13C"], Qv["13C"]) - E_n_min_248, 1.0, 9.0)
Ea_19F = "never" if En_max_alpha_n(8.785, *targets["19F"], Qv["19F"]) < E_n_min_248 else brentq(
    lambda Ea: En_max_alpha_n(Ea, *targets["19F"], Qv["19F"]) - E_n_min_248, 1.0, 8.785)
log(f"  alpha energy for 13C(a,n) to reach {E_n_min_248:.2f} MeV: {Ea_13C:.2f} MeV; 19F(a,n) with 8.785 MeV alpha reaches "
    f"{En_max_alpha_n(8.785, *targets['19F'], Qv['19F']):.2f} MeV -> 19F can {'never' if Ea_19F=='never' else ''} give 248 keV")
results["alpha_n_endpoints"] = dict(rows=end_rows, Ea_13C_for_248=Ea_13C, F_max_Th=En_max_alpha_n(8.785, *targets['19F'], Qv['19F']),
                                    F_max_U=En_max_alpha_n(7.687, *targets['19F'], Qv['19F']))

# semi-quantitative 13C share of PTFE (alpha,n) neutrons above 8.3 MeV (recalled yields; uncertain x3)
# yields per 1e6 alphas (thick target, elemental): Y_F(E) ~ 12*(E/5.3)^3.5 ; Y_C(E) ~ 0.11*(E/5.3)^3.5 (natural C)
def Y_F(E): return 12.0 * (E / 5.3) ** 3.5
def Y_C(E): return 0.11 * (E / 5.3) ** 3.5
U_alphas = [4.20, 4.78, 4.69, 5.49, 6.00, 7.69, 5.30]      # 238U,234U,230Th,226Ra,222Rn,218Po->214Po, 210Po (7.69 is 214Po)
Th_alphas = [(4.01, 1), (5.42, 1), (5.69, 1), (6.29, 1), (6.78, 1), (6.07, 0.36), (8.78, 0.64)]
# fraction of 13C g.s. neutrons above 8.3 MeV given E_alpha: emission-angle fraction (isotropic CM approx.)
def frac_above(Ea, Ethr=E_n_min_248):
    mA, mB = targets["13C"]; Q = Qv["13C"]
    MA = mA * U_MEV; MB = mB * U_MEV
    E_cm = Ea * MA / (M_ALPHA + MA); E_av = E_cm + Q
    if E_av <= 0: return 0.0
    E_ncm = E_av * MB / (M_N + MB); v = math.sqrt(2 * E_ncm / M_N); V = math.sqrt(2 * M_ALPHA * Ea) / (M_ALPHA + MA)
    # lab energy 0.5 m (v^2 + V^2 + 2 v V cos) -> cos_min for Ethr ; isotropic in CM: fraction = (1-cos_min)/2
    c = (2 * Ethr / M_N - v ** 2 - V ** 2) / (2 * v * V)
    return 0.0 if c > 1 else (1.0 if c < -1 else (1 - c) / 2)
num_U = sum(2 * Y_C(E) * 0.6 * frac_above(E) for E in U_alphas)     # 2 C atoms per C2F4; 0.6 = ground-state branch (recalled, uncertain)
den_U = sum(4 * Y_F(E) for E in U_alphas) + sum(2 * Y_C(E) for E in U_alphas)
num_Th = sum(w * 2 * Y_C(E) * 0.6 * frac_above(E) for E, w in Th_alphas)
den_Th = sum(w * 4 * Y_F(E) for E, w in Th_alphas) + sum(w * 2 * Y_C(E) for E, w in Th_alphas)
f_hi_U = num_U / den_U; f_hi_Th = num_Th / den_Th
log(f"  estimated fraction of PTFE (a,n) neutrons above {E_n_min_248:.2f} MeV: U chain {f_hi_U:.2e}, Th chain {f_hi_Th:.2e} (recalled yields, x3 uncertain)")
results["alpha_n_endpoints"]["f_hi_estimate"] = dict(U=f_hi_U, Th=f_hi_Th)

# 2b spectra definitions (dN/dE, MeV^-1, unnormalised) --------------------------------
F_T = 1.1; F_END = 6.3
def S_F(E):   # 19F(alpha,n) bulk: E exp(-E/T) truncated (shape recalled/uncertain; irrelevant above 6.3 MeV)
    E = np.asarray(E, float); return np.where((E > 0) & (E < F_END), E * np.exp(-E / F_T), 0.0)
C_LO, C_HI = 6.5, 10.5
def S_C(E):   # 13C(alpha,n) high-energy component: flat 6.5-10.5 MeV
    E = np.asarray(E, float); return np.where((E >= C_LO) & (E <= C_HI), 1.0, 0.0)
F_C = 2.5e-3
def S_alpha_n(E, f_C=F_C):
    E = np.asarray(E, float)
    nF = integrate.quad(S_F, 0, F_END)[0]; nC = C_HI - C_LO
    return (1 - f_C) * S_F(E) / nF + f_C * S_C(E) / nC
WATT_A, WATT_B = 0.988, 2.249
def S_watt(E):
    E = np.asarray(E, float); return np.where(E > 0, np.exp(-E / WATT_A) * np.sinh(np.sqrt(WATT_B * np.clip(E, 0, None))), 0.0)
def S_mu(E, index=1.0, lo=10.0, hi=1000.0):
    E = np.asarray(E, float); return np.where((E >= lo) & (E <= hi), E ** (-index), 0.0)

def norm(S, lo, hi):
    return integrate.quad(S, lo, hi, limit=400, points=None)[0]
nW = norm(S_watt, 0, 30)
watt_mean = integrate.quad(lambda E: E * S_watt(E), 0, 30, limit=400)[0] / nW
watt_gt83 = integrate.quad(S_watt, E_n_min_248, 30, limit=400)[0] / nW
watt_gt67 = integrate.quad(S_watt, E_n_min_200, 30, limit=400)[0] / nW
watt_gt10 = integrate.quad(S_watt, 10, 30, limit=400)[0] / nW
an_gt83 = integrate.quad(lambda E: S_alpha_n(E), E_n_min_248, 11, limit=200)[0]
an_gt67 = integrate.quad(lambda E: S_alpha_n(E), E_n_min_200, 11, limit=200)[0]
log(f"  238U SF Watt: mean {watt_mean:.3f} MeV; f(>{E_n_min_200:.2f}) = {watt_gt67:.2e}; f(>{E_n_min_248:.2f}) = {watt_gt83:.2e}; f(>10) = {watt_gt10:.2e}")
log(f"  (alpha,n) model (f_C={F_C}): f(>{E_n_min_200:.2f}) = {an_gt67:.2e}; f(>{E_n_min_248:.2f}) = {an_gt83:.2e}")
results["spectra"] = dict(watt=dict(a=WATT_A, b=WATT_B, mean=watt_mean, f_gt_En200=watt_gt67, f_gt_En248=watt_gt83, f_gt10=watt_gt10),
                          alpha_n=dict(f_C=F_C, F_T=F_T, F_END=F_END, C_range=[C_LO, C_HI], f_gt_En200=an_gt67, f_gt_En248=an_gt83),
                          muon=dict(index=[1, 2], range=[10, 1000], f_gt_En248=1.0))

# 2c cross sections (recalled) and first-scatter fractions ----------------------------
SIG_E = np.array([0.5, 1, 2, 3, 5, 8, 10, 14, 20, 50, 100, 300, 1000.])
SIG_EL = np.array([6.0, 5.5, 4.8, 4.2, 3.7, 3.4, 3.5, 2.9, 2.6, 2.3, 2.0, 1.3, 1.0])
SIG_NON = np.array([0.3, 0.9, 1.4, 1.7, 1.9, 1.9, 1.8, 1.8, 1.7, 1.6, 1.6, 1.5, 1.6])
def sig_el(E): return np.exp(np.interp(np.log(np.clip(E, SIG_E[0], SIG_E[-1])), np.log(SIG_E), np.log(SIG_EL)))
def sig_non(E): return np.exp(np.interp(np.log(np.clip(E, SIG_E[0], SIG_E[-1])), np.log(SIG_E), np.log(SIG_NON)))
def lam_cm(sig_b): return 1.0 / (N_XE * sig_b * 1e-24)
for E in [2, 10, 100, 1000]:
    log(f"  sigma(E={E} MeV): el {sig_el(E):.2f} b, nonel {sig_non(E):.2f} b -> lambda_el {lam_cm(sig_el(E)):.1f} cm, lambda_tot {lam_cm(sig_el(E)+sig_non(E)):.1f} cm")
results["cross_sections"] = dict(E=SIG_E.tolist(), el=SIG_EL.tolist(), non=SIG_NON.tolist(), n_Xe_cm3=N_XE,
                                 lambda_el_10MeV=lam_cm(sig_el(10)), lambda_tot_10MeV=lam_cm(sig_el(10) + sig_non(10)),
                                 lambda_el_100MeV=lam_cm(sig_el(100)), lambda_tot_100MeV=lam_cm(sig_el(100) + sig_non(100)))

def first_scatter_fraction(S, lo, hi, Elo, Ehi, model, Emin_n):
    """fraction of first elastic scatters with E_R in [Elo,Ehi] for source spectrum S on [lo,hi]; weight S*sigma_el."""
    Eg = np.geomspace(max(lo, 0.05), hi, 600)
    w = S(Eg) * sig_el(Eg)
    fr = np.array([frac_between(E, Elo, Ehi, model, n=4001) if E > Emin_n else 0.0 for E in Eg])
    return float(np.trapezoid(w * fr, Eg) / np.trapezoid(w, Eg))

fs_rows = []
for name, S, lo, hi in [("alpha_n", lambda E: S_alpha_n(E), 0.05, 10.5), ("fission_238U", S_watt, 0.05, 30),
                        ("muon_E-1", lambda E: S_mu(E, 1.0), 10, 1000), ("muon_E-2", lambda E: S_mu(E, 2.0), 10, 1000)]:
    row = dict(source=name)
    for model in ['iso', 'bd']:
        row[f'f_first_gt200_{model}'] = first_scatter_fraction(S, lo, hi, 0.200, 10, model, E_n_min_200)
        row[f'f_first_200_270_{model}'] = first_scatter_fraction(S, lo, hi, 0.200, 0.270, model, E_n_min_200)
        row[f'f_first_lt55_{model}'] = first_scatter_fraction(S, lo, hi, 0.0, 0.055, model, 0.0)
    fs_rows.append(row)
    log(f"  first-scatter fractions {name:12s}: >200 keV iso {row['f_first_gt200_iso']:.2e} bd {row['f_first_gt200_bd']:.2e}; "
        f"200-270 iso {row['f_first_200_270_iso']:.2e} bd {row['f_first_200_270_bd']:.2e}; <55 keV bd {row['f_first_lt55_bd']:.3f}")
pd.DataFrame(fs_rows).to_csv(f'{OUT}/first_scatter_fractions.csv', index=False)
results["first_scatter"] = fs_rows

# ----------------------------------------------------------------------------
# 3. Monte Carlo transport in a homogeneous LXe cylinder
# ----------------------------------------------------------------------------
log("=== 3. Monte Carlo transport ===")
R_ACT, H_ACT = 72.8, 145.6          # active TPC (recalled: 7 t active; likely)
FVS = {"fv63x131": dict(r=63.0, z0=4.0, z1=135.0), "fv60x115": dict(r=60.0, z0=15.0, z1=130.0)}
for k, v in FVS.items():
    v["mass_t"] = math.pi * v["r"] ** 2 * (v["z1"] - v["z0"]) * RHO_LXE / 1e6
    log(f"  {k}: r<{v['r']} cm, {v['z0']}<z<{v['z1']} cm -> {v['mass_t']:.2f} t")
THRS = [1.0, 3.0, 5.0]              # keV: second-deposit resolvability thresholds
E_CUT = 0.1                          # MeV: stop tracking (max deposit 3 keV)

# black-disk inverse CDF table in x = qR
XG = np.linspace(0, 130, 260001)
FX = f_blackdisk(XG) * XG
CDF = np.concatenate([[0], np.cumsum(0.5 * (FX[1:] + FX[:-1]) * np.diff(XG))])

def sample_energies(kind, n):
    if kind == "F_bulk":
        Eg = np.linspace(1e-3, F_END, 4000); c = np.cumsum(S_F(Eg)); c /= c[-1]
        return np.interp(rng.random(n), c, Eg)
    if kind == "C_high":
        return rng.uniform(C_LO, C_HI, n)
    if kind == "watt_bulk":
        Eg = np.linspace(1e-3, E_n_min_200, 4000); c = np.cumsum(S_watt(Eg)); c /= c[-1]
        return np.interp(rng.random(n), c, Eg)
    if kind == "watt_high":
        Eg = np.linspace(E_n_min_200, 30, 4000); c = np.cumsum(S_watt(Eg)); c /= c[-1]
        return np.interp(rng.random(n), c, Eg)
    if kind == "mu_E1":
        return 10.0 * 100.0 ** rng.random(n)
    if kind == "mu_E2":
        u = rng.random(n); return 1.0 / (1 / 10.0 - u * (1 / 10.0 - 1 / 1000.0))
    if kind.startswith("mono"):
        return np.full(n, float(kind[4:]))
    raise ValueError(kind)

def sample_source(geom, n):
    """positions (n,3) and inward directions for 'wall', 'ends', 'surface' (area-weighted)."""
    if geom == "surface":
        A_wall = 2 * math.pi * R_ACT * H_ACT; A_ends = 2 * math.pi * R_ACT ** 2
        nw = rng.binomial(n, A_wall / (A_wall + A_ends))
        pw, dw = sample_source("wall", nw); pe, de = sample_source("ends", n - nw)
        return np.vstack([pw, pe]), np.vstack([dw, de])
    if geom == "wall":
        phi = rng.uniform(0, 2 * np.pi, n); r = R_ACT - 1e-3
        pos = np.stack([r * np.cos(phi), r * np.sin(phi), rng.uniform(0, H_ACT, n)], 1)
        normal = np.stack([-np.cos(phi), -np.sin(phi), np.zeros(n)], 1)
    elif geom == "ends":
        rr = R_ACT * np.sqrt(rng.random(n)); phi = rng.uniform(0, 2 * np.pi, n)
        top = rng.random(n) < 0.5
        z = np.where(top, H_ACT - 1e-3, 1e-3)
        pos = np.stack([rr * np.cos(phi), rr * np.sin(phi), z], 1)
        normal = np.stack([np.zeros(n), np.zeros(n), np.where(top, -1.0, 1.0)], 1)
    # isotropic in the inward hemisphere
    ct = rng.random(n); st_ = np.sqrt(1 - ct ** 2); ph = rng.uniform(0, 2 * np.pi, n)
    # basis around normal
    a = np.where(np.abs(normal[:, 2:3]) < 0.9, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
    e1 = np.cross(normal, a); e1 /= np.linalg.norm(e1, axis=1, keepdims=True); e2 = np.cross(normal, e1)
    d = ct[:, None] * normal + st_[:, None] * (np.cos(ph)[:, None] * e1 + np.sin(ph)[:, None] * e2)
    return pos, d

def rotate(d, cos_t, phi):
    """rotate unit vectors d by polar angle acos(cos_t) and azimuth phi."""
    sin_t = np.sqrt(np.clip(1 - cos_t ** 2, 0, 1))
    a = np.where(np.abs(d[:, 2:3]) < 0.9, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
    e1 = np.cross(d, a); e1 /= np.linalg.norm(e1, axis=1, keepdims=True); e2 = np.cross(d, e1)
    return cos_t[:, None] * d + sin_t[:, None] * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2)

def dist_to_exit(pos, d):
    a = d[:, 0] ** 2 + d[:, 1] ** 2; b = pos[:, 0] * d[:, 0] + pos[:, 1] * d[:, 1]
    c = pos[:, 0] ** 2 + pos[:, 1] ** 2 - R_ACT ** 2
    disc = np.clip(b ** 2 - a * c, 0, None)
    t_lat = np.where(a > 1e-12, (-b + np.sqrt(disc)) / np.where(a > 1e-12, a, 1), np.inf)
    t_ax = np.where(d[:, 2] > 1e-12, (H_ACT - pos[:, 2]) / np.where(d[:, 2] > 1e-12, d[:, 2], 1),
                    np.where(d[:, 2] < -1e-12, -pos[:, 2] / np.where(d[:, 2] < -1e-12, d[:, 2], 1), np.inf))
    return np.minimum(t_lat, t_ax)

def transport(E, pos, d, model='bd', R=R_BD, max_steps=400):
    n = len(E)
    ndep = {t: np.zeros(n, int) for t in THRS}
    Elast = {t: np.zeros(n) for t in THRS}
    poslast = {t: np.zeros((n, 3)) for t in THRS}
    nonel = np.zeros(n, bool); nsub = np.zeros(n, int); nel = np.zeros(n, int)
    active = np.ones(n, bool)
    for step in range(max_steps):
        idx = np.flatnonzero(active)
        if len(idx) == 0: break
        Ei = E[idx]; se = sig_el(Ei); sn = sig_non(Ei); lam = lam_cm(se + sn)
        s = -lam * np.log(rng.random(len(idx)))
        dex = dist_to_exit(pos[idx], d[idx])
        exit_ = s >= dex
        active[idx[exit_]] = False
        go = idx[~exit_]; s = s[~exit_]; se = se[~exit_]; sn = sn[~exit_]; Ei = Ei[~exit_]
        pos[go] += s[:, None] * d[go]
        is_non = rng.random(len(go)) < sn / (se + sn)
        gnon = go[is_non]
        nonel[gnon] = True; active[gnon] = False
        for t in THRS:
            ndep[t][gnon] += 1; Elast[t][gnon] = 1e3  # a nonelastic deposit counts as a (non-NR-clean) deposit; energy flag 1000 keV
            poslast[t][gnon] = pos[gnon]
        gel = go[~is_non]; Ee = Ei[~is_non]
        if len(gel):
            k = np.sqrt(Ee ** 2 + 2 * M_N * Ee) * M_XE / (M_XE + M_N) / HBARC
            xmax = 2 * k * R
            if model == 'bd':
                cmax = np.interp(xmax, XG, CDF)
                x = np.interp(rng.random(len(gel)) * cmax, CDF, XG)
            else:  # isotropic: uniform in x^2 (i.e. in E_R)
                x = xmax * np.sqrt(rng.random(len(gel)))
            q = x / R
            ER = (q * HBARC) ** 2 / (2 * M_XE)          # MeV
            cos_t = np.clip(1 - x ** 2 / (2 * (k * R) ** 2), -1, 1)
            d[gel] = rotate(d[gel], cos_t, rng.uniform(0, 2 * np.pi, len(gel)))
            E[gel] = Ee - ER
            nel[gel] += 1
            ERk = 1e3 * ER
            for t in THRS:
                m = ERk > t
                ndep[t][gel[m]] += 1; Elast[t][gel[m]] = ERk[m]; poslast[t][gel[m]] = pos[gel[m]]
            nsub[gel[ERk <= THRS[0]]] += 1
            active[gel[E[gel] < E_CUT]] = False
    return dict(ndep=ndep, Elast=Elast, poslast=poslast, nonel=nonel, nsub=nsub, nel=nel)

def in_fv(p, fv):
    return (np.hypot(p[:, 0], p[:, 1]) < fv["r"]) & (p[:, 2] > fv["z0"]) & (p[:, 2] < fv["z1"])

def summarise(res, n, label):
    out = {}
    for t in THRS:
        nd = res["ndep"][t]; any_dep = nd > 0
        ss = (nd == 1) & (~res["nonel"])
        for fvk, fv in FVS.items():
            infv = in_fv(res["poslast"][t], fv)
            sel = ss & infv; E = res["Elast"][t][sel]
            roi = (E > 5.4) & (E < 270)
            d = dict(N=n, n_anydep=int(any_dep.sum()), n_ss_fv=int(sel.sum()), n_ss_roi=int(roi.sum()),
                     n_lt55=int(((E > 5.4) & (E < 55)).sum()), n_200_270=int(((E >= 200) & (E <= 270)).sum()),
                     n_gt200=int((E > 200).sum()), n_gt150=int((E > 150).sum()),
                     n_single_nonel_fv=int(((nd == 1) & res["nonel"] & infv).sum()),
                     p_ss_given_dep=float(sel.sum() / max(any_dep.sum(), 1)))
            out[f"thr{t:g}_{fvk}"] = d
    return out

MC = {}
N_HIGH = 1_500_000; N_BULK = 600_000
samples = [("F_bulk", "wall", N_BULK), ("C_high", "wall", N_HIGH), ("watt_bulk", "wall", N_BULK), ("watt_high", "wall", N_HIGH),
           ("mu_E1", "surface", N_HIGH), ("mu_E2", "surface", N_HIGH), ("C_high_ends", "ends", N_BULK)]
for kind, geom, n in samples:
    ekind = "C_high" if kind == "C_high_ends" else kind
    E = sample_energies(ekind, n); pos, d = sample_source(geom, n)
    res = transport(E, pos, d, 'bd')
    MC[kind] = summarise(res, n, kind)
    s = MC[kind]["thr3_fv63x131"]
    log(f"  {kind:12s} ({geom}, N={n}): any-dep {s['n_anydep']/n:.3f}; SS-in-FV {s['n_ss_fv']/n:.4f} "
        f"(P(SS|dep)={s['p_ss_given_dep']:.3f}); SS 5.4-55 {s['n_lt55']}, 200-270 {s['n_200_270']}, >200 {s['n_gt200']}  [{time.time()-t0:.0f}s]")
# isotropic-angle variant for the high-energy (alpha,n) component (upper bound on large-angle scattering)
E = sample_energies("C_high", N_HIGH); pos, d = sample_source("wall", N_HIGH)
MC["C_high_iso"] = summarise(transport(E, pos, d, 'iso'), N_HIGH, "C_high_iso")
s = MC["C_high_iso"]["thr3_fv63x131"]
log(f"  C_high_iso   : SS-in-FV {s['n_ss_fv']/N_HIGH:.4f}; SS 5.4-55 {s['n_lt55']}, 200-270 {s['n_200_270']}, >200 {s['n_gt200']}")
# monoenergetic SS probabilities
mono = {}
for En in [8.5, 10, 15, 50, 100, 200]:
    n = 300_000; E = sample_energies(f"mono{En}", n); pos, d = sample_source("wall", n)
    res = transport(E, pos, d, 'bd'); sm = summarise(res, n, f"mono{En}")["thr3_fv63x131"]
    mono[str(En)] = dict(sm, mean_n_el=float(res["nel"].mean()), frac_nonel=float(res["nonel"].mean()))
    log(f"  mono {En:6.1f} MeV: P(any dep) {sm['n_anydep']/n:.3f}; P(SS in FV | dep) {sm['p_ss_given_dep']:.3f}; "
        f"SS 200-270/SS-FV {sm['n_200_270']/max(sm['n_ss_fv'],1):.2e}; <n_el> {res['nel'].mean():.2f}; nonel frac {res['nonel'].mean():.2f}")
results["MC"] = dict(samples=MC, mono=mono, N_high=N_HIGH, N_bulk=N_BULK, FVs=FVS, thresholds=THRS, R_active=R_ACT, H_active=H_ACT)

# combine components into per-source SS spectra ----------------------------------------
def combine(parts, thr=3.0, fvk="fv63x131"):
    """parts: list of (weight fraction of source neutrons, MC key). returns per-source-neutron SS rates."""
    keys = ["n_ss_fv", "n_ss_roi", "n_lt55", "n_200_270", "n_gt200", "n_gt150", "n_single_nonel_fv"]
    tot = {k: 0.0 for k in keys}; err = {k: 0.0 for k in keys}
    for w, key in parts:
        s = MC[key][f"thr{thr:g}_{fvk}"]
        for k in keys:
            tot[k] += w * s[k] / s["N"]; err[k] += (w * math.sqrt(s[k]) / s["N"]) ** 2   # bulk parts (E_n < 6.6 MeV) are kinematically zero above 200 keV
    return tot, {k: math.sqrt(v) for k, v in err.items()}

nF_bulk = 1 - F_C
watt_lo = 1 - watt_gt67
SOURCES = {
  "alpha_n (f_C=2.5e-3)": [(1 - F_C, "F_bulk"), (F_C, "C_high")],
  "alpha_n (f_C=1e-2)":   [(1 - 1e-2, "F_bulk"), (1e-2, "C_high")],
  "alpha_n (f_C=5e-4)":   [(1 - 5e-4, "F_bulk"), (5e-4, "C_high")],
  "alpha_n iso-angles (f_C=2.5e-3)": [(1 - F_C, "F_bulk"), (F_C, "C_high_iso")],
  "fission_238U": [(watt_lo, "watt_bulk"), (watt_gt67, "watt_high")],
  "muon_E-1": [(1.0, "mu_E1")],
  "muon_E-2": [(1.0, "mu_E2")],
}
comb_rows = []
for name, parts in SOURCES.items():
    for thr in THRS:
        for fvk in FVS:
            tot, err = combine(parts, thr, fvk)
            F_hi = tot["n_200_270"] / tot["n_ss_roi"] if tot["n_ss_roi"] > 0 else float('nan')
            F_hi_all = tot["n_200_270"] / (tot["n_ss_roi"] + tot["n_single_nonel_fv"]) if tot["n_ss_roi"] > 0 else float('nan')
            ratio = tot["n_lt55"] / tot["n_200_270"] if tot["n_200_270"] > 0 else float('inf')
            ratio_gt200 = tot["n_lt55"] / tot["n_gt200"] if tot["n_gt200"] > 0 else float('inf')
            comb_rows.append(dict(source=name, thr_keV=thr, fv=fvk, p_ss_roi=tot["n_ss_roi"], p_ss_200_270=tot["n_200_270"],
                                  p_ss_200_270_err=err["n_200_270"], F_200_270_of_roiSS=F_hi, F_200_270_of_allSS=F_hi_all,
                                  companions_lt55_per_200_270=ratio, companions_lt55_per_gt200=ratio_gt200,
                                  p_single_nonel_fv=tot["n_single_nonel_fv"]))
            if thr == 3.0 and fvk == "fv63x131":
                log(f"  {name:34s}: P(SS in ROI)/n = {tot['n_ss_roi']:.2e}; P(SS 200-270)/n = {tot['n_200_270']:.2e} +- {err['n_200_270']:.1e}; "
                    f"F(200-270|ROI SS) = {F_hi:.2e}; companions(<55)/(200-270) = {ratio:.3g}; (<55)/(>200) = {ratio_gt200:.3g}")
comb_df = pd.DataFrame(comb_rows); comb_df.to_csv(f'{OUT}/ss_spectrum_fractions.csv', index=False)
results["combined"] = comb_rows

# ----------------------------------------------------------------------------
# 4. Expected counts in 200-270 keV from LZ's constraints
# ----------------------------------------------------------------------------
log("=== 4. Expected counts ===")
N_DET_NR_UL = lz.LZ["detector_NR_fit_interval"][1]     # 0.118 (science sample, all energies in ROI)
N_MS_XCHECK = (0.02, 0.02)                              # supplement: SS neutrons from MS analysis
N_MU_UL = lz.LZ["muon_neutron_UL90"]                    # 4.6e-4 SS of any energy
F_SF_OVER_AN = 1e-2                                     # "up to two orders of magnitude lower"
untag_an = 1 - lz.LZ["neutron_veto_eff"][0]; untag_rock = 0.20
lamDN, lamPN = 0.87, 0.05
N_delayed_if_1 = 1.0 * lamDN / untag_an; N_prompt_if_1 = 1.0 * lamPN / untag_an
log(f"  veto partition: science {untag_an:.2f}, delayed {lamDN}, prompt {lamPN}; 1 science neutron SS <-> {N_delayed_if_1:.1f} delayed + {N_prompt_if_1:.2f} prompt NR-band events;"
    f" delayed fit UL 1.3 -> science {1.3*untag_an/lamDN:.3f} (paper: {N_DET_NR_UL})")
# fission silent-companion factor
P_NU = np.array([0.05, 0.25, 0.37, 0.25, 0.07, 0.01]); P_NU /= P_NU.sum(); NU = np.arange(6); NUBAR = float((P_NU * NU).sum())
def silent_factor(p_n, p_g, ngam=7.0):
    f_n = float((P_NU * NU * p_n ** np.clip(NU - 1, 0, None)).sum() / (P_NU * NU).sum())
    f_g = math.exp(-ngam * (1 - p_g))
    return f_n * f_g, f_n, f_g
sf_rows = []
for p_n in [0.05, 0.07, 0.15]:
    for p_g in [0.10, 0.15, 0.30]:
        f, fn, fg = silent_factor(p_n, p_g); sf_rows.append(dict(p_n=p_n, p_gamma=p_g, f_neutrons=fn, f_gammas=fg, f_silent=f))
sf_df = pd.DataFrame(sf_rows); sf_df.to_csv(f'{OUT}/fission_silent_factor.csv', index=False)
f_silent_c = silent_factor(0.07, 0.15)[0]; f_silent_max = silent_factor(0.15, 0.30)[0]; f_silent_min = silent_factor(0.05, 0.10)[0]
log(f"  238U SF: nu_bar(model)={NUBAR:.2f}; silent-companion factor central {f_silent_c:.1e} (range {f_silent_min:.1e}-{f_silent_max:.1e})")

def pick(name, thr=3.0, fvk="fv63x131"):
    r = comb_df[(comb_df.source == name) & (comb_df.thr_keV == thr) & (comb_df.fv == fvk)].iloc[0]
    return r
exp_rows = []
def add(label, N_hi, note):
    exp_rows.append(dict(scenario=label, N_200_270=N_hi, P_ge1=1 - math.exp(-N_hi), note=note))
    log(f"  {label:60s} N(200-270) = {N_hi:.2e}  P(>=1) = {1-math.exp(-N_hi):.2e}   {note}")
for src in ["alpha_n (f_C=2.5e-3)", "alpha_n (f_C=1e-2)", "alpha_n (f_C=5e-4)", "alpha_n iso-angles (f_C=2.5e-3)"]:
    r = pick(src)
    add(f"{src} | fit UL 0.118", N_DET_NR_UL * r.F_200_270_of_roiSS, "science-sample total NR SS x spectral fraction")
    add(f"{src} | MS cross-check 0.02", N_MS_XCHECK[0] * r.F_200_270_of_roiSS, "0.02+-0.02 SS neutrons")
    add(f"{src} | <=3 low-E NR events", 3.0 / r.companions_lt55_per_200_270, "empty 5.4-55 keV NR band (recalled tolerance)")
r = pick("fission_238U")
p_ss_an = pick("alpha_n (f_C=2.5e-3)").p_ss_roi
add("fission 238U | 1% of (a,n) x silent factor (central)", N_DET_NR_UL * F_SF_OVER_AN * (r.p_ss_roi / p_ss_an) * f_silent_c * r.F_200_270_of_roiSS,
    f"SS-prob ratio SF/(a,n) {r.p_ss_roi/p_ss_an:.2f}; silent {f_silent_c:.1e}")
add("fission 238U | 1% of (a,n), no multiplicity penalty", N_DET_NR_UL * F_SF_OVER_AN * (r.p_ss_roi / p_ss_an) * r.F_200_270_of_roiSS, "upper bound ignoring companions")
add("fission 238U | silent factor at its maximum", N_DET_NR_UL * F_SF_OVER_AN * (r.p_ss_roi / p_ss_an) * f_silent_max * r.F_200_270_of_roiSS, "")
for src in ["muon_E-1", "muon_E-2"]:
    r = pick(src)
    add(f"{src} | UL 4.6e-4 SS of any energy", N_MU_UL * r.F_200_270_of_allSS, "all single deposits incl. nonelastic")
    add(f"{src} | UL 4.6e-4, elastic-only denominator", N_MU_UL * r.F_200_270_of_roiSS, "conservative")
    add(f"{src} | <=3 low-E NR events", 3.0 / r.companions_lt55_per_200_270, "")
# totals (central) and P(>=1)
N_tot_central = (pick("alpha_n (f_C=2.5e-3)").F_200_270_of_roiSS * N_DET_NR_UL
                 + N_DET_NR_UL * F_SF_OVER_AN * (pick("fission_238U").p_ss_roi / p_ss_an) * f_silent_c * pick("fission_238U").F_200_270_of_roiSS
                 + N_MU_UL * pick("muon_E-1").F_200_270_of_allSS)
N_tot_generous = (pick("alpha_n iso-angles (f_C=2.5e-3)").F_200_270_of_roiSS * N_DET_NR_UL * 4  # f_C x4 -> 1e-2
                  + N_DET_NR_UL * F_SF_OVER_AN * (pick("fission_238U").p_ss_roi / p_ss_an) * f_silent_max * pick("fission_238U").F_200_270_of_roiSS
                  + N_MU_UL * pick("muon_E-1").F_200_270_of_roiSS)
add("ALL SOURCES central (0.118 normalisation)", N_tot_central, "sum of central scenarios")
add("ALL SOURCES generous (iso angles, f_C=1e-2, max silent, elastic denom.)", N_tot_generous, "stacked conservative choices")
# what would have been needed
r = pick("alpha_n (f_C=2.5e-3)")
N_roi_needed = 1.0 / r.F_200_270_of_roiSS
log(f"  (a,n): one 200-270 keV SS requires {N_roi_needed:.3g} ROI NR SS in the science sample, i.e. {N_roi_needed*lamDN/untag_an:.3g} delayed-veto NR events (fit UL 1.3),"
    f" and {r.companions_lt55_per_200_270:.3g} science-sample 5.4-55 keV NR events")
# muon UL consistency check
UL_from_1200yr = 2.303 * (220 / 365.25) / 1200
log(f"  muon UL check: 0 events in 1200 yr -> 90% UL {UL_from_1200yr:.2e} per 220 d before tagging; x0.2 untagged -> {0.2*UL_from_1200yr:.2e}; paper quotes {N_MU_UL:.1e}")
pd.DataFrame(exp_rows).to_csv(f'{OUT}/expected_counts.csv', index=False)
results["expected"] = dict(rows=exp_rows, N_roi_needed_alpha_n=N_roi_needed, silent_factor=dict(central=f_silent_c, min=f_silent_min, max=f_silent_max),
                           veto=dict(untagged_alpha_n=untag_an, untagged_rock=untag_rock, lamDN=lamDN, lamPN=lamPN, delayed_per_science=N_delayed_if_1),
                           muon_UL_check=dict(raw_1200yr=UL_from_1200yr, times_untagged=0.2 * UL_from_1200yr, paper=N_MU_UL),
                           N_total_central=N_tot_central, N_total_generous=N_tot_generous)

# ----------------------------------------------------------------------------
# 5. (n,n'gamma) hybrid loci in (S1c, log10 S2c)
# ----------------------------------------------------------------------------
log("=== 5. Inelastic hybrids ===")
g1, g2 = lz.LZ["g1"], lz.LZ["g2"]
E_nr_grid = np.arange(20, 320, 5.0)
nr_yields = np.array([lz.nest_nr_yields(E) for E in E_nr_grid])
S1_nr = g1 * nr_yields[:, 0]; S2_nr = g2 * nr_yields[:, 1]
er_lines = {"129Xe 39.6 keV": 39.58, "131Xe 80.2 keV": 80.19}
er_y = {k: lz.nest_er_yields(v, params=lz.NEST_ER_LZ) for k, v in er_lines.items()}
hyb = {}
ev_S1, ev_logS2 = lz.LZ["ev_S1c"], lz.LZ["ev_log10S2c"]
# pure-NR median at the event's S1c
E_at_540 = float(np.interp(ev_S1, S1_nr, E_nr_grid)); logS2_nr_at_540 = float(np.interp(ev_S1, S1_nr, np.log10(S2_nr)))
sigma_band_dex = 0.033   # P009 MC band width at S1c~540 (corpus)
log(f"  pure NR: S1c={ev_S1} <-> E_NR={E_at_540:.1f} keV, median log10S2c={logS2_nr_at_540:.3f}; event {ev_logS2:.3f} -> {(ev_logS2-logS2_nr_at_540)/sigma_band_dex:+.2f} sigma (0.033 dex)")
hyb["pure_NR"] = dict(E_at_540=E_at_540, logS2_median_at_540=logS2_nr_at_540, event_offset_sigma=(ev_logS2 - logS2_nr_at_540) / sigma_band_dex)
for k, (nph_er, ne_er) in er_y.items():
    S1_h = g1 * (nr_yields[:, 0] + nph_er); S2_h = g2 * (nr_yields[:, 1] + ne_er)
    E_match = float(np.interp(ev_S1, S1_h, E_nr_grid)); logS2_match = float(np.interp(ev_S1, S1_h, np.log10(S2_h)))
    # 248 keV NR + line
    i248 = int(np.argmin(np.abs(E_nr_grid - 250)))
    nph248, ne248 = lz.nest_nr_yields(248.0)
    S1_248 = g1 * (nph248 + nph_er); S2_248 = g2 * (ne248 + ne_er)
    hyb[k] = dict(ER_Nph=nph_er, ER_Ne=ne_er, E_NR_for_S1c540=E_match, logS2_at_S1c540=logS2_match,
                  offset_from_NR_median_dex=logS2_match - logS2_nr_at_540, offset_sigma=(logS2_match - logS2_nr_at_540) / sigma_band_dex,
                  S1c_248_plus_line=S1_248, logS2c_248_plus_line=math.log10(S2_248), in_ROI_248_plus_line=bool(S1_248 < 600 and math.log10(S2_248) < 4.15))
    log(f"  {k}: ER quanta Nph={nph_er:.0f} Ne={ne_er:.0f}; 248 keV NR + line -> S1c={S1_248:.0f} phd, log10S2c={math.log10(S2_248):.3f} (ROI? {hyb[k]['in_ROI_248_plus_line']});"
        f" at S1c=540: E_NR={E_match:.0f} keV, log10S2c={logS2_match:.3f} = {hyb[k]['offset_sigma']:+.1f} sigma above NR median (event {hyb['pure_NR']['event_offset_sigma']:+.1f})")
results["hybrids"] = hyb

# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
log("=== Figures ===")
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
# Fig 1: recoil spectra
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for E_n, c in zip([8.5, 10, 15, 50, 200], ["C0", "C1", "C2", "C3", "C4"]):
    E, w = recoil_pdf(E_n, 'bd'); ax[0].plot(1e3 * E, w / 1e3, c, label=f"{E_n:g} MeV")
    E, wi = recoil_pdf(E_n, 'iso'); ax[0].plot(1e3 * E, wi / 1e3, c, ls=':', lw=0.8)
ax[0].axvspan(200, 270, color='gold', alpha=0.4); ax[0].set_yscale('log'); ax[0].set_ylim(1e-7, 1e-1); ax[0].set_xlim(0, 800)
ax[0].set_xlabel("E_R [keV]"); ax[0].set_ylabel("dP/dE_R [keV$^{-1}$]"); ax[0].legend(fontsize=7, title="black disk (solid), isotropic (dotted)")
ax[0].set_title("Elastic n-Xe recoil spectra")
En_g = np.geomspace(6.5, 1000, 200)
for model, ls in [('bd', '-'), ('iso', ':')]:
    ax[1].plot(En_g, [frac_between(E, 0.2, 10, model, n=4001) if E > E_n_min_200 else 0 for E in En_g], 'k' + ls, label=f">200 keV, {model}")
    ax[1].plot(En_g, [frac_between(E, 0.2, 0.27, model, n=4001) if E > E_n_min_200 else 0 for E in En_g], 'C3' + ls, label=f"200-270 keV, {model}")
ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_ylim(1e-5, 1); ax[1].set_xlabel("E_n [MeV]"); ax[1].set_ylabel("fraction of elastic scatters")
ax[1].axvline(E_n_min_248, color='gray', ls='--'); ax[1].legend(fontsize=7); ax[1].set_title("Fraction of recoils in the window")
fig.tight_layout(); fig.savefig(f'{FIG}/P013_fig1_recoil_spectra.png', dpi=150); plt.close(fig)

# Fig 2: source spectra
fig, ax = plt.subplots(figsize=(5.5, 3.6))
Eg = np.geomspace(0.1, 1000, 2000)
ax.plot(Eg, S_alpha_n(Eg), label="(α,n) PTFE model (¹⁹F bulk + 0.25% ¹³C high-E)")
ax.plot(Eg, S_watt(Eg) / nW, label="²³⁸U SF Watt (a=0.988, b=2.249)")
nmu = integrate.quad(lambda E: S_mu(E, 1.0), 10, 1000)[0]
ax.plot(Eg, S_mu(Eg, 1.0) / nmu, label="muon-induced ∝E⁻¹ (10 MeV–1 GeV)")
ax.axvline(E_n_min_248, color='k', ls='--', label=f"E_n,min(248 keV) = {E_n_min_248:.2f} MeV")
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e-6, 3); ax.set_xlabel("E_n [MeV]"); ax.set_ylabel("dN/dE_n [MeV⁻¹] (normalised)")
ax.legend(fontsize=7); ax.set_title("Neutron source spectra (recalled shapes)")
fig.tight_layout(); fig.savefig(f'{FIG}/P013_fig2_source_spectra.png', dpi=150); plt.close(fig)

# Fig 3: expected counts bar chart
fig, ax = plt.subplots(figsize=(7, 3.8))
sel = [r for r in exp_rows if ("fit UL" in r["scenario"] and "2.5e-3" in r["scenario"] and "iso" not in r["scenario"])
       or "silent factor (central)" in r["scenario"] or ("muon_E-1 | UL 4.6e-4 SS" in r["scenario"]) or "ALL SOURCES" in r["scenario"]
       or ("iso-angles" in r["scenario"] and "fit UL" in r["scenario"])]
labels = [r["scenario"].split(" | ")[0][:34] + ("\n" + r["scenario"].split(" | ")[1][:30] if " | " in r["scenario"] else "") for r in sel]
vals = [max(r["N_200_270"], 1e-14) for r in sel]
ax.barh(range(len(sel)), vals, color='C0'); ax.set_yticks(range(len(sel))); ax.set_yticklabels(labels, fontsize=7)
ax.set_xscale('log'); ax.set_xlabel("expected single scatters with 200 < E_R < 270 keV in 2.84 t yr"); ax.axvline(1, color='r', ls='--')
ax.set_xlim(1e-14, 3); ax.set_title("Expected 200–270 keV neutron single scatters")
fig.tight_layout(); fig.savefig(f'{FIG}/P013_fig3_expected_counts.png', dpi=150); plt.close(fig)

# Fig 4: hybrid loci
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.plot(S1_nr, np.log10(S2_nr), 'r-', label="pure NR median (LZ-tuned NEST)")
for (k, (nph_er, ne_er)), c in zip(er_y.items(), ["C0", "C2"]):
    ax.plot(g1 * (nr_yields[:, 0] + nph_er), np.log10(g2 * (nr_yields[:, 1] + ne_er)), c, label=f"NR + {k} (n,n'γ)")
ax.plot([ev_S1], [ev_logS2], 'k*', ms=12, label="event (540.1, 3.967)")
ax.axvline(600, color='gray', ls=':'); ax.axhline(4.15, color='gray', ls=':')
ax.set_xlim(150, 1000); ax.set_ylim(3.6, 4.9); ax.set_xlabel("S1c [phd]"); ax.set_ylabel("log10 S2c [phd]"); ax.legend(fontsize=7)
ax.set_title("Inelastic NR+ER hybrids move away from the event")
fig.tight_layout(); fig.savefig(f'{FIG}/P013_fig4_hybrid_loci.png', dpi=150); plt.close(fig)

with open(f'{OUT}/P013_results.json', 'w') as f:
    json.dump(results, f, indent=1, default=float)
log(f"done in {time.time()-t0:.0f} s; results -> {OUT}/P013_results.json")
