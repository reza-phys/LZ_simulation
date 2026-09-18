"""
P039 -- Neutron-star kinetic heating by TeV inelastic dark matter:
does the LZ-fit cross-section light up old neutron stars?

Run from the simulation root:  .venv/bin/python output/code/P039_ns_heating.py [--no-wimpyc]

Sections
  1. Neutron-star model: escape velocity (GR), capture geometry, sigma_crit, capture fraction f(sigma)
  2. Inelastic threshold inside the star; fate of chi_2 (re-scatter vs exit vs decay)
  3. Kinetic (and annihilation) heating temperature, near-IR flux, JWST detectability
  4. White dwarfs: delta_max on C/O, sigma_crit, capture, heating, M4 bound
  5. The Sun: delta_max per element, inelastic capture on Fe/Ni (own Gould integral + WimPyC), equilibrium
Outputs: output/work/P039/{results.json, *.csv, figures/*.png}
Every recalled input is collected in RECALLED and written to results.json.
"""
import sys, os, json, math, time, argparse
import numpy as np
sys.path.insert(0, 'output/code')
from common import lzcommon as lz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'output/work/P039'
FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument('--no-wimpyc', action='store_true'); ARGS = ap.parse_args()

RECALLED = []
def rec(item, source, rel):
    RECALLED.append(dict(item=item, presumed_source=source, reliability=rel))

# ----------------------------------------------------------------------------- constants (recalled, certain)
G = 6.674e-8            # cgs
c = 2.99792458e10       # cm/s
Msun = 1.989e33         # g
Rsun = 6.9634e10        # cm
sigSB = 5.670e-5        # erg cm^-2 s^-1 K^-4
h_pl = 6.626e-27; kB = 1.381e-16
pc = 3.0857e18
GeV_erg = 1.602e-3
GeV_g = 1.783e-24
m_n = 0.93957; m_p = 0.93827; amu = lz.AMU_GEV   # GeV
Jy = 1e-23
rec('G, c, M_sun, R_sun, sigma_SB, h, k_B, pc, GeV<->erg/g, nucleon masses, Jy', 'CODATA/PDG/IAU', 'certain')

# ----------------------------------------------------------------------------- halo (Baxter 2021 SHM via lzcommon)
rho_loc_list = [0.3, 0.4]
rec('local DM density 0.3-0.4 GeV/cm^3', 'Baxter et al. 2021 / de Salas & Widmark 2021', 'likely')
vmin_h, deta_h = lz.wd_halo()               # Sun-frame, annual average; vmin in km/s, delta_eta in (km/s)^-1
inv_v = float(np.sum(deta_h))               # <1/v>  (km/s)^-1
mean_v = float(np.sum(vmin_h**2 * deta_h))  # <v>    km/s
v_rms = math.sqrt(float(np.sum(vmin_h**3 * deta_h)))  # sqrt(<v^2>)
print(f'halo (Sun frame): <1/v>^-1 = {1/inv_v:.1f} km/s, <v> = {mean_v:.1f} km/s, v_rms = {v_rms:.1f} km/s')
# star with a 400 km/s kick velocity through the halo (variation)
inv_v_400 = lz.eta0(0.0, v_e=400.0)
rec('old isolated NS space velocities ~100-400 km/s (kicks)', 'Hobbs et al. 2005 pulsar velocity distribution', 'likely')

# LZ-fit cross sections from the corpus
SIG = {
    'Higgsino (P007) sigma_n': 7.4e-39,
    'dark photon (P011) sigma_p, delta=250': 1.2e-42,
    'dark photon (P011) sigma_p, delta=300': 8.6e-42,
    'dark photon (P011) sigma_p, delta=350': 3.0e-40,
    'dark photon (P011) sigma_p, delta=365': 2.3e-39,
}
sig_unit_P021 = 6.5e-38 / 2.19       # sigma_n per unit (c1 m_v^2)^2, from P021 (kappa=2.19 <-> 6.5e-38)
P021 = {300: 7.4e-5 * sig_unit_P021, 350: 3.9e-3 * sig_unit_P021, 380: 2.19 * sig_unit_P021}
for d, s in P021.items():
    SIG[f'P021 best-fit sigma_n, delta={d}'] = s
m_chi = 1000.0

# ============================================================================= 1. NEUTRON STAR
M_ns = 1.5 * Msun; R_ns = 12e5
rec('NS benchmark M=1.5 Msun, R=12 km', 'standard (Baryakhtar et al. 2017 use 1.5 Msun, 10-12 km)', 'likely')
rec('neutron Fermi momentum ~0.35-0.5 GeV in the NS core; proton fraction Y_p ~ 0.05-0.1', 'nuclear EoS texts', 'likely')
B = 1 - 2 * G * M_ns / (R_ns * c**2)             # g_tt
v_esc_loc = c * math.sqrt(1 - B)                   # locally measured escape speed
gam = 1 / math.sqrt(B)                            # local Lorentz factor of DM falling from rest at infinity
R_inf = R_ns / math.sqrt(B)                       # apparent radius
v_esc_newt = math.sqrt(2 * G * M_ns / R_ns)
N_n = M_ns / (m_n * GeV_g)
n_n = N_n / (4 / 3 * math.pi * R_ns**3)
sig_crit_n = math.pi * R_ns**2 / N_n              # per neutron (tau_bar = sigma/sigma_crit, mean chord 4R/3)
Y_p = 0.07
sig_crit_p = sig_crit_n / Y_p
print(f'\n=== NS: B={B:.4f}, v_esc,loc={v_esc_loc/c:.4f}c (Newtonian {v_esc_newt/c:.4f}c), gamma={gam:.4f}, R_inf={R_inf/1e5:.2f} km')
print(f'N_n={N_n:.3e}, n_n={n_n:.3e} cm^-3, sigma_crit,n={sig_crit_n:.3e} cm^2, sigma_crit,p(Y_p={Y_p})={sig_crit_p:.3e}')

def f_uniform(tau_bar):
    """Capture (>=1 scatter) probability averaged over straight chords through a uniform sphere,
    tau_bar = sigma/sigma_crit = mean optical depth (mean chord 4R/3).  y = 2 n sigma R = 1.5 tau_bar."""
    y = 1.5 * np.asarray(tau_bar, float)
    out = np.where(y < 1e-3, tau_bar, 1 - 2 / np.maximum(y, 1e-300)**2 * (1 - np.exp(-y) * (1 + y)))
    return out
def f_exp(tau_bar):
    return 1 - np.exp(-np.asarray(tau_bar, float))

# single-scatter capture: max energy transfer to a neutron at rest from a heavy DM at beta*gamma
bg2 = (1 - B) / B
dE_max = 2 * m_n * bg2                            # GeV
KE_inf = lambda m, v_kms: 0.5 * m * (v_kms / lz.C_KMS)**2   # GeV
m_single = dE_max / ((v_rms / lz.C_KMS)**2 / 2)    # mass up to which one scatter captures
q_typ = m_n * math.sqrt(bg2)                       # typical momentum transfer ~ m_n beta gamma
print(f'dE_max (one scatter) = {dE_max:.3f} GeV; single-scatter capture up to m_chi = {m_single:.2e} GeV; q_typ = {q_typ:.2f} GeV vs p_F~0.4')

# capture (mass) rate: Ndot = (rho/m) pi R_inf^2 [ B <v> + v_esc,loc^2 <1/v> ]   (GR-corrected b_max)
def Mdot_ns(rho_gev, inv_v_kms, mean_v_kms):
    rho = rho_gev * GeV_g
    foc = B * mean_v_kms * 1e5 + (v_esc_loc**2) * (inv_v_kms / 1e5)   # cm/s
    return rho * math.pi * R_inf**2 * foc                              # g/s
def Mdot_ns_newt(rho_gev, inv_v_kms, mean_v_kms):
    rho = rho_gev * GeV_g
    foc = mean_v_kms * 1e5 + (v_esc_newt**2) * (inv_v_kms / 1e5)
    return rho * math.pi * R_ns**2 * foc

def T_ns(Mdot, annihilation=False):
    """Returns (T_local, T_infinity) in K. Kinetic: L_inf = Mdot c^2 (1-sqrt(B)); with annihilation L_inf = Mdot c^2."""
    L_inf = Mdot * c**2 * (1.0 if annihilation else (1 - math.sqrt(B)))
    L_loc = L_inf / B
    T_loc = (L_loc / (4 * math.pi * R_ns**2 * sigSB))**0.25
    return T_loc, T_loc * math.sqrt(B), L_inf

ns_rows = []
for rho in rho_loc_list + [1.0]:
    for lab, iv in [('v_star=Sun', inv_v), ('v_star=400', inv_v_400)]:
        Md = Mdot_ns(rho, iv, mean_v)
        Tl, Ti, L = T_ns(Md); Tla, Tia, La = T_ns(Md, True)
        ns_rows.append(dict(rho=rho, star_velocity=lab, Mdot_g_s=Md, L_inf_kin=L, T_loc_kin=Tl, T_inf_kin=Ti, L_inf_ann=La, T_inf_ann=Tia,
                            Mdot_newt=Mdot_ns_newt(rho, iv, mean_v)))
        print(f'rho={rho} {lab}: Mdot={Md:.2f} g/s, L_inf={L:.3e} erg/s, T_loc={Tl:.0f} K, T_inf={Ti:.0f} K; with annihilation T_inf={Tia:.0f} K')
import pandas as pd
pd.DataFrame(ns_rows).to_csv(os.path.join(OUT, 'ns_heating_table.csv'), index=False)
ref = ns_rows[2]     # rho=0.4, Sun-like velocity
rec('kinetic-heating temperature of an old NS with full capture T ~ 1700 K (with annihilation ~ 2500 K)', 'Baryakhtar, Bramante, Li, Linden, Raj, PRL 119, 131801 (2017)', 'likely')

# T vs sigma (kinetic), rho = 0.4
sig_grid = np.logspace(-47, -37, 400)
T_sig_n = ref['T_inf_kin'] * f_uniform(sig_grid / sig_crit_n)**0.25
T_sig_p = ref['T_inf_kin'] * f_uniform(sig_grid / sig_crit_p)**0.25
T_sig_n_ann = ref['T_inf_ann'] * f_uniform(sig_grid / sig_crit_n)**0.25
pd.DataFrame(dict(sigma_cm2=sig_grid, f_n=f_uniform(sig_grid / sig_crit_n), f_exp_n=f_exp(sig_grid / sig_crit_n),
                  T_inf_kin_n=T_sig_n, T_inf_kin_p=T_sig_p, T_inf_kin_ann_n=T_sig_n_ann)).to_csv(os.path.join(OUT, 'T_vs_sigma.csv'), index=False)

# saturation table for the LZ-fit cross-sections
sat_rows = []
for k, s in SIG.items():
    is_p = 'sigma_p' in k
    sc = sig_crit_p if is_p else sig_crit_n
    tau = s / sc
    lam = 1 / (n_n * (Y_p if is_p else 1.0) * s)          # mean free path of chi_2 (same sigma for down-scatter)
    sat_rows.append(dict(model=k, sigma=s, sigma_crit=sc, sigma_over_crit=tau, f=float(f_uniform(tau)),
                         T_inf_kin=ref['T_inf_kin'] * float(f_uniform(tau))**0.25, mfp_cm=lam, n_scatter_R=R_ns / lam,
                         P_exit_unscattered=math.exp(-R_ns / lam) if R_ns / lam < 700 else 0.0,
                         f_with_FF_0p03=float(f_uniform(0.03 * tau))))
    print(f'{k}: sigma/sigma_crit = {tau:.2e}, f = {sat_rows[-1]["f"]:.4f}, mfp = {lam:.2e} cm, R/mfp = {R_ns/lam:.2e}; f(FF 0.03)={sat_rows[-1]["f_with_FF_0p03"]:.3f}')
pd.DataFrame(sat_rows).to_csv(os.path.join(OUT, 'saturation_table.csv'), index=False)
rec('nucleon form-factor / relativistic suppression of DM-nucleon scattering at q ~ 1 GeV (factor ~0.03-0.3)', 'Bell, Busoni, Robles et al. 2019-2021 (NS capture with form factors)', 'likely')

# ============================================================================= 2. INELASTIC THRESHOLD IN THE NS; chi_2 FATE
mu_n = lz.mu_red(m_chi, m_n)
dmax_NR = 0.5 * mu_n * (v_esc_loc / c)**2 * 1e6           # keV
s_inv = m_chi**2 + m_n**2 + 2 * gam * m_chi * m_n
dmax_rel = (math.sqrt(s_inv) - m_chi - m_n) * 1e6           # keV, available CM kinetic energy
print(f'\n=== inelastic threshold in NS: delta_max(NR) = {dmax_NR/1e3:.1f} MeV, relativistic CM KE = {dmax_rel/1e3:.1f} MeV; delta=380 keV is {380/dmax_rel:.2e} of it')
taus = {'Higgsino tau_gamma ~0.06-0.1 s (P014/P026)': 0.06, 'Higgsino tau_nunu 4.6e5 s (P014/P026)': 4.6e5, 'dark photon tau_nunu >= 7e16 s (P011/P026)': 7e16, 'assignment lower bracket 0.02 s': 0.02}
chi2_rows = []
for k, t in taus.items():
    chi2_rows.append(dict(channel=k, tau_s=t, decay_length_km=c * t / 1e5, R_over_ctau=R_ns / (c * t), light_crossing_us=R_ns / c * 1e6))
    print(f'{k}: c tau = {c*t/1e5:.2e} km, R/(c tau) = {R_ns/(c*t):.2e}')
pd.DataFrame(chi2_rows).to_csv(os.path.join(OUT, 'chi2_fate.csv'), index=False)

# ============================================================================= 3. NEAR-IR FLUX AND JWST
def F_nu_nJy(T_inf, R_app, d_pc, lam_um):
    nu = c / (lam_um * 1e-4)
    Bnu = 2 * h_pl * nu**3 / c**2 / (math.exp(h_pl * nu / (kB * T_inf)) - 1)
    return math.pi * Bnu * (R_app / (d_pc * pc))**2 / Jy * 1e9
def AB(F_nJy):
    return -2.5 * math.log10(F_nJy * 1e-9 / 3631)
JWST_F200W_10sig_10ks_nJy = 9.0
rec('JWST/NIRCam F200W point-source sensitivity ~9 nJy (AB~28.9) at 10 sigma in 10^4 s; F444W ~ 20-25 nJy', 'STScI JWST documentation (pre-launch ETC estimates)', 'uncertain')
rec('nearest known isolated NSs are >~ 100 pc away (RX J1856 ~120 pc, young/hot); local NS number density ~ (1-4)e-4 pc^-3', 'pulsar/INS population studies (Sartore et al. 2010)', 'uncertain')
flux_rows = []
for d in [10, 30, 100]:
    for lab, T in [('kinetic', ref['T_inf_kin']), ('kinetic+annihilation', ref['T_inf_ann']), ('hypothetical 2000 K', 2000.0)]:
        for lam in [1.5, 2.0, 4.4]:
            F = F_nu_nJy(T, R_inf, d, lam)
            t5 = 1e4 * (0.5 * JWST_F200W_10sig_10ks_nJy / F)**2 if lam == 2.0 else float('nan')   # background-limited scaling to 5 sigma
            flux_rows.append(dict(d_pc=d, case=lab, T_inf=T, lam_um=lam, F_nJy=F, AB=AB(F), t_5sigma_s_F200W=t5))
fl = pd.DataFrame(flux_rows); fl.to_csv(os.path.join(OUT, 'nir_flux_table.csv'), index=False)
print('\n=== near-IR flux (rho=0.4):'); print(fl[fl.lam_um == 2.0].to_string(index=False))
# what a 2000 K measurement implies
rho_2000_kin = 0.4 * (2000 / ref['T_inf_kin'])**4
rho_2000_ann = 0.4 * (2000 / ref['T_inf_ann'])**4
print(f'T_inf=2000 K needs rho = {rho_2000_kin:.2f} GeV/cm^3 (kinetic only) or {rho_2000_ann:.2f} (kinetic+annihilation)')
# ISM Bondi accretion comparison (recalled: n=1 cm^-3, v=30 km/s; magnetospheric barrier usually prevents)
n_ism, v_ism = 1.0, 30e5
Mdot_bondi = 4 * math.pi * (G * M_ns)**2 * (n_ism * 1.67e-24) / v_ism**3
L_bondi = Mdot_bondi * c**2 * (1 - math.sqrt(B))
rec('Bondi-Hoyle accretion of ISM (n~1 cm^-3, v~30 km/s) onto an old NS if not magnetically inhibited', 'Bondi 1952; Blaes & Madau 1993', 'likely')
print(f'ISM Bondi accretion: Mdot={Mdot_bondi:.2e} g/s, L={L_bondi:.2e} erg/s = {L_bondi/ref["L_inf_kin"]:.1e} x DM kinetic heating')
rec('old (>~10^7-10^8 yr) isolated NSs cool below ~10^3 K without a heating source', 'NS cooling theory (Yakovlev & Pethick 2004; Page et al.)', 'likely')

# ============================================================================= 4. WHITE DWARFS
WDm = dict(M=1.0 * Msun, R=8000e5)
rec('WD benchmark M~1 Msun, R~8000 km (assignment); realistic 1 Msun WD has R~5500-6000 km', 'WD mass-radius relation (Chandrasekhar)', 'likely')
v_esc_wd_s = math.sqrt(2 * G * WDm['M'] / WDm['R'])
WD = lz.wd()
r_wd = np.asarray(WD.White_Dwarf.r_vec); vesc_wd_prof = np.asarray(WD.White_Dwarf.v_esc)
ratio_c_s = vesc_wd_prof[0] / vesc_wd_prof[-1]      # centre/surface from WimPyC 0.49 Msun profile
v_esc_wd_c = v_esc_wd_s * ratio_c_s
print(f'\n=== WD: v_esc surface {v_esc_wd_s/1e5:.0f} km/s, centre (x{ratio_c_s:.2f} from WimPyC profile) {v_esc_wd_c/1e5:.0f} km/s; WimPyC WD: {vesc_wd_prof[-1]:.0f}/{vesc_wd_prof[0]:.0f} km/s')
u_typ = 1 / inv_v      # typical asymptotic speed
def dmax_kev(A, v_kms):
    mA = A * amu; mu = lz.mu_red(m_chi, mA)
    return 0.5 * mu * (v_kms / lz.C_KMS)**2 * 1e6
targets_dm = []
for body, vs, vc in [('NS (neutron)', v_esc_loc / 1e5, v_esc_loc / 1e5), ('WD 1 Msun (C-12)', v_esc_wd_s / 1e5, v_esc_wd_c / 1e5), ('WD 1 Msun (O-16)', v_esc_wd_s / 1e5, v_esc_wd_c / 1e5),
                     ('WimPyC WD 0.49 Msun (C-12)', vesc_wd_prof[-1], vesc_wd_prof[0])]:
    A = 1.0079 if 'neutron' in body else (12 if 'C-12' in body else 16)
    if 'neutron' in body:
        targets_dm.append(dict(body=body, A=1, dmax_surface_keV=dmax_rel, dmax_centre_keV=dmax_rel)); continue
    ws = math.sqrt(vs**2 + u_typ**2); wc = math.sqrt(vc**2 + u_typ**2)
    targets_dm.append(dict(body=body, A=A, dmax_surface_keV=dmax_kev(A, ws), dmax_centre_keV=dmax_kev(A, wc)))
# WD sigma_crit and capture probability with the Helm form factor and the capture condition E_R + delta >= KE_inf
N_C = WDm['M'] / (12 * amu * GeV_g)
sig_crit_wd_nuc = math.pi * WDm['R']**2 / N_C
mu_C = lz.mu_red(m_chi, 12 * amu); mu_nuc = lz.mu_red(m_chi, m_n)
coh_iso = 12**2 * (mu_C / mu_nuc)**2        # sigma_A/sigma_n isoscalar
coh_p = 6**2 * (mu_C / mu_nuc)**2           # sigma_A/sigma_p proton-only (dark photon)
GF_s2 = 1.1664e-5 / math.sqrt(2); s2w = 0.2312
cp_H = GF_s2 * (1 - 4 * s2w); cn_H = -GF_s2
coh_H = ((6 * cp_H + 6 * cn_H) / cn_H)**2 * (mu_C / mu_nuc)**2   # Higgsino on C relative to sigma_n
rec('G_F, sin^2 theta_W, Higgsino Z couplings c_p = (G_F/sqrt2)(1-4 s_W^2), c_n = -G_F/sqrt2', 'PDG; P007', 'certain')
def helm_F2v(E_keV, A):
    """Vectorised copy of lz.helm_F2 (Lewin-Smith Helm parametrisation: s=0.9, c=1.23A^(1/3)-0.60, a=0.52 fm)."""
    E = np.asarray(E_keV, float)
    q_fm = np.sqrt(2 * A * amu * E * 1e-6) / lz.HBARC_GEV_FM
    s, a = 0.9, 0.52; cc = 1.23 * A**(1 / 3) - 0.60
    rn = math.sqrt(cc * cc + 7 / 3 * math.pi**2 * a * a - 5 * s * s)
    x = np.maximum(q_fm * rn, 1e-6)
    j1 = (np.sin(x) - x * np.cos(x)) / x**2
    return (3 * j1 / x)**2 * np.exp(-(q_fm * s)**2)
assert abs(helm_F2v(100.0, 131.29) / lz.helm_F2(100.0, 131.29) - 1) < 1e-12

def P_capture(A, v_kms, u_kms, delta_kev, npts=4000):
    """Fraction of the flat (F^2-weighted) recoil distribution that removes >= KE_inf: int_{Elo}^{E+} F^2 dE / (E+ - E-)."""
    Em, Ep = lz.E_R_range_keV(m_chi, v_kms, A=A, delta_kev=delta_kev)
    if not np.isfinite(Em):
        return 0.0, float('nan'), float('nan')
    Elo = max(Em, KE_inf(m_chi, u_kms) * 1e6 - delta_kev)
    if Elo >= Ep:
        return 0.0, Em, Ep
    E = np.linspace(Elo, Ep, npts)
    return float(np.trapezoid(helm_F2v(E, A), E) / (Ep - Em)), Em, Ep
wd_rows = []
w_s = math.sqrt((v_esc_wd_s / 1e5)**2 + u_typ**2)
for d in [0, 300, 350, 366, 380]:
    P, Em, Ep = P_capture(12, w_s, u_typ, d)
    wd_rows.append(dict(delta=d, E_minus_keV=Em, E_plus_keV=Ep, P_cap=P, sigma_crit_n_iso=sig_crit_wd_nuc / coh_iso / max(P, 1e-30),
                        sigma_crit_p_darkphoton=sig_crit_wd_nuc / coh_p / max(P, 1e-30), sigma_crit_n_Higgsino=sig_crit_wd_nuc / coh_H / max(P, 1e-30)))
wdf = pd.DataFrame(wd_rows); wdf.to_csv(os.path.join(OUT, 'wd_capture_thresholds.csv'), index=False)
print(f'N_C={N_C:.2e}, sigma_crit per C nucleus = {sig_crit_wd_nuc:.2e} cm^2; coherence factors iso {coh_iso:.0f}, p-only {coh_p:.0f}, Higgsino {coh_H:.0f}')
print(wdf.to_string(index=False))
# WD capture fractions and heating for the LZ-fit cross-sections
def Mdot_wd(rho_gev, f):
    rho = rho_gev * GeV_g
    return f * rho * math.pi * WDm['R']**2 * (mean_v * 1e5 + v_esc_wd_s**2 * inv_v / 1e5)
wd_heat = []
for k, s in SIG.items():
    d = 366 if 'Higgsino' in k else int(k.split('=')[-1])
    row = wdf[wdf.delta == (d if d in wdf.delta.values else 366)].iloc[0]
    sc = row.sigma_crit_n_Higgsino if 'Higgsino' in k else (row.sigma_crit_p_darkphoton if 'sigma_p' in k else row.sigma_crit_n_iso)
    f = float(f_uniform(s / sc))
    for rho in [0.4, 1000.0]:
        Md = Mdot_wd(rho, f)
        L_kin = Md * v_esc_wd_s**2 / 2; L_ann = Md * c**2
        wd_heat.append(dict(model=k, delta=d, sigma=s, sigma_crit=sc, f=f, rho=rho, Mdot_g_s=Md, L_kin=L_kin, L_ann=L_ann,
                            T_kin=(L_kin / (4 * math.pi * WDm['R']**2 * sigSB))**0.25, T_ann=(L_ann / (4 * math.pi * WDm['R']**2 * sigSB))**0.25))
wdh = pd.DataFrame(wd_heat); wdh.to_csv(os.path.join(OUT, 'wd_heating_table.csv'), index=False)
print(wdh[['model', 'f', 'rho', 'Mdot_g_s', 'L_ann', 'T_kin', 'T_ann']].to_string(index=False))
L_M4_faint = 10**-4.5 * 3.828e33
rec('M4 globular cluster: assumed core DM density ~10^3 GeV/cm^3 (highly uncertain, GCs may hold little DM); faintest observed WDs L ~ 10^-4.5 L_sun', 'Bertone & Fairbairn 2008; McCullough & Fairbairn 2010; Hansen et al. 2004 (M4 WD sequence)', 'uncertain')
rec('coolest known field WDs have T_eff ~ 3000-4000 K', 'WD cooling sequences (Gaia)', 'likely')
print(f'faintest M4 WD L ~ {L_M4_faint:.2e} erg/s')

# ============================================================================= 5. THE SUN
# WimPyC normalises the loaded profile so that its volume integral is 1 in units of M/R^3 (WC_package celestial_body.__init__);
# convert back to g/cm^3 with M_sun/R_sun^3 = 5.89 g/cm^3 (checked: mass integral -> 1.000 M_sun)
unit_rho = Msun / Rsun**3
r_sun = np.asarray(WD.Sun.r_vec); vesc_sun = np.asarray(WD.Sun.v_esc); rho_tot_sun = np.asarray(WD.Sun.rho_tot) * unit_rho
rho_i = {k: np.asarray(v) * unit_rho for k, v in WD.Sun.rho_i.items()}
# validation of the profile: integrate mass
M_int = np.trapezoid(4 * math.pi * (r_sun * Rsun)**2 * rho_tot_sun, r_sun * Rsun) / Msun
print(f'\n=== Sun: profile mass integral = {M_int:.3f} Msun; v_esc centre {vesc_sun[0]:.0f}, surface {vesc_sun[-1]:.0f} km/s')
sun_targets = {'1H': 1.0079, '4He': 4.0026, '16O': 15.995, '56Fe': 55.935, '58Ni': 57.935}
for sym, A in sun_targets.items():
    targets_dm.append(dict(body=f'Sun ({sym})', A=A, dmax_surface_keV=dmax_kev(A, math.sqrt(vesc_sun[-1]**2 + u_typ**2)),
                           dmax_centre_keV=dmax_kev(A, math.sqrt(vesc_sun[0]**2 + u_typ**2))))
targets_dm.append(dict(body='Earth lab, Xe (LZ, 16 June)', A=131.29, dmax_surface_keV=lz.delta_max_kev(248.0, m_chi, v_kms=lz.vmax_kms(v_e=lz.v_earth_kms(167))), dmax_centre_keV=float('nan')))
tdm = pd.DataFrame(targets_dm); tdm.to_csv(os.path.join(OUT, 'delta_max_by_body.csv'), index=False)
print(tdm.to_string(index=False))
# threshold radius in the Sun for Fe/Ni
mu_Fe = lz.mu_red(m_chi, 55.935 * amu)
def r_threshold(delta_kev, A):
    mu = lz.mu_red(m_chi, A * amu)
    w_need = math.sqrt(2 * delta_kev * 1e-6 / mu) * lz.C_KMS
    v_need = math.sqrt(max(w_need**2 - u_typ**2, 0))
    idx = np.where(vesc_sun >= v_need)[0]
    if len(idx) == 0: return 0.0, w_need
    return float(r_sun[idx[-1]]), w_need
thr_rows = []
for d in [300, 350, 366, 380]:
    rF, wF = r_threshold(d, 55.935)
    mfrac = np.trapezoid((4 * math.pi * (r_sun * Rsun)**2 * rho_tot_sun)[r_sun <= rF], (r_sun * Rsun)[r_sun <= rF]) / Msun if rF > 0 else 0
    thr_rows.append(dict(delta=d, w_needed_kms=wF, r_thr_Rsun=rF, mass_fraction_inside=mfrac))
    print(f'delta={d}: need w >= {wF:.0f} km/s on Fe -> r <= {rF:.2f} Rsun (mass fraction {mfrac:.2f})')
pd.DataFrame(thr_rows).to_csv(os.path.join(OUT, 'sun_threshold_radius.csv'), index=False)

# --- own Gould-type capture integral
def capture_gould(sym, A, sigma_ref, coupling_sq_ratio, delta_kev, rho_gev=0.4, use_helm=True):
    """C [s^-1] = n_chi * sum_r N_i(r) * sum_u deta(u) * sigma_A c^2 * (m_A/(2 mu^2)) int_{Elo}^{E+} F^2 dE.
    sigma_A = sigma_ref * coupling_sq_ratio * (mu_A/mu_n)^2  (coupling_sq_ratio = A^2 isoscalar, Z^2 proton-only, ((Zc_p+Nc_n)/c_n)^2 Higgsino)."""
    mA = A * amu; mu = lz.mu_red(m_chi, mA)
    sigA = sigma_ref * coupling_sq_ratio * (mu / mu_nuc)**2
    n_chi = rho_gev / m_chi                                   # cm^-3
    rr = r_sun * Rsun; dN = 4 * math.pi * rr**2 * rho_i[sym] / (mA * GeV_g)   # nuclei per cm of radius
    u = vmin_h[1:]; de = deta_h[1:]
    tot = 0.0
    for j in range(len(rr)):
        w = np.sqrt(u**2 + vesc_sun[j]**2)
        beta2 = (w / lz.C_KMS)**2
        disc = 1 - 2 * delta_kev * 1e-6 / (mu * beta2)
        ok = disc > 0
        if not ok.any(): continue
        pref = mu**2 * beta2[ok] / mA; mid = 1 - delta_kev * 1e-6 / (mu * beta2[ok]); sq = np.sqrt(disc[ok])
        Em = pref * (mid - sq) * 1e6; Ep = pref * (mid + sq) * 1e6            # keV
        Elo = np.maximum(Em, KE_inf(m_chi, u[ok]) * 1e6 - delta_kev)
        good = Elo < Ep
        if not good.any(): continue
        # integrate F^2 on a common 60-point grid per stream
        Eg = Elo[good][:, None] + (Ep[good] - Elo[good])[:, None] * np.linspace(0, 1, 60)[None, :]
        F2 = helm_F2v(Eg, A) if use_helm else np.ones_like(Eg)
        I = np.trapezoid(F2, Eg, axis=1) * 1e-6                                # GeV
        X = mA / (2 * mu**2) * I                                               # dimensionless (w^2/c^2 units)
        tot += dN[j] * np.sum(de[ok][good] / 1e5 * X) * (rr[1] - rr[0] if j == 0 else rr[j] - rr[j - 1])
    return n_chi * sigA * c**2 * tot

t0 = time.time()
C_own = {}
C_own['Fe elastic iso 1e-42'] = capture_gould('56Fe', 55.935, 1e-42, 56**2, 0.0)
C_own['Fe delta=300 iso 1e-42'] = capture_gould('56Fe', 55.935, 1e-42, 56**2, 300.0)
C_own['H elastic 1e-42'] = capture_gould('1H', 1.0079, 1e-42, 1.0, 0.0, use_helm=False)
print(f'own Gould integral ({time.time()-t0:.1f} s): ' + '; '.join(f'{k}: {v:.3e}' for k, v in C_own.items()))
print('WimPyC probe values (elastic H1 2.53e18, elastic Fe56 2.42e20, inelastic Fe56 delta=300 2.14e20 at sigma=1e-42, rho=0.4)')

# --- WimPyC inelastic capture for the LZ-fit models
wimpyc_rows = []
if not ARGS.no_wimpyc:
    c0H, c1H = cp_H + cn_H, cp_H - cn_H                           # WimPyDD c^0 = c_p + c_n, c^1 = c_p - c_n (P003 convention)
    ham_H = lz.wd_hamiltonian('P039_Higgsino', {1: (c0H, c1H)})
    tgts = [WD.Fe56, WD.Ni58]
    C_geom = WD.wimp_capture_geom(WD.Sun, m_chi, vmin_h, deta_h, rho_loc=0.4)
    def run(ham, delta, label, sigma_label):
        t = time.time(); Cs = {}
        for tg in tgts:
            Cs[tg.symbol] = float(WD.wimp_capture(WD.Sun, ham, vmin_h, deta_h, mchi=m_chi, delta=delta, rho_loc=0.4, targets_list=[tg], verbose=False))
        C = sum(Cs.values())
        GA = float(WD.wimp_capture_annihilation(WD.Sun, m_chi, C, sigma_v=3e-26)) if C > 0 else 0.0
        teq = float(WD.wimp_capture_annihilation.t_eq) if C > 0 else float('nan')
        wimpyc_rows.append(dict(model=label, sigma=sigma_label, delta=delta, C_Fe=Cs['56Fe'], C_Ni=Cs['58Ni'], C_total=C, C_over_geom=C / C_geom,
                                Gamma_A_thermal_eq=GA, t_eq_yr=teq / 3.156e7))
        print(f'WimPyC {label} delta={delta}: C_Fe={Cs["56Fe"]:.3e} C_Ni={Cs["58Ni"]:.3e} s^-1 (C/geom={C/C_geom:.1e}); Gamma_A={GA:.3e}, t_eq={teq/3.156e7:.2e} yr  [{time.time()-t:.1f} s]')
    for d in [0, 300, 350, 366, 380]:
        run(ham_H, d, 'Higgsino (Z exchange)', 7.4e-39)
    for d, sp in [(250, 1.2e-42), (300, 8.6e-42), (350, 3.0e-40), (365, 2.3e-39)]:
        cp = lz.wd_c_SI_from_sigma_n(sp, m_chi, 1.0)[0] / 2      # c_p from sigma_p; c_n = 0 -> c0 = c1 = c_p
        run(lz.wd_hamiltonian(f'P039_DP{d}', {1: (cp, cp)}), d, 'dark photon (proton-only)', sp)
    for d, sn in P021.items():
        run(lz.wd_hamiltonian(f'P039_iso{d}', {1: lz.wd_c_SI_from_sigma_n(sn, m_chi, 1.0)}), d, 'isoscalar P021 fit', sn)
    # IceCube-scale reference: elastic SD-like capture on H at sigma = 1e-40 (recalled IceCube 1 TeV W+W- SD limit ~1e-40 cm^2)
    C_H_ref = float(WD.wimp_capture(WD.Sun, lz.wd_hamiltonian('P039_H', {1: lz.wd_c_SI_from_sigma_n(1e-40, m_chi, 1.0)}), vmin_h, deta_h, mchi=m_chi, rho_loc=0.4, targets_list=[WD.H1], verbose=False))
    rec('IceCube solar WIMP limit at 1 TeV (W+W-): sigma_SD ~ 1e-40 cm^2, i.e. Gamma_A ~ 1e20 s^-1', 'IceCube, EPJC 77, 146 (2017)', 'uncertain')
    wimpyc_rows.append(dict(model='reference: elastic on H, sigma=1e-40 (IceCube-scale)', sigma=1e-40, delta=0, C_Fe=0, C_Ni=0, C_total=C_H_ref, C_over_geom=C_H_ref / C_geom,
                            Gamma_A_thermal_eq=C_H_ref / 2, t_eq_yr=float('nan')))
    print(f'reference elastic H capture at 1e-40: C={C_H_ref:.3e} s^-1 -> Gamma_A ~ {C_H_ref/2:.2e}')
    # non-thermalised population: inelastic-only DM stops scattering once its speed at the centre falls below
    # v_thr = sqrt(2 delta/mu); its orbit then reaches the apoapsis r_a with v_esc(r_a)^2 = v_esc(0)^2 - v_thr^2.
    # Bracket the annihilation volume by 4/3 pi r_a^3 instead of the Griest-Seckel thermal volume.
    Tc = WD.Sun.T_c; rho_c = rho_tot_sun[0]
    V1 = (3 * 1.38e-23 * Tc / (2 * m_chi * 6.674e-11 * rho_c) * 5.62e27)**1.5; V2 = (3 * 1.38e-23 * Tc / (4 * m_chi * 6.674e-11 * rho_c) * 5.62e27)**1.5
    V_th = V1**2 / V2
    def r_apo(delta_kev, A=55.935):
        mu = lz.mu_red(m_chi, A * amu); vthr = math.sqrt(2 * delta_kev * 1e-6 / mu) * lz.C_KMS
        va = math.sqrt(max(vesc_sun[0]**2 - vthr**2, 0)); idx = np.where(vesc_sun >= va)[0]
        return float(r_sun[idx[-1]]) if len(idx) else 1.0
    for row in wimpyc_rows:
        if row['C_total'] > 0 and 'reference' not in row['model'] and row['delta'] > 0:
            ra = r_apo(row['delta']); V_orbit = 4 / 3 * math.pi * (ra * Rsun)**3
            CA_orb = 3e-26 / V_orbit; teq_orb = 1 / math.sqrt(row['C_total'] * CA_orb)
            row['V_thermal_cm3'] = V_th; row['r_apo_Rsun'] = ra; row['V_orbit_cm3'] = V_orbit; row['t_eq_orbit_yr'] = teq_orb / 3.156e7
            row['Gamma_A_orbit'] = row['C_total'] / 2 * math.tanh(4.6e9 * 3.156e7 / teq_orb)**2
            print(f'  {row["model"]} delta={row["delta"]}: r_apo={ra:.2f} Rsun, t_eq(orbit)={teq_orb/3.156e7:.2e} yr, Gamma_A(orbit)={row["Gamma_A_orbit"]:.2e}')
    print(f'thermal V_eff = {V_th:.2e} cm^3')
    pd.DataFrame(wimpyc_rows).to_csv(os.path.join(OUT, 'sun_capture_wimpyc.csv'), index=False)
    rec('thermalisation of inelastic-only DM in the Sun stalls once v < sqrt(2 delta/mu); loop-level elastic Higgsino sigma ~1e-49 cm^2', 'Nussinov, Wang, Yavin 2009; Menon et al. 2010; Hill & Solon 2014 (via P007)', 'likely')

# ============================================================================= FIGURES (Okabe-Ito fixed order; one axis per panel)
OI = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00', '#56B4E9']
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5})
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.plot(sig_grid, T_sig_n, color=OI[0], lw=2, label='kinetic heating, neutron target')
ax.plot(sig_grid, T_sig_p, color=OI[1], lw=2, ls='--', label='kinetic heating, proton-only coupling (Y$_p$=0.07)')
ax.plot(sig_grid, T_sig_n_ann, color=OI[2], lw=2, ls=':', label='kinetic + annihilation')
ax.axvline(sig_crit_n, color='0.5', lw=0.8); ax.text(sig_crit_n * 1.3, 120, f'σ$_{{crit,n}}$ = {sig_crit_n:.1e} cm²', rotation=90, va='bottom', color='0.35')
ax.axvline(sig_crit_p, color='0.5', lw=0.8, ls='--'); ax.text(sig_crit_p * 1.3, 120, f'σ$_{{crit,p}}$ = {sig_crit_p:.1e}', rotation=90, va='bottom', color='0.35')
ax.axvspan(1.2e-42, 2.3e-39, color=OI[1], alpha=0.12, lw=0); ax.text(5e-41, 260, 'P011 dark photon\nσ$_p$(δ=250–365 keV)', ha='center', color=OI[4], fontsize=8)
ax.axvline(7.4e-39, color=OI[3], lw=1.2); ax.text(7.4e-39 * 1.4, 400, 'P007 Higgsino\nσ$_n$ = 7.4e-39', color=OI[3], fontsize=8)
for d, s in P021.items():
    ax.plot([s], [ref['T_inf_kin'] * float(f_uniform(s / sig_crit_n))**0.25], marker='o', ms=6, color=OI[0], mec='white', mew=1)
    ax.annotate(f'P021 δ={d}', (s, ref['T_inf_kin']), textcoords='offset points', xytext=(0, -14 if d != 350 else 8), ha='center', fontsize=7.5, color=OI[0])
ax.axhline(2000, color='0.6', lw=0.8, ls='-.'); ax.text(1.5e-47, 2050, 'hypothetical JWST 2000 K', color='0.4', fontsize=8)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(1e-47, 1e-37); ax.set_ylim(100, 4000)
ax.set_xlabel('DM–nucleon cross-section σ [cm²]'); ax.set_ylabel('T$_\\infty$ of an old neutron star [K]')
ax.set_title('Kinetic heating of a 1.5 M$_\\odot$, 12 km neutron star (ρ$_χ$ = 0.4 GeV cm$^{-3}$, m$_χ$ = 1 TeV)', fontsize=9.5)
ax.legend(frameon=False, loc='lower right', fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P039_T_vs_sigma.png'), dpi=180); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 4.4))
labels = [r['body'] for r in targets_dm]; ys = np.arange(len(labels))
sur = np.array([r['dmax_surface_keV'] for r in targets_dm]); cen = np.array([r['dmax_centre_keV'] for r in targets_dm])
ax.axvspan(300, 387, color=OI[3], alpha=0.18, lw=0); ax.text(420, 0.9, 'LZ-fit\nδ = 300–387 keV', ha='left', va='center', fontsize=8, color=OI[3])
for i in range(len(labels)):
    if np.isfinite(cen[i]) and cen[i] != sur[i]:
        ax.plot([sur[i], cen[i]], [i, i], color='0.7', lw=1.5, zorder=1)
        ax.plot(cen[i], i, marker='o', ms=7, color=OI[1], mec='white', zorder=3)
    ax.plot(sur[i], i, marker='o', ms=7, color=OI[0], mec='white', zorder=3)
ax.plot([], [], 'o', color=OI[0], label='surface escape speed'); ax.plot([], [], 'o', color=OI[1], label='centre escape speed')
ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=8); ax.set_xscale('log'); ax.set_xlim(3, 1e6)
ax.set_xlabel('largest accessible splitting δ$_{max}$ = μ w²/2 [keV]  (w² = v$_{esc}$² + u², u = 1/⟨1/v⟩)')
ax.set_title('Which bodies can up-scatter δ ≈ 300–380 keV dark matter (m$_χ$ = 1 TeV)?', fontsize=9.5)
ax.legend(frameon=False, loc='upper right', fontsize=8); ax.grid(axis='y', alpha=0)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P039_delta_max_by_body.png'), dpi=180); plt.close(fig)

# ============================================================================= RESULTS JSON
res = dict(
    halo=dict(inv_mean_v_kms=1 / inv_v, mean_v_kms=mean_v, v_rms_kms=v_rms, inv_mean_v_400_kms=1 / inv_v_400),
    ns=dict(M_Msun=1.5, R_km=12, B=B, v_esc_local_c=v_esc_loc / c, v_esc_newton_c=v_esc_newt / c, gamma=gam, R_inf_km=R_inf / 1e5, N_n=N_n, n_n=n_n,
            sigma_crit_n=sig_crit_n, sigma_crit_p=sig_crit_p, Y_p=Y_p, dE_max_GeV=dE_max, m_single_scatter_GeV=m_single, q_typ_GeV=q_typ,
            kinetic_efficiency_1_minus_sqrtB=1 - math.sqrt(B), delta_max_NR_MeV=dmax_NR / 1e3, delta_max_rel_MeV=dmax_rel / 1e3,
            heating_rows=ns_rows, reference_rho0p4=ref, saturation=sat_rows, chi2_fate=chi2_rows,
            rho_for_2000K_kinetic=rho_2000_kin, rho_for_2000K_with_annihilation=rho_2000_ann, bondi_Mdot_g_s=Mdot_bondi, bondi_L=L_bondi),
    nir=dict(JWST_F200W_10sig_10ks_nJy=JWST_F200W_10sig_10ks_nJy, rows=flux_rows),
    wd=dict(M_Msun=1.0, R_km=8000, v_esc_surface_kms=v_esc_wd_s / 1e5, centre_over_surface=ratio_c_s, v_esc_centre_kms=v_esc_wd_c / 1e5,
            N_C=N_C, sigma_crit_per_nucleus=sig_crit_wd_nuc, coh_iso=coh_iso, coh_p=coh_p, coh_Higgsino=coh_H, thresholds=wd_rows, heating=wd_heat, L_M4_faintest=L_M4_faint),
    sun=dict(mass_integral_Msun=M_int, v_esc_centre=float(vesc_sun[0]), v_esc_surface=float(vesc_sun[-1]), thresholds=thr_rows, own_gould=C_own, wimpyc=wimpyc_rows),
    delta_max_by_body=targets_dm, sigmas=SIG, P021_sigma_n=P021, recalled=RECALLED)
def _conv(o):
    if isinstance(o, (np.floating, np.integer)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)
json.dump(res, open(os.path.join(OUT, 'results.json'), 'w'), indent=1, default=_conv)
print(f'\nwrote {OUT}/results.json; {len(RECALLED)} recalled items')
