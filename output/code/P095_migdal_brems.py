"""
P095 -- Migdal electrons and bremsstrahlung photons accompanying a 248 keV xenon recoil.

Run from the simulation root:  .venv/bin/python output/code/P095_migdal_brems.py   (~5 s, analytic only)

Sections
  1. Kinematics of the recoiling atom (v_A, q_e = m_e v_A, adiabaticity per shell)
  2. Migdal probabilities per shell, two dipole-limit estimates
       A. closure sum rule  P_all = q_e^2 <z^2>_nl  (hydrogenic <r^2>, Slater or binding-energy Z_eff)
       B. photoabsorption / oscillator-strength route  dP/dE_e = q_e^2 (df/dw)/(2w)   [Essig et al. 2020 relation]
       plus an exact hydrogenic (Stobbe) K-shell integral as a check of the continuum share.
     Energy distribution of the ejected electron and total ER energy E_ER = E_e + E_B (hole cascade absorbed).
  3. Nuclear bremsstrahlung: classical dipole radiation of a suddenly kicked screened nucleus,
     dN/dw = (4 alpha/3 pi) (E_R/M c^2) Z_eff(w)^2 / w  (Kouvaris-Pradler form), P(w > 1 keV).
  4. Double counting vs Lindhard/NEST electronic stopping: energy budgets, cascade adiabaticity.
  5. Effect in (S1c, log10 S2c): NR(248 keV, P009 paper scale) + ER quanta (lz.nest_er_yields, Table S3);
     shift from the NR median in units of P024's sigma = 0.0313 dex; probability-weighted shift;
     fraction of 248 keV NRs pushed above +2 sigma / above the ROI ceiling; multiple-scatter question.
  6. Reverse: mechanisms that make an NR charge-poor (recombination, drift loss, position residual, g2).
Outputs: output/work/P095/*.csv, P095_results.json, run_log.txt, figures/P095_fig1_migdal_shift.png
"""
import sys, os, json, math
sys.path.insert(0, 'output/code')
import numpy as np
import pandas as pd
from scipy import integrate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import lzcommon as lz

OUT = 'output/work/P095'; FIG = f'{OUT}/figures'
os.makedirs(FIG, exist_ok=True)
LOG = open(f'{OUT}/run_log.txt', 'w')
def log(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.write(s + '\n')
R = {}

# ---------------------------------------------------------------- constants (recalled, certain unless noted)
ALPHA = 1 / 137.035999          # fine-structure constant
HARTREE_EV = 27.211386          # eV
A0_CM = 0.529177e-8             # Bohr radius, cm
ME_KEV = 510.99895              # m_e c^2, keV
AMU_KEV = 931494.10             # keV
M_XE_KEV = 131.293 * AMU_KEV    # mean Xe atomic mass (nucleus + electrons; electron mass negligible here)
E_R = 248.0                     # keV, LZ paper
g1, g2 = lz.LZ['g1'], lz.LZ['g2']
SIG_BAND = 0.0313               # dex, P024 baseline NR-band sd at S1c = 540 phd
SIG_BAND_AMBE = 0.036           # dex, P024 AmBe residual sd
EV_LOG_S2 = lz.LZ['ev_log10S2c']
ROI_TOP = lz.LZ['log10S2c_max']

# Xe subshell binding energies (eV) and occupancies -- recalled (X-ray data booklet values), reliability: likely (+-2 %)
SHELLS = {  # name: (E_B eV, N_e, n, l)
    'K 1s':  (34561., 2, 1, 0),
    'L1 2s': (5453., 2, 2, 0), 'L2 2p': (5107., 2, 2, 1), 'L3 2p': (4786., 4, 2, 1),
    'M1 3s': (1149., 2, 3, 0), 'M2 3p': (1002., 2, 3, 1), 'M3 3p': (941., 4, 3, 1),
    'M4 3d': (689., 4, 3, 2),  'M5 3d': (676., 6, 3, 2),
    'N1 4s': (213., 2, 4, 0),  'N2 4p': (147., 2, 4, 1),  'N3 4p': (145., 4, 4, 1),
    'N4 4d': (69.5, 4, 4, 2),  'N5 4d': (67.5, 6, 4, 2),
    'O1 5s': (23.4, 2, 5, 0),  'O2 5p': (13.4, 2, 5, 1),  'O3 5p': (12.1, 4, 5, 1),
}
GROUP = {k: k[0] for k in SHELLS}   # K, L, M, N, O
# Slater's-rules effective charges for Xe (Z = 54), computed here from the rules (recalled rules, certain):
# groups [1s] [2s2p] [3s3p] [3d] [4s4p] [4d] [5s5p]; same-group 0.35 (0.30 for 1s), n-1 shell 0.85 (s,p) or 1.00 (d), lower 1.00
SLATER_Z = {'1s': 54 - 0.30, '2s': 54 - (0.35 * 7 + 0.85 * 2), '2p': 54 - (0.35 * 7 + 0.85 * 2),
            '3s': 54 - (0.35 * 7 + 0.85 * 8 + 2), '3p': 54 - (0.35 * 7 + 0.85 * 8 + 2),
            '3d': 54 - (0.35 * 9 + 18), '4s': 54 - (0.35 * 7 + 0.85 * 18 + 10), '4p': 54 - (0.35 * 7 + 0.85 * 18 + 10),
            '4d': 54 - (0.35 * 9 + 36), '5s': 54 - (0.35 * 7 + 0.85 * 18 + 28), '5p': 54 - (0.35 * 7 + 0.85 * 18 + 28)}

log('=== P095: Migdal + bremsstrahlung for a 248 keV Xe recoil ===')
# ---------------------------------------------------------------- 1. kinematics
v_A = math.sqrt(2 * E_R / M_XE_KEV)             # in units of c
v_A_kms = v_A * lz.C_KMS
q_e_keV = ME_KEV * v_A                          # m_e v_A in keV/c
q_e_au = v_A / ALPHA                            # in atomic units (m_e = hbar = 1, v in units of alpha c)
R['kinematics'] = dict(E_R_keV=E_R, M_Xe_GeV=M_XE_KEV / 1e6, v_A_over_c=v_A, v_A_kms=v_A_kms, q_e_keV=q_e_keV, q_e_au=q_e_au,
                       q_e_sq_au=q_e_au ** 2)
log(f'v_A = {v_A:.3e} c = {v_A_kms:.0f} km/s;  q_e = m_e v_A = {q_e_keV*1e3:.0f} eV/c = {q_e_au:.3f} a.u.;  q_e^2 = {q_e_au**2:.4f} a.u.')
# collision-time / adiabaticity: DM kick duration ~ R_N / v_rel; cascade Xe-Xe kick ~ a_TF / v_A
R_N_fm = 1.2 * 131.29 ** (1 / 3) ; v_rel_kms = 500.0
tau_dm = R_N_fm * 1e-15 / (v_rel_kms * 1e3); hbar_eVs = 6.582e-16
w_dm_keV = hbar_eVs / tau_dm / 1e3
a_TF = 0.8853 * A0_CM / (2 * 54 ** 0.5) ** (2 / 3)
tau_xx = a_TF * 1e-2 / (v_A_kms * 1e3); w_xx_eV = hbar_eVs / tau_xx
R['adiabaticity'] = dict(R_N_fm=R_N_fm, v_rel_kms=v_rel_kms, tau_DM_kick_s=tau_dm, hbar_over_tau_DM_keV=w_dm_keV,
                         a_TF_angstrom=a_TF * 1e8, tau_XeXe_s=tau_xx, hbar_over_tau_XeXe_eV=w_xx_eV)
log(f'DM kick: R_N = {R_N_fm:.1f} fm, tau = {tau_dm:.1e} s, hbar/tau = {w_dm_keV:.0f} keV (sudden for L, M, N, O; marginal for K)')
log(f'Cascade Xe-Xe kick: a_TF = {a_TF*1e8:.3f} A, tau = {tau_xx:.1e} s, hbar/tau = {w_xx_eV:.0f} eV (adiabatic for M and deeper)')

# ---------------------------------------------------------------- 2. Migdal probabilities
def r2_hydrogenic(n, l, Z):            # <r^2> in a0^2 for hydrogenic orbital
    return n ** 2 * (5 * n ** 2 + 1 - 3 * l * (l + 1)) / (2 * Z ** 2)

# exact hydrogenic 1s: Stobbe photoionisation cross-section (a.u.), sigma(w) = (2^9 pi^2 alpha / 3) (I/w)^4 exp(-4 eta acot eta)/(1-exp(-2 pi eta))
def stobbe_sigma_H(w):                 # hydrogen 1s, w in Hartree, I = 0.5
    I = 0.5
    if w <= I: return 0.0
    eta = math.sqrt(I / (w - I))
    return (2 ** 9 * math.pi ** 2 * ALPHA / 3) * (I / w) ** 4 * math.exp(-4 * eta * math.atan(1 / eta)) / (1 - math.exp(-2 * math.pi * eta))
dfdw_H = lambda w: stobbe_sigma_H(w) / (2 * math.pi ** 2 * ALPHA)
f_cont_H, _ = integrate.quad(dfdw_H, 0.5, np.inf, limit=200)                      # continuum oscillator strength (expect 0.435)
z2_cont_H, _ = integrate.quad(lambda w: dfdw_H(w) / (2 * w), 0.5, np.inf, limit=200)  # continuum share of <z^2> = 1 a.u. for 1s
sig_thr_H_cm2 = stobbe_sigma_H(0.5 + 1e-9) * A0_CM ** 2
# effective spectral slope p of the 1s continuum (sigma ~ w^-p) between threshold and 3 I: fit
ws = np.geomspace(0.5001, 1.5, 50); p_eff = -np.polyfit(np.log(ws), np.log([stobbe_sigma_H(w) for w in ws]), 1)[0]
R['hydrogenic_1s_check'] = dict(sigma_threshold_cm2=sig_thr_H_cm2, f_continuum=f_cont_H, z2_continuum_share=z2_cont_H, p_eff_threshold_to_3I=p_eff)
log(f'Hydrogen 1s check: sigma(threshold) = {sig_thr_H_cm2:.2e} cm^2 (textbook 6.3e-18), continuum oscillator strength {f_cont_H:.3f} '
    f'(textbook 0.435), continuum share of <z^2>: {z2_cont_H:.3f}, effective slope p = {p_eff:.2f}')

P_SLOPE = 2.7        # spectral slope used for all shells (hydrogenic 1s value; recalled/derived, uncertain +-0.3 for d shells)
rows = []
for name, (EB_eV, N, n, l) in SHELLS.items():
    EB = EB_eV / HARTREE_EV
    orb = name.split()[1]
    Z_sl = SLATER_Z[orb]; Z_eb = n * math.sqrt(EB_eV / 13.6057)
    z2_sl = r2_hydrogenic(n, l, Z_sl) / 3; z2_eb = r2_hydrogenic(n, l, Z_eb) / 3
    qr_sl = q_e_au * math.sqrt(3 * z2_sl)      # dipole-validity parameter q_e r_rms
    # A. closure: total excitation+ionisation probability (all final states; Pauli blocking ignored) and ionisation share
    PA_all_sl = q_e_au ** 2 * N * z2_sl;  PA_ion_sl = PA_all_sl * z2_cont_H
    PA_all_eb = q_e_au ** 2 * N * z2_eb;  PA_ion_eb = PA_all_eb * z2_cont_H
    # saturating form 1 - exp(-q^2 <z^2>) per electron for the outer shells (sudden approx. beyond the dipole limit, Gaussian orbital proxy)
    P_sat_sl = N * (1 - math.exp(-q_e_au ** 2 * z2_sl))
    # B. oscillator-strength / photoabsorption route with power-law continuum: P_ion = q^2 N f_c (p-1)/(2 p E_B)
    PB_ion = q_e_au ** 2 * N * f_cont_H * (P_SLOPE - 1) / (2 * P_SLOPE * EB)
    mean_Ee_eV = EB_eV / (P_SLOPE - 1)         # mean ejected-electron energy for sigma ~ w^-p
    median_Ee_eV = EB_eV * (2 ** (1 / P_SLOPE) - 1)
    v_shell_au = math.sqrt(2 * EB)             # orbital speed scale
    rows.append(dict(shell=name, group=GROUP[name], E_B_eV=EB_eV, N_e=N, Z_eff_slater=Z_sl, Z_eff_binding=Z_eb,
                     r_rms_slater_a0=math.sqrt(3 * z2_sl), q_r_rms=qr_sl, q_over_v_shell=q_e_au / v_shell_au,
                     PA_all_slater=PA_all_sl, PA_ion_slater=PA_ion_sl, PA_all_binding=PA_all_eb, PA_ion_binding=PA_ion_eb,
                     P_saturating_slater=P_sat_sl, PB_ion=PB_ion, mean_E_e_eV=mean_Ee_eV, median_E_e_eV=median_Ee_eV,
                     mean_E_ER_eV=EB_eV + mean_Ee_eV))
df = pd.DataFrame(rows); df.to_csv(f'{OUT}/P095_migdal_shells.csv', index=False)
log('\nPer-subshell Migdal estimates (q_e^2 = %.4f a.u.):' % q_e_au ** 2)
log(df[['shell', 'E_B_eV', 'N_e', 'Z_eff_slater', 'q_r_rms', 'PA_ion_slater', 'PB_ion', 'mean_E_ER_eV']].to_string(index=False, float_format=lambda x: f'{x:.3g}'))
grp = df.groupby('group').agg(E_B_min_eV=('E_B_eV', 'min'), E_B_max_eV=('E_B_eV', 'max'), N_e=('N_e', 'sum'),
                              PA_ion_slater=('PA_ion_slater', 'sum'), PA_all_slater=('PA_all_slater', 'sum'), PA_ion_binding=('PA_ion_binding', 'sum'),
                              PB_ion=('PB_ion', 'sum'), P_sat=('P_saturating_slater', 'sum'), q_r_max=('q_r_rms', 'max'))
grp['mean_E_ER_eV'] = df.groupby('group').apply(lambda d: (d.PB_ion * d.mean_E_ER_eV).sum() / d.PB_ion.sum(), include_groups=False)
grp['P_ion_low'] = grp[['PA_ion_slater', 'PB_ion']].min(axis=1); grp['P_ion_high'] = grp[['PA_ion_slater', 'PB_ion']].max(axis=1)
grp['P_ion_geo'] = np.sqrt(grp.P_ion_low * grp.P_ion_high)
grp['dipole_valid'] = grp.q_r_max < 0.3
grp.to_csv(f'{OUT}/P095_migdal_groups.csv')
log('\nPer-shell (K..O) Migdal ionisation probability per 248 keV recoil:')
log(grp[['E_B_min_eV', 'E_B_max_eV', 'N_e', 'PA_ion_slater', 'PB_ion', 'P_ion_geo', 'P_sat', 'q_r_max', 'dipole_valid', 'mean_E_ER_eV']].to_string(float_format=lambda x: f'{x:.3g}'))
R['migdal_groups'] = json.loads(grp.to_json(orient='index'))
# K-shell exact hydrogenic check: P_K = q^2 * 2 electrons * <z^2>_1s(Z) * continuum share, <z^2>_1s = 1/Z^2 (a.u.)
Z_K = SLATER_Z['1s']; P_K_exact = q_e_au ** 2 * 2 * (1 / Z_K ** 2) * z2_cont_H
R['P_K_exact_hydrogenic'] = P_K_exact
log(f'K shell, exact hydrogenic (Stobbe) with Z = {Z_K}: P_K = {P_K_exact:.2e};  route B gives {grp.loc["K","PB_ion"]:.2e}')
# ejected-electron spectrum for the M shell: fraction with E_ER above thresholds
def frac_Ee_above(EB_eV, x_eV): return (EB_eV / (EB_eV + x_eV)) ** P_SLOPE
M = df[df.group == 'M']
for thr_keV in [0.9, 1.5, 2.5, 3.5]:
    f = sum(r.PB_ion * frac_Ee_above(r.E_B_eV, max(thr_keV * 1e3 - r.E_B_eV, 0)) for r in M.itertuples()) / M.PB_ion.sum()
    R[f'M_frac_E_ER_above_{thr_keV}keV'] = f
    log(f'  M-shell events with E_ER = E_e + E_B > {thr_keV} keV: {f:.3f}')

# ---------------------------------------------------------------- 3. bremsstrahlung
# radiating charge at photon energy w: electrons with E_B < w cannot follow the nucleus -> Z_eff(w) = N(E_B < w)
EB_arr = np.array([v[0] for v in SHELLS.values()]); N_arr = np.array([v[1] for v in SHELLS.values()])
def Z_eff(w_eV): return float(N_arr[EB_arr < w_eV].sum())
pref = 4 * ALPHA / (3 * math.pi) * (E_R / M_XE_KEV)      # = (2 alpha/3 pi) (v_A/c)^2
def dNdw(w_eV): return pref * Z_eff(w_eV) ** 2 / w_eV      # per eV
brems = {}
for wmin_keV in [0.1, 1.0, 3.5, 10.0]:
    for wmax_keV in [100.0, 248.0]:
        edges = np.unique(np.concatenate([[wmin_keV * 1e3, wmax_keV * 1e3], EB_arr[(EB_arr > wmin_keV * 1e3) & (EB_arr < wmax_keV * 1e3)]]))
        P = sum(pref * Z_eff(0.5 * (a + b)) ** 2 * math.log(b / a) for a, b in zip(edges[:-1], edges[1:]))
        brems[f'P(w>{wmin_keV}keV, wmax={wmax_keV:.0f}keV)'] = P
P_brems_1keV = brems['P(w>1.0keV, wmax=100keV)']
# unscreened bare nucleus for comparison, and Z_eff -> 0 limit statement
P_bare = pref * 54 ** 2 * math.log(100 / 1)
R['brems'] = dict(prefactor_2alpha_3pi_v2=pref, Z_eff_at_1keV=Z_eff(1e3), Z_eff_at_6keV=Z_eff(6e3), Z_eff_at_40keV=Z_eff(4e4),
                  probabilities=brems, P_bare_Z54_1_to_100keV=P_bare)
log(f'\nBremsstrahlung: prefactor (2 alpha/3 pi)(v_A/c)^2 = {pref:.2e}; Z_eff(1 keV) = {Z_eff(1e3):.0f}, Z_eff(6 keV) = {Z_eff(6e3):.0f}, Z_eff(>34.6 keV) = 54')
for k, v in brems.items(): log(f'  {k}: {v:.2e}')
log(f'  bare Z = 54, 1-100 keV: {P_bare:.2e};  Migdal M-shell / brems(>1 keV) = {grp.loc["M","P_ion_geo"]/P_brems_1keV:.0f}')

# ---------------------------------------------------------------- 4. double-counting / energy budget
nph_nr, ne_nr = lz.nest_nr_yields(E_R)
s_paper = lz.LZ['ev_S1c'] / (g1 * nph_nr)              # P009/P024 paper-scale photon rescaling so that S1c(248) = 540.1
nph_nr_p = nph_nr * s_paper
Nq = nph_nr_p + ne_nr; E_vis_keV = Nq * lz.W_EV * 1e-3
L_lind = 0.334                                          # P064, Lindhard k = 0.166 at 248 keV
E_el_lind = L_lind * E_R
E_mig_outer = sum(grp.loc[g, 'PB_ion'] * grp.loc[g, 'mean_E_ER_eV'] for g in ['N', 'O']) / 1e3       # keV, mean per recoil
E_mig_inner = sum(grp.loc[g, 'P_ion_geo'] * grp.loc[g, 'mean_E_ER_eV'] for g in ['K', 'L', 'M']) / 1e3
E_brems = sum(pref * Z_eff(0.5 * (a + b)) ** 2 * (b - a) for a, b in zip(*(lambda e: (e[:-1], e[1:]))(np.unique(np.concatenate([[1e3, 1e5], EB_arr[(EB_arr > 1e3) & (EB_arr < 1e5)]]))))) / 1e3
R['energy_budget'] = dict(Nph_NR_tableS5=nph_nr, Ne_NR=ne_nr, paper_scale_s=s_paper, Nph_NR_paper=nph_nr_p, Nq=Nq, E_visible_keV=E_vis_keV,
                          L_lindhard_248=L_lind, E_electronic_lindhard_keV=E_el_lind, E_migdal_outer_mean_keV=E_mig_outer,
                          E_migdal_inner_mean_keV=E_mig_inner, E_brems_mean_keV_1_100=E_brems)
log(f'\nEnergy budget: NEST quanta N_q = {Nq:.0f} -> {E_vis_keV:.1f} keV visible; Lindhard L = {L_lind} -> {E_el_lind:.1f} keV electronic.')
log(f'  Mean Migdal energy per recoil: outer (N,O) {E_mig_outer*1e3:.1f} eV, inner (K,L,M) {E_mig_inner*1e3:.1f} eV; brems (1-100 keV) {E_brems*1e3:.2f} eV'
    f'  -> {(E_mig_outer+E_mig_inner+E_brems)/E_el_lind*1e4:.1f} x 1e-4 of the electronic energy')

# ---------------------------------------------------------------- 5. effect in (S1c, log10 S2c)
E_grid = np.arange(150, 340, 1.0)
nr = np.array([lz.nest_nr_yields(E) for E in E_grid]); NPH = nr[:, 0] * s_paper; NE = nr[:, 1]
S1_NR = g1 * NPH; L2_NR = np.log10(g2 * NE)
med_at = lambda s1: float(np.interp(s1, S1_NR, L2_NR))
med_540 = med_at(lz.LZ['ev_S1c']); slope = np.gradient(L2_NR, np.log10(S1_NR))[np.argmin(abs(E_grid - 248))]
event_nsig = (EV_LOG_S2 - med_540) / SIG_BAND
roi_nsig = (ROI_TOP - med_540) / SIG_BAND
log(f'\nNR median at S1c = 540.1 (paper scale, analytic means): {med_540:.4f} (P024 MC 4.0152); slope dlogS2/dlogS1 = {slope:.3f}; '
    f'event {event_nsig:+.2f} sigma; ROI ceiling 4.15 = {roi_nsig:+.2f} sigma')
def hybrid(E_ER, E_NR=E_R):
    nph, ne = lz.nest_nr_yields(E_NR); nph *= s_paper
    if E_ER > 0:
        nphe, nee = lz.nest_er_yields(E_ER, params=lz.NEST_ER_LZ)
    else: nphe = nee = 0.0
    s1 = g1 * (nph + nphe); l2 = math.log10(g2 * (ne + nee))
    return s1, l2, l2 - med_at(s1), nphe, nee
shift_rows = []
for E_ER, label in [(0.1, 'N/O-shell scale'), (0.5, ''), (0.7, 'M4/5 hole, E_e~0'), (1.0, 'M-shell typical'), (1.5, ''), (2.0, ''), (3.5, 'ROI-ceiling crossing'),
                    (5.0, 'L-shell typical'), (10.0, ''), (35.0, 'K-shell'), (39.58, 'P036 129Xe* check')]:
    s1, l2, d, nphe, nee = hybrid(E_ER)
    # S1-matched variant (P036 method): reduce E_NR so that S1c = 540.1
    f = lambda E: hybrid(E_ER, E)[0] - lz.LZ['ev_S1c']
    from scipy.optimize import brentq
    E_m = brentq(f, 100, 300); d_m = hybrid(E_ER, E_m)[2]
    shift_rows.append(dict(E_ER_keV=E_ER, note=label, Nph_ER=nphe, Ne_ER=nee, S1c=s1, log10S2c=l2, shift_dex=d, shift_sigma=d / SIG_BAND,
                           shift_sigma_ambe=d / SIG_BAND_AMBE, E_NR_S1matched_keV=E_m, shift_sigma_S1matched=d_m / SIG_BAND,
                           above_2sigma=d / SIG_BAND > 2, above_ROI=l2 > ROI_TOP))
sh = pd.DataFrame(shift_rows); sh.to_csv(f'{OUT}/P095_shift_table.csv', index=False)
log('\nNR(248 keV) + ER(E_ER) at one vertex: shift from the NR median at the hybrid S1c')
log(sh[['E_ER_keV', 'Nph_ER', 'Ne_ER', 'S1c', 'log10S2c', 'shift_dex', 'shift_sigma', 'shift_sigma_S1matched', 'above_2sigma', 'above_ROI']].to_string(index=False, float_format=lambda x: f'{x:.3f}'))
E_ceiling = brentq(lambda E: hybrid(E)[1] - ROI_TOP, 0.5, 20); E_2sig = brentq(lambda E: hybrid(E)[2] - 2 * SIG_BAND, 0.05, 20)
R['shift'] = dict(median_540=med_540, slope=slope, event_nsig=event_nsig, roi_ceiling_nsig=roi_nsig, E_ER_for_plus2sigma_keV=E_2sig,
                  E_ER_for_ROI_ceiling_keV=E_ceiling, table=shift_rows)
log(f'E_ER giving +2 sigma: {E_2sig:.2f} keV; E_ER lifting the event above the ROI ceiling: {E_ceiling:.2f} keV')
# probability-weighted expected shift and out-of-band fractions (Migdal + brems), using the mean E_ER of each shell
exp_shift = 0.0; frac_2sig = 0.0; frac_roi = 0.0; contrib = {}
for gname in ['K', 'L', 'M', 'N', 'O']:
    P = grp.loc[gname, 'P_ion_geo']; E_ER_mean = grp.loc[gname, 'mean_E_ER_eV'] / 1e3
    d_sig = hybrid(E_ER_mean)[2] / SIG_BAND if E_ER_mean > 0.05 else hybrid(0.1)[2] / SIG_BAND * E_ER_mean / 0.1
    exp_shift += P * d_sig; contrib[gname] = dict(P=P, mean_E_ER_keV=E_ER_mean, shift_sigma=d_sig, P_times_shift=P * d_sig)
    # fraction above thresholds: shell electrons with E_e + E_B beyond the threshold energies
    sub = df[df.group == gname]
    for r in sub.itertuples():
        w = r.PB_ion / sub.PB_ion.sum() * P
        frac_2sig += w * frac_Ee_above(r.E_B_eV, max(E_2sig * 1e3 - r.E_B_eV, 0))
        frac_roi += w * frac_Ee_above(r.E_B_eV, max(E_ceiling * 1e3 - r.E_B_eV, 0))
P_br_2sig = sum(pref * Z_eff(0.5 * (a + b)) ** 2 * math.log(b / a) for a, b in
        zip(*(lambda e: (e[:-1], e[1:]))(np.unique(np.concatenate([[E_2sig * 1e3, 1e5], EB_arr[(EB_arr > E_2sig * 1e3) & (EB_arr < 1e5)]])))))
P_br_roi = sum(pref * Z_eff(0.5 * (a + b)) ** 2 * math.log(b / a) for a, b in
        zip(*(lambda e: (e[:-1], e[1:]))(np.unique(np.concatenate([[E_ceiling * 1e3, 1e5], EB_arr[(EB_arr > E_ceiling * 1e3) & (EB_arr < 1e5)]])))))
from scipy.stats import norm
R['expected'] = dict(contributions=contrib, expected_shift_sigma=exp_shift, expected_shift_inner_only_sigma=sum(contrib[g]['P_times_shift'] for g in 'KLM'),
                     frac_above_2sigma_migdal=frac_2sig, frac_above_2sigma_brems=P_br_2sig, frac_above_ROI_migdal=frac_roi, frac_above_ROI_brems=P_br_roi,
                     gaussian_tail_above_2sigma=norm.sf(2), gaussian_tail_above_ROI=norm.sf(roi_nsig), P_below_event_gauss=norm.cdf(event_nsig))
log('\nProbability-weighted shift (sigma units): ' + ', '.join(f'{g}: {c["P"]:.2e} x {c["shift_sigma"]:+.2f} = {c["P_times_shift"]:+.4f}' for g, c in contrib.items()))
log(f'  total expected shift {exp_shift:+.3f} sigma (inner shells only {R["expected"]["expected_shift_inner_only_sigma"]:+.4f}); the median shift is 0; sign is UP')
log(f'  Fraction of 248 keV NRs lifted above +2 sigma: Migdal {frac_2sig:.2e}, brems {P_br_2sig:.2e} (Gaussian band tail alone {norm.sf(2):.3f});'
    f' above the ROI ceiling (+{roi_nsig:.1f} sigma): Migdal {frac_roi:.2e}, brems {P_br_roi:.2e} (Gaussian {norm.sf(roi_nsig):.1e})')
# multiple scatter? electron CSDA range and X-ray mean free path in LXe (recalled scalings, uncertain)
rho = 2.9
def e_range_um(E_keV): return 6e-6 * E_keV ** 1.7 / rho * 1e4       # g/cm^2 -> um  (ESTAR-like power law scaled to Xe; recalled, uncertain x2)
mu_rho = {29.7: 7.5, 4.1: 1000.0, 1.0: 6000.0}                      # cm^2/g photoelectric in Xe (recalled, uncertain x1.5)
ms = dict(range_1keV_um=e_range_um(1), range_5keV_um=e_range_um(5), range_35keV_um=e_range_um(35),
          mfp_Kalpha_29p7keV_mm=1 / (mu_rho[29.7] * rho) * 10, mfp_Lalpha_4p1keV_um=1 / (mu_rho[4.1] * rho) * 1e4, mfp_1keV_um=1 / (mu_rho[1.0] * rho) * 1e4,
          diffusion_sigma_T_mm_P041=3.1, drift_speed_mm_per_us=1.5)
R['multiple_scatter'] = ms
log(f"\nRanges in LXe: e- 1/5/35 keV: {ms['range_1keV_um']:.2f}/{ms['range_5keV_um']:.1f}/{ms['range_35keV_um']:.0f} um; "
    f"K-alpha 29.7 keV mfp {ms['mfp_Kalpha_29p7keV_mm']:.2f} mm; L-alpha 4.1 keV {ms['mfp_Lalpha_4p1keV_um']:.0f} um  << S2 diffusion sigma 3.1 mm -> one merged S2")

# ---------------------------------------------------------------- 6. reverse: charge-poor mechanisms
deficit_dex = med_540 - EV_LOG_S2; deficit_frac = 1 - 10 ** (-deficit_dex)
Ne_obs = lz.LZ['ev_S2c'] / g2
t_drift_us = 859.0                       # P041 (event drift time)
def tau_true_for_loss(tau_assumed_ms, loss):   # exp(-t (1/tau' - 1/tau)) = 1 - loss
    return 1 / (1 / tau_assumed_ms - math.log(1 - loss) / (t_drift_us / 1e3))
E_NO_mean = sum(grp.loc[g, 'P_ion_geo'] * grp.loc[g, 'mean_E_ER_eV'] for g in 'NO') / sum(grp.loc[g, 'P_ion_geo'] for g in 'NO') / 1e3
mech = [
    dict(mechanism='recombination + electron-counting fluctuation (band itself)', probability=norm.cdf(event_nsig), extra_ER_keV=0.0, shift_sigma=event_nsig,
         note='P024: 5.7 % from MC; binomial floor 0.023 dex; 4/31 AmBe points at/below the event'),
    dict(mechanism='S2 position-correction residual alone (2 % rms, P024 term 0.0087 dex)', probability=norm.sf(deficit_dex / 0.0087), extra_ER_keV=0.0,
         shift_sigma=event_nsig, note=f'needs a {deficit_dex/0.0087:.1f} sigma excursion of the residual'),
    dict(mechanism=f'electron-lifetime mis-correction at {t_drift_us:.0f} us ({deficit_frac*100:.0f} % charge loss)', probability=float('nan'), extra_ER_keV=0.0, shift_sigma=event_nsig,
         note=f'tau_true = {tau_true_for_loss(5, deficit_frac):.1f} ms if 5 ms assumed ({tau_true_for_loss(8, deficit_frac):.1f} if 8 ms); excluded by Kr-83m monitoring/P041 witnesses at the 78 % level, untested at 10 %'),
    dict(mechanism='g2 3 % low (systematic, not per event)', probability=float('nan'), extra_ER_keV=0.0, shift_sigma=-0.44, note='P043: g2 +-3 % moves the event by 0.44 sigma; P024: 1.10-1.99 sigma'),
    dict(mechanism='Migdal M-shell hole (this work)', probability=grp.loc['M', 'P_ion_geo'], extra_ER_keV=grp.loc['M', 'mean_E_ER_eV'] / 1e3, shift_sigma=contrib['M']['shift_sigma'], note='wrong sign: charge-rich'),
    dict(mechanism='Migdal L-shell hole (this work)', probability=grp.loc['L', 'P_ion_geo'], extra_ER_keV=grp.loc['L', 'mean_E_ER_eV'] / 1e3, shift_sigma=contrib['L']['shift_sigma'], note='wrong sign; above ROI ceiling'),
    dict(mechanism='Migdal K-shell hole (this work)', probability=grp.loc['K', 'P_ion_geo'], extra_ER_keV=grp.loc['K', 'mean_E_ER_eV'] / 1e3, shift_sigma=contrib['K']['shift_sigma'], note='wrong sign; above ROI ceiling'),
    dict(mechanism='Migdal N+O shells (in electronic stopping already)', probability=grp.loc['N', 'P_ion_geo'] + grp.loc['O', 'P_ion_geo'],
         extra_ER_keV=E_NO_mean, shift_sigma=hybrid(0.1)[2] / SIG_BAND * E_NO_mean / 0.1, note='dipole limit marginal (q r ~ 0.4-1.3); ~10 eV mean, 1e-4 of the electronic budget'),
    dict(mechanism='nuclear bremsstrahlung photon > 1 keV (this work)', probability=P_brems_1keV, extra_ER_keV=1.0, shift_sigma=hybrid(1.0)[2] / SIG_BAND, note='wrong sign; Z_eff = 40-54 above 1 keV'),
]
mech_df = pd.DataFrame(mech); mech_df.to_csv(f'{OUT}/P095_mechanisms.csv', index=False)
R['reverse'] = dict(deficit_dex=deficit_dex, deficit_fraction=deficit_frac, Ne_observed=Ne_obs, Ne_NR_mean=ne_nr, t_drift_us=t_drift_us,
                    tau_true_if_5ms=tau_true_for_loss(5, deficit_frac), tau_true_if_8ms=tau_true_for_loss(8, deficit_frac))
log(f'\nReverse: event is {deficit_dex:.4f} dex = {deficit_frac*100:.1f} % charge-poor ({Ne_obs:.0f} vs {ne_nr:.0f} electrons).')
log(mech_df[['mechanism', 'probability', 'extra_ER_keV', 'shift_sigma']].to_string(index=False, float_format=lambda x: f'{x:.3g}'))

# ---------------------------------------------------------------- figure
C1, C2, C3, INK, INK2 = '#2a78d6', '#eb6834', '#1baf7a', '#0b0b0b', '#52514e'
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor='#fcfcfb')
order = ['K', 'L', 'M', 'N', 'O']; x = np.arange(len(order)); w = 0.36
ax1.bar(x - w / 2, [grp.loc[g, 'PA_ion_slater'] for g in order], w, color=C1, label='A: closure sum rule (Slater Z_eff)', linewidth=0)
ax1.bar(x + w / 2, [grp.loc[g, 'PB_ion'] for g in order], w, color=C2, label='B: oscillator-strength route', linewidth=0)
ax1.axhline(P_brems_1keV, color=C3, lw=2, ls='--'); ax1.text(4.45, P_brems_1keV * 1.3, 'bremsstrahlung > 1 keV', ha='right', va='bottom', fontsize=9, color=INK2)
for i, g in enumerate(order):
    ax1.text(i, max(grp.loc[g, 'PA_ion_slater'], grp.loc[g, 'PB_ion']) * 1.6, f"{grp.loc[g,'mean_E_ER_eV']/1e3:.2g} keV", ha='center', fontsize=8.5, color=INK2)
ax1.set_yscale('log'); ax1.set_ylim(3e-6, 60); ax1.set_xticks(x); ax1.set_xticklabels([f'{g} shell\n({int(grp.loc[g,"N_e"])} e)' for g in order])
ax1.set_ylabel('ionisation probability per 248 keV recoil'); ax1.set_title('Migdal ionisation, dipole limit (labels: mean E_ER)', fontsize=10, color=INK)
ax1.axvspan(2.5, 4.5, color='#f0efec', zorder=0); ax1.text(3.5, 12, 'q r ~ 0.4-1.3: dipole limit fails;\npart of the electronic stopping', ha='center', fontsize=8.5, color=INK2)
ax1.legend(frameon=False, fontsize=8.5, loc='upper left', bbox_to_anchor=(0.0, 1.0))
E_curve = np.geomspace(0.1, 100, 140); d_curve = np.array([hybrid(E)[2] / SIG_BAND for E in E_curve])
ax2.plot(E_curve, d_curve, color=C1, lw=2)
for E_ER, lab, side in [(grp.loc['M', 'mean_E_ER_eV'] / 1e3, 'M', 1), (grp.loc['L', 'mean_E_ER_eV'] / 1e3, 'L', 1), (grp.loc['K', 'mean_E_ER_eV'] / 1e3, 'K', -1)]:
    d = hybrid(E_ER)[2] / SIG_BAND; ax2.plot(E_ER, d, 'o', color=C2, ms=8)
    ax2.text(E_ER * (1.15 if side > 0 else 0.87), d, f'{lab}-shell hole', va='center', ha='left' if side > 0 else 'right', fontsize=9, color=INK2)
ax2.axhline(2, color=INK2, lw=1, ls=':'); ax2.text(0.11, 2.2, '+2 sigma', fontsize=8.5, color=INK2)
ax2.axhline(roi_nsig, color=INK2, lw=1, ls='--'); ax2.text(0.11, roi_nsig + 0.25, f'ROI ceiling log10 S2c = 4.15 (+{roi_nsig:.1f} sigma)', fontsize=8.5, color=INK2)
ax2.axhline(event_nsig, color=C3, lw=2); ax2.text(0.11, event_nsig - 0.9, f'LZ event: {event_nsig:+.2f} sigma (charge-poor)', fontsize=9, color=C3)
ax2.axhline(0, color=INK2, lw=0.8); ax2.set_xscale('log'); ax2.set_xlabel('extra electron-recoil energy E_ER at the vertex [keV]')
ax2.set_ylabel('shift from NR median [band sigma = 0.0313 dex]'); ax2.set_ylim(-3, 21); ax2.set_xlim(0.1, 100)
ax2.set_title('NR(248 keV) + ER(E_ER): always upward', fontsize=10, color=INK)
for ax in (ax1, ax2):
    ax.set_facecolor('#fcfcfb'); ax.grid(axis='y', color='#e6e5e1', lw=0.6); ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); fig.savefig(f'{FIG}/P095_fig1_migdal_shift.png', dpi=160); plt.close(fig)
log(f'\nFigure written: {FIG}/P095_fig1_migdal_shift.png')

json.dump(R, open(f'{OUT}/P095_results.json', 'w'), indent=1, default=float)
log('Results written to P095_results.json'); LOG.close()
