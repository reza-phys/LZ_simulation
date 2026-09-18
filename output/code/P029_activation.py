"""
P029 -- Activation products eight days after AmBe: activities on 16 June 2023 and which
decays could produce an MSSI-like topology.

Run from the simulation root:
    .venv/bin/python output/code/P029_activation.py

Outputs -> output/work/P029/  (CSV/JSON tables) and output/work/P029/figures/ (PNG).

Sections
  1. Decay data (radioactivedecay, ICRP-107) + recalled emission scheme (flagged).
  2. Production & Bateman: captures per isotope ~ abundance x sigma (thermal set, and a
     fast-capture set as a bracket), 1-day irradiation ending on 8 June 12:00 UTC,
     decay to the event time (16 June 21:22 UTC, +8.39 d).  Absolute scale from LZ Table I:
     125I = 8.9 +- 2.7 ROI events in 220 d, through a NEST-LZ estimate of the 125I ROI
     acceptance (P010 charge-suppression f = 0.2, k sigma_p width).  Cross-check with
     127Xe + 125Xe = 1.5 +- 0.3.
  3. Gamma transport in LXe: recalled XCOM-like attenuation (iodine proxy, K-edge 34.56 keV)
     vs the longer 'assignment' bracket; P(travel >= 20 cm, >= 26.4/26.9 cm); Klein-Nishina
     fraction for a 12 +- 2 keV Compton deposit; reverse topology (decay in dead region,
     gamma into FV, 12 keV Compton, escape).
  4. Correlated MSSI: local (atomic/IC) deposit energies vs the 12 +- 2 keV vertex needed.
  5. Random coincidence: activation S1-only in RFR/wall layer x 10-14 keV ER in FV x dt.
  6. What activation does produce: ER lines, 133Xe at S1c > 500 phd, 127Xe RFR-MSSI near cathode.
"""
from __future__ import annotations
import sys, os, json, math
import numpy as np
import pandas as pd
sys.path.insert(0, 'output/code')
from common import lzcommon as lz
import radioactivedecay as rd
import periodictable as pt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import integrate, stats

OUT = 'output/work/P029'
FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)
RES = {}          # master results dict -> P029_results.json

DAY = 86400.0
LN2 = math.log(2)

# ----------------------------------------------------------------------------------------
# 0. Scenario parameters
# ----------------------------------------------------------------------------------------
T_IRR_D = 1.0            # assumed AmBe deployment length (d)            [assumption]
T_EV_D = 8.39            # 8 June 12:00 UTC -> 16 June 21:22:39 UTC       [paper dates; midday assumed]
T_RESUME_D = 1.0         # WS data resume after the end of the AmBe (d) [assumption; varied 0.5-2]
N_CAL = 3                # neutron calibrations in the run ("one of three") [paper]
M_LXE_T = 10.0           # total circulated LXe inventory (t)             [recalled: likely]
M_ACTIVE_T = 7.0         # active LXe (paper)
M_FV_T = lz.LZ['fiducial_mass_t']
R_TPC_CM = 72.8          # active radius (recalled, likely; P004/P013)
RFR_DEPTH_CM = 13.75     # paper
M_RFR_T = math.pi * R_TPC_CM**2 * RFR_DEPTH_CM * 2.9 / 1e6
M_WALL_T = lz.LZ['charge_dead_fraction'] * M_ACTIVE_T   # 0.3 % of active (paper)
RHO_LXE = 2.9
RES['scenario'] = dict(T_irr_d=T_IRR_D, t_event_after_irr_d=T_EV_D, t_resume_d=T_RESUME_D, N_cal=N_CAL,
                       M_LXe_t=M_LXE_T, M_active_t=M_ACTIVE_T, M_FV_t=M_FV_T, M_RFR_t=M_RFR_T, M_wall_t=M_WALL_T)

# ----------------------------------------------------------------------------------------
# 1. Decay data from radioactivedecay + recalled emission scheme
# ----------------------------------------------------------------------------------------
NUCS = ['Xe-125', 'I-125', 'Xe-127', 'Xe-129m', 'Xe-131m', 'Xe-133', 'Xe-133m', 'Xe-135', 'Xe-135m',
        'Xe-137', 'Cs-137', 'Kr-83m']
rows = []
for n in NUCS:
    nu = rd.Nuclide(n)
    rows.append(dict(nuclide=n, half_life_d=nu.half_life('d'), half_life_readable=nu.half_life('readable'),
                     decay_modes='; '.join(nu.decay_modes()),
                     branching='; '.join(f'{b:.4g}' for b in nu.branching_fractions()),
                     progeny='; '.join(nu.progeny())))
decay_df = pd.DataFrame(rows)
decay_df.to_csv(os.path.join(OUT, 'decay_data_icrp107.csv'), index=False)
HL = {r['nuclide']: r['half_life_d'] for r in rows}
LAM = {k: LN2 / (v * DAY) for k, v in HL.items()}   # 1/s

# Recalled emission data (ENSDF-like; reliability 'likely' unless noted).
# local = deposits confined to << 1 cm of the decay (atomic relaxation after EC, IC electrons + X-rays,
#         beta continuum);  gammas = (E_keV, intensity per decay).
EMIS = {
 'Xe-125': dict(mode='EC (beta+ 0.3%)', local=[(33.2, 0.80, 'I K-shell vacancy (K capture)'),
                                              (4.9, 0.17, 'I L-shell vacancy'), (55.0, 0.068, '55.0 keV transition, mostly IC')],
                gammas=[(188.4, 0.54), (243.4, 0.30), (55.0, 0.068)], beta_endpoint=None),
 'I-125':  dict(mode='EC to 35.5 keV level of 125Te', local=[(67.3, 0.80, 'K capture + 35.5 keV (IC 93%)'),
                                              (40.4, 0.17, 'L capture + 35.5'), (36.5, 0.03, 'M capture + 35.5')],
                gammas=[(35.5, 0.067)], beta_endpoint=None),
 'Xe-127': dict(mode='EC', local=[(33.2, 0.83, 'I K-shell vacancy'), (4.9, 0.14, 'I L-shell vacancy'),
                                  (57.6, 0.05, '57.6 keV transition, mostly IC')],
                gammas=[(202.9, 0.687), (172.1, 0.257), (375.0, 0.173), (145.3, 0.043), (57.6, 0.012)], beta_endpoint=None),
 'Xe-129m': dict(mode='IT 236.1 keV (196.6 + 39.6 cascade)', local=[(236.1, 0.88, 'both steps IC'),
                                  (196.6, 0.075, '39.6 keV gamma escapes (mm)'), (39.6, 0.046, '196.6 keV gamma escapes')],
                gammas=[(196.6, 0.046), (39.6, 0.075)], beta_endpoint=None),
 'Xe-131m': dict(mode='IT 163.9 keV', local=[(163.9, 0.98, 'IC electron + Xe X-rays/Auger')],
                gammas=[(163.9, 0.0195)], beta_endpoint=None),
 'Xe-133': dict(mode='beta- to 81.0 keV level (99%)', local=[('beta', 1.0, 'beta continuum, endpoint 346 keV'),
                                  (81.0, 0.62, 'IC of 81 keV transition (+Cs K X-rays)')],
                gammas=[(81.0, 0.38)], beta_endpoint=346.0),
 'Xe-133m': dict(mode='IT 233.2 keV', local=[(233.2, 0.90, 'IC')], gammas=[(233.2, 0.10)], beta_endpoint=None),
 'Xe-135': dict(mode='beta- to 249.8 keV level (96%)', local=[('beta', 1.0, 'beta continuum, endpoint 915 keV')],
                gammas=[(249.8, 0.90), (608.2, 0.029)], beta_endpoint=915.0),
 'Xe-135m': dict(mode='IT 526.6 keV (99.4%)', local=[(526.6, 0.19, 'IC')], gammas=[(526.6, 0.81)], beta_endpoint=None),
 'Xe-137': dict(mode='beta- (Q = 4.17 MeV)', local=[('beta', 1.0, 'beta continuum, endpoint 4.17 MeV')],
                gammas=[(455.5, 0.31)], beta_endpoint=4170.0),
 'Cs-137': dict(mode='beta- (94% to 137mBa)', local=[('beta', 1.0, 'endpoint 514 keV')], gammas=[(661.7, 0.85)], beta_endpoint=514.0),
 'Kr-83m': dict(mode='IT 32.1 + 9.4 keV (154 ns)', local=[(41.5, 1.0, 'two IC steps at the same site')], gammas=[], beta_endpoint=None),
}
em_rows = []
for n, d in EMIS.items():
    for (E, p, desc) in d['local']:
        em_rows.append(dict(nuclide=n, kind='local', energy_keV=E, prob=p, note=desc))
    for (E, p) in d['gammas']:
        em_rows.append(dict(nuclide=n, kind='gamma', energy_keV=E, prob=p, note='gamma escaping the site'))
pd.DataFrame(em_rows).to_csv(os.path.join(OUT, 'emission_scheme_recalled.csv'), index=False)

# ----------------------------------------------------------------------------------------
# 2. Production and decay
# ----------------------------------------------------------------------------------------
ABUND = {A: pt.Xe[A].abundance / 100.0 for A in [124, 126, 128, 129, 130, 131, 132, 134, 136]}
# thermal (n,gamma) cross-sections in barn -> product   [recalled, uncertain +-30%; isomer partials uncertain x2]
SIG_TH = {'Xe-125': (124, 165.0), 'Xe-127': (126, 3.5), 'Xe-129m': (128, 0.48), 'Xe-131m': (130, 0.45),
          'Xe-133': (132, 0.40), 'Xe-133m': (132, 0.05), 'Xe-135': (134, 0.26), 'Xe-135m': (134, 0.003),
          'Xe-137': (136, 0.26)}
SIG_TH_TOTAL = {124: 165.0, 126: 3.5, 128: 8.0, 129: 21.0, 130: 26.0, 131: 85.0, 132: 0.45, 134: 0.265, 136: 0.26}
# fast (~1 MeV) capture set in mb, with isomer shares                       [recalled, uncertain x2-3]
SIG_FAST = {'Xe-125': (124, 50.0), 'Xe-127': (126, 40.0), 'Xe-129m': (128, 30.0 * 0.4), 'Xe-131m': (130, 25.0 * 0.5),
            'Xe-133': (132, 15.0 * 0.9), 'Xe-133m': (132, 15.0 * 0.1), 'Xe-135': (134, 8.0 * 0.9), 'Xe-135m': (134, 8.0 * 0.1),
            'Xe-137': (136, 1.5)}
sig_nat_th = sum(ABUND[A] * s for A, s in SIG_TH_TOTAL.items())
RES['sigma_thermal_natural_Xe_b'] = sig_nat_th   # sanity: ~ 24 b recalled

def rel_captures(SIG):
    w = {n: ABUND[A] * s for n, (A, s) in SIG.items()}
    ref = w['Xe-125']
    return {n: v / ref for n, v in w.items()}

REL = {'thermal': rel_captures(SIG_TH), 'fast': rel_captures(SIG_FAST)}
# 'intermediate' set (added after the 127Xe Table-I check below): geometric interpolation between the two
# brackets with the weight w fixed so that the 127Xe/125Xe production ratio equals the Table-I-implied value.
def make_intermediate(w):
    return {n: REL['thermal'][n] ** (1 - w) * REL['fast'][n] ** w for n in REL['thermal']}
prod_rows = []
for n in SIG_TH:
    A = SIG_TH[n][0]
    prod_rows.append(dict(product=n, parent=f'Xe-{A}', abundance=ABUND[A], sigma_thermal_b=SIG_TH[n][1],
                          sigma_fast_mb=SIG_FAST[n][1], rel_captures_thermal=REL['thermal'][n],
                          rel_captures_fast=REL['fast'][n], half_life_d=HL[n]))
pd.DataFrame(prod_rows).to_csv(os.path.join(OUT, 'production_relative.csv'), index=False)

def saturation(lam, T):
    """Atoms at end of a constant-rate irradiation of length T per capture: (1-e^{-lam T})/(lam T)."""
    x = lam * T
    return (1 - math.exp(-x)) / x if x > 1e-9 else 1.0

# ---- 2a. 125I ROI acceptance with NEST-LZ yields (P010 f = 0.2, sigma(Ne) = k sigma_p N_i) -------------
ALPHA_EX = 0.067366 + 0.039693 * RHO_LXE      # NEST exciton/ion ratio (P010, recalled likely)
K_SIGP = 0.50 * 0.068                          # P010 calibrated width (k = 0.5, sigma_p = 0.068)
NE_ROI_MAX = 10 ** lz.LZ['log10S2c_max'] / lz.LZ['g2']

def eps_roi_line(E, f):
    nph, ne = lz.nest_er_yields(E, params=lz.NEST_ER_LZ)
    nq = nph + ne
    ni = nq / (1 + ALPHA_EX)
    ne_mean = ne * (1 - f)
    sig = K_SIGP * ni
    return stats.norm.cdf((NE_ROI_MAX - ne_mean) / sig), ne_mean, sig

eps_tab = []
for f in [0.0, 0.1, 0.2, 0.3]:
    tot = 0.0
    for (E, p, desc) in EMIS['I-125']['local']:
        pe, nem, sg = eps_roi_line(E, f)
        tot += p * pe
        eps_tab.append(dict(f=f, line_keV=E, branch=p, Ne_mean=nem, sigma_Ne=sg, P_in_ROI=pe, contrib=p * pe))
    eps_tab.append(dict(f=f, line_keV='total', branch=1.0, Ne_mean=np.nan, sigma_Ne=np.nan, P_in_ROI=tot, contrib=tot))
eps_df = pd.DataFrame(eps_tab)
eps_df.to_csv(os.path.join(OUT, 'I125_ROI_acceptance.csv'), index=False)
EPS_I125 = {f: float(eps_df[(eps_df.f == f) & (eps_df.line_keV == 'total')].P_in_ROI.iloc[0]) for f in [0.0, 0.1, 0.2, 0.3]}
RES['eps_ROI_I125'] = EPS_I125

# ROI edge energy for the ER-band median
def logS2c_median(E):
    nph, ne = lz.nest_er_yields(E, params=lz.NEST_ER_LZ)
    return math.log10(ne * lz.LZ['g2'])
from scipy.optimize import brentq
E_ROI_EDGE = brentq(lambda E: logS2c_median(E) - lz.LZ['log10S2c_max'], 5, 60)
RES['E_ee_ROI_edge_keV'] = E_ROI_EDGE
nph12, ne12 = lz.nest_er_yields(12.0, params=lz.NEST_ER_LZ)
RES['ER_12keV_S1c_log10S2c'] = (nph12 * lz.LZ['g1'], math.log10(ne12 * lz.LZ['g2']))
nph81, ne81 = lz.nest_er_yields(81.0, params=lz.NEST_ER_LZ)
RES['ER_81keV_S1c'] = nph81 * lz.LZ['g1']

# ---- 2b. 125Xe -> 125I chain with purification removal (numerical, production during irradiation) ----
lam_Xe125 = LAM['Xe-125']; lam_I_dec = LAM['I-125']
T_EFF_I = lz.LZ['I125_eff_halflife_d'][0]
lam_I_eff = LN2 / (T_EFF_I * DAY)
F_DECAY_IN_LXE = lam_I_dec / lam_I_eff
RES['I125_fraction_decaying_in_LXe'] = F_DECAY_IN_LXE

def chain_profile(T_irr_d, t_max_d=60.0, dt_d=0.005):
    """Unit capture rate on 124Xe during [0, T_irr]; returns t (d, from end of irradiation), N_Xe, N_I,
    125I decay rate in LXe (per unit total captures C = 1)."""
    t = np.arange(-T_irr_d, t_max_d + dt_d, dt_d)
    dt = dt_d * DAY
    NXe = np.zeros_like(t); NI = np.zeros_like(t)
    P = 1.0 / (T_irr_d * DAY)      # captures per second so that total = 1
    for i in range(1, len(t)):
        prod = P if t[i - 1] < 0 else 0.0
        # exact one-step update for linear ODEs with constant source over dt
        NXe[i] = NXe[i - 1] * math.exp(-lam_Xe125 * dt) + prod / lam_Xe125 * (1 - math.exp(-lam_Xe125 * dt))
        feed = lam_Xe125 * 0.5 * (NXe[i - 1] + NXe[i])
        NI[i] = NI[i - 1] * math.exp(-lam_I_eff * dt) + feed / lam_I_eff * (1 - math.exp(-lam_I_eff * dt))
    rate_I = lam_I_dec * NI       # decays per second in LXe per unit capture
    return t, NXe, NI, rate_I

t_pr, NXe_pr, NI_pr, rate_I_pr = chain_profile(T_IRR_D)
dt_s = (t_pr[1] - t_pr[0]) * DAY
tot_I_decays = np.sum(rate_I_pr) * dt_s          # should be ~ F_DECAY_IN_LXE
frac_after = {tr: float(np.sum(rate_I_pr[t_pr >= tr]) * dt_s / tot_I_decays) for tr in [0.5, 1.0, 2.0]}
i_peak = int(np.argmax(rate_I_pr)); t_peak = t_pr[i_peak]
rate_at_ev = float(np.interp(T_EV_D, t_pr, rate_I_pr)); rate_peak = float(rate_I_pr[i_peak])
RES['I125_profile'] = dict(total_decays_per_capture=float(tot_I_decays), check_lam_ratio=F_DECAY_IN_LXE,
                           frac_after_resume=frac_after, t_peak_d=float(t_peak), rate_event_over_peak=rate_at_ev / rate_peak)
pd.DataFrame(dict(t_d=t_pr, N_Xe125=NXe_pr, N_I125=NI_pr, I125_decay_rate_per_s=rate_I_pr)).iloc[::20].to_csv(
    os.path.join(OUT, 'I125_chain_profile.csv'), index=False)

# ---- 2c. Absolute normalisation from Table I -------------------------------------------------------
N_I125_ROI = lz.LZ['bkg_expected']['I125'][0]     # 8.9 +- 2.7
def captures_124_per_cal(eps, n_cal=N_CAL, t_res=T_RESUME_D):
    """124Xe captures per calibration in the whole LXe inventory implied by 8.9 ROI 125I events."""
    N_dec_FV = N_I125_ROI / eps                                    # 125I decays in FV in WS data
    N_dec_FV_per_cal = N_dec_FV / n_cal
    N_dec_all_per_cal = N_dec_FV_per_cal / frac_after[t_res]       # incl. decays before WS resumed
    C_FV = N_dec_all_per_cal / tot_I_decays
    return C_FV * M_LXE_T / M_FV_T, N_dec_FV

norm_rows = []
for eps_f, eps in EPS_I125.items():
    if eps_f == 0.0:
        continue
    for n_cal in [3, 1]:
        C, Ndec = captures_124_per_cal(eps, n_cal)
        norm_rows.append(dict(f=eps_f, eps_ROI=eps, N_cal=n_cal, I125_decays_FV_run=Ndec,
                              C124_per_cal_whole_LXe=C, C124_per_tonne=C / M_LXE_T))
norm_df = pd.DataFrame(norm_rows)
norm_df.to_csv(os.path.join(OUT, 'normalisation_from_TableI.csv'), index=False)
C124_CENTRAL, NDEC_CENTRAL = captures_124_per_cal(EPS_I125[0.2], 3)
C124_UPPER, _ = captures_124_per_cal(EPS_I125[0.2], 1)      # all 8.9 from the 8 June AmBe
RES['C124_per_cal_whole_LXe'] = dict(central=C124_CENTRAL, upper_one_calibration=C124_UPPER,
                                     f03_3cal=captures_124_per_cal(EPS_I125[0.3], 3)[0])
RES['I125_decays_FV_run_central'] = NDEC_CENTRAL

# ---- 2d. Inventory at the end of irradiation and Bateman decay to the event -----------------------------
def inventory_at_event(C124, relset, t_ev_d=T_EV_D):
    """Atoms at end of irradiation for each product; decay with radioactivedecay to t_ev.
    Returns dict nuclide -> (N_end, A_event_Bq_whole, N_event)."""
    rel = REL[relset]
    atoms_end = {}
    for n, r in rel.items():
        atoms_end[n] = C124 * r * saturation(LAM[n], T_IRR_D * DAY)
    inv = rd.Inventory({n: v for n, v in atoms_end.items() if v > 0}, 'num')
    inv_ev = inv.decay(t_ev_d, 'd')
    acts = inv_ev.activities('Bq'); nums = inv_ev.numbers()
    out = {}
    for n in list(rel.keys()) + ['I-125', 'Cs-137', 'Xe-129', 'Xe-131']:
        if n in acts:
            out[n] = dict(N_end=atoms_end.get(n, 0.0), A_event_Bq=float(acts[n]), N_event=float(nums[n]))
    # 125I: replace the rd (no purification) value with the removal-corrected chain
    out['I-125']['A_event_Bq'] = C124 * rate_at_ev
    out['I-125']['N_event'] = C124 * float(np.interp(t_ev_d, t_pr, NI_pr))
    return out

def decays_in_window(A_Bq, mass_t, hours=1.0):
    return A_Bq * (mass_t / M_LXE_T) * 2 * hours * 3600.0

act_rows = []
for relset in ['thermal', 'fast']:
    for label, C in [('central', C124_CENTRAL), ('upper', C124_UPPER)]:
        inv = inventory_at_event(C, relset)
        for n, v in inv.items():
            act_rows.append(dict(capture_set=relset, normalisation=label, nuclide=n, half_life_d=HL.get(n, np.nan),
                                 atoms_end_irr_whole_LXe=v['N_end'], atoms_event_whole_LXe=v['N_event'],
                                 activity_event_Bq_whole_LXe=v['A_event_Bq'],
                                 activity_event_uBq_per_kg=v['A_event_Bq'] / (M_LXE_T * 1000) * 1e6,
                                 decays_FV_pm1h=decays_in_window(v['A_event_Bq'], M_FV_T),
                                 decays_RFR_pm1h=decays_in_window(v['A_event_Bq'], M_RFR_T),
                                 decays_wall_layer_pm1h=decays_in_window(v['A_event_Bq'], M_WALL_T),
                                 remaining_fraction=(v['N_event'] / v['N_end']) if v['N_end'] > 0 else np.nan))
act_df = pd.DataFrame(act_rows)
act_df.to_csv(os.path.join(OUT, 'activities_16June.csv'), index=False)
cen = act_df[(act_df.capture_set == 'thermal') & (act_df.normalisation == 'central')].set_index('nuclide')
fas = act_df[(act_df.capture_set == 'fast') & (act_df.normalisation == 'central')].set_index('nuclide')
RES['activities_central_thermal_Bq_whole_LXe'] = cen['activity_event_Bq_whole_LXe'].to_dict()
RES['decays_FV_pm1h_central_thermal'] = cen['decays_FV_pm1h'].to_dict()
RES['decays_FV_pm1h_central_fast'] = fas['decays_FV_pm1h'].to_dict()
RES['total_activation_decays_FV_pm1h'] = dict(thermal=float(cen['decays_FV_pm1h'].drop(['Xe-129', 'Xe-131'], errors='ignore').sum()),
                                              fast=float(fas['decays_FV_pm1h'].drop(['Xe-129', 'Xe-131'], errors='ignore').sum()))
RES['total_activation_decays_RFR_pm1h'] = dict(thermal=float(cen['decays_RFR_pm1h'].drop(['Xe-129', 'Xe-131'], errors='ignore').sum()),
                                               fast=float(fas['decays_RFR_pm1h'].drop(['Xe-129', 'Xe-131'], errors='ignore').sum()))

# ---- 2e. 127Xe cross-check against Table I (127Xe + 125Xe = 1.5 +- 0.3) -----------------------------------
# rough SS-in-ROI acceptance for a 127Xe decay uniformly in the FV: the gamma must leave the active LXe
# without interacting (FV stand-off >= 8 cm from the wall; ~2 cm above the cathode -> into RFR gives MSSI, not SS)
# and the local deposit must fall in the ROI (L capture 4.9 keV: yes; K capture 33.2 keV: P(Ne<409) with f=0.2).
def lam_cm(E):   # forward reference to section 3 attenuation model
    return 1.0 / (mu_rho_xe(E) * RHO_LXE)

def mu_rho_xe(E):
    """Recalled XCOM-like total mass attenuation of xenon (cm^2/g), iodine proxy scaled x1.04, K-edge 34.56 keV.
    Table points (E keV: cm^2/g), log-log interpolation.  Reliability: likely (+-30 %)."""
    pts = [(20, 22.0), (30, 8.0), (34.5, 5.9), (34.6, 32.0), (40, 20.0), (50, 11.5), (60, 7.3), (80, 3.4), (100, 1.95),
           (122, 1.15), (150, 0.68), (200, 0.36), (250, 0.23), (300, 0.165), (400, 0.108), (500, 0.085),
           (600, 0.074), (662, 0.069), (800, 0.060), (1000, 0.053), (1500, 0.043), (2615, 0.036)]
    xs = np.log([p[0] for p in pts]); ys = np.log([p[1] for p in pts])
    return float(np.exp(np.interp(np.log(E), xs, ys)))

def p_escape_side(E, standoff=8.0):
    """Rough probability that a gamma from a decay in the FV leaves the active LXe through the wall side
    without interacting: (1/2) * E2-like average over the outward hemisphere for a slab of thickness 'standoff'."""
    lam = lam_cm(E)
    mus = np.linspace(0.02, 1, 200)
    return 0.5 * float(np.mean(np.exp(-standoff / (lam * mus))))

P_K_inROI, _, _ = eps_roi_line(33.2, 0.2)
esc = sum(p * p_escape_side(E) for E, p in EMIS['Xe-127']['gammas'] if E > 100)
EPS_127_SS = esc * (0.14 * 1.0 + 0.83 * P_K_inROI)
RES['Xe127_SS_ROI_acceptance_rough'] = dict(P_gamma_escape=esc, P_K33_in_ROI=P_K_inROI, eps=EPS_127_SS,
                                            note='factor ~3 uncertain; escape through top/bottom neglected')
N127_dec_FV_from_TableI = lz.LZ['bkg_expected']['Xe127_Xe125'][0] / EPS_127_SS
# 127Xe decays in FV over the run predicted from the 125I normalisation (thermal and fast)
def n127_decays_run(C124, relset):
    N_end = C124 * REL[relset]['Xe-127'] * saturation(LAM['Xe-127'], T_IRR_D * DAY)
    return N_end * (M_FV_T / M_LXE_T) * N_CAL * 0.9      # ~90 % decay within the remaining run
RES['Xe127_decays_FV_run'] = dict(from_TableI_1p5=N127_dec_FV_from_TableI,
                                  predicted_thermal=n127_decays_run(C124_CENTRAL, 'thermal'),
                                  predicted_fast=n127_decays_run(C124_CENTRAL, 'fast'))
ratio_127_125_implied = (N127_dec_FV_from_TableI / (N_CAL * 0.9 * M_FV_T / M_LXE_T)) / (C124_CENTRAL * saturation(LAM['Xe-127'], T_IRR_D * DAY))
RES['Xe127_over_Xe125_production_implied'] = dict(implied=ratio_127_125_implied, thermal=REL['thermal']['Xe-127'], fast=REL['fast']['Xe-127'])
# 127Xe activity on 16 June from the Table-I route (share of the 8 June calibration = 1/3, earlier ones decayed)
A127_TableI = N127_dec_FV_from_TableI / 0.9 / N_CAL * (M_LXE_T / M_FV_T) * math.exp(-LAM['Xe-127'] * T_EV_D * DAY) * LAM['Xe-127']
A127_TableI_with_earlier = A127_TableI * (1 + (math.exp(-LAM['Xe-127'] * 38 * DAY) + math.exp(-LAM['Xe-127'] * 68 * DAY)) / math.exp(-LAM['Xe-127'] * T_EV_D * DAY))
RES['Xe127_activity_16June_Bq_TableI_route'] = dict(one_calibration=A127_TableI, with_earlier_at_m30_m60d=A127_TableI_with_earlier,
                                                    decays_FV_pm1h=decays_in_window(A127_TableI_with_earlier, M_FV_T))

# ---- 2f. Table-I-anchored intermediate capture set and its activities ---------------------------------------
w_int = math.log(ratio_127_125_implied / REL['thermal']['Xe-127']) / math.log(REL['fast']['Xe-127'] / REL['thermal']['Xe-127'])
w_int = min(max(w_int, 0.0), 1.0)
REL['intermediate'] = make_intermediate(w_int)
RES['intermediate_set'] = dict(weight_w=w_int, rel_captures=REL['intermediate'])
for label, C in [('central', C124_CENTRAL), ('upper', C124_UPPER)]:
    inv = inventory_at_event(C, 'intermediate')
    for n, v in inv.items():
        act_rows.append(dict(capture_set='intermediate', normalisation=label, nuclide=n, half_life_d=HL.get(n, np.nan),
                             atoms_end_irr_whole_LXe=v['N_end'], atoms_event_whole_LXe=v['N_event'],
                             activity_event_Bq_whole_LXe=v['A_event_Bq'],
                             activity_event_uBq_per_kg=v['A_event_Bq'] / (M_LXE_T * 1000) * 1e6,
                             decays_FV_pm1h=decays_in_window(v['A_event_Bq'], M_FV_T),
                             decays_RFR_pm1h=decays_in_window(v['A_event_Bq'], M_RFR_T),
                             decays_wall_layer_pm1h=decays_in_window(v['A_event_Bq'], M_WALL_T),
                             remaining_fraction=(v['N_event'] / v['N_end']) if v['N_end'] > 0 else np.nan))
act_df = pd.DataFrame(act_rows)
act_df.to_csv(os.path.join(OUT, 'activities_16June.csv'), index=False)
inter = act_df[(act_df.capture_set == 'intermediate') & (act_df.normalisation == 'central')].set_index('nuclide')
SETS = {'thermal': cen, 'intermediate': inter, 'fast': fas}
RES['activities_central_Bq_whole_LXe'] = {k: v['activity_event_Bq_whole_LXe'].drop(['Xe-129', 'Xe-131'], errors='ignore').to_dict() for k, v in SETS.items()}
RES['decays_FV_pm1h_central'] = {k: v['decays_FV_pm1h'].drop(['Xe-129', 'Xe-131'], errors='ignore').to_dict() for k, v in SETS.items()}
RES['total_activation_decays_FV_pm1h'] = {k: float(v['decays_FV_pm1h'].drop(['Xe-129', 'Xe-131'], errors='ignore').sum()) for k, v in SETS.items()}
RES['total_activation_decays_RFR_pm1h'] = {k: float(v['decays_RFR_pm1h'].drop(['Xe-129', 'Xe-131'], errors='ignore').sum()) for k, v in SETS.items()}
RES['Xe127_decays_FV_run']['predicted_intermediate'] = n127_decays_run(C124_CENTRAL, 'intermediate')

# ----------------------------------------------------------------------------------------
# 3. Gamma transport in LXe
# ----------------------------------------------------------------------------------------
# 'assignment' bracket: longer recalled lengths handed to us (0.07 cm @40, 0.25 @80, 1.3 @164, 2.3 @203, 3.5 @250, 5.5 @375)
ASSIGN_LAM = [(40, 0.07), (80, 0.25), (164, 1.3), (203, 2.3), (250, 3.5), (375, 5.5)]
def lam_assign(E):
    xs = np.log([p[0] for p in ASSIGN_LAM]); ys = np.log([p[1] for p in ASSIGN_LAM])
    return float(np.exp(np.interp(np.log(E), xs, ys)))

ME = 510.999
def kn_dsigma_dT(E, T):
    """Klein-Nishina dsigma/dT (arbitrary units), T = electron kinetic energy (keV)."""
    Ep = E - T
    if Ep <= 0 or Ep < E / (1 + 2 * E / ME):
        return 0.0
    cos = 1 - ME * (1 / Ep - 1 / E)
    if abs(cos) > 1:
        return 0.0
    r = Ep / E
    dsig_dOmega = 0.5 * r**2 * (r + 1 / r - (1 - cos**2))
    dOmega_dT = 2 * math.pi * ME / (Ep**2)  # |dcos/dT| * 2pi
    return dsig_dOmega * dOmega_dT

def kn_fraction(E, T1, T2):
    Tmax = 2 * E**2 / (ME + 2 * E)
    tot = integrate.quad(lambda T: kn_dsigma_dT(E, T), 0, Tmax, limit=200)[0]
    if T1 >= Tmax:
        return 0.0, Tmax
    part = integrate.quad(lambda T: kn_dsigma_dT(E, T), T1, min(T2, Tmax), limit=200)[0]
    return part / tot, Tmax

def compton_share(E):
    """Compton fraction of the total attenuation (recalled: KN per electron x 0.2477 cm^2/g per barn)."""
    k = E / ME
    sig_kn = 2 * math.pi * (2.8179e-13)**2 * ((1 + k) / k**2 * (2 * (1 + k) / (1 + 2 * k) - math.log(1 + 2 * k) / k)
                                             + math.log(1 + 2 * k) / (2 * k) - (1 + 3 * k) / (1 + 2 * k)**2) * 1e24  # barn
    mu_c = sig_kn * 0.2477
    return min(mu_c / mu_rho_xe(E), 1.0)

D_WALL = lz.LZ['ev_r_from_true_wall_cm']; D_RFR = lz.LZ['ev_z_above_cathode_cm']
LINES = [(35.5, 'I-125'), (39.6, 'Xe-129m'), (55.0, 'Xe-125'), (57.6, 'Xe-127'), (81.0, 'Xe-133'), (145.3, 'Xe-127'),
         (163.9, 'Xe-131m'), (172.1, 'Xe-127'), (188.4, 'Xe-125'), (196.6, 'Xe-129m'), (202.9, 'Xe-127'),
         (233.2, 'Xe-133m'), (243.4, 'Xe-125'), (249.8, 'Xe-135'), (375.0, 'Xe-127'), (455.5, 'Xe-137'),
         (526.6, 'Xe-135m'), (661.7, 'Cs-137')]
g_rows = []
for E, src in LINES:
    lam = lam_cm(E); lamA = lam_assign(E) if 40 <= E <= 375 else np.nan
    fkn, Tmax = kn_fraction(E, 10.0, 14.0)
    cs = compton_share(E)
    # reverse topology B: decay in dead region; gamma travels >= 20 cm to the vertex, Compton 10-14 keV there,
    # scattered gamma escapes >= 26.4 cm (to leave the active volume without a second S2).
    Ep = E - 12.0
    pB = math.exp(-20.0 / lam) * cs * fkn * math.exp(-D_RFR / lam_cm(Ep))
    pB_A = (math.exp(-20.0 / lamA) * cs * fkn * math.exp(-D_RFR / lam_assign(min(max(Ep, 40), 375)))) if not np.isnan(lamA) else np.nan
    g_rows.append(dict(E_keV=E, source=src, mu_rho_cm2_g=mu_rho_xe(E), lambda_cm=lam, lambda_assign_cm=lamA,
                       P_20cm=math.exp(-20 / lam), P_26p9cm=math.exp(-D_WALL / lam), P_26p4cm=math.exp(-D_RFR / lam),
                       P_20cm_assign=(math.exp(-20 / lamA) if not np.isnan(lamA) else np.nan),
                       P_26p9cm_assign=(math.exp(-D_WALL / lamA) if not np.isnan(lamA) else np.nan),
                       compton_share=cs, KN_frac_10_14keV=fkn, T_compton_max_keV=Tmax,
                       topologyB_per_gamma=pB, topologyB_per_gamma_assign=pB_A))
g_df = pd.DataFrame(g_rows)
g_df.to_csv(os.path.join(OUT, 'gamma_transport.csv'), index=False)
RES['gamma_transport_key'] = {f"{r.E_keV:g}": dict(lam=r.lambda_cm, P20=r.P_20cm, P26p9=r.P_26p9cm, lamA=r.lambda_assign_cm,
                                                    P20A=r.P_20cm_assign, KN=r.KN_frac_10_14keV, topoB=r.topologyB_per_gamma)
                              for r in g_df.itertuples()}
# energy at which P(20 cm) = 1e-3 in each attenuation model
E_grid = np.linspace(100, 2600, 2501)
p20 = np.array([math.exp(-20 / lam_cm(E)) for E in E_grid])
E_1e3 = float(E_grid[np.argmax(p20 >= 1e-3)])
E_1e3_A = None
Eg2 = np.linspace(100, 375, 1000); p20A = np.array([math.exp(-20 / lam_assign(E)) for E in Eg2])
E_1e3_A = float(Eg2[np.argmax(p20A >= 1e-3)]) if p20A.max() >= 1e-3 else None
RES['E_gamma_for_P20cm_1e-3_keV'] = dict(recalled_XCOM=E_1e3, assignment_bracket=E_1e3_A)

# ----------------------------------------------------------------------------------------
# 4. Correlated MSSI: can the vertex deposit be 12 +- 2 keV?  (topology A)
# ----------------------------------------------------------------------------------------
S2_TARGET_E = lz.LZ['MSSI_first_scatter_keV']   # (12, 2)
topo_rows = []
for n, d in EMIS.items():
    if n in ('Cs-137', 'Kr-83m', 'Xe-137', 'Xe-135m'):
        pass
    for (E, p, desc) in d['local']:
        if E == 'beta':
            # fraction of the beta spectrum in 10-14 keV (allowed-shape approximation, Fermi function ~ const)
            Q = d['beta_endpoint']
            def beta_shape(T, Q=Q):
                W = 1 + T / ME; p_e = math.sqrt(W**2 - 1)
                return p_e * W * (Q - T)**2
            norm = integrate.quad(beta_shape, 0, Q)[0]
            fr = integrate.quad(beta_shape, 10, 14)[0] / norm
            ok = True
            note = f'beta 10-14 keV fraction {fr:.3f}; needs the gamma (if any) to reach a dead region >= 20 cm away'
        else:
            fr = 1.0 if abs(E - 12.0) <= 2.0 else 0.0
            ok = fr > 0
            note = desc
        # gamma that must reach the dead region: the most penetrating gamma of the scheme
        if d['gammas']:
            Eg, pg = max(d['gammas'], key=lambda x: x[0])
            p20 = math.exp(-20 / lam_cm(Eg)); p20A = math.exp(-20 / lam_assign(min(max(Eg, 40), 375)))
        else:
            Eg, pg, p20, p20A = np.nan, 0.0, 0.0, 0.0
        topo_rows.append(dict(nuclide=n, local_deposit_keV=E, prob=p, vertex_12keV_possible=ok, vertex_12keV_fraction=fr,
                              most_penetrating_gamma_keV=Eg, gamma_prob=pg, P_gamma_20cm=p20, P_gamma_20cm_assign=p20A,
                              topologyA_factor=fr * p * pg * p20, note=note))
topo_df = pd.DataFrame(topo_rows)
topo_df.to_csv(os.path.join(OUT, 'topologyA_correlated.csv'), index=False)
RES['topologyA_max_factor'] = float(topo_df.topologyA_factor.max())
RES['topologyA_max_row'] = topo_df.loc[topo_df.topologyA_factor.idxmax()].to_dict()

# Expected correlated (topology A) events over the whole run:
#   sum_i N_i(FV decays in WS data, all calibrations) x f_vertex(10-14 keV) x p_gamma x P(>= 20 cm) x F_DEAD,
# F_DEAD = probability that the gamma, having reached 20+ cm, deposits an energy compatible with the 471 phd
# S1-only pulse in a dead region (RFR: 166-269 keV; wall 3 mm layer: 70-84 keV) -- estimated 0.03 (x3 uncertain).
F_DEAD = 0.03
# 137Cs (daughter of 137Xe) is a chemically active species; LZ measures a 3.6 d effective removal half-life for
# the analogous 125I, so the central case removes Cs on that time scale; 'retained' keeps it for the whole run.
CS_REMOVAL_D = T_EFF_I
FRAC_CS_DECAY = {'removed': LAM['Cs-137'] / (LN2 / (CS_REMOVAL_D * DAY)),
                 'retained': LAM['Cs-137'] * (lz.LZ['live_days'] * DAY)}
RES['Cs137_fraction_decaying_in_run'] = FRAC_CS_DECAY
def n_fv_run_after_resume(n, C124, relset, cs_mode='removed'):
    if n == 'I-125':
        return C124 * tot_I_decays * frac_after[T_RESUME_D] * N_CAL * M_FV_T / M_LXE_T
    if n == 'Cs-137':
        return C124 * REL[relset]['Xe-137'] * FRAC_CS_DECAY[cs_mode] * N_CAL * M_FV_T / M_LXE_T
    if n == 'Kr-83m':
        return 0.0     # injected, not activation; handled qualitatively
    N_end = C124 * REL[relset][n] * saturation(LAM[n], T_IRR_D * DAY)
    return N_end * math.exp(-LAM[n] * T_RESUME_D * DAY) * N_CAL * M_FV_T / M_LXE_T
topoA_run = {}
for relset in ['thermal', 'intermediate', 'fast']:
    for label, C in [('central', C124_CENTRAL), ('upper', C124_UPPER)]:
        for cs_mode in ['removed', 'retained']:
            tot = 0.0; per = {}; tot_xe_only = 0.0
            for r in topo_df.itertuples():
                N = n_fv_run_after_resume(r.nuclide, C, relset, cs_mode)
                val = N * r.topologyA_factor * F_DEAD
                per[f'{r.nuclide}_{r.local_deposit_keV}'] = val
                tot += val
                if r.nuclide.startswith('Xe') or r.nuclide == 'I-125':
                    tot_xe_only += val
            topoA_run[f'{relset}_{label}_Cs{cs_mode}'] = dict(total=tot, total_without_Cs137=tot_xe_only,
                                                             leading=max(per, key=per.get), leading_value=max(per.values()))
RES['topologyA_expected_run'] = topoA_run
RES['topologyA_expected_run_central'] = topoA_run['intermediate_central_Csremoved']['total']
RES['topologyA_expected_run_max'] = max(v['total'] for v in topoA_run.values())
RES['topologyA_expected_run_max_without_Cs137'] = max(v['total_without_Cs137'] for v in topoA_run.values())
# time-resolved: expected correlated topologies in +-1 h around the event (decays in FV in the window x factor x F_DEAD)
topoA_window = {}
for relset, df_set in SETS.items():
    tot = 0.0
    for r in topo_df.itertuples():
        if r.nuclide in df_set.index:
            tot += df_set.loc[r.nuclide, 'decays_FV_pm1h'] * r.topologyA_factor * F_DEAD
    topoA_window[relset] = tot
RES['topologyA_expected_pm1h_event'] = topoA_window
RES['topologyA_expected_pm1h_event_max_upper'] = max(topoA_window.values()) * C124_UPPER / C124_CENTRAL
RES['N_FV_run_after_resume_central_intermediate'] = {n: n_fv_run_after_resume(n, C124_CENTRAL, 'intermediate')
                                                    for n in ['Xe-125', 'I-125', 'Xe-127', 'Xe-129m', 'Xe-131m', 'Xe-133', 'Xe-133m', 'Xe-135']}

# 127Xe 375 keV route explicitly (vertex 33.2 or 4.9 keV is NOT 12 keV; report the geometry factor anyway)
RES['Xe127_375_geometry_only'] = dict(P_26p9_recalled=math.exp(-D_WALL / lam_cm(375)), P_26p9_assign=math.exp(-D_WALL / 5.5),
                                      P_20_recalled=math.exp(-20 / lam_cm(375)), P_20_assign=math.exp(-20 / 5.5))
# S2c implied by the quantised local deposits vs observed 9268 phd
def s2c_of_ER(E, f=0.0):
    nph, ne = lz.nest_er_yields(E, params=lz.NEST_ER_LZ)
    return ne * (1 - f) * lz.LZ['g2']
RES['S2c_of_local_deposits'] = {f'{E} keV': s2c_of_ER(E) for E in [4.9, 12.0, 33.2, 40.4, 67.3]}
RES['S2c_of_local_deposits_f0p2'] = {f'{E} keV': s2c_of_ER(E, 0.2) for E in [4.9, 33.2]}

# ----------------------------------------------------------------------------------------
# 5. Random coincidence (topology C)
# ----------------------------------------------------------------------------------------
# ER rate in the FV in a 10-14 keV window: science sample 1710 events, ER band inside the ROI up to E_ROI_EDGE
N_SCI = lz.LZ['n_obs_science']
R_ER_per_keV = N_SCI / E_ROI_EDGE                  # events per keV over 220 d (flat approx.)
N_ER_12 = R_ER_per_keV * 4.0
R_ER_12 = N_ER_12 / (lz.LZ['live_days'] * DAY)    # per second
DT_MERGE = 0.2e-6
# S1-only rate from activation in the dead regions integrated over the run: total dead-region decays
# = sum over nuclides of all decays after the calibration in RFR + wall layer (all N_CAL calibrations)
def total_dead_decays(C124, relset):
    tot = 0.0
    for n, r in REL[relset].items():
        N_end = C124 * r * saturation(LAM[n], T_IRR_D * DAY)
        tot += N_end                                # every atom decays eventually within the run (all T1/2 << 220 d)
    # 125I decays in LXe
    tot += C124 * tot_I_decays
    return tot * (M_RFR_T + M_WALL_T) / M_LXE_T * N_CAL

coinc = {}
for relset in ['thermal', 'intermediate', 'fast']:
    for label, C in [('central', C124_CENTRAL), ('upper', C124_UPPER)]:
        Ndead = total_dead_decays(C, relset)
        for dt in [0.2e-6, 1.0e-6]:
            coinc[f'{relset}_{label}_dt{dt*1e6:g}us'] = dict(dead_region_decays_run=Ndead, N_ER_10_14keV_run=N_ER_12,
                                                             R_ER_per_s=R_ER_12, dt_s=dt, expected_coincidences=Ndead * R_ER_12 * dt)
RES['random_coincidence'] = coinc
RES['random_coincidence_max'] = max(v['expected_coincidences'] for v in coinc.values())
# S1-only rate from activation in the dead regions at the event time
RES['dead_region_S1only_rate_event_Hz'] = {
    k: float(v['activity_event_Bq_whole_LXe'].drop(['Xe-129', 'Xe-131'], errors='ignore').sum() * (M_RFR_T + M_WALL_T) / M_LXE_T)
    for k, v in SETS.items()}

# ----------------------------------------------------------------------------------------
# 6. What activation does produce
# ----------------------------------------------------------------------------------------
# 6a. ER lines: S1c and log10 S2c of full-energy deposits (beta-like yields; EC/IC lines sit a bit lower in S2)
line_rows = []
for n, E in [('I-125', 67.3), ('I-125', 40.4), ('Xe-127', 33.2), ('Xe-125', 33.2), ('Xe-129m', 236.1), ('Xe-131m', 163.9),
             ('Xe-133m', 233.2), ('Xe-133', 81.0), ('Xe-133', 200.0), ('Kr-83m', 41.5)]:
    nph, ne = lz.nest_er_yields(E, params=lz.NEST_ER_LZ)
    line_rows.append(dict(nuclide=n, E_keV=E, S1c_phd=nph * lz.LZ['g1'], log10S2c=math.log10(ne * lz.LZ['g2']),
                          in_WS_ROI=(nph * lz.LZ['g1'] <= 600) and (math.log10(ne * lz.LZ['g2']) <= 4.15),
                          in_HE_SB_S1=(800 <= nph * lz.LZ['g1'] <= 1700)))
lines_df = pd.DataFrame(line_rows)
lines_df.to_csv(os.path.join(OUT, 'activation_lines_S1S2.csv'), index=False)

# 6b. 133Xe at S1c > 500 phd: fraction of 133Xe decays with total deposit >= 81 keV  (all, since beta + 81 keV
#     either as gamma absorbed within ~mm or IC) -> S1c >= 499 phd.  Number in FV over the run:
N133_FV_run = {rs: C124_CENTRAL * REL[rs]['Xe-133'] * saturation(LAM['Xe-133'], T_IRR_D * DAY) * N_CAL * M_FV_T / M_LXE_T
               for rs in ['thermal', 'intermediate', 'fast']}
RES['Xe133_decays_FV_run'] = N133_FV_run

# 6c. 127Xe RFR-MSSI near the cathode: decay within the FV bottom layer, gamma absorbed in the RFR
FV_BOTTOM_CM = 2.0      # assumed FV stand-off above the cathode (recalled/assumed)
H_FV_CM = 130.0         # approximate FV height
def eff_layer(E, h0=FV_BOTTOM_CM):
    lam = lam_cm(E)
    # integral over h >= h0 of (1/2) int_0^1 exp(-h/(lam mu)) dmu dh = (lam/2) int_0^1 mu exp(-h0/(lam mu)) dmu
    mus = np.linspace(1e-3, 1, 2000)
    return 0.5 * lam * float(np.trapezoid(mus * np.exp(-h0 / (lam * mus)), mus))
frac_rfr_mssi = sum(p * eff_layer(E) for E, p in EMIS['Xe-127']['gammas'] if E > 100) / H_FV_CM
N127_FV_run_range = (RES['Xe127_decays_FV_run']['predicted_thermal'], N127_dec_FV_from_TableI)
RES['Xe127_RFR_MSSI'] = dict(fraction_per_decay=frac_rfr_mssi,
                             expected_events_range=[frac_rfr_mssi * N127_FV_run_range[0], frac_rfr_mssi * N127_FV_run_range[1]],
                             in_ROI_share_L_capture=0.14,
                             expected_in_ROI_range=[0.14 * frac_rfr_mssi * N127_FV_run_range[0], 0.14 * frac_rfr_mssi * N127_FV_run_range[1]],
                             S1c_L=RES['ER_12keV_S1c_log10S2c'][0] * 0 + lz.nest_er_yields(4.9, params=lz.NEST_ER_LZ)[0] * lz.LZ['g1'] + 471 * 202.9 / 204,
                             log10S2c_L=math.log10(s2c_of_ER(4.9)), z_range_cm='0-5 cm above the cathode (event: 26.4 cm)')

# ----------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------
# Fig 1: activity vs time after the AmBe for the central thermal normalisation (whole LXe), 125I with removal
tgrid = np.linspace(0.01, 40, 800)
fig, ax = plt.subplots(figsize=(7.2, 4.6))
colors = plt.cm.tab10(np.linspace(0, 1, 10))
plot_nucs = ['Xe-125', 'I-125', 'Xe-127', 'Xe-129m', 'Xe-131m', 'Xe-133', 'Xe-133m', 'Xe-135']
inv0 = {n: C124_CENTRAL * REL['thermal'][n] * saturation(LAM[n], T_IRR_D * DAY) for n in REL['thermal']}
inv_rd = rd.Inventory({n: v for n, v in inv0.items() if v > 0}, 'num')
curves = {n: [] for n in plot_nucs}
for t in tgrid:
    a = inv_rd.decay(t, 'd').activities('Bq')
    for n in plot_nucs:
        if n == 'I-125':
            curves[n].append(C124_CENTRAL * float(np.interp(t, t_pr, rate_I_pr)))
        else:
            curves[n].append(float(a.get(n, 0.0)))
for i, n in enumerate(plot_nucs):
    ax.plot(tgrid, curves[n], label=n, color=colors[i])
ax.axvline(T_EV_D, color='k', ls='--', lw=1); ax.text(T_EV_D + 0.3, 2e-5, '16 June 21:22', rotation=90, va='bottom', fontsize=8)
ax.set_yscale('log'); ax.set_ylim(1e-5, 1e0); ax.set_xlim(0, 40)
ax.set_xlabel('days after the end of the AmBe deployment (8 June 2023)')
ax.set_ylabel('activity in the 10 t LXe inventory (Bq)')
ax.set_title('Activation products after a 1-day AmBe: thermal-capture ratios, normalised to LZ Table I (125I = 8.9)', fontsize=9)
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3, which='both')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P029_fig1_activities_vs_time.png'), dpi=160); plt.close(fig)

# Fig 2: transmission over 20 cm and 26.9 cm vs gamma energy, both attenuation models; candidate lines marked
Eg = np.linspace(30, 700, 1400)
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.plot(Eg, [math.exp(-20 / lam_cm(E)) for E in Eg], 'k-', label='P(travel 20 cm), recalled XCOM-like')
ax.plot(Eg, [math.exp(-D_WALL / lam_cm(E)) for E in Eg], 'k--', label='P(travel 26.9 cm), recalled XCOM-like')
EgA = np.linspace(40, 375, 500)
ax.plot(EgA, [math.exp(-20 / lam_assign(E)) for E in EgA], 'r-', alpha=0.7, label='P(20 cm), longer bracket (assignment lengths)')
for E, src in LINES:
    if E <= 700:
        ax.axvline(E, color='grey', lw=0.6, alpha=0.6)
        ax.text(E, 2e-9, f'{src} {E:g}', rotation=90, fontsize=6, va='bottom', ha='right')
ax.axhline(1e-3, color='b', lw=0.8, ls=':'); ax.text(650, 1.3e-3, '10$^{-3}$', color='b', fontsize=8)
ax.set_yscale('log'); ax.set_ylim(1e-10, 1); ax.set_xlim(30, 700)
ax.set_xlabel('gamma-ray energy (keV)'); ax.set_ylabel('probability of travelling d without interacting, exp(-d/lambda)')
ax.set_title('LXe transparency for the activation gamma lines (event: 26.9 cm from wall, 26.4 cm above cathode)', fontsize=8.5)
ax.legend(fontsize=8, loc='upper left'); ax.grid(alpha=0.3, which='both')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P029_fig2_gamma_transmission.png'), dpi=160); plt.close(fig)

# ----------------------------------------------------------------------------------------
# Save master results
# ----------------------------------------------------------------------------------------
def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o
with open(os.path.join(OUT, 'P029_results.json'), 'w') as fh:
    json.dump(_clean(RES), fh, indent=1)

# console summary
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)
print('sigma_th natural Xe (b):', round(sig_nat_th, 2))
print('ER-band median crosses ROI edge at E_ee =', round(E_ROI_EDGE, 1), 'keV; 12 keV ER -> S1c, logS2c =', RES['ER_12keV_S1c_log10S2c'])
print('eps_ROI(125I) by f:', EPS_I125)
print('125I profile:', RES['I125_profile'])
print('C124 per calibration (whole LXe):', RES['C124_per_cal_whole_LXe'], ' 125I decays in FV over run:', NDEC_CENTRAL)
print(act_df[(act_df.normalisation == 'central')][['capture_set', 'nuclide', 'half_life_d', 'atoms_end_irr_whole_LXe',
      'activity_event_Bq_whole_LXe', 'decays_FV_pm1h', 'decays_RFR_pm1h', 'remaining_fraction']].to_string())
print('127Xe check:', RES['Xe127_decays_FV_run'], RES['Xe127_over_Xe125_production_implied'])
print('127Xe activity (Table I route):', RES['Xe127_activity_16June_Bq_TableI_route'])
print(g_df[['E_keV', 'source', 'lambda_cm', 'lambda_assign_cm', 'P_20cm', 'P_26p9cm', 'P_20cm_assign', 'KN_frac_10_14keV', 'topologyB_per_gamma']].to_string())
print('E for P20=1e-3:', RES['E_gamma_for_P20cm_1e-3_keV'])
print(topo_df[['nuclide', 'local_deposit_keV', 'prob', 'vertex_12keV_fraction', 'most_penetrating_gamma_keV', 'P_gamma_20cm', 'topologyA_factor']].to_string())
print('S2c of local deposits:', RES['S2c_of_local_deposits'])
print('random coincidences:')
for k, v in coinc.items():
    print(f"  {k:32s} dead decays {v['dead_region_decays_run']:.3g}  expected {v['expected_coincidences']:.3g}")
print('topology A over run:')
for k, v in topoA_run.items():
    print(f"  {k:24s} total {v['total']:.3g}  leading {v['leading']} {v['leading_value']:.3g}")
print('intermediate set w =', RES['intermediate_set']['weight_w'], RES['intermediate_set']['rel_captures'])
print('N_FV_run after resume (intermediate):', RES['N_FV_run_after_resume_central_intermediate'])
print('dead-region S1-only rate at event (Hz):', RES['dead_region_S1only_rate_event_Hz'])
print(lines_df.to_string())
print('133Xe decays FV run:', N133_FV_run)
print('127Xe RFR MSSI:', RES['Xe127_RFR_MSSI'])
