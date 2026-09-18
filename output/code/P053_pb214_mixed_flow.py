"""P053 -- 214Pb in the mixed-flow state: what the unavailable radon tag costs and whether
214Pb-driven MSSI could have been elevated around 16 June 2023.

Run from the simulation root:  .venv/bin/python output/code/P053_pb214_mixed_flow.py
Writes tables to output/work/P053/ and figures to output/work/P053/figures/.

Sections
  1. tag-active / mixed-flow fraction from the HE-SB radon-tag statistics (binomial)
  2. 222Rn chain timing (radioactivedecay) and daughter transport distances
  3. ROI ER window (NEST-LZ), Fermi beta acceptance, 214Pb activity implied by Table I
  4. gamma ray-tracing MC: probability that a 214Pb excited-state gamma reaches a charge-dead
     region (3 mm wall shell / 13.75 cm RFR) from the event position and from the FV average;
     HE-SB RFR cross-check against LZ's 21.5 predicted events
  5. elevation factor needed for a 10% 214Pb-MSSI explanation vs what the ER and alpha rates allow
  6. radon-tag counterfactual likelihood ratios
"""
import sys, os, json, math
sys.path.insert(0, 'output/code')
from common import lzcommon as lz
import numpy as np
from scipy import stats, integrate, optimize, special
import radioactivedecay as rd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'output/work/P053'
FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)
R = {}  # results dictionary written to JSON

LIVE_D = lz.LZ['live_days']          # 220 d
FV_KG = lz.LZ['fiducial_mass_t'] * 1000.0
N_ROI_SCI = lz.LZ['n_obs_science']   # 1710
N_INT_BETA = lz.LZ['bkg_expected']['internal_beta'][0]  # 1341

# ------------------------------------------------------------------ recalled inputs (flagged)
RECALLED = dict(
    TPC_radius_cm=72.8, TPC_drift_cm=145.6,          # likely (used by P004/P013)
    LXe_density=2.9,                                  # likely (lzcommon uses 2.9 for NEST)
    Pb214_Q_keV=1019.0,                               # likely (ENSDF Q_beta ~ 1018-1024 keV)
    Pb214_BR_gs=0.092, Pb214_BR_gs_alt=0.11,          # uncertain: ground-state branch 9-11%
    Pb214_levels_keV={'352': 351.9, '295': 295.2, '53': 53.2},
    Pb214_gamma_I={'352': 0.356, '295': 0.184, '242': 0.073, '53': 0.011},  # likely (per decay)
    Pb214_beta_feed={'352': 0.46, '295': 0.41, '53': 0.02, 'other': 0.018},  # uncertain
    lambda_cm={'352': 2.7, '295': 2.2, '242': 1.5},   # likely +-15% (XCOM-like total, LXe 2.9 g/cc)
    lambda_bracket={'352': (2.3, 3.1), '295': (1.9, 2.6), '242': (1.3, 1.8)},
    compton_fraction={'352': 0.55, '295': 0.45, '242': 0.35},  # uncertain (Xe, total-xsec share)
    FV_zmin_cm=9.0, FV_ztop_margin_cm=12.8, FV_wall_margin_cm=8.0, FV_wall_margin_mean_cm=10.7,  # paper l.137-139
    vol54_wall_margin_cm=5.5, vol54_zmin_cm=(2.0, 5.0),  # 5.4 t volume: radius derived; WS2024 z cuts uncertain
    purification_t_per_day=(1.0, 4.0),                # uncertain
    ion_mobility_cm2_Vs=(1e-4, 1e-3),                 # uncertain (positive ions in LXe)
    Rn222_LZ_uBq_kg=(1.0, 5.0),                       # uncertain (LZ SR1/WS2024 quoted values)
    false_tag_rates=(0.01, 0.03, 0.10),               # uncertain
    Pb214_share_of_internal_beta=(0.5, 0.7, 0.9),     # assumption: 214Pb+212Pb+85Kr make up 1341
)
R['recalled_inputs'] = RECALLED

def cp_interval(k, n, cl):
    """Clopper-Pearson central interval."""
    a = (1 - cl) / 2
    lo = 0.0 if k == 0 else stats.beta.ppf(a, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a, k + 1, n - k)
    return float(lo), float(hi)

# ================================================================== 1. tag-active fraction
sec1 = {}
for name, k, n in [('HE-SB RFR science', 12, 18), ('HE-SB prompt (1 wall + 2 RFR)', 3, 3),
                   ('combined', 15, 21)]:
    f = k / n
    sec1[name] = dict(k=k, n=n, f_active=f, ci68=cp_interval(k, n, 0.68), ci95=cp_interval(k, n, 0.95),
                      f_mixed=1 - f, mixed_days=(1 - f) * LIVE_D,
                      mixed_days_ci68=tuple(sorted(((1 - x) * LIVE_D for x in cp_interval(k, n, 0.68)))),
                      mixed_days_ci95=tuple(sorted(((1 - x) * LIVE_D for x in cp_interval(k, n, 0.95)))))
# tag efficiency check 7/12
sec1['tag_eff_7_of_12'] = dict(eff=7 / 12, ci68=cp_interval(7, 12, 0.68), ci95=cp_interval(7, 12, 0.95),
                               p_binom_ge7_if_0p6=float(stats.binom.sf(6, 12, 0.6)),
                               p_two_sided_vs_0p6=float(2 * min(stats.binom.cdf(7, 12, 0.6), stats.binom.sf(6, 12, 0.6))))
# rate-ratio degeneracy: RFR-MSSI rate mixed/active as a function of the (unknown) livetime fraction f_m
sec1['rate_ratio_vs_fm'] = {str(fm): (6 / fm) / (12 / (1 - fm)) for fm in (0.1, 0.2, 0.33, 0.5, 0.67)}
# P(the one candidate falls in mixed flow) if rates are flow independent = f_mixed
sec1['P_candidate_in_mixed_flow'] = dict(central=1 - 12 / 18, ci68=tuple(sorted(1 - np.array(cp_interval(12, 18, 0.68)))),
                                         ci95=tuple(sorted(1 - np.array(cp_interval(12, 18, 0.95)))))
R['sec1_tag_active'] = sec1

# ================================================================== 2. chain and transport
sec2 = {}
chain = ['Rn-222', 'Po-218', 'Pb-214', 'Bi-214', 'Po-214']
hl = {n: rd.Nuclide(n).half_life('s') for n in chain}
sec2['half_life_s'] = hl
sec2['mean_life_s'] = {n: hl[n] / math.log(2) for n in chain}
sec2['progeny'] = {n: rd.Nuclide(n).progeny() for n in chain}
# time for 218Po alpha -> 214Pb beta: mean delay = tau(Pb214) (218Po itself decays in 4.5 min mean)
tau_pb = hl['Pb-214'] / math.log(2)
tau_po = hl['Po-218'] / math.log(2)
sec2['mean_delay_Po218_alpha_to_Pb214_beta_min'] = tau_pb / 60
sec2['mean_delay_Rn222_alpha_to_Pb214_beta_min'] = (tau_po + tau_pb) / 60
# Bateman check with radioactivedecay: pure Rn-222 1 Bq -> Pb-214 activity vs time
inv = rd.Inventory({'Rn-222': 1.0}, 'Bq')
sec2['Pb214_activity_after_1Bq_Rn222_Bq'] = {str(t): float(inv.decay(t, 'h').activities('Bq')['Pb-214']) for t in (0.5, 1, 2, 4, 24)}
# transport distances during one 214Pb mean life and half-life
v_grid_mm_s = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
sec2['transport_distance_cm'] = {str(v): dict(per_half_life=v * 0.1 * hl['Pb-214'], per_mean_life=v * 0.1 * tau_pb) for v in v_grid_mm_s}
# bulk throughput velocity from recalled purification flow through the TPC cross-section
area_m2 = math.pi * (RECALLED['TPC_radius_cm'] / 100) ** 2
vb = [(t / RECALLED['LXe_density']) / 86400 / area_m2 * 1e3 for t in RECALLED['purification_t_per_day']]  # mm/s
sec2['bulk_velocity_mm_s'] = vb
sec2['bulk_distance_per_mean_life_cm'] = [v * 0.1 * tau_pb for v in vb]
# ion drift of charged daughters in the 96.5 V/cm drift field
vi = [mu * lz.DRIFT_FIELD_VCM * 10 for mu in RECALLED['ion_mobility_cm2_Vs']]  # cm/s -> mm/s
sec2['ion_drift_velocity_mm_s'] = vi
sec2['ion_drift_distance_per_mean_life_cm'] = [v * 0.1 * tau_pb for v in vi]
sec2['TPC_drift_length_cm'] = RECALLED['TPC_drift_cm']
R['sec2_chain_transport'] = sec2

# ================================================================== 3. ROI ER window, beta acceptance, activity
ME = 510.998950  # keV
ALPHA = 1 / 137.035999
Z_DAUGHTER = 83

def fermi_F(T):
    Et = T + ME
    p = np.sqrt(Et ** 2 - ME ** 2)
    eta = ALPHA * Z_DAUGHTER * Et / p
    return 2 * np.pi * eta / (1 - np.exp(-2 * np.pi * eta))

def beta_shape(T, Q):
    T = np.asarray(T, float)
    Et = T + ME
    p = np.sqrt(np.maximum(Et ** 2 - ME ** 2, 0))
    out = fermi_F(np.maximum(T, 1e-6)) * p * Et * (Q - T) ** 2
    return np.where((T > 0) & (T < Q), out, 0.0)

def beta_fraction(Q, lo, hi):
    tot = integrate.quad(beta_shape, 0, Q, args=(Q,), limit=200)[0]
    part = integrate.quad(beta_shape, lo, min(hi, Q), args=(Q,), limit=200)[0]
    return part / tot

def s1c_of_E(E):
    return lz.LZ['g1'] * lz.nest_er_yields(E)[0]

def log10s2c_of_E(E):
    return math.log10(lz.LZ['g2'] * lz.nest_er_yields(E)[1])

E_lo = optimize.brentq(lambda E: s1c_of_E(E) - lz.LZ['S1c_min'], 0.3, 5.0)
E_hi = optimize.brentq(lambda E: log10s2c_of_E(E) - lz.LZ['log10S2c_max'], 5.0, 60.0)
E_hi_HESB = optimize.brentq(lambda E: log10s2c_of_E(E) - 4.30, 5.0, 80.0)
E_S2raw = optimize.brentq(lambda E: lz.LZ['g2'] * lz.nest_er_yields(E)[1] - lz.LZ['S2_raw_min'], 0.05, 5.0)
sec3 = dict(ROI_ER_window_keV=(E_lo, E_hi), E_of_S2raw_645=E_S2raw, HESB_beta_window_keV=(E_lo, E_hi_HESB))

Q = RECALLED['Pb214_Q_keV']
lev = RECALLED['Pb214_levels_keV']
endpoints = {'gs': Q, '352': Q - lev['352'], '295': Q - lev['295'], '53': Q - lev['53']}
sec3['beta_endpoints_keV'] = endpoints
acc = {}
for br, Qb in endpoints.items():
    acc[br] = dict(ROI=beta_fraction(Qb, E_lo, E_hi), vertex_10_14=beta_fraction(Qb, 10, 14),
                   HESB=beta_fraction(Qb, E_lo, E_hi_HESB), flat_ROI=(E_hi - E_lo) / Qb)
sec3['beta_acceptance'] = acc
# uniform-in-energy comparison and Fermi enhancement at low energies
sec3['fermi_enhancement_ROI_vs_flat'] = acc['gs']['ROI'] / acc['gs']['flat_ROI']

# gammas per decay with the endpoint of the feeding branch
GAM = {'352': dict(E=351.9, I=RECALLED['Pb214_gamma_I']['352'], endpoint=endpoints['352']),
       '295': dict(E=295.2, I=RECALLED['Pb214_gamma_I']['295'], endpoint=endpoints['295']),
       '242': dict(E=242.0, I=RECALLED['Pb214_gamma_I']['242'], endpoint=endpoints['295'])}

# ================================================================== 4. gamma ray-tracing MC in the TPC cylinder
RT = RECALLED['TPC_radius_cm']; H = RECALLED['TPC_drift_cm']; SHELL = 0.3; RFR = 13.75
rng = np.random.default_rng(53)

def trace(r0, z0, ndir=200000):
    """Isotropic rays from (r0, 0, z0). Returns dict of arrays: path length L to the first active-volume
    boundary (inner face of wall shell at RT-SHELL, cathode z=0, gate z=H), which boundary, and the
    extra path inside the dead region (shell up to RT, or RFR up to 13.75 cm below the cathode)."""
    u = rng.uniform(-1, 1, ndir); phi = rng.uniform(0, 2 * np.pi, ndir)
    s = np.sqrt(1 - u ** 2)
    dx, dy, dz = s * np.cos(phi), s * np.sin(phi), u
    # cylinder r = Rin
    def t_cyl(Rc):
        a = dx ** 2 + dy ** 2
        b = 2 * r0 * dx
        c = r0 ** 2 - Rc ** 2
        disc = b ** 2 - 4 * a * c
        t = (-b + np.sqrt(np.maximum(disc, 0))) / (2 * a)
        return t
    t_in = t_cyl(RT - SHELL); t_out = t_cyl(RT)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_cath = np.where(dz < 0, -z0 / dz, np.inf)
        t_gate = np.where(dz > 0, (H - z0) / dz, np.inf)
    L = np.minimum.reduce([t_in, t_cath, t_gate])
    which = np.argmin(np.vstack([t_in, t_cath, t_gate]), axis=0)  # 0 wall, 1 cathode, 2 gate
    dead_path = np.zeros(ndir)
    m = which == 0
    dead_path[m] = np.minimum(t_out[m] - t_in[m], 1e3)  # path inside 3 mm shell (grazing rays capped)
    m = which == 1
    dead_path[m] = np.minimum(RFR / (-dz[m]), 1e3)       # path inside RFR before reaching the bottom grid
    return dict(L=L, which=which, dead=dead_path)

def geom_probs(tr, lam):
    surv = np.exp(-tr['L'] / lam)
    wall = tr['which'] == 0; cath = tr['which'] == 1; gate = tr['which'] == 2
    p_int_dead = 1 - np.exp(-tr['dead'] / lam)
    return dict(reach_wall=float(np.mean(surv * wall)), reach_rfr=float(np.mean(surv * cath)),
                reach_gate=float(np.mean(surv * gate)), escape_any=float(np.mean(surv)),
                int_wall_shell=float(np.mean(surv * wall * p_int_dead)),
                int_rfr=float(np.mean(surv * cath * p_int_dead)))

def plane_analytic(d, lam):
    """fraction of isotropic emission reaching an infinite plane at distance d unscattered = E2(d/lam)/2"""
    return 0.5 * special.expn(2, d / lam)

sec4 = {}
# validation: point far from the wall, close to the cathode: MC reach_rfr vs analytic half-space
tr_val = trace(0.0, 5.0, 400000)
sec4['validation_plane'] = {k: dict(mc=geom_probs(tr_val, v)['reach_rfr'], analytic=plane_analytic(5.0, v))
                            for k, v in RECALLED['lambda_cm'].items()}

# key points
r_ev = RT - lz.LZ['ev_r_from_true_wall_cm']; z_ev = lz.LZ['ev_z_above_cathode_cm']
points = {'event (r=45.9, z=26.4)': (r_ev, z_ev),
          'FV wall edge, event height (r=64.8, z=26.4)': (RT - RECALLED['FV_wall_margin_cm'], z_ev),
          'FV bottom, event radius (r=45.9, z=9.0)': (r_ev, RECALLED['FV_zmin_cm']),
          'reco-wall distance 23.4 cm (r=49.4, z=26.4)': (RT - lz.LZ['ev_r_from_reco_wall_cm'], z_ev),
          'FV corner (r=64.8, z=9.0)': (RT - RECALLED['FV_wall_margin_cm'], RECALLED['FV_zmin_cm']),
          '5.4t bottom, event radius (r=45.9, z=2.0)': (r_ev, 2.0)}
pt = {}
for name, (r0, z0) in points.items():
    tr = trace(r0, z0, 400000)
    pt[name] = {k: geom_probs(tr, v) for k, v in RECALLED['lambda_cm'].items()}
    # attenuation brackets for the 352 line
    pt[name]['352_bracket'] = [geom_probs(tr, v)['int_wall_shell'] + geom_probs(tr, v)['int_rfr'] for v in RECALLED['lambda_bracket']['352']]
    pt[name]['straight_line_exp'] = {k: dict(wall=math.exp(-(RT - SHELL - r0) / v), rfr=math.exp(-z0 / v)) for k, v in RECALLED['lambda_cm'].items()}
sec4['points'] = pt

# FV average (uniform 214Pb): grid in r (weight r) and z
def volume_average(r_max, z_min, z_max, nr=14, nz=14, ndir=20000):
    rs = np.linspace(0.5, r_max - 0.5, nr); zs = np.linspace(z_min + 0.5, z_max - 0.5, nz)
    acc_ = {k: dict(int_wall_shell=0.0, int_rfr=0.0, escape_any=0.0, reach_rfr=0.0) for k in RECALLED['lambda_cm']}
    wsum = 0.0
    for r0 in rs:
        for z0 in zs:
            tr = trace(r0, z0, ndir)
            for k, v in RECALLED['lambda_cm'].items():
                g = geom_probs(tr, v)
                for q in acc_[k]:
                    acc_[k][q] += r0 * g[q]
            wsum += r0
    for k in acc_:
        for q in acc_[k]:
            acc_[k][q] /= wsum
    vol_m3 = math.pi * (r_max / 100) ** 2 * (z_max - z_min) / 100
    return acc_, vol_m3

z_min = RECALLED['FV_zmin_cm']; z_max = H - RECALLED['FV_ztop_margin_cm']
fv_avg, fv_vol = volume_average(RT - RECALLED['FV_wall_margin_mean_cm'], z_min, z_max)          # mean stand-off 10.7 cm
fv_avg8, fv_vol8 = volume_average(RT - RECALLED['FV_wall_margin_cm'], z_min, z_max)              # min stand-off 8 cm (bracket)
v54_avg, v54_vol = volume_average(RT - RECALLED['vol54_wall_margin_cm'], RECALLED['vol54_zmin_cm'][1], z_max)   # 5.4 t, z_min 5 cm
v54_avg2, v54_vol2 = volume_average(RT - RECALLED['vol54_wall_margin_cm'], RECALLED['vol54_zmin_cm'][0], z_max)  # 5.4 t, z_min 2 cm
sec4['FV_average'] = dict(probs=fv_avg, cylinder_volume_m3=fv_vol, cylinder_mass_t=fv_vol * RECALLED['LXe_density'],
                          probs_wall8=fv_avg8, cylinder_mass_t_wall8=fv_vol8 * RECALLED['LXe_density'])
sec4['vol54_average'] = dict(probs_zmin5=v54_avg, cylinder_mass_t_zmin5=v54_vol * RECALLED['LXe_density'],
                             probs_zmin2=v54_avg2, cylinder_mass_t_zmin2=v54_vol2 * RECALLED['LXe_density'])

# ---- 214Pb activity implied by Table I --------------------------------------------------
# N_ROI(214Pb) = N_dec * [BR_gs*A_gs + sum_exc feed*A_exc*P_escape_gamma*(1-veto share)]
def n_decays_per_run(N_roi_pb, BR_gs, escape_weight=0.5):
    exc = 0.0
    for key, g in GAM.items():
        # escaped-gamma naked-like betas: use escape probability averaged over the FV; only half are assumed
        # to evade the Skin/OD prompt veto (escape_weight), bracketed 0..1 below
        exc += g['I'] * beta_fraction(g['endpoint'], E_lo, E_hi) * fv_avg[key]['escape_any'] * escape_weight
    per_decay = BR_gs * acc['gs']['ROI'] + exc
    return N_roi_pb / per_decay, per_decay, exc

act = {}
for share in RECALLED['Pb214_share_of_internal_beta']:
    for BRname, BR in [('BR9.2', RECALLED['Pb214_BR_gs']), ('BR11', RECALLED['Pb214_BR_gs_alt'])]:
        for ew in (0.0, 0.5, 1.0):
            Nd, pd, exc = n_decays_per_run(N_INT_BETA * share, BR, ew)
            act[f'share{share}_{BRname}_esc{ew}'] = dict(N_decays_run=Nd, per_decay_ROI_prob=pd, excited_term=exc,
                                                        decays_per_day=Nd / LIVE_D,
                                                        uBq_per_kg=Nd / (LIVE_D * 86400) / FV_KG * 1e6)
sec3['activity_grid'] = act
central = act['share0.7_BR9.2_esc0.5']
sec3['activity_central'] = central
vals = np.array([a['uBq_per_kg'] for a in act.values()])
sec3['activity_range_uBq_kg'] = (float(vals.min()), float(vals.max()))
sec3['recalled_LZ_Rn222_uBq_kg'] = RECALLED['Rn222_LZ_uBq_kg']
sec3['Pb214_ROI_ER_per_day'] = {str(s): N_INT_BETA * s / LIVE_D for s in RECALLED['Pb214_share_of_internal_beta']}
sec3['total_ROI_ER_per_day'] = N_ROI_SCI / LIVE_D
R['sec3_activity'] = sec3

# ---- MSSI expectations from our own model -----------------------------------------------
def kn_fraction_below(Egamma, Tmax):
    """Klein-Nishina fraction of Compton scatters depositing electron energy <= Tmax (keV)."""
    k = Egamma / ME
    def dsig(theta):
        eps = 1 / (1 + k * (1 - np.cos(theta)))
        return eps ** 2 * (eps + 1 / eps - np.sin(theta) ** 2) * np.sin(theta)
    tot = integrate.quad(dsig, 0, np.pi)[0]
    # T(theta) = E - E' ; find theta where T = Tmax
    def T(theta):
        return Egamma * (1 - 1 / (1 + k * (1 - np.cos(theta))))
    if T(np.pi) <= Tmax:
        return 1.0
    th = optimize.brentq(lambda t: T(t) - Tmax, 1e-6, np.pi)
    return integrate.quad(dsig, 0, th)[0] / tot

Nd_47 = central['N_decays_run']
mssi = {}
# WS-ROI wall MSSI (4.7 t): beta in ROI window, gamma reaches the shell unscattered, Compton in shell with
# T_e <= ~100 keV (second S1 <= ~600 phd at bulk g1), scattered gamma leaves outward (0.8), prompt veto survives 6%
wall_roi = 0.0; wall_roi_raw = 0.0
for key, g in GAM.items():
    fC = RECALLED['compton_fraction'][key] * kn_fraction_below(g['E'], 100.0) * 0.8
    term = Nd_47 * g['I'] * beta_fraction(g['endpoint'], E_lo, E_hi) * fv_avg[key]['int_wall_shell'] * fC
    wall_roi_raw += term
    wall_roi += term * (1 - lz.LZ['MSSI_veto_eff'][0])
mssi['wall_WSROI_4p7t_science_Pb214'] = wall_roi
mssi['wall_WSROI_4p7t_before_veto_Pb214'] = wall_roi_raw
mssi['LZ_wall_WSROI_4p7t_science_all_sources'] = 0.0048
mssi['Pb214_share_estimate_wall'] = min(wall_roi / 0.0048, 1.0)
# HE-SB RFR (5.4 t): beta in E_lo..E_hi_HESB, gamma reaches RFR unscattered and deposits its full energy there
# (RFR light yield ~2.3 phd/keV from LZ's 204 keV <-> 471 phd: 352 keV -> ~810 phd, at the 800 phd edge;
#  variant A: only the 352 line counts; variant B: all three lines count)
def rfr_full(key, g, Nd, avg):
    return Nd * g['I'] * beta_fraction(g['endpoint'], E_lo, E_hi_HESB) * avg[key]['int_rfr'] * 0.8
for tag, avg, vol in (('zmin5', v54_avg, v54_vol), ('zmin2', v54_avg2, v54_vol2)):
    Nd = Nd_47 * vol / fv_vol
    mssi[f'HESB_RFR_5p4t_Pb214_352only_{tag}'] = rfr_full('352', GAM['352'], Nd, avg)
    mssi[f'HESB_RFR_5p4t_Pb214_all_lines_{tag}'] = sum(rfr_full(k, g, Nd, avg) for k, g in GAM.items())
# same quantity for the 4.7 t FV alone (z_min = 9 cm): LZ predicts 0.5 HE-SB RFR events there
mssi['HESB_RFR_4p7t_Pb214_352only'] = rfr_full('352', GAM['352'], Nd_47, fv_avg)
mssi['HESB_RFR_4p7t_Pb214_all_lines'] = sum(rfr_full(k, g, Nd_47, fv_avg) for k, g in GAM.items())
mssi['HESB_RFR_4p7t_LZ_predicted'] = 0.5
# WS-ROI RFR MSSI in the 4.7 t (LZ: 0.0001): needs a gamma Compton in the RFR with the scattered photon escaping
rfr_roi = 0.0
for key, g in GAM.items():
    fC = RECALLED['compton_fraction'][key] * kn_fraction_below(g['E'], 250.0) * 0.2   # escape from 13.75 cm RFR unlikely (0.2, uncertain)
    rfr_roi += Nd_47 * g['I'] * beta_fraction(g['endpoint'], E_lo, E_hi) * fv_avg[key]['int_rfr'] * fC
mssi['RFR_WSROI_4p7t_Pb214'] = rfr_roi; mssi['LZ_RFR_WSROI_4p7t'] = 0.0001
mssi['HESB_RFR_5p4t_LZ_predicted'] = 21.5; mssi['HESB_RFR_5p4t_LZ_observed'] = 18
mssi['RFR_light_yield_phd_per_keV'] = 471 / 204; mssi['S1c_352keV_in_RFR_phd'] = 351.9 * 471 / 204
mssi['N_decays_4p7t_run'] = Nd_47; mssi['N_decays_5p4t_run_zmin5'] = Nd_47 * v54_vol / fv_vol
# note: the 5.4 t sample in LZ's table is the annulus outside the 4.7 t FV only; our volume average is for the
# full 5.4 t cylinder. RFR MSSI is dominated by decays low in the TPC and only weakly depends on radius.
sec4['mssi_expectations'] = mssi
R['sec4_geometry'] = sec4

# ================================================================== 5. elevation factors and rate constraints
sec5 = {}
mu_nb_all = 1.7e-4          # P004: MSSI expectation within +-2 sigma_NR at S1c>500 phd (all MSSI, k=1)
k_req_P004 = 619.0
shares = dict(low=0.3, central=0.5, high=0.7, ours=max(0.1, min(0.9, mssi['Pb214_share_estimate_wall'])))
sec5['Pb214_share_used'] = shares
P_target = 0.10; mu_target = -math.log(1 - P_target)
flat = {name: k_req_P004 / s for name, s in shares.items()}
sec5['k_flat_required_10pct'] = flat
windows_h = [1, 3, 6, 12, 24, 72, 24 * 7]
trans = {}
for name, s in shares.items():
    mu_pb = mu_nb_all * s
    trans[name] = {str(T): 1 + mu_target / mu_pb * (LIVE_D * 24 / T) for T in windows_h}
sec5['k_transient_required_10pct'] = trans
# position penalty: per-decay probability of the 12 keV-vertex + dead-region-gamma topology at the event vs FV average
def topo_prob(probs):
    p = 0.0
    for key, g in GAM.items():
        p += g['I'] * beta_fraction(g['endpoint'], 10, 14) * (probs[key]['int_wall_shell'] + probs[key]['int_rfr'])
    return p
p_ev = topo_prob(pt['event (r=45.9, z=26.4)']); p_fv = topo_prob(fv_avg); p_edge = topo_prob(pt['FV wall edge, event height (r=64.8, z=26.4)'])
p_ev_brkt = [sum(GAM[k]['I'] * beta_fraction(GAM[k]['endpoint'], 10, 14) * pt['event (r=45.9, z=26.4)']['352_bracket'][i] for k in ['352']) for i in (0, 1)]
p_fv8 = topo_prob(fv_avg8)
sec5['topology_prob_per_decay'] = dict(event=p_ev, FV_average=p_fv, FV_average_wall8=p_fv8, FV_wall_edge=p_edge, event_352only_lambda_bracket=p_ev_brkt)
sec5['position_penalty_event_over_FVavg'] = p_ev / p_fv
sec5['position_penalty_event_over_FVavg_wall8'] = p_ev / p_fv8
sec5['position_penalty_event_over_edge'] = p_ev / p_edge
# lambda bracket on the penalty: 352-only event probability bracket over the 352-only FV average (lambda_central)
p_fv_352 = GAM['352']['I'] * beta_fraction(GAM['352']['endpoint'], 10, 14) * (fv_avg['352']['int_wall_shell'] + fv_avg['352']['int_rfr'])
sec5['position_penalty_352only_lambda_bracket'] = [b / p_fv_352 for b in p_ev_brkt]
# mixed-flow-only consistency: 6 HE-SB RFR events in non-tag-active periods vs 21.5 f_m expected
mf = {}
for fm in (0.1, 0.2, 0.33, 0.5):
    mu = 21.5 * fm
    # 95% upper limit on an enhancement factor eps in mixed flow: P(N<=6 | eps*mu) = 0.05
    eps95 = optimize.brentq(lambda e: stats.poisson.cdf(6, e * mu) - 0.05, 1e-3, 1e3)
    mf[str(fm)] = dict(expected=mu, observed=6, p_ge6=float(stats.poisson.sf(5, mu)), p_le6=float(stats.poisson.cdf(6, mu)), eps_ML=6 / mu, eps_95UL=eps95)
sec5['mixed_flow_HESB_consistency'] = mf
# expected number of such topologies at the event's shell (within +-2 cm in r and z) per run at k=1
sec5['expected_12keV_topologies_per_run_whole_FV'] = Nd_47 * p_fv
sec5['expected_12keV_topologies_per_run_event_position_perdecay_x_Ndecays'] = Nd_47 * p_ev

# ER-rate detectability: minimum elevation k of the 214Pb activity over a window T detectable at 3 and 5 sigma
b_rate = N_ROI_SCI / LIVE_D / 24  # total ROI ER per hour
er = {}
for share in RECALLED['Pb214_share_of_internal_beta']:
    r_pb = N_INT_BETA * share / LIVE_D / 24
    er[str(share)] = {}
    for T in windows_h:
        b = b_rate * T
        row = {}
        for sig, pval in (('3sigma', 1.3499e-3), ('5sigma', 2.8665e-7)):
            n = int(stats.poisson.isf(pval, b)) + 1
            while stats.poisson.sf(n - 1, b) > pval:
                n += 1
            extra = n - b
            row[sig] = dict(n_needed=n, k_min=1 + extra / (r_pb * T))
        row['baseline_events'] = b
        row['extra_events_k10'] = 9 * r_pb * T
        er[str(share)][str(T)] = row
sec5['ER_rate_detectability'] = er
# alpha monitor: 218Po and 214Po alphas in the whole active LXe
active_kg = 7000.0
A_rn_Bq = central['uBq_per_kg'] * 1e-6 * active_kg
sec5['alpha_rate_per_hour_per_chain_member_active7t'] = A_rn_Bq * 3600
sec5['minutes_to_5sigma_for_k'] = {str(k): 25 / (A_rn_Bq * (k - 1) ** 2) / 60 for k in (1.5, 2, 3, 10)}
# gap between required and allowed
sec5['gap_required_over_allowed'] = {}
for T in (1, 24):
    kreq = trans['central'][str(T)] * (1 / sec5['position_penalty_event_over_FVavg'])
    kal = er['0.7'][str(T)]['3sigma']['k_min']
    sec5['gap_required_over_allowed'][str(T)] = dict(k_required_with_position=kreq, k_required_no_position=trans['central'][str(T)], k_allowed_3sigma=kal,
                                                     ratio_with_position=kreq / kal, ratio_no_position=trans['central'][str(T)] / kal)
R['sec5_elevation'] = sec5

# ================================================================== 6. tag counterfactual
sec6 = {}
eps_c = 0.6; eps_lo, eps_hi = cp_interval(7, 12, 0.68)
for f_false in RECALLED['false_tag_rates']:
    sec6[f'false_tag_{f_false}'] = dict(LR_negative=(1 - eps_c) / (1 - f_false), LR_negative_eps_range=((1 - eps_hi) / (1 - f_false), (1 - eps_lo) / (1 - f_false)),
                                        LR_positive=eps_c / f_false, LR_positive_eps_range=(eps_lo / f_false, eps_hi / f_false))
# expected information (nats) about H1=214Pb vs H0 from the binary tag, under each hypothesis
f_false = 0.03
KL_H1 = eps_c * math.log(eps_c / f_false) + (1 - eps_c) * math.log((1 - eps_c) / (1 - f_false))
KL_H0 = f_false * math.log(f_false / eps_c) + (1 - f_false) * math.log((1 - f_false) / (1 - eps_c))
sec6['expected_lnLR'] = dict(under_Pb214=KL_H1, under_not_Pb214=-KL_H0, f_false_used=f_false)
# posterior for illustrative priors (P004: wall MSSI k=1.64 neighbourhood 2.8e-4 -> P(MSSI|event) small)
for prior in (1e-3, 1e-2, 0.1):
    o = prior / (1 - prior)
    sec6[f'posterior_prior{prior}'] = dict(after_negative=(o * 0.4 / 0.97) / (1 + o * 0.4 / 0.97), after_positive=(o * 20) / (1 + o * 20))
R['sec6_tag'] = sec6

# ================================================================== figures
C1, C2, C3, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#1baf7a', '#0b0b0b', '#898781', '#e1e0d9'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})

# Fig 1: probability that a 214Pb gamma reaches a dead region unscattered vs distance (half-space analytic), with points
d = np.linspace(1, 40, 400)
fig, ax = plt.subplots(figsize=(6.2, 3.8), facecolor='#fcfcfb'); ax.set_facecolor('#fcfcfb')
for (key, lam), col in zip(RECALLED['lambda_cm'].items(), (C1, C2, C3)):
    ax.plot(d, plane_analytic(d, lam), color=col, lw=2, label=f'{key} keV, lambda = {lam} cm')
    pass
ax.axvline(RECALLED['FV_wall_margin_cm'], color=MUTED, lw=1, ls='--')
ax.text(8.4, 3e-8, 'FV wall\nmargin 8 cm', color=MUTED, fontsize=7.5, ha='left', va='bottom')
ax.axvline(z_ev, color=INK, lw=1, ls=':')
ax.text(26.9, 3e-8, 'event: 26.4 cm to cathode\n26.6 cm to wall shell', color=INK, fontsize=7.5, ha='left', va='bottom')
ev352 = pt['event (r=45.9, z=26.4)']['352']
ax.plot([z_ev], [ev352['reach_rfr'] + ev352['reach_wall']], 'o', color=INK, ms=6, zorder=5)
ax.annotate('ray-traced event\n(352 keV, wall+RFR)', xy=(z_ev, ev352['reach_rfr'] + ev352['reach_wall']), xytext=(15, 3e-6),
            fontsize=7.5, color=INK, arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
ax.set_yscale('log'); ax.set_ylim(1e-8, 1); ax.set_xlim(0, 40)
ax.set_xlabel('distance from decay to the charge-dead region (cm of LXe)'); ax.set_ylabel('P(gamma reaches dead region unscattered)')
ax.set_title('214Pb gamma reaching a charge-dead region: half-space survival E2(d/lambda)/2', fontsize=9, color=INK)
ax.grid(color=GRID, lw=0.6); ax.legend(frameon=False, fontsize=8, loc='upper right')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P053_fig1_gamma_reach.png'), dpi=160); plt.close(fig)

# Fig 2: elevation factor required vs window, against ER-rate and alpha-rate detectability
fig, ax = plt.subplots(figsize=(6.2, 3.8), facecolor='#fcfcfb'); ax.set_facecolor('#fcfcfb')
T = np.array(windows_h, float)
ax.plot(T, [trans['central'][str(int(t))] for t in windows_h], color=C1, lw=2, marker='o', ms=4, label='k required (10% explanation, 214Pb share 0.5, no position penalty)')
ax.plot(T, [trans['central'][str(int(t))] / sec5['position_penalty_event_over_FVavg'] for t in windows_h], color=C1, lw=2, ls='--', marker='o', ms=4, label='k required incl. position penalty at 27 cm')
ax.plot(T, [er['0.7'][str(int(t))]['3sigma']['k_min'] for t in windows_h], color=C2, lw=2, marker='s', ms=4, label='k detectable at 3 sigma from the ROI ER rate (214Pb 70% of 1341)')
ax.plot(T, [er['0.7'][str(int(t))]['5sigma']['k_min'] for t in windows_h], color=C2, lw=2, ls='--', marker='s', ms=4, label='k detectable at 5 sigma (ER rate)')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('duration of the transient 214Pb elevation (h)'); ax.set_ylabel('elevation factor k')
ax.set_ylim(0.5, 1e14)
ax.grid(color=GRID, lw=0.6); ax.legend(frameon=False, fontsize=7.2, loc='upper right')
ax.set_title('Transient 214Pb elevation: required for a 10% MSSI explanation vs detectable in the ER rate', fontsize=8.5, color=INK)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P053_fig2_elevation_vs_window.png'), dpi=160); plt.close(fig)

# ================================================================== write
def conv(o):
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, tuple): return list(o)
    raise TypeError(str(type(o)))
with open(os.path.join(OUT, 'P053_results.json'), 'w') as f:
    json.dump(R, f, indent=1, default=conv)

# compact CSV of the geometry table
import csv
with open(os.path.join(OUT, 'P053_geometry_points.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['point', 'gamma_keV', 'reach_wall', 'reach_rfr', 'int_wall_shell', 'int_rfr', 'escape_any', 'straight_wall', 'straight_rfr'])
    for name, dct in pt.items():
        for key in RECALLED['lambda_cm']:
            g = dct[key]; s = dct['straight_line_exp'][key]
            w.writerow([name, key, f"{g['reach_wall']:.3e}", f"{g['reach_rfr']:.3e}", f"{g['int_wall_shell']:.3e}", f"{g['int_rfr']:.3e}", f"{g['escape_any']:.3e}", f"{s['wall']:.3e}", f"{s['rfr']:.3e}"])
    for name, dct in (('FV average (10.7 cm, z>9)', fv_avg), ('FV average (8 cm, z>9)', fv_avg8), ('5.4t average (z>5)', v54_avg), ('5.4t average (z>2)', v54_avg2)):
        for key in RECALLED['lambda_cm']:
            g = dct[key]; w.writerow([name, key, '', f"{g['reach_rfr']:.3e}", f"{g['int_wall_shell']:.3e}", f"{g['int_rfr']:.3e}", f"{g['escape_any']:.3e}", '', ''])

# ================================================================== print summary
print('=== 1. tag-active fraction')
for k_, v in sec1.items():
    print(k_, json.dumps(v, default=conv))
print('\n=== 2. chain')
print(json.dumps({k: sec2[k] for k in ['half_life_s', 'mean_delay_Po218_alpha_to_Pb214_beta_min', 'bulk_velocity_mm_s', 'bulk_distance_per_mean_life_cm', 'ion_drift_velocity_mm_s', 'ion_drift_distance_per_mean_life_cm', 'Pb214_activity_after_1Bq_Rn222_Bq']}, default=conv, indent=0))
print('transport per mean life (cm):', {k: round(v['per_mean_life'], 1) for k, v in sec2['transport_distance_cm'].items()})
print('\n=== 3. ROI window and activity')
print('ROI ER window keV', sec3['ROI_ER_window_keV'], 'HE-SB beta window', sec3['HESB_beta_window_keV'], 'E(S2raw=645)', E_S2raw)
print('acceptances', json.dumps(acc, indent=0, default=conv))
print('Fermi enhancement', sec3['fermi_enhancement_ROI_vs_flat'])
print('central activity', json.dumps(central, default=conv)); print('activity range uBq/kg', sec3['activity_range_uBq_kg'])
print('\n=== 4. geometry')
print('validation', json.dumps(sec4['validation_plane'], default=conv))
for name, dct in pt.items():
    print(name);
    for key in RECALLED['lambda_cm']:
        print('   ', key, {k: f'{v:.3e}' for k, v in dct[key].items()}, 'straight', {k: f'{v:.2e}' for k, v in dct['straight_line_exp'][key].items()})
    print('    352 lambda bracket', dct['352_bracket'])
print('FV avg (10.7 cm)', {k: {q: f'{v:.3e}' for q, v in fv_avg[k].items()} for k in fv_avg}, 'mass', sec4['FV_average']['cylinder_mass_t'])
print('FV avg (8 cm)', {k: {q: f'{v:.3e}' for q, v in fv_avg8[k].items()} for k in fv_avg8}, 'mass', sec4['FV_average']['cylinder_mass_t_wall8'])
print('5.4t avg zmin5', {k: {q: f'{v:.3e}' for q, v in v54_avg[k].items()} for k in v54_avg}, 'mass', sec4['vol54_average']['cylinder_mass_t_zmin5'])
print('5.4t avg zmin2', {k: {q: f'{v:.3e}' for q, v in v54_avg2[k].items()} for k in v54_avg2}, 'mass', sec4['vol54_average']['cylinder_mass_t_zmin2'])
print('MSSI expectations', json.dumps(mssi, default=conv, indent=0))
print('\n=== 5. elevation')
print(json.dumps({k: sec5[k] for k in ['Pb214_share_used', 'k_flat_required_10pct', 'topology_prob_per_decay', 'position_penalty_event_over_FVavg', 'position_penalty_event_over_FVavg_wall8', 'position_penalty_352only_lambda_bracket', 'position_penalty_event_over_edge', 'mixed_flow_HESB_consistency', 'expected_12keV_topologies_per_run_whole_FV', 'alpha_rate_per_hour_per_chain_member_active7t', 'minutes_to_5sigma_for_k', 'gap_required_over_allowed']}, default=conv, indent=0))
print('k transient central', trans['central'])
for T in windows_h:
    print(f"T={T}h baseline {er['0.7'][str(T)]['baseline_events']:.2f}  3sig n={er['0.7'][str(T)]['3sigma']['n_needed']} k_min={er['0.7'][str(T)]['3sigma']['k_min']:.2f}  5sig n={er['0.7'][str(T)]['5sigma']['n_needed']} k_min={er['0.7'][str(T)]['5sigma']['k_min']:.2f}  extra(k=10)={er['0.7'][str(T)]['extra_events_k10']:.1f}")
print('\n=== 6. tag')
print(json.dumps(sec6, default=conv, indent=0))
