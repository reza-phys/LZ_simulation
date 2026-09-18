#!/usr/bin/env python
"""P064: Nuclear-recoil quenching at 250 keV -- Lindhard theory, the NEST power law and the
energy-scale systematic of the LZ 248 keV event.

Run from the simulation root:  .venv/bin/python output/code/P064_quenching.py

Sections
  1. Lindhard L(E) for xenon; Nq_Lindhard = L E / W; local exponent beta_eff; comparison with the
     NEST power laws (Table S5 as printed, LZ contour scale from P009, NEST default).
  2. Analytic NEST v2 NR yield model (validated against nestpy via lzcommon.nest_nr_yields).
  3. Energy-scale systematic at 250 keV: event energy under each model; anchored slope variations
     (beta 1.05-1.15 at E0 = 74.7 keV); Lindhard k +-20 %; "kink" variants; comparison with +-23 keV.
  4. Light/charge partition: Qy with/without the p(E) break vs the Thomas-Imel box model; electron
     fraction; band slope; the Ly-scale <-> (a,b) degeneracy in the AmBe band.
Outputs: output/work/P064/*.csv, results.json, figures/*.png
"""
import sys, os, json, math
sys.path.insert(0, 'output/code')
import numpy as np
from scipy.optimize import brentq, least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import lzcommon as lz

OUT = 'output/work/P064'
FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)

# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------
Z_XE, A_XE = 54, lz.A_XE_MEAN              # 131.29
FIELD = lz.DRIFT_FIELD_VCM                 # 96.5 V/cm (nestpy LZ_WS2024)
W_LIST = [13.7, 13.5]                      # eV per quantum (recalled: NEST v2 13.7; 13.5 also used; likely)
NQ_EVENT = lz.LZ['ev_S1c'] / lz.LZ['g1'] + lz.LZ['ev_S2c'] / lz.LZ['g2']   # 5179
NPH_EVENT = lz.LZ['ev_S1c'] / lz.LZ['g1']                                  # 4910
NE_EVENT = lz.LZ['ev_S2c'] / lz.LZ['g2']                                   # 268.6
E0 = 74.7
# P009 contour scale (digitised Fig. 4 labels): Nq = 11.3245 E^1.11167, W = 13.46 eV
ALPHA_P, BETA_P, W_P = 11.324544769218303, 1.1116662817198515, 13.46275612158634
NEST_CONTOUR = dict(lz.NEST_NR_LZ); NEST_CONTOUR.update(alpha=ALPHA_P, beta=BETA_P)
# Lindhard k values
K_LINDHARD = 0.133 * Z_XE ** (2 / 3) * A_XE ** (-0.5)      # 0.166 (Lindhard 1963; certain)
K_LENARDO = 0.1394   # recalled: Lenardo et al. 2015 global-fit Lindhard k with W = 13.7 eV (likely)
K_LUXDD = 0.174      # recalled: LUX D-D 2016 total-quanta Lindhard fit k ~ 0.17 (uncertain)

# ---------------------------------------------------------------------------------------------
# 1. Lindhard
# ---------------------------------------------------------------------------------------------
def lindhard_L(E, k=K_LINDHARD, Z=Z_XE):
    """Lindhard quenching L(E) = k g(eps)/(1 + k g(eps)), eps = 11.5 E[keV] Z^-7/3,
    g = 3 eps^0.15 + 0.7 eps^0.6 + eps  (Lindhard 1963 / Lewin-Smith 1996 parametrisation; certain)."""
    E = np.asarray(E, float)
    eps = 11.5 * E * Z ** (-7.0 / 3.0)
    g = 3 * eps ** 0.15 + 0.7 * eps ** 0.6 + eps
    return k * g / (1 + k * g)

def nq_lindhard(E, k=K_LINDHARD, W=13.7):
    return lindhard_L(E, k) * np.asarray(E, float) * 1e3 / W

def beta_eff(fun, E, h=1e-3):
    E = np.asarray(E, float)
    return (np.log(fun(E * (1 + h))) - np.log(fun(E * (1 - h)))) / (np.log(1 + h) - np.log(1 - h))

# ---------------------------------------------------------------------------------------------
# 2. Analytic NEST v2 NR yields (structure recalled from the NEST source, likely; validated below)
# ---------------------------------------------------------------------------------------------
def p_of_E(E, P):
    p = np.full_like(np.asarray(E, float), P['p'])
    if 'E0' in P:
        E = np.asarray(E, float)
        m = E > P['E0']
        p = np.where(m, 0.5 + P['a'] * np.log(1 + P['b'] * np.clip(E - P['E0'], 0, None)), p)
    return p

def nest_nr(E, P=lz.NEST_NR_LZ, F=FIELD, rho=2.9, use_break=True):
    """Return dict(Nq, Nph, Ne, Qy, Ly, p, TI) for the NEST v2 NR model with parameters P."""
    E = np.asarray(E, float)
    Pl = dict(P)
    if not use_break:
        Pl.pop('E0', None)
    p = p_of_E(E, Pl)
    Nq = Pl['alpha'] * E ** Pl['beta']
    TI = Pl['gamma'] * F ** Pl['delta'] * (rho / 2.9) ** 0.3
    Qy = 1 / (TI * (E + Pl['epsilon']) ** p) * (1 - 1 / (1 + (E / Pl['zeta']) ** Pl['eta']) ** Pl['f1'])
    Ly = np.clip(Nq / E - Qy, 0, None)
    Ne = Qy * E
    Nph = Ly * E * (1 - 1 / (1 + (E / Pl['theta']) ** Pl['iota']) ** Pl['f2'])
    return dict(Nq=Nq, Nph=Nph, Ne=Ne, Qy=Qy, Ly=Nph / E, p=p, TI=TI)

# validation against nestpy
val = []
for E in [5, 20, 50, 74.7, 100, 150, 248, 300, 330]:
    for name, P in [('LZ', lz.NEST_NR_LZ), ('DEF', lz.NEST_NR_DEFAULT)]:
        a = nest_nr(E, P); n = lz.nest_nr_yields(E, params=P)
        val.append(dict(model=name, E=E, Nph_an=float(a['Nph']), Ne_an=float(a['Ne']), Nph_nest=n[0], Ne_nest=n[1],
                        dNph=float(a['Nph']) / n[0] - 1, dNe=float(a['Ne']) / n[1] - 1))
val_max = max(max(abs(v['dNph']), abs(v['dNe'])) for v in val)
print(f"[validation] analytic NEST vs nestpy: max |rel diff| = {val_max*100:.2f} %")
import pandas as pd
pd.DataFrame(val).to_csv(os.path.join(OUT, 'nestpy_validation.csv'), index=False)

# ---------------------------------------------------------------------------------------------
# Model set for Nq(E)
# ---------------------------------------------------------------------------------------------
E_grid = np.logspace(0, np.log10(400), 400)
E_tab = np.array([50, 74.7, 150, 248, 300])

models_nq = {
    'Lindhard k=0.166, W=13.7': lambda E: nq_lindhard(E, K_LINDHARD, 13.7),
    'Lindhard k=0.166, W=13.5': lambda E: nq_lindhard(E, K_LINDHARD, 13.5),
    'Lindhard k=0.133 (-20%), W=13.7': lambda E: nq_lindhard(E, 0.8 * K_LINDHARD, 13.7),
    'Lindhard k=0.199 (+20%), W=13.7': lambda E: nq_lindhard(E, 1.2 * K_LINDHARD, 13.7),
    'Lindhard k=0.139 (Lenardo15 fit), W=13.7': lambda E: nq_lindhard(E, K_LENARDO, 13.7),
    'Lindhard k=0.174 (LUX DD fit), W=13.7': lambda E: nq_lindhard(E, K_LUXDD, 13.7),
    'NEST Table S5 (11.2 E^1.1)': lambda E: 11.2 * np.asarray(E, float) ** 1.1,
    'LZ contour scale (11.32 E^1.112)': lambda E: ALPHA_P * np.asarray(E, float) ** BETA_P,
    'NEST default (11.0 E^1.1)': lambda E: 11.0 * np.asarray(E, float) ** 1.1,
}
rows = []
for name, f in models_nq.items():
    for E in E_tab:
        nq = float(f(E)); be = float(beta_eff(f, E))
        rows.append(dict(model=name, E_keV=E, Nq=nq, L_eff_W13p7=nq * 13.7 / (E * 1e3),
                         beta_eff=be, Nq_over_TableS5=nq / (11.2 * E ** 1.1)))
tab1 = pd.DataFrame(rows)
tab1.to_csv(os.path.join(OUT, 'table1_L_Nq_beta.csv'), index=False)
print("\n=== Table 1: L, Nq, beta_eff at 50/74.7/150/248/300 keV ===")
print(tab1.pivot(index='model', columns='E_keV', values='Nq').round(0).to_string())
print(tab1.pivot(index='model', columns='E_keV', values='beta_eff').round(3).to_string())
print(tab1.pivot(index='model', columns='E_keV', values='L_eff_W13p7').round(3).to_string())

# Lindhard-specific: L and beta_eff for k = 0.166 at more energies; slope 50-300 keV
L_50, L_300 = float(lindhard_L(50)), float(lindhard_L(300))
slope_L = math.log(L_300 / L_50) / math.log(300 / 50)
print(f"\nLindhard k={K_LINDHARD:.4f}: L(50)={L_50:.3f}, L(74.7)={float(lindhard_L(74.7)):.3f}, "
      f"L(248)={float(lindhard_L(248)):.3f}, L(300)={L_300:.3f}; mean d ln L/d ln E (50-300) = {slope_L:.3f}")
# implied W that would make Lindhard (k=0.166) match Table S5 / contour scale at 248 keV
W_match_S5 = float(lindhard_L(248)) * 248e3 / (11.2 * 248 ** 1.1)
W_match_P = float(lindhard_L(248)) * 248e3 / (ALPHA_P * 248 ** BETA_P)
# implied k (W = 13.7) that reproduces Table S5 / contour at 248 keV and at 74.7 keV
def k_for_nq(E, nq, W=13.7):
    return brentq(lambda k: nq_lindhard(E, k, W) - nq, 0.02, 1.0)
k_S5_248, k_S5_75 = k_for_nq(248, 11.2 * 248 ** 1.1), k_for_nq(74.7, 11.2 * 74.7 ** 1.1)
k_P_248, k_P_75 = k_for_nq(248, ALPHA_P * 248 ** BETA_P), k_for_nq(74.7, ALPHA_P * 74.7 ** BETA_P)
print(f"W making Lindhard(0.166) = Table S5 at 248 keV: {W_match_S5:.2f} eV; = contour: {W_match_P:.2f} eV")
print(f"k (W=13.7) reproducing Table S5: {k_S5_75:.4f} (74.7 keV), {k_S5_248:.4f} (248 keV); "
      f"contour: {k_P_75:.4f}, {k_P_248:.4f}")

# ---------------------------------------------------------------------------------------------
# 3. Energy-scale systematic at 250 keV
# ---------------------------------------------------------------------------------------------
def solve_E(fun, target, lo=50, hi=800):
    return brentq(lambda E: float(fun(E)) - target, lo, hi)

# 3a. Nq-based energy of the event under each model (combined-quanta estimator)
e_rows = []
for name, f in models_nq.items():
    E_nq = solve_E(f, NQ_EVENT)
    # S1-based: Nph(E) = Nq_model(E) - Ne_LZ(E) (LZ-tuned charge yield with break; P043 convention)
    E_s1 = solve_E(lambda E: f(E) - nest_nr(E)['Ne'], NPH_EVENT)
    e_rows.append(dict(model=name, E_comb_keV=E_nq, E_S1_keV=E_s1, Nq_at_248=float(f(248)),
                       beta_eff_248=float(beta_eff(f, 248))))
tab3 = pd.DataFrame(e_rows)
tab3.to_csv(os.path.join(OUT, 'table3_event_energy_by_model.csv'), index=False)
print("\n=== Table 3: event energy (Nq = 5179; Nph = 4910) under each model ===")
print(tab3.round(1).to_string(index=False))

# 3b. anchored slope variations: Nq(E) = Nq_S5(E0) (E/E0)^beta'  (D-D anchors the yields at <= 74 keV)
NQ_E0 = 11.2 * E0 ** 1.1
anch = []
for b in [1.00, 1.05, 1.08, 1.10, 1.1117, 1.12, 1.15, 1.20]:
    f = lambda E, b=b: NQ_E0 * (np.asarray(E, float) / E0) ** b
    E_nq = solve_E(f, NQ_EVENT); E_s1 = solve_E(lambda E: f(E) - nest_nr(E)['Ne'], NPH_EVENT)
    anch.append(dict(beta=b, Nq_248=float(f(248)), Nq_ratio_248=float(f(248)) / (11.2 * 248 ** 1.1),
                     E_comb_keV=E_nq, E_S1_keV=E_s1))
tabA = pd.DataFrame(anch); tabA.to_csv(os.path.join(OUT, 'table3b_anchored_beta.csv'), index=False)
print("\n=== Table 3b: slope beta anchored at 74.7 keV ===")
print(tabA.round(3).to_string(index=False))
# analytic elasticity: dlnE/dbeta = -ln(E/E0)/beta_eff(Nq)  (S1: beta_eff of Nph ~ 1.15)
dlnE_dbeta = -math.log(248 / E0) / 1.1
print(f"analytic dlnE/dbeta (anchored, comb) = {dlnE_dbeta:.3f}  -> delta beta = 0.05: {dlnE_dbeta*0.05*248:.1f} keV; "
      f"0.08: {dlnE_dbeta*0.08*248:.1f} keV")
# what delta-beta reproduces the paper's +-23 keV (9.3 %)?
dbeta_23 = 23 / 248 / abs(dlnE_dbeta)
# what delta-beta corresponds to the P043 10.4 % light-yield scale?
dbeta_10 = 0.104 / math.log(248 / E0)
print(f"delta beta for +-23 keV: {dbeta_23:.3f}; delta beta equivalent to a 10.4 % Nq scale at 248 keV: {dbeta_10:.3f}")

# 3c. Lindhard k +-20 %: unanchored (fixed W = 13.7) and anchored to Table S5 at 74.7 keV (W floats)
lind_rows = []
for fac in [0.8, 0.9, 1.0, 1.1, 1.2]:
    k = fac * K_LINDHARD
    f_un = lambda E, k=k: nq_lindhard(E, k, 13.7)
    Wanch = float(lindhard_L(E0, k)) * E0 * 1e3 / NQ_E0
    f_an = lambda E, k=k, W=Wanch: nq_lindhard(E, k, W)
    lind_rows.append(dict(k=k, k_over_k0=fac, E_comb_unanchored=solve_E(f_un, NQ_EVENT),
                          W_anchor_eV=Wanch, E_comb_anchored=solve_E(f_an, NQ_EVENT),
                          Nq248_anch_over_S5=float(f_an(248)) / (11.2 * 248 ** 1.1),
                          beta_eff_75_248=math.log(float(f_an(248)) / float(f_an(E0))) / math.log(248 / E0)))
tabK = pd.DataFrame(lind_rows); tabK.to_csv(os.path.join(OUT, 'table3c_lindhard_k.csv'), index=False)
print("\n=== Table 3c: Lindhard k +-20 % ===")
print(tabK.round(4).to_string(index=False))

# 3d. "kink" variants: power law beta below E0 = 1.1 (D-D), above E0 slope beta_hi
kink = []
for bhi in [1.0, 1.05, 1.1, 1.16, 1.2, 1.3]:
    f = lambda E, b=bhi: np.where(np.asarray(E, float) <= E0, 11.2 * np.asarray(E, float) ** 1.1,
                                  NQ_E0 * (np.asarray(E, float) / E0) ** b)
    kink.append(dict(beta_hi=bhi, E_comb=solve_E(f, NQ_EVENT), Nq248_over_S5=float(f(248)) / (11.2 * 248 ** 1.1)))
tabKk = pd.DataFrame(kink); tabKk.to_csv(os.path.join(OUT, 'table3d_kink.csv'), index=False)
print("\n=== Table 3d: kink at E0 (beta_hi above 74.7 keV) ===")
print(tabKk.round(3).to_string(index=False))

# 3e. summary spreads
E_models = tab3.set_index('model')['E_comb_keV']
plausible = ['NEST Table S5 (11.2 E^1.1)', 'LZ contour scale (11.32 E^1.112)',
             'Lindhard k=0.139 (Lenardo15 fit), W=13.7', 'Lindhard k=0.174 (LUX DD fit), W=13.7',
             'Lindhard k=0.166, W=13.7']
spread_all = (E_models.max() - E_models.min()) / 2
spread_pl = (E_models[plausible].max() - E_models[plausible].min()) / 2
sd_pl = E_models[plausible].std()
anchored_pm = (tabA.set_index('beta').loc[1.05, 'E_comb_keV'] - tabA.set_index('beta').loc[1.15, 'E_comb_keV']) / 2
print(f"\nhalf-range of E_comb over all models: {spread_all:.1f} keV; over plausible calibrated models: {spread_pl:.1f} keV (sd {sd_pl:.1f})")
print(f"half-range for beta 1.05-1.15 anchored at 74.7 keV: {anchored_pm:.1f} keV")
# combine with P043's gain budget (3.8 keV) and its 4 % yield scale (8.7 keV)
tot_with_slope = math.sqrt(3.8 ** 2 + anchored_pm ** 2)
tot_with_slope_and_form = math.sqrt(3.8 ** 2 + anchored_pm ** 2 + (248 * 0.06 / 1.1) ** 2)
print(f"gains (3.8) + slope (+-{anchored_pm:.1f}) = {tot_with_slope:.1f} keV; + form (6 % Nq) = {tot_with_slope_and_form:.1f} keV; paper 23 keV")

# ---------------------------------------------------------------------------------------------
# 4. Partition: Qy, Ly, Thomas-Imel box model, band slope, degeneracy
# ---------------------------------------------------------------------------------------------
TI_LZ = nest_nr(100.0)['TI']            # 0.04076 at 96.5 V/cm
NEXNI_NEST = 1.399                      # P024: nestpy N_ex/N_i at 248 keV for NR (from GetQuanta means)
NEXNI_MEAS = 1.0                        # recalled: measured NR N_ex/N_i ~ 1 (likely)

def ti_box_Ne(E, nexni=NEXNI_NEST, sigma=TI_LZ, nq_fun=lambda E: 11.2 * np.asarray(E, float) ** 1.1):
    """Thomas-Imel box model: Ne/Ni = ln(1+xi)/xi, xi = sigma Ni/4  =>  Ne = (4/sigma) ln(1 + sigma Ni/4)."""
    Ni = nq_fun(E) / (1 + nexni)
    xi = sigma * Ni / 4
    return (4 / sigma) * np.log1p(xi)

part_rows = []
for E in [20, 50, 74.7, 100, 150, 200, 248, 300, 330]:
    lzb = nest_nr(E); lzn = nest_nr(E, use_break=False); dft = nest_nr(E, lz.NEST_NR_DEFAULT)
    ti14 = float(ti_box_Ne(E, NEXNI_NEST)); ti10 = float(ti_box_Ne(E, NEXNI_MEAS))
    part_rows.append(dict(E_keV=E, p_break=float(lzb['p']), Ne_break=float(lzb['Ne']), Ne_nobreak=float(lzn['Ne']),
                          Ne_default=float(dft['Ne']), Ne_TI_nexni1p4=ti14, Ne_TI_nexni1p0=ti10,
                          Nph_break=float(lzb['Nph']), Nph_nobreak=float(lzn['Nph']),
                          fe_break=float(lzb['Ne'] / lzb['Nq']), fe_nobreak=float(lzn['Ne'] / lzn['Nq']),
                          Qy_break=float(lzb['Qy']), Qy_nobreak=float(lzn['Qy']), Ly_break=float(lzb['Ly']),
                          dlog10Ne_break_vs_nobreak=math.log10(float(lzb['Ne'] / lzn['Ne']))))
tab4 = pd.DataFrame(part_rows); tab4.to_csv(os.path.join(OUT, 'table4_partition.csv'), index=False)
print("\n=== Table 4: partition ===")
print(tab4.round(3).to_string(index=False))

# band slope d log10 Ne / d log10 Nph at 248 keV
def band_slope(ne_fun, nph_fun, E=248.0, h=1e-3):
    return (math.log10(ne_fun(E * (1 + h))) - math.log10(ne_fun(E * (1 - h)))) / \
           (math.log10(nph_fun(E * (1 + h))) - math.log10(nph_fun(E * (1 - h))))
nph_lz = lambda E: float(nest_nr(E)['Nph'])
slopes = dict(
    LZ_break=band_slope(lambda E: float(nest_nr(E)['Ne']), nph_lz),
    LZ_nobreak=band_slope(lambda E: float(nest_nr(E, use_break=False)['Ne']), lambda E: float(nest_nr(E, use_break=False)['Nph'])),
    TI_box_nexni1p4=band_slope(lambda E: float(ti_box_Ne(E, NEXNI_NEST)), nph_lz),
    TI_box_nexni1p0=band_slope(lambda E: float(ti_box_Ne(E, NEXNI_MEAS)), nph_lz),
)
# convert to d log10 S2c / d S1c at S1c = 540 (P043 quotes 0.00058 dex/keV)
dlogNe_dE = (math.log10(float(nest_nr(250)['Ne'])) - math.log10(float(nest_nr(246)['Ne']))) / 4
print("\nband slopes d log10 Ne / d log10 Nph at 248 keV:", {k: round(v, 3) for k, v in slopes.items()},
      f"; d log10 Ne/dE = {dlogNe_dE:.5f} dex/keV")
# no-break offset in sigma units (P024 band sigma 0.0313, event 1.54 sigma with break)
d_nb = math.log10(float(nest_nr(248, use_break=False)['Ne']) / float(nest_nr(248)['Ne']))
print(f"no-break shift at 248 keV: {d_nb:.4f} dex = {d_nb/0.0313:.2f} sigma -> event at {1.54 + d_nb/0.0313:.2f} sigma (P024 MC: 5.02)")

# Fit the p(E) break form to the TI-box Ne(E) (does the log form emerge from TI saturation?)
E_fit = np.linspace(E0 + 0.5, 330, 200)
def ne_from_p(params, E, nexni_unused=None):
    a, b = params
    P = dict(lz.NEST_NR_LZ); P.update(a=a, b=b)
    return nest_nr(E, P)['Ne']
fits = {}
for label, nexni in [('nexni1p4', NEXNI_NEST), ('nexni1p0', NEXNI_MEAS)]:
    target = ti_box_Ne(E_fit, nexni)
    res = least_squares(lambda q: np.log(ne_from_p(q, E_fit)) - np.log(target), x0=[0.02, 0.03],
                        bounds=([0, 1e-4], [0.3, 1.0]))
    fits[label] = dict(a=float(res.x[0]), b=float(res.x[1]), rms_log=float(np.sqrt(np.mean(res.fun ** 2))),
                       Ne_248_fit=float(ne_from_p(res.x, 248.0)), Ne_248_TI=float(ti_box_Ne(248.0, nexni)))
print("\np(E)-break fits to the TI box model over 75-330 keV:", json.dumps(fits, indent=1))
print(f"LZ: a = 0.0230, b = 0.0289; Ne(248) = {float(nest_nr(248)['Ne']):.1f}")

# Degeneracy: scale Nq by s (light-yield scale), refit (a, b) so that Ne(Nph) is unchanged in the AmBe range
def ne_of_nph(P, nph_grid):
    out = []
    for nph in nph_grid:
        E = brentq(lambda E: float(nest_nr(E, P)['Nph']) - nph, 20, 2000)
        out.append(float(nest_nr(E, P)['Ne']))
    return np.array(out)
nph_grid = np.linspace(1500, 6300, 25)   # S1c ~ 165-690 phd
ne_base = ne_of_nph(lz.NEST_NR_LZ, nph_grid)
degen = []
for s in [0.90, 1.077, 1.10]:
    Ps = dict(lz.NEST_NR_LZ); Ps['alpha'] = 11.2 * s
    # before refit: band shift at fixed Nph
    shift0 = np.log10(ne_of_nph(Ps, nph_grid) / ne_base)
    def resid(q):
        Pq = dict(Ps); Pq.update(a=q[0], b=q[1])
        return np.log10(ne_of_nph(Pq, nph_grid) / ne_base)
    r = least_squares(resid, x0=[0.023, 0.0289], bounds=([0, 1e-4], [0.3, 1.0]))
    Pq = dict(Ps); Pq.update(a=r.x[0], b=r.x[1])
    degen.append(dict(s=s, shift_before_mean_dex=float(shift0.mean()), shift_before_at540=float(np.interp(4910, nph_grid, shift0)),
                      a_refit=float(r.x[0]), b_refit=float(r.x[1]), rms_after_dex=float(np.sqrt(np.mean(r.fun ** 2))),
                      max_after_dex=float(np.abs(r.fun).max()), E_comb_event=solve_E(lambda E: Pq['alpha'] * E ** 1.1, NQ_EVENT),
                      Ne_248_refit=float(nest_nr(248, Pq)['Ne'])))
tabD = pd.DataFrame(degen); tabD.to_csv(os.path.join(OUT, 'table4b_degeneracy.csv'), index=False)
print("\n=== Table 4b: light-yield scale s vs (a, b) refit -- band position at fixed S1c ===")
print(tabD.round(4).to_string(index=False))

# AmBe endpoint in S1c under each energy scale (E_R,max = 0.0301 * 11 MeV ~ 331 keV)
E_ambe_max = 4 * 1.008665 * A_XE / (1.008665 + A_XE) ** 2 * 11.0e3
s1_end = {name: float(f(E_ambe_max) - nest_nr(E_ambe_max)['Ne']) * lz.LZ['g1'] for name, f in models_nq.items()
          if name in ('NEST Table S5 (11.2 E^1.1)', 'LZ contour scale (11.32 E^1.112)', 'Lindhard k=0.166, W=13.7')}
print(f"\nAmBe kinematic endpoint {E_ambe_max:.0f} keV -> S1c: ", {k: round(v) for k, v in s1_end.items()},
      "; visible AmBe points in Fig. 2 end at ~600 phd (P024 digitisation max 599.6)")

# ---------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------
# Fig 1: L(E)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for k, lab, ls in [(K_LINDHARD, 'Lindhard k = 0.166', '-'), (0.8 * K_LINDHARD, 'k = 0.133 (-20 %)', ':'),
                   (1.2 * K_LINDHARD, 'k = 0.199 (+20 %)', ':'), (K_LENARDO, 'k = 0.139 (global-fit value, recalled)', '--'),
                   (K_LUXDD, 'k = 0.174 (LUX D-D fit, recalled)', '-.')]:
    ax[0].plot(E_grid, lindhard_L(E_grid, k), ls, label=lab)
for name, c in [('NEST Table S5 (11.2 E^1.1)', 'C3'), ('LZ contour scale (11.32 E^1.112)', 'C2'), ('NEST default (11.0 E^1.1)', 'C7')]:
    ax[0].plot(E_grid, models_nq[name](E_grid) * 13.7 / (E_grid * 1e3), color=c, lw=2, label=f'{name}, W=13.7')
ax[0].axvline(248, color='k', lw=0.6, alpha=0.5); ax[0].axvspan(0, 74, color='orange', alpha=0.12)
ax[0].axvspan(74, 330, color='purple', alpha=0.06)
ax[0].set_xscale('log'); ax[0].set_xlim(1, 400); ax[0].set_ylim(0.05, 0.45)
ax[0].set_xlabel('recoil energy [keV]'); ax[0].set_ylabel('L(E) = N_q W / E  (W = 13.7 eV)')
ax[0].legend(fontsize=7); ax[0].set_title('Quenching factor: Lindhard vs NEST power laws')
for name, c in [('NEST Table S5 (11.2 E^1.1)', 'C3'), ('LZ contour scale (11.32 E^1.112)', 'C2'), ('NEST default (11.0 E^1.1)', 'C7')]:
    ax[1].plot(E_grid, beta_eff(models_nq[name], E_grid), color=c, lw=2, label=name)
for k, lab, ls in [(K_LINDHARD, 'Lindhard k = 0.166', '-'), (K_LENARDO, 'k = 0.139', '--'), (0.8 * K_LINDHARD, 'k = 0.133', ':'), (1.2 * K_LINDHARD, 'k = 0.199', ':')]:
    ax[1].plot(E_grid, beta_eff(lambda E, k=k: nq_lindhard(E, k), E_grid), ls, color='C0', label=lab)
ax[1].axvline(248, color='k', lw=0.6, alpha=0.5); ax[1].axvline(E0, color='k', lw=0.6, alpha=0.5)
ax[1].set_xscale('log'); ax[1].set_xlim(1, 400); ax[1].set_ylim(1.0, 1.3)
ax[1].set_xlabel('recoil energy [keV]'); ax[1].set_ylabel(r'$\beta_{\rm eff} = d\ln N_q/d\ln E$'); ax[1].legend(fontsize=7)
ax[1].set_title('Local exponent')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P064_fig1_L_beta.png'), dpi=150); plt.close(fig)

# Fig 2: Nq ratio to Table S5, and event energy by model
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for name, f in models_nq.items():
    if 'W=13.5' in name: continue
    ax[0].plot(E_grid, f(E_grid) / (11.2 * E_grid ** 1.1), label=name, lw=1.5 if 'NEST' in name or 'contour' in name else 1)
ax[0].axhline(1, color='k', lw=0.6); ax[0].axvline(248, color='k', lw=0.6, alpha=0.5); ax[0].axvspan(0, 74, color='orange', alpha=0.12)
ax[0].set_xscale('log'); ax[0].set_xlim(5, 400); ax[0].set_ylim(0.7, 1.5)
ax[0].set_xlabel('recoil energy [keV]'); ax[0].set_ylabel('N_q(model) / N_q(Table S5)'); ax[0].legend(fontsize=6.5)
ax[0].set_title('Total quanta relative to Table S5 (D-D range shaded)')
y = np.arange(len(tab3))
ax[1].barh(y, tab3['E_comb_keV'], color='C0', alpha=0.7, label='combined quanta (N_q = 5179)')
ax[1].scatter(tab3['E_S1_keV'], y, color='C3', zorder=3, label='S1 only (N_ph = 4910, LZ N_e)')
ax[1].axvline(248, color='k'); ax[1].axvspan(248 - 23, 248 + 23, color='k', alpha=0.1, label='LZ 248 +- 23 (sys)')
ax[1].set_yticks(y); ax[1].set_yticklabels(tab3['model'], fontsize=7); ax[1].set_xlim(180, 300)
ax[1].set_xlabel('reconstructed recoil energy of the event [keV]'); ax[1].legend(fontsize=7, loc='lower right')
ax[1].set_title('Event energy under each quenching model')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P064_fig2_Nq_event_energy.png'), dpi=150); plt.close(fig)

# Fig 3: anchored slope systematics
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(tabA['beta'], tabA['E_comb_keV'], 'o-', label='combined quanta')
ax[0].plot(tabA['beta'], tabA['E_S1_keV'], 's--', label='S1 only')
ax[0].axhspan(248 - 23, 248 + 23, color='k', alpha=0.1, label='248 +- 23 keV (LZ sys)')
ax[0].axvline(1.1, color='C3', lw=0.8); ax[0].axvline(BETA_P, color='C2', lw=0.8)
ax[0].set_xlabel(r'slope $\beta$ above the D-D anchor (N_q fixed at 74.7 keV)'); ax[0].set_ylabel('event energy [keV]')
ax[0].legend(fontsize=8); ax[0].set_title(r'Energy vs slope: $d\ln E/d\beta = -\ln(E/E_0)/\beta$')
ax[1].plot(tabK['k'], tabK['E_comb_unanchored'], 'o-', label='Lindhard, W = 13.7 eV fixed')
ax[1].plot(tabK['k'], tabK['E_comb_anchored'], 's--', label='Lindhard anchored to Table S5 at 74.7 keV')
ax[1].axhspan(248 - 23, 248 + 23, color='k', alpha=0.1)
ax[1].set_xlabel('Lindhard k'); ax[1].set_ylabel('event energy [keV]'); ax[1].legend(fontsize=8)
ax[1].set_title('Lindhard k +- 20 %: unanchored vs D-D-anchored')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P064_fig3_slope_systematics.png'), dpi=150); plt.close(fig)

# Fig 4: partition
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
Eg = np.linspace(5, 400, 400)
ax[0].plot(Eg, nest_nr(Eg)['Ne'], 'C3', lw=2, label='LZ Table S5 with p(E) break')
ax[0].plot(Eg, nest_nr(Eg, use_break=False)['Ne'], 'C3--', label='LZ Table S5, no break (p = 0.5)')
ax[0].plot(Eg, nest_nr(Eg, lz.NEST_NR_DEFAULT)['Ne'], 'C7', label='NEST default')
ax[0].plot(Eg, ti_box_Ne(Eg, NEXNI_NEST), 'C0', label=f'Thomas-Imel box, N_ex/N_i = {NEXNI_NEST}')
ax[0].plot(Eg, ti_box_Ne(Eg, NEXNI_MEAS), 'C0:', label='Thomas-Imel box, N_ex/N_i = 1.0')
ax[0].axhline(NE_EVENT, color='k', lw=0.8); ax[0].axvline(248, color='k', lw=0.6, alpha=0.5); ax[0].axvline(E0, color='k', lw=0.6, alpha=0.5)
ax[0].set_xlabel('recoil energy [keV]'); ax[0].set_ylabel('N_e'); ax[0].legend(fontsize=7); ax[0].set_title('Electron yield: break vs Thomas-Imel saturation')
ax[1].plot(Eg, nest_nr(Eg)['Ne'] / nest_nr(Eg)['Nq'], 'C3', lw=2, label='with break')
ax[1].plot(Eg, nest_nr(Eg, use_break=False)['Ne'] / nest_nr(Eg, use_break=False)['Nq'], 'C3--', label='no break')
ax[1].plot(Eg, ti_box_Ne(Eg, NEXNI_NEST) / (11.2 * Eg ** 1.1), 'C0', label='TI box (1.4)')
ax[1].scatter([248], [NE_EVENT / NQ_EVENT], color='k', zorder=3, label='event: 269/5179 = 5.2 %')
ax[1].set_xlabel('recoil energy [keV]'); ax[1].set_ylabel('electron fraction N_e/N_q'); ax[1].legend(fontsize=7); ax[1].set_title('Electron fraction')
ax[2].plot(Eg, nest_nr(Eg)['p'], 'C3', lw=2, label='LZ p(E) = 0.5 + 0.023 ln[1 + 0.0289 (E - 74.7)]')
for lab, c in [('nexni1p4', 'C0'), ('nexni1p0', 'C9')]:
    P = dict(lz.NEST_NR_LZ); P.update(a=fits[lab]['a'], b=fits[lab]['b'])
    ax[2].plot(Eg, nest_nr(Eg, P)['p'], color=c, ls='--', label=f"fit to TI box ({lab}): a = {fits[lab]['a']:.4f}, b = {fits[lab]['b']:.4f}")
ax[2].set_xlabel('recoil energy [keV]'); ax[2].set_ylabel('charge-yield exponent p(E)'); ax[2].legend(fontsize=7); ax[2].set_title('Is the break Thomas-Imel physics?')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P064_fig4_partition.png'), dpi=150); plt.close(fig)

# ---------------------------------------------------------------------------------------------
# Results JSON
# ---------------------------------------------------------------------------------------------
res = dict(
    inputs=dict(Nq_event=NQ_EVENT, Nph_event=NPH_EVENT, Ne_event=NE_EVENT, field_Vcm=FIELD, W_list=W_LIST,
                k_lindhard=K_LINDHARD, k_lenardo_recalled=K_LENARDO, k_luxdd_recalled=K_LUXDD,
                contour_scale=dict(alpha=ALPHA_P, beta=BETA_P, W=W_P), E0=E0, TI_LZ=TI_LZ, nexni_nest=NEXNI_NEST),
    validation=dict(max_rel_diff_analytic_vs_nestpy=val_max),
    lindhard=dict(L_50=L_50, L_74p7=float(lindhard_L(74.7)), L_248=float(lindhard_L(248)), L_300=L_300,
                  mean_dlnL_dlnE_50_300=slope_L, W_match_S5_248=W_match_S5, W_match_contour_248=W_match_P,
                  k_S5_75=k_S5_75, k_S5_248=k_S5_248, k_contour_75=k_P_75, k_contour_248=k_P_248),
    table1=tab1.to_dict(orient='records'), table3=tab3.to_dict(orient='records'),
    table3b_anchored=tabA.to_dict(orient='records'), table3c_lindhard_k=tabK.to_dict(orient='records'),
    table3d_kink=tabKk.to_dict(orient='records'),
    systematics=dict(dlnE_dbeta=dlnE_dbeta, dE_dbeta0p05_keV=dlnE_dbeta * 0.05 * 248, dbeta_for_23keV=dbeta_23,
                     dbeta_equiv_10p4pct=dbeta_10, halfrange_all=spread_all, halfrange_plausible=spread_pl, sd_plausible=sd_pl,
                     halfrange_beta_1p05_1p15=anchored_pm, total_gains_slope=tot_with_slope, total_gains_slope_form=tot_with_slope_and_form),
    table4=tab4.to_dict(orient='records'), band_slopes=slopes, dlog10Ne_dE=dlogNe_dE, nobreak_shift_dex=d_nb,
    nobreak_sigma=1.54 + d_nb / 0.0313, ti_fits=fits, degeneracy=tabD.to_dict(orient='records'),
    ambe_endpoint=dict(E_max_keV=E_ambe_max, S1c_end=s1_end),
)
with open(os.path.join(OUT, 'results.json'), 'w') as fh:
    json.dump(res, fh, indent=1, default=float)
print("\nwritten:", OUT)
