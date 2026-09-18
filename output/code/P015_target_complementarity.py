"""
P015 -- Fluorine, argon, germanium, iodine and tungsten: which targets can confirm an
inelastic 300-390 keV splitting?

Run from the simulation root:   .venv/bin/python output/code/P015_target_complementarity.py

Parts
  1. Kinematic ceiling delta_max(A, m_chi) = mu v_max^2 / 2 and the accessible recoil window
     [E-, E+] at v_max for ten nuclei; "blind" table for delta = 250/300/350/366/380 keV.
  2. WimPyDD O1-isoscalar rates (LZ/Anand unit coupling c^s_1 = 1/m_v^2 <=> WimPyDD c^0 = 2/m_v^2,
     P003/P007 convention) per tonne-year of target element, integrated over the accessible
     window, annual-mean and 16-June halos; ratio to xenon.
  3. Expected counts in recalled exposures (PICO-60 C3F8 and CF3I, DEAP-3600, CRESST-II/III,
     DAMA/LIBRA, COSINE-100, ANAIS-112) when each spectrum is normalised to LZ's 1.0 event.
  4. Exposure of NaI and CaWO4 needed for 3 events at the LZ best fit; E+ for I and W.
  5. Digitisation of the comparison curves in LZ Fig. S7 (PDF vector paths) and their ratio to
     LZ's two-sided interval.

Outputs: output/work/P015/*.csv|json and output/work/P015/figures/*.png
"""
import sys, os, math, json, time, importlib
import numpy as np
from scipy import special
sys.path.insert(0, 'output/code')
from common import lzcommon as lz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'output/work/P015'; FIG = OUT + '/figures'
os.makedirs(FIG, exist_ok=True)
T0 = time.time()
WD = lz.wd()

# --- runtime fix (no file edited): WimPyDD's 18xW_func_w.py files do `from numpy import *` but
# call `np.`; WimPyDD loads them with __import__ (cached in sys.modules), so injecting `np` once is enough.
for a in (180, 182, 183, 184, 186):
    importlib.import_module(f'WimPyDD.Targets.Nuclear_response_functions.{a}W_func_w').np = np

MV = lz.M_V_GEV                      # 246.2 GeV (LZ/Anand unit coupling)
C = lz.C_KMS
VESC = lz.VESC_KMS                   # 544 km/s (Baxter 2021, via lzcommon)
V_JUNE = lz.v_earth_kms(167) + VESC  # 16 June 2023 (day 167)
V_MEAN = lz.v_earth_kms() + VESC     # annual mean |v_sun| + v_esc
V_DEC = lz.v_earth_kms(350) + VESC
MASSES = [400.0, 1000.0, 4000.0]
DELTAS = [250.0, 300.0, 350.0, 366.0, 380.0]   # 366 keV = P007 Higgsino delta(N=1) at 1 TeV
TABLE_DELTAS = [250.0, 300.0, 350.0, 380.0]

# targets: integer A as in the assignment; Z; WimPyDD element; nominal analysis threshold (keV_nr, recalled/assumed)
TARGETS = {
    'C':  dict(A=12,  Z=6,  wd=WD.C,  thr=3.3,  thr_note='PICO-60 C3F8 bubble-nucleation threshold 2.45-3.3 keV (recalled, likely)'),
    'O':  dict(A=16,  Z=8,  wd=WD.O,  thr=1.0,  thr_note='CRESST CaWO4, nominal 1 keV (assignment; real thresholds are lower)'),
    'F':  dict(A=19,  Z=9,  wd=WD.F,  thr=3.3,  thr_note='PICO-60 C3F8 (recalled, likely)'),
    'Na': dict(A=23,  Z=11, wd=WD.Na, thr=4.0,  thr_note='NaI ~1 keVee at Q_Na~0.25 (recalled, likely)'),
    'Ar': dict(A=40,  Z=18, wd=WD.Ar, thr=10.0, thr_note='nominal 10 keV (assignment); DEAP-3600 2019 ROI actually starts near 50 keV_nr (recalled, uncertain)'),
    'Ca': dict(A=40,  Z=20, wd=WD.Ca, thr=1.0,  thr_note='CRESST CaWO4, nominal'),
    'Ge': dict(A=73,  Z=32, wd=WD.Ge, thr=1.0,  thr_note='Ge bolometer, nominal'),
    'I':  dict(A=127, Z=53, wd=WD.I,  thr=11.0, thr_note='NaI ~1 keVee at Q_I~0.09 (recalled, likely)'),
    'Xe': dict(A=131, Z=54, wd=WD.Xe, thr=5.4,  thr_note='LZ 50% efficiency point (paper)'),
    'W':  dict(A=184, Z=74, wd=WD.W,  thr=1.0,  thr_note='CRESST CaWO4, nominal 1 keV'),
}
RATE_TARGETS = ['Xe', 'I', 'W', 'Ge', 'Ar', 'F']   # WimPyDD rates computed for these (others are kinematically blind)

# ==============================================================================================
# Part 1: kinematics
# ==============================================================================================
def mu_gev(m_chi, A):
    return lz.mu_red(m_chi, lz.m_nucleus_gev(A))

def delta_ceiling_kev(m_chi, A, v_kms):
    return mu_gev(m_chi, A) * (v_kms / C) ** 2 / 2.0 * 1e6

rows = []
for sym, t in TARGETS.items():
    for m in MASSES:
        for lab, v in [('june16', V_JUNE), ('annual_mean', V_MEAN), ('dec16', V_DEC)]:
            rows.append(dict(target=sym, A=t['A'], m_chi_GeV=m, halo=lab, v_max_kms=round(v, 1),
                             mu_GeV=round(mu_gev(m, t['A']), 3), delta_max_keV=round(delta_ceiling_kev(m, t['A'], v), 1)))
with open(f'{OUT}/delta_max_table.csv', 'w') as f:
    f.write(','.join(rows[0].keys()) + '\n')
    for r in rows:
        f.write(','.join(str(v) for v in r.values()) + '\n')

blind = []
for sym, t in TARGETS.items():
    for m in MASSES:
        for d in TABLE_DELTAS + [366.0]:
            dmax_j = delta_ceiling_kev(m, t['A'], V_JUNE); dmax_m = delta_ceiling_kev(m, t['A'], V_MEAN)
            Em_j, Ep_j = lz.E_R_range_keV(m, V_JUNE, A=t['A'], delta_kev=d)
            Em_m, Ep_m = lz.E_R_range_keV(m, V_MEAN, A=t['A'], delta_kev=d)
            blind.append(dict(target=sym, A=t['A'], m_chi_GeV=m, delta_keV=d,
                              blind_june=int(d > dmax_j), blind_mean=int(d > dmax_m),
                              E_minus_june_keV=None if math.isnan(Em_j) else round(Em_j, 1),
                              E_plus_june_keV=None if math.isnan(Ep_j) else round(Ep_j, 1),
                              E_minus_mean_keV=None if math.isnan(Em_m) else round(Em_m, 1),
                              E_plus_mean_keV=None if math.isnan(Ep_m) else round(Ep_m, 1),
                              threshold_keV=t['thr'],
                              accessible_above_threshold_june=int((not math.isnan(Ep_j)) and Ep_j > t['thr'])))
with open(f'{OUT}/blind_table.csv', 'w') as f:
    f.write(','.join(blind[0].keys()) + '\n')
    for r in blind:
        f.write(','.join('' if v is None else str(v) for v in r.values()) + '\n')

# A at which the ceiling equals delta (for the text): solve mu(A) v^2/2 = delta
def A_min_for_delta(m_chi, d_kev, v_kms):
    lo, hi = 1.0, 400.0
    if delta_ceiling_kev(m_chi, hi, v_kms) < d_kev:
        return float('inf')
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if delta_ceiling_kev(m_chi, mid, v_kms) < d_kev:
            lo = mid
        else:
            hi = mid
    return hi
amin = {f'{int(m)}GeV_{int(d)}keV': round(A_min_for_delta(m, d, V_JUNE), 1) for m in MASSES for d in TABLE_DELTAS}
# infinite-mass limit: mu -> m_N, so delta_max -> m_N v^2/2 ; A needed
amin_inf = {f'{int(d)}keV': round(2 * d * 1e-6 / (lz.AMU_GEV * (V_JUNE / C) ** 2), 1) for d in TABLE_DELTAS}
print('Part 1 done: v_max June/mean/Dec = %.1f / %.1f / %.1f km/s' % (V_JUNE, V_MEAN, V_DEC))
print('  delta_max (1000 GeV, June):', {s: round(delta_ceiling_kev(1000, t['A'], V_JUNE), 1) for s, t in TARGETS.items()})
print('  A_min(June):', amin, ' A_min(m->inf):', amin_inf)

# ==============================================================================================
# Part 2: WimPyDD rates
# ==============================================================================================
HAM = lz.wd_hamiltonian('O1_iso_unit', {1: (2.0 / MV ** 2, 0.0)})   # c_p = c_n = 1/m_v^2 (LZ unit coupling)
VGRID = np.linspace(0.0, VESC + 300.0, 1200)
days12 = 15.0 + 365.25 / 12 * np.arange(12)
halo_annual = (VGRID, np.mean([lz.wd_halo(day_of_year=d, vmin=VGRID)[1] for d in days12], axis=0))
halo_june = lz.wd_halo(day_of_year=167, vmin=VGRID)
HALOS = {'annual': halo_annual, 'june16': halo_june}
M_REF = 1000.0
NE = 60

def eff_LZ(E, sig_hi=11.5, sig_lo=2.5):
    """LZ NR efficiency model used in P005/P007: 0.96 plateau, 50% at 5.4 and 269.9 keV (paper), erf edges."""
    E = np.asarray(E, float)
    return lz.LZ['eff_plateau'] * 0.5 * (1 + special.erf((E - lz.LZ['E_50pct_low_keV']) / (math.sqrt(2) * sig_lo))) \
        * 0.5 * special.erfc((E - lz.LZ['E_50pct_high_keV']) / (math.sqrt(2) * sig_hi))

def window_rate(sym, d, halo_key, m_chi=M_REF, return_spectrum=False):
    """Integrated O1 rate [events / (t yr of element)] over the kinematic window at v_max(16 June)."""
    t = TARGETS[sym]
    Em, Ep = lz.E_R_range_keV(m_chi, V_JUNE, A=t['A'], delta_kev=d)
    if math.isnan(Em):
        return (0.0, None, None) if return_spectrum else 0.0
    E = np.linspace(max(0.5, 0.97 * Em), 1.03 * Ep, NE)
    r = lz.wd_rate(HAM, m_chi, E, halo=HALOS[halo_key], delta_kev=d, target=t['wd'])
    r = np.clip(r, 0.0, None)
    R = float(np.trapezoid(r, E))
    return (R, E, r) if return_spectrum else R

rates = {}
spectra = {}
t1 = time.time()
for sym in RATE_TARGETS:
    for d in DELTAS:
        for hk in HALOS:
            R, E, r = window_rate(sym, d, hk, return_spectrum=True)
            rates[(sym, d, hk)] = R
            if E is not None and d in (300.0, 350.0):
                spectra[(sym, d, hk)] = (E, r)
    print('  rates done for %s (%.0f s)' % (sym, time.time() - t1))

# LZ normalisation (xenon, annual-mean halo, LZ efficiency, 2.84 t yr)
EXPO_LZ = lz.LZ['exposure_tyr']
E_LZ = np.linspace(0.5, 330.0, 120)
norm = {}
for d in DELTAS:
    rXe = np.clip(lz.wd_rate(HAM, M_REF, E_LZ, halo=halo_annual, delta_kev=d), 0, None)
    n_unit = EXPO_LZ * float(np.trapezoid(rXe * eff_LZ(E_LZ), E_LZ))
    norm[d] = dict(N_LZ_unit_coupling=n_unit, scale_to_1_event=1.0 / n_unit,
                   c1s_mv2_sq_bestfit=1.0 / n_unit,   # rate is linear in (c m_v^2)^2; unit coupling has (c m_v^2)^2 = 1
                   sigma_SI_bestfit_cm2=(1.0 / n_unit) * lz.mu_red(M_REF, lz.M_NUCLEON_GEV) ** 2 / (math.pi * MV ** 4) * lz.GEV_TO_CM2,
                   Xe_window_rate_unit_annual=rates[('Xe', d, 'annual')],
                   in_LZ_eff_fraction=n_unit / (EXPO_LZ * rates[('Xe', d, 'annual')]) if rates[('Xe', d, 'annual')] > 0 else None)
print('Part 2 done (%.0f s). LZ unit-coupling events:' % (time.time() - T0), {d: round(v['N_LZ_unit_coupling'], 4) for d, v in norm.items()})

# isovector (Higgsino-like) target weighting relative to isoscalar: ((N - (1-4 s_W^2) Z)/A)^2 ; s_W^2 = 0.231 (recalled, certain)
SW2 = 0.231
isov = {s: ((t['A'] - t['Z']) - (1 - 4 * SW2) * t['Z']) ** 2 / t['A'] ** 2 for s, t in TARGETS.items()}

with open(f'{OUT}/rates_table.csv', 'w') as f:
    f.write('target,A,delta_keV,halo,E_minus_june_keV,E_plus_june_keV,rate_unit_coupling_per_tyr,ratio_to_Xe,'
            'events_per_tyr_element_at_LZ_bestfit,june_over_annual,isovector_weight_rel_Xe\n')
    for sym in RATE_TARGETS:
        for d in DELTAS:
            Em, Ep = lz.E_R_range_keV(M_REF, V_JUNE, A=TARGETS[sym]['A'], delta_kev=d)
            for hk in HALOS:
                R = rates[(sym, d, hk)]; RXe = rates[('Xe', d, hk)]
                f.write('%s,%d,%.0f,%s,%s,%s,%.6g,%.4g,%.4g,%.3f,%.3f\n' % (
                    sym, TARGETS[sym]['A'], d, hk,
                    '' if math.isnan(Em) else '%.1f' % Em, '' if math.isnan(Ep) else '%.1f' % Ep,
                    R, R / RXe if RXe > 0 else float('nan'), R * norm[d]['scale_to_1_event'],
                    rates[(sym, d, 'june16')] / rates[(sym, d, 'annual')] if rates[(sym, d, 'annual')] > 0 else float('nan'),
                    isov[sym] / isov['Xe']))

# spectra for figure/CSV
with open(f'{OUT}/spectra_1000GeV.csv', 'w') as f:
    f.write('target,delta_keV,halo,E_keV,dRdE_unit_coupling_per_tyr_keV\n')
    for (sym, d, hk), (E, r) in spectra.items():
        for e, v in zip(E, r):
            f.write('%s,%.0f,%s,%.2f,%.6g\n' % (sym, d, hk, e, v))

# ==============================================================================================
# Part 3: existing exposures (all recalled; reliability flagged)
# ==============================================================================================
M_F, M_C, M_I, M_Na, M_W, M_Ca, M_O = 18.998, 12.011, 126.904, 22.990, 183.84, 40.078, 15.999   # u (recalled, certain)
FRAC = {'C3F8_F': 8 * M_F / (3 * M_C + 8 * M_F), 'CF3I_I': M_I / (M_C + 3 * M_F + M_I),
        'NaI_I': M_I / (M_Na + M_I), 'CaWO4_W': M_W / (M_Ca + M_W + 4 * M_O)}
KGD_TO_TYR = 1.0 / (1000.0 * 365.25)
EXPTS = [
    dict(name='PICO-60 C3F8 (2019)', element='F', exposure_kgd=1404.0, mass_fraction=FRAC['C3F8_F'], reliability='likely',
         note='complete C3F8 exposure 1404 kg d, thresholds 2.45 and 3.3 keV; bubble chamber: no upper energy cut'),
    dict(name='PICO-60 CF3I (2016; recast in PICO 2023 inelastic paper)', element='I', exposure_kgd=1335.0, mass_fraction=FRAC['CF3I_I'], reliability='uncertain',
         note='CF3I run ~1335 kg d at 13.6 keV threshold; iodine recoils nucleate bubbles; no upper energy cut'),
    dict(name='DEAP-3600 (2019)', element='Ar', exposure_kgd=758e3, mass_fraction=1.0, reliability='likely',
         note='758 t d; ROI roughly 50-100 keV_nr (uncertain); irrelevant because argon is kinematically blind'),
    dict(name='DEAP-3600 PLR (2026)', element='Ar', exposure_kgd=1.0e6, mass_fraction=1.0, reliability='uncertain (placeholder ~1000 t d)',
         note='exposure not recalled; any value gives zero because argon is blind for delta >= 250 keV'),
    dict(name='CRESST-II Lise (2016)', element='W', exposure_kgd=52.0, mass_fraction=FRAC['CaWO4_W'], reliability='uncertain',
         note='~52 kg d CaWO4, ROI 0.3-40 keV; the Fig. S7 CRESST-II curve is the Bramante et al. (2016) recast'),
    dict(name='CRESST-III (2019)', element='W', exposure_kgd=3.64, mass_fraction=FRAC['CaWO4_W'], reliability='uncertain',
         note='~3.6 kg d, ROI to 16 keV'),
    dict(name='DAMA/LIBRA phase1+phase2', element='I', exposure_kgd=2.46 / KGD_TO_TYR, mass_fraction=FRAC['NaI_I'], reliability='likely',
         note='2.46 t yr NaI(Tl) total (1.33 + 1.13); no NR/ER discrimination; single-hit spectra published mainly <= 20 keVee'),
    dict(name='COSINE-100', element='I', exposure_kgd=0.2 / KGD_TO_TYR, mass_fraction=FRAC['NaI_I'], reliability='uncertain',
         note='~0.2 t yr NaI(Tl) (assignment); no NR/ER discrimination'),
    dict(name='ANAIS-112 (3 yr)', element='I', exposure_kgd=112.5 * 3 * 365.25, mass_fraction=FRAC['NaI_I'], reliability='uncertain',
         note='112.5 kg x 3 yr ~ 0.34 t yr NaI(Tl)'),
    dict(name='LZ (this paper)', element='Xe', exposure_kgd=EXPO_LZ / KGD_TO_TYR, mass_fraction=1.0, reliability='certain (paper)',
         note='2.84 t yr; counts here are the full-window (efficiency-free) counts, larger than the fitted 1.0 by 1/eff-fraction'),
]
QI = 0.09  # iodine quenching factor in NaI(Tl) (recalled, likely; DAMA uses 0.09)
exp_rows = []
for ex in EXPTS:
    X_el = ex['exposure_kgd'] * KGD_TO_TYR * ex['mass_fraction']
    row = dict(name=ex['name'], element=ex['element'], exposure_kgd=ex['exposure_kgd'], mass_fraction=round(ex['mass_fraction'], 4),
               element_exposure_tyr=X_el, reliability=ex['reliability'], note=ex['note'])
    for d in DELTAS:
        R = rates.get((ex['element'], d, 'annual'), 0.0)
        row[f'N_at_LZ_bestfit_delta{int(d)}'] = R * norm[d]['scale_to_1_event'] * X_el
    exp_rows.append(row)
with open(f'{OUT}/existing_exposures_counts.csv', 'w') as f:
    keys = list(exp_rows[0].keys())
    f.write(','.join(keys) + '\n')
    for r in exp_rows:
        f.write(','.join(('"%s"' % r[k]) if isinstance(r[k], str) else ('%.4g' % r[k] if isinstance(r[k], float) else str(r[k])) for k in keys) + '\n')

# ==============================================================================================
# Part 4: future exposures for 3 events; ROI upper edges
# ==============================================================================================
future = []
for d in DELTAS:
    s = norm[d]['scale_to_1_event']
    for sym, comp, frac in [('I', 'NaI', FRAC['NaI_I']), ('W', 'CaWO4', FRAC['CaWO4_W']), ('Xe', 'Xe', 1.0)]:
        R = rates[(sym, d, 'annual')]
        Em, Ep = lz.E_R_range_keV(M_REF, V_JUNE, A=TARGETS[sym]['A'], delta_kev=d)
        Em_m, Ep_m = lz.E_R_range_keV(M_REF, V_MEAN, A=TARGETS[sym]['A'], delta_kev=d)
        future.append(dict(delta_keV=d, element=sym, compound=comp, mass_fraction=round(frac, 4),
                           events_per_tyr_compound=R * s * frac,
                           exposure_tyr_compound_for_3_events=3.0 / (R * s * frac) if R > 0 else float('inf'),
                           E_minus_keV_june=Em, E_plus_keV_june=Ep, E_minus_keV_mean=Em_m, E_plus_keV_mean=Ep_m,
                           E_plus_keVee_NaI=(Ep * QI if sym == 'I' else float('nan'))))
with open(f'{OUT}/future_exposures.csv', 'w') as f:
    keys = list(future[0].keys()); f.write(','.join(keys) + '\n')
    for r in future:
        f.write(','.join(('%.4g' % r[k]) if isinstance(r[k], float) else str(r[k]) for k in keys) + '\n')
print('Part 4 done')

# ==============================================================================================
# Part 5: digitise the LZ Fig. S7 comparison curves (same axis calibration as P007)
# ==============================================================================================
import pymupdf
page = pymupdf.open('inputs/arXiv_2609.02823_source/FigS7_O1_as_SI.pdf')[0]
dr = page.get_drawings()
X0PX, X1PX, X0, X1 = 70.7, 491.2, 0.0, 350.0
YLAB = [(331.1, -47), (278.6, -45), (226.0, -43), (173.4, -41), (120.9, -39), (68.3, -37), (15.8, -35)]
slope, icpt = np.polyfit([p for p, _ in YLAB], [v for _, v in YLAB], 1)
def conv(idx):
    pts = []
    for it in dr[idx]['items']:
        if it[0] == 'l':
            pts += [(it[1].x, it[1].y), (it[2].x, it[2].y)]
    pts = sorted(set(pts))
    return [(X0 + (x - X0PX) / (X1PX - X0PX) * (X1 - X0), 10 ** (slope * y + icpt)) for x, y in pts]
CURVES = {'CRESST-II 2016 (orange)': 94, 'PICO 2023 (red)': 90, 'XENON1T 2018 (green)': 93,
          'PandaX-4T 2021 (blue)': 92, 'LZ 2024 (violet)': 91, 'LZ 2026 upper edge': 96, 'LZ 2026 lower edge': 97}
curves = {k: conv(i) for k, i in CURVES.items()}
def interp_log(curve, x):
    xs = np.array([p[0] for p in curve]); ys = np.log10([p[1] for p in curve])
    if x < xs.min() - 1 or x > xs.max() + 1:
        return float('nan')
    return float(10 ** np.interp(x, xs, ys))
figS7 = {}
for k, c in curves.items():
    xs = [p[0] for p in c]
    figS7[k] = dict(delta_min_keV=round(min(xs), 1), delta_max_keV=round(min(max(xs), 350.0), 1),
                    sigma_at={str(int(d)): interp_log(c, d) for d in (250.0, 300.0, 350.0)})
comp = []
for d in (250.0, 300.0, 350.0):
    up = figS7['LZ 2026 upper edge']['sigma_at'][str(int(d))]; lo = figS7['LZ 2026 lower edge']['sigma_at'][str(int(d))]
    for k in ['CRESST-II 2016 (orange)', 'PICO 2023 (red)', 'XENON1T 2018 (green)', 'PandaX-4T 2021 (blue)', 'LZ 2024 (violet)']:
        s = figS7[k]['sigma_at'][str(int(d))]
        comp.append(dict(delta_keV=d, curve=k, sigma_cm2=s, LZ_upper_cm2=up, LZ_lower_cm2=lo,
                         ratio_to_LZ_upper=s / up if s == s else float('nan'),
                         ratio_to_LZ_bestfit=s / norm[d]['sigma_SI_bestfit_cm2'] if s == s else float('nan')))
with open(f'{OUT}/figS7_comparison.csv', 'w') as f:
    keys = list(comp[0].keys()); f.write(','.join(keys) + '\n')
    for r in comp:
        f.write(','.join(('"%s"' % r[k]) if isinstance(r[k], str) else '%.4g' % r[k] for k in keys) + '\n')
json.dump(dict(curves_summary=figS7, curves_points={k: [(round(x, 2), float('%.4g' % y)) for x, y in v] for k, v in curves.items()}),
          open(f'{OUT}/figS7_curves.json', 'w'), indent=1)

# consistency check: count-based PICO CF3I expectation vs the digitised PICO-2023 / LZ-upper ratio
pico_I = [r for r in exp_rows if r['element'] == 'I' and 'CF3I' in r['name']][0]
check = {}
for d in (300.0, 350.0):
    N_pico_at_LZ_upper = pico_I[f'N_at_LZ_bestfit_delta{int(d)}'] * 3.65     # P007: LZ upper edge ~ 3.65 events (Sun-frame halo)
    sig_ratio = comp[[i for i, r in enumerate(comp) if r['delta_keV'] == d and 'PICO' in r['curve']][0]]['ratio_to_LZ_upper']
    check[str(int(d))] = dict(N_pico_CF3I_iodine_at_LZ_upper_edge=N_pico_at_LZ_upper,
                              implied_sigma_ratio_for_2p3_event_limit=2.3 / N_pico_at_LZ_upper if N_pico_at_LZ_upper > 0 else float('inf'),
                              digitised_sigma_ratio_PICO_over_LZ_upper=sig_ratio)
print('Part 5 done. Fig. S7 ratios to LZ upper edge:', [(r['curve'], int(r['delta_keV']), '%.3g' % r['ratio_to_LZ_upper']) for r in comp])
print('  PICO CF3I cross-check:', check)

# NaI(Tl) background context for DAMA/LIBRA (recalled, likely): single-hit rate ~1 cpd/kg/keVee in the 10-60 keVee range
DAMA_BKG_CPD_KG_KEV = 1.0
dama = [r for r in exp_rows if 'DAMA' in r['name']][0]
nai_context = {}
for d in DELTAS:
    Em, Ep = lz.E_R_range_keV(M_REF, V_JUNE, A=127, delta_kev=d)
    roi_keVee = (Em * QI, Ep * QI)
    nai_context[str(int(d))] = dict(roi_keVee=roi_keVee, signal_events=dama[f'N_at_LZ_bestfit_delta{int(d)}'],
                                    background_counts=DAMA_BKG_CPD_KG_KEV * dama['exposure_kgd'] * (roi_keVee[1] - roi_keVee[0]))

# ==============================================================================================
# Summary JSON
# ==============================================================================================
summary = dict(
    nai_dama_context=nai_context, dama_bkg_cpd_kg_keVee_recalled=DAMA_BKG_CPD_KG_KEV, Q_I_recalled=QI,
    v_max_kms=dict(june16=V_JUNE, annual_mean=V_MEAN, dec16=V_DEC), v_esc_kms=VESC, m_ref_GeV=M_REF, deltas_keV=DELTAS,
    delta_max_june_1000GeV={s: round(delta_ceiling_kev(1000, t['A'], V_JUNE), 1) for s, t in TARGETS.items()},
    delta_max_june_400GeV={s: round(delta_ceiling_kev(400, t['A'], V_JUNE), 1) for s, t in TARGETS.items()},
    delta_max_june_4000GeV={s: round(delta_ceiling_kev(4000, t['A'], V_JUNE), 1) for s, t in TARGETS.items()},
    delta_max_mean_1000GeV={s: round(delta_ceiling_kev(1000, t['A'], V_MEAN), 1) for s, t in TARGETS.items()},
    A_min_june=amin, A_min_infinite_mass=amin_inf,
    lz_normalisation={str(int(d)): v for d, v in norm.items()},
    window_rates_unit_coupling_per_tyr={f'{s}_{int(d)}_{h}': v for (s, d, h), v in rates.items()},
    ratio_to_Xe_annual={f'{s}_{int(d)}': (rates[(s, d, 'annual')] / rates[('Xe', d, 'annual')]) for s in RATE_TARGETS for d in DELTAS},
    june_over_annual={f'{s}_{int(d)}': (rates[(s, d, 'june16')] / rates[(s, d, 'annual')] if rates[(s, d, 'annual')] > 0 else None) for s in RATE_TARGETS for d in DELTAS},
    isovector_weight_rel_Xe={s: isov[s] / isov['Xe'] for s in TARGETS},
    mass_fractions=FRAC, existing_exposures=exp_rows, future=future, figS7=figS7, figS7_comparison=comp, pico_crosscheck=check,
    runtime_s=time.time() - T0)
json.dump(summary, open(f'{OUT}/P015_summary.json', 'w'), indent=1, default=float)

# ==============================================================================================
# Figures (palette: dataviz reference categorical order blue, orange, aqua, yellow, magenta)
# ==============================================================================================
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#898781', '#e6e5e1', '#fcfcfb'
COL = {'400': '#2a78d6', '1000': '#eb6834', '4000': '#1baf7a', 'Xe': '#2a78d6', 'I': '#eb6834', 'W': '#1baf7a', 'Ge': '#eda100'}
plt.rcParams.update({'font.size': 9.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': SURF, 'axes.facecolor': SURF})

# Fig 1: kinematic ceiling vs A
fig, ax = plt.subplots(figsize=(7.2, 4.4))
Agrid = np.linspace(8, 240, 300)
for m in MASSES:
    ax.plot(Agrid, [delta_ceiling_kev(m, a, V_JUNE) for a in Agrid], color=COL[str(int(m))], lw=2, label=f'{int(m)} GeV, 16 June')
ax.plot(Agrid, [delta_ceiling_kev(1000, a, V_MEAN) for a in Agrid], color=COL['1000'], lw=1.2, ls='--', label='1000 GeV, annual mean')
for d, lab in [(250, '250'), (300, '300'), (350, '350'), (380, '380 keV')]:
    ax.axhline(d, color=MUTED, lw=0.7, ls=':'); ax.text(242, d + 4, lab, color=INK2, fontsize=8, ha='right')
ax.axhspan(300, 390, color='#f0efec', alpha=0.8, lw=0, zorder=0)
for s, t in TARGETS.items():
    y = delta_ceiling_kev(1000, t['A'], V_JUNE)
    ax.plot(t['A'], y, 'o', color=INK, ms=4.5, zorder=5)
    ax.annotate(s, (t['A'], y), xytext=(4, -11 if s in ('Ca', 'Xe') else 5), textcoords='offset points', fontsize=8.5, color=INK)
ax.set_xlim(8, 245); ax.set_ylim(0, 700); ax.set_xlabel('mass number A'); ax.set_ylabel(r'kinematic ceiling $\delta_{\max}=\mu v_{\max}^2/2$ [keV]')
ax.set_title('Which nuclei can absorb a 300-390 keV splitting? (grey band = LZ-favoured window)', fontsize=10, loc='left', color=INK)
ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=8.5, loc='upper left')
fig.tight_layout(); fig.savefig(f'{FIG}/P015_fig1_delta_max_vs_A.png', dpi=170); plt.close(fig)

# Fig 2: spectra at delta = 300 and 350 keV (annual halo, 1000 GeV, LZ-best-fit normalisation), Xe / I / W
fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=True)
for ax, d in zip(axes, (300.0, 350.0)):
    s = norm[d]['scale_to_1_event']
    for sym in ('Xe', 'I', 'W'):
        E, r = spectra[(sym, d, 'annual')]
        ax.plot(E, r * s, color=COL[sym], lw=2, label=f'{sym} (A={TARGETS[sym]["A"]})')
        i = int(np.argmax(r)); ax.text(E[i] * 1.12, min(r[i] * s, 1.5), sym, color=INK, fontsize=8.5, ha='left', va='center')
    ax.axvspan(0, 5.4, color='#f0efec', lw=0); ax.axvspan(269.9, 1400, color='#f0efec', lw=0)
    ax.axvline(248, color='#e34948', lw=1); ax.text(255, 2.5e-6, 'LZ event', color='#e34948', fontsize=8, rotation=90, va='bottom')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(20, 1400); ax.set_ylim(1e-6, 5.0)
    ax.set_xlabel('nuclear recoil energy [keV]'); ax.set_title(fr'$\delta$ = {int(d)} keV, $m_\chi$ = 1 TeV, annual-mean halo', fontsize=10, loc='left', color=INK)
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
axes[0].set_ylabel('dR/dE at LZ best fit [events / (t yr keV) of element]'); axes[0].legend(frameon=False, fontsize=8.5, loc='lower left')
axes[1].text(300, 2.0, 'shaded: outside the LZ ROI (5.4-270 keV)', color=INK2, fontsize=8)
fig.tight_layout(); fig.savefig(f'{FIG}/P015_fig2_spectra_Xe_I_W.png', dpi=170); plt.close(fig)

# Fig 3: expected counts in existing exposures at the LZ best fit (delta = 300, 350 keV)
sel = [r for r in exp_rows if r['element'] in ('I', 'W', 'F', 'Ar') and 'PLR' not in r['name']]
fig, ax = plt.subplots(figsize=(7.6, 5.0))
y = np.arange(len(sel))[::-1]
for j, (d, c, mk) in enumerate([(300.0, '#2a78d6', 'o'), (350.0, '#eb6834', 's')]):
    vals = np.array([r[f'N_at_LZ_bestfit_delta{int(d)}'] for r in sel])
    ok = vals > 0
    ax.scatter(vals[ok], (y + (0.12 if j == 0 else -0.12))[ok], color=c, s=42, marker=mk, label=fr'$\delta$ = {int(d)} keV', zorder=4, edgecolor=SURF, lw=1)
for yi, r in zip(y, sel):
    if r['N_at_LZ_bestfit_delta300'] == 0:
        ax.text(1.3e-6, yi, '0 (kinematically blind)', color=INK2, fontsize=8.5, va='center')
ax.set_xscale('log'); ax.set_xlim(1e-6, 3)
ax.axvline(1.0, color=INK2, lw=0.8, ls='--'); ax.text(1.08, len(sel) - 0.9, 'LZ: 1 event', color=INK2, fontsize=8)
ax.set_yticks(y); ax.set_yticklabels([r['name'].split(' (')[0] + f"\n[{r['element']}: {r['element_exposure_tyr']:.2g} t yr]" for r in sel], fontsize=8.5)
ax.set_xlabel('expected events at the LZ best fit\n(all recoils in the kinematic window counted, efficiency 1; recalled exposures)')
ax.set_title('Existing exposures normalised to the LZ event', fontsize=10, loc='left', color=INK)
ax.grid(axis='x', color=GRID, lw=0.6); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=8.5, loc='lower left')
fig.tight_layout(); fig.savefig(f'{FIG}/P015_fig3_existing_exposures.png', dpi=170); plt.close(fig)
print('All done in %.0f s' % (time.time() - T0))
