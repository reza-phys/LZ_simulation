"""
P007 -- Higgsino-like inelastic dark matter confronting the LZ 248 keV event.

Run from the simulation root:   .venv/bin/python output/code/P007_higgsino_inelastic.py

Parts
  A  Z-exchange couplings of a pure Higgsino doublet -> O1 couplings (Anand and WimPyDD conventions),
     per-nucleon cross-sections, the equivalent isoscalar coupling and the paper's vector factor (~3.2).
  B  Validation of the WimPyDD coupling convention (delta = 0, isoscalar vs Helm; proton/neutron ratio).
  C  Digitisation of the LZ two-sided 90% intervals (Fig. 6 top, Fig. S7) from the PDF vector paths.
  D  Expected LZ counts N(m, delta) for the fixed Higgsino coupling, several halos and efficiency widths,
     using WimPyDD per-stream kernels dotted with the halo functions (exact, linear in delta_eta).
  E  delta giving N = 0.3, 1, 3 (and the one-event PLR band 0.105-3.65); comparison with the paper's
     intervals; June/December modulation; forecast for the next 1000 live days.
  F  Neutralino mass matrix: gaugino masses needed for delta = 300-390 keV; chargino-neutral splitting.
  G  Relic-density scaling, chi2 lifetime estimate (order of magnitude).
Outputs -> output/work/P007/*.csv, *.json  (figures are made by P007_higgsino_figures.py)
"""
from __future__ import annotations
import sys, os, json, math, time
import numpy as np
sys.path.insert(0, 'output/code')
from common import lzcommon as lz

OUT = 'output/work/P007'
os.makedirs(OUT, exist_ok=True)
T0 = time.time()

# ----------------------------------------------------------------------------------------------
# Part A: Z-exchange couplings of a pure Higgsino
# ----------------------------------------------------------------------------------------------
GF = 1.166e-5            # GeV^-2   (recalled, certain)
SW2 = 0.231              # sin^2 theta_W at low energy (recalled, certain)
MZ = 91.19               # GeV      (recalled, certain)
MW = 80.37               # GeV      (recalled, certain)
MV = lz.M_V_GEV          # 246.2 GeV, the paper's m_v
Z_XE = 54
GEV2_TO_CM2 = lz.GEV_TO_CM2
MN = lz.M_NUCLEON_GEV

# Vector charges of the nucleons under the Z (g_V^f = T3/2 - Q sin^2 theta_W, summed over valence quarks)
gV_p = 0.25 * (1.0 - 4.0 * SW2)
gV_n = -0.25
# Dirac neutral Higgsino: L_Z = (g/2c_W) psibar gamma^mu psi Z_mu (pure vector, |T3|=1/2, Q=0).
# Integrating out the Z with g^2/(c_W^2 m_Z^2) = 4 sqrt2 G_F gives
#   L_eff = 2 sqrt2 G_F g_V^N (psibar gamma^mu psi)(Nbar gamma_mu N)  ->  O1 coupling c_N = 2 sqrt2 G_F g_V^N
c_p = 2.0 * math.sqrt(2.0) * GF * gV_p        # = (G_F/sqrt2)(1-4 s_W^2)
c_n = 2.0 * math.sqrt(2.0) * GF * gV_n        # = -G_F/sqrt2
# Anand et al. isoscalar/isovector; WimPyDD's c^0 = c_p + c_n (validated in Part B)
c0_anand, c1_anand = 0.5 * (c_p + c_n), 0.5 * (c_p - c_n)
c0_wd, c1_wd = c_p + c_n, c_p - c_n

def sigma_N_cm2(cN, m_chi):
    mu = lz.mu_red(m_chi, MN)
    return cN**2 * mu**2 / math.pi * GEV2_TO_CM2

def vector_factor(A, Z=Z_XE):
    return (A / ((A - Z) - (1.0 - 4.0 * SW2) * Z))**2

def c_eq_isoscalar(A, Z=Z_XE):
    """Isoscalar O1 coupling giving the same nuclear amplitude c_p Z + c_n N on nucleus (A,Z)."""
    return abs(c_p * Z + c_n * (A - Z)) / A

A_mean = lz.A_XE_MEAN
partA = dict(
    GF_GeV_m2=GF, sin2thetaW=SW2, gV_p=gV_p, gV_n=gV_n,
    c_p_GeV_m2=c_p, c_n_GeV_m2=c_n, c0_anand=c0_anand, c1_anand=c1_anand, c1_over_c0=c1_anand / c0_anand,
    c0_wimpydd=c0_wd, c1_wimpydd=c1_wd,
    c0_anand_mv2_sq=(c0_anand * MV**2)**2, c1_anand_mv2_sq=(c1_anand * MV**2)**2,
    sigma_n_cm2={m: sigma_N_cm2(c_n, m) for m in [300, 500, 1000, 2000, 4000]},
    sigma_p_cm2_1000=sigma_N_cm2(c_p, 1000.0),
    sigma_p_over_sigma_n=(1 - 4 * SW2)**2,
    sigma_n_heavy_limit_cm2=GF**2 * MN**2 / (2 * math.pi) * GEV2_TO_CM2,
    vector_factor_Amean=vector_factor(A_mean),
    vector_factor_by_isotope={A: vector_factor(A) for A in lz.XE_ISOTOPES},
    vector_factor_abundance_weighted_amplitude=(A_mean / sum(f * abs((A - Z_XE) - (1 - 4 * SW2) * Z_XE) for A, f in lz.XE_ISOTOPES.items()))**2,
    c_eq_isoscalar_Amean_GeV_m2=c_eq_isoscalar(A_mean),
    c_eq_isoscalar_mv2_sq=(c_eq_isoscalar(A_mean) * MV**2)**2,
    sigma_SI_equiv_cm2_1000=sigma_N_cm2(c_eq_isoscalar(A_mean), 1000.0),
    sigma_SI_equiv_cm2_by_mass={m: sigma_N_cm2(c_eq_isoscalar(A_mean), m) for m in [300, 500, 1000, 2000, 4000]},
)
# nuclear-level cross-section at q->0 for 131Xe, 1000 GeV, for the record: sigma_A = G_F^2 mu_A^2/(2 pi) [(1-4s)Z-N]^2
muA = lz.mu_red(1000.0, lz.m_nucleus_gev(131))
partA['sigma_A131_q0_cm2_1000'] = GF**2 * muA**2 / (2 * math.pi) * ((1 - 4 * SW2) * Z_XE - 77)**2 * GEV2_TO_CM2
json.dump(partA, open(f'{OUT}/higgsino_couplings.json', 'w'), indent=1)
print('Part A: c_p=%.3e c_n=%.3e GeV^-2; sigma_n(1 TeV)=%.3e cm^2; (c_eq^s m_v^2)^2=%.4f; sigma_SI,eq=%.3e cm^2; vector factor=%.3f'
      % (c_p, c_n, partA['sigma_n_cm2'][1000], partA['c_eq_isoscalar_mv2_sq'], partA['sigma_SI_equiv_cm2_1000'], partA['vector_factor_Amean']))

# ----------------------------------------------------------------------------------------------
# Part B: WimPyDD convention validation
# ----------------------------------------------------------------------------------------------
WD = lz.wd()
VGRID = np.linspace(0.0, 830.0, 1661)          # common vmin grid (km/s), 0.5 km/s steps, beyond v_sun+v_E+v_esc
halo_sun = lz.wd_halo(vmin=VGRID)               # Sun-frame SHM (no Earth orbital motion) -- WimPyDD default usage
c0v, c1v = lz.wd_c_SI_from_sigma_n(1e-45, 1000.0)
ham_si = lz.wd_hamiltonian('si_test', {1: (c0v, c1v)})
valB = []
for e in [10.0, 20.0, 30.0, 40.0, 50.0]:
    wdv = lz.wd_rate(ham_si, 1000.0, e, halo=halo_sun)
    helm = lz.dRdE_SI(e, 1000.0, 1e-45, v_e=lz.v_earth_kms())
    valB.append(dict(E_keV=e, wimpydd=wdv, helm=helm, ratio=wdv / helm))
cN = c0v / 2
hp = lz.wd_hamiltonian('p_only', {1: (cN, cN)}); hn = lz.wd_hamiltonian('n_only', {1: (cN, -cN)})
iso = []
for e in [0.5, 2.0, 5.0]:
    iso.append(dict(E_keV=e, n_over_p=lz.wd_rate(hn, 1000.0, e, halo=halo_sun) / lz.wd_rate(hp, 1000.0, e, halo=halo_sun),
                    expected_q0=sum(f * (A - 54)**2 for A, f in lz.XE_ISOTOPES.items()) / 54**2))
json.dump(dict(isoscalar_vs_helm=valB, neutron_over_proton=iso), open(f'{OUT}/wimpydd_convention_check.json', 'w'), indent=1)
print('Part B: WimPyDD/Helm ratios', [round(v['ratio'], 3) for v in valB], '; n/p ratio', [round(v['n_over_p'], 3) for v in iso])

# ----------------------------------------------------------------------------------------------
# Part C: digitise the two-sided intervals from the PDF vector paths
# ----------------------------------------------------------------------------------------------
import pymupdf

def digitise(pdf, idx_upper, idx_lower, idx_median, x0px, x1px, x0val, x1val, ylabels):
    """ylabels: list of (pixel_y_centre, log10 value) for the decade ticks."""
    page = pymupdf.open(pdf)[0]
    dr = page.get_drawings()
    ys = np.array([p for p, _ in ylabels]); lv = np.array([v for _, v in ylabels])
    slope, icpt = np.polyfit(ys, lv, 1)
    def conv(idx):
        pts = []
        for it in dr[idx]['items']:
            if it[0] == 'l':
                pts.append((it[1].x, it[1].y)); pts.append((it[2].x, it[2].y))
        pts = sorted(set(pts))
        return [(x0val + (x - x0px) / (x1px - x0px) * (x1val - x0val), 10**(slope * y + icpt)) for x, y in pts]
    return dict(upper=conv(idx_upper), lower=conv(idx_lower), median=conv(idx_median) if idx_median is not None else None)

# Fig. 6 top: x tick centres 0 -> 66.7 pt, 350 -> 487.2 pt; decade labels 10^-9 at y=270.2 ... 10^-2 at y=32.2
fig6 = digitise('inputs/arXiv_2609.02823_source/Fig6_O1_L10_limit_stacked.pdf', 229, 230, 228,
                66.7, 487.2, 0.0, 350.0, [(270.2, -9), (236.2, -8), (202.1, -7), (168.2, -6), (134.1, -5), (100.2, -4), (66.2, -3), (32.2, -2)])
figS7 = digitise('inputs/arXiv_2609.02823_source/FigS7_O1_as_SI.pdf', 96, 97, None,
                 70.7, 491.2, 0.0, 350.0, [(331.1, -47), (278.6, -45), (226.0, -43), (173.4, -41), (120.9, -39), (68.3, -37), (15.8, -35)])
json.dump(dict(fig6_top_c1s_mv2_sq=fig6, figS7_sigma_SI_cm2=figS7), open(f'{OUT}/lz_intervals_digitised.json', 'w'), indent=1)

def interp_log(curve, x):
    xs = np.array([p[0] for p in curve]); ys = np.log10([p[1] for p in curve])
    return float(10**np.interp(x, xs, ys))

with open(f'{OUT}/lz_intervals_digitised.csv', 'w') as f:
    f.write('delta_keV,c1s_mv2_sq_lower,c1s_mv2_sq_median,c1s_mv2_sq_upper,sigmaSI_lower_cm2,sigmaSI_upper_cm2,sigmaSI_from_c_upper_cm2\n')
    for d in [0, 50, 100, 150, 200, 250, 300, 350]:
        lo = interp_log(fig6['lower'], d) if d >= 100 else float('nan')
        loS = interp_log(figS7['lower'], d) if d >= 100 else float('nan')
        cu = interp_log(fig6['upper'], d)
        mu1000 = lz.mu_red(1000.0, MN)
        f.write(f"{d},{lo:.4g},{interp_log(fig6['median'], d):.4g},{cu:.4g},{loS:.4g},{interp_log(figS7['upper'], d):.4g},{cu * mu1000**2 / (math.pi * MV**4) * GEV2_TO_CM2:.4g}\n")
print('Part C: Fig.6 upper/lower at 300 keV = %.3g / %.3g ; at 350 keV = %.3g / %.3g' % (
    interp_log(fig6['upper'], 300), interp_log(fig6['lower'], 300), interp_log(fig6['upper'], 350), interp_log(fig6['lower'], 350)))

# ----------------------------------------------------------------------------------------------
# Part D: expected counts N(m, delta) with per-stream kernels
# ----------------------------------------------------------------------------------------------
EXPOSURE = lz.LZ['exposure_tyr']               # 2.84 t yr
E50_HI, E50_LO, PLATEAU = lz.LZ['E_50pct_high_keV'], lz.LZ['E_50pct_low_keV'], lz.LZ['eff_plateau']
from scipy.special import erf, erfc

def efficiency(E, sig_hi=11.5, sig_lo=2.5):
    """LZ NR efficiency model: plateau 0.96 (paper), 50% at 5.4 and 269.9 keV (paper);
    error-function edges. sig_hi = 11.5 keV reproduces the Fig. S2 inset (75% at ~262, 25% at ~278 keV)."""
    return PLATEAU * 0.5 * (1 + erf((E - E50_LO) / (math.sqrt(2) * sig_lo))) * 0.5 * erfc((E - E50_HI) / (math.sqrt(2) * sig_hi))

# halos on the common grid
days12 = 15.0 + 365.25 / 12 * np.arange(12)
halos = {'sun': halo_sun[1],
         'annual': np.mean([lz.wd_halo(day_of_year=d, vmin=VGRID)[1] for d in days12], axis=0),
         'june16': lz.wd_halo(day_of_year=167, vmin=VGRID)[1],
         'dec16': lz.wd_halo(day_of_year=350, vmin=VGRID)[1],
         'annual_vesc528': np.mean([lz.wd_halo(day_of_year=d, vmin=VGRID, vesc=528.0)[1] for d in days12], axis=0),
         'annual_vesc560': np.mean([lz.wd_halo(day_of_year=d, vmin=VGRID, vesc=560.0)[1] for d in days12], axis=0)}
ONES = np.ones_like(VGRID)
ham_higgsino = lz.wd_hamiltonian('higgsino_Z', {1: (c0_wd, c1_wd)})
ham_iso_unit = lz.wd_hamiltonian('iso_unit', {1: (2.0 / MV**2, 0.0)})   # c_p = c_n = 1/m_v^2 (paper's unit coupling)
MASSES = [300.0, 500.0, 1000.0, 2000.0, 4000.0]
DELTAS = np.concatenate([np.arange(0, 250, 50.0), np.arange(250, 345, 10.0), np.arange(345, 402.5, 2.5)])
EFF_SIGS = [8.0, 11.5, 15.0]
E_MAX = 330.0

def spectrum_kernels(ham, m, delta):
    """Return E grid and dR/dE (events/t/yr/keV) for each halo, via per-stream kernels."""
    lo = lz.E_R_range_keV(m, VGRID[-1], A=124.0, delta_kev=delta)[0]
    if math.isnan(lo):
        return None, None
    E = np.arange(max(1.0, math.floor(lo) - 4.0), E_MAX + 1e-9, 2.0)
    K = np.array([WD.diff_rate(WD.Xe, ham, m, float(e), VGRID, ONES, j_chi=0.5, delta=float(delta), sum_over_streams=False) for e in E])
    K *= 1000.0 * 365.25       # /kg/day -> /t/yr
    return E, {name: K @ deta for name, deta in halos.items()}

rows = []
spectra_store = {}
CACHED = os.path.exists(f'{OUT}/N_events_grid.csv') and '--recompute' not in sys.argv
for m in ([] if CACHED else MASSES):
    for d in DELTAS:
        E, rates = spectrum_kernels(ham_higgsino, m, d)
        if E is None:
            for name in halos:
                for s in EFF_SIGS:
                    rows.append((m, d, name, s, 0.0, 0.0))
            continue
        if m == 1000.0 and d in (0.0, 200.0, 300.0, 350.0, 370.0, 380.0, 390.0):
            spectra_store[int(d)] = dict(E=E.tolist(), **{k: v.tolist() for k, v in rates.items()})
        for name, r in rates.items():
            raw = float(np.trapezoid(r, E)) * EXPOSURE
            for s in EFF_SIGS:
                rows.append((m, d, name, s, float(np.trapezoid(r * efficiency(E, sig_hi=s), E)) * EXPOSURE, raw))
    print(f'  mass {m:.0f} done, t={time.time() - T0:.0f}s', flush=True)

import pandas as pd
if CACHED:
    print('Part D: using cached N_events_grid.csv (pass --recompute to redo the WimPyDD loop)')
    df = pd.read_csv(f'{OUT}/N_events_grid.csv')
else:
    df = pd.DataFrame(rows, columns=['m_GeV', 'delta_keV', 'halo', 'eff_sigma_keV', 'N_events', 'N_noeff_all_E'])
    df.to_csv(f'{OUT}/N_events_grid.csv', index=False)
    json.dump(spectra_store, open(f'{OUT}/spectra_1000GeV.json', 'w'))

# isoscalar unit-coupling counts at delta = 250, 300, 350 for 1000 GeV -> implied N at the paper's interval edges
iso_rows = []
for d in [250.0, 300.0, 350.0]:
    E, rates = spectrum_kernels(ham_iso_unit, 1000.0, d)
    for name in ['sun', 'annual', 'june16']:
        N_unit = float(np.trapezoid(rates[name] * efficiency(E), E)) * EXPOSURE      # for (c^s m_v^2)^2 = 1
        for edge in ['upper', 'median', 'lower']:
            if edge == 'lower' and d < 100:
                continue
            cval = interp_log(fig6[edge], d)
            iso_rows.append(dict(delta_keV=d, halo=name, edge=edge, c1s_mv2_sq=cval, N_events_at_edge=N_unit * cval, N_unit=N_unit))
dfi = pd.DataFrame(iso_rows); dfi.to_csv(f'{OUT}/implied_N_at_LZ_interval_edges.csv', index=False)
print('Part D: implied N at Fig.6 edges (annual halo):')
print(dfi[dfi.halo == 'annual'].to_string(index=False))

# ----------------------------------------------------------------------------------------------
# Part E: delta for N = 0.3, 1, 3 (and PLR band); comparison; modulation; forecast
# ----------------------------------------------------------------------------------------------
def delta_for_N(sub, target):
    """Interpolate log N(delta) on the falling branch (delta >= 250) to find delta with N = target."""
    s = sub[(sub.delta_keV >= 250)].sort_values('delta_keV')
    x = s.delta_keV.values; y = np.log10(np.maximum(s.N_events.values, 1e-8))   # zeros -> floor, so a crossing is bracketed
    lt = math.log10(target)
    for i in range(len(x) - 1):
        if (y[i] - lt) * (y[i + 1] - lt) <= 0 and y[i] != y[i + 1]:
            return float(x[i] + (lt - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
    return float('nan')

TARGETS = {'N3.65': 3.65, 'N3': 3.0, 'N1': 1.0, 'N0.3': 0.3, 'N0.105': 0.105}
res_rows = []
for m in MASSES:
    vjune = lz.vmax_kms(lz.v_earth_kms(167)); vavg = lz.vmax_kms(lz.v_earth_kms())
    dmax248 = lz.delta_max_kev(248.0, m, v_kms=vjune); dmax270 = lz.delta_max_kev(E50_HI, m, v_kms=vjune)
    dmax_abs = lz.mu_red(m, lz.m_nucleus_gev(136)) * (vjune / lz.C_KMS)**2 / 2 * 1e6
    for name in halos:
        for s in EFF_SIGS:
            sub = df[(df.m_GeV == m) & (df.halo == name) & (df.eff_sigma_keV == s)]
            row = dict(m_GeV=m, halo=name, eff_sigma_keV=s, delta_max_248keV_june=dmax248, delta_max_270keV_june=dmax270, delta_max_absolute_june_136Xe=dmax_abs,
                       N_delta0=float(sub[sub.delta_keV == 0].N_events.iloc[0]), N_delta300=float(sub[sub.delta_keV == 300].N_events.iloc[0]),
                       N_delta350=float(sub[sub.delta_keV == 350].N_events.iloc[0]), N_delta370=float(sub[sub.delta_keV == 370].N_events.iloc[0]),
                       N_delta380=float(sub[sub.delta_keV == 380].N_events.iloc[0]))
            for k, t in TARGETS.items():
                row['delta_' + k] = delta_for_N(sub, t)
            res_rows.append(row)
dres = pd.DataFrame(res_rows); dres.to_csv(f'{OUT}/delta_for_N_events.csv', index=False)
print('Part E: delta(N=1) [keV], annual halo, sigma_eff=11.5:')
print(dres[(dres.halo == 'annual') & (dres.eff_sigma_keV == 11.5)][['m_GeV', 'N_delta300', 'N_delta350', 'delta_N3', 'delta_N1', 'delta_N0.3', 'delta_max_248keV_june']].to_string(index=False))

# modulation at delta(N=1): June/Dec and June/annual, and next-1000-live-day forecast
mod_rows = []
for m in MASSES:
    d1 = float(dres[(dres.m_GeV == m) & (dres.halo == 'annual') & (dres.eff_sigma_keV == 11.5)].delta_N1.iloc[0])
    dlist = [300.0, 350.0] + ([round(d1 / 2.5) * 2.5] if (not math.isnan(d1) and d1 >= 345) else ([round(d1 / 10) * 10.0] if not math.isnan(d1) else []))
    for d in dlist:
        g = lambda name: float(df[(df.m_GeV == m) & (df.halo == name) & (df.eff_sigma_keV == 11.5) & (np.isclose(df.delta_keV, d))].N_events.iloc[0])
        Na, Nj, Nd, Ns = g('annual'), g('june16'), g('dec16'), g('sun')
        mod_rows.append(dict(m_GeV=m, delta_keV=d, N_annual=Na, N_june16=Nj, N_dec16=Nd, N_sunframe=Ns,
                             june_over_dec=Nj / Nd if Nd > 0 else float('inf'), june_over_annual=Nj / Na if Na > 0 else float('nan'),
                             N_next_1000_livedays=Na * 1000.0 / lz.LZ['live_days'], N_next_1000_june_only_rate=Nj * 1000.0 / lz.LZ['live_days']))
dmod = pd.DataFrame(mod_rows); dmod.to_csv(f'{OUT}/modulation_and_forecast.csv', index=False)
print(dmod.to_string(index=False))

# comparison with paper: ratio of Higgsino coupling to interval edges (1000 GeV)
comp = {}
for d in [250, 300, 350]:
    comp[d] = dict(c_eq_sq=partA['c_eq_isoscalar_mv2_sq'], upper=interp_log(fig6['upper'], d), lower=interp_log(fig6['lower'], d),
                   ratio_to_upper=partA['c_eq_isoscalar_mv2_sq'] / interp_log(fig6['upper'], d),
                   N_higgsino_annual=float(df[(df.m_GeV == 1000) & (df.halo == 'annual') & (df.eff_sigma_keV == 11.5) & (df.delta_keV == d)].N_events.iloc[0]),
                   N_at_upper_annual=float(dfi[(dfi.delta_keV == d) & (dfi.halo == 'annual') & (dfi.edge == 'upper')].N_events_at_edge.iloc[0]))
    comp[d]['ratio_N'] = comp[d]['N_higgsino_annual'] / comp[d]['N_at_upper_annual']
json.dump(comp, open(f'{OUT}/comparison_with_LZ_intervals.json', 'w'), indent=1)
print('Part E: Higgsino/upper-edge ratios', {d: (round(v['ratio_to_upper'], 2), round(v['ratio_N'], 2)) for d, v in comp.items()})

# ----------------------------------------------------------------------------------------------
# Part F: neutralino mass matrix -> gaugino masses for the required splitting
# ----------------------------------------------------------------------------------------------
import sympy as sp
def neutralino_masses(M1, M2, mu, tanb):
    b = math.atan(tanb); sb, cb = math.sin(b), math.cos(b); sw, cw = math.sqrt(SW2), math.sqrt(1 - SW2)
    M = np.array([[M1, 0, -MZ * sw * cb, MZ * sw * sb], [0, M2, MZ * cw * cb, -MZ * cw * sb],
                  [-MZ * sw * cb, MZ * cw * cb, 0, -mu], [MZ * sw * sb, -MZ * cw * sb, -mu, 0]])
    ev = np.linalg.eigvalsh(M)          # symmetric real matrix; physical masses are |eigenvalues|
    return np.sort(np.abs(ev))

def chargino_masses(M2, mu, tanb):
    b = math.atan(tanb)
    X = np.array([[M2, math.sqrt(2) * MW * math.sin(b)], [math.sqrt(2) * MW * math.cos(b), mu]])
    return np.sqrt(np.sort(np.linalg.eigvalsh(X.T @ X)))

# symbolic leading-order check with sympy: effective 2x2 Higgsino matrix after integrating out gauginos
eps, mu_s, sb_s, cb_s = sp.symbols('epsilon mu s_beta c_beta', positive=True)
MH = sp.Matrix([[-eps * cb_s**2, -mu_s + eps * cb_s * sb_s], [-mu_s + eps * cb_s * sb_s, -eps * sb_s**2]])
lam = MH.eigenvals()
lams = list(lam.keys())
split = sp.simplify(sp.series(sp.Abs(lams[0]) - sp.Abs(lams[1]), eps, 0, 2).removeO().subs(sb_s**2 + cb_s**2, 1))
# numeric: for mu = 1 TeV, find M1 giving delta targets, several gaugino hierarchies
def delta_N_keV(M1, M2, mu, tanb):
    m = neutralino_masses(M1, M2, mu, tanb)
    return (m[1] - m[0]) * 1e6

from scipy.optimize import brentq
splitting_rows = []
for mu in [300.0, 1000.0, 4000.0]:
    for tanb in [2.0, 10.0]:
        for hier, (f1, f2) in {'bino_only(M2=1e9)': (1.0, None), 'M2=2M1': (1.0, 2.0), 'wino_only(M1=1e9)': (None, 1.0), 'M2=-2M1': (1.0, -2.0)}.items():
            for dtarget in [300.0, 350.0, 370.0, 385.0]:
                def fdel(logM):
                    M = 10**logM
                    M1 = M * f1 if f1 is not None else 1e9
                    M2 = M * f2 if f2 is not None else 1e9
                    return delta_N_keV(M1, M2, mu, tanb) - dtarget
                try:
                    logM = brentq(fdel, 3.5, 8.5)
                    M = 10**logM
                    M1 = M * f1 if f1 is not None else float('inf'); M2 = M * f2 if f2 is not None else float('inf')
                    lo = MZ**2 * (SW2 / M1 + (1 - SW2) / M2) * 1e6 if (M1 != float('inf') and M2 != float('inf')) else (MZ**2 * SW2 / M1 * 1e6 if M2 == float('inf') else MZ**2 * (1 - SW2) / M2 * 1e6)
                    mch = chargino_masses(M2 if M2 != float('inf') else 1e9, mu, tanb)[0]
                    mn = neutralino_masses(M1 if M1 != float('inf') else 1e9, M2 if M2 != float('inf') else 1e9, mu, tanb)
                    splitting_rows.append(dict(mu_GeV=mu, tanb=tanb, hierarchy=hier, delta_target_keV=dtarget, M1_TeV=M1 / 1e3, M2_TeV=M2 / 1e3,
                                               leading_order_delta_keV=lo, tree_charged_neutral_MeV=(mch - mn[0]) * 1e3))
                except ValueError:
                    splitting_rows.append(dict(mu_GeV=mu, tanb=tanb, hierarchy=hier, delta_target_keV=dtarget, M1_TeV=float('nan'), M2_TeV=float('nan'),
                                               leading_order_delta_keV=float('nan'), tree_charged_neutral_MeV=float('nan')))
dsp = pd.DataFrame(splitting_rows); dsp.to_csv(f'{OUT}/gaugino_masses_for_splitting.csv', index=False)
print('Part F: sympy leading-order splitting =', split)
print(dsp[(dsp.mu_GeV == 1000) & (dsp.tanb == 10)].to_string(index=False))
# reference numbers: M1 at 10 TeV
partF = dict(sympy_leading_order_splitting=str(split), delta_keV_M1_10TeV_bino_only=delta_N_keV(1e4, 1e9, 1000.0, 10.0),
             delta_keV_M1_100TeV_bino_only=delta_N_keV(1e5, 1e9, 1000.0, 10.0), delta_keV_M1eqM2_1000TeV=delta_N_keV(1e6, 1e6, 1000.0, 10.0),
             radiative_charged_neutral_MeV_recalled=355.0)
json.dump(partF, open(f'{OUT}/splitting_summary.json', 'w'), indent=1)

# ----------------------------------------------------------------------------------------------
# Part G: relic scaling and chi2 lifetime (order of magnitude)
# ----------------------------------------------------------------------------------------------
def omega_h2_thermal(m):        # pure Higgsino: Omega h^2 ~ 0.12 (m/1.1 TeV)^2  (recalled, likely)
    return 0.12 * (m / 1100.0)**2
HBAR_GEV_S = 6.582e-25
partG = dict(thermal_omega_h2={m: omega_h2_thermal(m) for m in MASSES}, thermal_fraction_of_DM={m: min(1.0, omega_h2_thermal(m) / 0.12) for m in MASSES},
             N1_events_rescaled_thermal_fraction={})
for m in MASSES:
    N1 = float(df[(df.m_GeV == m) & (df.halo == 'annual') & (df.eff_sigma_keV == 11.5) & (df.delta_keV == 350)].N_events.iloc[0])
    partG['N1_events_rescaled_thermal_fraction'][m] = N1 * partG['thermal_fraction_of_DM'][m]
for d in [300.0, 350.0, 375.0]:
    dg = d * 1e-6
    Gam = 3 * GF**2 * dg**5 / (192 * math.pi**3) * 2.0     # 3 nu flavours; x2 for the pure-vector chi coupling (order of magnitude)
    partG[f'chi2_lifetime_s_delta{int(d)}'] = HBAR_GEV_S / Gam
json.dump(partG, open(f'{OUT}/relic_and_lifetime.json', 'w'), indent=1)
print('Part G:', {k: (v if not isinstance(v, dict) else {kk: round(vv, 4) for kk, vv in v.items()}) for k, v in partG.items()})
print('total time %.0f s' % (time.time() - T0))
