"""
P036 -- Could the LZ event be a nuclear excitation?
DM-induced 129Xe(39.58 keV) and 131Xe(80.19 keV) transitions and where they land in S1-S2.

Sections
  1. Hybrid (NR + prompt ER) loci in (S1c, log10 S2c) with the LZ-tuned NEST yields; can any
     NR energy + 39.6/80.2 keV ER reproduce (540.1 phd, 9268 phd)?  Two-equation solve.
  2. Rates: WimPyDD O4 elastic rates for 129Xe / 131Xe at 1 TeV, inelastic kinematics via delta=E*,
     recalled inelastic/elastic structure-factor ratio R (bracketed) -> hybrid counts per 2.84 t yr
     for (a) a coupling giving one elastic O4 event in 200-270 keV, (b) a recalled SD-neutron limit.
  3. Kinematics: v_min with threshold E*, recoil window at v_max (16 June), minimum DM mass.
  4. Discussion numbers: ER photon fraction of the hybrid S1, containment (recalled attenuation).

Run from the simulation root:  .venv/bin/python output/code/P036_nuclear_excitation.py
Everything not in the LZ paper or lzcommon is RECALLED and flagged in details.md / P036.json.
"""
import sys, os, math, json, time
sys.path.insert(0, 'output/code')
from common import lzcommon as lz
import numpy as np
import pandas as pd
from scipy import optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

T0 = time.time()
OUT = 'output/work/P036'; FIG = f'{OUT}/figures'
os.makedirs(FIG, exist_ok=True)
LOG = open(f'{OUT}/run_log.txt', 'w')
def log(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.write(s + '\n'); LOG.flush()

R = {}   # results dict -> P036_results.json
g1, g2 = lz.LZ['g1'], lz.LZ['g2']
EV_S1, EV_S2 = lz.LZ['ev_S1c'], lz.LZ['ev_S2c']
EV_LOGS2 = math.log10(EV_S2)
SIG_NR = 0.0313          # dex, NR band at 540 phd (P024 baseline MC; P009 0.033)
SIG_NR_ALT = 0.033
SIG_ER = 150.0 / 1226.0 / math.log(10)   # ~0.053 dex: P010 calibrated sigma_Ne = 150 e at Ne = 1226 (ER band at ~70 keVee)
MCHI = 1000.0
EXPO = lz.LZ['exposure_tyr']

# Nuclear data (recalled; ENSDF) -----------------------------------------------------------
LEVELS = {129: dict(E=39.58, Jgs='1/2+', Jex='3/2+', T12_ns=0.97, alpha_IC=12.0, mult='M1'),
          131: dict(E=80.19, Jgs='3/2+', Jex='1/2+', T12_ns=0.48, alpha_IC=1.6, mult='M1(+E2)')}
log(f'=== P036  nuclear-excitation hybrids;  g1={g1}, g2={g2}, event ({EV_S1}, {EV_S2}) log10S2c={EV_LOGS2:.4f}')
log(f'event quanta: Nph={EV_S1/g1:.0f}, Ne={EV_S2/g2:.1f}')
R['event'] = dict(S1c=EV_S1, S2c=EV_S2, log10S2c=EV_LOGS2, Nph=EV_S1/g1, Ne=EV_S2/g2)

# ==========================================================================================
# 1. Yields and hybrid loci
# ==========================================================================================
log('\n=== 1. NEST-LZ yields and hybrid loci ===')
E_nr = np.arange(10.0, 320.0 + 1e-9, 2.0)
nr = np.array([lz.nest_nr_yields(E) for E in E_nr])            # Table S5 as printed (with p(E) break)
NPH_NR, NE_NR = nr[:, 0], nr[:, 1]
S1_NR, S2_NR = g1 * NPH_NR, g2 * NE_NR
E_er = np.concatenate([np.arange(2.0, 40.0, 1.0), np.arange(40.0, 200.0 + 1e-9, 2.0)])
er = np.array([lz.nest_er_yields(E, params=lz.NEST_ER_LZ) for E in E_er])
NPH_ER, NE_ER = er[:, 0], er[:, 1]
S1_ER, S2_ER = g1 * NPH_ER, g2 * NE_ER
pd.DataFrame(dict(E_keV=E_nr, Nph=NPH_NR, Ne=NE_NR, S1c=S1_NR, log10S2c=np.log10(S2_NR))).to_csv(f'{OUT}/nr_band_median.csv', index=False)
pd.DataFrame(dict(E_keV=E_er, Nph=NPH_ER, Ne=NE_ER, S1c=S1_ER, log10S2c=np.log10(S2_ER))).to_csv(f'{OUT}/er_band_median.csv', index=False)

def nr_logS2_at_S1(s1, scale=1.0):
    """pure-NR median log10 S2c at a given S1c (interpolated on the NEST curve, quanta scaled by `scale`)."""
    return float(np.interp(s1, scale * S1_NR, np.log10(scale * S2_NR)))
def E_nr_at_S1(s1, scale=1.0):
    return float(np.interp(s1, scale * S1_NR, E_nr))
def er_logS2_at_S1(s1):
    return float(np.interp(s1, S1_ER, np.log10(S2_ER)))

nr_med_540 = nr_logS2_at_S1(EV_S1)
er_med_540 = er_logS2_at_S1(EV_S1)
R['pure_NR'] = dict(E_at_S1c540_keV=E_nr_at_S1(EV_S1), logS2c_median_at_540=nr_med_540,
                    event_offset_dex=EV_LOGS2 - nr_med_540, event_offset_sigma=(EV_LOGS2 - nr_med_540) / SIG_NR,
                    ER_median_at_540=er_med_540, E_ER_at_S1c540_keV=float(np.interp(EV_S1, S1_ER, E_er)),
                    event_offset_ER_sigma=(EV_LOGS2 - er_med_540) / SIG_ER)
log(f"pure NR: S1c=540 <-> E_NR={R['pure_NR']['E_at_S1c540_keV']:.0f} keV (Table S5), median log10S2c={nr_med_540:.3f}; "
    f"event {EV_LOGS2 - nr_med_540:+.3f} dex = {R['pure_NR']['event_offset_sigma']:+.2f} sigma (sigma={SIG_NR})")
log(f"ER band at S1c=540: E_ER={R['pure_NR']['E_ER_at_S1c540_keV']:.0f} keV, median log10S2c={er_med_540:.3f}; event {R['pure_NR']['event_offset_ER_sigma']:+.1f} sigma_ER (sigma_ER={SIG_ER:.3f})")

# ER quanta of the de-excitation (full E* deposited locally: IC electron + x-rays, or the gamma)
ERQ = {}
for A, lv in LEVELS.items():
    nph, ne = lz.nest_er_yields(lv['E'], params=lz.NEST_ER_LZ)
    ERQ[A] = dict(E=lv['E'], Nph=nph, Ne=ne, S1c=g1 * nph, log10S2c=math.log10(g2 * ne), Nq=nph + ne, r=ne / (nph + ne))
    log(f"ER {lv['E']} keV (LZ Table S3 beta yields): Nph={nph:.0f}, Ne={ne:.0f} -> alone S1c={g1*nph:.0f} phd, log10S2c={math.log10(g2*ne):.3f}; "
        f"electron fraction {ne/(nph+ne):.3f}")
R['ER_quanta'] = ERQ

# hybrid loci: quanta add
HYB = {}
rows = []
for A, q in ERQ.items():
    S1h = g1 * (NPH_NR + q['Nph']); S2h = g2 * (NE_NR + q['Ne'])
    HYB[A] = dict(S1c=S1h, log10S2c=np.log10(S2h))
    for E, s1, s2 in zip(E_nr, S1h, S2h):
        rows.append(dict(isotope=A, E_star_keV=q['E'], E_NR_keV=E, S1c=s1, log10S2c=math.log10(s2),
                         in_ROI=bool(3 <= s1 <= 600 and 2.75 <= math.log10(s2) <= 4.15),
                         offset_from_NR_median_dex=math.log10(s2) - nr_logS2_at_S1(s1) if s1 <= S1_NR.max() else float('nan')))
df_h = pd.DataFrame(rows); df_h.to_csv(f'{OUT}/hybrid_loci.csv', index=False)

res1 = {}
for A, q in ERQ.items():
    # (i) match S1c: which E_NR, and where does log S2c land?
    S1h = g1 * (NPH_NR + q['Nph'])
    if S1h.min() > EV_S1:
        E_m = float('nan'); l_m = float('nan')
    else:
        E_m = float(np.interp(EV_S1, S1h, E_nr)); nph_m, ne_m = lz.nest_nr_yields(E_m)
        l_m = math.log10(g2 * (ne_m + q['Ne']))
    # (ii) match S2c: Ne_total = 268.6 -> need Ne_NR = 268.6 - Ne_ER
    ne_needed = EV_S2 / g2 - q['Ne']
    # (iii) 248 keV NR + line
    nph248, ne248 = lz.nest_nr_yields(248.0)
    s1_248 = g1 * (nph248 + q['Nph']); l_248 = math.log10(g2 * (ne248 + q['Ne']))
    # minimum log S2c of the locus (E_NR -> 0 limit is the ER alone)
    res1[A] = dict(E_NR_matching_S1c_keV=E_m, log10S2c_at_S1c540=l_m, offset_dex=l_m - nr_med_540,
                   offset_sigma=(l_m - nr_med_540) / SIG_NR, offset_sigma_alt=(l_m - nr_med_540) / SIG_NR_ALT,
                   offset_from_ER_median_sigma=(l_m - er_med_540) / SIG_ER,
                   S2c_ratio_pred_over_obs=10 ** (l_m - EV_LOGS2),
                   Ne_NR_needed_for_S2c=ne_needed, Ne_ER_alone=q['Ne'], S2c_solution_exists=bool(ne_needed > 0),
                   S1c_248_plus_line=s1_248, log10S2c_248_plus_line=l_248,
                   in_ROI_248_plus_line=bool(s1_248 <= 600 and l_248 <= 4.15),
                   min_log10S2c_of_locus=q['log10S2c'], any_locus_point_in_ROI=bool(df_h[(df_h.isotope == A)].in_ROI.any()),
                   ER_photon_fraction_at_540=q['Nph'] / (EV_S1 / g1))
    log(f"hybrid {A}Xe ({q['E']} keV): S1c=540 needs E_NR={E_m:.0f} keV -> log10S2c={l_m:.3f} = {res1[A]['offset_sigma']:+.1f} sigma above NR median "
        f"({res1[A]['offset_from_ER_median_sigma']:+.1f} sigma_ER from ER median), predicted S2c/observed = {res1[A]['S2c_ratio_pred_over_obs']:.2f}; "
        f"S2c match needs Ne_NR = {ne_needed:.0f} e (ER alone {q['Ne']:.0f} e vs observed {EV_S2/g2:.0f}) -> solution exists: {ne_needed > 0}; "
        f"248 keV + line -> ({s1_248:.0f} phd, {l_248:.3f}), in ROI: {res1[A]['in_ROI_248_plus_line']}; any locus point in ROI: {res1[A]['any_locus_point_in_ROI']}")
R['hybrid_solutions'] = res1

# two-equation solve with unknowns (E_NR, E_ER): minimise chi2 in (S1c, log10 S2c); scan E_ER on a grid incl. formal negative values
def hybrid_point(E_NR, E_ER):
    nph, ne = lz.nest_nr_yields(E_NR)
    if E_ER > 0.05:
        nphe, nee = lz.nest_er_yields(E_ER, params=lz.NEST_ER_LZ)
    else:
        nphe, nee = 0.0, 0.0
    return g1 * (nph + nphe), math.log10(g2 * (ne + nee))
SIG_S1 = 0.03 * EV_S1   # ~3% S1c resolution proxy (P009: g1 +-2%, photon statistics ~1.4%)
scan = []
for E_ER in [0.0, 1, 2, 5, 10, 20, 30, 39.58, 50, 60, 80.19, 100]:
    S1h = np.array([hybrid_point(E, E_ER)[0] for E in E_nr]);
    E_m = float(np.interp(EV_S1, S1h, E_nr)) if S1h.min() <= EV_S1 else float('nan')
    l = hybrid_point(E_m, E_ER)[1] if not math.isnan(E_m) else float('nan')
    scan.append(dict(E_ER_keV=E_ER, E_NR_at_S1c540_keV=E_m, log10S2c=l, residual_dex=l - EV_LOGS2, residual_sigma=(l - EV_LOGS2) / SIG_NR))
df_scan = pd.DataFrame(scan); df_scan.to_csv(f'{OUT}/two_equation_scan.csv', index=False)
slope = float(np.polyfit(df_scan.E_ER_keV[:5], df_scan.residual_dex[:5], 1)[0])
log('two-equation scan (S1c fixed at 540.1 by E_NR; residual in log10 S2c vs observed):')
for r_ in scan: log(f"   E_ER={r_['E_ER_keV']:6.2f} keV  E_NR={r_['E_NR_at_S1c540_keV']:6.1f} keV  log10S2c={r_['log10S2c']:.3f}  residual={r_['residual_dex']:+.3f} dex ({r_['residual_sigma']:+.1f} sigma)")
log(f'  d(residual)/dE_ER near 0 = {slope:+.4f} dex/keV; residual at E_ER=0 is {scan[0]["residual_dex"]:+.3f} dex -> a root would need E_ER = {-scan[0]["residual_dex"]/slope:+.1f} keV (unphysical)')
R['two_equation'] = dict(scan=scan, slope_dex_per_keV=slope, formal_root_E_ER_keV=-scan[0]['residual_dex'] / slope, S1_sigma_proxy=SIG_S1)

# variant: P009 paper energy scale (+7.7% quanta for NR)
SC = 1.077
res_sc = {}
for A, q in ERQ.items():
    S1h = g1 * (SC * NPH_NR + q['Nph'])
    E_m = float(np.interp(EV_S1, S1h, E_nr)); nph_m, ne_m = lz.nest_nr_yields(E_m)
    l_m = math.log10(g2 * (SC * ne_m + q['Ne'])); med = nr_logS2_at_S1(EV_S1, SC)
    res_sc[A] = dict(E_NR_matching_S1c_keV=E_m, log10S2c=l_m, NR_median=med, offset_sigma=(l_m - med) / SIG_NR, event_offset_sigma=(EV_LOGS2 - med) / SIG_NR)
    log(f"  variant P009 scale x{SC}: {A}Xe hybrid at 540 phd: E_NR={E_m:.0f}, log10S2c={l_m:.3f}, {res_sc[A]['offset_sigma']:+.1f} sigma above NR median {med:.3f} (event {res_sc[A]['event_offset_sigma']:+.2f})")
R['variant_paper_scale'] = res_sc
# variant: NEST default ER yields instead of LZ Table S3
res_def = {}
for A, lv in LEVELS.items():
    nph, ne = lz.nest_er_yields(lv['E'], params=None)
    res_def[A] = dict(Nph=nph, Ne=ne, log10S2c_alone=math.log10(g2 * ne))
    log(f"  variant NEST-default ER yields {lv['E']} keV: Nph={nph:.0f}, Ne={ne:.0f} (LZ: {ERQ[A]['Nph']:.0f}, {ERQ[A]['Ne']:.0f})")
R['variant_nest_default_er'] = res_def

# ==========================================================================================
# 2. WimPyDD rates: O4 elastic and threshold-shifted (excitation) spectra for 129Xe / 131Xe
# ==========================================================================================
log('\n=== 2. WimPyDD O4 rates at 1 TeV ===')
WD = lz.wd()
xe = WD.Xe
ISO = {}
for k, (m, name) in enumerate(zip(xe.mass, xe.isotopes)):
    A = int(round(float(m) / lz.AMU_GEV))
    ISO[A] = dict(idx=k, name=str(name), m=float(m))
log('WimPyDD Xe isotopes (A: index):', {A: d['idx'] for A, d in sorted(ISO.items())})
halo_avg = lz.wd_halo()                 # Baxter SHM, annual average
halo_jun = lz.wd_halo(day_of_year=167)  # 16 June
c0_unit = lz.wd_c_from_anand(1.0 / lz.M_V_GEV ** 2)[0]     # LZ unit coupling O4^s
ham_s = lz.wd_hamiltonian('O4s_unit', {4: (c0_unit, 0.0)})
E_grid = np.concatenate([np.arange(1.0, 60.0, 1.0), np.arange(60.0, 420.0 + 1e-9, 4.0)])

def rate_iso(ham, A, delta=0.0, halo=halo_avg):
    return lz.wd_rate(ham, MCHI, E_grid, halo=halo, delta_kev=delta, isotopes_list={0: [ISO[A]['idx']]})
def integ(E, r, lo, hi):
    m = (E >= lo) & (E <= hi)
    if m.sum() < 2: return 0.0
    return float(np.trapezoid(r[m], E[m]))

rates = {}
rates['nat_el'] = lz.wd_rate(ham_s, MCHI, E_grid, halo=halo_avg)            # natural Xe elastic O4s
for A in (129, 131):
    rates[f'{A}_el'] = rate_iso(ham_s, A)
    rates[f'{A}_inel'] = rate_iso(ham_s, A, delta=LEVELS[A]['E'])
    rates[f'{A}_inel_jun'] = rate_iso(ham_s, A, delta=LEVELS[A]['E'], halo=halo_jun)
    rates[f'{A}_el_jun'] = rate_iso(ham_s, A, halo=halo_jun)
pd.DataFrame(dict(E_keV=E_grid, **rates)).to_csv(f'{OUT}/O4_spectra_unit_coupling.csv', index=False)

EFF_ROI = 0.96; EFF_WIN = 0.93   # plateau efficiency (paper), 200-270 keV efficiency (P009)
N_unit = {}
N_unit['nat_el_ROI'] = EFF_ROI * integ(E_grid, rates['nat_el'], 5.4, 269.9) * EXPO
N_unit['nat_el_200_270'] = EFF_WIN * integ(E_grid, rates['nat_el'], 200, 270) * EXPO
N_unit['nat_el_225_271'] = EFF_WIN * integ(E_grid, rates['nat_el'], 225, 271) * EXPO
for A in (129, 131):
    N_unit[f'{A}_el_total'] = integ(E_grid, rates[f'{A}_el'], 0, 420) * EXPO
    N_unit[f'{A}_el_ROI'] = EFF_ROI * integ(E_grid, rates[f'{A}_el'], 5.4, 269.9) * EXPO
    N_unit[f'{A}_el_200_270'] = EFF_WIN * integ(E_grid, rates[f'{A}_el'], 200, 270) * EXPO
    N_unit[f'{A}_inel_total'] = integ(E_grid, rates[f'{A}_inel'], 0, 420) * EXPO     # structure ratio R = 1
    N_unit[f'{A}_inel_total_jun'] = integ(E_grid, rates[f'{A}_inel_jun'], 0, 420) * EXPO
    N_unit[f'{A}_el_total_jun'] = integ(E_grid, rates[f'{A}_el_jun'], 0, 420) * EXPO
share = {A: N_unit[f'{A}_el_200_270'] / N_unit['nat_el_200_270'] for A in (129, 131)}
log(f"unit-coupling O4s counts in {EXPO} t yr: natural Xe ROI {N_unit['nat_el_ROI']:.4g}, 200-270 keV {N_unit['nat_el_200_270']:.4g}; "
    f"129Xe/131Xe share of 200-270 keV: {share[129]:.2f}/{share[131]:.2f}")
f_kin = {A: N_unit[f'{A}_inel_total'] / N_unit[f'{A}_el_total'] for A in (129, 131)}
f_kin_jun = {A: N_unit[f'{A}_inel_total_jun'] / N_unit[f'{A}_el_total_jun'] for A in (129, 131)}
for A in (129, 131):
    log(f"{A}Xe: elastic total {N_unit[f'{A}_el_total']:.4g}, ROI {N_unit[f'{A}_el_ROI']:.4g}, 200-270 {N_unit[f'{A}_el_200_270']:.4g}; "
        f"threshold-shifted (delta={LEVELS[A]['E']} keV, same structure factor) total {N_unit[f'{A}_inel_total']:.4g} -> kinematic factor f_kin = {f_kin[A]:.3f} (June {f_kin_jun[A]:.3f})")
R['unit_coupling_counts'] = N_unit; R['f_kin'] = f_kin; R['f_kin_june'] = f_kin_jun; R['isotope_share_200_270'] = share
N_lo_unit = EFF_ROI * integ(E_grid, rates['nat_el'], 5.4, 55) * EXPO
R['N_lo_per_window_event_O4s'] = N_lo_unit / N_unit['nat_el_200_270']
log(f"O4s low-energy companions (5.4-55 keV) per 200-270 keV event: {R['N_lo_per_window_event_O4s']:.1f} (P003: 28)")

# spectral moments
mom = {}
for key in ['129_el', '129_inel', '131_el', '131_inel']:
    r = rates[key]; tot = integ(E_grid, r, 0, 420)
    mean = float(np.trapezoid(E_grid * r, E_grid) / tot)
    cdf = np.cumsum(np.concatenate([[0], 0.5 * (r[1:] + r[:-1]) * np.diff(E_grid)])) / tot
    med = float(np.interp(0.5, cdf, E_grid)); pk = float(E_grid[np.argmax(r)])
    f200 = integ(E_grid, r, 200, 420) / tot
    mom[key] = dict(mean_keV=mean, median_keV=med, peak_keV=pk, frac_above_200=f200, E_max_grid=float(E_grid[r > 0].max()) if (r > 0).any() else 0.0)
    log(f"  {key}: mean {mean:.1f}, median {med:.1f}, peak {pk:.0f} keV, fraction above 200 keV {f200:.3f}, last non-zero grid E {mom[key]['E_max_grid']:.0f}")
R['spectral_moments'] = mom

# structure-factor ratio bracket (RECALLED, uncertain: Baudis et al. 2013)
R_BR = [0.01, 0.03, 0.1, 0.3, 1.0]
# (a) coupling giving one elastic O4s event in 200-270 keV
kappa_1ev = 1.0 / N_unit['nat_el_200_270']       # coupling^2 multiplier on unit coupling
tab = []
for Rsf in R_BR:
    row = dict(R_struct=Rsf)
    for A in (129, 131):
        row[f'N_hyb_{A}'] = kappa_1ev * Rsf * N_unit[f'{A}_inel_total']
    row['N_hyb_total'] = row['N_hyb_129'] + row['N_hyb_131']
    row['N_lo_elastic'] = kappa_1ev * N_lo_unit
    tab.append(row)
df_hyb = pd.DataFrame(tab); df_hyb.to_csv(f'{OUT}/hybrid_counts_one_event_normalisation.csv', index=False)
log(f'coupling^2 for 1 elastic O4s event in 200-270 keV: (c4 m_v^2)^2 = {kappa_1ev:.4g}; hybrid counts in {EXPO} t yr:')
for row in tab: log(f"   R={row['R_struct']:<5} N_hyb 129Xe={row['N_hyb_129']:.2f}  131Xe={row['N_hyb_131']:.2f}  total={row['N_hyb_total']:.2f}   (elastic low-E companions {row['N_lo_elastic']:.1f})")
R['one_event_normalisation'] = dict(kappa_c4mv2_sq=kappa_1ev, table=tab,
                                    sigma_n_equiv_cm2=None)
# equivalent SD-neutron cross-section of that isoscalar coupling: c_n = c_p = c^0_Anand = kappa^0.5 / m_v^2 ;  sigma_n = 3 c_n^2 mu_n^2 / (16 pi)
mu_n = lz.mu_red(MCHI, lz.M_NUCLEON_GEV)
c_n_1ev = math.sqrt(kappa_1ev) / lz.M_V_GEV ** 2
sig_n_1ev = 3 * c_n_1ev ** 2 * mu_n ** 2 / (16 * math.pi) * lz.GEV_TO_CM2
R['one_event_normalisation']['sigma_n_equiv_cm2'] = sig_n_1ev
log(f'   equivalent per-nucleon SD cross-section (c_p=c_n): sigma_n^SD = {sig_n_1ev:.3g} cm^2')

# (b) recalled LZ SD-neutron limit at 1 TeV (uncertain): sigma_n^SD ~ 2e-42 cm^2 at 40 GeV scaling ~m_chi -> ~4e-41 at 1 TeV
SIG_SD_LIMIT_1TEV = 4.0e-41
c_n_lim = math.sqrt(16 * math.pi * SIG_SD_LIMIT_1TEV / lz.GEV_TO_CM2 / (3 * mu_n ** 2))     # GeV^-2
ham_n = lz.wd_hamiltonian('O4_neutron_only', {4: (c_n_lim, -c_n_lim)})   # WimPyDD c0=cp+cn, c1=cp-cn with cp=0
rn = {}
for A in (129, 131):
    rn[f'{A}_el'] = rate_iso(ham_n, A); rn[f'{A}_inel'] = rate_iso(ham_n, A, delta=LEVELS[A]['E'])
rn['nat_el'] = lz.wd_rate(ham_n, MCHI, E_grid, halo=halo_avg)
lim = dict(sigma_n_SD_cm2=SIG_SD_LIMIT_1TEV, c_n_GeV2=c_n_lim,
           nat_el_ROI_per_tyr=EFF_ROI * integ(E_grid, rn['nat_el'], 5.4, 269.9),
           nat_el_200_270_per_tyr=EFF_WIN * integ(E_grid, rn['nat_el'], 200, 270))
for A in (129, 131):
    lim[f'{A}_inel_per_tyr_R1'] = integ(E_grid, rn[f'{A}_inel'], 0, 420)
    lim[f'{A}_el_per_tyr'] = integ(E_grid, rn[f'{A}_el'], 0, 420)
lim['hybrid_per_tyr_R0.1'] = 0.1 * (lim['129_inel_per_tyr_R1'] + lim['131_inel_per_tyr_R1'])
lim['hybrid_in_2.84tyr_R0.1'] = lim['hybrid_per_tyr_R0.1'] * EXPO
lim['ratio_to_unit_isoscalar_ROI'] = lim['nat_el_ROI_per_tyr'] * EXPO / N_unit['nat_el_ROI']
# Normalisation check at q -> 0: standard SD formula  sigma_A = sigma_n (mu_A/mu_n)^2 (4/3) (J+1)/J <S_n>^2,
# dR/dE(0) = n_T (rho/m_chi) sigma_A m_A/(2 mu_A^2) eta(0).  <S_n> recalled (Menendez, Gazit, Schwenk 2012, 1b): 129Xe 0.329 (J=1/2), 131Xe -0.272 (J=3/2)
SPIN = {129: dict(J=0.5, Sn=0.329), 131: dict(J=1.5, Sn=-0.272)}
chk = {}
for A in (129, 131):
    mA = lz.m_nucleus_gev(A); muA = lz.mu_red(MCHI, mA)
    lam2 = 4.0 / 3.0 * (SPIN[A]['J'] + 1) / SPIN[A]['J'] * SPIN[A]['Sn'] ** 2
    sigA = SIG_SD_LIMIT_1TEV * (muA / mu_n) ** 2 * lam2                       # cm^2
    nT = 1e6 / (A * 1.66054e-24) * lz.XE_ISOTOPES[A]                            # nuclei per tonne of natural Xe
    eta0 = lz.eta0(1.0)                                                          # (km/s)^-1 at v_min ~ 0, annual-average v_E
    r0 = nT * (lz.RHO0_GEV_CM3 / MCHI) * sigA * mA / (2 * muA ** 2) * eta0 * lz.C_KMS ** 2 * 1e5 * 1e-6 * 3.15576e7   # /t/yr/keV
    r_wd = float(rate_iso(ham_n, A)[0])                                          # WimPyDD at E = 1 keV
    chk[A] = dict(analytic_dRdE0=r0, wimpydd_dRdE_1keV=r_wd, ratio_wd_over_analytic=r_wd / r0, lambda2=lam2)
    log(f"  q->0 check {A}Xe: analytic SD dR/dE(0) = {r0:.4g} /t/yr/keV vs WimPyDD(1 keV, neutron-only O4) = {r_wd:.4g}; ratio {r_wd/r0:.2f} (<S_n> recalled)")
lim['q0_normalisation_check'] = chk
# Convention-independent check: O4/O1 rate ratio at q->0 for neutron-only equal couplings should be
# (3/16)(4/3)(J+1)/J <S_n>^2 / N^2 = (J+1)/(4J) <S_n>^2 / N^2  (free-neutron limit of the same formula gives sigma_4/sigma_1 = 3/16)
ham_1n = lz.wd_hamiltonian('O1_neutron_only', {1: (c_n_lim, -c_n_lim)})
for A in (129, 131):
    r1 = float(rate_iso(ham_1n, A)[0]); r4 = float(rate_iso(ham_n, A)[0]); Nn = A - 54
    expect = (SPIN[A]['J'] + 1) / (4 * SPIN[A]['J']) * SPIN[A]['Sn'] ** 2 / Nn ** 2
    chk[A].update(O4_over_O1_wimpydd=r4 / r1, O4_over_O1_expected_Menendez1b=expect, implied_Sn=abs(SPIN[A]['Sn']) * math.sqrt((r4 / r1) / expect))
    log(f"  O4/O1 (neutron-only, q->0) {A}Xe: WimPyDD {r4/r1:.3e} vs (J+1)/(4J) <S_n>^2/N^2 = {expect:.3e} with <S_n>={SPIN[A]['Sn']}; implied |<S_n>| in WimPyDD's density matrices = {chk[A]['implied_Sn']:.3f}")
R['sd_limit_normalisation'] = lim
log(f"recalled SD-n limit at 1 TeV sigma_n={SIG_SD_LIMIT_1TEV:.1e} cm^2 -> c_n={c_n_lim:.3e} GeV^-2: elastic O4 ROI rate {lim['nat_el_ROI_per_tyr']:.3f} /t/yr "
    f"(200-270 keV {lim['nat_el_200_270_per_tyr']:.4f}); excitation rate (R=1) 129Xe {lim['129_inel_per_tyr_R1']:.3f}, 131Xe {lim['131_inel_per_tyr_R1']:.3f} /t/yr; "
    f"hybrids at R=0.1: {lim['hybrid_per_tyr_R0.1']:.3f} /t/yr = {lim['hybrid_in_2.84tyr_R0.1']:.2f} in {EXPO} t yr")
# inelastic/elastic rate ratio (whole spectrum) = R x f_kin, weighted over the two isotopes
w = {A: N_unit[f'{A}_el_total'] for A in (129, 131)}
f_kin_w = (w[129] * f_kin[129] + w[131] * f_kin[131]) / (w[129] + w[131])
R['inel_over_el_rate_ratio'] = {f'R={Rsf}': Rsf * f_kin_w for Rsf in R_BR}
R['f_kin_weighted'] = f_kin_w
log(f'excitation/elastic SD rate ratio in natural Xe = R x f_kin with f_kin(weighted) = {f_kin_w:.3f}; ' + ', '.join(f'R={k}: {v:.3g}' for k, v in R['inel_over_el_rate_ratio'].items()))

# ==========================================================================================
# 3. Kinematics with the excitation threshold
# ==========================================================================================
log('\n=== 3. Kinematics ===')
vE_jun = lz.v_earth_kms(167); vmax_jun = lz.vmax_kms(vE_jun)
kin = dict(v_E_16June=vE_jun, v_max_16June=vmax_jun)
for A in (129, 131):
    Es = LEVELS[A]['E']
    for m in (100.0, 400.0, 1000.0, 4000.0):
        lo, hi = lz.E_R_range_keV(m, vmax_jun, A=A, delta_kev=Es)
        lo0, hi0 = lz.E_R_range_keV(m, vmax_jun, A=A, delta_kev=0.0)
        kin[f'{A}_m{int(m)}_E_R_window_keV'] = (lo, hi); kin[f'{A}_m{int(m)}_E_R_max_elastic_keV'] = hi0
    kin[f'{A}_vmin_248_el'] = lz.vmin_kms(248, MCHI, A=A); kin[f'{A}_vmin_248_inel'] = lz.vmin_kms(248, MCHI, A=A, delta_kev=Es)
    E_m = res1[A]['E_NR_matching_S1c_keV']
    kin[f'{A}_vmin_ENRmatch_inel'] = lz.vmin_kms(E_m, MCHI, A=A, delta_kev=Es) if not math.isnan(E_m) else None
    kin[f'{A}_m_chi_min_248_inel_GeV'] = lz.m_chi_min_gev(248, A=A, v_kms=vmax_jun, delta_kev=Es)
    kin[f'{A}_m_chi_min_ENRmatch_inel_GeV'] = lz.m_chi_min_gev(E_m, A=A, v_kms=vmax_jun, delta_kev=Es) if not math.isnan(E_m) else None
    # excitation possible at all iff mu v_max^2/2 >= E*  ->  mu_min = 2 E*/v_max^2, m_chi,min = mu m_N/(m_N - mu)
    mN_A = lz.m_nucleus_gev(A); mu_min = 2 * Es * 1e-6 / (vmax_jun / lz.C_KMS) ** 2
    kin[f'{A}_m_chi_min_threshold_only_GeV'] = mu_min * mN_A / (mN_A - mu_min)
    kin[f'{A}_E_R_at_threshold_keV'] = Es * mu_min / mN_A   # recoil energy at the kinematic threshold (E_R = E* mu/m_N)
    log(f"{A}Xe E*={Es} keV: v_min(248 keV, 1 TeV) elastic {kin[f'{A}_vmin_248_el']:.0f} -> with threshold {kin[f'{A}_vmin_248_inel']:.0f} km/s; "
        f"E_R window at v_max={vmax_jun:.0f} km/s (1 TeV): {kin[f'{A}_m1000_E_R_window_keV'][0]:.1f}-{kin[f'{A}_m1000_E_R_window_keV'][1]:.1f} keV (elastic max {kin[f'{A}_m1000_E_R_max_elastic_keV']:.1f}); "
        f"m_chi,min(248 keV + E*) = {kin[f'{A}_m_chi_min_248_inel_GeV']:.0f} GeV; m_chi,min for any excitation {kin[f'{A}_m_chi_min_threshold_only_GeV']:.1f} GeV; "
        f"m_chi,min(E_NR={E_m:.0f} + E*) = {kin[f'{A}_m_chi_min_ENRmatch_inel_GeV']:.0f} GeV")
R['kinematics'] = kin

# ==========================================================================================
# 4. Discussion numbers (containment etc.; recalled attenuation data)
# ==========================================================================================
log('\n=== 4. Containment / pulse-shape numbers ===')
RHO_LXE = 2.9
ATT = {39.58: 20.0, 80.19: 3.5, 29.7: 8.0}    # mu/rho [cm^2/g] in Xe (recalled, uncertain x1.5): 39.6 keV above K edge 34.56 keV
cont = {}
for E, mr in ATT.items():
    mfp_mm = 10.0 / (mr * RHO_LXE)
    cont[f'{E}_keV'] = dict(mu_over_rho_cm2_g=mr, mfp_mm=mfp_mm, P_travel_5mm=math.exp(-5.0 / mfp_mm))
    log(f'  gamma/x-ray {E} keV: mu/rho={mr} cm^2/g -> mean free path {mfp_mm:.2f} mm, P(>5 mm) = {math.exp(-5.0/mfp_mm):.1e}')
cont['IC_electron_range_um'] = dict(value=30, note='CSDA range of a 40 keV electron in LXe, order 10-30 um (recalled, uncertain)')
for A in (129, 131):
    cont[f'{A}_ER_photon_fraction_at_S1c540'] = res1[A]['ER_photon_fraction_at_540']
    cont[f'{A}_IC_fraction'] = LEVELS[A]['alpha_IC'] / (1 + LEVELS[A]['alpha_IC'])
    log(f"  {A}Xe hybrid at S1c=540: {100*res1[A]['ER_photon_fraction_at_540']:.0f}% of the S1 photons are ER-type; IC fraction alpha/(1+alpha) = {cont[f'{A}_IC_fraction']:.2f}; T1/2 = {LEVELS[A]['T12_ns']} ns")
R['containment'] = cont; R['levels'] = LEVELS

# ==========================================================================================
# Figures (Okabe-Ito palette, fixed order)
# ==========================================================================================
OI = dict(orange='#E69F00', sky='#56B4E9', green='#009E73', yellow='#F0E442', blue='#0072B2', verm='#D55E00', purple='#CC79A7', black='#000000')
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})

# Fig 1: loci over the bands
fig, ax = plt.subplots(figsize=(6.4, 4.6))
z = 1.2816
ax.fill_between(S1_NR, np.log10(S2_NR) - z * SIG_NR, np.log10(S2_NR) + z * SIG_NR, color=OI['verm'], alpha=0.18, lw=0)
ax.plot(S1_NR, np.log10(S2_NR), color=OI['verm'], lw=2, label='NR band median (NEST-LZ), 10–90%')
ax.fill_between(S1_ER, np.log10(S2_ER) - z * SIG_ER, np.log10(S2_ER) + z * SIG_ER, color=OI['blue'], alpha=0.15, lw=0)
ax.plot(S1_ER, np.log10(S2_ER), color=OI['blue'], lw=2, label='ER band median (NEST-LZ), 10–90% approx.')
for A, c in ((129, OI['green']), (131, OI['purple'])):
    ax.plot(HYB[A]['S1c'], HYB[A]['log10S2c'], color=c, lw=2, label=f'NR + {LEVELS[A]["E"]} keV ({A}Xe*) hybrid, E_NR = 10–320 keV')
    # markers at E_NR = 50, 150, 248
    for E in (50, 150, 248):
        s1, l = hybrid_point(E, LEVELS[A]['E'])
        ax.plot(s1, l, 'o', ms=5, color=c, mec='white', mew=1)
        ax.annotate(f'{E}', (s1, l), textcoords='offset points', xytext=(4, 4), fontsize=7, color='#444')
ax.plot(EV_S1, EV_LOGS2, marker='*', ms=14, color=OI['black'], ls='none', label='LZ event (540.1 phd, 3.967)')
ax.axvline(600, color='#888', ls=':', lw=1); ax.axhline(4.15, color='#888', ls=':', lw=1)
ax.text(606, 4.86, 'ROI edge S1c = 600', fontsize=7, color='#666', rotation=90, va='top')
ax.text(160, 4.16, 'ROI top log10 S2c = 4.15', fontsize=7, color='#666')
ax.set_xlim(150, 1000); ax.set_ylim(3.6, 4.95)
ax.set_xlabel('S1c [phd]'); ax.set_ylabel('log10 S2c [phd]')
ax.set_title('Nuclear-excitation hybrids lie above the ROI, between the bands', fontsize=10)
ax.grid(color='#eee', lw=0.6); ax.legend(fontsize=7, loc='lower right', frameon=False)
fig.tight_layout(); fig.savefig(f'{FIG}/P036_fig1_hybrid_loci.png', dpi=160); plt.close(fig)

# Fig 2: recoil spectra elastic vs threshold-shifted (unit coupling)
fig, ax = plt.subplots(figsize=(6.4, 4.0))
for A, c in ((129, OI['green']), (131, OI['purple'])):
    ax.plot(E_grid, rates[f'{A}_el'], color=c, lw=2, label=f'{A}Xe elastic O4')
    ax.plot(E_grid, rates[f'{A}_inel'], color=c, lw=2, ls='--', label=f'{A}Xe with threshold E* = {LEVELS[A]["E"]} keV (R = 1)')
ax.axvspan(200, 270, color='#ddd', alpha=0.5, lw=0); ax.text(235, ax.get_ylim()[1] if False else 1, '', fontsize=7)
ax.set_yscale('log'); ax.set_xlim(0, 420)
ymax = max(rates['129_el'].max(), rates['131_el'].max())
ax.set_ylim(ymax * 1e-6, ymax * 3)
ax.set_xlabel('nuclear recoil energy E_R [keV]'); ax.set_ylabel('dR/dE_R [events / (t yr keV)], unit coupling')
ax.set_title('1 TeV, O4 isoscalar unit coupling, Baxter SHM (annual mean)', fontsize=10)
ax.grid(color='#eee', lw=0.6); ax.legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(f'{FIG}/P036_fig2_spectra.png', dpi=160); plt.close(fig)

# Fig 3: two-equation residual vs E_ER
fig, ax = plt.subplots(figsize=(5.6, 3.6))
ax.plot(df_scan.E_ER_keV, df_scan.residual_sigma, '-o', color=OI['blue'], lw=2, ms=5)
ax.axhline(0, color='#888', lw=1); ax.axvline(39.58, color=OI['green'], ls=':', lw=1); ax.axvline(80.19, color=OI['purple'], ls=':', lw=1)
ax.text(40.5, ax.get_ylim()[0] + 1, '129Xe*', fontsize=7, color=OI['green']); ax.text(81, ax.get_ylim()[0] + 1, '131Xe*', fontsize=7, color=OI['purple'])
ax.set_xlabel('prompt ER energy E_ER added at the vertex [keV]'); ax.set_ylabel('(log10 S2c predicted − observed) / σ_NR')
ax.set_title('At fixed S1c = 540.1 phd, any ER admixture raises S2c', fontsize=10)
ax.grid(color='#eee', lw=0.6)
fig.tight_layout(); fig.savefig(f'{FIG}/P036_fig3_two_equation.png', dpi=160); plt.close(fig)

R['runtime_s'] = time.time() - T0
with open(f'{OUT}/P036_results.json', 'w') as f:
    json.dump(R, f, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else (o.tolist() if isinstance(o, np.ndarray) else str(o)))
log(f'\ndone in {R["runtime_s"]:.1f} s; outputs in {OUT}')
LOG.close()
