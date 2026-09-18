"""
P012_magnetic_dipole.py -- Magnetic-dipole dark matter at 1 TeV: the coupling from the LZ fit and the low-energy shoulder.

Run from the simulation root:   .venv/bin/python output/code/P012_magnetic_dipole.py

Parts
  A  efficiency model (P003 form; P007 sigma_hi variant; hard cuts) and window integrals
  B  digitisation of LZ Fig. 6 (bottom): two-sided 90% interval on d10^s vs WIMP mass, from the PDF vector paths
  C  L10 spectra with WimPyDD for m = 100, 200, 400, 1000, 4000 GeV at d10 = 1 (Anand reduction, m_M = m_N,
     WimPyDD c0 = 2 x Anand), expected LZ counts per d10^2, d10 for N = 0.3 / 1.0 / 2.4 (Table I) and 0.105 / 3.65
     (single-count PLR band); comparison with the digitised Fig. 1 L10^s curve (shape, absolute scale -> the
     residual factor of P003) and with the Fig. 6 interval (implied N at the edges under each normalisation)
  D  low-energy shoulder: N_lo per high-energy event, absolute 5.4-55 keV counts in the 2024 exposure (4.2 t yr),
     tolerance scan (3, 5, 10 events); 55-200 keV intermediate counts
  E  photon-mediated magnetic-dipole DM (long-range, 1/q^2) in NREFT with WimPyDD: spectrum, N_lo, mu_chi for 1 event
  F  magnetic-moment conversions and the contact vs long-range comparison
Outputs -> output/work/P012/*.csv, *.json, figures/*.png
"""
from __future__ import annotations
import sys, os, json, math, time
sys.path.insert(0, 'output/code')
import numpy as np
import pandas as pd
from scipy import special, integrate, optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import lzcommon as lz

OUT = 'output/work/P012'
FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)
T0 = time.time()

MV2 = lz.M_V_GEV ** 2                 # m_v^2 (GeV^2), m_v = 246.2 GeV
MN = lz.M_NUCLEON_GEV                 # nucleon mass 0.938272 GeV; we set m_M = m_N (stated in the paper)
EXPOSURE = lz.LZ['exposure_tyr']      # 2.84 t yr (paper)
EXPOSURE_2024 = 4.2                   # t yr, LZ 2024 WS combined SR1+WS2024 (recalled, certain: 5.5 t x 280 live days)
MASSES = [100.0, 200.0, 400.0, 1000.0, 4000.0]
ALPHA_EM = 1.0 / 137.036              # recalled, certain
E_CHARGE = math.sqrt(4 * math.pi * ALPHA_EM)   # 0.3028 (Heaviside-Lorentz natural units)
G_P, G_N = 5.5857, -3.8261            # nucleon g-factors (recalled, certain)
MU_NUC_GEV = E_CHARGE / (2 * MN)      # nuclear magneton in GeV^-1 (natural units)
MU_NUC_ECM = lz.HBARC_GEV_FM / (2 * MN) * 1e-13   # nuclear magneton in e cm  (hbar c / 2 m_p c^2 = 0.1052 fm)
N_BESTFIT, N_UP68, N_LO68 = 1.0, 2.4, 0.3       # Table I: 1.0 +1.4 -0.7 events
N_PLR_LO, N_PLR_HI = 0.105, 3.65                # asymptotic two-sided 90% PLR interval for n=1, b~0 (mu - 1 - ln mu = 1.353)

# ----------------------------------------------------------------------------------------------
# Part A: efficiency model and window integrals (P003 form)
# ----------------------------------------------------------------------------------------------
def efficiency(E, sig_lo=3.4, sig_hi=8.0, eff0=lz.LZ['eff_plateau'], hard=False):
    E = np.asarray(E, dtype=float)
    lo, hi = lz.LZ['E_50pct_low_keV'], lz.LZ['E_50pct_high_keV']
    if hard:
        return eff0 * ((E >= lo) & (E <= hi))
    return eff0 * 0.5 * (1 + special.erf((E - lo) / (math.sqrt(2) * sig_lo))) * 0.5 * (1 - special.erf((E - hi) / (math.sqrt(2) * sig_hi)))

EFF_VARIANTS = {'P003': dict(), 'sig_hi_11p5': dict(sig_hi=11.5), 'hard': dict(hard=True)}

def window_rate(Egrid, dRdE, e1, e2, **effkw):
    Ef = np.linspace(e1, e2, 2001)
    y = np.asarray(dRdE, float)
    if np.all(y > 0):
        yf = 10 ** np.interp(Ef, Egrid, np.log10(y))
    else:
        yf = np.interp(Ef, Egrid, y)
    return float(integrate.trapezoid(yf * efficiency(Ef, **effkw), Ef))

def summarise(Egrid, dRdE, **effkw):
    R_lo = window_rate(Egrid, dRdE, 5.4, 55.0, **effkw)        # 2024 ROI (hard cut at 55 keV as in P003)
    R_mid = window_rate(Egrid, dRdE, 55.0, 200.0, **effkw)
    R_hi = window_rate(Egrid, dRdE, 200.0, 270.0, **effkw)
    R_roi = window_rate(Egrid, dRdE, 1.0, 330.0, **effkw)      # whole efficiency-weighted ROI
    return dict(R_lo=R_lo, R_mid=R_mid, R_hi=R_hi, R_roi=R_roi, N_lo_per_hi=R_lo / R_hi if R_hi > 0 else np.inf,
                N_mid_per_hi=R_mid / R_hi if R_hi > 0 else np.inf, f_lo=R_lo / R_roi, f_hi=R_hi / R_roi)

# ----------------------------------------------------------------------------------------------
# Part B: digitise Fig. 6 bottom (d10^s two-sided interval vs mass)
# ----------------------------------------------------------------------------------------------
import pymupdf
FIG6 = 'inputs/arXiv_2609.02823_source/Fig6_O1_L10_limit_stacked.pdf'
page = pymupdf.open(FIG6)[0]
drawings = page.get_drawings()
# decade tick marks on the left spine of the bottom panel (length-4 pt horizontal lines): 10^2, 10^1, 10^0, 10^-1
tick_y = sorted(d['rect'].y0 for d in drawings if d['rect'].y0 > 330 and d['rect'].width < 8 and d['rect'].height < 0.5 and d['rect'].x0 < 80)
decade_ticks = [355.46, 448.04, 540.61, 633.18]     # identified from the tick list (equal spacing 92.57 pt); see details.md
for t in decade_ticks:
    assert min(abs(t - y) for y in tick_y) < 0.05, 'decade tick not found'
slope_y, icpt_y = np.polyfit(decade_ticks, [2, 1, 0, -1], 1)     # log10(d10) = slope*y + icpt
def vertices(idx):
    pts = []
    for it in drawings[idx]['items']:
        if it[0] == 'l':
            pts += [(it[1].x, it[1].y), (it[2].x, it[2].y)]
    return sorted(set((round(x, 2), round(y, 2)) for x, y in pts))
v_up, v_lo, v_med = vertices(349), vertices(350), vertices(346)      # width-3 black (upper, lower), width-1.5 dashed (median)
mass_grid = np.array([10, 12, 14, 17, 21, 30, 40, 50, 100, 200, 400, 1000, 4000], float)
x_up = np.array([p[0] for p in v_up])
# x calibration: the 13 vertices of the upper curve are the 13 masses of Table S6 -> fit log10(m) vs x
slope_x, icpt_x = np.polyfit(x_up, np.log10(mass_grid), 1)
x_resid = np.log10(mass_grid) - (slope_x * x_up + icpt_x)
frame = drawings[237]['rect']       # bottom axes frame (white rectangle)
def to_mass(x): return 10 ** (slope_x * x + icpt_x)
def to_d(y): return 10 ** (slope_y * y + icpt_y)
dig_rows = []
for m, (xu, yu) in zip(mass_grid, v_up):
    med = [p for p in v_med if abs(p[0] - xu) < 0.5]
    lo = [p for p in v_lo if abs(p[0] - xu) < 0.5 and p[1] < 690]      # drop the vertical drop to the frame at 100 GeV
    dig_rows.append(dict(m_GeV=m, x_pt=xu, mass_from_x=to_mass(xu), d10_upper=to_d(yu),
                         d10_median_sens=to_d(med[0][1]) if med else np.nan, d10_lower=to_d(lo[0][1]) if lo else np.nan))
dig = pd.DataFrame(dig_rows)
dig.to_csv(os.path.join(OUT, 'fig6_bottom_digitised.csv'), index=False, float_format='%.5g')
fig6_cal = dict(decade_ticks_y=decade_ticks, pt_per_decade_y=float(-1 / slope_y), pt_per_decade_x=float(1 / slope_x),
                x_calibration_rms_log10=float(np.sqrt(np.mean(x_resid ** 2))), frame=[frame.x0, frame.y0, frame.x1, frame.y1],
                frame_mass_range=[to_mass(frame.x0), to_mass(frame.x1)], frame_d10_range=[to_d(frame.y1), to_d(frame.y0)],
                lower_curve_starts_at_GeV=to_mass(v_lo[0][0]))
print('Part B: Fig. 6 bottom digitised; x rms %.4f dex; frame masses %.1f-%.0f GeV; d10 range %.3f-%.1f'
      % (fig6_cal['x_calibration_rms_log10'], *fig6_cal['frame_mass_range'], *fig6_cal['frame_d10_range']))
print(dig[dig.m_GeV >= 100].to_string(index=False))

# ----------------------------------------------------------------------------------------------
# Part C: L10 spectra and couplings
# ----------------------------------------------------------------------------------------------
WD = lz.wd()
halo = lz.wd_halo()                                      # Sun-frame SHM (LZ/WimPyDD default usage; validated vs Fig. 1 by P003)
E = np.union1d(np.arange(1.0, 330.0 + 1e-9, 2.0), [5.4, 55.0, 200.0, 248.0, 269.9, 300.0])

# L10 = (d10/m_v^2) (chibar i sigma^{mu nu} q_nu / m_M chi)(Nbar i sigma_{mu rho} q^rho / m_M N)
#     -> (d10/m_v^2) * 4 (m_N^2/m_M^2) [ (q^2/m_N^2) O4 - O6 ]      (Anand et al. Table 1 entry 10, recalled/likely; P003 R6)
# with m_M = m_N:  c4^Anand(q) = 4 A q^2/m_N^2,  c6^Anand = -4 A,  A = d10/m_v^2  (isoscalar, c^0 = (c_p+c_n)/2)
# WimPyDD isospin convention c^0 = c_p + c_n  ->  multiply by 2 (lz.wd_c_from_anand)
A_UNIT = 1.0 / MV2
c4_wd = lz.wd_c_from_anand(4 * A_UNIT)[0]      # 8/m_v^2 ; times q^2/m_N^2
c6_wd = lz.wd_c_from_anand(-4 * A_UNIT)[0]     # -8/m_v^2
h_L10 = WD.eft_hamiltonian('P012_L10_dipole_dipole', {(4, 'q2'): lambda q, A=c4_wd: [A * q ** 2 / MN ** 2, 0.0],
                                                      6: lambda A6=c6_wd: [A6, 0.0]})
spec_L10 = {}
for m in MASSES:
    spec_L10[m] = lz.wd_rate(h_L10, m, E, halo)
    print(f'  L10 spectrum m={m:.0f} done ({time.time()-T0:.0f}s)', flush=True)
np.savez(os.path.join(OUT, 'P012_L10_spectra_d10_1.npz'), E=E, **{f'm{int(m)}': v for m, v in spec_L10.items()})

# comparison with the digitised Fig. 1 L10^s curves (P003, vector digitisation; unit coupling d10^s = 1/m_v^2)
fig1 = pd.read_csv('output/work/P003/fig1_digitised_bottom.csv')
def extrema(y, Egrid, mask):
    yy = np.where(mask, y, np.nan)
    i1 = np.nanargmax(np.where(Egrid < 120, yy, np.nan)); i2 = np.nanargmax(np.where(Egrid > 120, yy, np.nan))
    seg = (Egrid > 20) & (Egrid < 200) & mask
    idip = np.nanargmin(np.where(seg, yy, np.nan))
    return float(Egrid[i1]), float(Egrid[idip]), float(Egrid[i2])
cmp_rows, scale_fig = [], {}
for m, col in ((200.0, 'L10s_200GeV'), (1000.0, 'L10s_1000GeV')):
    fc = np.interp(E, fig1.E_keV, fig1[col], left=np.nan, right=np.nan)
    ok = np.isfinite(fc) & (E >= 10) & (E <= 260) & (fc > 0)
    sc = float(np.exp(np.nanmean(np.log(fc[ok] / spec_L10[m][ok]))))
    resid = np.log10(fc[ok] / (sc * spec_L10[m][ok]))
    pf, pw = extrema(fc, E, ok), extrema(spec_L10[m], E, ok)
    sf, sw = summarise(E, np.nan_to_num(fc)), summarise(E, spec_L10[m])
    scale_fig[m] = sc
    cmp_rows.append(dict(m_GeV=m, scale_fig1_over_wimpydd=sc, rms_log10_resid=float(np.sqrt(np.mean(resid ** 2))),
                         max_abs_log10_resid=float(np.max(np.abs(resid))), fig_peak1=pf[0], fig_dip=pf[1], fig_peak2=pf[2],
                         wd_peak1=pw[0], wd_dip=pw[1], wd_peak2=pw[2], N_lo_fig1=sf['N_lo_per_hi'], N_lo_wd=sw['N_lo_per_hi'],
                         N_unit_fig1_events=sf['R_roi'] * EXPOSURE, N_unit_wd_events=sw['R_roi'] * EXPOSURE,
                         R_hi_fig1=sf['R_hi'], R_hi_wd=sw['R_hi'], f_lo_fig1=sf['f_lo'], f_lo_wd=sw['f_lo']))
cmp = pd.DataFrame(cmp_rows)
cmp.to_csv(os.path.join(OUT, 'P012_L10_vs_fig1.csv'), index=False, float_format='%.5g')
SCALE = scale_fig[1000.0]        # LZ Fig. 1 / our normalisation at 1000 GeV (the residual factor; ~3.9)
print('Part C: Fig.1/WimPyDD(Anand x 2) scale = %.3f (200 GeV), %.3f (1000 GeV); rms resid %.3f dex' % (scale_fig[200.0], SCALE, cmp.rms_log10_resid.iloc[1]))

# expected counts per d10^2 and couplings, for each efficiency variant and both normalisations
rows = []
for m in MASSES:
    for ename, ekw in EFF_VARIANTS.items():
        s = summarise(E, spec_L10[m], **ekw)
        N_unit_wd = s['R_roi'] * EXPOSURE                  # events in 2.84 t yr at d10 = 1, our normalisation
        N_unit_lz = N_unit_wd * SCALE                      # LZ Fig. 1 normalisation (factor SCALE, from 1000 GeV)
        d = dig[dig.m_GeV == m].iloc[0]
        row = dict(m_GeV=m, eff_variant=ename, N_unit_wd=N_unit_wd, N_unit_lz=N_unit_lz, **{k: s[k] for k in ('R_lo', 'R_mid', 'R_hi', 'R_roi', 'N_lo_per_hi', 'N_mid_per_hi', 'f_lo', 'f_hi')})
        for tag, N in (('bestfit', N_BESTFIT), ('lo68', N_LO68), ('up68', N_UP68), ('plr_lo', N_PLR_LO), ('plr_hi', N_PLR_HI)):
            row[f'd10_lz_{tag}'] = math.sqrt(N / N_unit_lz)
            row[f'd10_wd_{tag}'] = math.sqrt(N / N_unit_wd)
        row.update(fig6_lower=d.d10_lower, fig6_upper=d.d10_upper, fig6_median=d.d10_median_sens,
                   N_at_fig6_lower_lz=N_unit_lz * d.d10_lower ** 2, N_at_fig6_upper_lz=N_unit_lz * d.d10_upper ** 2,
                   N_at_fig6_lower_wd=N_unit_wd * d.d10_lower ** 2, N_at_fig6_upper_wd=N_unit_wd * d.d10_upper ** 2,
                   N_at_fig6_median_lz=N_unit_lz * d.d10_median_sens ** 2)
        rows.append(row)
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, 'P012_coupling_table.csv'), index=False, float_format='%.5g')
main = res[res.eff_variant == 'P003']
print(main[['m_GeV', 'N_unit_wd', 'N_unit_lz', 'd10_lz_bestfit', 'd10_lz_lo68', 'd10_lz_up68', 'd10_wd_bestfit', 'fig6_lower', 'fig6_upper',
            'N_at_fig6_lower_lz', 'N_at_fig6_upper_lz', 'N_at_fig6_lower_wd', 'N_at_fig6_upper_wd']].to_string(index=False))

# ----------------------------------------------------------------------------------------------
# Part D: low-energy shoulder in the 2024 exposure
# ----------------------------------------------------------------------------------------------
sh_rows = []
for m in MASSES:
    for ename in EFF_VARIANTS:
        r = res[(res.m_GeV == m) & (res.eff_variant == ename)].iloc[0]
        # fraction of the fitted ROI count that lies in 5.4-55 keV, scaled to the 2024 exposure
        per_roi_event = r.f_lo * EXPOSURE_2024 / EXPOSURE
        row = dict(m_GeV=m, eff_variant=ename, f_lo=r.f_lo, N_lo_per_hi=r.N_lo_per_hi, N_mid_per_hi=r.N_mid_per_hi,
                   N2024_lo_per_ROI_event=per_roi_event,
                   N2024_lo_bestfit=per_roi_event * N_BESTFIT, N2024_lo_up68=per_roi_event * N_UP68,
                   N2024_lo_plr_hi=per_roi_event * N_PLR_HI, N2024_lo_fig6_upper=per_roi_event * r.N_at_fig6_upper_lz)
        for tol in (3, 5, 10):
            row[f'N_ROI_max_for_tol{tol}'] = tol / per_roi_event          # LZ ROI events allowed before the 2024 tolerance is hit
            row[f'd10_lz_max_for_tol{tol}'] = math.sqrt(tol / per_roi_event / r.N_unit_lz)
        sh_rows.append(row)
sh = pd.DataFrame(sh_rows)
sh.to_csv(os.path.join(OUT, 'P012_low_energy_shoulder.csv'), index=False, float_format='%.5g')
print('Part D (P003 efficiency):')
print(sh[sh.eff_variant == 'P003'][['m_GeV', 'f_lo', 'N_lo_per_hi', 'N_mid_per_hi', 'N2024_lo_bestfit', 'N2024_lo_up68', 'N2024_lo_plr_hi',
                                    'N_ROI_max_for_tol3', 'd10_lz_max_for_tol3', 'd10_lz_max_for_tol10']].to_string(index=False))
# digitised Fig. 1 shape as a cross-check for the 1000 GeV shoulder
fc = np.nan_to_num(np.interp(E, fig1.E_keV, fig1.L10s_1000GeV, left=np.nan, right=np.nan))
sf = summarise(E, fc)
shoulder_fig1 = dict(f_lo=sf['f_lo'], N_lo_per_hi=sf['N_lo_per_hi'], N_mid_per_hi=sf['N_mid_per_hi'],
                     N2024_lo_bestfit=sf['f_lo'] * EXPOSURE_2024 / EXPOSURE, N_unit_events=sf['R_roi'] * EXPOSURE)

# ----------------------------------------------------------------------------------------------
# Part E: photon-mediated magnetic-dipole DM (long range)
# ----------------------------------------------------------------------------------------------
# L = (mu_chi/2) chibar sigma^{mu nu} chi F_{mu nu}; photon exchange with the nucleon charge and magnetic-moment currents.
# NR reduction (Anand normalisation, per nucleon; recalled/likely -- derived in details.md and consistent with
# Fitzpatrick et al. 2012 / Gresham & Zurek 2014 / Banks-Fox-Weiner 2010):
#   c1^N = e mu_chi Q_N /(2 m_chi)         c5^N = 2 e mu_chi m_N Q_N / q^2
#   c4^N = e mu_chi g_N / m_N               c6^N = - e mu_chi g_N m_N / q^2
# (c4 + q^2/m_N^2 c6 = 0: the photon dipole-dipole term is L10's transverse-spin structure with 1/m_M^2 -> 1/q^2.)
# Isospin (Anand c^0 = (c_p+c_n)/2, c^1 = (c_p-c_n)/2 -> WimPyDD c^0 = c_p + c_n, c^1 = c_p - c_n):
MU_REF = MU_NUC_GEV           # compute at mu_chi = 1 nuclear magneton; rates scale as mu_chi^2
def h_photon(name, charge=True, spin=True):
    wc = {}
    if charge:
        wc[1] = lambda mchi, mu=MU_REF: [E_CHARGE * mu / (2 * mchi), E_CHARGE * mu / (2 * mchi)]          # c_p only: (cp+cn, cp-cn) = (cp, cp)
        wc[5] = lambda q, mu=MU_REF: [2 * E_CHARGE * mu * MN / q ** 2, 2 * E_CHARGE * mu * MN / q ** 2]
    if spin:
        wc[4] = lambda mu=MU_REF: [E_CHARGE * mu * (G_P + G_N) / MN, E_CHARGE * mu * (G_P - G_N) / MN]
        wc[6] = lambda q, mu=MU_REF: [-E_CHARGE * mu * (G_P + G_N) * MN / q ** 2, -E_CHARGE * mu * (G_P - G_N) * MN / q ** 2]
    return WD.eft_hamiltonian(name, wc)
h_ph = h_photon('P012_photon_dipole_full')
h_ph_charge = h_photon('P012_photon_dipole_charge', spin=False)
h_ph_spin = h_photon('P012_photon_dipole_spin', charge=False)
spec_ph, ph_rows = {}, []
for m in MASSES:
    spec_ph[m] = lz.wd_rate(h_ph, m, E, halo)
    s = summarise(E, spec_ph[m])
    N_unit = s['R_roi'] * EXPOSURE                    # events at mu_chi = 1 mu_nuc
    mu_1ev = math.sqrt(1.0 / N_unit)                  # mu_chi (in mu_nuc) giving 1 ROI event
    mu_1hi = math.sqrt(1.0 / (s['R_hi'] * EXPOSURE))  # mu_chi giving 1 event in 200-270 keV
    ph_rows.append(dict(m_GeV=m, N_unit_events_at_1muN=N_unit, mu_for_1_ROI_event_muN=mu_1ev, mu_for_1_ROI_event_ecm=mu_1ev * MU_NUC_ECM,
                        mu_for_1_hi_event_muN=mu_1hi, mu_for_1_hi_event_ecm=mu_1hi * MU_NUC_ECM,
                        N_lo_per_hi=s['N_lo_per_hi'], N_mid_per_hi=s['N_mid_per_hi'], f_lo=s['f_lo'], f_hi=s['f_hi'],
                        N2024_lo_if_1_hi_event=s['N_lo_per_hi'] * EXPOSURE_2024 / EXPOSURE,
                        dRdE_10keV=spec_ph[m][np.searchsorted(E, 10.0)], dRdE_248keV=spec_ph[m][np.searchsorted(E, 248.0)]))
    print(f'  photon dipole m={m:.0f}: N_lo/hi = {s["N_lo_per_hi"]:.3g}, mu(1 ROI ev) = {mu_1ev:.3g} mu_N ({time.time()-T0:.0f}s)', flush=True)
ph = pd.DataFrame(ph_rows)
ph.to_csv(os.path.join(OUT, 'P012_photon_dipole.csv'), index=False, float_format='%.5g')
# decomposition at 1000 GeV
spec_ph_charge = lz.wd_rate(h_ph_charge, 1000.0, E, halo)
spec_ph_spin = lz.wd_rate(h_ph_spin, 1000.0, E, halo)
decomp = {}
for lab, sp in (('full', spec_ph[1000.0]), ('charge_dipole_O1_O5', spec_ph_charge), ('dipole_dipole_O4_O6', spec_ph_spin)):
    s = summarise(E, sp)
    decomp[lab] = dict(R_lo=s['R_lo'], R_hi=s['R_hi'], N_lo_per_hi=s['N_lo_per_hi'], dRdE_10=float(sp[np.searchsorted(E, 10.0)]),
                       dRdE_248=float(sp[np.searchsorted(E, 248.0)]))
decomp['spin_fraction_at_248keV'] = decomp['dipole_dipole_O4_O6']['dRdE_248'] / decomp['full']['dRdE_248']
decomp['spin_fraction_of_R_hi'] = decomp['dipole_dipole_O4_O6']['R_hi'] / decomp['full']['R_hi']
decomp['sum_check_R_hi'] = (decomp['charge_dipole_O1_O5']['R_hi'] + decomp['dipole_dipole_O4_O6']['R_hi']) / decomp['full']['R_hi']
np.savez(os.path.join(OUT, 'P012_photon_spectra_1muN.npz'), E=E, **{f'm{int(m)}': v for m, v in spec_ph.items()},
         charge_1000=spec_ph_charge, spin_1000=spec_ph_spin)
print('Part E decomposition (1000 GeV):', json.dumps(decomp, indent=None, default=float)[:600])

# ----------------------------------------------------------------------------------------------
# Part F: magnetic-moment conversions for L10 and the contact vs long-range matching
# ----------------------------------------------------------------------------------------------
d10_best = float(main[main.m_GeV == 1000].d10_lz_bestfit.iloc[0])
d10_best_wd = float(main[main.m_GeV == 1000].d10_wd_bestfit.iloc[0])
q2_event = 2 * lz.m_nucleus_gev(lz.A_XE_MEAN) * 248e-6            # GeV^2 at E_R = 248 keV
G_dd = d10_best / (MV2 * MN ** 2)                                  # dipole-dipole contact strength d10/(m_v^2 m_M^2), GeV^-4
g0 = 0.5 * (G_P + G_N)
def mu_eff_photon_equiv(d10, q2):
    """mu_chi of a photon-mediated dipole whose dipole-dipole amplitude equals L10's at momentum transfer q (isoscalar g^0)."""
    return 4 * d10 * q2 / (E_CHARGE * g0 * MN * MV2)
conv = dict(
    e_natural=E_CHARGE, mu_nuc_GeV_inv=MU_NUC_GEV, mu_nuc_e_cm=MU_NUC_ECM, g_p=G_P, g_n=G_N, g0_isoscalar=g0,
    d10_bestfit_lz_norm_1000GeV=d10_best, d10_bestfit_wd_norm_1000GeV=d10_best_wd,
    A_anand_bestfit_GeV_m2=d10_best / MV2, G_dd_GeV_m4=G_dd, G_dd_scale_GeV=(1 / G_dd) ** 0.25,
    q_event_GeV=math.sqrt(q2_event), q2_event_GeV2=q2_event,
    mu_eff_photon_equiv_at_q_event_GeV_inv=mu_eff_photon_equiv(d10_best, q2_event),
    mu_eff_photon_equiv_at_q_event_muN=mu_eff_photon_equiv(d10_best, q2_event) / MU_NUC_GEV,
    mu_eff_photon_equiv_at_q_event_ecm=mu_eff_photon_equiv(d10_best, q2_event) / MU_NUC_GEV * MU_NUC_ECM,
    # heavy-mediator reading: mu_chi^V mu_N^V / M^2 = d10 / (m_v^2 m_M^2) (up to O(1) factors from the 1/2's)
    heavy_mediator_muchi_over_muN_for_M_1TeV_and_muN_1muN=G_dd * 1000.0 ** 2 / MU_NUC_GEV ** 2,
    photon_dipole_mu_for_1_ROI_event_1000GeV_muN=float(ph[ph.m_GeV == 1000].mu_for_1_ROI_event_muN.iloc[0]),
    photon_dipole_mu_for_1_ROI_event_1000GeV_ecm=float(ph[ph.m_GeV == 1000].mu_for_1_ROI_event_ecm.iloc[0]),
    photon_dipole_N_lo_per_hi_1000GeV=float(ph[ph.m_GeV == 1000].N_lo_per_hi.iloc[0]),
)
print('Part F:', json.dumps(conv, indent=1, default=float))

# ----------------------------------------------------------------------------------------------
# Part G: what a background-free single count implies for a two-sided 90% PLR interval (reference for Part C)
#   q(mu; n) = 2 [mu - n + n ln(n/mu)];  asymptotic threshold 2.706;  exact p-value from the Poisson distribution of n
# ----------------------------------------------------------------------------------------------
from scipy import stats
def q_stat(mu, n):
    n = np.asarray(n, float)
    return 2 * (mu - n + np.where(n > 0, n * np.log(np.maximum(n, 1e-300) / mu), 0.0))
def p_exact(mu, n_obs):
    ns = np.arange(0, 60)
    return float(np.sum(stats.poisson.pmf(ns, mu)[q_stat(mu, ns) >= q_stat(mu, n_obs) - 1e-12]))
def edges(n_obs, exact=True):
    f = (lambda mu: p_exact(mu, n_obs) - 0.10) if exact else (lambda mu: q_stat(mu, n_obs) - 2.706)
    lo = optimize.brentq(f, 1e-4, max(n_obs, 1e-3)) if n_obs > 0 else 0.0
    hi = optimize.brentq(f, max(n_obs, 1e-3) + 1e-6, 30.0)
    return lo, hi
plr = {}
for label, ex in (('exact_poisson', True), ('asymptotic', False)):
    lo1, hi1 = edges(1, ex)
    plr[label] = dict(n1_lower=lo1, n1_upper=hi1, n0_upper_expected=edges(0, ex)[1])
# implied single-count expectations vs the Fig. 6 interval (1000 GeV, LZ normalisation)
r1000 = main[main.m_GeV == 1000].iloc[0]
plr['fig6_1000GeV_implied_N'] = dict(lower_lz=r1000.N_at_fig6_lower_lz, upper_lz=r1000.N_at_fig6_upper_lz, median_lz=r1000.N_at_fig6_median_lz,
                                     lower_wd=r1000.N_at_fig6_lower_wd, upper_wd=r1000.N_at_fig6_upper_wd,
                                     median_wd=r1000.N_at_fig6_median_lz / SCALE)
print('Part G:', json.dumps(plr, indent=None, default=float))

# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------
# 1. L10 spectra normalised to 1 LZ event (LZ normalisation), overlay of digitised Fig. 1 at 1000 GeV
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for m, c in zip(MASSES, ['C0', 'C1', 'C2', 'C3', 'C4']):
    Nu = float(main[main.m_GeV == m].N_unit_lz.iloc[0])
    ax.plot(E, spec_L10[m] * SCALE * EXPOSURE / Nu, color=c, lw=1.4, label=f'{m:.0f} GeV (d$_{{10}}$ = {math.sqrt(1/Nu):.2f})')
Nu = float(main[main.m_GeV == 1000].N_unit_lz.iloc[0])
ax.plot(fig1.E_keV, fig1.L10s_1000GeV * EXPOSURE / Nu, color='k', lw=3, alpha=0.25, label='LZ Fig. 1 L$_{10}^s$ 1000 GeV (digitised), same scale')
ax.axvspan(0, 5.4, color='0.6', alpha=0.3); ax.axvspan(269.9, 330, color='0.6', alpha=0.3)
ax.axvspan(5.4, 55, color='C3', alpha=0.07); ax.axvspan(200, 270, color='C2', alpha=0.10)
ax.axvline(248, color='k', ls=':', lw=0.8)
ax.set_yscale('log'); ax.set_ylim(1e-5, 3e-2); ax.set_xlim(0, 330)
ax.set_xlabel('$E_R$ [keV]'); ax.set_ylabel('events / keV in 2.84 t yr (normalised to 1.0 ROI event)')
ax.set_title('P012: L$_{10}$ (dipole-dipole contact) spectra at the LZ best fit'); ax.legend(fontsize=7.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P012_L10_spectra_bestfit.png'), dpi=140); plt.close(fig)

# 2. d10 vs mass: our N = 1 (both normalisations), 68% band, and the digitised Fig. 6 interval
fig, ax = plt.subplots(figsize=(7.2, 4.8))
mm = main.m_GeV.values
ax.fill_between(mm, main.d10_lz_lo68, main.d10_lz_up68, color='C1', alpha=0.25, label='this work, N = 0.3-2.4 (Table I 68%), LZ normalisation')
ax.plot(mm, main.d10_lz_bestfit, 'o-', color='C1', label='this work, N = 1.0, LZ (Fig. 1) normalisation')
ax.plot(mm, main.d10_wd_bestfit, 's--', color='C0', label='this work, N = 1.0, Anand-reduction x WimPyDD normalisation')
ax.plot(mm, main.d10_lz_plr_lo, ':', color='C1', lw=1); ax.plot(mm, main.d10_lz_plr_hi, ':', color='C1', lw=1, label='N = 0.105 / 3.65 (single-count PLR 90%)')
dd = dig[dig.m_GeV >= 10]
ax.plot(dd.m_GeV, dd.d10_upper, 'k-', lw=2.5, label='LZ Fig. 6 two-sided 90% interval (digitised)')
ax.plot(dd.m_GeV, dd.d10_lower, 'k-', lw=2.5)
ax.plot(dd.m_GeV, dd.d10_median_sens, 'k--', lw=1, label='LZ median sensitivity')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(9, 4000); ax.set_ylim(0.08, 20)
ax.set_xlabel('WIMP mass [GeV]'); ax.set_ylabel('d$_{10}^s$ (dimensionless)'); ax.legend(fontsize=7)
ax.set_title('P012: coupling for one LZ event vs the LZ interval')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P012_d10_vs_mass.png'), dpi=140); plt.close(fig)

# 3. photon-mediated dipole vs contact L10 at 1000 GeV, each normalised to 1 event in 200-270 keV
fig, ax = plt.subplots(figsize=(7.2, 4.6))
s10 = summarise(E, spec_L10[1000.0]); sph = summarise(E, spec_ph[1000.0])
ax.plot(E, spec_L10[1000.0] / s10['R_hi'], color='C3', lw=1.6, label=f"L$_{{10}}$ contact dipole-dipole (N$_{{lo}}$/hi = {s10['N_lo_per_hi']:.2f})")
ax.plot(E, spec_ph[1000.0] / sph['R_hi'], color='C0', lw=1.6, label=f"photon-mediated magnetic dipole (N$_{{lo}}$/hi = {sph['N_lo_per_hi']:.0f})")
ax.plot(E, spec_ph_charge / sph['R_hi'], color='C0', lw=1, ls='--', label='  charge-dipole part (Z$^2$/E$_R$, O$_1$+O$_5$)')
ax.plot(E, spec_ph_spin / sph['R_hi'], color='C0', lw=1, ls=':', label='  dipole-dipole part (O$_4$-O$_6$)/q$^2$')
ax.axvspan(5.4, 55, color='C3', alpha=0.07); ax.axvspan(200, 270, color='C2', alpha=0.10)
ax.axvspan(0, 5.4, color='0.6', alpha=0.3); ax.axvspan(269.9, 330, color='0.6', alpha=0.3)
ax.set_yscale('log'); ax.set_xlim(0, 330); ax.set_ylim(1e-4, 1e4)
ax.set_xlabel('$E_R$ [keV]'); ax.set_ylabel('dR/dE per event in 200-270 keV [keV$^{-1}$]')
ax.set_title('P012: contact vs long-range dipole, 1000 GeV'); ax.legend(fontsize=7.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P012_contact_vs_photon_dipole.png'), dpi=140); plt.close(fig)

# ----------------------------------------------------------------------------------------------
# summary JSON
# ----------------------------------------------------------------------------------------------
summary = dict(
    settings=dict(masses=MASSES, exposure_tyr=EXPOSURE, exposure_2024_tyr=EXPOSURE_2024, halo='WimPyDD Sun-frame SHM Baxter-2021 (lz.wd_halo)',
                  vmax_kms=float(halo[0][-1]), E_grid='1-330 keV, 2 keV steps + reference energies', efficiency='P003 erf model (0.96; 5.4/269.9 keV; sig 3.4/8 keV); variants sig_hi=11.5, hard',
                  L10_reduction='(d10/m_v^2) 4 (m_N^2/m_M^2)[(q^2/m_N^2) O4 - O6], m_M = m_N; WimPyDD c0 = 2 x Anand (c4 = 8 q^2/(m_v^2 m_N^2), c6 = -8/m_v^2)'),
    fig6_calibration=fig6_cal, fig6_digitised=dig.to_dict(orient='records'),
    fig1_comparison=cmp.to_dict(orient='records'), scale_fig1_over_ours_1000GeV=SCALE,
    couplings_P003_eff=main.to_dict(orient='records'), shoulder_P003_eff=sh[sh.eff_variant == 'P003'].to_dict(orient='records'),
    shoulder_fig1_shape_1000GeV=shoulder_fig1, photon_dipole=ph.to_dict(orient='records'), photon_decomposition_1000GeV=decomp,
    conversions=conv, single_count_plr_reference=plr, runtime_s=time.time() - T0)
with open(os.path.join(OUT, 'P012_summary.json'), 'w') as f:
    json.dump(summary, f, indent=1, default=float)
print(f'done in {time.time()-T0:.0f} s')
