"""
P079 -- Optimal ROI and fiducial design for the next LZ high-energy analysis:
        trading fiducial mass against MSSI with a 3D likelihood.

Run from the simulation root:  .venv/bin/python output/code/P079_roi_fv_optimisation.py

What is computed
 1. Geometry.  The 4.7 t FV is P070's vector-digitised mean radial contour r_c(z) of Fig. 3
    (reconstructed coordinates), z = 9.0-132.8 cm (text), true wall R = 72.8 cm, LXe 2.86 g/cm^3.
    Three FV families parametrised by the minimum true-wall stand-off s and the bottom cut z_b:
      A "shift"    : r(z; s) = r_c(z) - (s - 8)        (keeps LZ's bottom flare; primary)
      B "cylinder" : r(z; s) = R - s                    (drops the flare; optimistic)
      C "reco-margin": r(z; m) = r_wall_reco(z) - m      (constant margin to the reconstructed wall)
    Masses are reported raw (P070: 4.36 t at LZ's cuts) and calibrated by kappa_M = 4.71/4.36.
 2. Backgrounds in the NR band (+-2 sigma) of the 200 keV - E50(edge) window, per 220 live days at
    the given FV:
      wall MSSI  N_w = 0.0048 * f_nb * I_w(FV)/I_w(FV_LZ),  I_w = int_FV exp(-(R - r)/lambda) dV
      RFR MSSI   N_r = 1e-4   * f_nb * J(FV)/J(FV_LZ),      J   = int_FV exp(-z/lambda_z) dV
      wall leakage (surface events, upper bound): 1e-4 -> 1e-2 between the 4.7 t and 5.4 t contours
      uniform   (accidentals + atm-nu + residual ER) 2e-4 (bracket 4e-4) x M/M_LZ, + 1.03e-4 per 100 keV
      S1c-edge dependence of MSSI from P038's table (x 0.15 NR-band share).
    lambda and lambda_z are checked against LZ's own 5.4 t annulus counts (0.03 wall, 0.003 RFR).
 3. Signal: L10 (1 TeV) and inelastic O1 (delta = 350, 366 keV) spectra from P050's WimPyDD cache,
    P038 edge efficiencies (E50, erf sigma), sigma_E = 11 sqrt(E/248) keV, normalised to 1.0 event per
    220 live days at LZ's cuts, scaled by mass and by the acceptance ratio A(edge)/A(600).
 4. Metrics: Asimov discovery Z in LZ's untouched 524 live days (P020), live time to 5 sigma,
    exclusion reach (M x A), M/sqrt(b + b0); optimum over (s, z_b, edge) with reconstruction floors.
 5. Soft FV: position-binned (d, z) Asimov likelihood versus hard cuts; equivalent hard-cut mass.
 6. Robustness: MSSI normalisation k = 0.5-2 (LZ's 100 % uncertainty), lambda = 3-6 cm,
    lambda_z = 2-6.5 cm, uniform-background bracket, P(>= 1 background event) per configuration.
Outputs: output/work/P079/*.csv, P079_results.json, figures/*.png
"""
from __future__ import annotations
import json, math, os, sys, time
import numpy as np
import pandas as pd
from scipy import special, optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, 'output/code')
from common import lzcommon as lz

OUT = 'output/work/P079'; FIG = f'{OUT}/figures'
os.makedirs(FIG, exist_ok=True)
LOG = open(f'{OUT}/run_log.txt', 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.write(s + '\n'); LOG.flush()

RES = {}
T0 = time.time()

# ----------------------------------------------------------------------------------------------
# 0. Inputs and constants
# ----------------------------------------------------------------------------------------------
R_TRUE = 72.8            # cm, TPC radius (recalled, likely; P033/P070; Fig. 3 frame ends at 73.0^2)
H_ACT = 145.6            # cm, gate-cathode (P070 confirmed from Fig. 3)
RHO = 2.86               # g/cm^3 LXe (recalled, likely)
Z_T = 132.8              # cm, upper FV boundary (12.8 cm below gate; paper text)
Z_B_LZ, S_LZ = 9.0, 8.0  # cm, LZ's bottom cut and minimum true-wall stand-off
LIVE_LZ = 220.0          # live days of the published search
LIVE_NEW = 884.0 * 220.0 / 371.0   # P020: untouched exposure, 524 live days to 2 Sep 2026
M_LZ_T = 4.71
F_NB = 0.035             # P004: fraction of WS-ROI MSSI within +-2 sigma of the NR median at S1c > 500 phd
N_WALL_LZ, N_RFR_LZ = 0.0048, 0.0001     # LZ MSSI table, science sample, 4.7 t, WS ROI
N_WALL_54, N_RFR_54 = 0.03, 0.003        # same, 5.4 t annulus (disjoint from the 4.7 t FV)
NB_SHARE_EXT = 0.15      # P038: 0.0027 of 0.018 added MSSI (600 -> 1000 phd) lies in NR +-2 sigma
B_UNI_LZ = {'central': 2.0e-4, 'high': 4.0e-4}   # uniform NR-band background per 220 d at 4.71 t (P022 1.5e-4 acc, P019 3.4e-5 nu, rest ER; P016 b_H - MSSI = 4.0e-4)
B_UNI_PER_KEV = 1.03e-6  # P050: 1.03e-4 per 100 keV of window extension above 270 keV (accidentals + nu)
LAM_W = {'central': 4.3, 'short': 3.3, 'long': 6.0, 'lz_table': 1.2}   # P033 MC 4.3; P070 prompt-veto data 3.3; long 6; LZ-table-implied (fitted below, overwritten)
LAM_Z = {'central': 2.7, 'short': 2.0, 'long': 6.5}          # RFR z-scale: 214Pb 352 keV E2 (P070), LZ-table fit (below), detector-gamma (P070)
D_RECO_FLOOR = 3.0       # cm: minimum margin to the reconstructed wall (10 sigma at sigma_r = 3 mm for S2c ~ 1e4 phd, P041)
SIG_E = lambda E: 11.0 * np.sqrt(np.asarray(E, float) / 248.0)   # P009/P021 resolution model
EDGES = {600: (271.64, 10.93), 700: (310.45, 11.72), 800: (348.67, 12.65), 1000: (423.24, 14.55), 1200: (495.12, 17.96)}  # P038 (E50 keV, erf sigma keV)
E_LOW_WINDOW = 200.0     # keV, lower edge of the high-energy window (observed energy)

cont = pd.read_csv('output/work/P070/fv_contours.csv')
zc = cont.z.values; r_c = cont.r_mean47.values; r_reco = cont.r_wall_reco.values
r54 = 0.5 * (cont.r_min54.values + cont.r_max54.values)
Z_B54, Z_T54 = 2.15, 135.75   # P070: drawn z-range of the 5.4 t contours
say(f'contour: {len(zc)} z-samples, r_c range {r_c.min():.2f}-{r_c.max():.2f} cm, min true stand-off {R_TRUE - r_c.max():.2f} cm')

mssi_edge = pd.read_csv('output/work/P038/P038_mssi_vs_edge.csv').set_index('S1c_edge')
def added_mssi(edge):
    """P038: MSSI (total, WS-ROI-like S2c) added by extending the S1c edge; mid of the exp/pow models. Returns (wall, RFR)."""
    row = mssi_edge.loc[float(edge)]
    return 0.5 * (row.wall_exp + row.wall_pow), 0.5 * (row.RFR_exp + row.RFR_pow)

# ----------------------------------------------------------------------------------------------
# 1. Geometry: FV families, masses, analytic radial integrals
# ----------------------------------------------------------------------------------------------
Z_GRID = np.arange(0.0, H_ACT + 1e-9, 0.1)
R_C = np.interp(Z_GRID, zc, r_c); R_RECO = np.interp(Z_GRID, zc, r_reco); R_54 = np.interp(Z_GRID, zc, r54)

def r_of_z(family, s):
    if family == 'A': return np.clip(R_C - (s - S_LZ), 0, R_TRUE)
    if family == 'B': return np.full_like(Z_GRID, R_TRUE - s)
    if family == 'C': return np.clip(R_RECO - s, 0, R_TRUE)       # here s = margin to the reconstructed wall
    if family == '54': return R_54
    raise ValueError(family)

def zmask(z_b, z_t=Z_T):
    return (Z_GRID >= z_b - 1e-9) & (Z_GRID <= z_t + 1e-9)

def volume_cm3(rz, z_b, z_t=Z_T):
    m = zmask(z_b, z_t); return float(np.trapezoid(np.pi * rz[m] ** 2, Z_GRID[m]))

def I_wall(rz, z_b, lam, z_t=Z_T):
    """int_FV exp(-(R-r)/lam) 2 pi r dr dz, radial part analytic."""
    m = zmask(z_b, z_t); r = rz[m]
    rad = 2 * np.pi * ((lam * r - lam ** 2) * np.exp(-(R_TRUE - r) / lam) + lam ** 2 * np.exp(-R_TRUE / lam))
    return float(np.trapezoid(rad, Z_GRID[m]))

def J_rfr(rz, z_b, lam_z, z_t=Z_T):
    m = zmask(z_b, z_t); return float(np.trapezoid(np.pi * rz[m] ** 2 * np.exp(-Z_GRID[m] / lam_z), Z_GRID[m]))

def mass_t(rz, z_b, z_t=Z_T): return volume_cm3(rz, z_b, z_t) * RHO / 1e6

M_RAW_LZ = mass_t(r_of_z('A', S_LZ), Z_B_LZ)
KAPPA_M = M_LZ_T / M_RAW_LZ
M_RAW_54 = mass_t(r_of_z('54', 0), Z_B54, Z_T54)
M_ACTIVE = mass_t(np.full_like(Z_GRID, R_TRUE), 0.0, H_ACT)
say(f'\n[1] geometry: raw mass at LZ cuts {M_RAW_LZ:.3f} t (P070 4.36; paper 4.71) -> kappa_M = {KAPPA_M:.3f}; '
    f'5.4 t contour raw {M_RAW_54:.3f} t (x kappa = {M_RAW_54*KAPPA_M:.2f}); active {M_ACTIVE:.2f} t (LZ 7.0)')
say(f'    stand-off of the mean contour from the true wall: min {R_TRUE - R_C.max():.2f}, mean {np.mean(R_TRUE - R_C[zmask(Z_B_LZ)]):.2f}, '
    f'max {R_TRUE - R_C[zmask(Z_B_LZ)].min():.2f} cm; from reco wall mean {np.mean(R_RECO[zmask(Z_B_LZ)] - R_C[zmask(Z_B_LZ)]):.2f} cm')
say(f'    reco-wall offset from the true wall: top {R_TRUE - R_RECO[-1]:.2f}, mid {R_TRUE - np.interp(72.8, Z_GRID, R_RECO):.2f}, bottom {R_TRUE - R_RECO[0]:.2f} cm')
RES['geometry'] = dict(M_raw_LZ_t=M_RAW_LZ, kappa_M=KAPPA_M, M_raw_54_t=M_RAW_54, M_active_t=M_ACTIVE,
                       standoff_true_min=float(R_TRUE - R_C.max()), standoff_true_mean=float(np.mean(R_TRUE - R_C[zmask(Z_B_LZ)])),
                       reco_offset_top_mid_bottom=[float(R_TRUE - R_RECO[-1]), float(R_TRUE - np.interp(72.8, Z_GRID, R_RECO)), float(R_TRUE - R_RECO[0])])

# mass vs s and z_b (family A), dM/ds
for fam in 'AB':
    say(f'    family {fam}: M(s) at z_b=9 [t, calibrated]: ' + ', '.join(f's={s:g}: {KAPPA_M*mass_t(r_of_z(fam, s), Z_B_LZ):.2f}' for s in [2, 3, 4, 5, 6, 8, 10, 12]))
say('    family A: M(z_b) at s=8: ' + ', '.join(f'z_b={zb:g}: {KAPPA_M*mass_t(r_of_z("A", S_LZ), zb):.2f}' for zb in [0, 2, 5, 7, 9, 12]))
say('    family C (reco margin m): ' + ', '.join(f'm={m:g}: {KAPPA_M*mass_t(r_of_z("C", m), Z_B_LZ):.2f}' for m in [1, 2, 3, 4, 6]))

# ----------------------------------------------------------------------------------------------
# 2. Backgrounds: calibrate lambda's against the 5.4 t annulus
# ----------------------------------------------------------------------------------------------
RZ_LZ = r_of_z('A', S_LZ); RZ_54 = r_of_z('54', 0)
def annulus_ratio_wall(lam):
    """(wall MSSI in 5.4 t volume minus 4.7 t FV) / (wall MSSI in 4.7 t FV) for exp(-(R-r)/lam)."""
    return (I_wall(RZ_54, Z_B54, lam, Z_T54) - I_wall(RZ_LZ, Z_B_LZ, lam)) / I_wall(RZ_LZ, Z_B_LZ, lam)
def annulus_ratio_rfr(lam_z):
    return (J_rfr(RZ_54, Z_B54, lam_z, Z_T54) - J_rfr(RZ_LZ, Z_B_LZ, lam_z)) / J_rfr(RZ_LZ, Z_B_LZ, lam_z)

say('\n[2] MSSI depth scales versus LZ\'s 5.4 t annulus (wall 0.03/0.0048 = 6.25; RFR 0.003/0.0001 = 30):')
for lam in [3.0, 4.3, 4.8, 6.0, 8.0]:
    say(f'    wall lambda = {lam:.1f} cm: annulus/FV = {annulus_ratio_wall(lam):.2f}')
lam_w_fit = optimize.brentq(lambda l: annulus_ratio_wall(l) - N_WALL_54 / N_WALL_LZ, 1.0, 60.0)
say(f'    -> wall lambda reproducing 6.25: {lam_w_fit:.2f} cm')
# radial-only version (same z range as the 4.7 t FV) to separate the radial from the bottom-slab contribution
def annulus_ratio_wall_radial(lam):
    return (I_wall(RZ_54, Z_B_LZ, lam) - I_wall(RZ_LZ, Z_B_LZ, lam)) / I_wall(RZ_LZ, Z_B_LZ, lam)
lam_w_fit_rad = optimize.brentq(lambda l: annulus_ratio_wall_radial(l) - N_WALL_54 / N_WALL_LZ, 0.5, 60.0)
say(f'    -> radial-only annulus (z 9-132.8): lambda = {lam_w_fit_rad:.2f} cm; with lambda = 4.3 the radial annulus alone gives {annulus_ratio_wall_radial(4.3):.2f}')
for lz_ in [2.0, 2.7, 4.0, 6.5]:
    say(f'    RFR lambda_z = {lz_:.1f} cm: annulus/FV = {annulus_ratio_rfr(lz_):.1f}')
lam_z_fit = optimize.brentq(lambda l: annulus_ratio_rfr(l) - N_RFR_54 / N_RFR_LZ, 0.5, 30.0)
say(f'    -> lambda_z reproducing 30: {lam_z_fit:.2f} cm (P070: 214Pb E2 2.7 cm, detector-gamma 6.5 cm)')
LAM_Z['fit'] = lam_z_fit; LAM_W['fit'] = lam_w_fit; LAM_W['lz_table'] = lam_w_fit
# the HE SB (800-1700 phd) wall rows give an independent ratio: 0.5 (5.4 t annulus) / 0.1 (4.7 t) = 5
lam_w_fit_hesb = optimize.brentq(lambda l: annulus_ratio_wall(l) - 0.5 / 0.1, 1.0, 60.0)
say(f'    -> HE SB wall rows (0.5/0.1 = 5.0) give lambda = {lam_w_fit_hesb:.2f} cm; prompt-veto rows (2.2/0.09 = 24) would give '
    f'{optimize.brentq(lambda l: annulus_ratio_wall(l) - 2.2/0.09, 0.3, 60.0):.2f} cm (prompt-veto wall MSSI is dominated by the annulus)')
RES['lambda_calibration'] = dict(lambda_wall_fit_cm=lam_w_fit, lambda_wall_fit_radial_only_cm=lam_w_fit_rad, lambda_wall_fit_HESB_cm=lam_w_fit_hesb,
                                 lambda_z_fit_cm=lam_z_fit, annulus_ratio_wall_at_4p3=annulus_ratio_wall(4.3),
                                 annulus_ratio_wall_radial_at_4p3=annulus_ratio_wall_radial(4.3), annulus_ratio_rfr_at_2p7=annulus_ratio_rfr(2.7),
                                 annulus_ratio_wall_at_3p3=annulus_ratio_wall(3.3))

# wall leakage (surface events reconstructed inside the contour): the two contour criteria (1e-4 in the 4.7 t FV,
# 1e-2 in the 5.4 t volume, both over the whole 3-600 phd ROI) fix an empirical scale in the reconstructed stand-off.
d_reco_47 = float(np.mean(R_RECO[zmask(Z_B_LZ)] - RZ_LZ[zmask(Z_B_LZ)]))
d_reco_54 = float(np.mean(R_RECO[zmask(Z_B_LZ)] - RZ_54[zmask(Z_B_LZ)]))
LAM_LEAK = (d_reco_47 - d_reco_54) / math.log(100.0)
say(f'    wall leakage (whole 3-600 phd ROI): mean reco stand-off 4.7 t {d_reco_47:.2f} cm, 5.4 t {d_reco_54:.2f} cm -> lambda_leak = {LAM_LEAK:.2f} cm (1e-4 -> 1e-2).')
say(f'    This population is low-S2 (sigma_r ~ cm); at S2c ~ 1e4 phd sigma_r ~ 3 mm (P041), so for the high-energy window the leakage is')
say(f'    treated as a reconstruction floor d_reco >= {D_RECO_FLOOR} cm everywhere, not as a background term; the low-E exponential is kept as a diagnostic.')
RES['leakage'] = dict(d_reco_47=d_reco_47, d_reco_54=d_reco_54, lambda_leak_cm=LAM_LEAK, d_reco_floor_cm=D_RECO_FLOOR)

def bkg_components(family, s, z_b, edge, lam=None, lam_z=None, k_mssi=1.0, uni='central', live=LIVE_LZ):
    """NR-band (+-2 sigma) background counts in 200 keV - E50(edge) for `live` live days. Returns dict."""
    lam = LAM_W['central'] if lam is None else lam; lam_z = LAM_Z['central'] if lam_z is None else lam_z
    rz = r_of_z(family, s)
    M = KAPPA_M * mass_t(rz, z_b)
    add_w, add_r = added_mssi(edge)
    wall_ref = (N_WALL_LZ * F_NB + add_w * NB_SHARE_EXT) * k_mssi
    rfr_ref = (N_RFR_LZ * F_NB + add_r * NB_SHARE_EXT) * k_mssi
    fw = I_wall(rz, z_b, lam) / I_wall(RZ_LZ, Z_B_LZ, lam)
    fr = J_rfr(rz, z_b, lam_z) / J_rfr(RZ_LZ, Z_B_LZ, lam_z)
    E50 = EDGES[edge][0]
    b_uni = (B_UNI_LZ[uni] + B_UNI_PER_KEV * max(E50 - 270.0, 0.0)) * M / M_LZ_T
    m = zmask(z_b); dr = R_RECO[m] - rz[m]
    d_reco = float(np.mean(dr)); d_reco_min = float(dr.min())
    leak_lowE = 1e-4 * math.exp(-(d_reco - d_reco_47) / LAM_LEAK)      # diagnostic only (whole-ROI, low-S2 population)
    sc = live / LIVE_LZ
    return dict(M=M, wall=wall_ref * fw * sc, rfr=rfr_ref * fr * sc, uni=b_uni * sc, leak_lowE=leak_lowE * sc,
                total=(wall_ref * fw + rfr_ref * fr + b_uni) * sc, fw=fw, fr=fr, d_reco=d_reco, d_reco_min=d_reco_min,
                feasible=d_reco_min >= D_RECO_FLOOR - 1e-9)

b0 = bkg_components('A', S_LZ, Z_B_LZ, 600)
say(f'    LZ cuts, 600 phd, per 220 d: wall {b0["wall"]:.2e}, RFR {b0["rfr"]:.2e}, uniform {b0["uni"]:.2e}, total {b0["total"]:.2e} '
    f'(low-E leakage diagnostic {b0["leak_lowE"]:.1e}; d_reco mean {b0["d_reco"]:.2f}, min {b0["d_reco_min"]:.2f} cm)')
b1 = bkg_components('A', S_LZ, Z_B_LZ, 1000)
say(f'    LZ cuts, 1000 phd: wall {b1["wall"]:.2e}, RFR {b1["rfr"]:.2e}, uniform {b1["uni"]:.2e}, total {b1["total"]:.2e} (P038 added NR-band MSSI 0.0027)')

# ----------------------------------------------------------------------------------------------
# 3. Signal spectra and acceptance versus edge
# ----------------------------------------------------------------------------------------------
cache = np.load('output/work/P050/P050_spectra_cache.npz')
SPEC = {'L10': (cache['E_l10'], cache['l10'])}
for d, row in zip(cache['deltas'], cache['inel']):
    if int(round(d)) in (300, 350, 366): SPEC[f'inel{int(round(d))}'] = (cache['E_in'], row)

def eff_edge(E, edge):
    E50, sig = EDGES[edge]
    lo = 0.5 * (1 + special.erf((E - 5.4) / (math.sqrt(2) * 2.5)))
    return 0.955 * lo * 0.5 * (1 - special.erf((E - E50) / (math.sqrt(2) * sig)))

E_OBS = np.arange(0.0, 1500.0, 1.0)
def observed_spectrum(E, dRdE, edge):
    w = dRdE * eff_edge(E, edge); dE = np.gradient(E)
    s = SIG_E(np.maximum(E, 1.0))
    K = np.exp(-0.5 * ((E_OBS[:, None] - E[None, :]) / s[None, :]) ** 2) / (math.sqrt(2 * math.pi) * s[None, :])
    return K @ (w * dE)                       # per keV of observed energy

ACC = {}
say('\n[3] acceptance of the 200 keV-plus window (fraction of the full spectrum, incl. 0.955 plateau):')
for m, (E, dR) in SPEC.items():
    tot = float(np.trapezoid(dR, E)); ACC[m] = {}
    for edge in EDGES:
        obs = observed_spectrum(E, dR, edge)
        ACC[m][edge] = float(np.trapezoid(obs[E_OBS >= E_LOW_WINDOW], E_OBS[E_OBS >= E_LOW_WINDOW])) / tot
    say(f'    {m:8s}: ' + ', '.join(f'{e}: {ACC[m][e]:.3f}' for e in EDGES) + f'  | ratio 1000/600 = {ACC[m][1000]/ACC[m][600]:.2f}, 800/600 = {ACC[m][800]/ACC[m][600]:.2f}')
RES['acceptance'] = {m: {str(e): v for e, v in d.items()} for m, d in ACC.items()}
RES['acceptance_ratio_to_600'] = {m: {str(e): ACC[m][e] / ACC[m][600] for e in EDGES} for m in ACC}

def signal(model, M, edge, live=LIVE_LZ, rate=1.0):
    """Expected signal counts: `rate` events per 220 live days at LZ's cuts in the LZ window, scaled."""
    return rate * (M / M_LZ_T) * (ACC[model][edge] / ACC[model][600]) * (live / LIVE_LZ)

# ----------------------------------------------------------------------------------------------
# 4. Metrics and the (s, z_b, edge) scan
# ----------------------------------------------------------------------------------------------
def Z_asimov(s, b):
    s = np.asarray(s, float); b = np.asarray(b, float)
    return np.sqrt(np.maximum(2 * ((s + b) * np.log1p(s / b) - s), 0.0))

from scipy import stats
P5 = 1 - stats.norm.cdf(5.0)
def N_for_5sigma(b):
    """smallest observed count N with P(>= N | b) <= 2.87e-7 (exact Poisson)."""
    N = 1
    while stats.poisson.sf(N - 1, b) > P5: N += 1
    return N
def discovery_stats(s, b):
    """exact-Poisson few-event metrics: P(5 sigma) = P(N >= N_5sigma(b) | s + b); median Z of Z_N = Phi^-1(1 - P(>= N | b))."""
    N5 = N_for_5sigma(b)
    p5 = float(stats.poisson.sf(N5 - 1, s + b))
    Ns = np.arange(0, 60); pN = stats.poisson.pmf(Ns, s + b)
    ZN = np.where(Ns > 0, stats.norm.isf(np.clip(stats.poisson.sf(Ns - 1, b), 1e-300, 1.0)), 0.0)
    cdf = np.cumsum(pN); Nmed = int(Ns[np.searchsorted(cdf, 0.5)])
    return dict(N_5sigma=N5, P_5sigma=p5, Z_median=float(ZN[Nmed]), N_median=Nmed, P_ge2=float(stats.poisson.sf(1, s + b)))

def live_for_Z(model, family, s, z_b, edge, Ztarget=5.0, rate=1.0, **kw):
    f = lambda T: Z_asimov(signal(model, bkg_components(family, s, z_b, edge, live=T, **kw)['M'], edge, T, rate),
                           bkg_components(family, s, z_b, edge, live=T, **kw)['total']) - Ztarget
    try: return optimize.brentq(f, 1.0, 1e7)
    except ValueError: return float('nan')

def evaluate(family, s, z_b, edge, **kw):
    b = bkg_components(family, s, z_b, edge, live=LIVE_NEW, **kw)
    out = dict(family=family, s=s, z_b=z_b, edge=edge, M=b['M'], b_wall=b['wall'], b_rfr=b['rfr'], b_uni=b['uni'], b_leak_lowE=b['leak_lowE'], b_total=b['total'],
               P_ge1_bkg=1 - math.exp(-b['total']), d_reco=b['d_reco'], d_reco_min=b['d_reco_min'], feasible=b['feasible'])
    for m in SPEC:
        sN = signal(m, b['M'], edge, LIVE_NEW)
        out[f's_{m}'] = sN; out[f'Z_{m}'] = float(Z_asimov(sN, b['total']))
        out[f'Zlo_{m}'] = float(Z_asimov(0.30 * sN, b['total']))      # lower 68 % rate band (Table I: 0.30)
        if m in ('L10', 'inel366'):
            ds = discovery_stats(sN, b['total'])
            out[f'P5_{m}'] = ds['P_5sigma']; out[f'N5_{m}'] = ds['N_5sigma']; out[f'Zmed_{m}'] = ds['Z_median']
            ds_lo = discovery_stats(0.30 * sN, b['total']); out[f'P5lo_{m}'] = ds_lo['P_5sigma']
    b220 = bkg_components(family, s, z_b, edge, **kw)
    out['b_total_220d'] = b220['total']
    out['FoM_M_sqrtb'] = b['M'] / math.sqrt(b['total'])        # background-limited figure of merit (untouched exposure)
    out['reach_MA_L10'] = (b['M'] / M_LZ_T) * ACC['L10'][edge] / ACC['L10'][600]     # zero-background exclusion reach
    out['reach_MA_inel366'] = (b['M'] / M_LZ_T) * ACC['inel366'][edge] / ACC['inel366'][600]
    return out

ref = evaluate('A', S_LZ, Z_B_LZ, 600)
say(f'\n[4] reference (LZ cuts, 600 phd) in the untouched {LIVE_NEW:.0f} live days: M = {ref["M"]:.2f} t, b = {ref["b_total"]:.2e} '
    f'(wall {ref["b_wall"]:.1e}, RFR {ref["b_rfr"]:.1e}, uni {ref["b_uni"]:.1e}); '
    f'Z_L10 = {ref["Z_L10"]:.2f}, Z_inel350 = {ref["Z_inel350"]:.2f}, Z_inel366 = {ref["Z_inel366"]:.2f}')
RES['reference'] = ref

# scan
S_GRID = np.arange(0.0, 16.01, 0.5); ZB_GRID = np.arange(0.0, 20.01, 1.0)
rows = []
for edge in EDGES:
    for s in S_GRID:
        for zb in ZB_GRID:
            rows.append(evaluate('A', float(s), float(zb), edge))
scan = pd.DataFrame(rows); scan.to_csv(f'{OUT}/P079_scan_familyA.csv', index=False)
say(f'    scan: {len(scan)} configurations (family A), {time.time()-T0:.0f} s')

def report_opt(df, col, label, floor_dreco=None, max_P=None, edges=None):
    d = df
    if floor_dreco is not None: d = d[d.d_reco_min >= floor_dreco - 1e-9]
    if max_P is not None: d = d[d.P_ge1_bkg <= max_P]
    if edges is not None: d = d[d.edge.isin(edges)]
    i = d[col].idxmax(); r = d.loc[i]
    say(f'    max {label:14s} (d_reco,min>={floor_dreco}, P(>=1 bkg)<={max_P}, edges {edges}): edge {int(r.edge)}, s = {r.s:.1f}, z_b = {r.z_b:.0f}: {col} = {r[col]:.3f}, '
        f'M = {r.M:.2f} t, b = {r.b_total:.2e}, P(>=1 bkg) = {r.P_ge1_bkg:.4f}, d_reco,min = {r.d_reco_min:.2f}')
    return r

say('\n    optima over the scan (family A, untouched exposure):')
OPT = {}
for col in ['Z_L10', 'Z_inel350', 'Z_inel366', 'FoM_M_sqrtb', 'reach_MA_L10']:
    OPT[col] = {}
    for fd, mp, ed in [(0.0, None, None), (D_RECO_FLOOR, None, None), (D_RECO_FLOOR, 0.01, None), (D_RECO_FLOOR, 0.025, None), (D_RECO_FLOOR, None, [600]), (D_RECO_FLOOR, None, [1000])]:
        r = report_opt(scan, col, col, floor_dreco=fd, max_P=mp, edges=ed)
        OPT[col][f'dreco{fd:g}_P{mp}_edges{ed}'] = r.to_dict()
RES['optima'] = OPT

# Z along s at z_b = 9, edge 600 and 1000, and along z_b
say('\n    Z_L10 vs s (z_b = 9): ' + ', '.join(f's={s:g}: {scan[(scan.edge==600)&(scan.z_b==9)&(np.isclose(scan.s,s))].Z_L10.iloc[0]:.3f}/{scan[(scan.edge==1000)&(scan.z_b==9)&(np.isclose(scan.s,s))].Z_L10.iloc[0]:.3f}' for s in [0, 2, 4, 6, 8, 10, 12, 16]) + '  (600/1000 phd)')
say('    b_total vs s (z_b = 9, 1000 phd): ' + ', '.join(f's={s:g}: {scan[(scan.edge==1000)&(scan.z_b==9)&(np.isclose(scan.s,s))].b_total.iloc[0]:.2e}' for s in [0, 2, 4, 6, 8, 12]))
say('    Z_L10 vs z_b (s = 8, 1000 phd): ' + ', '.join(f'z_b={zb:g}: {scan[(scan.edge==1000)&(scan.s==8)&(scan.z_b==zb)].Z_L10.iloc[0]:.3f}' for zb in [0, 2, 5, 9, 15]))
say('    FoM M/sqrt(b) vs s (z_b = 9, 600 phd): ' + ', '.join(f's={s:g}: {scan[(scan.edge==600)&(scan.z_b==9)&(np.isclose(scan.s,s))].FoM_M_sqrtb.iloc[0]:.0f}' for s in [0, 2, 4, 6, 8, 10, 12, 16]))

# 5.5 t counterfactual: what did the 14.5 % FV shrink buy for the high-energy window?
five5 = evaluate('54', 0.0, Z_B54, 600)
say(f'\n    counterfactual 5.4 t volume (drawn contour, z 2.15-135.75; calibrated M = {five5["M"]:.2f} t): b = {five5["b_total"]:.2e} '
    f'(wall {five5["b_wall"]:.1e}, RFR {five5["b_rfr"]:.1e}, uni {five5["b_uni"]:.1e}; whole-ROI wall leakage would be 1e-2 x {LIVE_NEW/LIVE_LZ:.2f}); '
    f'Z_L10 = {five5["Z_L10"]:.3f} vs {ref["Z_L10"]:.3f}; exclusion reach x{five5["reach_MA_L10"]:.2f}; d_reco,min = {five5["d_reco_min"]:.2f} cm')
# same, using LZ's own table numbers directly (wall 0.0048 + 0.03, RFR 0.0001 + 0.003) instead of the profile model
b_tab = ((N_WALL_LZ + N_WALL_54) * F_NB + (N_RFR_LZ + N_RFR_54) * F_NB + B_UNI_LZ['central'] * five5['M'] / M_LZ_T) * LIVE_NEW / LIVE_LZ
Z_tab = float(Z_asimov(signal('L10', five5['M'], 600, LIVE_NEW), b_tab))
say(f'    counterfactual with LZ table numbers directly: b = {b_tab:.2e}, Z_L10 = {Z_tab:.3f} (Delta Z vs 4.71 t = {Z_tab - ref["Z_L10"]:+.3f})')
five5['b_total_from_table'] = b_tab; five5['Z_L10_from_table'] = Z_tab
RES['counterfactual_5p4t'] = five5
# Z along s at z_b = 9 for 600 and 1000 phd: where does the maximum sit and how flat is it?
for edge in [600, 800, 1000]:
    d = scan[(scan.edge == edge) & (scan.z_b == 9)].sort_values('s')
    i = d.Z_L10.idxmax(); r = d.loc[i]
    within1 = d[d.Z_L10 >= 0.99 * r.Z_L10].s
    say(f'    edge {edge}: Z_L10(s) at z_b = 9 peaks at s = {r.s:.1f} (Z = {r.Z_L10:.3f}); within 1 % for s in [{within1.min():.1f}, {within1.max():.1f}] cm; Z(8) = {d[np.isclose(d.s, 8)].Z_L10.iloc[0]:.3f}')
    RES.setdefault('Z_vs_s_peak', {})[str(edge)] = dict(s_peak=float(r.s), Z_peak=float(r.Z_L10), s_within_1pct=[float(within1.min()), float(within1.max())], Z_at_8=float(d[np.isclose(d.s, 8)].Z_L10.iloc[0]))

# ----------------------------------------------------------------------------------------------
# 5. Soft FV: position-binned Asimov likelihood
# ----------------------------------------------------------------------------------------------
def soft_fv(model, edge, s_min, z_b, lam=None, lam_z=None, k_mssi=1.0, uni='central', d_bin=0.5, z_bin=2.0, live=LIVE_NEW, family='A'):
    """Bin the FV (family, s_min) in (distance-to-true-wall, z) cells; s_i uniform, b_i from the models. Returns Z_3D, Z_hard(s_min), M."""
    lam = LAM_W['central'] if lam is None else lam; lam_z = LAM_Z['central'] if lam_z is None else lam_z
    rz = r_of_z(family, s_min); m = zmask(z_b); zg = Z_GRID[m]; rmax = rz[m]
    add_w, add_r = added_mssi(edge)
    wall_ref = (N_WALL_LZ * F_NB + add_w * NB_SHARE_EXT) * k_mssi / I_wall(RZ_LZ, Z_B_LZ, lam)       # per unit I_w
    rfr_ref = (N_RFR_LZ * F_NB + add_r * NB_SHARE_EXT) * k_mssi / J_rfr(RZ_LZ, Z_B_LZ, lam_z)
    E50 = EDGES[edge][0]
    Mtot = KAPPA_M * mass_t(rz, z_b)
    uni_ref = (B_UNI_LZ[uni] + B_UNI_PER_KEV * max(E50 - 270.0, 0.0)) / M_LZ_T          # per tonne
    sc = live / LIVE_LZ
    s_tot = signal(model, Mtot, edge, live)
    # cells
    d_edges = np.arange(0.0, R_TRUE + d_bin, d_bin); z_edges = np.arange(z_b, Z_T + 1e-9, z_bin)
    S = []; B = []
    for zi in range(len(z_edges) - 1):
        mz = (zg >= z_edges[zi]) & (zg < z_edges[zi + 1])
        if not mz.any(): continue
        rmax_z = float(rmax[mz].mean()); dz = float(z_edges[zi + 1] - z_edges[zi]); zmid = 0.5 * (z_edges[zi] + z_edges[zi + 1])
        d_hi = np.minimum(d_edges[1:], R_TRUE); d_lo = d_edges[:-1]
        r_hi = np.clip(R_TRUE - d_lo, 0, rmax_z); r_lo = np.clip(R_TRUE - d_hi, 0, rmax_z)
        vol = np.pi * (r_hi ** 2 - r_lo ** 2) * dz
        ok = vol > 0
        iw = 2 * np.pi * (((lam * r_hi - lam ** 2) * np.exp(-(R_TRUE - r_hi) / lam)) - ((lam * r_lo - lam ** 2) * np.exp(-(R_TRUE - r_lo) / lam))) * dz
        b_cell = wall_ref * iw + rfr_ref * vol * math.exp(-zmid / lam_z) + uni_ref * vol * RHO / 1e6 * KAPPA_M
        s_cell = s_tot * (vol * RHO / 1e6 * KAPPA_M) / Mtot
        S.append(s_cell[ok]); B.append(b_cell[ok] * sc)
    S = np.concatenate(S); B = np.concatenate(B)
    Z3 = float(np.sqrt(np.sum(2 * ((S + B) * np.log1p(S / B) - S))))
    Zh = float(Z_asimov(S.sum(), B.sum()))
    return dict(Z_3D=Z3, Z_hard=Zh, M=Mtot, s=float(S.sum()), b=float(B.sum()), n_cells=int(len(S)))

say('\n[5] soft FV (position-binned likelihood, family A, untouched exposure):')
SOFT = {}
r600 = soft_fv('L10', 600, 8, 9); SOFT['edge600_s8_zb9_k1_lam4.3'] = r600
say(f'    600 phd, LZ volume: Z_3D = {r600["Z_3D"]:.3f} vs hard {r600["Z_hard"]:.3f} (P070: +0.1-0.2 sigma for the actual event)')
for s_min, zb in [(8, 9), (7, 9), (6, 9), (6, 5), (5, 5), (4, 5), (3, 5), (3, 2)]:
    for k, lam in [(1.0, 4.3), (2.0, 6.0), (1.0, LAM_W['lz_table']), (2.0, LAM_W['lz_table'])]:
        r = soft_fv('L10', 1000, s_min, zb, lam=lam, k_mssi=k)
        SOFT[f's{s_min}_zb{zb}_k{k:g}_lam{lam:.3g}'] = r
        say(f'    1000 phd, s_min={s_min}, z_b={zb}, k={k:g}, lambda={lam:.3g}: M = {r["M"]:.2f} t, s = {r["s"]:.2f}, b = {r["b"]:.2e}: Z_3D = {r["Z_3D"]:.3f}, Z_hard(same volume) = {r["Z_hard"]:.3f}')
RES['soft_fv'] = SOFT

# equivalent hard-cut mass: mass at which the hard-cut Z (nominal model, s = 8 shape) equals Z_3D
def Z_hard_mass(M, edge, model='L10'):
    bref = evaluate('A', S_LZ, Z_B_LZ, edge)
    b = bref['b_total'] * M / bref['M']       # scale all backgrounds with mass (conservative for the wall part)
    return float(Z_asimov(signal(model, M, edge, LIVE_NEW), b))
for key in ['s8_zb9_k1_lam4.3', 's6_zb9_k1_lam4.3', 's6_zb5_k1_lam4.3', 's5_zb5_k1_lam4.3', 's3_zb5_k1_lam4.3', 's3_zb2_k1_lam4.3', f's6_zb5_k1_lam{LAM_W["lz_table"]:.3g}', f's6_zb5_k2_lam{LAM_W["lz_table"]:.3g}']:
    r = SOFT[key]
    Meq = optimize.brentq(lambda M: Z_hard_mass(M, 1000) - r['Z_3D'], 1.0, 40.0)
    r['M_equivalent_hard_t'] = Meq
    say(f'    {key}: Z_3D = {r["Z_3D"]:.3f} equals a hard-cut analysis (LZ-shaped FV, 1000 phd) of M = {Meq:.2f} t (actual {r["M"]:.2f} t; LZ 4.71)')

# single-event position weight: effective background for an event observed at distance d from the wall
def LR_position(d, lam=4.3, s_min=3.0, zb=5.0):
    """uniform/wall-MSSI PDF ratio at true-wall distance d in the (s_min, z_b) volume (z-uniform exponential)."""
    rz = r_of_z('A', s_min); m = zmask(zb)
    V = volume_cm3(rz, zb); Iw = I_wall(rz, zb, lam)
    r = R_TRUE - d
    return (1.0 / V) / (math.exp(-d / lam) / Iw)
say('    position LR (uniform/wall, lambda 4.3, volume s_min=3, z_b=5) at d = 3, 5, 8, 12, 20, 27 cm: ' +
    ', '.join(f'{LR_position(d):.2f}' for d in [3, 5, 8, 12, 20, 27]))
RES['LR_position'] = {str(d): LR_position(d) for d in [3, 4, 5, 6, 8, 10, 12, 15, 20, 27]}

# ----------------------------------------------------------------------------------------------
# 6. Recommendation table with robustness variants
# ----------------------------------------------------------------------------------------------
CONFIGS = [('LZ 2026 (600 phd, s=8, z_b=9)', 'A', 8.0, 9.0, 600),
           ('edge 800 (s=8, z_b=9)', 'A', 8.0, 9.0, 800),
           ('edge 1000 (s=8, z_b=9)', 'A', 8.0, 9.0, 1000),
           ('edge 1200 (s=8, z_b=9)', 'A', 8.0, 9.0, 1200),
           ('1000 phd, s=7, z_b=9', 'A', 7.0, 9.0, 1000),
           ('1000 phd, s=6, z_b=9', 'A', 6.0, 9.0, 1000),
           ('1000 phd, s=6, z_b=5', 'A', 6.0, 5.0, 1000),
           ('1000 phd, s=5, z_b=5 (d_reco,min 3.2 cm)', 'A', 5.0, 5.0, 1000),
           ('1000 phd, reco-margin 3 cm, z_b=5', 'C', 3.0, 5.0, 1000),
           ('1000 phd, cylinder s=6, z_b=5 (infeasible at bottom)', 'B', 6.0, 5.0, 1000),
           ('600 phd, s=6, z_b=5', 'A', 6.0, 5.0, 600),
           ('600 phd, reco-margin 3 cm, z_b=5', 'C', 3.0, 5.0, 600)]
trows = []
say('\n[6] recommendation table (untouched exposure, 524 live days); worst case = k 2, lambda LZ-table 1.2 cm, lambda_z 2.0 cm, uniform high:')
for name, fam, s, zb, edge in CONFIGS:
    nom = evaluate(fam, s, zb, edge)
    worst = evaluate(fam, s, zb, edge, lam=LAM_W['lz_table'], lam_z=LAM_Z['short'], k_mssi=2.0, uni='high')
    worst_long = evaluate(fam, s, zb, edge, lam=LAM_W['long'], lam_z=LAM_Z['long'], k_mssi=2.0, uni='high')
    best = evaluate(fam, s, zb, edge, lam=LAM_W['short'], lam_z=LAM_Z['long'], k_mssi=0.5)
    T5 = {m: live_for_Z(m, fam, s, zb, edge) for m in ['L10', 'inel350', 'inel366']}
    T5_lo = live_for_Z('L10', fam, s, zb, edge, rate=0.30)
    soft = soft_fv('L10', edge, s, zb, family=fam); soft_w = soft_fv('L10', edge, s, zb, family=fam, lam=LAM_W['lz_table'], k_mssi=2.0, lam_z=LAM_Z['short'], uni='high')
    row = dict(config=name, family=fam, s=s, z_b=zb, edge=edge, M_t=nom['M'], mass_gain=nom['M'] / ref['M'] - 1, d_reco_min=nom['d_reco_min'], feasible=nom['feasible'],
               b_wall=nom['b_wall'], b_rfr=nom['b_rfr'], b_uni=nom['b_uni'], b_leak_lowE_diag=nom['b_leak_lowE'], b_total=nom['b_total'], b_total_worst=worst['b_total'], b_total_worst_long=worst_long['b_total'], b_total_best=best['b_total'],
               P_ge1_bkg=nom['P_ge1_bkg'], P_ge1_bkg_worst=worst['P_ge1_bkg'],
               Z_L10=nom['Z_L10'], Z_L10_worst=worst['Z_L10'], Z_L10_3D=soft['Z_3D'], Z_L10_3D_worst=soft_w['Z_3D'], Z_inel350=nom['Z_inel350'], Z_inel366=nom['Z_inel366'],
               N5sigma=nom['N5_L10'], N5sigma_worst=worst['N5_L10'], P5_L10=nom['P5_L10'], P5_L10_worst=worst['P5_L10'], P5lo_L10=nom['P5lo_L10'], Zmed_L10=nom['Zmed_L10'], Zmed_L10_worst=worst['Zmed_L10'],
               P5_inel366=nom['P5_inel366'], P5_inel366_worst=worst['P5_inel366'], P5lo_inel366=nom['P5lo_inel366'],
               Zlo_L10=nom['Zlo_L10'], Zlo_inel366=nom['Zlo_inel366'],
               s_L10=nom['s_L10'], s_inel350=nom['s_inel350'], s_inel366=nom['s_inel366'],
               live_5sigma_L10_d=T5['L10'], live_5sigma_inel350_d=T5['inel350'], live_5sigma_inel366_d=T5['inel366'], live_5sigma_L10_lo68_d=T5_lo,
               tyr_5sigma_L10=T5['L10'] / 365.25 * nom['M'], tyr_5sigma_inel366=T5['inel366'] / 365.25 * nom['M'],
               reach_MA_L10=nom['reach_MA_L10'], reach_MA_inel366=nom['reach_MA_inel366'], FoM_M_sqrtb=nom['FoM_M_sqrtb'], d_reco_mean=nom['d_reco'])
    trows.append(row)
    say(f'    {name:52s} M {nom["M"]:.2f} t ({100*row["mass_gain"]:+.1f} %), d_reco,min {nom["d_reco_min"]:.1f}, b {nom["b_total"]:.1e} [worst {worst["b_total"]:.1e}], '
        f'Z_L10 {nom["Z_L10"]:.2f} [worst {worst["Z_L10"]:.2f}; 3D {soft["Z_3D"]:.2f}/{soft_w["Z_3D"]:.2f}], Z_350 {nom["Z_inel350"]:.2f}, Z_366 {nom["Z_inel366"]:.2f}; '
        f'5sigma live d: L10 {T5["L10"]:.0f}, 350 {T5["inel350"]:.0f}, 366 {T5["inel366"]:.0f} (L10 lo68 {T5_lo:.0f}); reach x{nom["reach_MA_L10"]:.2f}/{nom["reach_MA_inel366"]:.2f}; P(>=1 bkg) {nom["P_ge1_bkg"]:.4f} [worst {worst["P_ge1_bkg"]:.4f}]')
    say(f'    {"":52s} exact Poisson: N_5sigma = {nom["N5_L10"]} [worst {worst["N5_L10"]}]; P(5sigma | L10 best) = {nom["P5_L10"]:.3f} [worst {worst["P5_L10"]:.3f}; lo68 rate {nom["P5lo_L10"]:.3f}], '
        f'median Z {nom["Zmed_L10"]:.2f} [worst {worst["Zmed_L10"]:.2f}]; P(5sigma | inel366) = {nom["P5_inel366"]:.3f} [worst {worst["P5_inel366"]:.3f}; lo68 {nom["P5lo_inel366"]:.3f}]')
table = pd.DataFrame(trows); table.to_csv(f'{OUT}/P079_recommendation_table.csv', index=False)
# P(5 sigma) map over the scan for L10 (nominal), to locate the few-event optimum
scan['P5_L10'] = [discovery_stats(sv, bv)['P_5sigma'] for sv, bv in zip(scan.s_L10, scan.b_total)]
scan.to_csv(f'{OUT}/P079_scan_familyA.csv', index=False)
for ed in [600, 800, 1000, 1200]:
    d = scan[(scan.edge == ed) & (scan.d_reco_min >= D_RECO_FLOOR - 1e-9)]
    i = d.P5_L10.idxmax(); r = d.loc[i]
    say(f'    P(5sigma | L10) optimum, edge {ed}: s = {r.s:.1f}, z_b = {r.z_b:.0f}: P5 = {r.P5_L10:.3f} (M {r.M:.2f} t, b {r.b_total:.1e}); at s=8,z_b=9: {d[(d.s==8)&(d.z_b==9)].P5_L10.iloc[0]:.3f}')
    RES.setdefault('P5_optimum', {})[str(ed)] = dict(s=float(r.s), z_b=float(r.z_b), P5=float(r.P5_L10), M=float(r.M), b=float(r.b_total), P5_at_LZ_cuts=float(d[(d.s==8)&(d.z_b==9)].P5_L10.iloc[0]))
RES['table'] = trows

# joint optimum incl. edge with the reconstruction floor, robust (worst-case) variant
say('\n    joint optimum incl. edge (family A, d_reco,min >= floor): nominal vs worst-case MSSI (k=2, lambda 1.2/2.0, uniform high):')
JOINT = {}
for m in ['L10', 'inel350', 'inel366']:
    for fd, mp in [(D_RECO_FLOOR, None), (D_RECO_FLOOR, 0.01), (D_RECO_FLOOR, 0.025)]:
        d = scan[(scan.d_reco_min >= fd - 1e-9)]
        if mp is not None: d = d[d.P_ge1_bkg <= mp]
        i = d[f'Z_{m}'].idxmax(); r = d.loc[i]
        w = evaluate('A', float(r.s), float(r.z_b), int(r.edge), lam=LAM_W['lz_table'], lam_z=LAM_Z['short'], k_mssi=2.0, uni='high')
        JOINT[f'{m}_dreco{fd:g}_P{mp}'] = dict(edge=int(r.edge), s=float(r.s), z_b=float(r.z_b), Z=float(r[f'Z_{m}']), Z_worst=w[f'Z_{m}'], M=float(r.M), b=float(r.b_total), b_worst=w['b_total'], P_worst=w['P_ge1_bkg'], gain_over_LZ=float(r[f'Z_{m}']) / ref[f'Z_{m}'])
        say(f'    {m:8s} P<={mp}: edge {int(r.edge)}, s {r.s:.1f}, z_b {r.z_b:.0f}: Z {r[f"Z_{m}"]:.2f} (LZ cuts {ref[f"Z_{m}"]:.2f}, x{JOINT[f"{m}_dreco{fd:g}_P{mp}"]["gain_over_LZ"]:.2f}); worst-case Z {w[f"Z_{m}"]:.2f}, b {r.b_total:.1e} -> {w["b_total"]:.1e}, P(>=1) {w["P_ge1_bkg"]:.3f}')
RES['joint_optimum'] = JOINT

# sensitivity of the wall-MSSI count at small stand-off to the profile (model dependence)
say(f'\n    wall MSSI (NR band, 1000 phd, 524 d) vs s for lambda = {LAM_W["lz_table"]:.2f} (LZ table) / 3.3 / 4.3 / 6 cm, k = 1, z_b = 5:')
MD = {}
for s in [2, 3, 4, 5, 6, 7, 8]:
    vals = [bkg_components('A', s, 5.0, 1000, lam=l, live=LIVE_NEW)['wall'] for l in (LAM_W['lz_table'], 3.3, 4.3, 6.0)]
    MD[str(s)] = vals
    say(f'    s = {s}: ' + ' / '.join(f'{v:.2e}' for v in vals) + f'   (spread x{max(vals)/min(vals):.1f})')
RES['wall_mssi_model_dependence'] = MD
say(f'    RFR MSSI (NR band, 1000 phd, 524 d) vs z_b for lambda_z = 2.0/2.7/6.5 (s = 6): ' + '; '.join(
    f'z_b={zb}: ' + '/'.join(f'{bkg_components("A", 6.0, zb, 1000, lam_z=l, live=LIVE_NEW)["rfr"]:.1e}' for l in (2.0, 2.7, 6.5)) for zb in [2, 5, 7, 9]))
RES['rfr_mssi_model_dependence'] = {str(zb): [bkg_components('A', 6.0, zb, 1000, lam_z=l, live=LIVE_NEW)['rfr'] for l in (2.0, 2.7, 6.5)] for zb in [2, 5, 7, 9]}
# L10 anchoring note: fraction of the LZ-ROI L10 spectrum above 200 keV observed (for converting to LZ's whole-ROI best fit)
E, dR = SPEC['L10']; obs600 = observed_spectrum(E, dR, 600)
frac_above200 = float(np.trapezoid(obs600[E_OBS >= 200], E_OBS[E_OBS >= 200]) / np.trapezoid(obs600, E_OBS))
say(f'    L10: fraction of accepted LZ-ROI events with E_obs > 200 keV = {frac_above200:.3f} (LZ whole-ROI best fit 1.0 event -> {frac_above200:.2f} above 200 keV)')
RES['L10_frac_above_200keV_LZ_ROI'] = frac_above200

# what fraction of the untouched-exposure background sits within 2 cm of the FV edge for s = 3?
r3 = soft_fv('L10', 1000, 3.0, 5.0)
RES['timing_s'] = time.time() - T0

# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------
# Fig 1: Z map in (s, z_b) for L10 at 1000 phd, with LZ's point and the recommended point
sub = scan[scan.edge == 1000].pivot(index='z_b', columns='s', values='Z_L10')
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
im = ax[0].pcolormesh(sub.columns.values, sub.index.values, sub.values, shading='nearest', cmap='viridis')
cs = ax[0].contour(sub.columns.values, sub.index.values, sub.values, levels=[6.0, 6.2, 6.4, 6.6], colors='w', linewidths=0.8)
ax[0].clabel(cs, fmt='%.1f', fontsize=8)
S_FLOOR_A = D_RECO_FLOOR + (S_LZ - b0['d_reco_min'])      # family-A stand-off at which d_reco,min = floor
RES['s_floor_familyA_cm'] = S_FLOOR_A
ax[0].plot([8], [9], marker='*', color='#E69F00', ms=14, label='LZ 2026 cuts (s = 8, z_b = 9)')
ax[0].plot([7], [9], marker='o', color='#D55E00', ms=9, label='robust hard cut (s = 7, z_b = 9)')
ax[0].plot([6], [9], marker='s', color='#D55E00', ms=7, mfc='none', label='with position likelihood (s = 6)')
ax[0].axvspan(0, S_FLOOR_A, color='k', alpha=0.25, lw=0, label=f'd_reco,min < {D_RECO_FLOOR:g} cm (reconstruction floor)')
ax[0].set_xlabel('minimum true-wall stand-off s [cm]'); ax[0].set_ylabel('bottom cut z_b above cathode [cm]')
ax[0].set_title('Asimov Z (L10 best fit), 1000 phd edge, 524 live d', fontsize=10); ax[0].legend(fontsize=7, loc='upper right')
fig.colorbar(im, ax=ax[0], label='Z [sigma]')
sub6 = scan[scan.edge == 600].pivot(index='z_b', columns='s', values='Z_L10')
ax[1].plot(sub.columns.values, sub.loc[9.0].values, color='#0072B2', lw=2, label='1000 phd, z_b = 9')
ax[1].plot(sub.columns.values, sub.loc[5.0].values, color='#0072B2', lw=1.2, ls='--', label='1000 phd, z_b = 5')
ax[1].plot(sub6.columns.values, sub6.loc[9.0].values, color='#009E73', lw=2, label='600 phd, z_b = 9')
ax[1].axvline(8, color='#E69F00', lw=1, ls=':'); ax[1].set_xlabel('s [cm]'); ax[1].set_ylabel('Z [sigma]'); ax[1].legend(fontsize=8)
ax[1].set_title('Z versus stand-off', fontsize=10); ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f'{FIG}/P079_fig1_Z_map.png', dpi=160); plt.close(fig)

# Fig 2: backgrounds and mass vs s (1000 phd, z_b = 5), with lambda bracket, and FoM
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ss = np.arange(0.0, 16.01, 0.25)
comp = {l: [bkg_components('A', s, 5.0, 1000, lam=l, live=LIVE_NEW) for s in ss] for l in (3.0, 4.3, 6.0)}
ax[0].fill_between(ss, [c['wall'] for c in comp[3.0]], [c['wall'] for c in comp[6.0]], color='#CC79A7', alpha=0.3, lw=0, label='wall MSSI (lambda 3-6 cm)')
ax[0].plot(ss, [c['wall'] for c in comp[4.3]], color='#CC79A7', lw=2, label='wall MSSI (lambda 4.3 cm)')
ax[0].plot(ss, [c['uni'] for c in comp[4.3]], color='#0072B2', lw=2, label='uniform (accidentals + nu + ER)')
ax[0].plot(ss, [c['rfr'] for c in comp[4.3]], color='#009E73', lw=2, label='RFR MSSI (z_b = 5)')
ax[0].plot(ss, [bkg_components('A', s, 5.0, 1000, lam=LAM_W['lz_table'], live=LIVE_NEW)['wall'] for s in ss], color='#CC79A7', lw=1.2, ls='--', label=f'wall MSSI (LZ-table lambda {LAM_W["lz_table"]:.1f} cm)')
ax[0].plot(ss, [c['total'] for c in comp[4.3]], color='k', lw=1.2, label='total (lambda 4.3)')
ax[0].axvspan(0, S_FLOOR_A, color='k', alpha=0.12, lw=0, label=f'd_reco,min < {D_RECO_FLOOR:g} cm (family A)')
ax[0].set_yscale('log'); ax[0].set_ylim(1e-6, 1e-1); ax[0].set_xlabel('s [cm]'); ax[0].set_ylabel('NR-band events, 200-423 keV, 524 live d')
ax[0].axvline(8, color='#E69F00', lw=1, ls=':'); ax[0].legend(fontsize=7, loc='upper right'); ax[0].grid(alpha=0.3, which='both')
ax[0].set_title('backgrounds versus stand-off (1000 phd)', fontsize=10)
ax2 = ax[1]; ax2.plot(ss, [c['M'] for c in comp[4.3]], color='k', lw=2, label='fiducial mass (family A)')
ax2.plot(ss, [KAPPA_M * mass_t(r_of_z('B', s), 5.0) for s in ss], color='k', lw=1.2, ls='--', label='cylinder family B')
ax2.set_xlabel('s [cm]'); ax2.set_ylabel('M [t]'); ax2.axvline(8, color='#E69F00', lw=1, ls=':'); ax2.grid(alpha=0.3)
ax3 = ax2.twinx()
fom = np.array([c['M'] / math.sqrt(c['total']) for c in comp[4.3]]); ax3.plot(ss, fom / fom[np.argmin(abs(ss - 8))], color='#D55E00', lw=2, label='M/sqrt(b) rel. to s = 8')
ax3.set_ylabel('M/sqrt(b) relative', color='#D55E00')
h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax3.get_legend_handles_labels(); ax2.legend(h1 + h2, l1 + l2, fontsize=7, loc='lower left')
ax2.set_title('mass and background-limited figure of merit', fontsize=10)
fig.tight_layout(); fig.savefig(f'{FIG}/P079_fig2_bkg_mass_vs_s.png', dpi=160); plt.close(fig)

with open(f'{OUT}/P079_results.json', 'w') as f:
    json.dump(RES, f, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
say(f'\ndone in {time.time()-T0:.0f} s; outputs in {OUT}')
