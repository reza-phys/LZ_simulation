"""
P041_charge_loss.py -- Detector artefacts: could partial charge loss turn a ~71 keV electron recoil
into the LZ 248 keV event (arXiv:2609.02823)?

Run from the simulation root:  .venv/bin/python output/code/P041_charge_loss.py

Sections
  0  inputs (LZ paper via lzcommon.LZ; corpus P010/P022/P024; recalled values flagged)
  1  required charge loss and the (S1c = 540) landing regions in log10 S2c, incl. a digitisation of the
     lower envelope of the grey ER population in Fig. 4 (the "empty strip" 4.15 < log S2c < envelope)
  2  electron-lifetime mechanism: implied tau_e, loss vs drift time, what the whole sample would show,
     monitor-event rates and the minimum undetectable excursion duration
  3  local absorber (bubble / particulate / impurity plume / field defect): electron-cloud footprint,
     absorbing-disk loss distribution, area fraction needed, population predicted in the empty regions
  4  extraction / gas-region dip versus the 83mKr S2c map; pulse pathologies (clipping, truncation)
  5  Bayesian channel summary (judgement-based factors, all printed)
Outputs: output/work/P041/*.csv, P041_results.json, figures/*.png
"""
import sys, json, os
sys.path.insert(0, 'output/code')
import numpy as np
from scipy import integrate, special, stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from common import lzcommon as lz

OUT = 'output/work/P041'
FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)
R = {}            # results dictionary -> JSON
log_lines = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); log_lines.append(s)

# ----------------------------------------------------------------------------------------------
# 0. Inputs
# ----------------------------------------------------------------------------------------------
LZ = lz.LZ
g1, g2, g2_err = LZ['g1'], LZ['g2'], LZ['g2_err']
S1c, S2c = LZ['ev_S1c'], LZ['ev_S2c']
Ne_obs = S2c / g2                                   # 268.6 electrons
# corpus
NE_ER_P010 = 1226.0        # P010: mean electrons of a 69.6 keV beta ER (r = 0.720, N_i = 4379)
SIG_NE_P010 = 150.0        # P010: band-calibrated sigma(N_e) at this energy
SIG_NR_DEX = 0.0313        # P024: NR-band sigma(log10 S2c) at S1c = 540
NR_MEDIAN = 4.015          # P024/P009: NR-band median log10 S2c at S1c = 540
T_MAX_US, T_EV_US = 1049.0, 859.0    # P022 (Fig. 3 axis): cathode drift time, event drift time
V_D_MM_US = 1.388                    # P022: drift speed
T_FV_LO, T_FV_HI = 93.0, 985.0       # P022: FV in drift time (12.8 cm below gate .. 9.0 cm above cathode)
L_DRIFT_CM, R_TPC_CM = 145.6, 72.8   # recalled (likely): gate-cathode distance, active radius
N_SCI, LIVE_D = LZ['n_obs_science'], LZ['live_days']
# recalled physics (flagged in details.md)
D_T_CM2S = 55.0; D_T_RANGE = (50.0, 60.0)   # transverse diffusion in LXe at ~100 V/cm (likely)
D_L_CM2S = 25.0                              # longitudinal diffusion (P022, likely)
TAU_CAL_MS = [np.inf, 10.0, 5.0]             # LZ electron lifetimes >~ 5-10 ms (likely)
EPS_EXT = 0.80                               # extraction efficiency (uncertain; P024 assumption)
MAP_VAR = 0.20                               # typical S2c (x,y) correction amplitude (likely)
XE136_ABUND, XE136_T12_YR = 0.0886, 2.165e21 # (certain / likely)
RN_UBQ_KG = (1.0, 0.5, 2.0)                  # 222Rn in LZ WS2024, uBq/kg (uncertain bracket)
KR_EVENTS_PER_INJ = (1e5, 1e6)               # 83mKr decays per injection (uncertain)
S2_RES_KR = 0.10                             # per-event S2 resolution of 83mKr (uncertain)

# nestpy cross-check of the ER electron count at the event's combined energy
E_ee = lz.combined_energy_keV(S1c, S2c, W_eV=13.44)   # P010 convention
Nph_n, Ne_n = lz.nest_er_yields(E_ee)
say(f"E_ee (W=13.44 eV) = {E_ee:.1f} keV; nestpy LZ-ER yields: Nph = {Nph_n:.0f}, Ne = {Ne_n:.0f}; P010 N_e = {NE_ER_P010:.0f}")
R['E_ee_keV'] = E_ee; R['Ne_ER_nestpy'] = Ne_n; R['Ne_ER_P010'] = NE_ER_P010; R['Ne_obs'] = Ne_obs

# ----------------------------------------------------------------------------------------------
# 1. Required loss and landing regions
# ----------------------------------------------------------------------------------------------
f_req = Ne_obs / NE_ER_P010            # survival fraction needed
f_req_nest = Ne_obs / Ne_n
L_req = 1 - f_req
logS2_ER = np.log10(NE_ER_P010 * g2)   # ER-band mean log10 S2c at S1c = 540 (P010 quanta)
say(f"survival needed f = {f_req:.3f} (nestpy variant {f_req_nest:.3f}); loss L = {L_req:.3f}; ER mean log10 S2c = {logS2_ER:.3f}")
R.update(f_req=f_req, f_req_nest=f_req_nest, L_req=L_req, logS2_ER_mean=logS2_ER)

# --- digitise Fig. 4: lower envelope of the grey ER population at S1c 450-600 and emptiness of the strip
im = np.array(Image.open('inputs/figures_png/Fig4_science_sample_wbands.png').convert('RGB')).astype(int)
XL, XR, ROW_415, ROW_275 = 143, 1185, 361, 898        # frame columns (0..800 phd), ROI top/bottom rows
def x_of(s1): return XL + s1 / 800.0 * (XR - XL)
slope = (ROW_275 - ROW_415) / (2.75 - 4.15)          # px per dex (negative)
def row_of(l): return ROW_415 + (l - 4.15) * slope
def l_of(row): return 4.15 + (row - ROW_415) / slope
r_, g_, b_ = im[:, :, 0], im[:, :, 1], im[:, :, 2]
grey_pts = (abs(r_ - g_) < 6) & (abs(g_ - b_) < 6) & (r_ > 100) & (r_ < 180)   # neutral grey data points
# colour census in the S1c 500-580 column block above the ROI, to show contour lines are not neutral grey
blk = im[int(row_of(4.9)):ROW_415, int(x_of(500)):int(x_of(580))].reshape(-1, 3)
cols, cnt = np.unique(blk, axis=0, return_counts=True)
top = np.argsort(cnt)[::-1][:6]
say("Fig.4 colour census (S1c 500-580, log S2c 4.15-4.9):", [(tuple(cols[i]), int(cnt[i])) for i in top])
env = []
for s1 in np.arange(450, 601, 10):
    x0 = int(round(x_of(s1)))
    colblk = grey_pts[int(row_of(5.0)):ROW_415 - 3, x0 - 4:x0 + 5]
    rows = np.where(colblk.any(axis=1))[0]
    env.append((s1, l_of(rows.max() + int(row_of(5.0))) if len(rows) else np.nan))
env = np.array(env)
env_540 = float(np.nanmedian(env[(env[:, 0] >= 500) & (env[:, 0] <= 580), 1]))
say(f"grey ER lower envelope at S1c 500-580: median log10 S2c = {env_540:.3f} (columns: "
    + ', '.join(f'{int(a)}:{b:.2f}' for a, b in env) + ')')
# envelope over the full high-S1c range and emptiness of the band just above the ROI ceiling
env_all = []
for s1 in np.arange(330, 801, 10):
    x0 = int(round(x_of(s1)))
    colblk = grey_pts[int(row_of(5.0)):ROW_415 - 3, x0 - 4:x0 + 5]
    rows = np.where(colblk.any(axis=1))[0]
    env_all.append((s1, l_of(rows.max() + int(row_of(5.0))) if len(rows) else np.nan))
env_all = np.array(env_all)
np.savetxt(os.path.join(OUT, 'P041_fig4_grey_envelope.csv'), env_all, delimiter=',', header='S1c_phd,lowest_grey_log10S2c', comments='')
strip = grey_pts[int(row_of(4.30)):ROW_415 - 3, int(x_of(520)):int(x_of(800))]
n_strip = int(strip.sum())
say(f"grey pixels at S1c 520-800 in log S2c 4.15-4.30: {n_strip} (one plotted point ~ 10-15 px); "
    f"envelope S1c 330/400/500/600/700/800: " + '/'.join(f'{v:.2f}' for v in np.interp([330, 400, 500, 600, 700, 800], env_all[:, 0], env_all[:, 1])))
blk_pts = ((r_ < 40) & (g_ < 40) & (b_ < 40))[ROW_415 + 3:ROW_275 - 3, int(x_of(330)):int(x_of(600))]
say(f"black pixels inside the ROI at S1c 330-600: {int(blk_pts.sum())} (the event alone ~ a few tens of px)")
R['fig4_envelope_logS2_540'] = env_540; R['fig4_strip_grey_pixels_520_800_415_430'] = n_strip
R['fig4_envelope_columns'] = env.tolist(); R['fig4_black_px_ROI_330_600'] = int(blk_pts.sum())

# landing regions at S1c = 540 (log10 S2c) -> survival-fraction windows f = 10^(l - logS2_ER)
regions = {
    'ER cloud (f > envelope)': (env_540, 5.0),
    'strip (out of ROI, empty)': (4.15, env_540),
    'gap G (in ROI, above NR+2sigma, empty)': (NR_MEDIAN + 2 * SIG_NR_DEX, 4.15),
    'V (NR band +-2sigma, 1 event)': (NR_MEDIAN - 2 * SIG_NR_DEX, NR_MEDIAN + 2 * SIG_NR_DEX),
    'below NR-2sigma (to S2 threshold, empty)': (np.log10(645.0), NR_MEDIAN - 2 * SIG_NR_DEX),
}
reg_f = {k: (10 ** (v[0] - logS2_ER), min(10 ** (v[1] - logS2_ER), 1.0)) for k, v in regions.items()}
say("\nLanding regions at S1c = 540 (log10 S2c -> survival fraction f):")
for k, v in regions.items():
    say(f"  {k:45s} log S2c [{v[0]:.3f}, {v[1]:.3f}]  f in [{reg_f[k][0]:.3f}, {reg_f[k][1]:.3f}]")
R['regions_logS2'] = {k: list(map(float, v)) for k, v in regions.items()}
R['regions_f'] = {k: list(map(float, v)) for k, v in reg_f.items()}
fV = reg_f['V (NR band +-2sigma, 1 event)']
fG = reg_f['gap G (in ROI, above NR+2sigma, empty)']
fS = reg_f['strip (out of ROI, empty)']
fB = reg_f['below NR-2sigma (to S2 threshold, empty)']
# flat-in-loss reference: probabilities proportional to window widths in f
flat = {k: v[1] - v[0] for k, v in reg_f.items()}
say("flat-in-loss probabilities:", {k: round(v, 3) for k, v in flat.items()})
R['flat_loss_probs'] = flat

# --- parent ER population (60-80 keV and S1c 250-600 windows) from Table I ROI counts
bk = LZ['bkg_expected']
N_ROI_ER = sum(bk[k][0] for k in ['internal_beta', 'nu_ER', 'Xe136', 'detector_ER', 'CH3T_C14',
                                   'Xe127_Xe125', 'Kr83m', 'Xe124', 'I125'])
E_cross = float(integrate.newton_cotes(1)[0][0]) if False else None
# ER-band median leaves the ROI (S2c = 10^4.15) where Ne*g2 = 10^4.15
Es = np.arange(2, 40, 0.5); Nes = np.array([lz.nest_er_yields(E)[1] for E in Es])
E_roi = float(np.interp(10 ** 4.15 / g2, Nes, Es))
dRdE = N_ROI_ER / (E_roi - 1.5)          # events per keV per 2.84 t yr (flat approximation; E_min ~1.5 keV)
N_par_60_80 = dRdE * 20 + 564            # + 124Xe KK line (P010: 564 decays)
N_par_30_78 = dRdE * 48 + 564            # S1c ~ 250-600 phd window (Fig. 4 contours 30-78 keVee)
say(f"\nROI ER count (Table I) = {N_ROI_ER:.0f}; ER band exits ROI at {E_roi:.1f} keV -> dR/dE ~ {dRdE:.0f}/keV/2.84 t yr")
say(f"parent ER populations (flat extrapolation, x0.5-2 bracket): 60-80 keV {N_par_60_80:.0f}; 30-78 keV (S1c 250-600) {N_par_30_78:.0f}")
R.update(N_ROI_ER=N_ROI_ER, E_roi_exit_keV=E_roi, dRdE_per_keV=dRdE, N_par_60_80=N_par_60_80, N_par_30_78=N_par_30_78)

# --- all-energy monitor rates in the 4.71 t FV (ER betas and alphas with large S2s), per hour
M_FV_KG = LZ['fiducial_mass_t'] * 1e3
N136 = M_FV_KG * 1e3 / 131.29 * 6.02214e23 * XE136_ABUND
A136 = N136 * np.log(2) / (XE136_T12_YR * 3.15576e7)           # Bq
rate_136_h = A136 * 3600
rn_h = np.array(RN_UBQ_KG) * 1e-6 * M_FV_KG * 3600            # Rn decays per hour (central, lo, hi)
rate_beta_h = rate_136_h + 2 * rn_h                            # 214Pb + 214Bi betas
rate_alpha_h = 3 * rn_h                                         # 222Rn, 218Po, 214Po
rate_mon_h = rate_beta_h + rate_alpha_h
say(f"136Xe 2nbb in FV: {A136*1e3:.1f} mBq = {rate_136_h:.0f}/h; Rn chain {rn_h[0]:.0f}/h (bracket {rn_h[1]:.0f}-{rn_h[2]:.0f})")
say(f"monitor events (beta + alpha, full FV, all energies): {rate_mon_h[0]:.0f}/h (bracket {rate_mon_h[1]:.0f}-{rate_mon_h[2]:.0f})")
R.update(rate_136Xe_per_h=rate_136_h, rate_Rn_per_h=rn_h.tolist(), rate_monitor_per_h=rate_mon_h.tolist())
N_all_ER_run = rate_beta_h[0] * LIVE_D * 24
R['N_all_ER_run'] = N_all_ER_run

# ----------------------------------------------------------------------------------------------
# 2. Electron lifetime
# ----------------------------------------------------------------------------------------------
say("\n=== 2. Electron lifetime ===")
tau_true = {}
for tc in TAU_CAL_MS:
    inv = (0 if np.isinf(tc) else 1 / (tc * 1e3)) + (-np.log(f_req)) / T_EV_US
    tau_true[str(tc)] = 1 / inv
    say(f"  calibration lifetime {tc} ms -> true lifetime needed {1/inv:.0f} us")
tau_e = tau_true['inf']
R['tau_true_us'] = tau_true
# where would 70 keV ERs land vs drift time under tau_e?
def t_of_f(f): return -tau_e * np.log(f)
win = {k: (t_of_f(v[1]), t_of_f(v[0])) for k, v in reg_f.items()}
frac_fv = {}
for k, (ta, tb) in win.items():
    lo, hi = max(ta, T_FV_LO), min(tb, T_FV_HI)
    frac_fv[k] = max(0.0, hi - lo) / (T_FV_HI - T_FV_LO)
    say(f"  {k:45s} drift window [{ta:6.0f}, {tb:6.0f}] us -> FV fraction {frac_fv[k]:.3f}")
R['lifetime_drift_windows_us'] = {k: list(map(float, v)) for k, v in win.items()}
R['lifetime_FV_fractions'] = frac_fv
# sample statistics during a 1 h excursion
rate_sci_h = N_SCI / (LIVE_D * 24)
frac_t800 = (T_FV_HI - 800) / (T_FV_HI - T_FV_LO)
say(f"  science-sample rate {rate_sci_h:.3f}/h; fraction at drift > 800 us {frac_t800:.3f} -> {rate_sci_h*frac_t800:.3f}/h")
say(f"  parent 30-78 keV ERs: {N_par_30_78/(LIVE_D*24):.2f}/h; monitor betas+alphas at drift > 800 us: {rate_mon_h[0]*frac_t800:.0f}/h")
# minimal undetectable excursion: P(no monitor event with drift > 500 us during dt)
frac_t500 = (T_FV_HI - 500) / (T_FV_HI - T_FV_LO)
R_mon_long = rate_mon_h[0] * frac_t500 / 3600.0      # per second
dts = np.array([10, 30, 60, 300, 600, 3600.0])
p_silent = np.exp(-R_mon_long * dts)
for dt, p in zip(dts, p_silent):
    say(f"  excursion of {dt:5.0f} s: P(no monitor event at drift > 500 us) = {p:.2e}")
R.update(rate_sci_per_h=rate_sci_h, frac_drift_gt800=frac_t800, R_monitor_long_drift_per_s=R_mon_long,
         silent_excursion=dict(zip(dts.tolist(), p_silent.tolist())))
# shift of the ER band at the cathode end and the ER median at 859 us
say(f"  under tau_e = {tau_e:.0f} us the ER median at 859 us drops by {-np.log10(f_req):.2f} dex; at 985 us by {T_FV_HI/tau_e/np.log(10):.2f} dex")
# residual correction error if the true lifetime were 2.5-5 ms and calibration assumed infinite
for tl in [2.5, 5.0, 10.0]:
    say(f"  lifetime {tl} ms uncorrected at 859 us: loss {(1-np.exp(-T_EV_US/(tl*1e3)))*100:.1f} %")
R['loss_if_uncorrected_pct'] = {str(tl): float((1 - np.exp(-T_EV_US / (tl * 1e3))) * 100) for tl in [2.5, 5.0, 10.0]}
pd_scan = np.linspace(T_FV_LO, T_FV_HI, 400)
np.savetxt(os.path.join(OUT, 'P041_lifetime_scan.csv'),
           np.c_[pd_scan, np.exp(-pd_scan / tau_e), logS2_ER + np.log10(np.exp(-pd_scan / tau_e))],
           delimiter=',', header='drift_us,survival_tau566us,logS2c_70keV_ER', comments='')

# ----------------------------------------------------------------------------------------------
# 3. Local absorber
# ----------------------------------------------------------------------------------------------
say("\n=== 3. Local absorber ===")
t_s = T_EV_US * 1e-6
sig_T = np.sqrt(2 * D_T_CM2S * t_s); sig_T_rng = [np.sqrt(2 * d * t_s) for d in D_T_RANGE]
sig_L = np.sqrt(2 * D_L_CM2S * t_s); sig_t_us = sig_L / (V_D_MM_US / 10)
A_tpc = np.pi * R_TPC_CM ** 2
A_2s = np.pi * (2 * sig_T) ** 2
say(f"  sigma_T = {sig_T*10:.2f} mm ({sig_T_rng[0]*10:.2f}-{sig_T_rng[1]*10:.2f}); sigma_L = {sig_L*10:.2f} mm -> S2 sigma_t = {sig_t_us:.2f} us (P022: 1.49)")
say(f"  2-sigma footprint {A_2s:.2f} cm2 vs TPC {A_tpc:.0f} cm2 -> {A_2s/A_tpc:.1e}")
R.update(sigma_T_mm=sig_T * 10, sigma_T_range_mm=[s * 10 for s in sig_T_rng], sigma_L_mm=sig_L * 10,
         S2_sigma_t_us=sig_t_us, footprint_2sigma_cm2=A_2s, A_TPC_cm2=A_tpc)

# absorbing disk of radius Rd: loss L(b) = P(Gaussian cloud inside disk) = 1 - Q1(b/s, Rd/s) (Marcum Q)
def loss_disk(b, Rd, s=1.0):
    f = lambda r: (r / s ** 2) * np.exp(-(r ** 2 + b ** 2) / (2 * s ** 2) + r * b / s ** 2) * special.i0e(r * b / s ** 2)
    return integrate.quad(f, 0, Rd, limit=200)[0]

Rd_tuned = np.sqrt(2 * np.log(1 / f_req))    # disk (in sigma units) giving L = L_req at b = 0
say(f"  disk radius giving exactly 78 % loss for a centred cloud: {Rd_tuned:.2f} sigma = {Rd_tuned*sig_T*10:.1f} mm")
rows = []
b_max_extra = 4.0
for Rd in [Rd_tuned, 2.0, 3.0, 5.0, 10.0, 20.0]:
    bmax = Rd + b_max_extra
    bs = np.linspace(0, bmax, 600)
    Ls = np.array([loss_disk(b, Rd) for b in bs])
    fs = 1 - Ls
    w = bs / np.trapezoid(bs, bs)          # dP = 2 pi b db / (pi bmax^2)
    def P(win): return float(np.trapezoid(w * ((fs >= win[0]) & (fs < win[1])), bs))
    pV, pG, pS, pB = P(fV), P(fG), P(fS), P(fB)
    pE = pG + pS + pB
    pN = 1 - pV - pE
    A_hit = np.pi * (bmax * sig_T) ** 2
    rows.append((Rd, Rd * sig_T * 10, A_hit, pV, pG, pS, pB, pE, pN, pV / (pV + pE) if pV + pE > 0 else 0))
    say(f"  Rd = {Rd:5.2f} sigma ({Rd*sig_T*10:5.1f} mm): hit-zone {A_hit:6.2f} cm2; p_V {pV:.3f} p_G {pG:.3f} p_strip {pS:.3f} p_below {pB:.3f} | p_empty {pE:.3f} | p_V/(p_V+p_empty) {pV/(pV+pE) if pV+pE>0 else 0:.3f}")
disk = np.array(rows)
np.savetxt(os.path.join(OUT, 'P041_defect_models.csv'), disk, delimiter=',',
           header='Rd_sigma,Rd_mm,hitzone_cm2,p_V,p_G,p_strip,p_below,p_empty,p_none,pV_over_pVpE', comments='')
R['disk_models'] = disk.tolist()
# required area fraction and population predictions (use N_par_30_78: any ER converted into the NR band
# at S1c 250-600 would have been "the event")
say(f"\n  Population budget with N_par = {N_par_30_78:.0f} ERs at S1c 250-600 (x0.5-2):")
budget = []
for Rd, Rd_mm, A_hit, pV, pG, pS, pB, pE, pN, ratio in rows:
    for targetV in [0.1, 1e-3]:
        eps = targetV / (N_par_30_78 * pV)            # area fraction of TPC covered by hit zones
        n_defects = eps * A_tpc / A_hit
        N_empty = N_par_30_78 * eps * pE
        N_all = N_all_ER_run * eps * (1 - pN)
        budget.append((Rd, targetV, eps, n_defects, N_empty, N_all))
        say(f"  Rd {Rd:5.2f}s target N_V={targetV:g}: area fraction {eps:.2e} ({n_defects:.1f} persistent hit-zones), "
            f"expected empty-region ERs {N_empty:.2f}, all-energy affected ERs {N_all:.0f}")
    # maximum-likelihood scenario given the observation (1 in V, 0 in empty): eps* = 1/(N_par (pV+pE))
    eps_star = 1 / (N_par_30_78 * (pV + pE))
    Pobs = ratio * np.exp(-1)
    say(f"     ML scenario: eps* = {eps_star:.2e}; P(1 in V, 0 in empty) = {Pobs:.3f}; all-energy affected {N_all_ER_run*eps_star*(1-pN):.0f}")
    budget.append((Rd, -1, eps_star, eps_star * A_tpc / A_hit, N_par_30_78 * eps_star * pE, N_all_ER_run * eps_star * (1 - pN)))
budget = np.array(budget)
np.savetxt(os.path.join(OUT, 'P041_defect_budget.csv'), budget, delimiter=',',
           header='Rd_sigma,target_NV(-1=ML),area_fraction,n_hitzones,N_empty_expected,N_allenergy_affected', comments='')
R['defect_budget'] = budget.tolist()
# flat-loss reference
pV_f, pE_f = flat['V (NR band +-2sigma, 1 event)'], flat['gap G (in ROI, above NR+2sigma, empty)'] + flat['strip (out of ROI, empty)'] + flat['below NR-2sigma (to S2 threshold, empty)']
say(f"  flat-in-loss: p_V {pV_f:.3f}, p_empty {pE_f:.3f}, p_V/(p_V+p_E) {pV_f/(pV_f+pE_f):.3f}; given 1 in V, ML expects {pE_f/pV_f:.1f} in empty regions, P(0) = {np.exp(-pE_f/pV_f):.3f}")
R['flat_pV'] = pV_f; R['flat_pE'] = pE_f; R['flat_P0_given_1V'] = float(np.exp(-pE_f / pV_f))

# persistent absorber vs the 83mKr map
kr_per_cm2 = np.array(KR_EVENTS_PER_INJ) / A_tpc
z_hole = L_req / (S2_RES_KR / np.sqrt(kr_per_cm2))
say(f"  83mKr per cm2 per injection {kr_per_cm2[0]:.0f}-{kr_per_cm2[1]:.0f} -> a 78 % deficit in a 1 cm2 cell is a {z_hole[0]:.0f}-{z_hole[1]:.0f} sigma hole")
R['kr_per_cm2'] = kr_per_cm2.tolist(); R['kr_hole_sigma'] = z_hole.tolist()
# impurity plume: local lifetime needed over a path length ell
for ell in [0.3, 1.0, 3.0]:
    tau_loc = ell / (V_D_MM_US / 10) / (-np.log(f_req))
    say(f"  plume of length {ell} cm: local lifetime {tau_loc:.1f} us (bulk >~ 5000 us) -> enrichment x{5000/tau_loc:.0f}")
R['plume_local_lifetime_us'] = {str(e): float(e / (V_D_MM_US / 10) / (-np.log(f_req))) for e in [0.3, 1.0, 3.0]}

# ----------------------------------------------------------------------------------------------
# 4. Extraction dip and pulse pathologies
# ----------------------------------------------------------------------------------------------
say("\n=== 4. Extraction / pulse pathologies ===")
eps_loc = EPS_EXT * f_req
say(f"  local extraction efficiency needed {eps_loc:.3f} (bulk {EPS_EXT}); required S2 gain deficit x{1/f_req:.2f} = {(1/f_req-1)*100:.0f} % vs map amplitude <~ {MAP_VAR*100:.0f} % -> {(1/f_req-1)/MAP_VAR:.0f}x typical")
R['eps_ext_local'] = eps_loc; R['gain_deficit_over_map'] = (1 / f_req - 1) / MAP_VAR
# g2 uncertainty and NR width as reference scales (fractional)
say(f"  g2 uncertainty {g2_err/g2*100:.1f} %; NR-band sigma {(10**SIG_NR_DEX-1)*100:.1f} %")
# truncation: cut a Gaussian S2 so that the surviving area is f_req -> width/asymmetry of the remnant
zc = stats.norm.ppf(f_req)                       # keep the part below z_c (leading edge kept)
xs = np.linspace(-6, zc, 4000); pdf = stats.norm.pdf(xs)
m = np.trapezoid(xs * pdf, xs) / np.trapezoid(pdf, xs)
sd = np.sqrt(np.trapezoid((xs - m) ** 2 * pdf, xs) / np.trapezoid(pdf, xs))
sk = np.trapezoid((xs - m) ** 3 * pdf, xs) / np.trapezoid(pdf, xs) / sd ** 3
say(f"  truncated S2 keeping {f_req:.2f} of area: remnant sigma = {sd:.2f} of the true sigma, skew {sk:.2f} (P022 width tolerance 10-20 %)")
R['truncation_remnant_sigma_ratio'] = sd; R['truncation_skew'] = sk
# clipping: the alpha template. 5.5 MeV alpha: Nq ~ E/W, Ne with recombination r_alpha ~ 0.9-0.95 (recalled)
Ne_alpha = 5.5e6 / 13.7 * np.array([0.05, 0.10])
say(f"  alpha S2 (template class): {Ne_alpha[0]:.0f}-{Ne_alpha[1]:.0f} electrons = {Ne_alpha[0]/Ne_obs:.0f}-{Ne_alpha[1]/Ne_obs:.0f} x the event's S2; hypothetical true ER S2 {NE_ER_P010*g2:.0f} phd ~ 83mKr class ({lz.nest_er_yields(41.5)[1]*g2:.0f} phd)")
R['alpha_S2_over_event'] = (Ne_alpha / Ne_obs).tolist(); R['S2_ER_true_phd'] = NE_ER_P010 * g2
R['S2_Kr83m_phd'] = float(lz.nest_er_yields(41.5)[1] * g2)

# ----------------------------------------------------------------------------------------------
# 5. Bayesian channel summary (judgement factors, explicit)
# ----------------------------------------------------------------------------------------------
say("\n=== 5. Channel summary ===")
# dossier prior for class F (artefact) = 0.18, split by judgement; survival factors S = P(checks pass & loss in V | channel)
pV_disk_mid = float(np.median(disk[:, 9]))     # p_V/(p_V+p_E) across disk models
# tau-window coincidence for a bulk excursion of random size: tau must fall in [t/ln(1/f_lo), t/ln(1/f_hi)]
tau_win = (T_EV_US / np.log(1 / fV[0]), T_EV_US / np.log(1 / fV[1]))
p_tau_win = np.log10(tau_win[1] / tau_win[0]) / 2.0          # log-uniform prior over 0.1-10 ms (2 decades)
say(f"  bulk excursion: tau must lie in [{tau_win[0]:.0f}, {tau_win[1]:.0f}] us -> window factor {p_tau_win:.3f}; "
    f"P(silent 300 s) = {p_silent[3]:.1e} (60 s: {p_silent[2]:.2f})")
R['tau_window_us'] = list(map(float, tau_win)); R['p_tau_window'] = float(p_tau_win)
chan = [
    # name, prior, factor, rationale
    ('bulk electron-lifetime excursion', 0.03, p_silent[3] * p_tau_win,
     'a bulk purity transient shorter than ~5 min is unphysical (circulation time-scales); P(no other long-drift event in 300 s) x tau-window coincidence'),
    ('local transient absorber (bubble/particulate/plume)', 0.05, pV_disk_mid * np.exp(-1) * 0.3,
     'loss must land in V (p_V/(p_V+p_E) ~ %.2f) x e^-1 population factor x 0.3 transient-only (Kr map excludes persistent)' % pV_disk_mid),
    ('field distortion / charge trapping (PTFE, floating conductor)', 0.03, 0.02,
     'static -> 83mKr (x,y,z) map and position-dependent band; only a transient localised region survives (as above)'),
    ('extraction / gas-gap dip', 0.02, 0.02,
     'x4.6 gain deficit = 23x map amplitude; static -> map; transient bubble/wave -> S2 shape'),
    ('S2 saturation / clipping / truncated window', 0.02, 1e-3,
     'S2 is 100-200x below the alpha template class; remnant width 0.4 sigma vs 10-20 % tolerance'),
    ('S1/S2 mis-pairing (accidental)', 0.02, 0.05,
     'P022: x708 mismodelling excluded at 4.3 sigma; cathode-emission loophole open'),
    ('unknown pathology', 0.01, 0.3, 'not testable with public information'),
]
tab = []
for name, pri, fac, why in chan:
    tab.append((name, pri, fac, pri * fac, why))
    say(f"  {name:60s} prior {pri:.2f} x factor {fac:.2e} = {pri*fac:.2e}")
tot = sum(t[3] for t in tab)
say(f"  total artefact residual mass {tot:.3f} (prior 0.18 -> shrink x{0.18/tot:.0f})")
R['channels'] = [dict(name=t[0], prior=t[1], factor=float(t[2]), residual=float(t[3]), rationale=t[4]) for t in tab]
R['artefact_residual_total'] = tot
with open(os.path.join(OUT, 'P041_bayes_table.csv'), 'w') as fh:
    fh.write('channel,prior,factor,residual,rationale\n')
    for t in tab:
        fh.write(f'"{t[0]}",{t[1]},{t[2]:.3e},{t[3]:.3e},"{t[4]}"\n')

# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------
# Fig 1: required loss vs plausible magnitude of each mechanism (fractional S2 deficit)
mech = [('required\n(ER hypothesis)', L_req * 100, L_req * 100),
        ('lifetime corr.\nerror (tau 2.5-10 ms)', R['loss_if_uncorrected_pct']['10.0'], R['loss_if_uncorrected_pct']['2.5']),
        ('(x,y) map\namplitude', 5, MAP_VAR * 100),
        ('g2\nuncertainty', g2_err / g2 * 100, g2_err / g2 * 100),
        ('NR band\nwidth (1 sigma)', (10 ** SIG_NR_DEX - 1) * 100, (10 ** SIG_NR_DEX - 1) * 100),
        ('S2 width\ntolerance', 10, 20)]
fig, ax = plt.subplots(figsize=(7.2, 4.0))
names = [m[0] for m in mech]; lo = np.array([m[1] for m in mech]); hi = np.array([m[2] for m in mech])
cols_ = ['#b2182b'] + ['#4393c3'] * 5
ax.bar(range(len(mech)), hi, color=cols_, alpha=0.35, width=0.6)
ax.bar(range(len(mech)), lo, color=cols_, width=0.6)
ax.axhline(L_req * 100, color='#b2182b', ls='--', lw=1)
ax.set_xticks(range(len(mech))); ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel('S2 deficit [%]'); ax.set_yscale('log'); ax.set_ylim(1, 200)
ax.set_title('Charge loss required (78 %) vs. plausible size of each effect (dark = low, light = high bracket)', fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P041_required_loss_vs_mechanism.png'), dpi=160); plt.close(fig)

# Fig 2: drift-time distribution + lifetime mechanism
fig, axs = plt.subplots(1, 2, figsize=(10, 4.0))
ax = axs[0]
ax.fill_between([T_FV_LO, T_FV_HI], 0, 1, color='#dddddd', step=None, label='FV (uniform in drift time)')
ax.axvline(T_EV_US, color='k', lw=2, label=f'event {T_EV_US:.0f} us')
ax.axvline(T_MAX_US, color='grey', ls=':', label=f'cathode {T_MAX_US:.0f} us')
ax.axvline(56, color='brown', ls=':', label='gas events ~56 us (Fig. 3)')
tt = np.linspace(0, T_MAX_US, 300)
ax.plot(tt, 1 - np.exp(-tt / tau_e), color='#b2182b', label=f'loss 1-exp(-t/tau), tau = {tau_e:.0f} us')
ax.plot(tt, 1 - np.exp(-tt / 5000), color='#4393c3', label='tau = 5 ms')
ax.set_xlabel('drift time [us]'); ax.set_ylabel('fraction of electrons lost'); ax.set_ylim(0, 1); ax.legend(fontsize=7, loc='upper left')
ax = axs[1]
colors = {'ER cloud (f > envelope)': '#bbbbbb', 'strip (out of ROI, empty)': '#fddbc7',
          'gap G (in ROI, above NR+2sigma, empty)': '#f4a582', 'V (NR band +-2sigma, 1 event)': '#2166ac',
          'below NR-2sigma (to S2 threshold, empty)': '#d6604d'}
for k, (a, b) in regions.items():
    ax.axhspan(a, min(b, 4.9), color=colors[k], alpha=0.5, label=k.split(' (')[0])
ax.plot(pd_scan, logS2_ER + np.log10(np.exp(-pd_scan / tau_e)), color='#b2182b', lw=2, label='70 keV ER, tau_e = 566 us')
ax.axhline(logS2_ER, color='k', ls='--', lw=1, label='ER mean (no loss)')
ax.plot([T_EV_US], [np.log10(S2c)], 'k*', ms=12, label='event')
ax.set_xlabel('drift time [us]'); ax.set_ylabel('log10 S2c at S1c = 540 phd'); ax.set_ylim(2.8, 4.9)
ax.legend(fontsize=6.5, loc='lower left')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P041_drift_time_and_lifetime.png'), dpi=160); plt.close(fig)

# Fig 3: loss distribution for absorbing disks
fig, ax = plt.subplots(figsize=(6.4, 4.0))
for Rd, c in zip([Rd_tuned, 3.0, 10.0], ['#2166ac', '#4393c3', '#92c5de']):
    bmax = Rd + b_max_extra; bs = np.linspace(0, bmax, 400)
    fs = 1 - np.array([loss_disk(b, Rd) for b in bs])
    w = bs / np.trapezoid(bs, bs)
    h, edges = np.histogram(fs, bins=np.linspace(0, 1, 41), weights=w)
    ax.step(edges[:-1], h / np.diff(edges), where='post', color=c, label=f'disk R = {Rd:.1f} sigma ({Rd*sig_T*10:.0f} mm)')
for k, (a, b) in reg_f.items():
    ax.axvspan(a, b, color=colors[k], alpha=0.35)
ax.set_xlabel('surviving electron fraction f'); ax.set_ylabel('probability density (per hit)')
ax.set_title('Loss distribution for a Gaussian cloud hitting an absorbing disk; shaded = landing regions', fontsize=9)
ax.legend(fontsize=8); ax.set_yscale('log'); ax.set_ylim(1e-2, 30)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P041_disk_loss_distribution.png'), dpi=160); plt.close(fig)

with open(os.path.join(OUT, 'P041_results.json'), 'w') as fh:
    json.dump(R, fh, indent=1, default=float)
with open(os.path.join(OUT, 'run_log.txt'), 'w') as fh:
    fh.write('\n'.join(log_lines) + '\n')
say("\nwritten:", OUT)
