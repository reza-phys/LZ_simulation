#!/usr/bin/env python
"""
P038 -- The efficiency edge: how the S1c < 600 phd boundary shapes inference at 250-270 keV and what an
extended ROI would have shown.

Parts
  1. Efficiency model: nestpy (LZ_WS2024, Table S5 yields + p(E) break) GetQuanta fluctuations on the paper-contour
     energy scale (P009: Nq = 11.32 E^1.112), binomial S1 detection (g1 = 0.110), SPE resolution, 2 % position
     residual; S2 with extraction 0.726, SE Fano 4, g2 = 34.5.  eps_X(E) = 0.955 * P(S1c < X and log10 S2c < 4.15)
     for X = 600, 700, 800, 1000, 1200 phd; 50 % points; validation against Fig. S2 (269.9 keV).
  2. Signal acceptance vs delta: WimPyDD O1 isoscalar inelastic spectra (1000 GeV, LZ unit coupling, Baxter halo,
     annual average and 16 June) over the FULL kinematic window (E_true to 800 keV); accepted fraction per cut;
     fraction above 270 keV; fitted coupling per event (1/S_acc); single-event profile Z(delta) per cut using the
     P021 formula q0 = 2[ln(f~/b) - 1 + b/f~], f~ = accepted density at E_obs per accepted event, b = 5.7e-4/70 keV^-1.
  3. Backgrounds in an extended ROI: MSSI interpolation between the WS ROI (3-600 phd; wall 0.0048, RFR 0.0001) and
     the HE SB (800-1700 phd; wall 0.1, RFR 0.5) for the 4.7 t FV (Table 'MSSI comparison'), exponential and power-law
     S1c densities; NR-band (+-2 sigma) fraction from Fig. S4 readings; signal-to-background of extending the edge.
  4. The E_true = 262 keV scenario: 1/eps, delta_max, P(S1c > 600), P(detected event within 25 keV of the 50 % point).
  5. Recommendation: S1c edge at which the NR-band MSSI reaches given levels; where the S2c < 10^4.15 cut bites.

Run from the simulation root:  .venv/bin/python output/code/P038_efficiency_edge.py
Outputs: output/work/P038/*.csv, *.json, figures/*.png ; WimPyDD kernels cached in output/work/P038/spectra_s_1000_full.npz
"""
import sys, os, json, math, time
import numpy as np
import pandas as pd
sys.path.insert(0, 'output/code')
from common import lzcommon as lz
from scipy import stats, special, optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nestpy

T0 = time.time()
OUT = 'output/work/P038'; FIG = OUT + '/figures'
os.makedirs(FIG, exist_ok=True)
LOG = open(f'{OUT}/run_log.txt', 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n'); LOG.flush()
say(f'===== P038 run {time.strftime("%Y-%m-%d %H:%M:%S")} =====')

rng = np.random.default_rng(38)
RES = {}

# ------------------------------------------------------------------------------------------------------
# 0. constants
# ------------------------------------------------------------------------------------------------------
G1, G2 = lz.LZ['g1'], lz.LZ['g2']                    # 0.110 phd/photon, 34.5 phd/electron (paper)
EXPO = lz.LZ['exposure_tyr']                         # 2.84 t yr
LOGS2_MAX = lz.LZ['log10S2c_max']                    # 4.15
S1_CUTS = [600, 700, 800, 1000, 1200]
PLATEAU = 0.955                                      # Fig. S2 green level (paper: 96 % average 14-250 keV)
E_EV = 248.0                                         # paper energy of the event
B_H, W_H = 5.7e-4, 70.0                              # P016/P021: NR-band background flat in 200-270 keV
BD = B_H / W_H                                       # background density per keV
MV, MN = lz.M_V_GEV, lz.M_NUCLEON_GEV
HBARC2_CM2 = 0.3894e-27
P009 = json.load(open('output/work/P009/results.json'))
ALPHA_P, BETA_P = P009['digitised']['alpha_paper'], P009['digitised']['beta_paper']   # Nq = 11.32 E^1.112
WIDTH_LZ = P009['inputs']['width_parameters']
EXT_EFF = P009['inputs']['ext_eff']                  # 0.726 (nestpy LZ_WS2024 CalculateG2)
POS_RES = P009['inputs']['pos_res']                  # 0.02
DET = nestpy.detectors.LZ_WS2024(); NC = nestpy.NESTcalc(DET)
FIELD, DENSITY = lz.DRIFT_FIELD_VCM, 2.9
S2_FANO, SPE_RES = DET.get_s2Fano(), DET.get_sPEres()
say(f'inputs: alpha_paper={ALPHA_P:.4f} beta_paper={BETA_P:.4f} ext_eff={EXT_EFF} s2Fano={S2_FANO} sPEres={SPE_RES}')

# ------------------------------------------------------------------------------------------------------
# 1. NEST Monte-Carlo efficiency for several S1c cuts
# ------------------------------------------------------------------------------------------------------
def quanta(E, n, variant='paper'):
    """(Nph, Ne) samples: nestpy GetQuanta with the LZ width vector around the Table S5 (+break) yields;
    'paper' shifts the photon mean so that Nq = alpha_p E^beta_p (P009 paper-contour scale)."""
    y = NC.GetYields(nestpy.interactions.NR, float(E), DENSITY, FIELD, 131.293, 54,
                     lz.nest_nr_params_vector(E, lz.NEST_NR_LZ))
    out = np.empty((n, 2))
    for i in range(n):
        q = NC.GetQuanta(y, DENSITY, WIDTH_LZ)
        out[i] = q.photons, q.electrons
    if variant == 'paper':
        out[:, 0] += (ALPHA_P * E ** BETA_P - y.ElectronYield) - y.PhotonYield
    return out

def detect(q):
    nph = np.clip(np.round(q[:, 0]).astype(int), 0, None)
    ne = np.clip(np.round(q[:, 1]).astype(int), 0, None)
    n1 = rng.binomial(nph, G1).astype(float)
    n1 = n1 * (1 + rng.normal(0, SPE_RES / np.sqrt(np.clip(n1, 1, None))))
    s1c = n1 * (1 + rng.normal(0, POS_RES, n1.size))
    nx = rng.binomial(ne, EXT_EFF).astype(float)
    s2 = nx * (G2 / EXT_EFF) * (1 + rng.normal(0, np.sqrt(S2_FANO / np.clip(nx * (G2 / EXT_EFF), 1, None))))
    s2 = s2 * (1 + rng.normal(0, POS_RES, s2.size))
    return s1c, np.log10(np.clip(s2, 1, None))

E_MC = np.arange(200.0, 604.0, 4.0)
N_MC = 6000
rows = []
for E in E_MC:
    s1c, l2 = detect(quanta(E, N_MC))
    r = dict(E_keV=E, S1c_median=float(np.median(s1c)), S1c_sd=float(s1c.std()), log10S2c_median=float(np.median(l2)),
             log10S2c_sd=float(l2.std()), P_S2c_pass=float((l2 < LOGS2_MAX).mean()), P_S1c_gt_600=float((s1c > 600).mean()))
    for X in S1_CUTS:
        r[f'P_pass_{X}'] = float(((s1c < X) & (l2 < LOGS2_MAX)).mean())
        r[f'P_S1_{X}'] = float((s1c < X).mean())
    rows.append(r)
MC = pd.DataFrame(rows)
MC.to_csv(f'{OUT}/P038_mc_efficiency.csv', index=False)
say(f'MC efficiency scan done ({len(E_MC)} energies x {N_MC}), t={time.time() - T0:.0f}s')

def eps_of_E(X):
    """Efficiency vs true NR energy for S1c < X (and log10 S2c < 4.15): plateau x low-edge erf below 200 keV,
    MC above; zero beyond the MC range (P_pass is ~0 there for every cut considered)."""
    Pp = MC[f'P_pass_{X}'].values
    def f(E):
        E = np.atleast_1d(np.asarray(E, float))
        lo = 0.5 * (1 + special.erf((E - 5.4) / (math.sqrt(2) * 2.5)))
        hi = np.interp(E, MC.E_keV.values, Pp, left=Pp[0], right=0.0)
        return PLATEAU * lo * hi
    return f
EPS = {X: eps_of_E(X) for X in S1_CUTS}
EPS['none'] = lambda E: PLATEAU * 0.5 * (1 + special.erf((np.asarray(E, float) - 5.4) / (math.sqrt(2) * 2.5)))

def e50(X):
    e = EPS[X](MC.E_keV.values)
    i = np.where(e < 0.5)[0][0]
    return float(np.interp(0.5, [e[i], e[i - 1]], [MC.E_keV.values[i], MC.E_keV.values[i - 1]]))
def erf_sigma(X):
    e = EPS[X](MC.E_keV.values); E50 = e50(X)
    m = (e > 0.02) & (e < PLATEAU - 0.02)
    s, _ = optimize.curve_fit(lambda E, s: PLATEAU * stats.norm.cdf((E50 - E) / s), MC.E_keV.values[m], e[m], p0=[12.0])
    return float(s[0])
EDGE = {}
for X in S1_CUTS:
    E50 = e50(X)
    # S1c-only 50 % point (no S2c cut) and the energy where the S2c cut alone removes 10 % / 50 %
    e_s1 = PLATEAU * MC[f'P_S1_{X}'].values
    i = np.where(e_s1 < 0.5)[0]
    E50_s1 = float(np.interp(0.5, [e_s1[i[0]], e_s1[i[0] - 1]], [MC.E_keV.values[i[0]], MC.E_keV.values[i[0] - 1]])) if len(i) else float('nan')
    EDGE[X] = dict(E50_keV=E50, E50_S1only_keV=E50_s1, sigma_erf_keV=erf_sigma(X),
                   S1c_median_at_E50=float(np.interp(E50, MC.E_keV, MC.S1c_median)))
ps2 = MC.P_S2c_pass.values
E_s2_90 = float(np.interp(0.9, ps2[::-1], MC.E_keV.values[::-1])); E_s2_50 = float(np.interp(0.5, ps2[::-1], MC.E_keV.values[::-1]))
RES['s2c_cut_bite'] = dict(E_P90_keV=E_s2_90, E_P50_keV=E_s2_50, S1c_at_P90=float(np.interp(E_s2_90, MC.E_keV, MC.S1c_median)),
                           S1c_at_P50=float(np.interp(E_s2_50, MC.E_keV, MC.S1c_median)))
RES['edges'] = EDGE
# g1 sensitivity of the edge: S1c is proportional to g1, so dE50 = E50 * (dg1/g1) / (d ln S1c / d ln E)
i1, i2 = np.searchsorted(MC.E_keV.values, 248.0), np.searchsorted(MC.E_keV.values, 300.0)
slope = float((np.log(MC.S1c_median.values[i2]) - np.log(MC.S1c_median.values[i1])) / (np.log(300.0) - np.log(248.0)))
RES['g1_sensitivity'] = dict(dlnS1c_dlnE_248_300=slope, dE50_per_2pct_g1_keV=float(EDGE[600]['E50_keV'] * 0.02 / slope),
                             dE50_per_2pct_g1_keV_1000=float(EDGE[1000]['E50_keV'] * 0.02 / slope))
say('d ln S1c / d ln E (248-300 keV) = %.3f; g1 +-2 %% moves E50(600) by -/+ %.1f keV' % (slope, RES['g1_sensitivity']['dE50_per_2pct_g1_keV']))
say('50 % points per S1c cut (with S2c<10^4.15):', json.dumps({k: {kk: round(vv, 1) for kk, vv in v.items()} for k, v in EDGE.items()}))
say('S2c<10^4.15 alone removes 10 %/50 % of NRs at E =', round(E_s2_90, 1), '/', round(E_s2_50, 1), 'keV; S1c median there',
    round(RES['s2c_cut_bite']['S1c_at_P90']), '/', round(RES['s2c_cut_bite']['S1c_at_P50']), 'phd')

# Fig. S2 inset readings (P009, by eye, +-0.03) for validation
INSET = np.array([[250, 0.93], [255, 0.88], [260, 0.79], [265, 0.66], [270, 0.50], [275, 0.35], [280, 0.22],
                  [285, 0.12], [290, 0.05], [295, 0.02], [300, 0.005]])
res600 = EPS[600](INSET[:, 0]) - INSET[:, 1]
RES['figS2_validation'] = dict(E50_MC=EDGE[600]['E50_keV'], E50_paper=269.9, max_abs_resid=float(np.abs(res600).max()),
                               rms_resid=float(np.sqrt((res600 ** 2).mean())), P009_E50_MC=271.7)
say('Fig. S2 validation: E50(600) =', round(EDGE[600]['E50_keV'], 1), 'keV; residuals vs inset readings max', round(float(np.abs(res600).max()), 3))

# high-statistics points at the event energies
PT = {}
for E in (246.0, 248.0, 262.0, 265.0, 269.9):
    s1c, l2 = detect(quanta(E, 40000))
    PT[str(E)] = dict(P_S1c_gt_600=float((s1c > 600).mean()), eps_600=float(PLATEAU * ((s1c < 600) & (l2 < LOGS2_MAX)).mean()),
                      S1c_median=float(np.median(s1c)), S1c_sd=float(s1c.std()),
                      **{f'eps_{X}': float(PLATEAU * ((s1c < X) & (l2 < LOGS2_MAX)).mean()) for X in S1_CUTS[1:]})
    PT[str(E)]['one_over_eps_600'] = 1.0 / PT[str(E)]['eps_600']
RES['event_energies'] = PT
say('event-energy points:', json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in PT.items()}))
# Table-S5-as-printed scale, for the record (P009: E50 = 291 keV)
s1c, l2 = detect(quanta(270.0, 20000, variant='tab')); s1b, l2b = detect(quanta(290.0, 20000, variant='tab'))
RES['tableS5_scale_check'] = dict(eps600_at_270=float(PLATEAU * (s1c < 600).mean()), eps600_at_290=float(PLATEAU * (s1b < 600).mean()))
say('Table S5 scale check: eps600(270)=%.3f eps600(290)=%.3f (paper scale would be ~0.5 at 270)' % tuple(RES['tableS5_scale_check'].values()))

# ------------------------------------------------------------------------------------------------------
# 2. WimPyDD O1 inelastic spectra over the full kinematic window (1000 GeV)
# ------------------------------------------------------------------------------------------------------
M_CHI = 1000.0
VGRID = np.linspace(0.0, 830.0, 1661); ONES = np.ones_like(VGRID)
E_T = np.arange(1.5, 800.0, 3.0); DE_T = 3.0
E_O = np.arange(0.5, 800.0, 1.0)
DELTAS = np.array([250., 275., 300., 310., 320., 330., 340., 350., 355., 360., 365., 370., 375., 380., 385., 390.])
DOY_JUNE = 167
days12 = 15.0 + 365.25 / 12 * np.arange(12)
SPEC_PATH = f'{OUT}/spectra_s_1000_full.npz'
if os.path.exists(SPEC_PATH):
    SP = dict(np.load(SPEC_PATH))
else:
    WD = lz.wd()
    HALOS = {'annual': np.mean([lz.wd_halo(day_of_year=d, vmin=VGRID)[1] for d in days12], axis=0),
             'june': lz.wd_halo(day_of_year=DOY_JUNE, vmin=VGRID)[1]}
    ham = lz.wd_hamiltonian('O1_s', {1: lz.wd_c_from_anand(1.0 / MV ** 2, 0.0)})     # (c_1^s m_v^2)^2 = 1
    R = {h: np.zeros((len(DELTAS), len(E_T))) for h in HALOS}
    ncalls = 0
    for i, d in enumerate(DELTAS):
        los, his = zip(*[lz.E_R_range_keV(M_CHI, VGRID[-1], A=A, delta_kev=float(d)) for A in (124.0, 136.0)])
        lo, hi = min(los), max(his)
        for j in np.where((E_T >= lo - DE_T) & (E_T <= hi + DE_T))[0]:
            K = WD.diff_rate(WD.Xe, ham, M_CHI, float(E_T[j]), VGRID, ONES, j_chi=0.5, delta=float(d),
                             sum_over_streams=False) * 1000.0 * 365.25
            ncalls += 1
            for h, deta in HALOS.items():
                R[h][i, j] = max(0.0, float(np.asarray(K) @ deta))
    np.savez(SPEC_PATH, deltas=DELTAS, E=E_T, **R)
    SP = dict(np.load(SPEC_PATH))
    say(f'WimPyDD spectra: {ncalls} kernel calls, t={time.time() - T0:.0f}s')
# cross-check against the P021 cache (same code path, E <= 337.5 keV)
P21 = dict(np.load('output/work/P021/spectra_s_1000.npz'))
i21 = np.where(P21['deltas'] == 300.0)[0][0]; i38 = np.where(SP['deltas'] == 300.0)[0][0]
n21 = len(P21['E']); m = (P21['E'] >= 100) & (P21['E'] <= 330)
assert np.allclose(P21['E'], E_T[:n21])
ratio = SP['annual'][i38][:n21][m] / np.where(P21['annual'][i21][m] > 0, P21['annual'][i21][m], np.nan)
RES['p021_spectrum_check'] = dict(delta=300, ratio_min=float(np.nanmin(ratio)), ratio_max=float(np.nanmax(ratio)))
say('spectrum check vs P021 cache (delta=300, 100-330 keV): ratio range %.4f-%.4f' % (np.nanmin(ratio), np.nanmax(ratio)))

# ------------------------------------------------------------------------------------------------------
# 3. acceptance, coupling and single-event Z per cut
# ------------------------------------------------------------------------------------------------------
SIG_E = 11.0 * np.sqrt(E_T / 248.0)                       # P009/P021 resolution
SMEAR = stats.norm.pdf((E_O[:, None] - E_T[None, :]) / SIG_E[None, :]) / SIG_E[None, :] * DE_T

def q0_single(ftil, bd=BD):
    return 2 * (math.log(ftil / bd) - 1 + bd / ftil) if ftil > bd else 0.0

acc_rows = []
for h in ('annual', 'june'):
    for i, d in enumerate(SP['deltas']):
        r = SP[h][i]
        S_full = float(r.sum() * DE_T * EXPO)
        if S_full <= 0:
            continue
        cum = np.cumsum(r) * DE_T * EXPO
        row = dict(halo=h, delta_keV=float(d), S_full_unit=S_full,
                   E_median_true=float(np.interp(0.5 * S_full, cum, E_T)), E_peak_true=float(E_T[np.argmax(r)]),
                   frac_true_gt_270=float(r[E_T > 270].sum() * DE_T * EXPO / S_full),
                   frac_true_gt_300=float(r[E_T > 300].sum() * DE_T * EXPO / S_full),
                   frac_true_gt_400=float(r[E_T > 400].sum() * DE_T * EXPO / S_full))
        for X in S1_CUTS + ['none']:
            w = r * EPS[X](E_T) * EXPO
            S_acc = float(w.sum() * DE_T)
            dobs = SMEAR @ w
            row[f'S_acc_{X}'] = S_acc
            row[f'A_{X}'] = S_acc / S_full
            row[f'kappa_hat_{X}'] = 1.0 / S_acc if S_acc > 0 else float('nan')
            for Eo, tag in ((248.0, ''), (262.0, '_E262')):
                f_ev = float((stats.norm.pdf((Eo - E_T) / SIG_E) / SIG_E * DE_T) @ w)
                ftil = f_ev / S_acc if S_acc > 0 else 0.0
                q0 = q0_single(ftil) if ftil > 0 else 0.0
                row[f'ftil_{X}{tag}'] = ftil; row[f'q0_{X}{tag}'] = q0; row[f'Z_{X}{tag}'] = math.sqrt(q0)
                if tag == '':
                    row[f'pct_obs_{X}'] = float(dobs[E_O < Eo].sum() / dobs.sum()) if dobs.sum() > 0 else float('nan')
            row[f'P_obs_245_295_{X}'] = float(dobs[(E_O > 245) & (E_O < 295)].sum() / dobs.sum()) if dobs.sum() > 0 else float('nan')
            row[f'P_obs_gt_270_{X}'] = float(dobs[E_O > 270].sum() / dobs.sum()) if dobs.sum() > 0 else float('nan')
        row['gain_1000_over_600'] = row['S_acc_1000'] / row['S_acc_600']
        row['gain_800_over_600'] = row['S_acc_800'] / row['S_acc_600']
        row['extra_events_600_1000_per_ROI_event'] = row['gain_1000_over_600'] - 1
        acc_rows.append(row)
ACC = pd.DataFrame(acc_rows)
ACC.to_csv(f'{OUT}/P038_acceptance_vs_delta.csv', index=False)
A = ACC[ACC.halo == 'annual'].set_index('delta_keV')
say('\nannual halo, 1000 GeV O1s: acceptance A(delta; cut) and single-event Z')
say(A[['A_600', 'A_800', 'A_1000', 'A_1200', 'frac_true_gt_270', 'E_median_true', 'Z_600', 'Z_800', 'Z_1000', 'Z_1200', 'gain_1000_over_600']].round(4).to_string())
say('\nJune halo:')
say(ACC[ACC.halo == 'june'].set_index('delta_keV')[['A_600', 'A_1000', 'frac_true_gt_270', 'Z_600', 'Z_1000']].round(4).to_string())

# comparison with P021 baseline (their erf efficiency, E_T <= 337 keV, low-energy bins included)
B21 = pd.read_csv('output/work/P021/P021_scan_baseline.csv')
B21 = B21[(B21.tag == 's') & (B21.m_GeV == 1000)].set_index('delta_keV')
cmp_rows = []
for d in A.index:
    if d in B21.index:
        cmp_rows.append(dict(delta_keV=d, Z_P021=float(B21.loc[d, 'Z']), Z_600_here=float(A.loc[d, 'Z_600']),
                             S_tot_P021=float(B21.loc[d, 'S_tot_unit']), S_acc_600_here=float(A.loc[d, 'S_acc_600']),
                             kappa_P021=float(B21.loc[d, 'kappa_hat']), kappa_600_here=float(A.loc[d, 'kappa_hat_600'])))
CMP = pd.DataFrame(cmp_rows); CMP.to_csv(f'{OUT}/P038_vs_P021.csv', index=False)
say('\ncomparison with P021 (Z and S_tot at the 600 phd cut):'); say(CMP.round(4).to_string(index=False))
RES['vs_P021'] = dict(max_abs_dZ=float((CMP.Z_P021 - CMP.Z_600_here).abs().max()),
                      S_ratio_range=[float((CMP.S_acc_600_here / CMP.S_tot_P021).min()), float((CMP.S_acc_600_here / CMP.S_tot_P021).max())])

# peak of Z(delta) per cut, and the delta-preference (Delta q0 peak vs 300 and 350)
PK = {}
for X in S1_CUTS + ['none']:
    z = A[f'Z_{X}']; dpk = float(z.idxmax())
    PK[str(X)] = dict(delta_peak=dpk, Z_peak=float(z.max()), Z_300=float(A.loc[300.0, f'Z_{X}']), Z_350=float(A.loc[350.0, f'Z_{X}']),
                      Z_380=float(A.loc[380.0, f'Z_{X}']), dq0_380_minus_300=float(A.loc[380.0, f'q0_{X}'] - A.loc[300.0, f'q0_{X}']),
                      dq0_peak_minus_300=float(A.loc[dpk, f'q0_{X}'] - A.loc[300.0, f'q0_{X}']),
                      LR_380_over_300=float(math.exp(0.5 * (A.loc[380.0, f'q0_{X}'] - A.loc[300.0, f'q0_{X}']))))
RES['Z_peaks'] = PK
say('\nZ(delta) peaks per cut:', json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in PK.items()}))

# ------------------------------------------------------------------------------------------------------
# 4. backgrounds in an extended ROI (4.7 t FV): MSSI interpolation
# ------------------------------------------------------------------------------------------------------
MSSI = dict(wall=dict(ROI=0.0048, HESB=0.1), RFR=dict(ROI=0.0001, HESB=0.5))    # Table 'MSSI comparison', science, 4.7 t
ROI_LO, ROI_HI, SB_LO, SB_HI = 3.0, 600.0, 800.0, 1700.0

def solve_exp(N1, N2):
    """density n(s) = a exp(s/lam): match integrals over the ROI and the HE SB."""
    ratio = N2 / N1
    f = lambda lam: (np.exp(SB_HI / lam) - np.exp(SB_LO / lam)) / (np.exp(ROI_HI / lam) - np.exp(ROI_LO / lam)) - ratio
    lam = optimize.brentq(f, 50.0, 5e4)
    a = N1 / (lam * (np.exp(ROI_HI / lam) - np.exp(ROI_LO / lam)))
    return a, lam
def solve_pow(N1, N2):
    """density n(s) = a s^p."""
    ratio = N2 / N1
    f = lambda p: (SB_HI ** (p + 1) - SB_LO ** (p + 1)) / (ROI_HI ** (p + 1) - ROI_LO ** (p + 1)) - ratio
    p = optimize.brentq(f, -0.99, 40.0)
    a = N1 * (p + 1) / (ROI_HI ** (p + 1) - ROI_LO ** (p + 1))
    return a, p
def N_exp(a, lam, s0, s1): return a * lam * (np.exp(s1 / lam) - np.exp(s0 / lam))
def N_pow(a, p, s0, s1): return a / (p + 1) * (s1 ** (p + 1) - s0 ** (p + 1))

BK = {}
for comp, v in MSSI.items():
    a_e, lam = solve_exp(v['ROI'], v['HESB']); a_p, p = solve_pow(v['ROI'], v['HESB'])
    BK[comp] = dict(exp=dict(a=a_e, lam=lam), pow=dict(a=a_p, p=p))
    say(f'MSSI {comp}: exponential lambda = {lam:.0f} phd; power-law index p = {p:.2f}')
S_EDGE = np.arange(600.0, 1701.0, 10.0)
bk_rows = []
for s in S_EDGE:
    r = dict(S1c_edge=s)
    for comp in MSSI:
        r[f'{comp}_exp'] = N_exp(BK[comp]['exp']['a'], BK[comp]['exp']['lam'], 600.0, s)
        r[f'{comp}_pow'] = N_pow(BK[comp]['pow']['a'], BK[comp]['pow']['p'], 600.0, s)
    r['MSSI_exp'] = r['wall_exp'] + r['RFR_exp']; r['MSSI_pow'] = r['wall_pow'] + r['RFR_pow']
    r['MSSI_mid'] = math.sqrt(r['MSSI_exp'] * r['MSSI_pow'])
    bk_rows.append(r)
BKG = pd.DataFrame(bk_rows); BKG.to_csv(f'{OUT}/P038_mssi_vs_edge.csv', index=False)
# NR-band (+-2 sigma) fraction of the HE-SB-type MSSI population: Fig. S4 (5.4 t, HE SB science): ~2-3 of 18 events within
# ~0.065 dex of the extrapolated NR median (visual reading); bracket with P004's 0.035 and a generous 0.25
F_NR = dict(central=0.15, low=0.035, high=0.25)
# S2c < 10^4.15 (extended ROI) vs 10^4.3 (HE SB): ~8 of 18 HE-SB events above 4.15 (visual reading) -> the NR-band subset is
# unaffected; the total MSSI entering the extended ROI is ~0.55 of the HE-SB-type population
F_S2 = 10.0 / 18.0
def bkg_at(edge, which='MSSI_mid'):
    return float(np.interp(edge, BKG.S1c_edge, BKG[which]))
EXT = {}
for X in (700, 800, 1000, 1200):
    EXT[str(X)] = dict(MSSI_total_exp=bkg_at(X, 'MSSI_exp'), MSSI_total_pow=bkg_at(X, 'MSSI_pow'), MSSI_total_mid=bkg_at(X),
                       MSSI_in_ROI_S2cut=bkg_at(X) * F_S2,
                       NR_band_central=bkg_at(X) * F_NR['central'], NR_band_low=bkg_at(X, 'MSSI_pow') * F_NR['low'],
                       NR_band_high=bkg_at(X, 'MSSI_exp') * F_NR['high'])
RES['extended_roi_background'] = dict(per_edge=EXT, f_NR=F_NR, f_S2=F_S2, fits=BK,
                                      other=dict(accidentals_600_1000_est=3e-4, ER_leakage='negligible (ER median >= 0.6 dex above 10^4.15 at S1c 600-800; Fig. 4)',
                                                 neutrons='negligible (P013)', atm_nu='negligible (P019)'))
say('\nextended-ROI MSSI (4.7 t, 2.84 t yr):', json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in EXT.items()}))
# edge at which the NR-band MSSI reaches given levels
lev = {}
for target in (0.001, 0.003, 0.01, 0.03, 0.1):
    nr = BKG.MSSI_mid.values * F_NR['central']
    lev[str(target)] = float(np.interp(target, nr, BKG.S1c_edge)) if nr.max() > target else float('inf')
RES['edge_for_NR_band_level'] = lev
say('S1c edge at which NR-band MSSI (central) reaches 0.001/0.003/0.01/0.03/0.1 events:', {k: (round(v) if math.isfinite(v) else 'beyond 1700') for k, v in lev.items()})

# signal-to-background of extending the edge, for one ROI event at each delta
SB = []
for d in (300., 330., 350., 360., 370., 380.):
    for X in (800, 1000, 1200):
        extra_sig = float(A.loc[d, f'S_acc_{X}'] / A.loc[d, 'S_acc_600'] - 1)
        b = EXT[str(X)]['NR_band_central']
        SB.append(dict(delta_keV=d, edge=X, extra_signal_per_ROI_event=extra_sig, NR_band_bkg=b,
                       P_zero_extra_if_signal=math.exp(-extra_sig), extra_signal_6p8tyr=extra_sig * 2.38, bkg_6p8tyr=b * 2.38))
SB = pd.DataFrame(SB); SB.to_csv(f'{OUT}/P038_signal_vs_background_extension.csv', index=False)
say('\nextension gain (1000 GeV O1s, annual):'); say(SB.round(4).to_string(index=False))

# ------------------------------------------------------------------------------------------------------
# 5. the E_true = 262 keV scenario
# ------------------------------------------------------------------------------------------------------
S262 = dict(eps_600_at_262=PT['262.0']['eps_600'], one_over_eps=PT['262.0']['one_over_eps_600'],
            eps_600_at_246=PT['246.0']['eps_600'], eps_600_at_248=PT['248.0']['eps_600'],
            P_S1c_gt_600_at_246=PT['246.0']['P_S1c_gt_600'], P_S1c_gt_600_at_262=PT['262.0']['P_S1c_gt_600'],
            P_S1c_gt_600_at_265=PT['265.0']['P_S1c_gt_600'],
            delta_max_248_june=lz.delta_max_kev(248, M_CHI, v_kms=lz.vmax_kms(lz.v_earth_kms(DOY_JUNE))),
            delta_max_262_june=lz.delta_max_kev(262, M_CHI, v_kms=lz.vmax_kms(lz.v_earth_kms(DOY_JUNE))),
            delta_max_265_june=lz.delta_max_kev(265, M_CHI, v_kms=lz.vmax_kms(lz.v_earth_kms(DOY_JUNE))),
            eps_800_at_262=PT['262.0']['eps_800'], eps_1000_at_262=PT['262.0']['eps_1000'])
# coupling shift: one event at E_obs = 262 instead of 248 changes nothing in kappa_hat = 1/S_acc (extended likelihood, mu_hat = 1);
# the Z(delta) at E_obs = 262 is in ACC (Z_600_E262); the "1/eps" correction applies to a counting (efficiency-corrected) estimate
S262['Z_600_E248_vs_E262'] = {str(d): (float(A.loc[d, 'Z_600']), float(A.loc[d, 'Z_600_E262'])) for d in (300., 350., 370., 380., 385., 390.)}
S262['peak_E262_600'] = float(A['Z_600_E262'].idxmax())
# probability that a detected event lies within 25 keV of the 50 % point (245-295 keV observed) vs anywhere in the ROI
S262['P_obs_245_295_600'] = {str(d): float(A.loc[d, 'P_obs_245_295_600']) for d in (300., 330., 350., 370., 380.)}
# flat-in-energy spectrum reference (what fraction of a flat accepted spectrum sits in 245-295)
wflat = EPS[600](E_T) * EXPO; dflat = SMEAR @ wflat
S262['P_obs_245_295_flat'] = float(dflat[(E_O > 245) & (E_O < 295)].sum() / dflat.sum())
S262['P_obs_gt_270_600'] = {str(d): float(A.loc[d, 'P_obs_gt_270_600']) for d in (300., 350., 380.)}
RES['E262_scenario'] = S262
say('\nE=262 scenario:', json.dumps(S262, indent=1, default=float)[:2500])

# ------------------------------------------------------------------------------------------------------
# 6. figures (categorical palette in fixed order: blue, orange, aqua, yellow, magenta; light surface)
# ------------------------------------------------------------------------------------------------------
PAL = {600: '#2a78d6', 700: '#eb6834', 800: '#1baf7a', 1000: '#eda100', 1200: '#e87ba4'}
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.25, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})
Ef = np.arange(200, 600, 1.0)
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for X in S1_CUTS:
    ax.plot(Ef, EPS[X](Ef), color=PAL[X], lw=2, label=f'S1c < {X} phd')
    ax.annotate(f'{EDGE[X]["E50_keV"]:.0f}', (EDGE[X]['E50_keV'], 0.5), textcoords='offset points', xytext=(4, 6), fontsize=8, color='#52514e')
ax.plot(Ef, PLATEAU * np.interp(Ef, MC.E_keV, MC.P_S2c_pass), color='#52514e', lw=1.2, ls=':', label=r'S2c < 10$^{4.15}$ only')
ax.errorbar(INSET[:, 0], INSET[:, 1], yerr=0.03, fmt='o', ms=4, color='#0b0b0b', label='LZ Fig. S2 inset (read)', zorder=5)
ax.axvline(269.9, color='#0b0b0b', lw=0.8, ls='--'); ax.axvline(248, color='#52514e', lw=0.8, ls='-.')
ax.text(232, 0.60, 'event\n248 keV', fontsize=8, color='#52514e', ha='center'); ax.text(271, 0.9, '269.9', fontsize=8)
ax.set_xlabel('true nuclear-recoil energy [keV]'); ax.set_ylabel('signal efficiency'); ax.set_ylim(0, 1.02); ax.set_xlim(200, 600)
ax.set_title('NR efficiency for alternative S1c edges (NEST-LZ MC, paper energy scale)', fontsize=10)
ax.legend(fontsize=8, loc='lower right', bbox_to_anchor=(1.0, 0.08), frameon=False)
fig.tight_layout(); fig.savefig(f'{FIG}/P038_fig1_efficiency_cuts.png', dpi=160); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
ax = axs[0]
for X in S1_CUTS:
    ax.plot(A.index, A[f'A_{X}'], color=PAL[X], lw=2, marker='o', ms=3, label=f'S1c < {X} phd')
ax.plot(A.index, 1 - A.frac_true_gt_270, color='#0b0b0b', lw=1, ls=':', label='true E < 270 keV (sharp)')
ax.set_xlabel(r'mass splitting $\delta$ [keV]'); ax.set_ylabel('accepted fraction of the inelastic spectrum')
ax.set_title(r'O$_1$ inelastic, 1000 GeV, annual halo', fontsize=10); ax.set_ylim(0, 1.0); ax.legend(fontsize=8, frameon=False, loc='lower left')
ax = axs[1]
for X in S1_CUTS:
    ax.plot(A.index, A[f'Z_{X}'], color=PAL[X], lw=2, marker='o', ms=3, label=f'S1c < {X} phd')
ax.plot(B21.index, B21.Z, color='#0b0b0b', lw=1, ls='--', label='P021 (600 phd, full likelihood)')
ax.set_xlabel(r'mass splitting $\delta$ [keV]'); ax.set_ylabel(r'single-event profile significance Z [$\sigma$]')
ax.set_title('one event at 248 keV; no events in the extension', fontsize=10); ax.set_xlim(245, 395); ax.set_ylim(0, 4)
ax.legend(fontsize=8, frameon=False, loc='lower left')
fig.tight_layout(); fig.savefig(f'{FIG}/P038_fig2_acceptance_Z_vs_delta.png', dpi=160); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
ax = axs[0]
ax.fill_between(BKG.S1c_edge, BKG.MSSI_pow, BKG.MSSI_exp, color='#2a78d6', alpha=0.25, label='all MSSI, 4.7 t (power-law to exponential)')
ax.plot(BKG.S1c_edge, BKG.MSSI_mid, color='#2a78d6', lw=2)
ax.fill_between(BKG.S1c_edge, BKG.MSSI_pow * F_NR['low'], BKG.MSSI_exp * F_NR['high'], color='#eb6834', alpha=0.25, label=r'within $\pm2\sigma$ of NR median')
ax.plot(BKG.S1c_edge, BKG.MSSI_mid * F_NR['central'], color='#eb6834', lw=2)
ax.axhline(0.0106, color='#0b0b0b', lw=0.8, ls='--'); ax.text(610, 0.012, 'current S1c>500 phd panel total 0.0106', fontsize=8)
ax.axhline(0.0049, color='#52514e', lw=0.8, ls=':'); ax.text(610, 0.0035, 'MSSI in 3-600 phd: 0.0049', fontsize=8, color='#52514e')
ax.set_yscale('log'); ax.set_ylim(1e-4, 3); ax.set_xlim(600, 1700)
ax.set_xlabel('upper S1c edge [phd]'); ax.set_ylabel('MSSI events in 600 phd - edge (2.84 t yr)')
ax.set_title('Background entering an extended ROI (4.7 t FV)', fontsize=10); ax.legend(fontsize=8, frameon=False, loc='lower right')
ax = axs[1]
cols = {300.: '#2a78d6', 350.: '#eb6834', 370.: '#1baf7a', 380.: '#eda100'}
edges_plot = np.array([700, 800, 1000, 1200])
for d, c in cols.items():
    g = [A.loc[d, f'S_acc_{X}'] / A.loc[d, 'S_acc_600'] - 1 for X in edges_plot]
    ax.plot(edges_plot, g, color=c, lw=2, marker='o', ms=4, label=fr'$\delta$ = {d:.0f} keV')
ax.plot(edges_plot, [EXT[str(X)]['NR_band_central'] for X in edges_plot], color='#0b0b0b', lw=1.2, ls='--', marker='s', ms=3,
        label='NR-band MSSI in the extension')
ax.set_yscale('log'); ax.set_ylim(1e-4, 100); ax.set_xlim(650, 1250); ax.set_xlabel('upper S1c edge [phd]')
ax.set_ylabel('extra signal events per event in 3-600 phd')
ax.set_title('Signal gained by extending the edge (1000 GeV O$_1$)', fontsize=10); ax.legend(fontsize=8, frameon=False, loc='upper left')
fig.tight_layout(); fig.savefig(f'{FIG}/P038_fig3_background_vs_edge.png', dpi=160); plt.close(fig)

# ------------------------------------------------------------------------------------------------------
# 7. save
# ------------------------------------------------------------------------------------------------------
RES['settings'] = dict(g1=G1, g2=G2, plateau=PLATEAU, E_MC=[200, 600, 4], N_MC=N_MC, N_MC_points=40000, S1_cuts=S1_CUTS, logS2_max=LOGS2_MAX,
                       alpha_paper=ALPHA_P, beta_paper=BETA_P, ext_eff=EXT_EFF, pos_res=POS_RES, s2Fano=S2_FANO, sPEres=SPE_RES,
                       m_chi_GeV=M_CHI, deltas=DELTAS.tolist(), E_T=[1.5, 800, 3], resolution='11 sqrt(E/248) keV', b_H=B_H, W_H=W_H,
                       exposure_tyr=EXPO, halos=['annual (12 monthly days)', 'june (day 167)'], seed=38)
RES['acceptance_annual'] = {str(d): {k: float(A.loc[d, k]) for k in ['A_600', 'A_700', 'A_800', 'A_1000', 'A_1200', 'frac_true_gt_270', 'E_median_true', 'E_peak_true',
                                                                   'Z_600', 'Z_800', 'Z_1000', 'Z_1200', 'Z_none', 'kappa_hat_600', 'kappa_hat_1000', 'gain_1000_over_600', 'pct_obs_600', 'pct_obs_1000']}
                            for d in A.index}
json.dump(RES, open(f'{OUT}/P038_results.json', 'w'), indent=1, default=float)
say(f'\nsaved results; total time {time.time() - T0:.0f}s')
