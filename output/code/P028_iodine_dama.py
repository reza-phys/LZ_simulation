"""
P028 -- Iodine as the second target: what DAMA/LIBRA's modulation data and COSINE/ANAIS already say
about an inelastic 300-390 keV splitting.

Computes, for O1 isoscalar inelastic DM (m_chi = 1000 GeV, delta = 300/350/366/380 keV) normalised to
LZ's one-event best fit (P015 convention: LZ unit coupling c_p = c_n = 1/m_v^2 <-> WimPyDD c^0 = 2/m_v^2;
xenon annual-mean rate x LZ efficiency x 2.84 t yr = 1 event):
  1. iodine (and sodium) recoil spectra in true recoil energy and in electron-equivalent energy for
     NaI(Tl) with a constant iodine quenching Q_I = 0.09 (range 0.06-0.12), in cpd/kg(NaI)/keVee;
  2. the time dependence through the year (73 days, WimPyDD halo with Earth's orbital velocity) and the
     resulting unmodulated rate S0, cosine amplitude S_m (DAMA phase t0 = 152.5 d and free phase) and
     harmonic content, per 1-keVee bin;
  3. comparison with the DAMA/LIBRA modulation amplitudes bundled with WimPyDD (digitisation of
     Bernabei et al. 2018 Fig. 11) and with Poisson estimates of the DAMA/COSINE/ANAIS sensitivities;
  4. the NaI exposure needed for a 3 sigma modulation detection at a DAMA-like background, and the
     exposure a hypothetical background-free NR-discriminating iodine detector would need.
Run from the simulation root:  .venv/bin/python output/code/P028_iodine_dama.py
"""
import sys, os, math, json, time
import numpy as np
from scipy import special
sys.path.insert(0, 'output/code')
from common import lzcommon as lz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'output/work/P028'; FIG = OUT + '/figures'
os.makedirs(FIG, exist_ok=True)
T0 = time.time()
WD = lz.wd()

# ------------------------------------------------------------------------------------------------
# constants and inputs
# ------------------------------------------------------------------------------------------------
MV = lz.M_V_GEV                                   # 246.2 GeV
C = lz.C_KMS
VESC = lz.VESC_KMS                                # 544 km/s (Baxter 2021 via lzcommon)
M_CHI = 1000.0
DELTAS = [300.0, 350.0, 366.0, 380.0]             # 366 keV = P007 Higgsino delta(N=1)
QI = 0.09                                         # iodine quenching, DAMA (recalled, likely; also WimPyDD DAMA folder)
QI_RANGE = [0.06, 0.09, 0.12]
QNA = 0.30                                        # sodium quenching (WimPyDD DAMA folder; recalled 0.25-0.3)
M_I, M_NA = 126.904, 22.990                       # u (recalled, certain)
F_I = M_I / (M_I + M_NA)                          # 0.8466 mass fraction of iodine in NaI
KGD_PER_TYR = 1000.0 * 365.25
T0_DAMA = 152.5                                   # DAMA phase (2 June), also WimPyDD default modulation_phase
OMEGA = 2 * math.pi / 365.25
B_DAMA = 1.0                                      # cpd/kg/keV single-hit rate at 10-60 keVee (recalled, likely)
EXPO = {  # NaI exposures in kg d  (recalled; reliability flagged)
    'DAMA/LIBRA phase1+2 (2.46 t yr)': dict(kgd=2.46 * KGD_PER_TYR, B=1.0, rel='likely'),
    'COSINE-100 (~0.2 t yr)': dict(kgd=0.2 * KGD_PER_TYR, B=3.0, rel='uncertain'),
    'ANAIS-112 (~0.3 t yr)': dict(kgd=0.3 * KGD_PER_TYR, B=3.0, rel='uncertain'),
}

HAM = lz.wd_hamiltonian('O1_iso_unit', {1: (2.0 / MV ** 2, 0.0)})   # LZ unit coupling (P003 convention)
VGRID = np.linspace(0.0, VESC + 300.0, 1200)
ONES = np.ones_like(VGRID)

# ------------------------------------------------------------------------------------------------
# Part 0: kinematics (iodine vs sodium vs xenon)
# ------------------------------------------------------------------------------------------------
V_JUNE = lz.v_earth_kms(167) + VESC
kin = {}
for sym, A in [('I', 127), ('Na', 23), ('Xe', lz.A_XE_MEAN)]:
    mu = lz.mu_red(M_CHI, lz.m_nucleus_gev(A))
    dmax = mu * (V_JUNE / C) ** 2 / 2 * 1e6
    kin[sym] = dict(A=A, delta_ceiling_june_keV=dmax)
    for d in DELTAS:
        Em, Ep = lz.E_R_range_keV(M_CHI, V_JUNE, A=A, delta_kev=d)
        Q = QI if sym == 'I' else (QNA if sym == 'Na' else 1.0)
        kin[sym][f'window_delta{int(d)}'] = None if math.isnan(Em) else dict(E_minus_keV=Em, E_plus_keV=Ep,
                                                                                E_minus_keVee=Em * Q, E_plus_keVee=Ep * Q)
print('Part 0: delta ceilings (1 TeV, 16 June): I %.1f  Na %.1f  Xe %.1f keV' % (
    kin['I']['delta_ceiling_june_keV'], kin['Na']['delta_ceiling_june_keV'], kin['Xe']['delta_ceiling_june_keV']))
print('  sodium windows:', [kin['Na'][f'window_delta{int(d)}'] for d in DELTAS])

# ------------------------------------------------------------------------------------------------
# Part 1: halo functions through the year and per-stream kernels
# ------------------------------------------------------------------------------------------------
NDAY = 73
DAYS = (np.arange(NDAY) + 0.5) * 365.25 / NDAY           # uniform sampling of the year (5.0 d)
DETA = np.array([lz.wd_halo(day_of_year=float(d), vmin=VGRID)[1] for d in DAYS])   # (73, 1200)
days12 = 15.0 + 365.25 / 12 * np.arange(12)
DETA_ANNUAL12 = np.mean([lz.wd_halo(day_of_year=float(d), vmin=VGRID)[1] for d in days12], axis=0)   # P015 convention
DETA_JUNE = lz.wd_halo(day_of_year=167.0, vmin=VGRID)[1]
print('Part 1a: %d halo functions (%.1f s)' % (NDAY + 13, time.time() - T0))

def kernel(target, E_keV, d):
    """Per-stream kernel K(E, vmin) such that rate = K . delta_eta  [events/kg/day/keV, unit coupling]."""
    return np.array([WD.diff_rate(target, HAM, M_CHI, float(e), VGRID, ONES, delta=d, sum_over_streams=False) for e in E_keV])

def eff_LZ(E, sig_hi=11.5, sig_lo=2.5):
    """LZ NR efficiency model (P005/P007/P015): 0.96 plateau, 50% at 5.4 and 269.9 keV."""
    E = np.asarray(E, float)
    return lz.LZ['eff_plateau'] * 0.5 * (1 + special.erf((E - lz.LZ['E_50pct_low_keV']) / (math.sqrt(2) * sig_lo))) \
        * 0.5 * special.erfc((E - lz.LZ['E_50pct_high_keV']) / (math.sqrt(2) * sig_hi))

# LZ normalisation (xenon, annual-mean halo, LZ efficiency, 2.84 t yr) -- reproduces P015
E_LZ = np.linspace(0.5, 330.0, 120)
norm = {}
p015 = json.load(open('output/work/P015/P015_summary.json'))['lz_normalisation']
for d in DELTAS:
    KXe = kernel(WD.Xe, E_LZ, d)
    rXe = np.clip(KXe @ DETA_ANNUAL12, 0, None) * KGD_PER_TYR          # events/(t yr keV)
    n_unit = lz.LZ['exposure_tyr'] * float(np.trapezoid(rXe * eff_LZ(E_LZ), E_LZ))
    rXe73 = np.clip(KXe @ DETA.mean(axis=0), 0, None) * KGD_PER_TYR
    n_unit73 = lz.LZ['exposure_tyr'] * float(np.trapezoid(rXe73 * eff_LZ(E_LZ), E_LZ))
    norm[d] = dict(N_LZ_unit_coupling=n_unit, N_LZ_unit_coupling_73day_mean=n_unit73,
                   scale_to_1_event=1.0 / n_unit, P015_N_LZ_unit_coupling=p015[str(int(d))]['N_LZ_unit_coupling'],
                   ratio_to_P015=n_unit / p015[str(int(d))]['N_LZ_unit_coupling'],
                   c1s_mv2_sq_bestfit=1.0 / n_unit,
                   sigma_SI_bestfit_cm2=(1.0 / n_unit) * lz.mu_red(M_CHI, lz.M_NUCLEON_GEV) ** 2 / (math.pi * MV ** 4) * lz.GEV_TO_CM2)
print('Part 1b: LZ unit-coupling events', {int(d): '%.4g (P015 %.4g)' % (v['N_LZ_unit_coupling'], v['P015_N_LZ_unit_coupling']) for d, v in norm.items()})

# iodine kernels on the June kinematic window
NE = 90
IOD = {}
for d in DELTAS:
    w = kin['I'][f'window_delta{int(d)}']
    E = np.linspace(0.97 * w['E_minus_keV'], 1.03 * w['E_plus_keV'], NE)
    K = kernel(WD.I, E, d)                                       # (NE, 1200) events/kg(I)/day/keV per unit eta
    R_day = np.clip(K @ DETA.T, 0, None)                         # (NE, 73)  unit coupling, per kg of iodine
    R_ann = np.clip(K @ DETA_ANNUAL12, 0, None)
    R_jun = np.clip(K @ DETA_JUNE, 0, None)
    IOD[d] = dict(E=E, R_day=R_day, R_ann=R_ann, R_jun=R_jun,
                  win_ann_per_tyr_I=float(np.trapezoid(R_ann, E)) * KGD_PER_TYR,
                  win_jun_per_tyr_I=float(np.trapezoid(R_jun, E)) * KGD_PER_TYR)
    # validation: WimPyDD's own DAMA target (Q_I = 0.09, Na + I per kg NaI) vs our manual conversion
    nai = WD.DAMA_LIBRA_2019.target
    checks = []
    for eee in [0.5 * (w['E_minus_keVee'] + w['E_plus_keVee']), 0.7 * w['E_plus_keVee']]:
        r_dama = WD.diff_rate(nai, HAM, M_CHI, float(eee), VGRID, DETA_JUNE, delta=d)
        r_man = float(np.interp(eee / QI, E, R_jun)) / QI * F_I
        checks.append(dict(E_keVee=eee, wimpydd_dama_target=r_dama, manual_I_only=r_man, ratio=r_man / r_dama if r_dama > 0 else None))
    IOD[d]['dama_target_check'] = checks
print('Part 1c: iodine kernels done (%.1f s); I/Xe annual window ratios:' % (time.time() - T0),
      {int(d): round(IOD[d]['win_ann_per_tyr_I'] / (float(np.trapezoid(np.clip(kernel(WD.Xe, np.linspace(0.97 * lz.E_R_range_keV(M_CHI, V_JUNE, delta_kev=d)[0], 1.03 * lz.E_R_range_keV(M_CHI, V_JUNE, delta_kev=d)[1], NE), d) @ DETA_ANNUAL12, 0, None), np.linspace(0.97 * lz.E_R_range_keV(M_CHI, V_JUNE, delta_kev=d)[0], 1.03 * lz.E_R_range_keV(M_CHI, V_JUNE, delta_kev=d)[1], NE))) * KGD_PER_TYR), 3) for d in DELTAS})
print('  DAMA-target check (ratio manual/WimPyDD):', [[round(c['ratio'], 4) for c in IOD[d]['dama_target_check']] for d in DELTAS])

# ------------------------------------------------------------------------------------------------
# Part 2: electron-equivalent spectra for NaI, per day; S0, S_m and harmonics per 1-keVee bin
# ------------------------------------------------------------------------------------------------
EEE_FINE = np.arange(0.5, 90.0, 0.05)      # fine E_ee grid
BIN_EDGES = np.arange(1.0, 91.0, 1.0)      # 1-keVee bins 1-90
BIN_C = 0.5 * (BIN_EDGES[1:] + BIN_EDGES[:-1])

def sigma_dama(E):
    """DAMA/LIBRA energy resolution sigma(E_ee) = 0.0091 E + 0.448 sqrt(E) keV (WimPyDD DAMA folder, from 1002.1028)."""
    return 0.0091 * E + 0.448 * np.sqrt(E)

def to_nai_ee(E_R, R_R, Q, scale):
    """dR/dE_ee [cpd/kg(NaI)/keVee] on EEE_FINE from an iodine spectrum dR/dE_R [events/kg(I)/day/keV] at unit coupling."""
    r = np.interp(EEE_FINE / Q, E_R, R_R, left=0.0, right=0.0)
    return F_I * scale * r / Q

def smear(spec):
    """Gaussian resolution folding on the fine grid (variable sigma)."""
    dE = EEE_FINE[1] - EEE_FINE[0]
    sig = sigma_dama(EEE_FINE)
    G = np.exp(-0.5 * ((EEE_FINE[:, None] - EEE_FINE[None, :]) / sig[None, :]) ** 2) / (math.sqrt(2 * math.pi) * sig[None, :])
    return (G * spec[None, :]).sum(axis=1) * dE

def binned(spec):
    idx = np.digitize(EEE_FINE, BIN_EDGES) - 1
    out = np.zeros(len(BIN_C))
    for i in range(len(BIN_C)):
        m = idx == i
        out[i] = spec[m].mean() if m.any() else 0.0
    return out

def harmonics(S_t):
    """S(t) sampled uniformly over the year -> S0, S_m at fixed DAMA phase, free-phase amplitude/phase, harmonics."""
    S0 = S_t.mean()
    ph = OMEGA * (DAYS - T0_DAMA)
    Sm_fixed = 2 * np.mean(S_t * np.cos(ph))
    a1 = 2 * np.mean(S_t * np.cos(OMEGA * DAYS)); b1 = 2 * np.mean(S_t * np.sin(OMEGA * DAYS))
    A1 = math.hypot(a1, b1); t_peak = (math.atan2(b1, a1) / OMEGA) % 365.25
    A = [A1]
    for k in (2, 3, 4):
        ak = 2 * np.mean(S_t * np.cos(k * OMEGA * DAYS)); bk = 2 * np.mean(S_t * np.sin(k * OMEGA * DAYS))
        A.append(math.hypot(ak, bk))
    var = np.mean((S_t - S0) ** 2)
    frac_fund = 0.5 * A1 ** 2 / var if var > 0 else float('nan')
    return dict(S0=S0, Sm_fixed_phase=Sm_fixed, A1=A1, t_peak=t_peak, A2=A[1], A3=A[2], A4=A[3],
                A2_over_A1=A[1] / A1 if A1 > 0 else float('nan'), frac_variance_fundamental=frac_fund,
                mod_fraction=Sm_fixed / S0 if S0 > 0 else float('nan'),
                max_over_min=(S_t.max() / S_t.min()) if S_t.min() > 0 else float('inf'),
                zero_fraction=float(np.mean(S_t <= 0)))

results = {}
rows_bins = []
rows_time = []
for d in DELTAS:
    sc = norm[d]['scale_to_1_event']
    I = IOD[d]
    res_d = {}
    for Q in QI_RANGE:
        spec_day = np.array([to_nai_ee(I['E'], I['R_day'][:, j], Q, sc) for j in range(NDAY)])   # (73, fine)
        spec_ann = spec_day.mean(axis=0)
        B_day = np.array([binned(s) for s in spec_day])                                           # (73, nbins)
        H = [harmonics(B_day[:, i]) for i in range(len(BIN_C))]
        # whole-window quantities (rate in window, cpd/kg NaI)
        win_t = spec_day.sum(axis=1) * (EEE_FINE[1] - EEE_FINE[0])
        Hwin = harmonics(win_t)
        entry = dict(Q=Q, bins=H, window=Hwin, spec_ann=spec_ann,
                     E_minus_keVee=kin['I'][f'window_delta{int(d)}']['E_minus_keV'] * Q,
                     E_plus_keVee=kin['I'][f'window_delta{int(d)}']['E_plus_keV'] * Q)
        S0b = np.array([h['S0'] for h in H]); Smb = np.array([h['Sm_fixed_phase'] for h in H])
        ip, im = int(np.argmax(S0b)), int(np.argmax(Smb))
        entry.update(peak_S0=dict(E_keVee=BIN_C[ip], S0=S0b[ip], Sm=Smb[ip]), peak_Sm=dict(E_keVee=BIN_C[im], Sm=Smb[im], S0=S0b[im]),
                     sum_Sm2=float(np.sum(Smb ** 2)), sum_S0=float(np.sum(S0b)))
        if Q == QI:
            # resolution-smeared variant (annual mean and modulation)
            spec_day_sm = np.array([smear(s) for s in spec_day])
            B_day_sm = np.array([binned(s) for s in spec_day_sm])
            Hs = [harmonics(B_day_sm[:, i]) for i in range(len(BIN_C))]
            entry['bins_smeared'] = Hs
            entry['peak_Sm_smeared'] = max(((h['Sm_fixed_phase'], BIN_C[i]) for i, h in enumerate(Hs)))
            for i, h in enumerate(H):
                rows_bins.append(dict(delta_keV=d, E_lo_keVee=BIN_EDGES[i], E_hi_keVee=BIN_EDGES[i + 1], **{k: h[k] for k in h},
                                      S0_smeared=Hs[i]['S0'], Sm_smeared=Hs[i]['Sm_fixed_phase']))
            for j, day in enumerate(DAYS):
                rows_time.append(dict(delta_keV=d, day=day, window_rate_cpd_kg=win_t[j],
                                      rate_peakbin_cpd_kg_keV=B_day[j, ip]))
        res_d[Q] = entry
    results[d] = res_d
    e = res_d[QI]
    print('Part 2: delta=%.0f keV: window %.1f-%.1f keVee; S0 peak %.3g at %.1f keVee; Sm peak %.3g cpd/kg/keV at %.1f keVee; '
          'window mod. fraction %.2f, t_peak %.0f d, A2/A1 %.2f, zero fraction %.2f' % (
              d, e['E_minus_keVee'], e['E_plus_keVee'], e['peak_S0']['S0'], e['peak_S0']['E_keVee'], e['peak_Sm']['Sm'],
              e['peak_Sm']['E_keVee'], e['window']['mod_fraction'], e['window']['t_peak'], e['window']['A2_over_A1'],
              e['window']['zero_fraction']))

# ------------------------------------------------------------------------------------------------
# Part 3: comparison with DAMA/LIBRA (bundled table) and Poisson sensitivities; required exposures
# ------------------------------------------------------------------------------------------------
dama_tab = np.loadtxt('WimPyDD/Experiments/DAMA_LIBRA_2019/modulation_amplitudes.tab')
dama_rows = [dict(E_lo=r[0], E_hi=r[1], band_lo=r[2], band_hi=r[3], central=0.5 * (r[2] + r[3]), half_width=0.5 * (r[3] - r[2])) for r in dama_tab]

def sigma_Sm(kgd, B, dE=1.0):
    """Poisson uncertainty of a cosine-amplitude fit: sigma_Sm = sqrt(2 B / (exposure dE))  [cpd/kg/keV]."""
    return math.sqrt(2 * B / (kgd * dE))

comparison = {}
for d in DELTAS:
    e = results[d][QI]
    Smb = np.array([h['Sm_fixed_phase'] for h in e['bins']]); S0b = np.array([h['S0'] for h in e['bins']])
    comp = dict(peak_Sm=e['peak_Sm'], window_keVee=[e['E_minus_keVee'], e['E_plus_keVee']])
    # DAMA bundled bins overlapping the signal window
    ov = []
    for r in dama_rows:
        m = (BIN_C >= r['E_lo']) & (BIN_C < r['E_hi'])
        if m.any() and Smb[m].max() > 0:
            ov.append(dict(E_lo=r['E_lo'], E_hi=r['E_hi'], dama_central=r['central'], dama_half_width=r['half_width'],
                           predicted_Sm=float(Smb[m].mean()), ratio_halfwidth_over_pred=r['half_width'] / float(Smb[m].mean())))
    comp['dama_bundled_overlap'] = ov
    exps = {}
    for name, x in EXPO.items():
        s1 = sigma_Sm(x['kgd'], x['B'])
        # expected signal counts in the window and the best-possible combined significance over all bins
        N_sig = e['sum_S0'] * x['kgd']
        z_comb = math.sqrt(e['sum_Sm2'] * x['kgd'] / (2 * x['B']))
        exps[name] = dict(exposure_kgd=x['kgd'], B_cpd_kg_keV=x['B'], reliability=x['rel'], sigma_Sm_1keV=s1,
                          sigma_over_peak_Sm=s1 / e['peak_Sm']['Sm'], z_peak_bin=e['peak_Sm']['Sm'] / s1,
                          z_all_bins_combined=z_comb, expected_signal_events_window=N_sig,
                          background_counts_window=x['B'] * (e['E_plus_keVee'] - e['E_minus_keVee']) * x['kgd'])
    comp['experiments'] = exps
    # required exposures
    Sm_pk = e['peak_Sm']['Sm']
    req = dict(
        dama_like_B1_peak_bin_3sigma_tyr=18 * B_DAMA / (Sm_pk ** 2) / KGD_PER_TYR,
        dama_like_B1_all_bins_3sigma_tyr=18 * B_DAMA / e['sum_Sm2'] / KGD_PER_TYR,
        bkgfree_modulation_3sigma_tyr=18 * e['window']['S0'] / (e['window']['Sm_fixed_phase'] ** 2) / KGD_PER_TYR,
        bkgfree_3_events_tyr=3.0 / e['window']['S0'] / KGD_PER_TYR,
        bkgfree_3_events_tyr_iodine=3.0 / e['window']['S0'] / KGD_PER_TYR * F_I,
    )
    comp['required_exposure'] = req
    comparison[d] = comp
    print('Part 3: delta=%.0f: DAMA sigma_Sm(1 keV)=%.2e -> sigma/Sm_peak=%.2e; z(all bins)=%.1e; N_sig(DAMA)=%.3f; '
          'E_3sigma(B=1, peak bin)=%.2e t yr, all bins %.2e; bkg-free modulation %.0f t yr, 3 events %.1f t yr NaI' % (
              d, exps['DAMA/LIBRA phase1+2 (2.46 t yr)']['sigma_Sm_1keV'], exps['DAMA/LIBRA phase1+2 (2.46 t yr)']['sigma_over_peak_Sm'],
              exps['DAMA/LIBRA phase1+2 (2.46 t yr)']['z_all_bins_combined'], exps['DAMA/LIBRA phase1+2 (2.46 t yr)']['expected_signal_events_window'],
              req['dama_like_B1_peak_bin_3sigma_tyr'], req['dama_like_B1_all_bins_3sigma_tyr'], req['bkgfree_modulation_3sigma_tyr'], req['bkgfree_3_events_tyr']))

# ------------------------------------------------------------------------------------------------
# outputs
# ------------------------------------------------------------------------------------------------
with open(f'{OUT}/bins_S0_Sm.csv', 'w') as f:
    keys = list(rows_bins[0].keys()); f.write(','.join(keys) + '\n')
    for r in rows_bins:
        f.write(','.join('%.6g' % r[k] if isinstance(r[k], float) else str(r[k]) for k in keys) + '\n')
with open(f'{OUT}/time_series.csv', 'w') as f:
    keys = list(rows_time[0].keys()); f.write(','.join(keys) + '\n')
    for r in rows_time:
        f.write(','.join('%.6g' % r[k] for k in keys) + '\n')
with open(f'{OUT}/spectra_true_energy.csv', 'w') as f:
    f.write('delta_keV,E_R_keV,dRdE_unit_annual_per_kgI_day_keV,dRdE_unit_june16_per_kgI_day_keV,dRdE_bestfit_annual_cpd_kgNaI_keVnr\n')
    for d in DELTAS:
        I = IOD[d]
        for e_, ra, rj in zip(I['E'], I['R_ann'], I['R_jun']):
            f.write('%.0f,%.3f,%.6g,%.6g,%.6g\n' % (d, e_, ra, rj, ra * norm[d]['scale_to_1_event'] * F_I))
with open(f'{OUT}/quenching_variants.csv', 'w') as f:
    f.write('delta_keV,Q_I,E_minus_keVee,E_plus_keVee,peak_S0_keVee,peak_S0_cpd_kg_keV,peak_Sm_keVee,peak_Sm_cpd_kg_keV,window_S0_cpd_kg,window_Sm_cpd_kg,window_mod_fraction\n')
    for d in DELTAS:
        for Q in QI_RANGE:
            e = results[d][Q]
            f.write('%.0f,%.2f,%.2f,%.2f,%.1f,%.4g,%.1f,%.4g,%.4g,%.4g,%.3f\n' % (
                d, Q, e['E_minus_keVee'], e['E_plus_keVee'], e['peak_S0']['E_keVee'], e['peak_S0']['S0'], e['peak_Sm']['E_keVee'],
                e['peak_Sm']['Sm'], e['window']['S0'], e['window']['Sm_fixed_phase'], e['window']['mod_fraction']))
with open(f'{OUT}/dama_comparison.csv', 'w') as f:
    f.write('delta_keV,experiment,exposure_kgd,B_cpd_kg_keV,reliability,sigma_Sm_1keV,sigma_over_peak_Sm,z_peak_bin,z_all_bins,N_signal_window,N_background_window\n')
    for d in DELTAS:
        for name, x in comparison[d]['experiments'].items():
            f.write('%.0f,"%s",%.4g,%.1f,%s,%.4g,%.4g,%.4g,%.4g,%.4g,%.4g\n' % (
                d, name, x['exposure_kgd'], x['B_cpd_kg_keV'], x['reliability'], x['sigma_Sm_1keV'], x['sigma_over_peak_Sm'],
                x['z_peak_bin'], x['z_all_bins_combined'], x['expected_signal_events_window'], x['background_counts_window']))

def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items() if k not in ('spec_ann', 'bins', 'bins_smeared', 'E', 'R_day', 'R_ann', 'R_jun')}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o

summary = dict(
    m_chi_GeV=M_CHI, deltas_keV=DELTAS, Q_I=QI, Q_I_range=QI_RANGE, Q_Na=QNA, iodine_mass_fraction=F_I,
    v_max_june_kms=V_JUNE, n_days=NDAY, kinematics=kin, lz_normalisation=norm,
    iodine_window_rates_unit_coupling_per_tyr_I={str(int(d)): dict(annual=IOD[d]['win_ann_per_tyr_I'], june16=IOD[d]['win_jun_per_tyr_I'],
                                                                   june_over_annual=IOD[d]['win_jun_per_tyr_I'] / IOD[d]['win_ann_per_tyr_I'],
                                                                   P015_annual=float({300: 3448.61, 350: 61.7831, 366: 6.29754, 380: 0.197544}[int(d)]))
                                                    for d in DELTAS},
    dama_target_validation={str(int(d)): IOD[d]['dama_target_check'] for d in DELTAS},
    results={str(int(d)): {str(Q): clean(results[d][Q]) for Q in QI_RANGE} for d in DELTAS},
    dama_bundled_table=dama_rows,
    dama_bundled_table_note='WimPyDD/Experiments/DAMA_LIBRA_2019/modulation_amplitudes.tab, header: taken from Fig. 11 of Nucl. Phys. At. Energy 19 (2018) 307; '
                            'columns 3-4 interpreted as the lower/upper edges of the 1-sigma band (cpd/kg/keV) -- interpretation ours',
    poisson_sigma_Sm_formula='sigma_Sm = sqrt(2 B / (exposure_kgd * dE_keV))',
    comparison={str(int(d)): clean(comparison[d]) for d in DELTAS},
    runtime_s=time.time() - T0,
)
json.dump(clean(summary), open(f'{OUT}/P028_summary.json', 'w'), indent=1)

# ------------------------------------------------------------------------------------------------
# figures
# ------------------------------------------------------------------------------------------------
COL = {300.0: '#2a78d6', 350.0: '#eb6834', 366.0: '#1baf7a', 380.0: '#eda100'}   # fixed categorical order
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#898781',
                     'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e', 'ytick.color': '#52514e', 'grid.color': '#e6e5e1'})

# Fig 1: S0 and Sm in E_ee vs DAMA scale
fig, axes = plt.subplots(2, 1, figsize=(6.4, 7.2), sharex=True)
ax = axes[0]
for d in DELTAS:
    e = results[d][QI]
    S0b = np.array([h['S0'] for h in e['bins']])
    ax.step(BIN_EDGES[:-1], np.where(S0b > 0, S0b, np.nan), where='post', color=COL[d], lw=1.8, label=f'δ = {d:.0f} keV')
    lo = np.array([h['S0'] for h in results[d][0.12]['bins']]); hi = np.array([h['S0'] for h in results[d][0.06]['bins']])
ax.set_yscale('log'); ax.set_ylabel('S$_0$ (unmodulated rate) [cpd/kg/keVee]')
ax.axhline(B_DAMA, color='#898781', ls=':', lw=1); ax.text(70, 1.4, 'DAMA single-hit rate ≈ 1 cpd/kg/keV (recalled)', color='#52514e', ha='right', fontsize=8)
ax.set_ylim(1e-8, 5); ax.legend(frameon=False, loc='center left', bbox_to_anchor=(0.02, 0.62), title='O$_1$ inelastic, 1 TeV, LZ 1-event fit', title_fontsize=8)
ax.grid(axis='y', lw=0.5)
ax = axes[1]
for d in DELTAS:
    e = results[d][QI]
    Smb = np.array([h['Sm_fixed_phase'] for h in e['bins']])
    ax.step(BIN_EDGES[:-1], np.where(Smb > 0, Smb, np.nan), where='post', color=COL[d], lw=1.8, label=f'δ = {d:.0f} keV')
    dx = {300.0: -3.0, 350.0: 0.0, 366.0: 9.0, 380.0: 0.0}[d]
    ax.annotate('%.1e' % e['peak_Sm']['Sm'], (e['peak_Sm']['E_keVee'] + dx, e['peak_Sm']['Sm'] * 1.6), color='#52514e', fontsize=7.5, ha='center')
# DAMA bundled band (1-16 keVee) and Poisson sensitivities
for r in dama_rows:
    ax.fill_between([r['E_lo'], r['E_hi']], max(r['band_lo'], 1e-4), r['band_hi'], color='#c3c2b7', alpha=0.7, lw=0)
ax.text(17.5, 2.5e-2, 'DAMA/LIBRA-phase2 S$_m$ ± 1σ (grey; bands reaching zero cut at 10$^{-4}$)\n(Bernabei et al. 2018, Fig. 11 as bundled in WimPyDD)', fontsize=7.5, color='#52514e')
s_dama = comparison[300.0]['experiments']['DAMA/LIBRA phase1+2 (2.46 t yr)']['sigma_Sm_1keV']
ax.axhline(s_dama, color='#0b0b0b', ls='--', lw=1); ax.text(88, s_dama * 1.4, 'Poisson σ(S$_m$), DAMA 2.46 t·yr, B = 1, 1 keV bin', ha='right', fontsize=7.5)
s_cos = comparison[300.0]['experiments']['COSINE-100 (~0.2 t yr)']['sigma_Sm_1keV']
ax.axhline(s_cos, color='#0b0b0b', ls='-.', lw=1); ax.text(88, s_cos * 1.4, 'COSINE-100 / ANAIS-112 (~0.2–0.3 t·yr, B ≈ 3)', ha='right', fontsize=7.5)
ax.set_yscale('log'); ax.set_ylim(1e-9, 0.1); ax.set_xlim(0, 90)
ax.set_xlabel('electron-equivalent energy E$_{ee}$ [keVee]  (Q$_I$ = 0.09)'); ax.set_ylabel('S$_m$ (cosine amplitude, t$_0$ = 2 June) [cpd/kg/keVee]')
ax.grid(axis='y', lw=0.5)
fig.suptitle('NaI(Tl) prediction from the LZ 248 keV event vs the DAMA/LIBRA modulation scale', fontsize=10)
fig.tight_layout(); fig.savefig(f'{FIG}/fig1_S0_Sm_vs_DAMA.png', dpi=170); plt.close(fig)

# Fig 2: time dependence of the window rate (normalised) with cosine fits
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ymax = 0.0
for d in DELTAS:
    e = results[d][QI]
    wt = np.array([r['window_rate_cpd_kg'] for r in rows_time if r['delta_keV'] == d])
    S0, Sm = e['window']['S0'], e['window']['Sm_fixed_phase']
    ymax = max(ymax, (wt / S0).max())
    ax.plot(DAYS, wt / S0, color=COL[d], lw=1.8, label=f'δ = {d:.0f} keV (S$_m$/S$_0$ = {Sm / S0:.2f})')
    ax.plot(DAYS, 1 + Sm / S0 * np.cos(OMEGA * (DAYS - T0_DAMA)), color=COL[d], lw=1, ls='--', alpha=0.8)
ax.axvline(167, color='#898781', lw=0.8, ls=':'); ax.text(169, 0.93 * 1.08 * ymax, '16 June', color='#52514e', fontsize=8)
ax.set_xlabel('day of year'); ax.set_ylabel('rate in iodine window / annual mean'); ax.set_xlim(0, 365); ax.set_ylim(0, 1.08 * ymax)
ax.legend(frameon=False, fontsize=8, loc='upper right'); ax.grid(axis='y', lw=0.5)
ax.set_title('Iodine inelastic rate through the year (solid) and its first harmonic (dashed)', fontsize=9.5)
fig.tight_layout(); fig.savefig(f'{FIG}/fig2_time_dependence.png', dpi=170); plt.close(fig)

# Fig 3: true-energy iodine spectra at the LZ best fit
fig, ax = plt.subplots(figsize=(6.4, 3.6))
for d in DELTAS:
    I = IOD[d]
    ax.plot(I['E'], I['R_ann'] * norm[d]['scale_to_1_event'] * F_I * KGD_PER_TYR, color=COL[d], lw=1.8, label=f'δ = {d:.0f} keV')
ax.set_yscale('log'); ax.set_xlabel('iodine recoil energy E$_R$ [keV]'); ax.set_ylabel('dR/dE$_R$ [events / (t·yr NaI · keV)]')
ax.legend(frameon=False, fontsize=8); ax.grid(axis='y', lw=0.5); ax.set_ylim(1e-7, 1e-1)
ax.set_title('Iodine recoil spectra (annual mean) normalised to LZ\'s one event', fontsize=9.5)
fig.tight_layout(); fig.savefig(f'{FIG}/fig3_true_energy_spectra.png', dpi=170); plt.close(fig)

print('done in %.1f s' % (time.time() - T0))
