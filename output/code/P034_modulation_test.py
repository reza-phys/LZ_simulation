"""
P034 -- Annual modulation as the decisive test for inelastic dark matter near the kinematic edge.

Run from the simulation root:  .venv/bin/python output/code/P034_modulation_test.py

What is computed
  1. Time PDFs p(t|delta) of the in-ROI LZ rate over the year for O1 isoscalar inelastic DM,
     m = 1000 GeV, delta = 300/330/350/366/380 keV (WimPyDD shell-model kernels x Baxter-2021
     Earth-frame halo evaluated on 36 days, periodic cubic spline), with the Fig. S2 efficiency.
     Fourier decomposition, duty cycle, modulation fraction.
  2. Test (i)  cosine-amplitude estimator (known phase) vs a time-flat signal.
  3. Test (ii) unbinned time-likelihood ratio p(t|delta) vs flat: exact N-fold convolution of
     the single-event statistic (FFT) + 1e4-toy cross-checks; N and exposure for 3 sigma/5 sigma.
  4. Test (iii) summer-window binomial counting test.
  5. Discovery against background only (rate + time), gain from time information.
  6. Degradations: halo variants (v_esc, v_0), efficiency roll-off variants, livetime gaps.
  7. Calendar translation for LZ (4.71 t, live fraction 0.593) and a 60 t detector.

Normalisation: every delta is normalised to LZ's best fit of 1.0 event per 2.84 t yr
(68% band 0.30-2.36 events, Table I / Poisson n=1), so the required exposure is
E = N_req x 2.84 t yr / mu.  The absolute WimPyDD rate at unit coupling is only used
for (a) the halo-rescaling discussion (fixed-coupling models) and (b) validation vs P018/P020.
"""
import sys, os, json, math, time
import numpy as np
from scipy import stats
from scipy.interpolate import CubicSpline
from scipy.fft import rfft, irfft, next_fast_len
sys.path.insert(0, 'output/code')
from common import lzcommon as lz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

T0 = time.time()
OUT = 'output/work/P034'; FIG = OUT + '/figures'
os.makedirs(FIG, exist_ok=True)
LOG = open(f'{OUT}/run_log.txt', 'w')
def log(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.write(s + '\n'); LOG.flush()

rng = np.random.default_rng(20260909)

# ----------------------------------------------------------------------------------------------
# 0. Inputs
# ----------------------------------------------------------------------------------------------
EXPO_LZ = lz.LZ['exposure_tyr']            # 2.84 t yr  [paper]
M_FID = 4.71                               # t          [paper]
LIVE_FRAC = 220.0 / 371.0                  # 0.593      [paper: 220 live d in 371 calendar d; P020]
TYR_PER_YEAR_LZ = M_FID * LIVE_FRAC        # 2.79 t yr per calendar year
MU_BEST, MU_LO, MU_HI = 1.0, 0.30, 2.36    # best fit and 68% band (events per 2.84 t yr) [paper Table I; P020]
B_PANEL, B_NB = 0.0106, 2.0e-4             # background per 2.84 t yr: whole S1c>500 phd panel / neighbourhood [paper Fig. 5; dossier]
M_CHI = 1000.0
DELTAS = [300.0, 330.0, 350.0, 366.0, 380.0]
YEAR = 365.25
M60_TYR_PER_YEAR = 60.0 * 0.80             # 60 t detector, assumed 80% live fraction -> 48 t yr / yr (assumption)
LZ_START_NEW = 2024.25                     # new data from 1 April 2024 [paper Conclusion; P020]

RES = dict(inputs=dict(exposure_LZ=EXPO_LZ, M_fid=M_FID, live_frac=LIVE_FRAC, tyr_per_year_LZ=TYR_PER_YEAR_LZ,
                       mu=[MU_BEST, MU_LO, MU_HI], b_panel=B_PANEL, b_nb=B_NB, m_chi=M_CHI, deltas=DELTAS,
                       m60_tyr_per_year=M60_TYR_PER_YEAR))

# ----------------------------------------------------------------------------------------------
# 1. WimPyDD kernels and time PDFs
# ----------------------------------------------------------------------------------------------
WD = lz.wd()
VGRID = np.linspace(0.0, 900.0, 1801)       # explicit v_min grid (0.5 km/s), beyond v_max for every halo variant
ONES = np.ones_like(VGRID)
E_GRID = np.arange(60.0, 400.1, 2.0)        # keV; inelastic onsets >= 85 keV for delta >= 300 keV (P002)
HAM = lz.wd_hamiltonian('O1s', {1: (1.0, 0.0)})

def kernel(delta):
    """K[E, v_i] in events/(t yr keV) per unit delta_eta stream weight; dR/dE = K @ delta_eta."""
    K = np.array([WD.diff_rate(WD.Xe, HAM, M_CHI, float(e), VGRID, ONES, j_chi=0.5, delta=float(delta),
                               sum_over_streams=False) for e in E_GRID])
    return K * 1000.0 * 365.25

def efficiency(E, edge=lz.LZ['E_50pct_high_keV'], width=15.0, plateau=lz.LZ['eff_plateau']):
    """LZ NR efficiency (P006/P020 parametrisation of Fig. S2): 50% at 5.4 keV, plateau 0.96,
    roll-off Phi((edge - E)/width) with edge = 269.9 keV, width 15 keV.  edge=None: no roll-off."""
    E = np.asarray(E, float)
    lo = 1.0 / (1.0 + np.exp(-(E - lz.LZ['E_50pct_low_keV']) / 1.0))
    hi = np.ones_like(E) if edge is None else stats.norm.cdf((edge - E) / width)
    return plateau * lo * hi

EFF_VARIANTS = {'nominal': dict(edge=269.9, width=15.0), 'sharp5': dict(edge=269.9, width=5.0),
                'edge250': dict(edge=250.0, width=15.0), 'edge290': dict(edge=290.0, width=15.0),
                'none': dict(edge=None, width=15.0)}
HALO_VARIANTS = {'std': dict(), 'vesc528': dict(vesc=528.0), 'vesc560': dict(vesc=560.0),
                 'v0_220': dict(v0=220.0), 'v0_250': dict(v0=250.0)}

DAYS = np.linspace(1.0, 366.0, 37)[:-1]     # 36 days, spacing 10.14 d
DAYS = np.unique(np.concatenate([DAYS, [152.0, 167.9]]))
T_FINE = np.arange(0.0, YEAR, 0.25)         # 1461 points, day-of-year from 0
N_T = len(T_FINE)

log(f'[1] kernels: {len(DELTAS)} deltas x {len(E_GRID)} energies x {len(VGRID)} streams')
KER = {d: kernel(d) for d in DELTAS}
log(f'    kernels done in {time.time()-T0:.1f} s')
HALOS = {hv: {d: lz.wd_halo(day_of_year=float(d), vmin=VGRID, **kw)[1] for d in DAYS} for hv, kw in HALO_VARIANTS.items()}
HALO_AVG = {hv: lz.wd_halo(vmin=VGRID, **kw)[1] for hv, kw in HALO_VARIANTS.items()}

def rate_curve(delta, hv='std', ev='nominal'):
    """In-ROI rate (events/t/yr) on DAYS and as a periodic spline on T_FINE; also unit-coupling
    annual-average events per 2.84 t yr (for normalisation checks)."""
    eff = efficiency(E_GRID, **EFF_VARIANTS[ev])
    K = KER[delta]
    R = np.array([np.trapezoid(np.clip(K @ HALOS[hv][d], 0, None) * eff, E_GRID) for d in DAYS])
    dd = np.concatenate([DAYS - YEAR, DAYS, DAYS + YEAR]); rr = np.concatenate([R, R, R])
    spl = CubicSpline(dd, rr)
    Rf = np.clip(spl(T_FINE + 1.0), 0, None)          # T_FINE=0 <-> day 1
    R_avg = np.trapezoid(np.clip(K @ HALO_AVG[hv], 0, None) * eff, E_GRID)
    return R, Rf, R_avg

def pdf_from_rate(Rf, live=None):
    """Normalised time PDF over live time (live: array of 0/1 on T_FINE; None = uniform)."""
    L = np.ones(N_T) if live is None else live.astype(float)
    p = Rf * L; p = p / (p.sum() * (T_FINE[1] - T_FINE[0]))
    p0 = L / (L.sum() * (T_FINE[1] - T_FINE[0]))
    return p, p0

def fourier(p, nharm=6):
    """p(t) = (1/T)[1 + sum_k a_k cos(k w (t - t1)) ...]; return amplitudes |c_k| (fraction of mean),
    phase of the fundamental (day of maximum) and fraction of power in the fundamental."""
    dt = T_FINE[1] - T_FINE[0]
    w = 2 * np.pi / YEAR
    amps, phases = [], []
    for k in range(1, nharm + 1):
        ck = np.sum(p * np.exp(-1j * k * w * T_FINE)) * dt * 2.0     # complex amplitude relative to mean (=1)
        amps.append(abs(ck)); phases.append(np.angle(ck))
    amps = np.array(amps)
    t1 = (-phases[0] / w) % YEAR + 1.0        # day-of-year of the fundamental maximum
    total_var = np.sum((p * YEAR - 1.0) ** 2) * dt / YEAR   # variance of T p(t) about 1 = sum_k a_k^2/2
    frac1 = (amps[0] ** 2 / 2.0) / total_var
    return amps, t1, frac1, total_var

def describe(delta, Rf, R_avg):
    p, p0 = pdf_from_rate(Rf)
    amps, t1, frac1, var = fourier(p)
    r = Rf / Rf.mean()
    return dict(delta=delta, peak_day=float(T_FINE[np.argmax(Rf)] + 1), t1_fundamental=float(t1),
                mod_fraction=float((Rf.max() - Rf.min()) / (Rf.max() + Rf.min())),
                june_dec=float(Rf[np.argmin(abs(T_FINE + 1 - 167.9))] / max(Rf[np.argmin(abs(T_FINE + 1 - 350))], 1e-300)),
                duty10=float(np.mean(Rf > 0.10 * Rf.max())), duty1=float(np.mean(Rf > 0.01 * Rf.max())),
                zero_frac=float(np.mean(Rf <= 1e-4 * Rf.max())), peak_over_mean=float(r.max()),
                a1=float(amps[0]), a2=float(amps[1]), a3=float(amps[2]), a4=float(amps[3]),
                power_frac_fundamental=float(frac1), KL_nats=float(np.sum(p[p > 0] * np.log(p[p > 0] / p0[p > 0])) * 0.25),
                N_unit_per_2p84_annualmean=float(Rf.mean() * EXPO_LZ), N_unit_per_2p84_avghalo=float(R_avg * EXPO_LZ))

CURVES = {}    # (delta, hv, ev) -> (R_days, Rf, R_avg)
for d in DELTAS:
    for hv in HALO_VARIANTS:
        CURVES[(d, hv, 'nominal')] = rate_curve(d, hv, 'nominal')
    for ev in EFF_VARIANTS:
        if ev != 'nominal':
            CURVES[(d, 'std', ev)] = rate_curve(d, 'std', ev)
log(f'    rate curves done in {time.time()-T0:.1f} s')

# interpolation check: 73-day grid vs 36-day spline at delta = 380
DAYS73 = np.linspace(1.0, 366.0, 74)[:-1]
h73 = {d: lz.wd_halo(day_of_year=float(d), vmin=VGRID)[1] for d in DAYS73}
eff_nom = efficiency(E_GRID)
R73 = np.array([np.trapezoid(np.clip(KER[380.0] @ h73[d], 0, None) * eff_nom, E_GRID) for d in DAYS73])
Rf380 = CURVES[(380.0, 'std', 'nominal')][1]
interp_dev = np.max(np.abs(np.interp(DAYS73, T_FINE + 1, Rf380) - R73)) / R73.max()
log(f'    interpolation check (delta=380, 73-day grid vs 36-day spline): max dev {interp_dev:.3e} of peak')

TABLE1 = []
for d in DELTAS:
    R, Rf, R_avg = CURVES[(d, 'std', 'nominal')]
    row = describe(d, Rf, R_avg); TABLE1.append(row)
    log(f"    delta={d:5.0f}: peak doy {row['peak_day']:.0f} (fund. {row['t1_fundamental']:.1f}), mod.frac {row['mod_fraction']:.3f}, "
        f"June/Dec {row['june_dec']:.3g}, duty(10%) {row['duty10']:.2f}, zero {row['zero_frac']:.2f}, peak/mean {row['peak_over_mean']:.2f}, "
        f"a1 {row['a1']:.3f} a2 {row['a2']:.3f} a3 {row['a3']:.3f}, P1 {row['power_frac_fundamental']:.2f}, KL {row['KL_nats']:.3f} nats, "
        f"N_unit {row['N_unit_per_2p84_annualmean']:.4g}")
RES['time_pdf_table'] = TABLE1
import csv
with open(f'{OUT}/time_pdf_summary.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(TABLE1[0].keys())); w.writeheader(); w.writerows(TABLE1)
with open(f'{OUT}/time_pdfs.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['doy'] + [f'p_delta{int(d)}' for d in DELTAS] + ['p_flat'])
    P = {d: pdf_from_rate(CURVES[(d, 'std', 'nominal')][1])[0] for d in DELTAS}
    for i in range(0, N_T, 4):
        w.writerow([f'{T_FINE[i]+1:.2f}'] + [f'{P[d][i]:.6e}' for d in DELTAS] + [f'{1/YEAR:.6e}'])

# ----------------------------------------------------------------------------------------------
# 2. Generic machinery: exact N-fold distribution of a per-event statistic
# ----------------------------------------------------------------------------------------------
DT = T_FINE[1] - T_FINE[0]

def single_event_pmf(stat, p_sig, p_null, nbins=800, dead_mask=None):
    """Bin a per-event statistic stat(t) (array on T_FINE) into a common grid; return
    (values, pmf_sig, pmf_null, q_dead_null, q_dead_sig).  dead_mask marks times where the
    statistic is -inf (event impossible under the signal); such events are tracked as an
    absorbing 'dead' state."""
    fin = np.ones(N_T, bool) if dead_mask is None else ~dead_mask
    s = stat[fin]; smin, smax = s.min(), s.max()
    if smax - smin < 1e-12: smax = smin + 1e-6
    edges = np.linspace(smin, smax, nbins + 1); vals = 0.5 * (edges[1:] + edges[:-1])
    ws = p_sig[fin] * DT; wn = p_null[fin] * DT
    pmf_s, _ = np.histogram(s, bins=edges, weights=ws); pmf_n, _ = np.histogram(s, bins=edges, weights=wn)
    qd_n = 1.0 - wn.sum(); qd_s = 1.0 - ws.sum()
    return vals, pmf_s, pmf_n, max(qd_n, 0.0), max(qd_s, 0.0)

def nfold(pmf, vals, N):
    """Exact distribution of the sum of N iid copies (FFT); returns (sum_values, pmf_N)."""
    L = len(pmf); M = N * (L - 1) + 1; nfft = next_fast_len(M)
    tot = pmf.sum()
    h = vals[1] - vals[0]
    if not tot > 0: return vals[0] * N + h * np.arange(M), np.zeros(M)
    F = rfft(pmf / tot, nfft)
    out = np.clip(irfft(F ** N, nfft)[:M], 0, None) * tot ** N
    h = vals[1] - vals[0]
    return vals[0] * N + h * np.arange(M), out

def z_of_p(p):
    p = min(max(p, 1e-300), 1.0)
    return float(stats.norm.isf(p))

def exact_test(stat, p_sig, p_null, N, dead_mask=None, nbins=800):
    """Median of sum_i stat(t_i) under the signal and the null tail probability at that median.
    Returns dict(median, p, Z)."""
    vals, ps, pn, qd_n, qd_s = single_event_pmf(stat, p_sig, p_null, nbins, dead_mask)
    sv, PS = nfold(ps, vals, N)
    cs = np.cumsum(PS) / max(PS.sum(), 1e-300)
    imed = int(np.searchsorted(cs, 0.5)); med = sv[min(imed, len(sv) - 1)]
    sv2, PN = nfold(pn, vals, N)
    # P_null(sum >= med AND all N events alive): pn sums to (1-qd_n), so nfold already carries
    # the factor (1-qd_n)^N via tot**N; an event in a dead zone makes the sum -inf (< med).
    tail = PN[sv2 >= med - 1e-9].sum()
    return dict(median=float(med), p=float(tail), Z=z_of_p(tail))

N_LIST = list(range(1, 41)) + list(range(45, 101, 5)) + list(range(110, 301, 10)) + list(range(320, 1001, 20)) + list(range(1050, 3001, 50))

def scan_N(stat, p_sig, p_null, dead_mask=None, targets=(3.0, 5.0), nmax=3000):
    """Z(N) on N_LIST until Z > 5.3; smallest N reaching each target (linear interpolation in N)."""
    zs = []
    for N in N_LIST:
        if N > nmax: break
        r = exact_test(stat, p_sig, p_null, N, dead_mask)
        zs.append((N, r['Z'], r['median'], r['p']))
        if r['Z'] > max(targets) + 0.3 and N >= 3: break
    zs = np.array(zs)
    out = dict(N=zs[:, 0].tolist(), Z=zs[:, 1].tolist())
    for tgt in targets:
        idx = np.where(zs[:, 1] >= tgt)[0]
        if len(idx) == 0:
            out[f'N_{int(tgt)}sigma'] = float('inf')
        else:
            i = idx[0]
            if i == 0: out[f'N_{int(tgt)}sigma'] = float(zs[0, 0])
            else:
                n0, z0, n1, z1 = zs[i - 1, 0], zs[i - 1, 1], zs[i, 0], zs[i, 1]
                out[f'N_{int(tgt)}sigma'] = float(n0 + (tgt - z0) / (z1 - z0) * (n1 - n0))
    return out

def sample_times(p, n, size):
    cdf = np.cumsum(p) * DT; cdf /= cdf[-1]
    u = rng.random((size, n))
    return np.interp(u, cdf, T_FINE)

# per-event statistics
def stat_cos(t_peak):
    return np.cos(2 * np.pi * (T_FINE + 1 - t_peak) / YEAR)

def stat_llr(p_sig, p_null, floor_ratio=1e-4):
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(p_null > 0, p_sig / np.where(p_null > 0, p_null, 1.0), 1.0)   # gap bins carry zero weight
    dead = (r < floor_ratio) & (p_null > 0)
    s = np.log(np.where(dead, floor_ratio, r))
    return s, dead

# ----------------------------------------------------------------------------------------------
# 3. Tests (i)-(iii) vs a time-flat signal, nominal halo/efficiency, uniform livetime
# ----------------------------------------------------------------------------------------------
log('[3] tests vs time-flat alternative')
TEST = {}
TOYS = {}
NTOY = 10000
for d in DELTAS:
    Rf = CURVES[(d, 'std', 'nominal')][1]
    p, p0 = pdf_from_rate(Rf)
    row = [r for r in TABLE1 if r['delta'] == d][0]
    t_peak = row['t1_fundamental']
    # (i) cosine estimator: known phase = fundamental phase of the model
    c = stat_cos(t_peak)
    Ec_null = float(np.sum(c * p0) * DT); Vc_null = float(np.sum(c ** 2 * p0) * DT - Ec_null ** 2)
    Ec_sig = float(np.sum(c * p) * DT); Vc_sig = float(np.sum(c ** 2 * p) * DT - Ec_sig ** 2)
    a1 = row['a1']
    N3_cos_asym = 2 * (3.0 / a1) ** 2; N5_cos_asym = 2 * (5.0 / a1) ** 2      # Z = a1 sqrt(N/2)
    cos_scan = scan_N(c, p, p0)
    # (ii) unbinned LLR
    s_llr, dead = stat_llr(p, p0)
    E_llr_sig = float(np.sum(s_llr * p) * DT); E_llr_null = float(np.sum(np.where(dead, 0, s_llr) * p0) * DT)   # (finite part)
    V_llr_null = float(np.sum(np.where(dead, 0, s_llr ** 2) * p0) * DT - E_llr_null ** 2)
    N3_gauss = 9.0 * V_llr_null / (E_llr_sig - E_llr_null) ** 2      # P006-style Gaussian estimate (no dead-zone bonus)
    llr_scan = scan_N(s_llr, p, p0, dead_mask=dead)
    # (iii) summer-window counting
    win = {}
    for name, (d0, d1) in {'May-Aug': (121, 243), 'May-Jul': (121, 212), 'Jun': (152, 181)}.items():
        m = ((T_FINE + 1) >= d0) & ((T_FINE + 1) <= d1 + 0.999)
        win[name] = m
    win['optimal(LR>1)'] = p > p0
    win_res = {}
    for name, m in win.items():
        f0 = float(np.sum(p0[m]) * DT); f1 = float(np.sum(p[m]) * DT)
        res = dict(f_flat=f0, f_sig=f1)
        for tgt, alpha in ((3, stats.norm.sf(3.0)), (5, stats.norm.sf(5.0))):
            Nreq = None
            for N in range(1, 5001):
                kcrit = int(stats.binom.isf(alpha, N, f0)) + 1     # smallest k with P(K>=k|f0) <= alpha
                if stats.binom.sf(kcrit - 1, N, f0) > alpha: kcrit += 1
                power = stats.binom.sf(kcrit - 1, N, f1)
                if power >= 0.5: Nreq = N; break
            res[f'N_{tgt}sigma'] = Nreq if Nreq is not None else float('inf')
        win_res[name] = res
    TEST[d] = dict(t_peak_used=t_peak, cos=dict(E_null=Ec_null, V_null=Vc_null, E_sig=Ec_sig, V_sig=Vc_sig, a1=a1,
                                                N3_asymptotic=N3_cos_asym, N5_asymptotic=N5_cos_asym, scan=cos_scan),
                   llr=dict(E_sig=E_llr_sig, E_null_finite=E_llr_null, V_null_finite=V_llr_null, dead_frac_null=float(np.sum(p0[dead]) * DT),
                            N3_gaussian=N3_gauss, scan=llr_scan),
                   window=win_res)
    log(f"    delta={d:5.0f}: cos a1={a1:.3f} N3/N5 exact {cos_scan['N_3sigma']:.1f}/{cos_scan['N_5sigma']:.1f} (asym {N3_cos_asym:.1f}/{N5_cos_asym:.1f}); "
        f"LLR N3/N5 {llr_scan['N_3sigma']:.1f}/{llr_scan['N_5sigma']:.1f} (Gauss N3 {N3_gauss:.1f}; dead frac {TEST[d]['llr']['dead_frac_null']:.2f}); "
        f"window May-Aug f {win_res['May-Aug']['f_flat']:.3f}->{win_res['May-Aug']['f_sig']:.3f} N3/N5 {win_res['May-Aug']['N_3sigma']}/{win_res['May-Aug']['N_5sigma']}; "
        f"optimal N3/N5 {win_res['optimal(LR>1)']['N_3sigma']}/{win_res['optimal(LR>1)']['N_5sigma']}")
    # toy cross-check (1e4 toys) for N = 2..30
    toy = {}
    for N in (2, 3, 5, 10, 20, 30):
        ts = sample_times(p, N, NTOY); tn = sample_times(p0, N, NTOY)
        ls = np.interp(ts, T_FINE, s_llr).sum(1) - 1e9 * np.interp(ts, T_FINE, dead.astype(float)).round().sum(1)
        ln = np.interp(tn, T_FINE, s_llr).sum(1) - 1e9 * np.interp(tn, T_FINE, dead.astype(float)).round().sum(1)
        med = np.median(ls); p_toy = np.mean(ln >= med)
        cs_ = np.interp(ts, T_FINE, c).sum(1); cn_ = np.interp(tn, T_FINE, c).sum(1)
        medc = np.median(cs_); pc_toy = np.mean(cn_ >= medc)
        ex = exact_test(s_llr, p, p0, N, dead); exc = exact_test(c, p, p0, N)
        toy[N] = dict(llr_median_toy=float(med), llr_median_exact=ex['median'], llr_p_toy=float(p_toy), llr_p_exact=ex['p'],
                      cos_median_toy=float(medc), cos_median_exact=exc['median'], cos_p_toy=float(pc_toy), cos_p_exact=exc['p'])
    TOYS[d] = toy
RES['tests_vs_flat'] = TEST; RES['toy_checks'] = TOYS
log(f'    tests done in {time.time()-T0:.1f} s')

# ----------------------------------------------------------------------------------------------
# 4. Exposure translation
# ----------------------------------------------------------------------------------------------
def exposure_rows(N, mu=MU_BEST):
    E = N * EXPO_LZ / mu
    return dict(N=N, E_tyr=E, LZ_years=E / TYR_PER_YEAR_LZ, LZ_multiples=E / EXPO_LZ,
                year_reached=(LZ_START_NEW + (E - EXPO_LZ) / TYR_PER_YEAR_LZ) if E > EXPO_LZ else 2024.25,
                det60t_years=E / M60_TYR_PER_YEAR)

EXPO = []
for d in DELTAS:
    for test, Nkey in (('cosine', TEST[d]['cos']['scan']), ('LLR', TEST[d]['llr']['scan']),
                       ('window May-Aug', TEST[d]['window']['May-Aug']), ('window optimal', TEST[d]['window']['optimal(LR>1)'])):
        for sig in (3, 5):
            N = Nkey[f'N_{sig}sigma']
            for mu, tag in ((MU_BEST, 'best'), (MU_HI, 'hi68'), (MU_LO, 'lo68')):
                r = exposure_rows(N, mu); r.update(delta=d, test=test, sigma=sig, rate=tag); EXPO.append(r)
with open(f'{OUT}/required_exposure.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(EXPO[0].keys())); w.writeheader(); w.writerows(EXPO)
for r in EXPO:
    if r['rate'] == 'best' and r['test'] in ('LLR', 'cosine'):
        log(f"    [E] delta={r['delta']:.0f} {r['test']:7s} {r['sigma']}sigma: N={r['N']:.1f} E={r['E_tyr']:.1f} t yr = {r['LZ_years']:.1f} LZ-yr -> year {r['year_reached']:.1f}; 60 t: {r['det60t_years']:.2f} yr")

# ----------------------------------------------------------------------------------------------
# 5. Discovery against background only: rate-only vs rate+time
# ----------------------------------------------------------------------------------------------
log('[5] discovery vs background-only')
def discovery_scan(d, b0, mu=MU_BEST, k_grid=np.arange(0.25, 20.01, 0.25), ntoy=20000):
    Rf = CURVES[(d, 'std', 'nominal')][1]; p, p0 = pdf_from_rate(Rf)
    r = p / p0                      # T p(t)
    out = []
    for k in k_grid:
        s, b = mu * k, b0 * k
        u = np.log1p((s / b) * r)   # per-event contribution
        # H1 toys: N ~ Poisson(s+b); times from mixture
        Ntoy = rng.poisson(s + b, ntoy)
        q = np.full(ntoy, -2.0 * s)
        nmax = Ntoy.max()
        if nmax > 0:
            mix = (s * p + b * p0) / (s + b)
            tt = sample_times(mix, nmax, ntoy)
            uu = np.interp(tt, T_FINE, u)
            mask = np.arange(nmax)[None, :] < Ntoy[:, None]
            q += 2.0 * (uu * mask).sum(1)
        q_med = float(np.median(q)); N_med = int(np.median(Ntoy))
        # H0 exact: P(q >= q_med) = sum_n Poi(n|b) P_n(sum u >= q_med/2 + s)
        thr = q_med / 2.0 + s
        vals, _, pn, _, _ = single_event_pmf(u, p, p0, 600)
        p_time = 0.0
        for n in range(0, 30):
            pois = stats.poisson.pmf(n, b)
            if n == 0:
                p_time += pois * (1.0 if 0.0 >= thr - 1e-9 else 0.0); continue
            sv, PN = nfold(pn, vals, n)
            p_time += pois * PN[sv >= thr - 1e-9].sum()
        p_count = float(stats.poisson.sf(N_med - 1, b)) if N_med > 0 else 1.0
        out.append(dict(k=float(k), E_tyr=float(k * EXPO_LZ), s=s, b=b, N_median=N_med, q_median=q_med,
                        p_time=float(p_time), Z_time=z_of_p(p_time), p_count=p_count, Z_count=z_of_p(p_count)))
    return out

DISC = {}
for b0, btag in ((B_PANEL, 'panel'), (B_NB, 'neighbourhood')):
    for d in DELTAS:
        rows = discovery_scan(d, b0)
        def first_k(rows, key, tgt):
            for r in rows:
                if r[key] >= tgt: return r['E_tyr']
            return float('inf')
        summ = dict(E5_time=first_k(rows, 'Z_time', 5.0), E5_count=first_k(rows, 'Z_count', 5.0),
                    E3_time=first_k(rows, 'Z_time', 3.0), E3_count=first_k(rows, 'Z_count', 3.0))
        summ['gain5'] = summ['E5_count'] / summ['E5_time'] if summ['E5_time'] > 0 else float('nan')
        DISC[f'{btag}_delta{int(d)}'] = dict(summary=summ, scan=rows)
        log(f"    b0={b0:.4g} delta={d:.0f}: 5sigma exposure count-only {summ['E5_count']:.2f} t yr, with time {summ['E5_time']:.2f} t yr (gain x{summ['gain5']:.2f}); "
            f"3sigma {summ['E3_count']:.2f} / {summ['E3_time']:.2f}")
RES['discovery'] = {k: v['summary'] for k, v in DISC.items()}
with open(f'{OUT}/discovery_scan.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['bkg', 'delta'] + list(DISC['panel_delta300']['scan'][0].keys()))
    for key, v in DISC.items():
        for r in v['scan']:
            w.writerow([key.split('_')[0], key.split('delta')[1]] + list(r.values()))
# Z_count vs Z_time at LZ's own exposure multiples (k=1..5), delta = 366, panel
log(f'    done in {time.time()-T0:.1f} s')

# ----------------------------------------------------------------------------------------------
# 6. Degradations
# ----------------------------------------------------------------------------------------------
log('[6] degradations')
DEG = []
def deg_row(tag, kind, d, Rf, R_avg, live=None, t_peak=None):
    p, p0 = pdf_from_rate(Rf, live)
    amps, t1, frac1, var = fourier(p)
    tp = t1 if t_peak is None else t_peak
    c = stat_cos(tp)
    s_llr, dead = stat_llr(p, p0)
    cs = scan_N(c, p, p0); ls = scan_N(s_llr, p, p0, dead_mask=dead)
    Ec_null = float(np.sum(c * p0) * DT); Vc_null = float(np.sum(c ** 2 * p0) * DT - Ec_null ** 2)
    row = dict(kind=kind, variant=tag, delta=d, mod_fraction=float((Rf.max() - Rf.min()) / (Rf.max() + Rf.min())),
               a1=float(amps[0]), duty10=float(np.mean(Rf > 0.1 * Rf.max())), t1=float(t1),
               N3_cos=cs['N_3sigma'], N5_cos=cs['N_5sigma'], N3_llr=ls['N_3sigma'], N5_llr=ls['N_5sigma'],
               N_unit_per_2p84=float(Rf.mean() * EXPO_LZ), cos_null_bias_per_sqrtN=float(Ec_null / math.sqrt(max(Vc_null, 1e-12))))
    DEG.append(row); return row

# (a) halo variants: shape (N required) and rate (fixed-coupling exposure rescaling)
for hv in HALO_VARIANTS:
    for d in DELTAS:
        R, Rf, R_avg = CURVES[(d, hv, 'nominal')]
        if Rf.max() <= 0:
            DEG.append(dict(kind='halo', variant=hv, delta=d, mod_fraction=float('nan'), a1=float('nan'), duty10=0.0, t1=float('nan'),
                            N3_cos=float('inf'), N5_cos=float('inf'), N3_llr=float('inf'), N5_llr=float('inf'), N_unit_per_2p84=0.0, cos_null_bias_per_sqrtN=0.0))
            log(f'    halo {hv} delta={d:.0f}: kinematically forbidden (zero rate)'); continue
        row = deg_row(hv, 'halo', d, Rf, R_avg)
        std = [r for r in DEG if r['kind'] == 'halo' and r['variant'] == 'std' and r['delta'] == d][0]
        log(f"    halo {hv:8s} delta={d:.0f}: mod.frac {row['mod_fraction']:.3f} a1 {row['a1']:.3f} duty {row['duty10']:.2f} N3 llr {row['N3_llr']:.1f} cos {row['N3_cos']:.1f}; "
            f"rate/std = {row['N_unit_per_2p84']/std['N_unit_per_2p84']:.3g} (log10 {math.log10(row['N_unit_per_2p84']/std['N_unit_per_2p84']):+.2f})")
# compare rate ratios with P018 (june16 epoch; P018 used the 16 June halo)
import pandas as pd
p18 = pd.read_csv('output/work/P018/rate_ratios_isoscalar.csv')
p18 = p18[(p18.m_GeV == 1000) & (p18.epoch == 'annual')]
P18CMP = []
for hv in ('vesc528', 'vesc560', 'v0_220', 'v0_250'):
    for d in (300, 350, 380):
        mine = [r for r in DEG if r['kind'] == 'halo' and r['variant'] == hv and r['delta'] == d][0]['N_unit_per_2p84']
        std = [r for r in DEG if r['kind'] == 'halo' and r['variant'] == 'std' and r['delta'] == d][0]['N_unit_per_2p84']
        sel = p18[(p18.delta_keV == d) & (p18.halo == hv)]
        p18r = float(sel.log10_ratio.iloc[0]) if len(sel) else float('nan')
        P18CMP.append(dict(halo=hv, delta=d, log10_ratio_mine=math.log10(mine / std) if mine > 0 else -np.inf, log10_ratio_P018_annual=p18r))
RES['P018_rate_ratio_comparison'] = P18CMP
log('    P018 comparison (annual, log10 N/N_SHM): ' + '; '.join(f"{r['halo']} d{r['delta']}: {r['log10_ratio_mine']:+.2f} vs {r['log10_ratio_P018_annual']:+.2f}" for r in P18CMP))
p18s = pd.read_csv('output/work/P018/rate_spread_summary.csv'); p18s = p18s[(p18s.m_GeV == 1000) & (p18s.epoch == 'annual')]
RES['P018_gaia_spread_dex'] = {int(r.delta_keV): float(r.spread_gaia_dex) for r in p18s.itertuples()}

# (b) efficiency roll-off variants
for ev in EFF_VARIANTS:
    for d in DELTAS:
        R, Rf, R_avg = CURVES[(d, 'std', ev)]
        row = deg_row(ev, 'efficiency', d, Rf, R_avg)
        log(f"    eff {ev:8s} delta={d:.0f}: mod.frac {row['mod_fraction']:.3f} a1 {row['a1']:.3f} duty {row['duty10']:.2f} N3 llr {row['N3_llr']:.1f} cos {row['N3_cos']:.1f} N_unit {row['N_unit_per_2p84']:.4g}")

# (c) livetime gaps: 21-day dead block
GAP = 21.0
def live_mask(start_doy):
    m = np.ones(N_T)
    doy = T_FINE + 1
    end = start_doy + GAP
    if end <= YEAR + 1: m[(doy >= start_doy) & (doy < end)] = 0
    else:
        m[(doy >= start_doy) | (doy < end - YEAR)] = 0
    return m
for d in DELTAS:
    R, Rf, R_avg = CURVES[(d, 'std', 'nominal')]
    t1_nom = [r for r in TABLE1 if r['delta'] == d][0]['t1_fundamental']
    gaps = {'gap_on_peak': live_mask(t1_nom - GAP / 2), 'gap_december': live_mask(335.0), 'gap_march': live_mask(60.0)}
    for tag, L in gaps.items():
        row = deg_row(tag, 'gap', d, Rf, R_avg, live=L, t_peak=t1_nom)
        log(f"    gap {tag:12s} delta={d:.0f}: N3 llr {row['N3_llr']:.1f} cos {row['N3_cos']:.1f}; naive-cosine null bias {row['cos_null_bias_per_sqrtN']:+.3f} sigma x sqrt(N)")
    # random gap phase: average over 12 positions
    N3l, N3c, bias = [], [], []
    for st in np.linspace(1, 365, 12, endpoint=False):
        L = live_mask(st); p, p0 = pdf_from_rate(Rf, L)
        s_llr, dead = stat_llr(p, p0); c = stat_cos(t1_nom)
        N3l.append(scan_N(s_llr, p, p0, dead_mask=dead, targets=(3.0,))['N_3sigma'])
        N3c.append(scan_N(c, p, p0, targets=(3.0,))['N_3sigma'])
        Ec = float(np.sum(c * p0) * DT); Vc = float(np.sum(c ** 2 * p0) * DT - Ec ** 2); bias.append(Ec / math.sqrt(Vc))
    DEG.append(dict(kind='gap', variant='gap_random_mean12', delta=d, mod_fraction=float('nan'), a1=float('nan'), duty10=float('nan'), t1=t1_nom,
                    N3_cos=float(np.mean(N3c)), N5_cos=float('nan'), N3_llr=float(np.mean(N3l)), N5_llr=float('nan'),
                    N_unit_per_2p84=float('nan'), cos_null_bias_per_sqrtN=float(np.max(np.abs(bias)))))
    log(f"    gap random(12) delta={d:.0f}: mean N3 llr {np.mean(N3l):.1f} (range {min(N3l):.1f}-{max(N3l):.1f}) cos {np.mean(N3c):.1f}; max |bias| {np.max(np.abs(bias)):.3f} sqrt(N)")
with open(f'{OUT}/degradations.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(DEG[0].keys())); w.writeheader(); w.writerows(DEG)
RES['degradations'] = DEG
log(f'    done in {time.time()-T0:.1f} s')

# ----------------------------------------------------------------------------------------------
# 7. Figures
# ----------------------------------------------------------------------------------------------
COL = {300.0: '#2a78d6', 330.0: '#eb6834', 350.0: '#1baf7a', 366.0: '#eda100', 380.0: '#e87ba4'}
GRAY = '#8a8985'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e6e5e1', 'grid.linewidth': 0.6, 'axes.edgecolor': '#b5b4ae', 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})

fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), constrained_layout=True)
ax = axes[0]
for d in DELTAS:
    p, p0 = pdf_from_rate(CURVES[(d, 'std', 'nominal')][1])
    ax.plot(T_FINE + 1, p * YEAR, color=COL[d], lw=2, label=f'δ = {d:.0f} keV')
ax.axhline(1.0, color=GRAY, lw=1.5, ls='--', label='time-flat')
ax.axvline(167.9, color='#0b0b0b', lw=0.8, ls=':'); ax.text(170, 3.35, '16 Jun 2023', fontsize=8, color='#52514e')
ax.set_xlabel('day of year'); ax.set_ylabel('T·p(t | δ)  (rate / annual mean)'); ax.set_xlim(1, 366); ax.set_ylim(0, 3.6)
ax.set_title('Expected arrival-time density, O1 inelastic, m = 1 TeV, LZ efficiency', loc='left', fontsize=9)
ax.legend(frameon=False, fontsize=8, loc='upper right')
ax = axes[1]
for d in DELTAS:
    row = [r for r in TABLE1 if r['delta'] == d][0]
    ax.bar(np.arange(1, 5) + (DELTAS.index(d) - 2) * 0.15, [row['a1'], row['a2'], row['a3'], row['a4']], width=0.14, color=COL[d], label=f'{d:.0f} keV')
ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel('harmonic k'); ax.set_ylabel('amplitude a_k (fraction of mean rate)')
ax.set_title('Fourier amplitudes (a₁ = fundamental; pure cosine would have a₂ = 0)', loc='left', fontsize=9)
ax.legend(frameon=False, fontsize=8, title='δ', title_fontsize=8)
fig.savefig(f'{FIG}/fig1_time_pdfs.png', dpi=160); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), constrained_layout=True)
tests = [('LLR', '#2a78d6', 'unbinned time likelihood'), ('cosine', '#eb6834', 'cosine amplitude'), ('window May-Aug', '#1baf7a', 'May–Aug counting')]
for ax, sig in zip(axes, (3, 5)):
    for test, col, lab in tests:
        rows = sorted([r for r in EXPO if r['test'] == test and r['sigma'] == sig and r['rate'] == 'best'], key=lambda r: r['delta'])
        lo = sorted([r for r in EXPO if r['test'] == test and r['sigma'] == sig and r['rate'] == 'hi68'], key=lambda r: r['delta'])
        hi = sorted([r for r in EXPO if r['test'] == test and r['sigma'] == sig and r['rate'] == 'lo68'], key=lambda r: r['delta'])
        x = [r['delta'] for r in rows]
        ax.plot(x, [r['E_tyr'] for r in rows], 'o-', color=col, lw=2, ms=5, label=lab)
        if test == 'LLR':
            ax.fill_between(x, [r['E_tyr'] for r in lo], [r['E_tyr'] for r in hi], color=col, alpha=0.15, lw=0, label='68% rate band (LLR)')
    ax.set_yscale('log'); ax.set_xlabel('δ (keV)'); ax.set_ylabel('exposure for median %dσ vs time-flat (t·yr)' % sig)
    ax.axhline(EXPO_LZ, color=GRAY, lw=1, ls='--'); ax.text(301, EXPO_LZ * 1.1, 'LZ 2.84 t·yr', fontsize=8, color='#52514e')
    ax.axhline(EXPO_LZ + TYR_PER_YEAR_LZ * (2028.0 - LZ_START_NEW), color=GRAY, lw=1, ls=':'); ax.text(301, (EXPO_LZ + TYR_PER_YEAR_LZ * (2028.0 - 2024.25)) * 1.1, 'LZ to end-2027', fontsize=8, color='#52514e')
    ax.set_title(f'{sig}σ time dependence: required exposure (best-fit rate 1 event / 2.84 t·yr)', loc='left', fontsize=9)
    ax.set_ylim(1, 3000); ax.set_xlim(295, 385)
    ax2 = ax.secondary_yaxis('right', functions=(lambda E: E / TYR_PER_YEAR_LZ, lambda y: y * TYR_PER_YEAR_LZ)); ax2.set_ylabel('LZ calendar years (2.79 t·yr/yr)')
    ax2.spines['right'].set_visible(True)
axes[0].legend(frameon=False, fontsize=8, loc='upper right')
fig.savefig(f'{FIG}/fig2_required_exposure.png', dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
for d in DELTAS:
    sc = TEST[d]['llr']['scan']; ax.plot(sc['N'], sc['Z'], '-', color=COL[d], lw=2, label=f'LLR, δ = {d:.0f} keV')
    sc = TEST[d]['cos']['scan']; ax.plot(sc['N'], sc['Z'], ':', color=COL[d], lw=1.5)
ax.plot([], [], ':', color='#0b0b0b', label='cosine estimator (dotted)')
for z in (3, 5): ax.axhline(z, color=GRAY, lw=0.8, ls='--')
ax.set_xscale('log'); ax.set_xlim(1, 1000); ax.set_ylim(0, 6); ax.set_xlabel('number of high-energy NR events N'); ax.set_ylabel('median significance vs time-flat (σ)')
ax.set_title('Median Z(N) against a time-flat population (exact N-fold convolution)', loc='left', fontsize=9)
ax.legend(frameon=False, fontsize=8)
fig.savefig(f'{FIG}/fig3_Z_vs_N.png', dpi=160); plt.close(fig)

# ----------------------------------------------------------------------------------------------
# 8. Save
# ----------------------------------------------------------------------------------------------
def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (isinstance(o, float) and (math.isnan(o))) else (float(o) if not math.isinf(float(o)) else 'inf')
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.ndarray): return clean(o.tolist())
    return o
RES['interp_check_delta380_maxdev_of_peak'] = float(interp_dev)
RES['runtime_s'] = time.time() - T0
json.dump(clean(RES), open(f'{OUT}/P034_results.json', 'w'), indent=1)
log(f'[done] {time.time()-T0:.1f} s; results in {OUT}')
