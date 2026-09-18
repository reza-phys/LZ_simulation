#!/usr/bin/env python
"""
P021 -- A full likelihood fit of the single LZ event in (m_chi, delta, sigma): where does inelastic DM prefer to sit?

Model (observed recoil energy E_obs, 2.84 t yr):
  dN/dE_obs(kappa) = kappa * EXPO * [ (dR/dE_true)_WimPyDD(unit LZ coupling) * eps(E_true) ] (x) Gauss(sigma_E(E_true))
  kappa = (c_1 m_v^2)^2 in LZ/Anand normalisation (WimPyDD c^0 = 2/m_v^2 for kappa = 1; lz.wd_c_from_anand).
  Efficiency: 0.96 plateau, erf edges 50% at 5.4 keV (sigma 2.5) and 269.9 keV (sigma 11.5 keV, Fig. S2 inset; P007).
  Resolution: sigma_E(E) = sigma_248 sqrt(E/248), sigma_248 = 11 keV baseline (P009), 8 and 15 keV variants.
Likelihood (extended, following P016 for the low-energy part):
  * Fig. 5 top panel (S1c < 250 phd ~ 5.4-125 keV NR): 0.5 sigma_NR bins from -8 to +2 sigma, model b_i (x theta, ER scale,
    Gaussian 30% constraint) and observed n_i (digitised by P016), signal fraction g_i = Gaussian in sigma_NR units;
  * 125-200 keV bin: b_M = 0.02, n = 0 (P016);
  * 200-290 keV: unbinned; background b_H = 5.7e-4 events flat in 200-270 keV (P016 anchor); one event at E_obs = 248 keV
    (variants 246, 262 keV from P009).
  q0 = 2 ln[L(kappa_hat)/L(0)], Z = sqrt(q0) (asymptotic, on P016's anchor scale; exact single-event p-value also given);
  68/90% intervals from the profile ratio (Delta(-2 ln L) = 1, 2.706), i.e. Feldman-Cousins-like two-sided.
Grid: m = 400, 1000, 4000 GeV; delta = 100-290 keV in 10 keV steps, 300 keV to the June kinematic ceiling mu v_max^2/2 in 5 keV
steps; E_true in 3 keV steps; halos: annual average (12 monthly days), 16 June (day 167), Sun frame (WimPyDD default).
Couplings: O1 isoscalar ('s'), O1 isovector ('v'), and the P007 pure-Higgsino Z-exchange couplings ('hig', 1000 GeV, fixed).

Run from the simulation root:  .venv/bin/python output/code/P021_full_likelihood.py [--spectra s v hig] [--no-analysis]
WimPyDD kernels are cached in output/work/P021/spectra_<tag>_<m>.npz.
"""
import sys, os, json, math, time, argparse
import numpy as np
sys.path.insert(0, 'output/code')
from common import lzcommon as lz
from scipy import stats, special

T0 = time.time()
OUT = 'output/work/P021'; FIG = OUT + '/figures'
os.makedirs(FIG, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument('--spectra', nargs='*', default=['s', 'v', 'hig'])
ap.add_argument('--no-analysis', action='store_true')
ARGS = ap.parse_args()

MV = lz.M_V_GEV                      # 246.2 GeV
EXPO = lz.LZ['exposure_tyr']         # 2.84 t yr
MN = lz.M_NUCLEON_GEV
HBARC2_CM2 = 0.3894e-27              # (hbar c)^2 in GeV^2 cm^2 (recalled, certain)
MASSES = [400.0, 1000.0, 4000.0]
VGRID = np.linspace(0.0, 830.0, 1661)  # km/s, beyond v_esc + v_sun + v_orb = 809 km/s (June)
ONES = np.ones_like(VGRID)
E_T = np.arange(1.5, 340.0, 3.0)     # true recoil energies (keV), 3 keV steps
DE_T = 3.0
E_O = np.arange(0.5, 340.0, 1.0)     # observed energies (keV), 1 keV bins
DOY_JUNE = 167
V_MAX_JUNE = lz.vmax_kms(lz.v_earth_kms(DOY_JUNE))
LOG = open(f'{OUT}/run_log.txt', 'a')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n'); LOG.flush()
say(f'\n===== P021 run {time.strftime("%Y-%m-%d %H:%M:%S")} args={sys.argv[1:]} =====')

# ---------------------------------------------------------------------------------------------
# 1. halos, couplings, delta grids
# ---------------------------------------------------------------------------------------------
days12 = 15.0 + 365.25 / 12 * np.arange(12)
HALOS = {'annual': np.mean([lz.wd_halo(day_of_year=d, vmin=VGRID)[1] for d in days12], axis=0),
         'june': lz.wd_halo(day_of_year=DOY_JUNE, vmin=VGRID)[1],
         'sun': lz.wd_halo(vmin=VGRID)[1]}
hig = json.load(open('output/work/P007/higgsino_couplings.json'))
COUPLINGS = {'s': lz.wd_c_from_anand(1.0 / MV**2, 0.0),        # (c_1^s m_v^2)^2 = 1
             'v': lz.wd_c_from_anand(0.0, 1.0 / MV**2),        # (c_1^v m_v^2)^2 = 1
             'hig': (hig['c0_wimpydd'], hig['c1_wimpydd'])}    # pure Higgsino, Z exchange (P007), fixed
KAPPA_HIG = hig['c_eq_isoscalar_mv2_sq']                        # 0.0777 isoscalar-equivalent
TAG_MASSES = {'s': MASSES, 'v': MASSES, 'hig': [1000.0]}

def ceiling_kev(m, v=V_MAX_JUNE):
    return 0.5 * lz.mu_red(m, lz.m_nucleus_gev(lz.A_XE_MEAN)) * (v / lz.C_KMS) ** 2 * 1e6

def delta_grid(m):
    c = ceiling_kev(m)
    return np.concatenate([np.arange(100.0, 300.0, 10.0), np.arange(300.0, c, 5.0)])

KIN = {int(m): dict(ceiling_june=ceiling_kev(m), ceiling_sun=ceiling_kev(m, lz.vmax_kms()),
                   dmax248_june=lz.delta_max_kev(248, m, v_kms=V_MAX_JUNE), dmax248_sun=lz.delta_max_kev(248, m),
                   dmax270_june=lz.delta_max_kev(270, m, v_kms=V_MAX_JUNE)) for m in MASSES}
say('kinematics:', json.dumps({k: {kk: round(vv, 1) for kk, vv in v.items()} for k, v in KIN.items()}))

# ---------------------------------------------------------------------------------------------
# 2. WimPyDD spectra via per-stream kernels (cached)
# ---------------------------------------------------------------------------------------------
def compute_spectra(tag, m):
    path = f'{OUT}/spectra_{tag}_{int(m)}.npz'
    if os.path.exists(path):
        return dict(np.load(path))
    WD = lz.wd()
    ham = lz.wd_hamiltonian(f'O1_{tag}', {1: COUPLINGS[tag]})
    deltas = delta_grid(m)
    R = {h: np.zeros((len(deltas), len(E_T))) for h in HALOS}
    ncalls = 0
    for i, d in enumerate(deltas):
        lo, hi = lz.E_R_range_keV(m, VGRID[-1], A=124.0, delta_kev=float(d))
        if math.isnan(lo):
            continue
        for j in np.where((E_T >= lo - DE_T) & (E_T <= hi + DE_T))[0]:
            K = WD.diff_rate(WD.Xe, ham, m, float(E_T[j]), VGRID, ONES, j_chi=0.5, delta=float(d),
                             sum_over_streams=False) * 1000.0 * 365.25       # /kg/day/keV -> /t/yr/keV
            ncalls += 1
            for h, deta in HALOS.items():
                R[h][i, j] = max(0.0, float(K @ deta))
    np.savez(path, deltas=deltas, E=E_T, **R)
    say(f'  spectra {tag} m={m:.0f}: {len(deltas)} deltas, {ncalls} kernel calls, t={time.time() - T0:.0f}s')
    return dict(np.load(path))

SPEC = {}
for tag in ARGS.spectra:
    for m in TAG_MASSES[tag]:
        SPEC[(tag, int(m))] = compute_spectra(tag, m)
if ARGS.no_analysis:
    sys.exit(0)
for tag in ['s', 'v', 'hig']:
    for m in TAG_MASSES[tag]:
        if (tag, int(m)) not in SPEC:
            SPEC[(tag, int(m))] = compute_spectra(tag, m)

# ---------------------------------------------------------------------------------------------
# 3. detector model
# ---------------------------------------------------------------------------------------------
def efficiency(E, sig_hi=11.5, sig_lo=2.5, hi_edge=True):
    lo = 0.5 * (1 + special.erf((E - 5.4) / (math.sqrt(2) * sig_lo)))
    hi = 0.5 * special.erfc((E - 269.9) / (math.sqrt(2) * sig_hi)) if hi_edge else 1.0
    return 0.96 * lo * hi

def sigma_E(sig248):
    return sig248 * np.sqrt(E_T / 248.0)

SMEAR = {s: stats.norm.pdf((E_O[:, None] - E_T[None, :]) / sigma_E(s)[None, :]) / sigma_E(s)[None, :] * DE_T
         for s in (8.0, 11.0, 15.0)}
E_OBS_CUT = 272.0     # observed-space S1c<600 phd cut for the 'obscut' variant (P009: MC E50 = 271.7 keV)

class Cfg:
    def __init__(self, name, sigE=11.0, E_ev=248.0, halo='annual', obscut=False, lowE='poisson', Nmax=3.0, b_H=5.7e-4):
        self.__dict__.update(name=name, sigE=sigE, E_ev=E_ev, halo=halo, obscut=obscut, lowE=lowE, Nmax=Nmax, b_H=b_H)

CFGS = [Cfg('baseline'), Cfg('sig8', sigE=8.0), Cfg('sig15', sigE=15.0), Cfg('E246', E_ev=246.0), Cfg('E262', E_ev=262.0),
        Cfg('june', halo='june'), Cfg('sun', halo='sun'), Cfg('obscut', obscut=True), Cfg('gauss3', lowE='gauss', Nmax=3.0),
        Cfg('bH_x2', b_H=1.14e-3), Cfg('bH_half', b_H=2.85e-4)]

def observe(r, cfg):
    """r: dR/dE_true (/t/yr/keV) for kappa = 1.  Returns observed-space quantities in events (2.84 t yr) per unit kappa."""
    eps = efficiency(E_T, hi_edge=not cfg.obscut)
    w = r * eps * EXPO
    d = SMEAR[cfg.sigE] @ w                              # events / keV_obs
    sig = sigma_E(cfg.sigE)
    f_ev = float((stats.norm.pdf((cfg.E_ev - E_T) / sig) / sig * DE_T) @ w)
    if cfg.obscut:
        d = d * (E_O < E_OBS_CUT)
        if cfg.E_ev >= E_OBS_CUT:
            f_ev = 0.0
    I = lambda a, b: float(np.sum(d[(E_O > a) & (E_O < b)]))   # 1 keV bins
    S_L, S_lo55, S_M, S_H = I(5.4, 125.0), I(5.4, 55.0), I(125.0, 200.0), I(200.0, 290.0)
    S_tot = S_L + S_M + S_H
    pct = I(5.4, cfg.E_ev) / S_tot if S_tot > 0 else float('nan')
    return dict(S_L=S_L, S_lo55=S_lo55, S_M=S_M, S_H=S_H, S_tot=S_tot, f_ev=f_ev, pct_ev=pct, d=d)

# ---------------------------------------------------------------------------------------------
# 4. likelihood
# ---------------------------------------------------------------------------------------------
P16 = json.load(open('output/work/P016/P016_results.json'))['fig5_top_digitised']
lo_edges = np.array(P16['sigma_bin_low']); sel = lo_edges < 2.0 - 1e-9
B_BINS = np.array(P16['total'])[sel]; N_BINS = np.array(P16['data_counts'])[sel]
G_BINS = stats.norm.cdf(lo_edges[sel] + 0.5) - stats.norm.cdf(lo_edges[sel])
B_M, N_M, W_H, SIG_TH = 0.02, 0, 70.0, 0.3
MU_GRID = np.concatenate([[0.0], np.logspace(-4, 2.5, 651)])
say(f'Fig.5 bins used: {sel.sum()} bins, b_sum={B_BINS.sum():.1f}, n_sum={N_BINS.sum():.0f}, signal fraction covered {G_BINS.sum():.3f}')

def theta_hat(C):
    """Profile the ER-scale nuisance theta for each row of C (signal counts per bin); dlnL/dtheta is monotonic."""
    lo = np.full(C.shape[0], 1e-3); hi = np.full(C.shape[0], 30.0)
    for _ in range(60):
        th = 0.5 * (lo + hi)
        dl = np.sum(N_BINS * B_BINS / (th[:, None] * B_BINS + C), axis=1) - B_BINS.sum() - (th - 1) / SIG_TH**2
        lo = np.where(dl > 0, th, lo); hi = np.where(dl > 0, hi, th)
    return 0.5 * (lo + hi)

def lnL_prof(mu, sig, cfg):
    """Profile ln L over theta at total expected accepted signal mu (array); kappa = mu / S_tot."""
    mu = np.atleast_1d(mu).astype(float)
    k = mu / sig['S_tot'] if sig['S_tot'] > 0 else np.zeros_like(mu)
    if cfg.lowE == 'poisson':
        C = k[:, None] * sig['S_L'] * G_BINS[None, :]
        th = theta_hat(C)
        m_i = th[:, None] * B_BINS + C
        ll = np.sum(N_BINS * np.log(m_i) - m_i, axis=1) - 0.5 * ((th - 1) / SIG_TH) ** 2
        mM = B_M + k * sig['S_M']; ll = ll + N_M * np.log(mM) - mM
    else:   # Gaussian penalty replacing the Fig. 5 bins: 90% UL of Nmax signal events in 5.4-125 keV; 125-200 keV bin kept
        ll = -0.5 * (k * sig['S_L'] / (cfg.Nmax / 1.2816)) ** 2
        mM = B_M + k * sig['S_M']; ll = ll + N_M * np.log(mM) - mM
    ll = ll + np.log(cfg.b_H / W_H + k * sig['f_ev']) - (cfg.b_H + k * sig['S_H'])
    return ll

def crossings(x, y, target, i_max):
    """Lower/upper x where y (concave around i_max) crosses target; x in log-space interpolation."""
    def cross(idx_range):
        for a, b in idx_range:
            if (y[a] - target) * (y[b] - target) <= 0 and y[a] != y[b]:
                if x[a] <= 0:
                    return 0.0
                la, lb = np.log(x[a]), np.log(x[b])
                return float(np.exp(la + (target - y[a]) * (lb - la) / (y[b] - y[a])))
        return None
    lo = cross([(i - 1, i) for i in range(i_max, 0, -1)])
    hi = cross([(i, i + 1) for i in range(i_max, len(x) - 1)])
    return (0.0 if lo is None else lo), (float('inf') if hi is None else hi)

def fit(sig, cfg):
    if sig['S_tot'] <= 0 or sig['f_ev'] <= 0:
        return dict(q0=0.0, Z=0.0, mu_hat=0.0, kappa_hat=float('nan'), mu68=(0, 0), mu90=(0, 0), ll0=float('nan'))
    ll = lnL_prof(MU_GRID, sig, cfg)
    i = int(np.argmax(ll)); ll0 = ll[0]; llmax = ll[i]
    mu_hat = MU_GRID[i]
    q0 = max(0.0, 2 * (llmax - ll0)) if i > 0 else 0.0
    mu68 = crossings(MU_GRID, ll, llmax - 0.5, i) if i > 0 else (0.0, crossings(MU_GRID, ll, llmax - 0.5, 0)[1])
    mu90 = crossings(MU_GRID, ll, llmax - 2.706 / 2, i) if i > 0 else (0.0, crossings(MU_GRID, ll, llmax - 1.353, 0)[1])
    return dict(q0=q0, Z=math.sqrt(q0), mu_hat=mu_hat, kappa_hat=mu_hat / sig['S_tot'], mu68=mu68, mu90=mu90, ll0=ll0, llmax=llmax)

def sigma_n_cm2(kappa, m):
    """LZ supplement: (c_1^s m_v^2)^2 = sigma_SI pi m_v^4 / mu_N^2  ->  sigma_SI = kappa mu_N^2 / (pi m_v^4)."""
    mu = lz.mu_red(m, MN)
    return kappa * mu**2 / (math.pi * MV**4) * HBARC2_CM2

def q0_single(ftil, bd):
    """Profile q0 for one event at a point where the accepted-signal density per event is ftil and the background
    density is bd (extended likelihood, Poisson term e^{-mu}): q0 = 2[ln(ftil/bd) - 1 + bd/ftil] if ftil > bd else 0."""
    return np.where(ftil > bd, 2 * (np.log(np.maximum(ftil, 1e-300) / bd) - 1 + bd / np.maximum(ftil, 1e-300)), 0.0)

def exact_p(sig, cfg, b_H):
    """Toy-free single-event p-value for a companion-free spectrum (S_L = 0): P(exactly one background event in the
    ROI lands at an energy E with q0(E) >= q0_obs) = sum_E bd(E) 1[q0(E) >= q0_obs] dE  (+O(b^2)); p = 1 if q0_obs = 0."""
    d = sig['d'] / sig['S_tot']; ftil_ev = sig['f_ev'] / sig['S_tot']
    bd = np.where((E_O > 200) & (E_O < 270), b_H / W_H, np.where((E_O > 125) & (E_O < 200), B_M / 75.0, 0.0))
    q_obs = float(q0_single(np.array([ftil_ev]), b_H / W_H)[0])
    if q_obs <= 0:
        return 1.0
    qE = q0_single(d, np.where(bd > 0, bd, 1.0))
    return float(np.sum(bd * (qE >= q_obs) * 1.0))

# ---------------------------------------------------------------------------------------------
# 5. scans
# ---------------------------------------------------------------------------------------------
rows = []
for (tag, m), S in SPEC.items():
    if tag == 'hig':
        continue
    for cfg in CFGS:
        for i, d in enumerate(S['deltas']):
            sig = observe(S[cfg.halo][i], cfg)
            f = fit(sig, cfg)
            kh = f['kappa_hat']
            row = dict(tag=tag, m_GeV=m, cfg=cfg.name, delta_keV=float(d), S_tot_unit=sig['S_tot'], S_H_unit=sig['S_H'],
                       S_L_unit=sig['S_L'], f_ev_unit=sig['f_ev'], f_ev_norm=(sig['f_ev'] / sig['S_tot'] if sig['S_tot'] > 0 else np.nan),
                       pct_ev=sig['pct_ev'], q0=f['q0'], Z=f['Z'], mu_hat=f['mu_hat'], kappa_hat=kh,
                       kappa_68lo=f['mu68'][0] / sig['S_tot'] if sig['S_tot'] > 0 else np.nan,
                       kappa_68hi=f['mu68'][1] / sig['S_tot'] if sig['S_tot'] > 0 else np.nan,
                       kappa_90lo=f['mu90'][0] / sig['S_tot'] if sig['S_tot'] > 0 else np.nan,
                       kappa_90hi=f['mu90'][1] / sig['S_tot'] if sig['S_tot'] > 0 else np.nan,
                       mu_68lo=f['mu68'][0], mu_68hi=f['mu68'][1], mu_90lo=f['mu90'][0], mu_90hi=f['mu90'][1],
                       lowE_events_at_fit=kh * sig['S_L'] if np.isfinite(kh) else np.nan)
            row['sigma_n_hat_cm2'] = sigma_n_cm2(kh, m) if np.isfinite(kh) else np.nan
            row['sigma_n_68lo_cm2'] = sigma_n_cm2(row['kappa_68lo'], m); row['sigma_n_68hi_cm2'] = sigma_n_cm2(row['kappa_68hi'], m)
            row['sigma_n_90lo_cm2'] = sigma_n_cm2(row['kappa_90lo'], m); row['sigma_n_90hi_cm2'] = sigma_n_cm2(row['kappa_90hi'], m)
            companion_free = sig['S_L'] < 1e-3 * max(sig['S_tot'], 1e-300)
            row['companion_free'] = companion_free
            if cfg.name == 'baseline' and companion_free and sig['S_tot'] > 0:
                p1 = exact_p(sig, cfg, cfg.b_H); p2 = exact_p(sig, cfg, stats.norm.sf(3.4))
                row['p_exact_bH'] = p1; row['Z_exact_bH'] = float(stats.norm.isf(p1)) if 0 < p1 < 1 else 0.0
                row['p_exact_bHpois'] = p2; row['Z_exact_bHpois'] = float(stats.norm.isf(p2)) if 0 < p2 < 1 else 0.0
            rows.append(row)
    say(f'  scan {tag} m={m}: done t={time.time() - T0:.0f}s')

import pandas as pd
df = pd.DataFrame(rows)
df.to_csv(f'{OUT}/P021_scan_all.csv', index=False)
base = df[df.cfg == 'baseline'].copy()
base.to_csv(f'{OUT}/P021_scan_baseline.csv', index=False)

RES = dict(kinematics=KIN, fig5_bins=dict(n_bins=int(sel.sum()), b_sum=float(B_BINS.sum()), n_sum=float(N_BINS.sum()), g_sum=float(G_BINS.sum())),
           settings=dict(exposure_tyr=EXPO, b_H=5.7e-4, b_M=B_M, w_H_keV=W_H, sig_theta=SIG_TH, eff='0.96 plateau, erf 5.4 keV (2.5), 269.9 keV (11.5)',
                         sigma_E_baseline=11.0, E_obs=248.0, halo='annual (12 monthly days)', E_true_step=DE_T, E_obs_step=1.0, mu_grid='0 + logspace(-4,2.5,651)'))

# --- 5a. Table S7 comparison -----------------------------------------------------------------
comp = []
for tag in ['s', 'v']:
    for m in MASSES:
        for j, d in enumerate(lz.OSIG_DELTAS):
            if d < 100:
                continue
            zlz = lz.OSIG['O1' + tag][int(m)][j]
            sub = base[(base.tag == tag) & (base.m_GeV == m) & (np.isclose(base.delta_keV, d))]
            zm = float(sub.Z.iloc[0]) if len(sub) else np.nan
            comp.append(dict(tag=tag, m_GeV=m, delta_keV=d, Z_LZ=zlz, Z_model=zm, diff=(zm - zlz) if zlz is not None else np.nan,
                             lowE_events_at_fit=float(sub.lowE_events_at_fit.iloc[0]) if len(sub) else np.nan))
comp = pd.DataFrame(comp); comp.to_csv(f'{OUT}/P021_tableS7_comparison.csv', index=False)
ok = comp.dropna(subset=['diff'])
RES['tableS7'] = dict(rms_all=float(np.sqrt(np.mean(ok['diff'] ** 2))), mean_offset=float(ok['diff'].mean()),
                      rms_s=float(np.sqrt(np.mean(ok[ok.tag == 's']['diff'] ** 2))), rms_v=float(np.sqrt(np.mean(ok[ok.tag == 'v']['diff'] ** 2))),
                      rms_delta_ge_250=float(np.sqrt(np.mean(ok[ok.delta_keV >= 250]['diff'] ** 2))),
                      dash_400_350=dict(Z_model=float(base[(base.tag == 's') & (base.m_GeV == 400) & (base.delta_keV == 350)].Z.iloc[0]) if len(base[(base.tag == 's') & (base.m_GeV == 400) & (base.delta_keV == 350)]) else None))
say('Table S7 comparison (O1s 1000 GeV):'); say(comp[(comp.tag == 's') & (comp.m_GeV == 1000)].to_string(index=False))
say('Table S7 comparison (O1v 1000 GeV):'); say(comp[(comp.tag == 'v') & (comp.m_GeV == 1000)].to_string(index=False))
say('rms:', {k: round(v, 3) for k, v in RES['tableS7'].items() if isinstance(v, float)})

# --- 5b. peaks -------------------------------------------------------------------------------
peaks = []
for (tag, m, cfgn), sub in df.groupby(['tag', 'm_GeV', 'cfg']):
    sub = sub.sort_values('delta_keV'); i = int(sub.Z.values.argmax()); r = sub.iloc[i]
    z350 = sub[np.isclose(sub.delta_keV, 350)].Z; z300 = sub[np.isclose(sub.delta_keV, 300)].Z
    # delta at which 248 keV sits at the median of the observed accepted spectrum
    pm = sub.dropna(subset=['pct_ev']).sort_values('pct_ev')
    dmed = float(np.interp(0.5, pm.pct_ev.values, pm.delta_keV.values)) if (pm.pct_ev.min() < 0.5 < pm.pct_ev.max()) else np.nan
    peaks.append(dict(tag=tag, m_GeV=m, cfg=cfgn, delta_peak_keV=r.delta_keV, Z_peak=r.Z, kappa_hat_peak=r.kappa_hat, sigma_n_peak_cm2=r.sigma_n_hat_cm2,
                      S_tot_unit_peak=r.S_tot_unit, mu_hat_peak=r.mu_hat, Z_300=float(z300.iloc[0]) if len(z300) else np.nan,
                      Z_350=float(z350.iloc[0]) if len(z350) else np.nan, delta_last_keV=sub.delta_keV.max(), Z_last=float(sub.Z.iloc[-1]),
                      delta_248_median_keV=dmed, ceiling_june=KIN[int(m)]['ceiling_june'], dmax248_june=KIN[int(m)]['dmax248_june'],
                      peak_minus_dmax248=r.delta_keV - KIN[int(m)]['dmax248_june'], peak_minus_ceiling=r.delta_keV - KIN[int(m)]['ceiling_june']))
peaks = pd.DataFrame(peaks); peaks.to_csv(f'{OUT}/P021_peaks.csv', index=False)
say('peaks (baseline):'); say(peaks[peaks.cfg == 'baseline'].to_string(index=False))
say('peaks 1000 GeV O1s all configs:'); say(peaks[(peaks.tag == 's') & (peaks.m_GeV == 1000)][['cfg', 'delta_peak_keV', 'Z_peak', 'Z_300', 'Z_350', 'kappa_hat_peak', 'delta_248_median_keV']].to_string(index=False))

# global significance of the scan maximum with P008's effective trials
pk = peaks[(peaks.cfg == 'baseline') & (peaks.tag == 's') & (peaks.m_GeV == 1000)].iloc[0]
p_loc = float(stats.norm.sf(pk.Z_peak))
RES['global'] = {f'Neff_{n}': dict(p_global=1 - (1 - p_loc) ** n, Z_global=float(stats.norm.isf(1 - (1 - p_loc) ** n))) for n in (9.3, 12.2, 14.5)}
RES['global']['p_local_peak'] = p_loc

# --- 5c. comparison with LZ Fig. 6 digitised intervals (P007) --------------------------------
lzint = pd.read_csv('output/work/P007/lz_intervals_digitised.csv')
cmpint = []
for cfgn in ['baseline', 'sun', 'june']:
    for d in [250.0, 300.0, 350.0]:
        r = df[(df.tag == 's') & (df.m_GeV == 1000) & (df.cfg == cfgn) & np.isclose(df.delta_keV, d)].iloc[0]
        L = lzint[lzint.delta_keV == d].iloc[0]
        cmpint.append(dict(cfg=cfgn, delta_keV=d, kappa_hat=r.kappa_hat, kappa_90lo=r.kappa_90lo, kappa_90hi=r.kappa_90hi, kappa_68lo=r.kappa_68lo, kappa_68hi=r.kappa_68hi,
                           LZ_lower=L.c1s_mv2_sq_lower, LZ_median_sens=L.c1s_mv2_sq_median, LZ_upper=L.c1s_mv2_sq_upper,
                           ratio_upper=r.kappa_90hi / L.c1s_mv2_sq_upper, ratio_lower=r.kappa_90lo / L.c1s_mv2_sq_lower,
                           N_at_LZ_upper=L.c1s_mv2_sq_upper * r.S_tot_unit, N_at_LZ_lower=L.c1s_mv2_sq_lower * r.S_tot_unit,
                           sigma_n_hat_cm2=r.sigma_n_hat_cm2, sigma_n_90hi_cm2=r.sigma_n_90hi_cm2, LZ_sigma_upper_cm2=L.sigmaSI_upper_cm2))
cmpint = pd.DataFrame(cmpint); cmpint.to_csv(f'{OUT}/P021_vs_LZ_intervals.csv', index=False)
say('vs LZ Fig.6 intervals (1000 GeV O1s):'); say(cmpint[['cfg', 'delta_keV', 'kappa_hat', 'kappa_90lo', 'kappa_90hi', 'LZ_lower', 'LZ_upper', 'ratio_upper', 'ratio_lower', 'N_at_LZ_upper']].to_string(index=False))

# --- 5d. 2D (delta, kappa) likelihood surface, 1000 GeV O1s baseline -------------------------
cfg0 = CFGS[0]; S = SPEC[('s', 1000)]
KAP = np.logspace(-6, 3, 451)
LL = np.full((len(S['deltas']), len(KAP)), np.nan)
for i, d in enumerate(S['deltas']):
    sig = observe(S['annual'][i], cfg0)
    if sig['S_tot'] <= 0:
        continue
    LL[i] = lnL_prof(KAP * sig['S_tot'], sig, cfg0)
ll0 = float(lnL_prof(np.array([0.0]), observe(S['annual'][0], cfg0), cfg0)[0])
DM2 = 2 * (np.nanmax(LL) - LL)
np.savez(f'{OUT}/P021_surface_1000GeV_O1s.npz', deltas=S['deltas'], kappa=KAP, lnL=LL, dm2=DM2, ll0=ll0)
imax = np.unravel_index(np.nanargmax(LL), LL.shape)
RES['surface_1000_O1s'] = dict(delta_best=float(S['deltas'][imax[0]]), kappa_best=float(KAP[imax[1]]), q0_best=2 * (float(np.nanmax(LL)) - ll0),
                               region68_2dof=2.30, region90_2dof=4.61)
# extent of the 2-dof regions in delta (any kappa) and kappa (any delta)
for lev, name in [(2.30, '68'), (4.61, '90')]:
    ok2 = DM2 <= lev
    dd = S['deltas'][ok2.any(axis=1)]; kk = KAP[ok2.any(axis=0)]
    RES['surface_1000_O1s'][f'delta_range_{name}'] = [float(dd.min()), float(dd.max())] if len(dd) else None
    RES['surface_1000_O1s'][f'kappa_range_{name}'] = [float(kk.min()), float(kk.max())] if len(kk) else None
say('2D surface 1000 GeV O1s:', json.dumps(RES['surface_1000_O1s']))

# --- 5e. Higgsino: fixed coupling likelihood over delta; crossing of the O1s bands ------------
H = SPEC[('hig', 1000)]
hrows = []
for cfg in [CFGS[0], Cfg('june', halo='june'), Cfg('sun', halo='sun'), Cfg('sig8', sigE=8.0), Cfg('sig15', sigE=15.0)]:
    for i, d in enumerate(H['deltas']):
        sig = observe(H[cfg.halo][i], cfg)
        ll = float(lnL_prof(np.array([sig['S_tot']]), sig, cfg)[0]) if sig['S_tot'] > 0 else float(lnL_prof(np.array([0.0]), sig, cfg)[0])
        hrows.append(dict(cfg=cfg.name, delta_keV=float(d), N_expected=sig['S_tot'], N_H=sig['S_H'], lnL=ll))
hdf = pd.DataFrame(hrows)
hsum = {}
for cfgn, sub in hdf.groupby('cfg'):
    sub = sub.sort_values('delta_keV'); llm = sub.lnL.max(); sub = sub.assign(dm2=2 * (llm - sub.lnL))
    hdf.loc[sub.index, 'dm2'] = sub.dm2
    best = sub.loc[sub.lnL.idxmax()]
    def rng(lev):
        ok3 = sub[sub.dm2 <= lev]; return [float(ok3.delta_keV.min()), float(ok3.delta_keV.max())]
    # delta at which N = 1 (falling branch)
    fb = sub[sub.delta_keV >= 300].sort_values('delta_keV')
    dN1 = float(np.interp(0.0, -np.log(np.maximum(fb.N_expected.values, 1e-300)), fb.delta_keV.values)) if (fb.N_expected.max() > 1 > fb.N_expected.min()) else np.nan
    hsum[cfgn] = dict(delta_best=float(best.delta_keV), N_at_best=float(best.N_expected), range68=rng(1.0), range90=rng(2.706), delta_N1=dN1,
                      q_vs_bkg_at_best=2 * (float(best.lnL) - float(lnL_prof(np.array([0.0]), observe(H[cfg0.halo][0], cfg0), cfg0)[0])) if cfgn == 'baseline' else None,
                      N_at_300=float(sub[np.isclose(sub.delta_keV, 300)].N_expected.iloc[0]), N_at_350=float(sub[np.isclose(sub.delta_keV, 350)].N_expected.iloc[0]))
hdf.to_csv(f'{OUT}/P021_higgsino_fixed_coupling.csv', index=False)
RES['higgsino_fixed'] = hsum
say('Higgsino fixed-coupling:', json.dumps(hsum))
# crossing of kappa_hig = 0.0777 with the O1s 68/90% bands (each mass, baseline)
cross = {}
for m in MASSES:
    sub = base[(base.tag == 's') & (base.m_GeV == m)].sort_values('delta_keV')
    pos = sub[sub.q0 > 0]
    for name, lo, hi in [('68', 'kappa_68lo', 'kappa_68hi'), ('90', 'kappa_90lo', 'kappa_90hi')]:
        inside = pos[(pos[lo] <= KAPPA_HIG) & (KAPPA_HIG <= pos[hi])]
        cross[f'{int(m)}_{name}'] = [float(inside.delta_keV.min()), float(inside.delta_keV.max())] if len(inside) else None
    # delta where kappa_hat = kappa_hig on the rising branch (delta >= 300 up to the Z peak)
    ipk = int(sub.Z.values.argmax()); fb = sub.iloc[:ipk + 1]; fb = fb[(fb.delta_keV >= 300) & (fb.kappa_hat > 0)]
    lk = np.log(fb.kappa_hat.values)
    cross[f'{int(m)}_kappahat_eq_hig'] = float(np.interp(np.log(KAPPA_HIG), lk, fb.delta_keV.values)) if (len(lk) and lk.min() < np.log(KAPPA_HIG) < lk.max()) else None
    cross[f'{int(m)}_Z_at_kappahat_eq_hig'] = float(np.interp(cross[f'{int(m)}_kappahat_eq_hig'], fb.delta_keV.values, fb.Z.values)) if cross[f'{int(m)}_kappahat_eq_hig'] else None
RES['higgsino_band_crossing'] = dict(kappa_hig=KAPPA_HIG, **cross)
say('Higgsino line vs O1s bands:', json.dumps(cross))

# --- 5f. mass degeneracy: Z(delta) shift between masses, isovector vs isoscalar --------------
deg = {}
for tag in ['s', 'v']:
    z = {int(m): base[(base.tag == tag) & (base.m_GeV == m)].set_index('delta_keV').Z for m in MASSES}
    common = sorted(set(z[400].index) & set(z[1000].index) & set(z[4000].index))
    deg[tag] = dict(max_abs_diff_400_vs_4000_delta_le_300=float(np.max(np.abs(z[400].loc[[c for c in common if c <= 300]] - z[4000].loc[[c for c in common if c <= 300]]))),
                    Z_at_300={int(m): float(z[int(m)].loc[300.0]) for m in MASSES}, Z_at_350={int(m): float(z[int(m)].loc[350.0]) if 350.0 in z[int(m)].index else None for m in MASSES})
zs = base[(base.tag == 's') & (base.m_GeV == 1000)].set_index('delta_keV').Z; zv = base[(base.tag == 'v') & (base.m_GeV == 1000)].set_index('delta_keV').Z
deg['v_minus_s_1000'] = {str(d): float(zv.loc[d] - zs.loc[d]) for d in (100.0, 200.0, 300.0, 350.0, 380.0)}
RES['degeneracy'] = deg
say('degeneracy:', json.dumps(deg))

# likelihood-ratio preference for the peak over the LZ grid points (1000 GeV O1s and O1v, baseline)
pref = {}
for tag in ['s', 'v']:
    sub = base[(base.tag == tag) & (base.m_GeV == 1000)].set_index('delta_keV')
    qpk = sub.q0.max(); dpk = float(sub.q0.idxmax())
    pref[tag] = dict(delta_peak=dpk, q0_peak=float(qpk), **{f'dq0_peak_minus_{int(d)}': float(qpk - sub.q0.loc[d]) for d in (300.0, 350.0)},
                     **{f'LR_peak_over_{int(d)}': float(np.exp(0.5 * (qpk - sub.q0.loc[d]))) for d in (300.0, 350.0)})
RES['delta_preference_1000'] = pref
say('delta preference:', json.dumps(pref))

# exact-p check summary
ex = base[(base.tag == 's') & (base.m_GeV == 1000)].dropna(subset=['Z_exact_bH']) if 'Z_exact_bH' in base else pd.DataFrame()
if len(ex):
    RES['exact_p_check_1000_O1s'] = ex[['delta_keV', 'Z', 'Z_exact_bH', 'Z_exact_bHpois']].to_dict('records')
    say('exact single-event Z (1000 GeV O1s):'); say(ex[['delta_keV', 'Z', 'Z_exact_bH', 'Z_exact_bHpois', 'f_ev_norm', 'pct_ev']].to_string(index=False))

# best-fit summary at selected deltas (1000 GeV O1s baseline)
sel_rows = base[(base.tag == 's') & (base.m_GeV == 1000) & base.delta_keV.isin([200, 250, 300, 350, 370, 380, 385, 390, 395])]
RES['selected_1000_O1s'] = sel_rows[['delta_keV', 'Z', 'kappa_hat', 'kappa_68lo', 'kappa_68hi', 'kappa_90lo', 'kappa_90hi', 'sigma_n_hat_cm2', 'S_tot_unit', 'f_ev_norm', 'pct_ev', 'lowE_events_at_fit']].to_dict('records')
say('selected 1000 GeV O1s:'); say(sel_rows[['delta_keV', 'Z', 'kappa_hat', 'kappa_68lo', 'kappa_68hi', 'kappa_90lo', 'kappa_90hi', 'sigma_n_hat_cm2', 'S_tot_unit', 'f_ev_norm', 'pct_ev']].to_string(index=False))

RES['runtime_s'] = time.time() - T0
json.dump(RES, open(f'{OUT}/P021_results.json', 'w'), indent=1, default=float)

# ---------------------------------------------------------------------------------------------
# 6. figures
# ---------------------------------------------------------------------------------------------
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
COL = {400: '#0072B2', 1000: '#D55E00', 4000: '#009E73'}   # Okabe-Ito, fixed order by mass
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25})

# Fig 1: Z(delta)
fig, ax = plt.subplots(figsize=(8.5, 5.2))
var_cfgs = ['sig8', 'sig15', 'E246', 'E262', 'june', 'sun', 'obscut', 'gauss3']
for m in MASSES:
    for tag, ls, lab in [('s', '-', 'O$_1^s$'), ('v', '--', 'O$_1^v$')]:
        sub = base[(base.tag == tag) & (base.m_GeV == m)].sort_values('delta_keV')
        ax.plot(sub.delta_keV, sub.Z, ls, color=COL[int(m)], lw=2, label=f'{lab} {m:.0f} GeV (this work)')
    if m == 1000:
        v = df[(df.tag == 's') & (df.m_GeV == m) & df.cfg.isin(var_cfgs)]
        zmin = v.groupby('delta_keV').Z.min(); zmax = v.groupby('delta_keV').Z.max()
        ax.fill_between(zmin.index, zmin.values, zmax.values, color=COL[1000], alpha=0.15, lw=0, label='1000 GeV O$_1^s$: variants (σ$_E$, E$_{obs}$, halo, edge, low-E)')
    for tag, mk in [('s', 'o'), ('v', 's')]:
        zz = [(d, z) for d, z in zip(lz.OSIG_DELTAS, lz.OSIG['O1' + tag][int(m)]) if z is not None and d >= 100]
        ax.plot([a for a, _ in zz], [b for _, b in zz], mk, color=COL[int(m)], mfc='white' if tag == 'v' else COL[int(m)], ms=7, mew=1.5,
                label=f'LZ Table S7 O$_1^{tag}$ {m:.0f} GeV' if m == 1000 else None)
    ax.axvline(KIN[int(m)]['dmax248_june'], color=COL[int(m)], ls=':', lw=1)
ax.text(KIN[1000]['dmax248_june'] + 1, 0.3, 'δ$_{max}$(248 keV, June)', rotation=90, color='#444', fontsize=8, va='bottom')
ax.axvspan(350, 440, color='#888', alpha=0.08, lw=0); ax.text(352, 0.15, 'beyond LZ grid', color='#555', fontsize=8)
ax.set_xlabel('mass splitting δ (keV)'); ax.set_ylabel('local significance Z (σ, asymptotic, P016 anchor)')
ax.set_xlim(95, 440); ax.set_ylim(0, 4.6)
ax.legend(fontsize=7.5, loc='upper left', ncol=2, frameon=False)
ax.set_title('Profile-likelihood significance of the 248 keV event vs mass splitting, inelastic O$_1$', fontsize=10.5)
fig.tight_layout(); fig.savefig(f'{FIG}/fig1_Z_vs_delta.png', dpi=160); plt.close(fig)

# Fig 2: (delta, kappa) plane, 1000 GeV O1s
fig, ax = plt.subplots(figsize=(8.0, 5.4))
Dg, Kg = np.meshgrid(S['deltas'], KAP, indexing='ij')
ax.contourf(Dg, Kg, np.nan_to_num(DM2, nan=1e9), levels=[0, 4.61], colors=[COL[1000]], alpha=0.25, antialiased=True)
ax.contourf(Dg, Kg, np.nan_to_num(DM2, nan=1e9), levels=[0, 2.30], colors=[COL[1000]], alpha=0.6, antialiased=True)
ax.plot([], [], 's', color=COL[1000], alpha=0.85, ms=10, label='68% region (2 dof)'); ax.plot([], [], 's', color=COL[1000], alpha=0.3, ms=10, label='90% region (2 dof)')
sub = base[(base.tag == 's') & (base.m_GeV == 1000)].sort_values('delta_keV')
ax.plot(sub.delta_keV, sub.kappa_hat, '-', color='#222', lw=1.8, label='best fit κ̂(δ) (1-dof profile)')
ax.plot(sub.delta_keV, sub.kappa_90lo.replace(0, np.nan), ':', color='#222', lw=1.2, label='90% interval at fixed δ')
ax.plot(sub.delta_keV, sub.kappa_90hi, ':', color='#222', lw=1.2)
Lz = lzint[lzint.delta_keV >= 100]
ax.errorbar(Lz.delta_keV + 1.5, np.sqrt(Lz.c1s_mv2_sq_lower * Lz.c1s_mv2_sq_upper), yerr=[np.sqrt(Lz.c1s_mv2_sq_lower * Lz.c1s_mv2_sq_upper) - Lz.c1s_mv2_sq_lower, Lz.c1s_mv2_sq_upper - np.sqrt(Lz.c1s_mv2_sq_lower * Lz.c1s_mv2_sq_upper)],
            fmt='none', ecolor=COL[400], elinewidth=3, capsize=4, label='LZ Fig. 6 two-sided 90% intervals (digitised, P007)')
ax.axhline(KAPPA_HIG, color=COL[4000], lw=2, ls='--', label='pure Higgsino, (c$_{eq}^s$m$_v^2$)$^2$ = 0.0777 (P007)')
ax.plot([RES['surface_1000_O1s']['delta_best']], [RES['surface_1000_O1s']['kappa_best']], '*', color='#222', ms=12, label='global maximum')
ax.axvline(KIN[1000]['dmax248_june'], color='#666', ls=':', lw=1); ax.axvline(KIN[1000]['ceiling_june'], color='#666', ls='-.', lw=1)
ax.text(KIN[1000]['dmax248_june'] - 2, 2e-6, 'δ$_{max}$(248)', rotation=90, fontsize=8, color='#555', ha='right')
ax.text(KIN[1000]['ceiling_june'] - 2, 2e-6, 'μv$_{max}^2$/2', rotation=90, fontsize=8, color='#555', ha='right')
ax.set_yscale('log'); ax.set_ylim(1e-6, 1e2); ax.set_xlim(95, 400)
ax.set_xlabel('mass splitting δ (keV)'); ax.set_ylabel('(c$_1^s$ m$_v^2$)$^2$')
sec = ax.secondary_yaxis('right', functions=(lambda k: sigma_n_cm2(k, 1000.0), lambda s: s / sigma_n_cm2(1.0, 1000.0)))
sec.set_ylabel('σ$_n$ (cm$^2$), 1000 GeV'); sec.spines['right'].set_visible(True)
ax.set_title('1000 GeV O$_1^s$: 68% / 90% (2-dof) likelihood regions in (δ, coupling)', fontsize=10.5)
ax.legend(fontsize=7.5, loc='upper left', frameon=False)
fig.tight_layout(); fig.savefig(f'{FIG}/fig2_delta_kappa_1000GeV.png', dpi=160); plt.close(fig)

# Fig 3: mechanism (observed-space densities) + Higgsino fixed-coupling
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
shades = ['#b3cde0', '#6497b1', '#03396c', '#011f4b', '#000000']
for c, d in zip(shades, [300.0, 350.0, 370.0, 385.0, 395.0]):
    i = int(np.argmin(np.abs(S['deltas'] - d)))
    sig = observe(S['annual'][i], cfg0)
    a1.plot(E_O, sig['d'] / sig['S_tot'], color=c, lw=1.8, label=f'δ = {S["deltas"][i]:.0f} keV')
a1.axhline(5.7e-4 / 70 / 1.0, color=COL[1000], ls='--', lw=1.2)
a1.text(3, 5.7e-4 / 70 * 1.35, 'background density (b$_H$ = 5.7×10$^{-4}$)/70 keV', color=COL[1000], fontsize=8)
a1.axvline(248, color='#444', lw=1); a1.text(249, 2e-1, 'event', fontsize=8, color='#444')
a1.set_yscale('log'); a1.set_ylim(1e-6, 0.5); a1.set_xlim(0, 320)
a1.set_xlabel('observed recoil energy (keV)'); a1.set_ylabel('accepted signal density per event (keV$^{-1}$)')
a1.set_title('1000 GeV O$_1^s$: observed-space spectra (ε ⊗ σ$_E$ = 11 keV)', fontsize=10); a1.legend(fontsize=8, frameon=False, loc='upper left')
hb = hdf[hdf.cfg == 'baseline'].sort_values('delta_keV'); hj = hdf[hdf.cfg == 'june'].sort_values('delta_keV'); hs = hdf[hdf.cfg == 'sun'].sort_values('delta_keV')
a2.plot(hb.delta_keV, hb.dm2, color=COL[4000], lw=2, label='annual-average halo')
a2.plot(hj.delta_keV, hj.dm2, color=COL[4000], lw=1.2, ls='--', label='16 June halo')
a2.plot(hs.delta_keV, hs.dm2, color=COL[4000], lw=1.2, ls=':', label='Sun-frame halo')
a2.axhline(1.0, color='#888', lw=1); a2.axhline(2.706, color='#888', lw=1); a2.text(301, 1.1, '68%', fontsize=8, color='#555'); a2.text(301, 2.8, '90%', fontsize=8, color='#555')
a2.axvspan(358, 380, color=COL[400], alpha=0.12, lw=0); a2.text(381, 5.0, 'P007 90% window\n358–380 keV\n(N = 3.65–0.105)', fontsize=8, color=COL[400])
a2.set_xlim(300, 400); a2.set_ylim(0, 10); a2.set_xlabel('mass splitting δ (keV)'); a2.set_ylabel('−2 Δln L (σ$_n$ = 7.4×10$^{-39}$ cm$^2$ fixed)')
a2.set_title('Pure Higgsino at 1000 GeV: likelihood over δ at fixed coupling', fontsize=10); a2.legend(fontsize=8, frameon=False, loc='upper left')
fig.tight_layout(); fig.savefig(f'{FIG}/fig3_mechanism_higgsino.png', dpi=160); plt.close(fig)
say(f'figures written; total runtime {time.time() - T0:.0f}s')
