"""
P072 - Exothermic scattering of a metastable excited state as the origin of the LZ 248 keV event:
kinematics, spectral shape, companions and the required relic fraction.  (Competes with P058 §5.6.)

Method.  For the O1 operator WimPyDD's per-stream kernel K(E, v) is, isotope by isotope, a pure step function
in v: K(E, v) = (1000/m) sum_i w_i(E) theta(v - v_min,i(E; m, delta)), where w_i(E) is the single-isotope rate per
unit delta_eta at m = 1000 GeV (verified here: rebuilt kernels agree with WimPyDD's own to 1e-16 and the
normalisation scales exactly as 1/m).  We therefore tabulate w_i(E) with WimPyDD once per coupling (9 isotopes x
641 energies x 3 couplings) and obtain EXACT WimPyDD spectra for any (m, delta, halo) as
dR/dE = (1000/m) sum_i w_i(E) eta(v_min,i(E)), with eta the halo mean inverse speed (WimPyDD's streamed halo on
24 days of the year, or analytic truncated Maxwellians for low-dispersion / dark-disk halos).

Parts
  A  per-isotope tables, validation against P058's cached kernels and R_ROI (916 iso / 859 p), P011 (858)
  B  (m, |delta|) grid, m = 0.3-10 TeV, |delta| = 100-500 keV: E*, width, percentile of 248 keV, companions
  C  statistics: five-term likelihood (5.4-125, 125-200, 600-1700 phd Poisson; 200-270 keV extended with the
     event's observed-energy density), profile Z, Bayes factors vs background and exothermic vs endothermic
  D  required f2*sigma, chi1 up-scattering at the same coupling, chi2 lifetime, dark-photon (m_A', eps) window
  E  annual modulation and the 16 June date factor
  F  two-population (f2, delta) plane
  G  low-dispersion and dark-disk halos: how narrow can the down-scatter line be?
  H  figures and results JSON
Run from the simulation root:  .venv/bin/python output/code/P072_exothermic_origin.py   (~3 min; w-tables cached)
"""
from __future__ import annotations
import sys, os, json, math, time
sys.path.insert(0, 'output/code')
import numpy as np
import pandas as pd
from scipy.special import erf
from scipy.stats import norm, poisson
from scipy.optimize import minimize_scalar, brentq
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import lzcommon as lz

T0 = time.time()
OUT = 'output/work/P072'; FIG = f'{OUT}/figures'; CACHE = f'{OUT}/cache'
for d in (OUT, FIG, CACHE):
    os.makedirs(d, exist_ok=True)
LOG = open(f'{OUT}/run_log.txt', 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n'); LOG.flush()

# ------------------------------------------------------------------------------------------------------------
# constants (recalled values flagged in details.md)
# ------------------------------------------------------------------------------------------------------------
MV = lz.M_V_GEV; C_UNIT = 1.0 / MV ** 2
GF = 1.1663788e-5; SW2 = 0.2312; ALPHA = 1 / 137.035999; MZ = 91.1876; MP = lz.M_NUCLEON_GEV
HBAR_GEV_S = 6.582119569e-25; T_U_S = 13.8e9 * 3.15576e7
EXPOSURE = lz.LZ['exposure_tyr']
C_P_HIG = (GF / math.sqrt(2)) * (1 - 4 * SW2); C_N_HIG = -(GF / math.sqrt(2))
F2_FO = {'iso': 0.5, 'p': 0.5, 'hig': 0.42}                     # P026 freeze-out fractions (dark photon 0.50, Higgsino 0.42)
ALPHA_D_RELIC = 0.0245                                           # P011: relic-density alpha_D at 1 TeV
MU_P = 1000.0 * MP / (1000.0 + MP)
SIGMA_UNIT = C_UNIT ** 2 * MU_P ** 2 / math.pi * lz.GEV_TO_CM2   # sigma_N for c_N = 1/m_v^2 at 1 TeV (2.96e-38 cm^2)

WD = lz.wd()
MN_ISO = np.array(WD.Xe.mass, dtype=float)         # 0.931*A, WimPyDD convention (9 isotopes)
A_ISO = np.array(WD.Xe.a, dtype=float)
MN_MEAN = 0.931 * float(WD.Xe.average_a)
M_REF = 1000.0
E_GRID = np.concatenate([np.arange(1.0, 700.0, 2.5), np.arange(700.0, 2505.0, 5.0)])   # same as P058
DE = np.gradient(E_GRID)
VGRID = np.linspace(0.0, 844.0, 1200)
HAMS = {'iso': lz.wd_hamiltonian('P072_iso', {1: (2 * C_UNIT, 0.0)}),
        'p': lz.wd_hamiltonian('P072_p', {1: (C_UNIT, C_UNIT)}),
        'hig': lz.wd_hamiltonian('P072_hig', {1: (C_P_HIG + C_N_HIG, C_P_HIG - C_N_HIG)})}

# ------------------------------------------------------------------------------------------------------------
# Part A: per-isotope step tables w_i(E) [events/(t yr keV) per unit delta_eta at m = 1000 GeV]
# ------------------------------------------------------------------------------------------------------------
def _w_job(args):
    tag, i = args
    ham = HAMS[tag]
    return tag, i, np.array([WD.diff_rate(WD.Xe, ham, M_REF, float(e), np.array([3000.0]), np.array([1.0]), j_chi=0.5, delta=0.0,
                                          sum_over_streams=False, isotopes_list={0: [i]})[0] for e in E_GRID]) * 1000.0 * 365.25

WFILE = f'{CACHE}/w_tables.npz'
if os.path.exists(WFILE):
    z = np.load(WFILE); W = {t: z[t] for t in HAMS}
    assert np.allclose(z['E'], E_GRID)
    say('Part A: per-isotope tables loaded from cache')
else:
    import multiprocessing as mp
    say('Part A: computing per-isotope tables with WimPyDD (27 jobs) ...')
    W = {t: np.zeros((9, len(E_GRID))) for t in HAMS}
    with mp.get_context('fork').Pool(9) as pool:
        for tag, i, w in pool.imap_unordered(_w_job, [(t, i) for t in HAMS for i in range(9)]):
            W[tag][i] = w
    np.savez_compressed(WFILE, E=E_GRID, **W)
    say(f'  done, t = {time.time() - T0:.0f} s')

def vmin_iso(E, m, delta):
    """WimPyDD's v_min per isotope [km/s], shape (9, len(E)); delta signed (keV), negative = exothermic."""
    mu = m * MN_ISO / (m + MN_ISO)
    return np.abs(MN_ISO[:, None] * E[None, :] / mu[:, None] + delta) / np.sqrt(2 * MN_ISO[:, None] * E[None, :]) * 300.0

def spectrum(tag, m, delta, eta):
    """dR/dE [events/(t yr keV)] on E_GRID for coupling tag, mass m, signed delta and halo function eta(v)."""
    return (M_REF / m) * np.sum(W[tag] * eta(vmin_iso(E_GRID, m, delta)), axis=0)

# halos: WimPyDD streamed halo (Baxter-2021 SHM) on 24 days -> step-function eta(v) exactly as diff_rate uses it
DAYS24 = 365.25 / 48 + 365.25 / 24 * np.arange(24)
def eta_from_streams(deta):
    rev = np.concatenate([np.cumsum(deta[::-1])[::-1], [0.0]])
    return lambda v: rev[np.searchsorted(VGRID, v, side='left')]
DETA_DAY = {d: lz.wd_halo(day_of_year=float(d), vmin=VGRID)[1] for d in DAYS24}
ETA_DAY = {d: eta_from_streams(x) for d, x in DETA_DAY.items()}
ETA = {'annual': eta_from_streams(np.mean(list(DETA_DAY.values()), axis=0)),
       'june16': eta_from_streams(lz.wd_halo(day_of_year=167, vmin=VGRID)[1]),
       'dec16': eta_from_streams(lz.wd_halo(day_of_year=350, vmin=VGRID)[1]),
       'sun': eta_from_streams(lz.wd_halo(vmin=VGRID)[1])}
def eta_maxwell(v_e, v0, vesc=544.0):
    """Analytic truncated-Maxwellian mean inverse speed (lzcommon.eta0) as a callable on arrays."""
    return lambda v: np.asarray(lz.eta0(np.ravel(v), v_e=v_e, v0=v0, vesc=vesc)).reshape(np.shape(v))

# validation against P058's exact cached kernels
say('Part A: validation against P058 cached WimPyDD kernels')
val = {}
for fn, tag, m, d in (('K_iso_1000_m300', 'iso', 1000.0, -300.0), ('K_p_1000_m380', 'p', 1000.0, -380.0),
                      ('K_iso_4000_m350', 'iso', 4000.0, -350.0), ('K_iso_400_p300', 'iso', 400.0, 300.0), ('K_hig_1000_m366', 'hig', 1000.0, -366.0)):
    f = f'output/work/P058/cache/{fn}.npz'
    if not os.path.exists(f):
        continue
    z = np.load(f); E58, K58 = z['E'], z['K']
    idx = np.searchsorted(E_GRID, E58); ok = np.abs(E_GRID[np.minimum(idx, len(E_GRID) - 1)] - E58) < 1e-9
    vm = vmin_iso(E58[ok], m, d)                                        # (9, nE)
    Kre = (M_REF / m) * np.einsum('ie,iev->ev', W[tag][:, idx[ok]], (VGRID[None, None, :] >= vm[:, :, None]).astype(float))
    rel = np.max(np.abs(Kre - K58[ok])) / K58.max()
    val[fn] = float(rel); say(f'  {fn}: max |K_rebuilt - K_WimPyDD| / max K = {rel:.2e}  ({ok.sum()} energies x 1200 streams)')

# efficiencies (P058/P038 conventions) and observed-energy smearing
def Phi(x): return 0.5 * (1 + erf(x / math.sqrt(2)))
E50_600, SIG_600, E50_1000, SIG_1000, S1_EXP, PLATEAU_HE = 271.6, 10.9, 423.2, 14.5, 1.1537, 0.955
E50_1700 = E50_1000 * (1700.0 / 1000.0) ** (1 / S1_EXP); SIG_1700 = SIG_1000 * E50_1700 / E50_1000
def eps_roi(E): return 0.96 * Phi((E - 5.4) / 2.5) * (1 - Phi((E - 269.9) / 11.5))
def eps_he(E, top=1700):
    E50t, st = (E50_1700, SIG_1700) if top == 1700 else (E50_1000, SIG_1000)
    return PLATEAU_HE * Phi((E - E50_600) / SIG_600) * (1 - Phi((E - E50t) / st))
EPS_ROI, EPS_HE17, EPS_HE10 = eps_roi(E_GRID), eps_he(E_GRID, 1700), eps_he(E_GRID, 1000)
EOBS = np.arange(0.5, 800.0, 1.0)
SIG_E = 11.0 * np.sqrt(np.maximum(E_GRID, 1.0) / 248.0)                 # P009/P021 resolution
SMEAR = norm.pdf(EOBS[:, None], E_GRID[None, :], SIG_E[None, :]) * DE[None, :]   # (nObs, nE): s_obs = SMEAR @ (r eps)
I248 = int(np.argmin(np.abs(EOBS - 248.0)))
def region_mask(lo, hi): return (EOBS >= lo) & (EOBS < hi)
MASKS = {'lo': region_mask(5.4, 55), 'mid': region_mask(55, 125), 'gap': region_mask(125, 200), 'hi': region_mask(200, 270), 'roi': region_mask(0, 800)}

def analyse(r):
    """Region counts (2.84 t yr), density at 248 keV, spectral descriptors for a spectrum r(E) [events/(t yr keV)]."""
    acc = r * EPS_ROI; s_obs = SMEAR @ acc                              # accepted spectrum in observed energy (per keV)
    out = {k: float(EXPOSURE * s_obs[msk].sum()) for k, msk in MASKS.items()}
    out['lomid'] = out['lo'] + out['mid']
    out['he1700'] = float(EXPOSURE * np.sum(r * EPS_HE17 * DE)); out['he1000'] = float(EXPOSURE * np.sum(r * EPS_HE10 * DE))
    out['all'] = float(EXPOSURE * np.sum(r * DE))
    # true-energy bins (P058 convention) for validation
    for k, (a, b) in {'gap_true': (125, 200), 'hi_true': (200, 270), 'lomid_true': (5.4, 125)}.items():
        mk = (E_GRID >= a) & (E_GRID < b); out[k] = float(EXPOSURE * np.sum(acc[mk] * DE[mk]))
    out['roi_true'] = float(EXPOSURE * np.sum(acc * DE))
    out['D248'] = float(EXPOSURE * s_obs[I248])                         # events per keV at 248 keV observed
    tot = s_obs.sum()
    if tot > 0:
        c = np.cumsum(s_obs) / tot
        out['pct248'] = float(c[I248]); out['p16'], out['p50'], out['p84'], out['p95'] = [float(np.interp(q, c, EOBS)) for q in (0.16, 0.5, 0.84, 0.95)]
        im = int(np.argmax(s_obs)); out['mode'] = float(EOBS[im]); half = s_obs[im] / 2
        l = im
        while l > 0 and s_obs[l] > half: l -= 1
        h = im
        while h < len(EOBS) - 1 and s_obs[h] > half: h += 1
        out['fwhm'] = float(EOBS[h] - EOBS[l]); out['f_hi_of_roi'] = out['hi'] / out['roi'] if out['roi'] > 0 else np.nan
    else:
        for k in ('pct248', 'p16', 'p50', 'p84', 'p95', 'mode', 'fwhm', 'f_hi_of_roi'): out[k] = np.nan
    return out

def estar(m, d): mu = m * MN_MEAN / (m + MN_MEAN); return abs(d) * mu / MN_MEAN
def halfwidth(m, d, v=250.0):
    mu = m * MN_MEAN / (m + MN_MEAN); b = v / lz.C_KMS
    x = abs(d) * 1e-6 / (mu * b ** 2); pref = mu ** 2 * b ** 2 / MN_MEAN * 1e6          # |delta| keV -> GeV; result in keV
    return pref * math.sqrt(1 + 2 * x)                                  # (E+ - E-)/2 [keV]

# R_ROI check vs P058 / P011 (true-energy ROI counts, 24-day annual halo)
rows = []
for tag, d in (('iso', 300.0), ('iso', 350.0), ('iso', 380.0), ('p', 300.0), ('p', 380.0), ('hig', 300.0)):
    ax_ = analyse(spectrum(tag, 1000.0, -d, ETA['annual'])); an_ = analyse(spectrum(tag, 1000.0, +d, ETA['annual']))
    rows.append(dict(tag=tag, delta=d, Nexo_roi_true=ax_['roi_true'], Nendo_roi_true=an_['roi_true'], R_roi=ax_['roi_true'] / an_['roi_true'],
                     R_gap=ax_['gap_true'] / an_['roi_true'], Nexo_roi_obs=ax_['roi'], gap_per_hi_true=ax_['gap_true'] / ax_['hi_true'], gap_per_hi_obs=ax_['gap'] / ax_['hi']))
vdf = pd.DataFrame(rows); vdf.to_csv(f'{OUT}/P072_validation_RROI.csv', index=False)
say('  R_ROI (exothermic ROI events per endothermic ROI event, f2 = 1, 1 TeV, annual): P058 916/2.78e4/8.11e6 (iso 300/350/380), 859 (p 300), 2.64e6 (p 380), 963 (hig 300); P011 858 (p 300)')
say(vdf.to_string(index=False, float_format='%.4g'))

# ------------------------------------------------------------------------------------------------------------
# Part C helpers: statistics
# ------------------------------------------------------------------------------------------------------------
OBS = {'lomid': dict(n=42, b=58.6, acc=0.866), 'gap': dict(n=0, b=0.02, acc=1.0), 'he1700': dict(n=0, b=0.606, acc=1.0)}
B_HI, W_HI = 5.7e-4, 70.0                                              # P016 anchor: 200-270 keV background, flat in energy
def nll0(use_density=True, regions=('lomid', 'gap', 'he1700')):
    """-lnL of the background-only hypothesis with the current OBS backgrounds (recomputed so that robustness scans stay consistent)."""
    return sum(OBS[r]['b'] - OBS[r]['n'] * math.log(OBS[r]['b']) for r in regions) + B_HI - math.log(B_HI / W_HI if use_density else B_HI)
NLL0 = nll0()

def nll(kappa, N, use_density=True, regions=('lomid', 'gap', 'he1700')):
    """-lnL: Poisson terms for lomid/gap/he1700, extended term for the 200-270 keV region with the event at 248 keV
    (density form) or the P058 binned form (n = 1)."""
    t = 0.0
    for reg in regions:
        mu = OBS[reg]['b'] + OBS[reg]['acc'] * kappa * N[reg]; t += mu - OBS[reg]['n'] * math.log(mu)
    mu_hi = B_HI + kappa * N['hi']
    t += mu_hi - (math.log(max(B_HI / W_HI + kappa * N['D248'], 1e-300)) if use_density else math.log(mu_hi))
    return t
N_HI_MIN = 1e-6      # events per unit coupling: below this the hypothesis cannot produce the event at any sensible coupling (kappa > 1e6)

def fit(N, use_density=True, regions=('lomid', 'gap', 'he1700')):
    """Profile over log10 kappa: returns Z (vs background only), kappa-hat, and fitted region expectations."""
    if not (N['hi'] > N_HI_MIN):
        return dict(Z=np.nan, kappa_hat=np.nan, mu_hi=np.nan, mu_gap=np.nan, mu_lomid=np.nan, mu_he=np.nan)
    res = minimize_scalar(lambda lk: nll(10 ** lk, N, use_density, regions), bounds=(-18, 6), method='bounded', options={'xatol': 1e-8})
    k = 10 ** res.x; base = nll0(use_density, regions)
    return dict(Z=math.sqrt(max(0.0, 2 * (base - res.fun))), kappa_hat=k, mu_hi=k * N['hi'], mu_gap=k * N['gap'], mu_lomid=0.866 * k * N['lomid'], mu_he=k * N['he1700'])

MU_GRID = np.concatenate([[0.0], np.logspace(-6, 1, 500)])            # prior variable: expected 200-270 keV signal events
def bayes(N, mu_max=10.0, prior='flat', use_density=True, regions=('lomid', 'gap', 'he1700')):
    """Marginal likelihood ratio vs background only, prior on mu_hi = kappa N_hi: flat on [0, mu_max] or log-uniform on [1e-3, mu_max]."""
    if not (N['hi'] > N_HI_MIN):
        return np.nan
    mus = MU_GRID[MU_GRID <= mu_max]
    base = nll0(use_density, regions)
    lr = np.array([math.exp(min(700.0, base - nll(mu / N['hi'], N, use_density, regions))) for mu in mus])
    if prior == 'flat':
        return float(np.trapezoid(lr, mus) / mu_max)
    m = mus >= 1e-3
    return float(np.trapezoid(lr[m] / mus[m], mus[m]) / math.log(mu_max / 1e-3))

# ------------------------------------------------------------------------------------------------------------
# Part B: (m, |delta|) grid
# ------------------------------------------------------------------------------------------------------------
MASSES = [300.0, 500.0, 1000.0, 2000.0, 4000.0, 10000.0]
DELTAS = np.arange(100.0, 501.0, 10.0)
say('Part B/C: (m, |delta|) grid, isoscalar O1 (and proton-only), annual-mean Baxter halo')
grid_rows = []; SPEC = {}
for tag in ('iso', 'p'):
    for m in MASSES:
        for d in DELTAS:
            rx = spectrum(tag, m, -d, ETA['annual']); rn = spectrum(tag, m, +d, ETA['annual'])
            ax_, an_ = analyse(rx), analyse(rn)
            SPEC[(tag, m, d)] = (rx, rn)
            row = dict(tag=tag, m_GeV=m, delta_keV=d, E_star=estar(m, d), halfwidth_v250=halfwidth(m, d),
                       v_min_248=float(np.abs(MN_MEAN * 248e-6 / (m * MN_MEAN / (m + MN_MEAN)) - d * 1e-6) / math.sqrt(2 * MN_MEAN * 248e-6) * lz.C_KMS))
            for k, v in ax_.items(): row[f'exo_{k}'] = v
            for k in ('lomid', 'gap', 'hi', 'roi', 'he1700', 'he1000', 'all', 'D248', 'pct248', 'p16', 'p50', 'p84', 'mode'): row[f'endo_{k}'] = an_[k]
            row['R_roi'] = ax_['roi'] / an_['roi'] if an_['roi'] > 0 else np.inf
            row['gap_per_hi'] = ax_['gap'] / ax_['hi'] if ax_['hi'] > 0 else np.nan
            row['lomid_per_hi'] = ax_['lomid'] / ax_['hi'] if ax_['hi'] > 0 else np.nan
            row['he1700_per_hi'] = ax_['he1700'] / ax_['hi'] if ax_['hi'] > 0 else np.nan
            row['D248_per_hi'] = ax_['D248'] / ax_['hi'] if ax_['hi'] > 0 else np.nan       # shape density at the event per 200-270 event [1/keV]
            row['endo_D248_per_hi'] = an_['D248'] / an_['hi'] if an_['hi'] > 0 else np.nan
            fx = fit(ax_); fx4 = fit(ax_, use_density=False); fn_ = fit(an_); fn4 = fit(an_, use_density=False)
            row.update(Z5_exo=fx['Z'], mu_hi_exo=fx['mu_hi'], mu_gap_exo=fx['mu_gap'], mu_lomid_exo=fx['mu_lomid'], mu_he_exo=fx['mu_he'], kappa_exo=fx['kappa_hat'],
                       Z4_exo=fx4['Z'], Z5_endo=fn_['Z'], mu_hi_endo=fn_['mu_hi'], kappa_endo=fn_['kappa_hat'], Z4_endo=fn4['Z'],
                       B_exo=bayes(ax_), B_endo=bayes(an_), B_exo_mu3=bayes(ax_, 3.0), B_endo_mu3=bayes(an_, 3.0), B_exo_log=bayes(ax_, prior='log'), B_endo_log=bayes(an_, prior='log'),
                       B_exo_noHE=bayes(ax_, regions=('lomid', 'gap')), B_endo_noHE=bayes(an_, regions=('lomid', 'gap')),
                       B_exo_bin=bayes(ax_, use_density=False), B_endo_bin=bayes(an_, use_density=False))
            # coupling for one exothermic ROI event at f2 = 1 (kappa*f2), sigma_N f2, chi1 up-scatter events at that coupling for f2 = f2,fo
            row['kappa_f2_one_event'] = 1.0 / ax_['roi'] if ax_['roi'] > 0 else np.inf
            row['sigmaN_f2_cm2'] = SIGMA_UNIT * row['kappa_f2_one_event']
            f2fo = F2_FO[tag]
            row['endo_events_at_exo_coupling_f2fo'] = (1 - f2fo) / f2fo * an_['roi'] / ax_['roi'] if ax_['roi'] > 0 else np.nan
            row['endo_gap_at_exo_coupling_f2fo'] = (1 - f2fo) / f2fo * an_['gap'] / ax_['roi'] if ax_['roi'] > 0 else np.nan
            grid_rows.append(row)
    say(f'  {tag} done, t = {time.time() - T0:.0f} s')
G = pd.DataFrame(grid_rows)
# 16 June date factor p(t_obs | H)/p_flat = R_ROI(day 167)/<R_ROI> for every isoscalar grid point (24-day halos, periodic spline)
def roi_true(r): return float(np.sum(r * EPS_ROI * DE))
def date_factor(y):
    if not (y.mean() > 0): return np.nan
    cs = CubicSpline(np.concatenate([DAYS24, [DAYS24[0] + 365.25]]), np.concatenate([y, [y[0]]]), bc_type='periodic')
    return float(cs(167.0) / y.mean())
dx, dn = [], []
for _, row in G.iterrows():
    if row.tag != 'iso':
        dx.append(np.nan); dn.append(np.nan); continue
    dx.append(date_factor(np.array([roi_true(spectrum('iso', row.m_GeV, -row.delta_keV, ETA_DAY[k])) for k in DAYS24])))
    dn.append(date_factor(np.array([roi_true(spectrum('iso', row.m_GeV, +row.delta_keV, ETA_DAY[k])) for k in DAYS24])))
G['date_exo'] = dx; G['date_endo'] = dn
G['B_exo_date'] = G.B_exo * G.date_exo; G['B_endo_date'] = G.B_endo * G.date_endo
G.to_csv(f'{OUT}/P072_grid.csv', index=False)
GI = G[G.tag == 'iso']
say(f'  date factors (iso grid): exothermic {np.nanmin(GI.date_exo):.3f}-{np.nanmax(GI.date_exo):.3f}; endothermic {np.nanmin(GI.date_endo):.2f}-{np.nanmax(GI.date_endo):.2f}, t = {time.time() - T0:.0f} s')
say(f'  grid-wide maxima of the accepted exothermic spectrum: mode {GI.exo_mode.max():.1f} keV, p84 {GI.exo_p84.max():.1f} keV, p95 {GI.exo_p95.max():.1f} keV; min percentile of 248 keV {GI.exo_pct248.min():.4f}; FWHM range {GI.exo_fwhm.min():.0f}-{GI.exo_fwhm.max():.0f} keV')

say('Part B: exothermic spectra, isoscalar, selected (m, |delta|): E*, half-width(250 km/s), observed accepted p16/p50/p84/mode/FWHM, percentile of 248, companions per 200-270 event')
cols = ['m_GeV', 'delta_keV', 'E_star', 'halfwidth_v250', 'exo_p16', 'exo_p50', 'exo_p84', 'exo_mode', 'exo_fwhm', 'exo_pct248', 'exo_f_hi_of_roi', 'gap_per_hi', 'lomid_per_hi', 'he1700_per_hi', 'R_roi']
sel = GI[GI.delta_keV.isin([100, 150, 200, 250, 280, 300, 350, 400, 450, 500]) & GI.m_GeV.isin([300, 1000, 4000, 10000])]
say(sel[cols].to_string(index=False, float_format='%.3g'))

# (m, |delta|) placing 248 keV at the mode / p84 / p95 of the accepted exothermic spectrum, and the |delta| giving E* = 248
edge_rows = []
for m in MASSES:
    g = GI[GI.m_GeV == m].sort_values('delta_keV')
    def cross(col, target=248.0):
        y = g[col].values - target; x = g.delta_keV.values; s = np.where(np.diff(np.sign(y)) != 0)[0]
        return [float(np.interp(0.0, [y[i], y[i + 1]], [x[i], x[i + 1]])) if y[i + 1] != y[i] else float(x[i]) for i in s]
    mu = m * MN_MEAN / (m + MN_MEAN)
    edge_rows.append(dict(m_GeV=m, delta_Estar248=248.0 * MN_MEAN / mu, delta_mode248=cross('exo_mode'), delta_p84_248=cross('exo_p84'), delta_p95_248=cross('exo_p95'),
                          mode_range=(float(g.exo_mode.min()), float(g.exo_mode.max())), best_pct248=float(g.exo_pct248.min()), delta_best_pct=float(g.delta_keV[g.exo_pct248.idxmin()]),
                          min_gap_per_hi=float(g.gap_per_hi.min()), delta_min_gap=float(g.delta_keV[g.gap_per_hi.idxmin()])))
EDGE = pd.DataFrame(edge_rows); EDGE.to_csv(f'{OUT}/P072_248_at_peak_or_edge.csv', index=False)
say('Part B: where does 248 keV sit at the mode / 84th / 95th percentile of the accepted exothermic spectrum?'); say(EDGE.to_string(index=False, float_format='%.4g'))

# best exothermic cases
say('Part C: five-term likelihood (density form) and Bayes factors; best exothermic and endothermic cases per mass (isoscalar)')
best_rows = []
for m in MASSES:
    g = GI[GI.m_GeV == m]
    bx = g.loc[g.B_exo.idxmax()]; bn = g.loc[g.B_endo.idxmax()]; zx = g.loc[g.Z5_exo.idxmax()]; zn = g.loc[g.Z5_endo.idxmax()]
    best_rows.append(dict(m_GeV=m, delta_bestB_exo=bx.delta_keV, B_exo_max=bx.B_exo, Z5_exo_at_bestB=bx.Z5_exo, mu_hi_exo=bx.mu_hi_exo, gap_per_hi=bx.gap_per_hi, pct248=bx.exo_pct248,
                          delta_bestZ_exo=zx.delta_keV, Z5_exo_max=zx.Z5_exo, Z4_exo_max=float(g.Z4_exo.max()),
                          delta_bestB_endo=bn.delta_keV, B_endo_max=bn.B_endo, Z5_endo_at_bestB=bn.Z5_endo, delta_bestZ_endo=zn.delta_keV, Z5_endo_max=zn.Z5_endo, Z4_endo_max=float(g.Z4_endo.max()),
                          B_exo_over_endo_best=bx.B_exo / bn.B_endo,
                          B_exo_marg=float(np.nanmean(g.B_exo)), B_endo_marg=float(np.nansum(g.B_endo) / len(g)),      # flat prior over delta in 100-500 keV (endothermic zero above delta_max)
                          ))
BEST = pd.DataFrame(best_rows); BEST.to_csv(f'{OUT}/P072_best_cases.csv', index=False)
say(BEST.to_string(index=False, float_format='%.3g'))
B_exo_all = float(np.nanmean(GI.B_exo)); B_endo_all = float(np.nansum(GI.B_endo) / len(GI))
B_exo_all_d = float(np.nanmean(GI.B_exo_date)); B_endo_all_d = float(np.nansum(GI.B_endo_date) / len(GI))
say(f'  marginal over m (6 masses, equal weight) and delta (flat 100-500 keV): B_exo/bkg = {B_exo_all:.3g}, B_endo/bkg = {B_endo_all:.3g}, B(exo/endo) = {B_exo_all / B_endo_all:.3g}')
say(f'  ... including the 16 June date factor: B_exo = {B_exo_all_d:.3g}, B_endo = {B_endo_all_d:.3g}, B(exo/endo) = {B_exo_all_d / B_endo_all_d:.3g}')
# restricted endothermic prior (delta in 250-390 keV, where the endothermic reading actually lives) vs exothermic over its full range
gi_e = GI[(GI.delta_keV >= 250) & (GI.delta_keV <= 390)]
B_endo_r = float(np.nansum(gi_e.B_endo) / len(gi_e)); B_endo_rd = float(np.nansum(gi_e.B_endo_date) / len(gi_e))
say(f'  endothermic marginal over delta = 250-390 keV only: B = {B_endo_r:.3g} (with date {B_endo_rd:.3g}); B(exo/endo) = {B_exo_all / B_endo_r:.3g} (with date {B_exo_all_d / B_endo_rd:.3g})')
# maxima with the date factor and prior variants, 1 TeV
g1 = GI[GI.m_GeV == 1000.0]
for col in ('B_exo', 'B_exo_date', 'B_exo_mu3', 'B_exo_log', 'B_exo_noHE', 'B_exo_bin', 'B_endo', 'B_endo_date', 'B_endo_mu3', 'B_endo_log', 'B_endo_noHE', 'B_endo_bin'):
    i = g1[col].idxmax(); say(f'  1 TeV max {col:12s} = {g1.loc[i, col]:.3g} at |delta| = {g1.loc[i, "delta_keV"]:.0f} keV')
bx = g1.loc[g1.B_exo_date.idxmax()]; bn = g1.loc[g1.B_endo_date.idxmax()]
say(f'  1 TeV best-vs-best with date: B(exo/endo) = {bx.B_exo_date / bn.B_endo_date:.3g}; without date {g1.B_exo.max() / g1.B_endo.max():.3g}; all-mass best-vs-best with date {GI.B_exo_date.max() / GI.B_endo_date.max():.3g}')

# P058 comparison table (1 TeV, iso and p)
cmp_rows = []
P058 = {('iso', 278.0): (0.997, 6.3, 2.90, 3.00), ('iso', 300.0): (0.996, 5.9, 2.94, 3.05), ('iso', 350.0): (0.993, 4.9, 3.00, 3.19), ('iso', 380.0): (0.992, 4.3, 3.03, 2.34),
        ('p', 300.0): (0.984, 2.9, 3.13, 3.23), ('p', 380.0): (0.971, 2.2, 3.20, 2.97)}
for (tag, d), (pc, gph, z4x, z4n) in P058.items():
    rx = spectrum(tag, 1000.0, -d, ETA['annual']); rn = spectrum(tag, 1000.0, +d, ETA['annual']); ax_, an_ = analyse(rx), analyse(rn)
    fx, fx4, fn_, fn4 = fit(ax_), fit(ax_, use_density=False), fit(an_), fit(an_, use_density=False)
    cmp_rows.append(dict(tag=tag, delta_keV=d, pct248_P058=pc, pct248_here=ax_['pct248'], gap_per_hi_P058=gph, gap_per_hi_here=ax_['gap'] / ax_['hi'], gap_per_hi_true=ax_['gap_true'] / ax_['hi_true'],
                         Z4_exo_P058=z4x, Z4_exo_here=fx4['Z'], Z5_exo_here=fx['Z'], Z4_endo_P058=z4n, Z4_endo_here=fn4['Z'], Z5_endo_here=fn_['Z'],
                         B_exo=bayes(ax_), B_endo=bayes(an_), mu_hi_exo=fx['mu_hi'], mu_hi_endo=fn_['mu_hi'], D248_per_hi_exo=ax_['D248'] / ax_['hi'], D248_per_hi_endo=an_['D248'] / an_['hi']))
CMP = pd.DataFrame(cmp_rows); CMP.to_csv(f'{OUT}/P072_vs_P058.csv', index=False)
say('Part C: comparison with P058 §5.6 (1 TeV, annual): percentile of 248 keV, gap companions per 200-270 event, four-bin Z (P058 form) and five-term Z (with event density)')
say(CMP.to_string(index=False, float_format='%.3g'))

# robustness of the Bayes factors to the background inputs and the high-energy edge (1 TeV, iso)
say('Part C: robustness (1 TeV iso): B_exo / B_endo under b_gap x0.5/x2, b_lomid x0.5/x2, 1000 phd edge instead of 1700')
rob = []
for d in (300.0, 350.0, 500.0):
    ax_ = analyse(spectrum('iso', 1000.0, -d, ETA['annual'])); an_ = analyse(spectrum('iso', 1000.0, +d, ETA['annual']))
    base = (bayes(ax_), bayes(an_)); r = dict(delta_keV=d, B_exo=base[0], B_endo=base[1])
    for key, fac in (('gap', 0.5), ('gap', 2.0), ('lomid', 0.5), ('lomid', 2.0), ('he1700', 0.5), ('he1700', 2.0)):
        b0 = OBS[key]['b']; OBS[key]['b'] = b0 * fac
        r[f'B_exo_{key}x{fac}'] = bayes(ax_); r[f'B_endo_{key}x{fac}'] = bayes(an_); OBS[key]['b'] = b0
    ax10 = dict(ax_); ax10['he1700'] = ax_['he1000']; an10 = dict(an_); an10['he1700'] = an_['he1000']
    b0 = OBS['he1700']['b']; OBS['he1700']['b'] = 0.018; r['B_exo_he1000'] = bayes(ax10); r['B_endo_he1000'] = bayes(an10); OBS['he1700']['b'] = b0
    rob.append(r)
ROB = pd.DataFrame(rob); ROB.to_csv(f'{OUT}/P072_robustness.csv', index=False); say(ROB.to_string(index=False, float_format='%.3g'))

# ------------------------------------------------------------------------------------------------------------
# Part D: required coupling, chi1 companions, lifetime, dark photon
# ------------------------------------------------------------------------------------------------------------
say('Part D: required f2*sigma for one exothermic ROI event, chi1 up-scatter events at that coupling, dark-photon window')
SW, CW = math.sqrt(SW2), math.sqrt(1 - SW2); G_WEAK = math.sqrt(4 * math.pi * ALPHA) / SW
GEFF_UNIT = (SW / CW) * G_WEAK / (4 * CW * MZ ** 2)                    # G_eff / (g_D eps)
def tau_dp(sigma_p_cm2, mA_GeV, d_keV):
    """chi2 -> chi1 nu nubar lifetime [s] for a dark photon with sigma_p fixed (eps^2 alpha_D = X m_A'^4; P011 eq. 3, P058)."""
    X = (sigma_p_cm2 / lz.GEV_TO_CM2) / (16 * math.pi * ALPHA * MU_P ** 2)
    Gam = 3 * (4 * math.pi * X * mA_GeV ** 4) * GEFF_UNIT ** 2 * (d_keV * 1e-6) ** 5 / (120 * math.pi ** 3)
    return HBAR_GEV_S / Gam
cpl_rows = []
for tag in ('iso', 'p'):
    for m in (1000.0, 4000.0):
        for d in (200.0, 250.0, 278.0, 300.0, 350.0, 380.0, 450.0):
            g = G[(G.tag == tag) & (G.m_GeV == m) & np.isclose(G.delta_keV, d)]
            if not len(g):
                continue
            g = g.iloc[0]; f2fo = F2_FO[tag]
            row = dict(tag=tag, m_GeV=m, delta_keV=d, kappa_f2=g.kappa_f2_one_event, sigmaN_f2_cm2=g.sigmaN_f2_cm2, sigmaN_at_f2fo=g.sigmaN_f2_cm2 / f2fo,
                       kappa_endo_fit=g.kappa_endo, ratio_endo_over_exo_coupling=g.kappa_endo * f2fo * g.exo_roi if g.exo_roi > 0 else np.nan,
                       chi1_endo_events_f2fo=g.endo_events_at_exo_coupling_f2fo, chi1_endo_gap_f2fo=g.endo_gap_at_exo_coupling_f2fo,
                       # required lifetime for f2(today) = x f2,fo
                       tau_over_tU_f2_0p9=1 / math.log(1 / 0.9), tau_over_tU_f2_0p5=1 / math.log(2.0), tau_over_tU_f2_0p1=1 / math.log(10.0))
            if tag == 'p':
                S = g.sigmaN_f2_cm2                                      # sigma_p f2 needed
                # tau = Cc/(sigma_p m_A'^4); one event: f2fo sigma_p exp(-t_U sigma_p m_A'^4 / Cc) = S; max m_A'^4 = Cc f2fo/(e S t_U)
                Cc = tau_dp(1.0, 1.0, d)                                 # tau at sigma_p = 1 cm^2, m_A' = 1 GeV
                mA_max = (Cc * f2fo / (math.e * S * T_U_S)) ** 0.25
                row.update(mA_max_GeV=mA_max, tau_at_mA_max_s=T_U_S, sigma_p_at_mA_max=math.e * S / f2fo)
                for mA in (0.1, 0.3, 1.0):
                    if mA < mA_max:
                        # long-lived branch: sigma_p slightly above S/f2fo
                        fsol = lambda s: f2fo * s * math.exp(-T_U_S / tau_dp(s, mA, d)) - S
                        s_lo = brentq(fsol, S / f2fo, math.e * S / f2fo)
                        row[f'sigma_p_mA{mA}'] = s_lo; row[f'tau_mA{mA}_s'] = tau_dp(s_lo, mA, d)
                        X = (s_lo / lz.GEV_TO_CM2) / (16 * math.pi * ALPHA * MU_P ** 2)
                        row[f'eps_mA{mA}_aD0.1'] = math.sqrt(X * mA ** 4 / 0.1); row[f'eps_mA{mA}_aDrelic'] = math.sqrt(X * mA ** 4 / ALPHA_D_RELIC)
                    else:
                        row[f'sigma_p_mA{mA}'] = np.nan
            cpl_rows.append(row)
CPL = pd.DataFrame(cpl_rows); CPL.to_csv(f'{OUT}/P072_coupling_lifetime.csv', index=False)
say(CPL.to_string(index=False, float_format='%.3g'))
# gamma-line constraint (P066): f2/tau_gamma < 5e-23 /s -> BR_gamma < 5e-23 tau / f2
BR_GAMMA_MAX_TU = 5e-23 * T_U_S / 0.5
say(f'  P066 SPI bound f2/tau_gamma < 5e-23 /s with f2 = 0.5 and tau = t_U: BR_gamma < {BR_GAMMA_MAX_TU:.1e} (scales with tau/t_U)')

# ------------------------------------------------------------------------------------------------------------
# Part E: annual modulation and the 16 June date factor
# ------------------------------------------------------------------------------------------------------------
say('Part E: annual modulation (24-day halos); ROI rate vs day; cosine amplitude a1 (phase 2 June), day of maximum, R(16 June)/mean')
mod_rows = []; MODC = {}
for tag, m, d in [('iso', 1000.0, dd) for dd in (150.0, 200.0, 278.0, 300.0, 350.0, 450.0)] + [('iso', 4000.0, 300.0), ('iso', 300.0, 300.0), ('p', 1000.0, 300.0)]:
    rx = np.array([analyse(spectrum(tag, m, -d, ETA_DAY[k]))['roi'] for k in DAYS24]); rn = np.array([analyse(spectrum(tag, m, +d, ETA_DAY[k]))['roi'] for k in DAYS24])
    rx_gap = np.array([analyse(spectrum(tag, m, -d, ETA_DAY[k]))['gap'] for k in DAYS24])
    MODC[(tag, m, d)] = (rx, rn)
    def stats(y):
        if y.mean() <= 0: return np.nan, np.nan, np.nan
        ph = 2 * math.pi * (DAYS24 - 152.5) / 365.25
        cs = CubicSpline(np.concatenate([DAYS24, [DAYS24[0] + 365.25]]), np.concatenate([y, [y[0]]]), bc_type='periodic')
        return 2 * np.mean(y * np.cos(ph)) / y.mean(), float(DAYS24[np.argmax(y)]), float(cs(167.0) / y.mean())
    a1x, dmx, f167x = stats(rx); a1n, dmn, f167n = stats(rn); a1g, dmg, f167g = stats(rx_gap)
    mod_rows.append(dict(tag=tag, m_GeV=m, delta_keV=d, a1_exo_roi=a1x, day_max_exo=dmx, R167_over_mean_exo=f167x, a1_exo_gap=a1g, day_max_exo_gap=dmg,
                         a1_endo_roi=a1n, day_max_endo=dmn, R167_over_mean_endo=f167n, date_factor_exo_over_endo=f167x / f167n if f167n > 0 else np.nan))
MOD = pd.DataFrame(mod_rows); MOD.to_csv(f'{OUT}/P072_modulation.csv', index=False)
say(MOD.to_string(index=False, float_format='%.3g'))

# ------------------------------------------------------------------------------------------------------------
# Part F: two-population (f2, delta) plane
# ------------------------------------------------------------------------------------------------------------
say('Part F: two-population plane (isoscalar): spectrum = kappa[(1-f2) endothermic + f2 exothermic]; profile Z, Bayes factor, companions at the fit')
F2S = np.concatenate([[0.0], np.logspace(-8, 0, 33)])
tp_rows = []
for m in (1000.0, 4000.0):
    for d in DELTAS:
        rx, rn = SPEC[('iso', m, d)]
        ax_, an_ = analyse(rx), analyse(rn)
        for f2 in F2S:
            N = {k: (1 - f2) * an_[k] + f2 * ax_[k] for k in ('lomid', 'gap', 'hi', 'roi', 'he1700', 'D248')}
            f = fit(N)
            tp_rows.append(dict(m_GeV=m, delta_keV=d, f2=f2, Z5=f['Z'], B=bayes(N), kappa_hat=f['kappa_hat'], mu_hi=f['mu_hi'], mu_gap=f['mu_gap'], mu_lomid=f['mu_lomid'], mu_he=f['mu_he'],
                                exo_fraction_of_hi=(f2 * ax_['hi'] / N['hi']) if N['hi'] > 0 else np.nan, exo_fraction_of_roi=(f2 * ax_['roi'] / N['roi']) if N['roi'] > 0 else np.nan))
TP = pd.DataFrame(tp_rows); TP.to_csv(f'{OUT}/P072_two_population_plane.csv', index=False)
for m in (1000.0, 4000.0):
    t = TP[TP.m_GeV == m]; b = t.loc[t.B.idxmax()]
    say(f'  m = {m:.0f}: best (f2, delta) = ({b.f2:.3g}, {b.delta_keV:.0f} keV): B = {b.B:.3g}, Z5 = {b.Z5:.2f}, mu_hi = {b.mu_hi:.3f}, mu_gap = {b.mu_gap:.3f}, exo fraction of 200-270 events = {b.exo_fraction_of_hi:.3g}')
    # f2 at which B drops by x2 and x10 from the f2=0 value, at delta = 300, 350, 366-ish
    for d in (300.0, 350.0, 380.0):
        tt = t[np.isclose(t.delta_keV, d)].sort_values('f2'); B0 = float(tt.B.iloc[0])
        below = tt[tt.B < 0.5 * B0]; f_half = float(below.f2.iloc[0]) if len(below) else np.nan
        below10 = tt[tt.B < 0.1 * B0]; f_10 = float(below10.f2.iloc[0]) if len(below10) else np.nan
        say(f'    delta = {d:.0f}: B(f2=0) = {B0:.3g}; B halves at f2 ~ {f_half:.2g}, falls x10 at f2 ~ {f_10:.2g}; B(f2=1) = {float(tt.B.iloc[-1]):.3g}')

# ------------------------------------------------------------------------------------------------------------
# Part G: low-dispersion and dark-disk halos (how narrow can the line be?)
# ------------------------------------------------------------------------------------------------------------
say('Part G: halo variants at 1 TeV with |delta| = 278 keV (E* = 248 keV) and 4 TeV / 256 keV: width, companions, likelihood')
hal_rows = []; HSPEC = {}
HALOS_VAR = {'SHM sun (v0=238, vE=251)': eta_maxwell(250.6, 238.0), 'SHM annual (WimPyDD)': ETA['annual'],
             'v0=150, vE=251': eta_maxwell(250.6, 150.0), 'v0=100, vE=251': eta_maxwell(250.6, 100.0), 'v0=50, vE=251': eta_maxwell(250.6, 50.0),
             'disk lag 100, sig 50': eta_maxwell(100.0, 50.0 * math.sqrt(2)), 'disk lag 50, sig 50': eta_maxwell(50.0, 50.0 * math.sqrt(2)),
             'disk lag 50, sig 20': eta_maxwell(50.0, 20.0 * math.sqrt(2)), 'disk lag 20, sig 20': eta_maxwell(20.0, 20.0 * math.sqrt(2)),
             'disk lag 10, sig 10': eta_maxwell(10.0, 10.0 * math.sqrt(2))}
for m, d in ((1000.0, 278.0), (4000.0, 256.0), (1000.0, 300.0)):
    for hname, eta in HALOS_VAR.items():
        rx = spectrum('iso', m, -d, eta); ax_ = analyse(rx); f = fit(ax_); HSPEC[(m, d, hname)] = rx
        hal_rows.append(dict(m_GeV=m, delta_keV=d, halo=hname, exo_roi=ax_['roi'], exo_all=ax_['all'], mode=ax_['mode'], fwhm=ax_['fwhm'], p16=ax_['p16'], p84=ax_['p84'], pct248=ax_['pct248'],
                             gap_per_hi=ax_['gap'] / ax_['hi'] if ax_['hi'] > 0 else np.nan, lomid_per_hi=ax_['lomid'] / ax_['hi'] if ax_['hi'] > 0 else np.nan,
                             he1700_per_hi=ax_['he1700'] / ax_['hi'] if ax_['hi'] > 0 else np.nan, f_hi_of_roi=ax_['f_hi_of_roi'], Z5=f['Z'], mu_hi=f['mu_hi'], mu_gap=f['mu_gap'], B=bayes(ax_),
                             frac_hi_of_all=ax_['hi'] / ax_['all'] if ax_['all'] > 0 else np.nan))
    # disk + SHM mixtures: rho_disk/rho_0 = 0.25, 1 with the (lag 50, sig 20) disk on top of the annual SHM
    for fr in (0.25, 1.0):
        rx = spectrum('iso', m, -d, ETA['annual']) + fr * spectrum('iso', m, -d, HALOS_VAR['disk lag 50, sig 20']); ax_ = analyse(rx); f = fit(ax_)
        hal_rows.append(dict(m_GeV=m, delta_keV=d, halo=f'SHM + disk(lag 50, sig 20) rho_d/rho_0 = {fr}', exo_roi=ax_['roi'], exo_all=ax_['all'], mode=ax_['mode'], fwhm=ax_['fwhm'], p16=ax_['p16'], p84=ax_['p84'],
                             pct248=ax_['pct248'], gap_per_hi=ax_['gap'] / ax_['hi'], lomid_per_hi=ax_['lomid'] / ax_['hi'], he1700_per_hi=ax_['he1700'] / ax_['hi'], f_hi_of_roi=ax_['f_hi_of_roi'],
                             Z5=f['Z'], mu_hi=f['mu_hi'], mu_gap=f['mu_gap'], B=bayes(ax_), frac_hi_of_all=ax_['hi'] / ax_['all']))
HAL = pd.DataFrame(hal_rows); HAL.to_csv(f'{OUT}/P072_halo_variants.csv', index=False)
say(HAL.to_string(index=False, float_format='%.3g'))
# total exothermic rate vs halo (rate ~ rho, independent of v to leading order): all-energy counts per unit coupling
say('  total exothermic counts (all energies, per unit coupling, 1 TeV, 278 keV) per halo: ' +
    ', '.join(f"{h}: {HAL[(HAL.m_GeV == 1000) & (HAL.delta_keV == 278) & (HAL.halo == h)].exo_all.iloc[0]:.3g}" for h in list(HALOS_VAR)[:6]))

# ------------------------------------------------------------------------------------------------------------
# Part H: figures
# ------------------------------------------------------------------------------------------------------------
PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
INK, INK2, MUTED, SURF, GRIDC = '#0b0b0b', '#52514e', '#898781', '#fcfcfb', '#e1e0d9'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'axes.facecolor': SURF, 'figure.facecolor': SURF, 'axes.grid': True, 'grid.color': GRIDC, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})

# Fig 1: observed accepted spectra, exo vs endo, plus narrow-line halo variants
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4))
ax = axs[0]
ax.axvspan(125, 200, color='#f0efec', lw=0, zorder=0); ax.text(162, 0.58, 'empty\n125–200 keV', ha='center', color=INK2, fontsize=8, transform=ax.get_xaxis_transform())
for i, d in enumerate((278.0, 300.0, 350.0, 450.0)):
    rx = spectrum('iso', 1000.0, -d, ETA['annual']); s = SMEAR @ (rx * EPS_ROI); s = s / s.sum()
    ax.plot(EOBS, s, color=PAL[i], lw=2, label=f'χ₂ down-scatter, |δ| = {d:.0f} keV (E* = {estar(1000.0, d):.0f})')
for j, d in enumerate((350.0, 380.0)):
    rn = spectrum('iso', 1000.0, +d, ETA['annual']); s = SMEAR @ (rn * EPS_ROI); s = s / s.sum()
    ax.plot(EOBS, s, color=[INK2, MUTED][j], lw=1.5, ls='--', label=f'χ₁ up-scatter, δ = {d:.0f} keV')
ax.axvline(248, color=INK, lw=1, ls=':'); ax.text(251, 0.0225, 'event', color=INK, fontsize=8)
ax.set_xlim(0, 300); ax.set_ylim(0, 0.025); ax.set_xlabel('observed recoil energy [keV]'); ax.set_ylabel('accepted spectrum, normalised to unit area [keV⁻¹]')
ax.set_title('Isoscalar O₁, 1 TeV, annual Baxter halo: where accepted events land', fontsize=9, loc='left'); ax.legend(fontsize=7.5, loc='upper left')
ax = axs[1]
for i, h in enumerate(('SHM annual (WimPyDD)', 'v0=100, vE=251', 'disk lag 100, sig 50', 'disk lag 50, sig 20', 'disk lag 10, sig 10')):
    rx = HSPEC[(1000.0, 278.0, h)]; s = SMEAR @ (rx * EPS_ROI); s = s / s.sum()
    ax.plot(EOBS, s, color=PAL[i], lw=2, label=h)
ax.axvspan(125, 200, color='#f0efec', lw=0, zorder=0); ax.axvline(248, color=INK, lw=1, ls=':')
ax.set_xlim(0, 300); ax.set_xlabel('observed recoil energy [keV]'); ax.set_ylabel('accepted spectrum, unit area [keV⁻¹]')
ax.set_title('|δ| = 278 keV (E* = 248 keV), 1 TeV: only a slow co-moving population gives a line', fontsize=8.5, loc='left'); ax.legend(fontsize=7.5)
fig.tight_layout(); fig.savefig(f'{FIG}/P072_fig1_spectra.png', dpi=160); plt.close(fig)

# Fig 2: (m, |delta|) maps: companions per 200-270 event, and log10 B_exo
fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
piv = lambda col: GI.pivot(index='m_GeV', columns='delta_keV', values=col)
for ax, col, lab, cmap, vmin, vmax in ((axs[0], 'gap_per_hi', 'events in 125–200 keV per event in 200–270 keV', 'Blues', 0, 12),
                                        (axs[1], 'exo_pct248', 'percentile of 248 keV in the accepted spectrum', 'Blues', 0.9, 1.0),
                                        (axs[2], 'B_exo', 'exothermic Bayes factor vs background (flat μ₂₀₀₋₂₇₀ ∈ [0, 10])', 'Blues', 0, 5)):
    P = piv(col); xe = np.concatenate([P.columns.values - 5.0, [P.columns.values[-1] + 5.0]])
    im = ax.pcolormesh(xe, np.arange(len(P.index) + 1) - 0.5, P.values, cmap=cmap, vmin=vmin, vmax=vmax, shading='flat')
    ax.set_yticks(np.arange(len(P.index))); ax.set_yticklabels([f'{m / 1000:g}' for m in P.index]); ax.set_ylabel('m_χ [TeV]'); ax.set_xlabel('|δ| [keV]')
    ax.set_title(lab, fontsize=9, loc='left'); ax.grid(False); fig.colorbar(im, ax=ax, shrink=0.85)
    for k, m in enumerate(P.index):
        mu = m * MN_MEAN / (m + MN_MEAN); ax.plot([248.0 * MN_MEAN / mu], [k], 'o', color=PAL[1], ms=5, mec=SURF)
axs[0].text(105, 5.3, 'orange: E* = 248 keV', color=PAL[1], fontsize=7.5)
Pn = piv('B_endo'); cs = axs[2].contour(Pn.columns.values, np.arange(len(Pn.index)), np.nan_to_num(Pn.values), levels=[2, 4, 8], colors=[PAL[1]], linewidths=1)
axs[2].clabel(cs, fmt='%g', fontsize=7); axs[2].text(105, -0.3, 'orange contours: endothermic B = 2, 4, 8', color=PAL[1], fontsize=7.5)
fig.savefig(f'{FIG}/P072_fig2_maps.png', dpi=160); plt.close(fig)

# Fig 3: (f2, delta) plane at 1 TeV: Bayes factor and fitted gap companions
fig, axs = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
t = TP[(TP.m_GeV == 1000.0) & (TP.f2 > 0)]
for ax, col, lab, vmin, vmax in ((axs[0], 'B', 'Bayes factor vs background (two populations, 1 TeV)', 0, 5), (axs[1], 'mu_gap', 'fitted expectation in 125–200 keV [events]', 0, 1.0)):
    P = t.pivot(index='f2', columns='delta_keV', values=col)
    im = ax.pcolormesh(P.columns.values, np.log10(P.index.values), P.values, cmap='Blues', vmin=vmin, vmax=vmax, shading='nearest')
    ax.set_xlabel('δ [keV]'); ax.set_ylabel('log₁₀ f₂ (χ₂ fraction today)'); ax.set_title(lab, fontsize=9, loc='left'); ax.grid(False); fig.colorbar(im, ax=ax, shrink=0.85)
for d, f in ((300, 2.8e-3), (350, 5.7e-5), (366, 4.8e-6), (380, 1.7e-7)):
    axs[0].plot([d], [math.log10(f)], 's', color=PAL[1], ms=5, mec=SURF)
axs[0].text(255, -7.6, 'orange: P058 90 % f₂ limits (μ_endo = 1)', color=PAL[1], fontsize=7.5)
fig.savefig(f'{FIG}/P072_fig3_f2_delta_plane.png', dpi=160); plt.close(fig)

# Fig 4: modulation
fig, ax = plt.subplots(figsize=(7.2, 4.0))
for i, d in enumerate((278.0, 300.0, 450.0)):
    rx, rn = MODC[('iso', 1000.0, d)]; ax.plot(DAYS24, rx / rx.mean(), color=PAL[i], lw=2, marker='o', ms=4, mec=SURF, label=f'χ₂ down-scatter ROI rate, |δ| = {d:.0f} keV')
rx, rn = MODC[('iso', 1000.0, 350.0)]; ax.plot(DAYS24, rn / rn.mean(), color=INK2, lw=1.4, ls='--', label='χ₁ up-scatter, δ = 350 keV')
ax.axvline(167, color=MUTED, lw=0.8, ls='-.'); ax.text(170, 0.3, '16 June', color=MUTED, fontsize=8)
ax.set_xlabel('day of year'); ax.set_ylabel('rate / annual mean'); ax.set_xlim(0, 365); ax.set_ylim(0, 2.6)
ax.set_title('A down-scattering population is time-flat to a few per cent', fontsize=9, loc='left'); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f'{FIG}/P072_fig4_modulation.png', dpi=160); plt.close(fig)

# ------------------------------------------------------------------------------------------------------------
# results JSON
# ------------------------------------------------------------------------------------------------------------
g1 = GI[GI.m_GeV == 1000.0]
res = dict(constants=dict(C_UNIT=C_UNIT, SIGMA_UNIT_cm2=SIGMA_UNIT, MN_MEAN_GeV=MN_MEAN, E50_1700=E50_1700, t_U_s=T_U_S, NLL0=NLL0),
           kernel_validation=val, observed=OBS, b_hi=B_HI, validation_RROI=vdf.to_dict(orient='records'), edges=EDGE.to_dict(orient='records'),
           best=BEST.to_dict(orient='records'), marginal=dict(B_exo=B_exo_all, B_endo=B_endo_all, B_exo_over_endo=B_exo_all / B_endo_all),
           vs_P058=CMP.to_dict(orient='records'), coupling=CPL.to_dict(orient='records'), modulation=MOD.to_dict(orient='records'),
           halo_variants=HAL.to_dict(orient='records'), grid_1TeV_iso=g1.to_dict(orient='records'), BR_gamma_max_at_tU=BR_GAMMA_MAX_TU, runtime_s=time.time() - T0)
json.dump(res, open(f'{OUT}/P072_results.json', 'w'), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
say(f'done in {time.time() - T0:.0f} s')
LOG.close()
