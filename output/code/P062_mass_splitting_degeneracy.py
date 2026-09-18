"""
P062 -- Mass-splitting degeneracy: why a single recoil cannot fix m_chi above 400 GeV, what the spectral
        endpoint and onset do, and the (m, delta) banana.

Run from the simulation root:   .venv/bin/python output/code/P062_mass_splitting_degeneracy.py  [--recompute]

Method
  For the velocity-independent operator O1 WimPyDD's differential rate factorises exactly (P046 method)
      dR/dE (E; m, delta, halo) = (1000 GeV / m) * sum_isotopes K_i(E) * eta(v_min,i(E; m, delta)),
  with K_i(E) obtained once (m = 1000 GeV) from a single high-velocity stream with delta_eta = 1 (km/s)^-1,
  and eta(v) = sum_{v_j > v} delta_eta_j from lzcommon's WimPyDD halo functions (Baxter-2021 SHM).  The exact
  1/m scaling of K_i is verified numerically below (WimPyDD O1 response carries no other m_chi dependence).
  Observed-energy pdfs: (dR/dE * eff) convolved with a Gaussian of sigma_E = 11 sqrt(E/248) keV (P009/P021),
  restricted to an "extended" xenon window 5-1000 keV (ideal efficiency) or an LZ-like window (0.96 plateau,
  erf edges at 5.4 / 269.9 keV with sigma 2.5 / 11.5 keV, P021/P046).

Parts
  1. (m, delta) grid spectra (Sun-frame, 16 June, annual-mean halos), pairwise KL divergences, KL maps and the
     "banana" delta_b(m) that best mimics the reference spectra (1000 GeV, 366 keV) and (1000 GeV, 300 keV);
     comparison with delta*mu/m_N = const (E* matched), onset E-(v_max) matched, v_min* matched, E+ matched.
  2. Fisher information in (ln m, delta) per event: shape only, plus the rate term for fixed-coupling models;
     eigen-decomposition, ridge slope, sigma(delta) with m fixed / free; N for 3 sigma separation along the banana.
  3. Annual modulation (24-day sampling) along the banana: amplitude, phase, June/Dec ratio, N to separate.
  4. Endpoint in xenon: E+(m, delta; v_max), fraction of the observed rate above 400 / 500 keV and within
     50 keV of E+.
  5. Fixed-coupling (Higgsino / multiplet) rate normalisation: delta(N=1) vs m (anchored to P007 at 1 TeV),
     d ln N / d ln m along the shape banana, and an interpretation table for 300 / 1100 / 4000 GeV.
Outputs: output/work/P062/*.csv|json|npz, figures in output/work/P062/figures/, run_log.txt
"""
import sys, os, math, json, time
import numpy as np
from scipy import special, optimize
sys.path.insert(0, 'output/code')
from common import lzcommon as lz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT = 'output/work/P062'; FIG = OUT + '/figures'
os.makedirs(FIG, exist_ok=True)
T0 = time.time()
LOG = open(f'{OUT}/run_log.txt', 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n'); LOG.flush()

WD = lz.wd()
Xe = WD.Xe
MN_I = np.array(Xe.mass, dtype=float)             # isotope masses [GeV]
ABUND = np.array(Xe.abundance, dtype=float)
MN = float(np.sum(MN_I * ABUND) / np.sum(ABUND))  # abundance-weighted nuclear mass used in the analytic kinematics
MV = lz.M_V_GEV
C_WD = 3.0e5                                       # WimPyDD's c in km/s (matches its get_vmin)
EXPO_LZ = lz.LZ['exposure_tyr']
say(f'Xe isotopes A={list(Xe.a)}; abundance-weighted m_N = {MN:.3f} GeV; LZ exposure {EXPO_LZ} t yr')

# --------------------------------------------------------------------------------------------------
# Part 0: kernels K_i(E) at 1000 GeV (unit LZ/Anand coupling c1^s = 1/m_v^2  <=> WimPyDD c^0 = 2/m_v^2, P003)
# --------------------------------------------------------------------------------------------------
E_FINE = np.arange(1.0, 1300.0 + 0.5, 1.0)          # 1 keV grid, true recoil energy
H_O1 = lz.wd_hamiltonian('P062_O1_iso_unit', {1: (2.0 / MV ** 2, 0.0)})
STREAM = (np.array([3000.0]), np.array([1.0]))
M_K = 1000.0
CACHE = f'{OUT}/kernels_xe_o1_1000gev.npz'
if os.path.exists(CACHE) and '--recompute' not in sys.argv:
    z = np.load(CACHE); KERN = [z[f'K{i}'] for i in range(Xe.n_isotopes)]
    say(f'kernels loaded from cache {CACHE}')
else:
    KERN = []
    for i in range(Xe.n_isotopes):
        k = lz.wd_rate(H_O1, M_K, E_FINE, halo=STREAM, delta_kev=0.0, isotopes_list={0: [i]})
        KERN.append(np.clip(np.atleast_1d(k), 0.0, None))
        say(f'  kernel isotope {Xe.a[i]} done ({time.time()-T0:.0f} s)')
    np.savez(CACHE, **{f'K{i}': KERN[i] for i in range(Xe.n_isotopes)}, E_FINE=E_FINE)
KSUM = np.sum(KERN, axis=0)
# checks: (a) sum of per-isotope kernels = full-target call; (b) exact 1/m scaling of the kernel
E_CHK = np.array([20., 100., 250., 400., 700., 1000.])
full = np.atleast_1d(lz.wd_rate(H_O1, M_K, E_CHK, halo=STREAM, delta_kev=0.0))
say('check (a) sum_i K_i / full-target kernel:', np.round(np.interp(E_CHK, E_FINE, KSUM) / full, 6))
SCALE_CHK = {}
for m in (200.0, 400.0, 4000.0, 10000.0):
    km = np.atleast_1d(lz.wd_rate(H_O1, m, E_CHK, halo=STREAM, delta_kev=0.0))
    SCALE_CHK[m] = (km / full * m / M_K).tolist()
    say(f'check (b) K(m={m:.0f})/K(1000) * m/1000 =', np.round(SCALE_CHK[m], 6))

# --------------------------------------------------------------------------------------------------
# halos
# --------------------------------------------------------------------------------------------------
class Halo:
    def __init__(self, vmin, deta, label):
        self.vmin = np.asarray(vmin); self.deta = np.asarray(deta); self.label = label
        self.cum = np.cumsum(self.deta[::-1])[::-1]
        nz = np.nonzero(self.deta > 0)[0]
        self.vmax = float(self.vmin[nz.max()]) if nz.size else 0.0
    def eta(self, v):
        idx = np.searchsorted(self.vmin, v, side='right')
        out = np.zeros(np.shape(v)); ok = idx < len(self.cum); out[ok] = self.cum[idx[ok]]; return out

VGRID = np.linspace(0.0, lz.VESC_KMS + 300.0, 1200)
def make_halo(day=None, annual=False, label=''):
    if annual:
        days12 = 15.0 + 365.25 / 12 * np.arange(12)
        deta = np.mean([lz.wd_halo(day_of_year=d, vmin=VGRID)[1] for d in days12], axis=0)
    else:
        _, deta = lz.wd_halo(day_of_year=day, vmin=VGRID)
    return Halo(VGRID, deta, label)

HALOS = {'sunframe': make_halo(None, label='Sun-frame (v_E = 250.6 km/s, no orbital motion)'),
         'june16': make_halo(167, label='16 June'), 'dec16': make_halo(350, label='16 December'),
         'annual': make_halo(annual=True, label='12-day annual mean')}
DAYS24 = 365.25 / 24 * (np.arange(24) + 0.5)
HALO_DAYS = [make_halo(d, label=f'day {d:.1f}') for d in DAYS24]
say('halos: v_max sunframe/june16/dec16/annual(max) = %.1f / %.1f / %.1f / %.1f km/s (%.0f s)' %
    (HALOS['sunframe'].vmax, HALOS['june16'].vmax, HALOS['dec16'].vmax, HALOS['annual'].vmax, time.time() - T0))

# --------------------------------------------------------------------------------------------------
# kinematics and spectra
# --------------------------------------------------------------------------------------------------
def mu_of(m, mN=MN):
    return m * mN / (m + mN)

def vmin_iso(E, mN, m, delta):
    mu = mu_of(m, mN)
    return (mN * E * 1e-6 / mu + delta * 1e-6) / np.sqrt(2 * mN * E * 1e-6) * C_WD

def spectrum(m, delta, halo, E=E_FINE):
    """dR/dE [events/(t yr keV)] for natural Xe, O1 isoscalar, unit LZ coupling."""
    r = np.zeros(np.shape(E))
    for i in range(Xe.n_isotopes):
        Ki = KERN[i] if E is E_FINE else np.interp(E, E_FINE, KERN[i], left=0.0, right=0.0)
        r += Ki * halo.eta(vmin_iso(E, MN_I[i], m, delta))
    return r * (M_K / m)

def ceiling(m, halo):
    """delta_max = mu v_max^2 / 2 [keV] -- above it no recoil at any energy."""
    return mu_of(m) * (halo.vmax / C_WD) ** 2 / 2 * 1e6

def E_star(m, delta):
    return mu_of(m) * delta / MN                    # keV, minimum of v_min(E)

def vmin_star(m, delta):
    return math.sqrt(2 * delta * 1e-6 / mu_of(m)) * C_WD   # km/s

def E_pm(m, delta, v):
    mu = mu_of(m); b2 = (v / C_WD) ** 2
    x = delta * 1e-6 / (mu * b2)
    if x > 0.5: return float('nan'), float('nan')
    pref = mu ** 2 * b2 / MN * 1e6
    return pref * (1 - x - math.sqrt(1 - 2 * x)), pref * (1 - x + math.sqrt(1 - 2 * x))

# validation of the factorised spectrum against direct WimPyDD calls
VAL = []
for (m, d, hk, Es) in [(1000., 300., 'sunframe', [100., 200., 400., 700.]), (400., 320., 'june16', [150., 250., 350.]),
                       (4000., 366., 'sunframe', [250., 350., 500.]), (1000., 366., 'annual', [230., 330., 450.])]:
    h = HALOS[hk]
    direct = np.atleast_1d(lz.wd_rate(H_O1, m, Es, halo=(h.vmin, h.deta), delta_kev=d))
    fact = spectrum(m, d, h, np.array(Es))
    for e, a, b in zip(Es, direct, fact):
        VAL.append(dict(m=m, delta=d, halo=hk, E=e, direct=float(a), factorised=float(b), ratio=float(b / a) if a else None))
say('validation factorised/direct:', [(v['m'], v['delta'], v['halo'], v['E'], None if v['ratio'] is None else round(v['ratio'], 4)) for v in VAL])

# --------------------------------------------------------------------------------------------------
# observed-energy pdfs
# --------------------------------------------------------------------------------------------------
def sigma_E(E):
    return 11.0 * np.sqrt(E / 248.0)
def smear_matrix(sig_fn):
    s = sig_fn(E_FINE)
    SM = np.exp(-0.5 * ((E_FINE[:, None] - E_FINE[None, :]) / s[None, :]) ** 2) / (math.sqrt(2 * math.pi) * s[None, :])
    return SM * 1.0    # 1 keV bins
SM_SQRT = smear_matrix(sigma_E)
SM_11 = smear_matrix(lambda E: 11.0 * np.ones_like(E))
def eff_LZ(E, sig_hi=11.5, sig_lo=2.5):
    return lz.LZ['eff_plateau'] * 0.5 * (1 + special.erf((E - lz.LZ['E_50pct_low_keV']) / (math.sqrt(2) * sig_lo))) \
        * 0.5 * (1 - special.erf((E - lz.LZ['E_50pct_high_keV']) / (math.sqrt(2) * sig_hi)))
EFF_LZ = eff_LZ(E_FINE)
W_EXT = (E_FINE >= 5.0) & (E_FINE <= 1000.0)
W_LZ = (E_FINE >= 1.0)
MODES = {'ext': (np.ones_like(E_FINE), W_EXT, SM_SQRT, 'extended xenon window 5-1000 keV, ideal efficiency, sigma_E = 11 sqrt(E/248) keV'),
         'ext11': (np.ones_like(E_FINE), W_EXT, SM_11, 'extended window, constant 11 keV resolution'),
         'lz': (EFF_LZ, W_LZ, SM_SQRT, 'LZ-like efficiency (0.96, erf edges 5.4/269.9 keV)')}

def obs_pdf(m, delta, halo, mode='ext'):
    """returns (p normalised on the window grid, rate R [events/(t yr)] in the window, unit coupling)"""
    eff, W, SM, _ = MODES[mode]
    s = spectrum(m, delta, halo) * eff
    o = SM @ s
    o = o[W]
    R = float(np.sum(o))
    if R <= 0: return None, 0.0
    return o / R, R

def kl(p, q, floor=1e-300):
    mk = p > 0
    return float(np.sum(p[mk] * np.log(p[mk] / np.maximum(q[mk], floor))))

def n_separate(p, q, z=3.0):
    """events for a z-sigma separation of two fixed hypotheses (mean LLR vs its sd, both directions)"""
    mk = (p > 1e-14) & (q > 1e-14)
    l = np.log(p[mk] / q[mk])
    kpq = float(np.sum(p[mk] * l)); vp = float(np.sum(p[mk] * l ** 2) - kpq ** 2)
    kqp = float(-np.sum(q[mk] * l)); vq = float(np.sum(q[mk] * l ** 2) - kqp ** 2)
    return (z * (math.sqrt(max(vp, 0)) + math.sqrt(max(vq, 0))) / (kpq + kqp)) ** 2, kpq, kqp

def ks_stat(p, q):
    return float(np.max(np.abs(np.cumsum(p) - np.cumsum(q))))

EW = E_FINE[W_EXT]

# --------------------------------------------------------------------------------------------------
# Part 1: grid, pairwise KL, banana
# --------------------------------------------------------------------------------------------------
M8 = [200., 300., 400., 700., 1000., 2000., 4000., 10000.]
D_GRID = np.arange(250.0, 400.1, 10.0)
grid_rows = []; PDFS = {}
for hk in ('sunframe', 'june16'):
    h = HALOS[hk]
    for m in M8:
        for d in D_GRID:
            if d >= ceiling(m, h):
                grid_rows.append(dict(halo=hk, m=m, delta=d, physical=False)); continue
            p, R = obs_pdf(m, d, h, 'ext')
            PDFS[(hk, m, d)] = p
            Em, Ep = E_pm(m, d, h.vmax)
            cdf = np.cumsum(p)
            grid_rows.append(dict(halo=hk, m=m, delta=d, physical=True, ceiling=ceiling(m, h), E_star=E_star(m, d),
                                  vmin_star=vmin_star(m, d), E_minus=Em, E_plus=Ep, R_unit=R,
                                  E_med=float(np.interp(0.5, cdf, EW)), E_p16=float(np.interp(0.16, cdf, EW)), E_p84=float(np.interp(0.84, cdf, EW)),
                                  frac_gt400=float(np.sum(p[EW > 400])), frac_gt500=float(np.sum(p[EW > 500])),
                                  frac_last50=float(np.sum(p[EW > Ep - 50])) if Ep == Ep else float('nan')))
import csv
def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow(r)
write_csv(f'{OUT}/grid_spectra_summary.csv', grid_rows)

# pairwise KL among physical Sun-frame grid points
keys_sf = [(m, d) for m in M8 for d in D_GRID if ('sunframe', m, d) in PDFS]
KLM = np.full((len(keys_sf), len(keys_sf)), np.nan)
for a, ka in enumerate(keys_sf):
    for b, kb in enumerate(keys_sf):
        KLM[a, b] = kl(PDFS[('sunframe',) + ka], PDFS[('sunframe',) + kb])
np.savez(f'{OUT}/pairwise_kl_sunframe.npz', keys=np.array(keys_sf), KL=KLM)
# same-delta mass comparisons (LZ's "degenerate above 400 GeV" statement) incl. KS and N_3sigma
same_d = []
for d in D_GRID:
    for (m1, m2) in [(400., 4000.), (400., 1000.), (1000., 4000.), (200., 4000.), (700., 4000.)]:
        if ('sunframe', m1, d) in PDFS and ('sunframe', m2, d) in PDFS:
            p, q = PDFS[('sunframe', m1, d)], PDFS[('sunframe', m2, d)]
            n3, kpq, kqp = n_separate(p, q)
            # true-energy KS in the LZ ROI 5.4-270 keV (P002's statistic)
            s1 = spectrum(m1, d, HALOS['sunframe']); s2 = spectrum(m2, d, HALOS['sunframe'])
            roi = (E_FINE >= 5.4) & (E_FINE <= 270.0)
            ks_roi = ks_stat(s1[roi] / s1[roi].sum(), s2[roi] / s2[roi].sum()) if s1[roi].sum() > 0 and s2[roi].sum() > 0 else float('nan')
            same_d.append(dict(delta=d, m1=m1, m2=m2, KL_12=kpq, KL_21=kqp, KS_ext_obs=ks_stat(p, q), KS_true_LZROI=ks_roi, N_3sigma_ext=n3))
write_csv(f'{OUT}/same_delta_mass_pairs_sunframe.csv', same_d)
say('same-delta pairs (Sun frame):', [(r['delta'], r['m1'], r['m2'], round(r['KS_true_LZROI'], 3), round(r['N_3sigma_ext'], 1)) for r in same_d if r['m1'] == 400. and r['m2'] == 4000.])

# KL maps and bananas
M_MAP = np.unique(np.concatenate([np.array(M8), np.geomspace(200., 10000., 33)]))
D_MAP = np.arange(150.0, 400.1, 2.0)
REFS = [(1000., 366.), (1000., 300.)]
BANANA = {}; KLMAP = {}
def refine_min(x, y):
    i = int(np.nanargmin(y))
    if 0 < i < len(x) - 1 and np.isfinite(y[i - 1]) and np.isfinite(y[i + 1]):
        y0, y1, y2 = y[i - 1], y[i], y[i + 1]; den = (y0 - 2 * y1 + y2)
        if den > 0:
            dx = 0.5 * (y0 - y2) / den * (x[1] - x[0])
            return float(x[i] + dx), float(y1 - 0.125 * (y0 - y2) ** 2 / den)
    return float(x[i]), float(y[i])

def solve_match(fn, target, m, h):
    """solve fn(m, delta) = target for delta in (0, ceiling)"""
    c = ceiling(m, h)
    f = lambda d: fn(m, d) - target
    a, b = 1e-3, c - 1e-6
    try:
        if f(a) * f(b) > 0: return float('nan')
        return float(optimize.brentq(f, a, b))
    except Exception:
        return float('nan')

def _load_part1():
    """reload KL maps and banana tables saved by an earlier run (identical grid), to avoid the ~5 min recomputation"""
    zf = f'{OUT}/kl_maps.npz'
    if not os.path.exists(zf): return False
    z = np.load(zf)
    if not (np.array_equal(z['M_MAP'], M_MAP) and np.array_equal(z['D_MAP'], D_MAP)): return False
    for ref in REFS:
        for hk in ('sunframe', 'june16', 'annual'):
            fn = f'{OUT}/banana_ref{int(ref[1])}_{hk}.csv'
            if not os.path.exists(fn): return False
            rows = []
            with open(fn) as f:
                for r in csv.DictReader(f):
                    rr = {}
                    for k, v in r.items():
                        if k == 'halo': rr[k] = v
                        elif k == 'at_ceiling': rr[k] = (v == 'True')
                        else: rr[k] = float(v) if v not in ('', None) else float('nan')
                    rows.append(rr)
            BANANA[(ref, hk)] = rows
            KLMAP[(ref, hk)] = z[f'ref{int(ref[1])}_{hk}']
    return True

PART1_CACHED = ('--recompute' not in sys.argv) and _load_part1()
if PART1_CACHED:
    say('Part 1 KL maps and banana tables reloaded from output/work/P062 (run with --recompute to regenerate)')
for ref in REFS:
    for hk in ('sunframe', 'june16', 'annual'):
        if PART1_CACHED: break
        h = HALOS[hk]
        pref, Rref = obs_pdf(ref[0], ref[1], h, 'ext')
        KM = np.full((len(M_MAP), len(D_MAP)), np.nan)
        rows = []
        for a, m in enumerate(M_MAP):
            c = ceiling(m, h)
            for b, d in enumerate(D_MAP):
                if d >= c - 0.5: continue
                p, R = obs_pdf(m, d, h, 'ext')
                if p is None: continue
                KM[a, b] = kl(pref, p)
            if np.all(np.isnan(KM[a])):
                rows.append(dict(ref_m=ref[0], ref_delta=ref[1], halo=hk, m=m, ceiling=c, delta_b=float('nan'), KL_min=float('nan'))); continue
            db, klmin = refine_min(D_MAP, np.where(np.isnan(KM[a]), np.inf, KM[a]))
            db = min(db, c - 0.5)
            at_ceiling = bool(np.nanargmin(KM[a]) >= np.max(np.nonzero(~np.isnan(KM[a]))[0]) - 0)
            mu_r = mu_of(ref[0]); mu_m = mu_of(m)
            Em_ref, Ep_ref = E_pm(ref[0], ref[1], h.vmax)
            rows.append(dict(ref_m=ref[0], ref_delta=ref[1], halo=hk, m=m, ceiling=c, delta_b=db, KL_min=klmin, at_ceiling=at_ceiling,
                             delta_Estar=ref[1] * mu_r / mu_m,                              # E* = delta mu / m_N matched
                             delta_vminstar=ref[1] * mu_m / mu_r,                           # v_min* = sqrt(2 delta/mu) matched
                             delta_onset=solve_match(lambda mm, dd: E_pm(mm, dd, h.vmax)[0], Em_ref, m, h),
                             delta_endpoint=solve_match(lambda mm, dd: E_pm(mm, dd, h.vmax)[1], Ep_ref, m, h),
                             E_star_b=E_star(m, db), E_minus_b=E_pm(m, db, h.vmax)[0], E_plus_b=E_pm(m, db, h.vmax)[1],
                             R_unit_b=obs_pdf(m, db, h, 'ext')[1], R_unit_ref=Rref))
        KLMAP[(ref, hk)] = KM
        BANANA[(ref, hk)] = rows
        write_csv(f'{OUT}/banana_ref{int(ref[1])}_{hk}.csv', rows)
        sel = [r for r in rows if r['m'] in M8]
        say(f'banana ref={ref} halo={hk}: ' + '; '.join(f"m={r['m']:.0f}: db={r['delta_b']:.1f} (E*={r['delta_Estar']:.1f}, onset={r['delta_onset']:.1f}, ceil={r['ceiling']:.1f}) KL={r['KL_min']:.4f}" for r in sel))
np.savez(f'{OUT}/kl_maps.npz', M_MAP=M_MAP, D_MAP=D_MAP, **{f'ref{int(r[1])}_{hk}': KLMAP[(r, hk)] for (r, hk) in KLMAP})
say('Part 1 done (%.0f s)' % (time.time() - T0))

# --------------------------------------------------------------------------------------------------
# Part 2: Fisher information in theta = (ln m, delta [keV]) per event
# --------------------------------------------------------------------------------------------------
def fisher(m, d, halo, mode='ext', h_lnm=0.02, h_d=1.0):
    c = ceiling(m, halo)
    h_d = min(h_d, 0.45 * (c - d))
    p0, R0 = obs_pdf(m, d, halo, mode)
    pm, Rm = obs_pdf(m * math.exp(-h_lnm), d, halo, mode); pp, Rp = obs_pdf(m * math.exp(h_lnm), d, halo, mode)
    qm, Sm = obs_pdf(m, d - h_d, halo, mode); qp, Sp = obs_pdf(m, d + h_d, halo, mode)
    dp_lnm = (pp - pm) / (2 * h_lnm); dp_d = (qp - qm) / (2 * h_d)
    mk = p0 > 1e-12
    I = np.zeros((2, 2))
    g = [dp_lnm, dp_d]
    for i in range(2):
        for j in range(2):
            I[i, j] = np.sum(g[i][mk] * g[j][mk] / p0[mk])
    # rate term for a fixed-coupling model: ln R derivatives (R includes the 1/m of the number density)
    dlnR = np.array([(math.log(Rp) - math.log(Rm)) / (2 * h_lnm), (math.log(Sp) - math.log(Sm)) / (2 * h_d)])
    I_rate = np.outer(dlnR, dlnR)
    out = dict(m=m, delta=d, halo=halo.label.split(' (')[0], mode=mode, h_d=h_d,
               I_lnm_lnm=I[0, 0], I_lnm_d=I[0, 1], I_d_d=I[1, 1], dlnR_dlnm=dlnR[0], dlnR_dd=dlnR[1])
    for tag, M in (('shape', I), ('shape+rate', I + I_rate)):
        w, V = np.linalg.eigh(M)
        Minv = np.linalg.inv(M) if np.linalg.cond(M) < 1e14 else np.full((2, 2), np.inf)
        out[f'{tag}_sig_d_mfixed'] = 1 / math.sqrt(M[1, 1]); out[f'{tag}_sig_d_mfree'] = math.sqrt(Minv[1, 1])
        out[f'{tag}_sig_lnm_dfixed'] = 1 / math.sqrt(M[0, 0]); out[f'{tag}_sig_lnm_dfree'] = math.sqrt(Minv[0, 0])
        out[f'{tag}_corr'] = -Minv[0, 1] / math.sqrt(Minv[0, 0] * Minv[1, 1]) if np.isfinite(Minv[0, 0]) else float('nan')
        out[f'{tag}_ridge_dd_dlnm'] = -M[0, 1] / M[1, 1]          # delta shift per unit ln m that keeps the likelihood maximal
        out[f'{tag}_eig_small'] = w[0]; out[f'{tag}_eig_large'] = w[1]
        out[f'{tag}_eigvec_small_dd_dlnm'] = V[1, 0] / V[0, 0] if abs(V[0, 0]) > 1e-12 else float('inf')
    out['banana_slope_Estar'] = -d * MN / (m + MN)                 # d delta / d ln m along E* = const
    return out

FISH = []
fisher_points = [(400., 300.), (1000., 300.), (4000., 300.), (1000., 350.), (4000., 350.), (1000., 366.), (4000., 366.), (2000., 366.),
                 (10000., 366.), (1000., 380.), (300., 300.), (1100., 366.)]
for hk in ('annual', 'sunframe'):
    for mode in ('ext', 'lz', 'ext11'):
        if hk == 'sunframe' and mode != 'ext': continue
        for (m, d) in fisher_points:
            if d >= ceiling(m, HALOS[hk]) - 2: continue
            FISH.append(fisher(m, d, HALOS[hk], mode))
# along the bananas (annual halo, ext): Fisher at (m, delta_b(m))
for ref in REFS:
    for r in BANANA[(ref, 'annual')]:
        if r['m'] in M8 and np.isfinite(r['delta_b']) and not r.get('at_ceiling', False):
            f = fisher(r['m'], r['delta_b'], HALOS['annual'], 'ext'); f['on_banana_ref'] = ref[1]; FISH.append(f)
write_csv(f'{OUT}/fisher_table.csv', FISH)
for f in FISH:
    if f['mode'] == 'ext' and f['halo'].startswith('12-day') and (f['m'], f['delta']) in [(1000., 300.), (1000., 366.), (4000., 300.), (400., 300.)]:
        say(f"Fisher ext annual m={f['m']:.0f} d={f['delta']:.0f}: sig_d(m fixed)={f['shape_sig_d_mfixed']:.1f} keV/sqrtN, sig_d(m free)={f['shape_sig_d_mfree']:.1f}, "
            f"sig_lnm(d free)={f['shape_sig_lnm_dfree']:.2f}, corr={f['shape_corr']:.3f}, ridge dd/dlnm={f['shape_ridge_dd_dlnm']:.1f} (E* banana {f['banana_slope_Estar']:.1f}); "
            f"with rate: sig_lnm={f['shape+rate_sig_lnm_dfree']:.3f}, sig_d={f['shape+rate_sig_d_mfree']:.1f}; dlnR/dlnm={f['dlnR_dlnm']:.2f}, dlnR/dd={f['dlnR_dd']:.3f}/keV")

# N for 3 sigma separation along the banana (shape), ext and lz windows, annual and sunframe
SEP = []
for ref in REFS:
    for hk in ('annual', 'sunframe'):
        rows = {r['m']: r for r in BANANA[(ref, hk)] if r['m'] in M8}
        for (m1, m2) in [(400., 4000.), (400., 1000.), (700., 4000.), (1000., 4000.), (2000., 4000.), (1000., 10000.), (4000., 10000.), (1000., 2000.), (300., 1000.), (700., 1000.)]:
            r1, r2 = rows.get(m1), rows.get(m2)
            if r1 is None or r2 is None or not (np.isfinite(r1['delta_b']) and np.isfinite(r2['delta_b'])): continue
            for mode in ('ext', 'lz'):
                p, _ = obs_pdf(m1, r1['delta_b'], HALOS[hk], mode); q, _ = obs_pdf(m2, r2['delta_b'], HALOS[hk], mode)
                n3, kpq, kqp = n_separate(p, q)
                SEP.append(dict(ref_delta=ref[1], halo=hk, mode=mode, m1=m1, delta1=r1['delta_b'], at_ceiling1=r1['at_ceiling'], m2=m2, delta2=r2['delta_b'], at_ceiling2=r2['at_ceiling'],
                                KL_12=kpq, KL_21=kqp, KS=ks_stat(p, q), N_3sigma=n3))
write_csv(f'{OUT}/separation_along_banana.csv', SEP)
for s in SEP:
    if s['halo'] == 'annual' and s['mode'] == 'ext' and (s['m1'], s['m2']) in [(400., 4000.), (1000., 4000.), (1000., 10000.), (2000., 4000.)]:
        say(f"N_3sigma shape (annual, ext) ref {s['ref_delta']:.0f}: {s['m1']:.0f}({s['delta1']:.1f}{'*' if s['at_ceiling1'] else ''}) vs {s['m2']:.0f}({s['delta2']:.1f}) -> N={s['N_3sigma']:.0f}, KL={s['KL_12']:.4f}/{s['KL_21']:.4f}")
say('Part 2 done (%.0f s)' % (time.time() - T0))

# --------------------------------------------------------------------------------------------------
# Part 3: annual modulation along the banana
# --------------------------------------------------------------------------------------------------
def window_rate(m, d, halo, mode='ext'):
    eff, W, SM, _ = MODES[mode]
    s = spectrum(m, d, halo) * eff
    return float(np.sum((SM @ s)[W]))

def modulation(m, d, mode='ext'):
    R = np.array([window_rate(m, d, h, mode) for h in HALO_DAYS])
    if R.sum() <= 0: return None
    om = 2 * math.pi / 365.25
    A = np.vstack([np.ones_like(DAYS24), np.cos(om * DAYS24), np.sin(om * DAYS24)]).T
    c, *_ = np.linalg.lstsq(A, R, rcond=None)
    amp = math.hypot(c[1], c[2]) / c[0]; phase = (math.atan2(c[2], c[1]) / om) % 365.25
    pt = R / R.sum()
    return dict(R_days=R, a1=amp, phase_day=phase, R_mean=float(R.mean()), R_max_over_min=float(R.max() / max(R.min(), 1e-300)),
                zero_frac=float(np.mean(R <= 1e-12 * R.max())), frac_MayAug=float(np.sum(pt[(DAYS24 >= 121) & (DAYS24 <= 243)])),
                june_over_dec=window_rate(m, d, HALOS['june16'], mode) / max(window_rate(m, d, HALOS['dec16'], mode), 1e-300), p_t=pt)

MOD = []; MODP = {}
for ref in REFS:
    rows = [r for r in BANANA[(ref, 'annual')] if r['m'] in M8 and np.isfinite(r['delta_b'])]
    for r in rows:
        mo = modulation(r['m'], r['delta_b'])
        if mo is None: continue
        MODP[(ref[1], r['m'])] = mo['p_t']
        MOD.append(dict(ref_delta=ref[1], m=r['m'], delta_b=r['delta_b'], at_ceiling=r['at_ceiling'], a1=mo['a1'], phase_day=mo['phase_day'],
                        max_over_min=mo['R_max_over_min'], zero_frac=mo['zero_frac'], frac_MayAug=mo['frac_MayAug'], june_over_dec=mo['june_over_dec'],
                        R_mean_unit=mo['R_mean']))
# also fixed (m, delta) LZ grid points for reference
for (m, d) in [(400., 300.), (1000., 300.), (4000., 300.), (1000., 350.), (4000., 350.), (1000., 366.), (4000., 366.)]:
    mo = modulation(m, d)
    if mo is None: continue
    MOD.append(dict(ref_delta=float('nan'), m=m, delta_b=d, at_ceiling=False, a1=mo['a1'], phase_day=mo['phase_day'], max_over_min=mo['R_max_over_min'],
                    zero_frac=mo['zero_frac'], frac_MayAug=mo['frac_MayAug'], june_over_dec=mo['june_over_dec'], R_mean_unit=mo['R_mean']))
write_csv(f'{OUT}/modulation_banana.csv', MOD)
MODSEP = []
for ref in REFS:
    for (m1, m2) in [(400., 4000.), (700., 4000.), (1000., 4000.), (2000., 4000.), (1000., 10000.), (400., 1000.)]:
        if (ref[1], m1) in MODP and (ref[1], m2) in MODP:
            n3, k12, k21 = n_separate(MODP[(ref[1], m1)], MODP[(ref[1], m2)])
            r1 = [x for x in MOD if x['ref_delta'] == ref[1] and x['m'] == m1][0]; r2 = [x for x in MOD if x['ref_delta'] == ref[1] and x['m'] == m2][0]
            MODSEP.append(dict(ref_delta=ref[1], m1=m1, a1_1=r1['a1'], m2=m2, a1_2=r2['a1'], KL_12=k12, KL_21=k21, N_3sigma_time=n3,
                               N_3sigma_amp_approx=18.0 / (r1['a1'] - r2['a1']) ** 2 if r1['a1'] != r2['a1'] else float('inf')))
write_csv(f'{OUT}/modulation_separation.csv', MODSEP)
for r in MOD:
    say(f"modulation ref={r['ref_delta']} m={r['m']:.0f} d={r['delta_b']:.1f}: a1={r['a1']:.3f} phase day {r['phase_day']:.0f}, June/Dec={r['june_over_dec']:.2f}, May-Aug frac={r['frac_MayAug']:.2f}")
for s in MODSEP:
    say(f"time-only N_3sigma ref={s['ref_delta']:.0f}: {s['m1']:.0f} (a1={s['a1_1']:.3f}) vs {s['m2']:.0f} (a1={s['a1_2']:.3f}) -> {s['N_3sigma_time']:.0f} (amp approx {s['N_3sigma_amp_approx']:.0f})")
say('Part 3 done (%.0f s)' % (time.time() - T0))

# --------------------------------------------------------------------------------------------------
# Part 4: endpoint in xenon
# --------------------------------------------------------------------------------------------------
END = []
for hk in ('sunframe', 'june16'):
    h = HALOS[hk]
    pts = [(1000., 300.), (1000., 350.), (1000., 366.), (1000., 380.), (400., 300.), (400., 330.), (4000., 300.), (4000., 366.), (4000., 400.), (10000., 366.)]
    pts += [(r['m'], round(r['delta_b'], 1)) for r in BANANA[((1000., 366.), hk)] if r['m'] in M8 and np.isfinite(r['delta_b']) and not r['at_ceiling']]
    for (m, d) in pts:
        if d >= ceiling(m, h) - 0.5: continue
        p, R = obs_pdf(m, d, h, 'ext')
        s = spectrum(m, d, h); st = s / s.sum()
        Em, Ep = E_pm(m, d, h.vmax)
        END.append(dict(halo=hk, m=m, delta=d, E_minus=Em, E_star=E_star(m, d), E_plus=Ep, E_plus_minus_50=Ep - 50,
                        frac_gt400_obs=float(np.sum(p[EW > 400])), frac_gt500_obs=float(np.sum(p[EW > 500])),
                        frac_gt400_true=float(np.sum(st[E_FINE > 400])), frac_gt500_true=float(np.sum(st[E_FINE > 500])),
                        frac_last50_true=float(np.sum(st[E_FINE > Ep - 50])), frac_first50_true=float(np.sum(st[E_FINE < Em + 50])),
                        E99_true=float(np.interp(0.99, np.cumsum(st), E_FINE)), E01_true=float(np.interp(0.01, np.cumsum(st), E_FINE)),
                        E_median_obs=float(np.interp(0.5, np.cumsum(p), EW))))
write_csv(f'{OUT}/endpoint_xe.csv', END)
for r in END:
    if r['halo'] == 'sunframe' and r['m'] == 1000.:
        say(f"endpoint Sun-frame 1000 GeV d={r['delta']:.0f}: E-={r['E_minus']:.0f} E*={r['E_star']:.0f} E+={r['E_plus']:.0f} keV; frac>400={r['frac_gt400_obs']:.3f} >500={r['frac_gt500_obs']:.4f}; last 50 keV {r['frac_last50_true']:.4f}, first 50 keV {r['frac_first50_true']:.3f}; E99={r['E99_true']:.0f}")
say('Part 4 done (%.0f s)' % (time.time() - T0))

# --------------------------------------------------------------------------------------------------
# Part 5: fixed-coupling rate normalisation (Higgsino anchor at 1 TeV, 366 keV; P007) and interpretation
# --------------------------------------------------------------------------------------------------
def N_LZ_unit(m, d, hk='annual'):
    return EXPO_LZ * window_rate(m, d, HALOS[hk], 'lz')
A_H = 1.0 / N_LZ_unit(1000., 366.)           # Higgsino normalisation: N(1 TeV, 366 keV) = 1 (P007/P021)
say(f'Higgsino anchor: unit-coupling N_LZ(1000, 366, annual) = {1/A_H:.4g}; anchor factor (c m_v^2)^2 = {A_H:.4g} (P007: 0.0777 for the Z-coupling in LZ normalisation)')

def delta_N1(m, factor=1.0, hk='annual', N=1.0):
    c = ceiling(m, HALOS[hk])
    f = lambda d: math.log(max(A_H * factor * N_LZ_unit(m, d, hk), 1e-300) / N)
    lo = 150.0
    if f(lo) < 0: return float('nan')            # fewer than N events even at delta = 150 keV
    if f(c - 0.05) > 0: return float('nan')      # more than N events even at the kinematic ceiling: excluded at this mass
    return float(optimize.brentq(f, lo, c - 0.05))
NORM = []
for m in (300., 500., 1000., 1100., 2000., 4000., 10000.):
    row = dict(m=m, ceiling_annual=ceiling(m, HALOS['annual']), delta_N1_Y12=delta_N1(m), delta_N1_Y1=delta_N1(m, 4.0), delta_N1_Y32=delta_N1(m, 9.0), delta_N1_Y2=delta_N1(m, 16.0),
               delta_N365=delta_N1(m, N=3.65), delta_N0105=delta_N1(m, N=0.105))
    row['E_star_N1'] = E_star(m, row['delta_N1_Y12']) if np.isfinite(row['delta_N1_Y12']) else float('nan')
    NORM.append(row)
write_csv(f'{OUT}/fixed_coupling_delta_N1.csv', NORM)
say('delta(N=1) Higgsino:', [(r['m'], round(r['delta_N1_Y12'], 1)) for r in NORM], ' (P007: 310/342/366/375/376 at 300/500/1000/2000/4000)')
say('delta(N=1) 1 TeV multiplets (2Y)^2=4/9/16:', [round(r[k], 1) for r in NORM if r['m'] == 1000. for k in ('delta_N1_Y1', 'delta_N1_Y32', 'delta_N1_Y2')], ' (P037: 374/379/383)')

# counts along the shape banana for a fixed coupling: how the rate lifts the degeneracy
LEV = []
for ref in REFS:
    for r in BANANA[(ref, 'annual')]:
        if not np.isfinite(r['delta_b']): continue
        N = A_H * N_LZ_unit(r['m'], r['delta_b'])
        Next = A_H * EXPO_LZ * window_rate(r['m'], r['delta_b'], HALOS['annual'], 'ext')
        LEV.append(dict(ref_delta=ref[1], m=r['m'], delta_b=r['delta_b'], at_ceiling=r['at_ceiling'], N_LZ_higgsino=N, N_ext_higgsino_per_2p84tyr=Next))
write_csv(f'{OUT}/rate_along_banana.csv', LEV)
for ref in REFS:
    rows = [x for x in LEV if x['ref_delta'] == ref[1] and x['m'] in M8]
    say(f'Higgsino-normalised LZ counts along banana ref {ref[1]:.0f}: ' + '; '.join(f"{x['m']:.0f}: {x['N_LZ_higgsino']:.3g}{'*' if x['at_ceiling'] else ''}" for x in rows))
    # local slope d ln N / d ln m along the banana between 1000 and 4000 GeV (and 1000-2000)
    d = {x['m']: x for x in rows}
    for (m1, m2) in [(1000., 2000.), (1000., 4000.), (2000., 4000.), (4000., 10000.)]:
        if m1 in d and m2 in d and not d[m1]['at_ceiling'] and not d[m2]['at_ceiling']:
            sl = math.log(d[m2]['N_LZ_higgsino'] / d[m1]['N_LZ_higgsino']) / math.log(m2 / m1)
            say(f'   d ln N / d ln m along banana {m1:.0f}-{m2:.0f}: {sl:.2f}  -> a x5 halo-rate uncertainty (P018) = {math.log(5)/abs(sl):.2f} in ln m (factor {math.exp(math.log(5)/abs(sl)):.2f} in m)')

# interpretation table: Higgsino at 300 (non-thermal), 1100 (thermal), 4000 GeV
INT = []
p_ref = None
for m in (300., 1100., 4000.):
    d1 = delta_N1(m)
    if not np.isfinite(d1): continue
    h = HALOS['annual']
    p, R = obs_pdf(m, d1, h, 'ext')
    plz, Rlz = obs_pdf(m, d1, h, 'lz')
    Em_s, Ep_s = E_pm(m, d1, HALOS['sunframe'].vmax); Em_j, Ep_j = E_pm(m, d1, HALOS['june16'].vmax)
    mo = modulation(m, d1)
    cdf_lz = np.cumsum(plz)
    row = dict(m=m, delta_N1=d1, ceiling_june=ceiling(m, HALOS['june16']), E_star=E_star(m, d1), vmin_star=vmin_star(m, d1),
               E_minus_sun=Em_s, E_plus_sun=Ep_s, E_minus_june=Em_j, E_plus_june=Ep_j,
               pct_248_LZaccepted=float(np.interp(248.0, E_FINE[W_LZ], cdf_lz)), frac_gt400_ext=float(np.sum(p[EW > 400])),
               a1=mo['a1'], june_over_dec=mo['june_over_dec'], frac_MayAug=mo['frac_MayAug'],
               N_ext_per_LZ_event=R / Rlz, E_median_ext=float(np.interp(0.5, np.cumsum(p), EW)))
    INT.append((row, p, mo['p_t']))
for i, (row, p, pt) in enumerate(INT):
    for j, (row2, q, qt) in enumerate(INT):
        if j != i:
            row[f'N3_shape_vs_{int(row2["m"])}'] = n_separate(p, q)[0]
            row[f'N3_time_vs_{int(row2["m"])}'] = n_separate(pt, qt)[0]
write_csv(f'{OUT}/interpretation_higgsino.csv', [r for r, _, _ in INT])
for r, _, _ in INT:
    say('Higgsino interp:', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
say('Part 5 done (%.0f s)' % (time.time() - T0))

# --------------------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------------------
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET, RED = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#4a3aa7', '#e34948'
INK, INK2, SURF = '#0b0b0b', '#52514e', '#fcfcfb'
cmap_blue = LinearSegmentedColormap.from_list('blues_p062', ['#fcfcfb', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.facecolor': SURF, 'figure.facecolor': SURF})

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
for ax, ref in zip(axes, [(1000., 300.), (1000., 366.)]):
    KM = KLMAP[(ref, 'sunframe')]
    with np.errstate(divide='ignore'):
        Z = np.log10(np.where(np.isnan(KM), np.nan, np.maximum(KM, 1e-4)))
    pc = ax.pcolormesh(M_MAP, D_MAP, Z.T, cmap=cmap_blue, vmin=-3, vmax=0.5, shading='nearest', rasterized=True)
    cs = ax.contour(M_MAP, D_MAP, KM.T, levels=[0.003, 0.01, 0.03, 0.1, 0.3], colors=[INK2], linewidths=0.7)
    ax.clabel(cs, fmt=lambda v: f'KL={v:g}', fontsize=7, colors=INK2)
    rows = BANANA[(ref, 'sunframe')]
    mm = np.array([r['m'] for r in rows]); db = np.array([r['delta_b'] for r in rows]); atc = np.array([r.get('at_ceiling', True) for r in rows])
    ax.plot(mm[~atc], db[~atc], color=ORANGE, lw=2, label='KL-minimising δ_b(m)')
    ax.plot(mm, [r['delta_Estar'] for r in rows], color=AQUA, lw=1.4, ls='--', label='E* = δμ/m_N matched (δ ∝ 1/μ)')
    ax.plot(mm, [r['delta_onset'] for r in rows], color=VIOLET, lw=1.2, ls=':', label='onset E₋(v_max) matched')
    ax.plot(mm, [r['ceiling'] for r in rows], color=RED, lw=1.2, label='ceiling δ_max = μv_max²/2')
    ax.plot([r['m'] for r in NORM], [r['delta_N1_Y12'] for r in NORM], color=MAGENTA, lw=1.4, ls='-.', label='Higgsino N_LZ = 1 (annual halo)')
    for m in (400., 1000., 4000.):
        for d in (300., 350.):
            ax.plot(m, d, marker='s', ms=4, color=INK, ls='none')
    ax.plot(ref[0], ref[1], marker='*', ms=11, color=YELLOW, mec=INK, ls='none', label='reference spectrum')
    ax.set_xscale('log'); ax.set_xlim(200, 10000); ax.set_ylim(200, 400)
    ax.set_xlabel('m_χ [GeV]'); ax.set_title(f'reference (1000 GeV, {ref[1]:.0f} keV), Sun-frame halo', fontsize=9)
    ax.grid(alpha=0.25)
axes[0].set_ylabel('δ [keV]')
axes[1].legend(fontsize=7, loc='lower right', framealpha=0.9)
cb = fig.colorbar(pc, ax=axes, pad=0.01, shrink=0.9); cb.set_label('log₁₀ KL [nats per event]')
fig.suptitle('P062 Fig. 1 — the (m, δ) banana: which splitting mimics the reference xenon spectrum (5–1000 keV, 11 keV resolution). Black squares: LZ grid points.', fontsize=9)
fig.savefig(f'{FIG}/fig1_banana_kl_maps.png', dpi=160, bbox_inches='tight'); plt.close(fig)

# Fig 2: Fisher ellipses + spectra along the banana + modulation along the banana
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
ax = axes[0]
def ellipse(M, N, center, ax, color, ls, label):
    Minv = np.linalg.inv(M * N)
    w, V = np.linalg.eigh(Minv)
    t = np.linspace(0, 2 * math.pi, 200)
    pts = (V @ np.diag(np.sqrt(w)) @ np.vstack([np.cos(t), np.sin(t)]))
    ax.plot(center[0] * np.exp(pts[0]), center[1] + pts[1], color=color, ls=ls, lw=1.6, label=label)
for f in FISH:
    if f['halo'].startswith('12-day') and (f['m'], f['delta']) == (1000., 366.) and 'on_banana_ref' not in f:
        I = np.array([[f['I_lnm_lnm'], f['I_lnm_d']], [f['I_lnm_d'], f['I_d_d']]])
        Ir = I + np.outer([f['dlnR_dlnm'], f['dlnR_dd']], [f['dlnR_dlnm'], f['dlnR_dd']])
        if f['mode'] == 'ext':
            ellipse(I, 30, (1000., 366.), ax, BLUE, '-', 'shape only, 5–1000 keV, N = 30')
            ellipse(Ir, 30, (1000., 366.), ax, ORANGE, '-', 'shape + rate (fixed coupling), N = 30')
        if f['mode'] == 'lz':
            ellipse(I, 30, (1000., 366.), ax, AQUA, '--', 'shape only, LZ-like window, N = 30')
rows = BANANA[((1000., 366.), 'annual')]
ax.plot([r['m'] for r in rows if not r['at_ceiling']], [r['delta_b'] for r in rows if not r['at_ceiling']], color=INK2, lw=0.9, ls=':', label='KL banana (annual halo)')
ax.set_xscale('log'); ax.set_xlim(300, 4000); ax.set_ylim(320, 400); ax.set_xlabel('m_χ [GeV]'); ax.set_ylabel('δ [keV]')
ax.set_xticks([300, 500, 1000, 2000, 4000]); ax.set_xticklabels(['300', '500', '1000', '2000', '4000']); ax.minorticks_off()
ax.set_title('1σ Fisher ellipses at (1000 GeV, 366 keV), annual halo', fontsize=9); ax.legend(fontsize=7, loc='upper right'); ax.grid(alpha=0.25)
ax = axes[1]
cols = {1000.: BLUE, 2000.: ORANGE, 4000.: AQUA, 10000.: VIOLET, 700.: MAGENTA}
for r in BANANA[((1000., 366.), 'sunframe')]:
    if r['m'] in cols and np.isfinite(r['delta_b']):
        p, _ = obs_pdf(r['m'], r['delta_b'], HALOS['sunframe'], 'ext')
        ax.plot(EW, p, color=cols[r['m']], lw=1.5, label=f"{r['m']:.0f} GeV, δ = {r['delta_b']:.0f} keV{' (ceiling)' if r['at_ceiling'] else ''}")
ax.set_xlim(100, 700); ax.set_xlabel('observed recoil energy [keV]'); ax.set_ylabel('pdf per keV'); ax.set_title('spectra along the 366 keV banana (Sun-frame)', fontsize=9)
ax.legend(fontsize=7); ax.grid(alpha=0.25)
ax = axes[2]
for ref, col, mk in [((1000., 300.), BLUE, 'o'), ((1000., 366.), ORANGE, 's')]:
    rows = [r for r in MOD if r['ref_delta'] == ref[1] and not r['at_ceiling']]
    ax.plot([r['m'] for r in rows], [r['a1'] for r in rows], color=col, marker=mk, ms=5, lw=1.4, label=f'banana of ({ref[0]:.0f} GeV, {ref[1]:.0f} keV)')
ax.set_xscale('log'); ax.set_xlabel('m_χ [GeV]'); ax.set_ylabel('cosine modulation amplitude a₁'); ax.set_title('modulation amplitude along the banana (5–1000 keV)', fontsize=9)
ax.axhline(1.0, color=INK2, lw=0.7, ls='--'); ax.text(210, 1.03, 'a₁ > 1: rate vanishes for part of the year (non-cosine)', fontsize=7, color=INK2)
ax.legend(fontsize=7, loc='lower left'); ax.grid(alpha=0.25); ax.set_ylim(0, 1.8)
fig.suptitle('P062 Fig. 2 — breaking the degeneracy: Fisher ellipses, near-identical spectra along the banana, and the modulation amplitude', fontsize=9)
fig.savefig(f'{FIG}/fig2_fisher_spectra_modulation.png', dpi=160, bbox_inches='tight'); plt.close(fig)

# summary JSON
summary = dict(m_N_GeV=MN, v_max={k: HALOS[k].vmax for k in HALOS}, kernel_scaling_check=SCALE_CHK, validation=VAL,
               higgsino_anchor_factor=A_H, fisher_selected=[f for f in FISH if f['mode'] == 'ext' and f['halo'].startswith('12-day')],
               banana_M8={f'ref{int(ref[1])}_{hk}': [r for r in BANANA[(ref, hk)] if r['m'] in M8] for ref in REFS for hk in ('sunframe', 'june16', 'annual')},
               same_delta_pairs=same_d, separation=SEP, modulation=MOD, modulation_sep=MODSEP, endpoint=END, delta_N1=NORM, rate_along_banana=LEV,
               interpretation=[r for r, _, _ in INT], runtime_s=time.time() - T0)
def _clean(o):
    if isinstance(o, dict): return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.ndarray): return _clean(o.tolist())
    return o
json.dump(_clean(summary), open(f'{OUT}/P062_summary.json', 'w'), indent=1)
say('all done (%.0f s)' % (time.time() - T0))
