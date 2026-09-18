"""
P097 -- A dark disk or co-rotating substructure: would a slow, co-rotating dark-matter component change anything
for the 248 keV event at delta ~ 300 keV?

Run from the simulation root:   .venv/bin/python output/code/P097_dark_disk.py        (~1-2 min; kernels cached)

Method (per-stream WimPyDD kernels, as P055/P068):  dR/dE(E, t) = sum_i K(E, v_i) * delta_eta_i(t), where K is the
WimPyDD `diff_rate` response for a unit-eta stream at speed v_i (sum_over_streams=False) and delta_eta_i(t) is built
analytically from the Earth-frame eta(v_min, t) of each halo component.  Composite halo at FIXED total local density
rho_0 = 0.3 GeV/cm^3:  (1 - f) SHM + f DD.

Parts
  0  Earth velocity vector in Galactic (U,V,W) (P055's from-scratch J2000 construction; re-validated vs WimPyDD).
  1  Dark-disk component: Galactic-frame isotropic Gaussian, mean (0, V0 - v_lag, 0), dispersion sigma_DD; lab speed
     of the mean through the year (phase, amplitude), Earth-frame speed distribution on 16 June, fraction above the
     inelastic thresholds (exact non-central chi-square; truncated numerical check with P055's angular grid).
  2  Kernels: O1 elastic isoscalar and O6 (computed here, cached); L10 and Higgsino-Z inelastic from P068's cache;
     isoscalar O1 endothermic/exothermic from P058's cache.  Validation of my SHM delta_eta against lz.wd_halo.
  3  Inelastic (endothermic) in-ROI rates: SHM, pure DD (zero), composite (1 - f): 16 June, annual mean, date LR.
  4  Elastic: low-energy (5.4-55 keV) and 200-270 keV rates, N_lo vs f for O1, L10, O6; low-energy modulation.
  5  Exothermic (delta < 0): DD boost, spectrum shape (line at E* = |delta| mu/m_N), companions, modulation.
  6  Density-fraction consequences: Higgsino delta(N=1) vs f, coupling rescaling, P021's kappa-hat.
  7  Summary table and figures.
"""
import sys, os, json, math, time, datetime as dt
import numpy as np
sys.path.insert(0, 'output/code')
from common import lzcommon as lz
from scipy.special import erf, erfc
from scipy.stats import ncx2
from numpy.polynomial.legendre import leggauss
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'output/work/P097'; FIG = OUT + '/figures'; CACHE = OUT + '/cache'
for d in (OUT, FIG, CACHE):
    os.makedirs(d, exist_ok=True)
T0 = time.time()
def log(*a):
    print('[%6.1fs]' % (time.time() - T0), *a, flush=True)

V0, VESC = lz.V0_KMS, lz.VESC_KMS                    # 238, 544 km/s (Baxter 2021)
V_LSR = np.array([0.0, V0, 0.0]); V_PEC = lz.V_SUN_PEC.copy(); V_SUN = V_LSR + V_PEC; VSUN = float(np.linalg.norm(V_SUN))
V_ORB = 29.79                                          # km/s (recalled, certain)
MASS = 1000.0
EXPOSURE = lz.LZ['exposure_tyr']
E50_HI, E50_LO, PLATEAU = lz.LZ['E_50pct_high_keV'], lz.LZ['E_50pct_low_keV'], lz.LZ['eff_plateau']
MN = lz.m_nucleus_gev(lz.A_XE_MEAN); MU = lz.mu_red(MASS, MN)
E_EVENT = 248.0
RES = {}

# --------------------------------------------------------------------------------------------------
# Part 0: Earth velocity (P055 construction)
# --------------------------------------------------------------------------------------------------
EQ2GAL = np.array([[-0.0548755604, -0.8734370902, -0.4838350155],
                   [ 0.4941094279, -0.4448296300,  0.7469822445],
                   [-0.8676661490, -0.1980763734,  0.4559837762]])   # J2000 eq -> Gal (recalled, certain)
OBLIQ = math.radians(23.4393)
def ecl2eq(v):
    x, y, z = v
    return np.array([x, y * math.cos(OBLIQ) - z * math.sin(OBLIQ), y * math.sin(OBLIQ) + z * math.cos(OBLIQ)])
YEAR = 2023
def doy_of(date):
    return (date - dt.datetime(YEAR, 1, 1)).total_seconds() / 86400.0 + 1.0
T_EQUINOX = doy_of(dt.datetime(2023, 3, 20, 21, 24))   # (recalled, likely)
T_EVENT = doy_of(dt.datetime(2023, 6, 16, 21, 22, 39)) # LZ paper
OMEGA = 2 * math.pi / 365.25
def v_orb_vec(doy):
    lam = OMEGA * (doy - T_EQUINOX)
    return EQ2GAL @ ecl2eq(V_ORB * np.array([math.sin(lam), -math.cos(lam), 0.0]))
def v_obs_vec(doy):
    return V_SUN + v_orb_vec(doy)

WD = lz.wd()
val0 = []
for d in [1.0, 100.0, T_EVENT, 250.0, 340.0]:
    mine = v_orb_vec(d); wdv = np.array(WD.v_earth_sun((d - 80.0) * 2 * math.pi / 365.0, v_rot_gal=V_LSR, v_sun_rot=V_PEC))
    val0.append(dict(doy=d, diff_kms=float(np.linalg.norm(mine - wdv)), vE_mine=float(np.linalg.norm(V_SUN + mine))))
RES['earth_velocity_validation'] = dict(rows=val0, max_diff_vs_WimPyDD_kms=max(r['diff_kms'] for r in val0))
VOBS_J = v_obs_vec(T_EVENT); VE_J = float(np.linalg.norm(VOBS_J))
DOYS = np.arange(1.0, 366.0, 4.0)                      # 92 regular days for annual means and harmonic fits
DOYS_ALL = np.sort(np.concatenate([DOYS, [T_EVENT]]))
iJ = int(np.argmin(np.abs(DOYS_ALL - T_EVENT))); REG = np.array([abs(d - T_EVENT) > 1e-9 for d in DOYS_ALL])
VOBS = np.array([v_obs_vec(d) for d in DOYS_ALL]); VE = np.linalg.norm(VOBS, axis=1)
log('Part 0: v_E(16 June) = %.2f km/s; |v_orb mine - WimPyDD| <= %.2f km/s' % (VE_J, RES['earth_velocity_validation']['max_diff_vs_WimPyDD_kms']))

# --------------------------------------------------------------------------------------------------
# Part 1: dark-disk configurations and Earth-frame kinematics
# --------------------------------------------------------------------------------------------------
# (v_lag, sigma) recalled from Read et al. 2008/2009 (v_lag ~ 0-150, sigma ~ 50-90 km/s; likely) and Bruch et al. 2009
# (fiducial v_lag = 50, sigma = 50; likely).  Densities: Read+2008 rho_DD/rho_halo ~ 0.25-1.5 (likely), Purcell+2009
# ~ 0.1-0.2 for a quiescent MW (likely); Gaia constraints (Schutz+2018, Buch+2019) are SURFACE-density limits
# Sigma_DD <~ 3-6 Msun/pc^2 for thin (h <~ 100 pc) disks (uncertain) -> f_DD <~ 0.1-0.3 is a plausible, not a firm, range.
CFG = [dict(key='cold', name='cold disk (v_lag 0, sigma 30)', vlag=0.0, sig=30.0),
       dict(key='fid',  name='fiducial disk (v_lag 50, sigma 50)', vlag=50.0, sig=50.0),
       dict(key='hot',  name='hot disk (v_lag 100, sigma 90)', vlag=100.0, sig=90.0)]
FRACS = [0.1, 0.2, 0.3, 0.5]

def mu_dd(c):
    return np.array([0.0, V0 - c['vlag'], 0.0])
def vlab_mean(c, vobs):
    return float(np.linalg.norm(mu_dd(c) - vobs))

def eta_iso(v, v_lab, sigma):
    """eta(v_min) = <1/w Theta(w - v_min)> for an isotropic Gaussian of dispersion sigma whose mean moves at v_lab
    in the lab (P030 analytic form; no v_esc truncation)."""
    s = math.sqrt(2.0) * sigma
    return (erf((v + v_lab) / s) - erf((v - v_lab) / s)) / (2.0 * v_lab)
def f_speed_iso(w, v_lab, sigma):
    return w / (math.sqrt(2 * math.pi) * sigma * v_lab) * (np.exp(-(w - v_lab)**2 / (2 * sigma**2)) - np.exp(-(w + v_lab)**2 / (2 * sigma**2)))
def frac_above(V, v_lab, sigma):
    """P(|x| > V), x ~ N(m, sigma^2 I_3), |m| = v_lab: non-central chi-square with 3 dof (exact, untruncated)."""
    return float(ncx2.sf((V / sigma)**2, 3, (v_lab / sigma)**2))

def cos_fit(t, y):
    X = np.column_stack([np.ones_like(t), np.cos(OMEGA * t), np.sin(OMEGA * t)])
    a0, a, b = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(a0), float(math.hypot(a, b)), float((math.atan2(b, a) / OMEGA) % 365.25)

# truncated numerical speed density (P055's angular grid) for the tail check
NTH, NPH = 96, 64
_x, _w = leggauss(NTH); _ph = (np.arange(NPH) + 0.5) * 2 * math.pi / NPH
def angular_grid(axis):
    e3 = axis / np.linalg.norm(axis)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(e3[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(e3, tmp); e1 /= np.linalg.norm(e1); e2 = np.cross(e3, e1)
    ct = np.repeat(_x, NPH); st = np.sqrt(1 - ct**2); ph = np.tile(_ph, NTH)
    n = ct[:, None] * e3 + st[:, None] * (np.cos(ph)[:, None] * e1 + np.sin(ph)[:, None] * e2)
    return n, np.repeat(_w, NPH) * (2 * math.pi / NPH)
WFINE = np.linspace(0.0, 900.0, 1801)
def speed_density_truncated(mu, sig, v_obs, w=WFINE):
    n, wgt = angular_grid(mu - v_obs); f = np.zeros_like(w)
    for i0 in range(0, len(w), 64):
        ww = w[i0:i0 + 64]
        U = ww[:, None, None] * n[None, :, :] + v_obs[None, None, :]
        g = np.exp(-0.5 * np.sum((U - mu)**2, axis=-1) / sig**2) * (np.sum(U**2, axis=-1) <= VESC**2)
        f[i0:i0 + 64] = ww**2 * (g @ wgt)
    return f / float(np.trapezoid(f, w))

THRESH = {'v_min(ROI, delta=250)': math.sqrt(2 * 250e-6 / MU) * lz.C_KMS,           # minimum over E of v_min at delta = 250
          'v_min(248 keV, 300)': lz.vmin_kms(E_EVENT, MASS, delta_kev=300.0), '700 km/s': 700.0,
          'v_min(248 keV, 350)': lz.vmin_kms(E_EVENT, MASS, delta_kev=350.0), 'v_min(248 keV, 380)': lz.vmin_kms(E_EVENT, MASS, delta_kev=380.0),
          'v_min(248 keV, elastic)': lz.vmin_kms(E_EVENT, MASS), 'v_min(200 keV, elastic)': lz.vmin_kms(200.0, MASS),
          'v_min(55 keV, elastic)': lz.vmin_kms(55.0, MASS), 'v_min(5.4 keV, elastic)': lz.vmin_kms(5.4, MASS)}
RES['thresholds_kms'] = THRESH
rows1 = []; VLAB = {}
for c in CFG:
    vl = np.array([vlab_mean(c, VOBS[i]) for i in range(len(DOYS_ALL))]); VLAB[c['key']] = vl
    a0, A, tpk = cos_fit(DOYS_ALL[REG], vl[REG])
    vlJ = vl[iJ]; s = c['sig']
    f_J = f_speed_iso(WFINE, vlJ, s); cdf = np.cumsum(f_J) * (WFINE[1] - WFINE[0]); cdf /= cdf[-1]
    row = dict(key=c['key'], name=c['name'], v_lag=c['vlag'], sigma=s, v_gal_mean=float(np.linalg.norm(mu_dd(c))),
               v_lab_June16=vlJ, v_lab_mean_fit=a0, v_lab_amp_fit=A, v_lab_peak_doy=tpk, v_lab_min=float(vl.min()), v_lab_max=float(vl.max()),
               v_lab_max_doy=float(DOYS_ALL[np.argmax(vl)]), v_lab_min_doy=float(DOYS_ALL[np.argmin(vl)]),
               mean_speed_June16=float(np.trapezoid(WFINE * f_J, WFINE)), v50_June16=float(np.interp(0.5, cdf, WFINE)), v98_June16=float(np.interp(0.98, cdf, WFINE)),
               inv_speed_June16=eta_iso(0.0, vlJ, s), inv_speed_SHM_June16=float(lz.eta0(0.0, v_e=VE_J)),
               E_max_elastic_mean_June16=lz.E_R_range_keV(MASS, vlJ)[1], E_max_elastic_v98=lz.E_R_range_keV(MASS, float(np.interp(0.98, cdf, WFINE)))[1])
    for k, V in THRESH.items():
        row['frac_above_' + k] = frac_above(V, vlJ, s)
        row['frac_above_' + k + '_annual_max'] = max(frac_above(V, v, s) for v in vl)
    rows1.append(row)
    log('  %-38s v_lab(16 Jun) %.1f  fit mean %.1f amp %.2f peak doy %.1f  <1/v>/<1/v>_SHM %.2f  frac>700: %.1e' %
        (c['name'], vlJ, a0, A, tpk, row['inv_speed_June16'] / row['inv_speed_SHM_June16'], row['frac_above_700 km/s']))
a0, A, tpk = cos_fit(DOYS_ALL[REG], VE[REG])
RES['SHM_vE'] = dict(v_E_June16=VE_J, mean_fit=a0, amp_fit=A, peak_doy=tpk, v_E_min=float(VE.min()), v_E_max=float(VE.max()))
df1 = pd.DataFrame(rows1); df1.to_csv(f'{OUT}/dark_disk_kinematics.csv', index=False, float_format='%.5g')
# truncated numerical check of the hot tail on 16 June
c = CFG[2]; ft = speed_density_truncated(mu_dd(c), c['sig'], VOBS_J)
fa = f_speed_iso(WFINE, vlab_mean(c, VOBS_J), c['sig'])
def tail(f, V):
    return float(np.trapezoid(f[WFINE >= V], WFINE[WFINE >= V]))
RES['hot_tail_check_June16'] = {k: dict(analytic=frac_above(V, vlab_mean(c, VOBS_J), c['sig']), numerical_untruncated=tail(fa, V), numerical_truncated=tail(ft, V))
                                for k, V in THRESH.items() if V > 300}
RES['hot_tail_check_June16']['hard_max_truncated_kms'] = float(WFINE[ft > 1e-12 * ft.max()].max())
RES['hot_tail_check_June16']['mean_speed_trunc_vs_untrunc'] = [float(np.trapezoid(WFINE * ft, WFINE)), float(np.trapezoid(WFINE * fa, WFINE))]
log('  hot disk truncated check: frac > 700 km/s analytic %.2e, numerical truncated %.2e; hard max %.0f km/s' %
    (RES['hot_tail_check_June16']['700 km/s']['analytic'], RES['hot_tail_check_June16']['700 km/s']['numerical_truncated'], RES['hot_tail_check_June16']['hard_max_truncated_kms']))
json.dump({k: v.tolist() for k, v in VLAB.items()} | {'doy': DOYS_ALL.tolist(), 'v_E': VE.tolist()}, open(f'{OUT}/vlab_timeseries.json', 'w'))

# --------------------------------------------------------------------------------------------------
# Part 2: kernels and delta_eta arrays
# --------------------------------------------------------------------------------------------------
def efficiency(E, sig_hi=11.5, sig_lo=2.5):
    return PLATEAU * 0.5 * (1 + erf((E - E50_LO) / (math.sqrt(2) * sig_lo))) * 0.5 * erfc((E - E50_HI) / (math.sqrt(2) * sig_hi))
VG_A = np.linspace(0.0, 844.0, 1200)                  # P058 grid (my O1/O6 kernels use it too)
VG_B = np.linspace(0.0, 950.0, 1901)                  # P068 grid
MV = lz.M_V_GEV
ham_iso = lz.wd_hamiltonian('P097_iso_unit', {1: (2.0 / MV**2, 0.0)})    # Anand unit coupling c1^s = 1/m_v^2 (P003 convention)
ham_o6 = lz.wd_hamiltonian('P097_o6_unit', {6: (2.0 / MV**2, 0.0)})
E_EL = np.concatenate([np.arange(0.5, 60.0, 0.5), np.arange(60.0, 332.0, 2.0)])
def kernel_elastic(tag, ham, grid):
    fn = f'{CACHE}/{tag}_m1000_d0.npz'
    if os.path.exists(fn):
        z = np.load(fn); return z['E'], z['K']
    ones = np.ones_like(grid)
    K = np.array([WD.diff_rate(WD.Xe, ham, MASS, float(e), grid, ones, j_chi=0.5, sum_over_streams=False) for e in E_EL]) * 1000.0 * 365.25
    np.savez_compressed(fn, E=E_EL, K=K); return E_EL, K
log('Part 2: kernels')
K_O1 = kernel_elastic('o1', ham_iso, VG_A); log('  O1 elastic kernel ready')
K_O6 = kernel_elastic('o6', ham_o6, VG_A); log('  O6 elastic kernel ready')
zl = np.load('output/work/P068/kernel_cache/L10_m1000_d0.0.npz'); K_L10 = (zl['E'], zl['K'])
def load58(tag):
    z = np.load(f'output/work/P058/cache/{tag}.npz'); return z['E'], z['K']
D_ENDO = [250.0, 300.0, 350.0, 366.0, 380.0]; D_EXO = [278.0, 300.0, 350.0, 366.0, 380.0]
K_ISO = {d: load58('K_iso_1000_p%d' % int(d)) for d in D_ENDO}
K_EXO = {d: load58('K_iso_1000_m%d' % int(d)) for d in D_EXO}
D_HIG = np.round(np.arange(250.0, 445.1, 5.0), 1)
K_HIG = {}
for d in D_HIG:
    z = np.load('output/work/P068/kernel_cache/hig_m1000_d%.1f.npz' % d); K_HIG[float(d)] = (z['E'], z['K'])
log('  loaded P058 iso endo/exo and P068 Higgsino/L10 kernels')

def deta_from_eta_fn(eta_fn, grid):
    eta = np.asarray(eta_fn(grid), float); de = np.zeros_like(eta); de[1:] = eta[:-1] - eta[1:]
    return np.clip(de, 0.0, None)
def DE_SHM(grid):
    return np.array([deta_from_eta_fn(lambda v: lz.eta0(v, v_e=VE[i]), grid) for i in range(len(DOYS_ALL))])
def DE_DD(c, grid):
    return np.array([deta_from_eta_fn(lambda v: eta_iso(v, VLAB[c['key']][i], c['sig']), grid) for i in range(len(DOYS_ALL))])
DE = {'A': {'SHM': DE_SHM(VG_A)}, 'B': {'SHM': DE_SHM(VG_B)}}
for c in CFG:
    DE['A'][c['key']] = DE_DD(c, VG_A); DE['B'][c['key']] = DE_DD(c, VG_B)
# sanity: eta(0) reproduced by sum of delta_eta
RES['deta_closure'] = dict(SHM_A=float(DE['A']['SHM'][iJ].sum() / lz.eta0(0.0, v_e=VE_J)), fid_A=float(DE['A']['fid'][iJ].sum() / eta_iso(0.0, VLAB['fid'][iJ], 50.0)))
# validation vs lz.wd_halo on 16 June (iso, delta = 300, 366)
DE_WD = lz.wd_halo(day_of_year=T_EVENT, vmin=VG_A)[1]
def N_roi(EK, de, lo=E50_LO, hi=1e9):
    E, K = EK; m = (E >= 0) & (E <= hi)
    return float(np.trapezoid((K[m] @ de) * efficiency(E[m]), E[m])) * EXPOSURE
RES['rate_validation_vs_wd_halo'] = {int(d): dict(mine=N_roi(K_ISO[d], DE['A']['SHM'][iJ]), wd_halo=N_roi(K_ISO[d], DE_WD)) for d in [300.0, 366.0]}
for d in RES['rate_validation_vs_wd_halo']:
    r = RES['rate_validation_vs_wd_halo'][d]; r['ratio'] = r['mine'] / r['wd_halo']
log('  SHM delta_eta vs wd_halo (16 June): ratio %.3f (300), %.3f (366); closure %.4f' % (RES['rate_validation_vs_wd_halo'][300]['ratio'], RES['rate_validation_vs_wd_halo'][366]['ratio'], RES['deta_closure']['SHM_A']))

# run-window date LR (P006/P055)
d_start = dt.datetime(2023, 3, 27); RUN_DAYS = 371
t_run = np.linspace(0.0, RUN_DAYS, RUN_DAYS * 8 + 1); doy_run = (doy_of(d_start) - 1.0 + t_run) % 365.25 + 1.0
def periodic(doys, vals, x):
    dd = np.concatenate([doys - 365.25, doys, doys + 365.25]); vv = np.concatenate([vals, vals, vals]); return np.interp(x, dd, vv)
def run_mean(vals):
    return float(np.mean(periodic(DOYS_ALL, vals, doy_run)))
def series(EK, deA, lo=0.0, hi=1e9):
    E, K = EK; m = (E >= lo) & (E <= hi)
    return np.array([np.trapezoid((K[m] @ deA[i]) * efficiency(E[m]), E[m]) * EXPOSURE for i in range(len(DOYS_ALL))])

# --------------------------------------------------------------------------------------------------
# Part 3: endothermic inelastic rates
# --------------------------------------------------------------------------------------------------
log('Part 3: endothermic inelastic')
rows3 = []
for d in D_ENDO:
    Rs = series(K_ISO[d], DE['A']['SHM'])
    lr_s = Rs[iJ] / run_mean(Rs); _, a1s, pks = cos_fit(DOYS_ALL[REG], (Rs / Rs[REG].mean())[REG])
    for c in CFG:
        Rp = series(K_ISO[d], DE['A'][c['key']])
        for f in [0.0] + FRACS:
            Rc = (1 - f) * Rs + f * Rp
            rows3.append(dict(delta=d, key=c['key'], f=f, N_SHM_June16=Rs[iJ], N_SHM_annual=Rs[REG].mean(), N_pure_June16=Rp[iJ], N_pure_annual_max=Rp.max(),
                              N_comp_June16=Rc[iJ], ratio_June16=Rc[iJ] / Rs[iJ], ratio_annual=Rc[REG].mean() / Rs[REG].mean(),
                              LR_SHM=lr_s, LR_comp=Rc[iJ] / run_mean(Rc), a1_SHM=a1s, peak_doy_SHM=pks, a1_comp=cos_fit(DOYS_ALL[REG], (Rc / Rc[REG].mean())[REG])[1]))
df3 = pd.DataFrame(rows3); df3.to_csv(f'{OUT}/inelastic_rates_vs_f.csv', index=False, float_format='%.6g')
# Higgsino counts (P068 kernels, grid B) for the same deltas
for d in [300.0, 350.0, 365.0, 380.0]:
    Rs = series(K_HIG[d], DE['B']['SHM']); Rp = series(K_HIG[d], DE['B']['hot'])
    RES.setdefault('higgsino_counts', {})[int(d)] = dict(SHM_June16=Rs[iJ], SHM_annual=Rs[REG].mean(), hot_pure_max=Rp.max())
s = df3[(df3.key == 'hot') & (df3.f == 0.0)]
log('  iso unit-coupling SHM June counts: ' + ', '.join('%.0f: %.4g' % (r.delta, r.N_SHM_June16) for r in s.itertuples()) + '; hot-disk pure max: ' + ', '.join('%.1e' % r.N_pure_annual_max for r in s.itertuples()))

# --------------------------------------------------------------------------------------------------
# Part 4: elastic channel -- low-energy companions N_lo, 200-270 keV band, modulation
# --------------------------------------------------------------------------------------------------
log('Part 4: elastic')
BANDS = dict(lo=(5.4, 55.0), hi=(200.0, 270.0), mid=(55.0, 125.0))
rows4 = []; el_ts = {'doy': DOYS_ALL.tolist()}; spec4 = {}
for tag, EK, grid in [('O1', K_O1, 'A'), ('L10', K_L10, 'B'), ('O6', K_O6, 'A')]:
    E, K = EK
    Rs = {b: series(EK, DE[grid]['SHM'], *BANDS[b]) for b in BANDS}
    r248_s = np.array([np.interp(248.0, E, K @ DE[grid]['SHM'][i]) for i in range(len(DOYS_ALL))])
    spec4[tag] = dict(E=E.tolist(), SHM=(K @ DE[grid]['SHM'][iJ]).tolist())
    for c in CFG:
        Rp = {b: series(EK, DE[grid][c['key']], *BANDS[b]) for b in BANDS}
        spec4[tag][c['key']] = (K @ DE[grid][c['key']][iJ]).tolist()
        r248_p = np.array([np.interp(248.0, E, K @ DE[grid][c['key']][i]) for i in range(len(DOYS_ALL))])
        el_ts[f'{tag}_{c["key"]}_lo_pure'] = Rp['lo'].tolist(); el_ts[f'{tag}_SHM_lo'] = Rs['lo'].tolist()
        for f in [0.0] + FRACS:
            Rc = {b: (1 - f) * Rs[b] + f * Rp[b] for b in BANDS}; r248_c = (1 - f) * r248_s + f * r248_p
            a0, a1, pk = cos_fit(DOYS_ALL[REG], (Rc['lo'] / Rc['lo'][REG].mean())[REG])
            _, a1h, pkh = cos_fit(DOYS_ALL[REG], (r248_c / r248_c[REG].mean())[REG])
            rows4.append(dict(op=tag, key=c['key'], f=f, R_lo_SHM_June=Rs['lo'][iJ], R_lo_pure_June=Rp['lo'][iJ], R_hi_SHM_June=Rs['hi'][iJ], R_hi_pure_June=Rp['hi'][iJ],
                              pure_over_SHM_lo=Rp['lo'][iJ] / Rs['lo'][iJ], pure_over_SHM_hi=Rp['hi'][iJ] / Rs['hi'][iJ] if Rs['hi'][iJ] > 0 else float('nan'),
                              pure_over_SHM_lo_annual=Rp['lo'][REG].mean() / Rs['lo'][REG].mean(),
                              N_lo_June16=Rc['lo'][iJ] / Rc['hi'][iJ], N_lo_annual=Rc['lo'][REG].mean() / Rc['hi'][REG].mean(), N_lo_SHM_annual=Rs['lo'][REG].mean() / Rs['hi'][REG].mean(),
                              N_mid_annual=Rc['mid'][REG].mean() / Rc['hi'][REG].mean(),
                              hi_band_ratio_June16=Rc['hi'][iJ] / Rs['hi'][iJ], dRdE248_ratio_June16=r248_c[iJ] / r248_s[iJ],
                              lo_a1=a1, lo_peak_doy=pk, lo_JuneDec=Rc['lo'][iJ] / periodic(DOYS_ALL, Rc['lo'], 336.0),
                              el248_a1=a1h, el248_peak_doy=pkh, el248_LR_16June=r248_c[iJ] / run_mean(r248_c)))
        _, a1p, pkp = cos_fit(DOYS_ALL[REG], (Rp['lo'] / Rp['lo'][REG].mean())[REG])
        rows4[-1].update(pure_lo_a1=a1p, pure_lo_peak_doy=pkp, pure_lo_JuneDec=Rp['lo'][iJ] / periodic(DOYS_ALL, Rp['lo'], 336.0),
                         pure_lo_max_over_min=Rp['lo'].max() / Rp['lo'].min())
df4 = pd.DataFrame(rows4); df4.to_csv(f'{OUT}/elastic_Nlo_vs_f.csv', index=False, float_format='%.5g')
json.dump(spec4, open(f'{OUT}/elastic_spectra_June16.json', 'w')); json.dump(el_ts, open(f'{OUT}/elastic_lowE_timeseries.json', 'w'))
print(df4[(df4.f.isin([0.0, 0.1, 0.3])) & (df4.key.isin(['fid', 'hot']))][['op', 'key', 'f', 'pure_over_SHM_lo', 'pure_over_SHM_hi', 'N_lo_annual', 'hi_band_ratio_June16', 'lo_a1', 'lo_peak_doy', 'el248_LR_16June']].to_string(index=False))

# --------------------------------------------------------------------------------------------------
# Part 5: exothermic (delta < 0), P058 kernels
# --------------------------------------------------------------------------------------------------
log('Part 5: exothermic')
rows5 = []; spec5 = {}
XB = dict(roi=(5.4, 270.0), b125=(125.0, 200.0), b200=(200.0, 270.0), he=(272.0, 670.0), b54_125=(5.4, 125.0))
def pct(E, y, ps):
    c = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(E)); c = np.concatenate([[0.0], c]) / c[-1]
    return [float(np.interp(p, c, E)) for p in ps]
for d in D_EXO:
    E, K = K_EXO[d]; Estar = d * MU / MN
    # 'he' (272-670 keV, P058's empty 600-1700 phd region) is evaluated with the plateau efficiency only (no 270 keV roll-off)
    def series_x(deA, b):
        if b == 'he':
            m = (E >= XB[b][0]) & (E <= XB[b][1]); return np.array([np.trapezoid(K[m] @ deA[i], E[m]) * PLATEAU * EXPOSURE for i in range(len(DOYS_ALL))])
        return series(K_EXO[d], deA, *XB[b])
    Rs = {b: series_x(DE['A']['SHM'], b) for b in XB}
    Rs['all_raw'] = np.array([np.trapezoid(K @ DE['A']['SHM'][i], E) * EXPOSURE for i in range(len(DOYS_ALL))])   # no efficiency, all energies
    ys = K @ DE['A']['SHM'][iJ]; ps = pct(E, ys * efficiency(E) * (E <= 270.0), [0.16, 0.5, 0.84]); psr = pct(E, ys, [0.16, 0.5, 0.84])
    spec5[int(d)] = dict(E=E.tolist(), SHM=ys.tolist(), E_star=Estar)
    for c in CFG:
        Rp = {b: series_x(DE['A'][c['key']], b) for b in XB}
        Rp['all_raw'] = np.array([np.trapezoid(K @ DE['A'][c['key']][i], E) * EXPOSURE for i in range(len(DOYS_ALL))])
        yp = K @ DE['A'][c['key']][iJ]; pp = pct(E, yp * efficiency(E) * (E <= 270.0), [0.16, 0.5, 0.84]); ppr = pct(E, yp, [0.16, 0.5, 0.84])
        spec5[int(d)][c['key']] = yp.tolist()
        _, a1p, pkp = cos_fit(DOYS_ALL[REG], (Rp['roi'] / Rp['roi'][REG].mean())[REG]); _, a1s, pks = cos_fit(DOYS_ALL[REG], (Rs['roi'] / Rs['roi'][REG].mean())[REG])
        for f in [0.0] + FRACS:
            Rc = {b: (1 - f) * Rs[b] + f * Rp[b] for b in list(XB) + ['all_raw']}
            _, a1c, pkc = cos_fit(DOYS_ALL[REG], (Rc['roi'] / Rc['roi'][REG].mean())[REG])
            rows5.append(dict(delta=d, E_star=Estar, key=c['key'], f=f, R_roi_SHM_annual=Rs['roi'][REG].mean(), R_roi_pure_annual=Rp['roi'][REG].mean(),
                              pure_over_SHM_all_raw=Rp['all_raw'][REG].mean() / Rs['all_raw'][REG].mean(), boost_all_raw_annual=Rc['all_raw'][REG].mean() / Rs['all_raw'][REG].mean(),
                              inv_speed_ratio_annual=float(np.mean([eta_iso(0.0, VLAB[c['key']][i], c['sig']) for i in range(len(DOYS_ALL)) if REG[i]]) / np.mean([lz.eta0(0.0, v_e=VE[i]) for i in range(len(DOYS_ALL)) if REG[i]])),
                              pure_over_SHM_roi=Rp['roi'][REG].mean() / Rs['roi'][REG].mean(), pure_over_SHM_b200=Rp['b200'][REG].mean() / Rs['b200'][REG].mean(),
                              pure_over_SHM_b125=Rp['b125'][REG].mean() / Rs['b125'][REG].mean(), pure_over_SHM_he=Rp['he'][REG].mean() / Rs['he'][REG].mean(),
                              boost_roi_annual=Rc['roi'][REG].mean() / Rs['roi'][REG].mean(), boost_b200_annual=Rc['b200'][REG].mean() / Rs['b200'][REG].mean(),
                              boost_b125_annual=Rc['b125'][REG].mean() / Rs['b125'][REG].mean(), boost_he_annual=Rc['he'][REG].mean() / Rs['he'][REG].mean(),
                              boost_roi_over_endo=(Rc['roi'][REG].mean() / Rs['roi'][REG].mean()) / (1 - f),
                              companions_125_200_per_200_270_SHM=Rs['b125'][REG].mean() / Rs['b200'][REG].mean(), companions_125_200_per_200_270_pure=Rp['b125'][REG].mean() / Rp['b200'][REG].mean(),
                              companions_125_200_per_200_270_comp=Rc['b125'][REG].mean() / Rc['b200'][REG].mean(),
                              frac_roi_SHM=Rs['roi'][REG].mean() / (Rs['roi'][REG].mean() + Rs['he'][REG].mean()), frac_roi_pure=Rp['roi'][REG].mean() / (Rp['roi'][REG].mean() + Rp['he'][REG].mean()),
                              pct16_50_84_SHM_accepted=ps, pct16_50_84_pure_accepted=pp, pct16_50_84_SHM_raw=psr, pct16_50_84_pure_raw=ppr,
                              a1_roi_SHM=a1s, peak_doy_SHM=pks, a1_roi_pure=a1p, peak_doy_pure=pkp, a1_roi_comp=a1c, peak_doy_comp=pkc,
                              pure_roi_max_over_min=Rp['roi'].max() / Rp['roi'].min()))
df5 = pd.DataFrame(rows5); df5.to_csv(f'{OUT}/exothermic_vs_f.csv', index=False, float_format='%.5g')
json.dump(spec5, open(f'{OUT}/exothermic_spectra_June16.json', 'w'))
print(df5[(df5.f.isin([0.0, 0.2])) & (df5.key.isin(['fid']))][['delta', 'E_star', 'f', 'pure_over_SHM_roi', 'pure_over_SHM_b200', 'boost_roi_annual', 'boost_roi_over_endo', 'companions_125_200_per_200_270_SHM', 'companions_125_200_per_200_270_pure', 'a1_roi_pure', 'peak_doy_pure']].to_string(index=False))

# --------------------------------------------------------------------------------------------------
# Part 6: density-fraction consequences for the Higgsino window and P021's coupling
# --------------------------------------------------------------------------------------------------
log('Part 6: Higgsino delta(N=1) vs f')
NH_ann = np.array([series(K_HIG[float(d)], DE['B']['SHM'])[REG].mean() for d in D_HIG])
NH_jun = np.array([series(K_HIG[float(d)], DE['B']['SHM'])[iJ] for d in D_HIG])
ok = NH_ann > 0
def delta_N1(N, frac):
    y = np.log((1 - frac) * N[ok]); x = D_HIG[ok]
    # log-linear interpolation; N decreasing with delta in the relevant range
    for i in range(len(x) - 1):
        if y[i] >= 0 > y[i + 1]:
            return float(x[i] + (x[i + 1] - x[i]) * (y[i] - 0.0) / (y[i] - y[i + 1]))
    return float('nan')
rows6 = []
for f in [0.0, 0.1, 0.2, 0.3, 0.5]:
    dN1 = delta_N1(NH_ann, f); dN1j = delta_N1(NH_jun, f)
    rows6.append(dict(f=f, delta_N1_annual=dN1, delta_N1_June=dN1j, coupling_factor=1 / (1 - f), events_factor=1 - f,
                      N_366_annual=(1 - f) * float(np.interp(366.0, D_HIG, NH_ann)), N_300_annual=(1 - f) * float(np.interp(300.0, D_HIG, NH_ann)), N_350_annual=(1 - f) * float(np.interp(350.0, D_HIG, NH_ann)),
                      P021_kappa_380=2.19 / (1 - f), P021_kappa_300=7.4e-5 / (1 - f), P021_sigma_n_380_cm2=6.5e-38 / (1 - f),
                      P007_exclusion_300=226 * (1 - f), P007_exclusion_350=1.9 * (1 - f)))
df6 = pd.DataFrame(rows6); df6.to_csv(f'{OUT}/higgsino_window_vs_f.csv', index=False, float_format='%.5g')
RES['higgsino_N_annual_grid'] = dict(delta=D_HIG.tolist(), N_annual=NH_ann.tolist(), N_June16=NH_jun.tolist())
dlnN = (math.log(NH_ann[D_HIG == 370.0][0]) - math.log(NH_ann[D_HIG == 360.0][0])) / 10.0
RES['higgsino_local_slope_dlnN_ddelta_360_370'] = dlnN
print(df6.to_string(index=False))

# --------------------------------------------------------------------------------------------------
# Part 7: summary table and figures
# --------------------------------------------------------------------------------------------------
def pick(df, **kw):
    s = df
    for k, v in kw.items():
        s = s[s[k] == v]
    return s.iloc[0]
S = []
def add(q, shm, f01, f03, unit='', note=''):
    S.append(dict(quantity=q, SHM=shm, SHM_plus_DD_f0p1=f01, SHM_plus_DD_f0p3=f03, unit=unit, note=note))
for d in [300.0, 366.0, 380.0]:
    r1 = pick(df3, delta=d, key='fid', f=0.1); r3 = pick(df3, delta=d, key='fid', f=0.3)
    add(f'inelastic O1 ROI events (unit coupling, 16 June), delta = {d:.0f}', r1.N_SHM_June16, r1.N_comp_June16, r3.N_comp_June16, 'events/2.84 t yr', 'exactly (1 - f): DD contributes 0')
    add(f'date LR(16 June), delta = {d:.0f}', r1.LR_SHM, r1.LR_comp, r3.LR_comp, '', 'unchanged')
for d in [300.0, 380.0]:
    r1 = pick(df3, delta=d, key='fid', f=0.1)
    add(f'inelastic a1 (first harmonic), delta = {d:.0f}', r1.a1_SHM, r1.a1_comp, pick(df3, delta=d, key='fid', f=0.3).a1_comp, '', 'unchanged')
for op in ['O1', 'L10', 'O6']:
    for key in ['fid', 'hot']:
        r1 = pick(df4, op=op, key=key, f=0.1); r3 = pick(df4, op=op, key=key, f=0.3)
        add(f'N_lo (5.4-55 per 200-270 keV, annual), {op}, {key} disk', r1.N_lo_SHM_annual, r1.N_lo_annual, r3.N_lo_annual, '', '')
r1 = pick(df4, op='O1', key='fid', f=0.1); r3 = pick(df4, op='O1', key='fid', f=0.3)
add('elastic 200-270 keV rate (fid disk, 16 June) / SHM', 1.0, r1.hi_band_ratio_June16, r3.hi_band_ratio_June16, '', '(1 - f) + tail')
add('elastic dR/dE(248) LR(16 June), O1, fid', pick(df4, op='O1', key='fid', f=0.0).el248_LR_16June, r1.el248_LR_16June, r3.el248_LR_16June, '', '')
add('low-energy (5.4-55 keV) O1 modulation a1, fid disk', pick(df4, op='O1', key='fid', f=0.0).lo_a1, r1.lo_a1, r3.lo_a1, 'fraction', 'peak day in note column of CSV')
add('low-energy O1 peak day, fid disk', pick(df4, op='O1', key='fid', f=0.0).lo_peak_doy, r1.lo_peak_doy, r3.lo_peak_doy, 'doy', '')
for d in [278.0, 300.0, 366.0]:
    r1 = pick(df5, delta=d, key='fid', f=0.1); r3 = pick(df5, delta=d, key='fid', f=0.3)
    add(f'exothermic ROI rate / SHM (annual), |delta| = {d:.0f}, fid', 1.0, r1.boost_roi_annual, r3.boost_roi_annual, '', 'f = 0.2: %.2f' % pick(df5, delta=d, key='fid', f=0.2).boost_roi_annual)
    add(f'exothermic TOTAL rate (all E, no eff.) / SHM, |delta| = {d:.0f}, fid', 1.0, r1.boost_all_raw_annual, r3.boost_all_raw_annual, '', 'f = 0.2: %.2f; pure DD/SHM %.2f' % (pick(df5, delta=d, key='fid', f=0.2).boost_all_raw_annual, r1.pure_over_SHM_all_raw))
    add(f'exothermic 272-670 keV rate / SHM, |delta| = {d:.0f}, fid', 1.0, r1.boost_he_annual, r3.boost_he_annual, '', 'f = 0.2: %.2f' % pick(df5, delta=d, key='fid', f=0.2).boost_he_annual)
    add(f'exothermic companions 125-200 per 200-270 keV, |delta| = {d:.0f}, fid', r1.companions_125_200_per_200_270_SHM, r1.companions_125_200_per_200_270_comp, r3.companions_125_200_per_200_270_comp, '', 'pure DD: %.2f' % r1.companions_125_200_per_200_270_pure)
r0 = pick(df6, f=0.0); r1 = pick(df6, f=0.1); r3 = pick(df6, f=0.3)
add('Higgsino delta(N=1), annual mean', r0.delta_N1_annual, r1.delta_N1_annual, r3.delta_N1_annual, 'keV', '')
add('P021 kappa-hat(380) = (c1 m_v^2)^2', r0.P021_kappa_380, r1.P021_kappa_380, r3.P021_kappa_380, '', 'x 1/(1 - f)')
add('P007 Higgsino excess over LZ 90% edge at 300 keV (events)', r0.P007_exclusion_300, r1.P007_exclusion_300, r3.P007_exclusion_300, '', 'x (1 - f)')
dfS = pd.DataFrame(S); dfS.to_csv(f'{OUT}/summary_table.csv', index=False, float_format='%.4g')
print(dfS.to_string(index=False))

COL = {'SHM': '#0b0b0b', 'cold': '#2a78d6', 'fid': '#eb6834', 'hot': '#1baf7a'}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25})
# Fig 1: (a) speed distributions on 16 June with thresholds; (b) v_lab(t) and low-energy rate through the year
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
fshm = np.array([lz.eta0(0.0, v_e=VE_J)]) * 0  # placeholder
w = WFINE
def shm_speed_density(w, vE):  # from -d eta/dv * v
    eta = np.asarray(lz.eta0(w, v_e=vE)); f = -np.gradient(eta, w) * w; return f / np.trapezoid(f, w)
ax[0].plot(w, shm_speed_density(w, VE_J), color=COL['SHM'], lw=2, label='SHM (16 June)')
for c in CFG:
    ax[0].plot(w, f_speed_iso(w, VLAB[c['key']][iJ], c['sig']), color=COL[c['key']], lw=1.6, label=c['name'])
for k, ls in [('v_min(55 keV, elastic)', ':'), ('v_min(248 keV, elastic)', '--'), ('v_min(248 keV, 300)', '-.')]:
    ax[0].axvline(THRESH[k], color='#52514e', lw=0.8, ls=ls); ax[0].text(THRESH[k] + 5, 0.0125, k.replace('v_min', '$v_{min}$'), rotation=90, fontsize=7, va='top')
ax[0].set_xlim(0, 850); ax[0].set_ylim(0, 0.0135); ax[0].set_xlabel('Earth-frame speed $w$ (km/s)'); ax[0].set_ylabel('$f(w)$ (km/s)$^{-1}$')
ax[0].legend(fontsize=7.5, frameon=False); ax[0].set_title('(a) Lab-frame speed distributions, 16 June 2023')
for c in CFG:
    ax[1].plot(DOYS_ALL, VLAB[c['key']], color=COL[c['key']], lw=1.6, label='%s: $v_{lab}$ of mean' % c['key'])
ax[1].plot(DOYS_ALL, VE, color=COL['SHM'], lw=2, label='SHM: $|v_{obs}|$')
ax[1].axvline(T_EVENT, color='#52514e', lw=0.8, ls=':'); ax[1].set_yscale('log'); ax[1].set_ylim(8, 400)
ax[1].set_xlabel('day of year (2023)'); ax[1].set_ylabel('lab speed (km/s)'); ax[1].legend(fontsize=7.5, frameon=False, ncol=2); ax[1].set_title('(b) Annual modulation of the lab speed')
fig.tight_layout(); fig.savefig(f'{FIG}/P097_fig1_speeds.png', dpi=170); plt.close(fig)
# Fig 2: (a) elastic O1 spectra SHM vs DD; (b) exothermic spectra at |delta| = 278 keV
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
E = np.array(spec4['O1']['E'])
ax[0].plot(E, spec4['O1']['SHM'], color=COL['SHM'], lw=2, label='SHM')
for c in CFG:
    ax[0].plot(E, spec4['O1'][c['key']], color=COL[c['key']], lw=1.6, label='pure ' + c['key'] + ' disk')
ax[0].set_yscale('log'); ax[0].set_ylim(1e-3 * max(spec4['O1']['SHM']), 30 * max(spec4['O1']['SHM'])); ax[0].set_xlim(0, 330)
ax[0].axvspan(5.4, 55, color='#2a78d6', alpha=0.08); ax[0].axvspan(200, 270, color='#eb6834', alpha=0.08)
ax[0].set_xlabel('$E_R$ (keV)'); ax[0].set_ylabel('dR/dE per unit density (events / t yr keV)'); ax[0].legend(fontsize=7.5, frameon=False); ax[0].set_title('(a) Elastic O$_1$ (1 TeV, 16 June): the disk only feeds the low-energy band')
E = np.array(spec5[278]['E']); m = E <= 700
ax[1].plot(E[m], np.array(spec5[278]['SHM'])[m], color=COL['SHM'], lw=2, label='SHM')
for c in CFG:
    ax[1].plot(E[m], np.array(spec5[278][c['key']])[m], color=COL[c['key']], lw=1.6, label='pure ' + c['key'] + ' disk')
ax[1].axvline(spec5[278]['E_star'], color='#52514e', lw=0.8, ls='--'); ax[1].axvspan(125, 200, color='#2a78d6', alpha=0.08); ax[1].axvspan(200, 270, color='#eb6834', alpha=0.08)
ax[1].set_yscale('log'); ax[1].set_ylim(1e-3 * max(spec5[278]['fid']), 3 * max(spec5[278]['fid'])); ax[1].set_xlabel('$E_R$ (keV)'); ax[1].set_ylabel('exothermic dR/dE (arb. per unit density)')
ax[1].legend(fontsize=7.5, frameon=False); ax[1].set_title('(b) Exothermic, $|\\delta|$ = 278 keV: slow disk peaks near $E^*$ = %.0f keV' % spec5[278]['E_star'])
fig.tight_layout(); fig.savefig(f'{FIG}/P097_fig2_spectra.png', dpi=170); plt.close(fig)
# Fig 3: N_lo vs f and exothermic boost vs f
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
ff = np.array([0.0] + FRACS)
for op, ls in [('O1', '-'), ('L10', '--'), ('O6', ':')]:
    for c in CFG:
        s = df4[(df4.op == op) & (df4.key == c['key'])].sort_values('f')
        ax[0].plot(s.f, s.N_lo_annual / s.N_lo_SHM_annual, ls=ls, color=COL[c['key']], lw=1.6, label=f'{op}, {c["key"]}' if op == 'O1' or c['key'] == 'fid' else None)
ax[0].set_xlabel('disk density fraction $f_{DD}$ (fixed $\\rho_0$)'); ax[0].set_ylabel('$N_{lo}$ / $N_{lo}$(SHM)'); ax[0].legend(fontsize=7.5, frameon=False, ncol=2); ax[0].set_title('(a) Elastic companion count vs disk fraction')
for d, mk in [(278.0, 'o'), (300.0, 's'), (366.0, '^')]:
    for c in CFG:
        s = df5[(df5.delta == d) & (df5.key == c['key'])].sort_values('f')
        ax[1].plot(s.f, s.boost_roi_annual, marker=mk, ms=4, color=COL[c['key']], lw=1.4, label=f'|delta| = {d:.0f}, {c["key"]}' if c['key'] == 'fid' or d == 278.0 else None)
ax[1].plot(ff, 1 - ff, color=COL['SHM'], lw=2, label='endothermic: (1 - f)')
ax[1].set_xlabel('$f_{DD}$'); ax[1].set_ylabel('rate / SHM rate'); ax[1].legend(fontsize=7.5, frameon=False, ncol=2); ax[1].set_title('(b) Exothermic ROI rate boost vs disk fraction')
fig.tight_layout(); fig.savefig(f'{FIG}/P097_fig3_Nlo_exo.png', dpi=170); plt.close(fig)

RES['runtime_s'] = time.time() - T0
json.dump(RES, open(f'{OUT}/P097_results.json', 'w'), indent=1, default=float)
log('done')
