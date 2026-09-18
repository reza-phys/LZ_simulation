"""
P068 -- A joint astrophysical uncertainty band on everything inferred from the LZ event:
rho0, v0, v_esc, v_sun, the tail shape and the halo frame.

Run from the simulation root:  .venv/bin/python output/code/P068_astro_band.py

Parts
  0  Joint prior over the halo parameters and Latin-hypercube sampling (three prior sets:
     A = main [v0-v_esc correlated, rho = +0.5 Gaussian copula], B = independent, C = Gaia-central v_esc 528-580).
  1  Earth-frame halo functions eta(v_min) for every sample in three frames (Sun frame, 16 June, 12-day
     annual mean) with P018's cap-limited Gauss-Legendre construction, reduced to one dimension because every
     model here is isotropic in the Galactic frame (the phi integral is 2 pi).  Validated against the analytic
     truncated Maxwellian (lz.eta0) and WimPyDD (lz.wd_halo).
     Tail shape: Lisanti-Strigari-Wacker-Wechsler form f ~ [exp((v_esc^2 - u^2)/(k v0^2)) - 1]^k, k in [0, 2]
     (k -> 0 is the sharp Maxwellian, k = 1 the King model); written in the overflow-safe factored form
     exp(x) [1 - exp(-x/k)]^k with x = (v_esc^2 - u^2)/v0^2, which also shows the bulk stays Maxwellian(v0).
  2  WimPyDD per-stream kernels K(E, v_i) (dR/dE = K @ delta_eta), cached to disk:
     O1 isoscalar LZ-unit coupling (WimPyDD c0 = 2/m_v^2, P003) at delta = 300/350/366/380 keV;
     P007 pure-Higgsino Z-exchange O1 on delta = 250-445 keV (2.5 keV steps);  L10 = 4[(q^2/m_N^2) O4 - O6]
     with WimPyDD c = 2 x Anand (P012), elastic.  m_chi = 1000 GeV throughout.
  3  Per-sample outputs: kappa_hat(delta) = 1/N_unit (one event), d10 (P012 LZ normalisation x rho0 scaling),
     Higgsino delta(N=1), delta_max(248 keV, 1 TeV, 16 June), modulation amplitude a1(delta) and phase from
     the 12 daily rates (P034 definition), efficiency-weighted percentile of 248 keV at delta = 300/350 keV.
  4  Marginals (median, 68/95 %), Spearman correlations, binned first-order (Sobol-like) variance indices,
     frame-conditional bands, comparison with the P043 detector systematics, the "astro-inflated" table.
  5  Figures.
Conventions: WimPyDD (vmin_i, delta_eta_i) are upper interval boundaries (P018).  rho0 enters as N ~ rho0/0.3.
"""
import sys, os, json, math, time
import numpy as np
sys.path.insert(0, 'output/code')
from common import lzcommon as lz
from scipy.special import erf, erfc
from scipy import stats
from scipy.stats import qmc
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'output/work/P068'; FIG = OUT + '/figures'; CACHE = OUT + '/kernel_cache'
for d in (OUT, FIG, CACHE):
    os.makedirs(d, exist_ok=True)
T0 = time.time()
LOGF = open(f'{OUT}/run_log.txt', 'w')
def log(*a):
    s = '[%6.1fs] ' % (time.time() - T0) + ' '.join(str(x) for x in a)
    print(s, flush=True); LOGF.write(s + '\n'); LOGF.flush()

WD = lz.wd()
V0_REF, VESC_REF = lz.V0_KMS, lz.VESC_KMS          # 238, 544 km/s (Baxter 2021)
VPEC_REF = lz.V_SUN_PEC.copy()                     # (11.1, 12.2, 7.3) km/s
RHO_REF = 0.3
VGRID = np.linspace(0.0, 950.0, 1901)              # upper boundaries, 0.5 km/s; covers v_max up to 600 + 250 + 12 + 30
DV = VGRID[1] - VGRID[0]
DOY_JUNE = 167
DAYS12 = 15.0 + 365.25 / 12 * np.arange(12)
EXPOSURE = lz.LZ['exposure_tyr']
E50_HI, E50_LO, PLATEAU = lz.LZ['E_50pct_high_keV'], lz.LZ['E_50pct_low_keV'], lz.LZ['eff_plateau']
M_CHI = 1000.0
DELTAS_ISO = [300.0, 350.0, 366.0, 380.0]
DELTAS_HIG = np.round(np.arange(250.0, 445.1, 5.0), 2)     # log-linear interpolation of N(delta) over 5 keV (N falls ~x5 per 10 keV near the edge)
FRAMES = ['sun', 'june', 'annual']
NSAMP = 400; NSAMP_VAR = 200          # main set A; variant sets B, C
HCACHE = OUT + '/halo_cache'; os.makedirs(HCACHE, exist_ok=True)
F_LZ_NORM = 12.842 / 3.3295          # P012: LZ Fig.1/Fig.6 normalisation of L10 relative to WimPyDD (sig_hi 11.5 row), provisional DR-001
D10_P012 = 0.279

# ==============================================================================================
# Part 0: joint prior and Latin-hypercube samples
# ==============================================================================================
PRIOR_DOC = dict(
    rho0='log-uniform 0.3-0.5 GeV cm^-3 (Gaia-era range; recalled, likely)',
    v0='uniform 220-250 km/s',
    vesc='uniform 500-600 km/s (set C: 528-580, Gaia central)',
    dVsun='Gaussian sigma = 5 km/s on the V (rotation) component of the solar peculiar velocity, truncated at 2.5 sigma',
    k='uniform 0-2 (Lisanti tail index; 0 = sharp Maxwellian, 1 = King)',
    frame='discrete equiprobable {Sun frame, 16 June, annual mean}',
    correlation='Gaussian copula rho(v0, v_esc) = +0.5 in sets A and C (assumption: v_esc rises with the circular speed in Galactic mass models), 0 in set B')

def sample_prior(n, seed, rho_corr=0.5, vesc_lo=500.0, vesc_hi=600.0):
    u = qmc.LatinHypercube(d=6, seed=seed).random(n)
    if rho_corr != 0.0:
        z1, z2 = stats.norm.ppf(u[:, 1]), stats.norm.ppf(u[:, 2])
        z2 = rho_corr * z1 + math.sqrt(1 - rho_corr**2) * z2
        u[:, 2] = stats.norm.cdf(z2)
    rho0 = 0.3 * (0.5 / 0.3)**u[:, 0]
    v0 = 220.0 + 30.0 * u[:, 1]
    vesc = vesc_lo + (vesc_hi - vesc_lo) * u[:, 2]
    a = stats.norm.cdf(-2.5); dV = 5.0 * stats.norm.ppf(a + (1 - 2 * a) * u[:, 3])
    k = 2.0 * u[:, 4]
    frame = np.minimum((3 * u[:, 5]).astype(int), 2)
    return pd.DataFrame(dict(rho0=rho0, v0=v0, vesc=vesc, dVsun=dV, k=k, frame_idx=frame, frame=[FRAMES[i] for i in frame]))

SETS = {'A': sample_prior(NSAMP, 68, 0.5, 500.0, 600.0),
        'B': sample_prior(NSAMP_VAR, 168, 0.0, 500.0, 600.0),
        'C': sample_prior(NSAMP_VAR, 268, 0.5, 528.0, 580.0)}
for name, df in SETS.items():
    log('prior set %s: n=%d, Spearman(v0, vesc) = %.3f, frames %s' % (name, len(df), stats.spearmanr(df.v0, df.vesc)[0], df.frame.value_counts().to_dict()))

# ==============================================================================================
# Part 1: fast isotropic Earth-frame halo functions
# ==============================================================================================
NC = 64
GLX, GLW = np.polynomial.legendre.leggauss(NC)
GLXU, GLWU = np.polynomial.legendre.leggauss(600)

def v_obs_vec(doy, v0, vpec):
    """Observer velocity in the Galactic rest frame (WimPyDD axes); doy None = Sun frame (no orbital motion)."""
    v_sun = np.array([0.0, v0, 0.0]) + vpec
    if doy is None:
        return v_sun
    lam = (doy - 80) * 2.0 * np.pi / 365.0          # WimPyDD.streamed_halo_function phase convention (P018)
    return v_sun + WD.v_earth_sun(lam, v_rot_gal=np.array([0.0, v0, 0.0]), v_sun_rot=vpec)

def f_lisanti_u2(u2, v0, vesc, k):
    """Un-normalised Galactic-frame speed distribution as a function of u^2; overflow-safe factored Lisanti form."""
    x = (vesc**2 - u2) / v0**2
    inside = x > 0
    if k < 1e-3:
        return np.where(inside, np.exp(-u2 / v0**2), 0.0)
    xs = np.where(inside, x, 0.0)
    # [exp(x/k) - 1]^k = exp(x) [1 - exp(-x/k)]^k; the constant exp(vesc^2/v0^2) cancels in the normalisation
    return np.where(inside, np.exp(-u2 / v0**2) * (1.0 - np.exp(-xs / k))**k, 0.0)

def norm_iso(v0, vesc, k):
    u = 0.5 * vesc * (GLXU + 1); wu = 0.5 * vesc * GLWU
    return 4 * math.pi * float(np.sum(wu * u**2 * f_lisanti_u2(u**2, v0, vesc, k)))

def shell_S_iso(vs, vo, v0, vesc, k):
    """S(v) = v * 2pi * Int_{cmin}^{1} f(v^2 + vo^2 - 2 v vo c) dc over the cap |v n + v_obs| <= v_esc."""
    with np.errstate(divide='ignore', invalid='ignore'):
        cmin = np.where(vs > 0, (vs**2 + vo**2 - vesc**2) / (2 * vs * vo), -1.0)
    cmin = np.clip(cmin, -1.0, 1.0)
    half = 0.5 * (1 - cmin)
    c = (0.5 * (1 + cmin))[:, None] + half[:, None] * GLX[None, :]
    wc = half[:, None] * GLW[None, :]
    u2 = vs[:, None]**2 + vo**2 - 2 * vs[:, None] * vo * c
    return vs * 2 * math.pi * (f_lisanti_u2(u2, v0, vesc, k) * wc).sum(-1)

def delta_eta_iso(vobs, v0, vesc, k, norm):
    vo = float(np.linalg.norm(vobs))
    mids = VGRID[1:] - 0.5 * DV
    Sg = shell_S_iso(VGRID, vo, v0, vesc, k) / norm
    Sm = shell_S_iso(mids, vo, v0, vesc, k) / norm
    de = np.zeros_like(VGRID)
    de[1:] = DV / 6.0 * (Sg[:-1] + 4 * Sm + Sg[1:])
    return de

def eta_from_deta(de):
    return np.cumsum(de[::-1])[::-1] - de

def halo_set(v0, vesc, k, dV):
    """delta_eta for the three frames: Sun, 16 June, and the 12 daily arrays (annual mean = their average)."""
    vpec = VPEC_REF + np.array([0.0, dV, 0.0])
    nrm = norm_iso(v0, vesc, k)
    de_sun = delta_eta_iso(v_obs_vec(None, v0, vpec), v0, vesc, k, nrm)
    de_june = delta_eta_iso(v_obs_vec(DOY_JUNE, v0, vpec), v0, vesc, k, nrm)
    de_days = np.array([delta_eta_iso(v_obs_vec(d, v0, vpec), v0, vesc, k, nrm) for d in DAYS12])
    return de_sun, de_june, de_days, vpec

# --- validation at the Baxter point ---
val = {}
de_sun0, de_june0, de_days0, _ = halo_set(V0_REF, VESC_REF, 0.0, 0.0)
vE_sun = float(np.linalg.norm(v_obs_vec(None, V0_REF, VPEC_REF))); vE_june = float(np.linalg.norm(v_obs_vec(DOY_JUNE, V0_REF, VPEC_REF)))
eta_sun0, eta_june0 = eta_from_deta(de_sun0), eta_from_deta(de_june0)
an_sun = lz.eta0(VGRID, v_e=vE_sun); an_june = lz.eta0(VGRID, v_e=vE_june)
sel = VGRID <= 780
val['eta_analytic_check'] = dict(v_E_sun=vE_sun, v_E_june=vE_june,
                                 max_rel_dev_sun_below_780=float(np.max(np.abs(eta_sun0[sel] / an_sun[sel] - 1))),
                                 max_rel_dev_june_below_780=float(np.max(np.abs(eta_june0[sel] / an_june[sel] - 1))),
                                 eta0_sun=float(eta_sun0[0]), eta0_analytic=float(an_sun[0]))
vm_wd, de_wd = lz.wd_halo(day_of_year=DOY_JUNE, vmin=VGRID)
eta_wd = eta_from_deta(de_wd)
val['eta_vs_wimpydd_june'] = [dict(vmin=v, mine=float(eta_june0[int(v / DV)]), wimpydd=float(eta_wd[int(v / DV)])) for v in (0, 400, 600, 700, 750, 780, 800)]
# Lisanti forms vs P018's k-form: eta ratio at 700 km/s, June
def eta_ratio_700(k):
    de = halo_set(V0_REF, VESC_REF, k, 0.0)[1]
    return float(eta_from_deta(de)[1400] / eta_june0[1400])
val['lisanti_eta_ratio_700_june'] = {str(k): eta_ratio_700(k) for k in (0.5, 1.0, 1.5, 2.0)}
val['P018_kform_eta_ratio_700_june'] = {'1': 0.56, '2': 0.018, '3': 5.0e-4}
log('validation: eta vs analytic max rel dev (sun/june, v<=780) = %.2e / %.2e' % (val['eta_analytic_check']['max_rel_dev_sun_below_780'], val['eta_analytic_check']['max_rel_dev_june_below_780']))
log('validation: eta/eta_SHM(700 km/s, June) Lisanti k=0.5/1/1.5/2 = %s' % val['lisanti_eta_ratio_700_june'])

# --- halo library for all sets (built after the kernels below; cached per set as float32 (n_v, 14, n) arrays) ---
HALO = {}      # set -> array (n_v, 14, n): index 0 Sun, 1 June, 2..13 the twelve days
def build_halo_library():
    for name, df in SETS.items():
        fn = f'{HCACHE}/halo_set{name}_n{len(df)}.npz'
        if os.path.exists(fn):
            HALO[name] = np.load(fn)['DE'].astype(float); log('  set %s loaded from cache' % name); continue
        DE = np.zeros((len(VGRID), 14, len(df)), dtype=np.float32)
        for j, (i, r) in enumerate(df.iterrows()):
            de_s, de_j, de_d, _ = halo_set(r.v0, r.vesc, r.k, r.dVsun)
            DE[:, 0, j] = de_s; DE[:, 1, j] = de_j; DE[:, 2:, j] = de_d.T
        np.savez_compressed(fn, DE=DE); HALO[name] = DE.astype(float)
        log('  set %s built (%d samples x 14 observer velocities)' % (name, len(df)))
def stack(name, which):
    """delta_eta matrix (n_v, n_samples) for a frame ('sun', 'june', 'annual') or a day index 0..11."""
    if which == 'sun':
        return HALO[name][:, 0, :]
    if which == 'june':
        return HALO[name][:, 1, :]
    if which == 'annual':
        return HALO[name][:, 2:, :].mean(axis=1)
    return HALO[name][:, 2 + which, :]

# ==============================================================================================
# Part 2: kernels (cached)
# ==============================================================================================
def efficiency(E, sig_hi=11.5, sig_lo=2.5):
    return PLATEAU * 0.5 * (1 + erf((E - E50_LO) / (math.sqrt(2) * sig_lo))) * 0.5 * erfc((E - E50_HI) / (math.sqrt(2) * sig_hi))

MV = lz.M_V_GEV; MN = lz.M_NUCLEON_GEV
GF, SW2 = 1.166e-5, 0.231                          # recalled, certain (P007)
c_p = (GF / math.sqrt(2)) * (1 - 4 * SW2); c_n = -GF / math.sqrt(2)
ham_iso = lz.wd_hamiltonian('P068_iso_unit', {1: (2.0 / MV**2, 0.0)})
ham_hig = lz.wd_hamiltonian('P068_higgsino_Z', {1: (c_p + c_n, c_p - c_n)})
def _make_q2(A):
    def f(q):
        return [A * q**2 / MN**2, 0.0]
    return f
def _make_const(A):
    def f():
        return [A, 0.0]
    return f
c4_wd = lz.wd_c_from_anand(4.0 / MV**2)[0]; c6_wd = lz.wd_c_from_anand(-4.0 / MV**2)[0]
ham_L10 = WD.eft_hamiltonian('P068_L10', {(4, 'q2'): _make_q2(c4_wd), 6: _make_const(c6_wd)})
ONES = np.ones_like(VGRID)

def kernel(tag, ham, m, delta, dE=2.0, E_max=330.0):
    fn = f'{CACHE}/{tag}_m{int(m)}_d{delta:.1f}.npz'
    if os.path.exists(fn):
        z = np.load(fn); return z['E'], z['K']
    if delta > 0:
        lo = min(lz.E_R_range_keV(m, VGRID[-1], A=float(A), delta_kev=delta)[0] for A in lz.XE_ISOTOPES)
        if math.isnan(lo):
            E = np.arange(1.0, E_max + 1e-9, dE); K = np.zeros((len(E), len(VGRID)))
        else:
            E = np.arange(max(1.0, math.floor(lo) - 4.0), E_max + 1e-9, dE)
            K = np.array([WD.diff_rate(WD.Xe, ham, m, float(e), VGRID, ONES, j_chi=0.5, delta=float(delta), sum_over_streams=False) for e in E]) * 1000.0 * 365.25
    else:
        E = np.arange(1.0, E_max + 1e-9, dE)
        K = np.array([WD.diff_rate(WD.Xe, ham, m, float(e), VGRID, ONES, j_chi=0.5, sum_over_streams=False) for e in E]) * 1000.0 * 365.25
    np.savez_compressed(fn, E=E, K=K)
    return E, K

log('kernels: iso x %d, Higgsino x %d, L10 x 1 (cache %s)' % (len(DELTAS_ISO), len(DELTAS_HIG), CACHE))
K_ISO = {d: kernel('iso', ham_iso, M_CHI, d) for d in DELTAS_ISO}
log('  iso kernels done')
K_HIG = {}
for j, d in enumerate(DELTAS_HIG):
    K_HIG[float(d)] = kernel('hig', ham_hig, M_CHI, float(d), dE=4.0)
    if j % 10 == 9:
        log('  Higgsino kernel %d/%d' % (j + 1, len(DELTAS_HIG)))
E_L10, K_L10 = kernel('L10', ham_L10, M_CHI, 0.0)
log('  all kernels done')
log('building halo library')
build_halo_library()

def N_from_spec(E, R):
    """in-ROI expected events (2.84 t yr) from dR/dE columns R (n_E, n_samples)."""
    w = efficiency(E)[:, None]
    return np.trapezoid(R * w, E, axis=0) * EXPOSURE
def pct248(E, R):
    w = efficiency(E)[:, None] * R
    cum = np.concatenate([np.zeros((1, R.shape[1])), np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(E)[:, None], axis=0)])
    tot = cum[-1]
    out = np.full(R.shape[1], np.nan)
    ok = tot > 0
    for j in np.where(ok)[0]:
        out[j] = 100.0 * np.interp(248.0, E, cum[:, j]) / tot[j]
    return out

# --- kernel validation at the Baxter point (reference halos in the three frames) ---
ref = {}
for fr, de in (('sun', de_sun0), ('june', de_june0), ('annual', de_days0.mean(axis=0))):
    de = de[:, None]
    ref[fr] = dict(N_iso={d: float(N_from_spec(K_ISO[d][0], K_ISO[d][1] @ de)[0]) for d in DELTAS_ISO},
                   N_hig={float(d): float(N_from_spec(K_HIG[float(d)][0], K_HIG[float(d)][1] @ de)[0]) for d in DELTAS_HIG},
                   N_L10_wd=float(N_from_spec(E_L10, K_L10 @ de)[0]),
                   pct248={d: float(pct248(K_ISO[d][0], K_ISO[d][1] @ de)[0]) for d in (300.0, 350.0)})
    ref[fr]['N_L10_lz'] = ref[fr]['N_L10_wd'] * F_LZ_NORM
    ref[fr]['d10_lz'] = 1.0 / math.sqrt(ref[fr]['N_L10_lz'])
    ref[fr]['kappa_hat'] = {d: 1.0 / ref[fr]['N_iso'][d] for d in DELTAS_ISO}
val['reference_baxter'] = ref
val['corpus_comparison'] = dict(
    P018_N_iso_1TeV=dict(june={300: 1.917e4, 350: 613.6, 380: 2.428}, annual={300: 1.335e4, 350: 253.3, 380: 0.6155}),
    P007_N_hig_1TeV_sunframe={300: 790, 350: 11.1, 370: 0.49, 380: 0.098},
    P012_N_L10_wd_sunframe_sig11p5=3.3295, P021_kappa_annual={300: 7.4e-5, 350: 3.9e-3, 380: 2.19},
    P002_pct248_june={300: 99.5, 350: 98.1}, P034_a1_annual={300: 0.428, 350: 1.179, 366: 1.521, 380: 1.656})
log('reference (Baxter) N_iso June: %s ; annual: %s' % ({d: '%.4g' % ref['june']['N_iso'][d] for d in DELTAS_ISO}, {d: '%.4g' % ref['annual']['N_iso'][d] for d in DELTAS_ISO}))
log('reference N_hig Sun-frame at 300/350/380 = %.1f / %.2f / %.3f (P007: 790/11.1/0.098)' % (ref['sun']['N_hig'][300.0], ref['sun']['N_hig'][350.0], ref['sun']['N_hig'][380.0]))
log('reference L10 N_unit_wd Sun-frame = %.4f (P012 3.3295); d10_lz = %.4f' % (ref['sun']['N_L10_wd'], ref['sun']['d10_lz']))
log('reference pct248 June: %s (P002 99.5/98.1)' % ref['june']['pct248'])

# ==============================================================================================
# Part 3: per-sample outputs
# ==============================================================================================
def delta_for_N(x, y, target=1.0):
    y = np.log10(np.maximum(y, 1e-12)); lt = math.log10(target)
    for i in range(len(x) - 1):
        if (y[i] - lt) * (y[i + 1] - lt) <= 0 and y[i] != y[i + 1] and y[i] > -11:
            return float(x[i] + (lt - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
    return float('nan')

def a1_from_days(Rdays):
    """Rdays: (12, n). Fundamental annual-modulation amplitude a1 = |2 <R/<R> e^{-i w t}>| and phase (day of year)."""
    mean = Rdays.mean(axis=0)
    w = 2 * np.pi / 365.25
    ph = np.exp(-1j * w * DAYS12)[:, None]
    with np.errstate(divide='ignore', invalid='ignore'):
        c = 2.0 * (Rdays / mean * ph).mean(axis=0)
    a1 = np.abs(c); t1 = (-np.angle(c) / w) % 365.25       # R ~ 1 + a1 cos(w(t - t1))  =>  c = a1 e^{-i w t1}
    a1[mean <= 0] = np.nan; t1[mean <= 0] = np.nan
    return a1, t1

HIG_STACK_E = [K_HIG[float(d)][0] for d in DELTAS_HIG]
HIG_STACK_K = np.concatenate([K_HIG[float(d)][1] for d in DELTAS_HIG], axis=0)
HIG_SLICES = np.cumsum([0] + [len(e) for e in HIG_STACK_E])

def outputs_for_set(name):
    df = SETS[name]; n = len(df)
    res = {}
    for fr in FRAMES:
        DE = stack(name, fr)
        o = {}
        for d in DELTAS_ISO:
            E, K = K_ISO[d]; R = K @ DE
            o[f'Niso_{int(d)}'] = N_from_spec(E, R) * df.rho0.values / RHO_REF
            if d in (300.0, 350.0):
                o[f'pct248_{int(d)}'] = pct248(E, R)
        RH = HIG_STACK_K @ DE
        Nh = np.array([N_from_spec(HIG_STACK_E[j], RH[HIG_SLICES[j]:HIG_SLICES[j + 1]]) for j in range(len(DELTAS_HIG))]) * (df.rho0.values / RHO_REF)[None, :]
        o['deltaH_N1'] = np.array([delta_for_N(DELTAS_HIG, Nh[:, i], 1.0) for i in range(n)])
        o['Nhig_300'] = Nh[list(DELTAS_HIG).index(300.0)]; o['Nhig_366'] = Nh[np.argmin(np.abs(DELTAS_HIG - 365.0))]
        o['NL10_lz'] = N_from_spec(E_L10, K_L10 @ DE) * df.rho0.values / RHO_REF * F_LZ_NORM
        res[fr] = o
    # modulation from the 12 daily halos (frame-independent by construction)
    Rd = {}
    for j in range(12):
        DEj = stack(name, j)
        for d in DELTAS_ISO:
            E, K = K_ISO[d]; Rd.setdefault(d, []).append(N_from_spec(E, K @ DEj))
        Rd.setdefault('L10', []).append(N_from_spec(E_L10, K_L10 @ DEj))
    mod = {}
    for key, arr in Rd.items():
        a1, t1 = a1_from_days(np.array(arr))
        lab = 'L10' if key == 'L10' else str(int(key))
        mod[f'a1_{lab}'] = a1; mod[f'phase_{lab}'] = t1
    # kinematics (16 June; Sun frame as variant)
    vpecV = VPEC_REF[None, :] + np.outer(df.dVsun.values, [0.0, 1.0, 0.0])
    vE_j = np.array([np.linalg.norm(v_obs_vec(DOY_JUNE, v0, vp)) for v0, vp in zip(df.v0.values, vpecV)])
    vE_s = np.array([np.linalg.norm(v_obs_vec(None, v0, vp)) for v0, vp in zip(df.v0.values, vpecV)])
    dmax_j = np.array([lz.delta_max_kev(248.0, M_CHI, v_kms=v + ve) for v, ve in zip(vE_j, df.vesc.values)])
    dmax_s = np.array([lz.delta_max_kev(248.0, M_CHI, v_kms=v + ve) for v, ve in zip(vE_s, df.vesc.values)])
    # assemble the joint table (frame drawn per sample) + frame-conditional columns
    T = df.copy()
    T['vE_june'] = vE_j; T['vE_sun'] = vE_s; T['vmax_june'] = vE_j + df.vesc.values
    T['deltamax_june'] = dmax_j; T['deltamax_sun'] = dmax_s
    for k_, v_ in mod.items():
        T[k_] = v_
    keys = list(res['sun'].keys())
    for k_ in keys:
        for fr in FRAMES:
            T[f'{k_}__{fr}'] = res[fr][k_]
        T[k_] = np.array([res[fr][k_][i] for i, fr in enumerate(df.frame.values)])
    # derived: couplings
    for d in DELTAS_ISO:
        for suf in [''] + [f'__{fr}' for fr in FRAMES]:
            T[f'log10kappa_{int(d)}{suf}'] = np.log10(1.0 / T[f'Niso_{int(d)}{suf}'].replace(0, np.nan))
    for suf in [''] + [f'__{fr}' for fr in FRAMES]:
        T[f'd10{suf}'] = 1.0 / np.sqrt(T[f'NL10_lz{suf}']); T[f'log10d10{suf}'] = np.log10(T[f'd10{suf}'])
    return T

TABLES = {}
for name in SETS:
    TABLES[name] = outputs_for_set(name)
    TABLES[name].to_csv(f'{OUT}/samples_set{name}.csv', index=False, float_format='%.6g')
    log('outputs for set %s done' % name)
T = TABLES['A']

# ==============================================================================================
# Part 4: statistics
# ==============================================================================================
OUTPUTS = ([f'log10kappa_{int(d)}' for d in DELTAS_ISO] + ['deltaH_N1', 'deltamax_june', 'log10d10']
           + [f'a1_{int(d)}' for d in DELTAS_ISO] + ['a1_L10', 'pct248_300', 'pct248_350'])
INPUTS = ['rho0', 'v0', 'vesc', 'dVsun', 'k', 'frame_idx']
FRAME_FREE = {'deltamax_june', 'a1_300', 'a1_350', 'a1_366', 'a1_380', 'a1_L10'}

def quantiles(x, nan_is_inf=False):
    """Quantiles over the finite samples, plus the undefined fraction; for couplings (nan_is_inf) also the quantiles
    with the undefined samples counted as +inf (rate zero => no coupling gives one event)."""
    x0 = np.asarray(x, float); x = x0[np.isfinite(x0)]
    if len(x) == 0:
        return dict(n=0, n_total=int(len(x0)), frac_undefined=1.0)
    q = np.percentile(x, [2.5, 16, 50, 84, 97.5])
    out = dict(n=int(len(x)), n_total=int(len(x0)), frac_undefined=float(1 - len(x) / len(x0)), p2p5=q[0], p16=q[1], median=q[2], p84=q[3], p97p5=q[4],
               half68=(q[3] - q[1]) / 2, half95=(q[4] - q[0]) / 2, mean=float(x.mean()), std=float(x.std()))
    if nan_is_inf:
        xi = np.where(np.isfinite(x0), x0, np.inf)
        qi = np.percentile(xi, [50, 84, 97.5], method='inverted_cdf')
        out.update(median_incl_inf=float(qi[0]), p84_incl_inf=float(qi[1]), p97p5_incl_inf=float(qi[2]))
    return out

def first_order_index(x, y, nb=8, discrete=False):
    """Binned first-order (Sobol-like) index Var(E[Y|X])/Var(Y); noise floor ~ (nb-1)/n."""
    ok = np.isfinite(y) & np.isfinite(x); x, y = x[ok], y[ok]
    if len(y) < 20 or y.var() == 0:
        return float('nan')
    if discrete:
        bins = x.astype(int)
    else:
        bins = np.minimum((stats.rankdata(x) - 1) * nb // len(x), nb - 1).astype(int)
    m = np.array([y[bins == b].mean() for b in np.unique(bins)]); w = np.array([(bins == b).sum() for b in np.unique(bins)])
    return float(np.sum(w * (m - y.mean())**2) / len(y) / y.var())

def stats_table(T, label, suffix=''):
    rows = []
    for o in OUTPUTS:
        col = o if (o in FRAME_FREE or suffix == '') else o + suffix
        y = T[col].values.astype(float)
        q = quantiles(y, nan_is_inf=o.startswith('log10kappa'))
        row = dict(set=label, output=o, **q)
        for inp in INPUTS:
            if inp == 'frame_idx' and (suffix != '' or o in FRAME_FREE):
                row[f'rho_{inp}'] = np.nan; row[f'S1_{inp}'] = np.nan; continue
            ok = np.isfinite(y)
            row[f'rho_{inp}'] = float(stats.spearmanr(T[inp].values[ok], y[ok])[0]) if ok.sum() > 20 else np.nan
            row[f'S1_{inp}'] = first_order_index(T[inp].values.astype(float), y, discrete=(inp == 'frame_idx'))
        rows.append(row)
    return pd.DataFrame(rows)

ST = pd.concat([stats_table(TABLES['A'], 'A_joint'), stats_table(TABLES['A'], 'A_sun', '__sun'), stats_table(TABLES['A'], 'A_june', '__june'),
                stats_table(TABLES['A'], 'A_annual', '__annual'), stats_table(TABLES['B'], 'B_joint_indep'), stats_table(TABLES['B'], 'B_annual_indep', '__annual'),
                stats_table(TABLES['C'], 'C_joint_gaia'), stats_table(TABLES['C'], 'C_annual_gaia', '__annual')], ignore_index=True)
ST.to_csv(f'{OUT}/marginals_and_sensitivities.csv', index=False, float_format='%.4g')
print(ST[ST.set == 'A_joint'][['output', 'n', 'p2p5', 'p16', 'median', 'p84', 'p97p5'] + [f'S1_{i}' for i in INPUTS]].to_string(index=False))
print(ST[ST.set == 'A_annual'][['output', 'n', 'p2p5', 'p16', 'median', 'p84', 'p97p5'] + [f'S1_{i}' for i in INPUTS]].to_string(index=False))

# dominant input per output (set A joint, and annual-only)
dom = {}
for lab in ('A_joint', 'A_annual'):
    s = ST[ST.set == lab].set_index('output')
    dom[lab] = {o: max(((i, s.loc[o, f'S1_{i}']) for i in INPUTS if np.isfinite(s.loc[o, f'S1_{i}'])), key=lambda t: t[1]) for o in OUTPUTS}

# correlation matrix of key outputs (set A joint)
KEY = ['log10kappa_366', 'deltaH_N1', 'deltamax_june', 'log10d10', 'a1_350', 'pct248_300']
CORR = TABLES['A'][KEY + INPUTS].corr(method='spearman')
CORR.to_csv(f'{OUT}/spearman_matrix_setA.csv', float_format='%.3f')

# --- detector systematics comparison (P043) ---
def pct_at(E, R, e0):
    w = efficiency(E) * R; cum = np.concatenate([[0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(E))])
    return 100.0 * np.interp(e0, E, cum) / cum[-1]
E3, K3 = K_ISO[300.0]; R3 = (K3 @ de_june0[:, None])[:, 0]
E35, K35 = K_ISO[350.0]; R35 = (K35 @ de_june0[:, None])[:, 0]
det_pct = {d: dict(minus_sys=pct_at(E, R, 248 - 9.6), plus_sys=pct_at(E, R, 248 + 9.6), minus_tot=pct_at(E, R, 248 - 13.7), plus_tot=pct_at(E, R, 248 + 13.7))
           for d, E, R in ((300, E3, R3), (350, E35, R35))}
DET = {  # 1-sigma detector half-widths in the units of the transformed outputs (P043)
    'log10kappa_300': (math.log10(1.00) + 0.004, 'kappa x/÷ 1.00 (g1), 1.01 (yield scale): <1 %'),
    'log10kappa_350': (math.log10(1.01) + 0.002, 'kappa x/÷ 1.01'),
    'log10kappa_366': (math.log10(1.07), 'interpolated between P043 370 keV (x1.03/1.06) and 350'),
    'log10kappa_380': (math.sqrt(math.log10(1.14)**2 + math.log10(1.30)**2), 'kappa x/÷ 1.14 (g1) and 1.30 (yield scale), in quadrature'),
    'deltaH_N1': (0.0, 'rate-set: unaffected by the energy scale (P043)'),
    'deltamax_june': (math.sqrt(2.1**2 + 2.1**2), 'delta_max 386.6 +- 2.1 (stat) +- 2.1 (sys) keV'),
    'log10d10': (math.log10(1.016), 'd10 +- 1.6 % (g1 0.6 %, yield scale 1.5 %)'),
    'pct248_300': ((det_pct[300]['plus_sys'] - det_pct[300]['minus_sys']) / 2, 'E_obs +- 9.6 keV (sys) through the Baxter June spectrum'),
    'pct248_350': ((det_pct[350]['plus_sys'] - det_pct[350]['minus_sys']) / 2, 'E_obs +- 9.6 keV (sys)'),
}
cmp_rows = []
for o, (hw, note) in DET.items():
    for lab in ('A_joint', 'A_annual', 'C_annual_gaia'):
        s = ST[(ST.set == lab) & (ST.output == o)].iloc[0]
        cmp_rows.append(dict(output=o, prior=lab, astro_half68=s.half68, astro_half95=s.half95, detector_1sigma=hw, ratio_astro68_over_det=(s.half68 / hw if hw > 0 else np.inf), detector_note=note))
CMP = pd.DataFrame(cmp_rows); CMP.to_csv(f'{OUT}/astro_vs_detector.csv', index=False, float_format='%.4g')
print(CMP[CMP.prior == 'A_joint'].to_string(index=False))

# --- the astro-inflated table ---
def band(lab, o, f=lambda x: x, fmt='%.3g'):
    s = ST[(ST.set == lab) & (ST.output == o)].iloc[0]
    b = dict(median=f(s['median']), lo68=f(s.p16), hi68=f(s.p84), lo95=f(s.p2p5), hi95=f(s.p97p5), n=int(s.n), frac_undefined=float(s.frac_undefined))
    if o.startswith('log10kappa'):
        b.update(median_incl_inf=f(s.median_incl_inf), hi68_incl_inf=f(s.p84_incl_inf), hi95_incl_inf=f(s.p97p5_incl_inf))
    return b
INFL = {}
for lab in ('A_joint', 'A_annual', 'C_annual_gaia', 'A_june', 'A_sun'):
    INFL[lab] = dict(d10=band(lab, 'log10d10', lambda x: 10**x), kappa_300=band(lab, 'log10kappa_300', lambda x: 10**x), kappa_350=band(lab, 'log10kappa_350', lambda x: 10**x),
                     kappa_366=band(lab, 'log10kappa_366', lambda x: 10**x), kappa_380=band(lab, 'log10kappa_380', lambda x: 10**x),
                     deltaH_N1=band(lab, 'deltaH_N1'), deltamax_june=band(lab, 'deltamax_june'),
                     a1_300=band(lab, 'a1_300'), a1_350=band(lab, 'a1_350'), a1_366=band(lab, 'a1_366'), a1_380=band(lab, 'a1_380'), a1_L10=band(lab, 'a1_L10'),
                     pct248_300=band(lab, 'pct248_300'), pct248_350=band(lab, 'pct248_350'))
# fraction of samples where the Higgsino window is undefined (never reaches N = 1 above 250 keV, or rate 0)
nan_frac = {lab: float(np.mean(~np.isfinite(TABLES['A'][('deltaH_N1' if lab == 'joint' else 'deltaH_N1__' + lab)].values))) for lab in ('joint', 'sun', 'june', 'annual')}
zero380 = {f'{int(d)}_{fr}': float(np.mean(TABLES['A'][f'Niso_{int(d)}__{fr}'].values <= 0)) for fr in FRAMES for d in DELTAS_ISO}
zero380.update({f'{int(d)}_joint': float(np.mean(TABLES['A'][f'Niso_{int(d)}'].values <= 0)) for d in DELTAS_ISO})
# frame-only spread at the Baxter point (convention effect)
frame_conv = {d: dict(sun=ref['sun']['N_iso'][d], june=ref['june']['N_iso'][d], annual=ref['annual']['N_iso'][d], june_over_annual=ref['june']['N_iso'][d] / ref['annual']['N_iso'][d] if ref['annual']['N_iso'][d] > 0 else np.inf,
                      sun_over_annual=ref['sun']['N_iso'][d] / ref['annual']['N_iso'][d] if ref['annual']['N_iso'][d] > 0 else np.inf) for d in DELTAS_ISO}
# variance share of the frame among total (joint) for each output
frame_share = {o: float(ST[(ST.set == 'A_joint') & (ST.output == o)].iloc[0]['S1_frame_idx']) for o in OUTPUTS if o not in FRAME_FREE}
# modulation phase summary
phase_summary = {d: quantiles(TABLES['A'][f'phase_{int(d)}'].values) for d in DELTAS_ISO}
phase_summary['L10'] = quantiles(TABLES['A']['phase_L10'].values)
# reference a1 at the Baxter point
Rd0 = {d: np.array([N_from_spec(K_ISO[d][0], K_ISO[d][1] @ de_days0[j][:, None])[0] for j in range(12)]) for d in DELTAS_ISO}
Rd0['L10'] = np.array([N_from_spec(E_L10, K_L10 @ de_days0[j][:, None])[0] for j in range(12)])
a1_ref = {str(k): dict(zip(('a1', 'phase'), [float(v[0]) for v in a1_from_days(np.array(Rd0[k])[:, None])])) for k in Rd0}
val['a1_reference_baxter'] = a1_ref
val['deltaH_N1_reference_baxter'] = {fr: delta_for_N(DELTAS_HIG, np.array([ref[fr]['N_hig'][float(d)] for d in DELTAS_HIG]), 1.0) for fr in FRAMES}
val['deltamax_reference'] = dict(june=lz.delta_max_kev(248.0, M_CHI, v_kms=vE_june + VESC_REF), sun=lz.delta_max_kev(248.0, M_CHI, v_kms=vE_sun + VESC_REF))
log('reference a1: %s (P034 0.43/1.18/1.52/1.66)' % {k: '%.3f' % v['a1'] for k, v in a1_ref.items()})
log('reference Higgsino delta(N=1): %s (P018 annual 365.7; P007 Sun 366)' % {k: '%.1f' % v for k, v in val['deltaH_N1_reference_baxter'].items()})

RESULTS = dict(prior=PRIOR_DOC, n_samples_per_set=NSAMP, sets={'A': 'main: correlated rho=0.5, vesc 500-600', 'B': 'independent', 'C': 'Gaia-central vesc 528-580, correlated'},
               validation=val, astro_inflated=INFL, dominant_input=dom, frame_variance_share=frame_share, frame_convention_at_baxter=frame_conv,
               higgsino_window_undefined_fraction=nan_frac, zero_rate_fraction=zero380, detector_pct_shift=det_pct, phase_summary=phase_summary,
               F_LZ_NORM=F_LZ_NORM, runtime_s=None)

# ==============================================================================================
# Part 5: figures
# ==============================================================================================
C = dict(blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', yellow='#eda100', magenta='#e87ba4', green='#008300', ink='#222222', muted='#777777')
FRCOL = {'sun': C['blue'], 'june': C['orange'], 'annual': C['aqua']}
FRLAB = {'sun': 'Sun frame', 'june': '16 June', 'annual': 'annual mean'}
INCOL = {'rho0': C['blue'], 'v0': C['orange'], 'vesc': C['aqua'], 'dVsun': C['yellow'], 'k': C['magenta'], 'frame_idx': C['green']}
INLAB = {'rho0': r'$\rho_0$', 'v0': r'$v_0$', 'vesc': r'$v_{\rm esc}$', 'dVsun': r'$\Delta V_\odot$', 'k': r'$k$ (tail)', 'frame_idx': 'frame'}
plt.rcParams.update({'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.2, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})

# Fig 1: corner-like plot of key outputs (set A, frame drawn), coloured by frame
KEYLAB = {'log10kappa_366': r'$\log_{10}\hat\kappa(366)$', 'deltaH_N1': r'$\delta_H(N{=}1)$ [keV]', 'deltamax_june': r'$\delta_{\max}$(June) [keV]',
          'log10d10': r'$\log_{10} d_{10}$', 'a1_350': r'$a_1(350)$', 'pct248_300': 'pct(248 | 300) [%]'}
n = len(KEY)
fig, axes = plt.subplots(n, n, figsize=(11, 11))
for i, yi in enumerate(KEY):
    for j, xj in enumerate(KEY):
        ax = axes[i, j]
        if j > i:
            ax.axis('off'); continue
        if i == j:
            for fr in FRAMES:
                v = T.loc[T.frame == fr, yi].values.astype(float); v = v[np.isfinite(v)]
                ax.hist(v, bins=18, histtype='step', lw=1.6, color=FRCOL[fr], label=FRLAB[fr])
            ax.set_yticks([])
            if i == 0:
                ax.legend(fontsize=7, frameon=False, loc='upper left')
        else:
            for fr in FRAMES:
                s = T[T.frame == fr]
                ax.scatter(s[xj], s[yi], s=9, alpha=0.75, color=FRCOL[fr], edgecolors='none')
        if i == n - 1:
            ax.set_xlabel(KEYLAB[xj])
        else:
            ax.set_xticklabels([])
        if j == 0 and i > 0:
            ax.set_ylabel(KEYLAB[yi])
        elif j > 0:
            ax.set_yticklabels([])
fig.suptitle('P068 Fig. 1: joint astrophysical prior propagated to six LZ-event quantities (set A, 400 halos, frame drawn per sample; 1 TeV)', fontsize=9.5)
fig.tight_layout(); fig.savefig(f'{FIG}/P068_fig1_corner.png', dpi=160); plt.close(fig)

# Fig 2: variance decomposition bars (set A joint and annual-only)
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
OUTLAB = {**KEYLAB, 'log10kappa_300': r'$\log_{10}\hat\kappa(300)$', 'log10kappa_350': r'$\log_{10}\hat\kappa(350)$', 'log10kappa_380': r'$\log_{10}\hat\kappa(380)$',
          'a1_300': r'$a_1(300)$', 'a1_366': r'$a_1(366)$', 'a1_380': r'$a_1(380)$', 'a1_L10': r'$a_1$(L10 elastic)', 'pct248_350': 'pct(248 | 350) [%]'}
for ax, lab, title in zip(axes, ('A_joint', 'A_annual'), ('(a) frame drawn from the prior', '(b) annual-mean frame only')):
    s = ST[ST.set == lab].set_index('output').loc[OUTPUTS]
    left = np.zeros(len(OUTPUTS))
    for inp in INPUTS:
        v = np.nan_to_num(s[f'S1_{inp}'].values.astype(float))
        ax.barh(np.arange(len(OUTPUTS)), v, left=left, color=INCOL[inp], label=INLAB[inp], height=0.62, edgecolor='#fcfcfb', linewidth=1.5)
        left += v
    ax.set_yticks(np.arange(len(OUTPUTS))); ax.set_yticklabels([OUTLAB[o] for o in OUTPUTS]); ax.invert_yaxis()
    ax.set_xlabel('first-order variance index  Var(E[Y|X_i]) / Var(Y)'); ax.set_xlim(0, 1.15); ax.axvline(1, color=C['muted'], lw=0.8, ls='--'); ax.set_title(title, fontsize=9)
axes[0].legend(fontsize=7.5, frameon=False, loc='lower right', ncol=2)
fig.suptitle('P068 Fig. 2: which astrophysical input controls each inferred quantity (binned Sobol-like indices; sums > 1 reflect the v0-v_esc correlation, < 1 interactions)', fontsize=9)
fig.tight_layout(); fig.savefig(f'{FIG}/P068_fig2_variance_decomposition.png', dpi=160); plt.close(fig)

# Fig 3: astro 68/95 % bands vs detector 1 sigma (P043), per quantity (set A joint), as multiples of the detector half-width
fig, ax = plt.subplots(figsize=(8.5, 4.6))
cm = CMP[CMP.prior == 'A_joint'].copy(); cm = cm[cm.detector_1sigma > 0]
y = np.arange(len(cm))
ax.barh(y - 0.18, cm.astro_half95 / cm.detector_1sigma, height=0.34, color='#9ec5f4', label='astro 95 % half-width')
ax.barh(y - 0.18, cm.astro_half68 / cm.detector_1sigma, height=0.34, color=C['blue'], label='astro 68 % half-width')
ax.barh(y + 0.18, np.ones(len(cm)), height=0.34, color=C['orange'], label='detector 1σ (P043)')
ax.set_yticks(y); ax.set_yticklabels([OUTLAB[o] for o in cm.output]); ax.invert_yaxis(); ax.set_xscale('log'); ax.set_xlim(0.1, 3e3)
ax.set_xlabel('half-width / detector 1σ half-width (P043)'); ax.axvline(1, color=C['ink'], lw=0.8)
for yy, r68 in zip(y, cm.astro_half68 / cm.detector_1sigma):
    ax.text(r68 * 1.15, yy - 0.18, '×%.1f' % r68, va='center', fontsize=7.5, color=C['ink'])
ax.legend(fontsize=7.5, frameon=False, loc='lower right')
ax.set_title('P068 Fig. 3: astrophysical vs detector systematics on the same quantities (set A; δ_H omitted: detector 0)', fontsize=9)
fig.tight_layout(); fig.savefig(f'{FIG}/P068_fig3_astro_vs_detector.png', dpi=160); plt.close(fig)

RESULTS['runtime_s'] = time.time() - T0
def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o
json.dump(_clean(RESULTS), open(f'{OUT}/P068_results.json', 'w'), indent=1)
log('done; runtime %.0f s' % RESULTS['runtime_s'])
LOGF.close()
