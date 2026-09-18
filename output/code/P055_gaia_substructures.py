"""
P055 -- Known substructures and the June event: Sagittarius, the Gaia Sausage, S1/S2 and the Helmi
streams as sources of a 248 keV recoil.

Run from the simulation root:   .venv/bin/python output/code/P055_gaia_substructures.py

Parts
  0  Frames.  Earth's orbital velocity vector in Galactic (U,V,W) coordinates from a from-scratch
     J2000 ecliptic -> equatorial -> Galactic rotation (circular orbit, lambda_sun = omega (t - t_equinox)),
     validated against wimprates.earth_velocity and WimPyDD.v_earth_sun.  v_obs(t) = v_LSR + v_pec + v_orb(t)
     with the Baxter-2021 values (238; 11.1, 12.2, 7.3).
  1  Substructure catalogue (recalled Gaia-era kinematics; every entry flagged).  Each component is a
     (possibly anisotropic) Gaussian in the Galactic frame, truncated at |u| <= v_esc = 544 km/s.  Its
     Earth-frame speed distribution f(w, t) is integrated numerically on a (w, cos theta, phi) grid, giving
     eta(v_min, t) and WimPyDD (v_min, delta_eta) arrays.  Validated against lz.eta0 (SHM) and against the
     analytic isotropic-stream eta of P030.
  2  Time dependence: v_lab(t) of each component, its first-harmonic phase and amplitude; delta_max(248 keV,
     1 TeV) on 16 June and at the component's own annual maximum (monochromatic, +2 sigma_parallel, hard edge).
  3  Rates: WimPyDD O1 per-stream kernels (P007 Higgsino Z coupling, 1 TeV; LZ efficiency; 2.84 t yr) at
     delta = 250-380 keV for the SHM and each component through the year; composite (1-f) SHM + f component;
     boosts on 16 June, annual means, June/December ratios, rate phase, and the run-window date likelihood
     ratio LR(16 June) = R(t_ev)/<R>_run (P006 method, uniform livetime, 27 Mar 2023 - 1 Apr 2024).
  4  The Gaia Sausage (SHM++ anisotropic form, beta = 0.9, f = 0.2): eta ratio vs P018, June/Dec and phase.
  5  Directional signature: mean Earth-frame arrival direction of each component (l, b), opening angle to the
     SHM wind (-v_obs, anti-Cygnus), tail-selected directions and 68 % cones for v > v_min(248 keV, delta).
"""
import sys, os, json, math, time, datetime as dt
import numpy as np
sys.path.insert(0, 'output/code')
from common import lzcommon as lz
from scipy.special import erf, erfc
from numpy.polynomial.legendre import leggauss
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'output/work/P055'; FIG = OUT + '/figures'
os.makedirs(FIG, exist_ok=True)
T0 = time.time()
def log(*a):
    print('[%6.1fs]' % (time.time() - T0), *a, flush=True)

V0, VESC = lz.V0_KMS, lz.VESC_KMS                    # 238, 544 km/s (Baxter 2021)
V_LSR = np.array([0.0, V0, 0.0]); V_PEC = lz.V_SUN_PEC.copy()
V_SUN = V_LSR + V_PEC
VSUN = float(np.linalg.norm(V_SUN))
V_ORB = 29.79                                          # km/s, mean orbital speed (recalled, certain)
VGRID = np.linspace(0.0, 1100.0, 1101); DV = 1.0       # WimPyDD upper bin boundaries
MASS = 1000.0
EXPOSURE = lz.LZ['exposure_tyr']
E50_HI, E50_LO, PLATEAU = lz.LZ['E_50pct_high_keV'], lz.LZ['E_50pct_low_keV'], lz.LZ['eff_plateau']
MN = lz.m_nucleus_gev(lz.A_XE_MEAN); MU = lz.mu_red(MASS, MN)
E_EVENT = 248.0
DELTAS = [250.0, 300.0, 330.0, 350.0, 366.0, 380.0]

# --------------------------------------------------------------------------------------------------
# Part 0: frames and the Earth's velocity vector
# --------------------------------------------------------------------------------------------------
# J2000 equatorial -> Galactic rotation matrix (Hipparcos / ESA 1997 definition; recalled, certain)
EQ2GAL = np.array([[-0.0548755604, -0.8734370902, -0.4838350155],
                   [ 0.4941094279, -0.4448296300,  0.7469822445],
                   [-0.8676661490, -0.1980763734,  0.4559837762]])
OBLIQ = math.radians(23.4393)                          # J2000 obliquity (recalled, certain)
def ecl2eq(v):
    x, y, z = v
    return np.array([x, y * math.cos(OBLIQ) - z * math.sin(OBLIQ), y * math.sin(OBLIQ) + z * math.cos(OBLIQ)])
YEAR = 2023
def doy_of(date):                                      # continuous day of year (1 Jan 00:00 = 1.0)
    return (date - dt.datetime(YEAR, 1, 1)).total_seconds() / 86400.0 + 1.0
T_EQUINOX = doy_of(dt.datetime(2023, 3, 20, 21, 24))    # March equinox 2023 (recalled, likely; +-1 h)
T_EVENT = doy_of(dt.datetime(2023, 6, 16, 21, 22, 39))  # LZ paper
OMEGA = 2 * math.pi / 365.25

def v_orb_vec(doy):
    """Earth heliocentric velocity (km/s) in Galactic (U,V,W): circular orbit, lambda_sun = omega (t - t_eq)."""
    lam = OMEGA * (doy - T_EQUINOX)
    v_ecl = V_ORB * np.array([math.sin(lam), -math.cos(lam), 0.0])   # Earth moves at longitude lambda_sun + 90 deg
    return EQ2GAL @ ecl2eq(v_ecl)

def v_obs_vec(doy):
    return V_SUN + v_orb_vec(doy)

def lb_of(vec):
    x, y, z = vec; r = np.linalg.norm(vec)
    return math.degrees(math.atan2(y, x)) % 360.0, math.degrees(math.asin(z / r))

# validation against wimprates and WimPyDD
import wimprates as wr, numericalunits as nu
WD = lz.wd()
def v_orb_wr(doy):
    t = wr.j2000(pd.Timestamp(dt.datetime(YEAR, 1, 1)) + pd.Timedelta(days=doy - 1.0))
    v = np.array(wr.earth_velocity(t, v_0=V0 * nu.km / nu.s)) / (nu.km / nu.s)
    return v - V_LSR - np.array([11.1, 12.2, 7.3])      # wimprates' default v_pec is (11.1, 12.2, 7.3)
def v_orb_wd(doy):
    lam = (doy - 80.0) * 2 * math.pi / 365.0
    return np.array(WD.v_earth_sun(lam, v_rot_gal=V_LSR, v_sun_rot=V_PEC))
val = dict(equinox_doy=T_EQUINOX, event_doy=T_EVENT, v_sun=VSUN, v_sun_lb=lb_of(V_SUN), rows=[])
for d in [1.0, 60.0, 100.0, 153.0, T_EVENT, 200.0, 250.0, 300.0, 350.0]:
    a, b, c = v_orb_vec(d), v_orb_wr(d), v_orb_wd(d)
    val['rows'].append(dict(doy=d, mine=a.tolist(), wimprates=b.tolist(), wimpydd=c.tolist(),
                            diff_wr_kms=float(np.linalg.norm(a - b)), diff_wd_kms=float(np.linalg.norm(a - c)),
                            vE_mine=float(np.linalg.norm(V_SUN + a)), vE_wr=float(np.linalg.norm(V_SUN + b)), vE_wd=float(np.linalg.norm(V_SUN + c))))
DOY_FINE = np.arange(1.0, 366.0, 0.25)
VE_FINE = np.array([np.linalg.norm(v_obs_vec(d)) for d in DOY_FINE])
val['vE_peak_doy'] = float(DOY_FINE[np.argmax(VE_FINE)]); val['vE_max'] = float(VE_FINE.max()); val['vE_min'] = float(VE_FINE.min())
val['vE_event'] = float(np.linalg.norm(v_obs_vec(T_EVENT))); val['vE_wr_event'] = float(np.linalg.norm(V_SUN + v_orb_wr(T_EVENT)))
val['max_diff_wr_kms'] = max(r['diff_wr_kms'] for r in val['rows']); val['max_diff_wd_kms'] = max(r['diff_wd_kms'] for r in val['rows'])
# Galactic-frame unit vectors e1 (equinox) and e2 (+quarter year) for comparison with the literature (0.9931, 0.1170, -0.0103), (-0.0670, 0.4927, -0.8676)
val['e1_e2'] = dict(e1=(v_orb_vec(T_EQUINOX) / V_ORB).tolist(), e2=(v_orb_vec(T_EQUINOX + 365.25 / 4) / V_ORB).tolist())
json.dump(val, open(f'{OUT}/earth_velocity_validation.json', 'w'), indent=1)
VOBS_J = v_obs_vec(T_EVENT); VE_J = float(np.linalg.norm(VOBS_J))
log('Part 0: v_E(event) = %.2f km/s (wimprates %.2f); peak doy %.1f, max %.2f; |mine - wimprates| <= %.2f km/s, |mine - WimPyDD| <= %.2f km/s'
    % (VE_J, val['vE_wr_event'], val['vE_peak_doy'], val['vE_max'], val['max_diff_wr_kms'], val['max_diff_wd_kms']))

# --------------------------------------------------------------------------------------------------
# Part 1: substructure catalogue (recalled; flagged) and numerical Earth-frame speed distributions
# --------------------------------------------------------------------------------------------------
# Galactic Cartesian axes: U toward the Galactic centre, V along rotation (l = 90), W toward the NGP.
SIG_SHM = V0 / math.sqrt(2.0)
BETA_GSE = 0.9
SIG_R_GSE = V0 * math.sqrt(3.0 / (2.0 * (3.0 - 2.0 * BETA_GSE)))     # SHM++ construction (Evans+2019): 266 km/s
SIG_T_GSE = SIG_R_GSE * math.sqrt(1.0 - BETA_GSE)                     # 84 km/s
VS_HAT = V_SUN / VSUN
CAT = [
 dict(key='SHM_num', name='SHM (numerical check)', mu=(0, 0, 0), sig=(SIG_SHM, SIG_SHM, SIG_SHM), f=0.0, f_lo=0.0, f_hi=0.0,
      flag='Baxter 2021 (certain)', ref='Baxter+2021', kind='check'),
 dict(key='Sgr_m', name='Sagittarius stream (-W)', mu=(0, 0, -300), sig=(30, 30, 30), f=0.03, f_lo=0.01, f_hi=0.05,
      flag='speed likely; vertical sense and density uncertain', ref='Freese+2004; Purcell+2012', kind='stream'),
 dict(key='Sgr_p', name='Sagittarius stream (+W)', mu=(0, 0, 300), sig=(30, 30, 30), f=0.03, f_lo=0.01, f_hi=0.05,
      flag='as above, opposite vertical sense', ref='Freese+2004; Purcell+2012', kind='stream'),
 dict(key='GSE', name='Gaia Sausage/Enceladus (SHM++ form)', mu=(0, 0, 0), sig=(SIG_R_GSE, SIG_T_GSE, SIG_T_GSE), f=0.20, f_lo=0.10, f_hi=0.30,
      flag='beta = 0.9, eta = 20 % (likely); zero mean, radially elongated', ref='Evans, O\'Hare, McCabe 2019; Necib+2019', kind='sausage'),
 dict(key='S1', name='S1 stream (retrograde)', mu=(30, -297, -73), sig=(83, 27, 59), f=0.10, f_lo=0.01, f_hi=0.10,
      flag='(v_r, v_phi, v_z) = (-30, -297, -73), sigma (83, 27, 59) likely; sign of U and f = 10 % claimed uncertain', ref='Myeong+2018; O\'Hare+2018', kind='stream'),
 dict(key='S2', name='S2 stream (prograde, vertical)', mu=(6, 164, -250), sig=(30, 20, 40), f=0.01, f_lo=0.003, f_hi=0.02,
      flag='uncertain (velocities and dispersions from memory of O\'Hare+2020)', ref='O\'Hare+2020', kind='stream'),
 dict(key='Helmi_p', name='Helmi streams (+W clump)', mu=(0, 150, 250), sig=(30, 30, 30), f=0.005, f_lo=0.002, f_hi=0.01,
      flag='v_phi ~ +150, |v_z| ~ 250 (likely); f uncertain', ref='Helmi+1999; Koppelman+2019', kind='stream'),
 dict(key='Helmi_m', name='Helmi streams (-W clump)', mu=(0, 150, -250), sig=(30, 30, 30), f=0.005, f_lo=0.002, f_hi=0.01,
      flag='as above, second clump', ref='Helmi+1999; Koppelman+2019', kind='stream'),
 dict(key='Shards', name='Retrograde shards (Rg-type)', mu=(0, -290, 0), sig=(60, 30, 60), f=0.01, f_lo=0.003, f_hi=0.02,
      flag='v_phi ~ -(250-300) uncertain; f uncertain', ref='O\'Hare+2020', kind='stream'),
 dict(key='Vesc_retro', name='Reference: retrograde stream at v_esc', mu=tuple((-VESC * VS_HAT).tolist()), sig=(20, 20, 20), f=0.01, f_lo=0.01, f_hi=0.01,
      flag='hypothetical (P030 limiting case)', ref='P030', kind='reference'),
]
pd.DataFrame([dict(key=c['key'], name=c['name'], U=c['mu'][0], V=c['mu'][1], W=c['mu'][2], v_gal=float(np.linalg.norm(c['mu'])),
                   sigU=c['sig'][0], sigV=c['sig'][1], sigW=c['sig'][2], f_nominal=c['f'], f_lo=c['f_lo'], f_hi=c['f_hi'], reliability=c['flag'], source=c['ref'])
              for c in CAT]).to_csv(f'{OUT}/substructure_catalogue.csv', index=False, float_format='%.4g')

NTH, NPH = 96, 64
_x, _w = leggauss(NTH)
_ph = (np.arange(NPH) + 0.5) * 2 * math.pi / NPH
def angular_grid(axis):
    """unit vectors and quadrature weights on the sphere with the polar axis along `axis` (GL in cos theta)."""
    e3 = axis / np.linalg.norm(axis)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(e3[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(e3, tmp); e1 /= np.linalg.norm(e1); e2 = np.cross(e3, e1)
    ct = np.repeat(_x, NPH); st = np.sqrt(1 - ct**2); ph = np.tile(_ph, NTH)
    n = ct[:, None] * e3 + st[:, None] * (np.cos(ph)[:, None] * e1 + np.sin(ph)[:, None] * e2)
    wgt = np.repeat(_w, NPH) * (2 * math.pi / NPH)
    return n, wgt

def lab_speed_density_iso(mu, sigma, v_obs, w=VGRID):
    """Exact Earth-frame speed density of an isotropic Gaussian stream (no truncation; valid when the stream is
    many sigma inside v_esc):  f(w) = w / (sqrt(2 pi) sigma v_lab) [exp(-(w-v_lab)^2/2s^2) - exp(-(w+v_lab)^2/2s^2)],
    the speed-space form of P030's analytic eta_s."""
    v_lab = float(np.linalg.norm(np.asarray(mu, float) - v_obs))
    f = w / (math.sqrt(2 * math.pi) * sigma * v_lab) * (np.exp(-(w - v_lab)**2 / (2 * sigma**2)) - np.exp(-(w + v_lab)**2 / (2 * sigma**2)))
    return f / float(np.trapezoid(f, w))

def lab_speed_density(mu, sig, v_obs, vesc=VESC, w=VGRID, return_grid=False, window=True):
    """Earth-frame speed density f(w) (normalised to 1 on w) of a Galactic-frame Gaussian N(mu, diag sig^2)
    truncated at |u| <= vesc, seen from an observer moving at v_obs.  Polar axis along the mean lab velocity.
    window=True restricts the evaluation to |w - v_lab| <= 7 max(sig) (+ the truncation range), f = 0 outside."""
    mu = np.asarray(mu, float); sig = np.asarray(sig, float)
    axis = mu - v_obs
    if np.linalg.norm(axis) < 1e-6:
        axis = -v_obs
    n, wgt = angular_grid(axis)
    f = np.zeros_like(w); G = None
    if return_grid:
        G = np.zeros((len(w), len(wgt)))
    v_lab = float(np.linalg.norm(mu - v_obs))
    lo_w, hi_w = (max(0.0, v_lab - 7 * sig.max()), min(w[-1], v_lab + 7 * sig.max())) if window else (0.0, w[-1])
    hi_w = min(hi_w, vesc + np.linalg.norm(v_obs) + 1.0)
    idx = np.where((w >= lo_w) & (w <= hi_w))[0]
    for i0 in idx[::64]:
        i1 = min(i0 + 64, idx[-1] + 1)
        ww = w[i0:i1]
        U = ww[:, None, None] * n[None, :, :] + v_obs[None, None, :]
        g = np.exp(-0.5 * np.sum(((U - mu) / sig)**2, axis=-1)) * (np.sum(U**2, axis=-1) <= vesc**2)
        f[i0:i1] = ww**2 * (g @ wgt)
        if return_grid:
            G[i0:i1] = ww[:, None]**2 * g * wgt[None, :]
    norm = float(np.trapezoid(f, w))
    f /= norm
    if return_grid:
        return f, G / norm, n
    return f

def eta_from_f(f, w=VGRID):
    """eta(w_i) = int_{w_i}^inf f(w)/w dw (trapezoid on the 1 km/s grid)."""
    g = np.where(w > 0, f / np.where(w > 0, w, 1.0), 0.0)
    seg = 0.5 * (g[1:] + g[:-1]) * np.diff(w)
    eta = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
    return eta

def deta_from_eta(eta):
    de = np.zeros_like(eta); de[1:] = eta[:-1] - eta[1:]
    return np.clip(de, 0.0, None)

def eta_of(de):
    return np.cumsum(de[::-1])[::-1] - de

def deta_shm(v_e):
    return deta_from_eta(np.asarray(lz.eta0(VGRID, v_e=v_e)))

def eta_stream_analytic(v, v_lab, sigma):          # P030 isotropic Gaussian stream (no truncation)
    v0s = math.sqrt(2.0) * sigma
    return (erf((v + v_lab) / v0s) - erf((v - v_lab) / v0s)) / (2.0 * v_lab)

# --- validation of the numerical construction (16 June) ---
log('Part 1: numerical speed distributions; validation')
f_shm_num = lab_speed_density((0, 0, 0), (SIG_SHM,) * 3, VOBS_J, window=False)
eta_num = eta_from_f(f_shm_num); eta_an = np.asarray(lz.eta0(VGRID, v_e=VE_J))
DE_SHM_J = deta_shm(VE_J)
val1 = dict(shm_eta_ratio={int(v): float(eta_num[int(v)] / eta_an[int(v)]) for v in [0, 200, 400, 600, 700, 750, 780, 800]},
            shm_vmax_numerical=float(VGRID[f_shm_num > 1e-10 * f_shm_num.max()].max()), shm_vmax_expected=VE_J + VESC,
            shm_eta0_ratio=float(eta_num[0] / eta_an[0]))
# isotropic stream vs P030 analytic (no truncation relevant: v_gal = 300, sigma = 30 -> 8 sigma below v_esc)
mu_t = np.array([0.0, 0.0, -300.0]); vlab_t = float(np.linalg.norm(mu_t - VOBS_J))
f_t = lab_speed_density(mu_t, (30, 30, 30), VOBS_J); eta_t = eta_from_f(f_t); eta_t_an = eta_stream_analytic(VGRID, vlab_t, 30.0)
f_t_iso = lab_speed_density_iso(mu_t, 30.0, VOBS_J); eta_t_iso = eta_from_f(f_t_iso)
val1['stream_eta_ratio_numerical'] = {int(v): float(eta_t[int(v)] / eta_t_an[int(v)]) for v in [0, 200, 300, 350, 400, 430, 460]}
val1['stream_eta_ratio_isoformula'] = {int(v): float(eta_t_iso[int(v)] / eta_t_an[int(v)]) for v in [0, 200, 300, 350, 400, 430, 460]}
val1['stream_vlab'] = vlab_t; val1['stream_mean_speed_numerical'] = float(np.trapezoid(f_t * VGRID, VGRID))
# anisotropic S1 with the window vs without (16 June)
c_s1 = [c for c in CAT if c['key'] == 'S1'][0]
f_a = lab_speed_density(c_s1['mu'], c_s1['sig'], VOBS_J); f_b = lab_speed_density(c_s1['mu'], c_s1['sig'], VOBS_J, window=False)
val1['S1_window_vs_full_eta_ratio'] = {int(v): float(eta_from_f(f_a)[int(v)] / eta_from_f(f_b)[int(v)]) for v in [0, 400, 600, 700, 750]}
json.dump(val1, open(f'{OUT}/speed_distribution_validation.json', 'w'), indent=1)
log('  SHM eta numerical/analytic at v_min = 0/600/750/800: %.5f/%.5f/%.5f/%.5f; stream eta ratio at 0/400/460: %.5f/%.5f/%.5f'
    % (val1['shm_eta_ratio'][0], val1['shm_eta_ratio'][600], val1['shm_eta_ratio'][750], val1['shm_eta_ratio'][800],
       val1['stream_eta_ratio_numerical'][0], val1['stream_eta_ratio_numerical'][400], val1['stream_eta_ratio_numerical'][460]))

# --------------------------------------------------------------------------------------------------
# Part 2: time dependence of v_lab, phase, amplitude, delta_max
# --------------------------------------------------------------------------------------------------
DOYS = np.concatenate([np.arange(1.0, 366.0, 12.0), [T_EVENT, 336.0]])   # 31 days + event + 2 Dec; periodic interpolation later
DOYS.sort()
def component_density(c, v_obs):
    """isotropic, untruncated components use the exact formula; anisotropic/truncated ones the numerical grid."""
    sig = np.array(c['sig'], float)
    if c['key'] == 'SHM_num':
        return lab_speed_density(c['mu'], c['sig'], v_obs, window=False)
    if np.ptp(sig) < 1e-9 and np.linalg.norm(c['mu']) + 6 * sig[0] < VESC:
        return lab_speed_density_iso(c['mu'], float(sig[0]), v_obs)
    return lab_speed_density(c['mu'], c['sig'], v_obs)
VOBS = np.array([v_obs_vec(d) for d in DOYS]); VE = np.linalg.norm(VOBS, axis=1)
DE_SHM = np.array([deta_shm(v) for v in VE])                          # [day, v]
DE_COMP = {}                                                          # key -> [day, v]
FLAB = {}
SLOPE = math.sqrt(2 * MN * E_EVENT * 1e-6) * 1e6 / lz.C_KMS          # d delta_max / d v (keV per km/s) = 0.82

def cos_fit(t, y):
    """first-harmonic fit y = a0 + A cos(omega (t - t_peak))."""
    X = np.column_stack([np.ones_like(t), np.cos(OMEGA * t), np.sin(OMEGA * t)])
    a0, a, b = np.linalg.lstsq(X, y, rcond=None)[0]
    A = math.hypot(a, b); tpk = (math.atan2(b, a) / OMEGA) % 365.25
    return float(a0), float(A), float(tpk), float(np.std(y - X @ [a0, a, b]))

rows = []; vlab_ts = {'doy': DOYS.tolist(), 'v_E_SHM': VE.tolist()}
for c in CAT:
    mu = np.array(c['mu'], float); sig = np.array(c['sig'], float)
    DE = np.zeros((len(DOYS), len(VGRID))); FL = np.zeros_like(DE)
    for i, d in enumerate(DOYS):
        if c['kind'] == 'check' and abs(d - T_EVENT) > 1e-6:
            FL[i] = FL[0] if i > 0 else f_shm_num; DE[i] = deta_from_eta(eta_from_f(FL[i])); continue   # SHM check: 16 June only
        f = component_density(c, VOBS[i]); FL[i] = f
        DE[i] = deta_from_eta(eta_from_f(f))
    DE_COMP[c['key']] = DE; FLAB[c['key']] = FL
    vlab = np.linalg.norm(mu[None, :] - VOBS, axis=1)                       # speed of the mean
    vlab_ts[c['key']] = vlab.tolist()
    iJ = int(np.argmin(np.abs(DOYS - T_EVENT)))
    a0, A, tpk, resid = cos_fit(DOYS, vlab)
    what = (mu - VOBS[iJ]); what /= np.linalg.norm(what)
    sig_par = float(math.sqrt(np.sum(sig**2 * what**2)))                    # dispersion along the lab velocity
    vmax_hard = np.array([VGRID[FL[i] > 1e-8 * FL[i].max()].max() for i in range(len(DOYS))])
    # 98th percentile of the speed distribution
    cdf = np.cumsum(FL[iJ]) * DV; v98 = float(np.interp(0.98, cdf / cdf[-1], VGRID))
    imax = int(np.argmax(vlab))
    rows.append(dict(key=c['key'], name=c['name'], v_gal=float(np.linalg.norm(mu)), sigma_parallel=sig_par,
                     v_lab_June16=float(vlab[iJ]), v_lab_mean_fit=a0, v_lab_amplitude_fit=A, v_lab_peak_doy=tpk, fit_resid_rms=resid,
                     v_lab_min=float(vlab.min()), v_lab_max=float(vlab.max()), v_lab_max_doy=float(DOYS[imax]),
                     v98_June16=v98, v_hard_max_June16=float(vmax_hard[iJ]), v_hard_max_year=float(vmax_hard.max()),
                     delta_max_mono_June16=lz.delta_max_kev(E_EVENT, MASS, v_kms=float(vlab[iJ])),
                     delta_max_2sig_June16=lz.delta_max_kev(E_EVENT, MASS, v_kms=float(vlab[iJ]) + 2 * sig_par),
                     delta_max_hard_June16=lz.delta_max_kev(E_EVENT, MASS, v_kms=float(vmax_hard[iJ])),
                     delta_max_mono_ownmax=lz.delta_max_kev(E_EVENT, MASS, v_kms=float(vlab.max())),
                     delta_max_2sig_ownmax=lz.delta_max_kev(E_EVENT, MASS, v_kms=float(vlab.max()) + 2 * sig_par),
                     delta_max_hard_year=lz.delta_max_kev(E_EVENT, MASS, v_kms=float(vmax_hard.max())),
                     E_max_elastic_June16=lz.E_R_range_keV(MASS, float(vlab[iJ]))[1], E_max_elastic_2sig=lz.E_R_range_keV(MASS, float(vlab[iJ]) + 2 * sig_par)[1]))
    log('  %-40s v_lab(16 Jun) %.1f  fit mean %.1f amp %.2f peak doy %.1f  d_max mono/2sig/hard %.0f/%.0f/%.0f keV' %
        (c['name'], vlab[iJ], a0, A, tpk, rows[-1]['delta_max_mono_June16'], rows[-1]['delta_max_2sig_June16'], rows[-1]['delta_max_hard_June16']))
a0, A, tpk, resid = cos_fit(DOYS, VE)
shm_row = dict(key='SHM', name='SHM Earth speed |v_obs|', v_lab_June16=VE_J, v_lab_mean_fit=a0, v_lab_amplitude_fit=A, v_lab_peak_doy=tpk, fit_resid_rms=resid,
               v_lab_min=float(VE.min()), v_lab_max=float(VE.max()), v_lab_max_doy=float(DOYS[np.argmax(VE)]), v_hard_max_June16=VE_J + VESC,
               delta_max_hard_June16=lz.delta_max_kev(E_EVENT, MASS, v_kms=VE_J + VESC), delta_max_hard_year=lz.delta_max_kev(E_EVENT, MASS, v_kms=float(VE.max()) + VESC))
df2 = pd.DataFrame([shm_row] + rows); df2.to_csv(f'{OUT}/modulation_phase_deltamax.csv', index=False, float_format='%.5g')
json.dump(vlab_ts, open(f'{OUT}/vlab_timeseries.json', 'w'))
print(df2[['key', 'v_lab_June16', 'v_lab_mean_fit', 'v_lab_amplitude_fit', 'v_lab_peak_doy', 'v_hard_max_June16', 'delta_max_mono_June16', 'delta_max_2sig_June16', 'delta_max_hard_June16', 'delta_max_hard_year']].to_string(index=False))

# --------------------------------------------------------------------------------------------------
# Part 3: rates through the year and the date likelihood ratio
# --------------------------------------------------------------------------------------------------
def efficiency(E, sig_hi=11.5, sig_lo=2.5):
    return PLATEAU * 0.5 * (1 + erf((E - E50_LO) / (math.sqrt(2) * sig_lo))) * 0.5 * erfc((E - E50_HI) / (math.sqrt(2) * sig_hi))
GF, SW2 = 1.166e-5, 0.231                                        # recalled, certain (P007)
c_p = (GF / math.sqrt(2)) * (1 - 4 * SW2); c_n = -GF / math.sqrt(2)
HAM = lz.wd_hamiltonian('higgsino_Z_P055', {1: (c_p + c_n, c_p - c_n)})   # sigma_n = 7.4e-39 cm^2 (P007); ratios are coupling-independent
ONES = np.ones_like(VGRID)
def kernel(delta, dE=2.0, E_max=330.0):
    if delta > 0:
        lo = min(lz.E_R_range_keV(MASS, VGRID[-1], A=float(A), delta_kev=delta)[0] for A in lz.XE_ISOTOPES)
        E = np.arange(max(1.0, math.floor(lo) - 4.0), E_max + 1e-9, dE)
    else:
        E = np.arange(1.0, E_max + 1e-9, dE)
    K = np.array([WD.diff_rate(WD.Xe, HAM, MASS, float(e), VGRID, ONES, j_chi=0.5, delta=float(delta), sum_over_streams=False) for e in E])
    return E, K * 1000.0 * 365.25
def N_roi(E, K, de):
    return float(np.trapezoid((K @ de) * efficiency(E), E)) * EXPOSURE

log('Part 3: kernels')
KERN = {d: kernel(d) for d in DELTAS}
E_el, K_el = kernel(0.0, dE=1.0)
# validation against lz.wd_halo on 16 June
DE_WD = lz.wd_halo(day_of_year=T_EVENT, vmin=VGRID)[1]
val3 = [dict(delta=d, N_mine=N_roi(*KERN[d], DE_SHM_J), N_wimpydd_halo=N_roi(*KERN[d], DE_WD)) for d in DELTAS]
for r in val3:
    r['ratio'] = r['N_mine'] / r['N_wimpydd_halo']
json.dump(val3, open(f'{OUT}/rate_validation.json', 'w'), indent=1)
log('  SHM June counts (Higgsino): ' + ', '.join('%.0f: %.4g (ratio to wd_halo %.3f)' % (r['delta'], r['N_mine'], r['ratio']) for r in val3))

# run window (P006): 27 Mar 2023 - 1 Apr 2024, uniform livetime
d_start = dt.datetime(2023, 3, 27); RUN_DAYS = 371
t_run = np.linspace(0.0, RUN_DAYS, RUN_DAYS * 8 + 1)
doy_run = (doy_of(d_start) - 1.0 + t_run) % 365.25 + 1.0
def periodic(doys, vals, x):
    dd = np.concatenate([doys - 365.25, doys, doys + 365.25]); vv = np.concatenate([vals, vals, vals])
    return np.interp(x, dd, vv)
def run_mean(vals):
    return float(np.mean(periodic(DOYS, vals, doy_run)))

iJ = int(np.argmin(np.abs(DOYS - T_EVENT))); iDec = int(np.argmin(np.abs(DOYS - 336.0)))
R_SHM = {d: np.array([N_roi(*KERN[d], DE_SHM[i]) for i in range(len(DOYS))]) for d in DELTAS}
R_PURE = {c['key']: {d: np.array([N_roi(*KERN[d], DE_COMP[c['key']][i]) for i in range(len(DOYS))]) for d in DELTAS} for c in CAT}
rows = []; ts = {'doy': DOYS.tolist()}
for d in DELTAS:
    Rs = R_SHM[d]; ts[f'SHM_d{int(d)}'] = Rs.tolist()
    lr_shm = Rs[iJ] / run_mean(Rs)
    _, a1s, pks, _ = cos_fit(DOYS, Rs / Rs.mean())
    rows.append(dict(key='SHM', name='SHM (Baxter 2021)', delta=d, f=0.0, N_June16=Rs[iJ], N_annual_mean=Rs.mean(), N_run_mean=run_mean(Rs), boost_June16=1.0, boost_per_unit_f=0.0,
                     June_Dec_ratio=Rs[iJ] / Rs[iDec] if Rs[iDec] > 1e-9 * Rs[iJ] else float('inf'), peak_doy_grid=float(DOYS[np.argmax(Rs)]), peak_doy_fit=pks, a1_fit=a1s,
                     mod_amp=(Rs.max() - Rs.min()) / (Rs.max() + Rs.min()), LR_16June=lr_shm, LR_ratio_to_SHM=1.0, frac_year_nonzero=float((Rs > 1e-6 * Rs.max()).mean())))
    for c in CAT:
        if c['kind'] == 'check':
            continue
        Rp = R_PURE[c['key']][d]; f = c['f']
        Rc = (1 - f) * Rs + f * Rp; ts[f'{c["key"]}_pure_d{int(d)}'] = Rp.tolist(); ts[f'{c["key"]}_comp_d{int(d)}'] = Rc.tolist()
        lr = Rc[iJ] / run_mean(Rc) if run_mean(Rc) > 0 else float('nan')
        _, a1c, pkc, _ = cos_fit(DOYS, Rc / Rc.mean())
        if Rp.mean() > 0:
            _, a1p, pkp, _ = cos_fit(DOYS, Rp / Rp.mean())
        else:
            a1p, pkp = float('nan'), float('nan')
        rows.append(dict(key=c['key'], name=c['name'], delta=d, f=f, N_June16=Rc[iJ], N_annual_mean=Rc.mean(), N_run_mean=run_mean(Rc),
                         N_pure_June16=Rp[iJ], N_pure_annual_mean=Rp.mean(), N_pure_max=Rp.max(), pure_peak_doy_fit=pkp, pure_a1_fit=a1p,
                         boost_June16=Rc[iJ] / Rs[iJ], boost_per_unit_f=(Rp[iJ] - Rs[iJ]) / Rs[iJ], boost_annual=Rc.mean() / Rs.mean(),
                         boost_f_lo=((1 - c['f_lo']) * Rs[iJ] + c['f_lo'] * Rp[iJ]) / Rs[iJ], boost_f_hi=((1 - c['f_hi']) * Rs[iJ] + c['f_hi'] * Rp[iJ]) / Rs[iJ],
                         June_Dec_ratio=Rc[iJ] / Rc[iDec] if Rc[iDec] > 1e-9 * Rc[iJ] else float('inf'), peak_doy_grid=float(DOYS[np.argmax(Rc)]), peak_doy_fit=pkc, a1_fit=a1c,
                         mod_amp=(Rc.max() - Rc.min()) / (Rc.max() + Rc.min()), LR_16June=lr, LR_ratio_to_SHM=lr / lr_shm, frac_year_nonzero=float((Rc > 1e-6 * Rc.max()).mean())))
df3 = pd.DataFrame(rows); df3.to_csv(f'{OUT}/rates_boosts_LR.csv', index=False, float_format='%.5g')
json.dump(ts, open(f'{OUT}/rate_timeseries.json', 'w'))
print(df3[['key', 'delta', 'f', 'N_June16', 'boost_June16', 'boost_per_unit_f', 'June_Dec_ratio', 'peak_doy_fit', 'a1_fit', 'LR_16June', 'LR_ratio_to_SHM']].to_string(index=False))

# elastic O1: dR/dE at 200/225/248/270 keV and 200-270 keV band, composite vs SHM (16 June); time dependence and date LR of dR/dE(248 keV)
el_rows = []
idx = [int(np.argmin(np.abs(E_el - e))) for e in [200.0, 225.0, 248.0, 270.0]]
band = (E_el >= 200) & (E_el <= 270)
r_shm = K_el @ DE_SHM_J
rs248 = np.array([K_el[idx[2]] @ DE_SHM[k] for k in range(len(DOYS))])
_, a1s, pks, _ = cos_fit(DOYS, rs248 / rs248.mean()); lr_s = rs248[iJ] / run_mean(rs248)
el_ts = {'doy': DOYS.tolist(), 'SHM_248': rs248.tolist()}
el_rows.append(dict(key='SHM', f=0.0, ratio_dRdE_248=1.0, el248_a1_fit=a1s, el248_peak_doy_fit=pks, el248_LR_16June=lr_s, el248_LR_ratio_to_SHM=1.0))
for c in CAT:
    if c['kind'] == 'check':
        continue
    r_p = K_el @ DE_COMP[c['key']][iJ]; f = c['f']; r_c = (1 - f) * r_shm + f * r_p
    rp248 = np.array([K_el[idx[2]] @ DE_COMP[c['key']][k] for k in range(len(DOYS))]); rc248 = (1 - f) * rs248 + f * rp248
    el_ts[f'{c["key"]}_pure_248'] = rp248.tolist()
    _, a1c, pkc, _ = cos_fit(DOYS, rc248 / rc248.mean()); lr_c = rc248[iJ] / run_mean(rc248)
    if rp248.mean() > 0:
        _, a1p, pkp, _ = cos_fit(DOYS, rp248 / rp248.mean())
    else:
        a1p, pkp = float('nan'), float('nan')
    el_rows.append(dict(key=c['key'], f=f, **{f'ratio_dRdE_{int(E_el[i])}': float(r_c[i] / r_shm[i]) for i in idx},
                        pure_over_SHM_248=float(r_p[idx[2]] / r_shm[idx[2]]), ratio_band_200_270=float(np.trapezoid(r_c[band], E_el[band]) / np.trapezoid(r_shm[band], E_el[band])),
                        pure_band_over_SHM=float(np.trapezoid(r_p[band], E_el[band]) / np.trapezoid(r_shm[band], E_el[band])),
                        el248_a1_fit=a1c, el248_peak_doy_fit=pkc, el248_LR_16June=lr_c, el248_LR_ratio_to_SHM=lr_c / lr_s, pure248_a1_fit=a1p, pure248_peak_doy_fit=pkp,
                        pure248_June_Dec=float(rp248[iJ] / rp248[iDec]) if rp248[iDec] > 0 else float('inf')))
df_el = pd.DataFrame(el_rows); df_el.to_csv(f'{OUT}/elastic_248keV_ratios.csv', index=False, float_format='%.4g')
json.dump(el_ts, open(f'{OUT}/elastic_248_timeseries.json', 'w'))
print(df_el.to_string(index=False))

# --------------------------------------------------------------------------------------------------
# Part 4: the Sausage -- eta ratio, June/Dec, phase, versus P018
# --------------------------------------------------------------------------------------------------
log('Part 4: Sausage')
eta_shm_J = eta_of(DE_SHM_J); eta_gse_J = eta_of(DE_COMP['GSE'][iJ])
sausage = dict(sigma_r=SIG_R_GSE, sigma_t=SIG_T_GSE, beta=BETA_GSE,
               eta_ratio_f0p2={int(v): float((0.8 * eta_shm_J[int(v)] + 0.2 * eta_gse_J[int(v)]) / eta_shm_J[int(v)]) for v in [300, 400, 500, 600, 650, 700, 750, 780, 800]},
               eta_gse_over_shm={int(v): float(eta_gse_J[int(v)] / eta_shm_J[int(v)]) if eta_shm_J[int(v)] > 0 else None for v in [300, 400, 500, 600, 650, 700, 750]},
               v_hard_max_gse_June16=float(VGRID[FLAB['GSE'][iJ] > 1e-8 * FLAB['GSE'][iJ].max()].max()), rows=[])
for d in DELTAS:
    Rs = R_SHM[d]; Rc = 0.8 * Rs + 0.2 * R_PURE['GSE'][d]
    a0s, As, ts_, _ = cos_fit(DOYS, Rs / Rs.mean()); a0c, Ac, tc, _ = cos_fit(DOYS, Rc / Rc.mean())
    sausage['rows'].append(dict(delta=d, N_SHM_June=Rs[iJ], N_comp_June=Rc[iJ], ratio_June=Rc[iJ] / Rs[iJ], ratio_annual=Rc.mean() / Rs.mean(),
                                June_Dec_SHM=Rs[iJ] / Rs[iDec] if Rs[iDec] > 0 else float('inf'), June_Dec_comp=Rc[iJ] / Rc[iDec] if Rc[iDec] > 0 else float('inf'),
                                a1_SHM=As, a1_comp=Ac, peak_doy_SHM=ts_, peak_doy_comp=tc, mod_amp_SHM=(Rs.max() - Rs.min()) / (Rs.max() + Rs.min()), mod_amp_comp=(Rc.max() - Rc.min()) / (Rc.max() + Rc.min()),
                                LR_SHM=Rs[iJ] / run_mean(Rs), LR_comp=Rc[iJ] / run_mean(Rc)))
# elastic modulation of the Sausage composite (low v_min): dR/dE at 10 and 50 keV vs day
for e0 in [10.0, 50.0, 248.0]:
    i = int(np.argmin(np.abs(E_el - e0)))
    rs = np.array([K_el[i] @ DE_SHM[k] for k in range(len(DOYS))]); rg = np.array([K_el[i] @ DE_COMP['GSE'][k] for k in range(len(DOYS))]); rc = 0.8 * rs + 0.2 * rg
    a0s, As, ts_, _ = cos_fit(DOYS, rs / rs.mean()); a0c, Ac, tc, _ = cos_fit(DOYS, rc / rc.mean())
    sausage[f'elastic_{int(e0)}keV'] = dict(a1_SHM=As, a1_comp=Ac, peak_doy_SHM=ts_, peak_doy_comp=tc, level_ratio_June=float(rc[iJ] / rs[iJ]))
json.dump(sausage, open(f'{OUT}/sausage.json', 'w'), indent=1)
print(pd.DataFrame(sausage['rows']).to_string(index=False))
log('  eta(comp)/eta(SHM) at 650/700/750: %.3f/%.3f/%.3f (P018: 0.80)' % tuple(sausage['eta_ratio_f0p2'][v] for v in [650, 700, 750]))

# --------------------------------------------------------------------------------------------------
# Part 5: directional signature
# --------------------------------------------------------------------------------------------------
log('Part 5: directions')
wind_J = -VOBS_J / VE_J                                          # SHM mean arrival (lab velocity) direction = anti-apex
apex_lb = lb_of(VOBS_J); wind_lb = lb_of(wind_J)
def cone68(G, n, mean_dir):
    """68 % containment half-angle of the density G[w, ang] about mean_dir."""
    wsum = G.sum(axis=0); cosang = n @ mean_dir
    order = np.argsort(-cosang); cum = np.cumsum(wsum[order]) / wsum.sum()
    return math.degrees(math.acos(float(np.clip(cosang[order][np.searchsorted(cum, 0.68)], -1, 1))))
dir_rows = []
for c in CAT:
    mu = np.array(c['mu'], float); sig = np.array(c['sig'], float)
    f, G, n = lab_speed_density(mu, sig, VOBS_J, return_grid=True, window=(c['kind'] != 'check'))
    row = dict(key=c['key'], name=c['name'])
    if np.linalg.norm(mu) > 0:
        wv = mu - VOBS_J; what = wv / np.linalg.norm(wv)
        row.update(mean_lab_dir_l=lb_of(what)[0], mean_lab_dir_b=lb_of(what)[1], opening_angle_to_SHM_wind=math.degrees(math.acos(float(np.clip(what @ wind_J, -1, 1)))),
                   angle_stream_to_solar_motion=math.degrees(math.acos(float(np.clip((mu / np.linalg.norm(mu)) @ VS_HAT, -1, 1)))))
        # annual range of the opening angle
        angs = [math.degrees(math.acos(float(np.clip(((mu - VOBS[i]) / np.linalg.norm(mu - VOBS[i])) @ (-VOBS[i] / VE[i]), -1, 1)))) for i in range(len(DOYS))]
        row.update(opening_angle_min=min(angs), opening_angle_max=max(angs))
    # all particles: density-weighted mean direction and 68 % cone
    m_all = (G.sum(axis=0) @ n); m_all /= np.linalg.norm(m_all)
    row.update(all_mean_l=lb_of(m_all)[0], all_mean_b=lb_of(m_all)[1], all_cone68=cone68(G, n, m_all), all_angle_to_wind=math.degrees(math.acos(float(np.clip(m_all @ wind_J, -1, 1)))))
    # tail-selected particles for elastic 248 keV (v > 339) and inelastic delta = 300/350/380 (v > v_min)
    for lab, vcut in [('el248', lz.vmin_kms(E_EVENT, MASS)), ('d300', lz.vmin_kms(E_EVENT, MASS, delta_kev=300.0)), ('d350', lz.vmin_kms(E_EVENT, MASS, delta_kev=350.0)), ('d380', lz.vmin_kms(E_EVENT, MASS, delta_kev=380.0))]:
        sel = VGRID >= vcut; Gs = G[sel]
        frac = float(Gs.sum() * DV)
        if frac > 1e-12:
            m = (Gs.sum(axis=0) @ n); m /= np.linalg.norm(m)
            # recoil direction: cos theta_R = vcut / w about the arrival direction -> maximal recoil angle for the 68 % fastest
            ws = VGRID[sel]; fw = f[sel]; cw = np.cumsum(fw) * DV; w68 = float(np.interp(0.68, cw / cw[-1], ws)) if cw[-1] > 0 else vcut
            row.update({f'{lab}_frac_above_vmin': frac, f'{lab}_mean_l': lb_of(m)[0], f'{lab}_mean_b': lb_of(m)[1], f'{lab}_cone68': cone68(Gs, n, m),
                        f'{lab}_angle_to_wind': math.degrees(math.acos(float(np.clip(m @ wind_J, -1, 1)))), f'{lab}_recoil_angle_68': math.degrees(math.acos(min(1.0, vcut / w68)))})
        else:
            row.update({f'{lab}_frac_above_vmin': 0.0})
    dir_rows.append(row)
df5 = pd.DataFrame(dir_rows); df5.to_csv(f'{OUT}/directional.csv', index=False, float_format='%.4g')
json.dump(dict(apex_lb=apex_lb, wind_lb=wind_lb, v_obs_June16=VOBS_J.tolist(), vmin_el248=lz.vmin_kms(E_EVENT, MASS), vmin_d300=lz.vmin_kms(E_EVENT, MASS, delta_kev=300.0),
               vmin_d350=lz.vmin_kms(E_EVENT, MASS, delta_kev=350.0), vmin_d380=lz.vmin_kms(E_EVENT, MASS, delta_kev=380.0), v_max_June16=VE_J + VESC,
               max_recoil_angle_d380=math.degrees(math.acos(lz.vmin_kms(E_EVENT, MASS, delta_kev=380.0) / (VE_J + VESC))),
               max_recoil_angle_d350=math.degrees(math.acos(lz.vmin_kms(E_EVENT, MASS, delta_kev=350.0) / (VE_J + VESC))),
               max_recoil_angle_d300=math.degrees(math.acos(lz.vmin_kms(E_EVENT, MASS, delta_kev=300.0) / (VE_J + VESC))),
               arrival_cone_d380_analytic=180.0 - math.degrees(math.acos((VESC**2 - lz.vmin_kms(E_EVENT, MASS, delta_kev=380.0)**2 - VE_J**2) / (2 * lz.vmin_kms(E_EVENT, MASS, delta_kev=380.0) * VE_J)))),
          open(f'{OUT}/directional_summary.json', 'w'), indent=1)
print(df5[['key', 'mean_lab_dir_l', 'mean_lab_dir_b', 'opening_angle_to_SHM_wind', 'all_cone68', 'el248_frac_above_vmin', 'el248_angle_to_wind', 'el248_cone68', 'd300_frac_above_vmin', 'd380_frac_above_vmin', 'd380_cone68', 'd380_recoil_angle_68']].to_string(index=False))

# --------------------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------------------
COL = {'SHM': '#0b0b0b', 'Sgr_m': '#2a78d6', 'Sgr_p': '#86b6ef', 'GSE': '#eb6834', 'S1': '#1baf7a', 'S2': '#eda100', 'Helmi_p': '#e87ba4', 'Helmi_m': '#e87ba4', 'Shards': '#4a3aa7', 'Vesc_retro': '#e34948'}
LBL = {c['key']: c['name'] for c in CAT}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25, 'grid.color': '#c3c2b7'})
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
ax[0].plot(DOYS, VE, color=COL['SHM'], lw=2.0, label='SHM: $|v_{obs}|$ (Earth speed)')
for k in ['Sgr_m', 'Sgr_p', 'S1', 'Shards', 'S2', 'Helmi_p', 'Vesc_retro']:
    ls = '--' if k in ('Sgr_p',) else '-'
    ax[0].plot(DOYS, vlab_ts[k], color=COL[k], lw=1.6, ls=ls, label=LBL[k])
ax[0].axvline(T_EVENT, color='#52514e', lw=0.8, ls=':'); ax[0].text(T_EVENT + 3, 640, '16 June', fontsize=7.5, color='#52514e')
ax[0].set_xlabel('day of year (2023)'); ax[0].set_ylabel('$v_{lab}$ of the component mean (km/s)'); ax[0].set_ylim(200, 850)
ax[0].legend(fontsize=6.8, frameon=False, loc='upper right', ncol=2); ax[0].set_title('(a) Lab-frame speed of each substructure through the year')
# (b) normalised: (v_lab - mean) for phase comparison
for k in ['Sgr_m', 'Sgr_p', 'S1', 'Shards', 'Vesc_retro']:
    y = np.array(vlab_ts[k]); ax[1].plot(DOYS, y - y.mean(), color=COL[k], lw=1.6, ls='--' if k == 'Sgr_p' else '-', label=LBL[k])
ax[1].plot(DOYS, VE - VE.mean(), color=COL['SHM'], lw=2.0, label='SHM')
ax[1].axvline(T_EVENT, color='#52514e', lw=0.8, ls=':'); ax[1].axvline(val['vE_peak_doy'], color='#52514e', lw=0.8, ls='-.')
ax[1].text(val['vE_peak_doy'] - 3, -27, 'SHM peak\n%s' % (dt.datetime(YEAR, 1, 1) + dt.timedelta(days=val['vE_peak_doy'] - 1)).strftime('%d %b'), fontsize=7, color='#52514e', ha='right')
ax[1].set_xlabel('day of year (2023)'); ax[1].set_ylabel('$v_{lab} - \\langle v_{lab}\\rangle$ (km/s)'); ax[1].set_ylim(-32, 32)
ax[1].set_title('(b) Annual modulation: phase and amplitude'); ax[1].legend(fontsize=6.8, frameon=False, loc='upper right', ncol=2)
fig.tight_layout(); fig.savefig(f'{FIG}/P055_fig1_vlab_year.png', dpi=170); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
for k in ['Sgr_m', 'GSE', 'S1', 'Shards', 'S2', 'Vesc_retro']:
    s = df3[df3.key == k].sort_values('delta')
    ax[0].plot(s.delta, s.boost_June16, 'o-', color=COL[k], lw=1.6, ms=4, label='%s ($f$ = %g)' % (LBL[k], s.f.iloc[0]))
ax[0].axhline(1, color=COL['SHM'], lw=1.0); ax[0].set_yscale('log'); ax[0].set_ylim(0.5, 200)
ax[0].set_xlabel('$\\delta$ (keV)   (O$_1$, 1 TeV, 16 June 2023)'); ax[0].set_ylabel('in-ROI rate / SHM rate')
ax[0].legend(fontsize=6.8, frameon=False, loc='upper left'); ax[0].set_title('(a) Inelastic rate boost of SHM + substructure')
for k in ['SHM', 'Sgr_m', 'GSE', 'S1', 'Shards', 'Vesc_retro']:
    s = df3[df3.key == k].sort_values('delta')
    ax[1].plot(s.delta, s.LR_16June, 's-', color=COL[k], lw=1.6 if k != 'SHM' else 2.2, ms=4, label=LBL[k] if k != 'SHM' else 'SHM only')
ax[1].axhline(1, color='#52514e', lw=0.8, ls=':'); ax[1].set_xlabel('$\\delta$ (keV)'); ax[1].set_ylabel('LR(16 June) = $R(t_{ev}) / \\langle R \\rangle_{run}$'); ax[1].set_ylim(0.8, 4)
ax[1].legend(fontsize=6.8, frameon=False, loc='upper left'); ax[1].set_title('(b) Date likelihood ratio (uniform livetime, 371-d run)')
fig.tight_layout(); fig.savefig(f'{FIG}/P055_fig2_boost_LR.png', dpi=170); plt.close(fig)

fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.2))
for d, col in zip([300.0, 350.0, 380.0], ['#2a78d6', '#eb6834', '#1baf7a']):
    Rs = R_SHM[d]; Rc = 0.8 * Rs + 0.2 * R_PURE['GSE'][d]; Rv = 0.99 * Rs + 0.01 * R_PURE['Vesc_retro'][d]
    ax.plot(DOYS, Rs / Rs.mean(), color=col, lw=2.0, label='SHM, $\\delta$ = %.0f keV' % d)
    ax.plot(DOYS, Rc / Rc.mean(), color=col, lw=1.2, ls='--')
    ax.plot(DOYS, Rv / Rv.mean(), color=col, lw=1.2, ls=':')
ax.axvline(T_EVENT, color='#52514e', lw=0.8, ls=':'); ax.set_xlabel('day of year'); ax.set_ylabel('rate / annual mean'); ax.set_ylim(0, 4.2)
ax.legend(fontsize=7.5, frameon=False, loc='upper right', title='solid SHM; dashed +20% Sausage; dotted +1% $v_{esc}$ stream', title_fontsize=7)
ax.set_title('Time dependence of the in-ROI inelastic rate'); fig.tight_layout(); fig.savefig(f'{FIG}/P055_fig3_rate_year.png', dpi=170); plt.close(fig)

json.dump(dict(runtime_s=time.time() - T0, n_days=len(DOYS), deltas=DELTAS, vgrid=[0, 1100, len(VGRID)], angular_grid=[NTH, NPH], mass=MASS, exposure=EXPOSURE),
          open(f'{OUT}/run_meta.json', 'w'), indent=1)
log('done')
