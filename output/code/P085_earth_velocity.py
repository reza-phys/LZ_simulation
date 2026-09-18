#!/usr/bin/env python
"""P085 -- Earth's velocity on 16 June 2023 21:22:39 UTC and gravitational focusing:
the exact kinematic inputs for the LZ event.

Run from the simulation root:  .venv/bin/python output/code/P085_earth_velocity.py
Outputs: output/work/P085/*.csv|json, output/work/P085/figures/*.png

Parts
  A  Earth heliocentric velocity/position at the event time (astropy 8.0.1, built-in ephemeris, offline),
     Earth rotation at SURF, solar motion variants -> v_obs (Galactic Cartesian), |v_E|, apex/wind direction.
  B  Comparison with wimprates, WimPyDD, lzcommon cosine model, P055/P006, the 2023 velocity maximum,
     the annual mean, and the WimPyDD Sun-frame value 250.6 km/s.
  C  Kinematics: delta_max(248 keV, m), E_max, sensitivity, uncertainty budget.
  D  Gravitational focusing by the Sun (Liouville mapping, validated by backward orbit integration):
     density enhancement, eta(v_min) ratios, hard-edge shift, through the year.
  E  Earth's own focusing and shadowing.
  F  WimPyDD per-stream kernels (1 TeV Higgsino Z coupling of P007): rates at delta = 300/350/366/380 keV
     and the Higgsino delta(N=1) for Sun-frame / June-16 / annual / focused halos.
  G  Fraction of the year (and of the LZ run window) with delta_max(248 keV) >= 380/385/387 keV.
"""
import sys, os, json, math, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, 'output/code')
from common import lzcommon as lz

from astropy.utils import iers
iers.conf.auto_download = False           # never touch the network
iers.conf.auto_max_age = 1e9
iers.conf.iers_degraded_accuracy = 'ignore'
import astropy
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import (get_body_barycentric_posvel, solar_system_ephemeris, EarthLocation, SkyCoord,
                                 CartesianRepresentation, ICRS, Galactic, AltAz, get_body, GeocentricTrueEcliptic)
from scipy.special import erf, erfc
from scipy.integrate import solve_ivp

T0 = time.time()
OUT = 'output/work/P085'
FIG = OUT + '/figures'
os.makedirs(FIG, exist_ok=True)
LOG = {}
def log(k, v):
    LOG[k] = v
    print(f'{k}: {v}')

# ---------------------------------------------------------------------------------------------
# constants (recalled, flagged in details.md)
# ---------------------------------------------------------------------------------------------
GM_SUN_KM3S2 = 1.32712440018e11     # km^3/s^2 (certain)
GM_EARTH_KM3S2 = 3.986004418e5      # km^3/s^2 (certain)
AU_KM = 1.495978707e8               # km (certain)
R_EARTH_KM = 6371.0                 # mean radius (certain)
SURF_LAT, SURF_LON = 44.352, -103.751   # deg (Lead, SD; likely)
SURF_HEIGHT_M = 100.0               # 4850 ft below a ~1600 m surface -> ~ +100 m a.s.l. (uncertain; irrelevant)
V0, VESC = lz.V0_KMS, lz.VESC_KMS   # 238, 544 (Baxter 2021 via lzcommon)
VPEC_BAXTER = np.array([11.1, 12.2, 7.3])
VPEC_SCHOENRICH = np.array([11.1, 12.24, 7.25])
SIG_V0 = 1.5                        # km/s (Baxter 2021 recommended uncertainty; recalled, uncertain)
SIG_PEC = np.array([1.2, 2.0, 0.6]) # km/s (Schoenrich 2010 stat+sys; recalled, uncertain)
C = lz.C_KMS
# J2000 equatorial -> Galactic rotation (Hipparcos; recalled, certain) for a cross-check of astropy
R_EQ2GAL = np.array([[-0.0548755604, -0.8734370902, -0.4838350155],
                     [0.4941094279, -0.4448296300, 0.7469822445],
                     [-0.8676661490, -0.1980763734, 0.4559837762]])

log('astropy_version', astropy.__version__)
log('ephemeris', solar_system_ephemeris.get())

# ---------------------------------------------------------------------------------------------
# A. Earth's heliocentric state and rotation at the event time
# ---------------------------------------------------------------------------------------------
t_ev = Time('2023-06-16T21:22:39', scale='utc')
log('t_event_utc', t_ev.isot)
log('t_event_jd_utc', t_ev.jd)
log('t_event_jd_tdb', t_ev.tdb.jd)
doy_ev = (t_ev - Time('2023-01-01T00:00:00', scale='utc')).to_value('day') + 1.0   # 1 Jan 00:00 = day 1.0
log('day_of_year_event', doy_ev)          # 167.89 (Jan 1 = 1)
log('UT1_minus_UTC_s', (t_ev.ut1.jd - t_ev.jd) * 86400.0)

def icrs_to_gal(vec):
    """Rotate ICRS Cartesian vectors (..., 3) to Galactic Cartesian with astropy's frame machinery."""
    vec = np.asarray(vec, dtype=float)
    sc = SkyCoord(CartesianRepresentation(vec[..., 0], vec[..., 1], vec[..., 2], unit=u.one), frame='icrs')
    g = sc.galactic.cartesian.xyz.value
    return np.moveaxis(g, 0, -1)

def helio_state_gal(t):
    """Earth position (km) and velocity (km/s) relative to the Sun, Galactic Cartesian (U,V,W)."""
    pe, ve = get_body_barycentric_posvel('earth', t)
    ps, vs = get_body_barycentric_posvel('sun', t)
    r = np.moveaxis((pe - ps).xyz.to_value(u.km), 0, -1)
    v = np.moveaxis((ve - vs).xyz.to_value(u.km / u.s), 0, -1)
    return icrs_to_gal(r), icrs_to_gal(v), v   # also return ICRS velocity

r_ev, vorb_ev, vorb_icrs = helio_state_gal(t_ev)
log('earth_helio_velocity_icrs_kms', vorb_icrs.tolist())
log('earth_helio_velocity_gal_kms', vorb_ev.tolist())
log('earth_helio_speed_kms', float(np.linalg.norm(vorb_ev)))
log('earth_helio_distance_AU', float(np.linalg.norm(r_ev) / AU_KM))
log('earth_helio_velocity_gal_recalled_matrix_kms', (R_EQ2GAL @ vorb_icrs).tolist())
log('astropy_vs_recalled_matrix_max_diff_kms', float(np.max(np.abs(R_EQ2GAL @ vorb_icrs - vorb_ev))))
# Earth-Moon barycentre vs Earth (size of the lunar term)
pb, vb = get_body_barycentric_posvel('earth-moon-barycenter', t_ev)
pe, ve = get_body_barycentric_posvel('earth', t_ev)
log('earth_minus_EMB_velocity_kms', float(np.linalg.norm((ve - vb).xyz.to_value(u.km / u.s))))
# Sun's apparent ecliptic longitude (for the WimPyDD comparison)
sun_ecl = get_body('sun', t_ev).transform_to(GeocentricTrueEcliptic(equinox=t_ev))
lam_sun = float(sun_ecl.lon.deg)
log('sun_apparent_ecliptic_longitude_deg', lam_sun)

# Earth rotation at SURF
surf = EarthLocation(lat=SURF_LAT * u.deg, lon=SURF_LON * u.deg, height=SURF_HEIGHT_M * u.m)
gp, gv = surf.get_gcrs_posvel(t_ev)
vrot_icrs = gv.xyz.to_value(u.km / u.s)
vrot_ev = icrs_to_gal(vrot_icrs)
zenith_ev = icrs_to_gal(gp.xyz.to_value(u.km) / np.linalg.norm(gp.xyz.to_value(u.km)))
log('rotation_velocity_gal_kms', vrot_ev.tolist())
log('rotation_speed_kms', float(np.linalg.norm(vrot_ev)))
lst = t_ev.sidereal_time('apparent', longitude=surf.lon)
log('local_apparent_sidereal_time_h', float(lst.hour))
log('zenith_direction_gal', zenith_ev.tolist())

def lb(vec):
    vec = np.asarray(vec, float)
    l = math.degrees(math.atan2(vec[1], vec[0])) % 360.0
    b = math.degrees(math.asin(vec[2] / np.linalg.norm(vec)))
    return l, b

# ---------------------------------------------------------------------------------------------
# solar-motion variants and the total observer velocity
# ---------------------------------------------------------------------------------------------
def v_sun_vec(v0=V0, pec=VPEC_BAXTER):
    return np.array([0.0, v0, 0.0]) + np.asarray(pec, float)

variants = {
    'Baxter (v0=238, pec 11.1/12.2/7.3)': (238.0, VPEC_BAXTER),
    'Schoenrich pec (11.1/12.24/7.25), v0=238': (238.0, VPEC_SCHOENRICH),
    'v0=220': (220.0, VPEC_BAXTER), 'v0=230': (230.0, VPEC_BAXTER),
    'v0=233 (SHM++)': (233.0, VPEC_BAXTER), 'v0=250': (250.0, VPEC_BAXTER),
}
rows_var = []
for name, (v0, pec) in variants.items():
    vs = v_sun_vec(v0, pec)
    vE_norot = vs + vorb_ev
    vE = vE_norot + vrot_ev
    l_ap, b_ap = lb(vE)
    rows_var.append(dict(variant=name, v0=v0, pecU=pec[0], pecV=pec[1], pecW=pec[2],
                         vsun_speed=float(np.linalg.norm(vs)),
                         vE_U=vE[0], vE_V=vE[1], vE_W=vE[2],
                         vE_speed=float(np.linalg.norm(vE)), vE_speed_norot=float(np.linalg.norm(vE_norot)),
                         apex_l=l_ap, apex_b=b_ap, wind_l=(l_ap + 180) % 360, wind_b=-b_ap,
                         vmax_544=float(np.linalg.norm(vE)) + 544.0))
import pandas as pd
df_var = pd.DataFrame(rows_var)
df_var.to_csv(OUT + '/observer_velocity_variants.csv', index=False)
print(df_var.to_string())

VSUN = v_sun_vec()                       # Baxter reference
VE_EV = VSUN + vorb_ev + vrot_ev         # reference observer velocity, event time, incl. rotation
VE_EV_NOROT = VSUN + vorb_ev
VE_SPEED = float(np.linalg.norm(VE_EV))
log('v_sun_speed_kms', float(np.linalg.norm(VSUN)))
log('v_E_event_vector_gal_kms', VE_EV.tolist())
log('v_E_event_speed_kms', VE_SPEED)
log('v_E_event_speed_norot_kms', float(np.linalg.norm(VE_EV_NOROT)))
log('rotation_projection_on_vE_kms', float(np.dot(vrot_ev, VE_EV_NOROT) / np.linalg.norm(VE_EV_NOROT)))
log('orbital_projection_on_vsun_kms', float(np.dot(vorb_ev, VSUN) / np.linalg.norm(VSUN)))
log('apex_lb_event_deg', list(lb(VE_EV)))
log('wind_lb_event_deg', list(lb(-VE_EV)))
log('sun_apex_lb_deg', list(lb(VSUN)))
# apex in equatorial coordinates and horizon coordinates at SURF
apex_icrs = R_EQ2GAL.T @ VE_EV
apex_sc = SkyCoord(CartesianRepresentation(*apex_icrs, unit=u.one), frame='icrs')
log('apex_radec_deg', [float(apex_sc.spherical.lon.deg), float(apex_sc.spherical.lat.deg)])
aa = apex_sc.transform_to(AltAz(obstime=t_ev, location=surf))
log('apex_altaz_at_SURF_deg', [float(aa.alt.deg), float(aa.az.deg)])
log('angle_rotation_vs_vE_deg', float(math.degrees(math.acos(np.dot(vrot_ev, VE_EV) / np.linalg.norm(vrot_ev) / VE_SPEED))))
# uncertainty budget on |v_E| from the solar motion (linear propagation)
uhat = VE_EV / VE_SPEED
sig_speed = math.sqrt((uhat[1] * SIG_V0) ** 2 + np.sum((uhat * SIG_PEC) ** 2))
log('sigma_vE_from_solar_motion_kms', float(sig_speed))
log('sigma_vE_v0_only_kms', float(abs(uhat[1]) * SIG_V0))

# ---------------------------------------------------------------------------------------------
# B. Comparisons: wimprates, WimPyDD, lzcommon, the annual curve
# ---------------------------------------------------------------------------------------------
import wimprates as wr
import numericalunits as nu
j2000_ev = wr.j2000_from_ymd(2023, 6, 16.0 + (21 + 22 / 60 + 39 / 3600) / 24.0)
vE_wr = wr.v_earth(j2000_ev) / (nu.km / nu.s)
vE_wr_vec = wr.earth_velocity(j2000_ev) / (nu.km / nu.s)
log('wimprates_j2000_event_days', float(j2000_ev))
log('wimprates_vE_event_kms', float(vE_wr))
log('wimprates_vE_vector_kms', np.asarray(vE_wr_vec).tolist())
WD = lz.wd()
lam_wd = (doy_ev - 80.0) * 2 * np.pi / 365.0
v_es_wd = WD.v_earth_sun(lam_wd, v_rot_gal=[0., V0, 0.], v_sun_rot=list(VPEC_BAXTER))
log('wimpydd_ecliptic_longitude_deg', float(np.degrees(lam_wd)))
log('wimpydd_vE_event_kms', float(np.linalg.norm(VSUN + v_es_wd)))
log('wimpydd_earth_sun_vector_kms', np.asarray(v_es_wd).tolist())
log('lzcommon_cosine_vE_event_kms', lz.v_earth_kms(doy_ev))
log('lzcommon_sunframe_kms', lz.v_earth_kms())
log('P055_vE_kms', 265.96); log('P006_vE_kms', 266.0)
log('orbital_vector_diff_astropy_minus_wimprates_kms', (vorb_ev - (np.asarray(vE_wr_vec) - VSUN)).tolist())
log('orbital_vector_diff_astropy_minus_wimpydd_kms', (vorb_ev - np.asarray(v_es_wd)).tolist())

# hourly scan over the LZ run window (27 Mar 2023 - 1 Apr 2024) and the calendar year 2023
t_run0, t_run1 = Time('2023-03-27T00:00:00'), Time('2024-04-01T00:00:00')
n_h = int(round((t_run1 - t_run0).to_value('day') * 24)) + 1
t_scan = t_run0 + np.arange(n_h) * u.hour
r_scan, v_scan, _ = helio_state_gal(t_scan)
vE_scan_vec = VSUN[None, :] + v_scan
vE_scan = np.linalg.norm(vE_scan_vec, axis=1)
doy_scan = (t_scan - Time('2023-01-01T00:00:00')).to_value('day') + 1.0
i_max = int(np.argmax(vE_scan)); i_min = int(np.argmin(vE_scan))
log('vE_max_run_kms', float(vE_scan[i_max])); log('t_vE_max_run', t_scan[i_max].isot)
log('vE_min_run_kms', float(vE_scan[i_min])); log('t_vE_min_run', t_scan[i_min].isot)
yr = t_scan < t_run0 + 365.25 * u.day          # one full year from the start of the run
log('vE_annual_mean_speed_kms', float(np.mean(vE_scan[yr])))
log('vE_run_mean_speed_kms', float(np.mean(vE_scan)))
log('vE_2_June_2023_00UTC_kms', float(np.linalg.norm(VSUN + helio_state_gal(Time('2023-06-02T00:00:00'))[1])))
log('days_event_after_vE_max', float((t_ev - t_scan[i_max]).to_value('day')))
log('vE_event_minus_max_kms', VE_SPEED - float(vE_scan[i_max]))
# first-harmonic fit over one year
yr_t = doy_scan[yr]; yr_v = vE_scan[yr]
A = np.column_stack([np.ones_like(yr_t), np.cos(2 * np.pi * yr_t / 365.25), np.sin(2 * np.pi * yr_t / 365.25)])
c, *_ = np.linalg.lstsq(A, yr_v, rcond=None)
amp = math.hypot(c[1], c[2]); phase_day = (math.degrees(math.atan2(c[2], c[1])) / 360.0 * 365.25) % 365.25
log('first_harmonic_mean_amp_peakday', [float(c[0]), float(amp), float(phase_day)])
log('first_harmonic_rms_residual_kms', float(np.std(yr_v - A @ c)))
pd.DataFrame(dict(isot=t_scan.isot, doy=doy_scan, vE_U=vE_scan_vec[:, 0], vE_V=vE_scan_vec[:, 1],
                  vE_W=vE_scan_vec[:, 2], vE=vE_scan)).iloc[::6].to_csv(OUT + '/vE_run_window_6h.csv', index=False)

# ---------------------------------------------------------------------------------------------
# C. Kinematics
# ---------------------------------------------------------------------------------------------
MASSES = [400.0, 1000.0, 4000.0]
VESCS = [500.0, 528.0, 544.0, 560.0, 580.0, 600.0]
E_EV = 248.0
rows_k = []
for m in MASSES:
    for ve in VESCS:
        vmax = ve + VE_SPEED
        rows_k.append(dict(m_GeV=m, v_esc=ve, v_max=vmax, delta_max_248=lz.delta_max_kev(E_EV, m, v_kms=vmax),
                           delta_max_248_norot=lz.delta_max_kev(E_EV, m, v_kms=ve + np.linalg.norm(VE_EV_NOROT)),
                           delta_max_248_annualmean=lz.delta_max_kev(E_EV, m, v_kms=ve + float(np.mean(vE_scan[yr]))),
                           delta_max_248_sunframe=lz.delta_max_kev(E_EV, m, v_kms=ve + lz.v_earth_kms()),
                           delta_max_248_vEmax=lz.delta_max_kev(E_EV, m, v_kms=ve + float(vE_scan[i_max])),
                           Emax_elastic=lz.E_R_range_keV(m, vmax)[1],
                           Emax_d300=lz.E_R_range_keV(m, vmax, delta_kev=300.)[1],
                           Emax_d350=lz.E_R_range_keV(m, vmax, delta_kev=350.)[1],
                           Emax_d366=lz.E_R_range_keV(m, vmax, delta_kev=366.)[1],
                           Emin_d366=lz.E_R_range_keV(m, vmax, delta_kev=366.)[0]))
df_k = pd.DataFrame(rows_k); df_k.to_csv(OUT + '/kinematics_event.csv', index=False)
print(df_k.to_string())
mN = lz.m_nucleus_gev(lz.A_XE_MEAN)
slope = math.sqrt(2 * mN * E_EV * 1e-6) / C * 1e6
log('d_deltamax_dv_keV_per_kms', slope)
log('deltamax_shift_from_rotation_keV', slope * float(np.dot(vrot_ev, VE_EV_NOROT) / np.linalg.norm(VE_EV_NOROT)))
log('deltamax_sigma_from_solar_motion_keV', slope * sig_speed)
log('deltamax_shift_v0_220_250_keV', [slope * (df_var.vE_speed[2] - VE_SPEED), slope * (df_var.vE_speed[5] - VE_SPEED)])
log('deltamax_shift_schoenrich_keV', slope * (df_var.vE_speed[1] - VE_SPEED))

# ---------------------------------------------------------------------------------------------
# D. Gravitational focusing by the Sun
# ---------------------------------------------------------------------------------------------
def vinf_map(v, rhat, gm_over_r):
    """Asymptotic (incoming) velocity of the hyperbolic Kepler orbit that has velocity v at position r (unit
    vector rhat, potential GM/r).  Alenazi & Gondolo 2006 / Lee-Lisanti-Peter-Safdi 2014 form (recalled;
    verified below by backward orbit integration).  v: (...,3); rhat: (3,).  Bound orbits -> nan."""
    v2 = np.sum(v * v, axis=-1)
    u2 = v2 - 2.0 * gm_over_r
    with np.errstate(invalid='ignore'):
        uu = np.sqrt(u2)
    vr = np.tensordot(v, rhat, axes=([-1], [0]))
    num = u2[..., None] * v + (uu * gm_over_r)[..., None] * rhat - (uu * vr)[..., None] * v
    den = u2 + gm_over_r - uu * vr
    return num / den[..., None]

def validate_map(n=6, seed=1):
    """Backward-integrate orbits from (r, v) at 1 AU under the Sun and compare the far-field velocity."""
    rng = np.random.default_rng(seed)
    rhat0 = np.array([1.0, 0.0, 0.0]); r0 = AU_KM * rhat0
    out = []
    for _ in range(n):
        v = rng.normal(size=3); v = v / np.linalg.norm(v) * rng.uniform(150, 800)
        def f(t, y):
            r = y[:3]; rr = np.linalg.norm(r)
            return np.concatenate([y[3:], -GM_SUN_KM3S2 * r / rr ** 3])
        sol = solve_ivp(f, [0, -3.0e9], np.concatenate([r0, v]), rtol=1e-11, atol=1e-6, method='DOP853')
        v_far = sol.y[3:, -1]; r_far = np.linalg.norm(sol.y[:3, -1])
        # residual potential correction at the final radius (speed only; direction converged)
        v_pred = vinf_map(v, rhat0, GM_SUN_KM3S2 / AU_KM)
        out.append(dict(v_in=v.tolist(), v_far_num=v_far.tolist(), v_inf_formula=v_pred.tolist(),
                        r_far_AU=r_far / AU_KM, abs_diff_kms=float(np.linalg.norm(v_far - v_pred)),
                        angle_diff_deg=float(math.degrees(math.acos(np.clip(np.dot(v_far, v_pred) / np.linalg.norm(v_far) / np.linalg.norm(v_pred), -1, 1))))))
    return out
val_map = validate_map()
json.dump(val_map, open(OUT + '/focusing_map_validation.json', 'w'), indent=1)
log('focusing_map_max_abs_diff_kms', max(d['abs_diff_kms'] for d in val_map))
log('focusing_map_max_angle_diff_deg', max(d['angle_diff_deg'] for d in val_map))

# angular quadrature (polar axis along the wind arrival direction) and speed grid
NMU, NPHI = 320, 48
mu_nodes, mu_w = np.polynomial.legendre.leggauss(NMU)
phi_nodes = np.arange(NPHI) * 2 * np.pi / NPHI; dphi = 2 * np.pi / NPHI
W_GRID = np.arange(0.0, 1100.0 + 1e-9, 2.0)
N_ESC = erf(VESC / V0) - 2 * (VESC / V0) / math.sqrt(math.pi) * math.exp(-(VESC / V0) ** 2)
F_NORM = 1.0 / (math.pi ** 1.5 * V0 ** 3 * N_ESC)      # unit density at infinity

def basis(zhat):
    zhat = zhat / np.linalg.norm(zhat)
    a = np.array([1.0, 0, 0]) if abs(zhat[0]) < 0.9 else np.array([0, 1.0, 0])
    xhat = np.cross(zhat, a); xhat /= np.linalg.norm(xhat); yhat = np.cross(zhat, xhat)
    return xhat, yhat, zhat

def lab_speed_density(vE_vec, r_vec_km=None, gm=0.0, earth=None, v0=V0, vesc=VESC, vsun=VSUN, chunk=40):
    """Speed density f_w(w) (per km/s, unit density at infinity) of the Earth-frame DM speed w, for an observer
    moving at vE_vec (Galactic frame) at heliocentric position r_vec_km, with Sun focusing (gm = GM_sun) and
    optionally Earth focusing earth=(vrot_vec, zenith_hat, GM_E/R_E).  f_lab(w) = f_gal(v_inf(w) + ...) (Liouville)."""
    xh, yh, zh = basis(-vE_vec)
    st = np.sqrt(1 - mu_nodes ** 2)
    dirs = (st[:, None, None] * np.cos(phi_nodes)[None, :, None] * xh + st[:, None, None] * np.sin(phi_nodes)[None, :, None] * yh
            + mu_nodes[:, None, None] * zh)                      # (NMU, NPHI, 3)
    weights = (mu_w[:, None] * dphi)                              # (NMU, NPHI)
    fw = np.zeros_like(W_GRID)
    rhat = None if r_vec_km is None else r_vec_km / np.linalg.norm(r_vec_km)
    for i0 in range(0, len(W_GRID), chunk):
        w = W_GRID[i0:i0 + chunk]
        wvec = w[:, None, None, None] * dirs[None]                # lab-frame velocity vectors
        if earth is not None:
            vrot, zen, gm_e_over_R = earth
            wgeo = wvec + vrot                                    # relative to Earth's centre
            wvec = vinf_map(wgeo, zen, gm_e_over_R)               # far from Earth
        # Sun-frame DM velocity: w + (observer velocity relative to the Sun, i.e. orbital part only)
        v = wvec + (vE_vec - vsun - (earth[0] if earth is not None else 0.0))
        if gm > 0:
            vinf = vinf_map(v, rhat, gm / np.linalg.norm(r_vec_km))
        else:
            vinf = v
        vgal = vinf + vsun
        g2 = np.sum(vgal * vgal, axis=-1)
        f = F_NORM * np.exp(-g2 / v0 ** 2) * (g2 < vesc ** 2)
        f = np.where(np.isnan(f), 0.0, f)
        fw[i0:i0 + chunk] = w ** 2 * np.sum(f * weights[None], axis=(1, 2))
    return fw

def eta_from_fw(fw, vgrid=W_GRID):
    """eta(v) = int_v^inf f_w/w dw on the grid (trapezoid from the top)."""
    integ = np.zeros_like(fw); integ[1:] = fw[1:] / vgrid[1:]
    seg = 0.5 * (integ[1:] + integ[:-1]) * np.diff(vgrid)
    eta = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
    return eta

def density(fw, vgrid=W_GRID):
    return float(np.trapezoid(fw, vgrid))

def hard_edge(fw, vgrid=W_GRID, rel=1e-8):
    m = fw > rel * fw.max()
    return float(vgrid[np.max(np.nonzero(m)[0])])

def exact_edge(vE_vec, r_vec_km=None, gm=0.0, earth=None, vsun=VSUN, vesc=VESC, cone_deg=12.0, nth=60, nph=36):
    """Largest lab-frame speed with f > 0 (|v_gal| = v_esc), by bisection along directions within a cone
    around the wind axis; resolves the ~1 km/s focusing shift that the 2 km/s grid cannot."""
    xh, yh, zh = basis(-vE_vec)
    th = np.linspace(0, math.radians(cone_deg), nth); ph = np.arange(nph) * 2 * np.pi / nph
    dirs = (np.sin(th)[:, None, None] * np.cos(ph)[None, :, None] * xh + np.sin(th)[:, None, None] * np.sin(ph)[None, :, None] * yh
            + np.cos(th)[:, None, None] * zh)
    rhat = None if r_vec_km is None else r_vec_km / np.linalg.norm(r_vec_km)
    def g(w):
        wvec = w[..., None] * dirs
        if earth is not None:
            wvec = vinf_map(wvec + earth[0], earth[1], earth[2])
        v = wvec + (vE_vec - vsun - (earth[0] if earth is not None else 0.0))
        vinf = vinf_map(v, rhat, gm / np.linalg.norm(r_vec_km)) if gm > 0 else v
        vg = vinf + vsun
        return np.sum(vg * vg, axis=-1) - vesc ** 2
    lo = np.full(dirs.shape[:2], 600.0); hi = np.full(dirs.shape[:2], 1000.0)
    for _ in range(40):
        mid = 0.5 * (lo + hi); inside = g(mid) < 0
        lo = np.where(inside, mid, lo); hi = np.where(inside, hi, mid)
    return float(np.max(lo))

# grid accuracy: unfocused numerical eta vs analytic (lz.eta0), event day
t0 = time.time()
fw_un = lab_speed_density(VE_EV, None, 0.0)
eta_un = eta_from_fw(fw_un)
V_TEST = np.array([0.0, 300.0, 500.0, 600.0, 700.0, 750.0, 780.0, 790.0, 800.0])
eta_an = lz.eta0(V_TEST, v_e=VE_SPEED)
eta_num = np.interp(V_TEST, W_GRID, eta_un)
log('grid_validation_eta_num_over_analytic', (eta_num / eta_an).tolist())
log('grid_validation_density_unfocused', density(fw_un))
log('one_lab_distribution_seconds', time.time() - t0)

# focused, event day (full observer velocity incl. rotation; Sun focusing uses the Sun-frame velocity)
fw_fo = lab_speed_density(VE_EV, r_ev, GM_SUN_KM3S2)
eta_fo = eta_from_fw(fw_fo)
V_ETA = np.array([600.0, 700.0, 750.0, 790.0])
def eta_ratio(eta_a, eta_b, vv=V_ETA):
    return (np.interp(vv, W_GRID, eta_a) / np.interp(vv, W_GRID, eta_b)).tolist()
log('event_density_enhancement', density(fw_fo) / density(fw_un))
log('event_eta_ratio_focused_over_unfocused_600_700_750_790', eta_ratio(eta_fo, eta_un))
log('event_hard_edge_unfocused_focused_kms', [hard_edge(fw_un), hard_edge(fw_fo)])
EDGE_UN, EDGE_FO = exact_edge(VE_EV), exact_edge(VE_EV, r_ev, GM_SUN_KM3S2)
log('event_exact_edge_unfocused_focused_kms', [EDGE_UN, EDGE_FO])
log('event_exact_edge_unfocused_minus_vesc_plus_vE', EDGE_UN - (VESC + VE_SPEED))
log('deltamax_shift_sun_focusing_keV_1TeV', lz.delta_max_kev(E_EV, 1000.0, v_kms=EDGE_FO) - lz.delta_max_kev(E_EV, 1000.0, v_kms=EDGE_UN))
v_esc_sun_1AU = math.sqrt(2 * GM_SUN_KM3S2 / np.linalg.norm(r_ev))
log('sun_escape_speed_at_earth_kms', v_esc_sun_1AU)
vmax_sun_inf = VESC + np.linalg.norm(VSUN)
log('analytic_vmax_shift_sun_focusing_kms', math.sqrt(vmax_sun_inf ** 2 + v_esc_sun_1AU ** 2) - vmax_sun_inf)
angle_r_wind = math.degrees(math.acos(np.dot(r_ev / np.linalg.norm(r_ev), -VSUN / np.linalg.norm(VSUN))))
log('event_angle_between_earth_position_and_downstream_deg', angle_r_wind)

# through the year: 36 dates (365.25/36 = 10.15-day steps from 27 Mar 2023) + event
dates = Time('2023-03-27T00:00:00') + np.arange(36) * (365.25 / 36) * u.day
rows_f = []
t0 = time.time()
ETA_STORE = {}
for tt in list(dates) + [t_ev]:
    rr, vv_orb, _ = helio_state_gal(tt)
    vE = VSUN + vv_orb
    fwu = lab_speed_density(vE, None, 0.0)
    fwf = lab_speed_density(vE, rr, GM_SUN_KM3S2)
    eu, ef = eta_from_fw(fwu), eta_from_fw(fwf)
    key = tt.isot
    ETA_STORE[key] = (eu, ef)
    ang = math.degrees(math.acos(np.dot(rr / np.linalg.norm(rr), -VSUN / np.linalg.norm(VSUN))))
    d = dict(isot=key, doy=float((tt - Time('2023-01-01')).to_value('day') + 1.0), vE=float(np.linalg.norm(vE)),
             r_AU=float(np.linalg.norm(rr) / AU_KM), angle_pos_downstream_deg=ang,
             density_ratio=density(fwf) / density(fwu), edge_unfoc=exact_edge(vE), edge_foc=exact_edge(vE, rr, GM_SUN_KM3S2))
    for vv_, rat in zip(V_ETA, eta_ratio(ef, eu)):
        d[f'eta_ratio_{int(vv_)}'] = rat
    for vv_ in V_ETA:
        d[f'eta_unfoc_{int(vv_)}'] = float(np.interp(vv_, W_GRID, eu))
        d[f'eta_foc_{int(vv_)}'] = float(np.interp(vv_, W_GRID, ef))
    rows_f.append(d)
df_f = pd.DataFrame(rows_f); df_f.to_csv(OUT + '/focusing_through_year.csv', index=False)
print(df_f[['isot', 'vE', 'angle_pos_downstream_deg', 'density_ratio', 'eta_ratio_600', 'eta_ratio_700', 'eta_ratio_750', 'eta_ratio_790']].to_string())
log('focusing_year_scan_seconds', time.time() - t0)
yr_rows = df_f.iloc[:36]                      # 36 equally spaced dates covering exactly one year
log('edge_shift_focusing_min_max_kms', [float((yr_rows.edge_foc - yr_rows.edge_unfoc).min()), float((yr_rows.edge_foc - yr_rows.edge_unfoc).max())])
i_dmax = int(np.argmax(yr_rows.density_ratio.values))
log('focusing_density_max_date_value', [yr_rows.isot.iloc[i_dmax], float(yr_rows.density_ratio.iloc[i_dmax])])
log('focusing_density_annual_mean', float(yr_rows.density_ratio.mean()))
for vv_ in V_ETA:
    col = f'eta_ratio_{int(vv_)}'
    fin = yr_rows[col].replace([np.inf, -np.inf], np.nan).dropna()   # dates where eta_unfocused(v) = 0 are excluded
    log(f'eta_ratio_mean_over_dates_{int(vv_)}', [float(fin.mean()), int(len(fin))])
    log(f'eta_ratio_min_max_{int(vv_)}', [float(fin.min()), float(fin.max())])
# annual-mean eta (unfocused/focused) from the 37 dates, and focused/unfocused ratio of annual means
eta_un_annual = np.mean([ETA_STORE[k][0] for k in yr_rows.isot], axis=0)
eta_fo_annual = np.mean([ETA_STORE[k][1] for k in yr_rows.isot], axis=0)
log('annual_mean_eta_ratio_focused_over_unfocused', eta_ratio(eta_fo_annual, eta_un_annual))
log('event_over_annual_eta_ratio_unfocused', eta_ratio(eta_un, eta_un_annual))
log('event_over_annual_eta_ratio_focused', eta_ratio(eta_fo, eta_fo_annual))
# March row (closest to the density maximum) for the paper's June vs March comparison
k_march = yr_rows.isot.iloc[i_dmax]
eta_un_march, eta_fo_march = ETA_STORE[k_march]
log('march_eta_ratio_focused_over_unfocused', eta_ratio(eta_fo_march, eta_un_march))

# ---------------------------------------------------------------------------------------------
# E. Earth focusing and shadowing
# ---------------------------------------------------------------------------------------------
gm_e_over_R = GM_EARTH_KM3S2 / (R_EARTH_KM + SURF_HEIGHT_M / 1000.0)
fw_fe = lab_speed_density(VE_EV, r_ev, GM_SUN_KM3S2, earth=(vrot_ev, zenith_ev, gm_e_over_R))
eta_fe = eta_from_fw(fw_fe)
log('earth_escape_speed_kms', math.sqrt(2 * gm_e_over_R))
log('earth_focusing_density_ratio', density(fw_fe) / density(fw_fo))
log('earth_focusing_eta_ratio_600_700_750_790', eta_ratio(eta_fe, eta_fo))
EDGE_FE = exact_edge(VE_EV, r_ev, GM_SUN_KM3S2, earth=(vrot_ev, zenith_ev, gm_e_over_R))
log('earth_focusing_exact_edge_kms', EDGE_FE)
log('deltamax_shift_earth_focusing_keV_1TeV', lz.delta_max_kev(E_EV, 1000.0, v_kms=EDGE_FE) - lz.delta_max_kev(E_EV, 1000.0, v_kms=EDGE_FO))
log('analytic_vmax_shift_earth_kms', math.sqrt(810.0 ** 2 + 2 * gm_e_over_R) - 810.0)
# shadowing: inelastic scattering in Earth materials is kinematically forbidden for delta >= 300 keV
mu_targets = {'O-16': 16, 'Si-28': 28, 'Fe-56': 56, 'Ni-58': 58, 'Pb-208': 208, 'Xe-131': 131.29}
thr = {k: C * math.sqrt(2 * 300e-6 / lz.mu_red(1000.0, lz.m_nucleus_gev(A))) for k, A in mu_targets.items()}
log('inelastic_threshold_speed_delta300_1TeV_kms', thr)
# column density of the Earth (diameter) and elastic interaction probability at a loop-level sigma
col_nucleons = 5.51 * 2 * R_EARTH_KM * 1e5 / 1.6605e-24    # nucleons/cm^2 through the diameter (mean density 5.51 g/cm3, certain)
log('earth_column_nucleons_cm2', col_nucleons)
log('elastic_interaction_prob_sigma_1e-48_A56coherent', col_nucleons / 56 * 56 ** 2 * (lz.mu_red(1000, 56 * lz.AMU_GEV) / lz.mu_red(1000, lz.M_NUCLEON_GEV)) ** 2 * 1e-48)

# ---------------------------------------------------------------------------------------------
# F. WimPyDD rates: Higgsino (P007) at 1 TeV; halos: Sun-frame, WimPyDD June-16, annual, numeric June-16
#    unfocused/focused, annual focused/unfocused, Earth-focused
# ---------------------------------------------------------------------------------------------
GF, SW2 = 1.166e-5, 0.231
c_p = 2 * math.sqrt(2) * GF * 0.25 * (1 - 4 * SW2); c_n = -2 * math.sqrt(2) * GF * 0.25
ham = lz.wd_hamiltonian('P085_higgsino_Z', {1: (c_p + c_n, c_p - c_n)})
VGRID, deta_sun = lz.wd_halo()
ONES = np.ones_like(VGRID)
EXPO = lz.LZ['exposure_tyr']
E50_HI, E50_LO, PLATEAU = lz.LZ['E_50pct_high_keV'], lz.LZ['E_50pct_low_keV'], lz.LZ['eff_plateau']
def efficiency(E, sig_hi=11.5, sig_lo=2.5):
    return PLATEAU * 0.5 * (1 + erf((E - E50_LO) / (math.sqrt(2) * sig_lo))) * 0.5 * erfc((E - E50_HI) / (math.sqrt(2) * sig_hi))

def deta_from_eta(eta_grid, vgrid=W_GRID):
    """WimPyDD delta-eta array on VGRID: delta_eta[i] = eta(VGRID[i-1]) - eta(VGRID[i])."""
    e = np.interp(VGRID, vgrid, eta_grid)
    d = np.zeros_like(e); d[1:] = e[:-1] - e[1:]
    return d

# check the convention against WimPyDD's own June halo
deta_wd_june = lz.wd_halo(day_of_year=doy_ev)[1]
deta_my_june = deta_from_eta(eta_un)
eta_wd_june = np.cumsum(deta_wd_june[::-1])[::-1]     # eta at lower bin edge
log('convention_check_sum_deta_wd_vs_mine', [float(deta_wd_june.sum()), float(deta_my_june.sum()), float(lz.eta0(0.0, v_e=VE_SPEED))])
for vv_ in [600, 700, 750, 790]:
    i = int(np.searchsorted(VGRID, vv_))
    log(f'eta_wd_over_mine_{vv_}', float(eta_wd_june[i] / np.interp(VGRID[i - 1], W_GRID, eta_un)))
days24 = 7.6 + 365.25 / 24 * np.arange(24)
deta_wd_annual = np.mean([lz.wd_halo(day_of_year=d)[1] for d in days24], axis=0)
halos = {'sun_frame_WimPyDD': deta_sun, 'june16_WimPyDD': deta_wd_june, 'annual_WimPyDD_24d': deta_wd_annual,
         'june16_num_unfocused': deta_my_june, 'june16_num_focused': deta_from_eta(eta_fo),
         'june16_num_focused_earth': deta_from_eta(eta_fe),
         'annual_num_unfocused': deta_from_eta(eta_un_annual), 'annual_num_focused': deta_from_eta(eta_fo_annual),
         'march_num_unfocused': deta_from_eta(eta_un_march), 'march_num_focused': deta_from_eta(eta_fo_march)}
DELTAS = sorted(set([300.0, 350.0, 366.0, 380.0] + list(np.arange(352.5, 392.6, 2.5))))
M_RATE = 1000.0
rows_r = []
t0 = time.time()
for d in DELTAS:
    lo = lz.E_R_range_keV(M_RATE, VGRID[-1], A=124.0, delta_kev=d)[0]
    if math.isnan(lo):
        continue
    E = np.arange(max(1.0, math.floor(lo) - 3.0), 330.0 + 1e-9, 3.0)
    K = np.array([WD.diff_rate(WD.Xe, ham, M_RATE, float(e), VGRID, ONES, j_chi=0.5, delta=float(d), sum_over_streams=False) for e in E])
    K *= 1000.0 * 365.25
    eff = efficiency(E)
    for name, de in halos.items():
        dR = K @ de
        rows_r.append(dict(delta=d, halo=name, N_events=float(EXPO * np.trapezoid(eff * dR, E))))
log('kernel_seconds', time.time() - t0)
df_r = pd.DataFrame(rows_r)
piv = df_r.pivot(index='delta', columns='halo', values='N_events')
piv.to_csv(OUT + '/higgsino_rates_1TeV.csv')
print(piv.to_string())
def delta_N1(col):
    x = piv.index.values; y = np.log(np.clip(piv[col].values, 1e-300, None))
    ok = piv[col].values > 0
    return float(np.interp(0.0, -y[ok], x[ok]))
dN1 = {c: delta_N1(c) for c in piv.columns}
log('delta_N1_keV', dN1)
ratios = {}
for d in [300.0, 350.0, 366.0, 380.0]:
    r = piv.loc[d]
    ratios[d] = dict(focused_over_unfocused_june=r['june16_num_focused'] / r['june16_num_unfocused'],
                     earth_focus_extra=r['june16_num_focused_earth'] / r['june16_num_focused'],
                     focused_over_unfocused_annual=r['annual_num_focused'] / r['annual_num_unfocused'],
                     focused_over_unfocused_march=r['march_num_focused'] / r['march_num_unfocused'],
                     june_over_annual_unfocused=r['june16_num_unfocused'] / r['annual_num_unfocused'],
                     june_over_annual_focused=r['june16_num_focused'] / r['annual_num_focused'],
                     my_june_over_WimPyDD_june=r['june16_num_unfocused'] / r['june16_WimPyDD'],
                     my_annual_over_WimPyDD_annual=r['annual_num_unfocused'] / r['annual_WimPyDD_24d'],
                     june_WimPyDD_over_sunframe=r['june16_WimPyDD'] / r['sun_frame_WimPyDD'])
log('rate_ratios', ratios)

# ---------------------------------------------------------------------------------------------
# G. Fraction of the year / run with delta_max(248 keV, m) >= thresholds
# ---------------------------------------------------------------------------------------------
rows_g = []
dm_ev = {m: lz.delta_max_kev(E_EV, m, v_kms=VESC + VE_SPEED) for m in MASSES}
for m in MASSES:
    for ve in [528.0, 544.0, 560.0]:
        dmax_scan = np.array([lz.delta_max_kev(E_EV, m, v_kms=ve + v) for v in vE_scan])
        for thr_ in [370.0, 380.0, 385.0, 387.0, 390.0]:
            rows_g.append(dict(m_GeV=m, v_esc=ve, threshold_keV=thr_,
                               frac_year=float(np.mean(dmax_scan[yr] >= thr_)), frac_run=float(np.mean(dmax_scan >= thr_)),
                               days_per_year=float(np.mean(dmax_scan[yr] >= thr_) * 365.25)))
        rows_g.append(dict(m_GeV=m, v_esc=ve, threshold_keV=float(lz.delta_max_kev(E_EV, m, v_kms=ve + VE_SPEED)),
                           frac_year=float(np.mean(vE_scan[yr] >= VE_SPEED)), frac_run=float(np.mean(vE_scan >= VE_SPEED)),
                           days_per_year=float(np.mean(vE_scan[yr] >= VE_SPEED) * 365.25)))
df_g = pd.DataFrame(rows_g); df_g.to_csv(OUT + '/fraction_of_year_deltamax.csv', index=False)
print(df_g[df_g.m_GeV == 1000].to_string())
log('frac_year_vE_ge_event', float(np.mean(vE_scan[yr] >= VE_SPEED)))
log('frac_run_vE_ge_event', float(np.mean(vE_scan >= VE_SPEED)))
log('delta_max_event_by_mass', dm_ev)

# ---------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
C1, C2, C3, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e5e5e5', 'grid.linewidth': 0.6})
fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.2), sharex=True)
ax = axes[0]
ax.plot(doy_scan, vE_scan, color=C1, lw=1.6, label='|v_E| (astropy, Baxter solar motion)')
ax.axvline(doy_ev, color=C2, lw=1.2, ls='--'); ax.plot([doy_ev], [VE_SPEED], 'o', color=C2, ms=6)
ax.annotate(f'16 Jun 21:22 UTC\n{VE_SPEED:.2f} km/s', (doy_ev, VE_SPEED), xytext=(doy_ev + 12, VE_SPEED - 4), fontsize=8, color=C2)
ax.axhline(lz.v_earth_kms(), color='#777', lw=0.9, ls=':'); ax.text(470, lz.v_earth_kms() + 0.6, 'Sun frame 250.6', fontsize=8, color='#555')
ax.set_ylabel('Earth speed in halo frame [km/s]')
ax.set_title('Earth velocity through the LZ run (27 Mar 2023 - 1 Apr 2024); day 1 = 1 Jan 2023', fontsize=9.5)
ax2 = axes[1]
dm1 = np.array([lz.delta_max_kev(E_EV, 1000.0, v_kms=VESC + v) for v in vE_scan])
ax2.plot(doy_scan, dm1, color=C3, lw=1.6, label='delta_max(248 keV, 1 TeV, v_esc = 544)')
for thr_, ls in [(380, ':'), (385, '-.'), (387, '--')]:
    ax2.axhline(thr_, color='#777', lw=0.8, ls=ls); ax2.text(470, thr_ + 0.4, f'{thr_} keV', fontsize=8, color='#555')
ax2.axvline(doy_ev, color=C2, lw=1.2, ls='--')
ax2.set_ylabel('delta_max [keV]'); ax2.set_xlabel('day of year 2023 (continuing into 2024)')
ax2.set_ylim(360, 392)
fig.tight_layout(); fig.savefig(FIG + '/P085_fig1_vE_run.png', dpi=160); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
ax = axes[0]
ax.plot(yr_rows.doy, (yr_rows.density_ratio - 1) * 100, color=C1, lw=1.6, marker='o', ms=3, label='DM density at Earth, n/n_inf - 1')
ax.axvline(doy_ev, color=C2, lw=1.2, ls='--'); ax.set_xlabel('day of year 2023'); ax.set_ylabel('percent')
ax.set_title('Solar gravitational focusing: density enhancement', fontsize=9.5)
ax = axes[1]
for vv_, col, lab in [(600, C1, 'v_min = 600'), (700, C3, '700'), (750, C4, '750'), (790, C2, '790 km/s')]:
    ax.plot(yr_rows.doy, (yr_rows[f'eta_ratio_{vv_}'] - 1) * 100, color=col, lw=1.6, marker='o', ms=3, label=lab)
ax.axvline(doy_ev, color=C2, lw=1.2, ls='--'); ax.set_xlabel('day of year 2023'); ax.set_ylabel('eta_focused / eta_unfocused - 1  [percent]')
ax.set_ylim(0, 40); ax.text(300, 37, '790 km/s curve exceeds 40 % (off scale) where the\nunfocused edge v_esc + v_E drops below ~795 km/s', fontsize=7.5, color='#555')
ax.set_title('Effect on the high-velocity tail eta(v_min)', fontsize=9.5); ax.legend(fontsize=8, frameon=False, loc='center left')
fig.tight_layout(); fig.savefig(FIG + '/P085_fig2_focusing.png', dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 3.6))
for name, col, lab in [('annual_num_unfocused', C1, 'annual mean'), ('june16_num_unfocused', C3, '16 June, no focusing'),
                       ('june16_num_focused', C2, '16 June, Sun focusing'), ('sun_frame_WimPyDD', '#777', 'WimPyDD Sun frame')]:
    ok = piv[name] > 0
    ax.semilogy(piv.index[ok], piv[name][ok], color=col, lw=1.6, label=lab)
ax.axhline(1.0, color='#999', lw=0.8, ls=':'); ax.set_xlabel('delta [keV]'); ax.set_ylabel('Higgsino events in 2.84 t yr (1 TeV)')
ax.set_title('Higgsino Z-exchange rate near the kinematic edge', fontsize=9.5); ax.legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(FIG + '/P085_fig3_higgsino_rates.png', dpi=160); plt.close(fig)

# ---------------------------------------------------------------------------------------------
# final kinematic-input table
# ---------------------------------------------------------------------------------------------
final = {
    'event_time_utc': t_ev.isot, 'day_of_year_(1Jan=1)': doy_ev, 'JD_UTC': t_ev.jd,
    'v_sun_gal_kms_(U,V,W)': VSUN.tolist(), 'v_orb_gal_kms': vorb_ev.tolist(), 'v_rot_SURF_gal_kms': vrot_ev.tolist(),
    'v_E_gal_kms_(U,V,W)': VE_EV.tolist(), '|v_E|_kms': VE_SPEED, '|v_E|_no_rotation_kms': float(np.linalg.norm(VE_EV_NOROT)),
    'sigma_|v_E|_solar_motion_kms': float(sig_speed),
    'apex_(l,b)_deg': list(lb(VE_EV)), 'wind_arrival_(l,b)_deg': list(lb(-VE_EV)),
    'apex_alt_az_SURF_deg': LOG['apex_altaz_at_SURF_deg'],
    'v_max_544_kms': VESC + VE_SPEED, 'v_max_544_with_sun_focusing_kms': EDGE_FO, 'v_max_544_with_sun_and_earth_focusing_kms': EDGE_FE,
    'delta_max_248_keV_by_mass_(400,1000,4000)': [dm_ev[m] for m in MASSES],
    'Emax_elastic_keV_by_mass': [lz.E_R_range_keV(m, VESC + VE_SPEED)[1] for m in MASSES],
    'vE_max_2023_kms': float(vE_scan[i_max]), 't_vE_max': t_scan[i_max].isot, 'vE_annual_mean_kms': float(np.mean(vE_scan[yr])),
    'vE_sun_frame_kms': lz.v_earth_kms(),
    'focusing_density_ratio_event': LOG['event_density_enhancement'],
    'focusing_eta_ratio_event_600_700_750_790': LOG['event_eta_ratio_focused_over_unfocused_600_700_750_790'],
    'higgsino_delta_N1_keV': dN1, 'frac_year_deltamax_ge_380_385_387_1TeV_544': [
        float(df_g[(df_g.m_GeV == 1000) & (df_g.v_esc == 544) & (df_g.threshold_keV == t)].frac_year.iloc[0]) for t in [380., 385., 387.]],
}
json.dump(final, open(OUT + '/P085_kinematic_inputs.json', 'w'), indent=1, default=float)
json.dump(LOG, open(OUT + '/P085_log.json', 'w'), indent=1, default=str)
print('total seconds', time.time() - T0)
