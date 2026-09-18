"""P092: Sidereal and diurnal effects on the high-velocity tail -- is there time-of-day information
in a delta ~ 300-385 keV inelastic reading of the LZ 248 keV event (16 June 2023 21:22:39 UTC)?

Parts
  1. Lab-frame velocity vector vs UTC on 16 June 2023: Sun (Baxter 2021) + Earth orbit (astropy ephemeris,
     validated against wimprates and P055) + Earth rotation at SURF (analytic GMST; no AltAz frame).
  2. delta_max(248 keV, m) and the O1 isoscalar inelastic in-ROI rate (1 TeV; delta = 300/350/366/380/385 keV)
     over the sidereal day.  Method: WimPyDD per-isotope response C_A(E) from a single fast unit stream
     (for O1, dR/dE = C_A(E) eta(v_min,A(E)) exactly) times the analytic Baxter-2021 halo function at the exact
     lab speed |v_lab(t)|; validated against WimPyDD's own day-of-year halo at the event time.
  3. Time-of-day likelihood ratio, information per event, N(3 sigma) for a diurnal modulation; ratio to P034's annual.
  4. Apex altitude/azimuth vs UTC at SURF (recoil direction for a directional detector).
  5. Gravitational focusing (Earth, Sun) and Earth shadowing: order-of-magnitude estimates.
  6. Hourly table of kinematic inputs.
Run from the simulation root:  .venv/bin/python output/code/P092_diurnal.py   (about 1-2 min; kernel cached)
"""
import sys, os, json, math, time
import numpy as np
sys.path.insert(0, 'output/code')
from common import lzcommon as lz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

T0 = time.time()
OUT = 'output/work/P092'; FIG = f'{OUT}/figures'
os.makedirs(FIG, exist_ok=True)
LOG = open(f'{OUT}/run_log.txt', 'w')
def log(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.write(s + '\n'); LOG.flush()
RECALLED = []
def rec(item, value, source, reliability):
    RECALLED.append(dict(item=item, value=str(value), presumed_source=source, reliability=reliability)); return value

# ------------------------------------------------------------------------------------------------
# 0. Inputs
# ------------------------------------------------------------------------------------------------
V0, VESC = lz.V0_KMS, lz.VESC_KMS                          # 238, 544 km/s (Baxter 2021 via lzcommon)
V_SUN_GAL = np.array([0.0, V0, 0.0]) + lz.V_SUN_PEC        # (11.1, 250.2, 7.3) km/s
M_CHI = 1000.0; E_EV = 248.0
DELTAS = [300.0, 350.0, 366.0, 380.0, 385.0]
MASSES = [400.0, 1000.0, 4000.0]
EV_UTC = lz.LZ['ev_time_utc']                              # '2023-06-16T21:22:39' (LZ paper)
LAT = rec('SURF Davis campus latitude 44.352 N, longitude 103.751 W', 44.352, 'geography (as P042/P067)', 'likely'); LON = -103.751
DEPTH_KM = rec('Davis campus depth 4850 ft = 1.478 km below surface', 1.478, 'SURF public description', 'likely')
OMEGA_E = rec('Earth sidereal rotation rate 7.2921150e-5 rad/s', 7.2921150e-5, 'IERS conventions', 'certain')
R_E = rec('Earth mean radius 6371.0 km; geocentric radius at 44.35 deg lat 6367.5 km (WGS84 a=6378.137, f=1/298.257)', 6367.5, 'WGS84', 'certain')
rec('GMST(h) = 18.697374558 + 24.06570982441908 D, D = JD(UT1) - 2451545.0; |UT1-UTC| < 1 s in 2023', 'formula', 'USNO', 'certain')
VESC_EARTH = rec('Earth surface escape speed 11.19 km/s', 11.19, 'standard', 'certain')
VESC_SUN_1AU = rec('Solar escape speed at 1 AU 42.1 km/s', 42.1, 'standard', 'certain')

# ------------------------------------------------------------------------------------------------
# 1. Frames and the lab velocity vector vs UTC
# ------------------------------------------------------------------------------------------------
from astropy.utils import iers
iers.conf.auto_download = False
from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel, SkyCoord, CartesianRepresentation
import astropy.units as u
import wimprates as wr, numericalunits as nu, pandas as pd

# ICRS -> Galactic fixed rotation from astropy (no IERS needed); columns = ICRS components of Galactic axes
R_G2I = np.array([SkyCoord(CartesianRepresentation(*e), frame='galactic').icrs.cartesian.xyz.value for e in np.eye(3)]).T
R_I2G = R_G2I.T
P055_EQ2GAL = np.array([[-0.0548755604, -0.8734370902, -0.4838350155],
                        [0.4941094279, -0.4448296300, 0.7469822445],
                        [-0.8676661490, -0.1980763734, 0.4559837762]])   # P055's recalled Hipparcos matrix (consistency check only)
log(f'[1] ICRS->Galactic matrix vs P055 Hipparcos matrix: max |diff| = {np.abs(R_I2G - P055_EQ2GAL).max():.2e}')

def v_orb_gal(t):
    """Earth heliocentric velocity (km/s, Galactic UVW) from astropy's built-in ephemeris (offline)."""
    _, ve = get_body_barycentric_posvel('earth', t); _, vs = get_body_barycentric_posvel('sun', t)
    v_icrs = (ve.xyz - vs.xyz).to(u.km / u.s).value            # shape (3,) or (3, N)
    return R_I2G @ v_icrs

def gmst_h(t):
    return (18.697374558 + 24.06570982441908 * (t.jd - 2451545.0)) % 24.0
def lst_deg(t):
    return (gmst_h(t) * 15.0 + LON) % 360.0

V_ROT = OMEGA_E * (R_E - DEPTH_KM) * math.cos(math.radians(LAT))   # km/s, eastward
rec('geodetic-geocentric latitude difference (0.19 deg) and 23-yr precession (0.32 deg) neglected: < 0.003 km/s', 'note', 'geometry', 'certain')
def v_rot_gal(t):
    """Rotational velocity of the SURF lab (km/s, Galactic), eastward: (-sin LST, cos LST, 0) in equatorial axes."""
    th = np.radians(lst_deg(t))
    v_eq = V_ROT * np.array([-np.sin(th), np.cos(th), np.zeros_like(th)])
    return R_I2G @ v_eq

T_EV = Time(EV_UTC, scale='utc')
T_DAY0 = Time('2023-06-16 00:00:00', scale='utc')
MIN = np.arange(0.0, 24.0 * 60 + 1.0, 1.0)                    # 1-min steps, 1441 points
T_GRID = T_DAY0 + MIN * u.min
HOURS = MIN / 60.0
V_ORB_EV = v_orb_gal(T_EV)
V_ORB_GRID = v_orb_gal(T_GRID)                                  # (3, N)
V_ROT_GRID = v_rot_gal(T_GRID)                                  # (3, N)
V_LAB_GEO_EV = V_SUN_GAL + V_ORB_EV                             # geocentric lab velocity at the event (no rotation)
V_LAB_ROT = V_LAB_GEO_EV[:, None] + V_ROT_GRID                  # rotation-only series (orbital velocity frozen at the event)
V_LAB_FULL = V_SUN_GAL[:, None] + V_ORB_GRID + V_ROT_GRID       # full series
S_ROT = np.linalg.norm(V_LAB_ROT, axis=0); S_FULL = np.linalg.norm(V_LAB_FULL, axis=0); S_GEO = np.linalg.norm(V_SUN_GAL[:, None] + V_ORB_GRID, axis=0)
V_LAB_EV_VEC = V_LAB_GEO_EV + v_rot_gal(T_EV); S_EV = float(np.linalg.norm(V_LAB_EV_VEC)); S_GEO_EV = float(np.linalg.norm(V_LAB_GEO_EV))

# sidereal-day mean of the rotation-only speed (23h56m04s = 1436.07 min)
SID_MIN = 1436.0682
msk = MIN <= SID_MIN
S_ROT_MEAN = float(np.trapezoid(S_ROT[msk], MIN[msk]) / (MIN[msk][-1] - MIN[msk][0]))
S_FULL_MEAN = float(np.trapezoid(S_FULL, MIN) / (MIN[-1] - MIN[0]))
i_ev = int(np.argmin(np.abs(HOURS - (21 + 22 / 60 + 39 / 3600))))

# wimprates validation of the geocentric velocity (Sun + orbit) at the event time
def v_wr(ts):
    v = np.array(wr.earth_velocity(wr.j2000(pd.Timestamp(ts)), v_0=V0 * nu.km / nu.s)) / (nu.km / nu.s)
    return v
V_WR_EV = v_wr(EV_UTC)
S_WR_EV = float(np.linalg.norm(V_WR_EV))
# ephemeris sanity: eccentric orbit (perihelion 4 Jan 2023, aphelion 6 July 2023; recalled, certain to +-1 d)
rec('Earth perihelion 4 Jan 2023 (~30.29 km/s), aphelion 6 Jul 2023 (~29.29 km/s); e = 0.0167', 'dates/speeds', 'almanac', 'certain')
EPH = {}
for lab, ts in [('perihelion 2023-01-04', '2023-01-04T16:00:00'), ('aphelion 2023-07-06', '2023-07-06T20:00:00'), ('event', EV_UTC)]:
    t = Time(ts, scale='utc'); pe, ve = get_body_barycentric_posvel('earth', t); ps, vs = get_body_barycentric_posvel('sun', t)
    r = (pe.xyz - ps.xyz).to(u.km).value; v = (ve.xyz - vs.xyz).to(u.km / u.s).value
    ang = math.degrees(math.acos(np.dot(r, v) / np.linalg.norm(r) / np.linalg.norm(v)))
    EPH[lab] = dict(speed_kms=float(np.linalg.norm(v)), r_AU=float(np.linalg.norm(r) / 1.495978707e8), r_v_angle_deg=ang)
    log(f'    ephemeris check {lab}: |v_orb| = {EPH[lab]["speed_kms"]:.3f} km/s, r = {EPH[lab]["r_AU"]:.4f} AU, angle(r,v) = {ang:.2f} deg')
DIFF_WR = V_LAB_GEO_EV - V_WR_EV
log(f'    (astropy - wimprates) orbital vector = {np.round(DIFF_WR,3)} km/s; its projection on the apex = {float(np.dot(DIFF_WR, V_LAB_GEO_EV/np.linalg.norm(V_LAB_GEO_EV))):+.3f} km/s '
    f'(= the 265.81 vs 265.98 difference); |v_orb| wimprates = {np.linalg.norm(V_WR_EV - V_SUN_GAL):.3f} (circular 29.79) vs astropy {np.linalg.norm(V_ORB_EV):.3f}')
P055_VE = 265.96311733275246     # P055 earth_velocity_validation.json (vE_mine at the event); wimprates there 265.977
log(f'    v_sun = {V_SUN_GAL} |v_sun| = {np.linalg.norm(V_SUN_GAL):.3f} km/s')
log(f'    v_orb(event) astropy = {np.round(V_ORB_EV,3)} |v_orb| = {np.linalg.norm(V_ORB_EV):.3f} km/s')
log(f'    geocentric |v_lab|(event) = {S_GEO_EV:.3f} km/s; wimprates {S_WR_EV:.3f}; P055 {P055_VE:.3f}; lzcommon cosine {lz.v_earth_kms(167.89):.2f}; '
    f'|astropy - wimprates| vector = {np.linalg.norm(V_LAB_GEO_EV - V_WR_EV):.3f} km/s')
log(f'    Earth rotation at SURF: v_rot = {V_ROT:.4f} km/s eastward (lat {LAT}, R = {R_E - DEPTH_KM:.1f} km)')
apex_gal = V_LAB_GEO_EV / S_GEO_EV
apex_icrs = R_G2I @ apex_gal
RA_AP = math.degrees(math.atan2(apex_icrs[1], apex_icrs[0])) % 360.0; DEC_AP = math.degrees(math.asin(apex_icrs[2]))
A_ROT = V_ROT * math.cos(math.radians(DEC_AP))                  # analytic diurnal amplitude of |v_lab|
log(f'    apex (geocentric lab velocity): RA {RA_AP:.2f} Dec {DEC_AP:.2f}; (l,b) = ({math.degrees(math.atan2(apex_gal[1],apex_gal[0]))%360:.2f}, {math.degrees(math.asin(apex_gal[2])):.2f})')
log(f'    analytic diurnal amplitude v_rot cos(Dec_apex) = {A_ROT:.4f} km/s (peak-to-peak {2*A_ROT:.4f})')
log(f'    rotation-only series: mean {S_ROT_MEAN:.4f}, min {S_ROT.min():.4f}, max {S_ROT.max():.4f}, p2p {S_ROT.max()-S_ROT.min():.4f} km/s; '
    f'UTC of max {HOURS[np.argmax(S_ROT[msk])]:.3f} h, of min {HOURS[np.argmin(S_ROT[msk])]:.3f} h')
log(f'    full series: 00:00 {S_FULL[0]:.4f} -> 24:00 {S_FULL[-1]:.4f} km/s (orbital drift {S_GEO[-1]-S_GEO[0]:+.4f} km/s/day); mean {S_FULL_MEAN:.4f}')
log(f'    at the event: |v_lab| = {S_EV:.4f} km/s = geocentric {S_GEO_EV:.4f} {S_EV - S_GEO_EV:+.4f}; relative to sidereal mean {S_EV - S_ROT_MEAN:+.4f} km/s; '
    f'rotation projection on apex {float(np.dot(v_rot_gal(T_EV), apex_gal)):+.4f} km/s')
# mean-speed check: rotation adds only <u_perp^2>/(2v) to the sidereal mean
log(f'    sidereal mean - geocentric = {S_ROT_MEAN - S_GEO_EV:+.2e} km/s (expected ~ v_rot^2/(4 v) = {V_ROT**2/(4*S_GEO_EV):.1e})')

# apex hour angle / altitude / azimuth vs UTC (analytic; P067 formulae)
def altaz(ra, dec, lst):
    ha = np.radians(lst - ra); lat = math.radians(LAT); d = math.radians(dec)
    alt = np.degrees(np.arcsin(math.sin(lat) * math.sin(d) + math.cos(lat) * math.cos(d) * np.cos(ha)))
    az = np.degrees(np.arctan2(-np.sin(ha) * math.cos(d), math.sin(d) * math.cos(lat) - math.cos(d) * math.sin(lat) * np.cos(ha))) % 360.0
    return alt, az, (np.degrees(ha) + 180.0) % 360.0 - 180.0
LST_GRID = lst_deg(T_GRID); LST_EV = float(lst_deg(T_EV))
ALT, AZ, HA = altaz(RA_AP, DEC_AP, LST_GRID)
alt_ev, az_ev, ha_ev = altaz(RA_AP, DEC_AP, LST_EV)
log(f'    LST(SURF, event) = {LST_EV/15:.4f} h (P067 8.120); apex HA {ha_ev/15:+.3f} h, alt {alt_ev:+.2f}, az {az_ev:.1f} (P067: -0.5, 347.4)')
log(f'    apex altitude range {ALT.min():+.2f} .. {ALT.max():+.2f} deg; below horizon {np.mean(ALT<0)*100:.1f} % of the day; '
    f'upper culmination at UTC {HOURS[np.argmax(ALT)]:.3f} h, lower at {HOURS[np.argmin(ALT)]:.3f} h; '
    f'apex due east (max lab speed) at UTC {HOURS[np.argmin(np.abs(HA+90))]:.3f} h, due west at {HOURS[np.argmin(np.abs(HA-90))]:.3f} h')

# ------------------------------------------------------------------------------------------------
# 2. Kinematics and the O1 inelastic rate over the sidereal day
# ------------------------------------------------------------------------------------------------
DDMAX_DV = math.sqrt(2 * lz.m_nucleus_gev(lz.A_XE_MEAN) * E_EV * 1e-6) / lz.C_KMS * 1e6   # keV per km/s
log(f'[2] d delta_max / d v_max = {DDMAX_DV:.4f} keV/(km/s) (P018: 0.8218)')
KIN = {}
for m in MASSES:
    dm = np.array([lz.delta_max_kev(E_EV, m, v_kms=s + VESC) for s in S_ROT])
    dm_ev = lz.delta_max_kev(E_EV, m, v_kms=S_EV + VESC); dm_mean = lz.delta_max_kev(E_EV, m, v_kms=S_ROT_MEAN + VESC)
    KIN[m] = dict(dmax_mean=dm_mean, dmax_event=dm_ev, dmax_min=float(dm.min()), dmax_max=float(dm.max()), p2p=float(dm.max() - dm.min()))
    log(f'    m = {m:.0f} GeV: delta_max(248) mean {dm_mean:.3f}, event {dm_ev:.3f} ({dm_ev-dm_mean:+.3f}), range {dm.min():.3f}..{dm.max():.3f} (p2p {dm.max()-dm.min():.3f} keV)')

WD = lz.wd()
HAM = lz.wd_hamiltonian('O1s', {1: (1.0, 0.0)})
XE_M = np.array(WD.Xe.mass); XE_AB = np.array(WD.Xe.abundance); XE_A = np.array(WD.Xe.a)
E_GRID = np.arange(60.0, 400.1, 2.0)
KER_FILE = f'{OUT}/P092_unit_stream_kernel.npz'
if os.path.exists(KER_FILE):
    C_A = np.load(KER_FILE)['C_A']; log(f'    loaded cached per-isotope response C_A(E) from {KER_FILE}')
else:
    t1 = time.time()
    V1 = np.array([2000.0]); ONE = np.array([1.0])
    C_A = np.zeros((len(XE_M), len(E_GRID)))
    for i in range(len(XE_M)):
        for j, e in enumerate(E_GRID):
            C_A[i, j] = WD.diff_rate(WD.Xe, HAM, M_CHI, float(e), V1, ONE, j_chi=0.5, delta=0.0, isotopes_list={0: [i]})
    C_A *= 1000.0 * 365.25       # events/(t yr keV) per unit eta (km/s)^-1
    np.savez(KER_FILE, C_A=C_A, E_GRID=E_GRID)
    log(f'    per-isotope unit-stream response computed: {C_A.shape} in {time.time()-t1:.1f} s')
# delta-independence check of the unit-stream response (O1 has no explicit v dependence)
chk = [WD.diff_rate(WD.Xe, HAM, M_CHI, e, np.array([2000.0]), np.array([1.0]), j_chi=0.5, delta=d) * 1000 * 365.25 / C_A[:, np.argmin(np.abs(E_GRID - e))].sum()
       for e in (150.0, 248.0, 300.0) for d in (0.0, 366.0)]
log(f'    unit-stream response ratio (all isotopes, delta 0/366 at E = 150/248/300) / cached sum: {np.round(chk, 6)}')

def vmin_A(E, mN, delta):
    mu = lz.mu_red(M_CHI, mN)
    return (mN * E * 1e-6 / mu + delta * 1e-6) / np.sqrt(2 * mN * E * 1e-6) * lz.C_KMS

def efficiency(E, edge=lz.LZ['E_50pct_high_keV'], width=15.0, plateau=lz.LZ['eff_plateau']):
    lo = 1.0 / (1.0 + np.exp(-(E - lz.LZ['E_50pct_low_keV']) / 1.0))
    hi = stats.norm.cdf((edge - E) / width)
    return plateau * lo * hi
EFF = efficiency(E_GRID)

def spectrum(delta, v_lab, vesc=VESC, v0=V0):
    """dR/dE (events/(t yr keV), unit WimPyDD coupling) for the Baxter halo boosted by speed v_lab."""
    dr = np.zeros_like(E_GRID)
    for i in range(len(XE_M)):
        vm = vmin_A(E_GRID, XE_M[i], delta)
        dr += C_A[i] * np.clip(lz.eta0(vm, v_e=v_lab, v0=v0, vesc=vesc), 0, None)
    return dr
def roi_rate(delta, v_lab, **kw):
    return float(np.trapezoid(spectrum(delta, v_lab, **kw) * EFF, E_GRID))

# validation against WimPyDD's own day-of-year halo (Earth velocity from WimPyDD's v_earth_sun) at the event day
DOY_EV = 167.8907
halo_day = lz.wd_halo(day_of_year=DOY_EV, vmin=np.linspace(0.0, 900.0, 1801))
v_wd_day = float(np.linalg.norm(V_SUN_GAL + np.array(WD.v_earth_sun((DOY_EV - 80.0) * 2 * math.pi / 365.0, v_rot_gal=np.array([0., V0, 0.]), v_sun_rot=lz.V_SUN_PEC))))
VAL = []
for d in DELTAS:
    Ecmp = np.arange(100.0, 340.1, 20.0)
    r_wd = lz.wd_rate(HAM, M_CHI, Ecmp, halo=halo_day, delta_kev=d)
    r_an = np.array([np.interp(e, E_GRID, spectrum(d, v_wd_day)) for e in Ecmp])
    N_wd = float(np.trapezoid(np.clip(np.array(lz.wd_rate(HAM, M_CHI, E_GRID, halo=halo_day, delta_kev=d)), 0, None) * EFF, E_GRID))
    N_an = roi_rate(d, v_wd_day)
    ok = r_wd > 0
    VAL.append(dict(delta=d, v_lab_wimpydd_day=v_wd_day, N_roi_wimpydd=N_wd, N_roi_analytic=N_an, ratio=N_an / N_wd,
                    spectrum_ratio_min=float((r_an[ok] / r_wd[ok]).min()), spectrum_ratio_max=float((r_an[ok] / r_wd[ok]).max())))
    log(f'    validation delta={d:.0f}: WimPyDD day-halo (v_lab {v_wd_day:.2f}) N_ROI = {N_wd:.4e}; analytic-eta method {N_an:.4e}; ratio {N_an/N_wd:.4f}; '
        f'spectrum ratio range {VAL[-1]["spectrum_ratio_min"]:.3f}-{VAL[-1]["spectrum_ratio_max"]:.3f}')
log(f'    (P034 annual-mean unit counts x 2.84: reference 1.24e13/2.39e11/1.95e10/6.7e8 per 2.84 t yr at 300/350/366/380 keV)')

# rate over the day: 5-min grid
STEP = 5
sel = np.arange(0, len(MIN), STEP)
H5 = HOURS[sel]
RATES_ROT = {d: np.array([roi_rate(d, s) for s in S_ROT[sel]]) for d in DELTAS}
RATES_FULL = {d: np.array([roi_rate(d, s) for s in S_FULL[sel]]) for d in DELTAS}
R_EV = {d: roi_rate(d, S_EV) for d in DELTAS}
R_GEO_EV = {d: roi_rate(d, S_GEO_EV) for d in DELTAS}
log(f'    rates on {len(sel)} time points done at {time.time()-T0:.1f} s')

def dlnR_dv(d, v, h=0.25):
    return (math.log(roi_rate(d, v + h)) - math.log(roi_rate(d, v - h))) / (2 * h)

def cos_fit(x_deg, y):
    """y = c0 + c1 cos(x) + s1 sin(x): fundamental amplitude and phase (degrees of the LST-like angle)."""
    X = np.c_[np.ones_like(x_deg), np.cos(np.radians(x_deg)), np.sin(np.radians(x_deg))]
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return c[0], math.hypot(c[1], c[2]), math.degrees(math.atan2(c[2], c[1])) % 360.0

MOD = {}
sid = MIN[sel] <= SID_MIN
for d in DELTAS:
    R = RATES_ROT[d]; Rs = R[sid]; t = MIN[sel][sid]
    mean = float(np.trapezoid(Rs, t) / (t[-1] - t[0]))
    c0, a1abs, ph = cos_fit(LST_GRID[sel][sid], Rs)
    a1 = a1abs / c0
    p = Rs / mean                               # normalised time PDF (T p(t) with T = 1 sidereal day)
    kl = float(np.trapezoid(p * np.log(np.clip(p, 1e-300, None)), t) / (t[-1] - t[0]))
    s = np.log(np.clip(p, 1e-300, None)); E_flat = float(np.trapezoid(s, t) / (t[-1] - t[0])); Var_flat = float(np.trapezoid((s - E_flat) ** 2, t) / (t[-1] - t[0]))
    N3_llr = 9.0 * Var_flat / (kl - E_flat) ** 2
    N3_cos = 18.0 / a1 ** 2
    lr_ev = R_EV[d] / mean
    lr_full = RATES_FULL[d][np.argmin(np.abs(H5 - HOURS[i_ev]))] / float(np.trapezoid(RATES_FULL[d], H5) / 24.0)
    MOD[d] = dict(mean_rate=mean, p2p_frac=float((Rs.max() - Rs.min()) / mean), a1=a1, phase_LST_deg_of_max=ph, UTC_of_max_h=float(H5[sid][np.argmax(Rs)]),
                  KL_nats=kl, N3_llr=N3_llr, N3_cos=N3_cos, LR_event=lr_ev, LR_event_fullseries=lr_full,
                  dlnR_dv_per_kms=dlnR_dv(d, S_ROT_MEAN), rate_event=R_EV[d], rate_event_over_geocentric=R_EV[d] / R_GEO_EV[d],
                  frac_of_day_zero=float(np.mean(Rs < 1e-6 * Rs.max())))
    log(f'    delta={d:.0f}: <R> {mean:.4e}/t/yr (unit c); p2p {MOD[d]["p2p_frac"]*100:.2f} %; a1 {a1*100:.3f} %; max at LST {ph:.1f} deg (UTC {MOD[d]["UTC_of_max_h"]:.2f} h); '
        f'dlnR/dv {MOD[d]["dlnR_dv_per_kms"]:.4f}/(km/s); KL {kl:.3e} nats; N3 LLR {N3_llr:.3e}, cos {N3_cos:.3e}; LR(event) {lr_ev:.4f} (full series {lr_full:.4f}); '
        f'R(event)/R(geocentric) {MOD[d]["rate_event_over_geocentric"]:.4f}')

# P034 annual comparison (their paper: a1 0.43/1.18/1.52/1.66; KL 0.047/0.39/0.72/0.90; N3 96.5/11.5/6.5/5.3 at 300/350/366/380)
P034 = {300.0: dict(a1=0.43, KL=0.047, N3=96.5), 350.0: dict(a1=1.18, KL=0.39, N3=11.5), 366.0: dict(a1=1.52, KL=0.72, N3=6.5), 380.0: dict(a1=1.66, KL=0.90, N3=5.3)}
for d in P034:
    MOD[d]['annual_P034'] = P034[d]
    MOD[d]['N3_ratio_diurnal_over_annual'] = MOD[d]['N3_llr'] / P034[d]['N3']
    MOD[d]['KL_ratio_annual_over_diurnal'] = P034[d]['KL'] / MOD[d]['KL_nats']
    log(f'    delta={d:.0f}: N3 diurnal/annual = {MOD[d]["N3_ratio_diurnal_over_annual"]:.0f}; KL annual/diurnal = {MOD[d]["KL_ratio_annual_over_diurnal"]:.0f}; '
        f'a1 annual/diurnal = {P034[d]["a1"]/MOD[d]["a1"]:.0f}')

# season dependence of the diurnal amplitude (same v_rot projection; different mean speed): 1 Sep (~annual mean), 1 Dec, 2 June
SEASON = {}
for label, v_mean in [('2 June (peak)', 266.44), ('16 June (event)', S_GEO_EV), ('1 Sep (mean)', 250.8), ('1 Dec (trough)', 237.4)]:
    row = {}
    for d in DELTAS:
        Rp, Rm = roi_rate(d, v_mean + A_ROT), roi_rate(d, v_mean - A_ROT)
        row[d] = float((Rp - Rm) / (Rp + Rm)) if Rp + Rm > 0 else float('nan')
    SEASON[label] = row
    log(f'    diurnal amplitude a1 ~ (R+ - R-)/(R+ + R-) on {label} (v {v_mean:.1f}): ' + ', '.join(f'{d:.0f}: {row[d]*100:.2f} %' for d in DELTAS))
# halo-parameter sensitivity of the diurnal amplitude at the event date (v_esc 528/560)
HALOVAR = {}
for vesc in (528.0, 560.0):
    HALOVAR[vesc] = {d: float((roi_rate(d, S_GEO_EV + A_ROT, vesc=vesc) - roi_rate(d, S_GEO_EV - A_ROT, vesc=vesc)) / max(roi_rate(d, S_GEO_EV + A_ROT, vesc=vesc) + roi_rate(d, S_GEO_EV - A_ROT, vesc=vesc), 1e-300)) for d in DELTAS}
    log(f'    v_esc = {vesc:.0f}: diurnal a1 ' + ', '.join(f'{d:.0f}: {HALOVAR[vesc][d]*100:.2f} %' for d in DELTAS))

# ------------------------------------------------------------------------------------------------
# 5. Gravitational focusing and Earth shadowing (order of magnitude)
# ------------------------------------------------------------------------------------------------
GRAV = {}
for v in (700.0, 750.0, 810.0):
    dv_E = VESC_EARTH ** 2 / (2 * v); dv_S = VESC_SUN_1AU ** 2 / (2 * v)
    GRAV[v] = dict(earth_blueshift_kms=dv_E, earth_blueshift_frac=dv_E / v, earth_focusing_density_anisotropy=(VESC_EARTH / v) ** 2,
                   sun_blueshift_kms=dv_S, sun_focusing_density=(VESC_SUN_1AU / v) ** 2,
                   sun_focusing_daily_change=(VESC_SUN_1AU / v) ** 2 * 2 * R_E / 1.496e8,
                   delta_max_shift_earth_keV=DDMAX_DV * dv_E, delta_max_shift_sun_keV=DDMAX_DV * dv_S)
    log(f'[5] v = {v:.0f} km/s: Earth blueshift {dv_E:.4f} km/s ({dv_E/v:.1e}; delta_max +{DDMAX_DV*dv_E:.3f} keV, time-constant); Earth focusing anisotropy (v_esc/v)^2 = {(VESC_EARTH/v)**2:.1e}; '
        f'Sun blueshift {dv_S:.3f} km/s (delta_max +{DDMAX_DV*dv_S:.2f} keV, annual, not daily); Sun focusing (v_esc,1AU/v)^2 = {(VESC_SUN_1AU/v)**2:.1e}, its daily change {(VESC_SUN_1AU/v)**2*2*R_E/1.496e8:.1e}')
rec('Sun gravitational focusing of DM changes annual-modulation phase mainly at low v_min (Lee, Lisanti, Peter, Safdi 2014)', 'qualitative', 'PRL 112, 011301', 'likely')

# Earth shadowing: (a) kinematic closure of the inelastic channel in Earth nuclei
EARTH_NUC = rec('Earth bulk composition by mass: O 0.30, Fe 0.32, Si 0.15, Mg 0.14, Ni 0.018, Ca 0.015, Al 0.014, S 0.029; mean density 5.51 g/cm3',
                {'O': (16, 0.30), 'Fe': (56, 0.32), 'Si': (28, 0.15), 'Mg': (24, 0.14), 'Ni': (58, 0.018), 'Ca': (40, 0.015), 'Al': (27, 0.014), 'S': (32, 0.029)}, 'geochemistry (McDonough & Sun 1995-type)', 'likely')
RHO_EARTH = 5.51
V_MAX = S_GEO_EV + VESC
CEIL = {}
for el, (A, f) in EARTH_NUC.items():
    mN = lz.m_nucleus_gev(A); mu = lz.mu_red(M_CHI, mN)
    CEIL[el] = dict(A=A, delta_ceiling_keV=mu * (V_MAX / lz.C_KMS) ** 2 / 2 * 1e6, vmin_delta300_kms=math.sqrt(2 * 300e-6 / mu) * lz.C_KMS)
log('    inelastic ceiling delta_max = mu v_max^2/2 in Earth nuclei (1 TeV, v_max %.1f): ' % V_MAX + ', '.join(f'{el} {CEIL[el]["delta_ceiling_keV"]:.0f} keV' for el in CEIL)
    + '; v_min for delta = 300 keV: ' + ', '.join(f'{el} {CEIL[el]["vmin_delta300_kms"]:.0f}' for el in ('O', 'Si', 'Fe', 'Ni')) + ' km/s')
# (b) elastic mean free path for a heavy WIMP with coherent sigma_A = sigma_n A^2 (mu_A/mu_n)^2 ~ sigma_n A^4 (F^2 = 1, conservative)
M_U_G = 1.6605e-24
SUM_FA3 = sum(f * A ** 3 for A, f in EARTH_NUC.values())
LAMBDA_EARTH_KM_AT_1E38 = 1.0 / (1e-38 * RHO_EARTH / M_U_G * SUM_FA3) / 1e5
ROCK = rec('crust rock: density 2.7 g/cm3; by mass O 0.47, Si 0.28, Al 0.08, Fe 0.05, Ca 0.04, Mg 0.02, K+Na 0.05', {'O': (16, 0.47), 'Si': (28, 0.28), 'Al': (27, 0.08), 'Fe': (56, 0.05), 'Ca': (40, 0.04), 'Mg': (24, 0.02), 'K': (39, 0.05)}, 'geochemistry', 'likely')
SUM_FA3_ROCK = sum(f * A ** 3 for A, f in ROCK.values())
LAMBDA_ROCK_KM_AT_1E38 = 1.0 / (1e-38 * 2.7 / M_U_G * SUM_FA3_ROCK) / 1e5
SIGMA_OVERBURDEN = 1e-38 * LAMBDA_ROCK_KM_AT_1E38 / DEPTH_KM       # sigma_n for which lambda_rock = overburden depth
# one-event inelastic cross-sections (unit WimPyDD c^0 = 1 GeV^-2: sigma_n = (c0/2)^2 mu_n^2/pi)
mu_n = lz.mu_red(M_CHI, lz.M_NUCLEON_GEV)
SIGMA_UNIT = 0.25 * mu_n ** 2 / math.pi * lz.GEV_TO_CM2
SIG1 = {}
for d in DELTAS:
    N_unit = MOD[d]['mean_rate'] * lz.LZ['exposure_tyr']      # unit-coupling events in 2.84 t yr at the 16 June halo (upper end of the year)
    SIG1[d] = dict(N_unit_16June_2p84tyr=N_unit, sigma_n_one_event_cm2=SIGMA_UNIT / N_unit)
    lam = LAMBDA_EARTH_KM_AT_1E38 * 1e-38 / SIG1[d]['sigma_n_one_event_cm2']
    SIG1[d]['lambda_earth_km_if_elastic'] = lam
    log(f'    delta={d:.0f}: unit-coupling events (16 June halo) {N_unit:.3e} -> one-event sigma_n = {SIG1[d]["sigma_n_one_event_cm2"]:.2e} cm2; if the same coupling were ELASTIC, lambda_Earth = {lam:.3e} km')
# path length through the Earth toward the apex vs UTC (detector at R - depth; direction at altitude alt)
r_d = R_E - DEPTH_KM
def path_km(alt_deg):
    a = np.radians(alt_deg)
    return -r_d * np.sin(a) + np.sqrt((r_d * np.sin(a)) ** 2 + R_E ** 2 - r_d ** 2)
PATH = path_km(ALT)
log(f'    Earth path toward the apex: at the event {path_km(alt_ev):.1f} km; min {PATH.min():.2f} km (zenith), horizon {path_km(0.0):.1f} km, max {PATH.max():.1f} km (alt {ALT.min():+.2f}); '
    f'lambda_Earth (coherent, F^2 = 1) = {LAMBDA_EARTH_KM_AT_1E38:.0f} km at sigma_n = 1e-38 cm2; overburden stops sigma_n > {SIGMA_OVERBURDEN:.1e} cm2')
SHADOW = {s: float(PATH.max() / (LAMBDA_EARTH_KM_AT_1E38 * 1e-38 / s)) for s in (1e-38, 1e-40, 1e-42, 1e-45)}
log('    diurnal shadow amplitude ~ L_max/lambda for an ELASTIC sigma_n: ' + ', '.join(f'{s:.0e}: {v:.1e}' for s, v in SHADOW.items()))
LZ_EL_LIMIT = rec('LZ elastic SI limit at 1 TeV ~ 1e-45 cm2 (2024 WS result scaled ~ m_chi from ~2e-47 at 40 GeV)', 1e-45, 'LZ 2024 WS paper', 'likely')
SIG_EL_1E4 = 1e-4 * LAMBDA_EARTH_KM_AT_1E38 * 1e-38 / PATH.max()
log(f'    an elastic component gives a shadow < 1e-4 for sigma_el < {SIG_EL_1E4:.1e} cm2; LZ\'s own elastic limit (~{LZ_EL_LIMIT:.0e}) makes it < {SHADOW[1e-45]:.0e}')

# ------------------------------------------------------------------------------------------------
# 6. Tables, figures, JSON
# ------------------------------------------------------------------------------------------------
import csv
with open(f'{OUT}/P092_hourly_inputs.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['UTC_h', 'GMST_h', 'LST_SURF_h', 'apex_HA_h', 'apex_alt_deg', 'apex_az_deg', 'v_rot_proj_kms', 'v_lab_rot_only_kms', 'v_lab_full_kms',
                'v_lab_minus_sidereal_mean_kms', 'delta_max_248_1TeV_keV', 'Earth_path_km'] + [f'R_rel_{d:.0f}' for d in DELTAS])
    rows_t = list(range(0, 1441, 60)) + [i_ev]
    for i in sorted(rows_t):
        t = T_GRID[i]; k = np.argmin(np.abs(H5 - HOURS[i]))
        w.writerow([f'{HOURS[i]:.4f}', f'{float(gmst_h(t)):.4f}', f'{LST_GRID[i]/15:.4f}', f'{HA[i]/15:+.3f}', f'{ALT[i]:+.2f}', f'{AZ[i]:.1f}',
                    f'{float(np.dot(V_ROT_GRID[:, i], apex_gal)):+.4f}', f'{S_ROT[i]:.4f}', f'{S_FULL[i]:.4f}', f'{S_ROT[i]-S_ROT_MEAN:+.4f}',
                    f'{lz.delta_max_kev(E_EV, M_CHI, v_kms=S_ROT[i]+VESC):.3f}', f'{PATH[i]:.1f}'] + [f'{RATES_ROT[d][k]/MOD[d]["mean_rate"]:.4f}' for d in DELTAS])
with open(f'{OUT}/P092_modulation_summary.csv', 'w', newline='') as f:
    w = csv.writer(f)
    keys = ['mean_rate', 'p2p_frac', 'a1', 'phase_LST_deg_of_max', 'UTC_of_max_h', 'dlnR_dv_per_kms', 'KL_nats', 'N3_llr', 'N3_cos', 'LR_event', 'LR_event_fullseries', 'rate_event_over_geocentric', 'N3_ratio_diurnal_over_annual', 'KL_ratio_annual_over_diurnal']
    w.writerow(['delta_keV'] + keys)
    for d in DELTAS:
        w.writerow([f'{d:.0f}'] + [f'{MOD[d].get(k, float("nan")):.5g}' for k in keys])
np.savetxt(f'{OUT}/P092_vlab_apex_vs_UTC.csv', np.c_[H5, S_ROT[sel], S_FULL[sel], ALT[sel], AZ[sel]], delimiter=',',
           header='UTC_h,v_lab_rot_only_kms,v_lab_full_kms,apex_alt_deg,apex_az_deg', comments='')
np.savetxt(f'{OUT}/P092_rates_rel_vs_UTC.csv', np.array([H5] + [RATES_ROT[d] / MOD[d]['mean_rate'] for d in DELTAS] + [RATES_FULL[d] / MOD[d]['mean_rate'] for d in DELTAS]).T,
           delimiter=',', header='UTC_h,' + ','.join(f'Rrot_{d:.0f}' for d in DELTAS) + ',' + ','.join(f'Rfull_{d:.0f}' for d in DELTAS), comments='')

COL = {'blue': '#3B6FB6', 'red': '#C8443F', 'green': '#3E8E5E', 'orange': '#D98A2B', 'purple': '#7D5BA6', 'grey': '#6E6E6E'}
fig, axs = plt.subplots(2, 1, figsize=(7.5, 7.0), sharex=True)
axs[0].plot(HOURS, S_ROT - S_ROT_MEAN, color=COL['blue'], lw=2, label='Earth rotation only (orbital velocity frozen)')
axs[0].plot(HOURS, S_FULL - S_ROT_MEAN, color=COL['orange'], lw=1.2, ls='--', label='rotation + orbital drift over the day')
axs[0].axvline(HOURS[i_ev], color=COL['red'], lw=1); axs[0].plot(HOURS[i_ev], S_EV - S_ROT_MEAN, 'o', color=COL['red'], label=f'event 21:22:39 UTC ({S_EV - S_ROT_MEAN:+.3f} km/s)')
axs[0].axhline(0, color=COL['grey'], lw=0.8, ls=':'); axs[0].set_ylabel('|v_lab| - sidereal mean  (km/s)'); axs[0].legend(fontsize=8, loc='lower left')
axs[0].set_title(f'SURF lab speed relative to the halo, 16 June 2023 (mean {S_ROT_MEAN:.2f} km/s; amplitude {A_ROT:.3f} km/s)', fontsize=10)
ax2 = axs[0].twinx(); ax2.set_ylim(np.array(axs[0].get_ylim()) * DDMAX_DV); ax2.set_ylabel('delta_max(248 keV) shift  (keV)')
axs[1].plot(HOURS, ALT, color=COL['green'], lw=2, label='apex altitude'); axs[1].axhline(0, color=COL['grey'], lw=0.8, ls=':')
axs[1].axvline(HOURS[i_ev], color=COL['red'], lw=1); axs[1].plot(HOURS[i_ev], alt_ev, 'o', color=COL['red'])
axs[1].set_ylabel('DM-wind apex altitude at SURF (deg)'); axs[1].set_xlabel('UTC hour, 16 June 2023'); axs[1].set_xlim(0, 24); axs[1].set_xticks(range(0, 25, 3))
ax3 = axs[1].twinx(); ax3.plot(HOURS, AZ, color=COL['purple'], lw=1, ls='--', label='apex azimuth'); ax3.set_ylabel('apex azimuth (deg, N through E)')
axs[1].legend(loc='upper left', fontsize=8); ax3.legend(loc='upper right', fontsize=8)
fig.tight_layout(); fig.savefig(f'{FIG}/fig1_vlab_apex_vs_UTC.png', dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 4.6))
cols = [COL['blue'], COL['green'], COL['orange'], COL['red'], COL['purple']]
for d, c in zip(DELTAS, cols):
    ax.plot(H5, 100 * (RATES_ROT[d] / MOD[d]['mean_rate'] - 1), color=c, lw=2, ls='-' if d != 385 else ':', label=f'delta = {d:.0f} keV (a1 = {MOD[d]["a1"]*100:.2f} %)')
ax.axvline(HOURS[i_ev], color='k', lw=1, ls='--', label='event 21:22:39 UTC'); ax.axhline(0, color=COL['grey'], lw=0.8, ls=':')
ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 3)); ax.set_xlabel('UTC hour, 16 June 2023'); ax.set_ylabel('in-ROI inelastic rate: deviation from sidereal mean (%)')
ax.set_title('Diurnal (sidereal) modulation of the O1 inelastic rate at LZ, 1 TeV, Earth rotation only', fontsize=10); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f'{FIG}/fig2_rate_modulation_vs_UTC.png', dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 4.2))
dd = [d for d in DELTAS if d in P034]
ax.semilogy(dd, [MOD[d]['N3_llr'] for d in dd], 'o-', color=COL['red'], label='diurnal (this work, 16 June amplitude)')
ax.semilogy(dd, [P034[d]['N3'] for d in dd], 's-', color=COL['blue'], label='annual (P034, likelihood ratio)')
ax.set_xlabel('delta (keV)'); ax.set_ylabel('events for 3 sigma vs time-flat'); ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')
ax.set_title('Events needed to establish the modulation at 3 sigma', fontsize=10)
fig.tight_layout(); fig.savefig(f'{FIG}/fig3_N3sigma_diurnal_vs_annual.png', dpi=150); plt.close(fig)

RES = dict(
    inputs=dict(v_sun_gal=V_SUN_GAL.tolist(), v0=V0, vesc=VESC, m_chi=M_CHI, deltas=DELTAS, event_utc=EV_UTC, lat=LAT, lon=LON, depth_km=DEPTH_KM,
                v_rot_kms=V_ROT, E_grid=[float(E_GRID[0]), float(E_GRID[-1]), 2.0], efficiency='P034 parametrisation of Fig. S2 (edge 269.9, width 15, plateau 0.96)'),
    frames=dict(v_orb_event_gal=V_ORB_EV.tolist(), v_lab_geocentric_event=V_LAB_GEO_EV.tolist(), s_geocentric_event=S_GEO_EV, s_wimprates_event=S_WR_EV,
                s_P055_event=P055_VE, astropy_minus_wimprates_vector_kms=float(np.linalg.norm(V_LAB_GEO_EV - V_WR_EV)), lzcommon_cosine=lz.v_earth_kms(167.89),
                ephemeris_check=EPH, astropy_minus_wimprates_apex_projection_kms=float(np.dot(DIFF_WR, V_LAB_GEO_EV / np.linalg.norm(V_LAB_GEO_EV))),
                apex_ra_dec=[RA_AP, DEC_AP], diurnal_amplitude_kms=A_ROT, s_rot_mean=S_ROT_MEAN, s_rot_min=float(S_ROT.min()), s_rot_max=float(S_ROT.max()),
                s_rot_p2p=float(S_ROT.max() - S_ROT.min()), utc_of_max_speed_h=float(HOURS[np.argmax(S_ROT[msk])]), utc_of_min_speed_h=float(HOURS[np.argmin(S_ROT[msk])]),
                s_event=S_EV, s_event_minus_mean=S_EV - S_ROT_MEAN, s_event_minus_geocentric=S_EV - S_GEO_EV, rot_projection_event=float(np.dot(v_rot_gal(T_EV), apex_gal)),
                orbital_drift_per_day_kms=float(S_GEO[-1] - S_GEO[0]), s_full_mean=S_FULL_MEAN, sidereal_mean_minus_geocentric=S_ROT_MEAN - S_GEO_EV,
                lst_event_h=LST_EV / 15, apex_ha_event_h=ha_ev / 15, apex_alt_event=float(alt_ev), apex_az_event=float(az_ev), apex_alt_min=float(ALT.min()), apex_alt_max=float(ALT.max()),
                frac_day_apex_below_horizon=float(np.mean(ALT < 0)), utc_upper_culmination_h=float(HOURS[np.argmax(ALT)]), utc_lower_culmination_h=float(HOURS[np.argmin(ALT)]),
                utc_apex_east_h=float(HOURS[np.argmin(np.abs(HA + 90))]), utc_apex_west_h=float(HOURS[np.argmin(np.abs(HA - 90))])),
    kinematics=dict(ddmax_dv_keV_per_kms=DDMAX_DV, per_mass={str(int(m)): KIN[m] for m in MASSES}),
    validation=VAL, modulation={str(int(d)): MOD[d] for d in DELTAS}, season_amplitude=SEASON, halo_variants={str(int(k)): v for k, v in HALOVAR.items()},
    gravity=GRAV, earth_ceilings=CEIL, shadowing=dict(lambda_earth_km_at_1e38=LAMBDA_EARTH_KM_AT_1E38, lambda_rock_km_at_1e38=LAMBDA_ROCK_KM_AT_1E38, sum_fA3_earth=SUM_FA3,
                                                     sigma_stopped_by_overburden=SIGMA_OVERBURDEN, path_event_km=float(path_km(alt_ev)), path_min_km=float(PATH.min()), path_horizon_km=float(path_km(0.0)),
                                                     path_max_km=float(PATH.max()), shadow_amplitude_if_elastic=SHADOW, sigma_el_for_1e4_shadow=SIG_EL_1E4, one_event_sigma=SIG1, sigma_unit_coupling=SIGMA_UNIT),
    recalled=RECALLED, runtime_s=time.time() - T0)
def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return o
json.dump(clean(RES), open(f'{OUT}/P092_results.json', 'w'), indent=1)
log(f'done in {time.time()-T0:.1f} s')
