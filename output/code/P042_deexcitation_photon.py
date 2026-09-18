#!/usr/bin/env python
"""P042 -- The de-excitation photon: what chi2 must do after scattering in LZ.

Run from the simulation root:  .venv/bin/python output/code/P042_deexcitation_photon.py

Parts
  1. Kinematics of the up-scattered chi2 (speed after the scatter, direction distribution from the
     Baxter-2021 SHM boosted to the lab on 16 June 2023 21:22:39 UTC, projected onto the SURF vertical).
  2. Simplified LZ geometry (TPC, RFR, PTFE, Skin, Ti vessels, GdLS OD, water) and a Woodcock-tracking
     photon Monte Carlo (photoabsorption + Klein-Nishina Compton) for the de-excitation photon.
  3. Topology probabilities vs lifetime tau: second site in the TPC, S1-only deposit in the RFR,
     Skin/OD prompt veto, Skin/OD delayed veto, clean (escape or sub-threshold).
  4. Single-event likelihood P_clean(tau) -> 90% CL lower limit on tau, BR_gamma limit, mapping to
     mu_tr(delta) with P023's LZ-rate moments, model points (MiDM, Higgsino, dark photon, L10 loop).
  5. Population predictions for N future events.
Outputs in output/work/P042/ (CSV/JSON) and output/work/P042/figures/ (PNG).
"""
import sys, os, json, math, time
sys.path.insert(0, 'output/code')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import lzcommon as lz

T0 = time.time()
OUT = 'output/work/P042'; FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)
LOG = open(os.path.join(OUT, 'P042_run.log'), 'w')
def log(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.write(s + '\n'); LOG.flush()

RECALLED = []
def rec(item, value, source, reliability):
    RECALLED.append(dict(item=item, value=str(value), presumed_source=source, reliability=reliability)); return value

rng = np.random.default_rng(42)

# ----------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------
C_KMS = lz.C_KMS
HBAR_GEV_S = rec('hbar = 6.582e-25 GeV s', 6.582119569e-25, 'PDG', 'certain')
ALPHA = rec('alpha = 1/137.036', 1 / 137.035999, 'PDG', 'certain')
E_CHARGE = math.sqrt(4 * math.pi * ALPHA)
M_P = lz.M_NUCLEON_GEV
MU_N_GEVINV = E_CHARGE / (2 * M_P)             # nuclear magneton in GeV^-1 (natural units)
M_E_KEV = rec('m_e = 510.999 keV', 510.999, 'PDG', 'certain')
M_CHI = 1000.0                                  # GeV, LZ reference mass
E_R = lz.LZ['ev_E_keV']                         # 248 keV
M_XE = lz.m_nucleus_gev(lz.A_XE_MEAN)           # 122.3 GeV
Q_GEV = math.sqrt(2 * M_XE * E_R * 1e-6)        # momentum transfer, 0.2463 GeV

# ----------------------------------------------------------------------------------------------
# Part 1: chi2 kinematics and direction distribution
# ----------------------------------------------------------------------------------------------
# Lab velocity in the Galactic frame on 16 June 2023 21:22:39 UTC
from astropy.utils import iers
iers.conf.auto_download = False
from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel, SkyCoord, CartesianRepresentation
import astropy.units as u
T_EV = Time(lz.LZ['ev_time_utc'], scale='utc')
_, vel = get_body_barycentric_posvel('earth', T_EV)
v_earth_icrs = vel.xyz.to(u.km / u.s).value
def icrs_to_gal(vec):
    c = SkyCoord(CartesianRepresentation(vec[0], vec[1], vec[2]), frame='icrs')
    g = c.galactic.cartesian.xyz.value
    return np.array(g)
def gal_to_icrs(vecs):
    c = SkyCoord(CartesianRepresentation(vecs[:, 0], vecs[:, 1], vecs[:, 2]), frame='galactic')
    i = c.icrs
    return np.asarray(i.ra.deg), np.asarray(i.dec.deg)
v_earth_gal = icrs_to_gal(v_earth_icrs)
V_LAB_GAL = np.array([0.0, lz.V0_KMS, 0.0]) + lz.V_SUN_PEC + v_earth_gal
log(f'Earth barycentric velocity (ICRS) {v_earth_icrs.round(2)} km/s; Galactic {v_earth_gal.round(2)}; '
    f'|v_lab,gal| = {np.linalg.norm(V_LAB_GAL):.1f} km/s (lzcommon cosine model {lz.v_earth_kms(167):.1f})')

# SURF horizon: altitude of a direction (RA, Dec) at the event time, from GMST (UT1 ~ UTC to < 1 s)
LAT = rec('SURF Davis campus latitude 44.352 N, longitude 103.751 W', 44.352, 'geography', 'likely')
LON = -103.751
D = T_EV.jd - 2451545.0
GMST_H = (18.697374558 + 24.06570982441908 * D) % 24.0        # recalled formula (USNO), certain
LST_DEG = (GMST_H * 15.0 + LON) % 360.0
rec('GMST = 18.697374558 + 24.06570982441908 D hours', 'formula', 'USNO approximate sidereal time', 'certain')
def altitude_deg(ra_deg, dec_deg):
    ha = np.radians(LST_DEG - ra_deg)
    lat = math.radians(LAT); dec = np.radians(dec_deg)
    return np.degrees(np.arcsin(np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(ha)))
ra_ap, dec_ap = gal_to_icrs(-V_LAB_GAL[None, :])     # direction fast DM *moves toward* (anti-apex)
alt_antiapex = float(altitude_deg(ra_ap, dec_ap)[0])
log(f'LST = {LST_DEG/15:.2f} h; DM-wind anti-apex RA {ra_ap[0]:.1f} Dec {dec_ap[0]:.1f}: altitude {alt_antiapex:+.1f} deg '
    f'(positive: fast DM moves upward through the detector)')

def sample_chi2(delta_kev, n, wind=True):
    """Sample chi2 velocities (km/s) after the up-scatter chi1 Xe -> chi2 Xe with E_R = 248 keV.
    Returns (speed km/s, cos(zenith) of the chi2 direction (+1 = up), weight) with rate weights 1/u."""
    vmin = lz.vmin_kms(E_R, M_CHI, delta_kev=delta_kev)
    vmax = lz.VESC_KMS + np.linalg.norm(V_LAB_GAL)
    if vmin >= vmax:
        return None
    # pure-numpy rejection sampling of the halo tail first (acceptance can be < 1e-4 near delta_max)
    acc = []
    got = 0
    while got < n:
        v = rng.normal(0, lz.V0_KMS / math.sqrt(2), size=(2_000_000, 3))
        v = v[np.einsum('ij,ij->i', v, v) < lz.VESC_KMS ** 2]
        uvec = v - V_LAB_GAL                       # DM velocity in the lab (Galactic axes)
        sel = np.einsum('ij,ij->i', uvec, uvec) >= vmin ** 2
        acc.append(uvec[sel]); got += sel.sum()
    uvec = np.concatenate(acc)[:n]
    out_v, out_cz, out_w = [], [], []
    for _ in range(1):
        uu = np.linalg.norm(uvec, axis=1)
        # momentum transfer direction: cos(theta_q) fixed by energy conservation, azimuth uniform
        Etot = (E_R + delta_kev) * 1e-6 + Q_GEV ** 2 / (2 * M_CHI)          # GeV
        cosq = Etot / (Q_GEV * uu / C_KMS)
        ok = np.abs(cosq) <= 1
        uvec, uu, cosq = uvec[ok], uu[ok], cosq[ok]
        uhat = uvec / uu[:, None]
        # orthonormal frame
        a = np.where(np.abs(uhat[:, 0:1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
        e1 = np.cross(uhat, a); e1 /= np.linalg.norm(e1, axis=1)[:, None]
        e2 = np.cross(uhat, e1)
        phi = rng.uniform(0, 2 * np.pi, uu.size)
        sinq = np.sqrt(1 - cosq ** 2)
        qhat = cosq[:, None] * uhat + sinq[:, None] * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2)
        v2 = uvec - (Q_GEV / M_CHI) * C_KMS * qhat                          # km/s, Galactic axes
        s2 = np.linalg.norm(v2, axis=1)
        if wind:
            ra, dec = gal_to_icrs(v2 / s2[:, None])
            cz = np.sin(np.radians(altitude_deg(ra, dec)))
        else:
            cz = rng.uniform(-1, 1, s2.size)
        out_v.append(s2); out_cz.append(cz); out_w.append(1.0 / uu)
    v = np.concatenate(out_v); cz = np.concatenate(out_cz); w = np.concatenate(out_w)
    return v, cz, w / w.sum()

kin_rows = []
for d in [250, 300, 350, 380]:
    v, cz, w = sample_chi2(d, 20000, wind=True)
    vmin = lz.vmin_kms(E_R, M_CHI, delta_kev=d)
    v1 = np.sqrt(v ** 2 + 2 * (E_R + d) * 1e-6 / M_CHI * C_KMS ** 2)
    kin_rows.append(dict(delta_keV=d, vmin_kms=vmin, v1_mean_kms=np.average(v1, weights=w), v2_mean_kms=np.average(v, weights=w),
                         v2_min_kms=v.min(), v2_max_kms=v.max(), speed_loss_frac=1 - np.average(v / v1, weights=w),
                         cos_zenith_mean=np.average(cz, weights=w), frac_upward=np.sum(w[cz > 0]),
                         cz_p10=np.quantile(cz, 0.1), cz_p90=np.quantile(cz, 0.9)))
    log(f'delta {d}: v_min {vmin:.0f}, <v1> {kin_rows[-1]["v1_mean_kms"]:.0f}, <v2> {kin_rows[-1]["v2_mean_kms"]:.0f} km/s '
        f'(range {v.min():.0f}-{v.max():.0f}), speed loss {100*kin_rows[-1]["speed_loss_frac"]:.1f}%, <cos zenith> {kin_rows[-1]["cos_zenith_mean"]:+.2f}, '
        f'upward fraction {kin_rows[-1]["frac_upward"]:.2f}')
pd.DataFrame(kin_rows).to_csv(os.path.join(OUT, 'P042_kinematics.csv'), index=False)

# ----------------------------------------------------------------------------------------------
# Part 2: geometry and photon transport
# ----------------------------------------------------------------------------------------------
# Geometry (cm; z = 0 at the cathode, axis vertical). Paper: RFR depth 13.75 cm; gate 145.6 cm above cathode
# [recalled/likely: LZ drift length 145.6 cm, active radius 72.8 cm]. Everything else recalled/uncertain.
R_TPC = rec('LZ active radius 72.8 cm', 72.8, 'LZ instrument paper', 'likely')
H_TPC = rec('LZ drift length (cathode-gate) 145.6 cm', 145.6, 'LZ instrument paper', 'likely')
Z_RFR = 13.75                                                   # paper (MSSI paragraph)
T_PTFE = rec('PTFE wall + field cage radial thickness ~3 cm', 3.0, 'LZ TDR (order of magnitude)', 'uncertain')
T_SKIN = rec('side Skin LXe thickness 4-8 cm (6 cm used)', 6.0, 'LZ instrument paper', 'uncertain')
Z_ARRAY = 15.0                                                  # PMT array + structure thickness (uncertain)
Z_SKIN_BOT = rec('bottom Skin/dome LXe ~25 cm below the bottom array', 25.0, 'LZ instrument paper', 'uncertain')
T_TI = rec('cryostat vessel walls: Ti, ~0.8 cm each, ~7 cm vacuum gap', 0.8, 'LZ instrument paper', 'uncertain')
GAP_VAC = 7.0
T_GDLS = rec('OD GdLS acrylic tanks ~61 cm thick (side), ~60 cm top/bottom', 61.0, 'LZ OD papers', 'likely')
R_WATER = rec('water tank radius 3.8 m', 380.0, 'LZ instrument paper', 'likely')
VERTEX = np.array([45.9, 0.0, 26.4])                            # paper (Fig. 3 caption)

R1 = R_TPC + T_PTFE                  # Skin inner radius
R2 = R1 + T_SKIN                     # Skin outer radius = ICV inner wall
R3 = R2 + T_TI                       # ICV outer
R4 = R3 + GAP_VAC                    # OCV inner
R5 = R4 + T_TI                       # OCV outer
R6 = R5 + 4.0                        # water gap before GdLS
R7 = R6 + T_GDLS                     # GdLS outer
Z_BOT_ARRAY = -Z_RFR - Z_ARRAY       # bottom of bottom array
Z_SKIN_LO = Z_BOT_ARRAY - Z_SKIN_BOT
Z_LIQ = H_TPC + 0.5                  # liquid surface 0.5 cm above gate
Z_TOP_ARRAY = Z_LIQ + 1.0 + Z_ARRAY  # gas gap 1 cm then array
Z_ICV_LO, Z_ICV_HI = Z_SKIN_LO - T_TI, Z_TOP_ARRAY + 10.0
Z_OCV_LO, Z_OCV_HI = Z_ICV_LO - GAP_VAC - T_TI, Z_ICV_HI + GAP_VAC + T_TI
Z_GD_LO, Z_GD_HI = Z_OCV_LO - 4.0 - 60.0, Z_OCV_HI + 4.0 + 60.0
Z_WORLD_LO, Z_WORLD_HI = -300.0, 320.0
# material codes
VAC, TPC, RFR, PTFE, SKIN, TI, GDLS, WATER, STRUCT, OUT_ = range(10)
def material(p):
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    r = np.hypot(x, y)
    m = np.full(r.shape, WATER, dtype=np.int8)
    m[(r >= R_WATER) | (z <= Z_WORLD_LO) | (z >= Z_WORLD_HI)] = OUT_
    inside_r7 = r < R7
    gd = inside_r7 & (((r >= R6) & (z > Z_GD_LO) & (z < Z_GD_HI)) | ((z > Z_OCV_HI + 4.0) & (z < Z_GD_HI)) | ((z < Z_OCV_LO - 4.0) & (z > Z_GD_LO)))
    m[gd] = GDLS
    ocv = (r < R5) & (z > Z_OCV_LO) & (z < Z_OCV_HI)
    m[ocv & ((r >= R4) | (z < Z_OCV_LO + T_TI) | (z > Z_OCV_HI - T_TI))] = TI
    m[ocv & (r < R4) & (z >= Z_OCV_LO + T_TI) & (z <= Z_OCV_HI - T_TI)] = VAC
    icv = (r < R3) & (z > Z_ICV_LO) & (z < Z_ICV_HI)
    m[icv & ((r >= R2) | (z < Z_ICV_LO + T_TI))] = TI
    inner = icv & (r < R2) & (z >= Z_ICV_LO + T_TI)
    m[inner] = VAC                                                    # gas region default
    m[inner & (z < Z_LIQ) & (z >= Z_SKIN_LO) & ((r >= R1) | (z < Z_BOT_ARRAY))] = SKIN
    m[inner & (r < R1) & (r >= R_TPC) & (z >= Z_BOT_ARRAY) & (z < Z_TOP_ARRAY)] = PTFE
    m[inner & (r < R_TPC) & (z >= Z_BOT_ARRAY) & (z < -Z_RFR)] = STRUCT
    m[inner & (r < R_TPC) & (z >= -Z_RFR) & (z < 0)] = RFR
    m[inner & (r < R_TPC) & (z >= 0) & (z < Z_LIQ)] = TPC
    m[inner & (r < R_TPC) & (z >= Z_LIQ + 1.0) & (z < Z_TOP_ARRAY)] = STRUCT
    return m

# Photon attenuation: mass attenuation coefficients (cm^2/g, coherent excluded) and photoelectric fraction.
# Xenon anchored on P004/P029 values (0.3 cm at 122 keV; 0.62/0.99/1.42/2.9 cm at 164/203/243/375 keV) and
# recalled XCOM-like values elsewhere [likely, +-20%].  Light materials: Compton-dominated [likely, +-10%].
E_TAB = np.array([10, 20, 30, 34.5, 35, 40, 50, 60, 80, 100, 122, 164, 203, 243, 300, 375, 400], float)
MU_XE = np.array([150, 25, 8.5, 6.0, 28, 20, 11, 7.0, 3.3, 1.9, 1.16, 0.56, 0.353, 0.246, 0.152, 0.121, 0.098])
FPE_XE = np.array([.99, .98, .97, .96, .99, .98, .97, .96, .93, .90, .88, .80, .72, .62, .50, .40, .36])
E_LT = np.array([10, 20, 30, 50, 100, 200, 300, 400], float)
MU_H2O = np.array([5.3, 0.81, 0.376, 0.227, 0.171, 0.137, 0.118, 0.106]); FPE_H2O = np.array([.95, .7, .4, .1, .02, .005, .002, .001])
MU_LS = np.array([3.5, 0.62, 0.32, 0.215, 0.165, 0.133, 0.115, 0.103]); FPE_LS = FPE_H2O
MU_PTFE = np.array([7.0, 1.0, 0.40, 0.21, 0.155, 0.123, 0.106, 0.095]); FPE_PTFE = np.array([.97, .8, .5, .15, .03, .008, .003, .002])
MU_TI = np.array([110, 15.6, 4.97, 1.21, 0.272, 0.131, 0.104, 0.090]); FPE_TI = np.array([1, .99, .97, .85, .45, .12, .05, .03])
rec('mass attenuation coefficients Xe/H2O/LAB/PTFE/Ti 10-400 keV', 'tables in script', 'NIST XCOM (from memory) + P004/P029 anchors', 'likely (+-20%)')
RHO = {TPC: 2.86, RFR: 2.86, SKIN: 2.86, PTFE: 2.2, TI: 4.51, GDLS: 0.86, WATER: 1.0, STRUCT: 1.5}
rec('densities LXe 2.86, PTFE 2.2, Ti 4.51, GdLS 0.86 g/cm3; PMT-array structure 1.5 g/cm3 steel-like', 'see script', 'handbook', 'likely (structure uncertain)')
def _interp(E, Et, tab):
    return np.exp(np.interp(np.log(E), np.log(Et), np.log(tab)))
def mu_lin(mat, E, scale=1.0):
    """linear attenuation coefficient (cm^-1) for arrays mat, E; scale multiplies the xenon coefficient."""
    out = np.zeros_like(E)
    for m in (TPC, RFR, SKIN):
        s = mat == m; out[s] = _interp(E[s], E_TAB, MU_XE) * RHO[m] * scale
    for m, tab in ((PTFE, MU_PTFE), (TI, MU_TI), (GDLS, MU_LS), (WATER, MU_H2O), (STRUCT, MU_TI)):
        s = mat == m; out[s] = _interp(E[s], E_LT, tab) * RHO[m]
    return out
def fpe(mat, E):
    out = np.zeros_like(E)
    for m in (TPC, RFR, SKIN):
        s = mat == m; out[s] = np.interp(np.log(E[s]), np.log(E_TAB), FPE_XE)
    for m, tab in ((PTFE, FPE_PTFE), (TI, FPE_TI), (GDLS, FPE_LS), (WATER, FPE_H2O), (STRUCT, FPE_TI)):
        s = mat == m; out[s] = np.interp(np.log(E[s]), np.log(E_LT), tab)
    return out

def isotropic(n):
    cz = rng.uniform(-1, 1, n); ph = rng.uniform(0, 2 * np.pi, n); st = np.sqrt(1 - cz ** 2)
    return np.stack([st * np.cos(ph), st * np.sin(ph), cz], axis=1)
def rotate(d, cos_t, phi):
    """new directions at angle theta (cos_t) and azimuth phi about d."""
    a = np.where(np.abs(d[:, 0:1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    e1 = np.cross(d, a); e1 /= np.linalg.norm(e1, axis=1)[:, None]; e2 = np.cross(d, e1)
    st = np.sqrt(np.clip(1 - cos_t ** 2, 0, 1))
    return cos_t[:, None] * d + st[:, None] * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2)
def klein_nishina_cos(E_kev):
    """sample cos(theta) from Klein-Nishina by rejection (vectorised)."""
    k = E_kev / M_E_KEV
    out = np.full(E_kev.shape, np.nan)
    todo = np.ones(E_kev.shape, bool)
    while todo.any():
        n = todo.sum(); c = rng.uniform(-1, 1, n); eps = 1 / (1 + k[todo] * (1 - c))
        f = eps ** 2 * (eps + 1 / eps - (1 - c ** 2)) / 2.0         # normalised to <= 1 (max at theta=0)
        acc = rng.uniform(0, 1, n) < f
        idx = np.where(todo)[0][acc]; out[idx] = c[acc]; todo[idx] = False
    return out

SCORE = {TPC: 0, RFR: 1, SKIN: 2, GDLS: 3}
def transport(x0, E0_kev, xe_scale=1.0, max_steps=80):
    """Woodcock (delta-tracking) photon transport. Returns energy deposited (keV) in [TPC, RFR, Skin, GdLS]
    and the position of the first TPC interaction (nan if none)."""
    n = x0.shape[0]
    pos = x0.copy(); d = isotropic(n); E = np.full(n, float(E0_kev))
    dep = np.zeros((n, 4)); first_tpc = np.full((n, 3), np.nan)
    alive = np.ones(n, bool)
    for _ in range(max_steps):
        idx = np.where(alive)[0]
        if idx.size == 0:
            break
        Ei = E[idx]
        mu_max = _interp(Ei, E_TAB, MU_XE) * 2.86 * max(xe_scale, 1.0) * 1.0001
        s = -np.log(rng.uniform(0, 1, idx.size)) / mu_max
        pos[idx] += s[:, None] * d[idx]
        mat = material(pos[idx])
        outw = mat == OUT_
        alive[idx[outw]] = False
        keep = ~outw
        idx, mat, Ei, mu_max = idx[keep], mat[keep], Ei[keep], mu_max[keep]
        mu_here = mu_lin(mat, Ei, xe_scale)
        accept = rng.uniform(0, 1, idx.size) < mu_here / mu_max
        idx, mat, Ei = idx[accept], mat[accept], Ei[accept]
        if idx.size == 0:
            continue
        photo = rng.uniform(0, 1, idx.size) < fpe(mat, Ei)
        # Compton kinematics for the others
        c = klein_nishina_cos(Ei)
        Eprime = Ei / (1 + Ei / M_E_KEV * (1 - c))
        Edep = np.where(photo, Ei, Ei - Eprime)
        for m, col in SCORE.items():
            s = mat == m
            dep[idx[s], col] += Edep[s]
            if m == TPC:
                new = s & np.isnan(first_tpc[idx, 0]); first_tpc[idx[new]] = pos[idx[new]]
        # continue Compton-scattered photons above 5 keV
        cont = (~photo) & (Eprime > 5.0)
        E[idx[cont]] = Eprime[cont]
        d[idx[cont]] = rotate(d[idx[cont]], c[cont], rng.uniform(0, 2 * np.pi, cont.sum()))
        killed = idx[~cont]
        # deposit the residual energy of low-energy scattered photons locally
        low = (~photo) & (Eprime <= 5.0)
        for m, col in SCORE.items():
            s = low & (mat == m); dep[idx[s], col] += Eprime[s]
        alive[killed] = False
    return dep, first_tpc

# Detector-response thresholds (recalled / assumptions, flagged)
E_SKIN_PROMPT = rec('Skin prompt threshold 2.5 phd ~ 5 keV deposited (Skin light yield ~0.5 phd/keV)', 5.0, 'LZ Skin performance (order of magnitude)', 'uncertain')
E_OD_PROMPT = rec('OD prompt threshold 4.5 phd ~ 30 keV deposited (OD light yield ~0.15 phd/keV)', 30.0, 'LZ OD performance (order of magnitude)', 'uncertain')
E_SKIN_DEL, E_OD_DEL = 300.0, 200.0                                  # paper (samples paragraph)
T_SKIN_PROMPT, T_OD_PROMPT, T_DELAYED = 0.25, 0.30, 600.0            # paper, microseconds
SKIN_RES = rec('Skin energy resolution ~10% at 300 keV', 0.10, 'LZ Skin calibrations', 'uncertain')
T_S1_MERGE = rec('two S1 pulses merge if separated by < 0.15 us', 0.15, 'LZ S1 pulse widths ~100-200 ns', 'uncertain')
DS = 3.0                                                             # cm step along the chi2 path
S_MAX = 900.0
REL_R, REL_ZLO, REL_ZHI = R7 + 40.0, Z_GD_LO - 40.0, Z_GD_HI + 40.0  # beyond this the photon cannot reach any detector

def run_config(delta_kev, n_traj=1500, wind=True, xe_scale=1.0, k_phot=2, label=''):
    """Trace chi2 trajectories from the vertex; emit photons every DS along the path; classify outcomes.
    Returns dict with per-emission-point arrays needed to integrate over tau."""
    v, cz, w = sample_chi2(delta_kev, n_traj, wind=wind)
    phi = rng.uniform(0, 2 * np.pi, n_traj); st = np.sqrt(1 - cz ** 2)
    dirs = np.stack([st * np.cos(phi), st * np.sin(phi), cz], axis=1)
    s_grid = np.arange(DS / 2, S_MAX, DS)
    P = VERTEX[None, None, :] + s_grid[None, :, None] * dirs[:, None, :]         # (traj, s, 3)
    mat_path = material(P.reshape(-1, 3)).reshape(n_traj, s_grid.size)
    rel = (np.hypot(P[..., 0], P[..., 1]) < REL_R) & (P[..., 2] > REL_ZLO) & (P[..., 2] < REL_ZHI)
    # path-length bookkeeping per trajectory
    in_tpc = (mat_path == TPC)
    L_tpc = in_tpc.sum(1) * DS
    s_exit_tpc = np.array([s_grid[np.where(~row)[0][0]] if (~row).any() else S_MAX for row in in_tpc])
    L_skin = (mat_path == SKIN).sum(1) * DS; L_gd = (mat_path == GDLS).sum(1) * DS; L_rfr = (mat_path == RFR).sum(1) * DS
    s_rel_end = np.array([s_grid[np.where(~row)[0][0]] if (~row).any() else S_MAX for row in rel])
    # emission points: all relevant path points
    ti, si = np.where(rel)
    x_em = P[ti, si]
    x_em = np.repeat(x_em, k_phot, axis=0); ti_p = np.repeat(ti, k_phot); si_p = np.repeat(si, k_phot)
    dep, ftpc = transport(x_em, delta_kev, xe_scale=xe_scale)
    t_us = s_grid[si_p] / v[ti_p] * 1e-5          # cm / (km/s) -> s: cm=1e-5 km; times 1e6 for us -> 1e-5*1e6/1e6... compute explicitly
    t_us = (s_grid[si_p] * 1e-5) / v[ti_p] * 1e6  # microseconds
    E_tpc, E_rfr, E_skin, E_od = dep.T
    skin_rec = E_skin * (1 + SKIN_RES * rng.normal(size=E_skin.size))
    cls = np.full(E_tpc.size, 6, dtype=np.int8)   # 6 = clean
    # precedence: TPC second site > RFR S1-only > prompt veto > delayed veto > clean
    delayed = ((skin_rec >= E_SKIN_DEL) & (t_us > T_SKIN_PROMPT) & (t_us <= T_DELAYED)) | ((E_od >= E_OD_DEL) & (t_us > T_OD_PROMPT) & (t_us <= T_DELAYED))
    prompt = ((E_skin >= E_SKIN_PROMPT) & (t_us <= T_SKIN_PROMPT)) | ((E_od >= E_OD_PROMPT) & (t_us <= T_OD_PROMPT))
    cls[delayed] = 4; cls[prompt] = 3
    cls[E_rfr > 1.0] = 2
    cls[E_tpc > 1.0] = 0
    # sub-classes of TPC second site: merged S1 (t < T_S1_MERGE) -> 1
    cls[(E_tpc > 1.0) & (t_us < T_S1_MERGE)] = 1
    # separation between vertex and the ER site for TPC conversions
    sep = np.linalg.norm(ftpc - VERTEX[None, :], axis=1)
    return dict(delta=delta_kev, label=label, v=v, w=w, cz=cz, s_grid=s_grid, ti=ti_p, si=si_p, cls=cls, t_us=t_us, k=k_phot,
                L_tpc=L_tpc, s_exit_tpc=s_exit_tpc, L_skin=L_skin, L_gd=L_gd, L_rfr=L_rfr, s_rel_end=s_rel_end, sep=sep,
                E_tpc=E_tpc, E_skin=E_skin, E_od=E_od, E_rfr=E_rfr)

CLASSES = ['TPC second site (resolved S1s)', 'TPC second site (merged S1)', 'RFR S1-only', 'prompt veto', 'delayed veto', '(unused)', 'clean']
def integrate_tau(cfg, tau_grid_s):
    """P(class | tau) by integrating the exponential decay-time density along each trajectory."""
    v_cm_us = cfg['v'] * 1e-1            # km/s -> cm/us
    n_traj = cfg['v'].size
    res = np.zeros((tau_grid_s.size, 7))
    ti, si, cls, w, k = cfg['ti'], cfg['si'], cfg['cls'], cfg['w'], cfg['k']
    s_lo = cfg['s_grid'][si] - DS / 2; s_hi = s_lo + DS
    for j, tau in enumerate(tau_grid_s):
        lam = 1.0 / (v_cm_us * tau * 1e6)                     # per cm, per trajectory
        pw = (np.exp(-lam[ti] * s_lo) - np.exp(-lam[ti] * s_hi)) * w[ti] / k
        for c in range(7):
            res[j, c] = pw[cls == c].sum()
        # decays beyond the relevant region (or never): clean
        res[j, 6] += (np.exp(-lam * cfg['s_rel_end']) * w).sum()
    return res

TAU = np.logspace(-9, 2, 221)
def summarize(cfg, name, save=True):
    res = integrate_tau(cfg, TAU)
    df = pd.DataFrame(res, columns=['P_tpc_resolved', 'P_tpc_merged', 'P_rfr', 'P_prompt', 'P_delayed', 'unused', 'P_clean'])
    df.insert(0, 'tau_s', TAU); df['P_tpc'] = df.P_tpc_resolved + df.P_tpc_merged
    df['P_notclean'] = 1 - df.P_clean
    if save:
        df.drop(columns='unused').to_csv(os.path.join(OUT, f'P042_topology_vs_tau_{name}.csv'), index=False)
    return df

def crossing(df, col, target):
    """tau at which df[col] crosses target (log interpolation), assuming monotonic."""
    y = df[col].values; x = np.log10(df.tau_s.values)
    order = np.argsort(y)
    return 10 ** np.interp(target, y[order], x[order])

def at_tau(df, tau):
    return {c: float(np.interp(np.log10(tau), np.log10(df.tau_s), df[c])) for c in df.columns if c != 'tau_s'}

# ---- baseline configuration: delta = 300 keV, DM-wind directions -----------------------------------------
log('\n=== Part 2/3: geometry MC ===')
log(f'geometry: TPC r<{R_TPC} cm, 0<z<{H_TPC}; RFR to -{Z_RFR}; Skin {R1:.1f}-{R2:.1f} cm; ICV {R2:.1f}-{R3:.1f}; OCV {R4:.1f}-{R5:.1f}; '
    f'GdLS {R6:.1f}-{R7:.1f} cm, z {Z_GD_LO:.0f}..{Z_GD_HI:.0f}; water to {R_WATER} cm')
lam_xe = {E: 1 / (_interp(np.array([E]), E_TAB, MU_XE)[0] * 2.86) for E in (250, 300, 350, 380)}
log('LXe attenuation length (cm): ' + ', '.join(f'{E} keV: {l:.2f}' for E, l in lam_xe.items()) +
    f'; GdLS 300 keV: {1/(_interp(np.array([300.]), E_LT, MU_LS)[0]*0.86):.1f} cm; water {1/(_interp(np.array([300.]), E_LT, MU_H2O)[0]):.1f} cm; Ti {1/(_interp(np.array([300.]), E_LT, MU_TI)[0]*4.51):.2f} cm')

base = run_config(300.0, n_traj=2000, wind=True, label='d300_wind')
df_base = summarize(base, 'd300_wind')
w = base['w']
def wmean(a, w=w): return float(np.sum(a * w))
def wq(a, q, w=w):
    o = np.argsort(a); cw = np.cumsum(w[o]); return float(a[o][np.searchsorted(cw, q)])
v_cm_us = base['v'] * 0.1
t_tpc = base['L_tpc'] / v_cm_us
geo = dict(L_tpc_mean_cm=wmean(base['L_tpc']), L_tpc_p10=wq(base['L_tpc'], .1), L_tpc_p90=wq(base['L_tpc'], .9),
           L_tpc_min=float(base['L_tpc'].min()), L_tpc_max=float(base['L_tpc'].max()),
           t_tpc_mean_us=wmean(t_tpc), t_tpc_p10=wq(t_tpc, .1), t_tpc_p90=wq(t_tpc, .9),
           L_skin_mean_cm=wmean(base['L_skin']), t_skin_mean_us=wmean(base['L_skin'] / v_cm_us),
           L_gdls_mean_cm=wmean(base['L_gd']), t_gdls_mean_us=wmean(base['L_gd'] / v_cm_us),
           frac_through_rfr=wmean((base['L_rfr'] > 0).astype(float)), L_rfr_mean_cm=wmean(base['L_rfr']),
           t_leave_relevant_us_mean=wmean(base['s_rel_end'] / v_cm_us), t_leave_relevant_us_max=float((base['s_rel_end'] / v_cm_us).max()),
           v2_mean_kms=wmean(base['v']))
log('chi2 path in active TPC: mean {L_tpc_mean_cm:.1f} cm (10-90%: {L_tpc_p10:.0f}-{L_tpc_p90:.0f}, range {L_tpc_min:.0f}-{L_tpc_max:.0f}); '
    'time in TPC mean {t_tpc_mean_us:.2f} us (10-90%: {t_tpc_p10:.2f}-{t_tpc_p90:.2f}); Skin {L_skin_mean_cm:.1f} cm / {t_skin_mean_us:.3f} us; '
    'GdLS {L_gdls_mean_cm:.1f} cm / {t_gdls_mean_us:.2f} us; through RFR: {frac_through_rfr:.2f} (mean {L_rfr_mean_cm:.1f} cm); '
    'leaves everything after {t_leave_relevant_us_mean:.1f} us (max {t_leave_relevant_us_max:.1f})'.format(**geo))

# photon conversion probability at the vertex itself (leakage) and the class shares at the tau->infinity limit
res_inf = integrate_tau(base, np.array([1.0]))[0]      # tau = 1 s: everything ~ proportional to time in region
share_inf = res_inf[:5] / res_inf[:5].sum()
log('tau >> transit: shares of the non-clean outcomes  TPC(resolved) {:.3f}, TPC(merged) {:.3f}, RFR S1-only {:.3f}, prompt veto {:.3f}, delayed veto {:.3f}'.format(*share_inf))
tau_notclean_1s = 1 - res_inf[6]
log(f'  P_notclean(tau) -> {tau_notclean_1s:.3e} x (1 s / tau)  =>  effective "sensitive time" {tau_notclean_1s*1e6:.3f} us')

# vertex leakage: photon emitted at the vertex reaching outside the TPC
dep0, _ = transport(np.repeat(VERTEX[None, :], 20000, axis=0), 300.0)
leak0 = float(np.mean(dep0[:, 0] < 1.0)); skin0 = float(np.mean((dep0[:, 0] < 1.0) & (dep0[:, 2] >= E_SKIN_PROMPT)))
log(f'photon emitted at the vertex: P(no TPC deposit) = {leak0:.4f}; P(no TPC & Skin >= {E_SKIN_PROMPT:.0f} keV) = {skin0:.4f}')

# limits from the single clean event
lim = dict(tau90_s=crossing(df_base, 'P_clean', 0.10), tau50_s=crossing(df_base, 'P_clean', 0.50), tau_exp90_s=crossing(df_base, 'P_clean', 0.90),
           tau_exp99_s=crossing(df_base, 'P_clean', 0.99))
log('single clean event, delta = 300 keV: P_clean = 0.10 at tau = {tau90_s:.3e} s (90% CL lower limit); 0.50 at {tau50_s:.3e}; '
    '0.90 at {tau_exp90_s:.3e} (P023-type "expected topology" line); 0.99 at {tau_exp99_s:.3e}'.format(**lim))
for tau in [1e-8, 1e-7, 1.4e-7, 5e-7, 1e-6, 1.4e-6, 5e-6, 1e-5, 6.6e-5, 1e-3, 6e-2]:
    a = at_tau(df_base, tau)
    log(f'  tau = {tau:.1e} s: P_tpc {a["P_tpc"]:.3e} (merged {a["P_tpc_merged"]:.2e}), P_rfr {a["P_rfr"]:.2e}, prompt {a["P_prompt"]:.2e}, delayed {a["P_delayed"]:.2e}, clean {a["P_clean"]:.4f}')
# BR_gamma limit: P_clean = 1 - BR (1 - P_clean_gamma) >= 0.10
df_base['BR_gamma_max_90'] = np.minimum(1.0, 0.9 / np.maximum(df_base.P_notclean, 1e-12))
# Bayesian: log-uniform prior on tau in [1e-9, 1e2]
post = df_base.P_clean.values / np.trapezoid(df_base.P_clean.values, np.log10(TAU))
cdf = np.concatenate([[0], np.cumsum(0.5 * (post[1:] + post[:-1]) * np.diff(np.log10(TAU)))])
lim['tau_bayes_10pct_s'] = float(10 ** np.interp(0.10, cdf, np.log10(TAU)))
log(f'  Bayesian (log-uniform prior 1e-9..1e2 s): 10th percentile of the posterior at tau = {lim["tau_bayes_10pct_s"]:.2e} s (prior-dominated)')

# ---- variants -----------------------------------------------------------------------------------
log('\n=== variants ===')
variants = {}
for name, kw in [('d300_iso', dict(wind=False)), ('d300_xe0p7', dict(xe_scale=0.7)), ('d300_xe1p4', dict(xe_scale=1.4))]:
    cfg = run_config(300.0, n_traj=1500, k_phot=1, label=name, **kw)
    dfv = summarize(cfg, name)
    variants[name] = dict(tau90_s=crossing(dfv, 'P_clean', 0.10), tau_exp90_s=crossing(dfv, 'P_clean', 0.90),
                          P_notclean_at_66us=at_tau(dfv, 6.6e-5)['P_notclean'], P_delayed_at_66us=at_tau(dfv, 6.6e-5)['P_delayed'],
                          t_tpc_mean_us=float(np.sum(cfg['L_tpc'] / (cfg['v'] * 0.1) * cfg['w'])))
    log(f'  {name}: tau90 {variants[name]["tau90_s"]:.2e} s, tau(P_clean=0.9) {variants[name]["tau_exp90_s"]:.2e} s, '
        f'P_notclean(66 us) {variants[name]["P_notclean_at_66us"]:.4f}, P_delayed(66 us) {variants[name]["P_delayed_at_66us"]:.2e}, <t_TPC> {variants[name]["t_tpc_mean_us"]:.2f} us')
# threshold variants (re-classify the baseline sample with different veto thresholds)
def reclassify(cfg, e_skin_prompt=E_SKIN_PROMPT, e_od_prompt=E_OD_PROMPT, skin_res=SKIN_RES):
    c = dict(cfg); E_tpc, E_rfr, E_skin, E_od, t_us = cfg['E_tpc'], cfg['E_rfr'], cfg['E_skin'], cfg['E_od'], cfg['t_us']
    skin_rec = E_skin * (1 + skin_res * rng.normal(size=E_skin.size))
    cls = np.full(E_tpc.size, 6, dtype=np.int8)
    delayed = ((skin_rec >= E_SKIN_DEL) & (t_us > T_SKIN_PROMPT) & (t_us <= T_DELAYED)) | ((E_od >= E_OD_DEL) & (t_us > T_OD_PROMPT) & (t_us <= T_DELAYED))
    prompt = ((E_skin >= e_skin_prompt) & (t_us <= T_SKIN_PROMPT)) | ((E_od >= e_od_prompt) & (t_us <= T_OD_PROMPT))
    cls[delayed] = 4; cls[prompt] = 3; cls[E_rfr > 1.0] = 2; cls[E_tpc > 1.0] = 0; cls[(E_tpc > 1.0) & (t_us < T_S1_MERGE)] = 1
    c['cls'] = cls; return c
for name, kw in [('d300_odprompt100keV', dict(e_od_prompt=100.0)), ('d300_skinres0', dict(skin_res=0.0))]:
    dfv = summarize(reclassify(base, **kw), name)
    variants[name] = dict(tau90_s=crossing(dfv, 'P_clean', 0.10), tau_exp90_s=crossing(dfv, 'P_clean', 0.90),
                          P_notclean_at_66us=at_tau(dfv, 6.6e-5)['P_notclean'], P_delayed_at_66us=at_tau(dfv, 6.6e-5)['P_delayed'])
    log(f'  {name}: tau90 {variants[name]["tau90_s"]:.2e} s, P_delayed(66 us) {variants[name]["P_delayed_at_66us"]:.2e}')

# ---- delta scan (photon energy and kinematics) ------------------------------------------------------
log('\n=== delta scan ===')
DELTAS = [250, 270, 290, 300, 310, 320, 330, 340, 350, 360, 366, 370, 380, 390]
scan = {}
for d in DELTAS:
    if lz.vmin_kms(E_R, M_CHI, delta_kev=d) >= lz.VESC_KMS + np.linalg.norm(V_LAB_GAL):
        log(f'  delta {d}: kinematically forbidden on 16 June (v_min {lz.vmin_kms(E_R, M_CHI, delta_kev=d):.0f} > v_max {lz.VESC_KMS + np.linalg.norm(V_LAB_GAL):.0f} km/s) -- skipped')
        continue
    cfg = base if d == 300 else run_config(float(d), n_traj=500, k_phot=1, label=f'd{d}')
    dfd = summarize(cfg, f'd{d}_wind', save=(d in (250, 350, 380)))
    scan[d] = dfd
    log(f'  delta {d}: tau90 {crossing(dfd, "P_clean", 0.1):.2e} s, tau(0.9) {crossing(dfd, "P_clean", 0.9):.2e} s, <v2> {np.sum(cfg["v"]*cfg["w"]):.0f} km/s, '
        f'<t_TPC> {np.sum(cfg["L_tpc"]/(cfg["v"]*0.1)*cfg["w"]):.2f} us')

# ----------------------------------------------------------------------------------------------
# Part 4: model mapping
# ----------------------------------------------------------------------------------------------
log('\n=== Part 4: models ===')
def tau_from_mu(mu_muN, delta_kev):
    mu = mu_muN * MU_N_GEVINV; d = delta_kev * 1e-6
    return HBAR_GEV_S * math.pi / (mu ** 2 * d ** 3)
def mu_from_tau(tau_s, delta_kev):
    d = delta_kev * 1e-6
    return math.sqrt(HBAR_GEV_S * math.pi / (tau_s * d ** 3)) / MU_N_GEVINV
rec('MiDM width Gamma(chi2 -> chi1 gamma) = mu_tr^2 delta^3 / pi', 'formula', 'Chang, Weiner, Yavin 2010 (as used by P023/P014)', 'likely')
log(f'check: mu_tr = 2.107e-4 mu_N at 300 keV -> tau = {tau_from_mu(2.107e-4, 300):.3e} s (P023: 6.626e-5 s)')
p023 = pd.read_csv('output/work/P023/P023_single_site_ceiling.csv')
rows = []
for d in DELTAS:
    if d not in scan:
        continue
    dfd = scan[d]
    for case in ['N=1', 'N=3.65', 'N=0.105']:
        r = p023[(p023.delta_keV == d) & (p023.case == case)]
        if r.empty:
            continue
        mu = float(r.mu_muN.iloc[0]); tau = tau_from_mu(mu, d)
        a = at_tau(dfd, tau)
        rows.append(dict(delta_keV=d, case=case, mu_tr_muN=mu, tau_s=tau, P_clean=a['P_clean'], P_tpc=a['P_tpc'], P_prompt=a['P_prompt'], P_delayed=a['P_delayed'],
                         tau90_s=crossing(dfd, 'P_clean', 0.1), tau_exp90_s=crossing(dfd, 'P_clean', 0.9),
                         mu_max90_muN=mu_from_tau(crossing(dfd, 'P_clean', 0.1), d), mu_exp90_muN=mu_from_tau(crossing(dfd, 'P_clean', 0.9), d)))
mod = pd.DataFrame(rows); mod.to_csv(os.path.join(OUT, 'P042_photonM1_delta_scan.csv'), index=False)
def delta_cross(case, target):
    s = mod[mod.case == case].sort_values('delta_keV')
    y = s.P_clean.values; x = s.delta_keV.values
    # P_clean decreases with delta; find first crossing
    for i in range(len(x) - 1):
        if (y[i] - target) * (y[i + 1] - target) <= 0:
            return float(x[i] + (target - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
    return float('nan')
ceil = {case: dict(delta_Pclean_0p9=delta_cross(case, 0.9), delta_Pclean_0p5=delta_cross(case, 0.5), delta_Pclean_0p1=delta_cross(case, 0.1)) for case in ['N=1', 'N=3.65', 'N=0.105']}
for case, c in ceil.items():
    log(f'  photon-M1 ({case}): P_clean = 0.9 at delta = {c["delta_Pclean_0p9"]:.0f} keV; 0.5 at {c["delta_Pclean_0p5"]:.0f}; 0.1 (90% CL exclusion) at {c["delta_Pclean_0p1"]:.0f} keV')
for d in [300, 330, 350, 360, 366, 380]:
    r = mod[(mod.delta_keV == d) & (mod.case == 'N=1')].iloc[0]
    log(f'  N=1 delta {d}: mu {r.mu_tr_muN:.2e} mu_N, tau {r.tau_s:.2e} s, P_clean {r.P_clean:.3f}, mu_max(90%) {r.mu_max90_muN:.2e} mu_N (headroom x{r.mu_max90_muN/r.mu_tr_muN:.1f})')

# Higgsino (P014): tau_gamma ~ 0.06 s (loop dipole), tau_nunu 4.6e5 s
a = at_tau(df_base, 0.06); higgs = dict(tau_gamma_s=0.06, P_notclean=a['P_notclean'], tau_nunu_s=4.6e5)
log(f'  Higgsino: tau_gamma 0.06 s -> P_notclean {a["P_notclean"]:.1e}; tau_nunu 4.6e5 s -> invisible')
# Dark photon (P011): chi2 -> chi1 nu nubar, 1.9e18 s at eps 1e-6; no photon channel
a = at_tau(df_base, 1.9e18 if 1.9e18 < TAU.max() else TAU.max()); dp = dict(tau_nunu_s=1.9e18, P_notclean_upper=float(tau_notclean_1s / 1.9e18))
log(f'  dark photon: tau(nu nu) 1.9e18 s -> P_notclean ~ {dp["P_notclean_upper"]:.1e}; invisible final state anyway')
# L10-type inelastic tensor contact: loop-induced transition moment (order-of-magnitude estimate)
d10 = 0.28; mv = lz.M_V_GEV
C = d10 / mv ** 2
mu_nuc = C * E_CHARGE * M_P / (16 * math.pi ** 2) * 2.0        # nucleon-level loop, log ~ 2
mu_quark = C * E_CHARGE * 0.005 / (16 * math.pi ** 2) * 2.0    # quark-level with m_q ~ 5 MeV, Q ~ 1 (order of magnitude)
rec('loop-induced transition dipole from a tensor contact operator: mu ~ e C m_f log/(16 pi^2), m_f = m_N (nucleon loop) or m_q (quark loop)', 'estimate', 'dimensional analysis / Schwinger-type estimate', 'uncertain (x10)')
l10 = dict(d10=d10, C_GeVm2=C, mu_nucleon_loop_muN=mu_nuc / MU_N_GEVINV, tau_nucleon_loop_s=tau_from_mu(mu_nuc / MU_N_GEVINV, 300),
           mu_quark_loop_muN=mu_quark / MU_N_GEVINV, tau_quark_loop_s=tau_from_mu(mu_quark / MU_N_GEVINV, 300))
l10['decay_length_nucleon_loop_km'] = l10['tau_nucleon_loop_s'] * 750.0
log(f'  L10-type inelastic contact (d10 = 0.28): loop mu_tr ~ {l10["mu_nucleon_loop_muN"]:.1e} mu_N (nucleon loop) -> tau ~ {l10["tau_nucleon_loop_s"]:.0f} s '
    f'(decay length {l10["decay_length_nucleon_loop_km"]:.1e} km); quark loop ~ {l10["mu_quark_loop_muN"]:.1e} mu_N -> {l10["tau_quark_loop_s"]:.1e} s')

# ----------------------------------------------------------------------------------------------
# Part 5: population predictions
# ----------------------------------------------------------------------------------------------
log('\n=== Part 5: population ===')
pop_rows = []
for label, tau, d in [('MiDM delta=300 (66 us)', 6.626e-5, 300), ('MiDM delta=330 (9.4 us)', 9.375e-6, 330), ('MiDM delta=350 (1.4 us)', 1.414e-6, 350),
                      ('MiDM delta=366 (0.14 us)', 1.403e-7, 366), ('Higgsino (0.06 s)', 0.06, 300), ('tau = 1 us', 1e-6, 300), ('tau = 10 us', 1e-5, 300)]:
    dfd = scan[d]; a = at_tau(dfd, tau)
    pop_rows.append(dict(model=label, tau_s=tau, delta_keV=d, f_clean=a['P_clean'], f_tpc_two_site=a['P_tpc'], f_tpc_merged=a['P_tpc_merged'], f_rfr=a['P_rfr'],
                         f_prompt=a['P_prompt'], f_delayed=a['P_delayed'], N_for_one_two_site=1 / max(a['P_tpc'], 1e-30), N_for_one_veto=1 / max(a['P_prompt'] + a['P_delayed'], 1e-30)))
pop = pd.DataFrame(pop_rows); pop.to_csv(os.path.join(OUT, 'P042_population.csv'), index=False)
for r in pop_rows:
    log('  {model}: clean {f_clean:.4f}, two-site {f_tpc_two_site:.3e} (merged {f_tpc_merged:.1e}), RFR {f_rfr:.1e}, prompt {f_prompt:.1e}, delayed {f_delayed:.1e}; '
        'events per two-site {N_for_one_two_site:.3g}, per veto-tag {N_for_one_veto:.3g}'.format(**r))
# geometry of the two-site companions (tau >> transit: uniform along the TPC chord)
tpc_sel = (base['cls'] <= 1)
sep = base['sep'][tpc_sel]; t2 = base['t_us'][tpc_sel]; wsel = base['w'][base['ti'][tpc_sel]]
comp = dict(sep_mean_cm=float(np.average(sep, weights=wsel)), sep_p10=float(np.quantile(sep, .1)), sep_p90=float(np.quantile(sep, .9)),
            dt_S1_mean_us=float(np.average(t2, weights=wsel)), dt_S1_p90_us=float(np.quantile(t2, .9)),
            frac_full_absorption=float(np.average(base['E_tpc'][tpc_sel] > 0.98 * 300, weights=wsel)),
            E_tpc_mean_keV=float(np.average(base['E_tpc'][tpc_sel], weights=wsel)), v_ratio_cm_per_us=float(np.average(sep / np.maximum(t2, 1e-3), weights=wsel)))
log('  two-site companions (tau >> transit): vertex-ER separation mean {sep_mean_cm:.0f} cm (10-90%: {sep_p10:.0f}-{sep_p90:.0f}); S1 time difference mean {dt_S1_mean_us:.2f} us (90%: {dt_S1_p90_us:.2f}); '
    'full-absorption fraction {frac_full_absorption:.2f}, mean ER energy {E_tpc_mean_keV:.0f} keV; separation/dt = {v_ratio_cm_per_us:.0f} cm/us'.format(**comp))
# LZ's 55 delayed-veto events: expected inelastic contribution given one science-sample event
for label, tau, d in [('MiDM 300', 6.626e-5, 300), ('MiDM 350', 1.414e-6, 350)]:
    a = at_tau(scan[d], tau)
    log(f'  given 1 clean science event, expected delayed-veto inelastic events ({label}): {a["P_delayed"]/a["P_clean"]:.2e}; prompt-veto: {a["P_prompt"]/a["P_clean"]:.2e}; two-site (rejected as SS): {a["P_tpc"]/a["P_clean"]:.2e}')

# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------
C1, C2, C3, C4, C5, C6, C7 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5})
fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
ax.plot(TAU, df_base.P_clean, color='#52514e', lw=2, label='clean single site (observed class)')
ax.plot(TAU, df_base.P_tpc, color=C1, lw=2, label='second site in TPC (fails SS)')
ax.plot(TAU, df_base.P_rfr, color=C7, lw=2, label='S1-only deposit in RFR')
ax.plot(TAU, df_base.P_prompt, color=C2, lw=2, label='prompt veto (Skin/OD, <0.3 us)')
ax.plot(TAU, df_base.P_delayed, color=C3, lw=2, label='delayed veto (Skin>300 / OD>200 keV)')
for tau, txt, col in [(6.626e-5, 'MiDM 300 keV', C4), (1.414e-6, 'MiDM 350 keV', C4), (1.403e-7, 'MiDM 366 keV', C4), (0.06, 'Higgsino loop', C5)]:
    ax.axvline(tau, color=col, lw=1, ls=':'); ax.text(tau, 1.25, txt, rotation=90, va='top', ha='right', fontsize=7, color='#52514e')
ax.axvline(lim['tau90_s'], color='#0b0b0b', lw=1, ls='--'); ax.text(lim['tau90_s'] * 1.15, 0.5, f'90% CL: tau > {lim["tau90_s"]*1e6:.2f} us', fontsize=7, color='#0b0b0b')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(1e-9, 1e2); ax.set_ylim(1e-6, 1.3)
ax.set_xlabel('chi2 lifetime tau [s]'); ax.set_ylabel('probability of event class')
ax.set_title('P042: fate of the de-excitation photon vs chi2 lifetime (delta = 300 keV, LZ vertex, DM-wind directions)', fontsize=8.5)
ax.legend(fontsize=7, loc='lower left', frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P042_fig1_topology_vs_tau.png')); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
s1 = mod[mod.case == 'N=1'].sort_values('delta_keV'); s2 = mod[mod.case == 'N=3.65'].sort_values('delta_keV'); s3 = mod[mod.case == 'N=0.105'].sort_values('delta_keV')
ax.fill_between(s1.delta_keV, s3.mu_tr_muN, s2.mu_tr_muN, color=C1, alpha=0.18, lw=0, label='mu_tr for 0.105-3.65 LZ events (P023)')
ax.plot(s1.delta_keV, s1.mu_tr_muN, color=C1, lw=2, label='mu_tr for N = 1 (P023)')
ax.plot(s1.delta_keV, s1.mu_max90_muN, color=C2, lw=2, label='excluded above: P(clean) < 10% (this work)')
ax.plot(s1.delta_keV, s1.mu_exp90_muN, color=C2, lw=1.5, ls='--', label='P(clean) = 90% line')
ax.axhline(2.1e-5, color=C7, lw=1, ls=':'); ax.text(252, 2.4e-5, 'P012 elastic photon dipole (excluded by low-E null)', fontsize=7, color='#52514e')
ax.set_yscale('log'); ax.set_xlabel('mass splitting delta [keV]'); ax.set_ylabel('transition moment mu_tr [mu_N]')
ax.set_xlim(250, 390); ax.set_ylim(1e-5, 3)
for k, lab in [('delta_Pclean_0p1', '90% CL'), ('delta_Pclean_0p9', 'P=0.9')]:
    x = ceil['N=1'][k]; ax.axvline(x, color='#0b0b0b', lw=0.8, ls='--'); ax.text(x + 1, 1.5, f'{lab}: {x:.0f} keV', fontsize=7, rotation=90, va='top')
ax.set_title('P042: photon-M1 transition moment - LZ rate requirement vs single-site ceiling (1 TeV)', fontsize=8.5)
ax.legend(fontsize=7, loc='upper left', frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P042_fig2_mu_tr_ceiling.png')); plt.close(fig)

# ----------------------------------------------------------------------------------------------
# Summary JSON
# ----------------------------------------------------------------------------------------------
summary = dict(event=dict(vertex_cm=VERTEX.tolist(), E_R_keV=E_R, m_chi_GeV=M_CHI, time=lz.LZ['ev_time_utc']),
               wind=dict(v_lab_gal_kms=V_LAB_GAL.tolist(), antiapex_ra_dec_deg=[float(ra_ap[0]), float(dec_ap[0])], antiapex_altitude_deg=alt_antiapex, LST_h=LST_DEG / 15),
               kinematics=kin_rows, geometry_paths=geo, lambda_LXe_cm=lam_xe, vertex_leakage=dict(P_no_TPC_deposit=leak0, P_skin_prompt_capable=skin0),
               shares_long_tau=dict(zip(['tpc_resolved', 'tpc_merged', 'rfr', 'prompt', 'delayed'], share_inf.tolist())), sensitive_time_us=tau_notclean_1s * 1e6,
               limits_d300=lim, variants=variants, photonM1_ceilings=ceil, higgsino=higgs, dark_photon=dp, L10_loop=l10, companions=comp,
               runtime_s=time.time() - T0)
def _clean(o):
    if isinstance(o, dict): return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    return o
json.dump(_clean(summary), open(os.path.join(OUT, 'P042_summary.json'), 'w'), indent=1)
json.dump(RECALLED, open(os.path.join(OUT, 'recalled_inputs.json'), 'w'), indent=1)
log(f'\ndone in {time.time()-T0:.0f} s; {len(RECALLED)} recalled items')
