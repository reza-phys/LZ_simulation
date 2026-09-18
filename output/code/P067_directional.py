"""P067: Directional and multi-target strategies to confirm a high-energy recoil population.

Parts
  1. Kinematics per candidate directional target nucleus (He, C, O, F, S, Ca, Br, Ag, I, Xe, W): inelastic
     ceiling mu v_max^2/2, delta_max(248 keV), A_min(delta), recoil windows, elastic end-points; recalled
     (flagged) track-length scalings -> directional detectability.
  2. Recoil-direction distributions (Monte Carlo Radon transform of the Baxter-2021 truncated Maxwellian boosted
     to the lab on 16 June 2023 21:22:39 UTC, and in the Sun frame) for (i) inelastic O1 at fixed E_R = 248 keV
     on Xe/I (delta = 300-380 keV), (ii) elastic L10 (q^4 Sigma') spectra on F and Xe (WimPyDD kernels), (iii)
     elastic Xe at 248 keV; dipole anisotropy and the number of events for a 3 sigma anisotropy with a 30 deg
     resolution / 70 % head-tail detector (analytic + toy MC).
  3. SURF horizon geometry of the DM wind at the event time (apex/anti-apex alt-az; expected recoil direction).
  4. Rates for directional heavy-gas TPCs / fluorine TPCs / emulsions at LZ's best fit and a decision table.

Run from the simulation root:  .venv/bin/python output/code/P067_directional.py
"""
import os, sys, json, math, time
import numpy as np
sys.path.insert(0, 'output/code')
from common import lzcommon as lz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'output/work/P067'; FIG = OUT + '/figures'
os.makedirs(FIG, exist_ok=True)
T0 = time.time()
LOG = []
def log(s):
    print(s, flush=True); LOG.append(s)

RECALLED = []
def rec(item, value, source, reliability):
    RECALLED.append(dict(item=item, value=value if isinstance(value, (int, float, str)) else str(value),
                         presumed_source=source, reliability=reliability))
    return value

C = lz.C_KMS
M_CHI = 1000.0                     # GeV (LZ best-fit mass class for L10 / inelastic; P002, P012)
E_EV = 248.0                       # keV, event energy under the NR hypothesis
V0, VESC = lz.V0_KMS, lz.VESC_KMS  # Baxter 2021 via lzcommon
rng = np.random.default_rng(67)

# ------------------------------------------------------------------------------------------------
# Lab velocity in the Galactic frame at the event time (astropy ephemeris, as P042) and Sun frame
# ------------------------------------------------------------------------------------------------
from astropy.utils import iers
iers.conf.auto_download = False
from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel, SkyCoord, CartesianRepresentation
import astropy.units as u
T_EV = Time(lz.LZ['ev_time_utc'], scale='utc')
_, vel = get_body_barycentric_posvel('earth', T_EV)
v_earth_icrs = vel.xyz.to(u.km / u.s).value
def icrs_to_gal_vec(vec):
    c = SkyCoord(CartesianRepresentation(vec[0], vec[1], vec[2]), frame='icrs')
    return np.array(c.galactic.cartesian.xyz.value)
def gal_to_icrs_radec(vecs):
    vecs = np.atleast_2d(vecs)
    c = SkyCoord(CartesianRepresentation(vecs[:, 0], vecs[:, 1], vecs[:, 2]), frame='galactic').icrs
    return np.asarray(c.ra.deg), np.asarray(c.dec.deg)
# Galactic -> ICRS rotation matrix from three basis vectors (fixed rotation; no IERS needed)
R_G2I = np.array([SkyCoord(CartesianRepresentation(*e), frame='galactic').icrs.cartesian.xyz.value
                  for e in np.eye(3)]).T          # columns = ICRS components of Galactic basis vectors
v_earth_gal = icrs_to_gal_vec(v_earth_icrs)
V_SUN_GAL = np.array([0.0, V0, 0.0]) + lz.V_SUN_PEC
V_LAB_GAL = V_SUN_GAL + v_earth_gal
V_LAB = float(np.linalg.norm(V_LAB_GAL)); V_SUN = float(np.linalg.norm(V_SUN_GAL))
VMAX_JUNE = V_LAB + VESC
log(f'lab velocity 16 June 2023 21:22:39 UTC: |v_lab| = {V_LAB:.1f} km/s (lzcommon cosine model {lz.v_earth_kms(167):.1f}); '
    f'Sun frame {V_SUN:.1f}; v_max(June) = {VMAX_JUNE:.1f} km/s')
W_HAT = -V_LAB_GAL / V_LAB           # direction the DM wind blows TOWARD (anti-apex); recoils go this way

# ------------------------------------------------------------------------------------------------
# Part 1: kinematics per target
# ------------------------------------------------------------------------------------------------
TARGETS = {  # symbol: (A used, isotopes note, medium note)
    'He': dict(A=4.0,   Z=2,  note='He (CYGNUS He:SF6 buffer)'),
    'C':  dict(A=12.0,  Z=6,  note='C (CF4, emulsion gel, diamond)'),
    'O':  dict(A=16.0,  Z=8,  note='O (CaWO4, emulsion gel)'),
    'F':  dict(A=19.0,  Z=9,  note='F (CF4, SF6, C3F8, CF3I)'),
    'S':  dict(A=32.0,  Z=16, note='S (SF6, CS2)'),
    'Ca': dict(A=40.0,  Z=20, note='Ca (CaWO4)'),
    'Br': dict(A=79.9,  Z=35, note='Br 79/81 (AgBr emulsion)'),
    'Ag': dict(A=107.9, Z=47, note='Ag 107/109 (AgBr emulsion)'),
    'I':  dict(A=126.9, Z=53, note='I (CF3I gas, NaI)'),
    'Xe': dict(A=lz.A_XE_MEAN, Z=54, note='Xe (LXe TPC; low-pressure / HP gas TPC)'),
    'W':  dict(A=183.8, Z=74, note='W (CaWO4 crystal)'),
}
rec('natural isotopes: Br 79/81 (50.7/49.3 %), Ag 107/109 (51.8/48.2 %), mean A 79.9 / 107.9', 'A', 'nuclear data tables', 'certain')
rec('nuclear masses approximated by A x amu (0.9315 GeV); error < 0.1 % in delta_max', 'approx', 'standard', 'certain')

def ceiling_kev(A, m_chi, v_kms):
    mu = lz.mu_red(m_chi, lz.m_nucleus_gev(A))
    return 0.5 * mu * (v_kms / C) ** 2 * 1e6
def elastic_emax_kev(A, m_chi, v_kms):
    mN = lz.m_nucleus_gev(A); mu = lz.mu_red(m_chi, mN)
    return 2 * mu ** 2 * (v_kms / C) ** 2 / mN * 1e6
def A_min_for_delta(delta, m_chi, v_kms):
    lo, hi = 1.0, 300.0
    if ceiling_kev(hi, m_chi, v_kms) < delta: return float('inf')
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if ceiling_kev(mid, m_chi, v_kms) < delta: lo = mid
        else: hi = mid
    return hi

DELTAS = [300.0, 350.0, 366.0, 380.0]
kin_rows = []
for sym, t in TARGETS.items():
    A = t['A']
    row = dict(target=sym, A=round(A, 1), note=t['note'])
    for m in (400.0, 1000.0, 4000.0):
        row[f'ceiling_keV_m{int(m)}'] = round(ceiling_kev(A, m, VMAX_JUNE), 1)
    row['ceiling_keV_1TeV_sunframe'] = round(ceiling_kev(A, 1000.0, V_SUN + VESC), 1)
    row['delta_max_248keV_1TeV_June'] = round(lz.delta_max_kev(E_EV, 1000.0, A=A, v_kms=VMAX_JUNE), 1)
    row['elastic_Emax_keV_1TeV_June'] = round(elastic_emax_kev(A, 1000.0, VMAX_JUNE), 1)
    row['vmin_248keV_elastic_1TeV_kms'] = round(lz.vmin_kms(E_EV, 1000.0, A=A), 1)
    for d in DELTAS:
        lo, hi = lz.E_R_range_keV(1000.0, VMAX_JUNE, A=A, delta_kev=d)
        row[f'window_d{int(d)}_keV'] = f'{lo:.0f}-{hi:.0f}' if not math.isnan(lo) else 'blind'
        row[f'live_d{int(d)}'] = bool(not math.isnan(lo))
    kin_rows.append(row)
A_min = {d: A_min_for_delta(d, 1000.0, VMAX_JUNE) for d in [250.0] + DELTAS}
log('A_min(delta) at 1 TeV, June: ' + ', '.join(f'{d:.0f} keV: A >= {a:.0f}' for d, a in A_min.items()))
import pandas as pd
df_kin = pd.DataFrame(kin_rows); df_kin.to_csv(f'{OUT}/P067_target_kinematics.csv', index=False)
log(df_kin[['target', 'A', 'ceiling_keV_m1000', 'delta_max_248keV_1TeV_June', 'elastic_Emax_keV_1TeV_June',
            'window_d300_keV', 'window_d366_keV']].to_string(index=False))

# --- track lengths (recalled SRIM-scale anchors; ALL flagged uncertain) ---------------------------------
# Range model: R(E) = R_100 * (E/100 keV)^0.8 in the anchor medium; gas ranges scale as 1/density.
rec('range exponent ~0.8 for 50-300 keV heavy ions (SRIM-like)', 0.8, 'SRIM systematics', 'uncertain')
R_ANCHORS = {   # medium: (R at 100 keV, unit, density note)
    'F in CF4 @ 40 Torr':      rec('F recoil range in CF4 at 40 Torr, 100 keV: ~2 mm (DMTPC ~1 mm at 75 Torr)', 2.0, 'DMTPC/SRIM', 'uncertain'),
    'F in He:SF6 (740:20 Torr)': 2.0 * (0.193 / 0.135),   # scaled by mass density ratio CF4(40 Torr)/He:SF6 mix (see below)
    'S in SF6 @ 20 Torr':      rec('S recoil in 40 Torr CS2, 100 keV: ~1.5 mm (DRIFT); SF6 at 20 Torr similar density -> ~1.5 mm', 1.5, 'DRIFT/SRIM', 'uncertain'),
    'Xe in LXe':               rec('Xe recoil range in LXe, 100 keV: ~0.07 um (projected; 0.05-0.1 um)', 0.07e-3, 'SRIM (LXe, 2.9 g/cm3)', 'uncertain'),
    'I in CF3I @ 40 Torr':     None,   # from LXe anchor scaled by density (I ~ Xe stopping)
    'Xe in Xe gas @ 40 Torr':  None,
    'Xe in Xe gas @ 10 bar':   None,
    'Ag in AgBr emulsion (NIT)': rec('Ag/Br recoil range in nuclear emulsion, 100 keV: ~0.1 um (NEWSdm: 100 nm ~ 30 keV C, ~100 keV Ag)', 0.10e-3, 'NEWSdm/SRIM', 'uncertain'),
    'W in CaWO4':              rec('W recoil range in CaWO4, 100 keV: ~0.03 um', 0.03e-3, 'SRIM', 'uncertain'),
}
# gas densities (ideal gas, 293 K): rho = P M / (R T)
def gas_density_kg_m3(P_torr, M_gmol, T_K=293.0):
    return (P_torr / 760.0) * M_gmol / (0.082057 * T_K)   # g/L = kg/m3
rec('ideal-gas densities at 293 K; molar masses CF4 88.0, SF6 146.1, CF3I 195.9, Xe 131.3, He 4.0 g/mol', 'formula', 'standard', 'certain')
RHO_LXE = rec('LXe density 2.9 g/cm3 (LZ conditions)', 2.9, 'LZ/NEST', 'likely')
rho = dict(CF4_40=gas_density_kg_m3(40, 88.0), SF6_20=gas_density_kg_m3(20, 146.1), He_740=gas_density_kg_m3(740, 4.0),
           CF3I_40=gas_density_kg_m3(40, 195.9), Xe_40=gas_density_kg_m3(40, 131.3), Xe_10bar=gas_density_kg_m3(7600, 131.3))
rho['HeSF6'] = rho['SF6_20'] + rho['He_740']
R_ANCHORS['F in He:SF6 (740:20 Torr)'] = 2.0 * rho['CF4_40'] / rho['HeSF6']
R_ANCHORS['Xe in Xe gas @ 40 Torr'] = R_ANCHORS['Xe in LXe'] * (RHO_LXE * 1000.0) / rho['Xe_40']
R_ANCHORS['Xe in Xe gas @ 10 bar'] = R_ANCHORS['Xe in LXe'] * (RHO_LXE * 1000.0) / rho['Xe_10bar']
R_ANCHORS['I in CF3I @ 40 Torr'] = R_ANCHORS['Xe in LXe'] * (RHO_LXE * 1000.0) / rho['CF3I_40']   # I ~ Xe stopping (uncertain)
rec('iodine stopping in CF3I gas approximated by xenon-in-xenon scaled by mass density', 'approx', 'Bragg/Lindhard scaling', 'uncertain')
THRESH = {'gas TPC (mm-scale 3D track)': 1.0, 'emulsion (NIT, 100 nm)': 0.1e-3, 'crystal/LXe (no track imaging)': float('inf')}
rec('directional thresholds: ~1 mm track for gas TPC head-tail, ~100 nm for NIT emulsions', 'thresholds', 'CYGNUS/NEWSdm design papers', 'likely')
track_rows = []
for med, R100 in R_ANCHORS.items():
    for E in (100.0, 250.0, 300.0):
        R = R100 * (E / 100.0) ** 0.8
        track_rows.append(dict(medium=med, E_keV=E, range_mm=R, range_um=R * 1e3))
df_track = pd.DataFrame(track_rows); df_track.to_csv(f'{OUT}/P067_track_lengths.csv', index=False)
log('gas densities kg/m3: ' + ', '.join(f'{k} {v:.3f}' for k, v in rho.items()))
log(df_track[df_track.E_keV == 250].to_string(index=False))

# ------------------------------------------------------------------------------------------------
# Part 2: recoil-direction Monte Carlo
# ------------------------------------------------------------------------------------------------
def sample_lab(n, vlab, vlo, rng):
    """Importance sampling of lab-frame DM velocities: speed uniform in [vlo, |vlab| + v_esc], isotropic direction;
    weight w0 = f(u) v^2 (v_max - vlo), u = v + vlab (Galactic frame), f = truncated Maxwellian exp(-u^2/v0^2) Theta(v_esc - |u|).
    Sum(w0) is proportional to the halo fraction above vlo; with vlo = 0 the full distribution is recovered."""
    vmax = float(np.linalg.norm(vlab)) + VESC
    v = rng.uniform(vlo, vmax, n)
    d = rng.normal(size=(n, 3)); d /= np.linalg.norm(d, axis=1)[:, None]
    vv = v[:, None] * d
    u2 = np.sum((vv + vlab) ** 2, axis=1)
    w0 = np.exp(-u2 / V0 ** 2) * (u2 < VESC ** 2) * v ** 2 * (vmax - vlo)
    return vv, w0

def perp_basis(vhat):
    a = np.where((np.abs(vhat[:, 0]) < 0.9)[:, None], np.array([1.0, 0, 0])[None, :], np.array([0, 1.0, 0])[None, :])
    e1 = np.cross(vhat, a); e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(vhat, e1)
    return e1, e2

def recoil_dirs(vvec, vmin, rng):
    """Recoil unit vectors for DM velocities vvec (lab) needing vmin: cos(theta_R) = vmin/|v|, uniform azimuth."""
    v = np.linalg.norm(vvec, axis=1); vhat = vvec / v[:, None]
    ct = np.clip(vmin / v, -1, 1); st = np.sqrt(1 - ct ** 2)
    e1, e2 = perp_basis(vhat); phi = rng.uniform(0, 2 * np.pi, len(v))
    return ct[:, None] * vhat + st[:, None] * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2), ct

def smear_dirs(q, sigma_deg, rng):
    """Rotate each direction by a Gaussian-distributed angle (sigma) about a random perpendicular axis."""
    alpha = np.abs(rng.normal(0.0, math.radians(sigma_deg), len(q)))
    e1, e2 = perp_basis(q); psi = rng.uniform(0, 2 * np.pi, len(q))
    return np.cos(alpha)[:, None] * q + np.sin(alpha)[:, None] * (np.cos(psi)[:, None] * e1 + np.sin(psi)[:, None] * e2)

SIGMA_RES = rec('angular resolution 30 deg and head-tail efficiency 70 % for a gas-TPC directional detector at 100-250 keV', '30 deg / 0.70', 'CYGNUS/DRIFT/DMTPC performance papers', 'uncertain (assumption)')
SIG_DEG, EPS_HT = 30.0, 0.70

def analyse(q, w, label, cases_out, rng, ct_R=None, extra=None):
    """Anisotropy statistics for weighted recoil directions q (Galactic axes) relative to W_HAT."""
    w = w / w.sum()
    cg = q @ W_HAT
    res = dict(case=label, n_mc=len(q))
    res['mean_cos_gamma'] = float(np.sum(w * cg))
    res['A_FB'] = float(np.sum(w * np.sign(cg)))
    res['frac_gamma_lt30'] = float(np.sum(w * (cg > math.cos(math.radians(30)))))
    res['frac_gamma_lt60'] = float(np.sum(w * (cg > 0.5)))
    res['mean_cos2_gamma'] = float(np.sum(w * cg ** 2))
    # weighted quantiles of gamma
    g = np.degrees(np.arccos(np.clip(cg, -1, 1))); o = np.argsort(g); cw = np.cumsum(w[o])
    res['gamma_median_deg'] = float(g[o][np.searchsorted(cw, 0.5)]); res['gamma_90pct_deg'] = float(g[o][np.searchsorted(cw, 0.9)])
    if ct_R is not None:
        th = np.degrees(np.arccos(np.clip(ct_R, -1, 1))); o2 = np.argsort(th); cw2 = np.cumsum(w[o2])
        res['thetaR_median_deg'] = float(th[o2][np.searchsorted(cw2, 0.5)]); res['thetaR_max_deg'] = float(th.max())
    # detector response: smear + head-tail flips
    qs = smear_dirs(q, SIG_DEG, rng)
    flip = rng.uniform(size=len(q)) > EPS_HT
    qo = np.where(flip[:, None], -qs, qs)
    cgo = qo @ W_HAT; cgs = qs @ W_HAT
    res['mean_cos_obs'] = float(np.sum(w * cgo)); res['A_FB_obs'] = float(np.sum(w * np.sign(cgo)))
    res['mean_cos2_axial_obs'] = float(np.sum(w * cgs ** 2))
    # analytic Gaussian N for Z = 3 (null: isotropy; var(cos) = 1/3, var(sign) = 1, var(cos^2) = 4/45)
    res['N3_ideal_dipole'] = 3.0 / res['mean_cos_gamma'] ** 2
    res['N3_ideal_FB'] = 9.0 / res['A_FB'] ** 2
    res['N3_obs_dipole'] = 3.0 / res['mean_cos_obs'] ** 2
    res['N3_obs_FB'] = 9.0 / res['A_FB_obs'] ** 2
    res['N3_obs_axial'] = 9.0 * (4.0 / 45.0) / (res['mean_cos2_axial_obs'] - 1.0 / 3.0) ** 2
    res['N5_obs_dipole'] = 25.0 / 3.0 / res['mean_cos_obs'] ** 2
    if extra: res.update(extra)
    cases_out.append(res)
    hist, edges = np.histogram(cg, bins=20, range=(-1, 1), weights=w, density=True)
    hist_o, _ = np.histogram(cgo, bins=20, range=(-1, 1), weights=w, density=True)
    HISTS[label] = (edges, hist, hist_o)
    log(f"{label:34s} <cos g>={res['mean_cos_gamma']:+.3f} A_FB={res['A_FB']:+.3f} f(<30deg)={res['frac_gamma_lt30']:.2f} "
        f"g50={res['gamma_median_deg']:.0f} g90={res['gamma_90pct_deg']:.0f} | obs <cos>={res['mean_cos_obs']:+.3f} "
        f"N3(dipole)={res['N3_obs_dipole']:.1f} N3(FB)={res['N3_obs_FB']:.1f} N3(axial)={res['N3_obs_axial']:.1f}"
        + (f" | thetaR med/max {res['thetaR_median_deg']:.1f}/{res['thetaR_max_deg']:.1f}" if ct_R is not None else ''))
    return res, (q, w, qo)

HISTS = {}; CASES = []; KEEP = {}
N_MC = 200_000     # per case (reduced from 400k at the coordinator's request; statistical precision on <cos gamma> ~ 1e-3)
for frame, vlab in (('June', V_LAB_GAL), ('Sun', V_SUN_GAL)):
    _, w0_full = sample_lab(N_MC, vlab, 0.0, rng); norm_full = w0_full.sum()
    # (i) fixed E_R = 248 keV, inelastic and elastic, Xe and I
    for A_sym in ('Xe', 'I'):
        A = TARGETS[A_sym]['A']
        for d in [0.0] + DELTAS:
            vmin = lz.vmin_kms(E_EV, M_CHI, A=A, delta_kev=d)
            if vmin >= np.linalg.norm(vlab) + VESC - 0.5:
                log(f'{frame} {A_sym} delta={d:.0f}: vmin={vmin:.0f} km/s >= v_max -> kinematically forbidden'); continue
            vv, w0 = sample_lab(N_MC, vlab, vmin, rng)
            speed = np.linalg.norm(vv, axis=1); sel = w0 > 0
            q, ct = recoil_dirs(vv[sel], vmin, rng); w = w0[sel] / speed[sel]
            lab = f'{frame} {A_sym} E=248 delta={d:.0f}'
            r, keep = analyse(q, w, lab, CASES, rng, ct_R=ct, extra=dict(vmin_kms=vmin, frac_halo_above_vmin=float(w0.sum() / norm_full), n_eff=int(sel.sum()),
                                                                          thetaR_max_analytic_deg=math.degrees(math.acos(min(1, vmin / (np.linalg.norm(vlab) + VESC))))))
            if frame == 'June' and A_sym == 'Xe' and d in (0.0, 366.0): KEEP[lab] = keep
    if frame == 'Sun':
        break

# (ii) elastic L10 spectra on F and Xe, and inelastic O1 spectrum on Xe at delta = 366 keV, via WimPyDD kernels
WD = lz.wd()
MV2 = lz.M_V_GEV ** 2; MN = lz.M_NUCLEON_GEV
c4 = lz.wd_c_from_anand(4 / MV2)[0]; c6 = lz.wd_c_from_anand(-4 / MV2)[0]
h_L10 = WD.eft_hamiltonian('P067_L10', {(4, 'q2'): lambda q, A=c4: [A * q ** 2 / MN ** 2, 0.0], 6: lambda A6=c6: [A6, 0.0]})
h_O1 = lz.wd_hamiltonian('P067_O1s', {1: lz.wd_c_from_anand(1.0 / MV2)})
FLAT = (np.array([0.0, 3000.0]), np.array([0.0, 1.0]))   # eta = 1 for all v_min: diff_rate returns the velocity-independent kernel K(E)
halo_sun = lz.wd_halo()
E_F = np.arange(0.5, 260.0, 0.5); E_XE = np.arange(1.0, 700.0, 1.0)
K_F_L10 = lz.wd_rate(h_L10, M_CHI, E_F, FLAT, target=WD.F)
K_XE_L10 = lz.wd_rate(h_L10, M_CHI, E_XE, FLAT)
K_XE_O1_366 = lz.wd_rate(h_O1, M_CHI, E_XE, FLAT, delta_kev=366.0)
# validation of the kernel trick: K * eta0 vs direct WimPyDD Sun-frame rate on F at a few energies
E_chk = np.array([10.0, 50.0, 100.0, 150.0, 200.0])
r_direct = lz.wd_rate(h_L10, M_CHI, E_chk, halo_sun, target=WD.F)
r_kernel = np.interp(E_chk, E_F, K_F_L10) * lz.eta0(np.array([lz.vmin_kms(e, M_CHI, A=19.0) for e in E_chk]), v_e=V_SUN)
log('kernel validation F L10 (K*eta0 / direct): ' + np.array2string(r_kernel / r_direct, precision=3))
np.savetxt(f'{OUT}/P067_kernels.csv', np.c_[E_XE, K_XE_L10, K_XE_O1_366, np.interp(E_XE, E_F, K_F_L10, right=0.0)],
           delimiter=',', header='E_keV,K_Xe_L10,K_Xe_O1_delta366,K_F_L10 (velocity-independent kernels, unit coupling, arbitrary norm)', comments='')

def spectrum_mc(A, Kgrid_E, Kgrid, delta, vlab, label, window=None, rng=rng, extra=None):
    """Joint (v, E_R) importance sampling: E uniform in the allowed window at each v, weight (E+ - E-)/v * K(E)."""
    mN = lz.m_nucleus_gev(A); mu = lz.mu_red(M_CHI, mN); d = delta * 1e-6
    vlo = math.sqrt(2 * d / mu) * C if delta > 0 else 0.0          # threshold speed for any recoil at this delta
    vv, w0 = sample_lab(N_MC, vlab, vlo, rng); speed = np.linalg.norm(vv, axis=1); beta = speed / C
    disc = 1 - 2 * d / (mu * beta ** 2); ok = disc > 0
    pref = mu ** 2 * beta ** 2 / mN; mid = 1 - d / (mu * beta ** 2)
    Elo = np.where(ok, pref * (mid - np.sqrt(np.abs(disc))) * 1e6, 0.0); Ehi = np.where(ok, pref * (mid + np.sqrt(np.abs(disc))) * 1e6, 0.0)
    E = rng.uniform(Elo, Ehi); w = w0 * (Ehi - Elo) / speed * np.interp(E, Kgrid_E, Kgrid, left=0.0, right=0.0) * ok
    if window is not None:
        w = w * ((E >= window[0]) & (E <= window[1]))
    sel = w > 0
    vmin = np.array([lz.vmin_kms(e, M_CHI, A=A, delta_kev=delta) for e in E[sel]]) if sel.sum() < 50000 else \
        (mN * E[sel] * 1e-6 / mu + d) / np.sqrt(2 * mN * E[sel] * 1e-6) * C
    q, ct = recoil_dirs(vv[sel], vmin, rng)
    # weighted energy quantiles
    Es = E[sel]; ws = w[sel] / w[sel].sum(); o = np.argsort(Es); cw = np.cumsum(ws[o])
    ex = dict(E_median_keV=float(Es[o][np.searchsorted(cw, 0.5)]), E_10_keV=float(Es[o][np.searchsorted(cw, 0.1)]),
              E_90_keV=float(Es[o][np.searchsorted(cw, 0.9)]))
    if extra: ex.update(extra)
    r, keep = analyse(q, w[sel], label, CASES, rng, ct_R=ct, extra=ex)
    return r, keep, (E, w)

r, keep, (E_all_F, w_all_F) = spectrum_mc(19.0, E_F, K_F_L10, 0.0, V_LAB_GAL, 'June F L10 elastic E>20 keV', window=(20.0, 1e4)); KEEP['F L10 >20'] = keep
spectrum_mc(19.0, E_F, K_F_L10, 0.0, V_LAB_GAL, 'June F L10 elastic E>50 keV', window=(50.0, 1e4))
spectrum_mc(19.0, E_F, K_F_L10, 0.0, V_LAB_GAL, 'June F L10 elastic 150-250 keV', window=(150.0, 250.0))
spectrum_mc(lz.A_XE_MEAN, E_XE, K_XE_L10, 0.0, V_LAB_GAL, 'June Xe L10 elastic 200-270 keV', window=(200.0, 270.0))
spectrum_mc(lz.A_XE_MEAN, E_XE, K_XE_L10, 0.0, V_LAB_GAL, 'June Xe L10 elastic E>5.4 keV', window=(5.4, 1e4))
r366, keep366, _ = spectrum_mc(lz.A_XE_MEAN, E_XE, K_XE_O1_366, 366.0, V_LAB_GAL, 'June Xe O1 inelastic d=366 full window'); KEEP['Xe d366 spectrum'] = keep366
spectrum_mc(lz.A_XE_MEAN, E_XE, K_XE_O1_366, 366.0, V_LAB_GAL, 'June Xe O1 inelastic d=366 200-270 keV', window=(200.0, 270.0))
# isotropic control (must give ~0 anisotropy)
q_iso = rng.normal(size=(200000, 3)); q_iso /= np.linalg.norm(q_iso, axis=1)[:, None]
analyse(q_iso, np.ones(len(q_iso)), 'isotropic control', CASES, rng)
pd.DataFrame(CASES).to_csv(f'{OUT}/P067_angular_cases.csv', index=False)

# MC spectrum validation: dR/dE from the joint sampler vs K*eta0 for F L10 (shape)
hE, edges = np.histogram(E_all_F, bins=np.arange(0, 260, 10.0), weights=w_all_F)   # weights already include the halo importance factor
cen = 0.5 * (edges[1:] + edges[:-1])
ref = np.interp(cen, E_F, K_F_L10) * lz.eta0(np.array([lz.vmin_kms(e, M_CHI, A=19.0) for e in cen]), v_e=V_LAB)
ratio = (hE / hE.sum()) / (ref / ref.sum())
log('MC spectrum / (K*eta0) for F L10 (10 keV bins, 5-245 keV): ' + np.array2string(ratio, precision=2, max_line_width=200))

# --- toy-MC check of N_3sigma for the two headline cases (median discovery, statistic = mean cos gamma) -------
def toy_N3(qo, w, rng, Ns=(5, 8, 10, 15, 20, 30, 50, 80, 120), ntoy=1500):
    p = w / w.sum(); cgo = qo @ W_HAT
    out = {}
    for N in Ns:
        # null distribution of mean cos for N isotropic events (100k toys; reduced statistics)
        null = rng.uniform(-1, 1, size=(100000, N)).mean(axis=1); crit = np.quantile(null, 1 - 1.35e-3)
        sig = np.array([np.mean(rng.choice(cgo, size=N, p=p)) for _ in range(ntoy)])
        out[N] = float(np.mean(sig > crit))
    return out
TOYS = {}
for lab in ('June Xe E=248 delta=366', 'F L10 >20', 'June Xe E=248 delta=0'):
    q, w, qo = KEEP[lab]
    TOYS[lab] = toy_N3(qo, w, rng)
    log(f'toy power (P[Z>3]) vs N for {lab}: ' + ', '.join(f'{k}:{v:.2f}' for k, v in TOYS[lab].items()))

# ------------------------------------------------------------------------------------------------
# Part 3: SURF geometry at the event time
# ------------------------------------------------------------------------------------------------
LAT = rec('SURF Davis campus latitude 44.352 N, longitude 103.751 W', 44.352, 'geography (as P042)', 'likely'); LON = -103.751
rec('GMST = 18.697374558 + 24.06570982441908 D hours (D = JD - 2451545.0), UT1 ~ UTC to < 1 s', 'formula', 'USNO', 'certain')
def lst_deg(t):
    D = t.jd - 2451545.0
    return ((18.697374558 + 24.06570982441908 * D) % 24.0) * 15.0 + LON
def altaz_deg(ra_deg, dec_deg, lst):
    ha = np.radians(lst - ra_deg); lat = math.radians(LAT); dec = np.radians(dec_deg)
    alt = np.degrees(np.arcsin(np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(ha)))
    az = np.degrees(np.arctan2(-np.sin(ha) * np.cos(dec), np.sin(dec) * math.cos(lat) - np.cos(dec) * math.sin(lat) * np.cos(ha))) % 360.0
    return alt, az
LST_EV = lst_deg(T_EV) % 360.0
ra_ap, dec_ap = gal_to_icrs_radec(V_LAB_GAL[None, :])        # apex: direction the lab moves toward = where DM comes FROM
ra_aa, dec_aa = gal_to_icrs_radec(-V_LAB_GAL[None, :])       # anti-apex: where recoils/wind go
alt_ap, az_ap = altaz_deg(ra_ap[0], dec_ap[0], LST_EV); alt_aa, az_aa = altaz_deg(ra_aa[0], dec_aa[0], LST_EV)
cyg = SkyCoord(ra=312.5 * u.deg, dec=45.0 * u.deg, frame='icrs')   # 'Cygnus' as commonly quoted: RA 20h50m, Dec +45
rec('Cygnus / solar apex direction ~ RA 20h50m, Dec +45 deg (l ~ 90, b ~ 0)', 'RA 312.5 Dec 45', 'standard', 'likely')
sep_cyg = float(cyg.separation(SkyCoord(ra=ra_ap[0] * u.deg, dec=dec_ap[0] * u.deg, frame='icrs')).deg)
l_ap, b_ap = SkyCoord(ra=ra_ap[0] * u.deg, dec=dec_ap[0] * u.deg, frame='icrs').galactic.l.deg, SkyCoord(ra=ra_ap[0] * u.deg, dec=dec_ap[0] * u.deg, frame='icrs').galactic.b.deg
log(f'LST(SURF) = {LST_EV/15:.3f} h; apex RA {ra_ap[0]:.1f} Dec {dec_ap[0]:.1f} (l={l_ap:.1f}, b={b_ap:.1f}; {sep_cyg:.1f} deg from RA 312.5/Dec 45): '
    f'alt {alt_ap:+.1f} az {az_ap:.1f}; anti-apex RA {ra_aa[0]:.1f} Dec {dec_aa[0]:.1f}: alt {alt_aa:+.1f} az {az_aa:.1f} (az from N through E)')
# apex altitude over 16 June 2023 (UTC) and the fraction of the sidereal day with |alt| < 30 deg
hours = np.linspace(0, 24, 289)
alts = np.array([altaz_deg(ra_ap[0], dec_ap[0], lst_deg(Time('2023-06-16 00:00:00', scale='utc') + h * u.hour) % 360.0)[0] for h in hours])
azs = np.array([altaz_deg(ra_ap[0], dec_ap[0], lst_deg(Time('2023-06-16 00:00:00', scale='utc') + h * u.hour) % 360.0)[1] for h in hours])
np.savetxt(f'{OUT}/P067_apex_altaz_16June2023.csv', np.c_[hours, alts, azs], delimiter=',', header='UTC_hour,apex_alt_deg,apex_az_deg', comments='')
log(f'apex altitude range on 16 June 2023: {alts.min():+.1f} to {alts.max():+.1f} deg; below the horizon {np.mean(alts<0)*100:.0f} % of the day; '
    f'|alt| < 20 deg {np.mean(np.abs(alts)<20)*100:.0f} % of the day')
# expected recoil directions of the event in the SURF horizon frame
def dirs_to_altaz(q_gal):
    v_icrs = (R_G2I @ q_gal.T).T
    ra = np.degrees(np.arctan2(v_icrs[:, 1], v_icrs[:, 0])) % 360.0; dec = np.degrees(np.arcsin(np.clip(v_icrs[:, 2], -1, 1)))
    return altaz_deg(ra, dec, LST_EV)
GEO = {}
for lab in ('June Xe E=248 delta=366', 'June Xe E=248 delta=0', 'F L10 >20'):
    q, w, qo = KEEP[lab]; p = w / w.sum()
    alt, az = dirs_to_altaz(q)
    o = np.argsort(alt); cw = np.cumsum(p[o]); alt_q = [float(alt[o][np.searchsorted(cw, x)]) for x in (0.1, 0.5, 0.9)]
    # circular statistics for azimuth
    azr = np.radians(az); az_mean = math.degrees(math.atan2(np.sum(p * np.sin(azr)), np.sum(p * np.cos(azr)))) % 360
    daz = ((az - az_mean + 180) % 360) - 180; o2 = np.argsort(np.abs(daz)); cw2 = np.cumsum(p[o2]); az68 = float(np.abs(daz)[o2][np.searchsorted(cw2, 0.68)])
    up = float(np.sum(p * (alt > 0)))
    GEO[lab] = dict(alt_10_50_90=alt_q, az_mean_deg=az_mean, az_68pct_halfwidth_deg=az68, frac_upward=up,
                    frac_within_30deg_of_antiapex=float(np.sum(p * ((q @ W_HAT) > math.cos(math.radians(30))))))
    log(f'{lab}: recoil alt 10/50/90 % = {alt_q[0]:+.0f}/{alt_q[1]:+.0f}/{alt_q[2]:+.0f} deg; az mean {az_mean:.0f} deg (68 % half-width {az68:.0f}); upward fraction {up:.2f}')

# ------------------------------------------------------------------------------------------------
# Part 4: rates at LZ's best fit and the decision table
# ------------------------------------------------------------------------------------------------
# L10 normalisation: LZ ROI count at unit coupling (Sun-frame halo as in P003/P012), simple efficiency model
def eff_lz(E):
    return 0.96 / (1 + np.exp(-(E - 5.4) / 1.5)) / (1 + np.exp((E - 269.9) / 8.0))
rec('LZ efficiency: 50 % at 5.4 and 269.9 keV, 96 % plateau (paper); logistic roll-off widths 1.5/8 keV assumed', 'model', 'LZ paper Fig. S2 / P002', 'likely')
R_xe_L10 = lz.wd_rate(h_L10, M_CHI, E_XE[E_XE <= 320], halo_sun)             # events / t yr keV, unit coupling
N_unit = lz.LZ['exposure_tyr'] * np.trapezoid(R_xe_L10 * eff_lz(E_XE[E_XE <= 320]), E_XE[E_XE <= 320]) if 'exposure_tyr' in lz.LZ else None
if N_unit is None:
    N_unit = 2.84 * np.trapezoid(R_xe_L10 * eff_lz(E_XE[E_XE <= 320]), E_XE[E_XE <= 320])
scale_L10 = 1.0 / N_unit
log(f'L10 unit-coupling LZ-ROI count in 2.84 t yr: {N_unit:.2f} (P012 "ours": 3.34) -> best-fit scale {scale_L10:.3f}')
R_F_L10_sun = lz.wd_rate(h_L10, M_CHI, E_F, halo_sun, target=WD.F) * scale_L10       # events / t yr keV at LZ best fit
R_XE_L10_bf = R_xe_L10 * scale_L10
rates_F = {thr: float(np.trapezoid(R_F_L10_sun[E_F >= thr], E_F[E_F >= thr])) for thr in (10.0, 20.0, 50.0, 100.0, 150.0)}
frac_F_window = rates_F[150.0] / rates_F[10.0]
log('F L10 rate at LZ best fit per tonne-year of F above 10/20/50/100/150 keV: ' + ', '.join(f'{k:.0f}: {v:.3f}' for k, v in rates_F.items()))
log(f'Xe L10 rate per t yr (E > 5.4 keV, no efficiency): {np.trapezoid(R_XE_L10_bf[E_XE[E_XE<=320] >= 5.4], E_XE[(E_XE<=320) & (E_XE>=5.4)]):.3f}; '
    f'F/Xe per tonne (E>20 keV) = {rates_F[20.0] / np.trapezoid(R_XE_L10_bf[E_XE[E_XE<=320] >= 5.4], E_XE[(E_XE<=320) & (E_XE>=5.4)]):.3f}')
# fluorine masses of example directional volumes (recalled designs, flagged)
rec('CYGNUS-1000: 1000 m3 He:SF6 at 740:20 Torr (design option); pure CF4 at 40 Torr as alternative', 'design', 'CYGNUS white paper (Vahsen et al. 2020)', 'uncertain')
m_F_HeSF6 = rho['SF6_20'] * 1000.0 * (6 * 19.0 / 146.1)      # kg F in 1000 m3 He:SF6
m_F_CF4 = rho['CF4_40'] * 1000.0 * (4 * 19.0 / 88.0)          # kg F in 1000 m3 CF4 @ 40 Torr
m_Xe_gas = rho['Xe_40'] * 1000.0                              # kg Xe in 1000 m3 @ 40 Torr
m_I_CF3I = rho['CF3I_40'] * 1000.0 * (126.9 / 195.9)          # kg I in 1000 m3 CF3I @ 40 Torr
log(f'1000 m3 target masses: F {m_F_HeSF6:.0f} kg (He:SF6) / {m_F_CF4:.0f} kg (CF4 40 Torr); Xe {m_Xe_gas:.0f} kg (40 Torr); I {m_I_CF3I:.0f} kg (CF3I 40 Torr)')
# inelastic rates: xenon full-window rate at LZ's fit from P015's ROI fractions (LZ ROI captures 82/45/17/2.8 % of the window)
FRAC_ROI = {300.0: 0.82, 350.0: 0.45, 366.0: 0.17, 380.0: 0.028}
I_OVER_XE = {300.0: 0.59, 350.0: 0.30, 366.0: 0.14, 380.0: 0.023}   # P015/P046 per tonne
W_OVER_XE = {300.0: 15.9, 350.0: 81.7, 366.0: 221.0, 380.0: 693.0}
xe_full = {d: 1.0 / (2.84 * f) for d, f in FRAC_ROI.items()}        # events per t yr, full window, best fit
# Ag/Xe at delta = 300 keV with Helm form factors on both (lzcommon dRdE_SI), June halo
def helm_window_rate(A, d, v_e):
    E = np.linspace(1.0, 1500.0, 1500)
    r = np.array([lz.dRdE_SI(e, M_CHI, 1e-45, A=A, delta_kev=d, v_e=v_e) for e in E])
    return float(np.trapezoid(np.nan_to_num(r), E))
ag_xe = {d: helm_window_rate(107.9, d, V_LAB) / helm_window_rate(lz.A_XE_MEAN, d, V_LAB) for d in (250.0, 300.0, 320.0)}
xe_helm_june_over_sun = helm_window_rate(lz.A_XE_MEAN, 300.0, V_LAB) / helm_window_rate(lz.A_XE_MEAN, 300.0, V_SUN)
log('Ag/Xe full-window rate per tonne (Helm both, June): ' + ', '.join(f'd={d:.0f}: {v:.3g}' for d, v in ag_xe.items()) +
    f'; Xe June/Sun-frame at 300 keV (Helm) = {xe_helm_june_over_sun:.2f}')
rec('NEWSdm nuclear-emulsion (NIT) programme: AgBr-C-N-O gel, 10 kg yr pilot to ~100 kg yr; Ag+Br ~ 78 % of the mass', 'design', 'NEWSdm LOI/papers', 'uncertain')
m_Ag_100kg = 100.0 * 0.78 * (107.9 / (107.9 + 79.9))          # kg Ag in 100 kg of emulsion (AgBr fraction 78 %, mass-weighted)

# directional N_3sigma lookups
def getcase(lab): return [c for c in CASES if c['case'] == lab][0]
N3_inel = {d: getcase(f'June Xe E=248 delta={d:.0f}')['N3_obs_dipole'] for d in DELTAS}
N3_inel_spec366 = getcase('June Xe O1 inelastic d=366 full window')['N3_obs_dipole']
N3_F = getcase('June F L10 elastic E>20 keV')['N3_obs_dipole']
N3_F50 = getcase('June F L10 elastic E>50 keV')['N3_obs_dipole']
N3_Xe_el = getcase('June Xe L10 elastic E>5.4 keV')['N3_obs_dipole']
rows = []
def add(strategy, target, observable, exposure, n_events, years, source, notes=''):
    rows.append(dict(strategy=strategy, target=target, observable=observable, exposure_for_3sigma=exposure, events_needed=n_events,
                     time_or_date=years, source=source, notes=notes))
add('Keep counting (present generation, LZ-like ROI)', 'Xe (LZ 4.7 t; XENONnT, PandaX-4T)', 'high-energy NR count vs <1e-3 bkg',
    '5 sigma at best fit: 12.3 (L10) - 5.7 (d380) t yr', '2-4', '2027-2028 (P050); untouched LZ data already 6.8 t yr (P020)', 'P050, P020')
add('Extend ROI to 400 keV (present generation)', 'Xe', 'count + spectral shape', 'acceptance 0.83-0.93 for d>=350', '2-4', 'by ~2030 (P050)', 'P050, P038')
add('60 t xenon, 400 keV ROI', 'Xe (XLZD-class)', 'count; L10 vs inelastic shape', '5 sigma: 8.5/9.6/3.6/1.1/0.13 t yr (L10/300/350/366/380)', '4-6 for shape', '2032 + 1 day to 2 months live', 'P050')
add('Annual modulation', 'Xe', 'June clustering vs flat', '3 sigma: 274/84/33/19/15 t yr (d=300/330/350/366/380)', '97/29/12/6.5/5.3', 'LZ 2029-30 only if d>=366; 60 t 0.3-5.7 yr', 'P034')
add('CaWO4 delta-meter', 'W (CRESST-like, 700 keV ROI)', 'W/Xe count ratio 16-693', '100 kg yr + 20 t yr Xe: sigma_delta 99/10/3.5/0.8 keV', '3 (2.1 kg yr for L10 vs d366)', 'kg yr-scale exposures: few years', 'P046, P015')
add('Isotopically enriched xenon', '90 % 136Xe', 'spin (x0.106) vs inelastic (x1.0) rate', '12.8 t yr per detector (natural + enriched)', '9 (5 at d=380)', 'nEXO-grade xenon; 2030s', 'P047')
add('NaI modulation', 'I (DAMA/COSINE/ANAIS)', 'S_m 1e-7 cpd/kg/keV', '2e8-8e9 t yr DAMA-like', '-', 'excluded as a route', 'P028, P015')
for d in DELTAS:
    ev_yr = xe_full[d] * m_Xe_gas / 1000.0
    add(f'Directional heavy-gas TPC (Xe, 40 Torr, 1000 m3 = {m_Xe_gas:.0f} kg), d={d:.0f}', 'Xe gas', 'recoil dipole toward anti-apex (30 deg, 70 % HT)',
        f'{N3_inel[d]/ev_yr:.0f} yr at {ev_yr:.2f} ev/yr', f'{N3_inel[d]:.0f}', f'{N3_inel[d]/ev_yr:.0f} years', 'this work (rate: P015/P050 xenon window fractions)',
        f'recoils within {getcase(f"June Xe E=248 delta={d:.0f}")["gamma_90pct_deg"]:.0f} deg of the wind for 90 %')
    ev_yr_I = xe_full[d] * I_OVER_XE[d] * m_I_CF3I / 1000.0
    add(f'Directional CF3I TPC (40 Torr, 1000 m3 = {m_I_CF3I:.0f} kg I), d={d:.0f}', 'I gas', 'recoil dipole', f'{N3_inel[d]/ev_yr_I:.0f} yr at {ev_yr_I:.3f} ev/yr',
        f'{N3_inel[d]:.0f}', f'{N3_inel[d]/ev_yr_I:.0f} years', 'this work; I/Xe from P015/P046')
ev_yr_Ag = xe_full[300.0] * ag_xe[300.0] * m_Ag_100kg / 1000.0
add('Nuclear emulsion (NEWSdm-like, 100 kg yr, Ag), d=300', 'Ag (ceiling 332 keV; Br blind)', 'sub-um track direction', f'{ev_yr_Ag:.1e} events per 100 kg yr',
    '-', 'not viable; blind for d > 332 keV', 'this work (Helm Ag/Xe)')
for lab, N3, thr in (('F L10 >20', N3_F, 20.0), ('F L10 >50', N3_F50, 50.0)):
    for mF, name in ((m_F_HeSF6, 'He:SF6'), (m_F_CF4, 'CF4 40 Torr')):
        ev_yr = rates_F[thr] * mF / 1000.0
        add(f'Directional fluorine TPC (1000 m3 {name}, {mF:.0f} kg F, E>{thr:.0f} keV), L10', 'F gas', 'recoil dipole (30 deg, 70 % HT)',
            f'{N3/ev_yr:.0f} yr at {ev_yr:.3f} ev/yr', f'{N3:.0f}', f'{N3/ev_yr:.0f} years; x{N3/ev_yr:.0f} target mass ({mF*N3/ev_yr/1000:.0f} t F) for ~1 yr', 'this work (WimPyDD L10 on 19F)')
pd.DataFrame(rows).to_csv(f'{OUT}/P067_decision_table.csv', index=False)
log(pd.DataFrame(rows)[['strategy', 'exposure_for_3sigma', 'events_needed', 'time_or_date']].to_string(index=False, max_colwidth=70))

# ------------------------------------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------------------------------------
COL = {'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73', 'red': '#D55E00', 'purple': '#CC79A7', 'grey': '#7f7f7f', 'sky': '#56B4E9'}
# Fig 1: inelastic ceiling vs A and track length vs detectability
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
Agrid = np.linspace(4, 200, 400)
for m, c, ls in ((400.0, COL['sky'], '--'), (1000.0, COL['blue'], '-'), (4000.0, COL['purple'], ':')):
    axs[0].plot(Agrid, [ceiling_kev(a, m, VMAX_JUNE) for a in Agrid], color=c, ls=ls, lw=2, label=f'ceiling mu v_max^2/2, {m/1000:g} TeV (16 June)')
for d, c in ((300, COL['green']), (366, COL['orange']), (380, COL['red'])):
    axs[0].axhline(d, color=c, lw=1, ls='-.'); axs[0].text(6, d + 5, f'delta = {d} keV', color=c, fontsize=8)
for sym, t in TARGETS.items():
    axs[0].plot(t['A'], ceiling_kev(t['A'], 1000.0, VMAX_JUNE), 'o', color='k', ms=4); axs[0].text(t['A'] + 2, ceiling_kev(t['A'], 1000.0, VMAX_JUNE) - 22, sym, fontsize=8)
axs[0].set_xlabel('mass number A'); axs[0].set_ylabel('largest accessible splitting [keV]'); axs[0].set_ylim(0, 620); axs[0].legend(fontsize=7.5, loc='upper left')
axs[0].set_title('(a) Who can scatter at all', fontsize=10)
meds = list(R_ANCHORS.keys()); ypos = np.arange(len(meds))
r250 = [R_ANCHORS[m] * 2.5 ** 0.8 for m in meds]
axs[1].barh(ypos, r250, color=[COL['blue'] if v >= 1.0 else (COL['orange'] if v >= 1e-4 else COL['grey']) for v in r250])
axs[1].set_xscale('log'); axs[1].set_yticks(ypos); axs[1].set_yticklabels(meds, fontsize=8)
axs[1].axvline(1.0, color=COL['blue'], ls='--', lw=1); axs[1].text(1.05, len(meds) - 0.8, 'gas-TPC head-tail (~1 mm)', fontsize=7.5, color=COL['blue'])
axs[1].axvline(1e-4, color=COL['orange'], ls='--', lw=1); axs[1].text(1.1e-4, len(meds) - 0.8, 'emulsion (100 nm)', fontsize=7.5, color=COL['orange'])
axs[1].set_xlabel('recoil range at 250 keV [mm] (recalled SRIM-scale anchors, factor ~2 uncertain)'); axs[1].set_title('(b) Whose track can be imaged', fontsize=10)
axs[1].set_xlim(1e-5, 30)
fig.tight_layout(); fig.savefig(f'{FIG}/P067_fig1_ceiling_and_tracks.png', dpi=160); plt.close(fig)

# Fig 2: angular distributions
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
for lab, c in (('June Xe E=248 delta=380', COL['red']), ('June Xe E=248 delta=366', COL['orange']), ('June Xe E=248 delta=300', COL['green']),
               ('June Xe E=248 delta=0', COL['blue']), ('June F L10 elastic E>20 keV', COL['purple'])):
    edges, h, ho = HISTS[lab]; cen = 0.5 * (edges[1:] + edges[:-1])
    axs[0].step(cen, h, where='mid', color=c, lw=2, label=lab.replace('June ', ''))
    axs[1].step(cen, ho, where='mid', color=c, lw=2, label=lab.replace('June ', ''))
for ax in axs:
    ax.axhline(0.5, color='k', ls=':', lw=1); ax.set_xlabel('cos(gamma): recoil direction vs DM wind (anti-apex = +1)'); ax.set_ylabel('dR/dcos(gamma) (normalised)')
axs[0].set_title('(a) True recoil directions, 16 June 2023', fontsize=10); axs[1].set_title('(b) With 30 deg resolution and 70 % head-tail', fontsize=10)
axs[0].legend(fontsize=7.5); axs[0].set_yscale('log'); axs[0].set_ylim(1e-2, 30); axs[1].set_ylim(0, 3.2)
fig.tight_layout(); fig.savefig(f'{FIG}/P067_fig2_angular.png', dpi=160); plt.close(fig)

# Fig 3: SURF geometry
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
axs[0].plot(hours, alts, color=COL['blue'], lw=2, label='DM-wind apex (Cygnus) altitude at SURF')
axs[0].axhline(0, color='k', lw=1); axs[0].axvline(21 + 22 / 60 + 39 / 3600, color=COL['red'], ls='--', lw=1.5, label='event 21:22:39 UTC')
axs[0].set_xlabel('UTC hour, 16 June 2023'); axs[0].set_ylabel('altitude [deg]'); axs[0].set_xlim(0, 24); axs[0].legend(fontsize=8)
axs[0].set_title(f'(a) Apex alt {alt_ap:+.1f} deg, az {az_ap:.0f} deg at the event', fontsize=10)
q, w, qo = KEEP['June Xe E=248 delta=366']; alt, az = dirs_to_altaz(q); p = w / w.sum()
q2, w2, _ = KEEP['June Xe E=248 delta=0']; alt2, az2 = dirs_to_altaz(q2); p2 = w2 / w2.sum()
axs[1].hist2d(az2, alt2, bins=[72, 36], range=[[0, 360], [-90, 90]], weights=p2, cmap='Greys')
idx = rng.choice(len(q), size=4000, p=p)
axs[1].scatter(az[idx], alt[idx], s=2, color=COL['orange'], alpha=0.5, label='inelastic delta = 366 keV (E_R = 248 keV)')
axs[1].plot(az_aa, alt_aa, '*', color=COL['red'], ms=12, label=f'anti-apex (alt {alt_aa:+.1f}, az {az_aa:.0f})')
axs[1].plot(az_ap, alt_ap, 'P', color=COL['blue'], ms=9, label=f'apex / Cygnus (alt {alt_ap:+.1f}, az {az_ap:.0f})')
axs[1].axhline(0, color='k', lw=1); axs[1].set_xlabel('azimuth [deg, N=0 E=90]'); axs[1].set_ylabel('altitude [deg]'); axs[1].set_xlim(0, 360); axs[1].set_ylim(-90, 90)
axs[1].legend(fontsize=7, loc='lower left'); axs[1].set_title('(b) Where the 248 keV recoil would have pointed (grey: elastic)', fontsize=10)
fig.tight_layout(); fig.savefig(f'{FIG}/P067_fig3_surf_geometry.png', dpi=160); plt.close(fig)

# ------------------------------------------------------------------------------------------------
# Save summary JSON
# ------------------------------------------------------------------------------------------------
summary = dict(
    v_lab_kms=V_LAB, v_sun_kms=V_SUN, vmax_june_kms=VMAX_JUNE, v_lab_gal_kms=V_LAB_GAL.tolist(),
    A_min_1TeV_June=A_min, gas_densities_kg_m3=rho, target_masses_1000m3_kg=dict(F_HeSF6=m_F_HeSF6, F_CF4_40Torr=m_F_CF4, Xe_40Torr=m_Xe_gas, I_CF3I_40Torr=m_I_CF3I),
    L10_unit_coupling_LZ_count=N_unit, L10_F_rates_per_tyr_bestfit=rates_F, xe_full_window_rate_per_tyr=xe_full, Ag_over_Xe_helm=ag_xe,
    N3_obs_dipole=dict(inelastic_248keV=N3_inel, inelastic_366_spectrum=N3_inel_spec366, F_L10_gt20=N3_F, F_L10_gt50=N3_F50, Xe_L10_elastic=N3_Xe_el),
    toys=TOYS, geometry=dict(LST_h=LST_EV / 15, apex_ra_dec=[float(ra_ap[0]), float(dec_ap[0])], apex_alt_az=[float(alt_ap), float(az_ap)],
                             antiapex_ra_dec=[float(ra_aa[0]), float(dec_aa[0])], antiapex_alt_az=[float(alt_aa), float(az_aa)],
                             apex_l_b=[float(l_ap), float(b_ap)], sep_from_RA312p5_Dec45_deg=sep_cyg, apex_alt_range_day=[float(alts.min()), float(alts.max())],
                             frac_day_apex_below_horizon=float(np.mean(alts < 0)), recoil_directions=GEO),
    recalled=RECALLED, runtime_s=time.time() - T0)
with open(f'{OUT}/P067_summary.json', 'w') as f:
    json.dump(summary, f, indent=1, default=float)
with open(f'{OUT}/P067_run_log.txt', 'w') as f:
    f.write('\n'.join(LOG))
log(f'done in {time.time() - T0:.0f} s; {len(RECALLED)} recalled items')
