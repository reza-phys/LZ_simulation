"""
P030 -- Streams and substructure: can a fast dark-matter stream produce the 248 keV event
without a low-energy population?

Run from the simulation root:   .venv/bin/python output/code/P030_streams.py

Parts
  0  Halo-function construction. A Galactic-frame Gaussian stream (bulk velocity u_s, per-axis
     dispersion sigma) seen from Earth (velocity v_obs) is a Maxwellian with v0_s = sqrt2 sigma boosted
     by v_lab = |u_s - v_obs|, whose mean inverse speed is analytic:
        eta_s(vmin) = [erf((vmin+v_lab)/v0_s) - erf((vmin-v_lab)/v0_s)] / (2 v_lab).
     WimPyDD's (vmin_i, delta_eta_i) arrays use upper bin boundaries with delta_eta_i = eta(v_{i-1}) - eta(v_i)
     (P018 convention check), so delta_eta is built by differencing eta on a 1 km/s grid.  The identical
     construction applied to the truncated Maxwellian (lz.eta0) is validated against lz.wd_halo (WimPyDD) both
     in eta and in the in-ROI inelastic rate.  Composite halo: (1-f_s) SHM + f_s stream at fixed rho_0.
  1  Elastic kinematics and shape: a monochromatic stream at v_lab gives a recoil spectrum that is the
     WimPyDD per-stream kernel K(E, v_lab) (flat x nuclear response for O1).  N_lo/N_hi = R(5.4-55)/R(200-270)
     for O1, O4, O6, O10 and for composite halos; Helm flat-spectrum estimate for comparison.
  2  Inelastic (Higgsino Z-exchange coupling of P007, m = 1000 GeV): N_stream(v_lab, delta) for pure streams,
     composite counts for v_lab = 600-1000 km/s, f_s = 0.01/0.05/0.2, delta = 300-500 keV; delta_max(v_lab);
     f_s needed for one event at delta = 380/400/450 keV; annual modulation of the stream signal for streams
     aligned / anti-aligned / perpendicular to the solar motion.
  3  Known substructures (recalled parameters, flagged): lab-frame speeds, delta_max, elastic E_max, and the
     Higgsino count they would add at delta = 300/350/380 keV.
  4  Escape-velocity ceiling: bound streams have v_lab <= v_esc + v_E = 810 km/s; tail-density and rate
     enhancement just below delta_max; unbound component requirements.
"""
import sys, os, json, math, time
import numpy as np
sys.path.insert(0, 'output/code')
from common import lzcommon as lz
from scipy.special import erf, erfc
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'output/work/P030'; FIG = OUT + '/figures'
os.makedirs(FIG, exist_ok=True)
T0 = time.time()
WD = lz.wd()

def log(*a):
    print('[%6.1fs]' % (time.time() - T0), *a, flush=True)

V0, VESC = lz.V0_KMS, lz.VESC_KMS                 # 238, 544 km/s (Baxter 2021)
VSUN_PEC = lz.V_SUN_PEC.copy()
VGRID = np.linspace(0.0, 1100.0, 1101)            # upper bin boundaries, 1 km/s
DV = VGRID[1] - VGRID[0]
DOY_JUNE = 167
MASS = 1000.0
SIGMA_S = 20.0                                    # stream per-axis dispersion (assignment)
EXPOSURE = lz.LZ['exposure_tyr']
E50_HI, E50_LO, PLATEAU = lz.LZ['E_50pct_high_keV'], lz.LZ['E_50pct_low_keV'], lz.LZ['eff_plateau']
MN = lz.m_nucleus_gev(lz.A_XE_MEAN)
MU = lz.mu_red(MASS, MN)

def efficiency(E, sig_hi=11.5, sig_lo=2.5):
    """LZ NR efficiency (P007/P018 model): plateau 0.96, 50% at 5.4 and 269.9 keV, erf edges."""
    return PLATEAU * 0.5 * (1 + erf((E - E50_LO) / (math.sqrt(2) * sig_lo))) * 0.5 * erfc((E - E50_HI) / (math.sqrt(2) * sig_hi))

# ---------------------------------------------------------------------------------------------
# Part 0: velocity vectors and halo functions
# ---------------------------------------------------------------------------------------------
def v_sun_vec():
    return np.array([0.0, V0, 0.0]) + VSUN_PEC

def v_orb_vec(doy):
    lam = (doy - 80) * 2.0 * np.pi / 365.0        # WimPyDD phase convention (P018)
    return WD.v_earth_sun(lam, v_rot_gal=np.array([0.0, V0, 0.0]), v_sun_rot=VSUN_PEC)

def v_obs_vec(doy=None):
    return v_sun_vec() if doy is None else v_sun_vec() + v_orb_vec(doy)

def eta_stream(v, v_lab, sigma=SIGMA_S):
    v0s = math.sqrt(2.0) * sigma
    v = np.asarray(v, float)
    return (erf((v + v_lab) / v0s) - erf((v - v_lab) / v0s)) / (2.0 * v_lab)

def deta_from_eta(eta):
    de = np.zeros_like(eta); de[1:] = eta[:-1] - eta[1:]
    return np.clip(de, 0.0, None)

def deta_shm(doy=DOY_JUNE, v0=V0, vesc=VESC):
    vE = float(np.linalg.norm(v_obs_vec(doy)))
    return deta_from_eta(np.asarray(lz.eta0(VGRID, v_e=vE, v0=v0, vesc=vesc)))

def deta_stream(v_lab, sigma=SIGMA_S):
    return deta_from_eta(eta_stream(VGRID, v_lab, sigma))

def eta_of(de):
    return np.cumsum(de[::-1])[::-1] - de

def composite(de_shm, de_str, f_s):
    return (1.0 - f_s) * de_shm + f_s * de_str

VOBS_J = v_obs_vec(DOY_JUNE); VE_J = float(np.linalg.norm(VOBS_J))
VSUN = float(np.linalg.norm(v_sun_vec()))
DE_SHM_J = deta_shm(DOY_JUNE)
DE_SHM_WD = lz.wd_halo(day_of_year=DOY_JUNE, vmin=VGRID)[1]

# --- validation of the differencing construction against WimPyDD (Maxwellian) ---
val = dict(v_obs_june16=VOBS_J.tolist(), v_E_june16=VE_J, v_sun=VSUN, v_max_june16=VE_J + VESC,
           eta0_mine=float(DE_SHM_J.sum()), eta0_wimpydd=float(DE_SHM_WD.sum()))
eta_m, eta_w = eta_of(DE_SHM_J), eta_of(DE_SHM_WD)
val['eta_vs_vmin'] = [dict(vmin=v, mine=float(eta_m[v]), wimpydd=float(eta_w[v]), ratio=float(eta_m[v] / eta_w[v]) if eta_w[v] > 0 else None)
                      for v in [0, 200, 400, 600, 700, 750, 780, 800]]
# stream normalisation check: sum delta_eta = 1/v_lab (cold limit) and the dispersion-broadened speed range
val['stream_checks'] = [dict(v_lab=v, sum_deta_times_vlab=float(deta_stream(v).sum() * v),
                             v_at_half_max_deta=[float(VGRID[np.argmax(deta_stream(v) > 0.5 * deta_stream(v).max())]),
                                                 float(VGRID[len(VGRID) - 1 - np.argmax(deta_stream(v)[::-1] > 0.5 * deta_stream(v).max())])])
                        for v in [600.0, 800.0, 1000.0]]
log('Part 0: v_E(16 June) = %.2f km/s, v_max = %.2f; eta0 mine/WimPyDD = %.6f' % (VE_J, VE_J + VESC, val['eta0_mine'] / val['eta0_wimpydd']))

# --- hamiltonians ---
MV = lz.M_V_GEV
GF, SW2 = 1.166e-5, 0.231                                   # recalled, certain (P007)
c_p = (GF / math.sqrt(2)) * (1 - 4 * SW2); c_n = -GF / math.sqrt(2)
HAM_HIG = lz.wd_hamiltonian('higgsino_Z', {1: (c_p + c_n, c_p - c_n)})      # sigma_n = 7.4e-39 cm^2 (P007)
HAM_ISO = {op: lz.wd_hamiltonian(f'iso_unit_O{op}', {op: (2.0 / MV**2, 0.0)}) for op in [1, 4, 6, 10]}
ONES = np.ones_like(VGRID)

def kernel(ham, m, delta, dE=2.0, E_max=330.0, E_min_floor=1.0):
    """K[E, v_i] = dR/dE (events/(t yr keV)) per unit delta_eta at stream speed v_i."""
    if delta > 0:
        lo = min(lz.E_R_range_keV(m, VGRID[-1], A=float(A), delta_kev=delta)[0] for A in lz.XE_ISOTOPES)
        if math.isnan(lo):
            return None, None
        E = np.arange(max(E_min_floor, math.floor(lo) - 4.0), E_max + 1e-9, dE)
    else:
        E = np.arange(E_min_floor, E_max + 1e-9, dE)
    K = np.array([WD.diff_rate(WD.Xe, ham, m, float(e), VGRID, ONES, j_chi=0.5, delta=float(delta), sum_over_streams=False) for e in E])
    return E, K * 1000.0 * 365.25

def N_roi(E, K, de, Elo=None, Ehi=None):
    r = K @ de
    w = efficiency(E)
    if Elo is not None:
        w = w * ((E >= Elo) & (E <= Ehi))
    return float(np.trapezoid(r * w, E)) * EXPOSURE

# rate validation at delta = 300/350/380 (Higgsino, June 16): my SHM delta_eta vs WimPyDD's
val['rate_checks'] = []
for d in [300.0, 350.0, 380.0]:
    E, K = kernel(HAM_HIG, MASS, d)
    Nm, Nw = N_roi(E, K, DE_SHM_J), N_roi(E, K, DE_SHM_WD)
    val['rate_checks'].append(dict(delta_keV=d, N_mine=Nm, N_wimpydd=Nw, ratio=Nm / Nw))
    log('  validation delta=%.0f: N mine %.4g  WimPyDD %.4g  ratio %.4f' % (d, Nm, Nw, Nm / Nw))
json.dump(val, open(f'{OUT}/validation.json', 'w'), indent=1)

# ---------------------------------------------------------------------------------------------
# Part 1: elastic scattering -- can a stream remove the low-energy population?
# ---------------------------------------------------------------------------------------------
log('Part 1: elastic kernels O1, O4, O6, O10')
V_ELASTIC = [300.0, 500.0, 555.0, 600.0, 700.0, 800.0, 900.0, 1000.0]
KEL = {op: kernel(HAM_ISO[op], MASS, 0.0, dE=1.0) for op in HAM_ISO}
E1 = KEL[1][0]

def nlo_nhi(E, r):
    w = efficiency(E)
    lo = float(np.trapezoid(r * w * ((E >= 5.4) & (E <= 55.0)), E))
    hi = float(np.trapezoid(r * w * ((E >= 200.0) & (E <= 270.0)), E))
    return lo, hi, (lo / hi if hi > 0 else float('inf'))

rows = []
for op, (E, K) in KEL.items():
    lo, hi, r_shm = nlo_nhi(E, K @ DE_SHM_J)
    rows.append(dict(operator=f'O{op}', halo='SHM (16 June)', v_lab_kms=float('nan'), f_s=0.0, E_max_keV=lz.E_R_range_keV(MASS, VE_J + VESC)[1],
                     R_lo=lo, R_hi=hi, N_lo_per_hi=r_shm))
    for v in V_ELASTIC:
        i = int(round(v / DV))
        lo, hi, rr = nlo_nhi(E, K[:, i])                      # monochromatic stream: spectrum = kernel column
        rows.append(dict(operator=f'O{op}', halo='monochromatic stream', v_lab_kms=v, f_s=1.0, E_max_keV=lz.E_R_range_keV(MASS, v)[1],
                         R_lo=lo, R_hi=hi, N_lo_per_hi=rr))
        for f in [0.01, 0.05, 0.2]:
            lo, hi, rr = nlo_nhi(E, K @ composite(DE_SHM_J, deta_stream(v), f))
            rows.append(dict(operator=f'O{op}', halo='SHM + stream (sigma 20)', v_lab_kms=v, f_s=f, E_max_keV=lz.E_R_range_keV(MASS, v + 3 * SIGMA_S)[1],
                             R_lo=lo, R_hi=hi, N_lo_per_hi=rr))
df1 = pd.DataFrame(rows); df1.to_csv(f'{OUT}/elastic_Nlo_Nhi.csv', index=False, float_format='%.5g')

# Helm flat-spectrum estimate: N_lo/N_hi = Int_5.4^55 F^2 eff / Int_200^270 F^2 eff (natural-Xe abundance-weighted F^2)
Ef = np.arange(1.0, 330.0, 0.5)
F2 = np.array([sum(f * lz.helm_F2(e, A) for A, f in lz.XE_ISOTOPES.items()) for e in Ef])
helm_lo = float(np.trapezoid(F2 * efficiency(Ef) * ((Ef >= 5.4) & (Ef <= 55)), Ef)); helm_hi = float(np.trapezoid(F2 * efficiency(Ef) * ((Ef >= 200) & (Ef <= 270)), Ef))
# shell-model O1 kernel at a speed above the 270 keV threshold (flat part): same integrals
i900 = int(round(900.0 / DV)); r900 = KEL[1][1][:, i900]
sm_lo, sm_hi, sm_ratio = nlo_nhi(E1, r900)
part1 = dict(v_min_248keV_elastic_1000GeV=lz.vmin_kms(248.0, MASS), v_min_270keV=lz.vmin_kms(270.0, MASS), v_min_200keV=lz.vmin_kms(200.0, MASS),
             E_max_keV={v: lz.E_R_range_keV(MASS, v)[1] for v in V_ELASTIC},
             helm_flat=dict(int_lo=helm_lo, int_hi=helm_hi, ratio=helm_lo / helm_hi, mean_F2_lo=helm_lo / (49.6 * PLATEAU), mean_F2_hi=helm_hi / (70 * PLATEAU * 0.83),
                            width_ratio=49.6 / 70.0),
             shell_model_flat_900kms=dict(ratio=sm_ratio, int_lo=sm_lo, int_hi=sm_hi),
             N_lo_SHM={f'O{op}': float(df1[(df1.operator == f'O{op}') & (df1.halo.str.startswith('SHM ('))].N_lo_per_hi.iloc[0]) for op in KEL},
             N_lo_stream={f'O{op}': {v: float(df1[(df1.operator == f'O{op}') & (df1.halo == 'monochromatic stream') & (df1.v_lab_kms == v)].N_lo_per_hi.iloc[0]) for v in V_ELASTIC} for op in KEL})
json.dump(part1, open(f'{OUT}/part1_elastic.json', 'w'), indent=1)
log('  N_lo(SHM) O1 = %.0f; monochromatic-stream O1 ratios: %s' % (part1['N_lo_SHM']['O1'], {v: round(x, 1) for v, x in part1['N_lo_stream']['O1'].items()}))
log('  Helm flat estimate %.1f; shell-model flat (900 km/s) %.1f' % (part1['helm_flat']['ratio'], sm_ratio))
print(df1[(df1.operator == 'O1')].to_string(index=False))

# ---------------------------------------------------------------------------------------------
# Part 2: inelastic Higgsino with streams
# ---------------------------------------------------------------------------------------------
DELTAS = np.arange(300.0, 500.1, 5.0)
VLABS = np.arange(600.0, 1000.1, 25.0)
VLAB_TAB = [600.0, 700.0, 800.0, 900.0, 1000.0]
FS = [0.01, 0.05, 0.2]
log('Part 2: Higgsino kernels for %d deltas' % len(DELTAS))
KERN = {}
DE_STR = {v: deta_stream(v) for v in VLABS}
rows = []
for d in DELTAS:
    E, K = kernel(HAM_HIG, MASS, d)
    KERN[d] = (E, K)
    Nshm = N_roi(E, K, DE_SHM_J)
    for v in VLABS:
        Nstr = N_roi(E, K, DE_STR[v])
        rows.append(dict(delta_keV=d, v_lab_kms=v, N_SHM=Nshm, N_stream_pure=Nstr, **{f'N_f{f}': (1 - f) * Nshm + f * Nstr for f in FS},
                         **{f'boost_f{f}': ((1 - f) * Nshm + f * Nstr) / Nshm if Nshm > 0 else float('inf') for f in FS}))
df2 = pd.DataFrame(rows); df2.to_csv(f'{OUT}/inelastic_stream_grid.csv', index=False, float_format='%.5g')
log('  grid done')

# delta_max(v_lab): analytic (monochromatic) and with the 2-sigma / 3-sigma dispersion tail; numerical delta at which N_stream falls to 1e-3 of its 300 keV value
dm_rows = []
for v in VLABS:
    s = df2[df2.v_lab_kms == v].sort_values('delta_keV')
    Np = s.N_stream_pure.values
    ref = Np[0]
    d_1e3 = float(np.interp(-3.0, np.log10(np.maximum(Np / ref, 1e-12))[::-1], s.delta_keV.values[::-1])) if Np.min() < 1e-3 * ref else float('nan')
    dm_rows.append(dict(v_lab_kms=v, v_gal_antialigned_june=v - VE_J, bound=(v - VE_J) <= VESC,
                        delta_max_mono=lz.delta_max_kev(248.0, MASS, v_kms=v), delta_max_2sigma=lz.delta_max_kev(248.0, MASS, v_kms=v + 2 * SIGMA_S),
                        delta_max_3sigma=lz.delta_max_kev(248.0, MASS, v_kms=v + 3 * SIGMA_S), delta_ceiling_muv2_over_2=MU * (v / lz.C_KMS)**2 / 2 * 1e6,
                        delta_N_1em3_of_300=d_1e3, E_minus_380=lz.E_R_range_keV(MASS, v, delta_kev=380.0)[0], E_plus_380=lz.E_R_range_keV(MASS, v, delta_kev=380.0)[1],
                        N_stream_300=float(s[s.delta_keV == 300].N_stream_pure.iloc[0]), N_stream_380=float(s[s.delta_keV == 380].N_stream_pure.iloc[0]),
                        N_stream_400=float(s[s.delta_keV == 400].N_stream_pure.iloc[0]), N_stream_450=float(s[s.delta_keV == 450].N_stream_pure.iloc[0])))
dfm = pd.DataFrame(dm_rows); dfm.to_csv(f'{OUT}/deltamax_vs_vlab.csv', index=False, float_format='%.5g')
print(dfm[['v_lab_kms', 'v_gal_antialigned_june', 'bound', 'delta_max_mono', 'delta_max_3sigma', 'delta_N_1em3_of_300', 'N_stream_300', 'N_stream_380', 'N_stream_400', 'N_stream_450']].to_string(index=False))

# f_s needed for N = 1 at delta = 380, 400, 450 (and 350, 366 for reference), per v_lab
fs_rows = []
for d in [350.0, 365.0, 380.0, 400.0, 450.0]:
    s = df2[df2.delta_keV == d]
    Nshm = float(s.N_SHM.iloc[0])
    for v in VLABS:
        Nstr = float(s[s.v_lab_kms == v].N_stream_pure.iloc[0])
        f_need = (1.0 - Nshm) / (Nstr - Nshm) if Nstr > Nshm and Nshm < 1 else (0.0 if Nshm >= 1 else float('inf'))
        fs_rows.append(dict(delta_keV=d, v_lab_kms=v, v_gal_antialigned_june=v - VE_J, N_SHM=Nshm, N_stream_pure=Nstr, f_s_for_one_event=f_need,
                            feasible=(f_need <= 1.0)))
dff = pd.DataFrame(fs_rows); dff.to_csv(f'{OUT}/fs_for_one_event.csv', index=False, float_format='%.4g')
# minimum v_lab for f_s <= 1, 0.2, 0.05, 0.01 at each delta (interpolated in log f)
vmin_rows = []
for d in [350.0, 365.0, 380.0, 400.0, 450.0]:
    s = dff[dff.delta_keV == d].sort_values('v_lab_kms')
    row = dict(delta_keV=d, N_SHM=float(s.N_SHM.iloc[0]), v_lab_kinematic_min=VE_J + VESC + (d - lz.delta_max_kev(248.0, MASS, v_kms=VE_J + VESC)) / (math.sqrt(2 * MN * 248e-6) * 1e6 / lz.C_KMS))
    f = s.f_s_for_one_event.values; v = s.v_lab_kms.values
    for ft in [1.0, 0.2, 0.05, 0.01, 0.001]:
        ok = np.isfinite(f) & (f > 0)
        if ok.sum() >= 2 and f[ok].min() <= ft:
            lf = np.log10(f[ok]); vv = v[ok]
            # f decreases with v_lab: find first crossing
            idx = np.where(lf <= math.log10(ft))[0][0]
            if idx == 0:
                row[f'v_lab_for_fs_{ft}'] = float(vv[0])
            else:
                row[f'v_lab_for_fs_{ft}'] = float(vv[idx - 1] + (math.log10(ft) - lf[idx - 1]) * (vv[idx] - vv[idx - 1]) / (lf[idx] - lf[idx - 1]))
        else:
            row[f'v_lab_for_fs_{ft}'] = float('nan')
    vmin_rows.append(row)
dfv = pd.DataFrame(vmin_rows); dfv.to_csv(f'{OUT}/vlab_needed_for_fs.csv', index=False, float_format='%.4g')
print(dfv.to_string(index=False))

# --- annual modulation of a stream signal (fixed Galactic-frame stream vector) ---
log('Part 2b: annual modulation of stream signals')
DOYS = np.arange(1.0, 366.0, 3.0)
vs_hat = v_sun_vec() / VSUN
# Earth orbital velocity plane: find the direction in that plane maximising the projection amplitude
VORB = np.array([v_orb_vec(d) for d in DOYS])
def mod_amplitude(direction):
    return 0.5 * (np.ptp(VORB @ direction))
# ecliptic-plane direction perpendicular to the solar motion direction: use the orbital vector on the day when it is most perpendicular to vs_hat
ecl_normal = np.cross(VORB[0], VORB[len(DOYS) // 4]); ecl_normal /= np.linalg.norm(ecl_normal)
in_plane_perp = np.cross(ecl_normal, vs_hat); in_plane_perp /= np.linalg.norm(in_plane_perp)
DIRS = {'anti-aligned (retrograde, -v_sun)': -vs_hat, 'aligned (prograde, +v_sun)': vs_hat,
        'perpendicular, vertical (+z)': np.array([0.0, 0.0, 1.0]), 'perpendicular, radial (+x, toward GC)': np.array([1.0, 0.0, 0.0]),
        'perpendicular, in ecliptic plane': in_plane_perp}
mod_rows = []
for name, dhat in DIRS.items():
    for vgal in [400.0, 500.0, 544.0, 600.0]:
        u = vgal * dhat
        vlab = np.array([np.linalg.norm(u - v_obs_vec(d)) for d in DOYS])
        vlab_june = float(np.linalg.norm(u - VOBS_J)); vlab_mean = float(np.linalg.norm(u - v_sun_vec()))
        row = dict(direction=name, v_gal_kms=vgal, v_lab_mean_kms=vlab_mean, v_lab_june16_kms=vlab_june, v_lab_min=float(vlab.min()), v_lab_max=float(vlab.max()),
                   v_lab_semi_amplitude=float(0.5 * np.ptp(vlab)), peak_doy=float(DOYS[np.argmax(vlab)]), orbital_projection_amplitude=float(mod_amplitude(dhat)),
                   delta_max_june16=lz.delta_max_kev(248.0, MASS, v_kms=vlab_june), delta_max_yearmax=lz.delta_max_kev(248.0, MASS, v_kms=float(vlab.max())))
        # rate modulation for a pure stream at delta = 350 and at delta just below its own June delta_max (rounded down to the grid)
        for d in [350.0, 380.0]:
            if d in KERN:
                E, K = KERN[d]
                N = np.array([N_roi(E, K, deta_stream(v)) for v in vlab])
                row[f'N_mean_d{int(d)}'] = float(N.mean()); row[f'N_june16_d{int(d)}'] = N_roi(E, K, deta_stream(vlab_june))
                row[f'frac_year_nonzero_d{int(d)}'] = float((N > 1e-6 * max(N.max(), 1e-30)).mean())
                row[f'mod_amp_d{int(d)}'] = float((N.max() - N.min()) / (N.max() + N.min())) if N.max() > 0 else float('nan')
                row[f'peak_doy_d{int(d)}'] = float(DOYS[np.argmax(N)]) if N.max() > 0 else float('nan')
        mod_rows.append(row)
dfmod = pd.DataFrame(mod_rows); dfmod.to_csv(f'{OUT}/stream_modulation.csv', index=False, float_format='%.4g')
print(dfmod[['direction', 'v_gal_kms', 'v_lab_mean_kms', 'v_lab_june16_kms', 'v_lab_semi_amplitude', 'peak_doy', 'orbital_projection_amplitude', 'delta_max_june16', 'mod_amp_d350', 'frac_year_nonzero_d380', 'peak_doy_d380']].to_string(index=False))
# save the anti-aligned v_gal = 544 and 600 time series for the figure
ts = {'doy': DOYS.tolist()}
for vgal in [544.0, 600.0]:
    u = -vgal * vs_hat
    vlab = np.array([np.linalg.norm(u - v_obs_vec(d)) for d in DOYS])
    ts[f'vlab_anti_{int(vgal)}'] = vlab.tolist()
    for d in [350.0, 380.0, 400.0]:
        E, K = KERN[d]
        ts[f'N_anti_{int(vgal)}_d{int(d)}'] = [N_roi(E, K, deta_stream(v)) for v in vlab]
E, K = KERN[350.0]
ts['N_SHM_d350'] = [N_roi(E, K, deta_shm(d)) for d in DOYS]
E, K = KERN[380.0]
ts['N_SHM_d380'] = [N_roi(E, K, deta_shm(d)) for d in DOYS]
json.dump(ts, open(f'{OUT}/modulation_timeseries.json', 'w'))

# ---------------------------------------------------------------------------------------------
# Part 3: known substructures (recalled; all flagged)
# ---------------------------------------------------------------------------------------------
log('Part 3: known substructures')
# WimPyDD axes: x toward the Galactic centre, y along rotation, z toward the NGP.
CANDS = [
 dict(name='Sagittarius stream (leading, -z)', u=(0.0, 0.0, -300.0), sigma=20.0, f_s_max=0.03, note='v~300 km/s roughly perpendicular to the disk; sign of v_z uncertain; f_s <~ few % (recalled, uncertain)'),
 dict(name='Sagittarius stream (+z variant)', u=(0.0, 0.0, 300.0), sigma=20.0, f_s_max=0.03, note='same speed, opposite vertical sense'),
 dict(name='S1 stream (Gaia, retrograde)', u=(-34.0, -306.0, -64.0), sigma=60.0, f_s_max=0.10, note='(v_r,v_phi,v_z)~(-34,-306,-64), dispersions ~(82,27,59); f_s up to ~10% claimed (recalled, likely speed / uncertain fraction)'),
 dict(name='S2 stream (prograde, vertical)', u=(6.0, 164.0, -250.0), sigma=40.0, f_s_max=0.01, note='prograde with large vertical velocity (recalled, uncertain)'),
 dict(name='Gaia-Enceladus/Sausage debris (radial, +x)', u=(250.0, 0.0, 0.0), sigma=80.0, f_s_max=0.20, note='v_phi~0, |v_r|~200-300, strongly radial; f_s~20% (SHM++) (recalled, likely)'),
 dict(name='Gaia-Enceladus/Sausage debris (radial, -x)', u=(-250.0, 0.0, 0.0), sigma=80.0, f_s_max=0.20, note='as above, inward-moving half'),
 dict(name='Retrograde shards (Rg-type)', u=(0.0, -280.0, 0.0), sigma=60.0, f_s_max=0.01, note='retrograde v_phi~-(250-300), small fractions (recalled, uncertain)'),
 dict(name='Helmi streams', u=(0.0, 150.0, 250.0), sigma=30.0, f_s_max=0.01, note='prograde with large |v_z| (recalled, uncertain); f_s <~ 1%'),
 dict(name='Hypothetical bound retrograde stream at v_esc', u=tuple((-VESC * vs_hat).tolist()), sigma=20.0, f_s_max=0.01, note='limiting bound case, anti-aligned with the solar motion'),
]
rows = []
for c in CANDS:
    u = np.array(c['u']); vgal = float(np.linalg.norm(u))
    vlab_j = float(np.linalg.norm(u - VOBS_J)); vlab_m = float(np.linalg.norm(u - v_sun_vec()))
    de_c = deta_stream(vlab_j, c['sigma'])
    row = dict(name=c['name'], v_gal_kms=vgal, bound=vgal <= VESC, sigma_kms=c['sigma'], f_s_max=c['f_s_max'], v_lab_mean_kms=vlab_m, v_lab_june16_kms=vlab_j,
               delta_max_mono_june=lz.delta_max_kev(248.0, MASS, v_kms=vlab_j), delta_max_2sigma_june=lz.delta_max_kev(248.0, MASS, v_kms=vlab_j + 2 * c['sigma']),
               E_max_elastic_keV=lz.E_R_range_keV(MASS, vlab_j)[1], E_max_elastic_2sigma_keV=lz.E_R_range_keV(MASS, vlab_j + 2 * c['sigma'])[1], note=c['note'])
    for d in [300.0, 350.0, 380.0]:
        E, K = KERN[d]
        Nshm = N_roi(E, K, DE_SHM_J); Nstr = N_roi(E, K, de_c)
        row[f'N_stream_pure_d{int(d)}'] = Nstr
        row[f'N_composite_fmax_d{int(d)}'] = (1 - c['f_s_max']) * Nshm + c['f_s_max'] * Nstr
        row[f'boost_fmax_d{int(d)}'] = row[f'N_composite_fmax_d{int(d)}'] / Nshm if Nshm > 0 else float('inf')
    # elastic O1 N_lo for the pure stream (Gaussian)
    E, K = KEL[1]
    lo, hi, rr = nlo_nhi(E, K @ de_c)
    row['elastic_O1_Nlo_per_hi_pure_stream'] = rr
    rows.append(row)
df3 = pd.DataFrame(rows); df3.to_csv(f'{OUT}/known_substructures.csv', index=False, float_format='%.4g')
print(df3[['name', 'v_gal_kms', 'v_lab_june16_kms', 'delta_max_mono_june', 'delta_max_2sigma_june', 'E_max_elastic_keV', 'boost_fmax_d300', 'boost_fmax_d350', 'boost_fmax_d380']].to_string(index=False))

# ---------------------------------------------------------------------------------------------
# Part 4: escape-velocity ceiling and tail-density enhancement
# ---------------------------------------------------------------------------------------------
log('Part 4: ceiling and tail enhancement')
eta_shm_j = eta_of(DE_SHM_J)
tail_rows = []
for v in [700.0, 750.0, 780.0, 800.0, 810.0]:
    de_s = deta_stream(v)
    for f in [0.001, 0.01, 0.05]:
        e_c = eta_of(composite(DE_SHM_J, de_s, f))
        tail_rows.append(dict(v_lab_kms=v, f_s=f, **{f'eta_ratio_{int(vm)}': float(e_c[int(vm)] / eta_shm_j[int(vm)]) if eta_shm_j[int(vm)] > 0 else float('inf') for vm in [600, 700, 750, 780, 800, 805]}))
dft = pd.DataFrame(tail_rows); dft.to_csv(f'{OUT}/tail_enhancement.csv', index=False, float_format='%.4g')
print(dft.to_string(index=False))
rate_rows = []
for v in [750.0, 780.0, 800.0, 810.0]:
    de_s = deta_stream(v)
    for d in [350.0, 365.0, 370.0, 380.0, 385.0]:
        E, K = KERN[d]
        Nshm = N_roi(E, K, DE_SHM_J); Nstr = N_roi(E, K, de_s)
        rate_rows.append(dict(v_lab_kms=v, delta_keV=d, N_SHM=Nshm, N_stream_pure=Nstr, **{f'boost_f{f}': ((1 - f) * Nshm + f * Nstr) / Nshm if Nshm > 0 else float('inf') for f in [0.001, 0.01, 0.05]},
                              f_s_for_one_event=(1 - Nshm) / (Nstr - Nshm) if (Nstr > Nshm and Nshm < 1) else float('nan')))
dfr = pd.DataFrame(rate_rows); dfr.to_csv(f'{OUT}/bound_stream_boost.csv', index=False, float_format='%.4g')
print(dfr.to_string(index=False))
slope = math.sqrt(2 * MN * 248e-6) * 1e6 / lz.C_KMS
part4 = dict(v_esc=VESC, v_E_june16=VE_J, v_lab_ceiling_bound_june=VESC + VE_J, v_lab_ceiling_bound_annual_mean=VESC + VSUN,
             delta_max_ceiling_bound_june=lz.delta_max_kev(248.0, MASS, v_kms=VESC + VE_J), slope_keV_per_kms=slope,
             v_lab_needed={d: VE_J + VESC + (d - lz.delta_max_kev(248.0, MASS, v_kms=VE_J + VESC)) / slope for d in [400.0, 420.0, 450.0, 500.0]},
             v_gal_needed_antialigned_june={d: VESC + (d - lz.delta_max_kev(248.0, MASS, v_kms=VE_J + VESC)) / slope for d in [400.0, 420.0, 450.0, 500.0]},
             v_gal_for_delta350_antialigned_june=(lz.vmin_kms(248.0, MASS, delta_kev=350.0) - VE_J), v_gal_for_delta300_antialigned_june=(lz.vmin_kms(248.0, MASS, delta_kev=300.0) - VE_J),
             vmin_248={d: lz.vmin_kms(248.0, MASS, delta_kev=d) for d in [300.0, 350.0, 380.0, 400.0, 450.0]})
json.dump(part4, open(f'{OUT}/part4_ceiling.json', 'w'), indent=1)
log('  bound ceiling: v_lab <= %.1f km/s -> delta_max = %.1f keV; delta = 400/450 need v_lab = %.0f/%.0f km/s (v_gal = %.0f/%.0f anti-aligned)' % (
    part4['v_lab_ceiling_bound_june'], part4['delta_max_ceiling_bound_june'], part4['v_lab_needed'][400.0], part4['v_lab_needed'][450.0],
    part4['v_gal_needed_antialigned_june'][400.0], part4['v_gal_needed_antialigned_june'][450.0]))

# --- Part 4b: dispersion sensitivity of the (v_lab, f_s) requirement above the SHM delta_max ---
log('Part 4b: dispersion sensitivity and truncated bound streams')
sig_rows = []
for sig in [5.0, 10.0, 20.0, 40.0]:
    for v in [725.0, 750.0, 775.0, 800.0, 825.0, 850.0, 900.0, 1000.0]:
        de_s = deta_stream(v, sig)
        row = dict(sigma_kms=sig, v_lab_kms=v)
        for d in [380.0, 400.0, 450.0]:
            E, K = KERN[d]
            Nshm = N_roi(E, K, DE_SHM_J); Nstr = N_roi(E, K, de_s)
            row[f'N_stream_d{int(d)}'] = Nstr
            row[f'fs_one_event_d{int(d)}'] = (1 - Nshm) / (Nstr - Nshm) if (Nstr > Nshm and Nshm < 1) else float('nan')
        sig_rows.append(row)
dfs = pd.DataFrame(sig_rows); dfs.to_csv(f'{OUT}/fs_dispersion_sensitivity.csv', index=False, float_format='%.4g')
print(dfs.to_string(index=False))

# --- bound stream truncated at |u| <= v_esc in the Galactic frame.  For a stream anti-parallel to v_obs the problem is
#     axisymmetric about v_obs: with polar axis e3 = -v_obs/|v_obs|, u = v n - v_o e3, u_s = +v_gal e3, so
#     |u - u_s|^2 = v^2 + V^2 - 2 v V c  (V = v_o + v_gal, c = cos theta) and |u|^2 = v^2 + v_o^2 - 2 v v_o c <= v_esc^2
#     restricts c >= c_min = (v^2 + v_o^2 - v_esc^2)/(2 v v_o).  The angular integral is analytic:
#     S(v) = 2 pi v exp(-(v^2+V^2)/(2 sigma^2)) (sigma^2/(v V)) [exp(v V/sigma^2) - exp(v V c_min/sigma^2)],  eta(vmin) = Int_vmin S/N dv.
def stream_axisym_S(v, V, sigma, vo, vcut):
    v = np.asarray(v, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        cmin = np.where(v > 0, (v**2 + vo**2 - vcut**2) / (2 * v * vo), -1.0)
    cmin = np.clip(cmin, -1.0, 1.0)
    a = v * V / sigma**2
    # exp(-(v-V)^2/2s^2) - exp(-(v^2+V^2-2vVcmin)/2s^2), written stably
    term = np.exp(-(v - V)**2 / (2 * sigma**2)) - np.exp(-(v**2 + V**2 - 2 * v * V * cmin) / (2 * sigma**2))
    S = np.where(a > 0, 2 * np.pi * v * sigma**2 / np.where(a > 0, v * V, 1.0) * term, 0.0)
    return np.where(cmin < 1.0, S, 0.0)

def stream_axisym_norm(vgal, sigma, vcut):
    u = np.linspace(0.0, vcut, 200001)
    with np.errstate(divide='ignore', invalid='ignore'):
        g = np.where(u > 0, 2 * np.pi * u * sigma**2 / vgal * (np.exp(-(u - vgal)**2 / (2 * sigma**2)) - np.exp(-(u + vgal)**2 / (2 * sigma**2))), 0.0)
    return float(np.trapezoid(g, u))

def deta_stream_truncated(vgal, sigma, vo, vcut):
    V = vo + vgal
    vf = np.linspace(0.0, VGRID[-1], 20 * (len(VGRID) - 1) + 1)                # 0.05 km/s
    S = stream_axisym_S(vf, V, sigma, vo, vcut) / stream_axisym_norm(vgal, sigma, vcut)
    eta_f = np.concatenate([np.cumsum((S[1:] + S[:-1])[::-1] * 0.5 * (vf[1] - vf[0]))[::-1], [0.0]])
    return deta_from_eta(np.interp(VGRID, vf, eta_f))

trunc_rows = []; trunc_eta = {'vmin_kms': VGRID.tolist()}
for vgal in [500.0, 530.0, 544.0]:
    vlab = VE_J + vgal                                                     # stream anti-parallel to v_obs(16 June)
    de_untr = deta_stream_truncated(vgal, SIGMA_S, VE_J, 1e4)              # no truncation: must reproduce the analytic stream
    de_tr = deta_stream_truncated(vgal, SIGMA_S, VE_J, VESC)               # truncated at v_esc, renormalised to the same local density
    de_an = deta_stream(vlab, SIGMA_S)
    trunc_eta[f'eta_untrunc_{int(vgal)}'] = eta_of(de_untr).tolist(); trunc_eta[f'eta_trunc_{int(vgal)}'] = eta_of(de_tr).tolist()
    row = dict(v_gal_kms=vgal, v_lab_june16=vlab, numeric_vs_analytic_eta0=float(de_untr.sum() / de_an.sum()),
               numeric_vs_analytic_max_abs_deta_diff_rel=float(np.max(np.abs(de_untr - de_an)) / de_an.max()),
               fraction_of_stream_beyond_vesc=1.0 - stream_axisym_norm(vgal, SIGMA_S, VESC) / (2 * np.pi * SIGMA_S**2)**1.5,
               v_max_truncated=float(VGRID[de_tr > 1e-12 * de_tr.max()].max()))
    for d in [350.0, 365.0, 380.0, 385.0, 390.0, 400.0]:
        E, K = KERN[d]
        Nshm = N_roi(E, K, DE_SHM_J)
        row[f'N_untrunc_pure_d{int(d)}'] = N_roi(E, K, de_untr); row[f'N_trunc_pure_d{int(d)}'] = N_roi(E, K, de_tr)
        row[f'boost_trunc_f0.01_d{int(d)}'] = ((0.99 * Nshm + 0.01 * row[f'N_trunc_pure_d{int(d)}']) / Nshm) if Nshm > 0 else (float('inf') if row[f'N_trunc_pure_d{int(d)}'] > 0 else float('nan'))
    trunc_rows.append(row)
dftr = pd.DataFrame(trunc_rows); dftr.to_csv(f'{OUT}/bound_truncated_streams.csv', index=False, float_format='%.4g')
json.dump(trunc_eta, open(f'{OUT}/truncated_stream_eta.json', 'w'))
print(dftr.T.to_string())

# ---------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------
C = dict(blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', yellow='#eda100', magenta='#e87ba4', violet='#4a3aa7', red='#e34948', ink='#222222', mute='#888888')
BLUES = ['#c6dbf0', '#8fb8e3', '#5a93d3', '#2a78d6', '#174f96']       # one hue, light -> dark for v_lab 600 -> 1000
LS_F = {0.01: ':', 0.05: '--', 0.2: '-'}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25})

# Fig 1: composite halo functions (16 June)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
ax[0].semilogy(VGRID, eta_shm_j, color=C['ink'], lw=2.2, label='SHM (Baxter 2021), 16 June')
for j, v in enumerate([700.0, 800.0, 900.0, 1000.0]):
    e_c = eta_of(composite(DE_SHM_J, deta_stream(v), 0.01))
    ax[0].semilogy(VGRID, e_c, color=BLUES[j + 1], lw=1.6, label=f'+1% stream, v_lab = {v:.0f} km/s')
    with np.errstate(divide='ignore', invalid='ignore'):
        ax[1].semilogy(VGRID, e_c / eta_shm_j, color=BLUES[j + 1], lw=1.6)
        ax[1].semilogy(VGRID, eta_of(composite(DE_SHM_J, deta_stream(v), 0.2)) / eta_shm_j, color=BLUES[j + 1], lw=1.2, ls='--')
for a in ax:
    for d in [300.0, 350.0, 380.0, 400.0, 450.0]:
        a.axvline(lz.vmin_kms(248.0, MASS, delta_kev=d), color=C['mute'], lw=0.7, ls='--')
    a.axvline(VE_J + VESC, color=C['red'], lw=1.0, ls='-.')
    a.set_xlim(400, 1080); a.set_xlabel('$v_{\\min}$ (km/s), Earth frame, 16 June 2023')
for d, y in [(300, 1.6e-3), (350, 1.6e-3), (380, 1.6e-3), (400, 1.6e-3), (450, 1.6e-3)]:
    ax[0].text(lz.vmin_kms(248.0, MASS, delta_kev=d) + 2, y, f'$\\delta$={d}', fontsize=6.5, color=C['mute'], rotation=90, va='top')
ax[0].text(VE_J + VESC + 3, 3e-8, 'bound ceiling\n$v_{esc}+v_E$ = %.0f km/s' % (VE_J + VESC), fontsize=7, color=C['red'])
ax[0].set_ylim(1e-8, 3e-3); ax[0].set_ylabel('$\\eta(v_{\\min})$ [(km/s)$^{-1}$]'); ax[0].legend(fontsize=7, frameon=False, loc='lower left')
ax[1].set_ylim(0.5, 1e6); ax[1].set_ylabel('$\\eta_{\\rm composite}/\\eta_{\\rm SHM}$'); ax[1].axhline(1, color=C['ink'], lw=1)
ax[1].text(405, 2e5, 'solid: $f_s$ = 1%;  dashed: $f_s$ = 20%\n($\\sigma_s$ = 20 km/s per axis)', fontsize=7.5, color=C['ink'])
ax[0].set_title('(a) Halo functions with a cold stream'); ax[1].set_title('(b) Tail enhancement; dashed verticals: $v_{\\min}$(248 keV, 1 TeV, $\\delta$)')
fig.tight_layout(); fig.savefig(f'{FIG}/P030_fig1_composite_eta.png', dpi=170); plt.close(fig)

# Fig 2: Higgsino expected counts vs delta for composites (a) and f_s needed for one event (b)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
s0 = df2[df2.v_lab_kms == 600.0].sort_values('delta_keV')
ax[0].plot(s0.delta_keV, s0.N_SHM, color=C['ink'], lw=2.2, label='SHM only')
for j, v in enumerate(VLAB_TAB):
    s = df2[df2.v_lab_kms == v].sort_values('delta_keV')
    for f in [0.01, 0.2]:
        ax[0].plot(s.delta_keV, s[f'N_f{f}'], color=BLUES[j], lw=1.5, ls=LS_F[f], label=(f'v_lab = {v:.0f} km/s' if f == 0.01 else None))
ax[0].axhspan(0.105, 3.65, color='#eeeeee', zorder=0); ax[0].axhline(1, color=C['ink'], lw=0.9)
ax[0].axvline(lz.delta_max_kev(248.0, MASS, v_kms=VE_J + VESC), color=C['red'], lw=1.0, ls='-.')
ax[0].text(lz.delta_max_kev(248.0, MASS, v_kms=VE_J + VESC) + 1.5, 3e3, 'SHM / bound\n$\\delta_{\\max}$ = %.0f keV' % lz.delta_max_kev(248.0, MASS, v_kms=VE_J + VESC), fontsize=7, color=C['red'])
ax[0].set_yscale('log'); ax[0].set_ylim(1e-3, 3e4); ax[0].set_xlim(300, 500)
ax[0].set_xlabel('$\\delta$ (keV)'); ax[0].set_ylabel('expected LZ events, pure Higgsino, 1 TeV, 16 June halo')
ax[0].legend(fontsize=7, frameon=False, loc='lower left', title='dotted $f_s$=1%, solid $f_s$=20%', title_fontsize=7)
ax[0].set_title('(a) Composite-halo counts (grey: one-event band 0.105-3.65)')
for k, (d, col) in enumerate(zip([380.0, 400.0, 450.0], [C['orange'], C['violet'], C['aqua']])):
    s = dff[dff.delta_keV == d].sort_values('v_lab_kms')
    ax[1].plot(s.v_lab_kms, s.f_s_for_one_event, color=col, lw=1.8, label=f'$\\delta$ = {d:.0f} keV')
ax[1].axhline(1, color=C['ink'], lw=0.9); ax[1].axhspan(1, 10, color='#eeeeee', zorder=0)
ax[1].axvline(VE_J + VESC, color=C['red'], lw=1.0, ls='-.'); ax[1].text(VE_J + VESC - 4, 2e-4, 'bound streams', fontsize=7.5, color=C['red'], ha='right')
ax[1].text(VE_J + VESC + 4, 2e-4, 'unbound', fontsize=7.5, color=C['red'])
ax[1].set_yscale('log'); ax[1].set_ylim(1e-4, 10); ax[1].set_xlim(600, 1000)
ax[1].set_xlabel('stream lab-frame speed $v_{\\rm lab}$ (km/s), 16 June'); ax[1].set_ylabel('stream density fraction $f_s$ giving one LZ event')
ax[1].legend(fontsize=7.5, frameon=False, loc='upper right'); ax[1].set_title('(b) Required $f_s$ (pure Higgsino, $\\sigma_n$ = 7.4e-39 cm$^2$)')
fig.tight_layout(); fig.savefig(f'{FIG}/P030_fig2_rate_boost.png', dpi=170); plt.close(fig)

# Fig 3: annual modulation of stream signals (anti-aligned bound and unbound streams)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(ts['doy'], ts['vlab_anti_544'], color=C['blue'], lw=1.8, label='retrograde stream, $v_{gal}$ = 544 km/s')
ax[0].plot(ts['doy'], ts['vlab_anti_600'], color=C['violet'], lw=1.8, label='retrograde stream, $v_{gal}$ = 600 km/s (unbound)')
ax[0].axvline(DOY_JUNE, color=C['red'], lw=0.9, ls='-.'); ax[0].text(DOY_JUNE + 3, 795, '16 June', fontsize=7.5, color=C['red'])
ax[0].set_xlabel('day of year'); ax[0].set_ylabel('$v_{\\rm lab}$ (km/s)'); ax[0].legend(fontsize=7.5, frameon=False, loc='lower left'); ax[0].set_title('(a) Lab-frame stream speed over the year')
for key, col, lab in [('N_SHM_d350', C['ink'], 'SHM, $\\delta$ = 350'), ('N_anti_544_d350', C['blue'], 'stream 544, $\\delta$ = 350'),
                      ('N_anti_544_d380', C['aqua'], 'stream 544, $\\delta$ = 380'), ('N_anti_600_d400', C['violet'], 'stream 600 (unbound), $\\delta$ = 400')]:
    y = np.array(ts[key]); m = y.mean()
    ax[1].plot(ts['doy'], y / m if m > 0 else y, color=col, lw=1.8, label=lab)
ax[1].axvline(DOY_JUNE, color=C['red'], lw=0.9, ls='-.')
ax[1].set_xlabel('day of year'); ax[1].set_ylabel('rate / annual mean (pure component)'); ax[1].set_ylim(0, 4.5)
ax[1].legend(fontsize=7.5, frameon=False, loc='upper right'); ax[1].set_title('(b) Modulation of the high-$\\delta$ signal')
fig.tight_layout(); fig.savefig(f'{FIG}/P030_fig3_modulation.png', dpi=170); plt.close(fig)

json.dump(dict(runtime_s=time.time() - T0, vgrid=[0.0, 1100.0, len(VGRID)], sigma_stream=SIGMA_S, deltas=DELTAS.tolist(), vlabs=VLABS.tolist(), mass=MASS),
          open(f'{OUT}/run_meta.json', 'w'), indent=1)
log('done')
