"""
P018 -- The high-velocity tail: escape velocity, non-Maxwellian tails and the reach of
inelastic dark matter at LZ.

Run from the simulation root:  .venv/bin/python output/code/P018_halo_tail.py

Parts
  1  delta_max(248 keV, m) vs v_esc (500-600 km/s) and Earth speed (June / mean / December), analytic.
  2  Numerical Earth-frame halo functions eta(v_min) for a family of Galactic-frame velocity
     distributions (truncated Maxwellian, King-like smooth truncations, SHM++-like two-component
     with an anisotropic Sausage), boosted with WimPyDD's Earth-velocity vector for 16 June (day 167)
     and the 12-day annual mean; validated against lz.wd_halo / lz.eta0 for the Maxwellian.
  3  O1 inelastic in-ROI event counts (WimPyDD per-stream kernels x delta_eta) for m = 400/1000/4000 GeV,
     delta = 250/300/350/380 keV, per halo variant, relative to the standard halo; coupling inferred from
     one event; shift of the Higgsino window delta(N=1) (P007) per halo variant.
  4  delta above which the halo-induced spread of the rate exceeds x10.

Conventions (checked in Part 2): WimPyDD's (vmin_i, delta_eta_i) are upper interval boundaries and
delta_eta_i = eta(v_{i-1}) - eta(v_i) >= 0 in (km/s)^-1, so eta(v_min) = sum_{v_i > v_min} delta_eta_i and
diff_rate(..., sum_over_streams=False) with unit weights gives a kernel K(E, v_i) with dR/dE = K @ delta_eta.
"""
import sys, os, json, math, time
import numpy as np
sys.path.insert(0, 'output/code')
from common import lzcommon as lz
from scipy.special import erf, erfc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'output/work/P018'; FIG = OUT + '/figures'
os.makedirs(FIG, exist_ok=True)
T0 = time.time()
WD = lz.wd()

V0, VESC = lz.V0_KMS, lz.VESC_KMS          # 238, 544 km/s (Baxter 2021, used by LZ)
VSUN_PEC = lz.V_SUN_PEC.copy()             # (11.1, 12.2, 7.3) km/s
VGRID = np.linspace(0.0, 900.0, 1801)      # upper boundaries, 0.5 km/s; beyond v_max for v_esc = 600 + v_E
DV = VGRID[1] - VGRID[0]
DOY_JUNE, DOY_DEC = 167, 350
DAYS12 = 15.0 + 365.25 / 12 * np.arange(12)
EXPOSURE = lz.LZ['exposure_tyr']
E50_HI, E50_LO, PLATEAU = lz.LZ['E_50pct_high_keV'], lz.LZ['E_50pct_low_keV'], lz.LZ['eff_plateau']
MASSES = [400.0, 1000.0, 4000.0]
DELTAS_TAB = [250.0, 300.0, 350.0, 380.0]
DELTAS_FINE = np.round(np.concatenate([np.arange(100.0, 250.0, 25.0), np.arange(250.0, 340.0, 10.0), np.arange(340.0, 402.5, 2.5), np.arange(405.0, 441.0, 5.0)]), 2)

def log(*a):
    print('[%6.1fs]' % (time.time() - T0), *a, flush=True)

# ==============================================================================================
# Part 1: delta_max(248 keV) vs v_esc and v_E  (analytic, lzcommon kinematics)
# ==============================================================================================
def v_sun_speed(v0):
    return float(np.linalg.norm(VSUN_PEC + np.array([0.0, v0, 0.0])))

VE = {'june16': lz.v_earth_kms(DOY_JUNE), 'mean': lz.v_earth_kms(), 'dec16': lz.v_earth_kms(DOY_DEC)}
vesc_scan = np.arange(500.0, 600.1, 5.0)
slope = math.sqrt(2 * lz.m_nucleus_gev(lz.A_XE_MEAN) * 248e-6) * 1e6 / lz.C_KMS   # keV per km/s of v_max
rows = []
for ep, vE in VE.items():
    for m in MASSES:
        for ve in vesc_scan:
            rows.append(dict(epoch=ep, v_E_kms=vE, m_GeV=m, v_esc_kms=ve, v_max_kms=vE + ve,
                             delta_max_keV=lz.delta_max_kev(248.0, m, v_kms=vE + ve)))
import pandas as pd
df1 = pd.DataFrame(rows); df1.to_csv(f'{OUT}/deltamax_vs_vesc.csv', index=False)
# v0 dependence enters only through v_E = |v_sun| (+ orbital term)
rows_v0 = []
for v0 in [220.0, 238.0, 250.0]:
    vs = v_sun_speed(v0)
    for m in MASSES:
        rows_v0.append(dict(v0_kms=v0, v_sun_kms=vs, m_GeV=m,
                            delta_max_june_keV=lz.delta_max_kev(248.0, m, v_kms=vs + 15.0 * math.cos(2 * math.pi * (DOY_JUNE - 153) / 365.25) + VESC),
                            delta_max_mean_keV=lz.delta_max_kev(248.0, m, v_kms=vs + VESC)))
df1b = pd.DataFrame(rows_v0); df1b.to_csv(f'{OUT}/deltamax_vs_v0.csv', index=False)
part1 = dict(slope_keV_per_kms=slope, v_E=VE,
             delta_max_std={ep: {int(m): lz.delta_max_kev(248.0, m, v_kms=vE + VESC) for m in MASSES} for ep, vE in VE.items()},
             delta_max_vesc500_june={int(m): lz.delta_max_kev(248.0, m, v_kms=VE['june16'] + 500.0) for m in MASSES},
             delta_max_vesc600_june={int(m): lz.delta_max_kev(248.0, m, v_kms=VE['june16'] + 600.0) for m in MASSES},
             delta_max_vesc500_dec={int(m): lz.delta_max_kev(248.0, m, v_kms=VE['dec16'] + 500.0) for m in MASSES},
             delta_max_vesc600_june_v0_250={int(m): lz.delta_max_kev(248.0, m, v_kms=v_sun_speed(250.0) + 14.6 + 600.0) for m in MASSES},
             delta_max_vesc500_dec_v0_220={int(m): lz.delta_max_kev(248.0, m, v_kms=v_sun_speed(220.0) - 14.6 + 500.0) for m in MASSES},
             mu_v2_ceiling_note='delta <= mu v_max^2/2 is the absolute ceiling; delta_max(248) is below it for all m here')
json.dump(part1, open(f'{OUT}/part1_deltamax.json', 'w'), indent=1)
log('Part 1: slope d(delta_max)/d(v_max) = %.4f keV/(km/s); delta_max June std = %s' % (slope, part1['delta_max_std']['june16']))

# ==============================================================================================
# Part 2: numerical halo functions
# ==============================================================================================
NC, NPHI = 48, 64
GLX, GLW = np.polynomial.legendre.leggauss(NC)
PHI = np.linspace(0.0, 2 * np.pi, NPHI, endpoint=False); WPHI = 2 * np.pi / NPHI

def v_obs_vec(doy, v0=V0):
    """Earth velocity in the Galactic rest frame (WimPyDD axes: x to GC, y rotation, z NGP), km/s."""
    v_sun = np.array([0.0, v0, 0.0]) + VSUN_PEC
    if doy is None:
        return v_sun
    lam = (doy - 80) * 2.0 * np.pi / 365.0                      # same phase convention as WimPyDD.streamed_halo_function
    return v_sun + WD.v_earth_sun(lam, v_rot_gal=np.array([0.0, v0, 0.0]), v_sun_rot=VSUN_PEC)

# --- Galactic-frame distributions f(ux,uy,uz), un-normalised, implicitly truncated at |u| < vesc ---
def f_maxwell(ux, uy, uz, v0, vesc):
    u2 = ux**2 + uy**2 + uz**2
    return np.exp(-u2 / v0**2) * (u2 <= vesc**2)

def f_king(ux, uy, uz, v0, vesc, k):
    """Lisanti et al. (2011) smoothly truncated form f ~ [exp(-u^2/v0^2) - exp(-vesc^2/v0^2)]^k."""
    u2 = ux**2 + uy**2 + uz**2
    g = np.exp(-u2 / v0**2) - math.exp(-vesc**2 / v0**2)
    return np.where(g > 0, np.abs(g)**k, 0.0)

def sausage_sigmas(v0, beta):
    """SHM++ (Evans, O'Hare, McCabe 2019): sigma_r^2 = 3 v0^2/(2(3-2beta)), sigma_theta^2 = sigma_phi^2 = 3 v0^2 (1-beta)/(2(3-2beta)).
    Total dispersion = 3 v0^2/2, same as the round Maxwellian."""
    sr2 = 3 * v0**2 / (2 * (3 - 2 * beta)); st2 = 3 * v0**2 * (1 - beta) / (2 * (3 - 2 * beta))
    return math.sqrt(sr2), math.sqrt(st2)

def f_sausage(ux, uy, uz, v0, vesc, beta):
    """Anisotropic Gaussian, radial axis = x (toward GC); tangential (y, rotation) and vertical (z) narrow."""
    sr, st = sausage_sigmas(v0, beta)
    u2 = ux**2 + uy**2 + uz**2
    return np.exp(-ux**2 / (2 * sr**2) - uy**2 / (2 * st**2) - uz**2 / (2 * st**2)) * (u2 <= vesc**2)

def norm_gal(f, vesc, nu=400):
    """Normalisation integral over the truncated sphere in the Galactic frame."""
    xu, wu = np.polynomial.legendre.leggauss(nu)
    u = 0.5 * vesc * (xu + 1); wu = 0.5 * vesc * wu
    c = GLX[None, :, None]; s = np.sqrt(1 - c**2)
    ph = PHI[None, None, :]
    U = u[:, None, None]
    val = f(U * s * np.cos(ph), U * s * np.sin(ph), U * c)
    inner = (val * WPHI).sum(-1); inner = (inner * GLW[None, :]).sum(-1)
    return float(np.sum(wu * u**2 * inner))

def shell_S(vs, f, vobs, vesc):
    """S(v) = v * Int dOmega f_gal(v n + v_obs) over the cap |v n + v_obs| <= vesc. Polar axis along -v_obs so the
    cap is cos(theta) in [c_min, 1] with c_min = (v^2 + vo^2 - vesc^2)/(2 v vo): the integrand is smooth on the cap."""
    vo = float(np.linalg.norm(vobs)); e3 = -vobs / vo
    e1 = np.cross(e3, [1.0, 0.0, 0.0]); e1 /= np.linalg.norm(e1); e2 = np.cross(e3, e1)
    S = np.zeros_like(vs)
    for i0 in range(0, len(vs), 300):
        v = vs[i0:i0 + 300]
        with np.errstate(divide='ignore', invalid='ignore'):
            cmin = np.where(v > 0, (v**2 + vo**2 - vesc**2) / (2 * v * vo), -1.0)
        cmin = np.clip(cmin, -1.0, 1.0)
        half = 0.5 * (1 - cmin)                                    # (nv,)
        c = (0.5 * (1 + cmin))[:, None] + half[:, None] * GLX[None, :]   # (nv, nc)
        wc = half[:, None] * GLW[None, :]
        s = np.sqrt(np.clip(1 - c**2, 0, None))
        n = (s[..., None, None] * np.cos(PHI)[None, None, :, None] * e1
             + s[..., None, None] * np.sin(PHI)[None, None, :, None] * e2
             + c[..., None, None] * e3)                              # (nv, nc, nphi, 3)
        u = v[:, None, None, None] * n + vobs
        val = f(u[..., 0], u[..., 1], u[..., 2])
        inner = (val * WPHI).sum(-1)
        inner = (inner * wc).sum(-1)
        S[i0:i0 + 300] = v * inner
    return S

def delta_eta_numeric(f, vobs, vesc, norm):
    """delta_eta_i on VGRID (upper boundaries) by Simpson on each interval; eta(v) = Int_v^inf S dv."""
    mids = VGRID[1:] - 0.5 * DV
    S_grid = shell_S(VGRID, f, vobs, vesc) / norm
    S_mid = shell_S(mids, f, vobs, vesc) / norm
    de = np.zeros_like(VGRID)
    de[1:] = DV / 6.0 * (S_grid[:-1] + 4 * S_mid + S_grid[1:])
    return de

def eta_from_deta(de):
    """eta at the grid points VGRID (eta(VGRID[i]) = sum_{j>i} de[j])."""
    return np.cumsum(de[::-1])[::-1] - de

class Halo:
    def __init__(self, name, label, family, comps, v0_obs):
        """comps: list of (weight, f(ux,uy,uz), vesc)."""
        self.name, self.label, self.family, self.comps, self.v0_obs = name, label, family, comps, v0_obs
        self.norms = [norm_gal(f, ve) for _, f, ve in comps]
        self.vesc_max = max(ve for _, _, ve in comps)
    def deta(self, doy):
        vobs = v_obs_vec(doy, self.v0_obs)
        return sum(w * delta_eta_numeric(f, vobs, ve, nrm) for (w, f, ve), nrm in zip(self.comps, self.norms))
    def deta_annual(self):
        return np.mean([self.deta(d) for d in DAYS12], axis=0)

def mk_maxwell(v0, vesc):
    return [(1.0, lambda x, y, z, v0=v0, ve=vesc: f_maxwell(x, y, z, v0, ve), vesc)]
def mk_king(v0, vesc, k):
    return [(1.0, lambda x, y, z, v0=v0, ve=vesc, k=k: f_king(x, y, z, v0, ve, k), vesc)]
def mk_shmpp(v0, vesc, beta, eta_s, isotropic=False):
    if isotropic:
        sr, _ = sausage_sigmas(v0, beta)
        v0b = math.sqrt(2.0) * sr                         # isotropic Maxwellian with per-axis sigma = sigma_r ("broader radial Maxwellian")
        saus = lambda x, y, z, v0b=v0b, ve=vesc: f_maxwell(x, y, z, v0b, ve)
    else:
        saus = lambda x, y, z, v0=v0, ve=vesc, b=beta: f_sausage(x, y, z, v0, ve, b)
    return [(1 - eta_s, lambda x, y, z, v0=v0, ve=vesc: f_maxwell(x, y, z, v0, ve), vesc), (eta_s, saus, vesc)]

HALOS = [
    Halo('std',        'SHM v0=238, v_esc=544 (sharp)',      'std',   mk_maxwell(V0, VESC), V0),
    Halo('vesc500',    'v_esc = 500',                        'vesc',  mk_maxwell(V0, 500.0), V0),
    Halo('vesc528',    'v_esc = 528',                        'vesc',  mk_maxwell(V0, 528.0), V0),
    Halo('vesc560',    'v_esc = 560',                        'vesc',  mk_maxwell(V0, 560.0), V0),
    Halo('vesc600',    'v_esc = 600',                        'vesc',  mk_maxwell(V0, 600.0), V0),
    Halo('v0_220',     'v0 = 220 (v_esc 544)',               'v0',    mk_maxwell(220.0, VESC), 220.0),
    Halo('v0_250',     'v0 = 250 (v_esc 544)',               'v0',    mk_maxwell(250.0, VESC), 250.0),
    Halo('king1',      'King k = 1',                         'king',  mk_king(V0, VESC, 1), V0),
    Halo('king2',      'King k = 2',                         'king',  mk_king(V0, VESC, 2), V0),
    Halo('king3',      'King k = 3',                         'king',  mk_king(V0, VESC, 3), V0),
    Halo('shmpp',      'SHM++ 80% round + 20% Sausage (beta=0.9, anisotropic)', 'shmpp', mk_shmpp(V0, VESC, 0.9, 0.2), V0),
    Halo('shmpp_iso',  'SHM++ Sausage approximated by isotropic sigma_r Maxwellian', 'shmpp', mk_shmpp(V0, VESC, 0.9, 0.2, isotropic=True), V0),
    Halo('shmpp_evans','SHM++ literal: v0=233, v_esc=528, beta=0.9, 20% Sausage', 'shmpp', mk_shmpp(233.0, 528.0, 0.9, 0.2), 233.0),
    Halo('env_low',    'envelope low: v0=220, v_esc=500, King k=3', 'env', mk_king(220.0, 500.0, 3), 220.0),
    Halo('env_high',   'envelope high: v0=250, v_esc=600, sharp',   'env', mk_maxwell(250.0, 600.0), 250.0),
]
SINGLE_PARAM = ['vesc500', 'vesc528', 'vesc560', 'vesc600', 'v0_220', 'v0_250', 'king1', 'king2', 'king3', 'shmpp', 'shmpp_evans']
TIERS = {'gaia': ['std', 'vesc528', 'vesc560', 'v0_220', 'v0_250', 'shmpp', 'shmpp_evans'],   # Gaia-era parameter range, sharp/SHM++ tails
         'shape': ['std', 'king1', 'king2', 'king3'],                                          # tail shape at fixed v0, v_esc
         'single': ['std'] + SINGLE_PARAM,                                                     # all single-parameter variations
         'envelope': ['env_low', 'env_high']}                                                  # combined extremes

# --- validation of the construction against WimPyDD / lzcommon for the standard Maxwellian ---
val = {}
std = HALOS[0]
an_norm = math.pi**1.5 * V0**3 * (erf(VESC / V0) - 2 * VESC / V0 / math.sqrt(math.pi) * math.exp(-(VESC / V0)**2))
val['norm_numeric_vs_analytic'] = dict(numeric=std.norms[0], analytic=an_norm, ratio=std.norms[0] / an_norm)
de_mine = std.deta(DOY_JUNE)
vm_wd, de_wd = lz.wd_halo(day_of_year=DOY_JUNE, vmin=VGRID)
eta_mine, eta_wd = eta_from_deta(de_mine), eta_from_deta(de_wd)
vobs_j = v_obs_vec(DOY_JUNE); vE_j = float(np.linalg.norm(vobs_j))
val['v_obs_june16'] = dict(vector=vobs_j.tolist(), speed=vE_j, cosine_model_speed=VE['june16'], v_max=vE_j + VESC)
val['eta0_total'] = dict(mine=float(de_mine.sum()), wimpydd=float(de_wd.sum()), analytic_lz_eta0=float(lz.eta0(0.0, v_e=vE_j)))
pts = [0, 200, 400, 500, 600, 700, 750, 780, 800]
val['eta_vs_vmin'] = [dict(vmin=p, mine=float(eta_mine[int(p / DV)]), wimpydd=float(eta_wd[int(p / DV)]), analytic=float(lz.eta0(float(p), v_e=vE_j)))
                      for p in pts]
val['max_abs_deta_diff_over_eta0'] = float(np.max(np.abs(de_mine - de_wd)) / de_mine.sum())
log('Part 2 validation: norm ratio %.6f; eta(0) mine/WD/analytic = %.5e / %.5e / %.5e' % (
    val['norm_numeric_vs_analytic']['ratio'], val['eta0_total']['mine'], val['eta0_total']['wimpydd'], val['eta0_total']['analytic_lz_eta0']))
for r in val['eta_vs_vmin']:
    log('   vmin=%4d  eta mine/WD = %.5f  mine/analytic = %.5f' % (r['vmin'], r['mine'] / r['wimpydd'] if r['wimpydd'] > 0 else float('nan'),
                                                                  r['mine'] / r['analytic'] if r['analytic'] > 0 else float('nan')))

# --- build all delta_eta (June and annual) ---
log('building %d halo variants x (June, annual 12 days) ...' % len(HALOS))
DETA = {}
for h in HALOS:
    DETA[(h.name, 'june16')] = h.deta(DOY_JUNE)
    DETA[(h.name, 'annual')] = h.deta_annual()
    log('  %-12s done; eta(0) June = %.4e, eta(700) June = %.3e, v_max(June) = %.1f' % (
        h.name, DETA[(h.name, 'june16')].sum(), eta_from_deta(DETA[(h.name, 'june16')])[1400], VGRID[DETA[(h.name, 'june16')] > 0].max()))
DETA[('std', 'dec16')] = std.deta(DOY_DEC)
DETA[('std_wimpydd', 'june16')] = de_wd
DETA[('std_wimpydd', 'annual')] = np.mean([lz.wd_halo(day_of_year=d, vmin=VGRID)[1] for d in DAYS12], axis=0)

# eta tables
tab = {'vmin_kms': VGRID}
for (name, ep), de in DETA.items():
    tab[f'eta_{name}_{ep}'] = eta_from_deta(de)
pd.DataFrame(tab).to_csv(f'{OUT}/eta_tails.csv', index=False, float_format='%.6e')

# tail fractions: eta(v)/eta_std(v) at v = 600, 700, 750, 780, 800 (June)
tail_rows = []
eta_std_j = eta_from_deta(DETA[('std', 'june16')])
for h in HALOS:
    e = eta_from_deta(DETA[(h.name, 'june16')])
    tail_rows.append(dict(halo=h.name, **{f'ratio_eta_{v}': float(e[int(v / DV)] / eta_std_j[int(v / DV)]) for v in [400, 600, 700, 750, 780, 800]},
                          vmax_june=float(VGRID[DETA[(h.name, 'june16')] > 0].max())))
pd.DataFrame(tail_rows).to_csv(f'{OUT}/eta_tail_ratios_june.csv', index=False, float_format='%.4g')

# ==============================================================================================
# Part 3: kernels and in-ROI counts
# ==============================================================================================
def efficiency(E, sig_hi=11.5, sig_lo=2.5):
    """LZ NR efficiency: plateau 0.96, 50% at 5.4 and 269.9 keV, erf edges (sig_hi = 11.5 keV reproduces Fig. S2 inset; P007)."""
    return PLATEAU * 0.5 * (1 + erf((E - E50_LO) / (math.sqrt(2) * sig_lo))) * 0.5 * erfc((E - E50_HI) / (math.sqrt(2) * sig_hi))

MV = lz.M_V_GEV
GF, SW2 = 1.166e-5, 0.231                      # recalled, certain (as P007)
c_p = (GF / math.sqrt(2)) * (1 - 4 * SW2); c_n = -GF / math.sqrt(2)
ham_iso = lz.wd_hamiltonian('iso_unit', {1: (2.0 / MV**2, 0.0)})              # c_p = c_n = 1/m_v^2 (LZ unit coupling)
ham_hig = lz.wd_hamiltonian('higgsino_Z', {1: (c_p + c_n, c_p - c_n)})       # P007 Higgsino, sigma_n = 7.4e-39 cm^2
ONES = np.ones_like(VGRID)

def kernel(ham, m, delta, dE=2.0, E_max=330.0):
    lo = min(lz.E_R_range_keV(m, VGRID[-1], A=float(A), delta_kev=delta)[0] for A in lz.XE_ISOTOPES)
    if math.isnan(lo):
        return None, None
    E = np.arange(max(1.0, math.floor(lo) - 4.0), E_max + 1e-9, dE)
    K = np.array([WD.diff_rate(WD.Xe, ham, m, float(e), VGRID, ONES, j_chi=0.5, delta=float(delta), sum_over_streams=False) for e in E])
    return E, K * 1000.0 * 365.25          # events/(t yr keV) per unit delta_eta

def N_roi(E, K, de):
    r = K @ de
    return float(np.trapezoid(r * efficiency(E), E)) * EXPOSURE

# validation of rates: my std vs WimPyDD std delta_eta
E, K = kernel(ham_iso, 1000.0, 300.0)
val['rate_check_1000GeV_delta300'] = dict(N_mine=N_roi(E, K, DETA[('std', 'june16')]), N_wimpydd=N_roi(E, K, DETA[('std_wimpydd', 'june16')]))
E, K = kernel(ham_iso, 1000.0, 380.0)
val['rate_check_1000GeV_delta380'] = dict(N_mine=N_roi(E, K, DETA[('std', 'june16')]), N_wimpydd=N_roi(E, K, DETA[('std_wimpydd', 'june16')]))
E, K = kernel(ham_iso, 1000.0, 350.0)
val['rate_check_1000GeV_delta350_annual'] = dict(N_mine=N_roi(E, K, DETA[('std', 'annual')]), N_wimpydd=N_roi(E, K, DETA[('std_wimpydd', 'annual')]))
for k_, v_ in val.items():
    if k_.startswith('rate_check'):
        v_['ratio'] = v_['N_mine'] / v_['N_wimpydd']
        log('  %s: mine %.4e  WimPyDD %.4e  ratio %.5f' % (k_, v_['N_mine'], v_['N_wimpydd'], v_['ratio']))
json.dump(val, open(f'{OUT}/validation.json', 'w'), indent=1)

# --- isoscalar unit coupling: tabulated deltas ---
log('Part 3a: isoscalar O1 counts at delta = %s' % DELTAS_TAB)
rows = []
for m in MASSES:
    for d in DELTAS_TAB:
        E, K = kernel(ham_iso, m, d)
        for ep in ['june16', 'annual']:
            Nstd = N_roi(E, K, DETA[('std', ep)]) if E is not None else 0.0
            for h in HALOS:
                N = N_roi(E, K, DETA[(h.name, ep)]) if E is not None else 0.0
                rows.append(dict(m_GeV=m, delta_keV=d, epoch=ep, halo=h.name, family=h.family, N_unit=N, N_std=Nstd,
                                 ratio=N / Nstd if Nstd > 0 else float('nan'),
                                 log10_ratio=(math.log10(N / Nstd) if N > 0 else -np.inf) if Nstd > 0 else float('nan'),   # N = 0: rate vanishes -> -inf
                                 c2_one_event=1.0 / N if N > 0 else float('inf'),          # (c^s m_v^2)^2 giving N = 1
                                 log10_c2_shift=(-math.log10(N / Nstd) if N > 0 else np.inf) if Nstd > 0 else float('nan')))
df3 = pd.DataFrame(rows); df3.to_csv(f'{OUT}/rate_ratios_isoscalar.csv', index=False, float_format='%.5g')

def spread(sub, names):
    s = sub[sub.halo.isin(names) & ~sub.log10_ratio.isna()]
    if len(s) == 0:
        return float('nan'), float('nan'), float('nan')
    return float(s.log10_ratio.max() - s.log10_ratio.min()), float(s.log10_ratio.min()), float(s.log10_ratio.max())

summ = []
for m in MASSES:
    for d in DELTAS_TAB:
        for ep in ['june16', 'annual']:
            sub = df3[(df3.m_GeV == m) & (df3.delta_keV == d) & (df3.epoch == ep)]
            row = dict(m_GeV=m, delta_keV=d, epoch=ep, N_std=float(sub.N_std.iloc[0]))
            for tier, names in TIERS.items():
                sp, lo, hi = spread(sub, names)
                row.update({f'spread_{tier}_dex': sp, f'min_{tier}': lo, f'max_{tier}': hi})
            row['spread_envelope_plus_rho_dex'] = row['spread_envelope_dex'] + math.log10(0.5 / 0.3)
            summ.append(row)
df3s = pd.DataFrame(summ); df3s.to_csv(f'{OUT}/rate_spread_summary.csv', index=False, float_format='%.4g')
print(df3s[df3s.epoch == 'june16'][['m_GeV', 'delta_keV', 'N_std', 'spread_gaia_dex', 'min_gaia', 'max_gaia', 'spread_shape_dex', 'spread_single_dex', 'spread_envelope_dex']].to_string(index=False))

# --- Higgsino coupling on the fine delta grid: N(delta) per halo; delta(N=1) ---
log('Part 3b: Higgsino N(delta) on fine grid, %d deltas x %d masses' % (len(DELTAS_FINE), len(MASSES)))
rows = []
KERN = {}
for m in MASSES:
    for d in DELTAS_FINE:
        E, K = kernel(ham_hig, m, float(d))
        KERN[(m, float(d))] = (E, K)
        for ep in ['june16', 'annual']:
            for h in HALOS:
                rows.append(dict(m_GeV=m, delta_keV=float(d), epoch=ep, halo=h.name, family=h.family,
                                 N=N_roi(E, K, DETA[(h.name, ep)]) if E is not None else 0.0))
    log('  mass %.0f done' % m)
dfh = pd.DataFrame(rows); dfh.to_csv(f'{OUT}/higgsino_N_vs_delta.csv', index=False, float_format='%.5g')

def delta_for_N(x, y, target):
    y = np.log10(np.maximum(y, 1e-12)); lt = math.log10(target)
    for i in range(len(x) - 1):
        if (y[i] - lt) * (y[i + 1] - lt) <= 0 and y[i] != y[i + 1] and y[i] > -11:
            return float(x[i] + (lt - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
    return float('nan')

win = []
for m in MASSES:
    for ep in ['june16', 'annual']:
        ref = dfh[(dfh.m_GeV == m) & (dfh.epoch == ep) & (dfh.halo == 'std')].sort_values('delta_keV')
        d1_std = delta_for_N(ref.delta_keV.values, ref.N.values, 1.0)
        for h in HALOS:
            s = dfh[(dfh.m_GeV == m) & (dfh.epoch == ep) & (dfh.halo == h.name)].sort_values('delta_keV')
            d1 = delta_for_N(s.delta_keV.values, s.N.values, 1.0)
            win.append(dict(m_GeV=m, epoch=ep, halo=h.name, family=h.family,
                            delta_N1=d1, delta_N3p65=delta_for_N(s.delta_keV.values, s.N.values, 3.65),
                            delta_N0p105=delta_for_N(s.delta_keV.values, s.N.values, 0.105),
                            shift_N1_vs_std=d1 - d1_std,
                            N_at_300=float(s[s.delta_keV == 300.0].N.iloc[0]), N_at_350=float(s[s.delta_keV == 350.0].N.iloc[0]),
                            N_at_366=float(np.interp(366.0, s.delta_keV.values, s.N.values))))
dfw = pd.DataFrame(win); dfw.to_csv(f'{OUT}/higgsino_window_shift.csv', index=False, float_format='%.4g')
print(dfw[(dfw.m_GeV == 1000.0) & (dfw.epoch == 'annual')].to_string(index=False))

# ==============================================================================================
# Part 4: spread of rate vs delta; delta above which astrophysics dominates (x10)
# ==============================================================================================
rows = []
for m in MASSES:
    for ep in ['june16', 'annual']:
        for d in DELTAS_FINE:
            sub = dfh[(dfh.m_GeV == m) & (dfh.epoch == ep) & (dfh.delta_keV == float(d))]
            Nstd = float(sub[sub.halo == 'std'].N.iloc[0])
            if Nstd <= 0:
                continue
            lr = {r.halo: math.log10(r.N / Nstd) if r.N > 0 else -np.inf for r in sub.itertuples()}
            row = dict(m_GeV=m, epoch=ep, delta_keV=float(d), N_std=Nstd)
            for tier, names in TIERS.items():
                row[f'spread_{tier}_dex'] = max(lr[n] for n in names) - min(lr[n] for n in names)
            row['spread_envelope_plus_rho_dex'] = row['spread_envelope_dex'] + math.log10(5 / 3)
            row.update({f'lr_{k}': v for k, v in lr.items()})
            rows.append(row)
df4 = pd.DataFrame(rows); df4.to_csv(f'{OUT}/spread_vs_delta.csv', index=False, float_format='%.4g')

def first_crossing(x, y, thr=1.0):
    """First delta at which the spread rises through thr (inf counts as above; NaN rows dropped)."""
    y = np.asarray(y, float); x = np.asarray(x, float)
    ok = ~np.isnan(y)
    x, y = x[ok], y[ok]
    if len(y) and y[0] >= thr:
        return float(x[0]) * -1.0          # negative flag: already above threshold at the lowest delta scanned
    for i in range(len(x) - 1):
        if y[i] < thr <= y[i + 1]:
            if np.isinf(y[i + 1]):
                return float(x[i + 1])
            return float(x[i] + (thr - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
    return float('nan')

cross = []
for m in MASSES:
    for ep in ['june16', 'annual']:
        s = df4[(df4.m_GeV == m) & (df4.epoch == ep)].sort_values('delta_keV')
        row = dict(m_GeV=m, epoch=ep, delta_max_248_june=lz.delta_max_kev(248.0, m, v_kms=VE['june16'] + VESC))
        for tier in TIERS:
            row[f'delta_x3_{tier}'] = first_crossing(s.delta_keV, s[f'spread_{tier}_dex'], math.log10(3))
            row[f'delta_x10_{tier}'] = first_crossing(s.delta_keV, s[f'spread_{tier}_dex'], 1.0)
            row[f'delta_x100_{tier}'] = first_crossing(s.delta_keV, s[f'spread_{tier}_dex'], 2.0)
        row['delta_x10_envelope_rho'] = first_crossing(s.delta_keV, s.spread_envelope_plus_rho_dex, 1.0)
        cross.append(row)
dfc = pd.DataFrame(cross); dfc.to_csv(f'{OUT}/astro_dominance_delta.csv', index=False, float_format='%.4g')
print(dfc.to_string(index=False))
print(df4[(df4.m_GeV == 1000.0) & (df4.epoch == 'june16')][['delta_keV', 'N_std', 'spread_gaia_dex', 'spread_shape_dex', 'spread_single_dex', 'spread_envelope_dex']].to_string(index=False))

# ==============================================================================================
# Figures
# ==============================================================================================
C = dict(blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', yellow='#eda100', magenta='#e87ba4', green='#008300', violet='#4a3aa7', red='#e34948')
STYLE = {'std': ('#222222', '-', 2.4), 'vesc500': (C['blue'], ':', 1.6), 'vesc528': (C['blue'], '--', 1.6), 'vesc560': (C['blue'], '-.', 1.6),
         'vesc600': (C['blue'], '-', 1.6), 'v0_220': (C['orange'], '--', 1.6), 'v0_250': (C['orange'], '-', 1.6),
         'king1': (C['aqua'], '-', 1.6), 'king2': (C['aqua'], '--', 1.6), 'king3': (C['aqua'], ':', 1.8),
         'shmpp': (C['magenta'], '-', 1.8), 'shmpp_iso': (C['magenta'], ':', 1.6), 'shmpp_evans': (C['violet'], '-', 1.6),
         'env_low': (C['red'], ':', 1.4), 'env_high': (C['red'], '-', 1.4)}
LAB = {h.name: h.label for h in HALOS}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25})

fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
for h in HALOS:
    e = eta_from_deta(DETA[(h.name, 'june16')]); col, ls, lw = STYLE[h.name]
    ax[0].semilogy(VGRID, e, color=col, ls=ls, lw=lw, label=LAB[h.name])
    with np.errstate(divide='ignore', invalid='ignore'):
        ax[1].semilogy(VGRID, e / eta_std_j, color=col, ls=ls, lw=lw)
for a in ax:
    for m, dd in [(1000, 300), (1000, 350), (1000, 380)]:
        vmn = lz.vmin_kms(248.0, m, delta_kev=dd)
        a.axvline(vmn, color='#888888', lw=0.8, ls='--')
    a.set_xlim(300, 880); a.set_xlabel('$v_{\\min}$ (km/s), Earth frame, 16 June')
ax[0].set_ylim(1e-8, 5e-3); ax[0].set_ylabel('$\\eta(v_{\\min})$  [(km/s)$^{-1}$]')
ax[1].set_ylim(1e-3, 1e3); ax[1].set_ylabel('$\\eta / \\eta_{\\rm SHM}$'); ax[1].axhline(1, color='#222222', lw=1)
ax[0].text(lz.vmin_kms(248.0, 1000, delta_kev=300) + 3, 2e-3, '$v_{\\min}$(248 keV, 1 TeV)\n$\\delta$=300', fontsize=7, color='#666666')
ax[0].text(lz.vmin_kms(248.0, 1000, delta_kev=350) + 3, 2e-3, '350', fontsize=7, color='#666666')
ax[0].text(lz.vmin_kms(248.0, 1000, delta_kev=380) + 3, 2e-3, '380', fontsize=7, color='#666666')
ax[0].legend(fontsize=6.3, loc='lower left', frameon=False)
ax[0].set_title('(a) Halo functions, 16 June 2023'); ax[1].set_title('(b) Ratio to the sharp SHM (v0=238, v_esc=544)')
fig.tight_layout(); fig.savefig(f'{FIG}/P018_fig1_eta_tails.png', dpi=170); plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(12.5, 4.3), sharey=True)
for i, m in enumerate(MASSES):
    s = df4[(df4.m_GeV == m) & (df4.epoch == 'june16')].sort_values('delta_keV')
    for h in HALOS:
        if h.name == 'std' or h.name == 'shmpp_iso':
            continue
        col, ls, lw = STYLE[h.name]
        ax[i].plot(s.delta_keV, 10**s[f'lr_{h.name}'], color=col, ls=ls, lw=lw, label=LAB[h.name] if i == 0 else None)
    ax[i].axhline(1, color='#222222', lw=1.2); ax[i].axhline(10, color='#888888', lw=0.8, ls='--'); ax[i].axhline(0.1, color='#888888', lw=0.8, ls='--')
    ax[i].set_yscale('log'); ax[i].set_ylim(1e-3, 1e3); ax[i].set_xlim(100, 420)
    ax[i].axvline(lz.delta_max_kev(248.0, m, v_kms=VE['june16'] + VESC), color='#222222', lw=0.8, ls=':')
    ax[i].set_title('m$_\\chi$ = %.0f GeV (16 June)' % m); ax[i].set_xlabel('$\\delta$ (keV)')
    dx = dfc[(dfc.m_GeV == m) & (dfc.epoch == 'june16')].delta_x10_gaia.iloc[0]
    if np.isfinite(dx) and dx > 0:
        ax[i].axvspan(dx, 400, color='#eeeeee', zorder=0)
        ax[i].text(dx + 1, 3e-3, 'Gaia-range spread > x10\n$\\delta$ > %.0f keV' % dx, fontsize=7, color='#555555')
ax[0].set_ylabel('in-ROI rate / rate(sharp SHM), O1 inelastic')
ax[0].legend(fontsize=6.3, loc='upper left', frameon=False)
fig.tight_layout(); fig.savefig(f'{FIG}/P018_fig2_rate_ratio_vs_delta.png', dpi=170); plt.close(fig)

# Higgsino window figure (1 TeV, annual)
fig, ax = plt.subplots(figsize=(6.2, 4.3))
s0 = dfh[(dfh.m_GeV == 1000.0) & (dfh.epoch == 'annual')]
for h in HALOS:
    if h.name == 'shmpp_iso':
        continue
    s = s0[s0.halo == h.name].sort_values('delta_keV'); col, ls, lw = STYLE[h.name]
    ax.plot(s.delta_keV, s.N, color=col, ls=ls, lw=lw, label=LAB[h.name])
ax.axhspan(0.105, 3.65, color='#eeeeee', zorder=0); ax.axhline(1, color='#222222', lw=1)
ax.set_yscale('log'); ax.set_ylim(1e-3, 1e4); ax.set_xlim(300, 400)
ax.set_xlabel('$\\delta$ (keV)'); ax.set_ylabel('expected LZ events, pure Higgsino, 1 TeV (annual mean)')
ax.legend(fontsize=6, frameon=False, loc='upper right'); fig.tight_layout(); fig.savefig(f'{FIG}/P018_fig3_higgsino_window.png', dpi=170); plt.close(fig)

json.dump(dict(runtime_s=time.time() - T0, n_halos=len(HALOS), vgrid=[float(VGRID[0]), float(VGRID[-1]), len(VGRID)], NC=NC, NPHI=NPHI,
               days12=DAYS12.tolist(), sausage_sigmas_238_beta0p9=sausage_sigmas(V0, 0.9)), open(f'{OUT}/run_meta.json', 'w'), indent=1)
log('done')
