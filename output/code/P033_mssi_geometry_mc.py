#!/usr/bin/env python
"""
P033 -- Geometric probability of a wall/RFR-MSSI topology with a ~12 keV Compton scatter
deep inside the LZ TPC: a weighted photon-transport Monte Carlo.

Run from the simulation root, one stage at a time (each stage < 10 min):
    .venv/bin/python output/code/P033_mssi_geometry_mc.py --stage validation
    .venv/bin/python output/code/P033_mssi_geometry_mc.py --stage grid --source wall     (also bottom, top, cathode)
    .venv/bin/python output/code/P033_mssi_geometry_mc.py --stage bulk
    .venv/bin/python output/code/P033_mssi_geometry_mc.py --stage variants
    .venv/bin/python output/code/P033_mssi_geometry_mc.py --stage summary      (tables, normalisations, figures)
Add --quick for a 10x smaller test.

Geometry (cm): active LXe cylinder R = 72.8, height H = 145.6 (z = 0 cathode, z = H gate);
charge-dead wall shell of thickness T_SHELL just inside the wall; reverse-field region (RFR)
-13.75 < z < 0 (charge-dead); fiducial volume r <= R - 8, 9 <= z <= H - 12.8 (simplified).
Photons undergo Compton (Klein-Nishina, free electrons), photoabsorption (local deposit) or
pair production (treated as local absorption).  Every energy deposit is classified by region.

Variance reduction (all unbiased, weights carried explicitly):
  * survival biasing through active non-FV LXe (a collision there makes an S2 outside the FV or a
    second S2 -> such histories can never be single-scatter FV events; their weight is dropped);
  * the FIRST FV collision is forced with a position uniform along the ray's FV chord, weight
    mu exp(-mu s) L_chord (importance sampling of the depth);
  * in mode 'forced12' the FV Compton deposit is forced into 8-16 keV with the exact KN weight;
  * in dead regions the ray is split into a pass-through branch (weight exp(-mu L), terminal
    if the ray then leaves the LXe) and a forced-collision branch (weight 1 - exp(-mu L));
  * Russian roulette below w = 1e-14.
Mode 'analog' is a plain analog MC used for validation of the weighted estimator.
"""
import sys, os, json, time, math, argparse
import numpy as np
sys.path.insert(0, 'output/code')
from common import lzcommon as lz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('--stage', default='summary'); ap.add_argument('--source', default='wall'); ap.add_argument('--quick', action='store_true')
ap.add_argument('--scale', type=float, default=1.0, help='multiply the photon counts (used for the slower sources)')
ap.add_argument('--lines', default='', help='comma-separated subset of gamma lines (keV) for the grid/variants stages')
ARGS = ap.parse_args()
QUICK = ARGS.quick
OUT = 'output/work/P033'; FIG = os.path.join(OUT, 'figures'); RUNS = os.path.join(OUT, 'runs')
os.makedirs(FIG, exist_ok=True); os.makedirs(RUNS, exist_ok=True)
rng = np.random.default_rng(20260909 + hash(ARGS.stage + ARGS.source) % 1000)

# ----------------------------------------------------------------------------- geometry
R = 72.8; H = 145.6; D_RFR = 13.75
T_SHELL = 0.3                       # dead shell thickness (paper: "up to 3 mm", scalloped)
R_FV = R - 8.0; Z_LO = 9.0; Z_HI = H - 12.8     # 64.8 cm, 9 cm, 132.8 cm
EPS = 1e-5
EVENT = dict(d_wall=lz.LZ['ev_r_from_true_wall_cm'], z=lz.LZ['ev_z_above_cathode_cm'])  # 26.9, 26.4
POS_CLASS = dict(dmin=25.0, zmin=20.0)          # "event position class"
CHUNK = 20000                                   # photons per transport call (memory control)

# ----------------------------------------------------------------------------- LXe physics
ME = 510.999                                    # keV (certain)
RHO = 2.86                                      # g/cm^3 LXe (recalled, likely)
NA = 6.02214e23; A_XE = 131.293; Z_XE = 54
RE2 = 7.9407e-26                                # r_e^2 in cm^2 (certain)
N_E = RHO * NA / A_XE * Z_XE                    # electrons / cm^3

def sigma_kn(E):
    """Klein-Nishina total cross-section per electron (cm^2). E in keV. Exact formula."""
    k = np.asarray(E, float) / ME
    t1 = (1 + k) / k**2 * (2 * (1 + k) / (1 + 2 * k) - np.log(1 + 2 * k) / k)
    t2 = np.log(1 + 2 * k) / (2 * k)
    t3 = (1 + 3 * k) / (1 + 2 * k)**2
    return 2 * np.pi * RE2 * (t1 + t2 - t3)

def dsig_dT(E, T):
    """KN differential cross-section in the electron kinetic energy T (cm^2/keV)."""
    Ep = E - T; eps = Ep / E
    cos = 1.0 - (E / Ep - 1.0) * ME / E
    return np.pi * RE2 * ME / E**2 * (eps + 1.0 / eps - (1.0 - cos**2))

def T_max(E):
    return E * (2 * E / ME) / (1 + 2 * E / ME)

def theta_for_T(E, T):
    cos = 1.0 - (E / (E - T) - 1.0) * ME / E
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

def P_kn_window(E, T1, T2, n=2001):
    E = np.atleast_1d(np.asarray(E, float)); out = np.zeros_like(E)
    for i, e in enumerate(E):
        t2 = min(T2, T_max(e))
        if t2 <= T1: continue
        T = np.linspace(T1, t2, n)
        out[i] = np.trapezoid(dsig_dT(e, T), T) / sigma_kn(e)
    return out

# recalled XCOM-like Xe photoelectric and pair mass-attenuation coefficients (cm^2/g); reliability "likely" (+-30%)
_E_PE = np.array([20, 30, 34.5, 34.6, 40, 50, 60, 80, 100, 122, 150, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000, 3000.])
_MU_PE = np.array([18., 6.5, 4.6, 26., 18., 9.7, 5.9, 2.7, 1.45, 0.85, 0.47, 0.21, 0.068, 0.030, 0.0165, 0.0100, 0.0048,
                   0.0028, 0.0012, 0.00068, 0.00032])
_E_PP = np.array([1022., 1200, 1500, 2000, 2615, 3000])
_MU_PP = np.array([0., 0.0002, 0.0015, 0.0040, 0.0072, 0.0100])  # recalled, uncertain (Xe total at 2.6 MeV ~0.038 cm^2/g -> lambda ~9 cm)
ATT_SCALE = 1.0    # global multiplier on all attenuation coefficients (bracket: 0.8 = 25% longer lambdas)

def mus(E):
    E = np.asarray(E, float)
    mu_c = N_E * sigma_kn(E)
    mu_pe = RHO * np.exp(np.interp(np.log(E), np.log(_E_PE), np.log(_MU_PE)))
    mu_pp = RHO * np.interp(E, _E_PP, _MU_PP, left=0.0)
    s = ATT_SCALE
    return s * (mu_c + mu_pe + mu_pp), s * mu_c, s * mu_pe, s * mu_pp

def kahn_cos(E):
    """Analog KN sampling of cos(theta) by Kahn's rejection method (exact)."""
    E = np.asarray(E, float); n = E.size
    cos = np.empty(n); todo = np.arange(n)
    while todo.size:
        a = E[todo] / ME
        r1, r2, r3 = rng.random((3, todo.size))
        b1 = r1 <= (1 + 2 * a) / (9 + 2 * a)
        rho = np.where(b1, 1 + 2 * a * r2, (1 + 2 * a) / (1 + 2 * a * r2))
        c = 1 - (rho - 1) / a
        acc = np.where(b1, r3 <= 4 * (1 / rho - 1 / rho**2), r3 <= 0.5 * (c**2 + 1 / rho))
        cos[todo[acc]] = c[acc]
        todo = todo[~acc]
    return cos

def rotate(ux, uy, uz, cos, phi):
    sin = np.sqrt(np.clip(1 - cos**2, 0, 1))
    small = np.abs(uz) > 0.99999
    ax = np.where(small, 1.0, -uy); ay = np.where(small, 0.0, ux); az = np.zeros_like(ux)
    nrm = np.sqrt(ax**2 + ay**2); ax = ax / nrm; ay = ay / nrm
    bx = uy * az - uz * ay; by = uz * ax - ux * az; bz = ux * ay - uy * ax
    cp, sp = np.cos(phi), np.sin(phi)
    nx = cos * ux + sin * (cp * ax + sp * bx); ny = cos * uy + sin * (cp * ay + sp * by); nz = cos * uz + sin * (cp * az + sp * bz)
    nrm = np.sqrt(nx**2 + ny**2 + nz**2)
    return nx / nrm, ny / nrm, nz / nrm

# ----------------------------------------------------------------------------- 214Pb beta spectrum
ALPHA_FS = 1 / 137.036; Z_DAU = 83
def beta_sampler(Q, n=4000):
    T = np.linspace(1e-3, Q, n); Ee = T + ME; p = np.sqrt(Ee**2 - ME**2)
    eta = ALPHA_FS * Z_DAU * Ee / p
    F = 2 * np.pi * eta / (1 - np.exp(-2 * np.pi * eta))
    pdf = p * Ee * (Q - T)**2 * F
    cdf = np.concatenate([[0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(T))]); cdf /= cdf[-1]
    P_win = float(np.interp(16, T, cdf) - np.interp(8, T, cdf))
    return (lambda k: np.interp(rng.random(k), cdf, T)), P_win
# 214Pb excited-state branches (recalled, likely): beta -> 352 keV level (endpoint 672 keV) ~42.5 %,
# beta -> 295 keV level (endpoint 729 keV) ~40 %; ground state ~9 % (no gamma: not MSSI). 242 keV cascade neglected.
PB214 = {352.0: dict(Q=672., frac=0.515), 295.0: dict(Q=729., frac=0.485)}

# ----------------------------------------------------------------------------- state handling
FIELDS = ['x', 'y', 'z', 'ux', 'uy', 'uz', 'E', 'w', 'E_fv', 'E_wall', 'E_rfr', 'fv_r', 'fv_z', 'fv_n', 'n_dead']
def empty_state(): return {k: np.empty(0) for k in FIELDS}
def sub(S, m): return {k: v[m] for k, v in S.items()}
def cat(states):
    states = [s for s in states if s['w'].size]
    if not states: return empty_state()
    return {k: np.concatenate([s[k] for s in states]) for k in FIELDS}

def region(x, y, z):
    """0 outside LXe, 1 RFR, 2 wall shell, 3 FV, 4 active non-FV."""
    r2 = x * x + y * y
    reg = np.full(x.shape, 4, dtype=np.int8)
    out = (r2 >= R * R) | (z >= H) | (z <= -D_RFR)
    rfr = (~out) & (z < 0)
    shell = (~out) & (~rfr) & (r2 >= (R - T_SHELL)**2)
    fv = (~out) & (~rfr) & (~shell) & (r2 <= R_FV**2) & (z >= Z_LO) & (z <= Z_HI)
    reg[rfr] = 1; reg[shell] = 2; reg[fv] = 3; reg[out] = 0
    return reg

def t_cyl(x, y, ux, uy, rad):
    a = ux * ux + uy * uy; b = 2 * (x * ux + y * uy); c = x * x + y * y - rad * rad
    disc = b * b - 4 * a * c
    ok = (disc > 0) & (a > 1e-14)
    sq = np.sqrt(np.where(ok, disc, 0.0)); a_s = np.where(ok, a, 1.0)
    t1 = (-b - sq) / (2 * a_s); t2 = (-b + sq) / (2 * a_s)
    t1 = np.where(ok & (t1 > EPS), t1, np.inf); t2 = np.where(ok & (t2 > EPS), t2, np.inf)
    return np.minimum(t1, t2)

def t_plane(z, uz, zp):
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (zp - z) / uz
    return np.where((np.abs(uz) > 1e-14) & (t > EPS), t, np.inf)

def t_next(S):
    x, y, z, ux, uy, uz = (S[k] for k in ['x', 'y', 'z', 'ux', 'uy', 'uz'])
    t = np.full(x.shape, np.inf)
    for rad in (R, R - T_SHELL, R_FV): t = np.minimum(t, t_cyl(x, y, ux, uy, rad))
    for zp in (-D_RFR, 0.0, Z_LO, Z_HI, H): t = np.minimum(t, t_plane(z, uz, zp))
    return t

def deposit(S, reg, T):
    """Add deposit T into the region of each collision (all entries)."""
    rfr = reg == 1; shell = reg == 2; fv = reg == 3
    S['E_rfr'][rfr] += T[rfr]; S['E_wall'][shell] += T[shell]; S['n_dead'][rfr | shell] += 1
    S['E_fv'][fv] += T[fv]; S['fv_n'][fv] += 1
    first = fv & (S['fv_n'] == 1)
    S['fv_r'][first] = np.hypot(S['x'][first], S['y'][first]); S['fv_z'][first] = S['z'][first]

# ----------------------------------------------------------------------------- classification & accumulation
S1_PER_KEV_FV = 69.0 / 12.0     # paper: 12 keV first scatter -> 69 phd
S1_PER_KEV_WALL = 471.0 / 77.0  # paper decomposition (same g1 as bulk)
S1_PER_KEV_RFR = 471.0 / 204.0  # RFR: suppressed recombination
CLASSES = ['wall12', 'rfr12', 'mixed12', 'ss_roi', 'wall_roi', 'rfr_roi', 'wall_any', 'rfr_any']
def classify(S, T_win=(8, 16), wall_win=(60, 95), rfr_win=(150, 260)):
    Efv = S['E_fv']; Ew = S['E_wall']; Er = S['E_rfr']
    one_fv = S['fv_n'] == 1
    S1 = S1_PER_KEV_FV * Efv + S1_PER_KEV_WALL * Ew + S1_PER_KEV_RFR * Er
    in12 = one_fv & (Efv >= T_win[0]) & (Efv <= T_win[1])
    c = {}
    c['wall12'] = in12 & (Ew >= wall_win[0]) & (Ew <= wall_win[1]) & (Er < 1)
    c['rfr12'] = in12 & (Er >= rfr_win[0]) & (Er <= rfr_win[1]) & (Ew < 1)
    c['mixed12'] = in12 & (Er >= 1) & (Ew >= 1) & (S1 >= 480) & (S1 <= 600)
    c['ss_roi'] = one_fv & (Ew < 1) & (Er < 1) & (Efv >= 1.5) & (Efv <= 75)
    c['wall_roi'] = one_fv & (Efv >= 1.5) & (Ew >= 1) & (Er < 1) & (S1 >= 3) & (S1 <= 600)
    c['rfr_roi'] = one_fv & (Efv >= 1.5) & (Er >= 1) & (Ew < 1) & (S1 >= 3) & (S1 <= 600)
    c['wall_any'] = one_fv & (Efv >= 1.5) & (Ew >= 1) & (Er < 1)
    c['rfr_any'] = one_fv & (Efv >= 1.5) & (Er >= 1) & (Ew < 1)
    return c

DBINS = np.concatenate([np.arange(8, 30, 1.0), np.arange(30, 66, 2.5), [R]])   # distance-to-wall bins
ZBINS = np.linspace(0, H, 30)
class Accum:
    def __init__(self):
        self.n0 = 0
        self.sum = {k: 0.0 for k in CLASSES}; self.sum2 = {k: 0.0 for k in CLASSES}; self.cnt = {k: 0 for k in CLASSES}
        self.sum_pos = {k: 0.0 for k in CLASSES}; self.sum2_pos = {k: 0.0 for k in CLASSES}; self.cnt_pos = {k: 0 for k in CLASSES}
        self.sum_near = {k: 0.0 for k in CLASSES}
        self.hd = {k: np.zeros(len(DBINS) - 1) for k in CLASSES}
        self.h2 = {k: np.zeros((len(DBINS) - 1, len(ZBINS) - 1)) for k in ['wall12', 'rfr12', 'wall_roi', 'rfr_roi']}
    def add(self, S):
        """S: terminal states (any). Only fv_n == 1 matter."""
        m1 = S['fv_n'] == 1
        if not m1.any(): return
        S = sub(S, m1); c = classify(S)
        d = R - S['fv_r']; z = S['fv_z']; w = S['w']
        pos = (d >= POS_CLASS['dmin']) & (z >= POS_CLASS['zmin']); near = d <= 10.0
        for k, m in c.items():
            if not m.any(): continue
            self.sum[k] += w[m].sum(); self.sum2[k] += (w[m]**2).sum(); self.cnt[k] += int(m.sum())
            mp = m & pos
            self.sum_pos[k] += w[mp].sum(); self.sum2_pos[k] += (w[mp]**2).sum(); self.cnt_pos[k] += int(mp.sum())
            self.sum_near[k] += w[m & near].sum()
            self.hd[k] += np.histogram(d[m], bins=DBINS, weights=w[m])[0]
            if k in self.h2: self.h2[k] += np.histogram2d(d[m], z[m], bins=[DBINS, ZBINS], weights=w[m])[0]
    def row(self, label, **meta):
        n0 = self.n0; out = dict(label=label, n_emitted=n0, **meta)
        for k in CLASSES:
            out[f'P_{k}'] = self.sum[k] / n0; out[f'err_{k}'] = math.sqrt(self.sum2[k]) / n0; out[f'n_{k}'] = self.cnt[k]
            out[f'P_{k}_pos'] = self.sum_pos[k] / n0; out[f'err_{k}_pos'] = math.sqrt(self.sum2_pos[k]) / n0; out[f'n_{k}_pos'] = self.cnt_pos[k]
            out[f'f_pos_{k}'] = self.sum_pos[k] / self.sum[k] if self.sum[k] > 0 else float('nan')
            out[f'f_near_{k}'] = self.sum_near[k] / self.sum[k] if self.sum[k] > 0 else float('nan')
        return out

# ----------------------------------------------------------------------------- transport
def transport(S, mode, acc, T_win=(8.0, 16.0), max_iter=80, w_rr=1e-14, E_cut=25.0):
    for it in range(max_iter):
        if S['w'].size == 0: break
        reg = region(S['x'] + EPS * S['ux'], S['y'] + EPS * S['uy'], S['z'] + EPS * S['uz'])
        esc = reg == 0
        acc.add(sub(S, esc))
        S = sub(S, ~esc); reg = reg[~esc]
        if S['w'].size == 0: break
        t = t_next(S); t = np.where(np.isfinite(t), t, 1e3)
        mu_t = mus(S['E'])[0]
        P_pass = np.exp(-mu_t * t)
        fv_done = S['fv_n'] > 0
        if mode == 'analog':
            s_free = -np.log(rng.random(S['w'].size)) / mu_t
            coll = s_free < t
            Sp = sub(S, ~coll); tp = t[~coll]
            Sp['x'] = Sp['x'] + (tp + EPS) * Sp['ux']; Sp['y'] = Sp['y'] + (tp + EPS) * Sp['uy']; Sp['z'] = Sp['z'] + (tp + EPS) * Sp['uz']
            cm = coll & (reg != 4) & ~((reg == 3) & fv_done)      # collisions in active non-FV or 2nd FV: invalid -> dropped
            Sc = sub(S, cm); s = s_free[cm]; wfac = np.ones(int(cm.sum())); regc = reg[cm]
        else:
            Sp = {k: v.copy() for k, v in S.items()}
            Sp['w'] *= P_pass
            Sp['x'] += (t + EPS) * Sp['ux']; Sp['y'] += (t + EPS) * Sp['uy']; Sp['z'] += (t + EPS) * Sp['uz']
            dead = (reg == 1) | (reg == 2); fvf = (reg == 3) & (~fv_done)
            cm = dead | fvf
            Sc = sub(S, cm); regc = reg[cm]; mt = mu_t[cm]; tt = t[cm]; Pp = P_pass[cm]
            u = rng.random(int(cm.sum()))
            s = np.where(dead[cm], -np.log(1 - u * (1 - Pp)) / mt, u * tt)
            wfac = np.where(dead[cm], 1 - Pp, mt * np.exp(-mt * s) * tt)
        if Sc['w'].size:
            Sc['x'] = Sc['x'] + s * Sc['ux']; Sc['y'] = Sc['y'] + s * Sc['uy']; Sc['z'] = Sc['z'] + s * Sc['uz']
            Sc['w'] = Sc['w'] * wfac
            mt, mc, mpe, mpp = mus(Sc['E'])
            f_abs = (mpe + mpp) / mt
            if mode == 'analog':
                isabs = rng.random(Sc['w'].size) < f_abs
                Sa = sub(Sc, isabs); deposit(Sa, regc[isabs], Sa['E'].copy()); acc.add(Sa)
                Sc = sub(Sc, ~isabs); regc = regc[~isabs]
            else:
                Sa = {k: v.copy() for k, v in Sc.items()}; Sa['w'] *= f_abs
                deposit(Sa, regc, Sa['E'].copy()); acc.add(Sa)
                Sc['w'] *= (1 - f_abs)
            nC = Sc['w'].size
            if nC:
                E = Sc['E']
                if mode == 'forced12':
                    infv = regc == 3
                    T = np.empty(nC); cos = np.empty(nC)
                    k = int(infv.sum())
                    if k:
                        T1, T2 = T_win
                        Tf = np.minimum(T1 + (T2 - T1) * rng.random(k), 0.999 * T_max(E[infv]))
                        Sc['w'][infv] *= dsig_dT(E[infv], Tf) * (T2 - T1) / sigma_kn(E[infv])
                        T[infv] = Tf; cos[infv] = 1 - (E[infv] / (E[infv] - Tf) - 1) * ME / E[infv]
                    if nC - k:
                        c = kahn_cos(E[~infv]); cos[~infv] = c
                        T[~infv] = E[~infv] - E[~infv] / (1 + E[~infv] / ME * (1 - c))
                else:
                    cos = kahn_cos(E); T = E - E / (1 + E / ME * (1 - cos))
                deposit(Sc, regc, T)
                Sc['E'] = E - T
                phi = 2 * np.pi * rng.random(nC)
                Sc['ux'], Sc['uy'], Sc['uz'] = rotate(Sc['ux'], Sc['uy'], Sc['uz'], cos, phi)
                low = Sc['E'] < E_cut                        # photon absorbed within << 1 mm: local deposit
                if low.any():
                    Sl = sub(Sc, low); deposit(Sl, regc[low], Sl['E'].copy()); acc.add(Sl); Sc = sub(Sc, ~low)
                multi = Sc['fv_n'] > 1                       # two FV deposits: multi-site, cannot be the event
                Sc = sub(Sc, ~multi)
        S = cat([Sp, Sc])
        if mode != 'analog' and S['w'].size:
            low = S['w'] < w_rr
            if low.any():
                keep = ~low | (rng.random(S['w'].size) < S['w'] / w_rr)
                S['w'] = np.where(low, w_rr, S['w']); S = sub(S, keep)

# ----------------------------------------------------------------------------- sources
def lambertian(n, ax):
    u1, u2 = rng.random((2, n)); cos = np.sqrt(u1); phi = 2 * np.pi * u2
    return rotate(ax[:, 0], ax[:, 1], ax[:, 2], cos, phi)
def hemi_iso(n, ax):
    u1, u2 = rng.random((2, n)); return rotate(ax[:, 0], ax[:, 1], ax[:, 2], u1, 2 * np.pi * u2)
def iso(n):
    cos = 2 * rng.random(n) - 1; phi = 2 * np.pi * rng.random(n); sin = np.sqrt(1 - cos**2)
    return sin * np.cos(phi), sin * np.sin(phi), cos

def make_source(kind, E, n, angular='lambert'):
    S = {k: np.zeros(n) for k in FIELDS}; S['w'][:] = 1.0; S['E'][:] = E
    S['fv_r'][:] = np.nan; S['fv_z'][:] = np.nan
    hemi = lambertian if angular == 'lambert' else hemi_iso
    if kind == 'wall':
        phi = 2 * np.pi * rng.random(n); r0 = R - 2 * EPS
        S['x'] = r0 * np.cos(phi); S['y'] = r0 * np.sin(phi); S['z'] = rng.random(n) * H
        S['ux'], S['uy'], S['uz'] = hemi(n, np.stack([-np.cos(phi), -np.sin(phi), np.zeros(n)], 1))
    elif kind in ('bottom', 'top', 'cathode'):
        r = R * np.sqrt(rng.random(n)); phi = 2 * np.pi * rng.random(n)
        S['x'] = r * np.cos(phi); S['y'] = r * np.sin(phi)
        if kind == 'bottom':
            S['z'][:] = -D_RFR + 2 * EPS; S['ux'], S['uy'], S['uz'] = hemi(n, np.tile([0, 0, 1.0], (n, 1)))
        elif kind == 'top':
            S['z'][:] = H - 2 * EPS; S['ux'], S['uy'], S['uz'] = hemi(n, np.tile([0, 0, -1.0], (n, 1)))
        else:
            S['ux'], S['uy'], S['uz'] = iso(n); S['z'] = np.where(S['uz'] > 0, 2 * EPS, -2 * EPS)
    elif kind == 'bulk214Pb':
        r = R * np.sqrt(rng.random(n)); phi = 2 * np.pi * rng.random(n)
        S['x'] = r * np.cos(phi); S['y'] = r * np.sin(phi); S['z'] = -D_RFR + (H + D_RFR) * rng.random(n)
        S['ux'], S['uy'], S['uz'] = iso(n)
        sample, _ = beta_sampler(PB214[E]['Q']); Tb = sample(n)
        reg = region(S['x'], S['y'], S['z'])
        keep = reg != 4                     # beta in active non-FV LXe -> S2 outside the FV: not our event
        S = sub(S, keep); deposit(S, reg[keep], Tb[keep])
    return S

def run(kind, E, mode, n, angular='lambert'):
    acc = Accum(); done = 0
    while done < n:
        m = min(CHUNK, n - done)
        S = make_source(kind, E, m, angular)
        transport(S, mode, acc); acc.n0 += m; done += m
    return acc

def save_run(acc, name, **meta):
    row = acc.row(name, **meta)
    np.savez_compressed(os.path.join(RUNS, name + '.npz'), dbins=DBINS, zbins=ZBINS,
                        **{f'hd_{k}': v for k, v in acc.hd.items()}, **{f'h2_{k}': v for k, v in acc.h2.items()})
    with open(os.path.join(RUNS, name + '.json'), 'w') as f: json.dump(row, f, indent=1, default=float)
    return row

# ----------------------------------------------------------------------------- run plan
LINES = [352., 609., 1120., 1173., 1332., 1461., 1764., 2615.]
# recalled per-decay gamma intensities for a generic U/Th/K/Co component mix (uncertain; results also given per line)
MIX_W = {352.: 0.356, 609.: 0.455, 1120.: 0.149, 1173.: 0.30, 1332.: 0.30, 1461.: 0.107, 1764.: 0.153, 2615.: 0.356}
SOURCES = ['wall', 'bottom', 'top', 'cathode']
N_F = int(1e5) if QUICK else int(5e5)       # forced12 mode, per (source, line)
N_G = int(5e4) if QUICK else int(2e5)       # general mode
N_B = int(2e5) if QUICK else int(1e6)       # bulk 214Pb per line
N_A = int(3e5) if QUICK else int(4e6)       # analog validation
N_F = int(N_F * ARGS.scale); N_G = int(N_G * ARGS.scale); N_B = int(N_B * ARGS.scale)

def stage_validation():
    t0 = time.time(); val = {}
    E = 1000.; c = kahn_cos(np.full(400000, E)); T = E - E / (1 + E / ME * (1 - c))
    h, edges = np.histogram(T, bins=np.linspace(0, T_max(E), 41)); mid = 0.5 * (edges[1:] + edges[:-1])
    ana = dsig_dT(E, mid) / sigma_kn(E) * np.diff(edges) * T.size
    val['kahn_chi2_dof'] = float(((h - ana)**2 / np.maximum(ana, 1)).sum() / (len(h) - 1))
    val['kahn_frac_8_16_sampled'] = float(((T >= 8) & (T <= 16)).mean()); val['kahn_frac_8_16_analytic'] = float(P_kn_window(E, 8, 16)[0])
    val['sigma_kn_numeric_over_analytic_1MeV'] = float(np.trapezoid(dsig_dT(E, np.linspace(1e-6, T_max(E) * (1 - 1e-9), 200001)), np.linspace(1e-6, T_max(E) * (1 - 1e-9), 200001)) / sigma_kn(E))
    # attenuation of a pencil beam through 10 cm (mean free path check of the ray marcher): fraction with no interaction
    acc_a = run('wall', 1461., 'analog', N_A); ra = acc_a.row('analog')
    acc_g = run('wall', 1461., 'general', N_G); rg = acc_g.row('general')
    for k in ['ss_roi', 'wall_roi', 'wall_any', 'rfr_roi', 'rfr_any']:
        val[k] = dict(analog=ra[f'P_{k}'], analog_err=ra[f'err_{k}'], weighted=rg[f'P_{k}'], weighted_err=rg[f'err_{k}'], analog_n=ra[f'n_{k}'], weighted_n=rg[f'n_{k}'],
                      pull=(ra[f'P_{k}'] - rg[f'P_{k}']) / math.sqrt(ra[f'err_{k}']**2 + rg[f'err_{k}']**2 + 1e-300))
    val['depth_bins'] = DBINS.tolist()
    val['depth_wall_any_analog'] = (acc_a.hd['wall_any'] / acc_a.n0).tolist(); val['depth_wall_any_weighted'] = (acc_g.hd['wall_any'] / acc_g.n0).tolist()
    val['depth_rfr_any_analog'] = (acc_a.hd['rfr_any'] / acc_a.n0).tolist(); val['depth_rfr_any_weighted'] = (acc_g.hd['rfr_any'] / acc_g.n0).tolist()
    val['N_analog'] = N_A; val['N_weighted'] = N_G; val['seconds'] = time.time() - t0
    with open(f'{OUT}/P033_validation.json', 'w') as f: json.dump(val, f, indent=1)
    print(json.dumps({k: v for k, v in val.items() if not k.startswith('depth')}, indent=1))

SEL_LINES = [float(x) for x in ARGS.lines.split(',')] if ARGS.lines else None
def stage_grid(kind):
    t0 = time.time()
    for E in (SEL_LINES or LINES):
        acc = run(kind, E, 'forced12', N_F); r1 = save_run(acc, f'{kind}_{E:.0f}_forced12', source=kind, E_keV=E, mode='forced12', att_scale=ATT_SCALE, angular='lambert', shell_cm=T_SHELL)
        acc = run(kind, E, 'general', N_G); r2 = save_run(acc, f'{kind}_{E:.0f}_general', source=kind, E_keV=E, mode='general', att_scale=ATT_SCALE, angular='lambert', shell_cm=T_SHELL)
        print(f'[{time.time()-t0:.0f}s] {kind} {E:.0f}: P(wall12)={r1["P_wall12"]:.3e} f_pos={r1["f_pos_wall12"]:.3e} (n_pos {r1["n_wall12_pos"]}) | P(rfr12)={r1["P_rfr12"]:.3e} f_pos={r1["f_pos_rfr12"]:.3e} | P(ss_roi)={r2["P_ss_roi"]:.3e} P(wall_roi)={r2["P_wall_roi"]:.3e} P(rfr_roi)={r2["P_rfr_roi"]:.3e}', flush=True)

def stage_bulk():
    t0 = time.time()
    for E in PB214:
        _, Pw = beta_sampler(PB214[E]['Q'])
        acc = run('bulk214Pb', E, 'forced12', N_B); r1 = save_run(acc, f'bulk214Pb_{E:.0f}_forced12', source='bulk214Pb', E_keV=E, mode='forced12', att_scale=ATT_SCALE, angular='iso', shell_cm=T_SHELL, P_beta_8_16=Pw)
        acc = run('bulk214Pb', E, 'general', N_B // 2); r2 = save_run(acc, f'bulk214Pb_{E:.0f}_general', source='bulk214Pb', E_keV=E, mode='general', att_scale=ATT_SCALE, angular='iso', shell_cm=T_SHELL, P_beta_8_16=Pw)
        print(f'[{time.time()-t0:.0f}s] bulk {E:.0f}: P_beta(8-16)={Pw:.4f} P(wall12)={r1["P_wall12"]:.3e} pos {r1["P_wall12_pos"]:.2e} | P(rfr12)={r1["P_rfr12"]:.3e} pos {r1["P_rfr12_pos"]:.2e} | ss_roi {r2["P_ss_roi"]:.3e} wall_roi {r2["P_wall_roi"]:.3e} rfr_roi {r2["P_rfr_roi"]:.3e}', flush=True)

def stage_variants():
    global ATT_SCALE, T_SHELL
    t0 = time.time()
    for E in (SEL_LINES or [609., 1461., 2615.]):
        ATT_SCALE = 0.8
        acc = run('wall', E, 'forced12', N_F // 2); r = save_run(acc, f'var_wall_{E:.0f}_forced12_att0.8', source='wall', E_keV=E, mode='forced12', att_scale=0.8, angular='lambert', shell_cm=T_SHELL)
        ATT_SCALE = 1.0
        acc = run('wall', E, 'forced12', N_F // 2, angular='iso'); r2 = save_run(acc, f'var_wall_{E:.0f}_forced12_iso', source='wall', E_keV=E, mode='forced12', att_scale=1.0, angular='iso', shell_cm=T_SHELL)
        print(f'[{time.time()-t0:.0f}s] {E:.0f}: att0.8 P={r["P_wall12"]:.2e} f_pos={r["f_pos_wall12"]:.3e}; iso P={r2["P_wall12"]:.2e} f_pos={r2["f_pos_wall12"]:.3e}', flush=True)
    if SEL_LINES: return
    T_SHELL = 0.11
    acc = run('wall', 1461., 'forced12', N_F // 2); r = save_run(acc, 'var_wall_1461_forced12_shell1.1mm', source='wall', E_keV=1461., mode='forced12', att_scale=1.0, angular='lambert', shell_cm=0.11)
    acc = run('wall', 1461., 'general', N_G // 2); r2 = save_run(acc, 'var_wall_1461_general_shell1.1mm', source='wall', E_keV=1461., mode='general', att_scale=1.0, angular='lambert', shell_cm=0.11)
    print(f'[{time.time()-t0:.0f}s] shell 1.1 mm: P(wall12)={r["P_wall12"]:.2e} f_pos={r["f_pos_wall12"]:.3e}; wall_roi {r2["P_wall_roi"]:.3e} ss_roi {r2["P_ss_roi"]:.3e}', flush=True)
    T_SHELL = 0.3
    # wider FV stand-off (mean 10.7 cm) as a variant of the FV definition
    global R_FV
    R_FV = R - 10.7
    acc = run('wall', 1461., 'forced12', N_F // 2); r = save_run(acc, 'var_wall_1461_forced12_standoff10.7', source='wall', E_keV=1461., mode='forced12', att_scale=1.0, angular='lambert', shell_cm=0.3, standoff=10.7)
    print(f'[{time.time()-t0:.0f}s] stand-off 10.7: P(wall12)={r["P_wall12"]:.2e} f_pos={r["f_pos_wall12"]:.3e}', flush=True)
    R_FV = R - 8.0

def stage_summary():
    import pandas as pd, glob
    rows = [json.load(open(f)) for f in sorted(glob.glob(f'{RUNS}/*.json'))]
    df = pd.DataFrame(rows); df.to_csv(f'{OUT}/P033_runs.csv', index=False)
    main_rows = [r for r in rows if not r['label'].startswith('var_')]
    # KN / attenuation tables
    kn = []
    for E in [242., 295., 352., 500., 609., 1000., 1173., 1461., 1764., 2615.]:
        m = mus(E)
        kn.append(dict(E_keV=E, theta12_deg=float(theta_for_T(E, 12.)), theta8_deg=float(theta_for_T(E, 8.)), theta16_deg=float(theta_for_T(E, 16.)),
                       theta77_deg=float(theta_for_T(E, 77.)), P_8_16=float(P_kn_window(E, 8, 16)[0]), P_10_14=float(P_kn_window(E, 10, 14)[0]),
                       P_60_95=float(P_kn_window(E, 60, 95)[0]), P_150_260=float(P_kn_window(E, 150, 260)[0]),
                       mu_tot_cm=float(m[0]), lambda_cm=float(1 / m[0]), f_compton=float(m[1] / m[0]), P_int_3mm=float(1 - math.exp(-0.3 * m[0]))))
    json.dump(kn, open(f'{OUT}/P033_kn_table.json', 'w'), indent=1)
    pd.DataFrame(kn).to_csv(f'{OUT}/P033_kn_table.csv', index=False)
    att = {str(E): dict(mu=float(mus(E)[0]), lam=float(1 / mus(E)[0]), f_C=float(mus(E)[1] / mus(E)[0])) for E in [100, 122, 200, 295, 352, 500, 1000, 1461, 1764, 2000, 2615]}
    json.dump(att, open(f'{OUT}/P033_attenuation.json', 'w'), indent=1)

    summary = {}
    def mixsum(sel, key, wfun):
        return sum(wfun(r) * r[key] for r in sel)
    for kind in SOURCES + ['surface_equal_mix']:
        srcs = SOURCES if kind == 'surface_equal_mix' else [kind]
        sf = [r for r in main_rows if r['mode'] == 'forced12' and r['source'] in srcs]
        sg = [r for r in main_rows if r['mode'] == 'general' and r['source'] in srcs]
        wf = (lambda r: MIX_W[r['E_keV']] / len(srcs))
        d = dict(n_runs_forced=len(sf), n_runs_general=len(sg))
        for k in ['wall12', 'rfr12', 'mixed12']:
            d[f'P_{k}'] = mixsum(sf, f'P_{k}', wf); d[f'P_{k}_pos'] = mixsum(sf, f'P_{k}_pos', wf)
            d[f'err_{k}_pos'] = math.sqrt(sum((wf(r) * r[f'err_{k}_pos'])**2 for r in sf))
            d[f'f_pos_{k}'] = d[f'P_{k}_pos'] / d[f'P_{k}'] if d[f'P_{k}'] > 0 else float('nan')
            d[f'f_near_{k}'] = mixsum(sf, f'P_{k}', lambda r: wf(r) * (r[f'f_near_{k}'] if r[f'f_near_{k}'] == r[f'f_near_{k}'] else 0)) / d[f'P_{k}'] if d[f'P_{k}'] > 0 else float('nan')
        for k in ['ss_roi', 'wall_roi', 'rfr_roi', 'wall_any', 'rfr_any']:
            d[f'P_{k}'] = mixsum(sg, f'P_{k}', wf); d[f'P_{k}_pos'] = mixsum(sg, f'P_{k}_pos', wf)
            d[f'f_pos_{k}'] = d[f'P_{k}_pos'] / d[f'P_{k}'] if d[f'P_{k}'] > 0 else float('nan')
        d['ratio_wall12pos_to_ssroi'] = d['P_wall12_pos'] / d['P_ss_roi']; d['ratio_rfr12pos_to_ssroi'] = d['P_rfr12_pos'] / d['P_ss_roi']
        d['ratio_wall12_to_ssroi'] = d['P_wall12'] / d['P_ss_roi']; d['ratio_rfr12_to_ssroi'] = d['P_rfr12'] / d['P_ss_roi']
        d['ratio_wallroi_to_ssroi'] = d['P_wall_roi'] / d['P_ss_roi']; d['ratio_rfrroi_to_ssroi'] = d['P_rfr_roi'] / d['P_ss_roi']
        d['rfr12_over_wall12_all'] = d['P_rfr12'] / d['P_wall12'] if d['P_wall12'] > 0 else float('inf')
        d['rfr12_over_wall12_at_pos'] = d['P_rfr12_pos'] / d['P_wall12_pos'] if d['P_wall12_pos'] > 0 else float('inf')
        summary[kind] = d
    bf = [r for r in main_rows if r['source'] == 'bulk214Pb' and r['mode'] == 'forced12']; bg = [r for r in main_rows if r['source'] == 'bulk214Pb' and r['mode'] == 'general']
    if bf:
        d = {}
        wb = lambda r: PB214[r['E_keV']]['frac']
        for k in ['wall12', 'rfr12', 'mixed12']:
            d[f'P_{k}'] = mixsum(bf, f'P_{k}', wb); d[f'P_{k}_pos'] = mixsum(bf, f'P_{k}_pos', wb)
            d[f'err_{k}_pos'] = math.sqrt(sum((wb(r) * r[f'err_{k}_pos'])**2 for r in bf))
        for k in ['ss_roi', 'wall_roi', 'rfr_roi']:
            d[f'P_{k}'] = mixsum(bg, f'P_{k}', wb); d[f'P_{k}_pos'] = mixsum(bg, f'P_{k}_pos', wb)
            d[f'f_pos_{k}'] = d[f'P_{k}_pos'] / d[f'P_{k}'] if d[f'P_{k}'] > 0 else float('nan')
        d['ratio_wall12pos_to_ssroi'] = d['P_wall12_pos'] / d['P_ss_roi'] if d['P_ss_roi'] > 0 else float('nan')
        d['ratio_rfr12pos_to_ssroi'] = d['P_rfr12_pos'] / d['P_ss_roi'] if d['P_ss_roi'] > 0 else float('nan')
        d['ratio_wallroi_to_ssroi'] = d['P_wall_roi'] / d['P_ss_roi']; d['ratio_rfrroi_to_ssroi'] = d['P_rfr_roi'] / d['P_ss_roi']
        d['P_beta_8_16'] = {str(r['E_keV']): r['P_beta_8_16'] for r in bf}
        summary['bulk214Pb'] = d

    # ---------- normalisations
    f_nb = 0.035; N_wall_lz = 0.0048; N_rfr_lz = 0.0001; N_detER_sci = 8.5; N_detER_prompt = 62.2
    sa = summary['surface_equal_mix']; norm = {}
    fpos_range = [summary[k]['f_pos_wall12'] for k in SOURCES]
    for key, fpos in [('wall12_class_equal_mix', sa['f_pos_wall12']), ('wall12_class_min_source', min(fpos_range)), ('wall12_class_max_source', max(fpos_range)),
                      ('wall_roi_general_equal_mix', sa['f_pos_wall_roi'])]:
        N_pos = N_wall_lz * f_nb * fpos
        norm[key] = dict(f_pos=fpos, N_expected_pos=N_pos, k_required_10pct=-math.log(0.9) / N_pos if N_pos > 0 else float('inf'),
                         k_required_50pct=math.log(2) / N_pos if N_pos > 0 else float('inf'), k_req_over_allowed_1p64=(-math.log(0.9) / N_pos) / 1.64 if N_pos > 0 else float('inf'))
    norm['via_detER'] = {}
    for lab, Nn in [('science_8.5_no_veto_correction', N_detER_sci), ('science_plus_prompt_70.7_x_0.06_untagged', (N_detER_sci + N_detER_prompt) * 0.06)]:
        e = {}
        for kind in SOURCES + ['surface_equal_mix']:
            s = summary[kind]
            e[kind] = dict(wall12_pos=Nn * s['ratio_wall12pos_to_ssroi'], rfr12_pos=Nn * s['ratio_rfr12pos_to_ssroi'], wall12_all=Nn * s['ratio_wall12_to_ssroi'],
                           rfr12_all=Nn * s['ratio_rfr12_to_ssroi'], wall_roi_total=Nn * s['ratio_wallroi_to_ssroi'], rfr_roi_total=Nn * s['ratio_rfrroi_to_ssroi'])
        norm['via_detER'][lab] = e
    norm['LZ_model'] = dict(wall_roi_science=N_wall_lz, rfr_roi_science=N_rfr_lz, detER_science=N_detER_sci, detER_prompt=N_detER_prompt, f_nb_P004=f_nb,
                            wall_nb_P004=N_wall_lz * f_nb)
    norm['P004'] = dict(f_pos_range=[1e-4, 0.12], f_pos_central=[0.003, 0.05], k_required_nb_10pct=619, k_allowed_95=1.64)
    summary['normalisation'] = norm
    summary['variants'] = {r['label']: dict(P_wall12=r['P_wall12'], f_pos_wall12=r['f_pos_wall12'], P_wall12_pos=r['P_wall12_pos'], P_rfr12=r['P_rfr12'], f_pos_rfr12=r['f_pos_rfr12'],
                                            P_wall_roi=r.get('P_wall_roi'), P_ss_roi=r.get('P_ss_roi')) for r in rows if r['label'].startswith('var_')}
    json.dump(summary, open(f'{OUT}/P033_summary.json', 'w'), indent=1, default=float)
    print(json.dumps({k: summary[k] for k in ['surface_equal_mix', 'normalisation']}, indent=1, default=float))
    for kind in SOURCES:
        s = summary[kind]; print(f"{kind:8s} P(wall12)={s['P_wall12']:.3e} f_pos={s['f_pos_wall12']:.3e} f_near={s['f_near_wall12']:.3f} | P(rfr12)={s['P_rfr12']:.3e} f_pos={s['f_pos_rfr12']:.3e} | rfr/wall at pos {s['rfr12_over_wall12_at_pos']:.1f} | wall_roi/ss {s['ratio_wallroi_to_ssroi']:.2e} rfr_roi/ss {s['ratio_rfrroi_to_ssroi']:.2e} | f_pos(wall_roi)={s['f_pos_wall_roi']:.3e}")
    if 'bulk214Pb' in summary: print('bulk214Pb', json.dumps(summary['bulk214Pb'], indent=0, default=float))
    make_figures(rows, kn)

def make_figures(rows, kn):
    import glob
    C = dict(wall='#0072B2', bottom='#E69F00', top='#009E73', cathode='#CC79A7', bulk214Pb='#D55E00', event='#000000')
    npz = {os.path.basename(f)[:-4]: np.load(f) for f in glob.glob(f'{RUNS}/*.npz')}
    def combined(cls, mode, key='h2', srcs=SOURCES):
        tot = None
        for name, z in npz.items():
            if name.startswith('var_'): continue
            parts = name.split('_'); src, E, md = parts[0], float(parts[1]), parts[2]
            if md != mode or src not in srcs: continue
            w = MIX_W[E] / len(srcs)
            arr = z[f'{key}_{cls}'] * w
            tot = arr if tot is None else tot + arr
        return tot
    dcent = 0.5 * (DBINS[1:] + DBINS[:-1]); zcent = 0.5 * (ZBINS[1:] + ZBINS[:-1])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, cls, title in zip(axes[:2], ['wall12', 'rfr12'], ['wall-MSSI-like: 12 keV in FV + 60-95 keV in dead shell', 'RFR-MSSI-like: 12 keV in FV + 150-260 keV in RFR']):
        Hs = combined(cls, 'forced12')
        if Hs is None or Hs.sum() <= 0: continue
        Hs = Hs / Hs.sum()
        # normalise per cm^2 (bins are uneven in d)
        area = np.outer(np.diff(DBINS), np.diff(ZBINS)); dens = Hs / area
        im = ax.pcolormesh(DBINS, ZBINS, np.log10(np.maximum(dens.T, 1e-12)), cmap='Blues', vmin=-9, vmax=np.log10(dens.max()), shading='flat')
        ax.add_patch(plt.Rectangle((8, Z_LO), R - 8, Z_HI - Z_LO, fill=False, ls='--', lw=1, color='0.4'))
        ax.plot(EVENT['d_wall'], EVENT['z'], marker='*', ms=14, color=C['event'], ls='none', label='LZ event')
        ax.axvline(25, color='0.5', lw=0.8); ax.axhline(20, color='0.5', lw=0.8)
        ax.set_xlabel('distance of FV deposit from true wall [cm]'); ax.set_ylabel('z above cathode [cm]'); ax.set_title(title, fontsize=9)
        ax.set_xlim(0, R); ax.set_ylim(0, H); ax.legend(loc='upper right', frameon=False)
        cb = fig.colorbar(im, ax=ax, pad=0.02); cb.set_label('log10 probability density [cm^-2]')
    ax = axes[2]
    for kind in SOURCES:
        h = combined('wall12', 'forced12', key='hd', srcs=[kind])
        if h is not None and h.sum() > 0:
            ax.step(DBINS[:-1], h / h.sum() / np.diff(DBINS), where='post', color=C[kind], lw=2, label=f'{kind} source: 12+77 keV class')
    h = combined('wall_roi', 'general', key='hd')
    if h is not None and h.sum() > 0:
        ax.step(DBINS[:-1], h / h.sum() / np.diff(DBINS), where='post', color='0.3', lw=1.2, ls='--', label='all wall MSSI in ROI (all surfaces)')
    h = combined('rfr12', 'forced12', key='hd')
    if h is not None and h.sum() > 0:
        ax.step(DBINS[:-1], h / h.sum() / np.diff(DBINS), where='post', color='0.3', lw=1.2, ls=':', label='RFR 12+200 keV class (all surfaces)')
    ax.axvline(EVENT['d_wall'], color=C['event'], lw=1); ax.text(EVENT['d_wall'] + 0.8, 3e-2, 'event', rotation=90, va='center')
    ax.set_yscale('log'); ax.set_ylim(1e-6, 1); ax.set_xlim(8, R)
    ax.set_xlabel('distance of FV deposit from true wall [cm]'); ax.set_ylabel('normalised density [1/cm]')
    ax.set_title('FV-deposit position of MSSI-like histories', fontsize=9); ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(f'{FIG}/P033_rz_map_and_depth.png', dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    Eg = np.array([k['E_keV'] for k in kn])
    ax = axes[0]
    ax.plot(Eg, [k['P_8_16'] for k in kn], 'o-', color='#0072B2', lw=2, label='T in 8-16 keV')
    ax.plot(Eg, [k['P_60_95'] for k in kn], 's-', color='#E69F00', lw=2, label='T in 60-95 keV')
    ax.plot(Eg, [k['P_150_260'] for k in kn], 'd-', color='#009E73', lw=2, label='T in 150-260 keV')
    ax.plot(Eg, [k['P_int_3mm'] for k in kn], '^-', color='#CC79A7', lw=2, label='P(interact in 3 mm LXe)')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('E_gamma [keV]'); ax.set_ylabel('probability per Compton scatter (or per crossing)'); ax.grid(alpha=0.2); ax.legend(fontsize=8, frameon=False)
    ax.set_title('Klein-Nishina deposit-window probabilities', fontsize=9)
    ax2 = axes[1]
    for kind in SOURCES:
        sel = sorted([r for r in rows if r['source'] == kind and r['mode'] == 'forced12' and not r['label'].startswith('var_')], key=lambda r: r['E_keV'])
        ax2.plot([r['E_keV'] for r in sel], [max(r['f_pos_wall12'], 1e-7) for r in sel], 'o-', color=C[kind], lw=2, label=f'{kind}: wall class')
        ax2.plot([r['E_keV'] for r in sel], [max(r['f_pos_rfr12'], 1e-7) for r in sel], 'o--', color=C[kind], lw=1, alpha=0.7, label=f'{kind}: RFR class')
    ax2.axhspan(1e-4, 0.12, color='0.9', label='P004 f_pos range'); ax2.axhspan(0.003, 0.05, color='0.8', label='P004 central range')
    ax2.set_xscale('log'); ax2.set_yscale('log'); ax2.set_xlabel('E_gamma [keV]'); ax2.set_ylabel('fraction of 12 keV class at d >= 25 cm, z >= 20 cm')
    ax2.set_ylim(1e-7, 1); ax2.legend(fontsize=6.5, frameon=False, ncol=2); ax2.grid(alpha=0.2); ax2.set_title('Position-class fraction f_pos per line and source', fontsize=9)
    fig.tight_layout(); fig.savefig(f'{FIG}/P033_kn_and_fpos.png', dpi=160); plt.close(fig)

def stage_post():
    """Derived numbers: depth-profile slopes, corner fractions, alternative normalisations, 214Pb budget."""
    import glob
    s = json.load(open(f'{OUT}/P033_summary.json'))
    npz = {os.path.basename(f)[:-4]: np.load(f) for f in glob.glob(f'{RUNS}/*.npz') if not os.path.basename(f).startswith('var_')}
    dc = 0.5 * (DBINS[1:] + DBINS[:-1]); dd = np.diff(DBINS)
    def comb(cls, mode, srcs, key='hd'):
        tot = 0
        for name, z in npz.items():
            src, E, md = name.split('_')[0], float(name.split('_')[1]), name.split('_')[2]
            if md != mode or src not in srcs: continue
            tot = tot + z[f'{key}_{cls}'] * (MIX_W[E] if src != 'bulk214Pb' else PB214[E]['frac']) / len(srcs)
        return tot
    post = {}
    # depth profiles: effective attenuation length from a straight-line fit of log(density) between 10 and 40 cm
    prof = {}
    for lab, cls, mode, srcs in [('wall12_wall', 'wall12', 'forced12', ['wall']), ('wall12_top', 'wall12', 'forced12', ['top']), ('wall12_bottom', 'wall12', 'forced12', ['bottom']),
                                 ('wall12_cathode', 'wall12', 'forced12', ['cathode']), ('wall12_equal', 'wall12', 'forced12', SOURCES), ('rfr12_equal', 'rfr12', 'forced12', SOURCES),
                                 ('wall_roi_equal', 'wall_roi', 'general', SOURCES), ('wall12_bulk214Pb', 'wall12', 'forced12', ['bulk214Pb'])]:
        h = comb(cls, mode, srcs); dens = h / h.sum() / dd
        m = (dc >= 10) & (dc <= 40) & (dens > 0)
        slope, icpt = np.polyfit(dc[m], np.log(dens[m]), 1)
        i27 = np.searchsorted(DBINS, 26.9) - 1
        prof[lab] = dict(lambda_eff_cm=float(-1 / slope), dens_8_9=float(dens[0]), dens_at_event=float(dens[i27]), ratio_8_9_over_event=float(dens[0] / dens[i27]) if dens[i27] > 0 else None,
                         frac_d_ge_25=float(h[DBINS[:-1] >= 25].sum() / h.sum()), frac_d_le_10=float(h[DBINS[1:] <= 10].sum() / h.sum()), frac_d_le_15=float(h[DBINS[1:] <= 15].sum() / h.sum()),
                         median_d_cm=float(np.interp(0.5, np.cumsum(h) / h.sum(), DBINS[1:])))
    post['depth_profiles'] = prof
    # bottom-corner fractions (d < 18 cm and z < 20 cm) for ROI MSSI classes; LZ's contour reaches 18.2 cm from the wall near the bottom
    corner = {}
    zc = 0.5 * (ZBINS[1:] + ZBINS[:-1])
    for cls, mode in [('wall_roi', 'general'), ('rfr_roi', 'general'), ('wall12', 'forced12'), ('rfr12', 'forced12')]:
        for srcs, lab in [(SOURCES, 'equal'), (['wall'], 'wall'), (['bottom'], 'bottom')]:
            h2 = comb(cls, mode, srcs, key='h2')
            if np.sum(h2) <= 0: continue
            mc = np.outer(dc < 18, zc < 20)
            corner[f'{cls}_{lab}'] = dict(frac_bottom_corner=float(h2[mc].sum() / h2.sum()), frac_z_lt_20=float(h2[:, zc < 20].sum() / h2.sum()), frac_z_gt_125=float(h2[:, zc > 125].sum() / h2.sum()))
    post['corner_fractions'] = corner
    # neighbourhood box around the event: |d - 26.9| <= 5 cm and |z - 26.4| <= 5 cm (approximate with bins)
    nb = {}
    for cls, mode in [('wall12', 'forced12'), ('rfr12', 'forced12')]:
        h2 = comb(cls, mode, SOURCES, key='h2')
        md = (dc >= 21.9) & (dc <= 31.9); mz = (zc >= 21.4) & (zc <= 31.4)
        nb[cls] = float(h2[np.ix_(md, mz)].sum() / h2.sum())
    post['frac_in_10x10cm_box_around_event'] = nb
    # alternative normalisations
    sa = s['surface_equal_mix']
    f_E_wall = sa['P_wall12'] / sa['P_wall_roi']; f_E_rfr = sa['P_rfr12'] / sa['P_rfr_roi']
    alt = dict(energy_class_fraction_wall=f_E_wall, energy_class_fraction_rfr=f_E_rfr, P004_f_nb=0.035)
    alt['wall12_pos_from_LZ_0.0048_own_fE'] = 0.0048 * f_E_wall * sa['f_pos_wall12']
    alt['rfr12_pos_from_LZ_0.0001_own_fE'] = 0.0001 * f_E_rfr * sa['f_pos_rfr12']
    alt['rfr12_pos_from_LZ_0.0001_bottom_source'] = 0.0001 * (s['bottom']['P_rfr12'] / s['bottom']['P_rfr_roi']) * s['bottom']['f_pos_rfr12']
    alt['k_required_10pct_wall_own_fE'] = -math.log(0.9) / alt['wall12_pos_from_LZ_0.0048_own_fE']
    # combined MSSI at the position class (wall + RFR, LZ normalisations)
    alt['total_MSSI12_pos_LZ_norm'] = alt['wall12_pos_from_LZ_0.0048_own_fE'] + alt['rfr12_pos_from_LZ_0.0001_own_fE']
    # 214Pb budget (recalled, uncertain): ground-state branch f_gs ~ 0.09; internal-beta ROI events ~ 1341 of which ~75 % 214Pb (assumption)
    sample_gs, _ = beta_sampler(1019.)
    Tgs = sample_gs(400000); P_gs_roi = float(((Tgs >= 1.5) & (Tgs <= 75)).mean())
    f_gs = 0.09; N_pb_roi = 0.75 * 1341
    N_gs_FV = N_pb_roi / P_gs_roi; N_exc_FV = N_gs_FV * (1 - f_gs) / f_gs
    V_all_t = math.pi * R**2 * (H + D_RFR) * RHO / 1e6; N_exc_all = N_exc_FV * V_all_t / 4.71
    b = s['bulk214Pb']
    alt['Pb214'] = dict(P_gs_beta_in_1p5_75=P_gs_roi, f_gs=f_gs, N_Pb214_ROI_assumed=N_pb_roi, N_gs_decays_FV=N_gs_FV, N_excited_decays_FV=N_exc_FV, V_LXe_t=V_all_t, N_excited_decays_all=N_exc_all,
                        wall12_all=N_exc_all * b['P_wall12'], wall12_pos=N_exc_all * b['P_wall12_pos'], rfr12_pos=N_exc_all * b['P_rfr12_pos'],
                        wall_roi_total_pre_veto=N_exc_all * b['P_wall_roi'], wall_roi_total_x0p06=N_exc_all * b['P_wall_roi'] * 0.06,
                        wall_roi_total_x0p06_shell1p1mm=N_exc_all * b['P_wall_roi'] * 0.06 * 0.37, f_pos_wall12=b['P_wall12_pos'] / b['P_wall12'])
    # detector-ER cross-check of the wall-MSSI total with shell and veto corrections
    alt['detER_wall_roi_total_equal_mix'] = dict(raw_3mm=8.5 * sa['ratio_wallroi_to_ssroi'], shell1p1mm=8.5 * sa['ratio_wallroi_to_ssroi'] * 0.37,
                                                  shell1p1mm_veto_ratio_0p5=8.5 * sa['ratio_wallroi_to_ssroi'] * 0.37 * 0.5, LZ=0.0048)
    alt['detER_rfr_roi_total_equal_mix'] = dict(raw=8.5 * sa['ratio_rfrroi_to_ssroi'], x0p06=8.5 * sa['ratio_rfrroi_to_ssroi'] * 0.06, LZ=0.0001)
    post['normalisation_alt'] = alt
    json.dump(post, open(f'{OUT}/P033_post.json', 'w'), indent=1, default=float)
    print(json.dumps(post, indent=1, default=float))

if __name__ == '__main__':
    st = ARGS.stage
    if st == 'post': stage_post(); raise SystemExit
    if st == 'validation': stage_validation()
    elif st == 'grid': stage_grid(ARGS.source)
    elif st == 'bulk': stage_bulk()
    elif st == 'variants': stage_variants()
    elif st == 'summary': stage_summary()
    else: raise SystemExit('unknown stage')
