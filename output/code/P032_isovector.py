"""P032: Why the isovector O1 gets 1.3 sigma at delta = 0 while the isoscalar gets none:
isospin, the M-response node and what isospin violation can buy.

Run from the simulation root:  .venv/bin/python output/code/P032_isovector.py  [--fast]

Physics
-------
O1 (spin-independent) scattering with nucleon couplings c_p, c_n.  WimPyDD convention (settled by P003):
  c^0 = c_p + c_n,  c^1 = c_p - c_n.   With r = f_n/f_p = c_n/c_p:  c^0 = c_p (1 + r),  c^1 = c_p (1 - r).
The rate is a quadratic form in (c^0, c^1):
  dR/dE(c0, c1) = c0^2 R00 + 2 c0 c1 R01 + c1^2 R11,
where R_tt' are the WimPyDD spectra built from the shell-model M^{tt'} responses (DMFormFactor-v6 density
matrices).  We therefore compute three WimPyDD spectra per mass ((1,0), (0,1), (1,1)) and obtain any r exactly.
  r = +1  isoscalar (LZ O1^s),  r = -1 isovector (LZ O1^v),  r = 0 proton-only (P011),  r = -Z/N ~ -0.7 xenophobic.

Outputs -> output/work/P032/ (CSV/JSON) and output/work/P032/figures/ (PNG).
"""
import os, sys, json, time, math, argparse
import numpy as np
import pandas as pd
from scipy import special, integrate, optimize, stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, 'output/code')
from common import lzcommon as lz

ap = argparse.ArgumentParser()
ap.add_argument('--fast', action='store_true', help='reuse cached WimPyDD spectra if present')
ARGS = ap.parse_args()

OUT = 'output/work/P032'; FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)
LOG = open(os.path.join(OUT, 'run_log.txt'), 'w')
def log(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.write(s + '\n'); LOG.flush()

# palette (dataviz skill reference instance, light mode)
C = dict(blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', yellow='#eda100', magenta='#e87ba4',
         green='#008300', violet='#4a3aa7', red='#e34948', ink='#0b0b0b', ink2='#52514e', muted='#898781',
         grid='#e1e0d9', surface='#fcfcfb')
plt.rcParams.update({'figure.facecolor': C['surface'], 'axes.facecolor': C['surface'], 'axes.edgecolor': C['muted'],
                     'axes.labelcolor': C['ink2'], 'xtick.color': C['ink2'], 'ytick.color': C['ink2'],
                     'grid.color': C['grid'], 'axes.grid': True, 'grid.linewidth': 0.6, 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})

WD = lz.wd(); xe = WD.Xe; NC = WD.nuclear_current
MASSES = [400.0, 1000.0, 4000.0]
DELTA_IN = 300.0                     # keV, inelastic reference splitting
E_EVENT = 248.0
Z_XE = 54
CP = 1.0                             # GeV^-2 reference proton coupling (ratios are coupling-independent)
E_ROI_LO, E_LO_MAX, E_HI, E_HI_MAX, E_EDGE = 5.4, 55.0, 200.0, 270.0, 269.9

# -------------------------------------------------------------------------------------------------
# 0. Efficiency (P003 model, used by P016/P021): 0.96 plateau, erf edges at 5.4 (sigma 3.4) and 269.9 keV (sigma 8)
# -------------------------------------------------------------------------------------------------
SIG_LO, SIG_HI, EFF0 = 3.4, 8.0, 0.96
def efficiency(E):
    E = np.asarray(E, float)
    return EFF0 * 0.5 * (1 + special.erf((E - E_ROI_LO) / (np.sqrt(2) * SIG_LO))) * \
           0.5 * (1 - special.erf((E - E_EDGE) / (np.sqrt(2) * SIG_HI)))

E_FINE = np.linspace(1.0, 300.0, 5981)   # 0.05 keV
def window(E, dR, e1, e2):
    f = np.interp(E_FINE, E, dR)
    m = (E_FINE >= e1) & (E_FINE <= e2)
    return float(np.trapezoid(f[m] * efficiency(E_FINE[m]), E_FINE[m]))

# -------------------------------------------------------------------------------------------------
# 1. WimPyDD basis spectra R00, R01, R11 (annual-average Baxter-2021 halo, natural Xe, events/(t yr keV))
# -------------------------------------------------------------------------------------------------
halo = lz.wd_halo()
log(f'halo: {len(halo[0])} v_min bins, v_max = {halo[0][-1]:.1f} km/s (annual average, Baxter 2021)')
E_EL = np.union1d(np.arange(1.0, 300.0 + 0.5, 1.0), [5.4, 55.0, 248.0, 269.9])
E_IN = np.union1d(np.arange(40.0, 300.0 + 0.5, 1.0), [248.0, 269.9])
HAMS = {'A': lz.wd_hamiltonian('O1_c0', {1: (1.0, 0.0)}),      # c0 = 1, c1 = 0
        'B': lz.wd_hamiltonian('O1_c1', {1: (0.0, 1.0)}),      # c0 = 0, c1 = 1
        'C': lz.wd_hamiltonian('O1_c0c1', {1: (1.0, 1.0)}),    # c0 = c1 = 1  (proton-only, c_p = 1)
        'D': lz.wd_hamiltonian('O1_chk', {1: (0.3, 1.7)})}     # bilinearity check point (r = -0.7)

CACHE = os.path.join(OUT, 'P032_basis_spectra.npz')
if ARGS.fast and os.path.exists(CACHE):
    S = dict(np.load(CACHE)); log('loaded cached basis spectra')
else:
    S = {'E_EL': E_EL, 'E_IN': E_IN}
    t0 = time.time()
    for m in MASSES:
        for k, h in HAMS.items():
            S[f'el_{k}_{int(m)}'] = lz.wd_rate(h, m, E_EL, halo)
            log(f'  elastic  {k} m={m:6.0f}  ({time.time()-t0:.0f} s)')
        for k in ('A', 'B', 'C'):
            S[f'in_{k}_{int(m)}'] = lz.wd_rate(HAMS[k], m, E_IN, halo, delta_kev=DELTA_IN)
            log(f'  inelastic {k} m={m:6.0f} delta={DELTA_IN:.0f} ({time.time()-t0:.0f} s)')
    # per-isotope basis rates at 248 keV (1 TeV, elastic and delta = 300)
    for j, name in enumerate(xe.isotopes):
        for k in ('A', 'B', 'C'):
            S[f'iso248_{k}_{name}'] = np.array([WD.diff_rate(xe, HAMS[k], 1000.0, E_EVENT, halo[0], halo[1],
                                                             isotopes_list={0: [j]}) * 1000.0 * 365.25])
            S[f'iso248in_{k}_{name}'] = np.array([WD.diff_rate(xe, HAMS[k], 1000.0, E_EVENT, halo[0], halo[1],
                                                               delta=DELTA_IN, isotopes_list={0: [j]}) * 1000.0 * 365.25])
            S[f'iso20_{k}_{name}'] = np.array([WD.diff_rate(xe, HAMS[k], 1000.0, 20.0, halo[0], halo[1],
                                                            isotopes_list={0: [j]}) * 1000.0 * 365.25])
    np.savez(CACHE, **S)
    log(f'basis spectra done in {time.time()-t0:.0f} s -> {CACHE}')

def basis(prefix, m=None, name=None):
    key = (lambda k: f'{prefix}_{k}_{int(m)}') if name is None else (lambda k: f'{prefix}_{k}_{name}')
    RA, RB, RC = S[key('A')], S[key('B')], S[key('C')]
    return RA, RB, 0.5 * (RC - RA - RB)          # R00, R11, R01

def rate_r(r, R00, R11, R01, cp=CP):
    c0, c1 = cp * (1 + r), cp * (1 - r)
    return c0 ** 2 * R00 + 2 * c0 * c1 * R01 + c1 ** 2 * R11

# normalisation validation: isoscalar WimPyDD rate for sigma_p = sigma_n = 1e-45 cm^2 vs the Helm dR/dE of lzcommon
# (P003: agreement <= 5 % below 100 keV).  c_p = sqrt(pi sigma / (hbar c)^2) / mu_p, WimPyDD c0 = 2 c_p.
_HBARC2 = 0.3894e-27
for m in (1000.0,):
    mu_p = m * lz.M_NUCLEON_GEV / (m + lz.M_NUCLEON_GEV)
    cp_1e45 = math.sqrt(math.pi * 1e-45 / _HBARC2) / mu_p
    R00, R11, R01 = basis('el', m)
    for E0 in (20.0, 50.0):
        wd_val = float(np.interp(E0, E_EL, rate_r(1.0, R00, R11, R01, cp=cp_1e45)))
        helm_val = float(lz.dRdE_SI(E0, m, 1e-45))
        log(f'normalisation check m={m:.0f} E={E0:.0f} keV sigma=1e-45: WimPyDD {wd_val:.4e}  Helm(lzcommon) {helm_val:.4e}  ratio {wd_val / helm_val:.3f}')

# bilinearity check against the directly computed (0.3, 1.7) spectrum
chk = []
for m in MASSES:
    R00, R11, R01 = basis('el', m)
    pred = 0.3 ** 2 * R00 + 2 * 0.3 * 1.7 * R01 + 1.7 ** 2 * R11
    direct = S[f'el_D_{int(m)}']
    sel = (E_EL >= 5) & (E_EL <= 265) & (direct > 0)
    chk.append(float(np.max(np.abs(pred[sel] / direct[sel] - 1))))
log(f'bilinearity check (c0,c1)=(0.3,1.7): max |pred/direct - 1| over 5-265 keV = {max(chk):.2e}')

# -------------------------------------------------------------------------------------------------
# 2. Scan in r = f_n/f_p: window rates, N_lo, effective-A, node positions
# -------------------------------------------------------------------------------------------------
R_GRID = np.round(np.arange(-2.0, 2.0001, 0.02), 4)
R_KEY = {'isoscalar (r=+1)': 1.0, 'proton-only (r=0)': 0.0, 'xenophobic (r=-0.7)': -0.7, 'isovector (r=-1)': -1.0}
# natural-Xe mean N for the xenophobic value
N_MEAN = sum(A * f for A, f in lz.XE_ISOTOPES.items()) / sum(lz.XE_ISOTOPES.values()) - Z_XE
R_XENO = -Z_XE / N_MEAN
log(f'<N> = {N_MEAN:.2f}, r_xeno = -Z/<N> = {R_XENO:.4f}')

def local_minima(E, y, lo, hi, depth=0.6, half_width=15.0):
    """positions of significant local minima of y(E) in (lo, hi), parabolic refinement.
    A minimum counts only if y_min < depth * y at both E -+ half_width (rejects grid ripples)."""
    out = []
    for i in range(1, len(E) - 1):
        if lo < E[i] < hi and y[i] < y[i - 1] and y[i] <= y[i + 1]:
            yl = np.interp(E[i] - half_width, E, y); yr = np.interp(E[i] + half_width, E, y)
            if not (y[i] < depth * yl and y[i] < depth * yr):
                continue
            x0, x1, x2 = E[i - 1], E[i], E[i + 1]; y0, y1, y2 = y[i - 1], y[i], y[i + 1]
            d = (y0 - 2 * y1 + y2)
            out.append(float(x1 + 0.5 * (y0 - y2) / d * (x1 - x0)) if d > 0 else float(x1))
    return out

rows = []
for m in MASSES:
    R00, R11, R01 = basis('el', m)
    R_iso = rate_r(1.0, R00, R11, R01)
    for r in R_GRID:
        R = rate_r(r, R00, R11, R01)
        R_lo, R_L2, R_M, R_hi = (window(E_EL, R, 5.4, 55), window(E_EL, R, 55, 125), window(E_EL, R, 125, 200),
                                 window(E_EL, R, 200, 270))
        mins = local_minima(E_EL, R, 60.0, 300.0)
        n1 = [x for x in mins if x < 150]; n2 = [x for x in mins if x >= 150]
        rows.append(dict(m_chi_GeV=m, r=r, R_lo=R_lo, R_55_125=R_L2, R_125_200=R_M, R_hi=R_hi,
                         N_lo=R_lo / R_hi if R_hi > 0 else np.inf, N_L2=R_L2 / R_hi, N_M=R_M / R_hi,
                         N_55_200=(R_L2 + R_M) / R_hi,
                         rho_lo=R_lo / window(E_EL, R_iso, 5.4, 55), rho_hi=R_hi / window(E_EL, R_iso, 200, 270),
                         rho_248=float(np.interp(248, E_EL, R) / np.interp(248, E_EL, R_iso)),
                         rho_q0=((Z_XE + r * N_MEAN) / (Z_XE + N_MEAN)) ** 2,
                         node1_keV=n1[0] if n1 else np.nan, node2_keV=n2[0] if n2 else np.nan,
                         dRdE_248=float(np.interp(248, E_EL, R)),
                         R_hi_per_cp2=R_hi))
scan = pd.DataFrame(rows)
scan.to_csv(os.path.join(OUT, 'P032_r_scan.csv'), index=False)

# key points
key_rows = []
for lab, r in R_KEY.items():
    for m in MASSES:
        q = scan[(scan.m_chi_GeV == m) & (np.isclose(scan.r, r))].iloc[0]
        key_rows.append(dict(label=lab, r=r, m_chi_GeV=m, **{k: q[k] for k in ['R_lo', 'R_55_125', 'R_125_200', 'R_hi', 'N_lo', 'N_55_200',
                                                                   'rho_lo', 'rho_hi', 'rho_248', 'rho_q0', 'node1_keV', 'node2_keV']}))
# extra points outside the grid: neutron-only (c_p = 0: c0 = c_n, c1 = -c_n) and the P007 Higgsino (c1/c0 = -1.17)
R_HIGGS = (1 + 1.17) / (1 - 1.17)          # r = c_n/c_p = (c0 - c1)/(c0 + c1) with c1/c0 = -1.17  -> -12.8
for lab, (c0, c1), r in (('neutron-only (c_p=0)', (1.0, -1.0), np.inf), ('Higgsino (c1/c0=-1.17)', (1.0, -1.17), R_HIGGS)):
    for m in MASSES:
        R00, R11, R01 = basis('el', m)
        R = c0 ** 2 * R00 + 2 * c0 * c1 * R01 + c1 ** 2 * R11
        # reference: isoscalar with c_p = c_n = |c_n| of the model (so 1/rho_q0 is LZ's "vector" conversion factor)
        cp_m, cn_m = (c0 + c1) / 2.0, (c0 - c1) / 2.0
        R_iso = rate_r(1.0, R00, R11, R01, cp=abs(cn_m))
        R_lo, R_L2, R_M, R_hi = (window(E_EL, R, 5.4, 55), window(E_EL, R, 55, 125), window(E_EL, R, 125, 200), window(E_EL, R, 200, 270))
        mn = local_minima(E_EL, R, 60, 300); n1 = [x for x in mn if x < 150]; n2 = [x for x in mn if x >= 150]
        key_rows.append(dict(label=lab, r=r, m_chi_GeV=m, R_lo=R_lo, R_55_125=R_L2, R_125_200=R_M, R_hi=R_hi, N_lo=R_lo / R_hi,
                             N_55_200=(R_L2 + R_M) / R_hi, rho_lo=R_lo / window(E_EL, R_iso, 5.4, 55), rho_hi=R_hi / window(E_EL, R_iso, 200, 270),
                             rho_248=float(np.interp(248, E_EL, R) / np.interp(248, E_EL, R_iso)),
                             rho_q0=((cp_m * Z_XE + cn_m * N_MEAN) / (cn_m * (Z_XE + N_MEAN))) ** 2,
                             node1_keV=n1[0] if n1 else np.nan, node2_keV=n2[0] if n2 else np.nan))
        # 1/rho is the sigma_n/sigma_SI,eq conversion: LZ quotes 3.2 (q -> 0) for the vector (Higgsino-like) coupling
        key_rows[-1]['conv_q0'] = 1.0 / key_rows[-1]['rho_q0']
        key_rows[-1]['conv_window_200_270'] = 1.0 / key_rows[-1]['rho_hi']
        key_rows[-1]['conv_248'] = 1.0 / key_rows[-1]['rho_248']
        key_rows[-1]['conv_lo_5_55'] = 1.0 / key_rows[-1]['rho_lo']
key = pd.DataFrame(key_rows)
log('\n[2] key r values (elastic):')
log(key.to_string(float_format=lambda x: f'{x:.4g}'))

# minimum N_lo
mins = []
for m in MASSES:
    q = scan[scan.m_chi_GeV == m]
    i = q.N_lo.idxmin()
    # refine with a 1D minimiser on the continuous r
    R00, R11, R01 = basis('el', m)
    f = lambda r: window(E_EL, rate_r(r, R00, R11, R01), 5.4, 55) / window(E_EL, rate_r(r, R00, R11, R01), 200, 270)
    res = optimize.minimize_scalar(f, bounds=(q.r[i] - 0.05, q.r[i] + 0.05), method='bounded')
    mins.append(dict(m_chi_GeV=m, r_min=float(res.x), N_lo_min=float(res.fun), N_lo_iso=float(q[np.isclose(q.r, 1)].N_lo.iloc[0]),
                     N_lo_vec=float(q[np.isclose(q.r, -1)].N_lo.iloc[0]), N_lo_p=float(q[np.isclose(q.r, 0)].N_lo.iloc[0]),
                     N_lo_xeno=float(q[np.isclose(q.r, -0.7)].N_lo.iloc[0]),
                     r_Nlo_le_5=[float(x) for x in q.r[q.N_lo <= 5]], r_Nlo_le_3=[float(x) for x in q.r[q.N_lo <= 3]],
                     r_Nlo_le_10=[float(x) for x in q.r[q.N_lo <= 10]]))
log('\n[2b] minimum N_lo:')
for d in mins:
    rng = lambda L: (f'{min(L):.2f}..{max(L):.2f}' if L else 'none')
    log(f"  m={d['m_chi_GeV']:.0f}: r_min={d['r_min']:.4f}  N_lo,min={d['N_lo_min']:.2f}   iso {d['N_lo_iso']:.0f}  vec {d['N_lo_vec']:.0f}  "
        f"p-only {d['N_lo_p']:.0f}  xeno(-0.7) {d['N_lo_xeno']:.1f};  N_lo<=3 for r in [{rng(d['r_Nlo_le_3'])}], <=5 [{rng(d['r_Nlo_le_5'])}], <=10 [{rng(d['r_Nlo_le_10'])}]")

# -------------------------------------------------------------------------------------------------
# 3. Isotope decomposition and nodes from the M response functions directly (q-space, fast)
# -------------------------------------------------------------------------------------------------
ISO = []
for k, (mass, name, fw, ab) in enumerate(zip(xe.mass, xe.isotopes, xe.func_w, xe.abundance)):
    A = int(name[:3]); w0 = fw(1e-6)
    if w0[NC['M'], 0, 0] <= 0:
        continue
    ISO.append(dict(idx=k, A=A, N=A - Z_XE, name=name, m=float(mass), fw=fw, ab=float(ab)))
log('\n[3] active isotopes:', [(d['A'], d['ab']) for d in ISO])

def q_gev(E_keV, m_gev):
    return np.sqrt(2.0 * m_gev * np.asarray(E_keV, float) * 1e-6)

E_Q = np.linspace(1.0, 400.0, 3991)
def W_M(d, tau, taup, E):
    return np.array([d['fw'](float(qq))[NC['M'], tau, taup] for qq in q_gev(E, d['m'])])

WM = {d['A']: {tt: W_M(d, *tt, E_Q) for tt in [(0, 0), (0, 1), (1, 1)]} for d in ISO}
def F2_r(A, r):
    w = WM[A]; c0, c1 = (1 + r), (1 - r)
    return c0 ** 2 * w[(0, 0)] + 2 * c0 * c1 * w[(0, 1)] + c1 ** 2 * w[(1, 1)]

def zeros_of(E, y, lo, hi):
    out = []
    for i in range(len(E) - 1):
        if lo < E[i] < hi and y[i] * y[i + 1] < 0:
            out.append(float(E[i] - y[i] * (E[i + 1] - E[i]) / (y[i + 1] - y[i])))
    return out

# per-isotope nodes, done exactly with the signed products.  The M response is a dyad:
#   W^{tt'} = F_t F_t'  ->  W_pp = (W00 + 2W01 + W11)/4,  W_nn = (W00 - 2W01 + W11)/4,  W_pn = (W00 - W11)/4 = F_p F_n.
# For F_c = Z F_p + r N F_n the zeros are the common zeros of  h_p = Z W_pp + r N W_pn (= F_p F_c)  and
# h_n = Z W_pn + r N W_nn (= F_n F_c), both exactly signed.  Zeros of F_p (F_n) alone are sign changes of W_pn at which
# W_pp (W_nn) is the vanishing factor.
# NOTE (found while debugging): W_M^{tt'} is a single dyad only for J = 0 and J = 1/2 isotopes.  For 131Xe (J = 3/2)
# the M_0 and M_2 multipoles both contribute and W01^2/(W00 W11) = 0.44 at 100 keV, so the "node" of a coupling
# combination is a (non-zero) minimum of the quadratic form, not a zero of an amplitude.  We therefore define the
# per-isotope node as the local minimum of F2_r(E) = c0^2 W00 + 2 c0 c1 W01 + c1^2 W11 (P017's definition) and record
# its depth relative to F2_r(0); depth << 1e-3 indicates a true zero.
DYAD_ISOTOPES = (128, 129, 130, 132, 134, 136)
def combo_zeros(A, r, lo=60, hi=400, with_depth=False):
    y = F2_r(A, r); mn = local_minima(E_Q, y, lo, hi)
    if with_depth:
        return [(round(x, 1), float(np.interp(x, E_Q, y) / y[0])) for x in mn]
    return mn

def single_zeros(A, which, lo=60, hi=400):
    w = WM[A]
    Wpp = 0.25 * (w[(0, 0)] + 2 * w[(0, 1)] + w[(1, 1)]); Wnn = 0.25 * (w[(0, 0)] - 2 * w[(0, 1)] + w[(1, 1)])
    Wpn = 0.25 * (w[(0, 0)] - w[(1, 1)])
    out = []
    for z in zeros_of(E_Q, Wpn, lo, hi):
        i = int(np.searchsorted(E_Q, z))
        p_small = Wpp[i] / Wpp[0] < Wnn[i] / Wnn[0]
        if (which == 'p') == p_small:
            out.append(z)
    return out

iso_rows = []
for d in ISO:
    A = d['A']; w = WM[A]
    Wpp = 0.25 * (w[(0, 0)] + 2 * w[(0, 1)] + w[(1, 1)]); Wnn = 0.25 * (w[(0, 0)] - 2 * w[(0, 1)] + w[(1, 1)])
    Wpn = 0.25 * (w[(0, 0)] - w[(1, 1)])
    if A in DYAD_ISOTOPES:
        nodes_p, nodes_n = single_zeros(A, 'p'), single_zeros(A, 'n')
    else:   # 131Xe: minima of W_pp, W_nn (not true zeros)
        nodes_p, nodes_n = local_minima(E_Q, Wpp, 60, 400), local_minima(E_Q, Wnn, 60, 400)
    # signed F_p by tracking through its own zeros; F_n = W_pn / F_p (exact for dyad isotopes where F_p != 0)
    s = np.ones_like(E_Q)
    for z in nodes_p:
        s[E_Q > z] *= -1
    Fp_s = s * np.sqrt(np.abs(Wpp))
    sn = np.ones_like(E_Q)
    for z in nodes_n:
        sn[E_Q > z] *= -1
    Fn_s = sn * np.sqrt(np.abs(Wnn))
    # consistency of the sign assignment: F_p F_n must reproduce W_pn (exact for dyad isotopes)
    if A in DYAD_ISOTOPES:
        cons = float(np.max(np.abs(Fp_s * Fn_s - Wpn)[E_Q < 350]) / np.max(np.abs(Wpn[E_Q < 350])))
        log(f'  {A}Xe: max |F_p F_n - W_pn| / max|W_pn| (E<350 keV) = {cons:.1e}')
    dy = float(np.interp(100.0, E_Q, w[(0, 1)] ** 2 / (w[(0, 0)] * w[(1, 1)])))
    iso_rows.append(dict(A=A, N=d['N'], ab=d['ab'], Z_minus_N=Z_XE - d['N'], dyad_ratio_100keV=dy,
                         node_Fp_keV=[round(x, 1) for x in nodes_p], node_Fn_keV=[round(x, 1) for x in nodes_n],
                         node_iso_keV=combo_zeros(A, 1.0, with_depth=True), node_p_keV=combo_zeros(A, 0.0, with_depth=True),
                         node_vec_keV=combo_zeros(A, -1.0, with_depth=True), node_xeno_keV=combo_zeros(A, R_XENO, with_depth=True),
                         F2_iso_248=float(np.interp(248, E_Q, F2_r(A, 1.0)) / F2_r(A, 1.0)[0]),
                         F2_vec_248=float(np.interp(248, E_Q, F2_r(A, -1.0)) / F2_r(A, 1.0)[0]),
                         F2_p_248=float(np.interp(248, E_Q, F2_r(A, 0.0)) / F2_r(A, 1.0)[0]),
                         F2_xeno_248=float(np.interp(248, E_Q, F2_r(A, R_XENO)) / F2_r(A, 1.0)[0]),
                         # normalised ratio f_n/f_p with f(0) = 1:  (Z/N) W_pn / W_pp
                         fn_over_fp_20keV=float(np.interp(20, E_Q, Wpn / Wpp)) * Z_XE / d['N'],
                         fn_over_fp_100keV=float(np.interp(100, E_Q, Wpn / Wpp)) * Z_XE / d['N'],
                         fn_over_fp_248keV=float(np.interp(248, E_Q, Wpn / Wpp)) * Z_XE / d['N']))
    d['Fp'], d['Fn'] = Fp_s / Fp_s[0], Fn_s / Fn_s[0]
iso_df = pd.DataFrame(iso_rows)
iso_df.to_csv(os.path.join(OUT, 'P032_isotope_nodes.csv'), index=False)
pd.set_option('display.width', 250)
log(iso_df.to_string(float_format=lambda x: f'{x:.4g}'))
# isotope-resolved q->0 suppression for the xenophobic ratio (vs the mean-N estimate)
supp_iso = sum(d['ab'] * (Z_XE + R_XENO * d['N']) ** 2 for d in ISO) / sum(d['ab'] * (Z_XE + d['N']) ** 2 for d in ISO)
log(f'  q->0 xenophobic suppression: mean-N estimate {((Z_XE + R_XENO * N_MEAN) / (Z_XE + N_MEAN)) ** 2:.2e}, isotope-resolved {supp_iso:.2e}')

# node of natural-Xe M-response (abundance-weighted, unit kinematics) vs r, from responses only
def natXe_F2(r):
    return sum(d['ab'] * F2_r(d['A'], r) for d in ISO)
node_resp = []
for r in R_GRID:
    y = natXe_F2(r); mn = local_minima(E_Q, y, 60, 400)
    node_resp.append(dict(r=r, node1=([x for x in mn if x < 150] or [np.nan])[0], node2=([x for x in mn if x >= 150] or [np.nan])[0],
                          depth2=float(np.interp(([x for x in mn if x >= 150] or [np.nan])[0], E_Q, y) / y[0]) if any(x >= 150 for x in mn) else np.nan))
node_resp = pd.DataFrame(node_resp); node_resp.to_csv(os.path.join(OUT, 'P032_node_vs_r_response.csv'), index=False)

# per-isotope rate shares at 248 keV and 20 keV (1 TeV) for the key r values (from WimPyDD diff_rate per isotope)
share_rows = []
for lab, r in R_KEY.items():
    tot248 = tot20 = tot248in = 0.0; parts = {}
    for d in ISO:
        R00, R11, R01 = basis('iso248', name=d['name']); v248 = float(rate_r(r, R00, R11, R01)[0])
        R00, R11, R01 = basis('iso20', name=d['name']); v20 = float(rate_r(r, R00, R11, R01)[0])
        R00, R11, R01 = basis('iso248in', name=d['name']); v248in = float(rate_r(r, R00, R11, R01)[0])
        parts[d['A']] = (v248, v20, v248in); tot248 += v248; tot20 += v20; tot248in += v248in
    for A, (v248, v20, v248in) in parts.items():
        share_rows.append(dict(label=lab, r=r, A=A, dRdE_248=v248, share_248=v248 / tot248, dRdE_20=v20, share_20=v20 / tot20,
                               dRdE_248_delta300=v248in, share_248_delta300=v248in / tot248in))
shares = pd.DataFrame(share_rows); shares.to_csv(os.path.join(OUT, 'P032_isotope_shares.csv'), index=False)
log('\n[3b] isotope shares at 248 keV (elastic, 1 TeV):')
log(shares.pivot(index='A', columns='label', values='share_248').to_string(float_format=lambda x: f'{x:.3f}'))
log('  shares at 20 keV:')
log(shares.pivot(index='A', columns='label', values='share_20').to_string(float_format=lambda x: f'{x:.3f}'))

# -------------------------------------------------------------------------------------------------
# 4. Local significance with P016's few-bin likelihood
# -------------------------------------------------------------------------------------------------
P16 = json.load(open('output/work/P016/P016_results.json'))
EDGES = np.array(P16['fig5_top_digitised']['sigma_bin_low'] + [8.0])
b_top = np.array(P16['fig5_top_digitised']['total']); n_top = np.array(P16['fig5_top_digitised']['data_counts'])
B_H = P16['baseline_settings']['b_H']; B_M, N_M_OBS = P16['baseline_settings']['b_M'], P16['baseline_settings']['n_M']
WIN_HI, SIG_THETA = P16['baseline_settings']['window_hi_sigma'], P16['baseline_settings']['sig_theta']
sel = EDGES[:-1] < WIN_HI - 1e-9
BL, BH_ = EDGES[:-1][sel], EDGES[1:][sel]
log(f'\n[4] P016 few-bin likelihood: b_H = {B_H:.4e}, {sel.sum()} low-energy bins (sum b = {b_top[sel].sum():.1f}, n = {n_top[sel].sum():.0f}), b_M = {B_M}')

class FewBin:
    def __init__(self, b_H=B_H, kappa=1.0):
        self.b = kappa * b_top[sel]; self.n = n_top[sel]
        self.g = stats.norm.cdf(BH_) - stats.norm.cdf(BL); self.b_H = b_H
    def lnL(self, s, th, N_top, N_M):
        mu = th * self.b + s * N_top * self.g
        ll = np.sum(self.n * np.log(mu) - mu) - 0.5 * ((th - 1) / SIG_THETA) ** 2
        muM = B_M + s * N_M; ll += N_M_OBS * np.log(muM) - muM
        muH = self.b_H + s; ll += np.log(muH) - muH
        return ll
    def prof(self, s, N_top, N_M):
        r = optimize.minimize_scalar(lambda th: -self.lnL(s, th, N_top, N_M), bounds=(0.02, 10.0), method='bounded',
                                     options={'xatol': 1e-5})
        return -r.fun
    def q0(self, N_lo, N_L2=0.0, N_M=0.0):
        N_top = N_lo + N_L2
        l0 = self.prof(0.0, N_top, N_M)
        grid = np.concatenate([[0.0], np.logspace(-5, 1.3, 90)])
        vals = np.array([self.prof(s, N_top, N_M) for s in grid]); i = int(np.argmax(vals))
        if i == 0:
            return dict(Z=0.0, s_hat=0.0, q0=0.0)
        lo, hi = grid[max(i - 1, 1)], grid[min(i + 1, len(grid) - 1)]
        r = optimize.minimize_scalar(lambda ls: -self.prof(np.exp(ls), N_top, N_M), bounds=(np.log(lo), np.log(hi)),
                                     method='bounded', options={'xatol': 1e-4})
        s_hat, lmax = float(np.exp(r.x)), -r.fun
        if lmax < vals[i]:
            s_hat, lmax = grid[i], vals[i]
        q = max(0.0, 2 * (lmax - l0))
        return dict(Z=float(np.sqrt(q)), s_hat=s_hat, q0=q, lo_events=s_hat * N_lo)

FB = FewBin()
LZ_O1v = {400: lz.OSIG['O1v'][400][0], 1000: lz.OSIG['O1v'][1000][0], 4000: lz.OSIG['O1v'][4000][0]}
LZ_L1v = dict(zip([400, 1000, 4000], lz.LSIG['L1v'][10:])); LZ_L5v = dict(zip([400, 1000, 4000], lz.LSIG['L5v'][10:]))
zrows = []
for m in MASSES:
    q = scan[scan.m_chi_GeV == m]
    for r in [1.0, 0.0, -0.7, -1.0] + [float(x) for x in np.round(np.arange(-2, 2.01, 0.1), 2)]:
        row = q[np.isclose(q.r, r)].iloc[0]
        res = FB.q0(row.N_lo, row.N_L2, row.N_M)
        zrows.append(dict(m_chi_GeV=m, r=r, N_lo=row.N_lo, N_L2=row.N_L2, N_M=row.N_M, Z=res['Z'], s_hat=res['s_hat'],
                          lo_events_hat=res.get('lo_events', 0.0),
                          Z_LZ_O1v=LZ_O1v[int(m)] if np.isclose(r, -1) else (0.0 if np.isclose(r, 1) else np.nan),
                          Z_LZ_L1v=LZ_L1v[int(m)] if np.isclose(r, -1) else (0.0 if np.isclose(r, 1) else np.nan),
                          Z_LZ_L5v=LZ_L5v[int(m)] if np.isclose(r, -1) else (0.0 if np.isclose(r, 1) else np.nan)))
zdf = pd.DataFrame(zrows).drop_duplicates(subset=['m_chi_GeV', 'r']).sort_values(['m_chi_GeV', 'r'])
zdf.to_csv(os.path.join(OUT, 'P032_local_Z.csv'), index=False)
log(zdf[zdf.r.isin([1.0, 0.0, -0.7, -1.0])].to_string(float_format=lambda x: f'{x:.3g}'))

# what N_lo would reproduce LZ's isovector Z? (effective isovector N_lo implied by Table S6/S7)
def Nlo_for_Z(Ztarget, ratio_L2, ratio_M):
    f = lambda lnN: FB.q0(np.exp(lnN), np.exp(lnN) * ratio_L2, np.exp(lnN) * ratio_M)['Z'] - Ztarget
    try:
        return float(np.exp(optimize.brentq(f, np.log(1.0), np.log(5e4), xtol=1e-3)))
    except ValueError:
        return np.nan
implied = []
for m in MASSES:
    row = scan[(scan.m_chi_GeV == m) & np.isclose(scan.r, -1)].iloc[0]
    rl2, rm = row.N_L2 / row.N_lo, row.N_M / row.N_lo
    for lab, Zt in (('O1v Table S7', LZ_O1v[int(m)]), ('L1v Table S6', LZ_L1v[int(m)]), ('L5v Table S6', LZ_L5v[int(m)])):
        implied.append(dict(m_chi_GeV=m, table=lab, Z_LZ=Zt, N_lo_implied=Nlo_for_Z(Zt, rl2, rm) if Zt > 0 else np.inf,
                            N_lo_WimPyDD=row.N_lo))
    # also the isoscalar: what N_lo gives Z=0 exactly? (Z -> 0 needs s_hat = 0)
implied = pd.DataFrame(implied); implied.to_csv(os.path.join(OUT, 'P032_implied_Nlo.csv'), index=False)
log('\n[4b] N_lo implied by LZ isovector significances vs WimPyDD isovector N_lo:')
log(implied.to_string(float_format=lambda x: f'{x:.3g}'))

# b_H sensitivity for the isovector Z at 1 TeV
sens = {}
for fac in (0.5, 1.0, 2.0):
    row = scan[(scan.m_chi_GeV == 1000) & np.isclose(scan.r, -1)].iloc[0]
    sens[f'b_H x{fac}'] = FewBin(b_H=B_H * fac).q0(row.N_lo, row.N_L2, row.N_M)['Z']
log('  isovector 1 TeV Z vs b_H:', {k: round(v, 3) for k, v in sens.items()})

# -------------------------------------------------------------------------------------------------
# 5. Inelastic delta = 300 keV: spectra vs r, node, percentile of 248 keV
# -------------------------------------------------------------------------------------------------
in_rows = []
for m in MASSES:
    R00, R11, R01 = basis('in', m)
    for r in [1.0, 0.0, R_XENO, -0.7, -1.0, R_HIGGS, np.inf]:
        if np.isinf(r):
            R = np.clip(R00 - 2 * R01 + R11, 0, None)          # neutron-only
        else:
            R = np.clip(rate_r(r, R00, R11, R01), 0, None)
        f = np.interp(E_FINE, E_IN, R, left=0.0) * efficiency(E_FINE)
        cum = integrate.cumulative_trapezoid(f, E_FINE, initial=0.0)
        pct = float(np.interp(248.0, E_FINE, cum) / cum[-1])
        pct_true = float(np.interp(248.0, E_FINE, integrate.cumulative_trapezoid(np.interp(E_FINE, E_IN, R, left=0), E_FINE, initial=0)) /
                         np.trapezoid(np.interp(E_FINE, E_IN, R, left=0), E_FINE))
        mn = local_minima(E_IN, R, 150, 300)
        # ratio of the local density at 248 keV to the peak density (in-ROI, efficiency weighted)
        peak_E = float(E_FINE[np.argmax(f)])
        in_rows.append(dict(m_chi_GeV=m, r=r, percentile_248_inROI=pct, percentile_248_true=pct_true, node2_keV=mn[0] if mn else np.nan,
                            peak_keV=peak_E, f248_over_fpeak=float(np.interp(248, E_FINE, f) / f.max()),
                            R_ROI_per_cp2=float(cum[-1]), dRdE_248=float(np.interp(248, E_IN, R)),
                            E_onset_keV=float(E_IN[np.argmax(R > 1e-12 * R.max())])))
inel = pd.DataFrame(in_rows); inel.to_csv(os.path.join(OUT, 'P032_inelastic_delta300.csv'), index=False)
log('\n[5] inelastic delta = 300 keV:')
log(inel.to_string(float_format=lambda x: f'{x:.4g}'))

# -------------------------------------------------------------------------------------------------
# 6. Required proton cross-section for 1 event in 200-270 keV (2.84 t yr) vs r  [informational]
# -------------------------------------------------------------------------------------------------
HBARC2_CM2 = 0.3894e-27      # GeV^2 cm^2 (recalled, certain)
def sigma_p_for_one_event(m, R_hi_cp1):
    mu_p = m * lz.M_NUCLEON_GEV / (m + lz.M_NUCLEON_GEV)
    cp2 = 1.0 / (R_hi_cp1 * lz.LZ['exposure_tyr']) if 'exposure_tyr' in lz.LZ else 1.0 / (R_hi_cp1 * 2.84)
    return cp2 * mu_p ** 2 / math.pi * HBARC2_CM2
sig_rows = []
for m in MASSES:
    for lab, r in R_KEY.items():
        row = scan[(scan.m_chi_GeV == m) & np.isclose(scan.r, r)].iloc[0]
        sig_rows.append(dict(m_chi_GeV=m, label=lab, r=r, sigma_p_1event_cm2=sigma_p_for_one_event(m, row.R_hi),
                             N_lo_events=row.N_lo))
sig = pd.DataFrame(sig_rows); sig.to_csv(os.path.join(OUT, 'P032_sigma_p_one_event.csv'), index=False)
log('\n[6] sigma_p for one 200-270 keV event in 2.84 t yr:')
log(sig.to_string(float_format=lambda x: f'{x:.3g}'))
# q -> 0 coherent suppression (Z + r N)^2 / A^2 of other targets for the xenophobic ratio (pure arithmetic)
OTHER = {'F': (9, 10), 'Ar': (18, 22), 'Ge': (32, 41), 'I': (53, 74), 'W': (74, 110), 'Xe(<N>)': (54, N_MEAN)}
other = pd.DataFrame([dict(target=k, Z=z, N=n, supp_r_m0p7=((z - 0.7 * n) / (z + n)) ** 2,
                           supp_r_min=((z + mins[1]['r_min'] * n) / (z + n)) ** 2) for k, (z, n) in OTHER.items()])
other.to_csv(os.path.join(OUT, 'P032_other_targets_q0.csv'), index=False)
log('  q->0 suppression (Z + rN)^2/A^2 for other targets at r = -0.7 and r_min(1 TeV):')
log(other.to_string(float_format=lambda x: f'{x:.3g}'))

# -------------------------------------------------------------------------------------------------
# 7. Figures
# -------------------------------------------------------------------------------------------------
RCOL = {1.0: C['blue'], 0.0: C['orange'], -0.7: C['aqua'], -1.0: C['violet'], -2.0: C['magenta']}
RLAB = {1.0: 'isoscalar r=+1', 0.0: 'proton-only r=0', -0.7: 'xenophobic r=-0.7', -1.0: 'isovector r=-1', -2.0: 'r=-2'}

# Fig 1: elastic spectra vs r (1 TeV), normalised to the same c_p, plus ratio to isoscalar
fig, axs = plt.subplots(1, 2, figsize=(10, 4.0))
R00, R11, R01 = basis('el', 1000.0); Riso = rate_r(1.0, R00, R11, R01)
for r in [1.0, 0.0, -0.7, -1.0, -2.0]:
    R = np.clip(rate_r(r, R00, R11, R01), 1e-30, None)
    axs[0].plot(E_EL, R, color=RCOL[r], lw=2, label=RLAB[r])
    axs[1].plot(E_EL, R / Riso, color=RCOL[r], lw=2)
axs[0].set_yscale('log'); axs[0].set_ylim(1e-9 * Riso.max(), 3 * Riso.max()); axs[0].set_xlim(0, 300)
axs[0].axvline(248, color=C['muted'], lw=1, ls='--'); axs[0].text(250, 1e-8 * Riso.max(), 'event', color=C['ink2'], fontsize=8)
axs[0].set_xlabel('$E_R$ [keV]'); axs[0].set_ylabel('dR/dE at fixed $c_p$ [t$^{-1}$ yr$^{-1}$ keV$^{-1}$]')
axs[0].set_title('O1 elastic spectra, m = 1 TeV, natural Xe (WimPyDD)', color=C['ink'], fontsize=9, loc='left')
axs[0].legend(fontsize=8)
axs[1].set_yscale('log'); axs[1].set_xlim(0, 300); axs[1].axvline(248, color=C['muted'], lw=1, ls='--')
axs[1].set_xlabel('$E_R$ [keV]'); axs[1].set_ylabel('rate / isoscalar rate (same $c_p$)')
axs[1].set_title('effective-A$^2$ ratio: isospin violation is energy dependent', color=C['ink'], fontsize=9, loc='left')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P032_fig1_spectra_vs_r.png'), dpi=150); plt.close(fig)

# Fig 2: N_lo vs r and Z vs r
fig, axs = plt.subplots(1, 2, figsize=(10, 4.0))
MCOL = {400.0: C['blue'], 1000.0: C['orange'], 4000.0: C['aqua']}
for m in MASSES:
    q = scan[scan.m_chi_GeV == m]
    axs[0].plot(q.r, q.N_lo, color=MCOL[m], lw=2, label=f'{m:.0f} GeV')
    zq = zdf[zdf.m_chi_GeV == m].sort_values('r')
    axs[1].plot(zq.r, zq.Z, color=MCOL[m], lw=2, label=f'{m:.0f} GeV (this work)')
for y, lab in ((3, 'N$_{lo}$ = 3'), (5, '5')):
    axs[0].axhline(y, color=C['muted'], lw=1, ls=':'); axs[0].text(1.6, y * 1.1, lab, color=C['ink2'], fontsize=8)
axs[0].set_yscale('log'); axs[0].set_xlabel('r = f$_n$/f$_p$'); axs[0].set_ylabel('N$_{lo}$: 5.4–55 keV events per 200–270 keV event')
axs[0].set_title('low-energy companions vs isospin ratio', color=C['ink'], fontsize=9, loc='left'); axs[0].legend(fontsize=8)
axs[0].axvline(R_XENO, color=C['muted'], lw=1, ls='--'); axs[0].text(R_XENO + 0.03, 2e3, 'r = −Z/⟨N⟩', color=C['ink2'], fontsize=8)
for m, mk in zip(MASSES, ('o', 's', 'D')):
    axs[1].scatter([-1.0], [LZ_O1v[int(m)]], color=MCOL[m], marker=mk, s=40, zorder=5, edgecolor=C['surface'])
    axs[1].scatter([1.0], [0.0], color=MCOL[m], marker=mk, s=40, zorder=5, edgecolor=C['surface'])
axs[1].scatter([], [], color=C['ink2'], marker='o', s=30, label='LZ Table S7 (O1$^v$ δ=0, O1$^s$)')
axs[1].set_xlabel('r = f$_n$/f$_p$'); axs[1].set_ylabel('local significance Z [σ]'); axs[1].set_ylim(0, 3.6)
axs[1].set_title('P016 few-bin likelihood, b$_H$ = 5.7e-4', color=C['ink'], fontsize=9, loc='left'); axs[1].legend(fontsize=8, loc='upper right')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P032_fig2_Nlo_Z_vs_r.png'), dpi=150); plt.close(fig)

# Fig 3: node positions vs r (natural Xe rate minimum, response-only minimum, per-isotope zeros) and F_p vs F_n
fig, axs = plt.subplots(1, 2, figsize=(10, 4.0))
q = scan[scan.m_chi_GeV == 1000]
axs[0].plot(node_resp.r, node_resp.node2, color=C['orange'], lw=2.2, label='natural Xe M-response minimum')
axs[0].plot(node_resp.r, node_resp.node1, color=C['orange'], lw=2.2, alpha=0.6)
qm = q[np.isclose(np.round(q.r * 4), q.r * 4) & np.isclose(q.r % 0.25, 0)]
axs[0].scatter(qm.r, qm.node2_keV, color=C['blue'], s=22, zorder=5, edgecolor=C['surface'], label='natural Xe dR/dE minimum (1 TeV)')
axs[0].scatter(qm.r, qm.node1_keV, color=C['blue'], s=22, zorder=5, edgecolor=C['surface'])
ISOCOL = {128: C['magenta'], 129: C['aqua'], 130: C['yellow'], 131: C['green'], 132: C['violet'], 134: C['red'], 136: C['ink2']}
for d in ISO:
    A = d['A']; nodes = []
    for r in np.arange(-2, 2.01, 0.05):
        z = combo_zeros(A, r, 150, 400)
        nodes.append(z[0] if z else np.nan)
    axs[0].plot(np.arange(-2, 2.01, 0.05), nodes, color=ISOCOL[A], lw=1, alpha=0.9, label=f'$^{{{A}}}$Xe')
axs[0].axhline(248, color=C['muted'], lw=1, ls=':'); axs[0].text(-1.95, 250, 'event 248 keV', color=C['ink2'], fontsize=8)
axs[0].set_xlabel('r = f$_n$/f$_p$'); axs[0].set_ylabel('node / minimum position [keV]'); axs[0].set_ylim(60, 340)
axs[0].set_title('M-response nodes vs isospin ratio', color=C['ink'], fontsize=9, loc='left'); axs[0].legend(fontsize=6.5, loc='upper right', ncol=3)
for d in ISO:
    if d['A'] in (129, 132, 136):
        axs[1].plot(E_Q, d['Fp'], color=ISOCOL[d['A']], lw=2, label=f"$^{{{d['A']}}}$Xe F$_p$")
        axs[1].plot(E_Q, d['Fn'], color=ISOCOL[d['A']], lw=2, ls='--', label=f"$^{{{d['A']}}}$Xe F$_n$")
axs[1].axhline(0, color=C['muted'], lw=1); axs[1].axvline(248, color=C['muted'], lw=1, ls=':')
axs[1].set_xlim(0, 400); axs[1].set_ylim(-0.12, 0.3); axs[1].set_xlabel('$E_R$ [keV]'); axs[1].set_ylabel('F(q)/F(0) (signed, shell model)')
axs[1].set_title('proton vs neutron M form factors', color=C['ink'], fontsize=9, loc='left'); axs[1].legend(fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P032_fig3_nodes.png'), dpi=150); plt.close(fig)

# Fig 4: inelastic delta = 300 keV spectra for r = 1, 0, -1 (1 TeV), efficiency-weighted, normalised to unit area
fig, ax = plt.subplots(figsize=(6.2, 4.0))
R00, R11, R01 = basis('in', 1000.0)
for r in [1.0, 0.0, -1.0]:
    R = np.clip(rate_r(r, R00, R11, R01), 0, None); f = np.interp(E_FINE, E_IN, R, left=0) * efficiency(E_FINE)
    ax.plot(E_FINE, f / np.trapezoid(f, E_FINE), color=RCOL[r], lw=2, label=RLAB[r])
ax.axvline(248, color=C['muted'], lw=1, ls='--'); ax.text(250, ax.get_ylim()[1] * 0.9, 'event', color=C['ink2'], fontsize=8)
ax.set_xlim(50, 300); ax.set_xlabel('$E_R$ [keV]'); ax.set_ylabel('normalised accepted spectrum [keV$^{-1}$]')
ax.set_title('inelastic O1, δ = 300 keV, m = 1 TeV: isospin moves the node', color=C['ink'], fontsize=9, loc='left'); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P032_fig4_inelastic_delta300.png'), dpi=150); plt.close(fig)

# -------------------------------------------------------------------------------------------------
# 8. Summary JSON
# -------------------------------------------------------------------------------------------------
summary = dict(
    halo='Baxter-2021 SHM annual average (lz.wd_halo)', efficiency=dict(eff0=EFF0, sig_lo=SIG_LO, sig_hi=SIG_HI),
    bilinearity_max_rel_dev=max(chk), N_mean=N_MEAN, r_xeno=R_XENO,
    key_points=key.to_dict(orient='records'), N_lo_minimum=mins,
    isotope_nodes=iso_df.to_dict(orient='records'),
    natXe_response_nodes={lab: dict(node1=float(node_resp[np.isclose(node_resp.r, r)].node1.iloc[0]),
                                     node2=float(node_resp[np.isclose(node_resp.r, r)].node2.iloc[0])) for lab, r in R_KEY.items()},
    local_Z=zdf[zdf.r.isin([1.0, 0.0, -0.7, -1.0])].to_dict(orient='records'),
    implied_Nlo=implied.to_dict(orient='records'), Z_isovector_1TeV_bH_sensitivity=sens,
    inelastic_delta300=inel.to_dict(orient='records'), sigma_p_one_event=sig.to_dict(orient='records'),
    P016_inputs=dict(b_H=B_H, b_M=B_M, n_M=N_M_OBS, sig_theta=SIG_THETA, window_hi_sigma=WIN_HI),
    LZ_tables=dict(O1v_delta0=LZ_O1v, L1v=LZ_L1v, L5v=LZ_L5v, O1s_delta0=0.0, L1s=0.0, L5s=0.0))
def _clean(o):
    if isinstance(o, dict): return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if (isinstance(o, float) and math.isnan(o)) else float(o)
    if isinstance(o, (np.integer,)): return int(o)
    return o
json.dump(_clean(summary), open(os.path.join(OUT, 'P032_summary.json'), 'w'), indent=1)
log('\nwritten:', os.listdir(OUT))
