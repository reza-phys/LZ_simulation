"""
P057_operator_degeneracy.py -- Operator interference and spectral degeneracy among the NREFT operators
compatible with a lone 248 keV xenon recoil: what a second (and a tenth) event would distinguish.

Parts
  A. WimPyDD spectra (Sun-frame Baxter-2021 halo; 1 TeV and 400 GeV) for the lone-event-compatible set
     O6, O10, O9, O14, O15, O5^v, O13^s, L10 = 4[(q^2/m_N^2)O4 - O6], O9 x q^2 (P044's L16 candidate),
     inelastic O1^s (delta = 300/350/366/380 keV; P050's 12-day annual-average cache and our Sun-frame
     recomputation) and inelastic O4^s (300/350 keV), plus the excluded references O1^s and O4^s.
  B. Observed-energy pdfs: efficiency (LZ edge E50 = 269.9 keV, or an extended 1000 phd edge E50 = 423 keV
     from P038) x Gaussian resolution sigma_E = 11 sqrt(E/248) keV (P009/P021).  Shape metrics: KL divergence,
     Hellinger distance, peak position, low/high ratios; hierarchical clustering (dendrogram).
  C. Events to separate each pair at 3 sigma: Gaussian LLR estimate (both directions) and toy MC.
  D. Two-operator Hamiltonians O1 + O6 (no interference: M vs Sigma'') and O4 + O6 (interference through the
     c4 c6 cross term of Sigma''): WimPyDD cross-term extraction, limits rho -> 0, infinity, and the maximum
     O1 / O4 admixture compatible with N_lo <= 2.2, 3, 5 low-energy companions.
  E. Second-event lookup table: P(E_2 | model) and likelihood ratios for E_2 = 100 ... 400 keV.

Run from the simulation root (WimPyDD needs cwd = root):
    .venv/bin/python output/code/P057_operator_degeneracy.py            # compute (cached per spectrum) + analyse
    .venv/bin/python output/code/P057_operator_degeneracy.py compute    # compute only (re-run until complete)
Per-spectrum caches live in output/work/P057/cache/ so interrupted runs resume.
"""
from __future__ import annotations
import sys, os, json, time, math, itertools
sys.path.insert(0, 'output/code')
import numpy as np
import pandas as pd
from scipy import special, stats
from scipy.cluster import hierarchy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import lzcommon as lz

OUT = 'output/work/P057'; FIG = OUT + '/figures'; CACHE = OUT + '/cache'
for d in (OUT, FIG, CACHE):
    os.makedirs(d, exist_ok=True)
LOG = open(OUT + '/P057_run.log', 'a')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.write(s + '\n'); LOG.flush()

MV2 = lz.M_V_GEV ** 2                 # m_v^2 (GeV^2); LZ/Anand unit coupling 1/m_v^2 = WimPyDD c^0 = 2/m_v^2 (P003)
MN = lz.M_NUCLEON_GEV
C_UNIT = 2.0 / MV2                    # WimPyDD c^0 (or c^1) for an LZ unit coupling
E_TRUE = np.arange(1.25, 600.0, 2.5)  # keV, 240 points (WimPyDD ~0.06-0.13 s per point)
E_OBS = np.arange(0.5, 700.0, 1.0)    # observed-energy grid, 1 keV bins
MASSES = [1000.0, 400.0]
EXPO_LZ = 2.84                        # t yr
SIGMA_E0 = 11.0                       # keV at 248 keV (P009/P021): sigma_E = 11 sqrt(E/248)
EDGES = {'LZ': dict(E50=269.9, sig=8.0, label='LZ edge (600 phd, E50 = 270 keV)'),
         'EXT': dict(E50=423.2, sig=12.5, label='extended edge (1000 phd, E50 = 423 keV; P038)')}
FLOOR_W = 1e-2                        # flat-pdf admixture regularising the KL (P050's choice); 1e-3 as variant

# ------------------------------------------------------------------------------------------------
# Part A: Hamiltonians and spectra.  Coefficient functions are closures without default arguments
# (WimPyDD treats default arguments as shared, named Hamiltonian parameters -- P031 pitfall).
# ------------------------------------------------------------------------------------------------
WD = None; HALO = None
def wd_init():
    global WD, HALO
    if WD is None:
        WD = lz.wd(); HALO = lz.wd_halo()          # Sun-frame Baxter-2021 SHM, explicit v_min grid (P002 fix)

def const_fn(c0, c1=0.0):
    c0, c1 = float(c0), float(c1)
    def f():
        return [c0, c1]
    return f
def q2_fn(c0, c1=0.0, scale=MN ** 2):
    """coefficient x q^2/scale (q in GeV)"""
    c0, c1, scale = float(c0), float(c1), float(scale)
    def f(q):
        fac = q ** 2 / scale
        return [c0 * fac, c1 * fac]
    return f

def build(name, wc):
    wd_init()
    return WD.eft_hamiltonian('P057_' + name, wc)

def single(op, iso):
    return {op: const_fn(C_UNIT, 0.0) if iso == 's' else const_fn(0.0, C_UNIT)}

# model catalogue: name -> (wc dict builder, delta_keV, note)
MODELS = {}
for op in [1, 4, 6, 9, 10, 14, 15]:
    MODELS[f'O{op}s'] = (lambda op=op: single(op, 's'), 0.0)
for op in [5, 6, 9, 10, 14, 15]:
    MODELS[f'O{op}v'] = (lambda op=op: single(op, 'v'), 0.0)
MODELS['O13s'] = (lambda: single(13, 's'), 0.0)
# L10 = 4[(q^2/m_N^2) O4 - O6] (Anand reduction as implemented by P003/P012; pure q^4 Sigma' response)
MODELS['L10'] = (lambda: {4: q2_fn(4 * C_UNIT), 6: const_fn(-4 * C_UNIT)}, 0.0)
# O9 x q^2/m_v^2 (P044's companion-free candidate for L16; q^6 Sigma')
MODELS['O9q2'] = (lambda: {9: q2_fn(C_UNIT, 0.0, MV2)}, 0.0)
for dlt in [300.0, 350.0, 366.0, 380.0]:
    MODELS[f'inel{dlt:.0f}_sun'] = (lambda: single(1, 's'), dlt)
for dlt in [300.0, 350.0]:
    MODELS[f'inelO4_{dlt:.0f}_sun'] = (lambda: single(4, 's'), dlt)
MODELS_400 = ['O1s', 'O4s', 'O6s', 'O9s', 'O10s', 'O14s', 'O15s', 'O5v', 'L10', 'O9q2']
INEL_400 = {'inel300_sun': 300.0, 'inel340_sun': 340.0}   # kinematic ceiling 341 keV at 400 GeV (P002)
RHO16 = [1e-3, 1e-2, 1.0]                  # c1/c6 for the O1 + O6 scan (1e-2 with both signs)
RHO46 = [0.01, 0.069, 0.3, 10.0]           # c4/c6 for the O4 + O6 scan (both signs); 0.069 puts the Sigma'' node at 248 keV

T_START = time.time(); BUDGET_S = 530.0      # foreground runs are limited to 600 s: stop computing after 530 s and resume on the next call
class Budget(Exception):
    pass
def cached_rate(key, wc_builder, m, delta):
    fn = f'{CACHE}/{key}.npy'
    if os.path.exists(fn):
        return np.load(fn)
    if time.time() - T_START > BUDGET_S:
        raise Budget(key)
    wd_init()
    t = time.time()
    r = lz.wd_rate(build(key, wc_builder()), m, E_TRUE, halo=HALO, delta_kev=delta)
    r = np.clip(np.asarray(r, float), 0.0, None)
    np.save(fn, r); say(f'  computed {key}: {time.time()-t:.0f} s; R(200-270)={np.trapezoid(r[(E_TRUE>200)&(E_TRUE<270)], E_TRUE[(E_TRUE>200)&(E_TRUE<270)]):.3e} /t/yr')
    return r

def compute_all():
    SP = {}
    for name, (wcb, dlt) in MODELS.items():
        SP[('1000', name)] = cached_rate(f'm1000_{name}', wcb, 1000.0, dlt)
    for name in MODELS_400:
        SP[('400', name)] = cached_rate(f'm400_{name}', MODELS[name][0], 400.0, 0.0)
    for name, dlt in INEL_400.items():
        SP[('400', name)] = cached_rate(f'm400_{name}', lambda: single(1, 's'), 400.0, dlt)
    # interference scans at 1 TeV: c6 = C_UNIT fixed, c1 (or c4) = sign * rho * c6
    for rho in RHO16:
        for sgn in ([+1, -1] if rho == 1e-2 else [+1]):
            key = f'mix16_rho{rho:g}_{"p" if sgn > 0 else "m"}'
            SP[('mix', key)] = cached_rate(key, lambda rho=rho, sgn=sgn: {1: const_fn(sgn * rho * C_UNIT), 6: const_fn(C_UNIT)}, 1000.0, 0.0)
    for rho in RHO46:
        for sgn in [+1, -1]:
            key = f'mix46_rho{rho:g}_{"p" if sgn > 0 else "m"}'
            SP[('mix', key)] = cached_rate(key, lambda rho=rho, sgn=sgn: {4: const_fn(sgn * rho * C_UNIT), 6: const_fn(C_UNIT)}, 1000.0, 0.0)
    # P050's 12-day annual-average inelastic O1s spectra (1 TeV) and its L10, re-gridded
    p50 = np.load('output/work/P050/P050_spectra_cache.npz')
    for i, dlt in enumerate(p50['deltas']):
        if float(dlt) in (300.0, 350.0, 366.0, 380.0):
            SP[('1000', f'inel{dlt:.0f}')] = np.clip(np.interp(E_TRUE, p50['E_in'], p50['inel'][i]), 0, None)
    SP[('1000', 'L10_P050annual')] = np.clip(np.interp(E_TRUE, p50['E_l10'], p50['l10']), 0, None)
    return SP

# ------------------------------------------------------------------------------------------------
# Part B: observed-energy pdfs and shape metrics
# ------------------------------------------------------------------------------------------------
def efficiency(E, edge='LZ', eff0=0.96, sig_lo=3.4):
    e = EDGES[edge]
    E = np.asarray(E, float)
    return eff0 * 0.5 * (1 + special.erf((E - 5.4) / (math.sqrt(2) * sig_lo))) * 0.5 * (1 - special.erf((E - e['E50']) / (math.sqrt(2) * e['sig'])))

def smear_matrix():
    sig = SIGMA_E0 * np.sqrt(E_TRUE / 248.0)
    K = stats.norm.pdf(E_OBS[:, None], loc=E_TRUE[None, :], scale=sig[None, :])   # [obs, true], per keV_obs
    return K * (E_OBS[1] - E_OBS[0])
KSM = smear_matrix()
DE_T = E_TRUE[1] - E_TRUE[0]

def accepted_true(r, edge):
    return r * efficiency(E_TRUE, edge)

def pdf_obs(r, edge):
    """observed-energy pdf (per 1 keV bin) of accepted events; returns (pdf, accepted rate /t/yr)"""
    acc = accepted_true(r, edge) * DE_T
    p = KSM @ acc
    tot = p.sum()
    if tot <= 0:
        return np.zeros_like(p), 0.0
    return p / tot, tot

def window_rate_true(r, e1, e2, edge='LZ'):
    m = (E_TRUE >= e1) & (E_TRUE <= e2)
    return float(np.trapezoid(r[m] * efficiency(E_TRUE[m], edge), E_TRUE[m]))

def shape_metrics(r):
    if not np.any(r > 0):
        return dict(N_lo=np.nan, note='empty spectrum (kinematically forbidden in the Sun frame)')
    R_lo = window_rate_true(r, 5.4, 55); R_hi = window_rate_true(r, 200, 270); R_mid = window_rate_true(r, 100, 200)
    R_ext = window_rate_true(r, 270, 423, 'EXT'); R_roi = window_rate_true(r, 1.25, 330)
    pL, aL = pdf_obs(r, 'LZ'); pX, aX = pdf_obs(r, 'EXT')
    peak_true = float(E_TRUE[np.argmax(r * (E_TRUE > 20))]) if np.any(r[E_TRUE > 20] > 0) else np.nan
    q_obs = np.cumsum(pX); q16, q50, q84 = [float(E_OBS[np.searchsorted(q_obs, x)]) for x in (0.16, 0.5, 0.84)]
    return dict(N_lo=R_lo / R_hi if R_hi > 0 else np.inf, R_100_200_over_hi=R_mid / R_hi if R_hi > 0 else np.inf,
                R_270_423_over_hi=R_ext / R_hi if R_hi > 0 else np.inf, f_above_LZedge_ext=R_ext / (R_ext + window_rate_true(r, 1.25, 270, 'EXT')),
                peak_true_keV=peak_true, obs_median_ext=q50, obs_16_ext=q16, obs_84_ext=q84,
                acc_LZ_per_tyr_unit=aL, acc_EXT_per_tyr_unit=aX, R_hi_unit=R_hi, R_lo_unit=R_lo)

def floored(p, w=FLOOR_W):
    return (1 - w) * p + w / p.size

def kl(p, q, w=FLOOR_W):
    pf, qf = floored(p, w), floored(q, w)
    lam = np.log(pf / qf)
    D = float(np.sum(pf * lam)); V = float(np.sum(pf * lam ** 2) - D ** 2)
    return D, V

def hellinger(p, q):
    return float(math.sqrt(max(0.0, 1 - np.sum(np.sqrt(p * q)))))

# ------------------------------------------------------------------------------------------------
# Part C: events to separate two shapes at 3 sigma
# ------------------------------------------------------------------------------------------------
N_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 130, 160, 200]
RNG = np.random.default_rng(57)
def n3_gauss(p, q, w=FLOOR_W):
    """Gaussian LLR estimate: smallest N with median(LLR | p) >= 3 sigma quantile of LLR | q  (reject q if p true)
    and the reverse; plus the symmetric 9/(D_pq + D_qp) estimate."""
    Dpq, Vp = kl(p, q, w); Dqp, Vq = kl(q, p, w)
    n_rej_q = 9 * Vq / (Dpq + Dqp) ** 2      # p true, reject q
    n_rej_p = 9 * Vp / (Dpq + Dqp) ** 2      # q true, reject p
    return dict(KL_pq=Dpq, KL_qp=Dqp, N_rej_q_if_p=n_rej_q, N_rej_p_if_q=n_rej_p, N_simple=9.0 / (Dpq + Dqp))

def n3_toys(p, q, w=FLOOR_W, ntoy=20000):
    """toy MC: smallest N in N_LIST for which the median of sum ln(pf/qf) under p exceeds the 99.865 % quantile under q."""
    pf, qf = floored(p, w), floored(q, w)
    lam = np.log(pf / qf)
    cp, cq = np.cumsum(p), np.cumsum(q)
    out = {}
    for direction, (ctrue, calt, sign) in {'rej_q_if_p': (cp, cq, +1), 'rej_p_if_q': (cq, cp, -1)}.items():
        found = None
        for N in N_LIST:
            it = np.searchsorted(ctrue, RNG.random((ntoy, N))); ia = np.searchsorted(calt, RNG.random((ntoy, N)))
            it = np.clip(it, 0, lam.size - 1); ia = np.clip(ia, 0, lam.size - 1)
            st = sign * lam[it].sum(axis=1); sa = sign * lam[ia].sum(axis=1)
            thr = np.quantile(sa, 1 - 0.00135)
            if np.median(st) >= thr:
                found = N; break
        out[direction] = found if found is not None else np.inf
    return out

# ------------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------------
def main(stage):
    say(f'\n===== P057 run {time.strftime("%Y-%m-%d %H:%M:%S")} stage={stage} =====')
    try:
        SP = compute_all()
    except Budget as b:
        say(f'time budget reached before {b}; {len(os.listdir(CACHE))} spectra cached -- re-run to continue'); return
    if stage == 'compute':
        say('compute stage complete'); return
    RES = {'settings': dict(E_true_grid=[float(E_TRUE[0]), float(E_TRUE[-1]), float(DE_T)], E_obs_bin_keV=1.0, sigma_E='11 sqrt(E/248) keV',
                            edges={k: {kk: vv for kk, vv in v.items()} for k, v in EDGES.items()}, floor_w=FLOOR_W,
                            coupling='WimPyDD c^0 = 2/m_v^2 (LZ unit coupling, P003); isovector c^1 = 2/m_v^2',
                            halo='Sun-frame Baxter-2021 SHM (lz.wd_halo(), v_E = 250.6 km/s); inelXXX (no suffix) = P050 12-day annual average')}

    # ---------- B. shape metrics ----------
    rows = []
    for (m, name), r in SP.items():
        if m == 'mix':
            continue
        d = dict(mass=m, model=name); d.update(shape_metrics(r)); rows.append(d)
    MET = pd.DataFrame(rows); MET.to_csv(OUT + '/P057_shape_metrics.csv', index=False)
    say('\nShape metrics (1 TeV):'); say(MET[MET.mass == '1000'].set_index('model')[['N_lo', 'R_100_200_over_hi', 'R_270_423_over_hi', 'f_above_LZedge_ext', 'peak_true_keV', 'obs_16_ext', 'obs_median_ext', 'obs_84_ext']].round(3).to_string())
    say('\nShape metrics (400 GeV):'); say(MET[MET.mass == '400'].set_index('model')[['N_lo', 'R_100_200_over_hi', 'R_270_423_over_hi', 'peak_true_keV', 'obs_median_ext']].round(3).to_string())

    # ---------- B'. exact response-level degeneracies: spectral ratios ----------
    ratios = {}
    def ratio_stats(a, b, lo=20, hi=400):
        m = (E_TRUE > lo) & (E_TRUE < hi) & (SP[('1000', b)] > 0) & (SP[('1000', a)] > 0)
        if m.sum() < 5:
            return dict(loglog_slope=np.nan, rms_dex_about_powerlaw=np.nan, ratio_at_50=np.nan, ratio_at_248=np.nan)
        rr = SP[('1000', a)][m] / SP[('1000', b)][m]
        slope = np.polyfit(np.log(E_TRUE[m]), np.log(rr), 1)[0]     # log-log slope of the ratio
        resid = np.std(np.log(rr) - np.polyval(np.polyfit(np.log(E_TRUE[m]), np.log(rr), 1), np.log(E_TRUE[m])))
        return dict(loglog_slope=float(slope), rms_dex_about_powerlaw=float(resid / math.log(10)), ratio_at_50=float(np.interp(50, E_TRUE[m], rr)), ratio_at_248=float(np.interp(248, E_TRUE[m], rr)))
    for a, b in [('O6s', 'O10s'), ('O6v', 'O10v'), ('O6s', 'O6v'), ('O10s', 'O10v'), ('O9s', 'O14s'), ('O9s', 'O9v'), ('O14s', 'O14v'), ('O15s', 'O15v'),
                 ('L10', 'O9q2'), ('L10', 'O6s'), ('O9s', 'O4s'), ('O9q2', 'O9s'), ('O13s', 'O10s'), ('O5v', 'O6s'), ('L10', 'L10_P050annual'), ('inel366_sun', 'inel366'), ('inel380_sun', 'inel380'), ('inel300_sun', 'inel300')]:
        ratios[f'{a}/{b}'] = ratio_stats(a, b)
    RES['spectral_ratios'] = ratios
    say('\nSpectral ratios a/b (20-400 keV): log-log slope, rms scatter about power law [dex]'); say(pd.DataFrame(ratios).T.round(4).to_string())

    # ---------- B''. pairwise KL / Hellinger / N_3sigma, both edges ----------
    SET_1TEV = ['L10', 'O9q2', 'O6s', 'O6v', 'O10s', 'O10v', 'O9s', 'O9v', 'O14s', 'O14v', 'O15s', 'O15v', 'O5v', 'O13s', 'O4s', 'O1s',
                'inel300', 'inel350', 'inel366', 'inel380', 'inelO4_300_sun', 'inelO4_350_sun']
    PDF = {edge: {n: pdf_obs(SP[('1000', n)], edge)[0] for n in SET_1TEV} for edge in EDGES}
    for n in ['inel300_sun', 'inel350_sun', 'inel366_sun', 'inel380_sun', 'L10_P050annual']:
        for edge in EDGES:
            PDF[edge][n] = pdf_obs(SP[('1000', n)], edge)[0]
    pair_rows = []
    t0 = time.time()
    for edge in EDGES:
        for a, b in itertools.combinations(SET_1TEV, 2):
            p, q = PDF[edge][a], PDF[edge][b]
            g = n3_gauss(p, q); g3 = n3_gauss(p, q, 1e-3)
            row = dict(edge=edge, A=a, B=b, hellinger=hellinger(p, q), **g, KL_pq_w1e3=g3['KL_pq'], KL_qp_w1e3=g3['KL_qp'],
                       N_rej_B_if_A_w1e3=g3['N_rej_q_if_p'], N_rej_A_if_B_w1e3=g3['N_rej_p_if_q'])
            if min(g['N_rej_q_if_p'], g['N_rej_p_if_q']) <= 150:
                toys = n3_toys(p, q); row['N_toy_rej_B_if_A'] = toys['rej_q_if_p']; row['N_toy_rej_A_if_B'] = toys['rej_p_if_q']
            else:
                row['N_toy_rej_B_if_A'] = np.nan; row['N_toy_rej_A_if_B'] = np.nan
            pair_rows.append(row)
    PAIRS = pd.DataFrame(pair_rows)
    PAIRS['N3_max_gauss'] = PAIRS[['N_rej_q_if_p', 'N_rej_p_if_q']].max(axis=1)
    PAIRS['N3_min_gauss'] = PAIRS[['N_rej_q_if_p', 'N_rej_p_if_q']].min(axis=1)
    PAIRS.to_csv(OUT + '/P057_pairwise.csv', index=False)
    say(f'\npairwise table: {len(PAIRS)} rows in {time.time()-t0:.0f} s')
    for edge in EDGES:
        sub = PAIRS[PAIRS.edge == edge].sort_values('N3_max_gauss', ascending=False)
        say(f'\nMost degenerate pairs, {edge} edge (N_3sigma Gaussian, worse direction; toys where <= 150):')
        say(sub.head(14)[['A', 'B', 'hellinger', 'KL_pq', 'KL_qp', 'N_rej_q_if_p', 'N_rej_p_if_q', 'N_toy_rej_B_if_A', 'N_toy_rej_A_if_B']].round(3).to_string(index=False))
    # halo-labelling systematic on inelastic shapes
    halo_rows = []
    for dlt in [300, 350, 366, 380]:
        for edge in EDGES:
            p, q = PDF[edge][f'inel{dlt}_sun'], PDF[edge][f'inel{dlt}']
            g = n3_gauss(p, q); halo_rows.append(dict(delta=dlt, edge=edge, hellinger=hellinger(p, q), KL_sun_vs_annual=g['KL_pq'], N3_gauss=max(g['N_rej_q_if_p'], g['N_rej_p_if_q'])))
    for edge in EDGES:
        g = n3_gauss(PDF[edge]['L10'], PDF[edge]['L10_P050annual']); halo_rows.append(dict(delta='L10', edge=edge, hellinger=hellinger(PDF[edge]['L10'], PDF[edge]['L10_P050annual']), KL_sun_vs_annual=g['KL_pq'], N3_gauss=max(g['N_rej_q_if_p'], g['N_rej_p_if_q'])))
    HALO = pd.DataFrame(halo_rows); HALO.to_csv(OUT + '/P057_halo_labelling.csv', index=False)
    say('\nSun-frame vs 12-day annual halo (same model):'); say(HALO.round(4).to_string(index=False))

    # key pairs summary
    KEY = [('O6s', 'O10s'), ('O9s', 'O14s'), ('L10', 'O9q2'), ('L10', 'O6s'), ('O6s', 'O9s'), ('O9s', 'O10s'), ('O10s', 'O14s'), ('O6s', 'O15v'), ('O6s', 'O5v'),
           ('L10', 'inel366'), ('L10', 'inel350'), ('L10', 'inel380'), ('O6s', 'inel366'), ('inel350', 'inel366'), ('inel366', 'inel380'), ('inel300', 'inelO4_300_sun'), ('inel350', 'inelO4_350_sun'),
           ('O6s', 'O6v'), ('O10s', 'O10v'), ('O9s', 'O9v'), ('L10', 'O4s'), ('O6s', 'O1s')]
    key_rows = []
    for a, b in KEY:
        for edge in EDGES:
            r = PAIRS[(PAIRS.edge == edge) & (((PAIRS.A == a) & (PAIRS.B == b)) | ((PAIRS.A == b) & (PAIRS.B == a)))].iloc[0]
            key_rows.append(dict(pair=f'{a} vs {b}', edge=edge, H=r.hellinger, KL_ab=r.KL_pq if r.A == a else r.KL_qp, KL_ba=r.KL_qp if r.A == a else r.KL_pq,
                                 N3_gauss_rej_b_if_a=r.N_rej_q_if_p if r.A == a else r.N_rej_p_if_q, N3_gauss_rej_a_if_b=r.N_rej_p_if_q if r.A == a else r.N_rej_q_if_p,
                                 N3_toy_rej_b_if_a=r.N_toy_rej_B_if_A if r.A == a else r.N_toy_rej_A_if_B, N3_toy_rej_a_if_b=r.N_toy_rej_A_if_B if r.A == a else r.N_toy_rej_B_if_A, N3_simple=r.N_simple))
    KEYT = pd.DataFrame(key_rows); KEYT.to_csv(OUT + '/P057_key_pairs.csv', index=False)
    say('\nKey pairs:'); say(KEYT.round(3).to_string(index=False))

    # ---------- clustering (Hellinger, average linkage) ----------
    CL = {}
    for edge in EDGES:
        names = SET_1TEV
        D = np.array([[hellinger(PDF[edge][a], PDF[edge][b]) for b in names] for a in names])
        Z = hierarchy.linkage(D[np.triu_indices(len(names), 1)], method='average')
        clusters = hierarchy.fcluster(Z, t=0.15, criterion='distance')   # H < 0.15: roughly N_3sigma >~ 100 (see details)
        CL[edge] = {int(c): [n for n, cc in zip(names, clusters) if cc == c] for c in sorted(set(clusters))}
        fig, ax = plt.subplots(figsize=(8, 4.2))
        hierarchy.dendrogram(Z, labels=names, ax=ax, color_threshold=0.15, above_threshold_color='#555555', leaf_rotation=60, leaf_font_size=8)
        ax.set_ylabel('Hellinger distance (average linkage)'); ax.set_title(f'Spectral clustering, 1 TeV, {EDGES[edge]["label"]}', fontsize=10)
        ax.axhline(0.15, color='#999999', lw=0.8, ls='--')
        for s in ('top', 'right'): ax.spines[s].set_visible(False)
        fig.tight_layout(); fig.savefig(f'{FIG}/P057_dendrogram_{edge}.png', dpi=160); plt.close(fig)
    RES['clusters_H_lt_0p15'] = CL
    say('\nClusters (H < 0.15):'); say(json.dumps(CL, indent=1))

    # ---------- 400 GeV vs 1 TeV ----------
    m400 = []
    for n in MODELS_400:
        for edge in EDGES:
            p, q = pdf_obs(SP[('400', n)], edge)[0], PDF[edge][n]
            g = n3_gauss(p, q); m400.append(dict(model=n, edge=edge, H_400_vs_1000=hellinger(p, q), N3_gauss=max(g['N_rej_q_if_p'], g['N_rej_p_if_q'])))
    for n, ref in [('inel300_sun', 'inel300_sun'), ('inel340_sun', 'inel350_sun')]:
        if not np.any(SP[('400', n)] > 0):
            say(f'  {n} at 400 GeV: empty spectrum in the Sun frame (v_max = 794.6 km/s); skipped'); continue
        for edge in EDGES:
            p, q = pdf_obs(SP[('400', n)], edge)[0], PDF[edge][ref]
            g = n3_gauss(p, q); m400.append(dict(model=f'{n}(400) vs {ref}(1000)', edge=edge, H_400_vs_1000=hellinger(p, q), N3_gauss=max(g['N_rej_q_if_p'], g['N_rej_p_if_q'])))
    M400 = pd.DataFrame(m400); M400.to_csv(OUT + '/P057_mass_dependence.csv', index=False)
    say('\n400 GeV vs 1 TeV same model:'); say(M400.round(3).to_string(index=False))
    # pairwise at 400 GeV among the compatible elastic set
    p400 = []
    for a, b in itertools.combinations(['L10', 'O9q2', 'O6s', 'O10s', 'O9s', 'O14s', 'O15s', 'O5v'], 2):
        for edge in EDGES:
            p, q = pdf_obs(SP[('400', a)], edge)[0], pdf_obs(SP[('400', b)], edge)[0]
            g = n3_gauss(p, q); p400.append(dict(A=a, B=b, edge=edge, hellinger=hellinger(p, q), N3_gauss_max=max(g['N_rej_q_if_p'], g['N_rej_p_if_q'])))
    P400 = pd.DataFrame(p400); P400.to_csv(OUT + '/P057_pairwise_400GeV.csv', index=False)

    # ---------- D. interference ----------
    INT = {}
    R1, R4, R6 = SP[('1000', 'O1s')], SP[('1000', 'O4s')], SP[('1000', 'O6s')]
    # O1 + O6: verify incoherent addition
    dev16 = {}
    for rho in RHO16:
        for sgn in ([+1, -1] if rho == 1e-2 else [+1]):
            key = f'mix16_rho{rho:g}_{"p" if sgn > 0 else "m"}'
            pred = R6 + rho ** 2 * R1
            m = pred > 0
            dev16[key] = float(np.max(np.abs(SP[('mix', key)][m] / pred[m] - 1)))
    INT['O1+O6_max_rel_dev_from_incoherent_sum'] = dev16
    say('\nO1+O6: max |WimPyDD/(R6 + rho^2 R1) - 1|:', {k: f'{v:.2e}' for k, v in dev16.items()})
    # O4 + O6: cross term X46(E) = [R(+rho) - R(-rho)] / (4 rho)   (rate = R6 + rho^2 R4 + 2 sgn rho X46)
    X = {}
    for rho in RHO46:
        rp, rm = SP[('mix', f'mix46_rho{rho:g}_p')], SP[('mix', f'mix46_rho{rho:g}_m')]
        X[rho] = (rp - rm) / (4 * rho)
        even = (rp + rm) / 2 - (R6 + rho ** 2 * R4)
        INT.setdefault('O4+O6_even_part_check_max_rel', {})[str(rho)] = float(np.max(np.abs(even[R6 > 0] / (R6 + rho ** 2 * R4)[R6 > 0])))
    X46 = np.mean([X[r] for r in [0.069, 0.3]], axis=0)
    INT['O4+O6_cross_term_rho_independence_mean_abs_rel'] = {str(r): float(np.mean(np.abs(X[r][(R6 > 0) & (E_TRUE > 10) & (E_TRUE < 400)] / X46[(R6 > 0) & (E_TRUE > 10) & (E_TRUE < 400)] - 1))) for r in RHO46}
    # the cross term should be proportional to q^2 Sigma'' ~ R6/(q^2) ~ R10 shape: compare with R10s
    R10 = SP[('1000', 'O10s')]
    m = (E_TRUE > 20) & (E_TRUE < 400)
    sl = np.polyfit(np.log(E_TRUE[m]), np.log(np.abs(X46[m]) / R10[m]), 1)[0]
    sl6 = np.polyfit(np.log(E_TRUE[m]), np.log(np.abs(X46[m]) / R6[m]), 1)[0]
    INT['cross_term_shape'] = dict(loglog_slope_X46_over_O10=float(sl), loglog_slope_X46_over_O6=float(sl6), sign_of_X46_for_c4c6_positive=float(np.sign(np.mean(X46[m]))))
    say('O4+O6 cross term: X46/R10 log-log slope', round(sl, 3), '; X46/R6 slope', round(sl6, 3), '; sign', np.sign(np.mean(X46[m])))
    # node of destructive constant-c4 admixture: (c4 + c6 q^2/m_N^2)^2 Sigma'' -> node at E = rho m_N^2 / (2 m_T)
    m_T = lz.m_nucleus_gev(lz.A_XE_MEAN)
    INT['node_formula_keV_per_unit_rho'] = float(MN ** 2 / (2 * m_T) * 1e6)
    # Sigma'' part of the destructive rho = 0.069 spectrum: total - c4^2 Sigma' part.  The Sigma' part of O4 is not separately
    # available, so locate the node as the minimum of [R(-rho) - R(+rho)]-symmetrised quantity: R6 + rho^2 R4 - 2 rho X46 has its
    # Sigma'' contribution (sqrt(R6) - rho sqrt(R4_Sigma''))^2 -> we report the energy where the destructive spectrum is most suppressed
    # relative to the constructive one, which is the node of the Sigma'' amplitude.
    rm069, rp069 = SP[('mix', 'mix46_rho0.069_m')], SP[('mix', 'mix46_rho0.069_p')]
    msk = (E_TRUE > 50) & (E_TRUE < 500)
    INT['destructive_rho0.069_min_of_Rminus_over_Rplus_keV'] = float(E_TRUE[msk][np.argmin(rm069[msk] / rp069[msk])])
    INT['destructive_rho0.069_Rminus_over_Rplus_min'] = float(np.min(rm069[msk] / rp069[msk]))
    # limits: rho -> 0 and rho -> infinity
    lim = {}
    for edge in EDGES:
        p0 = pdf_obs(SP[('mix', 'mix46_rho0.01_p')], edge)[0]; pinf = pdf_obs(SP[('mix', 'mix46_rho10_p')], edge)[0]
        lim[edge] = dict(H_rho0p01_vs_O6=hellinger(p0, PDF[edge]['O6s']), H_rho10_vs_O4=hellinger(pinf, PDF[edge]['O4s']),
                         H_O1mix_rho1e3_vs_O6=hellinger(pdf_obs(SP[('mix', 'mix16_rho0.001_p')], edge)[0], PDF[edge]['O6s']), H_O1mix_rho1_vs_O1=hellinger(pdf_obs(SP[('mix', 'mix16_rho1_p')], edge)[0], PDF[edge]['O1s']))
    INT['limits'] = lim; say('limits:', json.dumps(lim, indent=1))
    # admixture scans (analytic in rho using the pure spectra and the extracted cross term)
    def nlo_of(r):
        return window_rate_true(r, 5.4, 55) / window_rate_true(r, 200, 270)
    rho_grid = np.logspace(-6.5, 1.5, 2000)
    scan = []
    for rho in rho_grid:
        r16 = R6 + rho ** 2 * R1
        hi6 = window_rate_true(R6, 200, 270); hi1 = rho ** 2 * window_rate_true(R1, 200, 270)
        row = dict(rho=rho, N_lo_O1O6=nlo_of(r16), f_hi_O1=hi1 / (hi1 + hi6))
        for sgn, tag in [(+1, 'p'), (-1, 'm')]:
            r46 = np.clip(R6 + rho ** 2 * R4 + 2 * sgn * rho * X46, 0, None)
            row[f'N_lo_O4O6_{tag}'] = nlo_of(r46)
            row[f'f_hi_O4_incoh_{tag}'] = rho ** 2 * window_rate_true(R4, 200, 270) / window_rate_true(r46, 200, 270)
            row[f'R_hi_ratio_{tag}'] = window_rate_true(r46, 200, 270) / hi6
        scan.append(row)
    SCAN = pd.DataFrame(scan); SCAN.to_csv(OUT + '/P057_admixture_scan.csv', index=False)
    lims = {}
    for tol in [1.6, 2.2, 3.0, 5.0]:
        d = {}
        ok = SCAN.N_lo_O1O6 <= tol
        rho_max = SCAN.rho[ok].max(); d['O1O6_rho_max'] = float(rho_max); d['O1O6_f_hi_O1_max'] = float(SCAN.f_hi_O1[ok].max())
        # analytic check: f_hi(O1) <= (tol - N6)/(N1 - N6)
        for tag in ['p', 'm']:
            ok4 = SCAN[f'N_lo_O4O6_{tag}'] <= tol
            d[f'O4O6_{tag}_rho_max'] = float(SCAN.rho[ok4].max()) if ok4.any() else 0.0
            d[f'O4O6_{tag}_f_hi_O4_incoh_max'] = float(SCAN[f'f_hi_O4_incoh_{tag}'][ok4].max()) if ok4.any() else 0.0
        lims[str(tol)] = d
    N1, N6, N4 = nlo_of(R1), nlo_of(R6), nlo_of(R4)
    lims['analytic_f_hi_O1_max'] = {str(t): float((t - N6) / (N1 - N6)) for t in [1.6, 2.2, 3.0, 5.0]}
    lims['N_lo_pure'] = dict(O1s=N1, O4s=N4, O6s=N6)
    INT['admixture_limits'] = lims
    say('admixture limits:', json.dumps(lims, indent=1))
    # hidden SI cross-section: O6 normalised to 1.0 accepted LZ event in 2.84 t yr; c1 = rho c6
    acc6 = pdf_obs(R6, 'LZ')[1]                     # accepted rate at c6 = C_UNIT, per t yr
    c6_sq = 1.0 / (acc6 * EXPO_LZ) * C_UNIT ** 2    # c6^2 giving 1.0 event
    hbarc2 = 0.389e-27; mn = 0.931; mu = 1000.0 * mn / (1000.0 + mn)
    def sigma_n_from_c0(c0):                        # WimPyDD c^0 = 2 c_N (isoscalar)
        cN = c0 / 2.0
        return cN ** 2 * mu ** 2 / math.pi * hbarc2
    hid = {}
    for tol in [1.6, 2.2, 3.0, 5.0]:
        rho = lims[str(tol)]['O1O6_rho_max']
        hid[str(tol)] = dict(rho_max=rho, c1_GeV2=math.sqrt(c6_sq) * rho, sigma_n_cm2=sigma_n_from_c0(math.sqrt(c6_sq) * rho),
                             O1_events_in_ROI_2p84=float(rho ** 2 * c6_sq / C_UNIT ** 2 * pdf_obs(R1, 'LZ')[1] * EXPO_LZ))
    hid['c6_for_one_LZ_event_GeV-2'] = math.sqrt(c6_sq); hid['c6_times_mv2'] = math.sqrt(c6_sq) * MV2 / 2   # Anand-normalised
    INT['hidden_SI'] = hid; say('hidden SI:', json.dumps(hid, indent=1))
    # distinguishability of the maximal O1 admixture from pure O6 (2nd event)
    for tol in ['3.0', '5.0']:
        rho = lims[tol]['O1O6_rho_max']; rmix = R6 + rho ** 2 * R1
        for edge in EDGES:
            g = n3_gauss(pdf_obs(rmix, edge)[0], PDF[edge]['O6s'])
            INT.setdefault('mix_vs_pureO6_N3', {})[f'tol{tol}_{edge}'] = dict(H=hellinger(pdf_obs(rmix, edge)[0], PDF[edge]['O6s']), N3_gauss_max=max(g['N_rej_q_if_p'], g['N_rej_p_if_q']))
    RES['interference'] = INT

    # ---------- E. second-event lookup ----------
    E2 = [100, 150, 200, 248, 300, 350, 400]
    LOOK = {}
    look_rows = []
    SET2 = ['L10', 'O9q2', 'O6s', 'O10s', 'O9s', 'O14s', 'O15s', 'O15v', 'O5v', 'O13s', 'inel300', 'inel350', 'inel366', 'inel380', 'inelO4_300_sun', 'inelO4_350_sun', 'O4s', 'O1s']
    for edge in EDGES:
        for n in SET2:
            p = PDF[edge][n]
            row = dict(edge=edge, model=n)
            for e in E2:
                row[f'p_{e}'] = float(p[np.argmin(np.abs(E_OBS - e))])         # per keV
            row['P_lt55'] = float(p[E_OBS < 55].sum()); row['P_55_200'] = float(p[(E_OBS >= 55) & (E_OBS < 200)].sum())
            row['P_200_270'] = float(p[(E_OBS >= 200) & (E_OBS < 270)].sum()); row['P_gt270'] = float(p[E_OBS >= 270].sum())
            look_rows.append(row)
    LOOKT = pd.DataFrame(look_rows)
    for edge in EDGES:
        ref = LOOKT[(LOOKT.edge == edge) & (LOOKT.model == 'L10')].iloc[0]
        for e in E2:
            LOOKT.loc[LOOKT.edge == edge, f'LR_{e}_vs_L10'] = LOOKT.loc[LOOKT.edge == edge, f'p_{e}'] / ref[f'p_{e}']
    LOOKT.to_csv(OUT + '/P057_second_event_lookup.csv', index=False)
    say('\nSecond-event lookup, EXT edge, p(E2) per keV:'); say(LOOKT[LOOKT.edge == 'EXT'].set_index('model')[[f'p_{e}' for e in E2] + ['P_lt55', 'P_55_200', 'P_200_270', 'P_gt270']].to_string(float_format=lambda x: f'{x:.2e}'))
    say('\nLR vs L10, EXT edge:'); say(LOOKT[LOOKT.edge == 'EXT'].set_index('model')[[f'LR_{e}_vs_L10' for e in E2]].to_string(float_format=lambda x: f'{x:.3g}'))
    say('\nLR vs L10, LZ edge:'); say(LOOKT[LOOKT.edge == 'LZ'].set_index('model')[[f'LR_{e}_vs_L10' for e in E2[:5]]].to_string(float_format=lambda x: f'{x:.3g}'))
    # posterior over the compatible set (uniform prior) after a second event at E2
    COMPAT = ['L10', 'O9q2', 'O6s', 'O10s', 'O9s', 'O14s', 'O15s', 'O5v', 'inel300', 'inel350', 'inel366', 'inel380']
    post = {}
    for edge in EDGES:
        sub = LOOKT[(LOOKT.edge == edge) & LOOKT.model.isin(COMPAT)].set_index('model')
        for e in E2:
            w = sub[f'p_{e}'].values; w = w / w.sum() if w.sum() > 0 else w
            post[f'{edge}_{e}'] = {m: float(x) for m, x in zip(sub.index, w)}
    RES['second_event_posterior_uniform_prior'] = post
    # information a second event carries: expected KL between each pair for the compatible set = one-event information (already in PAIRS)

    # ---------- figures ----------
    OI = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']   # Okabe-Ito, fixed order
    LAB = {'L10': 'L10 (q⁴Σ′)', 'O9q2': 'O9×q² (q⁶Σ′)', 'O6s': 'O6 (q⁴Σ″)', 'O10s': 'O10 (q²Σ″)', 'O9s': 'O9 (q²Σ′)', 'O14s': 'O14 (q²v²Σ′)', 'O15s': 'O15ˢ', 'O5v': 'O5ᵛ',
           'inel300': 'inel. δ=300', 'inel350': 'δ=350', 'inel366': 'δ=366', 'inel380': 'δ=380', 'O4s': 'O4', 'O1s': 'O1'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, edge in zip(axes, EDGES):
        for i, n in enumerate(['L10', 'O9q2', 'O6s', 'O10s', 'O9s', 'O14s', 'O5v']):
            ax.plot(E_OBS, PDF[edge][n], color=OI[i % 8], lw=1.6, ls='-' if i < 8 else '--', label=LAB[n])
        for j, n in enumerate(['inel300', 'inel366', 'inel380']):
            ax.plot(E_OBS, PDF[edge][n], color=OI[(7 + j) % 8] if j else '#0072B2', lw=1.2, ls='--', label=LAB[n])
        ax.axvline(248, color='#888888', lw=0.8, ls=':'); ax.set_xlim(0, 520); ax.set_yscale('log'); ax.set_ylim(1e-5, 3e-2)
        ax.set_xlabel('observed recoil energy E_obs [keV]'); ax.set_title(EDGES[edge]['label'], fontsize=10)
        for s in ('top', 'right'): ax.spines[s].set_visible(False)
    axes[0].set_ylabel('pdf of accepted events [per keV]'); axes[1].legend(fontsize=7.5, ncol=2, frameon=False)
    fig.suptitle('Normalised observed-energy spectra, 1 TeV, σ_E = 11√(E/248) keV', fontsize=11); fig.tight_layout()
    fig.savefig(FIG + '/P057_spectra_overlay.png', dpi=160); plt.close(fig)

    # N_3sigma matrix (EXT edge, worse direction, Gaussian), sequential single hue
    for edge in EDGES:
        names = ['L10', 'O9q2', 'O6s', 'O10s', 'O9s', 'O14s', 'O15s', 'O15v', 'O5v', 'O13s', 'inel300', 'inel350', 'inel366', 'inel380', 'inelO4_300_sun', 'inelO4_350_sun']
        M = np.full((len(names), len(names)), np.nan)
        sub = PAIRS[PAIRS.edge == edge]
        for _, r in sub.iterrows():
            if r.A in names and r.B in names:
                i, j = names.index(r.A), names.index(r.B); M[i, j] = M[j, i] = r.N3_max_gauss
        fig, ax = plt.subplots(figsize=(8.2, 7))
        im = ax.imshow(np.log10(np.clip(M, 1, 1e4)), cmap='Blues', vmin=0, vmax=3)
        ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names))); ax.set_xticklabels(names, rotation=70, fontsize=8); ax.set_yticklabels(names, fontsize=8)
        for i in range(len(names)):
            for j in range(len(names)):
                if i != j and np.isfinite(M[i, j]):
                    v = M[i, j]; ax.text(j, i, f'{v:.0f}' if v < 100 else f'{v:.0e}'.replace('e+0', 'e'), ha='center', va='center', fontsize=6.2, color='white' if v > 60 else '#222222')
        cb = fig.colorbar(im, ax=ax, fraction=0.04); cb.set_label('log10 N_3σ (events; worse direction, Gaussian LLR)')
        ax.set_title(f'Events to separate two shapes at 3σ, 1 TeV, {EDGES[edge]["label"]}', fontsize=10)
        fig.tight_layout(); fig.savefig(f'{FIG}/P057_N3sigma_matrix_{edge}.png', dpi=160); plt.close(fig)

    # interference figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(E_TRUE, R6 / R6.max(), color=OI[0], lw=1.8, label='pure O6')
    for k, (rho, c) in enumerate([(0.01, OI[1]), (0.069, OI[2]), (0.3, OI[3])]):
        ax.plot(E_TRUE, SP[('mix', f'mix46_rho{rho:g}_p')] / R6.max(), color=c, lw=1.3, label=f'O4+O6, c4/c6 = +{rho}')
        ax.plot(E_TRUE, SP[('mix', f'mix46_rho{rho:g}_m')] / R6.max(), color=c, lw=1.3, ls='--', label=f'c4/c6 = −{rho}')
    ax.plot(E_TRUE, SP[('1000', 'L10')] / SP[('1000', 'L10')].max(), color=OI[5], lw=1.3, ls=':', label='L10: c4 = −(q²/m_N²) c6')
    ax.set_yscale('log'); ax.set_ylim(1e-3, 30); ax.set_xlim(0, 450); ax.set_xlabel('true recoil energy [keV]'); ax.set_ylabel('dR/dE (arb., O6 peak = 1)')
    ax.set_title('O4 + O6 interference (Σ″ cross term)', fontsize=10); ax.legend(fontsize=7, frameon=False)
    ax = axes[1]
    ax.plot(SCAN.rho, SCAN.N_lo_O1O6, color=OI[0], lw=1.8, label='O1 + O6 (incoherent)')
    ax.plot(SCAN.rho, SCAN.N_lo_O4O6_p, color=OI[1], lw=1.6, label='O4 + O6, same sign')
    ax.plot(SCAN.rho, SCAN.N_lo_O4O6_m, color=OI[1], lw=1.6, ls='--', label='O4 + O6, opposite sign')
    for tol, c in [(3, '#888888'), (5, '#bbbbbb')]: ax.axhline(tol, color=c, lw=0.8, ls=':')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('coupling ratio ρ = c1/c6 or c4/c6'); ax.set_ylabel('N_lo = R(5.4–55) / R(200–270 keV)')
    ax.set_title('Low-energy companions of an admixture', fontsize=10); ax.legend(fontsize=8, frameon=False)
    for a in axes:
        for s in ('top', 'right'): a.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG + '/P057_interference.png', dpi=160); plt.close(fig)

    # ---------- save ----------
    RES['n_pairs'] = int(len(PAIRS)); RES['toy_settings'] = dict(ntoy=20000, seed=57, N_list=N_LIST, criterion='median LLR under true model >= 99.865% quantile under alternative')
    with open(OUT + '/P057_results.json', 'w') as f:
        json.dump(RES, f, indent=1, default=float)
    np.savez(OUT + '/P057_spectra.npz', E_true=E_TRUE, E_obs=E_OBS, **{f'{m}__{n}': r for (m, n), r in SP.items()})
    say('\nsaved results; done.')

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'all')
