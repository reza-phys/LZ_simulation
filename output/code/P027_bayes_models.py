"""
P027 -- Bayesian model comparison across the LZ model space given one 248 keV event.

For every model spectrum m (elastic NREFT O1,O3-O15 s/v x 13 masses from the P008 WimPyDD cache; the
L10 transverse-spin combination (q^2/m_N^2)O4 - O6 recomputed with WimPyDD at 13 masses; inelastic O1/O4
s/v x {400,1000,4000} GeV on LZ's 8-point delta grid and on a 10 keV delta scan up to the kinematic
ceiling, recomputed with WimPyDD) we build the P016 few-bin likelihood of the data
   (i)   high-energy 200-270 keV NR-band bin: n_H = 1, b_H = 5.695e-4 (P016 anchor to LZ's 3.4 sigma)
   (ii)  low-energy NR-band bins (digitised Fig. 5 S1c<250 phd panel, P016): counts n_i, background theta*b_i
   (iii) 125-200 keV bin: n = 0, b_M = 0.02,
as a function of the signal strength, and marginalise the strength under several priors:
   s-parametrisation (s = expected 200-270 keV signal events; the assignment's choice, P001/P016 compatible):
       (a) log-uniform on [1e-3, 30]  (b) uniform on [0, 10]  (c) Jeffreys-like  1/sqrt(s) on [0, 10]
   mu-parametrisation (mu = expected signal events in the whole 5.4-270 keV ROI; same three priors).
Optionally the single event's position inside the 200-270 keV window enters through a shape factor
   rho_m = 70 keV * f_m^obs(248 keV)   (observed-energy pdf, 23 keV resolution, normalised in the window;
   flat background in the window).
Model priors: uniform over distinguishable spectra; uniform over 4 classes; physics-weighted (inelastic =
elastic); uniform over raw models; heavy-only.  Outputs: per-model Bayes factors, marginal B_DM, posterior
over models (entropy, top-10, class masses), continuous-delta class Bayes factors, posterior predictive
for the P020 exposure (6.76 t yr).

Run from the simulation root:  .venv/bin/python output/code/P027_bayes_models.py
Outputs: output/work/P027/*.csv, *.json, P027_spectra.npz; figures in output/work/P027/figures/
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from scipy import stats, special
from scipy.special import logsumexp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, 'output/code')
from common import lzcommon as lz

T0 = time.time()
OUT = 'output/work/P027'; FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)
R = {}
LOG = open(os.path.join(OUT, 'run_log.txt'), 'w')


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s); LOG.write(s + '\n'); LOG.flush()


# ------------------------------------------------------------------------------------------------
# 0. Data and likelihood ingredients (P016)
# ------------------------------------------------------------------------------------------------
P016 = json.load(open('output/work/P016/P016_results.json'))
dig = P016['fig5_top_digitised']
EDGES = np.array(dig['sigma_bin_low'] + [8.0])
b_top = np.array(dig['total']); n_top = np.array(dig['data_counts'])
WIN_HI = 2.0
sel = EDGES[:-1] < WIN_HI - 1e-9
B_LO, N_LO = b_top[sel], n_top[sel]
G = stats.norm.cdf(EDGES[1:][sel]) - stats.norm.cdf(EDGES[:-1][sel])     # Gaussian NR-band signal shape
B_H = P016['baseline_settings']['b_H']          # 5.695e-4
B_M, N_M_OBS = 0.02, 0
SIG_THETA = 0.3
TH = np.linspace(max(0.02, 1 - 4 * SIG_THETA), 1 + 4 * SIG_THETA, 41)
LNW_TH = stats.norm.logpdf(TH, 1, SIG_THETA); LNW_TH -= logsumexp(LNW_TH)
R['likelihood_inputs'] = dict(b_H=B_H, n_H=1, b_M=B_M, n_M=N_M_OBS, sig_theta=SIG_THETA, n_lowE_bins=int(sel.sum()),
                              b_lowE_sum=float(B_LO.sum()), n_lowE_sum=float(N_LO.sum()), window_hi_sigma=WIN_HI)
log(f'[0] likelihood inputs: b_H={B_H:.4e}, low-E bins {sel.sum()} (b={B_LO.sum():.1f}, n={N_LO.sum():.0f}), b_M={B_M}')

# efficiency (P003/P016 model)
SIG_LO, SIG_HI, EFF0 = 3.4, 8.0, 0.96


def efficiency(E):
    return EFF0 * 0.5 * (1 + special.erf((E - 5.4) / (np.sqrt(2) * SIG_LO))) * 0.5 * (1 - special.erf((E - 269.9) / (np.sqrt(2) * SIG_HI)))


E_FINE = np.arange(1.0, 400.001, 0.5)
EFF_FINE = efficiency(E_FINE)
E_EVENT, SIG_E, WIN = 248.0, 23.0, (200.0, 270.0)


def interp_spectrum(E, dR):
    """log-interpolate a WimPyDD spectrum onto E_FINE (zero outside support)."""
    dR = np.asarray(dR, float); ok = dR > 0
    if ok.sum() < 2:
        return np.zeros_like(E_FINE)
    f = np.interp(E_FINE, E[ok], np.log(dR[ok]), left=-np.inf, right=-np.inf)
    out = np.exp(f)
    # keep zeros inside the support where WimPyDD returned 0 (kinematic gaps)
    inside = (E_FINE >= E[ok].min()) & (E_FINE <= E[ok].max())
    return np.where(inside, out, 0.0)


def windows(dR_fine):
    """efficiency-weighted true-energy rates in the four ROI windows."""
    w = dR_fine * EFF_FINE
    def integ(a, b):
        m = (E_FINE >= a) & (E_FINE <= b)
        return float(np.trapezoid(w[m], E_FINE[m]))
    return integ(5.4, 55), integ(55, 125), integ(125, 200), integ(200, 270)


# resolution smearing (true -> observed) for the in-window shape factor
E_OBS = np.arange(150.0, 350.001, 1.0)
_K = stats.norm.pdf(E_OBS[:, None], loc=E_FINE[None, :], scale=SIG_E * np.sqrt(E_FINE[None, :] / 248.0)) * 0.5


def shape_factor(dR_fine, sigma_scale=1.0):
    """rho = 70 keV * f_obs(248) with f_obs the observed-energy pdf normalised over 200-270 keV (flat bkg)."""
    if sigma_scale == 1.0:
        obs = (_K @ dR_fine) * efficiency(E_OBS)
    else:
        K = stats.norm.pdf(E_OBS[:, None], loc=E_FINE[None, :], scale=sigma_scale * SIG_E * np.sqrt(E_FINE[None, :] / 248.0)) * 0.5
        obs = (K @ dR_fine) * efficiency(E_OBS)
    m = (E_OBS >= WIN[0]) & (E_OBS <= WIN[1])
    norm = np.trapezoid(obs[m], E_OBS[m])
    if norm <= 0:
        return 0.0
    return float(np.interp(E_EVENT, E_OBS, obs) / norm * (WIN[1] - WIN[0]))


# ------------------------------------------------------------------------------------------------
# 1. Spectra: P008 cache (elastic), WimPyDD recomputation (L10, inelastic grid + delta scan)
# ------------------------------------------------------------------------------------------------
p8 = np.load('output/work/P008/spectra_cache.npz', allow_pickle=True)
P8_NAMES, P8_RATES, E_WD8 = list(p8['names']), p8['rates'], p8['E_WD']
p8m = pd.read_csv('output/work/P008/models.csv').set_index('name')

E_WD = np.concatenate([np.geomspace(2.0, 150.0, 28), np.arange(155.0, 400.01, 5.0)])   # 78 points
INEL_MASSES = [400, 1000, 4000]
MN = lz.M_NUCLEON_GEV
MV2 = lz.M_V_GEV ** 2


def ceiling_kev(m):
    """kinematic ceiling mu v_max^2 / 2 for the annual-average halo (no scattering above it)."""
    return lz.mu_red(m, lz.m_nucleus_gev(lz.A_XE_MEAN)) * lz.vmax_kms() ** 2 / 2 / lz.C_KMS ** 2 * 1e6


CEIL = {m: ceiling_kev(m) for m in INEL_MASSES}
DELTA_GRID = {m: list(range(0, int(np.ceil(CEIL[m] / 10.0)) * 10 + 1, 10)) for m in INEL_MASSES}
R['kinematic_ceiling_keV_annual_halo'] = {str(m): float(v) for m, v in CEIL.items()}
R['delta_max_248keV_annual_halo'] = {str(m): float(lz.delta_max_kev(248.0, m)) for m in INEL_MASSES}
R['v_max_annual_kms'] = float(lz.vmax_kms())
log('[1] kinematic ceilings (annual halo):', {m: round(v, 1) for m, v in CEIL.items()}, 'v_max', round(lz.vmax_kms(), 1))

CACHE = os.path.join(OUT, 'P027_spectra.npz')
todo = []                                                # (name, cls, op, tau, mass, delta)
for m in lz.LSIG_MASSES:
    todo.append((f'L10s_m{m}', 'elastic', 'L10', 's', m, 0.0))
for op in (1, 4):
    for tau in ('s', 'v'):
        for m in INEL_MASSES:
            deltas = DELTA_GRID[m] if not (op == 4 and tau == 'v') else lz.OSIG_DELTAS   # O4v: LZ grid only (shape = O4s)
            for d in deltas:
                todo.append((f'O{op}{tau}_m{m}_d{d}', 'inelastic', f'O{op}', tau, m, float(d)))
if os.path.exists(CACHE):
    z = np.load(CACHE, allow_pickle=True)
    if list(z['names']) == [t[0] for t in todo] and np.allclose(z['E_WD'], E_WD):
        NEW_RATES = z['rates']; log('    loaded cached WimPyDD spectra', CACHE)
    else:
        NEW_RATES = None
else:
    NEW_RATES = None
if NEW_RATES is None:
    halo = lz.wd_halo()
    WD = lz.wd()
    hams = {'L10': WD.eft_hamiltonian('L10_q2O4_minus_O6', {(4, 'q2'): lambda q, A=1.0: [A * q ** 2 / MN ** 2, 0.0],
                                                          6: lambda A=1.0: [-A, 0.0]})}
    for op in (1, 4):
        hams[f'O{op}s'] = lz.wd_hamiltonian(f'O{op}s', {op: (2.0 / MV2, 0.0)})
        hams[f'O{op}v'] = lz.wd_hamiltonian(f'O{op}v', {op: (0.0, 2.0 / MV2)})
    # incremental, resumable cache (a full pass takes ~12 min; O4 spectra cost ~3 s each)
    PARTIAL = os.path.join(OUT, 'P027_spectra_partial.npz')
    done = {}
    if os.path.exists(PARTIAL):
        zp = np.load(PARTIAL, allow_pickle=True)
        if np.allclose(zp['E_WD'], E_WD):
            done = dict(zip(list(zp['names']), zp['rates']))
        log(f'    resuming: {len(done)} spectra already in partial cache')
    NEW_RATES = np.zeros((len(todo), len(E_WD)))
    n_new = 0
    for i, (name, cls, op, tau, m, d) in enumerate(todo):
        if name in done:
            NEW_RATES[i] = done[name]; continue
        if op == 'L10':
            NEW_RATES[i] = lz.wd_rate(hams['L10'], float(m), E_WD, halo=halo, A=1.0 / MV2)
        else:
            NEW_RATES[i] = lz.wd_rate(hams[f'{op}{tau}'], float(m), E_WD, halo=halo, delta_kev=d)
        done[name] = NEW_RATES[i]; n_new += 1
        if n_new % 20 == 0:
            np.savez(PARTIAL, names=np.array(list(done.keys())), rates=np.array(list(done.values())), E_WD=E_WD)
            log(f'    WimPyDD {i + 1}/{len(todo)} {name}  t={time.time() - T0:.0f}s (partial cache saved)')
        if time.time() - T0 > float(os.environ.get('P027_BUDGET_S', 1e9)):     # stay within a foreground pass; rerun to resume
            np.savez(PARTIAL, names=np.array(list(done.keys())), rates=np.array(list(done.values())), E_WD=E_WD)
            log(f'    time budget reached after {n_new} new spectra ({len(done)}/{len(todo)} cached); rerun to resume'); LOG.close()
            sys.exit(0)
    NEW_RATES = np.maximum(NEW_RATES, 0.0)
    np.savez(CACHE, names=np.array([t[0] for t in todo]), rates=NEW_RATES, E_WD=E_WD)
    log(f'    computed {n_new} new WimPyDD spectra ({len(todo)} total) in {time.time() - T0:.0f}s')
R['wimpydd_spectra_computed'] = len(todo)

# assemble the model table
rows = []
for i, name in enumerate(P8_NAMES):
    r = p8m.loc[name]
    if r.cls != 'elastic':
        continue
    rows.append(dict(name=name, cls='elastic', op=f'O{int(r.op)}', tau=r.tau, mass=int(r.mass), delta=0.0,
                     distinct=bool(r.distinct_rep_1pct), on_LZ_grid=True, spec=interp_spectrum(E_WD8, P8_RATES[i])))
for i, (name, cls, op, tau, m, d) in enumerate(todo):
    spec = interp_spectrum(E_WD, NEW_RATES[i])
    if cls == 'elastic':   # L10: LZ's rule -- masses >= 400 GeV degenerate, keep 1000 GeV
        distinct = m <= 200 or m == 1000
        rows.append(dict(name=name, cls=cls, op=op, tau=tau, mass=m, delta=0.0, distinct=distinct, on_LZ_grid=True, spec=spec))
    else:
        on_grid = d in lz.OSIG_DELTAS
        physical = spec.sum() > 0
        distinct = bool(p8m.loc[name].distinct_rep_1pct) if (on_grid and name in p8m.index) else False
        rows.append(dict(name=name, cls=cls, op=op, tau=tau, mass=m, delta=d, distinct=distinct and physical,
                         on_LZ_grid=on_grid and physical, spec=spec))
M = pd.DataFrame(rows)
# derived spectral quantities
q = np.array([windows(s) for s in M.spec])
tot = q.sum(axis=1)
with np.errstate(invalid='ignore', divide='ignore'):
    M['f_lo'], M['f_L2'], M['f_M'], M['f_hi'] = [np.where(tot > 0, q[:, j] / tot, 0.0) for j in range(4)]
    M['N_lo'] = np.where(q[:, 3] > 0, q[:, 0] / q[:, 3], np.inf)
    M['N_L2'] = np.where(q[:, 3] > 0, q[:, 1] / q[:, 3], np.inf)
    M['N_M'] = np.where(q[:, 3] > 0, q[:, 2] / q[:, 3], np.inf)
M['rho'] = [shape_factor(s) for s in M.spec]
M['rho_sig32'] = [shape_factor(s, sigma_scale=np.sqrt(2)) for s in M.spec]      # stat (+) sys resolution
M['physical'] = tot > 0
M['grid_model'] = M.on_LZ_grid & M.physical
# class labels
def class4(r):
    if r.cls == 'elastic':
        return 'elastic-isoscalar' if r.tau == 's' else 'elastic-isovector'
    return 'inelastic-O1' if r.op == 'O1' else 'inelastic-O4'
M['class4'] = M.apply(class4, axis=1)
Q2SPIN = {'O6', 'O9', 'O10', 'O14', 'O15', 'L10'}
M['q2spin'] = M.apply(lambda r: r.cls == 'elastic' and (r.op in Q2SPIN or (r.op == 'O5' and r.tau == 'v')), axis=1)
M['SIlike'] = M.apply(lambda r: r.cls == 'elastic' and r.op == 'O1', axis=1)
log(f'    models: {len(M)} total; LZ-grid physical {M.grid_model.sum()}; distinct {M.distinct.sum()} '
    f'(elastic {M[M.cls == "elastic"].distinct.sum()}, inelastic {M[M.cls == "inelastic"].distinct.sum()})')
R['n_models'] = dict(total=int(len(M)), LZ_grid_physical=int(M.grid_model.sum()), distinct=int(M.distinct.sum()),
                     distinct_elastic=int(M[M.cls == 'elastic'].distinct.sum()), distinct_inelastic=int(M[M.cls == 'inelastic'].distinct.sum()),
                     distinct_light_le50GeV=int(M[(M.cls == 'elastic') & (M.mass <= 50)].distinct.sum()),
                     deltascan_only=int((~M.on_LZ_grid).sum()), LZ_reported=dict(models=616, distinguishable=293))

# validation of N_lo against P016 (1000 GeV)
p16 = pd.read_csv('output/work/P016/P016_operator_table.csv').set_index('model')
val = []
for _, r in M[(M.cls == 'elastic') & (M.mass == 1000)].iterrows():
    key = 'L10_1000' if r.op == 'L10' else f'{r.op}{r.tau}_1000'
    if key in p16.index:
        val.append(dict(model=key, N_lo_P027=r.N_lo, N_lo_P016=p16.loc[key].N_lo, N_M_P027=r.N_M, N_M_P016=p16.loc[key].N_M, rho=r.rho))
val = pd.DataFrame(val); val['ratio'] = val.N_lo_P027 / val.N_lo_P016
val.to_csv(os.path.join(OUT, 'P027_validation_Nlo.csv'), index=False)
R['N_lo_validation_vs_P016_1000GeV'] = dict(median_ratio=float(val.ratio.median()), min=float(val.ratio.min()), max=float(val.ratio.max()),
                                            L10_N_lo=float(val[val.model == 'L10_1000'].N_lo_P027.iloc[0]))
log(f'    N_lo check vs P016 (1000 GeV): ratio median {val.ratio.median():.3f}, range {val.ratio.min():.3f}-{val.ratio.max():.3f}; '
    f'L10 N_lo={R["N_lo_validation_vs_P016_1000GeV"]["L10_N_lo"]:.3f} (P003 0.20)')

# ------------------------------------------------------------------------------------------------
# 2. Marginal likelihoods
# ------------------------------------------------------------------------------------------------
LN_L0 = logsumexp(LNW_TH + np.sum(N_LO[None, :] * np.log(TH[:, None] * B_LO[None, :]) - TH[:, None] * B_LO[None, :], axis=1)) - B_M


def lnLR_mu(mu, f_lo, f_L2, f_M, f_hi, rho):
    """ln [L(mu) / L(0)] for total-ROI signal mu (array), fractions per window, shape factor rho."""
    mu = np.atleast_1d(np.asarray(mu, float))
    f_top = f_lo + f_L2
    lam = TH[None, :, None] * B_LO[None, None, :] + mu[:, None, None] * f_top * G[None, None, :]     # (mu, th, bin)
    ll = np.sum(N_LO * np.log(lam) - lam, axis=2) - (B_M + mu[:, None] * f_M)                          # (mu, th)
    lnW = logsumexp(LNW_TH[None, :] + ll, axis=1) - LN_L0
    s = mu * f_hi
    return -s + np.log1p(rho * s / B_H) + lnW


S_LOG = np.geomspace(1e-3, 30.0, 500)
S_LIN = np.linspace(0.0, 10.0, 4001)
T_JEF = np.linspace(0.0, np.sqrt(10.0), 3001)          # s = t^2


def marginal_B(f, rho, param='s'):
    """Bayes factors vs background for the three priors; param 's' (high-energy events) or 'mu' (total ROI)."""
    f_lo, f_L2, f_M, f_hi = f
    out = {}
    if param == 's':
        if f_hi <= 1e-200:                       # model cannot produce a 200-270 keV event at all
            return dict(logu=0.0, flat=0.0, jeff=0.0, P016flat=0.0)
        conv = lambda s: s / f_hi
    else:
        conv = lambda x: x
    LR_log = np.exp(lnLR_mu(conv(S_LOG), f_lo, f_L2, f_M, f_hi, rho))
    out['logu'] = float(np.trapezoid(LR_log / np.log(S_LOG[-1] / S_LOG[0]), np.log(S_LOG)))
    LR_lin = np.exp(lnLR_mu(conv(S_LIN), f_lo, f_L2, f_M, f_hi, rho))
    out['flat'] = float(np.trapezoid(LR_lin / 10.0, S_LIN))
    LR_j = np.exp(lnLR_mu(conv(T_JEF ** 2), f_lo, f_L2, f_M, f_hi, rho))
    out['jeff'] = float(np.trapezoid(2.0 * LR_j / (2.0 * np.sqrt(10.0)), T_JEF))
    if param == 's':      # P016's approximate formula 1 + (1/b_H) int pi(s) s e^-s W ds (rho = 1) for cross-checking
        lnW = lnLR_mu(conv(S_LIN), f_lo, f_L2, f_M, f_hi, 0.0) + S_LIN          # = lnW(s)
        out['P016flat'] = float(1 + np.trapezoid(S_LIN * np.exp(-S_LIN + lnW) / 10.0, S_LIN) / B_H)
    return out


B_NONE = marginal_B((0.0, 0.0, 0.0, 1.0), 1.0)     # companion-free spectrum
R['B_no_low_energy_penalty'] = B_NONE
log(f'[2] companion-free Bayes factors: log-uniform {B_NONE["logu"]:.1f}, flat {B_NONE["flat"]:.1f}, Jeffreys {B_NONE["jeff"]:.1f}, P016-style {B_NONE["P016flat"]:.1f}')

for j, (_, r) in enumerate(M.iterrows()):
    f = (r.f_lo, r.f_L2, r.f_M, r.f_hi)
    for tag, rho in (('', 1.0), ('_rho', r.rho), ('_rho32', r.rho_sig32)):
        b = marginal_B(f, rho, 's')
        for k, v in b.items():
            M.loc[M.index[j], f'B_{k}{tag}'] = v
    bmu = marginal_B(f, 1.0, 'mu')
    for k in ('logu', 'flat', 'jeff'):
        M.loc[M.index[j], f'Bmu_{k}'] = bmu[k]
    bmu = marginal_B(f, r.rho, 'mu')
    M.loc[M.index[j], 'Bmu_logu_rho'] = bmu['logu']
log(f'    marginal likelihoods done for {len(M)} models, t={time.time() - T0:.0f}s')

# cross-check against P016's table
chk = []
for _, r in M[(M.cls == 'elastic') & (M.mass == 1000)].iterrows():
    key = 'L10_1000' if r.op == 'L10' else f'{r.op}{r.tau}_1000'
    if key in p16.index:
        chk.append(dict(model=key, B_P016=p16.loc[key].B10_flat, B_P027_P016formula=r.B_P016flat, B_flat_exact=r.B_flat, B_logu=r.B_logu, B_logu_rho=r.B_logu_rho))
chk = pd.DataFrame(chk); chk.to_csv(os.path.join(OUT, 'P027_validation_B_vs_P016.csv'), index=False)
R['B_validation_vs_P016'] = {k: dict(P016=float(chk.set_index('model').loc[k].B_P016), P027=float(chk.set_index('model').loc[k].B_P027_P016formula))
                             for k in ('L10_1000', 'O6s_1000', 'O10s_1000', 'O4s_1000', 'O1s_1000')}
log('    B cross-check (P016 flat / ours):', {k: (round(v['P016'], 2), round(v['P027'], 2)) for k, v in R['B_validation_vs_P016'].items()})

# ------------------------------------------------------------------------------------------------
# 3. Model priors, marginal B_DM, posterior over models
# ------------------------------------------------------------------------------------------------
LZ = M[M.on_LZ_grid & M.physical].copy()          # LZ-like model space (no delta-scan-only points)
DIST = LZ[LZ.distinct].copy()


def prior_weights(df, kind):
    w = np.zeros(len(df))
    if kind == 'uniform_spectra':
        w[:] = 1.0
    elif kind == 'uniform_classes':
        for c in df.class4.unique():
            m = (df.class4 == c).values; w[m] = 1.0 / m.sum() / df.class4.nunique()
    elif kind == 'physics_weighted':      # inelastic (both operators) = elastic (both isospins) = 1/2
        for c, wt in (('elastic', 0.5), ('inelastic', 0.5)):
            m = (df.cls == c).values; w[m] = wt / m.sum()
    elif kind == 'heavy_only':
        m = (df.mass >= 100).values; w[m] = 1.0
    return w / w.sum()


PRIORS = {'uniform over distinguishable spectra': ('uniform_spectra', DIST),
          'uniform over 4 classes': ('uniform_classes', DIST),
          'physics-weighted (inelastic = elastic)': ('physics_weighted', DIST),
          'uniform over raw models (no degeneracy merging)': ('uniform_spectra', LZ),
          'heavy WIMPs only (m >= 100 GeV), uniform': ('heavy_only', DIST)}
BCOLS = ['B_logu', 'B_flat', 'B_jeff', 'B_logu_rho', 'B_flat_rho', 'B_jeff_rho', 'B_logu_rho32', 'Bmu_logu', 'Bmu_flat', 'Bmu_jeff', 'Bmu_logu_rho']
bdm_rows, post = [], {}
for pname, (kind, df) in PRIORS.items():
    w = prior_weights(df, kind)
    row = dict(model_prior=pname, n_models=len(df))
    for c in BCOLS:
        B = df[c].values
        row[c] = float(np.sum(w * B))
        if c == 'B_logu':
            p = w * B / np.sum(w * B)
            H = -np.sum(p[p > 0] * np.log(p[p > 0]))
            row['post_entropy_nats'] = float(H); row['N_eff_posterior'] = float(np.exp(H))
            row['B_max'] = float(B.max()); row['bayes_trials_factor_Bmax_over_BDM'] = float(B.max() / row[c])
            row['prior_mass_zero_B'] = float(w[B == 0].sum())
            row['post_inel_delta_ge300'] = float(p[((df.cls == 'inelastic') & (df.delta >= 300)).values].sum())
            row['post_inel_delta_ge250'] = float(p[((df.cls == 'inelastic') & (df.delta >= 250)).values].sum())
            row['post_inelastic'] = float(p[(df.cls == 'inelastic').values].sum())
            row['post_q2spin'] = float(p[df.q2spin.values].sum())
            row['post_SIlike'] = float(p[df.SIlike.values].sum())
            row['post_L10'] = float(p[(df.op == 'L10').values].sum())
            srt = np.sort(p)[::-1]; cs = np.cumsum(srt)
            row['n_models_50pct'] = int(np.searchsorted(cs, 0.5) + 1); row['n_models_90pct'] = int(np.searchsorted(cs, 0.9) + 1)
            post[pname] = pd.Series(p, index=df.name.values)
        if c == 'B_logu_rho':
            p = w * B / np.sum(w * B)
            H = -np.sum(p[p > 0] * np.log(p[p > 0]))
            row['N_eff_posterior_rho'] = float(np.exp(H))
            row['post_inel_delta_ge300_rho'] = float(p[((df.cls == 'inelastic') & (df.delta >= 300)).values].sum())
            row['post_inelastic_rho'] = float(p[(df.cls == 'inelastic').values].sum())
            row['post_q2spin_rho'] = float(p[df.q2spin.values].sum())
            row['post_SIlike_rho'] = float(p[df.SIlike.values].sum())
            post[pname + ' [rho]'] = pd.Series(p, index=df.name.values)
    bdm_rows.append(row)
BDM = pd.DataFrame(bdm_rows); BDM.to_csv(os.path.join(OUT, 'P027_BDM_vs_prior.csv'), index=False)
log('[3] marginal Bayes factors B_DM:')
log(BDM[['model_prior', 'n_models', 'B_logu', 'B_flat', 'B_jeff', 'B_logu_rho', 'Bmu_logu', 'B_max', 'bayes_trials_factor_Bmax_over_BDM',
         'N_eff_posterior', 'n_models_50pct', 'n_models_90pct', 'prior_mass_zero_B', 'post_inelastic', 'post_inel_delta_ge300', 'post_q2spin', 'post_SIlike']].round(3).to_string(index=False))
R['BDM_table'] = BDM.to_dict(orient='records')
# reference numbers from P001 / P008
N_EFF_P008, N_EFF_LZ = 12.2, 13.86
R['reference'] = dict(P001_B_none_over_Neff=dict(b_H_anchor=B_NONE['flat'] / N_EFF_LZ, P008_Neff=B_NONE['flat'] / N_EFF_P008),
                      P016_B_L10=44.8, P001_flat_B10_range='101-501 for b=1e-3..2e-4', SBB_cap_global=14.7)
log(f"    reference: B_none/N_eff(LZ 13.9) = {B_NONE['flat'] / N_EFF_LZ:.1f}; B_none/N_eff(P008 12.2) = {B_NONE['flat'] / N_EFF_P008:.1f}")

# top-10 lists
TOP = {}
for pname in ['uniform over distinguishable spectra', 'uniform over 4 classes', 'physics-weighted (inelastic = elastic)',
              'uniform over distinguishable spectra [rho]', 'physics-weighted (inelastic = elastic) [rho]']:
    p = post[pname].sort_values(ascending=False)
    TOP[pname] = [(k, float(v)) for k, v in p.head(12).items()]
    log(f'    top-10 [{pname}]: ' + ', '.join(f'{k} {v:.3f}' for k, v in p.head(10).items()))
R['top_models'] = TOP
# per-class average B (uniform within class)
cls_rows = []
for c in DIST.class4.unique():
    d = DIST[DIST.class4 == c]
    cls_rows.append(dict(class4=c, n_distinct=len(d), B_logu_mean=d.B_logu.mean(), B_logu_rho_mean=d.B_logu_rho.mean(), B_flat_mean=d.B_flat.mean(),
                         Bmu_logu_mean=d.Bmu_logu.mean(), B_logu_max=d.B_logu.max(), frac_B_zero=(d.B_logu == 0).mean()))
CLS = pd.DataFrame(cls_rows); CLS.to_csv(os.path.join(OUT, 'P027_class_table.csv'), index=False)
log('    class table:\n' + CLS.round(3).to_string(index=False))
R['class_table'] = CLS.to_dict(orient='records')

# ------------------------------------------------------------------------------------------------
# 4. Continuous delta: class Bayes factors with delta integrated
# ------------------------------------------------------------------------------------------------
dsc_rows = []
for (op, tau, m), d in M[M.cls == 'inelastic'].groupby(['op', 'tau', 'mass']):
    d = d.sort_values('delta')
    if op == 'O4' and tau == 'v':
        continue
    dl, ceil = d.delta.values, CEIL[m]
    for tag in ('B_logu', 'B_logu_rho', 'B_flat', 'B_flat_rho', 'B_logu_rho32'):
        B = d[tag].values
        # uniform prior on delta in [0, ceiling]: trapezoid on the 10 keV grid (zero rate beyond ceiling already gives B=0 or B<1)
        mask = dl <= ceil + 10
        B_unif = float(np.trapezoid(B[mask], dl[mask]) / (dl[mask][-1] - dl[mask][0]))
        # log-uniform prior on delta in [10, ceiling]
        ml = (dl >= 10) & mask
        B_logd = float(np.trapezoid(B[ml] / dl[ml], dl[ml]) / np.log(dl[ml][-1] / dl[ml][0]))
        grid = d[d.on_LZ_grid & d.physical]
        B_grid = float(grid[tag].mean()) if len(grid) else np.nan
        B_ge300 = float(grid[grid.delta >= 300][tag].mean()) if (grid.delta >= 300).any() else np.nan
        i_max = int(np.argmax(B))
        # posterior over delta (uniform prior): mean, 68% interval
        pd_ = B[mask] / np.trapezoid(B[mask], dl[mask]); cdf = np.concatenate([[0], np.cumsum(0.5 * (pd_[1:] + pd_[:-1]) * np.diff(dl[mask]))])
        q16, q50, q84 = np.interp([0.16, 0.5, 0.84], cdf, dl[mask])
        dsc_rows.append(dict(op=op, tau=tau, mass=m, ceiling_keV=ceil, Bcol=tag, B_delta_uniform=B_unif, B_delta_loguniform=B_logd, B_LZgrid_mean=B_grid,
                             B_LZgrid_ge300_mean=B_ge300, B_max=float(B[i_max]), delta_at_Bmax=float(dl[i_max]), tuning_penalty_Bmax_over_Bunif=float(B[i_max] / B_unif) if B_unif > 0 else np.nan,
                             delta_post_16=q16, delta_post_50=q50, delta_post_84=q84, frac_delta_prior_with_B_gt_10=float(np.mean(B[mask] > 10))))
DSC = pd.DataFrame(dsc_rows); DSC.to_csv(os.path.join(OUT, 'P027_inelastic_class_delta.csv'), index=False)
M[M.cls == 'inelastic'][['name', 'op', 'tau', 'mass', 'delta', 'physical', 'on_LZ_grid', 'distinct', 'N_lo', 'N_L2', 'N_M', 'f_hi', 'rho', 'rho_sig32', 'B_logu', 'B_flat', 'B_jeff', 'B_logu_rho', 'B_flat_rho', 'Bmu_logu']].to_csv(
    os.path.join(OUT, 'P027_delta_scan.csv'), index=False)
log('[4] continuous-delta class Bayes factors (log-uniform s):')
log(DSC[DSC.Bcol == 'B_logu'][['op', 'tau', 'mass', 'ceiling_keV', 'B_delta_uniform', 'B_delta_loguniform', 'B_LZgrid_mean', 'B_LZgrid_ge300_mean', 'B_max', 'delta_at_Bmax', 'tuning_penalty_Bmax_over_Bunif', 'delta_post_16', 'delta_post_50', 'delta_post_84']].round(2).to_string(index=False))
log('    with shape factor rho:')
log(DSC[DSC.Bcol == 'B_logu_rho'][['op', 'tau', 'mass', 'B_delta_uniform', 'B_delta_loguniform', 'B_LZgrid_mean', 'B_LZgrid_ge300_mean', 'B_max', 'delta_at_Bmax', 'tuning_penalty_Bmax_over_Bunif', 'delta_post_16', 'delta_post_50', 'delta_post_84', 'frac_delta_prior_with_B_gt_10']].round(2).to_string(index=False))
# inelastic class (all op/tau/mass equally weighted) with grid vs continuous delta
inel_summary = {}
for tag in ('B_logu', 'B_logu_rho'):
    dd = DSC[DSC.Bcol == tag]
    inel_summary[tag] = dict(class_B_LZgrid=float(dd.B_LZgrid_mean.mean()), class_B_delta_uniform=float(dd.B_delta_uniform.mean()),
                             class_B_delta_loguniform=float(dd.B_delta_loguniform.mean()), class_B_max=float(dd.B_max.max()),
                             ratio_grid_over_uniform=float(dd.B_LZgrid_mean.mean() / dd.B_delta_uniform.mean()))
R['inelastic_class_summary'] = inel_summary
log('    inelastic class summary:', {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in inel_summary.items()})
# combined B_DM with continuous delta replacing the grid for the inelastic class (physics-weighted and 4-class priors)
el = DIST[DIST.cls == 'elastic']
for tag in ('B_logu', 'B_logu_rho'):
    dd = DSC[DSC.Bcol == tag]
    B_el_s = el[el.tau == 's'][tag].mean(); B_el_v = el[el.tau == 'v'][tag].mean()
    B_in1 = dd[dd.op == 'O1'].B_delta_uniform.mean(); B_in4 = dd[dd.op == 'O4'].B_delta_uniform.mean()
    R[f'BDM_continuous_delta_{tag}'] = dict(uniform_4_classes=float((B_el_s + B_el_v + B_in1 + B_in4) / 4),
                                            physics_weighted=float(0.5 * el[tag].mean() + 0.25 * (B_in1 + B_in4)),
                                            elastic_s=float(B_el_s), elastic_v=float(B_el_v), inel_O1=float(B_in1), inel_O4=float(B_in4))
    log(f'    B_DM with continuous delta ({tag}):', {k: round(v, 2) for k, v in R[f'BDM_continuous_delta_{tag}'].items()})

# ------------------------------------------------------------------------------------------------
# 5. Posterior predictive for the P020 exposure
# ------------------------------------------------------------------------------------------------
K_EXP = 6.76 / 2.84
NMAX = 8
def predictive(r, rho=1.0, prior='logu'):
    f = (r.f_lo, r.f_L2, r.f_M, r.f_hi)
    if r.f_hi <= 1e-200:
        return None
    if prior == 'logu':
        s = S_LOG; pr = 1.0 / (s * np.log(S_LOG[-1] / S_LOG[0])); xs = np.log(s); integrand_w = pr * s   # d ln s
    else:
        s = S_LIN; pr = np.full_like(s, 0.1); xs = s; integrand_w = pr
    LR = np.exp(lnLR_mu(s / r.f_hi, *f, rho))
    w = integrand_w * LR; w /= np.trapezoid(w, xs)
    s_mean = float(np.trapezoid(w * s, xs)); cdf = np.concatenate([[0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(xs))])
    s16, s50, s84 = np.interp([0.16, 0.5, 0.84], cdf, s)
    P = [float(np.trapezoid(w * stats.poisson.pmf(n, K_EXP * (s + B_H)), xs)) for n in range(NMAX)]
    return dict(s_mean=s_mean, s16=s16, s50=s50, s84=s84, N_mean=K_EXP * (s_mean + B_H), **{f'P{n}': P[n] for n in range(NMAX)}, P_ge2=1 - P[0] - P[1], P_ge3=1 - P[0] - P[1] - P[2])


pred_rows = []
p_main = post['uniform over distinguishable spectra']
names_top = list(p_main.sort_values(ascending=False).head(8).index) + ['L10s_m1000', 'O6s_m1000', 'O1s_m1000_d350', 'O4s_m1000_d300', 'O1v_m1000_d300']
seen = set()
for nm in names_top:
    if nm in seen or nm not in set(M.name):
        continue
    seen.add(nm)
    r = M[M.name == nm].iloc[0]
    for prior in ('logu', 'flat'):
        pr = predictive(r, 1.0, prior)
        if pr:
            pred_rows.append(dict(model=nm, prior=prior, rho=1.0, posterior_weight_uniform=float(p_main.get(nm, np.nan)), **pr))
    pr = predictive(r, r.rho, 'logu')
    if pr:
        pred_rows.append(dict(model=nm, prior='logu', rho=r.rho, posterior_weight_uniform=float(p_main.get(nm, np.nan)), **pr))
# model-averaged predictive (posterior over distinct models, log-uniform s) and background-only
for pname in ['uniform over distinguishable spectra', 'physics-weighted (inelastic = elastic)']:
    p = post[pname]; Pmix = np.zeros(NMAX); smean = 0.0
    for nm, pw in p.items():
        if pw < 1e-4:
            continue
        r = DIST[DIST.name == nm].iloc[0]; pr = predictive(r, 1.0, 'logu')
        if pr:
            Pmix += pw * np.array([pr[f'P{n}'] for n in range(NMAX)]); smean += pw * pr['s_mean']
    Pmix /= Pmix.sum()
    pred_rows.append(dict(model=f'model-averaged [{pname}]', prior='logu', rho=1.0, posterior_weight_uniform=1.0, s_mean=smean, N_mean=K_EXP * (smean + B_H),
                          **{f'P{n}': float(Pmix[n]) for n in range(NMAX)}, P_ge2=float(1 - Pmix[0] - Pmix[1]), P_ge3=float(1 - Pmix[:3].sum())))
Pb = stats.poisson.pmf(np.arange(NMAX), K_EXP * B_H)
pred_rows.append(dict(model='background only (b_H bin)', prior='-', rho=1.0, posterior_weight_uniform=0.0, s_mean=0.0, N_mean=K_EXP * B_H,
                      **{f'P{n}': float(Pb[n]) for n in range(NMAX)}, P_ge2=float(1 - Pb[0] - Pb[1]), P_ge3=float(1 - Pb[:3].sum())))
PRED = pd.DataFrame(pred_rows); PRED.to_csv(os.path.join(OUT, 'P027_posterior_predictive.csv'), index=False)
log(f'[5] posterior predictive for k = {K_EXP:.2f} x 2.84 t yr = 6.76 t yr:')
log(PRED[['model', 'prior', 'rho', 's_mean', 's16', 's50', 's84', 'N_mean', 'P0', 'P1', 'P_ge2', 'P_ge3']].round(3).to_string(index=False))
R['k_exposure'] = K_EXP

# ------------------------------------------------------------------------------------------------
# 6. Save the model table and results
# ------------------------------------------------------------------------------------------------
for pname, p in post.items():
    col = 'post_' + {'uniform over distinguishable spectra': 'uniform', 'uniform over 4 classes': 'classes', 'physics-weighted (inelastic = elastic)': 'physics',
                     'uniform over raw models (no degeneracy merging)': 'raw', 'heavy WIMPs only (m >= 100 GeV), uniform': 'heavy'}[pname.replace(' [rho]', '')] + ('_rho' if '[rho]' in pname else '')
    M[col] = M.name.map(p).fillna(0.0)
Mout = M.drop(columns=['spec'])
Mout.to_csv(os.path.join(OUT, 'P027_models.csv'), index=False, float_format='%.6g')
# a compact 1000-GeV elastic table for the paper
t1000 = Mout[(Mout.cls == 'elastic') & (Mout.mass == 1000)].sort_values('B_logu', ascending=False)
t1000[['name', 'N_lo', 'N_M', 'rho', 'B_logu', 'B_flat', 'B_jeff', 'B_logu_rho', 'Bmu_logu', 'post_uniform', 'post_physics']].to_csv(os.path.join(OUT, 'P027_elastic_1000GeV.csv'), index=False, float_format='%.4g')
R['elastic_1000GeV_top'] = t1000[['name', 'N_lo', 'rho', 'B_logu', 'B_flat', 'B_jeff', 'B_logu_rho', 'Bmu_logu']].head(12).round(3).to_dict(orient='records')
# summary numbers for the light models
light = DIST[(DIST.cls == 'elastic') & (DIST.mass <= 50)]
R['light_models'] = dict(n_distinct=int(len(light)), prior_mass_uniform=float(len(light) / len(DIST)), B_logu_all_zero=bool((light.B_logu == 0).all()),
                         Bmu_logu_mean=float(light.Bmu_logu.mean()), Bmu_logu_min=float(light.Bmu_logu.min()), Bmu_logu_max=float(light.Bmu_logu.max()))
log('    light models (<= 50 GeV):', R['light_models'])
# rho statistics
R['rho_stats'] = dict(elastic_heavy_min=float(DIST[(DIST.cls == 'elastic') & (DIST.mass >= 100)].rho.min()), elastic_heavy_max=float(DIST[(DIST.cls == 'elastic') & (DIST.mass >= 100)].rho.max()),
                      inel_by_delta_1000GeV_O1s={str(int(d)): float(M[(M.name == f'O1s_m1000_d{int(d)}')].rho.iloc[0]) for d in range(100, 380, 10) if (M.name == f'O1s_m1000_d{int(d)}').any()},
                      inel_by_delta_1000GeV_O4s={str(int(d)): float(M[(M.name == f'O4s_m1000_d{int(d)}')].rho.iloc[0]) for d in range(100, 380, 10) if (M.name == f'O4s_m1000_d{int(d)}').any()})
log('    rho (1000 GeV O1s by delta):', {k: round(v, 2) for k, v in R['rho_stats']['inel_by_delta_1000GeV_O1s'].items()})
R['runtime_s'] = time.time() - T0
json.dump(R, open(os.path.join(OUT, 'P027_results.json'), 'w'), indent=1, default=float)

# ------------------------------------------------------------------------------------------------
# 7. Figures
# ------------------------------------------------------------------------------------------------
C = dict(blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', yellow='#eda100', magenta='#e87ba4', green='#008300', violet='#4a3aa7', red='#e34948', ink='#0b0b0b', ink2='#52514e', grid='#e6e5e1')
CLASS_COL = {'elastic-isoscalar': C['blue'], 'elastic-isovector': C['aqua'], 'inelastic-O1': C['orange'], 'inelastic-O4': C['violet']}
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': C['ink2'], 'axes.labelcolor': C['ink'], 'xtick.color': C['ink2'], 'ytick.color': C['ink2'], 'axes.spines.top': False, 'axes.spines.right': False})


def pretty(nm):
    op, rest = nm.split('_', 1)
    if '_d' in rest:
        m, d = rest.split('_d'); return f"{op} {m[1:]} GeV δ={d}"
    return f"{op} {rest[1:]} GeV"


# Fig 1: posterior over models (top 20) under three model priors, log-uniform s
fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), sharey=False)
for ax, pname in zip(axes, ['uniform over distinguishable spectra', 'uniform over 4 classes', 'physics-weighted (inelastic = elastic)']):
    p = post[pname].sort_values(ascending=False).head(20)
    cols = [CLASS_COL[DIST.set_index('name').loc[k].class4] for k in p.index]
    y = np.arange(len(p))[::-1]
    ax.barh(y, p.values, color=cols, height=0.7)
    ax.set_yticks(y); ax.set_yticklabels([pretty(k) for k in p.index], fontsize=7)
    ax.set_xlabel('posterior probability of model')
    row = BDM[BDM.model_prior == pname].iloc[0]
    ax.set_title(f"{pname}\nB_DM = {row.B_logu:.1f}, N_eff(post) = {row.N_eff_posterior:.0f}, top-20 mass {p.sum():.2f}", fontsize=8)
    ax.grid(axis='x', color=C['grid'], lw=0.6); ax.set_axisbelow(True)
from matplotlib.patches import Patch
axes[0].legend(handles=[Patch(color=v, label=k) for k, v in CLASS_COL.items()], fontsize=7, loc='lower right', frameon=False)
fig.suptitle('Posterior over models given the 248 keV event and the low-energy null (log-uniform prior on s)', fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P027_fig1_posterior_models.png'), dpi=160); plt.close(fig)

# Fig 2: B_DM vs prior choice
fig, ax = plt.subplots(figsize=(8.2, 4.2))
labels = ['uniform\nspectra', 'uniform\n4 classes', 'physics-\nweighted', 'raw models', 'heavy only']
x = np.arange(len(BDM)); wd_ = 0.13
series = [('B_logu', 'log-uniform s', C['blue']), ('B_flat', 'uniform s [0,10]', C['orange']), ('B_jeff', 'Jeffreys 1/√s', C['aqua']),
          ('B_logu_rho', 'log-uniform s, with shape factor ρ', C['yellow']), ('Bmu_logu', 'log-uniform on total ROI events μ', C['magenta'])]
for i, (col, lab, colr) in enumerate(series):
    ax.bar(x + (i - 2) * wd_, BDM[col], wd_ * 0.92, color=colr, label=lab)
ax.axhline(B_NONE['flat'] / N_EFF_LZ, color=C['ink2'], ls='--', lw=1); ax.text(len(BDM) - 0.55, B_NONE['flat'] / N_EFF_LZ * 1.08, f'P001: B/N_eff = {B_NONE["flat"] / N_EFF_LZ:.1f}', fontsize=7, ha='right', color=C['ink2'])
ax.axhline(14.7, color=C['ink2'], ls=':', lw=1); ax.text(len(BDM) - 0.55, 14.7 * 0.8, 'SBB cap from 2.6σ global: 14.7', fontsize=7, ha='right', color=C['ink2'])
ax.set_yscale('log'); ax.set_ylim(1, 400)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('marginal Bayes factor B_DM (DM vs background)')
ax.set_title(f'Model-marginalised Bayes factor; single best model B_max = {B_NONE["logu"]:.0f} (log-uniform, companion-free)', fontsize=9)
ax.legend(fontsize=7, frameon=False, ncol=2, loc='upper left'); ax.grid(axis='y', color=C['grid'], lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P027_fig2_BDM_vs_prior.png'), dpi=160); plt.close(fig)

# Fig 3: inelastic B(delta) curves
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
mcol = {400: C['aqua'], 1000: C['blue'], 4000: C['violet']}
for ax, op in zip(axes, ('O1', 'O4')):
    for m in INEL_MASSES:
        d = M[(M.cls == 'inelastic') & (M.op == op) & (M.tau == 's') & (M.mass == m)].sort_values('delta')
        ax.plot(d.delta, d.B_logu, color=mcol[m], lw=2, label=f'{m} GeV, binned (ρ = 1)')
        ax.plot(d.delta, d.B_logu_rho, color=mcol[m], lw=1.6, ls='--', label=f'{m} GeV, with shape factor ρ')
        g = d[d.on_LZ_grid]; ax.plot(g.delta, g.B_logu, 'o', color=mcol[m], ms=4, mfc='white', mew=1.2)
        ax.axvline(CEIL[m], color=mcol[m], lw=0.7, ls=':')
    ax.set_yscale('log'); ax.set_ylim(0.3, 400); ax.set_xlabel('mass splitting δ [keV]'); ax.set_title(f'inelastic {op}$^s$: Bayes factor vs δ (log-uniform s)', fontsize=9)
    ax.grid(color=C['grid'], lw=0.6); ax.set_axisbelow(True)
axes[0].set_ylabel('Bayes factor vs background'); axes[0].legend(fontsize=6.5, frameon=False, loc='upper left', ncol=1)
axes[1].text(0.02, 0.04, 'circles: LZ δ grid; dotted: kinematic ceiling μv²_max/2', transform=axes[1].transAxes, fontsize=7, color=C['ink2'])
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P027_fig3_inelastic_delta.png'), dpi=160); plt.close(fig)

# Fig 4: posterior predictive
fig, ax = plt.subplots(figsize=(7.5, 3.8))
sel_pred = PRED[(PRED.prior.isin(['logu', '-'])) & (PRED.rho == 1.0)]
keys = ['L10s_m1000', 'O6s_m1000', 'O1s_m1000_d350', 'O4s_m1000_d300', 'model-averaged [uniform over distinguishable spectra]', 'background only (b_H bin)']
cols4 = [C['blue'], C['aqua'], C['orange'], C['violet'], C['yellow'], C['ink2']]
n = np.arange(6); wd_ = 0.13
for i, (k, colr) in enumerate(zip(keys, cols4)):
    r = sel_pred[sel_pred.model == k]
    if len(r) == 0:
        continue
    r = r.iloc[0]
    ax.bar(n + (i - 2.5) * wd_, [r[f'P{j}'] for j in n], wd_ * 0.92, color=colr, label=f"{k.split(' [')[0]} (mean {r.N_mean:.2f})")
ax.set_xticks(n); ax.set_xlabel('events in the 200-270 keV NR band in the next 6.76 t·yr'); ax.set_ylabel('posterior predictive probability')
ax.legend(fontsize=7, frameon=False); ax.grid(axis='y', color=C['grid'], lw=0.6); ax.set_axisbelow(True)
ax.set_title('Posterior predictive (log-uniform s, per model) vs background', fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P027_fig4_predictive.png'), dpi=160); plt.close(fig)
log(f'done in {time.time() - T0:.0f}s')
LOG.close()
