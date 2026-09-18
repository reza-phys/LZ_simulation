"""
P090 -- Propagating the 100 % MSSI uncertainty into the local and global significance:
        what a factor-2, factor-10 or mis-shaped wall/RFR background does to the 3.4 sigma.

Run from the simulation root:
  .venv/bin/python output/code/P090_mssi_systematic.py          (single stage, ~1-2 min)

Engine: P052's three-sample (S1c, d) likelihood (output/code/P052_reproduce_significance.py) is loaded by
executing its model-building header (everything before the stage dispatch) in a private namespace, with its
log and one CSV redirected so that nothing under output/work/P052/ is modified.  Its caches (yield table,
P003/P012/P021 spectra, P004/P016/P022 digitisations) are read unchanged.  Configuration 'lzlike' (Fig. 5 signal
d-shape, 23 keV resolution), model L10 (1000 GeV), i.e. P052's headline 3.11 sigma asymptotic / 3.03 toy.

Extensions (this work; every equation in work/P090/details.md):
  * t_MSSI is fitted as u = ln t with prior  Gaussian 100 %:  0.5 (e^u - 1)^2   (LZ / P052),
                                             log-normal:      0.5 (u / sigma_ln)^2,
                                             flat:            0.
  * FV rate factor f multiplies the MSSI expectation in the science and prompt samples (R_MSSI -> f R_MSSI).
  * Optional sideband Poisson terms for the six wall bins of LZ's MSSI comparison table (P004's common-k model):
    mu_i = k m_i with k = t (common) or k = t (sidebands) while the FV carries f t (FV-only / shape-driven).
  * Replaceable MSSI cell map (panel split, d-shape mixture, position re-weighting of the event's cell).
  * LEE: N_eff(Z_local) from P071's debiased table (4.87/6.12/9.01/14.2 at 2.0/2.5/3.0/3.4 sigma), power-law
    interpolation; p_global = 1 - (1 - p_local)^N_eff.
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
from iminuit import Minuit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

T0 = time.time()
ROOT = '/Users/reza/LZ_simulation'
os.chdir(ROOT)
OUT = 'output/work/P090'; FIG = os.path.join(OUT, 'figures'); os.makedirs(FIG, exist_ok=True)
LOGF = open(os.path.join(OUT, 'run_log.txt'), 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); LOGF.write(s + '\n'); LOGF.flush()
R = {}

# ----------------------------------------------------------------------------------------------
# 0. Load P052's engine (header only), without touching P052's output directory
# ----------------------------------------------------------------------------------------------
src = open('output/code/P052_reproduce_significance.py').read()
head = src.split('# ' + '=' * 94)[0]
assert 'if ARGS.stage' not in head
head = head.replace("ARGS = ap.parse_args()", "ARGS = ap.parse_args(['--stage', 'fit'])")
head = head.replace("LOG = open(os.path.join(OUT, f'run_log_{ARGS.stage}.txt'), 'a')",
                    "LOG = open('output/work/P090/P052_engine_log.txt', 'w')")
head = head.replace("os.path.join(OUT, 'P052_fig5_L10_digitised.csv'), index=False)", "os.devnull, index=False)")
head = head.replace("os.makedirs(FIG, exist_ok=True); os.makedirs(TOYDIR, exist_ok=True)", "pass")
assert "output/work/P090/P052_engine_log.txt" in head and "os.devnull" in head
ns = {'__name__': 'p052_engine'}
exec(compile(head, 'P052_engine', 'exec'), ns)
say(f'P052 engine loaded in {time.time()-T0:.1f} s: {len(ns["MODELS"])} models')
lz = ns['lz']
PNAMES, NOM, CON, LIMITS = ns['PNAMES'], ns['NOM'], ns['CON'], ns['LIMITS']
NS1, ND, S1_EDGES, D_EDGES, D_MID = ns['NS1'], ns['ND'], ns['S1_EDGES'], ns['D_EDGES'], ns['D_MID']
K_EV, J_EV = ns['K_EV'], ns['J_EV']
MSSI0, MSSI_TOT, R_MSSI0, F_DET, R_DET, R_XE = ns['MSSI'], ns['MSSI_TOT'], ns['R_MSSI'], ns['F_DET'], ns['R_DET'], ns['R_XE']
PROMPT_SMALL_ER, DELAYED_BKG, SIG_SPLIT = ns['PROMPT_SMALL_ER'], ns['DELAYED_BKG'], ns['SIG_SPLIT']
DATA, N_PROMPT, N_DELAYED = ns['DATA'], ns['N_PROMPT'], ns['N_DELAYED']
mssi_shape, MSSI_PANEL, assemble = ns['mssi_shape'], ns['MSSI_PANEL'], ns['assemble']
w_flat_mid, w_flat_bot, gauss_hist = ns['w_flat_mid'], ns['w_flat_bot'], ns['gauss_hist']
S_L10, _, _ = ns['sig_for']('L10_1000', 'lzlike')
CON['tMSSI'] = 1e9            # the base-class Gaussian on t_MSSI is switched off; the prior is supplied below

# LZ MSSI comparison table (supplement), wall rows outside the blind 4.7 t WS ROI: (simulated, observed)
WALL_SCI = [(0.03, 0), (0.1, 0), (0.5, 0)]        # WS ROI 5.4 t annulus; HE SB 4.7 t; HE SB 5.4 t annulus
WALL_PROMPT = [(0.1, 1), (0.3, 1), (2.2, 0)]
RFR_SCI = [(0.003, 0), (0.5, 0), (21.5, 18)]
RFR_PROMPT = [(1.9, 2), (0.5, 0), (5.2, 3)]
WALL_BINS = WALL_SCI + WALL_PROMPT
M_WALL, N_WALL = sum(m for m, n in WALL_BINS), sum(n for m, n in WALL_BINS)      # 3.23, 2
MSSI_FV_WALL, MSSI_FV_RFR = 0.0048, 0.0001
R['sideband_table'] = dict(wall_sci=WALL_SCI, wall_prompt=WALL_PROMPT, rfr_sci=RFR_SCI, rfr_prompt=RFR_PROMPT,
                           wall_total_sim=M_WALL, wall_total_obs=N_WALL)

# ----------------------------------------------------------------------------------------------
# 1. Extended likelihood
# ----------------------------------------------------------------------------------------------
class MLike(ns['Likelihood']):
    """P052 likelihood with: u = ln t_MSSI, choice of prior, FV rate factor f, sideband terms, MSSI map."""
    def __init__(self, S=S_L10, prior='gauss', sig_ln=1.0, f=1.0, sidebands=False, sb_common=True,
                 mssi_cells=None, sb_bins=WALL_BINS, event_pos_weight=1.0):
        super().__init__(S)
        self.prior, self.sig_ln, self.f = prior, sig_ln, f
        self.sidebands, self.sb_common, self.sb_bins = sidebands, sb_common, sb_bins
        if mssi_cells is not None:
            self.c['MSSI'] = np.asarray(mssi_cells, float).ravel().copy()
        if event_pos_weight != 1.0:
            M = self.c['MSSI'].reshape(NS1, ND).copy(); M[K_EV, J_EV] *= event_pos_weight; self.c['MSSI'] = M.ravel()
    def tval(self, p): return float(np.exp(p['tMSSI']))
    def mu_cells(self, p):
        c = self.c; t = self.tval(p)
        sci_mssi = t * self.f * R_MSSI0 * (1 - p['lMSSI']) / MSSI_TOT
        return (p['s'] * c['S'] + p['tER'] * (1 - F_DET) * (c['ERB'] + p['tTail'] * c['ERT']) + p['tAcc'] * c['ACC']
                + sci_mssi * c['MSSI'] + p['tNu'] * c['NU'] + p['Nn'] * c['NEUT']
                + (1 - p['lPG']) * (R_DET * p['tDet'] + R_XE * p['tXe']) * c['PLAT'])
    def mu_veto(self, p):
        t = self.tval(p); f_sci = max(1 - p['lPN'] - p['lDN'], 1e-3)
        mu_p = (PROMPT_SMALL_ER * p['tER'] + p['lPG'] * (R_DET * p['tDet'] + R_XE * p['tXe']) + p['lMSSI'] * R_MSSI0 * self.f * t
                + p['lPN'] / f_sci * p['Nn'] + SIG_SPLIT['prompt'] * p['s'])
        mu_d = DELAYED_BKG * p['tD'] + p['lDN'] / f_sci * p['Nn'] + SIG_SPLIT['delayed'] * p['s']
        return mu_p, mu_d
    def extra(self, p):
        u = p['tMSSI']; t = np.exp(u); val = 0.0
        if self.prior == 'gauss': val += 0.5 * (t - 1.0) ** 2
        elif self.prior == 'lognormal': val += 0.5 * (u / self.sig_ln) ** 2
        if self.sidebands:
            k = t * (self.f if self.sb_common else 1.0)
            for m, n in self.sb_bins:
                mu = max(k * m, 1e-300); val += mu - n * np.log(mu)
        return val
    def nll(self, *args):
        return super().nll(*args) + self.extra(dict(zip(PNAMES, args)))
    def fit(self, fix_s=None, start=None):
        st = dict(NOM) if start is None else dict(start)
        if start is None: st['tMSSI'] = 0.0
        if fix_s is not None: st['s'] = fix_s
        m = Minuit(self.nll, *[st[k] for k in PNAMES], name=PNAMES)
        for k in PNAMES: m.limits[k] = LIMITS[k]
        m.limits['tMSSI'] = (-9.0, 9.0)
        m.errordef = 0.5; m.strategy = 0; m.print_level = 0
        if fix_s is not None: m.fixed['s'] = True
        m.migrad()
        if not m.valid:
            m.strategy = 1; m.migrad()
        return m
    def q0_full(self):
        q, sh, m0, m1 = self.q0(return_fits=True)
        p0 = dict(zip(PNAMES, m0.values)); t0 = self.tval(p0)
        b_ev = float(self.mu_cells(p0).reshape(NS1, ND)[K_EV, J_EV])
        t1 = self.tval(dict(zip(PNAMES, m1.values))) if m1 is not None else t0
        return dict(q0=float(q), Z=float(np.sqrt(max(q, 0.0))), s_hat=float(sh), t_H0=t0, t_H1=t1, b_event_cell_H0=b_ev,
                    mssi_FV_H0=float(t0 * self.f * MSSI_TOT))

# ----------------------------------------------------------------------------------------------
# 2. LEE mapping (P071 debiased N_eff vs local threshold; P008 as alternative) and toy offset (P052)
# ----------------------------------------------------------------------------------------------
Z_T = np.array([2.0, 2.5, 3.0, 3.4]); N_P071 = np.array([4.87, 6.12, 9.01, 14.2]); N_P008 = np.array([4.8, 6.0, 8.3, 12.2])
def neff(z, table=N_P071):
    lz_, ln_ = np.log(Z_T), np.log(table)
    if z <= Z_T[0]: sl = (ln_[1] - ln_[0]) / (lz_[1] - lz_[0]); return float(np.exp(ln_[0] + sl * (np.log(max(z, 0.3)) - lz_[0])))
    if z >= Z_T[-1]: sl = (ln_[-1] - ln_[-2]) / (lz_[-1] - lz_[-2]); return float(np.exp(ln_[-1] + sl * (np.log(z) - lz_[-1])))
    return float(np.exp(np.interp(np.log(z), lz_, ln_)))
def zglobal(z, table=N_P071, dn=0.0):
    if z <= 0: return 0.0
    p = stats.norm.sf(z); n = max(neff(z, table) + dn, 1.0); pg = 1 - (1 - p) ** n
    return float(stats.norm.isf(pg)) if pg < 0.5 else 0.0
Z_TOY_OFFSET = 3.03 - 3.11        # P052 lzlike L10: toy-calibrated 3.03 [2.98, 3.10] vs asymptotic 3.11
R['lee_mapping'] = dict(Z_table=Z_T.tolist(), Neff_P071=N_P071.tolist(), Neff_P008=N_P008.tolist(),
                        Neff_at={z: dict(P071=neff(z), P008=neff(z, N_P008)) for z in (1.0, 1.5, 2.0, 2.5, 3.0, 3.11, 3.4)},
                        Zglobal_check_3p4=dict(P071=zglobal(3.4), P008=zglobal(3.4, N_P008), LZ=2.6), toy_offset=Z_TOY_OFFSET)
say('LEE mapping:', json.dumps(R['lee_mapping'], default=float))

Z_BASE_P052 = 3.108763718983941; LZ_ANCHOR = 3.4 - Z_BASE_P052      # +0.29: P052's residual to LZ's 3.4 (unresolved 250-500 phd S1c structure)
def row(label, res, **kw):
    d = dict(label=label, **res); d['Z_toy_est'] = d['Z'] + Z_TOY_OFFSET if d['Z'] > 0 else 0.0
    d['Neff'] = neff(d['Z']) if d['Z'] > 0 else np.nan; d['Z_global'] = zglobal(d['Z']); d['Z_global_P008'] = zglobal(d['Z'], N_P008)
    d['Z_global_toy_est'] = zglobal(d['Z_toy_est'])
    d['dZ_local'] = d['Z'] - Z_BASE_P052; d['Z_local_LZanch'] = max(d['Z'] + LZ_ANCHOR, 0.0) if d['Z'] > 0 else 0.0
    d['Z_global_LZanch'] = zglobal(d['Z_local_LZanch']); d.update(kw); return d

# ----------------------------------------------------------------------------------------------
# 3. Baseline validation against P052
# ----------------------------------------------------------------------------------------------
base = MLike().q0_full()
p052 = json.load(open('output/work/P052/P052_results.json'))
say('baseline (u = ln t, Gaussian prior):', json.dumps(base, default=float))
say('P052 lzlike L10: q0 %.3f Z %.3f s_hat %.3f b_ev %.3e' % (p052['L10_1000']['lzlike']['q0'], p052['L10_1000']['lzlike']['Z_asym'],
    p052['L10_1000']['lzlike']['s_hat'], p052['L10_1000']['lzlike']['b_event_cell_H0']))
x2 = MLike(f=2.0).q0_full()
say('MSSI x2: Z %.3f (P052 variant 2.975)' % x2['Z'])
nomssi = MLike(f=1e-6).q0_full()
M_ev = MSSI0[K_EV, J_EV]
R['baseline'] = dict(this=base, P052=dict(q0=p052['L10_1000']['lzlike']['q0'], Z=p052['L10_1000']['lzlike']['Z_asym'], s_hat=p052['L10_1000']['lzlike']['s_hat'],
                     b_event_cell_H0=p052['L10_1000']['lzlike']['b_event_cell_H0'], Z_toy=3.03, Z_toy_68=[2.98, 3.10], Z_LZ=3.4),
                     mssi_x2=x2, P052_mssi_x2_Z=2.975, no_mssi=nomssi,
                     mssi_event_cell_nominal=float(M_ev), mssi_share_of_event_cell=float(M_ev / base['b_event_cell_H0']),
                     mssi_event_cell_fraction_of_total=float(M_ev / MSSI_TOT))
# neighbourhood N = {S1c > 500, |d| < 2}: fraction of the MSSI model there and per-cell density
NB = np.zeros((NS1, ND), bool); NB[11:, (D_MID > -2) & (D_MID < 2)] = True
c_nb = float(MSSI0[NB].sum() / MSSI_TOT); n_cells_nb = int(NB.sum())
R['baseline'].update(neighbourhood_fraction_c0=c_nb, neighbourhood_cells=n_cells_nb, g_max_all_in_neighbourhood=float((1.0 / n_cells_nb) / (M_ev / MSSI_TOT)),
                     bottom_panel_frac_within_2sig=float(mssi_shape[(D_MID > -2) & (D_MID < 2)].sum()),
                     bottom_panel_frac_event_bin=float(mssi_shape[J_EV]), P004_f_nb=0.035, P033_f_nb=0.053)
say('baseline extras:', {k: v for k, v in R['baseline'].items() if not isinstance(v, dict)})

# ----------------------------------------------------------------------------------------------
# 4. Part 1: rate priors and the k ladder, with and without sideband anchoring
# ----------------------------------------------------------------------------------------------
PRIORS = [('gauss100', 'gauss', 1.0), ('LN0.7', 'lognormal', 0.7), ('LN1.5', 'lognormal', 1.5), ('LN2.3', 'lognormal', 2.3), ('flat', 'flat', 1.0)]
rows = []
for pl, pr, sg in PRIORS:
    for sb in (False, True):
        res = MLike(prior=pr, sig_ln=sg, sidebands=sb).q0_full()
        rows.append(row(f'{pl}{" + sidebands" if sb else ""}', res, prior=pl, sidebands=sb, f=1.0))
        say(f'prior {pl:9s} sidebands {sb!s:5s}: Z {res["Z"]:.3f} t_H0 {res["t_H0"]:.3f} t_H1 {res["t_H1"]:.3f} MSSI_FV(H0) {res["mssi_FV_H0"]:.4f}')
df_prior = pd.DataFrame(rows); df_prior.to_csv(os.path.join(OUT, 'P090_table1_priors.csv'), index=False)

LADDER = [1, 2, 5, 10, 20, 50, 100, 200, 500]
rows = []
for f in LADDER:
    for mode in ('common', 'FV-only'):
        # 'common': f scales FV and sidebands alike (a geometry error of the dead layer, P004); sidebands included.
        # 'FV-only': f scales the FV only (a shape/extrapolation error); sidebands see k only.
        res = MLike(prior='gauss', f=f, sidebands=True, sb_common=(mode == 'common')).q0_full()
        rows.append(row(f'x{f} {mode}', res, f=f, mode=mode))
    res = MLike(prior='gauss', f=f, sidebands=False).q0_full()
    rows.append(row(f'x{f} LZ-style (no sideband terms)', res, f=f, mode='none'))
    say(f'f={f:4d}: Z common {rows[-3]["Z"]:.3f} FV-only {rows[-2]["Z"]:.3f} no-SB {rows[-1]["Z"]:.3f}  MSSI_FV(H0) {rows[-1]["mssi_FV_H0"]:.4f}')
df_ladder = pd.DataFrame(rows); df_ladder.to_csv(os.path.join(OUT, 'P090_table2_rate_ladder.csv'), index=False)

def Z_of_f(f, mode='FV-only'):
    return MLike(prior='gauss', f=f, sidebands=True, sb_common=(mode == 'common')).q0_full()['Z']
R['f_required'] = {}
for mode in ('FV-only', 'common'):
    R['f_required'][mode] = {}
    for zt in (3.0, 2.0, 1.0):
        try: fz = brentq(lambda x: Z_of_f(x, mode) - zt, 1.0, 5000.0, xtol=1e-3, rtol=1e-3)
        except ValueError: fz = np.nan
        R['f_required'][mode][f'Z={zt}'] = float(fz)
    say(f'f required ({mode}):', R['f_required'][mode])

def sideband_pvalues(f):
    """Poisson p-values of LZ's own wall sideband counts if the wall-MSSI rate were f times the model everywhere."""
    p6 = float(stats.poisson.cdf(N_WALL, f * M_WALL))                       # P(N_6 <= 2 | 3.23 f)
    p_ann = float(stats.poisson.pmf(0, f * 0.03))                              # 5.4 t annulus, WS ROI science: 0 observed
    p_hesb = float(stats.poisson.cdf(0, f * (0.1 + 0.5)))                      # HE SB wall science bins: 0 observed
    p_prompt = float(stats.poisson.cdf(2, f * (0.1 + 0.3 + 2.2)))              # prompt wall bins: 2 observed
    p_prompt47 = float(stats.poisson.cdf(N_PROMPT, f * R_MSSI0 * 0.94 + (66.0 - R_MSSI0 * 0.94)))   # 4.7 t prompt total, MSSI part scaled
    # Baker-Cousins chi2 over the six wall bins
    chi2 = 2 * sum((f * m - n + (n * np.log(n / (f * m)) if n > 0 else 0.0)) for m, n in WALL_BINS)
    return dict(f=f, p_six_wall_bins=p6, p_annulus_sci=p_ann, p_HESB_wall_sci=p_hesb, p_wall_prompt=p_prompt, p_prompt_total_4p7t=p_prompt47,
                chi2_BC=float(chi2), Z_six_wall=float(stats.norm.isf(p6)) if p6 < 0.5 else 0.0)
sb_rows = [sideband_pvalues(f) for f in [1, 1.64, 2, 5, 10, 20, 50, 100, 200, 500]]
for mode in ('FV-only', 'common'):
    for zt in (3.0, 2.0, 1.0):
        fz = R['f_required'][mode][f'Z={zt}']
        if np.isfinite(fz): d = sideband_pvalues(fz); d['note'] = f'{mode}: Z_local = {zt}'; sb_rows.append(d)
df_sb = pd.DataFrame(sb_rows); df_sb.to_csv(os.path.join(OUT, 'P090_table3_sideband_pvalues.csv'), index=False)
say(df_sb.to_string())
# P004-style profile limit on the common k from the six wall bins alone (check)
kk = np.linspace(0.01, 6, 3000); nll_k = np.array([sum(k * m - n * np.log(k * m) for m, n in WALL_BINS) for k in kk])
k_ml = float(kk[np.argmin(nll_k)]); dn = nll_k - nll_k.min()
def k_upper(cl_two_sided_equiv):      # one-sided upper limit: Delta(2 nll) = chi2_1 quantile at 2 CL - 1
    thr = 0.5 * stats.chi2.ppf(cl_two_sided_equiv, 1); sel = (kk > k_ml) & (dn > thr)
    return float(kk[sel][0]) if np.any(sel) else np.nan
R['sideband_k_check'] = dict(k_ML=k_ml, k95_one_sided_LR=k_upper(0.90), k95_two_sided_LR=k_upper(0.95), P004=dict(k_ML=0.62, k95=1.64))
say('sideband k check:', R['sideband_k_check'])

# ----------------------------------------------------------------------------------------------
# 5. Part 2: shape systematics
# ----------------------------------------------------------------------------------------------
# 5a. wall depth scale lambda: (i) annulus-anchored FV rate factor f_lambda = ratio(1.23)/ratio(lambda) from P079's
#     digitised-contour integrals; (ii) position PDF of wall MSSI at the event's depth from P070's FV volume profile.
P079_RATIO = {1.23: 6.25, 3.0: 1.34, 4.3: 0.86, 4.8: 0.76, 6.0: 0.60, 8.0: 0.46}    # annulus(5.4 t)/FV(4.7 t) wall MSSI
lam_t = np.array(sorted(P079_RATIO)); rat_t = np.array([P079_RATIO[l] for l in lam_t])
def annulus_ratio(lam): return float(np.exp(np.interp(np.log(lam), np.log(lam_t), np.log(rat_t))))
def f_lambda(lam): return 6.25 / annulus_ratio(lam)
prof = pd.read_csv('output/work/P070/profiles_d.csv')
d_c = prof.d_centre.values; V = prof.uniform.values; V = V / V.sum()           # FV volume fraction per 1-cm shell in wall distance
D_EV, D_FV = 26.9, 8.0                                                           # event's distance from the wall (P033/P073); FV stand-off
def pos_pdf(lam):
    w = V * np.exp(-(d_c - D_FV) / lam); return w / w.sum()
def LR_pos(lam):
    i = int(np.argmin(np.abs(d_c - D_EV))); return float(V[i] / pos_pdf(lam)[i])
def frac_beyond(lam, d0=25.0): return float(pos_pdf(lam)[d_c >= d0].sum())
R['position_check_vs_P070'] = {f'lambda={l}': dict(LR_this=LR_pos(l), LR_P070=v) for l, v in ((3.0, 21.7), (4.3, 5.73), (6.0, 2.63))}
say('position LR check vs P070 (exp_true):', R['position_check_vs_P070'])
rows = []
for lam in (1.23, 2.0, 3.0, 4.3, 6.0, 8.0):
    fl = f_lambda(lam); lr = LR_pos(lam)
    blind = MLike(prior='gauss', f=fl, sidebands=True, sb_common=False).q0_full()
    aware = MLike(prior='gauss', f=fl, sidebands=True, sb_common=False, event_pos_weight=1.0 / lr).q0_full()
    rows.append(row(f'lambda={lam} cm position-blind', blind, lam=lam, f_lambda=fl, annulus_ratio=annulus_ratio(lam), LR_pos=lr, frac_beyond_25cm=frac_beyond(lam), position='blind'))
    rows.append(row(f'lambda={lam} cm position-aware', aware, lam=lam, f_lambda=fl, annulus_ratio=annulus_ratio(lam), LR_pos=lr, frac_beyond_25cm=frac_beyond(lam), position='aware'))
    say(f'lambda {lam:4.2f}: f_lambda {fl:6.2f}  LR_pos {lr:9.3g}  f(d>25cm) {frac_beyond(lam):.2e}  Z blind {blind["Z"]:.3f} aware {aware["Z"]:.3f}')
df_lam = pd.DataFrame(rows); df_lam.to_csv(os.path.join(OUT, 'P090_table4_depth_scale.csv'), index=False)

# 5b. d-distribution tail and S1c distribution of the MSSI model inside the ROI (total fixed at 0.0049)
def mssi_map(f_bot=None, a_gauss=0.0, sig_g=1.0, flat_band=False):
    shape = (1 - a_gauss) * mssi_shape + a_gauss * gauss_hist(0.0, sig_g)
    if flat_band:
        shape = np.where((D_MID > -2) & (D_MID < 2), 1.0, 0.0); shape /= shape.sum()
    panel = MSSI_PANEL.copy()
    if f_bot is not None:
        panel = np.array([0.5 * (1 - f_bot), 0.5 * (1 - f_bot), f_bot]) * MSSI_TOT
    return assemble(shape * panel[0], shape * panel[1], shape * panel[2], w_flat_mid, w_flat_bot), shape
rows = []
SHAPES = [('baseline (P004 d-shape, 10 % at S1c>500)', dict()),
          ('d: 25 % N(0,1) admixture', dict(a_gauss=0.25)), ('d: 50 % N(0,1) admixture', dict(a_gauss=0.5)),
          ('d: 100 % N(0,1) (MSSI on the NR band)', dict(a_gauss=1.0)), ('d: flat within |d|<2 only', dict(flat_band=True)),
          ('S1c: 20 % at S1c>500', dict(f_bot=0.2)), ('S1c: 50 % at S1c>500', dict(f_bot=0.5)), ('S1c: 100 % at S1c>500', dict(f_bot=1.0)),
          ('S1c 100 % >500 and d flat |d|<2 (all MSSI in the neighbourhood)', dict(f_bot=1.0, flat_band=True))]
for lab, kw in SHAPES:
    M, sh = mssi_map(**kw); res = MLike(prior='gauss', sidebands=True, sb_common=True, mssi_cells=M).q0_full()
    g = float(M[K_EV, J_EV] / M_ev)
    rows.append(row(lab, res, g_event_cell=g, frac_within_2sig_bottom=float(sh[(D_MID > -2) & (D_MID < 2)].sum()), frac_event_bin=float(sh[J_EV]),
                    neighbourhood_fraction=float(M[NB].sum() / MSSI_TOT), mssi_S1c_gt500=float(M[11:].sum())))
    say(f'{lab:70s} g {g:6.2f} f2sig {rows[-1]["frac_within_2sig_bottom"]:.3f} Z {res["Z"]:.3f}')
df_shape = pd.DataFrame(rows); df_shape.to_csv(os.path.join(OUT, 'P090_table5_dshape.csv'), index=False)

# 5c. RFR MSSI: FV expectation 1e-4 (49x below wall); scaling k_RFR with its own six sideband bins
RFR_BINS = RFR_SCI + RFR_PROMPT; M_RFR, N_RFR = sum(m for m, n in RFR_BINS), sum(n for m, n in RFR_BINS)
rfr_rows = []
for kr in (1, 10, 100, 500, 1000, 5000):
    f_eq = 1 + (kr - 1) * MSSI_FV_RFR / MSSI_TOT          # equivalent total-MSSI factor if RFR shares the wall (S1c, d) shape
    res = MLike(prior='gauss', f=f_eq, sidebands=False).q0_full()
    rfr_rows.append(row(f'k_RFR={kr}', res, k_RFR=kr, f_equiv=f_eq, p_HESB_RFR_sci=float(stats.poisson.cdf(18, kr * 21.5)),
                        p_six_rfr_bins=float(stats.poisson.cdf(N_RFR, kr * M_RFR))))
    say(f'RFR k={kr:5d}: f_equiv {f_eq:7.2f} Z {res["Z"]:.3f} p(HE-SB RFR 18 | {kr*21.5:.3g}) {rfr_rows[-1]["p_HESB_RFR_sci"]:.2e}')
df_rfr = pd.DataFrame(rfr_rows); df_rfr.to_csv(os.path.join(OUT, 'P090_table6_rfr.csv'), index=False)

# ----------------------------------------------------------------------------------------------
# 6. Part 4: the 'MSSI-only' explanation -- minimal joint (rate f x shape g) deformation
# ----------------------------------------------------------------------------------------------
# Z depends on the MSSI expectation at the event's cell, f*g*M_ev (the rest of the map carries < 0.01 events).
def Z_of_fg(f, g, sidebands=True, common=True):
    M = MSSI0.copy(); M[K_EV, J_EV] *= g       # shape deformation: concentrate MSSI into the event's cell (total fixed to first order)
    return MLike(prior='gauss', f=f, sidebands=sidebands, sb_common=common, mssi_cells=M).q0_full()
fg = np.array([1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 1000], float)
rows = []
for x in fg:
    a = Z_of_fg(x, 1.0, sidebands=False); b = Z_of_fg(1.0, x)
    rows.append(dict(fg=x, Z_rate_only_noSB=a['Z'], Z_shape_only=b['Z'], b_event_cell_H0=a['b_event_cell_H0'], mssi_event_cell=x * M_ev,
                     Z_global_rate=zglobal(a['Z']), Z_global_shape=zglobal(b['Z'])))
    say(f'f*g={x:6.1f}: Z (rate, no SB) {a["Z"]:.3f}  Z (shape only) {b["Z"]:.3f}  b_ev {a["b_event_cell_H0"]:.2e}')
df_fg = pd.DataFrame(rows); df_fg.to_csv(os.path.join(OUT, 'P090_table7_fg_product.csv'), index=False)
R['fg_required'] = {}
for zt in (3.0, 2.0, 1.0):
    R['fg_required'][f'Z={zt}'] = float(brentq(lambda x: Z_of_fg(1.0, x)['Z'] - zt, 1.0, 5000.0, xtol=1e-3, rtol=1e-3))
say('f*g product required for Z_local = 3/2/1:', R['fg_required'])
g_max = R['baseline']['g_max_all_in_neighbourhood']
R['mssi_only'] = {}
for zt in (2.0, 3.0):
    x = R['fg_required'][f'Z={zt}']
    R['mssi_only'][f'Z={zt}'] = dict(fg=x, rate_only_f=x, rate_only_sideband_p6=sideband_pvalues(x)['p_six_wall_bins'], rate_only_p_annulus=sideband_pvalues(x)['p_annulus_sci'],
                                     shape_only_g_needed=x, shape_max_g=g_max, shape_only_possible=bool(x <= g_max),
                                     f_at_gmax=x / g_max, sideband_p6_at_f_gmax=sideband_pvalues(max(x / g_max, 1.0))['p_six_wall_bins'],
                                     lambda43_f=f_lambda(4.3), g_needed_with_lambda43=x / f_lambda(4.3),
                                     lambda43_position_aware_g_needed=x / f_lambda(4.3) * LR_pos(4.3),
                                     mssi_at_S1c_gt500_needed_if_rate_only=float(x * MSSI0[11:].sum()),
                                     mssi_in_neighbourhood_needed_events=float(x * M_ev * n_cells_nb))
say('MSSI-only:', json.dumps(R['mssi_only'], default=float))

# ----------------------------------------------------------------------------------------------
# 7. Summary table (Part 5)
# ----------------------------------------------------------------------------------------------
def pick(df, lab): return df[df.label == lab].iloc[0]
summ = []
def add(name, r, sb_p=None, comment=''):
    summ.append(dict(hypothesis=name, Z_local=r['Z'], dZ_local=r['dZ_local'], Z_local_toy_est=r['Z_toy_est'], Neff=r['Neff'], Z_global=r['Z_global'],
                     Z_global_P008=r['Z_global_P008'], Z_local_LZanch=r['Z_local_LZanch'], Z_global_LZanch=r['Z_global_LZanch'],
                     mssi_FV_H0=r['mssi_FV_H0'], b_event_cell_H0=r['b_event_cell_H0'], sideband_p=sb_p, comment=comment))
# ROI self-constraint: the rate-only (no sideband) curve has a floor because the WS ROI itself has no events where MSSI would populate
sub_none = df_ladder[df_ladder['mode'] == 'none']
R['roi_self_constraint'] = dict(Z_min=float(sub_none.Z.min()), f_at_min=float(sub_none.f[sub_none.Z.idxmin()]),
                                t_H0_along_ladder={int(r_.f): float(r_.t_H0) for _, r_ in sub_none.iterrows()},
                                mssi_FV_H0_along_ladder={int(r_.f): float(r_.mssi_FV_H0) for _, r_ in sub_none.iterrows()})
say('ROI self-constraint:', R['roi_self_constraint'])
add('LZ model: 0.0049 +- 100 % Gaussian (P052 baseline)', pick(df_prior, 'gauss100'), sideband_pvalues(1.0)['p_six_wall_bins'], 'P052 3.11 asym / 3.03 toy; LZ 3.4')
add('MSSI removed (f -> 0)', row('nomssi', nomssi), None, 'upper bound from the MSSI channel')
for pl in ('LN0.7', 'LN1.5', 'LN2.3', 'flat'):
    add(f'prior {pl}, LZ-style (no sideband terms)', pick(df_prior, pl), None, 'profiled t_MSSI')
    add(f'prior {pl} + six wall sideband bins', pick(df_prior, f'{pl} + sidebands'), None, 'sideband-anchored')
for f in (2, 5, 10, 50):
    add(f'rate x{f}, common (sidebands scale too)', pick(df_ladder, f'x{f} common'), sideband_pvalues(f)['p_six_wall_bins'], 'P004 geometry error')
    add(f'rate x{f}, FV-only (sidebands unchanged)', pick(df_ladder, f'x{f} FV-only'), sideband_pvalues(1.0)['p_six_wall_bins'], 'shape/extrapolation error')
for lam in (4.3, 1.23):
    add(f'wall depth scale {lam} cm, annulus-anchored, position-blind', pick(df_lam, f'lambda={lam} cm position-blind'), sideband_pvalues(1.0)['p_six_wall_bins'], f'f_lambda = {f_lambda(lam):.2f}')
    add(f'wall depth scale {lam} cm, annulus-anchored, position-aware', pick(df_lam, f'lambda={lam} cm position-aware'), sideband_pvalues(1.0)['p_six_wall_bins'], f'LR_pos = {LR_pos(lam):.3g}')
add('d-shape: MSSI on the NR band (N(0,1))', pick(df_shape, 'd: 100 % N(0,1) (MSSI on the NR band)'), None, 'g = %.1f' % pick(df_shape, 'd: 100 % N(0,1) (MSSI on the NR band)')['g_event_cell'])
add('S1c: 100 % of MSSI at S1c>500', pick(df_shape, 'S1c: 100 % at S1c>500'), None, 'g = %.1f' % pick(df_shape, 'S1c: 100 % at S1c>500')['g_event_cell'])
add('all MSSI in the neighbourhood (g_max)', pick(df_shape, 'S1c 100 % >500 and d flat |d|<2 (all MSSI in the neighbourhood)'), None, 'g = %.1f' % g_max)
add('RFR MSSI x100 (HE-SB RFR 18 obs vs 2150)', pick(df_rfr, 'k_RFR=100'), pick(df_rfr, 'k_RFR=100')['p_HESB_RFR_sci'], 'RFR route')
df_sum = pd.DataFrame(summ); df_sum.to_csv(os.path.join(OUT, 'P090_summary_table.csv'), index=False)
say('\nSUMMARY\n' + df_sum[['hypothesis', 'Z_local', 'Z_global', 'mssi_FV_H0', 'sideband_p']].to_string())

# ----------------------------------------------------------------------------------------------
# 8. Figures
# ----------------------------------------------------------------------------------------------
C1, C2, C3, C4, INK, MUTED = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#0b0b0b', '#898781'
def clean(ax):
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.grid(True, color='#e8e7e3', lw=0.6); ax.set_axisbelow(True)
# Fig 1: Z_local and Z_global vs the MSSI rate factor, and the sideband p-value
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), dpi=160)
ax = axs[0]
for mode, col, lab in (('none', C1, 'LZ-style: 100 % prior only'), ('FV-only', C2, 'FV-only x f, sidebands at k'), ('common', C3, 'common x f (sidebands scale too)')):
    sub = df_ladder[df_ladder['mode'] == mode].sort_values('f')
    ax.plot(sub.f, sub.Z, '-o', color=col, ms=5, lw=2, label=lab)
    ax.plot(sub.f, sub.Z_global, '--', color=col, lw=1.4)
ax.axhline(3.4, color=MUTED, lw=1, ls=':'); ax.text(1.05, 3.45, 'LZ local 3.4', fontsize=7.5, color=MUTED)
ax.axhline(2.6, color=MUTED, lw=1, ls=':'); ax.text(1.05, 2.65, 'LZ global 2.6', fontsize=7.5, color=MUTED)
for zt in (3.0, 2.0, 1.0):
    fz = R['f_required']['FV-only'][f'Z={zt}']
    if np.isfinite(fz): ax.axvline(fz, color=C2, lw=0.8, ls='-.', alpha=0.6); ax.text(fz * 1.05, 0.15 + 0.25 * zt, f'x{fz:.0f}', fontsize=7.5, color=C2)
ax.set_xscale('log'); ax.set_xlabel('MSSI rate factor f (FV expectation = f x 0.0049 events)'); ax.set_ylabel('significance [sigma]  (solid local, dashed global)')
ax.set_ylim(0, 4); ax.legend(fontsize=7.5, frameon=False, loc='lower left'); clean(ax); ax.set_title('L10 (1 TeV), P052 (S1c, d) likelihood', fontsize=9)
ax = axs[1]
sub = df_sb[df_sb.f.isin([1, 1.64, 2, 5, 10, 20, 50, 100, 200, 500])]
ax.plot(sub.f, sub.p_six_wall_bins, '-o', color=C3, ms=5, lw=2, label='P(N <= 2 | 3.23 f), six wall bins')
ax.plot(sub.f, sub.p_annulus_sci, '-s', color=C4, ms=5, lw=2, label='P(0 | 0.03 f), 5.4 t annulus, science')
ax.plot(sub.f, sub.p_HESB_wall_sci, '-^', color=C1, ms=5, lw=2, label='P(0 | 0.6 f), HE SB wall, science')
ax.axhline(0.05, color=MUTED, lw=1, ls=':'); ax.text(1.05, 0.06, '5 %', fontsize=7.5, color=MUTED)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e-12, 2); ax.set_xlabel('common wall-MSSI rate factor f'); ax.set_ylabel('Poisson p-value of LZ\'s sideband counts')
ax.legend(fontsize=7.5, frameon=False, loc='lower left'); clean(ax); ax.set_title('what the same factor predicts in the calibration sidebands', fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P090_fig1_rate_ladder.png')); plt.close(fig)

# Fig 2: shape systematics -- depth scale (blind vs aware) and d/S1c re-shaping
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), dpi=160)
ax = axs[0]
for pos, col, lab in (('blind', C1, 'position-blind (LZ likelihood)'), ('aware', C2, 'position-aware (P070-style)')):
    sub = df_lam[df_lam.position == pos].sort_values('lam'); ax.plot(sub.lam, sub.Z, '-o', color=col, ms=5, lw=2, label=lab)
for _, r_ in df_lam[df_lam.position == 'blind'].iterrows():
    ax.annotate(f'x{r_.f_lambda:.1f}', (r_.lam, r_.Z), xytext=(0, -11), textcoords='offset points', fontsize=7, ha='center', color='#52514e')
for _, r_ in df_lam[df_lam.position == 'aware'].iterrows():
    ax.annotate(f'LR {r_.LR_pos:.3g}' if r_.LR_pos < 1e3 else f'LR {r_.LR_pos:.0e}', (r_.lam, r_.Z), xytext=(0, 7), textcoords='offset points', fontsize=7, ha='center', color='#52514e')
ax.text(4.6, 1.15, 'labels below: annulus-anchored FV rate factor f_lambda\nlabels above: position likelihood ratio at the event', fontsize=7, color='#52514e')
ax.axvline(1.23, color=MUTED, lw=0.8, ls='--'); ax.text(1.28, 0.3, 'LZ table (P079)', fontsize=7.5, color=MUTED, rotation=90)
ax.axvline(4.3, color=MUTED, lw=0.8, ls='--'); ax.text(4.4, 0.3, 'P033 MC', fontsize=7.5, color=MUTED, rotation=90)
ax.set_xlabel('wall-MSSI depth scale lambda [cm]'); ax.set_ylabel('Z_local [sigma]'); ax.set_ylim(0, 4); ax.legend(fontsize=7.5, frameon=False, loc='lower left'); clean(ax)
ax.set_title('depth scale: FV rate anchored to the 5.4 t annulus', fontsize=9)
ax = axs[1]
ax.plot(df_fg.fg, df_fg.Z_shape_only, '-', color=C1, lw=2, label='Z_local vs MSSI at the event cell (x g, total fixed)')
ax.plot(df_fg.fg, df_fg.Z_rate_only_noSB, '--', color=C2, lw=1.6, label='same via rate x f (no sideband terms)')
for _, r_ in df_shape.iterrows():
    ax.plot(r_.g_event_cell, r_.Z, 'o', color=C3, ms=6, mec='white')
ax.annotate('P004 d-shape\n(baseline)', (1, df_shape.Z.iloc[0]), xytext=(1.15, 3.55), fontsize=7)
ax.annotate('MSSI on the NR band', (pick(df_shape, 'd: 100 % N(0,1) (MSSI on the NR band)').g_event_cell, pick(df_shape, 'd: 100 % N(0,1) (MSSI on the NR band)').Z), xytext=(3.5, 3.3), fontsize=7)
ax.annotate('all MSSI at S1c>500', (pick(df_shape, 'S1c: 100 % at S1c>500').g_event_cell, pick(df_shape, 'S1c: 100 % at S1c>500').Z), xytext=(11, 2.9), fontsize=7)
ax.annotate('all MSSI in the\nneighbourhood (g_max)', (g_max, pick(df_shape, 'S1c 100 % >500 and d flat |d|<2 (all MSSI in the neighbourhood)').Z), xytext=(45, 2.3), fontsize=7)
ax.axvline(g_max, color=MUTED, lw=0.8, ls='--')
ax.set_xscale('log'); ax.set_xlabel('MSSI expectation at the event cell / LZ model  (f x g)'); ax.set_ylabel('Z_local [sigma]'); ax.set_ylim(0, 4)
ax.legend(fontsize=7.5, frameon=False, loc='lower left'); clean(ax); ax.set_title('shape deformations inside the ROI (green: explicit maps)', fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P090_fig2_shape.png')); plt.close(fig)

# Fig 3: the (f, g) plane: Z contours and the sideband exclusion
fgrid = np.logspace(0, 3, 61); ggrid = np.logspace(0, np.log10(60), 41)
def Zapprox(f, g): return float(np.interp(np.log(f * g), np.log(df_fg.fg), df_fg.Z_shape_only))
ZZ = np.array([[Zapprox(f, g) for f in fgrid] for g in ggrid])
fig, ax = plt.subplots(figsize=(5.6, 4.4), dpi=160)
cs = ax.contourf(fgrid, ggrid, ZZ, levels=[0, 1, 2, 3, 3.5], colors=['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5'], alpha=0.9)
cl = ax.contour(fgrid, ggrid, ZZ, levels=[1, 2, 3], colors=INK, linewidths=1); ax.clabel(cl, fmt='Z=%d', fontsize=7.5)
ax.axvspan(1.64, 1e3, color='#f0efec', alpha=0.0)
ax.axvline(1.64, color=C2, lw=1.5, ls='--'); ax.text(1.7, 1.08, 'k < 1.64 (95 %, six wall bins; P004)', color=C2, fontsize=7.5, rotation=90, va='bottom')
ax.axvline(f_lambda(4.3), color=C3, lw=1.5, ls='-.'); ax.text(f_lambda(4.3) * 1.05, 1.08, 'lambda = 4.3 cm, annulus-anchored', color=C3, fontsize=7.5, rotation=90, va='bottom')
ax.axhline(g_max, color=MUTED, lw=1, ls=':'); ax.text(1.05, g_max * 1.05, 'g_max: all MSSI in the neighbourhood', fontsize=7.5, color=MUTED)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('FV rate factor f'); ax.set_ylabel('shape concentration g at the event cell')
ax.set_title('Z_local(f, g) for L10 (1 TeV); shaded darker = higher Z', fontsize=9); clean(ax); ax.grid(False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P090_fig3_fg_plane.png')); plt.close(fig)

R['runtime_s'] = time.time() - T0
json.dump(R, open(os.path.join(OUT, 'P090_results.json'), 'w'), indent=1, default=float)
say(f'done in {R["runtime_s"]:.0f} s')
