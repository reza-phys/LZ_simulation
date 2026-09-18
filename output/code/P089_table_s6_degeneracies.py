"""
P089_table_s6_degeneracies.py -- Lagrangian-level (Table S6/S7) degeneracies of the LZ 248 keV event.

Run from the simulation root with .venv/bin/python (WimPyDD needs cwd = root):
    .venv/bin/python output/code/P089_table_s6_degeneracies.py          # compute (cached, <= 420 s per call) + analyse
Per-spectrum caches live in output/work/P089/cache/ (5 keV true-energy grid); P057's cache
(output/work/P057/cache/m1000_*.npy, 2.5 keV grid, identical halo/efficiency code) is sub-sampled where available.

What the script does
  A. Encodes the nonrelativistic reduction of the 20 Anand-Fitzpatrick-Haxton Lagrangians L1-L20 (4 scalar
     bilinear products + the 4x4 grid {V,T,A,T-gamma5}_chi x {V,T,A,T-gamma5}_N; structure recalled/likely, reductions
     derived in details.md sec. 2, heavy-DM limit, q^0 = 0, m_M = m_N), as WimPyDD Hamiltonians with q-dependent
     Wilson coefficients built as closures (P031 pitfall), for isoscalar and isovector couplings at 1 TeV.
  B. Computes natural-xenon spectra (WimPyDD 2.0.4 shell-model responses, Sun-frame Baxter-2021 halo) for the
     40 L_i^{s,v}, the 28 O_i^{s,v} (i = 1,3..15), sign variants of interfering Lagrangians, and check spectra.
  C. Folds LZ's efficiency (P003 model, 600 phd edge) and resolution sigma_E = 11 sqrt(E/248) keV (P009/P021), and
     computes pairwise Hellinger / KL distances of the observed-energy pdfs; average-linkage clustering at several
     thresholds; Gaussian N_3sigma (P057 convention) for same-class pairs.
  D. Coupling-ratio table: analytic Anand-level factors for exact pairs; rate-matched (200-270 keV) ratios for
     effective degeneracies; numerical check of the closures on L3/L8/L20 vs O11/O10/O6.
  E. Table S6 pattern: N_lo, R_hi at unit coupling, P016-calibrated Z(N_lo), residuals vs LZ at 1000 GeV, noise floor
     from LZ's own identical-spectrum pairs.
Outputs: output/work/P089/*.csv|json, figures in output/work/P089/figures/.
"""
import os, sys, json, math, time, itertools
import numpy as np, pandas as pd
sys.path.insert(0, 'output/code')
from common import lzcommon as lz
from scipy import special, stats
from scipy.cluster import hierarchy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'output/work/P089'; FIG = OUT + '/figures'; CACHE = OUT + '/cache'
for d in (OUT, FIG, CACHE):
    os.makedirs(d, exist_ok=True)
LOG = open(OUT + '/P089_run.log', 'a')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.write(s + '\n'); LOG.flush()
say(f'\n===== run {time.strftime("%Y-%m-%d %H:%M:%S")} =====')

MV2 = lz.M_V_GEV ** 2                 # m_v^2 (GeV^2); LZ/Anand unit coupling 1/m_v^2 = WimPyDD c^0 = 2/m_v^2 (P003)
MN = lz.M_NUCLEON_GEV                 # 0.938 GeV; also m_M (Anand's tensor mass scale, taken = m_N; recalled/likely)
MCHI = 1000.0
C_UNIT = 2.0 / MV2                    # WimPyDD c^0 (or c^1) for an LZ unit coupling d_i = 1/m_v^2
E_TRUE = np.arange(1.25, 600.0, 5.0)  # keV, 120 points = every second point of P057's 2.5 keV grid
E_OBS = np.arange(0.5, 700.0, 1.0)
SIGMA_E0 = 11.0
E50_HI, SIG_HI = 269.9, 8.0           # LZ 600 phd edge (P003 model)
FLOOR_W = 1e-2
BUDGET_S = 420.0; T_START = time.time()

# ------------------------------------------------------------------------------------------------
# Part A: Hamiltonians.  Coefficient closures WITHOUT default arguments (WimPyDD treats defaults as shared
# named parameters -- P031).  q-dependence is transferred by WimPyDD to the operator (q in GeV).
# ------------------------------------------------------------------------------------------------
WD = None; HALO = None
def wd_init():
    global WD, HALO
    if WD is None:
        WD = lz.wd(); HALO = lz.wd_halo()      # Sun-frame Baxter-2021 SHM with explicit v_min grid (P002 fix; P035 label)

def cfn(c, iso, qpow=0):
    """WimPyDD coefficient function for Anand-level coefficient c x (q/m_N)^qpow, isoscalar or isovector."""
    # Anand-level coefficient c (in units of the LZ unit coupling 1/m_v^2) -> WimPyDD c^tau = 2 c / m_v^2 (P003)
    c0 = C_UNIT * float(c) if iso == 's' else 0.0
    c1 = C_UNIT * float(c) if iso == 'v' else 0.0
    if qpow == 0:
        def f():
            return [c0, c1]
    else:
        p = float(qpow)
        def f(q):
            fac = (q / MN) ** p
            return [c0 * fac, c1 * fac]
    return f

R = MN / MCHI       # m_N / m_chi at 1 TeV
# Anand-level NR reduction of each Lagrangian at unit coupling d_i = 1/m_v^2: {operator: [(factor, q_power), ...]}
# (factor multiplies (q/m_N)^q_power; m_M = m_N).  'reliab' = reliability of the identification (see details.md).
LAG = {
 1:  dict(nr={1: [(1.0, 0)]},                                    txt='O1',                                    resp='M',                       reliab='certain'),
 2:  dict(nr={10: [(1.0, 0)]},                                   txt='O10',                                   resp='q2 Sigma\'\'',            reliab='certain'),
 3:  dict(nr={11: [(-R, 0)]},                                    txt='-(mN/mchi) O11',                        resp='q2 M',                    reliab='certain'),
 4:  dict(nr={6: [(-R, 0)]},                                     txt='-(mN/mchi) O6',                         resp='q4 Sigma\'\'',            reliab='certain'),
 5:  dict(nr={1: [(1.0, 0)]},                                    txt='O1',                                    resp='M',                       reliab='certain'),
 6:  dict(nr={1: [(0.5, 2)], 3: [(2.0, 0)]},                     txt='(q2/2mN2) O1 + 2 O3',                    resp='q4 M + q2 Phi\'\' + q4 Phi\'\'M + q2v2 Sigma\'', reliab='likely (sign of O1-O3 cross term uncertain)'),
 7:  dict(nr={7: [(-2.0, 0)], 9: [(2.0 * R, 0)]},                txt='-2 O7 + 2(mN/mchi) O9',                 resp='v2 Sigma\' + (mN/mchi)2 q2 Sigma\'', reliab='likely'),
 8:  dict(nr={10: [(-2.0, 0)]},                                  txt='-2 O10',                                resp='q2 Sigma\'\'',            reliab='likely (LZ: L2/L8 differ by a constant)'),
 9:  dict(nr={1: [(0.5 * R, 2)], 5: [(-2.0, 0)], 4: [(-2.0, 2)], 6: [(2.0, 0)]}, txt='(q2/2 mchi mN) O1 - 2 O5 - 2[(q2/mN2) O4 - O6]', resp='(mN/mchi)2 q4 M + q2v2 (M+Delta) + q4 Sigma\'', reliab='likely (sign of O5 vs spin-spin term uncertain)'),
 10: dict(nr={4: [(4.0, 2)], 6: [(-4.0, 0)]},                    txt='4[(q2/mN2) O4 - O6]',                   resp='q4 Sigma\'',              reliab='certain (P003/P012 vs LZ Fig. 1)'),
 11: dict(nr={9: [(-4.0, 0)]},                                   txt='-4 O9',                                 resp='q2 Sigma\'',              reliab='likely'),
 12: dict(nr={12: [(-4.0, 2)], 15: [(-4.0, 0)], 10: [(-R, 2)]},  txt='-4[(q2/mN2) O12 + O15] - (mN/mchi)(q2/mN2) O10', resp='q4 v2 Sigma\'\' + q4 Phi~\' (Sigma\', Phi\'\' cancel) + (mN/mchi)2 q4 Sigma\'\'', reliab='likely (derived; relative O12/O15 sign fixed by a vector identity)'),
 13: dict(nr={8: [(2.0, 0)], 9: [(-2.0, 0)]},                    txt='2 O8 - 2 O9',                           resp='v2 (M+Delta) + q2 Sigma\' - Delta Sigma\' cross', reliab='likely (sign of O8-O9 cross term uncertain)'),
 14: dict(nr={9: [(4.0, 0)]},                                    txt='4 O9',                                  resp='q2 Sigma\'',              reliab='likely (LZ: L11/L14 differ by a constant)'),
 15: dict(nr={4: [(-4.0, 0)]},                                   txt='-4 O4',                                 resp='Sigma\' + Sigma\'\'',     reliab='certain'),
 16: dict(nr={13: [(-4.0, 0)]},                                  txt='-4 O13',                                resp='q2 v2 Sigma\'\' + q2 Phi~\'', reliab='likely (derived, q^0 dropped)'),
 17: dict(nr={11: [(-2.0, 0)]},                                  txt='-2 O11',                                resp='q2 M',                    reliab='likely (LZ: L3/L17 differ by a constant)'),
 18: dict(nr={11: [(-1.0, 2)], 15: [(4.0, 0)]},                  txt='-(q2/mN2) O11 + 4 O15',                 resp='q6 M + q4 Phi\'\' + q6 Phi\'\'M + q4 v2 Sigma\'', reliab='likely (sign of O11-O15 cross term uncertain)'),
 19: dict(nr={14: [(4.0, 0)]},                                   txt='4 O14',                                 resp='q2 v2 Sigma\'',           reliab='likely'),
 20: dict(nr={6: [(4.0, 0)]},                                    txt='4 O6',                                  resp='q4 Sigma\'\'',            reliab='likely (LZ: L4/L20 differ by a constant)'),
}
# operators whose relative sign to the rest of the Lagrangian is convention-dependent -> both signs computed
SIGN_FLIP = {6: 3, 9: 5, 13: 9, 18: 15, 12: 15}

def wc_from_nr(nr, iso, flip_op=None):
    wc = {}
    for op, terms in nr.items():
        assert len(terms) == 1
        fac, qp = terms[0]
        if op == flip_op:
            fac = -fac
        wc[op] = cfn(fac, iso, qp)
    return wc

MODELS = {}   # name -> (wc builder, note)
for i, L in LAG.items():
    for iso in 'sv':
        MODELS[f'L{i}{iso}'] = (lambda i=i, iso=iso: wc_from_nr(LAG[i]['nr'], iso), f'L{i} {iso}')
        if i in SIGN_FLIP:
            MODELS[f'L{i}{iso}_flip'] = (lambda i=i, iso=iso: wc_from_nr(LAG[i]['nr'], iso, SIGN_FLIP[i]), f'L{i} {iso}, sign of O{SIGN_FLIP[i]} flipped')
for op in [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
    for iso in 'sv':
        MODELS[f'O{op}{iso}'] = (lambda op=op, iso=iso: {op: cfn(1.0, iso, 0)}, f'O{op} {iso}')
# L10 including the (q^4/4 mchi mN mM^2) O1 term dropped in the leading reduction (check of its size)
MODELS['L10s_full'] = (lambda: {4: cfn(4.0, 's', 2), 6: cfn(-4.0, 's', 0), 1: cfn(0.25 * R, 's', 4)}, 'L10 s with q^4 O1 term')

# single-operator L entries are exact multiples of O entries: reuse the O spectrum times factor^2
ALIAS = {}
for i, L in LAG.items():
    if len(L['nr']) == 1:
        op, [(fac, qp)] = list(L['nr'].items())[0]
        if qp == 0:
            for iso in 'sv':
                ALIAS[f'L{i}{iso}'] = (f'O{op}{iso}', fac ** 2)
# ... except three that we compute explicitly as a check of the closure/normalisation bookkeeping
CHECK_EXPLICIT = ['L3s', 'L8s', 'L20s']
for k in CHECK_EXPLICIT:
    ALIAS.pop(k)

class Budget(Exception):
    pass

def cached_rate(key, wc_builder):
    fn = f'{CACHE}/{key}.npy'
    if os.path.exists(fn):
        return np.load(fn)
    p57 = f'output/work/P057/cache/m1000_{key}.npy'
    if os.path.exists(p57):
        r57 = np.load(p57)                          # P057 grid: 1.25 + 2.5 k, 240 points -> every second point
        if r57.size == 240:
            r = r57[::2]; np.save(fn, r); say(f'  {key}: sub-sampled from P057 cache'); return r
    if time.time() - T_START > BUDGET_S:
        raise Budget(key)
    wd_init(); t = time.time()
    h = WD.eft_hamiltonian('P089_' + key, wc_builder())
    r = np.clip(np.asarray(lz.wd_rate(h, MCHI, E_TRUE, halo=HALO), float), 0.0, None)
    np.save(fn, r)
    say(f'  computed {key}: {time.time()-t:.0f} s; pairs {getattr(h, "coeff_squared_list", "?")}; R(200-270)={np.trapezoid(r[(E_TRUE>200)&(E_TRUE<270)], E_TRUE[(E_TRUE>200)&(E_TRUE<270)]):.3e} /t/yr')
    return r

P057_ALIAS = {'L10s': 'L10', 'L10s_p57': 'L10'}   # P057 cached L10 (same reduction, isoscalar)

def compute_all():
    SP = {}
    order = [k for k in MODELS if k not in ALIAS]
    for name in order:
        key = name
        if name == 'L10s' and os.path.exists('output/work/P057/cache/m1000_L10.npy') and not os.path.exists(f'{CACHE}/L10s.npy'):
            r = np.load('output/work/P057/cache/m1000_L10.npy')[::2]; np.save(f'{CACHE}/L10s.npy', r); say('  L10s: sub-sampled from P057 cache (m1000_L10)')
        SP[name] = cached_rate(key, MODELS[name][0])
    for name, (base, fac2) in ALIAS.items():
        SP[name] = SP[base] * fac2
    return SP

# ------------------------------------------------------------------------------------------------
# Part B: observed-energy pdfs and metrics (P057 conventions)
# ------------------------------------------------------------------------------------------------
def efficiency(E, eff0=0.96, sig_lo=3.4, e50=E50_HI, sig_hi=SIG_HI):
    E = np.asarray(E, float)
    return eff0 * 0.5 * (1 + special.erf((E - 5.4) / (math.sqrt(2) * sig_lo))) * 0.5 * (1 - special.erf((E - e50) / (math.sqrt(2) * sig_hi)))

def smear_matrix(scale=1.0):
    sig = scale * SIGMA_E0 * np.sqrt(E_TRUE / 248.0)
    K = stats.norm.pdf(E_OBS[:, None], loc=E_TRUE[None, :], scale=sig[None, :])
    return K * (E_OBS[1] - E_OBS[0])
KSM = smear_matrix(); KSM15 = smear_matrix(1.5)
DE_T = E_TRUE[1] - E_TRUE[0]

def pdf_obs(r, K=None):
    K = KSM if K is None else K
    acc = r * efficiency(E_TRUE) * DE_T
    p = K @ acc; tot = p.sum()
    return (p / tot if tot > 0 else np.zeros_like(p)), tot

def pdf_true(r):
    acc = r * efficiency(E_TRUE); tot = acc.sum()
    return acc / tot if tot > 0 else acc

def window(r, e1, e2):
    m = (E_TRUE >= e1) & (E_TRUE <= e2)
    return float(np.trapezoid(r[m] * efficiency(E_TRUE[m]), E_TRUE[m]))

def hellinger(p, q):
    return float(math.sqrt(max(0.0, 1 - np.sum(np.sqrt(p * q)))))

def kl(p, q, w=FLOOR_W):
    pf = (1 - w) * p + w / p.size; qf = (1 - w) * q + w / q.size
    lam = np.log(pf / qf); D = float(np.sum(pf * lam)); V = float(np.sum(pf * lam ** 2) - D ** 2)
    return D, V

def n3_gauss(p, q):
    Dpq, Vp = kl(p, q); Dqp, Vq = kl(q, p)
    if Dpq + Dqp <= 1e-15:
        return np.inf, 0.0
    return max(9 * Vq / (Dpq + Dqp) ** 2, 9 * Vp / (Dpq + Dqp) ** 2), Dpq + Dqp

# P016-calibrated Z(N_lo) at 1000 GeV (baseline curve file; falls back to the tabulated points used by P031/P044)
def load_p016():
    try:
        c = pd.read_csv('output/work/P016/P016_Z_vs_Nlo_curves.csv')
        return np.log10(c.iloc[:, 0].values), c.iloc[:, 1].values
    except Exception:
        return np.log10([0.20, 0.43, 3.1, 10.9, 28.0, 160.0, 258.0, 477.0, 937.0, 2752.0]), np.array([3.40, 3.26, 3.08, 3.00, 2.70, 2.00, 1.90, 1.65, 1.00, 0.48])
P016_LN, P016_Z = load_p016()
def z_pred(nlo):
    if not np.isfinite(nlo) or nlo <= 0:
        return float(P016_Z[0])
    return float(np.interp(math.log10(nlo), P016_LN, P016_Z, left=P016_Z[0], right=0.0))

# ------------------------------------------------------------------------------------------------
def main():
    try:
        SP = compute_all()
    except Budget as b:
        say(f'time budget reached before {b}; {len(os.listdir(CACHE))} spectra cached -- re-run to continue'); return
    say(f'{len(SP)} spectra available ({len(os.listdir(CACHE))} cached files)')
    RES = dict(settings=dict(E_true_grid_keV=[float(E_TRUE[0]), float(E_TRUE[-1]), float(DE_T)], E_obs_bin_keV=1.0, sigma_E='11 sqrt(E/248) keV',
                             efficiency='0.96 x erf(5.4, 3.4) x erf(269.9, 8.0)', halo='Sun-frame Baxter-2021 (lz.wd_halo())', m_chi_GeV=MCHI,
                             coupling='LZ unit d_i = c_i = 1/m_v^2 (Anand) = WimPyDD c^tau = 2/m_v^2 (P003)', KL_floor=FLOOR_W))

    # ---------- closure / bookkeeping checks ----------
    chk = []
    for lname, oname, fac in [('L3s', 'O11s', -R), ('L8s', 'O10s', -2.0), ('L20s', 'O6s', 4.0)]:
        ratio = SP[lname] / np.where(SP[oname] > 0, SP[oname], np.nan)
        m = (E_TRUE > 10) & (E_TRUE < 400)
        chk.append(dict(pair=f'{lname}/{oname}', predicted_rate_ratio=fac ** 2, wimpydd_ratio_median=float(np.nanmedian(ratio[m])),
                        rel_scatter=float(np.nanstd(ratio[m]) / np.nanmedian(ratio[m]))))
    # L10 with the q^4 O1 term
    m = (E_TRUE > 10) & (E_TRUE < 400)
    rr = SP['L10s_full'] / SP['L10s']
    chk.append(dict(pair='L10s_full/L10s', predicted_rate_ratio=1.0, wimpydd_ratio_median=float(np.median(rr[m])), rel_scatter=float(np.max(np.abs(rr[m] - 1))),
                    note='max |deviation| over 10-400 keV: size of the dropped (q^4/4 mchi mN mM^2) O1 term at 1 TeV'))
    # L12: cancellation of Sigma'/Phi'' -- compare with the sign-flipped version and with O6
    for iso in 'sv':
        a, b, o6 = SP[f'L12{iso}'], SP[f'L12{iso}_flip'], SP[f'O6{iso}']
        chk.append(dict(pair=f'L12{iso}_flip/L12{iso}', predicted_rate_ratio='>>1 if Sigma-prime/Phi-doubleprime cancel in the derived sign', wimpydd_ratio_median=float(np.median((b / a)[m])),
                        rel_scatter=float(np.std(np.log10((b / a)[m])))))
        lr = np.log10((a / o6)[m]); sl = np.polyfit(np.log10(E_TRUE[m]), lr, 1)[0]
        chk.append(dict(pair=f'L12{iso}/O6{iso}', predicted_rate_ratio='slowly varying (<v_perp^2> + Phi~/Sigma\'\')', wimpydd_ratio_median=float(10 ** np.median(lr)), rel_scatter=float(np.std(lr - np.polyval(np.polyfit(np.log10(E_TRUE[m]), lr, 1), np.log10(E_TRUE[m])))), note=f'log-log slope {sl:.3f}'))
    for iso in 'sv':   # L9: transverse cancellation check -- Sigma'' absent => compare L9 minus its O5/O1 part with L10 shape
        lr = np.log10((SP[f'L9{iso}'] / SP[f'L10{iso}'])[m]); sl = np.polyfit(np.log10(E_TRUE[m]), lr, 1)[0]
        chk.append(dict(pair=f'L9{iso}/L10{iso}', predicted_rate_ratio='1/4 + O5 admixture', wimpydd_ratio_median=float(10 ** np.median(lr)), rel_scatter=float(np.std(lr)), note=f'log-log slope {sl:.3f}'))
    CHK = pd.DataFrame(chk); CHK.to_csv(OUT + '/P089_closure_checks.csv', index=False); say('\nClosure checks:\n' + CHK.to_string(index=False))

    # ---------- observed pdfs, shape metrics ----------
    names_L = [f'L{i}{iso}' for i in range(1, 21) for iso in 'sv']
    names_O = [f'O{op}{iso}' for op in [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] for iso in 'sv']
    names_flip = [k for k in SP if k.endswith('_flip')]
    ALLN = names_L + names_O
    PDF, PDFT, PDF15, MET = {}, {}, {}, {}
    for n in ALLN + names_flip + ['L10s_full']:
        r = SP[n]; PDF[n], acc = pdf_obs(r); PDFT[n] = pdf_true(r); PDF15[n], _ = pdf_obs(r, KSM15)
        R_lo, R_hi, R_mid = window(r, 5.4, 55), window(r, 200, 270), window(r, 55, 200)
        pk = float(E_TRUE[np.argmax(r * efficiency(E_TRUE) * (E_TRUE > 15))])
        cs = np.cumsum(PDF[n]); med = float(E_OBS[np.searchsorted(cs, 0.5)])
        MET[n] = dict(N_lo=R_lo / R_hi if R_hi > 0 else np.inf, N_mid=R_mid / R_hi if R_hi > 0 else np.inf, R_hi_unit=R_hi, R_lo_unit=R_lo,
                      acc_unit=acc, N_2p84=acc * 2.84, peak_true_keV=pk, obs_median_keV=med, p248_per_keV=float(PDF[n][np.searchsorted(E_OBS, 248.0)]),
                      Z_pred_P016=z_pred(R_lo / R_hi if R_hi > 0 else np.inf))

    # ---------- pairwise distances and clustering ----------
    N = len(ALLN); H = np.zeros((N, N)); KLs = np.zeros((N, N))
    for i, j in itertools.combinations(range(N), 2):
        H[i, j] = H[j, i] = hellinger(PDF[ALLN[i]], PDF[ALLN[j]])
        KLs[i, j] = KLs[j, i] = kl(PDF[ALLN[i]], PDF[ALLN[j]])[0] + kl(PDF[ALLN[j]], PDF[ALLN[i]])[0]
    pd.DataFrame(H, index=ALLN, columns=ALLN).to_csv(OUT + '/P089_hellinger_matrix.csv', float_format='%.5f')
    pd.DataFrame(KLs, index=ALLN, columns=ALLN).to_csv(OUT + '/P089_symKL_matrix.csv', float_format='%.5g')
    Z = hierarchy.linkage(H[np.triu_indices(N, 1)], method='average')
    THR = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20]
    CL = {}
    for t in THR:
        CL[t] = hierarchy.fcluster(Z, t, criterion='distance')
    # robustness: true-energy pdf (no resolution) and 1.5x resolution, at H < 0.02 and 0.05
    def count_classes(P, thr, sub):
        n = len(sub); h = np.zeros((n, n))
        for i, j in itertools.combinations(range(n), 2):
            h[i, j] = h[j, i] = hellinger(P[sub[i]], P[sub[j]])
        z = hierarchy.linkage(h[np.triu_indices(n, 1)], method='average')
        return {t: int(hierarchy.fcluster(z, t, criterion='distance').max()) for t in thr}
    subsets = {'all_68': ALLN, 'TableS6_40L': names_L, 'S6_plus_elastic_S7_44': names_L + ['O1s', 'O1v', 'O4s', 'O4v'],
               'compatible_Z>=3.0_at_1TeV': [n for n in names_L if lz.LSIG[n][11] >= 3.0] }
    COUNTS = {}
    for sname, sub in subsets.items():
        COUNTS[sname] = dict(observed=count_classes(PDF, THR, sub), true_energy=count_classes(PDFT, THR, sub), resolution_x1p5=count_classes(PDF15, THR, sub))
    RES['class_counts'] = COUNTS
    say('\nNumber of spectral classes (average linkage, Hellinger threshold):')
    for sname, d in COUNTS.items():
        say(f'  {sname}: ' + '; '.join(f'{k}: ' + ','.join(f'H<{t}:{v}' for t, v in dd.items()) for k, dd in d.items()))

    # cluster membership table at the working threshold H < 0.02 (N_3sigma >~ 1500) and at 0.05, 0.15
    def members(t):
        lab = CL[t]; out = {}
        for c in sorted(set(lab)):
            mem = [ALLN[i] for i in range(N) if lab[i] == c]
            # representative: the member with the highest LZ Z at 1 TeV (L entries) or first
            def zz(n):
                return lz.LSIG[n][11] if n in lz.LSIG else (lz.OSIG[n][1000][0] if n in lz.OSIG else -1)
            rep = max(mem, key=lambda n: (zz(n), -len(n)))
            out[c] = dict(rep=rep, members=mem)
        return out
    MEMB = {t: members(t) for t in (0.02, 0.05, 0.15)}
    RES['clusters'] = {str(t): [dict(rep=v['rep'], members=v['members']) for v in MEMB[t].values()] for t in MEMB}
    say('\nClasses at H < 0.02:')
    for c, v in MEMB[0.02].items():
        say(f'  [{v["rep"]}] ' + ', '.join(v['members']))
    say('\nClasses at H < 0.05:')
    for c, v in MEMB[0.05].items():
        say(f'  [{v["rep"]}] ' + ', '.join(v['members']))

    # ---------- mapping table ----------
    rows = []
    lab02, lab05, lab15 = CL[0.02], CL[0.05], CL[0.15]
    idx = {n: i for i, n in enumerate(ALLN)}
    for n in ALLN:
        i = idx[n]
        d = dict(entry=n, kind='Lagrangian' if n.startswith('L') else 'operator', isospin=n[-1])
        if n.startswith('L'):
            k = int(n[1:-1]); L = LAG[k]
            d.update(NR_reduction=L['txt'], coefficients_Anand=json.dumps({f'c{op}': f'{fac:g} (q/mN)^{qp}' if qp else f'{fac:g}' for op, [(fac, qp)] in L['nr'].items()}),
                     responses=L['resp'], reliability=L['reliab'], Z_LZ_100=lz.LSIG[n][8], Z_LZ_200=lz.LSIG[n][9], Z_LZ_400=lz.LSIG[n][10], Z_LZ_1000=lz.LSIG[n][11], Z_LZ_4000=lz.LSIG[n][12])
        else:
            op = int(n[1:-1])
            OPRESP = {1: 'M', 3: 'q2 Phi\'\' + q2v2 Sigma\' + q2 Phi\'\'M', 4: 'Sigma\' + Sigma\'\'', 5: 'q2v2 (M + Delta)', 6: 'q4 Sigma\'\'', 7: 'v2 Sigma\'', 8: 'v2 (M + Delta)', 9: 'q2 Sigma\'',
                      10: 'q2 Sigma\'\'', 11: 'q2 M', 12: 'v2 (Sigma\' + Sigma\'\' + Phi\'\') + Phi~\'', 13: 'q2v2 Sigma\'\' + q2 Phi~\'', 14: 'q2v2 Sigma\'', 15: 'q4 v2 Sigma\' + q4 Phi\'\''}
            d.update(NR_reduction=f'O{op}', coefficients_Anand=json.dumps({f'c{op}': '1'}), responses=OPRESP[op], reliability='certain',
                     Z_LZ_1000=(lz.OSIG[n][1000][0] if n in lz.OSIG else np.nan), Z_LZ_400=(lz.OSIG[n][400][0] if n in lz.OSIG else np.nan), Z_LZ_4000=(lz.OSIG[n][4000][0] if n in lz.OSIG else np.nan))
        for t, lab in (('0.02', lab02), ('0.05', lab05), ('0.15', lab15)):
            c = lab[i]; rep = MEMB[float(t)][c]['rep']
            d[f'class_H{t}'] = int(c); d[f'class_rep_H{t}'] = rep
            if t == '0.02':
                d['H_to_rep'] = float(H[i, idx[rep]])
                d['N3sigma_gauss_vs_rep'] = (float(n3_gauss(PDF[n], PDF[rep])[0]) if rep != n else np.inf)
                # coupling ratio (Anand units) mapping the rep's coupling to this entry at equal 200-270 keV accepted rate
                d['coupling_ratio_entry_over_rep_at_equal_R200_270'] = float(math.sqrt(MET[rep]['R_hi_unit'] / MET[n]['R_hi_unit'])) if MET[n]['R_hi_unit'] > 0 else np.inf
        d.update({k: MET[n][k] for k in ('N_lo', 'N_mid', 'R_hi_unit', 'acc_unit', 'N_2p84', 'peak_true_keV', 'obs_median_keV', 'p248_per_keV', 'Z_pred_P016')})
        d['resid_Z_LZ_minus_pred_1000'] = (d['Z_LZ_1000'] - d['Z_pred_P016']) if np.isfinite(d.get('Z_LZ_1000', np.nan)) else np.nan
        rows.append(d)
    MAP = pd.DataFrame(rows); MAP.to_csv(OUT + '/P089_mapping_table.csv', index=False, float_format='%.5g')
    say('\nMapping table (selected columns):\n' + MAP[['entry', 'NR_reduction', 'responses', 'class_H0.02', 'class_rep_H0.02', 'H_to_rep', 'N_lo', 'N_2p84', 'Z_pred_P016', 'Z_LZ_1000', 'resid_Z_LZ_minus_pred_1000']].to_string(index=False, float_format=lambda x: f'{x:.3g}'))

    # ---------- exact pairs: analytic coupling ratios + LZ significance scatter (noise floor) ----------
    EX = []
    def zdiff(a, b):
        za, zb = np.array(lz.LSIG[a]), np.array(lz.LSIG[b]); dz = za - zb
        return dict(dZ_100_200_400_1000_4000=[float(x) for x in dz[8:]], max_abs_dZ=float(np.max(np.abs(dz))), rms_dZ_ge100=float(np.sqrt(np.mean(dz[8:] ** 2))))
    for (a, b, ratio_txt, ratio_val, op) in [('L1', 'L5', 'd1 = d5', 1.0, 'O1'), ('L2', 'L8', 'd2 = 2 d8', 2.0, 'O10'), ('L3', 'L17', 'd3 = 2 (mchi/mN) d17', 2.0 / R, 'O11'),
                                            ('L4', 'L20', 'd4 = 4 (mchi/mN) d20', 4.0 / R, 'O6'), ('L11', 'L14', 'd11 = d14 (opposite sign)', 1.0, 'O9')]:
        for iso in 'sv':
            EX.append(dict(pair=f'{a}{iso}/{b}{iso}', operator=op + iso, coupling_map=ratio_txt, ratio_d_a_over_d_b_1TeV=ratio_val, H_numerical=float(H[idx[a + iso], idx[b + iso]]), **zdiff(a + iso, b + iso)))
    # operator-level maps for single-operator Lagrangians
    OPMAP = [('L1/L5', 'c1 = d1 = d5'), ('L2', 'c10 = d2'), ('L8', 'c10 = -2 d8'), ('L3', 'c11 = -(mN/mchi) d3 = -9.38e-4 d3 at 1 TeV'), ('L17', 'c11 = -2 d17'),
             ('L4', 'c6 = -(mN/mchi) d4'), ('L20', 'c6 = 4 d20'), ('L11', 'c9 = -4 d11'), ('L14', 'c9 = 4 d14'), ('L15', 'c4 = -4 d15  (LZ Table S7 O4: (c4 mv^2)^2 = 16 (d15 mv^2)^2)'),
             ('L16', 'c13 = -4 d16'), ('L19', 'c14 = 4 d19'), ('L10', 'c4 = 4 (q^2/mN^2) d10, c6 = -4 d10'), ('L7', 'c7 = -2 d7, c9 = 2 (mN/mchi) d7'), ('L13', 'c8 = 2 d13, c9 = -2 d13'),
             ('L6', 'c1 = (q^2/2 mN^2) d6, c3 = 2 d6'), ('L9', 'c1 = (q^2/2 mchi mN) d9, c5 = -2 d9, c4 = -2 (q^2/mN^2) d9, c6 = 2 d9'), ('L12', 'c12 = -4 (q^2/mN^2) d12, c15 = -4 d12, c10 = -(mN/mchi)(q^2/mN^2) d12'),
             ('L18', 'c11 = -(q^2/mN^2) d18, c15 = 4 d18')]
    EXD = pd.DataFrame(EX); EXD.to_csv(OUT + '/P089_exact_pairs.csv', index=False)
    say('\nExact pairs (LZ statement) -- analytic coupling ratios and LZ Z scatter:\n' + EXD.to_string(index=False))
    allz = np.concatenate([np.array(e['dZ_100_200_400_1000_4000']) for e in EX])
    RES['noise_floor_identical_spectra'] = dict(n_differences=int(allz.size), rms=float(np.sqrt(np.mean(allz ** 2))), max_abs=float(np.max(np.abs(allz))),
                                                  note='differences of LZ local significances between entries LZ itself declares identical up to a constant (m >= 100 GeV)')
    RES['operator_coupling_maps'] = OPMAP
    # s/v pairs LZ declares indistinguishable
    SV = []
    for i in range(1, 21):
        a, b = f'L{i}s', f'L{i}v'
        SV.append(dict(L=i, LZ_declares_sv_indistinguishable=(i in (2, 4, 7, 8, 10, 11, 14, 15, 19, 20)), H_sv=float(H[idx[a], idx[b]]),
                       N3sigma_gauss=float(n3_gauss(PDF[a], PDF[b])[0]), same_class_H0p02=bool(lab02[idx[a]] == lab02[idx[b]]), same_class_H0p05=bool(lab05[idx[a]] == lab05[idx[b]]),
                       Z_LZ_1000_s=lz.LSIG[a][11], Z_LZ_1000_v=lz.LSIG[b][11], N_lo_s=MET[a]['N_lo'], N_lo_v=MET[b]['N_lo'],
                       rate_ratio_v_over_s_200_270=MET[b]['R_hi_unit'] / MET[a]['R_hi_unit'] if MET[a]['R_hi_unit'] > 0 else np.inf))
    SVD = pd.DataFrame(SV); SVD.to_csv(OUT + '/P089_isoscalar_isovector.csv', index=False, float_format='%.4g')
    say('\nIsoscalar vs isovector per Lagrangian:\n' + SVD.to_string(index=False, float_format=lambda x: f'{x:.3g}'))

    # ---------- sign-variant sensitivity ----------
    SG = []
    for n in names_flip:
        base = n.replace('_flip', '')
        SG.append(dict(model=base, flipped_operator=f'O{SIGN_FLIP[int(base[1:-1])]}', H_base_vs_flip=hellinger(PDF[base], PDF[n]), N_lo_base=MET[base]['N_lo'], N_lo_flip=MET[n]['N_lo'],
                       R_hi_flip_over_base=MET[n]['R_hi_unit'] / MET[base]['R_hi_unit'], Z_pred_base=MET[base]['Z_pred_P016'], Z_pred_flip=MET[n]['Z_pred_P016'], Z_LZ_1000=lz.LSIG[base][11]))
    SGD = pd.DataFrame(SG); SGD.to_csv(OUT + '/P089_sign_variants.csv', index=False, float_format='%.4g')
    say('\nSign variants of interfering Lagrangians:\n' + SGD.to_string(index=False, float_format=lambda x: f'{x:.3g}'))

    # ---------- Table S6 pattern summary ----------
    pat = MAP[MAP.kind == 'Lagrangian'][['entry', 'responses', 'N_lo', 'Z_pred_P016', 'Z_LZ_1000', 'resid_Z_LZ_minus_pred_1000']].copy()
    resid = pat.resid_Z_LZ_minus_pred_1000.values
    RES['tableS6_pattern'] = dict(rms_resid_all40=float(np.sqrt(np.mean(resid ** 2))), max_resid=float(np.max(np.abs(resid))), worst=[str(x) for x in pat.entry.values[np.argsort(-np.abs(resid))[:6]]],
                                  rms_resid_excluding_L16_L12=float(np.sqrt(np.mean(resid[~pat.entry.str.match(r'L1[26]')] ** 2))))
    say(f'\nTable S6 vs P016-curve prediction from N_lo: rms {RES["tableS6_pattern"]["rms_resid_all40"]:.2f} sigma, max {RES["tableS6_pattern"]["max_resid"]:.2f}; worst {RES["tableS6_pattern"]["worst"]}')

    # key ratios (spectral) for the paper
    def slope(a, b, lo=20, hi=270):
        # log-log slope of the true-energy ratio, 20-270 keV, excluding node regions (< 1e-3 of the maximum)
        m = (E_TRUE > lo) & (E_TRUE < hi) & (SP[a] > 1e-3 * SP[a].max()) & (SP[b] > 1e-3 * SP[b].max())
        lr = np.log10(SP[a][m] / SP[b][m]); x = np.log10(E_TRUE[m]); c = np.polyfit(x, lr, 1)
        return dict(slope=float(c[0]), rms_dex=float(np.std(lr - np.polyval(c, x))), ratio_248=float(10 ** np.interp(np.log10(248), x, lr)), H_obs=float(hellinger(PDF[a], PDF[b])))
    # two-variable proxy: Z_LZ(1 TeV) vs log N_lo and log p(248 keV) for the 40 Table S6 entries
    XN = np.array([math.log10(max(MET[n]['N_lo'], 1e-2)) for n in names_L]); XP = np.array([math.log10(MET[n]['p248_per_keV']) for n in names_L]); YZ = np.array([lz.LSIG[n][11] for n in names_L])
    A1 = np.vstack([np.ones_like(XN), XN]).T; c1 = np.linalg.lstsq(A1, YZ, rcond=None)[0]; r1 = YZ - A1 @ c1
    A2 = np.vstack([np.ones_like(XN), XN, XP]).T; c2 = np.linalg.lstsq(A2, YZ, rcond=None)[0]; r2 = YZ - A2 @ c2
    RES['Z_regression'] = dict(linear_in_logNlo=dict(coef=[float(v) for v in c1], rms=float(np.sqrt(np.mean(r1 ** 2)))),
                               logNlo_and_logp248=dict(coef=[float(v) for v in c2], rms=float(np.sqrt(np.mean(r2 ** 2))), resid={n: float(v) for n, v in zip(names_L, r2)}),
                               p016_curve=dict(rms=float(np.sqrt(np.mean(resid ** 2)))))
    say(f'\nZ regression over 40 entries: log N_lo only rms {RES["Z_regression"]["linear_in_logNlo"]["rms"]:.2f}; + log p248 rms {RES["Z_regression"]["logNlo_and_logp248"]["rms"]:.2f}; coefficients {c2}')
    say('  largest two-variable residuals: ' + ', '.join(f'{n}:{v:+.2f}' for n, v in sorted(zip(names_L, r2), key=lambda t: -abs(t[1]))[:6]))
    RES['spectral_ratios'] = {f'{a}/{b}': slope(a, b) for a, b in [('L12s', 'L20s'), ('L12v', 'L20v'), ('L19s', 'L11s'), ('L16s', 'L2s'), ('L16s', 'L20s'), ('L9s', 'L10s'), ('L9v', 'L10v'), ('L18s', 'L3s'), ('L6s', 'O3s'), ('L6v', 'O3v'), ('L7s', 'O7s'), ('L13s', 'O8s'), ('L13v', 'O9v'), ('L16s', 'L16v'), ('L12s', 'L12v')]}
    say('\nSpectral ratios (log-log slope over 20-400 keV, rms about power law, ratio at 248 keV):')
    for k, v in RES['spectral_ratios'].items():
        say(f'  {k}: slope {v["slope"]:+.3f}, rms {v["rms_dex"]:.4f} dex, ratio(248) {v["ratio_248"]:.4g}')
    RES['metrics'] = {n: {k: (float(v) if np.isfinite(v) else None) for k, v in MET[n].items()} for n in MET}
    RES['closure_checks'] = chk
    with open(OUT + '/P089_results.json', 'w') as f:
        json.dump(RES, f, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    np.savez(OUT + '/P089_spectra.npz', E_true=E_TRUE, E_obs=E_OBS, **{n: SP[n] for n in SP}, **{'pdf__' + n: PDF[n] for n in PDF})

    # ---------- figures ----------
    # 1. dendrogram of the 44 Table S6 + elastic S7 entries with LZ Z at 1 TeV
    sub = names_L + ['O1s', 'O1v', 'O4s', 'O4v']
    n = len(sub); h = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        h[i, j] = h[j, i] = H[idx[sub[i]], idx[sub[j]]]
    z = hierarchy.linkage(h[np.triu_indices(n, 1)], method='average')
    def zl(nm):
        return lz.LSIG[nm][11] if nm in lz.LSIG else lz.OSIG[nm][1000][0]
    labels = [f'{nm}  ({zl(nm):.1f}σ)' for nm in sub]
    fig, ax = plt.subplots(figsize=(7.5, 9.5))
    hierarchy.dendrogram(z, labels=labels, orientation='right', color_threshold=0.05, above_threshold_color='#777777', ax=ax, leaf_font_size=7.5)
    for t, c in [(0.02, '#D55E00'), (0.05, '#0072B2'), (0.15, '#009E73')]:
        ax.axvline(t, color=c, ls='--', lw=0.9); ax.text(t, -2.5, f'H = {t}', color=c, fontsize=7.5, ha='center', va='top')
    ax.set_xscale('log'); ax.set_xlim(3e-3, 1.0); ax.set_ylim(-9, n * 10 + 2)
    ax.set_xlabel('\nHellinger distance between observed-energy pdfs (1 TeV, LZ efficiency, σ_E = 11√(E/248) keV)')
    ax.set_title('P089: spectral classes of LZ Table S6 (L1–L20, s/v) and elastic O1/O4\nLZ local significance at 1 TeV in brackets; average linkage', fontsize=9)
    fig.tight_layout(); fig.savefig(FIG + '/P089_dendrogram.png', dpi=160); plt.close(fig)

    # 2. class-representative observed pdfs
    reps = [v['rep'] for v in MEMB[0.05].values() if v['rep'] in sub or any(mm in sub for mm in v['members'])]
    cols = plt.cm.tab20(np.linspace(0, 1, len(reps)))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    for k, rp in enumerate(reps):
        mem = [mm for mm in MEMB[0.05][[c for c, v in MEMB[0.05].items() if v['rep'] == rp][0]]['members'] if mm in sub]
        lab = rp + (f' (+{len(mem)-1})' if len(mem) > 1 else '')
        axs[0].plot(E_OBS, PDF[rp], color=cols[k], lw=1.3, label=lab)
        axs[1].plot(E_OBS, PDF[rp], color=cols[k], lw=1.3)
    for ax in axs:
        ax.axvline(248, color='k', ls=':', lw=0.8); ax.set_xlabel('observed energy [keV]'); ax.set_xlim(0, 330)
    axs[0].set_ylabel('pdf of accepted events [keV$^{-1}$]'); axs[0].set_ylim(0, None); axs[1].set_yscale('log'); axs[1].set_ylim(1e-5, 0.2)
    axs[0].legend(fontsize=6.5, ncol=2, frameon=False, title='class representatives (H < 0.05), members in brackets', title_fontsize=7)
    fig.suptitle('P089: one observed-energy spectrum per spectral class, 1 TeV, LZ efficiency and resolution', fontsize=9.5)
    fig.tight_layout(); fig.savefig(FIG + '/P089_class_spectra.png', dpi=160); plt.close(fig)

    # 3. Z_LZ(1 TeV) vs N_lo for the 40 Table S6 entries, colour by dominant response
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    xx = np.logspace(-1.5, 3.6, 200); ax.plot(xx, [z_pred(v) for v in xx], color='#777777', lw=1, label='P016 Z(N_lo) calibration')
    RESP_COL = {'M': '#0072B2', 'Sigma\'\'': '#D55E00', 'Sigma\'': '#009E73', 'Phi': '#CC79A7', 'Delta': '#E69F00', 'mixed': '#000000'}
    def dom(nm):
        r = LAG[int(nm[1:-1])]['resp']
        if 'Phi\'\'' in r and 'Sigma\'\'' not in r:
            return 'Phi'
        if r.count('Sigma\'\'') and 'M' not in r:
            return 'Sigma\'\''
        if 'Sigma\'' in r and 'M' not in r and 'Sigma\'\'' not in r:
            return 'Sigma\''
        if 'M' in r and 'Sigma' not in r and 'Phi' not in r:
            return 'M'
        return 'mixed'
    for nm in names_L:
        ax.scatter(max(MET[nm]['N_lo'], 0.03), lz.LSIG[nm][11], color=RESP_COL[dom(nm)], marker='o' if nm.endswith('s') else 's', s=28, zorder=3)
        ax.annotate(nm, (max(MET[nm]['N_lo'], 0.03), lz.LSIG[nm][11]), fontsize=6, xytext=(3, 2), textcoords='offset points')
    for k, c in RESP_COL.items():
        ax.scatter([], [], color=c, label=k)
    ax.scatter([], [], color='w', edgecolor='k', marker='o', label='isoscalar'); ax.scatter([], [], color='w', edgecolor='k', marker='s', label='isovector')
    ax.set_xscale('log'); ax.set_xlabel('N_lo = accepted rate 5.4–55 keV per event in 200–270 keV (WimPyDD, 1 TeV)'); ax.set_ylabel('LZ local significance at 1000 GeV [σ]')
    ax.legend(fontsize=7, frameon=False, ncol=2); ax.set_ylim(-0.1, 3.7); ax.set_title('P089: Table S6 significances follow the low-energy companion count', fontsize=9.5)
    fig.tight_layout(); fig.savefig(FIG + '/P089_Z_vs_Nlo.png', dpi=160); plt.close(fig)
    say('\nDone. Elapsed %.0f s' % (time.time() - T_START))

if __name__ == '__main__':
    main()
