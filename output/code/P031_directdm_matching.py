#!/usr/bin/env python
"""
P031 -- Relativistic operator matching for the LZ-preferred nonrelativistic operators with directdm.

Run from the simulation root:   .venv/bin/python output/code/P031_directdm_matching.py [--masses 1000,200,4000] [--final]

Stages (all cached in output/work/P031/cache/ so the script can be re-run in chunks):
  A  directdm (2.2.2) LO matching of every single relativistic operator structure (dim-5 dipoles, dim-6 quark
     currents, dim-7 gluon and m_q-weighted quark operators, dim-7 derivative "dipole-type" operators Q15-Q18,
     twist-2) onto the Anand et al. NR coefficients c_i^{p,n}(q) at q = 50 and 245 MeV for m_chi = 200/1000/4000 GeV.
  B  WimPyDD spectra for the full q-dependent NR Hamiltonian of each structure (pole terms kept as separate
     operator pieces), LZ efficiency, N_lo = R(5.4-55)/R(200-270 keV), ROI counts per unit Wilson coefficient,
     per-piece diagonal rate fractions at 20 and 248 keV (leading-operator identification).
  C  Wilson coefficient / scale Lambda giving 1.0 (0.3, 2.4) LZ events in 2.84 t yr; magnetic moment for the dipole.
  D  RG running MZ -> 2 GeV with directdm's 5->4->3 flavour chain (needs a numpy-2 workaround, see below).
  E  Comparison with LZ Table S6 through P016's Z(N_lo) calibration curve.
Outputs: CSV/JSON tables in output/work/P031/, figures in output/work/P031/figures/.
"""
import sys, os, json, math, time, argparse
sys.path.insert(0, 'output/code')
import numpy as np
import pandas as pd
from scipy import special, integrate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------------------------------------------
# numpy-2 workaround for directdm 2.2.2: WC_5flavor/WC_4flavor.__init__ call np.delete with index slices that run
# past the end of the coefficient array (e.g. np.s_[27:144] on a length-131 array).  numpy >= 1.19 raises
# IndexError; older numpy silently ignored out-of-range indices.  We wrap np.delete to drop out-of-range indices
# (this reproduces the old behaviour; the installed package is NOT modified).
# ---------------------------------------------------------------------------------------------------------------
_np_delete = np.delete
def _safe_delete(arr, obj, axis=None):
    a = np.asarray(arr)
    if axis is None and a.ndim == 1 and not isinstance(obj, (int, np.integer, slice)):
        idx = np.asarray(obj)
        obj = idx[(idx >= -a.shape[0]) & (idx < a.shape[0])]
    return _np_delete(arr, obj, axis=axis)
np.delete = _safe_delete
import directdm
import directdm.wilson_coefficients as _wcmod
_wcmod.np.delete = _safe_delete

from common import lzcommon as lz

ap = argparse.ArgumentParser()
ap.add_argument('--masses', default='1000', help='comma-separated DM masses in GeV for stage B')
ap.add_argument('--final', action='store_true', help='aggregate cached results, run stages C-E, make figures')
ap.add_argument('--skip-spectra', action='store_true')
ap.add_argument('--only', default='', help='comma-separated structure names to restrict stage B (used for the reduced 200/4000 GeV scan)')
args = ap.parse_args()
MASSES = [float(x) for x in args.masses.split(',')]

OUT = 'output/work/P031'; FIG = os.path.join(OUT, 'figures'); CACHE = os.path.join(OUT, 'cache')
for d in (OUT, FIG, CACHE):
    os.makedirs(d, exist_ok=True)
T0 = time.time()
def log(*a):
    print(f'[{time.time()-T0:6.0f}s]', *a, flush=True)

EXPOSURE = lz.LZ['exposure_tyr']          # 2.84 t yr
MN = lz.M_NUCLEON_GEV                     # 0.938272 GeV
MV2 = lz.M_V_GEV ** 2
ALPHA = 1 / 137.036                       # recalled, certain
E_CH = math.sqrt(4 * math.pi * ALPHA)     # 0.3028
Q_EVENT = math.sqrt(2 * lz.m_nucleus_gev(lz.A_XE_MEAN) * 248e-6)   # GeV, 248 keV on natural Xe
Q_LOW = 0.050
QS = {'q_event': Q_EVENT, 'q_50MeV': Q_LOW}
N_BEST, N_LO68, N_UP68 = 1.0, 0.3, 2.4    # LZ Table I: 1.0 +1.4 -0.7 events (L10^s, 1000 GeV)

# ---------------------------------------------------------------------------------------------------------------
# Relativistic operator structures (directdm / Bishara-Brod-Grinstein-Zupan basis, 1707.06998 + 1801.04240).
# Coefficient names: C51 (magnetic dipole), C52 (electric dipole); C61-C64 (V-V, A-V, V-A, A-A quark currents);
# C71-C74 (S GG, P GG, S GGdual, P GGdual); C75-C710 (m_q x S-S, P-S, S-P, P-P, T-T, PT-T); C715-C718 (dim-7
# derivative operators, chi-bar sigma^{mu nu} (i gamma5) chi x d_nu (q-bar gamma_mu (gamma5) q), identified from
# directdm's NR output -- see details.md); C723/C725 twist-2 quark/gluon.  Operator normalisation is directdm's
# (Q^(5) carry e/(8 pi^2), Q^(7)_{1-4} carry alpha_s/(12 pi) or alpha_s/(8 pi), Q^(7)_{5-10} carry m_q).
# ---------------------------------------------------------------------------------------------------------------
def univ(base, v=1.0, flav=('u', 'd', 's')):
    return {base + f: v for f in flav}
ZLIKE = {'u': 0.5, 'd': -0.5, 's': -0.5}   # SM axial charges of a Z-mediated axial current (T3), recalled, certain
QCHG = {'u': 2 / 3, 'd': -1 / 3, 's': -1 / 3}
STRUCTURES = [
    # name, coefficients, DM type, operator dimension, short description, Anand-L candidate (confidence)
    ('MDM', {'C51': 1.0}, 'D', 5, 'magnetic dipole e/(8pi^2) chibar sigma^{mu nu} chi F_{mu nu} (photon-mediated)', 'none (long-range)'),
    ('EDM', {'C52': 1.0}, 'D', 5, 'electric dipole e/(8pi^2) chibar sigma^{mu nu} i g5 chi F_{mu nu}', 'none (long-range)'),
    ('VV_univ', univ('C61'), 'D', 6, 'vector-vector chibar g^mu chi qbar g_mu q, u=d=s', 'L5^s (certain)'),
    ('VV_isov', {'C61u': 1.0, 'C61d': -1.0}, 'D', 6, 'vector-vector, u=-d', 'L5^v (certain)'),
    ('AV_univ', univ('C62'), 'D', 6, 'axial(DM)-vector(q) chibar g^mu g5 chi qbar g_mu q, u=d=s', 'A-V Lagrangian (L8 or L13, uncertain)'),
    ('AV_anapole', {'C62' + f: E_CH * QCHG[f] for f in 'uds'}, 'D', 6, 'anapole a chibar g^mu g5 chi d^nu F_{mu nu} -> e Q_q A-V', 'L16? (uncertain)'),
    ('VA_univ', univ('C63'), 'D', 6, 'vector(DM)-axial(q) chibar g^mu chi qbar g_mu g5 q, u=d=s', 'V-A Lagrangian (L7, likely)'),
    ('AA_univ', univ('C64'), 'D', 6, 'axial-axial chibar g^mu g5 chi qbar g_mu g5 q, u=d=s', 'L15^s (likely)'),
    ('AA_Zlike', {'C64' + f: ZLIKE[f] for f in 'uds'}, 'D', 6, 'axial-axial with Z (T3) quark charges', 'L15 (isovector-like)'),
    ('SGG', {'C71': 1.0}, 'D', 7, 'scalar-gluon alpha_s/(12pi) chibar chi G G', 'L1^s-like (certain)'),
    ('PGG', {'C72': 1.0}, 'D', 7, 'pseudoscalar(DM)-gluon chibar i g5 chi G G', 'L3-like'),
    ('SGGt', {'C73': 1.0}, 'D', 7, 'scalar(DM)-gluon-dual chibar chi G Gdual', 'L2-like'),
    ('PGGt', {'C74': 1.0}, 'D', 7, 'pseudoscalar-gluon-dual chibar i g5 chi G Gdual', 'L4-like'),
    ('SS_univ', univ('C75'), 'D', 7, 'm_q chibar chi qbar q, u=d=s', 'L1^s (certain)'),
    ('PS_univ', univ('C76'), 'D', 7, 'm_q chibar i g5 chi qbar q, u=d=s', 'L3 (certain)'),
    ('SP_univ', univ('C77'), 'D', 7, 'm_q chibar chi qbar i g5 q, u=d=s', 'L2 (certain)'),
    ('SP_isov', {'C77u': 1.0, 'C77d': -1.0}, 'D', 7, 'm_q chibar chi qbar i g5 q, u=-d', 'L2^v (certain)'),
    ('PP_univ', univ('C78'), 'D', 7, 'm_q chibar i g5 chi qbar i g5 q, u=d=s', 'L4 (certain)'),
    ('PP_isov', {'C78u': 1.0, 'C78d': -1.0}, 'D', 7, 'm_q chibar i g5 chi qbar i g5 q, u=-d', 'L4^v (certain)'),
    ('TT_univ', univ('C79'), 'D', 7, 'm_q chibar sigma^{mu nu} chi qbar sigma_{mu nu} q, u=d=s', 'tensor-tensor (no Anand L, see text)'),
    ('TT_isov', {'C79u': 1.0, 'C79d': -1.0}, 'D', 7, 'tensor-tensor, u=-d', 'tensor-tensor isovector'),
    ('PTT_univ', univ('C710'), 'D', 7, 'm_q chibar i sigma^{mu nu} g5 chi qbar sigma_{mu nu} q, u=d=s', 'pseudotensor-tensor'),
    ('Q15_univ', univ('C715'), 'D', 7, 'chibar sigma^{mu nu} chi d_nu(qbar g_mu q) [DM dipole x quark vector current], u=d=s', 'L6 (likely)'),
    ('Q15_charge', {'C715' + f: QCHG[f] for f in 'uds'}, 'D', 7, 'same with quark charges (contact analogue of the photon dipole)', 'L6 (likely)'),
    ('Q16_univ', univ('C716'), 'D', 7, 'chibar sigma^{mu nu} i g5 chi d_nu(qbar g_mu q) [DM EDM x quark vector current]', 'none secure'),
    ('Q17_univ', univ('C717'), 'D', 7, 'chibar sigma^{mu nu} chi d_nu(qbar g_mu g5 q) [DM dipole x quark axial current]', 'L9? (uncertain)'),
    ('Q18_univ', univ('C718'), 'D', 7, 'chibar sigma^{mu nu} i g5 chi d_nu(qbar g_mu g5 q)', 'none secure'),
    ('TW2_quark', univ('C723'), 'D', 7, 'twist-2 quark operator, u=d=s', 'L1/L5-like'),
    ('TW2_gluon', {'C725': 1.0}, 'D', 7, 'twist-2 gluon operator', 'L1-like'),
]
MAJORANA_OK = {'C62', 'C64', 'C71', 'C72', 'C73', 'C74', 'C75', 'C76', 'C77', 'C78'}
def majorana_allowed(coeffs):
    return all(any(k.startswith(b) for b in MAJORANA_OK) for k in coeffs)

# ---------------------------------------------------------------------------------------------------------------
# directdm -> NR pieces.  cNR_i(q) = const + q^2 * c_q2 + c_bq2 / q^2 + c_pi/(mpi^2+q^2) + c_eta/(meta^2+q^2)
#                                  + c_q2pi q^2/(mpi^2+q^2) + c_q2eta q^2/(meta^2+q^2)      (directdm cNR())
# Each piece with a distinct q-dependence becomes a separate WimPyDD operator (key (i, label)) so that WimPyDD's
# per-coefficient momentum factorisation is exact (verified: identical rates for a mixed-q single coefficient
# and its split into pieces).
# ---------------------------------------------------------------------------------------------------------------
_W3 = {}
def w3(coeffs, dm_type):
    key = (tuple(sorted(coeffs.items())), dm_type)
    if key not in _W3:
        _W3[key] = directdm.WC_3f(dict(coeffs), dm_type)
    return _W3[key]

def nr_pieces(coeffs, dm_type, m):
    w = w3(coeffs, dm_type)
    my = w._my_cNR(m)
    mpi2, meta2 = w.ip['mpi0'] ** 2, w.ip['meta'] ** 2
    pieces = {}   # (i, label) -> (c0_factor, c1_factor, kind)
    def add(i, label, cp, cn, kind):
        if abs(cp) + abs(cn) > 0:
            pieces[(i, label)] = (cp + cn, cp - cn, kind)
    for i in [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14]:
        add(i, 'c', my[f'cNR{i}p'], my[f'cNR{i}n'], 'const')
    add(1, 'q2', my['cNR1q2p'], my['cNR1q2n'], 'q2')
    add(4, 'q2', my['cNR4q2p'], my['cNR4q2n'], 'q2')
    add(5, 'bq2', my['cNR5bq2p'], my['cNR5bq2n'], 'bq2')
    add(6, 'bq2', my['cNR6bq2p'], my['cNR6bq2n'], 'bq2')
    add(11, 'bq2', my['cNR11bq2p'], my['cNR11bq2n'], 'bq2')
    add(6, 'pi', my['cNR6pip'], my['cNR6pin'], 'pi')
    add(6, 'eta', my['cNR6etap'], my['cNR6etan'], 'eta')
    add(6, 'q2pi', my['cNR6q2pip'], my['cNR6q2pin'], 'q2pi')
    add(6, 'q2eta', my['cNR6q2etap'], my['cNR6q2etan'], 'q2eta')
    add(10, 'pi', my['cNR10pip'], my['cNR10pin'], 'pi')
    add(10, 'eta', my['cNR10etap'], my['cNR10etan'], 'eta')
    add(10, 'q2pi', my['cNR10q2pip'], my['cNR10q2pin'], 'q2pi')
    add(10, 'q2eta', my['cNR10q2etap'], my['cNR10q2etan'], 'q2eta')
    return pieces, mpi2, meta2

def qfactor(kind, q, mpi2, meta2):
    return {'const': 1.0, 'q2': q ** 2, 'bq2': 1 / q ** 2, 'pi': 1 / (mpi2 + q ** 2), 'eta': 1 / (meta2 + q ** 2),
            'q2pi': q ** 2 / (mpi2 + q ** 2), 'q2eta': q ** 2 / (meta2 + q ** 2)}[kind]

# NOTE: WimPyDD inspects the *signature* of each coefficient function and treats every non-reserved argument
# (including default arguments) as a global Hamiltonian parameter shared by all coefficients.  Coefficient values
# must therefore be captured by closure, never passed as default arguments (a first version did that and the
# pieces silently overwrote each other; caught by the per-piece diagonal check, see details.md).
def _const_fn(c0, c1):
    def f():
        return [c0, c1]
    return f
def _q_fn(c0, c1, kind, mpi2, meta2):
    def f(q, q0=0.1):
        fac = qfactor(kind, q, mpi2, meta2)
        return [c0 * fac, c1 * fac]
    return f
def make_ham(name, pieces, mpi2, meta2, keys=None):
    wc = {}
    for key, (c0, c1, kind) in pieces.items():
        if keys is not None and key not in keys:
            continue
        wc[key] = _const_fn(c0, c1) if kind == 'const' else _q_fn(c0, c1, kind, mpi2, meta2)
    return lz.wd().eft_hamiltonian(name, wc)

# ---------------------------------------------------------------------------------------------------------------
# Efficiency and window integrals (P003/P012 form)
# ---------------------------------------------------------------------------------------------------------------
def efficiency(E, sig_lo=3.4, sig_hi=8.0, eff0=lz.LZ['eff_plateau']):
    E = np.asarray(E, float)
    lo, hi = lz.LZ['E_50pct_low_keV'], lz.LZ['E_50pct_high_keV']
    return eff0 * 0.5 * (1 + special.erf((E - lo) / (math.sqrt(2) * sig_lo))) * 0.5 * (1 - special.erf((E - hi) / (math.sqrt(2) * sig_hi)))

def window_rate(Egrid, dRdE, e1, e2):
    Ef = np.linspace(e1, e2, 1501)
    y = np.asarray(dRdE, float)
    yf = 10 ** np.interp(Ef, Egrid, np.log10(np.clip(y, 1e-300, None))) if np.all(y > 0) else np.interp(Ef, Egrid, y)
    return float(integrate.trapezoid(yf * efficiency(Ef), Ef))

def summarise(Egrid, dRdE):
    R_lo = window_rate(Egrid, dRdE, 5.4, 55.0); R_mid = window_rate(Egrid, dRdE, 55.0, 200.0)
    R_hi = window_rate(Egrid, dRdE, 200.0, 270.0); R_roi = window_rate(Egrid, dRdE, 1.0, 330.0)
    return dict(R_lo=R_lo, R_mid=R_mid, R_hi=R_hi, R_roi=R_roi, N_lo_per_hi=R_lo / R_hi if R_hi > 0 else np.inf,
                N_mid_per_hi=R_mid / R_hi if R_hi > 0 else np.inf, f_hi=R_hi / R_roi if R_roi > 0 else np.nan)

EGRID = np.union1d(np.arange(1.0, 330.0 + 1e-9, 4.0), [5.4, 20.0, 55.0, 200.0, 248.0, 269.9, 300.0])
WD = lz.wd(); HALO = lz.wd_halo()

# ---------------------------------------------------------------------------------------------------------------
# Stage A: NR coefficient tables
# ---------------------------------------------------------------------------------------------------------------
NR_NAMES = {1: 'O1', 3: 'O3', 4: 'O4', 5: 'O5', 6: 'O6', 7: 'O7', 8: 'O8', 9: 'O9', 10: 'O10', 11: 'O11', 12: 'O12', 14: 'O14'}
def stage_A():
    rows = []
    for name, coeffs, typ, dim, desc, anand in STRUCTURES:
        for m in [200.0, 1000.0, 4000.0]:
            w = w3(coeffs, typ)
            for qlab, q in QS.items():
                c = w.cNR(m, q)
                row = dict(structure=name, dm_type=typ, dim=dim, m_chi_GeV=m, q_label=qlab, q_GeV=q)
                for i in NR_NAMES:
                    row[f'c{i}p'] = c[f'cNR{i}p']; row[f'c{i}n'] = c[f'cNR{i}n']
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'P031_nr_coefficients.csv'), index=False, float_format='%.6g')
    log('stage A: NR coefficient table written', df.shape)
    return df

# ---------------------------------------------------------------------------------------------------------------
# Stage B: WimPyDD spectra per structure and mass (cached)
# ---------------------------------------------------------------------------------------------------------------
def stage_B(masses):
    only = [s for s in args.only.split(',') if s]
    for name, coeffs, typ, dim, desc, anand in STRUCTURES:
        if only and name not in only:
            continue
        for m in masses:
            f = os.path.join(CACHE, f'{name}_m{int(m)}.npz')
            if os.path.exists(f):
                continue
            pieces, mpi2, meta2 = nr_pieces(coeffs, typ, m)
            ham = make_ham(f'P031_{name}_{int(m)}', pieces, mpi2, meta2)
            spec = lz.wd_rate(ham, m, EGRID, HALO)
            # per-piece diagonal contributions at 20 and 248 keV (interference dropped in this diagnostic)
            diag = {}
            for key in pieces:
                h1 = make_ham(f'P031_{name}_{int(m)}_{key[0]}{key[1]}', pieces, mpi2, meta2, keys=[key])
                diag[f'{key[0]}_{key[1]}'] = lz.wd_rate(h1, m, np.array([20.0, 248.0]), HALO)
            full20, full248 = float(np.interp(20.0, EGRID, spec)), float(np.interp(248.0, EGRID, spec))
            dsum20, dsum248 = sum(v[0] for v in diag.values()), sum(v[1] for v in diag.values())
            np.savez(f, E=EGRID, spec=spec, pieces=json.dumps({f'{k[0]}_{k[1]}': [v[0], v[1], v[2]] for k, v in pieces.items()}),
                     diag=json.dumps({k: list(map(float, v)) for k, v in diag.items()}), full_over_diag=np.array([full20 / dsum20, full248 / dsum248]))
            s = summarise(EGRID, spec)
            log(f'{name:12s} m={m:5.0f}: N_lo={s["N_lo_per_hi"]:9.3g}  R_hi={s["R_hi"]:.3g}/t/yr  full/diag-sum(20,248)={full20/dsum20:.3f},{full248/dsum248:.3f}  pieces={list(diag)}')
    # reference: Anand L10 (dipole-dipole, m_M = m_N), P012 implementation, WimPyDD c0 = 2 x Anand, d10 = 1
    for m in masses:
        f = os.path.join(CACHE, f'L10ref_m{int(m)}.npz')
        if os.path.exists(f):
            continue
        c4 = lz.wd_c_from_anand(4 / MV2)[0]; c6 = lz.wd_c_from_anand(-4 / MV2)[0]
        h = WD.eft_hamiltonian(f'P031_L10ref_{int(m)}', {(4, 'q2'): lambda q, A=c4: [A * q ** 2 / MN ** 2, 0.0], 6: lambda A6=c6: [A6, 0.0]})
        spec = lz.wd_rate(h, m, EGRID, HALO)
        np.savez(f, E=EGRID, spec=spec)
        s = summarise(EGRID, spec)
        log(f'L10 reference m={m:.0f}: N_lo={s["N_lo_per_hi"]:.3f}, N(d10=1, WimPyDD norm)={s["R_roi"]*EXPOSURE:.3f} (P012: 3.336 at 1000 GeV)')

# ---------------------------------------------------------------------------------------------------------------
# Stages C-E: aggregation
# ---------------------------------------------------------------------------------------------------------------
def z_curve():
    zc = pd.read_csv('output/work/P016/P016_Z_vs_Nlo_curves.csv')
    x = np.log10(zc['N_lo'].values); y = zc['baseline (O4-like 55-200 keV companions)'].values
    def Z(nlo):
        nlo = np.clip(np.asarray(nlo, float), 10 ** x.min(), 10 ** x.max())
        return np.interp(np.log10(nlo), x, y)
    return Z

def final():
    Z_of = z_curve()
    rows, piece_rows = [], []
    for name, coeffs, typ, dim, desc, anand in STRUCTURES:
        for m in [200.0, 1000.0, 4000.0]:
            f = os.path.join(CACHE, f'{name}_m{int(m)}.npz')
            if not os.path.exists(f):
                continue
            d = np.load(f, allow_pickle=True)
            spec = d['spec']; s = summarise(d['E'], spec)
            diag = json.loads(str(d['diag'])); pieces = json.loads(str(d['pieces']))
            tot20 = sum(v[0] for v in diag.values()); tot248 = sum(v[1] for v in diag.values())
            lead248 = max(diag, key=lambda k: diag[k][1]); lead20 = max(diag, key=lambda k: diag[k][0])
            # group diagonal fractions by NR operator index
            byop248, byop20 = {}, {}
            for k, v in diag.items():
                i = int(k.split('_')[0]); byop248[i] = byop248.get(i, 0) + v[1]; byop20[i] = byop20.get(i, 0) + v[0]
            fr248 = {NR_NAMES[i]: byop248[i] / tot248 for i in byop248}; fr20 = {NR_NAMES[i]: byop20[i] / tot20 for i in byop20}
            lead_op248 = max(fr248, key=fr248.get); lead_op20 = max(fr20, key=fr20.get)
            N_unit = s['R_roi'] * EXPOSURE            # events in 2.84 t yr at unit Wilson coefficient (GeV^-(dim-4))
            C1 = 1 / math.sqrt(N_unit) if N_unit > 0 else np.inf
            p = dim - 4
            Lam = C1 ** (-1 / p) if np.isfinite(C1) else 0.0
            Lam_lo = (math.sqrt(N_UP68 / N_unit)) ** (-1 / p) if N_unit > 0 else 0.0     # more events -> larger C -> smaller Lambda
            Lam_hi = (math.sqrt(N_LO68 / N_unit)) ** (-1 / p) if N_unit > 0 else 0.0
            nlo = s['N_lo_per_hi']
            cls = 'compatible' if nlo <= 5 else ('marginal' if nlo <= 20 else 'excluded')
            row = dict(structure=name, description=desc, anand_candidate=anand, dm_type=typ, majorana_allowed=majorana_allowed(coeffs), dim=dim,
                       m_chi_GeV=m, N_lo=nlo, N_mid=s['N_mid_per_hi'], f_hi=s['f_hi'], R_lo=s['R_lo'], R_hi=s['R_hi'], R_roi=s['R_roi'],
                       N_events_unit_C=N_unit, C_1event=C1, Lambda_1event_GeV=Lam, Lambda_lo68_GeV=Lam_lo, Lambda_hi68_GeV=Lam_hi,
                       Lambda_over_mchi=Lam / m, leading_op_248keV=lead_op248, frac_lead_248=fr248[lead_op248],
                       leading_op_20keV=lead_op20, frac_lead_20=fr20[lead_op20], leading_piece_248=lead248, leading_piece_20=lead20,
                       fractions_248=json.dumps({k: round(v, 4) for k, v in fr248.items()}), fractions_20=json.dumps({k: round(v, 4) for k, v in fr20.items()}),
                       classification=cls, Z_pred_P016=float(Z_of(nlo)), dRdE_248_unit=float(np.interp(248.0, d['E'], spec)))
            if name == 'MDM':
                row['mu_chi_1event_muN'] = C1 * E_CH / (4 * math.pi ** 2) / (E_CH / (2 * MN))   # mu = e C51/(4 pi^2); mu_N = e/(2 m_N)
            if name == 'AV_anapole':
                row['anapole_a_1event_GeV-2'] = C1
            rows.append(row)
            for k, v in pieces.items():
                piece_rows.append(dict(structure=name, m_chi_GeV=m, piece=k, c0_factor=v[0], c1_factor=v[1], kind=v[2],
                                       diag_rate_20keV=diag[k][0], diag_rate_248keV=diag[k][1], frac_248=diag[k][1] / tot248, frac_20=diag[k][0] / tot20))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, 'P031_structure_table.csv'), index=False, float_format='%.5g')
    pd.DataFrame(piece_rows).to_csv(os.path.join(OUT, 'P031_nr_pieces.csv'), index=False, float_format='%.5g')
    ref = {}
    for m in [200.0, 1000.0, 4000.0]:
        f = os.path.join(CACHE, f'L10ref_m{int(m)}.npz')
        if os.path.exists(f):
            d = np.load(f); s = summarise(d['E'], d['spec'])
            ref[int(m)] = dict(N_lo=s['N_lo_per_hi'], N_unit_wd=s['R_roi'] * EXPOSURE, d10_wd_1event=1 / math.sqrt(s['R_roi'] * EXPOSURE),
                               d10_lz_1event=1 / math.sqrt(s['R_roi'] * EXPOSURE * 3.857), Z_pred_P016=float(Z_of(s['N_lo_per_hi'])))
    t1000 = df[df.m_chi_GeV == 1000].sort_values('N_lo')
    log('--- 1000 GeV summary ---')
    print(t1000[['structure', 'dim', 'N_lo', 'leading_op_248keV', 'frac_lead_248', 'leading_op_20keV', 'classification', 'Z_pred_P016',
                 'N_events_unit_C', 'Lambda_1event_GeV', 'Lambda_over_mchi']].to_string(index=False))
    print('L10 reference:', json.dumps(ref, indent=None))

    # ---------------- Stage D: RG running MZ -> 2 GeV (directdm 5->4->3 flavour chain, numpy workaround) --------------
    rg_rows = []
    RG_CASES = [('TT_univ', univ('C79') | {'C79c': 1.0, 'C79b': 1.0}), ('TT_isov', {'C79u': 1.0, 'C79d': -1.0}), ('PTT_univ', univ('C710')),
                ('PP_univ', univ('C78')), ('PP_isov', {'C78u': 1.0, 'C78d': -1.0}), ('SP_univ', univ('C77')), ('SP_isov', {'C77u': 1.0, 'C77d': -1.0}),
                ('AA_univ', univ('C64')), ('AA_Zlike', {'C64' + f: ZLIKE[f] for f in 'uds'}), ('AV_univ', univ('C62')), ('VV_univ', univ('C61')),
                ('SGGt', {'C73': 1.0}), ('PGGt', {'C74': 1.0}), ('MDM', {'C51': 1.0}), ('Q15_univ', univ('C715')), ('Q17_univ', univ('C717')), ('SS_univ', univ('C75'))]
    for name, coeffs in RG_CASES:
        try:
            w5 = directdm.WC_5f(dict(coeffs), 'D')
            d4 = w5.match(); d3 = directdm.WC_4f(d4, 'D').match()
            nz3 = {k: v for k, v in d3.items() if abs(v) > 1e-12 and not k.startswith('D') and not k.startswith('SM')}
            c_rg = w5.cNR(1000.0, Q_EVENT); c_no = w5.cNR(1000.0, Q_EVENT, RGE=False)
            keys = [k for k in c_rg if abs(c_rg[k]) > 1e-12 or abs(c_no[k]) > 1e-12]
            rg_rows.append(dict(structure=name, input_MZ=json.dumps(coeffs), WC_2GeV=json.dumps({k: float(f'{v:.4g}') for k, v in nz3.items()}),
                                cNR_RGE=json.dumps({k: float(f'{c_rg[k]:.4g}') for k in keys}), cNR_noRGE=json.dumps({k: float(f'{c_no[k]:.4g}') for k in keys}),
                                ratio_leading=float(c_rg[keys[0]] / c_no[keys[0]]) if keys and c_no[keys[0]] != 0 else np.nan,
                                new_ops_from_mixing=json.dumps([k for k in keys if abs(c_no[k]) < 1e-12 and abs(c_rg[k]) > 1e-12])))
            log(f'RG {name:10s}: 2 GeV WCs {nz3}')
        except Exception as e:
            rg_rows.append(dict(structure=name, input_MZ=json.dumps(coeffs), error=repr(e)[:200]))
            log(f'RG {name}: FAILED {e!r}')
    pd.DataFrame(rg_rows).to_csv(os.path.join(OUT, 'P031_rg_running.csv'), index=False)

    # ---------------- Stage E: Table S6 comparison for the secure identifications --------------------------------
    LS = lz.LSIG; im = lz.LSIG_MASSES.index(1000)
    ident = [('L1s', 'SS_univ', 'certain'), ('L1s', 'SGG', 'certain (gluon version)'), ('L5s', 'VV_univ', 'certain'), ('L5v', 'VV_isov', 'certain'),
             ('L2s', 'SP_univ', 'certain'), ('L2v', 'SP_isov', 'certain'), ('L3s', 'PS_univ', 'certain'), ('L4s', 'PP_univ', 'certain'), ('L4v', 'PP_isov', 'certain'),
             ('L15s', 'AA_univ', 'likely'), ('L15v', 'AA_Zlike', 'likely (isovector-like)'), ('L7s', 'VA_univ', 'likely'), ('L6s', 'Q15_univ', 'likely'),
             ('L8s', 'AV_univ', 'uncertain'), ('L13s', 'AV_univ', 'uncertain'), ('L16s', 'AV_anapole', 'uncertain (anapole)'), ('L9s', 'Q17_univ', 'uncertain')]
    cmp_rows = []
    for L, struct, conf in ident:
        r = df[(df.structure == struct) & (df.m_chi_GeV == 1000)]
        if len(r) == 0:
            continue
        r = r.iloc[0]
        cmp_rows.append(dict(anand_L=L, structure=struct, confidence=conf, Z_LZ_tableS6_1000GeV=LS[L][im], N_lo=r.N_lo, Z_pred_P016=r.Z_pred_P016,
                             leading_op_248=r.leading_op_248keV, classification=r.classification, Z_LZ_4000GeV=LS[L][-1], Z_LZ_200GeV=LS[L][9]))
    cmp = pd.DataFrame(cmp_rows); cmp.to_csv(os.path.join(OUT, 'P031_tableS6_comparison.csv'), index=False, float_format='%.3g')
    print(cmp.to_string(index=False))

    # summary json
    summ = dict(q_event_GeV=Q_EVENT, exposure_tyr=EXPOSURE, L10_reference=ref, n_structures=len(STRUCTURES),
                compatible_1000GeV=t1000[t1000.classification == 'compatible'].structure.tolist(),
                marginal_1000GeV=t1000[t1000.classification == 'marginal'].structure.tolist(),
                excluded_1000GeV=t1000[t1000.classification == 'excluded'].structure.tolist(),
                table_1000GeV=t1000.set_index('structure')[['N_lo', 'leading_op_248keV', 'frac_lead_248', 'leading_op_20keV', 'Z_pred_P016', 'N_events_unit_C',
                                                             'C_1event', 'Lambda_1event_GeV', 'Lambda_lo68_GeV', 'Lambda_hi68_GeV', 'Lambda_over_mchi', 'classification']].to_dict(orient='index'),
                mu_chi_MDM_1event_muN={int(r.m_chi_GeV): r.mu_chi_1event_muN for _, r in df[df.structure == 'MDM'].iterrows()},
                tableS6_comparison=cmp.to_dict(orient='records'))
    json.dump(summ, open(os.path.join(OUT, 'P031_summary.json'), 'w'), indent=1, default=float)

    # ---------------- Figures -------------------------------------------------------------------------------------
    ops = ['O1', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O14']
    names = t1000.structure.tolist()
    M = np.zeros((len(names), len(ops)))
    for i, n in enumerate(names):
        fr = json.loads(t1000[t1000.structure == n].fractions_248.iloc[0])
        for j, o in enumerate(ops):
            M[i, j] = fr.get(o, 0.0)
    fig, ax = plt.subplots(figsize=(8.5, 9.5))
    im_ = ax.imshow(M, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(ops))); ax.set_xticklabels(ops, rotation=45)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f'{n}  (N_lo={t1000[t1000.structure == n].N_lo.iloc[0]:.2g})' for n in names], fontsize=8)
    for i, n in enumerate(names):
        c = t1000[t1000.structure == n].classification.iloc[0]
        ax.get_yticklabels()[i].set_color({'compatible': 'green', 'marginal': 'darkorange', 'excluded': 'crimson'}[c])
    for j, o in enumerate(ops):
        ax.get_xticklabels()[j].set_color('green' if o in ('O5', 'O6', 'O9', 'O10', 'O14') else ('crimson' if o in ('O1', 'O4', 'O7', 'O8', 'O11', 'O12') else 'black'))
    plt.colorbar(im_, ax=ax, label='fraction of dR/dE at 248 keV from operator (diagonal terms), 1000 GeV')
    ax.set_title('P031: directdm NR reduction of relativistic operators -> LZ 248 keV event\n(rows sorted by N_lo; green = compatible, orange = marginal, red = excluded)', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P031_nr_coefficient_map.png'), dpi=140); plt.close(fig)

    # Lambda for 1 event vs mass for compatible/marginal structures
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sel = df[df.classification.isin(['compatible', 'marginal']) & (df.m_chi_GeV == 1000)].structure.unique()
    for n in sel:
        r = df[df.structure == n].sort_values('m_chi_GeV')
        ax.errorbar(r.m_chi_GeV, r.Lambda_1event_GeV, yerr=[r.Lambda_1event_GeV - r.Lambda_lo68_GeV, r.Lambda_hi68_GeV - r.Lambda_1event_GeV],
                    marker='o', capsize=2, label=f'{n} (dim {int(r.dim.iloc[0])})')
    mm = np.array([200, 1000, 4000]); ax.plot(mm, mm, 'k--', lw=1, label=r'$\Lambda = m_\chi$'); ax.axhline(91.2, color='grey', ls=':', lw=1); ax.text(210, 95, r'$M_Z$', color='grey')
    ax.axhline(2.0, color='grey', ls=':', lw=1); ax.text(210, 2.1, r'$\mu = 2$ GeV', color='grey')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'$m_\chi$ [GeV]'); ax.set_ylabel(r'$\Lambda$ for 1.0 LZ event (bars: 0.3-2.4 events) [GeV]')
    ax.legend(fontsize=7, ncol=2); ax.set_title('P031: scale of the relativistic operator reproducing the LZ fit', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P031_lambda_vs_mass.png'), dpi=140); plt.close(fig)

    # Z predicted vs LZ Table S6 for identified Lagrangians
    if len(cmp):
        fig, ax = plt.subplots(figsize=(6, 5))
        for conf, mk in [('certain', 'o'), ('likely', 's'), ('uncertain', 'x')]:
            c = cmp[cmp.confidence.str.startswith(conf)]
            ax.scatter(c.Z_LZ_tableS6_1000GeV, c.Z_pred_P016, marker=mk, s=60, label=conf)
            for _, r in c.iterrows():
                ax.annotate(f'{r.anand_L}={r.structure}', (r.Z_LZ_tableS6_1000GeV, r.Z_pred_P016), fontsize=6, xytext=(3, 3), textcoords='offset points')
        ax.plot([0, 3.6], [0, 3.6], 'k--', lw=1); ax.set_xlabel('LZ Table S6 local Z (1000 GeV)'); ax.set_ylabel('Z predicted from N_lo via P016 curve')
        ax.legend(); ax.set_title('P031: relativistic structures vs LZ Lagrangian significances', fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P031_Z_vs_tableS6.png'), dpi=140); plt.close(fig)
    log('final done')

if __name__ == '__main__':
    stage_A()
    if not args.skip_spectra:
        stage_B(MASSES)
    if args.final:
        final()
    log('done')
