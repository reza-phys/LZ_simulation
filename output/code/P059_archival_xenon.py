"""
P059 -- Archival xenon exposures reinterpreted for the extended window:
what LUX (2021 EFT), XENON1T (2018), PandaX-II (2019 SD-EFT) and LZ's own first run (SR1, 2023 NREFT)
already said about a signal of 1.0 event per 2.84 t yr in the LZ 3-600 phd window.

Run from the simulation root:  .venv/bin/python output/code/P059_archival_xenon.py
Outputs -> output/work/P059/ (CSV/JSON) and output/work/P059/figures/ (PNG)

Parts
  A  recalled exposure/ROI table (every entry flagged)
  B  spectra normalised to LZ's best fit (P005 WimPyDD spectra + own WimPyDD for delta = 366 keV, unit-coupling
     counts N_unit(delta) for the Fig. 6 consistency check, and L10 spectra for 5 masses)
  C  expected counts in each archival exposure with (i) LZ-like acceptance and (ii) their actual ROI edges; P(0)
  D  combined Poisson likelihood LZ(1 event) + archival(0 events): best-fit rate, 90% interval, local Z
  E  Fig. 6 previous-limit curves digitised (PyMuPDF vector paths): implied archival counts / effective exposures
  F  the SR1 look-back: expected counts at the new best fit and the effect of a joint SR1+SR3 fit
  G  figures
"""
import sys, os, json, math
import numpy as np, pandas as pd
from scipy import special, stats, optimize
sys.path.insert(0, 'output/code')
from common import lzcommon as lz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'output/work/P059'; FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)
LOG = open(os.path.join(OUT, 'run_log.txt'), 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n'); LOG.flush()

E_LZ = lz.LZ['exposure_tyr']                 # 2.84 t yr [paper]
N_BF = lz.LZ['L10s_1000_bestfit']            # (1.0, +1.4, -0.7) [paper Table I]
BAND = (N_BF[0] + N_BF[2], N_BF[0], N_BF[0] + N_BF[1])   # 0.3, 1.0, 2.4 events
MV = lz.M_V_GEV; mN = lz.M_NUCLEON_GEV

# =============================================================================================
# Part A: archival exposures (recalled; every number flagged in details.md / provenance)
# =============================================================================================
ARCH = {
    'LZ SR1 (2023 NREFT)': dict(expo_tyr=60.0 * 5.5 / 365.25, live_days=60.0, mass_t=5.5,
                                kind='lz_rolloff', E50_hi=269.9 * 0.110 / 0.114, plateau=0.96, E50_lo=5.4,
                                variants=dict(E50_hi=[250.0, 269.9], plateau=[0.85]),
                                flag='60 live days (likely), 5.5 t FV (likely); S1c 3-600 phd (paper l.116: same S1c range); '
                                     'g1(SR1)=0.114 phd/photon (likely) -> 600 phd ~ 260 keV; zero NR-band-like events at high S1c (uncertain)'),
    'XENON1T 2018': dict(expo_tyr=1.0, kind='hard', E_lo=4.9, E_hi=40.9, plateau=0.85,
                         variants=dict(E_hi=[50.0], plateau=[0.95]),
                         flag='1.0 t yr, ROI 4.9-40.9 keV (certain); plateau 0.85 (likely); inelastic recast ends at delta=231 keV (P035)'),
    'LUX 2021 EFT': dict(expo_tyr=3.35e4 / 1000.0 / 365.25, kind='hard', E_lo=3.0, E_hi=150.0, plateau=0.8,
                         variants=dict(E_hi=[100.0, 250.0], plateau=[0.5]),
                         flag='311.2 live days (certain, bib title); 3.35e4 kg d = 0.092 t yr (likely); ROI edge unknown (uncertain), scanned 100-250 keV; plateau 0.8 (uncertain)'),
    'PandaX-II 2019': dict(expo_tyr=54.0 / 365.25, kind='hard', E_lo=3.0, E_hi=100.0, plateau=0.8,
                           variants=dict(E_hi=[50.0, 150.0], plateau=[0.5]),
                           flag='54 t d = 0.148 t yr (likely); ROI edge unknown (uncertain), scanned 50-150 keV; plateau 0.8 (uncertain)'),
}
for k, v in ARCH.items():
    say(f'{k:22s} exposure {v["expo_tyr"]:.4f} t yr  ({v["flag"]})')
E_ARCH_TOTAL = sum(v['expo_tyr'] for v in ARCH.values())
say(f'Total archival exposure {E_ARCH_TOTAL:.3f} t yr (LZ new: {E_LZ} t yr)')

# =============================================================================================
# Part B: efficiencies and spectra
# =============================================================================================
def erf_up(E, e50, sig):
    return 0.5 * (1.0 + special.erf((np.asarray(E, float) - e50) / (math.sqrt(2.0) * sig)))

SIG_LO, SIG_HI = 2.5, 11.5    # keV (P005/P021 smooth model of LZ Fig. S2)
def eff_lz(E, plateau=0.96, E50_lo=5.4, E50_hi=269.9):
    return plateau * erf_up(E, E50_lo, SIG_LO) * (1.0 - erf_up(E, E50_hi, SIG_HI))

def eff_arch(E, cfg, **over):
    c = dict(cfg); c.update(over)
    if c['kind'] == 'lz_rolloff':
        return eff_lz(E, c['plateau'], c['E50_lo'], c['E50_hi'])
    E = np.asarray(E, float)
    return c['plateau'] * erf_up(E, c['E_lo'], 1.5) * (E < c['E_hi'])

# --- B1: spectra normalised to 1.0 LZ event (P005: WimPyDD, Sun-frame Baxter halo, 1000 GeV) ---
p005 = pd.read_csv('output/work/P005/spectra_normalised_LZ1.csv')
E5 = p005.E_keV.values
SPEC = {'L10 1 TeV': (E5, p005.L10_1000.values), 'O1 inel. 250': (E5, p005.O1s_d250.values),
        'O1 inel. 300': (E5, p005.O1s_d300.values), 'O1 inel. 350': (E5, p005.O1s_d350.values)}

# --- B2: own WimPyDD spectra: delta = 366 keV (P007 Higgsino), unit-coupling N_unit(delta), L10 masses ---
HALO = lz.wd_halo()          # SUN-FRAME halo (P035 labelling); LZ appears to use the time-averaged SHM (P007/P021)
ham_o1 = lz.wd_hamiltonian('o1_unit', {1: (2.0 / MV**2, 0.0)})    # c_p = c_n = 1/m_v^2 (LZ unit coupling, P003)
# REDUCED GRIDS (a first attempt with 0.5 keV steps and 8 deltas + 5 L10 masses exceeded the 600 s foreground limit at
# ~0.3 s per WimPyDD call): 366 keV spectrum on 1-800 keV step 2; unit-coupling spectra on a 0.5 keV (E<60) / 2 keV grid
# for delta = 0..250 keV (300/350 keV taken from P035's table); L10 for 200/1000/4000 GeV on 1-400 keV step 2.
E_FULL = np.arange(1.0, 800.0 + 0.01, 2.0)
E_UNIT = np.concatenate([np.arange(0.5, 60.0, 0.5), np.arange(60.0, 320.0 + 0.01, 2.0)])
E_L10 = np.arange(1.0, 400.0 + 0.01, 2.0)
CDIR = os.path.join(OUT, 'cache'); os.makedirs(CDIR, exist_ok=True)
def cached(key, fn):
    """per-item cache so that repeated foreground runs (each < 600 s) accumulate the WimPyDD results"""
    f = os.path.join(CDIR, key + '.npy')
    if os.path.exists(f) and '--recompute' not in sys.argv:
        return np.load(f)
    say(f'  WimPyDD: computing {key} ...')
    r = np.clip(np.nan_to_num(fn()), 0, None); np.save(f, r); return r
r366 = cached('O1s_1000GeV_delta366_Efull', lambda: lz.wd_rate(ham_o1, 1000.0, E_FULL, halo=HALO, delta_kev=366.0))
NUNIT = {}
for d in [0, 50, 100, 150, 200, 250]:
    NUNIT[d] = dict(E=E_UNIT, r=cached(f'O1s_1000GeV_delta{d}_Eunit', lambda d=d: lz.wd_rate(ham_o1, 1000.0, E_UNIT, halo=HALO, delta_kev=float(d))))
# L10 spectra per mass: taken from P012 (WimPyDD, d10 = 1, Sun-frame halo, 1-339 keV step 2) instead of recomputing
# (the q-dependent L10 Hamiltonian costs ~1 s per WimPyDD call; a 200 GeV spectrum computed here before the run was
#  stopped is kept only as a shape cross-check against P012)
p012npz = np.load('output/work/P012/P012_L10_spectra_d10_1.npz')
L10_MASSES = [100.0, 200.0, 400.0, 1000.0, 4000.0]
L10SPEC = {m: dict(E=p012npz['E'], r=np.clip(p012npz[f'm{m:.0f}'], 0, None)) for m in L10_MASSES}
own200 = os.path.join(CDIR, 'L10_200GeV_EL10.npy')
if os.path.exists(own200):
    o = np.load(own200); Ecommon = np.arange(101.0, 300.0, 2.0)
    ratio = np.interp(Ecommon, E_L10, o) / np.interp(Ecommon, p012npz['E'], p012npz['m200'])
    say(f'own L10 200 GeV / P012 (101-299 keV): {ratio.min():.4f}-{ratio.max():.4f} (constant ratio = same shape; absolute factor is the coupling normalisation)')

def norm_to_lz(E, r):
    """scale a true-energy spectrum so that 2.84 t yr x int r eff_LZ dE = 1.0 event"""
    return r / (E_LZ * np.trapezoid(r * eff_lz(E), E))
SPEC['O1 inel. 366'] = (E_FULL, norm_to_lz(E_FULL, r366))
MODELS = ['L10 1 TeV', 'O1 inel. 250', 'O1 inel. 300', 'O1 inel. 350', 'O1 inel. 366']

# unit-coupling counts (2.84 t yr, LZ efficiency) and check vs P035 (Sun frame)
p035 = pd.read_csv('output/work/P035/Nunit_table.csv'); p035 = p035[p035.m_GeV == 1000].set_index('delta_keV')
nunit_rows = []
for d, v in NUNIT.items():
    N = E_LZ * np.trapezoid(v['r'] * eff_lz(v['E']), v['E'])
    ref = float(p035.loc[d, 'Nunit_LZ_LZlike']) if d in p035.index else np.nan
    nunit_rows.append(dict(delta_keV=d, N_unit_2p84tyr=N, kappa_1event=1.0 / N, P035_Nunit=ref, ratio_to_P035=N / ref if ref == ref else np.nan, source='this work'))
for d in [300, 350]:
    ref = float(p035.loc[d, 'Nunit_LZ_LZlike']); nunit_rows.append(dict(delta_keV=d, N_unit_2p84tyr=ref, kappa_1event=1.0 / ref, P035_Nunit=ref, ratio_to_P035=1.0, source='P035'))
N366 = E_LZ * np.trapezoid(r366 * eff_lz(E_FULL), E_FULL)
N366_p035 = float(10 ** np.interp(366.0, [365.0, 370.0], np.log10([p035.loc[365.0, 'Nunit_LZ_LZlike'], p035.loc[370.0, 'Nunit_LZ_LZlike']])))
nunit_rows.append(dict(delta_keV=366, N_unit_2p84tyr=N366, kappa_1event=1.0 / N366, P035_Nunit=N366_p035, ratio_to_P035=N366 / N366_p035, source='this work (P035 log-interpolated 365/370)'))
NUNIT_T = pd.DataFrame(nunit_rows); NUNIT_T.to_csv(os.path.join(OUT, 'P059_Nunit_O1_1000GeV.csv'), index=False)
say('N_unit(delta):\n' + NUNIT_T.to_string(index=False))

# L10 unit normalisation (P012): N(d10 = 1) per 2.84 t yr, LZ (Fig. 1) normalisation and WimPyDD/Anand normalisation
p012 = pd.read_csv('output/work/P012/P012_coupling_table.csv').groupby('m_GeV').first()
L10_NUNIT_LZ = {m: float(p012.loc[m, 'N_unit_lz']) for m in [100, 200, 400, 1000, 4000]}
L10_NUNIT_WD = {m: float(p012.loc[m, 'N_unit_wd']) for m in [100, 200, 400, 1000, 4000]}
# own L10 unit count check at 1000 GeV (Anand 4/m_M^2 with m_M = m_N, d10 = 1/m_v^2 -> WimPyDD c^0 = 2 d10 x 4/m_v^2 ... only the ratio matters)
own_L10_1000 = E_LZ * np.trapezoid(L10SPEC[1000.0]['r'] * eff_lz(L10SPEC[1000.0]['E']), L10SPEC[1000.0]['E'])
say(f'own L10 unit-count (WimPyDD raw, 1000 GeV) {own_L10_1000:.4g}; P012 N_unit_wd 3.336 -> ratio {own_L10_1000 / 3.336:.3f} (coupling convention factors only)')

# =============================================================================================
# Part C: expected counts in the archival exposures at the LZ best fit
# =============================================================================================
def count(E, s, expo, effX):
    return expo * np.trapezoid(s * effX, E)

rows = []
for mk in MODELS:
    E, s = SPEC[mk]
    for xk, cfg in ARCH.items():
        n_lzlike = cfg['expo_tyr'] / E_LZ
        n_act = count(E, s, cfg['expo_tyr'], eff_arch(E, cfg))
        r = dict(model=mk, experiment=xk, expo_tyr=cfg['expo_tyr'], N_lzlike=n_lzlike, P0_lzlike=math.exp(-n_lzlike),
                 N_actual=n_act, P0_actual=math.exp(-n_act), N_actual_lo68=n_act * BAND[0], N_actual_hi68=n_act * BAND[2],
                 acceptance_ratio=n_act / n_lzlike)
        for key, vals in cfg['variants'].items():
            for val in vals:
                r[f'N_{key}={val:g}'] = count(E, s, cfg['expo_tyr'], eff_arch(E, cfg, **{key: val}))
        rows.append(r)
CNT = pd.DataFrame(rows); CNT.to_csv(os.path.join(OUT, 'P059_archival_counts.csv'), index=False, float_format='%.5g')
say('\nExpected counts at the LZ best fit (1.0 event / 2.84 t yr):\n' +
    CNT[['model', 'experiment', 'expo_tyr', 'N_lzlike', 'N_actual', 'P0_actual', 'acceptance_ratio']].to_string(index=False))
TOT = CNT.groupby('model')[['N_lzlike', 'N_actual']].sum()
TOT['P0_lzlike'] = np.exp(-TOT.N_lzlike); TOT['P0_actual'] = np.exp(-TOT.N_actual)
TOT['N_actual_excl_SR1'] = CNT[CNT.experiment != 'LZ SR1 (2023 NREFT)'].groupby('model').N_actual.sum()
TOT.to_csv(os.path.join(OUT, 'P059_archival_totals.csv'), float_format='%.5g')
say('\nTotals over the four archival exposures:\n' + TOT.to_string())

# =============================================================================================
# Part D: combined Poisson likelihood  LZ (n=1) + archival (n=0)
# =============================================================================================
def nll(s, expos, ns, bs):
    expos, ns, bs = (np.asarray(v, float) for v in (expos, ns, bs))
    keep = (expos > 0) | (bs > 0)                 # terms with zero acceptance and zero background contribute nothing
    mu = s * expos[keep] + bs[keep]
    with np.errstate(divide='ignore', invalid='ignore'):
        return float(np.sum(mu - np.where(ns[keep] > 0, ns[keep] * np.log(mu), 0.0)))

def fit(expos, ns, bs, cl_delta=2.706):
    """s = signal rate in LZ-ROI-equivalent events per t yr.  Returns s_hat, two-sided 90% interval (Delta(-2lnL)=2.706),
    one-sided 90% UL (Delta = 1.642), q0 and Z."""
    f = lambda s: nll(s, expos, ns, bs)
    res = optimize.minimize_scalar(f, bounds=(1e-9, 20.0), method='bounded', options=dict(xatol=1e-10))
    sh, fmin = res.x, res.fun
    def bound(target, lo, hi):
        g = lambda s: 2.0 * (f(s) - fmin) - target
        return optimize.brentq(g, lo, hi) if g(lo) * g(hi) < 0 else (lo if g(lo) > 0 else hi)
    lo90 = bound(cl_delta, 1e-12, sh) if 2 * (f(1e-12) - fmin) > cl_delta else 0.0
    hi90 = bound(cl_delta, sh, 50.0)
    ul90 = bound(1.642, sh, 50.0)
    q0 = 2.0 * (f(1e-12) - fmin)
    return dict(s_hat=sh, lo90=lo90, hi90=hi90, ul90_onesided=ul90, q0=q0, Z=math.sqrt(max(q0, 0.0)))

B_LZ = {'neighbourhood 2e-4': 2e-4, 'P016 anchor 5.7e-4': 5.7e-4, 'panel 0.0106': 0.0106}
comb_rows = []
for bname, b in B_LZ.items():
    base = fit([E_LZ], [1], [b])
    for mk in MODELS:
        sub = CNT[CNT.model == mk]
        for scen, col in [('LZ-like acceptance', 'N_lzlike'), ('actual ROIs', 'N_actual')]:
            e_eff = (E_LZ * sub[col]).values           # LZ-equivalent exposures
            b_arch = b * e_eff / E_LZ                  # background scaled with LZ-equivalent exposure (tiny)
            r = fit([E_LZ] + list(e_eff), [1] + [0] * len(e_eff), [b] + list(b_arch))
            comb_rows.append(dict(b_LZ=bname, model=mk, scenario=scen, E_eff_arch_tyr=float(e_eff.sum()),
                                  s_hat_LZ=base['s_hat'], s_hat=r['s_hat'], s_ratio=r['s_hat'] / base['s_hat'],
                                  lo90_LZ=base['lo90'], hi90_LZ=base['hi90'], lo90=r['lo90'], hi90=r['hi90'],
                                  ul90_LZ=base['ul90_onesided'], ul90=r['ul90_onesided'],
                                  Z_LZ=base['Z'], Z=r['Z'], dZ=r['Z'] - base['Z'],
                                  N_events_LZ_hat=r['s_hat'] * E_LZ))
COMB = pd.DataFrame(comb_rows); COMB.to_csv(os.path.join(OUT, 'P059_combined_likelihood.csv'), index=False, float_format='%.5g')
say('\nCombined likelihood (b_LZ = 2e-4):\n' + COMB[COMB.b_LZ == 'neighbourhood 2e-4'][
    ['model', 'scenario', 'E_eff_arch_tyr', 's_hat_LZ', 's_hat', 's_ratio', 'lo90', 'hi90', 'ul90_LZ', 'ul90', 'Z_LZ', 'Z', 'dZ']].to_string(index=False))

# exposure-only illustration: what if every archival t yr had LZ-like acceptance (E_arch = 2.14 t yr)
ILL = {}
for bname, b in B_LZ.items():
    base = fit([E_LZ], [1], [b]); allarch = fit([E_LZ, E_ARCH_TOTAL], [1, 0], [b, b * E_ARCH_TOTAL / E_LZ])
    sr1 = fit([E_LZ, ARCH['LZ SR1 (2023 NREFT)']['expo_tyr']], [1, 0], [b, b * ARCH['LZ SR1 (2023 NREFT)']['expo_tyr'] / E_LZ])
    ILL[bname] = dict(LZ_alone=base, LZ_plus_all_archival_LZlike=allarch, LZ_plus_SR1_LZlike=sr1)
say('\nIllustration (b=2e-4): LZ alone', {k: round(v, 4) for k, v in ILL['neighbourhood 2e-4']['LZ_alone'].items()})
say('  + all archival LZ-like', {k: round(v, 4) for k, v in ILL['neighbourhood 2e-4']['LZ_plus_all_archival_LZlike'].items()})

# =============================================================================================
# Part E: Fig. 6 previous-limit curves (violet LZ 2024a/b, red LUX 2021, blue PandaX-II 2019)
# =============================================================================================
import pymupdf
page = pymupdf.open('inputs/arXiv_2609.02823_source/Fig6_O1_L10_limit_stacked.pdf')[0]
DR = page.get_drawings()
def vertices(idx):
    pts = []
    for it in DR[idx]['items']:
        if it[0] == 'l':
            pts += [(it[1].x, it[1].y), (it[2].x, it[2].y)]
    return np.array(sorted(set((round(x, 3), round(y, 3)) for x, y in pts)))

# top panel calibration (P007): x 0 keV -> 66.7 pt, 350 keV -> 487.2 pt; decade ticks (y_pt, log10 kappa)
TOP_Y = [(270.2, -9), (236.2, -8), (202.1, -7), (168.2, -6), (134.1, -5), (100.2, -4), (66.2, -3), (32.2, -2)]
sy, iy = np.polyfit([p[0] for p in TOP_Y], [p[1] for p in TOP_Y], 1)
top_x = lambda x: (x - 66.7) / (487.2 - 66.7) * 350.0
top_y = lambda y: 10 ** (sy * y + iy)
# bottom panel calibration (P012): decade ticks y 355.46 (1e2) ... 633.18 (1e-1); x from the 13 upper-curve vertices vs Table S6 masses
BOT_TICKS = [355.46, 448.04, 540.61, 633.18]
sb, ib = np.polyfit(BOT_TICKS, [2, 1, 0, -1], 1)
p012dig = pd.read_csv('output/work/P012/fig6_bottom_digitised.csv')
sx, ix = np.polyfit(p012dig.x_pt, np.log10(p012dig.m_GeV), 1)
bot_x = lambda x: 10 ** (sx * x + ix)
bot_y = lambda y: 10 ** (sb * y + ib)

v235, v236, v347, v348 = vertices(235), vertices(236), vertices(347), vertices(348)
LZ24a = pd.DataFrame(dict(delta_keV=top_x(v235[:, 0]), kappa_lim=top_y(v235[:, 1])))
LUX21 = pd.DataFrame(dict(delta_keV=top_x(v236[:, 0]), kappa_lim=top_y(v236[:, 1])))
LZ24b = pd.DataFrame(dict(m_GeV=bot_x(v347[:, 0]), d10_lim=bot_y(v347[:, 1])))
inframe = (v348[:, 0] >= 66.7) & (v348[:, 0] <= 487.2) & (v348[:, 1] >= 351.2) & (v348[:, 1] <= 637.9)
PX2 = pd.DataFrame(dict(m_GeV=bot_x(v348[inframe, 0]), d10_lim=bot_y(v348[inframe, 1])))
for nm, df in [('LZ2024a_O1_inelastic', LZ24a), ('LUX2021_O1_inelastic', LUX21), ('LZ2024b_L10', LZ24b), ('PandaXII2019_L10', PX2)]:
    df.to_csv(os.path.join(OUT, f'fig6_digitised_{nm}.csv'), index=False, float_format='%.5g')
say('\nLZ 2024a (violet, top):\n' + LZ24a.to_string(index=False)); say('LUX 2021 (red, top):\n' + LUX21.to_string(index=False))
say('LZ 2024b (violet, bottom):\n' + LZ24b.to_string(index=False)); say('PandaX-II 2019 (blue, bottom, in frame): %d vertices, %.1f-%.0f GeV, d10 %.2f-%.1f' % (len(PX2), PX2.m_GeV.min(), PX2.m_GeV.max(), PX2.d10_lim.min(), PX2.d10_lim.max()))

# --- E1: O1 inelastic panel: implied SR1 counts at the SR1 limit, and where the new best fit sits ---
p007 = json.load(open('output/work/P007/lz_intervals_digitised.json'))['fig6_top_c1s_mv2_sq']
def interp_log(curve, x):
    xs = np.array([p[0] for p in curve]); ys = np.log10([p[1] for p in curve]); return float(10 ** np.interp(x, xs, ys))
SR1 = ARCH['LZ SR1 (2023 NREFT)']
e1 = []
for _, r in LZ24a.iterrows():
    d = int(round(r.delta_keV / 50.0) * 50); v = NUNIT[d]
    Nu = float(NUNIT_T.set_index('delta_keV').loc[d, 'N_unit_2p84tyr'])
    A_lz = np.trapezoid(v['r'] * eff_lz(v['E']), v['E']); A_sr1 = np.trapezoid(v['r'] * eff_arch(v['E'], SR1), v['E'])
    A_sr1_085 = np.trapezoid(v['r'] * eff_arch(v['E'], SR1, plateau=0.85), v['E'])
    n_lzlike = Nu * r.kappa_lim * SR1['expo_tyr'] / E_LZ
    e1.append(dict(delta_keV=d, kappa_SR1_limit=r.kappa_lim, N_unit=Nu, kappa_hat_new=1.0 / Nu,
                   kappa_upper_new=interp_log(p007['upper'], d), kappa_median_new=interp_log(p007['median'], d),
                   kappa_lower_new=interp_log(p007['lower'], d) if d >= 100 else np.nan,
                   SR1lim_over_kappa_hat=Nu * r.kappa_lim, SR1lim_over_new_upper=r.kappa_lim / interp_log(p007['upper'], d),
                   SR1lim_over_new_median=r.kappa_lim / interp_log(p007['median'], d),
                   N_SR1_at_limit_LZlike=n_lzlike, N_SR1_at_limit_260edge=n_lzlike * A_sr1 / A_lz,
                   N_SR1_at_limit_260edge_0p85=n_lzlike * A_sr1_085 / A_lz, acc_ratio_SR1_over_LZ=A_sr1 / A_lz,
                   N_SR1_at_new_bestfit=SR1['expo_tyr'] / E_LZ * A_sr1 / A_lz))
E1 = pd.DataFrame(e1); E1.to_csv(os.path.join(OUT, 'P059_fig6_top_consistency.csv'), index=False, float_format='%.4g')
say('\nFig. 6 top consistency (LZ 2024a vs new):\n' + E1[['delta_keV', 'kappa_SR1_limit', 'kappa_hat_new', 'SR1lim_over_kappa_hat', 'SR1lim_over_new_upper',
                                                          'SR1lim_over_new_median', 'N_SR1_at_limit_LZlike', 'N_SR1_at_limit_260edge', 'N_SR1_at_new_bestfit']].to_string(index=False))
# LUX 2021: implied LZ-equivalent effective exposure  E_eff = 2.3 x 2.84 / (N_unit kappa_lim)
lux = []
for _, r in LUX21.iterrows():
    d = int(round(r.delta_keV / 50.0) * 50); Nu = float(NUNIT_T.set_index('delta_keV').loc[d, 'N_unit_2p84tyr'])
    lux.append(dict(delta_keV=d, kappa_LUX_limit=r.kappa_lim, LUXlim_over_kappa_hat=Nu * r.kappa_lim,
                    E_eff_LZequiv_tyr_n0=2.303 * E_LZ / (Nu * r.kappa_lim), E_eff_over_nominal=2.303 * E_LZ / (Nu * r.kappa_lim) / ARCH['LUX 2021 EFT']['expo_tyr'],
                    N_LUX_at_limit_if_LZlike=Nu * r.kappa_lim * ARCH['LUX 2021 EFT']['expo_tyr'] / E_LZ))
LUXT = pd.DataFrame(lux); LUXT.to_csv(os.path.join(OUT, 'P059_fig6_top_LUX.csv'), index=False, float_format='%.4g')
say('\nLUX 2021 implied effective exposure:\n' + LUXT.to_string(index=False))

# --- E2: L10 panel: SR1 (LZ 2024b) implied counts and PandaX-II implied effective exposure ---
def l10_fracs(m):
    E, r = L10SPEC[m]['E'], L10SPEC[m]['r']
    A_lz = np.trapezoid(r * eff_lz(E), E)
    return dict(A_lz=A_lz, A_sr1=np.trapezoid(r * eff_arch(E, SR1), E),
                f_below_50=np.trapezoid(r * (E < 50), E) / np.trapezoid(r, E), f_below_100=np.trapezoid(r * (E < 100), E) / np.trapezoid(r, E),
                f_lzroi=A_lz / 0.96 / np.trapezoid(r, E))
e2 = []
for m in [int(x) for x in L10_MASSES]:
    fr = l10_fracs(float(m))
    d_sr1 = float(10 ** np.interp(np.log10(m), np.log10(LZ24b.m_GeV), np.log10(LZ24b.d10_lim)))
    d_px = float(10 ** np.interp(np.log10(m), np.log10(PX2.m_GeV), np.log10(PX2.d10_lim)))
    row = dict(m_GeV=m, d10_SR1_limit=d_sr1, d10_PandaXII_limit=d_px, d10_new_upper=float(p012dig.set_index('m_GeV').loc[m, 'd10_upper']),
               d10_new_median=float(p012dig.set_index('m_GeV').loc[m, 'd10_median_sens']), d10_new_bestfit=float(p012.loc[m, 'd10_lz_bestfit']),
               N_unit_lz=L10_NUNIT_LZ[m], N_unit_wd=L10_NUNIT_WD[m], acc_ratio_SR1=fr['A_sr1'] / fr['A_lz'],
               f_below_50=fr['f_below_50'], f_below_100=fr['f_below_100'])
    for tag, Nu in [('lz', L10_NUNIT_LZ[m]), ('wd', L10_NUNIT_WD[m])]:
        row[f'N_SR1_at_limit_{tag}'] = Nu * d_sr1**2 * SR1['expo_tyr'] / E_LZ * fr['A_sr1'] / fr['A_lz']
        row[f'E_eff_PandaX_tyr_{tag}'] = 2.303 * E_LZ / (Nu * d_px**2)
        row[f'PandaX_Eeff_over_expo_{tag}'] = row[f'E_eff_PandaX_tyr_{tag}'] / ARCH['PandaX-II 2019']['expo_tyr']
    row['SR1lim_sq_over_bestfit_sq'] = d_sr1**2 / row['d10_new_bestfit']**2
    row['SR1lim_over_new_upper'] = d_sr1 / row['d10_new_upper']; row['SR1lim_over_new_median'] = d_sr1 / row['d10_new_median']
    e2.append(row)
E2 = pd.DataFrame(e2); E2.to_csv(os.path.join(OUT, 'P059_fig6_bottom_consistency.csv'), index=False, float_format='%.4g')
say('\nFig. 6 bottom consistency (L10):\n' + E2[['m_GeV', 'd10_SR1_limit', 'd10_new_bestfit', 'SR1lim_sq_over_bestfit_sq', 'SR1lim_over_new_median',
                                                'N_SR1_at_limit_lz', 'N_SR1_at_limit_wd', 'd10_PandaXII_limit', 'E_eff_PandaX_tyr_lz', 'E_eff_PandaX_tyr_wd',
                                                'f_below_50', 'f_below_100']].to_string(index=False))

# =============================================================================================
# Part F: the SR1 look-back and a joint SR1 + SR3 fit
# =============================================================================================
sr1_rows = []
for mk in MODELS:
    E, s = SPEC[mk]
    for e50 in [250.0, SR1['E50_hi'], 269.9]:
        for pl in [0.96, 0.85]:
            n = count(E, s, SR1['expo_tyr'], eff_arch(E, SR1, E50_hi=e50, plateau=pl))
            sr1_rows.append(dict(model=mk, E50_hi=e50, plateau=pl, N_SR1_bestfit=n, N_SR1_lo68=n * BAND[0], N_SR1_hi68=n * BAND[2],
                                 P0=math.exp(-n), P_ge1=1 - math.exp(-n), P0_hi68=math.exp(-n * BAND[2])))
SR1T = pd.DataFrame(sr1_rows); SR1T.to_csv(os.path.join(OUT, 'P059_SR1_lookback.csv'), index=False, float_format='%.4g')
say('\nSR1 look-back (baseline rows):\n' + SR1T[(SR1T.plateau == 0.96) & (np.isclose(SR1T.E50_hi, SR1['E50_hi']))].to_string(index=False))
# joint SR1+SR3 fit for n_SR1 = 0 or 1, per model (actual SR1 acceptance), b = 2e-4 (scaled)
joint = []
for mk in MODELS:
    n_sr1 = float(SR1T[(SR1T.model == mk) & (SR1T.plateau == 0.96) & np.isclose(SR1T.E50_hi, SR1['E50_hi'])].N_SR1_bestfit.iloc[0])
    e_eff = E_LZ * n_sr1
    for b in [2e-4, 0.0106]:
        base = fit([E_LZ], [1], [b])
        for nobs in [0, 1]:
            r = fit([E_LZ, e_eff], [1, nobs], [b, b * e_eff / E_LZ])
            joint.append(dict(model=mk, b_LZ=b, E_eff_SR1_tyr=e_eff, n_SR1=nobs, s_hat_LZ=base['s_hat'], s_hat=r['s_hat'], s_ratio=r['s_hat'] / base['s_hat'],
                              lo90=r['lo90'], hi90=r['hi90'], hi90_LZ=base['hi90'], Z_LZ=base['Z'], Z=r['Z'], dZ=r['Z'] - base['Z']))
JOINT = pd.DataFrame(joint); JOINT.to_csv(os.path.join(OUT, 'P059_joint_SR1_SR3.csv'), index=False, float_format='%.4g')
say('\nJoint SR1+SR3 (b=2e-4):\n' + JOINT[JOINT.b_LZ == 2e-4].to_string(index=False))

# =============================================================================================
# Part G: figures
# =============================================================================================
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10})
# Fig 1: counts per archival exposure
fig, ax = plt.subplots(figsize=(7.2, 3.8))
exps = list(ARCH.keys()); w = 0.16; x = np.arange(len(exps))
cols = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
for i, mk in enumerate(MODELS):
    sub = CNT[CNT.model == mk].set_index('experiment').loc[exps]
    ax.bar(x + (i - 2) * w, np.maximum(sub.N_actual.values, 1e-4), w, color=cols[i], label=mk)
    ax.scatter(x + (i - 2) * w, sub.N_lzlike.values, marker='_', color='k', s=60, zorder=3)
ax.set_yscale('log'); ax.set_ylim(1e-4, 3); ax.set_xticks(x); ax.set_xticklabels([e.replace(' (', '\n(') for e in exps])
ax.set_ylabel('expected events at LZ best fit (1.0 / 2.84 t yr)')
ax.axhline(2.303, ls=':', color='grey'); ax.text(-0.4, 1.55, '2.3 events: 90% CL zero-event ceiling', ha='left', fontsize=8, color='grey')
ax.set_ylim(1e-4, 8)
ax.legend(fontsize=7.5, ncol=3, loc='upper right', title='bars: actual ROI; black ticks: LZ-like acceptance', title_fontsize=7.5)
ax.set_title('P059: what the archival xenon exposures should have contained')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P059_fig1_archival_counts.png'), dpi=160); plt.close(fig)

# Fig 2: combined likelihood profiles (L10 and delta=366 shown), b = 2e-4
fig, ax = plt.subplots(figsize=(6.4, 3.8))
sgrid = np.linspace(0.005, 2.0, 800); b = 2e-4
base = np.array([nll(s, [E_LZ], [1], [b]) for s in sgrid]); ax.plot(sgrid, 2 * (base - base.min()), 'k', label='LZ alone (1 event, 2.84 t yr)')
for mk, c in [('L10 1 TeV', cols[0]), ('O1 inel. 366', cols[4])]:
    sub = CNT[CNT.model == mk]; e_eff = (E_LZ * sub.N_actual).values
    prof = np.array([nll(s, [E_LZ] + list(e_eff), [1] + [0] * 4, [b] + list(b * e_eff / E_LZ)) for s in sgrid])
    ax.plot(sgrid, 2 * (prof - prof.min()), color=c, label=f'+ archival (actual ROIs), {mk}: E_eff = {e_eff.sum():.2f} t yr')
prof = np.array([nll(s, [E_LZ, E_ARCH_TOTAL], [1, 0], [b, b * E_ARCH_TOTAL / E_LZ]) for s in sgrid])
ax.plot(sgrid, 2 * (prof - prof.min()), '--', color='grey', label=f'+ all 2.14 t yr with LZ-like acceptance')
ax.axhline(2.706, ls=':', color='grey'); ax.text(1.95, 2.85, '90% two-sided', ha='right', fontsize=8, color='grey')
ax.set_ylim(0, 8); ax.set_xlim(0, 2); ax.set_xlabel('signal rate s [LZ-window events per t yr]'); ax.set_ylabel(r'$\Delta(-2\ln L)$')
ax.legend(fontsize=7.5); ax.set_title('P059: adding the zero-event archival exposures')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P059_fig2_combined_profile.png'), dpi=160); plt.close(fig)

# Fig 3: Fig. 6 top reconstruction with implied SR1 counts
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 6.2), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
dd = np.array([p[0] for p in p007['upper']]); a1.plot(dd, [p[1] for p in p007['upper']], 'k', lw=2, label='LZ 2026 90% upper (digitised, P007)')
a1.plot([p[0] for p in p007['lower']], [p[1] for p in p007['lower']], 'k', lw=2, label='LZ 2026 lower')
a1.plot([p[0] for p in p007['median']], [p[1] for p in p007['median']], 'k--', label='LZ 2026 median sensitivity')
a1.plot(LZ24a.delta_keV, LZ24a.kappa_lim, color='#8b00d3', label='LZ 2024a (SR1) limit, digitised')
a1.plot(LUX21.delta_keV, LUX21.kappa_lim, color='#dc143c', label='LUX 2021 limit, digitised')
nt = NUNIT_T.sort_values('delta_keV')
a1.plot(nt.delta_keV, nt.kappa_1event, 'o-', color='#2ca02c', ms=4, label=r'new best fit $\hat\kappa$ = 1/N$_{unit}$ (this work; 300/350 from P035)')
a1.set_yscale('log'); a1.set_ylim(1e-10, 3.0); a1.set_ylabel(r'$(c_1^s m_v^2)^2$'); a1.legend(fontsize=7, loc='lower right')
a1.set_title('P059: Fig. 6 top, previous limits versus the one-event fit (1000 GeV)')
a2.plot(E1.delta_keV, E1.N_SR1_at_limit_LZlike, 's-', color='#8b00d3', label='SR1 events at the SR1 limit (LZ-like acceptance)')
a2.plot(E1.delta_keV, E1.N_SR1_at_limit_260edge, 's--', color='#8b00d3', mfc='none', label='same, 260 keV edge')
a2.plot(E1.delta_keV, E1.N_SR1_at_new_bestfit, 'o-', color='#2ca02c', ms=4, label='SR1 events at the new best fit')
a2.axhspan(2.3, 3.3, color='grey', alpha=0.25, label='2.3-3.3 events (zero-event 90% CL)')
a2.set_xlabel('mass splitting [keV]'); a2.set_ylabel('events in 0.90 t yr'); a2.set_ylim(0, 10.5); a2.legend(fontsize=7, loc='upper right')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P059_fig3_fig6top_consistency.png'), dpi=160); plt.close(fig)

# =============================================================================================
# Summary JSON
# =============================================================================================
b2 = COMB[(COMB.b_LZ == 'neighbourhood 2e-4')]
summary = dict(
    exposures_tyr={k: v['expo_tyr'] for k, v in ARCH.items()}, archival_total_tyr=E_ARCH_TOTAL,
    LZ_bestfit_rate_per_tyr=1.0 / E_LZ, band_events=BAND,
    nunit_O1_1000GeV=NUNIT_T.to_dict(orient='records'),
    counts=CNT.to_dict(orient='records'), totals=TOT.reset_index().to_dict(orient='records'),
    combined_b2em4=b2.to_dict(orient='records'), illustration=ILL,
    fig6_top=E1.to_dict(orient='records'), fig6_LUX=LUXT.to_dict(orient='records'), fig6_bottom=E2.to_dict(orient='records'),
    sr1_lookback=SR1T.to_dict(orient='records'), joint=JOINT.to_dict(orient='records'),
    digitised=dict(LZ2024a=LZ24a.to_dict(orient='records'), LUX2021=LUX21.to_dict(orient='records'), LZ2024b=LZ24b.to_dict(orient='records'),
                   PandaXII_n=int(len(PX2)), PandaXII_range=[float(PX2.m_GeV.min()), float(PX2.m_GeV.max())]),
    settings=dict(eff_model='0.96 x erf_up(5.4 keV, 2.5) x [1 - erf_up(E50_hi, 11.5)]', SR1_E50_hi=SR1['E50_hi'], halo='Sun-frame Baxter SHM (WimPyDD default day=None)',
                  spectra='P005 spectra_normalised_LZ1.csv (L10, delta 250/300/350) + own WimPyDD delta=366 keV; 1000 GeV'))
json.dump(summary, open(os.path.join(OUT, 'P059_summary.json'), 'w'), indent=1, default=float)
say('\nDone. Summary written to', os.path.join(OUT, 'P059_summary.json'))
LOG.close()
