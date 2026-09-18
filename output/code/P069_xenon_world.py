"""
P069 -- The xenon world in one likelihood: exposure ledger, hidden-event table, joint Poisson scenarios,
coordinated-reanalysis protocol and timeline for LZ + XENONnT + PandaX-4T (+ archival LUX/PandaX-II/XENON1T).

Run from the simulation root:   .venv/bin/python output/code/P069_xenon_world.py

Inputs (all local): P050's WimPyDD spectra cache (1 TeV; L10 and inelastic O1 at delta = 300/350/366/380 keV,
annual-average Baxter halo, unit coupling), lzcommon constants, P001/P020 posterior framework, P034 timing
requirements, P016/P050 background rates.  Exposures of the other experiments are recalled (flags in LEDGER).
Outputs -> output/work/P069/ (CSV/JSON, run_log.txt) and output/work/P069/figures/ (PNG).
"""
from __future__ import annotations
import os, sys, math, json, time
import numpy as np
from scipy import stats, special, integrate, optimize
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, 'output/code')
from common import lzcommon as lz

WORK = 'output/work/P069'; FIG = WORK + '/figures'
os.makedirs(FIG, exist_ok=True)
LOG = open(WORK + '/run_log.txt', 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.write(s + '\n'); LOG.flush()

RES = {}
EXPO_LZ = lz.LZ['exposure_tyr']          # 2.84 t yr
S_BEST, S_LO68, S_HI68 = 1.0, 0.30, 2.36  # LZ Table I best fit; Poisson n=1 68% band (P020: [0.30, 2.36])
S_LO_FC = 0.105                            # Feldman-Cousins single-event lower limit (P021/P050)
B_PER284 = {'nb2sig': 2e-4, 'band200_270': 5.7e-4, 'panel': 0.0106}   # backgrounds per 2.84 t yr (P001/P016/LZ Fig. 5)
T_NOW = 2026 + (245 / 365.25)              # 2 Sep 2026 (LZ posting date) = 2026.67
T_END26, T_MID27, T_END27, T_END28 = 2027.0, 2027.5, 2028.0, 2029.0

say('P069 xenon-world ledger --', time.strftime('%Y-%m-%d %H:%M'))

# ==============================================================================================
# 1. Exposure ledger (t yr).  status: analysed = a published WIMP search exists for that data;
#    untouched = no published analysis.  HE = high-energy (200-270 keV NR) analysis exists?  Only LZ SR3.
#    start/end are decimal years for the timeline (recalled; 'likely' unless flagged).
# ==============================================================================================
LEDGER = [
 # name, experiment, exposure adopted, range (lo, hi), fiducial t, live frac, ROI edge keV (published), status, start, end, reliability note
 dict(name='LUX 2013-16', exp='LUX', E=0.092, Elo=0.092, Ehi=0.092, M=0.1, live=None, edge=50.0, status='analysed (archival)',
      t0=2013.3, t1=2016.4, rel='exposure 3.35e4 kg d (likely); ROI edge ~50 keV (uncertain)'),
 dict(name='PandaX-II 2016-18', exp='PandaX-II', E=0.36, Elo=0.36, Ehi=0.36, M=0.33, live=None, edge=50.0, status='analysed (archival)',
      t0=2016.2, t1=2018.6, rel='exposure 132 t d (likely); ROI edge ~50 keV (uncertain)'),
 dict(name='XENON1T 2018', exp='XENON1T', E=1.0, Elo=1.0, Ehi=1.0, M=1.3, live=None, edge=41.0, status='analysed (archival)',
      t0=2016.9, t1=2018.1, rel='1.0 t yr (certain); cS1 < 70 PE ~ 41 keV NR (likely)'),
 dict(name='LZ SR1 2022', exp='LZ', E=0.77, Elo=0.77, Ehi=0.90, M=4.71, live=None, edge=70.0, status='analysed, HE untouched',
      t0=2021.98, t1=2022.35, rel='60 live d x 5.5 t = 0.90 t yr (certain); 0.77 with the SR3 4.71 t FV; ROI ~ 70 keV (likely)'),
 dict(name='LZ SR3 (WS2024) analysed', exp='LZ', E=2.84, Elo=2.84, Ehi=2.84, M=4.71, live=0.593, edge=270.0, status='analysed to 270 keV (this result)',
      t0=2023.23, t1=2024.25, rel='paper (certain)'),
 dict(name='LZ untouched since Apr 2024', exp='LZ', E=6.76, Elo=5.7, Ehi=8.55, M=4.71, live=0.593, edge=None, status='untouched',
      t0=2024.25, t1=T_NOW, rel='P020: 884 cal d x 0.593 x 4.71 t; live fraction +-25%'),
 dict(name='XENONnT SR0+SR1 analysed', exp='XENONnT', E=3.1, Elo=3.1, Ehi=3.1, M=4.0, live=None, edge=60.0, status='analysed, HE untouched',
      t0=2021.5, t1=2023.6, rel='3.1 t yr (certain, cited by LZ); ROI cS1 < 100 PE ~ 60 keV (uncertain)'),
 dict(name='XENONnT untouched (SR2+)', exp='XENONnT', E=3.0, Elo=2.0, Ehi=5.0, M=4.0, live=0.25, edge=None, status='untouched (recalled)',
      t0=2023.6, t1=T_NOW, rel='UNCERTAIN: XENONnT continued after SR1 with long stops (neutron-veto Gd loading); 3.0 yr x 4.0 t x eff. live 0.17-0.42'),
 dict(name='PandaX-4T Run0+Run1 analysed', exp='PandaX-4T', E=1.54, Elo=1.54, Ehi=1.54, M=2.7, live=None, edge=100.0, status='analysed, HE untouched',
      t0=2020.9, t1=2022.4, rel='1.54 t yr (certain, cited by LZ); ROI edge ~100 keV (uncertain)'),
 dict(name='PandaX-4T untouched (Run2/3)', exp='PandaX-4T', E=2.5, Elo=1.5, Ehi=3.5, M=2.7, live=0.33, edge=None, status='untouched (recalled)',
      t0=2023.9, t1=T_NOW, rel='UNCERTAIN: running again since late 2023 after upgrade; 2.8 yr x 2.7 t x eff. live 0.2-0.46'),
]
# future accrual (t yr per calendar yr) from T_NOW to end-2028: LZ 4.71 x 0.593 (P020); XENONnT 4.0 x 0.6, PandaX-4T 2.7 x 0.6 (P050)
FUTURE_RATE = {'LZ': 4.71 * 0.593, 'XENONnT': 4.0 * 0.6, 'PandaX-4T': 2.7 * 0.6}
led = pd.DataFrame(LEDGER)
led['hidden_at_HE'] = led['status'] != 'analysed to 270 keV (this result)'
E_total_now = led['E'].sum()
E_new_core = led.loc[led['exp'].isin(['LZ', 'XENONnT', 'PandaX-4T']) & (led['name'] != 'LZ SR3 (WS2024) analysed') & (led['name'] != 'LZ SR1 2022'), 'E'].sum()
E_new_all = led.loc[led['hidden_at_HE'], 'E'].sum()
E_new_all_lo = led.loc[led['hidden_at_HE'], 'Elo'].sum(); E_new_all_hi = led.loc[led['hidden_at_HE'], 'Ehi'].sum()
say(f'[1] ledger total on disk (all xenon TPCs, to 2 Sep 2026): {E_total_now:.2f} t yr; analysed to 270 keV: {EXPO_LZ}; '
    f'hidden at high energy: {E_new_all:.2f} (range {E_new_all_lo:.2f}-{E_new_all_hi:.2f}); core running-three (excl. archival & SR1): {E_new_core:.2f}')
RES['exposure'] = dict(total_on_disk=E_total_now, analysed_HE=EXPO_LZ, hidden_HE=E_new_all, hidden_HE_lo=E_new_all_lo, hidden_HE_hi=E_new_all_hi,
                       core_running_three=E_new_core)

# ==============================================================================================
# 2. Spectra (P050 cache; WimPyDD, 1 TeV, annual Baxter halo, unit coupling) and efficiencies
# ==============================================================================================
CACHE = 'output/work/P050/P050_spectra_cache.npz'
SP = dict(np.load(CACHE)); say('[2] loaded', CACHE, 'deltas', SP['deltas'])
INEL = {float(d): np.clip(SP['inel'][i], 0, None) for i, d in enumerate(SP['deltas'])}
E_IN, E_L10, L10 = SP['E_in'], SP['E_l10'], np.clip(SP['l10'], 0, None)
MODELS = {'L10': (E_L10, L10)}
for d in (300.0, 350.0, 366.0, 380.0):
    MODELS[f'd{int(d)}'] = (E_IN, INEL[d])
LABELS = {'L10': 'L10 (1 TeV)', 'd300': 'inelastic 300 keV', 'd350': 'inelastic 350 keV', 'd366': 'inelastic 366 keV', 'd380': 'inelastic 380 keV'}

PLATEAU_LZ = 0.955
def eff_erf(E, E50_hi, sig_hi, plateau=PLATEAU_LZ, E50_lo=5.4, sig_lo=2.5):
    E = np.asarray(E, float)
    lo = 0.5 * (1 + special.erf((E - E50_lo) / (math.sqrt(2) * sig_lo)))
    hi = 0.5 * (1 - special.erf((E - E50_hi) / (math.sqrt(2) * sig_hi)))
    return plateau * lo * hi
def eff_LZ(E):  return eff_erf(E, 269.9, 11.5)                 # LZ Fig. S2 (P021/P050)
def eff_400(E): return eff_erf(E, 400.0, 17.0)                 # extended pre-registered window (P050 'ext400')
def eff_hard(E, edge, plateau=0.9, E50_lo=5.0):                # un-extended published ROI (P005 convention)
    E = np.asarray(E, float)
    lo = 0.5 * (1 + special.erf((E - E50_lo) / (math.sqrt(2) * 2.5)))
    return plateau * lo * (E < edge)

def count_per_tyr(model, weight):
    E, r = MODELS[model]
    return float(np.trapezoid(r * weight(E), E))

NORM = {}
for m in MODELS:
    N_LZ_unit = EXPO_LZ * count_per_tyr(m, eff_LZ)
    kappa = S_BEST / N_LZ_unit                                  # 1.0 event per 2.84 t yr in LZ's ROI
    NORM[m] = dict(N_LZ_unit=N_LZ_unit, kappa=kappa,
                   rate_LZwin=kappa * count_per_tyr(m, eff_LZ),                      # = 1/2.84 by construction
                   rate_200_270=kappa * count_per_tyr(m, lambda E: eff_LZ(E) * (E >= 200) * (E < 270)),
                   rate_150_270=kappa * count_per_tyr(m, lambda E: eff_LZ(E) * (E >= 150) * (E < 270)),
                   rate_ext400=kappa * count_per_tyr(m, eff_400),
                   rate_ext400_gt200=kappa * count_per_tyr(m, lambda E: eff_400(E) * (E >= 200)),
                   rate_full=kappa * count_per_tyr(m, lambda E: np.ones_like(E)))
    NORM[m]['frac_200_270'] = NORM[m]['rate_200_270'] / NORM[m]['rate_LZwin']
    NORM[m]['A_LZ'] = NORM[m]['rate_LZwin'] / NORM[m]['rate_full']
    NORM[m]['A_400'] = NORM[m]['rate_ext400'] / NORM[m]['rate_full']
norm = pd.DataFrame(NORM).T
say('[2] normalisation (best fit, per t yr):\n' + norm[['rate_LZwin', 'rate_200_270', 'frac_200_270', 'rate_ext400', 'rate_ext400_gt200', 'A_LZ', 'A_400']].round(4).to_string())
say('    cross-check vs P050: A_LZ 0.422/0.814/0.444/0.172/0.028; ext400 rates 0.447/0.401/0.687/1.70/10.5 per t yr')
norm.to_csv(WORK + '/P069_normalisation.csv')
RES['normalisation'] = {m: {k: float(v) for k, v in d.items()} for m, d in NORM.items()}

# ==============================================================================================
# 3. Hidden-events table: per dataset and model, expected events (best fit) recorded but not analysed at HE
#    hidden_270  : LZ-like 270 keV window, above the dataset's published ROI edge (all of it if untouched)
#    hidden_200_270 : true recoil energy 200-270 keV, LZ-like efficiency (the '200-270 keV NR-band' count)
#    tested_pub  : what the published ROI (hard edge, 0.9 plateau) already tested (P005 convention)
# ==============================================================================================
rows = []
for _, r in led.iterrows():
    for m in MODELS:
        k = NORM[m]['kappa']
        edge = r['edge']
        if r['status'].startswith('analysed to 270'):
            hid270 = 0.0; hid200 = 0.0; tested = S_BEST
        else:
            lo_edge = edge if edge is not None and not (isinstance(edge, float) and np.isnan(edge)) else 0.0
            hid270 = r['E'] * k * count_per_tyr(m, lambda E: eff_LZ(E) * (E >= lo_edge))
            hid200 = r['E'] * NORM[m]['rate_200_270']
            tested = r['E'] * k * count_per_tyr(m, lambda E: eff_hard(E, lo_edge)) if lo_edge > 0 else 0.0
        ext400 = r['E'] * NORM[m]['rate_ext400_gt200'] if not r['status'].startswith('analysed to 270') else 0.0
        rows.append(dict(dataset=r['name'], experiment=r['exp'], exposure_tyr=r['E'], model=m, published_edge_keV=edge,
                         tested_in_published_ROI=tested, hidden_270=hid270, hidden_200_270=hid200,
                         P_ge1_hidden_270=1 - math.exp(-hid270), hidden_ext400_gt200=ext400,
                         hidden_270_lo68=hid270 * S_LO68, hidden_270_hi68=hid270 * S_HI68))
hid = pd.DataFrame(rows)
hid.to_csv(WORK + '/P069_hidden_events.csv', index=False)
piv = hid.pivot(index='dataset', columns='model', values='hidden_270').reindex(led['name'])
piv200 = hid.pivot(index='dataset', columns='model', values='hidden_200_270').reindex(led['name'])
say('[3] hidden events in an LZ-like 270 keV window above the published edge (best fit):\n' + piv.round(3).to_string())
say('[3] hidden events with true E in 200-270 keV (best fit):\n' + piv200.round(3).to_string())
tot270 = hid.groupby('model')['hidden_270'].sum(); tot200 = hid.groupby('model')['hidden_200_270'].sum()
tested_tot = hid.groupby('model')['tested_in_published_ROI'].sum() - S_BEST
tot400 = hid.groupby('model')['hidden_ext400_gt200'].sum()
say('[3] totals (all hidden datasets): 270-window', tot270.round(3).to_dict(), '; 200-270 keV', tot200.round(3).to_dict(),
    '; already tested in published low-energy ROIs (excl. LZ SR3)', tested_tot.round(3).to_dict(), '; 200-400 keV window', tot400.round(3).to_dict())
# core (running three, excluding archival and LZ SR1)
core_mask = hid['dataset'].isin(['LZ untouched since Apr 2024', 'XENONnT SR0+SR1 analysed', 'XENONnT untouched (SR2+)',
                                 'PandaX-4T Run0+Run1 analysed', 'PandaX-4T untouched (Run2/3)'])
tot270_core = hid[core_mask].groupby('model')['hidden_270'].sum()
say('[3] core running-three totals, 270 window:', tot270_core.round(3).to_dict())

# Poisson-Gamma distribution of the total hidden count.  LZ's rate uncertainty: s ~ Gamma(1.5, 1) (Jeffreys posterior after
# one count; 68% central interval computed below, close to LZ's 0.3-2.4) and the flat-prior Gamma(2, 1) variant.
def nb_pmf(n, alpha, k):      # s ~ Gamma(alpha, scale 1) per 2.84 t yr, N ~ Poisson(k s): NB(alpha, p = 1/(1+k))
    return stats.nbinom.pmf(n, alpha, 1.0 / (1.0 + k))
g15 = stats.gamma(1.5, scale=1.0); g2 = stats.gamma(2.0, scale=1.0)
say(f'[3] Jeffreys Gamma(1.5,1): 68% central {g15.ppf(0.16):.2f}-{g15.ppf(0.84):.2f}, mean 1.5; flat Gamma(2,1): {g2.ppf(0.16):.2f}-{g2.ppf(0.84):.2f}, mean 2.0; LZ Table I 0.30-2.36')
dist_rows = []
for label, mu_map, Eref in [('all hidden (LZ-like 270 window)', tot270, E_new_all), ('all hidden (200-270 keV)', tot200, E_new_all),
                            ('core three (270 window)', tot270_core, E_new_core), ('all hidden (200-400 keV window)', tot400, E_new_all)]:
    for m in MODELS:
        mu = float(mu_map[m]); k = mu / S_BEST          # k = expected count per unit of LZ's rate
        pl = stats.poisson(mu)
        d = dict(set=label, model=m, mu_best=mu, mu_lo68=mu * S_LO68, mu_hi68=mu * S_HI68, exposure_tyr=Eref,
                 P0_best=pl.pmf(0), P1_best=pl.pmf(1), P2_best=pl.pmf(2), P3_best=pl.pmf(3), Pge4_best=pl.sf(3),
                 P_ge1_best=1 - pl.pmf(0), P_ge1_lo68=1 - math.exp(-mu * S_LO68), P_ge1_FC=1 - math.exp(-mu * S_LO_FC),
                 P_ge2_best=pl.sf(1), P_ge3_best=pl.sf(2),
                 PG_J_P0=nb_pmf(0, 1.5, k), PG_J_P1=nb_pmf(1, 1.5, k), PG_J_P2=nb_pmf(2, 1.5, k), PG_J_P3=nb_pmf(3, 1.5, k),
                 PG_J_Pge4=1 - sum(nb_pmf(i, 1.5, k) for i in range(4)), PG_J_mean=1.5 * k,
                 PG_flat_P0=nb_pmf(0, 2.0, k), PG_flat_mean=2.0 * k)
        dist_rows.append(d)
dist = pd.DataFrame(dist_rows); dist.to_csv(WORK + '/P069_hidden_distribution.csv', index=False)
say('[3] distribution of the hidden count (all hidden, 270 window):\n' +
    dist[dist['set'] == 'all hidden (LZ-like 270 window)'][['model', 'mu_best', 'mu_lo68', 'mu_hi68', 'P0_best', 'P_ge1_best', 'P_ge1_lo68', 'P_ge1_FC', 'P_ge3_best', 'PG_J_P0', 'PG_J_mean']].round(3).to_string())
say('[3] 200-270 keV only:\n' + dist[dist['set'] == 'all hidden (200-270 keV)'][['model', 'mu_best', 'P0_best', 'P_ge1_best', 'P_ge1_lo68', 'PG_J_P0']].round(3).to_string())
RES['hidden_totals'] = dict(window270={m: float(tot270[m]) for m in MODELS}, band200_270={m: float(tot200[m]) for m in MODELS},
                            core270={m: float(tot270_core[m]) for m in MODELS}, ext400_gt200={m: float(tot400[m]) for m in MODELS},
                            tested_published={m: float(tested_tot[m]) for m in MODELS})


# ==============================================================================================
# 4. Scenario analysis: N_new events found in the 200-270 keV NR band of the hidden exposure E_new.
#    The counting test is defined in the 200-270 keV band (LZ's event lies there; background b per 2.84 t yr
#    from P016/P001/LZ Fig. 5).  Frequentist Z and the band-rate interval are model-independent; the expected
#    count mu_new = E_new x rate_200_270(model) and hence the Bayesian update are model-dependent.
# ==============================================================================================
E_NEW = E_new_all                     # adopted: everything hidden on disk on 2 Sep 2026 (stated exactly in the paper)
K_NEW = E_NEW / EXPO_LZ               # exposure ratio (background scaling; model-independent 'counting' normalisation)
say(f'[4] adopted hidden exposure E_new = {E_NEW:.2f} t yr (k = {K_NEW:.2f} LZ exposures); core three {E_new_core:.2f}; range {E_new_all_lo:.2f}-{E_new_all_hi:.2f}')
K_MODEL = {m: E_NEW * NORM[m]['rate_200_270'] / S_BEST for m in MODELS}      # expected 200-270 keV events per unit LZ rate
K_MODEL['count'] = K_NEW                                                     # model-independent: 1 observed per 2.84 t yr
say('[4] k (expected hidden 200-270 keV events at LZ best fit):', {k: round(v, 2) for k, v in K_MODEL.items()})

def Z_of_p(p): return float(stats.norm.isf(min(max(p, 1e-300), 1.0)))
def z_poisson(n, b): return Z_of_p(stats.poisson.sf(n - 1, b)) if n > 0 else 0.0
def z_asimov(s, b): return math.sqrt(2 * ((s + b) * math.log(1 + s / b) - s)) if s > 0 else 0.0
def z_lr(n, b):      # profile-likelihood-ratio Z for observed n against b (Cowan et al.)
    return math.sqrt(2 * (n * math.log(n / b) - (n - b))) if n > b else 0.0
def p_global(p_loc, n_eff): return float(-np.expm1(n_eff * np.log1p(-p_loc)))   # 1-(1-p)^N without cancellation

# P001/P020 three-hypothesis posterior (DM with flat s in [0,10]; U = unknown background with fixed rate mu_U; K known bkg)
PI_DM, PI_U, L_U, S_MAX = 0.01, 0.10, 0.10, 10.0
MU_U = -math.log(1 - L_U)
def L_DM(n2, k, b):
    f = lambda s: stats.poisson.pmf(1, s + b) * stats.poisson.pmf(n2, k * (s + b)) / S_MAX
    return integrate.quad(f, 0, S_MAX, limit=200)[0]
def L_K(n2, k, b): return stats.poisson.pmf(1, b) * stats.poisson.pmf(n2, k * b)
def L_U_fixed(n2, k, b): return stats.poisson.pmf(1, b + MU_U) * stats.poisson.pmf(n2, k * (b + MU_U))
def posterior(n2, k, b):
    w = np.array([PI_DM * L_DM(n2, k, b), PI_U * L_U_fixed(n2, k, b), (1 - PI_DM - PI_U) * L_K(n2, k, b)])
    return w / w.sum()
chk = posterior(0, 1e-9, 2e-4); say(f'[4] P001 check: P(DM | LZ event alone) = {chk[0]:.3f} (P001: 0.090)')
chk2 = posterior(2, 2.38, 2e-4); say(f'[4] P020 check: P(DM | N=2, k=2.38, b=2e-4) = {chk2[0]:.3f} (P020: 0.358)')
P_DM_CAP = PI_DM / (PI_DM + PI_U)     # U with a free rate (same prior as DM): counting cannot separate them

def gamma_interval(n_tot, k_tot, cl):   # flat-prior posterior on the band rate s (per 2.84 t yr): Gamma(n_tot+1, scale 1/k_tot)
    g = stats.gamma(n_tot + 1, scale=1.0 / k_tot)
    return g.ppf((1 - cl) / 2), g.ppf(1 - (1 - cl) / 2), g.mean(), g.ppf(0.9)

# P034 timing test: N_3sigma (LLR vs time-flat population of equal rate); Z ~ sqrt(N) scaling (P034: rule errs <= 25% at small N)
N3_P034 = {'d300': 96.5, 'd350': 11.5, 'd366': 6.5, 'd380': 5.3}

scen = []
for N_new in [0, 1, 2, 3, 5]:
    N_tot = 1 + N_new
    for bname, b in B_PER284.items():
        b_new = b * K_NEW; b_tot = b * (1 + K_NEW)
        lo68, hi68, mean, ul90 = gamma_interval(N_tot, 1 + K_NEW, 0.68)
        lo90, hi90, _, _ = gamma_interval(N_tot, 1 + K_NEW, 0.90)
        d = dict(N_new=N_new, N_tot=N_tot, bkg=bname, b_per284=b, b_new=b_new, b_tot=b_tot,
                 Z_comb_poisson=z_poisson(N_tot, b_tot), Z_new_poisson=z_poisson(N_new, b_new) if N_new else 0.0,
                 Z_comb_LR=z_lr(N_tot, b_tot), s_hat=N_tot / (1 + K_NEW), s_lo68=lo68, s_hi68=hi68, s_lo90=lo90, s_hi90=hi90, s_UL90=ul90, s_mean=mean,
                 P_DM_cap_free_U=P_DM_CAP)
        for km, k in K_MODEL.items():
            post = posterior(N_new, k, b)
            d[f'P_DM_{km}'] = post[0]; d[f'P_U_{km}'] = post[1]
            d[f'mu_new_{km}'] = k * S_BEST
            d[f'P_thisN_best_{km}'] = stats.poisson.pmf(N_new, k * S_BEST)
            d[f'P_thisN_lo68_{km}'] = stats.poisson.pmf(N_new, k * S_LO68)
            d[f'P_thisN_PGJ_{km}'] = nb_pmf(N_new, 1.5, k)
        for m, n3 in N3_P034.items():
            d[f'Z_time_{m}'] = 3.0 * math.sqrt(N_tot / n3)     # all N_tot events carry dates (LZ's June event included)
        scen.append(d)
sc = pd.DataFrame(scen); sc.to_csv(WORK + '/P069_scenarios.csv', index=False)
show = sc[sc['bkg'] == 'band200_270'][['N_new', 'Z_comb_poisson', 'Z_new_poisson', 'Z_comb_LR', 's_hat', 's_lo68', 's_hi68', 's_UL90',
                                       'P_DM_L10', 'P_DM_d366', 'P_DM_count', 'P_thisN_best_L10', 'P_thisN_best_d366', 'P_thisN_PGJ_L10', 'Z_time_d366', 'Z_time_d380']]
say('[4] scenarios (b = 5.7e-4 per 2.84 t yr in 200-270 keV; E_new = %.2f t yr):\n' % E_NEW + show.round(3).to_string())
say('[4] Z_comb (Poisson) for the three backgrounds:\n' + sc.pivot(index='N_new', columns='bkg', values='Z_comb_poisson').round(2).to_string())
say('[4] P(DM), L10 k, for the three backgrounds:\n' + sc.pivot(index='N_new', columns='bkg', values='P_DM_L10').round(3).to_string())
say(f'[4] with a free-rate unknown background, counting caps P(DM) at pi_DM/(pi_DM+pi_U) = {P_DM_CAP:.3f} for any N')
ul_new_only = 2.303 / K_NEW
r0 = sc[(sc.N_new == 0) & (sc.bkg == 'band200_270')].iloc[0]
say(f'[4] N_new = 0: new data alone exclude a band rate > {ul_new_only:.3f} per 2.84 t yr (90%); combined flat-prior UL90 {r0.s_UL90:.3f}; '
    f'in LZ-ROI units (/f_200_270): L10 {ul_new_only/NORM["L10"]["frac_200_270"]:.2f}, d350 {ul_new_only/NORM["d350"]["frac_200_270"]:.2f}, d366 {ul_new_only/NORM["d366"]["frac_200_270"]:.2f}; '
    f'P(N_new=0 | best fit): L10 {math.exp(-K_MODEL["L10"]):.3f}, d366 {math.exp(-K_MODEL["d366"]):.3f}; | 0.30: L10 {math.exp(-0.3*K_MODEL["L10"]):.3f}, d366 {math.exp(-0.3*K_MODEL["d366"]):.3f}')
RES['scenario_E_new'] = dict(E_new=E_NEW, k_new=K_NEW, k_model=K_MODEL, UL90_band_new_only=ul_new_only,
                             UL90_LZROI_units={m: ul_new_only / NORM[m]['frac_200_270'] for m in MODELS},
                             P0_best={m: math.exp(-K_MODEL[m]) for m in K_MODEL}, P0_lo68={m: math.exp(-0.3 * K_MODEL[m]) for m in K_MODEL})
N5 = {}
for bname, b in B_PER284.items():
    b_tot = b * (1 + K_NEW); n = 1
    while z_poisson(n, b_tot) < 5.0: n += 1
    N5[bname] = n
say('[4] total events (LZ + new) for 5 sigma local, counting in 200-270 keV:', N5)
RES['N_total_for_5sigma'] = N5

# ==============================================================================================
# 5. Protocol: pre-registered window(s), three spectra; look-elsewhere accounting; Asimov exposures; exclusion
# ==============================================================================================
N_EFF = {'pre-registered count (1 window)': 1.0, 'three pre-registered spectra': 3.0, 'LZ 616-model set (P008)': 12.2,
         'LZ 616-model set (P001, from 2.6/3.4 sigma)': 13.9, 'three separate 616-model papers': 3 * 13.9}
lee = []
for N_new in [1, 2, 3, 5]:
    N_tot = 1 + N_new
    for bname in ['band200_270', 'panel']:
        b_tot = B_PER284[bname] * (1 + K_NEW)
        p_loc = float(stats.poisson.sf(N_tot - 1, b_tot))
        for lab, ne in N_EFF.items():
            pg = p_global(p_loc, ne)
            lee.append(dict(N_new=N_new, N_tot=N_tot, bkg=bname, N_eff=ne, scheme=lab, p_local=p_loc, Z_local=Z_of_p(p_loc), p_global=pg, Z_global=Z_of_p(pg)))
lee = pd.DataFrame(lee); lee.to_csv(WORK + '/P069_lee.csv', index=False)
say('[5] global Z for N_new new events (b = 5.7e-4 band):\n' + lee[lee.bkg == 'band200_270'].pivot(index='N_new', columns='scheme', values='Z_global').round(2).to_string())
say('[5] global Z (b = 0.0106 panel):\n' + lee[lee.bkg == 'panel'].pivot(index='N_new', columns='scheme', values='Z_global').round(2).to_string())
p34 = stats.norm.sf(3.4); say(f'[5] LZ check: 3.4 sigma local -> global {Z_of_p(p_global(p34, 12.2)):.2f} (N_eff 12.2) / {Z_of_p(p_global(p34, 13.9)):.2f} (13.9); LZ: 2.6')

# windows for the pre-registered analysis: 200-270 keV (LZ-like edge) and 200-400 keV (E50 = 400 keV edge, P050 'ext400')
b270_per_tyr = B_PER284['band200_270'] / EXPO_LZ          # 2.0e-4 per t yr (P016 anchor)
b400_per_tyr = 9.66e-4                                     # P050 ext400 LZ-like: b(>200 keV) per t yr (MSSI interpolation + accidentals + nu)
WIN = {'200-270 keV': ('rate_200_270', b270_per_tyr), '200-400 keV': ('rate_ext400_gt200', b400_per_tyr)}
prot = []
for m in MODELS:
    for wname, (rk, bper) in WIN.items():
        rate = NORM[m][rk]
        for sname, sfac in [('best', S_BEST), ('lo68', S_LO68)]:
            E5 = optimize.brentq(lambda E: z_asimov(rate * sfac * E, bper * E) - 5.0, 1e-3, 1e6)
            E3 = optimize.brentq(lambda E: z_asimov(rate * sfac * E, bper * E) - 3.0, 1e-3, 1e6)
            prot.append(dict(model=m, window=wname, rate=sname, rate_per_tyr=rate * sfac, b_per_tyr=bper, E_3sigma=E3, E_5sigma=E5,
                             mu_in_Enew=rate * sfac * E_NEW, Z_asimov_Enew=z_asimov(rate * sfac * E_NEW, bper * E_NEW)))
prot = pd.DataFrame(prot); prot.to_csv(WORK + '/P069_protocol_windows.csv', index=False)
say('[5] Asimov exposures (new data only, counting):\n' + prot[['model', 'window', 'rate', 'rate_per_tyr', 'E_3sigma', 'E_5sigma', 'mu_in_Enew', 'Z_asimov_Enew']].round(3).to_string())
excl = []
for m in MODELS:
    for wname, (rk, bper) in WIN.items():
        rate = NORM[m][rk]
        for sname, sfac in [('TableI_lo 0.30', S_LO68), ('FC_lo 0.105', S_LO_FC), ('best 1.0', S_BEST)]:
            excl.append(dict(model=m, window=wname, s_low=sname, E_90=2.303 / (rate * sfac), E_95=2.996 / (rate * sfac)))
excl = pd.DataFrame(excl); excl.to_csv(WORK + '/P069_zero_event_exclusion.csv', index=False)
say('[5] zero-event exclusion exposures E_90 (t yr):\n' + excl.pivot_table(index=['window', 's_low'], columns='model', values='E_90').round(2).to_string())
RES['protocol'] = dict(b270_per_tyr=b270_per_tyr, b400_per_tyr=b400_per_tyr)

# ==============================================================================================
# 6. Timeline: cumulative exposure vs calendar year; LZ track vs the whole xenon world
# ==============================================================================================
T = np.arange(2013.0, T_END28 + 0.01, 0.05)
def accrue(row, t):
    return row['E'] * np.clip((t - row['t0']) / (row['t1'] - row['t0']), 0, 1)
cum = {}
for exp in ['LUX', 'PandaX-II', 'XENON1T', 'LZ', 'XENONnT', 'PandaX-4T']:
    c = np.zeros_like(T)
    for _, r in led[led.exp == exp].iterrows(): c += accrue(r, T)
    if exp in FUTURE_RATE: c += FUTURE_RATE[exp] * np.clip(T - T_NOW, 0, None)
    cum[exp] = c
cum_all = sum(cum.values())
sr3 = led[led.name == 'LZ SR3 (WS2024) analysed'].iloc[0]
cum_LZ_track = np.where(T >= 2024.25, EXPO_LZ + FUTURE_RATE['LZ'] * (T - 2024.25), accrue(sr3, T))   # LZ's own high-energy-capable data
tl = pd.DataFrame({'year': T, **{f'cum_{k}': v for k, v in cum.items()}, 'cum_all_xenon': cum_all, 'cum_LZ_HE_track': cum_LZ_track})
tl.to_csv(WORK + '/P069_timeline.csv', index=False)
def at(t, arr): return float(np.interp(t, T, arr))
milestones = {}
for t, lab in [(T_NOW, '2026-09-02'), (T_END26, 'end-2026'), (T_MID27, 'mid-2027'), (T_END27, 'end-2027'), (T_END28, 'end-2028')]:
    milestones[lab] = dict(all_xenon=at(t, cum_all), LZ=at(t, cum['LZ']), XENONnT=at(t, cum['XENONnT']), PandaX4T=at(t, cum['PandaX-4T']),
                           new_beyond_LZ_SR3=at(t, cum_all) - EXPO_LZ, LZ_track=at(t, cum_LZ_track))
say('[6] cumulative exposure milestones (t yr):\n' + pd.DataFrame(milestones).T.round(2).to_string())
RES['milestones'] = milestones

def year_when(arr, target):
    idx = np.where(arr >= target)[0]
    return round(float(T[idx[0]]), 2) if len(idx) else None
years = {}
for m in ['L10', 'd350', 'd366']:
    rate = NORM[m]['rate_200_270']
    E5b = optimize.brentq(lambda E: z_asimov(rate * E, b270_per_tyr * E) - 5, 1e-3, 1e6)
    E5l = optimize.brentq(lambda E: z_asimov(rate * S_LO68 * E, b270_per_tyr * E) - 5, 1e-3, 1e7)
    E3l = optimize.brentq(lambda E: z_asimov(rate * S_LO68 * E, b270_per_tyr * E) - 3, 1e-3, 1e7)
    E90b = 2.303 / rate; E90l = 2.303 / (rate * S_LO68); E90fc = 2.303 / (rate * S_LO_FC)
    years[m] = {}
    for lab, arr in [('LZ track', cum_LZ_track), ('all xenon', cum_all)]:
        years[m][lab] = {f'5sigma best (E={E5b:.1f})': year_when(arr, E5b), f'5sigma lo68 (E={E5l:.0f})': year_when(arr, E5l),
                         f'3sigma lo68 (E={E3l:.1f})': year_when(arr, E3l), f'0 ev excl best (E={E90b:.1f})': year_when(arr, E90b),
                         f'0 ev excl 0.30 (E={E90l:.1f})': year_when(arr, E90l), f'0 ev excl 0.105 (E={E90fc:.0f})': year_when(arr, E90fc)}
    say(f'[6] {m}: ' + json.dumps(years[m]))
RES['milestone_years'] = years
say('    P050 cross-check (LZ-like window incl. 100-200 keV bins): E_5sigma L10 12.3, d350 8.5, d366 6.8 t yr; E_90(0.30) 21.8; E_90(0.105) 62.3')

# joint dataset for a mid-2027 publication = data on disk to end-2026
E_joint_end26 = milestones['end-2026']['all_xenon'] - EXPO_LZ
joint = {}
for m in MODELS:
    rate = NORM[m]['rate_200_270']; mu = rate * E_joint_end26; mu_lo = mu * S_LO68
    b_tot = B_PER284['band200_270'] * (1 + E_joint_end26 / EXPO_LZ)
    medN = int(stats.poisson.median(mu))
    joint[m] = dict(E_new=E_joint_end26, mu_best=mu, mu_lo68=mu_lo, P0_best=math.exp(-mu), P0_lo68=math.exp(-mu_lo),
                    P_ge2_best=float(stats.poisson.sf(1, mu)), P_Ntot_ge3_best=float(stats.poisson.sf(1, mu)), P_Ntot_ge3_lo68=float(stats.poisson.sf(1, mu_lo)),
                    median_N=medN, Z_comb_median=z_poisson(1 + medN, b_tot), Z_asimov_new=z_asimov(mu, b270_per_tyr * E_joint_end26),
                    Z_asimov_new_lo68=z_asimov(mu_lo, b270_per_tyr * E_joint_end26),
                    UL90_if_zero_LZROI_units=2.303 / (rate * E_joint_end26))   # zero events: LZ-ROI rate (units of the best fit) excluded at 90%
jd = pd.DataFrame(joint).T; jd.to_csv(WORK + '/P069_joint_end2026.csv')
say(f'[6] joint dataset for a mid-2027 paper (data to end-2026): {E_joint_end26:.1f} t yr new + 2.84 analysed = {E_joint_end26+EXPO_LZ:.1f} t yr:\n' + jd.round(3).to_string())
RES['joint_end2026'] = joint

# ==============================================================================================
# 7. Figures (Okabe-Ito palette, fixed order)
# ==============================================================================================
OI = {'LZ': '#0072B2', 'XENONnT': '#E69F00', 'PandaX-4T': '#009E73', 'XENON1T': '#CC79A7', 'PandaX-II': '#56B4E9', 'LUX': '#999999'}
fig, ax = plt.subplots(figsize=(8.2, 4.8))
order = ['LUX', 'PandaX-II', 'XENON1T', 'PandaX-4T', 'XENONnT', 'LZ']
base = np.zeros_like(T)
for exp in order:
    ax.fill_between(T, base, base + cum[exp], color=OI[exp], alpha=0.85, linewidth=0, label=exp)
    base = base + cum[exp]
ax.plot(T, cum_LZ_track, color='k', lw=2, label='LZ high-energy track (SR3 + untouched)')
ax.axvline(T_NOW, color='k', ls=':', lw=1); ax.text(T_NOW + 0.05, 2, 'LZ paper\n2 Sep 2026', fontsize=8)
E5_L10 = optimize.brentq(lambda E: z_asimov(NORM['L10']['rate_200_270'] * E, b270_per_tyr * E) - 5, 1e-3, 1e6)
E5_366 = optimize.brentq(lambda E: z_asimov(NORM['d366']['rate_200_270'] * E, b270_per_tyr * E) - 5, 1e-3, 1e6)
E90_30 = 2.303 / (NORM['L10']['rate_200_270'] * S_LO68)
for y, lab in [(E5_366, f'5σ, best fit, δ=366 keV ({E5_366:.1f} t·yr)'), (E5_L10, f'5σ, best fit, L10 ({E5_L10:.1f} t·yr)')]:
    ax.axhline(y, color='#444', lw=0.8, ls='--'); ax.text(2013.2, y + 0.6, lab, fontsize=7.5, color='#444')
ax.text(2013.2, 39.5, f'(0 events exclude LZ\'s lower edge 0.30 for L10 only at {E90_30:.0f} t·yr, off scale; δ=366 keV: 28 t·yr)', fontsize=7.5, color='#444')
ax.axvspan(T_NOW, T_END28, color='#f0f0f0', zorder=0)
ax.set_xlim(2013, T_END28); ax.set_ylim(0, 42); ax.set_xlabel('calendar year'); ax.set_ylabel('cumulative xenon-TPC exposure (t·yr)')
ax.set_title('Xenon-TPC exposure on disk vs. the 2.84 t·yr analysed above 200 keV (200–270 keV counting thresholds)', loc='left', fontsize=9.5)
ax.legend(fontsize=8, loc='upper left', frameon=False, ncol=2); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(FIG + '/P069_exposure_timeline.png', dpi=160); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.0))
ax = axs[0]
for bname, col, lab in [('nb2sig', '#0072B2', 'b = 2×10⁻⁴ (NR ±2σ)'), ('band200_270', '#E69F00', 'b = 5.7×10⁻⁴ (200–270 keV band)'), ('panel', '#009E73', 'b = 0.0106 (whole S1c>500 panel)')]:
    s = sc[sc.bkg == bname]
    ax.plot(s.N_new, s.Z_comb_poisson, '-o', color=col, ms=6, lw=2, label=lab)
ax.axhline(5, color='#444', ls='--', lw=0.8); ax.text(0.1, 5.1, '5σ', fontsize=8)
ax.set_xlabel(f'new 200–270 keV events N_new in {E_NEW:.1f} t·yr'); ax.set_ylabel('combined local significance (σ)')
ax.set_title('Counting significance, LZ event + N_new', loc='left', fontsize=10); ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.25)
ax = axs[1]
s = sc[sc.bkg == 'band200_270']
ax.plot(s.N_new, s.P_DM_L10, '-o', color='#0072B2', lw=2, ms=6, label='P(DM), L10 expectation (2.3 events)')
ax.plot(s.N_new, s.P_DM_d366, '-s', color='#CC79A7', lw=2, ms=6, label='P(DM), δ=366 keV expectation (5.2 events)')
ax.axhline(P_DM_CAP, color='#D55E00', ls='--', lw=1.2, label='cap if the unknown background has a free rate')
ax.bar(s.N_new - 0.15, s.P_thisN_best_L10, width=0.3, color='#E69F00', alpha=0.6, label='P(N_new | L10 best fit)')
ax.bar(s.N_new + 0.15, s.P_thisN_PGJ_L10, width=0.3, color='#009E73', alpha=0.6, label='P(N_new | L10, Poisson–Gamma)')
ax.set_xlabel('new 200–270 keV events N_new'); ax.set_ylabel('probability'); ax.set_ylim(0, 1)
ax.set_title('Posterior (P001 framework) and predictive', loc='left', fontsize=10); ax.legend(fontsize=7.5, frameon=False); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(FIG + '/P069_scenarios.png', dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(8.6, 4.4))
ds = led[led.hidden_at_HE]['name'].tolist()
x = np.arange(len(ds)); w = 0.16
cols = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00']
for i, m in enumerate(MODELS):
    vals = [float(hid[(hid.dataset == d) & (hid.model == m)].hidden_200_270.iloc[0]) for d in ds]
    ax.bar(x + (i - 2) * w, vals, w, color=cols[i], label=LABELS[m])
short = {'LUX 2013-16': 'LUX', 'PandaX-II 2016-18': 'PandaX-II', 'XENON1T 2018': 'XENON1T', 'LZ SR1 2022': 'LZ SR1',
         'LZ untouched since Apr 2024': 'LZ\nuntouched', 'XENONnT SR0+SR1 analysed': 'XENONnT\nSR0+SR1', 'XENONnT untouched (SR2+)': 'XENONnT\nuntouched',
         'PandaX-4T Run0+Run1 analysed': 'PandaX-4T\nRun0+1', 'PandaX-4T untouched (Run2/3)': 'PandaX-4T\nuntouched'}
ax.set_xticks(x); ax.set_xticklabels([short[d] for d in ds], fontsize=8)
ax.set_ylabel('expected 200–270 keV NR events on disk\n(best fit; ×0.30–2.36 for LZ\'s 68% band)'); ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.25, axis='y')
ax.set_title('Where the hidden events sit', loc='left', fontsize=10)
fig.tight_layout(); fig.savefig(FIG + '/P069_hidden_events.png', dpi=160); plt.close(fig)

with open(WORK + '/P069_results.json', 'w') as f: json.dump(RES, f, indent=1, default=float)
led.to_csv(WORK + '/P069_exposure_ledger.csv', index=False)
say('done; wrote', WORK)
