"""
P058 - If the excited state survives: exothermic down-scattering (chi2 N -> chi1 N, releasing delta) of a relic
chi2 fraction f2 in LZ's data, and the chi2-lifetime window it excludes.

Parts
  A  exothermic kinematics (E* = |delta| mu/m_N, E_-/E_+(v), width, |delta| giving E* = 248 keV); validation of
     lzcommon's E_R_range_keV / vmin_kms with negative delta and of WimPyDD's negative-delta handling
  B  WimPyDD O1 (E, v) kernels for endothermic (+delta) and exothermic (-delta) scattering, three isospin
     structures (LZ isoscalar, dark-photon proton-only, Higgsino Z), 400/1000/4000 GeV; Baxter-2021 halo on 24 days
  C  counts per unit coupling in LZ's regions (5.4-55, 55-125, 125-200, 200-270 keV in the WS ROI; 600-1000 and
     600-1700 phd empty high-energy region of Fig. S4), ratios exothermic/endothermic per unit f2
  D  statistics: per-region 90 % ULs on f2 and a joint profile likelihood in (kappa, f2); translation to the
     chi2 lifetime tau (f2 = f2,fo exp(-t_U/tau)), to the dark-photon eps floor and m_A' floor
  E  the event itself as an exothermic down-scatter: |delta|(E* = 248 keV), percentile of 248 keV, companions,
     predicted 600-1700 phd events per ROI event
  F  annual modulation of the exothermic rate vs the endothermic rate
  G  figures
Run from the simulation root:  .venv/bin/python output/code/P058_exothermic.py   (kernels cached in work/P058/cache)
"""
from __future__ import annotations
import sys, os, json, math, time
sys.path.insert(0, 'output/code')
import numpy as np
import pandas as pd
from scipy.special import erf
from scipy.stats import poisson, norm
from scipy.optimize import brentq, minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import lzcommon as lz

T0 = time.time()
OUT = 'output/work/P058'; FIG = f'{OUT}/figures'; CACHE = f'{OUT}/cache'
for d in (OUT, FIG, CACHE):
    os.makedirs(d, exist_ok=True)
LOG = open(f'{OUT}/run_log.txt', 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n'); LOG.flush()

# ------------------------------------------------------------------------------------------------------------
# constants (recalled values flagged)
# ------------------------------------------------------------------------------------------------------------
C_KMS = lz.C_KMS
AMU = lz.AMU_GEV                     # 0.9314941 GeV (recalled, certain)
A_XE = lz.A_XE_MEAN                  # 131.29
MN_MEAN = A_XE * AMU
MV = lz.M_V_GEV                      # 246.2 GeV (LZ/Anand unit coupling)
C_UNIT = 1.0 / MV ** 2               # GeV^-2 : Anand c_p (proton-only) or c_p = c_n (isoscalar unit coupling)
GF = 1.1663788e-5                    # GeV^-2 (recalled, certain)
SW2 = 0.2312                         # sin^2 theta_W at low energy (recalled, certain +-0.001)
ALPHA = 1 / 137.035999               # (recalled, certain)
MZ = 91.1876                         # GeV (recalled, certain)
MP = lz.M_NUCLEON_GEV
HBAR_GEV_S = 6.582119569e-25         # (recalled, certain)
T_U_S = 13.8e9 * 3.15576e7           # 4.35e17 s (recalled, certain)
EXPOSURE = lz.LZ['exposure_tyr']     # 2.84 t yr
C_P_HIG = (GF / math.sqrt(2)) * (1 - 4 * SW2)      # P007/P011 Higgsino Z couplings (WimPyDD c_p, c_n)
C_N_HIG = -(GF / math.sqrt(2))
F2_FO = {'higgsino': 0.42, 'darkphoton': 0.50}     # P026 freeze-out fractions

# ------------------------------------------------------------------------------------------------------------
# Part A: kinematics
# ------------------------------------------------------------------------------------------------------------
def mu_red(m, mN=MN_MEAN):
    return m * mN / (m + mN)

def estar_keV(delta_keV, m, mN=MN_MEAN):
    """Recoil energy at v -> 0 for exothermic scattering: E* = |delta| mu / m_N."""
    return abs(delta_keV) * mu_red(m, mN) / mN

def E_pm_exo(m, v_kms, delta_keV, mN=MN_MEAN):
    """[E-, E+] (keV) for exothermic scattering (delta -> -|delta|) at speed v; own formula."""
    mu = mu_red(m, mN); b = v_kms / C_KMS; d = abs(delta_keV) * 1e-6
    pref = mu ** 2 * b ** 2 / mN; x = d / (mu * b ** 2)
    return pref * (1 + x - math.sqrt(1 + 2 * x)) * 1e6, pref * (1 + x + math.sqrt(1 + 2 * x)) * 1e6

def vmin_exo(E_keV, m, delta_keV, mN=MN_MEAN):
    mu = mu_red(m, mN); E = E_keV * 1e-6; d = abs(delta_keV) * 1e-6
    return abs(mN * E / mu - d) / math.sqrt(2 * mN * E) * C_KMS

V_TYP, V_MAX_JUNE = 250.0, 810.0
kin_rows = []
for m in (400.0, 1000.0, 4000.0):
    for d in (250.0, 300.0, 350.0, 366.0, 380.0):
        Es = estar_keV(d, m); lo, hi = E_pm_exo(m, V_MAX_JUNE, d); lo_t, hi_t = E_pm_exo(m, V_TYP, d)
        lz_lo, lz_hi = lz.E_R_range_keV(m, V_MAX_JUNE, delta_kev=-d)
        kin_rows.append(dict(m_GeV=m, delta_keV=d, E_star_keV=Es, halfwidth_v250_keV=(hi_t - lo_t) / 2,
                             E_minus_v250=lo_t, E_plus_v250=hi_t, E_minus_vmax810=lo, E_plus_vmax810=hi,
                             lz_E_R_range_minus=lz_lo, lz_E_R_range_plus=lz_hi,
                             vmin_248keV_kms=vmin_exo(248.0, m, d), lz_vmin_248_raw=lz.vmin_kms(248.0, m, delta_kev=-d)))
kin = pd.DataFrame(kin_rows); kin.to_csv(f'{OUT}/P058_exo_kinematics.csv', index=False)
say('Part A: exothermic kinematics (A = 131.29)'); say(kin.to_string(index=False, float_format='%.4g'))
say('  check lz.E_R_range_keV(delta<0) vs own formula: max rel dev =',
    float(np.max(np.abs(kin.lz_E_R_range_minus / kin.E_minus_vmax810 - 1))), float(np.max(np.abs(kin.lz_E_R_range_plus / kin.E_plus_vmax810 - 1))))
say('  lz.vmin_kms(248, 1000, delta=-300) =', lz.vmin_kms(248.0, 1000.0, delta_kev=-300.0), '(negative: the library lacks abs(); physical v_min is |.|)')
# |delta| giving E* = 248 keV
ev_rows = []
for m in (200.0, 400.0, 1000.0, 2000.0, 4000.0, np.inf):
    mu = MN_MEAN if np.isinf(m) else mu_red(m)
    d248 = 248.0 * MN_MEAN / mu
    lo_t, hi_t = E_pm_exo(1e12 if np.isinf(m) else m, V_TYP, d248)
    ev_rows.append(dict(m_GeV=m, delta_for_Estar_248_keV=d248, halfwidth_v250_keV=(hi_t - lo_t) / 2,
                        E_star_at_delta300=estar_keV(300.0, 1e12 if np.isinf(m) else m),
                        E_star_at_delta366=estar_keV(366.0, 1e12 if np.isinf(m) else m)))
ev = pd.DataFrame(ev_rows); ev.to_csv(f'{OUT}/P058_delta_for_Estar248.csv', index=False)
say('  |delta| giving E* = 248 keV:'); say(ev.to_string(index=False, float_format='%.4g'))

# ------------------------------------------------------------------------------------------------------------
# Part B: WimPyDD kernels
# ------------------------------------------------------------------------------------------------------------
WD = lz.wd()
VGRID = np.linspace(0.0, 844.0, 1200); ONES = np.ones_like(VGRID)
DAYS24 = 365.25 / 48 + 365.25 / 24 * np.arange(24)
say('building halos (24 days + June 16 + Dec 16 + Sun frame) ...')
HALO_DAYS = {f'd{int(round(d)):03d}': lz.wd_halo(day_of_year=float(d), vmin=VGRID)[1] for d in DAYS24}
HALOS = {'annual': np.mean(list(HALO_DAYS.values()), axis=0), 'june16': lz.wd_halo(day_of_year=167, vmin=VGRID)[1],
         'dec16': lz.wd_halo(day_of_year=350, vmin=VGRID)[1], 'sun': lz.wd_halo(vmin=VGRID)[1]}
HAMS = {'iso': lz.wd_hamiltonian('P058_iso', {1: (2 * C_UNIT, 0.0)}),            # LZ unit isoscalar c_p = c_n = 1/m_v^2
        'p': lz.wd_hamiltonian('P058_p', {1: (C_UNIT, C_UNIT)}),                  # dark photon: c_p = 1/m_v^2, c_n = 0
        'hig': lz.wd_hamiltonian('P058_hig', {1: (C_P_HIG + C_N_HIG, C_P_HIG - C_N_HIG)})}   # Higgsino Z exchange
E_EXO = np.concatenate([np.arange(1.0, 700.0, 2.5), np.arange(700.0, 2505.0, 5.0)])
E_ENDO = np.arange(1.0, 452.0, 2.5)
CONFIGS = [('iso', 1000.0, (250.0, 278.0, 300.0, 350.0, 366.0, 380.0)), ('p', 1000.0, (250.0, 278.0, 300.0, 350.0, 366.0, 380.0)),
           ('hig', 1000.0, (300.0, 350.0, 366.0, 380.0)), ('iso', 400.0, (300.0, 350.0, 380.0)), ('iso', 4000.0, (300.0, 350.0, 380.0))]

def kernel(tag, m, delta, E):
    """(len(E), len(VGRID)) matrix: differential rate per stream [events/(t yr keV) per unit delta_eta]."""
    fn = f'{CACHE}/K_{tag}_{int(m)}_{"m" if delta < 0 else "p"}{int(round(abs(delta)))}.npz'
    if os.path.exists(fn):
        z = np.load(fn)
        if z['E'].shape == E.shape and np.allclose(z['E'], E):
            return z['K']
    K = np.array([WD.diff_rate(WD.Xe, HAMS[tag], m, float(e), VGRID, ONES, j_chi=0.5, delta=float(delta), sum_over_streams=False)
                  for e in E]) * 1000.0 * 365.25
    np.savez_compressed(fn, E=E, K=K)
    return K

# validation of WimPyDD negative-delta handling: per-isotope thresholds
say('Part B: WimPyDD negative-delta check at E = 200 keV, 1000 GeV, delta = -300 keV')
Kt = WD.diff_rate(WD.Xe, HAMS['iso'], 1000.0, 200.0, VGRID, ONES, j_chi=0.5, delta=-300.0, sum_over_streams=False)
v_first = float(VGRID[np.argmax(Kt > 0)])
iso_masses = np.atleast_1d(np.array(WD.Xe.mass if hasattr(WD.Xe, 'mass') else WD.Xe.element[0].mass, dtype=float))
thr = sorted(vmin_exo(200.0, 1000.0, 300.0, mN=float(mm)) for mm in (iso_masses if iso_masses is not None else [MN_MEAN]))
say(f'  first non-zero stream v = {v_first:.1f} km/s; per-isotope v_min(200 keV) from own formula: {np.round(thr, 1).tolist()} km/s')
say(f'  distinct partial-sum levels K/K_max: {np.unique(np.round(Kt[Kt > 0] / Kt.max(), 4)).tolist()}  (one per isotope threshold)')
val_wd = dict(v_first_nonzero_stream=v_first, isotope_vmin_kms=[float(x) for x in thr], levels=np.unique(np.round(Kt[Kt > 0] / Kt.max(), 4)).tolist())

say('computing kernels ...')
KERN = {}
for tag, m, deltas in CONFIGS:
    for d in deltas:
        KERN[(tag, m, -d)] = kernel(tag, m, -d, E_EXO)
        KERN[(tag, m, +d)] = kernel(tag, m, +d, E_ENDO)
        say(f'  {tag} {m:.0f} GeV delta={d:.0f}: done, t = {time.time() - T0:.0f} s')

# ------------------------------------------------------------------------------------------------------------
# Part C: efficiencies, regions, counts
# ------------------------------------------------------------------------------------------------------------
def Phi(x):
    return 0.5 * (1 + erf(x / math.sqrt(2)))
E50_600, SIG_600 = 271.6, 10.9          # P038 NEST-LZ rebuild of the 600 phd edge (Fig. S2: 269.9)
E50_1000, SIG_1000 = 423.2, 14.5        # P038
S1_EXP = 1.1537                         # P038: d ln S1c / d ln E at 250-300 keV
E50_1700 = E50_1000 * (1700.0 / 1000.0) ** (1 / S1_EXP); SIG_1700 = SIG_1000 * E50_1700 / E50_1000
PLATEAU_HE = 0.955                      # P038 plateau used above the ROI
def eps_roi(E):
    return 0.96 * Phi((E - 5.4) / 2.5) * (1 - Phi((E - 269.9) / 11.5))
def eps_he(E, top=1700):
    E50t, st = (E50_1700, SIG_1700) if top == 1700 else (E50_1000, SIG_1000)
    return PLATEAU_HE * Phi((E - E50_600) / SIG_600) * (1 - Phi((E - E50t) / st))
say(f'Part C: high-energy region: 600 phd edge E50 = {E50_600} keV; 1000 phd -> {E50_1000}; 1700 phd -> {E50_1700:.0f} keV (sigma {SIG_1700:.0f})')

REGIONS = {'lo': (5.4, 55.0, 'roi'), 'mid': (55.0, 125.0, 'roi'), 'gap': (125.0, 200.0, 'roi'), 'hi': (200.0, 270.0, 'roi'),
           'roi': (0.0, 3000.0, 'roi'), 'he1000': (0.0, 3000.0, 'he1000'), 'he1700': (0.0, 3000.0, 'he1700'), 'all': (0.0, 3000.0, 'none')}
def counts(E, r, region):
    lo, hi, effname = REGIONS[region]
    Ef = np.linspace(max(lo, E[0]), min(hi, E[-1]), 3001)
    y = np.interp(Ef, E, r)
    if effname == 'roi': y = y * eps_roi(Ef)
    elif effname == 'he1000': y = y * eps_he(Ef, 1000)
    elif effname == 'he1700': y = y * eps_he(Ef, 1700)
    return float(np.trapezoid(y, Ef)) * EXPOSURE

rows = []
SPEC = {}
for tag, m, deltas in CONFIGS:
    for d in deltas:
        for hname, deta in HALOS.items():
            r_exo = KERN[(tag, m, -d)] @ deta; r_endo = KERN[(tag, m, +d)] @ deta
            SPEC[(tag, m, d, hname)] = (r_exo, r_endo)
            row = dict(tag=tag, m_GeV=m, delta_keV=d, halo=hname)
            for reg in REGIONS:
                row[f'Nexo_{reg}'] = counts(E_EXO, r_exo, reg)
                row[f'Nendo_{reg}'] = counts(E_ENDO, r_endo, reg)
            # spectral descriptors of the exothermic spectrum (no efficiency)
            cdf = np.cumsum(0.5 * (r_exo[1:] + r_exo[:-1]) * np.diff(E_EXO)); cdf = np.insert(cdf, 0, 0.0); cdf /= cdf[-1]
            row['Eexo_peak_keV'] = float(E_EXO[np.argmax(r_exo)])
            row['Eexo_p16'], row['Eexo_p50'], row['Eexo_p84'] = [float(np.interp(p, cdf, E_EXO)) for p in (0.16, 0.5, 0.84)]
            row['frac_exo_below_270'] = float(np.interp(270.0, E_EXO, cdf)); row['frac_exo_270_670'] = float(np.interp(670.0, E_EXO, cdf)) - row['frac_exo_below_270']
            row['frac_exo_above_670'] = 1 - float(np.interp(670.0, E_EXO, cdf))
            # percentile of 248 keV in the accepted (ROI-efficiency-weighted) exothermic spectrum, true and observed energy
            acc = r_exo * eps_roi(E_EXO); ca = np.cumsum(0.5 * (acc[1:] + acc[:-1]) * np.diff(E_EXO)); ca = np.insert(ca, 0, 0); ca /= ca[-1]
            row['pct248_exo_accepted_true'] = float(np.interp(248.0, E_EXO, ca))
            Eo = np.arange(0.0, 800.0, 1.0); sig = 11.0 * np.sqrt(np.maximum(E_EXO, 1.0) / 248.0)
            obs = np.array([np.sum(acc * norm.pdf(eo, E_EXO, sig) * np.gradient(E_EXO)) for eo in Eo]); co = np.cumsum(obs); co /= co[-1]
            row['pct248_exo_accepted_obs'] = float(np.interp(248.0, Eo, co))
            row['obs_p16'], row['obs_p50'], row['obs_p84'] = [float(np.interp(p, co, Eo)) for p in (0.16, 0.5, 0.84)]
            rows.append(row)
df = pd.DataFrame(rows)
for reg in REGIONS:
    df[f'R_{reg}'] = df[f'Nexo_{reg}'] / df['Nendo_roi']        # exothermic events in region per unit f2 per endothermic ROI event
df.to_csv(f'{OUT}/P058_counts_per_unit_coupling.csv', index=False)
sel = df[df.halo == 'annual']
say('Part C: exothermic spectra (annual halo): peak, 16/50/84 percentiles [keV], fractions <270 / 270-670 / >670')
say(sel[['tag', 'm_GeV', 'delta_keV', 'Eexo_peak_keV', 'Eexo_p16', 'Eexo_p50', 'Eexo_p84', 'frac_exo_below_270', 'frac_exo_270_670', 'frac_exo_above_670']].to_string(index=False, float_format='%.3g'))
say('Part C: counts per unit coupling (annual): endothermic ROI vs exothermic regions; ratios R = Nexo/Nendo_roi')
say(sel[['tag', 'm_GeV', 'delta_keV', 'Nendo_roi', 'Nexo_roi', 'Nexo_lo', 'Nexo_gap', 'Nexo_hi', 'Nexo_he1000', 'Nexo_he1700', 'Nexo_all', 'R_roi', 'R_lo', 'R_gap', 'R_hi', 'R_he1000', 'R_he1700']].to_string(index=False, float_format='%.3g'))
# cross-check against P011 (proton-only, 1000 GeV, annual, ROI to 330 keV with efficiency)
try:
    p11 = pd.read_csv('output/work/P011/exothermic.csv'); p11 = p11[p11.halo == 'annual']
    p11g = pd.read_csv('output/work/P011/N_events_grid.csv'); p11g = p11g[(p11g.m_GeV == 1000.0) & (p11g.halo == 'annual') & (p11g.eff_sigma_keV == 11.5)]
    for d in (250.0, 300.0, 350.0, 365.0, 380.0):
        mine = sel[(sel.tag == 'p') & (sel.m_GeV == 1000.0) & (np.isclose(sel.delta_keV, d if d != 365.0 else 366.0))]
        theirs = p11[np.isclose(p11.delta_keV, d)]; th_endo = p11g[np.isclose(p11g.delta_keV, d)]
        if len(mine) and len(theirs) and len(th_endo):
            say(f'  P011 cross-check delta={d:.0f}: Nexo_ROI/unit {float(mine.Nexo_roi.iloc[0]):.4g} vs P011 {float(theirs.N_exo_unit_ROI.iloc[0]):.4g}; '
                f'Nendo_ROI/unit {float(mine.Nendo_roi.iloc[0]):.4g} vs P011 N_p {float(th_endo.N_p.iloc[0]):.4g}')
except Exception as e:
    say('  P011 cross-check skipped:', e)

# ------------------------------------------------------------------------------------------------------------
# Part D: statistics -> f2 upper limits -> tau, eps, m_A'
# ------------------------------------------------------------------------------------------------------------
# observed counts and backgrounds per region (sources: P016 digitised Fig. 5 and LZ Table I / MSSI table; P038)
OBS = {'lomid': dict(n=42, b=58.6, acc=0.866),      # 5.4-125 keV, within +-1.5 sigma of the NR median (P016); 86.6 % NR acceptance (recalled, certain)
       'gap': dict(n=0, b=0.02, acc=1.0),            # 125-200 keV (P016)
       'hi': dict(n=1, b=5.7e-4, acc=1.0),           # 200-270 keV (P016/P021 anchor)
       'he1700': dict(n=0, b=0.606, acc=1.0),        # 600-1700 phd, S2c < 10^4.3, 4.7 t science: 0 observed vs 0.1 wall + 0.5 RFR MSSI (LZ MSSI table) + 0.006 (P038, 600-800 phd)
       'he1000': dict(n=0, b=0.018, acc=1.0)}        # 600-1000 phd (P038 MSSI interpolation)
def s90_cls(n, b):
    """90 % CL upper limit on a signal s with n observed and background b (CLs / modified frequentist)."""
    f = lambda s: poisson.cdf(n, b + s) / poisson.cdf(n, b) - 0.10
    return brentq(f, 0.0, 200.0)
S90 = {k: s90_cls(v['n'], v['b']) for k, v in OBS.items()}
S90['hi_extra'] = s90_cls(1, 5.7e-4 + 1.0)   # one endothermic event already accounted for in 200-270 keV
say('Part D: per-region 90 % CLs upper limits on extra events:', {k: round(v, 3) for k, v in S90.items()})

def region_unit(row, reg):
    if reg == 'lomid': return row['Nexo_lo'] + row['Nexo_mid'], row['Nendo_lo'] + row['Nendo_mid']
    return row[f'Nexo_{reg}'], row[f'Nendo_{reg}']

def nll_bins(row, kappa, f2, regions, exo_only=False, endo_only=False):
    """-lnL (Poisson, constants dropped) over regions with mu_X = b_X + acc_X kappa (n_endo,X + f2 n_exo,X)."""
    tot = 0.0
    for reg in regions:
        nx, ne = region_unit(row, reg)
        if exo_only: ne = 0.0
        if endo_only: nx = 0.0
        mu = OBS[reg]['b'] + OBS[reg]['acc'] * kappa * (ne + f2 * nx)
        tot += mu - OBS[reg]['n'] * math.log(mu)
    return tot

def fixedkappa_f2_UL(row, regions=('lomid', 'gap', 'hi', 'he1700'), mu_endo=1.0, cl_q=2.706):
    """Joint likelihood in f2 with kappa fixed by the endothermic interpretation of the event: kappa = mu_endo / N_endo,ROI
    (LZ's best fit mu_endo = 1.0 event in the ROI). 90 % one-sided UL: Delta(-2lnL) = 2.706 above the minimum over f2 >= 0."""
    kappa = mu_endo / row['Nendo_roi']
    f2grid = np.concatenate([[0.0], np.logspace(-10, 1, 441)])
    q = np.array([2 * nll_bins(row, kappa, f, regions) for f in f2grid]); q -= q.min()
    idx = np.where(q > cl_q)[0]
    if len(idx) == 0: return np.nan
    i = idx[0]
    if i <= 1: return f2grid[1]
    x = np.interp(cl_q, [q[i - 1], q[i]], [math.log10(f2grid[i - 1]), math.log10(f2grid[i])])
    return 10 ** x

def profile_f2(row, regions=('lomid', 'gap', 'hi', 'he1700')):
    """kappa profiled at each f2: returns kappa-hat(f2=0), and (q, kappa-hat) at f2 = 1 relative to the f2 = 0 minimum."""
    def prof(f2):
        res = minimize_scalar(lambda lk: nll_bins(row, 10 ** lk, f2, regions), bounds=(-14, 6), method='bounded', options={'xatol': 1e-6})
        return res.fun, 10 ** res.x
    nll0, k0 = prof(0.0); nll1, k1 = prof(1.0)
    f2grid = np.logspace(-9, 0, 181); q = np.array([2 * (prof(f)[0] - nll0) for f in f2grid]); qmin = q.min()
    idx = np.where(q - qmin > 2.706)[0]
    ul = f2grid[idx[0]] if len(idx) else np.nan
    return k0, 2 * (nll1 - nll0), k1, ul

def local_Z(row, regions=('lomid', 'gap', 'hi', 'he1700'), exo_only=False, endo_only=False):
    """Four-bin local significance of a one-parameter (overall scale) hypothesis vs background only: Z = sqrt(2 (nll_bkg - nll_min))."""
    nll_b = nll_bins(row, 0.0, 0.0, regions)
    res = minimize_scalar(lambda lk: nll_bins(row, 10 ** lk, 1.0, regions, exo_only=exo_only, endo_only=endo_only), bounds=(-16, 6), method='bounded', options={'xatol': 1e-7})
    q0 = max(0.0, 2 * (nll_b - res.fun))
    return math.sqrt(q0), 10 ** res.x

stat_rows = []
for _, row in sel.iterrows():
    if not np.isfinite(row['Nendo_roi']) or row['Nendo_roi'] <= 0:
        continue
    out = dict(tag=row['tag'], m_GeV=row['m_GeV'], delta_keV=row['delta_keV'])
    # simple per-region ULs with mu_endo = 1 event (kappa = 1/Nendo_roi)
    out['f2_UL_lomid'] = S90['lomid'] / (0.866 * (row['R_lo'] + row['R_mid']))
    out['f2_UL_lo_Nmax3'] = 3.0 / row['R_lo'] if row['R_lo'] > 0 else np.inf    # P003/P016 recalled 2024 tolerance ~3 events in 5.4-55 keV
    out['f2_UL_gap'] = S90['gap'] / row['R_gap'] if row['R_gap'] > 0 else np.inf
    out['f2_UL_hi'] = S90['hi_extra'] / row['R_hi'] if row['R_hi'] > 0 else np.inf
    out['f2_UL_he1000'] = S90['he1000'] / row['R_he1000'] if row['R_he1000'] > 0 else np.inf
    out['f2_UL_he1700'] = S90['he1700'] / row['R_he1700'] if row['R_he1700'] > 0 else np.inf
    out['f2_UL_P011style_roi'] = 1.0 / row['R_roi'] if row['R_roi'] > 0 else np.inf   # <= 1 exothermic ROI event, mu_endo = 1
    # joint likelihood with kappa fixed by the endothermic interpretation (mu_endo,ROI = 1; LZ band 0.3-2.4)
    out['f2_UL_joint'] = fixedkappa_f2_UL(row)
    out['f2_UL_joint_mu0p3'] = fixedkappa_f2_UL(row, mu_endo=0.3)
    out['f2_UL_joint_mu2p4'] = fixedkappa_f2_UL(row, mu_endo=2.4)
    out['f2_UL_joint_he1000'] = fixedkappa_f2_UL(row, regions=('lomid', 'gap', 'hi', 'he1000'))
    out['f2_UL_joint_noLE'] = fixedkappa_f2_UL(row, regions=('hi', 'he1700'))
    out['f2_UL_joint_ROIonly'] = fixedkappa_f2_UL(row, regions=('lomid', 'gap', 'hi'))
    # kappa free (profiled): the exothermic population can replace the endothermic event -> degeneracy
    out['kappa_hat_f2_0'], out['q_f2_1_profiled'], out['kappa_hat_f2_1'], out['f2_UL_profiled'] = profile_f2(row)
    # lifetimes: f2(today) = f2,fo exp(-t_U/tau) ; excluded tau > tau_min
    for lab, f2fo in F2_FO.items():
        out[f'tau_min_s_{lab}'] = T_U_S / math.log(f2fo / out['f2_UL_joint']) if out['f2_UL_joint'] < f2fo else np.inf
    out['tau_min_s_darkphoton_mu0p3'] = T_U_S / math.log(0.5 / out['f2_UL_joint_mu0p3']) if out['f2_UL_joint_mu0p3'] < 0.5 else np.inf
    out['tau_min_s_darkphoton_mu2p4'] = T_U_S / math.log(0.5 / out['f2_UL_joint_mu2p4']) if out['f2_UL_joint_mu2p4'] < 0.5 else np.inf
    stat_rows.append(out)
stat = pd.DataFrame(stat_rows); stat.to_csv(f'{OUT}/P058_f2_limits.csv', index=False)
say('Part D: 90 % upper limits on f2 (annual halo, mu_endo = 1 for the simple limits; joint profile likelihood)')
say(stat.to_string(index=False, float_format='%.3g'))

# expected LZ counts (2.84 t yr) for given f2 at mu_endo = 1 (kappa = 1/Nendo_roi)
cnt_rows = []
for _, row in sel[(sel.m_GeV == 1000.0) & (sel.delta_keV.isin([300.0, 350.0, 366.0, 380.0]))].iterrows():
    for f2 in (1e-3, 1e-2, 0.1, 0.42):
        cnt_rows.append(dict(tag=row['tag'], delta_keV=row['delta_keV'], f2=f2, N_5_55=f2 * row['R_lo'], N_55_125=f2 * row['R_mid'], N_125_200=f2 * row['R_gap'],
                             N_200_270=f2 * row['R_hi'], N_600_1000phd=f2 * row['R_he1000'], N_600_1700phd=f2 * row['R_he1700'],
                             N_above_670keV_noeff=f2 * (row['Nexo_all'] - row['Nexo_roi'] / 0.96 - row['Nexo_he1700'] / PLATEAU_HE) / row['Nendo_roi']))
cnt = pd.DataFrame(cnt_rows); cnt.to_csv(f'{OUT}/P058_expected_counts_vs_f2.csv', index=False)
say('Part D: expected exothermic counts in 2.84 t yr for f2 = 1e-3, 1e-2, 0.1, 0.42 (mu_endo = 1)')
say(cnt.to_string(index=False, float_format='%.3g'))

# dark photon: sigma_p(N=1) from our proton-only kernels; chi2 -> chi1 nu nubar width (P011 eq. 3); eps and m_A' floors
def sigma_p_from_c(c, m):
    mup = m * MP / (m + MP)
    return c ** 2 * mup ** 2 / math.pi * lz.GEV_TO_CM2
SIGMA_P_UNIT = sigma_p_from_c(C_UNIT, 1000.0)
SW, CW = math.sqrt(SW2), math.sqrt(1 - SW2); G_WEAK = math.sqrt(4 * math.pi * ALPHA) / SW
def gamma_chi2(eps, aD, d_keV, n_nu=3):
    gD = math.sqrt(4 * math.pi * aD); Geff = gD * eps * (SW / CW) * G_WEAK / (4 * CW * MZ ** 2)
    return n_nu * Geff ** 2 * (d_keV * 1e-6) ** 5 / (120 * math.pi ** 3)
dp_rows = []
for _, s in stat[(stat.tag == 'p') & (stat.m_GeV == 1000.0)].iterrows():
    d = s['delta_keV']; row = sel[(sel.tag == 'p') & (sel.m_GeV == 1000.0) & np.isclose(sel.delta_keV, d)].iloc[0]
    sig_p_N1 = SIGMA_P_UNIT / row['Nendo_roi']                     # cm^2 for one endothermic ROI event
    X = (sig_p_N1 / lz.GEV_TO_CM2) / (16 * math.pi * ALPHA * (1000.0 * MP / (1000.0 + MP)) ** 2)   # eps^2 alpha_D / m_A'^4 [GeV^-4]
    tau_min = s['tau_min_s_darkphoton']
    # tau(m_A') = hbar/Gamma with eps^2 alpha_D = X m_A'^4 : Gamma = 3 (Geff/(gD eps))^2 4 pi X m_A'^4 delta^5/(120 pi^3)
    Geff_unit = (SW / CW) * G_WEAK / (4 * CW * MZ ** 2)
    tau0 = HBAR_GEV_S / (3 * Geff_unit ** 2 * 4 * math.pi * X * (d * 1e-6) ** 5 / (120 * math.pi ** 3))   # at m_A' = 1 GeV
    mA_min = (tau0 / tau_min) ** 0.25 if np.isfinite(tau_min) else np.nan
    eps_min_01 = math.sqrt(HBAR_GEV_S / gamma_chi2(1.0, 0.1, d) / tau_min) if np.isfinite(tau_min) else np.nan
    dp_rows.append(dict(delta_keV=d, sigma_p_N1_cm2=sig_p_N1, f2_UL_joint=s['f2_UL_joint'], tau_min_s=tau_min, tau_at_mA_1GeV_s=tau0,
                        mA_min_GeV=mA_min, eps_min_alphaD_0p1=eps_min_01, tau_min_s_P011style=T_U_S / math.log(1.0 / s['f2_UL_P011style_roi']) if s['f2_UL_P011style_roi'] < 1 else np.inf))
dp = pd.DataFrame(dp_rows); dp.to_csv(f'{OUT}/P058_darkphoton_floors.csv', index=False)
say('Part D: dark photon (1 TeV, proton-only): sigma_p(N=1), tau_min, tau(m_A=1 GeV), m_A floor, eps floor (alpha_D = 0.1)')
say(dp.to_string(index=False, float_format='%.3g'))

# ------------------------------------------------------------------------------------------------------------
# Part E: the event as an exothermic down-scatter (1000 GeV)
# ------------------------------------------------------------------------------------------------------------
ev_rows = []
for _, row in sel[sel.m_GeV == 1000.0].iterrows():
    d = row['delta_keV']; tag = row['tag']
    N_roi_exo_unit = row['Nexo_roi']
    kappa_exo = 1.0 / (0.5 * N_roi_exo_unit) if N_roi_exo_unit > 0 else np.nan     # coupling for 1 exothermic ROI event at f2 = 0.5
    endo_at_kappa_exo = kappa_exo * 0.5 * row['Nendo_roi']                            # (1-f2) chi1 endothermic events at that coupling
    Z_exo, k_exo = local_Z(row, exo_only=True)
    Z_endo, k_endo = (local_Z(row, endo_only=True) if row['Nendo_roi'] > 0 else (np.nan, np.nan))
    ev_rows.append(dict(tag=tag, delta_keV=d, E_star_keV=estar_keV(d, 1000.0), pct248_true=row['pct248_exo_accepted_true'], pct248_obs=row['pct248_exo_accepted_obs'],
                        obs_p16=row['obs_p16'], obs_p50=row['obs_p50'], obs_p84=row['obs_p84'],
                        frac_ROI_events_in_200_270=row['Nexo_hi'] / N_roi_exo_unit, gap_per_hi_event=row['Nexo_gap'] / row['Nexo_hi'],
                        lomid_per_hi_event=(row['Nexo_lo'] + row['Nexo_mid']) / row['Nexo_hi'], he1700_per_hi_event=row['Nexo_he1700'] / row['Nexo_hi'],
                        Z4bin_exo_origin=Z_exo, mu_hi_exo_origin=k_exo * row['Nexo_hi'], Z4bin_endo_origin=Z_endo, mu_hi_endo_origin=(k_endo * row['Nendo_hi'] if np.isfinite(k_endo) else np.nan),
                        companions_lo_per_ROI_event=row['Nexo_lo'] / N_roi_exo_unit, companions_5to125_per_ROI_event=(row['Nexo_lo'] + row['Nexo_mid']) / N_roi_exo_unit,
                        he1700_per_ROI_event=row['Nexo_he1700'] / N_roi_exo_unit, he1000_per_ROI_event=row['Nexo_he1000'] / N_roi_exo_unit,
                        P0_he1700=math.exp(-row['Nexo_he1700'] / N_roi_exo_unit), kappa_for_1_exo_event_f2_0p5=kappa_exo,
                        sigma_p_equiv_cm2=(SIGMA_P_UNIT * kappa_exo if tag == 'p' else np.nan), endo_events_at_that_coupling=endo_at_kappa_exo,
                        kappa_endo_N1=1.0 / row['Nendo_roi'] if row['Nendo_roi'] > 0 else np.nan))
evt = pd.DataFrame(ev_rows); evt.to_csv(f'{OUT}/P058_event_as_exothermic.csv', index=False)
say('Part E: the event as an exothermic down-scatter (1000 GeV, annual halo)')
say(evt.to_string(index=False, float_format='%.3g'))

# ------------------------------------------------------------------------------------------------------------
# Part F: annual modulation
# ------------------------------------------------------------------------------------------------------------
mod_rows = []; MODCURVES = {}
for tag, m, deltas in CONFIGS:
    for d in deltas:
        Kx, Kn = KERN[(tag, m, -d)], KERN[(tag, m, +d)]
        rx_roi = np.array([counts(E_EXO, Kx @ HALO_DAYS[k], 'roi') for k in HALO_DAYS]); rx_he = np.array([counts(E_EXO, Kx @ HALO_DAYS[k], 'he1700') for k in HALO_DAYS])
        rx_all = np.array([counts(E_EXO, Kx @ HALO_DAYS[k], 'all') for k in HALO_DAYS]); rn_roi = np.array([counts(E_ENDO, Kn @ HALO_DAYS[k], 'roi') for k in HALO_DAYS])
        MODCURVES[(tag, m, d)] = (rx_roi, rx_he, rx_all, rn_roi)
        def a1(y):
            if y.mean() <= 0: return np.nan, np.nan, np.nan
            ph = 2 * math.pi * (DAYS24 - 152.5) / 365.25
            c = 2 * np.mean(y * np.cos(ph)) / y.mean()
            return c, (y.max() - y.min()) / (y.max() + y.min()), float(DAYS24[np.argmax(y)])
        r = dict(tag=tag, m_GeV=m, delta_keV=d)
        for lab, y in (('exo_roi', rx_roi), ('exo_he1700', rx_he), ('exo_all', rx_all), ('endo_roi', rn_roi)):
            r[f'a1_{lab}'], r[f'peak_to_peak_{lab}'], r[f'day_max_{lab}'] = a1(y)
        mod_rows.append(r)
mod = pd.DataFrame(mod_rows); mod.to_csv(f'{OUT}/P058_modulation.csv', index=False)
say('Part F: annual modulation (cosine amplitude a1 relative to the mean, phase fixed at 2 June; peak-to-peak (max-min)/(max+min); day of maximum)')
say(mod.to_string(index=False, float_format='%.3g'))

# ------------------------------------------------------------------------------------------------------------
# Part G: figures  (palette: dataviz reference instance, fixed slot order)
# ------------------------------------------------------------------------------------------------------------
PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
INK, INK2, MUTED, SURF, GRID = '#0b0b0b', '#52514e', '#898781', '#fcfcfb', '#e1e0d9'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'axes.facecolor': SURF, 'figure.facecolor': SURF, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})

# Fig 1: spectra, isoscalar 1000 GeV, annual halo
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={'width_ratios': [1.35, 1]})
ax = axs[0]
ax.axvspan(5.4, 269.9, color='#f0efec', lw=0, zorder=0); ax.axvspan(E50_600, E50_1700, color='#e6e5e2', lw=0, zorder=0)
ax.text(120, 4e-5, 'WS ROI\n(5.4–270 keV)', ha='center', color=INK2, fontsize=8); ax.text(470, 4e-5, 'empty 600–1700 phd\n(272–670 keV, 0 events)', ha='center', color=INK2, fontsize=8)
for i, d in enumerate((300.0, 350.0, 380.0)):
    rx, rn = SPEC[('iso', 1000.0, d, 'annual')]
    ax.plot(E_EXO, rx, color=PAL[i], lw=2, label=f'exothermic, δ = {d:.0f} keV')
    ax.plot(E_ENDO, np.where(rn > 0, rn, np.nan), color=PAL[i], lw=1.6, ls='--', label=f'endothermic, δ = {d:.0f} keV')
    ax.plot([estar_keV(d, 1000.0)], [np.interp(estar_keV(d, 1000.0), E_EXO, rx)], 'o', color=PAL[i], ms=6, mec=SURF, mew=1)
ax.axvline(248, color=INK2, lw=1, ls=':'); ax.text(252, 2e-3, 'event\n248 keV', color=INK2, fontsize=8)
ax.set_yscale('log'); ax.set_xlim(0, 1000); ax.set_ylim(1e-5, 1e4)
ax.set_xlabel('true recoil energy E_R [keV]'); ax.set_ylabel('dR/dE per unit isoscalar coupling [events/(t·yr·keV)]')
ax.set_title('O₁ˢ, 1000 GeV, annual-mean Baxter halo: χ₂N→χ₁N (solid, f₂ = 1) vs χ₁N→χ₂N (dashed)', fontsize=9, loc='left')
ax.legend(fontsize=7.5, loc='lower right', ncol=1)
ax = axs[1]
for i, d in enumerate((300.0, 350.0, 380.0)):
    rx, _ = SPEC[('iso', 1000.0, d, 'annual')]
    cdf = np.cumsum(0.5 * (rx[1:] + rx[:-1]) * np.diff(E_EXO)); cdf = np.insert(cdf, 0, 0); cdf /= cdf[-1]
    ax.plot(E_EXO, cdf, color=PAL[i], lw=2, label=f'δ = {d:.0f} keV')
ax.axvline(269.9, color=INK2, lw=1, ls=':'); ax.axvline(E50_1700, color=INK2, lw=1, ls=':'); ax.axvline(248, color=INK2, lw=1, ls='-.')
ax.text(272, 0.05, '270', color=INK2, fontsize=8); ax.text(E50_1700 + 3, 0.05, f'{E50_1700:.0f}', color=INK2, fontsize=8)
ax.set_xlim(0, 1500); ax.set_ylim(0, 1); ax.set_xlabel('true recoil energy E_R [keV]'); ax.set_ylabel('cumulative fraction of the exothermic spectrum')
ax.set_title('Where the down-scattering recoils land', fontsize=9, loc='left'); ax.legend(fontsize=8, loc='lower right')
fig.tight_layout(); fig.savefig(f'{FIG}/P058_fig1_spectra.png', dpi=160); plt.close(fig)

# Fig 2: f2 limits, tau_min, m_A' floor vs delta
fig, axs = plt.subplots(1, 3, figsize=(13, 4.3), constrained_layout=True)
lab = {'iso': 'isoscalar O₁ˢ (LZ)', 'p': 'proton-only (dark photon)', 'hig': 'Higgsino Z (isovector-dominated)'}
ax = axs[0]
for i, tag in enumerate(('iso', 'p', 'hig')):
    s = stat[(stat.tag == tag) & (stat.m_GeV == 1000.0)].sort_values('delta_keV')
    ax.plot(s.delta_keV, s.f2_UL_joint, color=PAL[i], lw=2, marker='o', ms=5, label=lab[tag] + ', 1 TeV')
for j, m in enumerate((400.0, 4000.0)):
    s = stat[(stat.tag == 'iso') & (stat.m_GeV == m)].sort_values('delta_keV')
    ax.plot(s.delta_keV, s.f2_UL_joint, color=PAL[0], lw=1.2, ls=('--', ':')[j], marker=('s', '^')[j], ms=4, label=f'isoscalar, {m:.0f} GeV')
s = stat[(stat.tag == 'p') & (stat.m_GeV == 1000.0)].sort_values('delta_keV')
ax.plot(s.delta_keV, s.f2_UL_P011style_roi, color=MUTED, lw=1, ls='-.', label='P011 criterion (≤1 ROI event)')
ax.axhline(0.42, color=INK2, lw=0.8, ls=':'); ax.text(252, 0.5, 'freeze-out f₂ = 0.42–0.50 (P026)', color=INK2, fontsize=7.5)
ax.set_yscale('log'); ax.set_xlabel('δ [keV]'); ax.set_ylabel('90 % CL upper limit on f₂ (today)'); ax.set_title('Excited-state fraction', fontsize=9, loc='left'); ax.legend(fontsize=7)
ax = axs[1]
for i, tag in enumerate(('iso', 'p', 'hig')):
    s = stat[(stat.tag == tag) & (stat.m_GeV == 1000.0)].sort_values('delta_keV')
    ax.plot(s.delta_keV, s.tau_min_s_darkphoton, color=PAL[i], lw=2, marker='o', ms=5, label=lab[tag])
ax.axhline(T_U_S, color=INK2, lw=0.8, ls=':'); ax.text(252, T_U_S * 1.15, 't_U = 4.35×10¹⁷ s', color=INK2, fontsize=7.5)
ax.axhline(7.2e16, color=MUTED, lw=0.8, ls='-.'); ax.text(252, 7.2e16 * 0.6, 'P011/P026: 7×10¹⁶ s', color=MUTED, fontsize=7.5)
ax.set_yscale('log'); ax.set_xlabel('δ [keV]'); ax.set_ylabel('τ(χ₂) excluded above [s]  (f₂,fo = 0.5)'); ax.set_title('Lifetime: everything above the curve is excluded', fontsize=9, loc='left'); ax.legend(fontsize=7)
ax = axs[2]
ax.plot(dp.delta_keV, dp.mA_min_GeV, color=PAL[1], lw=2, marker='o', ms=5, label='this work (joint likelihood)')
ax.plot([300, 350, 365, 380], [2.45, 0.92, 0.55, 0.25], 's', color=MUTED, ms=5, label='P026 (P011 criterion)')
ax.set_yscale('log'); ax.set_xlabel('δ [keV]'); ax.set_ylabel("m_A′ floor [GeV]  (α_D-independent)"); ax.set_title('Dark-photon mass floor from χ₂ decay', fontsize=9, loc='left'); ax.legend(fontsize=7)
fig.savefig(f'{FIG}/P058_fig2_f2_tau_mA.png', dpi=160); plt.close(fig)

# Fig 3: modulation curves
fig, ax = plt.subplots(figsize=(7.2, 4.0))
for i, d in enumerate((300.0, 350.0, 380.0)):
    rx_roi, rx_he, rx_all, rn_roi = MODCURVES[('iso', 1000.0, d)]
    ax.plot(DAYS24, rx_roi / rx_roi.mean(), color=PAL[i], lw=2, marker='o', ms=3, label=f'exothermic (ROI), δ = {d:.0f} keV')
rx_roi, rx_he, rx_all, rn_roi = MODCURVES[('iso', 1000.0, 300.0)]
ax.plot(DAYS24, rn_roi / rn_roi.mean(), color=INK2, lw=1.4, ls='--', label='endothermic (ROI), δ = 300 keV')
ax.axvline(152.5, color=MUTED, lw=0.8, ls=':'); ax.text(155, 0.6, '2 June', color=MUTED, fontsize=8); ax.axvline(167, color=MUTED, lw=0.8, ls='-.'); ax.text(170, 0.52, 'event', color=MUTED, fontsize=8)
ax.set_xlabel('day of year'); ax.set_ylabel('rate / annual mean'); ax.set_xlim(0, 365); ax.set_title('Annual modulation: χ₂ down-scattering is nearly flat', fontsize=9, loc='left'); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f'{FIG}/P058_fig3_modulation.png', dpi=160); plt.close(fig)

# ------------------------------------------------------------------------------------------------------------
# results JSON
# ------------------------------------------------------------------------------------------------------------
res = dict(constants=dict(C_UNIT_GeV2=C_UNIT, sigma_p_unit_cm2_1TeV=SIGMA_P_UNIT, c_p_hig=C_P_HIG, c_n_hig=C_N_HIG, t_U_s=T_U_S, f2_freezeout=F2_FO,
                          E50_600=E50_600, E50_1000=E50_1000, E50_1700=E50_1700, sig_1700=SIG_1700, plateau_he=PLATEAU_HE),
           observed=OBS, s90=S90, wimpydd_negative_delta_check=val_wd, days24=DAYS24.tolist(),
           kinematics_1TeV={str(int(d)): dict(E_star=estar_keV(d, 1000.0), halfwidth_v250=float(kin[(kin.m_GeV == 1000) & (kin.delta_keV == d)].halfwidth_v250_keV.iloc[0]),
                                              E_minus_vmax=float(kin[(kin.m_GeV == 1000) & (kin.delta_keV == d)].E_minus_vmax810.iloc[0]),
                                              E_plus_vmax=float(kin[(kin.m_GeV == 1000) & (kin.delta_keV == d)].E_plus_vmax810.iloc[0])) for d in (250.0, 300.0, 350.0, 366.0, 380.0)},
           delta_for_Estar248=ev.to_dict(orient='records'), counts_annual=sel.to_dict(orient='records'), f2_limits=stat.to_dict(orient='records'),
           darkphoton=dp.to_dict(orient='records'), event_as_exothermic=evt.to_dict(orient='records'), modulation=mod.to_dict(orient='records'),
           runtime_s=time.time() - T0)
json.dump(res, open(f'{OUT}/P058_results.json', 'w'), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
say(f'done in {time.time() - T0:.0f} s')
LOG.close()
