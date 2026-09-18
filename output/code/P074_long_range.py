"""
P074_long_range.py -- Long-range operators (q^-2, q^-4) versus the LZ 248 keV event: a systematic map of which
mediator masses and operator structures can leave a lone 248 keV recoil.

For every NR operator O_i (i = 1, 3, ..., 15; spin-1/2 WIMP; isoscalar, plus O1 proton-only) the contact Wilson
coefficient is multiplied by a propagator factor
        c_i(q) = c_i^0 [ m_med^2 / (m_med^2 + q^2) ]^n ,      n = 1 (one mediator), n = 2 (two mediators / dipole-dipole),
for m_med = 0.01 ... 10 GeV and the pure long-range limit m_med -> 0 (c ~ q^-2n, rate ~ q^-4n).  Elastic scattering at
fixed E_R fixes q^2 = 2 m_A E_R, so the RATE with the propagator is the contact rate times P_n(E) = [m^2/(m^2+q^2)]^(2n)
(exact per isotope; we evaluate q at the mean xenon mass A = 131.29, which P051 validated at <= 6 % and which we re-validate
here against full WimPyDD q-dependent coefficient closures).

Parts / stages (run from the simulation root with .venv/bin/python; each stage caches its WimPyDD output):
  --stage contact   : contact spectra dR/dE (events/t/yr/keV, unit coupling) for O1, O3-O15 isoscalar and O1 proton-only at 1 TeV
  --stage validate  : direct WimPyDD spectra with c(q) closures for a few (operator, m_med, n) cases -> compare with factorisation
  --stage dipole    : photon-mediated magnetic dipole (P012 coefficients c1, c4 constant; c5, c6 ~ 1/q^2) split into pieces
  --stage analysis  : N_lo(i, m_med, n) table and heat map; m_med,min(i); survivors' percentile / couplings / 100-200 ratio;
                      analytic long-range scaling; operator collapse (O_i x q^-2 -> lower operator); dipole with a massive mediator;
                      UV summary table; figures; JSON/CSV under output/work/P074/
  --stage all       : everything (default)
N_lo = R(5.4-55 keV) / R(200-270 keV), efficiency-weighted with the P003 model (0.96 plateau, erf 50 % at 5.4 / 269.9 keV).
Halo: lz.wd_halo() = WimPyDD single-day SUN-FRAME Baxter-2021 SHM (v_E = 250.6 km/s, no Earth orbital motion), labelled as such.
"""
from __future__ import annotations
import sys, os, json, math, time
sys.path.insert(0, 'output/code')
import numpy as np
import pandas as pd
from scipy import special, integrate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import lzcommon as lz

OUT = 'output/work/P074'
FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)
STAGE = sys.argv[sys.argv.index('--stage') + 1] if '--stage' in sys.argv else 'all'
T0 = time.time()
LOG = open(os.path.join(OUT, 'P074_run.log'), 'a')
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()
log(f'===== P074 run {time.strftime("%Y-%m-%d %H:%M:%S")} stage={STAGE} =====')

# ------------------------------------------------------------------------------------------------
# A. constants, efficiency, window integrals
# ------------------------------------------------------------------------------------------------
MN = lz.M_NUCLEON_GEV                     # 0.938272 GeV
MV2 = lz.M_V_GEV ** 2                     # (246.2 GeV)^2
A_MEAN = lz.A_XE_MEAN                     # 131.29
M_XE = lz.m_nucleus_gev(A_MEAN)           # GeV
M_CHI = 1000.0
EXPOSURE = lz.LZ['exposure_tyr']          # 2.84 t yr
EXPOSURE_2024 = 4.2                       # t yr (recalled, certain; as P012/P044)
ALPHA_EM = 1.0 / 137.036                  # recalled, certain
E_CH = math.sqrt(4 * math.pi * ALPHA_EM)  # 0.30282
G_P, G_N = 5.5857, -3.8261                # recalled, certain
MU_NUC_GEV = 1.0 / (2 * MN)               # nuclear magneton in GeV^-1 (e = 1 units are carried by E_CH explicitly)
OPS = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
MMED = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, np.inf]      # GeV; inf = contact
MMED_FINE = np.concatenate([np.logspace(-2, 1, 61), [np.inf]])
NPOW = [1, 2]
C_UNIT_WD = lz.wd_c_from_anand(1.0 / MV2, 0.0)             # (2/m_v^2, 0): LZ/Anand unit isoscalar coupling in WimPyDD

def efficiency(E, sig_lo=3.4, sig_hi=8.0, eff0=lz.LZ['eff_plateau']):
    E = np.asarray(E, float)
    lo, hi = lz.LZ['E_50pct_low_keV'], lz.LZ['E_50pct_high_keV']
    return eff0 * 0.5 * (1 + special.erf((E - lo) / (math.sqrt(2) * sig_lo))) * 0.5 * (1 - special.erf((E - hi) / (math.sqrt(2) * sig_hi)))

def q_gev(E_kev, mA=M_XE):
    return np.sqrt(2.0 * mA * np.asarray(E_kev, float) * 1e-6)

def prop_rate_factor(E_kev, m_med, n):
    """rate factor [m^2/(m^2+q^2)]^(2n); m_med = inf -> 1; m_med = 0 -> (q_ref^2/q^2)^(2n) with q_ref = 1 GeV (shape only)."""
    q2 = q_gev(E_kev) ** 2
    if not np.isfinite(m_med):
        return np.ones_like(q2)
    if m_med == 0:
        return (1.0 / q2) ** (2 * n)
    return (m_med ** 2 / (m_med ** 2 + q2)) ** (2 * n)

# energy grid: fine at low energy (1/E^2 spectra), 3 keV above 30 keV, plus reference energies
E = np.union1d(np.union1d(np.arange(1.0, 30.0 + 1e-9, 1.0), np.arange(30.0, 330.0 + 1e-9, 3.0)),
               [5.4, 55.0, 100.0, 200.0, 248.0, 269.9, 300.0])
EF_LO, EF_HI = np.linspace(5.4, 55.0, 2001), np.linspace(200.0, 270.0, 1401)
EF_MID = np.linspace(100.0, 200.0, 2001)
EF_ROI = np.linspace(1.0, 330.0, 6581)

def interp_spec(dRdE, Ef, Egrid=E):
    y = np.asarray(dRdE, float)
    if np.all(y > 0):
        return 10 ** np.interp(Ef, Egrid, np.log10(y))
    return np.interp(Ef, Egrid, y)

def wrate(dRdE, Ef):
    return float(integrate.trapezoid(interp_spec(dRdE, Ef) * efficiency(Ef), Ef))

def summarise(dRdE):
    R_lo, R_hi, R_mid, R_roi = wrate(dRdE, EF_LO), wrate(dRdE, EF_HI), wrate(dRdE, EF_MID), wrate(dRdE, EF_ROI)
    y = interp_spec(dRdE, EF_ROI) * efficiency(EF_ROI)
    cum = integrate.cumulative_trapezoid(y, EF_ROI, initial=0.0)
    pct248 = float(np.interp(248.0, EF_ROI, cum) / cum[-1])
    # observed-energy percentile with Gaussian resolution sigma = 23 sqrt(E/248) keV (LZ stat error at 248 keV; sqrt(E) scaling recalled/likely)
    sig = 23.0 * np.sqrt(np.maximum(EF_ROI, 1.0) / 248.0)
    Eo = np.linspace(-50.0, 400.0, 901)
    ker = np.exp(-0.5 * ((Eo[:, None] - EF_ROI[None, :]) / sig[None, :]) ** 2) / (sig[None, :] * math.sqrt(2 * math.pi))
    yo = ker @ y * (EF_ROI[1] - EF_ROI[0])
    cumo = integrate.cumulative_trapezoid(yo, Eo, initial=0.0)
    pct248_obs = float(np.interp(248.0, Eo, cumo) / cumo[-1])
    Epk = float(EF_ROI[np.argmax(y)])
    return dict(R_lo=R_lo, R_hi=R_hi, R_100_200=R_mid, R_roi=R_roi, N_lo=R_lo / R_hi, ratio_100_200_over_hi=R_mid / R_hi,
                f_hi=R_hi / R_roi, pct248_true=pct248, pct248_obs=pct248_obs, E_peak_keV=Epk)

WD = None; HALO = None
def wd_init():
    global WD, HALO
    if WD is None:
        WD = lz.wd(); HALO = lz.wd_halo()          # Sun-frame Baxter SHM, explicit v_min grid (P002 fix)
        log(f'WimPyDD loaded; Sun-frame halo v_max = {HALO[0][-1]:.0f} km/s; E grid {len(E)} points')

def const_fn(c0, c1):
    c0, c1 = float(c0), float(c1)
    def f():
        return [c0, c1]
    return f
def prop_fn(c0, c1, m_med, n):
    c0, c1, m2, n = float(c0), float(c1), float(m_med) ** 2, int(n)
    def f(q):
        g = (m2 / (m2 + q ** 2)) ** n
        return [c0 * g, c1 * g]
    return f
def qpow_fn(c0, c1, power):
    c0, c1, power = float(c0), float(c1), float(power)
    def f(q):
        g = q ** power
        return [c0 * g, c1 * g]
    return f

# ------------------------------------------------------------------------------------------------
# B. contact spectra (cached)
# ------------------------------------------------------------------------------------------------
CPATH = os.path.join(OUT, 'P074_contact_spectra.npz')
def label(i, iso='s'):
    return f'O{i}{iso}'
if STAGE in ('contact', 'all'):
    wd_init()
    C = dict(np.load(CPATH)) if os.path.exists(CPATH) else {}
    todo = [(i, 's') for i in OPS] + [(1, 'p')]
    for i, iso in todo:
        key = label(i, iso)
        if key in C:
            continue
        c = C_UNIT_WD if iso == 's' else (1.0 / MV2, 1.0 / MV2)     # proton-only: c_p = 1/m_v^2, c_n = 0
        h = WD.eft_hamiltonian(f'P074_{key}', {i: const_fn(*c)})
        C[key] = lz.wd_rate(h, M_CHI, E, HALO)
        C['E'] = E
        np.savez(CPATH, **C)
        s = summarise(C[key])
        log(f'  contact {key:5s}: N_lo = {s["N_lo"]:9.3g}  dR/dE(248) = {C[key][np.searchsorted(E, 248.0)]:.3e}  ({time.time()-T0:.0f}s)')
    log(f'contact stage done ({time.time()-T0:.0f}s)')

# ------------------------------------------------------------------------------------------------
# C. validation of the propagator factorisation against full WimPyDD q-dependent closures (cached)
# ------------------------------------------------------------------------------------------------
VPATH = os.path.join(OUT, 'P074_validation.csv')
VAL_CASES = [(1, 's', 0.1, 1), (1, 's', 0.3, 2), (6, 's', 0.3, 1), (6, 's', 0.1, 2), (10, 's', 0.3, 1), (9, 's', 1.0, 1),
             (4, 's', 0.1, 1), (11, 's', 0.03, 1), (1, 'p', 0.1, 1), (5, 's', 0.3, 1)]
VAL_E = np.array([5.4, 10.0, 20.0, 40.0, 55.0, 100.0, 150.0, 200.0, 248.0, 269.9, 300.0])
if STAGE in ('validate', 'all') and not os.path.exists(VPATH):
    wd_init()
    C = dict(np.load(CPATH))
    rows = []
    for i, iso, m, n in VAL_CASES:
        c = C_UNIT_WD if iso == 's' else (1.0 / MV2, 1.0 / MV2)
        h = WD.eft_hamiltonian(f'P074_val_{label(i, iso)}_{m}_{n}', {i: prop_fn(c[0], c[1], m, n)})
        full = lz.wd_rate(h, M_CHI, VAL_E, HALO)
        fact = interp_spec(C[label(i, iso)], VAL_E) * prop_rate_factor(VAL_E, m, n)
        for e, a, b in zip(VAL_E, full, fact):
            rows.append(dict(op=label(i, iso), m_med=m, n=n, E_keV=e, wimpydd_full=a, factorised=b, ratio=a / b if b > 0 else np.nan))
        # window-integrated check
        Efull = np.union1d(np.arange(2.0, 330.0, 6.0), [5.4, 55.0, 200.0, 248.0, 269.9])
        fs = lz.wd_rate(h, M_CHI, Efull, HALO)
        Nlo_full = (integrate.trapezoid(np.interp(EF_LO, Efull, fs) * efficiency(EF_LO), EF_LO) /
                    integrate.trapezoid(np.interp(EF_HI, Efull, fs) * efficiency(EF_HI), EF_HI))
        Nlo_fact = summarise(C[label(i, iso)] * prop_rate_factor(E, m, n))['N_lo']
        rows.append(dict(op=label(i, iso), m_med=m, n=n, E_keV=-1, wimpydd_full=Nlo_full, factorised=Nlo_fact, ratio=Nlo_full / Nlo_fact))
        log(f'  validate {label(i, iso)} m={m} n={n}: N_lo full {Nlo_full:.4g} vs factorised {Nlo_fact:.4g}; '
            f'pointwise ratio range {min(r["ratio"] for r in rows if r["op"]==label(i,iso) and r["m_med"]==m and r["n"]==n and r["E_keV"]>0):.4f}-'
            f'{max(r["ratio"] for r in rows if r["op"]==label(i,iso) and r["m_med"]==m and r["n"]==n and r["E_keV"]>0):.4f}  ({time.time()-T0:.0f}s)')
    pd.DataFrame(rows).to_csv(VPATH, index=False, float_format='%.6g')
    log(f'validation stage done ({time.time()-T0:.0f}s)')

# ------------------------------------------------------------------------------------------------
# D. photon-mediated magnetic dipole, decomposed (cached).  P012 coefficients (recalled/likely; directdm-consistent per P031):
#    c1^N = e mu Q_N/(2 m_chi), c5^N = 2 e mu m_N Q_N/q^2, c4^N = e mu g_N/m_N, c6^N = -e mu g_N m_N/q^2 ;  mu = 1 mu_N
# ------------------------------------------------------------------------------------------------
DPATH = os.path.join(OUT, 'P074_dipole_spectra.npz')
def dipole_ham(name, pieces):
    mu = MU_NUC_GEV
    wc = {}
    if 'c1' in pieces:
        v = E_CH * mu / (2 * M_CHI); wc[1] = const_fn(v, v)                       # proton only -> (c_p, c_p)
    if 'c5' in pieces:
        v = 2 * E_CH * mu * MN; wc[5] = qpow_fn(v, v, -2)
    if 'c4' in pieces:
        wc[4] = const_fn(E_CH * mu * (G_P + G_N) / MN, E_CH * mu * (G_P - G_N) / MN)
    if 'c6' in pieces:
        wc[6] = qpow_fn(-E_CH * mu * (G_P + G_N) * MN, -E_CH * mu * (G_P - G_N) * MN, -2)
    return WD.eft_hamiltonian('P074_dip_' + name, wc)
DIP_PIECES = {'full': ('c1', 'c5', 'c4', 'c6'), 'charge_c1c5': ('c1', 'c5'), 'spin_c4c6': ('c4', 'c6'),
              'c1_only': ('c1',), 'c5_only': ('c5',), 'c4_only': ('c4',), 'c6_only': ('c6',)}
if STAGE in ('dipole', 'all'):
    wd_init()
    D = dict(np.load(DPATH)) if os.path.exists(DPATH) else {}
    for name, pieces in DIP_PIECES.items():
        if name in D:
            continue
        D[name] = lz.wd_rate(dipole_ham(name, pieces), M_CHI, E, HALO)
        D['E'] = E
        np.savez(DPATH, **D)
        s = summarise(D[name])
        log(f'  dipole {name:12s}: N_lo = {s["N_lo"]:9.4g}  R_lo = {s["R_lo"]:.3e} R_hi = {s["R_hi"]:.3e} ({time.time()-T0:.0f}s)')
    log(f'dipole stage done ({time.time()-T0:.0f}s)')

# ------------------------------------------------------------------------------------------------
# E. analysis
# ------------------------------------------------------------------------------------------------
if STAGE in ('analysis', 'all'):
    C = dict(np.load(CPATH))
    RES = dict(settings=dict(m_chi_GeV=M_CHI, halo='lz.wd_halo(): WimPyDD single-day Sun-frame Baxter-2021 SHM (v_E = 250.6 km/s)',
                             efficiency='P003 model: 0.96 plateau, erf 50% at 5.4 and 269.9 keV, widths 3.4/8 keV',
                             E_grid=f'{len(E)} points, 1 keV steps to 30 keV, 3 keV to 330 keV', m_med_GeV=[str(m) for m in MMED],
                             propagator='c(q) = c0 [m^2/(m^2+q^2)]^n, rate factor [m^2/(m^2+q^2)]^(2n), q^2 = 2 m_A E_R at A = 131.29',
                             coupling='isoscalar LZ/Anand unit c^s = 1/m_v^2 = WimPyDD c^0 = 2/m_v^2; O1p: c_p = 1/m_v^2, c_n = 0',
                             exposure_tyr=EXPOSURE))
    # ---- E1. N_lo(i, m_med, n) table -------------------------------------------------------------
    rows = []
    for key in [label(i) for i in OPS] + ['O1p']:
        base = C[key]
        for n in NPOW:
            for m in MMED + [0.0]:
                sp = base * prop_rate_factor(E, m, n)
                s = summarise(sp)
                N_hi_unit = s['R_hi'] * EXPOSURE          # events in 200-270 keV at unit coupling (c0 = q->0 coefficient)
                g_hi = math.sqrt(1.0 / N_hi_unit) if N_hi_unit > 0 and np.isfinite(m) and m > 0 else np.nan
                rows.append(dict(op=key, n=n, m_med_GeV=m, **s, N_hi_unit=N_hi_unit,
                                 c_anand_1hi=g_hi / MV2, c_mv2_1hi=g_hi, Lambda_1hi_GeV=(MV2 / g_hi) ** 0.5 if np.isfinite(g_hi) else np.nan,
                                 gchi_gN_1hi=(g_hi / MV2) * m ** 2 if (np.isfinite(g_hi) and n == 1) else np.nan))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, 'P074_Nlo_table.csv'), index=False, float_format='%.5g')
    # wide table for the paper
    for n in NPOW:
        W = T[T.n == n].pivot(index='op', columns='m_med_GeV', values='N_lo')
        W = W.reindex([label(i) for i in OPS] + ['O1p'])
        W.columns = [('contact' if not np.isfinite(c) else ('q^-%d (m->0)' % (2 * n) if c == 0 else f'{c:g}')) for c in W.columns]
        W.to_csv(os.path.join(OUT, f'P074_Nlo_wide_n{n}.csv'), float_format='%.4g')
        log(f'\nN_lo(op, m_med) for n = {n} (1 TeV, Sun-frame halo):\n' + W.to_string(float_format=lambda x: f'{x:.3g}'))
    # ---- E2. m_med,min per operator (fine grid), for N_lo <= 5 and <= 3; and the "contact-like" mass where N_lo = 1.5 x contact ---
    mrows = []
    for key in [label(i) for i in OPS] + ['O1p']:
        base = C[key]; Nc = summarise(base)['N_lo']
        for n in NPOW:
            Nf = np.array([summarise(base * prop_rate_factor(E, m, n))['N_lo'] for m in MMED_FINE])
            mf = MMED_FINE[:-1]; Nff = Nf[:-1]                # finite masses (log-spaced 0.01-10 GeV)
            def first_mass_below(thr):
                ok = np.where(Nff <= thr)[0]
                if len(ok) == 0:
                    return np.inf if Nc > thr else np.nan
                j = ok[0]
                if j == 0:
                    return 0.0
                # log-interpolate crossing between j-1 and j
                x0, x1, y0, y1 = math.log10(mf[j - 1]), math.log10(mf[j]), math.log10(Nff[j - 1]), math.log10(Nff[j])
                return 10 ** (x0 + (math.log10(thr) - y0) * (x1 - x0) / (y1 - y0))
            m5, m3 = first_mass_below(5.0), first_mass_below(3.0)
            m15 = first_mass_below(1.5 * Nc)           # contact-like: within 50 % of the contact N_lo
            m2 = first_mass_below(2.0 * Nc)
            N_lr = float(Nf[0]) if np.isfinite(Nf[0]) else np.nan
            # local log-slope of N_lo(m) between 0.1 and 0.3 GeV and the transition mass (log-midpoint between contact and m->0 values)
            N01, N03 = float(np.interp(math.log10(0.1), np.log10(mf), np.log10(Nff))), float(np.interp(math.log10(0.3), np.log10(mf), np.log10(Nff)))
            slope = (N03 - N01) / (math.log10(0.3) - math.log10(0.1))
            Nlr = summarise(base * prop_rate_factor(E, 0.0, n))['N_lo']
            mid = 0.5 * (math.log10(Nc) + math.log10(Nlr))
            j = np.where(np.log10(Nff) <= mid)[0]
            m_trans = 10 ** np.interp(mid, np.log10(Nff[::-1]), np.log10(mf[::-1])) if len(j) else np.nan
            mrows.append(dict(op=key, n=n, N_lo_contact=Nc, N_lo_longrange=Nlr, N_lo_0p01=float(Nf[0]), m_min_Nlo5_GeV=m5, m_min_Nlo3_GeV=m3,
                              m_contactlike_1p5_GeV=m15, m_contactlike_2_GeV=m2, m_transition_GeV=m_trans, logslope_0p1_0p3=slope))
    M = pd.DataFrame(mrows)
    M.to_csv(os.path.join(OUT, 'P074_mmed_min.csv'), index=False, float_format='%.4g')
    log('\nm_med,min per operator:\n' + M.to_string(index=False, float_format=lambda x: f'{x:.3g}'))
    # ---- E3. survivors ------------------------------------------------------------------------
    surv = T[(T.N_lo <= 5.0) & (T.m_med_GeV > 0)].copy()
    surv.to_csv(os.path.join(OUT, 'P074_survivors.csv'), index=False, float_format='%.5g')
    log('\nSurvivors (N_lo <= 5):\n' + surv[['op', 'n', 'm_med_GeV', 'N_lo', 'pct248_true', 'pct248_obs', 'E_peak_keV', 'ratio_100_200_over_hi',
                                         'N_hi_unit', 'c_anand_1hi', 'Lambda_1hi_GeV', 'gchi_gN_1hi']].to_string(index=False, float_format=lambda x: f'{x:.4g}'))
    # ---- E4. analytic understanding of the long-range limit for O1 ------------------------------
    base = C['O1s']
    Ncon = summarise(base)['N_lo']
    def wmean(sp, Ef, fn):
        y = interp_spec(sp, Ef) * efficiency(Ef)
        return float(integrate.trapezoid(y * fn(Ef), Ef) / integrate.trapezoid(y, Ef))
    kin_ratio = (1 / 5.4 - 1 / 55.0) / (1 / 200.0 - 1 / 270.0)                       # pure 1/E^2 spectrum, no F^2, no halo, no efficiency
    kin_ratio_eff = (float(integrate.trapezoid(efficiency(EF_LO) / EF_LO ** 2, EF_LO)) / float(integrate.trapezoid(efficiency(EF_HI) / EF_HI ** 2, EF_HI)))
    mE2_lo, mE2_hi = wmean(base, EF_LO, lambda x: x ** -2), wmean(base, EF_HI, lambda x: x ** -2)
    mE4_lo, mE4_hi = wmean(base, EF_LO, lambda x: x ** -4), wmean(base, EF_HI, lambda x: x ** -4)
    N_q4 = summarise(base * prop_rate_factor(E, 0.0, 1))['N_lo']
    N_q8 = summarise(base * prop_rate_factor(E, 0.0, 2))['N_lo']
    anal = dict(N_lo_contact=Ncon, kinematic_ratio_int_dE_over_E2=kin_ratio, kinematic_ratio_with_efficiency=kin_ratio_eff,
                rate_weighted_mean_Einv2_lo=mE2_lo, rate_weighted_mean_Einv2_hi=mE2_hi, E_eff_lo_keV=mE2_lo ** -0.5, E_eff_hi_keV=mE2_hi ** -0.5,
                predicted_ratio_q4_over_contact=mE2_lo / mE2_hi, N_lo_q4_predicted=Ncon * mE2_lo / mE2_hi, N_lo_q4_numerical=N_q4,
                predicted_ratio_q8_over_contact=mE4_lo / mE4_hi, N_lo_q8_predicted=Ncon * mE4_lo / mE4_hi, N_lo_q8_numerical=N_q8,
                q_at_30keV_GeV=float(q_gev(30.0)), q_at_248keV_GeV=float(q_gev(248.0)), q_at_5p4keV_GeV=float(q_gev(5.4)), q_at_55keV_GeV=float(q_gev(55.0)))
    # power-law scaling of N_lo(m) for O1 n=1 and n=2 on the fine grid
    for n in NPOW:
        Nf = np.array([summarise(base * prop_rate_factor(E, m, n))['N_lo'] for m in MMED_FINE[:-1]])
        anal[f'O1s_n{n}_Nlo_fine'] = dict(m_GeV=MMED_FINE[:-1].tolist(), N_lo=Nf.tolist())
        sl = np.gradient(np.log10(Nf), np.log10(MMED_FINE[:-1]))
        anal[f'O1s_n{n}_min_logslope'] = float(sl.min()); anal[f'O1s_n{n}_m_at_min_slope_GeV'] = float(MMED_FINE[:-1][np.argmin(sl)])
        anal[f'O1s_n{n}_expected_slope_intermediate'] = -4 * n
    log('\nAnalytic long-range check (O1s): ' + json.dumps({k: v for k, v in anal.items() if not isinstance(v, dict)}, default=float))
    # ---- E5. operator collapse: O_i x q^-2n in the m -> 0 limit vs. the contact operator with matching q power ------------
    Ncontact = {k: summarise(C[k])['N_lo'] for k in C if k != 'E'}
    collapse = []
    # Power counting: rate ~ |c(q)|^2 x (intrinsic q-power of the operator response).  One q^-2 propagator (n = 1) divides the RATE by
    # q^4, so only operators whose response carries q^4 (O6: Sigma'' q^4; O15: Sigma' q^4 + Phi'' q^4 v^2) return to a q^0 contact shape;
    # q^2-response operators (O9, O10, O11, O5, O3, O13, O14) overshoot to q^-2 and q^0 operators (O1, O4, O7, O8, O12) to q^-4.
    pairs = [('O6s', 1, 'O4s', 'rate Sigma\'\' q^4 / q^4 -> Sigma\'\' at q^0: O4-like shape (O4 = Sigma\' + Sigma\'\')'),
             ('O15s', 1, 'O12s', 'rate (Sigma\' q^4 + Phi\'\' q^4 v^2)/q^4 -> q^0 velocity/spin shape: O12-like'),
             ('O6s', 2, 'O4s', 'rate Sigma\'\' q^4 / q^8 -> Sigma\'\'/q^4: softer than any contact operator'),
             ('O10s', 1, 'O4s', 'rate Sigma\'\' q^2 / q^4 -> Sigma\'\'/q^2: overshoots O4 (reference) toward O4/q^2'),
             ('O9s', 1, 'O4s', 'rate Sigma\' q^2 / q^4 -> Sigma\'/q^2: overshoots O4'),
             ('O11s', 1, 'O1s', 'rate M q^2 / q^4 -> M/q^2: overshoots O1 (P044 photon EDM is the O11/q^2 amplitude, rate M/q^2 -> 18 100)'),
             ('O5s', 1, 'O8s', 'rate (M q^2 v^2 + Delta q^4)/q^4 -> M v^2/q^2 + Delta: overshoots O8 (M v^2 + Delta q^2)'),
             ('O3s', 1, 'O7s', 'rate (Phi\'\' q^2 + M q^2 v^2)/q^4 -> q^-2 velocity-suppressed: overshoots O7'),
             ('O14s', 1, 'O7s', 'rate Sigma\' q^2 v^2 / q^4 -> Sigma\' v^2/q^2: overshoots O7 (Sigma\' v^2)'),
             ('O13s', 1, 'O12s', 'rate (Sigma\'\' + Phi~\') q^2 v^2 / q^4 -> q^-2: overshoots O12'),
             ('O12s', 1, None, 'rate (Sigma v^2 + Phi q^2)/q^4 -> softer than any contact operator'),
             ('O4s', 1, None, 'rate Sigma/q^4 -> softer than any contact operator'),
             ('O1s', 1, None, 'rate M/q^4 (light-mediator SI) -> dR/dE ~ F^2 eta / E^2')]
    for key, n, ref, note in pairs:
        Nlr = summarise(C[key] * prop_rate_factor(E, 0.0, n))['N_lo']
        collapse.append(dict(op=key, n=n, N_lo_longrange=Nlr, N_lo_contact_same_op=Ncontact[key], reference_op=ref,
                             N_lo_reference_contact=Ncontact[ref] if ref else np.nan, ratio_to_reference=Nlr / Ncontact[ref] if ref else np.nan, note=note))
    COL = pd.DataFrame(collapse)
    COL.to_csv(os.path.join(OUT, 'P074_operator_collapse.csv'), index=False, float_format='%.4g')
    log('\nOperator collapse in the m -> 0 limit:\n' + COL[['op', 'n', 'N_lo_longrange', 'N_lo_contact_same_op', 'reference_op', 'N_lo_reference_contact', 'ratio_to_reference']].to_string(index=False, float_format=lambda x: f'{x:.3g}'))
    # ---- E6. photon dipole decomposition and a massive dark-photon-mediated dipole -----------------
    dip = {}
    if os.path.exists(DPATH):
        D = dict(np.load(DPATH))
        full = summarise(D['full'])
        for name in DIP_PIECES:
            s = summarise(D[name])
            dip[name] = dict(N_lo=s['N_lo'], R_lo=s['R_lo'], R_hi=s['R_hi'], frac_R_lo=s['R_lo'] / full['R_lo'], frac_R_hi=s['R_hi'] / full['R_hi'],
                             dRdE_10=float(D[name][np.searchsorted(E, 10.0)]), dRdE_248=float(D[name][np.searchsorted(E, 248.0)]),
                             frac_dRdE_10=float(D[name][np.searchsorted(E, 10.0)] / D['full'][np.searchsorted(E, 10.0)]),
                             frac_dRdE_248=float(D[name][np.searchsorted(E, 248.0)] / D['full'][np.searchsorted(E, 248.0)]), E_peak_keV=s['E_peak_keV'])
        dip['interference_R_lo'] = (full['R_lo'] - dip['charge_c1c5']['R_lo'] - dip['spin_c4c6']['R_lo']) / full['R_lo']
        dip['interference_R_hi'] = (full['R_hi'] - dip['charge_c1c5']['R_hi'] - dip['spin_c4c6']['R_hi']) / full['R_hi']
        dip['c4c6_interference_R_lo'] = (dip['spin_c4c6']['R_lo'] - dip['c4_only']['R_lo'] - dip['c6_only']['R_lo']) / dip['spin_c4c6']['R_lo']
        dip['c4c6_interference_R_hi'] = (dip['spin_c4c6']['R_hi'] - dip['c4_only']['R_hi'] - dip['c6_only']['R_hi']) / dip['spin_c4c6']['R_hi']
        # if the c5 (charge) piece is removed: N_lo of c1 + c4 + c6, and of spin alone (already)
        # massive mediator (dark photon of mass m coupling through the DM magnetic dipole and the SM charge/magnetic current via kinetic mixing):
        # all four coefficients acquire q^2/(q^2 + m^2): rate x [q^2/(q^2+m^2)]^2
        dm_rows = []
        for m in [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
            fac = np.ones_like(E) if m == 0 else (q_gev(E) ** 2 / (q_gev(E) ** 2 + m ** 2)) ** 2
            s_full, s_ch, s_sp = summarise(D['full'] * fac), summarise(D['charge_c1c5'] * fac), summarise(D['spin_c4c6'] * fac)
            dm_rows.append(dict(m_med_GeV=m, N_lo_full=s_full['N_lo'], N_lo_charge=s_ch['N_lo'], N_lo_spin=s_sp['N_lo'],
                                charge_frac_R_hi=s_ch['R_hi'] / s_full['R_hi'], charge_frac_R_lo=s_ch['R_lo'] / s_full['R_lo'], E_peak_keV=s_full['E_peak_keV']))
        DM = pd.DataFrame(dm_rows); DM.to_csv(os.path.join(OUT, 'P074_dipole_massive_mediator.csv'), index=False, float_format='%.4g')
        dip['massive_mediator'] = DM.to_dict(orient='records')
        mf = MMED_FINE[:-1]
        Nf = np.array([summarise(D['full'] * (q_gev(E) ** 2 / (q_gev(E) ** 2 + m ** 2)) ** 2)['N_lo'] for m in mf])
        for thr in (5.0, 3.0):
            j = np.where(Nf <= thr)[0]
            dip[f'massive_mediator_m_min_Nlo{int(thr)}_GeV'] = float(10 ** np.interp(math.log10(thr), np.log10(Nf[::-1]), np.log10(mf[::-1]))) if len(j) else np.inf
        dip['massive_mediator_N_lo_heavy_limit'] = float(Nf[-1])
        log('\nPhoton dipole decomposition (1 TeV): ' + json.dumps({k: v for k, v in dip.items() if k != 'massive_mediator'}, default=float))
        log('Dipole through a massive mediator:\n' + DM.to_string(index=False, float_format=lambda x: f'{x:.3g}'))
    # ---- E7. UV summary table ------------------------------------------------------------------
    def Nlo_at(key, n, m):
        return summarise(C[key] * prop_rate_factor(E, m, n))['N_lo']
    def mmin(key, n):
        r = M[(M.op == key) & (M.n == n)].iloc[0]; return r.m_min_Nlo5_GeV
    uv = []
    def add(model, route, key, n, note=''):
        uv.append(dict(uv_structure=model, nr_route=route, op=key, n=n, N_lo_contact=Ncontact[key], N_lo_m0p1=Nlo_at(key, n, 0.1), N_lo_m0p3=Nlo_at(key, n, 0.3),
                       N_lo_m1=Nlo_at(key, n, 1.0), N_lo_longrange=Nlo_at(key, n, 0.0), m_min_Nlo5_GeV=mmin(key, n),
                       verdict=('excluded at every mediator mass' if not np.isfinite(mmin(key, n)) or (Ncontact[key] > 5) else
                                ('allowed for m_med >= %.2g GeV' % mmin(key, n) if mmin(key, n) > 0 else 'allowed at every mediator mass')), note=note))
    add('scalar mediator, S-S (Higgs portal, light scalar)', 'O1 isoscalar', 'O1s', 1, 'light-mediator SI DM')
    add('vector mediator, V-V (Z\', B-L)', 'O1 isoscalar', 'O1s', 1)
    add('kinetically mixed dark photon (couples to charge)', 'O1 proton-only', 'O1p', 1, 'P011/P051 elastic analogue')
    add('two-mediator / dipole-dipole SI (e.g. scalar x scalar box)', 'O1 x q^-4', 'O1s', 2)
    add('pseudoscalar mediator, P-P (P031 -> O6)', 'O6 isoscalar', 'O6s', 1, 'P031: m_pi/eta poles already in the q^-2 class')
    add('S-P (scalar DM current x pseudoscalar quark current)', 'O10 isoscalar', 'O10s', 1)
    add('axial-vector mediator, A-A (dark Z\', light)', 'O4 isoscalar', 'O4s', 1, 'excluded already as a contact operator (P003: 28)')
    add('DM tensor x quark axial current via a light mediator', 'O9 isoscalar', 'O9s', 1, 'P031 route to O9 (Lambda = 62 GeV contact)')
    add('DM tensor(g5) x quark axial current via a light mediator', 'O14 isoscalar', 'O14s', 1)
    add('vector mediator with DM velocity coupling (V-A)', 'O7 isoscalar', 'O7s', 1)
    add('vector mediator, anapole-like DM current (A-V, charge part)', 'O8 isoscalar', 'O8s', 1, 'P044: photon anapole already 26.9')
    add('EDM-type DM x charge via a light mediator', 'O11 isoscalar', 'O11s', 1, 'P044: photon EDM 18 100')
    add('q^2-weighted spin (O15, O13, O12, O5, O3) via a light mediator', 'O15 / O13 / O12 / O5 / O3', 'O15s', 1)
    UV = pd.DataFrame(uv); UV.to_csv(os.path.join(OUT, 'P074_uv_summary.csv'), index=False, float_format='%.4g')
    log('\nUV summary:\n' + UV[['uv_structure', 'op', 'n', 'N_lo_contact', 'N_lo_m0p3', 'N_lo_longrange', 'm_min_Nlo5_GeV', 'verdict']].to_string(index=False, float_format=lambda x: f'{x:.3g}'))
    # ---- E8. figures ------------------------------------------------------------------------------
    oplist = [label(i) for i in OPS] + ['O1p']
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    for ax, n in zip(axs, NPOW):
        W = T[(T.n == n)].pivot(index='op', columns='m_med_GeV', values='N_lo').reindex(oplist)
        cols = [0.0] + MMED
        Z = np.log10(np.array([[W.loc[o, c] for c in cols] for o in oplist]))
        im = ax.imshow(Z, aspect='auto', cmap='RdYlGn_r', vmin=-1.5, vmax=4.5, origin='upper')
        ax.set_xticks(range(len(cols))); ax.set_xticklabels([('q$^{-%d}$' % (2 * n)) if c == 0 else ('contact' if not np.isfinite(c) else f'{c:g}') for c in cols], rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(oplist))); ax.set_yticklabels(oplist, fontsize=8)
        for a in range(len(oplist)):
            for b in range(len(cols)):
                v = 10 ** Z[a, b]
                ax.text(b, a, f'{v:.2g}' if v < 100 else f'{v:.0f}' if v < 1e4 else f'{v:.0e}', ha='center', va='center', fontsize=6, color='k' if v <= 5 else ('k' if v < 300 else 'w'))
                if v <= 5:
                    ax.add_patch(plt.Rectangle((b - 0.5, a - 0.5), 1, 1, fill=False, ec='k', lw=1.2))
        ax.set_xlabel('mediator mass m$_{med}$ [GeV]'); ax.set_title(f'n = {n}: c(q) = c$^0$ [m$^2$/(m$^2$+q$^2$)]$^{n}$')
    axs[0].set_ylabel('NR operator (isoscalar; O1p = proton-only)')
    cb = fig.colorbar(im, ax=axs, fraction=0.03, pad=0.02); cb.set_label('log$_{10}$ N$_{lo}$ (5.4-55 keV events per 200-270 keV event)')
    fig.suptitle('P074: low-energy companions of a 248 keV-class event vs mediator mass (1 TeV, Sun-frame SHM); boxed: N$_{lo}$ $\\leq$ 5', fontsize=10)
    fig.savefig(os.path.join(FIG, 'P074_heatmap_Nlo.png'), dpi=150, bbox_inches='tight'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for key, n, col in [('O1s', 1, 'C3'), ('O1s', 2, 'C1'), ('O4s', 1, 'C7'), ('O11s', 1, 'C6'), ('O10s', 1, 'C0'), ('O9s', 1, 'C2'), ('O6s', 1, 'C4'), ('O6s', 2, 'C5'), ('O15s', 1, 'C8'), ('O14s', 1, 'C9')]:
        d = anal.get(f'{key}_n{n}_Nlo_fine') if key == 'O1s' else None
        if d is None:
            mf = MMED_FINE[:-1]; Nf = [summarise(C[key] * prop_rate_factor(E, m, n))['N_lo'] for m in mf]
        else:
            mf, Nf = np.array(d['m_GeV']), np.array(d['N_lo'])
        ax.plot(mf, Nf, color=col, ls='-' if n == 1 else '--', lw=1.5, label=f'{key} n={n}')
    ax.axhspan(0.01, 5, color='C2', alpha=0.08); ax.axhline(5, color='k', ls=':', lw=0.8); ax.axhline(3, color='k', ls=':', lw=0.6)
    ax.axvline(float(q_gev(248.0)), color='k', ls='--', lw=0.7); ax.text(float(q_gev(248.0)) * 1.05, 2e5, 'q(248 keV)', fontsize=7)
    ax.axvline(float(q_gev(30.0)), color='k', ls='--', lw=0.7); ax.text(float(q_gev(30.0)) * 1.05, 2e5, 'q(30 keV)', fontsize=7)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('mediator mass m$_{med}$ [GeV]'); ax.set_ylabel('N$_{lo}$ per 200-270 keV event')
    ax.set_title('P074: N$_{lo}$(m$_{med}$) at 1 TeV; shaded = lone-event compatible', fontsize=10); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P074_Nlo_vs_mmed.png'), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for key, n, m, col, ls in [('O6s', 1, np.inf, 'C4', '-'), ('O6s', 1, 0.3, 'C4', '--'), ('O6s', 1, 0.1, 'C4', ':'), ('O10s', 1, np.inf, 'C0', '-'), ('O10s', 1, 0.3, 'C0', '--'),
                               ('O9s', 1, np.inf, 'C2', '-'), ('O9s', 1, 1.0, 'C2', '--'), ('O1s', 1, np.inf, 'C3', '-'), ('O1s', 1, 0.1, 'C3', '--'), ('O1s', 1, 0.0, 'C3', ':')]:
        sp = C[key] * prop_rate_factor(E, m, n); s = summarise(sp)
        y = sp * efficiency(E) / s['R_roi']
        lab = f'{key} ' + ('contact' if not np.isfinite(m) else ('q$^{-2}$ (m$\\to$0)' if m == 0 else f'm={m:g} GeV')) + f' (N$_{{lo}}$={s["N_lo"]:.3g})'
        ax.plot(E, y, color=col, ls=ls, lw=1.4, label=lab)
    ax.axvspan(5.4, 55, color='C3', alpha=0.06); ax.axvspan(200, 270, color='C2', alpha=0.10); ax.axvline(248, color='k', ls=':', lw=0.8)
    ax.set_yscale('log'); ax.set_xlim(0, 330); ax.set_ylim(1e-5, 3); ax.set_xlabel('E$_R$ [keV]'); ax.set_ylabel('efficiency-weighted dR/dE per ROI event [keV$^{-1}$]')
    ax.set_title('P074: survivors and the light-mediator SI spectrum (1 TeV, Sun-frame SHM), each normalised to one ROI event', fontsize=9); ax.legend(fontsize=6.5, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P074_spectra_survivors.png'), dpi=150); plt.close(fig)

    if dip:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for name, col, ls in [('full', 'k', '-'), ('charge_c1c5', 'C3', '-'), ('c5_only', 'C3', '--'), ('c1_only', 'C3', ':'), ('spin_c4c6', 'C0', '-'), ('c6_only', 'C0', '--'), ('c4_only', 'C0', ':')]:
            sp = D[name]; ax.plot(E, sp * efficiency(E), color=col, ls=ls, lw=1.4, label=f'{name} (N$_{{lo}}$={dip[name]["N_lo"]:.3g})')
        ax.axvspan(5.4, 55, color='C3', alpha=0.06); ax.axvspan(200, 270, color='C2', alpha=0.10)
        ax.set_yscale('log'); ax.set_xlim(0, 330); ax.set_xlabel('E$_R$ [keV]'); ax.set_ylabel('efficiency-weighted dR/dE at $\\mu_\\chi$ = 1 $\\mu_N$ [/t/yr/keV]')
        ax.set_title('P074: photon-mediated magnetic dipole at 1 TeV, decomposed (c1, c5 charge; c4, c6 spin)', fontsize=9); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P074_dipole_decomposition.png'), dpi=150); plt.close(fig)

    # validation summary
    val = {}
    if os.path.exists(VPATH):
        V = pd.read_csv(VPATH)
        Vp = V[V.E_keV > 0]; Vn = V[V.E_keV < 0]
        val = dict(pointwise_max_abs_dev=float((Vp.ratio - 1).abs().max()), pointwise_rms_dev=float(np.sqrt(((Vp.ratio - 1) ** 2).mean())),
                   Nlo_max_abs_dev=float((Vn.ratio - 1).abs().max()), n_points=int(len(Vp)), cases=Vn[['op', 'm_med', 'n', 'wimpydd_full', 'factorised', 'ratio']].to_dict(orient='records'))
        log('\nFactorisation validation: ' + json.dumps({k: v for k, v in val.items() if k != 'cases'}))
    RES.update(contact_N_lo=Ncontact, analytic_O1=anal, dipole=dip, validation=val, mmed_min=M.to_dict(orient='records'),
               collapse=COL.to_dict(orient='records'), survivors=surv.to_dict(orient='records'), uv_summary=UV.to_dict(orient='records'), runtime_s=time.time() - T0)
    with open(os.path.join(OUT, 'P074_results.json'), 'w') as f:
        json.dump(RES, f, indent=1, default=float)
    log(f'analysis done ({time.time()-T0:.0f}s)')
LOG.close()
