"""
P051_light_mediator.py -- Inelastic dark matter with a LIGHT mediator and the LZ 248 keV event.

Up-scattering chi1 N -> chi2 N through a mediator of mass m_med comparable to or below the momentum transfer
q = sqrt(2 m_N E_R) ~ 0.25 GeV at 248 keV.  The O1 Wilson coefficient becomes q-dependent,
    c(q) = c0 * m_med^2 / (m_med^2 + q^2)             (both isospin components),
so the rate carries the propagator factor P(E) = m_med^4/(m_med^2 + q^2)^2 and the recoil spectrum is tilted toward
low q inside the kinematically allowed window [E-, E+].

Parts
  1  WimPyDD O1 inelastic kernels (isoscalar with LZ unit coupling c^0 = 2/m_v^2; proton-only c^0 = c^1 = 1/m_v^2) for
     m_chi = 1 TeV, delta = 250..390 keV, on the full kinematic window (to 720 keV), for three halos on one v_min grid:
     Sun-frame (WimPyDD single-day default, no Earth motion), 16 June (day 167) and a 12-day annual mean.
     Light mediators are applied as the per-energy propagator factor P(E) at the mean xenon mass; this factorisation
     is VALIDATED against full WimPyDD q-dependent-coefficient kernels (Part 1b).  Exothermic (delta -> -delta)
     proton-only kernels for the chi2-survival test (Part 1c).
  2  Spectral observables per (delta, m_med, isospin, halo): E-, E+, acceptance of the S1c < 600 phd edge (P038 erf,
     E50 = 271.6 keV), observed-energy spectrum (sigma_E = 11 sqrt(E/248) keV), percentile of 248 keV, fraction within
     +-23 keV, events in 600-1000 phd (P038 1000-phd edge, E50 = 423.2 keV) per ROI event, N(100-200)/N(200-270),
     N(125-200)/N(200-270), Poisson P(0) for the empty NR band, and the single-event profile statistic
     q0 = 2[ln(f~/b) - 1 + b/f~] (P021/P038 companion-free form) -> Z(delta), peak and 68 % range.
  3  Coupling for one accepted LZ event: kappa = 1/S_acc; sigma_p(q->0) for the proton-only (dark-photon) case;
     eps^2 alpha_D = sigma_p m_A'^4 /(16 pi alpha mu_p^2) (P011); required eps(m_A') on a continuous m_A' grid;
     chi2-decay floor recomputed with the light-mediator exothermic rate (P011/P026 method, tau < t_U/ln N_exo);
     recalled accelerator limits (flagged); Planck (P025) as a vertical line.
  4  Figures and JSON/CSV tables under output/work/P051/.

Run from the simulation root:  .venv/bin/python output/code/P051_light_mediator.py   [--stage kernels|analysis|all]
"""
from __future__ import annotations
import sys, os, json, math, time
sys.path.insert(0, 'output/code')
import numpy as np
import pandas as pd
from scipy import stats, special
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import lzcommon as lz

T0 = time.time()
OUT = 'output/work/P051'
FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)
STAGE = sys.argv[sys.argv.index('--stage') + 1] if '--stage' in sys.argv else 'all'
LOG = open(f'{OUT}/run_log.txt', 'a')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n'); LOG.flush()
say(f'===== P051 run {time.strftime("%Y-%m-%d %H:%M:%S")} stage={STAGE} =====')

# ------------------------------------------------------------------------------------------------------
# 0. constants and settings
# ------------------------------------------------------------------------------------------------------
M_CHI = 1000.0
MV, MN = lz.M_V_GEV, lz.M_NUCLEON_GEV
A_MEAN = lz.A_XE_MEAN                                  # 131.29
M_XE = lz.m_nucleus_gev(A_MEAN)                        # GeV
EXPO = lz.LZ['exposure_tyr']                           # 2.84 t yr
DELTAS = np.array([250., 260., 270., 280., 290., 300., 310., 320., 330., 340., 350., 360., 370., 375., 380., 385., 390.])
MMED = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, np.inf]        # GeV
MMED_LABEL = {m: (f'{m:g} GeV' if np.isfinite(m) else 'contact') for m in MMED}
VGRID = np.linspace(0.0, 830.0, 831); ONES = np.ones_like(VGRID)
E_T = np.arange(40.0, 720.0 + 1e-9, 4.0); DE_T = 4.0    # true-energy grid (window edges: 47 keV at delta=250, A=136)
E_O = np.arange(0.5, 720.0, 1.0)                        # observed-energy grid
DOY_JUNE = 167
DAYS12 = 15.0 + 365.25 / 12 * np.arange(12)
B_H, W_H = 5.7e-4, 70.0                                 # P016/P021 NR-band background, flat in 200-270 keV
BD = B_H / W_H
E_EV = 248.0
# P038 efficiency edges (erf in true NR energy; 'edges' block of P038_results.json)
P38 = json.load(open('output/work/P038/P038_results.json'))['edges']
E50_600, SIG_600 = P38['600']['E50_keV'], P38['600']['sigma_erf_keV']        # 271.6, 10.9 keV
E50_1000, SIG_1000 = P38['1000']['E50_keV'], P38['1000']['sigma_erf_keV']    # 423.2, 14.5 keV
PLATEAU = 0.955
# dark-photon constants (P011; recalled, certain)
ALPHA = 1.0 / 137.035999
E_EM = math.sqrt(4 * math.pi * ALPHA)
SW2 = 0.2312; SW, CW = math.sqrt(SW2), math.sqrt(1 - SW2); MZ = 91.1876
G_WEAK = E_EM / SW
HBAR_GEV_S = 6.582119569e-25
T_UNIVERSE_S = 13.8e9 * 3.15576e7
GEV_TO_CM2 = lz.GEV_TO_CM2
MU_P = M_CHI * MN / (M_CHI + MN)
C_UNIT = 1.0 / MV ** 2
SIGMA_P_UNIT = C_UNIT ** 2 * MU_P ** 2 / math.pi * GEV_TO_CM2     # proton-only unit cross-section, cm^2
ALPHA_D_REF = 0.1
ALPHA_D_THERMAL = 0.035                                          # P025 (P011: 0.0245); eps scales as alpha_D^-1/2
EPS_BABAR = 1e-3                                                 # recalled, likely (BaBar visible 2014 / invisible 2017)
RES = dict(settings=dict(m_chi=M_CHI, deltas=DELTAS.tolist(), m_med=[str(m) for m in MMED], vgrid=[0, 830, 831],
                         E_T=[40, 720, 4], E_O=[0.5, 720, 1], eff_600=dict(E50=E50_600, sigma=SIG_600, plateau=PLATEAU),
                         eff_1000=dict(E50=E50_1000, sigma=SIG_1000), resolution='11 sqrt(E/248) keV', b_H=B_H, W_H=W_H,
                         exposure_tyr=EXPO, sigma_p_unit_cm2=SIGMA_P_UNIT, alpha_D_ref=ALPHA_D_REF))

def eff(E, E50, sig):
    E = np.asarray(E, float)
    lo = 0.5 * (1 + special.erf((E - 5.4) / (math.sqrt(2) * 2.5)))
    return PLATEAU * lo * 0.5 * special.erfc((E - E50) / (math.sqrt(2) * sig))
EFF600, EFF1000 = eff(E_T, E50_600, SIG_600), eff(E_T, E50_1000, SIG_1000)

def q_gev(E_kev, A=A_MEAN):
    return np.sqrt(2 * lz.m_nucleus_gev(A) * np.asarray(E_kev, float) * 1e-6)

def prop(E_kev, m_med, A=A_MEAN):
    """rate propagator factor m^4/(m^2+q^2)^2 at the given recoil energy (q from the mean xenon mass)."""
    if not np.isfinite(m_med):
        return np.ones_like(np.asarray(E_kev, float))
    q2 = q_gev(E_kev, A) ** 2
    return (m_med ** 2 / (m_med ** 2 + q2)) ** 2

# ------------------------------------------------------------------------------------------------------
# 1. WimPyDD kernels (cached)
# ------------------------------------------------------------------------------------------------------
KPATH = f'{OUT}/kernels_1000GeV.npz'
HPATH = f'{OUT}/halos.npz'
VALPATH = f'{OUT}/factorisation_validation.json'
EXOPATH = f'{OUT}/kernels_exothermic.npz'
HAMS = None
def hams():
    global HAMS
    if HAMS is None:
        HAMS = {'iso': lz.wd_hamiltonian('P051_iso', {1: lz.wd_c_from_anand(C_UNIT, 0.0)}),   # c^0 = 2/m_v^2 (LZ unit)
                'p': lz.wd_hamiltonian('P051_p', {1: (C_UNIT, C_UNIT)})}                    # c_p = 1/m_v^2, c_n = 0
    return HAMS

def kernel(ham, d, E):
    WD = lz.wd()
    return np.asarray(WD.diff_rate(WD.Xe, ham, M_CHI, float(E), VGRID, ONES, j_chi=0.5, delta=float(d),
                                   sum_over_streams=False)) * 1000.0 * 365.25          # /kg/day -> /t/yr per v-bin

def window_idx(d):
    los, his = zip(*[lz.E_R_range_keV(M_CHI, VGRID[-1], A=A, delta_kev=float(d)) for A in (124.0, 136.0)])
    lo, hi = min(los), max(his)
    return np.where((E_T >= lo - DE_T) & (E_T <= hi + DE_T))[0]

if STAGE in ('kernels', 'all'):
    if not os.path.exists(HPATH):
        halos = {'sun': lz.wd_halo(vmin=VGRID)[1], 'june': lz.wd_halo(day_of_year=DOY_JUNE, vmin=VGRID)[1],
                 'annual': np.mean([lz.wd_halo(day_of_year=dd, vmin=VGRID)[1] for dd in DAYS12], axis=0)}
        np.savez(HPATH, vgrid=VGRID, **halos)
        say('halos computed: v_max(sun/june) where eta>0:', {k: float(VGRID[np.max(np.where(v > 0))]) for k, v in halos.items()})
    K = dict(np.load(KPATH)) if os.path.exists(KPATH) else {}
    ncalls = 0
    for d in DELTAS:
        for tag in ('iso', 'p'):
            key = f'{tag}_{int(round(d))}'
            if key in K:
                continue
            arr = np.zeros((len(E_T), len(VGRID)))
            for j in window_idx(d):
                arr[j] = kernel(hams()[tag], d, E_T[j]); ncalls += 1
            K[key] = arr
            np.savez(KPATH, **K)
            say(f'  kernel {key}: {len(window_idx(d))} energies, calls so far {ncalls}, t={time.time() - T0:.0f}s')
    # 1b. validation of the propagator factorisation against full WimPyDD q-dependent coefficients
    if not os.path.exists(VALPATH):
        WD = lz.wd()
        def make_light(m_med, c0, c1):
            def f(q):
                g = m_med ** 2 / (m_med ** 2 + q ** 2)
                return [c0 * g, c1 * g]
            return f
        rows = []
        hal = dict(np.load(HPATH))
        for m_med in (0.05, 0.1, 0.3, 1.0):
            hl = {'iso': WD.eft_hamiltonian(f'P051_l_iso_{m_med}', {(1, 'q2'): make_light(m_med, 2 * C_UNIT, 0.0)}),
                  'p': WD.eft_hamiltonian(f'P051_l_p_{m_med}', {(1, 'q2'): make_light(m_med, C_UNIT, C_UNIT)})}
            for d, Es in ((300.0, (100., 150., 200., 248., 300., 400., 500.)), (380.0, (248., 300., 400.)), (-300.0, (50., 150., 248.))):
                for tag in ('iso', 'p'):
                    if d < 0 and tag == 'iso':
                        continue
                    for E in Es:
                        kl = kernel(hl[tag], d, E); kc = kernel(hams()[tag], d, E)
                        for hname in ('june', 'sun'):
                            rl, rc = float(kl @ hal[hname]), float(kc @ hal[hname])
                            if rc > 0:
                                rows.append(dict(m_med=m_med, delta=d, iso=tag, E_keV=E, halo=hname, ratio_wimpydd=rl / rc,
                                                 factor_meanA=float(prop(E, m_med)), rel_dev=rl / rc / float(prop(E, m_med)) - 1))
        V = pd.DataFrame(rows); V.to_csv(f'{OUT}/factorisation_validation.csv', index=False)
        json.dump(dict(max_abs_rel_dev=float(V.rel_dev.abs().max()), rms_rel_dev=float(np.sqrt((V.rel_dev ** 2).mean())),
                       by_m_med={str(m): float(V[V.m_med == m].rel_dev.abs().max()) for m in V.m_med.unique()},
                       n_points=len(V)), open(VALPATH, 'w'), indent=1)
        say('factorisation validation: max |rel dev| = %.4f, rms = %.4f (n=%d), t=%.0fs' % (V.rel_dev.abs().max(), np.sqrt((V.rel_dev ** 2).mean()), len(V), time.time() - T0))
    # 1c. exothermic proton-only kernels (chi2 N -> chi1 N), delta -> -delta
    if not os.path.exists(EXOPATH):
        E_X = np.arange(2.0, 350.0 + 1e-9, 4.0)
        KX = {'E': E_X}
        for d in (300.0, 350.0, 380.0):
            arr = np.zeros((len(E_X), len(VGRID)))
            for j, E in enumerate(E_X):
                arr[j] = kernel(hams()['p'], -d, E)
            KX[f'p_{int(d)}'] = arr
            say(f'  exothermic kernel delta=-{d:.0f}: {len(E_X)} energies, t={time.time() - T0:.0f}s')
        np.savez(EXOPATH, **KX)
    say(f'kernel stage done, t={time.time() - T0:.0f}s')

# 1d. exact WimPyDD light-mediator spectra (full q-dependent coefficient) for the two lightest mediators at delta = 300, 380
EXACTPATH = f'{OUT}/kernels_exact_light.npz'
if STAGE in ('exact', 'all') and not os.path.exists(EXACTPATH):
    WD = lz.wd()
    def make_light(m_med, c0, c1):
        def f(q):
            g = m_med ** 2 / (m_med ** 2 + q ** 2)
            return [c0 * g, c1 * g]
        return f
    KE = {}
    for m_med in (0.05, 0.1):
        hl = {'iso': WD.eft_hamiltonian(f'P051_x_iso_{m_med}', {(1, 'q2'): make_light(m_med, 2 * C_UNIT, 0.0)}),
              'p': WD.eft_hamiltonian(f'P051_x_p_{m_med}', {(1, 'q2'): make_light(m_med, C_UNIT, C_UNIT)})}
        for d in (300.0, 380.0):
            for tag in ('iso', 'p'):
                arr = np.zeros((len(E_T), len(VGRID)))
                for j in window_idx(d):
                    arr[j] = kernel(hl[tag], d, E_T[j])
                KE[f'{tag}_{int(d)}_{m_med}'] = arr
                say(f'  exact light kernel {tag} delta={d:.0f} m_med={m_med}: t={time.time() - T0:.0f}s')
    np.savez(EXACTPATH, **KE)

if STAGE in ('kernels', 'exact'):
    sys.exit(0)

# ------------------------------------------------------------------------------------------------------
# 2. spectra and observables
# ------------------------------------------------------------------------------------------------------
K = dict(np.load(KPATH)); HAL = dict(np.load(HPATH)); KX = dict(np.load(EXOPATH))
VAL = json.load(open(VALPATH)); RES['factorisation_validation'] = VAL
HALOS = ['annual', 'june', 'sun']
HALO_LABEL = {'annual': '12-day annual mean', 'june': '16 June (day 167)', 'sun': 'Sun-frame (WimPyDD single-day default, no Earth motion)'}
SIG_E = 11.0 * np.sqrt(E_T / 248.0)
SMEAR = stats.norm.pdf((E_O[:, None] - E_T[None, :]) / SIG_E[None, :]) / SIG_E[None, :] * DE_T     # (nO, nT)
ROW_EV = stats.norm.pdf((E_EV - E_T) / SIG_E) / SIG_E * DE_T

def q0_single(ftil):
    return 2 * (math.log(ftil / BD) - 1 + BD / ftil) if ftil > BD else 0.0

def spectrum(tag, d, hname, m_med):
    r = K[f'{tag}_{int(round(d))}'] @ HAL[hname]           # /t/yr/keV, contact
    r = np.clip(r, 0, None) * prop(E_T, m_med)
    return r

rows = []
for hname in HALOS:
    for tag in ('iso', 'p'):
        for d in DELTAS:
            Em, Ep = lz.E_R_range_keV(M_CHI, float(VGRID[np.max(np.where(HAL[hname] > 0))]), A=A_MEAN, delta_kev=float(d))
            for m_med in MMED:
                r = spectrum(tag, d, hname, m_med)
                S_full = float(r.sum() * DE_T * EXPO)
                if S_full <= 0:
                    continue
                w600 = r * EFF600 * EXPO; w1000 = r * EFF1000 * EXPO
                S600, S1000 = float(w600.sum() * DE_T), float(w1000.sum() * DE_T)
                dobs = SMEAR @ w600                                  # accepted observed-energy spectrum (per keV)
                dtrue_cum = np.cumsum(w600) * DE_T
                f_ev = float(ROW_EV @ w600)
                ftil = f_ev / S600 if S600 > 0 else 0.0
                q0 = q0_single(ftil) if ftil > 0 else 0.0
                tot_o = dobs.sum()
                def frac_o(a, b):
                    return float(dobs[(E_O >= a) & (E_O < b)].sum() / tot_o)
                N_200_270 = frac_o(200, 270); N_100_200 = frac_o(100, 200); N_125_200 = frac_o(125, 200)
                rows.append(dict(halo=hname, iso=tag, delta_keV=float(d), m_med_GeV=(float(m_med) if np.isfinite(m_med) else np.inf),
                                 E_minus_keV=Em, E_plus_keV=Ep, S_full_unit=S_full, S_acc600_unit=S600, S_acc1000_unit=S1000,
                                 A_600=S600 / S_full, A_1000=S1000 / S_full,
                                 E_peak_true=float(E_T[np.argmax(r)]), E_median_true_full=float(np.interp(0.5 * r.sum(), np.cumsum(r), E_T)),
                                 E_median_true_acc=float(np.interp(0.5 * dtrue_cum[-1], dtrue_cum, E_T)),
                                 pct248_true_acc=float(np.interp(E_EV, E_T, dtrue_cum) / dtrue_cum[-1]),
                                 pct248_true_full=float(np.interp(E_EV, E_T, np.cumsum(r)) / r.sum()),
                                 pct248_obs=float(dobs[E_O < E_EV].sum() / tot_o),
                                 frac_pm23_obs=frac_o(225, 271), frac_200_270_obs=N_200_270,
                                 ratio_100_200_over_200_270=N_100_200 / N_200_270 if N_200_270 > 0 else np.inf,
                                 ratio_125_200_over_200_270=N_125_200 / N_200_270 if N_200_270 > 0 else np.inf,
                                 P0_100_200_given_one_in_200_270=math.exp(-N_100_200 / N_200_270) if N_200_270 > 0 else 0.0,
                                 P0_125_200_given_one_in_200_270=math.exp(-N_125_200 / N_200_270) if N_200_270 > 0 else 0.0,
                                 N_600_1000phd_per_ROI_event=(S1000 - S600) / S600 if S600 > 0 else np.nan,
                                 kappa_hat=1.0 / S600 if S600 > 0 else np.nan, ftil=ftil, q0=q0, Z=math.sqrt(q0)))
DF = pd.DataFrame(rows)
DF.to_csv(f'{OUT}/P051_observables.csv', index=False)
say(f'observables: {len(DF)} rows, t={time.time() - T0:.0f}s')

# exact-vs-factorised comparison of the final observables (isotope-mix systematic of the propagator factorisation)
KE = dict(np.load(EXACTPATH)) if os.path.exists(EXACTPATH) else {}
cmp_rows = []
for key, arr in KE.items():
    tag, d, m_med = key.split('_'); d = float(d); m_med = float(m_med)
    for hname in HALOS:
        r_ex = np.clip(arr @ HAL[hname], 0, None); r_fa = spectrum(tag, d, hname, m_med)
        out = {}
        for lab, r in (('exact', r_ex), ('fact', r_fa)):
            w = r * EFF600 * EXPO; S = float(w.sum() * DE_T); dobs = SMEAR @ w
            ftil = float(ROW_EV @ w) / S
            fr = lambda a, b: float(dobs[(E_O >= a) & (E_O < b)].sum() / dobs.sum())
            out[lab] = dict(S_acc=S, Z=math.sqrt(q0_single(ftil)) if ftil > BD else 0.0, pct248_obs=float(dobs[E_O < E_EV].sum() / dobs.sum()),
                            ratio_100_200=fr(100, 200) / fr(200, 270), A_600=S / float(r.sum() * DE_T * EXPO))
        cmp_rows.append(dict(iso=tag, delta_keV=d, m_med_GeV=m_med, halo=hname, S_fact_over_exact=out['fact']['S_acc'] / out['exact']['S_acc'],
                             Z_exact=out['exact']['Z'], Z_fact=out['fact']['Z'], dZ=out['fact']['Z'] - out['exact']['Z'],
                             pct_exact=out['exact']['pct248_obs'], pct_fact=out['fact']['pct248_obs'],
                             ratio_exact=out['exact']['ratio_100_200'], ratio_fact=out['fact']['ratio_100_200'],
                             A600_exact=out['exact']['A_600'], A600_fact=out['fact']['A_600']))
if cmp_rows:
    CMPX = pd.DataFrame(cmp_rows); CMPX.to_csv(f'{OUT}/P051_exact_vs_factorised.csv', index=False)
    say('\nexact WimPyDD light-mediator spectra vs propagator factorisation:'); say(CMPX.round(4).to_string(index=False))
    RES['exact_vs_factorised'] = dict(max_abs_dZ=float(CMPX.dZ.abs().max()), S_ratio_range=[float(CMPX.S_fact_over_exact.min()), float(CMPX.S_fact_over_exact.max())],
                                      ratio_100_200_rel_range=[float((CMPX.ratio_fact / CMPX.ratio_exact).min()), float((CMPX.ratio_fact / CMPX.ratio_exact).max())],
                                      max_abs_dpct=float((CMPX.pct_fact - CMPX.pct_exact).abs().max()))

def sel(hname, tag, m_med):
    s = DF[(DF.halo == hname) & (DF.iso == tag) & (np.isinf(DF.m_med_GeV) if not np.isfinite(m_med) else np.isclose(DF.m_med_GeV, m_med))]
    return s.set_index('delta_keV').sort_index()

# Z(delta) peaks and 68 % ranges (Delta q0 <= 1 around the peak, linear interpolation on the delta grid)
def peak_and_range(s):
    dd, q = s.index.values, s.q0.values
    fine = np.arange(dd[0], dd[-1] + 1e-9, 1.0); qf = np.interp(fine, dd, q)
    i = int(np.argmax(qf)); ok = np.where(qf >= qf[i] - 1.0)[0]
    # contiguous segment containing the peak
    lo = i
    while lo > 0 and qf[lo - 1] >= qf[i] - 1.0: lo -= 1
    hi = i
    while hi < len(fine) - 1 and qf[hi + 1] >= qf[i] - 1.0: hi += 1
    return dict(delta_peak=float(fine[i]), Z_peak=float(math.sqrt(qf[i])), d68_lo=float(fine[lo]), d68_hi=float(fine[hi]),
                edge_lo=bool(lo == 0), edge_hi=bool(hi == len(fine) - 1))

pk_rows = []
for hname in HALOS:
    for tag in ('iso', 'p'):
        s_c = sel(hname, tag, np.inf)
        for m_med in MMED:
            s = sel(hname, tag, m_med)
            pr = peak_and_range(s)
            row = dict(halo=hname, iso=tag, m_med_GeV=(float(m_med) if np.isfinite(m_med) else np.inf), **pr,
                       Z_300=float(s.q0.get(300.0, np.nan) ** 0.5), Z_330=float(s.q0.get(330.0, np.nan) ** 0.5),
                       Z_350=float(s.q0.get(350.0, np.nan) ** 0.5), Z_380=float(s.q0.get(380.0, np.nan) ** 0.5),
                       dq0_peak_minus_300=float(pr['Z_peak'] ** 2 - s.q0.get(300.0, np.nan)),
                       dq0_peak_minus_330=float(pr['Z_peak'] ** 2 - s.q0.get(330.0, np.nan)),
                       dq0_peak_minus_350=float(pr['Z_peak'] ** 2 - s.q0.get(350.0, np.nan)),
                       LR_peak_over_300=float(math.exp(0.5 * (pr['Z_peak'] ** 2 - s.q0.get(300.0, np.nan)))),
                       shift_vs_contact_keV=float(pr['delta_peak'] - peak_and_range(s_c)['delta_peak']))
            pk_rows.append(row)
PK = pd.DataFrame(pk_rows); PK.to_csv(f'{OUT}/P051_Z_peaks.csv', index=False)
say('\nZ(delta) peaks, annual halo:'); say(PK[PK.halo == 'annual'].round(3).to_string(index=False))
say('\nZ(delta) peaks, June halo:'); say(PK[PK.halo == 'june'][['iso', 'm_med_GeV', 'delta_peak', 'Z_peak', 'd68_lo', 'd68_hi', 'Z_300', 'Z_350', 'Z_380']].round(3).to_string(index=False))
say('\nZ(delta) peaks, Sun-frame halo:'); say(PK[PK.halo == 'sun'][['iso', 'm_med_GeV', 'delta_peak', 'Z_peak', 'd68_lo', 'd68_hi', 'Z_300', 'Z_350', 'Z_380']].round(3).to_string(index=False))

# headline spectral table: annual halo, isoscalar & proton-only, delta = 300, 350, 380
cols = ['iso', 'm_med_GeV', 'E_minus_keV', 'E_plus_keV', 'A_600', 'E_peak_true', 'E_median_true_acc', 'pct248_true_acc', 'pct248_obs',
        'frac_pm23_obs', 'ratio_100_200_over_200_270', 'P0_100_200_given_one_in_200_270', 'ratio_125_200_over_200_270',
        'P0_125_200_given_one_in_200_270', 'N_600_1000phd_per_ROI_event', 'kappa_hat', 'Z']
for d in (300.0, 330.0, 350.0, 380.0):
    say(f'\n--- delta = {d:.0f} keV, annual halo ---')
    say(DF[(DF.halo == 'annual') & (DF.delta_keV == d)][cols].to_string(index=False, float_format=lambda x: '%.4g' % x))
say('\n--- delta = 300 keV, June halo, isoscalar ---')
say(DF[(DF.halo == 'june') & (DF.delta_keV == 300.0) & (DF.iso == 'iso')][cols].to_string(index=False, float_format=lambda x: '%.4g' % x))

# ------------------------------------------------------------------------------------------------------
# 3. couplings, eps(m_A'), chi2-decay floor with the light-mediator exothermic rate
# ------------------------------------------------------------------------------------------------------
E_X = KX['E']; EFFX = eff(E_X, E50_600, SIG_600)
def gamma_chi2(eps, aD, delta_kev, n_nu=3):
    gD = math.sqrt(4 * math.pi * aD)
    Geff = gD * eps * (SW / CW) * G_WEAK / (4 * CW * MZ ** 2)
    return n_nu * Geff ** 2 * (delta_kev * 1e-6) ** 5 / (120 * math.pi ** 3)
def tau_chi2_s(eps, aD, delta_kev):
    return HBAR_GEV_S / gamma_chi2(eps, aD, delta_kev)

MA_GRID = np.logspace(-1.5, 1.5, 181)
coup_rows, plane = [], {}
for hname in ('annual', 'june'):
    for d in (300.0, 330.0, 350.0, 380.0):
        rc = np.clip(K[f'p_{int(d)}'] @ HAL[hname], 0, None)              # contact proton-only endothermic spectrum, unit c_p
        if d in (300.0, 350.0, 380.0):
            rx = np.clip(KX[f'p_{int(d)}'] @ HAL[hname], 0, None)         # contact exothermic spectrum, unit c_p, f2 = 1
        else:
            rx = None
        eps_req, eps_floor, N_exo, sig_p, S_light = [], [], [], [], []
        for mA in MA_GRID:
            S600 = float((rc * prop(E_T, mA) * EFF600).sum() * DE_T * EXPO)
            kappa = 1.0 / S600
            sp = kappa * SIGMA_P_UNIT                                     # sigma_p(q -> 0) for one accepted event
            X = sp / GEV_TO_CM2 / (16 * math.pi * ALPHA * MU_P ** 2)      # eps^2 alpha_D / m_A'^4
            e_req = math.sqrt(X * mA ** 4 / ALPHA_D_REF)
            if rx is not None:
                Nx = kappa * float((rx * prop(E_X, mA) * EFFX).sum() * 4.0 * EXPO)   # exothermic ROI events if f2 = 1
                if Nx > 1:
                    tau_max = T_UNIVERSE_S / math.log(Nx)
                    e_min = math.sqrt(tau_chi2_s(1.0, ALPHA_D_REF, d) / tau_max)
                else:
                    e_min = 0.0
            else:
                Nx, e_min = np.nan, np.nan
            eps_req.append(e_req); eps_floor.append(e_min); N_exo.append(Nx); sig_p.append(sp)
            S_light.append(S600 / float((rc * EFF600).sum() * DE_T * EXPO))
        eps_req, eps_floor, N_exo, sig_p, S_light = map(np.array, (eps_req, eps_floor, N_exo, sig_p, S_light))
        plane[(hname, d)] = dict(mA=MA_GRID, eps_req=eps_req, eps_floor=eps_floor, N_exo=N_exo, sigma_p=sig_p, S=S_light)
        ratio = eps_req / np.where(eps_floor > 0, eps_floor, np.nan)
        # m_A' floor where eps_req crosses eps_floor (ratio = 1), searched from the top
        mA_floor = np.nan
        if np.all(np.isfinite(ratio)):
            idx = np.where((ratio[:-1] < 1) & (ratio[1:] >= 1))[0]
            if len(idx):
                i = idx[-1]; mA_floor = float(np.exp(np.interp(0.0, np.log(ratio[i:i + 2]), np.log(MA_GRID[i:i + 2]))))
        # m_A' where eps_req hits the recalled BaBar ceiling
        idx = np.where((eps_req[:-1] < EPS_BABAR) & (eps_req[1:] >= EPS_BABAR))[0]
        mA_babar = float(np.exp(np.interp(math.log(EPS_BABAR), np.log(eps_req[idx[0]:idx[0] + 2]), np.log(MA_GRID[idx[0]:idx[0] + 2])))) if len(idx) else np.nan
        for mA in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 3.0, 10.0):
            j = int(np.argmin(np.abs(np.log(MA_GRID) - math.log(mA))))
            coup_rows.append(dict(halo=hname, delta_keV=d, m_A_GeV=float(MA_GRID[j]), S_prop=float(S_light[j]), sigma_p_q0_N1_cm2=float(sig_p[j]),
                                  eps_req_alphaD0p1=float(eps_req[j]), eps_req_alphaD_thermal=float(eps_req[j] * math.sqrt(ALPHA_D_REF / ALPHA_D_THERMAL)),
                                  N_exo_if_f2_1=float(N_exo[j]), eps_floor_chi2_decay_alphaD0p1=float(eps_floor[j]),
                                  eps_req_over_floor=float(ratio[j]) if np.isfinite(ratio[j]) else np.nan,
                                  tau_at_eps_req_s=tau_chi2_s(float(eps_req[j]), ALPHA_D_REF, d), mA_floor_GeV=mA_floor, mA_babar_ceiling_GeV=mA_babar))
CP = pd.DataFrame(coup_rows); CP.to_csv(f'{OUT}/P051_couplings.csv', index=False)
say('\ncouplings (annual halo):'); say(CP[CP.halo == 'annual'].to_string(index=False, float_format=lambda x: '%.3g' % x))
say('\ncouplings (June halo) floors:'); say(CP[CP.halo == 'june'].groupby('delta_keV')[['mA_floor_GeV', 'mA_babar_ceiling_GeV']].first().to_string(float_format=lambda x: '%.3g' % x))
RES['mA_floor_GeV'] = {f'{h}_{int(d)}': float(CP[(CP.halo == h) & (CP.delta_keV == d)].mA_floor_GeV.iloc[0]) for h in ('annual', 'june') for d in (300.0, 350.0, 380.0)}
RES['mA_babar_ceiling_GeV'] = {f'{h}_{int(d)}': float(CP[(CP.halo == h) & (CP.delta_keV == d)].mA_babar_ceiling_GeV.iloc[0]) for h in ('annual', 'june') for d in (300.0, 330.0, 350.0, 380.0)}
# eps saturation (m_A' << q): eps^2 alpha_D -> sigma_p,heavy q^4/(16 pi alpha mu^2) x (exact spectral average)
RES['eps_saturation_alphaD0p1_annual'] = {int(d): float(plane[('annual', d)]['eps_req'][0]) for d in (300.0, 330.0, 350.0, 380.0)}
RES['P011_comparison'] = dict(note='P011 heavy-mediator eps(alpha_D=0.1, m_A=1 GeV, annual) = 8.7e-7 (300), 5.1e-6 (350); floors 5.4e-6 (300), 4.5e-6 (350), 4.5e-6 (380); m_A floors 2.6/0.94/0.09 GeV; P026 2.45/0.92/0.25 GeV',
                              this_work_eps_1GeV_300=float(CP[(CP.halo == 'annual') & (CP.delta_keV == 300) & np.isclose(CP.m_A_GeV, 1.0, rtol=0.03)].eps_req_alphaD0p1.iloc[0]),
                              this_work_floor_contact_300=float(plane[('annual', 300.0)]['eps_floor'][-1]),
                              this_work_floor_contact_350=float(plane[('annual', 350.0)]['eps_floor'][-1]),
                              this_work_floor_contact_380=float(plane[('annual', 380.0)]['eps_floor'][-1]))

# ------------------------------------------------------------------------------------------------------
# 4. summary numbers
# ------------------------------------------------------------------------------------------------------
def pick(hname, tag, m_med, d, col):
    return float(sel(hname, tag, m_med).loc[d, col])
RES['Z_peaks'] = PK.to_dict('records')
RES['spectral_300_annual'] = {MMED_LABEL[m]: {c: pick('annual', 'iso', m, 300.0, c) for c in
                              ('A_600', 'E_peak_true', 'E_median_true_acc', 'pct248_true_acc', 'pct248_obs', 'frac_pm23_obs', 'ratio_100_200_over_200_270',
                               'P0_100_200_given_one_in_200_270', 'ratio_125_200_over_200_270', 'P0_125_200_given_one_in_200_270', 'N_600_1000phd_per_ROI_event', 'kappa_hat', 'Z')} for m in MMED}
RES['spectral_350_annual'] = {MMED_LABEL[m]: {c: pick('annual', 'iso', m, 350.0, c) for c in ('A_600', 'pct248_obs', 'ratio_100_200_over_200_270', 'P0_100_200_given_one_in_200_270', 'N_600_1000phd_per_ROI_event', 'Z')} for m in MMED}
RES['spectral_380_annual'] = {MMED_LABEL[m]: {c: pick('annual', 'iso', m, 380.0, c) for c in ('A_600', 'pct248_obs', 'ratio_100_200_over_200_270', 'N_600_1000phd_per_ROI_event', 'Z')} for m in MMED}
RES['kinematic_window_june'] = {int(d): [pick('june', 'iso', np.inf, d, 'E_minus_keV'), pick('june', 'iso', np.inf, d, 'E_plus_keV')] for d in (250., 300., 350., 380., 390.)}
RES['contact_check_vs_P038'] = dict(note='P038 (annual, 600 phd): A=0.81/0.44/0.029 at 300/350/380; Z=2.67/3.12/3.47; extra 600-1000 phd per ROI event 0.16/1.06/30',
                                    A=[pick('annual', 'iso', np.inf, d, 'A_600') for d in (300., 350., 380.)],
                                    Z=[pick('annual', 'iso', np.inf, d, 'Z') for d in (300., 350., 380.)],
                                    N_ext=[pick('annual', 'iso', np.inf, d, 'N_600_1000phd_per_ROI_event') for d in (300., 350., 380.)],
                                    pct248_true_full_300=pick('june', 'iso', np.inf, 300., 'pct248_true_full'), P002='0.995 (June halo, true energy, in-ROI)')
RES['runtime_s'] = time.time() - T0
json.dump(RES, open(f'{OUT}/P051_results.json', 'w'), indent=1, default=float)

# ------------------------------------------------------------------------------------------------------
# 5. figures  (palette: dataviz reference instance; m_med as a single-hue sequential ramp, contact in ink)
# ------------------------------------------------------------------------------------------------------
INK, INK2, SURF = '#0b0b0b', '#52514e', '#fcfcfb'
SEQ = ['#bcd4f2', '#8fb6e8', '#5f95db', '#2a78d6', '#1d5aa8', '#123c74']      # 0.05 ... 1 GeV
COL = {m: c for m, c in zip(MMED[:-1], SEQ)}; COL[np.inf] = INK
ACC = ['#eb6834', '#1baf7a', '#eda100', '#e34948', '#4a3aa7']
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'text.color': INK, 'axes.facecolor': SURF, 'figure.facecolor': SURF, 'axes.grid': True, 'grid.color': '#e6e5e2',
                     'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False})

# Fig 1: spectra vs m_med at delta = 300, 350, 380 (isoscalar, 16 June halo), true energy, unit-normalised to the contact peak
fig, axs = plt.subplots(1, 3, figsize=(12, 3.9), sharey=False)
for ax, d in zip(axs, (300.0, 350.0, 380.0)):
    rc = spectrum('iso', d, 'june', np.inf)
    for m in MMED:
        r = spectrum('iso', d, 'june', m)
        ax.plot(E_T, r / rc.max(), color=COL[m], lw=2 if not np.isfinite(m) else 1.6, label=MMED_LABEL[m])
    ax.plot(E_T, EFF600 / PLATEAU * 1.05, color=INK2, lw=0.8, ls=':', label='S1c < 600 phd efficiency (P038), scaled' if d == 300 else None)
    ax.axvline(E_EV, color=ACC[0], lw=1.0); ax.text(E_EV + 4, 1.0, '248 keV', color=ACC[0], fontsize=7.5)
    ax.axvspan(E50_600, 720, color='#f0efec', zorder=0)
    ax.set_xlim(40, 620); ax.set_ylim(0, 1.12); ax.set_xlabel('true recoil energy E_R [keV]')
    ax.set_title(f'δ = {d:.0f} keV  (window {RES["kinematic_window_june"][int(d)][0]:.0f}–{RES["kinematic_window_june"][int(d)][1]:.0f} keV)', fontsize=9)
axs[0].set_ylabel('dR/dE, same c₀, relative to contact peak'); axs[0].legend(fontsize=7, title='mediator mass', title_fontsize=7.5, loc='upper right')
fig.suptitle('P051: light-mediator tilt of the inelastic O₁ spectrum, m_χ = 1 TeV, isoscalar, 16 June halo (grey: beyond the 600 phd 50 % edge)', fontsize=9.5)
fig.tight_layout(); fig.savefig(f'{FIG}/P051_fig1_spectra_vs_mmed.png', dpi=150); plt.close(fig)

# Fig 2: Z(delta) per m_med, isoscalar and proton-only, annual halo (P021 convention); contact in ink
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
for ax, tag, ttl in zip(axs, ('iso', 'p'), ('isoscalar O₁ (LZ convention)', 'proton-only O₁ (dark photon)')):
    for m in MMED:
        s = sel('annual', tag, m)
        ax.plot(s.index, s.Z, color=COL[m], lw=2 if not np.isfinite(m) else 1.6, marker='o', ms=3, label=MMED_LABEL[m])
    ax.axhline(3.4, color=INK2, lw=0.8, ls='--'); ax.text(252, 3.45, 'LZ local 3.4σ (best model)', fontsize=7.5, color=INK2)
    ax.axvline(386.6, color=INK2, lw=0.8, ls='-.'); ax.text(384, 0.3, 'δ_max(248 keV, 16 June)', rotation=90, fontsize=7, color=INK2, ha='right')
    ax.set_xlim(248, 392); ax.set_ylim(0, 4.0); ax.set_xlabel('mass splitting δ [keV]'); ax.set_title(ttl, fontsize=9)
axs[0].set_ylabel('single-event Z = √q₀ (κ profiled)'); axs[1].legend(fontsize=7, title='m_med', title_fontsize=7.5, loc='lower left')
fig.suptitle('P051: profile significance vs δ for contact and light mediators (1 TeV, annual-mean halo, 600 phd edge, σ_E = 11√(E/248) keV)', fontsize=9.5)
fig.tight_layout(); fig.savefig(f'{FIG}/P051_fig2_Z_vs_delta.png', dpi=150); plt.close(fig)

# Fig 3: constraint plane (m_A', eps), alpha_D = 0.1, annual halo, delta = 300/350/380
fig, ax = plt.subplots(figsize=(7.4, 5.0))
for i, d in enumerate((300.0, 350.0, 380.0)):
    P = plane[('annual', d)]
    ax.plot(P['mA'], P['eps_req'], color=ACC[i], lw=2, label=f'ε for one LZ event, δ = {d:.0f} keV')
    ax.plot(P['mA'], P['eps_floor'], color=ACC[i], lw=1.2, ls='--')
    ax.fill_between(P['mA'], 1e-10, P['eps_floor'], color=ACC[i], alpha=0.08, lw=0)
ax.plot([], [], color=INK2, lw=1.2, ls='--', label='χ₂-decay floor (≤ 1 exothermic event; same colour = same δ)')
ax.axhline(EPS_BABAR, color=INK2, lw=1.0, ls=':'); ax.text(0.034, 1.3e-3, 'BaBar visible/invisible A′ (recalled, likely): ε ≲ 10⁻³', fontsize=7.5, color=INK2)
# recalled beam-dump band (E137/E141/Orsay/NuCal/CHARM; visibly decaying A'): roughly eps 1e-7..1e-5 for m_A' <~ 0.1-0.4 GeV -- uncertain
bd_m = np.array([0.032, 0.1, 0.2, 0.3, 0.4]); bd_lo = np.array([2e-8, 1e-7, 3e-7, 1e-6, 3e-6]); bd_hi = np.array([1e-5, 1e-5, 6e-6, 4e-6, 3e-6])
ax.fill_between(bd_m, bd_lo, bd_hi, facecolor='none', hatch='///', edgecolor=INK2, lw=0.6, alpha=0.6)
ax.text(0.034, 1.4e-5, 'electron/proton beam dumps, visible A′ → e⁺e⁻\n(recalled, UNCERTAIN, ×3 in ε)', fontsize=7, color=INK2)
ax.axvline(9.2, color=ACC[4], lw=1.2); ax.text(8.6, 8e-3, 'Planck: m_A′ ≥ 9 GeV if α_D is thermal (P025)', rotation=90, fontsize=7.5, color=ACC[4], va='top', ha='right')
ax.axvline(0.246, color=INK2, lw=0.6, ls='-.'); ax.text(0.25, 3e-9, 'q(248 keV)', fontsize=7, color=INK2, rotation=90, va='bottom')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(0.032, 32); ax.set_ylim(1e-9, 1e-2)
ax.set_xlabel("mediator (dark-photon) mass m_A′ [GeV]"); ax.set_ylabel('kinetic mixing ε (α_D = 0.1; ε ∝ α_D^-1/2, floor likewise)')
ax.set_title('P051: (m_A′, ε) plane for the dark-photon reading of the LZ event, m_χ = 1 TeV, annual-mean halo', fontsize=9.5)
ax.legend(fontsize=7.5, loc='lower right')
fig.tight_layout(); fig.savefig(f'{FIG}/P051_fig3_mA_eps_plane.png', dpi=150); plt.close(fig)

# Fig 4: distinguishing predictions vs delta: N(100-200)/N(200-270) and 600-1000 phd events per ROI event
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.0))
ax = axs[0]
for m in MMED:
    s = sel('annual', 'iso', m)
    ax.plot(s.index, s.ratio_100_200_over_200_270, color=COL[m], lw=2 if not np.isfinite(m) else 1.6, label=MMED_LABEL[m])
for p0, lab in ((0.1, 'P(0) = 10 %'), (0.05, '5 %'), (0.01, '1 %')):
    ax.axhline(-math.log(p0), color=INK2, lw=0.7, ls=':'); ax.text(251, -math.log(p0) * 1.08, lab, fontsize=7, color=INK2)
ax.set_yscale('log'); ax.set_xlim(248, 392); ax.set_ylim(1e-3, 200); ax.set_xlabel('δ [keV]')
ax.set_ylabel('expected events in 100–200 keV per event in 200–270 keV'); ax.set_title('Low side of the window (LZ NR band empty 150–500 phd)', fontsize=9)
ax.legend(fontsize=7, title='m_med', title_fontsize=7.5, loc='upper right')
ax = axs[1]
for m in MMED:
    s = sel('annual', 'iso', m)
    ax.plot(s.index, s.N_600_1000phd_per_ROI_event, color=COL[m], lw=2 if not np.isfinite(m) else 1.6, label=MMED_LABEL[m])
ax.axhline(2.3, color=INK2, lw=0.7, ls=':'); ax.text(251, 2.5, 'P(0) = 10 %', fontsize=7, color=INK2)
ax.set_yscale('log'); ax.set_xlim(248, 392); ax.set_ylim(1e-2, 100); ax.set_xlabel('δ [keV]')
ax.set_ylabel('expected events in 600–1000 phd per ROI event'); ax.set_title('High side: the 600–1000 phd region LZ found empty (P038)', fontsize=9)
fig.suptitle('P051: what a light mediator predicts on either side of the 248 keV event (isoscalar, annual-mean halo)', fontsize=9.5)
fig.tight_layout(); fig.savefig(f'{FIG}/P051_fig4_predictions.png', dpi=150); plt.close(fig)
say(f'done, t={time.time() - T0:.0f}s')
