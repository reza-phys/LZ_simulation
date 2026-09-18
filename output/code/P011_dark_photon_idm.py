"""
P011_dark_photon_idm.py -- Dark-photon-mediated pseudo-Dirac (inelastic) dark matter and the LZ 248 keV event.

Model: pseudo-Dirac fermion chi = (chi1 + i chi2)/sqrt2 with Majorana mass splitting delta, coupled to a dark photon A'
(mass m_A', dark coupling alpha_D = g_D^2/4pi) that kinetically mixes with the photon (epsilon).  In the Majorana basis the
vector current is purely off-diagonal, chibar gamma^mu chi = i chi1bar gamma^mu chi2, so nuclear scattering is inelastic
(chi1 N -> chi2 N) and couples to the electric charge, i.e. to protons only.  Heavy-mediator limit (m_A' >> q ~ 250 MeV):
    c_p = e g_D eps / m_A'^2 = 4 pi sqrt(alpha alpha_D) eps / m_A'^2,   c_n = 0,     [O1 coupling, GeV^-2]
    sigma_p = c_p^2 mu_p^2 / pi = 16 pi alpha alpha_D eps^2 mu_p^2 / m_A'^4.
Light mediator: c_p(q) = c_p(0) m_A'^2/(m_A'^2 + q^2)  -> rate suppression m_A'^4/(m_A'^2+q^2)^2 (WimPyDD q-dependent WC).
WimPyDD convention (P003/P007): c^0 = c_p + c_n, c^1 = c_p - c_n  -> proton-only means c^0 = c^1 = c_p.

Parts
  A  couplings, unit cross-section (c_p = 1/m_v^2), Higgsino couplings (P007) for the comparison line
  B  halos (June 16, annual mean, Sun frame) on a common v_min grid; efficiency model (P007: 0.96 plateau, erf 50% at 269.9 keV)
  C  WimPyDD kernels for m = 300, 1000, 3000 GeV, delta = 200 ... delta_max in 5 keV steps, three O1 hamiltonians
     (proton-only unit, isoscalar unit, Higgsino) -> expected LZ events in 2.84 t yr, plus low-energy companions
  D  sigma_p(delta) giving N = 1 (bands 0.3-3 and 0.105-3.65), Higgsino-equivalent sigma_p, LZ Fig. 6/S7 edges converted
     to proton-only cross-sections (q->0 factor (A/Z)^2 and the effective WimPyDD factor)
  E  light mediators m_A' = 0.1, 0.3, 1 GeV at 1000 GeV: suppression S(delta, m_A') = N_light/N_heavy
  F  eps^2 alpha_D / m_A'^4 and the required eps for alpha_D = 0.1 and m_A' = 0.3, 1, 3, 10 GeV; comparison with recalled bounds
  G  relic density: alpha_D(m) from <sigma v>(chi chi -> A'A') = pi alpha_D^2/m^2 = 2.2e-26 cm^3/s; implied eps
  H  chi2 lifetime (chi2 -> chi1 nu nubar via A'-Z mixing; delta < 2 m_e) and the exothermic down-scattering of surviving chi2
  I  figures

Run from the simulation root:  .venv/bin/python output/code/P011_dark_photon_idm.py   [--recompute]
"""
from __future__ import annotations
import sys, os, json, math, time
sys.path.insert(0, 'output/code')
import numpy as np
import pandas as pd
from scipy.special import erf, erfc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import lzcommon as lz

T0 = time.time()
OUT = 'output/work/P011'
FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)
RECOMPUTE = '--recompute' in sys.argv

# ----------------------------------------------------------------------------------------------
# Part A: couplings and cross-sections
# ----------------------------------------------------------------------------------------------
ALPHA = 1.0 / 137.035999                     # recalled, certain
E_EM = math.sqrt(4 * math.pi * ALPHA)
MV = lz.M_V_GEV                              # 246.2 GeV (LZ/Anand unit coupling c = 1/m_v^2)
MP = lz.M_NUCLEON_GEV
GEV2_TO_CM2 = lz.GEV_TO_CM2
GF = 1.1663788e-5                            # GeV^-2, recalled certain
SW2 = 0.2312                                 # sin^2 theta_W (low-energy), recalled certain (+-0.001)
MZ = 91.1876                                 # GeV, recalled certain
Z_XE, A_XE = 54, lz.A_XE_MEAN
SIGV_THERMAL_CM3S = 2.2e-26                  # Steigman-Dasgupta-Beacom 2012 value for m >> 10 GeV, recalled certain
CM3S_PER_GEV2 = GEV2_TO_CM2 * 2.99792458e10  # 1 GeV^-2 (natural units, x c) = 1.167e-17 cm^3/s
T_UNIVERSE_S = 13.8e9 * 3.15576e7            # recalled certain
HBAR_GEV_S = 6.582119569e-25
M_E_MEV = 0.51099895

def mu_p(m):
    return m * MP / (m + MP)

def sigma_from_c(c_gev2, m):
    """Per-nucleon cross-section (cm^2) for O1 coupling c (GeV^-2): sigma = c^2 mu^2 / pi."""
    return c_gev2 ** 2 * mu_p(m) ** 2 / math.pi * GEV2_TO_CM2

def c_from_sigma(sigma_cm2, m):
    return math.sqrt(sigma_cm2 / GEV2_TO_CM2 * math.pi) / mu_p(m)

def eps2alphaD_over_mA4(sigma_p_cm2, m):
    """eps^2 alpha_D / m_A'^4 [GeV^-4] from sigma_p = 16 pi alpha alpha_D eps^2 mu^2 / m_A'^4 (heavy mediator)."""
    return sigma_p_cm2 / GEV2_TO_CM2 / (16 * math.pi * ALPHA * mu_p(m) ** 2)

def propagator_factor(mA, E_keV=248.0, A=A_XE):
    q = math.sqrt(2 * lz.m_nucleus_gev(A) * E_keV * 1e-6)
    return (mA ** 2 / (mA ** 2 + q ** 2)) ** 2

C_UNIT = 1.0 / MV ** 2                      # proton-only unit coupling c_p = 1/m_v^2
# Higgsino (P007): c_p = (G_F/sqrt2)(1-4 s_W^2), c_n = -G_F/sqrt2
C_P_HIG = 2 * math.sqrt(2) * GF * (0.25 - SW2)
C_N_HIG = -2 * math.sqrt(2) * GF * 0.25
Q_KEV248 = math.sqrt(2 * lz.m_nucleus_gev(A_XE) * 248e-6)
partA = dict(alpha=ALPHA, e=E_EM, m_v_GeV=MV, c_unit_GeV_m2=C_UNIT,
             q_248keV_GeV=Q_KEV248,
             sigma_p_unit_cm2={m: sigma_from_c(C_UNIT, m) for m in (300.0, 1000.0, 3000.0)},
             mu_p_GeV={m: mu_p(m) for m in (300.0, 1000.0, 3000.0)},
             higgsino_c_p=C_P_HIG, higgsino_c_n=C_N_HIG,
             higgsino_sigma_n_cm2_1000=sigma_from_c(C_N_HIG, 1000.0),
             higgsino_equiv_sigma_p_q0_cm2_1000=sigma_from_c(abs(C_P_HIG * Z_XE + C_N_HIG * (A_XE - Z_XE)) / Z_XE, 1000.0),
             A_over_Z_sq=(A_XE / Z_XE) ** 2,
             propagator_factor_248keV={mA: propagator_factor(mA) for mA in (0.1, 0.3, 1.0, 3.0, 10.0)})
print('Part A: q(248 keV) = %.3f GeV; sigma_p(c_p=1/m_v^2, 1 TeV) = %.3e cm^2; Higgsino sigma_n = %.3e, equivalent proton-only '
      'sigma_p(q->0) = %.3e cm^2; (A/Z)^2 = %.3f' % (Q_KEV248, partA['sigma_p_unit_cm2'][1000.0], partA['higgsino_sigma_n_cm2_1000'],
                                                   partA['higgsino_equiv_sigma_p_q0_cm2_1000'], partA['A_over_Z_sq']))
json.dump(partA, open(f'{OUT}/couplings.json', 'w'), indent=1, default=float)

# ----------------------------------------------------------------------------------------------
# Part B: halos, efficiency
# ----------------------------------------------------------------------------------------------
WD = lz.wd()
VGRID = np.linspace(0.0, 844.0, 1200)              # lzcommon default grid: v_esc + 300 km/s, 1200 points
ONES = np.ones_like(VGRID)
days12 = 15.0 + 365.25 / 12 * np.arange(12)
halos = {'june16': lz.wd_halo(day_of_year=167, vmin=VGRID)[1],
         'annual': np.mean([lz.wd_halo(day_of_year=d, vmin=VGRID)[1] for d in days12], axis=0),
         'sun': lz.wd_halo(vmin=VGRID)[1]}
EXPOSURE = lz.LZ['exposure_tyr']
E50_HI, E50_LO, PLATEAU = lz.LZ['E_50pct_high_keV'], lz.LZ['E_50pct_low_keV'], lz.LZ['eff_plateau']
EFF_SIGS = [8.0, 11.5, 15.0]

def efficiency(E, sig_hi=11.5, sig_lo=2.5):
    return PLATEAU * 0.5 * (1 + erf((E - E50_LO) / (math.sqrt(2) * sig_lo))) * 0.5 * erfc((E - E50_HI) / (math.sqrt(2) * sig_hi))

def window(E, r, lo, hi, eff=True, sig_hi=11.5):
    """Exposure-weighted events in [lo, hi] keV; r = 0 outside the computed E grid (onset handled by WimPyDD)."""
    Ef = np.linspace(lo, hi, 1501)
    y = np.interp(Ef, E, r, left=0.0, right=0.0)
    if eff:
        y = y * efficiency(Ef, sig_hi=sig_hi)
    return float(np.trapezoid(y, Ef)) * EXPOSURE

E_MAX = 330.0
E_STEP = 3.0
def egrid(m, delta):
    lo = lz.E_R_range_keV(m, VGRID[-1], A=124.0, delta_kev=delta)[0]
    if math.isnan(lo):
        return None
    return np.arange(max(1.0, math.floor(lo) - 4.0), E_MAX + 1e-9, E_STEP)

def kernel(ham, m, delta, E, **kw):
    K = np.array([WD.diff_rate(WD.Xe, ham, m, float(e), VGRID, ONES, j_chi=0.5, delta=float(delta), sum_over_streams=False, **kw)
                  for e in E])
    return K * 1000.0 * 365.25                       # /kg/day -> /t/yr

HAMS = {'p': lz.wd_hamiltonian('p_only_unit', {1: (C_UNIT, C_UNIT)}),           # c_p = 1/m_v^2, c_n = 0
        'iso': lz.wd_hamiltonian('iso_unit', {1: (2 * C_UNIT, 0.0)}),             # c_p = c_n = 1/m_v^2 (LZ unit coupling)
        'hig': lz.wd_hamiltonian('higgsino_Z', {1: (C_P_HIG + C_N_HIG, C_P_HIG - C_N_HIG)})}
MASSES = [300.0, 1000.0, 3000.0]
DELTA_STOP = {300.0: 320.0, 1000.0: 400.0, 3000.0: 415.0}     # just beyond delta_max(248 keV, June) of P002/P007

# validation (i): proton-only vs isoscalar at q -> 0 must give (Z/A)^2; (ii) kernel.dot(halo) vs wd_rate
val = {}
r_p = kernel(HAMS['p'], 1000.0, 0.0, np.array([3.0, 5.0])) @ halos['sun']
r_i = kernel(HAMS['iso'], 1000.0, 0.0, np.array([3.0, 5.0])) @ halos['sun']
val['p_over_iso_lowE'] = (r_p / r_i).tolist(); val['Z_over_A_sq'] = (Z_XE / A_XE) ** 2
kd = float((kernel(HAMS['p'], 1000.0, 300.0, np.array([200.0])) @ halos['june16'])[0])
wr = lz.wd_rate(HAMS['p'], 1000.0, 200.0, halo=(VGRID, halos['june16']), delta_kev=300.0)
val['kernel_dot_vs_wd_rate_200keV_d300'] = [kd, wr, kd / wr]
print('validation: p-only/isoscalar at 3,5 keV =', np.round(val['p_over_iso_lowE'], 4), ' (Z/A)^2 =', round(val['Z_over_A_sq'], 4),
      '; kernel/wd_rate =', round(kd / wr, 4))

# ----------------------------------------------------------------------------------------------
# Part C: event grid
# ----------------------------------------------------------------------------------------------
GRID_CSV = f'{OUT}/N_events_grid.csv'
SPEC_JSON = f'{OUT}/spectra_1000GeV.json'
if os.path.exists(GRID_CSV) and not RECOMPUTE:
    df = pd.read_csv(GRID_CSV)
    spectra_store = json.load(open(SPEC_JSON))
    print('Part C: cached grid loaded (use --recompute to redo the WimPyDD loop)')
else:
    rows, spectra_store = [], {}
    for m in MASSES:
        deltas = np.arange(200.0, DELTA_STOP[m] + 1e-9, 5.0)
        for d in deltas:
            E = egrid(m, d)
            rates = {}
            if E is not None:
                for key, ham in HAMS.items():
                    K = kernel(ham, m, d, E)
                    rates[key] = {name: K @ deta for name, deta in halos.items()}
            if m == 1000.0 and int(d) in (200, 250, 300, 350, 370, 380, 390):
                spectra_store[str(int(d))] = dict(E=E.tolist(), **{f'{k}_{h}': v.tolist() for k, hv in rates.items() for h, v in hv.items()})
            for name in halos:
                for s in EFF_SIGS:
                    row = dict(m_GeV=m, delta_keV=d, halo=name, eff_sigma_keV=s)
                    for key in HAMS:
                        if E is None:
                            row.update({f'N_{key}': 0.0, f'N_{key}_lo': 0.0, f'N_{key}_hi': 0.0, f'N_{key}_noeff': 0.0})
                        else:
                            r = rates[key][name]
                            row[f'N_{key}'] = window(E, r, 1.0, E_MAX, sig_hi=s)
                            row[f'N_{key}_lo'] = window(E, r, 5.4, 55.0, sig_hi=s)
                            row[f'N_{key}_hi'] = window(E, r, 200.0, 270.0, sig_hi=s)
                            row[f'N_{key}_noeff'] = window(E, r, 1.0, E_MAX, eff=False)
                    rows.append(row)
        print(f'  mass {m:.0f} done ({len(deltas)} deltas), t = {time.time() - T0:.0f} s', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(GRID_CSV, index=False)
    json.dump(spectra_store, open(SPEC_JSON, 'w'))

# validation (iii): Higgsino counts vs P007 (annual halo, sigma_eff 11.5, 1000 GeV)
p7 = pd.read_csv('output/work/P007/N_events_grid.csv')
def N7(m, d, halo='annual'):
    s = p7[(p7.m_GeV == m) & (np.isclose(p7.delta_keV, d)) & (p7.halo == halo) & (p7.eff_sigma_keV == 11.5)]
    return float(s.N_events.iloc[0]) if len(s) else np.nan
sel = lambda m, halo, s=11.5: df[(df.m_GeV == m) & (df.halo == halo) & (df.eff_sigma_keV == s)].sort_values('delta_keV')
val['higgsino_vs_P007_1000GeV_annual'] = {int(d): [float(sel(1000.0, 'annual')[np.isclose(sel(1000.0, 'annual').delta_keV, d)].N_hig.iloc[0]), N7(1000.0, d)]
                                          for d in (300.0, 350.0, 370.0, 380.0)}
print('validation: Higgsino N (this work vs P007), 1000 GeV annual:', {k: [round(a, 3), round(b, 3)] for k, (a, b) in val['higgsino_vs_P007_1000GeV_annual'].items()})

# ----------------------------------------------------------------------------------------------
# Part D: required sigma_p(delta), Higgsino-equivalent sigma_p, LZ edges converted to proton-only
# ----------------------------------------------------------------------------------------------
N_TARGETS = {'N1': 1.0, 'N0p3': 0.3, 'N3': 3.0, 'N0p105': 0.105, 'N3p65': 3.65}
req_rows = []
for m in MASSES:
    su = sigma_from_c(C_UNIT, m)
    dmax248 = lz.delta_max_kev(248.0, m, v_kms=lz.vmax_kms(lz.v_earth_kms(167)))
    for name in halos:
        for s in EFF_SIGS:
            sub = sel(m, name, s)
            for r in sub.itertuples():
                Np = r.N_p
                row = dict(m_GeV=m, delta_keV=r.delta_keV, halo=name, eff_sigma_keV=s, N_p_unit=Np, N_iso_unit=r.N_iso, N_hig=r.N_hig,
                           N_lo_per_event=(r.N_p_lo / r.N_p if r.N_p > 0 else np.nan), frac_hi=(r.N_p_hi / r.N_p if r.N_p > 0 else np.nan),
                           N_lo_per_hi_event=(r.N_p_lo / r.N_p_hi if r.N_p_hi > 0 else np.nan),
                           delta_max_248keV_june=dmax248)
                for k, t in N_TARGETS.items():
                    row[f'sigma_p_{k}_cm2'] = su * t / Np if Np > 0 else np.nan
                # Higgsino equivalent proton-only cross-section: same LZ count as the Higgsino at this delta
                row['sigma_p_higgsino_equiv_cm2'] = su * r.N_hig / Np if Np > 0 else np.nan
                row['effective_A_over_Z_sq'] = r.N_iso / Np if Np > 0 else np.nan
                req_rows.append(row)
req = pd.DataFrame(req_rows)
req.to_csv(f'{OUT}/sigma_p_required.csv', index=False)

def delta_for_N(sub, target, col='N_hig'):
    """delta at which the count column crosses target (log interpolation on the descending branch)."""
    d, N = sub.delta_keV.values, sub[col].values
    ok = N > 0
    d, N = d[ok], np.log(N[ok])
    i = np.where((N[:-1] >= math.log(target)) & (N[1:] < math.log(target)))[0]
    if len(i) == 0:
        return np.nan
    i = i[-1]
    return float(d[i] + (N[i] - math.log(target)) / (N[i] - N[i + 1]) * (d[i + 1] - d[i]))

summary_D = {}
for m in MASSES:
    su = sigma_from_c(C_UNIT, m)
    sub = sel(m, 'annual')
    subs = sel(m, 'sun'); subj = sel(m, 'june16')
    pick = lambda d, col, s=sub: float(s[np.isclose(s.delta_keV, d)][col].iloc[0]) if len(s[np.isclose(s.delta_keV, d)]) else np.nan
    dl = [250.0, 300.0, 350.0] + ([365.0, 380.0] if m == 1000.0 else [])
    summary_D[m] = dict(
        delta_max_248keV_june=lz.delta_max_kev(248.0, m, v_kms=lz.vmax_kms(lz.v_earth_kms(167))),
        higgsino_delta_N1={h: delta_for_N(sel(m, h), 1.0) for h in halos},
        higgsino_delta_N3p65={h: delta_for_N(sel(m, h), 3.65) for h in halos},
        higgsino_delta_N0p105={h: delta_for_N(sel(m, h), 0.105) for h in halos},
        sigma_p_N1_annual={d: su / pick(d, 'N_p') for d in dl if pick(d, 'N_p') > 0},
        sigma_p_N1_june16={d: su / pick(d, 'N_p', subj) for d in dl if pick(d, 'N_p', subj) > 0},
        sigma_p_N1_sun={d: su / pick(d, 'N_p', subs) for d in dl if pick(d, 'N_p', subs) > 0},
        N_p_unit_annual={d: pick(d, 'N_p') for d in dl},
        N_lo_per_event_annual={d: pick(d, 'N_p_lo') / pick(d, 'N_p') for d in dl},
        higgsino_equiv_sigma_p_annual={d: su * pick(d, 'N_hig') / pick(d, 'N_p') for d in dl if pick(d, 'N_p') > 0},
        higgsino_over_N1_ratio_annual={d: pick(d, 'N_hig') for d in dl},
        effective_A_over_Z_sq_annual={d: pick(d, 'N_iso') / pick(d, 'N_p') for d in dl if pick(d, 'N_p') > 0},
        june_over_dec_like_ratio_june_over_annual={d: pick(d, 'N_p', subj) / pick(d, 'N_p') for d in dl if pick(d, 'N_p') > 0})
    print(f'Part D m={m:.0f}: sigma_p(N=1, annual) =', {d: '%.2e' % v for d, v in summary_D[m]['sigma_p_N1_annual'].items()},
          '; Higgsino delta(N=1) annual = %.1f' % summary_D[m]['higgsino_delta_N1']['annual'],
          '; N_lo/event =', {d: '%.2g' % v for d, v in summary_D[m]['N_lo_per_event_annual'].items()})

# delta below which the low-energy companions (5.4-55 keV events per 200-270 keV event, P003's N_lo) exceed tolerances
def delta_for_Nlo(m, tol):
    s = sel(m, 'annual'); x = (s.N_p_lo / s.N_p_hi).values; d = s.delta_keV.values
    if x[0] <= tol:
        return float(d[0])            # already below tolerance at the first grid point (200 keV)
    i = np.where((x[:-1] > tol) & (x[1:] <= tol))[0]
    return float(np.interp(tol, [x[i[0] + 1], x[i[0]]], [d[i[0] + 1], d[i[0]]])) if len(i) else np.nan
summary_D['delta_where_N_lo_per_hi_event_below'] = {m: {tol: delta_for_Nlo(m, tol) for tol in (1.0, 3.0, 5.0, 10.0)} for m in MASSES}
s0 = sel(1000.0, 'annual')
summary_D['N_lo_per_hi_event_1000GeV'] = {int(d): dict(proton_only=float(s0[np.isclose(s0.delta_keV, d)].N_p_lo.iloc[0] / s0[np.isclose(s0.delta_keV, d)].N_p_hi.iloc[0]),
                                                      isoscalar=float(s0[np.isclose(s0.delta_keV, d)].N_iso_lo.iloc[0] / s0[np.isclose(s0.delta_keV, d)].N_iso_hi.iloc[0]))
                                          for d in (200.0, 210.0, 220.0, 250.0)}
print('Part D: N_lo per 200-270 keV event, 1000 GeV (P003 isoscalar value at 200 keV: 7.6):', {k: {a: round(b, 2) for a, b in v.items()} for k, v in summary_D['N_lo_per_hi_event_1000GeV'].items()},
      '; delta where N_lo <= 3:', {m: round(v[3.0], 1) for m, v in summary_D['delta_where_N_lo_per_hi_event_below'].items()})
# efficiency-width sensitivity (1000 GeV, annual): N_p at sigma_hi = 8 / 15 keV relative to 11.5 keV
summary_D['eff_width_sensitivity_1000GeV'] = {}
for d in (300.0, 350.0, 365.0, 380.0):
    n = {s: float(sel(1000.0, 'annual', s)[np.isclose(sel(1000.0, 'annual', s).delta_keV, d)].N_p.iloc[0]) for s in EFF_SIGS}
    summary_D['eff_width_sensitivity_1000GeV'][d] = {f'N_sig{s:g}_over_N_sig11.5': n[s] / n[11.5] for s in EFF_SIGS}
print('Part D: efficiency-width sensitivity (N at sigma_hi 8, 15 keV / N at 11.5), 1000 GeV:', {d: [round(v['N_sig8_over_N_sig11.5'], 3), round(v['N_sig15_over_N_sig11.5'], 3)] for d, v in summary_D['eff_width_sensitivity_1000GeV'].items()})

# LZ digitised intervals (P007, Fig. 6 in (c_1^s m_v^2)^2 and Fig. S7 in sigma_SI) converted to proton-only sigma_p
lzint = pd.read_csv('output/work/P007/lz_intervals_digitised.csv')
band_rows = []
for r in lzint.itertuples():
    d = float(r.delta_keV)
    if d < 200:
        continue
    for name in ('annual', 'sun'):
        s = sel(1000.0, name)
        if not len(s[np.isclose(s.delta_keV, d)]):
            continue
        Niso, Np = float(s[np.isclose(s.delta_keV, d)].N_iso.iloc[0]), float(s[np.isclose(s.delta_keV, d)].N_p.iloc[0])
        su = sigma_from_c(C_UNIT, 1000.0)
        for edge, cval in (('lower', r.c1s_mv2_sq_lower), ('median', r.c1s_mv2_sq_median), ('upper', r.c1s_mv2_sq_upper)):
            band_rows.append(dict(delta_keV=d, halo=name, edge=edge, c1s_mv2_sq=cval, sigma_SI_cm2=su * cval,
                                  N_events_at_edge=Niso * cval, effective_A_over_Z_sq=Niso / Np, A_over_Z_sq_q0=(A_XE / Z_XE) ** 2,
                                  sigma_p_edge_cm2=su * Niso * cval / Np, sigma_p_edge_q0_conversion_cm2=su * cval * (A_XE / Z_XE) ** 2))
band = pd.DataFrame(band_rows)
band.to_csv(f'{OUT}/lz_band_proton_only.csv', index=False)
print('Part D: LZ upper edges -> proton-only sigma_p (annual):')
print(band[(band.halo == 'annual') & (band.edge == 'upper')][['delta_keV', 'sigma_SI_cm2', 'N_events_at_edge', 'effective_A_over_Z_sq', 'sigma_p_edge_cm2']].to_string(index=False))

# ----------------------------------------------------------------------------------------------
# Part E: light mediators at 1000 GeV
# ----------------------------------------------------------------------------------------------
LIGHT_CSV = f'{OUT}/light_mediator_suppression.csv'
MA_LIGHT = [0.1, 0.3, 1.0]
if os.path.exists(LIGHT_CSV) and not RECOMPUTE:
    light = pd.read_csv(LIGHT_CSV)
    print('Part E: cached light-mediator table loaded')
else:
    lrows = []
    for mA in MA_LIGHT:
        ham_l = WD.eft_hamiltonian(f'p_light_{mA}', {(1, 'q2'): (lambda q, A=1.0, mA=mA: [A * mA ** 2 / (mA ** 2 + q ** 2) * C_UNIT,
                                                                                          A * mA ** 2 / (mA ** 2 + q ** 2) * C_UNIT])})
        for d in np.arange(200.0, 400.0 + 1e-9, 10.0):
            E = egrid(1000.0, d)
            K = kernel(ham_l, 1000.0, d, E, A=1.0)
            Kh = kernel(HAMS['p'], 1000.0, d, E)
            for name in halos:
                rl, rh = K @ halos[name], Kh @ halos[name]
                Nl, Nh = window(E, rl, 1.0, E_MAX), window(E, rh, 1.0, E_MAX)
                lrows.append(dict(m_A_GeV=mA, delta_keV=d, halo=name, N_light_unit=Nl, N_heavy_unit=Nh, S_light_over_heavy=Nl / Nh if Nh > 0 else np.nan,
                                  S_analytic_248keV=propagator_factor(mA)))
        print(f'  light mediator m_A = {mA} GeV done, t = {time.time() - T0:.0f} s', flush=True)
    light = pd.DataFrame(lrows)
    light.to_csv(LIGHT_CSV, index=False)
print('Part E: S = N_light/N_heavy (annual) at delta = 250/300/350:')
for mA in MA_LIGHT:
    s = light[(light.m_A_GeV == mA) & (light.halo == 'annual')]
    print('   m_A = %.1f GeV: %s  (analytic at 248 keV: %.3f)' % (mA, {int(d): round(float(s[np.isclose(s.delta_keV, d)].S_light_over_heavy.iloc[0]), 3) for d in (250, 300, 350)}, propagator_factor(mA)))

def S_factor(mA, d):
    """Rate suppression relative to the heavy-mediator formula at the same c_p(0) (annual halo, 1000 GeV).
    Exact WimPyDD values at m_A' = 0.1, 0.3, 1 GeV; elsewhere the analytic propagator at q(248 keV) times the
    WimPyDD/analytic ratio interpolated in log m_A' (clamped at the ends: the ratio -> 1 for heavy A')."""
    ratios = []
    for v in MA_LIGHT:
        s = light[(light.m_A_GeV == v) & (light.halo == 'annual')].sort_values('delta_keV')
        ratios.append(float(np.interp(d, s.delta_keV, s.S_light_over_heavy)) / propagator_factor(v))
    if mA >= 1.0:
        return propagator_factor(mA) * (1.0 + (ratios[-1] - 1.0) * max(0.0, 1.0 - math.log10(mA) / 1.0))   # ratio -> 1 by 10 GeV
    return propagator_factor(mA) * float(np.interp(math.log10(mA), np.log10(MA_LIGHT), ratios))

# ----------------------------------------------------------------------------------------------
# Part F and G: eps translation, relic density
# ----------------------------------------------------------------------------------------------
def alpha_D_thermal(m):
    """<sigma v>(chi chi -> A' A') ~ pi alpha_D^2 / m^2 (m_A' << m; recalled, likely) = 2.2e-26 cm^3/s."""
    return math.sqrt(SIGV_THERMAL_CM3S / CM3S_PER_GEV2 * m ** 2 / math.pi)

MA_LIST = [0.3, 1.0, 3.0, 10.0]
EPS_BABAR = 1e-3        # recalled, likely: BaBar visible (2014) and invisible (2017) A' searches, eps <~ 1e-3 for 0.02-10 GeV
eps_rows = []
for m in MASSES:
    su = sigma_from_c(C_UNIT, m)
    sub = sel(m, 'annual')
    aDth = alpha_D_thermal(m)
    dl = [250.0, 300.0, 350.0] + ([365.0, 380.0] if m == 1000.0 else [])
    for d in dl:
        row = sub[np.isclose(sub.delta_keV, d)]
        if not len(row) or float(row.N_p.iloc[0]) <= 0:
            continue
        Np = float(row.N_p.iloc[0])
        sig1 = su / Np
        X = eps2alphaD_over_mA4(sig1, m)
        for mA in MA_LIST:
            S = S_factor(mA, d)      # 1000 GeV WimPyDD ratio reused for 300/3000 GeV (q distribution nearly mass-independent above 300 GeV)
            for aD, lab in ((0.1, 'alpha_D=0.1'), (aDth, 'alpha_D=thermal')):
                eps = math.sqrt(X * mA ** 4 / aD / S)
                eps_rows.append(dict(m_GeV=m, delta_keV=d, sigma_p_N1_cm2=sig1, eps2alphaD_over_mA4_GeV_m4=X, m_A_GeV=mA, S_propagator=S,
                                     alpha_D_label=lab, alpha_D=aD, eps_required=eps, eps_over_babar=eps / EPS_BABAR))
eps_df = pd.DataFrame(eps_rows)
eps_df.to_csv(f'{OUT}/epsilon_required.csv', index=False)
print('Part F: required eps (alpha_D = 0.1), 1000 GeV, annual halo, N = 1:')
print(eps_df[(eps_df.m_GeV == 1000.0) & (eps_df.alpha_D_label == 'alpha_D=0.1')].pivot(index='delta_keV', columns='m_A_GeV', values='eps_required').to_string(float_format='%.2e'))
relic = {m: dict(alpha_D_thermal=alpha_D_thermal(m), g_D_thermal=math.sqrt(4 * math.pi * alpha_D_thermal(m))) for m in MASSES}
# s-channel chi1 chi2 -> A'* -> f fbar relative to A'A' (m_A' << m): ratio ~ (alpha eps^2 / alpha_D) * sum_f N_c Q_f^2 * O(1); recalled scaling
relic['s_channel_over_AA_ratio_scaling'] = 'sigma v(ff) / sigma v(A A) ~ (alpha eps^2/alpha_D) x Sum_f N_c Q_f^2 (~ 20/3 for f lighter than 1 TeV); with eps <= 1e-3 this is < 1e-3, negligible'
relic['s_channel_only_if_mA_above_mchi'] = 'if m_A > m_chi the A A channel closes and <sigma v>(chi1 chi2 -> f f) ~ 16 pi alpha alpha_D eps^2 N_eff m^2 / (3 m_A^4) needs eps ~ O(1) for alpha_D <= 1: excluded by dark-photon searches, so m_A < m_chi is required'

# ----------------------------------------------------------------------------------------------
# Part H: chi2 lifetime and exothermic down-scattering of surviving chi2
# ----------------------------------------------------------------------------------------------
# chi2 -> chi1 nu nubar through the A' coupling to the Z current, g_nu = eps tan(theta_W) (m_A'^2/m_Z^2) g/(4 cos theta_W) (m_A' << m_Z);
# the A' propagator 1/m_A'^2 cancels the m_A'^2 in the coupling:
#   G_eff = g_D eps tan(theta_W) g / (4 cos(theta_W) m_Z^2);  Gamma = N_nu G_eff^2 delta^5 / (120 pi^3)  (derived in details.md, NR limit)
SW, CW = math.sqrt(SW2), math.sqrt(1 - SW2)
G_WEAK = E_EM / SW
def gamma_chi2(eps, aD, delta_kev, n_nu=3):
    gD = math.sqrt(4 * math.pi * aD)
    Geff = gD * eps * (SW / CW) * G_WEAK / (4 * CW * MZ ** 2)
    return n_nu * Geff ** 2 * (delta_kev * 1e-6) ** 5 / (120 * math.pi ** 3)
def tau_chi2_s(eps, aD, delta_kev):
    return HBAR_GEV_S / gamma_chi2(eps, aD, delta_kev)

# exothermic (chi2 N -> chi1 N, delta -> -delta) expected events per unit chi2 fraction at the proton-only unit coupling, 1000 GeV
EXO_CSV = f'{OUT}/exothermic.csv'
if os.path.exists(EXO_CSV) and not RECOMPUTE:
    exo = pd.read_csv(EXO_CSV)
else:
    erows = []
    for d in (250.0, 300.0, 350.0, 365.0, 380.0):
        E = np.arange(1.0, E_MAX + 1e-9, E_STEP)
        K = kernel(HAMS['p'], 1000.0, -d, E)
        for name in halos:
            r = K @ halos[name]
            erows.append(dict(delta_keV=d, halo=name, N_exo_unit_ROI=window(E, r, 1.0, E_MAX), N_exo_unit_lo=window(E, r, 5.4, 55.0),
                              N_exo_unit_hi=window(E, r, 200.0, 270.0), N_exo_unit_noeff_to330=window(E, r, 1.0, E_MAX, eff=False),
                              E_onset_keV=float(E[np.argmax(r > 0)]) if np.any(r > 0) else np.nan))
    exo = pd.DataFrame(erows)
    exo.to_csv(EXO_CSV, index=False)
life_rows = []
for d in (250.0, 300.0, 350.0, 365.0, 380.0):
    sub = sel(1000.0, 'annual'); row = sub[np.isclose(sub.delta_keV, d)]
    Np = float(row.N_p.iloc[0]); su = sigma_from_c(C_UNIT, 1000.0); sig1 = su / Np
    Nexo_unit = float(exo[(exo.delta_keV == d) & (exo.halo == 'annual')].N_exo_unit_ROI.iloc[0])
    Nexo_at_sig1 = Nexo_unit / Np                      # exothermic ROI events if the chi2 fraction today were 1 (f2 = 1)
    f2_max = 1.0 / Nexo_at_sig1                        # <= 1 exothermic event
    tau_max = T_UNIVERSE_S / math.log(1.0 / f2_max)    # f2 = exp(-t_U/tau)
    for aD in (0.1, alpha_D_thermal(1000.0)):
        eps_min = math.sqrt(tau_chi2_s(1.0, aD, d) / tau_max)
        life_rows.append(dict(delta_keV=d, alpha_D=aD, sigma_p_N1_cm2=sig1, N_exo_if_f2_is_1=Nexo_at_sig1, f2_max=f2_max, tau_max_s=tau_max,
                              tau_at_eps_1em6_s=tau_chi2_s(1e-6, aD, d), tau_at_eps_1em4_s=tau_chi2_s(1e-4, aD, d), eps_min_for_no_exothermic=eps_min))
life = pd.DataFrame(life_rows)
life.to_csv(f'{OUT}/chi2_lifetime.csv', index=False)

# allowed dark-photon mass window: eps_req(m_A') between the chi2-decay floor and the (recalled) BaBar ceiling, 1000 GeV
mA_scan = np.logspace(-1.5, 2.0, 400)
win_rows = []
for d in (250.0, 300.0, 350.0, 365.0, 380.0):
    Np = float(sel(1000.0, 'annual')[np.isclose(sel(1000.0, 'annual').delta_keV, d)].N_p.iloc[0]); X = eps2alphaD_over_mA4(sigma_from_c(C_UNIT, 1000.0) / Np, 1000.0)
    for aD in (0.1, alpha_D_thermal(1000.0)):
        eps_req = np.array([math.sqrt(X * x ** 4 / aD / S_factor(x, d)) for x in mA_scan])
        e_min = float(life[(np.isclose(life.delta_keV, d)) & (np.isclose(life.alpha_D, aD))].eps_min_for_no_exothermic.iloc[0])
        ok = (eps_req >= e_min) & (eps_req <= EPS_BABAR)
        win_rows.append(dict(delta_keV=d, alpha_D=aD, eps_floor_chi2_decay=e_min, eps_ceiling_babar_recalled=EPS_BABAR,
                             mA_min_GeV=float(mA_scan[ok].min()) if ok.any() else np.nan, mA_max_GeV=float(mA_scan[ok].max()) if ok.any() else np.nan,
                             eps_req_at_1GeV=float(np.interp(0.0, np.log10(mA_scan), eps_req)), eps_req_at_10GeV=float(np.interp(1.0, np.log10(mA_scan), eps_req))))
win = pd.DataFrame(win_rows)
win.to_csv(f'{OUT}/mA_window.csv', index=False)
print('Part H: allowed m_A window (1000 GeV): eps between chi2-decay floor and BaBar 1e-3 (recalled):')
print(win.to_string(index=False, float_format='%.3g'))
print('Part H: chi2 lifetime and exothermic constraint (1000 GeV, annual):')
print(life.to_string(index=False, float_format='%.3g'))
json.dump(dict(relic=relic, chi2_width_formula='Gamma = 3 G_eff^2 delta^5/(120 pi^3), G_eff = g_D eps tan(theta_W) g/(4 cos(theta_W) m_Z^2)',
               tau_examples_s={f'eps={e:g},alpha_D=0.1,delta={d:g}': tau_chi2_s(e, 0.1, d) for e in (1e-6, 1e-5, 1e-4, 1e-3) for d in (250.0, 300.0, 365.0)},
               t_universe_s=T_UNIVERSE_S), open(f'{OUT}/relic_lifetime.json', 'w'), indent=1, default=float)

# ----------------------------------------------------------------------------------------------
# Part I: figures (palette: dataviz reference instance, fixed order)
# ----------------------------------------------------------------------------------------------
PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
INK, INK2, SURF = '#0b0b0b', '#52514e', '#fcfcfb'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'text.color': INK, 'axes.facecolor': SURF, 'figure.facecolor': SURF, 'axes.grid': True, 'grid.color': '#e6e5e2',
                     'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False})

# Fig 1: (delta, sigma_p) plane at 1000 GeV
m = 1000.0
sub = sel(m, 'annual'); subj = sel(m, 'june16'); subs = sel(m, 'sun')
fig, ax = plt.subplots(figsize=(7.2, 5.0))
ok = sub.N_p > 0
d = sub.delta_keV[ok].values; su = sigma_from_c(C_UNIT, m)
s1 = su / sub.N_p[ok].values
ax.fill_between(d, su * 0.105 / sub.N_p[ok].values, su * 3.65 / sub.N_p[ok].values, color=PAL[0], alpha=0.12, lw=0, label='0.105–3.65 events (90 % PLR band, one event)')
ax.fill_between(d, su * 0.3 / sub.N_p[ok].values, su * 3.0 / sub.N_p[ok].values, color=PAL[0], alpha=0.22, lw=0, label='0.3–3 events')
ax.plot(d, s1, color=PAL[0], lw=2, label='N = 1, annual-mean halo')
okj = subj.N_p > 0; ax.plot(subj.delta_keV[okj], su / subj.N_p[okj], color=PAL[0], lw=1.2, ls='--', label='N = 1, 16 June halo')
oks = subs.N_p > 0; ax.plot(subs.delta_keV[oks], su / subs.N_p[oks], color=PAL[0], lw=1.2, ls=':', label='N = 1, Sun-frame halo')
# Higgsino-equivalent line
okh = (sub.N_p > 0) & (sub.N_hig > 0)
ax.plot(sub.delta_keV[okh], su * sub.N_hig[okh] / sub.N_p[okh], color=PAL[1], lw=2, label='pure Higgsino (σ_n = 7.4e-39 cm², P007) as proton-only σ_p')
dH = summary_D[m]['higgsino_delta_N1']['annual']
ax.plot([dH], [np.interp(dH, sub.delta_keV[okh], su * sub.N_hig[okh] / sub.N_p[okh])], 'o', color=PAL[1], ms=7, mec=SURF, mew=1.5)
ax.annotate('Higgsino fits only here\nδ ≈ %.0f keV' % dH, (dH, np.interp(dH, sub.delta_keV[okh], su * sub.N_hig[okh] / sub.N_p[okh])),
            xytext=(-95, 28), textcoords='offset points', fontsize=8, color=INK2, arrowprops=dict(arrowstyle='-', color=INK2, lw=0.8))
# LZ digitised edges converted
bb = band[(band.halo == 'annual')]
for dd in sorted(bb.delta_keV.unique()):
    lo = float(bb[(bb.delta_keV == dd) & (bb.edge == 'lower')].sigma_p_edge_cm2.iloc[0]); hi = float(bb[(bb.delta_keV == dd) & (bb.edge == 'upper')].sigma_p_edge_cm2.iloc[0])
    ax.plot([dd, dd], [lo, hi], color=PAL[6], lw=2.5, solid_capstyle='butt', label='LZ Fig. 6/S7 two-sided 90 % interval, ×(A/Z)²_eff' if dd == 200 else None)
    ax.plot([dd - 3, dd + 3], [hi, hi], color=PAL[6], lw=1.2); ax.plot([dd - 3, dd + 3], [lo, lo], color=PAL[6], lw=1.2)
# light mediators (required sigma_p at q = 0)
for mA, c in ((0.3, PAL[2]), (0.1, PAL[3])):
    s = light[(light.m_A_GeV == mA) & (light.halo == 'annual')].sort_values('delta_keV'); okl = s.N_light_unit > 0
    ax.plot(s.delta_keV[okl], su / s.N_light_unit[okl], color=c, lw=1.6, label=f"N = 1, light A′ m_A′ = {mA:g} GeV (σ_p at q = 0)")
dmax = summary_D[m]['delta_max_248keV_june']
ax.axvline(dmax, color=INK2, lw=0.8, ls='-.'); ax.text(dmax - 2, 5e-37, 'δ_max(248 keV, 16 June) = %.0f keV' % dmax, rotation=90, va='top', ha='right', fontsize=7.5, color=INK2)
dlo = summary_D['delta_where_N_lo_per_hi_event_below'][m][3.0]
ax.axvspan(200, dlo, color='#e6e5e2', alpha=0.9, zorder=0); ax.text(201, 3e-37, '> 3 low-energy (5.4–55 keV)\nevents per 200–270 keV event\n(δ < %.0f keV)' % dlo, fontsize=7.5, color=INK2, va='top')
ax.set_yscale('log'); ax.set_xlim(200, 402); ax.set_ylim(1e-43, 1e-36)
ax.set_xlabel('mass splitting δ [keV]'); ax.set_ylabel('per-proton inelastic cross-section σ_p [cm²]')
ax.set_title('P011: dark-photon pseudo-Dirac DM, m_χ = 1000 GeV — σ_p giving one LZ event (2.84 t·yr)', fontsize=9.5)
ax.legend(fontsize=7, loc='lower right', framealpha=0.9)
fig.tight_layout(); fig.savefig(f'{FIG}/P011_fig1_sigma_p_plane_1000GeV.png', dpi=150); plt.close(fig)

# Fig 2: sigma_p(N=1) vs delta for three masses + Higgsino equivalents (small multiples share y)
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.8), sharey=True)
for ax, m, c in zip(axs, MASSES, PAL[:3]):
    sub = sel(m, 'annual'); su = sigma_from_c(C_UNIT, m); ok = sub.N_p > 0
    ax.fill_between(sub.delta_keV[ok], su * 0.3 / sub.N_p[ok], su * 3 / sub.N_p[ok], color=c, alpha=0.2, lw=0)
    ax.plot(sub.delta_keV[ok], su / sub.N_p[ok], color=c, lw=2, label='dark photon, N = 1 (0.3–3 band)')
    okh = ok & (sub.N_hig > 0)
    ax.plot(sub.delta_keV[okh], su * sub.N_hig[okh] / sub.N_p[okh], color=PAL[1], lw=1.6, label='Higgsino equivalent σ_p')
    dmax = summary_D[m]['delta_max_248keV_june']; ax.axvline(dmax, color=INK2, lw=0.8, ls='-.')
    dH = summary_D[m]['higgsino_delta_N1']['annual']
    if np.isfinite(dH):
        ax.plot([dH], [np.interp(dH, sub.delta_keV[okh], su * sub.N_hig[okh] / sub.N_p[okh])], 'o', color=PAL[1], ms=6, mec=SURF, mew=1.2)
    ax.set_yscale('log'); ax.set_xlim(200, 420); ax.set_ylim(1e-43, 1e-36); ax.set_title(f'm_χ = {m:.0f} GeV', fontsize=9)
    ax.set_xlabel('δ [keV]')
    ax.text(0.03, 0.95, 'δ_max(248 keV) = %.0f keV\nHiggsino δ(N=1) = %s' % (dmax, ('%.0f keV' % dH) if np.isfinite(dH) else 'none'), transform=ax.transAxes, va='top', fontsize=7.5, color=INK2)
axs[0].set_ylabel('σ_p for one LZ event [cm²]'); axs[0].legend(fontsize=7, loc='lower right')
fig.suptitle('P011: required proton cross-section vs δ (annual-mean SHM); Higgsino line from the gauge-fixed Z coupling', fontsize=9.5)
fig.tight_layout(); fig.savefig(f'{FIG}/P011_fig2_sigma_p_three_masses.png', dpi=150); plt.close(fig)

# Fig 3: required eps vs m_A' (1000 GeV) for several delta, alpha_D = 0.1, with the recalled BaBar line and the chi2-lifetime floor
fig, ax = plt.subplots(figsize=(7.0, 4.6))
mA_grid = np.logspace(-1.3, 1.3, 120)
sub = sel(1000.0, 'annual'); su = sigma_from_c(C_UNIT, 1000.0)
for i, dd in enumerate((250.0, 300.0, 350.0, 365.0, 380.0)):
    Np = float(sub[np.isclose(sub.delta_keV, dd)].N_p.iloc[0]); X = eps2alphaD_over_mA4(su / Np, 1000.0)
    Sg = np.array([S_factor(x, dd) for x in mA_grid])
    eps = np.sqrt(X * mA_grid ** 4 / 0.1 / Sg)
    ax.plot(mA_grid, eps, color=PAL[i], lw=2, label=f'δ = {dd:.0f} keV')
    ax.text(mA_grid[-1] * 1.05, eps[-1], f'{dd:.0f}', fontsize=7.5, color=INK2, va='center')
ax.axhline(EPS_BABAR, color=INK2, lw=1.2, ls='--'); ax.text(0.055, 1.25e-3, 'BaBar visible/invisible A′ (recalled, likely): ε ≲ 10⁻³', fontsize=7.5, color=INK2)
fl = life[np.isclose(life.alpha_D, 0.1)].eps_min_for_no_exothermic
ax.axhspan(fl.min(), fl.max(), color=PAL[7], alpha=0.25, lw=0)
ax.text(0.055, fl.min() * 0.55, 'χ₂ must decay before today (≤ 1 exothermic χ₂N → χ₁N event): ε ≳ (%.0f–%.0f)×10⁻⁶ (δ = 380–250 keV)' % (fl.min() * 1e6, fl.max() * 1e6), fontsize=7.5, color=PAL[7])
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(0.05, 22); ax.set_ylim(1e-9, 3e-2)
ax.set_xlabel("dark-photon mass m_A′ [GeV]"); ax.set_ylabel('kinetic mixing ε required for one LZ event (α_D = 0.1)')
ax.set_title('P011: m_χ = 1000 GeV — ε(m_A′) for N = 1; propagator floor below m_A′ ≈ q ≈ 0.25 GeV', fontsize=9.5)
ax.legend(fontsize=7.5, loc='lower right', title='splitting', title_fontsize=7.5)
fig.tight_layout(); fig.savefig(f'{FIG}/P011_fig3_epsilon_vs_mA.png', dpi=150); plt.close(fig)

# Fig 4: spectra at 1000 GeV for delta = 300 (p-only vs isoscalar vs Higgsino shapes) and low-energy companions
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
ax = axs[0]
for i, dd in enumerate(('250', '300', '350', '380')):
    sp = spectra_store[dd]; E = np.array(sp['E'])
    r = np.array(sp['p_annual']); r = r / r.max()
    ax.plot(E, r, color=PAL[i], lw=1.8, label=f'δ = {dd} keV, proton-only')
    ri = np.array(sp['iso_annual']); ax.plot(E, ri / ri.max(), color=PAL[i], lw=1.0, ls='--')
ax.axvspan(269.9, 330, color='#f0efec'); ax.axvline(248, color=INK2, lw=0.8)
ax.set_xlim(0, 330); ax.set_ylim(0, 1.15); ax.set_xlabel('E_R [keV]'); ax.set_ylabel('dR/dE (peak-normalised)')
ax.set_title('1000 GeV spectra: proton-only (solid) vs isoscalar (dashed)', fontsize=9); ax.legend(fontsize=7)
ax = axs[1]
for m, c in zip(MASSES, PAL[:3]):
    sub = sel(m, 'annual'); ok = sub.N_p > 0
    okh = ok & (sub.N_p_hi > 0)
    ax.plot(sub.delta_keV[okh], sub.N_p_lo[okh] / sub.N_p_hi[okh], color=c, lw=2, label=f'{m:.0f} GeV')
ax.axhline(3, color=INK2, lw=0.8, ls=':'); ax.text(240, 3.6, 'N_max ≈ 3 (P003 tolerance, recalled)', fontsize=7.5, color=INK2)
ax.set_yscale('log'); ax.set_xlim(200, 300); ax.set_ylim(1e-3, 30); ax.set_xlabel('δ [keV]'); ax.set_ylabel('5.4–55 keV events per 200–270 keV event')
ax.set_title('Low-energy companions (proton-only O1)', fontsize=9); ax.legend(fontsize=7.5)
fig.tight_layout(); fig.savefig(f'{FIG}/P011_fig4_spectra_and_companions.png', dpi=150); plt.close(fig)

# ----------------------------------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------------------------------
summary = dict(partA=partA, validation=val, partD=summary_D, relic=relic,
               eps_1000GeV_alphaD0p1_annual=eps_df[(eps_df.m_GeV == 1000.0) & (eps_df.alpha_D_label == 'alpha_D=0.1')][['delta_keV', 'm_A_GeV', 'S_propagator', 'eps_required']].to_dict('records'),
               eps_1000GeV_thermal_annual=eps_df[(eps_df.m_GeV == 1000.0) & (eps_df.alpha_D_label == 'alpha_D=thermal')][['delta_keV', 'm_A_GeV', 'alpha_D', 'eps_required']].to_dict('records'),
               chi2=life.to_dict('records'), exothermic=exo[exo.halo == 'annual'].to_dict('records'), mA_window=win.to_dict('records'),
               light_S_annual={mA: {int(d): float(light[(light.m_A_GeV == mA) & (light.halo == 'annual') & np.isclose(light.delta_keV, d)].S_light_over_heavy.iloc[0]) for d in (250, 300, 350, 380)} for mA in MA_LIGHT},
               settings=dict(E_step_keV=E_STEP, E_max_keV=E_MAX, delta_step_keV=5.0, vgrid=[0.0, 844.0, 1200], eff_sigma_hi_keV=EFF_SIGS, exposure_tyr=EXPOSURE),
               runtime_s=time.time() - T0)
json.dump(summary, open(f'{OUT}/summary.json', 'w'), indent=1, default=float)
print(f'done in {time.time() - T0:.0f} s')
