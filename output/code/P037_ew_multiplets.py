#!/usr/bin/env python
"""P037 - Which electroweak multiplets can be inelastic dark matter with a 100-400 keV neutral splitting?

Run from the simulation root:  .venv/bin/python output/code/P037_ew_multiplets.py

Parts
  A  group theory: SU(2)_L x U(1)_Y multiplets (n, Y) with a neutral component, Z vector coupling of the
     neutral Dirac state (prop. to Y), per-nucleon inelastic cross-section normalised to the Higgsino (P007)
  B  Majorana-splitting operators: lowest-dimension operator (chi chi H^dagger^{4Y})/Lambda^{4Y-1};
     Lambda for delta = 100/300/366/380 keV; gaugino-mixing cross-check (P007); Planck / EFT-consistency
  C  expected LZ events N_Y(delta) = (2Y)^2 N_Higgsino(delta) from P007's WimPyDD grid; delta(N = 3.65, 1, 0.105)
     for Y = 1/2, 1, 3/2, 2 and m = 300-4000 GeV; "inelastic evasion window"; live WimPyDD cross-check
  D  one-loop charged-neutral splittings for every multiplet (Cirelli-Fornengo-Strumia form, P014 method)
  E  summary table, figures, JSON
All outputs -> output/work/P037/ (figures in output/work/P037/figures/).
"""
import sys, os, math, json, time
sys.path.insert(0, 'output/code')
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import erf, erfc
from common import lzcommon as lz

T0 = time.time()
OUT = 'output/work/P037'
FIG = f'{OUT}/figures'
os.makedirs(FIG, exist_ok=True)
LOG = open(f'{OUT}/run_log.txt', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s); LOG.write(s + '\n'); LOG.flush()

# ---------------------------------------------------------------- constants (recalled, certain unless noted)
GF = 1.166e-5          # GeV^-2
S2W = 0.231            # sin^2 theta_W (MSbar at m_Z)
MZ, MW = 91.19, 80.37  # GeV
ALPHA_MZ = 1 / 127.95  # alpha(m_Z)
ALPHA2 = ALPHA_MZ / S2W          # g^2/4pi = 0.0338
GP2 = 4 * math.pi * ALPHA_MZ / (1 - S2W)   # g'^2 = e^2/c_W^2
V = 246.2              # GeV (lz.M_V_GEV)
MPL = 1.22e19          # GeV (Planck mass, recalled certain)
GEV2_TO_CM2 = 0.3894e-27
MN = lz.M_NUCLEON_GEV
Z_XE, A_XE = 54, lz.A_XE_MEAN
SIGMA_SI_2024 = 5e-47  # cm^2, LZ 2024 SI limit at 1 TeV (recalled: likely, as in P007)
N_UPPER90 = 3.65       # one observed event: PLR 90% two-sided upper edge in signal events (P007/P021)
N_LOWER90 = 0.105

# ---------------------------------------------------------------- Part A: multiplets
P('=== Part A: multiplets, Z couplings, cross-sections ===')
MULTIPLETS = [  # (n, Y, label, thermal mass TeV (recalled), reliability)
    (2, 0.5, 'doublet Y=1/2 (Higgsino-like)', 1.1, 'certain'),
    (3, 0.0, 'triplet Y=0 (wino-like)', 2.8, 'likely (2.7-3.0)'),
    (3, 1.0, 'triplet Y=1', 2.0, 'uncertain'),
    (4, 0.5, 'quadruplet Y=1/2', 2.4, 'uncertain'),
    (4, 1.5, 'quadruplet Y=3/2', 2.4, 'uncertain'),
    (5, 0.0, 'quintuplet Y=0 (MDM)', 9.4, 'likely'),
    (5, 1.0, 'quintuplet Y=1', 4.5, 'uncertain'),
    (5, 2.0, 'quintuplet Y=2', 4.5, 'uncertain'),
]

def mu_red(m, mt=MN):
    return m * mt / (m + mt)

def sigma_n_vector(Y, m):
    """per-neutron inelastic cross-section for the neutral state of a (n,Y) multiplet via Z exchange,
    sigma_n = G_F^2 mu^2/(2 pi) * (2Y)^2   [Higgsino: Y = 1/2]"""
    return GF**2 * mu_red(m)**2 / (2 * math.pi) * (2 * Y)**2 * GEV2_TO_CM2

VEC_FACTOR = (A_XE / ((A_XE - Z_XE) - (1 - 4 * S2W) * Z_XE))**2    # P007: 3.214
rowsA = []
for n, Y, lab, mth, rel in MULTIPLETS:
    j = (n - 1) / 2
    T3 = np.arange(-j, j + 1)
    Q = T3 + Y
    neutral = np.isclose(Q, 0).any()
    gZ_over_g_cW = -Y                        # (T3 - Q s_W^2) at Q = 0, T3 = -Y
    sig_n = sigma_n_vector(Y, 1000.0)
    sig_si_eq = sig_n / VEC_FACTOR
    nH = int(round(4 * Y))                   # Higgs fields in the lowest Majorana-splitting operator
    dim = 3 + nH
    rowsA.append(dict(n=n, Y=Y, label=lab, charges=' '.join(f'{q:+.0f}' for q in Q), has_neutral=neutral,
                      gZ_neutral_units_g_over_cW=gZ_over_g_cW, coupling_sq_rel_higgsino=(2 * Y)**2,
                      sigma_n_cm2_1TeV=sig_n, sigma_SI_eq_cm2_1TeV=sig_si_eq,
                      elastic_excess_vs_LZ2024=sig_si_eq / SIGMA_SI_2024 if Y else 0.0,
                      n_Higgs_in_splitting_op=nH, operator_dimension=dim,
                      thermal_mass_TeV_recalled=mth, thermal_mass_reliability=rel))
    P(f'  ({n},{Y:.1f}) {lab:32s} Q={rowsA[-1]["charges"]:>14s}  gZ/(g/cW)={gZ_over_g_cW:+.1f}  (2Y)^2={(2*Y)**2:.0f} '
      f'sigma_n={sig_n:.2e} cm2  sigma_SI,eq={sig_si_eq:.2e}  dim={dim}')
dfA = pd.DataFrame(rowsA)
dfA.to_csv(f'{OUT}/multiplets.csv', index=False)
P(f'  Higgsino check: sigma_n(1 TeV) = {sigma_n_vector(0.5, 1000):.3e} cm2 (P007: 7.40e-39); vector factor {VEC_FACTOR:.3f} (P007: 3.214)')
P(f'  Xe nuclear-level check (q->0, A=131): sigma_A = {GF**2*mu_red(1000, 131*lz.AMU_GEV)**2/(2*math.pi)*((1-4*S2W)*54-77)**2*GEV2_TO_CM2:.2e} cm2 (P007: 5.30e-31)')

# ---------------------------------------------------------------- Part B: splitting operators
P('\n=== Part B: Majorana splitting operators and Lambda ===')
def Lambda_GeV(Y, delta_keV, vev=V):
    """delta = vev^(4Y) / Lambda^(4Y-1)  ->  Lambda = (vev^(4Y)/delta)^(1/(4Y-1))"""
    d = delta_keV * 1e-6
    k = 4 * Y
    return (vev**k / d)**(1.0 / (k - 1))

def delta_from_Lambda_keV(Y, Lam, vev=V):
    k = 4 * Y
    return vev**k / Lam**(k - 1) * 1e6

DELTAS_B = [100.0, 200.0, 300.0, 366.0, 380.0]
rowsB = []
for Y in (0.5, 1.0, 1.5, 2.0):
    for d in DELTAS_B:
        L = Lambda_GeV(Y, d)
        L_half = Lambda_GeV(Y, d, vev=V / math.sqrt(2))     # <H> = v/sqrt2 convention
        rowsB.append(dict(Y=Y, operator_dimension=int(3 + 4 * Y), delta_keV=d, Lambda_TeV=L / 1e3,
                          Lambda_TeV_vev_over_sqrt2=L_half / 1e3, Lambda_over_1TeV=L / 1e3,
                          delta_if_Lambda_is_MPl_keV=delta_from_Lambda_keV(Y, MPL),
                          delta_if_Lambda_1TeV_keV=delta_from_Lambda_keV(Y, 1e3),
                          delta_if_Lambda_10TeV_keV=delta_from_Lambda_keV(Y, 1e4)))
dfB = pd.DataFrame(rowsB)
dfB.to_csv(f'{OUT}/lambda_table.csv', index=False)
for Y in (0.5, 1.0, 1.5, 2.0):
    s = dfB[dfB.Y == Y]
    P(f'  Y={Y}: dim-{int(3+4*Y)} op; Lambda(100/300/366/380 keV) = ' +
      ' / '.join(f'{x:.3g}' for x in s.Lambda_TeV.values[[0, 2, 3, 4]]) + ' TeV; '
      f'(<H>=v/sqrt2: x{s.Lambda_TeV_vev_over_sqrt2.values[2]/s.Lambda_TeV.values[2]:.2f}); '
      f'delta(Lambda=M_Pl) = {s.delta_if_Lambda_is_MPl_keV.values[0]:.2e} keV; delta(Lambda=1 TeV) = {s.delta_if_Lambda_1TeV_keV.values[0]:.3g} keV; '
      f'delta(10 TeV) = {s.delta_if_Lambda_10TeV_keV.values[0]:.3g} keV')

# gaugino-mixing cross-check (P007): delta = m_Z^2 (s_W^2/M1 + c_W^2/M2) = g'^2 v^2/(4 M1) + g^2 v^2/(4 M2)
g2 = 4 * math.pi * ALPHA2
P(f'  m_Z^2 check: sqrt((g^2+g\'^2) v^2/4) = {math.sqrt((g2 + GP2) * V**2 / 4):.2f} GeV (91.19)')
gaug = pd.read_csv('output/work/P007/gaugino_masses_for_splitting.csv')
sel = gaug[(gaug.mu_GeV == 1000) & (gaug.tanb == 10)]
rowsB2 = []
for _, r in sel.iterrows():
    if 'bino_only' in r.hierarchy:
        Lam_eff = 4 * r.M1_TeV / GP2               # TeV
        which = 'bino: Lambda_eff = 4 M1/g\'^2'
    elif 'wino_only' in r.hierarchy:
        Lam_eff = 4 * r.M2_TeV / g2
        which = 'wino: Lambda_eff = 4 M2/g^2'
    else:
        continue
    L_op = Lambda_GeV(0.5, r.delta_target_keV) / 1e3
    rowsB2.append(dict(hierarchy=r.hierarchy, delta_keV=r.delta_target_keV, M1_TeV=r.M1_TeV, M2_TeV=r.M2_TeV,
                       Lambda_eff_TeV=Lam_eff, Lambda_v2_over_delta_TeV=L_op, ratio=Lam_eff / L_op))
    P(f'  {which}: delta={r.delta_target_keV:.0f} keV, M={r.M1_TeV if "bino" in r.hierarchy else r.M2_TeV:.0f} TeV -> '
      f'Lambda_eff={Lam_eff:.3g} TeV vs v^2/delta={L_op:.3g} TeV (ratio {Lam_eff/L_op:.3f})')
pd.DataFrame(rowsB2).to_csv(f'{OUT}/gaugino_lambda_crosscheck.csv', index=False)

# ---------------------------------------------------------------- Part C: LZ counts scaled from P007
P('\n=== Part C: expected LZ events for Y-scaled Z couplings (P007 WimPyDD grid x (2Y)^2) ===')
grid = pd.read_csv('output/work/P007/N_events_grid.csv')
MASSES = [300.0, 500.0, 1000.0, 2000.0, 4000.0]
YS = [0.5, 1.0, 1.5, 2.0]
V_JUNE = lz.vmax_kms(lz.v_earth_kms(167))

def curve(m, halo='annual', sig=11.5):
    s = grid[(grid.m_GeV == m) & (grid.halo == halo) & (grid.eff_sigma_keV == sig)].sort_values('delta_keV')
    return s.delta_keV.values, s.N_events.values

def delta_for_N(d, N, target):
    """delta on the falling branch where N(delta) = target (log-linear interpolation); nan if never reached."""
    N = np.maximum(N, 1e-12)
    i0 = int(np.argmax(N))
    dd, NN = d[i0:], N[i0:]
    if NN[0] < target:
        return float('nan')
    for i in range(len(dd) - 1):
        if NN[i] >= target > NN[i + 1]:
            f = (math.log(NN[i]) - math.log(target)) / (math.log(NN[i]) - math.log(NN[i + 1]))
            return float(dd[i] + f * (dd[i + 1] - dd[i]))
    return float('nan')

rowsC = []
for m in MASSES:
    dmax248 = lz.delta_max_kev(248.0, m, v_kms=V_JUNE)
    dmax270 = lz.delta_max_kev(269.9, m, v_kms=V_JUNE)
    for Y in YS:
        f = (2 * Y)**2
        rec = dict(m_GeV=m, Y=Y, coupling_sq_rel=f, delta_max_248_june=dmax248, delta_max_270_june=dmax270)
        for halo in ('annual', 'sun', 'june16'):
            d, N = curve(m, halo)
            N = N * f
            rec[f'delta_N3.65_{halo}'] = delta_for_N(d, N, N_UPPER90)
            rec[f'delta_N1_{halo}'] = delta_for_N(d, N, 1.0)
            rec[f'delta_N0.105_{halo}'] = delta_for_N(d, N, N_LOWER90)
            if halo == 'annual':
                for dv in (0.0, 100.0, 200.0, 250.0, 300.0, 350.0):
                    rec[f'N_at_{dv:.0f}'] = float(N[np.argmin(abs(d - dv))])
        for vh in ('annual_vesc528', 'annual_vesc560'):
            d, N = curve(m, vh)
            rec[f'delta_N1_{vh}'] = delta_for_N(d, N * f, 1.0)
        rec['window_lo'] = rec['delta_N3.65_annual']
        rec['window_hi'] = dmax248
        rec['window_width_keV'] = dmax248 - rec['delta_N3.65_annual']
        rowsC.append(rec)
dfC = pd.DataFrame(rowsC)
dfC.to_csv(f'{OUT}/required_delta_by_Y.csv', index=False)
for m in MASSES:
    s = dfC[dfC.m_GeV == m]
    P(f'  m={m:.0f} GeV, delta_max(248 keV, June)={s.delta_max_248_june.values[0]:.1f} keV:')
    for _, r in s.iterrows():
        P(f'    Y={r.Y:.1f} (x{r.coupling_sq_rel:.0f}): delta(N=3.65/1/0.105) = {r["delta_N3.65_annual"]:.1f}/{r.delta_N1_annual:.1f}/{r["delta_N0.105_annual"]:.1f} keV '
          f'[sun {r.delta_N1_sun:.1f}, june {r.delta_N1_june16:.1f}, vesc528/560 {r.delta_N1_annual_vesc528:.1f}/{r.delta_N1_annual_vesc560:.1f}]; '
          f'window [{r.window_lo:.1f}, {r.window_hi:.1f}] = {r.window_width_keV:.1f} keV; N(0/200/300/350) = {r.N_at_0:.2e}/{r.N_at_200:.2e}/{r.N_at_300:.3g}/{r.N_at_350:.3g}')

# exclusion factors against LZ's digitised 90% upper edges (Fig. 6, 1 TeV) in coupling and events
lzint = pd.read_csv('output/work/P007/lz_intervals_digitised.csv')
C_EQ_H = 0.0777   # (c_eq^s m_v^2)^2 of the Higgsino (P007)
d1, N1 = curve(1000.0)
N_unit_1TeV = {250: 1.04e5, 300: 1.35e4, 350: 259.0}    # P007 isoscalar unit-coupling counts (annual)
rowsX = []
for _, r in lzint.iterrows():
    if not np.isfinite(r.c1s_mv2_sq_upper) or r.delta_keV < 100:
        continue
    for Y in YS:
        f = (2 * Y)**2
        rec = dict(delta_keV=r.delta_keV, Y=Y, coupling_ratio_to_upper=C_EQ_H * f / r.c1s_mv2_sq_upper,
                   N_model_annual=float(N1[np.argmin(abs(d1 - r.delta_keV))]) * f)
        if int(r.delta_keV) in N_unit_1TeV:
            rec['N_at_LZ_upper_edge'] = N_unit_1TeV[int(r.delta_keV)] * r.c1s_mv2_sq_upper
            rec['event_ratio'] = rec['N_model_annual'] / rec['N_at_LZ_upper_edge']
        rowsX.append(rec)
dfX = pd.DataFrame(rowsX)
dfX.to_csv(f'{OUT}/exclusion_vs_LZ_edges_1TeV.csv', index=False)
P('  Coupling excess over LZ Fig. 6 upper edge (1 TeV), (2Y)^2 x 0.0777 / upper:')
for dv in (100, 200, 250, 300, 350):
    s = dfX[dfX.delta_keV == dv]
    P(f'    delta={dv}: ' + ', '.join(f'Y={r.Y}: x{r.coupling_ratio_to_upper:.3g}' for _, r in s.iterrows()) +
      (f'; events x{s.event_ratio.values[0]:.3g}(Y=1/2) ... x{s.event_ratio.values[-1]:.3g}(Y=2)' if 'event_ratio' in s and s.event_ratio.notna().all() else ''))

# ---- live WimPyDD cross-check: recompute N(1 TeV; 300, 370 keV) in the Sun frame for Y=1/2 and Y=1
P('  WimPyDD cross-check (Sun-frame halo, 1 TeV):')
WD = lz.wd()
VGRID = np.linspace(0.0, 830.0, 1661)
halo_sun = lz.wd_halo(vmin=VGRID)
E50_HI, E50_LO, PLATEAU = lz.LZ['E_50pct_high_keV'], lz.LZ['E_50pct_low_keV'], lz.LZ['eff_plateau']
def efficiency(E, sig_hi=11.5, sig_lo=2.5):
    return PLATEAU * 0.5 * (1 + erf((E - E50_LO) / (math.sqrt(2) * sig_lo))) * 0.5 * erfc((E - E50_HI) / (math.sqrt(2) * sig_hi))
cp = GF / math.sqrt(2) * (1 - 4 * S2W); cn = -GF / math.sqrt(2)
c0_wd, c1_wd = cp + cn, cp - cn          # WimPyDD convention (P003/P007)
sun_ref = grid[(grid.m_GeV == 1000) & (grid.halo == 'sun') & (grid.eff_sigma_keV == 11.5)]
wchk = []
for Y in (0.5, 1.0):
    ham = lz.wd_hamiltonian(f'Zvec_Y{Y}', {1: (2 * Y * c0_wd, 2 * Y * c1_wd)})
    for dv in (300.0, 370.0):
        lo = lz.E_R_range_keV(1000.0, VGRID[-1], A=124.0, delta_kev=dv)[0]
        E = np.arange(max(1.0, math.floor(lo) - 4.0), 330.0 + 1e-9, 2.0)
        r = lz.wd_rate(ham, 1000.0, E, halo=halo_sun, delta_kev=dv)
        N = float(np.trapezoid(r * efficiency(E), E)) * lz.LZ['exposure_tyr']
        ref = float(sun_ref[sun_ref.delta_keV == dv].N_events.values[0]) * (2 * Y)**2
        wchk.append(dict(Y=Y, delta_keV=dv, N_wimpydd=N, N_from_P007_grid_scaled=ref, ratio=N / ref))
        P(f'    Y={Y}, delta={dv:.0f}: N = {N:.4g} vs (2Y)^2 x P007 = {ref:.4g}  (ratio {N/ref:.4f})')
pd.DataFrame(wchk).to_csv(f'{OUT}/wimpydd_crosscheck.csv', index=False)

# ---------------------------------------------------------------- Part D: radiative charged-neutral splittings
P('\n=== Part D: one-loop charged-neutral splittings (Cirelli-Fornengo-Strumia form; P014 method) ===')
def f_tilde(r):
    """f(r) - f(0) with f(r) = 2 int_0^1 dx (1+x) ln[x^2 + (1-x) r^2]; f_tilde -> 2 pi r as r -> 0."""
    val, _ = quad(lambda x: 2 * (1 + x) * math.log(x**2 + (1 - x) * r**2), 0, 1, limit=200)
    return val + 5.0

def dm_rad_MeV(M, Q, Qp, Y):
    fz, fw = f_tilde(MZ / M), f_tilde(MW / M)
    return ALPHA2 * M / (4 * math.pi) * ((Q**2 - Qp**2) * S2W * fz + (Q - Qp) * (Q + Qp - 2 * Y) * (fw - fz)) * 1e3

def dm_asymptotic_MeV(Q, Qp, Y):
    return ALPHA2 / 2 * ((Q**2 - Qp**2) * S2W * MZ + (Q - Qp) * (Q + Qp - 2 * Y) * (MW - MZ)) * 1e3

P(f'  checks: f~(r)/(2 pi r) at r=1e-3, 0.1 = {f_tilde(1e-3)/(2*math.pi*1e-3):.4f}, {f_tilde(0.1)/(2*math.pi*0.1):.4f}; '
  f'Higgsino 1 TeV {dm_rad_MeV(1000, 1, 0, 0.5):.1f} MeV (P014 341.8), asymptote {dm_asymptotic_MeV(1,0,0.5):.1f} (356.3); '
  f'wino 1 TeV {dm_rad_MeV(1000, 1, 0, 0.0):.1f} MeV (P014 on-shell 160.7 / MSbar 172.8)')
rowsD = []
for n, Y, lab, mth, rel in MULTIPLETS:
    j = (n - 1) / 2
    for T3 in np.arange(-j, j + 1):
        Q = T3 + Y
        if np.isclose(Q, 0):
            continue
        rec = dict(n=n, Y=Y, label=lab, Q=int(round(Q)))
        for M in (1000.0, 2000.0, 4000.0):
            rec[f'dm_MeV_{M:.0f}'] = dm_rad_MeV(M, Q, 0, Y)
        rec['dm_MeV_asymptotic'] = dm_asymptotic_MeV(Q, 0, Y)
        # crude chargino lifetime scaling for |Q| = 1 -> neutral: ctau ~ 0.712 cm (342/dm)^3 / kappa^2  [estimated]
        rec['lighter_than_neutral_1TeV'] = bool(rec['dm_MeV_1000'] < 0)
        rec['lighter_than_neutral_asymptotic'] = bool(rec['dm_MeV_asymptotic'] < 0)
        if abs(Q) == 1:
            kappa2 = (j - (-Y)) * (j + (-Y) + 1) if Q == 1 else (j + (-Y)) * (j - (-Y) + 1)   # |<T3+-1|T-+|T3=-Y>|^2
            rec['kappa2_W_coupling'] = kappa2
            rec['ctau_cm_scaled_1TeV'] = (0.712 * (341.8 / rec['dm_MeV_1000'])**3 / kappa2) if rec['dm_MeV_1000'] > 150 else float('nan')
        rowsD.append(rec)
dfD = pd.DataFrame(rowsD)
dfD.to_csv(f'{OUT}/radiative_splittings.csv', index=False)
for _, r in dfD.iterrows():
    extra = f"; kappa^2={r.kappa2_W_coupling:.0f}, ctau~{r.ctau_cm_scaled_1TeV:.2g} cm" if abs(r.Q) == 1 else ''
    flag = '  <-- LIGHTER than the neutral state (one loop)' if r.dm_MeV_asymptotic < 0 or r.dm_MeV_1000 < 0 else ''
    P(f'  ({r.n},{r.Y:.1f}) Q={r.Q:+d}: dm(1/2/4 TeV) = {r.dm_MeV_1000:.0f}/{r.dm_MeV_2000:.0f}/{r.dm_MeV_4000:.0f} MeV, asympt {r.dm_MeV_asymptotic:.0f}{extra}{flag}')
# mass at which the (4,1/2) Q=-1 state crosses the neutral one
from scipy.optimize import brentq
M_cross = brentq(lambda M: dm_rad_MeV(M, -1, 0, 0.5), 300.0, 20000.0)
P(f'  (4,1/2): Q=-1 becomes lighter than Q=0 for M > {M_cross:.0f} GeV (one loop, alpha(m_Z), s_W^2 = 0.231)')
CHARGED_LSP = {(4, 0.5): f'Q=-1 lighter than neutral for M > {M_cross/1e3:.1f} TeV (|dm| <= 10 MeV): needs tree-level lift',
               (5, 1.0): f'Q=-1 lighter than neutral by {abs(dm_rad_MeV(1000,-1,0,1.0)):.0f}-{abs(dm_asymptotic_MeV(-1,0,1.0)):.0f} MeV: needs tree-level lift'}

# ---------------------------------------------------------------- Part E: summary
P('\n=== Part E: summary table ===')
def pick(m, Y, col):
    return float(dfC[(dfC.m_GeV == m) & (dfC.Y == Y)][col].values[0])
rowsE = []
for n, Y, lab, mth, rel in MULTIPLETS:
    rec = dict(multiplet=lab, n=n, Y=Y, Z_coupling_rel_higgsino=2 * Y, thermal_mass_TeV=mth, thermal_rel=rel)
    if Y == 0:
        rec.update(sigma_n_cm2=0.0, operator_dimension=None, Lambda_300keV_TeV=None, delta_N1_1TeV=None,
                   window_1TeV='none (no Z coupling)', verdict='not an LZ scatterer without a new mediator (P014)')
    else:
        m_proxy = min(4000.0, max(300.0, mth * 1e3))
        m_grid = min(MASSES, key=lambda x: abs(math.log(x / m_proxy)))
        rec.update(sigma_n_cm2=sigma_n_vector(Y, 1000.0), operator_dimension=int(3 + 4 * Y),
                   Lambda_300keV_TeV=Lambda_GeV(Y, 300.0) / 1e3, Lambda_366keV_TeV=Lambda_GeV(Y, 366.0) / 1e3,
                   delta_N3p65_1TeV=pick(1000, Y, 'delta_N3.65_annual'), delta_N1_1TeV=pick(1000, Y, 'delta_N1_annual'),
                   delta_N0p105_1TeV=pick(1000, Y, 'delta_N0.105_annual'),
                   window_1TeV=f"[{pick(1000, Y, 'window_lo'):.0f}, {pick(1000, Y, 'window_hi'):.0f}] keV",
                   m_grid_for_thermal_GeV=m_grid, delta_N1_at_thermal_proxy=pick(m_grid, Y, 'delta_N1_annual'),
                   window_at_thermal_proxy=f"[{pick(m_grid, Y, 'window_lo'):.0f}, {pick(m_grid, Y, 'window_hi'):.0f}] keV",
                   dm_charged_MeV_1TeV=float(dfD[(dfD.n == n) & (dfD.Y == Y) & (dfD.Q == 1)].dm_MeV_1000.values[0]))
        Lam = Lambda_GeV(Y, 366.0)
        if Lam > 1e7:
            verdict = 'viable; splitting from PeV-scale physics (gaugino mixing)'
        elif Lam > mth * 1e3:
            verdict = f'viable; needs a mediator at {Lam/1e3:.0f} TeV (> M_DM)'
        else:
            verdict = f'EFT inconsistent: Lambda = {Lam/1e3:.1f} TeV < thermal mass; needs light mediator or mixing'
        if (n, Y) in CHARGED_LSP:
            verdict += '; ' + CHARGED_LSP[(n, Y)]
        rec['verdict'] = verdict
    rowsE.append(rec)
dfE = pd.DataFrame(rowsE)
dfE.to_csv(f'{OUT}/summary_table.csv', index=False)
P(dfE.to_string())

# ---------------------------------------------------------------- figures
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
COL = {300.0: '#eda100', 500.0: '#e87ba4', 1000.0: '#2a78d6', 2000.0: '#eb6834', 4000.0: '#1baf7a'}
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': '#0b0b0b',
                     'xtick.color': '#52514e', 'ytick.color': '#52514e', 'grid.color': '#e1e0d9', 'axes.grid': True,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})

# Fig 1: required delta vs Y
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.axhspan(360, 385, color='#e1e0d9', alpha=0.6, lw=0, label='P021 68% shape-preferred (1 TeV)')
xs = np.array(YS)
for k, m in enumerate([500.0, 1000.0, 2000.0, 4000.0]):
    s = dfC[dfC.m_GeV == m].sort_values('Y')
    off = (k - 1.5) * 0.045
    ax.vlines(xs + off, s['delta_N3.65_annual'], s['delta_N0.105_annual'], color=COL[m], lw=2)
    ax.plot(xs + off, s.delta_N1_annual, 'o', ms=6, color=COL[m], mec='#fcfcfb', mew=1, label=f'{m:.0f} GeV')
    ax.hlines(s.delta_max_248_june.values[0], xs[0] - 0.15, xs[-1] + 0.15, color=COL[m], lw=1, ls='--')
    ax.text(xs[-1] + 0.17, s.delta_max_248_june.values[0], f'δ_max(248 keV) {m:.0f}', color=COL[m], fontsize=7.5, va='center')
ax.set_xticks(xs); ax.set_xticklabels(['Y = 1/2\n(n = 2, 4)', 'Y = 1\n(n = 3, 5)', 'Y = 3/2\n(n = 4)', 'Y = 2\n(n = 5)'])
ax.set_ylabel('splitting δ giving one LZ event [keV]')
ax.set_xlim(0.3, 2.6); ax.set_ylim(320, 415)
ax.set_title('δ for N = 1 (dots) and the 90% one-event band (N = 3.65–0.105) vs hypercharge', fontsize=10, loc='left')
ax.legend(fontsize=8, loc='lower right', frameon=False, ncol=2)
fig.tight_layout(); fig.savefig(f'{FIG}/P037_fig1_delta_vs_Y.png', dpi=160); plt.close(fig)

# Fig 2: Lambda vs delta
fig, ax = plt.subplots(figsize=(7.2, 4.6))
dd = np.logspace(1, 3, 200)
YCOL = {0.5: '#2a78d6', 1.0: '#eb6834', 1.5: '#1baf7a', 2.0: '#4a3aa7'}
for Y in YS:
    ax.plot(dd, Lambda_GeV(Y, dd) / 1e3, color=YCOL[Y], lw=2, label=f'Y = {Y:g}: dim-{int(3+4*Y)} op, δ = v^{int(4*Y)}/Λ^{int(4*Y-1)}')
lo, hi = pick(1000, 0.5, 'window_lo'), pick(1000, 0.5, 'window_hi')
ax.axvspan(lo, hi, color='#e1e0d9', alpha=0.7, lw=0, label=f'LZ window, 1 TeV ({lo:.0f}–{hi:.0f} keV)')
ax.axhline(MPL / 1e3, color='#898781', lw=1, ls=':'); ax.text(11, MPL / 1e3 * 1.6, 'M_Planck', color='#898781', fontsize=8)
ax.axhline(1.0, color='#898781', lw=1, ls=':'); ax.text(11, 0.85, 'M_DM ~ 1 TeV (EFT validity: Λ > M_DM)', color='#898781', fontsize=8, va='top')
bino = [r for r in rowsB2 if 'bino' in r['hierarchy'] and r['delta_keV'] == 370][0]
ax.plot(370, bino['Lambda_eff_TeV'], 's', color='#2a78d6', ms=7, mec='#fcfcfb', label=f"P007 bino mixing, M₁ = {bino['M1_TeV']/1e3:.1f} PeV → Λ = 4M₁/g'²")
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(10, 1000); ax.set_ylim(0.2, 1e17)
ax.set_xlabel('neutral Majorana splitting δ [keV]'); ax.set_ylabel('operator scale Λ [TeV]')
ax.set_title('Scale of the Majorana-splitting operator versus δ', fontsize=10, loc='left')
ax.legend(fontsize=7.5, loc='upper right', frameon=False)
fig.tight_layout(); fig.savefig(f'{FIG}/P037_fig2_lambda_vs_delta.png', dpi=160); plt.close(fig)

# ---------------------------------------------------------------- JSON summary
summary = dict(
    sigma_n_higgsino_1TeV_cm2=sigma_n_vector(0.5, 1000), vector_factor=VEC_FACTOR,
    sigma_n_by_Y_1TeV={str(Y): sigma_n_vector(Y, 1000) for Y in YS},
    sigma_SI_eq_by_Y_1TeV={str(Y): sigma_n_vector(Y, 1000) / VEC_FACTOR for Y in YS},
    elastic_excess_vs_LZ2024_by_Y={str(Y): sigma_n_vector(Y, 1000) / VEC_FACTOR / SIGMA_SI_2024 for Y in YS},
    Lambda_TeV={f'Y{Y}': {f'{d:.0f}': Lambda_GeV(Y, d) / 1e3 for d in DELTAS_B} for Y in YS},
    Lambda_convention_factor_vev_over_sqrt2={f'Y{Y}': 2**(-2 * Y / (4 * Y - 1)) for Y in YS},
    delta_if_Lambda_MPl_keV={f'Y{Y}': delta_from_Lambda_keV(Y, MPL) for Y in YS},
    gaugino_crosscheck=rowsB2,
    required_delta_1TeV={f'Y{Y}': dict(N3p65=pick(1000, Y, 'delta_N3.65_annual'), N1=pick(1000, Y, 'delta_N1_annual'),
                                     N0p105=pick(1000, Y, 'delta_N0.105_annual'), sun=pick(1000, Y, 'delta_N1_sun'),
                                     june=pick(1000, Y, 'delta_N1_june16'), vesc528=pick(1000, Y, 'delta_N1_annual_vesc528'),
                                     vesc560=pick(1000, Y, 'delta_N1_annual_vesc560'), window=[pick(1000, Y, 'window_lo'), pick(1000, Y, 'window_hi')])
                         for Y in YS},
    required_delta_all=dfC[['m_GeV', 'Y', 'delta_N3.65_annual', 'delta_N1_annual', 'delta_N0.105_annual', 'delta_max_248_june', 'window_width_keV']].to_dict('records'),
    wimpydd_crosscheck=wchk,
    radiative_1TeV={f'({r.n},{r.Y}) Q={r.Q:+d}': r.dm_MeV_1000 for _, r in dfD.iterrows()},
    thermal_masses_recalled={lab: (mth, rel) for n, Y, lab, mth, rel in MULTIPLETS},
    runtime_s=time.time() - T0)
json.dump(summary, open(f'{OUT}/P037_results.json', 'w'), indent=1, default=float)
P(f'\nDone in {time.time()-T0:.0f} s. Files in {OUT}/')
LOG.close()
