"""P047: Isotope by isotope - which xenon nuclei produce a 248 keV recoil for each interaction, and what
isotopically modified xenon could reveal.

Run from the simulation root:  .venv/bin/python output/code/P047_isotopes.py  [--fast]

Physics
-------
WimPyDD's diff_rate(..., isotopes_list={0: [i]}) returns the differential rate from isotope i per kg of
*natural* xenon: the number of nuclei of isotope i per kg is nt_kg_i = a_i * 1000 / <A> * N_A  (package.py l.467),
a_i = natural number abundance, <A> = sum a_i A_i (incl. the response-less 124Xe/126Xe).  For a modified
composition with number fractions a'_i the rate per kg is therefore

    R'(E) = sum_i R_i(E) * (a'_i / a_i) * (<A> / <A'>),        <A'> = sum a'_i A_i .

Operators (LZ unit coupling c = 1/m_v^2 in the Anand convention = WimPyDD c^0 = 2/m_v^2, P003), m_chi = 1 TeV:
  O1 elastic; O1 inelastic delta = 300/350/366/380 keV (June-16 halo and annual halo); O4, O6, O10 elastic;
  L10-like (q^2/m_N^2) O4 - O6 (P003/P017 reduction).
Outputs -> output/work/P047/ (CSV/JSON, run_log.txt) and output/work/P047/figures/ (PNG).
"""
import os, sys, json, math, time, argparse
import numpy as np
import pandas as pd
from scipy import special, integrate, stats, optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, 'output/code')
from common import lzcommon as lz

ap = argparse.ArgumentParser(); ap.add_argument('--fast', action='store_true'); ARGS = ap.parse_args()
OUT = 'output/work/P047'; FIG = os.path.join(OUT, 'figures'); os.makedirs(FIG, exist_ok=True)
LOG = open(os.path.join(OUT, 'run_log.txt'), 'w')
def log(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.write(s + '\n'); LOG.flush()
pd.set_option('display.width', 250)

WD = lz.wd(); xe = WD.Xe; NC = WD.nuclear_current
MCHI = 1000.0
MN = lz.M_NUCLEON_GEV
E_EVENT, DE = 248.0, 23.0
WIN = (E_EVENT - DE, E_EVENT + DE)            # 225-271 keV
ROI = (5.4, 270.0)
DOY_JUNE = 167                                # 16 June
DELTAS = [300.0, 350.0, 366.0, 380.0]         # inelastic splittings (366 = P007 Higgsino, 380 = P021 peak)

# ------------------------------------------------------------------------------------------------------
# 0. Isotope bookkeeping (WimPyDD abundances; 124Xe/126Xe carry no density matrices -> zero response)
# ------------------------------------------------------------------------------------------------------
A_ALL = np.array([int(n[:3]) for n in xe.isotopes]); AB_ALL = np.array(xe.abundance, float)
AMEAN = float(np.sum(A_ALL * AB_ALL) / np.sum(AB_ALL))
ACTIVE = [k for k in range(len(A_ALL)) if xe.func_w[k](1e-6)[NC['M'], 0, 0] > 0]
A_ACT = A_ALL[ACTIVE]; AB_ACT = AB_ALL[ACTIVE]
J_OF = {129: 0.5, 131: 1.5}
log(f'WimPyDD xenon: isotopes {list(A_ALL)}, abundances {list(np.round(AB_ALL, 5))}, <A> = {AMEAN:.3f}')
log(f'active (response tables present): {list(A_ACT)}; inactive (zero response): {[int(A_ALL[k]) for k in range(len(A_ALL)) if k not in ACTIVE]}')
frac_missing_mass = float(np.sum(AB_ALL[[k for k in range(len(A_ALL)) if k not in ACTIVE]] * A_ALL[[k for k in range(len(A_ALL)) if k not in ACTIVE]]) / np.sum(AB_ALL * A_ALL))
frac_missing_A2 = float((0.00095 * 124**2 + 0.00089 * 126**2) / np.sum(AB_ALL * A_ALL**2))
log(f'124Xe+126Xe: mass fraction {100*frac_missing_mass:.3f} %, A^2-weighted SI share {100*frac_missing_A2:.3f} % (their rate is set to zero by WimPyDD)')

# ------------------------------------------------------------------------------------------------------
# 1. Hamiltonians and halos
# ------------------------------------------------------------------------------------------------------
c0u, c1u = lz.wd_c_from_anand(1.0 / lz.M_V_GEV**2)
H = {'O1': lz.wd_hamiltonian('P047_O1', {1: (c0u, c1u)}),
     'O4': lz.wd_hamiltonian('P047_O4', {4: (c0u, c1u)}),
     'O6': lz.wd_hamiltonian('P047_O6', {6: (c0u, c1u)}),
     'O10': lz.wd_hamiltonian('P047_O10', {10: (c0u, c1u)}),
     'L10': WD.eft_hamiltonian('P047_L10like', {(4, 'q2'): lambda q, A=c0u: [A * q**2 / MN**2, 0.0], 6: lambda A=c0u: [-A, 0.0]})}
halo_ann = lz.wd_halo(); halo_jun = lz.wd_halo(day_of_year=DOY_JUNE)
V_E_JUNE = lz.v_earth_kms(DOY_JUNE); VMAX_JUNE = V_E_JUNE + lz.VESC_KMS
V_E_ANN = lz.v_earth_kms(); VMAX_ANN = V_E_ANN + lz.VESC_KMS
log(f'v_E(16 June) = {V_E_JUNE:.1f} km/s, v_max = {VMAX_JUNE:.1f}; annual mean v_E = {V_E_ANN:.1f}, v_max = {VMAX_ANN:.1f} km/s')

# cases: (name, operator, delta, halo label)
CASES = [('O1_el', 'O1', 0.0, 'ann'), ('O4', 'O4', 0.0, 'ann'), ('O6', 'O6', 0.0, 'ann'), ('O10', 'O10', 0.0, 'ann'), ('L10', 'L10', 0.0, 'ann')]
for d in DELTAS:
    CASES.append((f'O1_d{int(d)}_jun', 'O1', d, 'jun'))
for d in (300.0, 350.0, 366.0):
    CASES.append((f'O1_d{int(d)}_ann', 'O1', d, 'ann'))
LABEL = {'O1_el': 'O1 elastic', 'O4': 'O4', 'O6': 'O6', 'O10': 'O10', 'L10': 'L10 (q²O4−O6)',
         'O1_d300_jun': 'O1 δ=300 (Jun)', 'O1_d350_jun': 'O1 δ=350 (Jun)', 'O1_d366_jun': 'O1 δ=366 (Jun)', 'O1_d380_jun': 'O1 δ=380 (Jun)',
         'O1_d300_ann': 'O1 δ=300 (ann)', 'O1_d350_ann': 'O1 δ=350 (ann)', 'O1_d366_ann': 'O1 δ=366 (ann)'}

E_GRID = np.union1d(np.union1d(np.arange(1.0, 200.0, 2.0), np.arange(200.0, 300.01, 1.0)), [5.4, 225.0, 248.0, 269.9, 271.0])
CACHE = os.path.join(OUT, 'P047_isotope_spectra.npz')
if ARGS.fast and os.path.exists(CACHE):
    S = dict(np.load(CACHE)); log('loaded cached per-isotope spectra')
else:
    S = {'E': E_GRID}; t0 = time.time()
    for cname, op, delta, hl in CASES:
        halo = halo_jun if hl == 'jun' else halo_ann
        for k in ACTIVE:
            S[f'{cname}_{A_ALL[k]}'] = np.array([WD.diff_rate(xe, H[op], MCHI, float(e), halo[0], halo[1], delta=delta, isotopes_list={0: [k]})
                                                 for e in E_GRID]) * 1000.0 * 365.25       # events/(t yr keV) of natural Xe
        log(f'  {cname:14s} done ({time.time()-t0:.0f} s)')
    np.savez(CACHE, **S)
E = S['E']
def spec(cname, A): return np.clip(S[f'{cname}_{A}'], 0.0, None)     # tiny negative interpolation artefacts near nodes -> 0

# efficiency (P003 model as used by P016/P021/P032)
def eff(E):
    E = np.asarray(E, float)
    return 0.96 * 0.5 * (1 + special.erf((E - 5.4) / (math.sqrt(2) * 3.4))) * 0.5 * (1 - special.erf((E - 269.9) / (math.sqrt(2) * 8.0)))
E_FINE = np.linspace(1.0, 300.0, 5981)
def integ(y, lo, hi, weighted=False):
    f = np.interp(E_FINE, E, y); m = (E_FINE >= lo) & (E_FINE <= hi)
    w = eff(E_FINE[m]) if weighted else 1.0
    return float(np.trapezoid(f[m] * w, E_FINE[m]))

# ------------------------------------------------------------------------------------------------------
# 2. Per-isotope fractions: 225-271 keV (plain), full ROI 5.4-270 keV (efficiency-weighted), dR/dE at 248 keV
# ------------------------------------------------------------------------------------------------------
rows = []
for cname, op, delta, hl in CASES:
    Rw = {A: integ(spec(cname, A), *WIN) for A in A_ACT}
    Rr = {A: integ(spec(cname, A), *ROI, weighted=True) for A in A_ACT}
    R248 = {A: float(np.interp(248.0, E, spec(cname, A))) for A in A_ACT}
    sw, sr, s248 = sum(Rw.values()), sum(Rr.values()), sum(R248.values())
    for A in A_ACT:
        rows.append(dict(case=cname, A=int(A), abundance=float(AB_ALL[A_ALL == A][0]), J=J_OF.get(int(A), 0.0),
                         R_win_per_tyr=Rw[A], f_win=Rw[A] / sw if sw > 0 else np.nan,
                         R_roi_per_tyr=Rr[A], f_roi=Rr[A] / sr if sr > 0 else np.nan,
                         dRdE_248=R248[A], f_248=R248[A] / s248 if s248 > 0 else np.nan,
                         f_win_per_abundance=(Rw[A] / sw) / AB_ALL[A_ALL == A][0] if sw > 0 else np.nan))
frac = pd.DataFrame(rows); frac.to_csv(os.path.join(OUT, 'P047_isotope_fractions.csv'), index=False, float_format='%.5g')
piv_w = frac.pivot(index='A', columns='case', values='f_win')[[c[0] for c in CASES]]
piv_r = frac.pivot(index='A', columns='case', values='f_roi')[[c[0] for c in CASES]]
piv_248 = frac.pivot(index='A', columns='case', values='f_248')[[c[0] for c in CASES]]
log('\n== isotope fractions of the rate in 225-271 keV ==\n' + piv_w.to_string(float_format=lambda x: f'{x:.3f}'))
log('\n== isotope fractions of the efficiency-weighted rate in 5.4-270 keV ==\n' + piv_r.to_string(float_format=lambda x: f'{x:.3f}'))
log('\n== isotope fractions of dR/dE at 248 keV (compare P017: L10 70/30, O4 52/48, O6 40/60) ==\n' + piv_248.to_string(float_format=lambda x: f'{x:.3f}'))
tot = frac.groupby('case')[['R_win_per_tyr', 'R_roi_per_tyr']].sum().loc[[c[0] for c in CASES]]
tot['N_win_2p84'] = tot.R_win_per_tyr * 2.84; tot['N_roi_2p84'] = tot.R_roi_per_tyr * 2.84
log('\n== natural-Xe totals (unit coupling, 1 TeV) ==\n' + tot.to_string(float_format=lambda x: f'{x:.4g}'))
# A^2-weighted reference for the SI pattern and 'per abundance' enhancement of the heavy isotopes
a2 = AB_ACT * A_ACT**2 / np.sum(AB_ACT * A_ACT**2)
log('\nA^2 x abundance reference shares: ' + ', '.join(f'{A}:{v:.3f}' for A, v in zip(A_ACT, a2)))
odd_share = {c[0]: float(piv_w.loc[[129, 131], c[0]].sum()) for c in CASES}
heavy_share = {c[0]: float(piv_w.loc[[134, 136], c[0]].sum()) for c in CASES}
log('odd-A (129+131) share of the window rate: ' + ', '.join(f'{k}:{v:.3f}' for k, v in odd_share.items()))
log('134+136 share of the window rate:        ' + ', '.join(f'{k}:{v:.3f}' for k, v in heavy_share.items()))

# ------------------------------------------------------------------------------------------------------
# 3. Kinematic isotope effect for inelastic scattering
# ------------------------------------------------------------------------------------------------------
kin = []
for A in A_ALL:
    dmax248 = lz.delta_max_kev(248.0, MCHI, A=float(A), v_kms=VMAX_JUNE)
    dmax225 = lz.delta_max_kev(225.0, MCHI, A=float(A), v_kms=VMAX_JUNE)
    dmax271 = lz.delta_max_kev(271.0, MCHI, A=float(A), v_kms=VMAX_JUNE)
    # absolute kinematic ceiling for any recoil: delta_max = mu v^2 / 2 (v = v_max), at E* = mu^2 v^2 / (2 m_N)... (E at which delta_max(E) peaks)
    mN = lz.m_nucleus_gev(float(A)); mu = lz.mu_red(MCHI, mN); beta = VMAX_JUNE / lz.C_KMS
    d_ceiling = 0.5 * mu * beta**2 * 1e6; E_star = mu**2 * beta**2 / (2 * mN) * 1e6
    row = dict(A=int(A), abundance=float(AB_ALL[A_ALL == A][0]), delta_max_248_june=dmax248, delta_max_225_june=dmax225, delta_max_271_june=dmax271,
               delta_max_248_annual=lz.delta_max_kev(248.0, MCHI, A=float(A), v_kms=VMAX_ANN), delta_ceiling_june=d_ceiling, E_star_keV=E_star)
    for d in DELTAS:
        lo, hi = lz.E_R_range_keV(MCHI, VMAX_JUNE, A=float(A), delta_kev=d)
        row[f'Emin_d{int(d)}'] = lo; row[f'Emax_d{int(d)}'] = hi
    kin.append(row)
kin = pd.DataFrame(kin); kin.to_csv(os.path.join(OUT, 'P047_kinematics_per_isotope.csv'), index=False, float_format='%.5g')
log('\n== inelastic kinematics per isotope (1 TeV, 16 June, v_max = %.1f km/s) ==\n' % VMAX_JUNE + kin.to_string(float_format=lambda x: f'{x:.1f}'))
log(f'natural-Xe mean-A reference: delta_max(248) = {lz.delta_max_kev(248.0, MCHI, v_kms=VMAX_JUNE):.1f} keV (June), {lz.delta_max_kev(248.0, MCHI, v_kms=VMAX_ANN):.1f} (annual)')
# isotope pattern of dR/dE(248) vs delta from WimPyDD on a finer delta scan (June halo), 7 isotopes
DSCAN = [300, 330, 350, 360, 366, 370, 375, 380, 383, 385, 387, 389, 391, 393, 395, 397]
E_WSCAN = np.arange(225.0, 271.01, 2.0)
scan_rows = []
t0 = time.time()
for d in DSCAN:
    r = {int(A_ALL[k]): max(float(WD.diff_rate(xe, H['O1'], MCHI, 248.0, halo_jun[0], halo_jun[1], delta=float(d), isotopes_list={0: [k]})) * 1000 * 365.25, 0.0) for k in ACTIVE}
    s = sum(r.values())
    # window-integrated (225-271 keV) per-isotope rate at this delta: the point value at 248 keV hides 136Xe, whose M node is at 251 keV
    rw = {}
    for k in ACTIVE:
        y = np.clip(np.array([WD.diff_rate(xe, H['O1'], MCHI, float(e), halo_jun[0], halo_jun[1], delta=float(d), isotopes_list={0: [k]}) for e in E_WSCAN]) * 1000 * 365.25, 0, None)
        rw[int(A_ALL[k])] = float(np.trapezoid(y, E_WSCAN))
    sw = sum(rw.values())
    scan_rows.append(dict(delta_keV=d, dRdE_248_total=s, R_win_total=sw, **{f'f248_{A}': (r[A] / s if s > 0 else np.nan) for A in r},
                          **{f'fwin_{A}': (rw[A] / sw if sw > 0 else np.nan) for A in rw}))
dscan = pd.DataFrame(scan_rows); dscan.to_csv(os.path.join(OUT, 'P047_share_vs_delta.csv'), index=False, float_format='%.5g')
log(f'\n== isotope shares of dR/dE(248 keV) vs delta (June, 1 TeV) [{time.time()-t0:.0f} s] ==\n' + dscan[['delta_keV', 'dRdE_248_total'] + [f'f248_{A}' for A in A_ACT]].to_string(float_format=lambda x: f'{x:.3g}'))
log('\n== isotope shares of the 225-271 keV window rate vs delta (June, 1 TeV) ==\n' + dscan[['delta_keV', 'R_win_total'] + [f'fwin_{A}' for A in A_ACT]].to_string(float_format=lambda x: f'{x:.3g}'))

# ------------------------------------------------------------------------------------------------------
# 4. Modified compositions
# ------------------------------------------------------------------------------------------------------
def composition(mod):
    """mod: dict A -> fixed number fraction; the remaining isotopes share the rest in natural proportion."""
    a = AB_ALL.copy()
    fixed = sum(mod.values()); free = np.array([A not in mod for A in A_ALL])
    a[free] = AB_ALL[free] / AB_ALL[free].sum() * (1.0 - fixed)
    for A, v in mod.items(): a[A_ALL == A] = v
    return a
COMPS = {'natural': composition({}),
         '136-depleted (0.1%)': composition({136: 0.001}),
         '129-enriched 80%': composition({129: 0.80}),
         '136-enriched 90%': composition({136: 0.90}),
         '134+136 enriched 90%': composition({134: 0.90 * 0.10436 / (0.10436 + 0.08857), 136: 0.90 * 0.08857 / (0.10436 + 0.08857)}),
         'even-A only (idealised)': composition({129: 0.001, 131: 0.001})}
comp_df = pd.DataFrame({k: v for k, v in COMPS.items()}, index=[f'{A}Xe' for A in A_ALL]).T
comp_df['<A>'] = [float(np.sum(v * A_ALL)) for v in COMPS.values()]
comp_df.to_csv(os.path.join(OUT, 'P047_compositions.csv'), float_format='%.4f')
log('\n== compositions (number fractions) ==\n' + comp_df.to_string(float_format=lambda x: f'{x:.4f}'))

def rate_mod(cname, a_new, lo, hi, weighted=False):
    Anew = float(np.sum(a_new * A_ALL)); tot = 0.0
    for A in A_ACT:
        i = int(np.where(A_ALL == A)[0][0])
        tot += integ(spec(cname, A), lo, hi, weighted) * (a_new[i] / AB_ALL[i]) * (AMEAN / Anew)
    return tot
ratio_rows = []
for cname, op, delta, hl in CASES:
    Rn_w = rate_mod(cname, COMPS['natural'], *WIN); Rn_r = rate_mod(cname, COMPS['natural'], *ROI, weighted=True)
    for lab, a in COMPS.items():
        ratio_rows.append(dict(case=cname, composition=lab, R_win=rate_mod(cname, a, *WIN), ratio_win=rate_mod(cname, a, *WIN) / Rn_w,
                               R_roi=rate_mod(cname, a, *ROI, weighted=True), ratio_roi=rate_mod(cname, a, *ROI, weighted=True) / Rn_r))
ratios = pd.DataFrame(ratio_rows); ratios.to_csv(os.path.join(OUT, 'P047_composition_ratios.csv'), index=False, float_format='%.5g')
piv_ratio = ratios.pivot(index='composition', columns='case', values='ratio_win').loc[list(COMPS)][[c[0] for c in CASES]]
log('\n== rate in 225-271 keV per tonne relative to natural xenon ==\n' + piv_ratio.to_string(float_format=lambda x: f'{x:.3f}'))
piv_ratio_roi = ratios.pivot(index='composition', columns='case', values='ratio_roi').loc[list(COMPS)][[c[0] for c in CASES]]
log('\n== full-ROI (5.4-270 keV, eff-weighted) rate per tonne relative to natural xenon ==\n' + piv_ratio_roi.to_string(float_format=lambda x: f'{x:.3f}'))

# discrimination: double ratio D = rho(H1)/rho(H2) and Poisson/binomial events needed
def n_for_3sigma(rho1, rho2, Z=3.0):
    """Two detectors (natural, modified), equal exposure. H1 predicts (N, rho1 N), H2 (N, rho2 N).
    (i) normalisation free -> binomial fraction test, f = rho/(1+rho); Asimov q = 2 N_tot KL(f1||f2).
    (ii) normalisation fixed by H (LZ rate known) -> Asimov Poisson LLR on both detectors with N_nat = N.
    Returns N_tot (total events in both detectors under H1) for median Z, both ways, and also H2-true."""
    f1, f2 = rho1 / (1 + rho1), rho2 / (1 + rho2)
    kl = f1 * math.log(f1 / f2) + (1 - f1) * math.log((1 - f1) / (1 - f2))
    if kl < 1e-12:      # identical composition response (e.g. two pure spin operators): no discrimination possible
        return dict(N_tot_free=np.inf, N_nat_fixed=np.inf, N_tot_fixed=np.inf)
    N_tot_free = Z**2 / (2 * kl)
    # (ii): counts (N, rho1 N) observed; LLR vs (N, rho2 N): Poisson Asimov 2*sum[mu1 ln(mu1/mu2) - mu1 + mu2] with mu1_nat = mu2_nat = N
    kl2 = rho1 * math.log(rho1 / rho2) - rho1 + rho2
    N_fixed = Z**2 / (2 * kl2)             # N = natural-detector expectation; total = N (1 + rho1)
    return dict(N_tot_free=N_tot_free, N_nat_fixed=N_fixed, N_tot_fixed=N_fixed * (1 + rho1))
PAIRS = [('O1_d366_jun', 'L10'), ('O1_d380_jun', 'L10'), ('O1_d350_jun', 'L10'), ('O1_d300_jun', 'L10'), ('O1_d366_jun', 'O6'), ('O1_d366_jun', 'O10'),
         ('O1_el', 'L10'), ('O1_d366_jun', 'O1_el'), ('L10', 'O6'), ('O4', 'L10')]
disc = []
for lab in COMPS:
    if lab == 'natural': continue
    for h1, h2 in PAIRS:
        r1 = float(piv_ratio.loc[lab, h1]); r2 = float(piv_ratio.loc[lab, h2])
        if r1 <= 0 or r2 <= 0: continue
        n = n_for_3sigma(r1, r2)
        disc.append(dict(composition=lab, H1=h1, H2=h2, rho_H1=r1, rho_H2=r2, D=r1 / r2,
                         N_tot_3sigma_free=n['N_tot_free'], exposure_per_det_free_tyr=n['N_tot_free'] / (1 + r1) * 2.84,
                         N_tot_3sigma_fixed=n['N_tot_fixed'], exposure_per_det_fixed_tyr=n['N_nat_fixed'] * 2.84))
disc = pd.DataFrame(disc); disc.to_csv(os.path.join(OUT, 'P047_discrimination.csv'), index=False, float_format='%.4g')
log('\n== discrimination between hypotheses with a natural + modified detector pair (equal exposure; LZ rate = 1 event / 2.84 t yr in 225-271 keV) ==')
log(disc.to_string(float_format=lambda x: f'{x:.3g}'))
# exact binomial version (normalisation free): smallest N_tot such that the median H1 count in the modified detector
# has a one-sided H2 p-value <= 1.35e-3 (3 sigma), and vice versa (H2 true, reject H1)
def exact_binomial_N(rho1, rho2, p_target=stats.norm.sf(3.0)):
    f1, f2 = rho1 / (1 + rho1), rho2 / (1 + rho2)
    for N in range(1, 20001):
        med = stats.binom.median(N, f1)
        p = stats.binom.sf(med - 1, N, f2) if f1 > f2 else stats.binom.cdf(med, N, f2)
        if p <= p_target: return N
    return np.inf
ex_rows = []
for lab in ('136-enriched 90%', '134+136 enriched 90%', '129-enriched 80%', '136-depleted (0.1%)', 'even-A only (idealised)'):
    for h1, h2 in (('O1_d366_jun', 'L10'), ('O1_d380_jun', 'L10'), ('O1_d350_jun', 'L10'), ('O1_d300_jun', 'L10'), ('O1_d366_jun', 'O1_el')):
        r1, r2 = float(piv_ratio.loc[lab, h1]), float(piv_ratio.loc[lab, h2])
        N12, N21 = exact_binomial_N(r1, r2), exact_binomial_N(r2, r1)
        ex_rows.append(dict(composition=lab, H1=h1, H2=h2, rho_H1=r1, rho_H2=r2, N_tot_reject_H2_when_H1=N12, N_tot_reject_H1_when_H2=N21,
                            exposure_per_det_H1_tyr=N12 / (1 + r1) * 2.84 if np.isfinite(N12) else np.inf,
                            exposure_per_det_H2_tyr=N21 / (1 + r2) * 2.84 if np.isfinite(N21) else np.inf))
exact = pd.DataFrame(ex_rows); exact.to_csv(os.path.join(OUT, 'P047_discrimination_exact_binomial.csv'), index=False, float_format='%.4g')
log('\n== exact binomial 3 sigma separation (median count under the true hypothesis; normalisation free) ==\n' + exact.to_string(float_format=lambda x: f'{x:.3g}'))
best = disc[(disc.H1 == 'O1_d366_jun') & (disc.H2 == 'L10')].sort_values('N_tot_3sigma_free')
log('\nbest composition for inelastic(366)/L10: ' + best.iloc[0].composition + f"  D = {best.iloc[0].D:.3g}, N_tot(3 sigma, free norm) = {best.iloc[0].N_tot_3sigma_free:.1f}, per-detector exposure {best.iloc[0].exposure_per_det_free_tyr:.0f} t yr")

# ------------------------------------------------------------------------------------------------------
# 5. Nuclear-structure caveats (quantified where possible)
# ------------------------------------------------------------------------------------------------------
# 5a. SI per-isotope shares in the window with Helm instead of shell-model M response: lzcommon.dRdE_SI per A (elastic only)
sig_unit = (1 / lz.M_V_GEV**2)**2 * lz.mu_red(MCHI, MN)**2 / math.pi * lz.GEV_TO_CM2
helm = {}
for A in A_ACT:
    ab = float(AB_ALL[A_ALL == A][0])
    y = np.array([lz.dRdE_SI(e, MCHI, sig_unit, A=float(A)) for e in E]) * ab * A / AMEAN    # per-isotope share of a natural-Xe tonne (mass fraction ab*A/<A>)
    helm[A] = y
hw = {A: integ(helm[A], *WIN) for A in A_ACT}; hs = sum(hw.values())
nuc = pd.DataFrame(dict(A=A_ACT, f_win_shell=[piv_w.loc[A, 'O1_el'] for A in A_ACT], f_win_helm=[hw[A] / hs for A in A_ACT],
                        f_roi_shell=[piv_r.loc[A, 'O1_el'] for A in A_ACT], f_win_A2ab=a2))
nuc['shell_over_helm'] = nuc.f_win_shell / nuc.f_win_helm
nuc.to_csv(os.path.join(OUT, 'P047_SI_shares_shell_vs_helm.csv'), index=False, float_format='%.4g')
log('\n== O1 elastic per-isotope shares in 225-271 keV: shell model vs Helm vs A^2-abundance ==\n' + nuc.to_string(float_format=lambda x: f'{x:.3f}'))
# per-isotope node positions (from P017 table) and distance of 248 keV from each node
p17 = pd.read_csv('output/work/P017/P017_M_nodes.csv'); p17 = p17[(p17.model == 'shell') & (p17.A > 0)][['A', 'E_node2_keV']]
log('P017 shell-model second M node per isotope [keV]: ' + ', '.join(f'{int(a)}:{e:.1f}' for a, e in zip(p17.A, p17.E_node2_keV)))
# 5b. spin shares under +-15 % response-shape variation (P017's Sigma' robustness band) applied to one isotope at a time
def share_var(cname, A_up, fac):
    Rw = {A: integ(spec(cname, A), *WIN) for A in (129, 131)}; Rw[A_up] *= fac
    return Rw[129] / (Rw[129] + Rw[131])
spin_var = []
for cname in ('L10', 'O4', 'O6', 'O10'):
    base = share_var(cname, 129, 1.0)
    spin_var.append(dict(case=cname, f129_win=base, f129_129x0p85=share_var(cname, 129, 0.85), f129_129x1p15=share_var(cname, 129, 1.15),
                         f129_131x0p85=share_var(cname, 131, 0.85), f129_131x1p15=share_var(cname, 131, 1.15),
                         ratio_129enr=float(piv_ratio.loc['129-enriched 80%', cname]), ratio_136enr=float(piv_ratio.loc['136-enriched 90%', cname])))
spin_var = pd.DataFrame(spin_var); spin_var.to_csv(os.path.join(OUT, 'P047_spin_share_variation.csv'), index=False, float_format='%.4g')
log('\n== 129Xe share of the spin-operator window rate under +-15 % single-isotope response variations ==\n' + spin_var.to_string(float_format=lambda x: f'{x:.3f}'))
# how the composition ratio for the spin operators depends on the 129 share: analytic
# rho(129-enr) = (a'_129/a_129 f129 + a'_131/a_131 (1-f129)) <A>/<A'>;  d rho / d f129 = (a'_129/a_129 - a'_131/a_131) <A>/<A'>
a_e = COMPS['129-enriched 80%']; Ae = float(np.sum(a_e * A_ALL))
slope = (a_e[A_ALL == 129][0] / AB_ALL[A_ALL == 129][0] - a_e[A_ALL == 131][0] / AB_ALL[A_ALL == 131][0]) * AMEAN / Ae
log(f'129-enriched: d rho/d f129 = {slope:.3f} -> a +-0.05 change of the 129Xe share moves rho by -+{0.05*abs(slope):.3f}')

# ------------------------------------------------------------------------------------------------------
# 6. Summary JSON
# ------------------------------------------------------------------------------------------------------
summ = dict(m_chi_GeV=MCHI, v_max_june=VMAX_JUNE, v_max_annual=VMAX_ANN, A_mean=AMEAN, missing_124_126=dict(mass_frac=frac_missing_mass, A2_share=frac_missing_A2),
            f_win=piv_w.to_dict(), f_roi=piv_r.to_dict(), f_248=piv_248.to_dict(), totals=tot.to_dict(),
            odd_share_win=odd_share, heavy_share_win=heavy_share,
            kinematics=kin.set_index('A').to_dict(orient='index'), share_vs_delta=dscan.set_index('delta_keV').to_dict(orient='index'),
            compositions={k: {f'{A}': float(v) for A, v in zip(A_ALL, a)} for k, a in COMPS.items()},
            ratio_win=piv_ratio.to_dict(), ratio_roi=piv_ratio_roi.to_dict(), discrimination=disc.to_dict(orient='records'),
            SI_shares_shell_vs_helm=nuc.to_dict(orient='records'), spin_share_variation=spin_var.to_dict(orient='records'), slope_rho_f129=float(slope))
def _c(o):
    if isinstance(o, dict): return {str(k): _c(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_c(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if (isinstance(o, float) and math.isnan(o)) else float(o)
    if isinstance(o, (np.integer,)): return int(o)
    return o
json.dump(_c(summ), open(os.path.join(OUT, 'P047_summary.json'), 'w'), indent=1)

# ------------------------------------------------------------------------------------------------------
# 7. Figures (dataviz palette; 7 fixed categorical slots = 7 active isotopes)
# ------------------------------------------------------------------------------------------------------
C = dict(blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', yellow='#eda100', magenta='#e87ba4', violet='#4a3aa7', red='#e34948', green='#008300',
         ink='#0b0b0b', ink2='#52514e', muted='#898781', grid='#e1e0d9', surface='#fcfcfb')
ISOCOL = {128: C['blue'], 129: C['orange'], 130: C['aqua'], 131: C['yellow'], 132: C['magenta'], 134: C['violet'], 136: C['red']}
plt.rcParams.update({'figure.facecolor': C['surface'], 'axes.facecolor': C['surface'], 'axes.edgecolor': C['muted'], 'axes.labelcolor': C['ink2'],
                     'xtick.color': C['ink2'], 'ytick.color': C['ink2'], 'grid.color': C['grid'], 'axes.grid': True, 'grid.linewidth': 0.6,
                     'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
SHOW = ['O1_el', 'O4', 'O10', 'O6', 'L10', 'O1_d300_jun', 'O1_d350_jun', 'O1_d366_jun', 'O1_d380_jun']
# Fig 1: stacked fractions, two panels
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for ax, piv, title in ((axs[0], piv_w, '225–271 keV (event window)'), (axs[1], piv_r, '5.4–270 keV (full ROI, efficiency-weighted)')):
    x = np.arange(len(SHOW)); bottom = np.zeros(len(SHOW))
    for A in A_ACT:
        vals = np.array([piv.loc[A, c] for c in SHOW])
        ax.bar(x, vals, 0.72, bottom=bottom, color=ISOCOL[A], edgecolor=C['surface'], linewidth=1.5, label=f'$^{{{A}}}$Xe')
        for xi, v, b in zip(x, vals, bottom):
            if v >= 0.07: ax.text(xi, b + v / 2, f'{100*v:.0f}', ha='center', va='center', fontsize=7.5, color=C['ink'])
        bottom += vals
    ax.set_xticks(x); ax.set_xticklabels([LABEL[c].replace(' (Jun)', '') for c in SHOW], rotation=35, ha='right'); ax.set_ylim(0, 1.0)
    ax.set_title(title, loc='left', color=C['ink'], fontsize=9.5); ax.grid(axis='x', visible=False)
axs[0].set_ylabel('fraction of the rate from each isotope'); axs[1].legend(fontsize=8, ncol=1, loc='center left', bbox_to_anchor=(1.01, 0.5))
fig.suptitle('Which xenon isotopes carry the rate?  m = 1 TeV, WimPyDD shell-model responses (inelastic: 16 June halo)', x=0.01, ha='left', fontsize=10, color=C['ink'])
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P047_fig1_isotope_fractions.png'), dpi=150); plt.close(fig)

# Fig 2: rate ratio vs composition (grouped bars, log), compositions as groups, operators as bars (fixed operator colours)
OPCOL = {'L10': C['blue'], 'O6': C['aqua'], 'O10': C['magenta'], 'O4': C['violet'], 'O1_el': C['yellow'], 'O1_d300_jun': '#f7b78c', 'O1_d366_jun': C['orange'], 'O1_d380_jun': C['red']}
OPS2 = ['L10', 'O6', 'O10', 'O4', 'O1_el', 'O1_d300_jun', 'O1_d366_jun', 'O1_d380_jun']
comps2 = [k for k in COMPS if k != 'natural']
fig, ax = plt.subplots(figsize=(11, 4.4)); x = np.arange(len(comps2)); w = 0.1
for j, op in enumerate(OPS2):
    vals = [max(float(piv_ratio.loc[lab, op]), 1e-3) for lab in comps2]
    ax.bar(x + (j - 3.5) * w, vals, w * 0.9, color=OPCOL[op], edgecolor=C['surface'], linewidth=0.8, label=LABEL[op].replace(' (Jun)', ''))
    for xi, v in zip(x, vals):
        ax.text(xi + (j - 3.5) * w, v * 1.12, f'{v:.2f}' if v >= 0.1 else f'{v:.3f}', ha='center', fontsize=6, rotation=90, color=C['ink2'])
ax.set_yscale('log'); ax.axhline(1, color=C['muted'], lw=1); ax.set_ylim(1e-3, 30)
ax.set_xticks(x); ax.set_xticklabels(comps2); ax.set_ylabel('rate in 225–271 keV per tonne, relative to natural Xe'); ax.grid(axis='x', visible=False)
ax.set_title('Isotopically modified xenon: how the 248 keV window rate changes for each interaction (1 TeV)', loc='left', color=C['ink'], fontsize=9.5)
ax.legend(fontsize=7.5, ncol=4, loc='lower left'); fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P047_fig2_composition_ratios.png'), dpi=150); plt.close(fig)

# Fig 3: kinematic isotope effect: share of dR/dE(248) vs delta (stacked area) + delta_max markers
fig, ax = plt.subplots(figsize=(8.2, 4.3))
dd = dscan.delta_keV.values; bottom = np.zeros(len(dd))
for A in A_ACT:
    v = np.nan_to_num(dscan[f'fwin_{A}'].values); dm = float(kin[kin.A == A].delta_max_248_june.iloc[0])
    ax.fill_between(dd, bottom, bottom + v, color=ISOCOL[A], linewidth=0, label=f'$^{{{A}}}$Xe   (δ_max(248 keV) = {dm:.0f} keV)'); bottom += v
    ax.axvline(dm, color=ISOCOL[A], lw=1, ls=':')
ax.set_xlim(300, 397); ax.set_ylim(0, 1.0); ax.set_xlabel('mass splitting δ [keV]'); ax.set_ylabel('share of the 225–271 keV rate')
ax.set_title('Inelastic O1, 1 TeV, 16 June: heavier isotopes take over as δ → δ_max\n(dotted lines: δ above which each isotope can no longer reach 248 keV)', loc='left', color=C['ink'], fontsize=9)
ax.legend(fontsize=7.5, loc='center left', bbox_to_anchor=(1.01, 0.5)); ax.grid(axis='x', visible=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P047_fig3_share_vs_delta.png'), dpi=150); plt.close(fig)
log('\nfigures written to ' + FIG); log('files: ' + ', '.join(sorted(os.listdir(OUT))))
LOG.close()
