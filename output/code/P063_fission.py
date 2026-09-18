#!/usr/bin/env python
"""
P063 -- Spontaneous fission of 238U in LZ's detector components: multiplicity, veto
probability and the odds of a lone 248 keV recoil.

Run from the simulation root:  .venv/bin/python output/code/P063_fission.py

Outputs -> output/work/P063/  (CSV/JSON tables, figures/*.png, console log P063_console.txt)

Structure
  1. 238U SF nuclear data (radioactivedecay for the SF branching; recalled multiplicities and spectra)
  2. Material inventory (recalled, bracketed) -> 238U activity -> SF rate in 220 d (route A);
     (alpha,n)-budget back-calculation via P013 transport factors (route B);
     multiple-scatter sample cap (route C)
  3. Prompt-gamma transport MC through a simplified LZ geometry -> per-gamma silent probability
     by source location -> f_gamma for Poisson and narrow multiplicity distributions
  4. Neutron-companion silent factor f_n; per-fission probability of a lone 200-270 keV
     single scatter; expected counts in 220 d; sensitivity table
  5. Detectability: SF-induced multi-neutron / gamma bursts, TPC multiple scatters, OD multiplicity
"""
import sys, json, math, os
sys.path.insert(0, 'output/code')
import numpy as np
import pandas as pd
from scipy import integrate, stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import radioactivedecay as rd
from common import lzcommon as lz

OUT = 'output/work/P063'
FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)
LOG = []
def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s); LOG.append(s)

rng = np.random.default_rng(63)
T_LIVE_S = lz.LZ['live_days'] * 86400.0     # 220 d
P13 = json.load(open('output/work/P013/P013_results.json'))

# ----------------------------------------------------------------------------------------------
# 1. Nuclear data
# ----------------------------------------------------------------------------------------------
say('=== 1. 238U spontaneous-fission data ===')
U238 = rd.Nuclide('U-238')
bf = dict(zip(U238.progeny(), U238.branching_fractions()))
BR_SF = bf['SF']                                   # 5.45e-7 (ICRP-107 via radioactivedecay)
T12_U238_y = U238.half_life('y')
say(f'radioactivedecay: 238U T1/2 = {T12_U238_y:.4g} y, SF branching = {BR_SF:.3g}, modes {U238.decay_modes()}')
Th232 = rd.Nuclide('Th-232')
say(f'radioactivedecay: 232Th modes {Th232.decay_modes()} (no SF branch tabulated; recalled SF/alpha ~1e-11, negligible)')
for x in ['Cm-244', 'Cf-252']:
    n = rd.Nuclide(x); d = dict(zip(n.progeny(), n.branching_fractions()))
    say(f'radioactivedecay: {x} T1/2 = {n.half_life("y"):.4g} y, SF branching = {d.get("SF", 0):.3g}')

# multiplicity of prompt neutrons: Terrell-type Gaussian, nu_bar = 2.01, sigma = 1.08 (recalled, likely/uncertain)
NU_BAR, SIG_NU = 2.01, 1.08
nus = np.arange(0, 8)
def terrell_pnu(nubar, sigma):
    # discretise a Gaussian over integer nu (Terrell 1957): P(nu) = Phi((nu+1/2-nubar)/sig) - Phi((nu-1/2-nubar)/sig), renormalised
    edges = np.arange(-0.5, 8.0, 1.0)
    cdf = stats.norm.cdf((edges - nubar) / sigma)
    p = np.diff(cdf); p[0] += cdf[0]      # fold the negative tail into nu=0
    return p / p.sum()
P_NU = terrell_pnu(NU_BAR, SIG_NU)
P_NU_P013 = np.array([0.05, 0.25, 0.37, 0.25, 0.07, 0.01, 0, 0]); P_NU_P013 /= P_NU_P013.sum()
say('P(nu) Terrell(2.01, 1.08):', np.array2string(P_NU, precision=4), ' mean', (nus * P_NU).sum().round(3))
say('P(nu) P013 recall       :', np.array2string(P_NU_P013, precision=4), ' mean', (nus * P_NU_P013).sum().round(3))

# Watt spectrum for 238U SF (a = 0.988 MeV, b = 2.249 MeV^-1; recalled likely, SOURCES-4C) -- same as P013
WA, WB = 0.988, 2.249
watt = lambda E: np.exp(-E / WA) * np.sinh(np.sqrt(WB * E))
wnorm = integrate.quad(watt, 0, 40)[0]
f_gt = lambda E0: integrate.quad(watt, E0, 40)[0] / wnorm
E_MIN_248 = P13['kinematics']['E_n_min_MeV']['248']
E_MIN_200 = P13['kinematics']['E_n_min_MeV']['200']
WATT = dict(a=WA, b=WB, mean=integrate.quad(lambda E: E * watt(E), 0, 40)[0] / wnorm,
            f_gt_200keV_recoil=f_gt(E_MIN_200), f_gt_248keV_recoil=f_gt(E_MIN_248), f_gt_10MeV=f_gt(10.0))
say(f'Watt: mean {WATT["mean"]:.3f} MeV; f(E_n > {E_MIN_200:.2f}) = {WATT["f_gt_200keV_recoil"]:.3e}; '
    f'f(E_n > {E_MIN_248:.2f}) = {WATT["f_gt_248keV_recoil"]:.3e}; f(>10 MeV) = {WATT["f_gt_10MeV"]:.3e}')

# prompt gammas: mean multiplicity 6.5, mean energy 0.9 MeV, total ~ 6 MeV (recalled, uncertain)
NG_BAR, EG_MEAN = 6.5, 0.9
def pk_poisson(nbar, kmax=25):
    k = np.arange(kmax + 1); return k, stats.poisson.pmf(k, nbar)
def pk_narrow(nbar, sigma=2.5, kmin=1, kmax=25):
    k = np.arange(kmax + 1)
    edges = np.arange(-0.5, kmax + 1.0, 1.0)
    p = np.diff(stats.norm.cdf((edges - nbar) / sigma)); p[k < kmin] = 0.0
    return k, p / p.sum()
GAMMA_MODELS = {'Poisson(6.5)': pk_poisson(NG_BAR), 'narrow G(6.5,2.5) k>=1': pk_narrow(NG_BAR, 2.5, 1),
                'narrow G(6.5,2.5) k>=2': pk_narrow(NG_BAR, 2.5, 2)}
for name, (k, p) in GAMMA_MODELS.items():
    say(f'gamma multiplicity {name}: P(0)={p[0]:.2e} P(1)={p[1]:.2e} P(2)={p[2]:.2e} mean={float((k*p).sum()):.2f}')

# ----------------------------------------------------------------------------------------------
# 2. Material inventory and SF rate (route A), (alpha,n) budget (route B), MS-sample cap (route C)
# ----------------------------------------------------------------------------------------------
say('\n=== 2. 238U inventory and SF rate ===')
# columns: component, mass or count, unit, 238U(early) specific activity central/low/high, location class,
# p_enter (fraction of emitted neutrons that enter the active LXe; geometric, recalled/estimated)
INV = [
 # name,            amount, unit,   A_c,   A_lo,   A_hi,    unit_A,        loc,      p_enter, note
 ('PTFE (TPC walls, reflectors)', 1000., 'kg',  10e-6,  3e-6, 100e-6, 'Bq/kg', 'wall',   0.50, 'mass 0.5-1.5 t; U_e 3-100 uBq/kg (recalled, uncertain)'),
 ('Ti cryostat (ICV+OCV)',        2200., 'kg',  0.5e-3, 0.1e-3, 1.6e-3, 'Bq/kg', 'cryostat', 0.35, 'U_e < 1.6 mBq/kg assay UL (recalled, uncertain); U_l 0.09'),
 ('R11410 PMTs (TPC 494 + Skin 131)', 625., 'PMT', 2.0e-3, 1.0e-3, 3.0e-3, 'Bq/PMT', 'pmt',  0.40, 'U_e 1-3 mBq/PMT (recalled, uncertain)'),
 ('cables, feedthroughs, PMT bases, misc.', 1., 'lot', 0.3, 0.1, 0.6, 'Bq', 'pmt',  0.20, 'lumped; 0.1-0.6 Bq U_e (recalled, uncertain)'),
]
rows = []
for name, amt, unit, Ac, Alo, Ahi, uA, loc, pent, note in INV:
    A = np.array([Ac, Alo, Ahi]) * amt
    nsf = A * BR_SF * T_LIVE_S
    rows.append(dict(component=name, amount=amt, unit=unit, A_U238_Bq=A[0], A_lo=A[1], A_hi=A[2],
                     N_SF_220d=nsf[0], N_SF_lo=nsf[1], N_SF_hi=nsf[2], location=loc, p_enter=pent, note=note))
inv = pd.DataFrame(rows)
tot = dict(A=inv.A_U238_Bq.sum(), A_lo=inv.A_lo.sum(), A_hi=inv.A_hi.sum(),
           N_SF=inv.N_SF_220d.sum(), N_SF_lo=inv.N_SF_lo.sum(), N_SF_hi=inv.N_SF_hi.sum())
inv.to_csv(os.path.join(OUT, 'P063_inventory.csv'), index=False)
say(inv[['component', 'A_U238_Bq', 'A_lo', 'A_hi', 'N_SF_220d', 'N_SF_lo', 'N_SF_hi', 'p_enter']].to_string(index=False))
say(f'TOTAL 238U(early) activity {tot["A"]:.2f} Bq ({tot["A_lo"]:.2f}-{tot["A_hi"]:.2f}); '
    f'SF in 220 d: {tot["N_SF"]:.1f} ({tot["N_SF_lo"]:.1f}-{tot["N_SF_hi"]:.1f})')
n_sf_neutrons = tot['N_SF'] * NU_BAR
say(f'SF neutrons emitted in 220 d: {n_sf_neutrons:.1f}; above {E_MIN_248:.2f} MeV: {n_sf_neutrons*WATT["f_gt_248keV_recoil"]:.3f} '
    f'(range {tot["N_SF_lo"]*NU_BAR*WATT["f_gt_248keV_recoil"]:.3f}-{tot["N_SF_hi"]*NU_BAR*WATT["f_gt_248keV_recoil"]:.3f})')

# route B: (alpha,n) budget back-calculation with P013 transport factors
say('\n--- route B: (alpha,n)-budget back-calculation ---')
UNTAGGED = 1.0 - lz.LZ['neutron_veto_eff'][0]          # 0.08
comb = {(r['source'], r['thr_keV'], r['fv']): r for r in P13['combined']}
p_ss_roi_an = comb[('alpha_n (f_C=2.5e-3)', 3.0, 'fv63x131')]['p_ss_roi']      # per neutron entering LXe
p_ss_roi_sf = comb[('fission_238U', 3.0, 'fv63x131')]['p_ss_roi']
p_win_sf = comb[('fission_238U', 3.0, 'fv63x131')]['p_ss_200_270']           # per neutron entering LXe
p_win_sf_rng = [comb[('fission_238U', t, f)]['p_ss_200_270'] for t in (1.0, 3.0, 5.0) for f in ('fv63x131', 'fv60x115')]
say(f'P013 MC (thr 3 keV, FV 63x131): p_ss_roi(alpha,n) = {p_ss_roi_an:.4f}, p_ss_roi(SF) = {p_ss_roi_sf:.4f}, '
    f'p_win(SF, 200-270) = {p_win_sf:.3e} (range {min(p_win_sf_rng):.2e}-{max(p_win_sf_rng):.2e})')
Y_AN_PTFE = dict(c=1.0e-4, lo=5e-5, hi=2e-4)        # (alpha,n) neutrons per 238U-chain (late, 7 alphas) decay in PTFE; recalled uncertain
R_LATE_EARLY = dict(c=1.0, lo=0.3, hi=3.0)           # A(226Ra chain)/A(238U) in PTFE; assumed
routeB = {}
for label, ss in (('fit UL 0.118', lz.LZ['detector_NR_fit_interval'][1]), ('MS cross-check 0.02', 0.02), ('MS cross-check +1sigma 0.04', 0.04)):
    pre_veto = ss / UNTAGGED
    n_enter = pre_veto / p_ss_roi_an
    n_emit = n_enter / 0.5                           # wall sources: half the neutrons go inward
    r_sf_an = BR_SF * NU_BAR / (Y_AN_PTFE['c'] * R_LATE_EARLY['c'])
    r_lo = BR_SF * NU_BAR / (Y_AN_PTFE['hi'] * R_LATE_EARLY['hi']); r_hi = BR_SF * NU_BAR / (Y_AN_PTFE['lo'] * R_LATE_EARLY['lo'])
    n_sf_neu = n_emit * r_sf_an
    routeB[label] = dict(SS_untagged=ss, SS_pre_veto=pre_veto, n_enter_LXe=n_enter, n_emitted=n_emit,
                         SF_over_alpha_n=r_sf_an, SF_over_alpha_n_range=[r_lo, r_hi],
                         N_SF_neutrons=n_sf_neu, N_SF=n_sf_neu / NU_BAR, N_SF_range=[n_emit * r_lo / NU_BAR, n_emit * r_hi / NU_BAR])
    say(f'{label}: pre-veto SS {pre_veto:.2f} -> {n_enter:.0f} (alpha,n) neutrons entering LXe -> {n_emit:.0f} emitted; '
        f'SF/(alpha,n) neutron ratio {r_sf_an:.2e} ({r_lo:.1e}-{r_hi:.1e}) -> SF neutrons {n_sf_neu:.2f}, fissions {n_sf_neu/NU_BAR:.2f} '
        f'({n_emit*r_lo/NU_BAR:.3f}-{n_emit*r_hi/NU_BAR:.2f})  [PTFE-like (alpha,n) emitters only]')
say('Route B bounds only the SF in (alpha,n)-producing (fluorine-bearing) components; Ti and PMT glass have negligible (alpha,n) yield')
say(f'Radioassay route A total ({tot["N_SF"]:.1f} SF) vs route-B PTFE share ({routeB["fit UL 0.118"]["N_SF"]:.2f}): '
    f'route A / (alpha,n) neutrons = {n_sf_neutrons/routeB["fit UL 0.118"]["n_emitted"]:.2f} (LZ: "up to two orders of magnitude lower")')

# route C: TPC multiple-scatter sample cap (recalled tolerance: <= 3 unexplained MS NR events, 1-5)
P_MS_GIVEN_ENTER = dict(c=0.5, lo=0.3, hi=0.7)      # P(>=2 deposits > 3 keV in the active LXe | neutron enters); estimated
def n_sf_from_ms(n_allowed, p_ms, p_enter=0.4):
    # per fission: P(>=1 MS event) = 1 - sum_nu P(nu) (1 - p_enter*p_ms)^nu
    q = 1 - p_enter * p_ms
    p1 = 1 - (P_NU * q ** nus).sum()
    return n_allowed / p1, p1
routeC = {}
for lab, nal in (('<=1 MS', 1), ('<=3 MS', 3), ('<=5 MS', 5)):
    nsf, p1 = n_sf_from_ms(nal, P_MS_GIVEN_ENTER['c'])
    routeC[lab] = dict(N_MS_allowed=nal, P_MS_per_SF=p1, N_SF_max=nsf,
                       N_SF_max_range=[n_sf_from_ms(nal, P_MS_GIVEN_ENTER['hi'])[0], n_sf_from_ms(nal, P_MS_GIVEN_ENTER['lo'])[0]])
    say(f'route C {lab} NR multiple scatters (any veto tag) -> P(MS per SF) = {p1:.3f} -> N_SF <= {nsf:.1f} '
        f'({routeC[lab]["N_SF_max_range"][0]:.1f}-{routeC[lab]["N_SF_max_range"][1]:.1f})')

# ----------------------------------------------------------------------------------------------
# 3. Prompt-gamma transport MC: per-gamma "silent" probability by source location
# ----------------------------------------------------------------------------------------------
say('\n=== 3. Prompt-gamma transport (simplified LZ geometry) ===')
# recalled mass attenuation coefficients (cm2/g), NIST-XCOM-like, +-20%: E (MeV) grid
E_GRID = np.array([0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0])
MU_RHO = {
 'LXe':   np.array([1.70, 0.30, 0.16, 0.088, 0.066, 0.058, 0.047, 0.040, 0.033, 0.028, 0.026]),   # xenon
 'LS':    np.array([0.170, 0.137, 0.119, 0.097, 0.079, 0.071, 0.058, 0.050, 0.040, 0.031, 0.026]), # LAB, water-like
 'Ti':    np.array([0.272, 0.131, 0.107, 0.081, 0.066, 0.059, 0.048, 0.042, 0.035, 0.030, 0.028]),
 'quartz':np.array([0.168, 0.125, 0.108, 0.087, 0.071, 0.064, 0.052, 0.045, 0.036, 0.029, 0.026]), # SiO2 ~ PMT arrays
 'water': np.array([0.171, 0.137, 0.119, 0.097, 0.079, 0.071, 0.058, 0.049, 0.040, 0.030, 0.026]),
}
RHO = dict(LXe=2.9, LS=0.86, Ti=4.5, quartz=1.0, water=1.0)      # quartz: PMT arrays as 1 g/cc effective (glass+Kovar+voids)
# region ids
VAC, TPC, SKIN, PMTA, TI, OD, WATER = 0, 1, 2, 3, 4, 5, 6
REGION_MAT = {TPC: 'LXe', SKIN: 'LXe', PMTA: 'quartz', TI: 'Ti', OD: 'LS', WATER: 'water'}
# visibility: LXe regions always visible (any interaction >= few keV gives S1/Skin light); OD visible with prob V_OD;
# Ti, PMT arrays, water: invisible absorbers (conservative: an interaction there kills the gamma silently)
GEO = dict(R_TPC=72.8, H_TPC=145.6, R_SKIN=79.0, Z_LXE_LO=-20.0, Z_LXE_HI=150.0, PMT_T=15.0, DOME_T=15.0,
           R_ICV=80.0, T_TI=0.8, R_OCV=92.0, R_OD_IN=95.0, T_OD_SIDE=61.0, T_OD_TOPBOT=30.0,
           Z_ICV_LO=-60.0, Z_ICV_HI=170.0, Z_OD_LO=-80.0, Z_OD_HI=190.0)
# Z_LXE_LO..0: reverse-field region + Skin LXe below the cathode; bottom PMT array in (Z_LXE_LO-PMT_T, Z_LXE_LO);
# LXe "dome" viewed by Skin PMTs below the bottom array (DOME_T); top array in gas above the liquid surface.

def region(x, y, z):
    r = np.hypot(x, y)
    g = GEO
    reg = np.full(x.shape, WATER, dtype=np.int8)
    # OD gadolinium-loaded scintillator: side annulus and top/bottom slabs
    side = (r >= g['R_OD_IN']) & (r < g['R_OD_IN'] + g['T_OD_SIDE']) & (z > g['Z_OD_LO'] - g['T_OD_TOPBOT']) & (z < g['Z_OD_HI'] + g['T_OD_TOPBOT'])
    topbot = (r < g['R_OD_IN']) & (((z >= g['Z_OD_HI']) & (z < g['Z_OD_HI'] + g['T_OD_TOPBOT'])) | ((z <= g['Z_OD_LO']) & (z > g['Z_OD_LO'] - g['T_OD_TOPBOT'])))
    reg[side | topbot] = OD
    # inside the OD inner boundary: vacuum by default
    inside = (r < g['R_OD_IN']) & (z > g['Z_OD_LO']) & (z < g['Z_OD_HI'])
    reg[inside] = VAC
    # Ti vessels: two thin shells (cylindrical walls + end caps)
    for Rv, zlo, zhi in ((g['R_ICV'], g['Z_ICV_LO'], g['Z_ICV_HI']), (g['R_OCV'], g['Z_ICV_LO'] - 10, g['Z_ICV_HI'] + 10)):
        wall = (r >= Rv) & (r < Rv + g['T_TI']) & (z > zlo) & (z < zhi)
        cap = (r < Rv + g['T_TI']) & (((z >= zhi) & (z < zhi + g['T_TI'])) | ((z <= zlo) & (z > zlo - g['T_TI'])))
        reg[wall | cap] = TI
    # PMT arrays (top and bottom), inside the ICV
    pmt = (r < g['R_SKIN']) & (((z >= g['Z_LXE_HI']) & (z < g['Z_LXE_HI'] + g['PMT_T'])) | ((z <= g['Z_LXE_LO']) & (z > g['Z_LXE_LO'] - g['PMT_T'])))
    reg[pmt] = PMTA
    # Skin / RFR LXe (everything in r<R_SKIN, Z_LXE_LO<z<Z_LXE_HI that is not TPC) and the LXe dome below the bottom array
    lxe = (r < g['R_SKIN']) & (z > g['Z_LXE_LO']) & (z < g['Z_LXE_HI'])
    dome = (r < g['R_SKIN']) & (z <= g['Z_LXE_LO'] - g['PMT_T']) & (z > g['Z_LXE_LO'] - g['PMT_T'] - g['DOME_T'])
    reg[lxe | dome] = SKIN
    tpc = (r < g['R_TPC']) & (z > 0) & (z < g['H_TPC'])
    reg[tpc] = TPC
    return reg

def mu_lin(mat, E):
    # log-log interpolation of the linear attenuation coefficient (cm^-1)
    return np.exp(np.interp(np.log(E), np.log(E_GRID), np.log(MU_RHO[mat] * RHO[mat])))

def sample_gamma_E(n):
    # fission prompt-gamma spectrum ~ exponential, mean 0.9 MeV, truncated to [0.1, 7] MeV (recalled shape, uncertain)
    E = rng.exponential(EG_MEAN, size=3 * n)
    E = E[(E > 0.1) & (E < 7.0)]
    while E.size < n:
        e2 = rng.exponential(EG_MEAN, size=n); E = np.concatenate([E, e2[(e2 > 0.1) & (e2 < 7.0)]])
    return E[:n]

def source_positions(loc, n):
    g = GEO
    phi = rng.uniform(0, 2 * np.pi, n)
    if loc == 'wall':            # PTFE panels between TPC and Skin
        r = np.full(n, g['R_TPC'] + 0.6); z = rng.uniform(0, g['H_TPC'], n)
    elif loc == 'cryostat':      # inner cryostat vessel wall (mid-thickness)
        r = np.full(n, g['R_ICV'] + 0.4); z = rng.uniform(g['Z_ICV_LO'] + 5, g['Z_ICV_HI'] - 5, n)
    elif loc == 'pmt':           # PMT arrays: 238U in the PMT bodies, uniform over the disc and the array thickness, top or bottom
        r = g['R_TPC'] * np.sqrt(rng.uniform(0, 1, n))
        top = rng.uniform(0, 1, n) < 0.5
        depth = rng.uniform(0.5, g['PMT_T'] - 0.5, n)
        z = np.where(top, g['Z_LXE_HI'] + depth, g['Z_LXE_LO'] - depth)
    elif loc == 'ocv':           # outer cryostat vessel (cables/conduits in the vacuum space ~ similar)
        r = np.full(n, g['R_OCV'] + 0.4); z = rng.uniform(g['Z_ICV_LO'], g['Z_ICV_HI'], n)
    else:
        raise ValueError(loc)
    return r * np.cos(phi), r * np.sin(phi), z

def f_photoelectric(mat, E):
    # recalled, uncertain: photoelectric share of the total attenuation (fraction); Ti (Z=22) and SiO2-like PMT material
    if mat == 'Ti':
        return np.interp(np.log(E), np.log([0.1, 0.2, 0.3, 0.5, 1.0, 3.0]), [0.55, 0.15, 0.06, 0.02, 0.005, 0.001])
    return np.interp(np.log(E), np.log([0.1, 0.2, 0.3, 0.5, 1.0, 3.0]), [0.05, 0.01, 0.004, 0.001, 3e-4, 1e-4])

def gamma_mc(loc, n=60000, v_od=0.9, pmt_visible=0.0, invisible_absorb=False, ds=0.5, smax=420.0):
    """Sampled straight-line transport of one prompt gamma. Returns (silent flag array, initial energies).
    Visible regions: TPC and Skin LXe (any interaction is visible), OD GdLS (interaction visible with prob v_od,
    i.e. deposit above the 4.5 phd prompt threshold), PMT arrays (visible with prob pmt_visible; default 0).
    Invisible regions (Ti vessels, PMT arrays, water): an interaction is a silent photoelectric absorption with the
    recalled photoelectric fraction, otherwise a Compton scatter after which the gamma continues along its line with
    a sampled reduced energy (direction change neglected). invisible_absorb=True reproduces the over-conservative
    'every interaction in an invisible region kills the gamma silently' bookkeeping (used as a bracket)."""
    x, y, z = source_positions(loc, n)
    E0 = sample_gamma_E(n); E = E0.copy()
    ct = rng.uniform(-1, 1, n); st = np.sqrt(1 - ct ** 2); ph = rng.uniform(0, 2 * np.pi, n)
    dx, dy, dz = st * np.cos(ph), st * np.sin(ph), ct
    alive = np.ones(n, bool)          # still tracking (no visible deposit, not absorbed, not escaped)
    visible = np.zeros(n, bool)
    vis = {TPC: 1.0, SKIN: 1.0, OD: v_od, PMTA: pmt_visible, TI: 0.0, WATER: 0.0}
    nstep = int(smax / ds)
    for _ in range(nstep):
        if not alive.any():
            break
        idx = np.nonzero(alive)[0]
        x[idx] += dx[idx] * ds; y[idx] += dy[idx] * ds; z[idx] += dz[idx] * ds
        reg = region(x[idx], y[idx], z[idx])
        # kill gammas that left the water tank envelope (r > 230 cm or |z| > 300): escaped, silent
        far = (np.hypot(x[idx], y[idx]) > 230) | (z[idx] > 300) | (z[idx] < -300)
        alive[idx[far]] = False
        for rid in (TPC, SKIN, PMTA, TI, OD, WATER):
            m = (reg == rid) & ~far
            if not m.any():
                continue
            ii = idx[m]
            pint = 1 - np.exp(-mu_lin(REGION_MAT[rid], E[ii]) * ds)
            hit = rng.uniform(0, 1, ii.size) < pint
            if not hit.any():
                continue
            jj = ii[hit]
            v = vis[rid]
            isvis = rng.uniform(0, 1, jj.size) < v
            visible[jj[isvis]] = True; alive[jj[isvis]] = False
            rest = jj[~isvis]
            if rest.size == 0:
                continue
            if rid == OD:
                continue                       # sub-threshold OD deposit: gamma continues (approximation)
            if invisible_absorb or rid == WATER:
                alive[rest] = False            # silently absorbed / lost in the water
                continue
            # invisible material (Ti, PMT arrays): photoelectric -> silent absorption; Compton -> continue, degraded
            fpe = f_photoelectric(REGION_MAT[rid], E[rest])
            pe = rng.uniform(0, 1, rest.size) < fpe
            alive[rest[pe]] = False
            cs = rest[~pe]
            if cs.size:
                emin = E[cs] / (1 + 2 * E[cs] / 0.511)        # backscatter energy
                E[cs] = rng.uniform(emin, E[cs])              # crude: uniform between backscatter and forward
    return ~visible, E0

LOCS = ['wall', 'cryostat', 'pmt', 'ocv']
gamma_res = {}
for loc in LOCS:
    ps, E = gamma_mc(loc)
    ps = ps.astype(float)
    gamma_res[loc] = dict(p_gamma_silent=float(ps.mean()), p_gamma_silent_err=float(ps.std() / np.sqrt(ps.size)),
                          p_gamma_silent_Egt1MeV=float(ps[E > 1.0].mean()), p_gamma_silent_Elt0p5MeV=float(ps[E < 0.5].mean()))
    say(f'{loc:9s}: <p_gamma silent> = {ps.mean():.4f} +- {ps.std()/np.sqrt(ps.size):.4f} (E>1 MeV {ps[E>1].mean():.4f}; E<0.5 MeV {ps[E<0.5].mean():.4f})')
# variants: OD visibility 0.7 and 1.0; PMT arrays partially visible; over-conservative 'invisible regions absorb'
VAR = {}
for loc in ('wall', 'cryostat', 'pmt'):
    for v_od in (0.7, 1.0):
        ps, _ = gamma_mc(loc, n=30000, v_od=v_od); VAR[(loc, f'v_od={v_od}')] = float(ps.mean())
    ps, _ = gamma_mc(loc, n=30000, pmt_visible=0.5); VAR[(loc, 'pmt_arrays_half_visible')] = float(ps.mean())
    ps, _ = gamma_mc(loc, n=30000, invisible_absorb=True); VAR[(loc, 'invisible_regions_absorb')] = float(ps.mean())
for k, v in VAR.items():
    say(f'  variant {k}: <p_gamma silent> = {v:.4f}')

# f_gamma = sum_k P(k) p^k  (independent gammas; p = mean per-gamma silent prob.) -- computed properly as
# E[ prod_i p_i ] = E[p]^k only if independent and identical; use the mean p (angular correlations ignored)
def f_gamma(p, model):
    k, pk = GAMMA_MODELS[model]; return float((pk * p ** k).sum())
FG = []
for loc in LOCS:
    p = gamma_res[loc]['p_gamma_silent']
    for model in GAMMA_MODELS:
        FG.append(dict(location=loc, model=model, p_gamma=p, f_gamma=f_gamma(p, model)))
# comparison with P013's e^{-7(1-0.15)}
FG.append(dict(location='P013 (p=0.15, Poisson 7)', model='Poisson(7)', p_gamma=0.15, f_gamma=math.exp(-7 * 0.85)))
fg = pd.DataFrame(FG); fg.to_csv(os.path.join(OUT, 'P063_f_gamma.csv'), index=False)
say('\nf_gamma (all prompt gammas silent):'); say(fg.to_string(index=False))

# ----------------------------------------------------------------------------------------------
# 4. Neutron companions, per-fission lone-recoil probability, expected counts
# ----------------------------------------------------------------------------------------------
say('\n=== 4. Lone-recoil probability per fission and expected counts ===')
P_N_SILENT = dict(c=0.10, lo=0.05, hi=0.20)      # per companion neutron: no TPC/Skin/OD prompt deposit and no capture tag; recalled/estimated
U_SELF = dict(c=0.10, lo=0.08, hi=0.20)          # the window neutron itself escapes untagged after its single scatter (LZ 8% (alpha,n), 20% rock n)
def f_neutrons(p_n, pnu=P_NU):
    # per emitted neutron: probability that the other nu-1 neutrons are all silent = sum_nu P(nu) nu p^(nu-1) / nu_bar
    nb = (nus * pnu).sum()
    return float((pnu * nus * p_n ** np.clip(nus - 1, 0, None)).sum() / nb)
FN = {k: f_neutrons(v) for k, v in P_N_SILENT.items()}
say(f'f_n (other neutrons silent, per emitted neutron): central {FN["c"]:.3f}, p_n=0.05: {FN["lo"]:.3f}, p_n=0.20: {FN["hi"]:.3f}; '
    f'with P013 P(nu) and p_n=0.07: {f_neutrons(0.07, P_NU_P013):.3f} (P013: 0.148)')

# fraction of fissions with exactly one neutron (the only way to avoid the neutron-companion penalty entirely)
say(f'P(nu=1) = {P_NU[1]:.3f}; P(nu=0) = {P_NU[0]:.3f}; share of emitted neutrons that are "lone" (nu=1): {P_NU[1]/NU_BAR:.3f}')

def lone_prob(p_enter, f_n, f_g, p_win=p_win_sf, u_self=U_SELF['c']):
    # per fission: nu_bar * p_enter * P(clean SS 200-270 in FV | enter) * P(self untagged) * f_n * f_gamma  (fragments: silent, see text)
    return NU_BAR * p_enter * p_win * u_self * f_n * f_g

GAMMA_CENTRAL = 'narrow G(6.5,2.5) k>=1'
res_rows = []
for _, r in inv.iterrows():
    loc = r.location
    fgP = f_gamma(gamma_res[loc]['p_gamma_silent'], 'Poisson(6.5)')
    fgN = f_gamma(gamma_res[loc]['p_gamma_silent'], GAMMA_CENTRAL)
    pl_c = lone_prob(r.p_enter, FN['c'], fgN)
    pl_P = lone_prob(r.p_enter, FN['c'], fgP)
    no_comp = NU_BAR * r.p_enter * p_win_sf * U_SELF['c']       # ignoring companions (nu-1 neutrons and gammas)
    res_rows.append(dict(component=r.component, location=loc, N_SF=r.N_SF_220d, p_enter=r.p_enter,
                         p_gamma=gamma_res[loc]['p_gamma_silent'], f_gamma_narrow=fgN, f_gamma_poisson=fgP,
                         f_n=FN['c'], P_lone_per_SF=pl_c, P_lone_per_SF_poissonG=pl_P,
                         N_lone_220d=r.N_SF_220d * pl_c, N_lone_220d_poissonG=r.N_SF_220d * pl_P,
                         N_lone_no_companion_penalty=r.N_SF_220d * no_comp))
res = pd.DataFrame(res_rows); res.to_csv(os.path.join(OUT, 'P063_expected_by_component.csv'), index=False)
say(res[['component', 'N_SF', 'p_gamma', 'f_gamma_narrow', 'f_gamma_poisson', 'P_lone_per_SF', 'N_lone_220d', 'N_lone_220d_poissonG', 'N_lone_no_companion_penalty']].to_string(index=False))
N_central = res.N_lone_220d.sum(); N_poisson = res.N_lone_220d_poissonG.sum(); N_nocomp = res.N_lone_no_companion_penalty.sum()
silent_eff = N_central / N_nocomp
say(f'TOTAL expected lone 200-270 keV SF recoils in 220 d: {N_central:.2e} (narrow gamma multiplicity), {N_poisson:.2e} (Poisson); '
    f'no-companion bound {N_nocomp:.2e}; effective silent-companion factor {silent_eff:.2e} (P013: 3.8e-4)')
say(f'P013 estimate 9.3e-11 -> ratio this/P013 = {N_central/9.268e-11:.2f} (narrow) / {N_poisson/9.268e-11:.2f} (Poisson)')

# sensitivity / stacked scenarios
def total_expected(sf_scale=1.0, ti_hi=False, pmt_hi=False, p_n='c', u='c', gmodel=GAMMA_CENTRAL, v_od_key=None, p_win=p_win_sf, pg_floor=None):
    tot_ = 0.0
    for _, r in inv.iterrows():
        nsf = r.N_SF_220d
        if ti_hi and r.location == 'cryostat': nsf = r.N_SF_hi
        if pmt_hi and r.location == 'pmt': nsf = r.N_SF_hi
        pg = gamma_res[r.location]['p_gamma_silent']
        if v_od_key is not None and (r.location, v_od_key) in VAR: pg = VAR[(r.location, v_od_key)]
        if pg_floor is not None: pg = max(pg, pg_floor)
        tot_ += nsf * sf_scale * lone_prob(r.p_enter, FN[p_n], f_gamma(pg, gmodel), p_win=p_win, u_self=U_SELF[u])
    return tot_
SCEN = [
 ('central (route A inventory, narrow gamma mult., p_n=0.1, u=0.1)', dict()),
 ('gamma multiplicity Poisson(6.5)', dict(gmodel='Poisson(6.5)')),
 ('gamma multiplicity narrow k>=2', dict(gmodel='narrow G(6.5,2.5) k>=2')),
 ('p_n = 0.20', dict(p_n='hi')), ('p_n = 0.05', dict(p_n='lo')),
 ('self-untagged 0.20', dict(u='hi')),
 ('OD visibility 0.7', dict(v_od_key='v_od=0.7')), ('OD visibility 1.0', dict(v_od_key='v_od=1.0')),
 ('Ti/PMT-array interactions all silent (over-conservative)', dict(v_od_key='invisible_regions_absorb')),
 ('PMT arrays half visible (Skin light)', dict(v_od_key='pmt_arrays_half_visible')),
 ('per-gamma silent floor 0.05 (unmodelled gaps)', dict(pg_floor=0.05)),
 ('per-gamma silent floor 0.15 (P013 value)', dict(pg_floor=0.15)),
 ('p_win at MC maximum', dict(p_win=max(p_win_sf_rng))), ('p_win at MC minimum', dict(p_win=min(p_win_sf_rng))),
 ('Ti and PMTs at assay upper limits', dict(ti_hi=True, pmt_hi=True)),
 ('inventory at all lower brackets', dict(sf_scale=tot['N_SF_lo'] / tot['N_SF'])),
 ('inventory at all upper brackets', dict(sf_scale=tot['N_SF_hi'] / tot['N_SF'])),
 ('route C cap (<=3 MS): N_SF = %.1f' % routeC['<=3 MS']['N_SF_max'], dict(sf_scale=routeC['<=3 MS']['N_SF_max'] / tot['N_SF'])),
 ('route B PTFE-only (fit UL): N_SF = %.2f' % routeB['fit UL 0.118']['N_SF'], dict(sf_scale=routeB['fit UL 0.118']['N_SF'] / tot['N_SF'])),
 ('STACKED HIGH: upper inventory, Poisson gammas, p_gamma floor 0.15, p_n 0.2, u 0.2, p_win max',
  dict(sf_scale=tot['N_SF_hi'] / tot['N_SF'], gmodel='Poisson(6.5)', pg_floor=0.15, p_n='hi', u='hi', p_win=max(p_win_sf_rng))),
 ('STACKED LOW: lower inventory, narrow k>=2, p_n 0.05, u 0.08, p_win min',
  dict(sf_scale=tot['N_SF_lo'] / tot['N_SF'], gmodel='narrow G(6.5,2.5) k>=2', p_n='lo', u='lo', p_win=min(p_win_sf_rng))),
]
sens = pd.DataFrame([dict(scenario=s, N_lone_220d=total_expected(**kw)) for s, kw in SCEN])
sens['ratio_to_central'] = sens.N_lone_220d / N_central
sens.to_csv(os.path.join(OUT, 'P063_sensitivity.csv'), index=False)
say('\nSensitivity:'); say(sens.to_string(index=False))

# ----------------------------------------------------------------------------------------------
# 5. Detectability of SF bursts
# ----------------------------------------------------------------------------------------------
say('\n=== 5. What SF bursts look like in LZ ===')
det = {}
# per fission: P(>=1 visible prompt gamma) by location
for loc in LOCS:
    p = gamma_res[loc]['p_gamma_silent']
    det[loc] = dict(P_ge1_visible_gamma=1 - f_gamma(p, GAMMA_CENTRAL),
                    mean_visible_gammas=NG_BAR * (1 - p))
# neutrons: P(>=1 TPC deposit), P(>=1 OD delayed capture), P(TPC NR + OD capture coincidence) per fission
def per_fission_neutron_sig(p_enter, p_dep=0.87, p_cap=0.6):
    # p_dep: P(any deposit > 3 keV in active LXe | enter)  (P013 MC 'watt_bulk' n_anydep/N)
    # p_cap: P(a neutron ends with a tagged OD/Skin capture within 600 us) (recalled/estimated 0.5-0.8)
    q_tpc = 1 - p_enter * p_dep
    P_tpc = 1 - (P_NU * q_tpc ** nus).sum()
    P_cap = 1 - (P_NU * (1 - p_cap) ** nus).sum()
    P_2cap = 1 - (P_NU * ((1 - p_cap) ** nus + nus * p_cap * (1 - p_cap) ** np.clip(nus - 1, 0, None))).sum()
    return dict(P_ge1_TPC_deposit=float(P_tpc), P_ge1_capture=float(P_cap), P_ge2_captures=float(P_2cap))
p_dep_watt = P13['MC']['samples']['watt_bulk']['thr3_fv63x131']['n_anydep'] / P13['MC']['samples']['watt_bulk']['thr3_fv63x131']['N']
burst_rows = []
for _, r in inv.iterrows():
    sig = per_fission_neutron_sig(r.p_enter, p_dep_watt)
    burst_rows.append(dict(component=r.component, N_SF=r.N_SF_220d, **sig,
                           N_TPC_deposit_events=r.N_SF_220d * sig['P_ge1_TPC_deposit'],
                           N_MS_events=r.N_SF_220d * (1 - (P_NU * (1 - r.p_enter * P_MS_GIVEN_ENTER['c']) ** nus).sum()),
                           N_TPC_and_ge1_capture=r.N_SF_220d * sig['P_ge1_TPC_deposit'] * sig['P_ge1_capture'],
                           N_ge2_captures=r.N_SF_220d * sig['P_ge2_captures'],
                           P_ge1_visible_gamma=det[r.location]['P_ge1_visible_gamma']))
burst = pd.DataFrame(burst_rows); burst.to_csv(os.path.join(OUT, 'P063_burst_signatures.csv'), index=False)
say(burst[['component', 'N_SF', 'P_ge1_TPC_deposit', 'P_ge1_capture', 'P_ge2_captures', 'N_TPC_deposit_events', 'N_MS_events', 'N_ge2_captures', 'P_ge1_visible_gamma']].to_string(index=False))
say(f'TOTAL (route A): SF with >=1 TPC deposit {burst.N_TPC_deposit_events.sum():.1f}; TPC multiple scatters {burst.N_MS_events.sum():.1f}; '
    f'>=2 OD/Skin captures {burst.N_ge2_captures.sum():.1f} in 220 d; per-fission P(>=1 visible prompt gamma) >= {min(d["P_ge1_visible_gamma"] for d in det.values()):.4f}')
# (alpha,n) comparison: MS from (alpha,n) pre-veto
an_ms = routeB['fit UL 0.118']['n_enter_LXe'] * P_MS_GIVEN_ENTER['c']
say(f'(alpha,n) at the fit UL: {routeB["fit UL 0.118"]["n_enter_LXe"]:.0f} neutrons entering LXe -> ~{an_ms:.0f} MS events (pre-veto); '
    f'at the MS cross-check 0.02: {routeB["MS cross-check 0.02"]["n_enter_LXe"]*P_MS_GIVEN_ENTER["c"]:.0f} MS events')

# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------
C1, C2, C3, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e6e6e6', 'grid.linewidth': 0.6, 'axes.axisbelow': True})
# Fig 1: per-gamma silent probability and f_gamma by location
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.4))
locs_lab = {'wall': 'PTFE wall', 'cryostat': 'Ti inner cryostat', 'pmt': 'PMT arrays', 'ocv': 'outer cryostat / cables'}
xs = np.arange(len(LOCS))
pg = [gamma_res[l]['p_gamma_silent'] for l in LOCS]
axs[0].bar(xs, pg, color=C1, width=0.6)
axs[0].axhline(0.15, color=C2, lw=1.2, ls='--'); axs[0].text(-0.3, 0.153, 'P013 assumption 0.15 (all locations)', color=C2, ha='left', fontsize=8)
axs[0].set_ylim(0, 0.175)
axs[0].set_xticks(xs); axs[0].set_xticklabels([locs_lab[l] for l in LOCS], rotation=15, ha='right')
axs[0].set_ylabel('per-gamma P(silent)'); axs[0].set_title('a) one prompt gamma leaves no visible deposit', fontsize=9, loc='left')
for x_, v in zip(xs, pg): axs[0].text(x_, v + 0.003, f'{v:.3f}', ha='center', fontsize=8)
w = 0.27
for i, (model, col) in enumerate(zip(GAMMA_MODELS, (C1, C3, C4))):
    vals = [f_gamma(gamma_res[l]['p_gamma_silent'], model) for l in LOCS]
    axs[1].bar(xs + (i - 1) * w, vals, width=w, color=col, label=model)
axs[1].axhline(math.exp(-7 * 0.85), color=C2, lw=1.2, ls='--', label='P013: exp(-7 x 0.85) = 2.6e-3')
axs[1].set_yscale('log'); axs[1].set_ylim(1e-7, 3.0)
axs[1].set_xticks(xs); axs[1].set_xticklabels([locs_lab[l] for l in LOCS], rotation=15, ha='right')
axs[1].set_ylabel('f_gamma = P(all prompt gammas silent)'); axs[1].set_title('b) all ~6.5 prompt gammas silent', fontsize=9, loc='left')
axs[1].legend(fontsize=7, frameon=False, loc='upper right')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P063_fig1_silent_gamma.png'), dpi=160); plt.close(fig)

# Fig 2: expected lone-recoil counts by component and scenario
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.6), gridspec_kw=dict(width_ratios=[1, 1.5]))
comp_lab = ['PTFE', 'Ti cryostat', 'PMTs', 'cables/misc']
axs[0].bar(np.arange(4) - 0.2, res.N_lone_220d_poissonG, width=0.4, color=C1, label='Poisson gamma multiplicity')
axs[0].bar(np.arange(4) + 0.2, res.N_lone_220d, width=0.4, color=C3, label='narrow multiplicity (k>=1)')
axs[0].set_yscale('log'); axs[0].set_xticks(np.arange(4)); axs[0].set_xticklabels(comp_lab, rotation=15, ha='right')
axs[0].set_ylabel('expected lone 200-270 keV recoils, 220 d'); axs[0].set_title('a) by component (route A inventory)', fontsize=9, loc='left')
axs[0].legend(fontsize=7, frameon=False)
keys = ['central', 'Poisson(6.5)', 'p_n = 0.20', 'self-untagged', 'over-conservative', 'floor 0.15', 'p_win at MC maximum',
        'Ti and PMTs at assay', 'upper brackets', 'route C', 'route B', 'STACKED HIGH', 'STACKED LOW']
sel = sens[sens.scenario.apply(lambda s: any(k in s for k in keys))]
ypos = np.arange(len(sel))[::-1]
axs[1].barh(ypos, sel.N_lone_220d, color=[C2 if 'STACKED HIGH' in s else (C4 if 'STACKED LOW' in s else C1) for s in sel.scenario], height=0.6)
axs[1].set_xscale('log'); axs[1].set_xlim(1e-14, 3)
axs[1].axvline(9.268e-11, color=C3, lw=1.2, ls='--'); axs[1].text(1.1e-10, len(sel) - 0.6, 'P013: 9e-11', color=C3, fontsize=8)
axs[1].axvline(1.0, color='k', lw=1.0); axs[1].text(0.5, len(sel) - 0.6, 'one event', fontsize=8, ha='right')
axs[1].set_yticks(ypos); axs[1].set_yticklabels([s.split(':')[0][:46] for s in sel.scenario], fontsize=7)
axs[1].set_xlabel('expected lone 200-270 keV recoils in 220 d'); axs[1].set_title('b) scenarios', fontsize=9, loc='left')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P063_fig2_expected.png'), dpi=160); plt.close(fig)

# ----------------------------------------------------------------------------------------------
# Save results JSON
# ----------------------------------------------------------------------------------------------
results = dict(
    nuclear=dict(BR_SF=BR_SF, T12_U238_y=T12_U238_y, nu_bar=NU_BAR, sigma_nu=SIG_NU, P_nu=P_NU.tolist(), P_nu_P013=P_NU_P013.tolist(),
                 watt=WATT, E_n_min_248=E_MIN_248, E_n_min_200=E_MIN_200, n_gamma_bar=NG_BAR, E_gamma_mean=EG_MEAN,
                 gamma_models={k: dict(P0=float(v[1][0]), P1=float(v[1][1]), P2=float(v[1][2])) for k, v in GAMMA_MODELS.items()}),
    inventory=inv.to_dict(orient='records'), inventory_total=tot,
    SF_neutrons_220d=n_sf_neutrons, SF_neutrons_above_8p2MeV_220d=n_sf_neutrons * WATT['f_gt_248keV_recoil'],
    routeB=routeB, routeC=routeC, P013_transport=dict(p_ss_roi_alpha_n=p_ss_roi_an, p_ss_roi_SF=p_ss_roi_sf, p_win_SF=p_win_sf,
                                                    p_win_SF_range=[min(p_win_sf_rng), max(p_win_sf_rng)], untagged=UNTAGGED),
    gamma_transport=dict(geometry=GEO, per_location=gamma_res, variants={f'{k[0]}|{k[1]}': v for k, v in VAR.items()}),
    f_gamma=FG, f_neutrons=FN, p_n_silent=P_N_SILENT, u_self=U_SELF,
    expected=dict(central=N_central, poisson_gamma=N_poisson, no_companion_penalty=N_nocomp, effective_silent_factor=silent_eff,
                  ratio_to_P013=N_central / 9.268e-11, by_component=res.to_dict(orient='records'),
                  sensitivity=sens.to_dict(orient='records')),
    bursts=dict(per_location=det, by_component=burst.to_dict(orient='records'), p_dep_watt=p_dep_watt, P_MS_given_enter=P_MS_GIVEN_ENTER),
)
json.dump(results, open(os.path.join(OUT, 'P063_results.json'), 'w'), indent=1, default=float)
open(os.path.join(OUT, 'P063_console.txt'), 'w').write('\n'.join(LOG) + '\n')
say('\nsaved', os.path.join(OUT, 'P063_results.json'))
