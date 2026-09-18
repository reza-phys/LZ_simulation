"""
P065 -- Magnetic inelastic dark matter (MiDM) at LZ: transition-dipole scattering spectrum, de-excitation photon,
        surviving parameter space, relic density and multi-target predictions.

Run from the simulation root:   .venv/bin/python output/code/P065_midm.py [--budget SECONDS] [--recompute]
The WimPyDD spectra are cached in output/work/P065/P065_spectra_cache.npz; the script computes missing spectra
until the time budget is used up (default 540 s), then exits asking to be re-run.  Once every spectrum is cached
the analysis and figures run (a few seconds).

Physics
  MiDM (Chang, Weiner, Yavin 2010): Majorana pair chi1, chi2 split by delta, transition magnetic moment mu_chi
  coupling to the photon, L = (mu_chi/2) chi2bar sigma^{mu nu} chi1 F_{mu nu} + h.c.  Scattering chi1 N -> chi2 N is
  photon exchange with the nuclear charge (long-range, dipole-charge) and with the nuclear magnetic moments
  (dipole-dipole).  In the Anand/Fitzpatrick NREFT (P012/P023, recalled/likely):
     c1^N = e mu Q_N /(2 m_chi)         c5^N = 2 e mu Q_N m_N / q^2        (dipole-charge; O1 velocity-suppressed part + O5)
     c4^N = e mu g_N / m_N              c6^N = - e mu g_N m_N / q^2         (dipole-dipole)
  WimPyDD isospin convention c^0 = c_p + c_n, c^1 = c_p - c_n (P003).  Q_p = 1, Q_n = 0; g_p = 5.5857, g_n = -3.8261.
  All rates are computed at mu = 1 mu_N and scaled by mu^2.
"""
from __future__ import annotations
import sys, os, math, json, time, argparse, importlib
import numpy as np
import pandas as pd
from scipy import special, optimize

sys.path.insert(0, 'output/code')
from common import lzcommon as lz

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'output/work/P065'; FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument('--budget', type=float, default=540.0, help='seconds of WimPyDD computing before exiting (re-run to continue)')
ap.add_argument('--recompute', action='store_true')
ARGS = ap.parse_args()
T0 = time.time()
LOG = []
def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.append(s)

# ----------------------------------------------------------------------------------------------
# 0. Constants (every recalled number flagged)
# ----------------------------------------------------------------------------------------------
RECALLED = []
def rec(item, value, source, reliability):
    RECALLED.append(dict(item=item, value=str(value) if not isinstance(value, (int, float)) else value,
                         presumed_source=source, reliability=reliability)); return value

ALPHA = rec('fine-structure constant 1/137.036', 1 / 137.036, 'PDG', 'certain')
E_CH = math.sqrt(4 * math.pi * ALPHA)                       # Heaviside-Lorentz e = 0.3028
M_P = lz.M_NUCLEON_GEV
G_P, G_N = rec('nucleon g-factors g_p = 5.5857, g_n = -3.8261', (5.5857, -3.8261), 'PDG', 'certain')
HBAR_GEV_S = rec('hbar = 6.5821e-25 GeV s', 6.5821e-25, 'PDG', 'certain')
GEV2_CM2 = lz.GEV_TO_CM2; C_CMS = 2.99792458e10
MU_N_GEV = E_CH / (2 * M_P)                                 # nuclear magneton in natural units, 0.1614 GeV^-1
MU_N_ECM = lz.HBARC_GEV_FM / (2 * M_P) * 1e-13              # 1.0515e-14 e cm
OMEGA_DM_H2 = rec('Omega_DM h^2 = 0.120', 0.120, 'Planck 2018', 'certain')
SV_THERMAL_MAJ = rec('thermal relic <sigma v> = 2.2e-26 cm^3/s for a self-conjugate WIMP, m >> 10 GeV', 2.2e-26,
                     'Steigman, Dasgupta, Beacom 2012', 'certain')
SV_THERMAL_DIRAC = rec('Dirac-like (two-species, chi1 chi2 only) thermal target 2 x 2.2e-26 = 4.4e-26 cm^3/s', 4.4e-26,
                       'standard Boltzmann bookkeeping (dn/dt = -<sv> n1 n2 x 2 with n1 = n2 = n/2)', 'likely')
rec('NREFT photon-dipole coefficients c1 = e mu Q/(2 m_chi), c5 = 2 e mu Q m_N/q^2, c4 = e mu g_N/m_N, c6 = -e mu g_N m_N/q^2 (Anand normalisation)',
    'formula', 'Fitzpatrick et al. 2012 / Anand et al. 2014, as implemented in P012 and P023', 'likely')
rec('MiDM de-excitation width Gamma(chi2 -> chi1 gamma) = mu_tr^2 delta^3/pi', 'formula', 'Chang, Weiner, Yavin 2010', 'likely')
rec('millicharge annihilation sigma v_rel(chi chibar -> f fbar) = pi alpha^2 eps^2 N_c Q_f^2 / m_chi^2 (s-wave, massless f)',
    'formula', 'standard QED (e+e- -> mu+mu- at threshold); used to normalise the dipole derivation', 'likely')
rec('sum over SM fermions N_c Q_f^2 = 8 (3 leptons + 3 up-type x 3 x 4/9 + 3 down-type x 3 x 1/9) at sqrt(s) = 2 TeV; W+W- neglected',
    8.0, 'SM charges', 'certain (W+W- channel omitted, +O(10%))', )
SUM_NCQ2 = 8.0
rec('MiDM literature moment for DAMA-scale fits ~ 1e-3 mu_N at m ~ 100 GeV', 1e-3, 'Chang, Weiner, Yavin 2010 (order of magnitude)', 'uncertain')
rec('LZ 2024 low-energy search exposure 4.2 t yr, 5.4-55 keV window, tolerance <= 3-5 events', (4.2, 3, 5), 'LZ PRL 135, 011802 (2025) via P003/P012', 'certain (exposure) / uncertain (tolerance)')
rec('XENONnT 3.1 t yr and PandaX-4T 1.54 t yr exposures, ROIs ending at <= 60-70 keV', (3.1, 1.54), 'XENONnT/PandaX-4T 2025 results via P005', 'certain (exposures) / uncertain (ROI edges)')
rec('local DM density 0.3 GeV/cm^3 (WimPyDD default rho_loc)', 0.3, 'standard / Baxter 2021', 'certain')
rec('P042 P(clean | tau) tables (Woodcock photon MC in an LZ model) and P(not clean) = 1.68 us/tau asymptote', 'tables', 'corpus P042 (work/P042/*.csv)', 'corpus')
rec('P046/P015 O1 target ratios W/Xe = 15.9/81.7/221/693, I/Xe = 0.591/0.295/0.142/0.0227 at 300/350/366/380 keV (annual halo, full windows)', 'table', 'corpus P046', 'corpus')

EXPOSURE = lz.LZ['exposure_tyr']
M_CHI = 1000.0
WD = lz.wd()
for a in (180, 182, 183, 184, 186):    # runtime fix from P015/P046 (no file edited): 18xW response files use np without importing it
    importlib.import_module(f'WimPyDD.Targets.Nuclear_response_functions.{a}W_func_w').np = np
TARGETS = {'Xe': WD.Xe, 'W': WD.W, 'I': WD.I}

# ----------------------------------------------------------------------------------------------
# 1. Hamiltonians (closures without default arguments -- P031 pitfall)
# ----------------------------------------------------------------------------------------------
def make_c1(mu):
    def f(mchi): return [E_CH * mu / (2 * mchi), E_CH * mu / (2 * mchi)]        # Q_p = 1, Q_n = 0 -> c^0 = c^1 = c_p
    return f
def make_c5(mu):
    def f(q): return [2 * E_CH * mu * M_P / q ** 2, 2 * E_CH * mu * M_P / q ** 2]
    return f
def make_c4(mu):
    def f(): return [E_CH * mu * (G_P + G_N) / M_P, E_CH * mu * (G_P - G_N) / M_P]
    return f
def make_c6(mu):
    def f(q): return [-E_CH * mu * (G_P + G_N) * M_P / q ** 2, -E_CH * mu * (G_P - G_N) * M_P / q ** 2]
    return f
MU_REF = MU_N_GEV
HAM = {
    'midm':   WD.eft_hamiltonian('P065_midm', {1: make_c1(MU_REF), 5: make_c5(MU_REF), 4: make_c4(MU_REF), 6: make_c6(MU_REF)}),
    'charge': WD.eft_hamiltonian('P065_midm_charge', {1: make_c1(MU_REF), 5: make_c5(MU_REF)}),
    'spin':   WD.eft_hamiltonian('P065_midm_spin', {4: make_c4(MU_REF), 6: make_c6(MU_REF)}),
    'O1':     lz.wd_hamiltonian('P065_O1_contact', {1: lz.wd_c_from_anand(1.0 / lz.M_V_GEV ** 2)}),
}

# ----------------------------------------------------------------------------------------------
# 2. Halos (labelled), energy grids, efficiency, spectrum cache
# ----------------------------------------------------------------------------------------------
VGRID = np.linspace(0.0, lz.VESC_KMS + 300.0, 1200)
days12 = np.arange(15, 366, 30.4)
HALOS = {'sun': lz.wd_halo(vmin=VGRID),                                 # WimPyDD single-day SUN-FRAME halo, v_E = 250.6 km/s (P035)
         'june16': lz.wd_halo(day_of_year=167, vmin=VGRID),
         'annual': (VGRID, np.mean([lz.wd_halo(day_of_year=d, vmin=VGRID)[1] for d in days12], axis=0))}
V_E_SUN = float(np.linalg.norm(np.array([0.0, lz.V0_KMS, 0.0]) + lz.V_SUN_PEC))
VMAX = {'sun': V_E_SUN + lz.VESC_KMS, 'june16': lz.v_earth_kms(167) + lz.VESC_KMS}
log(f'halo v_E: sun-frame {V_E_SUN:.1f} km/s, June 16 {lz.v_earth_kms(167):.1f} km/s; v_max {VMAX["sun"]:.0f} / {VMAX["june16"]:.0f} km/s')

E_LZ = np.arange(1.0, 331.0, 3.0)                    # 110 points, covers the LZ ROI and its roll-off
E_EXT = np.arange(335.0, 1001.0, 20.0)               # 34 points, the rest of the kinematic window (full-window target ratios)
E_FINE = np.arange(0.5, 1000.0, 0.5)

def efficiency(E, sig_lo=3.4, sig_hi=8.0, eff0=lz.LZ['eff_plateau']):      # P003/P012/P023 erf model of LZ Fig. S2
    lo, hi = lz.LZ['E_50pct_low_keV'], lz.LZ['E_50pct_high_keV']
    return eff0 * 0.5 * (1 + special.erf((E - lo) / (math.sqrt(2) * sig_lo))) * 0.5 * (1 - special.erf((E - hi) / (math.sqrt(2) * sig_hi)))

CACHE = os.path.join(OUT, 'P065_spectra_cache.npz')
SPEC = {}
if os.path.exists(CACHE) and not ARGS.recompute:
    SPEC = np.load(CACHE, allow_pickle=True)['spectra'].item(); log(f'loaded {len(SPEC)} cached spectra')
def key(target, model, halo, delta, grid): return f'{target}|{model}|{halo}|{delta:.0f}|{grid}'
GRIDS = {'lz': E_LZ, 'ext': E_EXT}
def need(target, model, halo, delta, grid='lz'):
    k = key(target, model, halo, delta, grid)
    if k in SPEC: return SPEC[k]
    if time.time() - T0 > ARGS.budget:
        raise TimeoutError
    t = time.time()
    r = lz.wd_rate(HAM[model], M_CHI, GRIDS[grid], halo=HALOS[halo], delta_kev=delta, target=TARGETS[target])
    SPEC[k] = np.asarray(r); np.savez(CACHE, spectra=np.array(SPEC, dtype=object))
    log(f'  computed {k} in {time.time()-t:.0f} s (total {time.time()-T0:.0f} s)')
    return SPEC[k]

DSCAN = [250.0, 260.0, 270.0, 280.0, 290.0, 300.0, 310.0, 320.0, 330.0, 340.0, 350.0, 360.0, 366.0, 370.0, 380.0]
D_RATIO = [300.0, 350.0, 366.0]
PLAN = ([('Xe', 'midm', 'sun', d, 'lz') for d in DSCAN] +
        [('Xe', 'midm', 'june16', d, 'lz') for d in [300.0, 350.0]] +
        [('Xe', 'midm', 'annual', 300.0, 'lz'), ('Xe', 'charge', 'sun', 300.0, 'lz'), ('Xe', 'spin', 'sun', 300.0, 'lz')] +
        [('Xe', 'O1', 'sun', d, 'lz') for d in D_RATIO] +
        [('Xe', 'midm', 'sun', 200.0, 'lz'), ('Xe', 'O1', 'sun', 200.0, 'lz')] +
        [('Xe', m, 'sun', d, 'lz') for d in (0.0, 200.0) for m in ('charge', 'spin')] + [('Xe', 'midm', 'sun', 0.0, 'lz')] +
        [('Xe', m, 'sun', d, 'ext') for d in D_RATIO for m in ('midm', 'O1')] +
        [(t, m, 'sun', d, 'lz') for t in ('I', 'W') for d in D_RATIO for m in ('midm', 'O1')] +
        [(t, m, 'sun', d, 'ext') for t in ('I', 'W') for d in D_RATIO for m in ('midm', 'O1')])
try:
    for p in PLAN: need(*p)
except TimeoutError:
    missing = [key(*p) for p in PLAN if key(*p) not in SPEC]
    log(f'time budget used; {len(missing)} spectra still missing -- re-run the script to continue'); sys.exit(0)
log(f'all {len(PLAN)} spectra available ({time.time()-T0:.0f} s)')

# ----------------------------------------------------------------------------------------------
# 3. Counting helpers
# ----------------------------------------------------------------------------------------------
def fine(E, r):
    """interpolate a spectrum onto the 0.5 keV grid (zero outside), for edge-accurate integrals"""
    return np.interp(E_FINE, E, r, left=0.0, right=0.0) * (E_FINE >= E.min() - 1.5) * (E_FINE <= E.max() + 1.5)
def counts(E, r, exposure=EXPOSURE):
    rf = fine(E, r); w = efficiency(E_FINE) * rf
    def integ(mask): return float(np.trapezoid(np.where(mask, w, 0.0), E_FINE)) * exposure
    N_roi = integ(E_FINE <= 330.0); N_hi = integ((E_FINE >= 200) & (E_FINE <= 270)); N_lo = integ((E_FINE >= 5.4) & (E_FINE <= 55))
    cdf = np.cumsum(np.where(E_FINE <= 330.0, w, 0.0)); cdf = cdf / max(cdf[-1], 1e-300)
    pct248 = float(np.interp(248.0, E_FINE, cdf)); pct200 = float(np.interp(200.0, E_FINE, cdf))
    q16, q50, q84 = [float(np.interp(p, cdf, E_FINE)) for p in (0.16, 0.5, 0.84)]
    peak = float(E_FINE[np.argmax(w)]) if w.max() > 0 else float('nan')
    return dict(N_roi=N_roi, N_hi=N_hi, N_lo=N_lo, pct248_accepted=pct248, frac_below200_accepted=pct200,
                q16=q16, q50=q50, q84=q84, peak_accepted=peak)
def full_window_rate(target, model, halo, delta):
    """events/(t yr) over the whole kinematic window (1-1000 keV, no efficiency)"""
    r = np.concatenate([need(target, model, halo, delta, 'lz'), need(target, model, halo, delta, 'ext')])
    E = np.concatenate([E_LZ, E_EXT]); rf = fine(E, r)
    return float(np.trapezoid(rf, E_FINE)), E, r
def kin_window(delta, halo):
    lo, hi = lz.E_R_range_keV(M_CHI, VMAX[halo], delta_kev=delta)
    return float(lo), float(hi)

# ----------------------------------------------------------------------------------------------
# 4. Part (a): spectra, percentile, N_lo, mu(N=1)
# ----------------------------------------------------------------------------------------------
log('=== (a) MiDM xenon spectra at 1 TeV ===')
rows = []
for halo in ('sun', 'june16', 'annual'):
    for d in DSCAN + [0.0, 200.0]:
        k = key('Xe', 'midm', halo, d, 'lz')
        if k not in SPEC: continue
        c = counts(E_LZ, SPEC[k]); lo, hi = kin_window(d, 'sun' if halo == 'annual' else halo)
        c.update(model='midm', halo=halo, delta_keV=d, E_minus_keV=lo, E_plus_keV=hi,
                 mu_N1_roi_muN=math.sqrt(1.0 / max(c['N_roi'], 1e-300)), mu_N1_hi_muN=math.sqrt(1.0 / max(c['N_hi'], 1e-300)),
                 N_lo_per_hi=c['N_lo'] / max(c['N_hi'], 1e-300))
        c['mu_N1_roi_ecm'] = c['mu_N1_roi_muN'] * MU_N_ECM; c['mu_N1_roi_e_over_2mchi'] = c['mu_N1_roi_muN'] * M_CHI / M_P
        c['mu_N1_roi_GeVinv'] = c['mu_N1_roi_muN'] * MU_N_GEV
        rows.append(c)
for d in [200.0] + D_RATIO:
    c = counts(E_LZ, SPEC[key('Xe', 'O1', 'sun', d, 'lz')]); lo, hi = kin_window(d, 'sun')
    c.update(model='O1_contact', halo='sun', delta_keV=d, E_minus_keV=lo, E_plus_keV=hi, N_lo_per_hi=c['N_lo'] / max(c['N_hi'], 1e-300)); rows.append(c)
for m in ('charge', 'spin'):
    for d in (0.0, 200.0, 300.0):
        c = counts(E_LZ, SPEC[key('Xe', m, 'sun', d, 'lz')]); c.update(model=f'midm_{m}_only', halo='sun', delta_keV=d, N_lo_per_hi=c['N_lo'] / max(c['N_hi'], 1e-300)); rows.append(c)
SC = pd.DataFrame(rows); SC.to_csv(os.path.join(OUT, 'P065_xenon_scan.csv'), index=False, float_format='%.5g')

def R(model, halo, d): return SC[(SC.model == model) & (SC.halo == halo) & (SC.delta_keV == d)].iloc[0]
s300 = R('midm', 'sun', 300.0); j300 = R('midm', 'june16', 300.0); a300 = R('midm', 'annual', 300.0); o300 = R('O1_contact', 'sun', 300.0)
log(f'  delta=300 keV, Sun frame: N_ROI(1 mu_N) = {s300.N_roi:.4g}; mu(N_ROI=1) = {s300.mu_N1_roi_muN:.3e} mu_N = {s300.mu_N1_roi_ecm:.2e} e cm = {s300.mu_N1_roi_e_over_2mchi:.3f} e/(2 m_chi)')
log(f'  June 16: mu(N=1) = {j300.mu_N1_roi_muN:.3e} mu_N (rate ratio June/Sun = {j300.N_roi/s300.N_roi:.3f}); annual mean: {a300.mu_N1_roi_muN:.3e} mu_N (P023: 2.107e-4; ratio {a300.mu_N1_roi_muN/2.107e-4:.3f})')
log(f'  248 keV percentile (accepted spectrum): MiDM {100*s300.pct248_accepted:.1f}%, O1 contact {100*o300.pct248_accepted:.1f}%; accepted peak {s300.peak_accepted:.0f} vs {o300.peak_accepted:.0f} keV; 68% {s300.q16:.0f}-{s300.q84:.0f} vs {o300.q16:.0f}-{o300.q84:.0f} keV')
log(f'  fraction of accepted rate below 200 keV: MiDM {s300.frac_below200_accepted:.3f}, O1 {o300.frac_below200_accepted:.3f}; kinematic window {s300.E_minus_keV:.0f}-{s300.E_plus_keV:.0f} keV')
# charge vs spin decomposition at 300 keV (rates at 248 keV and integrated)
r_full = SPEC[key('Xe', 'midm', 'sun', 300.0, 'lz')]; r_ch = SPEC[key('Xe', 'charge', 'sun', 300.0, 'lz')]; r_sp = SPEC[key('Xe', 'spin', 'sun', 300.0, 'lz')]
at = lambda r, e: float(np.interp(e, E_LZ, r))
cch, csp = R('midm_charge_only', 'sun', 300.0), R('midm_spin_only', 'sun', 300.0)
decomp = dict(rate248_full=at(r_full, 248), rate248_charge=at(r_ch, 248), rate248_spin=at(r_sp, 248),
              frac_charge_248=at(r_ch, 248) / at(r_full, 248), frac_spin_248=at(r_sp, 248) / at(r_full, 248),
              interference_248=(at(r_full, 248) - at(r_ch, 248) - at(r_sp, 248)) / at(r_full, 248),
              frac_charge_roi=cch.N_roi / s300.N_roi, frac_spin_roi=csp.N_roi / s300.N_roi, interference_roi=(s300.N_roi - cch.N_roi - csp.N_roi) / s300.N_roi,
              rate150_charge_frac=at(r_ch, 150) / at(r_full, 150), rate100_charge_frac=at(r_ch, 100) / max(at(r_full, 100), 1e-300))
log(f'  decomposition at 248 keV: charge {100*decomp["frac_charge_248"]:.1f}%, dipole-dipole {100*decomp["frac_spin_248"]:.1f}%, interference {100*decomp["interference_248"]:.2f}%; ROI: {100*decomp["frac_charge_roi"]:.1f}/{100*decomp["frac_spin_roi"]:.1f}/{100*decomp["interference_roi"]:.2f}%')
# validation of the v_perp suppression: the same decomposition for elastic (delta = 0; P012: charge part dominates, N_lo/hi 376) and delta = 200
DEC = []
for d in (0.0, 200.0, 300.0):
    rf_, rc_, rs_ = SPEC[key('Xe', 'midm', 'sun', d, 'lz')], SPEC[key('Xe', 'charge', 'sun', d, 'lz')], SPEC[key('Xe', 'spin', 'sun', d, 'lz')]
    m_, c_, s_ = R('midm', 'sun', d), R('midm_charge_only', 'sun', d), R('midm_spin_only', 'sun', d)
    vmin248 = lz.vmin_kms(248.0, M_CHI, delta_kev=d); vperp2_max = max(VMAX['sun'] ** 2 - vmin248 ** 2, 0.0)
    DEC.append(dict(delta_keV=d, frac_charge_248=at(rc_, 248) / at(rf_, 248), frac_spin_248=at(rs_, 248) / at(rf_, 248),
                    frac_charge_hi=c_.N_hi / m_.N_hi, frac_spin_hi=s_.N_hi / m_.N_hi, N_lo_per_hi_full=m_.N_lo / max(m_.N_hi, 1e-300),
                    N_lo_per_hi_charge=c_.N_lo / max(m_.N_hi, 1e-300), N_lo_per_hi_spin=s_.N_lo / max(m_.N_hi, 1e-300),
                    vmin248_kms=vmin248, vperp2_max_over_vmax2=vperp2_max / VMAX['sun'] ** 2))
DEC = pd.DataFrame(DEC); DEC.to_csv(os.path.join(OUT, 'P065_decomposition.csv'), index=False, float_format='%.4g')
for _, x in DEC.iterrows():
    log(f'  delta={x.delta_keV:.0f}: charge/spin fraction at 248 keV {x.frac_charge_248:.3f}/{x.frac_spin_248:.3f}; N_lo per hi (full/charge/spin) {x.N_lo_per_hi_full:.3g}/{x.N_lo_per_hi_charge:.3g}/{x.N_lo_per_hi_spin:.3g}; v_min(248) = {x.vmin248_kms:.0f} km/s, max v_perp^2/v_max^2 = {x.vperp2_max_over_vmax2:.3f}')
log('  (P012 elastic reference: N_lo/hi = 376, charge part 18000, spin part 32 -- different efficiency/halo details)')
# shape ratio MiDM/O1 across the window (normalised at 248 keV)
r_o1 = SPEC[key('Xe', 'O1', 'sun', 300.0, 'lz')]
shape = pd.DataFrame(dict(E_keV=E_LZ, midm=r_full / at(r_full, 248), O1=r_o1 / at(r_o1, 248), charge=r_ch / at(r_full, 248), spin=r_sp / at(r_full, 248)))
shape['midm_over_O1'] = shape.midm / shape.O1.replace(0, np.nan); shape.to_csv(os.path.join(OUT, 'P065_shape_d300.csv'), index=False, float_format='%.5g')
for e in (120.0, 150.0, 200.0, 300.0):
    v = float(np.interp(e, E_LZ, shape.midm_over_O1.fillna(0).values)); log(f'  (MiDM/O1) normalised at 248: {v:.2f} at {e:.0f} keV')
# low-energy repopulation where the window reaches < 55 keV
for d in (200.0, 250.0, 260.0):
    m_ = R('midm', 'sun', d)
    o_ = R('O1_contact', 'sun', d) if d == 200.0 else None
    log(f'  delta={d:.0f}: window {m_.E_minus_keV:.0f}-{m_.E_plus_keV:.0f} keV; N_lo per hi event MiDM {m_.N_lo_per_hi:.3g}' + (f', O1 {o_.N_lo_per_hi:.3g}' if o_ is not None else ''))

# ----------------------------------------------------------------------------------------------
# 5. Part (b): lifetime tau(delta) and P(clean) from P042
# ----------------------------------------------------------------------------------------------
log('=== (b) de-excitation lifetime and P(clean) ===')
P042 = {d: pd.read_csv(f'output/work/P042/P042_topology_vs_tau_d{d}_wind.csv') for d in (250, 300, 350, 380)}
def p042_table(delta): return P042[min(P042, key=lambda d: abs(d - delta))]
def p_clean(tau, delta):
    t = p042_table(delta); return float(np.interp(math.log(tau), np.log(t.tau_s.values), t.P_clean.values))
def p_class(tau, delta, col):
    t = p042_table(delta); return float(np.interp(math.log(tau), np.log(t.tau_s.values), t[col].values))
def gamma_M1(mu_GeVinv, delta_GeV): return mu_GeVinv ** 2 * delta_GeV ** 3 / math.pi
def tau_s(mu_muN, delta_keV): return HBAR_GEV_S / gamma_M1(mu_muN * MU_N_GEV, delta_keV * 1e-6)
lif = []
for halo in ('sun', 'june16'):
    sub = SC[(SC.model == 'midm') & (SC.halo == halo) & (SC.delta_keV >= 250)].sort_values('delta_keV')
    for _, s in sub.iterrows():
        for label, N in (('N=1', 1.0), ('N=0.3', 0.3), ('N=2.4', 2.4)):
            mu = s.mu_N1_roi_muN * math.sqrt(N); t = tau_s(mu, s.delta_keV)
            lif.append(dict(halo=halo, delta_keV=s.delta_keV, case=label, mu_muN=mu, mu_ecm=mu * MU_N_ECM, tau_s=t, decay_length_m=663e3 * t,
                            P_clean=p_clean(t, s.delta_keV), P_tpc=p_class(t, s.delta_keV, 'P_tpc'), P_delayed=p_class(t, s.delta_keV, 'P_delayed'),
                            P_prompt=p_class(t, s.delta_keV, 'P_prompt')))
LF = pd.DataFrame(lif); LF.to_csv(os.path.join(OUT, 'P065_lifetime_pclean.csv'), index=False, float_format='%.5g')
def delta_at_pclean(halo, case, target):
    t = LF[(LF.halo == halo) & (LF.case == case)].sort_values('delta_keV'); d = t.delta_keV.values; p = t.P_clean.values
    for i in range(len(d) - 1):
        if (p[i] - target) * (p[i + 1] - target) <= 0 and p[i] != p[i + 1]:
            return float(d[i] + (d[i + 1] - d[i]) * (p[i] - target) / (p[i] - p[i + 1]))
    return float('nan')
# June-16 halo: only two delta points were computed (300, 350 keV), so the June curve is obtained by scaling the Sun-frame
# mu(delta) with the interpolated (in ln) June/Sun rate ratio; a linear interpolation of P(clean) between two points 50 keV
# apart would be meaningless (checked: it moved the ceiling the wrong way).
jr = {d: R('midm', 'june16', d).N_roi / R('midm', 'sun', d).N_roi for d in (300.0, 350.0)}
def june_ratio(d): return math.exp(np.interp(d, [300.0, 350.0], [math.log(jr[300.0]), math.log(jr[350.0])]))
for _, s in SC[(SC.model == 'midm') & (SC.halo == 'sun') & (SC.delta_keV >= 250)].sort_values('delta_keV').iterrows():
    for label, N in (('N=1', 1.0), ('N=0.3', 0.3), ('N=2.4', 2.4)):
        mu = s.mu_N1_roi_muN * math.sqrt(N / june_ratio(s.delta_keV)); t = tau_s(mu, s.delta_keV)
        lif.append(dict(halo='june16_scaled', delta_keV=s.delta_keV, case=label, mu_muN=mu, mu_ecm=mu * MU_N_ECM, tau_s=t, decay_length_m=663e3 * t,
                        P_clean=p_clean(t, s.delta_keV), P_tpc=p_class(t, s.delta_keV, 'P_tpc'), P_delayed=p_class(t, s.delta_keV, 'P_delayed'), P_prompt=0.0))
LF = pd.DataFrame(lif); LF.to_csv(os.path.join(OUT, 'P065_lifetime_pclean.csv'), index=False, float_format='%.5g')
log(f'  June/Sun rate ratio {jr[300.0]:.3f} (300 keV), {jr[350.0]:.3f} (350 keV)')
CEIL = {}
for halo in ('sun', 'june16_scaled'):
    for case in ('N=1', 'N=0.3', 'N=2.4'):
        CEIL[f'{halo}|{case}'] = {f'P{int(100*p)}': delta_at_pclean(halo, case, p) for p in (0.9, 0.5, 0.1)}
        log(f'  {halo:13s} {case:6s}: P(clean) = 0.9/0.5/0.1 at delta = {CEIL[f"{halo}|{case}"]["P90"]:.0f}/{CEIL[f"{halo}|{case}"]["P50"]:.0f}/{CEIL[f"{halo}|{case}"]["P10"]:.0f} keV')
for d in (300.0, 350.0):
    lj = LF[(LF.halo == 'june16') & (LF.case == 'N=1') & (LF.delta_keV == d)].iloc[0]
    log(f'  June 16 direct, delta={d:.0f}: mu = {lj.mu_muN:.3e} mu_N, tau = {lj.tau_s*1e6:.2f} us, P(clean) = {lj.P_clean:.3f}')
l300 = LF[(LF.halo == 'sun') & (LF.case == 'N=1') & (LF.delta_keV == 300)].iloc[0]
log(f'  delta=300, N=1 (Sun): mu = {l300.mu_muN:.3e} mu_N, tau = {l300.tau_s*1e6:.1f} us, decay length {l300.decay_length_m:.0f} m, P(clean) = {l300.P_clean:.4f}, P(TPC 2nd site) = {l300.P_tpc:.4f}, P(delayed veto) = {l300.P_delayed:.4f}')
log('  (P042 reference with P023 moments: 321/344/358 keV for N=1)')

# ----------------------------------------------------------------------------------------------
# 6. Part (c): elastic and low-energy constraints
# ----------------------------------------------------------------------------------------------
log('=== (c) constraints from elastic and low-energy data ===')
EXPO = {'LZ2024 (5.4-55 keV)': 4.2, 'XENONnT (<=60 keV)': 3.1, 'PandaX-4T (<=60 keV)': 1.54}
con = []
for _, s in SC[(SC.model == 'midm') & (SC.halo == 'sun')].sort_values('delta_keV').iterrows():
    mu2 = s.mu_N1_roi_muN ** 2
    rf = fine(E_LZ, SPEC[key('Xe', 'midm', 'sun', s.delta_keV, 'lz')]) * mu2       # per t yr at mu(N_ROI = 1)
    n55 = float(np.trapezoid(np.where((E_FINE >= 5.4) & (E_FINE <= 55), rf * efficiency(E_FINE), 0), E_FINE))
    n60 = float(np.trapezoid(np.where((E_FINE >= 3.0) & (E_FINE <= 60), rf, 0), E_FINE))
    con.append(dict(delta_keV=s.delta_keV, E_minus_keV=s.E_minus_keV, N_lo_per_hi=s.N_lo_per_hi,
                    N_LZ2024_at_N1=n55 * 4.2, N_XENONnT_60keV_at_N1=n60 * 3.1, N_PandaX_60keV_at_N1=n60 * 1.54))
CON = pd.DataFrame(con); CON.to_csv(os.path.join(OUT, 'P065_low_energy_constraints.csv'), index=False, float_format='%.4g')
for d in (250.0, 260.0, 300.0):
    c = CON[CON.delta_keV == d].iloc[0]; log(f'  delta={d:.0f}: E_- = {c.E_minus_keV:.1f} keV, N_lo/hi = {c.N_lo_per_hi:.3g}, LZ-2024 events at mu(N=1) = {c.N_LZ2024_at_N1:.3g}, XENONnT(<=60 keV) {c.N_XENONnT_60keV_at_N1:.3g}')
# delta at which E_- crosses 55 keV (Sun frame and June)
d55 = {h: float(optimize.brentq(lambda d: lz.E_R_range_keV(M_CHI, VMAX[h], delta_kev=d)[0] - 55.0, 100.0, 380.0)) for h in ('sun', 'june16')}
d60 = {h: float(optimize.brentq(lambda d: lz.E_R_range_keV(M_CHI, VMAX[h], delta_kev=d)[0] - 60.0, 100.0, 380.0)) for h in ('sun', 'june16')}
log(f'  low-energy windows empty for delta > {d55["sun"]:.0f} (Sun) / {d55["june16"]:.0f} keV (June) [55 keV edge]; {d60["sun"]:.0f}/{d60["june16"]:.0f} keV [60 keV edge]')
# elastic: none at tree level; loop-level estimate (order of magnitude, flagged)
# two transition-dipole insertions with chi2 in the loop give an effective chi1-chi1-gamma-gamma (Rayleigh/polarizability) operator
# of size ~ alpha mu^2/(4 pi) x (q^2/m_chi) relative to ...; we only quote the parametric suppression of the amplitude ratio:
mu300 = s300.mu_N1_roi_GeVinv
loop_ratio = (mu300 * 0.25) ** 2 / (16 * math.pi ** 2)      # (mu q)^2/(16 pi^2), q = 0.25 GeV: elastic (two dipole vertices, loop) vs inelastic (one vertex, tree) amplitude
log(f'  elastic chi1 N -> chi1 N: absent at tree level (no chi1 chi1 gamma vertex); loop estimate amplitude ratio ~ (mu q)^2/(16 pi^2) = {loop_ratio:.1e} at q = 0.25 GeV -> rate ratio {loop_ratio**2:.0e} (order of magnitude, flagged)')

# ----------------------------------------------------------------------------------------------
# 7. Part (d): relic density via chi1 chi2 -> gamma* -> f fbar
# ----------------------------------------------------------------------------------------------
log('=== (d) relic density ===')
# numerical spinor check of the initial-state tensors at threshold (Dirac representation)
I2 = np.eye(2); Z2 = np.zeros((2, 2)); sx = np.array([[0, 1], [1, 0]], complex); sy = np.array([[0, -1j], [1j, 0]]); sz = np.array([[1, 0], [0, -1]], complex)
g0 = np.block([[I2, Z2], [Z2, -I2]]).astype(complex); gs = [np.block([[Z2, s], [-s, Z2]]) for s in (sx, sy, sz)]
def sigma_mn(a, b): return 0.5j * (a @ b - b @ a)
m = 1.0
us = [math.sqrt(2 * m) * np.array([1, 0, 0, 0], complex), math.sqrt(2 * m) * np.array([0, 1, 0, 0], complex)]        # chi1 at rest
vs = [math.sqrt(2 * m) * np.array([0, 0, 1, 0], complex), math.sqrt(2 * m) * np.array([0, 0, 0, 1], complex)]        # chi2 (antiparticle spinor) at rest
qmu = np.array([2 * m, 0, 0, 0])
L_dip = np.zeros((3, 3), complex); L_vec = np.zeros((3, 3), complex)
for u in us:
    for v in vs:
        ub = u.conj() @ g0
        Jd = np.array([sum(qmu[0] * (ub @ sigma_mn(g0, gs[i]) @ v) for _ in [0]) for i in range(3)])   # sigma^{0 i} q_0 (spatial components of the dipole current)
        Jv = np.array([ub @ gs[i] @ v for i in range(3)])
        L_dip += np.outer(Jd, Jd.conj()); L_vec += np.outer(Jv, Jv.conj())
log(f'  spinor check (m = 1): dipole tensor L^ij = {L_dip[0,0].real:.1f} delta^ij (analytic 32 m^4), vector L^ij = {L_vec[0,0].real:.1f} delta^ij (analytic 8 m^2)')
# sigma v_rel = (1/(4 m^2)) (1/(8 pi)) <|M|^2>_avg, with lepton tensor (16/3) m^2 delta_ij (angular average), 1/4 spin average:
#   dipole: |M|^2 = (e^2 Q_f^2 mu^2 / s^2) L^ij T_ij = 8 e^2 Q_f^2 mu^2 m^2  ->  sigma v = alpha mu^2 Q_f^2 (per colour)
#   vector (charge g): 2 g^2 e^2 Q_f^2  ->  sigma v = pi alpha alpha_chi Q_f^2/m^2  (matches the recalled millicharge formula)
def sv_ff_cm3s(mu_muN): return ALPHA * (mu_muN * MU_N_GEV) ** 2 * SUM_NCQ2 * GEV2_CM2 * C_CMS
def sv_gg_estimate_cm3s(mu_muN, m=M_CHI): return (mu_muN * MU_N_GEV) ** 4 * m ** 2 / (4 * math.pi) * GEV2_CM2 * C_CMS     # chi1 chi1 -> gamma gamma via chi2 exchange, O(mu^4): order of magnitude
mu_th = math.sqrt(SV_THERMAL_DIRAC / (ALPHA * SUM_NCQ2 * GEV2_CM2 * C_CMS)) / MU_N_GEV
rel = []
for _, s in SC[(SC.model == 'midm') & (SC.halo == 'sun') & (SC.delta_keV >= 250)].sort_values('delta_keV').iterrows():
    sv = sv_ff_cm3s(s.mu_N1_roi_muN)
    rel.append(dict(delta_keV=s.delta_keV, mu_N1_muN=s.mu_N1_roi_muN, sigmav_ff_cm3s=sv, sigmav_gg_est_cm3s=sv_gg_estimate_cm3s(s.mu_N1_roi_muN),
                    Omega_h2=OMEGA_DM_H2 * SV_THERMAL_DIRAC / sv, extra_sigmav_needed_cm3s=max(SV_THERMAL_DIRAC - sv, 0.0),
                    N_LZ_if_thermal=(mu_th / s.mu_N1_roi_muN) ** 2, tau_if_thermal_s=tau_s(mu_th, s.delta_keV), P_clean_if_thermal=p_clean(tau_s(mu_th, s.delta_keV), s.delta_keV)))
REL = pd.DataFrame(rel); REL.to_csv(os.path.join(OUT, 'P065_relic.csv'), index=False, float_format='%.4g')
r300 = REL[REL.delta_keV == 300].iloc[0]
log(f'  sigma v(chi1 chi2 -> f fbar) = alpha mu^2 sum N_c Q_f^2; at mu(N=1, 300 keV) = {r300.mu_N1_muN:.2e} mu_N: {r300.sigmav_ff_cm3s:.2e} cm^3/s -> Omega h^2 = {r300.Omega_h2:.2f} ({r300.Omega_h2/OMEGA_DM_H2:.0f}x over-abundant); gamma gamma O(mu^4) estimate {r300.sigmav_gg_est_cm3s:.1e}')
log(f'  thermal moment (Dirac-like target 4.4e-26): mu_th = {mu_th:.2e} mu_N = {mu_th*MU_N_ECM:.2e} e cm, mass-independent; LZ events at mu_th: {r300.N_LZ_if_thermal:.0f} (300 keV)')
def delta_thermal_N(target=1.0):
    t = REL.sort_values('delta_keV'); d = t.delta_keV.values; ln = np.log(t.N_LZ_if_thermal.values / target)
    for i in range(len(d) - 1):
        if ln[i] * ln[i + 1] <= 0: return float(d[i] + (d[i + 1] - d[i]) * ln[i] / (ln[i] - ln[i + 1]))
    return float('nan')
D_TH = delta_thermal_N(1.0); TAU_TH = tau_s(mu_th, D_TH); PC_TH = p_clean(TAU_TH, D_TH)
TH_BAND = {N: dict(delta_keV=delta_thermal_N(N), tau_us=tau_s(mu_th, delta_thermal_N(N)) * 1e6, P_clean=p_clean(tau_s(mu_th, delta_thermal_N(N)), delta_thermal_N(N))) for N in (0.3, 1.0, 2.4)}
log(f'  thermal MiDM gives N = 1 at delta = {D_TH:.0f} keV, where tau = {TAU_TH*1e6:.2f} us and P(clean) = {PC_TH:.2f}; 68% band (N = 2.4 .. 0.3): delta = {TH_BAND[2.4]["delta_keV"]:.0f}-{TH_BAND[0.3]["delta_keV"]:.0f} keV, P(clean) = {TH_BAND[2.4]["P_clean"]:.2f}-{TH_BAND[0.3]["P_clean"]:.2f}')
# mass dependence: sigma v is m-independent for fixed mu, and mu(N=1) at fixed delta varies weakly with m (P023: 4e2-4e3 GeV)

# ----------------------------------------------------------------------------------------------
# 8. Part (e): astrophysical 300 keV line (relic chi2, halo up-scattering)
# ----------------------------------------------------------------------------------------------
log('=== (e) astrophysical 300 keV photons ===')
# relic chi2: tau = 66 us << 1 s: any excited fraction left at decoupling (P026: f2 ~ 0.4-0.5) decays before BBN; f2(today) = 0.
# halo up-scattering chi1 chi1 -> chi2 chi2 (both legs flip; endothermic by 2 delta): threshold v_rel and an O(mu^4) cross-section estimate
v_thr = math.sqrt(4 * 300e-6 / (M_CHI / 2)) * lz.C_KMS          # (1/2) mu_red v^2 >= 2 delta with mu_red = m/2
sig_up = (mu300 ** 4 * M_CHI ** 2 / (4 * math.pi)) * GEV2_CM2   # cm^2 (order of magnitude)
n_loc = 0.3 / M_CHI; v_cm = 300e5
flux_gc = 0.5 * n_loc ** 2 * sig_up * v_cm * 2.5e22 / (4 * math.pi) * 10   # /cm^2/s/sr, factor 10 for the GC over-density along the line of sight (crude)
log(f'  chi1 chi1 -> chi2 chi2 threshold v_rel = {v_thr:.0f} km/s; sigma ~ mu^4 m^2/(4 pi) = {sig_up:.1e} cm^2; GC line flux ~ {flux_gc:.1e} ph/cm^2/s/sr (SPI sensitivity ~1e-5 ph/cm^2/s; recalled/uncertain)')

# ----------------------------------------------------------------------------------------------
# 9. Part (f): predictions -- delayed-veto fraction, target ratios
# ----------------------------------------------------------------------------------------------
log('=== (f) predictions ===')
ratios = []
for d in D_RATIO:
    out = dict(delta_keV=d)
    for model in ('midm', 'O1'):
        rx, Ex, rrx = full_window_rate('Xe', model, 'sun', d); rw = full_window_rate('W', model, 'sun', d)[0]; ri = full_window_rate('I', model, 'sun', d)[0]
        out[f'{model}_Xe_per_tyr'] = rx; out[f'{model}_W_per_tyr'] = rw; out[f'{model}_I_per_tyr'] = ri
        out[f'{model}_W_over_Xe'] = rw / rx if rx > 0 else float('nan'); out[f'{model}_I_over_Xe'] = ri / rx if rx > 0 else float('nan')
    out['W_over_Xe_midm_over_O1'] = out['midm_W_over_Xe'] / out['O1_W_over_Xe']; out['I_over_Xe_midm_over_O1'] = out['midm_I_over_Xe'] / out['O1_I_over_Xe']
    ratios.append(out)
RT = pd.DataFrame(ratios); RT.to_csv(os.path.join(OUT, 'P065_target_ratios.csv'), index=False, float_format='%.4g')
P046_WXE = {300.0: 15.9, 350.0: 81.7, 366.0: 221.0, 380.0: 693.0}; P046_IXE = {300.0: 0.591, 350.0: 0.295, 366.0: 0.142, 380.0: 0.0227}
for _, t in RT.iterrows():
    log(f'  delta={t.delta_keV:.0f}: W/Xe MiDM {t.midm_W_over_Xe:.3g} vs O1 {t.O1_W_over_Xe:.3g} (P046 annual {P046_WXE[t.delta_keV]}); I/Xe MiDM {t.midm_I_over_Xe:.3g} vs O1 {t.O1_I_over_Xe:.3g} (P046 {P046_IXE[t.delta_keV]}); MiDM/O1 ratio-of-ratios W {t.W_over_Xe_midm_over_O1:.2f}, I {t.I_over_Xe_midm_over_O1:.2f}')
# Z^2/A^2 expectation for the charge term: nuclei per tonne x Z^2 vs A^2
ZA = {'Xe': (54, 131.29), 'W': (74, 183.84), 'I': (53, 126.90)}
for t in ('W', 'I'):
    z, a = ZA[t]; zx, ax = ZA['Xe']
    log(f'  naive coherent scaling {t}/Xe: (Z^2/A)/(Z_Xe^2/A_Xe) = {(z**2/a)/(zx**2/ax):.3f} (charge) vs (A^2/A)/(A_Xe^2/A_Xe) = {a/ax:.3f} (O1) -> ratio {(z**2/a)/(zx**2/ax)/(a/ax):.3f}')
# delayed-veto / two-site fractions of a MiDM population at tau(delta), N = 1
pop = LF[(LF.halo == 'sun') & (LF.case == 'N=1')][['delta_keV', 'tau_s', 'P_clean', 'P_tpc', 'P_delayed', 'P_prompt']]
pop.to_csv(os.path.join(OUT, 'P065_population_topology.csv'), index=False, float_format='%.4g')
for d in (300.0, 320.0, 340.0, 350.0):
    p = pop[pop.delta_keV == d].iloc[0]; log(f'  population at delta={d:.0f}: tau {p.tau_s*1e6:.2f} us; clean {100*p.P_clean:.1f}%, second TPC site {100*p.P_tpc:.1f}%, OD delayed veto {100*p.P_delayed:.1f}%, prompt {100*p.P_prompt:.1f}%')

# ----------------------------------------------------------------------------------------------
# 10. Figures
# ----------------------------------------------------------------------------------------------
C1, C2, C3, C4, CG = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#52514e'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25})
# Fig 1: spectra at delta = 300 keV (Sun frame), and 350 keV
fig, axs = plt.subplots(1, 2, figsize=(10, 4.0))
ax = axs[0]
ax.plot(E_LZ, r_full / r_full.max(), color=C1, lw=2.2, label='MiDM (full)')
ax.plot(E_LZ, r_ch / r_full.max(), color=C1, lw=1.2, ls='--', label='dipole–charge part')
ax.plot(E_LZ, r_sp / r_full.max(), color=C1, lw=1.2, ls=':', label='dipole–dipole part')
ax.plot(E_LZ, r_o1 / r_o1.max(), color=C2, lw=2, label='contact O₁ inelastic')
ax.plot(E_LZ, efficiency(E_LZ), color=CG, lw=1, alpha=0.6, label='LZ efficiency')
ax.axvline(248, color=CG, lw=1, ls=':'); ax.text(250, 0.93, 'event', fontsize=8)
ax.set_xlabel('recoil energy [keV]'); ax.set_ylabel('dR/dE (peak-normalised)'); ax.set_title('δ = 300 keV, 1 TeV, Sun-frame halo'); ax.legend(fontsize=8, loc='upper left')
ax = axs[1]
for d, c in ((300.0, C1), (350.0, C3), (380.0, C4)):
    r = SPEC[key('Xe', 'midm', 'sun', d, 'lz')]; ax.plot(E_LZ, r * efficiency(E_LZ) / (r * efficiency(E_LZ)).max(), color=c, lw=2, label=f'MiDM δ = {d:.0f} keV')
ax.axvline(248, color=CG, lw=1, ls=':'); ax.set_xlabel('recoil energy [keV]'); ax.set_ylabel('accepted spectrum (peak-normalised)'); ax.set_title('Efficiency-folded MiDM spectra'); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P065_fig1_spectra.png'), dpi=140); plt.close(fig)
# Fig 2: tau(delta) and P(clean)
fig, ax = plt.subplots(figsize=(6.6, 4.4))
sub1 = LF[(LF.halo == 'sun') & (LF.case == 'N=1')].sort_values('delta_keV'); lo_ = LF[(LF.halo == 'sun') & (LF.case == 'N=0.3')].sort_values('delta_keV'); hi_ = LF[(LF.halo == 'sun') & (LF.case == 'N=2.4')].sort_values('delta_keV')
ax.plot(sub1.delta_keV, sub1.tau_s * 1e6, color=C1, lw=2.2, label='τ(δ) for N_ROI = 1 (Sun frame)')
ax.fill_between(sub1.delta_keV, hi_.tau_s.values * 1e6, lo_.tau_s.values * 1e6, color=C1, alpha=0.15, label='N = 0.3–2.4 (68 %)')
subj = LF[(LF.halo == 'june16_scaled') & (LF.case == 'N=1')].sort_values('delta_keV'); ax.plot(subj.delta_keV, subj.tau_s * 1e6, color=C1, lw=1.2, ls='--', label='June 16 halo (scaled)')
subd = LF[(LF.halo == 'june16') & (LF.case == 'N=1')]; ax.plot(subd.delta_keV, subd.tau_s * 1e6, 's', color=C1, ms=5)
ax.axhline(0.43, color=C2, lw=1.2, ls='--'); ax.text(252, 0.5, 'P042: τ > 0.43 μs (90 % CL)', color=C2, fontsize=8)
ax.axhline(TAU_TH * 1e6, color=C4, lw=1, ls=':');
ax.set_yscale('log'); ax.set_xlabel('δ [keV]'); ax.set_ylabel('τ(χ₂ → χ₁γ) [μs]'); ax.set_xlim(250, 390)
ax2 = ax.twinx(); ax2.plot(sub1.delta_keV, sub1.P_clean, color=C3, lw=2, label='P(clean | τ(δ)) [P042]'); ax2.set_ylabel('P(clean)', color=C3); ax2.set_ylim(0, 1.05); ax2.grid(False)
for p, lab in ((0.9, '0.9'), (0.5, '0.5'), (0.1, '0.1')):
    dd = CEIL['sun|N=1'][f'P{int(100*p)}']; ax2.plot([dd], [p], 'o', color=C3, ms=5); ax2.text(dd + 2, p, f'{dd:.0f} keV', color=C3, fontsize=8)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='lower left')
ax.set_title('MiDM: the LZ rate fixes μ(δ), hence τ(δ) and P(clean)')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P065_fig2_tau_pclean.png'), dpi=140); plt.close(fig)
# Fig 3: mu(N=1) vs delta
fig, ax = plt.subplots(figsize=(6.6, 4.4))
ss = SC[(SC.model == 'midm') & (SC.halo == 'sun') & (SC.delta_keV >= 250)].sort_values('delta_keV')
ax.plot(ss.delta_keV, ss.mu_N1_roi_muN, color=C1, lw=2.2, label='μ_χ(N_ROI = 1), Sun frame')
ax.fill_between(ss.delta_keV, ss.mu_N1_roi_muN * math.sqrt(0.3), ss.mu_N1_roi_muN * math.sqrt(2.4), color=C1, alpha=0.15, label='68 % band (0.3–2.4 events)')
sj = SC[(SC.model == 'midm') & (SC.halo == 'june16')].sort_values('delta_keV'); ax.plot(sj.delta_keV, sj.mu_N1_roi_muN, 'o--', color=C1, lw=1, ms=4, label='June 16 halo')
ax.axhline(mu_th, color=C4, lw=1.5, ls=':', label=f'thermal relic μ_th = {mu_th:.1e} μ_N')
ax.axhline(2.1e-5, color=C2, lw=1.2, ls='--', label='P012 elastic photon dipole (excluded)')
ax.axvspan(CEIL['sun|N=1']['P10'], 390, color='#f0efec', label='P(clean) < 0.1 (excluded by the clean event)')
ax.axvline(CEIL['sun|N=1']['P50'], color=C3, lw=1, ls='-.'); ax.text(CEIL['sun|N=1']['P50'] + 1, 3e-5, 'P(clean) = 0.5', color=C3, fontsize=8, rotation=90)
ax.set_yscale('log'); ax.set_xlabel('δ [keV]'); ax.set_ylabel('transition moment μ_χ [μ_N]'); ax.set_xlim(250, 390); ax.set_ylim(2e-5, 0.1)
ax.set_title('MiDM moment for one LZ event, 1 TeV'); ax.legend(fontsize=7.5, loc='upper left')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P065_fig3_mu_vs_delta.png'), dpi=140); plt.close(fig)

# ----------------------------------------------------------------------------------------------
# 11. Summary
# ----------------------------------------------------------------------------------------------
def _clean(o):
    if isinstance(o, dict): return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return float(o)
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)): return None
    return o
summary = dict(
    settings=dict(m_chi_GeV=M_CHI, E_grid='1-328 keV step 3 (+335-995 step 10 for full windows)', halos={'sun': f'WimPyDD single-day Sun frame, v_E = {V_E_SUN:.1f} km/s', 'june16': 'day 167', 'annual': '12-day mean'},
                  efficiency='P003 erf model, 0.96 plateau, 50% at 5.4/269.9 keV', exposure_tyr=EXPOSURE, coefficients='P012/P023 photon-dipole NREFT with WimPyDD c0 = cp+cn', runtime_s=time.time() - T0),
    part_a=dict(sun_300=s300.to_dict(), june_300=j300.to_dict(), annual_300=a300.to_dict(), O1_300=o300.to_dict(), decomposition_300=decomp,
                mu_annual_over_P023=a300.mu_N1_roi_muN / 2.107e-4),
    part_a_decomposition=DEC.to_dict(orient='records'),
    part_b=dict(ceilings_keV=CEIL, at_300_N1=l300.to_dict(), P042_reference_N1=[321, 344, 358], june_over_sun_rate=jr),
    part_d_thermal_band=TH_BAND,
    part_c=dict(delta_window_empty_55keV=d55, delta_window_empty_60keV=d60, loop_elastic_amplitude_ratio=loop_ratio),
    part_d=dict(mu_thermal_muN=mu_th, mu_thermal_ecm=mu_th * MU_N_ECM, Omega_h2_at_N1_300=r300.Omega_h2, sigmav_ff_300=r300.sigmav_ff_cm3s,
                delta_thermal_N1_keV=D_TH, tau_thermal_N1_us=TAU_TH * 1e6, P_clean_thermal_N1=PC_TH, spinor_check=dict(L_dip=L_dip[0, 0].real, L_vec=L_vec[0, 0].real)),
    part_e=dict(v_threshold_upscatter_kms=v_thr, sigma_upscatter_cm2=sig_up, GC_line_flux=flux_gc),
    part_f=dict(target_ratios=RT.to_dict(orient='records'), P046_O1_reference=dict(W_over_Xe=P046_WXE, I_over_Xe=P046_IXE)))
json.dump(_clean(summary), open(os.path.join(OUT, 'P065_summary.json'), 'w'), indent=1)
json.dump(RECALLED, open(os.path.join(OUT, 'recalled_inputs.json'), 'w'), indent=1)
open(os.path.join(OUT, 'P065_run.log'), 'w').write('\n'.join(LOG))
log(f'done in {time.time()-T0:.0f} s; {len(RECALLED)} recalled inputs')
