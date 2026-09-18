#!/usr/bin/env python
"""P076 -- Solar capture of TeV inelastic dark matter (delta = 250-380 keV) with the LZ-fitted couplings,
and the translation of neutrino-telescope (IceCube/ANTARES/Super-K) solar limits to the inelastic capture rate.

Run from the simulation root:  .venv/bin/python output/code/P076_solar_capture.py  [--no-wimpyc]

Physics (all derivations in output/work/P076/details.md):
  * Solar profile: WimPyC `Sun` celestial body = AGSS09ph standard solar model (Serenelli et al. 2009), 135 radial
    points, 20 isotopes, v_esc(r) from 1373 km/s (centre) to 618 km/s (surface). Densities are stored normalised
    to M_sun/R_sun^3 and are converted back to g/cm^3 (mass integral checked).
  * Halo: Sun-frame Baxter-2021 SHM streams (vmin, delta_eta) from lzcommon.wd_halo() (P035 labelling).
  * Kinematics: a DM particle with asymptotic speed u has local speed w^2 = u^2 + v_esc(r)^2; the inelastic recoil
    window on nucleus A is E_+- = (mu^2 w^2/m_A)[1 - delta/(mu w^2) +- sqrt(1 - 2 delta/(mu w^2))]; capture requires
    E_R + delta >= m_chi u^2/2.  delta_max(A, r) = mu w^2/2.
  * Capture (Gould 1987, per stream): C = n_chi sum_r N_A(r) sum_u delta_eta(u) sigma_A c^2 [m_A/(2 mu^2)] int F^2 dE.
  * Two-step (up-scatter then exothermic down-scatter of chi2 before exit) correction and chi2 fate.
  * Annihilation: Griest-Seckel equilibrium (thermal volume) and P039's orbit-confined bracket for a population whose
    thermalisation stalls at v_core < sqrt(2 delta/mu_Fe).
  * Neutrino telescopes: recalled IceCube/ANTARES/Super-K elastic limits converted to Gamma_A via our own elastic
    capture rates (SD on H, SI on all nuclei) at the same mass; channel factors recalled and flagged.
"""
import sys, os, io, json, math, time, argparse, contextlib
import numpy as np
import pandas as pd
sys.path.insert(0, 'output/code')
from common import lzcommon as lz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('--no-wimpyc', action='store_true', help='skip the live WimPyC validation points')
ARGS = ap.parse_args()

OUT = 'output/work/P076'; FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)
RECALLED = []
def rec(item, src, rel):
    RECALLED.append(dict(item=item, presumed_source=src, reliability=rel))

# ----------------------------------------------------------------------------- constants (WimPyC conventions)
Msun = 1.989e33; Rsun = 6.9634e10; c_cms = 2.99792458e10; C_KMS = lz.C_KMS
amu = lz.AMU_GEV; m_N = lz.M_NUCLEON_GEV; GeV_g = 1.0 / 5.62e23   # grams per GeV
t_sun_s = 4.603e9 * 3.156e7
rec('solar age 4.6 Gyr, M_sun = 1.989e33 g, R_sun = 6.9634e10 cm (WimPyC values)', 'standard constants', 'certain')

# ----------------------------------------------------------------------------- solar profile (WimPyC / AGSS09ph)
with contextlib.redirect_stdout(io.StringIO()):
    WD = lz.wd()
S = WD.Sun
unit_rho = Msun / Rsun**3
r_fr = np.asarray(S.r_vec, float); rr = r_fr * Rsun                     # cm
vesc = np.asarray(S.v_esc, float)                                        # km/s
rho_tot = np.asarray(S.rho_tot, float) * unit_rho                        # g/cm^3
rho_i = {k: np.asarray(v, float) * unit_rho for k, v in S.rho_i.items()}
M_int = np.trapezoid(4 * math.pi * rr**2 * rho_tot, rr) / Msun
dr = np.gradient(rr)
print(f'Solar profile (WimPyC Sun = AGSS09ph): {len(rr)} points, v_esc centre {vesc[0]:.0f} / surface {vesc[-1]:.0f} km/s, '
      f'rho_c = {rho_tot[0]:.1f} g/cm3, mass integral {M_int:.4f} Msun, T_c = {S.T_c:.2e} K')
rec('AGSS09ph standard solar model composition/density profile (as shipped with WimPyC); rho_c ~ 150 g/cm3, T_c = 1.4e7 K',
    'Serenelli, Basu, Ferguson, Asplund, ApJL 705, L123 (2009) via WimPyC', 'likely')

# isotope table (symbol -> A, Z); solar mass fractions from the profile
ISO = {'1H': (1.0078, 1), '4He': (4.0026, 2), '3He': (3.0160, 2), '12C': (12.0, 6), '13C': (13.0034, 6), '14N': (14.0031, 7),
       '15N': (15.0001, 7), '16O': (15.9949, 8), '17O': (16.9991, 8), '18O': (17.9992, 8), '20Ne': (19.9924, 10), '23Na': (22.9898, 11),
       '24Mg': (23.985, 12), '27Al': (26.9815, 13), '28Si': (27.9769, 14), '32S': (31.9721, 16), '40Ar': (39.9624, 18),
       '40Ca': (39.9626, 20), '56Fe': (55.9349, 26), '58Ni': (57.9353, 28)}
isos = [k for k in ISO if k in rho_i]
mass_frac = {k: np.trapezoid(4 * math.pi * rr**2 * rho_i[k], rr) / Msun / M_int for k in isos}
# number of nuclei per radial shell and radial column density (outward) per isotope
dN = {k: 4 * math.pi * rr**2 * rho_i[k] / (ISO[k][0] * amu * GeV_g) * dr for k in isos}      # nuclei per shell
n_i = {k: rho_i[k] / (ISO[k][0] * amu * GeV_g) for k in isos}                                 # cm^-3
Ncol_out = {k: np.array([np.trapezoid(n_i[k][j:], rr[j:]) if j < len(rr) - 1 else 0.0 for j in range(len(rr))]) for k in isos}

# ----------------------------------------------------------------------------- halo (Sun frame)
vmin_h, deta_h = lz.wd_halo()
u = vmin_h[1:]; de = deta_h[1:]                     # km/s, (km/s)^-1
inv_v = float(np.sum(de)); mean_v = float(np.sum(de * u**2)); u_max = float(u[de > 0].max())
print(f'Halo (Sun frame, Baxter SHM): <1/v>^-1 = {1/inv_v:.1f} km/s, <v> = {mean_v:.1f} km/s, u_max = {u_max:.0f} km/s, {len(u)} streams')
u_typ = 1 / inv_v
rec('local DM density 0.3-0.4 GeV/cm3 (we use 0.4 as P039; 0.3 shown as variation)', 'Baxter et al. 2021 / de Salas & Widmark 2021', 'likely')

# ----------------------------------------------------------------------------- helpers
def helm_F2v(E_keV, A):
    """Vectorised Helm form factor squared (Lewin-Smith), identical to lz.helm_F2."""
    if A < 1.5:
        return np.ones_like(E_keV)
    mA = lz.m_nucleus_gev(A)
    q = np.sqrt(2 * mA * E_keV * 1e-6) / lz.HBARC_GEV_FM
    s = 0.9; cc = 1.23 * A**(1 / 3) - 0.60; a = 0.52
    rn = math.sqrt(cc * cc + 7 / 3 * math.pi**2 * a * a - 5 * s * s)
    x = np.maximum(q * rn, 1e-9)
    j1 = (np.sin(x) - x * np.cos(x)) / x**2
    return (3 * j1 / x)**2 * np.exp(-(q * s)**2)

def window(mu, mA, w_kms, delta_kev):
    """Inelastic recoil window [E-, E+] in keV for local speed w (km/s); delta<0 is exothermic. Returns (Em, Ep, ok)."""
    b2 = (w_kms / C_KMS)**2
    x = delta_kev * 1e-6 / (mu * b2)
    disc = 1 - 2 * x
    ok = disc > 0
    sq = np.sqrt(np.where(ok, disc, 0))
    pref = mu**2 * b2 / mA * 1e6
    return pref * (1 - x - sq), pref * (1 - x + sq), ok

def sigma_nucleus(sigma_ref, fA, mu_A):
    """sigma_A = sigma_ref * fA * (mu_A/mu_N)^2 with sigma_ref the reference nucleon cross-section."""
    return sigma_ref * fA * (mu_A / lz.mu_red(M_CHI, m_N))**2

# ----------------------------------------------------------------------------- model couplings
GF = 1.1663788e-5; sW2 = 0.23122
rec('G_F = 1.1664e-5 GeV^-2, sin^2 theta_W = 0.2312 (MSbar)', 'PDG', 'certain')
cn_H = -GF / math.sqrt(2); cp_H = (GF / math.sqrt(2)) * (1 - 4 * sW2)
def fA_higgsino(A, Z): return ((Z * cp_H + (A - Z) * cn_H) / cn_H)**2      # relative to sigma_n
def fA_proton(A, Z): return float(Z)**2                                     # dark photon, c_n = 0
def fA_iso(A, Z): return float(A)**2                                        # Z' (c_p = c_n), also elastic SI reference
def fA_H_only(A, Z): return 1.0 if A < 1.5 else 0.0                         # SD-on-hydrogen reference

# P011 dark-photon grid (1 TeV, 'annual' = Sun-frame label, see P035) and P054 Z' grid
p011 = pd.read_csv('output/work/P011/sigma_p_required.csv')
p011 = p011[(p011.m_GeV == 1000) & (p011.halo == 'annual') & (p011.eff_sigma_keV == 8.0)].sort_values('delta_keV')
DP_delta = p011.delta_keV.values; DP_sig = p011.sigma_p_N1_cm2.values
p054 = pd.read_csv('output/work/P054/lz_requirement.csv'); p054 = p054[p054.m_chi_GeV == 1000].sort_values('delta_keV')
ZP_delta = p054.delta_keV.values; ZP_sig = p054.sigma_n_N1_cm2.values; ZP_G = p054.G_N1_GeV2.values
def sigma_DP(d): return float(np.exp(np.interp(d, DP_delta, np.log(DP_sig))))
def sigma_ZP(d): return float(np.exp(np.interp(d, ZP_delta, np.log(ZP_sig))))
def G_ZP(d): return float(np.exp(np.interp(d, ZP_delta, np.log(ZP_G))))
SIG_HIGGSINO = 7.4e-39
print(f'Higgsino: c_p = {cp_H:.3e}, c_n = {cn_H:.3e} GeV^-2; f_A(Fe56) = {fA_higgsino(56,26):.1f}, f_A(Ni58) = {fA_higgsino(58,28):.1f}, f_A(H) = {fA_higgsino(1,1):.4f}')
print(f'P011 dark photon sigma_p(N=1): 250 keV {sigma_DP(250):.2e}, 300 {sigma_DP(300):.2e}, 350 {sigma_DP(350):.2e}, 365 {sigma_DP(365):.2e} cm2')
print(f'P054 Z\' sigma_n(N=1): 250 keV {sigma_ZP(250):.2e}, 300 {sigma_ZP(300):.2e}, 350 {sigma_ZP(350):.2e}, 366 {sigma_ZP(366):.2e} cm2')

# ----------------------------------------------------------------------------- capture integral
M_CHI = 1000.0
NE = 48
def capture(fA_func, sigma_ref, delta_kev, m_chi=M_CHI, rho=0.4, targets=None, return_fail=False, u=u, de=de, NE=NE):
    """Gould capture rate [s^-1] per isotope for inelastic scattering with splitting delta (keV).
    If return_fail, also returns per-isotope arrays (n_r, n_u) of the up-scatter rate that does NOT capture and the
    F^2-weighted E_R grid needed for the two-step correction. u, de: halo streams (km/s, (km/s)^-1)."""
    global M_CHI
    M_CHI = m_chi
    n_chi = rho / m_chi
    KE = 0.5 * m_chi * (u / C_KMS)**2 * 1e6                     # keV, asymptotic kinetic energy per stream
    out = {}; fail = {}
    for k in (targets or isos):
        A, Z = ISO[k]; fA = fA_func(A, Z)
        if fA == 0: out[k] = 0.0; continue
        mA = A * amu; mu = lz.mu_red(m_chi, mA)
        sigA = sigma_nucleus(sigma_ref, fA, mu)
        w = np.sqrt(u[None, :]**2 + vesc[:, None]**2)              # (n_r, n_u)
        Em, Ep, ok = window(mu, mA, w, delta_kev)
        Elo = np.maximum(Em, KE[None, :] - delta_kev)
        good = ok & (Elo < Ep)
        # integrate F^2 over [Elo, Ep] on NE-point grids (only where good)
        I = np.zeros_like(w)
        if good.any():
            t = np.linspace(0, 1, NE)
            Eg = Elo[good][:, None] + (Ep[good] - Elo[good])[:, None] * t[None, :]
            I[good] = np.trapezoid(helm_F2v(Eg, A), Eg, axis=1)
        X = mA / (2 * mu**2) * I * 1e-6                              # dimensionless
        C = n_chi * sigA * c_cms**2 * np.sum(dN[k][:, None] * (de[None, :] / 1e5) * X)
        out[k] = float(C)
        if return_fail:
            # up-scatter that fails to capture: E_R in [Em, min(Elo, Ep)) where ok
            hi = np.minimum(Elo, Ep); fl = ok & (hi > Em)
            If = np.zeros_like(w); Ef = np.zeros((*w.shape, NE)); F2f = np.zeros((*w.shape, NE))
            if fl.any():
                t = np.linspace(0, 1, NE)
                Eg = Em[fl][:, None] + (hi[fl] - Em[fl])[:, None] * t[None, :]
                F2 = helm_F2v(Eg, A)
                If[fl] = np.trapezoid(F2, Eg, axis=1); Ef[fl] = Eg; F2f[fl] = F2
            rate_fail = n_chi * sigA * c_cms**2 * dN[k][:, None] * (de[None, :] / 1e5) * (mA / (2 * mu**2) * If * 1e-6)
            fail[k] = dict(rate=rate_fail, Eg=Ef, F2=F2f, mask=fl, KE=KE, w=w)
    return (out, fail) if return_fail else out

def two_step(fA_func, sigma_ref, delta_kev, up_targets=('56Fe', '58Ni', '40Ca', '40Ar', '32S', '28Si'), m_chi=M_CHI, rho=0.4):
    """Second-scatter capture: chi2 produced in a failed up-scatter (residual energy dE = KE - E_R - delta > 0) down-scatters
    exothermically (window with -delta) on any nucleus along the outward column N_col(r) and is captured if E_R' - delta >= dE.
    Returns the extra capture rate [s^-1] and the total failed up-scatter rate.
    Memory: uses a coarse-grained halo (streams binned by 8, delta_eta summed) and 16-point energy grids; the
    first-step rate on the coarse halo is returned for normalisation (it agrees with the fine one to ~5 %)."""
    nb = 8; n_c = (len(u) // nb) * nb
    u_c = u[:n_c].reshape(-1, nb).mean(axis=1); de_c = de[:n_c].reshape(-1, nb).sum(axis=1)
    NEc = 16
    C1, fail = capture(fA_func, sigma_ref, delta_kev, m_chi=m_chi, rho=rho, targets=list(up_targets), return_fail=True, u=u_c, de=de_c, NE=NEc)
    extra = 0.0; failed_total = 0.0
    for k, f in fail.items():
        mask = f['mask']
        if not mask.any(): continue
        Eg = f['Eg'][mask]; F2 = f['F2'][mask]                         # (n_pairs, NE)
        KEm = np.broadcast_to(f['KE'][None, :], mask.shape)[mask]; wm = f['w'][mask]
        ridx = np.broadcast_to(np.arange(len(rr))[:, None], mask.shape)[mask]
        dE = KEm[:, None] - Eg - delta_kev                              # residual energy (keV), >= 0 by construction
        dE = np.maximum(dE, 0)
        # probability of a capturing exothermic down-scatter along the outward column
        P2 = np.zeros_like(Eg)
        for k2 in isos:
            A2, Z2 = ISO[k2]; fA2 = fA_func(A2, Z2)
            if fA2 == 0: continue
            mA2 = A2 * amu; mu2 = lz.mu_red(m_chi, mA2); sig2 = sigma_nucleus(sigma_ref, fA2, mu2)
            Em2, Ep2, ok2 = window(mu2, mA2, wm, -delta_kev)           # exothermic window (always ok)
            lo = np.maximum(Em2[:, None], dE + delta_kev); hi = np.broadcast_to(Ep2[:, None], Eg.shape)
            g = hi > lo
            if not g.any(): continue
            t = np.linspace(0, 1, 16)
            Eg2 = lo[g][:, None] + (hi[g] - lo[g])[:, None] * t[None, :]
            I2 = np.trapezoid(helm_F2v(Eg2, A2), Eg2, axis=1) * 1e-6    # GeV
            # dsigma/dE = sigma_A m_A/(2 mu^2 w^2) F^2  -> sigma_cap = sigma_A * m_A/(2 mu^2 w^2) * int F^2 dE
            b2 = (np.broadcast_to(wm[:, None], Eg.shape)[g] / C_KMS)**2
            sig_cap = sig2 * mA2 / (2 * mu2**2 * b2) * I2
            Ncol = Ncol_out[k2][np.broadcast_to(ridx[:, None], Eg.shape)[g]]
            P2[g] += sig_cap * Ncol
        P2 = 1 - np.exp(-P2)
        # weight by the F^2 distribution of the failed up-scatter recoils
        wts = F2 / np.maximum(np.trapezoid(F2, Eg, axis=1), 1e-300)[:, None]
        Pbar = np.trapezoid(wts * P2, Eg, axis=1)
        rate = f['rate'][mask]
        extra += float(np.sum(rate * Pbar)); failed_total += float(np.sum(rate))
    return extra, failed_total, C1

# ----------------------------------------------------------------------------- 1. delta_max per element and threshold radii
print('\n=== delta_max = mu w^2/2 per solar isotope (u = <1/v>^-1 = %.0f km/s and u_max = %.0f km/s)' % (u_typ, u_max))
rows = []
for k in isos:
    A, Z = ISO[k]; mu = lz.mu_red(M_CHI, A * amu)
    def dm(w): return 0.5 * mu * (w / C_KMS)**2 * 1e6
    rows.append(dict(isotope=k, A=A, Z=Z, mass_fraction=mass_frac[k],
                     dmax_surface_utyp=dm(math.hypot(vesc[-1], u_typ)), dmax_centre_utyp=dm(math.hypot(vesc[0], u_typ)),
                     dmax_surface_umax=dm(math.hypot(vesc[-1], u_max)), dmax_centre_umax=dm(math.hypot(vesc[0], u_max))))
dmx = pd.DataFrame(rows); dmx.to_csv(os.path.join(OUT, 'delta_max_by_isotope.csv'), index=False)
print(dmx.to_string(index=False, float_format=lambda x: f'{x:.4g}'))
# threshold radius (typical u) for the main isotopes as a function of delta
thr_rows = []
for d in [250, 280, 300, 330, 350, 366, 380]:
    row = dict(delta_keV=d)
    for k in ['24Mg', '28Si', '32S', '40Ar', '40Ca', '56Fe', '58Ni']:
        A, Z = ISO[k]; mu = lz.mu_red(M_CHI, A * amu)
        w_need = math.sqrt(2 * d * 1e-6 / mu) * C_KMS
        v_need = math.sqrt(max(w_need**2 - u_typ**2, 0))
        idx = np.where(vesc >= v_need)[0]
        r_thr = float(r_fr[idx[-1]]) if len(idx) else 0.0
        mfrac = np.trapezoid((4 * math.pi * rr**2 * rho_tot)[r_fr <= r_thr], rr[r_fr <= r_thr]) / Msun if r_thr > 0 else 0.0
        row[f'r_thr_{k}'] = r_thr; row[f'w_need_{k}'] = w_need; row[f'Mfrac_{k}'] = mfrac
    thr_rows.append(row)
thr = pd.DataFrame(thr_rows); thr.to_csv(os.path.join(OUT, 'threshold_radius.csv'), index=False)
print('\nthreshold radius r_thr/R_sun (u_typ) for Fe56:', ', '.join(f'{r.delta_keV}: {r.r_thr_56Fe:.3f}' for r in thr.itertuples()))
print('P039 check (Fe, delta 300/350/366/380): 0.344/0.270/0.255/0.233')

# ----------------------------------------------------------------------------- 2. validation against WimPyC (P039 did 0.1-2 %)
val = {}
t0 = time.time()
val['H elastic iso 1e-42 (own)'] = capture(fA_iso, 1e-42, 0.0, targets=['1H'])['1H']
val['Fe56 elastic iso 1e-42 (own)'] = capture(fA_iso, 1e-42, 0.0, targets=['56Fe'])['56Fe']
val['Fe56 delta=300 iso 1e-42 (own)'] = capture(fA_iso, 1e-42, 300.0, targets=['56Fe'])['56Fe']
print(f'\n=== validation (own Gould, {time.time()-t0:.1f} s): ' + '; '.join(f'{k}: {v:.3e}' for k, v in val.items()))
print('P039 WimPyC: H 2.53e18, Fe elastic 2.42e20, Fe delta=300 2.14e20; P039 own: 2.53e18 / 2.47e20 / 2.16e20')
if not ARGS.no_wimpyc:
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ham = lz.wd_hamiltonian('P076_iso', {1: lz.wd_c_SI_from_sigma_n(1e-42, M_CHI, 1.0)})
            val['H elastic (WimPyC)'] = float(WD.wimp_capture(S, ham, vmin_h, deta_h, mchi=M_CHI, rho_loc=0.4, targets_list=[WD.H1], verbose=False))
            val['Fe56 elastic (WimPyC)'] = float(WD.wimp_capture(S, ham, vmin_h, deta_h, mchi=M_CHI, rho_loc=0.4, targets_list=[WD.Fe56], verbose=False))
            val['Fe56 delta=300 (WimPyC)'] = float(WD.wimp_capture(S, ham, vmin_h, deta_h, mchi=M_CHI, delta=300, rho_loc=0.4, targets_list=[WD.Fe56], verbose=False))
            val['C_geom (WimPyC)'] = float(WD.wimp_capture_geom(S, M_CHI, vmin_h, deta_h, rho_loc=0.4))
        print('WimPyC live: ' + '; '.join(f'{k}: {v:.3e}' for k, v in val.items() if 'WimPyC' in k))
    except Exception as e:
        print('WimPyC validation failed:', repr(e))
# geometric capture (own): C_geom = n_chi pi R^2 sum_u deta u^2 (u^2 + v_esc,s^2)/u ... = n_chi pi R^2 [<v> + v_esc^2 <1/v>]
C_geom_own = 0.4 / M_CHI * math.pi * Rsun**2 * (mean_v * 1e5 + (vesc[-1] * 1e5)**2 * inv_v / 1e5)
val['C_geom (own, surface v_esc)'] = C_geom_own
print(f'geometric capture rate (own) = {C_geom_own:.3e} s^-1')
json.dump(val, open(os.path.join(OUT, 'validation.json'), 'w'), indent=1)

# ----------------------------------------------------------------------------- 3. capture vs delta for the three models
MODELS = {
    'Higgsino': dict(fA=fA_higgsino, sigma=lambda d: SIG_HIGGSINO, label='Higgsino (Z exchange, sigma_n = 7.4e-39 fixed)'),
    'dark photon': dict(fA=fA_proton, sigma=sigma_DP, label="dark photon (proton-only, P011 sigma_p(N=1))"),
    "Z'": dict(fA=fA_iso, sigma=sigma_ZP, label="Z' (isoscalar, P054 sigma_n(N=1))"),
}
deltas = np.array([0] + list(range(200, 381, 10)) + [366, 250, 365], float); deltas = np.unique(deltas)
cap_rows = []
t0 = time.time()
for name, M in MODELS.items():
    for d in deltas:
        if name != 'Higgsino' and (d < 200): continue
        sig = M['sigma'](d) if d > 0 else M['sigma'](300.0)
        Cs = capture(M['fA'], sig, d)
        row = dict(model=name, delta_keV=d, sigma_ref_cm2=sig, C_total=sum(Cs.values()))
        row.update({f'C_{k}': v for k, v in Cs.items()})
        cap_rows.append(row)
cap = pd.DataFrame(cap_rows)
# elastic reference at the same coupling (delta = 0) for each model and delta -> C_inel/C_el
C_EL_UNIT = {name: sum(capture(MODELS[name]['fA'], 1.0, 0.0).values()) for name in MODELS}   # elastic capture per unit sigma_ref (linear)
def C_elastic_same_coupling(name, d):
    return C_EL_UNIT[name] * MODELS[name]['sigma'](d)
cap['C_elastic_same_coupling'] = [C_elastic_same_coupling(r.model, r.delta_keV) if r.delta_keV > 0 else r.C_total for r in cap.itertuples()]
cap['C_inel_over_C_el'] = cap.C_total / cap.C_elastic_same_coupling
print(f'\n=== capture vs delta ({time.time()-t0:.1f} s)')
for name in MODELS:
    sub = cap[cap.model == name]
    print(f'--- {name}')
    for r in sub.itertuples():
        top = sorted([(getattr(r, f'C_{k}'), k) for k in isos], reverse=True)[:3]
        print(f'  delta={r.delta_keV:5.0f}  sigma_ref={r.sigma_ref_cm2:.2e}  C={r.C_total:.3e} s^-1  C/C_el={r.C_inel_over_C_el:.3e}  '
              f'top: ' + ', '.join(f'{k} {v/r.C_total*100:.0f}%' for v, k in top if r.C_total > 0))
cap.to_csv(os.path.join(OUT, 'capture_vs_delta.csv'), index=False)
print('P039 (Fe+Ni only) Higgsino: 4.43e23 (0), 3.88e23 (300), 1.79e23 (350), 1.31e23 (366), 9.6e22 (380); DP 1.03e20 (250), 4.45e20 (300), 7.3e21 (350), 4.25e22 (365)')

# rho = 0.3 variation and mass variation for the Higgsino
var_rows = []
for m, d in [(300.0, 310.0), (1000.0, 366.0), (3000.0, 377.0)]:
    Cs = capture(fA_higgsino, SIG_HIGGSINO, d, m_chi=m); var_rows.append(dict(model='Higgsino', m_chi_GeV=m, delta_keV=d, rho=0.4, C_total=sum(Cs.values())))
M_CHI = 1000.0
for d in [300.0, 366.0]:
    Cs = capture(fA_higgsino, SIG_HIGGSINO, d, rho=0.3); var_rows.append(dict(model='Higgsino', m_chi_GeV=1000.0, delta_keV=d, rho=0.3, C_total=sum(Cs.values())))
var = pd.DataFrame(var_rows); var.to_csv(os.path.join(OUT, 'variations.csv'), index=False)
print('\nvariations:\n' + var.to_string(index=False))

# ----------------------------------------------------------------------------- 4. two-step (down-scatter) correction and chi2 fate
print('\n=== two-step correction (failed up-scatter followed by exothermic down-scatter before exit)')
ts_rows = []
t0 = time.time()
for name, dl in [('Higgsino', [300.0, 350.0, 366.0]), ('dark photon', [250.0, 300.0, 350.0]), ("Z'", [250.0, 300.0, 350.0])]:
    for d in dl:
        M = MODELS[name]
        extra, failed, C1 = two_step(M['fA'], M['sigma'](d), d)
        C1t = sum(C1.values())
        ts_rows.append(dict(model=name, delta_keV=d, C_first_step_heavy=C1t, failed_upscatter_rate=failed, two_step_extra=extra,
                            failed_over_C=failed / C1t, extra_over_C=extra / C1t))
        print(f'  {name} delta={d:.0f}: C1(heavy)={C1t:.3e}, failed up-scatters {failed:.3e} ({failed/C1t*100:.1f}% of C), two-step capture {extra:.3e} ({extra/C1t*100:.3f}% of C)')
ts = pd.DataFrame(ts_rows); ts.to_csv(os.path.join(OUT, 'two_step_correction.csv'), index=False)
print(f'({time.time()-t0:.1f} s)')

# chi2 fate: decay lengths, down-scatter times for bound chi2, energetics of the released delta
print('\n=== chi2 fate')
fate = []
n_H_c = n_i['1H'][0]; n_He_c = n_i['4He'][0]
for name, tau, note in [('Higgsino gamma channel', 0.06, 'P014/P026 tau_gamma ~ 0.01-10 s'), ('Higgsino nu nubar', 1.2e6, 'P026 (300 keV)'),
                        ('dark photon nu nubar (floor)', 7.2e16, 'P011/P026 exothermic floor'), ("Z' B-L nu nubar", 1.7e10, 'P054 (300 keV)')]:
    L = tau * 1000e5 / Rsun          # decay length at w = 1000 km/s in R_sun
    fate.append(dict(channel=name, tau_s=tau, decay_length_Rsun_at_1000kms=L, transit_time_s=2 * Rsun / 1000e5, note=note))
# exothermic down-scatter rate of a bound chi2 at the centre (all isotopes, exothermic window has no threshold): rate = sum n_A sigma_A^exo v
for name, d in [('Higgsino', 366.0), ('dark photon', 300.0), ("Z'", 300.0)]:
    M = MODELS[name]; sig = M['sigma'](d); v_b = 1000.0     # km/s typical bound speed in the core
    rate = 0.0
    for k in isos:
        A, Z = ISO[k]; fA = M['fA'](A, Z)
        if fA == 0: continue
        mA = A * amu; mu = lz.mu_red(M_CHI, mA); sA = sigma_nucleus(sig, fA, mu)
        Em, Ep, ok = window(mu, mA, np.array([v_b]), -d)
        Eg = np.linspace(Em[0], Ep[0], 200); I = np.trapezoid(helm_F2v(Eg, A), Eg) * 1e-6
        sig_exo = sA * mA / (2 * mu**2 * (v_b / C_KMS)**2) * I
        rate += n_i[k][0] * sig_exo * v_b * 1e5
    fate.append(dict(channel=f'{name} bound chi2 down-scatter at centre (delta={d:.0f})', tau_s=1 / rate, decay_length_Rsun_at_1000kms=float('nan'),
                     transit_time_s=float('nan'), note=f'exothermic rate {rate:.2e} s^-1 at v = 1000 km/s'))
E_bind_c = 0.5 * M_CHI * (vesc[0] / C_KMS)**2 * 1e6; E_bind_half = 0.5 * M_CHI * (np.interp(0.5, r_fr, vesc) / C_KMS)**2 * 1e6
fate.append(dict(channel='binding energy m v_esc^2/2 at centre / 0.5 R_sun [keV]', tau_s=E_bind_c, decay_length_Rsun_at_1000kms=E_bind_half, transit_time_s=float('nan'),
                 note='release of delta <= 380 keV cannot unbind a captured particle'))
pd.DataFrame(fate).to_csv(os.path.join(OUT, 'chi2_fate.csv'), index=False)
for f in fate: print('  ', f)

# ----------------------------------------------------------------------------- 5. annihilation and equilibrium
print('\n=== annihilation / equilibrium')
kB = 1.38e-23; Gn = 6.674e-11; conv = 5.62e27
rho_c = rho_tot[0]; Tc = S.T_c
V1 = (3 * kB * Tc / (2 * M_CHI * Gn * rho_c) * conv)**1.5; V2 = (3 * kB * Tc / (4 * M_CHI * Gn * rho_c) * conv)**1.5
V_th = V1**2 / V2
r_th = (3 * V_th / (4 * math.pi))**(1 / 3) / Rsun
print(f'Griest-Seckel thermal volume V_eff = {V_th:.3e} cm^3 (r_th = {r_th:.4f} R_sun); P039: 1.98e26')
def r_apo(delta_kev, A=55.935):
    mu = lz.mu_red(M_CHI, A * amu); vthr = math.sqrt(2 * delta_kev * 1e-6 / mu) * C_KMS
    va = math.sqrt(max(vesc[0]**2 - vthr**2, 0)); idx = np.where(vesc >= va)[0]
    return float(r_fr[idx[-1]]) if len(idx) else 1.0
SIGV = {'Higgsino': (9.6e-27, 2.9e-26, 'P025 WW+ZZ 9.6e-27 x Sommerfeld 1-3'),
        'dark photon': (4.4e-26, 4.4e-26 * 23, "P025 pi alpha_D^2/m^2 = 4.4e-26 x S = 1..23 (Planck-allowed saturation, m_A' >= 9 GeV)"),
        "Z'": (2.2e-26, 5.0e-26, "P054 chi chi -> Z'Z' relic 2.2e-26 x S0 = 1..2.2")}
rec('Sommerfeld enhancement of Higgsino WW/ZZ at v ~ 1e-3 c is O(1-3); dark-photon saturation S_sat = 6 alpha_D m/m_A\'', 'P025 (Hulthen), Cassel 2010', 'likely')
ann_rows = []
for r in cap.itertuples():
    if r.delta_keV < 200 and r.model != 'Higgsino': continue
    lo, hi, note = SIGV[r.model]
    for sv, tag in [(lo, 'sigv_lo'), (hi, 'sigv_hi')]:
        CA = sv / V_th; teq = 1 / math.sqrt(max(r.C_total, 1e-300) * CA)
        GA_th = r.C_total / 2 * math.tanh(t_sun_s / teq)**2
        ra = r_apo(r.delta_keV) if r.delta_keV > 0 else r_th
        Vo = 4 / 3 * math.pi * (ra * Rsun)**3; CAo = sv / Vo; teqo = 1 / math.sqrt(max(r.C_total, 1e-300) * CAo)
        GA_o = r.C_total / 2 * math.tanh(t_sun_s / teqo)**2
        ann_rows.append(dict(model=r.model, delta_keV=r.delta_keV, C_total=r.C_total, sigv=sv, sigv_case=tag, t_eq_thermal_yr=teq / 3.156e7,
                             Gamma_A_thermal=GA_th, r_apo_Rsun=ra, t_eq_orbit_yr=teqo / 3.156e7, Gamma_A_orbit=GA_o))
ann = pd.DataFrame(ann_rows); ann.to_csv(os.path.join(OUT, 'annihilation.csv'), index=False)
for name in MODELS:
    sub = ann[(ann.model == name)]
    print(f'--- {name}: {SIGV[name][2]}')
    for d in [250, 300, 350, 366, 380]:
        s = sub[np.isclose(sub.delta_keV, d)]
        if len(s) == 0: continue
        a, b = s[s.sigv_case == 'sigv_lo'].iloc[0], s[s.sigv_case == 'sigv_hi'].iloc[0]
        print(f'  delta={d}: C={a.C_total:.2e}; thermal eq: t_eq={a.t_eq_thermal_yr:.1e}-{b.t_eq_thermal_yr:.1e} yr, Gamma_A={a.Gamma_A_thermal:.2e}-{b.Gamma_A_thermal:.2e}; '
              f'orbit (r_a={a.r_apo_Rsun:.2f}): t_eq={a.t_eq_orbit_yr:.1e}-{b.t_eq_orbit_yr:.1e} yr, Gamma_A={a.Gamma_A_orbit:.2e}-{b.Gamma_A_orbit:.2e} s^-1')
# evaporation: irrelevant at TeV
E_evap = 3 * 1.38e-16 * Tc / 1.602e-9    # 3kT/2 *2 in keV ~ thermal energy scale
print(f'evaporation: thermal energy 3kT_c = {E_evap:.2f} keV vs m v_esc^2/2 = {E_bind_c/1e3:.1f} MeV at 1 TeV -> evaporation mass ~ 3-4 GeV (recalled), irrelevant')
rec('solar evaporation mass ~ 3-4 GeV for elastic WIMPs', 'Gould 1987; Busoni et al. 2013', 'likely')

# ----------------------------------------------------------------------------- 6. neutrino-telescope limits -> Gamma_A
print('\n=== neutrino telescopes: elastic capture rates for the limit conversions')
# elastic SD on H and SI on all nuclei, per unit cross-section, at 1 TeV and 500 GeV
conv_rows = []
for m in [1000.0, 500.0, 300.0]:
    C_SD = capture(fA_H_only, 1e-40, 0.0, m_chi=m)['1H']
    C_SI = sum(capture(fA_iso, 1e-43, 0.0, m_chi=m).values())
    conv_rows.append({'m_chi_GeV': m, 'C_SD_H_at_1e-40': C_SD, 'C_SI_all_at_1e-43': C_SI, 'Gamma_A_SD_1e-40': C_SD / 2, 'Gamma_A_SI_1e-43': C_SI / 2})
    print(f'  m={m:.0f}: C_SD(H, 1e-40) = {C_SD:.3e}, C_SI(all, 1e-43) = {C_SI:.3e} s^-1 -> Gamma_A = {C_SD/2:.2e} / {C_SI/2:.2e}')
M_CHI = 1000.0
conv = pd.DataFrame(conv_rows); conv.to_csv(os.path.join(OUT, 'elastic_conversion.csv'), index=False)
print('P039 WimPyC: C_H(1e-40, SI-like) = 2.5e20')
# recalled limits (flagged): IceCube 2017 (3 yr, IC79/86) W+W- at 1 TeV: sigma_SD ~ (1-10)e-41, sigma_SI ~ 1e-43 (0.3-3e-43)
LIM = dict(IceCube_WW_1TeV_SD=dict(central=3e-41, lo=1e-41, hi=1e-40), IceCube_WW_1TeV_SI=dict(central=1e-43, lo=3e-44, hi=3e-43),
           IceCube_tautau_1TeV_SD=dict(central=3e-41, lo=1e-41, hi=1e-40), IceCube_bb_1TeV_SD=dict(central=1e-39, lo=3e-40, hi=3e-39),
           ANTARES_WW_1TeV_SD=dict(central=1e-40, lo=5e-41, hi=3e-40), SuperK_max_mass_GeV=200.0)
rec('IceCube 2016-2022 solar WIMP searches (IC79/86, 3-7 yr): at 1 TeV sigma_SD(W+W-) ~ 1e-41 - 1e-40 cm^2, sigma_SI ~ 1e-43 cm^2; tau tau similar, b bbar ~30x weaker',
    'IceCube, EPJC 77, 146 (2017); IceCube PRD 105, 062004 (2022) [low mass]; JCAP 04 (2016) 022', 'uncertain')
rec('ANTARES 11-yr solar WIMP limits: sigma_SD(W+W-, 1 TeV) ~ 1e-40 cm^2 (few e-41 - 3e-40)', 'ANTARES, PLB 759, 69 (2016); PDU 2022', 'uncertain')
rec('Super-K solar WIMP search covers m <= 200 GeV (sigma_SD ~ 1e-39 at 200 GeV, b bbar); no TeV reach', 'Super-K, PRL 114, 141301 (2015)', 'likely')
rec('channel scaling of solar limits at 1 TeV: tau tau ~ W+W- (x1-2), b bbar ~ 20-50x weaker; A\'A\'->4 tau at E_A\' = m_chi behaves like tau tau at m_chi/2 with 2xBR(tau) yield',
    'IceCube 2017 channel comparison; Batell-Pospelov-Ritz-Shang 2010 (secluded DM in the Sun)', 'uncertain')
c1 = conv[conv.m_chi_GeV == 1000].iloc[0]; c05 = conv[conv.m_chi_GeV == 500].iloc[0]
GA_lim = {}
GA_lim['WW 1 TeV (SD)'] = {k: c1['Gamma_A_SD_1e-40'] * v / 1e-40 for k, v in LIM['IceCube_WW_1TeV_SD'].items()}
GA_lim['WW 1 TeV (SI)'] = {k: c1['Gamma_A_SI_1e-43'] * v / 1e-43 for k, v in LIM['IceCube_WW_1TeV_SI'].items()}
GA_lim['tautau 1 TeV (SD)'] = {k: c1['Gamma_A_SD_1e-40'] * v / 1e-40 for k, v in LIM['IceCube_tautau_1TeV_SD'].items()}
GA_lim['bb 1 TeV (SD)'] = {k: c1['Gamma_A_SD_1e-40'] * v / 1e-40 for k, v in LIM['IceCube_bb_1TeV_SD'].items()}
GA_lim['ANTARES WW 1 TeV (SD)'] = {k: c1['Gamma_A_SD_1e-40'] * v / 1e-40 for k, v in LIM['ANTARES_WW_1TeV_SD'].items()}
# dark photon: 2 A' per annihilation, each E = m_chi, BR(A'->tau tau) ~ 0.15 for m_A' = 9-30 GeV -> ~0.3 tau pairs at <E> like a 500 GeV tau tau annihilation
BR_tau = 1 / 6.5
GA_lim["A'A' -> 4f (tau content)"] = {k: c05['Gamma_A_SD_1e-40'] * v / 1e-40 / (2 * BR_tau) for k, v in LIM['IceCube_tautau_1TeV_SD'].items()}
# Z': U(1)_B -> quarks only (bb-like at 500 GeV): 2 Z' -> qq each; B-L: BR(nu nu) = 1.5/6.5, BR(tau tau) = 1/6.5 -> hard neutrinos; treat as tau-like with yield 2*(BR_nu*3 + BR_tau) ~ 1
GA_lim["Z'Z' U(1)_B (b bbar-like, 500 GeV)"] = {k: c05['Gamma_A_SD_1e-40'] * v / 1e-40 / 2 for k, v in LIM['IceCube_bb_1TeV_SD'].items()}
GA_lim["Z'Z' B-L (nu nu + tau tau)"] = {k: c05['Gamma_A_SD_1e-40'] * v / 1e-40 / (2 * (3 * 1.5 / 6.5 + 1 / 6.5)) for k, v in LIM['IceCube_tautau_1TeV_SD'].items()}
rec('a direct nu nubar line is ~3x more constraining than tau tau per annihilation at TeV energies (absorption in the Sun limits the gain)',
    'IceCube/ANTARES channel comparisons; Rott, Siegal-Gaskins, Beacom 2011', 'uncertain')
print('Gamma_A limits [s^-1] (central; lo-hi bracket):')
for k, v in GA_lim.items(): print(f'  {k:40s} {v["central"]:.2e}  ({v["lo"]:.1e} - {v["hi"]:.1e})')
json.dump(dict(limits_sigma=LIM, Gamma_A_limits=GA_lim, BR_tau_Aprime=BR_tau), open(os.path.join(OUT, 'neutrino_limits.json'), 'w'), indent=1, default=float)

# ----------------------------------------------------------------------------- 7. confrontation and classification
print('\n=== confrontation: model Gamma_A (orbit-bracket low .. thermal high) vs neutrino-telescope Gamma_A limit')
def classify(GA_lo, GA_hi, lim):
    """GA_lo: orbit-confined (stalled thermalisation) with the low <sigma v>; GA_hi: thermal equilibrium (= C/2 here).
    'excluded' is robust to the thermalisation question and to the recalled-limit bracket; 'excluded if thermalised' needs the
    Griest-Seckel equilibrium; 'marginal' means the thermal Gamma_A falls inside the limit bracket; 'unconstrained' means even
    the thermal Gamma_A is below the strongest recalled limit."""
    if GA_lo > lim['hi']: return 'excluded'
    if GA_hi > lim['hi']: return 'excluded if thermalised'
    if GA_hi < lim['lo']: return 'unconstrained'
    return 'marginal'
CH = {'Higgsino': ('WW 1 TeV (SD)', 'chi1 chi1 -> WW (54%) + ZZ (46%)'),
      'dark photon': ("A'A' -> 4f (tau content)", "chi1 chi1 -> A'A', A' -> tau tau (BR ~ 0.15, m_A' >= 9 GeV); mu/pi/K stop before decaying"),
      "Z'": ("Z'Z' B-L (nu nu + tau tau)", "chi1 chi1 -> Z'Z' -> 4f; B-L: nu nu 23%, tau tau 15%; U(1)_B: quarks only (b bbar-like)")}
conf_rows = []
for name in MODELS:
    sub = ann[ann.model == name]
    for d in sorted(sub.delta_keV.unique()):
        if d < 200: continue
        s = sub[sub.delta_keV == d]
        GA_lo = float(s.Gamma_A_orbit.min()); GA_hi = float(s.Gamma_A_thermal.max()); GA_eq = float(s.C_total.iloc[0] / 2)
        limkey, chan = CH[name]; lim = GA_lim[limkey]
        row = dict(model=name, delta_keV=d, C=float(s.C_total.iloc[0]), Gamma_A_lo=GA_lo, Gamma_A_hi=GA_hi, Gamma_A_eq=GA_eq,
                   limit_used=limkey, Gamma_lim_central=lim['central'], Gamma_lim_lo=lim['lo'], Gamma_lim_hi=lim['hi'],
                   ratio_lo=GA_lo / lim['central'], ratio_hi=GA_hi / lim['central'], status=classify(GA_lo, GA_hi, lim), channel=chan)
        if name == "Z'":
            limB = GA_lim["Z'Z' U(1)_B (b bbar-like, 500 GeV)"]
            row['status_U1B'] = classify(GA_lo, GA_hi, limB); row['ratio_hi_U1B'] = GA_hi / limB['central']
        conf_rows.append(row)
conf = pd.DataFrame(conf_rows); conf.to_csv(os.path.join(OUT, 'confrontation.csv'), index=False)
for name in MODELS:
    print(f'--- {name}: {CH[name][1]}')
    for r in conf[conf.model == name].itertuples():
        extra = f'  [U(1)_B: {r.status_U1B}, ratio_hi {r.ratio_hi_U1B:.1e}]' if name == "Z'" else ''
        print(f'  delta={r.delta_keV:4.0f}: C={r.C:.2e}, Gamma_A={r.Gamma_A_lo:.2e}..{r.Gamma_A_hi:.2e} vs lim {r.Gamma_lim_central:.1e} ({r.Gamma_lim_lo:.0e}-{r.Gamma_lim_hi:.0e}) '
              f'-> ratio {r.ratio_lo:.1e}..{r.ratio_hi:.1e}: {r.status}{extra}')

# ----------------------------------------------------------------------------- 8. dark photon decay length in the Sun
print("\n=== A' decay length for the P011 epsilon range (1 TeV, delta = 300 keV)")
alpha = 1 / 137.036
def Gamma_Ap(mA, eps):
    """A' -> f fbar width, sum over kinematically open SM fermions (leptons + quarks with N_c, Q_f^2, threshold factors)."""
    ferms = [(0.000511, 1, 1), (0.1057, 1, 1), (1.777, 1, 1), (0.0022, 3, 4 / 9), (0.0047, 3, 1 / 9), (0.095, 3, 1 / 9), (1.27, 3, 4 / 9), (4.18, 3, 1 / 9)]
    G = 0.0
    for mf, Nc, Q2 in ferms:
        if mA > 2 * mf:
            x = (2 * mf / mA)**2
            G += Nc * Q2 * (1 + x / 2) * math.sqrt(1 - x)
    return alpha * eps**2 * mA / 3 * G
dp_rows = []
sig300 = sigma_DP(300.0); mu_p = lz.mu_red(1000.0, 0.938272)
for aD in [0.1, 0.0245]:
    for mA in [1.0, 3.0, 9.0, 30.0]:
        eps = math.sqrt(sig300 / lz.GEV_TO_CM2 * mA**4 / (16 * math.pi * alpha * aD * mu_p**2))
        G = Gamma_Ap(mA, eps); tau = 6.582e-25 / G; gam = 1000.0 / mA
        L = gam * c_cms * tau
        dp_rows.append(dict(alpha_D=aD, m_Ap_GeV=mA, eps=eps, Gamma_GeV=G, tau_s=tau, gamma=gam, decay_length_cm=L, decay_length_over_Rsun=L / Rsun,
                            eps_for_L_eq_Rsun=eps * math.sqrt(L / Rsun)))
dp = pd.DataFrame(dp_rows); dp.to_csv(os.path.join(OUT, 'dark_photon_decay_length.csv'), index=False)
print(dp.to_string(index=False, float_format=lambda x: f'{x:.3g}'))
print('P011 eps(alpha_D=0.1, 1 TeV, 300 keV): 8.7e-7 (1 GeV), 8.3e-5 (10 GeV) -- check against column eps')
rec('muon range in the solar core: dE/dx ~ 2 MeV g^-1 cm^2 x 150 g/cm3 -> a 500 GeV muon stops in ~20 m, far below its 3e6 m decay length; pi/K likewise stop -> only tau, c, b give hard neutrinos',
    'standard Sun-annihilation neutrino lore (Jungman-Kamionkowski-Griest 1996)', 'likely')

# ----------------------------------------------------------------------------- 9. summary JSON
summary = dict(profile=dict(source='WimPyC Sun = AGSS09ph', n_r=len(rr), v_esc_centre=float(vesc[0]), v_esc_surface=float(vesc[-1]), rho_c=float(rho_c),
                            T_c=float(Tc), mass_integral=float(M_int), mass_fractions=mass_frac),
               halo=dict(inv_v_kms=1 / inv_v, mean_v_kms=mean_v, u_max_kms=u_max),
               higgsino=dict(cp=cp_H, cn=cn_H, sigma_n=SIG_HIGGSINO, fA_Fe56=fA_higgsino(56, 26), fA_Ni58=fA_higgsino(58, 28), fA_He4=fA_higgsino(4, 2)),
               validation=val, thermal_volume_cm3=V_th, r_thermal_Rsun=r_th, E_bind_centre_keV=E_bind_c,
               capture=cap.to_dict(orient='records'), confrontation=conf.to_dict(orient='records'), two_step=ts.to_dict(orient='records'),
               Gamma_A_limits=GA_lim, recalled=RECALLED)
json.dump(summary, open(os.path.join(OUT, 'results.json'), 'w'), indent=1, default=float)

# ----------------------------------------------------------------------------- FIGURES (Okabe-Ito, fixed order)
OI = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00', '#56B4E9']
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5})

# Fig 1: Gamma_A vs delta for the three models with the neutrino-telescope bands
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
for ax, (name, col) in zip(axes, zip(MODELS, OI)):
    sub = conf[conf.model == name].sort_values('delta_keV')
    ax.fill_between(sub.delta_keV, sub.Gamma_A_lo, sub.Gamma_A_hi, color=col, alpha=0.3, lw=0, label=r'$\Gamma_A$ (orbit-confined .. thermal eq.)')
    ax.plot(sub.delta_keV, sub.C / 2, color=col, lw=2, label=r'$C/2$ (full equilibrium)')
    lim = GA_lim[CH[name][0]]
    ax.axhspan(lim['lo'], lim['hi'], color='#7f7f7f', alpha=0.25, lw=0, label='recalled neutrino-telescope limit (bracket)')
    ax.axhline(lim['central'], color='#404040', lw=1, ls='--')
    if name == "Z'":
        limB = GA_lim["Z'Z' U(1)_B (b bbar-like, 500 GeV)"]
        ax.axhline(limB['central'], color='#404040', lw=1, ls=':', label="U(1)$_B$ (b$\\bar b$-like) limit")
    ax.set_yscale('log'); ax.set_xlim(200, 385); ax.set_ylim(1e15, 1e25)
    ax.set_title(name, fontsize=10); ax.set_xlabel(r'$\delta$ [keV]')
    ax.text(0.03, 0.03, CH[name][1].split(';')[0], transform=ax.transAxes, fontsize=7, color='#404040')
axes[0].set_ylabel(r'solar annihilation rate $\Gamma_A$ [s$^{-1}$]')
axes[0].legend(fontsize=7, loc='lower left', bbox_to_anchor=(0.0, 0.08), frameon=False)
fig.suptitle(r'Solar capture of LZ-fitted inelastic DM: $m_\chi$ = 1 TeV, $\rho_\chi$ = 0.4 GeV cm$^{-3}$, Sun-frame SHM', fontsize=9, y=0.995)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P076_GammaA_vs_delta.png'), dpi=160); plt.close(fig)

# Fig 2: delta_max per isotope (centre/surface, u_typ and u_max) and the 250-380 keV band
fig, ax = plt.subplots(figsize=(7.5, 3.8))
order = ['1H', '4He', '12C', '14N', '16O', '20Ne', '24Mg', '28Si', '32S', '40Ar', '40Ca', '56Fe', '58Ni']
x = np.arange(len(order)); d = dmx.set_index('isotope').loc[order]
ax.axhspan(250, 380, color='#7f7f7f', alpha=0.2, lw=0, label=r'LZ-fit $\delta$ = 250-380 keV')
ax.vlines(x, d.dmax_surface_utyp, d.dmax_centre_utyp, color=OI[0], lw=3, label=r'$\delta_{\max}$: surface $\to$ centre ($u$ = 288 km/s)')
ax.scatter(x, d.dmax_centre_umax, marker='_', s=120, color=OI[1], lw=2, label=rf'centre, $u_{{\max}}$ = {u_max:.0f} km/s (halo edge)', zorder=3)
ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels(order, rotation=45); ax.set_ylabel(r'$\delta_{\max} = \mu w^2/2$ [keV]')
ax.set_title(r'Kinematic reach of the Sun for 1 TeV inelastic DM (AGSS09ph, $v_{esc}$ = 618-1373 km/s)', fontsize=9)
ax.legend(fontsize=7, frameon=False, loc='lower right')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P076_delta_max_by_isotope.png'), dpi=160); plt.close(fig)

# Fig 3: isotope composition of the capture rate vs delta (Higgsino), and C_inel/C_el for the three models
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
sub = cap[(cap.model == 'Higgsino') & (cap.delta_keV >= 200)].sort_values('delta_keV')
comp = {k: sub[f'C_{k}'].values / sub.C_total.values for k in ['56Fe', '58Ni', '40Ca', '40Ar', '32S', '28Si']}
comp['other'] = 1 - sum(comp.values())
bottom = np.zeros(len(sub)); cols = OI + ['#7f7f7f']
for (k, v), col in zip(comp.items(), cols):
    axes[0].bar(sub.delta_keV, v, bottom=bottom, width=8, color=col, label=k, lw=0); bottom += v
axes[0].set_xlabel(r'$\delta$ [keV]'); axes[0].set_ylabel('fraction of solar capture rate'); axes[0].set_title('Higgsino: which nuclei capture', fontsize=9)
axes[0].legend(fontsize=7, ncol=7, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.22), handlelength=1.0, columnspacing=0.8)
axes[0].set_ylim(0, 1); axes[0].grid(False)
for (name, col) in zip(MODELS, OI):
    s = cap[(cap.model == name) & (cap.delta_keV >= 200)].sort_values('delta_keV')
    axes[1].plot(s.delta_keV, s.C_inel_over_C_el, color=col, lw=2, label=name)
axes[1].set_yscale('log'); axes[1].set_xlabel(r'$\delta$ [keV]'); axes[1].set_ylabel(r'$C_{\rm inelastic}/C_{\rm elastic}$ (same coupling)')
axes[1].set_title('Inelastic suppression of solar capture', fontsize=9); axes[1].legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P076_composition_and_ratio.png'), dpi=160); plt.close(fig)
print('\nfigures written to', FIG)
print(f'recalled items: {len(RECALLED)}')
