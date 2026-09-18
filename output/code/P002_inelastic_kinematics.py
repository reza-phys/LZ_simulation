"""
P002 -- Kinematic map of inelastic dark matter consistent with a 248 keV xenon recoil.

Run from the simulation root:   .venv/bin/python output/code/P002_inelastic_kinematics.py [--no-wimpydd]

Part 1  Kinematics: delta_max(m_chi) for E_R = 216/248/280 keV and several v_max; minimum masses;
        the LZ (m, delta) grid; per-isotope spread.                         -> kinematics_*.csv, fig1
Part 2  WimPyDD O1 isoscalar (unit coupling c^0 = 1/m_v^2) inelastic spectra on 16 June 2023 (day 167),
        m = 400/1000/4000 GeV, delta = 0..400 keV: peak, ROI fractions, central 68 % interval,
        comparison with Fig. 1 of the LZ paper (shape only).                -> spectra_summary.csv, fig2
Part 3  Total in-ROI rate vs delta with a simple efficiency model; June/Dec ratio.   -> rate_vs_delta.csv, fig3
Part 4  Mass degeneracy of the spectral shape for m >= 400 GeV (KS distance).       -> degeneracy.csv, fig4

All WimPyDD spectra are cached in output/work/P002/spectra_cache.npz so the script can be re-run quickly.
"""
from __future__ import annotations
import sys, os, json, math, time, argparse
import numpy as np
sys.path.insert(0, 'output/code')
from common import lzcommon as lz

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'output/work/P002'
FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--no-wimpydd', action='store_true', help='kinematics only')
args = parser.parse_args()

C = lz.C_KMS
E_EVENT, E_ERR = 248.0, math.hypot(23.0, 23.0)          # 248 +- 32.5 keV (stat, sys in quadrature)
E_LO, E_HI = 216.0, 280.0                                # the assignment's rounded +-1 sigma band
DOY_EVENT, DOY_DEC = 167, 336                            # 16 June 2023; 2 December (v_E minimum)
V_E_JUNE = lz.v_earth_kms(DOY_EVENT)                     # 265.1 km/s (lzcommon cosine model)
V_E_AVG = lz.v_earth_kms(None)                           # 250.6 km/s
V_E_DEC = lz.v_earth_kms(DOY_DEC)                        # 235.6 km/s
VESC = lz.VESC_KMS
SCEN = {                                                 # name -> v_max (km/s)
    'June_vesc544': V_E_JUNE + VESC,
    'avg_vesc544': V_E_AVG + VESC,
    'Dec_vesc544': V_E_DEC + VESC,
    'June_vesc500': V_E_JUNE + 500.0,
    'June_vesc600': V_E_JUNE + 600.0,
}
results = {'scenarios_vmax_kms': SCEN, 'E_event_keV': E_EVENT, 'E_err_keV': E_ERR,
           'v_E_kms': {'June16': V_E_JUNE, 'avg': V_E_AVG, 'Dec2': V_E_DEC}}

# ----------------------------------------------------------------------------------------------
# Part 1: kinematics
# ----------------------------------------------------------------------------------------------
def delta_max(E, m, v, A=lz.A_XE_MEAN):
    """delta_max = (v/c) sqrt(2 m_N E) - m_N E / mu   (keV), from v_min(E, delta) = v."""
    return lz.delta_max_kev(E, m, A=A, v_kms=v)

def delta_abs_max(m, v, A=lz.A_XE_MEAN):
    """Largest splitting reachable at any E_R: mu v^2 / 2 (keV), attained at E* = mu^2 v^2 / (2 m_N)."""
    mN = lz.m_nucleus_gev(A); mu = lz.mu_red(m, mN)
    return 0.5 * mu * (v / C) ** 2 * 1e6, mu ** 2 * (v / C) ** 2 / (2 * mN) * 1e6

m_grid = np.logspace(math.log10(40), 5, 400)
E_list = [E_LO, E_EVENT, E_HI]
rows = []
for m in m_grid:
    r = {'m_chi_GeV': m}
    for sc, v in SCEN.items():
        for E in E_list:
            r[f'dmax_E{int(E)}_{sc}'] = delta_max(E, m, v)
        r[f'dabs_{sc}'] = delta_abs_max(m, v)[0]
    rows.append(r)
import csv
with open(os.path.join(OUT, 'kinematics_delta_max.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# minimum mass for each delta and E_R
delta_list = list(range(0, 425, 25))
mm_rows = []
for d in delta_list:
    r = {'delta_keV': d}
    for sc, v in SCEN.items():
        for E in E_list:
            r[f'mmin_E{int(E)}_{sc}'] = lz.m_chi_min_gev(E, v_kms=v, delta_kev=d)
    mm_rows.append(r)
with open(os.path.join(OUT, 'kinematics_min_mass.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(mm_rows[0].keys())); w.writeheader(); w.writerows(mm_rows)

# LZ grid points (Table S7) and Table S6 masses (elastic)
LZ_M = [400, 1000, 4000]; LZ_D = lz.OSIG_DELTAS
grid_rows = []
for m in LZ_M:
    for d in LZ_D:
        r = {'m_chi_GeV': m, 'delta_keV': d}
        for sc in ('June_vesc544', 'avg_vesc544', 'Dec_vesc544'):
            v = SCEN[sc]
            for E in E_list:
                r[f'allowed_E{int(E)}_{sc}'] = int(d <= delta_max(E, m, v))
            r[f'dmax248_{sc}'] = delta_max(E_EVENT, m, v)
            lo, hi = lz.E_R_range_keV(m, v, delta_kev=d)
            r[f'Emin_{sc}'], r[f'Emax_{sc}'] = lo, hi
        r['table_S7_O1s'] = lz.OSIG['O1s'][m][LZ_D.index(d)]
        grid_rows.append(r)
with open(os.path.join(OUT, 'kinematics_grid_points.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(grid_rows[0].keys())); w.writeheader(); w.writerows(grid_rows)

# elastic floor for the Table S6 masses
elastic_rows = []
for m in lz.LSIG_MASSES:
    lo, hi = lz.E_R_range_keV(m, SCEN['June_vesc544'])
    elastic_rows.append({'m_chi_GeV': m, 'Emax_elastic_June_keV': hi,
                         'can_give_216': int(hi >= E_LO), 'can_give_248': int(hi >= E_EVENT), 'can_give_280': int(hi >= E_HI),
                         'L10s_local_sigma': lz.LSIG['L10s'][lz.LSIG_MASSES.index(m)]})
with open(os.path.join(OUT, 'kinematics_tableS6_masses.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(elastic_rows[0].keys())); w.writeheader(); w.writerows(elastic_rows)

# per-isotope spread of delta_max(248) at 1000 GeV, June
iso_rows = [{'A': A, 'abundance': fr, 'dmax248_m1000_June': delta_max(E_EVENT, 1000, SCEN['June_vesc544'], A=A),
             'dmax248_m400_June': delta_max(E_EVENT, 400, SCEN['June_vesc544'], A=A)} for A, fr in lz.XE_ISOTOPES.items()]
with open(os.path.join(OUT, 'kinematics_isotopes.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(iso_rows[0].keys())); w.writeheader(); w.writerows(iso_rows)

kin = {}
for sc, v in SCEN.items():
    kin[sc] = {'v_max_kms': v,
               'dmax_248': {m: delta_max(E_EVENT, m, v) for m in (100, 200, 400, 1000, 4000, 1e5)},
               'dmax_216': {m: delta_max(E_LO, m, v) for m in (400, 1000, 4000)},
               'dmax_280': {m: delta_max(E_HI, m, v) for m in (400, 1000, 4000)},
               'dabs_max': {m: delta_abs_max(m, v)[0] for m in (400, 1000, 4000, 1e5)},
               'E_star_keV': {m: delta_abs_max(m, v)[1] for m in (400, 1000, 4000)},
               'mmin_elastic': {E: lz.m_chi_min_gev(E, v_kms=v) for E in E_list},
               'mmin_delta': {d: lz.m_chi_min_gev(E_EVENT, v_kms=v, delta_kev=d) for d in (100, 200, 300, 350, 380, 400)},
               'mmin_delta_E216': {d: lz.m_chi_min_gev(E_LO, v_kms=v, delta_kev=d) for d in (300, 350)},
               'mmin_delta_E280': {d: lz.m_chi_min_gev(E_HI, v_kms=v, delta_kev=d) for d in (300, 350)},
               }
results['kinematics'] = kin
results['isotope_spread_dmax248_m1000_June_keV'] = [min(r['dmax248_m1000_June'] for r in iso_rows), max(r['dmax248_m1000_June'] for r in iso_rows)]
results['isotope_spread_dmax248_m400_June_keV'] = [min(r['dmax248_m400_June'] for r in iso_rows), max(r['dmax248_m400_June'] for r in iso_rows)]

# Figure 1: (m, delta) plane
fig, ax = plt.subplots(figsize=(6.4, 4.6))
vJ = SCEN['June_vesc544']
d248 = np.array([delta_max(E_EVENT, m, vJ) for m in m_grid])
d216 = np.array([delta_max(E_LO, m, vJ) for m in m_grid])
d280 = np.array([delta_max(E_HI, m, vJ) for m in m_grid])
dabs = np.array([delta_abs_max(m, vJ)[0] for m in m_grid])
ax.fill_between(m_grid, 0, np.clip(d248, 0, None), color='tab:blue', alpha=0.18, label=r'allowed for $E_R=248$ keV, 16 June ($v_{\max}=809$ km/s)')
ax.fill_between(m_grid, np.clip(np.minimum(d216, d280), 0, None), np.clip(np.maximum(d216, d280), 0, None), color='tab:blue', alpha=0.35,
                label=r'$\delta_{\max}$ for $E_R=216\ldots280$ keV ($\pm1\sigma$)')
ax.plot(m_grid, d248, color='tab:blue', lw=2)
ax.plot(m_grid, [delta_max(E_EVENT, m, SCEN['avg_vesc544']) for m in m_grid], color='k', ls='--', lw=1, label=r'248 keV, annual average ($v_{\max}=795$)')
ax.plot(m_grid, [delta_max(E_EVENT, m, SCEN['Dec_vesc544']) for m in m_grid], color='k', ls=':', lw=1, label=r'248 keV, December ($v_{\max}=780$)')
ax.plot(m_grid, [delta_max(E_EVENT, m, SCEN['June_vesc600']) for m in m_grid], color='tab:red', ls='-.', lw=1, label=r'248 keV, June, $v_{esc}=600$')
ax.plot(m_grid, [delta_max(E_EVENT, m, SCEN['June_vesc500']) for m in m_grid], color='tab:orange', ls='-.', lw=1, label=r'248 keV, June, $v_{esc}=500$')
ax.plot(m_grid, dabs, color='grey', lw=0.8, label=r'$\mu v_{\max}^2/2$ (any $E_R$)')
for m in LZ_M:
    for d in LZ_D:
        ok = d <= delta_max(E_EVENT, m, vJ)
        ax.plot(m, d, marker='o' if ok else 'x', color='tab:green' if ok else 'tab:red', ms=6 if ok else 9, mew=2, ls='none')
ax.annotate('(400 GeV, 350 keV): "-" in Table S7', xy=(400, 350), xytext=(60, 385), fontsize=8, arrowprops=dict(arrowstyle='->', lw=0.8))
ax.set_xscale('log'); ax.set_xlim(40, 1e5); ax.set_ylim(0, 450)
ax.set_xlabel(r'$m_\chi$ [GeV]'); ax.set_ylabel(r'mass splitting $\delta$ [keV]')
ax.axvline(kin['June_vesc544']['mmin_elastic'][E_EVENT], color='tab:blue', lw=0.8, ls='--')
ax.text(kin['June_vesc544']['mmin_elastic'][E_EVENT] * 1.05, 20, r'$m_{\min}^{el}=%.0f$ GeV' % kin['June_vesc544']['mmin_elastic'][E_EVENT], fontsize=8)
ax.legend(fontsize=6.5, loc='lower right', framealpha=0.9)
ax.set_title(r'Kinematically allowed $(m_\chi,\delta)$ for a 248 keV Xe recoil (SHM, Baxter 2021)', fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig1_kinematic_map.png'), dpi=160); plt.close(fig)

print('--- Part 1 kinematics ---')
for sc in SCEN:
    k = kin[sc]
    print(f"{sc:14s} vmax={k['v_max_kms']:.1f}  dmax248(400/1000/4000)= {k['dmax_248'][400]:.1f}/{k['dmax_248'][1000]:.1f}/{k['dmax_248'][4000]:.1f}"
          f"  mmin_el(216/248/280)= {k['mmin_elastic'][E_LO]:.1f}/{k['mmin_elastic'][E_EVENT]:.1f}/{k['mmin_elastic'][E_HI]:.1f}"
          f"  mmin(d=300)={k['mmin_delta'][300]:.0f} mmin(d=350)={k['mmin_delta'][350]:.0f}")

if args.no_wimpydd:
    json.dump(results, open(os.path.join(OUT, 'results.json'), 'w'), indent=1, default=float)
    sys.exit(0)

# ----------------------------------------------------------------------------------------------
# Part 2: WimPyDD spectra (O1 isoscalar, unit coupling), day 167, extended v_min grid
# ----------------------------------------------------------------------------------------------
VGRID = np.linspace(0, 830, 2000)[1:]     # WimPyDD's default grid stops at v_esc+|v_sun| = 794.6 km/s and
                                           # drops the June tail (795-810 km/s); an explicit grid fixes this.
halo_J = lz.wd_halo(day_of_year=DOY_EVENT, vmin=VGRID)
halo_D = lz.wd_halo(day_of_year=DOY_DEC, vmin=VGRID)
halo_A = lz.wd_halo(vmin=VGRID)
def eta_max_v(h):
    v, d = h; return float(v[d > 0].max())
results['wimpydd_vmax_kms'] = {'June': eta_max_v(halo_J), 'Dec': eta_max_v(halo_D), 'avg': eta_max_v(halo_A)}
results['wimpydd_default_grid_vmax_kms'] = float(lz.wd_halo(day_of_year=DOY_EVENT)[0].max())
print('WimPyDD v_max (eta>0) June/Dec/avg:', results['wimpydd_vmax_kms'], ' default grid max:', results['wimpydd_default_grid_vmax_kms'])

HAM = lz.wd_hamiltonian('O1s_unit', {1: (1.0 / lz.M_V_GEV ** 2, 0.0)})
E = np.arange(1.0, 421.0, 1.0)            # keV, 1 keV steps
MASSES = [400, 1000, 4000]
DELTAS = sorted(set(list(range(0, 425, 25)) + [310, 320, 330, 340, 360, 370, 380, 390]))
DELTAS_DEC = list(range(200, 425, 25))
EXTRA = [(100, d) for d in (0, 100, 200)] + [(200, d) for d in (0, 100, 200, 300)]

cache_path = os.path.join(OUT, 'spectra_cache.npz')
cache = dict(np.load(cache_path, allow_pickle=True)) if os.path.exists(cache_path) else {}
cache = {k: v for k, v in cache.items()}
def spec(key, m, d, halo):
    if key in cache:
        return cache[key]
    t = time.time()
    r = lz.wd_rate(HAM, m, E, halo=halo, delta_kev=d)
    r = np.where(r < 0, 0.0, r)            # tiny negative values can occur at the kinematic edge; clip
    cache[key] = r
    np.savez(cache_path, **cache)
    print(f'  computed {key} in {time.time()-t:.1f}s', flush=True)
    return r
if 'E' not in cache:
    cache['E'] = E

def moments(r, E=E):
    """peak, fractions and quantiles of a spectrum on the 1 keV grid."""
    tot_all = np.trapezoid(r, E)
    roi = (E >= 5.4) & (E <= 270.0)
    low = (E >= 5.4) & (E <= 55.0)
    hi = (E >= 200.0) & (E <= 270.0)
    above = E > 270.0
    out = {'peak_keV': float(E[np.argmax(r)]), 'peak_rate': float(r.max()),
           'onset_keV': float(E[r > 0].min()) if (r > 0).any() else float('nan'),
           'R_total_1_420': float(tot_all),
           'R_roi_5.4_270': float(np.trapezoid(r[roi], E[roi])),
           'R_5.4_55': float(np.trapezoid(r[low], E[low])),
           'R_200_270': float(np.trapezoid(r[hi], E[hi])),
           'R_above_270': float(np.trapezoid(r[above], E[above]))}
    out['f_200_270_of_roi'] = out['R_200_270'] / out['R_roi_5.4_270'] if out['R_roi_5.4_270'] > 0 else float('nan')
    out['f_5.4_55_of_roi'] = out['R_5.4_55'] / out['R_roi_5.4_270'] if out['R_roi_5.4_270'] > 0 else float('nan')
    out['ratio_200_270_over_5.4_55'] = out['R_200_270'] / out['R_5.4_55'] if out['R_5.4_55'] > 0 else float('inf')
    out['f_above_270_of_total'] = out['R_above_270'] / tot_all if tot_all > 0 else float('nan')
    # in-ROI quantiles (unweighted)
    if out['R_roi_5.4_270'] > 0:
        cdf = np.concatenate([[0], np.cumsum(0.5 * (r[roi][1:] + r[roi][:-1]) * np.diff(E[roi]))]); cdf /= cdf[-1]
        q = np.interp([0.16, 0.5, 0.84], cdf, E[roi])
        out['q16_keV'], out['q50_keV'], out['q84_keV'] = map(float, q)
        out['peak_in_roi_keV'] = float(E[roi][np.argmax(r[roi])])
        out['contains_248_68'] = int(q[0] <= 248.0 <= q[2])
        out['cdf_at_248'] = float(np.interp(248.0, E[roi], cdf))
        out['cdf_at_216'] = float(np.interp(216.0, E[roi], cdf))
        out['cdf_at_280'] = float(cdf[-1])
    else:
        for k in ('q16_keV', 'q50_keV', 'q84_keV', 'peak_in_roi_keV', 'cdf_at_248', 'cdf_at_216'):
            out[k] = float('nan')
        out['contains_248_68'] = 0
    return out

print('--- Part 2 spectra (June, day 167) ---')
summ = []
spectra_J = {}
for m in MASSES:
    for d in DELTAS:
        r = spec(f'J_{m}_{d}', m, d, halo_J); spectra_J[(m, d)] = r
        mo = moments(r); mo.update(m_chi_GeV=m, delta_keV=d, halo='June167')
        lo, hi = lz.E_R_range_keV(m, SCEN['June_vesc544'], delta_kev=d)
        mo['Emin_kin_keV'], mo['Emax_kin_keV'] = lo, hi
        summ.append(mo)
for m, d in EXTRA:
    r = spec(f'J_{m}_{d}', m, d, halo_J); spectra_J[(m, d)] = r
    mo = moments(r); mo.update(m_chi_GeV=m, delta_keV=d, halo='June167')
    lo, hi = lz.E_R_range_keV(m, SCEN['June_vesc544'], delta_kev=d); mo['Emin_kin_keV'], mo['Emax_kin_keV'] = lo, hi
    summ.append(mo)
spectra_D = {}
for m in MASSES:
    for d in DELTAS_DEC:
        r = spec(f'D_{m}_{d}', m, d, halo_D); spectra_D[(m, d)] = r
        mo = moments(r); mo.update(m_chi_GeV=m, delta_keV=d, halo='Dec336')
        lo, hi = lz.E_R_range_keV(m, SCEN['Dec_vesc544'], delta_kev=d); mo['Emin_kin_keV'], mo['Emax_kin_keV'] = lo, hi
        summ.append(mo)
spectra_A = {}
for d in range(0, 425, 25):
    r = spec(f'A_1000_{d}', 1000, d, halo_A); spectra_A[(1000, d)] = r
    mo = moments(r); mo.update(m_chi_GeV=1000, delta_keV=d, halo='annual_avg')
    lo, hi = lz.E_R_range_keV(1000, SCEN['avg_vesc544'], delta_kev=d); mo['Emin_kin_keV'], mo['Emax_kin_keV'] = lo, hi
    summ.append(mo)
np.savez(cache_path, **cache)
keys = ['halo', 'm_chi_GeV', 'delta_keV'] + [k for k in summ[0] if k not in ('halo', 'm_chi_GeV', 'delta_keV')]
with open(os.path.join(OUT, 'spectra_summary.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(summ)

# delta range where 248 keV lies inside the central 68 % (June, per mass)
rng68 = {}
for m in MASSES:
    ok = [d for d in DELTAS if next(s for s in summ if s['halo'] == 'June167' and s['m_chi_GeV'] == m and s['delta_keV'] == d)['contains_248_68']]
    rng68[m] = (min(ok), max(ok)) if ok else None
results['delta_range_248_in_central68_June'] = rng68
# Fig.1 comparison numbers (1000 GeV, June and avg): onset, peak, dip
def dip(r):
    msk = (E > 200) & (E < 300); return float(E[msk][np.argmin(r[msk])])
fig1cmp = {}
for d in (0, 200, 300):
    rJ = spectra_J[(1000, d)]; rA = spectra_A[(1000, d)]
    fig1cmp[d] = {'onset_June': float(E[rJ > 0].min()), 'peak_June': float(E[np.argmax(rJ)]), 'peak_rate_June': float(rJ.max()),
                  'dip_June': dip(rJ), 'onset_avg': float(E[rA > 0].min()), 'peak_rate_avg': float(rA.max()), 'dip_avg': dip(rA),
                  'rate_10keV_avg': float(rA[np.argmin(abs(E - 10))]), 'rate_248_June': float(rJ[np.argmin(abs(E - 248))]),
                  'rate_248_avg': float(rA[np.argmin(abs(E - 248))])}
    if d == 200:
        msk = (E > 50) & (E < 150); fig1cmp[d]['dip100_June'] = float(E[msk][np.argmin(rJ[msk])])
    if d == 0:
        msk = (E > 50) & (E < 150); fig1cmp[d]['dip100_June'] = float(E[msk][np.argmin(rJ[msk])])
results['fig1_comparison_1000GeV'] = fig1cmp
print('Fig.1 comparison:', json.dumps(fig1cmp, indent=0))

# Figure 2: spectra
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
ax = axs[0]
cols = plt.cm.viridis(np.linspace(0, 0.95, 9))
for i, d in enumerate([0, 100, 200, 250, 300, 325, 350, 375, 390]):
    r = spectra_J[(1000, d)]
    ax.plot(E, r, color=cols[i], lw=1.4, label=fr'$\delta={d}$ keV')
ax.axvspan(0, 5.4, color='grey', alpha=0.3); ax.axvspan(269.9, 420, color='grey', alpha=0.3)
ax.axvspan(E_LO, E_HI, color='tab:red', alpha=0.15); ax.axvline(248, color='tab:red', lw=1)
ax.set_yscale('log'); ax.set_ylim(1e-4, 1e8); ax.set_xlim(0, 420)
ax.set_xlabel('true recoil energy [keV]'); ax.set_ylabel(r'$dR/dE_R$ [/t/yr/keV], WimPyDD $c_1^0=1/m_v^2$ (normalisation: see P003)')
ax.set_title(r'$\mathcal{O}_1^s$ inelastic, $m_\chi=1000$ GeV, halo of 16 June 2023', fontsize=9)
ax.legend(fontsize=7, ncol=2)
ax = axs[1]
for i, d in enumerate([200, 250, 300, 325, 350, 375, 390]):
    r = spectra_J[(1000, d)]; roi = (E >= 5.4) & (E <= 270)
    n = np.trapezoid(r[roi], E[roi])
    if n > 0:
        ax.plot(E, r / n, color=cols[i + 2], lw=1.4, label=fr'$\delta={d}$ keV')
ax.axvspan(E_LO, E_HI, color='tab:red', alpha=0.15); ax.axvline(248, color='tab:red', lw=1); ax.axvspan(269.9, 420, color='grey', alpha=0.3)
ax.set_xlim(0, 420); ax.set_ylim(0, 0.02)
ax.set_xlabel('true recoil energy [keV]'); ax.set_ylabel('shape normalised to unit area in 5.4-270 keV')
ax.set_title('shape only: where the in-ROI spectrum sits relative to 248 keV', fontsize=9); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig2_spectra_1000GeV.png'), dpi=160); plt.close(fig)

# ----------------------------------------------------------------------------------------------
# Part 3: total in-ROI rate vs delta with a simple efficiency model
# ----------------------------------------------------------------------------------------------
def eff(E):
    """Piecewise-linear NR efficiency: 50% at 5.4 keV, 96% plateau 14-250 keV, 50% at 269.9 keV, 0 at 289.8 keV.
    (Anchors: LZ paper Fig. S2 caption and text; the plateau value is the paper's 96% average.)"""
    xp = [0, 5.4, 14.0, 250.0, 269.9, 289.8, 500]
    fp = [0, 0.50, 0.96, 0.96, 0.50, 0.0, 0.0]
    return np.interp(E, xp, fp)
EFF = eff(E)
def R_eff(r):
    return float(np.trapezoid(r * EFF, E))
def R_sharp(r):
    roi = (E >= 5.4) & (E <= 270.0); return float(0.96 * np.trapezoid(r[roi], E[roi]))
rate_rows = []
for m in MASSES:
    for d in DELTAS:
        rJ = spectra_J[(m, d)]
        row = {'m_chi_GeV': m, 'delta_keV': d, 'R_June_eff': R_eff(rJ), 'R_June_sharp': R_sharp(rJ),
               'R_June_200_270_eff': float(np.trapezoid((rJ * EFF)[(E >= 200) & (E <= 270)], E[(E >= 200) & (E <= 270)])),
               'R_June_total_noeff': float(np.trapezoid(rJ, E))}
        if (m, d) in spectra_D:
            rD = spectra_D[(m, d)]; row['R_Dec_eff'] = R_eff(rD); row['June_over_Dec'] = row['R_June_eff'] / row['R_Dec_eff'] if row['R_Dec_eff'] > 0 else float('inf')
        if (m, d) in spectra_A:
            rA = spectra_A[(m, d)]; row['R_avg_eff'] = R_eff(rA); row['June_over_avg'] = row['R_June_eff'] / row['R_avg_eff'] if row['R_avg_eff'] > 0 else float('inf')
        rate_rows.append(row)
keys = ['m_chi_GeV', 'delta_keV', 'R_June_eff', 'R_June_sharp', 'R_June_200_270_eff', 'R_June_total_noeff', 'R_Dec_eff', 'June_over_Dec', 'R_avg_eff', 'June_over_avg']
with open(os.path.join(OUT, 'rate_vs_delta.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rate_rows)
def R(m, d, key='R_June_eff'):
    return next(r for r in rate_rows if r['m_chi_GeV'] == m and r['delta_keV'] == d)[key]
ratios = {}
for m in MASSES:
    r300 = R(m, 300)
    ratios[m] = {'R300_June_eff': r300, 'R0_June_eff': R(m, 0), 'R200_June_eff': R(m, 200),
                 'r350_over_r300': R(m, 350) / r300, 'r380_over_r300': R(m, 380) / r300 if r300 > 0 else None,
                 'r325_over_r300': R(m, 325) / r300, 'r400_over_r300': R(m, 400) / r300,
                 'r300_over_r0': r300 / R(m, 0), 'r300_over_r200': r300 / R(m, 200),
                 'June_over_Dec_300': R(m, 300, 'June_over_Dec'), 'June_over_Dec_350': R(m, 350, 'June_over_Dec'),
                 'June_over_Dec_250': R(m, 250, 'June_over_Dec'), 'June_over_Dec_200': R(m, 200, 'June_over_Dec')}
    # delta at which the June in-ROI rate has fallen to 10% and 1% of R(300): log-interpolate on the delta grid
    dd = np.array([d for d in DELTAS if d >= 300]); rr = np.array([R(m, d) for d in dd])
    pos = rr > 0
    for frac, name in ((0.1, 'delta_10pct_of_R300'), (0.01, 'delta_1pct_of_R300')):
        if (rr[pos] / r300).min() <= frac:
            ratios[m][name] = float(np.interp(math.log(frac), np.log(rr[pos] / r300)[::-1], dd[pos][::-1]))
        else:
            ratios[m][name] = None
results['rate_vs_delta'] = ratios
results['June_over_avg_1000'] = {d: R(1000, d, 'June_over_avg') for d in range(0, 425, 25)}
print('--- Part 3 rate ratios ---'); print(json.dumps(ratios, indent=0, default=float))

fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2))
ax = axs[0]
for m, c in zip(MASSES, ('tab:orange', 'tab:blue', 'tab:green')):
    dd = np.array(DELTAS); rr = np.array([R(m, d) for d in dd])
    ax.plot(dd, rr, 'o-', ms=3, color=c, label=fr'$m_\chi={m}$ GeV, June')
    ddD = np.array(DELTAS_DEC); rrD = np.array([R(m, d, 'R_Dec_eff') for d in ddD])
    ax.plot(ddD, rrD, 's--', ms=3, color=c, alpha=0.6, label=fr'$m_\chi={m}$ GeV, Dec')
    ax.axvline(kin['June_vesc544']['dmax_248'][m], color=c, lw=0.8, ls=':')
ax.set_yscale('log'); ax.set_xlabel(r'$\delta$ [keV]'); ax.set_ylabel(r'in-ROI rate $\int \epsilon\, dR/dE$ [/t/yr], unit $c_1^0$ (norm.: P003)')
ax.set_title(r'Total efficiency-weighted rate vs $\delta$; dotted: $\delta_{\max}(248\,\mathrm{keV})$', fontsize=9); ax.legend(fontsize=7); ax.set_ylim(1e-6, 1e8)
ax = axs[1]
for m, c in zip(MASSES, ('tab:orange', 'tab:blue', 'tab:green')):
    dd = np.array(DELTAS); rr = np.array([R(m, d) / R(m, 300) for d in dd])
    ax.plot(dd, rr, 'o-', ms=3, color=c, label=fr'$m_\chi={m}$ GeV')
ax.set_yscale('log'); ax.set_xlim(250, 410); ax.set_ylim(1e-6, 30)
ax.set_xlabel(r'$\delta$ [keV]'); ax.set_ylabel(r'$R(\delta)/R(300\ \mathrm{keV})$, June')
ax.set_title('Rate collapse towards the kinematic edge', fontsize=9); ax.legend(fontsize=7); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig3_rate_vs_delta.png'), dpi=160); plt.close(fig)

# ----------------------------------------------------------------------------------------------
# Part 4: mass degeneracy (shape metric: KS distance of the in-ROI normalised CDFs)
# ----------------------------------------------------------------------------------------------
def cdf_roi(r):
    roi = (E >= 5.4) & (E <= 270)
    c = np.concatenate([[0], np.cumsum(0.5 * (r[roi][1:] + r[roi][:-1]) * np.diff(E[roi]))])
    return E[roi], c / c[-1] if c[-1] > 0 else c
def ks(r1, r2):
    x1, c1 = cdf_roi(r1); x2, c2 = cdf_roi(r2); return float(np.max(np.abs(c1 - c2)))
deg_rows = []
pairs = [(400, 1000), (1000, 4000), (400, 4000), (200, 1000), (100, 1000)]
for d in [0, 100, 200, 250, 300, 325, 350]:
    row = {'delta_keV': d}
    for a, b in pairs:
        if (a, d) in spectra_J and (b, d) in spectra_J and spectra_J[(a, d)].max() > 0 and spectra_J[(b, d)].max() > 0:
            row[f'KS_{a}_{b}'] = ks(spectra_J[(a, d)], spectra_J[(b, d)])
            row[f'dpeak_{a}_{b}_keV'] = moments(spectra_J[(a, d)])['peak_in_roi_keV'] - moments(spectra_J[(b, d)])['peak_in_roi_keV']
            row[f'f200_270_{a}'] = moments(spectra_J[(a, d)])['f_200_270_of_roi']; row[f'f200_270_{b}'] = moments(spectra_J[(b, d)])['f_200_270_of_roi']
        else:
            row[f'KS_{a}_{b}'] = float('nan')
    deg_rows.append(row)
allk = sorted({k for r in deg_rows for k in r})
with open(os.path.join(OUT, 'degeneracy.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['delta_keV'] + [k for k in allk if k != 'delta_keV']); w.writeheader(); w.writerows(deg_rows)
results['degeneracy_KS'] = {r['delta_keV']: {k: v for k, v in r.items() if k.startswith('KS')} for r in deg_rows}
print('--- Part 4 degeneracy ---')
for r in deg_rows:
    print(r['delta_keV'], {k: round(v, 4) for k, v in r.items() if k.startswith('KS') and v == v})

fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), sharey=False)
for ax, d in zip(axs, (0, 200, 300)):
    for m, c in ((100, 'tab:red'), (200, 'tab:purple'), (400, 'tab:orange'), (1000, 'tab:blue'), (4000, 'tab:green')):
        if (m, d) in spectra_J:
            x, cdf = cdf_roi(spectra_J[(m, d)]); ax.plot(x, cdf, color=c, label=f'{m} GeV')
    ax.axvline(248, color='k', lw=0.8); ax.set_title(fr'$\delta={d}$ keV: in-ROI CDF', fontsize=9); ax.set_xlabel('E_R [keV]')
axs[0].set_ylabel('normalised CDF (5.4-270 keV)'); axs[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig4_mass_degeneracy.png'), dpi=160); plt.close(fig)

# ----------------------------------------------------------------------------------------------
# Part 5: where does 248 +- 32 keV sit?  Fraction of the efficiency-weighted in-ROI spectrum inside 216-280 keV,
#         and the percentile of 248 keV, for each (m, delta), June halo.
# ----------------------------------------------------------------------------------------------
band_rows = []
for (m, d), r in sorted(spectra_J.items()):
    w = r * EFF
    tot = np.trapezoid(w, E)
    if tot <= 0:
        continue
    inb = (E >= E_LO) & (E <= E_HI)
    cdf = np.concatenate([[0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(E))]) / tot
    band_rows.append({'m_chi_GeV': m, 'delta_keV': d, 'f_band_216_280_eff': float(np.trapezoid(w[inb], E[inb]) / tot),
                      'percentile_248_eff': float(np.interp(248.0, E, cdf)), 'percentile_216_eff': float(np.interp(216.0, E, cdf)),
                      'f_above_248_eff': float(1 - np.interp(248.0, E, cdf))})
with open(os.path.join(OUT, 'band_fraction.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(band_rows[0].keys())); w.writeheader(); w.writerows(band_rows)
results['band_216_280'] = {f'{r["m_chi_GeV"]}_{r["delta_keV"]}': r for r in band_rows if r['delta_keV'] in (200, 250, 300, 325, 350, 375, 380, 390, 400)}
# delta at which the band fraction first exceeds 0.5 and 0.16 (June), per mass
for m in MASSES:
    br = [r for r in band_rows if r['m_chi_GeV'] == m]
    results[f'delta_first_band_gt_0.16_m{m}'] = next((r['delta_keV'] for r in br if r['f_band_216_280_eff'] > 0.16), None)
    results[f'delta_first_band_gt_0.50_m{m}'] = next((r['delta_keV'] for r in br if r['f_band_216_280_eff'] > 0.50), None)
print('--- Part 5 band fractions (June, eff-weighted) ---')
for r in band_rows:
    if r['delta_keV'] in (250, 300, 325, 350, 375, 380, 390, 400) and r['m_chi_GeV'] in MASSES:
        print(r)

# summary table for the paper: 1000 GeV June
tab = [{k: s[k] for k in ('m_chi_GeV', 'delta_keV', 'onset_keV', 'peak_in_roi_keV', 'q16_keV', 'q50_keV', 'q84_keV', 'contains_248_68',
                          'f_5.4_55_of_roi', 'f_200_270_of_roi', 'f_above_270_of_total', 'R_roi_5.4_270')}
       for s in summ if s['halo'] == 'June167' and s['delta_keV'] in range(0, 425, 25)]
results['summary_table_June'] = tab
json.dump(results, open(os.path.join(OUT, 'results.json'), 'w'), indent=1, default=float)
print('--- wrote', os.path.join(OUT, 'results.json'))
