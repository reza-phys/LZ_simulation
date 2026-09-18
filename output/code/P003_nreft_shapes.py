"""
P003_nreft_shapes.py -- Which NREFT operators can produce a lone high-energy xenon recoil?

For O_1, O_3..O_15 (isoscalar / isovector, WIMP spin 1/2) at m_chi = 200, 1000, 4000 GeV compute with WimPyDD
  * the recoil spectrum 1-300 keV (average halo, Baxter-2021 SHM parameters),
  * R_lo = efficiency-weighted rate in the 2024 low-energy ROI (5.4-55 keV),
  * R_hi = efficiency-weighted rate in the high-energy window (200-270 keV),
  * R_hi/R_lo and N_lo = R_lo/R_hi = number of low-energy events accompanying one high-energy event,
and classify the operators against the (recalled, uncertain) 2024 null result: N_lo <= N_max with N_max = 3, 5, 10.
Also builds the magnetic-dipole-like combination (q^2/m_N^2) O_4 - O_6 (pure transverse-spin Sigma' response) and
compares it with the L10^s curves digitised from Fig. 1 (P003_fig1_digitise.py must have been run first).

Run from the simulation root:  .venv/bin/python output/code/P003_nreft_shapes.py
"""
from __future__ import annotations
import sys, os, json, time
sys.path.insert(0, 'output/code')
import numpy as np
import pandas as pd
from scipy import special, integrate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import lzcommon as lz

OUT = 'output/work/P003'
FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)

MV2 = lz.M_V_GEV ** 2            # m_v^2, GeV^2 (unit coupling c = 1/m_v^2)
MN = lz.M_NUCLEON_GEV
OPS = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
MASSES = [200.0, 1000.0, 4000.0]
# keV grid: 3 keV steps (WimPyDD costs ~40 ms per point per spectrum; 84 spectra) plus the reference energies;
# window integrals interpolate log(dR/dE) on a fine grid, adequate for these smooth spectra (checked vs 1 keV steps for O1, O6)
E = np.union1d(np.arange(1.0, 300.0 + 0.5, 3.0), [5.4, 10.0, 50.0, 55.0, 150.0, 200.0, 248.0, 269.9, 300.0])
E_LO, E_LO_MAX = 5.4, 55.0                    # 2024 ROI (S1c 3-80 phd -> 5.4-55 keV at 50 % efficiency)
E_HI, E_HI_MAX = 200.0, 270.0                 # high-energy window
SIG_LO, SIG_HI = 3.4, 8.0                     # keV, erf roll-off widths (assumption, see details.md)
EFF0 = 0.96                                   # plateau efficiency, LZ paper (14-250 keV)

# leading nuclear response / momentum structure of each operator (Fitzpatrick et al. 2013; Anand et al. 2014) -- recalled
OP_INFO = {
    1: ('M', 'q^0', 'SI-like'),
    3: ("Phi''+ v^2 Sigma'", "q^4 Phi'', q^2 v^2 Sigma'", 'spin-orbit'),
    4: ("Sigma'+Sigma''", 'q^0', 'SD-like'),
    5: ('Delta + v^2 M', 'q^2 v^2 M, q^4 Delta', 'anapole-like'),
    6: ("Sigma''", 'q^4', 'pseudoscalar-pseudoscalar'),
    7: ("Sigma'", 'v^2', 'velocity-suppressed SD'),
    8: ('M + Delta', 'v^2 M, q^2 Delta', 'axial-vector-vector'),
    9: ("Sigma'", 'q^2', 'q-suppressed SD'),
    10: ("Sigma''", 'q^2', 'scalar-pseudoscalar'),
    11: ('M', 'q^2', 'pseudoscalar-scalar (q^2 SI)'),
    12: ("Phi''+ v^2 Sigma'", "q^2 Phi'', v^2 Sigma'", 'spin-orbit'),
    13: ("Sigma''", 'q^2 v^2', ''),
    14: ("Sigma'", 'q^2 v^2', ''),
    15: ("Phi''+ v^2 Sigma'", "q^6 Phi'', q^4 v^2 Sigma'", ''),
}


def efficiency(E, sig_lo=SIG_LO, sig_hi=SIG_HI, eff0=EFF0, hard=False):
    """Signal efficiency model: plateau eff0 with erf roll-offs centred at 5.4 and 269.9 keV.
    hard=True: step function (eff0 between 5.4 and 269.9 keV)."""
    E = np.asarray(E, dtype=float)
    if hard:
        return eff0 * ((E >= lz.LZ['E_50pct_low_keV']) & (E <= lz.LZ['E_50pct_high_keV']))
    s_lo = 0.5 * (1 + special.erf((E - lz.LZ['E_50pct_low_keV']) / (np.sqrt(2) * sig_lo)))
    s_hi = 0.5 * (1 - special.erf((E - lz.LZ['E_50pct_high_keV']) / (np.sqrt(2) * sig_hi)))
    return eff0 * s_lo * s_hi


def window_rate(Egrid, dRdE, e1, e2, **effkw):
    """Efficiency-weighted integral of dR/dE over [e1, e2] (trapezoid on a fine grid)."""
    Ef = np.linspace(e1, e2, 2001)
    dRdE = np.asarray(dRdE, dtype=float)
    if np.all(dRdE > 0):
        y = 10 ** np.interp(Ef, Egrid, np.log10(dRdE))          # log-linear interpolation for steep spectra
    else:
        y = np.interp(Ef, Egrid, dRdE)
    y = y * efficiency(Ef, **effkw)
    return float(integrate.trapezoid(y, Ef))


def summarise(Egrid, dRdE, **effkw):
    R_lo = window_rate(Egrid, dRdE, E_LO, E_LO_MAX, **effkw)
    R_hi = window_rate(Egrid, dRdE, E_HI, E_HI_MAX, **effkw)
    R_all = window_rate(Egrid, dRdE, E_LO, 300.0, **effkw)
    return dict(R_lo=R_lo, R_hi=R_hi, R_5_300=R_all, ratio_hi_lo=R_hi / R_lo if R_lo > 0 else np.inf,
                N_lo_per_hi=R_lo / R_hi if R_hi > 0 else np.inf,
                frac_hi=R_hi / R_all if R_all > 0 else np.nan)


t0 = time.time()
halo = lz.wd_halo()
print(f'halo: {len(halo[0])} vmin bins, vmax = {halo[0][-1]:.1f} km/s')

# ----------------------------------------------------------------------------
# 1. operator scan
# ----------------------------------------------------------------------------
rows, spectra = [], {}
for op in OPS:
    for iso, (c0, c1) in (('s', (1 / MV2, 0.0)), ('v', (0.0, 1 / MV2))):
        ham = lz.wd_hamiltonian(f'O{op}{iso}', {op: (c0, c1)})
        for m in MASSES:
            r = lz.wd_rate(ham, m, E, halo)
            spectra[(op, iso, m)] = r
            s = summarise(E, r)
            s_hard = summarise(E, r, hard=True)
            s_wide = summarise(E, r, sig_hi=15.0)
            s_narrow = summarise(E, r, sig_hi=4.0)
            rows.append(dict(operator=f'O{op}', isospin=iso, m_chi_GeV=m, response=OP_INFO[op][0], q_v_scaling=OP_INFO[op][1],
                             dRdE_10keV=r[np.searchsorted(E, 10.0)], dRdE_50keV=r[np.searchsorted(E, 50.0)],
                             dRdE_150keV=r[np.searchsorted(E, 150.0)], dRdE_248keV=r[np.searchsorted(E, 248.0)],
                             **s, N_lo_hardcut=s_hard['N_lo_per_hi'], N_lo_sighi15=s_wide['N_lo_per_hi'],
                             N_lo_sighi4=s_narrow['N_lo_per_hi']))
            print(f"O{op:<2d}{iso} m={m:6.0f}  R_lo={s['R_lo']:.3e}  R_hi={s['R_hi']:.3e}  R_hi/R_lo={s['ratio_hi_lo']:.3e}  "
                  f"N_lo/hi={s['N_lo_per_hi']:9.2f}   ({time.time()-t0:.0f}s)")
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, 'P003_operator_table.csv'), index=False, float_format='%.5g')
np.savez(os.path.join(OUT, 'P003_spectra.npz'), E=E, **{f'O{k[0]}{k[1]}_{int(k[2])}': v for k, v in spectra.items()})

# inelastic O1 reference rows (delta = 200, 300 keV at 1000 GeV) for context
inel = []
h1 = lz.wd_hamiltonian('O1s_in', {1: (1 / MV2, 0.0)})
for dl in (0.0, 100.0, 200.0, 300.0):
    r = lz.wd_rate(h1, 1000.0, E, halo, delta_kev=dl)
    s = summarise(E, r)
    inel.append(dict(model=f'O1s delta={dl:.0f} keV, 1000 GeV', **s))
    spectra[('O1s_inel', dl)] = r
pd.DataFrame(inel).to_csv(os.path.join(OUT, 'P003_inelastic_reference.csv'), index=False, float_format='%.5g')

# ----------------------------------------------------------------------------
# 2. classification table (per operator: worst / best case over masses and isospin)
# ----------------------------------------------------------------------------
cls = []
for op in OPS:
    sub = df[df.operator == f'O{op}']
    for iso in ('s', 'v'):
        ss = sub[sub.isospin == iso].sort_values('m_chi_GeV')
        N = ss.N_lo_per_hi.values
        cls.append(dict(operator=f'O{op}', isospin=iso, N_lo_200=N[0], N_lo_1000=N[1], N_lo_4000=N[2],
                        compatible_Nmax3_at_1000=bool(N[1] <= 3), compatible_Nmax5_at_1000=bool(N[1] <= 5),
                        compatible_Nmax10_at_1000=bool(N[1] <= 10),
                        verdict=('lone-event compatible (N_lo <= 5)' if N[1] <= 5 else
                                 'marginal (5 < N_lo <= 10)' if N[1] <= 10 else
                                 'excluded by absent low-energy population (N_lo > 10)')))
cdf = pd.DataFrame(cls)
cdf.to_csv(os.path.join(OUT, 'P003_classification.csv'), index=False, float_format='%.4g')
print(cdf.to_string())

# ----------------------------------------------------------------------------
# 3. magnetic-dipole-like combination  (q^2/m_N^2) O4 - O6  (transverse spin, pure Sigma')  vs digitised L10^s
# ----------------------------------------------------------------------------
h_mag = lz.wd()  # ensure loaded
h_mag = lz.wd().eft_hamiltonian('mag_q2O4_minus_O6', {(4, 'q2'): lambda q, A=1.0: [A * q ** 2 / MN ** 2, 0.0],
                                                      6: lambda A=1.0: [-A, 0.0]})
h_plus = lz.wd().eft_hamiltonian('q2O4_plus_O6', {(4, 'q2'): lambda q, A=1.0: [A * q ** 2 / MN ** 2, 0.0],
                                                  6: lambda A=1.0: [A, 0.0]})
h_q2O4 = lz.wd().eft_hamiltonian('q2O4_only', {(4, 'q2'): lambda q, A=1.0: [A * q ** 2 / MN ** 2, 0.0]})
mag_rows, mag_spec = [], {}
dig = pd.read_csv(os.path.join(OUT, 'fig1_digitised_bottom.csv'))
for m, col in ((50.0, 'L10s_50GeV'), (200.0, 'L10s_200GeV'), (1000.0, 'L10s_1000GeV')):
    r_mag = lz.wd_rate(h_mag, m, E, halo, A=1 / MV2)
    r_plus = lz.wd_rate(h_plus, m, E, halo, A=1 / MV2)
    r_q2o4 = lz.wd_rate(h_q2O4, m, E, halo, A=1 / MV2)
    r_o6 = spectra[(6, 's', 1000.0)] if m == 1000.0 else lz.wd_rate(lz.wd_hamiltonian('O6s', {6: (1 / MV2, 0.0)}), m, E, halo)
    mag_spec[m] = dict(mag=r_mag, plus=r_plus, q2o4=r_q2o4, o6=r_o6)
    fig_curve = np.interp(E, dig.E_keV, dig[col], left=np.nan, right=np.nan)
    ok = np.isfinite(fig_curve) & (E >= 10) & (E <= 260) & (fig_curve > 0)
    ok &= (E <= 110) if m == 50 else True
    scale = np.exp(np.nanmean(np.log(fig_curve[ok] / r_mag[ok])))            # geometric-mean scale factor
    resid = np.log10(fig_curve[ok] / (scale * r_mag[ok]))
    # peak / dip positions
    def extrema(y, Egrid, mask):
        yy = np.where(mask, y, np.nan)
        i1 = np.nanargmax(np.where(Egrid < 120, yy, np.nan)) if np.any(mask & (Egrid < 120)) else -1
        i2 = np.nanargmax(np.where(Egrid > 120, yy, np.nan)) if np.any(mask & (Egrid > 120)) else -1
        seg = (Egrid > 20) & (Egrid < 200) & mask
        idip = np.nanargmin(np.where(seg, yy, np.nan)) if np.any(seg) else -1
        return (float(Egrid[i1]) if i1 >= 0 else np.nan, float(Egrid[idip]) if idip >= 0 else np.nan,
                float(Egrid[i2]) if i2 >= 0 else np.nan)
    p_fig = extrema(fig_curve, E, ok)
    p_mag = extrema(r_mag, E, ok)
    s_fig = summarise(E, np.nan_to_num(fig_curve))
    s_mag = summarise(E, r_mag)
    mag_rows.append(dict(m_chi_GeV=m, scale_fig_over_wimpydd_A_1_over_mv2=scale, rms_log10_residual=float(np.sqrt(np.mean(resid ** 2))),
                         max_abs_log10_residual=float(np.max(np.abs(resid))),
                         fig_peak1_keV=p_fig[0], fig_dip_keV=p_fig[1], fig_peak2_keV=p_fig[2],
                         wd_peak1_keV=p_mag[0], wd_dip_keV=p_mag[1], wd_peak2_keV=p_mag[2],
                         N_lo_per_hi_fig1=s_fig['N_lo_per_hi'], N_lo_per_hi_wimpydd=s_mag['N_lo_per_hi'],
                         R_hi_fig1_per_tyr=s_fig['R_hi'], R_lo_fig1_per_tyr=s_fig['R_lo'],
                         frac_hi_fig1=s_fig['frac_hi'], frac_hi_wimpydd=s_mag['frac_hi']))
    print(f'L10 comparison m={m:.0f}: scale={scale:.3g}, rms log10 resid={mag_rows[-1]["rms_log10_residual"]:.3f}, '
          f'peaks fig {p_fig} vs wd {p_mag}; N_lo fig={s_fig["N_lo_per_hi"]:.2f} wd={s_mag["N_lo_per_hi"]:.2f}')
mdf = pd.DataFrame(mag_rows)
mdf.to_csv(os.path.join(OUT, 'P003_L10_comparison.csv'), index=False, float_format='%.5g')
# candidate absolute-normalisation factors for the digitised/wd scale (recalled reductions, see details.md)
cands = {'(2)^2 WimPyDD c=cp+cn convention': 4.0, '4^2 (Anand L10 -> 4[(q^2/mN^2)O4 - O6], m_M = m_N) x 2^2': 64.0,
         '4^2 only': 16.0}
mdf_note = {k: float(mdf.scale_fig_over_wimpydd_A_1_over_mv2.iloc[-1] / v) for k, v in cands.items()}

# digitised O1s delta=0 curve as pipeline cross-check (N_lo from Fig. 1 itself)
digt = pd.read_csv(os.path.join(OUT, 'fig1_digitised_top.csv'))
o1fig = np.interp(E, digt.E_keV, digt.O1s_delta0_1000GeV, left=np.nan, right=np.nan)
o1fig = np.where(np.isfinite(o1fig), o1fig, 0.0)
s_o1fig = summarise(E, o1fig)
xcheck = dict(O1s_1000_fig1=s_o1fig, O1s_1000_wimpydd=summarise(E, spectra[(1, 's', 1000.0)]),
              O1s_1000_wimpydd_c0_2_over_mv2=summarise(E, 4 * spectra[(1, 's', 1000.0)]))

# ----------------------------------------------------------------------------
# 4. figures
# ----------------------------------------------------------------------------
# 4a spectra, one panel per operator
fig, axs = plt.subplots(4, 4, figsize=(15, 13), sharex=True)
axs = axs.ravel()
cols = {200.0: 'C0', 1000.0: 'C1', 4000.0: 'C2'}
for i, op in enumerate(OPS):
    ax = axs[i]
    for m in MASSES:
        for iso, ls in (('s', '-'), ('v', '--')):
            ax.plot(E, spectra[(op, iso, m)], color=cols[m], ls=ls, lw=1.2,
                    label=f'{iso} {m:.0f} GeV' if op == 1 else None)
    ax.set_yscale('log'); ax.set_xlim(0, 300)
    ymax = max(spectra[(op, iso, m)].max() for m in MASSES for iso in 'sv')
    ax.set_ylim(ymax * 1e-6, ymax * 3)
    ax.axvspan(0, 5.4, color='0.6', alpha=0.3); ax.axvspan(269.9, 300, color='0.6', alpha=0.3)
    ax.axvspan(E_LO, E_LO_MAX, color='C3', alpha=0.07); ax.axvspan(E_HI, E_HI_MAX, color='C2', alpha=0.10)
    ax.set_title(f'$\\mathcal{{O}}_{{{op}}}$  [{OP_INFO[op][0]}; {OP_INFO[op][1]}]', fontsize=9)
    ax.text(0.97, 0.95, 'N$_{lo}$/hi (1000 GeV): s %.3g, v %.3g' % (
        df[(df.operator == f'O{op}') & (df.isospin == 's') & (df.m_chi_GeV == 1000)].N_lo_per_hi.iloc[0],
        df[(df.operator == f'O{op}') & (df.isospin == 'v') & (df.m_chi_GeV == 1000)].N_lo_per_hi.iloc[0]),
        transform=ax.transAxes, ha='right', va='top', fontsize=7)
axs[0].legend(fontsize=7, ncol=2)
for ax in axs[12:]:
    ax.set_xlabel('$E_R$ [keV]')
for ax in axs[::4]:
    ax.set_ylabel('dR/dE [/t/yr/keV], c = 1/m$_v^2$')
axs[14].axis('off'); axs[15].axis('off')
axs[14].text(0.0, 0.5, 'solid: isoscalar (c$^0$ = 1/m$_v^2$)\ndashed: isovector (c$^1$ = 1/m$_v^2$)\n'
             'red band: 2024 ROI 5.4-55 keV\ngreen band: 200-270 keV window\nWimPyDD 2.0.4, Baxter-2021 SHM, average halo',
             fontsize=9, va='center')
fig.suptitle('P003: NREFT operator spectra in xenon (WIMP spin 1/2)', y=0.995)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P003_operator_spectra.png'), dpi=130); plt.close(fig)

# 4b bar chart of N_lo per high-energy event
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(OPS)); w = 0.13
for j, (iso, m) in enumerate([(iso, m) for m in MASSES for iso in 'sv']):
    vals = [df[(df.operator == f'O{op}') & (df.isospin == iso) & (df.m_chi_GeV == m)].N_lo_per_hi.iloc[0] for op in OPS]
    ax.bar(x + (j - 2.5) * w, vals, w, color=cols[m], alpha=1.0 if iso == 's' else 0.45, edgecolor='k', lw=0.3,
           label=f'{iso}, {m:.0f} GeV')
for N, lab in ((3, 'N$_{max}$=3'), (5, 'N$_{max}$=5'), (10, 'N$_{max}$=10')):
    ax.axhline(N, color='k', ls=':' if N != 5 else '-', lw=0.8); ax.text(len(OPS) - 0.5, N * 1.08, lab, fontsize=8, ha='right')
ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels([f'O{op}' for op in OPS])
ax.set_ylabel('low-energy (5.4-55 keV) events per event in 200-270 keV')
ax.set_title('P003: N$_{lo}$ per high-energy event, natural Xe, LZ efficiency model')
ax.legend(fontsize=8, ncol=3); ax.grid(axis='y', alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P003_Nlo_bar.png'), dpi=140); plt.close(fig)

# 4c L10 overlay
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
ax = axs[0]
for m, col, c in ((50.0, 'L10s_50GeV', 'C3'), (200.0, 'L10s_200GeV', 'C4'), (1000.0, 'L10s_1000GeV', 'C5')):
    ax.plot(dig.E_keV, dig[col], color=c, lw=3, alpha=0.35, label=f'Fig. 1 L$_{{10}}^s$ {m:.0f} GeV (digitised)')
    sc = mdf[mdf.m_chi_GeV == m].scale_fig_over_wimpydd_A_1_over_mv2.iloc[0]
    ax.plot(E, sc * mag_spec[m]['mag'], color=c, lw=1.2, label=f'WimPyDD (q$^2$/m$_N^2$)O$_4$ - O$_6$ x {sc:.0f}')
ax.set_yscale('log'); ax.set_ylim(1e-3, 0.6); ax.set_xlim(0, 350); ax.set_xlabel('$E_R$ [keV]'); ax.set_ylabel('dR/dE [/t/yr/keV]')
ax.axvspan(0, 5.4, color='0.6', alpha=0.3); ax.axvspan(269.9, 350, color='0.6', alpha=0.3); ax.legend(fontsize=7)
ax.set_title('Magnetic-dipole-like combination vs Fig. 1 bottom')
ax = axs[1]
m = 1000.0
for key, lab, ls in (('mag', '(q$^2$/m$_N^2$)O$_4$ - O$_6$  (pure $\\Sigma\'$)', '-'), ('plus', '(q$^2$/m$_N^2$)O$_4$ + O$_6$', '--'),
                     ('q2o4', '(q$^2$/m$_N^2$)O$_4$ alone', ':'), ('o6', 'O$_6$ alone', '-.')):
    ax.plot(E, mag_spec[m][key] / mag_spec[m][key].max(), ls=ls, label=lab)
fc = np.interp(E, dig.E_keV, dig['L10s_1000GeV'], left=np.nan, right=np.nan)
ax.plot(E, fc / np.nanmax(fc), color='k', lw=3, alpha=0.3, label='Fig. 1 L$_{10}^s$ 1000 GeV')
ax.set_yscale('log'); ax.set_ylim(1e-2, 2); ax.set_xlim(0, 300); ax.set_xlabel('$E_R$ [keV]'); ax.set_ylabel('shape (peak-normalised)')
ax.legend(fontsize=7); ax.set_title('Why L$_{10}$ is double-peaked: 1000 GeV, shapes')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P003_L10_overlay.png'), dpi=140); plt.close(fig)

# 4d efficiency model
fig, ax = plt.subplots(figsize=(6, 3.2))
Ee = np.linspace(0, 300, 601)
ax.plot(Ee, efficiency(Ee), label=f'erf model (σ_lo={SIG_LO}, σ_hi={SIG_HI} keV)')
ax.plot(Ee, efficiency(Ee, sig_hi=15.0), '--', label='σ_hi = 15 keV'); ax.plot(Ee, efficiency(Ee, sig_hi=4.0), ':', label='σ_hi = 4 keV')
ax.plot(Ee, efficiency(Ee, hard=True), lw=0.8, color='k', label='hard cuts')
ax.set_xlabel('$E_R$ [keV]'); ax.set_ylabel('efficiency'); ax.legend(fontsize=7); ax.set_ylim(0, 1.05)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P003_efficiency_model.png'), dpi=140); plt.close(fig)

# ----------------------------------------------------------------------------
# 5. summary JSON
# ----------------------------------------------------------------------------
summary = dict(
    settings=dict(E_lo=[E_LO, E_LO_MAX], E_hi=[E_HI, E_HI_MAX], sig_lo=SIG_LO, sig_hi=SIG_HI, eff0=EFF0, masses=MASSES,
                  coupling='c^tau = 1/m_v^2 in WimPyDD convention (c0 = c_p + c_n); rates scale as c^2',
                  halo='WimPyDD streamed_halo_function, v0=238, vesc=544, v_sun_pec=(11.1,12.2,7.3), no Earth orbital velocity',
                  vmax_kms=float(halo[0][-1])),
    classification={f"O{r.operator[1:]}{r.isospin}": dict(N_lo_200=r.N_lo_200, N_lo_1000=r.N_lo_1000, N_lo_4000=r.N_lo_4000,
                                                         verdict=r.verdict) for r in cdf.itertuples()},
    inelastic_reference=inel,
    L10_comparison=mag_rows, L10_scale_hypotheses=mdf_note,
    pipeline_crosscheck_O1s_fig1_vs_wimpydd=xcheck,
    runtime_s=time.time() - t0)
with open(os.path.join(OUT, 'P003_summary.json'), 'w') as f:
    json.dump(summary, f, indent=1, default=float)
print(json.dumps(summary['pipeline_crosscheck_O1s_fig1_vs_wimpydd'], indent=1, default=float))
print(json.dumps(mdf_note, indent=1))
print(f'done in {time.time()-t0:.0f} s')
