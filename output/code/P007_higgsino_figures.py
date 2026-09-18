"""
P007 figures: (a) expected LZ events N(delta) for the fixed pure-Higgsino Z-exchange coupling, five masses;
(b) delta required for N = 1 (band 0.3-3) versus mass, with the kinematic limit delta_max(248 keV, 16 June).
Reads output/work/P007/N_events_grid.csv and delta_for_N_events.csv; writes output/work/P007/figures/*.png
Run from the simulation root:  .venv/bin/python output/code/P007_higgsino_figures.py
"""
import sys, json, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, 'output/code')
from common import lzcommon as lz

OUT = 'output/work/P007'
df = pd.read_csv(f'{OUT}/N_events_grid.csv')
dres = pd.read_csv(f'{OUT}/delta_for_N_events.csv')
comp = json.load(open(f'{OUT}/comparison_with_LZ_intervals.json'))
MASSES = [300, 500, 1000, 2000, 4000]
COL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']     # fixed categorical order (dataviz reference palette)
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e6e6e3'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False})

fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw=dict(width_ratios=[1.35, 1]))
sel = (df.halo == 'annual') & (df.eff_sigma_keV == 11.5)
for m, c in zip(MASSES, COL):
    s = df[sel & (df.m_GeV == m) & (df.delta_keV >= 250) & (df.N_events > 0)].sort_values('delta_keV')
    ax.plot(s.delta_keV, s.N_events, color=c, lw=2, label=f'{m} GeV')
    r = dres[(dres.m_GeV == m) & (dres.halo == 'annual') & (dres.eff_sigma_keV == 11.5)].iloc[0]
    ax.plot([r.delta_max_248keV_june] * 2, [1e-3, 2e-3], color=c, lw=2)
ax.legend(fontsize=8.5, frameon=False, loc='lower left', title='Higgsino mass', title_fontsize=8.5)
ax.axhspan(0.105, 3.65, color='#dddcd6', alpha=0.45, lw=0, zorder=0)
for y, lab in [(1.0, 'N = 1'), (0.3, '0.3'), (3.0, '3')]:
    ax.axhline(y, color=INK2, lw=0.8, ls='--', zorder=0)
    ax.text(401, y, lab, color=INK2, va='center', fontsize=8.5)
ax.text(252, 5.0, '90% one-event band (0.105-3.65 events)', color=INK2, fontsize=8)
ax.set_yscale('log'); ax.set_xlim(250, 400); ax.set_ylim(1e-3, 3e4)
ax.set_xlabel('mass splitting  $\\delta$  [keV]'); ax.set_ylabel('expected LZ events in 2.84 t yr')
ax.set_title('Pure Higgsino, Z exchange (fixed coupling); annual-mean SHM', fontsize=10, color=INK, loc='left')
ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.text(392, 1.4e-3, '$\\delta_{\\max}$(248 keV, 16 June)', color=INK2, fontsize=8, ha='right', va='bottom')

# panel b: required delta vs mass
for halo, ls, lab in [('annual', '-', 'annual mean'), ('june16', ':', '16 June only'), ('sun', '--', 'Sun frame')]:
    r = dres[(dres.halo == halo) & (dres.eff_sigma_keV == 11.5)].sort_values('m_GeV')
    bx.plot(r.m_GeV, r.delta_N1, color=INK if halo == 'annual' else INK2, lw=2 if halo == 'annual' else 1.2, ls=ls, label=f'$\\delta$(N=1), {lab}')
r = dres[(dres.halo == 'annual') & (dres.eff_sigma_keV == 11.5)].sort_values('m_GeV')
bx.fill_between(r.m_GeV, r.delta_N3, r['delta_N0.3'], color='#2a78d6', alpha=0.25, lw=0, label='N = 3 ... 0.3 (annual)')
bx.fill_between(r.m_GeV, r['delta_N3.65'], r['delta_N0.105'], color='#2a78d6', alpha=0.12, lw=0, label='90% one-event band')
bx.plot(r.m_GeV, r.delta_max_248keV_june, color='#e34948', lw=1.5, label='$\\delta_{\\max}$(248 keV, 16 June)')
bx.plot(r.m_GeV, r.delta_max_270keV_june, color='#e34948', lw=1.0, ls='--', label='$\\delta_{\\max}$(269.9 keV, 16 June)')
bx.set_xscale('log'); bx.set_xlim(250, 5000); bx.set_ylim(320, 420)
bx.set_xticks([300, 500, 1000, 2000, 4000]); bx.set_xticklabels(['300', '500', '1000', '2000', '4000']); bx.minorticks_off()
bx.set_xlabel('Higgsino mass  $\\mu$  [GeV]'); bx.set_ylabel('required $\\delta$  [keV]')
bx.set_title('Splitting that yields one LZ event', fontsize=10, color=INK, loc='left')
bx.grid(color=GRID, lw=0.6); bx.set_axisbelow(True)
bx.legend(fontsize=7.5, frameon=False, loc='lower right')
fig.tight_layout()
fig.savefig(f'{OUT}/figures/P007_fig1_N_vs_delta_and_required_delta.png', dpi=170)

# figure 2: spectra at 1000 GeV for several delta, with the efficiency curve
sp = json.load(open(f'{OUT}/spectra_1000GeV.json'))
fig2, cx = plt.subplots(figsize=(6.2, 4.2))
from scipy.special import erf, erfc
E = np.linspace(0, 330, 661)
eff = 0.96 * 0.5 * (1 + erf((E - 5.4) / (np.sqrt(2) * 2.5))) * 0.5 * erfc((E - 269.9) / (np.sqrt(2) * 11.5))
for (d, c) in zip(['300', '350', '370', '380', '390'], COL):
    if d in sp:
        cx.plot(sp[d]['E'], np.array(sp[d]['annual']) * 2.84, color=c, lw=2)
        i = int(np.argmax(sp[d]['annual']))
        cx.text(sp[d]['E'][i], sp[d]['annual'][i] * 2.84 * 1.3, f'$\\delta$ = {d} keV', color=c, fontsize=8.5, ha='center')
cx.set_yscale('log'); cx.set_xlim(0, 330); cx.set_ylim(1e-5, 1e2)
cx.set_xlabel('nuclear recoil energy [keV]'); cx.set_ylabel('events / keV in 2.84 t yr (before efficiency)')
cx.axvspan(269.9, 330, color='#dddcd6', alpha=0.5, lw=0); cx.axvspan(0, 5.4, color='#dddcd6', alpha=0.5, lw=0)
cx.text(300, 2e-5, 'efficiency\n< 50%', color=INK2, fontsize=8, ha='center')
cx.axvline(248, color='#e34948', lw=1); cx.text(246, 3e1, 'event 248 keV', color='#e34948', fontsize=8, ha='right')
cx.grid(axis='y', color=GRID, lw=0.6); cx.set_axisbelow(True)
cx.set_title('1000 GeV pure Higgsino: recoil spectra (annual-mean SHM)', fontsize=10, loc='left')
fig2.tight_layout(); fig2.savefig(f'{OUT}/figures/P007_fig2_spectra_1000GeV.png', dpi=170)
print('figures written')
