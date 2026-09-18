"""
P003_fig1_digitise.py -- digitise the vector curves of LZ Fig. 1 (arXiv:2609.02823,
inputs/arXiv_2609.02823_source/Fig1_combined_recoils.pdf) with PyMuPDF and compare the
O1^s (delta = 0, 1000 GeV) curve with WimPyDD to fix the coupling normalisation.

Run from the simulation root:   .venv/bin/python output/code/P003_fig1_digitise.py

Outputs (output/work/P003/):
  fig1_digitised_top.csv      E [keV], rate [/t/yr/keV] for O1s delta = 0, 200, 300 keV (1000 GeV)
  fig1_digitised_bottom.csv   E [keV], rate [/t/yr/keV] for L10s at 50, 200, 1000 GeV
  fig1_normalisation.json     ratios WimPyDD / digitised for c0 = 1/m_v^2 and 2/m_v^2
  figures/P003_fig1_overlay.png
"""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, 'output/code')
import numpy as np
import pymupdf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import lzcommon as lz

PDF = 'inputs/arXiv_2609.02823_source/Fig1_combined_recoils.pdf'
OUT = 'output/work/P003'
FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)

# ----------------------------------------------------------------------------
# 1. axis calibration from the tick labels
# ----------------------------------------------------------------------------
doc = pymupdf.open(PDF)
page = doc[0]
spans = []
for b in page.get_text('dict')['blocks']:
    for l in b.get('lines', []):
        for s in l['spans']:
            x0, y0, x1, y1 = s['bbox']
            spans.append(dict(text=s['text'].strip(), cx=0.5 * (x0 + x1), cy=0.5 * (y0 + y1),
                              x0=x0, y0=y0, x1=x1, y1=y1, size=s['size']))

# x axis: labels 0,50,...,350 at the bottom of the page (cy > 640), font size 19
xl = [s for s in spans if s['size'] > 18 and s['cy'] > 640 and s['text'].isdigit() and s['cx'] > 60]
xvals = np.array([float(s['text']) for s in xl]); xpos = np.array([s['cx'] for s in xl])
bx, ax = np.polyfit(xpos, xvals, 1)          # E = ax + bx * xpix
def E_of_x(x): return ax + bx * x

# y axes: mantissa '10' spans (size 19) with an exponent span (size 13.3) just to their right
mant = [s for s in spans if s['size'] > 18 and s['text'] == '10']
expo = [s for s in spans if 12 < s['size'] < 14]
ticks_top, ticks_bot = [], []
for m in mant:
    cand = [e for e in expo if abs(e['x0'] - m['x1']) < 3 and abs(e['cy'] - m['cy']) < 12]
    if not cand:
        continue
    ex = float(cand[0]['text'].replace('−', '-'))
    (ticks_top if m['cy'] < 340 else ticks_bot).append((m['cy'], ex))   # the 10^-1 label of the top panel sits at y = 332.7
ticks_top.sort(); ticks_bot.sort()
# panel boxes from the grey shaded regions (< 5.4 keV, > 269.9 keV); they span the full y range of each axes
draw = page.get_drawings()
grey = [d for d in draw if d['type'] == 'f' and d.get('fill') and abs(d['fill'][0] - 0.502) < 0.01]
top_box = (min(d['rect'].y0 for d in grey if d['rect'].y0 < 300), max(d['rect'].y1 for d in grey if d['rect'].y0 < 300))
bot_box = (min(d['rect'].y0 for d in grey if d['rect'].y0 > 300), max(d['rect'].y1 for d in grey if d['rect'].y0 > 300))
# A first fit to the label-bbox centres shows that the centres sit ~1.65 pt below the ticks (font ascender/descender
# asymmetry), i.e. a 0.05-decade bias on the top panel.  The decade spacing from the labels is offset-free, so we take
# the spacing from the labels and anchor the scale on the panel edges: the lowest label (10^-1 top, 10^-3 bottom)
# sits exactly at the bottom edge of its panel (matplotlib ylim), which the label positions confirm to 0.1 pt.
by_t = np.polyfit([t[0] for t in ticks_top], [t[1] for t in ticks_top], 1)[0]
by_b = np.polyfit([t[0] for t in ticks_bot], [t[1] for t in ticks_bot], 1)[0]
exp_bottom_top, exp_bottom_bot = ticks_top[-1][1], ticks_bot[-1][1]     # lowest tick exponent (largest y)
def logR_top(y): return exp_bottom_top + by_t * (y - top_box[1])
def logR_bot(y): return exp_bottom_bot + by_b * (y - bot_box[1])
label_offset_top = np.mean([t[0] - (top_box[1] + (t[1] - exp_bottom_top) / by_t) for t in ticks_top])
label_offset_bot = np.mean([t[0] - (bot_box[1] + (t[1] - exp_bottom_bot) / by_b) for t in ticks_bot])
calib = dict(x_pix_per_keV=1 / bx, x_pix_at_0keV=-ax / bx,
             top_pix_per_decade=1 / by_t, bottom_pix_per_decade=1 / by_b,
             top_box_y=top_box, bottom_box_y=bot_box,
             top_logR_range=(logR_top(top_box[1]), logR_top(top_box[0])),
             bottom_logR_range=(logR_bot(bot_box[1]), logR_bot(bot_box[0])),
             label_centre_minus_tick_pt=(label_offset_top, label_offset_bot),
             grey_band_edges_keV=[E_of_x(d['rect'].x0) for d in grey] + [E_of_x(d['rect'].x1) for d in grey])
print('axis calibration:', json.dumps({k: (list(map(float, v)) if hasattr(v, '__len__') else float(v)) for k, v in calib.items()}, indent=1))

# ----------------------------------------------------------------------------
# 2. curve extraction (matplotlib tab10 colours, stroked width-2 paths)
# ----------------------------------------------------------------------------
COLORS = {  # (r,g,b) of tab10, label, panel
    (0.122, 0.467, 0.706): ('O1s_delta0_1000GeV', 'top'),
    (1.000, 0.498, 0.055): ('O1s_delta200_1000GeV', 'top'),
    (0.173, 0.627, 0.173): ('O1s_delta300_1000GeV', 'top'),
    (0.839, 0.153, 0.157): ('L10s_50GeV', 'bottom'),
    (0.580, 0.404, 0.741): ('L10s_200GeV', 'bottom'),
    (0.549, 0.337, 0.294): ('L10s_1000GeV', 'bottom'),
}
curves = {}
for d in draw:
    if d['type'] != 's' or d.get('color') is None or d.get('width', 0) < 1.5:
        continue
    col = tuple(round(c, 3) for c in d['color'])
    match = [v for k, v in COLORS.items() if all(abs(k[i] - col[i]) < 0.01 for i in range(3))]
    if not match:
        continue
    label, panel = match[0]
    pts = []
    for it in d['items']:
        if it[0] == 'l':
            pts.append((it[1].x, it[1].y)); pts.append((it[2].x, it[2].y))
        elif it[0] == 'c':
            pts.append((it[1].x, it[1].y)); pts.append((it[4].x, it[4].y))
    pts = np.array(pts)
    if len(pts) < 20:          # legend handle, not a curve
        continue
    box = top_box if panel == 'top' else bot_box
    inside = (pts[:, 1] >= box[0] - 0.5) & (pts[:, 1] <= box[1] + 0.5)
    pts = pts[inside]
    E = E_of_x(pts[:, 0])
    logR = logR_top(pts[:, 1]) if panel == 'top' else logR_bot(pts[:, 1])
    order = np.argsort(E)
    E, logR = E[order], logR[order]
    # collapse duplicates
    Eu, idx = np.unique(np.round(E, 3), return_index=True)
    curves[label] = (Eu, 10 ** logR[idx])
    print(f'{label:24s} {len(Eu):5d} points, E {Eu.min():.1f}-{Eu.max():.1f} keV, '
          f'rate {10**logR.min():.3g}-{10**logR.max():.3g}')

for panel, fn in (('top', 'fig1_digitised_top.csv'), ('bottom', 'fig1_digitised_bottom.csv')):
    labels = [l for l, (c, p) in COLORS.items() if False]  # placeholder
    labels = [v[0] for v in COLORS.values() if v[1] == panel]
    Egrid = np.arange(1.0, 350.0, 1.0)
    cols = {'E_keV': Egrid}
    for lab in labels:
        E, R = curves[lab]
        cols[lab] = np.interp(Egrid, E, np.log10(R), left=np.nan, right=np.nan)
        cols[lab] = 10 ** cols[lab]
    import pandas as pd
    pd.DataFrame(cols).to_csv(os.path.join(OUT, fn), index=False, float_format='%.5g')

# ----------------------------------------------------------------------------
# 3. normalisation test: WimPyDD O1 (1000 GeV) with c0 = 1/m_v^2 and 2/m_v^2
# ----------------------------------------------------------------------------
halo = lz.wd_halo()
mv2 = lz.M_V_GEV ** 2
Etest = np.array([10.0, 20.0, 50.0, 80.0, 150.0, 200.0, 250.0])
E_dig, R_dig = curves['O1s_delta0_1000GeV']
R_dig_test = 10 ** np.interp(Etest, E_dig, np.log10(R_dig))
res = {'E_keV': Etest.tolist(), 'digitised': R_dig_test.tolist()}
for tag, c0 in (('c0=1/mv2', 1 / mv2), ('c0=2/mv2', 2 / mv2)):
    h = lz.wd_hamiltonian('O1_' + tag, {1: (c0, 0.0)})
    r = lz.wd_rate(h, 1000.0, Etest, halo)
    res[tag] = dict(rate=r.tolist(), ratio_wimpydd_over_fig=(r / R_dig_test).tolist())
# position of the form-factor (M-response) diffraction minimum: Fig. 1 vs WimPyDD
i_min = np.argmin(np.log10(R_dig[(E_dig > 60) & (E_dig < 140)]))
E_min_fig = E_dig[(E_dig > 60) & (E_dig < 140)][i_min]
Efine = np.arange(80.0, 120.01, 0.5)
rw = lz.wd_rate(lz.wd_hamiltonian('O1_min', {1: (2 / mv2, 0.0)}), 1000.0, Efine, halo)
res['M_response_minimum_keV'] = dict(fig1=float(E_min_fig), wimpydd=float(Efine[np.argmin(rw)]),
                                     depth_fig1=float(10 ** np.interp(E_min_fig, E_dig, np.log10(R_dig)) /
                                                      10 ** np.interp(E_min_fig - 10, E_dig, np.log10(R_dig))),
                                     depth_wimpydd=float(rw.min() / lz.wd_rate(lz.wd_hamiltonian('O1_min', {1: (2 / mv2, 0.0)}),
                                                                             1000.0, Efine[np.argmin(rw)] - 10, halo)))
# full ratio curve (2/m_v^2), saved for details.md
Er = np.arange(5.0, 300.0, 5.0)
rr = lz.wd_rate(lz.wd_hamiltonian('O1_r', {1: (2 / mv2, 0.0)}), 1000.0, Er, halo) / 10 ** np.interp(Er, E_dig, np.log10(R_dig))
import pandas as pd
pd.DataFrame(dict(E_keV=Er, ratio_wimpydd_2mv2_over_fig1=rr)).to_csv(os.path.join(OUT, 'fig1_O1_ratio_curve.csv'), index=False, float_format='%.4f')
res['ratio_curve_2mv2_median_10_90keV'] = float(np.median(rr[(Er >= 10) & (Er <= 90)]))
res['ratio_curve_2mv2_median_110_260keV'] = float(np.median(rr[(Er >= 110) & (Er <= 260)]))
# also the Helm SI formula for reference:  sigma_p for c_p = c_n = 1/m_v^2 (Anand normalisation)
mu_p = lz.mu_red(1000.0, lz.M_NUCLEON_GEV)
sigma_p_anand = mu_p ** 2 / np.pi / mv2 ** 2 * lz.GEV_TO_CM2     # c_p = 1/m_v^2 in Anand's c^0 = (c_p+c_n)/2 convention
res['sigma_p_cm2_for_cp_eq_1_over_mv2'] = float(sigma_p_anand)
res['helm_dRdE_sigma_p_anand'] = [float(lz.dRdE_SI(e, 1000.0, sigma_p_anand)) for e in Etest]
# inelastic curves as a shape check of the digitisation (delta = 200, 300 keV)
for dl in (200.0, 300.0):
    Ed, Rd = curves[f'O1s_delta{int(dl)}_1000GeV']
    Ecmp = np.array([200.0, 230.0, 260.0]) if dl == 200 else np.array([230.0, 260.0])
    h = lz.wd_hamiltonian('O1_in', {1: (1 / mv2, 0.0)})
    rw = lz.wd_rate(h, 1000.0, Ecmp, halo, delta_kev=dl)
    rd = 10 ** np.interp(Ecmp, Ed, np.log10(Rd))
    res[f'delta{int(dl)}_check'] = dict(E_keV=Ecmp.tolist(), wimpydd_c0_1_over_mv2=rw.tolist(), digitised=rd.tolist(),
                                        ratio=(rw / rd).tolist())
print(json.dumps(res, indent=1))
with open(os.path.join(OUT, 'fig1_normalisation.json'), 'w') as f:
    json.dump(dict(calibration={k: (list(map(float, v)) if hasattr(v, '__len__') else float(v)) for k, v in calib.items()},
                   normalisation=res), f, indent=1)

# ----------------------------------------------------------------------------
# 4. overlay figure
# ----------------------------------------------------------------------------
Eg = np.linspace(1, 349, 175)
fig, axs = plt.subplots(2, 1, figsize=(6.5, 8.5))
ax0, ax1 = axs
for lab, c in (('O1s_delta0_1000GeV', 'C0'), ('O1s_delta200_1000GeV', 'C1'), ('O1s_delta300_1000GeV', 'C2')):
    E, R = curves[lab]
    ax0.plot(E, R, color=c, lw=3, alpha=0.35, label=lab + ' (Fig. 1 digitised)')
h1 = lz.wd_hamiltonian('O1_a', {1: (1 / mv2, 0.0)})
h2 = lz.wd_hamiltonian('O1_b', {1: (2 / mv2, 0.0)})
for dl, c in ((0.0, 'C0'), (200.0, 'C1'), (300.0, 'C2')):
    r1 = lz.wd_rate(h1, 1000.0, Eg, halo, delta_kev=dl)
    ax0.plot(Eg, r1, color=c, lw=1.2, ls='-', label=f'WimPyDD c0=1/m_v², δ={dl:.0f}')
    if dl == 0:
        r2 = lz.wd_rate(h2, 1000.0, Eg, halo, delta_kev=dl)
        ax0.plot(Eg, r2, color='k', lw=1.0, ls='--', label='WimPyDD c0=2/m_v², δ=0')
ax0.set_yscale('log'); ax0.set_ylim(1e-1, 1e9); ax0.set_xlim(0, 350)
ax0.set_ylabel('dR/dE [/t/yr/keV]'); ax0.legend(fontsize=7, ncol=2); ax0.set_title('Fig. 1 top: O1s, 1000 GeV')
for lab, c in (('L10s_50GeV', 'C3'), ('L10s_200GeV', 'C4'), ('L10s_1000GeV', 'C5')):
    E, R = curves[lab]
    ax1.plot(E, R, color=c, lw=3, alpha=0.35, label=lab + ' (Fig. 1 digitised)')
ax1.set_yscale('log'); ax1.set_ylim(1e-3, 0.6); ax1.set_xlim(0, 350)
ax1.set_xlabel('E_R [keV]'); ax1.set_ylabel('dR/dE [/t/yr/keV]'); ax1.legend(fontsize=7)
ax1.set_title('Fig. 1 bottom: L10s (digitised; WimPyDD comparison in P003_nreft_shapes.py)')
for a in axs:
    a.axvspan(0, 5.4, color='0.5', alpha=0.3); a.axvspan(269.9, 350, color='0.5', alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'P003_fig1_overlay.png'), dpi=150)
print('wrote', os.path.join(FIG, 'P003_fig1_overlay.png'))
