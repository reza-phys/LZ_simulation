#!/usr/bin/env python
"""
P070 -- Position as a discriminant: the (r, z) likelihood ratio between a uniform signal and
wall/RFR MSSI (and the other backgrounds) at the LZ 248 keV event, and the counterfactual
3D (S1c, S2c, position) single-event significance.

Stages
 1. Exact vector digitisation of Fig. 3 and Fig. S5a (PyMuPDF): axis calibration from the major
    tick marks, the two solid FV contours (min/max radial extent), the dashed reconstructed wall,
    the 5.4 t contours, and every plotted event (1710 science, 66 prompt veto, 55 delayed veto,
    the star).  FV volume/mass, distance-to-wall distribution of a uniform signal.
 2. Position PDFs normalised over the digitised FV: uniform (signal, accidentals, atm-nu),
    wall-MSSI (P033 photon-transport r-z maps per source + analytic exponentials, lambda 3-6 cm,
    true/reconstructed/FV-edge reference), RFR-MSSI (P033 maps + 214Pb E2 profile), neutrons
    (exp from the nearest surface, lambda 10-15 cm), a cathode-emission accidental variant.
    Likelihood ratios at the event.
 3. Population checks: uniformity of the science/delayed samples, wall/top concentration of the
    prompt-veto sample, the >=5 sigma-from-ER events of Fig. S5a.
 4. Counterfactual: composite position likelihood ratio, b_eff, single-event local Z with and
    without position, LEE-corrected global Z, expected gain over a uniform signal.

Run from the simulation root:  .venv/bin/python output/code/P070_position_likelihood.py
"""
import sys, os, json, math, glob
sys.path.insert(0, 'output/code')
from common import lzcommon as lz
import numpy as np
import pandas as pd
import pymupdf
from scipy import stats, special, optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = '/Users/reza/LZ_simulation'
OUT = f'{ROOT}/output/work/P070'; FIG = f'{OUT}/figures'
SRC = f'{ROOT}/inputs/arXiv_2609.02823_source'
P033 = f'{ROOT}/output/work/P033/runs'
os.makedirs(FIG, exist_ok=True)

# ----------------------------------------------------------------------------- constants
R_TPC = 72.8          # cm, true wall radius (recalled, likely; the Fig. 3 axis ends at 73.0^2)
H_TPC = 145.6         # cm, gate-cathode (the dashed active-volume lines sit at z = 0 and 145.6 exactly)
RHO_LXE = 2.86        # g/cm^3 at LZ operating temperature (recalled, likely; 2.86-2.89 in the literature)
EV_R, EV_Z = 45.9, lz.LZ['ev_z_above_cathode_cm']          # paper: (45.9^2 cm^2, 26.4 cm)
EV_D_TRUE, EV_D_RECO = lz.LZ['ev_r_from_true_wall_cm'], lz.LZ['ev_r_from_reco_wall_cm']
Z_TOP_TEXT, Z_BOT_TEXT = H_TPC - 12.8, 9.0                  # paper l. 139
MAP_ZLO, MAP_ZHI, MAP_DLO = 9.0, 132.8, 8.0                 # support of the P033 simplified FV
# P033 line weights (its MIX_W) and 214Pb branch fractions, per emitted photon
MIX_W = {352.: 0.356, 609.: 0.455, 1120.: 0.149, 1173.: 0.30, 1332.: 0.30, 1461.: 0.107, 1764.: 0.153, 2615.: 0.356}
PB214_W = {352.: 0.515, 295.: 0.485}        # P033's PB214[E]['frac']
SOURCES = ['wall', 'top', 'bottom', 'cathode']
PAL = dict(blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', yellow='#eda100', magenta='#e87ba4',
           green='#008300', violet='#4a3aa7', red='#e34948', ink='#0b0b0b', ink2='#52514e', muted='#898781', surf='#fcfcfb')
RES = {}

# ============================================================================= Stage 1: digitisation
def path_points(d):
    pts = []
    for it in d['items']:
        if it[0] == 'l':
            pts.append((it[1].x, it[1].y)); pts.append((it[2].x, it[2].y))
        elif it[0] == 'c':
            pts.append((it[1].x, it[1].y)); pts.append((it[4].x, it[4].y))
    return np.array(pts)

def calibrate(dr):
    """Major ticks: width 0.8, filled, single 'l' item. x-ticks are vertical segments at the bottom axis,
    z-ticks horizontal segments at the left axis, drift ticks at the right axis."""
    maj = [d for d in dr if d.get('width') and abs(d['width'] - 0.8) < 1e-3 and len(d['items']) == 1 and d.get('fill')]
    xt, zt, tt = [], [], []
    for d in maj:
        p1, p2 = d['items'][0][1], d['items'][0][2]
        if abs(p1.x - p2.x) < 1e-6 and p1.y > 360:          # bottom x-axis ticks
            xt.append(p1.x)
        elif abs(p1.y - p2.y) < 1e-6 and p1.x < 60:          # left z-axis ticks
            zt.append(p1.y)
        elif abs(p1.y - p2.y) < 1e-6 and p1.x > 420:         # right drift-time ticks
            tt.append(p1.y)
    xt = np.array(sorted(set(np.round(xt, 4)))); zt = np.array(sorted(set(np.round(zt, 4)), reverse=True)); tt = np.array(sorted(set(np.round(tt, 4))))
    r2_vals = np.array([0, 20, 30, 40, 50, 60, 70], float) ** 2
    assert len(xt) == 7 and len(zt) == 8 and len(tt) == 6, (len(xt), len(zt), len(tt))
    ax_ = np.polyfit(xt, r2_vals, 1); az_ = np.polyfit(zt, np.arange(0, 160, 20), 1); at_ = np.polyfit(tt, np.arange(0, 1200, 200), 1)
    cal = dict(x_slope=ax_[0], x_icpt=ax_[1], z_slope=az_[0], z_icpt=az_[1], t_slope=at_[0], t_icpt=at_[1],
               x_resid_cm2=float(np.max(np.abs(np.polyval(ax_, xt) - r2_vals))), z_resid_cm=float(np.max(np.abs(np.polyval(az_, zt) - np.arange(0, 160, 20)))))
    return cal

def to_rz(cal, x, y):
    r2 = np.polyval([cal['x_slope'], cal['x_icpt']], x); z = np.polyval([cal['z_slope'], cal['z_icpt']], y)
    return r2, z

def digitise(fname):
    page = pymupdf.open(fname)[0]; dr = page.get_drawings(); cal = calibrate(dr)
    frame = [d for d in dr if d.get('color') is None and d.get('fill') == (1.0, 1.0, 1.0)][-1]['rect']
    cal['frame_r2_max'] = float(to_rz(cal, frame.x1, 0)[0]); cal['frame_z_top'] = float(to_rz(cal, 0, frame.y0)[1]); cal['frame_z_bot'] = float(to_rz(cal, 0, frame.y1)[1])
    cal['Tmax_us'] = float(np.polyval([cal['t_slope'], cal['t_icpt']], np.polyval([1 / cal['z_slope'], -cal['z_icpt'] / cal['z_slope']], 0.0)))
    out = dict(cal=cal)
    def is_black(c): return c is not None and max(c) < 1e-3
    def is_blue54(c): return c is not None and abs(c[0] - 0.0118) < 1e-2 and abs(c[1] - 0.443) < 1e-2
    def is_orange(c): return c is not None and abs(c[0] - 0.855) < 1e-2 and abs(c[1] - 0.392) < 1e-2
    def is_bluedel(c): return c is not None and abs(c[0] - 0.1216) < 1e-2 and abs(c[1] - 0.4667) < 1e-2
    def is_star(c): return c is not None and abs(c[0] - 0.647) < 1e-2
    contours = {'47': [], '54': []}; hlines = {'47': [], '54': []}; dashed = []
    pts = dict(science=[], prompt=[], delayed=[], star=[])
    for d in dr:
        c, f, w, n = d.get('color'), d.get('fill'), d.get('width') or 0, len(d['items'])
        dashed_flag = bool(d.get('dashes')) and d['dashes'] not in ('[] 0', '')
        if abs(w - 1.4) < 1e-2 and f is None:
            key = '47' if is_black(c) else ('54' if is_blue54(c) else None)
            if key is None: continue
            if dashed_flag: dashed.append(path_points(d))
            elif n > 20: contours[key].append(path_points(d))
            elif n == 1: hlines[key].append(d['items'][0][1].y)
        elif n == 8 and f is not None and is_black(f): pts['science'].append(d['rect'])
        elif n == 2 and f is not None and is_orange(f): pts['prompt'].append(d['rect'])
        elif n == 8 and f is None and is_bluedel(c): pts['delayed'].append(d['rect'])
        elif f is not None and is_star(f): pts['star'].append(d['rect'])
    rows = []
    for k, rects in pts.items():
        for rc in rects:
            r2, z = to_rz(cal, 0.5 * (rc.x0 + rc.x1), 0.5 * (rc.y0 + rc.y1))
            rows.append(dict(sample=k, r2=float(r2), r=float(math.sqrt(max(r2, 0))), z=float(z)))
    out['points'] = pd.DataFrame(rows)
    for key in contours:
        cs = []
        for p in contours[key]:
            r2, z = to_rz(cal, p[:, 0], p[:, 1]); o = np.argsort(z); cs.append(pd.DataFrame(dict(r2=r2[o], z=z[o])))
        if cs:
            cs.sort(key=lambda c: c['r2'].mean())     # min extent first
            out[f'contour{key}_min'], out[f'contour{key}_max'] = cs[0], cs[1]
            zs = sorted(to_rz(cal, 0, np.array(hlines[key]))[1]); out[f'z_bot{key}'], out[f'z_top{key}'] = float(zs[0]), float(zs[-1])
    if dashed:
        p = dashed[0]; r2, z = to_rz(cal, p[:, 0], p[:, 1]); m = r2 > 55 ** 2
        o = np.argsort(z[m]); out['wall_reco'] = pd.DataFrame(dict(r2=r2[m][o], z=z[m][o])).drop_duplicates('z')
    return out

F3 = digitise(f'{SRC}/Fig3_roi_events_in_fv.pdf'); S5 = digitise(f'{SRC}/FigS5a_LE_MSSI_rz.pdf')
RES['calibration_fig3'] = F3['cal']
pts = F3['points']; star = pts[pts['sample'] == 'star'].iloc[0]
RES['star_digitised'] = dict(r=float(star.r), z=float(star.z), note='marker bbox centre; the star glyph anchor is ~0.4 cm below its bbox centre')
zgrid = np.arange(0.0, H_TPC + 1e-9, 0.25)
def contour_fn(df):
    z, r = df['z'].values, np.sqrt(np.clip(df['r2'].values, 0, None))
    return lambda zz: np.interp(zz, z, r, left=r[0], right=r[-1])
r_min47, r_max47 = contour_fn(F3['contour47_min']), contour_fn(F3['contour47_max'])
r_mean47 = lambda zz: 0.5 * (r_min47(zz) + r_max47(zz))
r_wall_reco = contour_fn(F3['wall_reco'])
z_bot47, z_top47 = F3['z_bot47'], F3['z_top47']
r_min54, r_max54 = contour_fn(S5['contour54_min']), contour_fn(S5['contour54_max']); z_bot54, z_top54 = S5['z_bot54'], S5['z_top54']
pd.DataFrame(dict(z=zgrid, r_min47=r_min47(zgrid), r_max47=r_max47(zgrid), r_mean47=r_mean47(zgrid), r_wall_reco=r_wall_reco(zgrid),
                  r_min54=r_min54(zgrid), r_max54=r_max54(zgrid))).to_csv(f'{OUT}/fv_contours.csv', index=False)

def volume(rfn, zlo, zhi):
    zz = np.linspace(zlo, zhi, 4001); return float(np.trapezoid(np.pi * rfn(zz) ** 2, zz))
geo = dict(z_bot_drawn=z_bot47, z_top_drawn=z_top47, gap_below_gate_drawn=H_TPC - z_top47, z_bot_text=Z_BOT_TEXT, z_top_text=Z_TOP_TEXT,
           z_bot54_drawn=z_bot54, z_top54_drawn=z_top54,
           r_contour_at_top=dict(min=float(r_min47(z_top47)), max=float(r_max47(z_top47))),
           r_contour_at_bottom=dict(min=float(r_min47(z_bot47)), max=float(r_max47(z_bot47))),
           r_contour_at_event_z=dict(min=float(r_min47(EV_Z)), max=float(r_max47(EV_Z)), mean=float(r_mean47(EV_Z))),
           r_wall_reco_at_event_z=float(r_wall_reco(EV_Z)), r_wall_reco_mid=float(r_wall_reco(70.0)), r_wall_reco_bottom=float(r_wall_reco(z_bot47)), r_wall_reco_top=float(r_wall_reco(z_top47)))
zz = np.linspace(z_bot47, z_top47, 2001)
geo['stand_off_true'] = dict(min_contour=dict(min=float(np.min(R_TPC - r_min47(zz))), max=float(np.max(R_TPC - r_min47(zz))), mean=float(np.mean(R_TPC - r_min47(zz)))),
                             max_contour=dict(min=float(np.min(R_TPC - r_max47(zz))), max=float(np.max(R_TPC - r_max47(zz))), mean=float(np.mean(R_TPC - r_max47(zz)))))
geo['stand_off_reco'] = dict(min_contour=dict(min=float(np.min(r_wall_reco(zz) - r_min47(zz))), max=float(np.max(r_wall_reco(zz) - r_min47(zz))), mean=float(np.mean(r_wall_reco(zz) - r_min47(zz)))),
                             max_contour=dict(min=float(np.min(r_wall_reco(zz) - r_max47(zz))), max=float(np.max(r_wall_reco(zz) - r_max47(zz))), mean=float(np.mean(r_wall_reco(zz) - r_max47(zz)))))
masses = {}
for lab, rf in [('min', r_min47), ('mean', r_mean47), ('max', r_max47)]:
    masses[f'4.7t_{lab}_drawn_z'] = volume(rf, z_bot47, z_top47) * RHO_LXE / 1e6
    masses[f'4.7t_{lab}_text_z'] = volume(rf, Z_BOT_TEXT, Z_TOP_TEXT) * RHO_LXE / 1e6
for lab, rf in [('min', r_min54), ('mean', lambda q: 0.5 * (r_min54(q) + r_max54(q))), ('max', r_max54)]:
    masses[f'5.4t_{lab}_drawn_z'] = volume(rf, z_bot54, z_top54) * RHO_LXE / 1e6
masses['active_7.0t'] = math.pi * R_TPC ** 2 * H_TPC * RHO_LXE / 1e6
masses['rho_needed_for_4.71_mean_drawn'] = 4.71e6 / volume(r_mean47, z_bot47, z_top47)
masses['rho_needed_for_4.71_max_drawn'] = 4.71e6 / volume(r_max47, z_bot47, z_top47)
geo['masses_t'] = masses
RES['geometry'] = geo

# ============================================================================= Stage 2: grid & PDFs
dr_, dz_ = 0.2, 0.2
rc = np.arange(dr_ / 2, R_TPC, dr_); zc = np.arange(dz_ / 2, H_TPC, dz_)
RR, ZZ = np.meshgrid(rc, zc, indexing='ij')                  # shape (nr, nz)
dV = 2 * np.pi * RR * dr_ * dz_
FV_full = (RR <= r_mean47(ZZ)) & (ZZ >= z_bot47) & (ZZ <= z_top47)           # digitised FV (mean contour)
FV_lr = FV_full & (ZZ >= MAP_ZLO) & (ZZ <= MAP_ZHI)                          # domain where the P033 maps have support
RES['grid'] = dict(dr=dr_, dz=dz_, V_FV_full_cm3=float(dV[FV_full].sum()), V_FV_lr_cm3=float(dV[FV_lr].sum()),
                   frac_excluded_by_map_support=float(1 - dV[FV_lr].sum() / dV[FV_full].sum()), mass_FV_full_t=float(dV[FV_full].sum() * RHO_LXE / 1e6))
D_TRUE = R_TPC - RR; D_RECO = r_wall_reco(ZZ) - RR; D_EDGE = r_mean47(ZZ) - RR
# uniform-signal distance distributions
V = dV[FV_full].sum()
sig_dist = {}
for lab, D in [('true_wall', D_TRUE), ('reco_wall', D_RECO), ('fv_edge', D_EDGE)]:
    sig_dist[lab] = {f'within_{c}cm': float(dV[FV_full & (D <= c)].sum() / V) for c in (2, 5, 10, 15, 20, 25)}
    sig_dist[lab]['beyond_event'] = float(dV[FV_full & (D >= {'true_wall': EV_D_TRUE, 'reco_wall': EV_D_RECO, 'fv_edge': r_mean47(EV_Z) - EV_R}[lab])].sum() / V)
sig_dist['event_d_edge_mean'] = float(r_mean47(EV_Z) - EV_R); sig_dist['event_d_edge_min'] = float(r_min47(EV_Z) - EV_R); sig_dist['event_d_edge_max'] = float(r_max47(EV_Z) - EV_R)
sig_dist['event_d_reco_from_dashed'] = float(r_wall_reco(EV_Z) - EV_R)
sig_dist['pos_class_volume_fraction'] = float(dV[FV_full & (D_TRUE >= 25) & (ZZ >= 20)].sum() / V)
RES['uniform_signal_distances'] = sig_dist

# ---- P033 maps
DBINS = np.concatenate([np.arange(8, 30, 1.0), np.arange(30, 66, 2.5), [R_TPC]]); ZBINS = np.linspace(0, H_TPC, 30)
def load_map(cls, srcs, mode='forced12'):
    tot = np.zeros((len(DBINS) - 1, len(ZBINS) - 1))
    for f in glob.glob(f'{P033}/*_{mode}.npz'):
        name = os.path.basename(f)[:-4]
        if name.startswith('var_'): continue
        src, E = name.split('_')[0], float(name.split('_')[1])
        if src not in srcs: continue
        meta = json.load(open(f[:-4] + '.json')); w = (PB214_W[E] if src == 'bulk214Pb' else MIX_W[E]) / meta['n_emitted'] / len(srcs)
        tot += np.load(f)[f'h2_{cls}'] * w
    return tot
def map_density(h2):
    """per-volume density on the grid from a (d, z) histogram whose support is the simplified FV."""
    dd = np.diff(DBINS); zlo = np.maximum(ZBINS[:-1], MAP_ZLO); zhi = np.minimum(ZBINS[1:], MAP_ZHI); ov = np.clip(zhi - zlo, 1e-9, None)
    dens_area = h2 / dd[:, None] / ov[None, :]
    i = np.searchsorted(DBINS, D_TRUE, side='right') - 1; j = np.searchsorted(ZBINS, ZZ, side='right') - 1
    ok = (i >= 0) & (i < len(dd)) & (j >= 0) & (j < len(ov)) & (D_TRUE >= MAP_DLO) & (ZZ >= MAP_ZLO) & (ZZ <= MAP_ZHI)
    rho = np.zeros_like(RR); rho[ok] = dens_area[i[ok], j[ok]] / (2 * np.pi * RR[ok]); return rho
maps = {}
for s in SOURCES + ['equal', 'bulk214Pb']:
    srcs = SOURCES if s == 'equal' else [s]
    maps[f'wall12_{s}'] = load_map('wall12', srcs)
    if s in ('equal', 'wall', 'bottom', 'cathode'): maps[f'rfr12_{s}'] = load_map('rfr12', srcs)
maps['wall_roi_equal'] = load_map('wall_roi', SOURCES, mode='general')

def normed(rho, dom=FV_lr):
    n = (rho * dV)[dom].sum(); return rho / n if n > 0 else rho * np.nan
def E2(x): return special.expn(2, x)
PDF = {}
PDF['uniform'] = normed(np.ones_like(RR))
for k, h in maps.items(): PDF[f'map_{k}'] = normed(map_density(h))
for lam in (3.0, 4.3, 6.0):
    PDF[f'wall_exp_true_l{lam}'] = normed(np.exp(-(D_TRUE - MAP_DLO) / lam))
    PDF[f'wall_exp_reco_l{lam}'] = normed(np.exp(-np.clip(D_RECO, 0, None) / lam))
    PDF[f'wall_exp_edge_l{lam}'] = normed(np.exp(-np.clip(D_EDGE, 0, None) / lam))
for lam in (2.7, 2.2, 1.5):
    PDF[f'rfr_pb214_E2_l{lam}'] = normed(0.5 * E2(np.clip(ZZ, 1e-3, None) / lam))
for lam in (4.0, 6.0, 10.0):
    PDF[f'rfr_expz_l{lam}'] = normed(np.exp(-(ZZ - MAP_ZLO) / lam))
D_SURF = np.minimum(np.minimum(D_TRUE, ZZ), H_TPC - ZZ)
for lam in (10.0, 14.2, 15.0):
    PDF[f'neutron_surf_l{lam}'] = normed(np.exp(-D_SURF / lam)); PDF[f'neutron_wall_l{lam}'] = normed(np.exp(-D_TRUE / lam))
V_D = 1.388e-3 * 1e3 / 1e3  # mm/us -> cm/us: 1.388 mm/us = 0.1388 cm/us (P022)
Z_CATH_SLAB = H_TPC - 0.1388 * 850.0   # apparent drift > 850 us (P022 cathode-emission loophole, +-10 % width tolerance)
PDF['acc_cathode_slab'] = normed((ZZ <= Z_CATH_SLAB).astype(float))
Z_CATH_SLAB2 = H_TPC - 0.1388 * 672.0
PDF['acc_cathode_slab20pct'] = normed((ZZ <= Z_CATH_SLAB2).astype(float))

ie, je = np.argmin(np.abs(rc - EV_R)), np.argmin(np.abs(zc - EV_Z))
ie_reco = np.argmin(np.abs(rc - (R_TPC - EV_D_RECO)))
LR = {k: float(PDF['uniform'][ie, je] / p[ie, je]) if p[ie, je] > 0 else float('inf') for k, p in PDF.items()}
LR_reco_pos = {k: float(PDF['uniform'][ie_reco, je] / p[ie_reco, je]) if p[ie_reco, je] > 0 else float('inf') for k, p in PDF.items() if k.startswith('map_wall12') or k.startswith('wall_exp_true')}
RES['LR_at_event'] = LR; RES['LR_if_true_distance_were_23.4cm'] = LR_reco_pos
# smoothed alternative for the coarse-binned maps: average of the LR over the 3x3 neighbourhood of (d, z) bins (+-1 cm, +-5 cm)
def lr_box(p, dr_cm=1.0, dz_cm=5.0):
    m = (np.abs(RR - EV_R) <= dr_cm) & (np.abs(ZZ - EV_Z) <= dz_cm); return float(PDF['uniform'][m].mean() / p[m].mean())
RES['LR_box_pm1cm_pm5cm'] = {k: lr_box(p) for k, p in PDF.items() if k.startswith('map_')}
# position-class LR (coarse): V_class/V / f_pos(real FV)
cls = FV_lr & (D_TRUE >= 25) & (ZZ >= 20)
RES['position_class'] = {k: dict(f_pos_realFV=float((p * dV)[cls].sum()), LR_class=float((PDF['uniform'] * dV)[cls].sum() / (p * dV)[cls].sum())) for k, p in PDF.items() if k.startswith('map_') or k.startswith('wall_exp_true') or k.startswith('rfr_')}
# f_pos of the map classes over the *simplified* FV (should reproduce P033: 0.0098 / 0.0021) as a validation
simp = (D_TRUE >= MAP_DLO) & (ZZ >= MAP_ZLO) & (ZZ <= MAP_ZHI)
RES['validation_P033_fpos_simplified_FV'] = {k: float((normed(map_density(maps[k]), simp) * dV)[simp & (D_TRUE >= 25) & (ZZ >= 20)].sum()) for k in ('wall12_equal', 'rfr12_equal', 'wall12_wall', 'wall12_top', 'wall12_bottom', 'wall12_cathode', 'wall_roi_equal')}
# depth (d) and height (z) profiles of the map PDFs inside the real FV
def profile(p, coord, bins):
    w = (p * dV)[FV_lr]; h, _ = np.histogram(coord[FV_lr], bins=bins, weights=w); return h / h.sum() / np.diff(bins)
dbins = np.arange(8, 46, 1.0); zbins_p = np.arange(MAP_ZLO, MAP_ZHI + 1e-9, 4.0)
prof = {}
for k in ('map_wall12_equal', 'map_wall12_wall', 'map_wall12_top', 'map_wall12_bottom', 'map_wall12_bulk214Pb', 'map_rfr12_equal', 'map_rfr12_bottom', 'map_wall_roi_equal', 'uniform', 'neutron_surf_l14.2', 'rfr_pb214_E2_l2.7'):
    prof[k] = dict(d=profile(PDF[k], D_TRUE, dbins).tolist(), z=profile(PDF[k], ZZ, zbins_p).tolist())
# lambda fits
def lam_fit(dens, centres, lo, hi):
    m = (centres >= lo) & (centres <= hi) & (np.array(dens) > 0); s, _ = np.polyfit(centres[m], np.log(np.array(dens)[m]), 1); return float(-1 / s)
dcen = 0.5 * (dbins[1:] + dbins[:-1]); zcen = 0.5 * (zbins_p[1:] + zbins_p[:-1])
RES['lambda_fits'] = dict(wall12_equal_d_10_40=lam_fit(prof['map_wall12_equal']['d'], dcen, 10, 40), wall12_bulk214Pb_d_10_30=lam_fit(prof['map_wall12_bulk214Pb']['d'], dcen, 10, 30),
                          rfr12_equal_z_12_60=lam_fit(prof['map_rfr12_equal']['z'], zcen, 12, 60), rfr12_bottom_z_12_60=lam_fit(prof['map_rfr12_bottom']['z'], zcen, 12, 60),
                          rfr12_equal_frac_z_lt_20=float(sum(np.array(prof['map_rfr12_equal']['z'])[zcen < 20]) * 4.0),
                          wall12_equal_frac_d_le_10=float(sum(np.array(prof['map_wall12_equal']['d'])[dcen < 10])), wall12_equal_frac_d_le_15=float(sum(np.array(prof['map_wall12_equal']['d'])[dcen < 15])))
pd.DataFrame({'d_centre': dcen, **{k: v['d'] for k, v in prof.items()}}).to_csv(f'{OUT}/profiles_d.csv', index=False)
pd.DataFrame({'z_centre': zcen, **{k: v['z'] for k, v in prof.items()}}).to_csv(f'{OUT}/profiles_z.csv', index=False)

# ============================================================================= Stage 3: populations
pop = {}
def uniformity(df, rfn=r_max47):
    d = df[(df.z >= z_bot47 - 0.5) & (df.z <= z_top47 + 0.5)].copy()
    d['u'] = d.r2 / rfn(d.z) ** 2; d['v'] = (d.z - z_bot47) / (z_top47 - z_bot47)
    inside = d[d.u <= 1.0]
    ks_u = stats.kstest(inside.u, 'uniform'); ks_v = stats.kstest(np.clip(inside.v, 0, 1), 'uniform')
    d_edge = r_mean47(inside.z) - inside.r
    vol_edge10 = float(dV[FV_full & (D_EDGE <= 10)].sum() / V); vol_top20 = float(dV[FV_full & (ZZ >= z_top47 - 20)].sum() / V); vol_bot20 = float(dV[FV_full & (ZZ <= z_bot47 + 20)].sum() / V)
    n = len(inside)
    return dict(n_total=int(len(df)), n_inside_rmax=n, n_outside_rmax=int((d.u > 1).sum()), n_between_min_max=int(((d.r > r_min47(d.z)) & (d.u <= 1)).sum()),
                KS_u_stat=float(ks_u.statistic), KS_u_p=float(ks_u.pvalue), KS_v_stat=float(ks_v.statistic), KS_v_p=float(ks_v.pvalue),
                frac_within10cm_edge=float((d_edge <= 10).mean()), expected_uniform=vol_edge10, n_within10cm_edge=int((d_edge <= 10).sum()), n_exp_within10=float(n * vol_edge10),
                frac_top20=float((inside.z >= z_top47 - 20).mean()), expected_top20=vol_top20, n_top20=int((inside.z >= z_top47 - 20).sum()),
                frac_bot20=float((inside.z <= z_bot47 + 20).mean()), expected_bot20=vol_bot20,
                frac_top20_and_within15_edge=float(((inside.z >= z_top47 - 20) & (d_edge <= 15)).mean()), expected_top20_within15=float(dV[FV_full & (ZZ >= z_top47 - 20) & (D_EDGE <= 15)].sum() / V),
                mean_z=float(inside.z.mean()), median_d_edge=float(np.median(d_edge)))
for s in ('science', 'delayed', 'prompt'):
    pop[s] = uniformity(pts[pts['sample'] == s])
# prompt-veto: distance to nearest surface, exponential MLE
pr = pts[pts['sample'] == 'prompt']; sdist = np.minimum(np.minimum(R_TPC - pr.r, H_TPC - pr.z), pr.z)
smin = float(sdist.min()); pop['prompt']['nearest_surface_median_cm'] = float(np.median(sdist)); pop['prompt']['lambda_MLE_from_min_cm'] = float(sdist.mean() - smin); pop['prompt']['s_min_cm'] = smin
pop['prompt']['frac_nearest_surface_le_15cm'] = float((sdist <= 15).mean())
pop['prompt']['nearest_is_top'] = int(((H_TPC - pr.z) < (R_TPC - pr.r)).sum()); pop['prompt']['nearest_is_wall'] = int(((R_TPC - pr.r) <= (H_TPC - pr.z)).sum())
sd_u = np.minimum(np.minimum(D_TRUE, H_TPC - ZZ), ZZ); pop['prompt']['expected_uniform_le_15cm'] = float(dV[FV_full & (sd_u <= 15)].sum() / V)
pop['prompt']['binom_p_ge_obs_le15'] = float(stats.binom.sf(int((sdist <= 15).sum()) - 1, len(pr), pop['prompt']['expected_uniform_le_15cm']))
# binomial tests for science-sample edge fraction; z histogram to interpret the z-KS
sc = pop['science']; sc['binom_two_sided_p_within10'] = float(stats.binomtest(sc['n_within10cm_edge'], sc['n_inside_rmax'], sc['expected_uniform']).pvalue)
scp = pts[pts['sample'] == 'science']; zb = np.linspace(z_bot47, z_top47, 11); hz, _ = np.histogram(scp.z, bins=zb)
vz = np.array([dV[FV_full & (ZZ >= zb[i]) & (ZZ < zb[i + 1])].sum() / V for i in range(10)])
sc['z_hist_10bins'] = dict(edges=zb.tolist(), observed=hz.tolist(), expected_uniform=(vz * len(scp)).tolist(), pulls=((hz - vz * len(scp)) / np.sqrt(vz * len(scp))).tolist(),
                           chi2=float((((hz - vz * len(scp)) ** 2) / (vz * len(scp))).sum()), chi2_p=float(stats.chi2.sf((((hz - vz * len(scp)) ** 2) / (vz * len(scp))).sum(), 9)))
dbe = np.array([0, 2, 4, 6, 8, 10, 15, 20, 30, 45, 70]); de = r_mean47(scp.z) - scp.r; hd_, _ = np.histogram(de, bins=dbe)
vd = np.array([dV[FV_full & (D_EDGE >= dbe[i]) & (D_EDGE < dbe[i + 1])].sum() / V for i in range(10)])
sc['dedge_hist'] = dict(edges=dbe.tolist(), observed=hd_.tolist(), expected_uniform=(vd * len(scp)).tolist(), pulls=((hd_ - vd * len(scp)) / np.sqrt(vd * len(scp))).tolist())
# S5a events
s5 = S5['points'].copy(); s5['d_true_wall'] = R_TPC - s5.r; s5['d_reco_wall'] = r_wall_reco(s5.z) - s5.r; s5['d_cathode'] = s5.z; s5['d_edge47'] = r_mean47(s5.z) - s5.r
s5['inside_47_mean'] = (s5.r <= r_mean47(s5.z)) & (s5.z >= z_bot47) & (s5.z <= z_top47)
s5['nearest_dead_cm'] = np.minimum(s5.d_true_wall, s5.z)
s5.to_csv(f'{OUT}/figS5a_points.csv', index=False)
# uniform-in-5.4t expectation for "within 4 cm of cathode or 12 cm of the true wall" (cross-like)
FV54 = (RR <= 0.5 * (r_min54(ZZ) + r_max54(ZZ))) & (ZZ >= z_bot54) & (ZZ <= z_top54); V54 = dV[FV54].sum()
near = FV54 & ((ZZ <= 4.0) | (D_TRUE <= 12.0)); q_near = float(dV[near].sum() / V54)
ncross = int((s5['sample'] == 'prompt').sum()); ncross_near = int(((s5['sample'] == 'prompt') & (s5.nearest_dead_cm <= 12.0) & ((s5.z <= 4) | (s5.d_true_wall <= 12))).sum())
# KS on nearest-dead-region distance: uniform-in-5.4t reference CDF
sd54 = np.minimum(D_TRUE, ZZ)[FV54]; w54 = dV[FV54]; o = np.argsort(sd54); cdf_x = sd54[o]; cdf_y = np.cumsum(w54[o]) / w54.sum()
def ks_vs_ref(vals):
    vals = np.sort(np.asarray(vals)); n = len(vals); F = np.interp(vals, cdf_x, cdf_y); Dn = float(np.max(np.maximum(np.arange(1, n + 1) / n - F, F - np.arange(n) / n)))
    return Dn, float(stats.kstwo.sf(Dn, n))
# the single black science dot of Fig. S5a coincides with the star (it is the event itself, drawn under the star marker)
s5['is_event_duplicate'] = (s5['sample'] == 'science') & (np.hypot(s5.r - EV_R, s5.z - EV_Z) < 1.0)
nb = s5[s5['sample'] == 'prompt'].nearest_dead_cm.values; nb_star = np.append(nb, min(EV_D_TRUE, EV_Z))
D1, p1 = ks_vs_ref(nb); D2, p2 = ks_vs_ref(nb_star)
pop['figS5a'] = dict(points=s5.to_dict('records'), q_near_uniform_54t=q_near, n_cross=ncross, n_cross_near=ncross_near, P_all_cross_near_if_uniform=float(q_near ** ncross),
                     science_dot_is_event=bool(s5.is_event_duplicate.any()), n_science_ge5sigma_excluding_event=int(((s5['sample'] == 'science') & ~s5.is_event_duplicate).sum()),
                     KS_crosses_only=dict(D=D1, p=p1, n=len(nb)), KS_crosses_plus_event=dict(D=D2, p=p2, n=len(nb_star)), mass_54t_mean_t=float(V54 * RHO_LXE / 1e6),
                     P_event_farther_than_25cm_if_uniform_54t=float(dV[FV54 & (np.minimum(D_TRUE, ZZ) >= 25)].sum() / V54))
# LR for each S5a point under analytic PDFs normalised over the 5.4 t volume
def normed54(rho): return rho / (rho * dV)[FV54].sum()
P54 = dict(uniform=normed54(np.ones_like(RR)), wall_exp_l4p3=normed54(np.exp(-np.clip(D_TRUE - 6.0, 0, None) / 4.3)), rfr_pb214_l2p7=normed54(0.5 * E2(np.clip(ZZ, 1e-3, None) / 2.7)), rfr_expz_l6=normed54(np.exp(-np.clip(ZZ - 2.0, 0, None) / 6.0)))
rows = []
for _, p in s5.iterrows():
    i, j = np.argmin(np.abs(rc - p.r)), np.argmin(np.abs(zc - p.z))
    rows.append(dict(sample=p['sample'], r=p.r, z=p.z, **{f'LR_uniform_over_{k}': float(P54['uniform'][i, j] / v[i, j]) if v[i, j] > 0 else float('inf') for k, v in P54.items() if k != 'uniform'}))
pop['figS5a_LR_54t'] = rows
RES['populations'] = pop
pts['d_true_wall'] = R_TPC - pts.r; pts['d_reco_wall'] = r_wall_reco(pts.z) - pts.r; pts['d_edge47_mean'] = r_mean47(pts.z) - pts.r
pts.to_csv(f'{OUT}/fig3_points.csv', index=False)

# ============================================================================= Stage 4: counterfactual likelihood
# neighbourhood background composition (S1c > 500 phd, within +-2 sigma_NR of the NR median), events per 2.84 t yr
b = dict(wall_mssi=0.0048 * 0.035,                       # LZ 4.7 t wall MSSI x P004 f_nb
         rfr_mssi=1e-4 * 0.035 * (0.067 / 0.053),        # LZ RFR MSSI x f_nb x P033 energy-class ratio
         accidentals=1.49e-4,                            # P022
         atm_nu=3.4e-5,                                  # P019 (all of the S1c > 500 phd CEvNS taken inside the neighbourhood)
         neutrons=7e-6)                                  # P013 central
btot = sum(b.values()); w = {k: v / btot for k, v in b.items()}
def composite(assign):
    """assign: background -> PDF key (or list of (key, weight)); returns R_pos = sum w_i / LR_i (event) and the field b_eff(x)/b."""
    field = np.zeros_like(RR); Rev = 0.0
    for k, spec in assign.items():
        spec = spec if isinstance(spec, list) else [(spec, 1.0)]
        for key, fw in spec:
            field += w[k] * fw * PDF[key] / PDF['uniform']; Rev += w[k] * fw / LR[key]
    return Rev, field
SCEN = {
    'central (P033 detector-gamma maps, equal mix; RFR map; uniform acc/nu; neutron surf 14.2)':
        dict(wall_mssi='map_wall12_equal', rfr_mssi='map_rfr12_equal', accidentals='uniform', atm_nu='uniform', neutrons='neutron_surf_l14.2'),
    'wall = 50% 214Pb bulk map + 50% detector-gamma':
        dict(wall_mssi=[('map_wall12_equal', 0.5), ('map_wall12_bulk214Pb', 0.5)], rfr_mssi='rfr_pb214_E2_l2.7', accidentals='uniform', atm_nu='uniform', neutrons='neutron_surf_l14.2'),
    'flattest: analytic exp lambda 6 cm from true wall, RFR exp-z 10 cm':
        dict(wall_mssi='wall_exp_true_l6.0', rfr_mssi='rfr_expz_l10.0', accidentals='uniform', atm_nu='uniform', neutrons='neutron_surf_l15.0'),
    'steepest: analytic exp lambda 3 cm, RFR 214Pb 2.7 cm':
        dict(wall_mssi='wall_exp_true_l3.0', rfr_mssi='rfr_pb214_E2_l2.7', accidentals='uniform', atm_nu='uniform', neutrons='neutron_surf_l10.0'),
    'reconstructed-wall reference (lambda 4.3 from the dashed wall)':
        dict(wall_mssi='wall_exp_reco_l4.3', rfr_mssi='map_rfr12_equal', accidentals='uniform', atm_nu='uniform', neutrons='neutron_surf_l14.2'),
    'FV-edge reference (lambda 4.3 from the digitised contour)':
        dict(wall_mssi='wall_exp_edge_l4.3', rfr_mssi='map_rfr12_equal', accidentals='uniform', atm_nu='uniform', neutrons='neutron_surf_l14.2'),
    'adverse: accidentals = cathode-emission slab (drift > 850 us)':
        dict(wall_mssi='map_wall12_equal', rfr_mssi='map_rfr12_equal', accidentals='acc_cathode_slab', atm_nu='uniform', neutrons='neutron_surf_l14.2'),
    'top-source wall map (flattest P033 source)':
        dict(wall_mssi='map_wall12_top', rfr_mssi='map_rfr12_equal', accidentals='uniform', atm_nu='uniform', neutrons='neutron_surf_l14.2'),
    'pessimistic: bottom-source wall map (12 keV scatters concentrate at low z), RFR map':
        dict(wall_mssi='map_wall12_bottom', rfr_mssi='map_rfr12_bottom', accidentals='uniform', atm_nu='uniform', neutrons='neutron_surf_l14.2'),
    'pessimistic: cathode-source wall map, RFR map':
        dict(wall_mssi='map_wall12_cathode', rfr_mssi='map_rfr12_equal', accidentals='uniform', atm_nu='uniform', neutrons='neutron_surf_l14.2'),
    'optimistic: wall-source map, RFR = 214Pb E2 (2.2 cm)':
        dict(wall_mssi='map_wall12_wall', rfr_mssi='rfr_pb214_E2_l2.2', accidentals='uniform', atm_nu='uniform', neutrons='neutron_surf_l14.2'),
    'no MSSI at all (cap: only uniform backgrounds + neutrons remain)':
        dict(wall_mssi='map_wall12_top', rfr_mssi='rfr_pb214_E2_l1.5', accidentals='uniform', atm_nu='uniform', neutrons='neutron_surf_l14.2'),
}
def Zbox(bb): return math.sqrt(max(2 * (math.log(1 / bb) - 1 + bb), 0.0))
def Zpois(bb): return float(stats.norm.isf(-math.expm1(-bb)))
b_anchor = optimize.brentq(lambda x: Zbox(x) - lz.LZ['local_sig_max'], 1e-8, 0.5)
def globalZ(Zloc, neff):
    p = stats.norm.sf(Zloc); pg = -math.expm1(neff * math.log1p(-p)); return float(stats.norm.isf(pg)), float(pg)
cf = dict(b_components=b, b_total=btot, weights=w, b_anchor_from_3p4sigma=b_anchor, Z_box_at_btot=Zbox(btot), Z_pois_at_btot=Zpois(btot),
          Z_box_at_P016_bH=Zbox(5.70e-4), scenarios={})
for name, assign in SCEN.items():
    Rev, field = composite(assign)
    d = dict(R_pos=Rev, bayes_boost=1 / Rev, b_eff_anchor=b_anchor * Rev, b_eff_nb=btot * Rev)
    for lab, b0 in (('anchor', b_anchor), ('nb', btot)):
        Z0, Z1 = Zbox(b0), Zbox(b0 * Rev); d[f'Z_{lab}'] = Z0; d[f'Z_{lab}_with_pos'] = Z1; d[f'dZ_{lab}'] = Z1 - Z0
        for neff in (12.2, 13.9):
            g0, pg0 = globalZ(Z0, neff); g1, pg1 = globalZ(Z1, neff); d[f'Zglob_{lab}_N{neff}'] = g0; d[f'Zglob_{lab}_with_pos_N{neff}'] = g1
    # expected gain for a uniform signal: distribution of R_pos(x) over the FV
    Rx = field[FV_lr]; wx = dV[FV_lr] / dV[FV_lr].sum(); Zx = np.array([Zbox(b_anchor * max(r, 1e-12)) for r in Rx]) - Zbox(b_anchor)
    o = np.argsort(Zx); cw = np.cumsum(wx[o])
    d['expected_dZ_uniform_signal'] = dict(mean=float((Zx * wx).sum()), median=float(Zx[o][np.searchsorted(cw, 0.5)]), p16=float(Zx[o][np.searchsorted(cw, 0.16)]), p84=float(Zx[o][np.searchsorted(cw, 0.84)]),
                                           frac_negative=float(wx[Zx < 0].sum()), frac_gain_gt_0p2=float(wx[Zx > 0.2].sum()), mean_R_pos=float((Rx * wx).sum()))
    d['assign'] = {k: (v if isinstance(v, str) else [list(t) for t in v]) for k, v in assign.items()}
    cf['scenarios'][name] = d
# per-background LR table for the paper
# source mixing with equal *normalised PDF* weight (rather than P033's equal photon emission, which weights the class by its per-source yield)
_srcLR = [LR[f'map_wall12_{s}'] for s in SOURCES]
LR_pdf_equal = 1.0 / np.mean([1.0 / x if np.isfinite(x) else 0.0 for x in _srcLR])
cf['LR_table'] = dict(
    wall_mssi=dict(P033_equal_map=LR['map_wall12_equal'], equal_normalised_pdf_mix=float(LR_pdf_equal), P033_wall_src=LR['map_wall12_wall'], P033_top_src=LR['map_wall12_top'], P033_bottom_src=LR['map_wall12_bottom'], P033_cathode_src=LR['map_wall12_cathode'],
                   bulk214Pb_map=LR['map_wall12_bulk214Pb'], all_wall_ROI_general=LR['map_wall_roi_equal'],
                   exp_true_l3=LR['wall_exp_true_l3.0'], exp_true_l4p3=LR['wall_exp_true_l4.3'], exp_true_l6=LR['wall_exp_true_l6.0'],
                   exp_reco_l4p3=LR['wall_exp_reco_l4.3'], exp_edge_l4p3=LR['wall_exp_edge_l4.3'], exp_edge_l3=LR['wall_exp_edge_l3.0'], exp_edge_l6=LR['wall_exp_edge_l6.0'],
                   box_pm1_pm5=RES['LR_box_pm1cm_pm5cm']['map_wall12_equal'], class_level=RES['position_class']['map_wall12_equal']['LR_class']),
    rfr_mssi=dict(P033_equal_map=LR['map_rfr12_equal'], P033_bottom_src=LR['map_rfr12_bottom'], pb214_E2_l2p7=LR['rfr_pb214_E2_l2.7'], pb214_E2_l1p5=LR['rfr_pb214_E2_l1.5'],
                  expz_l4=LR['rfr_expz_l4.0'], expz_l6=LR['rfr_expz_l6.0'], expz_l10=LR['rfr_expz_l10.0'], class_level=RES['position_class']['map_rfr12_equal']['LR_class']),
    neutrons=dict(surf_l10=LR['neutron_surf_l10.0'], surf_l14p2=LR['neutron_surf_l14.2'], surf_l15=LR['neutron_surf_l15.0'], wall_l14p2=LR['neutron_wall_l14.2']),
    accidentals=dict(uniform=1.0, cathode_slab_850us=LR['acc_cathode_slab'], cathode_slab_672us=LR['acc_cathode_slab20pct'], z_slab_cm=Z_CATH_SLAB, z_slab2_cm=Z_CATH_SLAB2),
    atm_nu=dict(uniform=1.0))
# what a hard FV cut leaves on the table: 5.4 t vs 4.7 t
cf['exposure_left'] = dict(mass_54_over_47_paper=5.4 / 4.71, extra_tyr=(5.4 - 4.71) * 220 / 365.25, LZ_MSSI_54t_annulus_science=0.03 + 0.003,
                           MSSI_annulus_in_nb=(0.03 + 0.003) * 0.035)
RES['counterfactual'] = cf

# ============================================================================= Figures
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': PAL['muted'], 'axes.labelcolor': PAL['ink2'], 'xtick.color': PAL['ink2'], 'ytick.color': PAL['ink2'], 'axes.titlecolor': PAL['ink'], 'figure.facecolor': 'white', 'axes.facecolor': PAL['surf']})
fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
ax = axs[0]
# LR field (log10 f_sig / f_wall) as a sequential single-hue background in r^2 - z
field = np.where(FV_lr & (PDF['map_wall12_equal'] > 0), PDF['uniform'] / np.where(PDF['map_wall12_equal'] > 0, PDF['map_wall12_equal'], 1), np.nan)
r2c = rc ** 2
pc = ax.pcolormesh(r2c, zc, np.log10(field).T, cmap='Blues', vmin=-1.5, vmax=2.5, shading='nearest', rasterized=True)
cb = fig.colorbar(pc, ax=ax, pad=0.02, fraction=0.05); cb.set_label(r'log$_{10}$ [uniform-signal PDF / wall-MSSI PDF]  (P033 map, equal mix)', color=PAL['ink2'])
cb.ax.tick_params(color=PAL['ink2'], labelcolor=PAL['ink2'])
zz2 = np.linspace(z_bot47, z_top47, 400)
ax.plot(r_min47(zz2) ** 2, zz2, color=PAL['ink'], lw=1.2); ax.plot(r_max47(zz2) ** 2, zz2, color=PAL['ink'], lw=1.2)
ax.hlines([z_bot47, z_top47], 0, [r_min47(z_bot47) ** 2, r_min47(z_top47) ** 2], color=PAL['ink'], lw=1.2)
zw = np.linspace(0, H_TPC, 400); ax.plot(r_wall_reco(zw) ** 2, zw, color=PAL['ink2'], lw=1.0, ls='--')
zz5 = np.linspace(z_bot54, z_top54, 400); ax.plot(r_min54(zz5) ** 2, zz5, color=PAL['aqua'], lw=1.0, ls=':'); ax.plot(r_max54(zz5) ** 2, zz5, color=PAL['aqua'], lw=1.0, ls=':')
sci = pts[pts['sample'] == 'science']; ax.scatter(sci.r2, sci.z, s=3, color=PAL['ink2'], alpha=0.35, lw=0, label='science (1710)')
dl = pts[pts['sample'] == 'delayed']; ax.scatter(dl.r2, dl.z, s=14, facecolors='none', edgecolors=PAL['blue'], lw=0.8, label='delayed veto (55)')
ax.scatter(pr.r2, pr.z, s=16, marker='x', color=PAL['orange'], lw=1.0, label='prompt veto (66)')
ax.scatter([EV_R ** 2], [EV_Z], s=160, marker='*', color=PAL['red'], edgecolors='white', lw=0.6, zorder=5, label='248 keV event')
ax.axvline((R_TPC - 25) ** 2, color=PAL['violet'], lw=0.8, ls='-.'); ax.axhline(20, color=PAL['violet'], lw=0.8, ls='-.')
ax.set_xlim(0, 73 ** 2); ax.set_ylim(-3, 150); ax.set_xlabel(r'reconstructed $r^2$ [cm$^2$]'); ax.set_ylabel('z above cathode [cm]')
ticks = np.array([0, 20, 30, 40, 50, 60, 70]); ax.set_xticks(ticks ** 2); ax.set_xticklabels([f'{t}$^2$' for t in ticks])
ax.legend(loc='lower left', fontsize=7.5, frameon=False); ax.set_title('Fig. 3 (vector-digitised) with the position likelihood ratio', fontsize=9.5, loc='left')
ax = axs[1]
lines = [('map_wall12_equal', 'wall MSSI, P033 map (equal mix)', PAL['blue'], '-'), ('map_wall12_bulk214Pb', r'wall MSSI, bulk $^{214}$Pb', PAL['violet'], '-'),
         ('map_rfr12_equal', 'RFR MSSI, P033 map', PAL['orange'], '-'), ('neutron_surf_l14.2', r'neutrons, $\lambda$ = 14.2 cm', PAL['aqua'], '-'), ('uniform', 'uniform (signal, accidentals, atm-$\\nu$)', PAL['ink2'], '--')]
for k, lab, col, ls in lines:
    ax.step(dbins[:-1], prof[k]['d'], where='post', color=col, lw=1.6, ls=ls, label=lab)
for lam, col in ((3.0, PAL['magenta']), (6.0, PAL['yellow'])):
    pd_ = profile(PDF[f'wall_exp_true_l{lam}'], D_TRUE, dbins); ax.step(dbins[:-1], pd_, where='post', color=col, lw=1.0, ls=':', label=fr'exp, $\lambda$ = {lam:.0f} cm')
ax.axvline(EV_D_TRUE, color=PAL['red'], lw=1.0); ax.text(EV_D_TRUE + 0.4, 1.6e-5, 'event\n26.9 cm', color=PAL['red'], fontsize=8, va='bottom')
ax.set_yscale('log'); ax.set_ylim(1e-5, 0.6); ax.set_xlim(8, 45); ax.set_xlabel('distance to the true wall [cm]'); ax.set_ylabel('normalised density per cm (inside the digitised FV)')
ax.legend(fontsize=7.5, frameon=False, loc='upper right'); ax.set_title('Radial profiles of the position PDFs', fontsize=9.5, loc='left')
for a in axs: a.grid(color='#e5e4e0', lw=0.5); a.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f'{FIG}/P070_fig1_rz_and_profiles.png', dpi=170); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.3))
ax = axs[0]
bb = np.logspace(-6.5, -1.5, 300); ax.plot(bb, [Zbox(x) for x in bb], color=PAL['ink2'], lw=1.6, label=r'$Z=\sqrt{2[\ln(1/b)-1+b]}$ (one event)')
cen = cf['scenarios'][list(SCEN)[0]]
ax.scatter([b_anchor], [Zbox(b_anchor)], s=60, color=PAL['red'], zorder=5, label=f'LZ 3.4σ anchor: b = {b_anchor:.2e}')
ax.scatter([cen['b_eff_anchor']], [cen['Z_anchor_with_pos']], s=60, color=PAL['blue'], zorder=5, label=f'with position: b_eff = {cen["b_eff_anchor"]:.2e}, Z = {cen["Z_anchor_with_pos"]:.2f}σ')
ax.annotate('', xy=(cen['b_eff_anchor'], cen['Z_anchor_with_pos']), xytext=(b_anchor, Zbox(b_anchor)), arrowprops=dict(arrowstyle='->', color=PAL['blue'], lw=1.2))
ax.set_xscale('log'); ax.set_xlabel('effective background in the event\'s (S1c, S2c[, position]) neighbourhood'); ax.set_ylabel('local significance [σ]'); ax.legend(fontsize=7.5, frameon=False, loc='lower left')
ax.set_title('Single-event local significance versus b', fontsize=9.5, loc='left')
ax = axs[1]
Rev, field = composite(SCEN[list(SCEN)[0]]); Zx = np.array([Zbox(b_anchor * max(r, 1e-12)) for r in field[FV_lr]]) - Zbox(b_anchor); wx = dV[FV_lr] / dV[FV_lr].sum()
ax.hist(Zx, bins=np.linspace(-1.5, 0.6, 43), weights=wx, color=PAL['blue'], alpha=0.85, lw=0)
ax.axvline(cen['dZ_anchor'], color=PAL['red'], lw=1.2); ax.text(cen['dZ_anchor'] - 0.03, ax.get_ylim()[1] * 0.9, f'event: +{cen["dZ_anchor"]:.2f}σ', color=PAL['red'], ha='right', fontsize=8)
ax.axvline(0, color=PAL['muted'], lw=0.8, ls='--')
ax.set_xlabel('ΔZ from adding position (uniform signal event, central scenario)'); ax.set_ylabel('fraction of the fiducial volume')
ax.set_title(f"Expected gain: median {cen['expected_dZ_uniform_signal']['median']:+.2f}σ, {100 * cen['expected_dZ_uniform_signal']['frac_negative']:.0f}% of the FV loses", fontsize=9.5, loc='left')
for a in axs: a.grid(color='#e5e4e0', lw=0.5); a.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f'{FIG}/P070_fig2_counterfactual_Z.png', dpi=170); plt.close(fig)

# ============================================================================= save
def conv(o):
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.bool_,)): return bool(o)
    raise TypeError(str(type(o)))
json.dump(RES, open(f'{OUT}/P070_results.json', 'w'), indent=1, default=conv)
rows = []
for bk, dd in cf['LR_table'].items():
    for k, v in dd.items(): rows.append(dict(background=bk, variant=k, LR=v))
pd.DataFrame(rows).to_csv(f'{OUT}/LR_table.csv', index=False)
rows = []
for name, d in cf['scenarios'].items():
    rows.append(dict(scenario=name, R_pos=d['R_pos'], bayes_boost=d['bayes_boost'], b_eff_anchor=d['b_eff_anchor'], Z_anchor=d['Z_anchor'], Z_with_pos=d['Z_anchor_with_pos'], dZ=d['dZ_anchor'],
                     Zglob_N12=d['Zglob_anchor_N12.2'], Zglob_with_pos_N12=d['Zglob_anchor_with_pos_N12.2'], dZ_nb=d['dZ_nb'], exp_dZ_median=d['expected_dZ_uniform_signal']['median'], exp_dZ_mean=d['expected_dZ_uniform_signal']['mean'], frac_FV_negative=d['expected_dZ_uniform_signal']['frac_negative']))
pd.DataFrame(rows).to_csv(f'{OUT}/scenarios.csv', index=False)

# ============================================================================= console summary
print('== calibration', json.dumps(RES['calibration_fig3'], indent=0, default=conv))
print('== star digitised', RES['star_digitised'])
print('== geometry', json.dumps(geo, indent=1, default=conv))
print('== grid', RES['grid'])
print('== uniform-signal distances', json.dumps(sig_dist, indent=1, default=conv))
print('== P033 f_pos validation (simplified FV)', RES['validation_P033_fpos_simplified_FV'])
print('== lambda fits', RES['lambda_fits'])
print('== LR at event'); [print(f'  {k:32s} {v:10.3f}') for k, v in LR.items()]
print('== LR box', RES['LR_box_pm1cm_pm5cm'])
print('== LR if true distance were 23.4', LR_reco_pos)
print('== position class', json.dumps(RES['position_class'], indent=1, default=conv))
print('== populations', json.dumps({k: v for k, v in pop.items() if k not in ('figS5a',)}, indent=1, default=conv))
print('== S5a', json.dumps(pop['figS5a'], indent=1, default=conv))
print('== counterfactual'); print(json.dumps({k: v for k, v in cf.items() if k not in ('scenarios', 'LR_table')}, indent=1, default=conv))
print('== LR table'); [print(f'  {bk:12s} {k:28s} {v:12.3f}') for bk, dd in cf['LR_table'].items() for k, v in dd.items()]
for name, d in cf['scenarios'].items():
    print(f"-- {name}\n   R_pos={d['R_pos']:.4f} boost={d['bayes_boost']:.2f} b_eff={d['b_eff_anchor']:.3e} Z {d['Z_anchor']:.2f}->{d['Z_anchor_with_pos']:.2f} (dZ {d['dZ_anchor']:+.3f}; nb-based dZ {d['dZ_nb']:+.3f}) "
          f"glob N12.2 {d['Zglob_anchor_N12.2']:.2f}->{d['Zglob_anchor_with_pos_N12.2']:.2f}; N13.9 {d['Zglob_anchor_N13.9']:.2f}->{d['Zglob_anchor_with_pos_N13.9']:.2f}; expected dZ {d['expected_dZ_uniform_signal']}")
