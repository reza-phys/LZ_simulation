"""
P009_digitise_bands.py -- digitise Fig. 2 (calibrations) and Fig. 4 (science sample) of arXiv:2609.02823
from the PNG renderings in inputs/figures_png/.

Extracts, in {S1c, log10 S2c} space:
  * the NR-band median (solid red) and 10/90 % percentiles (dashed red) at chosen S1c values,
  * the event marker (black filled circle, Fig. 4),
  * the S1c at which each grey constant-energy contour (50,100,...,300 keV_nr) crosses the NR-band median,
    which is the paper's *own* S1c(E_nr) relation along the band,
  * the ER-equivalent energy label of each contour, taken from the figure text (typed in below).

Axis calibration: x from the frame (0 and 800 phd at the frame edges, verified with the dashed ROI edge
at S1c = 600); y from the dashed ROI box (log10 S2c = 4.15 top edge, 2.75 bottom edge) and the frame top (5.0).

Run from the simulation root:  .venv/bin/python output/code/P009_digitise_bands.py
Writes output/work/P009/digitised_fig.json and output/work/P009/figures/P009_fig_digitisation_check.png
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import image as mpimg

ROOT = "/Users/reza/LZ_simulation"
OUT = os.path.join(ROOT, "output/work/P009")
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)

# contour labels as printed on Fig. 2 / Fig. 4 (E_nr [keV], E_ee [keVee]) -- read from the PNG text
CONTOUR_LABELS = [(50.0, 11.8), (100.0, 25.5), (150.0, 40.0), (200.0, 55.1), (250.0, 70.6), (300.0, 86.5)]


def load(name):
    im = mpimg.imread(os.path.join(ROOT, "inputs/figures_png", name + ".png"))
    return (im[:, :, :3] * 255).astype(int)


def frame(rgb):
    dark = rgb.sum(axis=2) < 150
    H, W = dark.shape
    cols = np.where(dark.sum(axis=0) > 0.5 * H)[0]
    rows = np.where(dark.sum(axis=1) > 0.5 * W)[0]
    return cols.min(), cols.max(), rows.min(), rows.max()   # x_left, x_right, y_top, y_bottom


def roi_box(rgb, xl, xr, yt, yb):
    """Find the dashed grey ROI box: a mid-grey colour forming long dashed vertical/horizontal runs."""
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    grey = (np.abs(r - g) < 6) & (np.abs(g - b) < 6) & (r > 110) & (r < 175)
    # candidate vertical edge: column inside frame with many grey pixels
    colsum = grey[yt + 5:yb - 5, xl + 5:xr - 5].sum(axis=0)
    xc = np.argmax(colsum) + xl + 5
    rowsum = grey[yt + 5:yb - 5, xl + 5:xr - 5].sum(axis=1)
    order = np.argsort(-rowsum)
    # the two strongest, well separated rows are the top and bottom edges
    y1 = order[0] + yt + 5
    y2 = next(o for o in order if abs(o + yt + 5 - y1) > 50) + yt + 5
    return xc, min(y1, y2), max(y1, y2)


def main():
    res = {}
    figs = {}
    roi_cache = None
    for name in ["Fig2_Calibrations", "Fig4_science_sample_wbands"]:
        rgb = load(name)
        xl, xr, yt, yb = frame(rgb)
        # x calibration assuming frame spans 0..800 phd (labels 0 and 800 sit at the frame edges)
        def x_of(s1): return xl + (s1 / 800.0) * (xr - xl)
        def s1_of(x): return (x - xl) / (xr - xl) * 800.0
        if roi_cache is None:
            xc, yroi_top, yroi_bot = roi_box(rgb, xl, xr, yt, yb)
            roi_cache = (xc, yroi_top, yroi_bot, (xl, xr, yt, yb))
        else:
            # Fig. 4 has the same axes frame as Fig. 2 (checked below); its grey ER points confuse the
            # ROI-box finder, so reuse the Fig. 2 calibration.
            assert roi_cache[3] == (xl, xr, yt, yb), "frames differ; cannot reuse calibration"
            xc, yroi_top, yroi_bot = roi_cache[:3]
        s1_roi = s1_of(xc)
        # y calibration from ROI top (4.15) and bottom (2.75) edges
        slope = (4.15 - 2.75) / (yroi_top - yroi_bot)
        def l_of(y): return 4.15 + slope * (y - yroi_top)
        def y_of(l): return yroi_top + (l - 4.15) / slope
        ytop_check = l_of(yt)   # should be ~5.0
        # --- red band pixels
        red = (rgb[:, :, 0] > 170) & (rgb[:, :, 1] < 90) & (rgb[:, :, 2] < 120)
        # median curve: per column, the largest contiguous red run (solid line) -> take its centre
        med = {}
        for x in range(xl + 3, xr - 2):
            ys = np.where(red[:, x])[0]
            if ys.size == 0:
                continue
            # split into runs
            runs, cur = [], [ys[0]]
            for y in ys[1:]:
                if y == cur[-1] + 1:
                    cur.append(y)
                else:
                    runs.append(cur); cur = [y]
            runs.append(cur)
            med[x] = runs
        band = {}
        for s1 in [400.0, 450.0, 500.0, 540.1, 550.0, 600.0, 700.0]:
            x0 = int(round(x_of(s1)))
            allrows = []
            for x in range(x0 - 6, x0 + 7):
                for run in med.get(x, []):
                    allrows.append((np.mean(run), len(run)))
            if not allrows:
                continue
            rows = np.array([a for a, _ in allrows])
            # cluster into up to 3 groups by gap
            rows_sorted = np.sort(rows)
            groups, cur = [], [rows_sorted[0]]
            for rr in rows_sorted[1:]:
                if rr - cur[-1] < 6:
                    cur.append(rr)
                else:
                    groups.append(cur); cur = [rr]
            groups.append(cur)
            gl = sorted([np.mean(gp) for gp in groups])
            vals = [float(l_of(v)) for v in gl]
            band[str(s1)] = dict(n_groups=len(gl), log10S2c_groups_low_to_high=sorted(vals))
        # --- event marker (Fig. 4): black filled disc near (540, 3.97)
        ev = None
        if name.startswith("Fig4"):
            blk = rgb.sum(axis=2) < 120
            x0, y0 = int(x_of(540.0)), int(y_of(3.97))
            sub = blk[y0 - 40:y0 + 40, x0 - 40:x0 + 40]
            ys, xs = np.where(sub)
            ev = dict(S1c=float(s1_of(xs.mean() + x0 - 40)), log10S2c=float(l_of(ys.mean() + y0 - 40)),
                      n_pixels=int(len(xs)))
        # --- contour crossings with the NR median: use contour pixels just above/below the band
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        # contour colour as rendered: (150,166,166) core with lighter anti-aliased edges; require the teal tint
        cont = (np.abs(r - 150) < 30) & (np.abs(g - 166) < 30) & (np.abs(b - 166) < 30) & ((g - r) > 8)
        # NR median row per column from the solid (thickest) run, restricted to x > 150 phd
        ymed = {}
        for x, runs in med.items():
            if s1_of(x) < 120:
                continue
            runs2 = sorted(runs, key=len, reverse=True)
            ymed[x] = np.mean(runs2[0])
        xs_sorted = np.array(sorted(ymed))
        ys_med = np.array([ymed[x] for x in xs_sorted])
        # smooth median row (moving average) to remove dash artefacts
        k = 15
        ys_s = np.convolve(ys_med, np.ones(k) / k, mode="same")
        crossings = []
        for off in [-24, -18, 18, 24]:   # rows above (negative) and below the band, outside the dashed lines
            xs_hit = []
            for x, ym in zip(xs_sorted[k:-k], ys_s[k:-k]):
                yy = int(round(ym + off))
                if cont[yy - 1:yy + 2, x].any():
                    xs_hit.append(x)
            # cluster
            xs_hit = np.array(xs_hit)
            if xs_hit.size == 0:
                continue
            groups, cur = [], [xs_hit[0]]
            for xx in xs_hit[1:]:
                if xx - cur[-1] < 8:
                    cur.append(xx)
                else:
                    groups.append(cur); cur = [xx]
            groups.append(cur)
            crossings.append((off, [float(np.mean(gp)) for gp in groups]))
        # pair up: for each contour, interpolate x between the above and below hits to offset 0
        res_c = []
        above = [c for c in crossings if c[0] < 0]
        below = [c for c in crossings if c[0] > 0]
        if above and below:
            xa = np.array(sorted(above[0][1]))     # off=-24
            xa2 = np.array(sorted(above[1][1]))    # off=-18
            xb = np.array(sorted(below[0][1]))     # off=+18
            xb2 = np.array(sorted(below[1][1]))    # off=+24
            # match contours by nearest neighbour between the four rows (contours are ~15 px apart in x
            # between rows 18 and 24 px away, and > 100 px apart from each other)
            for x_a24 in xa:
                x_a18 = xa2[np.argmin(np.abs(xa2 - x_a24))]
                x_b18 = xb[np.argmin(np.abs(xb - x_a18))]
                x_b24 = xb2[np.argmin(np.abs(xb2 - x_b18))]
                if abs(x_b18 - x_a18) > 60:
                    continue
                # linear fit x(off) through the four points, evaluate at off = 0
                offs = np.array([-24, -18, 18, 24.0]); xx = np.array([x_a24, x_a18, x_b18, x_b24])
                p = np.polyfit(offs, xx, 1)
                res_c.append(float(s1_of(np.polyval(p, 0.0))))
        res_c = sorted(res_c)
        # Attach labels.  The contour crossing the NR median closest to the event (S1c ~ 544) is the
        # 250 keV_nr / 70.6 keVee contour (its label sits at S1c ~ 560 at the bottom of the frame); the
        # others are ~120-130 phd apart.  In Fig. 2 the AmBe calibration points produce spurious
        # 'crossings' at 450-520 phd, which are rejected by requiring a match to the expected spacing.
        labelled = []
        i250 = int(np.argmin(np.abs(np.array(res_c) - 544.0))) if res_c else None
        if i250 is not None:
            x250 = res_c[i250]
            for (E, Eee) in CONTOUR_LABELS:
                # expected crossing from the constant-Nq (combined-energy) nature of the contours:
                # S1c ~ x250 * (Eee/70.6) to first order; accept the nearest crossing within 12 %
                exp = x250 * (Eee / 70.6)
                cand = [c for c in res_c if abs(c - exp) < 0.06 * exp]
                if not cand:
                    continue
                s1 = min(cand, key=lambda c: abs(c - exp))
                labelled.append(dict(E_nr_keV=E, E_ee_keVee=Eee, S1c_at_NR_median=s1,
                                     log10S2c_at_NR_median=float(l_of(np.interp(x_of(s1), xs_sorted, ys_s)))))
        res[name] = dict(frame=dict(xl=int(xl), xr=int(xr), yt=int(yt), yb=int(yb)),
                         roi_edge_S1c=float(s1_roi), roi_rows=[int(yroi_top), int(yroi_bot)],
                         frame_top_log10S2c_check=float(ytop_check),
                         band_at_S1c=band, event=ev, raw_crossings_S1c=res_c, contours=labelled)
        figs[name] = (rgb, xl, xr, yt, yb, x_of, y_of, band, ev, labelled)
    with open(os.path.join(OUT, "digitised_fig.json"), "w") as f:
        json.dump(res, f, indent=1)
    # validation overlay
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, name in zip(axs, figs):
        rgb, xl, xr, yt, yb, x_of, y_of, band, ev, labelled = figs[name]
        ax.imshow(rgb)
        for s1, d in band.items():
            for v in d["log10S2c_groups_low_to_high"]:
                ax.plot(x_of(float(s1)), y_of(v), "c+", ms=8)
        if ev:
            ax.plot(x_of(ev["S1c"]), y_of(ev["log10S2c"]), "gx", ms=10)
        for c in labelled:
            ax.plot(x_of(c["S1c_at_NR_median"]), y_of(c["log10S2c_at_NR_median"]), "mo", mfc="none", ms=9)
        ax.set_xlim(x_of(300), x_of(800)); ax.set_ylim(y_of(3.8), y_of(4.2))
        ax.set_title(name + " (cyan: band samples; magenta: contour crossings; green: event)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures/P009_fig_digitisation_check.png"), dpi=130)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
