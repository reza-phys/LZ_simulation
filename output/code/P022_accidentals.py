"""
P022 -- Accidental coincidences at high S1: could an isolated 540 phd S1 and a 270-electron S2 fake the LZ event?

Run from the simulation root:  .venv/bin/python output/code/P022_accidentals.py

Sections
 1. Digitise the yellow "Accidentals" (and blue "Total") histograms in all three panels of Fig. 5
    (tick-calibrated pixel-colour method of P004, extended to the top and middle panels); per-panel
    integrals, the +-2 sigma_NR window of the S1c > 500 phd panel, the value near -1.5 sigma.
 2. Digitise Fig. S3 (accidentals model vs unphysical-drift-time (UDT) data): S1c projection (15 phd
    bins), log10 S2c projection (0.15 dex bins), UDT data points, and the viridis 2D map (colour ->
    value through the colour bar); fractions of the isolated-S1 spectrum above 500 phd and of the
    isolated-S2 spectrum in the event's log10 S2c window; the S2c shape at S1c = 500-600 phd from the
    2D map; 2D-map value at the event coordinate; cross-checks (projection sums vs 2.4 events).
 3. Factorisation N_acc = R_S1 R_S2 T_max T_live f_cuts: R_S1 R_S2 products implied by 2.7 events in
    220 live days with T_max digitised from Fig. 3 (right axis), and the implied isolated-S1 rate at
    500-600 phd for a bracket of isolated-S2 rates.
 4. Physical plausibility: RFR mass, ER-deposition rate in the RFR giving S1 in 500-600 phd (NEST-LZ ER
    yields, P010's ROI acceptance 19.4 keV), the S1 that must accompany any real 270-electron cluster,
    S2-width arithmetic for gate- and cathode-emission S2s at the event's drift time.
 5. Mismodelling factor k for P(>=1) = 10 %/50 % in the +-2 sigma neighbourhood; what the UDT data
    (pre-cut S3a and post-cut S3b) constrain at S1c > 500 phd; drift-time probabilities.
All numbers -> output/work/P022/*.json|csv ; figures -> output/work/P022/figures/.
"""
import csv
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage, stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "output/code")
from common import lzcommon as lz  # noqa: E402

OUT = "output/work/P022"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
R = {}

PNG = "inputs/figures_png/"
N_ACC_EXP, N_ACC_ERR = 2.7, 0.6      # Table I, science sample, expected
N_ACC_FIT = 2.6                      # Table I, fit result (Fig. 5 shows the post-fit model)
N_UDT_MODEL, N_UDT_MODEL_ERR, N_UDT_OBS = 2.4, 0.2, 3   # supplement, after all cuts
T_LIVE_S = lz.LZ["live_days"] * 86400.0
EV_S1C, EV_LOGS2C = lz.LZ["ev_S1c"], lz.LZ["ev_log10S2c"]
EV_SIG_NR = -1.5
SIG_NR_DEX = 0.022      # drawn NR band width at S1c ~ 540 (P009, digitised 10-90 % half-width / 1.28); MC 0.033
SIG_NR_DEX_ALT = 0.033
MU_NR = EV_LOGS2C + 1.5 * SIG_NR_DEX          # NR median at S1c = 540 implied by "1.5 sigma below"
MU_NR_ALT = EV_LOGS2C + 1.5 * SIG_NR_DEX_ALT
R["nr_band_at_event"] = {"sigma_dex": SIG_NR_DEX, "mu_NR": MU_NR, "sigma_alt": SIG_NR_DEX_ALT, "mu_NR_alt": MU_NR_ALT}


def load(name):
    return np.asarray(Image.open(PNG + name).convert("RGB")).astype(int)


# ======================================================================================
# 1. Fig. 5: three panels
# ======================================================================================
FIG5 = load("Fig5_NR_distance_3panels_no_sig.png")
# frames (long dark lines, found by scanning): rows 496 | 1299 | 2102 | 2905, cols 218 .. 2408
# x: major ticks at 492.5 (-6 sigma) ... 2135.5 (+6 sigma) -> 136.92 px / sigma ; -8 sigma at 218.7
X_L, X_R = 218.7, 2409.3
PX_PER_SIG = (2135.5 - 492.5) / 12.0
PANELS = {
    # name: (row_top, row_bottom, y_of_reference_major_tick, log10 value at that tick, px_per_decade)
    "S1c<250":     (496, 1299, 594.5, 2.0, (1158.5 - 594.5) / 4.0),     # ticks 10^2 .. 10^-2 ; bottom 10^-3
    "250<S1c<500": (1299, 2102, 1383.0, 1.0, (1982.5 - 1383.0) / 5.0),  # ticks 10^1 .. 10^-4 ; bottom 10^-5
    "S1c>500":     (2102, 2905, 2131.5, 2.0, (2711.5 - 2131.5) / 6.0),  # ticks 10^2,10^0,10^-2,10^-4 ; bottom 10^-6
}
COLOURS = {"Accidentals": (255, 220, 61), "Total": (0, 0, 255), "MSSI": (2, 81, 128),
           "NRs": (124, 174, 0), "ERs": (0, 194, 249), "Internal": (255, 0, 255)}
EDGES = np.arange(-8, 8.01, 0.5)
CENTRES = 0.5 * (EDGES[1:] + EDGES[:-1])


def digitise_panel(y_top, y_bot, y_ref, log_ref, ppd, tol=60):
    out = {}
    sub = FIG5[y_top + 3:y_bot - 2, :]
    for name, col in COLOURS.items():
        mask = np.abs(sub - np.array(col)).sum(axis=2) < tol
        vals = np.full(len(EDGES) - 1, np.nan)
        for i in range(len(EDGES) - 1):
            xa = int(X_L + (EDGES[i] + 8 + 0.12) * PX_PER_SIG)
            xb = int(X_L + (EDGES[i] + 8 + 0.38) * PX_PER_SIG)
            ys = []
            for x in range(xa, xb):
                yy = np.where(mask[:, x])[0]
                if len(yy):
                    ys.append(yy.min() + y_top + 3)     # top-most pixel of the line = histogram level
            if ys:
                vals[i] = 10 ** (log_ref - (np.median(ys) - y_ref) / ppd)
        out[name] = vals
    return out


fig5 = {}
for pname, (yt, yb, yref, lref, ppd) in PANELS.items():
    fig5[pname] = digitise_panel(yt, yb, yref, lref, ppd)
    # the panel's y range (for the record)
    fig5[pname]["_range"] = (10 ** (lref - (yb - yref) / ppd), 10 ** (lref - (yt - yref) / ppd))


def wsum(arr, lo, hi):
    sel = (EDGES[:-1] >= lo - 1e-9) & (EDGES[1:] <= hi + 1e-9)
    return np.nansum(arr[sel])


R["fig5"] = {}
for pname in PANELS:
    acc = fig5[pname]["Accidentals"]
    tot = fig5[pname]["Total"]
    nan_bins = int(np.isnan(acc).sum())
    # fill hidden accidental bins (line under another line) by log-interpolation, flagged
    acc_f = acc.copy()
    if nan_bins:
        ok = ~np.isnan(acc_f)
        acc_f[~ok] = 10 ** np.interp(CENTRES[~ok], CENTRES[ok], np.log10(acc_f[ok]))
    fig5[pname]["Accidentals_filled"] = acc_f
    R["fig5"][pname] = {
        "acc_sum_pm8": float(np.nansum(acc)), "acc_sum_pm8_filled": float(acc_f.sum()),
        "acc_hidden_bins": nan_bins, "acc_bins_hidden_at_sigma": [float(c) for c in CENTRES[np.isnan(acc)]],
        "acc_sum_pm2": float(wsum(acc_f, -2, 2)), "acc_sum_m3_p1": float(wsum(acc_f, -3, 1)),
        "acc_bin_m2_m15": float(acc_f[np.argmin(np.abs(CENTRES - (-1.75)))]),
        "acc_bin_m15_m1": float(acc_f[np.argmin(np.abs(CENTRES - (-1.25)))]),
        "acc_below_median": float(wsum(acc_f, -8, 0)), "acc_above_median": float(wsum(acc_f, 0, 8)),
        "total_sum_pm8": float(np.nansum(tot)), "total_sum_pm2": float(wsum(np.nan_to_num(tot), -2, 2)),
        "y_range": [float(v) for v in fig5[pname]["_range"]],
    }
tot_acc_fig5 = sum(R["fig5"][p]["acc_sum_pm8_filled"] for p in PANELS)
R["fig5"]["acc_sum_three_panels"] = tot_acc_fig5
R["fig5"]["acc_sum_vs_tableI_fit"] = tot_acc_fig5 / N_ACC_FIT
R["fig5"]["acc_sum_vs_tableI_exp"] = tot_acc_fig5 / N_ACC_EXP
R["fig5"]["frac_in_panel_S1c_lt250"] = R["fig5"]["S1c<250"]["acc_sum_pm8_filled"] / tot_acc_fig5
R["fig5"]["frac_in_panel_250_500"] = R["fig5"]["250<S1c<500"]["acc_sum_pm8_filled"] / tot_acc_fig5
R["fig5"]["frac_in_panel_gt500"] = R["fig5"]["S1c>500"]["acc_sum_pm8_filled"] / tot_acc_fig5
R["fig5"]["total_bottom_vs_caption"] = R["fig5"]["S1c>500"]["total_sum_pm8"] / lz.LZ["bkg_highS1_panel"][0]
N_NB = R["fig5"]["S1c>500"]["acc_sum_pm2"]                    # accidentals, S1c>500, |Delta|<=2 sigma_NR
N_PANEL = R["fig5"]["S1c>500"]["acc_sum_pm8_filled"]
R["fig5"]["acc_share_of_total_pm2_bottom"] = N_NB / R["fig5"]["S1c>500"]["total_sum_pm2"]
# the S1c>500 panel spans +-8 sigma_NR in log S2c: the log S2c interval it covers at S1c = 540
R["fig5"]["bottom_panel_logS2c_window"] = [MU_NR - 8 * SIG_NR_DEX, min(MU_NR + 8 * SIG_NR_DEX, lz.LZ["log10S2c_max"])]
R["fig5"]["pm2_window_logS2c"] = [MU_NR - 2 * SIG_NR_DEX, MU_NR + 2 * SIG_NR_DEX]

# ======================================================================================
# 2. Fig. S3 (a: normalisation stage, b: after all cuts)
# ======================================================================================
S3 = {
    "a": dict(png="FigS3a_accs_baseline.png", xl=152.4, xr=1164.0, yt=248, yb=1159, xr2=1437, cb=(1533, 1577),
              top_yt=19, top_ref_y=199.5, top_ref_log=0.0, top_ppd=(199.5 - 62.5) / 2.0,     # ticks 10^2 @62.5, 10^0 @199.5
              right_kind="lin", right_x0=1164.5, right_px_per_count=(1415.0 - 1164.5) / 100.0,  # "100" @1415
              cb_ticks=None),
    "b": dict(png="FigS3b_accs_all_cuts.png", xl=169.4, xr=1180.0, yt=248, yb=1159, xr2=1453, cb=(1550, 1594),
              top_yt=19, top_ref_y=56.0, top_ref_log=0.0, top_ppd=(235.5 - 56.0) / 3.0,     # ticks 10^0 @56, 10^-3 @235.5
              right_kind="log", right_ref_x=1405.0, right_ref_log=1.0, right_ppd=(1405.0 - 1180.0) / 4.0,  # 10^1 @1405, 10^-3 @ frame
              cb_ticks=[(431.0, -2), (656.0, -3), (882.0, -4), (1107.0, -5)]),
}
# shared log10 S2c axis: major ticks at 331.5 (4.0), 471.5 (3.8), ..., 1032.5 (3.0) -> 700.8 px / dex
Y_REF_S2, LOG_REF_S2, PX_PER_DEX = 331.5, 4.0, (1032.5 - 331.5) / 1.0
def y_to_logs2(y): return LOG_REF_S2 - (y - Y_REF_S2) / PX_PER_DEX
def logs2_to_y(l): return Y_REF_S2 - (l - LOG_REF_S2) * PX_PER_DEX
S1_BIN = 15.0
S1_EDGES = np.arange(0, 600.01, S1_BIN)
S2_BIN = 0.15
# log S2c bins: data-point centres at 2.854, 3.004, ... 4.053 -> edges
S2_EDGES = np.round(2.854 - 0.075 + S2_BIN * np.arange(0, 10), 4)     # 2.779 .. 4.129
VIR = matplotlib.colormaps["viridis"](np.linspace(0, 1, 256))[:, :3] * 255.0


def blobs(img, x0, x1, y0, y1, thresh=40, minsize=40):
    sub = img[y0:y1, x0:x1]
    black = sub.max(axis=2) < thresh
    lab, n = ndimage.label(black)
    cents = ndimage.center_of_mass(black, lab, range(1, n + 1))
    sizes = ndimage.sum(black, lab, range(1, n + 1))
    return [(c[1] + x0, c[0] + y0, int(s)) for c, s in zip(cents, sizes) if s >= minsize]


def digitise_S3(key):
    P = S3[key]
    im = load(P["png"])
    xl, xr, yt, yb = P["xl"], P["xr"], P["yt"], P["yb"]
    px_per_phd = (xr - xl) / 600.0
    res = {}
    # ---- top projection: blue (0,0,255) line, per 15 phd bin (top-most blue pixel in the middle 60 % of the bin)
    top = im[P["top_yt"] + 3:yt - 2, :]
    blue = np.abs(top - np.array([0, 0, 255])).sum(axis=2) < 60
    vals = np.full(len(S1_EDGES) - 1, np.nan)
    for i in range(len(S1_EDGES) - 1):
        xa = int(xl + (S1_EDGES[i] + 0.2 * S1_BIN) * px_per_phd)
        xb = int(xl + (S1_EDGES[i] + 0.8 * S1_BIN) * px_per_phd)
        ys = [np.where(blue[:, x])[0].min() + P["top_yt"] + 3 for x in range(xa, xb) if blue[:, x].any()]
        if ys:
            vals[i] = 10 ** (P["top_ref_log"] - (np.median(ys) - P["top_ref_y"]) / P["top_ppd"])
    res["top_model"] = vals
    # UDT data points in the top projection (black dots)
    dots = blobs(im, int(xl) + 3, int(xr) - 3, P["top_yt"] + 3, yt - 2)
    top_data = np.zeros(len(S1_EDGES) - 1)
    for x, y, s in dots:
        if s < 70:      # tick marks are smaller than dots (~80 px)
            continue
        s1 = (x - xl) / px_per_phd
        val = 10 ** (P["top_ref_log"] - (y - P["top_ref_y"]) / P["top_ppd"])
        i = int(np.clip(s1 // S1_BIN, 0, len(top_data) - 1))
        top_data[i] = round(val)
    res["top_data"] = top_data
    # ---- right projection: blue line, per 0.15 dex bin (right-most blue pixel over the middle 60 % of the bin rows)
    right = im[:, int(xr) + 3:P["xr2"] - 2]
    blue_r = np.abs(right - np.array([0, 0, 255])).sum(axis=2) < 60
    rvals = np.full(len(S2_EDGES) - 1, np.nan)
    for i in range(len(S2_EDGES) - 1):
        ya = int(logs2_to_y(S2_EDGES[i + 1] - 0.2 * S2_BIN)); yb_ = int(logs2_to_y(S2_EDGES[i] + 0.2 * S2_BIN))
        ya, yb_ = max(ya, yt + 3), min(yb_, yb - 3)
        xs = [np.where(blue_r[y, :])[0].max() + int(xr) + 3 for y in range(ya, yb_) if blue_r[y, :].any()]
        if xs:
            xm = np.median(xs)
            if P["right_kind"] == "lin":
                rvals[i] = (xm - P["right_x0"]) / P["right_px_per_count"]
            else:
                rvals[i] = 10 ** (P["right_ref_log"] - (P["right_ref_x"] - xm) / P["right_ppd"])
    res["right_model"] = rvals
    rdots = blobs(im, int(xr) + 3, P["xr2"] - 2, yt + 3, yb - 3)
    right_data = np.zeros(len(S2_EDGES) - 1)
    for x, y, s in rdots:
        if s < 70:
            continue
        l = y_to_logs2(y)
        val = (x - P["right_x0"]) / P["right_px_per_count"] if P["right_kind"] == "lin" else \
            10 ** (P["right_ref_log"] - (P["right_ref_x"] - x) / P["right_ppd"])
        i = int(np.clip((l - S2_EDGES[0]) // S2_BIN, 0, len(right_data) - 1))
        right_data[i] = round(val)
    res["right_data"] = right_data
    # ---- colour bar calibration
    cb0, cb1 = P["cb"]
    if P["cb_ticks"] is None:
        # scan for tick marks right of the bar (wider band); fall back to label rows read from the PNG
        m = im.sum(axis=2) < 250
        band = m[yt:yb + 1, cb1 + 3:cb1 + 40].any(axis=1)
        rows = np.where(band)[0] + yt
        groups = [g for g in np.split(rows, np.where(np.diff(rows) > 1)[0] + 1) if len(g)]
        tick_rows = [float(g.mean()) for g in groups]
        res["cb_tick_rows_found"] = tick_rows
        P["cb_ticks"] = [(505.0, 0), (795.0, -1), (1085.0, -2)] if len(tick_rows) < 2 else \
            [(r, 0 - k) for k, r in enumerate(tick_rows)]
        res["cb_ticks_used"] = P["cb_ticks"]
    ticks = np.array(P["cb_ticks"], dtype=float)
    slope = np.polyfit(ticks[:, 0], ticks[:, 1], 1)     # log10 value = a*y + b
    cb_top_log, cb_bot_log = np.polyval(slope, yt), np.polyval(slope, yb)
    res["cb_log_range"] = [float(cb_bot_log), float(cb_top_log)]
    # verify the colour bar is viridis: colours along the bar vs index
    bar = im[yt + 2:yb - 2, (cb0 + cb1) // 2]
    idx = np.array([np.argmin(((VIR - c) ** 2).sum(axis=1)) for c in bar])
    res["cb_viridis_monotonic_corr"] = float(np.corrcoef(idx, -np.arange(len(idx)))[0, 1])

    def colour_to_log(rgb):
        t = np.argmin(((VIR - rgb) ** 2).sum(axis=1)) / 255.0
        return cb_bot_log + t * (cb_top_log - cb_bot_log)

    # ---- 2D map: per-pixel log value in the S1c 500-600 phd column, binned in log S2c (0.05 dex)
    def map_col(s1lo, s1hi):
        xa = int(xl + s1lo * px_per_phd) + 1; xb = int(xl + s1hi * px_per_phd) - 1
        return np.array([[colour_to_log(im[y, x]) for x in range(xa, xb)] for y in range(yt + 3, yb - 3)])
    col = map_col(500, 600)
    ys = np.arange(yt + 3, yb - 3)
    logs2 = y_to_logs2(ys)
    # exclude rows contaminated by the red/cyan contour lines and black points (non-viridis colours)
    sub = im[yt + 3:yb - 3, int(xl + 500 * px_per_phd) + 1:int(xl + 600 * px_per_phd) - 1]
    dist = np.array([[((VIR - sub[j, i]) ** 2).sum(axis=1).min() for i in range(sub.shape[1])] for j in range(sub.shape[0])])
    good = dist < 400            # within ~20 RGB units of a viridis colour
    dens = np.where(good, 10 ** col, np.nan)
    row_dens = np.nanmedian(dens, axis=1)                 # per-pixel "counts per 2D bin" as a function of log S2c
    res["col_logs2"] = logs2; res["col_dens"] = row_dens
    # value at the event coordinate
    xe = int(xl + EV_S1C * px_per_phd); ye = int(logs2_to_y(EV_LOGS2C))
    res["map_log_at_event"] = float(colour_to_log(im[ye, xe]))
    res["map_rgb_at_event"] = [int(v) for v in im[ye, xe]]
    # column shape: fraction of the column's density in log S2c windows
    def frac_window(lo, hi):
        sel = (logs2 >= lo) & (logs2 <= hi) & ~np.isnan(row_dens)
        allsel = ~np.isnan(row_dens)
        return float(np.nansum(row_dens[sel]) / np.nansum(row_dens[allsel]))
    res["frac_col_pm2"] = frac_window(MU_NR - 2 * SIG_NR_DEX, MU_NR + 2 * SIG_NR_DEX)
    res["frac_col_pm2_alt"] = frac_window(MU_NR_ALT - 2 * SIG_NR_DEX_ALT, MU_NR_ALT + 2 * SIG_NR_DEX_ALT)
    res["frac_col_pm8"] = frac_window(MU_NR - 8 * SIG_NR_DEX, min(MU_NR + 8 * SIG_NR_DEX, 4.15))
    res["frac_col_39_40"] = frac_window(3.9, 4.0)
    res["col_logs2_range_plotted"] = [float(logs2.min()), float(logs2.max())]
    # density ratio top/bottom of the column (gradient)
    top_sel = (logs2 > 3.95) & (logs2 < 4.05); bot_sel = (logs2 > 2.9) & (logs2 < 3.0)
    res["col_dens_ratio_logS2c_4_over_3"] = float(np.nanmedian(row_dens[top_sel]) / np.nanmedian(row_dens[bot_sel]))
    # effective 2D bin height from the normalisation Sum_rows dens * (px/h_bin) = P_top(500-600): px per bin
    P_top_500_600 = np.nansum(vals[(S1_EDGES[:-1] >= 495)])   # bins 495-600 (7 bins) ~ 500-600
    n_cols_bins = 7.0
    px_per_row = 1.0 / PX_PER_DEX          # dex per pixel row
    # Sum over rows of density (per bin) * (dex per row / dex per bin) * (n S1 bins) = P_top -> dex per bin
    dex_per_bin = np.nansum(row_dens) * px_per_row * n_cols_bins / P_top_500_600 if P_top_500_600 > 0 else np.nan
    res["map_dex_per_bin_from_norm"] = float(dex_per_bin)
    # integral of the 2D density over the +-2 sigma box (S1c 500-600) in UDT normalisation
    sel = (logs2 >= MU_NR - 2 * SIG_NR_DEX) & (logs2 <= MU_NR + 2 * SIG_NR_DEX)
    res["map_box_pm2_counts_udt"] = float(np.nansum(row_dens[sel]) * px_per_row / dex_per_bin * n_cols_bins) if dex_per_bin > 0 else np.nan
    res["map_bins_S1_used"] = n_cols_bins
    return res


S3R = {k: digitise_S3(k) for k in ("a", "b")}
R["figS3"] = {}
for k, res in S3R.items():
    tm, td, rm, rd = res["top_model"], res["top_data"], res["right_model"], res["right_data"]
    hi = S1_EDGES[:-1] >= 495
    R["figS3"][k] = {
        "top_model_sum": float(np.nansum(tm)), "top_data_sum": float(td.sum()),
        "right_model_sum": float(np.nansum(rm)), "right_data_sum": float(rd.sum()),
        "top_model_ge495": float(np.nansum(tm[hi])), "top_data_ge495": float(td[hi].sum()),
        "frac_S1_ge495": float(np.nansum(tm[hi]) / np.nansum(tm)),
        "frac_S1_ge495_data": float(td[hi].sum() / td.sum()) if td.sum() else np.nan,
        "top_model_first_bin": float(tm[0]), "top_model_bins_ge495": [float(v) for v in tm[hi]],
        "top_data_bins_ge495": [float(v) for v in td[hi]],
        "right_model_bins": [float(v) for v in rm], "right_data_bins": [float(v) for v in rd],
        "right_bin_edges": [float(v) for v in S2_EDGES],
        "frac_S2_bin_containing_event": float(rm[int((EV_LOGS2C - S2_EDGES[0]) // S2_BIN)] / np.nansum(rm)),
        "frac_S2_top_two_bins_3p83_4p13": float(np.nansum(rm[-2:]) / np.nansum(rm)),
        "frac_col_pm2": res["frac_col_pm2"], "frac_col_pm2_alt": res["frac_col_pm2_alt"],
        "frac_col_pm8": res["frac_col_pm8"], "frac_col_39_40": res["frac_col_39_40"],
        "col_dens_ratio_logS2c_4_over_3": res["col_dens_ratio_logS2c_4_over_3"],
        "map_log_at_event": res["map_log_at_event"], "map_rgb_at_event": res["map_rgb_at_event"],
        "map_dex_per_bin_from_norm": res["map_dex_per_bin_from_norm"],
        "map_box_pm2_counts_udt": res["map_box_pm2_counts_udt"],
        "cb_log_range": res["cb_log_range"], "cb_viridis_monotonic_corr": res["cb_viridis_monotonic_corr"],
        "cb_ticks_used": res.get("cb_ticks_used"), "cb_tick_rows_found": res.get("cb_tick_rows_found"),
    }
b = R["figS3"]["b"]; a = R["figS3"]["a"]
b["top_model_sum_vs_2p4"] = b["top_model_sum"] / N_UDT_MODEL
b["right_model_sum_vs_2p4"] = b["right_model_sum"] / N_UDT_MODEL
a["data_over_model_ge495"] = a["top_data_ge495"] / a["top_model_ge495"]
a["data_over_model_ge495_stat_err"] = np.sqrt(a["top_data_ge495"]) / a["top_model_ge495"]
a["data_over_model_all"] = a["top_data_sum"] / a["top_model_sum"]
# cut survival at high S1 vs overall (b/a), and the FV-independent prediction of the science-sample accidentals at S1c>500
R["figS3"]["cut_survival_all"] = b["top_model_sum"] / a["top_model_sum"]
R["figS3"]["cut_survival_ge495"] = b["top_model_ge495"] / a["top_model_ge495"]
R["figS3"]["pred_science_acc_S1c_gt500_allS2"] = N_ACC_FIT * b["frac_S1_ge495"]
R["figS3"]["pred_science_acc_S1c_gt500_pm8"] = N_ACC_FIT * b["frac_S1_ge495"] * b["frac_col_pm8"]
R["figS3"]["pred_science_acc_S1c_gt500_pm2"] = N_ACC_FIT * b["frac_S1_ge495"] * b["frac_col_pm2"]
R["figS3"]["pred_science_acc_S1c_gt500_pm2_alt"] = N_ACC_FIT * b["frac_S1_ge495"] * b["frac_col_pm2_alt"]
R["figS3"]["ratio_fig5_over_S3pred_pm2"] = N_NB / R["figS3"]["pred_science_acc_S1c_gt500_pm2"]
R["figS3"]["ratio_fig5_over_S3pred_pm8"] = N_PANEL / R["figS3"]["pred_science_acc_S1c_gt500_pm8"]
R["figS3"]["ratio_fig5_over_S3map_pm2"] = N_NB / (b["map_box_pm2_counts_udt"] * N_ACC_FIT / N_UDT_MODEL)
# after-cut UDT constraint at S1c>500: expectation m_hi = 2.4 * f, observed 0
m_hi = N_UDT_MODEL * b["frac_S1_ge495"]
R["figS3"]["udt_postcut_expected_ge495"] = m_hi
R["figS3"]["udt_postcut_k_UL90"] = -np.log(0.10) / m_hi
R["figS3"]["udt_postcut_k_UL95"] = -np.log(0.05) / m_hi
R["figS3"]["udt_postcut_overall_k"] = [N_UDT_OBS / N_UDT_MODEL, np.sqrt(N_UDT_OBS) / N_UDT_MODEL]
# pre-cut constraint: Poisson interval on data/model at S1c>=495 (22 events)
n = a["top_data_ge495"]
R["figS3"]["udt_precut_ge495_ratio_68"] = [stats.gamma.ppf(0.16, n + 0.5) / a["top_model_ge495"],
                                            stats.gamma.ppf(0.84, n + 0.5) / a["top_model_ge495"]]
R["figS3"]["udt_precut_ge495_k_UL95"] = stats.gamma.ppf(0.95, n + 1) / a["top_model_ge495"]

# ======================================================================================
# 3. Fig. 3 right axis: maximum drift time; drift time of the event
# ======================================================================================
F3 = load("Fig3_roi_events_in_fv.png")
m3 = F3.sum(axis=2) < 250
H3, W3 = m3.shape
# frame: long dark lines
rows = [y for y in range(H3) if m3[y, int(0.3 * W3):int(0.7 * W3)].mean() > 0.95]
cols = [x for x in range(W3) if m3[int(0.3 * H3):int(0.7 * H3), x].mean() > 0.95]
f_top, f_bot, f_l, f_r = min(rows), max(rows), min(cols), max(cols)


def edge_ticks(xe, inward, y0, y1):
    res = []
    for y in range(y0, y1):
        L = 0
        for k in range(2, 40):
            if m3[y, xe + inward * k]:
                L += 1
            else:
                break
        if L >= 4:
            res.append((y, L))
    g = []
    for y, L in res:
        if g and y - g[-1][-1][0] <= 1:
            g[-1].append((y, L))
        else:
            g.append([(y, L)])
    return [(float(np.mean([a_ for a_, _ in grp])), max(b_ for _, b_ in grp)) for grp in g]


lt = edge_ticks(f_l, +1, f_top + 3, f_bot - 3); rt = edge_ticks(f_r, -1, f_top + 3, f_bot - 3)
lmaj_all = sorted([y for y, L in lt if L >= 15]); rmaj = sorted([y for y, L in rt if L >= 15])
# the left edge also carries the solid FV boundary lines (z = 9 cm and z = 135.6 cm) and the dashed cathode/gate
# lines, which look like long ticks: keep only the regularly spaced z ticks (20 cm = ~130 px apart), anchored on
# the lowest one (z = 0)
diffs = np.diff(lmaj_all); spacing = float(np.median([d for d in diffs if 110 < d < 150]))
y0 = lmaj_all[-1]
lmaj = [y for y in lmaj_all if abs((y0 - y) / spacing - round((y0 - y) / spacing)) < 0.08]
# left: z = 0,20,...,140 from the bottom up ; right: drift = 0,200,...,1000 (6 major, top down)
z_vals = [20 * round((y0 - y) / spacing) for y in lmaj]
z_fit = np.polyfit(lmaj, z_vals, 1) if len(lmaj) >= 2 else None
d_fit = np.polyfit(rmaj[:6], np.arange(0, 1001, 200)[:len(rmaj[:6])], 1) if len(rmaj) >= 2 else None
R["fig3"] = {"frame": [f_top, f_bot, f_l, f_r], "left_long_ticks_all": lmaj_all, "left_z_ticks_used": lmaj,
             "z_tick_values": z_vals, "right_major_ticks": rmaj, "z_tick_spacing_px": spacing}
if z_fit is not None and d_fit is not None:
    y_z0 = (0 - z_fit[1]) / z_fit[0]; y_zmax = (145.6 - z_fit[1]) / z_fit[0]
    T_MAX = float(np.polyval(d_fit, y_z0)); T_GATE = float(np.polyval(d_fit, y_zmax))
    y_ev = (26.4 - z_fit[1]) / z_fit[0]; T_EV = float(np.polyval(d_fit, y_ev))
    R["fig3"].update({"T_max_us_at_z0": T_MAX, "T_at_gate_us": T_GATE, "T_event_us": T_EV,
                      "drift_velocity_mm_per_us": 145.6 * 10 / (T_MAX - T_GATE),
                      "z_px_per_cm": float(-1 / z_fit[0]), "drift_px_per_us": float(d_fit[0])})
else:
    T_MAX, T_EV, T_GATE = 1070.0, 876.0, 0.0
DRIFT_LEN_CM = 145.6            # recalled (likely): LZ gate-cathode drift length
V_D = DRIFT_LEN_CM * 10 / (T_MAX - T_GATE)          # mm/us
T_FV_LO = 12.8 * 10 / V_D + T_GATE                  # FV upper boundary 12.8 cm below gate
T_FV_HI = (DRIFT_LEN_CM - 9.0) * 10 / V_D + T_GATE  # FV lower boundary 9 cm above cathode
R["drift"] = {"T_max_us": T_MAX, "T_event_us": T_EV, "v_drift_mm_us": V_D, "T_FV_range_us": [T_FV_LO, T_FV_HI],
              "P_t_ge_event_uniform_0_Tmax": (T_MAX - T_EV) / (T_MAX - T_GATE),
              "P_t_ge_event_uniform_FV": (T_FV_HI - T_EV) / (T_FV_HI - T_FV_LO),
              "P_t_ge_870_if_Tmax_951": (951 - 870) / 951.0,
              "frac_FV_in_drift": (T_FV_HI - T_FV_LO) / (T_MAX - T_GATE)}
# S2-width arithmetic: sigma_t = sqrt(2 D_L t)/v_d ; D_L recalled (likely) 25 cm^2/s at ~100 V/cm (LUX/LZ-like), +-40 %
D_L = 25.0
def sig_t_us(t_us, DL=D_L):
    return np.sqrt(2 * DL * t_us * 1e-6) * 10 / V_D    # cm -> mm -> us
NE_EVENT = lz.LZ["ev_S2c"] / lz.LZ["g2"]
R["s2_width"] = {
    "Ne_event": NE_EVENT,
    "sigma_t_us_at_event_drift": float(sig_t_us(T_EV)), "sigma_t_us_at_cathode": float(sig_t_us(T_MAX)),
    "ratio_cathode_over_event": float(np.sqrt(T_MAX / T_EV)),
    "stat_rel_err_width_from_Ne": float(1 / np.sqrt(2 * NE_EVENT)),
    "sigma_t_us_zero_drift": 0.0,
    "n_sigma_cathode_vs_event_width": float((np.sqrt(T_MAX / T_EV) - 1) * np.sqrt(2 * NE_EVENT)),
    # fraction of a uniform-in-apparent-drift population for which a cathode-emission S2 is within +-10 %/20 % in width
    "frac_drift_range_width_within_10pct_of_cathode": float(1 - (T_MAX * 0.9 ** 2) / T_MAX),
    "frac_drift_range_width_within_20pct_of_cathode": float(1 - 0.8 ** 2),
    "t_min_us_within_10pct": float(T_MAX * 0.81), "t_min_us_within_20pct": float(T_MAX * 0.64),
}

# ======================================================================================
# 4. Factorisation and rates
# ======================================================================================
prod_tot = N_ACC_EXP / (T_MAX * 1e-6 * T_LIVE_S)           # R_S1,tot R_S2,tot f_cuts  [Hz^2]
f_S1_hi = b["frac_S1_ge495"]
f_S2_box = b["frac_col_pm2"]                                # from the 2D map column at S1c 500-600
f_S2_39_40 = b["frac_col_39_40"]
R["factorisation"] = {
    "T_live_s": T_LIVE_S, "T_max_s": T_MAX * 1e-6,
    "R1R2_fcuts_total_Hz2": prod_tot,
    "R1R2_fcuts_event_box_Hz2": prod_tot * f_S1_hi * f_S2_box,
    "R1R2_fcuts_500_600_x_39_40_Hz2": prod_tot * f_S1_hi * f_S2_39_40,
    "f_S1_500_600": f_S1_hi, "f_S2_pm2_at_highS1": f_S2_box, "f_S2_39_40_at_highS1": f_S2_39_40,
    # pre-cut UDT sample (S3a) as a proxy for the pre-cut PDT population if the UDT window ~ T_max (assumption)
    "N_udt_precut_data": a["top_data_sum"], "N_udt_precut_model": a["top_model_sum"],
    "R1R2_precut_Hz2_if_UDTwindow_eq_Tmax": a["top_data_sum"] / (T_MAX * 1e-6 * T_LIVE_S),
    "f_cuts_incl_FV_if_UDTwindow_eq_Tmax": N_ACC_EXP / a["top_data_sum"],
}
# Pre-cut products (assumption A: UDT window length = T_max, so the 537 UDT events are also the pre-cut PDT count):
#   R_S1,tot x R_S2,tot = N_precut / (T_max T_live)  [both rates restricted to ROI pulse sizes + minimal quality cuts]
#   R_S1(500-600) x R_S2,tot = that x f_S1,precut(>=495)   (model 2.6 %, data 4.1 %)
prod_pre = R["factorisation"]["R1R2_precut_Hz2_if_UDTwindow_eq_Tmax"]
R["factorisation"]["R_S1_500_600_x_R_S2tot_Hz2_precut_model"] = prod_pre * a["frac_S1_ge495"]
R["factorisation"]["R_S1_500_600_x_R_S2tot_Hz2_precut_data"] = prod_pre * a["frac_S1_ge495_data"]
# isolated-S1 rate at 500-600 phd implied for a bracket of total isolated-S2 rates (>= 645 phd; the absolute
# isolated-S2 rate is not given in the paper; bracket spans the plausible range, recalled/uncertain)
R_S2_BRACKET = {"1 mHz": 1e-3, "10 mHz": 1e-2, "100 mHz": 1e-1, "1 Hz": 1.0}
impl = {}
for lab, r2 in R_S2_BRACKET.items():
    r1 = R["factorisation"]["R_S1_500_600_x_R_S2tot_Hz2_precut_model"] / r2
    impl[lab] = {"R_S2_tot_Hz": r2, "R_S2_in_pm2_box_Hz_postcut_shape": r2 * f_S2_box,
                 "R_S1_500_600_Hz": r1, "R_S1_500_600_per_day": r1 * 86400,
                 "R_S1_tot_Hz": prod_pre / r2}
R["factorisation"]["implied_R_S1_500_600_precut"] = impl

# ======================================================================================
# 5. Physical plausibility: isolated S1 of ~540 phd; isolated S2 of ~270 e
# ======================================================================================
R_TPC_CM = 72.8          # recalled (likely): LZ active radius (145.6 cm diameter)
RHO_LXE = 2.86           # g/cm^3 recalled (likely) at LZ operating temperature
RFR_DEPTH_CM = 13.75     # paper
M_ACTIVE_T = 7.0         # recalled (certain): LZ 7-tonne active mass
m_rfr = np.pi * R_TPC_CM ** 2 * RFR_DEPTH_CM * RHO_LXE / 1e6
R["rfr"] = {"mass_t": m_rfr, "frac_of_active": m_rfr / M_ACTIVE_T, "frac_of_fiducial": m_rfr / lz.LZ["fiducial_mass_t"]}
# ER rate density from the science sample (P010: 1655 continuum events / 19.4 keV acceptance = 85 per keVee in 2.84 t yr)
ER_PER_KEV_2p84 = 1655 / 19.4
R_ER_per_t_d_keV = ER_PER_KEV_2p84 / (lz.LZ["exposure_tyr"] * 365.25)
R["er_rate"] = {"events_per_keVee_in_2p84tyr": ER_PER_KEV_2p84, "per_t_d_keV": R_ER_per_t_d_keV,
                "per_kg_d_keV": R_ER_per_t_d_keV / 1e3, "naive_1710_over_220d_4p71t": 1710 / (220 * 4.71)}
# ER energy giving S1 in 500-600 phd: NEST-LZ beta yields (mean), g1 = 0.110, local light-collection factor f_L
try:
    Egrid = np.arange(20, 260, 2.0)
    nph = np.array([lz.nest_er_yields(E)[0] for E in Egrid])
    ok_nest = True
except Exception as e:  # pragma: no cover
    ok_nest = False
    R["er_rate"]["nest_error"] = str(e)
if ok_nest:
    def E_for_S1(S1, fL):
        return float(np.interp(S1 / (lz.LZ["g1"] * fL), nph, Egrid))
    windows = {}
    for fL in (1.0, 1.3, 0.8):
        e_lo, e_hi = E_for_S1(500, fL), E_for_S1(600, fL)
        windows[f"fL={fL}"] = {"E_lo_keV": e_lo, "E_hi_keV": e_hi, "dE_keV": e_hi - e_lo, "E_540_keV": E_for_S1(540.1, fL)}
    R["er_rate"]["ER_energy_for_S1_500_600"] = windows
    R["er_rate"]["Nph_per_keV_at_70_100_keV"] = [float(np.interp(70, Egrid, nph) / 70), float(np.interp(100, Egrid, nph) / 100)]
    dE = windows["fL=1.0"]["dE_keV"]
    # rate of ER deposits in the RFR with S1 in 500-600 phd: internal betas (uniform in LXe) x detector-gamma enhancement
    rate_rfr_beta = R_ER_per_t_d_keV * m_rfr * dE           # per day
    R["rfr"]["rate_S1_500_600_per_day_internal_beta"] = rate_rfr_beta
    R["rfr"]["rate_S1_500_600_Hz_internal_beta"] = rate_rfr_beta / 86400
    for enh in (10, 100, 1000):   # detector-gamma enhancement near the bottom PMT array (recalled, uncertain)
        R["rfr"][f"rate_S1_500_600_per_day_x{enh}"] = rate_rfr_beta * enh
    R["rfr"]["dE_used_keV"] = dE
    # enhancement over the internal-beta RFR rate needed to supply the model's isolated-S1 rate at 500-600 phd
    R["rfr"]["enhancement_needed_for_R_S2_10mHz"] = impl["10 mHz"]["R_S1_500_600_per_day"] / rate_rfr_beta
    R["rfr"]["enhancement_needed_for_R_S2_100mHz"] = impl["100 mHz"]["R_S1_500_600_per_day"] / rate_rfr_beta
    R["rfr"]["enhancement_needed_for_R_S2_1Hz"] = impl["1 Hz"]["R_S1_500_600_per_day"] / rate_rfr_beta
    # S1 that must accompany a real 270-electron cluster in the bulk (ER and NR hypotheses)
    Eg2 = np.arange(1, 400, 0.5)
    ne_er = np.array([lz.nest_er_yields(E)[1] for E in Eg2]); nph_er = np.array([lz.nest_er_yields(E)[0] for E in Eg2])
    E_er = float(np.interp(NE_EVENT, ne_er, Eg2)); S1_er = float(np.interp(E_er, Eg2, nph_er) * lz.LZ["g1"])
    ne_nr = np.array([lz.nest_nr_yields(E)[1] for E in Eg2]); nph_nr = np.array([lz.nest_nr_yields(E)[0] for E in Eg2])
    E_nr = float(np.interp(NE_EVENT, ne_nr, Eg2)); S1_nr = float(np.interp(E_nr, Eg2, nph_nr) * lz.LZ["g1"])
    R["s2_parent"] = {"Ne": NE_EVENT, "ER_energy_keV_for_Ne": E_er, "ER_expected_S1c_phd": S1_er,
                      "NR_energy_keV_for_Ne": E_nr, "NR_expected_S1c_phd": S1_nr, "S1_threshold_phd": lz.LZ["S1c_min"]}

# ======================================================================================
# 6. Mismodelling factor and probabilities
# ======================================================================================
def k_for(P, mu):
    return -np.log(1 - P) / mu
R["mismodelling"] = {
    "N_nb_model": N_NB, "N_panel_model": N_PANEL, "N_roi_model": N_ACC_FIT,
    "P_ge1_nb_k1": 1 - np.exp(-N_NB), "P_ge1_panel_k1": 1 - np.exp(-N_PANEL),
    "k_req_nb_10pct": k_for(0.10, N_NB), "k_req_nb_50pct": k_for(0.50, N_NB),
    "k_req_panel_10pct": k_for(0.10, N_PANEL), "k_req_roi_10pct": k_for(0.10, N_ACC_FIT),
    "k_allowed_postcut_UDT_95": R["figS3"]["udt_postcut_k_UL95"],
    "k_allowed_precut_UDT_95_highS1": R["figS3"]["udt_precut_ge495_k_UL95"],
    "P_ge1_nb_at_k_postcut95": 1 - np.exp(-N_NB * R["figS3"]["udt_postcut_k_UL95"]),
    "P_ge1_nb_at_k_precut95": 1 - np.exp(-N_NB * R["figS3"]["udt_precut_ge495_k_UL95"]),
    "ratio_kreq10_over_kallowed_postcut": k_for(0.10, N_NB) / R["figS3"]["udt_postcut_k_UL95"],
    # "shape" mismodelling: if the S2c distribution at high S1 were flat in log S2c instead of the map gradient
    "f_S2_pm2_if_flat_in_logS2c": (4 * SIG_NR_DEX) / (lz.LZ["log10S2c_max"] - lz.LZ["log10S2c_min"]),
    "N_nb_if_flat_in_logS2c": N_ACC_FIT * f_S1_hi * (4 * SIG_NR_DEX) / (lz.LZ["log10S2c_max"] - lz.LZ["log10S2c_min"]),
    # Poisson probability that a k=100 mismodelling at high S1 would still give 0 post-cut UDT events
    "P0_postcut_UDT_highS1_k100": float(np.exp(-100 * m_hi)),
}
MM = R["mismodelling"]
# uncertainty on N_nb from digitisation (factor 1.3) and from the model's own +-22 % (0.6/2.7)
MM["N_nb_range"] = [N_NB / 1.3 * (1 - N_ACC_ERR / N_ACC_EXP), N_NB * 1.3 * (1 + N_ACC_ERR / N_ACC_EXP)]
MM["k_req_nb_10pct_range"] = [k_for(0.10, MM["N_nb_range"][1]), k_for(0.10, MM["N_nb_range"][0])]
# what the post-cut UDT sample (0 events at S1c>495, expectation m_hi) says about the k required:
for lab, k in (("k_req_nb_10pct", MM["k_req_nb_10pct"]), ("k_req_panel_10pct", MM["k_req_panel_10pct"]),
               ("k_req_nb_10pct_lo", MM["k_req_nb_10pct_range"][0]), ("k_req_nb_10pct_hi", MM["k_req_nb_10pct_range"][1])):
    p0 = float(np.exp(-k * m_hi))
    MM[f"postcut_UDT_expected_at_{lab}"] = k * m_hi
    MM[f"P0_postcut_UDT_at_{lab}"] = p0
    MM[f"Z_postcut_UDT_at_{lab}"] = float(stats.norm.isf(p0))
# shape-only mismodelling: the largest plausible S2c-shape change at high S1 (flat in log S2c) gains
f_gain_flat = MM["f_S2_pm2_if_flat_in_logS2c"] / f_S2_box
MM["shape_gain_flat_over_map"] = f_gain_flat
MM["k_norm_needed_with_flat_shape"] = MM["k_req_nb_10pct"] / f_gain_flat
MM["P0_postcut_UDT_with_flat_shape"] = float(np.exp(-MM["k_norm_needed_with_flat_shape"] * m_hi))
# the extreme: every high-S1 isolated S2 inside the +-2 sigma window (unphysical upper bound on the shape gain)
MM["shape_gain_max"] = 1 / f_S2_box
MM["k_norm_needed_with_max_shape"] = MM["k_req_nb_10pct"] * f_S2_box
MM["P0_postcut_UDT_with_max_shape"] = float(np.exp(-MM["k_norm_needed_with_max_shape"] * m_hi))
# pre-cut high-S1 excess (22 vs 14.6) taken at face value as a normalisation factor
MM["N_nb_if_precut_excess_real"] = N_NB * a["data_over_model_ge495"]

# ======================================================================================
# Figures
# ======================================================================================
fig, axes = plt.subplots(3, 1, figsize=(7.5, 10), sharex=True)
for ax, (pname, d) in zip(axes, fig5.items()):
    acc = d["Accidentals"]; accf = d["Accidentals_filled"]; tot = d["Total"]
    ax.step(EDGES[:-1], np.where(tot > 0, tot, np.nan), where="post", color="tab:blue", lw=1.2, label="total model (digitised)")
    ax.step(EDGES[:-1], accf, where="post", color="goldenrod", lw=2, label="accidentals (digitised)")
    hid = np.isnan(acc)
    if hid.any():
        ax.plot(CENTRES[hid], accf[hid], "x", color="red", label="hidden bin, log-interpolated")
    ax.axvspan(-2, 2, color="0.9", zorder=0); ax.axvline(EV_SIG_NR, color="k", ls=":")
    ax.set_yscale("log"); ax.set_ylabel("events / 0.5σ bin")
    ax.set_title(f"{pname}: accidentals Σ(±8σ) = {R['fig5'][pname]['acc_sum_pm8_filled']:.3g}, Σ(±2σ) = {R['fig5'][pname]['acc_sum_pm2']:.3g}", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
axes[-1].set_xlabel(r"$(\log_{10}S2c-\mu_{NR})/\sigma_{NR}$")
axes[0].text(0.98, 0.05, f"three-panel accidental total {tot_acc_fig5:.2f} vs Table I fit 2.6", transform=axes[0].transAxes, ha="right", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "P022_fig5_accidentals_digitised.png"), dpi=150); plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for k, c in (("a", "tab:gray"), ("b", "tab:blue")):
    res = S3R[k]
    axes[0].step(S1_EDGES[:-1], res["top_model"], where="post", color=c, label=f"model S3{k} ({'normalisation stage' if k=='a' else 'after all cuts'})")
    nz = res["top_data"] > 0
    axes[0].plot(S1_EDGES[:-1][nz] + S1_BIN / 2, res["top_data"][nz], "o", color=c, ms=3, label=f"UDT data S3{k}")
axes[0].axvspan(500, 600, color="0.9", zorder=0); axes[0].set_yscale("log"); axes[0].set_xlabel("S1c [phd]"); axes[0].set_ylabel("counts / 15 phd (UDT normalisation)")
axes[0].set_title(f"isolated-S1 spectrum: f(S1c≥495) = {a['frac_S1_ge495']:.3f} (pre-cut), {b['frac_S1_ge495']:.3f} (post-cut)", fontsize=9)
axes[0].legend(fontsize=7)
for k, c in (("a", "tab:gray"), ("b", "tab:blue")):
    res = S3R[k]
    rm = res["right_model"]; norm = np.nansum(rm)
    axes[1].step(S2_EDGES[:-1], rm / norm, where="post", color=c, label=f"projection S3{k} (all S1c)")
res = S3R["b"]; dens = res["col_dens"]; good = ~np.isnan(dens)
# normalise the column shape to unit integral per 0.15 dex for comparison
norm_col = np.nansum(dens) / PX_PER_DEX / S2_BIN
axes[1].plot(res["col_logs2"][good], dens[good] / norm_col, color="tab:red", lw=1.5, label="2D map, S1c 500–600 phd column (S3b)")
axes[1].axvspan(MU_NR - 2 * SIG_NR_DEX, MU_NR + 2 * SIG_NR_DEX, color="0.85", zorder=0, label="±2σ_NR at S1c=540")
axes[1].axvline(EV_LOGS2C, color="k", ls=":")
axes[1].set_yscale("log"); axes[1].set_xlabel(r"$\log_{10}$S2c"); axes[1].set_ylabel("fraction per 0.15 dex")
axes[1].set_title(f"isolated-S2 shape: f(±2σ | S1c 500–600) = {b['frac_col_pm2']:.3f}; gradient 4/3 = {b['col_dens_ratio_logS2c_4_over_3']:.2f}", fontsize=9)
axes[1].legend(fontsize=7, loc="lower left")
kk = np.logspace(-0.5, 4, 400)
axes[2].plot(kk, 1 - np.exp(-N_NB * kk), color="tab:red", label=f"P(≥1 acc. in ±2σ nb), N={N_NB:.2e}·k")
axes[2].fill_between(kk, 1 - np.exp(-R["mismodelling"]["N_nb_range"][0] * kk), 1 - np.exp(-R["mismodelling"]["N_nb_range"][1] * kk), color="tab:red", alpha=0.2)
axes[2].plot(kk, 1 - np.exp(-N_PANEL * kk), color="goldenrod", label=f"anywhere S1c>500 (±8σ), N={N_PANEL:.2e}·k")
axes[2].axvspan(kk[0], R["figS3"]["udt_postcut_k_UL95"], color="green", alpha=0.15, label=f"allowed by post-cut UDT (0 obs at S1c>500): k<{R['figS3']['udt_postcut_k_UL95']:.0f}")
axes[2].axvspan(kk[0], R["figS3"]["udt_precut_ge495_k_UL95"], color="blue", alpha=0.15, label=f"allowed by pre-cut UDT (22 obs): k<{R['figS3']['udt_precut_ge495_k_UL95']:.2f}")
axes[2].axhline(0.1, color="0.5", lw=0.8); axes[2].axhline(0.5, color="0.5", lw=0.8)
axes[2].set_xscale("log"); axes[2].set_xlabel("accidentals mismodelling factor k at high S1"); axes[2].set_ylabel("P(≥1)")
axes[2].set_title(f"k for 10 % in neighbourhood: {R['mismodelling']['k_req_nb_10pct']:.0f}", fontsize=9)
axes[2].legend(fontsize=7, loc="upper left")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "P022_figS3_rates_and_k.png"), dpi=150); plt.close()

# ======================================================================================
# Save
# ======================================================================================
with open(os.path.join(OUT, "P022_results.json"), "w") as f:
    json.dump(R, f, indent=1, default=float)
with open(os.path.join(OUT, "P022_fig5_accidentals_digitised.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["sigma_low", "sigma_high"] + [f"{p}_{c}" for p in PANELS for c in ("acc", "acc_filled", "total")])
    for i in range(len(CENTRES)):
        w.writerow([EDGES[i], EDGES[i + 1]] + [f"{fig5[p][c][i]:.3e}" for p in PANELS for c in ("Accidentals", "Accidentals_filled", "Total")])
with open(os.path.join(OUT, "P022_figS3_projections.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["S1c_low", "S1c_high", "S3a_model", "S3a_data", "S3b_model", "S3b_data"])
    for i in range(len(S1_EDGES) - 1):
        w.writerow([S1_EDGES[i], S1_EDGES[i + 1], f"{S3R['a']['top_model'][i]:.4g}", int(S3R['a']['top_data'][i]), f"{S3R['b']['top_model'][i]:.4g}", int(S3R['b']['top_data'][i])])
    w.writerow([]); w.writerow(["logS2c_low", "logS2c_high", "S3a_model", "S3a_data", "S3b_model", "S3b_data"])
    for i in range(len(S2_EDGES) - 1):
        w.writerow([S2_EDGES[i], S2_EDGES[i + 1], f"{S3R['a']['right_model'][i]:.4g}", int(S3R['a']['right_data'][i]), f"{S3R['b']['right_model'][i]:.4g}", int(S3R['b']['right_data'][i])])
with open(os.path.join(OUT, "P022_figS3b_column_500_600.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["logS2c", "counts_per_2Dbin_UDTnorm"])
    for l, d in zip(S3R["b"]["col_logs2"], S3R["b"]["col_dens"]):
        if not np.isnan(d):
            w.writerow([f"{l:.4f}", f"{d:.4e}"])
print(json.dumps(R, indent=1, default=float))
