"""
P019 -- Atmospheric-neutrino processes at 248 keV: the CEvNS tail, loss of coherence,
and exotic neutrino interactions.

Run from the simulation root:   .venv/bin/python output/code/P019_atm_nu.py
Outputs -> output/work/P019/  (JSON, CSV, figures/*.png)

Sections
  1. Atmospheric-neutrino flux model (recalled, parametrised, bracketed) and CEvNS spectrum
     (Helm form factor; WimPyDD shell-model weak form factor as a second model), normalised so
     that the efficiency-weighted 5.4-270 keV rate in 2.84 t yr equals LZ Table I (0.11 events).
  2. Tail fractions above 100/150/200/248 keV, events above 200 keV and in 225-271 keV;
     Fig. 5 "Neutrino + Detector NRs" curve digitised in all three panels and compared.
  3. Coherence loss: F^2 at q = 246 MeV, node positions per isotope.
  4. Incoherent NC quasi-elastic knock-out (Llewellyn-Smith free-nucleon NC elastic, Pauli
     blocking), residual-nucleus Fermi-recoil channel, NC excitation of 129Xe/131Xe low-lying levels.
  5. Exotic: nuclear magnetic-moment scattering, light vector Z', heavy scalar, dipole-portal
     up-scattering to a heavy neutral lepton (cross-section derived in details.md).
  6. Solar 8B/hep kinematic end points.
"""
from __future__ import annotations
import json, math, os, sys
import numpy as np
from scipy import integrate, optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, "output/code")
from common import lzcommon as lz

OUT = "output/work/P019"
FIG = f"{OUT}/figures"
os.makedirs(FIG, exist_ok=True)
R: dict = {}          # results container -> P019_results.json
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.append(s)

# --------------------------------------------------------------------------------------
# 0. Constants (recalled, certain unless flagged)
# --------------------------------------------------------------------------------------
GF = 1.1663788e-5          # GeV^-2
S2W = 0.23867              # low-energy weak mixing angle (recalled, likely; MSbar at q~0)
GEV_TO_CM2 = lz.GEV_TO_CM2
HBARC = lz.HBARC_GEV_FM
ALPHA = 1 / 137.036
M_NUC = 0.938272           # GeV (nucleon)
EXPOSURE = lz.LZ["exposure_tyr"]      # 2.84
N_ATM_ROI = lz.LZ["bkg_expected"]["atm_nu"]       # (0.11, 0.02)
N_B8HEP_ROI = lz.LZ["bkg_expected"]["B8_hep"]     # (0.057, 0.006)
SEC_PER_YR = 3.15576e7
ISO = lz.XE_ISOTOPES       # {A: abundance}
Z_XE = 54
def mN(A): return lz.m_nucleus_gev(A)
def n_per_tonne(A, f): return 1e6 / (A * 1.66054e-24) * f     # nuclei of isotope per tonne natural Xe
def QW(A): return (A - Z_XE) - (1 - 4 * S2W) * Z_XE
def q_mev(E_keV, A): return math.sqrt(2 * mN(A) * E_keV * 1e-6) * 1e3
R["constants"] = dict(GF=GF, s2w=S2W, QW_131=QW(131), q_MeV_at_248keV_A131=q_mev(248, 131),
                      exposure_tyr=EXPOSURE, N_atm_TableI=N_ATM_ROI, N_B8hep_TableI=N_B8HEP_ROI)
log(f"Q_W(131Xe) = {QW(131):.2f};  q(248 keV, A=131) = {q_mev(248,131):.1f} MeV")

# --------------------------------------------------------------------------------------
# 1a. Efficiency (P009 fit to Fig. S2 inset) and S1c(E) scale (P009 'paper-contour' scale)
# --------------------------------------------------------------------------------------
from scipy.stats import norm
def eff(E):
    """NR detection efficiency vs true recoil energy: 0.955 plateau, 50% at 269.9 keV with
    sigma 11.8 keV (P009 Gaussian roll-off) and a low-energy turn-on with 50% at 5.4 keV
    (width 1.0 keV assumed)."""
    E = np.asarray(E, float)
    return 0.955 * norm.cdf((269.9 - E) / 11.8) * norm.cdf((E - 5.4) / 1.0)

_S1_CACHE = {}
def S1c_of_E(E):
    """NR-band median S1c (phd) at true energy E on the paper's own scale: nestpy LZ-tuned
    yields (Table S5 incl. p(E) break) rescaled so that S1c(250 keV) = 543.8 phd (P009 Fig. 4
    contour crossing)."""
    if not _S1_CACHE:
        Eg = np.concatenate([np.arange(1, 20, 1.0), np.arange(20, 320, 5.0)])
        nph = np.array([lz.nest_nr_yields(e)[0] for e in Eg])
        s1 = lz.LZ["g1"] * nph
        k = 543.8 / np.interp(250.0, Eg, s1)
        _S1_CACHE["E"], _S1_CACHE["S1"], _S1_CACHE["k"] = Eg, s1 * k, k
    return np.interp(E, _S1_CACHE["E"], _S1_CACHE["S1"])
S1c_of_E(250.0)
R["S1_scale_factor_vs_TableS5"] = float(_S1_CACHE["k"])
log(f"S1 scale factor (paper contour / nestpy Table S5) = {_S1_CACHE['k']:.3f};  S1c(248) = {S1c_of_E(248):.1f} phd; "
    f"E at S1c=250: {np.interp(250, _S1_CACHE['S1'], _S1_CACHE['E']):.1f} keV; at 500: {np.interp(500, _S1_CACHE['S1'], _S1_CACHE['E']):.1f} keV")
R["E_at_S1c_250_500_600"] = [float(np.interp(s, _S1_CACHE["S1"], _S1_CACHE["E"])) for s in (250, 500, 600)]

# --------------------------------------------------------------------------------------
# 1b. Atmospheric-neutrino flux model (RECALLED, uncertain; shape bracketed, normalisation
#     fixed to LZ's 0.11 events so that only the shape matters)
#     dPhi/dE = C (E/E_b)^-g1 for E_lo < E < E_b ;  C (E/E_b)^-g2 for E > E_b      [cm^-2 s^-1 MeV^-1]
#     Central (assignment): g1 = 1 (flat E dPhi/dE) between 10 and 100 MeV, g2 = 2.5 above.
# --------------------------------------------------------------------------------------
E_LO_MEV = 10.0
PHI_TOTAL_RECALLED = 10.5   # cm^-2 s^-1 above ~10 MeV, all flavours (Billard et al. 2014 Table I; recalled, likely)
def flux_shape(E, g1=1.0, Eb=100.0, g2=2.5):
    E = np.asarray(E, float)
    out = np.where(E < Eb, (E / Eb) ** (-g1), (E / Eb) ** (-g2))
    return np.where(E < E_LO_MEV, 0.0, out)
def flux_norm_C(g1, Eb, g2, total=PHI_TOTAL_RECALLED, Emax=1e4):
    I, _ = integrate.quad(lambda e: flux_shape(e, g1, Eb, g2), E_LO_MEV, Emax, limit=200)
    return total / I
SHAPES = {
    "central (g1=1, Eb=100, g2=2.5)": (1.0, 100.0, 2.5),
    "soft high-E (g1=1, Eb=100, g2=3)": (1.0, 100.0, 3.0),
    "hard high-E (g1=1, Eb=100, g2=2)": (1.0, 100.0, 2.0),
    "Honda-like (g1=1.5, Eb=1000, g2=2.7)": (1.5, 1000.0, 2.7),
    "steep low-E (g1=2, Eb=100, g2=2.5)": (2.0, 100.0, 2.5),
    "low break (g1=1, Eb=50, g2=2.5)": (1.0, 50.0, 2.5),
    "high break (g1=1, Eb=150, g2=2.5)": (1.0, 150.0, 2.5),
    "single power law E^-2 above 10 MeV": (2.0, 10.0001, 2.0),
}

# --------------------------------------------------------------------------------------
# 1c. CEvNS differential cross-section and recoil spectrum
# --------------------------------------------------------------------------------------
def Enu_min_MeV(E_keV, A):
    """Exact kinematic minimum: E_nu >= (E_R + sqrt(E_R^2 + 2 m_N E_R))/2."""
    ER = E_keV * 1e-3; M = mN(A) * 1e3
    return 0.5 * (ER + math.sqrt(ER * ER + 2 * M * ER))

# WimPyDD shell-model weak form factor: ratio of the WimPyDD O1 rate with weak-charge couplings
# (c_p = -(1-4 s2w), c_n = 1) to the Helm Q_W^2 F^2 sum, for a very heavy WIMP (velocity integral
# independent of E), normalised at 1 keV where both form factors -> 1.
_FW = {}
def build_wimpydd_weak_ff():
    cp, cn = -(1 - 4 * S2W), 1.0
    ham = lz.wd_hamiltonian("weak_charge", {1: (cp + cn, cp - cn)})
    halo = lz.wd_halo()
    Eg = np.concatenate([[1.0], np.arange(2, 30, 2.0), np.arange(30, 330, 3.0)])
    rw = lz.wd_rate(ham, 1e6, Eg, halo=halo)
    helm = np.array([sum(f * QW(A) ** 2 * lz.helm_F2(e, A) for A, f in ISO.items()) for e in Eg])
    ratio = rw / helm
    ratio /= ratio[0]
    # effective natural-Xe weak form factor squared (abundance- and Q_W^2-weighted)
    F2_helm_nat = helm / sum(f * QW(A) ** 2 for A, f in ISO.items())
    _FW["E"], _FW["F2"] = Eg, F2_helm_nat * ratio
    _FW["F2_helm_nat"] = F2_helm_nat
    return Eg, F2_helm_nat * ratio, F2_helm_nat
Eg_ff, F2_wd, F2_helm_nat = build_wimpydd_weak_ff()
def F2_natural(E, model):
    if model == "helm":
        return np.array([sum(f * QW(A) ** 2 * lz.helm_F2(e, A) for A, f in ISO.items()) / sum(f * QW(A) ** 2 for A, f in ISO.items())
                         for e in np.atleast_1d(E)])
    return np.interp(np.atleast_1d(E), _FW["E"], _FW["F2"])

def dsig_dER_cm2_per_keV(E_keV, Enu_MeV, A, F2=None):
    """CEvNS: dsigma/dE_R = G_F^2 m_N/(4 pi) Q_W^2 (1 - m_N E_R/(2 E_nu^2)) F^2   [cm^2/keV]."""
    M = mN(A); ER = E_keV * 1e-6; Enu = np.asarray(Enu_MeV, float) * 1e-3
    kin = 1 - M * ER / (2 * Enu ** 2)
    kin = np.where(kin > 0, kin, 0.0)
    f2 = lz.helm_F2(E_keV, A) if F2 is None else F2
    return GF ** 2 * M / (4 * math.pi) * QW(A) ** 2 * kin * f2 * GEV_TO_CM2 * 1e-6

def flux_integral(E_keV, A, shape, power=0, extra=None, Emax=1e4):
    """int_{Enu_min}^inf dE Phi_shape(E) (1 - m_N E_R/(2E^2)) E^power [extra(E)] (unnormalised shape)."""
    g1, Eb, g2 = shape
    M = mN(A) * 1e3; ER = E_keV * 1e-3
    emin = Enu_min_MeV(E_keV, A)
    def f(e):
        kin = 1 - M * ER / (2 * e * e)
        val = flux_shape(e, g1, Eb, g2) * max(kin, 0.0) * e ** power
        return val * (extra(e) if extra else 1.0)
    if emin >= Emax: return 0.0
    pts = [p for p in (Eb, 3 * emin, 10 * emin) if emin < p < Emax]
    I, _ = integrate.quad(f, emin, Emax, points=pts or None, limit=400)
    return I

def cevns_spectrum(E_grid, shape, ffmodel="helm", C=1.0):
    """dR/dE_R in events/(t yr keV) for natural Xe for flux dPhi/dE = C * shape(E)."""
    out = np.zeros_like(E_grid, dtype=float)
    F2nat_wd = F2_natural(E_grid, "wimpydd") if ffmodel == "wimpydd" else None
    for A, f in ISO.items():
        nT = n_per_tonne(A, f)
        for i, E in enumerate(E_grid):
            if ffmodel == "helm":
                f2 = lz.helm_F2(E, A)
            else:
                # apply the natural-Xe shell-model/Helm ratio to each isotope's Helm F^2
                f2 = lz.helm_F2(E, A) * (F2nat_wd[i] / F2_helm_nat_interp(E))
            pref = GF ** 2 * mN(A) / (4 * math.pi) * QW(A) ** 2 * f2 * GEV_TO_CM2 * 1e-6   # cm^2/keV per unit kin
            out[i] += nT * pref * C * flux_integral(E, A, shape) * SEC_PER_YR
    return out
def F2_helm_nat_interp(E):
    return float(np.interp(E, _FW["E"], _FW["F2_helm_nat"]))

E_GRID = np.concatenate([np.arange(0.5, 20, 0.5), np.arange(20, 100, 2.0), np.arange(100, 330, 2.0)])

def normalise_and_summarise(shape_name, shape, ffmodel):
    spec_unit = cevns_spectrum(E_GRID, shape, ffmodel, C=1.0)          # per unit C
    w = eff(E_GRID)
    N_roi_unit = np.trapezoid(spec_unit * w, E_GRID) * EXPOSURE
    C = N_ATM_ROI[0] / N_roi_unit                                          # cm^-2 s^-1 MeV^-1 at E_b
    spec = spec_unit * C
    C_recalled = flux_norm_C(*shape)
    N_pred_recalled = N_roi_unit * C_recalled
    # flux integrals implied
    phi_tot = C * integrate.quad(lambda e: flux_shape(e, *shape), E_LO_MEV, 1e4, limit=200)[0]
    phi_gt110 = C * integrate.quad(lambda e: flux_shape(e, *shape), 110, 1e4, limit=200)[0]
    def N_between(lo, hi, weighted=True):
        m = (E_GRID >= lo) & (E_GRID <= hi)
        Eg = np.concatenate([[lo], E_GRID[m], [hi]])
        sp = np.interp(Eg, E_GRID, spec) * (eff(Eg) if weighted else 1.0)
        return np.trapezoid(sp, Eg) * EXPOSURE
    tot_raw = N_between(0.5, 329, False)
    res = dict(shape=shape_name, ffmodel=ffmodel, C_norm=C, C_recalled=C_recalled,
               N_pred_with_recalled_flux=N_pred_recalled, phi_total_implied=phi_tot, phi_above_110MeV_implied=phi_gt110,
               N_roi_eff=N_between(0.5, 329, True), N_total_raw=tot_raw)
    for thr in (50, 100, 150, 200, 248):
        res[f"frac_above_{thr}_raw"] = N_between(thr, 329, False) / tot_raw
        res[f"N_above_{thr}_eff"] = N_between(thr, 329, True)
    res["N_225_271_eff"] = N_between(225, 271, True)
    res["N_200_270_eff"] = N_between(200, 270, True)
    res["N_lo_5p4_200_eff"] = N_between(5.4, 200, True)
    res["N_lo_per_hi"] = res["N_lo_5p4_200_eff"] / res["N_200_270_eff"]
    res["dRdE_at_248_per_tyr_keV"] = float(np.interp(248, E_GRID, spec))
    res["dRdE_at_20_per_tyr_keV"] = float(np.interp(20, E_GRID, spec))
    return res, spec

summary = []
spectra = {}
for name, shp in SHAPES.items():
    for ff in ("helm", "wimpydd"):
        res, spec = normalise_and_summarise(name, shp, ff)
        summary.append(res); spectra[(name, ff)] = spec
        log(f"[{ff:7s}] {name:42s} N>200={res['N_above_200_eff']:.2e}  N(225-271)={res['N_225_271_eff']:.2e}  "
            f"frac>200(raw)={res['frac_above_200_raw']:.2e}  N_lo/hi={res['N_lo_per_hi']:.0f}  Npred(recalled flux)={res['N_pred_with_recalled_flux']:.3f}")
import pandas as pd
pd.DataFrame(summary).to_csv(f"{OUT}/cevns_tail_summary.csv", index=False)
R["cevns_summary"] = summary
central_helm = [s for s in summary if s["shape"].startswith("central") and s["ffmodel"] == "helm"][0]
central_wd = [s for s in summary if s["shape"].startswith("central") and s["ffmodel"] == "wimpydd"][0]
# range over all shapes for the headline
for key in ("N_above_200_eff", "N_225_271_eff", "frac_above_200_raw", "N_lo_per_hi"):
    vals = [s[key] for s in summary]
    R[f"range_{key}"] = [min(vals), max(vals)]
pd.DataFrame({"E_keV": E_GRID, **{f"{n}|{ff}": s for (n, ff), s in spectra.items()}}).to_csv(f"{OUT}/cevns_spectra.csv", index=False)

# --------------------------------------------------------------------------------------
# 2. Fig. 5 green curve ("Neutrino + Detector NRs") digitised in the three panels
# --------------------------------------------------------------------------------------
def digitise_fig5_all(png="inputs/figures_png/Fig5_NR_distance_3panels_no_sig.png"):
    im = np.asarray(Image.open(png).convert("RGB")).astype(int)
    H, W, _ = im.shape
    x_l, x_r = 219, 2409                       # panel frame columns (P004; ticks -6..+6 at 492..2135)
    # legend swatch colour for the green line: legend row ~277, columns 262-447 (from the PNG layout)
    sw = im[265:292, 262:447].reshape(-1, 3)
    sw = sw[(sw.sum(axis=1) < 700)]
    green = np.median(sw, axis=0)
    # panel frames: rows where >90% of pixels between x_l and x_r are dark
    dark_rows = np.where((im[:, x_l:x_r].sum(axis=2) < 200).mean(axis=1) > 0.9)[0]
    groups = np.split(dark_rows, np.where(np.diff(dark_rows) > 3)[0] + 1)
    frame_rows = [int(np.median(g)) for g in groups]
    # panels are consecutive (top,bottom) pairs; middle/bottom share the frame line
    panels = {"top": (frame_rows[0], frame_rows[1]), "mid": (frame_rows[1], frame_rows[2]), "bot": (frame_rows[2], frame_rows[3])}
    bottoms_log = {"top": -3.0, "mid": -5.0, "bot": -6.0}        # axis labels at the panel bottoms (read from the figure)
    label_step = {"top": 1, "mid": 1, "bot": 2}                   # decades between labelled major ticks (read from the figure;
                                                                  # bottom panel labels 10^2,10^0,10^-2,10^-4,10^-6 -> P004: 96.75 px/decade)
    out = {}
    edges = np.arange(-8, 8.01, 0.5)
    blue = np.array([0, 0, 255])
    for pname, (yt, yb) in panels.items():
        # major y ticks: dark segments just right of the left frame that are longer than minor ticks
        seg = im[yt + 2:yb - 1, x_l + 2:x_l + 40].sum(axis=2) < 200
        lengths = seg.sum(axis=1)
        rows = np.where(lengths > 0)[0]
        grp = np.split(rows, np.where(np.diff(rows) > 2)[0] + 1)
        ticks = [(int(np.median(g)) + yt + 2, lengths[g].max()) for g in grp if len(g)]
        Lmax = max(l for _, l in ticks)
        major = sorted([y for y, l in ticks if l >= 0.7 * Lmax])
        px_per_decade = np.median(np.diff(major)) / label_step[pname]
        y0 = yb  # bottom frame = bottoms_log
        def y_to_log(y): return bottoms_log[pname] + (y0 - y) / px_per_decade
        sub = im[yt + 3:yb - 2, x_l + 3:x_r - 2]
        def read_curve(col, tol):
            mask = (np.abs(sub - col).sum(axis=2) < tol)
            vals = np.full(len(edges) - 1, np.nan)
            for i in range(len(edges) - 1):
                xa = int(x_l + (edges[i] + 8 + 0.1) / 16 * (x_r - x_l)); xb = int(x_l + (edges[i] + 8 + 0.4) / 16 * (x_r - x_l))
                ys = []
                for x in range(xa, xb):
                    yy = np.where(mask[:, x - (x_l + 3)])[0]
                    if len(yy): ys.append(yy.min() + yt + 3)
                if ys: vals[i] = 10 ** y_to_log(np.median(ys))
            return vals
        vals = read_curve(green, 70)
        tot = read_curve(blue, 60)
        out[pname] = dict(vals=vals, total=tot, frame=(yt, yb), px_per_decade=float(px_per_decade), n_major=len(major),
                          top_log=float(y_to_log(yt)))
    return edges, out, green.tolist(), frame_rows
edges5, green5, green_rgb, frames = digitise_fig5_all()
centres5 = 0.5 * (edges5[1:] + edges5[:-1])
fig5 = {}
for p, d in green5.items():
    v = np.nan_to_num(d["vals"]); t = np.nan_to_num(d["total"])
    fig5[p] = dict(total=float(v.sum()), in_pm2sigma=float(v[(centres5 > -2) & (centres5 < 2)].sum()),
                   at_event_bin=float(v[(centres5 > -2) & (centres5 < -1.5 + 1e-9)].sum()),
                   bkg_total_curve_sum=float(t.sum()), px_per_decade=d["px_per_decade"], frame=d["frame"], n_major=d["n_major"], top_log=d["top_log"])
    log(f"Fig.5 {p}: green sum = {v.sum():.3e}  (|sigma|<2: {fig5[p]['in_pm2sigma']:.3e}); blue total sum = {t.sum():.4g}; "
        f"frame {d['frame']}, {d['px_per_decade']:.1f} px/decade, {d['n_major']} major ticks, frame top = 10^{d['top_log']:.2f}")
# validation targets: top total ~ 1713 - (mid+bot) (Table I fit), bottom total 0.0106 (caption); P004 got 0.01053 in the bottom panel
R["fig5_green"] = dict(rgb=green_rgb, frames=frames, panels=fig5,
                       bins=dict(sigma_low=edges5[:-1].tolist(), **{p: np.nan_to_num(d["vals"]).tolist() for p, d in green5.items()}))
pd.DataFrame({"sigma_low": edges5[:-1], "sigma_high": edges5[1:], **{f"green_{p}": np.nan_to_num(d["vals"]) for p, d in green5.items()}}).to_csv(f"{OUT}/fig5_green_digitised.csv", index=False)
# validation: bottom-panel total from P004 digitisation reproduces the caption (0.01053 vs 0.0106) with the same axis calibration.

# Predicted split of the atm-nu CEvNS rate into the three S1c panels (S1 resolution 5%, P009)
def panel_split(spec, sigma_rel=0.05, extra_events=0.0):
    E = E_GRID; w = spec * eff(E) * EXPOSURE
    s1 = S1c_of_E(E)
    res = {}
    for pname, (lo, hi) in {"top": (3, 250), "mid": (250, 500), "bot": (500, 600)}.items():
        p = norm.cdf((hi - s1) / (sigma_rel * s1)) - norm.cdf((lo - s1) / (sigma_rel * s1))
        res[pname] = float(np.trapezoid(w * p, E))
    return res
split = {}
for name, shp in SHAPES.items():
    for ff in ("helm", "wimpydd"):
        split[f"{name}|{ff}"] = panel_split(spectra[(name, ff)])
R["panel_split_prediction"] = split
R["panel_split_central_helm"] = split[f"{list(SHAPES)[0]}|helm"]
R["panel_split_central_wd"] = split[f"{list(SHAPES)[0]}|wimpydd"]
log("Panel split (central, helm):", split[f"{list(SHAPES)[0]}|helm"])
log("Panel split (central, wimpydd):", split[f"{list(SHAPES)[0]}|wimpydd"])
# empirical shape test: ratio mid/top of the green curve vs prediction
ratio_obs = fig5["mid"]["total"] / max(fig5["top"]["total"], 1e-30)
R["fig5_mid_over_top_observed"] = ratio_obs
R["fig5_mid_over_top_predicted"] = {k: v["mid"] / v["top"] for k, v in split.items()}
# top panel also contains 8B+hep (0.057) and any detector-NR fit component; quote both
R["fig5_mid_over_top_observed_corrected_for_B8"] = fig5["mid"]["total"] / max(fig5["top"]["total"] - N_B8HEP_ROI[0], 1e-30)
rows = []
for k, v in split.items():
    rows.append(dict(model=k, pred_top=v["top"], pred_mid=v["mid"], pred_bot=v["bot"], obs_top_green=fig5["top"]["total"],
                     obs_top_minus_B8=fig5["top"]["total"] - N_B8HEP_ROI[0], obs_mid_green=fig5["mid"]["total"], obs_bot_green=fig5["bot"]["total"],
                     mid_ratio_obs_over_pred=fig5["mid"]["total"] / v["mid"], bot_ratio_obs_over_pred=fig5["bot"]["total"] / v["bot"]))
pd.DataFrame(rows).to_csv(f"{OUT}/fig5_panel_comparison.csv", index=False)

# --------------------------------------------------------------------------------------
# 3. Coherence loss: F^2 at the event, node positions
# --------------------------------------------------------------------------------------
coh = {}
for A in ISO:
    f2_248 = lz.helm_F2(248, A)
    # node: first minimum of Helm F^2 above 150 keV
    Es = np.arange(150, 400, 0.25)
    f2s = np.array([lz.helm_F2(e, A) for e in Es])
    i = int(np.argmin(f2s[Es < 350]))
    # first node (~100 keV) too
    Es1 = np.arange(60, 150, 0.25); f2s1 = np.array([lz.helm_F2(e, A) for e in Es1])
    coh[A] = dict(F2_at_248=f2_248, F2_at_20=lz.helm_F2(20, A), q_MeV_248=q_mev(248, A),
                  node2_keV=float(Es[i]), node2_F2=float(f2s[i]), node1_keV=float(Es1[np.argmin(f2s1)]), node1_F2=float(f2s1.min()))
R["coherence_per_isotope"] = coh
F2nat_248_helm = float(F2_natural(248, "helm")[0]); F2nat_248_wd = float(F2_natural(248, "wimpydd")[0])
R["F2_natural_at_248"] = dict(helm=F2nat_248_helm, wimpydd=F2nat_248_wd, helm_at_20=float(F2_natural(20, "helm")[0]),
                              wd_at_20=float(F2_natural(20, "wimpydd")[0]))
Es = np.arange(150, 350, 0.5)
f2n = F2_natural(Es, "helm"); f2w = F2_natural(Es, "wimpydd")
R["node2_natural_keV"] = dict(helm=float(Es[np.argmin(f2n)]), wimpydd=float(Es[np.argmin(f2w)]))
log(f"F^2(248 keV) natural Xe: Helm {F2nat_248_helm:.3e}, WimPyDD {F2nat_248_wd:.3e}; node2 at {R['node2_natural_keV']}")
log("per-isotope 248 keV F^2:", {A: f"{c['F2_at_248']:.2e} (node {c['node2_keV']:.0f} keV)" for A, c in coh.items()})
pd.DataFrame({"E_keV": Eg_ff, "F2_helm_nat": F2_helm_nat, "F2_wimpydd_nat": F2_wd}).to_csv(f"{OUT}/weak_form_factor.csv", index=False)

# --------------------------------------------------------------------------------------
# 4. Incoherent processes
# 4a. Free-nucleon NC elastic (Llewellyn Smith form; recalled, likely) with dipole form factors
# --------------------------------------------------------------------------------------
MA, MV = 1.03, 0.84   # GeV (recalled, likely)
gA = 1.267; mu_p, mu_n = 2.793, -1.913
def ff_nc(Q2, nucleon):
    GD_V = (1 + Q2 / MV ** 2) ** -2; GD_A = (1 + Q2 / MA ** 2) ** -2
    tau = Q2 / (4 * M_NUC ** 2)
    GEp, GMp, GEn, GMn = GD_V, mu_p * GD_V, 0.0, mu_n * GD_V
    F1p = (GEp + tau * GMp) / (1 + tau); F2p = (GMp - GEp) / (1 + tau)
    F1n = (GEn + tau * GMn) / (1 + tau); F2n = (GMn - GEn) / (1 + tau)
    s = +1 if nucleon == "p" else -1
    F1 = s * 0.5 * (F1p - F1n) - 2 * S2W * (F1p if nucleon == "p" else F1n)
    F2 = s * 0.5 * (F2p - F2n) - 2 * S2W * (F2p if nucleon == "p" else F2n)
    FA = s * 0.5 * gA * GD_A
    return F1, F2, FA
def dsig_dQ2_nce(Enu_GeV, Q2, nucleon, antinu=False):
    """Llewellyn Smith: dsigma/dQ^2 = G_F^2 M^2/(8 pi E^2) [A -/+ B (s-u)/M^2 + C (s-u)^2/M^4]."""
    M = M_NUC; F1, F2, FA = ff_nc(Q2, nucleon); tau = Q2 / (4 * M * M)
    A = 4 * tau * ((1 + tau) * FA ** 2 - (1 - tau) * F1 ** 2 + tau * (1 - tau) * F2 ** 2 + 4 * tau * F1 * F2)
    B = 4 * tau * FA * (F1 + F2)
    C = 0.25 * (FA ** 2 + F1 ** 2 + tau * F2 ** 2)
    su = 4 * M * Enu_GeV - Q2
    sign = +1 if not antinu else -1
    return GF ** 2 * M ** 2 / (8 * math.pi * Enu_GeV ** 2) * (A + sign * B * su / M ** 2 + C * su ** 2 / M ** 4) * GEV_TO_CM2
def Q2max(Enu):  # free nucleon at rest
    return 4 * Enu ** 2 * M_NUC / (M_NUC + 2 * Enu)
def sigma_nce(Enu_GeV, nucleon, pauli=True, Q2lo=0.0):
    """Integrated NC elastic cross-section; simple Pauli suppression factor for a Fermi gas
    (kF = 0.26 GeV, recalled/likely): P(Q2) = 1 - ... (Bodek-Ritchie-type estimate, uncertain)."""
    kF = 0.26
    def pauli_f(Q2):
        q = math.sqrt(Q2 + (Q2 / (2 * M_NUC)) ** 2)
        x = q / (2 * kF)
        return 1.0 if x >= 1 else 1.5 * x - 0.5 * x ** 3
    f = lambda Q2: 0.5 * (dsig_dQ2_nce(Enu_GeV, Q2, nucleon, False) + dsig_dQ2_nce(Enu_GeV, Q2, nucleon, True)) * (pauli_f(Q2) if pauli else 1.0)
    I, _ = integrate.quad(f, Q2lo, Q2max(Enu_GeV), limit=200)
    return I
def sigma_cevns(Enu_MeV, A, ffmodel="helm"):
    """Total coherent cross-section on one nucleus at E_nu."""
    ERmax = 2 * Enu_MeV ** 2 / (mN(A) * 1e3 + 2 * Enu_MeV) * 1e3  # keV
    f = lambda E: dsig_dER_cm2_per_keV(E, Enu_MeV, A, F2=(lz.helm_F2(E, A) if ffmodel == "helm" else lz.helm_F2(E, A) * float(F2_natural(E, "wimpydd")[0]) / F2_helm_nat_interp(E)))
    I, _ = integrate.quad(f, 0, ERmax, limit=200)
    return I
inc = []
for Enu in (30, 50, 100, 150, 200, 300, 500, 1000):
    E = Enu * 1e-3
    sp, sn = sigma_nce(E, "p"), sigma_nce(E, "n")
    sp0, sn0 = sigma_nce(E, "p", pauli=False), sigma_nce(E, "n", pauli=False)
    s_inc = Z_XE * sp + 77 * sn
    s_coh = sigma_cevns(Enu, 131)
    inc.append(dict(Enu_MeV=Enu, sig_p=sp, sig_n=sn, sig_p_free=sp0, sig_n_free=sn0, sig_incoh_nucleus=s_inc,
                    sig_coh_131=s_coh, incoh_over_coh=s_inc / s_coh, TN_keV_at_Q2max=Q2max(E) / (2 * M_NUC) * 1e6))
    log(f"E_nu={Enu:5d} MeV: sigma_NCE p={sp:.2e} n={sn:.2e} (free {sn0:.2e}) cm^2; nucleus incoh {s_inc:.2e}, coh {s_coh:.2e}, ratio {s_inc/s_coh:.3f}")
pd.DataFrame(inc).to_csv(f"{OUT}/incoherent_vs_coherent.csv", index=False)
R["incoherent_table"] = inc
# at fixed q = 246 MeV: coherent N^2 F^2 vs incoherent ~ A (Pauli-unblocked at this q)
q246 = q_mev(248, 131) * 1e-3
Q2_246 = q246 ** 2
TN_246 = Q2_246 / (2 * M_NUC) * 1e3   # MeV kinetic energy of the struck nucleon
R["q246"] = dict(Q2_GeV2=Q2_246, T_nucleon_MeV=TN_246, coherent_weight_QW2F2=QW(131) ** 2 * lz.helm_F2(248, 131),
                 incoherent_weight_approx=131 * 0.5 * (1 + 0.0) , comment="incoherent weight ~ sum_i |c_i|^2 ~ A/4 (axial) + N*(1/4)(vector)  order-of-magnitude")
log(f"q=246 MeV: Q^2 = {Q2_246:.4f} GeV^2; struck-nucleon T = {TN_246:.1f} MeV; coherent weight Q_W^2 F^2 = {R['q246']['coherent_weight_QW2F2']:.1f}")

# 4b. Lone-NR channel from quasi-elastic neutron knock-out: residual (A-1) nucleus recoils with the
#     struck neutron's Fermi momentum p:  E_res = p^2/(2 M_{A-1}).  Fermi sphere kF = 260 MeV/c (recalled, likely).
kF_MeV = 260.0
M_res = mN(130) * 1e3   # MeV
E_res_max = kF_MeV ** 2 / (2 * M_res) * 1e3   # keV
def frac_fermi(Elo, Ehi):
    plo = math.sqrt(2 * M_res * Elo * 1e-3); phi = min(math.sqrt(2 * M_res * Ehi * 1e-3), kF_MeV)
    if plo >= kF_MeV: return 0.0
    return (phi ** 3 - plo ** 3) / kF_MeV ** 3
# flux-weighted incoherent rate: R_inc = n_T * sum_E Phi(E) [Z sig_p + N sig_n] (using the normalised central flux)
shape_c = SHAPES[list(SHAPES)[0]]; C_c = central_helm["C_norm"]
def rate_incoherent_per_tyr(nucleon_only=None):
    Eg = np.geomspace(20, 3000, 60)
    sig = np.array([(77 * sigma_nce(e * 1e-3, "n") if nucleon_only in (None, "n") else 0) + (Z_XE * sigma_nce(e * 1e-3, "p") if nucleon_only in (None, "p") else 0) for e in Eg])
    phi = C_c * flux_shape(Eg, *shape_c)
    nT = sum(n_per_tonne(A, f) for A, f in ISO.items())
    return nT * np.trapezoid(sig * phi, Eg) * SEC_PER_YR
R_inc_n = rate_incoherent_per_tyr("n"); R_inc_p = rate_incoherent_per_tyr("p")
P_escape = 0.3     # ~30 MeV neutron, lambda ~ 27 cm in LXe (sigma_tot ~ 2.7 b, recalled/uncertain), path ~30-100 cm -> 0.1-0.35
P_untagged = 0.2   # neutron leaving the TPC is captured in the OD/skin with high probability; LZ tags 92+-4% of (alpha,n)
                   # neutrons that scatter in the TPC (paper l.178) -> untagged 0.08; allow 0.3 for top/bottom escape paths
P_gs = 0.3         # fraction of neutron removals leaving the A-1 nucleus in its ground state (mean-field valence-hole
                   # spectroscopic strength to the g.s.; recalled/uncertain, 0.1-0.5); excited daughters emit MeV gammas (ER)
f_win = frac_fermi(225, 271)
N_lone = R_inc_n * EXPOSURE * P_escape * P_untagged * P_gs * f_win
N_lone_hi = R_inc_n * EXPOSURE * 0.35 * 0.3 * 0.5 * f_win
N_lone_lo = R_inc_n * EXPOSURE * 0.1 * 0.08 * 0.1 * f_win
R["knockout"] = dict(kF_MeV=kF_MeV, E_res_max_keV=E_res_max, frac_fermi_225_271=f_win, frac_fermi_above_200=frac_fermi(200, 1e4),
                     R_inc_n_per_tyr=R_inc_n, R_inc_p_per_tyr=R_inc_p, N_inc_n_exposure=R_inc_n * EXPOSURE, N_inc_p_exposure=R_inc_p * EXPOSURE,
                     P_escape=P_escape, P_untagged=P_untagged, P_gs=P_gs, N_lone_NR_225_271=N_lone, N_lone_range=[N_lone_lo, N_lone_hi],
                     N_coherent_225_271_central=central_helm["N_225_271_eff"], ratio_lone_over_coherent_window=N_lone / central_helm["N_225_271_eff"])
log(f"Incoherent NC rates: n-knockout {R_inc_n:.2e}/t yr, p-knockout {R_inc_p:.2e}/t yr -> {R_inc_n*EXPOSURE:.2e}, {R_inc_p*EXPOSURE:.2e} in exposure")
log(f"Residual-nucleus recoil: E_max = {E_res_max:.0f} keV; fraction in 225-271: {f_win:.3f}; lone-NR events {N_lone:.2e} [{N_lone_lo:.1e}, {N_lone_hi:.1e}] (vs coherent window {central_helm['N_225_271_eff']:.2e})")

# 4c. NC excitation of low-lying levels (129Xe 39.58 keV 3/2+, 131Xe 80.19 keV 1/2+ ; recalled, likely)
#     sigma ~ (G_F^2 E_nu^2 / pi) (g_A^2/4) B(sigma) with B(sigma) ~ 0.05-0.2 (recalled, uncertain, allowed-approx.)
def sigma_excite(Enu_MeV, B_sigma=0.1):
    E = Enu_MeV * 1e-3
    return GF ** 2 * E ** 2 / math.pi * (gA ** 2 / 4) * B_sigma * GEV_TO_CM2
Eg = np.geomspace(10, 3000, 80)
phi = C_c * flux_shape(Eg, *shape_c)
for A, Ex, f in ((129, 39.58, ISO[129]), (131, 80.19, ISO[131])):
    nT = n_per_tonne(A, f)
    Rx = nT * np.trapezoid(np.array([sigma_excite(e) for e in Eg]) * phi, Eg) * SEC_PER_YR
    # signal location: ER of Ex keV (gamma/conversion electrons, fully absorbed) + NR of ~E_R (few keV..): use nestpy
    nph_er, ne_er = lz.nest_er_yields(Ex)
    nph_nr, ne_nr = lz.nest_nr_yields(20.0)
    S1 = lz.LZ["g1"] * (nph_er + nph_nr) * _S1_CACHE["k"]; S2 = lz.LZ["g2"] * (ne_er + ne_nr)
    R[f"excitation_{A}"] = dict(Ex_keV=Ex, N_events_exposure=Rx * EXPOSURE, ratio_to_atm_cevns=Rx * EXPOSURE / N_ATM_ROI[0],
                                S1c_phd_with_20keV_NR=S1, log10S2c=math.log10(S2), S1c_ER_only=lz.LZ["g1"] * nph_er * _S1_CACHE["k"], log10S2c_ER_only=math.log10(lz.LZ["g2"] * ne_er))
    log(f"NC excitation {A}Xe ({Ex} keV): {Rx*EXPOSURE:.2e} events (x{Rx*EXPOSURE/N_ATM_ROI[0]:.1e} of CEvNS); lands at S1c~{S1:.0f} phd, log10 S2c~{math.log10(S2):.2f}")
# event location and NR-band centre for reference
nph, ne = lz.nest_nr_yields(248.0)
R["nr_band_248"] = dict(S1c=lz.LZ["g1"] * nph * _S1_CACHE["k"], log10S2c=math.log10(lz.LZ["g2"] * ne))

# --------------------------------------------------------------------------------------
# 5. Exotic neutrino interactions: recoil-spectrum shapes and N_lo = N(5.4-200)/N(200-270)
# --------------------------------------------------------------------------------------
def spectrum_generic(E_grid, kernel, shape=shape_c, ffmodel="helm", emin_func=None):
    """dR/dE_R (arbitrary normalisation) for a cross-section dsigma/dE_R = kernel(E_R, E_nu, A) * F^2.
    emin_func(E_R, A) -> minimum neutrino energy (MeV) if different from elastic kinematics."""
    out = np.zeros_like(E_grid, float)
    for A, f in ISO.items():
        nT = n_per_tonne(A, f)
        for i, E in enumerate(E_grid):
            emin = Enu_min_MeV(E, A) if emin_func is None else emin_func(E, A)
            if emin is None or emin > 1e4: continue
            f2 = lz.helm_F2(E, A)
            g = lambda e: flux_shape(e, *shape) * kernel(E, e, A)
            pts = [p for p in (shape[1], 3 * emin) if emin < p < 1e4]
            I, _ = integrate.quad(g, emin, 1e4, points=pts or None, limit=300)
            out[i] += nT * f2 * I
    return out
def Nlo_from_spec(spec, lo=5.4, mid=200.0, hi=270.0, weighted=True):
    w = eff(E_GRID) if weighted else np.ones_like(E_GRID)
    def N(a, b):
        m = (E_GRID >= a) & (E_GRID <= b); Eg = np.concatenate([[a], E_GRID[m], [b]])
        return np.trapezoid(np.interp(Eg, E_GRID, spec * w), Eg)
    return N(lo, mid) / N(mid, hi), N(mid, hi) / N(0.5, 329)
exo = {}
# SM reference kernel (for consistency with section 1)
k_sm = lambda E, e, A: (1 - mN(A) * E * 1e-6 / (2 * (e * 1e-3) ** 2)) * QW(A) ** 2
spec_sm = spectrum_generic(E_GRID, k_sm)
exo["SM CEvNS (heavy mediator / rescaled)"] = Nlo_from_spec(spec_sm)
# (a) neutrino magnetic moment on the nucleus: Z^2 (1/E_R)(1 - E_R/(2 E_nu))^2
k_mm = lambda E, e, A: Z_XE ** 2 / (E * 1e-3) * (1 - E * 1e-3 / (2 * e)) ** 2
spec_mm = spectrum_generic(E_GRID, k_mm); exo["nu magnetic moment (nuclear, Z^2/E_R)"] = Nlo_from_spec(spec_mm)
# (b) light vector mediator (pure Z' limit): SM kernel x 1/(q^2 + m^2)^2, normalised
for mZp in (10, 30, 100, 246, 1000):
    k = lambda E, e, A, m=mZp: k_sm(E, e, A) / (2 * mN(A) * 1e3 * E * 1e-3 + m ** 2) ** 2 * (m ** 2) ** 2
    exo[f"vector Z' m={mZp} MeV"] = Nlo_from_spec(spectrum_generic(E_GRID, k))
# (c) scalar mediator: dsigma/dE_R ~ Q_s^2 m_N^2 E_R /(E_nu^2 (q^2+m^2)^2)  (Cerdeno et al. 2016 form; recalled likely)
for mphi in (30, 246, 1000, 1e5):
    k = lambda E, e, A, m=mphi: A ** 2 * (mN(A) * 1e3) ** 2 * E * 1e-3 / (e ** 2 * (2 * mN(A) * 1e3 * E * 1e-3 + m ** 2) ** 2) * (m ** 2) ** 2
    exo[f"scalar m={mphi:g} MeV"] = Nlo_from_spec(spectrum_generic(E_GRID, k))
# (d) dipole-portal up-scattering nu N -> N4 N (derived; see details.md):
#     dsigma/dE_R = Z^2 alpha d^2 F^2 * LH / (32 M^3 E_R^2 E_nu^2),
#     LH = 8 M E_R [2 M E_nu - M E_R - m4^2/2]^2 - (4 M^2 + 2 M E_R) m4^2 (2 M E_R + m4^2)
#     kinematic limits from the 2->2 CM frame.
def hnl_kernel_factory(m4_MeV):
    m4 = m4_MeV * 1e-3
    def ER_limits(Enu_GeV, A):
        M = mN(A); s = M * M + 2 * M * Enu_GeV
        if math.sqrt(s) < M + m4: return None
        E1 = (s - M * M) / (2 * math.sqrt(s)); E2 = (s + m4 * m4 - M * M) / (2 * math.sqrt(s)); p2 = math.sqrt(max(E2 * E2 - m4 * m4, 0))
        tmax = m4 * m4 - 2 * E1 * (E2 - p2); tmin = m4 * m4 - 2 * E1 * (E2 + p2)
        return (-tmax / (2 * M) * 1e6, -tmin / (2 * M) * 1e6)   # keV (ER_lo, ER_hi)
    def kern(E_keV, Enu_MeV, A):
        M = mN(A); ER = E_keV * 1e-6; Enu = Enu_MeV * 1e-3
        lim = ER_limits(Enu, A)
        if lim is None or not (lim[0] <= E_keV <= lim[1]): return 0.0
        LH = 8 * M * ER * (2 * M * Enu - M * ER - m4 * m4 / 2) ** 2 - (4 * M * M + 2 * M * ER) * m4 * m4 * (2 * M * ER + m4 * m4)
        return max(LH, 0.0) * Z_XE ** 2 / (32 * M ** 3 * ER ** 2 * Enu ** 2)
    def emin_func(E_keV, A):
        """Smallest E_nu (MeV) for which E_keV lies inside the allowed recoil range (bisection)."""
        M = mN(A); Ethr = (m4 + m4 * m4 / (2 * M)) * 1e3 + 1e-9   # MeV
        def ok(Enu_MeV):
            l = ER_limits(Enu_MeV * 1e-3, A)
            return l is not None and l[0] <= E_keV <= l[1]
        if not ok(1e4): return None
        lo, hi = Ethr, 1e4
        if ok(lo): return lo
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if ok(mid): hi = mid
            else: lo = mid
        return hi
    return kern, ER_limits, emin_func
hnl_spectra = {}
for m4 in (0.0, 0.3, 10, 50, 100, 200, 230, 240, 245):
    kern, lim, emf = hnl_kernel_factory(m4)
    sp = spectrum_generic(E_GRID, kern, emin_func=emf)
    hnl_spectra[m4] = sp
    exo[f"dipole HNL m4={m4:g} MeV"] = Nlo_from_spec(sp)
    lo, hi = lim(0.2, 131) or (float("nan"), float("nan"))
    lo3, hi3 = lim(0.3, 131) or (float("nan"), float("nan"))
    log(f"HNL m4={m4:g}: E_R range at E_nu=200 MeV {lo:.2e}-{hi:.1f} keV, at 300 MeV {lo3:.2e}-{hi3:.1f}; E_nu,min(248 keV)={emf(248,131)}; N_lo={exo[f'dipole HNL m4={m4:g} MeV'][0]:.3g}")
# check m4 -> 0 reproduces the magnetic-moment kernel shape
chk = hnl_spectra[0.0]; chk_mm = spec_mm
R["hnl_m4_0_vs_magnetic_moment_ratio_spread"] = float(np.std((chk / chk.max()) / (chk_mm / chk_mm.max())))
# threshold HNL: m4 such that E_R = 248 keV at threshold: m4 ~ E_nu = sqrt(2 M E_R)
R["hnl_threshold_mass_for_248keV_MeV"] = q_mev(248, 131)
R["exotic_Nlo"] = {k: dict(N_lo_per_hi=v[0], frac_200_270=v[1]) for k, v in exo.items()}
for k, v in exo.items(): log(f"{k:45s} N_lo = {v[0]:10.3g}   frac(200-270) = {v[1]:.3e}")
pd.DataFrame([dict(model=k, N_lo_per_hi=v[0], frac_200_270=v[1]) for k, v in exo.items()]).to_csv(f"{OUT}/exotic_Nlo.csv", index=False)

# --------------------------------------------------------------------------------------
# 6. Solar 8B / hep end points
# --------------------------------------------------------------------------------------
def ERmax_keV(Enu_MeV, A=131):
    M = mN(A) * 1e3
    return 2 * Enu_MeV ** 2 / (M + 2 * Enu_MeV) * 1e3
R["solar_endpoints_keV"] = dict(hep_18p8=ERmax_keV(18.8), B8_16p3=ERmax_keV(16.3), B8_15=ERmax_keV(15.0), A_used=131,
                                Enu_needed_for_248keV_MeV=Enu_min_MeV(248, 131), Enu_needed_for_200keV_MeV=Enu_min_MeV(200, 131))
log("Solar end points:", R["solar_endpoints_keV"])

# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
c0 = list(SHAPES)[0]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for (name, ff), sp in spectra.items():
    if ff != "helm": continue
    ax[0].plot(E_GRID, sp * EXPOSURE, lw=1.6 if name == c0 else 0.8, alpha=1 if name == c0 else 0.6, label=name if name == c0 else None, color="C0" if name == c0 else "0.6")
ax[0].plot(E_GRID, spectra[(c0, "wimpydd")] * EXPOSURE, color="C3", lw=1.4, ls="--", label="central, WimPyDD weak FF")
ax[0].plot(E_GRID, spectra[(c0, "helm")] * EXPOSURE * eff(E_GRID), color="C0", lw=1.0, ls=":", label="central × efficiency")
ax[0].axvspan(225, 271, color="gold", alpha=0.3, label="event ±23 keV")
ax[0].set_yscale("log"); ax[0].set_xlabel("E_R [keV]"); ax[0].set_ylabel("events / keV in 2.84 t·yr"); ax[0].set_ylim(1e-9, 1)
ax[0].set_title("Atmospheric-ν CEνNS, normalised to 0.11 events in the WS ROI"); ax[0].legend(fontsize=7, loc="lower left")
ax[0].text(0.98, 0.95, "grey: other flux shapes (Helm)", transform=ax[0].transAxes, ha="right", fontsize=7)
ax[1].plot(Eg_ff, F2_helm_nat, label="Helm (natural Xe, Q_W²-weighted)"); ax[1].plot(Eg_ff, F2_wd, "--", label="WimPyDD shell-model M response")
ax[1].axvline(248, color="k", ls=":", label="248 keV (q = 246 MeV)"); ax[1].set_yscale("log"); ax[1].set_ylim(1e-5, 1.2)
ax[1].set_xlabel("E_R [keV]"); ax[1].set_ylabel("weak form factor F²(q)"); ax[1].legend(fontsize=7); ax[1].set_title("Loss of coherence")
fig.tight_layout(); fig.savefig(f"{FIG}/P019_cevns_spectrum_and_formfactor.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)
for a, p in zip(ax, ("top", "mid", "bot")):
    v = np.nan_to_num(green5[p]["vals"]); a.step(edges5[:-1], np.where(v > 0, v, np.nan), where="post", color="#7cae00", label="Fig. 5 green (digitised)")
    a.set_yscale("log"); a.set_xlabel("(log10 S2c − μ_NR)/σ_NR"); a.set_title({"top": "S1c < 250 phd", "mid": "250 < S1c < 500", "bot": "S1c > 500 phd"}[p])
    a.axhline(split[f"{c0}|helm"][p] / 8, color="C0", ls="--", label=f"pred. CEνNS total/16 bins = {split[f'{c0}|helm'][p]:.1e}")
    a.axvline(-1.5, color="k", ls=":"); a.legend(fontsize=7)
ax[0].set_ylabel("events / bin")
fig.suptitle("LZ Fig. 5 'Neutrino + Detector NRs' digitised; sums: top %.3f, mid %.2e, bot %.2e" % (fig5["top"]["total"], fig5["mid"]["total"], fig5["bot"]["total"]), fontsize=9)
fig.tight_layout(); fig.savefig(f"{FIG}/P019_fig5_green_digitised.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 4.2))
def nrm(s):
    m = (E_GRID > 5) & (E_GRID < 270); return s / np.trapezoid(s[m], E_GRID[m])
ax.plot(E_GRID, nrm(spec_sm), label="SM CEνNS / heavy Z′")
ax.plot(E_GRID, nrm(spec_mm), label="ν magnetic moment (nucleus)")
ax.plot(E_GRID, nrm(spectrum_generic(E_GRID, lambda E, e, A: A ** 2 * (mN(A) * 1e3) ** 2 * E * 1e-3 / e ** 2)), label="heavy scalar mediator")
for m4 in (50, 100, 200, 240): ax.plot(E_GRID, nrm(hnl_spectra[m4]), ls="--", label=f"dipole HNL m4 = {m4} MeV")
ax.axvspan(200, 270, color="gold", alpha=0.3); ax.set_yscale("log"); ax.set_ylim(1e-7, 1); ax.set_xlabel("E_R [keV]"); ax.set_ylabel("normalised dR/dE_R (5–270 keV)")
ax.legend(fontsize=7); ax.set_title("Exotic ν–nucleus spectra (atmospheric flux, central shape)")
fig.tight_layout(); fig.savefig(f"{FIG}/P019_exotic_shapes.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
d = pd.DataFrame(inc)
ax.plot(d.Enu_MeV, d.sig_coh_131, "o-", label="coherent (¹³¹Xe, Helm)"); ax.plot(d.Enu_MeV, d.sig_incoh_nucleus, "s--", label="incoherent NC elastic Σ nucleons (Pauli-suppressed)")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("E_ν [MeV]"); ax.set_ylabel("σ per nucleus [cm²]"); ax.legend(fontsize=8); ax.set_title("Coherent vs incoherent NC on xenon")
fig.tight_layout(); fig.savefig(f"{FIG}/P019_coherent_vs_incoherent.png", dpi=150); plt.close(fig)

R["headline"] = dict(
    N_above_200_central_helm=central_helm["N_above_200_eff"], N_above_200_central_wd=central_wd["N_above_200_eff"],
    N_225_271_central_helm=central_helm["N_225_271_eff"], N_225_271_central_wd=central_wd["N_225_271_eff"],
    frac_above_200_raw_central=central_helm["frac_above_200_raw"], N_lo_per_hi_central=central_helm["N_lo_per_hi"],
    fig5_green_bottom_sum=fig5["bot"]["total"], fig5_green_mid_sum=fig5["mid"]["total"], fig5_green_top_sum=fig5["top"]["total"])
with open(f"{OUT}/P019_results.json", "w") as f:
    json.dump(R, f, indent=1, default=float)
with open(f"{OUT}/run_log.txt", "w") as f:
    f.write("\n".join(LOG))
print("done ->", OUT)
