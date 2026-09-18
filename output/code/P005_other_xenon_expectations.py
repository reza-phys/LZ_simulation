"""
P005 -- Would XENONnT and PandaX-4T have seen it?  Expected high-energy counts in their
published regions of interest under LZ's best-fit signal models.

Run from the simulation root:
    .venv/bin/python output/code/P005_other_xenon_expectations.py

Outputs (all under output/work/P005/):
    spectra_per_tyr.csv          -- WimPyDD spectra (events / t yr keV) before normalisation
    spectra_normalised_LZ1.csv   -- spectra normalised to 1.0 event in LZ (2.84 t yr, LZ efficiency)
    counts_table.csv             -- expected counts per experiment / ROI / model (+ P(0))
    fractions_table.csv          -- fraction of the LZ-normalised, efficiency-weighted spectrum below each edge
    summary.json                 -- all headline numbers
    fig1_L10_digitised.csv       -- colour-digitised brown curve of LZ Fig. 1 (bottom), for validation
    figures/*.png

Physics: all three detectors use natural xenon, so dR/dE per tonne-year is the same function of
true recoil energy for a given model and halo; the only differences are exposure and the energy-
dependent efficiency (threshold, ROI upper edge, analysis cuts).  We therefore compute one spectrum
per model with WimPyDD (the code LZ used) and fold it with three efficiency curves.
"""
from __future__ import annotations
import sys, os, json, math
import numpy as np
from scipy import special, integrate, stats, optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "output/code")
from common import lzcommon as lz

OUT = "output/work/P005"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

# ----------------------------------------------------------------------------------------------
# 0. Inputs
# ----------------------------------------------------------------------------------------------
M_CHI = 1000.0                          # GeV; LZ's representative mass (Table I, Fig. 1)
LZ_EXPO = lz.LZ["exposure_tyr"]         # 2.84 t yr  [paper, abstract]
LZ_BESTFIT = lz.LZ["L10s_1000_bestfit"] # (1.0, +1.4, -0.7) events [paper, Table I]
LZ_E50_LO, LZ_E50_HI = lz.LZ["E_50pct_low_keV"], lz.LZ["E_50pct_high_keV"]   # 5.4, 269.9 keV [paper]
LZ_PLATEAU = lz.LZ["eff_plateau"]       # 0.96 [paper]
SIG_LO = 2.5    # keV: width of the low-energy erf roll-off (Fig. S2: ~95% by 10 keV) [this work, read from Fig. S2]
SIG_HI = 12.0   # keV: width of the high-energy roll-off (Fig. S2 inset: 0.9 at 250, 0.5 at 270, 0.1 at ~283) [this work]

# Other experiments -- recalled from memory; reliabilities in details.md / provenance JSON
OTHER = {
    "XENONnT": dict(exposure_tyr=3.1,  ref="PRL 135, 221003 (2025); arXiv:2502.18005", expo_rel="certain (cited in LZ paper title)",
                    roi_note="cS1 <~ 100 PE, NR ROI roughly 3-60 keV", roi_rel="uncertain"),
    "PandaX-4T": dict(exposure_tyr=1.54, ref="PRL 134, 011805 (2025); arXiv:2408.00664", expo_rel="certain (cited in LZ paper title)",
                    roi_note="NR ROI upper edge ~100-120 keV (?)", roi_rel="uncertain"),
}
ROI_EDGES = [55.0, 70.0, 100.0, 150.0, 270.0]   # keV; 270 = 'LZ-like extended ROI'
EFF_OTHER_PLATEAU = 0.90                          # assumption (stated)
E50_OTHER_LO = 5.0                                # keV, generic threshold assumption (irrelevant for these spectra)

E_GRID = np.arange(0.5, 400.0 + 1e-9, 0.5)       # keV

# ----------------------------------------------------------------------------------------------
# 1. Efficiency curves
# ----------------------------------------------------------------------------------------------
def erf_step_up(E, e50, sig):
    return 0.5 * (1.0 + special.erf((E - e50) / (math.sqrt(2.0) * sig)))

def eff_LZ(E, plateau=LZ_PLATEAU):
    """Smooth model of LZ Fig. S2 (black curve): erf turn-on at 5.4 keV, erf roll-off at 269.9 keV."""
    E = np.asarray(E, float)
    return plateau * erf_step_up(E, LZ_E50_LO, SIG_LO) * (1.0 - erf_step_up(E, LZ_E50_HI, SIG_HI))

def eff_other(E, e_max, plateau=EFF_OTHER_PLATEAU, e50_lo=E50_OTHER_LO, sig_lo=2.0):
    """Generic efficiency for another LXe TPC: plateau inside ROI, erf turn-on at e50_lo, and either a hard
    upper edge at e_max (standard WIMP-search ROIs) or, for e_max == 270, LZ's smooth 269.9 keV roll-off."""
    E = np.asarray(E, float)
    lo = erf_step_up(E, e50_lo, sig_lo)
    if abs(e_max - 270.0) < 1e-6:
        hi = 1.0 - erf_step_up(E, LZ_E50_HI, SIG_HI)
    else:
        hi = (E <= e_max).astype(float)
    return plateau * lo * hi

# ----------------------------------------------------------------------------------------------
# 2. Signal models with WimPyDD
# ----------------------------------------------------------------------------------------------
mN = lz.M_NUCLEON_GEV
WD = lz.wd()
HALO = lz.wd_halo()      # Baxter-2021 SHM, time-averaged (no annual modulation)

def ham_const(name, op, c0=1.0):
    return lz.wd_hamiltonian(name, {op: (c0, 0.0)})

def ham_L10():
    """Anand et al. (2014) Table 1, entry 10 (recalled, 'likely'):
       L10 = chi-bar i sigma^{mu nu} q_nu/m_M chi  N-bar i sigma_{mu alpha} q^alpha/m_M N  (dipole-dipole)
           -> 4 (q^2/m_M^2) O4 - 4 (m_N^2/m_M^2) O6 .
       Overall factor 4/m_M^2 and the coupling d10 drop out after normalisation to 1 LZ event; only the
       relative coefficient (q^2 O4 - m_N^2 O6, i.e. transverse spin-spin) matters.  Check: c4 + (q^2/m_N^2) c6 = 0,
       so the longitudinal Sigma'' response cancels exactly and only Sigma' survives (physically the
       dipole-dipole tensor structure), which is why the isoscalar and isovector spectra coincide (LZ SM)."""
    c4 = lambda q: [q**2 / mN**2, 0.0]
    c6 = lambda q: [-1.0 + 0.0 * q, 0.0]
    return WD.eft_hamiltonian("L10_dipole_dipole", {(4, "q2"): c4, 6: c6})

def ham_L10_wrong_sign():
    """Same but with the O6 sign flipped (Sigma'' does not cancel) -- used only as a robustness/validation check."""
    c4 = lambda q: [q**2 / mN**2, 0.0]
    c6 = lambda q: [+1.0 + 0.0 * q, 0.0]
    return WD.eft_hamiltonian("L10_flipped", {(4, "q2"): c4, 6: c6})

MODELS = {
    # key: (label, hamiltonian, delta_keV)
    "O1s_d250": ("O1 inelastic, delta=250 keV", ham_const("o1s", 1), 250.0),
    "O1s_d300": ("O1 inelastic, delta=300 keV", ham_const("o1s", 1), 300.0),
    "O1s_d350": ("O1 inelastic, delta=350 keV", ham_const("o1s", 1), 350.0),
    "L10_1000": ("L10 magnetic dipole (q^2 O4 - m_N^2 O6)", ham_L10(), 0.0),
    "O4_1000":  ("O4 elastic (SD-like proxy)", ham_const("o4s", 4), 0.0),
    "O6_1000":  ("O6 elastic (q^4 Sigma'' proxy)", ham_const("o6s", 6), 0.0),
    "O1s_el":   ("O1 elastic (SI reference)", ham_const("o1s", 1), 0.0),
    "L10_flip": ("L10 with flipped O6 sign (check)", ham_L10_wrong_sign(), 0.0),
}

def spectrum(key):
    label, H, delta = MODELS[key]
    r = lz.wd_rate(H, M_CHI, E_GRID, halo=HALO, delta_kev=delta)
    r = np.where(np.isfinite(r), r, 0.0)
    return np.clip(r, 0.0, None)

print("Computing WimPyDD spectra ...", flush=True)
SPEC = {k: spectrum(k) for k in MODELS}
for k, s in SPEC.items():
    print(f"  {k:10s} peak {s.max():.3e} /t/yr/keV at {E_GRID[s.argmax()]:.1f} keV; first nonzero E = {E_GRID[s>0][0] if (s>0).any() else float('nan'):.1f} keV")

def integ(y, w=None):
    yy = y if w is None else y * w
    return float(np.trapezoid(yy, E_GRID))

# normalise every model to LZ_BESTFIT[0] = 1.0 expected events in LZ
EFF_LZ = eff_LZ(E_GRID)
NORM = {}
SPEC_N = {}
for k, s in SPEC.items():
    n_lz_unit = LZ_EXPO * integ(s, EFF_LZ)
    NORM[k] = LZ_BESTFIT[0] / n_lz_unit
    SPEC_N[k] = s * NORM[k]

# ----------------------------------------------------------------------------------------------
# 3. Digitise LZ Fig. 1 (bottom, brown = L10 1000 GeV) from the PNG for validation
# ----------------------------------------------------------------------------------------------
def digitise_fig1_L10():
    import matplotlib.image as mpimg
    img = mpimg.imread("inputs/figures_png/Fig1_combined_recoils.png")
    rgb = (img[..., :3] * 255).astype(int)
    # calibration established in scratch tests (details.md):
    #   left spine x=188.5 px (0 keV), right edge 1649 px (350 keV); check: gray edges at 5.39 and 270.0 keV
    #   y labels 1e-1, 1e-2, 1e-3 centred at rows 1145, 1470, 1785 (bottom spine 1795) -> 325 px/decade
    x_of = lambda px: (px - 188.5) / (1649.0 - 188.5) * 350.0
    y_of = lambda py: 10.0 ** (-1.0 - (py - 1145.0) / 325.0)
    brown = np.array([140, 86, 75])
    m = np.abs(rgb - brown).sum(-1) < 60
    m[:1061, :] = False; m[1795:, :] = False     # bottom panel only
    # the legend box (title + 3 line samples) occupies rows < ~1215 in the bottom panel; the brown curve itself never
    # exceeds ~0.035 /t/yr/keV (row ~1300), so dropping rows < 1250 (y > 0.047) removes the legend sample without touching the curve
    m[:1250, :] = False
    cols = np.nonzero(m.any(0))[0]
    xs, ys = [], []
    for c in cols:
        rows = np.nonzero(m[:, c])[0]
        xs.append(x_of(c)); ys.append(y_of(np.median(rows)))
    xs, ys = np.array(xs), np.array(ys)
    # keep only E >= 3 keV where the curve is above the axis bottom, and drop the legend region (E<~220 keV, y>0.08)
    keep = (xs >= 3.0) & (ys < 0.047)
    return xs[keep], ys[keep]

try:
    DX, DY = digitise_fig1_L10()
    np.savetxt(os.path.join(OUT, "fig1_L10_digitised.csv"), np.c_[DX, DY], delimiter=",",
               header="E_keV,dRdE_per_t_yr_keV (colour-digitised from inputs/figures_png/Fig1_combined_recoils.png, brown curve)", comments="")
    dig_ok = True
except Exception as e:  # pragma: no cover
    print("digitisation failed:", e); dig_ok = False

VALID = {}
if dig_ok:
    # interpolate the digitised curve onto E_GRID (only where covered), compare shapes with the WimPyDD L10
    lo, hi = DX.min(), DX.max()
    sel = (E_GRID >= max(lo, 3.0)) & (E_GRID <= min(hi, 350.0))
    dig_on_grid = np.interp(E_GRID[sel], DX, DY)
    w = SPEC["L10_1000"][sel]
    # scale factor from the ROI-weighted integral (5.4-269.9 keV) so shapes can be compared
    effw = EFF_LZ[sel]
    scale = np.trapezoid(dig_on_grid * effw, E_GRID[sel]) / np.trapezoid(w * effw, E_GRID[sel])
    ratio = dig_on_grid / (w * scale)
    inroi = (E_GRID[sel] > 10) & (E_GRID[sel] < 265)
    VALID = dict(
        scale_fig1_over_wimpydd_unitcoupling=float(scale),
        ratio_rms_10_265keV=float(np.sqrt(np.mean((ratio[inroi] - 1) ** 2))),
        ratio_min_10_265keV=float(ratio[inroi].min()), ratio_max_10_265keV=float(ratio[inroi].max()),
        dig_peak_keV=float(DX[np.argmax(DY * (DX > 100))]), wd_peak_keV=float(E_GRID[np.argmax(SPEC["L10_1000"] * (E_GRID > 100))]),
        dig_lowpeak_keV=float(DX[np.argmax(DY * (DX < 45))]), wd_lowpeak_keV=float(E_GRID[np.argmax(SPEC["L10_1000"] * (E_GRID < 45))]),
        dig_dip_keV=float(DX[(DX > 30) & (DX < 120)][np.argmin(DY[(DX > 30) & (DX < 120)])]),
        wd_dip_keV=float(E_GRID[(E_GRID > 30) & (E_GRID < 120)][np.argmin(SPEC["L10_1000"][(E_GRID > 30) & (E_GRID < 120)])]),
        dig_hi_over_lo_peak=float(DY[np.argmax(DY * (DX > 100))] / DY[np.argmax(DY * (DX < 45))]),
        wd_hi_over_lo_peak=float(SPEC["L10_1000"][np.argmax(SPEC["L10_1000"] * (E_GRID > 100))] / SPEC["L10_1000"][np.argmax(SPEC["L10_1000"] * (E_GRID < 45))]),
    )
    # also a digitised-curve version of the L10 model, normalised to 1 LZ event (used as a cross-check row)
    dig_full = np.interp(E_GRID, DX, DY, left=0.0, right=0.0)
    SPEC["L10_fig1dig"] = dig_full
    MODELS["L10_fig1dig"] = ("L10 1000 GeV digitised from LZ Fig. 1 (check)", None, 0.0)
    NORM["L10_fig1dig"] = LZ_BESTFIT[0] / (LZ_EXPO * integ(dig_full, EFF_LZ))
    SPEC_N["L10_fig1dig"] = dig_full * NORM["L10_fig1dig"]
    print("Fig.1 validation:", json.dumps(VALID, indent=1))

# ----------------------------------------------------------------------------------------------
# 4. Expected counts in XENONnT and PandaX-4T
# ----------------------------------------------------------------------------------------------
rows = []
frac_rows = []
for k in SPEC_N:
    s = SPEC_N[k]
    # fraction of LZ-detected (efficiency-weighted) signal below each edge
    tot_eff = integ(s, EFF_LZ)
    fr = {f"frac_below_{int(e)}": integ(s, EFF_LZ * (E_GRID <= e)) / tot_eff for e in ROI_EDGES[:-1]}
    fr.update(model=k, label=MODELS[k][0], N_LZ=LZ_EXPO * tot_eff,
              E_first_nonzero=float(E_GRID[s > 1e-12 * s.max()][0]) if s.max() > 0 else float("nan"),
              E_peak=float(E_GRID[s.argmax()]), E_median_LZ=float(E_GRID[np.searchsorted(np.cumsum(s * EFF_LZ), 0.5 * np.sum(s * EFF_LZ))]))
    frac_rows.append(fr)
    for exp_name, X in OTHER.items():
        for e_max in ROI_EDGES:
            effX = eff_other(E_GRID, e_max)
            N = X["exposure_tyr"] * integ(s, effX)
            rows.append(dict(model=k, experiment=exp_name, exposure_tyr=X["exposure_tyr"], roi_max_keV=e_max,
                             N_expected=N, N_lo=N * (LZ_BESTFIT[0] + LZ_BESTFIT[2]) / LZ_BESTFIT[0],
                             N_hi=N * (LZ_BESTFIT[0] + LZ_BESTFIT[1]) / LZ_BESTFIT[0],
                             P_zero=math.exp(-N), P_zero_hi=math.exp(-N * (LZ_BESTFIT[0] + LZ_BESTFIT[1]) / LZ_BESTFIT[0]),
                             P_ge1=1 - math.exp(-N)))
    # both experiments combined
    for e_max in ROI_EDGES:
        effX = eff_other(E_GRID, e_max)
        N = sum(X["exposure_tyr"] for X in OTHER.values()) * integ(s, effX)
        rows.append(dict(model=k, experiment="XENONnT+PandaX-4T", exposure_tyr=sum(X["exposure_tyr"] for X in OTHER.values()),
                         roi_max_keV=e_max, N_expected=N, N_lo=N * 0.3, N_hi=N * 2.4, P_zero=math.exp(-N),
                         P_zero_hi=math.exp(-2.4 * N), P_ge1=1 - math.exp(-N)))

import csv
with open(os.path.join(OUT, "counts_table.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
with open(os.path.join(OUT, "fractions_table.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(frac_rows[0].keys())); w.writeheader(); w.writerows(frac_rows)
hdr = "E_keV," + ",".join(SPEC.keys())
np.savetxt(os.path.join(OUT, "spectra_per_tyr.csv"), np.c_[E_GRID, np.array([SPEC[k] for k in SPEC]).T], delimiter=",", header=hdr, comments="")
np.savetxt(os.path.join(OUT, "spectra_normalised_LZ1.csv"), np.c_[E_GRID, np.array([SPEC_N[k] for k in SPEC_N]).T], delimiter=",", header=hdr, comments="")

# ----------------------------------------------------------------------------------------------
# 5. Xenon-world combination and required exposures
# ----------------------------------------------------------------------------------------------
EXPO_ALL = LZ_EXPO + sum(X["exposure_tyr"] for X in OTHER.values())     # 7.48 t yr
rate_LZroi_per_tyr = LZ_BESTFIT[0] / LZ_EXPO                            # events per t yr with LZ's ROI+efficiency (0.96)
mu_world_LZeff = rate_LZroi_per_tyr * EXPO_ALL
mu_world_others_0p9 = rate_LZroi_per_tyr * (EFF_OTHER_PLATEAU / LZ_PLATEAU) * (EXPO_ALL - LZ_EXPO)
world = dict(
    exposure_total_tyr=EXPO_ALL,
    mu_world_if_all_LZ_eff=mu_world_LZeff,
    P_exactly_one_total=float(stats.poisson.pmf(1, mu_world_LZeff)),
    P_exactly_one_total_lo=float(stats.poisson.pmf(1, mu_world_LZeff * 0.3)),
    P_exactly_one_total_hi=float(stats.poisson.pmf(1, mu_world_LZeff * 2.4)),
    P_zero_in_others_given_LZ_one=float(math.exp(-mu_world_others_0p9)),
    mu_others_270_0p9=mu_world_others_0p9,
    # P(LZ sees >=1 AND others see 0) vs P(others see >=1): the a-priori chance that the first event lands in LZ
    P_first_event_in_LZ=float(LZ_EXPO * LZ_PLATEAU / (LZ_EXPO * LZ_PLATEAU + (EXPO_ALL - LZ_EXPO) * EFF_OTHER_PLATEAU)),
)
# required exposures at 270 keV ROI with 0.9 plateau
rate270_0p9 = rate_LZroi_per_tyr * EFF_OTHER_PLATEAU / LZ_PLATEAU
mu_ge1_90 = -math.log(0.1)
mu_ge3_90 = optimize.brentq(lambda mu: stats.poisson.sf(2, mu) - 0.9, 0.1, 50)
req = {}
for tag, scale in [("bestfit", 1.0), ("lower68", 0.3), ("upper68", 2.4)]:
    req[tag] = dict(rate_per_tyr=rate270_0p9 * scale,
                    exposure_for_90pct_ge1_tyr=mu_ge1_90 / (rate270_0p9 * scale),
                    exposure_for_90pct_ge3_tyr=mu_ge3_90 / (rate270_0p9 * scale))
# and for the restricted ROIs, per model (best fit), for XENONnT-like efficiency
req_roi = {}
for k in ["O1s_d250", "O1s_d300", "O1s_d350", "L10_1000"]:
    req_roi[k] = {}
    for e_max in ROI_EDGES:
        r = integ(SPEC_N[k], eff_other(E_GRID, e_max))
        req_roi[k][str(int(e_max))] = dict(rate_per_tyr=r, exposure_for_90pct_ge1_tyr=(mu_ge1_90 / r if r > 0 else float("inf")))

summary = dict(
    inputs=dict(m_chi_GeV=M_CHI, LZ_exposure_tyr=LZ_EXPO, LZ_bestfit_events=LZ_BESTFIT, LZ_eff=dict(plateau=LZ_PLATEAU, E50_lo=LZ_E50_LO, E50_hi=LZ_E50_HI, sig_lo=SIG_LO, sig_hi=SIG_HI),
                other=OTHER, roi_edges_keV=ROI_EDGES, eff_other_plateau=EFF_OTHER_PLATEAU, e50_other_lo=E50_OTHER_LO),
    normalisation_factors=NORM, fig1_validation=VALID, fractions=frac_rows,
    counts=rows, xenon_world=world, required_exposure_270keV_0p9=dict(mu_ge1_90=mu_ge1_90, mu_ge3_90=mu_ge3_90, **req),
    required_exposure_by_roi_bestfit=req_roi,
)
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=1, default=float)

# ----------------------------------------------------------------------------------------------
# 6. Figures
# ----------------------------------------------------------------------------------------------
plt.rcParams.update({"font.size": 10})
main_keys = ["O1s_d250", "O1s_d300", "O1s_d350", "L10_1000", "O1s_el"]
colors = {"O1s_d250": "tab:orange", "O1s_d300": "tab:green", "O1s_d350": "tab:red", "L10_1000": "tab:brown", "O1s_el": "tab:blue",
          "O4_1000": "tab:purple", "O6_1000": "tab:gray", "L10_fig1dig": "k"}
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for k in main_keys:
    ax.plot(E_GRID, SPEC_N[k], color=colors[k], lw=1.6, label=MODELS[k][0])
if dig_ok:
    ax.plot(E_GRID, SPEC_N["L10_fig1dig"], color="k", lw=0.9, ls="--", label="L10 digitised from LZ Fig. 1 (check)")
for e, a in zip([55, 70, 100, 150], [0.28, 0.20, 0.13, 0.07]):
    ax.axvspan(0, e, color="tab:blue", alpha=a, lw=0)
    ax.text(e - 2, 4.5e-2, f"{e}", ha="right", va="top", fontsize=8, color="navy")
ax.axvspan(269.9, 400, color="0.5", alpha=0.35, lw=0)
ax.axvline(5.4, color="0.5", lw=0.8)
ax.set_yscale("log"); ax.set_xlim(0, 400); ax.set_ylim(1e-6, 6e-2)
ax.set_xlabel("True nuclear-recoil energy [keV]"); ax.set_ylabel("dR/dE  [events / (t yr keV)],  normalised to 1.0 LZ event")
ax.set_title("Spectra normalised to 1.0 LZ event (m = 1000 GeV); shaded: ROI upper edges", fontsize=10)
ax.legend(fontsize=7.5, loc="lower center", framealpha=0.85)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "spectra_with_ROIs.png"), dpi=160); plt.close(fig)

# efficiency curves
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.plot(E_GRID, EFF_LZ, "k", lw=1.8, label="LZ (model of Fig. S2)")
for e, c in zip([55, 100, 270], ["tab:blue", "tab:cyan", "tab:red"]):
    ax.plot(E_GRID, eff_other(E_GRID, e), color=c, lw=1.2, ls="--", label=f"other expt, ROI to {e} keV (0.9 plateau)")
ax.set_xlim(0, 320); ax.set_ylim(0, 1.05); ax.set_xlabel("NR energy [keV]"); ax.set_ylabel("efficiency"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "efficiency_models.png"), dpi=160); plt.close(fig)

# counts bar chart (best fit) for XENONnT and PandaX vs ROI edge
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
for ax, exp_name in zip(axes, OTHER):
    xpos = np.arange(len(ROI_EDGES)); wbar = 0.17
    for i, k in enumerate(["O1s_d250", "O1s_d300", "O1s_d350", "L10_1000"]):
        Ns = [next(r["N_expected"] for r in rows if r["model"] == k and r["experiment"] == exp_name and r["roi_max_keV"] == e) for e in ROI_EDGES]
        ax.bar(xpos + (i - 1.5) * wbar, Ns, wbar, color=colors[k], label=MODELS[k][0])
    ax.set_xticks(xpos); ax.set_xticklabels([f"{int(e)}" for e in ROI_EDGES]); ax.set_xlabel("ROI upper edge [keV]")
    ax.set_title(f"{exp_name}  ({OTHER[exp_name]['exposure_tyr']} t yr, 0.9 eff.)"); ax.set_yscale("log"); ax.set_ylim(1e-4, 3)
    ax.axhline(1.0, color="k", lw=0.6, ls=":")
axes[0].set_ylabel("expected events (LZ best fit = 1.0 event)"); axes[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "expected_counts.png"), dpi=160); plt.close(fig)

# validation figure: WimPyDD L10 vs digitised Fig.1
if dig_ok:
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    sc = VALID["scale_fig1_over_wimpydd_unitcoupling"]
    ax[0].plot(DX, DY, "k.", ms=2, label="LZ Fig. 1 bottom, 1000 GeV L10 (digitised)")
    ax[0].plot(E_GRID, SPEC["L10_1000"] * sc, color="tab:brown", lw=1.5, label="WimPyDD q$^2$O$_4$ - m$_N^2$O$_6$ (scaled)")
    ax[0].plot(E_GRID, SPEC["L10_flip"] * sc * SPEC["L10_1000"].max() / SPEC["L10_flip"].max(), color="tab:gray", lw=1, ls="--", label="flipped O6 sign (scaled to same peak)")
    ax[0].set_yscale("log"); ax[0].set_ylim(1e-3, 1e-1); ax[0].legend(fontsize=7.5); ax[0].set_ylabel("dR/dE [/t/yr/keV]")
    sel = (E_GRID >= 3) & (E_GRID <= 350)
    ax[1].plot(E_GRID[sel], np.interp(E_GRID[sel], DX, DY) / (SPEC["L10_1000"][sel] * sc), color="tab:brown")
    ax[1].axhline(1, color="k", lw=0.6); ax[1].set_ylim(0.5, 1.5); ax[1].set_ylabel("digitised / WimPyDD"); ax[1].set_xlabel("NR energy [keV]")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "L10_validation_vs_Fig1.png"), dpi=160); plt.close(fig)

# ----------------------------------------------------------------------------------------------
# 7. Console summary
# ----------------------------------------------------------------------------------------------
print("\nNormalisation factors (multiply WimPyDD unit-coupling rate to get 1.0 LZ event):")
for k, v in NORM.items():
    print(f"  {k:12s} {v:.4e}")
print("\nFractions of LZ-detectable signal below edge:")
for fr in frac_rows:
    print(f"  {fr['model']:12s} first E={fr['E_first_nonzero']:6.1f} peak={fr['E_peak']:6.1f} median={fr['E_median_LZ']:6.1f} | " +
          " ".join(f"<{int(e)}:{fr[f'frac_below_{int(e)}']:.4f}" for e in ROI_EDGES[:-1]))
print("\nExpected counts (best fit; [lo, hi] = LZ 68% band):")
for r in rows:
    if r["model"] in ["O1s_d250", "O1s_d300", "O1s_d350", "L10_1000", "L10_fig1dig", "O1s_el"]:
        print(f"  {r['model']:12s} {r['experiment']:18s} ROI<{int(r['roi_max_keV']):3d}: N={r['N_expected']:.4f} [{r['N_lo']:.4f},{r['N_hi']:.4f}]  P(0)={r['P_zero']:.3f}")
print("\nXenon world:", json.dumps(world, indent=1))
print("Required exposure (270 keV ROI, 0.9 eff):", json.dumps(summary["required_exposure_270keV_0p9"], indent=1))
print("Required exposure by ROI (best fit):", json.dumps(req_roi, indent=1))
