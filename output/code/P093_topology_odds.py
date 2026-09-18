#!/usr/bin/env python
"""P093 -- Event-topology odds for the LZ 248 keV candidate.

Combines every topological handle the corpus has quantified into a per-hypothesis
likelihood table L(handle | H) and a classification posterior

    P(H | event at D) = mu_H p_H(D) / sum_H' mu_H' p_H'(D)

for a marked Poisson process: mu_H is the expected number of events of type H in the
WS-ROI science sample and p_H(D) the density of the observed handles under H.  Each
p_H(D) is factorised into handles

    rate  : mu_ROI(H)                 expected events of type H in the WS ROI (science sample or pre-veto)
    E     : P(S1c > 500 phd | H, ROI)  'energy' handle (bottom Fig. 5 panel)
    B     : P(|d| <= 2 sigma_NR | H, S1c > 500)   band-distance handle (LZ's Fig. 5 variable)
    P     : p_H(r,z) / p_uniform(r,z)  position-density ratio at the event (P070 convention: L_P = 1/LR)
    W     : pulse/waveform handles as LZ states them (single-scatter S2, TBA, hit pattern, rise time)
    V     : veto quietness (science sample): (1 - tagging efficiency) where the pre-veto rate is quoted
    T     : timing (date, run conditions), relative to a time-flat background

Every number is traced to a corpus paper or to the LZ paper (see details.md).  Ranges are
propagated by Monte Carlo with log-uniform sampling inside the quoted brackets.

Run from the simulation root:
    .venv/bin/python output/code/P093_topology_odds.py
"""
import json, os, sys, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

sys.path.insert(0, "output/code")
from common import lzcommon as lz  # noqa: E402  (LZ dict for cross-checks)

OUT = "output/work/P093"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(93)
N_MC = 40000

HYP = ["NR", "WALL", "RFR", "ER", "ACC", "NEU", "ART", "NU"]
HYP_LABEL = {"NR": "single NR (DM-like)", "WALL": "wall MSSI", "RFR": "RFR MSSI",
             "ER": "charge-poor / leakage ER", "ACC": "accidental coincidence",
             "NEU": "neutron (radiogenic+cosmogenic)", "ART": "instrumental artefact",
             "NU": "atmospheric-nu NR"}
HANDLES = ["rate", "E", "B", "P", "W", "V", "T"]

# --------------------------------------------------------------------------------------
# 1. Handle table.  Each entry: (central, lo, hi, source).  lo == hi -> fixed.
#    Veto: for WALL/RFR/NEU the 'rate' is the PRE-veto ROI rate and V = 1 - eff, so that
#    rate*V reproduces the science-sample expectation.  For all bulk-TPC hypotheses the
#    random-veto loss (0.01 % prompt + 2.86 % delayed) is common and cancels -> V = 1.
# --------------------------------------------------------------------------------------
LZ_WALL_SCI = 0.0048          # P004: wall share of Table I MSSI 4.9e-3
LZ_RFR_SCI = 1.0e-4           # P033/P070: LZ RFR expectation in the 4.7 t ROI
MSSI_VETO = 0.94              # LZ: prompt-veto tagging efficiency for MSSI, 94 +- 2 %
NEU_VETO = 0.92               # LZ: 92 +- 4 % (alpha,n) tagging efficiency
NEU_SS_UL = 0.118             # LZ Table I: detector NR fit interval [0, 0.118]
ACC_TOT = 2.47                # P022: digitised Fig. 5 accidentals (Table I fit 2.6)
NU_TOT = 0.11 + 0.057         # Table I atm-nu + 8B/hep
ER_TOT = 1700.0               # science-sample ER-dominated total (1710 observed)


def T(c, lo=None, hi=None, src=""):
    if lo is None:
        lo = hi = c
    return dict(c=float(c), lo=float(lo), hi=float(hi), src=src)


# Corpus-revised table (case ii)
TAB = {
    "NR": {
        "rate": T(1.0, 1.0, 1.0, "s: LZ L10 best fit 1.0 (+1.4 -0.7); scanned 0.105 / 1.0 / 2.4"),
        "E": T(0.21, 0.15, 0.45, "P052: L10 split 0.21/0.59/0.21 across Fig. 5 panels; P038: delta>=350 keV spectra put more above 500 phd"),
        "B": T(0.90, 0.85, 0.95, "Gaussian +-2 sigma 0.954; LZ Fig. 5 L10 d-curve has heavier tails (P052)"),
        "P": T(1.0, 1.0, 1.0, "uniform in FV by construction (P070)"),
        "W": T(1.0, 1.0, 1.0, "LZ: S2 shape, TBA, hit pattern all consistent with a SS at the reconstructed position"),
        "V": T(1.0, 1.0, 1.0, "no companion expected (Higgsino/dark-photon/contact: P042 P(not clean) <= 3e-5)"),
        "T": T(1.0, 1.0, 1.0, "elastic 0.98; inelastic 1.4-2.6 (P006) treated as a variant"),
    },
    "WALL": {
        "rate": T(LZ_WALL_SCI / (1 - MSSI_VETO) * 0.62, LZ_WALL_SCI / (1 - MSSI_VETO) * 0.2, LZ_WALL_SCI / (1 - MSSI_VETO) * 1.64,
                  "pre-veto wall MSSI 0.0048/0.06 x k; P004 k_ML = 0.62, k < 1.64 (95 %); lower bracket 0.2"),
        "E": T(0.103, 0.08, 0.13, "P004 Fig. 5 digitisation: MSSI fraction at S1c > 500 phd = 0.103 (spectrum ~flat in S1c)"),
        "B": T(0.343, 0.30, 0.40, "P004: 1.74e-4 of 5.06e-4 within +-2 sigma_NR (MSSI ~flat in d)"),
        "P": T(1 / 15.1, 1 / 50.0, 1 / 2.3, "P070: LR(uniform/wall MSSI) 15.1 (P033 equal mix); 2.3 (bottom/cathode source) - 49.9 (wall source)"),
        "W": T(1.0, 0.3, 1.0, "LZ: S1 pattern cannot distinguish SS from wall-MSSI template; TBA of merged S1 requires wall deposit within ~7 cm in z (unquantified)"),
        "V": T(1 - MSSI_VETO, 1 - 0.96, 1 - 0.92, "LZ prompt-veto tagging 94 +- 2 % for MSSI"),
        "T": T(1.0, 1.0, 1.0, "214Pb mixed-flow coincidence unremarkable (P053); activation route <= 1e-9 (P029)"),
    },
    "RFR": {
        "rate": T(LZ_RFR_SCI / (1 - MSSI_VETO), LZ_RFR_SCI / (1 - MSSI_VETO) * 0.7, LZ_RFR_SCI / (1 - MSSI_VETO) * 1.4,
                  "pre-veto RFR MSSI 1e-4/0.06; HE-SB validated to ~20-30 % (LZ 21.5 vs 18; P053)"),
        "E": T(0.103, 0.05, 0.20, "assumed as wall MSSI; 204 +- 65 keV RFR deposit broadens the S1c distribution"),
        "B": T(0.343, 0.30, 0.40, "as wall MSSI (flat in d)"),
        "P": T(1 / 1.45, 1 / 60.0, 1 / 1.35, "P070: detector-gamma RFR map 1.45; 214Pb E2 profiles 17-60 (352/295 keV)"),
        "W": T(0.01, 1e-3, 0.1, "LZ: S1 pattern inconsistent with RFR template, rise time less consistent than NR; P077: 6-7.6 sigma effects (toy)"),
        "V": T(1 - MSSI_VETO, 1 - 0.96, 1 - 0.92, "LZ 94 +- 2 %"),
        "T": T(1.0, 1.0, 1.0, "flat"),
    },
    "ER": {
        "rate": T(ER_TOT, ER_TOT, ER_TOT, "Table I: 1713 +- 39 science-sample events, ER-dominated"),
        "E": T(5.5e-6, 4e-6, 7e-6, "ER at S1c > 500 phd inside the ROI: 0.0106 - MSSI - acc - nu = 9.4e-3 of 1700"),
        "B": T(4.4e-5 / 9.4e-3, 7e-7 / 9.4e-3, 2.8e-3 / 9.4e-3,
               "leakage to d ~ -1.5 sigma_NR: P056 Gaussian 7e-7 (at event) .. P010 flat extrapolation of LZ's own tail 2.8e-3; LZ model: 0 below +1 sigma_NR"),
        "P": T(1.0, 1.0, 1.0, "bulk ER uniform in FV"),
        "W": T(1.0, 1.0, 1.0, "a genuine SS ER passes all pulse checks; PSD inconclusive (LZ; P077 LR <= 2-5)"),
        "V": T(1.0, 1.0, 1.0, ""),
        "T": T(1.0, 1.0, 3.0, "activation-linked (125I) sub-case: activity at +8.4 d is 0.2-0.4 of peak, 2-4x the run-average density (P010/P029)"),
    },
    "ACC": {
        "rate": T(ACC_TOT, ACC_TOT * 0.3, ACC_TOT * 3.0, "P022: 2.47 (Table I 2.6); UDT zero above 495 phd gives k_ML = 0, k < 183 (95 %); bracket x0.3-3"),
        "E": T(6.3e-4 / ACC_TOT, 2.0e-4, 3.0e-4, "P022: 6.3e-4 at S1c > 500 phd (isolated-S1 spectrum falls, lambda ~280 phd)"),
        "B": T(1.49e-4 / 6.3e-4, 0.20, 0.30, "P022: 1.49e-4 within +-2 sigma_NR"),
        "P": T(1.0, 1.0, 7.6, "uniform pairing -> 1; cathode-emission S2 slab (drift > 850 us) favours the event's z by 1/0.13 = 7.6 (P070 adverse)"),
        "W": T(0.1, 0.01, 1.0, "TBA/hit pattern consistent with z = 26 cm requires the isolated S1 to originate at that depth; isolated S1s are RFR/bottom-array dominated (P022); unknown whether LZ's cuts already enforce this"),
        "V": T(1.0, 1.0, 1.0, ""),
        "T": T(1.0, 1.0, 1.0, "flat"),
    },
    "NEU": {
        "rate": T(NEU_SS_UL / (1 - NEU_VETO), NEU_SS_UL / (1 - NEU_VETO), NEU_SS_UL / (1 - NEU_VETO),
                  "pre-veto detector-NR SS: LZ fit interval upper edge 0.118 / 0.08"),
        "E": T(6.2e-5, 2.7e-5, 4.4e-3, "P013: F(200-270 | ROI SS) = 5.1e-5 (alpha,n black disk) + muon-induced (P049 1.3e-5); stacked-conservative 5e-4 total -> 4.4e-3"),
        "B": T(0.95, 0.95, 0.95, "genuine NR"),
        "P": T(1 / 1.4, 1 / 1.7, 1 / 1.1, "P070: neutrons from surfaces, exp(-d/14 cm): LR 1.1-1.7"),
        "W": T(1.0, 1.0, 1.0, "genuine SS NR"),
        "V": T(1 - NEU_VETO, 1 - 0.96, 1 - 0.88, "LZ 92 +- 4 %"),
        "T": T(1.0, 1.0, 1.0, "41-min OD muon uninformative (P049)"),
    },
    "ART": {
        "rate": T(0.1, 0.01, 0.5, "pi_U: prior weight of an LZ-specific unmodelled class (P061 hyper-prior range)"),
        "E": T(0.006, 0.002, 0.01, "P041 residual artefact likelihood for an event-like charge-poor ER (joint E x B; channel priors are judgement)"),
        "B": T(1.0, 1.0, 1.0, "included in E (joint)"),
        "P": T(1.0, 1.0, 3.0, "uniform; lifetime-type channels favour long drift (event at 859 of 1049 us), P041"),
        "W": T(1.0, 1.0, 1.0, "clipping/truncation already excluded inside the P041 residual (alpha template)"),
        "V": T(1.0, 1.0, 1.0, ""),
        "T": T(1.0, 1.0, 1.0, "'no abnormalities' and monitor counts already inside P041"),
    },
    "NU": {
        "rate": T(NU_TOT, NU_TOT, NU_TOT, "Table I: 0.11 atm-nu + 0.057 8B/hep"),
        "E": T(3.4e-5 / NU_TOT, 1.0e-5 / NU_TOT, 5.0e-4 / NU_TOT, "P019: coherent tail 3.4e-5 at S1c > 500 (Fig. 5); incoherent Fermi-recoil channel up to 5e-4"),
        "B": T(0.95, 0.95, 0.95, "genuine NR"),
        "P": T(1.0, 1.0, 1.0, "uniform"),
        "W": T(1.0, 1.0, 1.0, "genuine SS NR"),
        "V": T(1.0, 1.0, 1.0, ""),
        "T": T(1.0, 1.0, 1.0, "flat"),
    },
}

# LZ's own model (case i): Table I / Fig. 5 as published, no corpus revision, no unmodelled classes
TAB_LZ = json.loads(json.dumps(TAB))
TAB_LZ["WALL"]["rate"] = T(LZ_WALL_SCI / (1 - MSSI_VETO), src="k = 1")
TAB_LZ["RFR"]["rate"] = T(LZ_RFR_SCI / (1 - MSSI_VETO), src="k = 1")
TAB_LZ["ACC"]["rate"] = T(ACC_TOT, src="k = 1")
TAB_LZ["ER"]["B"] = T(0.0, src="LZ Fig. 5 ER PDF is truncated at +1 sigma_NR: 0 below")
TAB_LZ["NEU"]["E"] = T(0.0, src="LZ Fig. 5 NR curve digitises to 0 at S1c > 500 phd (19F(alpha,n) endpoint 191 keV)")
TAB_LZ["ART"]["rate"] = T(0.0, src="not in Table I")
TAB_LZ["NU"]["E"] = T(3.4e-5 / NU_TOT, src="Fig. 5 coherent tail")


def central(tab, s=1.0, handles=HANDLES):
    """Return dict H -> {handle: value} of central values (rate of NR scaled by s)."""
    out = {}
    for H in HYP:
        out[H] = {h: tab[H][h]["c"] for h in HANDLES}
        if H == "NR":
            out[H]["rate"] *= s
        for h in HANDLES:
            if h not in handles:
                out[H][h] = 1.0 if h != "rate" else out[H][h]
    return out


def sample(tab, n, s=1.0, handles=HANDLES):
    """Log-uniform samples inside [lo, hi] for every factor; returns dict H -> (n, 7) array."""
    out = {}
    for H in HYP:
        arr = np.empty((n, len(HANDLES)))
        for j, h in enumerate(HANDLES):
            e = tab[H][h]
            if h not in handles:
                arr[:, j] = 1.0
                continue
            if e["lo"] == e["hi"] or e["lo"] <= 0:
                arr[:, j] = e["c"]
            else:
                arr[:, j] = np.exp(rng.uniform(np.log(e["lo"]), np.log(e["hi"]), n))
        if H == "NR":
            arr[:, 0] *= s
        out[H] = arr
    return out


def posterior_from(vals):
    """vals: H -> array (..., 7).  Returns H -> posterior array."""
    w = {H: np.prod(vals[H], axis=-1) for H in HYP}
    tot = sum(w.values())
    return {H: w[H] / tot for H in HYP}, w


def summarise(post):
    rows = []
    for H in HYP:
        p = post[H]
        rows.append(dict(H=H, median=np.median(p), p16=np.quantile(p, 0.16), p84=np.quantile(p, 0.84),
                         p05=np.quantile(p, 0.05), p95=np.quantile(p, 0.95)))
    return pd.DataFrame(rows).set_index("H")


results = {}

# --------------------------------------------------------------------------------------
# 2. Neighbourhood expectations and the (S1c, d) likelihood ratio (cross-check with P052)
# --------------------------------------------------------------------------------------
c_ii = central(TAB)
c_i = central(TAB_LZ)
nb = {case: {H: c[H]["rate"] * c[H]["E"] * c[H]["B"] * c[H]["V"] for H in HYP} for case, c in [("LZ", c_i), ("corpus", c_ii)]}
b_LZ = sum(v for H, v in nb["LZ"].items() if H != "NR")
b_corpus = sum(v for H, v in nb["corpus"].items() if H != "NR")
LR_nb_per_signal_LZ = nb["LZ"]["NR"] / b_LZ
# event-bin version: NR 0.21 x 0.070 (LZ L10 d-curve fraction in [-2,-1.5], P052) vs Fig. 5 total 3.95e-5 (P004)
LR_bin_per_signal = 0.21 * 0.070 / 3.95e-5
results["neighbourhood"] = dict(nb_LZ=nb["LZ"], nb_corpus=nb["corpus"], b_LZ=b_LZ, b_corpus=b_corpus,
                                LR_pm2sigma_per_signal_event_LZ=LR_nb_per_signal_LZ,
                                LR_eventbin_per_signal_event=LR_bin_per_signal, P052_LR_ev=265.0,
                                P070_b=3.6e-4)
print("Neighbourhood (S1c>500, |d|<=2 sigma) expectations, LZ model:", {k: f"{v:.3g}" for k, v in nb["LZ"].items()})
print("  sum bkg LZ = %.3g (P070 used 3.6e-4); corpus central = %.3g" % (b_LZ, b_corpus))
print("  (S1c,d)-only LR per signal event: +-2sigma %.0f, event bin %.0f (P052: 265)" % (LR_nb_per_signal_LZ, LR_bin_per_signal))

# --------------------------------------------------------------------------------------
# 3. Posteriors: cases (i-a) LZ counts only, (i-b) LZ counts + corpus handles, (ii) corpus
# --------------------------------------------------------------------------------------
CASES = {
    "i-a LZ counts, (S1c,d) only": (TAB_LZ, ["rate", "E", "B", "V"]),
    "i-b LZ counts + topological handles": (TAB_LZ, HANDLES),
    "ii corpus counts + handles": (TAB, HANDLES),
}
S_SCAN = [0.001, 0.01, 0.105, 1.0, 2.4]   # 0.105 = LZ 90 % lower edge (P045); 2.4 = upper; 1e-3..1e-2 ~ prior-weighted expectation (P061)
post_tables = {}
for name, (tab, hs) in CASES.items():
    for s in S_SCAN:
        cv = central(tab, s=s, handles=hs)
        pc, wc = posterior_from({H: np.array([cv[H][h] for h in HANDLES]) for H in HYP})
        smp = sample(tab, N_MC, s=s, handles=hs)
        pm, wm = posterior_from(smp)
        df = summarise(pm)
        df["central"] = [pc[H] for H in HYP]
        df["weight_central"] = [wc[H] for H in HYP]
        df["case"] = name
        df["s"] = s
        post_tables[(name, s)] = df
        if s == 1.0:
            print(f"\n== {name}, s = 1 ==")
            print(df[["central", "median", "p16", "p84"]].to_string(float_format=lambda x: f"{x:.3g}"))
post_df = pd.concat(post_tables.values()).reset_index()
post_df.to_csv(os.path.join(OUT, "P093_posteriors.csv"), index=False)

# background-only posterior (conditional on 'not NR'): independent of s
bk_rows = []
for name, (tab, hs) in CASES.items():
    smp = sample(tab, N_MC, s=1.0, handles=hs)
    w = {H: np.prod(smp[H], axis=-1) for H in HYP if H != "NR"}
    tot = sum(w.values())
    cv = central(tab, handles=hs)
    wc = {H: np.prod([cv[H][h] for h in HANDLES]) for H in HYP if H != "NR"}
    totc = sum(wc.values())
    for H in w:
        p = w[H] / tot
        bk_rows.append(dict(case=name, H=H, central=wc[H] / totc, median=np.median(p),
                            p16=np.quantile(p, 0.16), p84=np.quantile(p, 0.84)))
bk_df = pd.DataFrame(bk_rows)
bk_df.to_csv(os.path.join(OUT, "P093_background_only_posterior.csv"), index=False)
print("\n== background-only posterior (given not NR) ==")
print(bk_df.pivot(index="H", columns="case", values="central").to_string(float_format=lambda x: f"{x:.3g}"))

# P(NR) vs s for the three cases (central and MC)
s_grid = np.logspace(-3.5, 1, 37)
pnr_rows = []
break_even = {}
for name, (tab, hs) in CASES.items():
    smp = sample(tab, N_MC, s=1.0, handles=hs)
    w_b = sum(np.prod(smp[H], axis=-1) for H in HYP if H != "NR")
    w_nr = np.prod(smp["NR"], axis=-1)
    cv = central(tab, handles=hs)
    wc_b = sum(np.prod([cv[H][h] for h in HANDLES]) for H in HYP if H != "NR")
    wc_nr = np.prod([cv["NR"][h] for h in HANDLES])
    sbe = w_b / w_nr   # s at which P(NR) = 1/2
    break_even[name] = dict(central=float(wc_b / wc_nr), median=float(np.median(sbe)), p16=float(np.quantile(sbe, 0.16)), p84=float(np.quantile(sbe, 0.84)))
    for s in s_grid:
        p = s * w_nr / (s * w_nr + w_b)
        pnr_rows.append(dict(case=name, s=s, central=s * wc_nr / (s * wc_nr + wc_b),
                             median=np.median(p), p16=np.quantile(p, 0.16), p84=np.quantile(p, 0.84)))
pnr_df = pd.DataFrame(pnr_rows)
pnr_df.to_csv(os.path.join(OUT, "P093_PNR_vs_s.csv"), index=False)
results["break_even_s"] = break_even
print("\n== break-even signal expectation s_1/2 (P(NR) = 0.5) ==")
for k, v in break_even.items():
    print(f"  {k:40s} central {v['central']:.2e}  MC median {v['median']:.2e} [{v['p16']:.2e}, {v['p84']:.2e}]")

# --------------------------------------------------------------------------------------
# 4. Odds matrix (central, case ii, s = 1) and MC bands on log10 odds
# --------------------------------------------------------------------------------------
cv = central(TAB, s=1.0)
wc = {H: np.prod([cv[H][h] for h in HANDLES]) for H in HYP}
odds = pd.DataFrame({H2: {H1: wc[H1] / wc[H2] for H1 in HYP} for H2 in HYP})  # odds[H1 row, H2 col] = P(H1)/P(H2)
odds.to_csv(os.path.join(OUT, "P093_odds_matrix_central.csv"))
smp = sample(TAB, N_MC, s=1.0)
w = {H: np.prod(smp[H], axis=-1) for H in HYP}
lo_rows = []
for H1, H2 in itertools.combinations(HYP, 2):
    lo = np.log10(w[H1] / w[H2])
    lo_rows.append(dict(H1=H1, H2=H2, log10_odds_central=np.log10(wc[H1] / wc[H2]), log10_odds_median=np.median(lo),
                        p16=np.quantile(lo, 0.16), p84=np.quantile(lo, 0.84), p05=np.quantile(lo, 0.05), p95=np.quantile(lo, 0.95)))
odds_mc = pd.DataFrame(lo_rows)
odds_mc.to_csv(os.path.join(OUT, "P093_odds_pairs_mc.csv"), index=False)
# same for LZ case (i-b)
cv_i = central(TAB_LZ, s=1.0)
wci = {H: np.prod([cv_i[H][h] for h in HANDLES]) for H in HYP}
odds_i = pd.DataFrame({H2: {H1: (wci[H1] / wci[H2] if wci[H2] > 0 else np.inf) for H1 in HYP} for H2 in HYP})
odds_i.to_csv(os.path.join(OUT, "P093_odds_matrix_LZ_ib.csv"))
print("\n== log10 odds matrix, central, case ii, s = 1 (row / column) ==")
print(np.log10(odds).to_string(float_format=lambda x: f"{x:+.2f}"))

# --------------------------------------------------------------------------------------
# 5. Most discriminating handle for each pair (central values, case ii)
# --------------------------------------------------------------------------------------
pair_rows = []
for H1, H2 in itertools.combinations(HYP, 2):
    contrib = {h: np.log10(cv[H1][h] / cv[H2][h]) for h in HANDLES}
    tops = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))
    topo = {h: v for h, v in contrib.items() if h != "rate"}
    top_topo = max(topo.items(), key=lambda kv: abs(kv[1]))
    pair_rows.append(dict(H1=H1, H2=H2, total_log10_odds=sum(contrib.values()), top_handle=tops[0][0], top_log10=tops[0][1],
                          top_topological_handle=top_topo[0], top_topological_log10=top_topo[1],
                          **{f"d_{h}": contrib[h] for h in HANDLES}))
pairs_df = pd.DataFrame(pair_rows)
pairs_df.to_csv(os.path.join(OUT, "P093_pair_discriminants.csv"), index=False)
print("\n== most discriminating handle per pair (case ii central) ==")
print(pairs_df[["H1", "H2", "total_log10_odds", "top_handle", "top_log10", "top_topological_handle", "top_topological_log10"]]
      .to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

# handle table export
rows = []
for H in HYP:
    for h in HANDLES:
        e = TAB[H][h]
        rows.append(dict(H=H, handle=h, central=e["c"], lo=e["lo"], hi=e["hi"], LZ_model=TAB_LZ[H][h]["c"], source=e["src"]))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "P093_handle_table.csv"), index=False)

# --------------------------------------------------------------------------------------
# 6. Which unpublished handle would change the odds most?  Binary-outcome model:
#    outcome o in {bkg-like, NR-like}; P(o = bkg-like | H in target) = p0, else p0 / LR.
#    Expected information gain I(O; H) under the current posterior (case ii, s = 1, MC).
# --------------------------------------------------------------------------------------
UNPUB = {
    "S1 pulse shape / photon timing (ER vs NR template LR)": (["WALL", "RFR", "ER", "ACC", "ART"], 3.0, 2.0, 5.0, "P077: LR <= 2-5 after LZ's AmBe-vs-ER statement"),
    "S2 width vs drift template (cathode-emission S2)": (["ACC"], 5.0, 3.0, 20.0, "P022: cathode S2 1.65 vs 1.49 us (2.4 sigma_stat)"),
    "S1 TBA + per-PMT hit pattern vs wall-MSSI / isolated-S1 templates": (["WALL", "ACC"], 1.5, 1.0, 3.0, "LZ: no discrimination found; P077: TBA sigma 0.035"),
    "(x, y) within FV vs gamma-source map (azimuth, PMT hot spots, 57Co tube)": (["WALL", "RFR"], 2.0, 1.0, 3.0, "P070: wall LR spans 2-50 with the source; P033 source maps"),
    "hour-resolved e-lifetime, 83mKr S2c-map cell at (x,y), per-PMT S2 footprint": (["ART"], 2.0, 1.3, 5.0, "P041: testable channels are 1.6e-3 of the 6e-3 residual"),
    "waveform-level release of all of the above": (["WALL", "RFR", "ER", "ACC", "ART"], 4.0, 2.5, 10.0, "product of the above, correlated"),
}
p0 = 0.8
unp_rows = []
S_EIG = [1.0, 0.01, float(break_even["ii corpus counts + handles"]["central"])]
for s_eig, (name, (targets, LRc, LRlo, LRhi, src)) in itertools.product(S_EIG, UNPUB.items()):
    smp = sample(TAB, N_MC, s=s_eig)
    post_mc, _ = posterior_from(smp)
    LRs = np.exp(rng.uniform(np.log(LRlo), np.log(LRhi), N_MC))
    for lab, LR in [("central", np.full(N_MC, LRc)), ("mc", LRs)]:
        pB = {H: (p0 if H in targets else p0 / LR) for H in HYP}
        P_B = sum(post_mc[H] * pB[H] for H in HYP)
        P_N = 1 - P_B
        eig = np.zeros(N_MC)
        for H in HYP:
            for o, po in [("B", pB[H]), ("N", 1 - pB[H])]:
                Po = P_B if o == "B" else P_N
                with np.errstate(divide="ignore", invalid="ignore"):
                    term = post_mc[H] * po * np.log(po / Po)
                eig += np.nan_to_num(term)
        pNR_now = post_mc["NR"]
        pNR_if_N = post_mc["NR"] * (1 - pB["NR"]) / P_N
        pNR_if_B = post_mc["NR"] * pB["NR"] / P_B
        unp_rows.append(dict(handle=name, variant=lab, s=s_eig, targets="+".join(targets), LR_central=LRc, LR_lo=LRlo, LR_hi=LRhi,
                             EIG_nats_median=np.median(eig), EIG_p16=np.quantile(eig, 0.16), EIG_p84=np.quantile(eig, 0.84),
                             PNR_now_median=np.median(pNR_now), PNR_if_NRlike_median=np.median(pNR_if_N),
                             PNR_if_bkglike_median=np.median(pNR_if_B),
                             dlog10_odds_NR_if_NRlike=np.median(np.log10((pNR_if_N / (1 - pNR_if_N)) / (pNR_now / (1 - pNR_now)))),
                             dlog10_odds_NR_if_bkglike=np.median(np.log10((pNR_if_B / (1 - pNR_if_B)) / (pNR_now / (1 - pNR_now)))),
                             source=src))
unp_df = pd.DataFrame(unp_rows)
unp_df.to_csv(os.path.join(OUT, "P093_unpublished_handles.csv"), index=False)
print("\n== value of unpublished handles (case ii) ==")
print(unp_df[unp_df.variant == "mc"][["s", "handle", "EIG_nats_median", "PNR_now_median", "PNR_if_NRlike_median", "PNR_if_bkglike_median",
                                      "dlog10_odds_NR_if_NRlike", "dlog10_odds_NR_if_bkglike"]]
      .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
# entropy of the hypothesis posterior at the break-even s, for scale
smp = sample(TAB, N_MC, s=S_EIG[2]); post_mc, _ = posterior_from(smp)
H_post = np.median(-sum(np.nan_to_num(post_mc[H] * np.log(np.clip(post_mc[H], 1e-300, None))) for H in HYP))
results["posterior_entropy_nats_median_at_break_even"] = float(H_post)
print("posterior entropy at break-even s: %.3f nats" % H_post)

# --------------------------------------------------------------------------------------
# 7. Variants: inelastic timing bonus, conservative ER (P061 value), no ART class, no ACC slab loophole
# --------------------------------------------------------------------------------------
variants = {}


def run_variant(tab, label, s=1.0):
    smp_v = sample(tab, N_MC, s=s)
    pm_v, _ = posterior_from(smp_v)
    cv_v = central(tab, s=s)
    pc_v, _ = posterior_from({H: np.array([cv_v[H][h] for h in HANDLES]) for H in HYP})
    variants[label] = dict(PNR_central=pc_v["NR"], PNR_median=float(np.median(pm_v["NR"])),
                           PNR_p16=float(np.quantile(pm_v["NR"], 0.16)), PNR_p84=float(np.quantile(pm_v["NR"], 0.84)),
                           top_bkg_central=max(((H, pc_v[H]) for H in HYP if H != "NR"), key=lambda kv: kv[1]))


run_variant(TAB, "baseline")
tv = json.loads(json.dumps(TAB)); tv["NR"]["T"] = T(2.1, 1.4, 2.6, "P006 inelastic delta 300-380 keV"); run_variant(tv, "inelastic timing bonus 1.4-2.6")
tv = json.loads(json.dumps(TAB)); tv["ER"]["B"] = T(2.8e-3 / 9.4e-3, src="P061 conservative 2.8e-3"); run_variant(tv, "ER at P010/P061 upper 2.8e-3")
tv = json.loads(json.dumps(TAB)); tv["ER"]["B"] = T(7e-7 / 9.4e-3, src="P056 Gaussian 7e-7"); run_variant(tv, "ER at P056 Gaussian 7e-7")
tv = json.loads(json.dumps(TAB)); tv["ART"]["rate"] = T(0.0, src="no artefact class"); run_variant(tv, "no artefact class")
tv = json.loads(json.dumps(TAB)); tv["ACC"]["P"] = T(1.0, src="no slab"); tv["ACC"]["W"] = T(1.0, src="TBA cut assumed already applied"); run_variant(tv, "ACC: no slab loophole, no TBA penalty")
tv = json.loads(json.dumps(TAB)); tv["WALL"]["P"] = T(1 / 2.3, src="bottom-source map"); tv["WALL"]["rate"] = T(LZ_WALL_SCI / 0.06 * 1.64, src="k = 1.64"); run_variant(tv, "WALL adverse: k = 1.64, bottom-source LR 2.3")
tv = json.loads(json.dumps(TAB)); tv["NEU"]["E"] = T(4.4e-3, src="P013 stacked-conservative"); run_variant(tv, "NEU stacked-conservative 5e-4")
results["variants"] = variants
print("\n== variants (P(NR) at s = 1) ==")
for k, v in variants.items():
    print(f"  {k:45s} central {v['PNR_central']:.3f}  MC median {v['PNR_median']:.3f} [{v['PNR_p16']:.3f}, {v['PNR_p84']:.3f}]  top bkg {v['top_bkg_central'][0]} {v['top_bkg_central'][1]:.3f}")

# --------------------------------------------------------------------------------------
# 8. Figures (palette: dataviz reference instance, fixed categorical order; thin marks)
# --------------------------------------------------------------------------------------
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
COL = dict(zip(HYP, PAL))
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})

# Fig 1: ranked posterior, three cases, s = 1, with 68 % bands
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=False)
for ax, (name, _) in zip(axes, CASES.items()):
    df = post_tables[(name, 1.0)].copy()
    df = df.sort_values("median", ascending=True)
    y = np.arange(len(df))
    vals = df["median"].clip(lower=1e-12).values
    lo_e = np.maximum(vals - df["p16"].clip(lower=1e-12).values, 0)
    hi_e = np.maximum(df["p84"].clip(lower=1e-12).values - vals, 0)
    ax.barh(y, vals, color=[COL[H] for H in df.index], height=0.55, left=1e-12)
    ax.errorbar(vals, y, xerr=[lo_e, hi_e], fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
    ax.scatter(df["central"].clip(lower=1e-12), y, marker="|", color=INK, s=60, zorder=3, label="central")
    ax.set_xscale("log"); ax.set_xlim(1e-9, 1.5)
    ax.set_yticks(y); ax.set_yticklabels([HYP_LABEL[H] for H in df.index], fontsize=8)
    ax.set_title(name, fontsize=9, color=INK)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("posterior P(H | event), s = 1")
axes[0].legend(loc="lower right", frameon=False, fontsize=8)
fig.suptitle("Classification posterior over event origins: MC median with 68 % band (log-uniform ranges), tick = central value", fontsize=9, color=INK2)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "P093_fig1_posteriors.png"), dpi=170); plt.close(fig)

# Fig 2: log10 odds matrix (case ii central) as a diverging heat map
lo_mat = np.log10(odds.values.astype(float))
cmap = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f0efec", "#2a78d6"])
fig, ax = plt.subplots(figsize=(6.2, 5.2))
im = ax.imshow(lo_mat, cmap=cmap, norm=TwoSlopeNorm(vcenter=0, vmin=-6, vmax=6))
ax.set_xticks(range(len(HYP))); ax.set_yticks(range(len(HYP)))
ax.set_xticklabels(HYP, fontsize=8); ax.set_yticklabels(HYP, fontsize=8)
for i in range(len(HYP)):
    for j in range(len(HYP)):
        if i != j:
            ax.text(j, i, f"{lo_mat[i, j]:+.1f}", ha="center", va="center", fontsize=7.5, color=INK)
ax.set_xlabel("column hypothesis H2"); ax.set_ylabel("row hypothesis H1")
ax.set_title("log10 odds P(H1)/P(H2), corpus counts + handles, s = 1 (central)", fontsize=9)
cb = fig.colorbar(im, ax=ax, fraction=0.046); cb.set_label("log10 odds (blue: row favoured)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "P093_fig2_odds_matrix.png"), dpi=170); plt.close(fig)

# Fig 3: handle contributions log10 L(handle|H)/L(handle|NR), case ii central, s = 1
fig, ax = plt.subplots(figsize=(9, 3.8))
hyps = [H for H in HYP if H != "NR"]
x = np.arange(len(hyps)); wdt = 0.11
HCOL = dict(zip(HANDLES, PAL[:7]))
for j, h in enumerate(HANDLES):
    vals = [np.log10(cv[H][h] / cv["NR"][h]) for H in hyps]
    ax.bar(x + (j - 3) * wdt, vals, width=wdt * 0.85, color=HCOL[h], label={"rate": "rate mu_ROI", "E": "E: S1c>500", "B": "B: |d|<2 sigma", "P": "P: position", "W": "W: pulse shape", "V": "V: veto", "T": "T: timing"}[h])
tot = [np.log10(wc[H] / wc["NR"]) for H in hyps]
ax.scatter(x, tot, marker="D", color=INK, s=22, zorder=4, label="total log10 odds vs NR")
ax.axhline(0, color="#c3c2b7", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([HYP_LABEL[H] for H in hyps], fontsize=8, rotation=12)
ax.set_ylabel("log10 [ L(handle | H) / L(handle | NR) ]"); ax.set_ylim(-7.5, 4)
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
ax.legend(ncol=4, fontsize=7.5, frameon=False, loc="lower left")
ax.set_title("Per-handle likelihood ratios relative to the single-NR hypothesis (corpus central values, s = 1)", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "P093_fig3_handle_contributions.png"), dpi=170); plt.close(fig)

# Fig 4: P(NR) vs s for the three cases
fig, ax = plt.subplots(figsize=(5.6, 3.6))
for k, (name, _) in enumerate(CASES.items()):
    d = pnr_df[pnr_df.case == name]
    ax.plot(d.s, d["median"], color=PAL[k], lw=2, label=name)
    ax.fill_between(d.s, d.p16, d.p84, color=PAL[k], alpha=0.18, lw=0)
ax.set_xscale("log"); ax.set_xlabel("expected DM-like NR events in the WS ROI, s"); ax.set_ylabel("P(single NR | event)")
ax.axvline(1.0, color="#c3c2b7", lw=0.8, ls="--"); ax.axvline(0.105, color="#c3c2b7", lw=0.8, ls=":")
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
ax.legend(fontsize=7.5, frameon=False, loc="lower right")
ax.set_title("P(NR) versus the assumed signal expectation (band: 68 % MC)", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "P093_fig4_PNR_vs_s.png"), dpi=170); plt.close(fig)

# --------------------------------------------------------------------------------------
# 9. Save summary JSON
# --------------------------------------------------------------------------------------
summ = {}
for (name, s), df in post_tables.items():
    summ[f"{name} | s={s}"] = {H: dict(central=float(df.loc[H, "central"]), median=float(df.loc[H, "median"]),
                                        p16=float(df.loc[H, "p16"]), p84=float(df.loc[H, "p84"])) for H in HYP}
results["posteriors"] = summ
results["odds_matrix_central_ii"] = odds.to_dict()
results["unpublished_handles"] = {f"{r.handle} | s={r.s:.3g}": dict(EIG=r.EIG_nats_median, PNR_now=r.PNR_now_median, PNR_if_NRlike=r.PNR_if_NRlike_median,
                                                                     PNR_if_bkglike=r.PNR_if_bkglike_median, dlog10_if_NRlike=r.dlog10_odds_NR_if_NRlike,
                                                                     dlog10_if_bkglike=r.dlog10_odds_NR_if_bkglike)
                                  for r in unp_df[unp_df.variant == "mc"].itertuples()}
results["N_MC"] = N_MC
results["lz_check"] = dict(ev_S1c=lz.LZ["ev_S1c"], ev_S2c=lz.LZ["ev_S2c"], MSSI_veto=lz.LZ["MSSI_veto_eff"], neutron_veto=lz.LZ["neutron_veto_eff"])
with open(os.path.join(OUT, "P093_results.json"), "w") as f:
    json.dump(results, f, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))
print("\nwrote", OUT)
