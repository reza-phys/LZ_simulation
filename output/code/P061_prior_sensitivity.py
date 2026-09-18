#!/usr/bin/env python
"""P061 -- Prior sensitivity of the dark-matter posterior for the LZ 248 keV event.

Three-hypothesis posterior (P001 form) with corpus-updated inputs:
  H_DM : L_DM = (B_DM / f_FP) * b_eff        (P027 model-marginalised Bayes factor, forking-paths penalty)
  H_B  : L_B  = b_eff                        (modelled background in the event's neighbourhood, P016 anchor)
  H_U  : L_U  = L_acc + L_rest               (accounted residual channels from the corpus + unaccounted free parameter)
  P(DM | event) = pi_DM L_DM / (pi_DM L_DM + pi_B L_B + pi_U L_U),  pi_B = 1 - pi_DM - pi_U.

Outputs (output/work/P061/): JSON/CSV tables and three PNG figures.
Run from the simulation root:  .venv/bin/python output/code/P061_prior_sensitivity.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

sys.path.insert(0, "output/code")
from common import lzcommon as lz  # noqa: E402

OUT = "output/work/P061"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(20260912)

# ----------------------------------------------------------------------------------------------
# 0. Inputs (all from the corpus ledger / papers; see details.md section 2 for sources)
# ----------------------------------------------------------------------------------------------
INP = dict(
    b_eff=5.7e-4,            # P016/P027 anchor: modelled background giving 3.4 sigma local for L10
    b_eff_range=(2.0e-4, 1.1e-3),   # P001: exact-Poisson to asymptotic bracket
    B_DM_central=16.4,       # P027 uniform-over-spectra, log-uniform s
    B_DM_class=28.7,         # P027 class priors
    B_DM_range=(6.6, 45.0),  # P027 uniform-s prior (6.6) to continuous-delta class B (45.7)
    N_eff=12.2,              # P008
    p_global=4.7e-3,         # LZ paper (2.6 sigma global)
    exposure_tyr=lz.LZ.get("exposure_tyr", 2.84),
    k_next=2.38,             # P020: 6.76 t yr / 2.84 t yr
    k_xenon=(3.1 + 1.54) / 2.84,   # XENONnT 3.1 + PandaX-4T 1.54 t yr (dossier, P005)
)
B_SBB = -1.0 / (np.e * INP["p_global"] * np.log(INP["p_global"]))   # Sellke-Bayarri-Berger cap

# Residual (unmodelled-excess) likelihood per accounted channel: expected event-like events per
# 2.84 t yr under the channel hypothesis at the corpus's central / 95%-allowed mismodelling.
CHANNELS = pd.DataFrame([
    # channel, central, upper, source
    ("wall/RFR MSSI at the event's position class (k = 1 / k = 1.64)", 2.0e-6, 3.3e-6, "P033 (2e-6 x k), P004 (k < 1.64 at 95%)"),
    ("accidental coincidence, high-S1 tail (as modelled / k < 183)", 1.5e-4, 2.7e-2, "P022 (1.5e-4 in +-2 sigma_NR at S1c > 500; UDT k < 183 at 95%)"),
    ("ER leakage incl. 124Xe/125I double vacancy (flat tail extrapolation / max)", 2.8e-3, 1.0e-2, "P010 (2.8e-3 extrapolation; <= 1e-2)"),
    ("neutrons of any origin (central / every conservative choice stacked)", 6.6e-6, 5.0e-4, "P013 (6.6e-6; <= 5e-4), P049 (1.3e-5 muon-induced)"),
    ("detector artefact / partial charge loss (P041 residual / x3)", 6.0e-3, 2.0e-2, "P041 residual 0.006 (upper bracket x3 is our judgement)"),
    ("AmBe/57Co activation, MSSI-like (per run / upper)", 1.5e-4, 8.0e-3, "P029 (1.5e-4 per run; 9e-6 - 8e-3)"),
    ("atmospheric-nu incoherent lone NR (central / upper)", 1.7e-4, 5.0e-4, "P019 (1.7e-4; 8e-6 - 5e-4)"),
    ("nuclear-excitation hybrid, boosted/fast light DM", 0.0, 0.0, "P036, P040 (excluded)"),
], columns=["channel", "L_central", "L_upper", "source"])
L_ACC = float(CHANNELS.L_central.sum())
L_ACC_UP = float(CHANNELS.L_upper.sum())
CHANNELS.to_csv(os.path.join(OUT, "P061_table1_channels.csv"), index=False)

# Community hyper-prior (log-uniform boxes)
HYPER = dict(pi_DM=(1e-3, 0.3), pi_U=(0.01, 0.5), L_rest=(1e-3, 0.3), f_FP=(1.0, 10.0), B_DM=INP["B_DM_range"])
CENTRAL = dict(pi_U=0.1, f_FP=3.0, B_DM=INP["B_DM_central"], b=INP["b_eff"], L_acc=L_ACC)


# ----------------------------------------------------------------------------------------------
# 1. Posterior
# ----------------------------------------------------------------------------------------------
def posterior(pi_DM, pi_U, L_rest, f_FP=3.0, B_DM=16.4, b=5.7e-4, L_acc=L_ACC):
    """Return (P_DM, P_B, P_U) for the three-hypothesis model (arrays broadcast)."""
    pi_DM, pi_U, L_rest, f_FP, B_DM = map(np.asarray, (pi_DM, pi_U, L_rest, f_FP, B_DM))
    pi_B = 1.0 - pi_DM - pi_U
    w_DM = pi_DM * (B_DM / f_FP) * b
    w_B = pi_B * b
    w_U = pi_U * (L_acc + L_rest)
    Z = w_DM + w_B + w_U
    return w_DM / Z, w_B / Z, w_U / Z


def h2(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


# Validation against P001's three-hypothesis numbers (B_DM -> 1 + E/b with E = 0.100, L_acc = 0)
val = {}
for (pu, lu, ref) in [(0.01, 0.1, 0.46), (0.1, 0.1, 0.09)]:
    p = posterior(0.01, pu, lu, f_FP=1.0, B_DM=1 + 0.100 / 2e-4, b=2e-4, L_acc=0.0)[0]
    val[f"P001_3hyp_piU{pu}_LU{lu}"] = dict(ours=float(p), P001=ref)
# P001 two-hypothesis (pi_U = 0): P(DM) = 0.84 for pi_DM = 0.01, b = 2e-4, B = 501
val["P001_2hyp_piDM0.01_b2e-4"] = dict(ours=float(posterior(0.01, 0.0, 0.0, 1.0, 501.0, 2e-4, 0.0)[0]), P001=0.84)

# ----------------------------------------------------------------------------------------------
# 2. Grid map in (pi_DM, L_rest) at central other values (+ optimistic panel)
# ----------------------------------------------------------------------------------------------
pi_grid = np.geomspace(1e-3, 0.3, 241)
Lr_grid = np.geomspace(1e-3, 0.3, 241)
PI, LR = np.meshgrid(pi_grid, Lr_grid, indexing="ij")
MAP_central = posterior(PI, 0.1, LR, **{k: CENTRAL[k] for k in ("f_FP", "B_DM", "b", "L_acc")})[0]
OPT = dict(pi_U=0.05, f_FP=1.0, B_DM=INP["B_DM_class"], b=INP["b_eff"], L_acc=L_ACC)
MAP_opt = posterior(PI, OPT["pi_U"], LR, OPT["f_FP"], OPT["B_DM"], OPT["b"], OPT["L_acc"])[0]
PES = dict(pi_U=0.3, f_FP=10.0, B_DM=INP["B_DM_range"][0], b=INP["b_eff"], L_acc=L_ACC_UP)
MAP_pes = posterior(PI, PES["pi_U"], LR, PES["f_FP"], PES["B_DM"], PES["b"], PES["L_acc"])[0]
np.savez(os.path.join(OUT, "P061_map.npz"), pi_DM=pi_grid, L_rest=Lr_grid, P_central=MAP_central, P_opt=MAP_opt, P_pes=MAP_pes)


def pi_needed(P_target, pi_U, L_rest, f_FP, B_DM, b, L_acc):
    """pi_DM at which P(DM) = P_target (analytic: pi_DM L_DM (1-P) = P [ (1-pi_DM-pi_U) b + pi_U L_U ])."""
    L_DM = B_DM / f_FP * b
    L_U = L_acc + L_rest
    # pi_DM [L_DM (1-P) + P b] = P [ (1 - pi_U) b + pi_U L_U ]
    return P_target * ((1 - pi_U) * b + pi_U * L_U) / (L_DM * (1 - P_target) + P_target * b)


bound = {}
for tag, cfg in [("central", CENTRAL), ("optimistic", OPT), ("pessimistic", PES)]:
    bound[tag] = {}
    for Lr in [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]:
        bound[tag][f"L_rest={Lr:g}"] = {f"pi_DM(P={P})": float(pi_needed(P, cfg["pi_U"], Lr, cfg["f_FP"], cfg["B_DM"], cfg["b"], cfg["L_acc"]))
                                        for P in (0.1, 0.5, 0.9)}
# Fraction of the (pi_DM, L_rest) log-area in each class at central values
area = {}
for tag, M in [("central", MAP_central), ("optimistic", MAP_opt), ("pessimistic", MAP_pes)]:
    area[tag] = dict(gt_0p5=float((M > 0.5).mean()), between=float(((M >= 0.1) & (M <= 0.5)).mean()), lt_0p1=float((M < 0.1).mean()))

# ----------------------------------------------------------------------------------------------
# 3. Monte Carlo over the community hyper-prior
# ----------------------------------------------------------------------------------------------
N = 400_000
def draw(N, fixed=None):
    th = {k: np.exp(rng.uniform(np.log(lo), np.log(hi), N)) for k, (lo, hi) in HYPER.items()}
    if fixed:
        for k, v in fixed.items():
            th[k] = np.full(N, v)
    return th

TH = draw(N)
P_DM, P_B, P_U = posterior(TH["pi_DM"], TH["pi_U"], TH["L_rest"], TH["f_FP"], TH["B_DM"], INP["b_eff"], L_ACC)


def summary(p):
    q = np.quantile(p, [0.025, 0.16, 0.5, 0.84, 0.975])
    return dict(median=float(q[2]), q16=float(q[1]), q84=float(q[3]), q2p5=float(q[0]), q97p5=float(q[4]), mean=float(p.mean()),
                frac_gt_0p5=float((p > 0.5).mean()), frac_0p1_0p5=float(((p >= 0.1) & (p <= 0.5)).mean()), frac_lt_0p1=float((p < 0.1).mean()),
                mean_binary_entropy_nats=float(h2(p).mean()))

MC = {"community_prior": summary(P_DM), "P_U": summary(P_U), "P_B": summary(P_B)}
# Variants
variants = {
    "f_FP=1 (no forking-paths penalty)": dict(fixed=dict(f_FP=1.0)),
    "f_FP=10": dict(fixed=dict(f_FP=10.0)),
    "B_DM=28.7 fixed (class priors)": dict(fixed=dict(B_DM=28.7)),
    "B_DM=6.6 fixed (uniform-s)": dict(fixed=dict(B_DM=6.6)),
    "pi_DM=0.2 fixed (dossier DM total)": dict(fixed=dict(pi_DM=0.2)),
    "pi_DM=0.01 fixed": dict(fixed=dict(pi_DM=0.01)),
    "L_rest=1e-3 fixed": dict(fixed=dict(L_rest=1e-3)),
    "L_rest=0.1 fixed": dict(fixed=dict(L_rest=0.1)),
    "pi_U=0.01 fixed": dict(fixed=dict(pi_U=0.01)),
    "pi_U=0.5 fixed": dict(fixed=dict(pi_U=0.5)),
}
for name, v in variants.items():
    t = draw(N // 2, v["fixed"])
    MC[name] = summary(posterior(t["pi_DM"], t["pi_U"], t["L_rest"], t["f_FP"], t["B_DM"], INP["b_eff"], L_ACC)[0])
for name, kw in {"L_acc upper bracket (0.066)": dict(L_acc=L_ACC_UP), "L_acc = 0 (no accounted channels)": dict(L_acc=0.0),
                 "b_eff=2e-4 (L_DM = B_DM b_H held fixed)": dict(b=2e-4), "b_eff=1.1e-3 (L_DM held fixed)": dict(b=1.1e-3)}.items():
    t = draw(N // 2)
    b_var = kw.get("b", INP["b_eff"])
    # P027's B_DM scales as 1/b_H, i.e. L_DM = B_DM b_H is the anchored quantity; only L_B = b changes here
    MC[name] = summary(posterior(t["pi_DM"], t["pi_U"], t["L_rest"], t["f_FP"], t["B_DM"] * INP["b_eff"] / b_var, b_var, kw.get("L_acc", L_ACC))[0])
# Uniform (not log-uniform) hyper-prior as a check
t = {k: rng.uniform(lo, hi, N // 2) for k, (lo, hi) in HYPER.items()}
MC["uniform (linear) hyper-prior"] = summary(posterior(t["pi_DM"], t["pi_U"], t["L_rest"], t["f_FP"], t["B_DM"], INP["b_eff"], L_ACC)[0])

# First-order variance-based sensitivity indices (binned conditional means) on logit P(DM)
def first_order(TH, y, nb=25):
    out = {}
    V = y.var()
    for k, x in TH.items():
        edges = np.quantile(x, np.linspace(0, 1, nb + 1))
        idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, nb - 1)
        cm = np.bincount(idx, weights=y, minlength=nb) / np.bincount(idx, minlength=nb)
        out[k] = float(cm[idx].var() / V)
    return out
logit = np.log(P_DM / (1 - P_DM))
SENS = dict(logit=first_order(TH, logit), linear=first_order(TH, P_DM))
# Analytic elasticities of the posterior odds: d ln O / d ln theta at the central point (pi_DM = 0.03, L_rest = 0.01)
c = dict(pi_DM=0.03, pi_U=0.1, L_rest=0.01, f_FP=3.0, B_DM=16.4)
def odds(**kw):
    p = posterior(kw["pi_DM"], kw["pi_U"], kw["L_rest"], kw["f_FP"], kw["B_DM"], INP["b_eff"], L_ACC)[0]
    return p / (1 - p)
ELAS = {}
for k in c:
    up = dict(c); up[k] = c[k] * 1.01
    dn = dict(c); dn[k] = c[k] / 1.01
    ELAS[k] = float((np.log(odds(**up)) - np.log(odds(**dn))) / (2 * np.log(1.01)))
ELAS["P_DM_at_point"] = float(posterior(**c, b=INP["b_eff"], L_acc=L_ACC)[0])

# ----------------------------------------------------------------------------------------------
# 4. Reference-class (historical base-rate) parametrisation
# ----------------------------------------------------------------------------------------------
# r = P(signal | ~3 sigma single-event anomaly) from history (P083 will estimate). A typical anomaly of the
# class carries B_typ ~ SBB cap at its global p; the LZ event carries B_DM / f_FP. Posterior odds =
# r/(1-r) x (B_DM/f_FP)/B_typ.
r_grid = np.geomspace(0.01, 0.3, 61)
REF = {}
for fp in (1.0, 3.0, 10.0):
    for Bdm in (6.6, 16.4, 28.7):
        Bp = Bdm / fp / B_SBB
        Pr = r_grid * Bp / (r_grid * Bp + 1 - r_grid)
        REF[f"f_FP={fp:g},B_DM={Bdm}"] = dict(B_rel=float(Bp), P_DM_at_r=dict(zip([f"{r:.3g}" for r in r_grid[::10]], [float(x) for x in Pr[::10]])))
# implied pi_DM making the structural posterior equal to the reference-class one (central others)
IMPLIED = {}
for r in (0.02, 0.05, 0.1, 0.2):
    Pr = r * (16.4 / 3 / B_SBB) / (r * (16.4 / 3 / B_SBB) + 1 - r)
    IMPLIED[f"r={r}"] = {"P_ref": float(Pr), "pi_DM_implied_Lrest0.01": float(pi_needed(Pr, 0.1, 0.01, 3.0, 16.4, INP["b_eff"], L_ACC)),
                         "pi_DM_implied_Lrest0.1": float(pi_needed(Pr, 0.1, 0.1, 3.0, 16.4, INP["b_eff"], L_ACC))}

# ----------------------------------------------------------------------------------------------
# 5. Dossier Section-5 table update (rule: posterior_i  proportional to  dossier P_i x L_i)
# ----------------------------------------------------------------------------------------------
DOSSIER = dict(A=0.10, B=0.19, C=0.13, D=0.08, E=0.05, F=0.18, G=0.05, H=0.02, I=0.10, J=0.06, K=0.04)
DM_SPLIT = dict(I=0.70, J=0.25, K=0.05)   # P027 class masses: inelastic 0.58-0.85, q^2-spin 0.11-0.30, other <~ 0.05
LABELS = dict(A="A fluctuation of modelled background", B="B wall/RFR MSSI mismodelled", C="C ER leakage / EC tail",
              D="D neutron", E="E accidental coincidence", F="F detector artefact + unknown unknowns", G="G calibration-related",
              H="H non-DM new physics", I="I inelastic DM", J="J heavy WIMP, q-suppressed EFT", K="K other DM")

def table_update(L_rest, f_FP, B_DM, b=INP["b_eff"], ch=CHANNELS.L_central.values):
    L = dict(A=b, B=ch[0], C=ch[2], D=ch[3], E=ch[1], F=ch[4] + L_rest, G=ch[5], H=b)  # H: fits no better than a background of the same rate
    L_DM = B_DM / f_FP * b
    w = {k: DOSSIER[k] * L[k] for k in L}
    for k in DM_SPLIT:
        w[k] = (DOSSIER["I"] + DOSSIER["J"] + DOSSIER["K"]) * DM_SPLIT[k] * L_DM
    Z = sum(w.values())
    return {k: w[k] / Z for k in DOSSIER}, L, L_DM

post_c, L_used, L_DM_c = table_update(0.01, 3.0, 16.4)
post_opt, _, _ = table_update(1e-3, 1.0, 28.7)
post_pes, _, _ = table_update(0.1, 10.0, 6.6, ch=CHANNELS.L_upper.values)
# hyper-prior ranges (L_rest, f_FP, B_DM from the community prior; channel likelihoods central)
Msub = 60_000
rows = {k: np.empty(Msub) for k in DOSSIER}
for i in range(Msub):
    p, _, _ = table_update(TH["L_rest"][i], TH["f_FP"][i], TH["B_DM"][i])
    for k in DOSSIER:
        rows[k][i] = p[k]
TABLE = pd.DataFrame({"explanation": [LABELS[k] for k in DOSSIER], "dossier_P": [DOSSIER[k] for k in DOSSIER],
                      "L_event_central": [L_used[k] if k in L_used else L_DM_c for k in DOSSIER],
                      "posterior_central": [post_c[k] for k in DOSSIER], "posterior_q16": [np.quantile(rows[k], 0.16) for k in DOSSIER],
                      "posterior_q84": [np.quantile(rows[k], 0.84) for k in DOSSIER], "posterior_optimistic": [post_opt[k] for k in DOSSIER],
                      "posterior_pessimistic": [post_pes[k] for k in DOSSIER]}, index=list(DOSSIER))
TABLE.loc["DM total (I+J+K)"] = ["DM total", 0.20, L_DM_c, sum(post_c[k] for k in "IJK"), np.quantile(rows["I"] + rows["J"] + rows["K"], 0.16),
                                 np.quantile(rows["I"] + rows["J"] + rows["K"], 0.84), sum(post_opt[k] for k in "IJK"), sum(post_pes[k] for k in "IJK")]
TABLE.to_csv(os.path.join(OUT, "P061_table3_dossier_update.csv"), float_format="%.4g")
# Comparison: the coordinator-style judgement mapping quoted in the assignment (not our rule; for reference)
JUDGE = dict(B=0.01, C=0.02, D=0.005, E=0.01, F=0.05, G=0.005, H=0.01)

# ----------------------------------------------------------------------------------------------
# 6. Value of information: expected entropy / spread reduction for four future measurements
# ----------------------------------------------------------------------------------------------
k = INP["k_next"]
def nb_pmf(r, k):
    """Negative-binomial predictive for a rate with Gamma(r,1) posterior per 2.84 t yr, scaled to k exposures: N=0,1,>=2."""
    p0 = (1 / (1 + k)) ** r
    p1 = r * k / (1 + k) ** (r + 1)
    return np.array([p0, p1, 1 - p0 - p1])

f_trans = 0.5   # share of H_U that is a one-off (transient) effect predicting zero further events
f_spec = 0.7    # share of steady H_U that is LZ-specific (predicts nothing in another detector)
muB = k * INP["b_eff"]
PRED = {
    "LZ next exposure 6.76 t yr (N = 0/1/>=2)": dict(
        DM=np.array([0.58, 0.22, 0.20]),                     # P027 model-averaged predictive
        B=np.array([np.exp(-muB), muB * np.exp(-muB), 1 - np.exp(-muB) - muB * np.exp(-muB)]),
        U=f_trans * np.array([1, 0, 0]) + (1 - f_trans) * nb_pmf(1.0, k)),   # log-uniform rate -> Exponential(1) posterior
    "LZ next exposure, if H_U is a one-off effect (f_trans = 1)": dict(
        DM=np.array([0.58, 0.22, 0.20]),
        B=np.array([np.exp(-muB), muB * np.exp(-muB), 1 - np.exp(-muB) - muB * np.exp(-muB)]),
        U=np.array([1.0, 0.0, 0.0])),
    "LZ next exposure, L10 plug-in predictive (P020)": dict(
        DM=np.array([0.092, 0.220, 0.688]),
        B=np.array([np.exp(-muB), muB * np.exp(-muB), 1 - np.exp(-muB) - muB * np.exp(-muB)]),
        U=f_trans * np.array([1, 0, 0]) + (1 - f_trans) * nb_pmf(1.0, k)),
    "XENONnT+PandaX-4T 270 keV reanalysis (0 / >=1)": dict(
        DM=np.array([np.exp(-0.62), 1 - np.exp(-0.62)]),   # 1.63 events at LZ best fit (P035) x 0.92/2.4 model-average scaling (P027/P020)
        B=np.array([1 - 1e-3, 1e-3]),
        U=np.array([1 - (1 - f_trans) * (1 - f_spec) * (1 - nb_pmf(1.0, INP["k_xenon"])[0]), (1 - f_trans) * (1 - f_spec) * (1 - nb_pmf(1.0, INP["k_xenon"])[0])])),
    "XENONnT+PandaX-4T at LZ best-fit coupling (1.63 events)": dict(
        DM=np.array([np.exp(-1.63), 1 - np.exp(-1.63)]),
        B=np.array([1 - 1e-3, 1e-3]),
        U=np.array([1 - (1 - f_trans) * (1 - f_spec) * (1 - nb_pmf(1.0, INP["k_xenon"])[0]), (1 - f_trans) * (1 - f_spec) * (1 - nb_pmf(1.0, INP["k_xenon"])[0])])),
    "wall-MSSI calibration (k <= 2 / k > 100)": dict(
        DM=np.array([1 - 1e-4, 1e-4]), B=np.array([1 - 1e-4, 1e-4]),
        U=None),   # filled per theta: share of MSSI in L_U
    "LZ internal artefact audit (nothing / artefact found)": dict(
        DM=np.array([0.99, 0.01]), B=np.array([0.99, 0.01]),
        U=None),   # share of artefact channel in L_U (false-positive 1% under DM/B is a judgement)
}

def voi(pred, TH, P3, name, nsub=100_000):
    pDM, pB, pU = (x[:nsub] for x in P3)
    L_U = L_ACC + TH["L_rest"][:nsub]
    if pred["U"] is None:
        share = (CHANNELS.L_central.values[0] * 1.64 if "MSSI" in name else CHANNELS.L_central.values[4]) / L_U
        pU_m = np.stack([1 - share, share], axis=1)
    else:
        pU_m = np.tile(pred["U"], (nsub, 1))
    pDM_m = np.tile(pred["DM"], (nsub, 1)); pB_m = np.tile(pred["B"], (nsub, 1))
    # predictive per theta and posterior per outcome
    Pm = pDM[:, None] * pDM_m + pB[:, None] * pB_m + pU[:, None] * pU_m
    post_DM = pDM[:, None] * pDM_m / Pm
    H0 = h2(pDM)
    H1 = (Pm * h2(post_DM)).sum(axis=1)
    Pm_bar = Pm.mean(axis=0)
    # community spread (68% width and IQR) of P(DM) before / after, weighted by the community-averaged predictive
    def width(x, w=None):
        q = np.quantile(x, [0.16, 0.84]) if w is None else _wq(x, w, [0.16, 0.84])
        return float(q[1] - q[0])
    def _wq(x, w, qs):
        o = np.argsort(x); cw = np.cumsum(w[o]) / w.sum()
        return np.interp(qs, cw, x[o])
    W0 = width(pDM)
    W1 = float(sum(Pm_bar[m] * width(post_DM[:, m], Pm[:, m]) for m in range(Pm.shape[1])))
    out = dict(H_before_nats=float(H0.mean()), H_after_nats=float(H1.mean()), expected_info_gain_nats=float((H0 - H1).mean()),
               width68_before=W0, width68_after_expected=W1, outcome_probabilities_community=[float(x) for x in Pm_bar],
               median_P_DM_after_each_outcome=[float(np.median(post_DM[:, m])) for m in range(Pm.shape[1])])
    # also at the central point (pi_DM = 0.03, pi_U = 0.1, L_rest = 0.01, f_FP = 3, B_DM = 16.4)
    c3 = posterior(0.03, 0.1, 0.01, 3.0, 16.4, INP["b_eff"], L_ACC)
    if pred["U"] is None:
        share_c = (CHANNELS.L_central.values[0] * 1.64 if "MSSI" in name else CHANNELS.L_central.values[4]) / (L_ACC + 0.01)
        pU_c = np.array([1 - share_c, share_c])
    else:
        pU_c = pred["U"]
    Pm_c = c3[0] * pred["DM"] + c3[1] * pred["B"] + c3[2] * pU_c
    out["central_point_P_DM_before"] = float(c3[0])
    out["central_point_P_DM_after_each_outcome"] = [float(x) for x in (c3[0] * pred["DM"] / Pm_c)]
    out["central_point_P_U_after_each_outcome"] = [float(x) for x in (c3[2] * pU_c / Pm_c)]
    return out

VOI = {}
for name, pred in PRED.items():
    VOI[name] = voi(pred, TH, (P_DM, P_B, P_U), name)
# transient-share sensitivity for the LZ next-exposure measurement
VOI_ftrans = {}
for ft in (0.0, 0.5, 1.0):
    name = "LZ next exposure 6.76 t yr (N = 0/1/>=2)"
    pred = dict(DM=np.array([0.58, 0.22, 0.20]), B=PRED[name]["B"], U=ft * np.array([1, 0, 0]) + (1 - ft) * nb_pmf(1.0, k))
    VOI_ftrans[f"f_trans={ft}"] = voi(pred, TH, (P_DM, P_B, P_U), name)

# ----------------------------------------------------------------------------------------------
# 7. Figures (dataviz palette: blue sequential ramp; categorical blue/orange/aqua; muted ink)
# ----------------------------------------------------------------------------------------------
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlecolor": INK, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
cmap = LinearSegmentedColormap.from_list("blue_seq", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0), sharey=True, constrained_layout=True)
for ax, M, ttl in zip(axes, (MAP_pes, MAP_central, MAP_opt),
                      (f"pessimistic: pi_U=0.3, f_FP=10, B_DM=6.6, L_acc=0.066", f"central: pi_U=0.1, f_FP=3, B_DM=16.4, L_acc=0.0095",
                       f"optimistic: pi_U=0.05, f_FP=1, B_DM=28.7, L_acc=0.0095")):
    pc = ax.pcolormesh(Lr_grid, pi_grid, M, cmap=cmap, vmin=0, vmax=1, shading="auto", rasterized=True)
    cs = ax.contour(Lr_grid, pi_grid, M, levels=[0.1, 0.5, 0.9], colors=[ORANGE, INK, AQUA], linewidths=1.6)
    ax.clabel(cs, fmt={0.1: "P=0.1", 0.5: "P=0.5", 0.9: "P=0.9"}, fontsize=8, colors=INK)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("L_U,rest")
    ax.set_title(ttl, fontsize=8.5)
    ax.grid(True, color=GRID, lw=0.5)
    ax.axhline(0.2, color=MUTED, lw=0.8, ls=":"); ax.text(1.1e-3, 0.21, "dossier DM total 0.20", color=MUTED, fontsize=7.5)
    if M.max() < 0.1:
        ax.text(0.5, 0.5, "P(DM) < 0.1 everywhere", transform=ax.transAxes, ha="center", color=INK2, fontsize=9)
axes[1].set_xlabel("L_U,rest  (unaccounted-background likelihood of the event, per 2.84 t yr)")
axes[0].set_ylabel("pi_DM  (prior P that DM gives any event here)")
cb = fig.colorbar(pc, ax=axes, shrink=0.9, pad=0.01); cb.set_label("P(DM | event)")
fig.suptitle("P061 Fig. 1: posterior DM probability over (pi_DM, L_U,rest); contours at P = 0.1, 0.5, 0.9", fontsize=10, color=INK)
fig.savefig(os.path.join(FIG, "P061_fig1_map.png"), dpi=170)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), constrained_layout=True)
ax = axes[0]
bins = np.geomspace(1e-4, 1, 61)
ax.hist(P_DM, bins=bins, color=BLUE, alpha=0.9, label="community prior (all five hyper-parameters log-uniform)", histtype="stepfilled", edgecolor=SURF)
t = draw(N // 2, dict(f_FP=1.0)); pf1 = posterior(t["pi_DM"], t["pi_U"], t["L_rest"], t["f_FP"], t["B_DM"], INP["b_eff"], L_ACC)[0]
ax.hist(pf1, bins=bins, color=ORANGE, histtype="step", lw=1.8, label="f_FP = 1 (no forking-paths penalty)")
t = draw(N // 2, dict(pi_DM=0.2)); pp2 = posterior(t["pi_DM"], t["pi_U"], t["L_rest"], t["f_FP"], t["B_DM"], INP["b_eff"], L_ACC)[0]
ax.hist(pp2, bins=bins, color=AQUA, histtype="step", lw=1.8, label="pi_DM = 0.2 fixed (dossier DM total)")
for x, lab in [(0.1, "0.1"), (0.5, "0.5")]:
    ax.axvline(x, color=INK2, lw=0.8, ls="--")
ax.set_xscale("log"); ax.set_xlabel("P(DM | event)"); ax.set_ylabel("hyper-prior samples per bin")
ax.set_title("Distribution of P(DM) over the community hyper-prior", fontsize=9.5)
ax.legend(frameon=False, fontsize=7.5, loc="upper left"); ax.grid(True, color=GRID, lw=0.5, axis="y")
s = MC["community_prior"]
ax.text(0.02, 0.62, f"median {s['median']:.3f}\n68%: {s['q16']:.3f}-{s['q84']:.2f}\nP>0.5: {100*s['frac_gt_0p5']:.0f}%   P<0.1: {100*s['frac_lt_0p1']:.0f}%",
        transform=ax.transAxes, fontsize=8, color=INK, va="top")
ax = axes[1]
keys = list(HYPER)
vals = [SENS["logit"][kk] for kk in keys]
ax.barh(range(len(keys)), vals, color=BLUE, height=0.55)
ax.set_yticks(range(len(keys))); ax.set_yticklabels(["pi_DM", "pi_U", "L_U,rest", "f_FP", "B_DM"])
ax.invert_yaxis(); ax.set_xlabel("first-order sensitivity index of logit P(DM)"); ax.set_xlim(0, 1)
for i, v in enumerate(vals):
    ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8, color=INK)
ax.set_title("Which hyper-parameter drives the spread", fontsize=9.5); ax.grid(True, color=GRID, lw=0.5, axis="x")
for sp in ("top", "right"):
    for a in axes: a.spines[sp].set_visible(False)
fig.suptitle("P061 Fig. 2: prior sensitivity of P(DM | event)", fontsize=10, color=INK)
fig.savefig(os.path.join(FIG, "P061_fig2_distribution.png"), dpi=170)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), constrained_layout=True, sharey=True)
names = list(VOI)
ig = [VOI[n]["expected_info_gain_nats"] for n in names]
H0 = VOI[names[0]]["H_before_nats"]
w0 = VOI[names[0]]["width68_before"]
w1 = [VOI[n]["width68_after_expected"] for n in names]
y = np.arange(len(names))
ax = axes[0]
ax.barh(y, ig, height=0.55, color=BLUE)
for i, v in enumerate(ig):
    ax.text(v + 0.001, i, f"{v:.3f} ({100*v/H0:.0f}%)", va="center", fontsize=7.5, color=INK)
ax.set_yticks(y); ax.set_yticklabels([n.replace(" (", "\n(").replace(", ", ",\n") for n in names], fontsize=7.5); ax.invert_yaxis()
ax.set_xlabel(f"expected information gain, DM-vs-not [nats] (prior entropy {H0:.3f})")
ax.set_xlim(0, max(ig) * 1.35); ax.grid(True, color=GRID, lw=0.5, axis="x")
ax.set_title("Expected entropy reduction", fontsize=9.5)
ax = axes[1]
ax.barh(y, w1, height=0.55, color=ORANGE)
ax.axvline(w0, color=INK2, lw=1.0, ls="--"); ax.text(w0 + 0.002, len(names) - 0.6, f"before: {w0:.3f}", color=INK2, fontsize=7.5)
for i, v in enumerate(w1):
    ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=7.5, color=INK)
ax.set_xlabel("expected 68% width of P(DM) across the community afterwards")
ax.set_xlim(0, max(w1 + [w0]) * 1.25); ax.grid(True, color=GRID, lw=0.5, axis="x")
ax.set_title("Expected community spread afterwards", fontsize=9.5)
for a in axes:
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
fig.suptitle("P061 Fig. 3: value of information of four future measurements", fontsize=10, color=INK)
fig.savefig(os.path.join(FIG, "P061_fig3_voi.png"), dpi=170)
plt.close(fig)

# ----------------------------------------------------------------------------------------------
# 8. Save results
# ----------------------------------------------------------------------------------------------
RES = dict(inputs=INP, B_SBB=float(B_SBB), L_acc_central=L_ACC, L_acc_upper=L_ACC_UP, hyper_prior=HYPER, central=CENTRAL,
           validation_vs_P001=val, map_area_fractions=area, pi_DM_boundaries=bound, monte_carlo=MC, sensitivity=SENS, elasticities=ELAS,
           reference_class=REF, implied_pi_DM=IMPLIED, dossier_update_central=post_c, dossier_update_optimistic=post_opt,
           dossier_update_pessimistic=post_pes, dossier_update_L_used=L_used, L_DM_central=float(L_DM_c), judgement_mapping_for_comparison=JUDGE,
           voi=VOI, voi_f_trans=VOI_ftrans, voi_assumptions=dict(f_trans=f_trans, f_spec=f_spec, k_next=k, k_xenon=INP["k_xenon"]))
with open(os.path.join(OUT, "P061_results.json"), "w") as f:
    json.dump(RES, f, indent=1, default=float)

# console summary
print(f"L_acc central = {L_ACC:.4g}, upper = {L_ACC_UP:.4g};  SBB cap = {B_SBB:.2f}")
print("validation:", json.dumps(val))
print("area fractions:", json.dumps(area))
for tag in bound:
    print(tag, json.dumps(bound[tag]["L_rest=0.01"]), json.dumps(bound[tag]["L_rest=0.1"]))
print("community prior:", json.dumps(MC["community_prior"]))
for kk in variants: print(f"  {kk}: median {MC[kk]['median']:.3f} [{MC[kk]['q16']:.3f},{MC[kk]['q84']:.3f}] >0.5 {MC[kk]['frac_gt_0p5']:.3f} <0.1 {MC[kk]['frac_lt_0p1']:.3f}")
for kk in ("L_acc upper bracket (0.066)", "L_acc = 0 (no accounted channels)", "b_eff=2e-4 (L_DM = B_DM b_H held fixed)", "b_eff=1.1e-3 (L_DM held fixed)", "uniform (linear) hyper-prior"):
    print(f"  {kk}: median {MC[kk]['median']:.3f} [{MC[kk]['q16']:.3f},{MC[kk]['q84']:.3f}] >0.5 {MC[kk]['frac_gt_0p5']:.3f} <0.1 {MC[kk]['frac_lt_0p1']:.3f}")
print("sensitivity (logit):", json.dumps(SENS["logit"]))
print("elasticities:", json.dumps(ELAS))
print("implied pi_DM:", json.dumps(IMPLIED))
print(TABLE.to_string(float_format=lambda x: f"{x:.3g}"))
for n in VOI:
    v = VOI[n]
    print(f"VOI {n}: IG {v['expected_info_gain_nats']:.4f} nats (H {v['H_before_nats']:.3f}->{v['H_after_nats']:.3f}); width68 {v['width68_before']:.3f}->{v['width68_after_expected']:.3f}; "
          f"P(m) {np.round(v['outcome_probabilities_community'],3).tolist()}; central P(DM) {v['central_point_P_DM_before']:.3f} -> {np.round(v['central_point_P_DM_after_each_outcome'],3).tolist()}; P(U) -> {np.round(v['central_point_P_U_after_each_outcome'],3).tolist()}")
for n in VOI_ftrans:
    print(f"VOI f_trans {n}: IG {VOI_ftrans[n]['expected_info_gain_nats']:.4f}; width {VOI_ftrans[n]['width68_after_expected']:.3f}; central after {np.round(VOI_ftrans[n]['central_point_P_DM_after_each_outcome'],3).tolist()}")
