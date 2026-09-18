"""
P091 - The corpus posterior after two weeks: updating the dossier's explanation
classes A-K for the LZ 248 keV event with the quantitative results of P001-P098.

Method
------
posterior_i(theta) = pi_i * L_i(theta) / sum_j pi_j * L_j(theta)

pi_i    : prior over the 11 dossier classes (dossier Section 5; alternatives scanned)
L_i     : corpus "event likelihood" of class i = expected number of event-like
          occurrences of class i in the event's neighbourhood (S1c > 500 phd,
          |d| <= 2 sigma_NR, single site, veto-silent, at the observed position and
          date), built as a product of factors, each attributed to the corpus paper
          that established it.  DM classes: L = (B_DM b_H / f_FP) * share * S * T.
theta   : every factor with a quoted corpus range is drawn log-uniformly within
          that range (Monte Carlo, N draws); k_B ~ Gamma(2, 1/3.23) (wall sideband
          posterior, P004); k_E ~ log-uniform[1,1e3] x exp(-0.016 k) (UDT zero, P022).

Outputs (output/work/P091/): P091_class_table.csv, P091_factor_table.csv,
P091_sensitivity.csv, P091_paper_contributions.csv, P091_results.json,
figures/P091_fig1_prior_posterior.png, figures/P091_fig2_sensitivity.png,
figures/P091_fig3_papers.png.  Run from the simulation root:
    .venv/bin/python output/code/P091_corpus_posterior.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "output/code")
from common import lzcommon as lz  # noqa: E402,F401  (LZ dict for b_H cross-check only)

OUT = "output/work/P091"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(91)
N = 200_000

CLASSES = list("ABCDEFGHIJK")
CLASS_NAMES = {
    "A": "fluctuation of modelled/known background",
    "B": "wall/RFR MSSI mismodelled",
    "C": "ER leakage / charge-poor ER",
    "D": "neutron (any origin)",
    "E": "accidental coincidence mismodelled",
    "F": "detector artefact + unknown unknowns",
    "G": "calibration-related (activation, 214Pb, scale)",
    "H": "non-DM new physics",
    "I": "inelastic DM",
    "J": "q^4-spin / momentum-suppressed elastic EFT DM",
    "K": "other DM (boosted, exothermic, composite, streams)",
}
PRIOR_DOSSIER = dict(A=0.10, B=0.19, C=0.13, D=0.08, E=0.05, F=0.18, G=0.05, H=0.02,
                     I=0.10, J=0.06, K=0.04)
B_H = 5.7e-4          # P016 high-energy modelled background anchor (P001: 2e-4 - 1e-3)
L_REF = 0.1           # "viable explanation" yardstick: 10 % chance of one event (P004/P033/P041 convention)

# ----------------------------------------------------------------------------
# Factor table.  Each factor: class, name, kind, lo, hi (log-uniform) or special,
# null value (state before the paper), primary paper (sequential credit), confirming papers.
# ----------------------------------------------------------------------------
FACTORS = [
    # class A: modelled / known-physics background as modelled
    dict(cls="A", name="b_nb  neighbourhood modelled background", kind="logu", lo=3.5e-4, hi=6.4e-4,
         null=L_REF, paper="P016", confirm="P001, P052, P070, P093", role="rate"),
    dict(cls="A", name="T_A  position: modelled mixture vs uniform", kind="logu", lo=0.56, hi=1.0,
         null=1.0, paper="P070", confirm="P093", role="mult"),
    dict(cls="A", name="nu_incoh  incoherent atm-nu lone NR (additive)", kind="logu", lo=8e-6, hi=5e-4,
         null=0.0, paper="P019", confirm="P060, P087", role="add"),
    # class B: wall / RFR MSSI
    dict(cls="B", name="R_B  wall MSSI as modelled in neighbourhood", kind="logu", lo=1.15e-4, hi=2.5e-4,
         null=L_REF, paper="P004", confirm="P033", role="rate"),
    dict(cls="B", name="k_B  sideband-allowed mismodelling (Gamma(2,1/3.23))", kind="gamma_kB", lo=0.62, hi=1.64,
         null=1.0, paper="P004", confirm="P033, P053, P090", role="mult"),
    dict(cls="B", name="P_pos  position LR (wall MSSI vs uniform)^-1", kind="logu", lo=0.02, hi=0.43,
         null=1.0, paper="P033", confirm="P070, P079", role="mult"),
    dict(cls="B", name="V_B  veto-silence penalty", kind="logu", lo=0.10, hi=0.79,
         null=1.0, paper="P073", confirm="", role="mult"),
    dict(cls="B", name="W_B  hit-pattern consistency", kind="logu", lo=0.3, hi=1.0,
         null=1.0, paper="P093", confirm="", role="mult"),
    dict(cls="B", name="RFR + 214Pb MSSI at position (additive)", kind="logu", lo=1e-9, hi=1e-7,
         null=0.0, paper="P033", confirm="P053, P070, P093", role="add"),
    # class C: ER leakage
    dict(cls="C", name="R_C  ER tail floor-to-plateau extrapolation", kind="logu", lo=1.6e-4, hi=2.8e-3,
         null=L_REF, paper="P010", confirm="P024, P095", role="rate"),
    dict(cls="C", name="F_C  recombination tail shape (Gaussian .. mirrored)", kind="logu", lo=0.014, hi=1.0,
         null=1.0, paper="P056", confirm="P093", role="mult"),
    # class D: neutrons
    dict(cls="D", name="L_D  neutron single scatters 200-270 keV", kind="logu", lo=5e-6, hi=2e-4,
         null=L_REF, paper="P013", confirm="P049, P063, P093", role="rate"),
    # class E: accidentals
    dict(cls="E", name="R_E  accidentals in neighbourhood as modelled", kind="logu", lo=1.0e-4, hi=2.0e-4,
         null=L_REF, paper="P022", confirm="P093", role="rate"),
    dict(cls="E", name="k_E  high-S1 mismodelling allowed by UDT zero", kind="kE", lo=1.0, hi=183.0,
         null=1.0, paper="P022", confirm="", role="mult"),
    dict(cls="E", name="T_E  S2-width/drift & cathode topology", kind="logu", lo=0.01, hi=1.0,
         null=1.0, paper="P093", confirm="P022", role="mult"),
    # class F: artefacts + unknown unknowns (judgemental; scanned)
    dict(cls="F", name="L_F  artefact residual incl. unknown unknowns", kind="logu", lo=6e-4, hi=3e-2,
         null=L_REF, paper="P041", confirm="P093, P061, P077", role="rate"),
    # class G: calibration-related
    dict(cls="G", name="R_G  activation MSSI-like events per run", kind="logu", lo=9e-6, hi=8e-3,
         null=L_REF, paper="P029", confirm="P053", role="rate"),
    dict(cls="G", name="P_pos,G  position penalty for MSSI-like topology", kind="logu", lo=0.01, hi=0.43,
         null=1.0, paper="P033", confirm="P070", role="mult"),
    # class H: non-DM new physics
    dict(cls="H", name="L_H  kinematically special non-DM new physics", kind="logu", lo=1e-5, hi=5.7e-4,
         null=L_REF, paper="P019", confirm="P036, P060, P080, P087", role="rate"),
    # DM common
    dict(cls="DM", name="R_DM  B10 x b_H (single-event Bayes factor after Occam)", kind="logu", lo=7 * B_H, hi=37 * B_H,
         null=L_REF, paper="P001", confirm="P008, P071", role="rate"),
    dict(cls="DM", name="B_DM refinement  model-marginalised 16.4-28.7 (replaces P001 range)", kind="logu",
         lo=16.4 * B_H, hi=28.7 * B_H, null=None, paper="P027", confirm="P016", role="rate_replace"),
    dict(cls="DM", name="1/f_FP  forking-paths divisor", kind="logu_inv", lo=1.0, hi=10.0,
         null=1.0, paper="P061", confirm="P071", role="mult"),
    # DM class shares and survival
    dict(cls="I", name="share_I  inelastic mass of P027 posterior", kind="logu", lo=0.58, hi=0.85,
         null=1.0, paper="P027", confirm="P021, P082", role="mult"),
    dict(cls="J", name="share_J  q^2-suppressed spin operators", kind="logu", lo=0.11, hi=0.30,
         null=1.0, paper="P027", confirm="P016, P003", role="mult"),
    dict(cls="K", name="share_K  remainder of model space", kind="logu", lo=0.02, hi=0.08,
         null=1.0, paper="P027", confirm="", role="mult"),
    dict(cls="I", name="S_I  UV-completion survival (Higgsino excluded, 360-385 keV corner closed)", kind="logu",
         lo=0.3, hi=1.0, null=1.0, paper="P076", confirm="P084, P025, P086, P075, P054, P038, P082", role="mult"),
    dict(cls="I", name="T_I  16 June date odds (delta 300-380 keV)", kind="logu", lo=1.4, hi=2.5,
         null=1.0, paper="P006", confirm="P034, P055, P085, P092", role="mult"),
    dict(cls="J", name="S_J  EFT consistency (Lambda << m_chi; L10 relic)", kind="logu", lo=0.2, hi=1.0,
         null=1.0, paper="P031", confirm="P025, P074, P044, P057, P089", role="mult"),
    dict(cls="J", name="T_J  date odds elastic", kind="fixed", lo=0.98, hi=0.98,
         null=1.0, paper="P006", confirm="", role="mult"),
    dict(cls="K", name="S_K  survival (CRDM/SIMP/MCP/streams excluded; exothermic B<=2)", kind="logu",
         lo=0.05, hi=0.5, null=1.0, paper="P040", confirm="P080, P072, P058, P030, P055, P097, P065, P023", role="mult"),
]


def sample_kE(n):
    """k ~ log-uniform[1,1e3] x exp(-0.016 k)  (UDT zero: 0.016 k expected, 0 observed; P022)."""
    grid = np.logspace(0, 3, 4001)
    w = np.exp(-0.016 * grid) / grid  # density in k (log-uniform prior = 1/k)
    cdf = np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(grid))
    cdf = np.concatenate([[0.0], cdf]) / cdf[-1]
    return np.interp(rng.uniform(size=n), cdf, grid)


def draw_factor(f, n, central=False):
    if f["kind"] == "fixed":
        return np.full(n, f["lo"])
    if f["kind"] == "logu":
        if central:
            return np.full(n, np.sqrt(f["lo"] * f["hi"]))
        return np.exp(rng.uniform(np.log(f["lo"]), np.log(f["hi"]), n))
    if f["kind"] == "logu_inv":
        if central:
            return np.full(n, 1.0 / np.sqrt(f["lo"] * f["hi"]))
        return 1.0 / np.exp(rng.uniform(np.log(f["lo"]), np.log(f["hi"]), n))
    if f["kind"] == "gamma_kB":
        if central:
            return np.full(n, 0.52)  # median of Gamma(2, 1/3.23) = 1.678/3.23
        return rng.gamma(2.0, 1.0 / 3.23, n)
    if f["kind"] == "kE":
        if central:
            return np.full(n, 6.0)  # median (computed below and checked)
        return sample_kE(n)
    raise ValueError(f["kind"])


def likelihoods(active, n, central=False, samples=None):
    """Return dict cls -> array L_i (n,) with the given set of active factor indices.
    Inactive factors take their null value.  `samples` caches MC draws for reuse."""
    vals = {}
    for idx, f in enumerate(FACTORS):
        if idx in active:
            if samples is not None:
                vals[idx] = samples[idx]
            else:
                vals[idx] = draw_factor(f, n, central)
        else:
            vals[idx] = None if f["null"] is None else np.full(n, f["null"])
    L = {}
    # DM rate: P001 range unless P027 refinement active
    idx_p001 = [i for i, f in enumerate(FACTORS) if f["paper"] == "P001"][0]
    idx_p027r = [i for i, f in enumerate(FACTORS) if f["role"] == "rate_replace"][0]
    R_DM = vals[idx_p027r] if vals[idx_p027r] is not None else vals[idx_p001]
    inv_fFP = vals[[i for i, f in enumerate(FACTORS) if f["name"].startswith("1/f_FP")][0]]
    L_DM = R_DM * inv_fFP
    for c in CLASSES:
        if c in "IJK":
            base = L_DM.copy()
        else:
            base = np.ones(n)
        add = np.zeros(n)
        for idx, f in enumerate(FACTORS):
            if f["cls"] != c:
                continue
            v = vals[idx]
            if f["role"] == "add":
                add = add + v
            else:
                base = base * v
        L[c] = base + add
    return L


def posterior(L, prior):
    num = np.vstack([prior[c] * L[c] for c in CLASSES])
    return num / num.sum(axis=0)


def summarize(post):
    med = np.median(post, axis=1)
    lo = np.quantile(post, 0.16, axis=1)
    hi = np.quantile(post, 0.84, axis=1)
    return med, lo, hi


def kl(p, q):
    p = np.clip(p, 1e-300, None)
    q = np.clip(q, 1e-300, None)
    return float(np.sum(p * np.log(p / q)))


# ----------------------------------------------------------------------------
# 1. Monte Carlo corpus posterior under the dossier prior
# ----------------------------------------------------------------------------
ALL = set(range(len(FACTORS)))
samples = {i: draw_factor(f, N) for i, f in enumerate(FACTORS)}
kE_med = float(np.median(samples[[i for i, f in enumerate(FACTORS) if f["kind"] == "kE"][0]]))
kE_mean = float(np.mean(samples[[i for i, f in enumerate(FACTORS) if f["kind"] == "kE"][0]]))
L_mc = likelihoods(ALL, N, samples=samples)
L_c = likelihoods(ALL, 1, central=True)
post_mc = posterior(L_mc, PRIOR_DOSSIER)
med, lo, hi = summarize(post_mc)
post_c = posterior(L_c, PRIOR_DOSSIER)[:, 0]

L_med = {c: float(np.median(L_mc[c])) for c in CLASSES}
L_lo = {c: float(np.quantile(L_mc[c], 0.16)) for c in CLASSES}
L_hi = {c: float(np.quantile(L_mc[c], 0.84)) for c in CLASSES}
LA = L_mc["A"]
rows = []
for k, c in enumerate(CLASSES):
    lam = L_mc[c] / LA
    rows.append(dict(
        cls=c, explanation=CLASS_NAMES[c], prior_dossier=PRIOR_DOSSIER[c],
        L_median=L_med[c], L_16=L_lo[c], L_84=L_hi[c],
        corpus_LR_vs_A_median=float(np.median(lam)),
        corpus_LR_vs_A_16=float(np.quantile(lam, 0.16)), corpus_LR_vs_A_84=float(np.quantile(lam, 0.84)),
        posterior_central=float(post_c[k]), posterior_median=float(med[k]),
        posterior_16=float(lo[k]), posterior_84=float(hi[k]),
        log10_post_over_prior=float(np.log10(med[k] / PRIOR_DOSSIER[c])),
    ))
tab = pd.DataFrame(rows)
tab.to_csv(os.path.join(OUT, "P091_class_table.csv"), index=False)

P_DM = post_mc[8:].sum(axis=0)
P_bkg_unmod = post_mc[[1, 2, 3, 4, 5, 6]].sum(axis=0)
groups = dict(
    DM=(float(np.median(P_DM)), float(np.quantile(P_DM, 0.16)), float(np.quantile(P_DM, 0.84)),
        float(np.quantile(P_DM, 0.025)), float(np.quantile(P_DM, 0.975))),
    frac_PDM_gt_0p5=float(np.mean(P_DM > 0.5)), frac_PDM_lt_0p1=float(np.mean(P_DM < 0.1)),
    mismodelled_bkg_B_to_G=(float(np.median(P_bkg_unmod)), float(np.quantile(P_bkg_unmod, 0.16)),
                            float(np.quantile(P_bkg_unmod, 0.84))),
)

# factor table with medians
frows = []
for i, f in enumerate(FACTORS):
    s = samples[i]
    frows.append(dict(cls=f["cls"], factor=f["name"], kind=f["kind"], lo=f["lo"], hi=f["hi"],
                      median=float(np.median(s)), q16=float(np.quantile(s, 0.16)), q84=float(np.quantile(s, 0.84)),
                      null=f["null"], primary_paper=f["paper"], confirming=f["confirm"], role=f["role"]))
pd.DataFrame(frows).to_csv(os.path.join(OUT, "P091_factor_table.csv"), index=False)

# ----------------------------------------------------------------------------
# 2. Sensitivity: priors x artefact likelihood (central factors elsewhere)
# ----------------------------------------------------------------------------
def make_prior(pi_dm, bkg="dossier"):
    p = {}
    if bkg == "dossier":
        for c in "ABCDEFGH":
            p[c] = PRIOR_DOSSIER[c] * (1 - pi_dm) / 0.80
    elif bkg == "uniform":
        for c in "ABCDEFGH":
            p[c] = (1 - pi_dm) / 8.0
    for c, w in zip("IJK", (0.5, 0.3, 0.2)):
        p[c] = pi_dm * w
    return p


PRIORS = {
    "sceptic pi_DM=0.01 (P083 implied 0.009)": make_prior(0.01),
    "P083 base rate pi_DM=0.062": make_prior(0.062),
    "dossier pi_DM=0.20": dict(PRIOR_DOSSIER),
    "theorist pi_DM=0.20, uniform backgrounds": make_prior(0.20, "uniform"),
    "flat 1/11 each (pi_DM=0.27)": {c: 1 / 11 for c in CLASSES},
}
LF_GRID = {"L_F=6e-4 (P093)": 6e-4, "L_F=4.2e-3 (range median)": np.sqrt(6e-4 * 3e-2), "L_F=3e-2 (P061 upper)": 3e-2}
idx_LF = [i for i, f in enumerate(FACTORS) if f["name"].startswith("L_F")][0]
sens = []
for pname, pr in PRIORS.items():
    for lname, lf in LF_GRID.items():
        Lc = {c: v.copy() for c, v in L_c.items()}
        Lc["F"] = np.array([lf])
        po = posterior(Lc, pr)[:, 0]
        # MC version: all other factors random, L_F fixed
        Lm = {c: v for c, v in L_mc.items()}
        Lm = dict(Lm)
        Lm["F"] = np.full(N, lf)
        pm = posterior(Lm, pr)
        pdm = pm[8:].sum(axis=0)
        sens.append(dict(prior=pname, LF_case=lname, L_F=lf, P_DM_central=float(po[8:].sum()),
                         P_DM_median=float(np.median(pdm)), P_DM_16=float(np.quantile(pdm, 0.16)),
                         P_DM_84=float(np.quantile(pdm, 0.84)),
                         P_F_central=float(po[5]), P_A_central=float(po[0]), P_C_central=float(po[2]),
                         P_I_central=float(po[8])))
    # full MC with L_F random too
    pm = posterior(L_mc, pr)
    pdm = pm[8:].sum(axis=0)
    sens.append(dict(prior=pname, LF_case="L_F log-uniform [6e-4, 3e-2]", L_F=np.nan,
                     P_DM_central=float(posterior(L_c, pr)[8:, 0].sum()), P_DM_median=float(np.median(pdm)),
                     P_DM_16=float(np.quantile(pdm, 0.16)), P_DM_84=float(np.quantile(pdm, 0.84)),
                     P_F_central=float(np.median(pm[5])), P_A_central=float(np.median(pm[0])),
                     P_C_central=float(np.median(pm[2])), P_I_central=float(np.median(pm[8]))))
sens = pd.DataFrame(sens)
sens.to_csv(os.path.join(OUT, "P091_sensitivity.csv"), index=False)

# variance decomposition of logit P(DM) over the MC factors (first-order, correlation-based)
logit = np.log(P_DM / (1 - P_DM))
var_shares = {}
for i, f in enumerate(FACTORS):
    s = samples[i]
    if f["kind"] == "fixed" or np.std(np.log(s)) < 1e-9:
        continue
    r = np.corrcoef(np.log(s), logit)[0, 1]
    var_shares[f["name"].split()[0] + f" ({f['paper']})"] = float(r ** 2)
tot = sum(var_shares.values())
var_shares = {k: v / tot for k, v in sorted(var_shares.items(), key=lambda kv: -kv[1])}

# ----------------------------------------------------------------------------
# 3. Paper contributions: sequential KL (P-number order), leave-one-out KL, dex moved
# ----------------------------------------------------------------------------
papers = sorted({f["paper"] for f in FACTORS}, key=lambda s: int(s[1:]))
paper_idx = {p: [i for i, f in enumerate(FACTORS) if f["paper"] == p] for p in papers}


def post_central(active, prior=PRIOR_DOSSIER):
    return posterior(likelihoods(active, 1, central=True), prior)[:, 0]


contrib = []
active = set()
prev = post_central(active)
for p in papers:
    active |= set(paper_idx[p])
    cur = post_central(active)
    dlo = np.log10(cur / (1 - cur)) - np.log10(prev / (1 - prev))
    touched = sorted({FACTORS[i]["cls"] for i in paper_idx[p]})
    # leave-one-out under the "as-modelled" reference (rate factors reset to b_H, not to the 10 % yardstick)
    _keep = FACTORS
    FACTORS = [dict(g, null=(B_H if g["null"] == L_REF else g["null"])) for g in _keep]
    loo = post_central(ALL - set(paper_idx[p]))
    FACTORS = _keep
    dex = 0.0
    for i in paper_idx[p]:
        f = FACTORS[i]
        v = draw_factor(f, 1, central=True)[0]
        if f["role"] == "add" or f["null"] in (None, 0.0):
            continue
        dex += abs(np.log10(v / f["null"]))
    contrib.append(dict(paper=p, classes=",".join(touched),
                        factors="; ".join(FACTORS[i]["name"].split("  ")[0] for i in paper_idx[p]),
                        KL_sequential_nats=kl(cur, prev), KL_leave_one_out_ref_bH_nats=kl(post_c, loo),
                        dex_moved=dex, max_abs_dlog10odds=float(np.max(np.abs(dlo))),
                        class_max_shift=CLASSES[int(np.argmax(np.abs(dlo)))],
                        P_DM_after=float(cur[8:].sum()), P_F_after=float(cur[5])))
    prev = cur
contrib = pd.DataFrame(contrib)
contrib["rank_sequential"] = contrib["KL_sequential_nats"].rank(ascending=False).astype(int)
contrib["rank_loo"] = contrib["KL_leave_one_out_ref_bH_nats"].rank(ascending=False).astype(int)
contrib = contrib.sort_values("KL_sequential_nats", ascending=False)
contrib.to_csv(os.path.join(OUT, "P091_paper_contributions.csv"), index=False)

# sequential path with the alternative reference L_REF = b_H (robustness of the ranking)
# (recompute with null rate = B_H)
FACTORS_ALT = [dict(f, null=(B_H if f["null"] == L_REF else f["null"])) for f in FACTORS]
_F0 = FACTORS
FACTORS = FACTORS_ALT
alt = []
active = set()
prev = post_central(active)
for p in papers:
    active |= set(paper_idx[p])
    cur = post_central(active)
    alt.append((p, kl(cur, prev)))
    prev = cur
FACTORS = _F0
alt = pd.DataFrame(alt, columns=["paper", "KL_sequential_nats_ref_bH"]).sort_values("KL_sequential_nats_ref_bH", ascending=False)
contrib = contrib.merge(alt, on="paper")
contrib.to_csv(os.path.join(OUT, "P091_paper_contributions.csv"), index=False)

# ----------------------------------------------------------------------------
# 4. Cross-checks and predictive
# ----------------------------------------------------------------------------
# P061-style binary formula at its central values for comparison
P061_central = 0.017 * (16.4 / 3 * B_H) / (0.017 * (16.4 / 3 * B_H) + 0.883 * B_H + 0.1 * (9.3e-3 + 0.01))
# P081 predictive for LZ's next 2.8 t yr, 200-270 keV: P(>=1 | DM) = 0.25 (corpus band-rate posterior),
# steady LZ-specific unknown (F): P(>=1) = k/(1+k) = 0.5 at k = 1 with f_trans = 0.5 one-off -> 0.25,
# other classes: modelled background 5.6e-4 (P081).
pF = post_mc[5]
pred = P_DM * 0.25 + pF * 0.25 + (1 - P_DM - pF) * 5.6e-4
pred_dossier = (float(np.median(pred)), float(np.quantile(pred, 0.16)), float(np.quantile(pred, 0.84)))

results = dict(
    N_draws=N, b_H=B_H, L_ref_sequential=L_REF, kE_median=kE_med, kE_mean=kE_mean,
    class_table=tab.to_dict(orient="records"),
    group_summaries=groups,
    likelihood_medians=L_med,
    P061_binary_formula_central=P061_central,
    variance_shares_logitPDM=var_shares,
    predictive_next_2p8tyr_200_270keV=dict(median=pred_dossier[0], q16=pred_dossier[1], q84=pred_dossier[2]),
    sensitivity=sens.to_dict(orient="records"),
    paper_contributions=contrib.to_dict(orient="records"),
)
with open(os.path.join(OUT, "P091_results.json"), "w") as fh:
    json.dump(results, fh, indent=1, default=float)

# ----------------------------------------------------------------------------
# 5. Figures (palette: dataviz reference instance; validator skipped - no node)
# ----------------------------------------------------------------------------
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False})

# Fig 1: prior vs posterior per class, log scale, 68 % bars
fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=160)
x = np.arange(len(CLASSES))
w = 0.38
pri = np.array([PRIOR_DOSSIER[c] for c in CLASSES])
ax.bar(x - w / 2, pri, w, color=BLUE, label="dossier prior (3 Sept)", zorder=3, linewidth=0)
ax.bar(x + w / 2, med, w, color=ORANGE, label="corpus posterior (median)", zorder=3, linewidth=0)
ax.errorbar(x + w / 2, med, yerr=[med - lo, hi - med], fmt="none", ecolor=INK2, elinewidth=1, capsize=2, zorder=4)
ax.set_yscale("log")
ax.set_ylim(1e-6, 1.5)
ax.set_xticks(x)
ax.set_xticklabels([f"{c}" for c in CLASSES])
ax.set_ylabel("probability")
ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
for k, c in enumerate(CLASSES):
    ax.text(x[k] + w / 2, hi[k] * 1.6, f"{med[k]:.2g}", ha="center", va="bottom", fontsize=6.5, color=INK2)
ax.legend(frameon=False, loc="lower left", fontsize=8)
ax.set_title("P091: dossier prior vs corpus posterior for explanation classes A-K (68 % bars: corpus ranges)",
             fontsize=9, color=INK, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "P091_fig1_prior_posterior.png"))
plt.close(fig)

# Fig 2: P(DM) heat table over prior x L_F
piv = sens[~sens["L_F"].isna()].pivot(index="prior", columns="LF_case", values="P_DM_central")
piv = piv.loc[list(PRIORS.keys()), list(LF_GRID.keys())]
fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=160)
im = ax.imshow(piv.values, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#0d366b"]),
               vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(3))
ax.set_xticklabels([k.replace(" (", "\n(") for k in piv.columns], fontsize=8)
ax.set_yticks(range(len(piv.index)))
ax.set_yticklabels(piv.index, fontsize=8)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color="white" if v > 0.5 else INK)
ax.set_title("P091: P(DM | corpus) versus prior choice and artefact-class likelihood L_F (central factors)",
             fontsize=9, loc="left")
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.set_label("P(DM)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "P091_fig2_sensitivity.png"))
plt.close(fig)

# Fig 3: top-10 papers by sequential KL, with leave-one-out KL
top = contrib.head(10).iloc[::-1]
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=160)
y = np.arange(len(top))
ax.barh(y + 0.2, top["KL_sequential_nats"], 0.4, color=BLUE, label="sequential KL (P-number order, L_ref = 0.1)", linewidth=0)
ax.barh(y - 0.2, top["KL_sequential_nats_ref_bH"], 0.4, color=AQUA, label="sequential KL, L_ref = b_H (as-modelled reference)", linewidth=0)
ax.set_yticks(y)
ax.set_yticklabels([f"{p}  ({c})" for p, c in zip(top["paper"], top["classes"])], fontsize=8)
ax.set_xlabel("information contribution to the A-K posterior (nats)")
ax.xaxis.grid(True, color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8, loc="center right")
ax.set_title("P091: the ten most consequential papers (classes touched in brackets)", fontsize=9, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "P091_fig3_papers.png"))
plt.close(fig)

# ----------------------------------------------------------------------------
# 6. Console summary
# ----------------------------------------------------------------------------
pd.set_option("display.width", 200)
print("k_E median/mean:", round(kE_med, 2), round(kE_mean, 2))
print("\n=== Class table (dossier prior) ===")
print(tab[["cls", "prior_dossier", "L_median", "corpus_LR_vs_A_median", "posterior_central", "posterior_median",
           "posterior_16", "posterior_84", "log10_post_over_prior"]].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
print("\nP(DM) median/16/84/2.5/97.5:", [f"{v:.3g}" for v in groups["DM"]],
      " frac>0.5:", f"{groups['frac_PDM_gt_0p5']:.3f}", " frac<0.1:", f"{groups['frac_PDM_lt_0p1']:.3f}")
print("P(B..G mismodelled backgrounds):", [f"{v:.3g}" for v in groups["mismodelled_bkg_B_to_G"]])
print("P061 binary formula central (check):", f"{P061_central:.4f}")
print("Predictive P(>=1 in 200-270 keV, next 2.8 t yr):", [f"{v:.3g}" for v in pred_dossier])
print("\n=== Variance shares of logit P(DM) ===")
for k, v in list(var_shares.items())[:8]:
    print(f"  {k}: {v:.3f}")
print("\n=== Sensitivity (central) ===")
print(sens[["prior", "LF_case", "P_DM_central", "P_DM_median", "P_DM_16", "P_DM_84", "P_F_central"]]
      .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
print("\n=== Paper contributions (sorted by sequential KL) ===")
print(contrib[["paper", "classes", "KL_sequential_nats", "KL_leave_one_out_ref_bH_nats", "KL_sequential_nats_ref_bH",
               "dex_moved", "max_abs_dlog10odds", "class_max_shift", "P_DM_after", "P_F_after"]]
      .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
print("\nlzcommon LZ dict keys available:", len(lz.LZ) if hasattr(lz, "LZ") else "n/a")
