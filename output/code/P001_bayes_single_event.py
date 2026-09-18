#!/usr/bin/env python
"""
P001 -- How much evidence is one event?  Bayesian and frequentist reassessment of the
LZ 248 keV recoil (arXiv:2609.02823).

Run from the simulation root:   .venv/bin/python output/code/P001_bayes_single_event.py

Everything here is a "Poisson-in-a-box" caricature of the LZ unbinned likelihood: one
observed event (n = 1) in a region whose modelled background expectation is b, tested
against a signal expectation s with signal efficiency 1 inside the box.  The LZ paper
gives (Fig. 5 caption) 0.0106 +/- 0.0008 events for the whole S1c > 500 phd panel, and the
dossier estimates ~2e-4 within +/-2 sigma of the NR median in that panel.  We use
b in {2e-4, 1e-3, 0.0106}.

Sections
  1. Poisson p-values, one-sided Z, likelihood ratios, profile-likelihood local significance
  2. Bayes factors B10 for several priors on s (closed form + numerical check)
  3. Posterior P(DM|event) for priors pi_DM, with and without an "unknown background" hypothesis
  4. Look-elsewhere / Occam: effective trials factor, mixture Bayes factor, p-value calibration bound
  5. Sensitivity to 0/1/2 further events in the next exposure
Outputs: output/work/P001/*.csv, P001_results.json, figures/*.png
"""
import json
import math
import os
import sys

import numpy as np
from scipy import integrate, optimize, stats

sys.path.insert(0, "output/code")
from common import lzcommon as lz  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

OUT = "output/work/P001"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

R = {}  # master results dict -> JSON

# ---------------------------------------------------------------------------
# Inputs (paper numbers via lzcommon; dossier numbers quoted)
# ---------------------------------------------------------------------------
B_PANEL, B_PANEL_ERR = lz.LZ["bkg_highS1_panel"]  # 0.0106 +/- 0.0008, Fig. 5 caption
B_BOX = 2e-4  # dossier Sec. 2/3: within +/-2 sigma of NR median at S1c > 500 phd (order of magnitude)
B_LIST = [B_BOX, 1e-3, B_PANEL]
Z_LOCAL, Z_GLOBAL = lz.LZ["local_sig_max"], lz.LZ["global_sig"]  # 3.4, 2.6
p_local = lz.sigma_to_p(Z_LOCAL)
p_global = lz.sigma_to_p(Z_GLOBAL)
R["inputs"] = dict(b_list=B_LIST, b_panel=B_PANEL, b_panel_err=B_PANEL_ERR, Z_local=Z_LOCAL,
                   Z_global=Z_GLOBAL, p_local=p_local, p_global=p_global,
                   n_models=lz.LZ["n_models"], n_distinct=lz.LZ["n_distinct"])

# ---------------------------------------------------------------------------
# 1. Poisson in a box
# ---------------------------------------------------------------------------
# Likelihood ratio for n=1:  LR(s) = P(1|s+b)/P(1|b) = ((s+b)/b) exp(-s)
# d/ds ln LR = 1/(s+b) - 1 = 0  ->  s_hat = 1 - b ;  LR_max = exp(b-1)/b
sec1 = []
for b in B_LIST:
    p_ge1 = lz.poisson_p_at_least(1, b)
    z_ge1 = lz.p_to_sigma(p_ge1)
    s_hat = 1.0 - b
    lr_max = math.exp(b - 1.0) / b
    q0 = 2.0 * math.log(lr_max)
    z_plr = math.sqrt(q0)
    lr_s1 = (1.0 + b) / b * math.exp(-1.0)  # LR at s = 1 exactly (the paper's best-fit L10 signal is 1.0)
    sec1.append(dict(b=b, P_ge1=p_ge1, Z_one_sided_exact=z_ge1, s_hat=s_hat, LR_at_s1=lr_s1,
                     LR_max=lr_max, q0=q0, Z_profile_asymptotic=z_plr, ratio_sb_over_b_at_shat=(s_hat + b) / b))
R["sec1_poisson_box"] = sec1

# Invert the paper's 3.4 sigma local into an effective single-event background b_eff:
#  (a) exact Poisson: P(>=1 | b_eff) = p_local     -> b_eff = -ln(1 - p_local)
#  (b) asymptotic profile LR: 2 ln LR_max(b_eff) = Z^2  -> solve exp(b-1)/b = exp(Z^2/2)
b_eff_exact = -math.log(1.0 - p_local)
f = lambda lb: (math.exp(math.exp(lb) - 1.0) / math.exp(lb)) - math.exp(Z_LOCAL ** 2 / 2)
b_eff_asym = math.exp(optimize.brentq(f, math.log(1e-8), math.log(0.5)))
R["sec1_b_eff_from_3p4sigma"] = dict(
    exact_poisson=b_eff_exact, asymptotic_profile_LR=b_eff_asym,
    implied_signal_fraction_in_box_exact=B_BOX / b_eff_exact,
    implied_signal_fraction_in_box_asym=B_BOX / b_eff_asym,
    note="b_eff = (background density at the event)/(signal density at the event per unit s) in the "
         "LZ unbinned likelihood; b_eff = b_box/eps_box where eps_box is the fraction of the signal "
         "PDF inside the box.  The paper's spectra spread signal over S1c ~100-600 phd, so eps_box < 1.")

# ---------------------------------------------------------------------------
# 2. Bayes factors.  B10 = int P(1|s+b) pi(s) ds / P(1|b) = 1 + E_pi[s e^{-s}] / b
# ---------------------------------------------------------------------------
def E_se_uniform(smax):
    return (1.0 - (1.0 + smax) * math.exp(-smax)) / smax


def E_se_loguniform(a, c):
    return (math.exp(-a) - math.exp(-c)) / math.log(c / a)


def E_se_gamma(alpha, beta):  # shape alpha, rate beta
    return alpha * beta ** alpha / (beta + 1.0) ** (alpha + 1.0)


PRIORS = {
    "uniform[0,3]": dict(kind="uniform", smax=3.0, E=E_se_uniform(3.0),
                         pdf=lambda s: (s <= 3.0) / 3.0),
    "uniform[0,10]": dict(kind="uniform", smax=10.0, E=E_se_uniform(10.0),
                          pdf=lambda s: (s <= 10.0) / 10.0),
    "uniform[0,30]": dict(kind="uniform", smax=30.0, E=E_se_uniform(30.0),
                          pdf=lambda s: (s <= 30.0) / 30.0),
    "log-uniform[0.01,30]": dict(kind="loguniform", a=0.01, c=30.0, E=E_se_loguniform(0.01, 30.0),
                                 pdf=lambda s: ((s >= 0.01) & (s <= 30.0)) / (s * math.log(3000.0))),
    "Gamma(k=2,theta=0.5) mean 1": dict(kind="gamma", alpha=2.0, beta=2.0, E=E_se_gamma(2.0, 2.0),
                                        pdf=lambda s: stats.gamma.pdf(s, a=2.0, scale=0.5)),
    "Gamma(k=2,theta=1) mean 2": dict(kind="gamma", alpha=2.0, beta=1.0, E=E_se_gamma(2.0, 1.0),
                                      pdf=lambda s: stats.gamma.pdf(s, a=2.0, scale=1.0)),
    "delta(s=1) [= ML bound]": dict(kind="delta", E=math.exp(-1.0), pdf=None),
}
CENTRAL_PRIOR = "uniform[0,10]"

# numerical check of the closed form
for name, P in PRIORS.items():
    if P["pdf"] is None:
        P["E_numeric"] = P["E"]
        continue
    lo = P.get("a", 0.0)
    hi = P.get("c", P.get("smax", 60.0))
    val, _ = integrate.quad(lambda s: s * math.exp(-s) * float(P["pdf"](s)), lo, hi, limit=200)
    P["E_numeric"] = val
    assert abs(val - P["E"]) < 1e-6 * max(1.0, abs(P["E"])), (name, val, P["E"])

def kass_raftery(B):
    if B < 1:
        return "favours H0"
    if B < 3:
        return "not worth more than a bare mention"
    if B < 20:
        return "positive"
    if B < 150:
        return "strong"
    return "very strong"

bf_rows = []
for name, P in PRIORS.items():
    row = dict(prior=name, E_s_exp_minus_s=P["E"], E_numeric=P["E_numeric"])
    for b in B_LIST:
        B10 = 1.0 + P["E"] / b
        row[f"B10_b={b:g}"] = B10
        row[f"KR_b={b:g}"] = kass_raftery(B10)
    bf_rows.append(row)
R["sec2_bayes_factors"] = bf_rows

# Sellke-Bayarri-Berger (2001) calibration: B10 <= -1/(e p ln p) for p < 1/e
sbb = lambda p: -1.0 / (math.e * p * math.log(p))
R["sec2_SBB_bound"] = dict(B_max_from_p_local=sbb(p_local), B_max_from_p_global=sbb(p_global),
                           B_max_from_P_ge1_panel=sbb(lz.poisson_p_at_least(1, B_PANEL)),
                           B_max_from_P_ge1_box=sbb(lz.poisson_p_at_least(1, B_BOX)))

# ---------------------------------------------------------------------------
# 3. Posterior probability of DM
# ---------------------------------------------------------------------------
PI_DM = [0.001, 0.01, 0.05, 0.2]
PI_U = [0.01, 0.1, 0.3]
L_U = [0.1, 0.5]


def post2(pi, B):
    return pi * B / (pi * B + 1.0 - pi)


def post3(pi_d, pi_u, B, b, LU):
    """Three hypotheses: DM (marginal likelihood B*P(1|b)), unknown background U (likelihood LU),
    known background K (P(1|b)); priors pi_d, pi_u, 1-pi_d-pi_u."""
    p1b = stats.poisson.pmf(1, b)
    num_d = pi_d * B * p1b
    num_u = pi_u * LU
    num_k = (1.0 - pi_d - pi_u) * p1b
    tot = num_d + num_u + num_k
    return num_d / tot, num_u / tot, num_k / tot


rows2 = []
for b in B_LIST:
    for name in ["uniform[0,3]", CENTRAL_PRIOR, "uniform[0,30]", "log-uniform[0.01,30]", "Gamma(k=2,theta=0.5) mean 1"]:
        B10 = 1.0 + PRIORS[name]["E"] / b
        for pi in PI_DM:
            rows2.append(dict(b=b, prior_s=name, B10=B10, pi_DM=pi, P_DM=post2(pi, B10)))
R["sec3_posterior_two_hypotheses"] = rows2

rows3 = []
for b in B_LIST:
    B10 = 1.0 + PRIORS[CENTRAL_PRIOR]["E"] / b
    for pi_d in PI_DM:
        for pi_u in PI_U:
            for LU in L_U:
                if pi_d + pi_u >= 1:
                    continue
                pd_, pu_, pk_ = post3(pi_d, pi_u, B10, b, LU)
                rows3.append(dict(b=b, prior_s=CENTRAL_PRIOR, B10=B10, pi_DM=pi_d, pi_U=pi_u, L_U=LU,
                                  P_DM=pd_, P_U=pu_, P_known=pk_))
R["sec3_posterior_three_hypotheses"] = rows3

# The small-b limit: P(DM) -> pi_d E / (pi_d E + pi_u L_U)  (known background drops out)
E_c = PRIORS[CENTRAL_PRIOR]["E"]
R["sec3_small_b_limit"] = dict(
    formula="P(DM) -> pi_DM*E[s e^-s] / (pi_DM*E[s e^-s] + pi_U*L_U) as b -> 0",
    E_central=E_c,
    examples={f"pi_DM={pd_},pi_U={pu_},L_U={lu}": pd_ * E_c / (pd_ * E_c + pu_ * lu)
              for pd_ in PI_DM for pu_ in PI_U for lu in L_U},
    condition_for_P_DM_gt_half=f"pi_DM > pi_U * L_U / E = pi_U * L_U / {E_c:.4f} = {1/E_c:.1f} * pi_U * L_U")

# ---------------------------------------------------------------------------
# 4. Look-elsewhere effect vs Bayesian Occam penalty
# ---------------------------------------------------------------------------
N_eff = math.log(1.0 - p_global) / math.log(1.0 - p_local)  # (1-p_l)^N = 1-p_g
N_eff_ratio = p_global / p_local


def flatten_sig(d):
    vals = []
    for v in (d.values() if isinstance(d, dict) else d):
        if isinstance(v, (dict, list, tuple)):
            vals.extend(flatten_sig(v))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            vals.append(float(v))
    return vals


sig_L = flatten_sig(lz.LSIG)
sig_O = flatten_sig(lz.OSIG)
sig_all = np.array(sig_L + sig_O)
sig_all = sig_all[np.isfinite(sig_all)]
n_cells = len(sig_all)
frac = {thr: float(np.mean(sig_all >= thr)) for thr in [2.0, 2.5, 3.0, 3.3, 3.4]}
n_zero = int(np.sum(sig_all < 0.05))

# Mixture Bayes factor over N "effectively distinct" spectra with equal prior weight:
#   B_mix = (1/N) sum_k B_k.   Scenario A: one spectrum fits (B), others uninformative (B_k = 1)
#   Scenario B: others predict unseen low-energy events (B_k -> 0)  => B_mix = B/N
#   Scenario C: weight by the paper's own table -- fraction f>=3.0 sigma get B, the rest 0
occam = []
for b in B_LIST:
    B10 = 1.0 + E_c / b
    occam.append(dict(b=b, B10=B10,
                      B_mix_A_others_uninformative=(B10 + N_eff - 1.0) / N_eff,
                      B_mix_B_others_excluded=B10 / N_eff,
                      B_mix_C_table_fraction=frac[3.0] * B10,
                      KR_B=kass_raftery(B10), KR_mixB=kass_raftery(B10 / N_eff),
                      KR_mixC=kass_raftery(frac[3.0] * B10)))
R["sec4_LEE_occam"] = dict(N_eff_trials=N_eff, N_eff_pratio=N_eff_ratio, ln_N_eff=math.log(N_eff),
                           table_cells_counted=n_cells, n_cells_below_0p05sigma=n_zero,
                           fraction_ge_threshold=frac, mixture=occam,
                           frequentist="p_global/p_local = %.1f; Bayesian: B -> B/N_eff (scenario B) or "
                                        "f(>=3sigma)*B (scenario C)" % N_eff_ratio)

# ---------------------------------------------------------------------------
# 5. Further events in the next exposure (scale k relative to 2.84 t yr)
# ---------------------------------------------------------------------------
def marg_like_dm(n1, n2, b, k, pdf, lo, hi):
    g = lambda s: stats.poisson.pmf(n1, s + b) * stats.poisson.pmf(n2, k * (s + b)) * float(pdf(s))
    val, _ = integrate.quad(g, lo, hi, limit=200)
    return val


def future_posterior(b, k, n2, pi_d, pi_u, LU, prior=CENTRAL_PRIOR):
    P = PRIORS[prior]
    lo, hi = P.get("a", 0.0), P.get("c", P.get("smax", 60.0))
    L_dm = marg_like_dm(1, n2, b, k, P["pdf"], lo, hi)
    L_k = stats.poisson.pmf(1, b) * stats.poisson.pmf(n2, k * b)
    mu_u = -math.log(1.0 - LU)  # unknown-background Poisson mean s.t. P(>=1|mu_u) = L_U in exposure 1
    L_u = stats.poisson.pmf(1, b + mu_u) * stats.poisson.pmf(n2, k * (b + mu_u))
    num = np.array([pi_d * L_dm, pi_u * L_u, (1 - pi_d - pi_u) * L_k])
    return num / num.sum(), mu_u


fut = []
for b in [B_BOX, 1e-3]:
    for k in [1.0, 2.0]:
        for (pi_d, pi_u, LU) in [(0.01, 0.0, 0.1), (0.05, 0.0, 0.1), (0.01, 0.1, 0.1), (0.05, 0.1, 0.1), (0.01, 0.01, 0.1)]:
            for n2 in [0, 1, 2, 3]:
                post, mu_u = future_posterior(b, k, n2, pi_d, pi_u, LU)
                fut.append(dict(b=b, k_exposure=k, pi_DM=pi_d, pi_U=pi_u, L_U=LU, mu_U=mu_u, n_new=n2,
                                P_DM=post[0], P_U=post[1], P_known=post[2]))
R["sec5_future_events"] = fut

# posterior predictive for the next equal exposure (k=1) under the 3-hypothesis posterior, central choices
def predictive(b, k, pi_d, pi_u, LU, prior=CENTRAL_PRIOR):
    P = PRIORS[prior]
    lo, hi = P.get("a", 0.0), P.get("c", P.get("smax", 60.0))
    mu_u = -math.log(1.0 - LU)
    p1b = stats.poisson.pmf(1, b)
    # posterior weights of hypotheses after the first event
    w = np.array([pi_d * (1 + P["E"] / b) * p1b, pi_u * stats.poisson.pmf(1, b + mu_u), (1 - pi_d - pi_u) * p1b])
    w /= w.sum()
    out = {}
    for n2 in [0, 1, 2]:
        L_dm = marg_like_dm(1, n2, b, k, P["pdf"], lo, hi) / ((1 + P["E"] / b) * p1b)
        L_u = stats.poisson.pmf(n2, k * (b + mu_u))
        L_k = stats.poisson.pmf(n2, k * b)
        out[f"P(n={n2})"] = float(w[0] * L_dm + w[1] * L_u + w[2] * L_k)
    out["P(n>=3)"] = 1.0 - sum(out.values())
    out["posterior_weights_DM_U_K"] = w.tolist()
    return out


R["sec5_posterior_predictive_k1"] = {f"b={b:g},pi_DM=0.01,pi_U=0.1,L_U=0.1": predictive(b, 1.0, 0.01, 0.1, 0.1) for b in [B_BOX, 1e-3]}
R["sec5_posterior_predictive_k1"]["b=0.0002,pi_DM=0.05,pi_U=0.0"] = predictive(B_BOX, 1.0, 0.05, 0.0, 0.1)

# ---------------------------------------------------------------------------
# Save tables
# ---------------------------------------------------------------------------
import csv  # noqa: E402


def write_csv(path, rows):
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


write_csv(os.path.join(OUT, "P001_table1_poisson_box.csv"), sec1)
write_csv(os.path.join(OUT, "P001_table2_bayes_factors.csv"), bf_rows)
write_csv(os.path.join(OUT, "P001_table3a_posterior_2hyp.csv"), rows2)
write_csv(os.path.join(OUT, "P001_table3b_posterior_3hyp.csv"), rows3)
write_csv(os.path.join(OUT, "P001_table4_occam.csv"), occam)
write_csv(os.path.join(OUT, "P001_table5_future_events.csv"), fut)
with open(os.path.join(OUT, "P001_results.json"), "w") as fh:
    json.dump(R, fh, indent=1, default=float)

# ---------------------------------------------------------------------------
# Figures (palette: dataviz reference instance; light surface)
# ---------------------------------------------------------------------------
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlecolor": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.3), gridspec_kw=dict(width_ratios=[1.15, 1]))
pis = np.logspace(-3, math.log10(0.5), 300)
for i, b in enumerate(B_LIST):
    B10 = 1.0 + E_c / b
    ax1.plot(pis, post2(pis, B10), color=C[i], lw=2, label=f"b = {b:g}  (B$_{{10}}$ = {B10:.0f}), vs modelled bkg only")
# with the unknown-background hypothesis, b = 2e-4
for ls, pi_u, LU in [("--", 0.1, 0.1), (":", 0.01, 0.1)]:
    B10 = 1.0 + E_c / B_BOX
    y = np.array([post3(p, pi_u, B10, B_BOX, LU)[0] for p in pis])
    ax1.plot(pis, y, color=C[0], lw=2, ls=ls, label=f"b = 2e-4, + unknown bkg ($\\pi_U$={pi_u}, L$_U$={LU})")
ax1.axhline(0.5, color=GRID, lw=1, zorder=0)
ax1.set_xscale("log")
ax1.set_xlabel("prior probability of DM in this window, $\\pi_{DM}$")
ax1.set_ylabel("posterior P(DM | one event)")
ax1.set_ylim(0, 1)
ax1.grid(axis="y", color=GRID, lw=0.6)
ax1.legend(fontsize=7.6, frameon=False, loc="upper left")
ax1.set_title("Posterior vs prior (signal prior: uniform s in [0, 10])", fontsize=10, loc="left")

# right: annotated grid of P(DM) for b = 2e-4, rows pi_DM, cols (pi_U, L_U)
cols = [(pu, lu) for pu in PI_U for lu in L_U]
B10c = 1.0 + E_c / B_BOX
M = np.array([[post3(pd_, pu, B10c, B_BOX, lu)[0] for (pu, lu) in cols] for pd_ in PI_DM])
cmap = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
im = ax2.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax2.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                 color="white" if M[i, j] > 0.45 else INK)
ax2.set_xticks(range(len(cols)))
ax2.set_xticklabels([f"$\\pi_U$={pu}\nL$_U$={lu}" for (pu, lu) in cols], fontsize=7.5)
ax2.set_yticks(range(len(PI_DM)))
ax2.set_yticklabels([f"$\\pi_{{DM}}$={p}" for p in PI_DM])
ax2.set_title(f"P(DM | event), b = 2e-4, B$_{{10}}$ = {B10c:.0f}, three hypotheses", fontsize=10, loc="left")
for s in ax2.spines.values():
    s.set_visible(False)
ax2.tick_params(length=0)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "P001_fig1_posterior_vs_prior.png"), dpi=170)
plt.close(fig)

# figure 2: Bayes factor vs b for the priors, with SBB bounds
fig, ax = plt.subplots(figsize=(6.2, 4.0))
bb = np.logspace(-4.5, -1.5, 200)
for i, name in enumerate(["uniform[0,3]", "uniform[0,10]", "uniform[0,30]", "log-uniform[0.01,30]"]):
    ax.plot(bb, 1.0 + PRIORS[name]["E"] / bb, color=C[i], lw=2, label=name)
ax.plot(bb, math.exp(-1) / bb + 1, color=INK2, lw=1, ls="--", label="maximum-likelihood bound (s = 1)")
ax.axhline(R["sec2_SBB_bound"]["B_max_from_p_local"], color=MUTED, lw=1, ls=":")
ax.text(bb[0], R["sec2_SBB_bound"]["B_max_from_p_local"] * 1.15, "SBB bound from local p (3.4σ)", fontsize=7.5, color=INK2)
ax.axhline(R["sec2_SBB_bound"]["B_max_from_p_global"], color=MUTED, lw=1, ls=":")
ax.text(bb[0], R["sec2_SBB_bound"]["B_max_from_p_global"] * 1.15, "SBB bound from global p (2.6σ)", fontsize=7.5, color=INK2)
for b in B_LIST:
    ax.axvline(b, color=GRID, lw=0.8, zorder=0)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("modelled background expectation in the box, b")
ax.set_ylabel("Bayes factor B$_{10}$ (one event)")
ax.grid(axis="y", color=GRID, lw=0.6)
ax.legend(fontsize=7.6, frameon=False)
ax.set_title("B$_{10}$ = 1 + E$_\\pi$[s e$^{-s}$]/b", fontsize=10, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "P001_fig2_bayes_factor_vs_b.png"), dpi=170)
plt.close(fig)

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
print("=== Section 1: Poisson in a box ===")
for r in sec1:
    print(f"b={r['b']:.4g}: P(>=1)={r['P_ge1']:.3e}  Z_exact={r['Z_one_sided_exact']:.2f}  s_hat={r['s_hat']:.4f}  "
          f"LR(s=1)={r['LR_at_s1']:.1f}  LR_max={r['LR_max']:.1f}  q0={r['q0']:.2f}  Z_PLR={r['Z_profile_asymptotic']:.2f}")
print("b_eff from 3.4 sigma: exact Poisson %.3e, asymptotic profile-LR %.3e; implied eps_box = %.2f / %.2f" % (
    b_eff_exact, b_eff_asym, B_BOX / b_eff_exact, B_BOX / b_eff_asym))
print("\n=== Section 2: Bayes factors ===")
for r in bf_rows:
    print(f"{r['prior']:30s} E={r['E_s_exp_minus_s']:.4f}  " + "  ".join(f"B(b={b:g})={r[f'B10_b={b:g}']:.1f}" for b in B_LIST))
print("SBB bounds:", {k: round(v, 1) for k, v in R["sec2_SBB_bound"].items()})
print("\n=== Section 3: posteriors (central prior uniform[0,10]) ===")
for b in B_LIST:
    B10 = 1.0 + E_c / b
    print(f"b={b:g} B10={B10:.1f}: " + "  ".join(f"pi={pi}: {post2(pi, B10):.3f}" for pi in PI_DM))
print("3-hyp, b=2e-4:")
for r in rows3:
    if r["b"] == B_BOX:
        print(f"  pi_DM={r['pi_DM']:<6} pi_U={r['pi_U']:<5} L_U={r['L_U']:<4} P_DM={r['P_DM']:.3f} P_U={r['P_U']:.3f} P_K={r['P_known']:.3f}")
print("small-b limit condition:", R["sec3_small_b_limit"]["condition_for_P_DM_gt_half"])
print("\n=== Section 4: LEE / Occam ===")
print(f"N_eff={N_eff:.2f} (ratio {N_eff_ratio:.2f}); cells={n_cells}; frac>=thr={frac}; zero-cells={n_zero}")
for r in occam:
    print(f"b={r['b']:g}: B={r['B10']:.1f} -> A {r['B_mix_A_others_uninformative']:.1f}, B {r['B_mix_B_others_excluded']:.1f}, C {r['B_mix_C_table_fraction']:.1f}")
print("\n=== Section 5: future events ===")
for r in fut:
    if r["k_exposure"] == 1.0:
        print(f"b={r['b']:g} k={r['k_exposure']} pi_DM={r['pi_DM']} pi_U={r['pi_U']} n_new={r['n_new']}: P_DM={r['P_DM']:.3f} P_U={r['P_U']:.3f} P_K={r['P_known']:.3f}")
print("predictive:", json.dumps(R["sec5_posterior_predictive_k1"], indent=1))
print("\nWrote", OUT)
