"""
P045 figures -- reads the tables written by P045_lower_limit.py and draws the three PNGs.
Run from the simulation root:  .venv/bin/python output/code/P045_figures.py
(also called at the end of P045_lower_limit.py).
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "output/work/P045"
FIG = f"{OUT}/figures"
CL = 0.90
# dataviz reference palette (categorical slots 1-7, fixed order)
C1, C2, C3, C4, C5, C6 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"
INK2 = "#52514e"


def make_figures():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "grid.linewidth": 0.6})
    RES = json.load(open(f"{OUT}/P045_results.json"))
    dfA = pd.read_csv(f"{OUT}/P045_single_event_intervals.csv")
    dfC = pd.read_csv(f"{OUT}/P045_raster_liftoff.csv")
    cov5 = pd.read_csv(f"{OUT}/P045_coverage_b0.00057.csv")
    cov1 = pd.read_csv(f"{OUT}/P045_coverage_b1.csv")
    d = RES["digitised"]
    P012, P017 = d["P012_events"], d["P017_events"]
    top_lo = min(d["top_panel_lower_edge_events_if_upper_3p65"].values())

    # ---- Fig 1: interval edges by construction (b = 5.7e-4) versus what LZ plotted ----
    rA = dfA[np.isclose(dfA.b, 5.7e-4)].iloc[0]
    items = [("Feldman-Cousins 90 %", rA.FC_lo, rA.FC_hi, C1),
             ("toy-calibrated $\\tilde t_\\mu$ 90 % (3$\\times$10$^4$ toys/point)", rA.toy_lo, rA.toy_hi, C1),
             ("naive $\\Delta\\chi^2 = 2.706$", rA.asym_lo, rA.asym_hi, C2),
             ("Cowan asymptotic $\\tilde t_\\mu$, $\\sigma^2 = \\mu + b$", rA.cowan_fisher_lo, rA.cowan_fisher_hi, C2),
             ("Cowan asymptotic $\\tilde t_\\mu$, Asimov $\\sigma$", rA.cowan_asimov_lo, rA.cowan_asimov_hi, C2),
             ("classical central 90 %", rA.central_lo, rA.central_hi, C6),
             ("FC 68.27 %", rA.FC68_lo, rA.FC68_hi, C3),
             ("$\\Delta\\chi^2 = 1$ (68 %)", rA.asym68_lo, rA.asym68_hi, C3),
             ("LZ Table I: 1.0 (+1.4, $-$0.7)", 0.3, 2.4, C4),
             ("LZ Fig. 6 bottom, 1000 GeV, digitised (P012)", P012["lower"], P012["upper"], C5),
             ("LZ Fig. 6 bottom, 1000 GeV, digitised (P017)", P017["lower"], P017["upper"], C5),
             ("LZ Fig. 6 top, $\\delta$ = 300 keV (P011 ratio $\\times$ 3.65)", top_lo, 3.65, C5)]
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for i, (lab, lo, hi, c) in enumerate(items[::-1]):
        ax.plot([lo, hi], [i, i], color=c, lw=2.2, solid_capstyle="round")
        ax.plot([lo], [i], marker="|", color=c, ms=9, mew=2); ax.plot([hi], [i], marker="|", color=c, ms=9, mew=2)
        ax.text(hi * 1.08, i, f"[{lo:.3g}, {hi:.3g}]", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([it[0] for it in items[::-1]], fontsize=8.5)
    ax.set_xscale("log"); ax.set_xlim(0.04, 30)
    ax.set_xlabel("expected signal events $s$  (one observed event, $b = 5.7\\times10^{-4}$)")
    ax.axvline(1.0, color=INK2, lw=0.8, ls=":"); ax.text(1.04, len(items) - 0.7, "$\\hat s = 1$", fontsize=8, color=INK2)
    ax.set_title("Two-sided intervals for one event: constructions versus LZ", fontsize=10.5)
    fig.tight_layout(); fig.savefig(f"{FIG}/P045_fig1_intervals_by_construction.png", dpi=180); plt.close(fig)

    # ---- Fig 2: coverage versus true mu ----
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.9), sharey=True)
    axs[0].plot(cov5.mu, cov5.FC, color=C1, lw=2, label="Feldman-Cousins (= toy-calibrated $\\tilde t_\\mu$)")
    axs[0].plot(cov5.mu, cov5.cowan, color=C3, lw=2, label="Cowan asymptotic $\\tilde t_\\mu$")
    axs[0].plot(cov5.mu, cov5.asym, color=C2, lw=2, label="naive $\\Delta\\chi^2 = 2.706$")
    axs[0].axhline(0.9, color=INK2, lw=0.8, ls="--"); axs[0].set_xlim(0, 8); axs[0].set_ylim(0.6, 1.005)
    axs[0].set_xlabel("true signal $\\mu$ (events)"); axs[0].set_ylabel("coverage of the 90 % interval")
    axs[0].set_title("$b = 5.7\\times10^{-4}$: single-model constructions", fontsize=10)
    axs[0].legend(fontsize=8, loc="lower right", frameon=False)
    axs[1].plot(cov1.mu, cov1.FC, color=C1, lw=2, label="FC always (LZ procedure)")
    axs[1].plot(cov1.mu, cov1.baxter_FC_UL_below_3sigma, color=C3, lw=2, ls="--", label="FC upper edge only below 3$\\sigma$ (Baxter et al.)")
    axs[1].plot(cov1.mu, cov1.flipflop_classical, color=C2, lw=2, label="flip-flop: 1-sided UL / central interval at 3$\\sigma$")
    axs[1].plot(cov1.mu, cov1.flipflop_FC_vs_1sidedUL, color=C4, lw=1.6, ls=":", label="flip-flop: 1-sided UL / FC at 3$\\sigma$")
    axs[1].axhline(0.9, color=INK2, lw=0.8, ls="--"); axs[1].set_xlim(0, 12)
    axs[1].set_xlabel("true signal $\\mu$ (events)"); axs[1].set_title("$b = 1$: reporting policies with a 3$\\sigma$ threshold", fontsize=10)
    axs[1].legend(fontsize=7.5, loc="lower right", frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIG}/P045_fig2_coverage.png", dpi=180); plt.close(fig)

    # ---- Fig 3: family-wise lift-off probability under H0 versus number of independent windows ----
    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    for con, c, lab in (("FC", C1, "FC / toy-calibrated (discrete counts, 12-bin toy)"), ("asym", C2, "naive $\\Delta\\chi^2$ (12-bin toy)")):
        dd = dfC[dfC.construction == con]
        ax.plot(dd.M, dd.p_family_liftoff_H0, marker="o", ms=5, lw=2, color=c, label=lab)
    Ne = np.arange(1, 13); ax.plot(Ne, 1 - CL**Ne, color=INK2, lw=1.2, ls="--", label="$1 - 0.9^{N_{\\rm eff}}$ (continuous statistic)")
    ax.axhline(4.7e-3, color=C4, lw=1.2, ls=":")
    ax.text(6.2, 3.3e-3, "LZ global $p = 4.7\\times10^{-3}$ (2.6$\\sigma$)", fontsize=8, color=INK2, va="top")
    ax.set_yscale("log"); ax.set_ylim(2e-4, 1.5); ax.set_xticks(range(1, 13))
    ax.set_xlabel("number $M$ of independent windows in the scan\n(1 = 230-270 keV only; 12 = the whole 5.4-270 keV NR band)")
    ax.set_ylabel("P(some 90 % lower limit lifts off | H$_0$)"); ax.legend(fontsize=8, frameon=False, loc="center right")
    ax.set_title("A non-zero lower limit somewhere is expected in 25-70 % of null experiments", fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(f"{FIG}/P045_fig3_liftoff_vs_M.png", dpi=180); plt.close(fig)
    print("figures written to", FIG)


if __name__ == "__main__":
    make_figures()
