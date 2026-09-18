"""
P045 -- What a lower limit that lifts off zero means for one event:
coverage, flip-flopping and the Baxter et al. conventions.

Run from the simulation root:   .venv/bin/python output/code/P045_lower_limit.py

Parts
  A  single-bin Poisson, n = 1, b in {0, 5.7e-4, 1e-3, 1e-2}: exact Feldman-Cousins (FC),
     naive Delta chi^2 = 2.706, Cowan et al. asymptotic t~_mu (boundary-aware), toy-calibrated t~_mu,
     68.27 % intervals (Table I check), and the "smeared-calibration" lower edge mu_lo(f).
  B  spectral information: one-event unbinned likelihood L(s) ~ e^{-s}(s + beta), beta = b f_b/f_s.
  C  coverage: exact coverage of FC / asymptotic intervals; flip-flopping policies; the raster-scan
     toy with N_eff = 12 independent bins (family-wise "lift-off" rate under H0 and coverage under s = 1).
  D  the low-mass upper limits versus the sensitivity band (power-constraint check) from the P012 digitisation.
Outputs: output/work/P045/*.csv, P045_results.json, figures/*.png
"""
from __future__ import annotations
import json, sys, time
import numpy as np
import pandas as pd
from scipy import stats, optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "output/code")
from common import lzcommon as lz  # noqa: E402

t0 = time.time()
OUT = "output/work/P045"
FIG = f"{OUT}/figures"
import os
os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(45045)
RES: dict = {}

CL = 0.90
CHI2_90 = stats.chi2.ppf(CL, 1)          # 2.7055
CHI2_68 = 1.0
NMAX = 200

# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def pois(n, lam):
    return stats.poisson.pmf(n, lam)

def tstat(n, mu, b, bounded=True):
    """PLR statistic -2 ln[L(mu)/L(mu_hat)] for a single Poisson count n with expectation mu + b.
    bounded=True: mu_hat = max(0, n-b)  (Cowan's t~_mu / the FC ordering); False: mu_hat = n-b (t_mu, no boundary;
    for n < b we let mu+b -> 0 be the supremum, i.e. L(mu_hat)=1 when n=0)."""
    n = np.asarray(n, float)
    lam = mu + b
    muhat = np.maximum(n - b, 0.0) if bounded else (n - b)
    lamhat = np.where(bounded, muhat + b, np.maximum(n, 1e-300))
    with np.errstate(divide="ignore", invalid="ignore"):
        ll = -lam + n * np.log(np.maximum(lam, 1e-300))
        llh = -lamhat + n * np.log(np.maximum(lamhat, 1e-300))
        llh = np.where(n == 0, -lamhat, llh)
        ll = np.where(n == 0, -lam, ll)
    return 2.0 * (llh - ll)

def fc_acceptance(mu, b, cl=CL, nmax=NMAX):
    """FC acceptance region for a single Poisson count at signal mu, background b. Returns boolean array over n."""
    n = np.arange(nmax + 1)
    p = pois(n, mu + b)
    R = tstat(n, mu, b, bounded=True)      # larger t = smaller likelihood ratio
    order = np.argsort(R, kind="stable")   # add n in order of decreasing LR
    acc = np.zeros(nmax + 1, bool)
    cum = 0.0
    for i in order:
        acc[i] = True
        cum += p[i]
        if cum >= cl - 1e-12:
            break
    return acc

def fc_interval(nobs, b, cl=CL, mu_max=60.0):
    """Exact Neyman construction with FC ordering; edges found by bisection on the acceptance of nobs."""
    def accepted(mu):
        return fc_acceptance(mu, b, cl)[nobs]
    # coarse scan
    grid = np.concatenate([np.linspace(0, 2, 2001), np.linspace(2, mu_max, 5801)])
    acc = np.array([accepted(m) for m in grid])
    if not acc.any():
        return (np.nan, np.nan)
    idx = np.where(acc)[0]
    lo_i, hi_i = idx[0], idx[-1]
    # refine lower edge
    if lo_i == 0:
        lo = 0.0
    else:
        a, c = grid[lo_i - 1], grid[lo_i]
        for _ in range(40):
            m = 0.5 * (a + c)
            if accepted(m): c = m
            else: a = m
        lo = c
    a, c = grid[hi_i], grid[min(hi_i + 1, len(grid) - 1)]
    for _ in range(40):
        m = 0.5 * (a + c)
        if accepted(m): a = m
        else: c = m
    hi = a
    return (lo, hi)

def asym_interval(nobs, b, crit=CHI2_90, bounded=True):
    """Naive likelihood-ratio interval: {mu >= 0 : t(mu) <= crit}."""
    f = lambda mu: tstat(nobs, mu, b, bounded) - crit
    muhat = max(nobs - b, 0.0)
    lo = 0.0 if f(0.0) <= 0 else optimize.brentq(f, 1e-12, muhat)
    hi = optimize.brentq(f, muhat, muhat + 20 + 10 * np.sqrt(muhat + 1))
    return (lo, hi)

def cowan_cdf_tmu_tilde(t, mu, sigma):
    """Cowan et al. (2011) eq. for the cdf of t~_mu under mu' = mu (their Eqs. 63-65 / Sec. 3.7)."""
    t = np.asarray(t, float)
    r = mu / sigma
    F1 = 2 * stats.norm.cdf(np.sqrt(t)) - 1
    F2 = stats.norm.cdf(np.sqrt(t)) + stats.norm.cdf((t + r**2) / (2 * r)) - 1 if r > 0 else stats.norm.cdf(np.sqrt(t))
    return np.where(t <= r**2, F1, F2)

def cowan_interval(nobs, b, sigma_mode="fisher"):
    """Two-sided interval from the boundary-aware asymptotic t~_mu distribution: accept mu if p(mu) > 1-CL."""
    def sig(mu):
        if sigma_mode == "fisher":
            return np.sqrt(mu + b)                   # Var(mu_hat) from the Fisher information at mu
        q = 2 * ((mu + b) - b - (b * np.log((mu + b) / b) if b > 0 else 0.0))   # Asimov (mu'=0) q_muA
        return mu / np.sqrt(q) if q > 0 else np.inf
    def pval(mu):
        tobs = float(tstat(nobs, mu, b, True))
        return 1.0 - float(cowan_cdf_tmu_tilde(tobs, mu, sig(mu)))
    muhat = max(nobs - b, 0.0)
    g = lambda mu: pval(mu) - (1 - CL)
    lo = 0.0 if g(1e-9) >= 0 else optimize.brentq(g, 1e-9, muhat)
    hi = optimize.brentq(g, muhat, muhat + 30)
    return (lo, hi)

def toy_interval(nobs, b, ntoy=30000, mu_grid=None):
    """Toy-MC calibrated t~_mu Neyman construction (single bin). Critical value = smallest t with
    P(t~ <= t) >= CL (inverted cdf); accept if t~_obs <= c."""
    if mu_grid is None:
        mu_grid = np.concatenate([np.arange(0.02, 1.0, 0.002), np.arange(1.0, 8.0, 0.01)])
    acc = np.zeros(len(mu_grid), bool)
    for i, mu in enumerate(mu_grid):
        n = rng.poisson(mu + b, ntoy)
        t = tstat(n, mu, b, True)
        c = np.quantile(t, CL, method="inverted_cdf")
        acc[i] = tstat(nobs, mu, b, True) <= c + 1e-9
    idx = np.where(acc)[0]
    return (mu_grid[idx[0]], mu_grid[idx[-1]])

def central_interval(nobs, b, cl=CL):
    """Classical central (Neyman) two-sided interval, alpha/2 in each tail."""
    a = (1 - cl) / 2
    lo = 0.0 if stats.poisson.sf(nobs - 1, b) > a else optimize.brentq(lambda m: stats.poisson.sf(nobs - 1, m + b) - a, 0, 100)
    fh = lambda m: stats.poisson.cdf(nobs, m + b) - a
    hi = 0.0 if fh(0.0) <= 0 else optimize.brentq(fh, 0, 200)     # empty/zero interval when n << b
    return (lo, hi)

def upper_limit_1s(nobs, b, cl=CL):
    fh = lambda m: stats.poisson.cdf(nobs, m + b) - (1 - cl)
    return 0.0 if fh(0.0) <= 0 else optimize.brentq(fh, 0, 200)

# ----------------------------------------------------------------------------------------------
# Part A: single-event intervals
# ----------------------------------------------------------------------------------------------
print("== Part A: single-bin intervals for n = 1 ==")
BLIST = [0.0, 5.7e-4, 1e-3, 1e-2]
rowsA = []
for b in BLIST:
    fc = fc_interval(1, b)
    asy = asym_interval(1, b)
    asy_unb = asym_interval(1, b, bounded=False)
    cow_f = cowan_interval(1, b, "fisher")
    cow_a = cowan_interval(1, b, "asimov0")
    toy = toy_interval(1, b, ntoy=30000)
    fc68 = fc_interval(1, b, cl=0.6827)
    asy68 = asym_interval(1, b, crit=CHI2_68)
    cen = central_interval(1, b)
    rowsA.append(dict(b=b, FC_lo=fc[0], FC_hi=fc[1], toy_lo=toy[0], toy_hi=toy[1],
                      asym_lo=asy[0], asym_hi=asy[1], asym_unbounded_lo=asy_unb[0], asym_unbounded_hi=asy_unb[1],
                      cowan_fisher_lo=cow_f[0], cowan_fisher_hi=cow_f[1], cowan_asimov_lo=cow_a[0], cowan_asimov_hi=cow_a[1],
                      central_lo=cen[0], central_hi=cen[1],
                      FC68_lo=fc68[0], FC68_hi=fc68[1], asym68_lo=asy68[0], asym68_hi=asy68[1],
                      FC_lo_analytic=max(-np.log(CL) - b, 0.0)))
    print(f"b={b:.1e}: FC [{fc[0]:.4f},{fc[1]:.3f}]  toy [{toy[0]:.3f},{toy[1]:.3f}]  asym [{asy[0]:.4f},{asy[1]:.3f}] "
          f"Cowan-F [{cow_f[0]:.3f},{cow_f[1]:.3f}] Cowan-A [{cow_a[0]:.3f},{cow_a[1]:.3f}] central [{cen[0]:.3f},{cen[1]:.3f}] "
          f"FC68 [{fc68[0]:.3f},{fc68[1]:.3f}] asym68 [{asy68[0]:.3f},{asy68[1]:.3f}]")
dfA = pd.DataFrame(rowsA)
dfA.to_csv(f"{OUT}/P045_single_event_intervals.csv", index=False)
RES["partA"] = dfA.to_dict("records")

# expected (n = 0) upper limits = median sensitivity in the background-free regime
b0 = 5.7e-4
RES["n0_upper_limits"] = dict(FC=fc_interval(0, b0)[1], asym=asym_interval(0, b0)[1], cowan_fisher=cowan_interval(0, b0)[1],
                              one_sided=upper_limit_1s(0, b0))
print("n=0 upper limits (b=5.7e-4):", RES["n0_upper_limits"])

# --- LZ digitised edges (P012, P017 for Fig. 6 bottom; P007/P011 for Fig. 6 top / Fig. S7) ---
dig = pd.read_csv("output/work/P012/fig6_bottom_digitised.csv")
N_UNIT = 12.9                      # events per unit d10^2 at 1000 GeV in LZ's normalisation (P012)
r1000 = dig[dig.m_GeV == 1000].iloc[0]
lz_events_P012 = dict(lower=N_UNIT * r1000.d10_lower**2, median=N_UNIT * r1000.d10_median_sens**2, upper=N_UNIT * r1000.d10_upper**2)
lz_events_P017 = dict(lower=N_UNIT * 0.154**2, median=N_UNIT * 0.375**2, upper=N_UNIT * 0.553**2)
ratio_bottom = (r1000.d10_lower / r1000.d10_upper) ** 2
# top panel (O1 inelastic, 1000 GeV) converted intervals from P011 (cm^2, proportional to events) and P007 upper edges in events
top_ratios = {"300 keV": 1.5e-42 / 3.0e-41, "350 keV": 1.1e-40 / 1.8e-39}
top_lower_events = {k: v * u for (k, v), u in zip(top_ratios.items(), [3.65, 3.65])}
RES["digitised"] = dict(P012_events=lz_events_P012, P017_events=lz_events_P017, ratio_lower_over_upper_bottom=ratio_bottom,
                        top_panel_ratio_lower_over_upper=top_ratios,
                        top_panel_lower_edge_events_if_upper_3p65=top_lower_events,
                        top_panel_lower_edge_events_if_upper_4p36={k: v * 4.36 for k, v in top_ratios.items()})
print("Fig.6 bottom (P012) events:", lz_events_P012, " ratio lo/up", ratio_bottom)
print("Fig.6 top ratios:", top_ratios, "-> lower edges", top_lower_events)

# CL that would give the digitised lower edge in an FC construction from one background-free count: mu_lo = -ln(CL)
RES["CL_equivalent_of_lower_edge"] = {f"{x:.2f} events": float(np.exp(-x)) for x in (0.105, 0.18, 0.2, 0.31, 0.38)}
print("CL equivalent (FC, n=1, b=0) of lower edges:", RES["CL_equivalent_of_lower_edge"])

# --- smeared-calibration lower edge mu_lo(f): p(mu) = f P(1|mu+b) + P(>=2|mu+b) = 0.10 ---
def mu_lo_smeared(f, b=b0):
    g = lambda mu: f * pois(1, mu + b) + stats.poisson.sf(1, mu + b) - (1 - CL)
    return optimize.brentq(g, 1e-6, 5.0)
fgrid = [1.0, 0.9, 0.75, 0.5, 0.31, 0.25, 0.15]
RES["smeared_lower_edge"] = {str(f): mu_lo_smeared(f) for f in fgrid}
f_req = {}
for target in (0.18, 0.2, 0.31, 0.38):
    f_req[str(target)] = float(optimize.brentq(lambda f: mu_lo_smeared(f) - target, 0.05, 1.0))
RES["f_required_for_lower_edge"] = f_req
print("mu_lo(f):", RES["smeared_lower_edge"]); print("f required:", f_req)

# ----------------------------------------------------------------------------------------------
# Part B: spectral information -- one-event unbinned likelihood  L(s) ~ e^{-s} (s + beta)
# ----------------------------------------------------------------------------------------------
print("\n== Part B: spectral information ==")
rowsB = []
for b_region, label in [(2.92, "NR band 5.4-270 keV (P008)"), (1713.0, "science sample (Table I)")]:
    for r in [1e2, 1e3, 1e4, 1e5, 1e6]:
        beta = b_region / r
        if beta >= 1:
            rowsB.append(dict(region=label, b_region=b_region, ratio_fs_over_fb=r, beta=beta, shat=0.0,
                              asym_lo=np.nan, asym_hi=np.nan, FC_lo=np.nan, FC_hi=np.nan, Z_asym=0.0)); continue
        asy = asym_interval(1, beta); fc = fc_interval(1, beta)
        q0 = float(tstat(1, 0.0, beta, True))
        rowsB.append(dict(region=label, b_region=b_region, ratio_fs_over_fb=r, beta=beta, shat=1 - beta,
                          asym_lo=asy[0], asym_hi=asy[1], FC_lo=fc[0], FC_hi=fc[1], Z_asym=np.sqrt(q0)))
dfB = pd.DataFrame(rowsB)
dfB.to_csv(f"{OUT}/P045_spectral_beta_intervals.csv", index=False)
print(dfB.to_string(index=False))
# beta from the paper's 3.4 sigma (asymptotic): 2[ln(1/beta) - 1 + beta] = 3.4^2
beta_34 = optimize.brentq(lambda be: 2 * (np.log(1 / be) - 1 + be) - 3.4**2, 1e-8, 0.5)
RES["partB"] = dict(beta_from_3p4sigma_asymptotic=beta_34, ratio_needed_science_sample=1713 / beta_34,
                    ratio_needed_NR_band=2.92 / beta_34, lower_edge_at_beta34=asym_interval(1, beta_34)[0])
print("beta(3.4 sigma) =", beta_34, " r needed:", RES["partB"])

# ----------------------------------------------------------------------------------------------
# Part C1: exact coverage of the single-bin constructions
# ----------------------------------------------------------------------------------------------
print("\n== Part C1: coverage ==")
mu_true = np.concatenate([np.linspace(0, 3, 301), np.linspace(3.02, 12, 450)])
NCOV = 60
def coverage_from_intervals(intervals, b, mus):
    cov = np.zeros(len(mus))
    for n, (lo, hi) in enumerate(intervals):
        inside = (mus >= lo - 1e-12) & (mus <= hi + 1e-12)
        cov += inside * pois(n, mus + b)
    return cov

cov_tables = {}
for b in [5.7e-4, 1.0, 3.0]:
    ints = {"FC": [fc_interval(n, b) for n in range(NCOV)],
            "asym": [asym_interval(n, b) for n in range(NCOV)],
            "cowan": [cowan_interval(n, b, "fisher") for n in range(NCOV)]}
    # flip-flopping policies (threshold 3 sigma on the local one-sided Poisson significance)
    Z = lambda n: stats.norm.isf(max(stats.poisson.sf(n - 1, b), 1e-300)) if n > 0 else -np.inf
    ints["flipflop_classical"] = [central_interval(n, b) if Z(n) >= 3 else (0.0, upper_limit_1s(n, b)) for n in range(NCOV)]
    ints["baxter_FC_UL_below_3sigma"] = [ints["FC"][n] if Z(n) >= 3 else (0.0, ints["FC"][n][1]) for n in range(NCOV)]
    ints["flipflop_FC_vs_1sidedUL"] = [ints["FC"][n] if Z(n) >= 3 else (0.0, upper_limit_1s(n, b)) for n in range(NCOV)]
    tab = pd.DataFrame({"mu": mu_true})
    for k, v in ints.items():
        tab[k] = coverage_from_intervals(v, b, mu_true)
    tab.to_csv(f"{OUT}/P045_coverage_b{b:g}.csv", index=False)
    cov_tables[b] = tab
    summ = {k: dict(min=float(tab[k].min()), at_mu=float(tab.mu[tab[k].idxmin()]),
                    mean_0_5=float(tab[k][tab.mu <= 5].mean())) for k in ints}
    RES[f"coverage_b{b:g}"] = summ
    print(f"b={b}:", {k: (round(v['min'], 3), round(v['at_mu'], 2)) for k, v in summ.items()})
    if b == 5.7e-4:
        for mu_q in (0.3, 0.5, 1.0, 1.5, 2.0, 3.0):
            i = int(np.argmin(np.abs(mu_true - mu_q)))
            print(f"   mu={mu_q}: FC {tab.FC[i]:.3f} asym {tab.asym[i]:.3f} cowan {tab.cowan[i]:.3f}")
        RES["coverage_b5.7e-4_points"] = {str(mu_q): {k: float(tab[k][int(np.argmin(np.abs(mu_true - mu_q)))]) for k in ("FC", "asym", "cowan")}
                                          for mu_q in (0.3, 0.5, 1.0, 1.5, 2.0, 3.0)}

# ----------------------------------------------------------------------------------------------
# Part C2: raster scan toy -- 12 independent resolution-wide bins with the NR-band background model
# ----------------------------------------------------------------------------------------------
print("\n== Part C2: raster-scan toy ==")
edges = np.array([5.4, 9, 14, 22, 33, 48, 66, 90, 118, 150, 190, 230, 270.0])
def exp_frac(E1, E2, lam, Elo=5.4, Ehi=270.0):
    return (np.exp(-E1 / lam) - np.exp(-E2 / lam)) / (np.exp(-Elo / lam) - np.exp(-Ehi / lam))
def bin_backgrounds(scale_high=1.0):
    b = np.zeros(len(edges) - 1)
    for i, (E1, E2) in enumerate(zip(edges[:-1], edges[1:])):
        acc = 2.7 * exp_frac(E1, E2, 15.0)                      # accidentals, 15 keV exponential (P008)
        atm = 0.11 * exp_frac(E1, E2, 28.6)                     # atmospheric-nu CEvNS, scale from P019 (1e-4 above 200 keV)
        mssi = 0.0049 * max(0.0, min(E2, 270) - max(E1, 50)) / 220.0   # MSSI flat above 50 keV
        neu = 0.05 * exp_frac(E1, E2, 50.0)                     # neutrons, 50 keV exponential (assumption)
        b[i] = acc + atm + mssi + neu
    b[-2:] *= scale_high
    return b
bbins = bin_backgrounds()
print("bin backgrounds:", np.round(bbins, 5), " total", bbins.sum(), " 190-270:", bbins[-2:].sum())

def liftoff_prob_bin(b, construction="FC"):
    """P(90% two-sided interval excludes 0 | H0) for one Poisson bin."""
    n = np.arange(0, 80)
    if construction == "FC":
        acc0 = fc_acceptance(0.0, b)            # acceptance region of mu = 0
        lifts = ~acc0[n]
    else:                                        # asymptotic Delta chi^2: 0 excluded iff t(0) > 2.706 with mu_hat > 0
        lifts = (n > b) & (tstat(n, 0.0, b, True) > CHI2_90)
    return float(np.sum(pois(n, b)[lifts]))

rowsC = []
for con in ("FC", "asym"):
    p_i = np.array([liftoff_prob_bin(b, con) for b in bbins])
    # order bins from high energy downward: M = 1 is the 230-270 keV bin (heavy models), M = 12 is everything
    p_rev = p_i[::-1]
    fam = 1 - np.cumprod(1 - p_rev)
    for M in range(1, 13):
        rowsC.append(dict(construction=con, M=M, bins_included=f"{edges[12 - M]:.0f}-270 keV", p_family_liftoff_H0=fam[M - 1]))
    RES[f"liftoff_per_bin_{con}"] = dict(zip([f"{a:g}-{c:g}" for a, c in zip(edges[:-1], edges[1:])], p_i.tolist()))
    print(con, "per-bin lift-off prob:", np.round(p_i, 4), " family-wise (M=12):", fam[-1])
dfC = pd.DataFrame(rowsC)
dfC.to_csv(f"{OUT}/P045_raster_liftoff.csv", index=False)

# variations: high-energy background scaled down to P016's b_H (5.7e-4 in 200-270) and up x3
for sc, lab in [(5.7e-4 / bbins[-2:].sum(), "bH_5.7e-4"), (3.0, "x3_high")]:
    bb = bin_backgrounds(sc)
    p_i = np.array([liftoff_prob_bin(b, "FC") for b in bb])
    RES[f"family_liftoff_FC_{lab}"] = float(1 - np.prod(1 - p_i))
# analytic N_eff version
RES["family_liftoff_from_Neff"] = {str(ne): float(1 - CL**ne) for ne in (1, 2, 3, 4, 6, 12)}
print("family lift-off variants:", {k: v for k, v in RES.items() if k.startswith("family_liftoff")})

# Monte Carlo of the full procedure under H0 and under s = 1 (signal in the 230-270 keV bin)
NEXP = 200000
def simulate(s_true, s_bin=11):
    lam = bbins.copy(); lam[s_bin] += s_true
    n = rng.poisson(lam, size=(NEXP, len(lam)))
    # lift-off per bin: n > n_crit(b_i), with n_crit from the FC acceptance at mu=0
    ncrit = np.array([np.max(np.where(fc_acceptance(0.0, b))[0]) for b in bbins])
    lifts = n > ncrit
    # coverage of the true-model interval (bin s_bin) for FC and asym
    fc_ints = [fc_interval(k, bbins[s_bin]) for k in range(40)]
    as_ints = [asym_interval(k, bbins[s_bin]) for k in range(40)]
    nn = np.minimum(n[:, s_bin], 39)
    cov_fc = np.array([(fc_ints[k][0] <= s_true <= fc_ints[k][1]) for k in range(40)])[nn]
    cov_as = np.array([(as_ints[k][0] <= s_true <= as_ints[k][1]) for k in range(40)])[nn]
    return dict(s_true=s_true, p_any_liftoff=float(lifts.any(1).mean()), p_liftoff_true_bin=float(lifts[:, s_bin].mean()),
                p_liftoff_other_bins=float(lifts[:, [i for i in range(12) if i != s_bin]].any(1).mean()),
                coverage_true_model_FC=float(cov_fc.mean()), coverage_true_model_asym=float(cov_as.mean()),
                mean_n_liftoff_models=float(lifts.sum(1).mean()))
RES["mc_procedure"] = [simulate(s) for s in (0.0, 1.0, 2.4)]
for r in RES["mc_procedure"]: print(r)

# ----------------------------------------------------------------------------------------------
# Part D: low-mass upper limits versus sensitivity band (power-constraint check), Gaussian regime
# ----------------------------------------------------------------------------------------------
print("\n== Part D: power constraint check ==")
rowsD = []
z90 = stats.norm.isf(0.10)  # 1.2816: one-sided 90 % / two-sided upper edge in the Gaussian regime uses 1.64 for 95%; see below
# upper edge of a 90 % two-sided Gaussian interval = mu_hat + 1.645 sigma; median sensitivity 1.645 sigma; -1 sigma band 0.645 sigma
for _, r in dig.iterrows():
    rho = (r.d10_upper / r.d10_median_sens) ** 2      # observed/median in events
    muhat_over_sigma = 1.645 * (rho - 1)
    pcl16 = max(rho, 0.645 / 1.645)                    # power constraint at the -1 sigma band (Baxter/PCL M_min = 0.16, recalled)
    rowsD.append(dict(m_GeV=r.m_GeV, ratio_d10=r.d10_upper / r.d10_median_sens, ratio_events=rho, muhat_over_sigma=muhat_over_sigma,
                      pcl_factor_events=pcl16 / rho, lifts_off=not np.isnan(r.d10_lower)))
dfD = pd.DataFrame(rowsD)
dfD.to_csv(f"{OUT}/P045_pcl_check.csv", index=False)
print(dfD.to_string(index=False))
RES["partD"] = dfD.to_dict("records")

RES["runtime_s"] = time.time() - t0
RES["bin_edges_keV"] = edges.tolist(); RES["bin_backgrounds"] = bbins.tolist()
with open(f"{OUT}/P045_results.json", "w") as f:
    json.dump(RES, f, indent=1, default=float)
from P045_figures import make_figures  # noqa: E402  (reads the tables written above)
make_figures()
print(f"\ndone in {RES['runtime_s']:.1f} s")
