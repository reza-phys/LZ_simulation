"""
P083 -- Base rates of ~2.5-4 sigma single-event / few-event / small-excess anomalies
in direct detection and other rare-event searches, and what they did next.

Everything in COMPILATION below is RECALLED KNOWLEDGE (no internet, no database).
Each item carries an `item_rel` flag (certain | likely | uncertain) for its existence and
outcome, and a `sig_rel` flag for the quoted significance. Years and times-to-resolution are
recalled to ~0.5 yr. The list is INCOMPLETE by construction; see the selection analysis.

Outputs (all under output/work/P083/):
  anomalies.csv                 the compilation
  P083_beta_table.csv           Beta posteriors for P(real) by stratum / variant
  P083_survival.csv             Kaplan-Meier curves
  P083_routes.csv               resolution routes
  P083_results.json             every headline number used in the paper
  figures/P083_fig1_timeline.png, P083_fig2_survival.png, P083_fig3_beta.png
Run:  .venv/bin/python output/code/P083_base_rates.py      (a few seconds)
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "output/code")
from common import lzcommon as lz  # noqa: F401  (LZ numbers; nothing else needed)

ROOT = "/Users/reza/LZ_simulation"
os.chdir(ROOT)
OUT = "output/work/P083"
FIG = f"{OUT}/figures"
os.makedirs(FIG, exist_ok=True)
NOW = 2026.7  # simulated date 2026-09-16, decimal year

# --------------------------------------------------------------------------------------
# 1. The compilation (recalled). Columns:
# id, name, year, domain, kind, n_events, sig_local, sig_global, sig_rel, prior_type,
# outcome, t_res_yr, route, indep_comparable, item_rel, notes
# kind: few_event (<= ~5 events) | small_excess | modulation | line | precision
# prior_type: new (new physics / unpredicted) | expected (a predicted SM/astrophysical process)
# outcome: real | statistical | background | artefact | theory_revision | contradicted | unresolved
# route: more_data_same | independent_expt | reanalysis | calibration_hardware | theory | none
# indep_comparable: was an independent experiment of comparable sensitivity available when claimed?
# --------------------------------------------------------------------------------------
C = []
def add(*a):
    C.append(dict(zip(["id", "name", "year", "domain", "kind", "n_events", "sig_local", "sig_global",
                       "sig_rel", "prior_type", "outcome", "t_res_yr", "route", "indep_comparable",
                       "item_rel", "notes"], a)))
NA = np.nan
# ---- direct detection (2008-2026) --------------------------------------------------------
add("A01", "DAMA/LIBRA annual modulation", 2008, "DD", "modulation", "many (modulation of ~1e4 events)",
    8.2, NA, "likely", "new", "contradicted", 10, "independent_expt", "no", "certain",
    "DAMA/NaI 1998 -> DAMA/LIBRA 2008 (~8 sigma), 12.9 sigma by 2018; same-target NaI(Tl) tests COSINE-100 (2018 Nature: no DM-compatible excess; later modulation null) and ANAIS-112 (2019-2025, excludes DAMA modulation at >3-4 sigma); cause still unidentified")
add("A02", "CoGeNT low-energy excess and 2.8 sigma modulation", 2010, "DD", "small_excess", "hundreds of events < 3 keVee",
    2.8, NA, "likely", "new", "background", 4, "reanalysis", "yes", "certain",
    "modulation claim 2011; surface-event (slow-pulse) leakage; CoGeNT 2014 reanalysis and Davis et al. 2014; excluded by CDMSlite/SuperCDMS/LUX")
add("A03", "CDMS-II silicon three events", 2013, "DD", "few_event", "3 (0.7 expected)",
    3.0, NA, "likely", "new", "statistical", 0.5, "independent_expt", "yes", "certain",
    "profile-likelihood p = 0.19 % (~3 sigma) for WIMP+bkg vs bkg; 5.4 % that known backgrounds give >= 3 events; LUX Oct 2013 and SuperCDMS 2014 excluded the region")
add("A04", "CRESST-II excess (Run 32)", 2011, "DD", "small_excess", "67 accepted events (~30 excess)",
    4.2, NA, "likely", "new", "background", 3, "calibration_hardware", "yes", "certain",
    "two maxima at 4.2 and 4.7 sigma; traced to alpha/Pb-recoil backgrounds from the bronze clamps; 2014 upgraded modules showed no excess")
add("A05", "XENON1T electronic-recoil excess", 2020, "DD", "small_excess", "285 obs vs 232 exp (53 excess) at 1-7 keV",
    3.3, NA, "certain", "new", "contradicted", 2, "independent_expt", "yes", "certain",
    "3.3 sigma solar axion, 3.2 sigma tritium alternative; XENONnT 2022 (5x lower ER background) saw no excess; cause never identified (tritium or fluctuation)")
add("A06", "XENON100 annual modulation (ER band)", 2015, "DD", "modulation", "single-scatter ER events",
    2.8, NA, "likely", "new", "statistical", 2, "more_data_same", "yes", "likely",
    "2.8 sigma periodicity; DAMA-like interpretation excluded at 4.8 sigma in the same paper; 4-year dataset (2017) reduced it to ~1.9 sigma")
add("A07", "EDELWEISS-II five events", 2011, "DD", "few_event", "5 (~3 expected)",
    NA, NA, "unknown", "new", "statistical", 1, "more_data_same", "yes", "uncertain",
    "recalled only vaguely; sub-2 sigma; superseded by EDELWEISS-III and the CDMS/EDELWEISS combination; EXCLUDED from primary counts")
add("A08", "CDMS-II germanium two events", 2009, "DD", "few_event", "2 (0.8 expected)",
    0.7, NA, "likely", "new", "statistical", 1, "independent_expt", "yes", "certain",
    "p = 0.23 for >= 2 background events; below the 2.5 sigma window; region excluded by XENON100 2010-2011")
add("A09", "Cryogenic low-energy excess (CRESST-III, EDELWEISS, SuperCDMS-CPD, NUCLEUS)", 2020, "DD", "small_excess",
    "thousands of events rising below ~200 eV", NA, NA, "unknown", "new", "unresolved", NA, "none", "yes", "certain",
    "EXCESS workshop 2021; leading hypothesis stress/relaxation artefact, not DM; no significance quoted; censored")
add("A10", "XENON1T 124Xe double electron capture", 2019, "DD", "small_excess", "126 events",
    4.4, NA, "likely", "expected", "real", 3, "independent_expt", "yes", "certain",
    "predicted SM process (half-life 1.8e22 yr); confirmed by XENONnT 2022 and LZ 2024; above the 4 sigma window")
add("A11", "8B solar-neutrino CEvNS (PandaX-4T 2.64 sigma, XENONnT 2.73 sigma)", 2024, "DD", "small_excess",
    "~3.5 (PandaX-4T) and ~11 (XENONnT) events", 2.7, NA, "certain", "expected", "unresolved", NA, "none", "yes", "certain",
    "predicted SM rate; two experiments each ~2.7 sigma; not yet 5 sigma as of Sept 2026 to our knowledge; censored")
# ---- broader rare-event / anomaly class ---------------------------------------------------
add("B01", "ATLAS/CMS 750 GeV diphoton", 2015, "collider", "small_excess", "O(10) excess pairs (ATLAS)",
    3.9, 2.1, "certain", "new", "statistical", 0.7, "more_data_same", "yes", "certain",
    "ATLAS 3.9 local / 2.1 global, CMS 2.6 local; gone with 2016 data (ICHEP Aug 2016)")
add("B02", "Fermi-LAT 130 GeV gamma-ray line", 2012, "astro", "line", "~50 excess photons",
    4.6, 3.2, "likely", "new", "statistical", 1.5, "more_data_same", "no", "certain",
    "Weniger 2012 on public data; Fermi-LAT 2013: weaker, similar feature in Earth-limb control; absent in Pass 8 (2015); partly instrumental")
add("B03", "3.5 keV X-ray line (stacked clusters, Andromeda, Perseus)", 2014, "astro", "line", "few-% of continuum in stacked spectra",
    4.4, NA, "likely", "new", "unresolved", NA, "none", "no", "certain",
    "Bulbul et al. 4-5 sigma, Boyarsky et al. 4.4 sigma; Hitomi 2017 Perseus null, Dessert et al. 2020 blank-sky null, disputed; censored; above window")
add("B04", "BICEP2 B-mode polarisation (r = 0.20)", 2014, "cosmo", "small_excess", "map-level excess",
    7.0, NA, "likely", "new", "artefact", 0.9, "independent_expt", "yes", "certain",
    "Galactic dust; Planck 353 GHz Sept 2014, joint BICEP2/Keck/Planck Jan 2015; above window")
add("B05", "OPERA superluminal neutrinos", 2011, "neutrino", "precision", "~16000 events, 60 ns early",
    6.0, NA, "certain", "new", "artefact", 0.7, "calibration_hardware", "yes", "certain",
    "loose optical-fibre connector and clock oscillator; ICARUS independent null 2012; above window")
add("B06", "LEP Higgs hint at 115 GeV", 2000, "collider", "few_event", "~4 ALEPH candidate events",
    2.9, NA, "likely", "expected", "statistical", 3, "reanalysis", "no", "likely",
    "LEP combination Nov 2000 ~2.9 sigma; final 2003 combination 1.7 sigma; Higgs found at 125 GeV in 2012")
add("B07", "ATLAS/CMS Higgs hints Dec 2011", 2011, "collider", "small_excess", "few tens of gamma-gamma / 4l events",
    3.6, 2.3, "certain", "expected", "real", 0.6, "more_data_same", "yes", "certain",
    "ATLAS 3.6 local / 2.3 global at 126 GeV, CMS 2.6 local; discovery July 2012")
add("B08", "CDF top-quark evidence 1994", 1994, "collider", "small_excess", "12 events, 5.7 expected",
    2.8, NA, "likely", "expected", "real", 0.9, "more_data_same", "yes", "likely",
    "discovery by CDF and D0 in Feb 1995")
add("B09", "IceCube two PeV cascades (2012 data)", 2013, "neutrino", "few_event", "2",
    2.8, NA, "likely", "expected", "real", 0.7, "more_data_same", "no", "certain",
    "2.8 sigma above atmospheric expectation; 28 events / 4.1 sigma later in 2013; 5.7 sigma 2014")
add("B10", "IceCube Glashow-resonance candidate (6 PeV)", 2021, "neutrino", "few_event", "1",
    2.3, NA, "likely", "expected", "unresolved", NA, "none", "no", "likely",
    "single event detected 2016, published 2021; consistent with SM expectation; a single event cannot be confirmed statistically; censored; below window")
add("B11", "KM3NeT KM3-230213A (~220 PeV muon)", 2025, "neutrino", "few_event", "1",
    2.7, NA, "likely", "expected", "unresolved", NA, "none", "yes", "certain",
    "tension of ~2.5-3 sigma with IceCube/Auger non-observation of a corresponding flux; censored")
add("B12", "ANITA anomalous upgoing air-shower events", 2016, "astro", "few_event", "2 (flights 2006, 2014)",
    NA, NA, "unknown", "new", "unresolved", NA, "none", "no", "certain",
    "steady-source interpretations constrained by IceCube 2020; reflection/artefact hypotheses; PUEO to test; censored")
add("B13", "MiniBooNE low-energy excess", 2007, "neutrino", "small_excess", "~130 excess (2009) -> ~460 (2018)",
    3.0, NA, "likely", "new", "unresolved", NA, "none", "no", "certain",
    "~3 sigma by 2009, 4.8 sigma 2018 (6 sigma with LSND); MicroBooNE 2021 saw no electron-like excess; photon-like channel open; censored")
add("B14", "LSND anti-nu_e appearance", 1996, "neutrino", "small_excess", "~88 excess events",
    3.8, NA, "likely", "new", "unresolved", NA, "none", "no", "certain",
    "KARMEN excluded part of the region; sterile-neutrino fits in tension; censored")
add("B15", "EDGES 21-cm absorption trough", 2018, "cosmo", "small_excess", "0.5 K trough (twice the standard maximum)",
    3.8, NA, "certain", "new", "contradicted", 4, "independent_expt", "yes", "certain",
    "SARAS 3 (2022) rejects the EDGES profile at 95.3 %; leading hypothesis ground-plane/systematics")
add("B16", "LHCb R_K lepton-universality anomaly", 2021, "collider", "precision", "hundreds of B+ -> K+ l+ l-",
    3.1, NA, "certain", "new", "artefact", 1.7, "reanalysis", "no", "certain",
    "3.1 sigma March 2021 (anomalies since 2014); Dec 2022 reanalysis: misidentified hadronic backgrounds, R_K SM-compatible")
add("B17", "Muon g-2 (BNL final -> FNAL)", 2004, "precision", "precision", "precision measurement",
    2.7, NA, "likely", "new", "theory_revision", 21, "theory", "yes", "likely",
    "2.7 sigma (2004), 3.3-3.7 with later theory, 4.2 (2021), 5.0 (2023) vs 2020 white paper; lattice HVP (BMW 2020) and the 2025 theory update remove most of the discrepancy (recalled: likely)")
add("B18", "CDF W-boson mass (7 sigma)", 2022, "precision", "precision", "precision measurement",
    7.0, NA, "certain", "new", "contradicted", 1.5, "independent_expt", "yes", "certain",
    "ATLAS 2023 and CMS 2024 SM-consistent; cause unidentified; above window")
add("B19", "ATOMKI X17 (8Be, 4He angular correlations)", 2016, "nuclear", "small_excess", "angular-correlation bump",
    6.8, NA, "certain", "new", "unresolved", NA, "none", "yes", "certain",
    "MEG II 2024 finds no signal (recalled: likely); several other null/partial tests; censored; above window")
add("B20", "Reactor antineutrino anomaly", 2011, "neutrino", "precision", "~6 % flux deficit",
    3.0, NA, "likely", "new", "theory_revision", 7, "theory", "yes", "certain",
    "Daya Bay fuel-evolution 2017, Kurchatov 235U/238U ratio 2021: 235U flux prediction was off; largely a flux-model error")
add("B21", "Gallium anomaly (GALLEX/SAGE -> BEST)", 2010, "neutrino", "precision", "source-calibration deficit",
    2.7, NA, "likely", "new", "unresolved", NA, "none", "no", "certain",
    "~2.5-3 sigma 2010; BEST 2022 ~4 sigma; censored")
add("B22", "Heidelberg-Moscow 0nubb claim", 2001, "nuclear", "small_excess", "~15 counts in the Q-value peak",
    3.1, NA, "likely", "new", "contradicted", 12, "independent_expt", "no", "certain",
    "2.2-3.1 sigma (2001), 4.2 (2004), 6 (2006); GERDA 2013 excluded the claim; KamLAND-Zen/EXO-200 likewise")
add("B23", "DAMPE 1.4 TeV electron-positron peak", 2017, "astro", "small_excess", "few tens of events",
    NA, NA, "unknown", "new", "unresolved", NA, "none", "no", "uncertain",
    "significance and status recalled only vaguely; EXCLUDED from primary counts")
add("B24", "CDF W+jj 145 GeV bump", 2011, "collider", "small_excess", "~250 excess events",
    3.2, NA, "likely", "new", "artefact", 1.2, "independent_expt", "yes", "likely",
    "4.1 sigma with more data; D0 null; CDF 2012 traced to jet-energy-scale / background modelling")
add("B25", "Tevatron top forward-backward asymmetry", 2011, "collider", "precision", "precision measurement",
    3.4, NA, "likely", "new", "theory_revision", 4, "theory", "yes", "likely",
    "CDF high-mass A_FB; NNLO QCD (2015) and D0/more data closed the gap")
add("B26", "HERA high-Q2 events (H1/ZEUS 1997)", 1997, "collider", "small_excess", "O(10) events vs ~5 expected",
    NA, NA, "unknown", "new", "statistical", 2, "more_data_same", "yes", "likely",
    "leptoquark interpretation; excess did not grow with 1998-2000 data")
add("B27", "ALEPH four-jet events at 105 GeV", 1996, "collider", "small_excess", "~16 excess 4-jet events",
    NA, NA, "unknown", "new", "statistical", 1, "independent_expt", "yes", "likely",
    "not seen by DELPHI/L3/OPAL; faded in 1997")
add("B28", "Pentaquark Theta+(1540)", 2003, "nuclear", "small_excess", "LEPS ~19 events",
    4.6, NA, "likely", "new", "statistical", 3, "independent_expt", "yes", "likely",
    "several 4-5 sigma claims 2003-2004; CLAS high-statistics 2006 null; above window")
add("B29", "AGASA super-GZK events", 1998, "astro", "small_excess", "11 events above 1e20 eV",
    NA, NA, "unknown", "new", "contradicted", 6, "independent_expt", "yes", "likely",
    "HiRes (2004-2008) and Auger (2008) observe the GZK suppression; AGASA energy scale implicated")
add("B30", "CMS 95 GeV diphoton excess", 2018, "collider", "small_excess", "small excess",
    2.9, NA, "likely", "new", "unresolved", NA, "none", "yes", "likely",
    "2.8 sigma (2018), 2.9 sigma local (2023); ATLAS 1.7 sigma; censored")
add("B31", "KOTO K_L -> pi0 nu nubar candidates", 2019, "collider", "few_event", "4 (0.05 expected)",
    NA, NA, "unknown", "new", "background", 1.5, "reanalysis", "no", "likely",
    "2019 conference result; 2020-2021 reanalysis: charged-kaon and halo-neutron backgrounds; upper limit published")
add("B32", "R_D / R_D* (BaBar 2012 onward)", 2012, "collider", "precision", "precision ratio",
    3.4, NA, "likely", "new", "unresolved", NA, "none", "yes", "certain",
    "BaBar 3.4 sigma 2012; world average ~3 sigma 2023; censored")
add("B33", "AMS-02 antihelium candidates", 2016, "astro", "few_event", "~8 candidates",
    NA, NA, "unknown", "new", "unresolved", NA, "none", "no", "likely",
    "conference presentations only; no refereed paper to our knowledge; censored")
add("B34", "D0 like-sign dimuon asymmetry", 2010, "collider", "precision", "precision asymmetry",
    3.2, NA, "likely", "new", "contradicted", 4, "independent_expt", "yes", "likely",
    "3.2 sigma (2010), 3.9 (2011); LHCb a_sl measurements SM-consistent")

df = pd.DataFrame(C)
df["few_event"] = df["kind"].eq("few_event")
df["resolved"] = df["outcome"].ne("unresolved")
df["real"] = df["outcome"].eq("real")
df["in_window"] = df["sig_local"].between(2.5, 4.0)
df["window_flag"] = np.where(df["sig_local"].isna(), "unknown",
                     np.where(df["in_window"], "in", np.where(df["sig_local"] > 4, "above", "below")))
df["t_obs"] = np.where(df["resolved"], df["t_res_yr"], NOW - df["year"])  # KM time
df["primary"] = df["item_rel"].isin(["certain", "likely"])
df.to_csv(f"{OUT}/anomalies.csv", index=False)

# --------------------------------------------------------------------------------------
# 2. Beta posteriors for P(real | anomaly)
# --------------------------------------------------------------------------------------
def beta_summary(k, n, a=0.5, b=0.5):
    post = stats.beta(a + k, b + n - k)
    return dict(k=int(k), n=int(n), mean=post.mean(), median=post.median(),
                lo68=post.ppf(0.16), hi68=post.ppf(0.84), lo95=post.ppf(0.025), hi95=post.ppf(0.975),
                up95_onesided=post.ppf(0.95), post=post)

P = df[df.primary]
strata = {
    "DD, new physics": P[(P.domain == "DD") & (P.prior_type == "new")],
    "DD, new physics, in window": P[(P.domain == "DD") & (P.prior_type == "new") & P.in_window],
    "broader, new physics": P[(P.domain != "DD") & (P.prior_type == "new")],
    "broader, new physics, in window": P[(P.domain != "DD") & (P.prior_type == "new") & P.in_window],
    "all, new physics": P[P.prior_type == "new"],
    "all, new physics, in window": P[(P.prior_type == "new") & P.in_window],
    "all, new physics, few-event": P[(P.prior_type == "new") & P.few_event],
    "all, expected process": P[P.prior_type == "expected"],
    "all, expected process, in window": P[(P.prior_type == "expected") & P.in_window],
    "all items": P,
    "all items, in window": P[P.in_window],
}
rows = []
POST = {}
for name, S in strata.items():
    for variant in ["resolved only", "unresolved as null", "unresolved as real"]:
        if variant == "resolved only":
            k, n = S.real.sum(), S.resolved.sum()
        elif variant == "unresolved as null":
            k, n = S.real.sum(), len(S)
        else:
            k, n = S.real.sum() + (~S.resolved).sum(), len(S)
        for prior, (a, b) in {"Jeffreys": (0.5, 0.5), "uniform": (1, 1)}.items():
            s = beta_summary(k, n, a, b)
            POST[(name, variant, prior)] = s.pop("post")
            rows.append(dict(stratum=name, variant=variant, prior=prior, n_items=len(S),
                             n_unresolved=int((~S.resolved).sum()), **s))
beta_tab = pd.DataFrame(rows)
beta_tab.to_csv(f"{OUT}/P083_beta_table.csv", index=False, float_format="%.4f")

# --------------------------------------------------------------------------------------
# 3. Selection / recall corrections
#   (a) recall bias: observed odds = true odds * f_real/f_null  ->  true odds = obs odds * f_null/f_real
#   (b) dilution by never-written-up ~3 sigma excursions (applies to the class "any local excursion",
#       NOT to the class "anomaly that reached a dedicated paper", which is the LZ event's class)
# --------------------------------------------------------------------------------------
def odds(p): return p / (1 - p)
def inv_odds(o): return o / (1 + o)
sel_rows = []
for name in ["DD, new physics", "all, new physics", "all, new physics, in window"]:
    post = POST[(name, "resolved only", "Jeffreys")]
    r_obs = post.mean()
    for ratio in [1.0, 0.5, 0.33]:
        r_a = inv_odds(odds(r_obs) * ratio)
        for dil in [0, 2, 5]:
            sel_rows.append(dict(stratum=name, r_obs=r_obs, f_null_over_f_real=ratio, unrecalled_per_recalled=dil,
                                 r_corrected=r_a / (1 + dil)))
sel = pd.DataFrame(sel_rows)
sel.to_csv(f"{OUT}/P083_selection.csv", index=False, float_format="%.4f")

# Observed rate of published >= 2.5 sigma-ish DD anomalies vs a pure-fluctuation expectation
dd_new = P[(P.domain == "DD") & (P.prior_type == "new") & (P.year >= 2008)]
n_dd_claims = len(dd_new)
rate_obs = n_dd_claims / (NOW - 2008)
# pure-statistics expectation: N_results/yr DD results, each with N_trials effective independent bins,
# P(>= 3 sigma local somewhere) = 1-(1-p3)^N_trials.  Assumptions flagged 'uncertain' in the JSON.
p3 = stats.norm.sf(3.0)
expect = {f"{nres} results/yr x {ntr} trials": nres * (1 - (1 - p3) ** ntr)
          for nres in [10, 20] for ntr in [10, 30]}

# --------------------------------------------------------------------------------------
# 4. Kaplan-Meier survival (time from claim to resolution; unresolved censored at 2026.7)
# --------------------------------------------------------------------------------------
def km(times, events):
    times = np.asarray(times, float); events = np.asarray(events, bool)
    order = np.argsort(times); times, events = times[order], events[order]
    t_grid, S = [0.0], [1.0]
    s = 1.0
    for t in np.unique(times[events]):
        n_at_risk = np.sum(times >= t)
        d = np.sum((times == t) & events)
        s *= 1 - d / n_at_risk
        t_grid.append(t); S.append(s)
    return np.array(t_grid), np.array(S)

def S_at(t_grid, S, t):
    return S[np.searchsorted(t_grid, t, side="right") - 1]

def km_median(t_grid, S):
    idx = np.where(S <= 0.5)[0]
    return t_grid[idx[0]] if len(idx) else np.inf

surv_sets = {
    "all new physics": P[P.prior_type == "new"],
    "new physics, independent comparable test available": P[(P.prior_type == "new") & (P.indep_comparable == "yes")],
    "new physics, no comparable independent test": P[(P.prior_type == "new") & (P.indep_comparable == "no")],
    "direct detection (all)": P[P.domain == "DD"],
    "few-event (n <= 5), all": P[P.few_event],
    "expected process": P[P.prior_type == "expected"],
    "new physics, in window": P[(P.prior_type == "new") & P.in_window],
}
surv_rows, curves = [], {}
for name, S in surv_sets.items():
    tg, sv = km(S.t_obs.values, S.resolved.values)
    curves[name] = (tg, sv)
    surv_rows.append(dict(stratum=name, n=len(S), n_resolved=int(S.resolved.sum()),
                          F1=1 - S_at(tg, sv, 1.0), F2=1 - S_at(tg, sv, 2.0), F3=1 - S_at(tg, sv, 3.0),
                          F5=1 - S_at(tg, sv, 5.0), F10=1 - S_at(tg, sv, 10.0), median_yr=km_median(tg, sv)))
surv = pd.DataFrame(surv_rows)
surv.to_csv(f"{OUT}/P083_survival.csv", index=False, float_format="%.3f")
# raw curves
with open(f"{OUT}/P083_survival_curves.json", "w") as f:
    json.dump({k: dict(t=list(map(float, v[0])), S=list(map(float, v[1]))) for k, v in curves.items()}, f, indent=1)

# routes
res_new = P[(P.prior_type == "new") & P.resolved]
routes = res_new.route.value_counts()
routes_few = P[P.few_event & P.resolved].route.value_counts()
routes_dd = P[(P.domain == "DD") & P.resolved].route.value_counts()
route_tab = pd.DataFrame({"new physics (resolved)": routes, "few-event (resolved)": routes_few,
                          "DD (resolved)": routes_dd}).fillna(0).astype(int)
route_tab["share_new_physics"] = route_tab["new physics (resolved)"] / route_tab["new physics (resolved)"].sum()
route_tab.to_csv(f"{OUT}/P083_routes.csv", float_format="%.3f")
outcome_tab = pd.crosstab(P.prior_type, P.outcome)
outcome_tab.to_csv(f"{OUT}/P083_outcomes_by_prior.csv")
median_t_by_route = res_new.groupby("route").t_res_yr.agg(["count", "median", "mean"])

# --------------------------------------------------------------------------------------
# 5. Prior for the LZ event and posterior via P061 Eq. 3:
#    P_ref = r B' / (r B' + 1 - r),  B' = (B_DM/f_FP)/B_typ,  B_typ = 14.60 (SBB bound at LZ's global p)
#    r ~ Beta posterior of the stratum.  Naive version: B_DM applied to r directly (double-counts,
#    because the reference-class anomalies also had B ~ 10-100 against their modelled backgrounds).
# --------------------------------------------------------------------------------------
B_TYP = 14.60           # P061 details 4.3 (SBB bound at p_global = 4.7e-3)
BDM = {"P027 uniform spectra": 16.4, "P027 class priors": 28.7}
FFP = {"f_FP=1": 1.0, "f_FP=3": 3.0, "f_FP=10": 10.0}
rng = np.random.default_rng(20260916)
lz_rows = []
for stratum in ["DD, new physics", "all, new physics", "all, new physics, in window", "all, new physics, few-event"]:
    post = POST[(stratum, "resolved only", "Jeffreys")]
    r = post.rvs(200000, random_state=rng)
    for bname, B in BDM.items():
        for fname, f in FFP.items():
            Bp = (B / f) / B_TYP
            p = r * Bp / (r * Bp + 1 - r)
            pn = r * B / (r * B + 1 - r)
            lz_rows.append(dict(stratum=stratum, B_DM=bname, f_FP=fname, Bprime=Bp,
                                r_mean=r.mean(), P_median=np.median(p), P_lo68=np.quantile(p, 0.16),
                                P_hi68=np.quantile(p, 0.84), P_hi95=np.quantile(p, 0.975),
                                P_naive_median=np.median(pn), P_naive_hi68=np.quantile(pn, 0.84)))
lzp = pd.DataFrame(lz_rows)
lzp.to_csv(f"{OUT}/P083_LZ_posterior.csv", index=False, float_format="%.4f")

# Implied pi_DM for P061's structural model, from P061 details 4.3 table (log-log interpolation)
p061_r = np.array([0.02, 0.05, 0.10, 0.20])
p061_pi_lo = np.array([0.0060, 0.0154, 0.0323, 0.0721])   # L_rest = 0.01
p061_pi_hi = np.array([0.028, 0.072, 0.152, 0.338])       # L_rest = 0.1
def pi_implied(r, table):
    # log-log linear fit to P061's four-point table (slope ~1.08), extrapolated beyond its edges
    slope, icpt = np.polyfit(np.log(p061_r), np.log(table), 1)
    return np.exp(icpt + slope * np.log(r))
def r_implied(pi, table):
    slope, icpt = np.polyfit(np.log(p061_r), np.log(table), 1)
    return np.exp((np.log(pi) - icpt) / slope)
pi_fit_check = {f"r={r}": float(pi_implied(r, p061_pi_lo) / t) for r, t in zip(p061_r, p061_pi_lo)}
post_dd = POST[("DD, new physics", "resolved only", "Jeffreys")]
post_all = POST[("all, new physics", "resolved only", "Jeffreys")]
r_dd = post_dd.rvs(200000, random_state=rng)
pi_dd_lo = pi_implied(r_dd, p061_pi_lo); pi_dd_hi = pi_implied(r_dd, p061_pi_hi)
p061_logu_median = np.sqrt(1e-3 * 0.3)
# P061: P(DM) > 0.5 needs pi_DM >= 0.089 (optimistic corner) -> r >= ?
r_for_pi089 = float(r_implied(0.089, p061_pi_lo))
r_equiv_p061 = float(r_implied(p061_logu_median, p061_pi_lo))   # base rate equivalent to P061's prior median
prob_r_exceeds = {"DD": float(post_dd.sf(r_for_pi089)), "all": float(post_all.sf(r_for_pi089))}
# fraction of P061's log-uniform pi_DM range [1e-3,0.3] lying above the DD 95% upper bound (translated)
pi_up95 = pi_implied(post_dd.ppf(0.95), p061_pi_lo)
frac_logu_above = (np.log(0.3) - np.log(pi_up95)) / (np.log(0.3) - np.log(1e-3))

# --------------------------------------------------------------------------------------
# 6. LZ timeline: the event's covariates place it in "new physics, independent comparable test available"
# --------------------------------------------------------------------------------------
tg, sv = curves["new physics, independent comparable test available"]
lz_time = {f"P_resolved_within_{t}yr": float(1 - S_at(tg, sv, t)) for t in [1, 2, 3, 5]}
lz_time["median_yr"] = float(km_median(tg, sv))
lz_time["claim_date"] = "2026-09-02"
# route split for that stratum
stratum_routes = P[(P.prior_type == "new") & (P.indep_comparable == "yes") & P.resolved].route.value_counts(normalize=True)
# few-event items: how were they resolved
few = P[P.few_event]

# --------------------------------------------------------------------------------------
# 7. Figures
# --------------------------------------------------------------------------------------
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
COL = {"real": "#2a78d6", "not real": "#eb6834", "unresolved": "#1baf7a"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})
def outcome_group(o):
    return "real" if o == "real" else ("unresolved" if o == "unresolved" else "not real")

# Fig 1: timeline
d1 = pd.concat([df[df.domain == "DD"].sort_values("year"), df[df.domain != "DD"].sort_values("year")])
fig, ax = plt.subplots(figsize=(8.5, 11.5))
y = np.arange(len(d1))[::-1]
for yi, (_, r) in zip(y, d1.iterrows()):
    g = outcome_group(r.outcome); c = COL[g]
    t_end = r.year + r.t_res_yr if r.resolved else NOW
    ax.plot([r.year, t_end], [yi, yi], color=c, lw=2, solid_capstyle="round", alpha=0.9)
    ax.scatter([r.year], [yi], s=34 if r.few_event else 18, marker="D" if r.few_event else "o",
               color=c, edgecolor=SURF, linewidth=0.8, zorder=3)
    if not r.resolved:
        ax.annotate("", xy=(NOW + 0.8, yi), xytext=(NOW, yi), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.2))
    nm = r["name"]
    lab = nm if len(nm) <= 52 else nm[:50] + "..."
    if r.item_rel == "uncertain": lab += "  [uncertain]"
    ax.text(1993.2, yi, lab, ha="right", va="center", fontsize=7.2, color=INK if r.primary else INK2)
    sig = "" if np.isnan(r.sig_local) else f"{r.sig_local:.1f}σ"
    ax.text(NOW + 1.2, yi, sig, ha="left", va="center", fontsize=7, color=INK2)
ysep = y[(d1.domain == "DD").sum() - 1] - 0.5
ax.axhline(ysep, color=INK2, lw=0.6, ls=":")
ax.text(1994, ysep - 0.15, "direct detection above the line;  other rare-event searches below",
        fontsize=7.5, color=INK2, va="top")
ax.axvline(2026.67, color=INK2, lw=0.6, ls="--"); ax.text(2026.4, ysep + 1.2, "LZ event\npaper", ha="right", fontsize=7, color=INK2)
ax.set_xlim(1993, 2029); ax.set_ylim(-1, len(d1)); ax.set_yticks([])
ax.set_xlabel("year of claim  →  year of resolution (arrow = still unresolved, Sept 2026)")
for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
ax.grid(axis="x", color="#e6e5e1", lw=0.6); ax.set_axisbelow(True)
from matplotlib.lines import Line2D
h = [Line2D([], [], color=COL[k], lw=3, label=k) for k in COL] + \
    [Line2D([], [], marker="D", color=INK2, lw=0, label="few-event (≤ 5 events)"),
     Line2D([], [], marker="o", color=INK2, lw=0, label="excess / modulation / line / precision")]
ax.legend(handles=h, loc="upper left", bbox_to_anchor=(0.01, 0.985), fontsize=7.5, frameon=False, ncol=1)
ax.set_title("Recalled reference class of anomalies and what they did next\n(P083; compiled from memory, incomplete; σ = reported local significance)", fontsize=9, loc="left")
plt.tight_layout(); plt.savefig(f"{FIG}/P083_fig1_timeline.png", dpi=170); plt.close()

# Fig 2: KM survival
fig, ax = plt.subplots(figsize=(6.4, 4.2))
sel_curves = [("new physics, independent comparable test available", "#2a78d6", "new physics, comparable independent test existed"),
              ("new physics, no comparable independent test", "#eb6834", "new physics, no comparable independent test"),
              ("direct detection (all)", "#1baf7a", "direct detection, all"),
              ("expected process", "#eda100", "expected (predicted) process")]
for key, c, lab in sel_curves:
    tg, sv = curves[key]
    tt = np.append(tg, 30); ss = np.append(sv, sv[-1])
    ax.step(tt, ss, where="post", color=c, lw=2, label=f"{lab} (n = {len(surv_sets[key])})")
ax.axvline(2, color=INK2, lw=0.6, ls=":"); ax.axvline(5, color=INK2, lw=0.6, ls=":")
ax.set_xscale("log"); ax.set_xlim(0.4, 30); ax.set_ylim(0, 1.02)
ax.set_xlabel("years since the claim"); ax.set_ylabel("fraction still unresolved (Kaplan–Meier)")
ax.legend(fontsize=7.5, frameon=False, loc="upper right")
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.grid(color="#e6e5e1", lw=0.6); ax.set_axisbelow(True)
ax.set_title("Time to resolution, recalled reference class (censored at Sept 2026)", fontsize=9.5, loc="left")
plt.tight_layout(); plt.savefig(f"{FIG}/P083_fig2_survival.png", dpi=170); plt.close()

# Fig 3: Beta posteriors
fig, ax = plt.subplots(figsize=(6.4, 4.0))
x = np.geomspace(1e-3, 1, 2000)
for key, c, lab in [("DD, new physics", "#2a78d6", "direct detection, new physics"),
                    ("all, new physics", "#eb6834", "all new-physics anomalies"),
                    ("all, new physics, in window", "#1baf7a", "new physics, 2.5–4σ local"),
                    ("all, expected process", "#eda100", "expected process")]:
    post = POST[(key, "resolved only", "Jeffreys")]
    kk = beta_tab[(beta_tab.stratum == key) & (beta_tab.variant == "resolved only") & (beta_tab.prior == "Jeffreys")].iloc[0]
    ax.plot(x, post.cdf(x), color=c, lw=2, label=f"{lab}: {kk.k} real of {kk.n} resolved")
ax.axhline(0.5, color=INK2, lw=0.6, ls=":"); ax.axhline(0.95, color=INK2, lw=0.6, ls=":")
ax.text(1.05e-3, 0.51, "median", fontsize=7, color=INK2); ax.text(1.05e-3, 0.96, "95 %", fontsize=7, color=INK2)
ax.axvline(r_equiv_p061, color=INK2, lw=0.8, ls="--")
ax.text(r_equiv_p061 * 1.06, 0.04, f"r = {r_equiv_p061:.3f}: equivalent of\nP061's log-uniform π_DM median", fontsize=7, color=INK2)
ax.set_xscale("log"); ax.set_xlim(1e-3, 1); ax.set_ylim(0, 1.02)
ax.set_xlabel("base rate r = P(real | anomaly reached a paper)"); ax.set_ylabel("posterior CDF (Jeffreys prior)")
ax.legend(fontsize=7.5, frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.72)); ax.grid(color="#e6e5e1", lw=0.6); ax.set_axisbelow(True)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.set_title("Base rate of 'real' outcomes by stratum (resolved items only)", fontsize=9.5, loc="left")
plt.tight_layout(); plt.savefig(f"{FIG}/P083_fig3_beta.png", dpi=170); plt.close()

# --------------------------------------------------------------------------------------
# 8. Console report + results JSON
# --------------------------------------------------------------------------------------
def bt(stratum, variant="resolved only", prior="Jeffreys"):
    r = beta_tab[(beta_tab.stratum == stratum) & (beta_tab.variant == variant) & (beta_tab.prior == prior)].iloc[0]
    return {k: (float(r[k]) if isinstance(r[k], (float, np.floating)) else int(r[k])) for k in
            ["k", "n", "n_items", "n_unresolved", "mean", "median", "lo68", "hi68", "lo95", "hi95", "up95_onesided"]}

results = dict(
    compilation=dict(n_total=len(df), n_primary=int(df.primary.sum()), n_uncertain=int((~df.primary).sum()),
                     n_DD=int((df.domain == "DD").sum()), n_few_event=int(df.few_event.sum()),
                     n_in_window=int(df.in_window.sum()), n_unresolved=int((~df.resolved).sum()),
                     n_sig_unknown=int(df.sig_local.isna().sum()),
                     outcomes=df[df.primary].outcome.value_counts().to_dict(),
                     outcomes_by_prior=outcome_tab.to_dict()),
    beta={k: bt(k) for k in strata},
    beta_unresolved_as_null={k: bt(k, "unresolved as null") for k in ["DD, new physics", "all, new physics", "all, new physics, in window"]},
    beta_unresolved_as_real={k: bt(k, "unresolved as real") for k in ["DD, new physics", "all, new physics", "all, new physics, in window"]},
    beta_uniform_prior={k: bt(k, "resolved only", "uniform") for k in ["DD, new physics", "all, new physics", "all, new physics, in window"]},
    selection=sel.to_dict(orient="records"),
    dd_claim_rate=dict(n_DD_new_physics_claims_2008_2026=int(n_dd_claims), rate_per_yr=float(rate_obs),
                       pure_fluctuation_expectation_per_yr=expect),
    survival=surv.to_dict(orient="records"),
    routes=route_tab.reset_index().rename(columns={"index": "route"}).to_dict(orient="records"),
    median_t_by_route=median_t_by_route.reset_index().to_dict(orient="records"),
    few_event_items=few[["id", "name", "n_events", "outcome", "t_res_yr", "route"]].to_dict(orient="records"),
    lz_posterior=lzp.to_dict(orient="records"),
    lz_prior_translation=dict(B_typ=B_TYP,
                              pi_DM_implied_DD_Lrest001=dict(median=float(np.median(pi_dd_lo)), lo68=float(np.quantile(pi_dd_lo, 0.16)), hi68=float(np.quantile(pi_dd_lo, 0.84)), up95=float(np.quantile(pi_dd_lo, 0.95))),
                              pi_DM_implied_DD_Lrest01=dict(median=float(np.median(pi_dd_hi)), lo68=float(np.quantile(pi_dd_hi, 0.16)), hi68=float(np.quantile(pi_dd_hi, 0.84)), up95=float(np.quantile(pi_dd_hi, 0.95))),
                              P061_loguniform_pi_median=float(p061_logu_median), loglog_fit_check_ratio=pi_fit_check,
                              r_needed_for_pi_0p089=float(r_for_pi089), prob_r_exceeds_that=prob_r_exceeds,
                              r_equivalent_to_P061_median=r_equiv_p061,
                              pi_up95_translated=float(pi_up95), frac_P061_logu_range_above_up95=float(frac_logu_above)),
    lz_timeline=dict(**lz_time, route_shares_in_stratum=stratum_routes.to_dict()),
)
with open(f"{OUT}/P083_results.json", "w") as f:
    json.dump(results, f, indent=1, default=float)

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
print("=== compilation ===");  print(results["compilation"])
print("\n=== outcomes by prior type (primary) ===\n", outcome_tab)
print("\n=== Beta posteriors (Jeffreys, resolved only) ===")
print(beta_tab[(beta_tab.variant == "resolved only") & (beta_tab.prior == "Jeffreys")][["stratum", "k", "n", "n_unresolved", "mean", "lo68", "hi68", "hi95", "up95_onesided"]].to_string(index=False))
print("\n=== variants (DD & all new physics) ===")
print(beta_tab[beta_tab.stratum.isin(["DD, new physics", "all, new physics", "all, new physics, in window"])][["stratum", "variant", "prior", "k", "n", "mean", "lo68", "hi68", "up95_onesided"]].to_string(index=False))
print("\n=== selection corrections ===\n", sel.to_string(index=False))
print("\n=== DD claim rate ===", results["dd_claim_rate"])
print("\n=== survival ===\n", surv.to_string(index=False))
print("\n=== routes ===\n", route_tab); print(median_t_by_route)
print("\n=== few-event items ===\n", few[["id", "name", "n_events", "outcome", "t_res_yr", "route", "indep_comparable"]].to_string(index=False))
print("\n=== LZ posterior (P061 Eq. 3) ===\n", lzp.to_string(index=False))
print("\n=== prior translation ===", json.dumps(results["lz_prior_translation"], indent=1))
print("\n=== LZ timeline ===", results["lz_timeline"])
