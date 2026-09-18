# P083 — Base rates: single-event and few-event anomalies in direct detection (2008–2026) and other rare-event searches, and what they did next

Research record. Simulated date 2026-09-16. Category BKG (meta/statistics), physics.data-an (cross-list hep-ex).
Author profile: history-and-sociology-of-physics group working with a statistics group.
Builds on the LZ paper (arXiv:2609.02823), dossier §4 (which asserts qualitatively that "single-event anomalies at this level
have historically resolved as backgrounds or artifacts"), P001 (three-hypothesis posterior), P027 (B_DM = 16.4–28.7), P061
(hyper-prior over pi_DM; reference-class equation and implied-pi_DM table), and P069/P050 (independent-xenon timelines).

**Everything in the compilation is recalled knowledge.** No database, web or literature access was used. The list is
incomplete by construction; every item carries a reliability flag (certain | likely | uncertain) for its existence and outcome,
and a separate flag for the quoted significance. Two items flagged "uncertain" (EDELWEISS-II five events; DAMPE 1.4 TeV peak)
are shown but excluded from all counts. Years and times to resolution are recalled to about ±0.5 yr.

## 1. Motivation and framework

P001 and P061 showed that P(DM | LZ event) is governed by the prior odds of DM against unmodelled backgrounds, not by the
modelled background; P061 found that pi_DM alone carries 56 % of the variance of the posterior and parametrised a
"reference-class" base rate r ∈ [0.01, 0.3] without adopting a value, leaving the historical compilation to this paper.
The reference-class (outside-view) argument: the LZ event is one member of a population of ~2.5–4σ local anomalies that
reached a dedicated paper; the empirical frequency with which such anomalies turned out to be real is a defensible prior
for this one, provided the event's covariates place it in the right stratum of the population.

Definitions used throughout:
- **Item**: a claimed excess, event set, line, modulation or precision deviation that was published (or, for two items,
  presented at conferences) and discussed as a possible new-physics or first-detection signal.
- **Domain**: DD (direct dark-matter detection) or other (collider, neutrino, astro, cosmology, nuclear, precision).
- **Kind**: few_event (≤ 5 events) | small_excess | modulation | line | precision.
- **Prior type**: `new` (new physics, or a process with no predicted rate) versus `expected` (a Standard-Model or
  astrophysical process that was predicted to appear at roughly that rate: Higgs, top, astrophysical neutrinos, 124Xe
  double electron capture, 8B CEνNS, cosmogenic neutrinos).
- **Outcome**: real | statistical (faded with more data, no cause identified) | background (identified physical
  background) | artefact (identified instrumental/analysis cause) | contradicted (independent null, cause never
  identified) | theory_revision (the prediction moved) | unresolved (still open in Sept 2026). For the base rate, all
  outcomes other than `real` and `unresolved` are "not real".
- **Route**: what resolved it — more_data_same | independent_expt | reanalysis | calibration_hardware | theory.
- **indep_comparable**: whether an independent experiment of comparable sensitivity existed at the time of the claim.
- **Window**: reported local significance 2.5–4.0σ (the LZ event is 3.4σ local, 2.6σ global). Items above, below or
  with unknown significance are kept for the timeline and route statistics but excluded from "in-window" counts.

## 2. The compilation (Table 1; `anomalies.csv`; Fig. 1)

Columns: id | item | year | kind | events | σ_local (rel.) | prior | outcome | t_res [yr] | route | indep. | item rel.

| id | item | year | kind | events | σ_local | prior | outcome | t_res | route | indep. | rel. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A01 | DAMA/LIBRA annual modulation | 2008 | modulation | ~1e4 (modulated) | 8.2 (likely) | new | contradicted (COSINE-100 2018, ANAIS-112 2019–25) | 10 | independent | no | certain |
| A02 | CoGeNT excess + 2.8σ modulation | 2010 | excess | hundreds | 2.8 (likely) | new | background (surface events; 2014 reanalyses) | 4 | reanalysis | yes | certain |
| A03 | CDMS-II Si three events | 2013 | few-event | 3 (0.7 exp.) | 3.0 (likely; p = 0.19 %) | new | statistical (LUX 2013, SuperCDMS 2014 exclude) | 0.5 | independent | yes | certain |
| A04 | CRESST-II Run-32 excess | 2011 | excess | 67 (~30 excess) | 4.2 (likely) | new | background (clamp α/Pb recoils; 2014 modules) | 3 | hardware | yes | certain |
| A05 | XENON1T ER excess | 2020 | excess | 285 vs 232 | 3.3 (certain) | new | contradicted (XENONnT 2022; tritium never proven) | 2 | independent (successor) | yes | certain |
| A06 | XENON100 ER-band modulation | 2015 | modulation | ER singles | 2.8 (likely) | new | statistical (4-yr data 2017: 1.9σ) | 2 | more data | yes | likely |
| A07 | EDELWEISS-II five events | 2011 | few-event | 5 (~3 exp.) | unknown | new | statistical | 1 | more data | yes | **uncertain — excluded** |
| A08 | CDMS-II Ge two events | 2009 | few-event | 2 (0.8 exp.) | 0.7 (p = 0.23) | new | statistical (XENON100 2010–11) | 1 | independent | yes | certain |
| A09 | Cryogenic low-energy excess (CRESST-III, EDELWEISS, SuperCDMS-CPD, NUCLEUS) | 2020 | excess | thousands | none quoted | new | unresolved (stress/relaxation artefact favoured) | — | — | yes | certain |
| A10 | XENON1T 124Xe 2νECEC | 2019 | excess | 126 | 4.4 (likely) | expected | real (XENONnT 2022, LZ 2024) | 3 | independent | yes | certain |
| A11 | 8B CEνNS: PandaX-4T 2.64σ, XENONnT 2.73σ | 2024 | excess | ~3.5 and ~11 | 2.7 (certain) | expected | unresolved (not yet 5σ) | — | — | yes | certain |
| B01 | ATLAS/CMS 750 GeV diphoton | 2015 | excess | O(10) pairs | 3.9 local / 2.1 global (certain) | new | statistical (2016 data) | 0.7 | more data | yes | certain |
| B02 | Fermi-LAT 130 GeV line | 2012 | line | ~50 photons | 4.6 / 3.2 (likely) | new | statistical/instrumental (Earth-limb feature; Pass 8) | 1.5 | more data + reanalysis | no | certain |
| B03 | 3.5 keV line | 2014 | line | few % of continuum | 4.4 (likely) | new | unresolved (Hitomi 2017, Dessert 2020 null; disputed) | — | — | no | certain |
| B04 | BICEP2 r = 0.20 | 2014 | excess | map-level | 7.0 (likely) | new | artefact (dust; Planck 353 GHz) | 0.9 | independent | yes | certain |
| B05 | OPERA superluminal ν | 2011 | precision | 1.6e4 events | 6.0 (certain) | new | artefact (fibre connector, oscillator) | 0.7 | hardware | yes | certain |
| B06 | LEP Higgs hint 115 GeV | 2000 | few-event | ~4 ALEPH candidates | 2.9 (likely) | expected | statistical (final LEP 1.7σ; Higgs at 125 GeV) | 3 | reanalysis | no | likely |
| B07 | ATLAS/CMS Higgs hints Dec 2011 | 2011 | excess | tens | 3.6 / 2.3 (certain) | expected | real (July 2012) | 0.6 | more data | yes | certain |
| B08 | CDF top evidence 1994 | 1994 | excess | 12 (5.7 exp.) | 2.8 (likely) | expected | real (1995) | 0.9 | more data | yes | likely |
| B09 | IceCube two PeV cascades | 2013 | few-event | 2 | 2.8 (likely) | expected | real (28 events 4.1σ; 5.7σ 2014) | 0.7 | more data | no | certain |
| B10 | IceCube Glashow candidate | 2021 | few-event | 1 | 2.3 (likely) | expected | unresolved (single event) | — | — | no | likely |
| B11 | KM3NeT KM3-230213A | 2025 | few-event | 1 | 2.7 tension (likely) | expected | unresolved | — | — | yes | certain |
| B12 | ANITA anomalous upgoing events | 2016 | few-event | 2 | unknown | new | unresolved (IceCube constraints; PUEO) | — | — | no | certain |
| B13 | MiniBooNE low-energy excess | 2007 | excess | ~130 → ~460 | 3.0 (2009; 4.8 in 2018) (likely) | new | unresolved (MicroBooNE 2021 no e-like excess) | — | — | no | certain |
| B14 | LSND | 1996 | excess | ~88 | 3.8 (likely) | new | unresolved | — | — | no | certain |
| B15 | EDGES 21-cm trough | 2018 | excess | 0.5 K profile | 3.8 (certain) | new | contradicted (SARAS 3 2022) | 4 | independent | yes | certain |
| B16 | LHCb R_K | 2021 | precision | hundreds | 3.1 (certain) | new | artefact (misID backgrounds; Dec 2022) | 1.7 | reanalysis | no | certain |
| B17 | Muon g−2 | 2004 | precision | — | 2.7 → 5.0 (likely) | new | theory revision (lattice HVP; 2025 update) (likely) | 21 | theory | yes | likely |
| B18 | CDF W mass | 2022 | precision | — | 7.0 (certain) | new | contradicted (ATLAS 2023, CMS 2024) | 1.5 | independent | yes | certain |
| B19 | ATOMKI X17 | 2016 | excess | angular bump | 6.8 (certain) | new | unresolved (MEG II 2024 null, likely) | — | — | yes | certain |
| B20 | Reactor antineutrino anomaly | 2011 | precision | 6 % deficit | 3.0 (likely) | new | theory revision (235U flux) | 7 | theory | yes | certain |
| B21 | Gallium anomaly → BEST | 2010 | precision | — | 2.7 → 4 (likely) | new | unresolved | — | — | no | certain |
| B22 | Heidelberg–Moscow 0νββ claim | 2001 | excess | ~15 counts | 3.1 (2001) → 4.2 → 6 (likely) | new | contradicted (GERDA 2013) | 12 | independent | no | certain |
| B23 | DAMPE 1.4 TeV peak | 2017 | excess | few tens | unknown | new | unresolved | — | — | no | **uncertain — excluded** |
| B24 | CDF W+jj 145 GeV | 2011 | excess | ~250 | 3.2 → 4.1 (likely) | new | artefact (JES/background modelling; D0 null) | 1.2 | independent | yes | likely |
| B25 | Tevatron top A_FB | 2011 | precision | — | 3.4 (likely) | new | theory revision (NNLO 2015) | 4 | theory | yes | likely |
| B26 | HERA high-Q² events | 1997 | excess | O(10) vs ~5 | unknown | new | statistical | 2 | more data | yes | likely |
| B27 | ALEPH 4-jet 105 GeV | 1996 | excess | ~16 | unknown | new | statistical (other LEP expts) | 1 | independent | yes | likely |
| B28 | Pentaquark Θ+ | 2003 | excess | ~19 (LEPS) | 4.6 (likely) | new | statistical (CLAS 2006) | 3 | independent | yes | likely |
| B29 | AGASA super-GZK | 1998 | excess | 11 events | unknown | new | contradicted (HiRes, Auger; energy scale) | 6 | independent | yes | likely |
| B30 | CMS 95 GeV diphoton | 2018 | excess | small | 2.9 (likely) | new | unresolved (ATLAS 1.7σ) | — | — | yes | likely |
| B31 | KOTO K_L→π0νν̄ candidates | 2019 | few-event | 4 (0.05 exp.) | unknown | new | background (K± and halo-neutron; 2020–21) | 1.5 | reanalysis | no | likely |
| B32 | R_D / R_D* | 2012 | precision | — | 3.4 (likely) | new | unresolved | — | — | yes | certain |
| B33 | AMS-02 antihelium candidates | 2016 | few-event | ~8 | unknown | new | unresolved (conference only) | — | — | no | likely |
| B34 | D0 like-sign dimuon asymmetry | 2010 | precision | — | 3.2 → 3.9 (likely) | new | contradicted (LHCb a_sl) | 4 | independent | yes | likely |

Items the assignment suggested but which we could **not** recall with enough confidence to list: a XENON1T "S2-only"
excess (we recall the 2019 ionisation-only limit, not an excess claim); PICO/COUPP excesses; LUX, PandaX or XENON1T
single high-energy events; EDELWEISS beyond the vague A07. They are absent rather than invented.

Counts (`P083_results.json` → `compilation`): 45 items, 43 primary (2 uncertain excluded), 11 DD, 10 few-event,
24 in the 2.5–4.0σ window, 9 with unknown significance, 14 unresolved (13 primary).
Outcomes among the 43 primary items: statistical 9, contradicted 7, artefact 4, background 3, theory revision 3, real 4, unresolved 13.

Outcomes by prior type (primary):

| prior type | real | statistical | background | artefact | contradicted | theory rev. | unresolved | total |
|---|---|---|---|---|---|---|---|---|
| new physics | **0** | 8 | 3 | 4 | 7 | 3 | 10 | 35 |
| expected process | **4** | 1 | 0 | 0 | 0 | 0 | 3 | 8 |

## 3. Base rates: Beta posteriors (`P083_beta_table.csv`; Fig. 3)

For k real outcomes among n resolved items the posterior for r = P(real) under a Beta(a, b) prior is Beta(a + k, b + n − k).
Primary prior: Jeffreys (a = b = ½); variant: uniform (a = b = 1). Unresolved items are excluded in the primary
estimate, and counted as null or as real to bracket. Intervals are central 68 % / 95 % and the one-sided 95 % upper bound.

| stratum | k/n resolved (unresolved) | mean | 68 % | 95 % one-sided upper | uniform-prior mean |
|---|---|---|---|---|---|
| DD, new physics | 0/7 (1) | 0.062 | 0.003–0.127 | 0.232 | 0.111 |
| DD, new physics, in window | 0/4 (0) | 0.100 | 0.005–0.207 | 0.362 | — |
| broader (non-DD), new physics | 0/18 (9) | 0.026 | 0.001–0.053 | 0.100 | — |
| broader, new physics, in window | 0/9 (5) | 0.050 | 0.002–0.101 | 0.187 | — |
| all new physics | 0/25 (10) | 0.019 | 0.001–0.038 | 0.073 | 0.037 |
| all new physics, in window | 0/13 (5) | 0.036 | 0.002–0.072 | 0.135 | 0.067 |
| all new physics, few-event | 0/3 (2) | 0.125 | 0.006–0.261 | 0.444 | — |
| expected process | 4/5 (3) | 0.750 | 0.579–0.915 | (lower 5 %: 0.36) | — |
| expected process, in window | 3/4 (2) | 0.700 | 0.500–0.894 | — | — |
| all items | 4/30 (13) | 0.145 | 0.084–0.207 | 0.259 | — |
| all items, in window | 3/17 (7) | 0.194 | 0.104–0.285 | 0.362 | — |

Brackets from the unresolved items (Jeffreys): DD new physics, unresolved as null 0/8 → 0.056 (upper 0.208); as real
1/8 → 0.167 (upper 0.397). All new physics: as null 0/35 → 0.014 (upper 0.053); as real 10/35 → 0.292 (0.217–0.367).
The "as real" envelope is the maximum any reading of the record allows, and requires every open anomaly (LSND,
MiniBooNE, 3.5 keV, X17, ANITA, gallium, R_D, 95 GeV, antihelium, cryogenic LEE) to be new physics.

Observation on the covariate: `prior_type` separates the outcomes completely (0/25 versus 4/5). This is the one variable
in the compilation that predicts survival; significance level within 2.5–4σ does not (the in-window and out-of-window
new-physics rates are both zero), and neither does domain (DD 0/7, other 0/18).

## 4. Selection and recall (`P083_selection.csv`)

Two selection effects, treated separately:

(a) **Recall/publication bias in this compilation.** We remember memorable cases. If a real anomaly is recalled with
probability f_real and a null one with f_null, the observed odds are the true odds × f_real/f_null, so
odds_true = odds_obs × f_null/f_real. Confirmed discoveries are essentially always remembered (f_real ≈ 1), quiet
fizzles are not, so f_null/f_real ≤ 1. For f_null/f_real = 1, 0.5, 0.33 the DD rate 0.062 becomes 0.062, 0.032, 0.022;
the all-new-physics rate 0.019 becomes 0.019, 0.010, 0.006. Because the observed real count in the new-physics strata is
zero, this correction only moves the prior-driven mean; it cannot raise the rate.

(b) **Dilution by excursions that never reached a dedicated paper.** For the class "any ~3σ local excursion in a
rare-event search", the denominator should include the many that were quietly absorbed. With 2–5 unrecalled per recalled
item, r falls by ×1/3 to ×1/6 (DD: 0.021–0.010). This correction does **not** apply to the LZ event, which has already
passed the selection to a dedicated paper; its reference class is the one we compiled. We record it because the
distinction is the whole content of the "selection" question: base rates are class-relative.

Side check (assumption-dominated, recalled inputs uncertain): 8 new-physics DD claims in 2008–2026.7 is 0.43 yr⁻¹. A
pure-fluctuation expectation of N_results/yr × [1 − (1 − p_3σ)^N_trials] gives 0.13 (10 × 10), 0.27 (20 × 10),
0.40 (10 × 30), 0.79 (20 × 30) yr⁻¹, bracketing the observed rate; the compilation cannot tell whether DD anomalies
appear more often than chance. The outcome mix (DD resolved: 3 statistical, 2 background, 2 contradicted, 1 hardware)
says roughly half were not pure fluctuations.

## 5. Time to resolution: Kaplan–Meier (`P083_survival.csv`, `P083_survival_curves.json`; Fig. 2)

S(t) = Π_{t_i ≤ t} (1 − d_i/n_i) with resolved items as events at t_res and unresolved items censored at
2026.7 − year_of_claim. F(t) = 1 − S(t) is the fraction resolved by t years.

| stratum | n (resolved) | F(1 yr) | F(2 yr) | F(3 yr) | F(5 yr) | F(10 yr) | median |
|---|---|---|---|---|---|---|---|
| all new physics | 35 (25) | 0.17 | 0.40 | 0.46 | 0.57 | 0.66 | 4.0 yr |
| new physics, comparable independent test available | 24 (20) | 0.25 | 0.46 | 0.54 | 0.71 | 0.80 | 3.0 yr |
| new physics, no comparable independent test | 11 (5) | 0.00 | 0.27 | 0.27 | 0.27 | 0.36 | undefined (> 50 % open after 10 yr) |
| direct detection, all | 10 (8) | 0.20 | 0.40 | 0.64 | 0.76 | 1.00 | 3.0 yr |
| few-event (≤ 5), all | 9 (5) | 0.33 | 0.44 | 0.58 | 0.58 | 0.58 | 3.0 yr |
| expected process | 8 (5) | 0.38 | 0.38 | 0.79 | 0.79 | 0.79 | 3.0 yr |
| new physics, in window | 18 (13) | 0.11 | 0.33 | 0.33 | 0.56 | 0.61 | 4.0 yr |

Routes among the 25 resolved new-physics items (`P083_routes.csv`): independent experiment 13 (52 %), more data from the
same experiment 4 (16 %), reanalysis 3 (12 %), theory revision 3 (12 %), calibration/hardware 2 (8 %). Median time by
route: independent 2.0 yr (mean 3.6), more data 1.75, reanalysis 1.7, hardware 1.85, theory 7.0 (mean 10.7).
Among the 8 resolved DD items: independent 5, more data 1, reanalysis 1, hardware 1.

Few-event items (n ≤ 5; 9 primary): 5 resolved (CDMS Si and Ge by independent exclusion, LEP 115 by the final
combination, KOTO by background identification, IceCube PeV pair confirmed by more data); 4 unresolved (Glashow,
KM3NeT, ANITA, AMS antihelium). Both **single-event** items (Glashow, KM3NeT) are unresolved; no single-event anomaly
in the compilation has ever been resolved, and few-event anomalies were resolved only by exclusion (independent exposure)
or by identifying a background in a reanalysis — never by the anomaly "growing".

## 6. The LZ event on the reference-class covariates

| covariate | LZ event | what the reference class says |
|---|---|---|
| prior type | new physics (inelastic δ ≥ 250 keV or q²-suppressed operators, P027; no predicted rate) | 0/25 resolved new-physics anomalies real |
| number of events | 1 | 0/2 single-event items resolved; few-event 0/3 resolved real |
| significance | 3.4σ local, 2.6σ global | mid-window; in-window new-physics 0/13 |
| background modelled? | modelled b ≈ 2e-4–1e-3 (P001) but P061's unmodelled channel L_acc ≈ 9e-3 dominates | items with identified backgrounds/artefacts (7/25) were those where a mis-modelled channel was later found |
| independent comparable test | yes: XENONnT + PandaX-4T hold ~19 t·yr unexamined above 200 keV (P069); a 60 t detector later (P050) | places LZ in the "comparable test available" stratum: median 3 yr, 71 % by 5 yr |
| route for a real signal | more data (LZ 6.76 t·yr, P(0 | DM) = 0.58, P027) or independent xenon | reference-class real cases were confirmed by more data within ≤ 1 yr, always for predicted processes |

## 7. Prior and posterior for the LZ event (`P083_LZ_posterior.csv`)

P061 Eq. 3: P_ref = r B' / (r B' + 1 − r), B' = (B_DM / f_FP) / B_typ, B_typ = 14.60 (Sellke–Bayarri–Berger bound at LZ's
global p = 4.7e-3; P061 §4.3). B' measures how much stronger LZ's evidence is than the typical reference-class anomaly;
B' ≈ 1 means "typical", so P_ref ≈ r. We draw r from the stratum's Jeffreys posterior (2 × 10⁵ samples, seed 20260916).

| stratum for r | B_DM, f_FP | B' | P(DM) median | 68 % | 97.5 % |
|---|---|---|---|---|---|
| DD new physics (0/7) | 16.4, 3 (P061 central) | 0.374 | **0.012** | 0.001–0.052 | 0.133 |
| DD new physics | 16.4, 1 | 1.12 | 0.035 | 0.003–0.141 | 0.316 |
| DD new physics | 28.7, 1 (optimistic) | 1.97 | **0.059** | 0.005–0.223 | 0.447 |
| DD new physics | 28.7, 3 | 0.655 | 0.021 | 0.002–0.087 | 0.212 |
| DD new physics | 16.4, 10 (pessimistic) | 0.112 | 0.0036 | 0.0003–0.016 | 0.044 |
| all new physics (0/25) | 16.4, 3 | 0.374 | 0.0034 | 0.0003–0.015 | 0.038 |
| all new physics | 28.7, 1 | 1.97 | 0.017 | 0.002–0.073 | 0.170 |
| new physics in window (0/13) | 16.4, 3 | 0.374 | 0.0064 | 0.0006–0.028 | 0.072 |
| new physics in window | 28.7, 1 | 1.97 | 0.033 | 0.003–0.131 | 0.290 |
| few-event new physics (0/3) | 16.4, 3 | 0.374 | 0.026 | 0.002–0.117 | 0.301 |
| few-event new physics | 28.7, 1 | 1.97 | 0.124 | 0.012–0.411 | 0.693 |

Naive alternative, P = r B_DM / (r B_DM + 1 − r), i.e. applying P027's Bayes factor against the *modelled* background
directly to the base rate: median 0.34 (B = 16.4) to 0.48 (B = 28.7) for the DD stratum. This double-counts: every
reference-class anomaly also had a Bayes factor of order 10–100 against its own modelled background (that is what made it
an anomaly), and the base rate already integrates over that. Only the excess of LZ's evidence over the class norm (B')
should update r. We report the naive number to make the size of the double-counting explicit (×30–40 in odds).

**Feeding P061.** P061's structural model maps r to an implied pi_DM (its Table in §4.3: r = 0.02/0.05/0.10/0.20 →
pi_DM = 0.0060/0.0154/0.0323/0.0721 at L_rest = 0.01; 0.028/0.072/0.152/0.338 at L_rest = 0.1). A log-log line fits the
four points to within 3 % (slope 1.08). Propagating the DD posterior for r: pi_DM = 0.0093 (68 % 0.0007–0.043; 95 % upper
0.083) at L_rest = 0.01, and 0.044 (0.003–0.20; upper 0.39) at L_rest = 0.1. P061's log-uniform community prior on
pi_DM ∈ [1e-3, 0.3] has median 0.0173, which corresponds to r = 0.055 — essentially the DD base-rate mean (0.062): P061's
central prior is the outside-view prior. The upper 23 % of P061's log-uniform range (pi_DM > 0.082) lies above the DD 95 %
bound. P061 found P(DM) > 0.5 requires pi_DM ≥ 0.089 even in the optimistic corner, i.e. r ≥ 0.25: the DD posterior puts
4.1 % probability there, the all-new-physics posterior 0.01 %. The base rate therefore removes the P > 0.5 region of P061's
hyper-prior volume (2.4 %) almost entirely and shifts the community median (0.015) toward 0.01.

## 8. Expected timeline for the LZ event

Stratum "new physics, comparable independent test available" (n = 24): P(resolved within 1 / 2 / 3 / 5 yr) =
0.25 / 0.46 / 0.54 / 0.71; median 3 yr, i.e. around 2029, with a 29 % chance of remaining open in 2031. Routes in that
stratum: independent experiment 55 %, more data 15 %, theory 15 %, hardware 10 %, reanalysis 5 %. Mapped onto the
corpus: (i) a XENONnT/PandaX-4T high-energy reanalysis (P069: P(≥ 1 event | best fit) = 0.90–0.995; a null excludes the
L10 best fit at 90 %) is the modal route and the fastest (median 2 yr in the class); (ii) LZ's own 6.76 t·yr (P027:
P(0 | DM) = 0.58, so a null is only weakly informative; P050: 5σ at the best fit by 2027–28 if real); (iii) an internal
artefact/background audit (P061's 0.011 nats). The single-event covariate argues for the slower tail: no n = 1 anomaly in
the compilation has been resolved, and a xenon null at the 90 % level would leave the event "contradicted" rather than
explained — the modal outcome (7/25) for new-physics anomalies with an independent test.

## 9. Validation and robustness

- Beta posteriors checked against closed forms: for k = 0, n = 7, Jeffreys mean = 0.5/8 = 0.0625 ✓; uniform 1/9 = 0.111 ✓.
- KM code checked on the "DD, all" stratum by hand: 10 items, 8 events; S(0.5) = 0.9, S(1) = 0.8, S(2) = 0.6, S(3) = 0.36 ✓ (Fig. 2).
- Log-log fit to P061's table reproduces the four points to 0.97–1.03 ✓ (`loglog_fit_check_ratio`).
- Prior sensitivity: Jeffreys → uniform raises zero-count means by ×1.8 (DD 0.062 → 0.111) and upper bounds by ×1.3.
- Window sensitivity: restricting to 2.5–4.0σ items raises the DD mean to 0.10 (n = 4) — the change is n, not the count.
- Recall sensitivity: removing all "likely"-reliability items leaves DD 0/6 and all-new-physics 0/16; conclusions unchanged.
- Coding sensitivity: recoding "contradicted" as unresolved (cause never identified) gives all-new-physics 0/18 resolved, mean 0.026; DD 0/5, mean 0.083.
- Reference-class heterogeneity: precision anomalies (7 items) removed → all-new-physics 0/19; few-event only → 0/3.

## 10. Failed or abandoned approaches

- A logistic regression of outcome on covariates (σ, n_events, domain, prior type, indep. test) was abandoned: the outcome
  is perfectly separated by `prior_type`, so coefficients diverge; we report the stratified counts instead.
- A parametric (Weibull) survival fit was dropped in favour of Kaplan–Meier: with 25 events and heavy censoring in the
  "no comparable test" stratum the shape parameter was unconstrained.
- An attempt to estimate the number of unpublished ~3σ excursions from the number of DD results per year was kept only as
  a bracket (§4) because every input is a recalled guess.

## 11. Figures

- Fig. 1 `figures/P083_fig1_timeline.png` — every item as a bar from year of claim to year of resolution (arrow: still open);
  colour = outcome group (real / not real / unresolved); diamond = few-event; right-hand column = reported local σ.
- Fig. 2 `figures/P083_fig2_survival.png` — Kaplan–Meier fraction unresolved versus years since claim for four strata.
- Fig. 3 `figures/P083_fig3_beta.png` — posterior CDFs of the base rate r for four strata (Jeffreys prior, resolved items);
  dashed line: r = 0.055, the equivalent of P061's log-uniform pi_DM median.

## 12. Extended discussion

The outside view is not a substitute for the inside view (P001–P061 and the background papers) but a check on it. Three
lessons from the record. First, the covariate that matters is not the significance but the existence of a prediction: the
four confirmed cases were all processes that theory said must be there at about that rate, and their hints were confirmed
within a year by the same instrument. LZ's event has the opposite profile — an unpredicted rate in a window chosen after the
fact (dossier §4) — and P027's finding that the posterior is spread over ~70–100 models is the Bayesian statement of the same
thing. Second, resolution is a social as much as a statistical process: anomalies without a comparable independent test
(DAMA, LSND, MiniBooNE, ANITA, gallium) stay open for decades, and "contradicted" (independent null, cause never found) is
the single most common fate. LZ is fortunate here: two comparable xenon experiments hold more unexamined exposure than LZ
has analysed (P069). Third, the number that the reference class supplies, r ≈ 0.06 with a 95 % bound of 0.23, is neither
negligible nor small enough to dismiss the event; it is, however, almost exactly the centre of the prior P061 already used,
so the corpus's median P(DM) ≈ 0.01–0.02 is the outside-view answer as well as the inside-view one.

## 13. References

LZ Collaboration, arXiv:2609.02823 (2026). Sellke, Bayarri & Berger, Am. Stat. 55, 62 (2001). Kaplan & Meier, J. Am. Stat.
Assoc. 53, 457 (1958). Ioannidis, PLoS Med. 2, e124 (2005). Lyons, arXiv:1310.1284 (2013). Franklin, *Shifting Standards:
Experiments in Particle Physics in the Twentieth Century* (Univ. Pittsburgh Press, 2013). Dorigo, EPJ Web Conf. 95, 02003
(2015). Kass & Raftery, J. Am. Stat. Assoc. 90, 773 (1995). Corpus: dossier §4–5; P001, P027, P050, P061, P069.
The 45 compiled items are recalled knowledge and are cited by experiment and year in Table 1, not as references.

## 14. Tools and provenance (mirrors `provenance/P083.json`)

Agent tools: Read (PAPER_GUIDE full; P001.md, P027.md, P061.md full; P061/details.md §4.2–4.3 and Eq. 3; P061.json as
format template; three figure PNGs, twice each), Bash (ledger rows P001/P027/P061 and titles of P041–P070 via pandas;
environment versions; grep of dossier/plan/papers for prior mentions of historical anomalies; grep of the dataviz palette;
script runs ×4), Write (script, details.md, P083.json, P083.md), Edit (script ×8: row-name bug, dead line, log-log
translation, r_equiv, Fig. 1 title/legend/separator, Fig. 3 CDF), Skill (dataviz; JS validator skipped per PAPER_GUIDE).
Software: python 3.12.13; numpy 2.5.3 (polyfit, geomspace, searchsorted, unique, quantile, default_rng); scipy 1.18.1
(stats.beta, stats.norm.sf); pandas 3.0.5 (DataFrame, crosstab, value_counts, to_csv); matplotlib 3.11.2 (Agg; step,
scatter, annotate); common/lzcommon.py (imported for the LZ dictionary; no numerical function used). Kaplan–Meier and the
Beta algebra: derived by hand in the script.
Script: `output/code/P083_base_rates.py` — `.venv/bin/python output/code/P083_base_rates.py` (≈ 4 s) → `anomalies.csv`,
`P083_beta_table.csv`, `P083_selection.csv`, `P083_survival.csv`, `P083_survival_curves.json`, `P083_routes.csv`,
`P083_outcomes_by_prior.csv`, `P083_LZ_posterior.csv`, `P083_results.json`, three figures, `run_log.txt`.
Local inputs: PAPER_GUIDE; P001/P027/P061 papers; P061 details §4.3 table and Eq. 3; results_ledger (P001, P027, P046, P050,
P059, P061, P069 rows); dossier §4; ENVIRONMENT_versions.txt.
Recalled knowledge: 45 compiled anomalies (28 certain, 15 likely, 2 uncertain-excluded) plus the SBB bound, Kaplan–Meier
estimator, Beta–binomial conjugacy and the Jeffreys prior (all certain) = 49 items.
Datasets: none. Data requests: none. WimPyDD-generated files: none.
