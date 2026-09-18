# P069 — The xenon world in one likelihood: research record

*Simulated arXiv date 2026-09-12. Category XEXP (hep-ex, cross-list hep-ph). Author profile: cross-collaboration phenomenologists. All numbers below are produced by `output/code/P069_xenon_world.py` (log `run_log.txt`, tables `P069_*.csv`, `P069_results.json`) unless marked [paper] (arXiv:2609.02823), [P0XX] (corpus) or [recall] (training knowledge, with reliability).*

## 1. Motivation and framework

LZ's 248 keV event (1.0 (+1.4, −0.7) signal events in 2.84 t·yr, local 3.4σ, global 2.6σ [paper]) is a *counting* hint: one event in a region whose modelled background is 2×10⁻⁴–10⁻² events. Because every tonne-scale dark-matter TPC uses natural xenon and (by convention) the same halo, the recoil spectrum per tonne-year of any interaction is detector-independent; only exposure and acceptance differ (P005, P035). That makes the whole xenon programme one Poisson experiment. This paper asks three things:

1. **Ledger.** How much xenon-TPC exposure exists, how much of it has been examined above 200 keV, and how much is sitting on disk unexamined (either never analysed, or analysed only in a low-energy WIMP ROI)?
2. **Hidden events.** Under LZ's best-fit models, how many 200–270 keV NR-band events should that unexamined exposure contain, dataset by dataset, and what would 0/1/2/3/5 found events mean (significance, rate interval, P001/P020 posterior, P034 timing power)?
3. **Protocol.** What a coordinated, pre-registered, blind joint reanalysis would settle by mid-2027, and what a joint publication buys over separate ones in look-elsewhere terms.

Corpus context: P005 computed expected counts in XENONnT and PandaX-4T's *published* ROIs (≤0.23 events combined) and with 270 keV ROIs (1.53 events); P035 recast those two experiments to inelastic limits (zero events remove the upper 27% of LZ's interval); P020 predicted LZ's untouched 6.76 t·yr (2.4 events, P(0) = 9%); P050 projected the present generation and a 60 t detector (5σ at 12.3 t·yr L10 in an LZ-like window, ≈31 t·yr present-generation exposure by 2030); P038 gave the acceptance of LZ's 270 keV edge versus δ (0.81/0.44/0.17/0.028 at δ = 300/350/366/380 keV); P034 the timing-test requirements; P001 the Bayesian framework; P008 N_eff = 12.2 for LZ's 616 models. This paper joins those pieces into one ledger and one likelihood.

## 2. Inputs

### 2.1 LZ numbers [paper] via `lzcommon.LZ`
Exposure 2.84 t·yr (220 live d × 4.71 t); efficiency plateau 0.96 (0.955 used, P050), 50% at 5.4 and 269.9 keV; L10ˢ 1000 GeV best fit 1.0 (+1.4, −0.7) events; Poisson n = 1 68% band 0.30–2.36 (Table I equals the n = 1 interval, P020); Feldman–Cousins lower limit 0.105 (P021/P050); backgrounds per 2.84 t·yr: 2×10⁻⁴ (within ±2σ of the NR median at S1c > 500 phd, dossier), 5.7×10⁻⁴ (200–270 keV NR band, P016's b_H, adopted by P021/P038/P050), 0.0106 (whole S1c > 500 phd panel, Fig. 5).

### 2.2 Exposure ledger (`P069_exposure_ledger.csv`)

| dataset | exposure adopted (range) t·yr | fiducial t | published ROI edge | status at high energy | source / reliability |
|---|---|---|---|---|---|
| LUX 2013–16 | 0.092 | ≈0.1 | ≈50 keV | archival, never examined | 3.35×10⁴ kg·d [recall, likely]; edge [recall, uncertain] |
| PandaX-II 2016–18 | 0.36 | ≈0.33 | ≈50 keV | archival | 132 t·d [recall, likely]; edge uncertain |
| XENON1T 2018 | 1.0 | 1.3 | ≈41 keV (cS1 < 70 PE) | archival | 1.0 t·yr [recall, certain]; edge [recall, likely] |
| LZ SR1 2022 | 0.77 (0.77–0.90) | 4.71 (5.5) | ≈70 keV (S1c < 80 phd) | low-E analysed, HE untouched | 60 live d × 5.5 t = 0.90 [recall, certain]; 0.77 if the SR3 4.71 t FV is applied; edge [recall, likely] |
| LZ SR3 (WS2024) | 2.84 | 4.71 | 270 keV | **analysed to 270 keV** | [paper] |
| LZ untouched since 1 Apr 2024 | 6.76 (5.7–8.55) | 4.71 | — | untouched | P020: 884 cal d × 0.593 × 4.71 t; live fraction ±25% |
| XENONnT SR0+SR1 | 3.1 | ≈4.0 | ≈60 keV (cS1 ≲ 100 PE) | low-E analysed, HE untouched | 3.1 t·yr [recall, certain; cited in the LZ title/intro]; edge [recall, uncertain] (P005) |
| XENONnT untouched (SR2+) | 3.0 (2.0–5.0) | ≈4.0 | — | untouched | **[recall, uncertain]**: data-taking continued after SR1 (≈mid-2023) with long stops (neutron-veto Gd loading); 3.0 yr × 4.0 t × effective live fraction 0.17–0.42 |
| PandaX-4T Run0+Run1 | 1.54 | ≈2.7 | ≈100 keV | low-E analysed, HE untouched | 1.54 t·yr [recall, certain]; edge [recall, uncertain] (P005) |
| PandaX-4T untouched (Run2/3) | 2.5 (1.5–3.5) | ≈2.7 | — | untouched | **[recall, uncertain]**: running again since late 2023 after an upgrade; 2.8 yr × 2.7 t × 0.2–0.46 |

Totals: **22.0 t·yr on disk** (2 Sep 2026), of which 2.84 analysed above 200 keV and **19.1 t·yr (range 16.1–24.0) hidden at high energy**; the "core" of the three running experiments alone (excluding archival data and LZ SR1) is 16.9 t·yr. Not counted: the 14.5% of the WS2024 fiducial volume that LZ removed for MSSI reasons (≈0.48 t·yr), which is not usable for a clean high-energy search.

Future accrual (t·yr per calendar year, from 2 Sep 2026): LZ 4.71 × 0.593 = 2.79 (P020); XENONnT 4.0 × 0.6 = 2.4 and PandaX-4T 2.7 × 0.6 = 1.62 (P050's fiducial masses [recall, likely] and 60% live-fraction assumption). Cumulative all-xenon exposure: 24.2 (end-2026), 27.6 (mid-2027), 31.0 (end-2027), 37.8 (end-2028) t·yr; LZ's own high-energy track (SR3 + untouched): 9.6 now, 10.5, 11.9, 13.3, 16.1 (`P069_timeline.csv`). P050's "≈31 t·yr by 2030" counted only LZ + the running XENONnT/PandaX-4T from 2026; our ledger adds the untouched XENONnT/PandaX-4T and archival data, hence the larger totals.

### 2.3 Signal spectra and efficiencies
WimPyDD spectra from P050's cache (`output/work/P050/P050_spectra_cache.npz`; 1 TeV; annual-average Baxter-2021 halo (12-day average); unit coupling): L10 = dipole–dipole 4[(q²/m_N²)O4 − O6] (P003/P012 convention) and isoscalar O1 inelastic at δ = 300, 350, 366, 380 keV. Efficiencies: LZ-like ε_LZ(E) = 0.955·½[1+erf((E−5.4)/(√2·2.5))]·½[1−erf((E−269.9)/(√2·11.5))] (P021/P050, matches Fig. S2); extended window ε_400 with E50 = 400 keV, σ = 17 keV (P050 'ext400', from P038's edge study); un-extended published ROI ε_hard(E) = 0.9·θ(E < E_edge) with a 5 keV threshold (P005 convention). Each model is normalised to **1.0 event per 2.84 t·yr in LZ's ROI**: κ_m = 1/[2.84 ∫ dR_m/dE ε_LZ dE].

Derived per-t·yr best-fit rates (`P069_normalisation.csv`):

| model | LZ-ROI rate | 200–270 keV rate | f(200–270) | 200–400 keV (E > 200) rate | A_LZ | A_400 |
|---|---|---|---|---|---|---|
| L10 | 0.352 | 0.1205 | 0.342 | 0.220 | 0.422 | 0.536 |
| δ = 300 | 0.352 | 0.0628 | 0.178 | 0.112 | 0.814 | 0.928 |
| δ = 350 | 0.352 | 0.178 | 0.506 | 0.516 | 0.444 | 0.867 |
| δ = 366 | 0.352 | 0.274 | 0.777 | 1.634 | 0.172 | 0.832 |
| δ = 380 | 0.352 | 0.278 | 0.791 | 10.49 | 0.028 | 0.826 |

A_LZ reproduces P050/P038 (0.422/0.814/0.444/0.172/0.028) and the 400 keV-edge rates reproduce P050 (0.447/0.401/0.687/1.70/10.5 per t·yr; ours 0.447/0.401/0.688/1.703/10.50 for the full ε_400 window). f(200–270) is the fraction of LZ-detectable events with true recoil energy in 200–270 keV: for L10 only 34% (its spectrum peaks at ≈196 keV and has a low-energy shoulder), for δ ≥ 366 keV ≈ 78%.

## 3. Method

### 3.1 Hidden-event counts
For dataset i with exposure E_i and published edge E_edge,i (none for untouched data):
- N_270,i = E_i κ_m ∫_{E>E_edge} dR/dE ε_LZ dE — events an LZ-like 270 keV analysis would find *above* what the published ROI already examined;
- N_200–270,i = E_i κ_m ∫_{200}^{270} dR/dE ε_LZ dE — the "200–270 keV NR band" count (true energy; the 11 keV resolution moves ≈10% of events across the edges, ignored);
- N_tested,i = E_i κ_m ∫ dR/dE ε_hard dE — what the published low-energy analysis already tested (P005);
- N_400,i = E_i κ_m ∫_{E>200} dR/dE ε_400 dE — the pre-registered 200–400 keV window.
P(≥1) = 1 − e^{−N}. The 68% band is N × (0.30, 2.36).

### 3.2 Distribution of the total
Plug-in Poisson at the best fit, and Poisson–Gamma (negative binomial) marginalising LZ's rate: s ~ Gamma(α, 1) per 2.84 t·yr with α = 1.5 (Jeffreys posterior after one count; central 68% 0.42–2.58, mean 1.5 — close to LZ's 0.30–2.36) or α = 2 (flat prior; 0.71–3.29). N ~ NB(α, p = 1/(1+k)) with k = μ_best/1.0.

### 3.3 Scenario statistics (counting in 200–270 keV)
E_new = 19.12 t·yr (k_E = E_new/2.84 = 6.73). Backgrounds scale with exposure: b_tot = b (1 + k_E), b_new = b k_E.
- Combined local significance: Z = Φ⁻¹[1 − P(≥N_tot | b_tot)], N_tot = 1 + N_new; also the PLR form Z = √(2[N ln(N/b) − (N − b)]) and the new-data-only Z.
- Band rate (per 2.84 t·yr) with a flat prior: s | N_tot ~ Gamma(N_tot + 1, scale 1/(1 + k_E)); central 68%/90% and 90% upper limit; ŝ = N_tot/(1 + k_E). Conversion to LZ-ROI units divides by f(200–270), i.e. is model-dependent.
- Expected new count μ_m = E_new × rate_200–270,m (model-dependent) and the model-independent "counting" normalisation μ = k_E × 1 (one observed event per 2.84 t·yr).
- P001/P020 posterior: hypotheses DM (π = 0.01; s flat on [0, 10]), U = unknown background with fixed rate μ_U = −ln(1 − 0.1) = 0.105 per 2.84 t·yr (π = 0.1, so that L_U = 0.1 for the first event), K = modelled background; L_DM = ∫ ds/10 Pois(1 | s + b) Pois(N_new | k(s + b)), L_U = Pois(1 | b + μ_U) Pois(N_new | k(b + μ_U)), L_K = Pois(1 | b) Pois(N_new | k b), with k = μ_m (per unit LZ rate). Checks: N_new = 0, k → 0 gives 0.094 (P001: 0.090; the difference is b = 2×10⁻⁴ vs P001's exact setting); N_new = 2, k = 2.38, b = 2×10⁻⁴ gives 0.358 (P020: 0.358).
- U with a *free* rate and the same prior as DM: L_U ∝ L_DM for any N, so P(DM) ≤ π_DM/(π_DM + π_U) = 0.091 forever — counting cannot separate a DM signal from an unknown background of unknown rate (P001's point).
- Timing power: P034's LLR requirements against a time-flat population of equal rate, N_3σ = 96.5/11.5/6.5/5.3 at δ = 300/350/366/380 keV, scaled as Z ≈ 3√(N_tot/N_3σ) (P034: the √N rule errs ≤25% at small N); all N_tot events (LZ's June event included) carry dates.

### 3.4 Look-elsewhere accounting
p_global = 1 − (1 − p_local)^{N_eff} (computed as −expm1(N_eff log1p(−p))), with N_eff = 1 (pre-registered count in one window), 3 (three pre-registered spectra, treated as independent — an upper bound), 12.2 (P008 for LZ's 616 models), 13.9 (P001 from LZ's 2.6/3.4σ), 3 × 13.9 = 41.7 (three collaborations each publishing a 616-model scan). Check: 3.4σ local → 2.64σ (12.2) / 2.60σ (13.9), LZ: 2.6σ.

### 3.5 Asimov exposures and exclusion
Z_A = √(2[(s + b) ln(1 + s/b) − s]) with s = rate × E, b = b_rate × E; b_rate = 5.7×10⁻⁴/2.84 = 2.0×10⁻⁴ per t·yr (200–270 keV) or 9.66×10⁻⁴ per t·yr above 200 keV for the 400 keV edge (P050 'ext400': P038 MSSI interpolation × NR-band share + accidentals + ν). Zero-event exclusion: E_90 = 2.303/(rate × s_low), E_95 = 2.996/(rate × s_low) (CLs = Poisson for negligible b), for s_low = 1.0, 0.30 (Table I lower edge), 0.105 (FC).

### 3.6 Timeline
Cumulative exposure accrues uniformly between each dataset's start and end (decimal years, recalled/likely: LUX 2013.3–2016.4, PandaX-II 2016.2–2018.6, XENON1T 2016.9–2018.1, PandaX-4T Run0/1 2020.9–2022.4, XENONnT SR0/1 2021.5–2023.6, LZ SR1 2021.98–2022.35, LZ WS2024 2023.23–2024.25 [paper], untouched periods to 2026.67) and at the future rates of §2.2 to end-2028. Milestone years are the times at which a cumulative curve crosses an Asimov or exclusion exposure; for the all-xenon curve they mean "the needed exposure was on disk by then", not "was analysed by then".

## 4. Results

### 4.1 Hidden-event table (`P069_hidden_events.csv`; figure `figures/P069_hidden_events.png`)
Best-fit expected events in the 200–270 keV NR band (L10 / δ=300 / δ=350 / δ=366 / δ=380) and, in brackets, in an LZ-like 270 keV window above the published edge:

| dataset | L10 | δ300 | δ350 | δ366 | δ380 | [270-window, L10 / inel.] | P(≥1), 270-window L10 |
|---|---|---|---|---|---|---|---|
| LUX | 0.011 | 0.006 | 0.016 | 0.025 | 0.026 | [0.030 / 0.032] | 0.03 |
| PandaX-II | 0.043 | 0.023 | 0.064 | 0.099 | 0.100 | [0.118 / 0.127] | 0.11 |
| XENON1T | 0.121 | 0.063 | 0.178 | 0.274 | 0.278 | [0.331 / 0.352] | 0.28 |
| LZ SR1 | 0.093 | 0.048 | 0.137 | 0.211 | 0.214 | [0.248 / 0.271] | 0.22 |
| LZ untouched | 0.815 | 0.425 | 1.203 | 1.850 | 1.882 | [2.380 / 2.380] | 0.91 |
| XENONnT SR0+SR1 | 0.374 | 0.195 | 0.552 | 0.848 | 0.863 | [1.009 / 1.092] | 0.64 |
| XENONnT untouched | 0.362 | 0.188 | 0.534 | 0.821 | 0.835 | [1.056 / 1.056] | 0.65 |
| PandaX-4T Run0+1 | 0.186 | 0.097 | 0.274 | 0.421 | 0.429 | [0.463 / 0.542] | 0.37 |
| PandaX-4T untouched | 0.301 | 0.157 | 0.445 | 0.684 | 0.696 | [0.880 / 0.880] | 0.59 |
| per experiment (L10 / δ366): LZ 0.907 / 2.061 (untouched 0.815 / 1.850), XENONnT 0.735 / 1.669, PandaX-4T 0.487 / 1.106, archival LUX+PandaX-II+XENON1T 0.175 / 0.398, archival + LZ SR1 0.268 / 0.609 | | | | | | | |
| **total** | **2.30** | **1.20** | **3.40** | **5.23** | **5.32** | **[6.52 / 6.73]** | |
| core three (no archival, no SR1) 270-window | | | | | | [5.79 / 5.95] | |
| already tested by the published low-E ROIs | 0.20 | 0 | 0 | 0 | 0 | | |
| 200–400 keV window (E > 200 keV) | 4.20 | 2.15 | 9.87 | 31.3 | 200.7 | | |

Reading: the published low-energy searches of XENONnT, PandaX-4T, XENON1T, PandaX-II, LUX and LZ SR1 have together tested 0.20 (L10) or exactly zero (inelastic) of the events LZ's fit implies (P005's conclusion, now for the whole world); the 270-window counts for XENONnT SR0+SR1 and PandaX-4T Run0+1 (1.09 + 0.54 = 1.63) reproduce P035; the LZ untouched entry (2.38) reproduces P020. The 200–400 keV numbers for δ ≥ 366 keV (31, 201 events) are P038's point restated: those couplings predict tens of events in LZ's own empty 600–1000 phd sideband and are already excluded by LZ's data; they are tabulated for completeness only.

### 4.2 Distribution of the hidden total (`P069_hidden_distribution.csv`)
200–270 keV, all hidden datasets (μ = 2.30 / 1.20 / 3.40 / 5.23 / 5.32):
- Plug-in P(N = 0, 1, 2, 3, ≥4): L10 0.100/0.230/0.265/0.204/0.202; δ300 0.301/0.361/0.217/0.087/0.034; δ350 0.033/0.113/0.193/0.219/0.443; δ366 0.005/0.028/0.073/0.127/0.766.
- P(≥1 somewhere): best fit 0.90/0.70/0.97/0.995/0.995; lower 68% edge (×0.30) 0.50/0.30/0.64/0.79/0.80; FC edge (×0.105) 0.21/0.12/0.30/0.42/0.43 (L10: 1−e^{−0.242} = 0.215).
- Poisson–Gamma (Jeffreys α = 1.5): P(0) = 0.166/0.306/0.108/0.064/0.063, mean 3.46/1.80/5.11/7.85/7.98; flat α = 2: P(0) = 0.092/0.206/0.052/0.026/0.025.
- P(≥3 new) at best fit: 0.41 (L10), 0.12 (δ300), 0.66 (δ350), 0.89 (δ366).
LZ-like 270 window, everything (μ = 6.5–6.7): P(0) = 0.001 (best fit), 0.13–0.14 at ×0.30, 0.49–0.51 at ×0.105; Poisson–Gamma P(0) = 0.047–0.049.

### 4.3 Scenarios (`P069_scenarios.csv`; figure `figures/P069_scenarios.png`), E_new = 19.12 t·yr, b = 5.7×10⁻⁴ per 2.84 t·yr

| N_new | N_tot | Z_comb (Poisson) | Z_comb (PLR) | Z_new only | ŝ band rate (68%) [90%] | UL90 | P(DM) k=L10 2.30 | P(DM) k=δ366 5.23 | P(DM) k=count 6.73 | P(N_new│L10 best) | P(N_new│L10 P–G) | Z_time δ366 / δ380 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 2.62 | 2.98 | — | 0.13 (0.09–0.43) [0.05–0.62] | 0.50 | 0.011 | 0.004 | 0.003 | 0.100 | 0.166 | (1.2 / 1.3) |
| 1 | 2 | 4.27 | 4.53 | 2.67 | 0.26 (0.18–0.60) | 0.69 | 0.065 | 0.014 | 0.009 | 0.230 | 0.174 | (1.7 / 1.8) |
| 2 | 3 | 5.55 | 5.76 | 4.33 | 0.39 (0.27–0.76) | 0.86 | 0.375 | 0.061 | 0.031 | 0.265 | 0.152 | 2.0 / 2.3 |
| 3 | 4 | 6.64 | 6.82 | 5.62 | 0.52 (0.37–0.92) | 1.03 | 0.873 | 0.282 | 0.135 | 0.204 | 0.124 | 2.4 / 2.6 |
| 5 | 6 | 8.49 | 8.64 | 7.70 | 0.78 (0.57–1.24) | 1.36 | 0.999 | 0.964 | 0.875 | 0.054 | 0.074 | 2.9 / 3.2 |

Backgrounds bracket: Z_comb for b = 2×10⁻⁴ / 5.7×10⁻⁴ / 0.0106 = 2.96/2.62/1.41 (N_new = 0), 4.72/4.27/2.73 (1), 6.08/5.55/3.76 (2), 7.23/6.64/4.64 (3), 9.19/8.49/6.15 (5). P(DM) is insensitive to b (e.g. L10 row N_new = 3: 0.874/0.873/0.832). With a free-rate unknown background P(DM) ≤ 0.091 for any N. 5σ local by counting needs **N_tot = 3** for b ≤ 5.7×10⁻⁴ and N_tot = 5 for the whole-panel background. Timing: the √N-scaled P034 requirement gives 2.4σ (N_tot = 4) and 2.9σ (6) at δ = 366 keV, 1.8/2.2σ at 350 keV, 0.6/0.7σ at 300 keV against a time-flat population of equal rate; P020 quotes LR > 10 with probability 0.39/0.74 for N = 3/5 at 366 keV.

Zero events: new data alone give a band rate < 0.342 per 2.84 t·yr at 90%, i.e. in LZ-ROI units (÷f) **< 1.00 (L10), 0.68 (δ350), 0.44 (δ366)** of the best fit — the L10 best fit just excluded (P(N_new = 0 | best fit) = 0.10), 98% of Table I's 0.30–2.36 interval excluded for δ366 (P(0) = 0.005). The combined flat-prior UL90 is 0.50 band units (1.47 L10-ROI units, 0.65 δ366). At the lower 68% edge P(0) = 0.50 (L10) and 0.21 (δ366).

### 4.4 Look-elsewhere (`P069_lee.csv`)
Global Z for N_new new events (b = 5.7×10⁻⁴ band | 0.0106 panel):

| N_new | local | pre-registered count (N_eff 1) | three spectra (3) | LZ 616-model set (12.2 / 13.9) | three separate 616-model papers (41.7) |
|---|---|---|---|---|---|
| 1 | 4.27 │ 2.73 | 4.27 │ 2.73 | 4.02 │ 2.34 | 3.68/3.64 │ 1.77/1.71 | 3.35 │ 1.15 |
| 2 | 5.55 │ 3.76 | 5.55 │ 3.76 | 5.36 │ 3.47 | 5.10/5.07 │ 3.07/3.04 | 4.86 │ 2.69 |
| 3 | 6.64 │ 4.64 | 6.64 │ 4.64 | 6.48 │ 4.41 | 6.26/6.24 │ 4.09/4.06 | 6.07 │ 3.80 |
| 5 | 8.49 │ 6.15 | 8.49 │ 6.15 | 8.36 │ 5.97 | 8.20/8.18 │ 5.74/5.72 | 8.05 │ 5.53 |

A three-event world sample (LZ + 2) is 5σ-class only with a pre-registered analysis (5.4–5.6σ); run through LZ's 616-model machinery it is 5.1σ, and three uncoordinated 616-model papers would leave 4.9σ (and, with the whole-panel background, 3.5 → 2.7σ). The LEE cost of the model scan is 0.2–0.5σ per step at these significances; it is largest exactly in the 3–4σ range where the first new event would land (4.3 → 3.4σ).

### 4.5 Protocol windows (`P069_protocol_windows.csv`, `P069_zero_event_exclusion.csv`)
Asimov exposures (new data only, counting):

| model | window | rate/t·yr (best) | E_3σ | E_5σ | E_5σ at ×0.30 | μ in 19.1 t·yr | Z_A(19.1) | E_90 zero events: best / 0.30 / 0.105 |
|---|---|---|---|---|---|---|---|---|
| L10 | 200–270 | 0.121 | 6.9 | 19.2 | 82 | 2.30 | 5.0 | 19.1 / 63.7 / 182 |
| L10 | 200–400 | 0.220 | 4.6 | 12.8 | 57 | 4.20 | 6.1 | 10.5 / 35.0 / 100 |
| δ300 | 200–270 | 0.063 | 15.0 | 41.7 | 184 | 1.20 | 3.4 | 36.7 / 122 / 349 |
| δ300 | 200–400 | 0.112 | 10.5 | 29.2 | 138 | 2.15 | 4.0 | 20.5 / 68.3 / 195 |
| δ350 | 200–270 | 0.178 | 4.4 | 12.1 | 51 | 3.40 | 6.3 | 12.9 / 43.1 / 123 |
| δ350 | 200–400 | 0.516 | 1.6 | 4.6 | 19.6 | 9.9 | 10.2 | 4.5 / 14.9 / 42.5 |
| δ366 | 200–270 | 0.274 | 2.6 | 7.3 | 30 | 5.23 | 8.1 | 8.4 / 28.1 / 80 |
| δ366 | 200–400 | 1.634 | 0.43 | 1.19 | 4.9 | 31 | 20 | 1.4 / 4.7 / 13.4 |

Cross-check with P050 (LZ-like window including the 100–200 keV bins, Asimov): E_5σ = 12.3/11.5/8.5/6.8 t·yr for L10/δ300/δ350/δ366; ours in the 200–270 keV band alone are 19.2/41.7/12.1/7.3 because L10 and δ300 put most of their LZ-ROI events below 200 keV (f = 0.34, 0.18), where the background is 0.04 per t·yr; for δ ≥ 350 keV the two agree within 15–40%. Our E_90 for the Table I lower edge in LZ-ROI units reproduces P050 exactly when expressed in the LZ window (21.8 t·yr at 0.30; 62.3 at 0.105; log line [6]).

### 4.6 Timeline and the mid-2027 joint dataset (`P069_timeline.csv`, `P069_joint_end2026.csv`; figure `figures/P069_exposure_timeline.png`)
Milestone years (exposure on disk crossing the Asimov/exclusion exposure; LZ track vs all xenon):
- L10: E_5σ(best) = 19.2 t·yr — all xenon crossed it in 2026.1; LZ alone does not reach it by end-2028 (16.1 t·yr). Zero-event exclusion of the best fit (19.1 t·yr): same. Exclusion of the 0.30 edge (64 t·yr): neither.
- δ350: E_5σ = 12.1 — LZ alone 2027.6; all xenon 2024.6. 3σ at ×0.30 (18.3): all xenon 2025.9; LZ alone never by 2028.
- δ366: E_5σ = 7.3 — LZ alone 2025.9 (i.e. already on LZ's disk, cf. P020/P050); all xenon 2023.5; 5σ at ×0.30 (30 t·yr) all xenon 2027.9; 0.30-edge exclusion (28 t·yr) all xenon 2027.6.

Joint dataset available to a paper written in mid-2027 (data to end-2026): 21.4 t·yr new + 2.84 analysed = 24.2 t·yr.

| model | μ_new best | μ_new ×0.30 | P(0) best / ×0.30 | P(N_tot ≥ 3 → 5σ) best / ×0.30 | median N_new, Z_comb | Z_A best / ×0.30 | UL90 if zero (LZ-ROI units) |
|---|---|---|---|---|---|---|---|
| L10 | 2.58 | 0.77 | 0.076 / 0.46 | 0.73 / 0.18 | 2, 5.5σ | 5.3 / 2.6 | 0.89 |
| δ300 | 1.34 | 0.40 | 0.26 / 0.67 | 0.39 / 0.06 | 1, 4.2σ | 3.6 / 1.7 | 1.72 |
| δ350 | 3.80 | 1.14 | 0.022 / 0.32 | 0.89 / 0.32 | 4, 7.5σ | 6.6 / 3.2 | 0.61 |
| δ366 | 5.85 | 1.75 | 0.003 / 0.17 | 0.98 / 0.52 | 6, 9.2σ | 8.5 / 4.2 | 0.39 |

So by mid-2027 a joint 200–270 keV reanalysis reaches 5σ with probability 0.73 (L10) to 0.98 (δ366) if LZ's best fit is true, and zero events would exclude the best fit at 90% (UL 0.89 for L10, 0.39 for δ366; 1.72 for δ300 — the low-δ inelastic case is the hardest because only 18% of its events lie above 200 keV). The 0.30 lower edge is excludable by zero events only for δ ≥ 366 keV (28 t·yr, all-xenon end-2027) in the 200–270 keV band, or with a 400 keV edge (L10 35 t·yr ≈ end-2028; δ350 14.9 t·yr — already on disk).

### 4.7 Proposed protocol (qualitative, quantified above)
1. **Pre-registration** (public, time-stamped) of: ROI = NR band (±2σ of the NEST-tuned median, LZ-like) for true recoil energies 200–270 keV (baseline; identical to LZ so the SR3 event enters without re-selection) and 200–400 keV (extension; S2c ceiling tracking the NR band, P038/P050); fiducial volume with LZ-like wall margins (LZ removed 14.5% of its FV for MSSI; each experiment fixes its wall/cathode cuts before unblinding); three signal spectra only — L10 (1 TeV), O1 inelastic δ = 350 keV and δ = 366 keV (the corpus's physical-coupling maximum, P007/P021) — with the counting statistic as the primary test (N_eff = 1) and the three spectra as pre-registered secondary shape tests (N_eff ≤ 3).
2. **Common MSSI validation**: each experiment publishes its high-energy sideband (LZ: 800–1700 phd, RFR 5.4 t, 18 obs vs 21.5 pred) and wall-sideband counts before opening the ROI; a ×2 disagreement in any sideband vetoes the unblinding (P004: LZ's wall MSSI is constrained to k < 1.64 at 95%).
3. **Blindness**: LZ's salt failed above 55 keV; the other experiments should blind by masking the 200–400 keV NR band rather than by salting.
4. **Joint likelihood**: product of per-experiment Poisson terms with a common rate per t·yr and the exposure ratios of §2.2; per-event energies and dates are released (P035: one event fixes δ to ±15 keV; P034/P020: dates test the June clustering).
5. **Reporting**: one joint paper with the pre-registered global significance; separate technical notes on acceptances.

## 5. Validation and robustness
- Normalisation: A_LZ and 400 keV rates match P050 to ≤0.3%; the XENONnT+PandaX-4T 270-window count 1.63 matches P035; LZ untouched 2.38 matches P020; the P001 (0.094 vs 0.090) and P020 (0.358) posteriors are reproduced; LZ's 3.4 → 2.6σ LEE is reproduced with N_eff 12.2–13.9; E_90 = 21.8/62.3 t·yr match P050.
- Exposure uncertainty: hidden exposure 16.1–24.0 t·yr around 19.1 (−16%/+26%) from the untouched XENONnT/PandaX-4T entries and LZ's live fraction; all hidden counts and μ_new scale linearly; the archival sets add only 1.45 t·yr (0.18 L10 events) and LZ SR1 0.77 t·yr.
- Background bracket 2×10⁻⁴–0.0106 changes Z_comb by ±0.4σ at N_tot = 3 (5.55 → 6.08 / 3.76) and the P(DM) by <5%.
- Rate band: μ × 0.30–2.36; Poisson–Gamma with α = 1.5 vs 2: P(0) 0.166 vs 0.092 (L10).
- Resolution: counts use true energy; folding σ_E = 11√(E/248) keV would move ≈10% of the 200–270 keV events across the edges (P021/P050 fold it; our counts are consistent with P050's observed-energy medians to that level).

## 6. Failed or abandoned approaches
- A first version defined the counting test on the whole LZ-like window (rate 0.352 per t·yr) but with the 200–270 keV background only (2×10⁻⁴ per t·yr), giving E_5σ = 5.5 t·yr — inconsistent (the L10 window includes 100–200 keV where the NR-band background is 0.04 per t·yr, P050) and abandoned in favour of the 200–270 keV band statistic.
- 1 − (1 − p)^N_eff evaluated directly underflowed for p ≈ 10⁻¹⁵ (returned Z = 37); replaced by −expm1(N_eff·log1p(−p)).
- A first attempt quoted the zero-event 90% limit of the end-2026 joint dataset as 2.303/(rate × 2.84) instead of 2.303/(rate × E_joint); corrected.
- The timeline "5σ year" for the all-xenon curve was initially read as a discovery date; it is only the date at which the required exposure existed on disk, and is labelled so.

## 7. Discussion
The xenon programme has, without knowing it, already run the experiment that would test LZ's hint: 19 t·yr of data taken with the same target, of which 3.1 + 1.54 + 1.0 t·yr were even analysed, but in ROIs that end below the energies where LZ's fitted spectra live (0.20 or 0 expected events tested). The expected 200–270 keV NR yield of that data is 2.3 (L10) to 5.2 (δ = 366 keV) events at the best fit, so the probability that at least one such event already exists on somebody's disk is 90–99.5% (50–79% at LZ's lower 68% edge). Two found events would make the world sample 5σ-class locally; zero would exclude the L10 best fit at 90% and cut the δ ≥ 350 keV interval to below half of it. By mid-2027 the joint dataset grows to 24 t·yr and settles the best-fit hypothesis either way with ≥73% probability. None of this needs a new detector (cf. P050): it needs three collaborations to open the same window with the same pre-registered rules, and to publish once — the look-elsewhere cost of three separate 616-model scans is 0.5–0.7σ at exactly the 3–5σ level where the decision will be made. What counting cannot do, in this or any exposure, is separate dark matter from an unknown background of unknown rate (P(DM) ≤ 0.09 in P001's framework); that requires the spectral shape (P035, P050: 4–6 events), the June clustering (P034: 5–12 events for δ ≥ 350 keV) and a second target (P015, P046).

## 8. Figures
- `figures/P069_exposure_timeline.png` — cumulative xenon-TPC exposure by experiment (stacked, Okabe–Ito colours) 2013–2028, the LZ high-energy track (black), the LZ posting date, and the 200–270 keV counting thresholds (5σ best fit: 7.3 t·yr δ = 366, 19.2 t·yr L10). Future accrual shaded.
- `figures/P069_scenarios.png` — left: combined local Z vs N_new for three backgrounds; right: P(DM) in P001's framework for the L10 and δ = 366 keV expectations, the free-rate cap 0.091, and the predictive P(N_new) (plug-in and Poisson–Gamma, L10).
- `figures/P069_hidden_events.png` — expected 200–270 keV events per dataset and model.

## 9. References
LZ Collaboration, arXiv:2609.02823 (2026). E. Aprile et al. (XENON), PRL 135, 221003 (2025) [3.1 t·yr]. Z. Bo et al. (PandaX), PRL 134, 011805 (2025) [1.54 t·yr]. E. Aprile et al. (XENON1T), PRL 121, 111302 (2018) [1.0 t·yr]. D. S. Akerib et al. (LUX), PRL 118, 021303 (2017). X. Cui et al. (PandaX-II), PRL 119, 181302 (2017); Q. Wang et al., Chin. Phys. C 44, 125001 (2020). J. Aalbers et al. (LZ), PRL 131, 041002 (2023) [SR1]. G. Cowan, K. Cranmer, E. Gross, O. Vitells, EPJC 71, 1554 (2011). G. J. Feldman, R. D. Cousins, PRD 57, 3873 (1998). E. Gross, O. Vitells, EPJC 70, 525 (2010). D. Baxter et al., EPJC 81, 907 (2021). I. Jeong et al. (WimPyDD), CPC 276, 108342 (2022). Corpus: dossier 00, P001, P003, P004, P005, P007, P008, P012, P015, P016, P020, P021, P034, P035, P038, P046, P050.

## 10. Tools and provenance (mirrors `output/provenance/P069.json`)
- Agent tools: Read ×22 (PAPER_GUIDE, dossier, ledger (partial), P001/P005/P012/P020/P021/P034/P035/P038/P050 papers, lzcommon.py, P050 and P020 scripts, figures); Bash ×23 (directory listings, table extraction from P005/P020/P034/P035/P038/P050 work files, npz inspection, grep of P050/P020 code, four script runs, figure/table checks); Write ×2 (script, scratch tail); Edit ×3; Skill ×1 (dataviz; JS validator skipped per guide).
- Software: python 3.12.13; numpy 2.5.3 (trapezoid, interp, expm1/log1p); scipy 1.18.1 (stats.poisson/gamma/nbinom/norm, special.erf, integrate.quad, optimize.brentq); pandas 3.0.5; matplotlib 3.11.2; common/lzcommon.py (LZ constants); WimPyDD 2.0.4 only through P050's cached spectra (no new WimPyDD calls; no WimPyDD files generated).
- Script: `output/code/P069_xenon_world.py`, command `.venv/bin/python output/code/P069_xenon_world.py` (≈5 s).
- Local inputs: `output/work/P050/P050_spectra_cache.npz` (spectra), P050 tables (normalisation, background rates, discovery/exclusion/timeline), P038 acceptance table, P005 counts/fractions, P020 exposure/decision/UL tables and code (posterior), P034 required-exposure table, P035 summary, lzcommon.py, dossier, ledger, P001/P005/P012/P020/P021/P034/P035/P038/P050 papers.
- Recalled knowledge (13 items): XENONnT 3.1 and PandaX-4T 1.54 t·yr (certain); XENON1T 1.0 t·yr (certain); LZ SR1 60 live d × 5.5 t (certain); LUX 3.35×10⁴ kg·d and PandaX-II 132 t·d (likely); XENONnT ≈4.0 t and PandaX-4T ≈2.7 t fiducial (likely); XENONnT/PandaX-4T continued running after their published runs, exposures 2–5 and 1.5–3.5 t·yr (uncertain); ROI edges XENONnT ≈60, PandaX-4T ≈100, XENON1T ≈41, LZ SR1 ≈70, LUX/PandaX-II ≈50 keV (uncertain/likely); run dates (likely); Poisson–Gamma/negative-binomial, Jeffreys Gamma(1.5) (certain); Asimov Z, CLs 2.303/2.996 (certain); Gross–Vitells trials factor formula (certain).
- Datasets: none. Data requests: none. WimPyDD-generated files: none.
