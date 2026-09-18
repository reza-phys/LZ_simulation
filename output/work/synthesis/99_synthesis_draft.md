# 99 · Synthesis: the simulated community response to the LZ 248 keV event, 3–16 September 2026

*Coordinator's synthesis of the 100-paper corpus (P001–P100), written 16 September 2026 (simulated). All numbers are traceable to the ledger (`results_ledger.csv`) and the cited papers; recalled external numbers keep the reliability flags of the papers that used them.*

## 1. Where the simulated community stands after two weeks

The corpus converged fast on three facts and one open question.

**Fact 1: the event is a single nuclear recoil at 246–248 keV, not a mundane background.** Every modelled background was pushed far below the event. Wall multiple-scatter-single-ionisation (MSSI) needs a mismodelling factor k ≥ 620 (P004), 6×10⁴ with photon transport (P033) and 1.4×10⁵ once the vetoes' silence is added (P073), against a sideband-allowed k < 1.64. Electronic-recoil leakage into the event's position needs a recombination tail that LZ's own Fig. 5 and the band-shape symmetry exclude at the 10⁻⁵–10⁻³ level (P010, P056); Migdal or bremsstrahlung companions push recoils *up*, not down, the band (P095); the event's charge-poor position is a 6 % fluctuation for a genuine NR (P024). Neutrons of every origin give ≤ 10⁻⁵ lone events (P013, P049, P063), accidentals are excluded at 4.3σ (P022), activation products cannot reach 248 keV (P029), and the atmospheric-neutrino floor is 3×10⁻⁵ events per tonne-year (P019, P087). The combined topological odds favour a single NR over each modelled class by 10^3.6–10^6.9 (P093). The energy scale is self-consistent: the "265 keV" reading of Table S5 is a printing artefact, LZ's own scale gives 246–248 keV (P009, P064), and the statistical resolution is 11.4 keV rather than the quoted 23 keV (P098).

**Fact 2: the statistical case is exactly what LZ said, and no stronger.** LZ's 3.4σ local was reproduced to 3.0–3.2σ from the published summaries (P052), the look-elsewhere correction to N_eff = 12–14 (P008, P071), and the global 2.6σ survives every alternative treatment; adding the ~200 spectra the community itself introduced lowers it to 2.50σ (P071). One event is strong evidence against the *modelled* background (B₁₀ ≈ 10²–10³) but only positive evidence for dark matter once the model space and unmodelled backgrounds are admitted (B_DM = 16–29 model-marginalised, P027; B₁₀/N_eff ≈ 7–37, P001). Under any community-wide prior the posterior probability of dark matter is small: median 0.015, 68 % interval 0.002–0.14 (P061), consistent with a historical base rate of 0.06 for unpredicted ~3σ hints in direct detection, none of which has ever turned out real (P083).

**Fact 3: the only DM readings that fit *naturally* are inelastic scattering near the kinematic edge and q⁴-suppressed spin operators.** Elastic spin-independent scattering is excluded by the absence of ~2 700 low-energy companions (P003, P016). The inelastic isoscalar O₁ likelihood peaks at δ ≈ 380 keV for 1 TeV (P021), a preference the S1c < 600 phd edge manufactures (P038); the joint direct-detection region is m ≥ 400 GeV, δ = 335–390 keV (P082). The alternative is a companion-free elastic operator of the L10/O6/O9 family, which needs a UV completion with Λ ≪ m_χ (P031, P074).

**The open question is whether the event is real at all**, and the corpus agrees that it cannot be settled from the 2.84 t·yr already analysed. The residual probability mass sits in the *unmodelled* classes — an instrumental artefact of a kind LZ has not characterised (P041, P077, P093) and a genuine one-off — and only new data or waveform-level information moves it. Three demands crystallised, exactly as forecast: analyse LZ's untouched exposure (P020, P079, P088), re-open XENONnT's and PandaX-4T's existing data above 200 keV (P005, P035, P069), and release enough of the likelihood to allow independent fits (DR-001, DR-002, DR-003).

## 2. Updated explanation probabilities

The dossier (2 September) assigned prior probabilities to eleven explanation classes. P091 multiplied that prior by thirty corpus-derived likelihood factors (background rates at the event's location, sideband constraints, position and veto ratios, tail shapes, the date, the model-marginalised DM evidence) and propagated the quoted ranges by Monte Carlo. The coordinator's adopted values (last column) follow P091 except where noted below.

| Class | Explanation | Dossier prior | Corpus LR vs A (P091) | P091 posterior median (68 %) | Adopted |
|---|---|---|---|---|---|
| A | fluctuation of the modelled background | 0.10 | 1 | 0.033 (0.013–0.076) | **0.13** |
| B | wall / RFR MSSI mismodelled | 0.19 | 2.6×10⁻³ | 1.6×10⁻⁴ | 0.001 |
| C | ER leakage / charge-poor ER | 0.13 | 0.17 | 0.0073 (0.0013–0.041) | 0.02 |
| D | neutron of any origin | 0.08 | 0.069 | 0.0018 | 0.005 |
| E | accidental coincidence | 0.05 | 0.22 | 0.0036 | 0.01 |
| F | instrumental artefact or uncharacterised unknown | 0.18 | 9.3 | 0.64 (0.31–0.87) | **0.60** |
| G | calibration / energy-scale error | 0.05 | 0.038 | 6×10⁻⁴ | 0.005 |
| H | non-DM new physics | 0.02 | 0.17 | 0.0011 | 0.004 |
| I | inelastic dark matter | 0.10 | 6.1 | 0.23 (0.07–0.50) | **0.19** |
| J | q⁴-spin elastic EFT dark matter | 0.06 | 0.68 | 0.014 (0.0045–0.037) | 0.03 |
| K | other dark matter | 0.04 | 0.053 | 7×10⁻⁴ | 0.005 |
| | **dark matter, I + J + K** | 0.20 | | 0.25 (0.08–0.54; 95 % 0.03–0.76) | **0.225** |

Where I depart from P091: class A is raised from 0.03 to 0.13 because P091's likelihood for A is the *local* background density at the event, whereas the honest reference for "a fluctuation" is the whole 293-spectrum search (global p ≈ 0.006, P008/P071) and the base rate of unpredicted ~3σ hints (P083); class F is correspondingly lowered. The DM total is prior-dominated: 0.015 under P061's sceptical community prior and P083's base rate, 0.25 under the dossier prior, 0.22 adopted. Sixty percent of the variance sits in the artefact likelihood L_F and 29 % in the forking-paths divisor (P091) — both quantities only LZ can pin down.

What changed the numbers. The four detector classes B–E lost almost all their weight to quantitative arguments that did not exist on 2 September: the MSSI position and veto likelihood ratios (P033, P070, P073, P093), the recombination-tail bound (P056), the accidental UDT count (P022) and the neutron transport chain (P013, P049, P063). Class G collapsed once P009/P064/P098 showed the scale is consistent and the resolution is 11 keV. Class F is the one class the corpus could not shrink: the S1 pulse shape at 540 phd is intrinsically weak (P077), LZ published no waveform-level quantities (DR-003), and every artefact scenario that *was* computable (charge loss, P041; activation, P029) was excluded, so what remains is the uncharacterised residual. Within the DM classes, weight moved from the pure Higgsino to generic pseudo-Dirac fermions with a 9–35 GeV dark photon, because solar capture excludes the fixed-coupling Higgsino at every δ (P076) and Sommerfeld-enhanced annihilation excludes the higher electroweak multiplets (P084). Class K shrank as cosmic-ray-boosted, exothermic-origin, millicharged and composite relics were each excluded on companion counts (P040, P072, P080). The ten papers that carried most of the information are, by sequential Kullback–Leibler gain (P091): P041, P004, P016, P029, P010, P013, P022, P061, P001, P019 — mostly the background papers.

## 3. The ten most important papers

1. **P021 + P038 (inelastic likelihood and the efficiency edge).** Located the isoscalar-O₁ maximum beyond LZ's grid (δ ≈ 380 keV, Z = 3.6σ) and then showed the preference is an artefact of the 600 phd edge (1000 phd → 355 keV, 2.9σ) — the single most important methodological correction to how the event is read.
2. **P004 / P033 / P073 (wall MSSI).** Turned "MSSI is the leading background" into a quantitative exclusion: k_required rises from 620 to 1.4×10⁵ while sidebands allow < 1.64.
3. **P076 (solar capture).** The only paper that *removed* a popular DM reading: the fixed-coupling Higgsino is captured by core Fe/Ni and annihilates 10²–10³ above IceCube's limits at every δ, and the 360–385 keV corner is closed for all three mediator families.
4. **P052 (LZ likelihood reproduced).** Rebuilt LZ's three-sample 2D likelihood from Tables I/S1/S2 and Fig. 5 to rms 0.42σ, with 3.03σ toy-calibrated for L10, and localised the residual 0.3σ in the unpublished within-panel S1c dependence (DR-002).
5. **P027 + P061 + P083 (model marginalisation, prior sensitivity, base rates).** Together they set the community's probability language: B_DM = 16–29, P(DM) median 0.015, and an outside-view base rate of 0.06 that coincides with the community prior.
6. **P003 + P016 (companion counting).** Established N_lo, the number of low-energy events per 200–270 keV event, as the decisive spectral statistic (SI 2 750; only q⁴-spin operators ≤ 5) and explained why LZ's likelihood still gave 2.7σ to spectra with tens of companions.
7. **P082 (global inelastic combination).** The surviving region after every archival xenon zero: m ≥ 400 GeV, δ = 335–390 keV, σ_n = 8×10⁻⁴²–3×10⁻³⁶ cm² at 1 TeV; the archival zeros cost only 0.06–0.11σ.
8. **P069 + P088 (the xenon world and the timeline).** 17.9–19.1 t·yr of xenon data already on disk have never been examined above 200 keV and should hold 2–5 events at the best fit; a 5σ population-vs-one-off verdict is on disk for δ ≥ 350 keV and comes by 2027 for L10.
9. **P068 + P018 + P085 (astrophysical inputs).** Escape velocity is the dominant systematic of every inelastic inference (κ spans 3×10⁻³–150 at δ = 366 keV); the exact observer velocity at the event is 265.74 km/s and solar focusing adds ×1.04–1.16 to near-edge rates.
10. **P093 (topology odds).** The synthesis of every published handle: single NR favoured over each modelled background by 10^3.6–10^6.9, but over an uncharacterised artefact by only 10^2.5 — which is where the argument now lives.

## 4. The most viable dark-matter scenarios

**Inelastic pseudo-Dirac fermion with a heavy or GeV-scale mediator (the leading class).** m_χ ≥ 400 GeV with a flat likelihood above 1 TeV (P062, P082); δ = 335–390 keV at 90 % with the LZ edge, 300–360 keV if the 1000 phd re-analysis moves the peak as P038 predicts; σ_n(N = 1) = 8×10⁻⁴²–3×10⁻³⁶ cm² at 1 TeV (P082), i.e. 1.2×10⁻⁴²–2×10⁻³⁹ cm² proton-only for the dark photon (P011). Viability conditions: m_A′ = 9–35 GeV and ε = 7×10⁻⁵–10⁻³ after Planck, BBN/dark-Higgs and BaBar (P025, P086); thermal relic via secluded annihilation for δ ≤ 356 keV at 1 TeV, any δ with a dilution factor 1.3–2.3 (P075); χ₂ must have decayed, τ < 3–9×10¹⁶ s (P026, P058, P066). The heavy-Z′ variant survives only for m_Z′ ≲ m_χ and δ ≤ 300–340 keV (P054, P075). What kills it: solar-neutrino limits already exclude the fixed-coupling Higgsino at every δ and both mediator families for δ ≥ 360–370 keV (P076).

**Electroweak multiplets.** The pure Higgsino (δ = 366–380 keV at 1–1.3 TeV; P007, P037, P075) is excluded by solar capture (P076); the Y ≥ 1 multiplets are excluded by Sommerfeld-enhanced annihilation unless they are a ≤ 20 % sub-component (P084). Collider tests are out of reach before FCC-hh or a muon collider (P014, P048).

**Companion-free elastic q⁴-spin operators (L10/O6/O9/O14/O15).** d₁₀ ≈ 0.28 (0.15–0.43) in LZ's normalisation at 1 TeV (P012; provisional pending DR-001), N_lo = 0.2–0.4 (P003, P012), predicts 2.4 events in LZ's untouched 6.8 t·yr (P020). Requires a UV scale Λ = 1.5–62 GeV ≪ m_χ (P031); no light-mediator dressing rescues any excluded operator (P074). Spectrally degenerate families (O6 = E_R × O10; O9/O14; L12, L16 = O13) need 15–60 events to separate (P057, P089).

**Magnetic inelastic (MiDM) and composite dark hadrons.** μ_χ ≈ 2×10⁻⁴ μ_N at δ = 300 keV; the de-excitation photon excludes δ > 355 keV and the thermal moment fits only at δ ≈ 351 keV with P(clean) = 0.15 (P023, P042, P065). Alive but contrived.

**Excluded DM readings.** Elastic SI and any velocity-independent contact interaction (N_lo ≥ 350–2 750; P003, P030, P040); exothermic down-scattering as the event's origin (B ≤ 2, 2–7 companions in the empty 125–200 keV bin; P058, P072); nuclear-excitation hybrids (P036); photon-mediated anapole and EDM (P044); cosmic-ray-boosted, millicharged, SIMP and superheavy relics (P040, P080). Halo substructure cannot help: no known stream or disk component reaches 700 km/s (P030, P055, P097).

## 5. The most viable non-DM explanations

1. **A statistical fluctuation of the modelled background** at 2.5–2.6σ global (P008, P071), rate 3.5–9×10⁻⁴ in the event's neighbourhood (P052, P093) — the default hypothesis, and the reference-class base rate says it is the likeliest single explanation (P083).
2. **An uncharacterised instrumental artefact** — the only class the corpus could not bound below 10⁻²·⁵ relative odds (P093). Computable variants (bulk or local charge loss, P041; activation, P029; accidentals, P022) are excluded; the residual is whatever LZ's waveform quantities could still reveal (DR-003; P077 shows the S1 pulse shape at 540 phd carries a likelihood ratio ≤ 2–5).
3. **Wall or RFR MSSI with a grossly wrong spatial model.** Requires k ≥ 10³–10⁵ (P004, P033, P073) and is contradicted by the sidebands; P079 nevertheless found that LZ's own MSSI table implies a wall depth scale of 1.2–1.4 cm, seven times steeper than photon transport gives, so the *shape* of the MSSI model is genuinely uncertain even if its normalisation is not.
4. **A charge-poor electronic recoil** (¹²⁴Xe or ¹²⁵I double-vacancy line with anomalous charge suppression): ≤ 10⁻² events, excluded by the empty gap between the bands (P010, P056).

Neutrons, astrophysical neutrinos and nuclear excitations are not viable (P013, P049, P063, P060, P087, P036).

## 6. Decisive tests

| Test | Who / what | Expected signal at the LZ best fit | Timeline | Source |
|---|---|---|---|---|
| LZ's untouched exposure, same 600 phd edge | LZ, ~6.8 t·yr since April 2024 | 2.4 events (L10), P(0) = 9 %; 5σ (with the first event) by Oct 2031 for L10 | release 2027 | P020, P088 |
| Same data with a 1000 phd edge | LZ re-analysis | ×1.9–6.3 signal; P(5σ) = 0.83 (L10) to 1.00 (δ = 366 keV) in 524 live days; 8 cm stand-off already optimal | 2027 | P038, P079 |
| The 600–1000 phd sideband already on disk | LZ | 3.6/13/103 events at δ = 350/366/380 keV on 0.009 background; SR3's empty sideband already gives P(0) = 0.02 at δ = 366 keV | now | P038, P088 |
| High-energy re-analysis of existing XENONnT and PandaX-4T data | 17.9 t·yr unexamined above 200 keV | 2.2/1.1/3.2/4.9 events (L10/δ 300/350/366); two events = 4.3σ new-only, BF > 100 against a one-off | 2027 | P005, P035, P069, P088 |
| Annual modulation | LZ + others, June-phased | 3σ needs 5–12 events for δ ≥ 350 keV, 97 for δ = 300 keV | 2029–2030 (LZ alone) | P034, P006 |
| Spectral shape, elastic vs inelastic | any xenon detector | 2–7 events; O6 vs O10 15–20; L10 vs O6 30–60 | with the re-analyses | P050, P057 |
| CaWO₄ (tungsten) | CRESST-scale 10 kg·yr | 2.8/0.37/0.03 events at δ = 366/350/300 keV; decisive only at 100 kg·yr | 2030s | P015, P046, P088 |
| Isotopic (¹³⁶Xe-enriched vs natural) | hypothetical 5 t enriched TPC | nine shared events separate spin from inelastic at 3σ | > 2030 | P047, P078 |
| Directional detectors | gas TPCs, emulsions | kinematically blind below A ≈ 100; xenon gas 1000 m³ needs 41 yr | not decisive | P067 |
| Data releases | LZ collaboration | DR-001 (L10 normalisation, interval tables), DR-002 (event list and 2D PDFs), DR-003 (waveform quantities) | on request | P012, P052, P093 |

Neither the atmospheric-neutrino floor (one event per 3×10⁴ t·yr; P087) nor liquid scintillators (P096) nor indirect detection (P025, P084) can decide the question; the neutron-star heating signal (P039) is real but unobservable before JWST-class surveys of old neutron stars.

## 7. A falsifiable prediction for LZ's next data release

Pre-registered here (P081, P088, P099 give the machinery; all counts refer to the 200–270 keV NR-band region with LZ's current selections unless stated). Assume LZ releases the ≈ 6.8 t·yr taken since April 2024 (≈ 524 live days; 2.4 × the published exposure).

1. **Background or one-off.** Zero events in 200–270 keV with probability ≥ 0.998 (modelled 5.6×10⁻⁴ per 2.8 t·yr, 9×10⁻⁴ with the corpus's revised residuals); zero in 55–270 keV NR band with probability ≥ 0.99.
2. **Dark matter at the corpus posterior rate.** Per 2.8 t·yr: P(0/1/≥2) = 0.75/0.17/0.08 in 200–270 keV, P(≥1) = 0.50 in 55–270 keV; for the full 6.8 t·yr P(≥1) ≈ 0.5 in the band and ≈ 0.8 in 55–270 keV. The band-anchored best fit (P020, P079) is ×2.9 higher than the ROI-anchored posterior mean (P069, P081) and predicts 2.4 events with P(0) = 0.09; the corpus quotes both as a bracket (P099).
3. **Decision rule.** N = 0 → P(DM) falls from 0.015 to < 10⁻³ by August 2027 for the elastic reading and by November 2030 for δ = 300 keV inelastic (with the present xenon generation); N = 1 in the band → P(DM) ≈ 0.025 (Bayes factor 300 against the modelled background but 0.67 against a steady LZ-specific unknown); N ≥ 2 with at least one event in 270–420 keV (extended edge) or in 55–200 keV → P(DM) = 0.57–0.99; two events both in the band → 0.011 (it is then a steady unknown).
4. **The extended sideband is the sharpest knife.** With a 1000 phd edge the same data must show 3.6/13/103 events at 600–1000 phd if δ = 350/366/380 keV and none if the reading is elastic; the already-published empty SR3 sideband gives P(0 | δ = 366 keV) = 0.02, and the untouched data push it to 1.2×10⁻⁴ (P038, P088, P099). δ ≥ 366 keV is therefore falsified by LZ's next release unless the sideband is populated.
5. **Outside LZ.** The 17.9 t·yr of XENONnT, PandaX-4T and LZ SR1 data already on disk contain 2.2/1.1/3.2/4.9 events (L10/δ = 300/350/366 keV) at the best fit, P(0) = 0.12/0.32/0.04/0.007; two found events are 4.4σ new-only (P069, P088, P099).

## 8. Self-assessment of the Phase-2 forecast

**What I think was wrong.**
- *The wall-MSSI and ER-tail explanations were over-weighted in the dossier* (B 0.19, C 0.13). Both collapsed under the first quantitative attack (P004, P010) and never recovered; the residual detector weight belongs to the uncharacterised-artefact class (F), which the dossier had at 0.18 and which the corpus could barely move. The forecast that the community "will regard the event as most likely a background or artefact (with wall MSSI and an unmodelled ER tail as the leading candidates)" was right about the conclusion and wrong about the candidates.
- *I expected the pure Higgsino to remain the flagship model.* It is dead as a fixed-coupling explanation: solar capture on core Fe/Ni gives annihilation rates 10²–10³ above IceCube (P076) — a constraint I had listed nowhere in the plan's COMP entries — and the higher multiplets fall to Sommerfeld-enhanced annihilation (P084). The generic pseudo-Dirac fermion with a 9–35 GeV dark photon is what survives.
- *I under-estimated how much the S1c < 600 phd edge shapes the inference.* The dossier treated δ ≈ 380 keV as the natural inelastic preference; P038 showed it is manufactured by the acceptance edge, and P079/P088 showed the single most valuable analysis LZ can do is to move the edge to 1000 phd.
- *I forecast "no comparable high-energy xenon analysis exists" as a reason exclusion claims would be premature.* True, but I missed the corollary that P069/P088 made central: ≈ 18 t·yr of XENONnT, PandaX-4T and LZ SR1 data are already on disk and should contain 2–5 events if the fit is right. The decisive test is a re-analysis, not new exposure.
- *The ±23 keV energy uncertainty and the "265 keV" reading were taken at face value in the dossier.* P009/P064 resolved the latter as a Table-S5 artefact, and P098 found the statistical resolution is 11.4 keV; the corpus has used 11 keV since P021.
- *I planned P055/P085/P092/P097 as if halo substructure could rescue or ruin the inelastic reading.* Every one of them found deficits of a few percent and phase-neutral effects; only v_esc and the tail shape matter (P018, P068). Two of the four were still worth writing, but four was too many.
- *Volume and category mix.* The forecast of ≈ 28 real papers with IDM 25 %, MODEL 12 %, EFT 12 %, BKG+RESP 6 % still looks right for the real arXiv; the deliberate re-weighting of this corpus towards BKG/RESP/STAT (34 of 100 papers) is where its most robust results came from, so the re-weighting was correct.

**What held up.** The three community demands (analyse the untouched LZ data, extend XENONnT/PandaX ROIs, release the likelihood) were exactly the corpus's conclusions; the "intriguing but not evidence" consensus; the identification of inelastic DM at δ ≈ 250–390 keV and q⁴-spin operators as the only natural fits; the statistical critique landing at 2.5–2.6σ global.

## 9. Process-level tool summary

**Coordination.** One general-purpose subagent per paper (100 agents), each following `provenance/PAPER_GUIDE.md`, launched in batches of 5–10 with the ledger updated after every batch (steps 001–NNN of `provenance/process_log.md`). The coordinator wrote the dossier, plan, shared library, ledger tooling and this synthesis, validated every page with `ledger_tools.py check`, and kept the process log, tool inventory and data-request index.

**Tool calls by the paper agents** (from the 100 provenance JSONs): Read NNNN, Bash NNNN, Write NNN, Edit NNNN, Skill (dataviz) NN, ToolSearch NN, TaskStop NN, Monitor N — about NN tool uses per paper. Bash calls were script runs, greps of the LZ source, word counts and JSON checks; no network, no installs, no interpreter other than `.venv/bin/python` (one deviation: `node` executed by the dataviz palette validator in P046/P047 before it was banned).

**Software, and which kinds of papers used it** (papers out of 100):
- numpy, matplotlib and the shared `common/lzcommon.py` in every paper; scipy in ≈ 95 (integration, ODEs, special functions, statistics, clustering); pandas in ≈ 87.
- WimPyDD 2.0.4 in ≈ 64 papers — all IDM, EFT, HALO and XEXP papers, most PROJ/MODEL/STAT papers that needed spectra (shell-model nuclear responses, NREFT Hamiltonians with q-dependent closures, per-isotope and per-stream kernels, inelastic and exothermic kinematics, WimPyC solar capture). Its coupling convention (c⁰ = c_p + c_n) was settled by P003 against the digitised Fig. 1, and three pitfalls were found and documented (default v_min grid truncation, Sun-frame vs annual halo, shared lambda defaults).
- nestpy 2.1.1 (LZ_WS2024 with the paper's Table S3–S5 parameters) in ≈ 21 papers — all RESP and most BKG/NUC papers (yields, band widths, quanta fluctuations, energy reconstruction).
- Digitisation: PyMuPDF 1.28.2 (vector paths of Figs. 1, 3, 6, S7) in 7 papers and Pillow 12.3.0 (pixel masks of Figs. 2, 4, 5, S2–S6) in 7, mostly BKG/STAT.
- Specialist packages: iminuit (P052 profile likelihood), wimprates (halo cross-checks), astropy (ephemerides; AltAz unusable offline), radioactivedecay (²¹⁴Pb/²³⁸U chains), periodictable, healpy (sky maps), directdm (P031; 5/4-flavour classes crash under numpy 2.5), uncertainties, sympy, numericalunits — each in 1–4 papers.
- Hand derivations were the primary tool in ≈ 50 papers (relic formulae, NR reductions, kinematics, Fisher/KL statistics).

**Recalled knowledge.** About 1 150 items were flagged across the corpus with reliability labels (roughly 45 % certain, 35 % likely, 20 % uncertain), catalogued as R01–R98 in `provenance/tool_inventory.md`. The most consequential uncertain items are external experimental limits (IceCube/ANTARES solar, H.E.S.S./Fermi/AMS, dijet Z′ ceilings, archival ROI edges) and LZ geometry/light-yield numbers not in the paper.

**Unavailable tools and workarounds.** No internet (all external numbers recalled and flagged; three data requests filed instead); no GEANT4 (photon and neutron transport written as weighted ray-march Monte Carlos in numpy: P033, P042, P063, P073, P077); no NEST beyond nestpy (ER skew model reconstructed in P056); `ps`/`pgrep`/`timeout` denied or absent; harness limits (600 s foreground, 20 concurrent agents, stream watchdog) shaped the batch cadence and forced caching of WimPyDD kernels in 14 papers.

## 10. Data-request summary (ranked by priority)

| Rank | ID | Requested by | Priority | Content | Papers and results affected | Status |
|---|---|---|---|---|---|---|
| 1 | DR-002 | P052 | important | LZ Data Release: science-sample event list (S1c, log₁₀S2c, position), per-component background and L10/O1 signal PDFs in {S1c, log₁₀S2c} or finer Fig. 5 S1c slices, NR-band median/width, toy q₀/p₀ distributions | P052 (toy-calibrated 3.0–3.2σ could move to LZ's 3.4σ; the −0.5σ isoscalar-inelastic offset); P093 (within-panel MSSI/accidental shapes); P090 (MSSI shape systematic); P016, P008, P071 would re-anchor their few-bin likelihoods | pending |
| 2 | DR-001 | P012 (also P045, P068) | important | LZ Data Release: L10ˢ signal normalisation (events per unit d₁₀² vs mass) and the two-sided 90 % CL interval tables of Fig. 6 | P012 (d₁₀ = 0.28 could change ×2), P045 (identification of LZ's lower-edge construction), P068 (d₁₀ scale of the astrophysical band), P020/P088 (L10 event-rate predictions inherit the scale) | pending |
| 3 | DR-003 | P093 | important | Waveform-level quantities of the candidate (summed S1/S2 waveforms, per-PMT areas, S1 prompt fraction/rise time, top–bottom asymmetry, S2 width, drift time) and the Waveform-Analysis templates | P093 (NR:ER and NR:artefact odds ±0.5 dex — the two smallest margins in the corpus), P077 (PSD likelihood ratio), P041 (artefact class), P022 (accidental S2 width) | pending; may not be publicly released |

No dataset was fetched or used; every result in the corpus rests on the LZ paper's text, tables and figures, the corpus's own outputs, and flagged recalled knowledge. P060 considered and deliberately did not file a GCN/SNEWS/GOES request (a Galactic supernova or flare coincidence cannot change its conclusion).
