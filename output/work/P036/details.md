# P036 — Could the LZ event be a nuclear excitation? Research record

Simulated date 2026-09-09. Category NUC (nucl-th, cross-list hep-ph). Author profile: nuclear theorists working on DM–nucleus inelastic transitions. Script: `output/code/P036_nuclear_excitation.py` (run log `run_log.txt`, results `P036_results.json`).

## 1. Motivation and framework

Spin-dependent (SD) dark-matter scattering can leave the odd-A xenon isotopes in their first excited states: ¹²⁹Xe (ground 1/2⁺ → 3/2⁺ at 39.58 keV, T½ = 0.97 ns, M1, α_IC ≈ 12) and ¹³¹Xe (3/2⁺ → 1/2⁺ at 80.19 keV, T½ = 0.48 ns, M1+E2, α_IC ≈ 1.6) [recalled, ENSDF; energies/half-lives certain, conversion coefficients uncertain]. The de-excitation (IC electron + x-rays, or the γ) is prompt on the S1 time scale and deposits E* at the scattering vertex as an electron recoil (ER). The observable is therefore a *hybrid*: an NR of energy E_R plus an ER of energy E*. This channel was studied theoretically by Baudis, Kessler, Klos, Lang, Menéndez, Reichard and Schwenk (2013) and by McCabe (2016), and searched for by XENON100 (2017) and XENON1T (2021) [recalled, likely]. LZ's 2609.02823 does not consider it. We ask:

(a) where hybrids fall in (S1c, log₁₀S2c) and whether one can mimic the event at (540.1 phd, 9268 phd);
(b) the expected excitation-to-elastic SD rate ratio for a 1 TeV WIMP;
(c) whether the event is a lower-energy elastic NR plus a 39.6 keV IC electron;
(d) what a hybrid population would look like in LZ.

**Hybrid response model.** Quanta add at a common vertex: S1c = g₁(N_ph,NR + N_ph,ER), S2c = g₂(N_e,NR + N_e,ER), with g₁ = 0.110 phd/photon, g₂ = 34.5 phd/electron (LZ paper). NR yields: LZ-tuned NEST (Table S5 incl. the p(E) break above 74.7 keV) via `lz.nest_nr_yields`; ER yields: LZ Table S3 β model via `lz.nest_er_yields(params=lz.NEST_ER_LZ)`. Recombination of the two clouds is treated independently (the NR track ~100 nm and the 40 keV electron track ~10–30 μm are spatially separate; cross-recombination would only move electrons of the ER cloud into photons and cannot reduce the ER charge below the NR-like fraction — see §7). Band widths: σ_NR(log₁₀S2c at 540 phd) = 0.0313 dex (P024 baseline; P009 0.033); σ_ER ≈ 0.053 dex, from P010's calibrated σ_Ne = 150 e at N_e = 1226 (150/1226/ln10) — approximate, used only to place hybrids relative to the ER band.

**Rate model.** For a heavy WIMP the excitation rate is the elastic SD (O₄) rate with (i) the elastic SD structure factor replaced by the inelastic one, S_inel(q) = R(q) S_el(q), and (ii) inelastic kinematics with threshold E*: v_min = (m_N E_R/μ + E*)/√(2 m_N E_R), identical to inelastic-DM kinematics with δ = E* (the excitation energy is drawn from the relative kinetic energy; the change of the nuclear mass by E*/m_N ≈ 3×10⁻⁴ is negligible). WimPyDD (`lz.wd_rate`, `delta_kev=E*`, `isotopes_list` per isotope) gives (ii) exactly with the DMFormFactor-v6 shell-model responses LZ uses; R is *not* in the corpus and is recalled with a bracket (§4).

## 2. Inputs

| input | value | source |
|---|---|---|
| g₁, g₂, event S1c/S2c, ROI (3–600 phd, log₁₀S2c 2.75–4.15), exposure 2.84 t·yr, efficiency plateau 0.96 (5.4–269.9 keV) | see `lz.LZ` | LZ paper (Data Analysis, Fig. 1 caption) |
| NEST NR Table S5 (with break), ER Table S3 | `lz.NEST_NR_LZ`, `lz.NEST_ER_LZ` | LZ supplement |
| σ_NR = 0.0313 dex at 540 phd; NR median 4.015 (paper scale) / 4.027 (Table S5 as printed) | P024, P009, P013 | corpus |
| σ_Ne(ER) = 150 e at 1226 e | P010 | corpus |
| 200–270 keV efficiency 0.93 | P009 | corpus |
| Baxter-2021 SHM, 16 June v_E = 266 km/s, v_max = 809 km/s | `lz.wd_halo`, `lz.v_earth_kms(167)` | library |
| WimPyDD coupling convention c⁰ = c_p + c_n; LZ unit coupling ↔ c⁰ = 2/m_v² | `lz.wd_c_from_anand` | P003/P007 (settled) |
| ¹²⁹Xe 39.58 keV (0.97 ns), ¹³¹Xe 80.19 keV (0.48 ns); α_IC ≈ 12 / 1.6 | recalled | ENSDF (certain / uncertain for α) |
| ⟨S_n⟩ = 0.329 (¹²⁹Xe), −0.272 (¹³¹Xe), 1-body | recalled, likely | Menéndez, Gazit, Schwenk 2012 |
| R = S_inel/S_el | 0.01–1, central 0.1 | recalled, **uncertain** (Baudis et al. 2013 computed it; our memory: order 0.1 at low q, rising towards ~1 at q ≈ 200–250 MeV where the elastic response has fallen) |
| LZ SD-neutron limit at 1 TeV ≈ 4×10⁻⁴¹ cm² (≈2×10⁻⁴² at 40 GeV, ∝ m_χ above 100 GeV) | recalled, uncertain (×2) | LZ 2024 WS result |
| σ_n^SD = 3 c_4² μ_n²/(16π) for a spin-½ WIMP | derived (spin traces: (1/4)Σ|⟨S_χ·S_N⟩|² = 3/16), consistent with WimPyDD's O₁ convention σ = c² μ²/π | — |
| μ/ρ in Xe: 20 (39.6 keV, above the 34.56 keV K edge), 3.5 (80.2 keV), 8 cm²/g (29.7 keV Kα); ρ = 2.9 g/cm³ | recalled, uncertain ×1.5 | XCOM-type tables |
| CSDA range of a 40 keV electron in LXe ≈ 10–30 μm | recalled, uncertain | ESTAR-type tables |

## 3. Results — hybrid loci (script §1; `hybrid_loci.csv`, `two_equation_scan.csv`, Fig. 1)

**Event quanta.** N_ph = 540.1/0.110 = 4910, N_e = 9268/34.5 = 268.6.

**Pure NR (Table S5 as printed).** S1c = 540.1 ↔ E_NR = 266 keV, NR median log₁₀S2c = 4.027; event −0.060 dex = −1.9σ (σ = 0.0313). P024's paper-scale band (S1/keV +8%) gives median 4.015 and −1.54σ, the paper's 1.5σ; P013 quoted −1.8σ with σ = 0.033. The scale ambiguity is irrelevant below.

**ER yields of the de-excitation (Table S3).** 39.58 keV: N_ph = 2243, N_e = 702 → alone (247 phd, log₁₀S2c 4.384); electron fraction 0.238. 80.19 keV: N_ph = 4497, N_e = 1470 → alone (495 phd, 4.705); fraction 0.246. NEST-default ER parameters give 2192/753 and 4363/1603 (≤9% difference).

**Loci** (E_NR = 10–320 keV): the 39.6 keV hybrid runs from (247, 4.38) to (915, 4.55); the 80.2 keV hybrid from (495, 4.70) to (1163, 4.79). **No point of either locus lies inside the ROI** (all have log₁₀S2c > 4.15): the ER charge alone (702 or 1470 e) exceeds g₂⁻¹ × 10^4.15 = 409 e. Hybrids populate the region above the ROI ceiling, between the NR and ER bands (¹²⁹Xe*) or on the ER band (¹³¹Xe*).

| case | S1c [phd] | log₁₀S2c | comment |
|---|---|---|---|
| pure NR median at 540.1 | 540.1 | 4.027 | E_NR = 266 keV; event −1.9σ (−1.5σ on the paper scale) |
| NR + 39.58 keV at S1c = 540.1 | 540.1 | 4.520 | E_NR = 157 keV; **+15.8σ** above NR median (+15.2σ with quanta ×1.077); −4.6σ_ER below ER median (4.766); predicted S2c = 3.57 × observed |
| NR + 80.19 keV at S1c = 540.1 | 540.1 | 4.740 | E_NR = 34 keV; **+22.8σ**; −0.5σ_ER; S2c = 5.93 × observed |
| 248 keV NR + 39.58 keV | 745 | 4.539 | outside ROI (P013: 745, 4.54) |
| 248 keV NR + 80.19 keV | 992 | 4.786 | outside ROI |

**Two-equation solve.** Unknowns (E_NR, E_ER): fix S1c = 540.1 by E_NR for each E_ER and compute the log₁₀S2c residual. Residual = +0.060 dex at E_ER = 0 (Table S5; +0.048 on the paper scale), +0.138 (1 keV), +0.234 (5 keV), +0.308 (10 keV), +0.553 (39.58 keV), +0.773 (80.19 keV); slope near zero +0.0222 dex/keV. The residual is positive and monotonically increasing: the two equations have a solution only at the formal value **E_ER = −2.7 keV** (−2.2 keV on the paper scale), i.e. the event needs *less* charge than a pure NR, whereas any ER admixture adds charge. Matching S2c alone requires N_e,NR = 268.6 − N_e,ER = −433 e (¹²⁹Xe*) or −1201 e (¹³¹Xe*).

Answer to (a) and (c): no NR energy exists for which a hybrid reproduces the event; a hybrid at the event's S1c would have S2c 3.6–5.9× larger and sit 16–23σ above the NR median. The S1-matching hybrids would have E_NR = 157 keV (¹²⁹Xe*) or 34 keV (¹³¹Xe*).

## 4. Results — rates (script §2; `O4_spectra_unit_coupling.csv`, `hybrid_counts_one_event_normalisation.csv`, Fig. 2)

WimPyDD Xe isotope indices: 124:0, 126:1, 128:2, 129:3, 130:4, 131:5, 132:6, 134:7, 136:8. O₄^s at LZ unit coupling (c⁰ = 2/m_v²), 1 TeV, annual-average Baxter halo; E_R grid 1–420 keV; counts for 2.84 t·yr with efficiency 0.96 (ROI) / 0.93 (200–270 keV).

| quantity | ¹²⁹Xe | ¹³¹Xe | natural Xe |
|---|---|---|---|
| elastic, all E_R | 1533 | 1047 | — |
| elastic, ROI 5.4–269.9 keV | 1073 | 874 | 1947 |
| elastic, 200–270 keV | 22.2 | 25.2 | 47.4 (shares 0.47/0.53) |
| threshold-shifted (δ = E*, R = 1), all E_R | 654.6 | 266.4 | — |
| **f_kin = inel/el (R = 1)** | **0.427** (June 0.451) | **0.255** (June 0.276) | weighted 0.357 |
| spectrum mean / median [keV], elastic | 26.6 / 12.6 | 63.3 / 42.7 | |
| spectrum mean / median / peak [keV], excitation | 38.3 / 19.8 / 10 | 94.6 / 77.4 / 44 | |
| fraction above 200 keV: elastic → excitation | 0.023 → 0.037 | 0.043 → 0.074 | |

Low-energy companions of elastic O₄^s: 29.8 events in 5.4–55 keV per 200–270 keV event (P003: 28; agreement 6%).

**Excitation/elastic SD rate ratio** (natural Xe, all recoil energies, 1 TeV): R × 0.357, i.e. 0.0036 (R = 0.01), 0.036 (0.1), 0.107 (0.3), 0.36 (1). The kinematic factor is modest for a TeV WIMP because μ v²/2 ≈ 100 keV-scale energies are available; the threshold matters mainly below 20–50 keV (Fig. 2).

**(a) Coupling for one elastic O₄^s event in 200–270 keV**: (c₄ m_v²)² = 0.0211, equivalent σ_n^SD (c_p = c_n) = 1.17×10⁻⁴⁰ cm². Hybrid events in 2.84 t·yr = 0.0211 × R × (654.6 + 266.4):

| R | ¹²⁹Xe* | ¹³¹Xe* | total hybrids | elastic 5.4–55 keV companions |
|---|---|---|---|---|
| 0.01 | 0.14 | 0.06 | 0.19 | 29.8 |
| 0.03 | 0.41 | 0.17 | 0.58 | 29.8 |
| 0.1 | 1.38 | 0.56 | 1.94 | 29.8 |
| 0.3 | 4.14 | 1.69 | 5.83 | 29.8 |
| 1 | 13.8 | 5.6 | 19.4 | 29.8 |

**(b) Recalled LZ SD-neutron limit at 1 TeV, σ_n = 4×10⁻⁴¹ cm²** → c_n = 1.40×10⁻⁶ GeV⁻² (WimPyDD c⁰ = c_n, c¹ = −c_n): elastic O₄ ROI rate 4.69 /t/yr (200–270 keV 0.112 /t/yr); excitation rate at R = 1: 1.56 (¹²⁹Xe*) + 0.65 (¹³¹Xe*) /t/yr; at R = 0.1: 0.22 /t/yr = 0.63 hybrids in 2.84 t·yr. The elastic ROI rate of 4.7 /t/yr at a "limit" coupling is ~4× more than a 90% limit should allow in LZ's 2024 exposure, so the recalled limit is probably a factor of a few too weak or the SD efficiency differs; treat (b) as order-of-magnitude.

**Normalisation checks at q → 0.** Analytic SD formula dR/dE(0) = n_T (ρ/m_χ) σ_A m_A/(2μ_A²) η(0) with σ_A = σ_n (μ_A/μ_n)² (4/3)(J+1)/J ⟨S_n⟩²: 0.432 (¹²⁹Xe) and 0.131 (¹³¹Xe) /t/yr/keV vs WimPyDD 0.226 and 0.066 — ratios 0.52 and 0.50. A convention-free test, O₄/O₁ for neutron-only couplings at q → 0, should equal (J+1)/(4J)⟨S_n⟩²/N²: WimPyDD gives 8.14×10⁻⁶ vs 1.44×10⁻⁵ (¹²⁹Xe) and 2.76×10⁻⁶ vs 5.20×10⁻⁶ (¹³¹Xe), i.e. the shipped density matrices carry |⟨S_n⟩|_eff = 0.247 and 0.198 — 0.75 and 0.73 of the Menéndez 1-body values, with the ¹²⁹/¹³¹ ratio 1.25 vs 1.21 (P017 found the same 4% ratio agreement). The factor ≈0.55 in rate is consistent with the two-body-current quenching LZ 2023 applied [recalled 10–30% in amplitude, P017]; it is a nuclear-structure normalisation, not a coupling-convention error, and affects only the absolute conversion in (b) (×2), not the ratios in (a).

## 5. Results — kinematics (script §3)

v_E(16 June) = 266 km/s, v_max = 809 km/s (Baxter SHM, `lzcommon`).

| | ¹²⁹Xe, E* = 39.58 keV | ¹³¹Xe, E* = 80.19 keV |
|---|---|---|
| v_min(248 keV, 1 TeV): elastic → with threshold | 341 → 390 km/s | 339 → 437 km/s |
| recoil window at v_max, 1 TeV | 0.9–1324 keV (elastic max 1395) | 4.0–1265 keV (elastic max 1412) |
| minimum WIMP mass for any excitation (μ v_max²/2 = E*) | 11.9 GeV (E_R at threshold 3.6 keV) | 26.9 GeV (14.5 keV) |
| m_χ,min for 248 keV + E* | 80 GeV (elastic 73) | 90 GeV |
| m_χ,min for the S1-matching hybrid (157 / 34 keV + E*) | 58 GeV | 31 GeV |

The threshold reshapes the spectrum (Fig. 2): the excitation spectrum vanishes at E_R → 0, peaks at 10 keV (¹²⁹Xe*) / 44 keV (¹³¹Xe*) and has a 1.4–1.7× larger fraction above 200 keV than the elastic one, but for m_χ ≫ m_N the upper edge is essentially unchanged (E₊ shifts by ≈ −E*·(1 + m_N/μ)/2 ≈ −70/−150 keV out of 1400).

## 6. Results — what a hybrid looks like (script §4)

- **Containment.** Mean free paths in LXe (ρ = 2.9): 39.6 keV γ 0.17 mm (P(>5 mm) = 2.5×10⁻¹³), 29.7 keV Kα x-ray 0.43 mm (9×10⁻⁶), 80.2 keV γ 0.99 mm (6×10⁻³). IC electrons (branch α/(1+α) = 0.92 for ¹²⁹Xe*, 0.62 for ¹³¹Xe*) travel ≲ 30 μm. A hybrid is a single-site event within LZ's ~mm S2 resolution; it would not be flagged as MSSI or multiple-scatter.
- **Timing.** T½ = 0.97 / 0.48 ns ≪ the 21–28 ns triplet lifetime and LZ's S1 pulse: the two S1 components are unresolved.
- **S1 pulse shape.** At S1c = 540 the ER part supplies 46% (¹²⁹Xe*) or 92% (¹³¹Xe*) of the photons; the singlet/triplet mix would be intermediate-to-ER-like. LZ states PSD is inconclusive at 550 phd, so this handle is weak in practice.
- **S2/S1.** The hybrid's charge fraction is set by the ER cloud: N_e/N_q = 0.24 for the ER part vs 0.055 for the event (268.6/(4910+268.6)). The event's charge-poor S2/S1 is what excludes a hybrid — a hybrid can only be *charge-rich* relative to an NR.
- **Where a population would appear.** All hybrids lie above the WS ROI (log₁₀S2c > 4.15), i.e. outside the analysis LZ performed. In (S1c, log₁₀S2c) the ¹²⁹Xe* branch forms a distinct band 0.2–0.25 dex below the ER median (−4 to −5σ_ER) running from (247, 4.38) to (915, 4.55); the ¹³¹Xe* branch merges with the ER band. In ER-equivalent energy the ¹²⁹Xe* branch spans ≈ 45–130 keVee, where LZ's β/γ continuum (²¹⁴Pb, ¹³⁶Xe 2νββ, ⁸⁵Kr…) contributes O(10³) events per t·yr per 100 keVee within the ER band, so a search would rely on the ¹²⁹Xe* branch sitting several σ_ER below the ER median (a "leakage-like" locus) and on the 39.6 keV line structure in the ER-part energy. With ≈2 hybrids per 2.84 t·yr at the one-event O₄ coupling and R = 0.1 this is not a promising confirmation channel for LZ; it is ≈10 hybrids per 2.84 t·yr only for R ≈ 0.5.

## 7. Robustness and failed approaches

- Table S5 as printed vs P024 paper scale: pure-NR median 4.027 vs 4.015; the hybrid offsets change from +15.8/+22.8σ to +15.2/+22.1σ (variant with all NR quanta ×1.077, `variant_paper_scale` in the JSON — a crude proxy; P024's proper rescaling moves S1 not S2).
- NEST-default vs LZ ER yields: ≤9% in N_e; the ER charge (702–753 e) always exceeds the total observed 268.6 e.
- Cross-recombination between the NR and ER clouds could only convert ER electrons into photons; to bring the ER charge from 702 e down to the ≤ 269 − N_e,NR available would require recombination ≥ 0.96 of the ER cloud at S1c-matched E_NR — the same unphysical ≈0.94–0.95 recombination that P010 excluded for a pure ER; we did not model it further.
- Halo: June vs annual-average changes f_kin by +6–8%; ROI efficiencies of 0.96/0.93 change counts by ≤ 4%.
- R(q): applied as q-independent. If R rises with q (elastic response falling faster), the excitation spectrum is harder than shown and the fraction above 200 keV larger; the conclusion for the event is unaffected because it rests on the ER charge, not on rates.
- Abandoned: a first attempt to obtain the minimum WIMP mass for any excitation by bisection at fixed E_R = 0.5 keV returned ∞ (that E_R is below the threshold recoil E* μ/m_N); replaced by the analytic condition μ v_max²/2 ≥ E*.

## 8. Figures

- `figures/P036_fig1_hybrid_loci.png` — (S1c, log₁₀S2c): NR and ER medians with 10–90% bands (σ 0.0313 / 0.053 dex), the ¹²⁹Xe* and ¹³¹Xe* hybrid loci for E_NR = 10–320 keV (dots at 50, 150, 248 keV), the event and the ROI edges. Hybrids lie entirely above log₁₀S2c = 4.15.
- `figures/P036_fig2_spectra.png` — O₄^s unit-coupling spectra for ¹²⁹Xe and ¹³¹Xe, elastic vs threshold-shifted (R = 1), 1 TeV; the 200–270 keV window shaded.
- `figures/P036_fig3_two_equation.png` — residual (log₁₀S2c predicted − observed)/σ_NR at fixed S1c = 540.1 as a function of the ER energy added at the vertex; positive everywhere, root at −2.7 keV.

## 9. References

1. LZ Collaboration, arXiv:2609.02823 (2026).
2. L. Baudis, G. Kessler, P. Klos, R. F. Lang, J. Menéndez, S. Reichard, A. Schwenk, Phys. Rev. D 88, 115014 (2013) — inelastic SD structure factors for ¹²⁹Xe and ¹³¹Xe.
3. C. McCabe, JCAP 05 (2016) 033 — prospects for inelastic xenon transitions.
4. J. Menéndez, D. Gazit, A. Schwenk, Phys. Rev. D 86, 103511 (2012); P. Klos, J. Menéndez, D. Gazit, A. Schwenk, Phys. Rev. D 88, 083516 (2013).
5. XENON Collaboration, Phys. Rev. D 96, 022008 (2017); Phys. Rev. D 103, 063028 (2021) — searches for ¹²⁹Xe inelastic scattering [recalled, likely].
6. A. L. Fitzpatrick, W. Haxton, E. Katz, N. Lubbers, Y. Xu, JCAP 02 (2013) 004.
7. Corpus: P003, P009, P010, P013, P017, P019, P024.

## 10. Tools and provenance (mirrors `provenance/P036.json`)

- Agent tools: Read ×18 (PAPER_GUIDE; dossier; ledger ×2 pages; P013/P019/P017/P009/P024.md; tex l.35-70, 684-697, 820-911; lzcommon.py; P013.json; own figures ×4 views), Bash ×17 (grep tex; lzcommon API; versions/ls; P013 hybrid code; WimPyDD signature; P017 isotope bookkeeping; P024 baseline; script runs ×3; log/JSON inspection ×2; word counts ×4), Write ×4 (script, details.md, P036.json, P036.md), Edit ×10 (script ×4: threshold mass, q→0 check, O4/O1 check, figure label; details ×1; paper ×5 trims; JSON ×1), Skill ×1 (dataviz).
- Software: python 3.12.13; numpy 2.5.3; scipy 1.18.1 (via lzcommon); pandas 3.0.5; matplotlib 3.11.2 (Agg); nestpy 2.1.1 (LZ_WS2024 GetYields NR/beta via lzcommon); WimPyDD 2.0.4 (Xe target, eft_hamiltonian, streamed_halo_function with explicit v_min grid, diff_rate with delta and isotopes_list); common/lzcommon.py.
- Script/command: `.venv/bin/python output/code/P036_nuclear_excitation.py` (17 s).
- WimPyDD-generated files: none (diff_rate only; checked with `find WimPyDD -newer`).
- Recalled knowledge: 9 items listed in §2 with reliabilities.
- Data requests: none. Datasets: none.
