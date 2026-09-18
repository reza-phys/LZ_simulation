# P095 — Migdal electrons and bremsstrahlung photons accompanying a 248 keV xenon recoil

Simulated date 2026-09-16 · category NUC (RESP aspects) · hep-ph (cross-list physics.ins-det) · author profile: atomic/nuclear theory group.
Script: `output/code/P095_migdal_brems.py` (run from the root with `.venv/bin/python`; 4 s; fully analytic, no random numbers).
Machine-readable results: `output/work/P095/P095_results.json`; tables `P095_migdal_shells.csv` (17 subshells), `P095_migdal_groups.csv` (K–O),
`P095_shift_table.csv`, `P095_mechanisms.csv`; console log `run_log.txt`; figure `figures/P095_fig1_migdal_shift.png`.
All numbers below are produced by the script unless marked [paper], [corpus PXXX] or [recalled].

## 1. Motivation and framework

LZ's candidate (S1c = 540.1 phd, S2c = 9268 phd) sits 1.5σ *below* the NR-band median in log₁₀S2c [paper; P009: 1.46σ; P024: 1.54σ, P(≤ event | NR) = 5.7 %].
P036 showed that nuclear-excitation hybrids (NR + 39.6/80.2 keV ER at one vertex) land 16–23σ *above* the median. Two further prompt companions of any
nuclear recoil have not been treated in the corpus: (a) Migdal ionisation/excitation of the recoiling atom's own electrons (Ibe, Nakano, Shoji, Suzuki 2018),
and (b) nuclear bremsstrahlung from the suddenly accelerated, partially screened nucleus (Kouvaris & Pradler 2017). We ask: with what probability does a
248 keV Xe recoil carry such a companion, how much electron-recoil (ER) energy does it deposit, where does that move the event in (S1c, log₁₀S2c) in units of
the NR-band σ, and could either mechanism be relevant to the observed −1.5σ position or to the NR signal acceptance? We also (c) clarify what NEST/Lindhard
quenching already contains, and (e) list the mechanisms that make an NR *charge-poor*.

## 2. Kinematics of the recoiling atom (script §1)

M_Xe = 131.293 u = 122.30 GeV [recalled constants, certain]. Recoil speed v_A = √(2E_R/M) = 2.014×10⁻³ c = **604 km/s**.
Migdal momentum parameter q_e = m_e v_A = **1029 eV/c**, i.e. **q_e = 0.276 a.u.**, q_e² = 0.0762 a.u. (a.u.: ħ = m_e = e = 1, velocity unit αc).

Suddenness. The DM–nucleus momentum transfer lasts τ ≈ R_N/v_rel = 6.1 fm / 500 km/s = 1.2×10⁻²⁰ s → ħ/τ = 54 keV: the kick is sudden for the L, M, N, O
shells (E_B ≤ 5.5 keV) and marginal for the K shell (34.6 keV; our K number is therefore an upper estimate). In the subsequent atomic cascade, each
Xe–Xe collision transfers momentum over τ ≈ a_TF/v_A with a_TF = 0.8853 a₀/(2√54)^{2/3} = 0.078 Å [recalled Firsov screening length, likely] → 1.3×10⁻¹⁷ s,
ħ/τ = 51 eV: cascade collisions are adiabatic for the M shell and deeper (E_B τ/ħ = 13–680) and cannot create inner-shell holes with appreciable
probability. Inner-shell Migdal companions are therefore a property of the *primary* DM kick only.

Adiabaticity per shell (q_e/v_shell with v_shell = √(2E_B)): K 0.005, L 0.014, M 0.03, N 0.07–0.12, O 0.2–0.3.

## 3. Migdal probabilities (script §2)

### 3.1 Formalism
In the recoiling atom's frame the electrons receive a boost; the sudden-approximation amplitude is ⟨f| exp(−i q_e·Σ_j r_j) |i⟩ (Migdal; Ibe et al. 2018).
For q_e r ≪ 1 (dipole limit) P_f = q_e² |⟨f|Σ_j z_j|i⟩|² [certain]. Two consequences we use:

(A) **Closure sum rule.** Σ_{f≠i} P_f = q_e² (⟨i|(Σz)²|i⟩ − |⟨i|Σz|i⟩|²) = q_e² Σ_electrons ⟨z²⟩_nl (one-electron orbitals, definite parity), i.e. the
total excitation + ionisation probability of an orbital is q_e² ⟨z²⟩ = q_e² ⟨r²⟩/3. We evaluate ⟨r²⟩ hydrogenically, ⟨r²⟩ = n²[5n² + 1 − 3l(l+1)]/(2Z_eff²),
with Slater's-rules Z_eff (computed in the script from the rules [recalled, certain]) as the baseline and Z_eff = n√(E_B/13.6 eV) as a variant (larger radii,
×2–7 larger probabilities; listed in `P095_migdal_shells.csv` as `PA_*_binding`). The *ionisation* part is the continuum share of ⟨z²⟩; for hydrogenic 1s we
compute it exactly (§3.2): 0.283, and apply the same share to all shells (approximation; bound transitions to empty levels also create a hole, see §3.4).

(B) **Oscillator-strength (photoabsorption) route.** With f_{if} = 2ω_{fi}|⟨f|Σz|i⟩|², dP/dω = q_e² (df/dω)/(2ω) and, since σ_ph(ω) = 2π²α df/dω,
dP/dE_e = q_e² σ_ph(E_e + E_B)/(4π²α (E_e + E_B)) — the Migdal spectrum is the shell's photoabsorption cross-section divided by the photon energy
(relation stated by Essig, Pradler, Sholapurkar & Yu 2020 [recalled, likely]; derived here independently). For a continuum σ ∝ ω^{−p} with total continuum
oscillator strength N f_c this integrates to

  P_ion(nl) = q_e² N_nl f_c (p − 1) / (2 p E_B)      (a.u.),

depending only on the binding energy, occupancy, continuum share f_c and slope p. We use f_c = 0.435 and p = 2.7 from the hydrogenic 1s continuum
(§3.2; p_eff fitted between threshold and 3E_B is 2.81). Mean ejected-electron energy ⟨E_e⟩ = E_B/(p−1) = 0.59 E_B; median E_e = E_B(2^{1/p} − 1) = 0.29 E_B;
P(E_e > x) = (E_B/(E_B + x))^p.

### 3.2 Hydrogenic check (Stobbe formula)
σ_1s(ω) = (2⁹π²α/3)(I/ω)⁴ exp(−4η arccot η)/(1 − e^{−2πη}), η = √(I/(ω−I)) [recalled, likely]. Numerically: σ(threshold) = 6.30×10⁻¹⁸ cm² (textbook
6.3×10⁻¹⁸ [certain]); continuum oscillator strength 0.4350 (textbook 0.435 [certain]); continuum share of ⟨z²⟩: ∫(df/dω)/(2ω)dω = 0.283; p_eff = 2.81.
The formula is therefore reproduced to <0.1 % and both recalled anchors are recovered.
Exact hydrogenic K-shell Migdal ionisation for Xe (Z_eff = 53.7, 2 electrons): P_K = q_e² · 2 · Z_eff⁻² · 0.283 = **1.50×10⁻⁵**; route B: 1.64×10⁻⁵.

### 3.3 Inputs
Xe subshell binding energies (eV) [recalled, X-ray data booklet, likely ±2 %]: K 34561; L1 5453, L2 5107, L3 4786; M1 1149, M2 1002, M3 941, M4 689, M5 676;
N1 213, N2 147, N3 145, N4 69.5, N5 67.5; O1 23.4, O2 13.4, O3 12.1. Occupancies 2/2,2,4/2,2,4,4,6/2,2,4,4,6/2,2,4. Slater Z_eff: 1s 53.7, 2s2p 49.9,
3s3p 42.8, 3d 32.9, 4s4p 26.2, 4d 14.9, 5s5p 8.25.

### 3.4 Results per shell (per 248 keV recoil)

| shell | E_B | N_e | q_e r_rms | P_ion (A, Slater) | P_ion (B) | geometric mean | P_all (A, closure) | ⟨E_ER⟩ = E_B + ⟨E_e⟩ |
|---|---|---|---|---|---|---|---|---|
| K | 34.56 keV | 2 | 0.009 | 1.50×10⁻⁵ | 1.64×10⁻⁵ | **1.6×10⁻⁵** | 5.3×10⁻⁵ | 54.9 keV (median 44.7) |
| L | 4.79–5.45 keV | 8 | 0.030–0.036 | 7.6×10⁻⁴ | 4.5×10⁻⁴ | **5.9×10⁻⁴** | 2.7×10⁻³ | 8.0 keV |
| M | 0.68–1.15 keV | 18 | 0.087–0.094 | 1.43×10⁻² | 6.4×10⁻³ | **9.6×10⁻³** | 5.0×10⁻² | 1.26 keV |
| N | 67–213 eV | 18 | 0.26–0.42 | 0.22 | 0.056 | 0.11 | 0.76 | 0.145 keV |
| O | 12–23 eV | 8 | 1.3 | 1.28 (>1: dipole invalid) | 0.16 | 0.45 | 4.5 | 22 eV |

Routes A and B agree within ×1.1 (K), ×1.7 (L), ×2.2 (M); we quote the geometric mean with a factor-2 uncertainty. Dipole validity (q_e r_rms < 0.3) holds
for K, L, M; for N (0.26–0.42) it is marginal and for O (1.3) it fails, so the N/O numbers only say "order unity": the outer shells are shaken in essentially
every recoil, as expected for an atom moving at 0.28 a.u. (the saturating proxy N(1 − e^{−q²⟨z²⟩}) gives 0.74 for N and 3.5 for O electrons).
Bound excitations to empty levels (4f, 5d, 6p, …) also leave a hole whose cascade deposits ≈E_B; the closure value P_all is the upper bound including them
(M: 5.0 %). Pauli blocking of transitions into occupied levels is ignored in P_all, so the truth for "M-shell hole" lies between 0.6 % and 5 %.
M-shell electron spectrum: fraction of M events with E_ER > 0.9/1.5/2.5/3.5 keV = 0.66/0.20/0.050/0.020.

Comparison with Ibe et al. (2018): we did not reproduce their tabulated normalisation from memory [flagged uncertain]; their Fig. 4 rates for isolated Xe
scale as q_e² and, at q_e = 1 keV, are of the same order (10⁻² for n = 3) as our route A. Our numbers are an independent dipole-limit estimate.

### 3.5 Energy deposited and topology
The hole de-excites by Auger electrons (dominant for L, M, N, O) or X-rays (K: fluorescence yield ≈0.89 [recalled, likely], Kα 29.7 keV). The total ER
energy at the vertex is E_ER = E_e + E_B. Ranges in LXe (ρ = 2.9): electron CSDA ≈ 6×10⁻⁶ (E/keV)^{1.7} g cm⁻² [recalled ESTAR-like scaling, uncertain ×2]
→ 0.02 μm (1 keV), 0.3 μm (5 keV), 9 μm (35 keV); photon mean free paths from μ/ρ = 6000/1000/7.5 cm² g⁻¹ at 1/4.1/29.7 keV [recalled, uncertain ×1.5]
→ 0.6 μm, 3 μm, 0.46 mm. All are far below the S2 transverse diffusion σ_T ≈ 3.1 mm at 859 μs drift [P041] and below the ~1 μs S2 width in drift time
(0.46 mm ≈ 0.3 μs at 1.5 mm/μs [recalled, uncertain]). A Migdal- or bremsstrahlung-accompanied recoil is therefore a *single* merged S2 — it cannot be
misclassified as a multiple scatter, and LZ's alpha-template S2-shape check [paper, Waveform Analysis] would pass.

## 4. Nuclear bremsstrahlung (script §3)

A charge Q whose velocity changes suddenly by Δv radiates the classical dipole spectrum dI/dω = 2Q²|Δv|²/(3πc³) [certain], i.e. photon number
dN/dω = (2α/3π) (Q/e)² (Δv/c)²/ω = (4α/3π)(E_R/Mc²) Z_eff(ω)²/ω, the form of Kouvaris & Pradler (2017) with |f(ω)|² → Z_eff(ω)² [recalled form, likely].
Screening: electrons bound more tightly than ħω follow the nucleus adiabatically and cancel its charge; electrons with E_B < ħω stay behind. Hence
Z_eff(ω) = N(E_B < ħω): 0 for ω ≲ 12 eV (neutral atom — the radiation vanishes at long wavelength), 40 at 1 keV, 44 above 1.15 keV, 52 above 5.45 keV, 54 above
34.6 keV. Prefactor (2α/3π)(v_A/c)² = 6.28×10⁻⁹. Integrating piecewise:

| threshold | ω_max = 100 keV | ω_max = 248 keV |
|---|---|---|
| ω > 0.1 keV | 8.2×10⁻⁵ | 9.9×10⁻⁵ |
| **ω > 1 keV** | **7.2×10⁻⁵** | 8.8×10⁻⁵ |
| ω > 3.5 keV | 5.7×10⁻⁵ | 7.3×10⁻⁵ |
| ω > 10 keV | 4.1×10⁻⁵ | 5.7×10⁻⁵ |

A bare Z = 54 nucleus gives 8.4×10⁻⁵ (1–100 keV): screening reduces the keV-scale yield by only ≈15 %, because above 1 keV most electrons cannot follow;
the "tiny" radiation is at ω below the valence binding energies. The M-shell Migdal probability exceeds P(ω > 1 keV) by ×130, consistent with the
literature statement that Migdal dominates bremsstrahlung by orders of magnitude [recalled, likely]. Upper cut-off: the sudden spectrum is flat only for
ω ≲ ħ/τ ≈ 54 keV; the 100/248 keV brackets change the result by 20 %. Mean radiated energy (1–100 keV): 1.75 eV per recoil.

## 5. What NEST/Lindhard already contains (script §4)

LZ's N_q = 5211 quanta at 248 keV (P009 paper scale) ↔ 71.4 keV visible electronic energy; Lindhard with k = 0.166 gives L = 0.334 → 82.8 keV electronic
[P064]. This electronic energy is the integrated electronic stopping of the primary and all cascade atoms (≈10³–10⁴ collisions of ~1–10 eV electronic
transfer each) — it includes, in an averaged way, the continuous perturbation of the outer (N, O) shells of every moving atom, of which the primary
Migdal shake-off is the first instance. Mean Migdal energy per recoil: outer shells 11.7 eV, inner shells 17.6 eV, bremsstrahlung 1.8 eV → 31 eV total =
3.8×10⁻⁴ of the electronic budget. Conclusion: (i) whether or not one calls the outer-shell shake "Migdal", it is 10⁻⁴ of the yield and irrelevant;
(ii) inner-shell holes are *not* in a smooth stopping model, but their mean contribution (18 eV) is 10⁻⁴ of the yield, so the NEST means are unaffected;
they matter only as a rare (≈1 %) charge-rich tail, which the D-D/AmBe calibration samples contain but the NEST fluctuation model does not describe.
No double counting arises when adding a K/L/M hole to the NEST NR quanta; adding N/O shake-off would be double counting at the 10⁻⁴ level.

## 6. Effect in (S1c, log₁₀S2c) (script §5)

Quanta add at the vertex (P036 method): S1c = g₁(s·N_ph,NR + N_ph,ER), S2c = g₂(N_e,NR + N_e,ER), with g₁ = 0.110, g₂ = 34.5 [paper], NR yields from
`lz.nest_nr_yields` (Table S5 with p(E) break) and the photon count rescaled by s = 540.1/(g₁·4525) = 1.085 so that S1c(248 keV) = 540.1 phd (P009/P024
paper scale; N_e = 301.1 unchanged); ER yields from `lz.nest_er_yields` with LZ Table S3 β parameters. Shift = log₁₀S2c(hybrid) − NR median at the hybrid S1c
(analytic means; median at 540.1 = 4.0165 vs P024's MC 4.0152; slope d log S2/d log S1 = 0.286); σ = 0.0313 dex [P024]. On this analytic median the event
is −1.58σ (P024 MC: −1.54σ); the ROI ceiling log₁₀S2c = 4.15 is +4.26σ.

| E_ER [keV] | N_ph,ER | N_e,ER | S1c [phd] | log₁₀S2c | shift [dex] | shift [σ] | S1-matched variant [σ] | comment |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 0 | 8.5 | 540.1 | 4.029 | 0.012 | +0.39 | +0.39 | N/O scale |
| 0.5 | 0 | 38.5 | 540.1 | 4.069 | 0.052 | +1.67 | +1.67 | |
| 0.7 | 2.9 | 49.2 | 540.4 | 4.082 | 0.066 | +2.10 | +2.10 | M4/5 hole, E_e → 0 |
| 1.0 | 12.9 | 61.5 | 541.5 | 4.097 | 0.080 | +2.57 | +2.57 | M typical |
| 1.5 | 34 | 77.5 | 543.9 | 4.116 | 0.099 | +3.15 | +3.16 | |
| 2.0 | 58 | 91 | 546.5 | 4.131 | 0.113 | +3.62 | +3.63 | |
| 3.5 | 135 | 126 | 554.9 | 4.168 | 0.148 | +4.73 | +4.76 | above ROI |
| 5.0 | 216 | 156 | 563.8 | 4.198 | 0.176 | +5.63 | +5.68 | L typical |
| 10 | 498 | 247 | 594.8 | 4.276 | 0.248 | +7.92 | +8.07 | |
| 35 | 1971 | 633 | 756.9 | 4.508 | 0.450 | +14.4 | +15.1 | K shell |
| 39.58 | 2243 | 702 | 786.9 | 4.539 | 0.479 | +15.3 | +16.0 | P036 check: +15.8σ (Table S5 scale) |

Thresholds: E_ER = **0.65 keV** lifts a 248 keV NR above +2σ; **2.72 keV** lifts it above the ROI ceiling. With P024's AmBe width (0.036 dex) all σ values
scale by 0.87. The NEST β model returns zero photons below ≈0.7 keV (an artefact of the Table S3 parametrisation; the charge yield, which drives the shift,
is 60–80 e/keV there and is the quantity that matters).

Probability-weighted mean shift (shell probability × shift at the shell's mean E_ER): K 1.6×10⁻⁵ × 18.3σ = +0.0003σ; L 5.9×10⁻⁴ × 7.1σ = +0.004σ;
M 9.6×10⁻³ × 2.9σ = +0.028σ; (N 0.11 × 0.55σ = +0.06σ; O 0.45 × 0.09σ = +0.04σ, both already in the stopping budget). Inner shells: **+0.03σ**;
all shells: +0.13σ. The median shift is zero and every contribution is positive.

Out-of-band fractions for 248 keV NRs (Migdal, using the per-subshell E_e spectra): above +2σ **1.1×10⁻²** (Gaussian band tail alone 2.3×10⁻²; the two add),
above the ROI ceiling **1.0×10⁻³** (Gaussian 1×10⁻⁵). Bremsstrahlung: 7.5×10⁻⁵ and 6.0×10⁻⁵. Hence: the signal-acceptance loss from prompt companions
is ≈10⁻³ (negligible against LZ's 0.96 plateau [paper]), and a ±2σ band would lose ≈1 % of high-energy NRs to a charge-rich Migdal tail that a Gaussian
NEST band does not model. Using the closure upper bound (P_all,M = 5 %) these fractions rise to ≈4 % and ≈2×10⁻³.

The event lies at −1.58σ. No prompt electronic companion can lower log₁₀S2c: extra quanta are ER-like (charge fraction 0.6–1.0 versus 0.06 for the NR) and
raise S2c far more than S1c. The only way an ER companion could reduce charge is cross-recombination of its electrons on the NR's ion column; P036 showed
this would require r ≥ 0.96 and P010 excluded such values. Migdal and bremsstrahlung therefore have the wrong sign for the observed position.

## 7. Reverse: what makes an NR charge-poor (script §6; `P095_mechanisms.csv`)

Deficit: 0.0495 dex = 10.8 % (268.6 observed vs 301.1 model electrons).

| mechanism | probability per 248 keV NR | extra ER energy | shift [σ] | source / note |
|---|---|---|---|---|
| recombination + electron-counting fluctuation | 5.7 % | 0 | −1.5 | P024 MC (band σ 0.0313; binomial floor 0.023 dex); 4/31 AmBe points at/below the event |
| S2 position-correction residual alone (2 % rms) | 6×10⁻⁹ | 0 | −1.5 | needs a 5.7σ excursion of the 0.0087 dex term [P024]; already inside the band σ |
| electron-lifetime mis-correction at 859 μs (11 % loss) | not quantifiable here | 0 | −1.5 | needs τ_e = 3.0 ms where 5 ms assumed (3.9 if 8 ms); Kr-83m monitoring and P041's witness argument exclude 78 % loss, are untested at 10 % |
| g₂ 3 % low (global systematic) | — | 0 | −0.44 | P043; P024: g₂ ± 3 % → 1.10–1.99σ |
| ER leakage (β, EC lines) | ≤4×10⁻⁵ events in 500–600 phd | — | — | P056: Gaussian recombination model; 7×10⁻⁷ at the event |
| Migdal M / L / K hole | 9.6×10⁻³ / 5.9×10⁻⁴ / 1.6×10⁻⁵ | 1.26 / 8.0 / 55 keV | +2.9 / +7.1 / +18 | this work — wrong sign |
| Migdal N + O shells | ≈1 (0.56 in the dipole proxy) | 46 eV mean | +0.18 | this work — in the electronic stopping |
| nuclear bremsstrahlung > 1 keV | 7.2×10⁻⁵ | ≥1 keV | ≥ +2.6 | this work — wrong sign |

The charge-poor position is therefore what P024 said it is: a 6 % recombination/counting fluctuation of a normal NR (or partly a g₂/position systematic of
±0.5σ, P043), not the signature of any prompt companion.

## 8. Validation and robustness
- Stobbe formula reproduces σ_H(threshold) = 6.30×10⁻¹⁸ cm² and f_cont = 0.435 (both textbook values) — the oscillator-strength normalisation is right.
- K shell: exact hydrogenic integral (1.50×10⁻⁵) vs power-law route B (1.64×10⁻⁵): 9 % apart.
- Routes A and B agree within ×2.2 for K, L, M; binding-energy Z_eff instead of Slater raises A by ×2–7 (bracket in `P095_migdal_shells.csv`).
- Slope p = 2.7 ± 0.3 changes P_ion by ∓7 % and ⟨E_e⟩ by ∓20 %.
- Hybrid at 39.58 keV: +15.3σ (paper scale) vs P036 +15.8σ (Table S5 scale): consistent.
- Analytic NR median 4.0165 vs P024 MC 4.0152 (0.04σ).
- Brems: ω_max 100 → 248 keV: +23 %; unscreened vs screened: −15 %.

## 9. Failed or abandoned approaches
- Reproducing Ibe et al.'s tabulated dp/dE_e normalisation from memory: abandoned (not reliably recalled); replaced by the two first-principles dipole estimates.
- Per-shell photoabsorption cross-sections from recalled XCOM/Henke tables: abandoned as too uncertain (±50 %); the sum-rule normalisation (route B) is used instead,
  with the K-shell threshold cross-section (hydrogenic 4.4 kb, ≈20 cm²/g) as the only cross-check against recalled attenuation data.

## 10. Figure
`figures/P095_fig1_migdal_shift.png` — left: Migdal ionisation probability per 248 keV recoil for the K–O shells, routes A and B, with the mean E_ER labels,
the bremsstrahlung P(ω > 1 keV) line and the shaded region where the dipole limit fails; right: shift of a 248 keV NR from the NR median (in σ = 0.0313 dex)
versus the ER energy added at the vertex, with the M/L/K mean points, the +2σ and ROI-ceiling (+4.3σ) lines and the event at −1.58σ.

## 11. References
1. LZ Collaboration, arXiv:2609.02823 (2026).
2. M. Ibe, W. Nakano, Y. Shoji, K. Suzuki, "Migdal effect in dark matter direct detection experiments", JHEP 03 (2018) 194, arXiv:1707.07258.
3. C. Kouvaris, J. Pradler, "Probing sub-GeV dark matter with conventional detectors", PRL 118, 031803 (2017), arXiv:1607.01789.
4. R. Essig, J. Pradler, M. Sholapurkar, T.-T. Yu, "Relation between the Migdal effect and dark matter–electron scattering in isolated atoms and semiconductors", PRL 124, 021801 (2020).
5. H. A. Bethe, E. E. Salpeter, Quantum Mechanics of One- and Two-Electron Atoms (Springer, 1957) — Stobbe photoionisation formula, hydrogen oscillator strengths.
6. J. Lindhard et al., Mat. Fys. Medd. Dan. Vid. Selsk. 33, 10 (1963).
7. M. Szydagis et al., NEST review, arXiv:2211.10726 (2024).
8. Corpus: P009, P010, P024, P036, P041, P043, P056, P064.

## 12. Tools and provenance (mirrors provenance/P095.json)
- Agent tools: Read ×11 (PAPER_GUIDE; P009, P024, P043, P036, P056, P064 papers; lzcommon.py l. 18–75 and 256–295; figure PNG ×2), Bash ×13 (API/ledger/tex greps,
  P024 details and P036 script inspection, nestpy probes, three script runs, results dump, four word-count/JSON checks), Write ×5 (script, details, JSON, paper ×2),
  Edit ×15 (script ×6, paper ×7, JSON ×2; one failed no-op on details.md), Skill ×1 (dataviz; JS validator skipped per PAPER_GUIDE).
- Software: python 3.12.13; numpy 2.5.3; scipy 1.18.1 (integrate.quad, optimize.brentq, stats.norm); pandas 3.0.5; matplotlib 3.11.2 (Agg); nestpy 2.1.1 via
  common/lzcommon.py (nest_nr_yields, nest_er_yields, LZ, W_EV, C_KMS).
- Script: `output/code/P095_migdal_brems.py`, command `.venv/bin/python output/code/P095_migdal_brems.py`.
- Local inputs: PAPER_GUIDE.md; papers P009/P024/P043/P036/P056/P064 (+P041/P010 headlines); output/work/P024/details.md §2–4 and results.json (band σ, median, detector
  terms); output/code/P036_nuclear_excitation.py (hybrid method); lzcommon.py; LZ tex l. 93–105, 165, 690–695 (band definition, event, S2-shape/SS classification);
  results_ledger.csv header and P010/P013 rows.
- Recalled knowledge: 16 items (listed in the JSON), of which the Ibe normalisation is explicitly *not* used; reliabilities certain/likely/uncertain as marked.
- Datasets: none. Data requests: none. WimPyDD files: none.
