# P047 — Isotope by isotope: which xenon nuclei produce a 248 keV recoil for each interaction, and what isotopically modified xenon could reveal

*Research record (simulated date 2026-09-10). Category NUC, nucl-th (cross-list hep-ex). Author profile: nuclear theorists and xenon-isotope experimentalists. Script: `output/code/P047_isotopes.py` (run from the simulation root; `--fast` re-uses the cached per-isotope spectra in `output/work/P047/P047_isotope_spectra.npz`). All numbers below are copied from `output/work/P047/run_log.txt` and the CSV/JSON files listed in §9.*

## 1. Motivation and framework

LZ (arXiv:2609.02823) fits its lone 248 ± 23 ± 23 keV nuclear-recoil-like event with a whole family of spectra that are degenerate for one event (dossier §4.7): a magnetic-dipole-like L10 (transverse-spin, ¹²⁹Xe/¹³¹Xe only), the q²-suppressed spin operators O6, O10 (also odd isotopes only), O4, spin-independent O1, and inelastic O1 at δ = 300–350 keV (P021: the likelihood peaks at δ ≈ 380 keV; P007: a pure Higgsino gives one event at δ = 366 keV). A natural-xenon detector cannot break this degeneracy by counting; a **second detector with a different isotope mixture** can, because

* spin-dependent responses (Σ′, Σ″) exist only for the odd isotopes ¹²⁹Xe (J = 1/2, 26.4 %) and ¹³¹Xe (J = 3/2, 21.2 %);
* the spin-independent M response is coherent (∝ A²) but, at q ≈ 246 MeV, sits on the isotope-dependent second diffraction node (P017: 251 keV for ¹³⁶Xe to 278 keV for ¹²⁸Xe);
* inelastic kinematics near δ_max favour the heaviest isotopes (δ_max grows with the reduced mass), so ¹³⁴Xe and ¹³⁶Xe carry the rate as δ → δ_max (P030 noted the ¹³⁶Xe ceiling of 402 keV for a 270 keV recoil).

We compute all three effects with WimPyDD's per-isotope shell-model responses and translate them into rate ratios for realistic isotopically modified xenon.

## 2. Method

### 2.1 Per-isotope spectra with WimPyDD

`WD.diff_rate(xe, H, m_χ, E, vmin, Δη, delta=δ, isotopes_list={0: [i]})` returns the differential rate from isotope *i* per kg of **natural** xenon (WimPyDD `package.py` l. 467: `nt_kg = abundance·1000/average_a·N_A`, i.e. the number of nuclei of isotope *i* per kg of the natural mixture). Multiplying by 1000 × 365.25 gives events/(t·yr·keV) of natural xenon. WimPyDD ships the DMFormFactor-v6 one-body density matrices for ¹²⁸Xe, ¹²⁹Xe, ¹³⁰Xe, ¹³¹Xe, ¹³²Xe, ¹³⁴Xe, ¹³⁶Xe; **¹²⁴Xe and ¹²⁶Xe (0.095 %, 0.089 %) have identically zero responses** (P017). Their mass fraction is 0.175 % and their A²-weighted SI share would be 0.166 %; for spin operators they contribute nothing (even-even), and for inelastic scattering near δ_max they are the lightest nuclei and drop out first (δ_max(248 keV) = 369/374 keV). All natural-xenon rates quoted here are therefore low by ≤ 0.2 %, far below every other uncertainty.

Inputs: m_χ = 1000 GeV; LZ unit coupling c = 1/m_v² (Anand convention) = WimPyDD c⁰ = 2/m_v², c¹ = 0 (P003, `lz.wd_c_from_anand`); Baxter-2021 SHM via `lz.wd_halo()` (annual average, v_E = 250.6 km/s, v_max = 794.6 km/s) and `lz.wd_halo(day_of_year=167)` (16 June, v_E = 265.1 km/s, v_max = 809.1 km/s), both with the explicit v_min grid (P002 fix). Operators:

| case | WimPyDD Hamiltonian | halo |
|---|---|---|
| O1 elastic | {1: (c⁰, 0)} | annual |
| O4, O6, O10 elastic | {4/6/10: (c⁰, 0)} | annual |
| L10-like | {(4,'q2'): c⁰ q²/m_N², 6: −c⁰} (P003/P017 reduction of the magnetic tensor Lagrangian to a pure q⁴Σ′ response) | annual |
| O1 inelastic δ = 300, 350, 366, 380 keV | {1: (c⁰, 0)}, `delta=δ` | 16 June |
| O1 inelastic δ = 300, 350, 366 keV | same | annual |

Energy grid: 2 keV steps in 1–200 keV, 1 keV steps in 200–300 keV plus 5.4, 225, 248, 269.9, 271 keV (206 points); 12 cases × 7 isotopes × 206 = 17 304 `diff_rate` calls (201 s). Spectra are clipped at zero: near the M node WimPyDD's interpolated response can be very slightly negative (e.g. −0.1 % for ¹³⁶Xe at 248 keV, cf. P017's isotope-share table).

### 2.2 Windows

* **Event window** W = 225–271 keV (248 ± 23 keV), plain integral (trapezoid on a 0.05 keV grid).
* **Full ROI** 5.4–270 keV, weighted by the P003/P016/P021/P032 efficiency model ε(E) = 0.96 · ½[1 + erf((E−5.4)/(√2·3.4))] · ½[1 − erf((E−269.9)/(√2·8))].
* Point value dR/dE(248 keV) for comparison with P017.

Fractions f_i = R_i/ΣR_i per case.

### 2.3 Modified compositions

For number fractions a′_i the rate per kg is
R′(E) = Σ_i R_i(E) · (a′_i/a_i) · (⟨A⟩/⟨A′⟩), ⟨A⟩ = Σ a_i A_i = 131.388 (WimPyDD abundances, including ¹²⁴Xe/¹²⁶Xe), ⟨A′⟩ = Σ a′_i A_i.
The factor ⟨A⟩/⟨A′⟩ converts "per nucleus" to "per unit mass". Compositions (the non-fixed isotopes share the remainder in natural proportion):

| composition | ¹²⁸ | ¹²⁹ | ¹³⁰ | ¹³¹ | ¹³² | ¹³⁴ | ¹³⁶ | ⟨A′⟩ |
|---|---|---|---|---|---|---|---|---|
| natural | 0.0191 | 0.2640 | 0.0407 | 0.2123 | 0.2691 | 0.1044 | 0.0886 | 131.39 |
| ¹³⁶Xe-depleted (0.1 %) | 0.0209 | 0.2894 | 0.0446 | 0.2327 | 0.2949 | 0.1144 | 0.0010 | 130.94 |
| ¹²⁹Xe-enriched 80 % | 0.0052 | 0.8000 | 0.0111 | 0.0577 | 0.0731 | 0.0284 | 0.0241 | 129.65 |
| ¹³⁶Xe-enriched 90 % (nEXO-like) | 0.0021 | 0.0290 | 0.0045 | 0.0233 | 0.0295 | 0.0115 | 0.9000 | 135.49 |
| ¹³⁴Xe+¹³⁶Xe 90 % (natural ratio) | 0.0024 | 0.0327 | 0.0050 | 0.0263 | 0.0333 | 0.4868 | 0.4132 | 134.48 |
| even-A only (idealised: ¹²⁹, ¹³¹ → 0.1 %) | 0.0364 | 0.0010 | 0.0776 | 0.0010 | 0.5128 | 0.1989 | 0.1688 | 132.74 |

(`P047_compositions.csv`.) The ¹³⁶Xe-enriched and ¹³⁴+¹³⁶ mixtures are what centrifuge enrichment for 0νββ delivers; the ¹³⁶-depleted mixture is its tails stream; 80 % ¹²⁹Xe is the grade sold for hyperpolarised-xenon MRI; "even-A only" is a physics reference, not a product (¹²⁹Xe cannot be removed while keeping ¹²⁸Xe and ¹³⁰Xe).

### 2.4 Discrimination statistics

Two dual-phase detectors, natural and modified, equal exposure. Hypothesis H predicts (N, ρ_H N) events in the window, ρ_H = R′_H/R_H(natural). With the absolute rate unknown (LZ's ±100 % normalisation), the test is binomial on the fraction f_H = ρ_H/(1+ρ_H) of all events found in the modified detector.

* Asimov: q = 2 N_tot [f₁ ln(f₁/f₂) + (1−f₁) ln((1−f₁)/(1−f₂))], Z = √q → N_tot(3σ) = 9/(2 KL). Also with the normalisation fixed by LZ's rate: Poisson Asimov on both detectors, N_nat(3σ) = 9/(2[ρ₁ ln(ρ₁/ρ₂) − ρ₁ + ρ₂]).
* **Exact** (used in the paper): the smallest N_tot for which the median H₁ count in the modified detector has a one-sided binomial p-value ≤ 1.35 × 10⁻³ under H₂ (and the converse). Exposure per detector = N_tot/(1+ρ₁) × 2.84 t·yr, taking LZ's observed rate (1 event in 2.84 t·yr in the window) as the natural-xenon rate.

### 2.5 Inelastic kinematics per isotope

δ_max(E_R, A) = v_max √(2 m_N E_R) − m_N E_R/μ (`lz.delta_max_kev(E, 1000, A=A, v_kms=v_max)`), recoil window [E₋, E₊] from `lz.E_R_range_keV`, and the absolute ceiling δ_ceiling = μ v_max²/2 reached at E* = μ² v_max²/(2 m_N).

## 3. Results — isotope fractions (natural xenon, 1 TeV)

`P047_isotope_fractions.csv`; Fig. 1.

**225–271 keV window (plain):**

| A | abund. | A²·ab | O1 el. | O4 | O6 | O10 | L10 | δ=300 J | δ=350 J | δ=366 J | δ=380 J | δ=300 ann | δ=350 ann | δ=366 ann |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 0.019 | 0.018 | 0.026 | 0 | 0 | 0 | 0 | 0.023 | 0.018 | 0.011 | 0.000 | 0.022 | 0.013 | 0.000 |
| 129 | 0.264 | 0.255 | **0.402** | 0.510 | 0.397 | 0.400 | **0.688** | 0.369 | 0.317 | 0.241 | 0.025 | 0.364 | 0.269 | 0.042 |
| 130 | 0.041 | 0.040 | 0.044 | 0 | 0 | 0 | 0 | 0.043 | 0.040 | 0.035 | 0.010 | 0.042 | 0.037 | 0.015 |
| 131 | 0.212 | 0.211 | 0.232 | 0.490 | 0.603 | 0.600 | 0.312 | 0.236 | 0.239 | 0.239 | 0.137 | 0.236 | 0.240 | 0.177 |
| 132 | 0.269 | 0.272 | 0.198 | 0 | 0 | 0 | 0 | 0.210 | 0.229 | 0.254 | 0.252 | 0.212 | 0.245 | 0.281 |
| 134 | 0.104 | 0.109 | 0.066 | 0 | 0 | 0 | 0 | 0.078 | 0.098 | 0.128 | 0.269 | 0.080 | 0.116 | 0.240 |
| 136 | 0.089 | 0.095 | 0.031 | 0 | 0 | 0 | 0 | 0.042 | 0.060 | 0.092 | **0.307** | 0.043 | 0.078 | 0.243 |

Odd-A (¹²⁹+¹³¹) share of the window rate: O1 el. 0.634, spin operators 1.000, δ = 300/350/366/380 keV (June) 0.605/0.556/0.480/0.162. ¹³⁴+¹³⁶ share: O1 el. 0.097, δ = 300/350/366/380: 0.120/0.157/0.220/0.576.

**Full ROI 5.4–270 keV (efficiency-weighted):** O1 el. 0.019/0.261/0.041/0.212/0.271/0.106/0.091 (¹²⁸…¹³⁶) — within 0.01 of the A²·abundance pattern 0.018/0.255/0.040/0.211/0.272/0.109/0.095, as it must be when the rate is dominated by low energies where F² ≈ 1. O4 0.545/0.455, O6 0.270/0.730, O10 0.312/0.688, L10 0.428/0.572 (¹²⁹/¹³¹). Inelastic δ = 366 keV: ¹³⁶Xe 0.226, ¹³⁴Xe 0.196; δ = 380 keV: ¹³⁶Xe 0.456, ¹³⁴Xe 0.265, ¹²⁹Xe 0.014.

**dR/dE at 248 keV** (validation against P017): L10 0.700/0.300, O4 0.520/0.480, O6 0.402/0.598 (P017: 70/30, 52/48, 40/60 — identical). O1 elastic: 0.033/0.490/0.048/0.235/0.159/0.036/**0.000** — ¹³⁶Xe contributes nothing at exactly 248 keV because its own M node is at 251.1 keV (P017), whereas across the ±23 keV window it gives 3.1 %. This is why we quote window shares, not point shares.

**Reading.** (i) Spin-dependent: L10 is ¹²⁹Xe-dominated in the window (69 %) because its q⁴Σ′ kernel peaks at 233 keV for ¹²⁹Xe (P017) while the ¹³¹Xe Σ′ node at 292 keV suppresses ¹³¹Xe; O6/O10 (Σ″-type) are ¹³¹Xe-dominated (60 %); O4 is even. Over the full ROI L10 flips to ¹³¹Xe-dominated (57 %) because its low-energy shoulder is ¹³¹Xe-rich. (ii) Spin-independent: in the window the shell-model nodes redistribute the coherent A² pattern — ¹²⁹Xe rises from 25.5 % to 40.2 % (×1.58; its node is at 276 keV, above the window), ¹³⁶Xe falls from 9.5 % to 3.1 % (×0.33; node at 251 keV inside the window), ¹³²Xe from 27.2 % to 19.8 % (node 263 keV). (iii) Inelastic: the SI pattern at δ = 300 keV is close to elastic (¹³⁶Xe 4.2 %), but the kinematic weight shifts steadily to the heavy isotopes: ¹³⁶Xe 9.2 % at δ = 366 keV, 30.7 % at 380 keV.

Natural-xenon unit-coupling totals (events/(t·yr) in the window | full ROI): O1 el. 3.83×10⁴ | 5.33×10⁸; O4 9.48 | 667; O6 0.0282 | 0.184; O10 1.64 | 26.3; L10 0.0159 | 0.0727; δ = 300/350/366/380 (June) 290 | 6842, 30.6 | 223, 7.36 | 24.0, 0.546 | 0.817; annual δ = 300/350/366: 202 | 4421, 11.7 | 53.8, 0.872 | 1.57 (June/annual = 1.44, 2.6, 8.4 — consistent with P006/P020).

## 4. Results — inelastic kinematics per isotope (1 TeV)

`P047_kinematics_per_isotope.csv`, `P047_share_vs_delta.csv`; Fig. 3.

| A | δ_max(248, June) | δ_max(248, annual) | δ_max(225) | δ_max(271) | δ_ceiling (June) at E* | window at δ=366 | window at δ=380 |
|---|---|---|---|---|---|---|---|
| 124 | 369.4 | 357.7 | 364.3 | 373.0 | 377.1 @ 338 keV | 232–464 | forbidden |
| 126 | 374.1 | 362.4 | 368.9 | 377.9 | 382.6 @ 342 | 215–500 | 289–401 |
| 128 | 378.8 | 367.0 | 373.3 | 382.8 | 388.0 @ 347 | 201–531 | 254–453 |
| 129 | 381.1 | 369.2 | 375.6 | 385.2 | 390.7 @ 349 | 195–546 | 243–474 |
| 130 | 383.4 | 371.5 | 377.8 | 387.6 | 393.4 @ 351 | 190–561 | 233–492 |
| 131 | 385.7 | 373.8 | 380.0 | 390.0 | 396.1 @ 353 | 185–575 | 225–510 |
| 132 | 388.0 | 376.0 | 382.2 | 392.4 | 398.8 @ 355 | 181–588 | 218–526 |
| 134 | 392.6 | 380.5 | 386.6 | 397.2 | 404.2 @ 359 | 172–614 | 205–557 |
| 136 | 397.1 | 384.9 | 390.9 | 401.9 | 409.5 @ 364 | 165–639 | 195–585 |

Natural mean-A reference δ_max(248) = 386.6 keV (June), 374.6 (annual) — the corpus value (P002). The isotope spread is 18 keV (¹²⁸Xe 379 → ¹³⁶Xe 397 keV, June); slope ≈ 2.3 keV per mass unit. δ_max(271 keV, ¹³⁶Xe) = 401.9 keV reproduces P030's "402 keV ceiling". Note that at δ = 380 keV the window for ¹²⁸Xe (254–453 keV) already excludes 248 keV, while ¹³⁶Xe's lower edge is 195 keV.

**Isotope shares of the 225–271 keV rate versus δ (16 June):**

| δ [keV] | R_win (unit) | ¹²⁸ | ¹²⁹ | ¹³⁰ | ¹³¹ | ¹³² | ¹³⁴ | ¹³⁶ |
|---|---|---|---|---|---|---|---|---|
| 300 | 290 | 0.023 | 0.369 | 0.043 | 0.236 | 0.210 | 0.078 | 0.042 |
| 350 | 30.7 | 0.018 | 0.317 | 0.040 | 0.239 | 0.229 | 0.098 | 0.060 |
| 366 | 7.35 | 0.011 | 0.241 | 0.035 | 0.239 | 0.253 | 0.129 | 0.092 |
| 375 | 1.80 | 0.003 | 0.111 | 0.025 | 0.219 | 0.282 | 0.192 | 0.169 |
| 380 | 0.552 | 0.0002 | 0.025 | 0.010 | 0.139 | 0.255 | 0.270 | 0.302 |
| 385 | 0.130 | 0 | 0.0001 | 0.001 | 0.025 | 0.089 | 0.298 | 0.587 |
| 389 | 0.037 | 0 | 0 | 0 | 0.0003 | 0.013 | 0.174 | 0.813 |
| 393 | 0.011 | 0 | 0 | 0 | 0 | 0.00003 | 0.082 | 0.918 |
| 397 | 0.0027 | 0 | 0 | 0 | 0 | 0 | 0.005 | 0.995 |

¹²⁸Xe stops contributing to 248 keV at δ = 379 keV, ¹²⁹Xe at 381, ¹³⁰Xe at 383, ¹³¹Xe at 386, ¹³²Xe at 388, ¹³⁴Xe at 393, ¹³⁶Xe at 397 keV (June); the window rate is > 50 % ¹³⁶Xe for δ ≥ 384 keV and > 80 % for δ ≥ 389 keV. (The point shares at exactly 248 keV, also in the CSV, show ¹³⁶Xe = 0 at all δ because of its node; ¹³⁴Xe then reaches 100 % at δ ≥ 389 keV.) At P021's likelihood peak (δ = 380 keV, natural Xe) the window rate is already 58 % ¹³⁴Xe+¹³⁶Xe and only 2.5 % ¹²⁹Xe.

## 5. Results — modified compositions

`P047_composition_ratios.csv`; Fig. 2. Rate per tonne in 225–271 keV relative to natural xenon (ρ):

| composition | O1 el. | O4 | O6 | O10 | L10 | δ=300 J | δ=350 J | δ=366 J | δ=380 J | δ=300 a | δ=350 a | δ=366 a |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ¹³⁶-depleted | 1.066 | 1.100 | 1.100 | 1.100 | 1.100 | 1.054 | 1.035 | 1.000 | 0.766 | 1.053 | 1.015 | 0.835 |
| ¹²⁹-enriched 80 % | 1.400 | 1.700 | 1.386 | 1.393 | **2.198** | 1.306 | 1.161 | 0.950 | 0.345 | 1.292 | 1.028 | 0.394 |
| ¹³⁶-enriched 90 % | 0.413 | 0.106 | 0.106 | 0.106 | **0.106** | 0.514 | 0.690 | **1.004** | **3.095** | 0.529 | 0.869 | 2.476 |
| ¹³⁴+¹³⁶ 90 % | 0.553 | 0.121 | 0.121 | 0.121 | 0.121 | 0.653 | 0.819 | 1.097 | 2.677 | 0.668 | 0.985 | 2.267 |
| even-A only | 0.692 | 0.004 | 0.004 | 0.004 | 0.004 | 0.748 | 0.840 | 0.983 | 1.581 | 0.757 | 0.927 | 1.472 |

Full ROI (efficiency-weighted), same order: ¹³⁶-depleted 1.001/1.100/1.100/1.100/1.100/0.966/0.924/0.854/0.604; ¹²⁹-enriched 1.004/1.800/1.031/1.147/1.471/0.902/0.771/0.607/0.316; ¹³⁶-enriched 0.994/0.106/0.106/0.106/0.106/1.308/1.682/**2.308**/**4.550**; ¹³⁴+¹³⁶ 0.994/0.121/…/1.278/1.554/1.994/3.321; even-A 0.997/0.004/…/1.061/1.170/1.323/1.696.

**Analytic checks.** For any pure spin operator ρ = [(a′₁₂₉/a₁₂₉) f₁₂₉ + (a′₁₃₁/a₁₃₁)(1−f₁₂₉)] ⟨A⟩/⟨A′⟩. In the ¹³⁶-enriched mixture both odd isotopes are scaled by the same factor 0.1098, so ρ_spin = 0.1098 × 131.39/135.49 = 0.1065 **independently of f₁₂₉** — the ¹³⁶-enriched discriminant for spin operators is nuclear-structure-free (it is just the odd-A fraction 5.2 % vs 47.6 %, per unit mass). For the ¹³⁶-depleted mixture ρ_spin = (1/0.9114) × 131.39/130.94 = 1.100. For ¹²⁹-enrichment ρ_spin = (3.030 f₁₂₉ + 0.2718 (1−f₁₂₉)) × 1.0134: L10 (f = 0.688) → 2.198, O4 (0.510) → 1.700, O6 (0.397) → 1.386 ✓. For inelastic δ = 366 keV in the ¹³⁶-enriched mixture ρ = 0.970 × [0.1098 (1 − f₁₃₆) + 10.16 f₁₃₆] = 0.970 × (0.1098 + 10.05 × 0.0922) = 1.004 ✓ — linear in the ¹³⁶Xe window share.

**Reading.** ¹³⁶Xe depletion (the cheapest option, the tails of 0νββ enrichment) changes every rate by ≤ 10 % except at δ ≥ 380 keV (−23 %): it is **not** a useful discriminant. ¹²⁹Xe enrichment raises L10 ×2.2 but SI/inelastic rates by ×1.3–1.4, a double ratio of only 1.6–2.3 (0.43 for δ = 366). The **¹³⁶Xe-enriched (nEXO-grade) mixture** is the discriminant: every spin operator falls ×0.106, elastic O1 falls ×0.41 (the ¹³⁶Xe node), the Higgsino-like δ = 366 keV rate is unchanged (×1.00) and δ = 380 keV rises ×3.1; in the full ROI the inelastic counts rise ×2.3 (366) and ×4.6 (380). Even-A xenon would be ideal (spin ×0.004) but is not producible.

### 5.1 Discrimination

`P047_discrimination.csv` (Asimov) and `P047_discrimination_exact_binomial.csv` (exact). Double ratio D = ρ_H1/ρ_H2 and the total number of window events (both detectors, equal exposure) for a 3σ separation; exposures use LZ's rate (1 event/2.84 t·yr in natural xenon):

| composition | H₁ vs H₂ | ρ₁ | ρ₂ | D | N_tot exact (H₁ true → reject H₂) | N_tot exact (H₂ true → reject H₁) | exposure/detector (H₁ true) | Asimov N_tot free / fixed norm |
|---|---|---|---|---|---|---|---|---|
| ¹³⁶-enr 90 % | δ=366 vs L10 | 1.004 | 0.106 | 9.44 | **9** | 14 | **12.8 t·yr** | 8.5 / 6.7 |
| ¹³⁶-enr 90 % | δ=380 vs L10 | 3.095 | 0.106 | 29.1 | 5 | 5 | 3.5 t·yr | 3.6 / 2.5 |
| ¹³⁶-enr 90 % | δ=350 vs L10 | 0.690 | 0.106 | 6.48 | 14 | 22 | 23.5 t·yr | 13.3 / 10.8 |
| ¹³⁶-enr 90 % | δ=300 vs L10 | 0.514 | 0.106 | 4.83 | 23 | 33 | 43.1 t·yr | 20.4 / 16.9 |
| ¹³⁶-enr 90 % | δ=366 vs O6 or O10 | 1.004 | 0.106 | 9.44 | 9 | 14 | 12.8 t·yr | 8.5 / 6.7 |
| ¹³⁶-enr 90 % | O1 el. vs L10 | 0.413 | 0.106 | 3.88 | — | — | — | 29.6 / 25.1 |
| ¹³⁶-enr 90 % | δ=366 vs O1 el. | 1.004 | 0.413 | 2.43 | 49 | 53 | 69 t·yr | 47.1 / 30.0 |
| ¹³⁴+¹³⁶ 90 % | δ=366 vs L10 | 1.097 | 0.121 | 9.07 | 9 | 13 | 12.2 t·yr | 8.5 / 6.5 |
| ¹³⁴+¹³⁶ 90 % | δ=380 vs L10 | 2.677 | 0.121 | 22.1 | 5 | 6 | 3.9 t·yr | 4.2 / 2.9 |
| ¹²⁹-enr 80 % | δ=366 vs L10 | 0.950 | 2.198 | 0.432 | 54 | 56 | 79 t·yr | 52.2 / 19.4 |
| ¹²⁹-enr 80 % | δ=380 vs L10 | 0.345 | 2.198 | 0.157 | 13 | 13 | 27 t·yr | 11.5 / 5.0 |
| ¹³⁶-depleted | δ=366 vs L10 | 1.000 | 1.100 | 0.909 | 3980 | 3960 | 5650 t·yr | 3940 / 1910 |
| ¹³⁶-depleted | δ=380 vs L10 | 0.766 | 1.100 | 0.697 | 280 | 278 | 450 t·yr | 277 / 140 |
| even-A only | δ=366 vs L10 | 0.983 | 0.004 | 244 | 4 | 10 | 5.7 t·yr | 2.2 / 2.0 |

Among the spin operators (L10, O4, O6, O10) no composition discriminates (identical ρ; D = 1, N = ∞), because any mass-separated mixture scales ¹²⁹Xe and ¹³¹Xe together or nearly so (the ¹²⁹-enriched case gives L10/O6 D = 1.59, N ≈ 190 events — impractical). The Asimov formula agrees with the exact binomial to ~1 event for N ≳ 9 and overestimates the power at N ≲ 4 (even-A case: 2.2 vs 4).

**Best realistic choice:** ¹³⁶Xe-enriched (or ¹³⁴+¹³⁶) xenon. Nine window events in total (≈ 13 t·yr per detector at LZ's rate, i.e. two 5-tonne-class detectors running ~2.5 years) separate a Higgsino-like δ = 366 keV interpretation from L10 (and from O6/O10) at 3σ; five events suffice if δ = 380 keV. Separating inelastic from *elastic* O1 needs ~50 events, and separating the spin operators from one another is impossible with isotopes alone.

## 6. Nuclear-structure caveats (quantified)

`P047_SI_shares_shell_vs_helm.csv`, `P047_spin_share_variation.csv`.

* **SI per-isotope shares.** Replacing the shell-model M response by the Helm form factor (`lz.dRdE_SI` per isotope, mass-fraction weighted) changes the O1 window shares to 0.030/0.362/0.049/0.220/0.238/0.065/0.037 (¹²⁸…¹³⁶) — shell/Helm per isotope 0.88/1.11/0.91/1.06/0.83/1.02/0.86. P017 showed the shell-model node positions vary 251–278 keV across isotopes and Helm/2pF move them by +12/+42 keV; the per-isotope window shares are therefore uncertain by **±10–20 % relative** (¹³⁶Xe's share in particular, sitting on its own node, could be 3–6 %). For inelastic δ = 366 keV, ρ(¹³⁶-enr) = 0.970 (0.1098 + 10.05 f₁₃₆) is linear in the ¹³⁶Xe window share; f₁₃₆ = 0.092 ± 20 % gives ρ = 0.83–1.18 and D(366/L10) = 7.8–11.1; the conclusion is unaffected.
* **¹³¹Xe non-dyad structure (P032).** The J = 3/2 M response has M₀ and M₂ multipoles, so its "node" is a minimum, not a zero; this only affects the ¹³¹Xe SI share (23 %) at the ~10 % level.
* **Spin shares.** Varying one odd isotope's Σ′/Σ″ response by ±15 % (P017's robustness band) moves f₁₂₉ for L10 from 0.688 to 0.652–0.722, for O4 0.469–0.550, for O6 0.359–0.437. This matters only for the ¹²⁹-enriched mixture: dρ/df₁₂₉ = 2.80, so ±0.035 in f₁₂₉ moves ρ(L10) = 2.20 by ∓0.10 (5 %). The ¹³⁶-enriched ρ_spin = 0.106 is exactly independent of f₁₂₉.
* **Two-body currents (recalled, likely).** Chiral two-body currents reduce the neutron-spin structure factors of ¹²⁹Xe and ¹³¹Xe by ~10–30 % at low q with a q-dependence that differs between the two isotopes (Menéndez–Gazit–Schwenk 2012; Klos et al. 2013); for SI, two-body (isoscalar) currents add a few–30 % smooth in q (Hoferichter et al. 2016). They are absent from WimPyDD's one-body responses and from LZ's spectra alike. They change f₁₂₉ by ≲ 0.05 and the isotope-ratio predictions for ¹³⁶-enriched xenon not at all (spin) or by ≲ 10 % (SI).
* **LZ's density-matrix modifications** ("as in LZ 2023") are unknown (P017); our shares use the stock DMFormFactor-v6 tables.

## 7. Practicality (recalled, qualitative)

* ¹³⁶Xe enrichment by gas centrifuge is industrial: EXO-200 used ~200 kg at 80.6 %, KamLAND-Zen ~745 kg at ~91 %, and nEXO plans ~5 t at ~90 % (likely). The enrichment cascade removes the odd isotopes to ~2–3 % automatically — our "¹³⁶-enriched 90 %" mixture (¹²⁹Xe 2.9 %, ¹³¹Xe 2.3 %) is precisely this material. A dual-phase TPC filled with it would be a DM detector with 1/9 of the spin-dependent sensitivity of natural xenon per tonne and 2–5× the inelastic-DM sensitivity near δ_max (full ROI), so the synergy runs both ways; nEXO itself (single-phase, no S2) is not such a detector.
* ¹³⁶Xe-depleted xenon is the tails stream of that enrichment and has been discussed for future DM TPCs to remove the ¹³⁶Xe 2νββ background (uncertain); it is useless as a discriminant here (≤ 10 % changes).
* ¹²⁹Xe enriched to ~80–86 % is sold for hyperpolarised-xenon MRI in litre (gram) quantities (likely); tonne-scale ¹²⁹Xe enrichment does not exist (uncertain), and it discriminates poorly (D ≤ 2.3) anyway.
* Cost: natural xenon ~ a few thousand USD per kg; enriched ¹³⁶Xe several times more (uncertain). Two 5-t-class detectors × 2.5 yr is the scale of XLZD/nEXO-era programmes, not of LZ.

## 8. Failed or abandoned approaches

* Point shares of dR/dE at exactly 248 keV were the first δ-scan observable; ¹³⁶Xe vanished at all δ because 248 keV sits on its M node (251 keV). Replaced by window-integrated (225–271 keV) shares (16 δ values × 7 isotopes × 24 energies).
* Asimov N_tot for the even-A mixture (2.2 events) is outside the asymptotic regime; the exact binomial count (4) is quoted instead.
* Pairs of pure spin operators gave KL = 0 (ZeroDivisionError) in the Asimov formula; guarded (N = ∞), which is the correct physical statement.

## 9. Files

* `output/code/P047_isotopes.py` — everything (≈ 3.5 min full, ≈ 40 s with `--fast`).
* `output/work/P047/run_log.txt` — all printed tables.
* `P047_isotope_fractions.csv` (per case × isotope: window, ROI, 248 keV rates and fractions), `P047_kinematics_per_isotope.csv`, `P047_share_vs_delta.csv`, `P047_compositions.csv`, `P047_composition_ratios.csv`, `P047_discrimination.csv`, `P047_discrimination_exact_binomial.csv`, `P047_SI_shares_shell_vs_helm.csv`, `P047_spin_share_variation.csv`, `P047_summary.json`, `P047_isotope_spectra.npz` (our cache of WimPyDD outputs).
* Figures: `figures/P047_fig1_isotope_fractions.png` (stacked isotope fractions per operator, window and full ROI), `figures/P047_fig2_composition_ratios.png` (window rate per tonne relative to natural xenon for five compositions × eight interactions, log scale), `figures/P047_fig3_share_vs_delta.png` (isotope shares of the window rate vs δ with per-isotope δ_max(248 keV) lines).

## 10. References

LZ Collaboration, arXiv:2609.02823 (2026). A. L. Fitzpatrick, W. Haxton, E. Katz, N. Lubbers, Y. Xu, JCAP 02 (2013) 004. N. Anand, A. L. Fitzpatrick, W. C. Haxton, PRC 89, 065501 (2014). I. Jeong, S. Kang, S. Scopel, G. Tomar, CPC 276, 108342 (2022) (WimPyDD). J. Menéndez, D. Gazit, A. Schwenk, PRD 86, 103511 (2012); P. Klos, J. Menéndez, D. Gazit, A. Schwenk, PRD 88, 083516 (2013). M. Hoferichter, P. Klos, J. Menéndez, A. Schwenk, PRD 94, 063505 (2016). EXO-200 Collaboration, JINST 7, P05010 (2012). KamLAND-Zen Collaboration, PRL 130, 051801 (2023). D. Baxter et al., EPJC 81, 907 (2021). Corpus: P002, P003, P006, P007, P017, P020, P021, P030, P032.

## 11. Tools and provenance (mirrors `output/provenance/P047.json`)

* **Agent tools:** Read ×18 (PAPER_GUIDE; dossier; ledger 3 pages; papers P017, P032, P003, P030, P021; code P017_nuclear_response.py, P032_isovector.py; lzcommon.py sections 108–217 and 305–385; provenance/P032.json; figures fig1, fig2, fig3 ×2), Bash ×11 (grep lzcommon function index + ls work/provenance; cat P017 CSVs/run_log + ENVIRONMENT_versions + P032 files; WimPyDD package.py grep and Xe object inspection; dataviz palette grep + `node validate_palette.js`; full run 3.5 min; two `--fast` re-runs; three `wc -w` budget checks), Write ×4 (script, details, JSON, paper), Edit ×15 (script ×4: KL guard, exact binomial block, window-integrated δ scan, Fig. 3; paper ×10 word-budget trims; JSON ×1 tool counts), Skill ×1 (dataviz).
* **Software:** python 3.12.13; WimPyDD 2.0.4 (`streamed_halo_function` via `lz.wd_halo` with explicit v_min grid, annual and day 167; `eft_hamiltonian` incl. q-dependent O4 coefficient for L10; `diff_rate` with `isotopes_list` and `delta`; `Xe.func_w`, `nuclear_current`, `Xe.abundance/isotopes`); numpy 2.5.3; scipy 1.18.1 (`special.erf`, `stats.binom`, `stats.norm`, `integrate/np.trapezoid`); pandas 3.0.5; matplotlib 3.11.2 (Agg); common/lzcommon.py (`wd`, `wd_halo`, `wd_hamiltonian`, `wd_c_from_anand`, `delta_max_kev`, `E_R_range_keV`, `v_earth_kms`, `mu_red`, `m_nucleus_gev`, `dRdE_SI`, constants); node (dataviz palette validator only, no chart output).
* **Recalled knowledge (10 items):** see JSON.
* **Datasets / data requests:** none. **WimPyDD-generated files:** none by WimPyDD itself; our cache `P047_isotope_spectra.npz`.
