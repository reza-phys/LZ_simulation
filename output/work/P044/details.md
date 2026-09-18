# P044 — Anapole and electric-dipole dark matter versus the LZ event: contact and photon-mediated versions

Research record (simulated date 2026-09-10). Script: `output/code/P044_anapole_edm.py`; outputs in `output/work/P044/` (`P044_results_table.csv`, `P044_distinguishability_vs_L10.csv`, `P044_spectra_unit_coupling.npz`, `P044_summary.json`, `P044_run.log`, `figures/`). All rates: WimPyDD 2.0.4 through `lzcommon.wd_rate` (natural xenon, shell-model responses), Sun-frame Baxter-2021 SHM from `lz.wd_halo()` (v_max = 844 km/s on the explicit v_min grid), P003 efficiency model (plateau 0.96, 50 % at 5.4 and 269.9 keV, erf widths 3.4 / 8 keV), exposure 2.84 t·yr.

## 1. Motivation and framework

LZ's Table S6 gives its maximum local significance, 3.4σ, to two Lagrangians at 400–1000 GeV: L10 (magnetic tensor–tensor, treated by P003 and P012) and L16. Electromagnetic moments are the most economical way for neutral dark matter to talk to nuclei, so we ask which of the three classic moment interactions — anapole, magnetic dipole (P012), electric dipole — can produce a *lone* 248 keV recoil, i.e. one event in 200–270 keV with ≲ 3–5 companions in the 5.4–55 keV window of LZ's 2024 search (P003 criterion; P016 shows LZ's own profile fit tolerates only 1.6–2.2 companions, with Z falling logarithmically with N_lo).

P031 (directdm matching; its paper is in preparation, we use its `work/P031` tables) obtained the NR coefficients of the anapole and of the photon EDM numerically. We use those numbers as an independent check of the analytic reductions below, and go further: contact analogues, the coupling values for one LZ event in physical units, bounds from the 2024 null, the identity of L16, and the number of events needed to separate each structure from L10.

## 2. Non-relativistic reductions (all recalled/likely unless stated)

Conventions: Anand–Fitzpatrick–Haxton operators O_i; per-nucleon coefficients c_i^p, c_i^n; WimPyDD isospin coefficients c^0 = c_p + c_n, c^1 = c_p − c_n (P003, settled by digitising LZ Fig. 1). e = √(4πα) = 0.30282; g_p = 5.5857, g_n = −3.8261; m_N = 0.938272 GeV; ħc = 0.19733 GeV·fm; 1 μ_N = 1.052 × 10⁻¹⁴ e·cm.

**Anapole.** L = 𝒜 χ̄γ^μγ⁵χ ∂^νF_{μν}. The photon equation of motion ∂^νF_{μν} = e J^EM_μ turns this into the *contact* interaction e𝒜 χ̄γ^μγ⁵χ J^EM_μ — the photon propagator is cancelled, so there is no 1/q². The nucleon EM current is N̄[F₁γ_μ + F₂ iσ_{μν}q^ν/(2m_N)]N with F₁ = Q_N and F₁ + F₂ = g_N/2. The axial-DM × vector-nucleon bilinear reduces to 2O8 + 2O9 (Fitzpatrick et al. 2012, Table 1; certain up to sign conventions) and the tensor piece adds 2F₂O9, hence

  c8^N = 2e𝒜 Q_N  (proton only: velocity-suppressed coherent term),  c9^N = e𝒜 g_N  (spin term with the *full* magnetic moment).

Check against directdm (work/P031 §3.1, C62q = eQ_q): c8^p = 0.606 = 2e, c8^n = 0, c9^p = 1.69 = e g_p, c9^n = −1.16 = e g_n. Exact agreement, so the reduction is "likely" for the relative sign O8/O9 and effectively certain for the magnitudes. The relative sign is tested by a flipped-sign Hamiltonian.

**Electric dipole.** L = (d_E/2) χ̄σ^{μν}iγ⁵χ F_{μν}. Photon exchange with the nuclear charge gives the long-range operator

  c11^N = 2e d_E Q_N m_N / q²  (O11 = i S_χ·q/m_N; proton only).

This is the iγ⁵ partner of P012's charge–magnetic-dipole term c5 = 2eμ m_N Q_N/q². Check: directdm's C52 (= e/(8π²) χ̄σ^{μν}iγ⁵χF_{μν}, so d_E = eC52/(4π²) = 7.67 × 10⁻³ GeV⁻¹ at C52 = 1) gives c11^p = 0.072 GeV⁻² at q = 245 MeV (work/P031); our formula: 2 × 0.3028 × 7.67 × 10⁻³ × 0.938/0.0600 = 0.0726. Agreement to 1 %.

**Contact analogues.** (i) EDM-type with a heavy mediator of mass M: c11^p = 2e d_E m_N/M², constant (this is also Anand's L3-type O11 structure and P031's Q16). (ii) One more derivative: c11^p ∝ q²/m_v² ("O11 × q²"). (iii) Anapole-type heavy vector coupled to the proton vector charge with no tensor form factor (F₂ = 0): c8 = c9 = 2e𝒜_c (Fitzpatrick 2O8 + 2O9). (iv) A dimension-8 anapole with both coefficients × q²/m_v². (v) Pure O9 (isoscalar unit c⁰ = 1/m_v², WimPyDD 2/m_v²) — the axial-DM × tensor-nucleon structure χ̄γ^μγ⁵χ N̄iσ_{μν}q^ν N reduces to O9 alone (Anand L9-type; P031's Q17) — and O9 × q²/m_N². (vi) O14 (a velocity-suppressed spin operator, to which axial × pseudotensor structures reduce at leading non-vanishing order — our own estimate, uncertain). (vii) L10 reference: P012's 4[(q²/m_N²)O4 − O6] d10/m_v², WimPyDD normalisation.

Anand's L16 (1308.6288, Table 1) — we could not reconstruct its exact form from memory (uncertain). We therefore test candidate identities against Table S6 rather than assert one.

## 3. Method

Spectra dR/dE at 115 energies (1–330 keV, 3 keV steps plus 5.4, 55, 100, 200, 248, 269.9, 300 keV) for each Hamiltonian at unit coupling; 200/1000/4000 GeV for the main structures, 1000 GeV for the diagnostics. Efficiency-weighted window rates R_lo (5.4–55), R_mid (55–200), R_100–200, R_hi (200–270), R_roi (1–330 keV, efficiency-weighted). N_lo = R_lo/R_hi (coupling-independent). Couplings: g(1 ROI event) = √(1.0/N_roi,unit) — the LZ best-fit convention (Table I: 1.0 +1.4/−0.7 events in the whole ROI) — and g(1 hi event) = √(1/N_hi,unit) for one event in 200–270 keV; 68 % range from 0.3–2.4 events. The 2024 bound: g_max = g(1 hi) × √(tol/N_lo,2024) with N_lo,2024 = N_lo × 4.2/2.84 (2024 exposure 4.2 t·yr recalled, certain; tolerance 3 or 5 events recalled, uncertain). Z_pred(N_lo) interpolates P016's calibrated (N_lo, Z) points in log N_lo (as P031 did). Distinguishability from L10: normalised efficiency-weighted pdfs on 1–330 keV, KL divergences, and N_3σ = 9/Z₁², Z₁ = (KL_pq + KL_qp)/(√Var_p + √Var_q) (per-event symmetric log-likelihood-ratio test), without and with Gaussian smearing σ(E) = 23 keV √(E/248 keV) (LZ's ±23 keV stat at 248 keV; √E scaling recalled, likely).

WimPyDD pitfall (found by P031, respected here): coefficient functions must capture constants by closure; any non-reserved argument (even a default) becomes a shared global parameter. Reserved names: q, mchi, delta, q0*.

## 4. Results — spectral shapes and companions (1000 GeV; 200 / 4000 GeV in brackets)

| structure | N_lo | N_mid/hi | R(100–200)/R(200–270) | f_hi | Z_pred (P016) | eff.-weighted peak |
|---|---|---|---|---|---|---|
| photon anapole O8+O9 | **26.9** (82.5 / 23.1) | 3.76 | 2.13 | 0.030 | 2.71 (2.27 / 2.76) | 10 keV |
|  – O8 (charge) only | 1264 | 71.7 | 8.49 | 7 × 10⁻⁴ | 0.86 | 10 |
|  – O9 (magnetic) only | 2.22 | 2.58 | 2.04 | 0.170 | 3.11 | 13 |
|  – relative sign flipped | 25.2 | 3.97 | 2.18 | 0.031 | 2.73 | 10 |
| heavy-vector A–V, F₂ = 0 (c9 = c8) | 946 | 60.2 | 9.52 | 9 × 10⁻⁴ | 1.00 | 10 |
| anapole × q²/m_v² (dim-8 contact) | 0.29 | 1.15 | 0.98 | 0.405 | 3.33 | 211 |
| photon EDM O11/q² | **18 100** (54 100 / 15 600) | 173 | 7.19 | 3.8 × 10⁻⁵ | 0.0 | 1 |
| contact EDM-type O11 (proton) | 118 (338 / 102) | 18.2 | 3.73 | 0.007 | 2.12 | 19 |
| O11 × q²/m_v² | 2.19 (6.09 / 1.91) | 3.38 | 2.05 | 0.152 | 3.11 | 46 |
| pure O9, isoscalar | 2.07 (6.36 / 1.77) | 2.69 | 2.07 | 0.171 | 3.12 | 13 |
| O9 × q²/m_N² | 0.024 | 1.04 | 0.96 | 0.478 | 3.45 | 214 |
| O14 isoscalar | 2.35 | 2.83 | 2.15 | 0.160 | 3.11 | 13 |
| L10 (P012) | 0.199 (0.601 / 0.171) | 1.61 | 1.39 | 0.353 | 3.4 | 199 |

Anapole decomposition (1000 GeV): the O8 charge term is 97.8 % of R_lo (95.5 % of dR/dE at 20 keV) but 2.1 % of R_hi; the O9 term is 9.0 % of R_lo and 109 % of R_hi (110 % of dR/dE at 248 keV), the O8–O9 interference (Δ–Σ′ response) being −6.8 % of R_lo and −10.8 % of R_hi. Flipping the relative sign changes R_hi by +22 % and R_lo by +14 % and N_lo from 26.9 to 25.2 — the conclusion is insensitive to the sign convention. The anapole is therefore a *two-regime* interaction: velocity-suppressed coherent scattering rules the low-energy window (v⊥² peaks at small q, so O8 is not q-suppressed), the spin term rules the high-energy window. The 248 keV event would be an O9 (spin) scatter, but the same coupling puts 27 O8 scatters into 5.4–55 keV per such event — 40 events in the 2024 exposure.

Cross-checks (`cross_checks` in the summary JSON): pure O9 N_lo 2.07 vs P003 2.1; L10 N_lo 0.199 and N(d10 = 1) = 3.335 events vs P012 0.20 and 3.34; anapole N_lo 26.91 vs P031 26.9; photon EDM 18 100 vs P031 18 200; anapole 𝒜(1 ROI event) 2.73 × 10⁻⁶ GeV⁻² vs P031 2.73 × 10⁻⁶. The proton-only contact O11 gives N_lo = 118, between the isoscalar (258, P003) and isovector (34) values, as expected for a Z-weighted coherent response.

## 5. Results — couplings and physical moments

| structure | m [GeV] | coupling for 1.0 ROI event (68 %: 0.3–2.4 ev.) | for 1 event in 200–270 keV | 2024-null bound (≤ 3 / ≤ 5 events) |
|---|---|---|---|---|
| photon anapole 𝒜 | 200 | 1.32 × 10⁻⁶ GeV⁻² | 1.31 × 10⁻⁵ | 2.05 / 2.64 × 10⁻⁶ |
|  | 1000 | **2.73 × 10⁻⁶ GeV⁻²** (1.49–4.23) = (605 GeV)⁻² = 1.06 × 10⁻⁷ e·fm² | 1.59 × 10⁻⁵ → 39.8 low-E events in 2024 | 4.36 / 5.63 × 10⁻⁶ |
|  | 4000 | 5.38 × 10⁻⁶ = (431 GeV)⁻² | 2.92 × 10⁻⁵ | 8.65 / 11.2 × 10⁻⁶ |
|  | (200) | (1.32 × 10⁻⁶ = (871 GeV)⁻² = 5.1 × 10⁻⁸ e·fm²) | | |
| photon EDM d_E | 200 | 7.62 × 10⁻¹¹ GeV⁻¹ = 1.50 × 10⁻²⁴ e·cm | 4.24 × 10⁻²² e·cm | 2.60 / 3.35 × 10⁻²⁴ e·cm |
|  | 1000 | **1.66 × 10⁻¹⁰ GeV⁻¹ = 3.28 × 10⁻²⁴ e·cm** | 5.32 × 10⁻²² e·cm → 26 800 low-E events in 2024 | 5.63 / 7.27 × 10⁻²⁴ e·cm |
|  | 4000 | 3.32 × 10⁻¹⁰ = 6.54 × 10⁻²⁴ e·cm | 9.82 × 10⁻²² | 1.12 / 1.45 × 10⁻²³ e·cm |
| contact EDM-type c11^p | 1000 | 2.44 × 10⁻⁸ GeV⁻² → d_E/M² = 4.29 × 10⁻⁸ GeV⁻³ (d_E = 8.5 × 10⁻¹⁶ e·cm for M = 1 TeV) | 2.87 × 10⁻⁷ | 3.76 × 10⁻⁸ |
| O11 × q²/m_v² (c11^p at q = m_v) | 1000 | 0.126 → c11^p(q_event) = 1.26 × 10⁻⁷ GeV⁻² | 0.322 | 0.310 |
| pure O9 c9^s m_v² | 1000 | 0.379 (200: 0.214; 4000: 0.729) | 0.916 | 0.907 |
| O9 × q²/m_N², c9^s m_v² at q = m_N | 1000 | 10.1 → 0.69 at q_event | 14.6 | – (N_lo 0.024) |
| anapole × q²/m_v², 𝒜₈ | 1000 | 11.0 GeV⁻² → 𝒜_eff(q_event) = 1.1 × 10⁻⁵ GeV⁻² | 17.4 | 
| L10 d10 (WimPyDD norm.) | 1000 | 0.548 (= 0.28 in LZ's normalisation, P012 factor 1.96) | 0.922 | 2.95 |

(q_event² = 2 m_Xe × 248 keV = 0.0607 GeV², q_event = 246 MeV; q²/m_v² = 1.0 × 10⁻⁶, q²/m_N² = 0.069.)

Reading. The anapole moment that gives LZ its 1.0 fitted event is 𝒜 = 2.7 × 10⁻⁶ GeV⁻² — a perfectly natural loop-induced size, Λ ≈ 600 GeV — but 97 % of that event sits below 55 keV; the moment that would make the *observed* 248 keV recoil is 1.6 × 10⁻⁵ GeV⁻², 3.6× above what the 2024 window allows (4.4 × 10⁻⁶ at ≤ 3 events) and predicting 40 low-energy events there. The photon EDM is hopeless: 1 high-energy event needs d_E = 5.3 × 10⁻²² e·cm and brings 2.7 × 10⁴ low-energy scatters; LZ's low-energy null alone bounds d_E ≤ 5.6 × 10⁻²⁴ e·cm at 1 TeV (2.6 × 10⁻²⁴ at 200 GeV). Recalled literature values for photon-EDM DM from XENON100/LUX-era recasts are ~10⁻²²–10⁻²¹ e·cm at 100 GeV–1 TeV (Banks–Fox–Weiner 2010; Barger–Keung–Marfatia 2011; Del Nobile et al. 2012; **uncertain**); an LZ-scale exposure (~100× larger) should improve them by ~10 in d_E, so our 5 × 10⁻²⁴ e·cm is of the expected order — we quote it as our own derived number, not as a published limit. For the magnetic dipole P012 found the same structure: μ(1 hi event) = 2.1 × 10⁻⁵ μ_N with N_lo = 376.

## 6. What is L16? (see also the addendum in §6a)

Table S6 (tex lines 852–866): L16^s = 0/…/2.8/3.3/3.4/3.4/3.2 at 50/…/100/200/400/1000/4000 GeV, L16^v = 2.4/3.2/3.3/3.2/3.2; L10^s = 2.9/3.1/3.4/3.4/3.4; L9^s = 2.1/2.9/2.9/3.0/3.1, L9^v = 2.7/3.0/3.1/3.2/3.1; L18^s = 2.2/2.9/3.1/3.1/3.1, L19 = 2.2/2.9/3.0/3.0/3.1; L11 = L14 (2.5/3.0/3.1/3.2/3.2); L12 3.2–3.3. LZ lists the s/v-degenerate Lagrangians (L2, L4, L7, L8, L10, L11, L14, L15, L19, L20): L16 is *not* among them, so its isoscalar and isovector spectra differ, which excludes a pure single-spin-operator identity with equal p/n weights but is consistent with a spin operator carrying g_p ≠ g_n weights or an admixture of a coherent piece.

Using P016's calibration Z(N_lo): 3.4σ requires N_lo ≲ 0.3 (L10: 0.20 → 3.40; O6: 0.43 → 3.26). Hence:
- L16 ≠ photon anapole: N_lo = 27 → Z_pred = 2.7 (P031 reached the same conclusion). The anapole's O8 charge term is fatal.
- L16 ≠ pure O9 (any g-weighting): N_lo = 2.1–2.2 → Z_pred 3.1. Pure O9 instead reproduces L9 (3.0/3.2) and L11 = L14 (3.2) and, with its velocity-suppressed sibling O14 (2.35 → 3.1), L18/L19 (2.9–3.1).
- L16 ≈ a q⁴-weighted spin structure: O9 × q²/m_N² (N_lo 0.024, peak 214 keV) or anapole × q² (0.29, peak 211 keV) give Z_pred ≥ 3.3. If Anand's L16 is χ̄γ^μγ⁵χ N̄iσ_{μν}q^νγ⁵N/m_M (assignment hypothesis, flagged "likely" by the coordinator, "uncertain" by us), the pseudotensor nucleon current carries an extra q relative to L9's tensor current, which would produce exactly such a q²O9-type operator; we could not verify the reduction from memory. The 4 TeV behaviour (L16 drops to 3.2 while L10 stays 3.4) is a further discriminant: a spectrum harder than L10 loses more of its rate above the 270 keV edge as the kinematic endpoint rises with mass — tested in §6a.

### 6a. Addendum — hardness versus mass (second run; `f_edge_270_330`, `f_raw_above270`, `rho_248_per_roi_event` in the results CSV)

| structure | m [GeV] | eff.-weighted peak | f_hi (200–270 keV / ROI) | raw fraction above 270 keV | ρ(248 ± 10 keV) per ROI event [keV⁻¹] | N_lo |
|---|---|---|---|---|---|---|
| L10 | 200 / 1000 / 4000 | 175 / 199 / 199 keV | 0.233 / 0.353 / 0.370 | 0.048 / 0.106 / 0.117 | 0.0026 / 0.0045 / 0.0048 | 0.60 / 0.20 / 0.17 |
| O9 × q²/m_N² | 200 / 1000 / 4000 | 196 / 214 / 214 | 0.365 / 0.478 / 0.493 | 0.093 / 0.172 / 0.185 | 0.0045 / 0.0065 / 0.0068 | 0.072 / 0.024 / 0.021 |
| pure O9 | 200 / 1000 / 4000 | 13 | 0.081 / 0.171 / 0.187 | 0.013 / 0.041 / 0.048 | 0.0008 / 0.0020 / 0.0022 | 6.4 / 2.1 / 1.8 |
| photon anapole | 200 / 1000 / 4000 | 10 | 0.010 / 0.030 / 0.034 | 0.002 / 0.009 / 0.011 | 0.0001 / 0.0003 / 0.0004 | 82 / 27 / 23 |

O9 × q² is harder than L10 (17 % vs 11 % of the raw rate above 270 keV at 1 TeV; peak 214 vs 199 keV) and its in-window density per ROI event is 1.45× L10's, consistent with a Z at least as high as L10's at 400–1000 GeV. But the mass dependence between 1 and 4 TeV is only +7 % (relative) in the above-edge fraction and +5 % in ρ for *both* structures, so this simple proxy does **not** explain why Table S6 has L16 dropping from 3.4σ to 3.2σ at 4 TeV while L10 stays at 3.4σ; the drop may come from LZ's full two-dimensional (S1, S2) likelihood, from a q-dependence steeper than q⁴, or from the isoscalar/isovector composition. The L16 identification remains: a companion-free (N_lo ≲ 0.3), L10-like or harder spin structure — not the anapole, not pure O9. Z_pred values of 3.45 in the tables are the cap of the P016 interpolation (its calibration ends at L10's 0.20 → 3.40) and should be read as "≥ 3.4".

Coupling for O9 × q² (Anand unit c9^s m_v², coefficient evaluated at q = m_N): 7.3 / 10.1 / 18.8 for 1.0 ROI event at 200 / 1000 / 4000 GeV (12.1 / 14.6 / 26.8 for one 200–270 keV event), i.e. c9^s m_v² = 0.50 / 0.69 / 1.29 at q_event.

## 7. Distinguishability from L10 (1000 GeV)

| structure | KL(p‖L10) ideal / smeared | N_3σ ideal / smeared (σ_E = 23 keV √(E/248)) |
|---|---|---|
| photon anapole | 2.27 / 2.09 | 5.2 / 5.1 |
| heavy-vector A–V (F₂ = 0) | 2.76 / 2.56 | 2.6 / 2.5 |
| photon EDM | 4.45 / 3.48 | 2.0 / 1.6 |
| contact EDM-type O11 | 2.18 / 2.09 | 3.2 / 3.5 |
| O11 × q² | 0.71 / 0.57 | 28 / 34 |
| pure O9 (or anapole O9 part, O14) | 0.46–0.52 / 0.43–0.49 | 41–46 / 43–48 |
| O9 × q² | 0.083 / 0.078 | 187 / 200 |
| anapole × q² | 0.051 / 0.038 | 372 / 485 |

Structures with a low-energy population separate from L10 in 2–5 events (they would in fact have been separated by the 2024 data already); the compatible spin operators need 30–50 events; the q⁴-spin candidates for L16 are nearly degenerate with L10 (peaks 211–214 vs 199 keV; R(100–200)/R(200–270) 0.96–0.98 vs 1.39) and need 200–500 events — i.e. they are indistinguishable for any foreseeable xenon exposure (P020: ~2.4 events expected in the untouched 6.76 t·yr at the best-fit rate). Resolution smearing changes N_3σ by ≤ 30 %.

## 8. Failed or abandoned approaches

- Reconstructing Anand et al.'s L16, L18, L19 Lagrangians and their reductions from memory: abandoned as unreliable; replaced by the pattern test of §6/§6a against Table S6 with flagged candidates.
- A first idea to quote published anapole/EDM limits directly was replaced by a self-computed 2024-null bound, because the recalled values (10⁻²²–10⁻²¹ e·cm) are of uncertain provenance and refer to much smaller exposures.
- The first run lacked the mass-dependent hardness columns and O9 × q² at 200/4000 GeV; the script was extended and rerun (the rerun exceeded the 600 s foreground limit and completed in the background; all first-run numbers were reproduced identically).

## 9. Discussion

Three electromagnetic moments, three fates. The magnetic dipole (P012) and the anapole are both "contact-like at high q, coherent at low q": for the dipole the charge–dipole O5/q² term carries Z²/E_R, for the anapole the charge term O8 carries Z² v⊥² with no q-suppression. Both therefore predict hundreds (376) or tens (27) of low-energy companions per 248 keV recoil and are excluded as the origin of a lone event by the 2024 null — with the anapole the marginal case: its Z_pred = 2.7 is what LZ gives to O4/L15, and P016 shows such spectra retain ~2.7σ only because the profile fit shrinks the signal to 0.06 events at 248 keV. The photon EDM is the extreme opposite (18 000 companions) — its 1/q² is untamed by any spin factor. Only the *pure spin, q-weighted* structures survive: O9 (Z_pred 3.1, matching L9/L11/L14), O11 × q² (3.1) and, at the 3.4σ level, q⁴-spin structures such as O9 × q² (peak 214 keV, N_lo 0.02), which are what L16 must effectively be if Table S6 is taken at face value. Among UV completions, this points away from photon-mediated moments (all excluded here or in P012) toward heavy-mediator dipole × axial-current or axial × pseudotensor contact operators at Λ ~ tens of GeV (P031's Q17 at 62 GeV), or the P012 L10 dipole–dipole at (21 GeV)⁻⁴ — none of which is a natural heavy-mediator EFT. The coupling values in §5 (𝒜 = 2.7 × 10⁻⁶ GeV⁻², d_E = 3 × 10⁻²⁴ e·cm for one *ROI* event) remain useful as LZ-derived bounds: they are what the fitted 1.0-event normalisation implies once the 2024 window is included, and are 10–100× stronger than recalled XENON100/LUX-era recasts.

## 10. Figures

- `figures/P044_spectra_1TeV.png` — efficiency-weighted spectra at 1 TeV, each normalised to 1.0 ROI event: L10, photon anapole (full, O8, O9), photon EDM, contact O11, O11 × q², pure O9.
- `figures/P044_Nlo_bar.png` — N_lo per 200–270 keV event for all structures (1 TeV) with the 3/5-event lines and Z_pred labels.
- `figures/P044_moment_vs_mass.png` — anapole moment (left) and EDM (right) giving 1.0 (0.3–2.4) LZ ROI events and 1 event in 200–270 keV versus mass, against the 2024-null bound.

## 11. References

LZ Collaboration, arXiv:2609.02823 (2026). A. L. Fitzpatrick, W. Haxton, E. Katz, N. Lubbers, Y. Xu, JCAP 02 (2013) 004. N. Anand, A. L. Fitzpatrick, W. C. Haxton, PRC 89 (2014) 065501. C. M. Ho, R. J. Scherrer, PLB 722 (2013) 341 (anapole DM). T. Banks, J. F. Fox, N. Weiner, PRD 82 (2010) 075004. V. Barger, W.-Y. Keung, D. Marfatia, PLB 696 (2011) 74. E. Del Nobile, C. Kouvaris, P. Panci, F. Sannino, J. Virkajärvi, JCAP 08 (2012) 010. M. I. Gresham, K. M. Zurek, PRD 89 (2014) 123521. F. Bishara, J. Brod, B. Grinstein, J. Zupan, JCAP 02 (2017) 009 (directdm). S. Kang, S. Scopel, G. Tomar, J.-H. Yoon, CPC 276 (2022) 108342 (WimPyDD). Corpus: P003, P012, P016, P020, P027, P031 (work directory).

## 12. Tools and provenance

Mirrors `output/provenance/P044.json`. Agent tools: Read ×13 (PAPER_GUIDE, dossier, ledger, P003/P012/P016/P027 papers, P012 script, lzcommon wrapper, 3 figures), Bash ×24 (listings; tex Theory paragraph, Table S6, LEE section; P031 tables/details/script; WimPyDD reserved-argument check; two script runs; word counts), Write ×5 (script, details, provenance, paper ×2), Edit ×13 (script ×3, details ×3, paper ×5, provenance ×2), ToolSearch ×1 (Monitor, not used). Software: python 3.12.13, WimPyDD 2.0.4 (eft_hamiltonian with q-dependent coefficients, diff_rate, streamed_halo_function), numpy 2.5.3, scipy 1.18.1 (special.erf, integrate.trapezoid), pandas 3.0.5, matplotlib 3.11.2, common/lzcommon.py. Recalled knowledge: 10 items (anapole and EDM reductions — likely, directdm-confirmed; Fitzpatrick 2O8+2O9 — certain; e, g-factors, m_N, ħc — certain; 2024 exposure 4.2 t·yr — certain; 2024 tolerance 3–5 events — uncertain; literature EDM limits 10⁻²²–10⁻²¹ e·cm — uncertain; √E resolution scaling — likely; Anand L16 form — uncertain). Datasets: none. Data requests: none (DR-001 of P012 affects only the L10 d10 normalisation quoted for comparison).
