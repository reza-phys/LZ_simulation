# P048 · Chargino lifetimes, disappearing tracks and the road to a thermal Higgsino: HL-LHC, FCC-hh and a muon collider after LZ — research record

Simulated date 2026-09-10. Author profile: future-collider phenomenologists. Category COMP, hep-ph. Competes with P014 (P014 = present-day
status; this paper = future reach and the δ-related signatures). Script: `output/code/P048_future_colliders.py`, run from the simulation
root with `.venv/bin/python`; every number below is printed in `output/work/P048/run_log.txt` or stored in the CSV/JSON files of §11.
Recalled inputs are flagged inline and collected in `recalled_knowledge.json` (21 items). Nothing was fetched. Result type: estimated.

## 1. Motivation and framework

P007 fits the LZ event with a pure Higgsino at δ ≈ 366 keV (1 TeV; 310–376 keV over 300–4000 GeV) with PeV gauginos; P037 shows that any
Y ≠ 0 multiplet does the same with δ = 366–383 keV, the Y = 1 triplet with a 22 TeV mediator being the only other heavy-mediator-consistent
case. P014 computed the collider-visible splitting, Δm± = 342 ± 10 MeV at 1 TeV, cτ = 0.71 cm, independent of δ to 0.2 %, and showed that
LEP, LHC and HL-LHC (12 cm tracklets) do not reach any P007 fit point. We ask what the *next* machines would measure:

1. how many doublet pairs HL-LHC (14 TeV, 3 ab⁻¹), FCC-hh (100 TeV, 30 ab⁻¹) and a 10 TeV μ⁺μ⁻ collider (10 ab⁻¹) produce;
2. what fraction of the cτ = 0.71 cm charginos decays beyond the first tracker layers (5, 10, 12, 30 cm) given each machine's βγ_T
   distribution, and the resulting mass reach against recalled fake-tracklet backgrounds;
3. whether mono-photon at the muon collider adds anything;
4. why δ itself is invisible at any collider (χ₂ decay kinematics and lifetime; how δ enters Δm±) and what the collider–direct-detection
   "identification" chain therefore looks like;
5. a timeline.

Notation: m = doublet mass (μ), Δm± = chargino–χ₁ splitting, δ = χ₂–χ₁ splitting, βγ_T = p_T/m of the chargino, r = transverse decay radius.

## 2. Cross-sections (Part A)

### 2.1 Hadron colliders by τ-scaling of the P014 anchors

P014 recalled the 13 TeV NLO+NLL pure-wino χ̃₁±χ̃₂⁰ cross-sections (12 pb at 100 GeV … 1.3 fb at 1 TeV; likely, factor ~1.5) and fitted a
quadratic in ln m; the Higgsino χ±χ⁰ channels are half of that (W coupling g/2 per Majorana neutral state), with χ⁺χ⁻ + χ₁χ₂ adding ≈ 50 %.
We reuse the fit (reproduces 201.4 / 23.0 / 0.666 fb at 300 / 500 / 1000 GeV vs P014's 201 / 23 / 0.67) and scale to other energies with the
collinear-factorisation identity [own derivation; likely, ×/÷1.5 because PDF Q² evolution is neglected]

  σ(m, s₂) = σ(m √(s₁/s₂), s₁) × s₁/s₂,

valid when σ̂ ∝ 1/ŝ and the parton luminosity depends on τ = 4m²/s only. Results (`hadron_xsec_pairs.csv`; χ±χ⁰ channel, fb):

| m (GeV) | 13 TeV | 14 TeV | 14/13 | 100 TeV | 100/13 | pairs HL-LHC 3 ab⁻¹ (all channels) | pairs FCC-hh 30 ab⁻¹ |
|---|---|---|---|---|---|---|---|
| 500 | 23.0 | 27.8 | 1.21 | (242, clipped) | — | 1.25×10⁵ | — |
| 800 | 2.24 | 2.85 | 1.27 | 93.2 | 42 | 1.28×10⁴ | 4.2×10⁶ |
| 1000 | 0.666 | 0.866 | 1.30 | 53.0 | 80 | 3.9×10³ | 2.4×10⁶ |
| 1100 | 0.388 | 0.509 | 1.31 | 40.8 | 105 | 2.3×10³ | 1.8×10⁶ |
| 1500 | 0.061 | 0.083 | 1.36 | 15.9 | 260 | 372 | 7.1×10⁵ |
| 2000 | 0.0097 | 0.0136 | 1.40 | 5.85 | 604 | 61 | 2.6×10⁵ |
| 3000 | 5.9×10⁻⁴ | 8.7×10⁻⁴ | 1.46 | 1.17 | 1970 | 3.9 | 5.3×10⁴ |

The 14/13 TeV ratio of 1.2–1.4 is the familiar one. At 100 TeV the method needs the 13 TeV curve at m × 0.13; below 60 GeV we freeze the
curve (rows with `clipped_100TeV = True`, m < 460 GeV, are lower bounds and enter no reach statement). At 1 TeV we obtain
σ(χ⁺χ⁻ + χ±χ⁰) ≈ 71 fb and σ(all) ≈ 80 fb, versus the assignment's recalled 30 fb and Low–Wang's ≈ 60 fb [uncertain, factor 3]; we
therefore carry a ×/÷3 band on every FCC-hh number (§4.4). The local slope d ln σ/d ln m at 100 TeV is −2.7 (1 TeV) to −4.3 (3 TeV),
versus −5.6 at 13 TeV; it sets the pair-mass spectrum in §3.

### 2.2 Muon collider at tree level

For unpolarised μ⁺μ⁻ → f f̄ through s-channel γ and Z, with a Dirac fermion of mass m, charge Q and Z couplings g_L, g_R (v = g_L + g_R,
a = g_L − g_R, lepton v_μ = −½ + 2s_W², a_μ = −½) [certain; textbook]:

  σ = (4πα²/3s) β { (3−β²)/2 [Q² − 2Q v_μ v Re χ + (v_μ² + a_μ²) v² |χ|²] + β² (v_μ² + a_μ²) a² |χ|² },
  χ(s) = s / [4 s_W² c_W² (s − m_Z² + i m_Z Γ_Z)],  β = √(1 − 4m²/s).

The (3−β²)/2 factor multiplies the vector–vector terms and β² the axial–axial term; the V–A cross terms integrate to zero. Check: the
massless μμ → ee ratio to σ_point = 4πα²/3s is 1.1290 numerically and 1 + 2v_μ² Re χ + (v_μ² + a_μ²)²|χ|² = 1.1290 analytically.
σ_point(10 TeV) = 0.996 fb with α(m_Z).

Couplings [certain]: the pure-Higgsino chargino is a Dirac fermion whose left- and right-handed components both sit in doublets with
T₃ = +½, Q = +1, so g_L = g_R = ½ − s_W² (vector-like, a = 0, v = 1 − 2s_W² = 0.538). The neutral Dirac Higgsino ψ⁰ = (χ₁ + iχ₂)/√2 has
g_L = g_R = −½; its vector current is purely off-diagonal, ψ̄γ^μψ = i χ̄₁γ^μχ₂, so χ₁χ₂ production equals Dirac ψψ̄ production. The Y = 1
triplet (P037's alternative) has (T₃, Q) = (1, 2), (0, 1), (−1, 0), i.e. g = 1 − 2s_W², −s_W², −1 for the three Dirac states.

Coupling factor for the chargino at 10 TeV: photon 1, γ–Z interference +0.057, Z² +0.144 (|χ| → 1.407); so σ(χ⁺χ⁻) ≈ 1.20 σ_point β(3−β²)/2.
Results (`muon_collider_xsec.csv`):

| m (GeV) | √s (TeV) | χ⁺χ⁻ (fb) | χ₁χ₂ (fb) | doublet total | Y=1 triplet total (Q=2, Q=1, neutral) | ratio | βγ of χ± |
|---|---|---|---|---|---|---|---|
| 300–1000 | 10 | 1.196–1.195 | 0.496 | 1.69 | 7.82 | 4.62 | 16.6–4.9 |
| 1100 | 10 | 1.195 | 0.495 | 1.69 | 7.81 | 4.62 | 4.43 |
| 2000 | 10 | 1.184 | 0.491 | 1.68 | 7.74 | 4.62 | 2.29 |
| 3000 | 10 | 1.129 | 0.468 | 1.60 | 7.38 | 4.62 | 1.33 |
| 4500 | 10 | 0.733 | 0.304 | 1.04 | 4.79 | 4.62 | 0.48 |
| 1100 | 3 | 11.5 | 4.76 | 16.2 | 75.0 | 4.62 | 0.93 |
| 1100 | 14 | 0.610 | 0.253 | 0.863 | 3.99 | 4.62 | 6.29 |

In 10 ab⁻¹ at 10 TeV: 11 950 χ⁺χ⁻ and 4 950 χ₁χ₂ events at 1.1 TeV (16 900 pairs), for any mass below ≈ 3 TeV (the cross-section is flat
in m until β drops). The Y = 1 triplet gives 4.6× more (78 100 pairs), 79 % of it from the doubly charged state (photon coupling 4×).

## 3. Kinematics and tracklet survival (Part B)

### 3.1 Why only the transverse βγ matters

The decay point lies at L = βγ cτ_exp along the flight direction; its transverse radius is r = L sin θ = (p_T/m) c τ_exp. Hence
P(r > r_min) = exp(−r_min m/(p_T cτ)) depends on βγ_T ≡ p_T/m only, which is invariant under the longitudinal boost of the pair system.
We therefore need only the p_T/m distribution of the chargino.

### 3.2 Toy Drell–Yan model (hadron colliders)

With dL/dτ ∝ τ^{−a} and σ̂ ∝ β(3−β²)/(2ŝ) (§2.2), the pair-mass spectrum is dσ/dM ∝ M^{−(2a+1)} β_M (3 − β_M²)/2 with β_M = √(1 − 4m²/M²),
and σ(m) ∝ m^{−2a}; we fix a = −½ d ln σ/d ln m from the anchor fit at the equivalent 13 TeV mass (a = 1.9–3.3 at 14 TeV for
m = 300–2000 GeV; 0.8–1.8 at 100 TeV). The pair-frame angular distribution for vector-coupled fermions is dσ/dcos θ* ∝ 2 − β_M² sin²θ*
(integrates to (4/3)(3−β²), consistent with §2.2), and βγ_T = (M β_M/2m) sin θ*. Pair p_T from ISR is neglected (noted as a limitation;
it would fatten the high-βγ_T tail somewhat). 2×10⁵ samples per point (`betagamma_T_quantiles.csv`):

| machine | m (GeV) | a | βγ_T 16 % | median | 84 % | 99 % | f(βγ_T > 2) | f(> 3) |
|---|---|---|---|---|---|---|---|---|
| HL-LHC 14 TeV | 300 | 1.89 | 0.28 | 0.61 | 1.19 | 2.96 | 0.038 | 0.0095 |
| HL-LHC 14 TeV | 1100 | 2.82 | 0.24 | 0.50 | 0.90 | 1.92 | 0.0082 | 0.0010 |
| FCC-hh 100 TeV | 1100 | 1.41 | 0.32 | 0.72 | 1.48 | 4.32 | 0.082 | 0.029 |
| FCC-hh 100 TeV | 2000 | 1.84 | 0.29 | 0.62 | 1.21 | 3.09 | 0.041 | 0.011 |

So at HL-LHC the median chargino has βγ_T ≈ 0.5 (P014's "βγ = 1" was already generous) and only 0.1 % exceed βγ_T = 3; at FCC-hh the
smaller τ makes the spectrum harder (median 0.72, 3 % above 3). At a 10 TeV muon collider βγ = √(s/4 − m²)/m is fixed by kinematics:
4.43 at 1.1 TeV, 2.29 at 2 TeV, 1.33 at 3 TeV; the transverse radius is βγ cτ sin θ with the same 2 − β² sin²θ angular law and a 10°
nozzle cut (acceptance 0.889).

### 3.3 Survival fractions

Higgsino cτ(m) is interpolated (log–log) from P014's `chargino_lifetimes.csv`: 0.953 / 0.811 / 0.712 / 0.664 / 0.641 cm at
300 / 500 / 1000 / 2000 / 4000 GeV (0.705 cm at 1.1 TeV). For the Y = 1 triplet's Q = 1 state, Γ ∝ κ²Δm³ with κ² = 2 and Δm = 511 MeV
(P037) gives cτ = 0.71 × (342/511)³/2 = 0.107 cm. (`tracklet_survival.csv`)

| machine | m (GeV) | cτ (cm) | P(r > 5 cm) | P(r > 10 cm) | P(r > 12 cm) | P(r > 30 cm) | triplet P(r > 5 cm) |
|---|---|---|---|---|---|---|---|
| HL-LHC | 300 | 0.953 | 1.0×10⁻² | 1.2×10⁻³ | 6.5×10⁻⁴ | 1.3×10⁻⁵ | 1.3×10⁻⁶ |
| HL-LHC | 500 | 0.811 | 4.0×10⁻³ | 3.2×10⁻⁴ | 1.5×10⁻⁴ | 1.4×10⁻⁶ | 3×10⁻⁷ |
| HL-LHC | 1100 | 0.705 | 1.1×10⁻³ | 4.7×10⁻⁵ | 1.8×10⁻⁵ | 3×10⁻⁸ | 1×10⁻⁸ |
| FCC-hh | 1100 | 0.705 | 9.9×10⁻³ | 1.4×10⁻³ | 7.5×10⁻⁴ | 1.2×10⁻⁵ | 7×10⁻⁶ |
| FCC-hh | 2000 | 0.664 | 4.0×10⁻³ | 3.8×10⁻⁴ | 1.9×10⁻⁴ | 1.9×10⁻⁶ | 1.5×10⁻⁶ |
| MuC 10 TeV | 500 | 0.811 | 0.418 | 0.191 | 0.142 | 0.011 | 3.8×10⁻³ |
| MuC 10 TeV | 1100 | 0.705 | 0.125 | 0.020 | 9.8×10⁻³ | 2×10⁻⁵ | 7.7×10⁻⁶ |
| MuC 10 TeV | 2000 | 0.664 | 0.019 | 5.3×10⁻⁴ | 1.3×10⁻⁴ | 6×10⁻¹⁰ | 3×10⁻¹⁰ |
| MuC 10 TeV | 3000 | 0.651 | 1.3×10⁻³ | 3×10⁻⁶ | 3×10⁻⁷ | — | — |

These are the population-averaged numbers behind P014's fixed-βγ statement (P(r > 12 cm) = 5×10⁻⁸ at βγ = 1): averaged over the toy spectrum
the HL-LHC value at 1.1 TeV is 1.8×10⁻⁵ because the 1 % high-βγ_T tail dominates the exponential. At the muon collider the fixed βγ = 4.4
raises P(r > 5 cm) to 12.5 %, a factor 10⁴ above HL-LHC — the single most important number of this paper.

### 3.4 Calibration of the selection efficiency

Following P014, the ATLAS 136 fb⁻¹ wino exclusion at 660 GeV [likely] is taken to correspond to ≈ 5 selected events [uncertain, ×2] out of
1.5 σ_C1N2 × 136 fb⁻¹ = 2466 produced C1N2 + C1C1 events with n_ch = 4/3 charginos per event. With the toy 13 TeV βγ_T spectrum at 660 GeV
(a = 2.51, median 0.53) and cτ_wino = 6.82 cm, P(r > 12 cm) = 0.077, so ε_kin = 5/(2466 × 4/3 × 0.077) = 0.020 (P014: 0.012 with βγ = 1).
ε_kin absorbs trigger (ISR jet + E_T^miss), tracklet quality, |η| and isolation requirements; it is applied unchanged to HL-LHC and FCC-hh,
which is optimistic for FCC-hh (pile-up 1000) and pessimistic for a dedicated HL-LHC trigger. For the muon collider we assume a 50 %
tracklet reconstruction efficiency after beam-induced-background (BIB) cuts [uncertain].

### 3.5 Expected tracklets and mass reach

S(m) = N_charginos(m) × ⟨P(r > r_min)⟩ × ε, with N_charginos = L × (σ_cn × 1 + σ_cn/3 × 2) at hadron colliders (χ±χ⁰ with one chargino,
χ⁺χ⁻ ≈ σ_cn/3 with two, χ₁χ₂ ≈ σ_cn/6 with none [uncertain split]) and 2 × σ(χ⁺χ⁻) × L at the muon collider. Recalled fake-tracklet
backgrounds [uncertain]: HL-LHC 30 (10–100), FCC-hh 100 (10–1000), MuC 3 (1–30). Reach = largest mass with S ≥ max(2√B, 3) (95 % CL) or
S ≥ max(5√B, 5) (5σ). (`reach_vs_mass.csv`, `expected_tracklets_vs_mass.csv`, `reach_xsec_band.csv`; Fig. 1)

| machine | r_min | S(300 GeV) | S(500) | S(1100) | 95 % CL reach, B central (low–high B) | 5σ reach | 95 % reach for σ ×/÷ band |
|---|---|---|---|---|---|---|---|
| HL-LHC 3 ab⁻¹ | 12 cm | 14.7 | 0.39 | 0.0011 | **300** (275–325) GeV | 250 GeV | 275–325 |
| HL-LHC 3 ab⁻¹ | 5 cm | 234 | 10.6 | 0.056 | **475** (450–525) GeV | 425 GeV | 425–550 |
| FCC-hh 30 ab⁻¹ | 10 cm | 3 560 | 1 870 | 55 | **1 300** (1 050–1 600) GeV | 1 100 GeV | 1 050–1 600 |
| FCC-hh 30 ab⁻¹ | 5 cm | 1.5×10⁴ | 8 730 | 396 | **2 050** (1 600–2 500) GeV | 1 700 GeV | 1 650–2 500 |
| MuC 10 ab⁻¹ | 10 cm | 4 930 | 2 290 | 238 | **2 100** (1 850–2 150) GeV | 1 900 GeV | 2 100–2 150 |
| MuC 10 ab⁻¹ | 5 cm | 7 540 | 5 000 | 1 490 | **3 350** (3 050–3 400) GeV | 3 150 GeV | 3 350–3 400 |

Reading: HL-LHC with 12 cm tracklets reaches 300 GeV, i.e. the LZ floor (259 GeV at δ = 300 keV) and not the 453 / 596 / 821 GeV floors at
δ = 350 / 366 / 380 keV, in line with the recalled literature (≈ 200–300 GeV; Fukuda et al. 2018, Mahbubani et al. 2017). A 5 cm
short-tracklet strategy (ITk innermost layer at 3.4 cm [uncertain]) would extend it to ≈ 475 GeV, still below the thermal point by a
factor 2.3 in mass. FCC-hh reaches 1.3 TeV (10 cm) to 2 TeV (5 cm) at 95 % CL, with 5σ at 1.1 TeV (10 cm) — the thermal Higgsino sits
exactly at the FCC-hh 5σ edge, and the ×/÷3 cross-section band moves the 95 % reach between 1.05 and 1.6 TeV (10 cm); Saito et al.'s
recalled ≈ 1 TeV discovery reach is reproduced in spirit. The muon collider yields 1 490 tracklets with r > 5 cm at 1.1 TeV (S/√B ≈ 860)
and discovers the doublet up to ≈ 3.1 TeV with 5 cm tracklets (1.9 TeV with 10 cm), covering every P007 fit point up to 3 TeV; the 4 TeV
fit point has βγ = 0.75 and P(r > 5 cm) = 10⁻⁵, so it would need a higher √s. The Y = 1 triplet (cτ = 0.11 cm) yields no tracklets
anywhere (P(r > 5 cm) ≤ 10⁻⁵) and must be sought through its 4.6× larger pair rate in mono-photon/soft-pion channels.

## 4. Mono-X (Part C)

Hadron-collider mono-jet reach for compressed electroweakinos is recalled only: ≈ 200 GeV at HL-LHC and ≈ 0.9 TeV at FCC-hh for the
Higgsino [uncertain; Low & Wang 2014] — both below the disappearing-track numbers above.

At the muon collider we estimate μ⁺μ⁻ → γ χχ with the collinear ISR structure function dP/dx dθ² = (α/2π)(1 + (1−x)²)/x /θ² integrated over
10°–170° (ln(1/θ_min²) = 3.49 per beam) and E_γ > 100 GeV [likely, factor 2 for wide-angle photons], convolved with §2.2 at the reduced
√s. Background: μ⁺μ⁻ → νν̄γ with σ(νν̄) ≈ 70 pb taken constant (t-channel W exchange; uncertain, factor 2), same photon cuts.
(`monophoton_muc.csv`)

| m (GeV) | σ(γ + χχ) (fb) | fraction of pair σ | S (10 ab⁻¹) | B (10 ab⁻¹) | S/√B |
|---|---|---|---|---|---|
| 500 | 0.163 | 0.097 | 1 630 | 3.9×10⁷ | 0.26 |
| 1100 | 0.139 | 0.082 | 1 390 | 3.8×10⁷ | 0.22 |
| 2000 | 0.118 | 0.070 | 1 180 | 3.8×10⁷ | 0.19 |

With inclusive cuts the mono-photon channel gives S/√B ≈ 0.2 for the thermal Higgsino: recoil-mass shape, beam polarisation and angular
information are needed to approach the recalled ≈ 95 % CL sensitivity [Han–Liu–Wang–Wang 2020; uncertain], and in no case does it rival
the 1 490 tracklets of §3.5. The disappearing-track channel is the discovery channel at a muon collider, as Capdevilla et al. (2021)
concluded [likely].

## 5. δ-blindness (Part D)

### 5.1 χ₂ decays in the laboratory

At δ = 366 keV the e⁺e⁻ channel is closed (δ < 2m_e = 1022 keV). P014 gives τ(χ₂ → χ₁γ) ≈ 0.06 s [uncertain, O(1) loop factor] and
τ(χ₂ → χ₁νν̄) = 4.6×10⁵ s. A χ₂ of Lorentz factor γ emits a photon whose lab energy is uniform in [γ(1−β)δ, γ(1+β)δ] with mean γδ
(`chi2_lab_decay.csv`):

| γ | case | E_lab range | mean | decay length (γ channel) | P(decay within 10 m) |
|---|---|---|---|---|---|
| 1.5 | slow | 140 keV – 0.96 MeV | 0.55 MeV | 2.0×10⁷ m | 5×10⁻⁷ |
| 3 | typical hadron-collider χ₂ | 63 keV – 2.1 MeV | 1.1 MeV | 5.0×10⁷ m | 2×10⁻⁷ |
| 4.55 | MuC, 1.1 TeV | 41 keV – 3.3 MeV | 1.7 MeV | 7.9×10⁷ m | 1.3×10⁻⁷ |
| 30 | extreme tail | 6 keV – 22 MeV | 11 MeV | 5.3×10⁸ m | 2×10⁻⁸ |

(The assignment's "boosted to ~GeV" would need γ ~ 3000; for realistic boosts the photon is 0.1–3 MeV.) Produced χ₂: 4 950 at the muon
collider (10 ab⁻¹) and 8.2×10⁵ at FCC-hh (30 ab⁻¹; ½σ_cn + σ_cn/6). In-detector photon decays: 6×10⁻⁴ and 0.16 over the whole
programmes — an MeV photon at a random point in a 100 TeV pile-up environment, once. The νν̄ channel is 10⁷ times longer still. δ is not
observable through χ₂ decays at any collider.

### 5.2 How δ enters the chargino splitting

P014 found Δm± = Δm_rad(m) + c_tree δ with c_tree = (0.1–2.1)×10⁻³ depending on the gaugino hierarchy and tan β. A collider measurement of
Δm± with uncertainty σ_Δm (dominated by the one-loop scheme spread ±10 MeV; a two-loop calculation plus a 1 % cτ measurement would give
≈ 4 MeV) bounds δ < σ_Δm/c_tree (`deltam_to_delta_bound.csv`):

| σ_Δm | wino-only, tan β = 2 | M₂ = 2M₁ | bino-only, tan β = 2 | bino-only, tan β = 10 | M₂ = −2M₁ | ratio to δ_LZ |
|---|---|---|---|---|---|---|
| 10 MeV | 104 GeV | 25 GeV | 11 GeV | 17 GeV | 4.7 GeV | (1.3–29)×10⁴ |
| 4.2 MeV | 44 GeV | 11 GeV | 4.8 GeV | 7.1 GeV | 2.0 GeV | (0.5–12)×10⁴ |
| 1.4 MeV | 15 GeV | 3.6 GeV | 1.6 GeV | 2.4 GeV | 0.67 GeV | (0.2–4)×10⁴ |

Even the aspirational row leaves δ unconstrained above ≈ 1 GeV, 2 000–40 000 times the LZ value. Conversely, the *observation* of
Δm± = 342 ± 10 MeV confirms gauginos heavier than ~10 TeV (P014: a few-TeV gaugino would shift Δm± by O(GeV) and make δ ~ MeV, killing
the LZ fit) — a necessary but far from sufficient condition. At the muon collider the tracklet radius distribution with known βγ gives
σ(cτ)/cτ ≈ 1/√1490 = 2.6 % and, through cτ ∝ Δm⁻³, σ(Δm±) ≈ 3 MeV: the experimental error will be below the theory error.

### 5.3 Precision electroweak (FCC-ee, CEPC, ILC)

A heavy vector-coupled fermion contributes to the oblique W parameter through the q⁴ term of the W³ self-energy. From the QED vacuum
polarisation Π̂(q²) − Π̂(0) = (α/15π) q²/m² + …, replacing e² → g² T₃² per Dirac fermion and using Ŵ = (g² m_W²/2) Π″₃₃(0)
[Barbieri et al. 2004; formula: likely]:

  Ŵ = g² m_W² Σ T₃² / (60 π² m²),  Σ T₃² = ½ (doublet: two Dirac fermions with T₃² = ¼), 2 (Y = 1 triplet).

Ŵ(doublet) = 2.6×10⁻⁵, 9.3×10⁻⁶, 1.9×10⁻⁶, 5.8×10⁻⁷ at m = 300, 500, 1100, 2000 GeV (`W_parameter.csv`). With a recalled FCC-ee/CEPC
sensitivity |W| ≈ 3×10⁻⁵ [uncertain; Di Luzio–Gröber–Panico 2018] the doublet is probed only below 278 GeV (481 GeV for 10⁻⁵); the
Y = 1 triplet below 556 (963) GeV. Direct pair production at 250–500 GeV lepton colliders stops at √s/2 = 125–250 GeV. None of these
machines touches the LZ region; the Ŷ parameter is of the same size and does not change the conclusion.

## 6. Identification logic (Part E)

What each side measures, and the precision that matters:

| observable | collider | direct detection |
|---|---|---|
| m | MuC: threshold and βγ of tracklets, ≲ 1 %; FCC-hh: mono-jet/tracklet shapes, ~10 % | only through δ_max(m): a 5 % (20 %) mass error moves δ(N=1) by 0.65 (2.6) keV (dδ/d ln m = 13 keV from P007), negligible against the halo systematic −17/+11 keV (P018) |
| multiplet (Y) | σ_prod: doublet 1.69 fb vs Y = 1 triplet 7.81 fb at 10 TeV (ratio 4.6): a 20 % measurement separates them at ln(4.6)/0.2 = 7.7σ; the triplet's Q = 2 state is a further discriminant | degenerate: (2Y)² only shifts δ(N=1) by +8 keV (P037) |
| Δm± | cτ from tracklet radii: 3 MeV at MuC; pion momentum p*_π = 312 MeV in principle | none |
| δ | none: bound δ ≲ 1–100 GeV (§5.2); χ₂ decays invisible (§5.1) | N(δ) falls ×5 per 10 keV (P007): σ_δ,stat = 6.2 keV/√N (6.2 / 3.1 / 2.0 / 1.1 keV for N = 1 / 4 / 10 / 30 events), systematic ±10–17 keV from the halo tail; annual modulation (P034) as the shape test |
| σ_n | no | fixed by G_F for Y = ½ (P007): the rate at known m and δ is a zero-parameter prediction; target scaling (P015) tests the Z-mediated nature |

The chain is therefore: (i) a collider discovers a charged state with cτ ≈ 0.7 cm and measures m to ≲ 5 % and σ_prod to ≲ 20 %, fixing
the multiplet; (ii) the identification of the LZ scatterer with that state then requires, from direct detection alone, that δ lies in the
window [δ(N = 3.65), δ_max(m)] of P007/P037 for that (m, Y) — 358–387 keV for a 1.1 TeV doublet — and that the rate tracks the G_F-fixed
prediction in exposure and season. No collider observable can ever confirm or refute the value of δ; the "inelastic" part of the
interpretation is direct-detection-only, while the "electroweak doublet" part is collider-only.

## 7. Timeline (`timeline_table.csv`)

| facility | years (recalled) | doublet reach | Y = 1 triplet | verdict |
|---|---|---|---|---|
| HL-LHC, 14 TeV, 3 ab⁻¹ | 2030–2041 (likely) | tracklets 300 GeV (12 cm) / 475 GeV (5 cm); mono-jet ≈ 200 GeV | no tracklets (cτ 0.11 cm); mono-jet only | at the LZ floor; cannot test the fit |
| FCC-ee / CEPC / ILC | 2045–2065 (uncertain) | W parameter m < 278 GeV; pairs < √s/2 | m < 556 GeV | no sensitivity |
| FCC-hh, 100 TeV, 30 ab⁻¹ | 2070s (likely) | 95 % CL 1.3 TeV (10 cm) / 2.05 TeV (5 cm); 5σ 1.1 / 1.7 TeV; mono-jet ≈ 0.9 TeV | 4.6× rate, mono-jet/soft pions | exclusion of the thermal Higgsino plausible; discovery marginal |
| muon collider, 10 TeV, 10 ab⁻¹ | 2050s at the earliest (uncertain) | 16 900 pairs; 1 490 tracklets (r > 5 cm) at 1.1 TeV; 5σ to 3.1 TeV; m, Y, Δm± measured | 78 100 pairs; mono-photon / soft pions; no tracklets | full coverage of the P007 points ≤ 3 TeV; blind to δ |

## 8. Figures

- `figures/P048_fig1_reach_vs_mass.png` — expected disappearing-track events versus doublet mass for HL-LHC (12 / 5 cm), FCC-hh (10 / 5 cm)
  and the 10 TeV muon collider (10 / 5 cm), with the 5√B lines for the central backgrounds, the LZ floor (259 GeV), the thermal mass and
  the five P007 fit points. The 5 cm curves lie 1–2 decades above the 10–12 cm curves because cτ = 0.71 cm sits in the exponential regime.
- `figures/P048_fig2_survival_betagamma.png` — left: P(r > r_min) versus βγ_T for the Higgsino (r_min = 5, 10, 12, 30 cm), the wino
  (12 cm) and the Y = 1 triplet (5 cm), with the muon-collider βγ = 4.4 marked; right: toy Drell–Yan βγ_T distributions at 14 and 100 TeV
  for 300 and 1100 GeV, and the fixed muon-collider value.

## 9. Validation and robustness

- Cross-section machinery reproduces the P014 13 TeV anchors to 0.5 % and the familiar 14/13 TeV ratio (1.2–1.4).
- The μμ → ff̄ formula reproduces the analytic massless ratio 1.1290 and σ_point = 0.996 fb at 10 TeV.
- The wino calibration gives ε_kin = 0.020 versus P014's 0.012 (different βγ treatment); a factor 2 on ε_kin acts like the σ ×2 band and
  shifts the hadron-collider reaches by ≲ 10 % in mass (HL-LHC 12 cm: 300 → 325 GeV).
- Backgrounds ×/÷3–10 move the 95 % CL reaches by ±8 % (HL-LHC 12 cm; −5/+11 % at 5 cm), −19 to +23 % (FCC-hh) and −12 to +2 % (MuC); the
  cross-section band moves FCC-hh by −19 to +23 %; MuC is insensitive (tree level, β(3−β²)/2 flat).
- Neglecting pair p_T (ISR) underestimates the high-βγ_T tail at hadron colliders; since the tail carries the signal this makes our
  HL-LHC/FCC-hh yields conservative by an O(1) factor, not by decades.
- The 100 TeV τ-scaling is frozen below m = 460 GeV; no reach statement uses that region.

## 10. Failed or abandoned approaches

- First run failed on a hyphen inside a dictionary keyword (renamed).
- A parton-level PDF convolution was not attempted (no PDF sets offline); the τ-scaling and power-law luminosity are the substitutes.
- A full mono-photon analysis with recoil-mass and polarisation cuts was not attempted; the inclusive estimate is reported with its S/√B.
- The figure-2 muon-collider label needed two placement iterations (hidden by the legend, then clipped by the axes).

## 11. Result files

`hadron_xsec_pairs.csv`, `muon_collider_xsec.csv`, `betagamma_T_quantiles.csv`, `tracklet_survival.csv`, `reach_vs_mass.csv`,
`reach_xsec_band.csv`, `expected_tracklets_vs_mass.csv`, `monophoton_muc.csv`, `chi2_lab_decay.csv`, `deltam_to_delta_bound.csv`,
`W_parameter.csv`, `timeline_table.csv`, `P048_summary.json`, `recalled_knowledge.json`, `run_log.txt`, two figures.

## 12. Discussion

The LZ-fitting doublet has a lifetime that is wrong for every existing tracker and right for a muon collider. At hadron colliders the
chargino is produced near threshold (median βγ_T ≈ 0.5–0.7) and only the 1–3 % tail above βγ_T ≈ 3 survives past 5–10 cm, so the reach is
driven by luminosity and by how close to the beam pipe a tracklet can be reconstructed: HL-LHC stops at 300–475 GeV, FCC-hh reaches
1.3–2 TeV. At √s = 10 TeV the chargino is born with βγ = 4.4 and 12.5 % of the LZ-fitting charginos travel beyond 5 cm; with 10 ab⁻¹ this is
1 490 tracklets against a handful of fakes, discovery to 3 TeV and a 1 % measurement of cτ, i.e. a 3 MeV measurement of Δm±. What no
machine will measure is δ: its decay products are MeV photons emitted 10⁷ m downstream, and its imprint on Δm± is 10⁴ below the theory
error. The corpus' inelastic interpretation therefore splits cleanly into a collider half (m, Y, Δm±) and a direct-detection half
(δ, σ_n, modulation); the collider half can be completed in the 2050s–2070s, the direct-detection half needs the exposures of P020/P034.
The Y = 1 triplet of P037 is the mirror case: 4.6× the production, but cτ = 0.11 cm and no tracklets anywhere.

## 13. References

1. LZ Collaboration, arXiv:2609.02823 (2026).
2. H. Fukuda, N. Nagata, H. Otono, S. Shirai, Phys. Lett. B 781, 306 (2018) — Higgsino disappearing tracks at HL-LHC (recalled).
3. R. Mahbubani, P. Schwaller, J. Zurita, JHEP 06 (2017) 119 — HL-LHC Higgsino prospects (recalled).
4. H. Saito, R. Sawada, K. Terashi, S. Asai, Eur. Phys. J. C 79, 469 (2019) — 100 TeV disappearing tracks (recalled).
5. M. Low, L.-T. Wang, JHEP 08 (2014) 161 — electroweakinos at 14 and 100 TeV (recalled).
6. R. Capdevilla, F. Meloni, R. Simoniello, J. Zurita, JHEP 06 (2021) 133 — wino/Higgsino disappearing tracks at a muon collider (recalled).
7. T. Han, Z. Liu, L.-T. Wang, X. Wang, Phys. Rev. D 103, 075004 (2021), arXiv:2009.11287 — WIMPs at high-energy muon colliders (recalled).
8. R. Barbieri, A. Pomarol, R. Rattazzi, A. Strumia, Nucl. Phys. B 703, 127 (2004) — W, Y parameters (recalled).
9. L. Di Luzio, R. Gröber, G. Panico, JHEP 01 (2019) 011 — EW states via precision measurements (recalled).
10. ATLAS Collaboration, Eur. Phys. J. C 82, 606 (2022) — disappearing-track search (recalled, via P014).
11. Corpus: P002, P007, P014, P015, P018, P020, P025, P034, P037, dossier 00.

## 14. Tools and provenance (mirrors `output/provenance/P048.json`)

- Agent tools: Read (PAPER_GUIDE.md; 00_evidence_dossier.md; results_ledger.csv in three pages; papers P014, P007, P037, P025;
  work/P014/details.md; provenance/P014.json; the two output figures, three iterations), Bash (directory listings, environment versions,
  P014 CSV/JSON dump, lzcommon grep, palette grep, five script runs, one word count), Write (script, details.md, P048.json, P048.md),
  Edit (script: keyword rename, clipping flag, check message, cross-section band, two figure-label placements; paper trimming),
  Skill (dataviz).
- Software: python 3.12.13; numpy 2.5.3 (polyfit/polyval, percentile, random.default_rng, trapezoid); scipy 1.18.1 (integrate.quad);
  pandas 3.0.5; matplotlib 3.11.2 (Agg); common/lzcommon.py (GEV_TO_CM2). Hand derivations: τ-scaling identity, (3−β²)/2 and 2 − β²sin²θ
  structure, Majorana off-diagonal current, W-parameter one-loop coefficient, transverse-radius argument.
- Recalled knowledge: 21 items in `recalled_knowledge.json` (constants certain; textbook σ(ff̄) certain; P014 anchors and channel split
  likely; τ-scaling likely; 100 TeV literature values, tracklet radii, backgrounds, efficiencies, ISR wide-angle log, σ(νν̄), FCC-ee
  sensitivity and all dates uncertain; corpus values from P007/P014/P018/P037).
- Datasets: none. Data requests: none. WimPyDD-generated files: none.
