# P014 · Collider constraints on electroweak-multiplet inelastic dark matter after the LZ event — research record

Simulated date 2026-09-05. Author profile: collider phenomenologists. Category COMP, hep-ph. Competes with P048 (later).
Script: `output/code/P014_collider_constraints.py` (run from the simulation root with `.venv/bin/python`). Every number below is printed in
`output/work/P014/run_log.txt` or stored in the CSV/JSON files listed in §10. Recalled experimental bounds are flagged in §6 and in
`recalled_knowledge.json`; none was fetched.

## 1. Motivation and framework

P007 showed that a pure Higgsino doublet reproduces one LZ event only for a neutral splitting δ ≈ 358–380 keV at μ = 1 TeV
(336–351 keV at 500 GeV, 366–394 keV at 4 TeV), which requires bino/wino masses of 3–30 PeV; P002 showed that a 248 keV recoil with
δ ≥ 300 keV needs m_χ ≥ 259 GeV (453 GeV at δ = 350 keV). The question for collider physicists is whether any LEP or LHC search
already excludes, or could soon test, those masses and splittings. Two facts organise the answer:

1. The collider-visible splitting is not δ (a 10⁻⁴ GeV effect between two neutral Majorana states) but the chargino–neutralino
   splitting Δm± = m(χ±) − m(χ₁), which is dominated by the one-loop electroweak term and is O(300 MeV). It fixes the chargino
   lifetime and therefore which search topology (disappearing track, soft lepton, monojet) applies.
2. The Higgsino masses that fit LZ start at 259 GeV, above every LHC Higgsino limit ever set (≤ 250 GeV), so the problem is one of
   *reach*, not of *exclusion*.

We (i) compute Δm±(M) for the doublet and the triplet, (ii) add the tree-level O(δ) piece from P007's gaugino table, (iii) derive the
chargino widths and lifetimes, (iv) recall the LEP/LHC bounds with reliability flags and map them onto the P002/P007 region,
(v) estimate produced pairs at the LHC from recalled cross-sections and (vi) compute disappearing-track survival probabilities to show
why the natural Higgsino lifetime escapes those searches. A short section treats the excited state χ₂.

## 2. One-loop chargino–neutralino splitting (Part A)

For a heavy fermion coupled to a vector boson of mass m_V with coupling g_V, the on-shell self-energy in Feynman gauge gives a mass
shift whose m_V dependence is (Peskin–Schroeder eq. 7.30 structure after x → 1−x)

  δM_V = −(α_V M/4π) f(m_V/M) + const,  f(r) = 2∫₀¹ dx (1+x) ln[x² + (1−x) r²],  α_V = g_V²/4π.

Because Σ_V g_V² is the same for every member of a multiplet (gauge invariance), the divergent constant cancels in mass differences and
only f̃(r) ≡ f(r) − f(0) matters (f(0) = −5). We evaluate f̃ numerically (`scipy.integrate.quad`). Checks: f̃(r)/(2πr) = 0.9995,
0.9953, 0.9555, 0.8812 at r = 10⁻³, 10⁻², 0.1, 0.3, i.e. f̃ → 2πr for r → 0 as required, so δM_V → −α_V m_V/2 in the heavy limit.
Summing photon, Z and W loops with the couplings of each component gives the Cirelli–Fornengo–Strumia form
Δm = (α₂M/4π){(Q²−Q'²)s_W² f̃(m_Z/M) + (Q−Q')(Q+Q'−2Y)[f̃(m_W/M) − f̃(m_Z/M)]} [recalled: likely], which reduces to

- doublet (Y = ½, Q = 1 vs 0):  Δm_H = (α M/4π) f̃(m_Z/M) → α m_Z/2 = **356.3 MeV** (α(m_Z) = 1/127.95; with α(0) = 1/137.036: 332.7 MeV);
  no dependence on s_W (α₂ s_W² = α). Goldstone loops vanish because the Higgsino mass μ is gauge invariant.
- triplet (Y = 0):  Δm_W = (α₂M/4π)[f̃(m_W/M) − c_W² f̃(m_Z/M)] → α₂ m_W (1 − c_W)/2 = **161.1 MeV** with on-shell
  c_W = m_W/m_Z (167.4 MeV with MSbar c_W² = 0.7688). We adopt on-shell c_W in the W/Z-difference term because the physical
  combination is c² m_Z → m_W²/m_Z.

Results (`radiative_splitting_vs_mass.csv`):

| M (GeV) | 100 | 200 | 300 | 500 | 1000 | 2000 | 4000 |
|---|---|---|---|---|---|---|---|
| Δm_rad Higgsino (MeV) | 258.0 | 296.9 | 313.5 | 328.9 | 341.8 | 348.8 | 352.5 |
| Δm_rad wino, on-shell (MeV) | 147.2 | 155.8 | 158.2 | 159.9 | 160.7 | 161.0 | 161.0 |

Scheme systematics: Higgsino at 1 TeV 341.8 MeV (α(m_Z)) vs 319.1 MeV (α(0)); wino at 1 TeV 160.7 (on-shell) vs 172.8 MeV (MSbar).
The literature two-loop wino value ≈ 164 MeV [Ibe–Matsumoto–Sato 2013, recalled: likely] lies between our two one-loop schemes, and
the usual "≈ 340–355 MeV" Higgsino quotes bracket our 342 MeV (1 TeV) and 356 MeV (asymptote). We therefore assign ±10 MeV to the
Higgsino Δm±, as the assignment anticipated, and use the computed on-shell wino curve with 165 MeV as the reference lifetime point.

## 3. Tree-level piece and the total Δm± at the P007 fit points (Part B) — the computed δ-independence

P007's `gaugino_masses_for_splitting.csv` tabulates, from the full 4×4 neutralino and 2×2 chargino diagonalisation, the tree-level
m(χ±) − m(χ₁) for four gaugino hierarchies (bino-only, M₂ = 2M₁, wino-only, M₂ = −2M₁), tan β = 2 and 10, μ = 300/1000/4000 GeV
and δ = 300–385 keV. The ratio (tree piece)/δ is constant for each hierarchy (both scale as m_Z²/M):

| hierarchy | tan β = 2 | tan β = 10 |
|---|---|---|
| bino only | 0.885 × 10⁻³ | 0.595 × 10⁻³ |
| M₂ = 2M₁ | 0.395 × 10⁻³ | 0.474 × 10⁻³ |
| wino only | 0.096 × 10⁻³ | 0.400 × 10⁻³ |
| M₂ = −2M₁ | 2.126 × 10⁻³ | 0.902 × 10⁻³ |

so the tree piece is (0.1–2.1) × 10⁻³ δ = **0.03–0.8 MeV** for δ = 300–380 keV. Totals at the fit points (`P014_results.json`,
`fit_points`):

| μ (GeV) | δ(N=1) (keV) | Δm_rad (MeV) | tree (MeV) | **Δm± (MeV)** |
|---|---|---|---|---|
| 300 | 310 | 313.5 | 0.03–0.66 | 313.9 |
| 500 | 342 | 328.9 | 0.03–0.73 | 329.2 |
| 1000 | 366 | 341.8 | 0.04–0.78 | 342.2 |
| 2000 | 375 | 348.8 | 0.04–0.80 | 349.3 |
| 4000 | 376 | 352.5 | 0.04–0.80 | 352.9 |

Scanning δ = 100, 200, 300, 366, 400 keV at 1 TeV with the *largest* tree coefficient gives Δm± = 342.01, 342.22, 342.44, 342.58,
342.65 MeV: a **0.19 % change over the whole δ range**, far below the ±3 % scheme uncertainty. The relation is
δ ≃ m_Z²(s_W²/M₁ + c_W²/M₂) (P007) while Δm± ≃ α m_Z/2 + O(m_Z²/M): the gaugino mass that LZ fixes to the PeV scale enters Δm± only
through the second term, at the 10⁻³ level. Conclusion: **collider limits on the LZ Higgsino are independent of δ**, and conversely the
LZ fit *predicts* the chargino splitting, Δm± = 342 ± 10 MeV at 1 TeV. Had the gauginos been at a few TeV (Δm± shifted by O(GeV)),
δ would be O(MeV) and the LZ event impossible, so the two observables are rigidly linked.

## 4. Chargino widths and lifetimes (Part C)

Integrating out the W between the chargino current and the quark or lepton current gives L_eff = 2√2 G_F ⟨T⁺⟩ (χ̄⁰γ^μχ⁺)(ū γ_μ P_L d)
+ h.h., with ⟨T⁺⟩ = 1 for the doublet (⟨+½|T⁺|−½⟩) and √2 for the triplet. For the Higgsino the neutral Dirac state is
ψ = (χ₁ + iχ₂)/√2, so each Majorana state couples with g/2 and both final states are open (δ ≪ Δm±): the summed rate equals the Dirac
result, κ² ≡ ⟨T⁺⟩² = 1 (doublet), 2 (triplet). The currents are pure vector, so the pion matrix element ⟨0|d̄γ_μγ₅u|π⟩ = i f_π p_μ
contracted with ū₀γ^μu₊(p₊ − p₀)_μ = Δm ū₀u₊ gives, in the heavy limit (|p_π| → √(Δm² − m_π²)),

  Γ(χ± → χ⁰π±) = κ² (G_F² f_π² |V_ud|²/π) Δm³ (1 − m_π²/Δm²)^{1/2},   f_π = 130.2 MeV.

For the leptonic channel the heavy current reduces to its time component (2M δ_ss′) and the three-body phase space gives

  Γ(χ± → χ⁰ ℓ± ν) = κ² (G_F²/15π³) Δm⁵ I(m_ℓ/Δm),  I = (30/Δm⁵)∫₀^{p_max} p² (Δm − √(p² + m_ℓ²))² dp  (I = 1 for m_ℓ = 0).

Both prefactors were derived here; they coincide with the standard Chen–Drees–Gunion expressions [recalled: likely] and are validated
by the wino: at Δm = 165 MeV, τ = 0.193 ns, cτ = 5.77 cm, BR(π) = 0.977, BR(e) = 0.021, BR(μ) = 0.002 — the textbook ~0.2 ns, ~6 cm,
97.7 %. Kaon and multi-pion channels are closed or negligible below 400 MeV. Results (`chargino_lifetimes.csv`):

| model | M (GeV) | Δm± (MeV) | BR(π) | BR(e) | BR(μ) | τ (ns) | cτ (cm) | cτ for Δm ∓ 10 MeV |
|---|---|---|---|---|---|---|---|---|
| Higgsino | 300 | 313.9 | 0.932 | 0.043 | 0.025 | 0.0318 | **0.953** | 1.064 / 0.857 |
| Higgsino | 500 | 329.2 | 0.925 | 0.047 | 0.028 | 0.0270 | **0.811** | 0.900 / 0.733 |
| Higgsino | 1000 | 342.2 | 0.919 | 0.050 | 0.031 | 0.0237 | **0.712** | 0.787 / 0.646 |
| Higgsino | 2000 | 349.3 | 0.916 | 0.051 | 0.033 | 0.0222 | **0.664** | 0.733 / 0.604 |
| Higgsino | 4000 | 352.9 | 0.914 | 0.052 | 0.034 | 0.0214 | **0.641** | 0.707 / 0.584 |
| wino (on-shell) | 200 / 660 / 1000 / 2900 | 155.8 / 160.3 / 160.7 / 161.0 | 0.976–0.977 | 0.021 | 0.002 | 0.275 / 0.228 / 0.224 / 0.222 | 8.2 / 6.8 / 6.7 / 6.65 | — |
| wino (165 MeV) | — | 165 | 0.977 | 0.021 | 0.002 | 0.193 | 5.77 | — |

cτ ∝ Δm^{-3}: the ±10 MeV uncertainty on the Higgsino splitting is ∓10 % on cτ; the α(0) scheme (319 MeV at 1 TeV) would give
≈ 0.87 cm. Everything the LZ fit allows sits at **cτ ≈ 0.6–1.0 cm, τ ≈ 0.02–0.03 ns**, a factor 6–9 shorter than the wino.

## 5. Production at the LHC (Part D) — recalled cross-sections, flagged

Anchors [recalled: likely, factor ~1.5]: pure-wino χ̃₁±χ̃₂⁰ at 13 TeV, NLO+NLL (LHC SUSY cross-section WG): 12 pb (100 GeV),
1.8 (200), 0.39 (300), 0.12 (400), 0.046 (500), 0.020 (600), 0.0046 (800), 0.0013 pb (1000 GeV). A quadratic in ln m fits them to
≤ 4 % with local slopes d ln σ/d ln m = −3.9 (300 GeV), −5.6 (1 TeV), −6.6 (2 TeV, extrapolated). For the Higgsino the χ±χ⁰ channels
carry ½ of the wino χ̃₁±χ̃₂⁰ rate at equal mass (W coupling g/2 per Majorana neutral state versus g; derived); χ⁺χ⁻ and χ₁χ₂ add roughly
50 % more [uncertain]. We quote the χ±χ⁰ channels as central and treat the total as ×/÷2. The assignment's alternative parametrisation
(20 fb at 300 GeV, 1 fb at 600, 0.1 fb at 1 TeV) is 5–10 times lower than our recollection; we carry it as a pessimistic curve
(`sigma_alt_fb`) and note that no conclusion changes between the two. (`production_and_pair_counts.csv`)

| m (GeV) | σ wino C1N2 (fb) | σ Higgsino χ±χ⁰ (fb) | total ≈ | alt (fb) | pairs 140 fb⁻¹ | 300 fb⁻¹ | 3000 fb⁻¹ |
|---|---|---|---|---|---|---|---|
| 300 | 403 | **201** | 322 | 20.3 | 2.8 × 10⁴ | 6.0 × 10⁴ | 6.0 × 10⁵ |
| 500 | 46.0 | **23.0** | 36.8 | 2.15 | 3.2 × 10³ | 6.9 × 10³ | 6.9 × 10⁴ |
| 660 | 12.1 | 6.0 | 9.7 | 0.63 | 850 | 1.8 × 10³ | 1.8 × 10⁴ |
| 1000 | 1.33 | **0.67** | 1.07 | 0.10 | 93 | 200 | 2.0 × 10³ |
| 1100 (extrap.) | 0.78 | 0.39 | 0.62 | 0.067 | 54 | 116 | 1.2 × 10³ |
| 2000 (extrap.) | 0.019 | 0.010 | 0.015 | 0.005 | 1.4 | 2.9 | 29 |

With the pessimistic curve the 1 TeV counts are 14 / 31 / 306. So HL-LHC produces O(10³) thermal-Higgsino pairs; the issue is entirely
whether any of them can be *seen*.

## 6. Recalled constraints and their applicability (Part E, F)

| bound | value | reliability | applies to the LZ Higgsino? |
|---|---|---|---|
| LEP2 chargino | m(χ±) > 92 GeV for Δm ≳ 0.1 GeV; 103.5 GeV at large Δm | certain | yes, but 92 GeV ≪ 259 GeV floor |
| ATLAS disappearing track, 136 fb⁻¹ (EPJC 82 (2022) 606) | wino excluded < 660 GeV; with Higgsino cross-section < 210 GeV; both quoted at τ = 0.2 ns | likely (masses); "both at 0.2 ns": likely | **no**: natural τ = 0.024 ns (§7) |
| CMS disappearing track, Run 2 | wino ≈ 880 GeV at τ = 3 ns, ≈ 470 GeV at 0.2 ns | uncertain | no |
| soft-lepton compressed searches (ATLAS 139 fb⁻¹ and update, CMS) | Higgsino < 210–250 GeV for Δm ≈ 5–30 GeV | likely | **no**: leptons have p ≲ 0.35 GeV |
| monojet-type limits on degenerate electroweakinos | ≲ 100–150 GeV | uncertain | no (m > 259 GeV) |
| thermal masses | Higgsino 1.1 TeV, wino 2.7–3.0 TeV | certain / likely | targets |

Kinematic floors recomputed with `lz.m_chi_min_gev` on 16 June (v_max = 809.1 km/s): 259 / 453 / 596 / 821 GeV at δ = 300 / 350 /
366 / 380 keV, identical to P002. Constraint map (`constraint_map.csv`):

| model | masses | LEP | disappearing track | soft lepton | monojet | verdict |
|---|---|---|---|---|---|---|
| Higgsino (doublet) | 300, 500, 1000, 1100, 2000, 4000 GeV | allowed | unprobed (cτ = 0.71 cm, P(r > 12 cm) = 5 × 10⁻⁸ at βγ = 1) | not applicable | unprobed | **allowed / unprobed by every collider search** |
| wino (triplet) | 300, 500, 660 | allowed | excluded (ATLAS < 660 GeV) | n/a | unprobed | excluded |
| wino (triplet) | 1000, 2000, 2900 | allowed | beyond reach | n/a | unprobed | allowed / unprobed |

A remark on the "wino-like" case: a Y = 0 multiplet's neutral component has Z coupling (g/c_W)(T₃ − Q s_W²) = 0, and the MSSM wino is a
single Majorana neutral state with no O(100 keV) partner; a Z-mediated inelastic LZ signal therefore needs Y ≠ 0 (the doublet) or a
new mediator. The triplet is included as the reference collider case, not as an LZ candidate.

## 7. Why disappearing-track searches miss the natural Higgsino (Part E, computed)

An ATLAS pixel tracklet needs four pixel hits, i.e. a decay radius r ≳ 12 cm [recalled: likely]; CMS-style tracks need r ≳ 30 cm
[uncertain]. With P(r > r_min) = exp(−r_min/(βγ cτ)) (`tracklet_survival.csv`):

| r_min | βγ | P(Higgsino, cτ = 0.71 cm) | P(wino, cτ = 6.8 cm) | ratio |
|---|---|---|---|---|
| 12 cm | 0.5 | 2.3 × 10⁻¹⁵ | 0.030 | 8 × 10⁻¹⁴ |
| 12 cm | 1.0 | **4.8 × 10⁻⁸** | 0.17 | 3 × 10⁻⁷ |
| 12 cm | 2.0 | 2.2 × 10⁻⁴ | 0.42 | 5 × 10⁻⁴ |
| 12 cm | 3.0 | 3.6 × 10⁻³ | 0.56 | 7 × 10⁻³ |
| 30 cm | 1.0 | 4.9 × 10⁻¹⁹ | 0.012 | 4 × 10⁻¹⁷ |

Reaching P = 10⁻³ at 12 cm needs βγ = 2.4, i.e. p_T ≈ 2.4 m — a rare ISR-boosted tail. Toy calibration: the ATLAS wino exclusion at
660 GeV (136 fb⁻¹) corresponds to ≈ 2470 produced C1N2 + C1C1 events (1.5 × σ_C1N2); assuming the exclusion needed ≈ 5 selected events
gives ε_tot ≈ 2 × 10⁻³ and, dividing out P_wino(βγ = 1) = 0.17, a kinematic/selection efficiency ε_kin ≈ 0.012. Applying ε_kin and
P_Higgsino to HL-LHC (3000 fb⁻¹, total σ, 2 charginos per event as an upper bound; `toy_tracklet_yield.csv`):

| m (GeV) | tracklets if every chargino has βγ = 1 | … βγ = 2 |
|---|---|---|
| 200 | 5 × 10⁻³ | 21 |
| 300 | 1 × 10⁻³ | 5.0 |
| 500 | 1 × 10⁻⁴ | 0.57 |
| 1000 | 4 × 10⁻⁶ | 0.016 |

Even the unrealistic assumption that every chargino carries βγ = 2 yields ≲ 5 tracklets at 300 GeV and 0.02 at 1 TeV: the standard
tracklet strategy cannot reach the LZ region at HL-LHC. This is consistent with the literature, which finds HL-LHC Higgsino reach of
≈ 200–300 GeV only with shorter tracklets/dedicated triggers [Fukuda–Nagata–Otono–Shirai 2018; Mahbubani–Schwaller–Zurita 2017;
recalled: uncertain], FCC-hh (100 TeV, 30 ab⁻¹) discovery reach ≈ 1 TeV for the Higgsino and ≈ 3 TeV for the wino [Saito et al. 2019;
uncertain], and full coverage of both thermal targets at a 10 TeV muon collider [Capdevilla et al. 2021; likely].

## 8. The excited state χ₂ at colliders (Part G)

At the LZ-favoured δ = 310–376 keV: (i) δ < 2m_e = 1022 keV, so χ₂ → χ₁e⁺e⁻ is **closed** (the assignment's "soft e⁺e⁻" does not exist);
(ii) χ₂ → χ₁νν̄ through the Z: L_eff = √2 G_F (χ̄₁γ^μχ₂)(ν̄γ_μP_Lν) per flavour gives Γ = G_F²δ⁵/(20π³) summed over three flavours
(derived; ¼ of the chargino–lepton coefficient per flavour), τ = 1.1 × 10⁶, 6.4 × 10⁵, 4.6 × 10⁵, 4.1 × 10⁵, 4.0 × 10⁵ s at the five fit
points — consistent with P007's order-of-magnitude 10⁶ s; (iii) χ₂ → χ₁γ through the one-loop transition dipole: the Dirac Higgsino's
loop moment ψ̄σ^{μν}ψ F_{μν} becomes i χ̄₁σ^{μν}χ₂ F_{μν} for Majorana components, and with a Schwinger-like estimate
μ₁₂ ≈ (α₂/2π) e/(2μ) = 8 × 10⁻⁷ GeV⁻¹ (1 TeV) and Γ = μ₁₂²δ³/π [recalled: likely] we get τ_γ ≈ 0.06 s (0.009 s at 300 GeV, 0.9 s at
4 TeV) [uncertain to an O(1) loop factor]. If correct, the photon channel dominates and P007's 10⁶ s should be read as an upper bound;
either way χ₂ is stable over any detector transit (10⁻⁸ s) and decays long before reaching the halo age, so it is invisible at
colliders and absent as a relic. The only collider handle on the LZ splitting is therefore indirect: Δm± is fixed at 342 ± 10 MeV.

## 9. Figures

- `figures/P014_fig1_mass_vs_ctau.png` — chargino cτ versus multiplet mass: computed Higgsino (blue) and wino (orange) curves, the five
  P007 fit points, the LEP band, the LZ mass floor (259 GeV), the ATLAS disappearing-track exclusions drawn at the lifetime they assume
  (τ = 0.2 ns), the 12 cm pixel-tracklet radius, thermal masses, and recalled prospects. The Higgsino curve lies an order of magnitude
  below the tracklet radius everywhere.
- `figures/P014_fig2_pairs_vs_mass.png` — produced Higgsino χ±χ⁰ pairs versus mass for 140, 300 and 3000 fb⁻¹ (central recalled
  cross-sections, ×/÷2 band, pessimistic parametrisation dashed), with the LZ floor and the thermal mass marked.

## 10. Result files

`P014_results.json` (all numbers), `radiative_splitting_vs_mass.csv`, `chargino_lifetimes.csv`, `production_and_pair_counts.csv`,
`tracklet_survival.csv`, `toy_tracklet_yield.csv`, `constraint_map.csv`, `chi2_decays.csv`, `recalled_knowledge.json`, `run_log.txt`.

## 11. Failed or abandoned approaches

- First run failed on a Python keyword containing a hyphen in the summary dictionary (renamed).
- The wino splitting was first evaluated with the MSbar cos²θ_W in the W/Z-difference term (173 MeV at 1 TeV, 5 % above the
  literature); switched to on-shell c_W = m_W/m_Z (161 MeV) and kept the MSbar value as the scheme systematic.
- Figure 1 annotations needed three placement iterations to avoid label collisions.
- A parton-level βγ spectrum for Drell–Yan charginos was not attempted (no PDFs available offline); fixed representative βγ values are used
  and the toy yield is labelled an upper bound.

## 12. Discussion

The LZ Higgsino occupies a corner that colliders have not touched. LEP is irrelevant (92 GeV against a 259 GeV floor). The LHC has
two Higgsino strategies, both tuned to a different splitting than the one LZ implies: soft-lepton searches need Δm ≳ 5 GeV, and
disappearing-track searches need cτ ≳ few cm, whereas the PeV-gaugino spectrum that produces δ ≈ 360 keV forces Δm± onto its
radiative value, 342 ± 10 MeV, and cτ ≈ 0.7 cm. The published "Higgsino to 210 GeV" tracklet limit is quoted at τ = 0.2 ns, eight
times the natural lifetime, and does not transfer. Monojet searches stop far below 259 GeV. HL-LHC will produce ~2000 thermal-Higgsino
pairs but, with 12 cm tracklets, will identify none; the literature's HL-LHC Higgsino projections (≲ 300 GeV) barely cross the LZ
floor and do not approach the 1.1 TeV thermal point. Only a 100 TeV hadron collider or a 10 TeV muon collider reaches it. Two
positive statements survive: the LZ fit makes a sharp collider prediction (a Higgsino chargino with τ ≈ 0.02–0.03 ns, BR(π) ≈ 0.92,
BR(e) ≈ 0.05, BR(μ) ≈ 0.03), so a future Higgsino discovery with τ ≈ 0.2 ns would refute the PeV-gaugino LZ interpretation; and the
triplet alternative is not an LZ candidate at all without a new mediator, while where it is a collider candidate (> 660 GeV) it is as
unprobed as the doublet.

## 13. References

1. LZ Collaboration, arXiv:2609.02823 (2026).
2. S. D. Thomas, J. D. Wells, Phys. Rev. Lett. 81, 34 (1998) — radiative doublet splitting.
3. M. Cirelli, N. Fornengo, A. Strumia, Nucl. Phys. B 753, 178 (2006) — general multiplet splitting formula (recalled).
4. M. Ibe, S. Matsumoto, R. Sato, Phys. Lett. B 721, 252 (2013) — two-loop wino splitting (recalled).
5. C.-H. Chen, M. Drees, J. F. Gunion, Phys. Rev. D 55, 330 (1997) — chargino decay widths (recalled).
6. ATLAS Collaboration, Eur. Phys. J. C 82, 606 (2022), arXiv:2201.02472 — disappearing-track search (recalled).
7. H. Fukuda, N. Nagata, H. Otono, S. Shirai, Phys. Lett. B 781, 306 (2018) — Higgsino disappearing tracks at HL-LHC (recalled).
8. R. Mahbubani, P. Schwaller, J. Zurita, JHEP 06 (2017) 119 (recalled).
9. H. Saito, R. Sawada, K. Terashi, S. Asai, Eur. Phys. J. C 79, 469 (2019) — 100 TeV disappearing tracks (recalled).
10. R. Capdevilla, F. Meloni, R. Simoniello, J. Zurita, JHEP 06 (2021) 133 — muon collider (recalled).
11. Corpus: P002 (kinematic floors), P007 (fit points, gaugino table, χ₂ lifetime), dossier 00; P048 (later, competing).

## 14. Tools and provenance (mirrors `output/provenance/P014.json`)

- Agent tools: Read (PAPER_GUIDE.md; 00_evidence_dossier.md; results_ledger.csv; papers/P007.md; papers/P002.md; work/P007/details.md;
  work/P007/gaugino_masses_for_splitting.csv; dataviz palette reference; the two output figures, several iterations), Bash (directory
  listings and ENVIRONMENT_versions.txt; grep of lzcommon.py; five script runs; run-log inspection; word count), Write (script,
  details.md, P014.json, P014.md), Edit (script fixes and figure-label iterations), Skill (dataviz).
- Software: python 3.12.13; numpy 2.5.3; scipy 1.18.1 (integrate.quad); pandas 3.0.5; matplotlib 3.11.2 (Agg); common/lzcommon.py
  (m_chi_min_gev, vmax_kms, v_earth_kms). No WimPyDD calls, no WimPyDD-generated files.
- Recalled knowledge: 17 items listed in `recalled_knowledge.json` with presumed sources and reliabilities (constants: certain;
  splitting/width formulae: likely, asymptotes and wino lifetime reproduced; LEP bound: certain; ATLAS/CMS disappearing-track and
  soft-lepton limits: likely/uncertain; cross-sections: likely, factor ~1.5–2; prospects: uncertain; thermal masses: certain/likely;
  dipole-decay estimate: uncertain).
- Datasets: none. Data requests: none.
