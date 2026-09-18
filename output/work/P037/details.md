# P037 · Which electroweak multiplets can be inelastic dark matter with a 100–400 keV neutral splitting? — research record

Simulated date 2026-09-09. Author profile: electroweak dark-matter model builders. Category MODEL, hep-ph.
Script: `output/code/P037_ew_multiplets.py` (run from the simulation root with `.venv/bin/python`; 6 s; log `run_log.txt`).
Every number below is printed in `run_log.txt` or stored in the CSV/JSON files listed in §10.

## 1. Motivation and framework

P007 showed that a pure Higgsino (SU(2)_L doublet, Y = 1/2) scatters in LZ only inelastically through the Z vector current,
with a gauge-fixed σ_n = 7.4×10⁻³⁹ cm², and therefore gives one LZ event only for δ ≈ 358–380 keV at 1 TeV. P014 noted that a
Y = 0 triplet's neutral state has no Z coupling at all. P021's shape-only likelihood prefers δ ≈ 380 keV (68 %: 360–385 keV).
The natural generalisation is the whole family of SU(2)_L × U(1)_Y fermion multiplets (n, Y) considered in "minimal dark matter"
(Cirelli, Fornengo, Strumia 2006; recalled: certain that the framework exists, the specific table entries are flagged below):
which of them can be the LZ scatterer, how is the Majorana splitting δ between the two neutral states generated, what scale Λ
gives δ = 100–400 keV, and how does the required δ move for the larger Z couplings of Y = 1, 3/2, 2?

Key structural facts used throughout (derived here; standard):
- A multiplet of dimension n = 2j+1 with hypercharge Y has components of charge Q = T₃ + Y, T₃ = −j…j. A neutral component exists
  iff Y ∈ {−j, …, j} with Y ≡ j (mod 1): (2, ½), (3, 0), (3, 1), (4, ½), (4, 3/2), (5, 0), (5, 1), (5, 2). Fermion multiplets with
  n ≥ 6 make g₂ non-perturbative below M_Pl (recalled: likely) and are not considered.
- The neutral Dirac component couples to the Z with g_Z = (g/c_W)(T₃ − Q s_W²) = −(g/c_W) Y. For Y = 0 it vanishes (P014).
  For Y ≠ 0 the current is pure vector (both chiralities carry the same T₃), so after the neutral Dirac state ψ is split into two
  Majorana states χ₁, χ₂ the Z current is purely off-diagonal, ψ̄γ^μψ = iχ̄₁γ^μχ₂ (P007 §2): the scattering is *only* inelastic.
- Because the whole Y dependence sits in one overall factor, c_p/c_n is Y-independent and the recoil-spectrum shape is identical to
  P007's Higgsino spectrum; only the normalisation scales as (2Y)².
- Electroweak gauge loops conserve the Dirac fermion number of ψ and cannot split χ₁ from χ₂ (P007 §7); the splitting needs an
  operator that violates the U(1) of ψ by two units, i.e. carries hypercharge 2Y, absorbed by 4Y Higgs fields (H^† has Y = −½):
  O = χχ (H^†)^{4Y}/Λ^{4Y−1}, dimension 3 + 4Y (5, 7, 9, 11 for Y = ½, 1, 3/2, 2), giving δ ≃ v^{4Y}/Λ^{4Y−1}.

## 2. Part A — Z couplings and cross-sections (`multiplets.csv`)

Inputs (recalled, certain): G_F = 1.166×10⁻⁵ GeV⁻², s_W² = 0.231, m_Z = 91.19 GeV, m_W = 80.37 GeV, α(m_Z) = 1/127.95,
v = 246.2 GeV (`lz.M_V_GEV`), ħ²c² = 0.3894 GeV² mb; Xe: Z = 54, ⟨A⟩ = 131.29 (`lz.A_XE_MEAN`). Nucleon vector charges
g_V^p = ¼(1 − 4s_W²), g_V^n = −¼ (P007).

σ_n(Y) = (2Y)² G_F² μ_n²/(2π); at 1 TeV:

| (n, Y) | charges | g_Z /(g/c_W) | (2Y)² | σ_n [cm²] | σ_SI,eq = σ_n/3.214 [cm²] | elastic excess vs LZ-2024 5×10⁻⁴⁷ (recalled, likely) | splitting op. dim |
|---|---|---|---|---|---|---|---|
| (2, ½) doublet | 0, +1 | −½ | 1 | 7.40×10⁻³⁹ | 2.30×10⁻³⁹ | 4.6×10⁷ | 5 |
| (3, 0) triplet (wino) | −1, 0, +1 | 0 | 0 | 0 | 0 | – | – |
| (3, 1) triplet | 0, +1, +2 | −1 | 4 | 2.96×10⁻³⁸ | 9.21×10⁻³⁹ | 1.8×10⁸ | 7 |
| (4, ½) quadruplet | −1, 0, +1, +2 | −½ | 1 | 7.40×10⁻³⁹ | 2.30×10⁻³⁹ | 4.6×10⁷ | 5 |
| (4, 3/2) quadruplet | 0, +1, +2, +3 | −3/2 | 9 | 6.66×10⁻³⁸ | 2.07×10⁻³⁸ | 4.1×10⁸ | 9 |
| (5, 0) quintuplet (MDM) | −2…+2 | 0 | 0 | 0 | 0 | – | – |
| (5, 1) quintuplet | −1, 0, +1, +2, +3 | −1 | 4 | 2.96×10⁻³⁸ | 9.21×10⁻³⁹ | 1.8×10⁸ | 7 |
| (5, 2) quintuplet | 0, +1, +2, +3, +4 | −2 | 16 | 1.18×10⁻³⁷ | 3.69×10⁻³⁸ | 7.4×10⁸ | 11 |

Checks: σ_n(Y = ½, 1 TeV) = 7.404×10⁻³⁹ cm² (P007 7.40); vector factor (A/((A−Z) − (1−4s_W²)Z))² = 3.214 (P007 3.214);
nuclear-level σ_A(¹³¹Xe, q→0) = 5.30×10⁻³¹ cm² (P007 5.30). "Elastic excess" is the excess an *unsplit* Dirac neutral state
would have over LZ's 2024 SI limit: every Y ≠ 0 multiplet is excluded by 8–9 orders of magnitude unless split (the classic
minimal-dark-matter statement, recalled: certain). Note the results organise by Y, not n: (2, ½) and (4, ½) are identical for LZ,
as are (3, 1) and (5, 1).

## 3. Part B — Splitting operators and the scale Λ (`lambda_table.csv`, `gaugino_lambda_crosscheck.csv`)

Λ(Y, δ) = (v^{4Y}/δ)^{1/(4Y−1)} with v = 246.2 GeV (nominal coefficient 1; if ⟨H⟩ = v/√2 is used the same δ needs
Λ × 2^{−2Y/(4Y−1)} = ×0.50, 0.63, 0.66, 0.67 for Y = ½, 1, 3/2, 2 — an O(1) convention/coefficient uncertainty we carry):

| Y (dim) | Λ(100 keV) | Λ(200) | Λ(300) | Λ(366) | Λ(380) | δ if Λ = M_Pl | δ if Λ = 10 TeV | δ if Λ = 1 TeV |
|---|---|---|---|---|---|---|---|---|
| ½ (5) | 6.06×10⁵ TeV | 3.03×10⁵ | 2.02×10⁵ | 1.66×10⁵ | 1.60×10⁵ | 5.0×10⁻⁹ keV | 6.1 GeV | 61 GeV |
| 1 (7) | 33.2 TeV | 26.3 | 23.1 | 21.6 | 21.3 | 2×10⁻⁴² keV | 3.7 MeV | 3.7 GeV |
| 3/2 (9) | 4.67 TeV | 4.07 | 3.75 | 3.60 | 3.58 | 8×10⁻⁷⁶ keV | 2.2 keV | 223 MeV |
| 2 (11) | 2.02 TeV | 1.82 | 1.72 | 1.67 | 1.67 | 3×10⁻¹⁰⁹ keV | 1.4 eV | 13.5 MeV |

Reading: (i) a Planck-suppressed operator gives a negligible δ for every Y — the required scale is a genuine intermediate scale.
(ii) For the doublet, Λ ≈ 1.6–2.0×10⁵ TeV. **Cross-check against P007's gaugino mixing**: δ = m_Z²(s_W²/M₁ + c_W²/M₂) =
g′²v²/(4M₁) + g²v²/(4M₂), so bino mixing is the dimension-5 operator with Λ_eff = 4M₁/g′² and wino mixing has Λ_eff = 4M₂/g².
With P007's numerically diagonalised masses (μ = 1 TeV, tan β = 10): bino M₁ = 6540/5589/5281/5072 TeV for δ = 300/350/370/385 keV
→ Λ_eff = 2.05/1.75/1.65/1.59×10⁵ TeV versus v²/δ = 2.02/1.73/1.64/1.57×10⁵ TeV (ratio 1.009–1.014); wino M₂ = 21453–16693 TeV →
ratio 0.998–0.999. (The 1 % offset is the O(ε²/μ) term in P007's series and m_Z from the input couplings, 91.53 vs 91.19 GeV.)
(iii) For Y = 1 the dimension-7 operator needs Λ ≈ 21–33 TeV: a heavy mediator well above the DM mass — a consistent EFT and a
concrete target scale. (iv) For Y = 3/2 the required Λ ≈ 3.6–4.7 TeV is above a 1 TeV DM mass but *below* the recalled thermal
mass of the (4, 3/2) multiplet (≈ 2.4 TeV, uncertain) only by ×1.5; for Y = 2 the required Λ ≈ 1.7–2.0 TeV is *below* the recalled
thermal mass (≈ 4.5 TeV, uncertain) and only marginally above 1 TeV: the "heavy mediator integrated out" picture fails, and the
splitting must come from a mediator lighter than the DM or from tree-level mixing with another multiplet. Conversely, with the
mediator at Λ ≥ 10 TeV the natural δ is 2.2 keV (Y = 3/2) or 1.4 eV (Y = 2) — effectively elastic and excluded (§2). This is the
quantitative content of the folk statement that higher-Y multiplets "cannot be split enough".

## 4. Part C — Expected LZ events and the required δ (`required_delta_by_Y.csv`, `exclusion_vs_LZ_edges_1TeV.csv`, `wimpydd_crosscheck.csv`)

N_Y(m, δ) = (2Y)² N_H(m, δ), with N_H from P007's WimPyDD grid `output/work/P007/N_events_grid.csv` (shell-model Xe responses,
Higgsino couplings c⁰_WD = −7.62×10⁻⁶, c¹_WD = +8.87×10⁻⁶ GeV⁻², Baxter-2021 SHM annual mean of 12 day-of-year halos on an explicit
0–830 km/s grid, LZ efficiency 0.96 with erf edges at 5.4/269.9 keV (σ_hi = 11.5 keV), 2.84 t·yr). The scaling is exact because
the spectral shape is Y-independent (§1). **Live WimPyDD cross-check** (Sun-frame halo, 1 TeV, same efficiency and 2 keV grid):
Y = ½: N(300 keV) = 727.1, N(370) = 0.2053; Y = 1 (couplings doubled): 2908 and 0.8212 — ratios to (2Y)² × P007 grid = 1.0000 at
all four points. δ(N) is obtained by log-linear interpolation on the falling branch; the 90 % one-event band is 0.105–3.65 signal
events (PLR, P007/P021); δ_max(248 keV) from `lz.delta_max_kev` with v_max on 16 June (810.5 km/s).

Annual halo, σ_hi = 11.5 keV:

| m | δ_max(248) | Y | δ(N=3.65) | **δ(N=1)** | δ(N=0.105) | window [δ(N=3.65), δ_max] | width | δ(N=1) Sun / June | v_esc 528 / 560 | N(δ=300) | N(350) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 300 | 315.8 | ½ | 306.8 | 310.2 | 311.0 | [306.8, 315.8] | 9.0 | 302.6 / 310.6 | 300.2 / 311.0 | 23.2 | 0 |
| 300 | | 1 | 310.2 | 310.6 | 311.4 | [310.2, 315.8] | 5.6 | 304.9 / 311.0 | 302.3 / 311.4 | 92.7 | 0 |
| 300 | | 3/2 | 310.4 | 310.9 | 311.6 | [310.4, 315.8] | 5.4 | 306.3 / 311.3 | 303.5 / 311.6 | 209 | 0 |
| 300 | | 2 | 310.6 | 311.0 | 311.8 | [310.6, 315.8] | 5.2 | 307.2 / 311.4 | 304.4 / 311.7 | 371 | 0 |
| 500 | 356.3 | ½ | 336.4 | 342.4 | 351.1 | [336.4, 356.3] | 19.8 | 335.8 / 347.4 | 331.4 / 350.9 | 439 | 0.14 |
| 500 | | 1 | 342.7 | 347.8 | 356.4 | [342.7, 356.3] | 13.5 | 340.9 / 352.8 | 336.3 / 356.7 | 1750 | 0.55 |
| 500 | | 3/2 | 345.9 | 350.8 | 357.5 | [345.9, 356.3] | 10.3 | 343.7 / 356.2 | 339.1 / 357.6 | 3950 | 1.25 |
| 500 | | 2 | 348.1 | 353.1 | 357.6 | [348.1, 356.3] | 8.1 | 345.6 / 357.5 | 341.1 / 357.6 | 7020 | 2.22 |
| 1000 | 386.6 | ½ | 357.8 | **365.8** | 379.5 | [357.8, 386.6] | 28.8 | 360.4 / 372.0 | 355.0 / 374.7 | 790 | 11.1 |
| 1000 | | 1 | 366.3 | **374.1** | 387.0 | [366.3, 386.6] | 20.3 | 368.8 / 381.3 | 362.7 / 384.1 | 3160 | 44.5 |
| 1000 | | 3/2 | 371.2 | **379.2** | 389.4 | [371.2, 386.6] | 15.5 | 373.4 / 386.2 | 367.4 / 388.8 | 7110 | 100 |
| 1000 | | 2 | 374.7 | **382.7** | 390.4 | [374.7, 386.6] | 12.0 | 376.2 / 388.4 | 370.5 / 390.3 | 12600 | 178 |
| 2000 | 401.8 | ½ | 365.4 | 374.9 | 391.2 | [365.4, 401.8] | 36.4 | 370.1 / 381.6 | 364.4 / 383.9 | 644 | 21.3 |
| 2000 | | 1 | 375.6 | 384.7 | – | [375.6, 401.8] | 26.2 | 380.0 / 392.3 | 373.7 / 394.3 | 2580 | 85 |
| 2000 | | 3/2 | 381.3 | 390.8 | – | [381.3, 401.8] | 20.5 | 385.9 / 398.7 | 379.3 / – | 5790 | 191 |
| 2000 | | 2 | 385.4 | 394.9 | – | [385.4, 401.8] | 16.4 | 389.5 / – | 383.1 / – | 10300 | 340 |
| 4000 | 409.4 | ½ | 365.8 | 376.4 | 393.9 | [365.8, 409.4] | 43.6 | 372.1 / 383.4 | 366.3 / 385.5 | 401 | 18.1 |
| 4000 | | 1 | 377.1 | 387.0 | – | [377.1, 409.4] | 32.2 | 382.4 / 394.3 | 376.3 / 396.1 | 1600 | 72.3 |
| 4000 | | 3/2 | 383.4 | 393.4 | – | [383.4, 409.4] | 26.0 | 389.0 / – | 382.3 / – | 3610 | 163 |
| 4000 | | 2 | 387.7 | 398.1 | – | [387.7, 409.4] | 21.6 | 393.4 / – | 386.6 / – | 6420 | 289 |

("–": N never falls to that value before the grid/kinematic end at 400 keV.) Elastic counts N(δ = 0) = 1.2×10⁸ (2Y)² at 1 TeV;
N(δ = 200 keV) = 6.4×10⁴ (2Y)².

Interpretation. (i) Because N(δ) falls by ×5 per 10 keV near the ceiling (P007), a factor (2Y)² = 4, 9, 16 in the rate moves the
required δ by only +8.3, +13.4, +16.9 keV at 1 TeV (365.8 → 374.1, 379.2, 382.7 keV). (ii) All four δ(N=1) values at 1 TeV lie
inside P021's 68 % shape-preferred band 360–385 keV; the larger multiplets sit closer to P021's maximum (380 keV) than the Higgsino
does. (iii) The "inelastic evasion window" [δ(N=3.65), δ_max(248 keV)] — the range where the multiplet is neither excluded at 90 %
by LZ's single event nor kinematically unable to produce it — narrows from 28.8 keV (Y = ½) to 12.0 keV (Y = 2) at 1 TeV, and from
9.0 to 5.2 keV at 300 GeV. Above 2 TeV δ(N=1) for Y ≥ 1 approaches or exceeds δ_max(248 keV) — the 248 keV recoil would then be at
the extreme low-energy end of the accessible spectrum — while δ(N=1) stays below δ_max(269.9 keV) (P007: 390.8 keV at 1 TeV,
407 keV at 2 TeV), so the event remains producible but atypical. (iv) Exclusion at LZ's grid points (1 TeV; digitised Fig. 6 upper
edges, P007 `lz_intervals_digitised.csv`): the Y-scaled coupling (2Y)² × 0.0777 exceeds the 90 % upper edge by ×1.0×10⁶ (2Y)² at
δ = 100 keV, ×1.27×10⁴ (2Y)² at 200 keV, ×2.0×10³ (2Y)² at 250, ×299 (2Y)² at 300 and ×3.46 (2Y)² at 350 keV; in events the excess
at 350 keV is ×1.91 (Y = ½) … ×30.6 (Y = 2). Every Y ≠ 0 multiplet is thus excluded at all of LZ's tabulated δ ≤ 350 keV at 1 TeV,
and the LZ event sits exactly at the edge of the excluded region: δ(N=3.65) is the 90 % exclusion boundary and δ(N=1) is 8 keV above it.

Halo systematics (1 TeV): v_esc 528/560 km/s shifts δ(N=1) by −2.8/+8.9 keV (Y = ½) and −3.6/+10.0 keV (Y = 1); the Sun-frame halo
+2.6…+4.7 keV; the June-only halo +5.7…+7.2 keV — the same pattern as P007/P018. The efficiency-edge width (8/15 keV) changes δ(N=1)
by < 0.5 keV (P007).

## 5. Part D — One-loop charged–neutral splittings (`radiative_splittings.csv`)

Cirelli–Fornengo–Strumia form (recalled: likely; P014 method, reproduced here):
Δm(Q) − Δm(Q′) = (α₂M/4π){(Q² − Q′²) s_W² f̃(m_Z/M) + (Q − Q′)(Q + Q′ − 2Y)[f̃(m_W/M) − f̃(m_Z/M)]},
f̃(r) = 2∫₀¹dx (1+x) ln[x² + (1−x)r²] + 5 (numerical `quad`; f̃/(2πr) = 0.9995, 0.9555 at r = 10⁻³, 0.1), α₂ = α(m_Z)/s_W² = 0.0338.
Heavy-mass limit: Δm → (α₂/2)[(Q² − Q′²) s_W² m_Z + (Q − Q′)(Q + Q′ − 2Y)(m_W − m_Z)].
Checks: doublet 1 TeV 341.8 MeV (P014 341.8), asymptote 356.4 (P014 356.3); wino 1 TeV 172.5 MeV (P014 MSbar-c_W 172.8; on-shell
160.7) — we keep s_W² = 0.231 throughout, so our Y = 0 values are at the upper end of P014's 7 % scheme spread. Quintuplet Y = 0:
172 (Q = ±1), 690 MeV (Q = ±2), the familiar 166·Q² MeV pattern (recalled: likely) within the scheme spread.

| (n, Y) | Q | Δm (1 / 2 / 4 TeV) [MeV] | asymptote | κ² = |⟨T₃±1|T^±|−Y⟩|² | cτ(Q→0) scaled [cm] |
|---|---|---|---|---|---|
| (2, ½) | +1 | 342 / 349 / 353 | 356 | 1 | 0.71 (P014 value) |
| (3, 0) | ±1 | 172 / 173 / 173 | 173 | 2 | 2.8 (P014 full calc. 5.8) |
| (3, 1) | +1 | 511 / 525 / 532 | 539 | 2 | 0.11 |
| (3, 1) | +2 | 1367 / 1395 / 1410 | 1425 | | |
| (4, ½) | −1 | +3 / −3 / −6 | −10 | 3 | – (lighter for M > 1.36 TeV) |
| (4, ½) | +1 | 342 / 349 / 353 | 356 | 4 | 0.18 |
| (4, ½) | +2 | 1029 / 1044 / 1051 | 1059 | | |
| (4, 3/2) | +1 / +2 / +3 | 680 / 1706 / 3076 (1 TeV) | 722 / 1791 / 3207 | 3 | 0.03 |
| (5, 0) | ±1 / ±2 | 172 / 690 | 173 / 693 | 6 | 0.9 |
| (5, 1) | −1 | −166 / −179 / −186 | −193 | 4 | – (lighter than neutral) |
| (5, 1) | +1 / +2 / +3 | 511 / 1367 / 2568 (1 TeV) | 539 / 1425 / 2658 | 6 | 0.035 |
| (5, 2) | +1 / +2 / +3 / +4 | 850 / 2045 / 3584 / 5469 (1 TeV) | 905 / 2158 / 3756 / 5702 | 4 | 0.012 |

cτ is a crude scaling estimate, cτ ≈ 0.712 cm × (342 MeV/Δm)³/κ² (pion-channel width ∝ κ² f_π² Δm³; recalled: likely), calibrated
on P014's Higgsino value; it reproduces P014's wino cτ only to ×0.5 (leptonic channels and phase space omitted), so treat it as an
order of magnitude. Conclusions: (a) for Y ≥ 1 the singly-charged partner is 0.5–0.9 GeV heavier and decays within ≲ 1 mm — even
further from disappearing-track sensitivity than the Higgsino (P014: needs r > 12 cm); LEP's chargino bound (~100 GeV, certain)
lies below the LZ kinematic floors (259 GeV at δ = 300 keV, P002/P014). Higher-charge states decay in cascades of soft pions.
(b) **Two multiplets have a charged state lighter than the neutral one at one loop**: the (4, ½) Q = −1 component (T₃ = −3/2) is
degenerate with the neutral one to within ±10 MeV and lighter for M > 1.36 TeV; the (5, 1) Q = −1 component (T₃ = −2) is lighter by
166–193 MeV at every mass. For these two the dark-matter candidate would be charged unless a tree-level T₃-dependent term — e.g.
(χ̄T^aχ)(H^†τ^aH)/Λ′, which needs Λ′ ≈ v²/(10 MeV) ≈ 6×10³ TeV for the quadruplet and ≈ 300 TeV for the quintuplet — lifts the charged
state; the dimension-7 Majorana operator of §3 (δ ~ 0.4 MeV) cannot do it. We flag this as a computed obstacle for (4, ½) and (5, 1)
consistent with remarks in the minimal-dark-matter literature (recalled: uncertain which multiplets were flagged there).

## 6. Summary table (`summary_table.csv`)

| multiplet | Z coupling / Higgsino | σ_n [cm²] | splitting op. | Λ(δ = 300 keV) | δ(N=1), 1 TeV | 90 % window, 1 TeV | thermal mass (recalled) | δ(N=1) at nearest grid mass | Δm(Q=1), 1 TeV | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| (2, ½) Higgsino-like | 1 | 7.4×10⁻³⁹ | dim-5, v²/Λ | 2.0×10⁵ TeV | 365.8 keV | [358, 387] | 1.1 TeV (certain) | 365.8 (1 TeV) | 342 MeV | viable; Λ = 4M₁/g′², PeV gauginos (P007) |
| (3, 0) wino | 0 | 0 | – | – | – | none | 2.7–3.0 TeV (likely) | – | 172 MeV | not an LZ scatterer without a new mediator (P014) |
| (3, 1) | 2 | 3.0×10⁻³⁸ | dim-7, v⁴/Λ³ | 23 TeV | 374.1 keV | [366, 387] | ~2 TeV (uncertain) | 384.7 (2 TeV) | 511 MeV | viable; needs a 20–30 TeV mediator |
| (4, ½) | 1 | 7.4×10⁻³⁹ | dim-5 | 2.0×10⁵ TeV | 365.8 keV | [358, 387] | ~2.4 TeV (uncertain) | 374.9 (2 TeV) | 342 MeV | LZ-identical to the Higgsino; Q = −1 state lighter for M > 1.36 TeV → needs a tree-level lift |
| (4, 3/2) | 3 | 6.7×10⁻³⁸ | dim-9, v⁶/Λ⁵ | 3.8 TeV | 379.2 keV | [371, 387] | ~2.4 TeV (uncertain) | 390.8 (2 TeV) | 680 MeV | viable only with a mediator at 3.6–4.7 TeV ≈ (1.5–2) M_DM: marginal EFT |
| (5, 0) MDM | 0 | 0 | – | – | – | none | 9.4 TeV (likely) | – | 172 MeV | not an LZ scatterer without a new mediator |
| (5, 1) | 2 | 3.0×10⁻³⁸ | dim-7 | 23 TeV | 374.1 keV | [366, 387] | ~4.5 TeV (uncertain) | 387.0 (4 TeV) | 511 MeV | LZ-identical to (3, 1); Q = −1 lighter by 170–190 MeV → needs a ≳ 200 MeV tree-level lift |
| (5, 2) | 4 | 1.2×10⁻³⁷ | dim-11, v⁸/Λ⁷ | 1.7 TeV | 382.7 keV | [375, 387] | ~4.5 TeV (uncertain) | 398.1 (4 TeV) | 850 MeV | Λ < M_DM: no heavy-mediator EFT; needs a light mediator or mixing |

## 7. Figures

- `figures/P037_fig1_delta_vs_Y.png` — δ(N = 1) (dots) and the 90 % one-event band δ(N = 3.65)…δ(N = 0.105) (bars) versus Y for
  500, 1000, 2000, 4000 GeV (annual halo); dashed lines δ_max(248 keV, 16 June) per mass; grey band P021's 68 % shape-preferred
  360–385 keV (1 TeV).
- `figures/P037_fig2_lambda_vs_delta.png` — Λ(δ) for the dimension-5/7/9/11 operators (log–log), the 1 TeV LZ window 358–387 keV,
  M_Planck and M_DM ≈ 1 TeV reference lines, and P007's bino-mixing point (M₁ = 5.3 PeV → Λ = 4M₁/g′² at δ = 370 keV) lying on the
  dimension-5 line.

## 8. Failed or abandoned approaches / caveats

- The first version of the chargino-lifetime scaling produced a negative cτ for the (5, 1) Q = −1 state because that state is
  lighter than the neutral one; the scaling is now suppressed when Δm < 150 MeV and the lighter-than-neutral cases are flagged.
- We did not recompute the full N(m, δ) grid with WimPyDD for each Y: the (2Y)² scaling is exact (identical c_p/c_n), verified live
  at four points to 10⁻⁴. A recomputation would add nothing but runtime.
- Thermal relic masses for multiplets other than the doublet are recalled from the minimal-dark-matter literature and flagged
  uncertain (±30 %); Sommerfeld enhancement raises the larger-multiplet values. We therefore tabulate δ(N=1) at 1, 2 and 4 TeV
  rather than at a single "thermal" mass, and treat the mass as a free parameter ≥ 259 GeV (P002 floor).
- The operator coefficient is taken as 1; with ⟨H⟩ = v/√2 or with O(1) Clebsch factors Λ moves by ×0.5–0.67 (tabulated).
- The one-loop splittings use s_W² = 0.231 in both terms (7 % scheme spread for Y = 0, P014); the tree-level O(δ) piece is ≤ 1 MeV
  (P014) and neglected.
- Not treated: loop-induced elastic scattering (Hill–Solon; 10⁻⁴⁹–10⁻⁴⁷ cm², recalled likely, irrelevant here), indirect
  detection, the χ₂ lifetime (P007/P011: 10⁵–10⁶ s for Z-mediated χ₂ → χ₁νν̄, scaling as (2Y)²), and scalar multiplets (their neutral
  component also has a Z vector coupling ∝ Y and the same analysis applies with a dimension-4 Majorana-like splitting term
  λ(H^†χ)² for Y = ½ — a different, generically larger δ; left for future work).

## 9. Discussion

The LZ event does not single out the Higgsino. Every Y ≠ 0 electroweak multiplet has the same Z-vector inelastic interaction up to
(2Y)², and because the LZ rate collapses ×5 per 10 keV near the kinematic ceiling, the factor 4–16 in rate is absorbed by a shift
of only 8–17 keV in δ: the required splittings, 366–383 keV at 1 TeV, all lie inside P021's shape-preferred band, and they
converge on δ_max at higher mass. What does distinguish the multiplets is the *origin* of δ. The doublet's dimension-5 operator
needs Λ ≈ 1.7×10⁵ TeV, exactly the 4M₁/g′² of P007's PeV bino — a scale with a known UV story (split SUSY). The Y = 1 triplet or
quintuplet needs a dimension-7 operator with Λ ≈ 22 TeV: a heavy mediator (e.g. a scalar with Y = 2 coupling to χχ and H⁴) that is
itself a target for future colliders. For Y = 3/2 the scale drops to ≈ 3.6 TeV, comparable to plausible DM masses, and for Y = 2 to
≈ 1.7 TeV, below them: the heavy-mediator EFT breaks down and the splitting must come from light mediators or tree-level mixing.
Independently, the (4, ½) and (5, 1) multiplets have a charged component that is radiatively degenerate with or lighter than the
neutral one and need an additional tree-level lift. Taken together: the Higgsino-like doublet and the Y = 1 triplet are the two
clean electroweak-multiplet readings of the event; both predict the same June-phased 4–8 events in LZ's next 1000 live days
(P007's forecast scales with the rate, hence identically once δ is fixed by N = 1), so LZ alone cannot separate them — the
difference lies in the mediator scale (PeV vs 20 TeV) and the chargino spectrum (342 vs 511 MeV, cτ 7 mm vs 1 mm).

## 10. References

1. LZ Collaboration, arXiv:2609.02823 (2026) — Theory paragraph ("O₁^s resembles a Higgsino model"), Fig. 6, Table S7.
2. M. Cirelli, N. Fornengo, A. Strumia, "Minimal dark matter", Nucl. Phys. B 753, 178 (2006) — multiplet classification, one-loop
   splittings, Y ≠ 0 exclusion by Z exchange (recalled).
3. M. Cirelli, A. Strumia, "Minimal Dark Matter: model and results", New J. Phys. 11, 105005 (2009) (recalled).
4. P. W. Graham, H. Ramani, S. S. Y. Wong, Phys. Rev. D 111, 055030 (2025) — Higgsino inelastic direct detection.
5. D. Tucker-Smith, N. Weiner, Phys. Rev. D 64, 043502 (2001) — inelastic dark matter.
6. N. Nagata, S. Shirai, JHEP 01 (2015) 029 — Higgsino splittings in high-scale SUSY (recalled).
7. R. J. Hill, M. P. Solon, Phys. Rev. Lett. 112, 211602 (2014) — loop-level electroweak-multiplet scattering (recalled).
8. Corpus: P002 (kinematics), P003 (WimPyDD convention), P007 (Higgsino grid, couplings, gaugino masses), P011 (dark-photon
   alternative), P014 (Y = 0 remark, one-loop method), P018 (halo tail), P021 (likelihood peak), P023 (composite alternative).

## 11. Tools and provenance (mirrors `output/provenance/P037.json`)

- Agent tools: Read (PAPER_GUIDE.md; 00_evidence_dossier.md; results_ledger.csv (2 pages); P007.md, P014.md, P021.md, P023.md,
  P011.md; work/P007/details.md; lzcommon.py lines 300–373; work/P014/details.md lines 25–65; P007_higgsino_inelastic.py lines
  150–210; the two P037 figures, twice each), Bash (directory listings of work/P007, work/P014, papers, data_requests; head/wc of
  N_events_grid.csv; cat of gaugino_masses_for_splitting.csv, higgsino_couplings.json, delta_for_N_events.csv,
  lz_intervals_digitised.csv, comparison_with_LZ_intervals.json; pandas print of the 1 TeV grid; grep of lzcommon.py, P014 details,
  radiative_splitting_vs_mass.csv, data_requests/index.csv, P007 script, fulltext.tex, dataviz palette; three script runs; log/CSV
  inspection), Write (script, details.md, P037.json, P037.md), Edit (7 script fixes), Skill (dataviz).
- Software: python 3.12.13; numpy 2.5.3; scipy 1.18.1 (integrate.quad, special.erf/erfc, optimize.brentq); pandas 3.0.5;
  matplotlib 3.11.2 (Agg); WimPyDD 2.0.4 (eft_hamiltonian, streamed_halo_function via lz.wd_halo with explicit v_min grid,
  diff_rate via lz.wd_rate with delta); common/lzcommon.py (LZ, A_XE_MEAN, AMU_GEV, M_NUCLEON_GEV, M_V_GEV, v_earth_kms, vmax_kms,
  delta_max_kev, E_R_range_keV, wd, wd_halo, wd_hamiltonian, wd_rate). Operator dimension counting, Λ(δ) inversion and the
  Λ_eff = 4M/g² identification: derived by hand.
- Recalled knowledge (14 items): G_F, s_W², m_Z, m_W, α(m_Z), v = 246 GeV, M_Pl (certain); Z coupling (g/c_W)(T₃ − Qs_W²) and
  nucleon vector charges (certain); SU(2) Clebsch–Gordan |⟨T₃±1|T^±|T₃⟩|² = (j∓T₃)(j±T₃+1) (certain); Cirelli–Fornengo–Strumia
  one-loop splitting formula (likely; reproduces P014 and the 166·Q² MeV pattern); perturbativity bound n ≤ 5 for fermion
  multiplets (likely); thermal masses: doublet 1.1 TeV (certain), wino 2.7–3.0 TeV (likely), (3,1) ≈ 2 TeV, (4,½) and (4,3/2) ≈ 2.4
  TeV, (5,1) and (5,2) ≈ 4.5 TeV (uncertain), (5,0) 9.4 TeV (likely); LZ 2024 SI limit ≈ 5×10⁻⁴⁷ cm² at 1 TeV (likely); chargino
  pion-channel width ∝ κ²f_π²Δm³ (likely); Y ≠ 0 multiplets excluded by Z-exchange elastic scattering unless split (certain);
  Hill–Solon loop-level σ_SI (likely, unused quantitatively).
- WimPyDD-generated files: none (diff_rate does not write response-function files; `wimp_dd_rate` not called).
- Datasets: none. Data requests: none.
