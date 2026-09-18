# P060 — Diffuse supernova, Galactic supernova, solar-flare and other astrophysical neutrinos at 248 keV: could a neutrino burst on 16 June 2023 explain the event?

Simulated date 2026-09-11 · astro-ph.HE (cross-list hep-ph) · neutrino astrophysicists · category EXO · competes with P019 (atmospheric flux); here: every *other* astrophysical neutrino flux and transient sources.

All numbers below are produced by `output/code/P060_astro_nu.py` (results in `output/work/P060/P060_results.json`, tables in `P060_*.csv`, log in `run_log.txt`) unless marked [recalled] or [paper]/[corpus].

## 1. Motivation and framework

LZ's background model contains two neutrino components (Table I [paper]): atmospheric-ν CEνNS, 0.11 ± 0.02 events in the ROI, and ⁸B+hep solar CEνNS, 0.057 ± 0.006. P019 showed that the atmospheric flux gives 1–3 × 10⁻⁵ coherent events in the 225–271 keV window, that a 248 keV recoil needs E_ν ≥ 123 MeV, and identified one incoherent path to a *lone* nuclear recoil: neutral-current quasi-elastic knock-out of a neutron, whose escape leaves the (A−1) daughter recoiling with the struck neutron's initial (hole) momentum, up to k_F²/2M = 279 keV; P019 estimated 1.7 × 10⁻⁴ (8 × 10⁻⁶–5 × 10⁻⁴) such events. LZ's own statement (Discussion, l.313–314 [paper]) is that "the reconstructed energy of the event is not consistent with scattering coherence".

This paper asks whether any *astrophysical* neutrino source other than the steady atmospheric flux could have produced the event: the diffuse supernova neutrino background (DSNB), a Galactic core-collapse supernova, the nearest 2023 extragalactic supernova (SN 2023ixf), solar-flare neutrinos, the IceCube diffuse astrophysical flux, reactor/geo/solar neutrinos, and a hypothetical unidentified burst coincident with 21:22:39 UTC on 16 June 2023. Two distinct channels are evaluated per source: coherent CEνNS into 200–270 keV and the incoherent Fermi-recoil channel into 225–271 keV. Finally the "burst" hypothesis is inverted: what fluence at Earth would give one window event, what does it cost energetically, and what would it have done in Super-Kamiokande and in LZ itself?

## 2. Inputs

| Input | Value | Source |
|---|---|---|
| Fiducial mass, live time, exposure | 4.71 t, 220 d, 2.84 t·yr | LZ paper (lzcommon `LZ`) |
| Atmospheric ν, ⁸B+hep in ROI | 0.11 ± 0.02, 0.057 ± 0.006 | LZ Table I |
| Efficiency | 0.955 plateau, 50 % at 5.4 and 269.9 keV (σ = 11.8 keV) | P009 fit, as used by P019 |
| Event | 248 ± 23 ± 23 keV, 16 June 2023 21:22:39 UTC | LZ paper |
| Xe isotopes, Helm F², m_N = A·u | lzcommon | corpus library |
| G_F = 1.1664 × 10⁻⁵ GeV⁻², sin²θ_W = 0.23867 | PDG [recalled, certain / likely] | same as P019 |
| Q_W = N − (1 − 4 sin²θ_W) Z; Q_W(¹³¹Xe) = 74.55 | Freedman 1974 [recalled, certain] | |
| Nuclei per tonne natural Xe: 4.584 × 10²⁷; in 4.71 t: 2.159 × 10²⁸; nucleons in FV: 2.836 × 10³⁰ | computed | |
| Keil–Raffelt spectrum f ∝ E^α e^{−(α+1)E/⟨E⟩}, α = 2.5 | Keil, Raffelt, Janka 2003 [recalled, likely; α = 2–3] | |
| DSNB all-flavour number flux 30 cm⁻² s⁻¹; ⟨E⟩ = 9/11/13 MeV (redshift-softened ν_e/ν̄_e/ν_x) | Beacom 2010 review [recalled, uncertain ×3] | |
| SK ν̄_e DSNB limit 2.7 cm⁻² s⁻¹ above 17.3 MeV | Bays et al. 2012 [recalled, likely] | used as a per-species saturating bracket |
| Galactic SN: 3 × 10⁵³ erg in ν, equipartition over 6 species, ⟨E⟩ = 12/15/18 MeV (hard: 15/18/25) | standard [recalled, certain to ×1.5 / likely] | |
| SN 2023ixf: Type II in M101, discovered 19 May 2023, d ≈ 6.4 Mpc | [recalled, likely] | |
| Betelgeuse distance ≈ 0.2 kpc | [recalled, likely] | |
| Llewellyn-Smith NC elastic, M_A = 1.03, M_V = 0.84 GeV, g_A = 1.267, μ_p, μ_n; Pauli factor 1.5x − 0.5x³, x = q/2k_F, k_F = 260 MeV/c | P019 (copied verbatim for consistency) [recalled, likely] | |
| Neutron separation energy of Xe isotopes S_n = 8.5 MeV (7.9–9.6 across isotopes) | [recalled, likely] | new here |
| P019 lone-NR factors P_esc = 0.3 (0.1–0.35), P_untagged = 0.2 (0.08–0.3), P_gs = 0.3 (0.1–0.5) | corpus P019 | |
| IceCube diffuse: E²Φ = 1.0 × 10⁻⁸ GeV cm⁻² s⁻¹ sr⁻¹ per flavour at 100 TeV, γ = 2.5 | IceCube 2015–2020 [recalled, likely ×2] | |
| ν–N cross-section: σ_CC = 0.68 × 10⁻³⁸ E[GeV] (1 + E/3.7 × 10⁴ GeV)^−0.637 cm², NC = 0.4 σ_CC | Gandhi et al. 1998 asymptotics 5.53 × 10⁻³⁶ E^0.363 (matched at 10 PeV) [recalled, likely ×2] | |
| Magnetar giant flare energy 2 × 10⁴⁶ erg at ~10 kpc (SGR 1806−20, 2004) | [recalled, likely] | |
| GRB neutrino fluence E²F ~ 10⁻³ GeV cm⁻² at 100 TeV per bright burst | IceCube GRB limits [recalled, uncertain] | |
| Solar-flare ν fluence: models ≲ 10²–10⁴ cm⁻²; SK per-flare limits ~10⁵–10⁷ cm⁻² | [recalled, uncertain] | |
| Super-K fiducial 22.5 kt; IBD σ ≈ 9.52 × 10⁻⁴⁴ E_e p_e cm² | Vogel–Beacom 1999 [recalled, certain in form; overestimates by ≲ 2 at 150–300 MeV] | |
| P019 numbers: C_norm, knock-outs 0.0412, lone-NR 1.73 × 10⁻⁴, coherent 1.66 × 10⁻⁵ | `output/work/P019/P019_results.json` | validation |

## 3. Kinematics (Section A of the script)

Two-body elastic scattering of a massless neutrino on a nucleus of mass M:

E_R,max = 2E_ν²/(M + 2E_ν),  E_ν,min(E_R) = ½[E_R + √(E_R² + 2 M E_R)].

| E_ν [MeV] | 1.8 | 10 | 15 | 18.8 (hep) | 20 | 30 | 50 | 60 | 80 | 100 | 123.1 | 150 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E_R,max [keV], natural Xe | 0.053 | 1.63 | 3.68 | 5.77 | 6.54 | 14.7 | 40.8 | **58.8** | 104 | 163 | 247 | 367 |

E_ν,min: 200 keV → 110.7 MeV, 225 → 117.5, 248 → **123.3**, 270 → 128.7, 271 → 128.9 MeV (natural Xe; ¹²⁴Xe 119.8, ¹³⁶Xe 125.5 MeV at 248 keV). These agree with P019 (123.1 MeV for A = 131) and P040 (p_min = 123 MeV/c). Hence DSNB, Galactic-SN, reactor, geo and solar neutrinos are kinematically blind to the window coherently: the hardest plausible SN tail (60–80 MeV) reaches 59–104 keV. Figure 1 (`figures/P060_ERmax_vs_Enu.png`) shows E_R,max(E_ν) with the source energy ranges.

## 4. Source spectra (Section B)

Keil–Raffelt pinched spectra, normalised numerically. For the DSNB, three species groups (ν_e, ν̄_e, 4 × ν_x) share the number flux 1/6 : 1/6 : 4/6; the "SK-saturating" variants set *each* species to the ν̄_e limit (2.7 cm⁻² s⁻¹ above 17.3 MeV) — a deliberately absurd upper bracket. For a supernova the fluence per species is E_tot/6/⟨E⟩/(4πd²).

| Source (spectrum) | total flux [cm⁻² s⁻¹] or fluence [cm⁻²] | above 17.3 MeV | above 30 MeV | above 110 MeV |
|---|---|---|---|---|
| DSNB central (9/11/13) | 30 | 5.6 | 0.52 | 4.3 × 10⁻⁹ |
| DSNB hard (12/15/18) | 30 | 11.7 | 2.6 | 7.4 × 10⁻⁶ |
| DSNB very hard (12/15/25) | 30 | 16.1 | 6.3 | 1.4 × 10⁻³ |
| DSNB SK-saturating (12/15/18) | 46.6 | 16.2 | 3.3 | 8.8 × 10⁻⁶ |
| DSNB SK-saturating very hard (12/15/25) | 38.9 | 16.2 | 5.4 | 1.1 × 10⁻³ |
| Galactic SN 10 kpc (12/15/18) | 9.71 × 10¹¹ | 3.6 × 10¹¹ | 7.7 × 10¹⁰ | 2.1 × 10⁵ |
| Galactic SN 10 kpc hard (15/18/25) | 7.36 × 10¹¹ | 4.1 × 10¹¹ | 1.5 × 10¹¹ | 2.8 × 10⁷ |
| Galactic SN 1 kpc | 9.71 × 10¹³ | 3.6 × 10¹³ | 7.7 × 10¹² | 2.1 × 10⁷ |
| Betelgeuse-like 0.2 kpc | 2.43 × 10¹⁵ | 9.0 × 10¹⁴ | 1.9 × 10¹⁴ | 5.4 × 10⁸ |
| SN 2023ixf 6.4 Mpc | 2.37 × 10⁶ | 8.8 × 10⁵ | 1.9 × 10⁵ | 0.52 |

The 10 kpc all-flavour fluence ≈ 10¹² cm⁻² (1.6 × 10¹¹ per species) is the familiar number (the assignment's "~10¹¹" is the per-species value).

## 5. Coherent CEνNS counts (Section C)

dσ/dE_R = (G_F² M/4π) Q_W² (1 − M E_R/2E_ν²) F_Helm²(E_R); dN/dE_R = Σ_A N_A ∫_{E_min} Φ_A(E) dσ/dE_R dE (× live time for steady sources). Counts are efficiency-weighted except "all E_R". Grid 0.25–330 keV.

| Source | all E_R | ROI 5.4–270 (eff.) | > 50 keV | > 100 keV | **200–270 keV** | 225–271 keV |
|---|---|---|---|---|---|---|
| DSNB central | 0.042 | 3.6 × 10⁻³ | 8.2 × 10⁻⁸ | 1.3 × 10⁻¹¹ | **4.2 × 10⁻¹⁵** | 2.7 × 10⁻¹⁶ |
| DSNB hard | 0.077 | 0.015 | 4.1 × 10⁻⁶ | 5.2 × 10⁻⁹ | 1.3 × 10⁻¹¹ | 1.3 × 10⁻¹² |
| DSNB very hard | 0.120 | 0.037 | 6.2 × 10⁻⁵ | 4.1 × 10⁻⁷ | 3.8 × 10⁻⁹ | 5.2 × 10⁻¹⁰ |
| DSNB SK-sat. | 0.109 | 0.019 | 4.9 × 10⁻⁶ | 6.2 × 10⁻⁹ | 1.5 × 10⁻¹¹ | 1.5 × 10⁻¹² |
| DSNB SK-sat. very hard | 0.119 | 0.032 | 4.9 × 10⁻⁵ | 3.3 × 10⁻⁷ | **3.0 × 10⁻⁹** | 4.1 × 10⁻¹⁰ |
| Galactic SN 10 kpc | 127 | 23.3 | 6.3 × 10⁻³ | 8.0 × 10⁻⁶ | **1.9 × 10⁻⁸** | 1.9 × 10⁻⁹ |
| Galactic SN 10 kpc hard | 155 | 46.1 | 0.069 | 4.5 × 10⁻⁴ | **4.2 × 10⁻⁶** | 5.7 × 10⁻⁷ |
| Galactic SN 1 kpc | 1.27 × 10⁴ | 2330 | 0.63 | 8.0 × 10⁻⁴ | 1.9 × 10⁻⁶ | 1.9 × 10⁻⁷ |
| Betelgeuse-like 0.2 kpc | 3.2 × 10⁵ | 5.8 × 10⁴ | 15.6 | 0.020 | 4.8 × 10⁻⁵ | 4.8 × 10⁻⁶ |
| SN 2023ixf | 3.1 × 10⁻⁴ | 5.7 × 10⁻⁵ | 1.5 × 10⁻⁸ | 1.9 × 10⁻¹¹ | 4.6 × 10⁻¹⁴ | 4.7 × 10⁻¹⁵ |

Sanity checks. (i) The DSNB in the ROI (3.6 × 10⁻³–0.037 events) is ≪ LZ's ⁸B+hep 0.057 and atmospheric 0.11; the DSNB is not in LZ's model and need not be. (ii) A 10 kpc SN gives 26.9 CEνNS recoils per tonne of xenon (all E_R), 23 in the ROI with efficiency — consistent with the recalled literature scale of tens of events per tonne for a 10 kpc burst [recalled, likely], so the normalisation machinery is right. (iii) The window counts are the tails of exponentially falling spectra above 110 MeV and are, as expected, ≥ 10⁻⁹ below one event even in the most contrived DSNB case; even a Betelgeuse-distance supernova gives 5 × 10⁻⁵ coherent window events while flooding LZ with 6 × 10⁴ ROI recoils in ten seconds.

## 6. Incoherent lone-NR channel with an explicit threshold (Section D)

P019's channel: ν + n(bound, momentum **p**) → ν′ + n(free), the daughter (A−1) nucleus recoiling with −**p**, E_res = p²/2M_{A−1}, up to k_F²/2M = 279 keV; the window 225–271 keV corresponds to p = 233–256 MeV/c, i.e. the outer 23 % of the Fermi sphere (f_win = (p_hi³ − p_lo³)/k_F³ = 0.233). P019 applied this to the atmospheric flux (mostly ≥ 100 MeV) without an energy gate. Here the sources are soft, so the gate matters.

**Threshold (derived here, impulse approximation).** Energy conservation E_ν = E_ν′ + T_f + S_n + E_res with the freed neutron's kinetic energy T_f = |**p** + **q**|²/2m_n and |**q**| ≤ E_ν + E_ν′. For a hole momentum p, the largest p reachable at given E_ν is obtained by minimising T_f (anti-parallel **q**) and maximising q, i.e. at E_ν′ → 0:

p_max(E_ν) = E_ν + √(2 m_n (E_ν − S_n − E_res)).

(The function E_ν′ + √(2m_n(E_ν − E_ν′ − S_n − E_res)) decreases with E_ν′ for T_f < m_n/2, so the maximum is at the boundary.) Solving p_max(E_thr) = p(E_res) with S_n = 8.5 MeV gives

| E_res [keV] | 100 | 150 | 200 | **225** | **248** | 271 | 279 |
|---|---|---|---|---|---|---|---|
| E_thr [MeV] | 18.6 | 23.5 | 28.3 | **30.6** | **32.7** | 34.8 | 35.6 |

versus 123.3 MeV for the coherent route. The gated window fraction f_win(E_ν) is 0 below 30.6 MeV, 0.074 at 32 MeV and the full 0.233 from 35 MeV up. Because the threshold configuration has vanishing phase space (E_ν′ → 0) and because the Pauli factor of P019 is a Fermi-gas average, the gated rate is an *upper* estimate near threshold.

Rate: N_lone = N_n · t · ∫ Φ(E) σ_n^{NCE,Pauli}(E) f_win(E) dE · P_esc P_untagged P_gs, with N_n = 1.66 × 10³⁰ neutrons in the FV, σ_n^{NCE,Pauli} = 7.1 × 10⁻⁴³, 2.3 × 10⁻⁴², 9.5 × 10⁻⁴², 6.0 × 10⁻⁴¹, 2.9 × 10⁻⁴⁰ cm² at 20/30/50/100/200 MeV (Llewellyn-Smith, copied from P019), and the P019 factor product 0.018 (range 8 × 10⁻⁴–0.0525).

**Validation.** Applying the gated formula to P019's central atmospheric flux (C_norm = 0.0534 cm⁻² s⁻¹ MeV⁻¹ at 100 MeV, shape E⁻¹ below 100 MeV, E⁻²·⁵ above, cut at 10 MeV) gives 0.0418 neutron knock-outs and 1.77 × 10⁻⁴ lone-NR events, versus P019's 0.0412 and 1.73 × 10⁻⁴: agreement to 1–2 % (the small difference is the gate and the σ interpolation grid).

| Source | n knock-outs (Pauli only) | lone-NR 225–271 keV, central | range (P019 factor extremes) |
|---|---|---|---|
| DSNB central | 2.9 × 10⁻⁴ | **1.8 × 10⁻⁷** | 7.9 × 10⁻⁹–5.2 × 10⁻⁷ |
| DSNB hard | 7.5 × 10⁻⁴ | 1.3 × 10⁻⁶ | 5.7 × 10⁻⁸–3.7 × 10⁻⁶ |
| DSNB very hard | 1.7 × 10⁻³ | 4.9 × 10⁻⁶ | 2.2 × 10⁻⁷–1.4 × 10⁻⁵ |
| DSNB SK-sat. | 1.0 × 10⁻³ | 1.6 × 10⁻⁶ | 7.1 × 10⁻⁸–4.7 × 10⁻⁶ |
| DSNB SK-sat. very hard | 1.5 × 10⁻³ | **4.1 × 10⁻⁶** | 1.8 × 10⁻⁷–1.2 × 10⁻⁵ |
| Galactic SN 10 kpc (per burst) | 1.21 | **2.0 × 10⁻³** | 8.9 × 10⁻⁵–5.8 × 10⁻³ |
| Galactic SN 10 kpc hard | 2.11 | **5.9 × 10⁻³** | 2.6 × 10⁻⁴–1.7 × 10⁻² |
| Galactic SN 1 kpc | 121 | 0.20 | 8.9 × 10⁻³–0.58 |
| Betelgeuse-like 0.2 kpc | 3.0 × 10³ | 5.0 | 0.22–15 |
| SN 2023ixf | 2.9 × 10⁻⁶ | 4.9 × 10⁻⁹ | 2.2 × 10⁻¹⁰–1.4 × 10⁻⁸ |
| atmospheric (P019 flux, gated) | 0.0418 | 1.77 × 10⁻⁴ | (P019: 1.73 × 10⁻⁴, 8 × 10⁻⁶–5 × 10⁻⁴) |

The DSNB's incoherent channel (≤ 4 × 10⁻⁶ even when saturating the Super-K limit with a 25 MeV ν_x spectrum) is 40–1000× below the atmospheric channel of P019, because only the ≳ 31 MeV tail contributes and σ_NCE ∝ E². A Galactic supernova would be a different matter — 2–6 × 10⁻³ lone high-energy recoils per burst, ≈ 5 for a Betelgeuse-distance event — but it would announce itself with 23–46 (10 kpc) to 6 × 10⁴ (0.2 kpc) ordinary CEνNS recoils in LZ within ~10 s and with 10³–10⁷ events in Super-K, and no Galactic supernova neutrino burst has been recorded since SN 1987A (SNEWS has never issued an alert) [recalled, certain to the best of our knowledge as of Sept 2026].

## 7. IceCube diffuse flux (Section E)

Per-flavour E²Φ = 10⁻⁸ (E/100 TeV)^−0.5 GeV cm⁻² s⁻¹ sr⁻¹, three flavours, 4π sr (Earth absorption at ≥ 100 TeV for through-centre paths would reduce this by ≲ 30 %; ignored). σ_tot(10 TeV / 100 TeV / 1 PeV) = 8.2 × 10⁻³⁵ / 4.1 × 10⁻³⁴ / 1.1 × 10⁻³³ cm² per nucleon. Interactions in 2.84 × 10³⁰ nucleons over 220 d, 1 TeV–10 PeV: **1.4 × 10⁻⁶** (8 × 10⁻⁸ from ≥ 100 TeV). Each deposits a GeV–PeV hadronic cascade, not 248 keV: the fraction of NC events with hadronic energy < 300 keV at 100 TeV is y < 3 × 10⁻¹² (dσ/dy roughly flat at small y), i.e. 10⁻¹⁸ events. The assignment's guessed σ ~ 10⁻³⁵ cm² is ~40× below the recalled value; the conclusion is the same either way.

## 8. The transient hypothesis (Section F)

For a monoenergetic fluence F of neutrinos at E_ν, the expected window count is N = N_T F σ_win(E_ν) with σ_win = Σ_A f_A ∫₂₀₀²⁷⁰ ε(E_R) dσ_A/dE_R dE_R (abundance-weighted, efficiency-weighted), N_T = 2.159 × 10²⁸ nuclei. Setting N = 1:

| E_ν [MeV] | σ_win [cm²] | σ_tot [cm²] | **F_req [cm⁻²]** | N_lo(5.4–200)/window event | N_lo(5.4–55) | E_iso at 10 kpc [erg] | at 1 kpc | at 100 pc |
|---|---|---|---|---|---|---|---|---|
| 125 | 1.44 × 10⁻⁴¹ | 5.0 × 10⁻³⁸ | 3.2 × 10¹² | 2367 | 2294 | 7.7 × 10⁵⁴ | 7.7 × 10⁵² | 7.7 × 10⁵⁰ |
| 130 | 2.05 × 10⁻⁴¹ | 5.0 × 10⁻³⁸ | 2.3 × 10¹² | 1671 | 1618 | 5.6 × 10⁵⁴ | 5.6 × 10⁵² | 5.6 × 10⁵⁰ |
| 150 | 3.93 × 10⁻⁴¹ | 5.1 × 10⁻³⁸ | **1.18 × 10¹²** | 893 | 861 | **3.4 × 10⁵⁴** | 3.4 × 10⁵² | 3.4 × 10⁵⁰ |
| 200 | 6.41 × 10⁻⁴¹ | 5.2 × 10⁻³⁸ | 7.2 × 10¹¹ | 564 | 541 | 2.8 × 10⁵⁴ | 2.8 × 10⁵² | 2.8 × 10⁵⁰ |
| 300 | 8.18 × 10⁻⁴¹ | 5.3 × 10⁻³⁸ | **5.7 × 10¹¹** | 451 | 431 | 3.3 × 10⁵⁴ | 3.3 × 10⁵² | 3.3 × 10⁵⁰ |
| 1000 | 9.47 × 10⁻⁴¹ | 5.4 × 10⁻³⁸ | 4.9 × 10¹¹ | 395 | 377 | 9.4 × 10⁵⁴ | 9.4 × 10⁵² | 9.4 × 10⁵⁰ |
| 3000 | 9.58 × 10⁻⁴¹ | 5.4 × 10⁻³⁸ | 4.8 × 10¹¹ | 391 | 373 | 2.8 × 10⁵⁵ | 2.8 × 10⁵³ | 2.8 × 10⁵¹ |

Spectrum-averaged: an atmospheric-like E⁻²·⁵ tail above 110 MeV needs 9.0 × 10¹¹ cm⁻² (N_lo = 696); a flat-in-log-E spectrum 130 MeV–1 GeV needs 6.0 × 10¹¹ (N_lo = 472). N_lo(5.4–55) → 373–391 at high E_ν reproduces P040's response-only floor (350 Helm) once the (1 − ME_R/2E²) factor becomes negligible, and P019's SM value 1631 sits between our 125 and 130 MeV rows, as it should for a flux dominated by the threshold region.

Comparison fluences and what they would give if placed *entirely* at 150 (300) MeV — a gross overestimate for every real source:

| Reference | fluence [cm⁻²] | events at 150 MeV | at 300 MeV |
|---|---|---|---|
| Galactic SN 10 kpc, all flavours (real energies ≈ 15 MeV) | 9.7 × 10¹¹ | 0.82 | 1.7 |
| Galactic SN 10 kpc hard, only the part above 110 MeV | 2.8 × 10⁷ | 2.4 × 10⁻⁵ | 5.0 × 10⁻⁵ |
| SN 2023ixf, all flavours | 2.4 × 10⁶ | 2.0 × 10⁻⁶ | 4.2 × 10⁻⁶ |
| magnetar giant flare, all 2 × 10⁴⁶ erg into 200 MeV ν at 10 kpc | 5.2 × 10³ | 4.4 × 10⁻⁹ | 9.2 × 10⁻⁹ |
| GRB neutrino number fluence (TeV–PeV) | 10⁻⁸ | 8 × 10⁻²¹ | 2 × 10⁻²⁰ |
| solar flare, model upper range | 10⁴ | 8.5 × 10⁻⁹ | 1.8 × 10⁻⁸ |
| solar flare, SK-limit-like | 10⁷ | 8.5 × 10⁻⁶ | 1.8 × 10⁻⁵ |

The first row is the instructive one: a Galactic supernova has *exactly* the right fluence but at ten times too low an energy; nature would have to release a supernova's worth of neutrinos at 150–300 MeV within 10 kpc — 3 × 10⁵⁴ erg isotropic-equivalent, ten times a core-collapse total — or 3 × 10⁵² erg at 1 kpc, 3 × 10⁵⁰ erg at 100 pc. Nothing known does this. Even a Sun-located source (1 AU) would need 8 × 10³⁵ erg in 150 MeV neutrinos, 10⁴ times a large flare's total bolometric energy (~10³² erg [recalled, likely]).

**What other detectors would have seen.** For F_req and a democratic flavour mix (ν̄_e = 1/6), Super-Kamiokande's 1.5 × 10³³ free protons give 6.2 × 10⁵ (150 MeV) to 1.2 × 10⁶ (300 MeV) inverse-β events within the burst duration, versus a background of order 10⁻² per second; charged-current reactions on ¹⁶O and the other flavours roughly double this. IceCube's supernova trigger, KamLAND, Borexino, SNO+ and LVD were all operating in June 2023 [recalled, likely], and SNEWS would have alerted. **What LZ itself would have seen.** The same burst deposits 450–2400 CEνNS recoils in 5.4–200 keV (Table above) within seconds of the event. LZ reports that the science data around the event were examined and "no anomalous populations were observed" (l.302 [paper]), and the event is one of 1710 science-sample events spread over 220 days. The burst hypothesis therefore fails three independent tests — energetics, external detectors, and LZ's own low-energy record — the last two of which do not rely on any recalled astrophysical number.

**On checking the catalogues.** Confirming that no GCN circular, SNEWS alert, GOES X-ray flare or IceCube/Super-K burst was logged at 21:22:39 UTC on 16 June 2023 is a trivial step for anyone with network access, but its outcome cannot change the conclusion: a positive coincidence would still be excluded by the fluence/energy budget and by the absence of low-energy companions in LZ; a null result merely confirms it. We therefore file no data request.

## 9. Summary table (`P060_summary_table.csv`)

| Source | E_ν [MeV] | kinematic reach | coherent 200–270 keV | incoherent lone-NR 225–271 | verdict |
|---|---|---|---|---|---|
| reactor / geo ν̄ | ≤ 10 | 1.6 keV | 0 | 0 (below 30.6 MeV) | excluded by kinematics |
| solar ⁸B / hep | ≤ 18.8 | 5.8 keV | 0 (LZ: 0.057 in ROI) | 0 | excluded (P019) |
| DSNB | 5–60, tail | 59 keV at 60 MeV | 4 × 10⁻¹⁵–3 × 10⁻⁹ | 2 × 10⁻⁷–4 × 10⁻⁶ | negligible |
| Galactic SN 10 kpc | 5–80 | 59 keV at 60 MeV | 2 × 10⁻⁸–4 × 10⁻⁶ per burst | 2–6 × 10⁻³ per burst | none occurred; 23–46 companions in 10 s |
| SN 2023ixf, 6.4 Mpc, 19 May 2023 | 5–60 | 59 keV | 5 × 10⁻¹⁴ | 5 × 10⁻⁹ | 28 d early, 10⁶ too faint |
| solar-flare ν | 10–300 | window only above 123 MeV | ≤ 8 × 10⁻⁶ (SK-limit fluence at 150 MeV); models ~10⁻⁸ | smaller | negligible |
| atmospheric (P019) | 10–10⁴ | unbounded | 6.5 × 10⁻⁵ (1.7 × 10⁻⁵ in 225–271) | 1.7 × 10⁻⁴ (8 × 10⁻⁶–5 × 10⁻⁴) | dominant but ≤ 5 × 10⁻⁴ |
| IceCube diffuse | 10⁶–10¹⁰ | unbounded | 1.4 × 10⁻⁶ interactions, GeV–PeV; 10⁻¹⁸ below 300 keV | – | negligible |
| hypothetical burst, E ≥ 130 MeV | ≥ 123 | ≥ 248 keV | needs 6 × 10¹¹–3 × 10¹² cm⁻² | – | 3 × 10⁵⁴ erg at 10 kpc; 10⁶ SK events; 450–2400 LZ companions |

## 10. Figures

- `figures/P060_ERmax_vs_Enu.png` — Fig. 1: maximum xenon recoil energy versus neutrino energy (coherent), with the LZ window, the incoherent lone-NR ceiling (279 keV) and its 31 MeV threshold, the 123 MeV coherent threshold, and the energy ranges of reactor/geo, solar, DSNB/SN, solar-flare, atmospheric and IceCube neutrinos.
- `figures/P060_expected_counts.png` — Fig. 2: expected window counts per source (coherent 200–270 keV, blue; incoherent 225–271 keV, orange), 4.71 t × 220 d (bursts per burst), against the one-event line.
- `figures/P060_required_fluence.png` — Fig. 3: fluence at Earth needed for one 200–270 keV CEνNS event versus neutrino energy, compared with a 10 kpc supernova (all flavours; and its > 110 MeV part), a Super-K-limit solar-flare fluence and a magnetar giant flare converted entirely to 200 MeV neutrinos.

## 11. Robustness and failed approaches

- DSNB normalisation ×3 and hardness (⟨E_x⟩ 13 → 25 MeV) move the coherent window count over six decades (4 × 10⁻¹⁵ → 3 × 10⁻⁹) and the incoherent count by ×25 (1.8 × 10⁻⁷ → 4.9 × 10⁻⁶); no choice approaches 10⁻⁵.
- S_n = 7.9–9.6 MeV shifts the lone-NR threshold by ∓0.6/+1.1 MeV around 30.6 MeV; f_win saturates by 35–36 MeV in all cases, so the DSNB incoherent counts change by < 10 %.
- Removing the gate entirely (P019's treatment) raises the DSNB central incoherent count from 1.8 × 10⁻⁷ to ~3 × 10⁻⁶ (dominated by unphysical 10–30 MeV contributions); the gate is therefore essential for soft sources and irrelevant for the atmospheric flux (2 % effect), as the validation shows.
- Using the WimPyDD shell-model weak form factor instead of Helm (P019: ×15 lower at 248 keV) would lower every coherent number and raise every F_req by ~×10; we quote Helm as the conservative (higher-count) choice.
- No approach was abandoned; an initial figure draft placed the IceCube band off-axis and had overlapping labels and wrong legend handles, fixed before the final run.

## 12. Discussion

The result is a clean negative that complements P019 rather than competing with it: the atmospheric flux remains the *only* astrophysical neutrino population with a non-negligible (yet ≤ 5 × 10⁻⁴) route to a lone 248 keV recoil, because it alone has substantial flux above ~30 MeV (incoherent) and above 123 MeV (coherent). The new ingredient is the explicit threshold for the Fermi-recoil channel, E_thr = 30.6 MeV for the window, which converts P019's channel into a usable tool for soft sources and shows that a future *Galactic* supernova would give a xenon detector 2–6 × 10⁻³ high-energy lone recoils per 10 kpc burst (0.2 at 1 kpc, ~5 at Betelgeuse distance) — an interesting, if academic, addition to supernova CEνNS phenomenology. The transient hypothesis is excluded by an energy budget (≥ 3 × 10⁵⁴ erg at 10 kpc in ≥ 130 MeV neutrinos), by Super-K (≥ 6 × 10⁵ events), and by LZ's own science sample (≥ 450 low-energy recoils in the same seconds), so the neutrino sector as a whole — steady or transient, standard or hard — is closed as an explanation of the event, and the dossier's hypothesis H (non-DM new physics via neutrinos) inherits P019's ≥ 790-companion argument unchanged.

## 13. References

LZ Collaboration, arXiv:2609.02823 (2026). J. F. Beacom, Annu. Rev. Nucl. Part. Sci. 60, 439 (2010). M. T. Keil, G. G. Raffelt, H.-T. Janka, ApJ 590, 971 (2003). K. Bays et al. (Super-Kamiokande), Phys. Rev. D 85, 052007 (2012). R. F. Lang, C. McCabe, S. Reichard, M. Selvi, I. Tamborra, Phys. Rev. D 94, 103009 (2016). R. Gandhi, C. Quigg, M. H. Reno, I. Sarcevic, Phys. Rev. D 58, 093009 (1998). P. Antonioli et al. (SNEWS), New J. Phys. 6, 114 (2004). P. Vogel, J. F. Beacom, Phys. Rev. D 60, 053003 (1999). Corpus: P003, P009, P013, P019, P040, P049.

## 14. Tools and provenance (mirrors `output/provenance/P060.json`)

- Agent tools: Read ×12 (PAPER_GUIDE.md; 00_evidence_dossier.md; results_ledger.csv rows 1–14; P019.md; P019_atm_nu.py; P013.md; P049.md; P040.md; fulltext.tex l.172–246 and 305–316; lzcommon.py l.1–62, 108–131, 218–235; three P060 figures ×2 rounds), Bash ×9 (ledger print + tex grep + ls + versions; lzcommon API grep + P019 JSON/provenance + data-request index; P019 details structure; palette grep; script run ×3 (initial, after figure fixes, final) with result print; wc -w budget checks ×3), Skill ×1 (dataviz), Write ×5 (script, details.md, P060.json, P060.md, ledger_row.json; P060.md rewritten once for the word budget), Edit ×12 (4 figure fixes in the script; 4 paper trims; 3 provenance count updates; this line).
- Software: python 3.12.13; numpy 2.5.3 (geomspace, trapezoid, interp); scipy 1.18.1 (integrate.quad, optimize.brentq, stats.norm); pandas 3.0.5 (CSV tables); matplotlib 3.11.2 (Agg); common/lzcommon.py (LZ constants, XE_ISOTOPES, m_nucleus_gev, helm_F2, GEV_TO_CM2, A_XE_MEAN). No WimPyDD or nestpy calls.
- Recalled knowledge: 21 items (see §2 and JSON). Datasets: none. Data requests: none (deliberately; §8). WimPyDD-generated files: none.
