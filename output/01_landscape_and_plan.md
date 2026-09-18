# 01 · Research landscape, forecast and the 100-paper plan

*Frozen on 3 September 2026 (simulated). Not revised afterwards.*

## 1. Taxonomy of research directions (two-week horizon)

| Code | Category | What it covers | Why it matters here |
|---|---|---|---|
| **STAT** | Statistical reinterpretation | Single-event inference, Bayes factors, look-elsewhere effect, coverage of two-sided intervals, joint fits with the 2024 low-energy search, posterior predictive checks, base rates of anomalies | The paper's 3.4σ→2.6σ is the only quantitative evidence; its meaning is contested |
| **BKG** | Background explanations | MSSI (wall/RFR), ER leakage (EC double-vacancy, recombination tails, activation lines), neutrons ((α,n), fission, muon-induced), accidentals, radon in mixed flow, veto information, event-topology odds | The event sits in the NR–MSSI overlap; the mundane hypotheses have the largest prior |
| **RESP** | Detector response and calibration | NEST retuning, energy reconstruction (248 vs 265 keV), band width at 250 keV, g₁/g₂ propagation, S1 pulse-shape, resolution, charge loss, artefacts | Every interpretation depends on the energy and on band positions |
| **IDM** | Inelastic dark matter phenomenology | Kinematic (m, δ) maps, endothermic/exothermic scattering, light vs heavy mediators, isospin structure, modulation, global fits, endpoint physics | Inelastic O₁/O₄ at δ = 300–350 keV give the maximal 3.4σ and explain the absent low-energy population |
| **EFT** | Elastic NREFT/relativistic operators | Which of L₁–L₂₀ produce a lone high-energy recoil, magnetic/anapole/electric dipoles, matching with `directdm`, interference and degeneracies, light-mediator variants | L₁₀ and L₁₆ share the 3.4σ maximum |
| **MODEL** | UV model building and relic abundance | Higgsino/electroweak multiplets, dark-photon pseudo-Dirac fermions, Z′ models, composite DM, magnetic inelastic DM, multi-component DM, thermal history | Gives the required couplings physical meaning and cross-constraints |
| **NUC** | Nuclear-physics inputs | Xe response functions at q ≈ 250 MeV, isotope contributions, spin-dependent structure, nuclear excitation channels, Migdal/bremsstrahlung, quenching | Form-factor zeros and isotope mixtures shape the high-energy spectrum |
| **HALO** | Astrophysical and halo uncertainties | v_esc, v₀, ρ₀, high-velocity tails, streams, dark disk, gravitational focusing, the 16 June date | δ_max and the inelastic rate are exponentially sensitive to the tail |
| **XEXP** | Consistency with other experiments and targets | XENONnT, PandaX-4T, LUX/XENON1T/PandaX-II archives, PICO-60, CRESST, DEAP-3600, DAMA/LIBRA (iodine), combined xenon-world counts | Any DM interpretation must survive comparable exposures |
| **COMP** | Complementary probes | LHC/LEP, disappearing tracks, indirect detection (γ, p̄, lines), CMB/BBN, neutron stars, solar capture (IceCube), excited-state decays | Model-dependent but potentially decisive |
| **EXO** | Non-DM exotic explanations | Atmospheric/solar/DSNB neutrino processes, boosted DM, millicharged/strongly interacting heavy particles | Small prior, but the community always checks |
| **PROJ** | Prospects and decisive tests | LZ post-2024 data, modulation tests, ROI/FV optimisation, XLZD, target complementarity, timelines, falsifiable forecasts | What settles the question and when |

## 2. Forecast of the real-world response (3–16 September 2026)

**Volume.** I expect **≈ 28 arXiv papers** (plausible range 18–45) citing arXiv:2609.02823 by 16 September 2026, plus conference slides and blog commentary that do not count. Two weeks is short: hep-ph groups with ready-made inelastic-DM or NREFT codes can post within 3–7 days; experimental collaborations will not respond formally within two weeks; independent statisticians/former experimentalists produce a handful of critical notes.

**Distribution across categories** (share of the ~28): IDM 25%, MODEL 12%, EFT 12%, COMP 12%, XEXP recasts 10%, STAT 8%, HALO 6%, BKG+RESP 6% (mostly by ex-collaboration members writing carefully), EXO 5%, PROJ 4%.

**Order of appearance.** Days 1–4: kinematic (m, δ) maps, Higgsino/pseudo-Dirac fits, EFT operator recasts, "would XENONnT/PandaX have seen it" notes, a quick Bayesian reassessment. Days 5–9: relic-density and collider consistency papers, halo-tail and modulation papers, first detector-critical notes (MSSI, ER tails). Days 10–14: global fits, indirect-detection constraints, projections for LZ's next release, and a first review-style summary.

**Most-cited ideas.** (1) Inelastic DM with δ ≈ 200–390 keV at TeV mass, especially Higgsino-like electroweak doublets and dark-photon-mediated pseudo-Dirac fermions; (2) magnetic-dipole (L₁₀) and anapole (L₁₆) DM at m ≳ 400 GeV; (3) the statistical critique of a non-blind single event at 2.6σ global.

**Consensus after two weeks.** "Intriguing but not evidence": the community will regard the event as most likely a background or artefact (with wall MSSI and an unmodelled ER tail as the leading candidates), will note that inelastic DM at δ ≈ 250–390 keV is the only DM scenario that fits *naturally*, and will converge on three demands: LZ should analyse its post-April-2024 data (already ≳ 2× the exposure), XENONnT and PandaX-4T should extend their ROIs to ~300 keV, and the LZ Data Release should allow independent likelihood fits. No one will claim discovery; a few will claim exclusion of the DM interpretation by other experiments (prematurely, since no comparable high-energy xenon analysis exists).

**Deliberate re-weighting in this corpus.** Relative to the forecast, I allocate *more* papers to BKG/RESP/STAT (≈ 33 instead of ≈ 14%), because those are where scientific correctness is decided, and slightly fewer to pure model-building. Competing/overlapping papers on the most popular ideas are marked ★.

## 3. The 100 papers (chronological)

Categories as in §1. Dates are simulated arXiv posting dates. "★" marks deliberately competing/overlapping papers on the same popular idea.

| ID | Date | Cat | Working title | One-line question |
|---|---|---|---|---|
| P001 | 09-03 | STAT | How much evidence is one event? A Bayesian and frequentist reassessment | What Bayes factor and posterior DM probability does one event in a 2×10⁻⁴-background region support, and how does it depend on priors? |
| P002 | 09-03 | IDM | Kinematic map of inelastic dark matter consistent with a 248 keV xenon recoil | For which (m_χ, δ) is a 248 ± 32 keV recoil possible on 16 June 2023, and where does the rate peak? |
| P003 | 09-03 | EFT | Which NREFT operators can produce a lone high-energy recoil? Spectral shape ratios in xenon | For O₁–O₁₅ (and L₁–L₂₀), what is the ratio of rate above 200 keV to rate below 55 keV, and which survive the 2024 low-energy null? Reproduces Fig. 1 with WimPyDD |
| P004 | 09-03 | BKG | Wall MSSI as the origin: required mismodelling factor and what the sidebands allow | How much larger than modelled must the wall-MSSI rate be, and is that excluded by the 0–1-event sidebands? |
| P005 | 09-03 | XEXP | Would XENONnT and PandaX-4T have seen it? Expected counts in their standard ROIs | Under LZ's best-fit models, how many events do the 3.1 and 1.54 t·yr exposures expect within their published ROIs? |
| P006 | 09-03 | HALO | Sixteenth of June: the event date under inelastic dark matter and under background | What is the likelihood ratio of the event date for modulating inelastic signals versus flat backgrounds? |
| P007 | 09-04 | IDM | Higgsino-like inelastic dark matter confronting the LZ event ★ | What Higgsino mass, splitting and Z-mediated inelastic cross-section fit one event, and is it consistent with the elastic (loop-level) limits? |
| P008 | 09-04 | STAT | An independent look-elsewhere estimate for 616 models | What trials factor do the 293 distinct spectra imply, and does 3.4σ→2.6σ follow? |
| P009 | 09-04 | RESP | 248 or 265 keV? Reconstructing the recoil energy with the tuned NEST model | What energy and uncertainty follow from S1c, S2c and the tuned NEST yields, and how close is the event to the efficiency edge? |
| P010 | 09-04 | BKG | ER leakage: the recombination fraction required and the ¹²⁴Xe/¹²⁵I double-vacancy hypothesis | What recombination fluctuation would move a 64–71 keV ER to the event, and how many such decays occurred? |
| P011 | 09-04 | IDM/MODEL | Dark-photon-mediated pseudo-Dirac dark matter: fitting the event ★ | What (m_χ, δ, m_A′, ε α_D) reproduce one event, and is it compatible with dark-photon searches? |
| P012 | 09-04 | EFT | Magnetic-dipole dark matter at 1 TeV: the coupling from the LZ fit and the low-energy shoulder ★ | What dipole moment gives 1 event, and does the predicted low-energy shoulder survive LZ 2024? |
| P013 | 09-05 | BKG | Neutron origins of a 248 keV single scatter: spectra and multiplicity | What (α,n), fission and muon-induced neutron spectra give 248 keV, and how many lower-energy companions are expected? |
| P014 | 09-05 | COMP | Collider constraints on electroweak-multiplet inelastic dark matter ★ | Do LEP/LHC bounds allow the Higgsino/wino masses that fit? |
| P015 | 09-05 | XEXP | Fluorine, argon and tungsten: PICO-60, DEAP-3600 and CRESST sensitivity to the inelastic best fit | Which targets are kinematically blind to δ ≈ 300 keV and which could confirm? |
| P016 | 09-05 | STAT | Joint constraint from the 2024 low-energy search and the extended window | How does adding the 3–80 phd null (no excess) restrict the spectral shape of any signal explaining the event? |
| P017 | 09-05 | NUC | Xenon nuclear response at q ≈ 250 MeV: form-factor uncertainties for O₁, O₄ and dipole operators | How uncertain are the high-q response functions and the inferred couplings? |
| P018 | 09-05 | HALO | The high-velocity tail: escape velocity and non-Maxwellian tails versus δ_max | How do v_esc = 500–600 km/s and tail shapes change δ_max and the inelastic rate? |
| P019 | 09-05 | EXO | Atmospheric-neutrino processes at 248 keV: CEνNS tail and beyond | What atmospheric-ν rate is expected at 248 keV and could a non-coherent process fake an NR? |
| P020 | 09-05 | PROJ | What LZ's data since April 2024 should show | Under the best-fit models, what event counts and dates are predicted for the untouched exposure? |
| P021 | 09-08 | IDM | A full likelihood fit of the single event in (m_χ, δ, σ) ★ | With the tuned response and efficiency, where do the likelihood and 90% regions lie, including δ > 350 keV? |
| P022 | 09-08 | BKG | Accidental coincidences at high S1: isolated 540 phd S1s and 270-electron S2s | What is the rate of accidental pairs at the event's coordinates given the paper's isolated-pulse model? |
| P023 | 09-08 | MODEL | Composite dark-baryon inelastic dark matter with 300 keV splittings | Can a dark-QCD hyperfine splitting naturally give δ ≈ 300 keV at TeV mass? |
| P024 | 09-08 | RESP | The NR charge-yield break and the NR-band width at 250 keV | How sensitive is the "1.5σ below the NR median" to the new (a, b, E₀) parameters and to fluctuation models? |
| P025 | 09-08 | COMP | Indirect detection of TeV inelastic dark matter: γ-ray and antiproton constraints | Are the annihilation cross-sections implied by thermal production excluded by Fermi/H.E.S.S./AMS? |
| P026 | 09-08 | COMP | Cosmological constraints on the excited state: BBN, CMB and the χ₂ → χ₁ transition | For δ ≈ 300 keV, what lifetimes and abundances of χ₂ are allowed by BBN and CMB? |
| P027 | 09-08 | STAT | Bayesian model comparison across 293 spectra given one event | Which model classes are preferred a posteriori, and how flat is the posterior? |
| P028 | 09-08 | XEXP | Iodine as a second target: DAMA/LIBRA, COSINE and the inelastic interpretation | Does the iodine kinematic similarity to xenon predict a visible high-energy signal in NaI experiments? |
| P029 | 09-08 | BKG | Activation products eight days after AmBe: activities and MSSI-capable branches | What ¹²⁵Xe, ¹²⁷Xe, ¹²⁹ᵐXe, ¹³¹ᵐXe, ¹³³Xe activities were present on 16 June 2023 and which decays can produce MSSI topologies? |
| P030 | 09-08 | HALO | Streams and substructure: boosting the 248 keV rate without low-energy events ★ | Can a fast stream reconcile a TeV-scale rate with the absence of a low-energy population? |
| P031 | 09-09 | EFT | Relativistic operator matching for the preferred Lagrangians with `directdm` | Which UV operators map onto L₂, L₄, L₆ᵛ, L₉–L₁₂, L₁₆, and what are their Wilson coefficients at the fit? |
| P032 | 09-09 | IDM | Isovector inelastic couplings and the non-zero O₁ᵛ significance at δ = 0 | Why does O₁ᵛ (δ = 0) show 1.3σ while O₁ˢ shows 0, and what does isospin violation buy? |
| P033 | 09-09 | BKG | Geometric probability of a wall-MSSI topology with a 12 keV Compton scatter 27 cm inside ★ | What fraction of wall γ-ray MSSIs can have their ionising scatter at the event position? |
| P034 | 09-09 | PROJ | Annual modulation as the decisive test for inelastic dark matter ★ | What exposure detects the predicted June–December asymmetry at 3σ? |
| P035 | 09-09 | XEXP | Recasting XENONnT 2025 and PandaX-4T 2025 to inelastic dark matter ★ | What approximate (δ, σ) exclusion would those exposures give if extended to 300 keV? |
| P036 | 09-09 | NUC | Nuclear excitation channels: could the recoil be an NR plus a de-excitation γ? | Where would DM-induced ¹²⁹Xe (39.6 keV) or ¹³¹Xe (80 keV) excitations land in S1c–S2c? |
| P037 | 09-09 | MODEL | Electroweak multiplets: radiative splittings versus the required δ | Do wino-like triplets and larger multiplets produce neutral splittings of 100–400 keV? |
| P038 | 09-09 | STAT | The efficiency edge: how the S1c < 600 phd cut shapes inference at 250–270 keV | How do limits and significance change if the true energy is 265 keV and the ROI boundary is varied? |
| P039 | 09-09 | COMP | Neutron-star heating by TeV inelastic dark matter | Do NS kinetic-heating bounds constrain δ ≈ 300 keV, σ ≈ 10⁻⁴³ cm² models? |
| P040 | 09-09 | EXO | Cosmic-ray-boosted light dark matter and the extended window | Can boosted sub-GeV DM give a 248 keV recoil without a steeply rising low-energy population? |
| P041 | 09-10 | BKG | Detector artefacts: partial charge loss and the 71 keV ER hypothesis | How much charge loss is needed, and can electron-lifetime or field effects at 870 μs drift supply it? |
| P042 | 09-10 | IDM | The de-excitation photon: χ₂ → χ₁ γ inside LZ | For a magnetic transition with δ = 300 keV, what is the decay length and would the photon be seen? |
| P043 | 09-10 | RESP | Propagating g₁ and g₂ uncertainties into the recoil energy and δ_max | How do ±2% (g₁) and ±3% (g₂) shift the energy and the inelastic bounds? |
| P044 | 09-10 | EFT | Anapole and electric-dipole dark matter: fits and low-energy constraints ★ | Do L₁₆-type anapole couplings fit the event and evade the 2024 low-energy search? |
| P045 | 09-10 | STAT | What a lower limit that lifts off zero means for one event | What is the coverage of the two-sided 90% interval, and does the lower limit imply evidence? |
| P046 | 09-10 | XEXP | Target complementarity for δ ≈ 300 keV: which nucleus confirms? ★ | For Xe, I, W, Ge, Ar, F, what are the rate ratios at fixed coupling? |
| P047 | 09-10 | NUC | Isotope-by-isotope contributions: ¹²⁹Xe/¹³¹Xe spin-dependent O₄ versus O₁ at 248 keV | Which isotopes carry the rate for O₄ and how does that alter the spectrum? |
| P048 | 09-10 | COMP | Chargino lifetimes and disappearing-track searches for Higgsino-like fits ★ | What charged–neutral splitting accompanies δ ≈ 300 keV and is it probed by LHC disappearing tracks? |
| P049 | 09-10 | BKG | Muon-induced high-energy neutrons at 4850 ft reaching the fiducial volume untagged ★ | What is the rate of ≥ 8 MeV neutrons scattering once in the FV with no veto? |
| P050 | 09-10 | PROJ | Next-generation (XLZD-scale) sensitivity to the inelastic and dipole best fits | How quickly would a 60–80 t detector confirm or exclude? |
| P051 | 09-11 | IDM | Inelastic dark matter with a light mediator: momentum dependence at 250 keV | How does a mediator lighter than the momentum transfer reshape the spectrum and the fit? |
| P052 | 09-11 | STAT | Reproducing the 3.4σ local significance from the published background numbers | Does a toy likelihood with the paper's PDF summaries reproduce the quoted local significance? |
| P053 | 09-11 | BKG | ²¹⁴Pb in the mixed-flow state and untagged MSSI | What fraction of the exposure lacked the radon tag and how much does that change the MSSI expectation? |
| P054 | 09-11 | MODEL | Pseudo-Dirac fermion dark matter with a Z′: coupling relations and relic density ★ | What Z′ mass and couplings give both the relic density and the fitted cross-section? |
| P055 | 09-11 | HALO | Known stream candidates (Sagittarius, S1, Gaia-Enceladus debris) and the June event ★ | What modulation phase and rate boost would specific streams give? |
| P056 | 09-11 | RESP | ER/NR discrimination at 250 keV from the double-skew-Gaussian recombination model | What leakage fraction to −6.7σ does the tuned fluctuation model predict? |
| P057 | 09-11 | EFT | Operator interference and the degeneracy among L₂, L₄, L₉–L₁₂: what a second event would distinguish | How different are the spectra, and how many events separate them? |
| P058 | 09-11 | COMP | Survival of the excited state and exothermic down-scattering signals ★ | What fraction of χ₂ survives to today, and would exothermic scattering appear in low-energy data? |
| P059 | 09-11 | XEXP | Archival xenon exposures (LUX, XENON1T, PandaX-II) reinterpreted for the extended window | What do the older high-energy analyses already imply for the fitted couplings? |
| P060 | 09-11 | EXO | Diffuse supernova and other astrophysical neutrinos at 248 keV | Can any astrophysical neutrino flux produce a 248 keV NR at an interesting rate? |
| P061 | 09-12 | STAT | Prior sensitivity of the dark-matter posterior | How does the posterior probability of DM vary over reasonable priors on model space and background mismodelling? |
| P062 | 09-12 | IDM | Mass–splitting degeneracy and the spectral endpoint | Why can a single recoil not fix m_χ above 400 GeV, and what does the endpoint do? |
| P063 | 09-12 | BKG | Spontaneous fission of ²³⁸U in detector components: multiplicity and veto probability ★ | What is the probability that a fission neutron gives a lone 248 keV NR with no other deposit? |
| P064 | 09-12 | NUC | Nuclear-recoil quenching at 250 keV: Lindhard versus the tuned NEST yields | Do the tuned yields agree with Lindhard-based expectations and what energy-scale systematic follows? |
| P065 | 09-12 | MODEL | Magnetic inelastic dark matter (MiDM) at LZ | Can a dipole transition between χ₁ and χ₂ fit the event, and what does the ER-like de-excitation predict? |
| P066 | 09-12 | COMP | Gamma-ray lines from long-lived excited states in the halo | For χ₂ lifetimes exceeding the age of the Universe, what 300 keV line flux is expected and is it excluded? |
| P067 | 09-12 | PROJ | Directional and multi-target strategies to confirm a high-energy recoil population | What would a directional or multi-target programme add for a δ ≈ 300 keV signal? |
| P068 | 09-12 | HALO | Joint astrophysical uncertainty band on the inferred cross-section | How do ρ₀, v₀, v_esc uncertainties propagate to σ and δ_max? |
| P069 | 09-12 | XEXP | A xenon-world exposure model: expected high-energy events across LZ, XENONnT and PandaX-4T | Under the best fit, how many events should the combined ~8 t·yr contain? |
| P070 | 09-12 | BKG | Position-dependent MSSI probability: a geometric PDF in (r, z) ★ | How does the event's position compare with the spatial distribution expected for wall and RFR MSSI? |
| P071 | 09-15 | STAT | Alternative look-elsewhere treatments: Bonferroni and Gross–Vitells ★ | How robust is the 2.6σ global value to the LEE method? |
| P072 | 09-15 | IDM | Exothermic scattering of a metastable excited state ★ | Could down-scattering with δ < 0 produce the event, and at what abundance? |
| P073 | 09-15 | BKG | Veto information and the MSSI tagging efficiency | How does the 94 ± 2% prompt-veto efficiency for MSSI propagate to the untagged science-sample rate? |
| P074 | 09-15 | EFT | Long-range operators (q⁻², q⁻⁴): excluded by the spectral shape? | Do light-mediator elastic spectra allow a lone high-energy event? |
| P075 | 09-15 | MODEL | Thermal history of 1 TeV pseudo-Dirac dark matter with the fitted coupling | Is the fitted coupling consistent with thermal freeze-out, coannihilation or non-thermal production? |
| P076 | 09-15 | COMP | Solar capture of inelastic dark matter and neutrino-telescope constraints | Can the Sun capture δ ≈ 300 keV DM, and what does IceCube/Super-K imply? |
| P077 | 09-15 | RESP | S1 pulse-shape discrimination at 550 phd: expected separation | What NR/ER separation do singlet/triplet ratios and photon statistics predict at this S1 size? |
| P078 | 09-15 | NUC | Isospin violation and the isovector suppression factor for xenon ★ | How much can isovector couplings enhance high-energy rates relative to isoscalar? |
| P079 | 09-15 | PROJ | Optimal ROI and fiducial design for the next LZ high-energy analysis | What ROI/FV maximises sensitivity at fixed MSSI expectation? |
| P080 | 09-15 | EXO | Millicharged and strongly interacting heavy particles | Could exotic charged or strongly interacting relics give a single 248 keV recoil? |
| P081 | 09-15 | STAT | Posterior predictive counts for LZ's next release | What are P(0), P(1), P(≥2) events in the next exposure under the corpus posterior? |
| P082 | 09-16 | IDM | Global direct-detection combination for inelastic dark matter ★ | Combining all available exposures, what (m_χ, δ, σ) region remains? |
| P083 | 09-16 | BKG | Base rates: single-event anomalies in direct detection 2008–2026 | How often have ~3σ single-event anomalies turned out to be signal? |
| P084 | 09-16 | MODEL/COMP | Sommerfeld-enhanced annihilation and CMB/dwarf constraints on Higgsino-like fits ★ | Are TeV electroweak multiplets that fit the event excluded by indirect probes? |
| P085 | 09-16 | HALO | Earth's velocity on 16 June 2023 and gravitational focusing | What is the exact v_E and its effect on δ_max and the June/December ratio? |
| P086 | 09-16 | COMP | BBN and CMB constraints on the mediator required for the inelastic cross-section | Is the light mediator needed for large σ compatible with cosmology? |
| P087 | 09-16 | XEXP | The atmospheric-neutrino floor in the extended window | What CEνNS rate and uncertainty does the extended ROI face and does it limit future tests? |
| P088 | 09-16 | PROJ | Decisive-test timeline: LZ, XENONnT and PandaX-4T through 2028 | When does each experiment reach 3σ/5σ discrimination for the best fit? |
| P089 | 09-16 | EFT | Identical spectra, different couplings: the isoscalar/isovector degeneracies of Table S6 | Which Lagrangians are spectrally degenerate and what coupling ratios relate them? |
| P090 | 09-16 | STAT | Propagating the 100% MSSI uncertainty into the global significance | How does a factor-2 (or larger) MSSI systematic change the local and global p-values? |
| P091 | 09-16 | STAT | The corpus posterior after two weeks | Combining P001–P090, what are the updated explanation probabilities? |
| P092 | 09-16 | IDM | Sidereal and diurnal effects on the high-velocity tail | Is there any time-of-day information in a δ ≈ 300 keV event? |
| P093 | 09-16 | BKG | Event-topology odds: single NR versus wall MSSI versus charge-lost ER versus accidental | Combining all topological handles, what are the odds ratios among the mundane hypotheses and the NR hypothesis? |
| P094 | 09-16 | MODEL | Multi-component dark matter with a populated excited state | If χ₂ is a stable relic fraction, what direct-detection signals follow? |
| P095 | 09-16 | NUC | Migdal and bremsstrahlung accompaniment of a 248 keV nuclear recoil | How many extra electrons/photons accompany the NR and can they explain the −1.5σ position? |
| P096 | 09-16 | COMP | Large liquid-scintillator detectors as high-energy recoil detectors | Could Borexino/JUNO/SNO+ see the same model through ¹²C recoils? |
| P097 | 09-16 | HALO | A dark disk or co-rotating substructure ★ | Would a slow co-rotating component change anything for δ ≈ 300 keV? |
| P098 | 09-16 | RESP | Energy resolution at 250 keV from quanta statistics | Does the ±23 keV statistical uncertainty follow from Nq fluctuations and band widths? |
| P099 | 09-16 | PROJ | Decisive tests and a falsifiable forecast for LZ's next release | Which analysis, exposure and timeline settle the question, and what do we predict? |
| P100 | 09-16 | STAT/review | Two weeks after: a critical review of interpretations of the LZ event | Where does the evidence stand across all categories? |

### Allocation summary
STAT 13 · BKG 14 · RESP 6 · IDM 11 · EFT 7 · MODEL 7 · NUC 6 · HALO 7 · XEXP 8 · COMP 10 · EXO 4 · PROJ 7 = 100.
Posting dates: 09-03 (6), 09-04 (6), 09-05 (8), 09-08 (10), 09-09 (10), 09-10 (10), 09-11 (10), 09-12 (10), 09-15 (11), 09-16 (19).

### Execution notes for Phase 3
- Batches of ten in ID order. Each paper: `code/P0XX_*.py` → `work/P0XX/details.md` → `provenance/P0XX.json` → `papers/P0XX.md` → ledger row.
- Shared utilities live in `code/common/lzcommon.py` (validated against wimprates to ≤ 5% and against the paper's kinematic table entries).
- Later papers must read `results_ledger.csv` and cite/dispute earlier headline results by P-number.
- Data requests are filed only when a paper's headline depends on a public dataset (expected candidates: LZ Data Release/HEPData for P021, P052, P082; XENONnT/PandaX-4T HEPData for P035; DAMA/LIBRA spectra for P028).
