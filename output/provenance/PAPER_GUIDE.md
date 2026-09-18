# Paper-writing guide (for every paper P001–P100)

You are writing ONE paper of a simulated 100-paper community response to the LZ paper arXiv:2609.02823
(posted 2 September 2026), which reports a single ~248 keV nuclear-recoil-like event (S1c = 540.1 phd,
S2c = 9268 phd, 16 June 2023 21:22:39 UTC) in 2.84 t·yr, local 3.4σ, global 2.6σ. The simulated date of
your paper is given in your assignment. You are a research group of the stated profile. No real names.

## Absolute rules
1. **No internet, no web tools, no installs.** Inputs are: the LZ paper (`inputs/LZ_arXiv_2609.02823_fulltext.tex`,
   the source `.tex` files and figure PDFs in `inputs/arXiv_2609.02823_source/`, PNGs in `inputs/figures_png/`),
   the corpus files under `output/` (dossier `output/00_evidence_dossier.md`, plan `output/01_landscape_and_plan.md`,
   earlier papers `output/papers/P0XX.md`, ledger `output/results_ledger.csv`), and your own prior knowledge.
   Never claim knowledge of anything released after 2 September 2026. If you use a number from memory
   (another experiment's limit, an astrophysical parameter, a nuclear datum, a formula), flag it as recalled with
   a reliability (certain | likely | uncertain). **Never invent experimental data.**
2. **Python only via `.venv/bin/python`**, run from the simulation root `/Users/reza/LZ_simulation`
   (WimPyDD needs cwd = root). Never use `/usr/bin/python3`, pip, uv, curl, brew, `node`, or any other interpreter or
   program not listed in PROMPT.md §2b (if the dataviz skill suggests running a JavaScript palette validator, skip it).
3. **Write only under `output/`** (plus WimPyDD-generated files, which you must list). Do NOT edit shared files
   (`results_ledger.*`, `process_log.md`, `tool_inventory.md`, other papers). You return your ledger row and tool list in your final report.
4. Shared library: `output/code/common/lzcommon.py` — import with
   `import sys; sys.path.insert(0,'output/code'); from common import lzcommon as lz`. It has: LZ paper numbers (`lz.LZ`),
   tuned NEST parameters, the local-significance tables (`lz.LSIG`, `lz.OSIG`), Baxter-2021 SHM, kinematics
   (`vmin_kms`, `E_R_range_keV`, `delta_max_kev`, `m_chi_min_gev`), Helm FF, `dRdE_SI` (Helm approximation, validated
   vs wimprates ≤5% below 100 keV), nestpy wrappers (`nest_nr_yields`, `nest_er_yields`, with the LZ p(E) break),
   stats helpers, and a WimPyDD wrapper (`wd_halo(day_of_year=…)`, `wd_hamiltonian(name,{op:(c0,c1)})`, `wd_rate(...)`
   in events/(t·yr·keV)). WimPyDD (shell-model response functions) is the reference for NREFT and inelastic spectra;
   it differs from Helm by ×3–6 at 200–250 keV. WimPyDD coupling convention (SETTLED by P003 via digitising Fig. 1, confirmed by P007): WimPyDD c^0 = c_p + c_n,
   so LZ/Anand unit coupling c_i^s = 1/m_v^2 corresponds to WimPyDD c^0 = 2/m_v^2 (factor 4 in rate); use
   `lz.wd_c_from_anand(c0, c1)`. ALSO: `wd_halo` now passes an explicit v_min grid (P002 found the default grid
   truncated the June halo at 795 km/s); rates near delta_max are only reliable with the patched wrapper.
   HALO LABELLING (P035 finding): `lz.wd_halo()` with no day is WimPyDD's single-day SUN-FRAME halo (v_E = 250.6 km/s,
   no Earth orbital motion), not an annual average; call it "Sun-frame" in your paper, or average over `day_of_year`
   for a true annual mean (they differ by x1.08/1.7/10 at delta = 300/350/380 keV). LZ itself appears to have used the
   time-averaged SHM (P007, P021).
   WimPyDD PITFALL (P031): `eft_hamiltonian` treats default lambda arguments as shared, named Hamiltonian parameters, so
   `{1: lambda c=v: [...]}` for several operators can silently share/overwrite `c`; build coefficient functions with
   closures (def make(v): return lambda: [...]) or distinct parameter names, and check one piece at a time.
   Also: directdm 2.2.2's 5/4-flavour classes crash under numpy 2.5.3 (`np.delete` out-of-range); use WC_3flavor or the
   in-script wrapper from `output/code/P031_directdm_matching.py`.
   Prefer `WD.diff_rate` (used by `wd_rate`) over `wimp_dd_rate`, which writes response-function files.
   Do not modify `lzcommon.py`; put paper-specific code in your own script.
5. **Do the physics honestly.** Compute what you claim; state approximations; report the numbers you actually obtained
   with the precision the method supports. Null and sceptical results are welcome. Later papers should use or dispute
   earlier headline results (cite by P-number; read `output/results_ledger.csv` and the relevant `output/papers/P0XX.md`).

## Deliverables (in this order)
1. `output/code/P0XX_<slug>.py` — one or more scripts, runnable from the root; save result tables (CSV/JSON) into
   `output/work/P0XX/` and figures into `output/work/P0XX/figures/` (matplotlib Agg; PNG). Run them and check them.
2. `output/work/P0XX/details.md` — complete research record: motivation & framework; every equation/derivation;
   all inputs with source (paper section/table/figure, library function, or recalled with flag); every intermediate
   and final number; validation checks and robustness variations; figures with captions; result tables; failed
   approaches; extended discussion; full reference list; and a final section **"Tools and provenance"** mirroring the JSON.
3. `output/provenance/P0XX.json` with EXACTLY these keys:
   `id, title, date, agent_tools, software, scripts_and_commands, local_inputs, recalled_knowledge, datasets,
   data_requests, failed_or_abandoned, wimpydd_generated_files`.
   - `agent_tools`: list of {tool, count, detail} e.g. {"tool":"Read","count":3,"detail":"fulltext.tex lines 590-690; P002.md; ledger"},
     Bash entries listing each command run.
   - `software`: list of {name, version, functions} e.g. {"name":"scipy","version":"1.18.1","functions":"integrate.quad, stats.poisson"};
     versions from `environment/ENVIRONMENT_versions.txt` (python 3.12.13, numpy 2.5.3, scipy 1.18.1, sympy 1.14.0,
     matplotlib 3.11.2, pandas 3.0.5, iminuit 2.32.0, emcee 3.1.6, wimprates 0.5.0, nestpy 2.1.1, WimPyDD 2.0.4,
     directdm 2.2.2, astropy 8.0.1, periodictable 2.1.0, radioactivedecay 0.6.1, pymupdf 1.28.2, uncertainties 3.2.3, …).
     Include `common/lzcommon.py` as software. Say "none: derived by hand" for pure algebra.
   - `scripts_and_commands`: list of {path, purpose, command}.
   - `local_inputs`: list of {file, parts_used}.
   - `recalled_knowledge`: list of {item, presumed_source, reliability}.
   - `datasets`: "none" or list; `data_requests`: "none" or list of {id, status}; `failed_or_abandoned`: list of strings.
4. `output/papers/P0XX.md` — strictly one page, EXACT structure below. Body (Abstract→Conclusion) ≤ 550 words; check with
   `wc -w` on the body. WHOLE FILE ≤ 950 words (`wc -w output/papers/P0XX.md`): HEADLINE RESULT ≤ 50 words (one sentence),
   KEY NUMBERS ≤ 6 bullets and ≤ 110 words in total, CONFIDENCE ≤ 25 words, TOOLS line ≤ 60 words, References ≤ 90 words.
   Every number must appear in details.md or a script output referenced there.

```
# P0XX: <Title>
- Simulated arXiv date: 2026-09-DD
- Primary arXiv category: <hep-ph | hep-ex | astro-ph.CO | astro-ph.HE | nucl-th | physics.data-an | physics.ins-det>
- Author profile: <kind of group; no real names>
- Category (from your taxonomy): <STAT | BKG | RESP | IDM | EFT | MODEL | NUC | HALO | XEXP | COMP | EXO | PROJ>
- Builds on / responds to: <LZ paper; P-numbers>

**Abstract.** (≤ 80 words)
**Question and approach.** (≤ 120 words)
**Results.** (≤ 220 words; key equation(s) and numbers; at most one small table or one figure reference)
**Caveats.** (≤ 80 words)
**Conclusion.** (≤ 50 words)
**References.** (≤ 6 real works you are confident exist and predate Sept 2026, plus the LZ paper and corpus P-numbers)

---
**HEADLINE RESULT:** <one sentence, specific and quantitative>
**KEY NUMBERS:** <bullets with units>
**RESULT TYPE:** computed | estimated | qualitative
**STANCE ON THE EVENT:** supports DM interpretation | favours background/artifact | statistical reassessment (strengthens/weakens) | neutral (tool, projection, or constraint)
**CONFIDENCE:** high | medium | low, and one sentence why
**DATA DEPENDENCE:** none | provisional pending DR-### (say which results would change)

**TOOLS (summary):** <one line: every tool/package/version/function, files Read, count of recalled values; "complete record in provenance/P0XX.json">
**DETAILS:** work/P0XX/details.md · provenance/P0XX.json · code/P0XX_*.py · data requests: <none | DR-###>
```

## Data requests
Only if your headline genuinely needs a public dataset that was public on or before 2 Sept 2026 (e.g. LZ Data Release/HEPData,
another experiment's HEPData, a nuclear-data table). Do NOT fetch anything. Write `output/data_requests/DR-###.md` using the
template in PROMPT.md §4 (check existing DR numbers in `output/data_requests/` first and use the next free number; if two papers
collide, the coordinator will renumber), mark affected results "provisional pending DR-###", and mention it in your report.
Do not edit `output/data_requests/index.csv` (the coordinator does).

## Final report (your last message; keep it under ~450 words plus the JSON)
1. Headline result and 3–6 key numbers.
2. Files written.
3. A JSON object for the ledger with keys: id, version ("v1"), date, arxiv_category, taxonomy_category, title, question,
   headline_result, key_numbers, model_or_mechanism, stance, result_type, confidence, builds_on, agent_tools (short string),
   software_and_packages (short string), scripts (paths), local_inputs (short string), recalled_knowledge (count + short string),
   datasets_used, data_requests, data_dependence.
4. A compact list of every tool call you made (tool + target/command), so the coordinator can log it.

**Exothermic pitfall (P058):** `lz.vmin_kms(E, m, delta)` returns a *negative* value for delta < 0 at E below E* = |delta| mu/m_N; take `abs()` (WimPyDD's own `diff_rate` with delta < 0 and `lz.E_R_range_keV` are correct).
**astropy offline (P085):** `Time.ut1`/precise frames fail until `astropy.utils.iers.conf.auto_max_age` is set explicitly (e.g. to a large number) and `auto_download=False`; `get_body_barycentric_posvel` with the built-in ephemeris works offline; AltAz does not (P067) — use analytic GMST.
