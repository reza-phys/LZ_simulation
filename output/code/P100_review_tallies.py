"""P100 -- Two weeks after: a critical review of the interpretations of the LZ 248 keV event.

Builds the per-category verdict table (hand-authored from reading the corpus; machine-readable
CSV/MD/JSON), and tallies -- from output/results_ledger.csv plus the P090 page, which exists but
has no ledger row yet -- stances, confidences, result types, categories, data-request
dependencies, recalled-knowledge reliability flags and halo-frame conventions.  Nothing new is
computed beyond counts and ratios.  Runtime ~2 s.

Run from the simulation root:
    .venv/bin/python output/code/P100_review_tallies.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(".")
sys.path.insert(0, "output/code")
from common import lzcommon as lz  # noqa: E402  (imported for the LZ constants only)

WORK = ROOT / "output/work/P100"
FIG = WORK / "figures"
TAB = WORK / "tables"
for d in (WORK, FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

CATS = ["STAT", "BKG", "RESP", "IDM", "EFT", "MODEL", "NUC", "HALO", "XEXP", "COMP", "EXO", "PROJ"]

# --------------------------------------------------------------------------------------------
# 1. Ledger + P090 page (P091 absent at the time of writing -- checked, see details.md)
# --------------------------------------------------------------------------------------------
led = pd.read_csv(ROOT / "output/results_ledger.csv")
N_LEDGER = len(led)  # 97 when this review started; the coordinator appended P090 (98) while it was written

p090_path = ROOT / "output/papers/P090.md"
p091_path = ROOT / "output/papers/P091.md"
extra_rows = []
if p090_path.exists() and "P090" not in set(led["id"]):
    extra_rows.append(
        dict(
            id="P090",
            taxonomy_category="STAT",
            stance="statistical reassessment (neutral to slightly strengthens: MSSI systematic cannot lower the local significance below 2.7 sigma)",
            confidence="medium - exact within P052's surrogate likelihood",
            result_type="computed",
            data_requests="none",
            data_dependence="none",
            recalled_knowledge="4 recalled items",
            date="2026-09-16",
            title="Propagating the 100 % MSSI uncertainty into the local and global significance",
        )
    )
P091_PRESENT = p091_path.exists()
if P091_PRESENT and "P091" not in set(led["id"]):
    extra_rows.append(
        dict(
            id="P091",
            taxonomy_category="STAT",
            stance="statistical reassessment (neutral: conventional backgrounds eliminated; the residual is prior-dominated between artefact/unknown and inelastic DM)",
            confidence="medium - algebra exact; class likelihoods inherit corpus judgements",
            result_type="computed (posterior algebra, Monte Carlo) / estimated",
            data_requests="none",
            data_dependence="none (L_F would tighten with DR-003 via P093)",
            recalled_knowledge="3 recalled items (all certain)",
            date="2026-09-16",
            title="The corpus posterior after two weeks: updating the explanation probabilities with P001-P099",
        )
    )
allp = pd.concat([led, pd.DataFrame(extra_rows)], ignore_index=True) if extra_rows else led.copy()
N_PAPERS = len(allp)


def primary_cat(s: str) -> str:
    s = str(s).strip()
    tok = re.split(r"[/ (]", s)[0].strip()
    return tok if tok in CATS else "OTHER"


def stance_class(s: str) -> str:
    s = str(s).lower()
    if s.startswith("supports"):
        return "supports DM (by elimination)"
    if s.startswith("favours"):
        return "favours background/artefact"
    if s.startswith("statistical"):
        if "weakens" in s:
            return "stat. reassessment: weakens"
        if "strengthens" in s:
            return "stat. reassessment: strengthens"
        return "stat. reassessment: neutral"
    if s.startswith("neutral"):
        return "neutral (tool/projection/constraint)"
    return "other"


def conf_class(c: str) -> str:
    c = str(c).lower().strip()
    for k in ("high", "medium", "low"):
        if c.startswith(k):
            return k
    return "other"


allp["cat"] = allp["taxonomy_category"].map(primary_cat)
allp["stance_class"] = allp["stance"].map(stance_class)
allp["conf_class"] = allp["confidence"].map(conf_class)
allp["rtype"] = allp["result_type"].astype(str).str.split(r"[ (/]").str[0]
allp["dr_provisional"] = allp["data_dependence"].astype(str).str.contains("provisional")
allp["dr_inherits"] = allp["data_dependence"].astype(str).str.contains("inherit")
allp["dr_ids"] = allp["data_dependence"].astype(str).str.findall(r"DR-00\d").map(lambda x: ",".join(sorted(set(x))))
allp["recalled_uncertain"] = allp["recalled_knowledge"].astype(str).str.lower().str.contains("uncertain")


def recalled_count(s: str) -> float:
    m = re.match(r"\s*(\d+)", str(s))
    return float(m.group(1)) if m else np.nan


allp["recalled_n"] = allp["recalled_knowledge"].map(recalled_count)

tallies = {
    "n_papers_tallied": int(N_PAPERS),
    "n_ledger_rows": int(len(led)),
    "added_from_pages_not_in_ledger": [r["id"] for r in extra_rows],
    "P091_present_at_finalisation": bool(P091_PRESENT),
    "stance": allp["stance_class"].value_counts().to_dict(),
    "stance_coarse": {
        "neutral": int((allp["stance_class"].str.startswith("neutral")).sum()),
        "statistical reassessment": int((allp["stance_class"].str.startswith("stat")).sum()),
        "supports DM": int((allp["stance_class"].str.startswith("supports")).sum()),
        "favours background/artefact": int((allp["stance_class"].str.startswith("favours")).sum()),
    },
    "confidence": allp["conf_class"].value_counts().to_dict(),
    "result_type": allp["rtype"].value_counts().to_dict(),
    "category": allp["cat"].value_counts().reindex(CATS).fillna(0).astype(int).to_dict(),
    "dr_provisional_ids": allp.loc[allp["dr_provisional"], ["id", "dr_ids"]].values.tolist(),
    "dr_inherits_ids": allp.loc[allp["dr_inherits"], ["id", "dr_ids"]].values.tolist(),
    "n_recalled_uncertain_papers": int(allp["recalled_uncertain"].sum()),
    "recalled_items_total_declared": float(np.nansum(allp["recalled_n"])),
    "recalled_items_median_per_paper": float(np.nanmedian(allp["recalled_n"])),
    "recalled_items_max": [allp.loc[allp["recalled_n"].idxmax(), "id"], float(allp["recalled_n"].max())],
}

# stance x category and confidence x category matrices
sc = pd.crosstab(allp["cat"], allp["stance_class"]).reindex(CATS).fillna(0).astype(int)
cc = pd.crosstab(allp["cat"], allp["conf_class"]).reindex(CATS).fillna(0).astype(int)
sc.to_csv(TAB / "P100_stance_by_category.csv")
cc.to_csv(TAB / "P100_confidence_by_category.csv")

# supports-DM papers: are they all eliminations?
sup = allp[allp["stance_class"].str.startswith("supports")]
tallies["supports_DM_all_by_elimination"] = bool(all("elimin" in s.lower() or "weakly" in s.lower() for s in sup["stance"]))
tallies["supports_DM_ids"] = sup["id"].tolist()

# statistical-reassessment sub-labels
tallies["stat_reassessment_ids"] = allp.loc[allp["stance_class"].str.startswith("stat"), ["id", "stance_class"]].values.tolist()

# --------------------------------------------------------------------------------------------
# 2. Halo-frame convention scan over the one-page papers (P035 finding)
# --------------------------------------------------------------------------------------------
PAT = {
    "Sun-frame": re.compile(r"sun[- ]?frame", re.I),
    "annual/time-averaged": re.compile(r"annual[- ](average|averaged|mean)|time-averaged", re.I),
    "16 June (event date)": re.compile(r"16 june|june halo", re.I),
}
frame_rows = []
for f in sorted((ROOT / "output/papers").glob("P0*.md")) + sorted((ROOT / "output/papers").glob("P1*.md")):
    if f.stem == "P100":
        continue
    txt = f.read_text(encoding="utf-8", errors="ignore")
    row = {"id": f.stem}
    for k, p in PAT.items():
        row[k] = len(p.findall(txt))
    row["any"] = any(row[k] for k in PAT)
    frame_rows.append(row)
fr = pd.DataFrame(frame_rows).set_index("id")
fr.to_csv(TAB / "P100_halo_frame_mentions.csv")
frame_summary = {
    "papers_mentioning_any_frame": int(fr["any"].sum()),
    "Sun-frame_only": fr.index[(fr["Sun-frame"] > 0) & (fr["annual/time-averaged"] == 0)].tolist(),
    "annual_only": fr.index[(fr["Sun-frame"] == 0) & (fr["annual/time-averaged"] > 0)].tolist(),
    "both_Sun_and_annual": fr.index[(fr["Sun-frame"] > 0) & (fr["annual/time-averaged"] > 0)].tolist(),
    "June_date_only": fr.index[(fr["16 June (event date)"] > 0) & (fr["Sun-frame"] == 0) & (fr["annual/time-averaged"] == 0)].tolist(),
}
tallies["halo_frame"] = frame_summary

# --------------------------------------------------------------------------------------------
# 3. Convention spreads flagged in the corpus (numbers taken from the cited pages; ratios only)
# --------------------------------------------------------------------------------------------
conv = pd.DataFrame(
    [
        dict(item="'best-fit' L10 rate in 200-270 keV per t.yr", a_label="band-anchored (P020/P079)", a=0.352,
             b_label="ROI-anchored (P069/P081/P088/P099)", b=0.121, ratio=0.352 / 0.121,
             consequence="P020: 2.4 events expected in 6.76 t.yr, P(0)=9%; P081: P(>=1)=0.25. Both called 'best fit'. LZ's Table I (1.0 event in the whole ROI) is the ROI-anchored one; P079 concedes the x2.7."),
        dict(item="energy resolution used for spectral smearing (keV at 248 keV)", a_label="P008/P027/P052-variant (LZ's quoted stat)", a=23.0,
             b_label="P009/P021/P038/P089/P098 (quanta statistics)", b=11.4, ratio=23.0 / 11.4,
             consequence="P052's reproduction of Tables S6/S7 improves (rms 0.52 -> 0.42 sigma) with 23 keV, while P098 shows 23 keV needs 4x the physical variance. Either LZ smears with its systematic or the tables encode something else; DR-002."),
        dict(item="wall-MSSI depth scale lambda (cm)", a_label="P033 photon-transport MC (12 keV Compton class)", a=4.3,
             b_label="P079 from LZ's 5.4 t / 4.7 t MSSI table", b=1.23, ratio=4.3 / 1.23,
             consequence="P090: lambda=4.3 cm implies a fiducial MSSI factor 7.3 and 2.76 sigma position-blind (3.16 sigma position-aware). The two lambdas may describe different populations (Compton class vs low-S2 leakage); unreconciled."),
        dict(item="halo frame: inelastic rate ratio annual/Sun-frame at delta=300/350/380 keV", a_label="annual average", a=1.0,
             b_label="Sun-frame (WimPyDD default)", b=1.0, ratio=np.nan,
             consequence="x1.08 / x1.7 / x10 (P035, P068). Moves P021's peak 380 -> 370 keV, P007's delta(N=1) by +-5 keV, P035's absolute limits above 360 keV by x1.7-10, P082's region width by 22%. LZ used the time-averaged SHM (P007, P021)."),
        dict(item="NR-band width at 540 phd (dex)", a_label="LZ drawn 10-90% lines (P024)", a=0.021,
             b_label="NEST / AmBe (P024, P098)", b=0.031, ratio=0.021 / 0.031,
             consequence="Paper's '1.5 sigma' is right; drawn lines 1.5x too narrow. P052: using 0.021 dex would cost 1.4 sigma. Corpus converged on 0.031-0.036."),
        dict(item="neighbourhood background at S1c>500 phd, |d|<2 sigma (events/2.84 t.yr)", a_label="LZ model (P093 case i)", a=3.5e-4,
             b_label="corpus-revised incl. artefact/ER classes (P093 case ii)", b=9.4e-4, ratio=9.4e-4 / 3.5e-4,
             consequence="The corpus raised LZ's own background near the event x2.7, dominated (64%) by an artefact class with no identified mechanism (P041). P001: b 2e-4 -> 1e-3 costs ~0.4 sigma. No paper labelled this 'favours background'."),
    ]
)
conv.to_csv(TAB / "P100_convention_spreads.csv", index=False)

# --------------------------------------------------------------------------------------------
# 4. Recalled-with-'uncertain' dependencies (which headline would move)
# --------------------------------------------------------------------------------------------
recalled_dep = pd.DataFrame(
    [
        dict(paper="P076", recalled="IceCube/ANTARES W+W- and channel factors (x3 each way)", headline_at_risk="Higgsino excluded at every delta by >=15-770x; dark photon/Z' verdicts at delta 290-370 keV", robust="Higgsino exclusion survives x3 (margin >=15x when stalled); dark-photon/Z' boundaries move by ~20-30 keV"),
        dict(paper="P084", recalled="Fermi dSph, H.E.S.S., Planck, line anchors (x2-10)", headline_at_risk="Doublet 'allowed' at 0.16-0.8 of limits (H.E.S.S. bracket 0.4-3.2 could flip it); triplet excluded x4.7 [2.3-9.3]", robust="Y>=3/2 exclusions (x36-8300) survive any bracket"),
        dict(paper="P054", recalled="dijet/dilepton Z' ceilings (uncertain)", headline_at_risk="thermal U(1)_B reaches delta<=343/319/277 keV; B-L 298/231 keV", robust="contact-limit count 0.1-0.3 events and m_Z'<~m_chi requirement are coupling-independent"),
        dict(paper="P082/P059", recalled="archival ROI edges, SR1 zero high-S1c events (uncertain)", headline_at_risk="kappa_hat x0.76, Z -0.06..-0.11 sigma", robust="region shrinks only 2-3%; joint region itself is LZ-driven"),
        dict(paper="P069/P088/P099", recalled="untouched XENONnT/PandaX-4T exposures (+-50%), XLZD start", headline_at_risk="2.3-5.2 hidden events; 5-sigma dates Apr 2027 / 2033", robust="'verdict on disk' qualitatively; dates +-1 yr"),
        dict(paper="P003/P016/P058", recalled="2024 low-energy tolerance N_max ~3-5 events (uncertain)", headline_at_risk="compatible/excluded operator boundary (O13s, O3v, O8v at N_lo 5-10)", robust="operator ordering and O1/O4/O11 exclusion (N_lo 28-2750)"),
        dict(paper="P005/P015/P028/P035", recalled="other experiments' ROI edges, DAMA rate/exposure, PICO/CRESST exposures", headline_at_risk="0-0.23 expected events elsewhere; NaI 7e3-4e4 below sensitivity", robust="kinematic blindness of light targets (exact)"),
        dict(paper="P083", recalled="45-anomaly compilation (28 certain, 15 likely, 2 uncertain)", headline_at_risk="r = 0.062 (95% < 0.23) and P(DM)=0.012", robust="only the sign: unpredicted ~3 sigma hints rarely survive; confidence self-rated low"),
        dict(paper="P024/P041/P056/P098", recalled="extraction efficiency 0.80, position residual 2%, D_T/D_L (uncertain)", headline_at_risk="band 1.2-1.7 sigma; ER leakage at the event 7e-7; sigma_E 10.2-12.4 keV", robust="floor 0.023 dex and 11 vs 23 keV conclusion"),
        dict(paper="P073/P042/P077", recalled="Skin/OD geometry and light yields (uncertain)", headline_at_risk="veto factor 0.46 (0.10-0.79); PSD separation 1.4-8 sigma", robust="direction of the veto penalty; RFR-vs-event 6-8 sigma"),
    ]
)
recalled_dep.to_csv(TAB / "P100_recalled_uncertain_dependencies.csv", index=False)

# --------------------------------------------------------------------------------------------
# 5. Data-request dependency map
# --------------------------------------------------------------------------------------------
dr_map = pd.DataFrame(
    [
        dict(dr="DR-001", requested_by="P012 (P045)", content="L10 signal normalisation and two-sided interval tables", provisional=["P012 d10 = 0.28 (factor 2)", "P045 lower-edge identification (0.31-0.38 vs 0.105)"], inherits=["P025 L10 annihilation estimate", "P044 d10 comparison", "P068 d10 scale", "P046 L10 spin response"], status="pending"),
        dict(dr="DR-002", requested_by="P052", content="event list, per-component 2D PDFs, NR-band mu/sigma, toy q0", provisional=["P052 toy Z 3.0-3.2 sigma and isoscalar-inelastic -0.5 sigma offset"], inherits=["P090 (P052 engine, 0.3 sigma deficit)", "P016/P021/P027/P038/P081 (b_H anchor and 1D proxy would be replaced)"], status="pending"),
        dict(dr="DR-003", requested_by="P093", content="waveform-level quantities and calibration templates for the candidate", provisional=["P093 NR:ER and NR:ART odds (+0.5/-0.6 dex)", "P099 artefact kill condition"], inherits=["P061 L_acc (artefact 0.006, ER 2.8e-3 residuals)", "P081 mixture weights via P061", "P077 PSD LR <= 2-5"], status="pending"),
    ]
)
dr_map.to_json(TAB / "P100_data_request_map.json", orient="records", indent=1)

# --------------------------------------------------------------------------------------------
# 6. The verdict table (hand-authored from the corpus; the central deliverable)
# --------------------------------------------------------------------------------------------
verdict = pd.DataFrame(
    [
        dict(category="STAT", n_papers=int(tallies["category"]["STAT"]),
             established="Local 3.4 sigma reproduced from published summaries to 3.0-3.2 sigma; global 2.6 sigma confirmed by two independent toy calibrations (N_eff 12-14; 2.5 sigma once corpus spectra are added); local Z is only logarithmically sensitive to the spectrum; model-marginalised B_DM = 16-29; P(DM|event) = 0.015 (68% 0.002-0.14) under a sceptical community prior (P061; base rate 0.06, P083) but 0.25 (0.08-0.54) under the dossier prior (P091), i.e. prior-dominated by a factor 17; explanation-class posterior: artefact/unknown 0.64, inelastic DM 0.23, everything else < 0.04 (P091); MSSI systematic cannot move Z below 2.7 sigma",
             load_bearing="P052, P016, P008, P071, P027, P061, P083, P090, P091",
             confidence="high (LEE, reproduction); medium (posteriors)",
             open="0.3 sigma residual in the 250-500 phd background (DR-002); P(DM) hinges on pi_DM and on an unsized artefact likelihood L_F (60% of P091's variance) for a class with no identified mechanism; 23 vs 11 keV smearing"),
        dict(category="BKG", n_papers=int(tallies["category"]["BKG"]),
             established="Every modelled background is dead as origin: wall MSSI needs k >= 620-1.4e5 vs sidebands k < 1.64; RFR MSSI 1e-8; neutrons <= 5e-4 (any origin, incl. muons), fission 1e-9, accidentals need x708 high-S1 mismodelling, activation <= 1e-9, 214Pb transient >= 2.7e5; ER leakage < 0.3% but its tail is untestable (1e-14-1e-2); a single NR beats every modelled class by >= 10^3.6 per unit signal, the artefact class by only 10^2.5",
             load_bearing="P004, P033, P073, P013, P049, P022, P010, P056, P041, P093",
             confidence="high (modelled backgrounds); medium-low (ER tail, artefact residual)",
             open="unmodelled artefact/charge-poor-ER classes (85% of the corpus's residual background mass) testable only with waveforms (DR-003); wall depth scale 1.2 vs 4.3 cm; x2.5 prompt-MSSI count discrepancy (P073)"),
        dict(category="RESP", n_papers=int(tallies["category"]["RESP"]),
             established="Event energy 246-248 keV on the efficiency plateau (eps 0.93-0.95) once LZ's own contour scale (Nq +8% over Table S5) is used; statistical resolution 11 keV, not 23; NR band 0.031-0.036 dex (drawn lines 1.5x too narrow), event a 1.5 sigma (P~6%) band member; gains contribute <= 4 keV; Migdal/brems shift the wrong way; PSD at 540 phd cannot exceed LR 2-5",
             load_bearing="P009, P064, P098, P024, P043, P056, P077, P095",
             confidence="high (energy, band width); medium (tails, PSD)",
             open="cause of the 8% Nq offset (beta rounding, W, NEST version); AmBe 200-330 keV band data unpublished; ER NR-ward tail shape; +-23 keV (sys) meaning"),
        dict(category="IDM", n_papers=int(tallies["category"]["IDM"]),
             established="Kinematically allowed for delta <= 387 keV at 1 TeV on 16 June; the isoscalar-O1 likelihood peaks at delta = 380 keV (3.6 sigma) beyond LZ's grid, but this preference is manufactured by the 600 phd edge (1000 phd: 355 keV, 2.9 sigma) and delta >= 366 keV is disfavoured by LZ's empty 600-1000 phd sideband (P(0) = 0.02); global region m >= 400 GeV, delta 335-390 keV, sigma_n 1e-41-1e-36 cm2; Higgsino window 358-380 keV; exothermic origin B <= 2; mass unmeasurable",
             load_bearing="P002, P085, P021, P038, P082, P007, P072, P062",
             confidence="medium",
             open="halo frame (+-10 keV on every delta); 1D observed-energy proxies throughout; whether LZ will extend the edge; light-mediator variants change little (P051)"),
        dict(category="EFT", n_papers=int(tallies["category"]["EFT"]),
             established="A lone 248 keV recoil selects q^2/q^4-suppressed spin operators (O6, O9, O10, O14, O15, L10-like; N_lo <= 5 low-energy companions) and excludes O1/O4/O7/O11/O12 by 1-3 dex; WimPyDD c0 = 2/m_v^2 settled; L10 d10 = 0.28 (0.15-0.43); Table S6's 40 entries are 24 shapes (14 at >= 3 sigma, = N_eff); light mediators soften but rescue nothing; directdm: only pseudoscalar and dipole x current UV structures",
             load_bearing="P003, P016, P031, P074, P012, P089, P057, P044",
             confidence="high (ordering); medium (boundary: N_max recalled; d10 factor 2)",
             open="DR-001 normalisation; L18v's 2.9 sigma unexplained; operator identity needs 15-60 events"),
        dict(category="MODEL", n_papers=int(tallies["category"]["MODEL"]),
             established="Every UV model fits the event only by tuning delta within 5-30 keV of delta_max or by fitting a free coupling: Higgsino/doublet needs PeV gauginos and is excluded by solar capture; dark photon needs m_A' >= 2.6 GeV (chi2 decay) and 9-35 GeV (Planck/BBN); heavy Z' reaches only delta <= 340 keV; MiDM delta <= 355 keV; Y >= 1 multiplets excluded by indirect detection; composite/dark-baryon and multi-component variants viable but unpredictive",
             load_bearing="P007, P037, P011, P054, P065, P023, P094, P075",
             confidence="medium (recalled collider/indirect limits everywhere)",
             open="no surviving model predicts the rate; thermal relic + LZ + indirect + solar is satisfied only by secluded dark-photon models at delta <= 340 keV"),
        dict(category="NUC", n_papers=int(tallies["category"]["NUC"]),
             established="The event sits 19 keV below the shell-model M node (267 keV; isotopes 251-278), so SI couplings from it carry a x3-5 nuclear-structure factor while Sigma'/Sigma'' couplings are robust to 15%; isospin violation cannot bring N_lo below 7 (r = -0.744); nuclear-excitation hybrids excluded (+16-23 sigma above the NR median); 136Xe-enriched xenon separates spin from inelastic with ~9 events; Lindhard quenching reproduces 248 keV",
             load_bearing="P017, P032, P078, P036, P047, P064, P095",
             confidence="high",
             open="two-body currents and node position (+-10 keV); LZ's density-matrix modifications unknown"),
        dict(category="HALO", n_papers=int(tallies["category"]["HALO"]),
             established="The 16 June date multiplies odds by 1.4-2.6 for delta >= 300 keV (p = 0.086, 1.4 sigma) and is worthless for elastic; v_esc dominates every inelastic quantity (delta_H = 355 +30/-26 keV, kappa(366) spans 5 dex, 13% of halos give no event); no stream, substructure or dark disk removes elastic companions or raises delta_max; exact v_E = 265.7 km/s; diurnal effects nil",
             load_bearing="P006, P020, P018, P068, P030, P055, P097, P085, P092",
             confidence="high (kinematics); medium (tails)",
             open="frame convention used inconsistently across the corpus (Sun-frame vs annual vs June: x1.08/1.7/10 at delta = 300/350/380 keV); Gaia v_esc prior"),
        dict(category="XEXP", n_papers=int(tallies["category"]["XEXP"]),
             established="Nothing published constrains LZ's interval (XENON1T 2018 lies 11-1e5 x above it; NaI, F, Ar, Ge, scintillators kinematically blind); 19 t.yr of xenon data on disk are unexamined above 200 keV and should hold 2.3-5.2 events at the best fit; a 270 keV XENONnT+PandaX-4T reanalysis is the most informative next measurement; the atmospheric-nu floor matters only beyond 30-40 t.yr",
             load_bearing="P035, P005, P015, P028, P059, P069, P087, P096",
             confidence="medium (recalled exposures +-50%; acceptance assumed LZ-like)",
             open="other TPCs' high-S1c acceptance and MSSI; whether a pre-registered joint analysis happens"),
        dict(category="COMP", n_papers=int(tallies["category"]["COMP"]),
             established="Colliders cannot reach the LZ-fitting doublet (c tau 0.7 cm; FCC-hh/muon collider only); indirect detection allows the doublet (S = 1.7) but excludes Y >= 1 multiplets x5-8000; solar capture on core Fe/Ni excludes the Higgsino at every delta (>= 15-770x) and dark photon/Z' above 350-370 keV; Planck removes m_A' < 9 GeV; BBN/FIRAS blind; neutron stars heat to 1600 K for any LZ-fit coupling; no MeV line predicted",
             load_bearing="P014, P048, P025, P084, P076, P026, P086, P039, P066",
             confidence="medium (every external limit recalled, x2-10)",
             open="P076 vs P084 give the doublet opposite verdicts, both on recalled anchors; thermalisation in the Sun bracketed"),
        dict(category="EXO", n_papers=int(tallies["category"]["EXO"]),
             established="Atmospheric CEvNS gives 1-3e-5 events near the event (coherence lost at q = 246 MeV; incoherent Fermi-recoil channel 1.7e-4); no astrophysical neutrino burst, boosted/fast light DM (N_lo >= 350), millicharged, SIMP or Planck-scale relic reproduces a lone recoil",
             load_bearing="P019, P087, P060, P040, P080",
             confidence="high",
             open="the incoherent neutrino channel (8e-6-5e-4) is the only exotic rate within a decade of the corpus background"),
        dict(category="PROJ", n_papers=int(tallies["category"]["PROJ"]),
             established="The decisive data exist: LZ's untouched 6.8 t.yr (0.8 ROI-anchored / 2.4 band-anchored L10 events; posterior P(>=1) = 0.25) plus 10 t.yr in XENONnT/PandaX-4T; a 1000 phd edge is worth more than any FV change (P(5 sigma) 0.43 -> 0.83); at the best fit the present generation gives a 5 sigma population verdict by 2027-28, at x0.3 only XLZD (2033); modulation needs 15-274 t.yr; direction and tungsten add nothing before the 2030s; one new band event must not be read as confirmation (P(DM) 0.03)",
             load_bearing="P020, P081, P099, P038, P079, P088, P069, P034, P050",
             confidence="medium",
             open="the x2.9 'best-fit' normalisation spread; recalled exposures and live fractions; whether LZ will publish position-aware, extended-edge results"),
    ]
)
verdict.to_csv(WORK / "verdict_table.csv", index=False)
verdict.to_json(WORK / "verdict_table.json", orient="records", indent=1)
with open(WORK / "verdict_table.md", "w") as fh:
    fh.write("| category | n | established | load-bearing | confidence | open |\n|---|---|---|---|---|---|\n")
    for _, r in verdict.iterrows():
        fh.write(f"| {r.category} | {r.n_papers} | {r.established} | {r.load_bearing} | {r.confidence} | {r.open} |\n")

# --------------------------------------------------------------------------------------------
# 7. Bias arithmetic (simple ratios from the tallies and quoted corpus numbers)
# --------------------------------------------------------------------------------------------
n_neutral = tallies["stance_coarse"]["neutral"]
n_sup = tallies["stance_coarse"]["supports DM"]
n_fav = tallies["stance_coarse"]["favours background/artefact"]
n_stat = tallies["stance_coarse"]["statistical reassessment"]
n_weak = tallies["stance"].get("stat. reassessment: weakens", 0)
n_str = tallies["stance"].get("stat. reassessment: strengthens", 0)
bias = {
    "fraction_neutral": n_neutral / N_PAPERS,
    "supports_minus_favours": n_sup - n_fav,
    "weakens_minus_strengthens": n_weak - n_str,
    "coupling_fits_from_one_event_(IDM+EFT+MODEL papers)": int(sum(tallies["category"][c] for c in ("IDM", "EFT", "MODEL"))),
    "background_eliminations_(BKG+EXO)": int(tallies["category"]["BKG"] + tallies["category"]["EXO"]),
    "corpus_vs_LZ_neighbourhood_background_ratio": 9.4e-4 / 3.5e-4,
    "P061_weight_on_unknown_U": 0.789,
    "P061_L_acc_artefact_share": 0.006 / 9.3e-3,
    "P_DM_central_P061_P083_P091": [0.015, 0.012, 0.25],
    "P_DM_P091_over_P061": 0.25 / 0.015,
    "P091_artefact_class_posterior": 0.64,
    "P091_inelastic_DM_posterior": 0.23,
    "P_DM_if_U_dropped_(P001, b=1e-3, pi_DM=0.01)": 0.51,
    "N_eff_corpus_vs_LZ": 18.7 / 13.9,
}
tallies["bias"] = bias

with open(WORK / "P100_tallies.json", "w") as fh:
    json.dump(tallies, fh, indent=1, default=str)

# --------------------------------------------------------------------------------------------
# 8. Figure: category x stance (left) and category x confidence (right)
# --------------------------------------------------------------------------------------------
# dataviz reference palette (light surface): categorical slots + sequential blue steps
C_NEUTRAL = "#c3c2b7"   # baseline gray (chrome) for the 'neutral' mass
C_STAT_W = "#2a78d6"    # slot 1 blue
C_STAT_N = "#86b6ef"    # sequential blue 250
C_STAT_S = "#1baf7a"    # slot 3 aqua
C_SUP = "#eb6834"       # slot 2 orange
C_HIGH, C_MED, C_LOW = "#1c5cab", "#6da7ec", "#cde2fb"
ink, ink2, grid = "#0b0b0b", "#52514e", "#e1e0d9"

stance_order = [
    ("neutral (tool/projection/constraint)", "neutral", C_NEUTRAL),
    ("stat. reassessment: weakens", "stat. reassessment: weakens", C_STAT_W),
    ("stat. reassessment: neutral", "stat. reassessment: neutral", C_STAT_N),
    ("stat. reassessment: strengthens", "stat. reassessment: strengthens", C_STAT_S),
    ("supports DM (by elimination)", "supports DM (by elimination only)", C_SUP),
]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8), facecolor="#fcfcfb", gridspec_kw=dict(width_ratios=[1.35, 1]))
y = np.arange(len(CATS))[::-1]
ax = axes[0]
left = np.zeros(len(CATS))
for col, lab, c in stance_order:
    vals = sc[col].values if col in sc else np.zeros(len(CATS))
    ax.barh(y, vals, left=left, color=c, edgecolor="#fcfcfb", linewidth=2, height=0.72, label=lab)
    for yi, v, l in zip(y, vals, left):
        if v >= 1 and col != "neutral (tool/projection/constraint)":
            ax.text(l + v / 2, yi, str(int(v)), ha="center", va="center", fontsize=8, color=ink)
    left += vals
for yi, tot in zip(y, left):
    ax.text(tot + 0.15, yi, str(int(tot)), va="center", fontsize=8.5, color=ink2)
ax.set_yticks(y)
ax.set_yticklabels(CATS, fontsize=9.5, color=ink)
extra_lab = " + " + "/".join(r["id"] for r in extra_rows) + " page" if extra_rows else ""
ax.set_xlabel(f"papers (n = {N_PAPERS}: {N_LEDGER} ledger rows{extra_lab})", fontsize=9, color=ink2)
ax.set_title("Declared stance on the event, by category", fontsize=10.5, color=ink, loc="left")
ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
ax.set_xlim(0, left.max() + 1.5)

ax = axes[1]
left = np.zeros(len(CATS))
for col, lab, c in (("high", "high", C_HIGH), ("medium", "medium", C_MED), ("low", "low", C_LOW)):
    vals = cc[col].values if col in cc else np.zeros(len(CATS))
    ax.barh(y, vals, left=left, color=c, edgecolor="#fcfcfb", linewidth=2, height=0.72, label=lab)
    for yi, v, l in zip(y, vals, left):
        if v >= 2:
            ax.text(l + v / 2, yi, str(int(v)), ha="center", va="center", fontsize=8, color="#ffffff" if col == "high" else ink)
    left += vals
ax.set_yticks(y)
ax.set_yticklabels([])
ax.set_xlabel("papers", fontsize=9, color=ink2)
ax.set_title("Self-declared confidence", fontsize=10.5, color=ink, loc="left")
ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
ax.set_xlim(0, left.max() + 1.0)
for a in axes:
    a.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        a.spines[s].set_color("#c3c2b7")
    a.tick_params(colors=ink2, labelsize=8.5)
    a.xaxis.grid(True, color=grid, linewidth=0.8)
    a.set_axisbelow(True)
fig.suptitle(
    f"LZ 248 keV corpus after two weeks: {n_neutral} neutral, {n_stat} statistical reassessments "
    f"({n_weak} weaken, {n_str} strengthen), {n_sup} 'supports DM' (all by elimination), {n_fav} 'favours background'",
    fontsize=9.5, color=ink2, x=0.01, ha="left", y=0.995,
)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(FIG / "P100_fig1_stance_confidence_by_category.png", dpi=180, facecolor=fig.get_facecolor())

# --------------------------------------------------------------------------------------------
# 9. Console summary
# --------------------------------------------------------------------------------------------
print(f"papers tallied: {N_PAPERS} (ledger {N_LEDGER}; added from pages: {[r['id'] for r in extra_rows]}); "
      f"P090 in ledger: {'P090' in set(led['id'])}; P091 page present: {P091_PRESENT}; P091 in ledger: {'P091' in set(led['id'])}")
print("stance:", tallies["stance_coarse"])
print("stance detail:", tallies["stance"])
print("confidence:", tallies["confidence"])
print("result type:", tallies["result_type"])
print("category:", tallies["category"])
print("DR provisional:", tallies["dr_provisional_ids"])
print("DR inherits:", tallies["dr_inherits_ids"])
print("recalled-uncertain papers:", tallies["n_recalled_uncertain_papers"],
      "| declared items total:", tallies["recalled_items_total_declared"],
      "| median/paper:", tallies["recalled_items_median_per_paper"], "| max:", tallies["recalled_items_max"])
print("halo frame:", json.dumps(frame_summary, indent=0))
print("bias:", json.dumps(bias, indent=0))
print("supports-DM all by elimination:", tallies["supports_DM_all_by_elimination"], tallies["supports_DM_ids"])
print("LZ exposure used for sanity:", lz.LZ.get("exposure_tyr", "n/a") if isinstance(lz.LZ, dict) else "n/a")
print("wrote:", sorted(str(p) for p in WORK.rglob("*") if p.is_file()))
