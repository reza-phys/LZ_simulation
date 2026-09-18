"""
inventory_tools.py -- aggregate the per-paper provenance JSONs into the consolidated tables of
provenance/tool_inventory.md (sections 2-5). Run from the simulation root:

  .venv/bin/python output/code/common/inventory_tools.py > output/work/synthesis/inventory_tables.md

Output: Markdown tables (software x papers, agent-tool totals, local inputs by type, recalled-knowledge
counts by reliability and paper, datasets / data requests).
"""
import json, os, re, collections, csv

ROOT = os.getcwd()
LEDGER = os.path.join(ROOT, "output/results_ledger.json")
PROV = os.path.join(ROOT, "output/provenance")

CANON = [("wimpydd", "WimPyDD 2.0.4"), ("wimpyc", "WimPyC (WimPyDD)"), ("nestpy", "nestpy 2.1.1"), ("wimprates", "wimprates 0.5.0"),
         ("iminuit", "iminuit 2.32.0"), ("emcee", "emcee 3.1.6"), ("directdm", "directdm 2.2.2"), ("astropy", "astropy 8.0.1"),
         ("pymupdf", "PyMuPDF 1.28.2"), ("fitz", "PyMuPDF 1.28.2"), ("pillow", "Pillow 12.3.0"), ("pil", "Pillow 12.3.0"),
         ("radioactivedecay", "radioactivedecay 0.6.1"), ("healpy", "healpy 1.20.0"), ("periodictable", "periodictable 2.1.0"),
         ("uncertainties", "uncertainties 3.2.3"), ("sympy", "sympy 1.14.0"), ("numericalunits", "numericalunits 1.28"),
         ("scipy", "scipy 1.18.1"), ("numpy", "numpy 2.5.3"), ("pandas", "pandas 3.0.5"), ("matplotlib", "matplotlib 3.11.2"),
         ("lzcommon", "common/lzcommon.py"), ("python", "python 3.12.13"), ("multiprocessing", "multiprocessing (stdlib)"),
         ("hand", "none: derived by hand"), ("derived", "none: derived by hand")]


def canon(name):
    n = str(name).lower()
    for k, v in CANON:
        if k in n:
            return v
    return None


def main():
    ledger = json.load(open(LEDGER))
    cat = {r["id"]: r["taxonomy_category"].split()[0].split("/")[0] for r in ledger}
    ids = [r["id"] for r in ledger]
    sw = collections.defaultdict(set)
    swfun = collections.defaultdict(set)
    tools = collections.Counter()
    tools_by_cat = collections.defaultdict(collections.Counter)
    rk_rel = collections.Counter()
    rk_per_paper = {}
    inputs = collections.Counter()
    drs = []
    datasets = []
    wdfiles = []
    failed = 0
    for pid in ids:
        p = os.path.join(PROV, pid + ".json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        for s in d.get("software", []) or []:
            if isinstance(s, dict):
                c = canon(s.get("name", ""))
                if c:
                    sw[c].add(pid)
                    f = str(s.get("functions", ""))[:60]
                    if f:
                        swfun[c].add(f)
        for t in d.get("agent_tools", []) or []:
            if isinstance(t, dict):
                try:
                    n = int(t.get("count", 0) or 0)
                except Exception:
                    n = 0
                tools[t.get("tool")] += n
                tools_by_cat[cat[pid]][t.get("tool")] += n
        rk = d.get("recalled_knowledge", [])
        if isinstance(rk, list):
            rk_per_paper[pid] = len(rk)
            for r in rk:
                if isinstance(r, dict):
                    rel = str(r.get("reliability", "")).lower()
                    key = "certain" if rel.startswith("certain") else "likely" if rel.startswith("likely") else "uncertain" if rel.startswith("uncertain") else "mixed/other"
                    rk_rel[key] += 1
        for li in d.get("local_inputs", []) or []:
            if isinstance(li, dict):
                f = str(li.get("file", ""))
                key = ("LZ tex" if "tex" in f.lower() else "LZ figure PNG/PDF" if ("fig" in f.lower() or "png" in f.lower() or "pdf" in f.lower())
                       else "corpus paper" if "papers/" in f else "corpus work file" if "work/" in f else "corpus code" if "code/" in f
                       else "dossier/plan/guide/ledger" if any(k in f.lower() for k in ["dossier", "plan", "guide", "ledger"]) else "other")
                inputs[key] += 1
        dr = d.get("data_requests", "none")
        if dr not in ("none", "", None) and not (isinstance(dr, str) and dr.lower().startswith("none")):
            drs.append((pid, dr))
        ds = d.get("datasets", "none")
        if ds not in ("none", "", None, []) and not (isinstance(ds, str) and ds.lower().startswith("none")):
            datasets.append((pid, ds))
        wd = d.get("wimpydd_generated_files", "none")
        if wd not in ("none", "", None, []) and not (isinstance(wd, str) and wd.lower().startswith("none")):
            wdfiles.append((pid, wd))
        fa = d.get("failed_or_abandoned", [])
        failed += len(fa) if isinstance(fa, list) else 0

    print(f"Papers with provenance JSON: {len(rk_per_paper)} of {len(ids)}\n")
    print("### Software and packages (consolidated)\n")
    print("| Package / version | Papers (count) | Categories | Typical functions |")
    print("|---|---|---|---|")
    for c, ps in sorted(sw.items(), key=lambda x: -len(x[1])):
        cats = collections.Counter(cat[p] for p in ps)
        catstr = ", ".join(f"{k} {v}" for k, v in sorted(cats.items(), key=lambda x: -x[1]))
        fun = "; ".join(sorted(swfun[c])[:3])
        print(f"| {c} | {len(ps)} | {catstr} | {fun} |")
    print("\n### Agent tool calls (totals from provenance JSONs)\n")
    print("| Tool | Calls | Per paper |")
    print("|---|---|---|")
    n = max(1, len(rk_per_paper))
    for t, v in tools.most_common():
        print(f"| {t} | {v} | {v / n:.1f} |")
    print(f"| **all** | {sum(tools.values())} | {sum(tools.values()) / n:.1f} |")
    print("\n### Tool calls by taxonomy category\n")
    print("| Category | Papers | Read | Bash | Write | Edit | other |")
    print("|---|---|---|---|---|---|---|")
    npc = collections.Counter(cat[p] for p in rk_per_paper)
    for c in sorted(tools_by_cat, key=lambda k: -npc[k]):
        tc = tools_by_cat[c]
        other = sum(v for k, v in tc.items() if k not in ("Read", "Bash", "Write", "Edit"))
        print(f"| {c} | {npc[c]} | {tc['Read']} | {tc['Bash']} | {tc['Write']} | {tc['Edit']} | {other} |")
    print("\n### Local inputs by type (entries in provenance JSONs)\n")
    print("| Input type | Entries |")
    print("|---|---|")
    for k, v in inputs.most_common():
        print(f"| {k} | {v} |")
    print("\n### Recalled knowledge\n")
    tot = sum(rk_per_paper.values())
    print(f"Total items: {tot} in {len(rk_per_paper)} papers (mean {tot / n:.1f}; min {min(rk_per_paper.values())}, max {max(rk_per_paper.values())}).")
    print("| Reliability | Items | Share |")
    print("|---|---|---|")
    t2 = sum(rk_rel.values())
    for k, v in rk_rel.most_common():
        print(f"| {k} | {v} | {100 * v / max(1, t2):.0f} % |")
    top = sorted(rk_per_paper.items(), key=lambda x: -x[1])[:10]
    print("\nMost recall-dependent papers: " + ", ".join(f"{p} ({n})" for p, n in top))
    print(f"\nFailed or abandoned approaches recorded: {failed}.")
    print("\n### Datasets and data requests\n")
    print("Datasets used: " + ("none" if not datasets else "; ".join(f"{p}: {d}" for p, d in datasets)))
    print("Data requests: " + ("none" if not drs else "; ".join(f"{p}: {str(d)[:80]}" for p, d in drs)))
    print("\n### WimPyDD-generated files declared\n")
    for p, w in wdfiles:
        print(f"- {p}: {str(w)[:160]}")


if __name__ == "__main__":
    main()
