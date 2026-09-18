"""
ledger_tools.py -- coordinator utilities for the corpus.

Usage (from the simulation root):
  .venv/bin/python output/code/common/ledger_tools.py check P001 [P002 ...]
      -> verifies papers/P0XX.md exists, body word count (Abstract..Conclusion) <= 550 (warn > 560),
         required header/footer fields present, work/P0XX/details.md and provenance/P0XX.json exist
         with the required keys, scripts referenced exist.
  .venv/bin/python output/code/common/ledger_tools.py add path/to/row.json [more.json ...]
      -> appends rows to output/results_ledger.csv and output/results_ledger.json (replacing an existing
         row with the same id+version).
  .venv/bin/python output/code/common/ledger_tools.py summary
      -> prints id, date, category, stance, headline for all rows (compact corpus overview).
"""
import sys, os, json, re, csv

ROOT = os.getcwd()
LEDGER_CSV = os.path.join(ROOT, "output/results_ledger.csv")
LEDGER_JSON = os.path.join(ROOT, "output/results_ledger.json")
FIELDS = ["id", "version", "date", "arxiv_category", "taxonomy_category", "title", "question", "headline_result",
          "key_numbers", "model_or_mechanism", "stance", "result_type", "confidence", "builds_on", "agent_tools",
          "software_and_packages", "scripts", "local_inputs", "recalled_knowledge", "datasets_used", "data_requests",
          "data_dependence"]
PROV_KEYS = ["id", "title", "date", "agent_tools", "software", "scripts_and_commands", "local_inputs",
             "recalled_knowledge", "datasets", "data_requests", "failed_or_abandoned", "wimpydd_generated_files"]
FOOTER_FIELDS = ["HEADLINE RESULT", "KEY NUMBERS", "RESULT TYPE", "STANCE ON THE EVENT", "CONFIDENCE",
                 "DATA DEPENDENCE", "TOOLS (summary)", "DETAILS"]


def body_word_count(text):
    m = re.search(r"\*\*Abstract\.\*\*(.*?)\*\*References\.\*\*", text, re.S)
    if not m:
        return None
    body = m.group(1)
    body = re.sub(r"\*\*(Question and approach|Results|Caveats|Conclusion)\.\*\*", " ", body)
    return len(body.split())


def check(pid):
    problems, notes = [], []
    p = os.path.join(ROOT, f"output/papers/{pid}.md")
    if not os.path.exists(p):
        return [f"{pid}: paper missing"], []
    text = open(p).read()
    wc = body_word_count(text)
    if wc is None:
        problems.append("could not locate Abstract..References body")
    else:
        notes.append(f"body words={wc}")
        if wc > 560:
            problems.append(f"body word count {wc} > 550")
    total = len(text.split())
    notes.append(f"total words={total}")
    if total > 1000:
        problems.append(f"whole page {total} words: may not fit one page")
    for f in FOOTER_FIELDS:
        if f"**{f}" not in text:
            problems.append(f"missing footer field {f}")
    for h in ["Simulated arXiv date", "Primary arXiv category", "Author profile", "Category (from your taxonomy)", "Builds on"]:
        if h not in text:
            problems.append(f"missing header {h}")
    if not os.path.exists(os.path.join(ROOT, f"output/work/{pid}/details.md")):
        problems.append("details.md missing")
    pj = os.path.join(ROOT, f"output/provenance/{pid}.json")
    if not os.path.exists(pj):
        problems.append("provenance json missing")
    else:
        try:
            d = json.load(open(pj))
            for k in PROV_KEYS:
                if k not in d:
                    problems.append(f"provenance key missing: {k}")
        except Exception as e:
            problems.append(f"provenance json unreadable: {e}")
    scripts = [s for s in os.listdir(os.path.join(ROOT, "output/code")) if s.startswith(pid + "_")]
    notes.append(f"scripts={scripts}")
    if not scripts and "none: derived by hand" not in text.lower() and "derived by hand" not in text.lower():
        notes.append("no script found (check RESULT TYPE)")
    return problems, notes


def load_json_ledger():
    if os.path.exists(LEDGER_JSON):
        try:
            return json.load(open(LEDGER_JSON))
        except Exception:
            return []
    return []


def add(rows):
    ledger = load_json_ledger()
    for row in rows:
        row = {k: ("" if row.get(k) is None else row.get(k)) for k in FIELDS} | {k: v for k, v in row.items() if k not in FIELDS}
        row = {k: row[k] for k in FIELDS}
        for k, v in row.items():
            if isinstance(v, (list, dict)):
                row[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else "; ".join(str(x) for x in v)
        row.setdefault("version", "v1")
        ledger = [r for r in ledger if not (r.get("id") == row["id"] and r.get("version") == row["version"])]
        ledger.append(row)
    ledger.sort(key=lambda r: (r["id"], r["version"]))
    json.dump(ledger, open(LEDGER_JSON, "w"), indent=1, ensure_ascii=False)
    with open(LEDGER_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in ledger:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    return len(ledger)


def summary():
    for r in load_json_ledger():
        print(f"{r['id']} {r['version']} {r['date']} [{r['taxonomy_category']}] ({r['stance']}) {r['title']}\n    -> {r['headline_result']}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "check":
        bad = 0
        for pid in sys.argv[2:]:
            probs, notes = check(pid)
            print(pid, "OK" if not probs else "PROBLEMS", "|", "; ".join(notes))
            for pr in probs:
                print("   -", pr)
                bad += 1
        sys.exit(1 if bad else 0)
    elif cmd == "add":
        rows = []
        for path in sys.argv[2:]:
            d = json.load(open(path))
            rows.extend(d if isinstance(d, list) else [d])
        n = add(rows)
        print(f"ledger now has {n} rows")
    elif cmd == "summary":
        summary()
