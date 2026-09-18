# QA trim, batch 1–2 (P001–P020): one-page fit

Rule applied: whole file ≤ 950 words (`wc -w`); body (**Abstract.** → **References.**) untouched; only footer fields after `---` condensed (HEADLINE ≤ 50 words, KEY NUMBERS ≤ 6 bullets / ≤ 110 words, CONFIDENCE ≤ 25, TOOLS ≤ 60 keeping every package name+version); DETAILS / RESULT TYPE / STANCE / DATA DEPENDENCE lines unchanged; no number altered or introduced. References lines were not edited in any paper (not needed for length).

| Paper | Words before | Words after | Body words | Fields edited |
|---|---|---|---|---|
| P001 | 1043 | 868 | 520 | HEADLINE, KEY NUMBERS, CONFIDENCE, TOOLS |
| P002 | 1033 | 878 | 541 | HEADLINE, KEY NUMBERS, CONFIDENCE, TOOLS |
| P003 | 961 | 866 | 522 | HEADLINE, KEY NUMBERS, CONFIDENCE, TOOLS |
| P004 | 884 | 884 | 512 | none (already ≤ 950) |
| P005 | 1028 | 904 | 515 | KEY NUMBERS, CONFIDENCE, TOOLS |
| P006 | 1036 | 913 | 549 | HEADLINE, KEY NUMBERS, CONFIDENCE, TOOLS |
| P007 | 999 | 935 | 538 | HEADLINE, CONFIDENCE, TOOLS |
| P008 | 859 | 859 | 524 | none (already ≤ 950) |
| P009 | 997 | 863 | 519 | HEADLINE, KEY NUMBERS, CONFIDENCE, TOOLS |
| P010 | 965 | 878 | 514 | HEADLINE, KEY NUMBERS, CONFIDENCE, TOOLS |
| P011 | 1165 | 942 | 546 | HEADLINE, KEY NUMBERS, CONFIDENCE, TOOLS |
| P012 | 1173 | 947 | 549 | HEADLINE, KEY NUMBERS, CONFIDENCE, TOOLS |
| P013 | 1073 | 933 | 533 | HEADLINE, KEY NUMBERS, CONFIDENCE, TOOLS |
| P014 | 1126 | 922 | 546 | HEADLINE, KEY NUMBERS, CONFIDENCE, TOOLS |
| P015 | 1153 | 938 | 547 | HEADLINE, KEY NUMBERS, CONFIDENCE, TOOLS |
| P016 | 1151 | 929 | 542 | HEADLINE, KEY NUMBERS, CONFIDENCE, TOOLS |
| P017 | 1133 | 906 | 512 | HEADLINE, KEY NUMBERS, CONFIDENCE, TOOLS |
| P018 | 1054 | 903 | 543 | HEADLINE, KEY NUMBERS, CONFIDENCE, TOOLS |
| P019 | 1148 | 948 | 549 | HEADLINE, KEY NUMBERS, CONFIDENCE, TOOLS |
| P020 | 1159 | 901 | 516 | HEADLINE, KEY NUMBERS, CONFIDENCE, TOOLS |

Body word counts were verified identical before and after editing for all 20 papers. No body exceeds 550 words (max 549: P006, P012, P019).

Notes
- P004 and P008 were already ≤ 950 and were left untouched; their footers therefore still exceed some per-field limits (P004: KEY NUMBERS 113 words, CONFIDENCE 32 words; P008: CONFIDENCE 27 words). Flagged, not edited, per the "only if > 950" rule.
- P017 References line lists Menéndez–Gazit–Schwenk and Klos et al. as one entry separated by ";" — 7 external references if counted separately. Not edited (References edits were not needed for length; Klos is not cited in the body).
- All final per-field counts for edited papers: HEADLINE 45–50 words; KEY NUMBERS 104–110 words, 5–6 bullets (P007 keeps its original single-paragraph KEY NUMBERS, 108 words, unedited); CONFIDENCE 16–25 words; TOOLS 16–34 words with every package name and version retained.
- Numbers dropped from footers were redundant variants or secondary figures already present in the body; no number was altered and none was introduced.

## Tool calls

Bash (run from /Users/reza/LZ_simulation)
1. Bash — `wc -w` whole-file and body counts for P001–P020 (initial).
2. Bash — print References + footer of P001–P003 (awk).
3. Bash — References line numbers for P001–P020 (grep/wc).
4. Bash — `.venv/bin/python` inline checker: whole/body/per-field counts for P001, P002, P003, P005, P006, P007.
5. Bash — `.venv/bin/python` inline checker for P001–P020 + `wc -w`.
6. Bash — `.venv/bin/python` inline checker for P001–P020 + `wc -w` + grep Klos in P017.
7. Bash — final `wc -w` confirmation P001–P020.

Read (footer region, from the References line)
- output/papers/P001.md, P002.md, P003.md, P005.md, P006.md, P007.md, P009.md, P010.md, P011.md, P012.md, P013.md, P014.md, P015.md, P016.md, P017.md, P018.md, P019.md, P020.md

Edit (exact string replacement, footer only)
- output/papers/P001.md ×2
- output/papers/P002.md ×3
- output/papers/P003.md ×3
- output/papers/P005.md ×3
- output/papers/P006.md ×3
- output/papers/P007.md ×2
- output/papers/P009.md ×3
- output/papers/P010.md ×2
- output/papers/P011.md ×3
- output/papers/P012.md ×3
- output/papers/P013.md ×2
- output/papers/P014.md ×2
- output/papers/P015.md ×2
- output/papers/P016.md ×3
- output/papers/P017.md ×2
- output/papers/P018.md ×3
- output/papers/P019.md ×2
- output/papers/P020.md ×5

Write
- output/provenance/QA_trim_batch1-2.md (this report)
