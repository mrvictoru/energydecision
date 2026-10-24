# Agent task: audit repository docs and `index.html`

You are auditing the `energydecision` repository at
`/mnt/rescue-hdd/energydecision`. Work read-only by default. Produce a findings
report; apply only mechanical, unambiguous fixes and list every edit.

## Context (read this first)

This repo is a research codebase with two tracks:

- **AEMO / grid-scale** (`AEMOBatteryTradingEnv`): energy + 8 FCAS markets,
  degradation-aware. The shipped policy is a standalone Decision Transformer
  distilled from an honest (non-clairvoyant) SDP planning teacher with a
  state-dependent `J_t(soc)` return-to-go prompt.
- **Household** (`SolarBatteryEnv`): residential solar-battery.

A preprint manuscript lives in `paper/manuscript/` (`main.tex`,
`references.bib`, `figures/`). The paper's framing is:
*we set out to test whether an offline DT can control a multi-market battery;
policy-generated data stalls at a behaviour-cloning ceiling; distilling an
honest SDP teacher breaks it.*

**Authoritative sources — do not treat prose docs as truth:**

1. `paper/manuscript/main.tex` — the manuscript (do **not** edit; the author is
   reading it).
2. `paper/audit/provenance.tsv` — every headline number → artifact + command.
3. `paper/audit/artifacts/*.json` — significance, impact gate, retention.
4. `paper/audit/artifacts/spotcheck_standard.txt` — reproducibility record.
5. `AGENTS.md` — project conventions and current status.
6. `docs/website_requirements.md` — the spec for `index.html`.
7. `report.md` — long-form technical record (source for many numbers).

## Known recent corrections — verify these propagated

These were changed recently. Any doc or page still showing the old value is a
finding:

- Impact-gate Hornsdale ratio: **4.70×**, not 4.71× (artifact `4.7047`).
- `abdulla2016optimal`: issue year **2018**, not 2016 (vol. 9, no. 3, pp.
  2086–2096, May 2018).
- `subramanya2022review`: pages **54484–54506**, not 54456–54476.
- The paper does **not** implement DAgger; DAgger is related work only.
- Headline surfaces are the **extended** ones: n = 30 / 12 / 30 / 6
  (standard 2024, dispatch-matched, expanded broad-2024, 2025 OOD), not the
  earlier small-n single-month runs.
- The behaviour-cloning ceiling is scoped to the **tested** data sources and
  levers; avoid unqualified "no amount" / "never" claims.
- `return_scale = 23,284.83` is correct for inference but does **not** reproduce
  exactly as `p90(|J_t|)/10` of the released corpus (recomputed ≈21,067).
- Household track: only the **unconstrained 7-day real-OOD** DT is net-positive;
  constrained/long-horizon surfaces are net-negative (see `AGENTS.md` and
  `docs/household/workflow.md` §H4.9). Do not describe household as a uniform win.

## Scope A — documentation freshness and readability

Audit at least:

- `README.md`, `AGENTS.md`, `report.md`
- `docs/README.md`, `docs/DOC_README.md`, `docs/architecture.md`,
  `docs/development.md`, `docs/evaluation_guide.md`, `docs/known_issues.md`,
  `docs/FUTURE_PLAN.md`, `docs/aemo_dt_preferred_policy_plan.md`,
  `docs/physics_v2_rebaseline_commands.md`, `docs/website_requirements.md`,
  `docs/huggingface/*.md`
- `paper/README.md`, `paper/manuscript/README.md`,
  `paper/manuscript/WRITING_PLAN.md`, `paper/arxiv_submission_checklist.md`,
  `paper/anticipated_reviewer_questions.md`, `paper/outreach_emails.md`

Check:

1. **Stale numbers/claims.** Any headline number that contradicts
   `provenance.tsv` / artifacts (ratios, dollar values, n, CIs, p-values,
   retention, architecture, corpus size).
2. **Stale status.** Sections describing old results as current (v1 checkpoints,
   small-n surfaces, "PPO is the broad-year leader", pre-physics-v2 numbers,
   `aemo_dt_sdp_jtsoc_fullcorpus.pt` as shipped).
3. **Broken references.** `report.md §X.Y` targets that no longer exist;
   markdown links to files that do not exist; commands referencing missing
   paths; hard-coded `/home/...` or `/mnt/...` paths in code blocks.
4. **Duplication/contradiction.** The same fact stated differently in two docs.
5. **Readability.** Heading hierarchy, overly long paragraphs, undefined
   acronyms on first use, inconsistent terminology (student/teacher/PPO,
   "control loop", "auto" mode), mixed British/American spelling (the project
   uses British/Australian English).
6. **Freshness markers.** "Today", "currently", "pending" statements that are
   now false; dates older than the shipped result.

Suggested commands:

```bash
grep -rn "4\.71\|54456\|2016" --include='*.md' .
grep -rn "/home/\|/mnt/" --include='*.md' .
grep -rn "TODO\|TBD\|FIXME" --include='*.md' .
python3 - <<'PY'   # markdown link check (relative links only)
import re, pathlib
for p in pathlib.Path('.').rglob('*.md'):
    if '.git' in p.parts: continue
    for m in re.finditer(r'\]\(([^)#]+)(?:#[^)]*)?\)', p.read_text(errors='ignore')):
        t=m.group(1)
        if t.startswith(('http','mailto:')): continue
        q=(p.parent/t).resolve()
        if not q.exists(): print(f"{p}:{t} -> MISSING")
PY
```

## Scope B — `index.html`

`index.html` is a single-file interactive project page (~2,700 lines) with two
tracks (AEMO and household), charts, and embedded JS data arrays. Audit it
against `docs/website_requirements.md` (especially §3 sections, §4 design,
§5 technical quality, §6 data-integrity rules, §8 acceptance criteria).

Check:

1. **HTML validity.** Unclosed tags, duplicate `id`s, invalid nesting, `<a href="#...">`
   targets with no matching `id` (already clean once, re-check), `<img>` without
   `alt`, buttons without labels.
2. **Navigation.** Every nav link resolves to a section; the AEMO/household tab
   switch shows/hides the right sections; the skip link works.
3. **Data integrity.** Numbers embedded in HTML and in the JS data arrays must
   match the artifacts: 16,501/4,303; 33,791/18,818; 29,124/4,350;
   30,791/4,817; ratios 3.84/1.80/6.70/6.39; impact 9/9, Δ $167,099,
   CI [101,213, 227,362], p=0.0039; per-battery 4.36/4.70/4.67. Flag any
   mismatch, including rounded variants that imply a different number.
4. **Honesty.** Limitations section present and accurate; negative results
   (forecast DT, GRPO, failed fixes) presented as prominently as wins; the
   household caveat stated; "no solver in the control loop" not overstated.
5. **Stale framing.** Does the page still frame the ceiling as the original
   goal, or as an intermediate finding? Does it claim a uniform household win?
6. **Links.** GitHub, Hugging Face, paper/arXiv links resolve; no `href="#"`.
7. **Performance/robustness.** Charts have fallbacks (`data-chart-fallback`);
   no console-breaking JS; works from `file://`.
8. **Accessibility/readability.** Contrast, font sizes, aria labels, keyboard
   focus, mobile layout at ~375 px.

Suggested commands:

```bash
python3 - <<'PY'   # duplicate ids + broken anchors
import re
s=open('index.html').read()
ids=re.findall(r'\bid="([^"]+)"',s); hrefs=re.findall(r'href="#([^"]+)"',s)
print("duplicate ids:", sorted({i for i in ids if ids.count(i)>1}))
print("missing anchors:", sorted(set(hrefs)-set(ids)-{''}))
PY
grep -n "4\.71\|4\.7×\|2016\|54456\|TODO\|coming soon" index.html
```

## Output format

Write `paper/audit/doc_audit_2026-10-24.md` with:

```
## Summary
<3–6 lines: overall state, blockers, what is safe>

## Blockers (must fix before submission)
- [file:line] issue — evidence — proposed fix

## High / Medium / Low
...

## Docs that are up to date
<list, so the author knows what not to re-check>

## Edits applied
<exact list, or "none">

## Open questions for the author
<only genuine ambiguities>
```

Rules:

- **Never invent a number.** If a claim cannot be traced to an artifact, mark it
  `UNVERIFIED` and quote the source.
- **Do not edit** `paper/manuscript/main.tex`, `paper/manuscript/references.bib`,
  or any artifact JSON.
- Mechanical fixes allowed in docs/`index.html`: typos, broken anchors, stale
  links, duplicated IDs, obvious date/status corrections backed by an artifact.
  Anything touching a numeric claim needs an explicit evidence pointer.
- Keep the author's voice and British/Australian spelling.
- Do not run GPU jobs or long evaluations.
