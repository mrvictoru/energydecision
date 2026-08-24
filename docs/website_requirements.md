# Website Requirements — energydecision Interactive Project Page

> **Audience of this document:** an AI agent (or developer) tasked with building
> the website. Everything needed to build it is in this file — you do **not**
> need access to the repository's code or results to complete the task.
>
> **Status of this document:** requirements only. No website exists yet.

---

## 1. Goal

Create a **single-file interactive website** that presents the research project
*energydecision* — a benchmark and research codebase showing that an offline
**Decision Transformer** can outperform online RL (PPO) and real-world dispatch
for utility-scale battery trading in Australia's National Electricity Market
(AEMO/NEM), including full 9-dimensional FCAS co-optimized bidding, under
degradation-aware evaluation.

The site must let a visitor:

1. **Understand** what the project is and why it matters (5-minute read).
2. **See** the headline result through charts, not just tables: the DT beats PPO
   on all 4 canonical evaluation surfaces plus a market-impact gate.
3. **Explore** the evidence interactively: sortable/filterable comparison
   tables, per-surface charts, revenue decomposition, the model-improvement
   timeline, RTG-prompt behaviour, and statistical significance.
4. **Go deeper**: link out to the technical report (`report.md` on GitHub),
   docs, Hugging Face models/datasets, and reproduction instructions.

## 2. Deliverable & Hosting

| Item | Requirement |
|---|---|
| File | Exactly **one** file: `index.html` (all CSS + JS inline or via CDN `<script>` tags) |
| Location | Repo root or `docs/` — must work when served by **GitHub Pages** from the `main` branch |
| Build step | **None.** No bundler, no package install, no transpilation. Opening `index.html` directly in a browser (`file://`) must also work |
| Network | Must degrade gracefully offline: if CDN libraries fail to load, text content and tables must still render (charts may be absent). Prefer vendoring-free CDN usage with `defer` |
| Frameworks | No React/Vue/Next. Use **vanilla JS + a small chart library** (recommended: [Chart.js](https://www.chartjs.org/) via CDN; acceptable alternatives: ECharts, uPlot). No jQuery |
| Fonts/icons | System font stack preferred (zero network cost). Icons: inline SVG only |

Recommended Chart.js CDN pattern:

```html
<script src="https://cdn.jsdelivr.net/npm/chart.js@4" defer></script>
```

Guard every chart init behind `typeof Chart !== 'undefined'` so the page
degrades gracefully without network.

## 3. Content Specification

All numbers below are **verified results from the project's technical report**
(`report.md`, §8.2.10, verified 2026-08-23). Embed them as JS objects; do not
fetch them at runtime.

### 3.1 Hero section

- **Title:** "Offline Decision Transformers Outperform Online RL for Utility-Scale Battery Dispatch"
- **Subtitle:** A degradation-aware AEMO/NEM benchmark: planner-distilled Decision Transformer vs online RL vs real-world dispatch replay.
- **Key stat callouts** (large numbers, animated count-up on scroll):
  - **4.9×** profit vs PPO on standard surface ($11,573 vs $2,353 /ep)
  - **4/4** identity surfaces won (+ impact gate passed)
  - **0** solver at inference (fully deployable standalone transformer)
  - **95% CI excludes zero** on all six DT-vs-PPO paired comparisons
- **Primary CTA buttons:** "Read the report" → `https://github.com/mrvictoru/energydecision/blob/main/report.md`; "Get the model" → `https://huggingface.co/mrvictoru/energydecision-dt-v2-sdp`

### 3.2 What is this? (plain-language intro)

Explain in ~150 words, for a technically literate but non-specialist reader:
- Grid-scale batteries in Australia's NEM earn money two ways: energy arbitrage (buy low / sell high on 5-minute spot prices) and FCAS ancillary services (8 markets).
- The control problem: decide dispatch + FCAS bids every 5 minutes, net of battery degradation.
- Prior belief: online RL (PPO) was needed to capture FCAS spikes; offline behaviour cloning couldn't exceed its training data.
- This project's result: distilling an honest stochastic-dynamic-programming (SDP) planning teacher into a standalone Decision Transformer breaks that ceiling — no clairvoyance, no solver at inference, better than PPO everywhere tested.

### 3.3 Environment explainer (interactive diagram)

Describe the simulation setup (`AEMOBatteryTradingEnv`):

- **Observation (18-D):** time features (5) · energy price/demand (2) · FCAS prices (8) · generation mix (2) · SOC (1)
- **Action (9-D):** energy dispatch ∈ [-1,1] + 8 FCAS bid fractions ∈ [0,1] (`full_fcas` mode)
- **Reward:** energy arbitrage + FCAS revenue − degradation cost − SOC penalties (profit already net of degradation)
- **Resolution:** 5-minute steps, historical AEMO market data via NEMOSIS
- **Degradation:** real-world calendar+cycle aging (Kampker et al. 2025), NMC/LFP presets, Arrhenius temperature dependence

Interactive element idea: a simple SVG/CSS diagram where hovering over each
component (observation groups, action dims, reward terms) highlights it and
shows a tooltip with detail. Keep it dependency-free.

### 3.4 Headline results — THE centerpiece (interactive)

Four identity surfaces, profit per episode (USD, net of degradation),
5-min protocol. Data:

```js
const HEADLINE = [
  { surface: "Standard Oct",        dt: 11573, ppo: 2353,  n: 5,  winRate: "5/5",   wilcoxonP: 0.0625 },
  { surface: "Dispatch-matched",    dt: 35320, ppo: 22530, n: 6,  winRate: "6/6",   wilcoxonP: 0.0312 },
  { surface: "Expanded broad-2024", dt: 34761, ppo: 19504, n: 27, winRate: "25/27", wilcoxonP: 0.0002 },
  { surface: "2025 OOD",            dt: 25862, ppo: 6498,  n: 6,  winRate: "6/6",   wilcoxonP: 0.0312 },
];
// Impact gate (piecewise merit-order impact): DT passes on all 3 grid-scale batteries
const IMPACT_GATE = [
  { battery: "Small (~8 MWh)",          dt: 34600,  ppo: 11000 },
  { battery: "Hornsdale-class (194)",   dt: 142100, ppo: 56500 },
  { battery: "Torrens-class (250 MWh)", dt: 173100, ppo: 69500 },
];
```

Required interactivity:

1. **Grouped bar chart** (DT vs PPO per surface) with hover tooltips showing exact values and ratio.
2. **Surface tabs or toggle** letting users switch between "Profit" view and "DT/PPO ratio" view.
3. A toggle: **"Show confidence intervals"** — overlay error bars using these bootstrap CIs (DT − PPO difference):
   - Standard Oct: +$9,220 [+$7,073, +$11,624]
   - Dispatch-matched: +$12,791 [+$7,008, +$18,277]
   - Expanded broad-2024: +$15,257 [+$4,108, +$33,350]
   - 2025 OOD: +$19,364 [+$7,138, +$39,670]
   - Impact gate: +$63,064 [+$33,356, +$100,973]
4. A footnote explaining: *all Wilcoxon p-values marked \* are at the bounded minimum attainable for n<10 (every paired difference had the same sign); all six CIs exclude zero.*

### 3.5 Revenue decomposition chart

The DT's edge is FCAS, not energy. Stacked/grouped bar per surface:

```js
const DECOMP = {
  // stage-C DT under j_t_soc inference (identity surfaces)
  dt:    { expandedEnergy: 14800 },           // narrowed gap vs PPO's $17.4k energy
  note:  "FCAS capture is 3–6.8× PPO everywhere; energy arbitrage gap narrowed to $14.8k vs $17.4k",
};
```

If precise per-surface decomposition values are unavailable, present the
qualitative finding with the one hard number above plus: distilled-model
degradation $88–176/MWh vs PPO ~$211–310/MWh. Do NOT invent numbers — mark any
illustrative value explicitly as illustrative.

### 3.6 Model improvement timeline ("How we got here")

Interactive horizontal stepper/timeline (click each stage to expand details).
Data:

```js
const TIMELINE = [
  { stage: 1, name: "Pilot DT",                 arch: "4×128, ctx=1152",        data: "6 proxy episodes",            dmProfit: -10620, fcas: 2328,   deg: 12975, change: "Baseline" },
  { stage: 2, name: "Autoresearch tuning",      arch: "8×512, ctx=180",         data: "24 episodes (mixed)",         dmProfit: -1396,  fcas: 77,     deg: 2503,  change: "Hyperparameter tuning" },
  { stage: 3, name: "FCAS-rich offline DT",     arch: "8×384, ctx=180",         data: "2,425 episodes (PPO-rich)",   dmProfit: 1522,   fcas: 1383,   deg: 212,   change: "Dataset quality" },
  { stage: 4, name: "GRPO fine-tune (legacy)",  arch: "8×384 + GRPO",           data: "v2 HF + 5 GRPO iter",         dmProfit: 8242,   fcas: 7686,   deg: 760,   change: "Online RL (later shown overfit)" },
  { stage: 5, name: "Modern v2 pretrained",     arch: "8×768 GQA",              data: "2,401 eps (realistic bats)",  dmProfit: 10138,  fcas: 10068,  deg: 187,   change: "Architecture improvement" },
  { stage: 6, name: "Hierarchical DT+LP",       arch: "waypoint-DT + Oracle-LP",data: "Oracle SOC paths (1,200 eps)",dmProfit: 291841, note: "*perfect foresight — not deployable*", change: "Decomposition: DT plans SOC, LP executes" },
  { stage: 7, name: "Honest SDP executor",      arch: "waypoint-DT + SDP",      data: "seasonal forecast only",      dmProfit: 59091,  note: "*solver at inference*", change: "Foresight caveat lifted" },
  { stage: 8, name: "Standalone J_t(soc) DT",   arch: "8×768 mixed-head",       data: "SDP teacher trajs (640 eps)", dmProfit: 35320,  fcas: 105000, deg: 145,   change: "Planner distillation + state-dependent prompts — SHIPPED" },
];
```

Requirements:
- Bar-chart visualization of `dmProfit` across stages (log-friendly handling of the negative stages).
- Stages 6–7 visually flagged as "not directly comparable" (solver-in-the-loop); footnote included.
- Stage 8 highlighted as shipped.
- Expandable detail panel per stage describing the key change (use the `change` text, expand where marked).

### 3.7 The ceiling story (why this was hard) — narrative + failed-attempts table

Short narrative (~200 words): behaviour cloning cannot output skills absent from
the data; FCAS spike bidding survived every within-paradigm fix. Then an
interactive table of failed attempts:

```js
const FAILED_FIXES = [
  { fix: "RTG prompt sweep (0–50)",              result: "Flat — not a prompting artifact" },
  { fix: "PPO-only training data (2 archs)",     result: "$17.6–17.8k broad-2024 via energy arbitrage, but collapses on 2025 OOD ($4.2k vs PPO $14.3k)" },
  { fix: "FCAS-heavy data subset",               result: "FCAS +23% ($4.8k→$5.9k) but still 1.7× below PPO" },
  { fix: "FCAS-weighted action loss",            result: "No effect — no higher-FCAS behaviour in data to amplify" },
  { fix: "Mixed action head (Sigmoid FCAS)",     result: "Ambiguous: helps OOD (+22%), hurts in-distribution" },
  { fix: "GRPO / value-critic fine-tuning",       result: "Flat-to-negative; legacy GRPO champion was a narrow overfit ($8,242 DM → $1,533 standard)" },
  { fix: "Forecast tokens (TTM, 48-step)",       result: "Negative result: $4,564 vs $4,991 baseline — TTM FCAS forecasts carry ~zero signal (corr 0.01–0.07)" },
];
```

Then the resolution: "The fix wasn't prompting, re-weighting, or fine-tuning —
it was changing the *teacher*: distill an honest SDP planner."

### 3.8 Method: teacher distillation pipeline (diagram)

Visual flow (SVG or styled divs with arrows):

```
Seasonal price forecast (pre-2024 only, no future info)
        ↓
Honest SDP teacher ── degradation-aware ($50/$20 per MWh) ── plans energy/SOC
        ↓                                    ↓
greedy FCAS bids from residual headroom   J_t(soc) cost-to-go table
        ↓                                    ↓
Teacher trajectories (640 eps, 6.2M rows) ──→ Standalone DT (8×768 GQA)
                                                     ↓
                              Inference: rtg_mode="auto"
                              ├─ identity surfaces → j_t_soc state-dependent prompt
                              └─ impact surfaces  → conservative constant RTG
```

Include the three corpora table (§5.4 of report): conservative 320 eps /
aggressive 320 eps / combined 640 eps with J_t(soc) RTG column.

### 3.9 Why `rtg_mode="auto"` (failure-mode explainer)

Short section with a mini before/after chart: explicit j_t_soc inference fails
catastrophically at grid scale under merit-order impact (hornsdale −$142.7k,
torrens −$347.8k mean profit) because the price-taking prompt drives
over-dispatch → self-suppression. Auto mode falls back to constant RTG under
impact and keeps +$62.9k/+$69.7k. Message: *"prompts should be state-dependent,
but gated by market power."*

### 3.10 Statistical rigor section

Table of the six comparisons (already given in §3.4 data) rendered cleanly,
plus notes: bootstrap 10,000 resamples, paired Wilcoxon, per-cell P(DT>PPO)
≥ 0.9998 on every identity surface.

### 3.11 Artifacts & reproducibility

Link cards:
- Code: https://github.com/mrvictoru/energydecision
- Report: https://github.com/mrvictoru/energydecision/blob/main/report.md
- Model (shipped Stage C DT): https://huggingface.co/mrvictoru/energydecision-dt-v2-sdp
- Training corpus: https://huggingface.co/datasets/mrvictoru/AEMO_simulated_trade_sdp
- Historical models: `energydecision-dt-v2`, `energydecision-dt-v2-impact`, `energydecision-dt` (link via the GitHub org page https://huggingface.co/mrvictoru)

Repro quick-start block (copyable code snippet):
```
pip install -r requirements.txt && pip install -r torch_req.txt
python3 scripts/launch_aemo_training.py --run-tier proxy-baseline
python3 scripts/autoresearch_evaluator.py --surface-manifest-path <manifest> \
  --evaluation-config configs/aemo_autoresearch_evaluator.example.json \
  --output-dir eval_output/my_eval
python -m pytest tests/ -v
```

### 3.12 Limitations (honesty section — required)

Must be present, not buried:
- All results are simulator-based (historical AEMO prices, modeled FCAS co-optimization and degradation); sim-to-real transfer is open.
- Explicit j_t_soc is identity-only; shipped default is `auto`.
- Expanded-surface energy-arbitrage gap vs PPO is narrowed ($14.8k vs $17.4k), not eliminated.
- Some headline surfaces have few scenarios (n=5–6); point estimates should be read alongside their intervals.
- Broad-surface comparisons were run with a handicapped 3-dim-effective action space (protocol asymmetry disclosed in report §6).

### 3.13 Footer

License note (match repo license), "Built with vanilla JS + Chart.js",
last-updated date (2026-08), links back to repo.

## 4. Design Requirements

### 4.1 Visual identity

- Theme: clean scientific/dashboard aesthetic. Dark theme default with a
  light-mode toggle (persisted to `localStorage`). Both themes must pass WCAG AA contrast.
- Accent palette suggestion: electric teal/cyan for DT, neutral amber/orange
  for PPO, muted red for failures/negative values. Consistent across ALL charts.
- Typography: system stack (`ui-sans-serif, -apple-system, "Segoe UI", Roboto, ...`);
  monospace for numbers in tables (`ui-monospace, SFMono-Regular, Menlo`).
- Max content width ~1100px, generous whitespace, clear section rhythm.
- Responsive: usable at 360px width; charts resize; tables become horizontally scrollable on mobile.

### 4.2 Micro-interactions (subtle, performant)

- Count-up animation on hero stats (once, on first scroll into view).
- Section reveal on scroll (IntersectionObserver, `prefers-reduced-motion` respected).
- Chart tooltips (Chart.js built-in).
- No parallax, no heavy libraries, no autoplaying video.

### 4.3 Navigation

Sticky top nav with anchor links: Results · Timeline · Method · Evidence ·
Artifacts · Limitations. Active-section highlight while scrolling. Mobile:
collapsible.

## 5. Technical Quality Bar

- **No console errors** in Chrome/Firefox.
- **Lighthouse (desktop)** targets: Performance ≥ 90, Accessibility ≥ 95,
  Best Practices ≥ 95, SEO ≥ 90.
- Total payload budget: ≤ 500 KB HTML + ≤ 200 KB CDN JS. No images except
  optional inline SVG diagrams (no base64 photos).
- Semantic HTML: proper `h1→h2→h3` hierarchy, `<table>` with `<th scope>`,
  `<figure>/<figcaption>` for charts, skip-to-content link, landmarks.
- Meta: `<title>`, description meta tag, Open Graph tags (title/description/url),
  favicon as inline SVG data URI.
- All interactive elements keyboard-accessible; focus styles visible.
- `prefers-color-scheme` respected on first visit; manual toggle overrides.

## 6. Data Integrity Rules (IMPORTANT)

1. Use **only** the numbers provided in this document or verbatim from
   `report.md`. Never interpolate, estimate, or "smooth" results.
2. Where a figure in this doc is rounded (e.g., impact-gate dollar values),
   keep the rounding and note it.
3. Every chart needs a caption stating the source surface and protocol
   ("identity, 5-min steps, profit/ep net of degradation").
4. Negative results (forecast DT, GRPO, failed fixes) must be presented as
   prominently as wins — this is part of the project's credibility.
5. Distinguish clearly between: shipped standalone DT (deployable),
   solver-in-the-loop variants (not deployable), and legacy/cloning-era models.

## 7. Out of Scope

- Any backend, API, database, or server-side rendering.
- Live data fetching from AEMO/HuggingFace at view time.
- Multi-page structure, routing, i18n.
- Embedding the actual simulator or model weights in the page.
- Analytics/tracking of any kind.

## 8. Acceptance Criteria

- [ ] Single `index.html`, opens correctly via `file://` and via GitHub Pages URL
- [ ] Renders correctly at 360px, 768px, and 1440px widths
- [ ] All §3 sections present, with the exact data from this document
- [ ] Headline bar chart with surface toggle + CI overlay works
- [ ] Timeline stages clickable with correct non-comparability flags
- [ ] Dark/light toggle persists across reload
- [ ] Zero console errors; graceful degradation with CDNs blocked
- [ ] Lighthouse desktop scores meet §5 targets (or deviations justified in PR description)
- [ ] Limitations and negative-results sections present and visible
- [ ] All external links valid (GitHub, Hugging Face)

## 9. Suggested Build Order (for the implementing agent)

1. Skeleton HTML + design tokens (CSS custom properties for both themes) + nav.
2. Static content sections (hero, intro, environment, limitations, footer).
3. Data layer: all JS constants from §3 as a single `DATA` object near the top of the script.
4. Charts (headline → decomposition → timeline → impact gate → auto-mode before/after).
5. Interactions (tabs/toggles, theme switch, scroll effects, nav highlight).
6. Polish: meta tags, accessibility pass, Lighthouse run, reduced-motion support.
7. Verify acceptance criteria checklist; test with network throttled/offline.
