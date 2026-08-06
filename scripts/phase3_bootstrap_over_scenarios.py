"""
Phase 3 bootstrap CIs over scenarios (v2 model).

Because a single 14-day scenario has no room for random episode starts
(max_start_idx = len - max_step - 1 < 0), seed-resampling is degenerate.
Instead each SCENARIO is one resample unit: we bootstrap the 3 scenario
profits per (policy, battery, impact) cell with replacement.

Wilcoxon paired test needs >= ~10 matched scenarios to be meaningful; with
n=3 we report it as indicative only (and flag for the 5-region expansion).

Usage: python3 scripts/phase3_bootstrap_over_scenarios.py <sweep_log>
"""

import sys, re
import numpy as np


def parse_sweep(path):
    """Parse a phase3_impact_eval summary into rows."""
    txt = open(path).read()
    if '── SUMMARY ──' in txt:
        summary = txt.split('── SUMMARY ──')[1]
    else:
        summary = txt
    rows = []
    for line in summary.splitlines():
        line = line.strip()
        if not line or line.startswith('--') or line.startswith('Impact'): continue
        m_b = re.search(r'[\s_](small|hornsdale|torrens)_', line)
        if not m_b: continue
        bat = m_b.group(1); left = line[:m_b.start()]; right = line[m_b.end():]
        ls = left.strip()
        mi = re.match(r'(identity|piecewise\s+merit\s+order)[_\s]+(.*)', ls)
        if not mi: continue
        impact = 'identity' if mi.group(1) == 'identity' else 'piecewise_merit_order'
        pol = mi.group(2).replace(' ', '_')
        m = re.match(r'(\S+)\s+\$\s*(-?[\d,]+)', right.strip())
        if not m: continue
        rows.append((impact, pol, bat, m.group(1), int(m.group(2).replace(',', ''))))
    return rows


def bootstrap_ci(values, n_boot=5000, seed=42, conf=0.95):
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=float)
    boots = np.array([rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)])
    alpha = 1.0 - conf
    return dict(mean=float(boots.mean()),
                ci_lower=float(np.percentile(boots, 100 * alpha / 2)),
                ci_upper=float(np.percentile(boots, 100 * (1 - alpha / 2))))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'eval_output/phase3_v2/sweep_full.txt'
    rows = parse_sweep(path)
    print(f"parsed {len(rows)} rows from {path}")

    # DT best per cell, and fixed policies
    dt_best = {}; ppo = {}; orac = {}; fcas = {}
    for imp, pol, b, s, p in rows:
        key = (imp, b, s)
        if pol.startswith('dt_rtg'):
            dt_best[key] = max(dt_best.get(key, -1e18), p)
        elif pol == 'ppo': ppo[key] = p
        elif pol == 'oracle': orac[key] = p
        elif pol == 'fcasrule': fcas[key] = p

    scenarios = ['sa1_oct_2024', 'sa1_nov_2024', 'vic1_oct_2024']

    print("\n=== Bootstrap 95% CI over scenarios (v2) ===")
    for impact in ['identity', 'piecewise_merit_order']:
        for bat in ['small', 'hornsdale', 'torrens']:
            print(f"  {impact:>20} {bat:>10}:")
            for pol, store in [('dt', dt_best), ('ppo', ppo), ('oracle', orac), ('fcasrule', fcas)]:
                vals = [store.get((impact, bat, s)) for s in scenarios]
                vals = [v for v in vals if v is not None]
                if not vals: continue
                c = bootstrap_ci(vals)
                print(f"      {pol:>10}: ${c['mean']:>10,.0f} "
                      f"[${c['ci_lower']:>9,.0f}, ${c['ci_upper']:>9,.0f}]  (n={len(vals)})")

    # Impact resilience with CI (ratio of means)
    print("\n=== DT impact resilience CI (ratio, mean impact / mean identity) ===")
    for bat in ['small', 'hornsdale', 'torrens']:
        rng = np.random.default_rng(7)
        idv = np.array([dt_best[('identity', bat, s)] for s in scenarios])
        imv = np.array([dt_best[('piecewise_merit_order', bat, s)] for s in scenarios])
        ratios = []
        for _ in range(5000):
            ii = rng.choice(3, size=3, replace=True)
            ratios.append(imv[ii].mean() / idv[ii].mean())
        ratios = np.array(ratios)
        print(f"  {bat:>10}: {100*ratios.mean():.0f}% [{100*np.percentile(ratios,2.5):.0f}%, "
              f"{100*np.percentile(ratios,97.5):.0f}%]")

    print("\nNOTE: n=3 scenarios per cell. Wilcoxon needs >= ~10 matched scenarios;")
    print("expand to the 5-region surface for the paired test.")


if __name__ == '__main__':
    main()
