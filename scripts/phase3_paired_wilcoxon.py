"""Paired Wilcoxon DT vs PPO over matched (scenario x battery) cells from the v2 sweep."""
import re, sys
import numpy as np
from scipy import stats

path = sys.argv[1] if len(sys.argv) > 1 else 'eval_output/phase3_v2/sweep_full.txt'
txt = open(path).read()
summary = txt.split('── SUMMARY ──')[1]
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
    impact = 'identity' if mi.group(1)=='identity' else 'piecewise_merit_order'
    pol = mi.group(2).replace(' ','_')
    m = re.match(r'(\S+)\s+\$\s*(-?[\d,]+)', right.strip())
    if not m: continue
    rows.append((impact, pol, bat, m.group(1), int(m.group(2).replace(',',''))))

# Best DT per (impact, battery, scenario); PPO direct
dt = {}; ppo = {}
for imp, pol, b, s, p in rows:
    k = (imp, b, s)
    if pol.startswith('dt_rtg'):
        dt[k] = max(dt.get(k, -1e18), p)
    elif pol == 'ppo':
        ppo[k] = p

scen = ['sa1_oct_2024','sa1_nov_2024','vic1_oct_2024']
bats = ['small','hornsdale','torrens']

print("=== Paired Wilcoxon: DT (best RTG) vs PPO ===")
for impact in ['identity', 'piecewise_merit_order']:
    diffs = []
    cells = []
    for b in bats:
        for s in scen:
            k = (impact, b, s)
            if k in dt and k in ppo:
                diffs.append(dt[k] - ppo[k])
                cells.append(f"{b}/{s.split('_')[0]}")
    diffs = np.array(diffs)
    n = len(diffs)
    if n >= 2 and np.any(diffs != 0):
        w, p = stats.wilcoxon(diffs)
        print(f"  {impact:>20}: n={n}, mean_diff=${diffs.mean():,.0f}, "
              f"median_diff=${np.median(diffs):,.0f}, W={w:.0f}, p={p:.4f}")
        pos = int((diffs > 0).sum()); neg = int((diffs < 0).sum())
        print(f"      DT wins {pos}/{n} cells, PPO wins {neg}/{n}")
        if n < 6:
            print(f"      NOTE: n={n} < 6 — Wilcoxon p-value is not discriminating; "
                  f"5-region expansion needed (target n>=10).")
    else:
        print(f"  {impact:>20}: insufficient data (n={n})")

# Also report per-battery for context
print("\n=== DT-PPO mean diff by battery (impact) ===")
for b in bats:
    ds = [dt[('piecewise_merit_order',b,s)]-ppo[('piecewise_merit_order',b,s)]
          for s in scen if ('piecewise_merit_order',b,s) in dt and ('piecewise_merit_order',b,s) in ppo]
    if ds:
        print(f"  {b:>10}: mean=${np.mean(ds):,.0f} across {len(ds)} scenarios")
