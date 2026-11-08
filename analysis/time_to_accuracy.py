"""CIFAR-10 time to accuracy (Section 7.1): test accuracy of the periodic consensus models (every 5 rounds)
against wall-clock time at the end of that round.  Run from the repository root."""
import os, glob
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RUNS = os.path.join(ROOT, 'results', 'runs')
SEL = {'bsp': ['v4', 'v6'], 'deadline': ['v4', 'v6'], 'rule': ['v4', 'v6'], 'fedproxmu01': ['v8']}
rows = []
for meth, camps in SEL.items():
    for camp in camps:
        E = pd.read_csv(os.path.join(RUNS, camp, 'cifar_ckpt_eval.csv'))
        for run, e in E[(E['round'] >= 0) & E.run.str.contains(f'cif_{meth}_s')].groupby('run'):
            f = glob.glob(os.path.join(RUNS, camp, run, 'Node_A', run, 'cifar10', 'results', 'node1_rounds.csv'))
            if not f: continue
            x = pd.read_csv(f[0]).set_index('round')
            for r, a in zip(e['round'], e.test_acc):
                rows.append(dict(method=meth, run=run, t=0.0 if r == 0 else x.loc[int(r), 't_elapsed'], acc=a))
c = pd.DataFrame(rows); c.to_csv(os.path.join(ROOT, 'results', 'cifar_test_accuracy_vs_time.csv'), index=False)
for a in [0.66, 0.68, 0.685]:
    z = c.groupby(['method', 'run']).apply(lambda g: g[g.acc >= a].t.min(), include_groups=False)
    print(f'time to test accuracy >= {a}:', {m: f'{v.mean():.1f} s ({v.notna().sum()}/{len(v)} runs)' for m, v in z.groupby(level=0)})
