"""Prints every number of the result tables and the statistical tests of the paper from results/*.csv.
Run from the repository root after analysis/summarize_runs.py:  python analysis/make_tables.py"""
import os, json
import numpy as np, pandas as pd
from scipy import stats
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = pd.read_csv(os.path.join(ROOT, 'results', 'per_run_summary.csv'))
# the main comparison: rule/BSP/Deadline/Fixed/Schedule/SARSA/ablations from v4-v6, the new baselines from v8;
# the two rule runs of v8 are the same-session check (Section 7.5)
main = d[~((d.campaign == 'v8') & (d.method == 'rule'))]
same_session = d[(d.campaign == 'v8') & (d.method == 'rule')]
NAMES = {'bsp': 'BSP', 'deadline': 'Deadline', 'fixed50': 'Fixed 50%', 'fedproxmu01': 'FedProx (mu=0.1)', 'gossip': 'Gossip',
         'local': 'Local only', 'central': 'Centralized', 'schbest': 'Schedule (tuned)', 'rule': 'VITA-DFL',
         'noshape': 'w/o deadline shape', 'noint8': 'w/o 8-bit compression', 'noes': 'w/o early stopping',
         'rlint': 'SARSA-1', 'rlint2': 'SARSA-2', 'rulep2': 'Rule p=2', 'rulep5': 'Rule p=5',
         'fedproxmu0001': 'FedProx (mu=0.001)', 'fedproxmu001': 'FedProx (mu=0.01)'}
welch = lambda a, b: stats.ttest_ind(a, b, equal_var=False).pvalue
out = {}
pd.set_option('display.width', 250)
for task in ['vehicle', 'cifar10']:
    x = main[main.task == task]; r = x[x.method == 'rule']; rows = []
    for m in NAMES:
        g = x[x.method == m]
        if not len(g): continue
        rows.append(dict(method=NAMES[m], n=len(g), rounds=g.rounds.mean(), time_s=g.wall.mean(), time_sd=g.wall.std(),
                         wait_s=g.idle.mean(), MB=g.MB.mean(), best_val=g.best_val.mean(), test=g.test.mean(), test_sd=g.test.std(),
                         test_final=g.test_final.mean(), p_time=np.nan if m in ('rule', 'local', 'central') else welch(r.wall, g.wall),
                         p_test=np.nan if m == 'rule' else welch(r.test, g.test)))
    t = pd.DataFrame(rows).set_index('method'); out[task] = t
    print(f'\n================ {task} ================'); print(t.round(4).to_string())
print('\nsame-session rule runs (campaign v8):'); print(same_session[['task', 'seed', 'wall', 'test']].to_string(index=False))

L = pd.read_csv(os.path.join(ROOT, 'results', 'loeo_per_run.csv')); lo = L[L.method != 'rule_seen']
print('\n================ leave-one-town-out ================')
print(lo.groupby('method')[['wall', 'idle', 'MB', 'r2_heldout', 'rmse_heldout', 'r2_town07']].mean().round(4).to_string())
P = lo.pivot_table(index=['held_out', 'seed'], columns='method', values=['wall', 'r2_heldout', 'r2_town07'])
for b in ['bsp', 'deadline']:
    print(f'paired t-test rule vs {b}:', {v: float(stats.ttest_rel(P[(v, 'rule')], P[(v, b)]).pvalue) for v in ['wall', 'r2_heldout', 'r2_town07']})
seen = L[L.method == 'rule_seen'].groupby('held_out').r2_heldout_valsplit.mean()
unseen = lo[lo.method == 'rule'].groupby('held_out').r2_heldout_valsplit.mean()
print('R2 drop when the town is excluded (same 20% split):', (seen - unseen).round(4).to_dict())

C = pd.read_csv(os.path.join(ROOT, 'results', 'compression_error.csv'))
print('\n================ compression error delta ================')
print(C.groupby(['task', C.method.str.replace(r'mu.*', '', regex=True)]).delta.describe(percentiles=[.5, .99])[['count', '50%', '99%', 'max']].to_string())
print('largest delta over all messages:', C.delta.max(), ' c_delta =', 2 * C.delta.max() * (1 + C.delta.max()) / (1 - C.delta.max()) ** 2)
