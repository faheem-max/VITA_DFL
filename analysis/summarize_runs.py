"""Builds the per-run result tables from the raw run data (results/runs.zip, unzipped to results/runs/).
Outputs: results/per_run_summary.csv, results/loeo_per_run.csv, results/compression_error.csv
Run from the repository root:  python analysis/summarize_runs.py"""
import os, json, glob, sys
import numpy as np, pandas as pd, torch
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'code')); from models import build_model
RUNS = os.path.join(ROOT, 'results', 'runs')
T = pd.read_csv(os.path.join(ROOT, 'data', 'vehicle', 'test_town07.csv'))
XT = torch.tensor(T.drop(columns=['speed_kmh']).values, dtype=torch.float32); YT = T.speed_kmh.values
MLP = build_model({'model': 'mlp', 'input_dim': 20}); MLP.eval()
def pred(p, X):
    MLP.load_state_dict(torch.load(p, map_location='cpu', weights_only=False))
    with torch.no_grad(): return MLP(X).numpy().ravel()
r2 = lambda p: r2_score(YT, pred(p, XT))

def node_dirs(campaign, run, task, letters):
    return {L: os.path.join(RUNS, campaign, run, f'Node_{L}', run, task) for L in letters}

rows, deltas = [], []
for camp in ['v4', 'v5', 'v6', 'v8']:
    E = pd.read_csv(os.path.join(RUNS, camp, 'cifar_ckpt_eval.csv'))
    for run in sorted(os.listdir(os.path.join(RUNS, camp))):
        if not run.startswith('v') or run.split('_')[0].endswith('tr'): continue      # skip SARSA training episodes
        task = 'vehicle' if 'veh' in run else 'cifar10'; N = 'ABCDEFG' if task == 'vehicle' else 'ABCDEFGH'
        B = node_dirs(camp, run, task, N)
        S = {L: json.load(open(f'{B[L]}/results/node{i+1}_summary.json')) for i, L in enumerate(N)}
        D = {L: pd.read_csv(f'{B[L]}/results/node{i+1}_rounds.csv') for i, L in enumerate(N) if S[L].get('role') != 'data_sender'}
        tr = list(D)                                       # nodes that trained
        mode = S[tr[0]].get('mode', 'allavg'); meth = run.split('_')[1]; seed = int(run.split('_s')[-1])
        cons = all(len({D[L].global_hash[r] for L in N}) == 1 for r in range(len(D['A']))) if mode == 'allavg' else None
        if mode == 'allavg' and 'comp_delta' in D['A']:
            for L in N: deltas += [dict(task=task, method=meth, seed=seed, node=L, round=r, delta=v) for r, v in enumerate(D[L].comp_delta, 1)]
        ck = os.path.join(B['A'], 'checkpoints')
        if task == 'vehicle':
            if mode in ('allavg', 'central'):
                tl = [r2(f'{ck}/global_valbest.pt')]
                tf = r2(f'{ck}/global_final_node1.pt') if mode == 'allavg' else np.nan
            else:
                tl = [r2(f'{ck}/local_valbest_node{i}.pt') for i in range(1, len(N) + 1)]; tf = np.nan
            tavg = r2(f'{ck}/avg_valbest.pt') if mode == 'gossip' else np.nan
        else:
            e = E[E.run == run].set_index('round')
            if mode in ('allavg', 'central'):
                tl = [e.loc[-1, 'test_acc']]; pos = [i for i in e.index if i > 0]
                tf = e.loc[max(pos), 'test_acc'] if (mode == 'allavg' and pos) else np.nan
            else:
                tl = [e.loc[-(100 + i), 'test_acc'] for i in range(1, len(N) + 1)]; tf = np.nan
            tavg = e.loc[-2, 'test_acc'] if mode == 'gossip' else np.nan
        rows.append(dict(task=task, method=meth, seed=seed, campaign=camp, mode=mode, consensus=cons,
                         rounds=max(len(D[L]) for L in tr), wall=max(S[L]['wall_total'] for L in tr),
                         idle=np.mean([D[L].idle.mean() for L in tr]) if mode in ('allavg', 'gossip') else 0.0,
                         MB=sum(D[L].bytes_sent.sum() for L in tr) / 1e6, best_val=S[tr[0]].get('best_goodness'),
                         test=float(np.mean(tl)), test_min=min(tl), test_final=tf, test_avgmodel=tavg))
d = pd.DataFrame(rows)
# campaign v8 also contains two same-session repetitions of the rule (seeds 42-43); make_tables.py keeps them separate
d.to_csv(os.path.join(ROOT, 'results', 'per_run_summary.csv'), index=False)
pd.DataFrame(deltas).to_csv(os.path.join(ROOT, 'results', 'compression_error.csv'), index=False)
print('per-run rows:', len(d), '| bit-identical models in all all-to-all runs:', bool(d[d['mode'] == 'allavg'].consensus.all()))

# ---------------- leave-one-town-out (campaign v7) ----------------
HO = {}
for k in range(1, 8):
    c = pd.read_csv(os.path.join(ROOT, 'data', 'vehicle', f'client_{k}.csv'))
    X = c.drop(columns=['speed_kmh']).values.astype(np.float32); y = c.speed_kmh.values.astype(np.float32)
    _, Xv, _, yv = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)    # same split as code/data.py
    HO[k] = dict(X=torch.tensor(X), y=y, Xv=torch.tensor(Xv), yv=yv)
L7 = []
for run in sorted(os.listdir(os.path.join(RUNS, 'v7'))):
    if not run.startswith('v7veh'): continue
    meth = run.split('_')[1][:-3]; ho = int(run.split('_')[1][-1]); seed = int(run.split('_s')[-1])
    ids = [k for k in range(1, 8) if k != ho]; B = {k: os.path.join(RUNS, 'v7', run, f'Node_{"ABCDEFG"[k-1]}', run, 'vehicle') for k in ids}
    D = {k: pd.read_csv(f'{B[k]}/results/node{k}_rounds.csv') for k in ids}
    S = {k: json.load(open(f'{B[k]}/results/node{k}_summary.json')) for k in ids}
    ck = f'{B[ids[0]]}/checkpoints/global_valbest.pt'
    ph = pred(ck, HO[ho]['X'])
    L7.append(dict(method=meth, held_out=ho, seed=seed, rounds=len(D[ids[0]]), wall=max(S[k]['wall_total'] for k in ids),
                   idle=np.mean([D[k].idle.mean() for k in ids]), MB=sum(D[k].bytes_sent.sum() for k in ids) / 1e6,
                   r2_heldout=r2_score(HO[ho]['y'], ph), rmse_heldout=mean_squared_error(HO[ho]['y'], ph) ** 0.5,
                   r2_heldout_valsplit=r2_score(HO[ho]['yv'], pred(ck, HO[ho]['Xv'])), r2_town07=r2_score(YT, pred(ck, XT))))
# reference: same 20% split of each town when the town WAS part of training (campaign v4, 7 nodes)
for run in sorted(os.listdir(os.path.join(RUNS, 'v4'))):
    if not run.startswith('v4veh_rule_s'): continue
    p = os.path.join(RUNS, 'v4', run, 'Node_A', run, 'vehicle', 'checkpoints', 'global_valbest.pt')
    for k in range(1, 8):
        L7.append(dict(method='rule_seen', held_out=k, seed=int(run.split('_s')[-1]), r2_heldout_valsplit=r2_score(HO[k]['yv'], pred(p, HO[k]['Xv']))))
pd.DataFrame(L7).to_csv(os.path.join(ROOT, 'results', 'loeo_per_run.csv'), index=False)
print('LOEO rows:', len(L7))
