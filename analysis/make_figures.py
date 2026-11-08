"""Draws Figures 2-5 of the paper from the raw run data. Run from the repository root."""
import pandas as pd, numpy as np, glob, os, json, sys, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'code')); from models import build_model
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RUNS = os.path.join(ROOT, 'results', 'runs'); OUT = os.path.join(ROOT, 'results', 'figures'); os.makedirs(OUT, exist_ok=True)
C = {'rule': '#2a78d6', 'bsp': '#eb6834', 'deadline': '#1baf7a', 'fedproxmu01': '#eda100', 'gossip': '#6250d6'}
LBL = {'rule': 'VITA-DFL', 'bsp': 'BSP', 'deadline': 'Deadline', 'fedproxmu01': 'FedProx', 'gossip': 'Gossip'}
MK = {'rule': 'o', 'bsp': 's', 'deadline': '^', 'fedproxmu01': 'D', 'gossip': 'v'}
LS = {'rule': '-', 'bsp': '--', 'deadline': '-.', 'fedproxmu01': ':', 'gossip': (0, (5, 1, 1, 1))}
plt.rcParams.update({'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e6e5e0', 'grid.linewidth': 0.6, 'axes.edgecolor': '#8a8984', 'axes.labelcolor': '#2b2a27',
                     'xtick.color': '#5c5b56', 'ytick.color': '#5c5b56', 'legend.frameon': False, 'pdf.fonttype': 42})
# ---------- Fig A: test score vs wall-clock time ----------
T = pd.read_csv(os.path.join(ROOT, 'data', 'vehicle', 'test_town07.csv'))
XT = torch.tensor(T.drop(columns=['speed_kmh']).values, dtype=torch.float32); YT = T.speed_kmh.values
mdl = build_model({'model': 'mlp', 'input_dim': 20}); mdl.eval()
def r2(p):
    mdl.load_state_dict(torch.load(p, map_location='cpu', weights_only=False))
    with torch.no_grad(): return r2_score(YT, mdl(XT).numpy().ravel())
runs = {'rule': [('v4', 42), ('v4', 43), ('v4', 44), ('v6', 45), ('v6', 46)], 'bsp': [('v4', 42), ('v4', 43), ('v4', 44), ('v6', 45), ('v6', 46)],
        'deadline': [('v4', 42), ('v4', 43), ('v4', 44), ('v6', 45), ('v6', 46)], 'fedproxmu01': [('v8', s) for s in range(42, 47)]}
def curve(task, meth, D, s):
    tag = 'veh' if task == 'vehicle' else 'cif'; v = D
    run = f'{v}{tag}_{meth}_s{s}'
    x = pd.read_csv(glob.glob(f'{RUNS}/{D}/{run}/Node_A/{run}/{task}/results/node1_rounds.csv')[0]).set_index('round')
    pts = []
    if task == 'vehicle':
        for p in glob.glob(f'{RUNS}/{D}/{run}/Node_A/{run}/vehicle/checkpoints/global_r*.pt'):
            r = int(p.split('global_r')[1][:-3]); pts.append((0.0 if r == 0 else x.loc[r, 't_elapsed'], r2(p)))
    else:
        E = pd.read_csv(f'{RUNS}/{D}/cifar_ckpt_eval.csv'); e = E[(E.run == run) & (E['round'] >= 0)]
        pts = [(0.0 if r == 0 else x.loc[int(r), 't_elapsed'], a) for r, a in zip(e['round'], e.test_acc)]
    return pd.DataFrame(sorted(pts), columns=['t', 'y'])
CUR = {}
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
for ax, task, yl, ylim in [(axs[0], 'vehicle', 'Test $R^2$ (Town07)', (0.80, 0.95)), (axs[1], 'cifar10', 'Test accuracy', (0.55, 0.71))]:
    for m in ['bsp', 'deadline', 'fedproxmu01', 'rule']:
        cs = [curve(task, m, D, s) for D, s in runs[m]]
        CUR[(task, m)] = [c.to_dict('list') for c in cs]
        tmax = np.mean([c.t.max() for c in cs]); grid = np.linspace(0, tmax, 200)
        Y = np.array([np.interp(grid, c.t, c.y) for c in cs]); mean = Y.mean(0)
        ax.plot(grid, mean, color=C[m], lw=2, ls=LS[m], label=LBL[m])
        idx = np.linspace(0, len(grid) - 1, 6).astype(int)[1:]
        ax.plot(grid[idx], mean[idx], MK[m], color=C[m], ms=4.5, mec='white', mew=0.8)
    ax.set_ylim(*ylim); ax.set_xlabel('Wall-clock time (s)'); ax.set_ylabel(yl)
    ax.set_title('Vehicle speed regression' if task == 'vehicle' else 'CIFAR-10', fontsize=9, color='#2b2a27')
axs[1].legend(loc='lower right', handlelength=2.6)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_time_curves.pdf'); fig.savefig(f'{OUT}/fig_time_curves.png', dpi=200); plt.close(fig)
json.dump({f'{k[0]}|{k[1]}': v for k, v in CUR.items()}, open(os.path.join(OUT, 'fig_time_curves_data.json'), 'w'))
# ---------- Fig B: waiting time per device ----------
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4))
W = {}
for ax, task, N in [(axs[0], 'vehicle', 'ABCDEFG'), (axs[1], 'cifar10', 'ABCDEFGH')]:
    tag = 'veh' if task == 'vehicle' else 'cif'; ms_ = ['bsp', 'deadline', 'rule']; w = 0.26
    for j, m in enumerate(ms_):
        vals = []
        for i, L in enumerate(N):
            v = []
            for D, s in runs[m]:
                run = f'{D}{tag}_{m}_s{s}'; v.append(pd.read_csv(glob.glob(f'{RUNS}/{D}/{run}/Node_{L}/{run}/{task}/results/node{i+1}_rounds.csv')[0]).idle.mean())
            vals.append(np.mean(v))
        W[f'{task}|{m}'] = dict(zip(N, vals))
        ax.bar(np.arange(len(N)) + (j - 1) * w, vals, w * 0.9, color=C[m], label=LBL[m])
    dev = {'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 'G': 'G', 'H': 'H'}
    ax.set_xticks(range(len(N))); ax.set_xticklabels([f'{L}\n{"Pi" if L in "ABGH" else "Jet"}' for L in N])
    ax.set_ylabel('Mean waiting per round (s)'); ax.grid(axis='x', visible=False)
    ax.set_title('Vehicle speed regression' if task == 'vehicle' else 'CIFAR-10', fontsize=9, color='#2b2a27')
axs[0].legend(loc='upper right')
fig.tight_layout(); fig.savefig(f'{OUT}/fig_waiting.pdf'); fig.savefig(f'{OUT}/fig_waiting.png', dpi=200); plt.close(fig)
json.dump(W, open(os.path.join(OUT, 'fig_waiting_data.json'), 'w'), indent=1)
# ---------- Fig C: one VITA-DFL run: shared validation, intensity and local steps ----------
run = 'v4cif_rule_s42'; N = 'ABCDEFGH'
X = {L: pd.read_csv(glob.glob(f'{RUNS}/v4/{run}/Node_{L}/{run}/cifar10/results/node{i+1}_rounds.csv')[0]) for i, L in enumerate(N)}
fig, axs = plt.subplots(2, 1, figsize=(7.0, 3.6), sharex=True, gridspec_kw={'height_ratios': [1, 1.3]})
a = X['A']; axs[0].plot(a['round'], a.val_good_avg, color='#2a78d6', lw=2, label='Shared validation accuracy $G_t$')
ax2 = axs[0].twinx() if False else None
for r_, gg in zip(a['round'], a.g):
    pass
chg = a[a.g.diff().fillna(0) != 0]
for _, rr in chg.iterrows():
    for ax in axs: ax.axvline(rr['round'], color='#8a8984', lw=0.8, ls='--')
    axs[0].text(rr['round'] + 0.3, a.val_good_avg.min() + 0.02, f'$g_t\\to{rr.g:g}$', fontsize=8, color='#5c5b56')
axs[0].set_ylabel('Shared val. acc.')
for i, L in enumerate(N):
    pi = L in 'ABGH'
    axs[1].plot(X[L]['round'], X[L].steps, color='#eb6834' if pi else '#2a78d6', lw=1.2, alpha=0.9, ls='--' if pi else '-')
axs[1].plot([], [], color='#2a78d6', lw=1.2, label='Jetson Orin Nano Super (C, D, E, F)')
axs[1].plot([], [], color='#eb6834', lw=1.2, ls='--', label='Raspberry Pi 5 (A, B, G, H)')
axs[1].set_ylabel('Local steps $\\tau_{k,t}$'); axs[1].set_xlabel('Round $t$'); axs[1].legend(loc='upper right')
es = a['round'].max(); axs[0].set_xlim(0.5, es + 0.5)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_trace.pdf'); fig.savefig(f'{OUT}/fig_trace.png', dpi=200); plt.close(fig)
json.dump({'run': run, 'g_changes': chg[['round', 'g']].to_dict('list'), 'rounds': int(es),
           'steps_round1': {L: int(X[L].steps.iloc[0]) for L in N}, 'steps_round2': {L: int(X[L].steps.iloc[1]) for L in N},
           'steps_last': {L: int(X[L].steps.iloc[-1]) for L in N}}, open(os.path.join(OUT, 'fig_trace_data.json'), 'w'), indent=1)
# ---------- Fig D: LOEO per fold ----------
lo = pd.read_csv(os.path.join(ROOT, 'results', 'loeo_per_run.csv')).query('method != "rule_seen"').rename(columns={'held_out': 'ho', 'r2_heldout': 'r2_ho'}); TOWN = {1: 'Town01', 2: 'Town02', 3: 'Town03', 4: 'Town04', 5: 'Town05', 6: 'Town06', 7: 'Town10HD'}
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5))
for ax, v, yl in [(axs[0], 'wall', 'Time to early stop (s)'), (axs[1], 'r2_ho', '$R^2$ on held-out town')]:
    g = lo.groupby(['ho', 'method'])[v].mean().unstack()
    for j, m in enumerate(['bsp', 'deadline', 'rule']):
        ax.bar(np.arange(7) + (j - 1) * 0.26, g[m].values, 0.26 * 0.9, color=C[m], label=LBL[m])
    ax.set_xticks(range(7)); ax.set_xticklabels([TOWN[h].replace('Town', 'T') for h in g.index]); ax.set_xlabel('Held-out town')
    ax.set_ylabel(yl); ax.grid(axis='x', visible=False)
    if v == 'r2_ho': ax.set_ylim(0.80, 1.0)
axs[0].legend(loc='upper right')
fig.tight_layout(); fig.savefig(f'{OUT}/fig_loeo.pdf'); fig.savefig(f'{OUT}/fig_loeo.png', dpi=200); plt.close(fig)
print('figures done')
