"""Build the paper figures from the experiment results.

Figure 1 (fig_margin_traj.eps): margin trajectories vs. explicit theoretical
bounds, separately for the exact continuous flow (exponential loss) and the
exact discrete algorithm (logistic loss), at a small-gamma and a large-gamma
setting; Monte Carlo mean over replicates with +/-1 sd band.

Figure 2 (fig_realdata.eps): redone after results_e3.json is complete.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
PAPER = os.environ.get("PAPER_DIR", "")   # optional: also copy figures there
L0_LOG = float(np.log(2.0))
QUANTILES = (0.02, 0.05, 0.10, 0.25, 0.50)


def inv_exp(z):
    return -np.log(z)


def inv_log(z):
    return np.log(1.0 / (np.exp(np.minimum(z, 500)) - 1.0))


def flow_bound_traj(recs, p):
    """Theorem 3.2 bound at coverage level 1-p along each replicate's own
    trajectory: B_p(t) = (gamma/2) * l^{-1}(eps(t)/p) / l^{-1}(eps(t)),
    exp loss (l^{-1}(z) = -log z).  NaN where eps/p > l(0) = 1."""
    B = []
    for x in recs:
        g, eps = x['gamma'], np.array(x['flow_eps'])
        ratio = np.where(eps <= p, 1.0 - np.log(1.0 / p) / np.log(1.0 / eps),
                         np.nan)
        B.append((g / 2.0) * ratio)
    return np.array(B)


def discrete_bound_traj(recs, p):
    """Theorem 4.4 bound at coverage level 1-p along each replicate's own
    trajectory (logistic loss).  NaN where eps/p > l(0) = log 2."""
    B = []
    for x in recs:
        g, eps = x['gamma'], np.array(x['log_eps'])
        ok = (eps / p) < L0_LOG * (1 - 1e-9)
        num = np.log(1.0 / (np.exp(eps / p) - 1.0))
        den = np.log(1.0 / (np.exp(eps) - 1.0))
        b = (g / 2.0) * np.where(ok, num / np.where(ok, den, 1.0), np.nan)
        B.append(b)
    return np.array(B)


def fig_margin_traj():
    r = json.load(open('results/results_e1.json'))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))

    # ---- panels (a),(b): flow + exponential loss, risk & margin
    for col, key in enumerate(['m800_d5_gt0.05', 'm800_d5_gt0.45']):
        recs = r[key]
        gam = np.mean([x['gamma'] for x in recs])
        t = np.array(recs[0]['flow_t'])
        eps = np.array([x['flow_eps'] for x in recs])          # reps x grid
        bd = 1.0 / np.maximum(gam ** 2 * t, 1e-12)
        ax = axes[0, col]
        ax.loglog(t, eps.mean(0), 'k-', lw=2, label=r'flow risk $\epsilon(t)$')
        ax.fill_between(t, eps.mean(0) - eps.std(0), np.minimum(eps.mean(0) + eps.std(0), 1),
                        color='gray', alpha=0.3, label=r'$\pm$1 sd')
        ax.loglog(t, np.minimum(bd, 1), 'r--', lw=1.5,
                  label=r'risk bound $1/(\gamma^2 t)$')
        ax.set_xlabel('flow time $t$')
        ax.set_title(f'({chr(97+col)}) flow, exp. loss, '
                     + ('small' if col == 0 else 'large') + rf' $\gamma$'
                     + rf' ($\gamma\approx{gam:.2f}$)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which='both')

        # margins: 25% quantile (75% of points) vs the theorem bound p=0.25
        mq = np.array([x['flow_marginq'] for x in recs])       # reps x grid x 5
        q25 = mq[:, :, QUANTILES.index(0.25)]
        B = flow_bound_traj(recs, 0.25)
        ax = axes[1, col]
        ax.semilogx(t, q25.mean(0), 'k-', lw=2,
                    label='75% margin quantile')
        ax.fill_between(t, q25.mean(0) - q25.std(0), q25.mean(0) + q25.std(0),
                        color='gray', alpha=0.3, label=r'$\pm$1 sd')
        ax.semilogx(t, B.mean(0), 'r--', lw=2,
                    label=r'Thm. bound ($p=0.25$)')
        ax.fill_between(t, np.maximum(B.mean(0) - B.std(0), 0), B.mean(0) + B.std(0),
                        color='red', alpha=0.15)
        ax.set_xlabel('flow time $t$')
        ax.set_ylabel('normalized margin')
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(alpha=0.3, which='both')

    fig.tight_layout()
    fig.savefig('figures/fig_margin_traj.pdf')
    if PAPER: fig.savefig(os.path.join(PAPER, 'fig_margin_traj.pdf'))
    fig.savefig('fig_margin_traj.png', dpi=150)
    print('saved fig_margin_traj')


def fig_discrete_traj():
    """Discrete algorithm (logistic, eta=1): margin quantiles vs bound."""
    r = json.load(open('results/results_e1.json'))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for col, key in enumerate(['m800_d5_gt0.05', 'm800_d5_gt0.45']):
        recs = r[key]
        gam = np.mean([x['gamma'] for x in recs])
        Ts = np.array(recs[0]['log_T'])
        mq = np.array([x['log_marginq'] for x in recs])
        q25 = mq[:, :, QUANTILES.index(0.25)]
        B = discrete_bound_traj(recs, 0.25)
        ax = axes[col]
        ax.semilogx(Ts, q25.mean(0), 'k-', lw=2,
                    label='75% margin quantile')
        ax.fill_between(Ts, q25.mean(0) - q25.std(0), q25.mean(0) + q25.std(0),
                        color='gray', alpha=0.3, label=r'$\pm$1 sd')
        ax.semilogx(Ts, B.mean(0), 'r--', lw=2,
                    label=r'Thm. bound ($p=0.25$)')
        ax.fill_between(Ts, np.maximum(B.mean(0) - B.std(0), 0), B.mean(0) + B.std(0),
                        color='red', alpha=0.15)
        ax.set_xlabel('iteration $T$')
        ax.set_ylabel('normalized margin')
        ax.set_ylim(bottom=0)
        ax.set_title(f'({chr(99+col)}) discrete GD, logistic loss, '
                     + ('small' if col == 0 else 'large') + rf' $\gamma$ ($\approx{gam:.2f}$)')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig('figures/fig_discrete_traj.pdf')
    if PAPER: fig.savefig(os.path.join(PAPER, 'fig_discrete_traj.pdf'))
    fig.savefig('fig_discrete_traj.png', dpi=150)
    print('saved fig_discrete_traj')


if __name__ == '__main__':
    fig_margin_traj()
    fig_discrete_traj()
