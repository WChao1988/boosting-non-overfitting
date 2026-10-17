"""E1: Monte-Carlo validation of the margin lower bounds (revised Theorems
3.2/3.4 for the flow + exponential loss, 4.3/4.4 for discrete GD + logistic
loss), plus the risk-decay lemmas.

Design (responding to Report 2, bullet 3):
  - repeated Monte Carlo samples (n_rep repeats), mean +/- sd reported
  - vary m in {200, 800, 3200}, d in {5, 50}, and the separation kappa so
    that gamma spans ~ {0.05, 0.25, 0.45} (the margin of the data)
  - evaluate coverage on the full admissible grid of p in [eps/l(0), 1]
  - margin trajectories across iterations compared with explicit bounds
  - the exact analyzed algorithms are used (small-step GD for the flow,
    discrete GD with eta = 1 for the logistic loss); stump AdaBoost is also
    tracked as an empirical beyond-scope check.

Margin quantiles are recorded so that the coverage of the alpha-corollaries
can be evaluated for any alpha in post-processing (Report 3, alpha question).
"""
import json
import numpy as np
from common import make_separable, svm_margin, gd_track, StumpAdaBoost

RNG = np.random.default_rng(2026)
RESULTS = "results/results_e1.json"
L0 = {'exp': 1.0, 'log': float(np.log(2.0))}
QUANTILES = (0.02, 0.05, 0.10, 0.25, 0.50)


def inv_loss(loss, z):
    z = min(z, 1e5)
    if loss == 'exp':
        return -np.log(z)
    return np.log(1.0 / (np.exp(z) - 1.0))


def margin_bound_theory(loss, eps, p, gamma):
    """Theorem margin lower bound: (gamma/2) l^{-1}(eps/p) / l^{-1}(eps).

    Degenerate only when eps = l(0) (both inverse values vanish)."""
    if eps >= L0[loss] - 1e-12 or p <= 0 or p > 1:
        return np.nan
    num = inv_loss(loss, eps / p)
    den = inv_loss(loss, eps)
    return (gamma / 2.0) * (num / den)


def run_setting(m, d, gamma_target, n_rep, T_discrete, flow_time):
    recs = []
    for r in range(n_rep):
        X, y, kappa, gamma, wstar = make_separable(m, d, 2.0, RNG,
                                                   gamma_target=gamma_target)
        rec = dict(gamma=gamma, kappa=kappa)

        # ---- discrete GD, logistic loss, eta = 1 (exact analyzed algorithm)
        tr = gd_track(X, y, 'log', 1.0, T_discrete)
        Tgrid = np.array([10, 30, 100, 300, 1000, 3000])
        idx = [int(np.searchsorted(tr['T'], t)) for t in Tgrid]
        idx = [min(i, len(tr['T']) - 1) for i in idx]
        eps_traj, mq_traj = [], []
        for i in idx:
            W = tr['W'][i]
            eps_traj.append(float(tr['eps'][i]))
            mg = y * (X @ W) / np.linalg.norm(W)
            mq_traj.append(np.quantile(mg, QUANTILES).tolist())
        rec['log_T'] = Tgrid.tolist()
        rec['log_eps'] = eps_traj
        rec['log_marginq'] = mq_traj

        # coverage on the admissible p grid at the final iterate
        eps_log = tr['eps'][-1]
        W = tr['W'][-1]
        margins = y * (X @ W) / np.linalg.norm(W)
        pgrid = np.linspace(max(eps_log / L0['log'], 0.05), 0.95, 10)
        cov, thr = [], []
        for p in pgrid:
            b = margin_bound_theory('log', eps_log, p, gamma)
            thr.append(b)
            cov.append(float(np.mean(margins >= b - 1e-12)))
        rec['log_pgrid'] = pgrid.tolist()
        rec['log_cover'] = cov
        rec['log_thr'] = thr
        rec['log_eps_final'] = eps_log

        # ---- gradient flow, exponential loss (small-step GD, step -> 0)
        step = 0.01
        n_it = int(flow_time / step)
        every = max(n_it // 50, 1)
        trf = gd_track(X, y, 'exp', step, n_it, record_every=every)
        tgrid = trf['T'] * step
        rec['flow_t'] = tgrid.tolist()
        rec['flow_eps'] = trf['eps'].tolist()
        rec['flow_bound_eps'] = [1.0 / max(gamma * gamma * t, 1e-12)
                                 for t in tgrid]
        mqf = []
        for Wt in trf['W']:
            nw = np.linalg.norm(Wt)
            mg = y * (X @ Wt) / nw if nw > 1e-12 else np.zeros(m)
            mqf.append(np.quantile(mg, QUANTILES).tolist())
        rec['flow_marginq'] = mqf

        # ---- stump AdaBoost (empirical check, beyond theory scope)
        ada = StumpAdaBoost(300).fit(X, y)
        F = ada.F_path[-1]
        nrm = np.linalg.norm(F)
        margins_ada = y * F / (nrm if nrm > 0 else 1.0)
        rec['ada_marginq'] = np.quantile(margins_ada, QUANTILES).tolist()
        if ada.n_rounds > 0:
            edges = 0.5 - np.array(ada.err_hist)
            rec['ada_mean_edge'] = float(edges.mean())
            rec['ada_min_edge'] = float(edges.min())
            rec['ada_n_rounds'] = int(ada.n_rounds)
        else:
            rec['ada_mean_edge'] = None
            rec['ada_min_edge'] = None
            rec['ada_n_rounds'] = 0
        rec['ada_eps_exp'] = float(np.exp(-y * F).mean())
        rec['ada_train_err'] = float(np.mean(np.sign(F) != y))

        recs.append(rec)
        ame = rec['ada_min_edge']
        print(f"  rep {r+1}/{n_rep}: kappa={kappa:.1f} gamma={gamma:.4f} "
              f"g2m={gamma**2*m:.1f} eps_log={eps_log:.1e} "
              f"ada_rounds={rec['ada_n_rounds']} "
              f"ada_min_edge={ame if ame is not None else float('nan'):.4f}",
              flush=True)
    return recs


if __name__ == "__main__":
    settings = [
        dict(m=200,  d=5,  gamma_target=0.05),   # small gamma
        dict(m=800,  d=5,  gamma_target=0.05),
        dict(m=3200, d=5,  gamma_target=0.05),
        dict(m=800,  d=5,  gamma_target=0.25),   # medium gamma
        dict(m=800,  d=5,  gamma_target=0.45),   # large gamma
        dict(m=800,  d=50, gamma_target=0.03),   # higher dimension
    ]
    n_rep = 20
    import os
    out = {}
    if os.path.exists(RESULTS):
        out = json.load(open(RESULTS))
    for s in settings:
        key = f"m{s['m']}_d{s['d']}_gt{s['gamma_target']}"
        if key in out and len(out[key]) >= n_rep:
            print(f"setting {key}: already done, skipping", flush=True)
            continue
        print(f"setting {key}", flush=True)
        out[key] = run_setting(s['m'], s['d'], s['gamma_target'], n_rep,
                               T_discrete=5000, flow_time=2000)
        with open(RESULTS, "w") as f:      # save incrementally
            json.dump(out, f)
    print("saved", RESULTS)
