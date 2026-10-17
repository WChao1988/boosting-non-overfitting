"""E2: explicit generalization-bound comparison (replaces the old Table 3).

Fixes to Report 1, point 4 and Report 2:
  - the margin gamma is measured by a hard-margin SVM in exactly the geometry
    of the revised theory (unit ball, unit-norm linear separator);
  - our bound is reported with EXPLICIT constants (no tilde-O dropped):
        err <= 2 p + 1/m + 16 ln m/((1-a)^2 g^2 m) + 2 ln(2 sqrt(m)/d)/m,
    p = (g^2 T)^{-a}, optimized over a, delta = 0.05;
  - the competing Ghojogh-Crowley bound is applied with the correct
    interpretation of xi as the per-round EDGE (their eq. (111) requires
    weighted error <= 1/2 - xi for every round), using both the minimum edge
    over rounds (the only assumption-valid value) and the mean edge (as in
    the original submission).

Designs:
  A: d=5,  moderate separation -> stumps have a large edge;
  B: d=50, signal spread thinly -> stumps have a tiny edge but the linear
     margin gamma is large (the boosting-relevant "many weak features"
     regime);
  C: d=5, strong separation, large m -> the non-vacuous regime of our bound.
"""
import json
import numpy as np
from common import (make_separable, svm_margin, gd_final, StumpAdaBoost,
                    pacbayes_bound, ghojogh_bound, loss_of)

RNG = np.random.default_rng(7)
T_GRID = [10, 30, 100, 300, 1000, 3000]
THETA = 0.01
DELTA = 0.05


def run_design(name, m_train, m_test, d, gamma_target, T_ada=1000):
    Xtr, ytr, kappa, gamma, wstar = make_separable(m_train, d, 2.0, RNG,
                                                   gamma_target=gamma_target)
    Xte, yte, _, _, _ = make_separable(m_test, d, kappa, RNG,
                                       gamma_target=gamma_target)
    out = dict(name=name, d=d, kappa=kappa, gamma=gamma,
               g2m=gamma ** 2 * m_train, m_train=m_train)

    # ---- analyzed algorithm: discrete GD, logistic loss, eta = 1
    Tg = T_GRID + [10000]
    test_err, train_eps = [], []
    for T in Tg:
        w = gd_final(Xtr, ytr, 'log', 1.0, T)
        test_err.append(float(np.mean(np.sign(Xte @ w) != yte)))
        train_eps.append(float(loss_of('log', ytr * (Xtr @ w))))
    out['T'] = Tg
    out['gd_test_err'] = test_err
    out['gd_train_eps'] = train_eps

    # our explicit bound
    our = [pacbayes_bound(gamma, m_train, T, DELTA)[0] for T in Tg]
    our_a = [pacbayes_bound(gamma, m_train, T, DELTA)[1] for T in Tg]
    out['our_bound'] = our
    out['our_alpha'] = our_a

    # ---- stump AdaBoost: edges and Ghojogh bounds
    ada = StumpAdaBoost(T_ada).fit(Xtr, ytr)
    edges = 0.5 - np.array(ada.err_hist)
    out['ada_mean_edge'] = float(edges.mean())
    out['ada_min_edge'] = float(edges.min())
    out['ada_edge_first'] = float(edges[0])
    out['ada_n_rounds'] = int(ada.n_rounds)
    out['ada_edges_head'] = edges[:10].tolist()
    out['ada_edges_tail'] = edges[-5:].tolist()
    ada_err = [float(np.mean(np.sign(ada.decision_function(Xte)) != yte))]
    # test error along the AdaBoost path
    for T in T_GRID[1:]:
        a2 = StumpAdaBoost(T).fit(Xtr, ytr)
        ada_err.append(float(np.mean(np.sign(a2.decision_function(Xte)) != yte)))
    out['ada_T'] = T_GRID
    out['ada_test_err'] = ada_err
    out['ghojogh_min_edge'] = [min(ghojogh_bound(max(out['ada_min_edge'], 1e-9),
                                                 THETA, T), 99.0)
                               for T in T_GRID]
    out['ghojogh_mean_edge'] = [min(ghojogh_bound(max(out['ada_mean_edge'], 1e-9),
                                                  THETA, T), 99.0)
                                for T in T_GRID]
    show = {k: v for k, v in out.items() if k not in ('gd_train_eps',)}
    show['our_bound'] = [min(b, 2.0) for b in out['our_bound']]
    print(json.dumps(show, indent=1)[:1500], flush=True)
    return out


if __name__ == "__main__":
    res = []
    res.append(run_design("A_lowD_strongStumps", 2000, 3000, 5, 0.25))
    res.append(run_design("B_highD_weakStumps", 2000, 3000, 50, 0.30))
    res.append(run_design("C_largeGamma_largeM", 5000, 3000, 5, 0.55))
    res.append(run_design("D_largeGamma_veryLargeM", 20000, 5000, 5, 0.55))
    with open("results/results_e2.json", "w") as f:
        json.dump(res, f, indent=1)
    print("saved results_e2.json")
