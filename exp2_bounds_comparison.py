"""E2: numerical values of the explicit generalization bound (Table 5.2).

For the four synthetic designs (A--D) of the paper's Table 5.2 this script
  - generates separable two-class Gaussian data with the protocol of
    ``common.make_separable`` (kappa raised until the sample is separable
    with the target margin; gamma measured by a hard-margin SVM on the
    unit-ball-rescaled training set, so the measured gamma is exactly the
    theoretical quantity),
  - runs the analyzed algorithm (B: gradient descent with step size 1 on
    the logistic loss) and, as a beyond-scope check, stagewise Discrete
    AdaBoost with decision stumps,
  - evaluates the fully explicit bound (``common.pacbayes_bound``) with
    delta = 0.05 and alpha optimized numerically for each (gamma, m, T).

Designs:
  A: d=5,   moderate separation;
  B: d=50,  signal spread over all coordinates ("many weak features");
  C: d=5,   strong separation, large m;
  D: d=5,   very large m -- the regime gamma^2 m >> log m in which the
            bound falls below one and decreases with T.
"""
import json
import numpy as np
from common import (make_separable, gd_final, StumpAdaBoost,
                    pacbayes_bound, loss_of)

RNG = np.random.default_rng(7)
T_GRID = [10, 30, 100, 300, 1000, 3000]
DELTA = 0.05


def run_design(name, m_train, m_test, d, gamma_target):
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

    # our explicit bound (alpha optimized)
    our = [pacbayes_bound(gamma, m_train, T, DELTA)[0] for T in Tg]
    our_a = [pacbayes_bound(gamma, m_train, T, DELTA)[1] for T in Tg]
    out['our_bound'] = our
    out['our_alpha'] = our_a

    # ---- stump AdaBoost (beyond-scope check): test error along the path
    ada_err = []
    for T in T_GRID:
        a2 = StumpAdaBoost(T).fit(Xtr, ytr)
        ada_err.append(float(np.mean(np.sign(a2.decision_function(Xte)) != yte)))
    out['ada_T'] = T_GRID
    out['ada_test_err'] = ada_err

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
