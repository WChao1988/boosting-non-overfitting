"""Generate LaTeX tables for the paper from the experiment results."""
import json
import numpy as np

QUANTILES = (0.02, 0.05, 0.10, 0.25, 0.50)


def table_coverage():
    """Table 5.1: Monte Carlo validation of the margin bound (Theorem 4.4),
    logistic loss, final iterate T=5000, coverage on the p-grid."""
    r = json.load(open('results/results_e1.json'))
    print(r'% Table 5.1: margin-bound coverage, %%NREP%% replicates')
    print(r'\begin{tabular}{|c|c|c|c|c|c|c|}')
    print(r'\hline')
    print(r'$m$ & $d$ & $\kappa$ & $\gamma$ & $\gamma^2 m$ & '
          r'$\min_p(C_p-(1-p))$ & worst rep \\ \hline')
    for key, recs in r.items():
        m = int(key.split('_')[0][1:])
        d = int(key.split('_')[1][1:])
        kap = np.array([x['kappa'] for x in recs])
        gam = np.array([x['gamma'] for x in recs])
        covs = np.array([x['log_cover'] for x in recs])   # reps x p
        need = 1 - np.array(recs[0]['log_pgrid'])
        exc = covs - need[None, :]
        print(f"{m} & {d} & {kap.mean():.1f}$\\pm${kap.std():.1f} & "
              f"{gam.mean():.3f}$\\pm${gam.std():.3f} & "
              f"{(gam**2*m).mean():.1f} & "
              f"{exc.min(1).mean():.3f}$\\pm${exc.min(1).std():.3f} & "
              f"{exc.min():.3f} \\\\")
    print(r'\hline')
    print(r'\end{tabular}')


def table_alpha():
    """Table 5.3: alpha trade-off (Remark 3.7) at the large-gamma setting."""
    r = json.load(open('results/results_e1.json'))
    from common import pacbayes_bound
    recs = r['m3200_d5_gt0.05']
    m = 3200
    gam = float(np.mean([x['gamma'] for x in recs]))
    print(f"% alpha table, setting m=3200, gamma={gam:.4f}")
    print(r'\begin{tabular}{|c|c|c|c|c|c|}')
    print(r'\hline')
    print(r'$\alpha$ & $T(\alpha)$ entry & $1-p$ at $T{=}10^4$ & '
          r'coverage & bound plateau & $\alpha^\star$ range \\ \hline')
    for alpha in (0.25, 0.5, 0.75):
        T = 10000
        p = (gam**2 * T) ** (-alpha)
        # empirical coverage: fraction with margin >= (1-alpha) gamma/2
        # from the last recorded margin quantiles at T=3000 (grid max)
        frac = []
        for x in recs:
            mg_q = np.array(x['log_marginq'][-1])   # quantiles at T=3000
            thr = (1 - alpha) * x['gamma'] / 2
            # interpolate the fraction with margin >= thr from quantiles
            q = np.array(QUANTILES)
            frac.append(1 - np.interp(thr, mg_q, q))
        frac = np.array(frac)
        Tentry = (gam**2)**(1/alpha - 1) * m**(1/alpha)
        plateau = 32 * np.log(m) / ((1 - alpha)**2 * gam**2 * m)
        print(f"{alpha} & {Tentry:.2g} & {1-p:.3f} & "
              f"{frac.mean():.3f}$\\pm${frac.std():.3f} & "
              f"{plateau:.3f} & -- \\\\")
    print(r'\hline')
    print(r'\end{tabular}')


def table_e2():
    r = json.load(open('results/results_e2.json'))
    print(r'% Table 5.2: explicit bound comparison')
    for d in r:
        name = {'A_lowD_strongStumps': 'A', 'B_highD_weakStumps': 'B',
                'C_largeGamma_largeM': 'C',
                'D_largeGamma_veryLargeM': 'D'}[d['name']]
        print(f"% design {name}: m={d['m_train']} d={d['d']} "
              f"kappa={d['kappa']:.1f} gamma={d['gamma']:.3f} "
              f"g2m={d['g2m']:.0f} edges first/mean/min="
              f"{d['ada_edge_first']:.4f}/{d['ada_mean_edge']:.4f}/"
              f"{d['ada_min_edge']:.4f}")
        for i, T in enumerate(d['T']):
            if T not in (100, 1000, 3000):
                continue
            j = d['ada_T'].index(T)
            print(f"%   T={T}: our={d['our_bound'][i]:.3f} (a*={d['our_alpha'][i]:.2f}) "
                  f"ghojMin={d['ghojogh_min_edge'][j]:.2e} "
                  f"ghojMean={d['ghojogh_mean_edge'][j]:.2e} "
                  f"gdErr={d['gd_test_err'][i]:.4f} "
                  f"adaErr={d['ada_test_err'][j]:.4f}")


if __name__ == '__main__':
    print('=' * 30, 'TABLE 5.1 (coverage)')
    table_coverage()
    print('=' * 30, 'TABLE 5.3 (alpha)')
    table_alpha()
    print('=' * 30, 'TABLE 5.2 (E2)')
    table_e2()
