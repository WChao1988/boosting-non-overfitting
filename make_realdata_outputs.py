"""Real-data figure and LaTeX tables from results_e3.json."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
PAPER = os.environ.get("PAPER_DIR", "")   # optional: also copy there
ORDER = ['sonar', 'ionosphere', 'breast-cancer', 'waveform', 'glass',
         'segmentation', 'letter']
NICE = {'sonar': 'Sonar', 'ionosphere': 'Ionosphere',
        'breast-cancer': 'Breast Cancer', 'waveform': 'Waveform',
        'glass': 'Glass', 'segmentation': 'Segmentation',
        'letter': 'Letter'}


def fig_realdata():
    r = json.load(open('results/results_e3.json'))
    ds = [d for d in ORDER if d in r]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.4))
    axes = axes.ravel()
    for k, name in enumerate(ds):
        d = r[name]
        T = np.array(d['T'])
        mean = np.array(d['mean'])
        sd = np.array(d['sd'])
        tr = np.array(d['train_mean'])
        ax = axes[k]
        ax.errorbar(T, mean, yerr=sd, fmt='o-', color='red', ms=3,
                    lw=1.4, capsize=2, label='test error')
        ax.plot(T, tr, 'ks--', ms=3, lw=1, label='training error')
        ax.set_xscale('log')
        ax.set_title(f"{NICE[name]} ({d['n']}, {d['classes']} cl.)",
                     fontsize=10)
        ax.set_xlabel('iterations $T$')
        if k % 4 == 0:
            ax.set_ylabel('0-1 error')
        ax.grid(alpha=0.3, which='both')
        if k == 0:
            ax.legend(fontsize=8)
    for k in range(len(ds), 8):
        axes[k].axis('off')
    fig.tight_layout()
    fig.savefig('figures/fig_realdata_new.pdf')
    if PAPER: fig.savefig(os.path.join(PAPER, 'fig_realdata_new.pdf'))
    fig.savefig('fig_realdata_new.png', dpi=130)
    print('saved fig_realdata_new')


def latex_tables():
    r = json.load(open('results/results_e3.json'))
    # Table: mean +- sd at selected T
    cols = [1, 10, 50, 100, 500, 1000]
    print(r'\begin{table}[!ht]')
    print(r'\caption{Test error (mean $\pm$ sd over '
          f"{r[ORDER[0]]['folds']}"r' folds) across iterations on the UCI '
          r'datasets ($10\times5$-fold stratified CV, decision stumps).}'
          r'\label{tb3}')
    print(r'\footnotesize\setlength{\tabcolsep}{3pt}')
    print(r'\begin{tabular}{|c|c|c|c|c|c|c|}')
    print(r'\hline')
    print('Dataset & ' + ' & '.join(f'$T{{=}}{t}$' for t in cols)
          + r' \\ \hline')
    for name in ORDER:
        d = r[name]
        cells = []
        for t in cols:
            i = d['T'].index(t)
            cells.append(f"{d['mean'][i]:.3f}$\\pm${d['sd'][i]:.3f}")
        print(f"{NICE[name]} ({d['n']}) & " + ' & '.join(cells) + r' \\')
    print(r'\hline')
    print(r'\end{tabular}')
    print(r'\end{table}')
    print()
    # Table: overfitting analysis
    print(r'\begin{table}[!ht]')
    print(r'\caption{Prespecified overfitting analysis: overfitting index '
          r'OI $= \mathrm{err}(T_{\max})-\min_T \mathrm{err}(T)$ with '
          r'bootstrap 95\% CIs, and paired Wilcoxon $p$-values '
          r'(T$_{\max}{=}1000$ vs.\ best $T$; Holm-corrected in brackets for '
          r'the adjacent pairs).}\label{tb4}')
    print(r'\begin{tabular}{|c|c|c|c|c|}')
    print(r'\hline')
    print(r'Dataset & $T_{\text{best}}$ & OI [95\% CI] & '
          r'$p$ ($T_{\max}$ vs $T_{\text{best}}$) & flagged pairs \\ \hline')
    for name in ORDER:
        d = r[name]
        w = d['wilcoxon_Tmax_vs_Tbest']
        p = w[1]
        ps = f'{p:.3f}' if p == p else 'n/a'
        pairs = '; '.join(
            f"{pr['Ta']}$\\to${pr['Tb']}: {pr['mean_diff']:+.3f} "
            f"(p={pr['p']:.3f})" for pr in d['pairs'])
        print(f"{NICE[name]} & {d['T_best']} & "
              f"{d['oi_mean']:.3f} [{d['oi_ci'][0]:.3f},{d['oi_ci'][1]:.3f}] & "
              f"{ps} & {pairs} \\\\")
    print(r'\hline')
    print(r'\end{tabular}')
    print(r'\end{table}')


if __name__ == '__main__':
    fig_realdata()
    os.makedirs('tables', exist_ok=True)
    with open('tables/tables_e3.tex', 'w') as f:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            latex_tables()
        f.write(buf.getvalue())
    print('wrote tables/tables_e3.tex')
