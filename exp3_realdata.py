"""E3: real-data experiments (Section 6 of the paper).

Protocol: 10 x 5-fold stratified cross-validation (Letter: 4 x 5) with
decision stumps, a prespecified overfitting index
OI = err(T_max) - min_T err(T) with bootstrap CIs, and paired Wilcoxon
signed-rank tests for the primary comparison T=1000 vs. the best mean-error
T (Holm-corrected across the seven datasets) plus prespecified
adjacent-pair comparisons (reported as raw p-values).

Data handling: datasets are read from the local ``data/`` folder when
present; any missing file is downloaded from the UCI repository and cached
there, so the repository is self-contained.
"""
import io
import json
import os
import time
import urllib.request
import numpy as np
from scipy import stats
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

T_GRID = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
N_REPEATS = 10
RNG = np.random.default_rng(2026)
RESULTS = "results/results_e3.json"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases"


def fetch(url, tries=5):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:
            print(f"  retry {i+1} for {url}: {e}", flush=True)
            time.sleep(5 * (i + 1))
    raise RuntimeError(url)


def local_or_download(fname, url):
    """Read data/<fname> if present; otherwise download and cache it."""
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return f.read()
    raw = fetch(url)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(raw)
    print(f"  downloaded and cached {fname}", flush=True)
    return raw


def load_dataset(name):
    if name == 'sonar':
        raw = local_or_download('sonar.all-data',
                                f"{BASE}/undocumented/connectionist-bench/sonar/sonar.all-data")
        A = np.array([l.strip().split(',') for l in io.StringIO(raw.decode()) if l.strip()])
        X = A[:, :-1].astype(float)
        y = np.where(A[:, -1] == 'R', 1.0, -1.0)
    elif name == 'ionosphere':
        raw = local_or_download('ionosphere.data',
                                f"{BASE}/ionosphere/ionosphere.data")
        A = np.array([l.strip().split(',') for l in io.StringIO(raw.decode()) if l.strip()])
        X = A[:, :-1].astype(float)
        X = X[:, 2:]                      # first two columns are constant 1/0
        y = np.where(A[:, -1] == 'g', 1.0, -1.0)
    elif name == 'breast-cancer':
        raw = local_or_download('breast-cancer.data',
                                f"{BASE}/breast-cancer/breast-cancer.data")
        A = np.array([l.strip().split(',') for l in io.StringIO(raw.decode()) if l.strip()])
        # class is the FIRST column; ordinal-encode the 9 categorical attributes
        from sklearn.preprocessing import OrdinalEncoder
        X = OrdinalEncoder().fit_transform(A[:, 1:]).astype(float)
        y = np.where(A[:, 0] == 'no-recurrence-events', 1.0, -1.0)
    elif name == 'waveform':
        path = os.path.join(DATA_DIR, 'waveform.data')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                raw = f.read()
        else:
            from unlzw3 import unlzw
            zbytes = local_or_download('waveform.data.Z',
                                       f"{BASE}/waveform/waveform.data.Z")
            raw = unlzw(zbytes)
            with open(path, 'wb') as f:      # cache the decompressed copy
                f.write(raw)
        rows = [r for r in io.StringIO(raw.decode(errors='replace')) if r.strip()]
        X = np.array([[float(v) for v in r.strip().split(',')[:21]] for r in rows])
        y = np.array([float(r.strip().split(',')[-1]) for r in rows])
    elif name == 'glass':
        raw = local_or_download('glass.data', f"{BASE}/glass/glass.data")
        A = np.array([l.strip().split(',') for l in io.StringIO(raw.decode()) if l.strip()])
        X = A[:, 1:-1].astype(float)
        y = A[:, -1].astype(int)
    elif name == 'segmentation':
        raw = local_or_download('segmentation.data',
                                f"{BASE}/image/segmentation.data")
        rows = [l.strip().split(',') for l in io.StringIO(raw.decode())]
        # keep only the 210 data rows (label + 19 features); header and
        # comment lines have different field counts
        rows = [r for r in rows if len(r) == 20]
        A = np.array(rows)
        y = A[:, 0]
        X = A[:, 1:].astype(float)
    elif name == 'letter':
        raw = local_or_download('letter-recognition.data',
                                f"{BASE}/letter-recognition/letter-recognition.data")
        A = np.array([l.strip().split(',') for l in io.StringIO(raw.decode()) if l.strip()])
        X = A[:, 1:].astype(float)
        y = A[:, 0]
    else:
        raise ValueError(name)
    return X, y


def run_dataset(name, X, y):
    print(f"== {name}: X={X.shape}, classes={len(np.unique(y))}", flush=True)
    n_reps = N_REPEATS if name != 'letter' else 4   # letter is 20k x 26 classes
    errs = np.zeros((n_reps * 5, len(T_GRID)))
    trains = np.zeros_like(errs)
    k = 0
    for rep in range(n_reps):
        skf = StratifiedKFold(n_splits=5, shuffle=True,
                              random_state=int(RNG.integers(1e6)))
        for tr, te in skf.split(X, y):
            for j, T in enumerate(T_GRID):
                clf = AdaBoostClassifier(
                    estimator=DecisionTreeClassifier(max_depth=1),
                    n_estimators=T)
                clf.fit(X[tr], y[tr])
                errs[k, j] = np.mean(clf.predict(X[te]) != y[te])
                trains[k, j] = np.mean(clf.predict(X[tr]) != y[tr])
            k += 1
        print(f"  rep {rep+1}/{n_reps} done", flush=True)

    mean, sd = errs.mean(0), errs.std(0, ddof=1)
    i_best = int(np.argmin(mean))

    # prespecified overfitting analysis
    # OI: err(T_max) - min_T err(T), paired per fold
    oi = errs[:, -1] - errs.min(1)
    boot = np.array([RNG.choice(oi, size=len(oi), replace=True).mean()
                     for _ in range(5000)])
    res = dict(name=name, n=int(X.shape[0]), d=int(X.shape[1]),
               classes=int(len(np.unique(y))), folds=int(k),
               T=T_GRID, mean=mean.tolist(), sd=sd.tolist(),
               train_mean=trains.mean(0).tolist(),
               oi_mean=float(oi.mean()),
               oi_ci=[float(np.quantile(boot, 0.025)),
                      float(np.quantile(boot, 0.975))],
               T_best=int(T[i_best]))
    # primary paired test: T_max vs T_best
    try:
        w = stats.wilcoxon(errs[:, -1], errs[:, i_best])
        res['wilcoxon_Tmax_vs_Tbest'] = [float(w.statistic), float(w.pvalue)]
    except ValueError:
        res['wilcoxon_Tmax_vs_Tbest'] = [float('nan'), float('nan')]

    # prespecified adjacent-pair comparisons
    pairs = []
    for (a, b) in [(1, 10), (50, 100), (500, 1000), (100, 1000)]:
        if a in T_GRID and b in T_GRID:
            ia, ib = T_GRID.index(a), T_GRID.index(b)
            diff = errs[:, ib] - errs[:, ia]
            if np.allclose(diff, 0):
                p = 1.0
            else:
                try:
                    p = float(stats.wilcoxon(errs[:, ib], errs[:, ia]).pvalue)
                except ValueError:
                    p = 1.0
            pairs.append(dict(Ta=a, Tb=b, mean_diff=float(diff.mean()),
                              frac_increase=float(np.mean(diff > 0)), p=p))
    res['pairs'] = pairs
    print(json.dumps({k2: v for k2, v in res.items() if k2 != 'train_mean'},
                     indent=1), flush=True)
    return res


def holm_adjust(pvals):
    """Holm correction across the primary tests; returns adjusted p-values."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 0.0
    for rank, i in enumerate(order):
        a = min(1.0, (m - rank) * pvals[i])
        adj[i] = max(a, prev)
        prev = adj[i]
    return adj


if __name__ == "__main__":
    all_res = {}
    if os.path.exists(RESULTS):
        all_res = json.load(open(RESULTS))
    for name in ['sonar', 'ionosphere', 'breast-cancer', 'waveform',
                 'glass', 'segmentation', 'letter']:
        if name in all_res and 'mean' in all_res[name]:
            print(f"{name}: already done, skipping", flush=True)
            continue
        for attempt in range(3):
            try:
                X, y = load_dataset(name)
                break
            except Exception as e:
                print(f"load {name} failed ({e}), attempt {attempt+1}", flush=True)
                time.sleep(10)
        else:
            continue
        all_res[name] = run_dataset(name, X, y)
        os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
        with open(RESULTS, "w") as f:
            json.dump(all_res, f, indent=1)

    # Holm correction across the seven primary tests
    names = [n for n in all_res if 'wilcoxon_Tmax_vs_Tbest' in all_res[n]]
    raw = [all_res[n]['wilcoxon_Tmax_vs_Tbest'][1] for n in names]
    adj = holm_adjust(raw)
    print("\nHolm-adjusted p-values (T_max vs T_best):")
    for n, r_, a_ in zip(names, raw, adj):
        print(f"  {n}: raw={r_:.4f}  Holm={a_:.4f}")
    print("saved", RESULTS)
