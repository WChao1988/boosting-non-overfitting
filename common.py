"""Shared utilities for the revised experiments.

Theory setting (revised paper):
  - data (x_i, y_i), x in the unit ball ||x||_2 <= 1, y in {-1, +1}
  - linear predictors F_w(x) = <w, x>
  - gamma = max-margin of a unit-norm linear separator (measured by hard-margin SVM)
  - analyzed algorithms:
      (A) gradient flow on the exponential loss (approximated by small-step GD),
      (B) discrete gradient descent with step size eta = 1 on the logistic loss
          (exactly the algorithm of Ji & Telgarsky 2018, Thm 3.1),
      (C) stagewise Discrete AdaBoost with decision stumps (beyond the strict
          scope of the theory; empirical check).
"""
import numpy as np
from sklearn.svm import LinearSVC


# ----------------------------------------------------------------------
# data generation
# ----------------------------------------------------------------------
def make_gaussian_two_class(m, d, kappa, rng, unit_ball=True):
    """Two-class Gaussian data with equal class sizes.

    Class centers are +/-(kappa/sqrt(d)) * 1_d, so the signal norm is kappa
    and the per-coordinate signal is kappa/sqrt(d) (weak features when d is
    large).  If unit_ball, the whole sample is rescaled by max_i ||x_i||_2 so
    that ||x_i|| <= 1 as required by the theory.
    """
    mu = (kappa / np.sqrt(d)) * np.ones(d)
    y = np.ones(m)
    y[m // 2:] = -1.0
    X = rng.standard_normal((m, d)) + y[:, None] * mu[None, :]
    if unit_ball:
        X = X / np.linalg.norm(X, axis=1).max()
    return X, y


def make_separable(m, d, kappa0, rng, unit_ball=True, gamma_target=1e-4):
    """Gaussian two-class data, increasing kappa until the sample is linearly
    separable with margin at least gamma_target (kappa is chosen at the
    data-generation stage so that the training set is separable with the
    target margin).  Returns (X, y, kappa_used, gamma, w_star)."""
    kappa = kappa0
    while True:
        X, y = make_gaussian_two_class(m, d, kappa, rng, unit_ball)
        gamma, w = svm_margin(X, y)
        if gamma >= gamma_target:
            return X, y, kappa, gamma, w
        kappa *= 1.3


def svm_margin(X, y):
    """Hard-margin linear SVM margin (gamma of Assumption 1 in the paper).

    Returns (gamma, w_star) with ||w_star||_2 = 1 such that
    min_i y_i <w_star, x_i> = gamma (up to solver accuracy).
    LinearSVC with a huge C approximates the hard-margin solution; a negative
    value signals that the sample is not linearly separable.
    """
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf = LinearSVC(C=1e10, fit_intercept=False, tol=1e-10,
                        max_iter=200000, dual=True)
        clf.fit(X, y)
    w = clf.coef_.ravel()
    w = w / np.linalg.norm(w)
    margins = y * (X @ w)
    return float(margins.min()), w


# ----------------------------------------------------------------------
# losses
# ----------------------------------------------------------------------
def exp_loss(z):
    return np.exp(-z)


def log_loss(z):
    return np.log1p(np.exp(-np.clip(z, -500, 500)))


# ----------------------------------------------------------------------
# analyzed algorithms (linear predictors)
# ----------------------------------------------------------------------
def gd_track(X, y, loss, step, T, record_every=1):
    """Gradient descent on (1/m) sum_i loss(y_i <w, x_i>), w_0 = 0.

    step: step size eta (scalar). For the *flow* approximation use a small
    step; time = iteration * step.
    Returns dict with recorded iterate paths.
    """
    m, d = X.shape
    w = np.zeros(d)
    out_T, out_eps, out_w = [], [], []
    for t in range(1, T + 1):
        z = y * (X @ w)
        if loss == 'exp':
            g = -(X.T @ (y * np.exp(-z))) / m          # l' = -l
        else:  # logistic
            s = 1.0 / (1.0 + np.exp(np.clip(z, -500, 500)))
            g = -(X.T @ (y * s)) / m
        w = w - step * g
        if t % record_every == 0:
            zt = y * (X @ w)
            out_T.append(t)
            out_eps.append(float(loss_of(loss, zt)))
            out_w.append(w.copy())
    return dict(T=np.array(out_T), eps=np.array(out_eps),
                W=np.array(out_w), step=step)


def loss_of(loss, z):
    return exp_loss(z).mean() if loss == 'exp' else log_loss(z).mean()


def gd_final(X, y, loss, step, T):
    """Same as gd_track but keeps only the final iterate (faster)."""
    m, d = X.shape
    w = np.zeros(d)
    for t in range(T):
        z = y * (X @ w)
        if loss == 'exp':
            g = -(X.T @ (y * np.exp(-z))) / m
        else:
            s = 1.0 / (1.0 + np.exp(np.clip(z, -500, 500)))
            g = -(X.T @ (y * s)) / m
        w = w - step * g
    return w


# ----------------------------------------------------------------------
# stagewise AdaBoost with decision stumps (empirical, beyond theory scope)
# ----------------------------------------------------------------------
class StumpAdaBoost:
    """Discrete AdaBoost with exhaustive-search decision stumps."""

    def __init__(self, T):
        self.T = T

    def fit(self, X, y):
        m, d = X.shape
        w = np.full(m, 1.0 / m)
        self.models = []
        self.alphas = []
        self.err_hist = []          # weighted training error of each stump
        F = np.zeros(m)
        self.F_path = np.zeros((self.T + 1, m))
        for t in range(self.T):
            feat, thr, flip, err = best_stump(X, y, w)
            if err >= 0.5 - 1e-8:
                break
            if err <= 1e-8 and t > 0:
                break          # later degenerate rounds would dominate F
            # (at t == 0 a perfect stump is kept: it is the whole ensemble)
            g = flip * np.where(X[:, feat] <= thr, 1.0, -1.0)
            alpha = 0.5 * np.log((1 - err) / err)
            w = w * np.exp(-alpha * y * g)
            w = w / w.sum()
            self.models.append((feat, thr, flip))
            self.alphas.append(alpha)
            self.err_hist.append(err)
            F += alpha * g
            self.F_path[t + 1] = F
        self.n_rounds = len(self.alphas)
        self.F_path = self.F_path[:self.n_rounds + 1]   # trim unused tail
        return self

    def decision_function(self, X):
        F = np.zeros(X.shape[0])
        for (feat, thr, flip), alpha in zip(self.models, self.alphas):
            F += alpha * flip * np.where(X[:, feat] <= thr, 1.0, -1.0)
        return F

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1.0, -1.0)


def best_stump(X, y, w):
    """Exhaustive search for the weighted-error-minimizing decision stump."""
    m, d = X.shape
    best = (0, 0.0, 1.0, 0.5)
    best_err = np.inf
    for j in range(d):
        order = np.argsort(X[:, j])
        xs, ys, ws = X[order, j], y[order], w[order]
        # cumulative weighted error if stump predicts -1 on left, +1 on right
        wpos_left = np.cumsum(ws * (ys > 0))
        wtot_left = np.cumsum(ws)
        wpos_tot = wpos_left[-1]
        wtot = wtot_left[-1]
        # errors of rule "-1 on left, +1 on right" = positives on left + negatives on right
        e1 = wpos_left + (wtot - wtot_left - (wpos_tot - wpos_left))
        # errors of rule "+1 on left, -1 on right" = negatives on left + positives on right
        e2 = (wtot_left - wpos_left) + (wpos_tot - wpos_left)
        # e1 is realized by flip=-1 (g=-1 on left), e2 by flip=+1
        for e, flip in ((e1, -1.0), (e2, 1.0)):
            k = int(np.argmin(e))
            if e[k] < best_err:
                thr = xs[k]
                best_err = e[k]
                best = (j, thr, flip, e[k])
    j, thr, flip, err = best
    return j, thr, flip, min(max(err, 1e-12), 0.5 - 1e-12)


# ----------------------------------------------------------------------
# explicit generalization bound (PAC-Bayes lemma of the revised paper)
# ----------------------------------------------------------------------
def pacbayes_bound(gamma, m, T, delta=0.05, alpha=None, n_alpha=41):
    """Explicit bound of the revised Lemma (eq. thm11-explicit):

        err <= 2 p + 1/m + 32 ln(m) / ((1-alpha)^2 gamma^2 m)
               + 2 ln(2 sqrt(m)/delta) / m ,   p = (gamma^2 T)^{-alpha}.

    Minimized over alpha in (0,1) if alpha is None.  Returns
    (bound, alpha_star); bound is uncapped during optimization.  Valid only
    for gamma^2 T > 1 (else p >= 1).
    """
    if gamma ** 2 * T <= 1:
        return np.inf, np.nan
    p = (gamma ** 2 * T) ** (-alpha if alpha is not None else 0.5)
    if alpha is None:
        alphas = np.linspace(0.01, 0.95, n_alpha)
        vals = [(pacbayes_bound(gamma, m, T, delta, a)[0], a) for a in alphas]
        return min(vals)
    b = (2 * p + 1.0 / m + 32 * np.log(m) / ((1 - alpha) ** 2 * gamma ** 2 * m)
         + 2 * np.log(2 * np.sqrt(m) / delta) / m)
    return float(b), alpha


def ghojogh_bound(edge, theta, T):
    """Ghojogh & Crowley (2019), Eq. (111): with per-round edge xi
    (weighted error <= 1/2 - xi for every round) and margin threshold theta:
        P[y f(x) <= theta] <= ( sqrt((1-2xi)^{1-theta} (1+2xi)^{1+theta}) )^T.
    """
    base = np.sqrt((1 - 2 * edge) ** (1 - theta) * (1 + 2 * edge) ** (1 + theta))
    return float(base ** T)
