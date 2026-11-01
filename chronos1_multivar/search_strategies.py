import numpy as np


# ============================================================
# Cac chien luoc SEARCH test-time (baseline tu DEFT) - THUAN TUY tren 1
# objective den (minimize), KHONG phu thuoc backbone/physics, de UNIT-TEST
# duoc bang ham synthetic truoc khi cam vao pipeline that.
#
# QUY UOC:
#   - objective(theta: np.ndarray[dim]) -> float, CANG NHO CANG TOT (energy).
#     (physics/composite verifier E cua ta la energy, minimize.)
#   - budget = TONG so lan goi objective (so query scorer) - de so cong bang
#     giua cac chien luoc dung dung tinh than DEFT (match query budget).
#   - bounds = (lb, ub), moi cai shape (dim,).
#   - rng = np.random.Generator (seed co dinh de tai lap).
#
# Tat ca tra ve dict: {best_theta, best_val, n_queries, history(list val)}.
#
# Khong gian search o pipeline that: per-cluster offset coefficients (output
# -space cap cum, bo CDF) - xem search_baselines.py (buoc tich hop sau).
# ============================================================


class _BudgetedObjective:
    """Boc objective de DEM so lan goi va CHAN khi vuot budget."""

    def __init__(self, fn, budget):
        self.fn = fn
        self.budget = int(budget)
        self.n = 0
        self.best_val = np.inf
        self.best_theta = None
        self.history = []

    def __call__(self, theta):
        if self.n >= self.budget:
            raise StopIteration("budget exhausted")
        val = float(self.fn(np.asarray(theta, dtype=np.float64)))
        self.n += 1
        self.history.append(val)
        if val < self.best_val:
            self.best_val = val
            self.best_theta = np.array(theta, dtype=np.float64)
        return val

    def result(self):
        return {
            "best_theta": self.best_theta,
            "best_val": self.best_val,
            "n_queries": self.n,
            "history": self.history,
        }


def _clip(x, lb, ub):
    return np.minimum(np.maximum(x, lb), ub)


# ------------------------------------------------------------
# 1) Random search
# ------------------------------------------------------------
def random_search(objective, dim, bounds, budget, rng):
    lb, ub = bounds
    obj = _BudgetedObjective(objective, budget)
    try:
        for _ in range(budget):
            theta = rng.uniform(lb, ub)
            obj(theta)
    except StopIteration:
        pass
    return obj.result()


# ------------------------------------------------------------
# 4) Direct CEM (Cross-Entropy Method) - de Boer et al., 2005
# ------------------------------------------------------------
def cem(objective, dim, bounds, budget, rng, pop=None, elite_frac=0.25, init_sigma_frac=0.25):
    lb, ub = bounds
    pop = int(pop or max(5, budget // 4))
    n_elite = max(1, int(round(pop * elite_frac)))

    mu = (lb + ub) / 2.0
    sigma = (ub - lb) * init_sigma_frac

    obj = _BudgetedObjective(objective, budget)
    try:
        while obj.n < budget:
            remaining = budget - obj.n
            cur_pop = min(pop, remaining)
            samples = rng.normal(mu, np.maximum(sigma, 1e-9), size=(cur_pop, dim))
            samples = _clip(samples, lb, ub)
            vals = np.array([obj(s) for s in samples])
            elite_idx = np.argsort(vals)[:n_elite]
            elites = samples[elite_idx]
            mu = elites.mean(axis=0)
            sigma = elites.std(axis=0) + 1e-6  # tranh suy bien ve 0
    except StopIteration:
        pass
    return obj.result()


# ------------------------------------------------------------
# 6) Surrogate CEM - CEM nhung cham so bo bang SURROGATE re (GP/ridge),
#    chi ton query scorer THAT cho vai ung vien tot nhat moi vong.
# ------------------------------------------------------------
def surrogate_cem(objective, dim, bounds, budget, rng, pop_virtual=256, real_per_iter=4,
                   elite_frac=0.25, init_sigma_frac=0.25, n_warmup=None):
    lb, ub = bounds
    n_warmup = int(n_warmup or max(4, budget // 3))
    n_warmup = min(n_warmup, budget)

    obj = _BudgetedObjective(objective, budget)
    X, y = [], []

    # warm-up: lay mau ngau nhien de co du lieu train surrogate
    try:
        for _ in range(n_warmup):
            theta = rng.uniform(lb, ub)
            v = obj(theta)
            X.append(theta); y.append(v)
    except StopIteration:
        return obj.result()

    def fit_surrogate(X, y):
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
            k = ConstantKernel(1.0) * RBF(length_scale=np.ones(dim)) + WhiteKernel(1e-3)
            m = GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=0)
            m.fit(np.array(X), np.array(y))
            return lambda Z: m.predict(Z)
        except Exception:  # noqa: BLE001  - fallback: ridge tren RBF features ngau nhien
            Xa = np.array(X); ya = np.array(y)
            centers = Xa[rng.choice(len(Xa), size=min(len(Xa), 64), replace=False)]
            gamma = 1.0 / (dim * (Xa.std() + 1e-6) ** 2)

            def feat(Z):
                d2 = ((Z[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
                return np.exp(-gamma * d2)
            F = feat(Xa)
            w = np.linalg.lstsq(F.T @ F + 1e-3 * np.eye(F.shape[1]), F.T @ ya, rcond=None)[0]
            return lambda Z: feat(Z) @ w

    mu = np.array(X)[np.argmin(y)].copy()
    sigma = (ub - lb) * init_sigma_frac
    n_elite = max(1, int(round(pop_virtual * elite_frac)))

    try:
        while obj.n < budget:
            surrogate = fit_surrogate(X, y)
            # sinh population ao LON, cham bang surrogate (re)
            virt = _clip(rng.normal(mu, np.maximum(sigma, 1e-9), size=(pop_virtual, dim)), lb, ub)
            pred = np.asarray(surrogate(virt))
            order = np.argsort(pred)
            # chi EVAL THAT vai ung vien surrogate tot nhat
            k_real = min(real_per_iter, budget - obj.n)
            chosen = virt[order[:k_real]]
            for theta in chosen:
                v = obj(theta)
                X.append(theta); y.append(v)
            # cap nhat phan phoi CEM tu top-elite theo SURROGATE (dung pop ao)
            elites = virt[order[:n_elite]]
            mu = elites.mean(axis=0)
            sigma = elites.std(axis=0) + 1e-6
    except StopIteration:
        pass
    return obj.result()


# ------------------------------------------------------------
# 5) TuRBO-1 (Trust Region BO, 1 trust region) - Eriksson et al., NeurIPS 2019
#    Dung botorch/gpytorch (user da chon ban chuan). Import TRE (lazy).
# ------------------------------------------------------------
def turbo1(objective, dim, bounds, budget, rng, n_init=None):
    import torch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood

    lb, ub = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    span = np.maximum(ub - lb, 1e-12)
    n_init = int(n_init or max(4, min(2 * dim, budget // 2)))
    n_init = min(n_init, budget)
    seed = int(rng.integers(0, 2**31 - 1))
    torch.manual_seed(seed)

    def to_unit(x):
        return (x - lb) / span

    def from_unit(z):
        return lb + z * span

    obj = _BudgetedObjective(objective, budget)

    # ---- khoi tao Latin-ish bang uniform ----
    Z, Y = [], []
    try:
        for _ in range(n_init):
            z = rng.uniform(0, 1, size=dim)
            v = obj(from_unit(z))
            Z.append(z); Y.append(v)
    except StopIteration:
        return obj.result()

    Z = np.array(Z); Y = np.array(Y)

    # ---- tham so trust region (theo paper) ----
    L = 0.8               # do dai TR ban dau (trong [0,1] space)
    L_MIN, L_MAX = 0.5 ** 7, 1.6
    succ, fail = 0, 0
    SUCC_TOL, FAIL_TOL = 3, max(4, dim)

    try:
        while obj.n < budget:
            # chuan hoa Y de GP on dinh
            Yt = (Y - Y.mean()) / (Y.std() + 1e-9)
            train_x = torch.tensor(Z, dtype=torch.double)
            train_y = torch.tensor(Yt, dtype=torch.double).unsqueeze(-1)
            gp = SingleTaskGP(train_x, train_y)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            try:
                fit_gpytorch_mll(mll)
            except Exception:  # noqa: BLE001
                pass

            # tam TR tai diem tot nhat; dung lengthscale GP de co dan theo chieu
            x_center = Z[np.argmin(Y)].copy()
            try:
                ls = gp.covar_module.base_kernel.lengthscale.detach().numpy().ravel()
                ls = ls / ls.mean()
            except Exception:  # noqa: BLE001
                ls = np.ones(dim)
            tr_lb = np.clip(x_center - L * ls / 2.0, 0.0, 1.0)
            tr_ub = np.clip(x_center + L * ls / 2.0, 0.0, 1.0)

            # Thompson sampling: boc nhieu candidate trong TR, chon theo posterior sample
            n_cand = min(2000, 100 * dim)
            cand = rng.uniform(tr_lb, tr_ub, size=(n_cand, dim))
            with torch.no_grad():
                post = gp.posterior(torch.tensor(cand, dtype=torch.double))
                ts = post.rsample().squeeze(-1).numpy().ravel()  # 1 mau posterior
            z_next = cand[int(np.argmin(ts))]  # minimize (Yt da cung chieu voi Y)

            v = obj(from_unit(z_next))
            improved = v < Y.min() - 1e-3 * abs(Y.min())
            Z = np.vstack([Z, z_next]); Y = np.append(Y, v)

            if improved:
                succ += 1; fail = 0
            else:
                fail += 1; succ = 0
            if succ >= SUCC_TOL:
                L = min(L_MAX, 2.0 * L); succ = 0
            if fail >= FAIL_TOL:
                L = max(L_MIN, L / 2.0); fail = 0
            if L < L_MIN:  # TR qua nho -> coi nhu hoi tu, dung
                break
    except StopIteration:
        pass
    return obj.result()


# ------------------------------------------------------------
# 2)/3) Best-of-N SELECT (sample / quantile) - khong toi uu, chi chon trong
#       tap candidate cho truoc. candidates: list theta; score_fn: energy.
# ------------------------------------------------------------
def best_of_n_select(candidate_scores):
    """candidate_scores: array energy (nho = tot). Tra ve index argmin + val."""
    scores = np.asarray(candidate_scores, dtype=np.float64)
    i = int(np.argmin(scores))
    return {"best_index": i, "best_val": float(scores[i]), "n_queries": len(scores),
            "history": scores.tolist()}


SEARCH_REGISTRY = {
    "random_search": random_search,
    "cem": cem,
    "surrogate_cem": surrogate_cem,
    "turbo1": turbo1,
}
