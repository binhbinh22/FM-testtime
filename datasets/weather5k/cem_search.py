import numpy as np

"""
Direct CEM (Cross-Entropy Method, khong surrogate - khac "Surrogate CEM"
trong paper DEFT arXiv:2607.19659) toi uu ham score_fn(x) -> scalar (CANG
THAP CANG TOT) tren khong gian x in [-1,1]^dim, voi budget truy van CO DINH.

Cung khong gian "don vi sigma chuan hoa" nhu turbo1_search.py, de so sanh
cong bang voi Random search / TuRBO-1 (xem ghi chu trong turbo1_search.py).

Vong lap chuan: sample population tu N(mean, std) (mean=0, std=init_std
luc dau), cham diem TRUC TIEP bang score_fn, giu elite (ty le elite_frac
tot nhat), refit mean/std tu elite, lap lai cho den het budget.
"""


def cem_minimize(score_fn, dim, budget=20, population=5, elite_frac=0.4, seed=0,
                  init_std=0.5):
    rng = np.random.default_rng(seed)
    mean = np.zeros(dim)
    std = np.full(dim, init_std)

    n_elite = max(1, int(np.ceil(population * elite_frac)))
    n_iters = max(1, budget // population)

    all_X, all_y = [], []
    for _ in range(n_iters):
        X = rng.normal(mean, std, size=(population, dim))
        X = np.clip(X, -1.0, 1.0)
        y = np.array([score_fn(X[i]) for i in range(population)], dtype=np.float64)
        all_X.append(X)
        all_y.append(y)

        elite_idx = np.argsort(y)[:n_elite]
        elite = X[elite_idx]
        mean = elite.mean(axis=0)
        std = elite.std(axis=0) + 1e-6

    all_X = np.vstack(all_X)
    all_y = np.concatenate(all_y)
    best_idx = int(np.argmin(all_y))
    return all_X[best_idx], float(all_y[best_idx]), {"X": all_X, "y": all_y}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    dim = 10
    target = rng.uniform(-0.5, 0.5, size=dim)

    def score_fn(x):
        return float(np.sum((x - target) ** 2))

    x_best, y_best, _ = cem_minimize(score_fn, dim=dim, budget=20, seed=1)
    print("target :", np.round(target, 3))
    print("x_best :", np.round(x_best, 3))
    print("y_best :", y_best)
