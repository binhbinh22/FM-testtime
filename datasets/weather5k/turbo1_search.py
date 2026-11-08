import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

"""
TuRBO-1 (Trust Region Bayesian Optimization, 1 trust region - Eriksson et
al. 2019, "Algorithm 6" trong paper DEFT arXiv:2607.19659) - toi uu 1 ham
score_fn(x) -> scalar (CANG THAP CANG TOT) tren khong gian x in [-1,1]^dim,
voi budget truy van CO DINH (bao gom ca n_init).

Khong gian x la "don vi sigma chuan hoa", KHONG rang buoc gi ve y nghia vat
ly - ben ngoai (kernel WEATHER-5K) se nhan x roi nhan voi sigma THAT cua
tung bien (uoc luong tu do phan tan N candidate Chronos) de ra nhieu thuc,
giong HET cach random_search dang lam - dam bao TuRBO-1 duoc so sanh CONG
BANG voi Random search (cung 1 khong gian edit, chi khac chien luoc tim
kiem: random thuan tuy vs GP-surrogate + trust region).

GHI CHU QUAN TRONG (gioi han co chu dich, khong phai bug):
- GP dung Matern(nu=2.5) ISOTROPIC (1 lengthscale chung, khong ARD) va
  n_restarts_optimizer=0 - vi budget mac dinh=20 << dim (48-144 chieu cho
  P+V tuy prediction_steps), ARD/multi-restart se khong fit noi hoac cham
  vo ich. Day la TuRBO-1 DON GIAN HOA, khong phai ban chuan cua Eriksson.
- Voi budget nho (vd 20), success/failure streak gan nhu khong du de trust
  region co giat/mo rong dang ke trong 15 vong lap con lai sau n_init - ky
  vong hieu nang se GAN VOI Random search hon la mot phuong phap BO day du
  tiem nang (da nguoc vao bang ket qua mock cua du an, TuRBO-1/CEM deu
  TE HON Random search o budget nay - khop du doan ly thuyet).
"""


def turbo1_minimize(score_fn, dim, budget=20, n_init=5, seed=0,
                     length_init=0.8, length_min=0.05, length_max=1.6,
                     success_tol=3, n_candidates=500):
    rng = np.random.default_rng(seed)
    n_init = min(n_init, budget)

    X = rng.uniform(-1.0, 1.0, size=(n_init, dim))
    y = np.array([score_fn(X[i]) for i in range(n_init)], dtype=np.float64)

    best_idx = int(np.argmin(y))
    center = X[best_idx].copy()
    best_y = float(y[best_idx])

    length = length_init
    fail_tol = max(1, dim)  # budget nho -> gan nhu khong bao gio cham nguong nay
    n_success, n_fail = 0, 0

    for _ in range(max(0, budget - n_init)):
        kernel = (Matern(nu=2.5, length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
                  + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e0)))
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=0)
        gp.fit(X, y)

        lo = np.clip(center - length / 2.0, -1.0, 1.0)
        hi = np.clip(center + length / 2.0, -1.0, 1.0)
        candidates = rng.uniform(lo, hi, size=(n_candidates, dim))

        mu, std = gp.predict(candidates, return_std=True)
        thompson = mu + std * rng.standard_normal(size=mu.shape)
        x_next = candidates[int(np.argmin(thompson))]

        y_next = float(score_fn(x_next))
        X = np.vstack([X, x_next[None, :]])
        y = np.append(y, y_next)

        if y_next < best_y - 1e-12:
            best_y = y_next
            center = x_next.copy()
            n_success += 1
            n_fail = 0
        else:
            n_fail += 1
            n_success = 0

        if n_success >= success_tol:
            length = min(length_max, 2.0 * length)
            n_success = 0
        elif n_fail >= fail_tol:
            length = max(length_min, length / 2.0)
            n_fail = 0

    best_idx = int(np.argmin(y))
    return X[best_idx], float(y[best_idx]), {"X": X, "y": y}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    dim = 10
    target = rng.uniform(-0.5, 0.5, size=dim)

    def score_fn(x):
        return float(np.sum((x - target) ** 2))

    x_best, y_best, _ = turbo1_minimize(score_fn, dim=dim, budget=20, seed=1)
    print("target :", np.round(target, 3))
    print("x_best :", np.round(x_best, 3))
    print("y_best :", y_best)
