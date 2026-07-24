import numpy as np
from scipy.optimize import minimize
from smo import sequential_minimal_optimization


def make_random_tensor_cost(n_params, seed=0):
    """
    Build a cost function that is EXACTLY of the form assumed by Eq. (12):
    L(theta) = b . bigotimes_j (cos theta_j, sin theta_j, 1)
    for a random real coefficient tensor b of shape (3,)*n_params.
    This is exactly what you'd get from a real quantum circuit satisfying
    the paper's preconditions, so it's a fair test of the optimizer.
    """
    rng = np.random.default_rng(seed)
    b = rng.normal(size=(3,) * n_params)

    def cost(theta):
        vecs = [np.array([np.cos(t), np.sin(t), 1.0]) for t in theta]
        val = b
        for v in vecs:
            val = np.tensordot(val, v, axes=([0], [0]))
        return float(val)

    return cost, b


def brute_force_min(cost, n_params, grid=200):
    grids = [np.linspace(-np.pi, np.pi, grid, endpoint=False) for _ in range(n_params)]
    best = np.inf
    if n_params == 1:
        for t in grids[0]:
            v = cost([t])
            best = min(best, v)
    elif n_params == 2:
        for t1 in grids[0]:
            for t2 in grids[1]:
                v = cost([t1, t2])
                best = min(best, v)
    else:
        raise NotImplementedError
    return best


print("=" * 70)
print("Test 1: 4 parameters, single-parameter updates (subset_size=1)")
print("=" * 70)
cost4, b4 = make_random_tensor_cost(4, seed=1)
x0 = np.random.default_rng(1).uniform(-np.pi, np.pi, size=4)

res = minimize(
    cost4, x0,
    method=sequential_minimal_optimization,
    options={"subset_size": 1, "maxiter": 30, "tol": 1e-12, "disp": False},
)
print("final x:", res.x)
print("final f:", res.fun, " nfev:", res.nfev, " nit:", res.nit, " success:", res.success)

print()
print("=" * 70)
print("Test 2: 2 parameters, JOINT update (subset_size=2) -- should")
print("converge to the exact minimum in a SINGLE sweep (9 evals + 1 check)")
print("=" * 70)
cost2, b2 = make_random_tensor_cost(2, seed=2)
x0_2 = np.array([1.234, -2.1])

res2 = minimize(
    cost2, x0_2,
    method=sequential_minimal_optimization,
    options={"subset_size": 2, "maxiter": 5, "tol": 1e-12, "disp": True},
)
print("final x:", res2.x)
print("final f:", res2.fun, " nfev:", res2.nfev, " nit:", res2.nit)

brute_min = brute_force_min(cost2, 2, grid=300)
print("brute-force reference min (300x300 grid):", brute_min)
print("match within tolerance:", abs(res2.fun - brute_min) < 1e-2)

print()
print("=" * 70)
print("Test 3: sanity check vs Nelder-Mead on the same 4-param problem")
print("=" * 70)
res_nm = minimize(cost4, x0, method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-10, "fatol": 1e-10})
print("Nelder-Mead   f =", res_nm.fun, " nfev =", res_nm.nfev)
print("SMO (size=1)  f =", res.fun, " nfev =", res.nfev)