"""
Generalized Sequential Minimal Optimization (SMO) for quantum-classical
hybrid algorithms, following:

    K. M. Nakanishi, K. Fujii, S. Todo,
    "Sequential minimal optimization for quantum-classical hybrid algorithms",
    arXiv:1903.12166.

This implements the *generalized* multi-parameter update of Sec. II C / Eq. (12)
of the paper: for any subset M of parameters, if the cost function restricted
to those parameters (holding all others fixed) has the exact form

    L_M({theta_j}_{j in M}) = b_M . bigotimes_{j in M} (cos(theta_j), sin(theta_j), 1)

(which holds whenever every parameter enters the circuit only through rotation
gates R_j(theta_j) = exp(-i theta_j/2 A_j) with A_j^2 = I, per the paper's
preconditions), then the 3^|M| unknown coefficients in b_M can be reconstructed
exactly from 3^|M| function evaluations on a regular grid via an |M|-dimensional
3-point discrete Fourier transform, and the resulting trig polynomial can be
minimized in closed form / cheaply, with NO further calls to the (expensive)
cost function.

Choosing |M| = 1 for every subset recovers the original single-parameter method
of Sec. II B (a.k.a. "NFT" / "Rotosolve"). Choosing |M| > 1 lets you jointly
update parameters that interact non-trivially (e.g. two angles entering through
commuting generators, as in a Givens/hopping rotation for free fermions) in one
shot instead of needing repeated single-parameter sweeps to converge.

Usage
-----
This is written as a scipy.optimize.minimize *custom method* (scipy supports
passing a callable as `method`), so it is invoked with the exact same
interface as any other scipy optimizer:

    from scipy.optimize import minimize
    from smo import sequential_minimal_optimization

    result = minimize(
        cost_fn, x0,
        method=sequential_minimal_optimization,
        options={"subset_size": 2, "maxiter": 50},
    )

`result` is a standard `scipy.optimize.OptimizeResult`.

Caveats
-------
- The cost function must actually have the periodic, per-parameter-sinusoidal
  structure Eq. (12) assumes. This holds for parameterized quantum circuits
  built from Pauli-rotation-type gates (R_x, R_y, R_z, or any exp(-i theta/2 A)
  with A^2=I) and expectation values of Hermitian observables; it generally
  does NOT hold for arbitrary classical objective functions.
- Box `bounds` and `constraints` are not supported (the method relies on
  exact 2*pi-periodicity of each parameter); a ValueError is raised if given.
- `jac`, `hess`, `hessp` are ignored (the method is gradient-free by
  construction) with a warning if a jac is supplied.
"""

from __future__ import annotations

import itertools
import warnings
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import OptimizeResult, minimize as _scipy_minimize


def _subsets_from_size(n_params: int, subset_size: int, order: Sequence[int] | None) -> list[tuple[int, ...]]:
    """Partition parameter indices into consecutive chunks of size `subset_size`."""
    idx = list(range(n_params)) if order is None else list(order)
    return [tuple(idx[i : i + subset_size]) for i in range(0, n_params, subset_size)]


def _reconstruct_and_minimize(
    fun: Callable,
    x: np.ndarray,
    subset: tuple[int, ...],
    args: tuple,
    n_restarts: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """
    For the given parameter subset, evaluate `fun` on the 3^|M| grid needed to
    reconstruct the exact trig-polynomial dependence (Eq. 12), then find the
    displacement `delta` (added to x[subset]) that minimizes it.

    Returns (delta, n_new_function_evals).
    """
    m = len(subset)
    grid_shape = (3,) * m
    values = np.empty(grid_shape, dtype=float)

    base = x[list(subset)].copy()
    offsets = (0.0, 2 * np.pi / 3, 4 * np.pi / 3)

    for k in itertools.product(range(3), repeat=m):
        x_trial = x.copy()
        for pos, j in enumerate(subset):
            x_trial[j] = base[pos] + offsets[k[pos]]
        values[k] = fun(x_trial, *args)

    n_new_evals = 3 ** m

    # m-dimensional 3-point DFT. numpy's fftn bin ordering (0, 1, ..., N-1) maps
    # to integer frequencies (0, 1, ..., -1) for N=3, i.e. bin 2 <-> freq -1,
    # exactly the {-1, 0, 1} frequency set the paper's ansatz requires.
    coeffs = np.fft.fftn(values) / (3 ** m)  # complex tensor, shape (3,)*m

    freqs = np.array([0, 1, -1])  # frequency corresponding to bin index 0,1,2

    def reconstructed(delta: np.ndarray) -> float:
        # delta: length-m real vector (displacement from `base`)
        total = 0.0 + 0.0j
        for k in itertools.product(range(3), repeat=m):
            f = freqs[list(k)]
            total += coeffs[k] * np.exp(1j * np.dot(f, delta))
        return float(np.real(total))

    # Minimize the cheap closed-form trig polynomial (no new fun() calls).
    if m == 1:
        # Closed form: reconstructed(d) = c0 + 2|c1| cos(d - phase(c1))  [real]
        c1 = coeffs[(1,)]
        # reconstructed(d) = c0 + 2*Re(c1 * exp(i d)) = c0 + 2|c1|*cos(d + angle(c1)),
        # minimized when d + angle(c1) = pi, i.e. d = pi - angle(c1).
        phase = np.angle(c1)
        best_delta = np.array([np.pi - phase])
    else:
        best_val = np.inf
        best_delta = np.zeros(m)
        starts = [np.zeros(m)] + [rng.uniform(-np.pi, np.pi, size=m) for _ in range(n_restarts)]
        for x0 in starts:
            res = _scipy_minimize(reconstructed, x0, method="L-BFGS-B")
            if res.fun < best_val:
                best_val = res.fun
                best_delta = res.x

    return best_delta, n_new_evals


def sequential_minimal_optimization(
    fun: Callable,
    x0: np.ndarray,
    args: tuple = (),
    jac=None,
    hess=None,
    hessp=None,
    bounds=None,
    constraints=(),
    tol: float | None = None,
    callback: Callable | None = None,
    options: dict | None = None,
    **unused,
) -> OptimizeResult:
    """
    Generalized sequential minimal optimizer (Nakanishi-Fujii-Todo, 2019),
    compatible with the `method=` callable interface of
    `scipy.optimize.minimize`.

    Parameters
    ----------
    fun, x0, args, jac, hess, hessp, bounds, constraints, tol, callback, options
        Same meaning as in `scipy.optimize.minimize`. `jac`, `hess`, `hessp`
        are ignored (method is analytic/gradient-free). `bounds` and
        `constraints` are not supported.

    options : dict, optional
        - "subset_size" : int, default 1
              Size |M| of each jointly-updated parameter group. 1 reproduces
              the original single-parameter method (Sec. II B). Larger values
              exploit joint sinusoidal structure (Sec. II C, Eq. 12) between
              parameters that interact through commuting/related generators.
        - "subsets" : list of tuples of int, optional
              Explicit parameter groups to update each sweep, overriding
              `subset_size` (groups need not partition all indices, and can
              repeat indices across groups).
        - "order" : sequence of int, optional
              Permutation of parameter indices used to build default
              (chunked) subsets when "subsets" is not given.
        - "maxiter" : int, default 100
              Maximum number of full sweeps over all subsets.
        - "tol" : float, default 1e-8
              Sweep is considered converged when the true cost function
              changes by less than this between successive sweeps.
        - "n_restarts" : int, default 4
              Random restarts used for minimizing the (cheap, closed-form)
              reconstructed trig polynomial when |M| > 1.
        - "seed" : int, optional
              RNG seed for restarts.
        - "disp" : bool, default False
              Print progress each sweep.

    Returns
    -------
    scipy.optimize.OptimizeResult
        Standard fields: x, fun, nit, nfev, success, message.
    """
    if bounds is not None:
        raise ValueError("sequential_minimal_optimization does not support `bounds` "
                          "(parameters are assumed exactly 2*pi-periodic).")
    if constraints:
        raise ValueError("sequential_minimal_optimization does not support `constraints`.")
    if jac is not None:
        warnings.warn("`jac` is ignored: this method reconstructs the cost function's "
                       "exact trigonometric dependence and does not use gradients.")

    options = dict(options or {})
    subset_size = int(options.get("subset_size", 1))
    explicit_subsets = options.get("subsets")
    order = options.get("order")
    maxiter = int(options.get("maxiter", 100))
    tol_ = float(tol) if tol is not None else float(options.get("tol", 1e-8))
    n_restarts = int(options.get("n_restarts", 4))
    seed = options.get("seed")
    disp = bool(options.get("disp", False))

    print(f"Igot:{disp} and {subset_size} and {options}")

    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=float).copy()
    n_params = x.size

    subsets = explicit_subsets if explicit_subsets is not None else _subsets_from_size(
        n_params, subset_size, order
    )

    nfev = 0
    f_prev = float(fun(x, *args))
    nfev += 1

    success = False
    message = "Maximum number of sweeps reached."
    nit = 0

    for it in range(1, maxiter + 1):
        for subset in subsets:
            delta, n_new = _reconstruct_and_minimize(fun, x, subset, args, n_restarts, rng)
            nfev += n_new
            for pos, j in enumerate(subset):
                x[j] = x[j] + delta[pos]

        f_curr = float(fun(x, *args))
        nfev += 1
        nit = it

        if callback is not None:
            callback(np.copy(x))

        if disp:
            print(f"sweep {it:4d}  f = {f_curr:.10g}  (Delta f = {f_prev - f_curr:.3g})")

        if abs(f_prev - f_curr) < tol_:
            success = True
            message = "Converged: change in cost function below tol."
            f_prev = f_curr
            break

        f_prev = f_curr

    return OptimizeResult(
        x=x,
        fun=f_prev,
        nit=nit,
        nfev=nfev,
        success=success,
        message=message,
    )