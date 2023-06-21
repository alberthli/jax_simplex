import jax.numpy as jnp
from jax import lax

from jax_simplex.pivoting import lex_min_ratio_test, pivoting

"""
Implementation ported into Jax from the QuantEcon package:
https://github.com/QuantEcon/QuantEcon.py

Reference: KC Border 2004, "The Gauss–Jordan and Simplex Algorithms".

Notation
--------
Standard form LP:
    maximize    c @ x
    subject to  A_ineq @ x <= b_ineq
                A_eq @ x == b_eq
                x >= 0

A_ineq, shape=(..., m, n)
A_eq, shape=(..., k, n)
L := m + k (number of basis variables)
"""

FEAS_TOL = 1e-6
MAX_ITER = int(1e6)

# ################# #
# LINPROG + HELPERS #
# ################# #


def linprog(
    c: jnp.ndarray,
    A_ineq: jnp.ndarray,
    b_ineq: jnp.ndarray,
    A_eq: jnp.ndarray,
    b_eq: jnp.ndarray,
) -> jnp.ndarray:
    """Solves a linear program in standard form using the simplex method.

    maximize_x   c @ x
    subject to   A_ineq @ x <= b_ineq
                   A_eq @ x == b_eq
                          x >= 0

    Parameters
    ----------
    c : jnp.ndarray, shape=(n,)
        Cost vector.
    A_ineq : jnp.ndarray, shape=(m, n)
        Inequality constraint matrix.
    b_ineq : jnp.ndarray, shape=(m,)
        Inequality constraint vector.
    A_eq : jnp.ndarray, shape=(k, n)
        Equality constraint matrix.
    b_eq : jnp.ndarray, shape=(k,)
        Equality constraint vector.

    Returns
    -------
    x_opt : jnp.ndarray, shape=(n,)
        Optimal primal solution.
    lambda_opt : jnp.ndarray, shape=(L,)
        Optimal dual solution (inequality before equality multipliers).
    val_opt : jnp.ndarray, shape=(,)
        Optimal value.
    success : jnp.ndarray[bool], shape=(,)
        Whether the solver succeeded in finding an optimal solution.
    """
    # initialize tableau and execute phase 1
    tableau, basis = initialize_tableau(A_ineq, b_ineq, A_eq, b_eq)
    tableau, basis, success = solve_tableau(tableau, basis, jnp.array(False))

    # check whether phase 1 terminates, otherwise execute phase 2
    return lax.cond(
        jnp.logical_or(~success, tableau[-1, -1] > FEAS_TOL),
        _linprog_true_fun,
        _phase2_func,
        tableau,
        basis,
        c,
        b_ineq,
        b_eq,
    )


def _linprog_true_fun(tableau, basis, c, b_ineq, b_eq):
    """Helper function for `linprog`. True function of the conditional."""
    n, m, k = c.shape[0], b_ineq.shape[0], b_eq.shape[0]
    L = m + k
    return jnp.zeros(n), jnp.zeros(L), jnp.array(jnp.inf), jnp.array(False)


def _phase2_func(tableau, basis, c, b_ineq, b_eq):
    """Helper function to execute phase 2."""
    # modify criterion row and execute phase 2
    tableau = set_criterion_row(c, basis, tableau)
    tableau, basis, success = solve_tableau(tableau, basis, jnp.array(True))

    # retrieving solution
    b = jnp.concatenate((b_ineq, b_eq))
    b_signs = b >= 0.0
    x_opt, lambda_opt, val_opt = get_solution(tableau, basis, b_signs, c)
    return x_opt, lambda_opt, val_opt, success


# ######################### #
# FUNCTIONS WITHOUT HELPERS #
# ######################### #


def initialize_tableau(
    A_ineq: jnp.ndarray,
    b_ineq: jnp.ndarray,
    A_eq: jnp.ndarray,
    b_eq: jnp.ndarray,
) -> jnp.ndarray:
    """Initializes the tableau.

    Parameters
    ----------
    A_ineq : jnp.ndarray, shape=(m, n)
        Inequality constraint matrix.
    b_ineq : jnp.ndarray, shape=(m,)
        Inequality constraint vector.
    A_eq : jnp.ndarray, shape=(k, n)
        Equality constraint matrix.
    b_eq : jnp.ndarray, shape=(k,)
        Equality constraint vector.

    Returns
    -------
    tableau : jnp.ndarray, shape=(L + 1, n + m + L + 1)
        The initial tableau.
    basis : jnp.ndarray[int], shape=(L,)
        Indices of the basic variables.
    """
    m, k, n = A_ineq.shape[0], A_eq.shape[0], A_ineq.shape[1]
    L = m + k
    tableau = jnp.zeros(((L + 1, n + m + L + 1)))
    tableau = tableau.at[:m, :n].set(A_ineq)
    tableau = tableau.at[m : m + k, :n].set(A_eq)
    tableau = tableau.at[:L, n:-1].set(0.0)

    tableau = tableau.at[:m, n : m + n].set(jnp.eye(m))
    tableau = tableau.at[:m, -1].set(jnp.abs(b_ineq))
    sub_tableau = tableau[:m, : m + n]
    sub_tableau = jnp.where(
        (b_ineq < 0.0)[..., None],  # (m, 1)
        -sub_tableau,
        sub_tableau,
    )
    tableau = tableau.at[:m, : m + n].set(sub_tableau)
    tableau = tableau.at[:m, m + n : 2 * m + n].set(jnp.eye(m))

    tableau = tableau.at[m : m + k, 2 * m + n : 2 * m + n + k].set(jnp.eye(k))
    tableau = tableau.at[m : m + k, -1].set(jnp.abs(b_eq))
    sub_tableau = tableau[m : m + k, : m + n]
    sub_tableau = jnp.where(
        (b_eq < 0.0)[..., None],  # (k, 1)
        -sub_tableau,
        sub_tableau,
    )
    tableau = tableau.at[m : m + k, : m + n].set(sub_tableau)

    tableau = tableau.at[-1, :].set(0.0)
    tableau = tableau.at[-1, : m + n].set(
        tableau[-1, : m + n] + jnp.sum(tableau[:L, : m + n], axis=0)
    )
    tableau = tableau.at[-1, -1].set(tableau[-1, -1] + jnp.sum(tableau[:L, -1]))

    basis = jnp.arange(m + n, m + n + L)
    return tableau, basis


def set_criterion_row(
    c: jnp.ndarray, basis: jnp.ndarray, tableau: jnp.ndarray
) -> jnp.ndarray:
    """
    Modify the criterion row of the tableau for Phase 2.

    Parameters
    ----------
    c : jnp.ndarray, shape=(n,)
        Cost vector.
    basis : jnp.ndarray[int], shape=(L,)
        Phase 1 basis vectors.
    tableau : jnp.ndarray, shape=(L + 1, n + m + L + 1)
        Tableau from phase 1.

    Returns
    -------
    new_tableau : jnp.ndarray, shape=(L + 1, n + m + L + 1)
        New tableau post-modification.
    """
    n = c.shape[0]
    L = basis.shape[0]

    tableau = tableau.at[-1, :n].set(c)
    tableau = tableau.at[-1, n:].set(0.0)

    inds = basis[:L][..., None]  # multiplier indices
    multipliers = tableau[-1, inds]  # (L, 1)
    subtract_term = jnp.sum(multipliers * tableau[:L, : tableau.shape[1]], axis=0)
    tableau = tableau.at[-1, : tableau.shape[1]].set(
        tableau[-1, : tableau.shape[1]] - subtract_term
    )
    return tableau


def get_solution(
    tableau: jnp.ndarray,
    basis: jnp.ndarray,
    b_signs: jnp.ndarray,
    c: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Computes the optimal primal/dual solutions and values from the optimal tableau.

    Parameters
    ----------
    tableau : jnp.ndarray, shape=(L + 1, n + m + L + 1)
        The optimal tableau.
    basis : jnp.ndarray[int], shape=(L,)
        The optimal basis.
    b_signs : jnp.ndarray[bool], shape=(L,)
        Indicates whether elements of (b_ineq, b_eq) are non-negative.
    c : jnp.ndarray, shape=(n,)
        The cost vector. Only used for the shape n.

    Returns
    -------
    x_opt : jnp.ndarray, shape=(n,)
        Optimal primal solution.
    lambda_opt : jnp.ndarray, shape=(L,)
        Optimal dual solution. Inequality multipliers returned first.
    val_opt : jnp.ndarray, shape=(,)
        The optimal value.
    """
    n = c.shape[0]
    L = basis.shape[0]
    aux_start = tableau.shape[1] - L - 1

    # primal optimum
    x_opt = jnp.zeros_like(c)
    sub_x = jnp.where(
        basis < n,
        tableau[:L, -1],
        jnp.zeros_like(basis),
    )
    x_opt = x_opt.at[basis].set(sub_x)

    # dual optimum
    lambda_opt = tableau[-1, aux_start : aux_start + L]
    lambda_opt = jnp.where(
        jnp.logical_and(lambda_opt != 0, b_signs), -lambda_opt, lambda_opt
    )

    # optimal value
    val_opt = -tableau[-1, -1]

    return x_opt, lambda_opt, val_opt


# ################### #
# PIVOT_COL + HELPERS #
# ################### #


def pivot_col(
    tableau: jnp.ndarray, skip_aux: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Chooses the column containing the pivot element.

    Parameters
    ----------
    tableau : jnp.ndarray, shape=(L + 1, m + n + L + 1)
        The tableau.
    skip_aux : jnp.ndarray[bool], shape=(,)
        Whether to skip the coefficients of the aux variables in pivot col selection.

    Returns
    -------
    found : jnp.ndarray[bool], shape=(,)
        True iff there is a positive element in the last row of the tableau.
    pivcol : jnp.ndarray[int], shape=(,)
        The index of column containing the pivot element. -1 if not found.
    """
    L = tableau.shape[0] - 1
    criterion_row_stop = tableau.shape[1] - 1
    _tableau = tableau.at[-1, criterion_row_stop - L : criterion_row_stop].set(
        tableau[-1, criterion_row_stop - L : criterion_row_stop] * (1 - skip_aux)
    )  # removing auxiliary variables from consideration

    return lax.cond(
        jnp.sum(_tableau[-1, :criterion_row_stop] > FEAS_TOL) > 0,
        _pivot_col_true_fun,
        _pivot_col_false_fun,
        _tableau,
    )


def _pivot_col_true_fun(tableau):
    """Helper function for `pivot_col`. True outcome of conditional."""
    criterion_row_stop = tableau.shape[1] - 1
    return jnp.array(True), jnp.argmax(tableau[-1, :criterion_row_stop])


def _pivot_col_false_fun(tableau):
    """Helper function for `pivot_col`. False outcome of conditional."""
    return jnp.array(False), jnp.array(-1, dtype=int)


# ####################### #
# SOLVE_TABLEAU + HELPERS #
# ####################### #


def solve_tableau(
    tableau: jnp.ndarray, basis: jnp.ndarray, skip_aux: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Performs the simplex algorithm on a given tableau.

    Parameters
    ----------
    tableau : jnp.ndarray, shape=(L + 1, n + m + L + 1)
        The initial tableau.
    basis : jnp.ndarray[int], shape=(L,)
        Indices of the basic variables.
    skip_aux : jnp.ndarray[bool], shape=(,)
        Whether to skip the coefficients of the aux variables in pivot col selection.

    Returns
    -------
    new_tableau : jnp.ndarray, shape=(L + 1, n + m + L + 1)
        The new tableau.
    new_basis : jnp.ndarray, shape=(L,)
        Indices of the updated basic variables.
    success : jnp.ndarray[bool], shape=(,)
        Whether the algorithm succeeded in finding an optimal solution.
    """
    # `val` = (index, tableau, basis, skip_aux, success, terminate)
    init_val = 0, tableau, basis, skip_aux, jnp.array(False), jnp.array(False)

    _, tableau, basis, _, success, _, = lax.while_loop(
        _solve_tableau_cond_fun,
        _solve_tableau_body_fun,
        init_val,
    )
    return tableau, basis, success


def _solve_tableau_cond_fun(val):
    """Helper function for `solve_tableau`. Condition function for while loop."""
    i, _, _, _, _, terminate = val
    return jnp.logical_and(i < MAX_ITER, ~terminate)


def _solve_tableau_body_fun(val):
    """Helper function for `solve_tableau`. Body function for while loop."""
    i, _tableau, _basis, skip_aux, _, _ = val

    L = _tableau.shape[0] - 1
    aux_start = _tableau.shape[1] - L - 1

    pivcol_found, pivcol = pivot_col(_tableau, skip_aux)
    pivrow_found, pivrow = lex_min_ratio_test(_tableau[:-1, :], pivcol, aux_start)

    cases = jnp.array([~pivcol_found, ~pivrow_found, True])
    branches = [
        _solve_tableau_branch1_fun,
        _solve_tableau_branch2_fun,
        _solve_tableau_branch3_fun,
    ]
    return lax.switch(
        jnp.argmax(cases),
        branches,
        i,
        _tableau,
        _basis,
        skip_aux,
        pivcol,
        pivrow,
    )


def _solve_tableau_branch1_fun(i, _tableau, _basis, skip_aux, pivcol, pivrow):
    """Helper function for `_solve_tableau`. Branch 1 function."""
    return i + 1, _tableau, _basis, skip_aux, jnp.array(True), jnp.array(True)


def _solve_tableau_branch2_fun(i, _tableau, _basis, skip_aux, pivcol, pivrow):
    """Helper function for `_solve_tableau`. Branch 2 function."""
    return i + 1, _tableau, _basis, skip_aux, jnp.array(False), jnp.array(True)


def _solve_tableau_branch3_fun(i, _tableau, _basis, skip_aux, pivcol, pivrow):
    """Helper function for `_solve_tableau`. Branch 3 function."""
    return (
        i + 1,
        pivoting(_tableau, pivcol, pivrow),
        _basis.at[pivrow].set(pivcol),
        skip_aux,
        jnp.array(False),
        jnp.array(False),
    )


# if __name__ == "__main__":
#     # import numpy as np
#     # # np.random.seed(0)

#     # m = 5
#     # n = 4
#     # k = 3
#     # A_ineq = np.random.rand(m, n)
#     # b_ineq = np.random.rand(m)
#     # A_eq = np.random.rand(k, n)
#     # b_eq = np.random.rand(k)
#     # c = np.random.rand(n)

#     # # transforming problem into canonical form
#     # n = 2 * n
#     # A_ineq = np.concatenate((A_ineq, -A_ineq), axis=-1)
#     # A_eq = np.concatenate((A_eq, -A_eq), axis=-1)
#     # c = np.concatenate((c, -c), axis=-1)

#     # # testing linprog solve
#     # from quantecon.optimize import linprog_simplex
#     # from jax import jit
#     # res = linprog_simplex(c, A_ineq, b_ineq, A_eq, b_eq)
#     # compiled_linprog = jit(linprog)

#     # x_opt, lambda_opt, val_opt, success = compiled_linprog(
#     #     jnp.array(c),
#     #     jnp.array(A_ineq),
#     #     jnp.array(b_ineq),
#     #     jnp.array(A_eq),
#     #     jnp.array(b_eq),
#     # )

#     # print("COMPARING TO QUANTECON")
#     # print(f"QE status: {res.status}")
#     # if res.status == 0 and success:
#     #     print(res.x - x_opt)
#     #     print(res.lambd - lambda_opt)
#     #     print(res.fun - val_opt)
#     # print()

#     # from scipy.optimize import linprog as linprog_scipy
#     # res_scipy = linprog_scipy(
#     #     -c, A_ub=A_ineq, b_ub=b_ineq, A_eq=A_eq, b_eq=b_eq, method="highs-ds"
#     # )

#     # print("COMPARING TO SCIPY")
#     # print(f"scipy status: {res_scipy.status}")
#     # if res_scipy.success and success:
#     #     print(res_scipy.x - x_opt)
#     #     lambda_opt_scipy = -np.concatenate(
#     #         (res_scipy.ineqlin.marginals, res_scipy.eqlin.marginals)
#     #     )
#     #     print(lambda_opt_scipy - lambda_opt)
#     #     print(-res_scipy.fun - val_opt)
#     # print()

#     # import cvxpy as cp
#     # x = cp.Variable(n)
#     # obj = cp.Maximize(c @ x)
#     # constraints = [x >= 0]
#     # if m > 0:
#     #     constraints += [A_ineq @ x <= b_ineq]
#     # if k > 0:
#     #     constraints += [A_eq @ x == b_eq]
#     # prob = cp.Problem(obj, constraints)
#     # prob.solve(solver=cp.GUROBI)

#     # print("COMPARING TO CVXPY")
#     # if prob.status == "optimal" and success:
#     #     _lambda_opt = np.concatenate(
#     #         (constraints[1].dual_value, constraints[2].dual_value)
#     #     )
#     #     print(x.value - x_opt)
#     #     print(_lambda_opt - lambda_opt)
#     #     print(prob.value - val_opt)
#     # print()

#     # breakpoint()

#     # TIMING #
#     import numpy as np
#     from quantecon.optimize import linprog_simplex
#     from scipy.optimize import linprog as linprog_scipy
#     import time
#     from jax import jit, vmap
#     B = 16
#     m = 14
#     n = 32
#     k = 0
#     A_ineq = np.random.rand(B, m, n)
#     b_ineq = np.random.rand(B, m)
#     A_eq = np.random.rand(B, k, n)
#     b_eq = np.random.rand(B, k)
#     c = np.random.rand(B, n)

#     # transforming problem into canonical form
#     n = 2 * n
#     A_ineq = np.concatenate((A_ineq, -A_ineq), axis=-1)
#     A_eq = np.concatenate((A_eq, -A_eq), axis=-1)
#     c = np.concatenate((c, -c), axis=-1)

#     # pre-compiling
#     c_jax = jnp.array(c)
#     A_ineq_jax = jnp.array(A_ineq)
#     b_ineq_jax = jnp.array(b_ineq)
#     A_eq_jax = jnp.array(A_eq)
#     b_eq_jax = jnp.array(b_eq)
#     batch_linprog = jit(vmap(linprog))
#     x_opt, lambda_opt, val_opt, success = batch_linprog(
#         c_jax,
#         A_ineq_jax,
#         b_ineq_jax,
#         A_eq_jax,
#         b_eq_jax,
#     )
#     x_opt.block_until_ready()
#     lambda_opt.block_until_ready()
#     val_opt.block_until_ready()
#     success.block_until_ready()

#     # generating data one more time
#     B = 16
#     m = 14
#     n = 32
#     k = 0
#     A_ineq = np.random.rand(B, m, n)
#     b_ineq = np.random.rand(B, m)
#     A_eq = np.random.rand(B, k, n)
#     b_eq = np.random.rand(B, k)
#     c = np.random.rand(B, n)

#     # transforming problem into canonical form
#     n = 2 * n
#     A_ineq = np.concatenate((A_ineq, -A_ineq), axis=-1)
#     A_eq = np.concatenate((A_eq, -A_eq), axis=-1)
#     c = np.concatenate((c, -c), axis=-1)

#     # serial
#     # start = time.time()
#     # for i in range(B):
#     #     x_opt, lambda_opt, val_opt, success = compiled_linprog(
#     #         jnp.array(c[i]),
#     #         jnp.array(A_ineq[i]),
#     #         jnp.array(b_ineq[i]),
#     #         jnp.array(A_eq[i]),
#     #         jnp.array(b_eq[i]),
#     #     )
#     # end = time.time()
#     # print(f"Serial My Impl: {(end - start) / B}")

#     # batched
#     num_reps = 100
#     start = time.time()
#     for j in range(num_reps):
#         x_opt, lambda_opt, val_opt, success = batch_linprog(
#             c_jax,
#             A_ineq_jax,
#             b_ineq_jax,
#             A_eq_jax,
#             b_eq_jax,
#         )
#         x_opt.block_until_ready()
#         lambda_opt.block_until_ready()
#         val_opt.block_until_ready()
#         success.block_until_ready()
#     end = time.time()
#     print(f"Batched My Impl: {(end - start) / (B * num_reps)}")

#     start = time.time()
#     qe_successes = 0
#     for i in range(B):
#         res = linprog_simplex(c[i], A_ineq[i], b_ineq[i], A_eq[i], b_eq[i])
#         if res.success:
#             qe_successes += 1
#     end = time.time()
#     print(f"Serial quantecon: {(end - start) / B}")

#     start = time.time()
#     for i in range(B):
#         linprog_scipy(
#             -c[i],
#             A_ub=A_ineq[i],
#             b_ub=b_ineq[i],
#             A_eq=A_eq[i],
#             b_eq=b_eq[i],
#             method="highs-ds",
#         )
#     end = time.time()
#     print(f"Serial scipy: {(end - start) / B}")

#     breakpoint()
