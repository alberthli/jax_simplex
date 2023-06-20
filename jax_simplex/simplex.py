import jax.numpy as jnp
from jax import lax

from jax_simplex.pivoting import _lex_min_ratio_test, _pivoting

FEAS_TOL = 1e-6
MAX_ITER = int(1e6)


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
    fun : jnp.ndarray, shape=(,)
        Optimal value.
    success : jnp.ndarray[bool], shape=(,)
        Whether the solver succeeded in finding an optimal solution.
    """
    n, m, k = c.shape[0], A_ineq.shape[0], A_eq.shape[0]
    L = m + k

    # initialize tableau and execute phase 1
    tableau, basis = _initialize_tableau(A_ineq, b_ineq, A_eq, b_eq)
    tableau, basis, success = solve_tableau(tableau, basis, jnp.array(False))

    cases = jnp.array([~success, tableau[-1, -1] > FEAS_TOL, True])
    branches = [
        lambda: (jnp.zeros(n), jnp.zeros(L), jnp.array(jnp.inf), jnp.array(False)),
        lambda: (jnp.zeros(n), jnp.zeros(L), jnp.array(jnp.inf), jnp.array(False)),
        lambda: _phase2_func(tableau, basis, c, b_ineq, b_eq),
    ]
    return lax.switch(jnp.argmax(cases), branches)


def _phase2_func(tableau, basis, c, b_ineq, b_eq):
    """Helper function to execute phase 2."""
    # modify criterion row and execute phase 2
    tableau = _set_criterion_row(c, basis, tableau)
    tableau, basis, success = solve_tableau(tableau, basis, jnp.array(True))

    # retrieving solution
    b = jnp.concatenate((b_ineq, b_eq))
    b_signs = b >= 0.0
    x_opt, lambda_opt, val_opt = get_solution(tableau, basis, b_signs, c)
    return x_opt, lambda_opt, val_opt, success


def _initialize_tableau(
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


def _pivot_col(
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
        lambda: (jnp.array(True), jnp.argmax(_tableau[-1, :criterion_row_stop])),
        lambda: (jnp.array(False), jnp.array(-1, dtype=int)),
    )


def _mrt_helper(tableau, basis, skip_aux):
    """Helper function for performing the min ratio test."""
    L = tableau.shape[0] - 1
    aux_start = tableau.shape[1] - L - 1

    pivcol_found, pivcol = _pivot_col(tableau, skip_aux)
    pivrow_found, pivrow = _lex_min_ratio_test(tableau[:-1, :], pivcol, aux_start)

    cases = jnp.array([~pivrow_found, ~pivcol_found, True])
    branches = [
        lambda: (tableau, basis, jnp.array(False), jnp.array(True)),
        lambda: (tableau, basis, jnp.array(True), jnp.array(True)),
        lambda: (
            _pivoting(tableau, pivcol, pivrow),
            basis.at[pivrow].set(pivcol),
            jnp.array(False),
            jnp.array(False),
        ),
    ]
    return lax.switch(jnp.argmax(cases), branches)


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
    init_val = tableau, basis, jnp.array(False), jnp.array(False)

    def loop_func(i, val):
        """For loop function."""
        _tableau, _basis, _success, _skip = val
        return lax.cond(
            _skip,
            lambda: val,
            lambda: _mrt_helper(_tableau, _basis, skip_aux),
        )

    tableau, basis, success, _ = lax.fori_loop(0, MAX_ITER, loop_func, init_val)
    return tableau, basis, success


def _set_criterion_row(
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


# # ############## #
# # QUANTECON IMPL #
# # ############## #

# import numpy as np
# from jax_simplex.pivoting import _lex_min_ratio_test_qe, _pivoting_qe

# def _initialize_tableau_qe(A_ub, b_ub, A_eq, b_eq):
#     m, k, n = A_ineq.shape[0], A_eq.shape[0], A_ineq.shape[1]
#     L = m + k
#     tableau = np.zeros(((L + 1, n + m + L + 1)))

#     for i in range(m):
#         for j in range(n):
#             tableau[i, j] = A_ub[i, j]
#     for i in range(k):
#         for j in range(n):
#             tableau[m+i, j] = A_eq[i, j]

#     tableau[:L, n:-1] = 0

#     for i in range(m):
#         tableau[i, -1] = b_ub[i]
#         if tableau[i, -1] < 0:
#             for j in range(n):
#                 tableau[i, j] *= -1
#             tableau[i, n+i] = -1
#             tableau[i, -1] *= -1
#         else:
#             tableau[i, n+i] = 1
#         tableau[i, n+m+i] = 1
#     for i in range(k):
#         tableau[m+i, -1] = b_eq[i]
#         if tableau[m+i, -1] < 0:
#             for j in range(n):
#                 tableau[m+i, j] *= -1
#             tableau[m+i, -1] *= -1
#         tableau[m+i, n+m+m+i] = 1

#     tableau[-1, :] = 0
#     for i in range(L):
#         for j in range(n+m):
#             tableau[-1, j] += tableau[i, j]
#         tableau[-1, -1] += tableau[i, -1]

#     basis = np.arange(m + n, m + n + L)

#     return tableau, basis

# def _pivot_col_qe(tableau, skip_aux):
#     L = tableau.shape[0] - 1
#     criterion_row_stop = tableau.shape[1] - 1
#     if skip_aux:
#         criterion_row_stop -= L

#     found = False
#     pivcol = -1
#     coeff = FEAS_TOL
#     for j in range(criterion_row_stop):
#         if tableau[-1, j] > coeff:
#             coeff = tableau[-1, j]
#             pivcol = j
#             found = True

#     return found, pivcol

# def solve_tableau_qe(tableau, basis, skip_aux=True):
#     L = tableau.shape[0] - 1

#     success = False
#     status = 1
#     num_iter = 0

#     while num_iter < MAX_ITER:
#         num_iter += 1

#         pivcol_found, pivcol = _pivot_col_qe(tableau, skip_aux)

#         if not pivcol_found:  # Optimal
#             success = True
#             status = 0
#             break

#         aux_start = tableau.shape[1] - L - 1
#         pivrow_found, pivrow = _lex_min_ratio_test_qe(
#             tableau[:-1, :], pivcol, aux_start
#         )

#         if not pivrow_found:  # Unbounded
#             success = False
#             status = 3
#             break

#         tableau = _pivoting_qe(tableau, pivcol, pivrow)
#         basis[pivrow] = pivcol

#     return tableau, basis, success

# def _set_criterion_row_qe(c, basis, tableau):
#     """
#     Modify the criterion row of the tableau for Phase 2.

#     Parameters
#     ----------
#     c : ndarray(float, ndim=1)
#         ndarray of shape (n,).

#     basis : ndarray(int, ndim=1)
#         ndarray of shape (L,) containing the basis obtained by Phase 1.

#     tableau : ndarray(float, ndim=2)
#         ndarray of shape (L+1, n+m+L+1) containing the tableau obtained
#         by Phase 1. Modified in place.

#     Returns
#     -------
#     tableau : ndarray(float, ndim=2)
#         View to `tableau`.

#     """
#     n = c.shape[0]
#     L = basis.shape[0]

#     for j in range(n):
#         tableau[-1, j] = c[j]
#     tableau[-1, n:] = 0

#     for i in range(L):
#         multiplier = tableau[-1, basis[i]]
#         for j in range(tableau.shape[1]):
#             tableau[-1, j] -= tableau[i, j] * multiplier

#     return tableau

# def get_solution_qe(tableau, basis, b_signs, c):
#     n = c.shape[0]
#     L = basis.shape[0]
#     aux_start = tableau.shape[1] - L - 1

#     x = np.zeros(n)
#     for i in range(L):
#         if basis[i] < n:
#             x[basis[i]] = tableau[i, -1]
#     lambd = np.zeros(L)
#     for j in range(L):
#         lambd[j] = tableau[-1, aux_start+j]
#         if lambd[j] != 0 and b_signs[j]:
#             lambd[j] *= -1
#     fun = tableau[-1, -1] * (-1)

#     return x, lambd, fun

# if __name__ == "__main__":
#     import numpy as np
#     # np.random.seed(0)

#     m = 5
#     n = 5
#     k = 2
#     A_ineq = np.random.rand(m, n)
#     b_ineq = np.random.rand(m)
#     A_eq = np.random.rand(k, n)
#     b_eq = np.random.rand(k)
#     c = np.random.rand(n)

#     # transforming problem into canonical form
#     n = 2 * n
#     A_ineq = np.concatenate((A_ineq, -A_ineq), axis=-1)
#     A_eq = np.concatenate((A_eq, -A_eq), axis=-1)
#     c = np.concatenate((c, -c))

#     # # testing tableau initialization
#     # tableau, basis = _initialize_tableau_qe(A_ineq, b_ineq, A_eq, b_eq)
#     # tableau_jax, basis_jax = _initialize_tableau(A_ineq, b_ineq, A_eq, b_eq)

#     # # testing pivot col
#     # found, pivcol = _pivot_col_qe(tableau, True)
#     # found_jax, pivcol_jax = _pivot_col(tableau_jax, jnp.array(True))

#     # # testing solve_tableau
#     # tableau_jax, basis_jax, success_jax = solve_tableau(
#     #     tableau_jax, basis_jax, jnp.array(False)
#     # )
#     # tableau, basis, success = solve_tableau_qe(
#     #     tableau, basis, False
#     # )

#     # # testing set_criterion_row
#     # tableau_jax = jit(_set_criterion_row)(jnp.array(c), basis_jax, tableau_jax)
#     # tableau = _set_criterion_row_qe(c, basis, tableau)

#     # # testing get_solution
#     # b = np.concatenate((b_ineq, b_eq))
#     # b_signs = b >= 0
#     # x_opt, lambda_opt, opt_val = get_solution_qe(tableau, basis, b_signs, c)
#     # x_opt_jax, lambda_opt_jax, opt_val_jax = jit(get_solution)(
#     #     tableau_jax, basis_jax, jnp.array(b_signs), jnp.array(c)
#     # )

#     # testing linprog solve
#     from quantecon.optimize import linprog_simplex
#     res = linprog_simplex(c, A_ineq, b_ineq, A_eq, b_eq)

#     x_opt, lambda_opt, val_opt, success = jit(linprog)(c, A_ineq, b_ineq, A_eq, b_eq)

#     print("COMPARING TO QUANTECON")
#     if res.status == 0 and success:
#         print(res.x - x_opt)
#         print(res.lambd - lambda_opt)
#         print(res.fun - val_opt)
#     print()

#     from scipy.optimize import linprog as linprog_scipy
#     res_scipy = linprog_scipy(
#         -c, A_ub=A_ineq, b_ub=b_ineq, A_eq=A_eq, b_eq=b_eq, method="highs-ds"
#     )

#     print("COMPARING TO SCIPY")
#     if res_scipy.success and success:
#         print(res_scipy.x - x_opt)
#         lambda_opt_scipy = -np.concatenate(
#             (res_scipy.ineqlin.marginals, res_scipy.eqlin.marginals)
#         )
#         print(lambda_opt_scipy - lambda_opt)
#         print(-res_scipy.fun - val_opt)
#     print()

#     import cvxpy as cp
#     x = cp.Variable(n)
#     obj = cp.Maximize(c @ x)
#     constraints = [x >= 0]
#     if m > 0:
#         constraints += [A_ineq @ x <= b_ineq]
#     if k > 0:
#         constraints += [A_eq @ x == b_eq]
#     prob = cp.Problem(obj, constraints)
#     prob.solve()

#     print("COMPARING TO CVXPY")
#     if prob.status == "optimal" and success:
#         _lambda_opt = np.concatenate(
#             (constraints[1].dual_value, constraints[2].dual_value)
#         )
#         print(x.value - x_opt)
#         print(_lambda_opt - lambda_opt)
#         print(prob.value - val_opt)
#     print()

#     breakpoint()
