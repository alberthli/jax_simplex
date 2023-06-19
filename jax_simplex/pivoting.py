from functools import partial

import jax.numpy as jnp
from jax import config, lax

config.update("jax_enable_x64", True)
TOL_PIV = 1e-10
TOL_RATIO_DIFF = 1e-15

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


def _pivoting(tableau: jnp.ndarray, pc: jnp.ndarray, pr: jnp.ndarray) -> jnp.ndarray:
    """Perform a pivoting step and returns the new tableau with the replacement op.

    [TESTED VS. QUANTECON]

    Parameters
    ----------
    tableau : jnp.ndarray, shape=(L + 1, n + m + L + 1)
        The tableau.
    pc : jnp.ndarray[int], shape=(,)
        The pivot column.
    pr : jnp.ndarray[int], shape=(,)
        The pivot row.

    Returns
    -------
    new_tableau : jnp.ndarray, shape=(L + 1, n + m + L + 1)
        The new tableau.
    """
    pivot_elt = tableau[pr, pc]
    _tableau = tableau.at[pr, :].set(tableau[pr, :] / pivot_elt)
    col = _tableau.at[pr, :].get()[None, :]
    row = _tableau.at[:, pc].get()[:, None]
    _subtract_term = row * col
    subtract_term = _subtract_term.at[pr, :].set(0.0)
    new_tableau = _tableau - subtract_term
    return new_tableau


def _min_ratio_test_no_tie_breaking(
    tableau: jnp.ndarray,
    pivot: jnp.ndarray,
    test_col: jnp.ndarray,
    argmins: jnp.ndarray,
    num_candidates: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Performs the min ratio test without tie breaks.

    [TESTED VS. QUANTECON]

    Parameters
    ----------
    tableau : jnp.ndarray, shape=(L + 1, n + m + L + 1)
        The tableau.
    pivot : jnp.ndarray[int], shape=(,)
        The pivot.
    test_col : jnp.ndarray[int], shape=(,)
        Index of the column used in the test.
    argmins : jnp.ndarray[int], shape=(L + 1,)
        The indices corresponding to the minimum ratio.
    num_candidates : jnp.ndarray[int], shape=(,)
        The number of candidates to consider (determines loop length).

    Returns
    -------
    num_argmins : jnp.ndarray[int], shape=(,)
        Number of minimizing rows.
    argmins : jnp.ndarray[int], shape=(L + 1,)
        The modified indices of the minimum ratio.
    """

    def _argmin_update_func(carry, argmins, pivot, test_col):
        """Logic in single loop iteration."""
        # unpacking carry state
        k, ratio_min, num_argmins, argmins = carry

        i = argmins[k]

        def outer_branch1():
            """Early stop if one of two conditions.

            Conditions:
            (1) tableau[i, pivot] <= TOL_PIV
            (2) the total number of candidates has already been considered

            Cond (2) is necessary because dynamically changing the number of loop iters
            to run doesn't play nice with JIT, so we always run L+1 iters but tack on
            dummy operations to pad the loop.
            """
            return (k + 1, ratio_min, num_argmins, argmins), argmins

        def outer_branch2():
            """If not early stopped, then do a 3-way case-switch base on ratio.

            Cases:
            (1) ratio > ratio_min + TOL_RATIO_DIFF, the curr ratio is larger than min
                In this case, we just continue the loop.
            (2) ratio < ratio_min - TOL_RATIO_DIFF, the curr ratio is lower than min
                In this case, we have a new minimum, so we reset the argmin to the
                current index and reset the argmin counter to 1.
            (3) else, the curr ratio is same as min
                In this case, we add 1 to the argmin counter and tack on this candidate
                to the ordering of argmin indices.
            """
            ratio = tableau[i, test_col] / tableau[i, pivot]

            conds = jnp.array(
                [
                    ratio > ratio_min + TOL_RATIO_DIFF,
                    ratio < ratio_min - TOL_RATIO_DIFF,
                    True,
                ]
            )

            def inner_branch2():
                new_argmins = argmins.at[0].set(i)
                return (k + 1, ratio, 1, new_argmins), new_argmins

            def inner_branch3():
                new_argmins = argmins.at[num_argmins].set(i)
                return (k + 1, ratio_min, num_argmins + 1, new_argmins), new_argmins

            branches = [
                lambda: ((k + 1, ratio_min, num_argmins, argmins), argmins),
                inner_branch2,
                inner_branch3,
            ]
            return lax.switch(jnp.argmax(conds), branches)

        return lax.cond(
            jnp.logical_or(tableau[i, pivot] <= TOL_PIV, k >= num_candidates),
            outer_branch1,
            outer_branch2,
        )

    # the static values are the pivot and the test column
    argmin_update_func = partial(_argmin_update_func, pivot=pivot, test_col=test_col)

    # unroll the min ratio test loop
    carry_init = (0, jnp.inf, 0, argmins)
    carry_final, all_argmins = lax.scan(
        argmin_update_func, init=carry_init, xs=jnp.arange(tableau.shape[0])
    )
    num_argmins = carry_final[2]
    argmins = all_argmins[-1, :]  # scan returns all outputs of the loop, take last
    return num_argmins, argmins


def _lex_min_ratio_test(
    tableau: jnp.ndarray,
    pivot: jnp.ndarray,
    slack_start: jnp.ndarray,
) -> jnp.ndarray:
    """Performs the lexico-minimum ratio test.

    Parameters
    ----------
    tableau : jnp.ndarray, shape=(L + 1, n + m + L + 1)
        The tableau.
    pivot : jnp.ndarray[int], shape=(,)
        The pivot.
    slack_start : jnp.ndarray[int], shape=(,)
        The index where slack variables start.

    Returns
    -------
    found : jnp.ndarray[bool], shape=(,)
        Whether there was a positive entry in the pivot column.
    row_min : jnp.ndarray[int], shape=(,)
        Index of the row with the lexico-minimum ratio.
    """
    nrows = tableau.shape[0]
    num_candidates = jnp.array(nrows)

    argmins = jnp.arange(nrows)
    num_argmins, argmins = _min_ratio_test_no_tie_breaking(
        tableau, pivot, jnp.array(-1), argmins, num_candidates
    )

    # inner branching logic
    def inner_func(argmins):
        """Runs when num_argmins >= 2.

        Runs the lexicographic min ratio test until there's a unique choice.
        """

        def argmin_update_func(carry, argmins):
            # unpacking loop state
            j, num_argmins, argmins, _ = carry

            def branches_inner3(argmins):
                num_argmins_inner, argmins_inner = _min_ratio_test_no_tie_breaking(
                    tableau, pivot, j, argmins, num_argmins
                )
                carry_inner = (j + 1, num_argmins_inner, argmins_inner, False)
                return carry_inner, argmins_inner

            conds_inner = jnp.array(
                [
                    j == pivot,  # skip the pivot
                    num_argmins == 1,  # terminate the loop by only using dummy ops
                    True,  # else, run min ratio test w/o tiebreaks
                ]
            )
            branches_inner = [
                lambda argmins: ((j + 1, num_argmins, argmins, False), argmins),
                lambda argmins: ((j + 1, num_argmins, argmins, True), argmins),
                lambda argmins: branches_inner3(argmins),
            ]
            return lax.switch(jnp.argmax(conds_inner), branches_inner, argmins)

        # execute the lexicographic min ratio test loop
        xs = jnp.arange(nrows)
        carry_init = (slack_start, num_argmins, argmins, False)
        carry_final, all_argmins = lax.scan(argmin_update_func, init=carry_init, xs=xs)
        found = carry_final[3]
        argmins = all_argmins[-1, :]
        return found, argmins[0]

    # outer branching logic
    conds_outer = jnp.array([num_argmins == 1, num_argmins >= 2, True])
    branches_outer = [
        lambda argmins: (True, argmins[0]),
        lambda argmins: inner_func(argmins),
        lambda argmins: (False, argmins[0]),
    ]
    return lax.switch(jnp.argmax(conds_outer), branches_outer, argmins)


# # ############## #
# # QUANTECON IMPL #
# # ############## #

# def _pivoting_qe(tableau, pivot_col, pivot_row):
#     """
#     Perform a pivoting step. Modify `tableau` in place.

#     Parameters
#     ----------
#     tableau : ndarray(float, ndim=2)
#         Array containing the tableau.

#     pivot_col : scalar(int)
#         Pivot column index.

#     pivot_row : scalar(int)
#         Pivot row index.

#     Returns
#     -------
#     tableau : ndarray(float, ndim=2)
#         View to `tableau`.

#     """
#     nrows, ncols = tableau.shape

#     pivot_elt = tableau[pivot_row, pivot_col]
#     for j in range(ncols):
#         tableau[pivot_row, j] /= pivot_elt

#     for i in range(nrows):
#         if i == pivot_row:
#             continue
#         multiplier = tableau[i, pivot_col]
#         if multiplier == 0:
#             continue
#         for j in range(ncols):
#             tableau[i, j] -= tableau[pivot_row, j] * multiplier

#     return tableau

# def _min_ratio_test_no_tie_breaking_qe(tableau, pivot, test_col,
#                                     argmins, num_candidates):
#     """
#     Perform the minimum ratio test, without tie breaking, for the
#     candidate rows in `argmins[:num_candidates]`. Return the number
#     `num_argmins` of the rows minimizing the ratio and store thier
#     indices in `argmins[:num_argmins]`.

#     Parameters
#     ----------
#     tableau : ndarray(float, ndim=2)
#         Array containing the tableau.

#     pivot : scalar(int)
#         Pivot.

#     test_col : scalar(int)
#         Index of the column used in the test.

#     argmins : ndarray(int, ndim=1)
#         Array containing the indices of the candidate rows. Modified in
#         place to store the indices of minimizing rows.

#     num_candidates : scalar(int)
#         Number of candidate rows in `argmins`.

#     Returns
#     -------
#     num_argmins : scalar(int)
#         Number of minimizing rows.

#     """
#     ratio_min = np.inf
#     num_argmins = 0

#     for k in range(num_candidates):
#         i = argmins[k]
#         if tableau[i, pivot] <= TOL_PIV:  # Treated as nonpositive
#             continue
#         ratio = tableau[i, test_col] / tableau[i, pivot]
#         if ratio > ratio_min + TOL_RATIO_DIFF:  # Ratio large for i
#             continue
#         elif ratio < ratio_min - TOL_RATIO_DIFF:  # Ratio smaller for i
#             ratio_min = ratio
#             num_argmins = 1
#         else:  # Ratio equal
#             num_argmins += 1
#         argmins[num_argmins-1] = i

#     return num_argmins

# def _lex_min_ratio_test_qe(tableau, pivot, slack_start,
#                         tol_piv=TOL_PIV, tol_ratio_diff=TOL_RATIO_DIFF):
#     """
#     Perform the lexico-minimum ratio test.

#     Parameters
#     ----------
#     tableau : ndarray(float, ndim=2)
#         Array containing the tableau.

#     pivot : scalar(int)
#         Pivot.

#     slack_start : scalar(int)
#         First index for the slack variables.

#     Returns
#     -------
#     found : bool
#         False if there is no positive entry in the pivot column.

#     row_min : scalar(int)
#         Index of the row with the lexico-minimum ratio.

#     """
#     nrows = tableau.shape[0]
#     num_candidates = nrows

#     found = False

#     # Initialize `argmins`
#     argmins = np.arange(nrows)

#     num_argmins = _min_ratio_test_no_tie_breaking_qe(
#         tableau, pivot, -1, argmins, num_candidates
#     )
#     if num_argmins == 1:
#         found = True
#     elif num_argmins >= 2:
#         for j in range(slack_start, slack_start+nrows):
#             if j == pivot:
#                 continue
#             num_argmins = _min_ratio_test_no_tie_breaking_qe(
#                 tableau, pivot, j, argmins, num_argmins,
#             )
#             if num_argmins == 1:
#                 found = True
#                 break
#     return found, argmins[0]

# if __name__ == "__main__":
#     import numpy as np
#     from jax import jacobian, jacrev, jacfwd
#     tableau = np.random.rand(5, 4, 3)
#     pr = np.array([0,1,2,3,0], dtype=int)
#     pc = np.array([0,1,2,0,1], dtype=int)

#     # testing pivoting
#     tableau_jax = jit(vmap(_pivoting))(jnp.array(tableau), jnp.array(pc), jnp.array(pr))
#     for i in range(5):
#         _pivoting_qe(tableau[i, ...], pc[i], pr[i])

#     grad_tableau_jax = jit(vmap(jacobian(_pivoting)))(
#         jnp.array(tableau), jnp.array(pc), jnp.array(pr)
#     )
#     breakpoint()

#     # testing the min ratio test without tiebreaks
#     tableau = np.random.rand(5, 4, 3)
#     pivot = np.array([0, 1, 2, 0, 1])
#     test_col = np.array([2, 1, 0, 2, 1])
#     num_candidates = np.array([1, 2, 3, 4, 3])
#     na_jax, am_jax = jit(vmap(_min_ratio_test_no_tie_breaking))(
#         jnp.array(tableau),
#         jnp.array(pivot),
#         jnp.array(test_col),
#         jnp.repeat(jnp.arange(4)[None, ...], 5, axis=0),
#         jnp.array(num_candidates),
#     )
#     na = []
#     am = []
#     for i in range(5):
#         _am = np.arange(4)
#         _na = _min_ratio_test_no_tie_breaking_qe(
#             tableau[i, ...], pivot[i], test_col[i], _am, num_candidates[i]
#         )
#         am.append(_am)
#         na.append(_na)
#     am = np.stack(am)
#     na = np.stack(na)

#     grad_mrt_jax = jit(vmap(jacfwd(_min_ratio_test_no_tie_breaking)))(
#         jnp.array(tableau),
#         jnp.array(pivot),
#         jnp.array(test_col),
#         jnp.repeat(jnp.arange(4)[None, ...], 5, axis=0),
#         jnp.array(num_candidates),
#     )  # [TODO] returns all 0s using fwd mode
#     breakpoint()

#     # testing the lexicographic min ratio test
#     tableau = np.random.rand(5, 4, 3)
#     pivot = np.array([0, 1, 2, 0, 1])
#     slack_start = np.array([0, 0, 0, 0, 0])
#     found_jax, row_min_jax = jit(vmap(_lex_min_ratio_test))(
#         jnp.array(tableau), jnp.array(pivot), jnp.array(slack_start)
#     )

#     found = []
#     row_min = []
#     for i in range(5):
#         _found, _row_min = _lex_min_ratio_test_qe(
#             tableau[i, ...], pivot[i], 0
#         )
#         found.append(_found)
#         row_min.append(_row_min)
#     found = np.stack(found)
#     row_min = np.stack(row_min)

#     # grad_lmrt_jax = jit(vmap(jacfwd(_lex_min_ratio_test)))(
#     #     jnp.array(tableau), jnp.array(pivot), jnp.array(slack_start)
#     # )  # [TODO] breaks!
#     breakpoint()
