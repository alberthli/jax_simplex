from functools import partial
from typing import Tuple

import jax.numpy as jnp
from jax import config, lax

config.update("jax_enable_x64", True)
TOL_PIV = 1e-7
TOL_RATIO_DIFF = 1e-13


def pivoting(tableau: jnp.ndarray, pc: jnp.ndarray, pr: jnp.ndarray) -> jnp.ndarray:
    """Perform a pivoting step and returns the new tableau with the replacement op.

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


# ######################## #
# MIN_RATIO_TEST + HELPERS #
# ######################## #


def min_ratio_test_no_tie_breaking(
    tableau: jnp.ndarray,
    pivot: jnp.ndarray,
    test_col: jnp.ndarray,
    argmins: jnp.ndarray,
    num_candidates: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Performs the min ratio test without tie breaks.

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
    # initializing cond and body functions for min ratio test loop
    mrt_cond_fun = partial(_mrt_cond_fun, num_candidates=num_candidates)
    mrt_body_fun = partial(
        _mrt_body_fun,
        pivot=pivot,
        test_col=test_col,
        tableau=tableau,
        num_candidates=num_candidates,
    )

    # `val` = (k, ratio_min, num_argmins, argmins)
    init_val = 0, jnp.inf, 0, argmins
    num_argmins, argmins = lax.while_loop(mrt_cond_fun, mrt_body_fun, init_val)[-2:]
    return num_argmins, argmins


def _mrt_cond_fun(val, num_candidates):
    """MRT helper function. Condition function of while loop."""
    k = val[0]
    return k < num_candidates


def _mrt_body_fun(val, pivot, test_col, tableau, num_candidates):
    """MRT helper function. Body function of while loop."""
    k, ratio_min, num_argmins, argmins = val
    i = argmins[k]
    ratio = tableau[i, test_col] / tableau[i, pivot]

    # three cases:
    # (1) non-positive element OR ratio larger than min -> continue loop
    # (2) ratio less than min -> reassign argmin, reset num_argmins to 1
    # (3) else (ratio equal) -> increment num_argmins, add duplicate argmin to order
    case1 = jnp.logical_or(
        tableau[i, pivot] <= TOL_PIV, ratio > ratio_min + TOL_RATIO_DIFF
    )
    case2 = ratio < ratio_min - TOL_RATIO_DIFF
    case3 = True
    cases = jnp.array([case1, case2, case3])

    # computing elements to pass to the branches
    # [NOTE] also tried passing `i` into the branch helpers and computing the updated
    # `argmins` in each branch, but that seems slower than pre-computing all 3 and
    # simply branching on the selection.
    ratios_tuple = (ratio_min, ratio)
    argmins_tuple = (argmins, argmins.at[0].set(i), argmins.at[num_argmins].set(i))
    branches = [
        _mrt_branch1_helper,
        _mrt_branch2_helper,
        _mrt_branch3_helper,
    ]
    return lax.switch(
        jnp.argmax(cases),
        branches,
        k,
        ratios_tuple,
        num_argmins,
        argmins_tuple,
    )


def _mrt_branch1_helper(k, ratios_tuple, num_argmins, argmins_tuple):
    return k + 1, ratios_tuple[0], num_argmins, argmins_tuple[0]


def _mrt_branch2_helper(k, ratios_tuple, num_argmins, argmins_tuple):
    return k + 1, ratios_tuple[1], 1, argmins_tuple[1]


def _mrt_branch3_helper(k, ratios_tuple, num_argmins, argmins_tuple):
    return k + 1, ratios_tuple[0], num_argmins + 1, argmins_tuple[2]


# ####################### #
# LEX_MIN_RATIO + HELPERS #
# ####################### #


def lex_min_ratio_test(
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
        Whether there was a positive entry in the piv column. True iff num_argmins==1.
    row_min : jnp.ndarray[int], shape=(,)
        Index of the row with the lexico-minimum ratio.
    """
    nrows = tableau.shape[0]
    num_candidates = jnp.array(nrows)

    # initial while loop value
    # `val` = (j, num_argmins, argmins)
    argmins = jnp.arange(nrows)

    num_argmins, argmins = min_ratio_test_no_tie_breaking(
        tableau, pivot, jnp.array(-1), argmins, num_candidates
    )
    init_val = slack_start, num_argmins, argmins

    # executing while loop
    lex_mrt_cond_fun = partial(_lex_mrt_cond_fun, slack_start=slack_start, nrows=nrows)
    lex_mrt_body_fun = partial(
        _lex_mrt_body_fun,
        tableau=tableau,
        pivot=pivot,
    )
    _, num_argmins, argmins = lax.while_loop(
        lex_mrt_cond_fun,
        lex_mrt_body_fun,
        init_val,
    )
    return num_argmins == 1, argmins[0]


def _lex_mrt_cond_fun(val, slack_start, nrows):
    """Helper function for `lex_min_ratio_test`. Cond function of while loop."""
    # Continue looping until num_argmins == 1 or until n_rows iterations have passed.
    j, num_argmins, _ = val
    return jnp.logical_and(num_argmins != 1, j < slack_start + nrows)


def _lex_mrt_body_fun(val, tableau, pivot):
    """Helper function for `lex_min_ratio_test`. Body function of while loop."""
    # If j == pivot, continue the loop. Otherwise, compute a new argmin.
    j, num_argmins, argmins = val
    return lax.cond(
        j == pivot,
        lambda: (j + 1, num_argmins, argmins),
        lambda: (j + 1,)
        + min_ratio_test_no_tie_breaking(tableau, pivot, j, argmins, num_argmins),
    )
