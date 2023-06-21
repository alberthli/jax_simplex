from functools import partial

import jax.numpy as jnp
from jax import config, lax

config.update("jax_enable_x64", True)
TOL_PIV = 1e-10
TOL_RATIO_DIFF = 1e-15


def _pivoting(tableau: jnp.ndarray, pc: jnp.ndarray, pr: jnp.ndarray) -> jnp.ndarray:
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


def _min_ratio_test_no_tie_breaking(
    tableau: jnp.ndarray,
    pivot: jnp.ndarray,
    test_col: jnp.ndarray,
    argmins: jnp.ndarray,
    num_candidates: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
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

    # inner branching logic
    def inner_func(argmins):
        """Runs when num_argmins >= 2.

        Runs the lexicographic min ratio test until there's a unique choice.
        """
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
