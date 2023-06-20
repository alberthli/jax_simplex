import time

# import cvxpy as cp
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from jax.experimental.compilation_cache import compilation_cache as cc

from jax_simplex.simplex import linprog

cc.initialize_cache("jax_cache")

"""Times the jit-compiled simplex method vs. Gurobi via the cvxpy interface."""

# compiled+batched linprog
B = 100
n = 20
m = 20
k = 5

# pre-compiling
c = jnp.zeros((B, n))
A_ineq = jnp.zeros((B, m, n))
b_ineq = jnp.zeros((B, m))
A_eq = jnp.zeros((B, k, n))
b_eq = jnp.zeros((B, k))
batch_linprog = jit(vmap(linprog))
x_opt, lambda_opt, val_opt, success = batch_linprog(c, A_ineq, b_ineq, A_eq, b_eq)

print("PRE-COMPILED")

# timing
start = time.time()
c = jnp.array(np.random.randn((B, n)))
A_ineq = jnp.array(np.random.randn((B, m, n)))
b_ineq = jnp.array(np.random.randn((B, m)))
A_eq = jnp.array(np.random.randn((B, k, n)))
b_eq = jnp.array(np.random.randn((B, k)))
batch_linprog(c, A_ineq, b_ineq, A_eq, b_eq)[0].block_until_ready()
end = time.time()
print((end - start) / B)

breakpoint()
