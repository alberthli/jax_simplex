# JAX Simplex
This repo contains an implementation of the simplex method for solving linear programs written entirely in JAX. It is a direct port of the implementation in [quantecon](https://github.com/QuantEcon/QuantEcon.py/tree/main/quantecon/optimize).

To install, clone the repository (`conda` environment recommended) and in the repository root run
```
pip install -r requirements.txt
pip install -e .
```
By default, the requirements file installs `jax` assuming your system has Cuda 11. To install for Cuda 12, simply uncomment the correct dependencies before running the above command.

## Example Usage
```
import jax.numpy as jnp
from jax import jit
from jax_simplex.simplex import linprog

# example problem from quantecon tests
# this example isn't batched, but you can easily vmap the call as well!
c = -jnp.array([-1., 8, 4, -6])
A_ineq = jnp.array(
    [
        [-7., -7, 6, 9],
        [1, -1, -3, 0],
        [10, -10, -7, 7],
        [6, -1, 3, 4],
    ]
)
b_ineq = jnp.array([-3., 6, -6, 6])
A_eq = jnp.array([[-10., 1, 1, -8]])
b_eq = jnp.array([-4.])
x_opt, lambda_opt, val_opt, success = jit(linprog)(c, A_ineq, b_ineq, A_eq, b_eq)
```