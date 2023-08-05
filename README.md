# JAX Simplex
This repo contains an implementation of the simplex method for solving linear programs written entirely in JAX. It is a direct port of the implementation in [quantecon](https://github.com/QuantEcon/QuantEcon.py/tree/main/quantecon/optimize).

To install, clone the repository (`conda` environment recommended) and in the repository root run
```
pip install -r requirements.txt
pip install -e .
```
By default, the requirements file installs `jax` assuming your system has Cuda 11. To install for Cuda 12, simply uncomment the correct dependencies before running the above command.