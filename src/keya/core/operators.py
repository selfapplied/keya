"""
Core symbolic operators for the kéya engine.

Each function will be designed to operate on JAX arrays,
allowing for JIT compilation and hardware acceleration.
"""

from typing import Any, Callable, List

import jax
import jax.numpy as jnp
import math

# --- Foundational Attractors ---

PI: Any = jnp.pi
GOLD: Any = (1 + jnp.sqrt(5)) / 2

# --- Core Operators (to be implemented) ---


def fuse(a: jax.Array, b: jax.Array) -> jax.Array:
    """Fusion operator (⊕): Combines two elements."""
    # Defaulting to simple addition for now.
    return a + b


def tensor(a: jax.Array, b: jax.Array) -> jax.Array:
    """Tensor operator (⊗): Binds two elements."""
    # Defaulting to multiplication.
    return a * b


def reflect(a: jax.Array) -> jax.Array:
    """Reflection operator (~): Inverts phase or orientation."""
    # Defaulting to negation.
    return -a


def descent(a: jax.Array) -> jax.Array:
    """Descent operator (ℓ): Folds or regularizes a process."""
    # This will eventually be a complex function, e.g., Ramanujan summation proxy.
    # For now, a simple identity or compression.
    return a


def growth(a: jax.Array, n: Any) -> jax.Array:
    """Growth operator (↑): Increases dimensionality or scale."""
    # Defaulting to exponentiation.
    return a**n


def curvature(a: jax.Array) -> jax.Array:
    """
    Curvature operator (κ): Computes the discrete Laplacian across all dimensions
    using periodic boundary conditions.
    """
    dim = a.ndim
    lap = sum(
        jnp.roll(a, shift=1, axis=i) + jnp.roll(a, shift=-1, axis=i) - 2 * a
        for i in range(dim)
    )
    return lap

def taylorphasewalk(f: Callable[[jnp.ndarray], jnp.ndarray],
                     x0: jnp.ndarray,
                     order: int) -> jnp.ndarray:
    """
    Compute the univariate Taylor series coefficients of f around x0 up to the given order.
    Returns an array of shape (order+1,) where coeffs[k] = f^{(k)}(x0) / k!.
    """
    deriv_fn = f
    coeffs: List[jnp.ndarray] = []
    for k in range(order + 1):
        if k == 0:
            val = deriv_fn(x0)
        else:
            deriv_fn = jax.grad(deriv_fn)
            val = deriv_fn(x0)
        coeffs.append(val / math.factorial(k))
    return jnp.stack(coeffs)


def taylorphasewalk_inverse(coeffs: jnp.ndarray) -> jnp.ndarray:
    """
    Given univariate Taylor coefficients coeffs of f at x0 (coeffs[k] = f^{(k)}(x0)/k!),
    compute the coefficients of 1/f up to the same order via formal power series inversion.
    """
    order = coeffs.shape[0] - 1
    inv_coeffs: List[jnp.ndarray] = [1.0 / coeffs[0]]
    for n in range(1, order + 1):
        s = 0.0
        for k in range(1, n + 1):
            s += coeffs[k] * inv_coeffs[n - k]
        inv_coeffs.append(-s / coeffs[0])
    return jnp.stack(inv_coeffs)


def taylorphasewalk_multivariate(f: Callable,
                                 x0: jnp.ndarray,
                                 order: int) -> List[jnp.ndarray]:
    """
    Compute multivariate Taylor series of f around x0 up to total degree `order`.
    Returns a list of arrays where the k-th element is the k-th order derivative tensor
    f^{(k)}(x0)/k!.
    """
    deriv_fn = f
    coeffs: List[jnp.ndarray] = []
    # zeroth order
    coeffs.append(deriv_fn(x0))
    for k in range(1, order + 1):
        deriv_fn = jax.jacfwd(deriv_fn)
        coeffs.append(deriv_fn(x0) / math.factorial(k))
    return coeffs
