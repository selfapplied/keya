"""
Polynomial Operator Definitions for the Pascal-Sierpinski Engine.

This module defines the core Operator class that represents a transformation
as a polynomial. These operators are applied via convolution by the PascalKernel.
"""
from __future__ import annotations
import jax.numpy as jnp
from typing import Union

class Operator:
    """
    Represents a system operator as a polynomial.
    The operator's action is defined by the convolution of its polynomial
    representation with a state's polynomial representation.
    """
    def __init__(self, name: str, coefficients: Union[list[int], jnp.ndarray]):
        """
        Initializes an operator with a name and its polynomial coefficients.
        
        Args:
            name: The human-readable name of the operator.
            coefficients: A list of integers (for 1D) or a JAX array (for 2D)
                          representing the polynomial, e.g., [1, 0, 1] for x^2 + 1.
        """
        self.name = name
        if isinstance(coefficients, list):
            self.coeffs = jnp.array(coefficients, dtype=jnp.int64)
        else:
            self.coeffs = coefficients

    def __repr__(self) -> str:
        return f"Operator({self.name}, coeffs={self.coeffs})"

# --- Pre-defined Standard Operators ---

def Fuse() -> Operator:
    """
    The Fuse operator, representing addition/fusion.
    Its polynomial is `x + 1`. Applying it via convolution with a state
    is equivalent to adding the state to a shifted version of itself,
    which creates a fusion or superposition of neighboring states.
    """
    return Operator("Fuse", [1, 1])

def Diff() -> Operator:
    """
    The Differential operator.
    Its polynomial is `x - 1`. Applying it is equivalent to taking a
    finite difference of the state's components.
    """
    return Operator("Diff", [-1, 1])

def Identity() -> Operator:
    """The Identity operator, which causes no change. Its polynomial is `1`."""
    return Operator("Identity", [1])

def GoldenCurvature() -> Operator:
    """
    The Golden Curvature operator, which rotates a state in the complex plane.

    This action is defined by multiplication by the complex number e^(i*pi/phi),
    representing the rotational force of the Golden Spiral's limit cycle.
    """
    phi = (1 + jnp.sqrt(5)) / 2
    # The complex coefficient representing the rotation.
    turn_angle = jnp.pi / phi
    coeff = jnp.cos(turn_angle) + 1j * jnp.sin(turn_angle)
    return Operator("GoldenCurvature", jnp.array([coeff], dtype=jnp.complex128)) 