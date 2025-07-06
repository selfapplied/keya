"""
Polynomial Operator Definitions for the Pascal-Sierpinski Engine.

This module defines the core Operator class that represents a transformation
as a polynomial. These operators are applied via convolution by the PascalKernel.
"""
from __future__ import annotations
import jax.numpy as jnp

class Operator:
    """
    Represents a system operator as a polynomial.
    The operator's action is defined by the convolution of its polynomial
    representation with a state's polynomial representation.
    """
    def __init__(self, name: str, coefficients: list[int]):
        """
        Initializes an operator with a name and its polynomial coefficients.
        
        Args:
            name: The human-readable name of the operator.
            coefficients: A list of integers representing the polynomial,
                          e.g., [1, 0, 1] for x^2 + 1.
        """
        self.name = name
        self.coeffs = jnp.array(coefficients, dtype=jnp.int64)

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
    return Operator(name="Fuse", coefficients=[1, 1])

def Diff() -> Operator:
    """
    The Differential operator.
    Its polynomial is `x - 1`. Applying it is equivalent to taking a
    finite difference of the state's components.
    """
    return Operator(name="Diff", coefficients=[-1, 1])

def Identity() -> Operator:
    """The Identity operator, which causes no change. Its polynomial is `1`."""
    return Operator(name="Identity", coefficients=[1]) 