from keya.kernel.kernel import PascalKernel, CombinatorialPosition
from keya.kernel.cyclotomic import CyclotomicBinary
from keya.kernel.operators import Operator, Fuse, Diff
from typing import TypeAlias
from math import comb, gcd
import jax.numpy as jnp
from jax import jit
from functools import partial
from sympy import symbols, Poly, expand, cyclotomic_poly
from sympy.core.symbol import Symbol
from typing import Type
import os
from tqdm import tqdm

"""
Core implementation of the PascalGaugeField engine.

This module defines the primary classes for the Pascal space simulation,
including the hierarchical layers and the main engine that drives evolution.
"""

# A type alias for clarity in function signatures
SymbolicExpression: TypeAlias = CyclotomicBinary

# --- Utility Functions ---

def to_poly(coeffs: jnp.ndarray, var: SymbolicExpression) -> Poly:
    """Converts a numpy array of coefficients into a SymPy polynomial."""
    return Poly(coeffs, var)

# --- Core Architectural Components ---

class PascalGaugeField:
    """
    The main engine for simulating dynamics on a Pascal-Sierpinski substrate.

    This class uses a PascalKernel as its computational fabric and evolves
    states represented by CyclotomicBinary numbers.
    """
    def __init__(self):
        self.kernel = PascalKernel()
        # The state is now a list of CyclotomicBinary numbers.
        self.state: list[CyclotomicBinary] = []
        # The operators are now polynomial-based.
        self.operators: list[Operator] = []

    def initialize_state(self, values: list[int], operators: list[Operator]):
        """Initializes the field with values and a chain of operators."""
        self.state = [CyclotomicBinary(v) for v in values]
        self.operators = operators

    def evolve(self, max_steps: int = 10) -> list[CyclotomicBinary]:
        """
        Evolves the system state by applying the operator chain.
        """
        print(f"Evolving {len(self.state)} states with operators: {[op.name for op in self.operators]}")

        # Process each state through the operator chain
        final_states = []
        for state in self.state:
            current_coeffs = state.components
            for op in self.operators:
                current_coeffs = self.kernel.apply_polynomial(current_coeffs, op.coeffs)
            
            # Create a new CyclotomicBinary from the final coefficients
            new_state = CyclotomicBinary.from_vector(current_coeffs)
            final_states.append(new_state)
        
        self.state = final_states
        print("Evolution complete.")
        return self.state

    def visualize(self, output_path: str | None = None):
        """
        Visualizes the kernel's state.
        This will need to be adapted to the new model.
        """
        print("Visualization is not yet implemented for the new kernel-based engine.")
        pass

# The old OperatorPlugin system is no longer compatible and has been removed.
# A new system will be designed for the PascalKernel architecture.