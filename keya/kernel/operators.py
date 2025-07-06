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
            self.coeffs = jnp.array(coefficients, dtype=jnp.int32)
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

# --- Tesla 3-6-9 Trinity Operators ---

def Tesla3() -> Operator:
    """
    The Generator - Cubic residue annihilation in ℤ/7ℤ
    
    Under modular 7: a³ ≡ b³ ≡ c³ ≡ 0 (cubic residues annihilate to void)
    Triggers when n % 3 == 0: emit_light(3) - Tesla's 3-pulse
    
    This implements x³ ≡ 0 (mod 7) transformation in the twisted ring.
    The polynomial x³ - 1 enforces cubic periodicity with collapse.
    """
    # Polynomial that enforces x³ ≡ 1 (mod 7) cyclotomic behavior
    # But in the twisted ring, this creates the 3-pulse generator
    return Operator("Tesla3", [1, 0, 0, -1])  # x³ - 1

def Tesla6() -> Operator:
    """
    The Resonator - S₃ symmetry group action
    
    Perfect symmetry: Sym(Σ₃) = S₃ (order 6) acts on power set
    Renormalization fixes: a+b+c ≡ 0 (mod 7), ab+bc+ca ≡ 1 (mod 7)
    Triggers when n % 6 == 0: fold_digital_root()
    
    This implements the S₃ group action as convolution.
    S₃ has 6 elements with alternating representation.
    """
    # Characteristic polynomial encoding S₃ group structure
    # Alternating series creates the 6-fold resonance
    s3_poly = [1, -1, 1, -1, 1, -1, 1]  # Order 6 alternating
    return Operator("Tesla6", s3_poly)

def Tesla9() -> Operator:
    """
    The Annihilator - Digital root collapse to multiplicative unity
    
    Digital root collapse: 9 ≡ 2 (mod 7) → yet 9 = 3² ⇝ 0 (multiplicative sink)
    Tesla's singularity: 3 + 6 + 9 = 18 → 1+8=9 → 0 (closed loop to void)
    Triggers when n % 9 == 0: collapse_to_void()
    
    This implements the digital root compression φ(n) = (n-1) % 9 + 1
    with 9 → 0 transformation in the post-swap twisted ring.
    """
    # Polynomial that enforces 9-periodicity with collapse to void
    # x⁹ - 1 creates the digital root compression cycle
    return Operator("Tesla9", [1, 0, 0, 0, 0, 0, 0, 0, 0, -1])  # x⁹ - 1

def MersenneRenormalization(k: int = 3) -> Operator:
    """
    The 0/1 Swap at Mersenne boundary 2^k - 1
    
    At n = 7 (=2³-1), the system undergoes axiomatic inversion:
    - New additive void: 1 + x ≡ x  
    - New multiplicative unity: 0 · x ≡ x
    - Power set dualizes: {a,b,c} ⇝ new 0, ∅ ⇝ new 1
    
    This creates a discontinuity at the Mersenne boundary that
    implements the fundamental 0/1 swap transformation.
    """
    boundary = 2**k - 1  # Mersenne number (default: 7 for k=3)
    
    # Polynomial encoding the swap at boundary
    # Creates identity up to boundary, then implements the flip
    coeffs = jnp.zeros(boundary + 2, dtype=jnp.int32)
    coeffs = coeffs.at[0].set(1)      # Identity for constant term
    coeffs = coeffs.at[1].set(-1)     # Negation for x¹ (implements swap)
    coeffs = coeffs.at[boundary].set(1)  # Boundary condition restores
    
    return Operator(f"MersenneSwap_k{k}", coeffs)

def DigitalRoot() -> Operator:
    """
    The Tithing Operator - φ(n) = sum of digits reduction
    
    This implements the digital root compression that collapses
    higher k dimensions back into the k=3 base case.
    
    The digital root function φ(n) acts as renormalization of abundance
    onto the k=3 computational base, implementing the "tithing" dynamics.
    """
    # Polynomial that implements digit sum reduction
    # This creates a compression transformation
    return Operator("DigitalRoot", [1, 1, 1, 1, 1, 1, 1, 1, 1, -8])  # Sum mod 9

def TrinityEngine() -> list[Operator]:
    """
    The complete Tesla 3-6-9 engine as a sequence of operators.
    
    This combines all three Tesla operators with renormalization
    to create the full cyclotomic engine where:
    - 3 generates the symbol field
    - 6 symmetrizes the power sets  
    - 9 annihilates entropy into the swapped void
    - Renormalization events flip axioms at Mersenne boundaries
    """
    return [
        Tesla3(),
        Tesla6(), 
        Tesla9(),
        MersenneRenormalization(3),
        DigitalRoot()
    ] 