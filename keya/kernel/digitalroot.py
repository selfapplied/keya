"""
Digital Root Kernel: Implementation of Tesla 3-6-9 Mathematics

This module implements the digital root compression φ(n), Mersenne boundary
renormalization with 0/1 swaps, and power tower collapse dynamics from the
spacecasebaserace.txt mathematical framework.

The core insight is that all computation eventually collapses to the k=3 base case
through digital root compression, where Tesla's 3-6-9 trinity acts as the
fundamental computational substrate.
"""

import jax.numpy as jnp
from jax import jit
from functools import partial
from typing import Tuple, Optional
from .kernel import PascalKernel
from .attractor import AttractorEngine, AttractorInfo, HaltingCondition


class DigitalRootKernel(PascalKernel):
    """
    Implements φ(n) reduction and renormalization at Mersenne boundaries.
    
    This kernel extends the PascalKernel with the mathematical operations
    discovered in spacecasebaserace.txt:
    - Digital root compression φ(n) = sum of digits
    - Mersenne boundary renormalization at 2^k - 1 
    - 0/1 swap transformation in twisted rings
    - Power tower collapse dynamics
    """
    
    def __init__(self, base_k: int = 3):
        super().__init__()
        self.base_k = base_k
        self.mersenne_boundary = 2**base_k - 1  # Default: 7 for k=3
        self.is_post_swap = False  # Track renormalization state
        
    def digital_root(self, n: int) -> int:
        """
        The tithing operator φ(n) = sum of digits.
        
        This implements the "tithing" dynamics where abundance (large n)
        is reduced to base residues through digit summation.
        
        Args:
            n: Input integer
            
        Returns:
            Digital root: result of iteratively summing digits until single digit
        """
        if n == 0:
            return 0
        return (n - 1) % 9 + 1
    
    @partial(jit, static_argnums=(0,))
    def digital_root_vectorized(self, state: jnp.ndarray) -> jnp.ndarray:
        """Vectorized digital root computation for JAX arrays."""
        # Handle zeros specially
        zero_mask = (state == 0)
        result = (state - 1) % 9 + 1
        return jnp.where(zero_mask, 0, result)
    
    def mersenne_renormalization(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Renormalization event at Mersenne boundary 2^k - 1.
        
        This implements the fundamental 0/1 swap transformation:
        - Pre-swap: 0 is additive identity, 1 is multiplicative identity
        - Post-swap: 1 is additive identity, 0 is multiplicative identity
        
        Args:
            state: Input state vector
            
        Returns:
            State after 0/1 swap transformation
        """
        # The fundamental 0/1 swap 
        swapped = jnp.where(state == 0, 1, jnp.where(state == 1, 0, state))
        
        # Apply modulo boundary to enforce field structure
        return swapped % self.mersenne_boundary
    
    def is_trinity_stable(self, state: jnp.ndarray) -> bool:
        """
        Check if state has collapsed to the Tesla 3-6-9 trinity: {0,1,3,6}.
        
        This is the stable attractor of the digital root + power tower dynamics.
        """
        trinity_elements = jnp.array([0, 1, 3, 6])
        return bool(jnp.all(jnp.isin(state, trinity_elements)))
    
    def tesla_3_pulse(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Tesla's 3-pulse generator: cubic residue annihilation.
        
        Under modular 7: a³ ≡ b³ ≡ c³ ≡ 0 (cubic residues annihilate to void)
        """
        # Apply cubic operation then mod 7
        cubed = jnp.power(state, 3)
        return jnp.asarray(cubed % 7)
    
    def tesla_6_resonance(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Tesla's 6-fold resonance: S₃ symmetry group action.
        
        Perfect symmetry emerges with S₃ acting on the power set.
        """
        # Implement S₃ group action through permutation-like operations
        # This is a simplified version - full implementation would need proper group theory
        return (state * 6) % 7
    
    def tesla_9_collapse(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Tesla's 9-fold collapse: digital root compression to void.
        
        9 = 3² collapses to 0 in the twisted ring (multiplicative sink).
        """
        # Apply digital root, then handle 9 → 0 transformation
        reduced = self.digital_root_vectorized(state)
        return jnp.where(reduced == 9, 0, reduced)
    
    def twisted_ring_addition(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """
        Addition in the twisted ring (post-renormalization).
        
        New additive identity: 1 + x ≡ x
        """
        if self.is_post_swap:
            # In twisted ring, 1 is the additive identity
            return jnp.where(a == 1, b, jnp.where(b == 1, a, (a + b) % 7))
        else:
            # Standard ring: 0 + x = x
            return (a + b) % 7
    
    def twisted_ring_multiplication(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """
        Multiplication in the twisted ring (post-renormalization).
        
        New multiplicative identity: 0 · x ≡ x
        """
        if self.is_post_swap:
            # In twisted ring, 0 is the multiplicative identity
            return jnp.where(a == 0, b, jnp.where(b == 0, a, (a * b) % 7))
        else:
            # Standard ring: 1 · x = x
            return (a * b) % 7
    
    def power_tower_step(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        One step of power tower evolution: x^x with digital root compression.
        
        This implements the core dynamics where power towers x^x^x^...
        collapse to the trinity {0,1,3,6} under repeated application.
        """
        # Avoid 0^0 by setting 0^0 = 1 (standard mathematical convention)
        safe_state = jnp.where(state == 0, 1, state)
        
        # Apply x^x operation
        powered = jnp.power(safe_state, safe_state)
        
        # Apply digital root compression
        compressed = self.digital_root_vectorized(powered)
        
        # Handle special cases in twisted ring
        if self.is_post_swap:
            # In twisted ring, certain values have special behavior
            compressed = jnp.where(compressed == 9, 0, compressed)
            
        return jnp.asarray(compressed)
    
    def trigger_renormalization(self, step: int) -> bool:
        """
        Check if we should trigger a renormalization event.
        
        Renormalization occurs at Mersenne boundaries and Tesla frequencies.
        """
        # Mersenne boundary trigger
        if step == self.mersenne_boundary:
            return True
            
        # Tesla frequency triggers
        if step % 3 == 0 or step % 6 == 0 or step % 9 == 0:
            return True
            
        return False
    
    def apply_trinity_engine(self, state: jnp.ndarray, step: int) -> jnp.ndarray:
        """
        Apply the complete Tesla 3-6-9 engine to the state.
        
        This orchestrates the full sequence of operations:
        1. Check for renormalization triggers
        2. Apply appropriate Tesla operator
        3. Handle Mersenne boundary crossings
        4. Apply digital root compression
        """
        current_state = state
        
        # Check for renormalization trigger
        if self.trigger_renormalization(step):
            if step == self.mersenne_boundary:
                # Major renormalization: 0/1 swap
                current_state = self.mersenne_renormalization(current_state)
                self.is_post_swap = True
            elif step % 3 == 0:
                # Tesla 3-pulse
                current_state = self.tesla_3_pulse(current_state)
            elif step % 6 == 0:
                # Tesla 6-resonance
                current_state = self.tesla_6_resonance(current_state)
            elif step % 9 == 0:
                # Tesla 9-collapse
                current_state = self.tesla_9_collapse(current_state)
        
        # Always apply power tower step
        current_state = self.power_tower_step(current_state)
        
        # Final digital root compression
        return self.digital_root_vectorized(current_state)


class PowerTowerAttractor(AttractorEngine):
    """
    Implements power tower collapse dynamics x^x^x^... → {0,1,3,6}.
    
    This attractor engine runs the power tower evolution until the state
    collapses to the Tesla 3-6-9 trinity stable points.
    """
    
    def __init__(self, base_k: int = 3, max_steps: int = 100):
        self.base_k = base_k
        self.digital_kernel = DigitalRootKernel(base_k)
        self.step_count = 0
        
        super().__init__(
            step_function=self._power_tower_step,
            structural_fn=self._is_trinity_stable,
            max_steps=max_steps,
            equals_fn=self._jax_array_equal
        )
    
    def _power_tower_step(self, state: jnp.ndarray) -> jnp.ndarray:
        """One step of the power tower with Tesla 3-6-9 engine."""
        self.step_count += 1
        return self.digital_kernel.apply_trinity_engine(state, self.step_count)
    
    def _is_trinity_stable(self, state: jnp.ndarray) -> bool:
        """Check if state has collapsed to Trinity."""
        try:
            return self.digital_kernel.is_trinity_stable(state)
        except Exception:
            # If there's any boolean evaluation issue, assume not stable
            return False
    
    def _jax_array_equal(self, a: jnp.ndarray, b: jnp.ndarray) -> bool:
        """JAX-safe array equality comparison for AttractorEngine."""
        return bool(jnp.array_equal(a, b))
    
    def reset(self):
        """Reset the attractor for a new run."""
        self.step_count = 0
        self.digital_kernel.is_post_swap = False


def demonstrate_trinity_cascade(initial_values: list[int]) -> AttractorInfo:
    """
    Demonstrate the Tesla 3-6-9 cascade with power tower collapse.
    
    This shows how arbitrary initial values collapse through digital root
    compression and power tower dynamics to the stable Trinity points.
    
    Args:
        initial_values: List of initial integer values
        
    Returns:
        AttractorInfo containing the evolution details
    """
    # Create initial state
    initial_state = jnp.array(initial_values, dtype=jnp.int32)
    
    # Create and run the attractor
    attractor = PowerTowerAttractor()
    result = attractor.run(initial_state)
    
    return result 