#!/usr/bin/env python3
"""
The Keya Canonical State Resolver.

This module provides the highest-level interface for the engine. Its purpose
is not just to simulate, but to take any initial state and find its most
optimal, canonical, and irreducible form by using the AttractorEngine
configured for maximal descent.
"""
from .kernel.kernel import PascalKernel
from .kernel.attractor import AttractorEngine
from .kernel.operators import GoldenCurvature
import jax.numpy as jnp

class KeyaResolver:
    """
    A high-level resolver to find the canonical form of a state by applying
    the Golden Curvature, guiding the state along its geodesic toward a stable,
    optimal form.
    """
    def __init__(self):
        self.kernel = PascalKernel()
        self.flow_operator = GoldenCurvature()

    def resolve(self, initial_state: jnp.ndarray) -> jnp.ndarray:
        """
        Resolves a state to its most optimal, irreducible form.

        This method uses the AttractorEngine, configured to repeatedly apply
        the 'GoldenCurvature' operator until the state no longer changes.

        Args:
            initial_state: A JAX numpy array representing the state.

        Returns:
            The final, canonical state vector.
        """
        # Define the step function for the Golden Curvature flow.
        def flow_step(state: jnp.ndarray) -> jnp.ndarray:
            return self.kernel.apply_polynomial(state, self.flow_operator.coeffs)

        # Configure the engine to find the end of the flow.
        engine = AttractorEngine(
            step_function=flow_step,
            equals_fn=lambda a, b: bool(jnp.array_equal(a, b))
        )

        # Run the resolution process.
        attractor_info = engine.run(initial_state)

        # The final state of the attractor is the optimal, resolved form.
        return attractor_info.final_state 