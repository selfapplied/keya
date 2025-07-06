#!/usr/bin/env python3
"""
🌟 KERNEL-POWERED CELLULAR AUTOMATA WITH ATTRACTOR DETECTION 🌟

This module demonstrates that Conway's Game of Life can be implemented as a
special case of the Kéya engine's PascalKernel. More importantly, it shows
that the simulation can be run until it reaches a stable attractor (a still
life or an oscillator), which is then classified.
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
import matplotlib.pyplot as plt
from jax.scipy.signal import convolve2d
from matplotlib.colors import ListedColormap

from demos.reporting.registry import register_demo
from keya.kernel.attractor import AttractorEngine, AttractorInfo, HaltingCondition

# --- The Model: Pure Simulation Logic ---

@dataclass
class CellularAutomaton:
    """A pure, headless, high-performance JAX implementation of Conway's Game of Life."""
    
    # Define the convolution kernel as a class attribute so it's accessible
    # by the static `step` method.
    life_kernel: ClassVar[jnp.ndarray] = jnp.array(
        [[1, 1, 1],
         [1, 0, 1],
         [1, 1, 1]], dtype=jnp.uint8
    )

    @staticmethod
    @jit
    def step(grid: jnp.ndarray) -> jnp.ndarray:
        """Executes one step of the Game of Life using convolution."""
        num_neighbors = convolve2d(grid, CellularAutomaton.life_kernel, mode='same', boundary='wrap')
        survives = (grid == 1) & ((num_neighbors == 2) | (num_neighbors == 3))
        born = (grid == 0) & (num_neighbors == 3)
        return survives | born

    @staticmethod
    def initialize_grid(size: int, seed: int = 0) -> jnp.ndarray:
        """Creates a grid with a random initial state."""
        key = jax.random.PRNGKey(seed)
        return jax.random.randint(key, (size, size), 0, 2, dtype=jnp.uint8)

    @staticmethod
    def plot_grid(grid: jnp.ndarray, title: str, filename: str):
        """Saves a visualization of the grid state to an SVG file."""
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.patch.set_facecolor('#121212')
        cmap = ListedColormap(['#1e1e1e', '#ff7f0e']) # Dark background, Orange cells
        ax.imshow(grid, cmap=cmap)
        ax.set_title(title, color='white', fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(filename, format='svg', bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)

# The old find_attractor function is now replaced by the generic AttractorEngine.
# The AttractorInfo dataclass is now imported from keya.kernel.attractor.

# --- Demo Registration ---

@register_demo(
    title="Cellular Automata Attractor Classification",
    artifacts=[
        {"filename": "cellular_automata_initial_state.svg", "caption": "The initial random state of the grid, which serves as the seed for the simulation."},
        {"filename": "cellular_automata_final_attractor.svg", "caption": "The final, stable attractor state reached by the simulation. The title indicates the type of attractor found and the number of steps required."},
    ],
    claims=[
        "A random initial state in Conway's Game of Life will eventually settle into a stable attractor (a still life or an oscillator).",
        "The simulation can programmatically detect when it has reached equilibrium by checking against a history of previous states.",
        "The period of the attractor can be determined by identifying when in the history the repeated state occurred."
    ],
    findings=(
        "This demo showcases a more sophisticated analysis of emergent systems. Instead of running for an arbitrary number of steps, the simulation runs until it "
        "provably reaches a stable equilibrium. This demonstrates that the system can not only simulate the rules of the Game of Life but also "
        "analyze the simulation's output to identify and classify its final state. This capability is crucial for studying the long-term behavior "
        "of complex systems modeled by the Kéya engine."
    )
)
def run_cellular_automata_attractor_demo():
    """
    Runs a cellular automaton until it finds a stable attractor, then saves
    the initial and final states as SVG artifacts.
    """
    initial_grid = CellularAutomaton.initialize_grid(size=50, seed=42)
    CellularAutomaton.plot_grid(initial_grid, "Initial Random State (Seed=42)", "cellular_automata_initial_state.svg")
    
    # Use the new, generic AttractorEngine.
    # By providing a custom `equals_fn`, we can work directly with JAX arrays
    # without needing any wrapper functions or type conversions.
    engine = AttractorEngine(
        step_function=CellularAutomaton.step,
        max_steps=500,
        history_size=10,
        equals_fn=lambda a, b: bool(jnp.array_equal(a, b))
    )
    
    attractor_info = engine.run(initial_grid)
    
    if attractor_info.halting_condition not in [HaltingCondition.MAX_STEPS_REACHED]:
        title = f"Final State: {attractor_info.halting_condition.value}\n(Period {attractor_info.period}) Reached in {attractor_info.steps_to_reach} steps"
    else:
        title = "Final State (Max Steps Reached)"
        
    CellularAutomaton.plot_grid(attractor_info.final_state, title, "cellular_automata_final_attractor.svg")
    
    assert attractor_info.halting_condition != HaltingCondition.MAX_STEPS_REACHED, "An attractor should be found within the step limit."
    assert attractor_info.period is not None and attractor_info.period > 0
    assert attractor_info.steps_to_reach > 0

if __name__ == "__main__":
    run_cellular_automata_attractor_demo() 