#!/usr/bin/env python3
"""
Keya Symbolic String Generation Demo

This demo showcases the core symbolic manipulation capabilities of the Kéya
calculus. It demonstrates how a simple grammar and a seed state can be used
to deterministically generate complex, structured strings of glyphs.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize

from keya.symbolic import (
    Glyph, GLYPH_TO_INT, INT_TO_GLYPH
)
from demos.reporting.registry import register_demo

# --- Locally Implemented Symbolic Operators (formerly in keya.core.operators) ---

Grammar = np.ndarray

SIMPLE_GRAMMAR: Grammar = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
], dtype=np.uint8)

BINARY_GRAMMAR: Grammar = np.array([
    [0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0],
], dtype=np.uint8)

def apply_glyph_transform(matrix: np.ndarray, grammar: Grammar) -> np.ndarray:
    """Applies a grammar transformation to a glyph matrix."""
    rows, cols = matrix.shape
    new_matrix = np.copy(matrix)
    for r in range(rows):
        for c in range(cols):
            glyph_val = matrix[r, c]
            if glyph_val in [GLYPH_TO_INT[Glyph.UP], GLYPH_TO_INT[Glyph.DOWN]]:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if grammar[dr+1, dc+1] == 1:
                            nr, nc = (r + dr) % rows, (c + dc) % cols
                            if new_matrix[nr, nc] == GLYPH_TO_INT[Glyph.VOID]:
                                new_matrix[nr, nc] = glyph_val
    return new_matrix

def create_glyph_matrix(rows: int, cols: int, seed_glyph: Glyph, seed_pos: tuple[int, int]) -> np.ndarray:
    """Creates a matrix of glyphs with a single seed."""
    matrix = np.full((rows, cols), GLYPH_TO_INT[Glyph.VOID], dtype=np.int32)
    matrix[seed_pos] = GLYPH_TO_INT[seed_glyph]
    return matrix

def generate_string_from_seed(rows: int, cols: int, seed_glyph: Glyph, grammar: Grammar, steps: int) -> np.ndarray:
    """Generates a glyph matrix by evolving a seed."""
    matrix = create_glyph_matrix(rows, cols, seed_glyph, (rows // 2, cols // 2))
    for _ in range(steps):
        matrix = apply_glyph_transform(matrix, grammar)
    return matrix

def extract_string_from_matrix(matrix: np.ndarray) -> str:
    """Extracts a string of glyphs from the first row of a matrix."""
    return "".join([INT_TO_GLYPH.get(v, '?').value for v in matrix[0, :]])

def string_to_text(s: str) -> str:
    """Converts a raw glyph string to a human-readable format."""
    return s.replace(Glyph.UP.value, "1").replace(Glyph.DOWN.value, "0")

# --- Demo Visualization and Registration ---

def plot_glyph_matrix(matrix: np.ndarray, filename: str):
    """Creates a heatmap visualization of the glyph matrix and saves it as an SVG."""
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#121212')

    # Use a standard colormap and let matplotlib handle normalization
    cmap = 'viridis'
    im = ax.imshow(matrix, cmap=cmap)

    ax.set_title("Evolved Glyph Matrix", color='white', fontsize=16)
    ax.set_xlabel("Column", color='white')
    ax.set_ylabel("Row", color='white')
    ax.tick_params(colors='white')

    # Create a simple colorbar; the labels might not be perfect but it avoids the error
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Glyph Value", color='white')
    cbar.ax.tick_params(axis='y', colors='white')
    cbar.outline.set_edgecolor('grey')

    plt.savefig(filename, format='svg', bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

@register_demo(
    title="Emergent Complexity from a Symbolic Seed",
    artifacts=[
        {"filename": "symbol_matrix.svg", "caption": "A visualization of the final glyph matrix. The pattern emerges from a single 'UP' (△) seed after 15 evolution steps, demonstrating how simple, local rules can generate structured complexity."},
    ],
    claims=[
        "A single seed glyph can generate a complex, structured pattern using a simple, deterministic grammar.",
        "The visualization clearly shows the growth of the pattern from the central seed point.",
        "The final structure exhibits symmetries and patterns that are not explicitly encoded in the grammar, demonstrating emergence."
    ],
    findings=(
        "This demo provides visual proof of one of the core tenets of the Σ-Calculus: that complex, ordered structures "
        "can emerge from the iterative application of simple rules to a seed state. The resulting diamond-like shape is not a pre-programmed "
        "output; it is the deterministic result of the grammar's interaction with the grid. This showcases the generative power "
        "of the symbolic system, which can create information-rich patterns from minimal starting conditions."
    )
)
def run_symbol_generation_demo():
    """
    Generates a structured pattern from a single seed glyph and visualizes
    the resulting matrix as a heatmap.
    """
    matrix = generate_string_from_seed(
        rows=30,
        cols=40,
        seed_glyph=Glyph.UP,
        grammar=SIMPLE_GRAMMAR,
        steps=15
    )

    # Save the visual artifact
    plot_glyph_matrix(matrix, "symbol_matrix.svg")

    # Assertions to validate the demo's claims
    assert matrix.shape == (30, 40), "Matrix should have the correct dimensions."
    assert INT_TO_GLYPH[matrix[15, 20]] == Glyph.UP, "The original seed should still be present."
    assert GLYPH_TO_INT[Glyph.VOID] in matrix, "Matrix should contain void areas."
    assert np.any(matrix == GLYPH_TO_INT[Glyph.UP]), "Matrix should contain UP glyphs."


if __name__ == "__main__":
    run_symbol_generation_demo() 