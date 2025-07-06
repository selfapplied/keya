"""
Pascal's Triangle and Combinatorial Iterators Demo

This demonstrates the key insight: combinatorics are like having two iterators 
adding up terms across a row forwards and backwards.

In Pascal's triangle: C(n,k) = C(n-1,k-1) + C(n-1,k)
This can be seen as:
- Iterator 1: moves "backwards" from position k to k-1 in row above
- Iterator 2: stays at position k in row above  
- Sum: C(n-1,k-1) + C(n-1,k) = C(n,k)

This dual-iterator pattern connects to Keya's Ϟ§ operators and creates
the fractal Sierpinski triangle when viewed modulo 2.
"""

import os
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import jax.numpy as jnp
from matplotlib.colors import ListedColormap

from keya.symbolic import Glyph
from demos.reporting.registry import register_demo


def build_pascal_triangle(max_rows: int) -> np.ndarray:
    """Builds Pascal's triangle using the combinatorial formula."""
    triangle = np.zeros((max_rows, max_rows), dtype=np.int64)
    for n in range(max_rows):
        for k in range(n + 1):
            triangle[n, k] = comb(n, k, exact=True)
    return triangle

def plot_pascal_and_sierpinski(pascal_triangle: np.ndarray, sierpisnki_triangle: np.ndarray, filename: str):
    """Plots the standard and mod 2 triangles side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor('#121212')
    fig.suptitle("Pascal's Triangle and its Modulo 2 Form (Sierpinski)", color='white', fontsize=18)

    for ax in axes:
        ax.set_xlabel("k (column)", color='white')
        ax.set_ylabel("n (row)", color='white')
        ax.tick_params(colors='white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('grey')
        ax.spines['left'].set_color('grey')

    # Pascal's Triangle
    im1 = axes[0].imshow(pascal_triangle, cmap='viridis', origin='upper')
    axes[0].set_title("Standard C(n,k)", color='white')
    fig.colorbar(im1, ax=axes[0])

    # Sierpinski Triangle
    axes[1].imshow(sierpisnki_triangle, cmap=ListedColormap(['#202020', '#F0F0F0']), origin='upper')
    axes[1].set_title("C(n,k) mod 2", color='white')
    
    plt.savefig(filename, format='svg', bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

def plot_iterator_contribution(pascal_triangle: np.ndarray, filename: str):
    """Creates a heatmap of the dual-iterator contribution ratio."""
    size = pascal_triangle.shape[0]
    iterator_ratios = np.full((size, size), np.nan)
    for n in range(1, size):
        for k in range(1, n):
            back_val = pascal_triangle[n - 1, k - 1]
            total = pascal_triangle[n, k]
            if total > 0:
                iterator_ratios[n, k] = back_val / total
    
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('#121212')
    im = ax.imshow(iterator_ratios, cmap='RdBu', origin='upper', vmin=0, vmax=1)
    
    ax.set_title("Dual Iterator Contribution Ratio\n(C(n-1,k-1) / C(n,k))", color='white', fontsize=16)
    ax.set_xlabel("k (column)", color='white')
    ax.set_ylabel("n (row)", color='white')
    ax.tick_params(colors='white')
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Backward Iterator Contribution %", color='white')
    cbar.ax.yaxis.set_tick_params(color='white')

    plt.savefig(filename, format='svg', bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

def plot_pascal_fft(pascal_triangle: np.ndarray, row_index: int, filename: str):
    """Plots the FFT of a specific row of Pascal's triangle."""
    pascal_vector = pascal_triangle[row_index]
    fft_result = np.abs(np.fft.fft(pascal_vector, n=256)) # Pad for smooth plot

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#121212')
    ax.set_facecolor('#2e2e2e')
    
    ax.plot(fft_result, color='cyan')
    ax.set_yscale('log')
    ax.set_title(f"FFT Magnitude of Pascal's Triangle Row n={row_index}", color='white')
    ax.set_xlabel("Frequency Component", color='white')
    ax.set_ylabel("Magnitude (log scale)", color='white')
    ax.tick_params(colors='white')
    ax.grid(True, which="both", linestyle='--', color='grey', alpha=0.5)

    plt.savefig(filename, format='svg', bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

@register_demo(
    title="Multi-Faceted Analysis of Pascal's Triangle",
    artifacts=[
        {"filename": "pascal_sierpinski.svg", "caption": "Side-by-side view of Pascal's triangle and its modulo 2 form, revealing the emergent Sierpinski fractal."},
        {"filename": "pascal_iterator_contribution.svg", "caption": "A heatmap visualizing the contribution ratio of the two 'iterators' from the previous row, illustrating the core combinatorial construction C(n,k) = C(n-1,k-1) + C(n-1,k)."},
        {"filename": "pascal_row_fft.svg", "caption": "The frequency spectrum of a single row, revealing patterns in its coefficients."}
    ],
    claims=[
        "Pascal's triangle contains the Sierpinski fractal in its modulo 2 structure.",
        "The construction of Pascal's triangle can be understood as a dual-iterator system.",
        "The coefficients of a given row have a distinct and analyzable frequency spectrum.",
        "These interconnected patterns demonstrate how simple generative rules produce layers of mathematical structure."
    ],
    findings=(
        "This demo successfully visualizes three distinct but interconnected mathematical properties of Pascal's "
        "triangle. The first artifact shows the classic emergence of the Sierpinski fractal. The second provides "
        "a novel visualization of the 'dual iterator' insight, the core recursive definition of the triangle. The "
        "third reveals the frequency-domain structure of the coefficients. Together, they demonstrate that even "
        "a simple combinatorial object like Pascal's triangle is rich with complex patterns, reinforcing the "
        "Kéya project's core thesis that simple rules can generate profound, multi-faceted structures."
    )
)
def run_pascal_analysis_demo():
    """
    Demonstrates multiple interconnected mathematical patterns within Pascal's
    triangle, including the emergence of the Sierpinski fractal, the dual-iterator
    construction, and the frequency spectrum of its coefficients.
    """
    max_rows = 64
    
    # --- Generate Data ---
    pascal_triangle = build_pascal_triangle(max_rows)
    sierpinski_triangle = pascal_triangle % 2

    # --- Create Visualizations ---
    plot_pascal_and_sierpinski(pascal_triangle, sierpinski_triangle, "pascal_sierpinski.svg")
    plot_iterator_contribution(pascal_triangle, "pascal_iterator_contribution.svg")
    plot_pascal_fft(pascal_triangle, row_index=max_rows // 2, filename="pascal_row_fft.svg")


if __name__ == "__main__":
    run_pascal_analysis_demo() 