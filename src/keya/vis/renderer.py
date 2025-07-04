"""
Functions for rendering the output of kéya D-C simulations.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Dict, List, Optional, Union

from ..dsl.ast import Glyph


def plot_wavefunction(
    psi: np.ndarray, title: str = "Electron Probability Orbital", alpha_scale: float = 20.0
):
    """
    Renders a 3D wavefunction probability density |ψ|² using a 3D scatter plot.

    Args:
        psi: A 3D numpy array representing the wavefunction.
        title: The title of the plot.
        alpha_scale: A scaling factor for the opacity of the points.
    """
    psi = np.asarray(psi)

    prob_density = np.abs(psi) ** 2
    prob_density /= prob_density.max()

    x, y, z = np.mgrid[-1 : 1 : psi.shape[0] * 1j, -1 : 1 : psi.shape[1] * 1j, -1 : 1 : psi.shape[2] * 1j]

    points = prob_density > 0.01

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    phase = np.angle(psi[points])

    scatter_size = 5
    sc = ax.scatter(
        x[points],
        y[points],
        z[points],
        c=phase,
        s=scatter_size,
        alpha=(prob_density[points] * alpha_scale).clip(0, 1),
        cmap="hsv",
    )

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    # Type checker doesn't recognize 3D axis methods, but this is correct
    ax.set_zlabel("Z")  # type: ignore
    fig.colorbar(sc, label="Phase (radians)")

    plt.show()


def plot_dc_matrix(matrix: np.ndarray, title: str = "D-C Matrix Visualization", 
                   glyph_mapping: Optional[Dict[float, str]] = None):
    """
    Visualize a D-C matrix with glyph symbols and color mapping.
    
    Args:
        matrix: 2D numpy array representing the matrix
        title: Plot title
        glyph_mapping: Optional mapping from values to glyph symbols
    """
    if glyph_mapping is None:
        glyph_mapping = {
            0.0: '∅',   # void
            -1.0: '▽',  # down
            1.0: '△',   # up
            0.5: '⊙',   # unity
            2.0: '⊕',   # flow
        }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Heatmap visualization
    im1 = ax1.imshow(matrix, cmap='viridis', aspect='equal')
    ax1.set_title(f"{title} - Heatmap")
    ax1.set_xlabel("Column")
    ax1.set_ylabel("Row")
    fig.colorbar(im1, ax=ax1, label="Value")
    
    # Glyph visualization
    ax2.imshow(matrix, cmap='viridis', alpha=0.3, aspect='equal')
    
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            # Find closest glyph
            closest_val = min(glyph_mapping.keys(), key=lambda x: abs(x - value))
            glyph = glyph_mapping[closest_val]
            ax2.text(j, i, glyph, ha='center', va='center', fontsize=14, fontweight='bold')
    
    ax2.set_title(f"{title} - Glyphs")
    ax2.set_xlabel("Column")
    ax2.set_ylabel("Row")
    ax2.set_xticks(range(matrix.shape[1]))
    ax2.set_yticks(range(matrix.shape[0]))
    
    plt.tight_layout()
    plt.show()


def plot_dc_transformation(before: np.ndarray, after: np.ndarray, 
                          operation: str = "D-C Transform"):
    """
    Visualize before and after D-C transformation.
    
    Args:
        before: Matrix before transformation
        after: Matrix after transformation
        operation: Name of the operation performed
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Before heatmap
    im1 = axes[0, 0].imshow(before, cmap='viridis', aspect='equal')
    axes[0, 0].set_title("Before - Heatmap")
    fig.colorbar(im1, ax=axes[0, 0])
    
    # After heatmap
    im2 = axes[0, 1].imshow(after, cmap='viridis', aspect='equal')
    axes[0, 1].set_title("After - Heatmap")
    fig.colorbar(im2, ax=axes[0, 1])
    
    # Difference map
    diff = after - before
    im3 = axes[1, 0].imshow(diff, cmap='RdBu', aspect='equal')
    axes[1, 0].set_title("Difference (After - Before)")
    fig.colorbar(im3, ax=axes[1, 0])
    
    # Statistical comparison
    axes[1, 1].bar(['Before', 'After'], [before.mean(), after.mean()], 
                   color=['blue', 'red'], alpha=0.7)
    axes[1, 1].set_title("Mean Values")
    axes[1, 1].set_ylabel("Mean Value")
    
    plt.suptitle(f"{operation} Visualization", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_dc_cycle(matrices: List[np.ndarray], operation: str = "D-C Cycle"):
    """
    Visualize a sequence of matrices through D-C cycle iterations.
    
    Args:
        matrices: List of matrices at each iteration
        operation: Name of the cycle operation
    """
    n_iterations = len(matrices)
    cols = min(4, n_iterations)
    rows = (n_iterations + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, matrix in enumerate(matrices):
        if i < len(axes):
            im = axes[i].imshow(matrix, cmap='viridis', aspect='equal')
            axes[i].set_title(f"Iteration {i}")
            fig.colorbar(im, ax=axes[i])
    
    # Hide unused subplots
    for i in range(len(matrices), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f"{operation} Evolution", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_glyph_distribution(matrix: np.ndarray, title: str = "Glyph Distribution"):
    """
    Plot the distribution of glyph values in a matrix.
    
    Args:
        matrix: Matrix to analyze
        title: Plot title
    """
    glyph_names = {
        0.0: 'Void (∅)',
        -1.0: 'Down (▽)',
        1.0: 'Up (△)',
        0.5: 'Unity (⊙)',
        2.0: 'Flow (⊕)',
    }
    
    # Count occurrences
    unique, counts = np.unique(matrix, return_counts=True)
    
    # Map to glyph names
    labels = []
    for val in unique:
        closest = min(glyph_names.keys(), key=lambda x: abs(x - val))
        if abs(val - closest) < 0.1:  # Close enough
            labels.append(glyph_names[closest])
        else:
            labels.append(f"Value: {val:.2f}")
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, counts, color=['purple', 'blue', 'red', 'orange', 'green'][:len(unique)])
    plt.title(title)
    plt.xlabel("Glyph Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def visualize_dc_program_results(results: Dict[str, any], program_name: str = "D-C Program"):
    """
    Create a comprehensive visualization of D-C program execution results.
    
    Args:
        results: Dictionary of results from program execution
        program_name: Name of the program
    """
    print(f"\n=== {program_name} Visualization ===")
    
    for section_name, section_results in results.items():
        print(f"\nSection: {section_name}")
        
        for i, result in enumerate(section_results):
            if isinstance(result, np.ndarray) and result.ndim == 2:
                plot_dc_matrix(result, f"{section_name} - Result {i+1}")
            elif isinstance(result, (list, tuple)) and len(result) > 1:
                # Check if it's a sequence of matrices (DC cycle)
                is_matrix_sequence = True
                for r in result:
                    if not (isinstance(r, np.ndarray) and r.ndim == 2):
                        is_matrix_sequence = False
                        break
                if is_matrix_sequence and isinstance(result, list):
                    plot_dc_cycle(result, f"{section_name} - Cycle {i+1}")
            elif isinstance(result, (int, float)):
                print(f"  Scalar result {i+1}: {result}")
    
    print(f"=== End {program_name} Visualization ===\n")
