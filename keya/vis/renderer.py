"""
Functions for rendering the output of kéya simulations.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import os
import subprocess
import sys

# Terminal capability detection will be handled locally in each function

# Set non-interactive backend for file saving by default
try:
    matplotlib.use('Agg')
except Exception:
    pass  # Ignore if backend already set

def _ensure_output_dir() -> Path:
    """Ensure .out directory exists and return path."""
    out_dir = Path('.out')
    out_dir.mkdir(exist_ok=True)
    return out_dir

def _check_terminal_image_support() -> bool:
    """Check if terminal supports image display using capability detection."""
    # Try to detect actual capabilities rather than hardcoded terminal names
    
    # Check for Kitty graphics protocol
    if _probe_kitty_graphics():
        return True
    
    # Check for iTerm2 inline images
    if _probe_iterm2_images():
        return True
    
    # Check for sixel support
    if _probe_sixel_support():
        return True
    
    # Check for general image display capability
    if _probe_image_display():
        return True
        
    return False

def _probe_kitty_graphics() -> bool:
    """Probe for Kitty graphics protocol support."""
    try:
        import termios
        import tty
        import select
        
        # Save terminal settings
        if not sys.stdin.isatty():
            return False
            
        old_settings = termios.tcgetattr(sys.stdin)
        
        try:
            # Send Kitty graphics query
            sys.stdout.write('\x1b_Gi=1,a=q;\x1b\\')
            sys.stdout.flush()
            
            # Check for response (with timeout)
            if select.select([sys.stdin], [], [], 0.1)[0]:
                tty.setcbreak(sys.stdin)
                response = sys.stdin.read(1)
                return response == '\x1b'
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            
    except (ImportError, OSError, AttributeError):
        pass
    
    # Fallback: check environment hint
    return os.environ.get('TERM', '').startswith('xterm-kitty')

def _probe_iterm2_images() -> bool:
    """Probe for iTerm2 inline image support."""
    try:
        # Check for iTerm2 session ID (reliable indicator)
        if 'ITERM_SESSION_ID' in os.environ:
            return True
        
        # Check for iTerm2 profile (another indicator)
        if 'ITERM_PROFILE' in os.environ:
            return True
            
    except Exception:
        pass
    
    return False

def _probe_sixel_support() -> bool:
    """Probe for sixel graphics support."""
    try:
        import termios
        import tty
        import select
        
        if not sys.stdin.isatty():
            return False
            
        old_settings = termios.tcgetattr(sys.stdin)
        
        try:
            # Query terminal capabilities
            sys.stdout.write('\x1b[c')  # Device Attributes query
            sys.stdout.flush()
            
            if select.select([sys.stdin], [], [], 0.1)[0]:
                tty.setcbreak(sys.stdin)
                response = sys.stdin.read(20)  # Read response
                # Look for sixel capability indicator (4;...)
                if '4;' in response:
                    return True
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            
    except (ImportError, OSError, AttributeError):
        pass
    
    # Fallback: check environment variables
    term_features = os.environ.get('TERM_FEATURES', '')
    return 'sixel' in term_features

def _probe_image_display() -> bool:
    """Probe for general image display capability."""
    # Check if we're in a graphical environment
    if os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
        # We're in a graphical session, images might be supported
        return True
    
    # Check for terminal multiplexers that might support images
    if os.environ.get('TMUX') and _probe_tmux_image_support():
        return True
    
    return False

def _probe_tmux_image_support() -> bool:
    """Check if tmux session supports image passthrough."""
    try:
        # Check tmux version and configuration
        result = subprocess.run(['tmux', 'show-options', '-g', 'allow-passthrough'],
                              capture_output=True, text=True, timeout=1)
        return 'on' in result.stdout.lower()
    except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False

def _save_or_show(fig, title: str, save_to: Optional[str] = None, interactive: bool = False):
    """Handle saving plot to file or showing interactively."""
    if save_to is None:
        out_dir = _ensure_output_dir()
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title.replace(' ', '_')
        save_path = out_dir / f"{safe_title}.png"
    else:
        save_path = Path(save_to)
    
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    print(f"📊 Plot saved: {save_path}")
    
    if interactive and _check_terminal_image_support():
        plt.show()
    else:
        plt.close(fig)

def plot_wavefunction(
    psi: np.ndarray, 
    title: str = "Electron Probability Orbital", 
    alpha_scale: float = 20.0,
    save_to: Optional[str] = None,
    interactive: bool = False
):
    """Renders a 3D wavefunction probability density |ψ|² using a 3D scatter plot."""
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
        x[points], y[points], z[points],
        c=phase, s=scatter_size,
        alpha=(prob_density[points] * alpha_scale).clip(0, 1),
        cmap="hsv",
    )

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.colorbar(sc, label="Phase (radians)")

    _save_or_show(fig, "wavefunction", save_to, interactive)

def plot_dc_matrix(matrix: np.ndarray, 
                   title: str = "Matrix Visualization", 
                   glyph_mapping: Optional[Dict[float, str]] = None,
                   save_to: Optional[str] = None,
                   interactive: bool = False):
    """Visualize a matrix with glyph symbols and color mapping."""
    if glyph_mapping is None:
        glyph_mapping = {0.0: '∅', -1.0: '▽', 1.0: '△', 0.5: '⊙', 2.0: '⊕'}
    
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
            closest_val = min(glyph_mapping.keys(), key=lambda x: abs(x - value))
            glyph = glyph_mapping[closest_val]
            ax2.text(j, i, glyph, ha='center', va='center', fontsize=14, fontweight='bold')
    
    ax2.set_title(f"{title} - Glyphs")
    ax2.set_xlabel("Column")
    ax2.set_ylabel("Row")
    ax2.set_xticks(range(matrix.shape[1]))
    ax2.set_yticks(range(matrix.shape[0]))
    
    plt.tight_layout()
    _save_or_show(fig, f"matrix_{title}", save_to, interactive)

def plot_dc_transformation(before: np.ndarray, after: np.ndarray, 
                          operation: str = "Transform",
                          save_to: Optional[str] = None,
                          interactive: bool = False):
    """Visualize before and after transformation."""
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
    _save_or_show(fig, f"transform_{operation}", save_to, interactive)

def plot_dc_cycle(matrices: List[np.ndarray], operation: str = "Cycle",
                  save_to: Optional[str] = None,
                  interactive: bool = False):
    """Visualize a sequence of matrices through cycle iterations."""
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
    _save_or_show(fig, f"cycle_{operation}", save_to, interactive)

def plot_glyph_distribution(matrix: np.ndarray, title: str = "Glyph Distribution",
                           save_to: Optional[str] = None,
                           interactive: bool = False):
    """Plot the distribution of glyph values in a matrix."""
    glyph_names = {
        0.0: 'Void (∅)', -1.0: 'Down (▽)', 1.0: 'Up (△)', 
        0.5: 'Unity (⊙)', 2.0: 'Flow (⊕)',
    }
    
    # Count occurrences
    unique, counts = np.unique(matrix, return_counts=True)
    
    # Map to glyph names
    labels = []
    for val in unique:
        closest = min(glyph_names.keys(), key=lambda x: abs(x - val))
        if abs(val - closest) < 0.1:
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
    fig = plt.gcf()
    _save_or_show(fig, f"distribution_{title}", save_to, interactive)

def visualize_dc_program_results(results: Dict[str, Any], program_name: str = "Program",
                                save_to_dir: Optional[str] = None,
                                interactive: bool = False):
    """Create a comprehensive visualization of program execution results."""
    print(f"\n=== {program_name} Visualization ===")
    
    if save_to_dir is None:
        save_dir = _ensure_output_dir()
    else:
        save_dir = Path(save_to_dir)
        save_dir.mkdir(exist_ok=True)
    
    for section_name, section_results in results.items():
        print(f"\nSection: {section_name}")
        
        for i, result in enumerate(section_results):
            if isinstance(result, np.ndarray) and result.ndim == 2:
                safe_section = "".join(c for c in section_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_section = safe_section.replace(' ', '_')
                filename = save_dir / f"{safe_section}_result_{i+1}.png"
                plot_dc_matrix(result, f"{section_name} - Result {i+1}", 
                             save_to=str(filename), interactive=interactive)
            elif isinstance(result, (list, tuple)) and len(result) > 1:
                # Check if it's a sequence of matrices (DC cycle)
                if all(isinstance(r, np.ndarray) and r.ndim == 2 for r in result):
                    safe_section = "".join(c for c in section_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    safe_section = safe_section.replace(' ', '_')
                    filename = save_dir / f"{safe_section}_cycle_{i+1}.png"
                    matrices = list(result) if isinstance(result, tuple) else result
                    plot_dc_cycle(matrices, f"{section_name} - Cycle {i+1}",
                                save_to=str(filename), interactive=interactive)
            elif isinstance(result, (int, float)):
                print(f"  Scalar result {i+1}: {result}")
    
    print(f"=== End {program_name} Visualization ===")
    print(f"📁 All plots saved to: {save_dir}")
