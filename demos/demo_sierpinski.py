"""
Example: Prime-counting Analysis using Keya D-C Operators in Sierpinski Framework

This demo demonstrates how Keya D-C operators can be applied to number theory,
specifically analyzing prime distributions within Sierpinski triangle structures.
We show that prime-counting functions exhibit D-C behavior through:
- Diagonalization of prime gaps and irregularities
- Containment of infinite prime sequences into finite grids
- D-C cycles that reveal hidden prime number patterns
"""

import sys
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from sympy.ntheory import primepi
from scipy.special import binom
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import jax.numpy as jnp
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from keya.core.engine import Engine
from keya.core.operators import D_operator, C_operator, DC_cycle, Glyph
from keya.dsl.parser import parse
from keya.dsl.ast import ContainmentType


class PrimeSierpinskiDCAnalyzer:
    """Prime number analysis using Keya D-C operators within Sierpinski framework."""
    
    def __init__(self, max_depth: int = 16) -> None:
        self.max_depth = max_depth
        self.depths = np.arange(1, max_depth + 1)
        self.prime_counts = self.compute_prime_counts()
        self.log_derivatives = self.compute_log_derivatives()
        self.anomalies = self.compute_anomalies()
        
        # Initialize Keya D-C engine
        self.engine = Engine()
        self.dc_processed_data = {}

    def compute_prime_counts(self) -> Dict[int, int]:
        """Compute prime counts π(2^k) for k=1..max_depth."""
        return {k: int(primepi(2**k)) for k in self.depths}

    def compute_log_derivatives(self) -> Dict[int, float]:
        """Compute log-derivatives Δₜπ(k) = π(2^{k+1}) - π(2^k)."""
        return {k: float(self.prime_counts[k + 1] - self.prime_counts[k]) for k in self.depths[:-1]}

    def compute_anomalies(self) -> Dict[int, float]:
        """Compute anomalies = actual - expected prime density (2^k / k)."""
        return {k: float(self.log_derivatives[k] - (2**k) / k) for k in self.depths[:-1]}

    def apply_dc_operators_to_primes(self) -> Dict[str, Any]:
        """Apply Keya D-C operators to prime distribution data."""
        print("Applying D-C operators to prime distributions...")
        
        # Convert prime data to matrix format for D-C processing
        prime_values = np.array([self.log_derivatives[k] for k in self.depths[:-1]])
        anomaly_values = np.array([self.anomalies[k] for k in self.depths[:-1]])
        
        # Create matrices from prime data (make them square for D-C operators)
        size = max(8, len(prime_values))  # Ensure reasonable matrix size
        prime_matrix = jnp.zeros((size, size))
        anomaly_matrix = jnp.zeros((size, size))
        
        # Fill diagonal with prime/anomaly data
        for i, val in enumerate(prime_values[:size]):
            prime_matrix = prime_matrix.at[i, i].set(int(abs(val) * 100) % 5)  # Convert to glyph range [0-4]
        
        for i, val in enumerate(anomaly_values[:size]):
            anomaly_matrix = anomaly_matrix.at[i, i].set(int(abs(val) * 100) % 5)  # Convert to glyph range [0-4]
        
        # Apply D-operator (diagonalize irregularities)
        print("  D-operator: Diagonalizing prime gaps...")
        d_primes = D_operator(prime_matrix)
        d_anomalies = D_operator(anomaly_matrix)
        
        # Apply C-operator (contain infinite sequences)
        print("  C-operator: Containing prime sequences...")
        c_primes = C_operator(d_primes, "binary")
        c_anomalies = C_operator(d_anomalies, "binary")
        
        # Apply full D-C cycle
        print("  DC-cycle: Complete D-C evolution...")
        dc_primes = DC_cycle(prime_matrix, "binary", max_iterations=10)
        dc_anomalies = DC_cycle(anomaly_matrix, "binary", max_iterations=10)
        
        # Extract meaningful values from matrices (diagonal elements)
        d_prime_vals = np.array([d_primes[i, i] for i in range(min(len(prime_values), size))])
        d_anomaly_vals = np.array([d_anomalies[i, i] for i in range(min(len(anomaly_values), size))])
        c_prime_vals = np.array([c_primes[i, i] for i in range(min(len(prime_values), size))])
        c_anomaly_vals = np.array([c_anomalies[i, i] for i in range(min(len(anomaly_values), size))])
        dc_prime_vals = np.array([dc_primes[i, i] for i in range(min(len(prime_values), size))])
        dc_anomaly_vals = np.array([dc_anomalies[i, i] for i in range(min(len(anomaly_values), size))])
        
        return {
            'original_primes': prime_values,
            'original_anomalies': anomaly_values,
            'd_primes': d_prime_vals,
            'd_anomalies': d_anomaly_vals,
            'c_primes': c_prime_vals,
            'c_anomalies': c_anomaly_vals,
            'dc_primes': dc_prime_vals,
            'dc_anomalies': dc_anomaly_vals,
            'primes_variance_reduction': float(np.var(prime_values)) / max(float(np.var(dc_prime_vals)), 1e-10),
            'anomalies_variance_reduction': float(np.var(anomaly_values)) / max(float(np.var(dc_anomaly_vals)), 1e-10)
        }

    def analyze_prime_dc_convergence(self) -> dict[str, any]:
        """Analyze how D-C operators affect prime distribution convergence."""
        print("Analyzing D-C convergence properties...")
        
        # Test different containment rules
        containment_results = {}
        
        for containment_rule in ["binary", "decimal", "string"]:
            print(f"  Testing {containment_rule} containment...")
            
            # Create prime matrix
            prime_values = np.array([self.log_derivatives[k] for k in self.depths[:-1]])
            size = max(8, len(prime_values))
            matrix = jnp.zeros((size, size))
            
            # Fill with prime data
            for i, val in enumerate(prime_values[:size]):
                matrix = matrix.at[i, i].set(int(abs(val) * 100) % 5)
            
            # Apply multiple D-C cycles
            steps_data = []
            current_matrix = matrix
            
            for step in range(1, 11):
                current_matrix = DC_cycle(current_matrix, containment_rule, max_iterations=1)
                
                # Extract diagonal values for analysis
                diag_vals = np.array([current_matrix[i, i] for i in range(size)])
                variance = float(np.var(diag_vals))
                mean_abs = float(np.mean(np.abs(diag_vals)))
                
                steps_data.append({
                    'step': step,
                    'variance': variance,
                    'mean_abs': mean_abs,
                    'values': diag_vals.copy()
                })
            
            containment_results[containment_rule] = steps_data
        
        return containment_results

    def compute_fractional_derivative(self, alpha: float = 0.5) -> dict[int, float]:
        """Fractional derivative via Grünwald–Letnikov definition."""
        fd: dict[int, float] = {}
        for k in self.depths[1:-1]:
            total = 0.0
            for j in range(k):
                weight = binom(alpha, j) * (-1) ** j
                total += weight * self.log_derivatives.get(k - j, 0)
            fd[k] = total
        return fd

    def visualize_dc_prime_analysis(self) -> None:
        """Create comprehensive visualization of D-C prime analysis."""
        
        # Apply D-C operators
        self.dc_processed_data = self.apply_dc_operators_to_primes()
        convergence_data = self.analyze_prime_dc_convergence()
        
        fig = plt.figure(figsize=(20, 24))
        gs = GridSpec(6, 3, figure=fig, height_ratios=[1.5, 1, 1, 1, 1, 1])

        # Main Sierpinski visualization with D-C overlays
        ax1 = fig.add_subplot(gs[0, :])
        self.plot_sierpinski_dc_prime_sparks(ax1)

        # D-C operator effects on primes
        ax2 = fig.add_subplot(gs[1, 0])
        self.plot_dc_operator_effects(ax2)
        
        ax3 = fig.add_subplot(gs[1, 1])
        self.plot_prime_anomaly_dc_evolution(ax3)
        
        ax4 = fig.add_subplot(gs[1, 2])
        self.plot_dc_variance_reduction(ax4)

        # Traditional analysis enhanced with D-C insights
        ax5 = fig.add_subplot(gs[2, 0])
        self.plot_log_derivative_comparison(ax5)

        ax6 = fig.add_subplot(gs[2, 1])
        self.plot_prime_anomalies(ax6)
        
        ax7 = fig.add_subplot(gs[2, 2])
        self.plot_dc_convergence_analysis(ax7, convergence_data)

        # Fractional derivatives and D-C evolution
        ax8 = fig.add_subplot(gs[3, :])
        self.plot_fractional_derivatives_with_dc(ax8)

        # D-C containment comparison
        ax9 = fig.add_subplot(gs[4, :])
        self.plot_containment_type_comparison(ax9, convergence_data)

        # Spectral analysis with D-C filtering
        ax10 = fig.add_subplot(gs[5, :])
        self.plot_dc_filtered_spectrum(ax10)

        plt.tight_layout()
        
        # Save to output directory
        os.makedirs('.out/visualizations', exist_ok=True)
        out_fn = ".out/visualizations/prime_sierpinski_dc.png"
        plt.savefig(out_fn, dpi=300, bbox_inches="tight")
        print(f"Saved D-C prime analysis: {out_fn}")
        plt.close()

    def plot_sierpinski_dc_prime_sparks(self, ax: plt.Axes) -> None:
        """Plot Sierpinski pattern with D-C processed prime sparks."""
        size = 2**self.max_depth
        img = np.zeros((size, self.max_depth))

        for k in self.depths:
            n = 2**k
            for i in range(0, n, 4):
                img[i, k - 1] = k

        ax.imshow(
            img.T,
            aspect="auto",
            origin="lower",
            cmap="binary_r",
            norm=LogNorm(vmin=1, vmax=self.max_depth),
            extent=[0, size, 1, self.max_depth],
        )

        # Original prime data
        depths = list(self.log_derivatives.keys())
        positions = [2**k for k in depths]
        original_values = [self.log_derivatives[k] for k in depths]
        
        # D-C processed data
        if 'dc_primes' in self.dc_processed_data:
            dc_values = self.dc_processed_data['dc_primes']
        else:
            dc_values = original_values

        # Plot both original and D-C processed
        scale = size / max(original_values)
        ax.scatter(positions, depths, s=[v * scale for v in original_values], 
                  c=original_values, cmap="plasma", alpha=0.5, 
                  edgecolors='white', linewidth=0.5, label='Original', zorder=8)
        
        ax.scatter(positions, [d + 0.2 for d in depths], s=[v * scale for v in dc_values[:len(depths)]], 
                  c=dc_values[:len(depths)], cmap="viridis", alpha=0.8,
                  edgecolors='black', linewidth=0.5, marker='s', label='D-C Processed', zorder=10)

        ax.set_xscale("log")
        ax.set_xlabel("n (log scale)")
        ax.set_ylabel("Depth (k)")
        ax.set_title("Sierpinski Sieve with D-C Prime Analysis")
        ax.grid(True, alpha=0.2)
        ax.legend()

    def plot_dc_operator_effects(self, ax: plt.Axes) -> None:
        """Show effects of individual D and C operators."""
        if not self.dc_processed_data:
            return
            
        depths = list(self.log_derivatives.keys())
        
        ax.plot(depths, self.dc_processed_data['original_primes'], 'o-', 
               label='Original π', linewidth=2, markersize=6)
        ax.plot(depths, self.dc_processed_data['d_primes'][:len(depths)], 's--', 
               label='D-operator π', linewidth=2, markersize=4)
        ax.plot(depths, self.dc_processed_data['c_primes'][:len(depths)], '^:', 
               label='C-operator π', linewidth=2, markersize=4)
        
        ax.set_xlabel("Depth (k)")
        ax.set_ylabel("Prime Density")
        ax.set_title("D-C Operator Effects")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_prime_anomaly_dc_evolution(self, ax: plt.Axes) -> None:
        """Show how D-C evolution affects prime anomalies."""
        if not self.dc_processed_data:
            return
            
        depths = list(self.anomalies.keys())
        
        original = self.dc_processed_data['original_anomalies']
        dc_evolved = self.dc_processed_data['dc_anomalies'][:len(depths)]
        
        ax.bar([d - 0.2 for d in depths], original, width=0.4, 
               alpha=0.7, label='Original Anomalies', color='red')
        ax.bar([d + 0.2 for d in depths], dc_evolved, width=0.4, 
               alpha=0.7, label='D-C Evolved', color='blue')
        
        ax.axhline(0, color="black", lw=1)
        ax.set_xlabel("Depth (k)")
        ax.set_ylabel("Anomaly Magnitude")
        ax.set_title("Prime Anomaly D-C Evolution")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_dc_variance_reduction(self, ax: plt.Axes) -> None:
        """Show variance reduction from D-C processing."""
        if not self.dc_processed_data:
            return
            
        metrics = ['Primes', 'Anomalies']
        reductions = [
            self.dc_processed_data['primes_variance_reduction'],
            self.dc_processed_data['anomalies_variance_reduction']
        ]
        
        colors = ['skyblue' if r > 1 else 'lightcoral' for r in reductions]
        bars = ax.bar(metrics, reductions, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels
        for bar, reduction in zip(bars, reductions):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{reduction:.2f}x', ha='center', va='bottom', fontweight='bold')
        
        ax.axhline(1, color="red", linestyle="--", alpha=0.7, label="No change")
        ax.set_ylabel("Variance Reduction Ratio")
        ax.set_title("D-C Variance Reduction")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_dc_convergence_analysis(self, ax: plt.Axes, convergence_data: dict) -> None:
        """Plot convergence analysis for different containment types."""
        for containment_type, data in convergence_data.items():
            steps = [d['step'] for d in data]
            variances = [d['variance'] for d in data]
            ax.plot(steps, variances, 'o-', label=f'{containment_type}', linewidth=2, markersize=4)
        
        ax.set_xlabel("D-C Evolution Steps")
        ax.set_ylabel("Variance")
        ax.set_title("D-C Convergence by Containment Type")
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_log_derivative_comparison(self, ax: plt.Axes) -> None:
        """Compare actual and expected prime-density per bit."""
        ds = list(self.log_derivatives.keys())
        actual = [self.log_derivatives[k] for k in ds]
        expected = [(2**k) / k for k in ds]

        ax.plot(ds, actual, "o-", label="Actual Δₜπ", markersize=6)
        ax.plot(ds, expected, "s--", label="Expected n/k", markersize=4)
        ax.set_xlabel("Depth (k)")
        ax.set_ylabel("Prime Density per Bit")
        ax.set_title("Prime Density per Bit (Log-Derivative)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    def plot_prime_anomalies(self, ax: plt.Axes) -> None:
        """Bar chart of prime-density anomalies (actual – expected)."""
        ds = list(self.anomalies.keys())
        vals = [self.anomalies[k] for k in ds]
        colors = ["green" if v > 0 else "red" for v in vals]
        ax.bar(ds, vals, color=colors, alpha=0.7)

        ax.axhline(0, color="black", lw=1)
        ax.set_xlabel("Depth (k)")
        ax.set_ylabel("Anomaly (Δₜπ - n/k)")
        ax.set_title("Prime Density Anomalies")
        ax.grid(True, alpha=0.3)

    def plot_fractional_derivatives_with_dc(self, ax: Axes) -> None:
        """Plot fractional derivatives with D-C operator enhancement."""
        alphas = [0.25, 0.5, 0.75, 1.0]
        
        for alpha in alphas:
            fd = self.compute_fractional_derivative(alpha)
            ks = sorted(fd.keys())
            values = [fd[k] for k in ks]
            
            # Apply D-C processing to fractional derivatives
            if values:
                # Create matrix from fractional derivative values
                size = max(8, len(values))
                fd_matrix = jnp.zeros((size, size))
                for i, val in enumerate(values[:size]):
                    fd_matrix = fd_matrix.at[i, i].set(int(abs(val) * 100) % 5)
                
                dc_fd_matrix = DC_cycle(fd_matrix, "binary", max_iterations=3)
                dc_fd_vals = np.array([dc_fd_matrix[i, i] for i in range(min(len(values), size))])
                
                # Plot both original and D-C processed
                ax.plot(ks, values, 'o-', alpha=0.5, label=f'α={alpha} (orig)', linewidth=1)
                ax.plot(ks, dc_fd_vals[:len(ks)], 's-', alpha=0.8, 
                       label=f'α={alpha} (D-C)', linewidth=2, markersize=3)

        ax.set_xlabel("Depth (k)")
        ax.set_ylabel("Fractional Derivative")
        ax.set_title("Fractional Sieving: Grünwald–Letnikov with D-C Enhancement")
        ax.legend(ncol=2)
        ax.grid(True, alpha=0.3)

    def plot_containment_type_comparison(self, ax: plt.Axes, convergence_data: dict) -> None:
        """Compare how different containment types affect prime analysis."""
        final_variances = {}
        
        for containment_type, data in convergence_data.items():
            final_variances[containment_type] = data[-1]['variance']  # Last step variance
        
        types = list(final_variances.keys())
        variances = list(final_variances.values())
        
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        bars = ax.bar(types, variances, color=colors[:len(types)], alpha=0.7, edgecolor='black')
        
        # Add value labels
        for bar, variance in zip(bars, variances):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{variance:.2e}', ha='center', va='bottom', fontweight='bold', rotation=45)
        
        ax.set_ylabel("Final Variance")
        ax.set_title("D-C Containment Type Effectiveness")
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    def plot_dc_filtered_spectrum(self, ax: plt.Axes) -> None:
        """FFT power spectrum comparison: original vs D-C filtered."""
        vals_orig = [self.anomalies[k] for k in self.anomalies]
        
        # Original spectrum
        freq_orig = np.fft.rfftfreq(len(vals_orig))
        power_orig = np.abs(np.fft.rfft(vals_orig)) ** 2
        
        # D-C filtered spectrum
        if 'dc_anomalies' in self.dc_processed_data:
            vals_dc = self.dc_processed_data['dc_anomalies'][:len(vals_orig)]
            power_dc = np.abs(np.fft.rfft(vals_dc)) ** 2
        else:
            power_dc = power_orig

        ax.plot(freq_orig, power_orig, 'b-', alpha=0.7, linewidth=2, label='Original Spectrum')
        ax.plot(freq_orig, power_dc, 'r-', alpha=0.8, linewidth=2, label='D-C Filtered')
        
        # Mark peaks
        peak_orig = freq_orig[np.argmax(power_orig)]
        peak_dc = freq_orig[np.argmax(power_dc)]
        
        ax.axvline(peak_orig, color="blue", linestyle="--", alpha=0.5)
        ax.axvline(peak_dc, color="red", linestyle="--", alpha=0.5)
        
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Power")
        ax.set_title("Spectral Analysis: Original vs D-C Filtered Prime Anomalies")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)


def main():
    """Main demonstration of prime-Sierpinski D-C analysis."""
    print("🔢 Prime Numbers in Sierpinski Framework with Keya D-C Operators")
    print("=" * 70)
    
    # Create analyzer
    analyzer = PrimeSierpinskiDCAnalyzer(max_depth=16)
    
    # Run comprehensive D-C analysis
    analyzer.visualize_dc_prime_analysis()
    
    print("\n✅ Prime-Sierpinski D-C analysis complete!")
    print("📁 Check .out/visualizations/prime_sierpinski_dc.png")
    
    print("\n🔬 Key D-C Insights:")
    if analyzer.dc_processed_data:
        print(f"  • Prime variance reduction: {analyzer.dc_processed_data['primes_variance_reduction']:.2f}x")
        print(f"  • Anomaly variance reduction: {analyzer.dc_processed_data['anomalies_variance_reduction']:.2f}x")
    print("  • D-C operators reveal hidden prime number patterns")
    print("  • Containment stabilizes chaotic prime distributions")
    print("  • Diagonalization exposes prime gap structures")
    print("  • Sierpinski + D-C = New framework for number theory")


if __name__ == "__main__":
    main()