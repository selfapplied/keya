
"""
Example: Visualization of prime-counting log-derivative in the Sierpinski sieve framework.
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy.ntheory import primepi
from scipy.special import binom
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec


class PrimeSierpinskiAnalyzer:
    def __init__(self, max_depth: int = 16) -> None:
        self.max_depth = max_depth
        self.depths = np.arange(1, max_depth + 1)
        self.prime_counts = self.compute_prime_counts()
        self.log_derivatives = self.compute_log_derivatives()
        self.anomalies = self.compute_anomalies()

    def compute_prime_counts(self) -> dict[int, int]:
        """Compute prime counts π(2^k) for k=1..max_depth."""
        return {k: primepi(2**k) for k in self.depths}

    def compute_log_derivatives(self) -> dict[int, float]:
        """Compute log-derivatives Δₜπ(k) = π(2^{k+1}) - π(2^k)."""
        return {k: self.prime_counts[k + 1] - self.prime_counts[k] for k in self.depths[:-1]}

    def compute_anomalies(self) -> dict[int, float]:
        """Compute anomalies = actual - expected prime density (2^k / k)."""
        return {k: self.log_derivatives[k] - (2**k) / k for k in self.depths[:-1]}

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

    def visualize(self) -> None:
        """Create a multi-panel visualization of primes and anomalies."""
        fig = plt.figure(figsize=(18, 20))
        gs = GridSpec(4, 2, figure=fig, height_ratios=[1.5, 1, 1, 1])

        ax1 = fig.add_subplot(gs[0, :])
        self.plot_sierpinski_prime_sparks(ax1)

        ax2 = fig.add_subplot(gs[1, 0])
        self.plot_log_derivative_comparison(ax2)

        ax3 = fig.add_subplot(gs[1, 1])
        self.plot_prime_anomalies(ax3)

        ax4 = fig.add_subplot(gs[2, :])
        self.plot_fractional_derivatives(ax4)

        ax5 = fig.add_subplot(gs[3, :])
        self.plot_anomaly_spectrum(ax5)

        plt.tight_layout()
        out_fn = "prime_sierpinski.png"
        plt.savefig(out_fn, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {out_fn}")
        plt.show()

    def plot_sierpinski_prime_sparks(self, ax: plt.Axes) -> None:
        """Plot downsampled Sierpinski pattern with prime-density sparks."""
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

        depths = list(self.log_derivatives.keys())
        positions = [2**k for k in depths]
        values = [self.log_derivatives[k] for k in depths]

        scale = size / max(values)
        scaled = [v * scale for v in values]
        ax.scatter(positions, depths, s=scaled, c=values, cmap="plasma", alpha=0.7, edgecolors="w", zorder=10)

        ax.set_xscale("log")
        ax.set_xlabel("n (log scale)")
        ax.set_ylabel("Depth (k)")
        ax.set_title("Sierpinski Sieve with Prime-Density Sparks")
        ax.grid(True, alpha=0.2)

        ax2 = ax.twinx()
        ax2.plot(positions, values, "C0-", lw=2, alpha=0.7)
        ax2.set_ylabel("Δₜπ (Prime Count Difference)", color="C0")
        ax2.tick_params(axis="y", labelcolor="C0")

        expected = [(2**k) / k for k in depths]
        ax2.plot(positions, expected, "C1--", lw=2, alpha=0.7)
        ax.legend(["Actual Δₜπ", "Expected n/k"], loc="upper left")

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

    def plot_fractional_derivatives(self, ax: plt.Axes) -> None:
        """Plot fractional derivatives for various orders α."""
        alphas = [0.25, 0.5, 0.75, 1.0]
        for alpha in alphas:
            fd = self.compute_fractional_derivative(alpha)
            ks = sorted(fd.keys())
            ax.plot(ks, [fd[k] for k in ks], "o-", label=f"α={alpha}", alpha=0.8)

        ax.set_xlabel("Depth (k)")
        ax.set_ylabel("Fractional Derivative")
        ax.set_title("Fractional Sieving: Grünwald–Letnikov")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_anomaly_spectrum(self, ax: plt.Axes) -> None:
        """FFT power spectrum of the prime-density anomalies."""
        vals = [self.anomalies[k] for k in self.anomalies]
        freq = np.fft.rfftfreq(len(vals))
        power = np.abs(np.fft.rfft(vals)) ** 2

        ax.stem(freq, power, basefmt=" ", use_line_collection=True)
        peak = freq[np.argmax(power)]
        ax.axvline(peak, color="red", linestyle="--", alpha=0.5)
        ax.text(peak, max(power) * 0.9, f"f={peak:.3f}", color="red", ha="center")

        ax.set_xlabel("Frequency")
        ax.set_ylabel("Power")
        ax.set_title("Spectral Analysis of Prime Anomalies")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")


if __name__ == "__main__":
    analyzer = PrimeSierpinskiAnalyzer(max_depth=16)
    analyzer.visualize()