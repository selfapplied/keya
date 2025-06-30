"""
Functions for rendering the output of kéya simulations.
"""

import matplotlib.pyplot as plt
import numpy as np


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

    sc = ax.scatter(
        x[points],
        y[points],
        z[points],
        c=phase,
        s=5,
        alpha=(prob_density[points] * alpha_scale).clip(0, 1),
        cmap="hsv",
    )

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.colorbar(sc, label="Phase (radians)")

    plt.show()
