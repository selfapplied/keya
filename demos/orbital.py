"""
Example: Rendering a hydrogen electron orbital by solving the
Schrödinger equation as a symbolic equilibrium problem.
"""

import jax
import jax.numpy as jnp

from keya.core.engine import EquilibriumOperator
from keya.vis.renderer import plot_wavefunction

# --- Define the Physical Operators ---


def laplacian_3d(psi: jax.Array) -> jax.Array:
    """
    Computes the 3D Laplacian ∇²ψ using a finite difference stencil.
    This represents the kinetic energy operator.
    """
    # jnp.roll implements periodic boundary conditions, which is a good
    # approximation for a wavefunction that decays at the boundaries.
    return (
        jnp.roll(psi, shift=1, axis=0)
        + jnp.roll(psi, shift=-1, axis=0)
        + jnp.roll(psi, shift=1, axis=1)
        + jnp.roll(psi, shift=-1, axis=1)
        + jnp.roll(psi, shift=1, axis=2)
        + jnp.roll(psi, shift=-1, axis=2)
        - 6 * psi
    )


def coulomb_potential_3d(grid_size: int) -> jax.Array:
    """
    Creates the Coulomb potential V = -1/r.
    This represents the potential energy from the nucleus.
    """
    coords = jnp.linspace(-5, 5, grid_size)
    x, y, z = jnp.meshgrid(coords, coords, coords)
    r = jnp.sqrt(x**2 + y**2 + z**2)
    # Add a small epsilon to avoid division by zero at the center
    potential = -1 / (r + 1e-6)
    return potential


# --- Main Simulation ---


def render_orbital(grid_size: int = 50, energy: float = -0.5):
    """
    Initializes and solves for a hydrogen orbital.

    Args:
        grid_size: The resolution of the simulation grid.
        energy: The energy level of the orbital to solve for.
    """
    print("Initializing system...")
    # The Schrödinger equation is (∇² + V)ψ = Eψ
    # We can rewrite this as: ∇²ψ = (E - V)ψ
    # This fits our forward/reverse operator model perfectly.

    # The forward process is the application of the potential.
    potential = coulomb_potential_3d(grid_size)
    forward_op = lambda psi: (energy - potential) * psi

    # The reverse process is the kinetic energy operator.
    # In our equation, ∇² is on the other side, so it acts as the "reverse".
    reverse_op = laplacian_3d

    # Create the equilibrium operator for the hydrogen atom
    hydrogen_atom = EquilibriumOperator(forward_op, reverse_op)

    # Start with an initial guess for the wavefunction (a random state)
    key = jax.random.PRNGKey(0)
    # Use a complex dtype for the wavefunction to capture phase
    initial_psi = jax.random.normal(key, (grid_size, grid_size, grid_size), dtype=jnp.complex64)

    print("Resolving equilibrium state...")
    # Use a large number of steps to allow the system to converge
    final_psi = hydrogen_atom.resolve(initial_psi, t_max=2000)

    # Normalize the wavefunction
    norm = jnp.sqrt(jnp.sum(jnp.abs(final_psi) ** 2))
    final_psi /= norm

    print("Rendering wavefunction...")
    # Visualize the result
    plot_wavefunction(final_psi, title=f"Hydrogen Orbital (E={energy})")


if __name__ == "__main__":
    # Let's solve for the 2p orbital, which has a characteristic dumbbell shape
    # and non-zero angular momentum, resulting in interesting phase properties.
    # The energy for n=2 is E = -13.6 / 2² = -3.4, but we use normalized units.
    # Let's try an energy level that might produce a p-orbital shape.
    render_orbital(grid_size=40, energy=-0.12)
