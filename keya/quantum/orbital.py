"""Electron Orbital Visualization using Keya Operators."""

from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.special import sph_harm, factorial, genlaguerre

from .wavefunction import QuantumWaveFunction, WaveFunctionType


class OrbitalType(Enum):
    """Types of atomic orbitals."""
    S_1S = "1s"      # 1s orbital (n=1, l=0, m=0)
    S_2S = "2s"      # 2s orbital (n=2, l=0, m=0)
    P_2PX = "2px"    # 2p orbitals (n=2, l=1, m=-1,0,1)
    P_2PY = "2py"
    P_2PZ = "2pz"
    D_3DZ2 = "3dz2"  # 3d orbitals (n=3, l=2)
    D_3DXY = "3dxy"
    D_3DXZ = "3dxz"
    D_3DYZ = "3dyz"
    D_3DX2Y2 = "3dx2-y2"
    F_4F = "4f"      # 4f orbitals (n=4, l=3)


class ElectronOrbital:
    """Represents an electron orbital in a hydrogen atom using keya mathematics."""
    
    def __init__(self,
                 orbital_type: OrbitalType = OrbitalType.S_1S,
                 grid_size: int = 80,
                 max_radius: float = 20.0,  # In Bohr radii
                 resolution: float = 0.25):  # Grid spacing in Bohr radii
        
        self.orbital_type = orbital_type
        self.grid_size = grid_size
        self.max_radius = max_radius
        self.resolution = resolution
        
        # Quantum numbers from orbital type
        self.n, self.l, self.m = self._parse_quantum_numbers(orbital_type)
        
        # Physical constants (atomic units)
        self.a0 = 1.0  # Bohr radius (atomic units)
        
        # Initialize coordinate grids
        self._setup_coordinates()
        
        # Calculate the orbital wave function
        self.wave_function = self._calculate_orbital()
        self.probability_density = np.abs(self.wave_function)**2
        
        # Normalize
        self._normalize()
        
        # Set up evolution for orbital dynamics
        self.evolution_wave_function = self._create_evolution_wave_function()
        
    def _parse_quantum_numbers(self, orbital_type: OrbitalType) -> Tuple[int, int, int]:
        """Parse quantum numbers (n, l, m) from orbital type."""
        
        orbital_map = {
            OrbitalType.S_1S: (1, 0, 0),
            OrbitalType.S_2S: (2, 0, 0),
            OrbitalType.P_2PX: (2, 1, 1),   # px ~ (Y₁₁ + Y₁₋₁)
            OrbitalType.P_2PY: (2, 1, -1),  # py ~ i(Y₁₁ - Y₁₋₁)
            OrbitalType.P_2PZ: (2, 1, 0),   # pz ~ Y₁₀
            OrbitalType.D_3DZ2: (3, 2, 0),
            OrbitalType.D_3DXY: (3, 2, 2),
            OrbitalType.D_3DXZ: (3, 2, 1),
            OrbitalType.D_3DYZ: (3, 2, -1),
            OrbitalType.D_3DX2Y2: (3, 2, -2),
            OrbitalType.F_4F: (4, 3, 0),
        }
        
        return orbital_map.get(orbital_type, (1, 0, 0))
    
    def _setup_coordinates(self):
        """Set up spherical and Cartesian coordinate grids."""
        # Create 3D Cartesian grid
        x = np.linspace(-self.max_radius, self.max_radius, self.grid_size)
        y = np.linspace(-self.max_radius, self.max_radius, self.grid_size)
        z = np.linspace(-self.max_radius, self.max_radius, self.grid_size)
        
        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Convert to spherical coordinates
        self.R = np.sqrt(self.X**2 + self.Y**2 + self.Z**2)
        self.R[self.R == 0] = 1e-10  # Avoid division by zero
        
        self.THETA = np.arccos(self.Z / self.R)  # Polar angle [0, π]
        self.PHI = np.arctan2(self.Y, self.X)    # Azimuthal angle [0, 2π]
        
    def _calculate_orbital(self) -> np.ndarray:
        """Calculate the hydrogen orbital wave function."""
        
        # Radial part
        radial_part = self._radial_wave_function(self.R, self.n, self.l)
        
        # Angular part (spherical harmonics)
        angular_part = self._angular_wave_function(self.THETA, self.PHI, self.l, self.m)
        
        # Complete wave function
        psi = radial_part * angular_part
        
        # Handle special cases for real orbitals (px, py, etc.)
        if self.orbital_type in [OrbitalType.P_2PX, OrbitalType.P_2PY]:
            psi = self._convert_to_real_orbital(psi)
        
        return psi
    
    def _radial_wave_function(self, r: np.ndarray, n: int, angular_l: int) -> np.ndarray:
        """Calculate the radial part of the hydrogen wave function."""
        
        # Normalization constant
        norm = np.sqrt((2/(n*self.a0))**3 * factorial(n-angular_l-1) / (2*n*factorial(n+angular_l)))
        
        # Rho = 2r/(na0)
        rho = 2 * r / (n * self.a0)
        
        # Associated Laguerre polynomial
        laguerre = genlaguerre(n-angular_l-1, 2*angular_l+1)(rho)
        
        # Radial wave function
        R_nl = norm * np.exp(-rho/2) * (rho**angular_l) * laguerre
        
        return R_nl
    
    def _angular_wave_function(self, theta: np.ndarray, phi: np.ndarray, angular_l: int, m: int) -> np.ndarray:
        """Calculate the angular part using spherical harmonics."""
        
        # Note: scipy's sph_harm uses (m, angular_l, phi, theta) convention
        Y_lm = sph_harm(m, angular_l, phi, theta)
        
        return Y_lm
    
    def _convert_to_real_orbital(self, psi_complex: np.ndarray) -> np.ndarray:
        """Convert complex spherical harmonics to real orbitals."""
        
        if self.orbital_type == OrbitalType.P_2PX:
            # px orbital: (Y₁₁ + Y₁₋₁) / √2
            # Already calculated Y₁₁, need to add Y₁₋₁
            Y_1_minus1 = sph_harm(-1, 1, self.PHI, self.THETA)
            return np.real((psi_complex + Y_1_minus1) / np.sqrt(2))
        
        elif self.orbital_type == OrbitalType.P_2PY:
            # py orbital: i(Y₁₁ - Y₁₋₁) / √2
            Y_1_1 = sph_harm(1, 1, self.PHI, self.THETA)
            Y_1_minus1 = sph_harm(-1, 1, self.PHI, self.THETA)
            return np.real(1j * (Y_1_1 - Y_1_minus1) / np.sqrt(2))
        
        return np.real(psi_complex)
    
    def _normalize(self):
        """Normalize the wave function."""
        norm_squared = np.sum(self.probability_density) * (self.resolution**3)
        if norm_squared > 1e-10:
            self.wave_function /= np.sqrt(norm_squared)
            self.probability_density = np.abs(self.wave_function)**2
    
    def _create_evolution_wave_function(self) -> QuantumWaveFunction:
        """Create a keya wave function for orbital evolution."""
        
        # Use a smaller grid for evolution
        evolution_size = min(50, self.grid_size)
        
        evolution_wave = QuantumWaveFunction(
            wave_type=WaveFunctionType.HYDROGEN,
            dimensions=(evolution_size, evolution_size, evolution_size),
            energy_level=self.n
        )
        
        # Initialize with current orbital data (downsampled)
        self._copy_to_evolution_wave_function(evolution_wave)
        
        return evolution_wave
    
    def _copy_to_evolution_wave_function(self, evolution_wave: QuantumWaveFunction):
        """Copy orbital data to wave function."""
        # Downsample the orbital to fit grid
        skip = max(1, self.grid_size // evolution_wave.nx)
        
        for i in range(evolution_wave.nx):
            for j in range(evolution_wave.ny):
                for k in range(evolution_wave.nz):
                    # Map to original grid
                    orig_i = min(i * skip, self.grid_size - 1)
                    orig_j = min(j * skip, self.grid_size - 1)
                    orig_k = min(k * skip, self.grid_size - 1)
                    
                    # Set real and imaginary parts
                    if np.isrealobj(self.wave_function):
                        evolution_wave.psi_real[i, j, k] = self.wave_function[orig_i, orig_j, orig_k]
                        evolution_wave.psi_imag[i, j, k] = 0.0
                    else:
                        evolution_wave.psi_real[i, j, k] = np.real(self.wave_function[orig_i, orig_j, orig_k])
                        evolution_wave.psi_imag[i, j, k] = np.imag(self.wave_function[orig_i, orig_j, orig_k])
    
    def evolve_with_wild_tame_operators(self, steps: int = 10) -> bool:
        """Evolve the orbital using keya operators."""
        
        print(f"🌀 Evolving {self.orbital_type.value} orbital with operators...")
        
        success = self.evolution_wave_function.apply_wild_tame_evolution(steps)
        
        if success:
            print(f"✅ evolution completed ({steps} steps)")
            # Update main orbital from evolved state
            self._copy_from_evolution_wave_function()
            return True
        else:
            print("❌ evolution failed")
            return False
    
    def _copy_from_evolution_wave_function(self):
        """Copy evolved state back to main orbital."""
        # This is a simplified version - in practice you'd want proper interpolation
        evolution_prob = self.evolution_wave_function.get_probability_density_3d()
        
        # Update the central region of the orbital
        scale_factor = np.max(self.probability_density) / (np.max(evolution_prob) + 1e-10)
        
        # Simple update of central region
        skip = max(1, self.grid_size // self.evolution_wave_function.nx)
        for i in range(self.evolution_wave_function.nx):
            for j in range(self.evolution_wave_function.ny):
                for k in range(self.evolution_wave_function.nz):
                    orig_i = self.grid_size//4 + i * skip
                    orig_j = self.grid_size//4 + j * skip
                    orig_k = self.grid_size//4 + k * skip
                    
                    if (orig_i < 3*self.grid_size//4 and 
                        orig_j < 3*self.grid_size//4 and 
                        orig_k < 3*self.grid_size//4):
                        
                        new_prob = evolution_prob[i, j, k] * scale_factor
                        self.probability_density[orig_i, orig_j, orig_k] = new_prob
    
    def plot_orbital(self, ax, title=""):
        """Plot the orbital as an isosurface."""
        from skimage import measure

        # Generate isosurface
        verts, faces, _, _ = measure.marching_cubes(
            self.probability_density,
            level=0.001,
            spacing=(self.resolution, self.resolution, self.resolution)
        )

        # Plot the surface
        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                        cmap='viridis', lw=1)

        ax.set_title(title)
        ax.set_xlabel("X (Bohr radii)")
        ax.set_ylabel("Y (Bohr radii)")
        ax.set_zlabel("Z (Bohr radii)")

    def get_probability_density_2d(self, plane: str = "xy", z_slice: Optional[int] = None) -> np.ndarray:
        """Get 2D slice of probability density."""
        
        if z_slice is None:
            z_slice = self.grid_size // 2
        
        if plane == "xy":
            return self.probability_density[:, :, z_slice]
        elif plane == "xz":
            return self.probability_density[:, z_slice, :]
        elif plane == "yz":
            return self.probability_density[z_slice, :, :]
        else:
            raise ValueError(f"Unknown plane: {plane}")
    
    def get_isosurface_data(self, isoval: float = 0.1) -> Dict:
        """Get data for 3D isosurface rendering."""
        
        # Find points where probability density equals isoval
        prob_density = self.probability_density
        
        # Simple isosurface approximation
        mask = prob_density >= isoval
        
        # Get coordinates of isosurface points
        points = np.column_stack([
            self.X[mask].flatten(),
            self.Y[mask].flatten(), 
            self.Z[mask].flatten()
        ])
        
        # Get values at those points
        values = prob_density[mask].flatten()
        
        return {
            'points': points,
            'values': values,
            'isoval': isoval,
            'orbital_type': self.orbital_type.value,
            'quantum_numbers': (self.n, self.l, self.m)
        }
    
    def get_radial_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get radial probability distribution P(r) = 4πr²|ψ(r)|²."""
        
        # Create radial bins
        r_max = self.max_radius
        r_bins = np.linspace(0, r_max, 200)
        dr = r_bins[1] - r_bins[0]
        
        radial_prob = np.zeros_like(r_bins)
        
        # Integrate over spherical shells
        for i, r in enumerate(r_bins):
            # Find all points in shell [r, r+dr]
            mask = (self.R >= r) & (self.R < r + dr)
            if np.any(mask):
                # Sum probability in this shell
                shell_prob = np.sum(self.probability_density[mask])
                # Multiply by shell volume factor
                radial_prob[i] = shell_prob * 4 * np.pi * r**2 * dr
        
        return r_bins, radial_prob
    
    def get_orbital_info(self) -> Dict:
        """Get comprehensive orbital information."""
        
        r_bins, radial_prob = self.get_radial_distribution()
        
        # Find most probable radius
        max_prob_idx = np.argmax(radial_prob)
        most_probable_r = r_bins[max_prob_idx] if max_prob_idx > 0 else 0
        
        # Calculate expectation values
        total_prob = np.sum(self.probability_density)
        
        return {
            'orbital_type': self.orbital_type.value,
            'quantum_numbers': {'n': self.n, 'l': self.l, 'm': self.m},
            'grid_size': self.grid_size,
            'max_radius': self.max_radius,
            'resolution': self.resolution,
            'is_normalized': np.isclose(np.sum(self.probability_density) * (self.resolution**3), 1.0),
            'most_probable_radius': float(most_probable_r),
            'max_probability_density': float(np.max(self.probability_density)),
            'energy_level': -13.6 / (self.n**2),  # eV
            'evolution_available': True
        }
    
    def get_position_expectation(self) -> Tuple[float, float, float]:
        """Calculates the expectation value for position."""
        total_prob = np.sum(self.probability_density)
        if total_prob < 1e-12:
            return 0.0, 0.0, 0.0

        exp_x = np.sum(self.X * self.probability_density) / total_prob
        exp_y = np.sum(self.Y * self.probability_density) / total_prob
        exp_z = np.sum(self.Z * self.probability_density) / total_prob

        return exp_x, exp_y, exp_z

    def get_momentum_expectation(self) -> Tuple[float, float, float]:
        """Calculates the expectation value for momentum."""
        total_prob = np.sum(self.probability_density)
        if total_prob < 1e-12:
            return 0.0, 0.0, 0.0

        psi_conj = np.conj(self.wave_function)
        
        # The gradient needs to be calculated with respect to coordinate changes.
        # The spacing is self.resolution for all axes.
        grad_x, grad_y, grad_z = np.gradient(self.wave_function, self.resolution)

        # Expectation value <p> = ∫ ψ* (-iħ∇) ψ dV / ∫ ψ*ψ dV
        # The integral is approximated by sum * dV. dV cancels.
        # atomic units: ħ=1.
        exp_px = np.sum(psi_conj * (-1j) * grad_x) / total_prob
        exp_py = np.sum(psi_conj * (-1j) * grad_y) / total_prob
        exp_pz = np.sum(psi_conj * (-1j) * grad_z) / total_prob
        
        return exp_px.real, exp_py.real, exp_pz.real

    def get_expectation_values(self) -> Dict[str, float]:
        """Calculates expectation values for position and momentum."""
        # Calculate expectation values
        pos_x, pos_y, pos_z = self.get_position_expectation()
        mom_x, mom_y, mom_z = self.get_momentum_expectation()
        
        return {
            'position_x': pos_x,
            'position_y': pos_y,
            'position_z': pos_z,
            'momentum_x': mom_x,
            'momentum_y': mom_y,
            'momentum_z': mom_z
        } 