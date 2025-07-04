"""Quantum Wave Function Representation using Keya Operators."""

import cmath
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Tuple

import numpy as np

from ..core.engine import Engine
from ..dsl.ast import ContainmentType


class WaveFunctionType(Enum):
    """Types of quantum wave functions."""
    GAUSSIAN = "gaussian"           # Gaussian wave packet
    HARMONIC = "harmonic"          # Harmonic oscillator states
    HYDROGEN = "hydrogen"          # Hydrogen atom orbitals
    FREE_PARTICLE = "free"         # Free particle plane waves
    SUPERPOSITION = "superposition" # Quantum superposition states


@dataclass
class QuantumState:
    """Represents a quantum state with amplitude and phase."""
    amplitude: float
    phase: float
    position: Tuple[float, float, float]  # 3D position
    
    @property
    def complex_amplitude(self) -> complex:
        """Get the complex amplitude."""
        return self.amplitude * cmath.exp(1j * self.phase)
    
    @property
    def probability_density(self) -> float:
        """Get the probability density |ψ|²."""
        return self.amplitude ** 2


class QuantumWaveFunction:
    """A quantum wave function represented using keya mathematical operators."""
    
    def __init__(self,
                 wave_type: WaveFunctionType = WaveFunctionType.GAUSSIAN,
                 dimensions: Tuple[int, int, int] = (50, 50, 50),
                 containment_type: ContainmentType = ContainmentType.GENERAL,
                 energy_level: int = 1):
        
        self.wave_type = wave_type
        self.dimensions = dimensions
        self.containment_type = containment_type
        self.energy_level = energy_level
        
        # Initialize keya engine for quantum evolution
        self.engine = Engine()
        
        # 3D grid for wave function
        self.nx, self.ny, self.nz = dimensions
        self.grid_spacing = 0.1  # Spatial resolution
        
        # Initialize wave function amplitudes and phases
        self.psi_real = np.zeros(dimensions)  # Real part
        self.psi_imag = np.zeros(dimensions)  # Imaginary part
        
        # Initialize quantum state
        self._initialize_wave_function()
        
        # Evolution parameters
        self.time = 0.0
        self.dt = 0.01  # Time step
        self.hbar = 1.0  # Reduced Planck constant (natural units)
        
    def _initialize_wave_function(self):
        """Initialize the wave function based on type."""
        
        if self.wave_type == WaveFunctionType.GAUSSIAN:
            self._create_gaussian_packet()
        elif self.wave_type == WaveFunctionType.HARMONIC:
            self._create_harmonic_oscillator()
        elif self.wave_type == WaveFunctionType.HYDROGEN:
            self._create_hydrogen_orbital()
        elif self.wave_type == WaveFunctionType.FREE_PARTICLE:
            self._create_free_particle()
        elif self.wave_type == WaveFunctionType.SUPERPOSITION:
            self._create_superposition()
    
    def _create_gaussian_packet(self):
        """Create a Gaussian wave packet."""
        # Center the packet
        x0, y0, z0 = self.nx // 2, self.ny // 2, self.nz // 2
        sigma = min(self.dimensions) / 8  # Width of the packet
        
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    x = (i - x0) * self.grid_spacing
                    y = (j - y0) * self.grid_spacing  
                    z = (k - z0) * self.grid_spacing
                    
                    r_squared = x*x + y*y + z*z
                    
                    # Gaussian envelope with momentum
                    amplitude = math.exp(-r_squared / (2 * sigma**2))
                    phase = 0.5 * x  # Add momentum in x direction
                    
                    self.psi_real[i, j, k] = amplitude * math.cos(phase)
                    self.psi_imag[i, j, k] = amplitude * math.sin(phase)
    
    def _create_harmonic_oscillator(self):
        """Create 3D harmonic oscillator wave function."""
        # Center coordinates
        x0, y0, z0 = self.nx // 2, self.ny // 2, self.nz // 2
        
        # Oscillator length scale
        a = min(self.dimensions) / 6
        
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    x = (i - x0) * self.grid_spacing
                    y = (j - y0) * self.grid_spacing
                    z = (k - z0) * self.grid_spacing
                    
                    # 3D harmonic oscillator ground state
                    r_squared = x*x + y*y + z*z
                    amplitude = math.exp(-r_squared / (2 * a**2))
                    
                    # Add quantum number dependence
                    if self.energy_level > 1:
                        # First excited state (one quantum in x)
                        amplitude *= (x / a)
                    
                    self.psi_real[i, j, k] = amplitude
                    self.psi_imag[i, j, k] = 0.0
    
    def _create_hydrogen_orbital(self):
        """Create hydrogen atom orbital (simplified)."""
        # Center at nucleus
        x0, y0, z0 = self.nx // 2, self.ny // 2, self.nz // 2
        
        # Bohr radius scale
        a0 = min(self.dimensions) / 8
        
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    x = (i - x0) * self.grid_spacing
                    y = (j - y0) * self.grid_spacing
                    z = (k - z0) * self.grid_spacing
                    
                    r = math.sqrt(x*x + y*y + z*z)
                    
                    if r < 0.01:  # Avoid singularity at origin
                        r = 0.01
                    
                    if self.energy_level == 1:
                        # 1s orbital
                        amplitude = 2 * math.exp(-r / a0) / math.sqrt(4 * math.pi * a0**3)
                    else:
                        # 2p orbital (simplified)
                        amplitude = (r / a0) * math.exp(-r / (2 * a0)) * (z / r)
                    
                    self.psi_real[i, j, k] = amplitude
                    self.psi_imag[i, j, k] = 0.0
    
    def _create_free_particle(self):
        """Create free particle plane wave."""
        # Wave vector
        k = 0.5  # Momentum
        
        for i in range(self.nx):
            for j in range(self.ny):
                for k_idx in range(self.nz):
                    x = i * self.grid_spacing
                    phase = k * x
                    
                    self.psi_real[i, j, k_idx] = math.cos(phase)
                    self.psi_imag[i, j, k_idx] = math.sin(phase)
    
    def _create_superposition(self):
        """Create quantum superposition of states."""
        # Create superposition of Gaussian packets
        self._create_gaussian_packet()
        
        # Add second packet offset
        temp_real = np.zeros_like(self.psi_real)
        temp_imag = np.zeros_like(self.psi_imag)
        
        # Second packet shifted
        x0, y0, z0 = self.nx // 3, self.ny // 2, self.nz // 2
        sigma = min(self.dimensions) / 8
        
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    x = (i - x0) * self.grid_spacing
                    y = (j - y0) * self.grid_spacing
                    z = (k - z0) * self.grid_spacing
                    
                    r_squared = x*x + y*y + z*z
                    amplitude = math.exp(-r_squared / (2 * sigma**2))
                    phase = -0.5 * x  # Opposite momentum
                    
                    temp_real[i, j, k] = amplitude * math.cos(phase)
                    temp_imag[i, j, k] = amplitude * math.sin(phase)
        
        # Superposition: |ψ⟩ = (|ψ₁⟩ + |ψ₂⟩) / √2
        self.psi_real = (self.psi_real + temp_real) / math.sqrt(2)
        self.psi_imag = (self.psi_imag + temp_imag) / math.sqrt(2)
    
    def apply_wild_tame_evolution(self, time_steps: int = 1) -> bool:
        """Evolve the wave function using keya operators."""
        
        # Convert wave function to matrix for processing
        # Use probability density as the basis for evolution
        prob_density = self.get_probability_density_2d()
        
        # Create keya program for quantum evolution
        keya_program = f"""
matrix quantum_evolution {{
    time_step {{
        evolved_psi = ∮(psi_matrix, {self.containment_type.value}, {time_steps})
    }}
}}
"""
        
        try:
            # Set wave function as engine variable
            self.engine.variables['psi_matrix'] = prob_density
            
            # Execute evolution
            self.engine.execute_program(keya_program.strip())
            
            # Get evolved state
            if 'evolved_psi' in self.engine.variables:
                evolved = self.engine.variables['evolved_psi']
                if isinstance(evolved, np.ndarray):
                    # Update wave function based on evolved probability
                    self._update_from_evolved_matrix(evolved)
                    self.time += time_steps * self.dt
                    return True
                    
        except Exception as e:
            print(f"Quantum evolution error: {e}")
            
        return False
    
    def plot_wave_function(self, ax, title=""):
        """Plot the 3D probability density of the wave function."""
        prob_density = self.get_probability_density_3d()
        
        # Threshold to remove "quantum foam" of low-probability points
        threshold = 0.05 * np.max(prob_density)
        mask = prob_density > threshold

        # Create a meshgrid for plotting
        x = np.linspace(0, self.dimensions[0], self.dimensions[0])
        y = np.linspace(0, self.dimensions[1], self.dimensions[1])
        z = np.linspace(0, self.dimensions[2], self.dimensions[2])
        x, y, z = np.meshgrid(x, y, z)

        # Use scatter plot for visualization, applying the mask
        ax.scatter(x[mask], y[mask], z[mask], c=prob_density[mask].flatten(), marker='.', alpha=0.5, cmap='viridis')
        
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    def _update_from_evolved_matrix(self, evolved_matrix: np.ndarray):
        """Update wave function from evolved matrix (2D to 3D)."""
        # Simple projection for now
        if evolved_matrix.shape[0] <= self.nx and evolved_matrix.shape[1] <= self.ny:
            z_center = self.nz // 2
            
            # Update real part based on evolved probabilities
            for i in range(min(evolved_matrix.shape[0], self.nx)):
                for j in range(min(evolved_matrix.shape[1], self.ny)):
                    # Convert evolved probability back to amplitude
                    new_amplitude = math.sqrt(abs(evolved_matrix[i, j]))
                    
                    # Preserve phase information 
                    old_phase = math.atan2(self.psi_imag[i, j, z_center], 
                                         self.psi_real[i, j, z_center] + 1e-10)
                    
                    self.psi_real[i, j, z_center] = new_amplitude * math.cos(old_phase)
                    self.psi_imag[i, j, z_center] = new_amplitude * math.sin(old_phase)
    
    def get_probability_density_3d(self) -> np.ndarray:
        """Get 3D probability density |ψ|²."""
        return self.psi_real**2 + self.psi_imag**2
    
    def get_probability_density_2d(self, axis: int = 2) -> np.ndarray:
        """Get 2D probability density by integrating along one axis."""
        prob_3d = self.get_probability_density_3d()
        return np.sum(prob_3d, axis=axis)
    
    def get_phase_2d(self, axis: int = 2) -> np.ndarray:
        """Get 2D phase by averaging along one axis."""
        real_2d = np.mean(self.psi_real, axis=axis)
        imag_2d = np.mean(self.psi_imag, axis=axis)
        return np.arctan2(imag_2d, real_2d + 1e-10)
    
    def measure_expectation(self, operator: str = "position") -> Tuple[float, float, float]:
        """Measure expectation value of an operator."""
        prob_density = self.get_probability_density_3d()
        total_prob = np.sum(prob_density)
        
        if total_prob < 1e-10:
            return (0.0, 0.0, 0.0)
        
        if operator == "position":
            # ⟨x⟩, ⟨y⟩, ⟨z⟩
            x_exp = 0.0
            y_exp = 0.0
            z_exp = 0.0
            
            for i in range(self.nx):
                for j in range(self.ny):
                    for k in range(self.nz):
                        x = i * self.grid_spacing
                        y = j * self.grid_spacing
                        z = k * self.grid_spacing
                        
                        weight = prob_density[i, j, k] / total_prob
                        x_exp += x * weight
                        y_exp += y * weight
                        z_exp += z * weight
            
            return (x_exp, y_exp, z_exp)
        
        return (0.0, 0.0, 0.0)
    
    def collapse_wave_function(self, measurement_pos: Tuple[int, int, int]):
        """Simulate wave function collapse upon measurement."""
        # D operator: Creates dissonance/collapse at measurement point
        i, j, k = measurement_pos
        
        if 0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz:
            # Collapse to delta function at measurement point
            self.psi_real.fill(0.0)
            self.psi_imag.fill(0.0)
            
            # Peak at measurement location
            self.psi_real[i, j, k] = 1.0
            self.psi_imag[i, j, k] = 0.0
            
            print(f"⚡ Wave function collapsed at ({i}, {j}, {k})")
    
    def normalize(self):
        """Normalize the wave function."""
        norm_squared = np.sum(self.psi_real**2 + self.psi_imag**2)
        if norm_squared > 1e-10:
            norm = math.sqrt(norm_squared)
            self.psi_real /= norm
            self.psi_imag /= norm
    
    def get_quantum_stats(self) -> Dict[str, Any]:
        """Get quantum mechanical statistics."""
        prob_density = self.get_probability_density_3d()
        total_prob = np.sum(prob_density)
        
        position_exp = self.measure_expectation("position")
        
        return {
            'wave_type': self.wave_type.value,
            'energy_level': self.energy_level,
            'time': self.time,
            'total_probability': float(total_prob),
            'position_expectation': position_exp,
            'max_probability': float(np.max(prob_density)),
            'dimensions': self.dimensions,
            'containment_type': self.containment_type.value,
            'is_normalized': abs(total_prob - 1.0) < 0.01
        } 