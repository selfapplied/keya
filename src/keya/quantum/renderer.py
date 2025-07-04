"""3D Quantum Phenomena Renderer using Keya D-C Operators."""

from enum import Enum
from typing import Optional, Tuple, Any

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from .wavefunction import QuantumWaveFunction, WaveFunctionType
from .orbital import ElectronOrbital, OrbitalType


class PhenomenaType(Enum):
    """Types of quantum phenomena to render."""
    WAVE_PACKET = "wave_packet"           # Gaussian wave packet evolution
    ELECTRON_ORBITAL = "electron_orbital" # Hydrogen atom orbitals
    PROBABILITY_CLOUD = "probability_cloud" # 3D probability clouds
    SUPERPOSITION = "superposition"       # Quantum superposition states
    WAVE_COLLAPSE = "wave_collapse"      # Wave function collapse
    DC_EVOLUTION = "dc_evolution"        # D-C operator evolution


class QuantumRenderer:
    """3D renderer for quantum phenomena using keya D-C mathematical framework."""
    
    def __init__(self, 
                 phenomena_type: PhenomenaType = PhenomenaType.ELECTRON_ORBITAL,
                 figure_size: Tuple[int, int] = (12, 10)):
        
        self.phenomena_type = phenomena_type
        self.figure_size = figure_size
        
        # Set up matplotlib figure with 3D subplot
        self.fig = plt.figure(figsize=figure_size)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Quantum objects
        self.orbital: Optional[ElectronOrbital] = None
        self.wave_function: Optional[QuantumWaveFunction] = None
        
        # Rendering state
        self.current_frame = 0
        self.animation: Optional[Any] = None  # FuncAnimation type varies with matplotlib version
        self.is_animating = False
        
        # Visualization parameters
        self.isosurface_value = 0.1
        self.colormap = self._create_quantum_colormap()
        self.transparency = 0.7
        
        # Setup display
        self._setup_display()
        
    def _create_quantum_colormap(self) -> LinearSegmentedColormap:
        """Create a beautiful quantum-themed colormap."""
        colors = [
            (0.0, 0.0, 0.1),      # Deep blue (low probability)
            (0.0, 0.2, 0.8),      # Bright blue
            (0.2, 0.8, 0.8),      # Cyan
            (0.8, 0.8, 0.2),      # Yellow
            (1.0, 0.4, 0.0),      # Orange
            (1.0, 0.0, 0.0),      # Red (high probability)
        ]
        return LinearSegmentedColormap.from_list("quantum", colors)
    
    def _setup_display(self) -> None:
        """Setup the 3D display."""
        self.ax.set_xlabel('X (Bohr radii)')
        self.ax.set_ylabel('Y (Bohr radii)')
        self.ax.set_zlabel('Z (Bohr radii)')
        
        # Dark background for quantum feel
        self.fig.patch.set_facecolor('black')
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        
        # Grid styling
        self.ax.grid(True, alpha=0.3)
        
        # Title
        self.ax.set_title("Keya D-C Quantum Phenomena Renderer", 
                         color='white', fontsize=16, pad=20)
    
    def load_orbital(self, orbital_type: OrbitalType = OrbitalType.S_1S, grid_size: int = 60):
        """Load an electron orbital for visualization."""
        print(f"🌀 Loading {orbital_type.value} orbital...")
        
        self.orbital = ElectronOrbital(
            orbital_type=orbital_type,
            grid_size=grid_size,
            max_radius=15.0
        )
        
        print(f"✅ Orbital loaded: {self.orbital.get_orbital_info()}")
        
    def load_wave_function(self, 
                          wave_type: WaveFunctionType = WaveFunctionType.GAUSSIAN,
                          dimensions: Tuple[int, int, int] = (40, 40, 40)):
        """Load a quantum wave function for visualization."""
        print(f"🌊 Loading {wave_type.value} wave function...")
        
        self.wave_function = QuantumWaveFunction(
            wave_type=wave_type,
            dimensions=dimensions
        )
        
        print(f"✅ Wave function loaded: {self.wave_function.get_quantum_stats()}")
    
    def render_orbital_2d_slices(self) -> None:
        """Render 2D slices of the orbital probability density."""
        if not self.orbital:
            print("❌ No orbital loaded!")
            return
            
        # Clear previous plots
        self.ax.clear()
        self._setup_display()
        
        # Get 2D slices
        xy_slice = self.orbital.get_probability_density_2d("xy")
        xz_slice = self.orbital.get_probability_density_2d("xz") 
        yz_slice = self.orbital.get_probability_density_2d("yz")
        
        # Create coordinate grids for plotting
        max_r = self.orbital.max_radius
        x = np.linspace(-max_r, max_r, xy_slice.shape[0])
        y = np.linspace(-max_r, max_r, xy_slice.shape[1])
        z = np.linspace(-max_r, max_r, xz_slice.shape[1])
        
        X, Y = np.meshgrid(x, y)
        X_z, Z = np.meshgrid(x, z)
        Y_z, Z_y = np.meshgrid(y, z)
        
        # Plot the three slices
        offset = max_r * 0.8
        
        # XY plane at z = -offset
        self.ax.contourf(X, Y, xy_slice.T, zdir='z', offset=-offset, 
                        cmap=self.colormap, alpha=0.8, levels=20)
        
        # XZ plane at y = offset
        self.ax.contourf(X_z, xz_slice.T, Z, zdir='y', offset=offset,
                        cmap=self.colormap, alpha=0.8, levels=20)
        
        # YZ plane at x = -offset
        self.ax.contourf(yz_slice, Y_z, Z_y, zdir='x', offset=-offset,
                        cmap=self.colormap, alpha=0.8, levels=20)
        
        # Set limits
        self.ax.set_xlim(-max_r, max_r)
        self.ax.set_ylim(-max_r, max_r) 
        self.ax.set_zlim(-max_r, max_r)
        
        # Add info text
        info = self.orbital.get_orbital_info()
        info_text = (f"{info['orbital_type']} orbital\n"
                    f"n={info['quantum_numbers']['n']}, "
                    f"l={info['quantum_numbers']['l']}, "
                    f"m={info['quantum_numbers']['m']}\n"
                    f"Energy: {info['energy_level']:.2f} eV")
        
        self.ax.text2D(0.02, 0.98, info_text, transform=self.ax.transAxes,
                      fontsize=10, color='white', verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        plt.tight_layout()
        
    def render_orbital_3d_isosurface(self, isoval: float = 0.02):
        """Render 3D isosurface of orbital probability density."""
        if not self.orbital:
            print("❌ No orbital loaded!")
            return
            
        # Clear previous plots
        self.ax.clear()
        self._setup_display()
        
        print(f"🎨 Rendering 3D isosurface (isoval={isoval})...")
        
        # Get probability density
        prob_density = self.orbital.probability_density
        
        # Create coordinate grids
        max_r = self.orbital.max_radius
        x = np.linspace(-max_r, max_r, self.orbital.grid_size)
        y = np.linspace(-max_r, max_r, self.orbital.grid_size)
        z = np.linspace(-max_r, max_r, self.orbital.grid_size)
        
        # Find points above isosurface threshold
        mask = prob_density >= isoval
        
        # Get coordinates and values
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Extract surface points
        x_surf = X[mask]
        y_surf = Y[mask]
        z_surf = Z[mask]
        colors = prob_density[mask]
        
        # Create 3D scatter plot with color-coded probability
        scatter = self.ax.scatter(x_surf, y_surf, z_surf, 
                                 c=colors, cmap=self.colormap, 
                                 alpha=self.transparency, s=2)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=self.ax, shrink=0.8)
        cbar.set_label('Probability Density |ψ|²', color='white')
        cbar.ax.yaxis.label.set_color('white')
        cbar.ax.tick_params(colors='white')
        
        # Set equal aspect ratio
        max_range = max_r
        self.ax.set_xlim(-max_range, max_range)
        self.ax.set_ylim(-max_range, max_range)
        self.ax.set_zlim(-max_range, max_range)
        
        # Add orbital info
        info = self.orbital.get_orbital_info()
        info_text = (f"{info['orbital_type']} orbital 3D\n"
                    f"Isosurface: {isoval}\n"
                    f"Points: {len(x_surf)}")
        
        self.ax.text2D(0.02, 0.98, info_text, transform=self.ax.transAxes,
                      fontsize=10, color='white', verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        plt.tight_layout()
        
    def render_wave_function_evolution(self, steps: int = 50):
        """Render real-time wave function evolution using D-C operators."""
        if not self.wave_function:
            print("❌ No wave function loaded!")
            return
            
        print(f"🌊 Starting wave function evolution ({steps} steps)...")
        
        def animate(frame):
            # Clear and setup
            self.ax.clear()
            self._setup_display()
            
            # Safety check for wave function
            if not self.wave_function:
                return []
            
            # Evolve one step using D-C operators
            if frame > 0:
                self.wave_function.apply_dc_evolution(1)
            
            # Get current probability density
            prob_2d = self.wave_function.get_probability_density_2d()
            
            # Create surface plot
            x = np.linspace(-5, 5, prob_2d.shape[0])
            y = np.linspace(-5, 5, prob_2d.shape[1])
            X, Y = np.meshgrid(x, y)
            
            # 3D surface
            surf = self.ax.plot_surface(X, Y, prob_2d, cmap=self.colormap,
                                       alpha=0.8, edgecolor='none')
            
            # Add contour projection
            self.ax.contour(X, Y, prob_2d, zdir='z', offset=-0.1, 
                           cmap=self.colormap, alpha=0.5)
            
            # Update title with evolution info
            stats = self.wave_function.get_quantum_stats()
            self.ax.set_title(f"D-C Quantum Evolution - Step {frame}\n"
                             f"Time: {stats['time']:.3f}, "
                             f"Total Prob: {stats['total_probability']:.3f}",
                             color='white', fontsize=14)
            
            return [surf]
        
        # Create animation
        self.animation = animation.FuncAnimation(
            self.fig, animate, frames=steps, interval=100, blit=False, repeat=True)
        
        self.is_animating = True
        
    def render_dc_orbital_evolution(self, evolution_steps: int = 20):
        """Render orbital evolution using keya D-C operators."""
        if not self.orbital:
            print("❌ No orbital loaded!")
            return
            
        print(f"🌀 Starting D-C orbital evolution ({evolution_steps} steps)...")
        
        def animate(frame):
            # Clear and setup
            self.ax.clear()
            self._setup_display()
            
            # Safety check for orbital
            if not self.orbital:
                return []
            
            # Apply D-C evolution every few frames
            if frame > 0 and frame % 3 == 0:
                self.orbital.evolve_with_dc_operators(1)
            
            # Get current 2D slices
            xy_slice = self.orbital.get_probability_density_2d("xy")
            
            # Create surface
            max_r = self.orbital.max_radius
            x = np.linspace(-max_r, max_r, xy_slice.shape[0])
            y = np.linspace(-max_r, max_r, xy_slice.shape[1])
            X, Y = np.meshgrid(x, y)
            
            # 3D surface of probability density
            surf = self.ax.plot_surface(X, Y, xy_slice.T, cmap=self.colormap,
                                       alpha=0.9, edgecolor='none')
            
            # Add wireframe for structure
            self.ax.plot_wireframe(X, Y, xy_slice.T, alpha=0.3, color='white', linewidth=0.5)
            
            # Update title
            info = self.orbital.get_orbital_info()
            dc_stats = self.orbital.dc_wave_function.get_quantum_stats()
            
            self.ax.set_title(f"D-C {info['orbital_type']} Evolution - Frame {frame}\n"
                             f"DC Time: {dc_stats['time']:.3f}, "
                             f"Containment: {dc_stats['containment_type']}",
                             color='white', fontsize=14)
            
            return [surf]
        
        # Create animation
        self.animation = animation.FuncAnimation(
            self.fig, animate, frames=evolution_steps * 3, interval=200, 
            blit=False, repeat=False)
        
        self.is_animating = True
        
    def render_quantum_superposition(self) -> None:
        """Render quantum superposition visualization."""
        print("🌈 Rendering quantum superposition...")
        
        # Load superposition wave function
        self.load_wave_function(WaveFunctionType.SUPERPOSITION, (30, 30, 30))
        
        # Clear and setup
        self.ax.clear()
        self._setup_display()
        
        # Safety check for wave function
        if not self.wave_function:
            print("❌ No wave function loaded for superposition!")
            return
        
        # Get probability density and phase
        prob_2d = self.wave_function.get_probability_density_2d()
        phase_2d = self.wave_function.get_phase_2d()
        
        # Create coordinate grids
        x = np.linspace(-3, 3, prob_2d.shape[0])
        y = np.linspace(-3, 3, prob_2d.shape[1])
        X, Y = np.meshgrid(x, y)
        
        # Plot probability surface
        self.ax.plot_surface(X, Y, prob_2d, cmap=self.colormap,
                            alpha=0.7, edgecolor='none')
        
        # Plot phase as wireframe offset
        self.ax.plot_wireframe(X, Y, phase_2d + np.max(prob_2d) + 0.5,
                              alpha=0.5, color='cyan', linewidth=1)
        
        # Add interference pattern projection
        interference = prob_2d * np.cos(phase_2d)
        self.ax.contour(X, Y, interference, zdir='z', offset=-0.2,
                       levels=10, alpha=0.6, colors='yellow')
        
        self.ax.set_title("Quantum Superposition\nProbability (surface) + Phase (wireframe)",
                         color='white', fontsize=14)
        
        plt.tight_layout()
    
    def save_quantum_snapshot(self, filename: str):
        """Save current quantum visualization."""
        plt.savefig(filename, dpi=150, facecolor='black', bbox_inches='tight')
        print(f"📸 Quantum snapshot saved: {filename}")
    
    def show(self, auto_rotate: bool = True):
        """Display the quantum visualization."""
        
        if auto_rotate and not self.is_animating:
            # Add gentle rotation for better 3D perception
            def rotate(frame):
                self.ax.view_init(elev=30, azim=frame * 2)
                return []
            
            self.animation = animation.FuncAnimation(
                self.fig, rotate, frames=180, interval=50, blit=False, repeat=True)
        
        # Add interactive controls text
        controls_text = ("Controls:\n"
                        "Mouse: Rotate view\n"
                        "Scroll: Zoom\n"
                        "Press Q to quit")
        
        self.fig.text(0.02, 0.02, controls_text, fontsize=9, color='white',
                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        plt.show()
    
    def stop_animation(self):
        """Stop any running animation."""
        if self.animation:
            self.animation.event_source.stop()
            self.is_animating = False
            print("🛑 Animation stopped")


def create_quantum_demo(demo_type: str = "orbital") -> QuantumRenderer:
    """Create a quantum demo with predefined settings."""
    
    renderer = QuantumRenderer(figure_size=(14, 10))
    
    if demo_type == "orbital":
        # Hydrogen orbital demo
        renderer.load_orbital(OrbitalType.P_2PZ, grid_size=60)
        renderer.render_orbital_3d_isosurface(0.02)
        
    elif demo_type == "evolution":
        # Wave function evolution demo
        renderer.load_wave_function(WaveFunctionType.GAUSSIAN, (25, 25, 25))
        renderer.render_wave_function_evolution(30)
        
    elif demo_type == "superposition":
        # Quantum superposition demo
        renderer.render_quantum_superposition()
        
    elif demo_type == "dc_orbital":
        # D-C orbital evolution demo
        renderer.load_orbital(OrbitalType.S_2S, grid_size=50)
        renderer.render_dc_orbital_evolution(15)
        
    else:
        # Default: multiple orbitals
        renderer.load_orbital(OrbitalType.D_3DZ2, grid_size=50)
        renderer.render_orbital_2d_slices()
    
    return renderer 