"""Widget Renderer for Keya Cellular Automata."""

from typing import Tuple, Optional, Any

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import numpy as np

from .cellular import CellularWidget, InteractionMode
from ..dsl.ast import ContainmentType


class WidgetRenderer:
    """Renders and manages cellular automata widgets with real-time interaction."""
    
    def __init__(self, widget: CellularWidget, cell_size: int = 20):
        self.widget = widget
        self.cell_size = cell_size
        
        # Set up the matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.suptitle("Keya Cellular Automata Widget", fontsize=16)
        
        # Create custom colormap for glyphs
        self.colormap = ListedColormap([
            '#000000',  # Black - void (∅)
            '#6400C8',  # Purple - down (▽) 
            '#FF6400',  # Orange - up (△)
            '#969696',  # Gray - unity (⊙)
            '#FFFF00'   # Yellow - flow (⊕)
        ])
        
        # Initialize the display
        self.image = None
        self.text_display = None
        self.animation: Optional[Any] = None  # FuncAnimation type
        self.is_running = False
        
        # Mouse interaction state
        self.mouse_pressed = False
        self.last_mouse_pos: Optional[Tuple[int, int]] = None
        
    def _grid_to_image_array(self) -> np.ndarray:
        """Convert the widget grid to a numpy array for matplotlib."""
        grid = self.widget.state.matrix
        
        # Map values to color indices (0-4)
        image_array = np.zeros_like(grid, dtype=int)
        
        # Map glyph values to color indices
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                value = grid[i, j]
                if abs(value) < 0.1:
                    image_array[i, j] = 0  # Void
                elif value < -0.5:
                    image_array[i, j] = 1  # Down
                elif value > 1.5:
                    image_array[i, j] = 4  # Flow
                elif 0.3 <= value <= 0.7:
                    image_array[i, j] = 3  # Unity
                else:
                    image_array[i, j] = 2  # Up
                    
        return image_array
    
    def _on_mouse_press(self, event):
        """Handle mouse press events."""
        if event.inaxes != self.ax:
            return
            
        self.mouse_pressed = True
        x, y = int(event.xdata), int(event.ydata)
        self.last_mouse_pos = (x, y)
        
        # Handle interaction based on mode
        if self.widget.handle_interaction(x, y):
            self.update_display()
    
    def _on_mouse_release(self, event):
        """Handle mouse release events."""
        self.mouse_pressed = False
        self.last_mouse_pos = None
    
    def _on_mouse_move(self, event):
        """Handle mouse move events (for drawing/erasing)."""
        if not self.mouse_pressed or event.inaxes != self.ax:
            return
            
        x, y = int(event.xdata), int(event.ydata)
        
        # For draw/erase modes, continue interaction on mouse move
        if self.widget.interaction_mode in [InteractionMode.DRAW, InteractionMode.ERASE]:
            if self.widget.handle_interaction(x, y):
                self.update_display()
        
        self.last_mouse_pos = (x, y)
    
    def _on_key_press(self, event):
        """Handle keyboard shortcuts."""
        if event.key == 'space':
            # Toggle evolution
            if self.widget.state.evolution_active:
                self.widget.stop_evolution()
            else:
                self.widget.start_evolution()
        elif event.key == 'r':
            # Reset widget
            self.widget.reset()
            self.update_display()
        elif event.key == '1':
            self.widget.interaction_mode = InteractionMode.CLICK_TOGGLE
            print("🔧 Mode: Click Toggle")
        elif event.key == '2':
            self.widget.interaction_mode = InteractionMode.RIPPLE
            print("🔧 Mode: Ripple")
        elif event.key == '3':
            self.widget.interaction_mode = InteractionMode.DRAW
            print("🔧 Mode: Draw")
        elif event.key == '4':
            self.widget.interaction_mode = InteractionMode.ERASE
            print("🔧 Mode: Erase")
        elif event.key == 'q':
            # Quit
            self.stop()
    
    def update_display(self):
        """Update the visual display with current widget state."""
        # Update the grid image
        image_array = self._grid_to_image_array()
        
        if self.image is None:
            # First time setup
            self.image = self.ax.imshow(
                image_array, 
                cmap=self.colormap, 
                vmin=0, 
                vmax=4,
                interpolation='nearest'
            )
            self.ax.set_xlim(-0.5, self.widget.width - 0.5)
            self.ax.set_ylim(-0.5, self.widget.height - 0.5)
            self.ax.set_aspect('equal')
        else:
            # Update existing image
            self.image.set_array(image_array)
        
        # Update stats display
        stats = self.widget.get_stats()
        status = "▶️ EVOLVING" if stats['evolution_active'] else "⏸️ PAUSED"
        
        stats_text = (
            f"{status} | Gen: {stats['generation']} | "
            f"Mode: {stats['interaction_mode']} | "
            f"Type: {stats['containment_type']}\n"
            f"∅:{stats['void_cells']} ▽:{stats['down_cells']} △:{stats['up_cells']} "
            f"⊙:{stats['unity_cells']} ⊕:{stats['flow_cells']}"
        )
        
        if self.text_display is None:
            self.text_display = self.fig.text(0.02, 0.02, stats_text, fontsize=10, 
                                            fontfamily='monospace', verticalalignment='bottom')
        else:
            self.text_display.set_text(stats_text)
        
        # Redraw
        self.fig.canvas.draw()
    
    def _animation_func(self, frame):
        """Animation function for matplotlib animation."""
        if self.widget.state.evolution_active:
            self.widget.step_evolution()
        self.update_display()
        # Both image and text_display are guaranteed to be Artists after update_display()
        artists = []
        if self.image is not None:
            artists.append(self.image)
        if self.text_display is not None:
            artists.append(self.text_display)
        return artists
    
    def show(self, auto_evolve: bool = False):
        """Show the widget renderer window."""
        print("🎮 KEYA CELLULAR AUTOMATA WIDGET 🎮")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("Controls:")
        print("  SPACE  - Toggle evolution")
        print("  R      - Reset grid")
        print("  1      - Click toggle mode")
        print("  2      - Ripple mode")
        print("  3      - Draw mode") 
        print("  4      - Erase mode")
        print("  Q      - Quit")
        print("  CLICK  - Interact with grid")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        # Set up event handlers
        self.fig.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        # Initial display
        self.update_display()
        
        # Start evolution if requested
        if auto_evolve:
            self.widget.start_evolution()
        
        # Start animation for real-time updates
        self.is_running = True
        self.animation = animation.FuncAnimation(
            self.fig, 
            self._animation_func,
            interval=int(self.widget.evolution_speed * 1000),  # Convert to milliseconds
            blit=False,
            cache_frame_data=False
        )
        
        # Show the window
        plt.show()
    
    def stop(self):
        """Stop the renderer and close the window."""
        self.is_running = False
        if self.animation:
            self.animation.event_source.stop()
        self.widget.stop_evolution()
        plt.close(self.fig)
        print("👋 Widget renderer stopped")
    
    def save_snapshot(self, filename: str):
        """Save a snapshot of the current state."""
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"📸 Snapshot saved to {filename}")
    
    def export_animation(self, filename: str, duration: int = 10, fps: int = 10):
        """Export an animation of the evolution."""
        print(f"🎬 Exporting {duration}s animation to {filename}...")
        
        # Start evolution
        was_evolving = self.widget.state.evolution_active
        self.widget.start_evolution()
        
        # Create animation
        frames = duration * fps
        anim = animation.FuncAnimation(
            self.fig, 
            self._animation_func,
            frames=frames,
            interval=1000//fps,
            blit=False
        )
        
        # Save animation
        anim.save(filename, writer='pillow', fps=fps)
        
        # Restore evolution state
        if not was_evolving:
            self.widget.stop_evolution()
            
        print(f"✅ Animation exported to {filename}")


def create_demo_widget(widget_type: str = "ripple") -> Tuple[CellularWidget, WidgetRenderer]:
    """Create a demo widget with preset configurations."""
    
    if widget_type == "ripple":
        widget = CellularWidget(
            width=30, 
            height=30,
            containment_type=ContainmentType.BINARY,
            interaction_mode=InteractionMode.RIPPLE,
            evolution_speed=0.2
        )
    elif widget_type == "draw":
        widget = CellularWidget(
            width=25,
            height=25, 
            containment_type=ContainmentType.DECIMAL,
            interaction_mode=InteractionMode.DRAW,
            evolution_speed=0.15
        )
    elif widget_type == "toggle":
        widget = CellularWidget(
            width=20,
            height=20,
            containment_type=ContainmentType.STRING,
            interaction_mode=InteractionMode.CLICK_TOGGLE,
            evolution_speed=0.3
        )
    else:
        # Default
        widget = CellularWidget()
    
    renderer = WidgetRenderer(widget)
    return widget, renderer 