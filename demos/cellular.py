#!/usr/bin/env python3
"""
🌟 KEYA INTERACTIVE CELLULAR AUTOMATA 🌟

This module demonstrates a cellular automata widget system, built following a
Model-View-Controller (MVC) pattern. It combines a JAX-based simulation backend
(the Model) with a Matplotlib-based interactive GUI (the View), mediated by
a state management class (the Controller).
"""
import argparse
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from jax.scipy.signal import convolve2d
from matplotlib.colors import ListedColormap

from demos.reporting.registry import register_demo

# --- 1. The Controller: State Management and Application Logic ---

class InteractionMode(Enum):
    """Defines how the user interacts with the grid."""
    DRAW = "draw"
    ERASE = "erase"
    TOGGLE = "toggle"

@dataclass
class GridState:
    """A simple dataclass to hold the UI's state."""
    matrix: jnp.ndarray
    generation: int
    evolution_active: bool

class CellularWidget:
    """
    The Controller. It connects the simulation (Model) to the GUI (View).
    
    It holds the application state (generation count, interaction mode) and
    interprets user input to command the model and update the view.
    """
    def __init__(self, width: int, height: int, evolution_speed: float = 0.1):
        self.width = width
        self.height = height
        self.evolution_speed = evolution_speed
        self.interaction_mode = InteractionMode.DRAW
        
        self.automaton = CellularAutomaton(height, width) # The Model
        self.state = GridState(
            matrix=self.automaton.create_random_grid(),
            generation=0,
            evolution_active=False,
        )

    def step_evolution(self):
        """Commands the model to compute the next state."""
        if self.state.evolution_active:
            self.state.matrix = self.automaton.step(self.state.matrix)
            self.state.generation += 1

    def handle_interaction(self, x: int, y: int):
        """Interprets raw user input and updates the model's grid."""
        if not (0 <= y < self.height and 0 <= x < self.width):
            return

        if self.interaction_mode == InteractionMode.DRAW:
            self.state.matrix = self.state.matrix.at[y, x].set(1)
        elif self.interaction_mode == InteractionMode.ERASE:
            self.state.matrix = self.state.matrix.at[y, x].set(0)
        elif self.interaction_mode == InteractionMode.TOGGLE:
            current_val = self.state.matrix[y, x]
            self.state.matrix = self.state.matrix.at[y, x].set(1 - current_val)

    def start_evolution(self):
        self.state.evolution_active = True
    
    def stop_evolution(self):
        self.state.evolution_active = False

    def reset(self):
        self.state.matrix = self.automaton.create_random_grid()
        self.state.generation = 0
        self.state.evolution_active = False

# --- 2. The Model: Pure Simulation Logic ---

class CellularAutomaton:
    """
    The Model. A pure, headless, high-performance JAX implementation of
    Conway's Game of Life. It knows nothing about GUIs or user interaction.
    """
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
        # The 3x3 kernel to count neighbors
        self.kernel = jnp.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=jnp.int32)

    def create_random_grid(self) -> jax.Array:
        """Creates a grid with a random initial state."""
        key = jax.random.PRNGKey(0)
        return jax.random.randint(key, (self.height, self.width), 0, 2, dtype=jnp.int32)

    @jax.jit
    def step(self, grid: jax.Array) -> jax.Array:
        """Performs a single, JIT-compiled step of the simulation."""
        num_neighbors = convolve2d(grid, self.kernel, mode='same', boundary='wrap')
        survives = (grid == 1) & ((num_neighbors == 2) | (num_neighbors == 3))
        born = (grid == 0) & (num_neighbors == 3)
        return (survives | born).astype(jnp.int32)

# --- 3. The View: Rendering and Raw Input Handling ---

class WidgetRenderer:
    """
    The View. Renders the grid and captures raw user input.
    
    It knows how to draw a grid and report mouse/key events, but does not
    know what those events mean.
    """
    def __init__(self, widget: CellularWidget):
        self.widget = widget # A reference to the controller to send events to
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.colormap = ListedColormap(['#2E2E2E', '#F0F0F0']) # Dark grey for 0, Light grey for 1
        self.image: Optional[Any] = None
        self.animation: Optional[Any] = None
        self.mouse_pressed = False

    def _on_mouse_event(self, event):
        """Generic handler for mouse press and move events."""
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        self.widget.handle_interaction(x, y) # Send raw coordinates to controller
        self.update_display()

    def _on_key_press(self, event):
        """Handles keyboard shortcuts by changing the controller's state."""
        if event.key == ' ':
            if self.widget.state.evolution_active:
                self.widget.stop_evolution()
            else:
                self.widget.start_evolution()
        elif event.key == 'r':
            self.widget.reset()
        elif event.key == '1':
            self.widget.interaction_mode = InteractionMode.DRAW
        elif event.key == '2':
            self.widget.interaction_mode = InteractionMode.ERASE
        elif event.key == '3':
            self.widget.interaction_mode = InteractionMode.TOGGLE
        elif event.key == 'q':
            self.stop()
        self.update_display() # Redraw to reflect mode change

    def update_display(self):
        """Updates the matplotlib view from the controller's state."""
        status_text = "▶️ EVOLVING" if self.widget.state.evolution_active else "⏸️ PAUSED"
        title = (f"JAX Cellular Automata | Gen: {self.widget.state.generation} | "
                 f"Mode: {self.widget.interaction_mode.name} (Keys: 1-3) | "
                 f"{status_text}")
        self.ax.set_title(title)
        
        if self.image is None:
            self.image = self.ax.imshow(self.widget.state.matrix, cmap=self.colormap, interpolation='nearest')
        else:
            self.image.set_data(self.widget.state.matrix)
        self.fig.canvas.draw_idle()

    def _animation_func(self, frame):
        """Function called by the animation timer."""
        self.widget.step_evolution()
        self.update_display()
        return [self.image] if self.image else []

    def show(self):
        """Connects event handlers and shows the main window."""
        self.fig.canvas.mpl_connect('button_press_event', lambda e: (setattr(self, 'mouse_pressed', True), self._on_mouse_event(e)))
        self.fig.canvas.mpl_connect('button_release_event', lambda e: setattr(self, 'mouse_pressed', False))
        self.fig.canvas.mpl_connect('motion_notify_event', lambda e: self.mouse_pressed and self._on_mouse_event(e))
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        self.update_display()
        self.animation = animation.FuncAnimation(
            self.fig, self._animation_func, interval=int(self.widget.evolution_speed * 1000), blit=True, cache_frame_data=False
        )
        plt.show()

    def stop(self):
        if self.animation:
            self.animation.event_source.stop()
        plt.close(self.fig)

# --- Demo Registration and Main Execution ---

@register_demo(
    title="Interactive Cellular Automata (MVC)",
    artifacts=[],
    claims=[
        "A JAX backend (Model) can power a real-time interactive simulation.",
        "A Matplotlib GUI (View) can be cleanly separated from the simulation logic.",
        "A Controller class can mediate between the Model and View, managing application state."
    ],
    findings="This demo successfully implements a clean Model-View-Controller (MVC) architecture. The `CellularAutomaton` class handles the high-performance computation, the `WidgetRenderer` provides the interactive GUI, and the `CellularWidget` acts as the controller. This robust model proves the core logic is decoupled from its presentation."
)
def main():
    """
    This demo showcases an interactive 2D cellular automaton (Conway's Game of Life)
    built with a JAX backend and a Matplotlib frontend, following an MVC pattern.
    Users can draw, erase, and toggle cells, and watch the simulation evolve in real time.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the Interactive Cellular Automata Demo.")
    parser.add_argument('--width', type=int, default=80, help='Width of the grid.')
    parser.add_argument('--height', type=int, default=60, help='Height of the grid.')
    parser.add_argument('--speed', type=float, default=0.05, help='Evolution speed in seconds per step.')
    
    args = parser.parse_args()

    # Create the components and run the application
    widget = CellularWidget(width=args.width, height=args.height, evolution_speed=args.speed)
    renderer = WidgetRenderer(widget)
    renderer.show()

if __name__ == "__main__":
    main() 