"""Cellular Automata Widget using Keya D-C Language."""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..core.engine import Engine
from ..dsl.ast import ContainmentType, Glyph


class InteractionMode(Enum):
    """Different modes of user interaction with the cellular grid."""
    CLICK_TOGGLE = "click_toggle"     # Click to toggle cell state
    RIPPLE = "ripple"                 # Click creates expanding ripples
    DRAW = "draw"                     # Click and drag to draw
    ERASE = "erase"                   # Click and drag to erase


@dataclass
class GridState:
    """Represents the state of a cellular automata grid."""
    matrix: np.ndarray
    generation: int
    evolution_active: bool
    last_interaction: Optional[Tuple[int, int]]  # Last clicked position
    

class CellularWidget:
    """A widget that displays evolving cellular automata using keya D-C rules."""
    
    def __init__(self, 
                 width: int = 20, 
                 height: int = 20,
                 containment_type: ContainmentType = ContainmentType.BINARY,
                 interaction_mode: InteractionMode = InteractionMode.RIPPLE,
                 evolution_speed: float = 0.1):  # Seconds between generations
        
        self.width = width
        self.height = height
        self.containment_type = containment_type
        self.interaction_mode = interaction_mode
        self.evolution_speed = evolution_speed
        
        # Initialize the keya D-C engine
        self.engine = Engine()
        
        # Create initial grid state
        self.state = GridState(
            matrix=self._create_initial_grid(),
            generation=0,
            evolution_active=False,
            last_interaction=None
        )
        
        # Track evolution history for visualization
        self.history: List[np.ndarray] = []
        self.max_history = 100  # Keep last 100 generations
        
    def _create_initial_grid(self) -> np.ndarray:
        """Create the initial grid state."""
        # Start with mostly void (∅) cells
        grid = np.full((self.height, self.width), self._glyph_to_number(Glyph.VOID))
        
        # Add some random seed points
        num_seeds = max(1, (self.width * self.height) // 20)
        for _ in range(num_seeds):
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            grid[y, x] = self._glyph_to_number(Glyph.UP)  # △ seed points
            
        return grid
    
    def _glyph_to_number(self, glyph: Glyph) -> float:
        """Convert glyphs to numeric values for computation."""
        glyph_values = {
            Glyph.VOID: 0.0,     # ∅ - empty space
            Glyph.DOWN: -1.0,    # ▽ - negative energy
            Glyph.UP: 1.0,       # △ - positive energy
            Glyph.UNITY: 0.5,    # ⊙ - neutral/boundary
            Glyph.FLOW: 2.0,     # ⊕ - high energy flow
        }
        return glyph_values.get(glyph, 0.0)
    
    def _number_to_glyph(self, value: float) -> Glyph:
        """Convert numeric values back to glyphs."""
        # Map values to glyphs based on ranges
        if abs(value) < 0.1:
            return Glyph.VOID      # ∅
        elif value < -0.5:
            return Glyph.DOWN      # ▽
        elif value > 1.5:
            return Glyph.FLOW      # ⊕
        elif 0.3 <= value <= 0.7:
            return Glyph.UNITY     # ⊙
        else:
            return Glyph.UP        # △
    
    def handle_interaction(self, x: int, y: int) -> bool:
        """Handle user interaction at grid position (x, y). Returns True if grid changed."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
            
        self.state.last_interaction = (x, y)
        
        if self.interaction_mode == InteractionMode.CLICK_TOGGLE:
            # Simple toggle between void and up
            current = self.state.matrix[y, x]
            if current == self._glyph_to_number(Glyph.VOID):
                self.state.matrix[y, x] = self._glyph_to_number(Glyph.UP)
            else:
                self.state.matrix[y, x] = self._glyph_to_number(Glyph.VOID)
            return True
            
        elif self.interaction_mode == InteractionMode.RIPPLE:
            # Create ripple effect using D operator (dissonance)
            self._create_ripple(x, y)
            return True
            
        elif self.interaction_mode == InteractionMode.DRAW:
            # Draw positive energy
            self.state.matrix[y, x] = self._glyph_to_number(Glyph.UP)
            return True
            
        elif self.interaction_mode == InteractionMode.ERASE:
            # Erase to void
            self.state.matrix[y, x] = self._glyph_to_number(Glyph.VOID)
            return True
            
        return False
    
    def _create_ripple(self, center_x: int, center_y: int, radius: int = 3):
        """Create a ripple effect centered at (center_x, center_y)."""
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x = center_x + dx
                y = center_y + dy
                
                if 0 <= x < self.width and 0 <= y < self.height:
                    distance = np.sqrt(dx*dx + dy*dy)
                    if distance <= radius:
                        # Create wave pattern based on distance
                        intensity = (radius - distance) / radius
                        if distance < 1:
                            # Center gets high energy
                            self.state.matrix[y, x] = self._glyph_to_number(Glyph.FLOW)
                        elif distance < 2:
                            # Middle ring gets positive energy
                            self.state.matrix[y, x] = self._glyph_to_number(Glyph.UP)
                        else:
                            # Outer ring gets unity (boundary)
                            self.state.matrix[y, x] = self._glyph_to_number(Glyph.UNITY)
    
    def start_evolution(self):
        """Start the cellular automata evolution."""
        self.state.evolution_active = True
        print(f"🔄 Started cellular evolution with {self.containment_type.value} containment")
    
    def stop_evolution(self):
        """Stop the cellular automata evolution."""
        self.state.evolution_active = False
        print("⏸️  Stopped cellular evolution")
    
    def step_evolution(self) -> bool:
        """Perform one evolution step using keya D-C operators. Returns True if changed."""
        if not self.state.evolution_active:
            return False
            
        # Store current state in history
        self.history.append(self.state.matrix.copy())
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
        # Use keya D-C language to evolve the grid
        # Create a keya program that applies one DC cycle
        keya_program = f"""
matrix evolution {{
    step {{
        result = DC(grid, {self.containment_type.value}, 1)
    }}
}}
"""
        
        try:
            # Set the current grid as a variable in the engine
            self.engine.variables['grid'] = self.state.matrix
            
            # Execute the evolution step
            result = self.engine.execute_program(keya_program.strip())
            
            # Get the evolved matrix
            if 'grid' in self.engine.variables:
                new_matrix = self.engine.variables.get('result', self.state.matrix)
                if isinstance(new_matrix, np.ndarray):
                    self.state.matrix = new_matrix
                    self.state.generation += 1
                    return True
                    
        except Exception as e:
            print(f"Evolution error: {e}")
            
        return False
    
    def get_display_grid(self) -> List[List[str]]:
        """Get the grid as displayable glyphs."""
        grid = []
        for row in self.state.matrix:
            glyph_row = []
            for value in row:
                glyph = self._number_to_glyph(value)
                glyph_symbols = {
                    Glyph.VOID: '∅',
                    Glyph.DOWN: '▽', 
                    Glyph.UP: '△',
                    Glyph.UNITY: '⊙',
                    Glyph.FLOW: '⊕'
                }
                glyph_row.append(glyph_symbols[glyph])
            grid.append(glyph_row)
        return grid
    
    def get_color_grid(self) -> List[List[Tuple[int, int, int]]]:
        """Get the grid as RGB color values for rendering."""
        colors = []
        for row in self.state.matrix:
            color_row = []
            for value in row:
                glyph = self._number_to_glyph(value)
                # Map glyphs to colors
                glyph_colors = {
                    Glyph.VOID: (0, 0, 0),         # Black - empty
                    Glyph.DOWN: (100, 0, 200),     # Purple - negative energy  
                    Glyph.UP: (255, 100, 0),       # Orange - positive energy
                    Glyph.UNITY: (150, 150, 150),  # Gray - neutral
                    Glyph.FLOW: (255, 255, 0),     # Yellow - high energy
                }
                color_row.append(glyph_colors[glyph])
            colors.append(color_row)
        return colors
    
    def reset(self):
        """Reset the widget to initial state."""
        self.state = GridState(
            matrix=self._create_initial_grid(),
            generation=0,
            evolution_active=False,
            last_interaction=None
        )
        self.history.clear()
        print("🔄 Widget reset to initial state")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the current state."""
        grid = self.state.matrix
        total_cells = grid.size
        
        # Count each glyph type
        void_count = np.sum(np.abs(grid) < 0.1)
        down_count = np.sum(grid < -0.5)
        up_count = np.sum((grid > 0.1) & (grid < 1.5))
        unity_count = np.sum((grid >= 0.3) & (grid <= 0.7))
        flow_count = np.sum(grid > 1.5)
        
        return {
            'generation': self.state.generation,
            'total_cells': total_cells,
            'void_cells': int(void_count),
            'down_cells': int(down_count), 
            'up_cells': int(up_count),
            'unity_cells': int(unity_count),
            'flow_cells': int(flow_count),
            'evolution_active': self.state.evolution_active,
            'containment_type': self.containment_type.value,
            'interaction_mode': self.interaction_mode.value
        } 