"""
The Kéya Symbolic System.

This module defines the fundamental vocabulary of the Σ-Calculus. It contains:
- The `Glyph` enum, the core symbolic alphabet.
- Mappings between `Glyph` objects and their integer representations for computation.
- The `omega` (ω) transformation, the most basic symbolic operator.
"""
from enum import Enum
from typing import Dict

# --- The Symbolic Alphabet ---

class Glyph(Enum):
    """The fundamental symbols (glyphs) in the resonance field."""
    
    VOID = "∘"     # Emptiness, zero
    DOWN = "∇"     # Primal glyph, initiator
    UP = "∆"       # Transformed glyph, responder
    UNITY = "I"    # Contained/stable glyph, identity
    FLOW = "∿"     # Dynamic glyph, connection
    FLUX = "↯"     # Higher-energy state
    STASIS = "⌽"    # Stable, non-interactive state

    def __str__(self):
        """Return the actual symbol character for printing."""
        return self.value

# --- Computational Mappings ---

# Mapping from symbolic glyphs to their integer representation for computation.
GLYPH_TO_INT: Dict[Glyph, int] = {
    Glyph.VOID: 0,
    Glyph.DOWN: 1,
    Glyph.UP: 2,
    Glyph.UNITY: 3,
    Glyph.FLOW: 4,
    Glyph.FLUX: 5,
    Glyph.STASIS: 6,
}

# Reverse mapping for converting computed integers back to symbolic glyphs.
INT_TO_GLYPH: Dict[int, Glyph] = {v: k for k, v in GLYPH_TO_INT.items()}

# --- Foundational Transformation (ω) ---

# Defines the state transitions for the fundamental omega (ω) operator.
_OMEGA_TRANSFORMS: Dict[Glyph, Glyph] = {
    Glyph.VOID: Glyph.DOWN,      # Void becomes Down
    Glyph.DOWN: Glyph.UP,        # Down becomes Up
    Glyph.UP: Glyph.DOWN,        # Up reverts to Down
    Glyph.UNITY: Glyph.UNITY,    # Unity is a fixed point
    Glyph.FLOW: Glyph.FLOW,      # Flow is a fixed point
    Glyph.FLUX: Glyph.STASIS,    # Flux and Stasis are a binary pair
    Glyph.STASIS: Glyph.FLUX,
}

def omega(glyph: Glyph) -> Glyph:
    """
    The fundamental glyph transformation ω.
    
    This operator represents the most basic, predictable state change in the
    symbolic system. It maps a glyph to its successor state.
    """
    return _OMEGA_TRANSFORMS[glyph] 