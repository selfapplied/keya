"""
SymbolExpander - Provides glyph replacement logic for the REPL.

- Expands words like 'void', 'wild', or 'unity' into ∅, Ϟ, ⊙.
- Supports:
    - word → glyph on space key
    - autosuggestion during typing
    - tab completion preview
- Used by both auto-suggest and keybindings.

You can optionally expose:
- expand_word(word: str) → Optional[glyph]
- get_replacements() → Dict[str, str]
"""
from typing import Dict, Optional


class SymbolExpander:
    """Expands symbol words into their corresponding glyphs for the DC language."""

    def __init__(self):
        # Core DC glyphs
        self.replacements = {
            'void': '∅',
            'empty': '∅',
            'null': '∅',
            'up': '△',
            'triangle': '△',
            'tri': '△',
            'down': '▽',
            'inverted': '▽',
            'unity': '⊙',
            'one': '⊙',
            'unit': '⊙',
            'flow': '⊕',
            'plus': '⊕',
            'cross': '⊕',
            
            # Extended mathematical glyphs
            'tensor': '⊗',
            'times': '⊗',
            'mult': '⊗',
            'growth': '↑',
            'ascent': '↑',
            'descent': 'ℓ',
            'down_arrow': 'ℓ',
            'reflect': '~',
            'wave': '~',
            'tilde': '~',
            'dissonance': '𝔻',
            'discord': '𝔻',
            'chaos': '𝔻',
            'containment': 'ℂ',
            'complex': 'ℂ',
            'field': 'ℂ',
            'infinity': '∞',
            'inf': '∞',
            'infinite': '∞',
            
            # Tesla 3-6-9 Trinity Symbols
            'tesla3': '③',
            'three': '③',
            'generator': '③',
            'vibration': '③',
            'tesla6': '⑥', 
            'six': '⑥',
            'resonator': '⑥',
            'resonance': '⑥',
            'symmetry': '⑥',
            'tesla9': '⑨',
            'nine': '⑨',
            'annihilator': '⑨',
            'singularity': '⑨',
            'collapse': '⑨',
            
            # Digital Root and Renormalization Symbols
            'phi': 'φ',
            'tithing': 'φ', 
            'digitalroot': 'φ',
            'compression': 'φ',
            'mersenne': 'M',
            'boundary': 'M',
            'renorm': 'M',
            'swap': '⇄',
            'flip': '⇄',
            'invert': '⇄',
            'tower': '🗼',
            'powertower': '🗼',
            'iteration': '🗼',
            'sierpinski': '🔺',
            'fractal': '🔺',
            'triangle': '🔺',
            'scrub': '🧽',
            'clean': '🧽',
            'reduction': '🧽',
            
            # Trinity State Symbols
            'trinity': '⧬',
            'stable': '⧬',
            'attractor': '⧬',
            'basecase': '⧬',
            'twisted': '🌀',
            'ring': '🌀',
            'postswap': '🌀',
            'glitched': '⚡',
            'anomaly': '⚡',
            'euler': '⚡',
        }

    def expand_word(self, word: str) -> Optional[str]:
        """Expands a word to its corresponding glyph if a replacement exists."""
        return self.replacements.get(word.lower())

    def get_replacements(self) -> Dict[str, str]:
        """Returns the dictionary of all symbol replacements."""
        return self.replacements 