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
    """Provides glyph replacement logic for the REPL."""

    SYMBOL_REPLACEMENTS: Dict[str, str] = {
        # Core glyphs
        'void': '∅', 'empty': '∅',
        'down': '▽', 'primal': '▽',
        'up': '△', 'transformed': '△',
        'unity': '⊙', 'contained': '⊙', 'stable': '⊙',
        'flow': '⊕', 'dynamic': '⊕',

        # Operator symbols
        'tensor': '⊗',
        'growth': '↑',
        'descent': 'ℓ',
        'reflect': '~', 'reflection': '~',
        'wild': 'Ϟ',
        'tame': '§',
        'wildtame': '∮', 'cycle': '∮'
    }

    def expand_word(self, word: str) -> Optional[str]:
        """Expands a word to its corresponding glyph if a replacement exists."""
        return self.SYMBOL_REPLACEMENTS.get(word.lower())

    def get_replacements(self) -> Dict[str, str]:
        """Returns the dictionary of all symbol replacements."""
        return self.SYMBOL_REPLACEMENTS 