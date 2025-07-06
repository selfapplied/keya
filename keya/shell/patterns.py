"""
PatternLibrary - Central location for Keya syntax snippets.

- Stores named program templates like:
    - matrix_dims
    - wildtame_cycle
    - grammar program

This lets both completions and meta-help pull from the same structure.
"""
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class PatternLibrary:
    """Central location for Keya syntax snippets."""
    patterns: Dict[str, str] = field(default_factory=lambda: {
        "matrix_dims": "[{rows}, {cols}, {fill}]",
        "wildtame_cycle": "∮({matrix}, {type}, {iterations})",
        "wild": "Ϟ({matrix})",
        "tame": "§({matrix}, {type})",
        "matrix_program": "matrix {name} {\n  ops {\n    {content}\n  }\n}",
        "grammar_program": "grammar {name} {\n  rules {\n    {content}\n  }\n}",
    })

    def get_pattern(self, name: str) -> str:
        """Returns a specific pattern by name."""
        return self.patterns.get(name, "")

    def get_all_patterns(self) -> Dict[str, str]:
        """Returns all patterns."""
        return self.patterns 