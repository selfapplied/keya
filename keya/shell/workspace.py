"""Workspace state management for the Keya REPL."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from ..dsl.ast import Definition


@dataclass
class WorkspaceState:
    """Represents the current workspace/session state."""
    name: str = "default"
    variables: Dict[str, Any] = field(default_factory=dict)
    programs: Dict[str, Definition] = field(default_factory=dict)
    history: List[str] = field(default_factory=list)

    def save_to_file(self, path: Path):
        """Save workspace state to file."""
        try:
            data = {
                'name': self.name,
                'variables': {k: str(v) for k, v in self.variables.items()},
                'history': self.history[-50:]  # Last 50 commands
            }
            path.write_text(json.dumps(data, indent=2))
        except (OSError, PermissionError, UnicodeEncodeError):
            # Ignore filesystem and encoding errors during workspace save
            pass

    def load_from_file(self, path: Path):
        """Load workspace state from file."""
        if path.exists():
            try:
                data = json.loads(path.read_text())
                self.name = data.get('name', 'default')
                self.history = data.get('history', [])
            except (json.JSONDecodeError, OSError, UnicodeDecodeError, KeyError):
                # Ignore corrupted or malformed workspace files
                pass 