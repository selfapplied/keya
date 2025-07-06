"""
CommandRouter - Determines how to interpret and route input in the Keya REPL.

- If the input is a meta-command (starts with ':'), route to the appropriate REPL handler.
- If the input contains program keywords (`matrix`, `grammar`, `resonance`), treat it as a full program.
- If the input is a simple expression, wrap it in a temporary matrix program and evaluate.
- If all else fails, attempt variable lookup or raise a parse error.
- Records output into the workspace if appropriate.

This module should be decoupled from prompt_toolkit.
"""
from typing import Callable, Any

from ..core.engine import Engine
from ..dsl.parser import ParseError
from .workspace import WorkspaceState


class CommandRouter:
    """Determines how to interpret and route input in the Keya REPL."""

    def __init__(
        self,
        engine: Engine,
        workspace: WorkspaceState,
        display_result: Callable[[Any], None],
        display_error: Callable[[str, str, str], None],
        handle_meta_command: Callable[[str], None],
    ):
        self.engine = engine
        self.workspace = workspace
        self.display_result = display_result
        self.display_error = display_error
        self.handle_meta_command = handle_meta_command

    def process_input(self, line: str):
        """Process a complete input line or program."""
        # Add to history
        self.workspace.history.append(line)

        # Handle meta-commands
        if line.startswith(':'):
            self.handle_meta_command(line[1:])
            return

        # Handle simple commands
        if line.lower() in ['exit', 'quit']:
            raise EOFError()

        # Try to parse and execute as language
        try:
            # Check if it's a simple expression or full program
            if any(keyword in line for keyword in ['matrix', 'grammar', 'resonance']):
                # Full program
                result = self.engine.execute_program(line)
                if result:
                    self.display_result(result)
            else:
                # Simple expression - wrap in a minimal program
                wrapped = f"matrix temp {{ ops {{ result = {line} }} }}"
                try:
                    result = self.engine.execute_program(wrapped)
                    if result and 'result' in result:
                        self.display_result(result['result'])
                except Exception:
                    # Fallback: try as variable lookup
                    if line in self.workspace.variables:
                        self.display_result(self.workspace.variables[line])
                    else:
                        raise ParseError(f"Unknown variable or invalid expression: {line}")

        except ParseError as e:
            self.display_error("Parse Error", str(e), line)
        except Exception as e:
            self.display_error("Execution Error", str(e), line) 