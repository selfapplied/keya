"""Modern Keya D-C REPL - Language-first interactive shell."""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML, StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.shortcuts import print_formatted_text
from prompt_toolkit.styles import Style

from ..core.engine import Engine
from ..dsl.parser import parse, ParseError
from ..dsl.ast import (
    Glyph, ContainmentType, MatrixProgram, GrammarProgram, 
    ResonanceProgram, Definition
)


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
        except Exception:
            # Ignore save errors
            pass
    
    def load_from_file(self, path: Path):
        """Load workspace state from file."""
        if path.exists():
            try:
                data = json.loads(path.read_text())
                self.name = data.get('name', 'default')
                self.history = data.get('history', [])
            except (json.JSONDecodeError, Exception):
                # Ignore corrupted workspace files
                pass


class KeyaDCCompleter(Completer):
    """Advanced completer for keya D-C language."""
    
    def __init__(self, engine: Engine, workspace: WorkspaceState):
        self.engine = engine
        self.workspace = workspace
        
        # Core language keywords
        self.keywords = {
            'matrix', 'grammar', 'resonance', 'verify', 'trace',
            'arithmetic', 'strings', 'pattern', 'match', 'concat',
            'from', 'seed', 'length', 'rule'
        }
        
        # Operators and types
        self.operators = {'D', 'C', 'DC'}
        self.containment_types = {'binary', 'decimal', 'string', 'general'}
        
        # Glyph replacements
        self.glyph_replacements = {
            'void': '∅', 'empty': '∅',
            'down': '▽', 'primal': '▽',
            'up': '△', 'transformed': '△',
            'unity': '⊙', 'contained': '⊙', 'stable': '⊙',
            'flow': '⊕', 'dynamic': '⊕'
        }
        
        # Common patterns
        self.patterns = {
            'matrix_dims': '[{rows}, {cols}, {fill}]',
            'dc_cycle': 'DC({matrix}, {type}, {iterations})',
            'dissonance': 'D({matrix})',
            'containment': 'C({matrix}, {type})',
            'matrix_program': 'matrix {name} {\n  ops {\n    {content}\n  }\n}',
            'grammar_program': 'grammar {name} {\n  rules {\n    {content}\n  }\n}',
        }
    
    def get_completions(self, document: Document, complete_event):
        """Generate completions based on context."""
        word = document.get_word_before_cursor(WORD=True)
        line = document.current_line_before_cursor
        
        # Glyph replacements
        if word in self.glyph_replacements:
            yield Completion(
                self.glyph_replacements[word],
                start_position=-len(word),
                display_meta="glyph"
            )
        
        # Keywords
        for keyword in self.keywords:
            if keyword.startswith(word.lower()):
                yield Completion(
                    keyword,
                    start_position=-len(word),
                    display_meta="keyword"
                )
        
        # Operators
        for op in self.operators:
            if op.startswith(word.upper()):
                yield Completion(
                    op,
                    start_position=-len(word),
                    display_meta="operator"
                )
        
        # Containment types
        for ctype in self.containment_types:
            if ctype.startswith(word.lower()):
                yield Completion(
                    ctype,
                    start_position=-len(word),
                    display_meta="type"
                )
        
        # Session variables
        for var_name in self.workspace.variables:
            if var_name.startswith(word):
                var_value = self.workspace.variables[var_name]
                meta = f"{type(var_value).__name__}"
                if hasattr(var_value, 'shape'):
                    meta += f" {var_value.shape}"
                yield Completion(
                    var_name,
                    start_position=-len(word),
                    display_meta=meta
                )
        
        # Context-aware pattern suggestions
        if 'matrix' in line and not word:
            yield Completion(
                self.patterns['matrix_dims'].format(rows=3, cols=3, fill='∅'),
                start_position=0,
                display_meta="matrix pattern"
            )


class KeyaDCAutoSuggest(AutoSuggest):
    """Intelligent auto-suggestions for keya D-C."""
    
    def __init__(self, engine: Engine, workspace: WorkspaceState):
        self.engine = engine
        self.workspace = workspace
    
    def get_suggestion(self, buffer: Buffer, document: Document) -> Optional[Suggestion]:
        """Generate smart suggestions based on context and history."""
        text = document.text
        
        # Suggest closing braces
        if text.count('{') > text.count('}'):
            return Suggestion(' }')
        
        # Suggest closing brackets  
        if text.count('[') > text.count(']'):
            return Suggestion(']')
        
        # Suggest common patterns from history
        for cmd in reversed(self.workspace.history):
            if cmd.startswith(text) and cmd != text:
                return Suggestion(cmd[len(text):])
        
        return None


class KeyaDCREPL:
    """Modern, language-first REPL for keya D-C."""
    
    def __init__(self, engine: Engine):
        self.engine = engine
        self.workspace = WorkspaceState()
        self._setup_session()
        self._buffer = ""  # For multi-line input
        self._in_multiline = False
    
    def _setup_session(self):
        """Initialize the prompt session with all modern features."""
        completer = KeyaDCCompleter(self.engine, self.workspace)
        auto_suggest = KeyaDCAutoSuggest(self.engine, self.workspace)
        
        self.session = PromptSession(
            completer=completer,
            auto_suggest=auto_suggest,
            key_bindings=self._create_keybindings(),
            style=self._get_style(),
            multiline=False,
            wrap_lines=True,
        )
    
    def _create_keybindings(self) -> KeyBindings:
        """Create enhanced key bindings."""
        kb = KeyBindings()
        
        @kb.add('c-c')
        def _(event):
            """Cancel multi-line input."""
            if self._in_multiline:
                self._buffer = ""
                self._in_multiline = False
                event.app.output.write("\nCancelled.\n")
            else:
                event.app.exit(exception=KeyboardInterrupt)
        
        @kb.add('c-d')
        def _(event):
            """Exit on Ctrl+D."""
            event.app.exit()
        
# Removed custom enter handling - using default behavior
        
        # Quick glyph insertions
        glyph_shortcuts = {
            'c-v': '∅',  # Ctrl+V for void
            'c-u': '△',  # Ctrl+U for up  
            'c-d': '▽',  # Ctrl+D for down
            'c-o': '⊙',  # Ctrl+O for unity
            'c-f': '⊕',  # Ctrl+F for flow
        }
        
        for key, glyph in glyph_shortcuts.items():
            @kb.add(key)
            def _(event, g=glyph):
                event.app.current_buffer.insert_text(g)
        
        return kb
    
    def _needs_continuation(self, text: str) -> bool:
        """Determine if input needs continuation."""
        if not text:
            return False
            
        # Check for unclosed braces/brackets
        open_braces = text.count('{') - text.count('}')
        open_brackets = text.count('[') - text.count(']')
        
        # Check for incomplete program structures
        if any(keyword in text for keyword in ['matrix', 'grammar', 'resonance']):
            if open_braces > 0:
                return True
        
        return open_braces > 0 or open_brackets > 0
    
    def _get_style(self) -> Style:
        """Define the visual style."""
        return Style.from_dict({
            # Prompt styling
            'prompt.kappa': '#FF6B6B bold',
            'prompt.bracket': '#5D8AA8',
            'prompt.workspace': '#98D8C8',
            
            # Syntax highlighting
            'keyword': '#FF6B6B bold',
            'operator': '#F7DC6F bold', 
            'glyph': '#BB8FCE bold',
            'number': '#85C1E9',
            'string': '#82E0AA',
            'comment': '#6C757D italic',
            
            # UI elements
            'autosuggestion': '#6A6A6A',
            'completion-menu.completion': 'bg:#333333 #ffffff',
            'completion-menu.completion.current': 'bg:#5D8AA8 #ffffff bold',
            'completion-menu.meta.completion': 'bg:#333333 #bbbbbb',
            'completion-menu.meta.completion.current': 'bg:#5D8AA8 #ffffff',
        })
    
    def _get_prompt(self) -> StyleAndTextTuples:
        """Generate the dynamic prompt."""
        if self._in_multiline:
            return [('class:prompt.bracket', '...〉 ')]
        
        # Show current workspace and active variables
        context = ""
        if self.workspace.name != "default":
            context = f"|{self.workspace.name}"
        
        active_vars = list(self.workspace.variables.keys())[:3]
        if active_vars:
            vars_str = ",".join(active_vars)
            if len(self.workspace.variables) > 3:
                vars_str += ",+"
            context += f":[{vars_str}]"
        
        if context:
            return [
                ('class:prompt.bracket', 'keya'),
                ('class:prompt.workspace', context),
                ('class:prompt.bracket', '〉 ')
            ]
        else:
            return [
                ('class:prompt.bracket', '|'),
                ('class:prompt.kappa', 'κ'),
                ('class:prompt.bracket', '〉 ')
            ]
    
    def run(self):
        """Main REPL loop."""
        self._show_welcome()
        
        while True:
            try:
                if self._in_multiline:
                    line = self.session.prompt(self._get_prompt())
                    self._buffer += "\n" + line
                    
                    if not self._needs_continuation(self._buffer):
                        self._process_input(self._buffer.strip())
                        self._buffer = ""
                        self._in_multiline = False
                else:
                    line = self.session.prompt(self._get_prompt())
                    if line.strip():
                        if self._needs_continuation(line):
                            self._buffer = line
                            self._in_multiline = True
                        else:
                            self._process_input(line.strip())
                
            except (KeyboardInterrupt, EOFError):
                if self._in_multiline:
                    self._buffer = ""
                    self._in_multiline = False
                    print("Cancelled.")
                    continue
                else:
                    break
            except Exception as e:
                print(f"Unexpected error: {e}")
                self._buffer = ""
                self._in_multiline = False
        
        self._show_goodbye()
    
    def _show_welcome(self):
        """Display welcome message."""
        print("Keya D-C Shell - Modern Language-First REPL")
        print()
        print("Features:")
        print("  * Full D-C syntax: DC([3,3,△], binary, 5)")
        print("  * Smart glyph completion: void → ∅, up → △") 
        print("  * Matrix visualization: :show matrix_name")
        print("  * Workspace management: :workspace quantum_demo")
        print()
        print("Quick help: :help syntax | :help operators | :help examples")
        print("Exit: Ctrl+D or 'exit'")
        print()
    
    def _show_goodbye(self):
        """Display goodbye message."""
        print("Collapsing wavefunction... Farewell!")
    
    def _process_input(self, line: str):
        """Process a complete input line or program."""
        # Add to history
        self.workspace.history.append(line)
        
        # Handle meta-commands
        if line.startswith(':'):
            self._handle_meta_command(line[1:])
            return
        
        # Handle simple commands
        if line.lower() in ['exit', 'quit']:
            raise EOFError()
        
        # Try to parse and execute as D-C language
        try:
            # Check if it's a simple expression or full program
            if any(keyword in line for keyword in ['matrix', 'grammar', 'resonance']):
                # Full program
                result = self.engine.execute_program(line)
                if result:
                    self._display_result(result)
            else:
                # Simple expression - wrap in a minimal program
                wrapped = f"matrix temp {{ ops {{ result = {line} }} }}"
                try:
                    result = self.engine.execute_program(wrapped)
                    if result and 'result' in result:
                        self._display_result(result['result'])
                except:
                    # Fallback: try as variable lookup
                    if line in self.workspace.variables:
                        self._display_result(self.workspace.variables[line])
                    else:
                        raise ParseError(f"Unknown variable or invalid expression: {line}")
        
        except ParseError as e:
            self._display_error("Parse Error", str(e), line)
        except Exception as e:
            self._display_error("Execution Error", str(e), line)
    
    def _handle_meta_command(self, cmd: str):
        """Handle meta-commands like :help, :show, etc."""
        parts = cmd.split()
        command = parts[0]
        args = parts[1:]
        
        if command == 'help':
            self._show_help(args[0] if args else None)
        elif command == 'show':
            self._show_variable(args[0] if args else None)
        elif command == 'vars':
            self._show_variables()
        elif command == 'workspace':
            self._manage_workspace(args[0] if args else None)
        elif command == 'save':
            self._save_session(args[0] if args else None)
        elif command == 'load':
            self._load_session(args[0] if args else None)
        elif command == 'clear':
            self._clear_session()
        elif command == 'history':
            self._show_history()
        else:
            print(f"Unknown command: :{command}")
            print("Available commands: help, show, vars, workspace, save, load, clear, history")
    
    def _show_help(self, topic: Optional[str]):
        """Show contextual help."""
        if topic == 'syntax':
            print_formatted_text(HTML("""
<style fg="#FF6B6B" bold>Keya D-C Syntax Reference</style>

<style fg="#98D8C8">Program Types:</style>
  <style fg="#F7DC6F">matrix</style> program_name { ops { ... } }
  <style fg="#F7DC6F">grammar</style> program_name { rules { ... } }
  <style fg="#F7DC6F">resonance</style> program_name { traces { ... } }

<style fg="#98D8C8">Operators:</style>
  <style fg="#F7DC6F">D</style>(matrix)                    - Dissonance (symmetry breaking)
  <style fg="#F7DC6F">C</style>(matrix, type)             - Containment (binary|decimal|string|general)
  <style fg="#F7DC6F">DC</style>(matrix, type, iterations) - Full D-C cycle

<style fg="#98D8C8">Glyphs:</style>
  <style fg="#BB8FCE">∅</style> void    <style fg="#BB8FCE">▽</style> down    <style fg="#BB8FCE">△</style> up    <style fg="#BB8FCE">⊙</style> unity    <style fg="#BB8FCE">⊕</style> flow
            """))
        elif topic == 'operators':
            print_formatted_text(HTML("""
<style fg="#FF6B6B" bold>D-C Operators</style>

<style fg="#F7DC6F" bold>D(matrix)</style> - Dissonance Operator
  Breaks symmetry and creates interference patterns
  Example: <style fg="#85C1E9">D([3,3,△])</style>

<style fg="#F7DC6F" bold>C(matrix, type)</style> - Containment Operator  
  Resolves patterns into stable forms
  Types: binary, decimal, string, general
  Example: <style fg="#85C1E9">C(evolved_matrix, binary)</style>

<style fg="#F7DC6F" bold>DC(matrix, type, iterations)</style> - D-C Cycle
  Iterative dissonance→containment evolution
  Example: <style fg="#85C1E9">DC([4,4,∅], binary, 10)</style>
  Infinite: <style fg="#85C1E9">DC(wave, general, ∞)</style>
            """))
        elif topic == 'examples':
            print_formatted_text(HTML("""
<style fg="#FF6B6B" bold>Examples</style>

<style fg="#98D8C8">Simple matrix operations:</style>
  m = [3, 3, ∅]
  result = D(m)
  evolved = DC(m, binary, 5)

<style fg="#98D8C8">Full program:</style>
  matrix evolution {
    ops {
      grid = [10, 10, ∅]
      grid[5,5] = △
      final = DC(grid, binary, 20)
    }
  }

<style fg="#98D8C8">Grammar program:</style>
  grammar cellular {
    rules {
      ∅ → [△, ▽]
      △ → [⊙]
    }
  }
            """))
        else:
            print_formatted_text(HTML("""
<style fg="#FF6B6B" bold>Keya D-C Help</style>

<style fg="#98D8C8">Help topics:</style>
  :help syntax     - Language syntax reference
  :help operators  - D-C operator details  
  :help examples   - Code examples

<style fg="#98D8C8">Commands:</style>
  :show VAR        - Display variable/matrix
  :vars            - List all variables
  :workspace NAME  - Switch workspace
  :save [FILE]     - Save session
  :load [FILE]     - Load session
  :clear           - Clear variables
  :history         - Show command history

<style fg="#98D8C8">Quick keys:</style>
  Ctrl+V → ∅  Ctrl+U → △  Ctrl+D → ▽  Ctrl+O → ⊙  Ctrl+F → ⊕
            """))
    
    def _show_variable(self, name: Optional[str]):
        """Display variable with enhanced visualization."""
        if not name:
            print("Usage: :show variable_name")
            return
        
        if name not in self.workspace.variables:
            print(f"Variable '{name}' not found.")
            return
        
        value = self.workspace.variables[name]
        print(f"\n{name}:")
        
        # Enhanced display based on type
        if hasattr(value, 'shape') and len(value.shape) == 2:
            # Matrix visualization
            self._display_matrix(value, name)
        else:
            print(f"  Type: {type(value).__name__}")
            print(f"  Value: {value}")
    
    def _display_matrix(self, matrix, name: str):
        """ASCII art matrix display."""
        print(f"  Matrix {matrix.shape[0]}×{matrix.shape[1]}:")
        
        # Convert numeric values back to glyphs for display
        glyph_map = {0.0: '∅', 1.0: '▽', 2.0: '△', 3.0: '⊙', 4.0: '⊕'}
        
        for i in range(matrix.shape[0]):
            row = "    "
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                glyph = glyph_map.get(float(val), '?')
                row += f"{glyph} "
            print(row)
    
    def _show_variables(self):
        """List all session variables."""
        if not self.workspace.variables:
            print("No variables in current session.")
            return
        
        print(f"\nVariables in workspace '{self.workspace.name}':")
        for name, value in self.workspace.variables.items():
            if hasattr(value, 'shape'):
                print(f"  {name}: {value.shape} matrix")
            else:
                print(f"  {name}: {type(value).__name__} = {value}")
    
    def _manage_workspace(self, name: Optional[str]):
        """Manage workspace/session."""
        if not name:
            print(f"Current workspace: {self.workspace.name}")
            return
        
        if name != self.workspace.name:
            # Save current workspace
            self._save_current_workspace()
            
            # Switch to new workspace
            self.workspace.name = name
            self.workspace.variables.clear()
            
            # Try to load workspace
            workspace_file = Path(f".keya_workspace_{name}.json")
            if workspace_file.exists():
                self.workspace.load_from_file(workspace_file)
                print(f"Switched to workspace '{name}' (loaded from file)")
            else:
                print(f"Switched to new workspace '{name}'")
    
    def _save_current_workspace(self):
        """Save current workspace state."""
        if self.workspace.name and self.workspace.variables:
            workspace_file = Path(f".keya_workspace_{self.workspace.name}.json")
            self.workspace.save_to_file(workspace_file)
    
    def _save_session(self, filename: Optional[str]):
        """Save session to file."""
        if not filename:
            filename = f"{self.workspace.name}_session.keya"
        
        path = Path(filename)
        self.workspace.save_to_file(path)
        print(f"Session saved to {path}")
    
    def _load_session(self, filename: Optional[str]):
        """Load session from file."""
        if not filename:
            print("Usage: :load filename")
            return
        
        path = Path(filename)
        if not path.exists():
            print(f"File not found: {path}")
            return
        
        self.workspace.load_from_file(path)
        print(f"Session loaded from {path}")
    
    def _clear_session(self):
        """Clear all session variables."""
        self.workspace.variables.clear()
        print("Session variables cleared.")
    
    def _show_history(self):
        """Show command history."""
        if not self.workspace.history:
            print("No command history.")
            return
        
        print("\nRecent commands:")
        for i, cmd in enumerate(self.workspace.history[-10:], 1):
            print(f"  {i:2d}. {cmd}")
    
    def _display_result(self, result):
        """Display execution result with enhanced formatting."""
        if result is None:
            return
        
        if hasattr(result, 'shape') and len(result.shape) == 2:
            print("\nResult matrix:")
            self._display_matrix(result, "result")
        elif isinstance(result, dict):
            print("\nResults:")
            for key, value in result.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: {value.shape} matrix")
                    # Store in workspace
                    self.workspace.variables[key] = value
                else:
                    print(f"  {key}: {value}")
                    self.workspace.variables[key] = value
        else:
            print(f"\nResult: {result}")
            # Store simple results as 'result' variable
            self.workspace.variables['result'] = result
    
    def _display_error(self, error_type: str, message: str, line: str):
        """Display enhanced error messages."""
        print_formatted_text(HTML(f"""
<style fg="#FF6B6B" bold>{error_type}</style>: {message}

<style fg="#6C757D">Input:</style> {line}

<style fg="#98D8C8">Suggestions:</style>
  • Check syntax with <style fg="#F7DC6F">:help syntax</style>
  • View examples with <style fg="#F7DC6F">:help examples</style>  
  • List variables with <style fg="#F7DC6F">:vars</style>
        """))
    
    def run_script(self, filepath: str):
        """Execute a keya D-C script file."""
        path = Path(filepath)
        if not path.exists():
            print(f"Script file not found: {filepath}")
            return
        
        try:
            content = path.read_text()
            print(f"Executing {path.name}...")
            self._process_input(content)
            print(f"Script {path.name} completed.")
        except Exception as e:
            print(f"Error executing script {path.name}: {e}") 