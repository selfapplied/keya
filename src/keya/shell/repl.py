from pathlib import Path
from typing import Optional, Sequence

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.styles import Style

from ..core.engine import Engine
from ..dsl.parser import ParseError


class KeyaAutoSuggest(AutoSuggest):
    """Auto-suggestion provider for keya D-C language."""

    def __init__(self, engine: Engine):
        self.engine = engine
        # Legacy symbol table
        self.legacy_words = sorted(self.engine.symbol_table.keys())
        
        # Keya D-C language keywords and constructs
        self.keya_keywords = [
            'matrix', 'grammar', 'resonance',
            'verify', 'arithmetic', 'strings', 'trace',
            'binary', 'decimal', 'string', 'general',
            'D', 'C', 'DC'
        ]
        
        # Glyph symbols
        self.glyph_symbols = ['∅', '▽', '△', '⊙', '⊕']
        
        # Variable names from current session
        self.session_variables = []
        
        self.all_suggestions = self.legacy_words + self.keya_keywords + self.glyph_symbols

    def get_suggestion(self, buffer: Buffer, document: Document) -> Optional[Suggestion]:
        """Generates suggestions for keya D-C language."""
        word = document.get_word_before_cursor(WORD=True)
        if not word:
            return None

        # Update session variables from engine
        if hasattr(self.engine, 'variables'):
            self.session_variables = list(self.engine.variables.keys())
            
        all_words = self.all_suggestions + self.session_variables
        
        for suggestion in sorted(set(all_words)):
            if suggestion.startswith(word) and suggestion != word:
                return Suggestion(suggestion[len(word):])

        return None


class KeyaREPL:
    """The interactive shell environment for Kéya."""

    def __init__(self, engine: Engine):
        self.engine = engine
        auto_suggester = KeyaAutoSuggest(self.engine)
        bindings = self._create_key_bindings()

        self.prompt_session = PromptSession(
            auto_suggest=auto_suggester,
            key_bindings=bindings,
            style=self._get_styles(),
        )

    def _create_key_bindings(self) -> KeyBindings:
        """Binds space and tab to handle symbol completion and keya D-C syntax."""
        bindings = KeyBindings()

        @bindings.add(" ")
        def _(event: KeyPressEvent):
            buffer = event.app.current_buffer
            word = buffer.document.get_word_before_cursor(WORD=True)

            # Handle legacy symbols
            if word and word in self.engine.symbol_table:
                buffer.delete_before_cursor(count=len(word))
                symbol, _ = self.engine.symbol_table[word]
                buffer.insert_text(symbol + " ")
            else:
                buffer.insert_text(" ")

        @bindings.add("tab")
        def _(event: KeyPressEvent):
            buffer = event.app.current_buffer
            if suggestion := buffer.suggestion:
                word_before_cursor = buffer.document.get_word_before_cursor(WORD=True)
                full_word = word_before_cursor + suggestion.text
                
                # Handle legacy symbols if applicable
                if full_word in self.engine.symbol_table:
                    symbol, _ = self.engine.symbol_table[full_word]
                    buffer.delete_before_cursor(count=len(word_before_cursor))
                    buffer.insert_text(symbol)
                else:
                    # Accept the suggestion as-is for keya D-C syntax
                    buffer.insert_text(suggestion.text)

        return bindings

    def _get_styles(self) -> Style:
        return Style.from_dict(
            {
                "prompt-ket": "#5D8AA8",
                "prompt-kappa": "#FF6B6B",
                "autosuggestion": "#6A6A6A",
            }
        )

    def _get_prompt(self) -> StyleAndTextTuples:
        return [
            ("class:prompt-ket", "|"),
            ("class:prompt-kappa", "κ"),
            ("class:prompt-ket", "〉"),
            ("", " "),
        ]

    def run_interactive(self):
        """The main execution loop for the interactive REPL."""
        print("Welcome to Kshell ⸻ The Kéya D-C Shell 🐚")
        print("Features: D-C operators, glyph matrices, grammar transformations")
        print("Type 'exit' to leave, 'help' for keya D-C syntax guide.")
        print()

        while True:
            try:
                line = self.prompt_session.prompt(self._get_prompt())
                if line.lower() == "exit":
                    break
                elif line.lower() == "help":
                    self._show_help()
                elif line.lower() == "vars":
                    self._show_variables()
                elif line.lower() == "clear":
                    self._clear_session()
                else:
                    try:
                        result = self.engine.process_line(line)
                        if result is not None:
                            print(f"Result: {result}")
                    except ParseError as e:
                        print(f"Parse error: {e}")
                    except Exception as e:
                        print(f"Execution error: {e}")
            except (KeyboardInterrupt, EOFError):
                break

        print("Collapsing wavefunction... Farewell!")
    
    def _show_help(self):
        """Display keya D-C syntax help."""
        print("""
Kéya D-C Language Quick Reference:

Program Types:
  matrix program_name { ... }      - Matrix operations and D-C transformations
  grammar program_name { ... }     - Grammar rules and string generation  
  resonance program_name { ... }   - Resonance analysis and verification

Operators:
  D matrix                         - Dissonance operator (symmetry breaking)
  C(matrix, type)                  - Containment operator (binary|decimal|string|general)
  DC(matrix, type, iterations)     - D-C cycle transformation

Glyphs:
  ∅ (void)  ▽ (down)  △ (up)  ⊙ (unity)  ⊕ (flow)

Matrices:
  [rows, cols, fill_glyph]         - Matrix with dimensions and fill
  [[▽, △], [⊙, ⊕]]                - Explicit matrix values

Commands:
  help    - Show this help
  vars    - Show current variables  
  clear   - Clear session variables
  exit    - Exit shell
        """)
    
    def _show_variables(self):
        """Display current session variables."""
        if hasattr(self.engine, 'variables') and self.engine.variables:
            print("Current variables:")
            for name, value in self.engine.variables.items():
                if hasattr(value, 'shape'):
                    print(f"  {name}: {value.shape} matrix")
                else:
                    print(f"  {name}: {type(value).__name__} = {value}")
        else:
            print("No variables defined in current session.")
    
    def _clear_session(self):
        """Clear all session variables."""
        if hasattr(self.engine, 'variables'):
            self.engine.variables.clear()
        if hasattr(self.engine, 'grammars'):
            self.engine.grammars.clear()
        print("Session variables cleared.")

    def run_script(self, filename: str):
        """Executes commands from a file using the engine."""
        filepath = Path(filename)
        if not filepath.is_file():
            print(f"Error: Script file not found at `{filename}`.")
            return

        with filepath.open("r") as f:
            for line in f:
                self.engine.process_line(line)
