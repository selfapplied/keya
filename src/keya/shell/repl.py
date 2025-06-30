from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.styles import Style

from ..core.engine import Engine


class KeyaAutoSuggest(AutoSuggest):
    """Auto-suggestion provider based on the engine's symbol table."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self.words = sorted(self.engine.symbol_table.keys())

    def get_suggestion(self, buffer: Buffer, document: Document) -> Suggestion | None:
        """Generates a suggestion based on the word before the cursor."""
        word = document.get_word_before_cursor(WORD=True)
        if not word:
            return None

        for suggestion in self.words:
            if suggestion.startswith(word) and suggestion != word:
                return Suggestion(suggestion[len(word) :])

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
        """Binds space and tab to handle symbol completion."""
        bindings = KeyBindings()

        @bindings.add(" ")
        def _(event: KeyPressEvent):
            buffer = event.app.current_buffer
            word = buffer.document.get_word_before_cursor(WORD=True)

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
                symbol, _ = self.engine.symbol_table[full_word]
                buffer.delete_before_cursor(count=len(word_before_cursor))
                buffer.insert_text(symbol)

        return bindings

    def _get_styles(self) -> Style:
        return Style.from_dict(
            {
                "prompt-ket": "#5D8AA8",
                "prompt-kappa": "#FF6B6B",
                "autosuggestion": "#6A6A6A",
            }
        )

    def _get_prompt(self):
        return [
            ("class:prompt-ket", "|"),
            ("class:prompt-kappa", "κ"),
            ("class:prompt-ket", "〉"),
            ("", " "),
        ]

    def run_interactive(self):
        """The main execution loop for the interactive REPL."""
        print("Welcome to Kshell ⸻ The Kéya Shell 🐚")
        print("Type 'exit' to leave.")

        while True:
            try:
                line = self.prompt_session.prompt(self._get_prompt())
                if line.lower() == "exit":
                    break
                self.engine.process_line(line)
            except (KeyboardInterrupt, EOFError):
                break

        print("Collapsing wavefunction... Farewell!")

    def run_script(self, filename: str):
        """Executes commands from a file using the engine."""
        filepath = Path(filename)
        if not filepath.is_file():
            print(f"Error: Script file not found at `{filename}`.")
            return

        with filepath.open("r") as f:
            for line in f:
                self.engine.process_line(line)
