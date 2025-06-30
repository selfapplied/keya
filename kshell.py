import sys

from src.keya.core.engine import Engine
from src.keya.shell.repl import KeyaREPL


def main():
    """The main entrypoint for the Kshell."""
    engine = Engine()
    repl = KeyaREPL(engine)

    if len(sys.argv) > 1:
        # Script mode: execute the file provided as an argument.
        script_path = sys.argv[1]
        repl.run_script(script_path)
    else:
        # Interactive mode: start the REPL.
        repl.run_interactive()


if __name__ == "__main__":
    main()
