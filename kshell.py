import sys

import os

# Ensure the src directory is on the path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from keya.core.engine import Engine
from keya.shell.repl import KeyaREPL


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
