#!/usr/bin/env python3
"""
Kshell - The Kéya D-C Language Shell

A command-line interface for the Kéya D-C language supporting:
- Matrix operations with D-C operators
- Glyph-based symbolic computation  
- Grammar transformations
- Resonance analysis
- Interactive visualization
"""

import sys
import os
from pathlib import Path

# Ensure the src directory is on the path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from keya.core.engine import Engine
from keya.shell.repl import KeyaREPL


def print_version():
    """Print version information."""
    print("Kshell - Kéya D-C Language Shell")
    print("Version: 1.0.0")
    print("Features: D-C operators, glyph matrices, grammar transformations")


def print_usage():
    """Print usage information."""
    print("Usage:")
    print("  python kshell.py                    # Start interactive shell")
    print("  python kshell.py <script.keya>     # Run script file")
    print("  python kshell.py --version         # Show version")
    print("  python kshell.py --help            # Show this help")
    print()
    print("Examples:")
    print("  python kshell.py examples/dc_test.py")
    print("  python kshell.py examples/string_generation_test.py")


def main():
    """The main entrypoint for the Kshell."""
    # Handle command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        if arg in ['--help', '-h']:
            print_usage()
            return
        elif arg in ['--version', '-v']:
            print_version()
            return
        elif arg.startswith('--'):
            print(f"Unknown option: {arg}")
            print_usage()
            return 1
        else:
            # Script mode: execute the file provided as an argument
            script_path = Path(arg)
            if not script_path.exists():
                print(f"Error: File '{script_path}' not found.")
                return 1
            
            try:
                engine = Engine()
                repl = KeyaREPL(engine)
                print(f"Executing script: {script_path}")
                repl.run_script(str(script_path))
            except Exception as e:
                print(f"Error executing script: {e}")
                return 1
    else:
        # Interactive mode: start the REPL
        try:
            engine = Engine()
            repl = KeyaREPL(engine)
            repl.run_interactive()
        except KeyboardInterrupt:
            print("\nKshell interrupted by user.")
        except Exception as e:
            print(f"Error starting Kshell: {e}")
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
