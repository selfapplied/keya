#!/usr/bin/env python3
"""
Kshell - The Keya Language Shell

Interactive shell for Keya language:
- Matrix operations with glyph symbols
- Grammar transformations  
- Resonance analysis
- Visualization
"""

import sys
import os
from pathlib import Path

# Ensure the src directory is on the path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# The REPL is temporarily disabled as it's tied to the old DSL.
# from keya.shell.repl import KeyaDCREPL
from keya.kshell.parser import parse_kshell_file
from keya.kshell.engine import KShellEngine


def print_version():
    """Print version information."""
    print("Kshell - Keya Language Shell")
    print("Version: 2.0.0 - Modern Language-First REPL")
    print("Features: Glyph matrices, grammar transformations")


# Extensible command definitions
COMMANDS = {
    'interactive': {
        'usage': 'python kshell.py',
        'description': 'Start interactive shell',
        'examples': []
    },
    'script': {
        'usage': 'python kshell.py <script.keya>',
        'description': 'Run script file',
        'examples': [
            'python kshell.py demos/evolution.keya',
            'python kshell.py demos/symbol-translation.keya'
        ]
    },
    'version': {
        'usage': 'python kshell.py --version, -v',
        'description': 'Show version information',
        'examples': []
    },
    'help': {
        'usage': 'python kshell.py --help, -h',
        'description': 'Show this help',
        'examples': []
    },
    'tab': {
        'usage': 'python kshell.py --tab <word>, -t <word>',
        'description': 'Test tab completion behavior',
        'examples': [
            'python kshell.py -t ten          # Should output: ⊗',
            'python kshell.py -t v            # Should list: void, etc.'
        ]
    }
}

def print_usage():
    """Print usage information from structured command definitions."""
    print("Kshell - Keya Language Shell")
    print()
    print("Usage:")
    for cmd_info in COMMANDS.values():
        print(f"  {cmd_info['usage']:<35} # {cmd_info['description']}")
    
    # Collect all examples
    all_examples = []
    for cmd_info in COMMANDS.values():
        all_examples.extend(cmd_info['examples'])
    
    if all_examples:
        print()
        print("Examples:")
        for example in all_examples:
            print(f"  {example}")
    
    print()
    print("Symbol Completions:")
    print("  Core glyphs: void→∅, up→△, down→▽, unity→⊙, flow→⊕")
    print("  Operators: tensor→⊗, growth→↑, descent→ℓ, reflect→~, dissonance→𝔻, containment→ℂ")


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
        # The --tab command was tied to the old REPL's symbol completer.
        # It is disabled for now.
        # elif arg in ['--tab', '-t']:
        #     if len(sys.argv) < 3:
        #         print("Error: --tab/-t requires a word argument")
        #         print("Usage: python kshell.py --tab <word> OR python kshell.py -t <word>")
        #         return 1
        #     return test_completion(sys.argv[2])
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
                print(f"--- Running Script: {script_path.name} ---")
                pipeline = parse_kshell_file(script_path)
                engine = KShellEngine()
                final_state = engine.run(pipeline)
                print(f"\n--- Script Complete ---")
                print(f"Final State: {final_state}")

            except Exception as e:
                print(f"Error executing script: {e}")
                return 1
    else:
        # Interactive mode is disabled pending a redesign for the new engine.
        print_version()
        print("\nInteractive REPL is temporarily disabled.")
        print("Please run a script file instead, e.g.: python kshell.py demos/kshell/evolution.keya")
        # try:
        #     engine = Engine()
        #     repl = KeyaDCREPL(engine)
        #     repl.run()
        # except KeyboardInterrupt:
        #     print("\nKshell interrupted by user.")
        # except Exception as e:
        #     print(f"Error starting Kshell: {e}")
        #     return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
