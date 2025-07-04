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
from keya.shell.repl import KeyaDCREPL, SYMBOL_REPLACEMENTS


def print_version():
    """Print version information."""
    print("Kshell - Kéya D-C Language Shell")
    print("Version: 2.0.0 - Modern Language-First REPL")
    print("Features: D-C operators, glyph matrices, grammar transformations")


def test_completion(word: str):
    """Test tab completion for a given word."""
    if not word:
        print("Error: No word provided for completion testing")
        return 1
    
    # Find all symbol matches
    symbol_matches = []
    
    # Exact matches
    if word.lower() in SYMBOL_REPLACEMENTS:
        symbol_matches.append((word.lower(), SYMBOL_REPLACEMENTS[word.lower()]))
    
    # Prefix matches (only if no exact match)
    if not symbol_matches:
        for symbol_word in SYMBOL_REPLACEMENTS:
            if symbol_word.startswith(word.lower()) and symbol_word != word.lower():
                symbol_matches.append((symbol_word, SYMBOL_REPLACEMENTS[symbol_word]))
    
    # Output results
    if len(symbol_matches) == 0:
        print(f"No symbol completions for '{word}'")
        return 0
    elif len(symbol_matches) == 1:
        # Single match - output the symbol (like tab would do)
        symbol_word, symbol = symbol_matches[0]
        print(symbol)
        return 0
    else:
        # Multiple matches - output column-style list
        print(f"Multiple completions for '{word}':")
        for symbol_word, symbol in sorted(symbol_matches):
            print(f"  {symbol_word:<12} → {symbol}")
        return 0


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
    print("Kshell - Kéya D-C Language Shell")
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
        elif arg in ['--tab', '-t']:
            if len(sys.argv) < 3:
                print("Error: --tab/-t requires a word argument")
                print("Usage: python kshell.py --tab <word> OR python kshell.py -t <word>")
                return 1
            return test_completion(sys.argv[2])
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
                repl = KeyaDCREPL(engine)
                repl.run_script(str(script_path))
            except Exception as e:
                print(f"Error executing script: {e}")
                return 1
    else:
        # Interactive mode: start the REPL
        try:
            engine = Engine()
            repl = KeyaDCREPL(engine)
            repl.run()
        except KeyboardInterrupt:
            print("\nKshell interrupted by user.")
        except Exception as e:
            print(f"Error starting Kshell: {e}")
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
