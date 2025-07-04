#!/usr/bin/env python3
"""Demo of different display modes in Kéya REPL."""

import sys
import os

import numpy as np
from keya.core.engine import Engine
from keya.shell.repl import KeyaDCREPL

def demo_display_modes():
    """Demo different display modes."""
    print("🎮 Kéya Display Modes Demo")
    print("=" * 40)
    
    # Create engine and REPL
    engine = Engine()
    repl = KeyaDCREPL(engine)
    
    # Create a test matrix with different glyph values
    test_matrix = np.array([
        [0.0, -1.0, 1.0],      # ∅, ▽, △
        [0.5, 2.0, -0.8417],   # ⊙, ⊕, dissonance value
        [1.0, 0.0, 0.5]        # △, ∅, ⊙
    ])
    
    # Add matrix to workspace
    repl.workspace.variables['test'] = test_matrix
    
    print("\n1. DEFAULT GLYPH MODE:")
    repl._show_variable('test')
    
    print("\n2. NUMBERS MODE:")
    repl._set_display_mode('numbers')
    repl._show_variable('test')
    
    print("\n3. MIXED MODE (glyph + number):")
    repl._set_display_mode('mixed')
    repl._show_variable('test')
    
    print("\n4. BACK TO GLYPHS:")
    repl._set_display_mode('glyphs')
    repl._show_variable('test')
    
    print("\n5. TESTING :plot COMMAND:")
    try:
        repl._plot_variable('test', 'matrix')
        print("✅ Matrix plot should display")
    except ImportError:
        print("⚠️  Matplotlib not available for plotting")
    except Exception as e:
        print(f"❌ Plot error: {e}")
    
    print("\n6. TESTING :display COMMAND:")
    repl._set_display_mode(None)  # Show current mode
    
    print("\n✅ Display mode demo complete!")
    print("\nAvailable REPL commands:")
    print("  :display glyphs   # Show symbols ∅ ▽ △ ⊙ ⊕")
    print("  :display numbers  # Show raw numbers") 
    print("  :display mixed    # Show both: △(1.0)")
    print("  :plot matrix_name # Matplotlib visualization")

if __name__ == "__main__":
    demo_display_modes() 