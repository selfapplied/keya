#!/usr/bin/env python3
"""
🌟 KEYA CELLULAR AUTOMATA WIDGETS DEMO 🌟

This demonstrates the revolutionary cellular automata widget system 
powered by the keya mathematical language!

Features:
- Real-time cellular evolution using keya operators
- Interactive widgets with multiple interaction modes
- Infinite iteration support (∞) for continuous evolution  
- Mathematical harmonic transformations
- Beautiful visual rendering with matplotlib
"""

import sys
import os
import argparse

from keya.widgets.renderer import create_demo_widget, WidgetRenderer
from keya.widgets.cellular import CellularWidget, InteractionMode
from keya.dsl.ast import ContainmentType


def demo_ripple_widget():
    """Demo the ripple interaction widget."""
    print("🌊 RIPPLE WIDGET DEMO")
    print("Click anywhere to create energy ripples that evolve via operators!")
    print("Press SPACE to start/stop evolution, R to reset")
    
    widget, renderer = create_demo_widget("ripple")
    renderer.show(auto_evolve=False)


def demo_infinite_evolution():
    """Demo infinite evolution with ∞ iterations."""
    print("♾️  INFINITE EVOLUTION DEMO")
    print("This widget uses DC(grid, binary, ∞) for continuous cellular evolution!")
    
    # Create widget with infinite evolution
    widget = CellularWidget(
        width=25,
        height=25,
        containment_type=ContainmentType.BINARY,
        interaction_mode=InteractionMode.RIPPLE,
        evolution_speed=0.1  # Fast evolution
    )
    
    # Manually test infinite cycle
    program = """
matrix infinite_test {
    cellular_evolution {
        grid = [5, 5, △]
        infinite_result = DC(grid, binary, ∞)  
    }
}
"""
    
    print("Testing infinite cycles...")
    result = widget.engine.execute_program(program.strip())
    if result:
        print("✅ Infinite cycles working!")
    
    renderer = WidgetRenderer(widget)
    renderer.show(auto_evolve=True)


def demo_multi_containment():
    """Demo different containment types."""
    print("🎛️  MULTI-CONTAINMENT DEMO")
    print("Press 1-4 to switch interaction modes while evolution runs!")
    
    widget = CellularWidget(
        width=20,
        height=20,
        containment_type=ContainmentType.DECIMAL,  # Different containment
        interaction_mode=InteractionMode.DRAW,
        evolution_speed=0.2
    )
    
    renderer = WidgetRenderer(widget)
    renderer.show(auto_evolve=True)


def demo_console_widget():
    """Demo a console-based widget for systems without GUI."""
    print("💻 CONSOLE WIDGET DEMO")
    print("Watch evolution in the terminal!")
    
    widget = CellularWidget(width=15, height=10)
    
    print("Initial state:")
    display_console_grid(widget)
    
    # Add some manual interactions
    widget.handle_interaction(7, 5)  # Center ripple
    widget.handle_interaction(3, 2)  # Another ripple
    
    print("\nAfter interactions:")
    display_console_grid(widget)
    
    # Start evolution and show a few steps
    widget.start_evolution()
    for step in range(5):
        widget.step_evolution()
        print(f"\nEvolution step {step + 1}:")
        display_console_grid(widget)
        
        stats = widget.get_stats()
        print(f"Gen: {stats['generation']} | Active cells: {stats['total_cells'] - stats['void_cells']}")


def demo_all_features():
    """Run all console-based demos in sequence."""
    print("🎉 Running all console demos in sequence!")
    demo_console_widget()
    print("\n" + "="*50)
    print("✅ All demos completed successfully!")


def display_console_grid(widget: CellularWidget):
    """Display widget grid in console."""
    grid = widget.get_display_grid()
    for row in grid:
        print(' '.join(row))


def main():
    """Main demo selector - non-interactive by default."""
    parser = argparse.ArgumentParser(description='Keya Cellular Automata Widgets Demo')
    parser.add_argument('-i', '--interactive', action='store_true',
                       help='Enable interactive mode with menu selection')
    parser.add_argument('--choice', type=int, choices=range(1, 6), 
                       help='Specific demo choice (1-5)')
    
    args = parser.parse_args()
    
    # Non-interactive mode (default) - run console demo
    if not args.interactive:
        print("🤖 NON-INTERACTIVE MODE (use -i for interactive)")
        print("Running console demo automatically...")
        demo_console_widget()
        return
    
    # Interactive mode - show menu
    print("🚀 KEYA CELLULAR AUTOMATA WIDGETS 🚀")
    print("=" * 50)
    print("Select a demo:")
    print("1. Ripple Widget (Interactive)")
    print("2. Infinite Evolution (∞ iterations)")  
    print("3. Multi-Containment Types")
    print("4. Console Demo (No GUI)")
    print("5. All Features Test")
    
    try:
        if args.choice:
            choice = str(args.choice)
        else:
            choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            demo_ripple_widget()
        elif choice == '2':
            demo_infinite_evolution()
        elif choice == '3':
            demo_multi_containment()
        elif choice == '4':
            demo_console_widget()
        elif choice == '5':
            demo_all_features()
        else:
            print("Invalid choice. Showing console demo by default...")
            demo_console_widget()
            
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print("Falling back to console demo...")
        demo_console_widget()


if __name__ == "__main__":
    main() 