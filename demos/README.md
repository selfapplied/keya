# Keya Demos

This directory contains demonstration scripts showcasing different aspects of the keya D-C programming language.

## Demo Scripts

### Pure Keya D-C Language Demos

**`demo-test-evolution.keya`** - *Basic Matrix Evolution*
- Simple demonstration of D-C operators on glyph matrices
- Shows basic syntax: matrix creation, D operator, and DC cycles
- Perfect introduction to keya D-C language fundamentals
- Run with: `python kshell.py demos/demo-test-evolution.keya`

**`demo-symbol-translation.keya`** - *Symbol-to-Number Translation*
- Demonstrates how keya symbols (∅ ▽ △ ⊙ ⊕) map to numbers (0-4)
- Shows mixed glyph matrices and their mathematical evolution  
- Illustrates binary containment convergence patterns
- Run with: `python kshell.py demos/demo-symbol-translation.keya`

### Core D-C Demonstrations

**`demo_floating_point_dc.py`** - *Scientific Testing*
- Tests Deepseek's hypothesis that floating-point arithmetic embodies D-C operations
- Rigorous numerical verification of IEEE 754 as containment operations
- Outputs: `.out/tests/floating_point_dc_tests.out`, `.out/visualizations/floating_point_dc_tests.png`
- Status: ✅ All 8 tests pass (100% success rate)

### Quantum Mechanics Integration

**`demo_quantum_phenomena.py`** - *Quantum Visualization*
- Demonstrates electron orbital rendering using keya D-C operators
- 3D visualization of hydrogen atom wave functions
- Real-time quantum evolution via D-C cycles
- Requires: matplotlib, numpy

**`demo_mantissa_quantum.py`** - *Floating-Point ↔ Quantum Connection*
- Shows relationship between mantissa normalization and wave function normalization
- Demonstrates that D-C operators are fundamentally quantum operators
- Explores the deep connection between computation and physics

### Interactive Systems

**`demo_cellular_widgets.py`** - *Cellular Automata*
- Interactive cellular automata using keya D-C language
- Real-time pattern evolution through D-C cycles
- Click-to-edit grid interface
- Demonstrates infinite iteration capabilities

### Development & Testing

**`debug_infinity.py`** - *Infinity Handling*
- Tests keya's ability to handle infinite D-C iterations
- Debugging utilities for cellular automata systems

**`test_infinity.py`** - *Infinity Verification*
- Verification tests for infinite iteration behavior
- Unit tests for cellular automata convergence

## Running Demos

### Keya D-C Language Demos (.keya files)

Use the modern keya shell (kshell) to run pure D-C language programs:

```bash
# Run from project root directory
python kshell.py demos/demo-test-evolution.keya
python kshell.py demos/demo-symbol-translation.keya

# Or start interactive REPL for exploration
python kshell.py
# Then try: grid = [3, 3, ∅]; DC(grid, binary, 5)
```

### Python Integration Demos (.py files)

All Python demos can be run from the demos directory:

```bash
# Navigate to demos directory
cd demos

# List available demos
python run_demo.py list

# Run specific demo (scientific testing of floating-point D-C hypothesis)
python run_demo.py demo_floating_point_dc.py

# Run all demos
python run_demo.py all

# Or run directly
python demo_quantum_phenomena.py
python demo_cellular_widgets.py
python demo_mantissa_quantum.py
```

## Output Organization

All demo outputs are automatically saved to organized directories:

```
.out/
├── tests/          # Test results and logs (.out files)
└── visualizations/ # Generated plots and images (.png files)
```

## Dependencies

- **Core**: numpy, matplotlib
- **Keya**: All demos require the keya package (`src/keya/`)
- **Optional**: JAX (for hardware acceleration)

## Notes

- All demos use non-interactive matplotlib backend (no windows)
- Outputs are saved to files rather than displayed
- Run from project root directory for proper import paths
- Some demos may take several seconds to complete due to mathematical computations 