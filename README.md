# Kéya: A Universal Computational Engine

Kéya is an experimental computational framework built on a profound and elegant insight: that a vast range of computational and mathematical systems—from number theory to quantum mechanics to floating-point arithmetic—can be modeled as emergent behaviors of a single, universal, symbolic system.

At its heart is the **`PascalKernel`**, a pure, parameter-free mathematical object based on the binary logic of Pascal's triangle modulo 2 (the Sierpinski triangle). This kernel is not just a tool; it is a hypothesis about the nature of computation itself. It asserts that complex systems can be understood by representing their states as symbolic vectors and applying the kernel's fundamental, combinatorial transformation rules.

The philosophy of Kéya is not to change the engine to fit the problem, but to find a new **representation** of the problem that the engine can naturally solve.

## Core Concepts

*   **Universal Kernel**: The `PascalKernel` is the unchanging core of the system. Its logic, based on the `modulus=2` properties of Pascal's triangle, is treated as a fundamental computational substrate.
*   **Symbolic Representation**: All problems are approached by converting their states into symbolic vectors. A floating-point number becomes a vector of its binary digits; a cellular automaton becomes a grid of states.
*   **Transformation as Computation**: Computation is the process of applying the kernel's built-in transformation rules (`apply_polynomial` for convolutions, `reduce_with_carries` for normalization) to these state vectors.
*   **Emergent Behavior**: Complex, high-level behaviors (like rounding error cycles or the shapes of quantum orbitals) are shown to be emergent properties of the engine's simple, low-level binary rules.

## Getting Started: The Demos

The best way to understand Kéya is to explore the demos. They are not just examples; they are rigorous, assertion-backed proofs of the engine's capabilities.

We have built a decorator-based reporting system that makes running and understanding these proofs simple.

### Generating the Demo Report

To see a comprehensive overview of the engine's capabilities, generate the interactive HTML report:

```bash
python3 demos/report.py
```

This will run all registered demos and produce a detailed report at `docs/report.html`, complete with visualizations, claims, and findings for each experiment.

### Key Demonstrations

*   **`demos/floatingpoint.py`**: Proves that the subtle behaviors of floating-point arithmetic (quantization, rounding cycles) can be perfectly simulated by the engine's binary logic.
*   **`demos/cellular.py`**: Implements a high-performance 2D cellular automaton using JAX convolutions, demonstrating the engine's applicability to parallel systems.
*   **`demos/quantum.py` & `demos/orbital.py`**: Show how the shapes and evolution of quantum wavefunctions and atomic orbitals can be modeled.
*   **`demos/kshell.py`**: Introduces a declarative DSL for defining experimental pipelines, showcasing a higher-level way to interact with the engine.

## License

Kéya is released under the [GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0). 

### Why AGPL v3?

We chose AGPL v3 to foster **reinvestment and locality** in software development:

🔄 **Reinvestment**: Any improvements you make must be shared back with the community  
🏘️ **Locality**: Network services using Kéya must provide source code to their users  
🛡️ **Anti-Extraction**: Prevents "take and run" patterns that don't contribute back  
🔬 **Research Friendly**: Academics and researchers can use Kéya freely  
🤝 **Community Building**: Creates a commons-based ecosystem where everyone benefits

**In Practice**: You can use Kéya for any purpose, but if you modify it or run it as a web service, you must share your source code. This ensures the knowledge stays in the commons and benefits everyone.