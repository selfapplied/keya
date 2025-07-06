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

## Style Guide

This project follows a specific set of conventions to maintain clarity and consistency.

1.  **Output Directory**: All generated output files should be placed in the `.out/` directory.
2.  **File Naming**: Python filenames should be concise and avoid underscores. For example, `hydrogen_orbital.py` is renamed to `orbital.py`.
    -   *Exception*: Test files must use `test_*.py` pattern as required by pytest and Python testing conventions.
3.  **Code Structure**: We prefer code with minimal indentation depth. This is achieved through:
    -   **Early returns** to reduce nested `if` statements.
    -   **Small, focused functions** that adhere to the single-responsibility principle.
    -   **Factoring out common expressions** to keep the code DRY (Don't Repeat Yourself). 
4.  **Purposeful Comments**: Code should be self-documenting. Avoid comments that explain *what* the code is doing. Instead, reserve comments for explaining the *why*—the complex logic, the reasoning behind a choice, or the physical model being implemented.
5.  **No `isinstance`**: The use of `isinstance` is a signal that code complexity is growing in a disorderly way. It should be replaced by more robust patterns like `typing.Protocol` or `match/case` to restore order.
    -   Exception: numpy arrays and external code patterns may necessitate the pattern. Use DRY patterns where possible to contain the surface area.
6.  **Universal Type Hinting**: All function and method signatures **must** include type hints for every parameter and the return value. It serves as documentation and catches errors very early in the pipeline.
7.  **Structured Return Values**: Avoid returning raw tuples or dictionaries. Instead, use a `@dataclass(slots=True)` to define a clear, typed return object. This prevents ambiguity and leverages the type checker.
    -   *Exception*: This rule does not apply when an external library's API explicitly requires a tuple format (e.g., the step function in `jax.lax.scan`).
8.  **Concise Syntax**: Employ modern Python syntax (e.g., list comprehensions, context managers, decorators) to enhance clarity and reduce verbosity.
9.  **Strategic Docstrings**: Limit docstrings to heavily-used public APIs. Rely on file-level and class-level docstrings for broader context.
10. **Hypothesis-Driven Error Resolution**: When encountering linter errors or stack traces, avoid making broad, stylistic changes. Instead, treat debugging as an experiment.
11. **Avoid Hyper-Defensive Coding**: Trust the type system and handle errors at the appropriate level.

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