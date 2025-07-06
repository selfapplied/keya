# Kéya: An Engine for Symbolic Calculus

Kéya is an experimental computational framework for exploring a core hypothesis: that a vast range of computational systems can be modeled as emergent behaviors of a single, universal, symbolic calculus.

The philosophy of Kéya is not to build a specific tool for a specific problem, but to develop a fundamental "computational substrate" and then discover novel **representations** of problems that can be solved by the substrate's intrinsic rules.

## The Σ-Calculus

The core theory behind Kéya is the "Σ-Calculus," a paradigm built on a few key principles:

*   **Symbolic Fields**: The state of any system is represented not as a single value, but as a field or vector of symbolic units. A number can be a vector of its digits; a physical system can be a field of quantum states.
*   **Universal Transformations**: A minimal set of universal operators are applied to these fields. These operators are not complex functions but fundamental, combinatorial transformations.
*   **Computation as Normalization**: The result of a transformation is often an unstable, "un-normalized" state. The crucial step of computation is applying a universal "carry" or "reduction" rule that propagates through the state until it re-stabilizes.
*   **Emergent Complexity**: Complex, high-level behaviors—the rules of arithmetic, the shapes of orbitals, the patterns of a cellular automaton—are hypothesized to be emergent properties of the simple, underlying normalization rules.

## The `PascalKernel`: A Concrete Implementation

The primary engine implementing the Σ-Calculus today is the `PascalKernel`.

This is a pure, parameter-free mathematical object whose normalization rules are derived from the combinatorial properties of Pascal's triangle modulo 2 (the Sierpinski triangle). It provides a concrete, powerful, and surprisingly versatile foundation for testing the calculus's claims.

## Exploring the Proofs

The best way to understand Kéya is to explore the demos. They are not just examples; they are rigorous, assertion-backed proofs that test the core hypothesis against real-world computational systems.

To see a comprehensive overview, generate the interactive HTML report:

```bash
python -m keya.reporting.builder
```

This command runs all registered demos and creates a detailed report in `.out/report.html`, complete with visualizations, claims, and findings for each experiment. The demos prove that the engine can successfully model:

*   **Formal Arithmetic**: Simulating the subtle, emergent behaviors of floating-point arithmetic using the engine's fundamental binary logic.
*   **Physical Phenomena**: Generating the shapes of quantum atomic orbitals and modeling the evolution of wavefunctions.
*   **Complex Systems**: Running cellular automata and other generative models to show how complex patterns can emerge from simple, local rules.
*   **Declarative Pipelines**: Executing high-level, declarative experimental pipelines via the K-Shell DSL.

## License

Kéya is released under the [GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0), chosen to foster community, sharing, and reinvestment in the project's development.