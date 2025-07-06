# Agent Development Guide

This document outlines principles and style guides for the AI agent working on this project.

## Demos

- **Claims Must Be Asserted:** When creating or modifying demos, any print statement or comment that makes a claim about the demo's behavior or the engine's properties must be backed by a corresponding `assert` statement. This ensures our demos are rigorous and self-validating.
## Development Style Guide

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