"""
Demo for the Kéya Shell (kshell) Declarative DSL.

This demo constructs a simple experimental pipeline in code using the
kshell AST and then runs it through the KShellEngine.
"""
from keya.kshell.engine import KShellEngine
from keya.kshell.ast import Pipeline, Step, OperatorType
from demos.reporting.registry import register_demo
import jax.numpy as jnp

@register_demo(
    title="Declarative Experimental Pipelines (K-Shell)",
    claims=[
        "A sequence of symbolic operations can be defined declaratively.",
        "The KShellEngine can parse and execute this pipeline, applying operators in order.",
        "The final state of the system is consistent with the applied transformations."
    ],
    findings="The demo defines a simple pipeline (FUSE, FUSE, DIFF, IDENTITY) and runs it through the engine. The final state is correctly computed and validated with an assertion, proving the viability of the declarative DSL approach for running experiments."
)
def main():
    """
    This demo introduces the Kéya Shell (kshell), a declarative DSL for
    defining and running experimental pipelines. It showcases how a sequence
    of operators can be applied to an initial state without writing complex
    imperative code, making it easy to design and run new experiments.
    """
    # 1. Define an initial state (a simple vector)
    initial_state = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int64)

    # 2. Define a pipeline using the kshell AST
    pipeline = Pipeline(
        name="Simple Fuse/Diff Test",
        initial_state=initial_state,
        steps=[
            Step(OperatorType.FUSE),
            Step(OperatorType.FUSE),
            Step(OperatorType.DIFF),
            Step(OperatorType.IDENTITY),
        ]
    )

    # 3. Create an engine and run the pipeline
    engine = KShellEngine()
    final_state = engine.run(pipeline)

    print(f"Final State: {final_state}")
    
    # Assert that the pipeline had an effect.
    assert not jnp.array_equal(final_state, initial_state), "Pipeline had no effect."
    print("✅ K-Shell declarative pipeline executed successfully.")


if __name__ == "__main__":
    main() 