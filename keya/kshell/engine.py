"""
The Kéya Shell (kshell) Engine.

This module provides the execution engine for running declarative pipelines
defined in the kshell AST.
"""
from .ast import Pipeline, Step, OperatorType
from keya.kernel.kernel import PascalKernel
from keya.kernel.operators import Fuse, Diff, Identity, Operator

class KShellEngine:
    """
    Executes a kshell pipeline.
    """
    def __init__(self):
        self.kernel = PascalKernel()
        self.operator_map = {
            OperatorType.FUSE: Fuse,
            OperatorType.DIFF: Diff,
            OperatorType.IDENTITY: Identity,
        }

    def _get_operator(self, step: Step) -> Operator:
        """Maps a pipeline step to a concrete Operator instance."""
        op_func = self.operator_map.get(step.operator)
        if not op_func:
            raise ValueError(f"Unknown operator type: {step.operator}")
        return op_func(*step.args)

    def run(self, pipeline: Pipeline):
        """
        Runs the full pipeline and returns the final state.
        """
        print(f"--- Running Pipeline: {pipeline.name} ---")
        current_state = pipeline.initial_state
        print(f"Initial State: {current_state}")

        for i, step in enumerate(pipeline.steps):
            print(f"Step {i+1}: Applying {step.operator.name} operator...")
            op = self._get_operator(step)
            current_state = self.kernel.apply_polynomial(current_state, op.coeffs)
            print(f"  -> State: {current_state}")

        print("--- Pipeline Complete ---")
        return current_state 