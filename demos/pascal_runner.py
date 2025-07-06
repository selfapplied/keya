"""
PascalGaugeField Simulation Runner (Kernel-based)

This script demonstrates how to use the new PascalKernel-based engine.
"""
import os
os.environ['JAX_ENABLE_X64'] = 'True'

from keya.kernel.field import PascalGaugeField
from keya.kernel.operators import Fuse

def run_simulation():
    """
    Initializes and runs a simulation using the new kernel-based engine.
    """
    print("Initializing kernel-based PascalGaugeField...")
    # The depth of the kernel determines the max size of numbers.
    field = PascalGaugeField(depth=16)

    # 1. Initialize the state with a single value and a Fuse operator.
    initial_values = [5]
    operator_chain = [Fuse()]
    field.initialize_state(initial_values, operator_chain)
    print(f"Engine state initialized with: {initial_values}")
    print(f"Operator chain: {[op.name for op in operator_chain]}")

    # 2. Run the evolution.
    print("\nStarting field evolution...")
    final_states = field.evolve()
    print("Evolution complete.")

    # 3. Print the result.
    print("\n--- Simulation Result ---")
    if final_states:
        final_state = final_states[0]
        print(f"Final State Value: {final_state.value}")
        print(f"Final State Representation: {final_state}")
    else:
        print("No final state produced.")
    print("-------------------------\n")

if __name__ == "__main__":
    run_simulation() 