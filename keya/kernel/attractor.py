#!/usr/bin/env python3
"""
A generic, reusable engine for finding attractors in dynamic systems.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Generic, TypeVar, List, Optional, Tuple
import operator

# A generic type for the state of the system.
# It must support equality comparison (==).
TState = TypeVar("TState")

class HaltingCondition(Enum):
    """The reason the simulation halted."""
    STABLE_STATE_REACHED = "Stable state reached (Still Life)"
    OSCILLATOR_REACHED = "Oscillator reached"
    MAX_STEPS_REACHED = "Maximum steps reached"
    CONVERGENCE_CRITERIA_MET = "Convergence criteria met (e.g., variance)"
    PROCESS_EXHAUSTED = "Process mathematically exhausted (e.g., descent)"
    STRUCTURAL_CONDITION_MET = "A specific structural condition was met"

@dataclass
class AttractorInfo(Generic[TState]):
    """Holds information about the detected attractor."""
    halting_condition: HaltingCondition
    final_state: TState
    steps_to_reach: int
    period: Optional[int] = None
    history: Optional[List[TState]] = None

class AttractorEngine(Generic[TState]):
    """
    A generic engine for running simulations until they reach an attractor
    or a defined halting condition.
    """
    def __init__(
        self,
        step_function: Callable[[TState], TState],
        history_size: int = 20,
        max_steps: Optional[int] = None,
        convergence_fn: Optional[Callable[[TState, TState], bool]] = None,
        structural_fn: Optional[Callable[[TState], bool]] = None,
        equals_fn: Optional[Callable[[TState, TState], bool]] = None,
    ):
        """
        Initializes the AttractorEngine.

        Args:
            step_function: A function that takes a state and returns the next state.
            history_size: The number of previous states to store for oscillator detection.
            max_steps: A hard limit on the number of simulation steps.
            convergence_fn: An optional function that takes (prev_state, current_state)
                            and returns True if a convergence criterion is met.
            structural_fn: An optional function that takes a single state and returns
                           True if a desired structural property is found.
            equals_fn: An optional function for comparing two states for equality.
                       Defaults to the `==` operator.
        """
        self.step_function = step_function
        self.history_size = history_size
        self.max_steps = max_steps
        self.convergence_fn = convergence_fn
        self.structural_fn = structural_fn
        self.equals_fn = equals_fn or operator.eq

    def run(self, initial_state: TState) -> AttractorInfo[TState]:
        """
        Runs the simulation from an initial state until a halting condition is met.
        """
        state = initial_state
        history: List[TState] = [state]
        
        # Use the user-provided max_steps or a very large number as a fallback.
        effective_max_steps = self.max_steps or 1_000_000 

        for step in range(1, effective_max_steps + 1):
            
            # Sane default: if no max_steps is provided by the user, we implement a
            # dynamic timeout to prevent near-infinite loops in chaotic systems.
            # The heuristic is that a system should not run for more than twice the
            # number of steps it has already taken to get to its current state
            # without repeating.
            if self.max_steps is None and step > 2 * len(history) and len(history) > self.history_size:
                return AttractorInfo(
                    halting_condition=HaltingCondition.MAX_STEPS_REACHED,
                    final_state=state,
                    steps_to_reach=step -1, # It didn't complete this step
                    history=history
                )

            prev_state = state
            state = self.step_function(state)

            # 0. Check for a user-defined structural condition first.
            if self.structural_fn and self.structural_fn(state):
                return AttractorInfo(
                    halting_condition=HaltingCondition.STRUCTURAL_CONDITION_MET,
                    final_state=state,
                    steps_to_reach=step,
                    history=history
                )

            # 1. Check for mathematical exhaustion (the most common case for stable states)
            if self.equals_fn(state, prev_state):
                 return AttractorInfo(
                    halting_condition=HaltingCondition.PROCESS_EXHAUSTED,
                    final_state=state,
                    steps_to_reach=step,
                    period=1,
                    history=history
                )

            # 2. Check for state repetition (Oscillator) by looking in history
            for i, old_state in enumerate(reversed(history)):
                if self.equals_fn(state, old_state):
                    period = i + 1
                    # A period of 1 would have been caught by the exhaustion check above
                    return AttractorInfo(
                        halting_condition=HaltingCondition.OSCILLATOR_REACHED,
                        final_state=state,
                        steps_to_reach=step,
                        period=period,
                        history=history
                    )

            # 3. Check for custom statistical convergence
            if self.convergence_fn and self.convergence_fn(prev_state, state):
                 return AttractorInfo(
                    halting_condition=HaltingCondition.CONVERGENCE_CRITERIA_MET,
                    final_state=state,
                    steps_to_reach=step,
                    history=history
                )
            
            history.append(state)
            if len(history) > self.history_size:
                history.pop(0)
                
        return AttractorInfo(
            halting_condition=HaltingCondition.MAX_STEPS_REACHED,
            final_state=state,
            steps_to_reach=effective_max_steps,
            history=history,
        ) 