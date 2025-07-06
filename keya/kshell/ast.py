"""
Abstract Syntax Tree (AST) for the Kéya Shell (kshell) DSL.

This module defines the data structures that represent the declarative
pipelines for running experiments.
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Union, Any

import jax.numpy as jnp

# Represents the data being transformed by the pipeline
State = Union[jnp.ndarray, Any]

class OperatorType(Enum):
    """The types of operators available in the DSL."""
    FUSE = auto()
    DIFF = auto()
    IDENTITY = auto()
    # In the future, we can add more complex operators like 'GAME_OF_LIFE'

@dataclass
class Step:
    """Represents a single step in an experimental pipeline."""
    operator: OperatorType
    # We can add arguments here later, e.g., for rule-specific params
    args: List[Any] = field(default_factory=list)

@dataclass
class Pipeline:
    """Represents a full experimental pipeline."""
    name: str
    initial_state: State
    steps: List[Step] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"Pipeline(name='{self.name}', steps={len(self.steps)})" 