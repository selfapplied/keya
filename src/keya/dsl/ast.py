from abc import ABC
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Union


class Operator(Enum):
    """Enumeration of all symbolic operators in the Σ-Calculus."""

    # Binary Operators
    FUSION = auto()  # ⊕
    TENSOR = auto()  # ⊗
    GROWTH = auto()  # ↑

    # Unary Operators
    DESCENT = auto()  # ℓ
    REFLECTION = auto()  # ~


class ASTNode(ABC):
    """Base class for all AST nodes."""

    pass


# --- Expressions ---


class Expression(ASTNode):
    """Base class for all expression nodes."""

    pass


@dataclass(slots=True)
class Literal(Expression):
    """Represents a literal value like a number, string, or color."""

    value: Union[int, float, str]


@dataclass(slots=True)
class Variable(Expression):
    """Represents a variable."""

    name: str


@dataclass(slots=True)
class FunctionCall(Expression):
    """Represents a function call, e.g., abs(z)."""

    name: str
    args: List[Expression]


@dataclass(slots=True)
class BinaryOp(Expression):
    """Represents a binary operation, e.g., a + b or |z| > 2.0."""

    left: Expression
    op: Union[str, Operator]
    right: Expression


@dataclass(slots=True)
class UnaryOp(Expression):
    """Represents a unary operation, e.g., ~abs(z) or ℓ(x)."""

    op: Union[str, Operator]
    operand: Expression


# --- Top-level Language Constructs ---


class Statement(ASTNode):
    """Base class for all statements."""

    pass


@dataclass(slots=True)
class Boundary(Statement):
    """Represents a single boundary condition: condition → consequence."""

    condition: Expression
    consequence: Union["Assignment", "Action"]


@dataclass(slots=True)
class Assignment(ASTNode):
    """Represents an assignment, e.g., color = #000000."""

    target: Variable
    value: Expression


@dataclass(slots=True)
class Action(ASTNode):
    """Represents a keyword action, e.g., stop."""

    name: str


# --- Main Program Structure ---


@dataclass(slots=True)
class Section(ASTNode):
    """A named section like 'boundary' or 'operators'."""

    name: str
    statements: List[Statement]


@dataclass(slots=True)
class Definition(ASTNode):
    """The root of the AST, representing a full definition block."""

    def_type: str  # e.g., "image", "grammar"
    name: str
    sections: List[Section]
