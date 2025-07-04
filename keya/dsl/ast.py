from abc import ABC
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Union


class Operator(Enum):
    """Enumeration of all symbolic operators in the Σ-Calculus."""

    # Binary Operators
    FUSION = auto()  # ⊕
    TENSOR = auto()  # ⊗
    GROWTH = auto()  # ↑

    # Unary Operators
    DESCENT = auto()  # ℓ
    REFLECTION = auto()  # ~
    
    #  Fundamental Operators
    WILD = auto()  # Ϟ - symmetry breaking
    TAME = auto()  # § - resonance creation
    
    # Matrix Operations
    MATRIX_ADD = auto()      # matrix addition
    MATRIX_MULT = auto()     # matrix multiplication
    EXTRACT_STRING = auto()  # extract string from matrix
    
    # String Operations
    STRING_CONCAT = auto()   # string concatenation
    PATTERN_MATCH = auto()   # pattern recognition


class Glyph(Enum):
    """The fundamental symbols in the resonance field."""
    
    VOID = "∅"     # emptiness
    DOWN = "▽"     # primal glyph  
    UP = "△"       # transformed glyph
    UNITY = "⊙"    # contained/stable glyph
    FLOW = "⊕"     # dynamic glyph


class ContainmentType(Enum):
    """Types of containment for the C operator."""
    
    BINARY = "binary"
    DECIMAL = "decimal"  
    STRING = "string"
    GENERAL = "general"


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
class GlyphLiteral(Expression):
    """Represents a glyph literal like ∅, ▽, △, ⊙, ⊕."""
    
    glyph: Glyph


@dataclass(slots=True)
class MatrixLiteral(Expression):
    """Represents a matrix literal with explicit glyph values."""
    
    rows: int
    cols: int
    fill_glyph: Optional[Glyph] = None
    values: Optional[List[List[Glyph]]] = None  # If specific values are given


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


@dataclass(slots=True)
class WildOp(Expression):
    """Represents application of the Ϟ (Wild) operator."""
    
    operand: Expression  # Matrix expression


@dataclass(slots=True) 
class TameOp(Expression):
    """Represents application of the § (Tame) operator."""
    
    operand: Expression  # Matrix expression
    containment_type: ContainmentType


@dataclass(slots=True)
class WildTameCycle(Expression):
    """Represents a full ∮ () cycle operation."""
    
    operand: Expression  # Matrix expression
    containment_type: ContainmentType
    max_iterations: Optional[Union[int, float]] = None  # Support infinity (float('inf'))


@dataclass(slots=True)
class MatrixBinaryArithmetic(Expression):
    """Binary arithmetic using emergent base systems."""
    
    left: Expression   # Matrix expression
    right: Expression  # Matrix expression
    operation: str     # "add" or "multiply"


@dataclass(slots=True)
class StringFromSeed(Expression):
    """Generate string from seed glyph using grammar."""
    
    seed_glyph: GlyphLiteral
    grammar: Union[Expression, "GrammarDef"]  # Can be expression that resolves to GrammarDef
    length: Expression  # Number expression


@dataclass(slots=True)
class PatternMatch(Expression):
    """Pattern matching in strings."""
    
    string_expr: Expression
    pattern: List[Glyph]


@dataclass(slots=True)
class StringConcat(Expression):
    """String concatenation operation."""
    
    left: Expression   # String/matrix expression
    right: Expression  # String/matrix expression


# --- Grammar and Language Constructs ---

@dataclass(slots=True)
class GrammarRule(ASTNode):
    """A single grammar production rule: glyph -> [glyphs]."""
    
    from_glyph: Glyph
    to_glyphs: List[Glyph]


@dataclass(slots=True)
class GrammarDef(Expression):
    """Grammar definition with production rules."""
    
    name: str
    rules: List[GrammarRule]


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
class Assignment(Statement):
    """Represents an assignment, e.g., color = #000000."""

    target: Variable
    value: Expression


@dataclass(slots=True)
class MatrixAssignment(Statement):
    """Assignment of matrix expressions."""
    
    target: Variable
    value: Expression  # Matrix expression


@dataclass(slots=True)
class GrammarAssignment(Statement):
    """Assignment of grammar definitions."""
    
    target: Variable
    grammar: GrammarDef


@dataclass(slots=True)
class Action(Statement):
    """Represents a keyword action, e.g., stop."""

    name: str


@dataclass(slots=True)
class ResonanceTrace(Statement):
    """Compute and output resonance trace of a matrix."""
    
    matrix_expr: Expression


@dataclass(slots=True)
class VerifyArithmetic(Statement):
    """Verify that base system emergence works correctly."""
    
    test_size: Optional[Expression] = None


@dataclass(slots=True)
class VerifyStrings(Statement):
    """Verify that string generation works correctly."""
    
    pass


# --- Main Program Structure ---


@dataclass(slots=True)
class Section(ASTNode):
    """A named section like 'boundary' or 'operators'."""

    name: str
    statements: List[Statement]


@dataclass(slots=True)
class Definition(ASTNode):
    """The root of the AST, representing a full definition block."""

    name: str
    sections: List[Section]
    def_type: str = "definition"  # e.g., "image", "grammar", "matrix", "resonance"


# --- Program Types ---

@dataclass(slots=True)
class MatrixProgram(Definition):
    """A program that works with glyph matrices."""
    
    def_type: str = "matrix"


@dataclass(slots=True)
class GrammarProgram(Definition):
    """A program that defines and uses string grammars."""
    
    def_type: str = "grammar"


@dataclass(slots=True)
class ResonanceProgram(Definition):
    """A program that explores resonance and symbolic emergence."""
    
    def_type: str = "resonance"
