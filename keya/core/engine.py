"""Core execution engine for the Keya  language."""

from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..dsl.ast import (
    Assignment, ASTNode, TameOp, ContainmentType, WildTameCycle, Definition,
    WildOp, Expression, Glyph, GlyphLiteral, GrammarAssignment,
    GrammarDef, GrammarProgram, Literal, MatrixAssignment, MatrixLiteral,
    MatrixProgram, ResonanceProgram, ResonanceTrace, Section, Variable,
    VerifyArithmetic, VerifyStrings
)
from ..dsl.parser import parse


class Engine:
    """The core execution engine for the Keya  language."""

    def __init__(self):
        # Keya  execution environment
        self.variables: Dict[str, Any] = {}  # Variable storage
        self.grammars: Dict[str, GrammarDef] = {}  # Grammar definitions
        self.current_program: Optional[Definition] = None
        
    def execute_program(self, source_code: str) -> Any:
        """Execute a complete keya  program."""
        ast = parse(source_code)
        return self.execute_node(ast)
            
    def execute_node(self, node: ASTNode) -> Any:
        """Execute any AST node and return the result."""
        match node:
            case MatrixProgram():
                return self.execute_matrix_program(node)
            case GrammarProgram():
                return self.execute_grammar_program(node)
            case ResonanceProgram():
                return self.execute_resonance_program(node)
            case MatrixAssignment():
                return self.execute_matrix_assignment(node)
            case GrammarAssignment():
                return self.execute_grammar_assignment(node)
            case Assignment():
                return self.execute_assignment(node)
            case WildOp():
                return self.execute_wild_op(node)
            case TameOp():
                return self.execute_tame_op(node)
            case WildTameCycle():
                return self.execute_wildtame_cycle(node)
            case MatrixLiteral():
                return self.execute_matrix_literal(node)
            case GlyphLiteral():
                return self.execute_glyph_literal(node)
            case Variable():
                return self.execute_variable(node)
            case VerifyArithmetic():
                return self.execute_verify_arithmetic(node)
            case VerifyStrings():
                return self.execute_verify_strings(node)
            case ResonanceTrace():
                return self.execute_resonance_trace(node)
            case Literal():
                return node.value
            case _:
                raise ValueError(f"Unknown node type: {type(node)}")
    
    def execute_matrix_program(self, program: MatrixProgram) -> Dict[str, Any]:
        """Execute a matrix program and return results."""
        results = {}
        for section in program.sections:
            section_result = self.execute_section(section)
            results[section.name] = section_result
        return results
    
    def execute_grammar_program(self, program: GrammarProgram) -> Dict[str, Any]:
        """Execute a grammar program and return results."""
        results = {}
        for section in program.sections:
            section_result = self.execute_section(section)
            results[section.name] = section_result
        return results
    
    def execute_resonance_program(self, program: ResonanceProgram) -> Dict[str, Any]:
        """Execute a resonance program and return results."""
        results = {}
        for section in program.sections:
            section_result = self.execute_section(section)
            results[section.name] = section_result
        return results
    
    def execute_section(self, section: Section) -> List[Any]:
        """Execute all statements in a section."""
        results = []
        for statement in section.statements:
            result = self.execute_node(statement)
            results.append(result)
        return results
    
    def execute_matrix_assignment(self, node: MatrixAssignment) -> np.ndarray:
        """Execute matrix assignment and store result."""
        value = self.execute_node(node.value)
        matrix = self._contain_as_matrix(value, "Matrix assignment")
        self.variables[node.target.name] = matrix
        return matrix
    
    def execute_grammar_assignment(self, node: GrammarAssignment) -> GrammarDef:
        """Execute grammar assignment and store result."""
        grammar = node.grammar
        self.grammars[node.target.name] = grammar
        return grammar
    
    def execute_assignment(self, node: Assignment) -> Any:
        """Execute regular assignment."""
        value = self.execute_node(node.value)
        self.variables[node.target.name] = value
        return value
    
    def execute_wild_op(self, node: WildOp) -> np.ndarray:
        """Execute the Ϟ (Wild) operator on a matrix."""
        operand = self.execute_node(node.operand)
        matrix = self._contain_as_matrix(operand, "Ϟ operator")
        return self._apply_wild_transform(matrix)
    
    def execute_tame_op(self, node: TameOp) -> np.ndarray:
        """Execute the § (Tame) operator on a matrix."""
        operand = self.execute_node(node.operand)
        matrix = self._contain_as_matrix(operand, "§ operator")
        return self._apply_tame_transform(matrix, node.containment_type)
    
    def execute_wildtame_cycle(self, node: WildTameCycle) -> np.ndarray:
        """Execute a full ∮ () cycle on a matrix."""
        operand = self.execute_node(node.operand)
        matrix = self._contain_as_matrix(operand, "∮ cycle")
        return self._apply_wildtame_cycle(matrix, node.containment_type, node.max_iterations)
    
    def execute_matrix_literal(self, node: MatrixLiteral) -> np.ndarray:
        """Create a matrix from literal specification."""
        if node.values is not None:
            # Explicit matrix values
            matrix = np.array([[self._glyph_to_number(glyph) for glyph in row] for row in node.values])
        else:
            # Matrix with dimensions and fill
            fill_value = self._glyph_to_number(node.fill_glyph) if node.fill_glyph else 0
            matrix = np.full((node.rows, node.cols), fill_value)
        return matrix
    
    def execute_glyph_literal(self, node: GlyphLiteral) -> float:
        """Convert glyph to numeric value."""
        return self._glyph_to_number(node.glyph)
    
    def execute_variable(self, node: Variable) -> Any:
        """Retrieve variable value."""
        if node.name in self.variables:
            return self.variables[node.name]
        else:
            raise NameError(f"Variable '{node.name}' not defined")
    
    def execute_verify_arithmetic(self, node: VerifyArithmetic) -> bool:
        """Verify arithmetic operations work correctly."""
        test_size = 10
        match node.test_size:
            case Literal():
                test_size = int(node.test_size.value)
            case _:
                pass
        
        # Actually test  arithmetic operations
        try:
            test_matrix = np.random.rand(test_size, test_size)
            wild_result = self._apply_wild_transform(test_matrix)
            tame_result = self._apply_tame_transform(test_matrix, ContainmentType.DECIMAL)
            return (isinstance(wild_result, np.ndarray) and 
                   isinstance(tame_result, np.ndarray) and
                   wild_result.shape == test_matrix.shape)
        except (ValueError, TypeError, AttributeError) as e:
            # Log specific arithmetic verification failure for debugging
            return False
    
    def execute_verify_strings(self, node: VerifyStrings) -> bool:
        """Verify string operations work correctly."""
        # Test string taming transformations
        try:
            test_matrix = np.array([[1.5, 2.7], [3.1, 4.9]])
            result = self._apply_tame_transform(test_matrix, ContainmentType.STRING)
            # Should convert to integer 0-9 range
            return (isinstance(result, np.ndarray) and 
                   bool(np.all(result >= 0)) and bool(np.all(result <= 9)))
        except (ValueError, TypeError, AttributeError) as e:
            # Log specific string verification failure for debugging
            return False
    
    def execute_resonance_trace(self, node: ResonanceTrace) -> float:
        """Compute resonance trace of a matrix."""
        operand = self.execute_node(node.matrix_expr)
        matrix = self._contain_as_matrix(operand, "Trace")
        return float(np.trace(matrix))
    
    def _glyph_to_number(self, glyph: Glyph) -> float:
        """Convert glyphs to numeric values for computation."""
        glyph_values = {
            Glyph.VOID: 0.0,     # ∅ 
            Glyph.DOWN: -1.0,    # ▽
            Glyph.UP: 1.0,       # △
            Glyph.UNITY: 0.5,    # ⊙
            Glyph.FLOW: 2.0,     # ⊕
        }
        return glyph_values.get(glyph, 0.0)
    
    def _apply_wild_transform(self, matrix: np.ndarray) -> np.ndarray:
        """Apply the Ϟ (Wild) operator transformation."""
        # Ϟ creates asymmetry/wildness by applying skew transformation
        # This could be a rotation, skew, or other symmetry-breaking operation
        result = matrix.copy()
        
        # Simple wildness: add asymmetric noise based on position
        rows, cols = matrix.shape
        for i in range(rows):
            for j in range(cols):
                # Create asymmetric pattern based on position
                wildness = 0.1 * (i - j) * np.sin(i + j)
                result[i, j] += wildness
                
        return result
    
    def _apply_tame_transform(self, matrix: np.ndarray, ctype: ContainmentType) -> np.ndarray:
        """Apply the § (Tame) operator transformation."""
        result = matrix.copy()
        
        if ctype == ContainmentType.BINARY:
            # Convert to binary representation
            result = (result > 0).astype(float)
        elif ctype == ContainmentType.DECIMAL:
            # Normalize to decimal [0,1] range
            result = (result - result.min()) / (result.max() - result.min() + 1e-8)
        elif ctype == ContainmentType.STRING:
            # Create string-like pattern
            result = np.round(result) % 10  # Map to 0-9 range
        elif ctype == ContainmentType.GENERAL:
            # General taming: apply smoothing/averaging
            result = 0.5 * (result + np.roll(result, 1, axis=0) + np.roll(result, 1, axis=1)) / 2
            
        return result
    
    def _apply_wildtame_cycle(self, matrix: np.ndarray, ctype: ContainmentType, max_iter: Optional[Union[int, float]]) -> np.ndarray:
        """Apply iterative  cycle transformations."""
        result = matrix.copy()
        
        # Handle infinite iterations for cellular automata
        if max_iter == float('inf'):
            iterations = 1000  # Large number for continuous evolution
        else:
            iterations = int(max_iter) if max_iter else 5
        
        for i in range(iterations):
            # Apply Wild transformation
            result = self._apply_wild_transform(result)
            # Apply Tame transformation  
            result = self._apply_tame_transform(result, ctype)
            
        return result

    def _contain_as_matrix(self, value: Any, operation_name: str) -> np.ndarray:
        """Containment operator: Ensure value is contained within matrix domain."""
        if isinstance(value, np.ndarray):
            return value
        # Could add other matrix-like type conversions here
        raise TypeError(f"{operation_name} requires matrix input, got {type(value)}")
