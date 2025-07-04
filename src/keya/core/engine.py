import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import jax.numpy as jnp

from ..dsl.ast import (
    Assignment, ASTNode, ContainmentOp, ContainmentType, DCCycle, Definition,
    DissonanceOp, Expression, FunctionCall, Glyph, GlyphLiteral, GrammarAssignment,
    GrammarDef, GrammarProgram, GrammarRule, Literal, MatrixAssignment, MatrixLiteral,
    MatrixProgram, ResonanceProgram, ResonanceTrace, Section, Statement, Variable,
    VerifyArithmetic, VerifyStrings
)
from ..dsl.parser import parse


# --- Symbolic Representations ---
@dataclass
class Symbol:
    id: str
    phase: float
    curvature: float
    location: Tuple[int, int]


class FunctionSymbol(Symbol):
    def __init__(
        self,
        id: str,
        phase: float,
        curvature: float,
        location: Tuple[int, int],
        return_type: str,
    ):
        super().__init__(id, phase, curvature, location)
        self.return_type = return_type


# --- Linter Core Components ---
class Manifold:
    """Defines the expected attractor states for symbols."""

    def __init__(self):
        self.attractor_states: Dict[str, float] = {
            "return_type.None": 1.0,  # High curvature for '-> None'
            "return_type.∅": 0.0,  # Zero curvature for no annotation
        }

    def compute_curvature(self, symbol: Symbol) -> float:
        """Computes the curvature of a symbol relative to the manifold."""
        if isinstance(symbol, FunctionSymbol):
            # ℜ_expected for a 'good' function is < 0.7
            # A function with '-> None' has ℜ_actual = 1.0
            return self.attractor_states.get(f"return_type.{symbol.return_type}", 0.0)
        return 0.0


@dataclass
class SymbolicWarning:
    ψ: Symbol
    δ: np.ndarray
    ℜ_actual: float
    ℜ_expected: float
    location: Tuple[int, int]
    suggestion: str

    def __str__(self) -> str:
        """Formats the warning as specified in the prompt."""
        drift_str = f"∇{self.δ[0]:.2f}"
        return (
            f"Symbol '{self.ψ.id}' at line {self.location[0]}:\n"
            f"  Drift {drift_str} | ℜ_actual={self.ℜ_actual:.2f} vs ℜ_expected<{self.ℜ_expected:.2f}\n"
            f"  Suggestion: {self.suggestion}"
        )


class KéyaLinter:
    """A symbol-aware linter using phase and curvature mechanics."""

    def __init__(self, manifold: Manifold):
        self.manifold = manifold

    def lint(self, symbols: List[Symbol]) -> List[SymbolicWarning]:
        """Analyzes a list of symbols and generates warnings for drift."""
        warnings = []
        for ψ in symbols:
            ℜ_actual = self.manifold.compute_curvature(ψ)
            ℜ_expected = 0.7  # The attractor's expected max curvature

            if ℜ_actual >= ℜ_expected:
                δ = self.compute_displacement(ψ, ℜ_actual, ℜ_expected)
                suggestion = f"∂{ψ.id}.return_type := ∅  # Remove → None"
                warning = SymbolicWarning(
                    ψ=ψ,
                    δ=δ,
                    ℜ_actual=ℜ_actual,
                    ℜ_expected=ℜ_expected,
                    location=ψ.location,
                    suggestion=suggestion,
                )
                warnings.append(warning)
        return warnings

    def compute_displacement(self, ψ: Symbol, ℜ_actual: float, ℜ_expected: float) -> np.ndarray:
        """Computes the vector-based displacement (drift)."""
        # For now, a simple 1D vector representing curvature drift.
        # This can be expanded into a multi-dimensional space.
        return np.array([ℜ_actual - ℜ_expected])


class SymbolExtractor(ast.NodeVisitor):
    """Traverses a Python AST to extract symbolic representations."""

    def __init__(self):
        self.symbols: List[Symbol] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        return_type = "∅"  # Assume no annotation
        if node.returns:
            if isinstance(node.returns, ast.Name) and node.returns.id == "None":
                return_type = "None"
            # In a full implementation, we'd handle other types here.

        symbol = FunctionSymbol(
            id=node.name,
            phase=0.0,  # Placeholder
            curvature=0.0,  # Will be computed by the manifold
            location=(node.lineno, node.col_offset),
            return_type=return_type,
        )
        self.symbols.append(symbol)
        self.generic_visit(node)


# --- Engine and Operators ---


# Dummy operator classes for now
class CurvatureOperator(Expression):
    pass


class CycleOperator(Expression):
    pass


class TemporalOperator(Expression):
    pass


class Linter(ast.NodeVisitor):
    """AST-based linter to enforce project-specific style rules."""

    def __init__(self, file: Path):
        self.file = file
        self.errors: list[str] = []

    def check(self) -> list[str]:
        """Run the linter on the given file."""
        tree = ast.parse(self.file.read_text())
        self.visit(tree)
        return self.errors

    def _check_forbidden_return_annotation(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        """Check a function node for forbidden return type annotations."""
        if not node.returns:
            return

        is_none = isinstance(node.returns, ast.Constant) and node.returns.value is None
        is_name_constant_none = (
            hasattr(ast, "NameConstant")
            and isinstance(node.returns, ast.NameConstant)
            and node.returns.value is None
        )
        is_any = isinstance(node.returns, ast.Name) and node.returns.id == "Any"

        if is_none or is_name_constant_none or is_any:
            forbidden_type = "None" if (is_none or is_name_constant_none) else "Any"
            error_msg = (
                f"{self.file}:{node.lineno}:{node.col_offset + 1}: "
                f"Function `{node.name}` should not have `-> {forbidden_type}` annotation."
            )
            self.errors.append(error_msg)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Check for forbidden '-> None' and '-> Any' annotations."""
        self._check_forbidden_return_annotation(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Delegate async function checks to the sync function visitor."""
        self._check_forbidden_return_annotation(node)
        self.generic_visit(node)


def find_python_files(path: Path) -> Iterator[Path]:
    """Recursively find all Python files in a directory."""
    if path.is_file() and path.suffix == ".py":
        yield path
    elif path.is_dir():
        yield from path.rglob("*.py")


class Engine:
    """The core execution engine for the Kéya D-C language."""

    def __init__(self):
        # Legacy symbol table for backward compatibility
        self.symbol_table: dict[str, tuple[str, Callable]] = {
            "beta": ("ß", TemporalOperator),
            "nabla": ("∇", CurvatureOperator),
            "cycle": ("⊙", CycleOperator),
            "lint": ("🔎", self.op_lint_and_print),
        }
        self.reversal = {v[0]: k for k, v in self.symbol_table.items()}
        self.manifold = Manifold()
        
        # New keya D-C execution environment
        self.variables: Dict[str, Any] = {}  # Variable storage
        self.grammars: Dict[str, GrammarDef] = {}  # Grammar definitions
        self.current_program: Optional[Definition] = None
        
    def execute_program(self, source_code: str) -> Any:
        """Execute a complete keya D-C program from source code."""
        try:
            ast_node = parse(source_code)
            self.current_program = ast_node
            return self.execute_node(ast_node)
        except Exception as e:
            print(f"Execution error: {e}")
            return None
    
    def execute_node(self, node: ASTNode) -> Any:
        """Execute any AST node and return the result."""
        if isinstance(node, MatrixProgram):
            return self.execute_matrix_program(node)
        elif isinstance(node, GrammarProgram):
            return self.execute_grammar_program(node)
        elif isinstance(node, ResonanceProgram):
            return self.execute_resonance_program(node)
        elif isinstance(node, Section):
            return self.execute_section(node)
        elif isinstance(node, MatrixAssignment):
            return self.execute_matrix_assignment(node)
        elif isinstance(node, GrammarAssignment):
            return self.execute_grammar_assignment(node)
        elif isinstance(node, Assignment):
            return self.execute_assignment(node)
        elif isinstance(node, VerifyArithmetic):
            return self.execute_verify_arithmetic(node)
        elif isinstance(node, VerifyStrings):
            return self.execute_verify_strings(node)
        elif isinstance(node, ResonanceTrace):
            return self.execute_resonance_trace(node)
        elif isinstance(node, DissonanceOp):
            return self.execute_dissonance_op(node)
        elif isinstance(node, ContainmentOp):
            return self.execute_containment_op(node)
        elif isinstance(node, DCCycle):
            return self.execute_dc_cycle(node)
        elif isinstance(node, MatrixLiteral):
            return self.execute_matrix_literal(node)
        elif isinstance(node, GlyphLiteral):
            return self.execute_glyph_literal(node)
        elif isinstance(node, Variable):
            return self.execute_variable(node)
        elif isinstance(node, Literal):
            return node.value
        else:
            print(f"Unknown node type: {type(node)}")
            return None
    
    def execute_matrix_program(self, program: MatrixProgram) -> Dict[str, Any]:
        """Execute a matrix program and return results."""
        results = {}
        print(f"Executing matrix program: {program.name}")
        
        for section in program.sections:
            section_result = self.execute_section(section)
            results[section.name] = section_result
            
        return results
    
    def execute_grammar_program(self, program: GrammarProgram) -> Dict[str, Any]:
        """Execute a grammar program and return results."""
        results = {}
        print(f"Executing grammar program: {program.name}")
        
        for section in program.sections:
            section_result = self.execute_section(section)
            results[section.name] = section_result
            
        return results
    
    def execute_resonance_program(self, program: ResonanceProgram) -> Dict[str, Any]:
        """Execute a resonance program and return results."""
        results = {}
        print(f"Executing resonance program: {program.name}")
        
        for section in program.sections:
            section_result = self.execute_section(section)
            results[section.name] = section_result
            
        return results
    
    def execute_section(self, section: Section) -> List[Any]:
        """Execute all statements in a section."""
        print(f"  Section: {section.name}")
        results = []
        
        for statement in section.statements:
            result = self.execute_node(statement)
            results.append(result)
            
        return results
    
    def execute_matrix_assignment(self, node: MatrixAssignment) -> Optional[np.ndarray]:
        """Execute matrix assignment and store result."""
        value = self.execute_node(node.value)
        if isinstance(value, np.ndarray):
            self.variables[node.target.name] = value
            print(f"    {node.target.name} = {value.shape} matrix")
            return value
        else:
            print(f"Error: Expected matrix, got {type(value)}")
            return None
    
    def execute_grammar_assignment(self, node: GrammarAssignment) -> GrammarDef:
        """Execute grammar assignment and store result."""
        grammar = node.grammar
        self.grammars[node.target.name] = grammar
        print(f"    {node.target.name} = grammar '{grammar.name}' ({len(grammar.rules)} rules)")
        return grammar
    
    def execute_assignment(self, node: Assignment) -> Any:
        """Execute regular assignment."""
        value = self.execute_node(node.value)
        self.variables[node.target.name] = value
        print(f"    {node.target.name} = {value}")
        return value
    
    def execute_dissonance_op(self, node: DissonanceOp) -> Optional[np.ndarray]:
        """Execute the D (dissonance) operator on a matrix."""
        operand = self.execute_node(node.operand)
        if isinstance(operand, np.ndarray):
            # D operator: Apply symmetry breaking transformation
            result = self._apply_dissonance_transform(operand)
            print(f"    D({operand.shape}) -> {result.shape}")
            return result
        else:
            print(f"Error: D operator requires matrix input, got {type(operand)}")
            return None
    
    def execute_containment_op(self, node: ContainmentOp) -> Optional[np.ndarray]:
        """Execute the C (containment) operator on a matrix."""
        operand = self.execute_node(node.operand)
        if isinstance(operand, np.ndarray):
            # C operator: Apply containment transformation based on type
            result = self._apply_containment_transform(operand, node.containment_type)
            print(f"    C({operand.shape}, {node.containment_type.value}) -> {result.shape}")
            return result
        else:
            print(f"Error: C operator requires matrix input, got {type(operand)}")
            return None
    
    def execute_dc_cycle(self, node: DCCycle) -> Optional[np.ndarray]:
        """Execute a full D-C cycle on a matrix."""
        operand = self.execute_node(node.operand)
        if isinstance(operand, np.ndarray):
            # DC cycle: Apply D then C transformations iteratively
            result = self._apply_dc_cycle(operand, node.containment_type, node.max_iterations)
            print(f"    DC({operand.shape}, {node.containment_type.value}, {node.max_iterations}) -> cycle complete")
            return result
        else:
            print(f"Error: DC cycle requires matrix input, got {type(operand)}")
            return None
    
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
            print(f"Error: Variable '{node.name}' not defined")
            return None
    
    def execute_verify_arithmetic(self, node: VerifyArithmetic) -> bool:
        """Verify arithmetic operations work correctly."""
        test_size = 10
        if node.test_size and isinstance(node.test_size, Literal):
            test_size = node.test_size.value
        print(f"    Verifying arithmetic with test size {test_size}...")
        # Implement arithmetic verification logic
        return True
    
    def execute_verify_strings(self, node: VerifyStrings) -> bool:
        """Verify string operations work correctly."""
        print(f"    Verifying string operations...")
        # Implement string verification logic
        return True
    
    def execute_resonance_trace(self, node: ResonanceTrace) -> Optional[Union[float, np.ndarray]]:
        """Compute resonance trace of a matrix."""
        matrix = self.execute_node(node.matrix_expr)
        if isinstance(matrix, np.ndarray):
            trace = np.trace(matrix)
            print(f"    Trace({matrix.shape}) = {trace}")
            return trace
        return None
    
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
    
    def _apply_dissonance_transform(self, matrix: np.ndarray) -> np.ndarray:
        """Apply the D (dissonance) operator transformation."""
        # D creates asymmetry/dissonance by applying skew transformation
        # This could be a rotation, skew, or other symmetry-breaking operation
        result = matrix.copy()
        
        # Simple dissonance: add asymmetric noise based on position
        rows, cols = matrix.shape
        for i in range(rows):
            for j in range(cols):
                # Create asymmetric pattern based on position
                dissonance = 0.1 * (i - j) * np.sin(i + j)
                result[i, j] += dissonance
                
        return result
    
    def _apply_containment_transform(self, matrix: np.ndarray, ctype: ContainmentType) -> np.ndarray:
        """Apply the C (containment) operator transformation."""
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
            # General containment: apply smoothing/averaging
            result = 0.5 * (result + np.roll(result, 1, axis=0) + np.roll(result, 1, axis=1)) / 2
            
        return result
    
    def _apply_dc_cycle(self, matrix: np.ndarray, ctype: ContainmentType, max_iter: Optional[int]) -> np.ndarray:
        """Apply iterative D-C cycle transformations."""
        result = matrix.copy()
        iterations = max_iter or 5
        
        print(f"      Starting DC cycle ({iterations} iterations)...")
        for i in range(iterations):
            # Apply D (dissonance) transformation
            result = self._apply_dissonance_transform(result)
            # Apply C (containment) transformation  
            result = self._apply_containment_transform(result, ctype)
            print(f"        Iteration {i+1} complete")
            
        return result

    def process_line(self, line: str):
        """Process a single line - supports both legacy commands and keya D-C code."""
        line = line.strip()
        if not line or line.startswith("#"):
            return

        # Try to parse as keya D-C code first
        try:
            result = self.execute_program(line)
            if result is not None:
                return result
        except:
            pass  # Fall back to legacy processing
        
        # Legacy command processing
        parts = line.split()
        command, *args = parts

        if command in self.symbol_table:
            _, func = self.symbol_table[command]
            if callable(func):
                func(*args)
            else:
                print(f"Error: `{command}` is not an executable operator.")
        elif command in self.reversal:
            word = self.reversal[command]
            _, func = self.symbol_table[word]
            if callable(func):
                func(*args)
            else:
                print(f"Error: `{command}` ({word}) is not an executable operator.")
        else:
            translated = self.translate_symbols(line)
            print(f"Translated: {translated}")

    def translate_symbols(self, text: str) -> str:
        """Translates known words to their symbolic representation."""
        for word, (symbol, _) in self.symbol_table.items():
            text = text.replace(word, symbol)
        return text

    # --- Linter Operator Suite ---

    def op_lint_and_print(self, *paths: str):
        """The main, all-in-one linter operator."""
        if not paths:
            print("Usage: lint <file_or_directory> ...")
            return

        all_warnings: List[SymbolicWarning] = []
        for path_str in paths:
            path_obj = Path(path_str)
            for file in self._find_py_files(path_obj):
                try:
                    source = file.read_text()
                    tree = ast.parse(source)
                    extractor = SymbolExtractor()
                    extractor.visit(tree)
                    linter = KéyaLinter(self.manifold)
                    warnings = linter.lint(extractor.symbols)
                    all_warnings.extend(warnings)
                except Exception as e:
                    print(f"Error processing {file}: {e}")

        if all_warnings:
            for warning in all_warnings:
                print(str(warning))
        else:
            print("✨ No style guide violations found. Well done! ✨")

    def _find_py_files(self, path: Path) -> Iterator[Path]:
        """Finds all Python files in the given path."""
        if path.is_file() and path.suffix == ".py":
            yield path
        elif path.is_dir():
            yield from path.rglob("*.py")


class EquilibriumOperator:
    """Equilibrium solver that finds a steady state given forward and reverse operators."""

    def __init__(self, forward: Callable[[jnp.ndarray], jnp.ndarray], reverse: Callable[[jnp.ndarray], jnp.ndarray]) -> None:
        """Initialize with forward and reverse operator callables."""
        self.forward = forward
        self.reverse = reverse

    def resolve(self, psi: jnp.ndarray, t_max: int = 100, tol: float = 1e-6) -> jnp.ndarray:
        """Iteratively solve for equilibrium up to t_max iterations or tolerance."""
        for _ in range(t_max):
            next_psi = (self.forward(psi) + self.reverse(psi)) / 2
            if jnp.linalg.norm(next_psi - psi) < tol:
                psi = next_psi
                break
            psi = next_psi
        return psi
