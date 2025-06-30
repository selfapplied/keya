import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Tuple

import numpy as np

from ..dsl.ast import Expression


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
    """The core execution engine for the Kéya language."""

    def __init__(self):
        self.symbol_table: dict[str, tuple[str, Callable]] = {
            "beta": ("ß", TemporalOperator),
            "nabla": ("∇", CurvatureOperator),
            "cycle": ("⊙", CycleOperator),
            "lint": ("🔎", self.op_lint_and_print),
        }
        self.reversal = {v[0]: k for k, v in self.symbol_table.items()}
        self.manifold = Manifold()

    def process_line(self, line: str):
        """Processes a single line of Kéya code."""
        line = line.strip()
        if not line or line.startswith("#"):
            return

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
