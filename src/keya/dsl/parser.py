"""Parser for the keya D-C language."""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from .ast import (
    Action,
    # Statements
    Assignment,
    # Core AST types
    ASTNode,
    # Operations
    BinaryOp,
    Boundary,
    ContainmentOp,
    ContainmentType,
    DCCycle,
    Definition,
    DissonanceOp,
    Expression,
    FunctionCall,
    # Enums
    Glyph,
    GlyphLiteral,
    GrammarAssignment,
    GrammarDef,
    GrammarProgram,
    # Grammar constructs
    GrammarRule,
    # Literals and basic expressions
    Literal,
    MatrixAssignment,
    MatrixBinaryArithmetic,
    MatrixLiteral,
    MatrixProgram,
    Operator,
    PatternMatch,
    ResonanceProgram,
    ResonanceTrace,
    # Program types
    Section,
    Statement,
    StringConcat,
    StringFromSeed,
    UnaryOp,
    Variable,
    VerifyArithmetic,
    VerifyStrings,
)


class ParseError(Exception):
    """Exception raised when parsing fails."""
    
    def __init__(self, message: str, line: int = 0, column: int = 0):
        self.message = message
        self.line = line
        self.column = column
        super().__init__(f"Parse error at line {line}, column {column}: {message}")


@dataclass
class Token:
    """A token in the input stream."""
    
    type: str
    value: str
    line: int
    column: int


class Lexer:
    """Tokenizes keya D-C source code."""
    
    # Token patterns
    TOKENS = [
        ('COMMENT', r'#.*$'),
        ('NEWLINE', r'\n'),
        ('WHITESPACE', r'[ \t]+'),
        
        # Glyph literals
        ('GLYPH_VOID', r'∅'),
        ('GLYPH_DOWN', r'▽'),
        ('GLYPH_UP', r'△'),
        ('GLYPH_UNITY', r'⊙'),
        ('GLYPH_FLOW', r'⊕'),
        
        # D-C operators (order matters - longer patterns first)
        ('DC_CYCLE', r'\b(?:DC|dc)\b'),
        ('D_OP', r'𝔻|D'),
        ('C_OP', r'ℂ|C'),
        
        # Matrix brackets
        ('MATRIX_START', r'\['),
        ('MATRIX_END', r'\]'),
        
        # Containment types (with word boundaries)
        ('BINARY_TYPE', r'\bbinary\b'),
        ('DECIMAL_TYPE', r'\bdecimal\b'),
        ('STRING_TYPE', r'\bstring\b'),
        ('GENERAL_TYPE', r'\bgeneral\b'),
        
        # Keywords (with word boundaries to prevent partial matches)
        ('MATRIX', r'\bmatrix\b'),
        ('GRAMMAR', r'\bgrammar\b'),
        ('RESONANCE', r'\bresonance\b'),
        ('RULE', r'\brule\b'),
        ('VERIFY', r'\bverify\b'),
        ('TRACE', r'\btrace\b'),
        ('ARITHMETIC', r'\barithmetic\b'),
        ('STRINGS', r'\bstrings\b'),
        ('PATTERN', r'\bpattern\b'),
        ('MATCH', r'\bmatch\b'),
        ('CONCAT', r'\bconcat\b'),
        ('FROM', r'\bfrom\b'),
        ('SEED', r'\bseed\b'),
        ('LENGTH', r'\blength\b'),
        
        # Symbols
        ('ARROW', r'→|->'),
        ('ASSIGN', r'='),
        ('LPAREN', r'\('),
        ('RPAREN', r'\)'),
        ('LBRACE', r'\{'),
        ('RBRACE', r'\}'),
        ('SEMICOLON', r';'),
        ('COMMA', r','),
        ('PIPE', r'\|'),
        ('PLUS', r'\+'),
        ('MULTIPLY', r'\*'),
        ('DOT', r'\.'),
        
        # Identifiers and literals
        ('NUMBER', r'\d+(\.\d+)?'),
        ('STRING', r'"[^"]*"'),
        ('IDENTIFIER', r'[a-zA-Z_][a-zA-Z0-9_]*'),
    ]
    
    def __init__(self, text: str):
        self.text = text
        self.tokens = []
        self.current = 0
        self.line = 1
        self.column = 1
        
    def tokenize(self) -> List[Token]:
        """Tokenize the input text."""
        
        patterns = [(name, re.compile(pattern, re.MULTILINE)) for name, pattern in self.TOKENS]
        
        pos = 0
        while pos < len(self.text):
            matched = False
            
            for token_type, pattern in patterns:
                match = pattern.match(self.text, pos)
                if match:
                    value = match.group(0)
                    
                    if token_type not in ['WHITESPACE', 'COMMENT']:
                        self.tokens.append(Token(token_type, value, self.line, self.column))
                    
                    if token_type == 'NEWLINE':
                        self.line += 1
                        self.column = 1
                    else:
                        self.column += len(value)
                    
                    pos = match.end()
                    matched = True
                    break
            
            if not matched:
                raise ParseError(f"Unexpected character: {self.text[pos]}", self.line, self.column)
        
        return self.tokens


class Parser:
    """Parses keya D-C source code into an AST."""
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.current = 0
        
    def parse(self) -> Definition:
        """Parse the token stream into an AST."""
        
        if not self.tokens:
            raise ParseError("Empty input")
        
        return self.parse_program()
    
    def peek(self, offset: int = 0) -> Optional[Token]:
        """Peek at the token at the current position + offset."""
        
        pos = self.current + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return None
    
    def consume(self, expected_type: Optional[str] = None) -> Token:
        """Consume and return the current token."""
        
        if self.current >= len(self.tokens):
            raise ParseError("Unexpected end of input")
        
        token = self.tokens[self.current]
        if expected_type and token.type != expected_type:
            raise ParseError(f"Expected {expected_type}, got {token.type}", token.line, token.column)
        
        self.current += 1
        return token
    
    def match(self, *token_types: str) -> bool:
        """Check if the current token matches any of the given types."""
        
        token = self.peek()
        return token is not None and token.type in token_types
    
    def skip_newlines(self):
        """Skip any NEWLINE tokens."""
        
        while self.peek() and self.peek().type == 'NEWLINE':
            self.current += 1
    
    def parse_program(self) -> Definition:
        """Parse a complete program definition."""
        
        # Determine program type
        if self.match('MATRIX'):
            return self.parse_matrix_program()
        elif self.match('GRAMMAR'):
            return self.parse_grammar_program()
        elif self.match('RESONANCE'):
            return self.parse_resonance_program()
        else:
            raise ParseError("Expected program type: matrix, grammar, or resonance")
    
    def parse_matrix_program(self) -> MatrixProgram:
        """Parse a matrix program."""
        
        self.consume('MATRIX')  # Consume the MATRIX token that was matched
        name_token = self.consume('IDENTIFIER')
        self.skip_newlines()
        self.consume('LBRACE')  # Consume opening brace for the program
        self.skip_newlines()
        
        sections = []
        while not self.match('RBRACE'):
            sections.append(self.parse_section())
            self.skip_newlines()
        
        self.consume('RBRACE')  # Consume closing brace for the program
        return MatrixProgram(name=name_token.value, sections=sections)
    
    def parse_grammar_program(self) -> GrammarProgram:
        """Parse a grammar program."""
        
        self.consume('GRAMMAR')  # Consume the GRAMMAR token that was matched
        name_token = self.consume('IDENTIFIER')
        self.skip_newlines()
        self.consume('LBRACE')  # Consume opening brace for the program
        self.skip_newlines()
        
        sections = []
        while not self.match('RBRACE'):
            sections.append(self.parse_section())
            self.skip_newlines()
        
        self.consume('RBRACE')  # Consume closing brace for the program
        return GrammarProgram(name=name_token.value, sections=sections)
    
    def parse_resonance_program(self) -> ResonanceProgram:
        """Parse a resonance program."""
        
        self.consume('RESONANCE')  # Consume the RESONANCE token that was matched
        name_token = self.consume('IDENTIFIER')
        self.skip_newlines()
        self.consume('LBRACE')  # Consume opening brace for the program
        self.skip_newlines()
        
        sections = []
        while not self.match('RBRACE'):
            sections.append(self.parse_section())
            self.skip_newlines()
        
        self.consume('RBRACE')  # Consume closing brace for the program
        return ResonanceProgram(name=name_token.value, sections=sections)
    
    def parse_section(self) -> Section:
        """Parse a section within a program."""
        
        name_token = self.consume('IDENTIFIER')
        self.skip_newlines()
        self.consume('LBRACE')
        self.skip_newlines()
        
        statements = []
        while not self.match('RBRACE'):
            statements.append(self.parse_statement())
            self.skip_newlines()
        
        self.consume('RBRACE')
        
        return Section(name=name_token.value, statements=statements)
    
    def parse_statement(self) -> Statement:
        """Parse a statement."""
        
        if self.match('VERIFY'):
            return self.parse_verify_statement()
        elif self.match('TRACE'):
            return self.parse_trace_statement()
        elif self.match('IDENTIFIER') and self.peek(1) and self.peek(1).type == 'ASSIGN':
            return self.parse_assignment()
        else:
            # Could be a boundary condition or other statement
            expr = self.parse_expression()
            
            if self.match('ARROW'):
                # Boundary condition
                self.consume('ARROW')
                consequence = self.parse_statement()
                return Boundary(condition=expr, consequence=consequence)
            else:
                # Standalone expression - convert to action
                if isinstance(expr, Variable):
                    return Action(name=expr.name)
                else:
                    raise ParseError("Expected statement")
    
    def parse_verify_statement(self) -> Union[VerifyArithmetic, VerifyStrings]:
        """Parse a verify statement."""
        
        self.consume('VERIFY')
        
        if self.match('ARITHMETIC'):
            self.consume('ARITHMETIC')
            test_size = None
            if self.match('NUMBER'):
                test_size = Literal(int(self.consume('NUMBER').value))
            return VerifyArithmetic(test_size=test_size)
        
        elif self.match('STRINGS'):
            self.consume('STRINGS')
            return VerifyStrings()
        
        else:
            raise ParseError("Expected 'arithmetic' or 'strings' after 'verify'")
    
    def parse_trace_statement(self) -> ResonanceTrace:
        """Parse a trace statement."""
        
        self.consume('TRACE')
        matrix_expr = self.parse_expression()
        return ResonanceTrace(matrix_expr=matrix_expr)
    
    def parse_assignment(self) -> Statement:
        """Parse an assignment statement."""
        
        target = Variable(self.consume('IDENTIFIER').value)
        self.consume('ASSIGN')
        
        if self.match('GRAMMAR'):
            # Grammar assignment
            grammar = self.parse_grammar_def()
            return GrammarAssignment(target=target, grammar=grammar)
        else:
            # Regular or matrix assignment
            value = self.parse_expression()
            if self.is_matrix_expression(value):
                return MatrixAssignment(target=target, value=value)
            else:
                return Assignment(target=target, value=value)
    
    def parse_grammar_def(self) -> GrammarDef:
        """Parse a grammar definition."""
        
        self.consume('GRAMMAR')
        name_token = self.consume('IDENTIFIER')
        self.skip_newlines()
        
        self.consume('LBRACE')
        self.skip_newlines()
        rules = []
        
        while not self.match('RBRACE'):
            rules.append(self.parse_grammar_rule())
            self.skip_newlines()
        
        self.consume('RBRACE')
        
        return GrammarDef(name=name_token.value, rules=rules)
    
    def parse_grammar_rule(self) -> GrammarRule:
        """Parse a grammar rule."""
        
        from_glyph = self.parse_glyph()
        self.skip_newlines()
        self.consume('ARROW')
        self.skip_newlines()
        
        to_glyphs = []
        if self.match('MATRIX_START'):
            self.consume('MATRIX_START')
            while not self.match('MATRIX_END'):
                to_glyphs.append(self.parse_glyph())
                if self.match('COMMA'):
                    self.consume('COMMA')
            self.consume('MATRIX_END')
        else:
            to_glyphs.append(self.parse_glyph())
        
        return GrammarRule(from_glyph=from_glyph, to_glyphs=to_glyphs)
    
    def parse_expression(self) -> Expression:
        """Parse an expression."""
        
        return self.parse_binary_op()
    
    def parse_binary_op(self) -> Expression:
        """Parse binary operations with precedence."""
        
        left = self.parse_unary_op()
        
        while self.match('PLUS', 'MULTIPLY', 'CONCAT'):
            op_token = self.consume()
            right = self.parse_unary_op()
            
            if op_token.type == 'PLUS':
                left = MatrixBinaryArithmetic(left=left, right=right, operation="add")
            elif op_token.type == 'MULTIPLY':
                left = MatrixBinaryArithmetic(left=left, right=right, operation="multiply")
            elif op_token.type == 'CONCAT':
                left = StringConcat(left=left, right=right)
        
        return left
    
    def parse_unary_op(self) -> Expression:
        """Parse unary operations."""
        
        if self.match('D_OP'):
            self.consume('D_OP')
            operand = self.parse_primary()
            return DissonanceOp(operand=operand)
        
        elif self.match('C_OP'):
            self.consume('C_OP')
            self.consume('LPAREN')
            operand = self.parse_expression()
            self.consume('COMMA')
            containment_type = self.parse_containment_type()
            self.consume('RPAREN')
            return ContainmentOp(operand=operand, containment_type=containment_type)
        
        elif self.match('DC_CYCLE'):
            self.consume('DC_CYCLE')
            self.consume('LPAREN')
            operand = self.parse_expression()
            self.consume('COMMA')
            containment_type = self.parse_containment_type()
            
            max_iterations = None
            if self.match('COMMA'):
                self.consume('COMMA')
                max_iterations = int(self.consume('NUMBER').value)
            
            self.consume('RPAREN')
            return DCCycle(operand=operand, containment_type=containment_type, max_iterations=max_iterations)
        
        else:
            return self.parse_primary()
    
    def parse_primary(self) -> Expression:
        """Parse primary expressions."""
        
        if self.match('NUMBER'):
            value = self.consume('NUMBER').value
            if '.' in value:
                return Literal(float(value))
            else:
                return Literal(int(value))
        
        elif self.match('STRING'):
            value = self.consume('STRING').value[1:-1]  # Remove quotes
            return Literal(value)
        
        elif self.match('IDENTIFIER'):
            name = self.consume('IDENTIFIER').value
            
            if self.match('LPAREN'):
                # Function call
                self.consume('LPAREN')
                args = []
                
                while not self.match('RPAREN'):
                    args.append(self.parse_expression())
                    if self.match('COMMA'):
                        self.consume('COMMA')
                
                self.consume('RPAREN')
                return FunctionCall(name=name, args=args)
            else:
                return Variable(name)
        
        elif self.is_glyph_token():
            return GlyphLiteral(glyph=self.parse_glyph())
        
        elif self.match('MATRIX_START'):
            return self.parse_matrix_literal()
        
        elif self.match('LPAREN'):
            self.consume('LPAREN')
            expr = self.parse_expression()
            self.consume('RPAREN')
            return expr
        
        elif self.match('SEED'):
            return self.parse_string_from_seed()
        
        elif self.match('PATTERN'):
            return self.parse_pattern_match()
        
        else:
            token = self.peek()
            raise ParseError(f"Unexpected token: {token.type if token else 'EOF'}")
    
    def parse_glyph(self) -> Glyph:
        """Parse a glyph literal."""
        
        if self.match('GLYPH_VOID'):
            self.consume('GLYPH_VOID')
            return Glyph.VOID
        elif self.match('GLYPH_DOWN'):
            self.consume('GLYPH_DOWN')
            return Glyph.DOWN
        elif self.match('GLYPH_UP'):
            self.consume('GLYPH_UP')
            return Glyph.UP
        elif self.match('GLYPH_UNITY'):
            self.consume('GLYPH_UNITY')
            return Glyph.UNITY
        elif self.match('GLYPH_FLOW'):
            self.consume('GLYPH_FLOW')
            return Glyph.FLOW
        else:
            raise ParseError("Expected glyph literal")
    
    def parse_matrix_literal(self) -> MatrixLiteral:
        """Parse a matrix literal."""
        
        self.consume('MATRIX_START')
        
        # Parse matrix dimensions or explicit values
        if self.match('NUMBER'):
            rows = int(self.consume('NUMBER').value)
            self.consume('COMMA')
            cols = int(self.consume('NUMBER').value)
            
            fill_glyph = None
            if self.match('COMMA'):
                self.consume('COMMA')
                fill_glyph = self.parse_glyph()
            
            self.consume('MATRIX_END')
            return MatrixLiteral(rows=rows, cols=cols, fill_glyph=fill_glyph)
        
        else:
            # Parse explicit matrix values
            values = []
            while not self.match('MATRIX_END'):
                row = []
                if self.match('MATRIX_START'):
                    self.consume('MATRIX_START')
                    while not self.match('MATRIX_END'):
                        row.append(self.parse_glyph())
                        if self.match('COMMA'):
                            self.consume('COMMA')
                    self.consume('MATRIX_END')
                else:
                    row.append(self.parse_glyph())
                
                values.append(row)
                
                if self.match('COMMA'):
                    self.consume('COMMA')
            
            self.consume('MATRIX_END')
            rows = len(values)
            cols = len(values[0]) if values else 0
            return MatrixLiteral(rows=rows, cols=cols, values=values)
    
    def parse_containment_type(self) -> ContainmentType:
        """Parse a containment type."""
        
        if self.match('BINARY_TYPE'):
            self.consume('BINARY_TYPE')
            return ContainmentType.BINARY
        elif self.match('DECIMAL_TYPE'):
            self.consume('DECIMAL_TYPE')
            return ContainmentType.DECIMAL
        elif self.match('STRING_TYPE'):
            self.consume('STRING_TYPE')
            return ContainmentType.STRING
        elif self.match('GENERAL_TYPE'):
            self.consume('GENERAL_TYPE')
            return ContainmentType.GENERAL
        else:
            raise ParseError("Expected containment type: binary, decimal, string, or general")
    
    def parse_string_from_seed(self) -> StringFromSeed:
        """Parse a string generation from seed."""
        
        self.consume('SEED')
        self.consume('LPAREN')
        seed_glyph = GlyphLiteral(glyph=self.parse_glyph())
        self.consume('COMMA')
        grammar = self.parse_expression()  # Should be a grammar reference
        self.consume('COMMA')
        length = self.parse_expression()
        self.consume('RPAREN')
        
        return StringFromSeed(seed_glyph=seed_glyph, grammar=grammar, length=length)
    
    def parse_pattern_match(self) -> PatternMatch:
        """Parse a pattern match expression."""
        
        self.consume('PATTERN')
        self.consume('LPAREN')
        string_expr = self.parse_expression()
        self.consume('COMMA')
        
        pattern = []
        self.consume('MATRIX_START')
        while not self.match('MATRIX_END'):
            pattern.append(self.parse_glyph())
            if self.match('COMMA'):
                self.consume('COMMA')
        self.consume('MATRIX_END')
        
        self.consume('RPAREN')
        
        return PatternMatch(string_expr=string_expr, pattern=pattern)
    
    def is_glyph_token(self) -> bool:
        """Check if the current token is a glyph."""
        
        return self.match('GLYPH_VOID', 'GLYPH_DOWN', 'GLYPH_UP', 'GLYPH_UNITY', 'GLYPH_FLOW')
    
    def is_matrix_expression(self, expr: Expression) -> bool:
        """Check if an expression is matrix-related."""
        
        return isinstance(expr, (MatrixLiteral, DissonanceOp, ContainmentOp, DCCycle, MatrixBinaryArithmetic))


def parse(text: str) -> Definition:
    """Parse keya D-C source code."""
    
    lexer = Lexer(text)
    tokens = lexer.tokenize()
    
    parser = Parser(tokens)
    return parser.parse() 