"""
Parser for the Kéya Shell (.keya) file format.

This parser reads a .keya file and constructs an Abstract Syntax Tree (AST)
representing the experimental pipeline.
"""
from pathlib import Path
import jax.numpy as jnp
from .ast import Pipeline, Step, OperatorType

def parse_kshell_file(filepath: Path) -> Pipeline:
    """
    Parses a .keya file and returns a Pipeline AST object.

    The expected format is:
    pipeline: <name>
    initial_state: [1, 0, 1, ...]
    step { op: FUSE; args: [1]; }
    step { op: DIFF; }
    """
    content = filepath.read_text()
    lines = [line.strip() for line in content.splitlines() if line.strip() and not line.strip().startswith('#')]

    if not lines:
        raise ValueError(f"File is empty or contains only comments: {filepath}")

    # Parse pipeline name
    name_line = lines.pop(0)
    if not name_line.startswith("pipeline:"):
        raise ValueError("File must start with 'pipeline: <name>'")
    pipeline_name = name_line.split(":", 1)[1].strip()

    # Parse initial state
    state_line = lines.pop(0)
    if not state_line.startswith("initial_state:"):
        raise ValueError("Pipeline must declare 'initial_state:' after the name.")
    state_str = state_line.split(":", 1)[1].strip()
    initial_state = jnp.array(eval(state_str), dtype=jnp.int32)

    # Parse steps
    steps = []
    for line in lines:
        if line.startswith("step {") and line.endswith("}"):
            step_content = line[len("step {"):-1].strip()
            parts = [p.strip() for p in step_content.split(';')]
            
            op_str = parts[0]
            if not op_str.startswith("op:"):
                raise ValueError(f"Step must contain 'op:'. Found: {line}")
            op_name = op_str.split(":", 1)[1].strip()
            
            op_type = OperatorType[op_name.upper()]
            
            args = []
            if len(parts) > 1 and parts[1]:
                arg_str_part = parts[1]
                if not arg_str_part.startswith("args:"):
                    raise ValueError(f"Step arguments must be declared with 'args:'. Found: {line}")
                arg_str = arg_str_part.split(":", 1)[1].strip()
                args = eval(arg_str)

            steps.append(Step(operator=op_type, args=args))
        else:
            raise ValueError(f"Invalid step definition: {line}")
            
    return Pipeline(name=pipeline_name, initial_state=initial_state, steps=steps) 