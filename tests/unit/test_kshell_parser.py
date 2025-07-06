import pytest
from pathlib import Path
import jax.numpy as jnp

from keya.kshell.parser import parse_kshell_file
from keya.kshell.ast import Pipeline, Step, OperatorType

@pytest.fixture
def create_test_file(tmp_path: Path):
    """A pytest fixture to create temporary .keya files for testing."""
    def _create_file(content: str, name: str = "test.keya") -> Path:
        file_path = tmp_path / name
        file_path.write_text(content)
        return file_path
    return _create_file

def test_parse_valid_pipeline(create_test_file):
    """Tests parsing a valid .keya file with all features."""
    content = """
# This is a comment, it should be ignored.
pipeline: my_test_pipeline
initial_state: [1, 0, 1, 0]
step { op: FUSE; args: [1, 2] }
step { op: DIFF }
    """
    test_file = create_test_file(content)
    pipeline = parse_kshell_file(test_file)

    assert isinstance(pipeline, Pipeline)
    assert pipeline.name == "my_test_pipeline"
    assert jnp.array_equal(pipeline.initial_state, jnp.array([1, 0, 1, 0], dtype=jnp.int32))
    assert len(pipeline.steps) == 2

    # Check first step
    assert isinstance(pipeline.steps[0], Step)
    assert pipeline.steps[0].operator == OperatorType.FUSE
    assert pipeline.steps[0].args == [1, 2]

    # Check second step
    assert isinstance(pipeline.steps[1], Step)
    assert pipeline.steps[1].operator == OperatorType.DIFF
    assert pipeline.steps[1].args == []

def test_parse_minimal_pipeline(create_test_file):
    """Tests parsing the simplest possible valid .keya file."""
    content = """
pipeline: minimal
initial_state: [0]
step { op: IDENTITY }
    """
    test_file = create_test_file(content)
    pipeline = parse_kshell_file(test_file)

    assert pipeline.name == "minimal"
    assert jnp.array_equal(pipeline.initial_state, jnp.array([0], dtype=jnp.int32))
    assert len(pipeline.steps) == 1
    assert pipeline.steps[0].operator == OperatorType.IDENTITY
    assert pipeline.steps[0].args == []

def test_parser_error_handling(create_test_file):
    """Tests that the parser raises appropriate ValueErrors for malformed files."""
    invalid_contents = {
        "missing_pipeline_name": "initial_state: [0]",
        "missing_initial_state": "pipeline: my_pipe",
        "invalid_step": "pipeline: my_pipe\ninitial_state: [0]\nstep op: FUSE",
        "invalid_op_keyword": "pipeline: p\ninitial_state: [0]\nstep { operator: FUSE }",
        "unknown_op": "pipeline: p\ninitial_state: [0]\nstep { op: UNKNOWN }",
        "empty_file": "# Only comments",
    }

    for key, content in invalid_contents.items():
        test_file = create_test_file(content, name=f"{key}.keya")
        with pytest.raises(ValueError, match=".*"):
            parse_kshell_file(test_file)

def test_empty_file_error(create_test_file):
    """Ensures a completely empty file raises a ValueError."""
    test_file = create_test_file("")
    with pytest.raises(ValueError, match="File is empty"):
        parse_kshell_file(test_file) 