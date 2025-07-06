import pytest
from pathlib import Path
import jax.numpy as jnp

from keya.kshell.parser import parse_kshell_file
from keya.kshell.engine import KShellEngine

@pytest.fixture
def create_test_file(tmp_path: Path):
    """A pytest fixture to create a temporary .keya file for testing."""
    def _create_file(content: str) -> Path:
        file_path = tmp_path / "pipeline.keya"
        file_path.write_text(content)
        return file_path
    return _create_file

def test_full_pipeline_execution(create_test_file):
    """
    Tests the full end-to-end pipeline:
    1. Parsing a .keya file into an AST.
    2. Executing the AST with the KShellEngine.
    3. Verifying the final state.
    """
    # This pipeline starts with [1, 1], applies Fuse, then Diff.
    # The result of Fuse is [1, 2, 1].
    # The result of Diff on that is [-1, -1, 1, 1].
    content = """
pipeline: fuse_and_diff
initial_state: [1, 1]
step { op: FUSE }
step { op: DIFF }
    """
    test_file = create_test_file(content)

    # 1. Parse the file
    pipeline = parse_kshell_file(test_file)
    assert pipeline.name == "fuse_and_diff"

    # 2. Execute the pipeline
    engine = KShellEngine()
    final_state = engine.run(pipeline)

    # 3. Verify the final state
    # Fuse([1,1]) -> [1,2,1]
    # Diff([1,2,1]) -> convolve with [-1, 1] -> [-1, -1, 1, 1]
    expected_state = jnp.array([-1, -1, 1, 1], dtype=jnp.int32)
    assert jnp.array_equal(final_state, expected_state)

def test_identity_pipeline(create_test_file):
    """Tests that a pipeline with only an IDENTITY step doesn't change the state."""
    content = """
pipeline: identity_test
initial_state: [1, 0, 1, 1, 0]
step { op: IDENTITY }
    """
    test_file = create_test_file(content)
    
    pipeline = parse_kshell_file(test_file)
    engine = KShellEngine()
    final_state = engine.run(pipeline)
    
    # The final state should be the same as the initial state
    initial_state = jnp.array([1, 0, 1, 1, 0], dtype=jnp.int32)
    assert jnp.array_equal(final_state, initial_state) 