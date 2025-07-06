# Matplotlib type stubs for keya project
"""
Main matplotlib type stubs to fix 3D API recognition issues.
Exports properly typed 3D functionality.
"""

# Import classes defined in our stubs
from typing import Any

def use(backend: str, *, force: bool = True) -> None:
    """
    Select the matplotlib backend to use for rendering and GUI integration.
    
    Parameters
    ----------
    backend : str
        The backend to switch to. Common backends include:
        - 'Agg': Anti-Grain Geometry (raster) for file output
        - 'Qt5Agg': Qt5 with Anti-Grain Geometry
        - 'TkAgg': Tk with Anti-Grain Geometry
    force : bool, default: True
        If True, force the backend switch even if already set.
    """
    ...

def get_backend() -> str:
    """
    Return the name of the current backend.
    """
    ...

# Type stubs are automatically discovered by mypy
# No need to export specific symbols 