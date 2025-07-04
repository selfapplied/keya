"""
Type stubs for matplotlib.figure to ensure 3D functionality is properly recognized.
"""
from typing import Any, Optional, Union, Tuple
from typing import Any

class Figure:
    """Type stub for matplotlib Figure with 3D support."""
    
    def add_subplot(
        self, 
        *args: Any,
        projection: Optional[str] = None,
        **kwargs: Any
    ) -> Any:
        """
        Add subplot with proper 3D typing when projection='3d'.
        
        When projection='3d', returns Axes3D with all 3D methods available.
        """
        ... 