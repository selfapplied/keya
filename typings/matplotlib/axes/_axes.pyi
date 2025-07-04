"""
Type stubs for matplotlib 3D axes functionality.
Fixes Pylance issues with 3D API method recognition.
"""
from typing import Any, Optional, Union, Tuple, List
from matplotlib.artist import Artist
from matplotlib.text import Text
from matplotlib.transforms import Transform
import numpy as np
from numpy.typing import ArrayLike

class Axes3D:
    """Type stub for matplotlib 3D axes to fix Pylance issues."""
    
    # Core 3D axis methods that Pylance doesn't recognize
    def set_zlabel(
        self, 
        zlabel: str, 
        fontdict: Optional[dict] = None, 
        labelpad: Optional[float] = None,
        *,
        loc: Optional[str] = None,
        **kwargs: Any
    ) -> Text: ...
    
    def set_zlim(
        self,
        bottom: Optional[float] = None,
        top: Optional[float] = None,
        *,
        emit: bool = True,
        auto: bool = False
    ) -> Tuple[float, float]: ...
    
    def set_zlim3d(
        self,
        bottom: Optional[float] = None, 
        top: Optional[float] = None,
        *,
        emit: bool = True,
        auto: bool = False
    ) -> Tuple[float, float]: ...
    
    def text2D(
        self,
        x: float,
        y: float, 
        s: str,
        fontdict: Optional[dict] = None,
        *,
        transform: Optional[Transform] = None,
        **kwargs: Any
    ) -> Text: ...
    
    def text3D(
        self,
        x: float,
        y: float,
        z: float,
        s: str,
        zdir: Optional[str] = None,
        **kwargs: Any
    ) -> Text: ...

    # 3D axis pane properties that Pylance doesn't recognize
    @property
    def xaxis(self) -> 'XAxis3D': ...
    
    @property  
    def yaxis(self) -> 'YAxis3D': ...
    
    @property
    def zaxis(self) -> 'ZAxis3D': ...

    # Standard matplotlib Axes methods (inherits from 2D)
    def scatter(
        self,
        xs: ArrayLike,
        ys: ArrayLike,
        zs: Optional[ArrayLike] = None,
        *,
        s: Optional[Union[float, ArrayLike]] = None,
        c: Optional[Union[str, ArrayLike]] = None,
        marker: Optional[str] = None,
        cmap: Optional[str] = None,
        norm: Optional[Any] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        alpha: Optional[float] = None,
        linewidths: Optional[Union[float, ArrayLike]] = None,
        edgecolors: Optional[Union[str, ArrayLike]] = None,
        plotnonfinite: bool = False,
        **kwargs: Any
    ) -> Any: ...
    
    def plot_surface(
        self,
        X: ArrayLike,
        Y: ArrayLike, 
        Z: ArrayLike,
        *,
        norm: Optional[Any] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        lightsource: Optional[Any] = None,
        **kwargs: Any
    ) -> Any: ...
    
    def plot_wireframe(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        Z: ArrayLike,
        *,
        rstride: int = 1,
        cstride: int = 1,
        **kwargs: Any
    ) -> Any: ...
    
    def contour(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        Z: ArrayLike,
        *,
        levels: Optional[Union[int, ArrayLike]] = None,
        zdir: str = 'z',
        offset: Optional[float] = None,
        **kwargs: Any
    ) -> Any: ...
    
    def contourf(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        Z: ArrayLike,
        *,
        levels: Optional[Union[int, ArrayLike]] = None,
        zdir: str = 'z', 
        offset: Optional[float] = None,
        **kwargs: Any
    ) -> Any: ...


class XAxis3D:
    """Type stub for 3D X-axis with pane properties."""
    @property
    def pane(self) -> 'Pane3D': ...


class YAxis3D:
    """Type stub for 3D Y-axis with pane properties."""
    @property
    def pane(self) -> 'Pane3D': ...


class ZAxis3D:
    """Type stub for 3D Z-axis with pane properties."""
    @property
    def pane(self) -> 'Pane3D': ...


class Pane3D:
    """Type stub for 3D axis pane properties."""
    fill: bool
    color: Any
    alpha: float


# For backward compatibility, also provide under common names
Axes = Axes3D  # Common alias 