"""
Type stubs for matplotlib.pyplot to fix scatter API parameter conflicts.
"""
from typing import Any, Optional, Union, Sequence, Tuple, List
import numpy as np
from numpy.typing import ArrayLike

def scatter(
    x: ArrayLike,
    y: ArrayLike,
    *,
    s: Optional[Union[float, ArrayLike]] = None,
    c: Optional[Union[str, ArrayLike]] = None,
    marker: Optional[str] = None,
    cmap: Optional[Union[str, Any]] = None,
    norm: Optional[Any] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    alpha: Optional[float] = None,
    linewidths: Optional[Union[float, ArrayLike]] = None,
    edgecolors: Optional[Union[str, ArrayLike]] = None,
    plotnonfinite: bool = False,
    **kwargs: Any
) -> Any:
    """
    Fixed scatter function signature to avoid parameter conflicts.
    
    Parameters:
    -----------
    s : float or array-like, optional
        The marker size in points**2 (typographic points are 1/72 inch).
        Can be scalar or array of same length as x and y.
    """
    ...

def figure(
    num: Optional[Union[int, str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[float] = None,
    facecolor: Optional[str] = None,
    edgecolor: Optional[str] = None,
    frameon: bool = True,
    FigureClass: Optional[Any] = None,
    clear: bool = False,
    **kwargs: Any
) -> Any:
    """Create a new figure or activate an existing figure."""
    ...

def subplots(
    nrows: int = 1,
    ncols: int = 1,
    *,
    sharex: Union[bool, str] = False,
    sharey: Union[bool, str] = False,
    squeeze: bool = True,
    subplot_kw: Optional[dict[str, Any]] = None,
    gridspec_kw: Optional[dict[str, Any]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs: Any
) -> Tuple[Any, Any]:
    """Create a figure and subplots."""
    ...

def show(block: Optional[bool] = None) -> None:
    """Display all open figures."""
    ...

def close(fig: Optional[Any] = None) -> None:
    """Close a figure window."""
    ...

def savefig(
    fname: str,
    dpi: Optional[Union[str, float]] = None,
    facecolor: Optional[str] = None,
    edgecolor: Optional[str] = None,
    orientation: str = 'portrait',
    papertype: Optional[str] = None,
    format: Optional[str] = None,
    transparent: bool = False,
    bbox_inches: Optional[Union[str, Any]] = None,
    pad_inches: float = 0.1,
    **kwargs: Any
) -> None:
    """Save the current figure."""
    ...

def tight_layout(
    pad: float = 1.08,
    h_pad: Optional[float] = None,
    w_pad: Optional[float] = None,
    rect: Optional[Tuple[float, float, float, float]] = None
) -> None:
    """Automatically adjust subplot parameters."""
    ...

def colorbar(mappable: Any, ax: Optional[Any] = None, **kwargs: Any) -> Any:
    """Add a colorbar to a plot."""
    ...

def suptitle(t: str, **kwargs: Any) -> Any:
    """Add a centered suptitle to the figure."""
    ...

def title(label: str, **kwargs: Any) -> Any:
    """Set a title for the current axes."""
    ...

def xlabel(xlabel: str, **kwargs: Any) -> Any:
    """Set the label for the x-axis."""
    ...

def ylabel(ylabel: str, **kwargs: Any) -> Any:
    """Set the label for the y-axis."""
    ...

def xticks(ticks: Optional[ArrayLike] = None, labels: Optional[Sequence[str]] = None, **kwargs: Any) -> Any:
    """Get or set the current tick locations and labels of the x-axis."""
    ...

def yticks(ticks: Optional[ArrayLike] = None, labels: Optional[Sequence[str]] = None, **kwargs: Any) -> Any:
    """Get or set the current tick locations and labels of the y-axis."""
    ...

def text(x: float, y: float, s: str, **kwargs: Any) -> Any:
    """Add text to the axes."""
    ...

def bar(
    x: ArrayLike,
    height: ArrayLike,
    width: Union[float, ArrayLike] = 0.8,
    bottom: Optional[ArrayLike] = None,
    *,
    align: str = 'center',
    color: Optional[Union[str, ArrayLike]] = None,
    alpha: Optional[float] = None,
    **kwargs: Any
) -> Any:
    """Make a bar plot."""
    ...

def imshow(
    X: ArrayLike,
    *,
    cmap: Optional[Union[str, Any]] = None,
    norm: Optional[Any] = None,
    aspect: Optional[Union[str, float]] = None,
    interpolation: Optional[str] = None,
    alpha: Optional[float] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    origin: Optional[str] = None,
    extent: Optional[Tuple[float, float, float, float]] = None,
    **kwargs: Any
) -> Any:
    """Display an image on the axes."""
    ...

def gcf() -> Any:
    """
    Get the current figure.
    
    If no current figure exists, a new one is created using figure().
    
    Returns
    -------
    figure : Figure
        The current figure.
    """
    ...

rcParams: dict[str, Any] 