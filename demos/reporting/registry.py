"""
Demo Registration System

This module provides a decorator-based system for registering demos
and their metadata for automatic inclusion in reports.
"""
from dataclasses import dataclass, field
from typing import List, Callable, Dict, Optional

@dataclass
class Artifact:
    """Represents a single output artifact from a demo."""
    filename: str
    caption: str = ""

@dataclass
class DemoInfo:
    """Holds all metadata for a registered demo."""
    title: str
    script_path: str
    description: str
    func: Optional[Callable] = None
    claims: List[str] = field(default_factory=list)
    findings: str = ""
    artifacts: List[Artifact] = field(default_factory=list)

# The global registry that will be populated by the decorators.
DEMO_REGISTRY: List[DemoInfo] = []

def register_demo(title: str, artifacts: Optional[List[Dict[str, str]]] = None, claims: Optional[List[str]] = None, findings: str = ""):
    """
    A decorator to register a demo function with the reporting system.
    
    The decorator extracts the description from the function's docstring.
    """
    final_artifacts = artifacts or []
    final_claims = claims or []

    def decorator(func: Callable):
        # Extract description from docstring
        description = func.__doc__
        if description:
            description = description.strip()
        else:
            description = "No description provided."

        # Get the path of the script where the demo is defined
        import inspect
        script_path = inspect.getfile(func)
        
        # Create Artifact objects
        processed_artifacts = [Artifact(filename=a['filename'], caption=a.get('caption', '')) for a in final_artifacts]

        demo_info = DemoInfo(
            title=title,
            script_path=script_path,
            description=description,
            func=func,
            claims=final_claims,
            findings=findings,
            artifacts=processed_artifacts
        )
        DEMO_REGISTRY.append(demo_info)
        
        return func
    return decorator 