#!/usr/bin/env python3
"""
Environment setup utilities for Cosmos-Prefer2.5

This module handles:
- PYTHONPATH configuration for cosmos-predict2.5 and cosmos-transfer2.5
- Environment variable setup
- Dependency checking
"""

import os
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()


def setup_pythonpath(
    workspace_dir: Optional[Path] = None,
    verbose: bool = True
) -> tuple[Path, Path]:
    """
    Setup PYTHONPATH to include cosmos-predict2.5 and cosmos-transfer2.5 repositories.
    
    Args:
        workspace_dir: Root directory containing the cosmos repositories.
                      Defaults to /workspace (Docker environment)
        verbose: Print setup information
    
    Returns:
        Tuple of (predict2_path, transfer2_path)
    
    Example:
        >>> from cosmos_prefer2.utils.environment import setup_pythonpath
        >>> predict_path, transfer_path = setup_pythonpath()
        >>> print(f"Cosmos-Predict2.5: {predict_path}")
        >>> print(f"Cosmos-Transfer2.5: {transfer_path}")
    """
    if workspace_dir is None:
        # Default to Docker workspace
        workspace_dir = Path("/workspace")
    
    workspace_dir = Path(workspace_dir).resolve()
    
    # Paths to the cosmos repositories
    
    prefer2_path = workspace_dir / "cosmos-prefer2.5"
    predict2_path = workspace_dir / "cosmos-predict2.5"
    transfer2_path = workspace_dir / "cosmos-transfer2.5"
    
    # Check if paths exist
    paths_to_add = []
    for path, name in [ (prefer2_path, "cosmos-prefer2.5"),
                        (predict2_path, "cosmos-predict2.5"), 
                        (transfer2_path, "cosmos-transfer2.5")]:
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
                paths_to_add.append((name, path))
        else:
            console.print(f"[yellow]Warning: {name} not found at {path}[/yellow]")
    
    # Also add PYTHONPATH environment variable
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    new_paths = [str(p) for _, p in paths_to_add]
    
    if current_pythonpath:
        new_paths.append(current_pythonpath)
    
    os.environ["PYTHONPATH"] = ":".join(new_paths)
    
    if verbose and paths_to_add:
        console.print("\n[bold green]✓ PYTHONPATH configured successfully![/bold green]\n")
        console.print("[bold cyan]Added Paths:[/bold cyan]\n")
        
        for name, path in paths_to_add:
            console.print(f"  • [cyan]{name}[/cyan]: [yellow]{path}[/yellow]")
        
        console.print()
    
    return predict2_path, transfer2_path


def check_environment(verbose: bool = True) -> dict[str, bool]:
    """
    Check if the environment is properly configured.
    
    Args:
        verbose: Print detailed status information
    
    Returns:
        Dictionary with status of various environment components
    
    Example:
        >>> from cosmos_prefer2.utils.environment import check_environment
        >>> status = check_environment()
        >>> if all(status.values()):
        ...     print("Environment is ready!")
    """
    import importlib.util
    
    status = {}
    
    # Check PyTorch and CUDA
    try:
        import torch
        status["torch"] = True
        status["cuda"] = torch.cuda.is_available()
        status["gpu_count"] = torch.cuda.device_count() if status["cuda"] else 0
    except ImportError:
        status["torch"] = False
        status["cuda"] = False
        status["gpu_count"] = 0
    
    # Check key dependencies
    dependencies = [
        "transformers",
        "diffusers",
        "mediapy",
        "einops",
        "rich",
        "av",
    ]
    
    for dep in dependencies:
        status[dep] = importlib.util.find_spec(dep) is not None
    
    # Check if cosmos repositories are accessible
    status["cosmos_predict2"] = importlib.util.find_spec("cosmos_predict2") is not None
    status["cosmos_transfer2"] = importlib.util.find_spec("cosmos_transfer2") is not None
    
    if verbose:
        console.print("\n[bold]Environment Status[/bold]\n")
        
        # PyTorch and CUDA status
        console.print("[bold cyan]PyTorch & CUDA:[/bold cyan]")
        console.print(f"  • PyTorch: {'✓ Installed' if status['torch'] else '✗ Not found'}")
        console.print(f"  • CUDA: {'✓ Available' if status['cuda'] else '✗ Not available'}")
        if status["cuda"]:
            console.print(f"  • GPU Count: {status['gpu_count']}")
            if status["torch"]:
                import torch
                console.print(f"  • PyTorch Version: {torch.__version__}")
                console.print(f"  • CUDA Version: {torch.version.cuda or 'N/A'}")
        console.print()
        
        # Dependencies status
        console.print("[bold cyan]Dependencies:[/bold cyan]")
        for dep in dependencies:
            status_text = "✓ Installed" if status[dep] else "✗ Not found"
            color = "green" if status[dep] else "red"
            console.print(f"  • {dep}: [{color}]{status_text}[/{color}]")
        console.print()
        
        # Cosmos repositories status
        console.print("[bold cyan]Cosmos Repositories:[/bold cyan]")
        for repo in ["cosmos_predict2", "cosmos_transfer2"]:
            status_text = "✓ Accessible" if status[repo] else "✗ Not found"
            color = "green" if status[repo] else "yellow"
            console.print(f"  • {repo}: [{color}]{status_text}[/{color}]")
        console.print()
        
        # Overall status
        all_critical = (
            status["torch"] and 
            status["cuda"] and 
            all(status[dep] for dep in dependencies[:4])  # Core deps
        )
        
        if all_critical:
            console.print("[bold green]✓ Environment is ready![/bold green]")
        else:
            console.print("[bold yellow]⚠ Some components are missing. Please install dependencies.[/bold yellow]")
    
    return status


def get_workspace_dir() -> Path:
    """Get the workspace directory (usually /workspace in Docker)."""
    # Try to find workspace dir
    possible_paths = [
        Path("/workspace"),
        Path.home() / "workspace",
        Path.cwd().parent,
    ]
    
    for path in possible_paths:
        if (path / "cosmos-predict2.5").exists():
            return path
    
    # Default to /workspace
    return Path("/workspace")


if __name__ == "__main__":
    console.print("[bold cyan]Cosmos-Prefer2.5 Environment Setup[/bold cyan]\n")
    
    # Setup PYTHONPATH
    predict_path, transfer_path = setup_pythonpath()
    
    # Check environment
    status = check_environment()

