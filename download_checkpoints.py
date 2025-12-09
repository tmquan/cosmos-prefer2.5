#!/usr/bin/env python3
"""
Download Cosmos model checkpoints from Hugging Face.

This script downloads checkpoints using the HuggingFace cache structure,
ensuring compatibility with cosmos-predict2.5 and cosmos-transfer2.5 inference.

The checkpoints are stored in the HuggingFace cache format at:
    {HF_HOME}/hub/models--nvidia--Cosmos-Predict2.5-2B/snapshots/...
    
For example, with HF_HOME=/workspace/checkpoints:
    /workspace/checkpoints/hub/models--nvidia--Cosmos-Predict2.5-2B/snapshots/.../base/post-trained/*.pt

This is the same location that cosmos inference will look for, preventing duplicate downloads.

Usage:
    # Set HF_HOME before running inference to use these cached files
    export HF_HOME=/workspace/checkpoints
    
    python download_checkpoints.py --help
    python download_checkpoints.py --model predict2.5-2b-posttrained
    python download_checkpoints.py --model transfer2.5-2b-depth --cache-dir /custom/path
"""

import argparse
import os
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download, snapshot_download
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn, TimeRemainingColumn

console = Console()


def print_tree(directory: Path, prefix: str = "", is_last: bool = True):
    """Print directory tree structure, hiding hidden files/folders (starting with .)."""
    if not directory.exists():
        return
    
    # Print current directory/file
    connector = "└── " if is_last else "├── "
    console.print(f"{prefix}{connector}[cyan]{directory.name}[/cyan]")
    
    if directory.is_dir():
        # Get all items in directory, excluding hidden files/folders (starting with .)
        items = [item for item in directory.iterdir() if not item.name.startswith('.')]
        items = sorted(items, key=lambda x: (not x.is_dir(), x.name))
        
        # Update prefix for children
        extension = "    " if is_last else "│   "
        
        for i, item in enumerate(items):
            is_last_item = i == len(items) - 1
            print_tree(item, prefix + extension, is_last_item)


# Model checkpoint configurations
MODELS = {
    # Cosmos-Predict2.5 models
    "predict2.5-2b-pretrained": {
        "repo_id": "nvidia/Cosmos-Predict2.5-2B",
        "files": ["base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt"],
        "description": "Cosmos-Predict2.5-2B Pre-trained Base Model",
    },
    "predict2.5-2b-posttrained": {
        "repo_id": "nvidia/Cosmos-Predict2.5-2B",
        "files": ["base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt"],
        "description": "Cosmos-Predict2.5-2B Post-trained Base Model",
    },
    "predict2.5-2b-auto-multiview": {
        "repo_id": "nvidia/Cosmos-Predict2.5-2B",
        "files": [
            "auto/multiview/524af350-2e43-496c-8590-3646ae1325da_ema_bf16.pt",
            "auto/multiview/6b9d7548-33bb-4517-b5e8-60caf47edba7_ema_bf16.pt",
        ],
        "description": "Cosmos-Predict2.5-2B Auto Multiview (Autonomous Driving)",
    },
    "predict2.5-2b-robot-action": {
        "repo_id": "nvidia/Cosmos-Predict2.5-2B",
        "files": [
            "robot/action-cond/38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt",
            "robot/action-cond/cr1_empty_string_text_embeddings.pt",
        ],
        "description": "Cosmos-Predict2.5-2B Robot Action-Conditioned",
    },
    
    # Cosmos-Transfer2.5 models
    "transfer2.5-2b-depth": {
        "repo_id": "nvidia/Cosmos-Transfer2.5-2B",
        "files": [
            "general/depth/0f214f66-ae98-43cf-ab25-d65d09a7e68f_ema_bf16.pt",
            "general/depth/626e6618-bfcd-4d9a-a077-1409e2ce353f_ema_bf16.pt",
        ],
        "description": "Cosmos-Transfer2.5-2B Depth Control",
    },
    "transfer2.5-2b-edge": {
        "repo_id": "nvidia/Cosmos-Transfer2.5-2B",
        "files": [
            "general/edge/61f5694b-0ad5-4ecd-8ad7-c8545627d125_ema_bf16.pt",
            "general/edge/ecd0ba00-d598-4f94-aa09-e8627899c431_ema_bf16.pt",
        ],
        "description": "Cosmos-Transfer2.5-2B Edge Control",
    },
    "transfer2.5-2b-seg": {
        "repo_id": "nvidia/Cosmos-Transfer2.5-2B",
        "files": [
            "general/seg/5136ef49-6d8d-42e8-8abf-7dac722a304a_ema_bf16.pt",
            "general/seg/fcab44fe-6fe7-492e-b9c6-67ef8c1a52ab_ema_bf16.pt",
        ],
        "description": "Cosmos-Transfer2.5-2B Segmentation Control",
    },
    "transfer2.5-2b-auto-multiview": {
        "repo_id": "nvidia/Cosmos-Transfer2.5-2B",
        "files": [
            "auto/multiview/4ecc66e9-df19-4aed-9802-0d11e057287a_ema_bf16.pt",
            "auto/multiview/b5ab002d-a120-4fbf-a7f9-04af8615710b_ema_bf16.pt",
        ],
        "description": "Cosmos-Transfer2.5-2B Auto Multiview (Autonomous Driving)",
    },
}


def list_available_models():
    """Display a list of available models."""
    console.print("\n[bold cyan]Available Cosmos Model Checkpoints:[/bold cyan]\n")
    
    for key, info in MODELS.items():
        model_type = "Predict" if "predict" in key else "Transfer"
        console.print(f"  • [cyan]{key}[/cyan]")
        console.print(f"    Type: [green]{model_type}[/green]")
        console.print(f"    Description: [yellow]{info['description']}[/yellow]")
        console.print(f"    Files: [white]{len(info['files'])} files[/white]")
        console.print()


def download_model(model_key: str, cache_dir: Optional[Path] = None):
    """Download a specific model checkpoint using HuggingFace cache."""
    if model_key not in MODELS:
        console.print(f"[red]Error: Model '{model_key}' not found![/red]")
        console.print("\nAvailable models:")
        list_available_models()
        return False
    
    model_info = MODELS[model_key]
    repo_id = model_info["repo_id"]
    files = model_info["files"]
    
    # Set HF_HOME environment variable to control where HuggingFace caches files
    # HuggingFace will create a 'hub' subdirectory: {HF_HOME}/hub/models--org--repo/...
    if cache_dir is None:
        # Default to /workspace/checkpoints in Docker environment
        if Path("/workspace").exists():
            cache_dir = Path("/workspace/checkpoints")
        else:
            cache_dir = Path("./checkpoints")
    else:
        cache_dir = Path(cache_dir)
    
    # Set HF_HOME to our cache directory
    # HuggingFace will automatically use {HF_HOME}/hub/ for downloads
    os.environ['HF_HOME'] = str(cache_dir)
    console.print(f"[dim]HF_HOME set to: {cache_dir}[/dim]")
    console.print(f"[dim]Files will be cached in: {cache_dir}/hub/[/dim]")
    
    console.print(f"\n[bold cyan]Downloading {model_info['description']}[/bold cyan]")
    console.print(f"Repository: [green]{repo_id}[/green]")
    console.print(f"Number of files: [blue]{len(files)}[/blue]\n")
    
    downloaded_paths = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        
        for file_path in files:
            task_id = progress.add_task(f"Downloading {Path(file_path).name}", total=None)
            
            try:
                # Use hf_hub_download with NO cache_dir parameter
                # This respects HF_HOME and uses the standard structure
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=file_path,
                    # Don't pass cache_dir - let HF_HOME control the location
                )
                downloaded_paths.append(Path(local_path))
                progress.update(task_id, completed=True, description=f"✓ {Path(file_path).name}")
                
            except Exception as e:
                console.print(f"[red]Error downloading {file_path}: {e}[/red]")
                progress.update(task_id, completed=True, description=f"✗ {Path(file_path).name}")
                return False
    
    console.print(f"\n[bold green]✓ Successfully downloaded {len(downloaded_paths)} files![/bold green]\n")
    console.print("[bold cyan]Downloaded files location:[/bold cyan]")
    for path in downloaded_paths:
        console.print(f"  • [dim]{path}[/dim]")
    console.print()
    
    return True


def download_all_models(cache_dir: Optional[Path] = None):
    """Download all available model checkpoints."""
    console.print("[bold yellow]Downloading ALL model checkpoints...[/bold yellow]")
    console.print("[yellow]This will download several GB of data![/yellow]\n")
    
    success_count = 0
    fail_count = 0
    
    for model_key in MODELS.keys():
        if download_model(model_key, cache_dir):
            success_count += 1
        else:
            fail_count += 1
        console.print("")
    
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  ✓ Success: [green]{success_count}[/green]")
    console.print(f"  ✗ Failed: [red]{fail_count}[/red]")


def main():
    parser = argparse.ArgumentParser(
        description="Download Cosmos model checkpoints from Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available models
  python download_checkpoints.py --list
  
  # Download a specific model
  python download_checkpoints.py --model predict2.5-2b-posttrained
  
  # Download to a custom directory
  python download_checkpoints.py --model transfer2.5-2b-depth --cache-dir /custom/path
  
  # Download all models
  python download_checkpoints.py --all

Important:
  This script sets HF_HOME to ensure cosmos inference uses the same cache.
  To use these checkpoints for inference, set the environment variable:
  
    export HF_HOME=/workspace/checkpoints
    
  Then run your inference scripts. This prevents duplicate downloads.
  
  The files are stored in HuggingFace's standard cache format:
    /workspace/checkpoints/models--nvidia--Cosmos-Predict2.5-2B/snapshots/...
        """
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available models"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Model key to download (use --list to see available models)"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available models"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Download directory for models (default: /workspace/checkpoints in Docker, ./checkpoints otherwise)"
    )
    
    args = parser.parse_args()
    
    # Display header
    console.print("[bold magenta]   Cosmos Model Checkpoint Downloader[/bold magenta]")
    
    if args.list:
        list_available_models()
    elif args.all:
        download_all_models(args.cache_dir)
    elif args.model:
        download_model(args.model, args.cache_dir)
    else:
        parser.print_help()
        console.print("\n[yellow]Tip: Use --list to see available models[/yellow]")
        return
    
    # Print directory tree at the end
    final_cache_dir = Path(args.cache_dir) if args.cache_dir else (
        Path("/workspace/checkpoints") if Path("/workspace").exists() else Path("./checkpoints")
    )
    
    # Check for the hub subdirectory where HF actually stores files
    hub_dir = final_cache_dir / "hub"
    
    if hub_dir.exists():
        console.print(f"\n[bold cyan]HF_HOME:[/bold cyan] [yellow]{final_cache_dir}[/yellow]")
        console.print(f"[bold cyan]Cache Directory:[/bold cyan] [yellow]{hub_dir}[/yellow]\n")
        
        # Show HF cache structure (models--org--repo format)
        if list(hub_dir.glob("models--*")):
            console.print("[bold cyan]Downloaded Models (HuggingFace Cache Format):[/bold cyan]\n")
            for model_dir in sorted(hub_dir.glob("models--nvidia--Cosmos-*")):
                console.print(f"  • [cyan]{model_dir.name}[/cyan]")
            console.print()
        
        # Print directory tree
        console.print("[bold cyan]Directory Structure:[/bold cyan]\n")
        print_tree(hub_dir, prefix="", is_last=True)
        console.print()
        
        console.print("[bold green]✓ To use these checkpoints for inference, set:[/bold green]")
        console.print(f"[yellow]  export HF_HOME={final_cache_dir}[/yellow]\n")
        console.print("[dim]This ensures cosmos inference uses these cached files instead of re-downloading.[/dim]\n")


if __name__ == "__main__":
    main()
