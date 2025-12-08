#!/usr/bin/env python3
"""
Download Cosmos model checkpoints from Hugging Face.

This script helps download checkpoints for both Cosmos-Predict2.5 and Cosmos-Transfer2.5 models.

Usage:
    python download_checkpoints.py --help
    python download_checkpoints.py --model predict2.5-2b-base
    python download_checkpoints.py --model transfer2.5-2b-depth --cache-dir ./checkpoints
"""

import argparse
import os
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download, snapshot_download
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn, TimeRemainingColumn
from rich.table import Table

console = Console()

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
    """Display a table of available models."""
    table = Table(title="Available Cosmos Model Checkpoints", show_header=True, header_style="bold magenta")
    table.add_column("Model Key", style="cyan", no_wrap=True)
    table.add_column("Type", style="green")
    table.add_column("Description", style="yellow")
    table.add_column("Files", style="white")
    
    for key, info in MODELS.items():
        model_type = "Predict" if "predict" in key else "Transfer"
        table.add_row(
            key,
            model_type,
            info["description"],
            str(len(info["files"])) + " files"
        )
    
    console.print(table)


def download_model(model_key: str, cache_dir: Optional[Path] = None):
    """Download a specific model checkpoint."""
    if model_key not in MODELS:
        console.print(f"[red]Error: Model '{model_key}' not found![/red]")
        console.print("\nAvailable models:")
        list_available_models()
        return False
    
    model_info = MODELS[model_key]
    repo_id = model_info["repo_id"]
    files = model_info["files"]
    
    # Set default cache directory
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    else:
        cache_dir = Path(cache_dir)
    
    console.print(f"\n[bold cyan]Downloading {model_info['description']}[/bold cyan]")
    console.print(f"Repository: [green]{repo_id}[/green]")
    console.print(f"Cache directory: [yellow]{cache_dir}[/yellow]")
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
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=file_path,
                    cache_dir=str(cache_dir),
                    local_dir_use_symlinks=False,
                )
                downloaded_paths.append(local_path)
                progress.update(task_id, completed=True, description=f"✓ {Path(file_path).name}")
                
            except Exception as e:
                console.print(f"[red]Error downloading {file_path}: {e}[/red]")
                progress.update(task_id, completed=True, description=f"✗ {Path(file_path).name}")
                return False
    
    console.print(f"\n[bold green]✓ Successfully downloaded {len(downloaded_paths)} files![/bold green]\n")
    console.print("[bold]Downloaded files:[/bold]")
    for path in downloaded_paths:
        console.print(f"  • {path}")
    
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
  python download_checkpoints.py --model transfer2.5-2b-depth --cache-dir ./checkpoints
  
  # Download all models
  python download_checkpoints.py --all
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
        help="Cache directory for downloaded models (default: ~/.cache/huggingface/hub)"
    )
    
    args = parser.parse_args()
    
    # Display header
    console.print("\n[bold magenta]═══════════════════════════════════════[/bold magenta]")
    console.print("[bold magenta]   Cosmos Model Checkpoint Downloader[/bold magenta]")
    console.print("[bold magenta]═══════════════════════════════════════[/bold magenta]\n")
    
    if args.list:
        list_available_models()
    elif args.all:
        download_all_models(args.cache_dir)
    elif args.model:
        download_model(args.model, args.cache_dir)
    else:
        parser.print_help()
        console.print("\n[yellow]Tip: Use --list to see available models[/yellow]")


if __name__ == "__main__":
    main()
