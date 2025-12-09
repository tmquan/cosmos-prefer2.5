#!/usr/bin/env python3
"""
Example: Video-to-Video Generation with Cosmos-Transfer2.5

This demonstrates controlled video generation using depth, edge, or segmentation maps.
IMPORTANT: HF_HOME must be set before importing cosmos modules!
"""

import os
from pathlib import Path

# CRITICAL: Set HF_HOME BEFORE any cosmos imports to avoid duplicate downloads
if not os.environ.get('HF_HOME'):
    hf_cache = Path("/workspace/checkpoints") if Path("/workspace").exists() else Path("./checkpoints")
    os.environ['HF_HOME'] = str(hf_cache)
    print(f"Set HF_HOME to: {hf_cache}")

import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cosmos_prefer2.utils import setup_pythonpath, check_environment, check_downloads, write_video
from cosmos_prefer2.inference.transfer_helpers import (
    setup_transfer_environment,
    build_transfer_pipeline,
    yield_video,
)
from rich.console import Console

console = Console()


def main():
    console.print("[bold cyan]🎬 Cosmos-Transfer2.5: Video-to-Video Generation[/bold cyan]\n")
    
    # Step 1: Setup environment
    console.print("[bold]Step 1:[/bold] Setting up environment...\n")
    setup_pythonpath()
    status = check_environment()
    
    if not all([status.get("torch"), status.get("cuda")]):
        console.print("[bold red]✗ Error: PyTorch with CUDA is required![/bold red]")
        return
    
    # Step 2: Select control type
    console.print("\n[bold]Step 2:[/bold] Select control type...\n")
    
    control_types = {
        "1": ("depth", "Depth maps (structure preservation)"),
        "2": ("edge", "Edge maps (sharp boundaries)"),
        "3": ("seg", "Segmentation maps (semantic control)"),
    }
    
    console.print("[cyan]Available control types:[/cyan]")
    for key, (name, desc) in control_types.items():
        console.print(f"  [yellow]{key}.[/yellow] {name.capitalize()} - {desc}")
    
    console.print()
    choice = input("Enter choice (1-3, default=1): ").strip() or "1"
    control_type, _ = control_types.get(choice, control_types["1"])
    
    console.print(f"\n[green]Using control type:[/green] [bold]{control_type}[/bold]\n")
    
    # Step 3: Check for checkpoint
    console.print("[bold]Step 3:[/bold] Verifying checkpoint cache...\n")
    
    checkpoint_path = check_downloads(
        model_type="Transfer2.5",
        variant=f"general/{control_type}",
        preferred_training="post-trained"
    )
    
    if checkpoint_path:
        console.print(f"[green]✓ Found checkpoint in cache:[/green] [dim]{checkpoint_path}[/dim]\n")
    else:
        console.print("[yellow]No checkpoint found in cache. Will download from HuggingFace...[/yellow]")
        console.print("[dim]To avoid long waits, pre-download with:[/dim]")
        console.print(f"  [cyan]python download_checkpoints.py --model transfer2.5-2b-{control_type}[/cyan]\n")
    
    # Step 4: Check for input video
    console.print("[bold]Step 4:[/bold] Preparing input...\n")
    
    # Check for downloaded examples
    inputs_dir = Path.cwd() / "data" / "inputs" / "transfer2.5"
    example_dirs = []
    
    if inputs_dir.exists():
        example_dirs = [
            ("sim2real", "Sim2Real: Kitchen robot arm", inputs_dir / "sim2real"),
            ("real2real", "Real2Real: Snow storm driving", inputs_dir / "real2real"),
        ]
    
    # Also check /workspace/datasets
    dataset_videos = []
    if Path("/workspace/datasets").exists():
        dataset_videos = list(Path("/workspace/datasets").glob(f"**/*{control_type}*.mp4"))
    
    if example_dirs:
        console.print("[cyan]Available examples:[/cyan]")
        for i, (name, desc, path) in enumerate(example_dirs, 1):
            console.print(f"  [yellow]{i}.[/yellow] {desc}")
        if dataset_videos:
            console.print(f"  [yellow]{len(example_dirs)+1}.[/yellow] Custom from /workspace/datasets/")
        console.print()
        
        choice = input(f"Enter choice (1-{len(example_dirs) + (1 if dataset_videos else 0)}): ").strip() or "1"
        choice_idx = int(choice) - 1 if choice.isdigit() else 0
        
        if choice_idx < len(example_dirs):
            # Use example
            example_name, _, example_path = example_dirs[choice_idx]
            
            # Load prompt
            prompt_file = example_path / "prompt.txt"
            prompt = prompt_file.read_text().strip()
            console.print(f"\n[green]Loaded prompt:[/green] [italic]{prompt[:80]}...[/italic]")
            
            # Find control video matching control_type
            control_video = example_path / f"control_{control_type}.mp4"
            if not control_video.exists():
                # Try input_video.mp4 if control not found
                control_video = example_path / "input_video.mp4"
            
            console.print(f"[green]Using control video:[/green] {control_video.name}\n")
            use_custom_prompt = False
        else:
            # Use custom from datasets
            console.print(f"[cyan]Found {len(dataset_videos)} video(s) in /workspace/datasets/:[/cyan]")
            for i, vid in enumerate(dataset_videos[:5], 1):
                console.print(f"  {i}. {vid.name}")
            console.print()
            
            vid_choice = input(f"Select video (1-{min(5, len(dataset_videos))}): ").strip() or "1"
            vid_idx = int(vid_choice) - 1 if vid_choice.isdigit() else 0
            control_video = dataset_videos[vid_idx]
            console.print(f"\n[green]Using:[/green] {control_video.name}\n")
            use_custom_prompt = True
    
    elif dataset_videos:
        console.print(f"[cyan]Found {len(dataset_videos)} video(s) with {control_type}:[/cyan]")
        for i, vid in enumerate(dataset_videos[:5], 1):
            console.print(f"  {i}. {vid.name}")
        console.print()
        
        choice = input(f"Select video (1-{min(5, len(dataset_videos))}): ").strip() or "1"
        choice_idx = int(choice) - 1 if choice.isdigit() else 0
        control_video = dataset_videos[choice_idx]
        console.print(f"\n[green]Using:[/green] {control_video.name}\n")
        use_custom_prompt = True
    
    else:
        console.print(f"[yellow]No example or dataset videos found.[/yellow]")
        console.print("[dim]Please provide path to control video:[/dim]")
        path = input(f"Path to {control_type} map video: ").strip()
        
        if not path or not Path(path).exists():
            console.print(f"[red]✗ Invalid path. Please prepare a {control_type} map video first.[/red]\n")
            console.print(f"[dim]Run: python download_example_data.py to get examples[/dim]")
            return
        
        control_video = Path(path)
        console.print(f"\n[green]Using:[/green] {control_video.name}\n")
        use_custom_prompt = True
    
    # Step 5: Initialize pipeline
    console.print("[bold]Step 5:[/bold] Initializing Transfer2.5 pipeline...\n")
    
    try:
        # Setup Transfer2.5 environment
        original_cwd, transfer_dir = setup_transfer_environment()
        console.print(f"[dim]Working directory: {transfer_dir}[/dim]")
        
        # Create Transfer pipeline
        pipe = build_transfer_pipeline(
            checkpoint_path=checkpoint_path,
            control_type=control_type,
        )
        
        console.print("[green]✓ Pipeline initialized successfully![/green]\n")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error initializing pipeline:[/bold red] {e}\n")
        import traceback
        traceback.print_exc()
        return
    
    # Step 6: Get prompt
    console.print("[bold]Step 6:[/bold] Configure prompt...\n")
    
    if not use_custom_prompt:
        # Already loaded from example
        console.print(f"[green]Using example prompt:[/green] [italic]{prompt[:80]}...[/italic]")
        console.print()
        use_default = input("Press Enter to use example prompt, or type new prompt: ").strip()
        if use_default:
            prompt = use_default
            console.print(f"\n[green]Updated prompt:[/green] [italic]{prompt[:80]}...[/italic]\n")
    else:
        # Need to get prompt from user
        default_prompts = {
            "depth": "A cinematic scene with dramatic lighting and realistic textures",
            "edge": "A stylized artistic scene with bold lines and vivid colors",
            "seg": "A photorealistic scene with accurate materials and lighting",
        }
        
        default_prompt = default_prompts[control_type]
        console.print(f"[dim]Default:[/dim] {default_prompt}\n")
        
        prompt = input("Enter prompt (or press Enter for default): ").strip()
        prompt = prompt if prompt else default_prompt
        console.print(f"\n[green]Using prompt:[/green] [italic]{prompt}[/italic]\n")
    
    # Step 7: Generate video
    console.print("[bold]Step 7:[/bold] Generating video...\n")
    
    # Generation parameters
    # Note: num_frames=93 is the default for Transfer2.5 (matches state_t=24)
    # For longer videos, the system will process them in chunks automatically
    num_frames = 93  # Frames per chunk (matches model's state_t=24)
    resolution = (704, 1280)
    guidance = 7.5
    num_steps = 50
    seed = 42
    
    console.print("[dim]Parameters:[/dim]")
    console.print(f"  • Control type: {control_type}")
    console.print(f"  • Frames: {num_frames}")
    console.print(f"  • Resolution: {resolution[0]}x{resolution[1]}")
    console.print(f"  • Guidance: {guidance}")
    console.print(f"  • Steps: {num_steps}")
    console.print(f"  • Seed: {seed}")
    console.print(f"  • Input: {control_video.name}")
    console.print()
    
    try:
        console.print("[yellow]Generating... (this may take several minutes)[/yellow]\n")
        
        video = yield_video(
            pipe=pipe,
            prompt=prompt,
            control_video_path=control_video,
            control_type=control_type,
            num_frames=num_frames,
            resolution=resolution,
            guidance=guidance,
            num_steps=num_steps,
            seed=seed,
        )
        
        console.print("[green]✓ Video generated successfully![/green]\n")
        
        # Restore original directory
        os.chdir(original_cwd)
        
    except Exception as e:
        os.chdir(original_cwd)
        console.print(f"[bold red]✗ Error during generation:[/bold red] {e}\n")
        import traceback
        traceback.print_exc()
        return
    
    # Step 8: Save result
    console.print("[bold]Step 8:[/bold] Saving video...\n")
    
    output_dir = Path.cwd() / "data" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"transfer_{control_type}_{seed}.mp4"
    
    try:
        write_video(video, output_path, fps=24)
        
        console.print(f"[bold green]✓ Complete![/bold green]")
        console.print(f"[cyan]Output saved to:[/cyan] {output_path.absolute()}\n")
        console.print(f"[dim]Control type:[/dim] {control_type}")
        console.print(f"[dim]Input video:[/dim] {control_video.name}\n")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error saving video:[/bold red] {e}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]✗ Interrupted by user[/yellow]\n")
    except Exception as e:
        console.print(f"\n[bold red]✗ Unexpected error:[/bold red] {e}\n")
        import traceback
        traceback.print_exc()

