#!/usr/bin/env python3
"""
Example: Text-to-Video Generation with Cosmos-Predict2.5

This demonstrates the simplest way to generate videos from text prompts.
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
from cosmos_prefer2.inference.predict_helpers import (
    setup_inference_environment,
    build_inference_pipeline,
    yield_video,
)
from rich.console import Console

console = Console()


def main():
    console.print("[bold cyan]  Cosmos-Predict2.5: Text-to-Video Generation[/bold cyan]\n")
    
    # Step 1: Setup environment
    console.print("[bold]Step 1:[/bold] Setting up environment...\n")
    setup_pythonpath()
    status = check_environment()
    
    if not all([status.get("torch"), status.get("cuda")]):
        console.print("[bold red]✗ Error: PyTorch with CUDA is required![/bold red]")
        return
    
    # Step 2: Check HF cache
    console.print("\n[bold]Step 2:[/bold] Verifying checkpoint cache...\n")
    
    # Check if checkpoints are available in HF cache
    checkpoint_path = check_downloads(
        model_type="Predict2.5",
        variant="base",
        preferred_training="post-trained"
    )
    
    if checkpoint_path:
        console.print(f"[green]✓ Found checkpoint in cache:[/green] [dim]{checkpoint_path}[/dim]\n")
    else:
        console.print("[yellow]No checkpoint found in cache. Will download from HuggingFace...[/yellow]")
        console.print("[dim]To avoid downloads, run:[/dim]")
        console.print("  [cyan]python download_checkpoints.py --model predict2.5-2b-posttrained[/cyan]\n")
    
    # Step 3: Initialize pipeline
    console.print("[bold]Step 3:[/bold] Initializing pipeline...\n")
    
    try:
        # Setup environment (changes directory to cosmos-predict2.5)
        original_cwd, cosmos_dir = setup_inference_environment()
        console.print(f"[dim]Working directory: {cosmos_dir}[/dim]")
        
        # Create pipeline using checkpoint database (respects HF_HOME, no duplicate downloads!)
        # Don't pass checkpoint_path - let it use the checkpoint database
        pipe = build_inference_pipeline(
            checkpoint_path=None,  # Use checkpoint database
            model_name="nvidia/Cosmos-Predict2.5-2B",  # HuggingFace repo ID
            experiment=None,  # Auto-detect from model_name
            context_parallel_size=1,
            offload_diffusion_model=False,
            offload_text_encoder=False,
            offload_tokenizer=False,
        )
        
        console.print("[green]✓ Pipeline initialized successfully![/green]\n")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error initializing pipeline:[/bold red] {e}\n")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Select example or custom prompt
    console.print("[bold]Step 4:[/bold] Select generation mode...\n")
    
    # Check for downloaded examples
    inputs_dir = original_cwd / "data" / "inputs" / "predict2.5"
    has_examples = inputs_dir.exists()
    
    examples = []
    if has_examples:
        examples = [
            ("text2world", "Text-to-video (custom prompt)", None),
            ("image2world", "Image-to-video (bus terminal example)", inputs_dir / "image2world"),
            ("video2world", "Video-to-video (robot arm example)", inputs_dir / "video2world"),
        ]
    else:
        examples = [("text2world", "Text-to-video (custom prompt)", None)]
    
    console.print("[cyan]Choose generation mode:[/cyan]")
    for i, (mode, desc, path) in enumerate(examples, 1):
        console.print(f"  [yellow]{i}.[/yellow] {desc}")
    
    console.print()
    choice = input(f"Enter choice (1-{len(examples)}): ").strip() or "1"
    
    mode_idx = int(choice) - 1 if choice.isdigit() and 1 <= int(choice) <= len(examples) else 0
    mode, _, example_path = examples[mode_idx]
    
    # Get prompt and input
    prompt = None
    input_video = None
    
    if mode == "text2world":
        console.print("\n[cyan]Sample prompts:[/cyan]")
        sample_prompts = [
            "A robotic arm slowly picks up a red cube from a wooden table in a well-lit laboratory",
            "A sleek sports car drives along a winding mountain road at golden hour sunset",
            "A person walks through a futuristic neon-lit city street at night with reflections on wet pavement",
        ]
        for i, p in enumerate(sample_prompts, 1):
            console.print(f"  {i}. {p}")
        console.print()
        
        prompt_choice = input("Enter choice (1-3) or type your own prompt: ").strip()
        if prompt_choice.isdigit() and 1 <= int(prompt_choice) <= 3:
            prompt = sample_prompts[int(prompt_choice) - 1]
        else:
            prompt = prompt_choice if prompt_choice else sample_prompts[0]
    
    elif mode in ["image2world", "video2world"]:
        # Load prompt from example
        prompt_file = example_path / "prompt.txt"
        prompt = prompt_file.read_text().strip()
        console.print(f"\n[green]Loaded example prompt from:[/green] {prompt_file.name}")
        
        # For image2world/video2world, we need input
        if mode == "video2world":
            input_video = example_path / "input_video.mp4"
            console.print(f"[green]Using input video:[/green] {input_video.name}")
        else:
            input_video = example_path / "input_image.png"
            console.print(f"[green]Using input image:[/green] {input_video.name}")
    
    console.print(f"\n[green]Prompt:[/green] [italic]{prompt[:100]}...[/italic]\n")
    
    # Step 5: Generate video
    console.print(f"[bold]Step 5:[/bold] Generating video...\n")
    
    # Generation parameters
    num_frames = 121
    resolution = (704, 1280)
    guidance = 7.5
    num_steps = 50
    seed = 42
    
    console.print("[dim]Parameters:[/dim]")
    console.print(f"  • Mode: {mode}")
    console.print(f"  • Frames: {num_frames}")
    console.print(f"  • Resolution: {resolution[0]}x{resolution[1]}")
    console.print(f"  • Guidance: {guidance}")
    console.print(f"  • Steps: {num_steps}")
    console.print(f"  • Seed: {seed}")
    if input_video:
        console.print(f"  • Input: {input_video.name}")
    console.print()
    
    try:
        console.print("[yellow]Generating... (this may take several minutes)[/yellow]\n")
        
        # Generate video using helper function
        video = yield_video(
            pipe=pipe,
            prompt=prompt,
            num_frames=num_frames,
            resolution=resolution,
            guidance=guidance,
            num_steps=num_steps,
            seed=seed,
            input_video=input_video,
        )
        
        console.print("[green]✓ Video generated successfully![/green]\n")
        
        # Restore original directory
        os.chdir(original_cwd)
        
    except Exception as e:
        os.chdir(original_cwd)  # Restore directory
        console.print(f"[bold red]✗ Error during generation:[/bold red] {e}\n")
        import traceback
        traceback.print_exc()
        return
    
    # Step 6: Save result
    console.print("[bold]Step 6:[/bold] Saving video...\n")
    
    output_dir = original_cwd / "data" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"predict_{mode}_{seed}.mp4"
    
    try:
        write_video(video, output_path, fps=24)
        
        console.print(f"[bold green]✓ Complete![/bold green]")
        console.print(f"[cyan]Output saved to:[/cyan] {output_path.absolute()}\n")
        
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

