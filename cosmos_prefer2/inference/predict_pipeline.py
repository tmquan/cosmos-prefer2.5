#!/usr/bin/env python3
"""
Cosmos-Predict2.5 Inference Pipeline

This module provides a user-friendly interface for running inference with Cosmos-Predict2.5 models.
Features:
- Simple, intuitive API
- Comprehensive documentation
- Step-by-step visualization
- Checkpoint management
"""

import sys
from pathlib import Path
from typing import Optional, Union

import torch
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)

console = Console()


class Predict2Pipeline:
    """
    User-friendly wrapper for Cosmos-Predict2.5 inference.
    
    This class provides a simple interface for text-to-video and video-to-video generation
    with Cosmos-Predict2.5 models.
    
    Example:
        ```python
        from cosmos_prefer2.inference.predict_pipeline import Predict2Pipeline
        
        # Initialize pipeline
        pipeline = Predict2Pipeline(
            checkpoint_path="/workspace/checkpoints/predict2-posttrained.pt",
            experiment="Stage-c_pt_4-Index-2-Size-2B-Res-720-Fps-16-Note-rf_with_edm_ckpt"
        )
        
        # Generate video
        video = pipeline.generate(
            prompt="A robot arm picking up a red cube",
            num_frames=121,
            guidance=7.5,
            num_steps=50,
        )
        
        # Save output
        pipeline.save_video(video, "output.mp4")
        ```
    """
    
    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        experiment: str = "Stage-c_pt_4-Index-2-Size-2B-Res-720-Fps-16-Note-rf_with_edm_ckpt",
        config_file: Optional[Union[str, Path]] = None,
        context_parallel_size: int = 1,
        offload_diffusion_model: bool = False,
        offload_text_encoder: bool = False,
        offload_tokenizer: bool = False,
        device: str = "cuda",
        verbose: bool = True,
    ):
        """
        Initialize the Predict2.5 inference pipeline.
        
        Args:
            checkpoint_path: Path to model checkpoint
            experiment: Experiment configuration name
            config_file: Optional path to config file (auto-detected if None)
            context_parallel_size: Number of GPUs for context parallelism
            offload_diffusion_model: Offload diffusion model to CPU when not in use
            offload_text_encoder: Offload text encoder to CPU
            offload_tokenizer: Offload tokenizer to CPU
            device: Device to run on (default: "cuda")
            verbose: Print detailed progress
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.experiment = experiment
        self.device = device
        self.verbose = verbose
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        if verbose:
            self._print_header()
            self._print_config(locals())
        
        # Initialize the pipeline with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            disable=not verbose
        ) as progress:
            task = progress.add_task("Initializing pipeline...", total=4)
            
            # Disable gradients
            torch.enable_grad(False)
            progress.update(task, advance=1, description="Setting up environment...")
            
            # Initialize inference pipeline
            try:
                # Import here to avoid loading unless needed
                sys.path.insert(0, str(Path("/workspace/cosmos-predict2.5")))
                from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
                
                # Build kwargs for Video2WorldInference
                inference_kwargs = {
                    "experiment_name": experiment,
                    "ckpt_path": str(self.checkpoint_path),
                    "s3_credential_path": "",
                    "context_parallel_size": context_parallel_size,
                    "offload_diffusion_model": offload_diffusion_model,
                    "offload_text_encoder": offload_text_encoder,
                    "offload_tokenizer": offload_tokenizer,
                }
                
                # Only add config_file if provided
                if config_file:
                    inference_kwargs["config_file"] = str(config_file)
                
                self.pipe = Video2WorldInference(**inference_kwargs)
                progress.update(task, advance=1, description="Loading model...")
            except Exception as e:
                console.print(f"[bold red]✗ Error initializing pipeline:[/bold red] {e}")
                raise
            
            progress.update(task, advance=1, description="Loading text encoder...")
            progress.update(task, advance=1, description="✓ Pipeline ready!")
        
        if verbose:
            console.print("\n[bold green]✓ Pipeline initialized successfully![/bold green]\n")
            self._print_model_info()
    
    def generate(
        self,
        prompt: str,
        input_video: Optional[Union[str, Path]] = None,
        num_frames: int = 121,
        resolution: tuple[int, int] = (704, 1280),
        guidance: float = 7.5,
        num_steps: int = 50,
        seed: Optional[int] = None,
        negative_prompt: str = "",
        num_input_frames: int = 1,
    ) -> torch.Tensor:
        """
        Generate a video from text prompt.
        
        Args:
            prompt: Text description of desired video
            input_video: Optional path to input video for video2video
            num_frames: Number of frames to generate
            resolution: Output resolution (height, width)
            guidance: Classifier-free guidance scale (higher = more prompt adherence)
            num_steps: Number of denoising steps (more = better quality, slower)
            seed: Random seed for reproducibility
            negative_prompt: What to avoid in generation
            num_input_frames: Number of frames to condition on
        
        Returns:
            Generated video tensor (1, C, T, H, W) in range [-1, 1]
        
        Example:
            >>> video = pipeline.generate(
            ...     prompt="A robot picks up a cube",
            ...     num_frames=121,
            ...     guidance=7.5,
            ...     seed=42
            ... )
        """
        if self.verbose:
            console.print("\n[bold cyan]Generation Parameters:[/bold cyan]")
            console.print(f"  • [bold]Prompt:[/bold] {prompt}")
            if input_video:
                console.print(f"  • [bold]Input:[/bold] {Path(input_video).name}")
            console.print(f"  • Frames: {num_frames} | Resolution: {resolution[0]}x{resolution[1]}")
            console.print(f"  • Steps: {num_steps} | Guidance: {guidance}")
            console.print()
        
        # Format resolution as string
        resolution_str = f"{resolution[0]},{resolution[1]}"
        
        # Setup progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
            disable=not self.verbose
        ) as progress:
            
            # Text encoding
            text_task = progress.add_task("Encoding text prompt...", total=1)
            progress.update(text_task, advance=1)
            
            # Video generation
            gen_task = progress.add_task(f"Generating video ({num_steps} steps)...", total=num_steps)
            
            try:
                video = self.pipe.generate_vid2world(
                    prompt=prompt,
                    input_path=str(input_video) if input_video else None,
                    guidance=guidance,
                    num_video_frames=num_frames,
                    num_latent_conditional_frames=num_input_frames,
                    resolution=resolution_str,
                    seed=seed,
                    negative_prompt=negative_prompt,
                    num_steps=num_steps,
                )
                
                progress.update(gen_task, completed=num_steps)
                
            except Exception as e:
                console.print(f"[bold red]✗ Error during generation:[/bold red] {e}")
                raise
        
        if self.verbose:
            console.print(f"\n[bold green]✓ Video generated successfully![/bold green]")
            console.print(f"[dim]Output shape: {video.shape}[/dim]\n")
        
        return video
    
    def save_video(
        self,
        video: torch.Tensor,
        output_path: Union[str, Path],
        fps: int = 24,
    ) -> None:
        """
        Save generated video to file.
        
        Args:
            video: Video tensor (1, C, T, H, W) in range [-1, 1]
            output_path: Path to save video
            fps: Frames per second
        
        Example:
            >>> pipeline.save_video(video, "output.mp4", fps=24)
        """
        from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert from [-1, 1] to [0, 1]
        video = (1.0 + video[0]) / 2.0
        
        if self.verbose:
            console.print(f"[cyan]Saving video to:[/cyan] {output_path}")
        
        save_img_or_video(video, str(output_path), fps=fps)
        
        if self.verbose:
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            console.print(f"[green]✓ Video saved ({file_size_mb:.1f}MB)[/green]\n")
    
    def _print_header(self):
        """Print welcome header."""
        console.print("\n" + "="*60)
        console.print("[bold cyan]Cosmos-Predict2.5 Inference Pipeline[/bold cyan]")
        console.print("="*60 + "\n")
    
    def _print_config(self, config: dict):
        """Print configuration."""
        console.print("[bold cyan]Configuration:[/bold cyan]")
        
        important_keys = ['checkpoint_path', 'experiment', 'context_parallel_size', 'device']
        for key in important_keys:
            if key in config and key != 'self':
                console.print(f"  • {key}: [yellow]{config[key]}[/yellow]")
        console.print()
    
    def _print_model_info(self):
        """Print model information."""
        try:
            console.print("[bold cyan]Model Information:[/bold cyan]")
            console.print(f"  • Experiment: {self.experiment}")
            console.print(f"  • Checkpoint: {self.checkpoint_path.name}")
            console.print(f"  • Device: {self.device}")
            console.print()
        except Exception:
            pass


if __name__ == "__main__":
    # Example usage
    console.print("[bold]Cosmos-Predict2.5 Pipeline - Example Usage[/bold]\n")
    
    example_code = '''
from cosmos_prefer2.inference.predict_pipeline import Predict2Pipeline

# Initialize
pipeline = Predict2Pipeline(
    checkpoint_path="/workspace/checkpoints/predict2-posttrained.pt"
)

# Generate
video = pipeline.generate(
    prompt="A robot arm picking up a red cube on a wooden table",
    num_frames=121,
    num_steps=50,
    guidance=7.5,
    seed=42
)

# Save
pipeline.save_video(video, "output.mp4")
'''
    
    console.print("[bold]Usage Example:[/bold]\n")
    console.print(example_code)
