#!/usr/bin/env python3
"""
Cosmos-Transfer2.5 Inference Pipeline

This module provides a user-friendly interface for running controllable video generation
with Cosmos-Transfer2.5 models using depth, edge, or segmentation control.
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


class Transfer2Pipeline:
    """
    User-friendly wrapper for Cosmos-Transfer2.5 inference.
    
    This class provides a simple interface for controllable video generation
    using depth maps, edge maps, or segmentation maps.
    
    Supported control types:
    - depth: Depth maps for structure preservation
    - edge: Edge maps for sharp boundaries
    - seg: Segmentation maps for semantic control
    - multiview: Multiple camera views
    
    Example:
        ```python
        from cosmos_prefer2.inference.transfer_pipeline import Transfer2Pipeline
        
        # Initialize pipeline
        pipeline = Transfer2Pipeline(
            checkpoint_path="/workspace/checkpoints/transfer2-depth.pt",
            control_type="depth"
        )
        
        # Generate video from depth map
        video = pipeline.generate(
            prompt="A car driving on a highway with dramatic sunset lighting",
            control_video="depth_maps.mp4",
            num_frames=93,
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
        control_type: str = "depth",
        experiment: Optional[str] = None,
        device: str = "cuda",
        verbose: bool = True,
    ):
        """
        Initialize Transfer2.5 pipeline.
        
        Args:
            checkpoint_path: Path to checkpoint
            control_type: Type of control ("depth", "edge", "seg", "multiview")
            experiment: Experiment name (auto-selected if None)
            device: Device to use (default: "cuda")
            verbose: Print progress messages
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.control_type = control_type
        self.device = device
        self.verbose = verbose
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        # Auto-select experiment based on control type
        if experiment is None:
            # Default experiment for Transfer2.5 general models
            experiment = "vid2vid_2B_control_720p_t24_control_layer4_cr1pt1_embedding_rectified_flow"
        
        self.experiment = experiment
        
        if verbose:
            self._print_header()
            self._print_config(locals())
        
        # Initialize pipeline
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            disable=not verbose
        ) as progress:
            task = progress.add_task("Loading model...", total=3)
            
            try:
                # Import here to avoid loading unless needed
                sys.path.insert(0, str(Path("/workspace/cosmos-transfer2.5")))
                from cosmos_transfer2._src.transfer2.inference.inference_pipeline import ControlVideo2WorldInference
                
                torch.enable_grad(False)
                progress.update(task, advance=1, description="Initializing...")
                
                self.pipe = ControlVideo2WorldInference(
                    registered_exp_name=experiment,
                    checkpoint_paths=str(self.checkpoint_path),
                    s3_credential_path="",
                    process_group=None,
                    cache_dir=None,
                )
                
                progress.update(task, advance=1, description="Loading weights...")
                progress.update(task, advance=1, description="✓ Model loaded")
            except Exception as e:
                console.print(f"[bold red]✗ Error loading model:[/bold red] {e}")
                raise
        
        if verbose:
            console.print("\n[bold green]✓ Pipeline ready![/bold green]\n")
            self._print_model_info()
    
    def generate(
        self,
        prompt: str,
        control_video: Union[str, Path],
        num_frames: int = 93,
        resolution: tuple[int, int] = (704, 1280),
        guidance: float = 7.5,
        num_steps: int = 50,
        seed: Optional[int] = None,
        negative_prompt: str = "",
    ) -> torch.Tensor:
        """
        Generate video with control signal.
        
        Args:
            prompt: Text description of desired output
            control_video: Path to control signal video (depth/edge/seg map)
            num_frames: Number of frames per chunk (default=93, matching state_t=24)
            resolution: Output resolution (height, width)
            guidance: Guidance scale (higher = more prompt adherence)
            num_steps: Denoising steps (more = better quality, slower)
            seed: Random seed for reproducibility
            negative_prompt: What to avoid in generation
        
        Returns:
            Generated video tensor (B, C, T, H, W) in range [-1, 1]
        
        Example:
            >>> video = pipeline.generate(
            ...     prompt="A cinematic urban scene at night",
            ...     control_video="edge_maps.mp4",
            ...     num_frames=93,
            ...     guidance=7.5,
            ...     seed=42
            ... )
        
        Note:
            The default num_frames=93 corresponds to state_t=24 in the Transfer2.5 model.
            If your control video has more frames, it will be processed in chunks.
        """
        if self.verbose:
            console.print("\n[bold cyan]Generation Parameters:[/bold cyan]")
            console.print(f"  • [bold]Prompt:[/bold] {prompt}")
            console.print(f"  • [bold]Control:[/bold] {self.control_type} ({Path(control_video).name})")
            console.print(f"  • Frames: {num_frames} | Resolution: {resolution[0]}x{resolution[1]}")
            console.print(f"  • Steps: {num_steps} | Guidance: {guidance}")
            if seed is not None:
                console.print(f"  • Seed: {seed}")
            console.print()
        
        # Map resolution tuple to resolution string
        resolution_map = {
            (704, 1280): "720",
            (480, 854): "480",
            (1080, 1920): "1080",
        }
        resolution_str = resolution_map.get(resolution, "720")
        
        # Build control video paths dictionary
        input_control_video_paths = {
            self.control_type: str(control_video)
        }
        
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
                # Call the inference pipeline
                result = self.pipe.generate_img2world(
                    prompt=prompt,
                    video_path=str(control_video),
                    guidance=int(guidance),
                    seed=seed if seed is not None else 1,
                    resolution=resolution_str,
                    num_steps=num_steps,
                    num_video_frames_per_chunk=num_frames,
                    hint_key=[self.control_type],
                    input_control_video_paths=input_control_video_paths,
                    negative_prompt=negative_prompt if negative_prompt else None,
                )
                
                # Extract video from result tuple
                video = result[0]
                
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
            video: Video tensor (B, C, T, H, W) in range [-1, 1]
            output_path: Path to save video
            fps: Frames per second
        
        Example:
            >>> pipeline.save_video(video, "output.mp4", fps=24)
        """
        from cosmos_transfer2._src.imaginaire.visualize.video import save_img_or_video
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert from [-1, 1] to [0, 1]
        # Handle both (B, C, T, H, W) and (C, T, H, W) formats
        if video.dim() == 5:
            video = (1.0 + video[0]) / 2.0
        else:
            video = (1.0 + video) / 2.0
        
        if self.verbose:
            console.print(f"[cyan]Saving video to:[/cyan] {output_path}")
        
        save_img_or_video(video, str(output_path), fps=fps)
        
        if self.verbose:
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            console.print(f"[green]✓ Video saved ({file_size_mb:.1f}MB)[/green]\n")
    
    def _print_header(self):
        """Print welcome header."""
        console.print("\n" + "="*60)
        console.print(f"[bold cyan]Cosmos-Transfer2.5 Inference Pipeline ({self.control_type})[/bold cyan]")
        console.print("="*60 + "\n")
    
    def _print_config(self, config: dict):
        """Print configuration."""
        console.print("[bold cyan]Configuration:[/bold cyan]")
        
        important_keys = ['checkpoint_path', 'control_type', 'experiment', 'device']
        for key in important_keys:
            if key in config and key != 'self':
                console.print(f"  • {key}: [yellow]{config[key]}[/yellow]")
        console.print()
    
    def _print_model_info(self):
        """Print model information."""
        try:
            console.print("[bold cyan]Model Information:[/bold cyan]")
            console.print(f"  • Experiment: {self.experiment}")
            console.print(f"  • Control type: {self.control_type}")
            console.print(f"  • Checkpoint: {self.checkpoint_path.name}")
            console.print(f"  • Device: {self.device}")
            console.print()
        except Exception:
            pass


if __name__ == "__main__":
    # Example usage
    console.print("[bold]Cosmos-Transfer2.5 Pipeline - Example Usage[/bold]\n")
    
    example_code = '''
from cosmos_prefer2.inference.transfer_pipeline import Transfer2Pipeline

# Initialize with depth control
pipeline = Transfer2Pipeline(
    checkpoint_path="/workspace/checkpoints/transfer2-depth.pt",
    control_type="depth"
)

# Generate video from depth map
video = pipeline.generate(
    prompt="A car driving on a highway with dramatic sunset lighting",
    control_video="depth_maps.mp4",
    num_frames=93,
    num_steps=50,
    guidance=7.5,
    seed=42
)

# Save
pipeline.save_video(video, "output.mp4")
'''
    
    console.print("[bold]Usage Example:[/bold]\n")
    console.print(example_code)
