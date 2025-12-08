#!/usr/bin/env python3
"""
Cosmos-Transfer2.5 Inference Pipeline with Rich Progress

Similar to Predict2Pipeline but for controllable video generation with depth/edge/seg control.
"""

import sys
from pathlib import Path
from typing import Optional, Union, List

import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()

# Setup PYTHONPATH
sys.path.insert(0, str(Path("/workspace/cosmos-transfer2.5")))

from cosmos_transfer2._src.transfer2.inference.inference_pipeline import ControlVideo2WorldInference
from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.experiment.experiment_list import EXPERIMENTS


class Transfer2Pipeline:
    """
    User-friendly wrapper for Cosmos-Transfer2.5 inference.
    
    Supports:
    - Depth control
    - Edge control  
    - Segmentation control
    - Multi-control (combining multiple)
    
    Example:
        ```python
        from cosmos_prefer2.inference import Transfer2Pipeline
        
        # Initialize pipeline
        pipeline = Transfer2Pipeline(
            checkpoint_path="/workspace/checkpoints/transfer2-depth.pt",
            control_type="depth"
        )
        
        # Generate video from depth map
        video = pipeline.generate(
            prompt="A car driving on a highway",
            control_video="depth_maps.mp4",
            guidance=7.5,
            num_steps=50,
        )
        ```
    """
    
    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        control_type: str = "depth",  # "depth", "edge", "seg", or "multicontrol"
        experiment: Optional[str] = None,
        device: str = "cuda",
        verbose: bool = True,
    ):
        """
        Initialize Transfer2 pipeline.
        
        Args:
            checkpoint_path: Path to checkpoint
            control_type: Type of control ("depth", "edge", "seg", "multicontrol")
            experiment: Experiment name (auto-selected if None)
            device: Device to use
            verbose: Print progress
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.control_type = control_type
        self.device = device
        self.verbose = verbose
        
        if experiment is None:
            # Auto-select experiment based on control type
            experiment = f"{control_type}_720p_t24"
        
        self.experiment = experiment
        
        if verbose:
            console.print(f"\n[bold cyan]Cosmos-Transfer2.5 Pipeline ({control_type})[/bold cyan]\n")
        
        # Initialize pipeline
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                     console=console, disable=not verbose) as progress:
            task = progress.add_task("Loading model...", total=None)
            
            torch.enable_grad(False)
            
            self.pipe = ControlVideo2WorldInference(
                registered_exp_name=EXPERIMENTS[experiment].registered_exp_name,
                checkpoint_paths=[str(self.checkpoint_path)],
                s3_credential_path="",
                exp_override_opts=EXPERIMENTS[experiment].command_args,
            )
            
            progress.update(task, completed=True, description="✓ Model loaded")
        
        if verbose:
            console.print("[green]✓ Pipeline ready![/green]\n")
    
    def generate(
        self,
        prompt: str,
        control_video: Union[str, Path],
        num_frames: int = 24,
        resolution: tuple[int, int] = (704, 1280),
        guidance: float = 7.5,
        num_steps: int = 50,
        seed: Optional[int] = None,
        negative_prompt: str = "",
    ) -> torch.Tensor:
        """
        Generate video with control signal.
        
        Args:
            prompt: Text description
            control_video: Path to control signal video (depth/edge/seg map)
            num_frames: Number of frames
            resolution: Output resolution
            guidance: Guidance scale
            num_steps: Denoising steps
            seed: Random seed
            negative_prompt: Negative prompt
        
        Returns:
            Generated video tensor
        """
        if self.verbose:
            console.print("\n[bold cyan]Generation Parameters:[/bold cyan]")
            console.print(f"  • [bold]Prompt:[/bold] {prompt}")
            console.print(f"  • [bold]Control:[/bold] {self.control_type}")
            console.print(f"  • Frames: {num_frames} | Steps: {num_steps}")
            console.print()
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                     BarColumn(), console=console, disable=not self.verbose) as progress:
            task = progress.add_task("Generating...", total=num_steps)
            
            video = self.pipe.generate_control2world(
                prompt=prompt,
                control_path=str(control_video),
                guidance=guidance,
                num_video_frames=num_frames,
                resolution=resolution,
                seed=seed,
                negative_prompt=negative_prompt,
                num_steps=num_steps,
            )
            
            progress.update(task, completed=num_steps)
        
        if self.verbose:
            console.print("[green]✓ Generation complete![/green]\n")
        
        return video
    
    def save_video(self, video: torch.Tensor, output_path: Union[str, Path], fps: int = 24):
        """Save video to file."""
        from cosmos_transfer2._src.imaginaire.visualize.video import save_img_or_video
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        video = (1.0 + video[0]) / 2.0
        save_img_or_video(video, str(output_path), fps=fps)
        
        if self.verbose:
            console.print(f"[green]✓ Saved to {output_path}[/green]\n")

