#!/usr/bin/env python3
"""
I/O utilities for Cosmos-Prefer2.5.

This module provides utilities for checkpoint discovery and video I/O operations.
"""

import os
from pathlib import Path
from typing import Optional
from glob import glob

from rich.console import Console

console = Console()


def check_downloads(
    checkpoint_dir: Path = Path("/workspace/checkpoints"),
    model_type: str = "Predict2.5",
    variant: str = "base",
    preferred_training: str = "post-trained",
) -> Optional[Path]:
    """
    Find a Cosmos checkpoint in the HuggingFace cache structure.
    
    The HuggingFace cache uses the format:
        {HF_HOME}/hub/models--{org}--{repo}/snapshots/{revision}/{filename}
    
    Args:
        checkpoint_dir: Base checkpoint directory (HF_HOME)
        model_type: "Predict2.5" or "Transfer2.5"
        variant: "base", "auto/multiview", "robot/action-cond", etc.
        preferred_training: "post-trained" or "pre-trained"
    
    Returns:
        Path to checkpoint if found, None otherwise
    
    Example:
        >>> from cosmos_prefer2.utils.io import check_downloads
        >>> checkpoint = check_downloads()
        >>> checkpoint = check_downloads(variant="robot/action-cond")
    """
    # HuggingFace stores files in {HF_HOME}/hub/
    hub_dir = checkpoint_dir / "hub"
    
    # Build search patterns for HuggingFace cache structure
    patterns = []
    
    if model_type == "Predict2.5":
        # HF cache structure: hub/models--nvidia--Cosmos-Predict2.5-{SIZE}/snapshots/*/...
        if variant == "base":
            patterns = [
                # Search in HF cache structure (hub subdirectory)
                f"{hub_dir}/models--nvidia--Cosmos-Predict2.5-2B/snapshots/*/base/{preferred_training}/*.pt",
                f"{hub_dir}/models--nvidia--Cosmos-Predict2.5-2B/snapshots/*/base/*/*.pt",
                f"{hub_dir}/models--nvidia--Cosmos-Predict2.5-14B/snapshots/*/base/{preferred_training}/*.pt",
                f"{hub_dir}/models--nvidia--Cosmos-Predict2.5-14B/snapshots/*/base/*/*.pt",
            ]
        else:
            patterns = [
                # HF cache structure
                f"{hub_dir}/models--nvidia--Cosmos-Predict2.5-2B/snapshots/*/{variant}/*.pt",
                f"{hub_dir}/models--nvidia--Cosmos-Predict2.5-14B/snapshots/*/{variant}/*.pt",
                # Custom structure
                f"{checkpoint_dir}/nvidia/Cosmos-Predict2.5-2B/{variant}/*.pt",
                f"{checkpoint_dir}/nvidia/Cosmos-Predict2.5-14B/{variant}/*.pt",
            ]
    elif model_type == "Transfer2.5":
        patterns = [
            # HF cache structure (hub subdirectory)
            f"{hub_dir}/models--nvidia--Cosmos-Transfer2.5-2B/snapshots/*/{variant}/*.pt",
        ]
    
    # Search for checkpoints
    for pattern in patterns:
        matches = glob(pattern, recursive=True)
        if matches:
            return Path(matches[0])
    
    return None


def write_video(
    video_tensor,
    output_path: Path,
    fps: int = 24,
):
    """
    Save a video tensor to file using torchvision.
    
    Args:
        video_tensor: Video tensor from generation (in range [-1, 1])
                     Shape: [B, C, T, H, W] or [B, T, C, H, W]
        output_path: Output file path
        fps: Frames per second
    
    Example:
        >>> from cosmos_prefer2.utils.io import write_video
        >>> write_video(video, Path("/workspace/outputs/video.mp4"))
    """
    import torch
    from einops import rearrange
    from torchvision.io import write_video as torchvision_write_video
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove batch dimension (take first sample if batched)
    if video_tensor.ndim == 5:
        video_tensor = video_tensor[0]  # Now shape: [C, T, H, W] or [T, C, H, W]
    
    # Detect the format and convert to [T, H, W, C]
    if video_tensor.shape[0] == 3:  # [C, T, H, W]
        video = rearrange(video_tensor, 'c t h w -> t h w c')
    else:  # [T, C, H, W]
        video = rearrange(video_tensor, 't c h w -> t h w c')
    
    # Convert from [-1, 1] to [0, 255] uint8
    video_normalized = ((video + 1.0) * 127.5).clamp(0, 255).byte().cpu()
    
    # Write video
    torchvision_write_video(
        str(output_path),
        video_normalized,
        fps=fps,
        video_codec='h264',
        options={'crf': '18'}  # High quality (lower = better, 18 is visually lossless)
    )
    
    console.print(f"[green]✓ Video saved:[/green] [dim]{output_path}[/dim]")

