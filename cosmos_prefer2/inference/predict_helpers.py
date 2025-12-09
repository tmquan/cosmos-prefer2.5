#!/usr/bin/env python3
"""
Inference helper utilities for Cosmos-Predict2.5.

This module provides helper functions for setting up and running inference
with Cosmos models, bridging the gap between checkpoint management and inference.
"""

import os
from pathlib import Path
from typing import Optional
from glob import glob

from rich.console import Console

# Import I/O utilities
from cosmos_prefer2.utils.io import check_downloads, write_video

console = Console()


def setup_hf_cache(cache_dir: Path = Path("/workspace/checkpoints")):
    """
    Setup HuggingFace cache directory to avoid duplicate downloads.
    
    Args:
        cache_dir: Directory where HF cache should be located
    
    Example:
        >>> setup_hf_cache(Path("/workspace/checkpoints"))
        >>> # Now all HF downloads will use this cache
    """
    if not os.environ.get('HF_HOME'):
        os.environ['HF_HOME'] = str(cache_dir)
        console.print(f"[dim]Set HF_HOME to: {cache_dir}[/dim]")
    else:
        console.print(f"[dim]Using existing HF_HOME: {os.environ['HF_HOME']}[/dim]")


def get_experiment_for_checkpoint(checkpoint_path: Path) -> str:
    """
    Get the correct experiment name for a checkpoint.
    
    This maps checkpoint UUIDs/paths to their registered experiment names.
    
    Args:
        checkpoint_path: Path to the checkpoint file
    
    Returns:
        Experiment name string
    
    Example:
        >>> checkpoint = Path("/workspace/checkpoints/.../81edfebe-..._ema_bf16.pt")
        >>> experiment = get_experiment_for_checkpoint(checkpoint)
    """
    checkpoint_str = str(checkpoint_path)
    
    # Cosmos-Predict2.5-2B base models
    if "81edfebe" in checkpoint_str or ("post-trained" in checkpoint_str and "2B" in checkpoint_str):
        return "Stage-c_pt_4-Index-2-Size-2B-Res-720-Fps-16-Note-rf_with_edm_ckpt"
    elif "d20b7120" in checkpoint_str or ("pre-trained" in checkpoint_str and "2B" in checkpoint_str):
        return "Stage-c_pt_4-Index-2-Size-2B-Res-720-Fps-16-Note-rf_with_edm_ckpt"
    
    # Cosmos-Predict2.5-14B base models
    elif "54937b8c" in checkpoint_str or ("pre-trained" in checkpoint_str and "14B" in checkpoint_str):
        return "Stage-c_pt_4-Index-3-Size-14B-Res-720-Fps-16-Note-rf_with_edm_ckpt"
    elif "e21d2a49" in checkpoint_str or ("post-trained" in checkpoint_str and "14B" in checkpoint_str):
        return "Stage-c_pt_4-Index-3-Size-14B-Res-720-Fps-16-Note-rf_with_edm_ckpt"
    
    # Auto/Multiview
    elif "524af350" in checkpoint_str or "6b9d7548" in checkpoint_str or "auto/multiview" in checkpoint_str:
        return "predict2_multiview_training_2b_auto_multiview_gcp"
    
    # Robot Action-Conditioned
    elif "38c6c645" in checkpoint_str or "robot/action-cond" in checkpoint_str:
        return "predict2_action_training_2b_robot_action_gcp"
    
    # Robot Multiview Agibot
    elif "f740321e" in checkpoint_str or "robot/multiview-agibot" in checkpoint_str:
        return "predict2_multiview_agibot_training"
    
    # Default to 2B base model experiment
    else:
        console.print(f"[yellow]Warning: Unknown checkpoint, using default experiment[/yellow]")
        return "Stage-c_pt_4-Index-2-Size-2B-Res-720-Fps-16-Note-rf_with_edm_ckpt"


def setup_inference_environment(cosmos_predict_dir: Optional[Path] = None) -> tuple[Path, Path]:
    """
    Setup the environment for inference.
    
    This function:
    1. Sets HF_HOME to /workspace/checkpoints (if not already set)
    2. Changes to cosmos-predict2.5 directory for Hydra config resolution
    
    Args:
        cosmos_predict_dir: Path to cosmos-predict2.5 directory
    
    Returns:
        Tuple of (original_cwd, cosmos_predict_dir)
    
    Example:
        >>> original_cwd, predict_dir = setup_inference_environment()
        >>> # ... do inference ...
        >>> os.chdir(original_cwd)  # restore
    """
    original_cwd = Path.cwd()
    
    # Setup HF cache to avoid duplicate downloads
    setup_hf_cache(Path("/workspace/checkpoints"))
    
    if cosmos_predict_dir is None:
        if Path("/workspace/cosmos-predict2.5").exists():
            cosmos_predict_dir = Path("/workspace/cosmos-predict2.5")
        elif (Path.home() / "cosmos-predict2.5").exists():
            cosmos_predict_dir = Path.home() / "cosmos-predict2.5"
        else:
            raise FileNotFoundError(
                "cosmos-predict2.5 directory not found. "
                "Expected at /workspace/cosmos-predict2.5 or ~/cosmos-predict2.5"
            )
    
    os.chdir(cosmos_predict_dir)
    return original_cwd, cosmos_predict_dir


def build_inference_pipeline(
    checkpoint_path: Optional[Path] = None,
    model_name: str = "nvidia/Cosmos-Predict2.5-2B",
    experiment: Optional[str] = None,
    context_parallel_size: int = 1,
    offload_diffusion_model: bool = False,
    offload_text_encoder: bool = False,
    offload_tokenizer: bool = False,
):
    """
    Create a Video2WorldInference pipeline with proper configuration.
    
    IMPORTANT: If checkpoint_path is None, the system will use the checkpoint database
    which respects HF_HOME. This is the recommended approach to avoid duplicate downloads.
    
    Args:
        checkpoint_path: Path to checkpoint (None = use checkpoint database, recommended!)
        model_name: HuggingFace repo ID (e.g., "nvidia/Cosmos-Predict2.5-2B", "nvidia/Cosmos-Predict2.5-14B")
        experiment: Experiment name (auto-detected if None)
        context_parallel_size: Number of GPUs for context parallelism
        offload_diffusion_model: Offload diffusion model to CPU
        offload_text_encoder: Offload text encoder to CPU
        offload_tokenizer: Offload tokenizer to CPU
    
    Returns:
        Video2WorldInference instance
    
    Example:
        >>> # Recommended: Let system use checkpoint database (respects HF_HOME)
        >>> pipe = build_inference_pipeline(model_name="nvidia/Cosmos-Predict2.5-2B")
        
        >>> # Or specify checkpoint path (may trigger downloads)
        >>> checkpoint = check_downloads()
        >>> pipe = build_inference_pipeline(checkpoint_path=checkpoint)
    """
    from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
    
    # If checkpoint_path is provided, auto-detect experiment
    if checkpoint_path is not None:
        if experiment is None:
            experiment = get_experiment_for_checkpoint(checkpoint_path)
            console.print(f"[dim]Auto-detected experiment: {experiment}[/dim]")
        checkpoint_str = str(checkpoint_path)
        use_checkpoint_db = False
    else:
        # Use checkpoint database - find checkpoint from HF cache or let imaginaire download
        if experiment is None:
            # Map HuggingFace repo ID to experiment name
            if "Cosmos-Predict2.5-2B" in model_name:
                experiment = "Stage-c_pt_4-Index-2-Size-2B-Res-720-Fps-16-Note-rf_with_edm_ckpt"
            elif "Cosmos-Predict2.5-14B" in model_name:
                experiment = "Stage-c_pt_4-Index-43-Size-14B-Res-720-Fps-16-Note-T24_HQV5_from_40"
            elif "Cosmos-Transfer2.5-2B" in model_name:
                # Placeholder for Transfer model - update when available
                experiment = "Stage-c_pt_4-Index-2-Size-2B-Res-720-Fps-16-Note-rf_with_edm_ckpt"
            elif "Cosmos-Transfer2.5-14B" in model_name:
                # Placeholder for Transfer model - update when available
                experiment = "Stage-c_pt_4-Index-43-Size-14B-Res-720-Fps-16-Note-T24_HQV5_from_40"
            else:
                # Default to 2B post-trained
                experiment = "Stage-c_pt_4-Index-2-Size-2B-Res-720-Fps-16-Note-rf_with_edm_ckpt"
        
        # Try to find checkpoint in HF cache
        console.print(f"[dim]Searching for checkpoint in HF cache for experiment: {experiment}[/dim]")
        checkpoint_path = check_downloads()
        
        if checkpoint_path is not None:
            # Found in cache - use it directly
            checkpoint_str = str(checkpoint_path)
            use_checkpoint_db = False
            console.print(f"[green]✓ Found checkpoint in cache:[/green] [dim]{checkpoint_str}[/dim]")
        else:
            # Not found in cache - use checkpoint database (don't pass ckpt_path at all)
            checkpoint_str = None
            use_checkpoint_db = True
            console.print(f"[yellow]Checkpoint not found in cache. Using imaginaire checkpoint database...[/yellow]")
            console.print(f"[dim]This will download from HuggingFace to HF_HOME: {os.environ.get('HF_HOME', 'default')}[/dim]")
    
    config_file = "cosmos_predict2/_src/predict2/configs/video2world/config.py"
    
    # Build kwargs based on whether we're using checkpoint database
    inference_kwargs = {
        "experiment_name": experiment,
        "s3_credential_path": "",
        "context_parallel_size": context_parallel_size,
        "config_file": config_file,
        "offload_diffusion_model": offload_diffusion_model,
        "offload_text_encoder": offload_text_encoder,
        "offload_tokenizer": offload_tokenizer,
    }
    
    # Only add ckpt_path if we're not using checkpoint database
    if not use_checkpoint_db:
        inference_kwargs["ckpt_path"] = checkpoint_str
    
    return Video2WorldInference(**inference_kwargs)


def yield_video(
    pipe,
    prompt: str,
    num_frames: int = 121,
    resolution: tuple[int, int] = (704, 1280),
    guidance: float = 7.5,
    num_steps: int = 50,
    seed: Optional[int] = None,
    input_video: Optional[Path] = None,
):
    """
    Generate a video using the inference pipeline.
    
    Args:
        pipe: Video2WorldInference instance
        prompt: Text prompt
        num_frames: Number of frames to generate
        resolution: Output resolution (height, width)
        guidance: Guidance scale
        num_steps: Number of denoising steps
        seed: Random seed
        input_video: Optional input video for video2video
    
    Returns:
        Generated video tensor
    
    Example:
        >>> video = yield_video(
        ...     pipe,
        ...     prompt="A robot picks up a cube",
        ...     num_frames=121,
        ...     seed=42
        ... )
    """
    resolution_str = f"{resolution[0]},{resolution[1]}"
    
    return pipe.generate_vid2world(
        prompt=prompt,
        input_path=str(input_video) if input_video else None,
        guidance=guidance,
        num_video_frames=num_frames,
        num_latent_conditional_frames=1,
        resolution=resolution_str,
        seed=seed,
        negative_prompt="",
        num_steps=num_steps,
    )

