#!/usr/bin/env python3
"""
Transfer helper utilities for Cosmos-Transfer2.5.

This module provides helper functions for setting up and running inference
with Cosmos Transfer models for controlled video generation.
"""

import os
from pathlib import Path
from typing import Optional

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


def get_experiment_for_control_type(control_type: str, checkpoint_path: Optional[Path] = None) -> str:
    """
    Get the correct experiment name for a Transfer2.5 control type.
    
    Args:
        control_type: Type of control ("depth", "edge", "seg", "multiview")
        checkpoint_path: Optional checkpoint path for UUID-based mapping
    
    Returns:
        Experiment name string
    
    Example:
        >>> experiment = get_experiment_for_control_type("depth")
        'vid2vid_2B_control_720p_t24_control_layer4_cr1pt1_embedding_rectified_flow'
    """
    # Transfer2.5 uses a single experiment name for all control types
    # The control type is specified via the hint_keys parameter during initialization
    return "vid2vid_2B_control_720p_t24_control_layer4_cr1pt1_embedding_rectified_flow"


def setup_transfer_environment(cosmos_transfer_dir: Optional[Path] = None) -> tuple[Path, Path]:
    """
    Setup the environment for Transfer2.5 inference.
    
    This function:
    1. Sets HF_HOME to /workspace/checkpoints (if not already set)
    2. Changes to cosmos-transfer2.5 directory for Hydra config resolution
    
    Args:
        cosmos_transfer_dir: Path to cosmos-transfer2.5 directory
    
    Returns:
        Tuple of (original_cwd, cosmos_transfer_dir)
    
    Example:
        >>> original_cwd, transfer_dir = setup_transfer_environment()
        >>> # ... do inference ...
        >>> os.chdir(original_cwd)  # restore
    """
    original_cwd = Path.cwd()
    
    # Setup HF cache to avoid duplicate downloads
    setup_hf_cache(Path("/workspace/checkpoints"))
    
    if cosmos_transfer_dir is None:
        if Path("/workspace/cosmos-transfer2.5").exists():
            cosmos_transfer_dir = Path("/workspace/cosmos-transfer2.5")
        elif (Path.home() / "cosmos-transfer2.5").exists():
            cosmos_transfer_dir = Path.home() / "cosmos-transfer2.5"
        else:
            raise FileNotFoundError(
                "cosmos-transfer2.5 directory not found. "
                "Expected at /workspace/cosmos-transfer2.5 or ~/cosmos-transfer2.5"
            )
    
    os.chdir(cosmos_transfer_dir)
    return original_cwd, cosmos_transfer_dir


def build_transfer_pipeline(
    checkpoint_path: Optional[Path] = None,
    model_name: str = "nvidia/Cosmos-Transfer2.5-2B",
    control_type: str = "depth",
    experiment: Optional[str] = None,
    context_parallel_size: int = 1,
    offload_diffusion_model: bool = False,
    offload_text_encoder: bool = False,
    offload_tokenizer: bool = False,
):
    """
    Create a Transfer2.5 pipeline for controlled video generation.
    
    IMPORTANT: If checkpoint_path is None, the system will use the checkpoint database
    which respects HF_HOME. This is the recommended approach to avoid duplicate downloads.
    
    Args:
        checkpoint_path: Path to checkpoint (None = use checkpoint database, recommended!)
        model_name: HuggingFace repo ID (e.g., "nvidia/Cosmos-Transfer2.5-2B")
        control_type: Type of control ("depth", "edge", "seg", "multiview")
        experiment: Experiment name (auto-detected if None)
        context_parallel_size: Number of GPUs for context parallelism
        offload_diffusion_model: Offload diffusion model to CPU
        offload_text_encoder: Offload text encoder to CPU
        offload_tokenizer: Offload tokenizer to CPU
    
    Returns:
        ControlVideo2WorldInference instance for Transfer2.5
    
    Example:
        >>> # Recommended: Let system use checkpoint database (respects HF_HOME)
        >>> pipe = build_transfer_pipeline(
        ...     model_name="nvidia/Cosmos-Transfer2.5-2B",
        ...     control_type="depth"
        ... )
        
        >>> # Or specify checkpoint path
        >>> checkpoint = check_downloads(model_type="Transfer2.5", variant="general/depth")
        >>> pipe = build_transfer_pipeline(checkpoint_path=checkpoint, control_type="depth")
    """
    try:
        from cosmos_transfer2._src.transfer2.inference.inference_pipeline import ControlVideo2WorldInference
    except ModuleNotFoundError as e:
        console.print(f"[bold red]✗ Error importing Transfer2.5 dependencies:[/bold red] {e}")
        console.print("\n[yellow]Missing dependency detected![/yellow]")
        
        if 'sam2' in str(e):
            console.print("\n[cyan]The SAM2 (Segment Anything Model 2) package is required.[/cyan]")
            console.print("\n[bold]Installation steps:[/bold]")
            console.print("  1. Install SAM2:")
            console.print("     [white]pip install git+https://github.com/facebookresearch/segment-anything-2.git[/white]")
            console.print("\n  2. Or if already cloned:")
            console.print("     [white]cd /workspace/segment-anything-2[/white]")
            console.print("     [white]pip install -e .[/white]")
        else:
            console.print("\n[cyan]Please ensure all Transfer2.5 dependencies are installed:[/cyan]")
            console.print("  [white]cd /workspace/cosmos-transfer2.5[/white]")
            console.print("  [white]pip install -r requirements.txt[/white]")
        
        raise
    
    # If checkpoint_path is provided, auto-detect experiment
    if checkpoint_path is not None:
        if experiment is None:
            experiment = get_experiment_for_control_type(control_type, checkpoint_path)
            console.print(f"[dim]Auto-detected experiment: {experiment}[/dim]")
        checkpoint_str = str(checkpoint_path)
        use_checkpoint_db = False
    else:
        # Use checkpoint database - find checkpoint from HF cache or let imaginaire download
        if experiment is None:
            experiment = get_experiment_for_control_type(control_type)
        
        # Try to find checkpoint in HF cache
        console.print(f"[dim]Searching for checkpoint in HF cache for experiment: {experiment}[/dim]")
        checkpoint_path = check_downloads(
            model_type="Transfer2.5",
            variant=f"general/{control_type}",
        )
        
        if checkpoint_path is not None:
            # Found in cache - use it directly
            checkpoint_str = str(checkpoint_path)
            use_checkpoint_db = False
            console.print(f"[green]✓ Found checkpoint in cache:[/green] [dim]{checkpoint_str}[/dim]")
        else:
            # Not found in cache - use checkpoint database (don't pass ckpt_path at all)
            checkpoint_str = None
            use_checkpoint_db = True
            console.print(f"[yellow]Checkpoint not found in cache. Using checkpoint database...[/yellow]")
            console.print(f"[dim]This will download from HuggingFace to HF_HOME: {os.environ.get('HF_HOME', 'default')}[/dim]")
    
    config_file = "cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/config.py"
    
    # Build kwargs for ControlVideo2WorldInference
    # The Transfer2.5 API is different from Predict2.5
    inference_kwargs = {
        "registered_exp_name": experiment,
        "checkpoint_paths": checkpoint_str if not use_checkpoint_db else None,
        "s3_credential_path": "",
        "process_group": None,  # Set if using context parallelism
        "cache_dir": None,
    }
    
    # Note: Transfer2.5 uses ControlVideo2WorldInference which has different initialization
    # It doesn't support the same offload options as Predict2.5
    return ControlVideo2WorldInference(**inference_kwargs)


def yield_video(
    pipe,
    prompt: str,
    control_video_path: Path,
    control_type: str = "depth",
    num_frames: int = 93,
    resolution: tuple[int, int] = (704, 1280),
    guidance: float = 7.5,
    num_steps: int = 50,
    seed: Optional[int] = None,
):
    """
    Generate video with control signal using Transfer2.5.
    
    Args:
        pipe: ControlVideo2WorldInference instance (Transfer2.5)
        prompt: Text prompt describing desired output
        control_video_path: Path to control video (depth/edge/seg maps)
        control_type: Type of control signal ("depth", "edge", "seg")
        num_frames: Frames per chunk (default=93, matching state_t=24)
        resolution: Output resolution (height, width)
        guidance: Guidance scale
        num_steps: Number of denoising steps
        seed: Random seed for reproducibility
    
    Returns:
        Generated video tensor (B, C, T, H, W) in range [-1, 1]
    
    Example:
        >>> video = yield_video(
        ...     pipe,
        ...     prompt="A cinematic scene with dramatic lighting",
        ...     control_video_path=Path("/workspace/datasets/depth_map.mp4"),
        ...     control_type="depth",
        ...     num_frames=93,
        ...     seed=42
        ... )
    
    Note:
        The default num_frames=93 corresponds to state_t=24 in the Transfer2.5 model.
        If your control video has more frames, it will be processed in chunks automatically.
        For best results, ensure your control video matches the expected frame count.
    """
    # Map resolution tuple to resolution string
    # Transfer2.5 uses "720", "480", etc.
    resolution_map = {
        (704, 1280): "720",
        (480, 854): "480",
        (1080, 1920): "1080",
    }
    resolution_str = resolution_map.get(resolution, "720")
    
    # Build input_control_video_paths dict
    # This maps the control modality to the video path
    input_control_video_paths = {
        control_type: str(control_video_path)
    }
    
    # Transfer2.5 uses generate_img2world method
    # It returns a tuple: (video, control_dict, mask_dict, fps, original_hw)
    # Note: video_path is used to determine base properties (dims, fps, frame count)
    # while input_control_video_paths provides the actual control signals
    # IMPORTANT: Do NOT pass max_frames - let it read the full video to ensure
    # base video and control video have matching frame counts
    result = pipe.generate_img2world(
        prompt=prompt,
        video_path=str(control_video_path),
        guidance=int(guidance),
        seed=seed if seed is not None else 1,
        resolution=resolution_str,
        num_steps=num_steps,
        num_video_frames_per_chunk=num_frames,
        hint_key=[control_type],
        input_control_video_paths=input_control_video_paths,
    )
    
    # Return only the video tensor (first element of tuple)
    return result[0]
