#!/usr/bin/env python3
"""
Comprehensive Automated Test for Cosmos-Prefer2.5

This script automatically tests all possible inference cases:
- Predict2.5: text2world, image2world, video2world
- Transfer2.5: depth, edge, seg control

No user interaction required - uses predefined prompts and inputs.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import traceback

# Set HF_HOME before any imports
if not os.environ.get('HF_HOME'):
    hf_cache = Path("/workspace/checkpoints") if Path("/workspace").exists() else Path("./checkpoints")
    os.environ['HF_HOME'] = str(hf_cache)
    print(f"Set HF_HOME to: {hf_cache}")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from cosmos_prefer2.utils import setup_pythonpath, check_environment, check_downloads, write_video
from cosmos_prefer2.inference.predict_helpers import (
    setup_inference_environment,
    build_inference_pipeline,
    yield_video as predict_yield_video,
)
from cosmos_prefer2.inference.transfer_helpers import (
    setup_transfer_environment,
    build_transfer_pipeline,
    yield_video as transfer_yield_video,
)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()


class TestCase:
    """Represents a test case."""
    
    def __init__(self, name, model_type, mode, prompt, input_path=None, control_type=None):
        self.name = name
        self.model_type = model_type  # "predict" or "transfer"
        self.mode = mode  # "text2world", "image2world", "video2world", or control type
        self.prompt = prompt
        self.input_path = input_path
        self.control_type = control_type
        self.status = "pending"
        self.output_path = None
        self.error = None
        self.duration = None


class TestRunner:
    """Manages and runs all test cases."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_cases = []
        self.results = {"passed": [], "failed": [], "skipped": []}
        self.predict_pipe = None
        self.transfer_pipes = {}
        self.original_cwd = Path.cwd()
    
    def add_test_case(self, test_case: TestCase):
        """Add a test case."""
        self.test_cases.append(test_case)
    
    def setup_predict_pipeline(self):
        """Setup Predict2.5 pipeline."""
        console.print("\n[bold cyan]Setting up Predict2.5 Pipeline...[/bold cyan]\n")
        
        try:
            # Check environment
            if not Path("/workspace/cosmos-predict2.5").exists():
                console.print("[yellow]⊘ cosmos-predict2.5 not found, skipping Predict tests[/yellow]")
                return False
            
            # Setup environment
            original_cwd, predict_dir = setup_inference_environment()
            console.print(f"[dim]Working directory: {predict_dir}[/dim]")
            
            # Check for checkpoint
            checkpoint_path = check_downloads(
                model_type="Predict2.5",
                variant="base",
                preferred_training="post-trained"
            )
            
            if checkpoint_path:
                console.print(f"[green]✓ Found checkpoint:[/green] [dim]{checkpoint_path.name}[/dim]")
            else:
                console.print("[yellow]No checkpoint found, will download...[/yellow]")
            
            # Build pipeline
            self.predict_pipe = build_inference_pipeline(
                checkpoint_path=checkpoint_path,
                model_name="nvidia/Cosmos-Predict2.5-2B",
            )
            
            console.print("[green]✓ Predict2.5 pipeline ready![/green]\n")
            return True
            
        except Exception as e:
            console.print(f"[red]✗ Error setting up Predict2.5:[/red] {e}\n")
            traceback.print_exc()
            return False
    
    def setup_transfer_pipeline(self, control_type: str):
        """Setup Transfer2.5 pipeline for a specific control type."""
        if control_type in self.transfer_pipes:
            return True
        
        console.print(f"\n[bold cyan]Setting up Transfer2.5 Pipeline ({control_type})...[/bold cyan]\n")
        
        try:
            # Check environment
            if not Path("/workspace/cosmos-transfer2.5").exists():
                console.print("[yellow]⊘ cosmos-transfer2.5 not found, skipping Transfer tests[/yellow]")
                return False
            
            # Setup environment (if not already done)
            if not hasattr(self, 'transfer_dir'):
                self.transfer_original_cwd, self.transfer_dir = setup_transfer_environment()
                console.print(f"[dim]Working directory: {self.transfer_dir}[/dim]")
            
            # Check for checkpoint
            checkpoint_path = check_downloads(
                model_type="Transfer2.5",
                variant=f"general/{control_type}",
            )
            
            if checkpoint_path:
                console.print(f"[green]✓ Found checkpoint:[/green] [dim]{checkpoint_path.name}[/dim]")
            else:
                console.print("[yellow]No checkpoint found, will download...[/yellow]")
            
            # Build pipeline
            self.transfer_pipes[control_type] = build_transfer_pipeline(
                checkpoint_path=checkpoint_path,
                control_type=control_type,
            )
            
            console.print(f"[green]✓ Transfer2.5-{control_type} pipeline ready![/green]\n")
            return True
            
        except Exception as e:
            console.print(f"[red]✗ Error setting up Transfer2.5-{control_type}:[/red] {e}\n")
            traceback.print_exc()
            return False
    
    def run_predict_test(self, test_case: TestCase):
        """Run a Predict2.5 test case."""
        console.print(f"\n[bold]Running: {test_case.name}[/bold]\n")
        
        try:
            # Determine parameters
            num_frames = 93  # Use 93 for faster testing, can be 121 for full
            resolution = (704, 1280)
            guidance = 7.5
            num_steps = 50
            seed = 42
            
            console.print("[dim]Parameters:[/dim]")
            console.print(f"  • Mode: {test_case.mode}")
            console.print(f"  • Prompt: {test_case.prompt[:60]}...")
            console.print(f"  • Frames: {num_frames}")
            console.print(f"  • Resolution: {resolution[0]}x{resolution[1]}")
            if test_case.input_path:
                console.print(f"  • Input: {test_case.input_path.name}")
            console.print()
            
            import time
            start_time = time.time()
            
            # Generate video
            console.print("[yellow]Generating...[/yellow]\n")
            video = predict_yield_video(
                pipe=self.predict_pipe,
                prompt=test_case.prompt,
                num_frames=num_frames,
                resolution=resolution,
                guidance=guidance,
                num_steps=num_steps,
                seed=seed,
                input_video=test_case.input_path,
            )
            
            test_case.duration = time.time() - start_time
            
            # Save video
            output_filename = f"predict_{test_case.mode}_{seed}.mp4"
            test_case.output_path = self.output_dir / output_filename
            
            write_video(video, test_case.output_path, fps=24)
            
            file_size_mb = test_case.output_path.stat().st_size / (1024 * 1024)
            
            console.print(f"[green]✓ Success![/green]")
            console.print(f"[dim]Output: {test_case.output_path}[/dim]")
            console.print(f"[dim]Size: {file_size_mb:.1f}MB | Duration: {test_case.duration:.1f}s[/dim]\n")
            
            test_case.status = "passed"
            self.results["passed"].append(test_case)
            
        except Exception as e:
            test_case.status = "failed"
            test_case.error = str(e)
            self.results["failed"].append(test_case)
            console.print(f"[red]✗ Failed: {e}[/red]\n")
            traceback.print_exc()
    
    def run_transfer_test(self, test_case: TestCase):
        """Run a Transfer2.5 test case."""
        console.print(f"\n[bold]Running: {test_case.name}[/bold]\n")
        
        try:
            # Determine parameters
            num_frames = 93  # Must be 93 for Transfer2.5 (state_t=24)
            resolution = (704, 1280)
            guidance = 7.5
            num_steps = 50
            seed = 42
            
            console.print("[dim]Parameters:[/dim]")
            console.print(f"  • Control: {test_case.control_type}")
            console.print(f"  • Prompt: {test_case.prompt[:60]}...")
            console.print(f"  • Frames: {num_frames}")
            console.print(f"  • Resolution: {resolution[0]}x{resolution[1]}")
            console.print(f"  • Control video: {test_case.input_path.name}")
            console.print()
            
            import time
            start_time = time.time()
            
            # Generate video
            console.print("[yellow]Generating...[/yellow]\n")
            video = transfer_yield_video(
                pipe=self.transfer_pipes[test_case.control_type],
                prompt=test_case.prompt,
                control_video_path=test_case.input_path,
                control_type=test_case.control_type,
                num_frames=num_frames,
                resolution=resolution,
                guidance=guidance,
                num_steps=num_steps,
                seed=seed,
            )
            
            test_case.duration = time.time() - start_time
            
            # Save video
            output_filename = f"transfer_{test_case.control_type}_{seed}.mp4"
            test_case.output_path = self.output_dir / output_filename
            
            write_video(video, test_case.output_path, fps=24)
            
            file_size_mb = test_case.output_path.stat().st_size / (1024 * 1024)
            
            console.print(f"[green]✓ Success![/green]")
            console.print(f"[dim]Output: {test_case.output_path}[/dim]")
            console.print(f"[dim]Size: {file_size_mb:.1f}MB | Duration: {test_case.duration:.1f}s[/dim]\n")
            
            test_case.status = "passed"
            self.results["passed"].append(test_case)
            
        except Exception as e:
            test_case.status = "failed"
            test_case.error = str(e)
            self.results["failed"].append(test_case)
            console.print(f"[red]✗ Failed: {e}[/red]\n")
            traceback.print_exc()
    
    def run_all_tests(self):
        """Run all test cases."""
        console.print("\n[bold cyan]Running All Test Cases[/bold cyan]\n")
        
        for test_case in self.test_cases:
            if test_case.model_type == "predict":
                if self.predict_pipe is None:
                    test_case.status = "skipped"
                    test_case.error = "Predict pipeline not available"
                    self.results["skipped"].append(test_case)
                    console.print(f"[yellow]⊘ Skipping: {test_case.name}[/yellow]\n")
                else:
                    # Change to predict directory
                    os.chdir(self.transfer_dir if hasattr(self, 'transfer_dir') else self.original_cwd)
                    self.run_predict_test(test_case)
                    
            elif test_case.model_type == "transfer":
                if test_case.control_type not in self.transfer_pipes:
                    test_case.status = "skipped"
                    test_case.error = f"Transfer-{test_case.control_type} pipeline not available"
                    self.results["skipped"].append(test_case)
                    console.print(f"[yellow]⊘ Skipping: {test_case.name}[/yellow]\n")
                else:
                    # Change to transfer directory
                    os.chdir(self.transfer_dir)
                    self.run_transfer_test(test_case)
        
        # Restore original directory
        os.chdir(self.original_cwd)
    
    def print_summary(self):
        """Print test summary."""
        console.print("\n" + "="*70)
        console.print("[bold cyan]Test Summary[/bold cyan]")
        console.print("="*70 + "\n")
        
        # Create summary table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Test Case", style="dim", width=40)
        table.add_column("Status", width=12)
        table.add_column("Duration", justify="right", width=10)
        
        for test_case in self.test_cases:
            if test_case.status == "passed":
                status = f"[green]✓ Passed[/green]"
                duration = f"{test_case.duration:.1f}s" if test_case.duration else "-"
            elif test_case.status == "failed":
                status = f"[red]✗ Failed[/red]"
                duration = "-"
            else:
                status = f"[yellow]⊘ Skipped[/yellow]"
                duration = "-"
            
            table.add_row(test_case.name, status, duration)
        
        console.print(table)
        console.print()
        
        # Print statistics
        total = len(self.test_cases)
        passed = len(self.results["passed"])
        failed = len(self.results["failed"])
        skipped = len(self.results["skipped"])
        
        stats_table = Table(show_header=True, header_style="bold")
        stats_table.add_column("Status", width=15)
        stats_table.add_column("Count", justify="right", width=10)
        stats_table.add_column("Percentage", justify="right", width=15)
        
        stats_table.add_row(
            "[green]Passed[/green]",
            f"[green]{passed}[/green]",
            f"[green]{passed/total*100:.1f}%[/green]" if total > 0 else "0%"
        )
        stats_table.add_row(
            "[red]Failed[/red]",
            f"[red]{failed}[/red]",
            f"[red]{failed/total*100:.1f}%[/red]" if total > 0 else "0%"
        )
        stats_table.add_row(
            "[yellow]Skipped[/yellow]",
            f"[yellow]{skipped}[/yellow]",
            f"[yellow]{skipped/total*100:.1f}%[/yellow]" if total > 0 else "0%"
        )
        stats_table.add_row(
            "[bold]Total[/bold]",
            f"[bold]{total}[/bold]",
            "[bold]100.0%[/bold]"
        )
        
        console.print(stats_table)
        console.print()
        
        # Print outputs location
        if passed > 0:
            console.print(f"[cyan]Generated videos saved to:[/cyan] {self.output_dir.absolute()}\n")
        
        # Print failed test details
        if failed > 0:
            console.print("[bold red]Failed Tests:[/bold red]")
            for test_case in self.results["failed"]:
                console.print(f"  • {test_case.name}: {test_case.error}")
            console.print()
        
        return 0 if failed == 0 else 1


def main():
    # Print header
    console.print(Panel.fit(
        "[bold cyan]Cosmos-Prefer2.5 Comprehensive Test Suite[/bold cyan]\n"
        "[dim]Automated testing of all Predict and Transfer cases[/dim]",
        border_style="cyan"
    ))
    
    # Setup
    console.print("\n[bold]Step 1:[/bold] Checking environment...\n")
    setup_pythonpath()
    status = check_environment()
    
    if not status.get("torch"):
        console.print("[red]✗ PyTorch not available![/red]")
        return 1
    
    if not status.get("cuda"):
        console.print("[yellow]⚠ CUDA not available, tests may be slow[/yellow]")
    
    console.print("[green]✓ Environment ready[/green]")
    
    # Create test runner
    output_dir = Path.cwd() / "data" / "outputs" 
    runner = TestRunner(output_dir)
    
    # Define Predict2.5 test cases
    console.print("\n[bold]Step 2:[/bold] Defining test cases...\n")
    
    # Predict: Text-to-World
    runner.add_test_case(TestCase(
        name="Predict2.5: Text-to-World",
        model_type="predict",
        mode="text2world",
        prompt="A robotic arm slowly picks up a red cube from a wooden table in a well-lit laboratory with dramatic lighting"
    ))
    
    # Predict: Image-to-World (if example exists)
    inputs_dir = Path.cwd() / "data" / "inputs" / "predict2.5"
    image2world_dir = inputs_dir / "image2world"
    if image2world_dir.exists():
        input_image = image2world_dir / "input_image.png"
        prompt_file = image2world_dir / "prompt.txt"
        
        if input_image.exists() and prompt_file.exists():
            prompt = prompt_file.read_text().strip()
            runner.add_test_case(TestCase(
                name="Predict2.5: Image-to-World",
                model_type="predict",
                mode="image2world",
                prompt=prompt,
                input_path=input_image
            ))
    
    # Predict: Video-to-World (if example exists)
    video2world_dir = inputs_dir / "video2world"
    if video2world_dir.exists():
        input_video = video2world_dir / "input_video.mp4"
        prompt_file = video2world_dir / "prompt.txt"
        
        if input_video.exists() and prompt_file.exists():
            prompt = prompt_file.read_text().strip()
            runner.add_test_case(TestCase(
                name="Predict2.5: Video-to-World",
                model_type="predict",
                mode="video2world",
                prompt=prompt,
                input_path=input_video
            ))
    
    # Transfer2.5 test cases
    transfer_inputs = Path.cwd() / "data" / "inputs" / "transfer2.5"
    
    # Define prompts for each control type
    transfer_prompts = {
        "depth": "A cinematic scene with dramatic lighting and realistic textures, photorealistic style",
        "edge": "A stylized artistic scene with bold colors and vivid details, high contrast",
        "seg": "A photorealistic scene with accurate materials and natural lighting, ultra detailed",
    }
    
    # Check for transfer examples
    for example_dir in ["sim2real", "real2real"]:
        example_path = transfer_inputs / example_dir
        if example_path.exists():
            prompt_file = example_path / "prompt.txt"
            
            # Use example prompt if available
            if prompt_file.exists():
                example_prompt = prompt_file.read_text().strip()
            else:
                example_prompt = None
            
            # Add test for each control type
            for control_type in ["depth", "edge", "seg"]:
                control_video = example_path / f"control_{control_type}.mp4"
                
                # Fallback to input_video.mp4 if control video not found
                if not control_video.exists():
                    control_video = example_path / "input_video.mp4"
                
                if control_video.exists():
                    prompt = example_prompt if example_prompt else transfer_prompts[control_type]
                    
                    runner.add_test_case(TestCase(
                        name=f"Transfer2.5: {control_type.upper()} ({example_dir})",
                        model_type="transfer",
                        mode=control_type,
                        prompt=prompt,
                        input_path=control_video,
                        control_type=control_type
                    ))
    
    console.print(f"[green]✓ Defined {len(runner.test_cases)} test cases[/green]")
    
    # Setup pipelines
    console.print("\n[bold]Step 3:[/bold] Setting up pipelines...\n")
    
    # Determine which pipelines we need
    needs_predict = any(tc.model_type == "predict" for tc in runner.test_cases)
    transfer_control_types = set(tc.control_type for tc in runner.test_cases if tc.model_type == "transfer")
    
    if needs_predict:
        runner.setup_predict_pipeline()
    
    for control_type in transfer_control_types:
        runner.setup_transfer_pipeline(control_type)
    
    # Run tests
    console.print("\n[bold]Step 4:[/bold] Running tests...\n")
    console.print("="*70 + "\n")
    
    runner.run_all_tests()
    
    # Print summary
    exit_code = runner.print_summary()
    
    # Final message
    if exit_code == 0:
        console.print("[bold green]✨ All tests completed successfully! ✨[/bold green]\n")
    else:
        console.print("[bold yellow]⚠ Some tests failed or were skipped[/bold yellow]\n")
    
    return exit_code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Tests interrupted by user[/yellow]\n")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]Unexpected error:[/bold red] {e}\n")
        traceback.print_exc()
        sys.exit(1)

