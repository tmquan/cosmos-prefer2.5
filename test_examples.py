#!/usr/bin/env python3
"""
Test Suite for Cosmos-Prefer2.5 Examples and Inference Modules

This script tests all examples, helpers, and pipelines to ensure they work correctly.
Can run in interactive or automated mode.
"""

import os
import sys
from pathlib import Path
from typing import Optional
import argparse

# Set HF_HOME before any imports
if not os.environ.get('HF_HOME'):
    hf_cache = Path("/workspace/checkpoints") if Path("/workspace").exists() else Path("./checkpoints")
    os.environ['HF_HOME'] = str(hf_cache)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


class TestResults:
    """Track test results."""
    
    def __init__(self):
        self.passed = []
        self.failed = []
        self.skipped = []
    
    def add_pass(self, test_name: str):
        self.passed.append(test_name)
        console.print(f"[green]✓ {test_name}[/green]")
    
    def add_fail(self, test_name: str, error: str):
        self.failed.append((test_name, error))
        console.print(f"[red]✗ {test_name}[/red]")
        console.print(f"[dim]  Error: {error}[/dim]")
    
    def add_skip(self, test_name: str, reason: str):
        self.skipped.append((test_name, reason))
        console.print(f"[yellow]⊘ {test_name}[/yellow]")
        console.print(f"[dim]  Reason: {reason}[/dim]")
    
    def print_summary(self):
        """Print test summary table."""
        console.print("\n" + "="*60)
        console.print("[bold cyan]Test Summary[/bold cyan]")
        console.print("="*60 + "\n")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Status", style="dim", width=12)
        table.add_column("Count", justify="right")
        
        table.add_row("[green]Passed[/green]", f"[green]{len(self.passed)}[/green]")
        table.add_row("[red]Failed[/red]", f"[red]{len(self.failed)}[/red]")
        table.add_row("[yellow]Skipped[/yellow]", f"[yellow]{len(self.skipped)}[/yellow]")
        table.add_row("[bold]Total[/bold]", f"[bold]{len(self.passed) + len(self.failed) + len(self.skipped)}[/bold]")
        
        console.print(table)
        
        if self.failed:
            console.print("\n[bold red]Failed Tests:[/bold red]")
            for test_name, error in self.failed:
                console.print(f"  • {test_name}: {error}")
        
        if self.skipped:
            console.print("\n[bold yellow]Skipped Tests:[/bold yellow]")
            for test_name, reason in self.skipped:
                console.print(f"  • {test_name}: {reason}")
        
        console.print()
        
        # Return exit code
        return 0 if len(self.failed) == 0 else 1


def test_imports(results: TestResults):
    """Test that all modules can be imported."""
    console.print("\n[bold]Testing Imports...[/bold]\n")
    
    # Test utils
    try:
        from cosmos_prefer2.utils import setup_pythonpath, check_environment, check_downloads
        results.add_pass("Import cosmos_prefer2.utils")
    except Exception as e:
        results.add_fail("Import cosmos_prefer2.utils", str(e))
    
    # Test predict helpers
    try:
        from cosmos_prefer2.inference.predict_helpers import (
            setup_hf_cache,
            get_experiment_for_checkpoint,
            setup_inference_environment,
            build_inference_pipeline,
            yield_video,
        )
        results.add_pass("Import cosmos_prefer2.inference.predict_helpers")
    except Exception as e:
        results.add_fail("Import cosmos_prefer2.inference.predict_helpers", str(e))
    
    # Test transfer helpers
    try:
        from cosmos_prefer2.inference.transfer_helpers import (
            setup_hf_cache,
            get_experiment_for_control_type,
            setup_transfer_environment,
            build_transfer_pipeline,
            yield_video,
        )
        results.add_pass("Import cosmos_prefer2.inference.transfer_helpers")
    except Exception as e:
        results.add_fail("Import cosmos_prefer2.inference.transfer_helpers", str(e))
    
    # Test pipeline classes
    try:
        from cosmos_prefer2.inference import Predict2Pipeline, Transfer2Pipeline
        results.add_pass("Import pipeline classes")
    except Exception as e:
        results.add_fail("Import pipeline classes", str(e))


def test_environment(results: TestResults):
    """Test environment setup."""
    console.print("\n[bold]Testing Environment...[/bold]\n")
    
    try:
        from cosmos_prefer2.utils import check_environment
        
        status = check_environment()
        
        if status.get("torch"):
            results.add_pass("PyTorch available")
        else:
            results.add_fail("PyTorch available", "PyTorch not found")
        
        if status.get("cuda"):
            results.add_pass("CUDA available")
        else:
            results.add_skip("CUDA available", "CUDA not available (CPU mode)")
        
        if status.get("imaginaire"):
            results.add_pass("Imaginaire available")
        else:
            results.add_skip("Imaginaire available", "Imaginaire not installed")
        
    except Exception as e:
        results.add_fail("Environment check", str(e))


def test_checkpoint_detection(results: TestResults):
    """Test checkpoint detection."""
    console.print("\n[bold]Testing Checkpoint Detection...[/bold]\n")
    
    try:
        from cosmos_prefer2.utils import check_downloads
        
        # Test Predict2.5 checkpoint detection
        predict_ckpt = check_downloads(
            model_type="Predict2.5",
            variant="base",
            preferred_training="post-trained"
        )
        
        if predict_ckpt:
            results.add_pass(f"Predict2.5 checkpoint found: {predict_ckpt.name}")
        else:
            results.add_skip("Predict2.5 checkpoint found", "Not in cache (will download on first use)")
        
        # Test Transfer2.5 checkpoint detection
        for control_type in ["depth", "edge", "seg"]:
            transfer_ckpt = check_downloads(
                model_type="Transfer2.5",
                variant=f"general/{control_type}"
            )
            
            if transfer_ckpt:
                results.add_pass(f"Transfer2.5-{control_type} checkpoint found: {transfer_ckpt.name}")
            else:
                results.add_skip(f"Transfer2.5-{control_type} checkpoint found", 
                               "Not in cache (will download on first use)")
        
    except Exception as e:
        results.add_fail("Checkpoint detection", str(e))


def test_predict_helpers(results: TestResults, run_inference: bool = False):
    """Test Predict2.5 helpers."""
    console.print("\n[bold]Testing Predict2.5 Helpers...[/bold]\n")
    
    try:
        from cosmos_prefer2.inference.predict_helpers import (
            setup_hf_cache,
            setup_inference_environment,
        )
        
        # Test HF cache setup
        setup_hf_cache(Path(os.environ.get('HF_HOME', './checkpoints')))
        results.add_pass("setup_hf_cache()")
        
        # Test environment setup
        if Path("/workspace/cosmos-predict2.5").exists():
            original_cwd, predict_dir = setup_inference_environment()
            results.add_pass("setup_inference_environment()")
            os.chdir(original_cwd)  # Restore
        else:
            results.add_skip("setup_inference_environment()", 
                           "cosmos-predict2.5 not found")
        
        # Test pipeline building
        if run_inference and Path("/workspace/cosmos-predict2.5").exists():
            from cosmos_prefer2.inference.predict_helpers import build_inference_pipeline
            from cosmos_prefer2.utils import check_downloads
            
            checkpoint = check_downloads(model_type="Predict2.5")
            if checkpoint:
                try:
                    os.chdir("/workspace/cosmos-predict2.5")
                    pipe = build_inference_pipeline(checkpoint_path=checkpoint)
                    results.add_pass("build_inference_pipeline()")
                    os.chdir(original_cwd)
                except Exception as e:
                    results.add_fail("build_inference_pipeline()", str(e))
                    os.chdir(original_cwd)
            else:
                results.add_skip("build_inference_pipeline()", "No checkpoint available")
        else:
            results.add_skip("build_inference_pipeline()", "Inference test disabled")
        
    except Exception as e:
        results.add_fail("Predict2.5 helpers", str(e))


def test_transfer_helpers(results: TestResults, run_inference: bool = False):
    """Test Transfer2.5 helpers."""
    console.print("\n[bold]Testing Transfer2.5 Helpers...[/bold]\n")
    
    try:
        from cosmos_prefer2.inference.transfer_helpers import (
            setup_hf_cache,
            setup_transfer_environment,
            get_experiment_for_control_type,
        )
        
        # Test HF cache setup
        setup_hf_cache(Path(os.environ.get('HF_HOME', './checkpoints')))
        results.add_pass("setup_hf_cache()")
        
        # Test experiment detection
        experiment = get_experiment_for_control_type("depth")
        if experiment:
            results.add_pass(f"get_experiment_for_control_type(): {experiment[:50]}...")
        else:
            results.add_fail("get_experiment_for_control_type()", "No experiment returned")
        
        # Test environment setup
        if Path("/workspace/cosmos-transfer2.5").exists():
            original_cwd, transfer_dir = setup_transfer_environment()
            results.add_pass("setup_transfer_environment()")
            os.chdir(original_cwd)  # Restore
        else:
            results.add_skip("setup_transfer_environment()", 
                           "cosmos-transfer2.5 not found")
        
        # Test pipeline building
        if run_inference and Path("/workspace/cosmos-transfer2.5").exists():
            from cosmos_prefer2.inference.transfer_helpers import build_transfer_pipeline
            from cosmos_prefer2.utils import check_downloads
            
            checkpoint = check_downloads(model_type="Transfer2.5", variant="general/depth")
            if checkpoint:
                try:
                    os.chdir("/workspace/cosmos-transfer2.5")
                    pipe = build_transfer_pipeline(checkpoint_path=checkpoint, control_type="depth")
                    results.add_pass("build_transfer_pipeline()")
                    os.chdir(original_cwd)
                except Exception as e:
                    results.add_fail("build_transfer_pipeline()", str(e))
                    os.chdir(original_cwd)
            else:
                results.add_skip("build_transfer_pipeline()", "No checkpoint available")
        else:
            results.add_skip("build_transfer_pipeline()", "Inference test disabled")
        
    except Exception as e:
        results.add_fail("Transfer2.5 helpers", str(e))


def test_pipeline_classes(results: TestResults):
    """Test pipeline class initialization."""
    console.print("\n[bold]Testing Pipeline Classes...[/bold]\n")
    
    # Test Predict2Pipeline
    try:
        from cosmos_prefer2.inference import Predict2Pipeline
        results.add_pass("Predict2Pipeline class importable")
        
        # Test initialization would require checkpoint
        results.add_skip("Predict2Pipeline initialization", "Requires checkpoint")
        
    except Exception as e:
        results.add_fail("Predict2Pipeline", str(e))
    
    # Test Transfer2Pipeline
    try:
        from cosmos_prefer2.inference import Transfer2Pipeline
        results.add_pass("Transfer2Pipeline class importable")
        
        # Test initialization would require checkpoint
        results.add_skip("Transfer2Pipeline initialization", "Requires checkpoint")
        
    except Exception as e:
        results.add_fail("Transfer2Pipeline", str(e))


def test_example_data(results: TestResults):
    """Test for example data availability."""
    console.print("\n[bold]Testing Example Data...[/bold]\n")
    
    # Check for Predict2.5 example data
    predict_data = Path.cwd() / "data" / "inputs" / "predict2.5"
    if predict_data.exists():
        results.add_pass(f"Predict2.5 example data found: {predict_data}")
        
        # Check subdirectories
        for subdir in ["image2world", "video2world"]:
            if (predict_data / subdir).exists():
                results.add_pass(f"  - {subdir} examples found")
            else:
                results.add_skip(f"  - {subdir} examples", "Not found")
    else:
        results.add_skip("Predict2.5 example data", 
                       "Run download_example_data.py to get examples")
    
    # Check for Transfer2.5 example data
    transfer_data = Path.cwd() / "data" / "inputs" / "transfer2.5"
    if transfer_data.exists():
        results.add_pass(f"Transfer2.5 example data found: {transfer_data}")
        
        # Check subdirectories
        for subdir in ["sim2real", "real2real"]:
            if (transfer_data / subdir).exists():
                results.add_pass(f"  - {subdir} examples found")
                
                # Check for control videos
                for control_type in ["depth", "edge", "seg"]:
                    control_file = transfer_data / subdir / f"control_{control_type}.mp4"
                    if control_file.exists():
                        results.add_pass(f"    - control_{control_type}.mp4 found")
            else:
                results.add_skip(f"  - {subdir} examples", "Not found")
    else:
        results.add_skip("Transfer2.5 example data", 
                       "Run download_example_data.py to get examples")


def test_documentation(results: TestResults):
    """Test that all modules have proper documentation."""
    console.print("\n[bold]Testing Documentation...[/bold]\n")
    
    modules_to_check = [
        ("cosmos_prefer2.inference.predict_helpers", 
         ["setup_hf_cache", "setup_inference_environment", "build_inference_pipeline", "yield_video"]),
        ("cosmos_prefer2.inference.transfer_helpers",
         ["setup_hf_cache", "setup_transfer_environment", "build_transfer_pipeline", "yield_video"]),
    ]
    
    for module_name, functions in modules_to_check:
        try:
            module = __import__(module_name, fromlist=functions)
            
            # Check module docstring
            if module.__doc__:
                results.add_pass(f"{module_name} has module docstring")
            else:
                results.add_fail(f"{module_name} module docstring", "Missing")
            
            # Check function docstrings
            for func_name in functions:
                func = getattr(module, func_name, None)
                if func and func.__doc__:
                    results.add_pass(f"{module_name}.{func_name}() has docstring")
                else:
                    results.add_fail(f"{module_name}.{func_name}() docstring", "Missing")
        
        except Exception as e:
            results.add_fail(f"Documentation check for {module_name}", str(e))


def main():
    parser = argparse.ArgumentParser(
        description="Test Cosmos-Prefer2.5 examples and inference modules"
    )
    parser.add_argument(
        "--run-inference",
        action="store_true",
        help="Run inference tests (requires checkpoints and GPU)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only quick tests (skip inference and documentation)"
    )
    args = parser.parse_args()
    
    # Print header
    console.print(Panel.fit(
        "[bold cyan]Cosmos-Prefer2.5 Test Suite[/bold cyan]\n"
        "[dim]Testing examples, helpers, and pipelines[/dim]",
        border_style="cyan"
    ))
    
    results = TestResults()
    
    # Run tests
    test_imports(results)
    test_environment(results)
    test_checkpoint_detection(results)
    test_predict_helpers(results, run_inference=args.run_inference)
    test_transfer_helpers(results, run_inference=args.run_inference)
    test_pipeline_classes(results)
    test_example_data(results)
    
    if not args.quick:
        test_documentation(results)
    
    # Print summary
    exit_code = results.print_summary()
    
    # Print recommendations
    if results.failed or results.skipped:
        console.print("[bold cyan]Recommendations:[/bold cyan]")
        
        if any("checkpoint" in str(item).lower() for item, _ in results.skipped):
            console.print("  • Run [cyan]python download_checkpoints.py[/cyan] to download model checkpoints")
        
        if any("example data" in str(item).lower() for item, _ in results.skipped):
            console.print("  • Run [cyan]python download_example_data.py[/cyan] to download example inputs")
        
        if any("cosmos-predict" in str(item).lower() for item, _ in results.skipped):
            console.print("  • Ensure cosmos-predict2.5 is installed at /workspace/cosmos-predict2.5")
        
        if any("cosmos-transfer" in str(item).lower() for item, _ in results.skipped):
            console.print("  • Ensure cosmos-transfer2.5 is installed at /workspace/cosmos-transfer2.5")
        
        console.print()
    
    if len(results.failed) == 0:
        console.print("[bold green]All tests passed! ✨[/bold green]\n")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Tests interrupted by user[/yellow]\n")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]Unexpected error:[/bold red] {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

