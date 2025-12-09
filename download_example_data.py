#!/usr/bin/env python3
"""
Download example videos and images from cosmos-predict2.5 and cosmos-transfer2.5 README files.

This script downloads all the example media files along with their prompts
to data/inputs/ for testing inference.
"""

import os
from pathlib import Path
import urllib.request
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn

console = Console()


def download_file(url: str, output_path: Path, description: str):
    """Download a file with progress bar."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
        ) as progress:
            task = progress.add_task(f"[cyan]{description}", total=None)
            
            def reporthook(blocknum, blocksize, totalsize):
                if totalsize > 0:
                    progress.update(task, total=totalsize, completed=blocknum * blocksize)
            
            urllib.request.urlretrieve(url, output_path, reporthook=reporthook)
        
        console.print(f"[green]✓[/green] Downloaded: {output_path.name}")
        return True
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to download {description}: {e}")
        return False


def save_prompt(prompt: str, output_path: Path):
    """Save prompt to a text file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(prompt.strip())
    console.print(f"[green]✓[/green] Saved prompt: {output_path.name}")


def main():
    """Download all example data from README files."""
    console.print("\n[bold cyan]Cosmos Example Data Downloader[/bold cyan]")
    console.print("Downloading videos and images from README examples...\n")
    
    base_dir = Path("data/inputs")
    
    # ===== Cosmos-Predict2.5 Examples =====
    console.print("[bold yellow]Downloading Predict2.5 Examples[/bold yellow]")
    
    # Image2World example
    console.print("\n[bold]1. Image2World Example[/bold]")
    image2world_dir = base_dir / "predict2.5" / "image2world"
    
    image2world_prompt = """A nighttime city bus terminal gradually shifts from stillness to subtle movement. At first, multiple double-decker buses are parked under the glow of overhead lights, with a central bus labeled '87D' facing forward and stationary. As the video progresses, the bus in the middle moves ahead slowly, its headlights brightening the surrounding area and casting reflections onto adjacent vehicles. The motion creates space in the lineup, signaling activity within the otherwise quiet station. It then comes to a smooth stop, resuming its position in line. Overhead signage in Chinese characters remains illuminated, enhancing the vibrant, urban night scene."""
    
    save_prompt(image2world_prompt, image2world_dir / "prompt.txt")
    download_file(
        "https://github.com/user-attachments/assets/c855f468-0577-475d-a2bb-5673b9d8ae91",
        image2world_dir / "input_image.png",
        "Image2World input image"
    )
    download_file(
        "https://github.com/user-attachments/assets/a233567b-9eb4-405a-ab36-c0bf902d2988",
        image2world_dir / "output_reference.mp4",
        "Image2World output video (reference)"
    )
    
    # Video2World example
    console.print("\n[bold]2. Video2World Example[/bold]")
    video2world_dir = base_dir / "predict2.5" / "video2world"
    
    video2world_prompt = """A robotic arm, primarily white with black joints and cables, is shown in a clean, modern indoor setting with a white tabletop. The arm, equipped with a gripper holding a small, light green pitcher, is positioned above a clear glass containing a reddish-brown liquid and a spoon. The robotic arm is in the process of pouring a transparent liquid into the glass. To the left of the pitcher, there is an opened jar with a similar reddish-brown substance visible through its transparent body. In the background, a vase with white flowers and a brown couch are partially visible, adding to the contemporary ambiance. The lighting is bright, casting soft shadows on the table. The robotic arm's movements are smooth and controlled, demonstrating precision in its task. As the video progresses, the robotic arm completes the pour, leaving the glass half-filled with the reddish-brown liquid. The jar remains untouched throughout the sequence, and the spoon inside the glass remains stationary. The other robotic arm on the right side also stays stationary throughout the video. The final frame captures the robotic arm with the pitcher finishing the pour, with the glass now filled to a higher level, while the pitcher is slightly tilted but still held securely by the gripper."""
    
    save_prompt(video2world_prompt, video2world_dir / "prompt.txt")
    download_file(
        "https://github.com/user-attachments/assets/ddca366e-b30f-44bb-9def-b4a8386d8d23",
        video2world_dir / "input_video.mp4",
        "Video2World input video"
    )
    download_file(
        "https://github.com/user-attachments/assets/62c0800d-036a-4dbc-b0a6-199ee25d8e31",
        video2world_dir / "output_reference.mp4",
        "Video2World output video (reference)"
    )
    
    # ===== Cosmos-Transfer2.5 Examples =====
    console.print("\n[bold yellow]Downloading Transfer2.5 Examples[/bold yellow]")
    
    # Sim2Real example
    console.print("\n[bold]3. Sim2Real Augmentation Example[/bold]")
    sim2real_dir = base_dir / "transfer2.5" / "sim2real"
    
    sim2real_prompt = """A contemporary luxury kitchen with marble tabletops. window with beautiful sunset outside. There is an esspresso coffee maker on the table in front of the white robot arm. Robot arm interacts with a coffee cup and coffee maker on the kitchen table."""
    
    save_prompt(sim2real_prompt, sim2real_dir / "prompt.txt")
    download_file(
        "https://github.com/user-attachments/assets/20d63162-0fd5-483a-a306-7b8021df5ed9",
        sim2real_dir / "input_video.mp4",
        "Sim2Real input video"
    )
    download_file(
        "https://github.com/user-attachments/assets/131ffe81-cca0-44cd-8547-7b0e49d5253f",
        sim2real_dir / "control_depth.mp4",
        "Sim2Real control depth"
    )
    download_file(
        "https://github.com/user-attachments/assets/e4dd3b80-4696-4930-8b05-6d41e37974c2",
        sim2real_dir / "control_seg.mp4",
        "Sim2Real control segmentation"
    )
    download_file(
        "https://github.com/user-attachments/assets/5a816f4d-fdc3-4939-b2b9-141c6ee64d2b",
        sim2real_dir / "control_normal.mp4",
        "Sim2Real control normal"
    )
    download_file(
        "https://github.com/user-attachments/assets/56f76740-ea36-4916-9e94-c983d6b84d28",
        sim2real_dir / "output_reference.mp4",
        "Sim2Real output video (reference)"
    )
    
    # Real2Real example
    console.print("\n[bold]4. Real2Real Augmentation Example[/bold]")
    real2real_dir = base_dir / "transfer2.5" / "real2real"
    
    real2real_prompt = """Dashcam video, driving through a modern urban environment, winter with heavy snow storm, trees and sidewalks covered in snow."""
    
    save_prompt(real2real_prompt, real2real_dir / "prompt.txt")
    download_file(
        "https://github.com/user-attachments/assets/4705c192-b8c6-4ba3-af7f-fd968c4a3eeb",
        real2real_dir / "input_video.mp4",
        "Real2Real input video"
    )
    download_file(
        "https://github.com/user-attachments/assets/ba92fa5d-2972-463e-af2e-a637a810a463",
        real2real_dir / "control_depth.mp4",
        "Real2Real control depth"
    )
    download_file(
        "https://github.com/user-attachments/assets/f8e6c351-78b5-4bd6-949b-e1845aa19f63",
        real2real_dir / "control_seg.mp4",
        "Real2Real control segmentation"
    )
    download_file(
        "https://github.com/user-attachments/assets/7edf3f46-c4da-403f-b630-d8853a165602",
        real2real_dir / "control_normal.mp4",
        "Real2Real control normal"
    )
    download_file(
        "https://github.com/user-attachments/assets/ba59f926-c4c2-4232-bdbf-392c53f29a97",
        real2real_dir / "control_edge.mp4",
        "Real2Real control edge"
    )
    download_file(
        "https://github.com/user-attachments/assets/8e62af23-3ca4-4e72-97fe-7a337a31d306",
        real2real_dir / "output_reference.mp4",
        "Real2Real output video (reference)"
    )
    
    console.print("\n[bold green]✓ Download complete![/bold green]")
    console.print(f"\nAll files saved to: [cyan]{base_dir.absolute()}[/cyan]")
    
    # Print directory tree
    console.print("\n[bold]Directory Structure:[/bold]")
    console.print("data/inputs/")
    console.print("├── predict2.5/")
    console.print("│   ├── image2world/")
    console.print("│   │   ├── input_image.png")
    console.print("│   │   ├── prompt.txt")
    console.print("│   │   └── output_reference.mp4")
    console.print("│   └── video2world/")
    console.print("│       ├── input_video.mp4")
    console.print("│       ├── prompt.txt")
    console.print("│       └── output_reference.mp4")
    console.print("└── transfer2.5/")
    console.print("    ├── sim2real/")
    console.print("    │   ├── input_video.mp4")
    console.print("    │   ├── prompt.txt")
    console.print("    │   ├── control_depth.mp4")
    console.print("    │   ├── control_seg.mp4")
    console.print("    │   ├── control_normal.mp4")
    console.print("    │   └── output_reference.mp4")
    console.print("    └── real2real/")
    console.print("        ├── input_video.mp4")
    console.print("        ├── prompt.txt")
    console.print("        ├── control_depth.mp4")
    console.print("        ├── control_seg.mp4")
    console.print("        ├── control_normal.mp4")
    console.print("        ├── control_edge.mp4")
    console.print("        └── output_reference.mp4")


if __name__ == "__main__":
    main()

