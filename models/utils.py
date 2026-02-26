import os
from PIL import Image
from typing import List, Tuple
#import matplotlib.pyplot as plt
import textwrap
import numpy as np

def get_frame_paths_from_folder(folder_path: str) -> Tuple[List[str], int]:
    """
    Get all image paths from a folder without loading them into memory.
    """
    image_paths = sorted(
        os.path.join(folder_path, fname)
        for fname in os.listdir(folder_path)
        if fname.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    
    num_frames = len(image_paths)
    return image_paths, num_frames

def load_all_frames_from_folder(
    folder_path: str,
) -> Tuple[List[Image.Image], List[str], int]:
    """
    Load all image frames from a folder.

    Args:
        folder_path (str): Path to the folder containing frame images.

    Returns:
        images (List[PIL.Image.Image]): List of loaded RGB images.
        image_paths (List[str]): Full paths to each image file, sorted.
        num_frames (int): Number of frames loaded.
    """
    image_paths = sorted(
        os.path.join(folder_path, fname)
        for fname in os.listdir(folder_path)
        if fname.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    images: List[Image.Image] = []

    for path in image_paths:
        # Convert to RGB to ensure consistent 3-channel input
        img = Image.open(path).convert("RGB")
        images.append(img)

    num_frames = len(images)

    return images, image_paths, num_frames


def build_chunks_from_full_video(
    total_frames: int,
    chunk_size: int = 16,
    stride: int | None = None,
):
    """
    Returns list of (start_idx, end_idx) based on full length
    """
    if stride is None:
        stride = chunk_size

    chunks = []
    for start in range(0, total_frames, stride):
        end = start + chunk_size
        if end > total_frames:
            end = total_frames
        chunks.append((start, end))

        if end == total_frames:
            break

    return chunks


def plot_timeline_filmstrip(
    frames,
    timeline,
    samples=8,
    max_chunks=100
):
    """
    Visualizes the timeline with filmstrips + validity check reasoning.

    Args:
        frames: List[PIL.Image.Image] of length T
        timeline: List of tuples (start, end, description, validity_answer)
        samples: Number of frames per filmstrip
        max_chunks: Safety limit for number of timeline rows
    """

    # Safety limit
    if len(timeline) > max_chunks:
        print(f"Showing first {max_chunks} chunks (out of {len(timeline)})...")
        display_timeline = timeline[:max_chunks]
    else:
        display_timeline = timeline

    # Create figure
    fig, axes = plt.subplots(
        len(display_timeline),
        1,
        figsize=(15, 4.5 * len(display_timeline))
    )

    if len(display_timeline) == 1:
        axes = [axes]

    active_description = "No description available"
    num_frames = len(frames)

    for i, (start, end, desc, reason) in enumerate(display_timeline):
        ax = axes[i]

        # --- 1. Filmstrip Generation ---
        real_end = min(end, num_frames)
        if real_end <= start:
            indices = [min(start, num_frames - 1)] * samples
        else:
            indices = np.linspace(start, real_end - 1, samples, dtype=int)

        selected_frames = []
        for idx in indices:
            img = frames[idx]
            if not isinstance(img, Image.Image):
                raise TypeError("All frames must be PIL.Image.Image")
            selected_frames.append(np.array(img))

        # Concatenate frames horizontally
        filmstrip = np.concatenate(selected_frames, axis=1)
        ax.imshow(filmstrip)

        # --- 2. Logic for Description vs Reason ---
        status_tag = ""
        main_color = "black"

        is_skipped = (desc == "skipped") or (reason and "YES" in str(reason).upper())

        if is_skipped:
            display_desc = f"(Scene Continues) {active_description}"
            status_tag = " [NO CHANGE]"
            main_color = "#555555"
        else:
            active_description = desc
            display_desc = desc
            status_tag = " [NEW EVENT]"
            main_color = "black"

        # --- 3. Text Formatting ---
        wrapped_desc = "\n".join(textwrap.wrap(display_desc, width=120))

        reason_text = ""
        if reason:
            clean_reason = str(reason).strip()
            wrapped_reason = "\n".join(
                textwrap.wrap(f"Model Reasoning: {clean_reason}", width=120)
            )
            reason_text = f"\n\n{wrapped_reason}"

        final_label = f"{wrapped_desc}{reason_text}"

        # --- 4. Plotting Labels ---
        ax.set_title(
            f"Chunk {i}: Frames {start} - {end}{status_tag}",
            loc="left",
            fontsize=10,
            fontweight="bold",
            color=main_color
        )

        ax.set_xlabel(
            final_label,
            fontsize=11,
            labelpad=8,
            color=main_color,
            loc="left"
        )

        ax.set_yticks([])
        ax.set_xticks([])

    plt.tight_layout()
    plt.show()