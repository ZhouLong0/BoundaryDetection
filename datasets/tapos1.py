import os
from typing import List, Dict, Any

from torch.utils.data import Dataset

from models.utils import (
    load_all_frames_from_folder,
    build_chunks_from_full_video,
)
from .annot_utils import load_gt_annotations, get_video_gt_info
import numpy as np
import warnings

class TaposDataset(Dataset):
    def __init__(
        self,
        base_dir: str,
        split_file: str,
        gt_annotation_path: str,
        chunk_duration: float = 1.0,  
        num_frames_per_chunk: int = 8, 
    ):
        self.base_dir = base_dir
        self.chunk_duration = chunk_duration
        self.num_frames_per_chunk = num_frames_per_chunk
        self.video_paths = self._load_split(split_file)
        self.full_annotations = load_gt_annotations(gt_annotation_path)


    def _load_split(self, split_file: str) -> List[str]:
        video_paths = []
        with open(split_file, "r") as f:
            for line in f:
                video_id = line.strip()
                if not video_id:
                    continue
                video_paths.append(os.path.join(self.base_dir, video_id))
        return video_paths


    def __len__(self) -> int:
        return len(self.video_paths)


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_path = self.video_paths[idx]
        video_id = os.path.basename(video_path)

        video_info = get_video_gt_info(self.full_annotations, video_path)
        fps = video_info["fps"]
        gt_boundaries = video_info["boundaries"]

        frames, _, num_frames = load_all_frames_from_folder(video_path)

        # 1. Calculate how many frames are in one "time chunk"
        # Example: 30 fps * 2 seconds = 60 frames per chunk window
        frames_per_chunk_window = int(self.chunk_duration * int(fps))

        # 2. Build the windows (start/end indices)
        # We slide by the window size (no overlap here, but you could add a stride)
        # skip the last incomplete window
        chunk_ranges = []
        for start in range(0, num_frames, frames_per_chunk_window):
            end = min(start + frames_per_chunk_window, num_frames)
            if end - start < frames_per_chunk_window:
                break  # Skip incomplete window
            chunk_ranges.append((start, end))

        # 3. Sample exactly N frames from each window
        frame_chunks = []
        for start, end in chunk_ranges:
            if (end - start) + 1 < self.num_frames_per_chunk:
                # If the window has fewer frames than we want to sample, just take them all
                sampled_chunk = frames[start:end]
                warnings.warn(f"Chunk from frame {start} to {end} has only {end - start} frames, less than the desired {self.num_frames_per_chunk}. Taking all available frames.")
            else:
                window_indices = np.linspace(start, end - 1, self.num_frames_per_chunk).astype(int)
                # Access frames (assuming frames is a list or tensor)
                sampled_chunk = [frames[i] for i in window_indices]
            frame_chunks.append(sampled_chunk)

        # 4. Build labels (remains similar, checking if boundary falls in range)
        chunk_labels = []
        for annotator_boundaries in gt_boundaries:
            boundaries_set = set(annotator_boundaries)
            annotator_labels = [
                1 if any(start <= b < end for b in boundaries_set) else 0
                for start, end in chunk_ranges
            ]
            chunk_labels.append(annotator_labels)

        return {
            "video_id": video_id,
            "video_path": video_path,
            "num_frames": num_frames,
            "fps": self.num_frames_per_chunk / self.chunk_duration,
            "gt_boundaries": gt_boundaries,
            "chunks": frame_chunks,
            "chunk_ranges": chunk_ranges,
            "chunk_labels": chunk_labels,
        }