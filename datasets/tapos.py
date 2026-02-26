import os
from typing import List, Dict, Any

from torch.utils.data import Dataset

from models.utils import (
    load_all_frames_from_folder,
    build_chunks_from_full_video,
)
from .annot_utils import load_gt_annotations, get_video_gt_info



class TAPOSDataset(Dataset):
    """
    TAPOS dataset that returns an entire video and its chunks per sample.
    """

    def __init__(
        self,
        base_dir: str,
        split_file: str,
        gt_annotation_path: str,
        chunk_size: int = 16,
    ):
        """
        Args:
            base_dir: Root directory containing video frame folders
            split_file: Text file with one video ID per line
            gt_annotation_path: Path to TAPOS GT annotations
            chunk_size: Number of frames per chunk
        """
        self.base_dir = base_dir
        self.chunk_size = chunk_size

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
        """
        Get video data and its chunks.
        Args:
            idx: Index of the video sample
        Returns:
            dict with keys:
                - video_id (str)
                - video_path (str): directory path to video frames
                - num_frames (int)
                - fps (int or float or None)
                - gt_boundaries (list of list): Frame indices of substage boundaries for each annotator (1 annotator in TAPOS)
                - chunks (list of lists): Each inner list contains frames for a chunk
                - chunk_ranges (list of tuples): Each tuple is (start_frame, end_frame)
                - chunk_labels (list of int): 1 if chunk contains a boundary, else 0
        """
        video_path = self.video_paths[idx]
        video_id = os.path.basename(video_path)

        # Load GT info
        video_info = get_video_gt_info(self.full_annotations, video_path)
        fps = video_info["fps"]
        gt_boundaries = video_info["boundaries"]

        # Load all frames once
        frames, frame_files, num_frames = load_all_frames_from_folder(video_path)

        # Build chunks
        chunks_idx = build_chunks_from_full_video(
            total_frames=num_frames,
            chunk_size=self.chunk_size,
        )

        # Slice frames into chunks
        frame_chunks = [
            frames[start:end] for (start, end) in chunks_idx
        ]

        # Build chunk labels
        chunk_labels = []
        for annotator_boundaries in gt_boundaries:
            annotator_labels = []
            boundaries_set = set(annotator_boundaries)

            for start, end in chunks_idx:
                has_boundary = any(start <= b < end for b in boundaries_set)
                annotator_labels.append(1 if has_boundary else 0)

            chunk_labels.append(annotator_labels)

        return {
            "video_id": video_id,
            "video_path": video_path,
            "num_frames": num_frames,
            "fps": fps,
            "gt_boundaries": gt_boundaries,
            "chunks": frame_chunks,
            "chunk_ranges": chunks_idx,
            "chunk_labels": chunk_labels,
        }
    
