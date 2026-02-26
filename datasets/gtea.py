import os
import numpy as np
import warnings
from typing import List, Dict, Any
from torch.utils.data import Dataset
from PIL import Image
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models.utils import get_frame_paths_from_folder



class GTEADataset(Dataset):
    def __init__(
        self,
        base_dir: str = "data/GTEA/frames",
        split_root: str = "data/GTEA/data/gtea/splits",
        annotation_root: str = "data/GTEA/data/gtea/groundTruth",
        mapping_path: str = "data/GTEA/data/gtea/mapping.txt",
        split_files: List[str] | None = None,
        chunk_duration: float = 2.0,
        num_frames_per_chunk: int = 4,
        fps: int = 15,
    ):
        """
        Dataset for GTEA where each sample is a full video split into chunks.

        Returns the same keys as EgoperDataset in __getitem__.
        """
        self.base_dir = base_dir
        self.split_root = split_root
        self.annotation_root = annotation_root
        self.mapping_path = mapping_path
        self.chunk_duration = chunk_duration
        self.num_frames_per_chunk = num_frames_per_chunk
        self.fps = fps

        if split_files is None:
            split_files = [
                "test.split1.bundle",
                "test.split2.bundle",
                "test.split3.bundle",
                "test.split4.bundle",
            ]
        self.split_files = split_files

        self.mapping = self._load_mapping()
        self.video_list = self._load_all_splits()

        if not self.video_list:
            warnings.warn("No videos were loaded. Check split files and frame paths.")

    def _load_mapping(self) -> Dict[str, int]:
        mapping = {}

        if not os.path.exists(self.mapping_path):
            warnings.warn(f"Mapping file not found: {self.mapping_path}")
            return mapping

        with open(self.mapping_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue

                idx_str, label = parts
                try:
                    mapping[label] = int(idx_str)
                except ValueError:
                    continue

        return mapping

    def _load_split(self, split_file: str) -> List[Dict[str, str]]:
        videos = []
        split_path = os.path.join(self.split_root, split_file)

        if not os.path.exists(split_path):
            warnings.warn(f"Split file not found: {split_path}")
            return videos

        with open(split_path, "r") as f:
            for line in f:
                filename = line.strip()
                if not filename:
                    continue

                video_id = filename.replace(".txt", "")
                videos.append(
                    {
                        "id": video_id,
                        "path": os.path.join(self.base_dir, video_id),
                    }
                )

        return videos

    def _load_all_splits(self) -> List[Dict[str, str]]:
        video_meta = []
        seen = set()

        for split_file in self.split_files:
            split_videos = self._load_split(split_file)
            for v in split_videos:
                vid = v["id"]
                if vid in seen:
                    continue
                seen.add(vid)
                video_meta.append(v)

        return video_meta

    def _get_annotation_info(self, video_id: str) -> Dict[str, Any]:
        annot_path = os.path.join(self.annotation_root, f"{video_id}.txt")

        if not os.path.exists(annot_path):
            return {"boundaries": [], "frame_labels": [], "frame_ids": []}

        with open(annot_path, "r") as f:
            frame_labels = [line.strip() for line in f.readlines() if line.strip()]

        boundaries = []
        for i in range(1, len(frame_labels)):
            if frame_labels[i] != frame_labels[i - 1]:
                boundaries.append(i)

        frame_ids = [self.mapping.get(label, -1) for label in frame_labels]

        return {
            "boundaries": [boundaries],
            "frame_labels": frame_labels,
            "frame_ids": frame_ids,
        }

    def get_list_steps_by_video_id(self, video_id: str) -> List[str]:
        # In GTEA there is a single global mapping (no recipe split)
        return list(self.mapping.keys())

    def __len__(self) -> int:
        return len(self.video_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_meta = self.video_list[idx]
        video_id = video_meta["id"]
        video_path = video_meta["path"]

        all_frame_paths, num_frames = get_frame_paths_from_folder(video_path)

        annot_info = self._get_annotation_info(video_id)
        gt_boundaries = annot_info["boundaries"]
        dense_frame_ids = annot_info["frame_ids"]
        dense_frame_labels = annot_info["frame_labels"]

        frames_per_window = int(self.chunk_duration * self.fps)
        chunk_ranges = []
        for start in range(0, num_frames, frames_per_window):
            end = min(start + frames_per_window, num_frames)
            if end - start < frames_per_window:
                break
            chunk_ranges.append((start, end))

        frame_chunks = []
        chunk_labels_boundary = []
        chunk_class_ids = []
        chunk_class_names = []

        boundaries_set = set(gt_boundaries[0]) if gt_boundaries else set()

        for start, end in chunk_ranges:
            if (end - start) < self.num_frames_per_chunk:
                indices = np.linspace(start, end - 1, end - start).astype(int)
            else:
                indices = np.linspace(start, end - 1, self.num_frames_per_chunk).astype(int)

            current_chunk_frames = []
            for i in indices:
                img_path = all_frame_paths[i]
                img = Image.open(img_path).convert("RGB")
                current_chunk_frames.append(img)

            frame_chunks.append(current_chunk_frames)

            has_boundary = any(start <= b < end for b in boundaries_set)
            chunk_labels_boundary.append(1 if has_boundary else 0)

            current_chunk_ids = []
            current_chunk_names = []
            for i in indices:
                if i < len(dense_frame_ids):
                    current_chunk_ids.append(dense_frame_ids[i])
                    current_chunk_names.append(dense_frame_labels[i])
                else:
                    current_chunk_ids.append(-1)
                    current_chunk_names.append("Unknown")

            chunk_class_ids.append(current_chunk_ids)
            chunk_class_names.append(current_chunk_names)

        chunk_labels_boundary = [chunk_labels_boundary]

        return {
            "video_id": video_id,
            "video_path": video_path,
            "num_frames": num_frames,
            "fps": self.num_frames_per_chunk / self.chunk_duration,
            "gt_boundaries": gt_boundaries,
            "chunks": frame_chunks,
            "chunk_ranges": chunk_ranges,
            "chunk_labels": chunk_labels_boundary,
            "chunk_class_ids": chunk_class_ids,
            "chunk_class_names": chunk_class_names,
            "recipe": "gtea",
        }


if __name__ == "__main__":
    dataset = GTEADataset()
    print(f"Loaded {len(dataset)} videos from test split bundles 1..4")

    if len(dataset) > 0:
        sample = dataset[0]
        print("Sample video metadata:")
        print(f"Video ID: {sample['video_id']}")
        print(f"Video Path: {sample['video_path']}")
        print(f"Number of Frames: {sample['num_frames']}")
        print(f"Chunks: {len(sample['chunks'])}")
        print(f"First chunk labels: {sample['chunk_class_names'][0] if sample['chunk_class_names'] else []}")
        print(f"First chunk contents: {sample['chunks'][0][0] if sample['chunks'] else []}")
