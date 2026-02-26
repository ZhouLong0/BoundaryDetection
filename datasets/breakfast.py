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


class BreakfastDataset(Dataset):
    def __init__(
        self,
        base_dir: str = "data/Breakfast/frames",
        split_root: str = "data/Breakfast/splits",
        annotation_root: str = "data/Breakfast/groundTruth",
        mapping_path: str = "data/Breakfast/mapping.txt",
        mapping_root: str = "data/Breakfast/mapping",
        split_files: List[str] | None = None,
        chunk_duration: float = 2.0,
        num_frames_per_chunk: int = 4,
        fps: int = 15,
    ):
        """
        Dataset for Breakfast where each sample is a full video split into chunks.

        Frame folder example:
            data/Breakfast/frames/P03_cam01_P03_cereals/00001.jpg
        Split file example:
            data/Breakfast/splits/test.split1.bundle
        Ground truth example:
            data/Breakfast/groundTruth/P03_cam01_P03_cereals.txt
        """
        self.base_dir = base_dir
        self.split_root = split_root
        self.annotation_root = annotation_root
        self.mapping_path = mapping_path
        self.mapping_root = mapping_root
        self.chunk_duration = chunk_duration
        self.num_frames_per_chunk = num_frames_per_chunk
        self.fps = fps

        if split_files is None:
            split_files = [
                "test.split1.bundle",
                # "test.split2.bundle",
                # "test.split3.bundle",
                # "test.split4.bundle",
            ]
        self.split_files = split_files

        self.mapping = self._load_mapping(self.mapping_path)
        self.recipe_mappings = self._load_recipe_mappings()
        self.video_list = self._load_all_splits()

        if not self.video_list:
            warnings.warn("No videos were loaded. Check split files and frame paths.")

    def _load_mapping(self, mapping_file: str) -> Dict[str, int]:
        mapping = {}

        if not os.path.exists(mapping_file):
            warnings.warn(f"Mapping file not found: {mapping_file}")
            return mapping

        with open(mapping_file, "r") as f:
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

    def _load_recipe_mappings(self) -> Dict[str, Dict[str, int]]:
        recipe_maps: Dict[str, Dict[str, int]] = {}

        if not os.path.isdir(self.mapping_root):
            return recipe_maps

        for recipe in os.listdir(self.mapping_root):
            recipe_dir = os.path.join(self.mapping_root, recipe)
            if not os.path.isdir(recipe_dir):
                continue
            recipe_mapping_file = os.path.join(recipe_dir, "mapping.txt")
            if not os.path.exists(recipe_mapping_file):
                continue
            recipe_maps[recipe] = self._load_mapping(recipe_mapping_file)

        return recipe_maps

    def _infer_recipe_from_video_id(self, video_id: str) -> str:
        # Example: P03_cam01_P03_cereals -> cereals
        parts = video_id.split("_")
        if len(parts) < 4:
            return "unknown"
        return parts[-1]

    def _get_recipe_mapping(self, recipe: str) -> Dict[str, int]:
        # Prefer per-recipe mapping generated under data/Breakfast/mapping/<recipe>/mapping.txt
        # Fallback to global mapping.txt if recipe mapping is missing.
        return self.recipe_mappings.get(recipe, self.mapping)

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
                        "recipe": self._infer_recipe_from_video_id(video_id),
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

    def _get_annotation_info(self, recipe: str, video_id: str) -> Dict[str, Any]:
        annot_path = os.path.join(self.annotation_root, f"{video_id}.txt")

        if not os.path.exists(annot_path):
            return {"boundaries": [], "frame_labels": [], "frame_ids": []}

        with open(annot_path, "r") as f:
            frame_labels = [line.strip() for line in f.readlines() if line.strip()]

        boundaries = []
        for i in range(1, len(frame_labels)):
            if frame_labels[i] != frame_labels[i - 1]:
                boundaries.append(i)

        recipe_mapping = self._get_recipe_mapping(recipe)
        frame_ids = [recipe_mapping.get(label, -1) for label in frame_labels]

        return {
            "boundaries": [boundaries],
            "frame_labels": frame_labels,
            "frame_ids": frame_ids,
        }

    def get_list_steps_by_video_id(self, video_id: str, full: bool = False) -> List[str]:
        if full:
            return list(self.mapping.keys())
        recipe = None
        for video_meta in self.video_list:
            if video_meta["id"] == video_id:
                recipe = video_meta.get("recipe")
                break

        if recipe is None:
            recipe = self._infer_recipe_from_video_id(video_id)

        return list(self._get_recipe_mapping(recipe).keys())

    def get_all_steps(self) -> List[str]:
        """
        Returns all Breakfast steps from the global mapping file
        (same behavior as before recipe-specific mappings were introduced).
        """
        return list(self.mapping.keys())

    def __len__(self) -> int:
        return len(self.video_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_meta = self.video_list[idx]
        video_id = video_meta["id"]
        video_path = video_meta["path"]
        recipe = video_meta.get("recipe", self._infer_recipe_from_video_id(video_id))

        all_frame_paths, num_frames = get_frame_paths_from_folder(video_path)

        annot_info = self._get_annotation_info(recipe, video_id)
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
            "recipe": recipe,
        }


if __name__ == "__main__":
    dataset = BreakfastDataset()
    print(f"Loaded {len(dataset)} videos from test split bundles 1..4")
    print(f"Loaded {len(dataset.recipe_mappings)} recipe-specific mappings from {dataset.mapping_root}")
    print(f"Global step count: {len(dataset.get_all_steps())}")

    if len(dataset) > 0:
        sample = dataset[0]
        print("Sample video metadata:")
        print(f"Video ID: {sample['video_id']}")
        print(f"Recipe: {sample['recipe']}")
        print(f"Video Path: {sample['video_path']}")
        print(f"Number of Frames: {sample['num_frames']}")
        print(f"Chunks: {len(sample['chunks'])}")
        print(f"First chunk labels: {sample['chunk_class_names'][0] if sample['chunk_class_names'] else []}")

        recipe_steps = dataset.get_list_steps_by_video_id(sample['video_id'])
        all_steps = dataset.get_all_steps()
        print(f"Recipe-specific step count for {sample['recipe']}: {len(recipe_steps)}")
        print(f"First 10 recipe steps: {recipe_steps[:10]}")
        print(f"First 10 global steps: {all_steps[:10]}")

        # Check that recipe steps are a subset of global steps
        recipe_not_in_global = sorted(set(recipe_steps) - set(all_steps))
        if recipe_not_in_global:
            print(f"[WARN] Recipe steps missing from global mapping: {recipe_not_in_global}")
        else:
            print("[OK] Recipe-specific steps are consistent with global mapping")

        # Check frame-id mapping quality on first sample
        mapped_ids = [i for chunk in sample['chunk_class_ids'] for i in chunk]
        unknown_count = sum(1 for i in mapped_ids if i == -1)
        print(f"Mapped frame IDs in sampled chunks: {len(mapped_ids) - unknown_count}/{len(mapped_ids)}")
        if unknown_count > 0:
            print(f"[WARN] Unknown frame labels encountered: {unknown_count}")
        else:
            print("[OK] No unknown frame labels in sampled chunks")
