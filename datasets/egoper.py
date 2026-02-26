import sys
import os
import numpy as np
import warnings
from typing import List, Dict, Any
from torch.utils.data import Dataset

# 1. Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from models.utils import load_all_frames_from_folder, get_frame_paths_from_folder
from PIL import Image

def read_file(path):
        with open(path, 'r') as f:
            content = f.read()
            f.close()
        return content


class EgoperDataset(Dataset):
    def __init__(
        self,
        base_dir: str,                # Path to extracted frames
        split_root: str,              # Root path to splits
        annotation_root: str,         # Path to frame annotations
        mapping_root: str,            # Path to idx_action_mapping
        recipes: List[str],           # List of recipes
        split_name: str = "test.split1.bundle",
        chunk_duration: float = 1.0,  
        num_frames_per_chunk: int = 8, 
        fps: int = 10,
    ):
        self.base_dir = base_dir
        self.annotation_root = annotation_root
        self.mapping_root = mapping_root
        self.chunk_duration = chunk_duration
        self.num_frames_per_chunk = num_frames_per_chunk
        self.fps = fps
        self.recipes = [r.lower() for r in recipes]

        # 1. Load Mappings (Action Name -> ID) for each recipe
        self.mappings = self._load_all_mappings()

        # 2. Load Videos
        self.video_list = []
        for recipe in self.recipes:
            split_path = os.path.join(split_root, recipe, split_name)
            if not os.path.exists(split_path):
                warnings.warn(f"Split file not found for {recipe}: {split_path}")
                continue
            self.video_list.extend(self._load_split(split_path, recipe))

        if not self.video_list:
            warnings.warn("No videos were loaded. Check your split paths and recipe names.")

    def _load_all_mappings(self) -> Dict[str, Dict[str, int]]:
        """
        Loads mappings for all requested recipes.
        """
        mappings = {}
        for recipe in self.recipes:
            mapping_path = os.path.join(self.mapping_root, f"{recipe.capitalize()}.txt")
            
            if not os.path.exists(mapping_path):
                warnings.warn(f"Mapping file not found for {recipe}: {mapping_path}")
                mappings[recipe] = {}
                continue
            
            recipe_map = {}
            with open(mapping_path, 'r') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        idx_str, label = line.strip().split('|', 1)
                        recipe_map[label] = int(idx_str)
                    except ValueError:
                        continue 
            mappings[recipe] = recipe_map
        return mappings

    def _load_split(self, split_file: str, recipe: str) -> List[Dict[str, str]]:
        videos = []
        with open(split_file, "r") as f:
            for line in f:
                filename = line.strip()
                if not filename: continue
                video_id = filename.replace(".txt", "")
                videos.append({
                    "id": video_id,
                    "recipe": recipe,
                    "path": os.path.join(self.base_dir, video_id)
                })
        return videos

    def _get_annotation_info(self, recipe: str, video_id: str) -> Dict[str, Any]:
        """
        Reads the annotation file.
        Returns:
            - boundaries: List[List[int]] (indices where action changes)
            - frame_labels: List[str] (raw text label for every frame)
            - frame_ids: List[int] (mapped ID for every frame)
        """
        annot_path = os.path.join(self.annotation_root, recipe, f"{video_id}.txt")
        
        # Default empty return if missing
        if not os.path.exists(annot_path):
            return {"boundaries": [], "frame_labels": [], "frame_ids": []}

        with open(annot_path, 'r') as f:
            frame_labels = [line.strip() for line in f.readlines()]
        
        # 1. Detect Boundaries
        boundaries = []
        for i in range(1, len(frame_labels)):
            if frame_labels[i] != frame_labels[i-1]:
                boundaries.append(i)

        # 2. Map to IDs
        recipe_map = self.mappings.get(recipe, {})
        frame_ids = []
        for label in frame_labels:
            # Default to -1 if label not found in mapping
            frame_ids.append(recipe_map.get(label, -1))

        return {
            "boundaries": [boundaries], # Wrapped in list for consistency
            "frame_labels": frame_labels,
            "frame_ids": frame_ids
        }
    
    def get_gt_annotations(self, video_id: str) -> Dict[str, Any]:
        ground_truth_path = "./data/"+dataset+"/groundTruth/"
        gt_file = ground_truth_path + video_id + ".txt"
        gt_content = read_file(gt_file).split('\n')[0:-1]
        return gt_content
    
    def get_list_steps_of_recipe(self, recipe: str) -> List[str]:
        return_list = []
        recipe_dict = self.mappings[recipe]
        for key in recipe_dict.keys():
            return_list.append(key)
        return return_list

    def get_list_steps_by_video_id(self, video_id: str, full: bool = False) -> List[str]:
        """
        Given a video_id, returns the list of steps for its recipe.
        Returns an empty list if the video_id is not found.
        """
        recipe = None
        for video_meta in self.video_list:
            if video_meta["id"] == video_id:
                recipe = video_meta["recipe"]
                break

        if recipe is None:
            warnings.warn(f"Video ID not found: {video_id}")
            return []

        return self.get_list_steps_of_recipe(recipe)



    def __len__(self) -> int:
        return len(self.video_list)

    # def __getitem__(self, idx: int) -> Dict[str, Any]:
    #     video_meta = self.video_list[idx]
    #     video_id = video_meta["id"]
    #     video_path = video_meta["path"]
    #     recipe = video_meta["recipe"]

    #     # 1. Load Frames
    #     frames, _, num_frames = load_all_frames_from_folder(video_path)

    #     # 2. Get Annotation Info (Boundaries + Labels)
    #     annot_info = self._get_annotation_info(recipe, video_id)
    #     gt_boundaries = annot_info["boundaries"]
    #     dense_frame_ids = annot_info["frame_ids"]
    #     dense_frame_labels = annot_info["frame_labels"]

    #     # 3. Build Windows
    #     frames_per_window = int(self.chunk_duration * self.fps)
    #     chunk_ranges = []
    #     for start in range(0, num_frames, frames_per_window):
    #         end = min(start + frames_per_window, num_frames)
    #         if end - start < frames_per_window:
    #             break
    #         chunk_ranges.append((start, end))

    #     # 4. Process Chunks (Sample Frames + Extract Labels)
    #     frame_chunks = []
    #     chunk_labels_boundary = [] # Binary boundary labels
    #     chunk_class_ids = []       # List[List[int]]: Action IDs for every frame in chunk
    #     chunk_class_names = []     # List[List[str]]: Action Names for every frame in chunk

    #     # Helper for boundary labels
    #     boundaries_set = set(gt_boundaries[0]) if gt_boundaries else set()

    #     for start, end in chunk_ranges:
    #         # A. Calculate Indices
    #         if (end - start) < self.num_frames_per_chunk:
    #             indices = np.linspace(start, end - 1, end - start).astype(int)
    #         else:
    #             indices = np.linspace(start, end - 1, self.num_frames_per_chunk).astype(int)
            
    #         # B. Sample Frames
    #         frame_chunks.append([frames[i] for i in indices])

    #         # C. Boundary Label (Binary: does this chunk contain a boundary?)
    #         has_boundary = any(start <= b < end for b in boundaries_set)
    #         chunk_labels_boundary.append(1 if has_boundary else 0)

    #         # D. Class Labels (Get label for EACH sampled frame)
    #         current_chunk_ids = []
    #         current_chunk_names = []
            
    #         for i in indices:
    #             # Check bounds (annotation file might be shorter than video frames)
    #             if i < len(dense_frame_ids):
    #                 current_chunk_ids.append(dense_frame_ids[i])
    #                 current_chunk_names.append(dense_frame_labels[i])
    #             else:
    #                 # Fallback
    #                 current_chunk_ids.append(-1)
    #                 current_chunk_names.append("Unknown")
            
    #         chunk_class_ids.append(current_chunk_ids)
    #         chunk_class_names.append(current_chunk_names)

    #     # Wrap boundary labels to list of lists (for multi-annotator compatibility)
    #     chunk_labels_boundary = [chunk_labels_boundary]

    #     return {
    #         "video_id": video_id,
    #         "video_path": video_path,
    #         "num_frames": num_frames,
    #         "fps": self.num_frames_per_chunk / self.chunk_duration,
    #         "gt_boundaries": gt_boundaries,
    #         "chunks": frame_chunks,
    #         "chunk_ranges": chunk_ranges,
    #         "chunk_labels": chunk_labels_boundary, # List of boundary labels for each chunk (0/1) (List[List[int]]) where first list is for annotators
    #         "chunk_class_ids": chunk_class_ids,    # List[List[int]]
    #         "chunk_class_names": chunk_class_names # List[List[str]]
    #     }
        

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_meta = self.video_list[idx]
        video_id = video_meta["id"]
        video_path = video_meta["path"]
        recipe = video_meta["recipe"]

        # 1. Get Frame Paths (Lightweight: strings only, no images yet)
        all_frame_paths, num_frames = get_frame_paths_from_folder(video_path)

        # 2. Get Annotation Info
        annot_info = self._get_annotation_info(recipe, video_id)
        gt_boundaries = annot_info["boundaries"]
        dense_frame_ids = annot_info["frame_ids"]
        dense_frame_labels = annot_info["frame_labels"]

        # 3. Build Windows
        frames_per_window = int(self.chunk_duration * self.fps)
        chunk_ranges = []
        for start in range(0, num_frames, frames_per_window):
            end = min(start + frames_per_window, num_frames)
            if end - start < frames_per_window:
                break
            chunk_ranges.append((start, end))

        # 4. Process Chunks
        frame_chunks = []
        chunk_labels_boundary = [] 
        chunk_class_ids = []       
        chunk_class_names = []     

        boundaries_set = set(gt_boundaries[0]) if gt_boundaries else set()

        for start, end in chunk_ranges:
            # A. Calculate Indices
            if (end - start) < self.num_frames_per_chunk:
                indices = np.linspace(start, end - 1, end - start).astype(int)
            else:
                indices = np.linspace(start, end - 1, self.num_frames_per_chunk).astype(int)
            
            # B. Sample Frames (LAZY LOADING HAPPENS HERE)
            current_chunk_frames = []
            for i in indices:
                # Load only the specific frames selected by linspace
                img_path = all_frame_paths[i]
                img = Image.open(img_path).convert("RGB")
                current_chunk_frames.append(img)

            frame_chunks.append(current_chunk_frames)

            # C. Boundary Label
            has_boundary = any(start <= b < end for b in boundaries_set)
            chunk_labels_boundary.append(1 if has_boundary else 0)

            # D. Class Labels
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
            "recipe": recipe
        }


if __name__ == "__main__":
    # --- CONFIGURATION ---
    BASE_DIR = "data/Egoper/extracted_frames"
    SPLIT_ROOT = "data/Egoper/splits"
    ANNOT_ROOT = "data/Egoper/frame_annotations"
    MAPPING_ROOT = "data/Egoper/idx_action_mapping" 
    
    RECIPES = ["coffee", "pinwheels", "oatmeal", "quesadilla", "tea"]

    print("Initializing EgoperDataset...")
    
    dataset = EgoperDataset(
        base_dir=BASE_DIR,
        split_root=SPLIT_ROOT,
        annotation_root=ANNOT_ROOT,
        mapping_root=MAPPING_ROOT,
        recipes=RECIPES,
        split_name="test.split1.bundle",
        chunk_duration=1.0,
        num_frames_per_chunk=8,
        fps=10
    )

    print(f"Dataset initialized with {len(dataset)} videos.")

    if len(dataset) > 0:
        sample = dataset[0]
        print("\n--- Sample Item ---")
        print(f"Video ID: {sample['video_id']}")
        print(f"Total Chunks: {len(sample['chunks'])}")
        
        # Verify the list-of-lists structure
        if len(sample['chunks']) > 0:
            # Inspect the first chunk
            first_chunk_frames = sample['chunks'][0]
            first_chunk_ids = sample['chunk_class_ids'][0]
            first_chunk_names = sample['chunk_class_names'][0]
            
            print("\n--- First Chunk Details ---")
            print(f"Number of frames: {len(first_chunk_frames)}")
            print(f"Number of labels: {len(first_chunk_ids)}")
            print(f"Frame IDs: {first_chunk_ids}")
            print(f"Frame Names: {first_chunk_names}")
            
            # Verify consistency
            if len(first_chunk_frames) == len(first_chunk_ids):
                print("\n[SUCCESS] Frame count matches label count.")
            else:
                print(f"\n[ERROR] Frame count ({len(first_chunk_frames)}) != Label count ({len(first_chunk_ids)})")

        # print examples
        for i in range(min(1, len(sample['chunks']))):
            print(f"\nChunk {i}:")
            print(f"  Frames in chunk: {len(sample['chunks'][i])}")
            print(f"  Range: {sample['chunk_ranges'][i]}")
            print(f"  Boundary Label: {sample['chunk_labels'][0][i]}") # Accessing first annotator's labels
            print(f"  Class IDs: {sample['chunk_class_ids'][i]}")
            print(f"  Class Names: {sample['chunk_class_names'][i]}")