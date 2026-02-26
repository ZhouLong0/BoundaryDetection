# RUN THIS IN THE VERY FIRST CELL
#%env HF_HUB_OFFLINE=1

import os
from typing import Dict, List, Any, Tuple
os.environ["HF_HUB_OFFLINE"] = "1"

print("Offline mode forced.")

from models import Qwen3VL
from models.utils import load_all_frames_from_folder, build_chunks_from_full_video
import json
from datasets.annot_utils import load_gt_annotations, get_video_gt_info
from eval import evaluate_gebd, do_eval
from datetime import datetime
import warnings
from datasets.tapos import TAPOSDataset



def eval_video(
    video_data: Dict[str, Any], 
    prompt_templates: Dict, 
    model: Qwen3VL, 
    chunk_size: int,
    update_threshold: float = 0.5,
    verbose: bool = False
) -> Tuple[List, List, List]:
    """
    Process a single video using the loaded video_data.
    
    Returns:
        timeline, boundaries_frame_indices, boundaries_timestamps
    """
    
    # Extract data
    chunks = video_data['chunks']
    chunk_ranges = video_data['chunk_ranges']
    fps = video_data['fps']
    if fps is None: fps = 30

    # --- Prompt Settings & Warnings ---
    if "skip_token" not in prompt_templates:
        warnings.warn("[WARNING] 'skip_token' not found in prompts. Defaulting to 'SAME'.")
    skip_token = prompt_templates.get("skip_token", "SAME")

    if "update_token" not in prompt_templates:
        warnings.warn("[WARNING] 'update_token' not found in prompts. Defaulting to 'NEW'.")
    update_token = prompt_templates.get("update_token", "NEW")
    
    current_description = None
    timeline = []
    boundaries_frame_indices = []
    boundaries_timestamps = []
    
    for idx, (frame_chunk, (start, end)) in enumerate(zip(chunks, chunk_ranges)):
        
        # --- Step 1: Initial Chunk ---
        if idx == 0:
            prompt = prompt_templates["initial_description"]
            current_description = model.infer(frame_chunk, prompt, sample_fps=fps)
            if verbose: print(f"Chunk {idx}: Initial: {current_description}")
            
            # Original timeline format: (start, end, description, answer/decision)
            timeline.append((start, end, current_description, None))
            continue

        # --- Step 2: Validity Check (Probabilistic) ---
        prompt_check = prompt_templates["validity_check"].format(
            previous_description=current_description
        )

        probs = model.get_next_token_probs(
            frame_chunk, 
            prompt_check, 
            candidate_strings=[skip_token, update_token], 
            sample_fps=fps
        )

        p_skip = probs.get(skip_token, 0.0)
        p_update = probs.get(update_token, 0.0)

        if p_skip + p_update == 0.0:
            warnings.warn(
                f"\nChunk {idx}: Both candidate strings '{skip_token}' and '{update_token}' "
                "have zero probability. Defaulting to skip (SAME).\n"
            )
            p_skip = 1.0

        # --- Step 3: Strict Decision Logic ---
        total_mass = p_skip + p_update
        
        if total_mass > 0:
            relative_p_update = p_update / total_mass
        else:
            relative_p_update = 0.0

        if verbose:
            print(f"Chunk {idx}: {skip_token}={p_skip:.4f}, {update_token}={p_update:.4f} "
                  f"| Rel Update Score: {relative_p_update:.2f} (Thresh: {update_threshold})")

        # Decision
        if relative_p_update > update_threshold:
            # UPDATE EVENT
            prompt_update = prompt_templates["update_description"]
            current_description = model.infer(frame_chunk, prompt_update, sample_fps=fps)
            
            timeline.append((start, end, current_description, update_token))
            
            # Original boundary logic
            boundary_frame = end - chunk_size / 2
            boundaries_frame_indices.append(boundary_frame)
            boundaries_timestamps.append(boundary_frame / fps)
            
            if verbose: print(f"  -> Update TRIGGERED")
            
        else:
            # SKIP UPDATE
            timeline.append((start, end, "skipped", skip_token))
            if verbose: print(f"  -> Update SKIPPED")

    return timeline, boundaries_frame_indices, boundaries_timestamps

BASE_DIR = "data/TAPOS/images/val"
SPLIT_FILE = "random_100_val_split.txt"

if __name__ == "__main__":
    # -----------------------
    # Settings
    # -----------------------
    qwen_model = Qwen3VL()

    GT_ANNOTATION_PATH = "data/TAPOS/tapos_gt_val.pkl"

    # Example chunk sizes to test
    CHUNK_SIZES = [4, 8, 12, 16]

    # Thresholds for evaluation
    REL_DIS_THRESH = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    # Prompt templates JSON file to test
    PROMPT_FILE = "prompts_prob.json"

    # -----------------------
    # Load data
    # -----------------------
   # Load Full Annotations
    full_annotations_dict = load_gt_annotations(GT_ANNOTATION_PATH)

    # Load Prompts
    with open(PROMPT_FILE, "r") as f:
        PROMPT_SETS = json.load(f)
    print(f"Loaded {len(PROMPT_SETS)} prompt sets.")

    all_experiments_results = []

    for prompt_idx, prompt_set in enumerate(PROMPT_SETS):
        
        # Extract threshold from prompt JSON, default to 0.5 if missing
        if "update_threshold" in prompt_set:
            current_threshold = prompt_set["update_threshold"]
        else:
            warnings.warn(
                f"[WARNING] 'update_threshold' not found in prompt set index {prompt_idx}. "
                "Defaulting to 0.5."
            )
            current_threshold = 0.5


        for chunk_size in CHUNK_SIZES:
            print(f"\n=== Exp: Prompt {prompt_idx+1}, chunk_size={chunk_size}, thresh={current_threshold} ===")

            tapos_dataset = TAPOSDataset(
                base_dir=BASE_DIR,
                split_file=SPLIT_FILE,
                gt_annotation_path=GT_ANNOTATION_PATH,
                chunk_size=chunk_size
            )
            
            predictions = {}

            for i in range(len(tapos_dataset)):
                try:
                    video_data = tapos_dataset[i]
                    video_id = video_data['video_id']
                    
                    # Unpack tuple return
                    timeline, boundary_frames, boundary_times = eval_video(
                        video_data=video_data,
                        prompt_templates=prompt_set,
                        model=qwen_model,
                        chunk_size=chunk_size,
                        update_threshold=current_threshold, 
                        verbose=False
                    )

                    predictions[video_id] = boundary_frames

                except Exception as e:
                    print(f"Error processing video index {i}: {e}")
                    if 'video_id' in locals():
                        predictions[video_id] = {"error": str(e)}

            # Evaluate Metrics
            metrics_per_threshold = {}
            for thres in REL_DIS_THRESH:
                f1, rec, prec = do_eval(full_annotations_dict, predictions, threshold=thres)
                metrics_per_threshold[thres] = {
                    "f1": f1, "recall": rec, "precision": prec
                }

            # Log Results
            experiment_entry = {
                "prompt_index": prompt_idx,
                "prompt_text": prompt_set,
                "chunk_size": chunk_size,
                "update_threshold": current_threshold,
                "metrics_per_threshold": metrics_per_threshold,
            }
            all_experiments_results.append(experiment_entry)

    # Save Results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_FILE = f"all_experiments_results_withprob_{timestamp}.json"
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_experiments_results, f, indent=4)

    print(f"All experiments saved to {OUTPUT_FILE}")