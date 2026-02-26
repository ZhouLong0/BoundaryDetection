# RUN THIS IN THE VERY FIRST CELL
#%env HF_HUB_OFFLINE=1

import os
os.environ["HF_HUB_OFFLINE"] = "1"

print("Offline mode forced.")

from models import Qwen3VL
from models.utils import load_all_frames_from_folder, build_chunks_from_full_video
import json
from datasets.annot_utils import load_gt_annotations, get_video_gt_info
from eval import evaluate_gebd, do_eval
from datetime import datetime



def eval_video(video_path, chunk_size=16, fps=30, verbose=False, prompt_templates=None, model=None):
    list_frames, list_files, n_frames = load_all_frames_from_folder(video_path)
    print(f"Loaded {n_frames} frames from {video_path}.")

    chunks = build_chunks_from_full_video(
        total_frames=n_frames,
        chunk_size=chunk_size,
    )

    current_description = None
    timeline = []
    boundaries_frame_indices = []
    boundaries_timestamps = []

    for idx, (start, end) in enumerate(chunks):
        frame_chunk = list_frames[start:end]

        if idx == 0:
            prompt = prompt_templates["initial_description"]
            current_description = model.infer(frame_chunk, prompt, sample_fps=fps)
            if verbose:
                print(f"Chunk {idx}, start{start} - end{end}: Initial description: {current_description}")
            timeline.append((start, end, current_description, None))
            continue

        prompt = prompt_templates["validity_check"].format(
            previous_description=current_description
        )

        answer = model.infer(frame_chunk, prompt, sample_fps=fps)

        if verbose:
            print(f"Chunk {idx}, start{start} - end{end}: Validity check answer: {answer}")

        # if answer.upper().startswith("YES"):
        #     continue

        if "YES" in answer.upper():
            timeline.append((start, end, "skipped", answer))
            continue

        prompt = prompt_templates["update_description"]

        current_description = model.infer(frame_chunk, prompt, sample_fps=fps)

        if verbose:
            print(f"Chunk {idx}, start{start} - end{end}: Updated description: {current_description}")

        timeline.append((start, end, current_description, answer))
        boundaries_frame_indices.append(end - chunk_size/2)
        boundaries_timestamps.append((end - chunk_size/2) / fps) 


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
    PROMPT_FILE = "prompts.json"

    # -----------------------
    # Load data
    # -----------------------
    full_annotations_dict = load_gt_annotations(GT_ANNOTATION_PATH)

    video_paths = []
    with open(SPLIT_FILE, "r") as f:
        for line in f:
            video_id = line.strip()
            if video_id:
                video_paths.append(os.path.join(BASE_DIR, video_id))

    print(f"Loaded {len(video_paths)} videos to process from {SPLIT_FILE}.")

    # Load prompts
    with open(PROMPT_FILE, "r") as f:
        PROMPT_SETS = json.load(f)

    print(f"Loaded {len(PROMPT_SETS)} prompt sets from {PROMPT_FILE}")

    # -----------------------
    # Run experiments
    # -----------------------
    all_experiments_results = []

    for prompt_idx, prompt_set in enumerate(PROMPT_SETS):
        for chunk_size in CHUNK_SIZES:
            print(f"\n=== Experiment: Prompt {prompt_idx+1}, chunk_size={chunk_size} ===")

            # Store predictions per video
            predictions = {}

            for video_path in video_paths:
                video_id = os.path.basename(video_path)

                try:
                    # Load per-video GT info
                    video_annotations = get_video_gt_info(full_annotations_dict, video_path)
                    video_fps = video_annotations["fps"]

                    # Run video evaluation
                    timeline, boundaries_frame_indices, boundaries_timestamps = eval_video(
                        video_path,
                        chunk_size=chunk_size,
                        fps=video_fps,
                        model=qwen_model,
                        prompt_templates=prompt_set
                    )

                    predictions[video_id] = boundaries_frame_indices

                except Exception as e:
                    print(f"Error processing video {video_id}: {e}")
                    predictions[video_id] = {"error": str(e)}

            # -----------------------
            # Evaluate metrics per threshold
            # -----------------------
            metrics_per_threshold = {}

            for thres in REL_DIS_THRESH:
                f1, rec, prec = do_eval(full_annotations_dict, predictions, threshold=thres)
                metrics_per_threshold[thres] = {
                    "f1": f1,
                    "recall": rec,
                    "precision": prec
                }

            # Save experiment results
            experiment_entry = {
                "prompt_text": prompt_set,
                "chunk_size": chunk_size,
                "metrics_per_threshold": metrics_per_threshold,
                #"predictions": predictions
            }

            all_experiments_results.append(experiment_entry)

    # -----------------------
    # Save all experiments to JSON
    # -----------------------
    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Include it in the filename
    OUTPUT_FILE = f"all_experiments_results_{timestamp}.json"

    # Save JSON
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_experiments_results, f, indent=4)

    print(f"All experiments saved to {OUTPUT_FILE}")