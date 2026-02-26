import os
import json
import random
import argparse
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List
import torch
import numpy as np
from methods.boundary_to_tas import create_step_annotation_list, evaluate_tas


# Force Offline Mode for HF
os.environ["HF_HUB_OFFLINE"] = "1"

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _normalize_baseline_timeline(timeline: List[Any]) -> List[Dict[str, Any]]:
    """
    Convert baseline timeline tuples to the event_start dict format used downstream.

    Baseline timeline format (per chunk):
        (start_frame, end_frame, description_or_skipped, decision)
    We keep only true event descriptions (initial and updates), skipping "skipped" entries.
    """
    normalized = []
    for item in timeline:
        if not isinstance(item, (list, tuple)) or len(item) < 4:
            continue
        start_frame, _, description, _ = item
        if description == "skipped":
            continue
        normalized.append(
            {
                "start_frame": start_frame,
                "description": description,
                "type": "event_start",
            }
        )
    return normalized


def main():
    parser = argparse.ArgumentParser(description="Run BSurprise/Baseline on Egoper/GTEA/Breakfast datasets")
    parser.add_argument("--method", type=str, default="bsurprise", choices=["bsurprise", "baseline"], help="Method to run")
    parser.add_argument("--dataset", type=str, default="egoper", choices=["egoper", "gtea", "breakfast"], help="Dataset to process")
    parser.add_argument("--recipe", type=str, default="quesadilla", help="Recipe for Egoper (ignored for GTEA/Breakfast)")
    parser.add_argument("--split_name", type=str, default="test.split1.bundle", help="Single split file name for Egoper")
    parser.add_argument("--split_files", type=str, default="test.split1.bundle", help="Comma-separated split files for GTEA/Breakfast")
    parser.add_argument("--exp_id", type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"), help="Experiment identifier used for output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--threshold", type=float, default=0.3, help="Detection threshold for BSurprise")
    parser.add_argument("--base_dir", type=str, default=None, help="Dataset frame root. If omitted, a dataset-specific default is used")
    parser.add_argument("--chunk_duration", type=float, default=2.0, help="Duration of video chunks in seconds")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging during classification")
    parser.add_argument("--full_steps", action="store_true", help="When set, all the steps for mapping step")
    parser.add_argument("--history_conditioning", action="store_true", default=True, help="Enable history-conditioned classification")
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print(f"Ignoring unknown arguments: {unknown_args}")

    set_seed(args.seed)
    print(f"Offline mode forced. Seed set to {args.seed}. Dataset={args.dataset}. Method={args.method}")

    # --- Imports for Models and Datasets ---
    # These are imported here to ensure env vars are set first
    from models import Qwen3VL
    from datasets.egoper import EgoperDataset
    from datasets.gtea import GTEADataset
    from datasets.breakfast import BreakfastDataset
    from methods.bsurprise import BSurprise
    from methods.baseline import eval_video
    from methods.baseline_config import EgoperBaselineConfig, GTEABaselineConfig, BreakfastBaselineConfig
    from methods.bsurprise.config import EgoperConfig, GTEAConfig, BreakfastConfig

    # Initialize Model
    print("Initializing Qwen3VL...")
    qwen_model3 = Qwen3VL()

    # Initialize Dataset
    if args.dataset == "egoper":
        base_dir = args.base_dir or "data/Egoper/extracted_frames"
        print(f"Initializing EgoperDataset for recipe: {args.recipe}...")
        dataset = EgoperDataset(
            base_dir=base_dir,
            split_root="data/Egoper/splits",
            annotation_root="data/Egoper/frame_annotations",
            mapping_root="data/Egoper/idx_action_mapping",
            recipes=[args.recipe],
            split_name=args.split_name,
            chunk_duration=args.chunk_duration,
            num_frames_per_chunk=8,
            fps=10
        )
        config = EgoperConfig()
        baseline_config = EgoperBaselineConfig()
    elif args.dataset == "gtea":
        base_dir = args.base_dir or "data/GTEA/frames"
        split_files = [s.strip() for s in args.split_files.split(',') if s.strip()]
        print(f"Initializing GTEADataset with split files: {split_files}...")
        dataset = GTEADataset(
            base_dir=base_dir,
            split_root="data/GTEA/data/gtea/splits",
            annotation_root="data/GTEA/data/gtea/groundTruth",
            mapping_path="data/GTEA/data/gtea/mapping.txt",
            split_files=split_files,
            chunk_duration=args.chunk_duration,
            num_frames_per_chunk=8,
            fps=15,
        )
        config = GTEAConfig()
        baseline_config = GTEABaselineConfig()
    else:  # breakfast
        base_dir = args.base_dir or "data/Breakfast/frames"
        split_files = [s.strip() for s in args.split_files.split(',') if s.strip()]
        print(f"Initializing BreakfastDataset with split files: {split_files}...")
        dataset = BreakfastDataset(
            base_dir=base_dir,
            split_root="data/Breakfast/splits",
            annotation_root="data/Breakfast/groundTruth",
            mapping_path="data/Breakfast/mapping.txt",
            split_files=split_files,
            chunk_duration=args.chunk_duration,
            num_frames_per_chunk=8,
            fps=15,
        )
        config = BreakfastConfig()
        baseline_config = BreakfastBaselineConfig()

    # Always initialize bsurprise for downstream step-matching utilities.
    bsurprise = BSurprise(
        qwen_interface=qwen_model3,
        divergence_type='kl',
        detection_threshold=args.threshold,
        config=config
    )
    prompt_set = None

    # Initialize baseline prompt set only when needed
    if args.method == "baseline":
        prompt_set = baseline_config
        print(f"Using {type(baseline_config).__name__} for baseline prompts")

    # Setup Output Directory
    exp_output_root = os.path.join("results/tas", args.method, args.dataset, str(args.exp_id))
    os.makedirs(exp_output_root, exist_ok=True)

    # Save run configuration for reproducibility
    config_dump = {
        "exp_id": args.exp_id,
        "dataset": args.dataset,
        "method": args.method,
        "recipe": args.recipe,
        "seed": args.seed,
        "threshold": args.threshold,
        "base_dir": base_dir,
        "split_name": args.split_name,
        "split_files": args.split_files,
        "chunk_duration": args.chunk_duration,
        "baseline_prompt_config": asdict(prompt_set) if prompt_set is not None else None,
        "config": asdict(config),
        "history_conditioning": args.history_conditioning,
        "full_steps": args.full_steps,
    }
    config_path = os.path.join(exp_output_root, "run_config.json")
    with open(config_path, "w") as f:
        json.dump(config_dump, f, indent=4)
    print(f"Saved run configuration to {config_path}")

    print(f"Experiment configuration:")
    for key, value in config_dump.items():
        print(f"  {key}: {value}")
    print()

    for video in dataset:
        video_id = video['video_id']
        video_recipe = video['recipe']
        if args.dataset == "egoper":  # For GTEA/Breakfast, we can organize by split instead of recipe
            output_dir = os.path.join(exp_output_root, video_recipe)
        else:
            output_dir = exp_output_root
        print(f"\nProcessing {video_id}...")

        # --- Run Classification ---
        if args.method == "baseline":
            print("Running baseline classification...")
            baseline_timeline, boundaries_frames, boundaries_timestamps = eval_video(
                video_data=video,
                prompt_templates=prompt_set,
                model=qwen_model3,
                update_threshold=args.threshold,
                verbose=args.verbose,
            )
            timeline = _normalize_baseline_timeline(baseline_timeline)
        elif args.history_conditioning:
            print("Running history-conditioned classification...")
            timeline, boundaries_frames, boundaries_timestamps = bsurprise.classify_video_chunks_history_conditioned(
                video_data=video,
                verbose=args.verbose,
            )
        else:
            print("Running standard classification...")
            timeline, boundaries_frames, boundaries_timestamps = bsurprise.classify_video_chunks(
                video_data=video, 
                verbose=args.verbose,
            )

        # --- Prepare Data ---
        output_data = {
            "video_id": video_id,
            "num_frames": video['num_frames'],
            "timeline": timeline,
            "boundaries_frames": boundaries_frames,
            "boundaries_timestamps": boundaries_timestamps
        }

        # --- Save Initial Results ---
        output_filename = os.path.join(output_dir, f"{video_id}.json")
        if not os.path.exists(os.path.dirname(output_filename)):
            os.makedirs(os.path.dirname(output_filename))
        with open(output_filename, 'w') as f:
            json.dump(output_data, f, indent=4)
        
        print(f"Saved results to {output_filename}")

        ##### --- Match to Recipe Steps ---
        print(f"Matching descriptions for {video_id}...")
        recipe_steps = dataset.get_list_steps_by_video_id(video_id, full=args.full_steps)
        
        # This updates the json or creates a matched version depending on method implementation
        output_data_matched = bsurprise.match_timeline_descriptions_to_recipe_steps(
            result_json_path=output_filename, 
            recipe_steps=recipe_steps
        )
        output_filename = output_filename.replace(".json", "_matched.json")
        with open(output_filename, 'w') as f:
            json.dump(output_data_matched, f, indent=4)
        print(f"Matching completed for {video_id}")

        ##### --- Create Step Annotation List ---

        tas_results = create_step_annotation_list(matched_data=output_data_matched)
        tas_output_path = os.path.join(output_dir, f"{video_id}_matched_tas.json")
        with open(tas_output_path, 'w') as f:
            json.dump(tas_results, f, indent=4)
        print(f"Step annotation list created for {video_id}, saved to {tas_output_path}")

    #### --- evaluate against GT ---
    eval_result_dir = exp_output_root if args.dataset == "egoper" else os.path.join(exp_output_root, args.dataset)  # For GTEA/Breakfast
    evaluate_tas(dataset=dataset, result_dir=eval_result_dir, dataset_name=args.dataset, exp_id=args.exp_id)
    print(f"Evaluation completed for {args.dataset}. Results saved in {exp_output_root}.")



if __name__ == "__main__":
    main()