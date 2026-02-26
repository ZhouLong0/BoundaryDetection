# RUN THIS IN THE VERY FIRST CELL
#%env HF_HUB_OFFLINE=1

import os
import random
from typing import Dict, List, Any, Tuple
from dataclasses import asdict

import numpy as np
import torch
os.environ["HF_HUB_OFFLINE"] = "1"

print("Offline mode forced.")

from models import Qwen3VL
from models.qwen2_5vl import Qwen2_5VL
from models.utils import load_all_frames_from_folder, build_chunks_from_full_video
import json
from datasets.annot_utils import load_gt_annotations, get_video_gt_info
from eval import evaluate_gebd, do_eval
from datetime import datetime
import warnings
from datasets.tapos1 import TaposDataset
import argparse
from methods.baseline import eval_video
from methods.baseline_config import BaselineConfig
from methods.bsurprise import BSurprise
from methods.bsurprise.config import TaposConfig

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TAPOS experiments with Qwen models.")
    
    # 1. Add model selection argument
    parser.add_argument("--dataset", type=str, default="tapos", choices=["tapos"], help="The dataset to evaluate on.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct", choices=["Qwen/Qwen3-VL-8B-Instruct", "Qwen/Qwen2.5-VL-3B-Instruct"], help="The Hugging Face path or local path for the Qwen model.")
    parser.add_argument("--duration", type=float, default=1, help="Chunk duration in seconds")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--det_threshold", type=float, default=0.5)
    parser.add_argument("--method", type=str, default="baseline", choices=["baseline", "bsurprise"], help="The method to use for evaluation")
    parser.add_argument("--verbose", action="store_true", help="Whether to print detailed logs during processing")
    parser.add_argument("--split_file", type=str, default="random_100_val_split.txt", help="Path to the split file containing video IDs for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--exp_id", type=str, default="exp1", help="Experiment ID for organizing results")
    parser.add_argument("--history_conditioning", action="store_true", help="Enable history-conditioned classification")

    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print(f"Ignoring unknown arguments: {unknown_args}")

    set_seed(args.seed)
    # -----------------------
    # Initialize Model
    # -----------------------
    print(f"Loading model: {args.model_name}...")
    # Initialize your model wrapper with the specific name from args
    if args.model_name == "Qwen/Qwen3-VL-8B-Instruct":
        qwen_model = Qwen3VL()
    elif args.model_name == "Qwen/Qwen2.5-VL-3B-Instruct":
        qwen_model = Qwen2_5VL()
    else:
        raise ValueError(f"Unsupported model name: {args.model_name}")
    
    exp_output_root = os.path.join("results/boundary", args.method, "tapos", args.exp_id)
    os.makedirs(exp_output_root, exist_ok=True)

    # -----------------------
    # Baseline prompt configuration
    # -----------------------
    prompt_config = BaselineConfig()
    print(f"Using {type(prompt_config).__name__} for baseline prompts")

    config_dump = {
        **vars(args),
        "baseline_prompt_config": asdict(prompt_config),
    }
    config_path = os.path.join(exp_output_root, "run_config.json")
    with open(config_path, "w") as f:
        json.dump(config_dump, f, indent=4)


    print(f"Experiment configuration:")
    for key, value in config_dump.items():
        print(f"  {key}: {value}")

    # -----------------------
    # Load Dataset
    # -----------------------
    if args.dataset == "tapos":
        tapos_dataset = TaposDataset(
            base_dir="data/TAPOS/images/val",
            split_file=args.split_file,
            gt_annotation_path="data/TAPOS/tapos_gt_val.pkl",
            chunk_duration=args.duration,
            num_frames_per_chunk=args.samples
        )
        full_annotations_dict = load_gt_annotations("data/TAPOS/tapos_gt_val.pkl")
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    
    # ... (Loop through dataset and collect predictions) ...
    predictions = {}

    for i in range(len(tapos_dataset)):
        video_data = tapos_dataset[i]
        video_id = video_data['video_id']

        if args.method == "baseline":
            timeline, boundary_frames, boundary_times = eval_video(
                video_data=video_data,
                prompt_templates=prompt_config,
                model=qwen_model,
                update_threshold=args.det_threshold, 
                verbose=args.verbose
            )

        elif args.method == "bsurprise":
            bsurprise_detector = BSurprise(qwen_interface=qwen_model, divergence_type="kl", config=TaposConfig(), detection_threshold=args.det_threshold)
            if args.history_conditioning:
                print(f"Running BSurprise with history-conditioned classification...")
                timeline, boundary_frames, boundary_times = bsurprise_detector.classify_video_chunks_history_conditioned(video_data=video_data, verbose=args.verbose)
            else:
                timeline, boundary_frames, boundary_times = bsurprise_detector.classify_video_chunks(video_data, verbose=args.verbose)

        predictions[video_id] = boundary_frames

    # Evaluate Metrics
    metrics_per_threshold = {}
    REL_DIS_THRESH = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    for thres in REL_DIS_THRESH:
        f1, rec, prec = do_eval(full_annotations_dict, predictions, threshold=thres)
        metrics_per_threshold[thres] = {
            "f1": f1, "recall": rec, "precision": prec
        }

    # -----------------------
    # Save Results
    # -----------------------
    results_file = os.path.join(exp_output_root, f"eval.json")
    
    with open(results_file, "w") as f:
        json.dump(metrics_per_threshold, f, indent=4)