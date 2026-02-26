import argparse
import json
import os

from datasets.breakfast import BreakfastDataset
from methods.boundary_to_tas import evaluate_tas


def build_eval_video_list(split_files):
    dataset = BreakfastDataset(
        base_dir="data/Breakfast/frames",
        split_root="data/Breakfast/splits",
        annotation_root="data/Breakfast/groundTruth",
        mapping_path="data/Breakfast/mapping.txt",
        split_files=split_files,
        chunk_duration=2.0,
        num_frames_per_chunk=8,
        fps=15,
    )

    # Avoid loading frames: evaluation only needs video_id (+ recipe for compatibility)
    return [{"video_id": v["id"], "recipe": v.get("recipe", "") } for v in dataset.video_list]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate already-saved Breakfast TAS predictions"
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="results/bsurprise/breakfast/20260225_004245",
        help="Directory containing *_matched_tas.json prediction files",
    )
    args = parser.parse_args()

    result_dir = args.result_dir
    run_config_path = os.path.join(result_dir, "run_config.json")

    # Default if run_config is unavailable
    split_files = ["test.split1.bundle"]
    exp_id = os.path.basename(os.path.normpath(result_dir))

    if os.path.exists(run_config_path):
        with open(run_config_path, "r") as f:
            run_config = json.load(f)
        split_files_str = run_config.get("split_files", "test.split1.bundle")
        split_files = [s.strip() for s in split_files_str.split(",") if s.strip()]
        exp_id = run_config.get("exp_id", exp_id)

    eval_dataset = build_eval_video_list(split_files=split_files)

    print(f"Evaluating predictions from: {result_dir}")
    print(f"Using split files: {split_files}")
    print(f"Number of videos to evaluate: {len(eval_dataset)}")

    evaluate_tas(
        dataset=eval_dataset,
        result_dir=result_dir,
        dataset_name="breakfast",
        exp_id=exp_id,
    )

    print("Done.")


if __name__ == "__main__":
    main()
