import argparse
import json
import os
import glob
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm

from methods.boundary_to_tas import create_step_annotation_list


def load_gt_labels(gt_file: str) -> List[str]:
    with open(gt_file, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def load_pred_from_matched_json(matched_json_file: str) -> Tuple[List[str], Dict[str, Any]]:
    with open(matched_json_file, "r") as f:
        matched_data = json.load(f)
    return create_step_annotation_list(matched_data), matched_data


def align_sequences(pred: List[str], gt: List[str]) -> Tuple[List[str], List[str]]:
    min_len = min(len(pred), len(gt))
    return pred[:min_len], gt[:min_len]


def encode_labels(pred: List[str], gt: List[str]) -> Tuple[np.ndarray, np.ndarray, dict]:
    labels = []
    seen = set()
    for x in pred + gt:
        if x not in seen:
            seen.add(x)
            labels.append(x)

    label_to_id = {label: i for i, label in enumerate(labels)}
    pred_ids = np.array([label_to_id[x] for x in pred], dtype=np.int32)
    gt_ids = np.array([label_to_id[x] for x in gt], dtype=np.int32)
    return pred_ids, gt_ids, label_to_id


def plot_comparison(
    video_id: str,
    pred: List[str],
    gt: List[str],
    matched_data: Dict[str, Any],
    output_png: str,
) -> None:
    pred_ids, gt_ids, label_to_id = encode_labels(pred, gt)
    num_labels = len(label_to_id)
    x_min = -0.5
    x_max = len(pred_ids) - 0.5

    acc = float(np.mean(pred_ids == gt_ids)) if len(pred_ids) > 0 else 0.0

    fig = plt.figure(figsize=(34, 11))
    gs = fig.add_gridspec(2, 2, width_ratios=[4.8, 1.5], height_ratios=[1.35, 1.85], hspace=0.5, wspace=0.18)
    ax_gt = fig.add_subplot(gs[0, 0])
    ax_pred = fig.add_subplot(gs[1, 0], sharex=ax_gt)

    ax_legend = fig.add_subplot(gs[:, 1])

    base_cmap = plt.get_cmap("tab20")
    color_list = [base_cmap(i % base_cmap.N) for i in range(max(num_labels, 1))]
    cmap = ListedColormap(color_list)
    norm = BoundaryNorm(np.arange(-0.5, max(num_labels, 1) + 0.5, 1), cmap.N)

    ax_gt.imshow(
        gt_ids[np.newaxis, :],
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
    )
    ax_gt.set_yticks([])
    ax_gt.set_ylabel("GT")
    ax_gt.set_xlim(x_min, x_max)
    ax_gt.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    ax_pred.imshow(
        pred_ids[np.newaxis, :],
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
    )
    ax_pred.set_yticks([])
    ax_pred.set_ylabel("Pred")
    ax_pred.set_xlim(x_min, x_max)

    timeline = matched_data.get("timeline", [])
    valid_events = []
    for event in timeline:
        start_frame = int(event.get("start_frame", 0))
        if 0 <= start_frame < len(pred_ids):
            valid_events.append(event)

    # Show ALL descriptions as x-ticks directly on prediction strip.
    tick_events = valid_events

    tick_pos = []
    tick_labels = []
    for i, event in enumerate(tick_events, start=1):
        start_frame = int(event.get("start_frame", 0))
        desc = str(event.get("description", "")).strip() or "<no description>"
        if len(desc) > 34:
            desc = desc[:31] + "..."
        tick_pos.append(start_frame)
        tick_labels.append(f"{i}. {desc}")

        # visual event marker aligned to same x in GT and Pred
        ax_pred.axvline(start_frame, ymin=0.0, ymax=1.0, color="black", alpha=0.15, linewidth=0.8)
        ax_gt.axvline(start_frame, ymin=0.0, ymax=1.0, color="black", alpha=0.08, linewidth=0.6)

    ax_pred.set_xticks(tick_pos)
    ax_pred.set_xticklabels(tick_labels, rotation=-45, ha="left", fontsize=7)
    ax_pred.set_xlabel("Frame index / all predicted descriptions (on prediction plot)")

    # Legend (color -> class label)
    id_to_label = {v: k for k, v in label_to_id.items()}
    handles = []
    for idx in sorted(id_to_label.keys()):
        color = cmap(idx)
        handles.append(Patch(facecolor=color, edgecolor="none", label=f"{idx}: {id_to_label[idx]}"))

    ax_legend.axis("off")
    ax_legend.set_title("Color legend", fontsize=13)
    if handles:
        ax_legend.legend(
            handles=handles,
            loc="upper left",
            fontsize=9,
            frameon=False,
            ncol=1,
            labelspacing=0.45,
            handlelength=1.6,
            borderaxespad=0.3,
        )

    fig.suptitle(
        f"{video_id} | Frames={len(pred_ids)} | Frame Acc={acc * 100:.2f}% | Classes={len(label_to_id)}",
        fontsize=10,
    )

    fig.subplots_adjust(left=0.04, right=0.985, top=0.92, bottom=0.18, wspace=0.18, hspace=0.5)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Pred vs GT timeline strips for Egoper matched results.")
    parser.add_argument(
        "--result_root",
        type=str,
        default="results/bsurprise/egoper/20260224_182615",
        help="Experiment folder containing recipe subfolders and *_matched.json files.",
    )
    parser.add_argument(
        "--gt_root",
        type=str,
        default="data/Egoper/frame_annotations",
        help="Root folder containing ground-truth txt files by recipe.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to save comparison plots. Default: <result_root>/comparison_plots",
    )
    args = parser.parse_args()

    result_root = args.result_root
    output_dir = args.output_dir or os.path.join(result_root, "comparison_plots")
    os.makedirs(output_dir, exist_ok=True)

    matched_files = sorted(glob.glob(os.path.join(result_root, "*", "*_matched.json")))

    if not matched_files:
        print(f"No *_matched.json files found under: {result_root}")
        return

    ok = 0
    skipped = 0

    for matched_file in matched_files:
        recipe = os.path.basename(os.path.dirname(matched_file))
        video_id = os.path.basename(matched_file).replace("_matched.json", "")
        gt_file = os.path.join(args.gt_root, recipe, f"{video_id}.txt")

        if not os.path.exists(gt_file):
            print(f"[SKIP] GT missing: {gt_file}")
            skipped += 1
            continue

        try:
            pred, matched_data = load_pred_from_matched_json(matched_file)
            gt = load_gt_labels(gt_file)
            pred, gt = align_sequences(pred, gt)

            if len(pred) == 0:
                print(f"[SKIP] Empty aligned sequence: {video_id}")
                skipped += 1
                continue

            output_png = os.path.join(output_dir, f"{recipe}__{video_id}__pred_vs_gt.png")
            plot_comparison(
                video_id=video_id,
                pred=pred,
                gt=gt,
                matched_data=matched_data,
                output_png=output_png,
            )
            print(f"[OK] {output_png}")
            ok += 1
        except Exception as e:
            print(f"[SKIP] {video_id} failed: {e}")
            skipped += 1

    print(f"Done. Saved {ok} plots, skipped {skipped}.")
    print(f"Output folder: {output_dir}")


if __name__ == "__main__":
    main()
