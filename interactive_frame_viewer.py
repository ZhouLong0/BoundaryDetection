import argparse
import json
import os
from typing import List

import matplotlib
import numpy as np
from PIL import Image


def _sorted_frame_paths(frames_dir: str) -> List[str]:
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
    files = [f for f in os.listdir(frames_dir) if f.lower().endswith(valid_ext)]

    def key_fn(name: str):
        stem = os.path.splitext(name)[0]
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    files.sort(key=key_fn)
    return [os.path.join(frames_dir, f) for f in files]


def _read_gt_labels(path: str) -> List[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def _read_pred_labels(path: str) -> List[str]:
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [str(x) for x in data]

    raise ValueError("Prediction file must be a JSON list of frame-level labels.")


def _build_label_map(gt: List[str], pred: List[str]):
    labels = []
    seen = set()
    for lab in gt + pred:
        if lab not in seen:
            labels.append(lab)
            seen.add(lab)

    label_to_id = {lab: i for i, lab in enumerate(labels)}
    return labels, label_to_id


def run_viewer(frames_dir: str, gt_path: str, pred_path: str, start_idx: int = 0):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from matplotlib.widgets import Slider

    frame_paths = _sorted_frame_paths(frames_dir)
    gt_labels = _read_gt_labels(gt_path)
    pred_labels = _read_pred_labels(pred_path)

    if len(frame_paths) == 0:
        raise ValueError(f"No frames found in {frames_dir}")

    n = min(len(frame_paths), len(gt_labels), len(pred_labels))
    if n == 0:
        raise ValueError("At least one frame/label pair is required.")

    if len(frame_paths) != len(gt_labels) or len(frame_paths) != len(pred_labels):
        print(
            "[Warning] Length mismatch: "
            f"frames={len(frame_paths)}, gt={len(gt_labels)}, pred={len(pred_labels)}. "
            f"Using first {n} elements."
        )

    frame_paths = frame_paths[:n]
    gt_labels = gt_labels[:n]
    pred_labels = pred_labels[:n]

    all_labels, label_to_id = _build_label_map(gt_labels, pred_labels)
    gt_ids = np.array([label_to_id[x] for x in gt_labels], dtype=np.int32)
    pred_ids = np.array([label_to_id[x] for x in pred_labels], dtype=np.int32)

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % cmap.N) for i in range(len(all_labels))]
    listed_cmap = ListedColormap(colors)

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(8, 1, height_ratios=[5, 0.3, 0.8, 0.3, 0.8, 0.25, 0.25, 0.1])

    ax_img = fig.add_subplot(gs[0, 0])
    ax_gt = fig.add_subplot(gs[2, 0])
    ax_pred = fig.add_subplot(gs[4, 0], sharex=ax_gt)
    ax_slider = fig.add_subplot(gs[6, 0])

    init_idx = int(np.clip(start_idx, 0, n - 1))

    first_img = np.asarray(Image.open(frame_paths[init_idx]).convert("RGB"))
    img_artist = ax_img.imshow(first_img)
    title_artist = ax_img.set_title("")
    ax_img.axis("off")

    gt_bar = gt_ids[np.newaxis, :]
    pred_bar = pred_ids[np.newaxis, :]

    ax_gt.imshow(gt_bar, aspect="auto", cmap=listed_cmap, vmin=0, vmax=max(1, len(all_labels) - 1))
    ax_gt.set_yticks([0])
    ax_gt.set_yticklabels(["GT"])
    ax_gt.set_ylabel("Label")
    ax_gt.set_xticks([])

    ax_pred.imshow(pred_bar, aspect="auto", cmap=listed_cmap, vmin=0, vmax=max(1, len(all_labels) - 1))
    ax_pred.set_yticks([0])
    ax_pred.set_yticklabels(["Pred"])
    ax_pred.set_ylabel("Label")
    ax_pred.set_xlabel("Frame index")

    gt_cursor = ax_gt.axvline(init_idx, color="white", linewidth=2)
    pred_cursor = ax_pred.axvline(init_idx, color="white", linewidth=2)

    slider = Slider(
        ax=ax_slider,
        label="Frame",
        valmin=0,
        valmax=n - 1,
        valinit=init_idx,
        valstep=1,
    )

    def update(idx):
        i = int(idx)
        new_img = np.asarray(Image.open(frame_paths[i]).convert("RGB"))
        img_artist.set_data(new_img)

        gt_cursor.set_xdata([i, i])
        pred_cursor.set_xdata([i, i])

        title_artist.set_text(
            f"Frame {i + 1}/{n} | GT: {gt_labels[i]} | Pred: {pred_labels[i]}"
        )
        fig.canvas.draw_idle()

    def on_key(event):
        cur = int(slider.val)
        if event.key in ("right", "d"):
            slider.set_val(min(n - 1, cur + 1))
        elif event.key in ("left", "a"):
            slider.set_val(max(0, cur - 1))

    slider.on_changed(update)
    fig.canvas.mpl_connect("key_press_event", on_key)

    update(init_idx)

    # Legend with color swatches on the right side
    legend_handles = [
        Patch(facecolor=colors[i], edgecolor="black", label=f"{i}: {lab}")
        for i, lab in enumerate(all_labels)
    ]
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.99, 0.5),
        fontsize=8,
        frameon=True,
        title="Label colors",
    )

    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive frame-level GT vs prediction viewer")
    parser.add_argument(
        "--video_id",
        type=str,
        required=True,
        help="Video id, e.g. quesadilla_u1_a2_normal_006",
    )
    parser.add_argument(
        "--exp_id",
        type=str,
        required=True,
        help="Experiment id, e.g. 20260220_151515",
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        default=None,
        help="Optional override for video frame folder",
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default=None,
        help="Optional override for GT frame-level annotation text file",
    )
    parser.add_argument(
        "--pred_path",
        type=str,
        default=None,
        help="Optional override for predicted frame-level labels JSON file",
    )
    parser.add_argument("--start_idx", type=int, default=0, help="Initial frame index")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "webagg", "gui"],
        default="auto",
        help="Matplotlib backend mode. Use webagg on remote machines.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="WebAgg bind host")
    parser.add_argument("--port", type=int, default=8988, help="WebAgg port")
    return parser.parse_args()


def configure_backend(backend: str, host: str, port: int):
    display_missing = os.environ.get("DISPLAY", "") == ""

    use_webagg = backend == "webagg" or (backend == "auto" and display_missing)
    if use_webagg:
        matplotlib.use("WebAgg", force=True)
        matplotlib.rcParams["webagg.address"] = host
        matplotlib.rcParams["webagg.port"] = port
        matplotlib.rcParams["webagg.open_in_browser"] = False
        print(f"Using WebAgg backend on http://{host}:{port}")
        print("If remote, forward this port in VS Code and open it in your local browser.")
    else:
        # Let matplotlib pick the default interactive GUI backend
        print("Using GUI backend.")


def build_paths(video_id: str, exp_id: str, frames_dir: str | None, gt_path: str | None, pred_path: str | None):
    recipe = video_id.split("_")[0]

    resolved_frames_dir = frames_dir or os.path.join(
        "data", "Egoper", "extracted_frames", video_id
    )
    resolved_gt_path = gt_path or os.path.join(
        "data", "Egoper", "frame_annotations", recipe, f"{video_id}.txt"
    )
    resolved_pred_path = pred_path or os.path.join(
        "results", "bsurprise", "egoper", str(exp_id), recipe, f"{video_id}_matched_tas.json"
    )

    return resolved_frames_dir, resolved_gt_path, resolved_pred_path


if __name__ == "__main__":
    args = parse_args()
    configure_backend(args.backend, args.host, args.port)

    frames_dir, gt_path, pred_path = build_paths(
        video_id=args.video_id,
        exp_id=args.exp_id,
        frames_dir=args.frames_dir,
        gt_path=args.gt_path,
        pred_path=args.pred_path,
    )

    print(f"video_id={args.video_id} | exp_id={args.exp_id}")
    print(f"frames_dir={frames_dir}")
    print(f"gt_path={gt_path}")
    print(f"pred_path={pred_path}")

    run_viewer(
        frames_dir=frames_dir,
        gt_path=gt_path,
        pred_path=pred_path,
        start_idx=args.start_idx,
    )
