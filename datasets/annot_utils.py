import os

import pickle

def load_gt_annotations(gt_annotation_path):
    """
    Load ground-truth annotations from a pickle file.

    Args:
        gt_annotation_path (str): Path to the GT annotation pickle file.

    Returns:
        dict: Dictionary mapping video_id to annotation entries.
    """
    with open(gt_annotation_path, "rb") as f:
        annotations = pickle.load(f)

    print(f"Loaded annotations for {len(annotations)} videos.")
    return annotations

def get_video_gt_info(annotations, video_path):
    """
    Retrieve ground-truth information for a given video.

    Args:
        annotations (dict):
            Loaded GT annotations mapping video_id to annotation entries.

        video_path (str):
            Path to the video directory or file.

    Returns:
        dict:
            Dictionary containing:
                - video_id (str)
                - boundaries (list): Frame indices of substage boundaries
                - fps (int or float or None)
    """
    video_id = os.path.splitext(os.path.basename(video_path))[0]

    if video_id not in annotations:
        raise KeyError(f"No annotation found for video_id: {video_id}")

    entry = annotations[video_id]

    return {
        "video_id": video_id,
        "boundaries": entry.get("substages_myframeidx", []),
        "fps": entry.get("fps", None),
    }