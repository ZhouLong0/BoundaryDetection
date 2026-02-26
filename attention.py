import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import warnings

def visualize_video_spatial_attention(
    model,
    inputs,
    tokenizer,
    frames,                 # List[PIL.Image]
    alpha=0.5,
    cmap=plt.cm.jet,
    n_cols=4,               # number of images per row
    normalize_per_frame=False
):
    """
    Visualize spatial attention over video frames for a SINGLE generated token.
    
    Args:
        normalize_per_frame (bool): 
            If True, attention is normalized (0-1) individually for each frame 
            (good for seeing spatial focus per frame).
            If False, attention is normalized globally across all frames 
            (good for seeing which frames are most important).
    """

    model.set_attn_implementation("eager")

    # ---- 1. SETUP ----
    curr_inputs = {
        k: v.clone() if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    frames_np = np.stack(
        [np.array(f.convert("RGB")) for f in frames],
        axis=0
    )  # (T_orig, H, W, 3)

    T_orig, H, W, _ = frames_np.shape

    # Qwen video grid
    # Note: Ensure "video_grid_thw" exists in inputs or handle gracefully
    t_vis, h, w = curr_inputs["video_grid_thw"][0].tolist()
    eff_h, eff_w = h // 2, w // 2

    if T_orig != t_vis * 2:
        warnings.warn(
            f"Expected {t_vis * 2} original frames, got {T_orig}, maybe due to non-even frame count."
        )

    # Identify vision tokens (specific to Qwen-VL architecture usually)
    vision_indices = torch.where(
        curr_inputs["input_ids"][0] == 151656
    )[0]

    # ---- 2. FORWARD PASS (ONE TOKEN) ----
    with torch.no_grad():
        outputs = model(**curr_inputs, output_attentions=True)

        next_token_id = torch.argmax(
            outputs.logits[:, -1, :], dim=-1
        ).item()

        generated_token = tokenizer.decode([next_token_id])

        # Get attention from the last layer, averaged over heads
        last_layer_attn = outputs.attentions[-1][0].mean(dim=0)
        
        # Extract attention weights pointing to vision tokens
        vision_attn = last_layer_attn[-1, vision_indices]
        vision_attn = vision_attn.float().cpu().numpy()

    # ---- 3. RESHAPE TO VISUAL FRAMES ----
    # Shape: (t_vis, eff_h, eff_w)
    attn_vis = vision_attn.reshape(t_vis, eff_h, eff_w)

    # ---- PRE-CALCULATE GLOBAL MIN/MAX IF NEEDED ----
    if not normalize_per_frame:
        global_min = attn_vis.min()
        global_max = attn_vis.max()
        global_denom = global_max - global_min + 1e-8

    # ---- 4. PROCESS AND OVERLAY ATTENTION ----
    n_rows = (T_orig + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = axes.flatten()

    for i in range(T_orig):
        # Determine which temporal token index this original frame belongs to
        t_idx = i // 2
        
        # Guard against index out of bounds if frame count mismatch
        if t_idx >= len(attn_vis):
            break

        # Get the spatial map for this token group
        attn_map = attn_vis[t_idx]

        # ---- NORMALIZE ----
        if normalize_per_frame:
            # Local normalization (per frame)
            local_min = attn_map.min()
            local_max = attn_map.max()
            attn_norm = (attn_map - local_min) / (local_max - local_min + 1e-8)
        else:
            # Global normalization (across video)
            attn_norm = (attn_map - global_min) / global_denom

        # Convert to tensor for high-quality upscaling
        attn_tensor = torch.from_numpy(attn_norm).unsqueeze(0).unsqueeze(0)
        
        # Upscale to the original image dimensions (H, W)
        upscaled_attn = F.interpolate(
            attn_tensor, 
            size=(H, W), 
            mode='bilinear', 
            align_corners=False
        ).squeeze().numpy()

        # Display original frame
        axes[i].imshow(frames[i])
        
        # Overlay heatmap
        axes[i].imshow(upscaled_attn, cmap=cmap, alpha=alpha)
        
        # Title logic
        norm_str = "Local" if normalize_per_frame else "Global"
        axes[i].set_title(f"Frame {i} (T-{t_idx}) [{norm_str}]")
        axes[i].axis('off')

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f"Spatial Attention ('{generated_token}') - Norm: {'Per-Frame' if normalize_per_frame else 'Global'}", fontsize=16)
    plt.tight_layout()
    plt.show()

    # <--- FEATURE 2: Return fig, weights, and token
    return fig, vision_attn, generated_token