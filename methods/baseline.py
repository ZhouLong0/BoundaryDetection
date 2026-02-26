import warnings
from typing import Dict, List, Any, Tuple, Union
from models import Qwen3VL
from methods.baseline_config import BaselineConfig, EgoperBaselineConfig, GTEABaselineConfig, BreakfastBaselineConfig


def eval_video(
    video_data: Dict[str, Any],
    prompt_templates: Union[BaselineConfig, Dict],
    model: Qwen3VL,
    chunk_size: int = 16,
    update_threshold: float = 0.5,
    verbose: bool = False,
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

    # --- Prompt Settings ---
    if isinstance(prompt_templates, BaselineConfig):
        skip_token = prompt_templates.skip_token
        update_token = prompt_templates.update_token
        initial_prompt = prompt_templates.initial_description
        validity_check_template = prompt_templates.validity_check
        update_prompt = prompt_templates.update_description
    else:
        # Legacy dict support
        if "skip_token" not in prompt_templates:
            warnings.warn("[WARNING] 'skip_token' not found in prompts. Defaulting to 'SAME'.")
        skip_token = prompt_templates.get("skip_token", "SAME")
        if "update_token" not in prompt_templates:
            warnings.warn("[WARNING] 'update_token' not found in prompts. Defaulting to 'NEW'.")
        update_token = prompt_templates.get("update_token", "NEW")
        initial_prompt = prompt_templates["initial_description"]
        validity_check_template = prompt_templates["validity_check"]
        update_prompt = prompt_templates["update_description"]
    
    current_description = None
    timeline = []
    boundaries_frame_indices = []
    boundaries_timestamps = []
    
    for idx, (frame_chunk, (start, end)) in enumerate(zip(chunks, chunk_ranges)):
        
        # --- Step 1: Initial Chunk ---
        if idx == 0:
            prompt = initial_prompt
            current_description = model.infer(frame_chunk, prompt, sample_fps=fps)
            if verbose: print(f"Chunk {idx}: Initial: {current_description}")
            
            # Original timeline format: (start, end, description, answer/decision)
            timeline.append((start, end, current_description, None))
            continue

        # --- Step 2: Validity Check (Probabilistic) ---
        prompt_check = validity_check_template.format(
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
            prompt_update = update_prompt
            current_description = model.infer(frame_chunk, prompt_update, sample_fps=fps)
            
            timeline.append((start, end, current_description, update_token))
            
            # Original boundary logic
            boundary_frame = start + (end - start) // 2
            boundaries_frame_indices.append(boundary_frame)
            boundaries_timestamps.append(boundary_frame / fps)
            
            if verbose: print(f"  -> Update TRIGGERED")
            
        else:
            # SKIP UPDATE
            timeline.append((start, end, "skipped", skip_token))
            if verbose: print(f"  -> Update SKIPPED")

    return timeline, boundaries_frame_indices, boundaries_timestamps