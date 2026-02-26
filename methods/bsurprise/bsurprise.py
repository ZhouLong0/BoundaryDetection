import PIL.Image
from typing import List, Union, Dict, Any
import json
import re
import numpy as np
import torch
from .config import ReasoningConfig
from .hypothesis_generator import HypothesisGenerator
from .hypothesis_verifier import HypothesisVerifier
from.divergence_utils import kl_divergence, js_divergence, total_variation


class BSurprise:
    def __init__(self, qwen_interface, divergence_type: str = ["kl", "js"], detection_threshold: float = 0.5, config: ReasoningConfig = None):
        self.config = config if config is not None else ReasoningConfig()
        self.qwen_interface = qwen_interface
        self.model = qwen_interface.model
        self.processor = qwen_interface.processor
        self.divergence_type = divergence_type
        self.detection_threshold = detection_threshold
        self.generator_component = HypothesisGenerator(self.model, self.processor, self.config)
        self.verifier_component = HypothesisVerifier(self.model, self.processor, self.config)

    def generate(self, memory_text: str, video_ctx: list[PIL.Image.Image]) -> List[str]:
        # 1. Generate raw text
        raw_generations = self.generator_component.generate(
            memory_text=memory_text,
            video_ctx=video_ctx,
        )

        # 2. Split into list
        hypotheses_list = [h.strip() for h in raw_generations.split("\n") if h.strip()]
        
        # 3. Append last memory state (Baseline)
        if memory_text:
            last_memory = memory_text.split('\n')[-1].strip()
            if last_memory:
                hypotheses_list.append(last_memory)

        return hypotheses_list
    
    def score(self, hypotheses_list: List[str], video_chunk: list[PIL.Image.Image], context: str = None) -> torch.Tensor:
        # Delegate to verifier component
        return self.verifier_component.score(hypotheses_list, video_chunk, context=context)  # You can modify this to pass actual context if needed
    
    def _calculate_divergence(self, old_scores: torch.Tensor, new_scores: torch.Tensor) -> float:
        if self.divergence_type == "kl":
            return kl_divergence(old_scores, new_scores)
        elif self.divergence_type == "js":
            return js_divergence(old_scores, new_scores)
        elif self.divergence_type == "tv":
            return total_variation(old_scores, new_scores)
        else:
            raise ValueError(f"Unsupported divergence type: {self.divergence_type}")
    
    def classify_video_chunks(self, video_data: Dict[str, Any], verbose: bool = False):
        """
        Processes video chunks to find event boundaries based on hypothesis surprise.
        """
        chunks = video_data['chunks']   # list of video chunks (each chunk is a list of PIL Images)
        chunk_ranges = video_data['chunk_ranges']
        video_fps = video_data['fps']
        chunk_labels = video_data["chunk_labels"][0]
        video_id = video_data['video_id']

        
        timeline = []
        boundaries_frames = []
        boundaries_timestamps = []

        # --- State Variables ---
        # "Anchor" variables represent the state at the start of the current event
        current_description = None
        current_hypotheses = []
        last_scores = None # The scores of hypotheses on the PREVIOUS chunk

        history_descriptions = []
        
        if verbose:
            print(f"Processing Video: {video_id} | Total Chunks: {len(chunks)} | FPS: {video_fps}")
            print(f"GT chunk labels: {np.where(np.array(chunk_labels) == 1)[0].tolist()}")


        # We use a while loop index to allow manual control (staying on same chunk)
        i = 0
        while i < len(chunks):
            chunk = chunks[i]
            start_frame, end_frame = chunk_ranges[i]
            
            # --- PHASE 1: Initialization (Start of Video or New Event) ---
            if current_description is None:
                # 1. Describe the new event anchor
                desc_prompt = self.config.DESCR_PROMPT_TEMPLATE
                current_description = self.qwen_interface.infer(chunk, desc_prompt, sample_fps=video_fps)
                if verbose: print(f"Chunk {i} | New Event Description: {current_description}")

                history_descriptions.append(current_description)
                
                # 2. Predict what happens next (Hypotheses)
                # We use the description as the 'memory' context
                memory_text = history_descriptions[-1]
                current_hypotheses = self.generate(memory_text=memory_text, video_ctx=chunk)
                if verbose: print(f"Chunk {i} | Generated Hypotheses: {current_hypotheses} | Memory Context: {memory_text}")
                
                # 3. Establish Baseline Scores
                # How well do these hypotheses fit the CURRENT chunk?
                last_scores = self.score(current_hypotheses, chunk)
                if verbose: print(f"Chunk {i} | Baseline Scores: {last_scores}")
                
                # Record to timeline
                timeline.append({
                    "start_frame": start_frame,
                    "description": current_description,
                    "type": "event_start"
                })
                
                i += 1
                continue

            # --- PHASE 2: Monitor for Surprise (Subsequent Chunks) ---
            
            # 1. Score the OLD hypotheses against the NEW chunk
            # "Do my previous predictions still hold true now?"
            new_scores = self.score(current_hypotheses, chunk)
            if verbose: print(f"Chunk {i} | New Scores: {new_scores}")
            
            # 2. Calculate Surprise (Divergence)
            divergence = self._calculate_divergence(last_scores, new_scores)
            
            if verbose: print(f"Chunk {i} | Divergence: {divergence:.4f}")

            # 3. Decision
            # You might need to tune this threshold in your ReasoningConfig
            threshold = self.detection_threshold

            if divergence > threshold:
                # --- SURPRISE! Boundary Detected ---
                if verbose: print(f"   -> SURPRISE DETECTED (Div {divergence:.2f} > {threshold})")

                # timeline.append({
                #     "start_frame": start_frame,
                #     "description": current_description,
                #     "type": "boundary",})
                
                # Record Boundary
                boundary_frame = start_frame + (end_frame - start_frame) // 2  # The midpoint of this chunk is the boundary
                boundaries_frames.append(boundary_frame)
                boundaries_timestamps.append(boundary_frame / video_fps)
                
                # Reset State
                # We set description to None so the loop re-initializes 
                # strictly on the *next* pass, but we actually want to re-process *this* chunk
                # as the anchor of the new event.
                current_description = None
                current_hypotheses = []
                last_scores = None
                
                # IMPORTANT: Do not increment 'i'. 
                # We loop back and treat chunk 'i' as the INIT chunk for the new event.
                continue

            else:
                # --- NO SURPRISE. Event Continues ---
                i += 1

        return timeline, boundaries_frames, boundaries_timestamps
    

    def classify_video_chunks_history_conditioned(self, video_data: Dict[str, Any], verbose: bool = False):
        """
        Processes video chunks to find event boundaries based on hypothesis surprise.
        """
        chunks = video_data['chunks']   # list of video chunks (each chunk is a list of PIL Images)
        chunk_ranges = video_data['chunk_ranges']
        video_fps = video_data['fps']
        chunk_labels = video_data["chunk_labels"][0]
        boundary_ids = np.where(np.array(chunk_labels) == 1)[0].tolist()
        video_id = video_data['video_id']

        
        timeline = []
        boundaries_frames = []
        boundaries_timestamps = []

        description_history = []  # We keep a history of descriptions to condition the generation on the full context, not just the last description.

        # --- State Variables ---
        # "Anchor" variables represent the state at the start of the current event
        current_description = None
        current_hypotheses = []
        last_scores = None # The scores of hypotheses on the PREVIOUS chunk
        
        if verbose:
            print(f"Processing Video: {video_id} | Total Chunks: {len(chunks)} | FPS: {video_fps}")

        # We use a while loop index to allow manual control (staying on same chunk)
        i = 0
        while i < len(chunks):
            chunk = chunks[i]
            start_frame, end_frame = chunk_ranges[i]
            
            # --- PHASE 1: Initialization (Start of Video or New Event) ---
            if current_description is None:                
                # 1. Describe the new event anchor
                desc_prompt = self.config.DESCR_PROMPT_TEMPLATE
                current_description = self.qwen_interface.infer(chunk, desc_prompt, sample_fps=video_fps)
                if verbose: print(f"Chunk {i} | New Event Description: {current_description}")
                description_history.append(current_description)
                
                # 2. Predict what happens next (Hypotheses)
                # We use the description as the 'memory' context
                memory_text = ",".join(description_history[-5:])  # Condition on the last 5 descriptions for more context
                current_hypotheses = self.generate(memory_text=memory_text, video_ctx=chunk)
                if verbose: print(f"Chunk {i} | Generated Hypotheses: {current_hypotheses} \n| Memory Context (last 5 desc: {memory_text}")
                
                # 3. Establish Baseline Scores
                # How well do these hypotheses fit the CURRENT chunk?
                last_scores = self.score(current_hypotheses, chunk, context=memory_text)
                if verbose: print(f"Chunk {i} | Baseline Scores: {last_scores} | Memory Context: {memory_text}")
                
                # Record to timeline
                timeline.append({
                    "start_frame": start_frame,
                    "description": current_description,
                    "type": "event_start"
                })
                
                i += 1
                continue

            # --- PHASE 2: Monitor for Surprise (Subsequent Chunks) ---
            
            # 1. Score the OLD hypotheses against the NEW chunk
            # "Do my previous predictions still hold true now?"
            context = ",".join(description_history[-5:])  # Use the same context for scoring
            new_scores = self.score(current_hypotheses, chunk, context=context)
            if verbose: print(f"Chunk {i} | New Scores: {new_scores} | Memory Context: {context}")
            
            # 2. Calculate Surprise (Divergence)
            divergence = self._calculate_divergence(last_scores, new_scores)
            
            if verbose: print(f"Chunk {i} | Divergence: {divergence:.4f}")

            # 3. Decision
            # You might need to tune this threshold in your ReasoningConfig
            threshold = self.detection_threshold

            if divergence > threshold:
                # --- SURPRISE! Boundary Detected ---
                if verbose: print(f"   -> SURPRISE DETECTED (Div {divergence:.2f} > {threshold})")

                # Record Boundary
                boundary_frame = start_frame + (end_frame - start_frame) // 2  # The midpoint of this chunk is the boundary
                boundaries_frames.append(boundary_frame)
                boundaries_timestamps.append(boundary_frame / video_fps)
                
                # Reset State
                # We set description to None so the loop re-initializes 
                # strictly on the *next* pass, but we actually want to re-process *this* chunk
                # as the anchor of the new event.
                current_description = None
                current_hypotheses = []
                last_scores = None
                
                # IMPORTANT: Do not increment 'i'. 
                # We loop back and treat chunk 'i' as the INIT chunk for the new event.
                continue

            else:
                # --- NO SURPRISE. Event Continues ---
                i += 1

        return timeline, boundaries_frames, boundaries_timestamps

    def _match_description_to_recipe_step(self, description: str, recipe_steps: List[str], verbose: bool = False) -> int:
        original_steps = list(recipe_steps)
        recipe_steps = [step.replace("BG", "Background") for step in recipe_steps]
        recipe_steps = [step.replace("SIL", "Background") for step in recipe_steps]

        def _find_background_idx() -> int:
            for i, s in enumerate(original_steps):
                if str(s).strip().lower() in {"bg", "sil", "background"}:
                    return i
            for i, s in enumerate(recipe_steps):
                if str(s).strip().lower() in {"bg", "sil", "background"}:
                    return i
            return 0

        bg_idx = _find_background_idx()

        steps_block = "\n".join([f"{i}: {s}" for i, s in enumerate(recipe_steps)])

        match_template = self.config.MATCH_PROMPT_TEMPLATE

        prompt = match_template.format(
            steps_block=steps_block,
            description=description,
            bg_idx=bg_idx,
        )
        response = self.qwen_interface.infer_text(prompt)
        response_str = str(response).strip()

        # Robust parsing: keep only the first integer in noisy outputs (e.g., "3a", "step 2", "2.")
        match = re.search(r"-?\d+", response_str)
        if match is None:
            print(
                f"[WARN] Could not parse numeric step index from response '{response_str}'. "
                f"Falling back to background index {bg_idx}."
            )
            return bg_idx

        extracted = match.group(0)
        if response_str != extracted:
            print(
                f"[WARN] Non-numeric response normalized from '{response_str}' to '{extracted}'."
            )

        step_idx = int(extracted)
        if 0 <= step_idx < len(recipe_steps):
            return step_idx

        print(
            f"[WARN] Parsed step index {step_idx} out of range [0, {len(recipe_steps)-1}]. "
            f"Falling back to background index {bg_idx}."
        )
        return bg_idx

    # def match_timeline_descriptions_to_recipe_steps(
    #     self,
    #     result_json_path: str,
    #     recipe_steps: List[str],
    # ) -> Dict[str, Any]:
    #     """
    #     Given a result file (with `timeline`), match each timeline description
    #     to one recipe step by querying Qwen.
    #     """
    #     with open(result_json_path, "r") as f:
    #         data = json.load(f)

    #     timeline = data.get("timeline", [])
    #     matches = []  # simple list aligned with timeline; duplicates are allowed

    #     for i, event in enumerate(timeline):
    #         description = event.get("description", "")
    #         step_idx = self._match_description_to_recipe_step(description, recipe_steps) if description else -1

    #         recipe_step = recipe_steps[step_idx]

    #         matches.append({
    #             "description": description,
    #             "matched_step_idx": step_idx,
    #             "matched_step": recipe_step,
    #             "start_frame": event.get("start_frame", None),
    #         })

    #     return matches

    def match_timeline_descriptions_to_recipe_steps(
            self,
            result_json_path: str,
            recipe_steps: List[str],
            verbose: bool = False
        ) -> Dict[str, Any]:
            """
            Given a result file (with `timeline`), match each timeline description
            to one recipe step. Modifies the timeline in-place and returns the full JSON data.
            """
            with open(result_json_path, "r") as f:
                data = json.load(f)

            # Get the reference to the timeline list
            timeline = data.get("timeline", [])

            # Iterate through the actual dictionaries in the list
            for event in timeline:
                description = event["description"]  # This is a reference to the description in the original timeline event

                step_idx = self._match_description_to_recipe_step(description, recipe_steps, verbose=verbose)
                # Safely get the recipe step, handling potential out-of-bounds
                if 0 <= step_idx < len(recipe_steps):
                    recipe_step = recipe_steps[step_idx]
                else:
                    raise ValueError(f"Matched step index {step_idx} is out of bounds for recipe steps list: {recipe_steps}.")

                if verbose:
                    print(f"Matching description '{description}' to step index {step_idx} ('{recipe_step}')")

                # Inject the new data directly into the existing event dictionary
                event["matched_step_idx"] = step_idx
                event["matched_step"] = recipe_step

            # Return the entirely preserved JSON structure, now enriched with matched steps
            return data