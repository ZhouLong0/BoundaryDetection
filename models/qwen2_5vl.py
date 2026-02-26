from qwen_vl_utils import process_vision_info
from .base import VisionLanguageModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
import warnings

class Qwen2_5VL(VisionLanguageModel):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str = "cuda",
        torch_dtype="auto",
    ):
        # Qwen2.5-VL uses a specific model class in newer transformers
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        self.device = device

    def _build_messages(self, video, prompt, **video_kwargs):
        """
        Builds the message list. Qwen2.5-VL supports passing 
        FPS and pixel constraints directly in the content dict.
        """
        return [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video, **video_kwargs},
                    {"type": "text", "text": prompt},
                ]
            }
        ]

    def _prepare_inputs(
        self,
        video,
        prompt,
        total_pixels=20480 * 28 * 28, # Qwen2.5 uses 28 as base patch factor
        min_pixels=16 * 28 * 28,
        max_frames=128,
        sample_fps=1.0,
    ):
        messages = self._build_messages(
            video,
            prompt,
            total_pixels=total_pixels,
            min_pixels=min_pixels,
            max_frames=max_frames,
            fps=sample_fps,
        )

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # process_vision_info handles the extraction and resizing logic
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, 
            return_video_kwargs=True
        )

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs, # This now includes 'fps' and 'video_grid_thw'
        ).to(self.device)

        return inputs
    
    def infer(
        self,
        video,
        prompt,
        max_new_tokens=2048,
        sample_fps=1.0,
        do_sample=False,
    ):
        inputs = self._prepare_inputs(video, prompt, sample_fps=sample_fps)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample
        )

        # Trim the input tokens from the output
        generated_ids = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, output_ids)
        ]

        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
    
    def get_next_token_probs(
        self,
        video,
        prompt,
        candidate_strings: list,
        sample_fps=1.0,
    ):
        inputs = self._prepare_inputs(video, prompt, sample_fps=sample_fps)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Logits for the last generated position
            next_token_logits = outputs.logits[:, -1, :]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)

        candidate_probs = {}
        tokenizer = self.processor.tokenizer

        for cand in candidate_strings:
            cand_ids = tokenizer.encode(cand, add_special_tokens=False)
            
            if not cand_ids:
                candidate_probs[cand] = 0.0
                continue
            
            if len(cand_ids) > 1:
                token_strs = tokenizer.convert_ids_to_tokens(cand_ids)
                warnings.warn(f"String '{cand}' is multiple tokens: {token_strs}. Only first token prob returned.")

            target_token_id = cand_ids[0]
            candidate_probs[cand] = next_token_probs[0, target_token_id].item()

        return candidate_probs