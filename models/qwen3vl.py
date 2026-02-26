from qwen_vl_utils import process_vision_info  # wherever this lives
from .base import VisionLanguageModel
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
import warnings

class Qwen3VL(VisionLanguageModel):
    def __init__(
        self,
        model_name: str="Qwen/Qwen3-VL-8B-Instruct", # Qwen/Qwen2-VL-7B-Instruct
        device: str = "cuda",
        torch_dtype="auto"
    ):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto"
        )
        self.device = device

    def _build_messages(self, video, prompt, **video_kwargs):
        return [
            {
                "role": "user",
                "content": [
                    {"video": video, **video_kwargs},
                    {"type": "text", "text": prompt},
                ]
            }
        ]

    def _prepare_inputs(
        self,
        video,
        prompt,
        max_new_tokens=2048,
        total_pixels=20480 * 32 * 32,
        min_pixels=64 * 32 * 32,
        max_frames=2048,
        sample_fps=2,
        do_sample=False
    ):
        messages = self._build_messages(
            video,
            prompt,
            total_pixels=total_pixels,
            min_pixels=min_pixels,
            max_frames=max_frames,
            sample_fps=sample_fps,
        )

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            [messages],
            return_video_kwargs=True,
            image_patch_size=16,
            return_video_metadata=True
        )

        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs = list(video_inputs)
            video_metadatas = list(video_metadatas)
        else:
            video_metadatas = None

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadatas,
            **video_kwargs,
            do_resize=False,
            return_tensors="pt"
        ).to(self.device)

        return inputs
    
    def infer(
        self,
        video,
        prompt,
        max_new_tokens=2048,
        total_pixels=20480 * 32 * 32,
        min_pixels=64 * 32 * 32,
        max_frames=2048,
        sample_fps=2,
        do_sample=False
    ):
        """
        Perform multimodal inference on input video and text prompt to generate model response.

        Args:
            video (str or list/tuple): Video input, supports two formats:
                - str: Path or URL to a video file. The function will automatically read and sample frames.
                - list/tuple: Pre-sampled list of video frames (PIL.Image or url). 
                In this case, `sample_fps` indicates the frame rate at which these frames were sampled from the original video.
            prompt (str): User text prompt to guide the model's generation.
            max_new_tokens (int, optional): Maximum number of tokens to generate. Default is 2048.
            total_pixels (int, optional): Maximum total pixels for video frame resizing (upper bound). Default is 20480*32*32.
            min_pixels (int, optional): Minimum total pixels for video frame resizing (lower bound). Default is 16*32*32.
            sample_fps (int, optional): ONLY effective when `video` is a list/tuple of frames!
                Specifies the original sampling frame rate (FPS) from which the frame list was extracted.
                Used for temporal alignment or normalization in the model. Default is 2.

        Returns:
            str: Generated text response from the model.

        Notes:
            - When `video` is a string (path/URL), `sample_fps` is ignored and will be overridden by the video reader backend.
            - When `video` is a frame list, `sample_fps` informs the model of the original sampling rate to help understand temporal density.
        """
        messages = self._build_messages(
            video,
            prompt,
            total_pixels=total_pixels,
            min_pixels=min_pixels,
            max_frames=max_frames,
            sample_fps=sample_fps,
        )

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            [messages],
            return_video_kwargs=True,
            image_patch_size=16,
            return_video_metadata=True
        )

        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs = list(video_inputs)
            video_metadatas = list(video_metadatas)
        else:
            video_metadatas = None

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadatas,
            **video_kwargs,
            do_resize=False,
            return_tensors="pt"
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample
        )

        generated_ids = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, output_ids)
        ]

        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]
    
    def infer_text(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        do_sample: bool = False,
    ) -> str:
        """
        Text-only inference (no visual inputs).
        """
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            return_tensors="pt"
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample
        )

        generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0].strip()

    def get_next_token_probs(
        self,
        video,
        prompt,
        candidate_strings: list,
        total_pixels=20480 * 32 * 32,
        min_pixels=64 * 32 * 32,
        max_frames=2048,
        sample_fps=2,
    ):
        """
        Calculates the probability that the next generated token matches the start of 
        each string in candidate_strings. Warns if a candidate splits into multiple tokens.
        """

        # 1. Prepare inputs
        inputs = self._prepare_inputs(
            video,
            prompt,
            total_pixels=total_pixels,
            min_pixels=min_pixels,
            max_frames=max_frames,
            sample_fps=sample_fps
        )

        # 2. Forward pass (get logits for the last token)
        with torch.no_grad():
            outputs = self.model(**inputs)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)

        # 3. Extract probabilities with validation
        candidate_probs = {}
        tokenizer = self.processor.tokenizer

        for cand in candidate_strings:
            # Encode without special tokens (BOS/EOS)
            cand_ids = tokenizer.encode(cand, add_special_tokens=False)
            
            if not cand_ids:
                candidate_probs[cand] = 0.0
                warnings.warn(
                    f"\nCandidate string '{cand}' tokenizes to an empty sequence. "
                    "Probability set to 0.0.\n"
                )
                continue
            
            # --- WARNING CHECK ---
            if len(cand_ids) > 1:
                token_strs = tokenizer.convert_ids_to_tokens(cand_ids)
                warnings.warn(
                    f"\nCandidate string '{cand}' tokenizes to {len(cand_ids)} tokens: {token_strs} (IDs: {cand_ids}).\n"
                    f"Probability will ONLY be calculated for the first token '{token_strs[0]}'. "
                    "This may result in inaccurate classification."
                )
            # ---------------------

            target_token_id = cand_ids[0]
            prob = next_token_probs[0, target_token_id].item()
            candidate_probs[cand] = prob

        return candidate_probs