import torch
import PIL.Image
from typing import List, Dict
from .config import ReasoningConfig

class HypothesisGenerator:
    def __init__(self, model, processor, config: ReasoningConfig):
        self.model = model
        self.processor = processor
        self.config = config

    def generate(self, memory_text: str, video_ctx: list[PIL.Image.Image]) -> str:
        """
        Generates raw hypothesis text block.
        """
        conv = self._build_conv(memory_text)
        
        prompt = self.processor.apply_chat_template(
            conv, add_generation_prompt=True
        )

        inputs = self.processor(
            text=prompt,
            videos=video_ctx,
            return_tensors="pt",
            padding=True,
            padding_side="left",  # Crucial for generation
            add_special_tokens=False,
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs.to(self.model.device),
                do_sample=False,  # Deterministic output for testing
                num_return_sequences=1,
                max_new_tokens=self.config.max_new_tokens,
                return_dict_in_generate=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                use_cache=False,
            )

        # Decode
        decoded_text = self.processor.tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True
        )
        
        return self._postprocess(decoded_text)

    def _build_conv(self, memory_text: str) -> list[dict]:
        text_content = self.config.GEN_PROMPT_TEMPLATE.format(
            memory_text=memory_text,
            n=self.config.n_hypotheses
        )
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_content},
                    {"type": "video"},
                ],
            }
        ]

    def _postprocess(self, text: str) -> str:
        """Parses the model output to extract the assistant's response."""
        text = text.strip()
        # Robust splitting
        if self.config.SPLIT_TOKEN in text:
            text = text.split(self.config.SPLIT_TOKEN, 1)[1]
        elif "assistant" in text: # Fallback if newline is missing
            text = text.split("assistant", 1)[1]
            
        return text.replace(":", "").strip()