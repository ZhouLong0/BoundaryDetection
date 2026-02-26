import torch
import PIL.Image
from typing import List
from .config import ReasoningConfig
from qwen_vl_utils import process_vision_info  # wherever this lives
from torch.nn import functional as F


class HypothesisVerifier:
    def __init__(self, model, processor, config: ReasoningConfig):
        self.model = model
        self.processor = processor
        self.config = config
        
        self.yes_id = self._single_id(self.config.TOKEN_YES)
        self.no_id = self._single_id(self.config.TOKEN_NO)
    
    def _single_id(self, tok: str) -> int:
        ids = self.processor.tokenizer.encode(tok, add_special_tokens=False)
        # Warning instead of assert to prevent crash on minor tokenization drifts
        if len(ids) != 1:
            print(f"Warning: Token '{tok}' splits into {ids}. Using first ID.")
        return ids[0]

    @torch.no_grad()
    def score_old(self, hypotheses: List[str], video_chunk: list[PIL.Image.Image]) -> torch.Tensor:
        """
        Returns normalized posterior probabilities across hypotheses.
        Math: Softmax( Log(P(yes)) - Log(P(no)) )
        """
        if not hypotheses:
            return torch.tensor([])

        # --- Batch Preparation ---
        prompts = []
        # Duplicate video list for every hypothesis in the batch
        # (Assuming processor accepts list of video-lists or broadcasts automatically)
        videos_batch = [video_chunk for _ in hypotheses]

        for hyp in hypotheses:
            conv = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {
                            "type": "text",
                            "text": self.config.VERIFY_PROMPT_TEMPLATE.format(hypothesis=hyp),
                        },
                    ],
                }
            ]
            prompt = self.processor.apply_chat_template(conv, add_generation_prompt=True)
            prompts.append(prompt)

        # --- Batched Inference ---
        inputs = self.processor(
            text=prompts,
            videos=videos_batch,
            return_tensors="pt",
            padding=True,
        )

        logits = self.model(
            **inputs.to(self.model.device)
        ).logits[:, -1, :] # (Batch_Size, Vocab_Size)

        generated_words = [self.processor.tokenizer.decode([torch.argmax(logit).item()]) for logit in logits]

        vocab_prob = torch.softmax(logits, dim=-1) # (Batch_Size, Vocab_Size)
        vocab_yes_no_prob = vocab_prob[:, [self.yes_id, self.no_id]] # (Batch_Size, 2)

        # Select Yes/No logits
        sub_logits = logits[:, [self.yes_id, self.no_id]] # (Batch_Size, 2)
        
        # --- Probability Calculation ---
        # 1. Log Softmax between Yes and No
        log_probs = torch.log_softmax(sub_logits, dim=-1)
        
        # 2. Log Odds (Yes - No)
        log_odds = log_probs[:, 0] - log_probs[:, 1]
        
        # 3. Posterior (Softmax across all hypotheses)
        posterior = torch.softmax(log_odds, dim=0)

        return posterior
    

    @torch.no_grad()
    def score(self, hypotheses: List[str], video_chunk: list, context: str = None) -> torch.Tensor:
        """
        Returns normalized posterior probabilities across hypotheses using the
        working input preparation pipeline.
        
        Math: Softmax( Log(P(yes)) - Log(P(no)) )
        which is equivalent to Softmax( Logit(yes) - Logit(no) )
        """
        if not hypotheses:
            return torch.tensor([])

        scores = []

        if context and self.config.VERIFY_PROMPT_TEMPLATE_W_HISTORY:
            prompt_template = self.config.VERIFY_PROMPT_TEMPLATE_W_HISTORY
        else:
            prompt_template = self.config.VERIFY_PROMPT_TEMPLATE

        # Iterate through hypotheses to ensure safe processing of vision inputs
        # (Batching variable-length video tokens is complex and error-prone; 
        # sequential processing guarantees correctness using the working pipeline).
        for hyp in hypotheses:
            # 1. Format the prompt  if the template does not include the placeholder for context, it will be ignored
            prompt = prompt_template.format(hypothesis=hyp, memory_text=context)

            # 2. Use the WORKING CODE's input preparation
            # We assume video_chunk is a list of PIL images or URLs
            inputs = self._prepare_inputs(
                video=video_chunk,
                prompt=prompt,
                # You can adjust these defaults or pass them in via **kwargs if needed
                max_new_tokens=1, 
                do_sample=False
            ).to(self.model.device)

            # 3. Forward Pass
            # We only need the logits for the last token to determine the next word
            outputs = self.model(**inputs)
            
            # Get logits for the last token: Shape (1, Vocab_Size)
            next_token_logits = outputs.logits[:, -1, :] 
            next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)
            yes_no_log_probs = next_token_log_probs[:, [self.yes_id, self.no_id]]

            generated_word = self.processor.tokenizer.decode(
                torch.argmax(next_token_logits, dim=-1).item())


            # 4. Calculate Log Odds (Score)
            # Mathematical Simplification: 
            # Log(Softmax(Yes)) - Log(Softmax(No)) == Logit(Yes) - Logit(No)
            # We do not need to calculate the full softmax over the vocab.
            yes_logit = next_token_logits[0, self.yes_id]
            no_logit = next_token_logits[0, self.no_id]
            log_odds = yes_logit - no_logit

            yes_log_prob = yes_no_log_probs[0, 0]

            scores.append(yes_log_prob) # log_odds or yes_log_prob depending on your preference for scoring

        # 5. Posterior Calculation (Softmax across all hypotheses)
        # Stack scores into a tensor of shape (Num_Hypotheses,)
        scores_tensor = torch.stack(scores)
        
        # Normalize across the hypotheses
        posterior = torch.softmax(scores_tensor, dim=0)

        return posterior
    


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
        )

        return inputs