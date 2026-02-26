import PIL
import torch
class _HypothesisGenerator:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def _build_conv(self, memory_text: str, n_hypotheses: int = 3) -> list[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Here is what happened so far from the beginning of the video: "
                            f"{memory_text}"
                        ),
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Based on this information and recent frames, "
                            f"predict {n_hypotheses} plausible next event, each one in 8-10 words "
                            "that is different the current action. "
                            "Each event should be separated with linebreak. "
                            "Focus on the main subject and main action. "
                        ),
                    },
                    {"type": "video"},
                ],
            }
        ]

    def _postprocess(self, text: str) -> str:
        text = text.strip().lower()
        if "assistant\n" in text:
            text = text.split("assistant\n", 1)[1]
        return text.replace(":", "").strip()

    def generate(
        self,
        memory_text: str,
        video_ctx: list[PIL.Image.Image],
        n_hypotheses: int,
    ):
        conv = self._build_conv(memory_text, n_hypotheses=n_hypotheses)

        prompt = self.processor.apply_chat_template(
            conv, add_generation_prompt=True
        )

        inputs = self.processor(
            text=prompt,
            videos=video_ctx,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )

        outputs = self.model.generate(
                **inputs.to(self.model.device),
                do_sample=True,
                num_return_sequences=1,
                max_new_tokens=100,
                return_dict_in_generate=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                use_cache=False,
            )
        hyp = self.processor.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        hyp = hyp.split("assistant\n")[1].replace(":", "")

        return hyp
        

class _HypothesisVerifier:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.yes_id = self._single_id(" yes")
        self.no_id = self._single_id(" no")
    
    def _single_id(self, tok: str) -> int:
        ids = self.processor.tokenizer.encode(tok, add_special_tokens=False)
        assert len(ids) == 1, f"‘{tok}’ splits into {ids}"
        return ids[0]

    @torch.no_grad()
    def score(self, hypotheses: str, video_chunk) -> float:
        """
        Returns log-odds: log P(yes) - log P(no)
        """
        logits_list = []
        log_like_list = []
        for hyp in hypotheses:
            conv = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {
                            "type": "text",
                            "text": (
                                f"Statement: {hyp}\n"
                                "Is this statement true in the CURRENT video? "
                                "Answer 'yes' or 'no'."
                            ),
                        },
                    ],
                }
            ]

            prompt = self.processor.apply_chat_template(
                conv, add_generation_prompt=True
            )

            inputs = self.processor(
                text=prompt,
                videos=video_chunk,
                return_tensors="pt",
            )

            logits = self.model(
                **inputs.to(self.model.device)
            ).logits[0, -1]

            sub_logits = logits[[self.yes_id, self.no_id]]

            logits_list.append(sub_logits)
            
            logprob_yes = sub_logits.log_softmax(dim=-1)[0]  # log P(yes|hyp, obs)
            log_like_list.append(logprob_yes)

        logits_yes_no = torch.stack(logits_list)  # (N, 2)
        log_probs = torch.log_softmax(logits_yes_no, dim=-1)
        log_odds = log_probs[:, 0] - log_probs[:, 1]
        posterior = torch.softmax(log_odds, dim=0)

        # log-odds
        return posterior
    

class QwenVideoReasoner:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

        self._generator = _HypothesisGenerator(model, processor)
        self._verifier = _HypothesisVerifier(model, processor)

    def generate(self, memory_text, video_ctx, n_hypotheses: int):
        # --- Generation ---
        generations = self._generator.generate(
            memory_text=memory_text,
            video_ctx=video_ctx,
            n_hypotheses=n_hypotheses,
        )

        # you currently only use the first generation
        hypotheses_list = generations.split("\n")
        hypotheses_list.append(memory_text.split('\n')[-1])

        return hypotheses_list
    
    def score(self, hypotheses_list, video_chunk):
        # --- Verification ---
        score = self._verifier.score(hypotheses_list, video_chunk)
        return score
