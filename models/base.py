from abc import ABC, abstractmethod

class VisionLanguageModel(ABC):
    @abstractmethod
    def infer(
        self,
        video,
        prompt: str,
        max_new_tokens: int = 2048,
        **kwargs
    ) -> str:
        pass