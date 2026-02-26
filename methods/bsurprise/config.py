from dataclasses import dataclass

@dataclass
class ReasoningConfig:
    """Central configuration for prompts and generation parameters."""
    
    # Generation Settings
    max_new_tokens: int = 100
    n_hypotheses: int = 3
    
    # Prompts
    # {memory_text} and {n} are placeholders filled at runtime
    GEN_PROMPT_TEMPLATE: str = (
        "Here is what happened so far in the video: {memory_text}\n"
        "Based on this information and recent frames, predict {n} plausible next events that are semantically different from the current action. "
        "Each event should be in 8-10 words. "
        "Each event should be separated with linebreak. "
        "Focus on the main subject and main action."
    )
    
    # {hypothesis} is filled at runtime
    VERIFY_PROMPT_TEMPLATE: str = (
        "Statement: {hypothesis}\n"
        "Is this statement true in the CURRENT video? "
        "Answer only with 'yes' or 'no'."
    )

    DESCR_PROMPT_TEMPLATE: str = (
        "Describe the single observable main action or event with 8-10 words in this 0.5 second video segment."
    )

    # Magic Tokens & Parsing
    TOKEN_YES: str = "yes"
    TOKEN_NO: str = "no"
    SPLIT_TOKEN: str = "assistant\n"


@dataclass
class EgoperConfig(ReasoningConfig):
    """Central configuration for prompts and generation parameters."""
    
    # Generation Settings
    max_new_tokens: int = 500
    n_hypotheses: int = 5
    
    # Prompts
    # {memory_text} and {n} are placeholders filled at runtime
    GEN_PROMPT_TEMPLATE: str = (
        "Context so far in this first view video: {memory_text}\n"
        "Based on this information and recent frames, predict {n} plausible next events that are semantically different from the current action. "
        "Each event should be in 8-10 words. "
        "Each event should be separated with linebreak. "
        "Focus on the main action."
    )
    
    # {hypothesis} is filled at runtime
    VERIFY_PROMPT_TEMPLATE: str = (
        "Statement: {hypothesis}\n"
        "Is this statement true in the CURRENT video? "
        "Answer only with 'yes' or 'no'."
    )

    DESCR_PROMPT_TEMPLATE: str = (
        "The video shows the first view of a person performing an action. "
        "Describe the action being performed by the person in 8-10 words in this 2 second video segment."
        "Describe only what is directly observable in the video."
    )

    VERIFY_PROMPT_TEMPLATE_W_HISTORY: str = (
        "You are given a textual summary of the video so far, the hypothesis of what is happening and the current video. "
        "Your task is to evaluate whether each hypothesis generated from the prior context holds in the current video."
        "Context so far: {memory_text}"
        "Hypothesis: {hypothesis}"
        "Question: Is this hypothesis true in the current video? Answer with a single word: yes or no.")

    # MATCH_PROMPT_TEMPLATE: str = (
    #     "You are given a list of numbers and the corresponding recipe steps.\n"
    #     "You are also given a description of an event that occurred in a video.\n"
    #     "Your task:\n"
    #     "Match the description to one of the recipe steps.\n"
    #     "The description may be matched to a step if it:\n"
    #     "Paraphrases a step using different wording.\n"
    #     "Describes only part of a step or an action that may belong within a step.\n"
    #     "Contains minor visual misidentifications of objects but the main action, intent, and meaning still clearly corresponds to a step.\n"
    #     "If the description clearly does NOT match any step's action, or is a background activity, it should be considered as Background.\n\n"
    #     "Recipe Steps:\n{steps_block}\n"
    #     "Description:\n{description}\n"
    #     "Important: Answer ONLY with the exact step number (e.g., 0, 1, 2, 3). Do not include any extra text."
    # )

    MATCH_PROMPT_TEMPLATE: str = """
        Task: choose exactly one step index for the event description.

        You are given:
        - A numbered list of recipe steps.
        - One event description from a video.

        Rules:
        1) Return the index of the BEST matching step.
        2) If no step clearly matches, return the Background index.
        3) Match by action semantics, not exact wording.
        4) Ignore minor object/detail errors if the core action matches.
        5) If unsure between multiple steps, choose the closest action in time/order.

        Recipe steps:
        {steps_block}

        Background index: {bg_idx}

        Event description:
        {description}

        Output format:
        Return ONLY one integer (example: 3). No words, no punctuation.
        """

    # Magic Tokens & Parsing
    TOKEN_YES: str = "yes"
    TOKEN_NO: str = "no"
    SPLIT_TOKEN: str = "assistant\n"




@dataclass
class GTEAConfig(ReasoningConfig):
    """Central configuration for prompts and generation parameters."""
    
    # Generation Settings
    max_new_tokens: int = 500
    n_hypotheses: int = 5
    
    # Prompts
    # {memory_text} and {n} are placeholders filled at runtime
    GEN_PROMPT_TEMPLATE: str = (
        "Context so far in this first view video: {memory_text}\n"
        "Based on this information and recent frames, predict {n} plausible next events that are semantically different from the current action. "
        "Each event should be in 8-10 words. "
        "Each event should be separated with linebreak. "
        "Focus on the main action."
    )
    
    # {hypothesis} is filled at runtime
    VERIFY_PROMPT_TEMPLATE: str = (
        "Statement: {hypothesis}\n"
        "Is this statement true in the CURRENT video? "
        "Answer only with 'yes' or 'no'."
    )

    DESCR_PROMPT_TEMPLATE: str = (
        "The video shows the first view of a person performing an action. "
        "Describe the action being performed by the person in 8-10 words in this 2 second video segment."
        "Describe only what is directly observable in the video."
    )

    VERIFY_PROMPT_TEMPLATE_W_HISTORY: str = (
        "You are given a textual summary of the video so far, the hypothesis of what is happening and the current video. "
        "Your task is to evaluate whether each hypothesis generated from the prior context holds in the current video."
        "Context so far: {memory_text}"
        "Hypothesis: {hypothesis}"
        "Question: Is this hypothesis true in the current video? Answer with a single word: yes or no.")

    MATCH_PROMPT_TEMPLATE: str = (
        "You are given a list of numbers and the corresponding recipe steps.\n"
        "You are also given a description of an event that occurred in a video.\n"
        "Your task:\n"
        "Match the description to one of the recipe steps.\n"
        "The description may be matched to a step if it:\n"
        "Paraphrases a step using different wording.\n"
        "Describes only part of a step or an action that may belong within a step.\n"
        "Contains minor visual misidentifications of objects but the main action, intent, and meaning still clearly corresponds to a step.\n"
        "If the description clearly does NOT match any step's action, or is a background activity, it should be considered as Background.\n\n"
        "Recipe Steps:\n{steps_block}\n"
        "Description:\n{description}\n"
        "Important: Answer ONLY with the exact step number (e.g., 0, 1, 2, 3). Do not include any extra text."
    )

    # Magic Tokens & Parsing
    TOKEN_YES: str = "yes"
    TOKEN_NO: str = "no"
    SPLIT_TOKEN: str = "assistant\n"


@dataclass
class BreakfastConfig(ReasoningConfig):
    """Configuration for Breakfast dataset prompts."""

    max_new_tokens: int = 500
    n_hypotheses: int = 5

    GEN_PROMPT_TEMPLATE: str = (
        "Context so far in this kitchen video: {memory_text}\n"
        "Based on this information and recent frames, predict {n} plausible next events that are semantically different from the current action. "
        "Each event should be in 8-10 words. "
        "Each event should be separated with linebreak. "
        "Focus on the main action."
    )

    VERIFY_PROMPT_TEMPLATE: str = (
        "Statement: {hypothesis}\n"
        "Is this statement true in the CURRENT video? "
        "Answer only with 'yes' or 'no'."
    )

    DESCR_PROMPT_TEMPLATE: str = (
        "The video shows a person preparing breakfast in the kitchen. "
        "Describe the action being performed in 8-10 words in this 2 second video segment."
        "Describe only what is directly observable in the video."
    )

    VERIFY_PROMPT_TEMPLATE_W_HISTORY: str = (
    "You are given a textual summary of the video so far, the hypothesis of what is happening and the current video. "
    "Your task is to evaluate whether each hypothesis generated from the prior context holds in the current video."
    "Context so far: {memory_text}"
    "Hypothesis: {hypothesis}"
    "Question: Is this hypothesis true in the current video? Answer with a single word: yes or no.")


    # MATCH_PROMPT_TEMPLATE: str = (
    #     "You are given a list of numbers and the corresponding breakfast recipe steps.\n"
    #     "You are also given a description of an event that occurred in a video.\n"
    #     "Your task:\n"
    #     "Match the description to one of the recipe steps.\n"
    #     "The description may be matched to a step if it:\n"
    #     "Paraphrases a step using different wording.\n"
    #     "Describes only part of a step or an action that may belong within a step.\n"
    #     "Contains minor visual misidentifications of objects but the main action, intent, and meaning still clearly corresponds to a step.\n"
    #     "If the description clearly does NOT match any step's action, or is a background activity, it should be considered as Background.\n\n"
    #     "Recipe Steps:\n{steps_block}\n"
    #     "Description:\n{description}\n"
    #     "Important: Answer ONLY with the exact step number (e.g., 0, 1, 2, 3). Do not include any extra text."
    # )

    MATCH_PROMPT_TEMPLATE: str = """
        Task: choose exactly one step index for the event description.

        You are given:
        - A numbered list of recipe steps.
        - One event description from a video.

        Rules:
        1) Return the index of the BEST matching step.
        2) If no step clearly matches, return the Background index.
        3) Match by action semantics, not exact wording.
        4) Ignore minor object/detail errors if the core action matches.
        5) If unsure between multiple steps, choose the closest action in time/order.

        Recipe steps:
        {steps_block}

        Background index: {bg_idx}

        Event description:
        {description}

        Output format:
        Return ONLY one integer (example: 3). No words, no punctuation.
        """

    # Magic Tokens & Parsing
    TOKEN_YES: str = "yes"
    TOKEN_NO: str = "no"
    SPLIT_TOKEN: str = "assistant\n"



@dataclass
class TaposConfig(ReasoningConfig):
    """Configuration for TAPOS dataset prompts."""

    max_new_tokens: int = 500
    n_hypotheses: int = 5

    GEN_PROMPT_TEMPLATE: str = (
        "Context so far: {memory_text}\n"
        "Based on this information and recent frames, predict {n} plausible next events that are semantically different from the current action. "
        "Each event should be in 8-10 words. "
        "Each event should be separated with linebreak. "
    )

    VERIFY_PROMPT_TEMPLATE: str = (
        "Statement: {hypothesis}\n"
        "Is this statement true in the CURRENT video? "
        "Answer only with 'yes' or 'no'."
    )

    DESCR_PROMPT_TEMPLATE: str = (
        "Describe the single observable action or event in this 0.5 second video segment with 8-10 words."
    )

    VERIFY_PROMPT_TEMPLATE_W_HISTORY: str = (
    "You are given a textual summary of the video so far, the hypothesis of what is happening and the current video."
    "Your task is to if the current video show exactly the same event as the hypothesis."
    "Context so far: {memory_text}"
    "Hypothesis: {hypothesis}"
    "Question: Is this hypothesis true in the current video? Answer with a single word: yes or no.")

    MATCH_PROMPT_TEMPLATE: str = """
        Task: choose exactly one step index for the event description.

        You are given:
        - A numbered list of recipe steps.
        - One event description from a video.

        Rules:
        1) Return the index of the BEST matching step.
        2) If no step clearly matches, return the Background index.
        3) Match by action semantics, not exact wording.
        4) Ignore minor object/detail errors if the core action matches.
        5) If unsure between multiple steps, choose the closest action in time/order.

        Recipe steps:
        {steps_block}

        Background index: {bg_idx}

        Event description:
        {description}

        Output format:
        Return ONLY one integer (example: 3). No words, no punctuation.
        """

    # Magic Tokens & Parsing
    TOKEN_YES: str = "yes"
    TOKEN_NO: str = "no"
    SPLIT_TOKEN: str = "assistant\n"