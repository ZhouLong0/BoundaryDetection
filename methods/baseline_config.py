from dataclasses import dataclass


@dataclass
class BaselineConfig:
    """Base configuration for Baseline prompts and decision tokens."""

    initial_description: str = (
        "Briefly describe the action/event that just started in the current video segment. "
        "Describe only the main actions or events taking place. "
        "Describe only what is in the current segment."
    )
    validity_check: str = (
        "Given the previous event description: {previous_description}, "
        "is this the same event or a new one? "
        "Respond ONLY with 'SAME' if it's the same or 'NEW' if it's a new event."
    )
    update_description: str = (
        "Briefly describe the action/event that just started in the current video segment. "
        "Describe only the main actions or events taking place. "
        "Describe only what is in the current segment."
    )
    update_token: str = "NEW"
    skip_token: str = "SAME"

class TaposBaselineConfig(BaselineConfig):
    """Baseline configuration for TAPOS dataset (sports videos)."""

    initial_description: str = (
        "Briefly describe the action/event that just started in the current video segment. "
        "Describe only the main actions or events taking place. "
        "Describe only what is in the current segment."
    )
    validity_check: str = (
        "Given the previous event description: {previous_description}, "
        "is this the same event or a new one? "
        "Respond ONLY with 'SAME' if it's the same or 'NEW' if it's a new event."
    )
    update_description: str = (
        "Briefly describe the action/event that just started in the current video segment. "
        "Describe only the main actions or events taking place. "
        "Describe only what is in the current segment."
    )


@dataclass
class EgoperBaselineConfig(BaselineConfig):
    """Baseline configuration for Egoper (egocentric, first-person view)."""

    initial_description: str = (
        "The video shows the first view of a person performing an action. "
        "Briefly describe the action/event that just started in the current video segment. "
        "Describe only what is directly observable in the video."
    )
    update_description: str = (
        "The video shows the first view of a person performing an action. "
        "Briefly describe the new action/event that just started in the current video segment. "
        "Describe only what is directly observable in the video."
    )


@dataclass
class GTEABaselineConfig(BaselineConfig):
    """Baseline configuration for GTEA (egocentric kitchen actions)."""

    initial_description: str = (
        "The video shows the first view of a person performing a kitchen action. "
        "Briefly describe the action/event that just started in the current video segment. "
        "Describe only what is directly observable in the video."
    )
    update_description: str = (
        "The video shows the first view of a person performing a kitchen action. "
        "Briefly describe the new action/event that just started in the current video segment. "
        "Describe only what is directly observable in the video."
    )


@dataclass
class BreakfastBaselineConfig(BaselineConfig):
    """Baseline configuration for Breakfast dataset (kitchen, breakfast preparation)."""

    initial_description: str = (
        "The video shows a person preparing breakfast in the kitchen. "
        "Briefly describe the action/event that just started in the current video segment. "
        "Describe only what is directly observable in the video."
    )
    update_description: str = (
        "The video shows a person preparing breakfast in the kitchen. "
        "Briefly describe the new action/event that just started in the current video segment. "
        "Describe only what is directly observable in the video."
    )
