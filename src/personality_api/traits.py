"""Single source of truth for the Big Five traits and their interpretations."""

from __future__ import annotations

# Order matters: the model outputs 5 logits in exactly this order.
TRAIT_NAMES: tuple[str, ...] = (
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
)

# Dataset column keys (O, C, E, A, N) aligned to TRAIT_NAMES.
TRAIT_KEYS: tuple[str, ...] = ("O", "C", "E", "A", "N")

INTERPRETATIONS: dict[str, dict[int, str]] = {
    "Openness": {0: "Practical, conventional", 1: "Creative, open to new experiences"},
    "Conscientiousness": {0: "Spontaneous, flexible", 1: "Organized, disciplined"},
    "Extraversion": {0: "Reserved, introverted", 1: "Outgoing, energetic"},
    "Agreeableness": {0: "Competitive, skeptical", 1: "Cooperative, compassionate"},
    "Neuroticism": {0: "Emotionally stable", 1: "Sensitive, anxious"},
}
