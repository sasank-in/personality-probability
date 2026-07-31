"""
Big Five Personality Classifier — model architecture.
Input:  1024-dim Mistral text embedding (StandardScaler normalized)
Output: 5 logits (O, C, E, A, N) — apply sigmoid for probabilities

A single regularized linear layer (== 5 L2-penalized logistic regressions).
With only ~1.6k training essays, deeper MLPs overfit (train AUC ~1.0,
test AUC ~0.6) and score worse on held-out data than this linear model
(~0.62 mean acc vs ~0.58).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PersonalityClassifier(nn.Module):
    def __init__(self, input_dim: int = 1024):
        super().__init__()
        self.fc_out = nn.Linear(input_dim, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_out(x)
